"""Host-side tests of the Isaac v2 scene and camera description (TASK-025).

The Isaac transport runs only inside the Isaac container; these tests cover what must hold on
every host: the module stays out of the core import path, the committed scene and camera
manifests are exactly what the MuJoCo v2 scene yields plus the pinned Isaac constants, the
Isaac reset placement and task truth reproduce ``MuJoCoSimulation``'s rules, and the declared
image metric behaves.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import isaac_scene as scene
from embodied_jepa.contracts import ContractError
from embodied_jepa.isaac_transport import onboard_camera_offset

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "configs/isaac/apple_to_plate_v2_scene_v1.json"
CAMERA = ROOT / "configs/isaac/onboard_camera_v1.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_module_imports_without_isaac_torch_or_mujoco():
    code = (
        "import sys, embodied_jepa.isaac_scene; "
        "bad = [m for m in ('torch', 'warp', 'isaaclab', 'omni', 'mujoco', 'pxr') "
        "if m in sys.modules]; assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.fixture(scope="module")
def v2_sim():
    pytest.importorskip("mujoco")
    from embodied_jepa.apple_to_plate_v2 import apply_v2_scene
    from embodied_jepa.simulation import MuJoCoSimulation

    sim = MuJoCoSimulation(width=112, height=112, render=False)
    apply_v2_scene(sim.model)
    yield sim
    sim.close()


def test_committed_manifests_match_mujoco_and_pinned_constants(v2_sim):
    assert scene.build_scene_manifest(v2_sim) == load(SCENE)
    assert scene.build_camera_manifest(v2_sim) == load(CAMERA)


def test_scene_manifest_describes_the_v2_apple_and_plate():
    m = scene.validate_scene_manifest(load(SCENE))
    assert m["apple"]["condim"] == 6 and m["apple"]["friction"] == [1.0, 0.01, 0.001]
    assert m["apple"]["radius"] == pytest.approx(0.027)
    assert m["apple"]["mass"] == pytest.approx(0.08)
    # Solid-sphere inertia, which PhysX also derives from the shape.
    assert m["apple"]["inertia_diag"] == pytest.approx([0.4 * 0.08 * 0.027**2] * 3, rel=1e-6)
    rims = m["plate"]["rim"]["capsules"]
    radial = [np.linalg.norm(np.asarray(c["from"])[:2]) for c in rims]
    np.testing.assert_allclose(radial, 0.067, atol=1e-9)
    assert all(c["from"][2] == pytest.approx(0.009) for c in rims)
    assert m["task_constants"]["table_top_z"] == pytest.approx(0.74)
    assert m["isaac_physx"]["apple"]["rolling_friction"] is None
    assert any("rolling friction" in u for u in m["unmatched_in_physx"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(version="other"),
        lambda m: m.pop("isaac_physx"),
        lambda m: m["apple"].update(condim=3),
        lambda m: m["apple"].update(mass=float("nan")),
        lambda m: m["plate"]["rim"]["capsules"].pop(),
    ],
)
def test_scene_manifest_validation_rejects(mutate):
    m = copy.deepcopy(load(SCENE))
    mutate(m)
    with pytest.raises(ContractError):
        scene.validate_scene_manifest(m)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(version="other"),
        lambda m: m.pop("image_parity"),
        lambda m: m["camera"].update(quat_wxyz=[1.0, 1.0, 0.0, 0.0]),
        lambda m: m["camera"].update(width=0),
        lambda m: m["camera"].update(znear_m=100.0),
    ],
)
def test_camera_manifest_validation_rejects(mutate):
    m = copy.deepcopy(load(CAMERA))
    mutate(m)
    with pytest.raises(ContractError):
        scene.validate_camera_manifest(m)


def test_camera_manifest_pose_and_intrinsics():
    m = load(CAMERA)
    pos, quat = onboard_camera_offset()
    assert tuple(m["camera"]["pos_m"]) == pos
    np.testing.assert_allclose(scene.camera_quat_xyzw(m), quat, atol=1e-12)
    focal, h_ap, v_ap = scene.pinhole_from_fovy(75.0, 112, 112)
    assert np.rad2deg(2 * np.arctan(v_ap / (2 * focal))) == pytest.approx(75.0)
    assert h_ap == pytest.approx(v_ap)
    assert m["isaac_render"]["status"] == "pinned"
    assert m["isaac_render"]["carb_settings"]["/rtx/rendermode"] == "PathTracing"


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"object_xy": [0.30, -0.25], "plate_xy": [0.50, 0.10]},
        {"object_on_container": True, "plate_xy": [0.45, 0.0]},
        {"object_xy": [0.46, 0.01], "plate_xy": [0.45, 0.0], "object_on_container": True},
        {"object_xy": [0.9, 0.0]},
        {"object_xy": [0.48, -0.12]},
        {"object_xy": [0.50, 0.03], "plate_xy": [0.45, 0.0], "object_on_container": True},
        {"object_on_container": 1},
    ],
)
@pytest.mark.parametrize("seed", [0, 7])
def test_reset_layout_reproduces_mujoco_reset(v2_sim, kwargs, seed):
    try:
        apple, plate = scene.reset_layout(seed, **kwargs)
    except ContractError as exc:
        with pytest.raises(ContractError):
            v2_sim.reset(seed, **kwargs)
        assert str(exc)
        return
    truth = v2_sim.reset(seed, **kwargs)
    np.testing.assert_allclose(apple, truth["object_position"], atol=1e-12)
    np.testing.assert_allclose(plate, truth["plate_position"], atol=1e-12)


def test_task_truth_from_state_matches_mujoco(v2_sim):
    import mujoco

    for kwargs, steps in (({}, 0), ({"object_on_container": True}, 200)):
        v2_sim.reset(0, **kwargs)
        for _ in range(steps):
            mujoco.mj_step(v2_sim.model, v2_sim.data)
        want = v2_sim.task_truth()
        got = scene.task_truth_from_state(
            object_position=want["object_position"],
            object_velocity=want["object_velocity"],
            plate_position=want["plate_position"],
            hand_contact=want["hand_contact"],
            base_position=want["base_position_world"],
            base_rotation=want["base_rotation_world"],
            timestamp=want["timestamp"],
        )
        assert set(got) == set(want)
        for key, value in want.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(got[key], value)
            else:
                assert got[key] == value, key
    assert got["placed"] is True


def test_hand_body_rule_and_settle_time():
    assert scene.is_hand_body("right_hand_middle_1_link")
    assert scene.is_hand_body("left_wrist_yaw_link")
    assert not scene.is_hand_body("table") and not scene.is_hand_body("apple")
    t = [0.0, 0.05, 0.1, 0.15]
    assert scene.settle_time(t, [0.5, 0.01, 0.0005, 0.0]) == 0.1
    assert scene.settle_time(t, [0.5, 0.0, 0.0, 0.01]) is None
    assert scene.settle_time(t, [0.0, 0.0, 0.0, 0.0]) == 0.0


def test_image_metric_on_identical_and_shifted_frames():
    sys.path.insert(0, str(ROOT / "scripts/isaac"))
    import image_parity

    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    cls = np.zeros((32, 32), dtype=np.int64)
    cls[8:16, 8:16] = 3
    same = image_parity.compare(rgb, rgb, cls, cls)
    assert same["mad"] == 0 and same["ssim_luma"] == pytest.approx(1.0)
    assert same["mask_iou"]["apple"] == 1.0 and same["centroid_px"]["apple"] == 0.0
    shifted = np.roll(cls, 2, axis=1)
    other = image_parity.compare(rgb, rgb, shifted, cls)
    assert other["mask_iou"]["apple"] == pytest.approx(48 / 80)
    assert other["centroid_px"]["apple"] == pytest.approx(2.0)


def test_render_freshness_check_rejects_the_pre_fix_behaviour():
    good = {
        "post_reset_read_vs_last_pre_reset_frame": {"max_abs_diff": 149},
        "post_reset_read_vs_first_reset_frame": {"max_abs_diff": 0},
    }
    scene.check_render_freshness(good)
    stale = copy.deepcopy(good)
    stale["post_reset_read_vs_last_pre_reset_frame"]["max_abs_diff"] = 0
    with pytest.raises(RuntimeError, match="stale"):
        scene.check_render_freshness(stale)
    other = copy.deepcopy(good)
    other["post_reset_read_vs_first_reset_frame"]["max_abs_diff"] = 3
    with pytest.raises(RuntimeError):
        scene.check_render_freshness(other)


def _returned_dict_keys(method: str) -> set[str]:
    import ast

    tree = ast.parse((ROOT / "src/embodied_jepa/isaac_transport.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IsaacTransport")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    keys = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Return):
            assert isinstance(node.value, ast.Dict), f"{method} must return a dict literal"
            keys |= {k.value for k in node.value.keys}
    return keys


def test_read_and_reset_carry_no_object_state():
    """Simulator truth is evaluator-only: read() and reset() return robot state and clocks."""
    assert _returned_dict_keys("read") == {
        "rgb",
        "qpos",
        "qvel",
        "joint_names",
        "timestamp",
        "rgb_timestamp",
        "clock",
        "sensor_valid",
    }
    assert _returned_dict_keys("reset") == {"timestamp", "clock"}
