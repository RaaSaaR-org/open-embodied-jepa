"""SO-101 S1 (TASK-097): pinned SO-101 MJCF, tabletop transport and joint-delta embodiment.

Headless control tests replace only RGB with explicit fixture pixels (qpos, qvel and time are
real physics) and never claim renderer coverage; the camera test opts in with
JEPA_TEST_RENDER=1. Physics tests skip only when the pinned assets are not fetched
(scripts/fetch_so101.py, which scripts/fetch_assets.py also runs, so the integration CI job
fetches them).
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from embodied_jepa import config as config_module
from embodied_jepa.action_schemas import JOINT_DELTA_SO101_V0
from embodied_jepa.config import ExperimentConfig
from embodied_jepa.contracts import ContractError
from embodied_jepa.registry import EMBODIMENTS
from embodied_jepa.so101 import DEFAULT_ACTION_MANIFEST, load_action_manifest

ROOT = Path(__file__).resolve().parents[1]
ASSET_MANIFEST = json.loads((ROOT / "assets/so101_manifest.json").read_text())
FETCHED = all(
    (ROOT / ASSET_MANIFEST["local_path"] / relative).is_file()
    for relative in ASSET_MANIFEST["sha256"]
)
needs_assets = pytest.mark.skipif(not FETCHED, reason="pinned SO-101 assets have not been fetched")


def _fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_so101", ROOT / "scripts/fetch_so101.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----- no MuJoCo needed -------------------------------------------------------------------------
def test_asset_manifest_pins_revision_licence_and_every_file():
    m = ASSET_MANIFEST
    assert m["source"] == "https://github.com/google-deepmind/mujoco_menagerie.git"
    assert len(m["revision"]) == 40 and int(m["revision"], 16) >= 0
    assert m["license"] == "Apache-2.0" and m["license_file"] in m["sha256"]
    assert m["robot"] in m["sha256"]
    assert all(k.startswith(m["sparse_path"] + "/") for k in m["sha256"])
    assert all(len(v) == 64 and int(v, 16) >= 0 for v in m["sha256"].values())
    # The checkout goes to the ignored third_party/; nothing upstream is committed.
    assert m["local_path"].startswith("third_party/")
    assert "/third_party" in (ROOT / ".gitignore").read_text().splitlines()


def test_fetch_verification_rejects_a_modified_missing_or_extra_file(tmp_path):
    fetch = _fetch_module()
    sparse = ASSET_MANIFEST["sparse_path"]
    manifest = {"sparse_path": sparse, "sha256": {}}
    for relative, data in ((f"{sparse}/a.xml", b"<mujoco/>"), (f"{sparse}/LICENSE", b"text")):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest["sha256"][relative] = hashlib.sha256(data).hexdigest()
    fetch.verify(tmp_path, manifest)
    path = tmp_path / next(iter(manifest["sha256"]))
    path.write_bytes(b"<mujoco></mujoco>")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        fetch.verify(tmp_path, manifest)
    extra = tmp_path / manifest["sparse_path"] / "extra.stl"
    extra.write_bytes(b"")
    with pytest.raises(RuntimeError, match="file set differs"):
        fetch.verify(tmp_path, manifest)
    extra.unlink()
    path.unlink()
    with pytest.raises(RuntimeError, match="file set differs"):
        fetch.verify(tmp_path, manifest)


def test_action_manifest_is_valid_and_rejects_bad_fields(tmp_path):
    m = load_action_manifest()
    assert m["action_schema"] == "joint_delta_so101_v0"
    assert tuple(m["joint_limits_rad"]) == JOINT_DELTA_SO101_V0.names
    # Powers of two keep the arm channels exact under float32 scaling.
    for scale in m["joint_delta_per_step_rad"].values():
        assert np.log2(scale) == np.round(np.log2(scale))
    good = json.loads(DEFAULT_ACTION_MANIFEST.read_text())
    for change, message in (
        ({"action_schema": "ee_delta_grasp_v0"}, "identity"),
        ({"calibration": "hardware"}, "identity"),
        ({"control_dt_s": 0}, "control_dt_s"),
        ({"gripper_max_delta_per_step_rad": float("nan")}, "gripper_max_delta"),
        ({"joint_delta_per_step_rad": {"shoulder_pan": 0.1}}, "joint_delta_per_step_rad"),
        ({"gripper_open_rad": 3.0}, "gripper range"),
        ({"home_rad": [0, 0, 0, 0, 0, 9.0]}, "home_rad"),
    ):
        bad = copy.deepcopy(good)
        bad.update(change)
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ContractError, match=message):
            load_action_manifest(path)


def test_so101_is_registered_lazily_and_core_import_stays_clean():
    assert config_module.EMBODIMENT_ACTION_SCHEMAS["so101"] == "joint_delta_so101_v0"
    assert EMBODIMENTS._factories["so101"] == "embodied_jepa.so101:SO101Embodiment"
    assert config_module.EMBODIMENT_ACTION_SCHEMAS["unitree_g1_dex3"] == "ee_delta_grasp_v0"
    code = (
        "import sys, embodied_jepa, embodied_jepa.config, embodied_jepa.so101, "
        "embodied_jepa.so101_simulation, embodied_jepa.so101_scripted;"
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_so101_config_is_refused_at_load_until_it_has_a_task(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "native.pt").touch()
    shutil.copy(DEFAULT_ACTION_MANIFEST, tmp_path / "action.json")
    raw = {
        "schema_version": 0,
        "world_model": {"backend": "native_jepa", "checkpoints": {"native_jepa": "native.pt"}},
        "embodiment": {
            "backend": "so101",
            "transport": "mujoco",
            "action_schema": "joint_delta_so101_v0",
            "action_manifest": "action.json",
        },
        "planner": {"horizon": 3, "samples": 16, "elites": 4},
        "dataset": {"root": "data"},
        "task": {"name": "reach"},
    }
    path = tmp_path / "so101.yaml"
    path.write_text(yaml.safe_dump(raw))
    # reach and apple_to_plate are G1 tasks run by the G1-only benchmark: refuse early.
    with pytest.raises(ValueError, match="has no task 'reach'"):
        ExperimentConfig.load(path)
    raw["embodiment"]["action_schema"] = "ee_delta_grasp_v0"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="does not execute action schema"):
        ExperimentConfig.load(path)


# ----- MuJoCo physics (headless) ----------------------------------------------------------------
@pytest.fixture
def robot(monkeypatch):
    pytest.importorskip("mujoco")
    if not FETCHED:
        pytest.skip("pinned SO-101 assets have not been fetched")
    from embodied_jepa.so101_simulation import SO101Simulation

    sim = SO101Simulation(render=False)
    # Only RGB is replaced in these control tests; qpos/qvel/time are actual physics.
    monkeypatch.setattr(sim, "render", lambda camera="wrist_rgb": np.zeros((8, 8, 3), np.uint8))
    embodiment = EMBODIMENTS.create("so101", simulation=sim)
    embodiment.reset(0)
    yield embodiment
    embodiment.close()


def _run(robot, action, steps):
    results = []
    for _ in range(steps):
        robot.observe()
        results.append(robot.execute(np.asarray(action, dtype=np.float32)))
    return results


@needs_assets
def test_scene_matches_the_pinned_model_and_satisfies_the_protocols(robot):
    from embodied_jepa.contracts import Embodiment, Transport

    assert isinstance(robot, Embodiment) and isinstance(robot.sim, Transport)
    sim = robot.sim
    assert sim.joint_names == JOINT_DELTA_SO101_V0.names
    assert sim.model.nu == 6 and sim.substeps == 10
    assert sim.model.camera("wrist_cam").id >= 0 and sim.model.camera("overview").id >= 0
    assert robot.action_schema is JOINT_DELTA_SO101_V0
    assert robot.state_schema.version == "so101_proprio_v0"
    assert robot.state_schema.dimension == 12
    observation = robot.observe()
    assert set(observation.images) == {"wrist_rgb", "overview_rgb"}
    # Simulator truth never enters the observation.
    assert all("cube" not in n and "target" not in n for n in robot.state_schema.names)
    np.testing.assert_allclose(observation.robot_state[0, :6], robot.home, atol=1e-6)


@needs_assets
def test_asset_hash_is_checked_before_loading(tmp_path):
    pytest.importorskip("mujoco")
    from embodied_jepa.so101_simulation import SO101Simulation, default_asset_root

    copy_root = tmp_path / "menagerie"
    shutil.copytree(
        default_asset_root() / ASSET_MANIFEST["sparse_path"], copy_root / "robotstudio_so101"
    )
    SO101Simulation(asset_root=copy_root, render=False).close()
    xml = copy_root / ASSET_MANIFEST["robot"]
    xml.write_text(xml.read_text().replace('damping="1"', 'damping="2"'))
    with pytest.raises(ContractError, match="fetch_so101"):
        SO101Simulation(asset_root=copy_root, render=False)


@needs_assets
def test_normalization_round_trip_is_exact_on_the_arm(robot):
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, (256, 6)).astype(np.float32)
    actions[:4] = [[-1] * 6, [1] * 6, [0] * 6, [0.5, -0.5, 0.25, -0.25, 0.125, -0.125]]
    for action in actions:
        physical = robot.denormalize_action(action)
        back = robot.normalize_action(physical)
        np.testing.assert_array_equal(back[:5], action[:5])  # power-of-two scales: exact
        assert abs(float(back[5]) - float(action[5])) <= 2e-7  # affine gripper map: 1-2 ulp
    np.testing.assert_array_equal(robot.denormalize_action(actions[1])[:5], robot.scales)
    assert robot.denormalize_action(actions[0])[5] == pytest.approx(robot.gripper_closed)
    assert robot.denormalize_action(actions[1])[5] == pytest.approx(robot.gripper_open)
    with pytest.raises(ContractError):
        robot.denormalize_action(np.zeros(14, dtype=np.float32))
    with pytest.raises(ContractError):
        robot.normalize_action(np.zeros(6))  # float64
    with pytest.raises(ContractError):
        robot.normalize_action(np.full(6, 9, np.float32))  # outside [-1, 1]


@needs_assets
def test_zero_arm_action_holds_pose(robot):
    start = robot.sim.data.qpos[robot.sim.qadr].copy()
    targets = robot.sim.targets.copy()
    results = _run(robot, robot.hold_action(), 20)
    assert all(r.status == "applied" for r in results)
    np.testing.assert_array_equal(robot.sim.targets[:5], targets[:5])
    assert abs(robot.sim.targets[5] - targets[5]) < 1e-6
    np.testing.assert_allclose(robot.sim.data.qpos[robot.sim.qadr], start, atol=2e-3)


@needs_assets
def test_joint_limit_is_clipped_and_recorded_never_silently(robot):
    action = robot.hold_action()
    action[0] = 1.0  # shoulder_pan +1/16 rad per step toward its 1.91986 rad limit
    results = _run(robot, action, 40)
    upper = robot.limits[0, 1]
    assert robot.sim.targets[0] == upper
    clipped = [r for r in results if r.status == "clipped"]
    assert clipped and all("shoulder_pan target clipped" in r.reason for r in clipped)
    for result in clipped:
        np.testing.assert_array_equal(result.requested_action, action)  # stored unchanged
        assert 0 <= result.applied_action[0] < 1
    assert results[-1].applied_action[0] == 0  # at the limit nothing more is applied
    # The transport refuses out-of-range targets outright.
    bad = robot.sim.targets.copy()
    bad[1] = 5.0
    with pytest.raises(ContractError, match="limits"):
        robot.sim.send_joint_targets(bad, joint_names=robot.sim.joint_names, deadline=1e9)


@needs_assets
def test_gripper_is_rate_limited_and_closes_through_physics(robot):
    start = robot.sim.data.qpos[robot.sim.qadr[5]]
    action = robot.hold_action()
    action[5] = -1.0
    results = _run(robot, action, 12)
    step = robot.manifest["gripper_max_delta_per_step_rad"]
    first = results[0]
    assert first.status == "clipped" and "gripper target rate-limited" in first.reason
    expected = robot._gripper_normalized(robot.home[5] - step)
    assert first.applied_action[5] == pytest.approx(expected, abs=1e-6)
    assert results[-1].status == "applied"  # the target reached the closed end
    assert robot.sim.targets[5] == pytest.approx(robot.gripper_closed)
    assert robot.sim.data.qpos[robot.sim.qadr[5]] < start - 0.5


@needs_assets
def test_scripted_joint_trajectory_moves_the_arm_and_is_deterministic(robot):
    plan = [
        np.array([0.5, -0.5, 0.5, 0.0, 0.25, 0.8], np.float32),
        np.array([-0.5, 0.0, -0.5, 0.5, 0.0, -0.2], np.float32),
    ]

    def trace(seed):
        robot.reset(seed)
        rows = []
        for action in plan:
            for result in _run(robot, action, 8):
                assert result.status in ("applied", "clipped")
                rows.append(robot.sim.data.qpos.copy())
        return np.array(rows)

    first, second = trace(3), trace(3)
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
    moved = np.abs(first[-1][robot.sim.qadr] - first[0][robot.sim.qadr])
    assert moved[0] > 0.05 and moved[5] > 0.1
    assert robot.sim.data.time == pytest.approx(0.05 * 16)


@needs_assets
def test_stale_or_consumed_observations_stop_and_hold(robot):
    action = robot.hold_action()
    action[1] = 0.5
    robot.observe()
    assert robot.execute(action).status == "applied"
    result = robot.execute(action)  # the snapshot was consumed by the previous step
    assert result.status == "stopped" and "observe" in result.reason
    assert result.applied_action is None
    result = robot.execute(action)
    assert result.status == "stopped" and result.reason == "no synchronized observation"


@needs_assets
def test_measured_joint_velocity_limit_stops_and_holds(robot):
    robot.observe()
    limit = robot.manifest["measured_joint_velocity_stop_rad_s"]
    robot.sim.data.qvel[robot.sim.vadr[1]] = 2 * limit
    result = robot.execute(robot.hold_action())
    assert result.status == "stopped" and result.reason == "measured joint velocity limit exceeded"
    assert result.applied_action is None and robot.sim.stopped_reason == result.reason


@needs_assets
def test_wall_clock_planning_deadline_stops_and_holds(robot, monkeypatch):
    robot.observe()
    import embodied_jepa.so101 as so101_module

    later = so101_module.time.monotonic() + robot.manifest["max_wall_snapshot_age_s"] + 1.0
    monkeypatch.setattr(so101_module.time, "monotonic", lambda: later)
    result = robot.execute(robot.hold_action())
    assert result.status == "stopped" and result.reason == "wall-clock planning deadline expired"
    assert result.applied_action is None


@needs_assets
def test_transport_rejects_an_expired_command_deadline_and_holds(robot):
    sim = robot.sim
    before = sim.data.time
    ack = sim.send_joint_targets(sim.targets.copy(), joint_names=sim.joint_names, deadline=-1.0)
    assert ack["status"] == "rejected" and ack["reason"] == "command deadline expired"
    assert sim.data.time == before  # no physics step was taken
    np.testing.assert_allclose(sim.targets, sim.data.qpos[sim.qadr])
    ack = sim.send_joint_targets(
        sim.targets.copy(), joint_names=sim.joint_names, deadline=float("nan")
    )
    assert ack["status"] == "rejected"
    # Through the embodiment, a transport rejection surfaces as a stop with its reason.
    robot.observe()
    original = sim.send_joint_targets
    sim.send_joint_targets = lambda targets, *, joint_names, deadline: original(
        targets, joint_names=joint_names, deadline=-1.0
    )
    result = robot.execute(robot.hold_action())
    assert result.status == "stopped" and result.reason == "command deadline expired"


@needs_assets
def test_offscreen_buffer_fits_a_wide_render_request():
    pytest.importorskip("mujoco")
    from embodied_jepa.so101_simulation import SO101Simulation

    sim = SO101Simulation(width=800, height=480, render=False)
    try:
        assert sim.model.vis.global_.offwidth >= 800
        assert sim.model.vis.global_.offheight >= 640
    finally:
        sim.close()


@needs_assets
def test_reset_validates_poses_and_seeds_are_reproducible(robot):
    a = robot.reset(5)
    b = robot.reset(5)
    np.testing.assert_array_equal(a["cube_position"], b["cube_position"])
    np.testing.assert_array_equal(a["target_position"], b["target_position"])
    assert not np.array_equal(a["cube_position"], robot.reset(6)["cube_position"])
    assert a["cube_position"][2] == pytest.approx(0.015)
    with pytest.raises(ContractError, match="cube reset"):
        robot.reset(0, cube_xy=[2.0, 0.0])
    with pytest.raises(ContractError, match="joint positions"):
        robot.reset(0, joint_positions=[0, 0, 0, 0, 0, 9])


@needs_assets
def test_privileged_scripted_lift_raises_the_cube(robot):
    """Development check only: the scene is reachable and the jaw can hold the cube."""
    from embodied_jepa.so101_scripted import scripted_lift

    record = scripted_lift(robot, 0)
    assert record["lift"]["jaw_contacts"] == [True, True]
    assert record["lifted_cm"] > 5


# ----- graphics ---------------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("JEPA_TEST_RENDER") != "1", reason="graphics opt-in")
def test_both_cameras_render_the_cube_repeatably():
    pytest.importorskip("mujoco")
    from embodied_jepa.so101_simulation import SO101Simulation

    sim = SO101Simulation(width=96, height=96)
    try:
        robot = EMBODIMENTS.create("so101", simulation=sim)
        robot.reset(0, cube_xy=[0.24, 0.0], cube_yaw=0.0)
        first = robot.observe()
        for name in ("wrist_rgb", "overview_rgb"):
            image = first.images[name][0]
            assert image.shape == (96, 96, 3) and image.std() > 5
            red = (image[..., 0] > 140) & (image[..., 1] < 80) & (image[..., 2] < 80)
            assert red.sum() >= 4, name  # the red cube is in view from the home pose
        robot.reset(0, cube_xy=[0.24, 0.0], cube_yaw=0.0)
        again = robot.observe()
        for name in first.images:
            np.testing.assert_array_equal(first.images[name], again.images[name])
    finally:
        sim.close()
