import subprocess
import sys

import numpy as np
import pytest

from embodied_jepa import arena_transport as at
from embodied_jepa.contracts import ContractError

NAMES = [f"j{i}" for i in range(at.G1_NUM_JOINTS)]


def test_import_stays_light():
    code = (
        "import sys, embodied_jepa.arena_transport\n"
        "bad = [m for m in ('torch', 'isaaclab', 'isaaclab_arena', 'omni', 'warp', 'mujoco')"
        " if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_tutorial_argv_matches_the_tutorial_command():
    argv = at.tutorial_argv()
    assert argv == [
        "--headless",
        "--enable_cameras",
        "galileo_g1_static_pick_and_place",
        "--object",
        "apple_01_objaverse_robolab",
        "--destination",
        "clay_plates_hot3d_robolab",
        "--embodiment",
        "g1_wbc_agile_joint",
    ]
    assert "--num_envs" in at.tutorial_argv(num_envs=5)


def test_wbc_action_layout():
    targets = {n: 0.01 * i for i, n in enumerate(NAMES)}
    a = at.wbc_action(targets, list(reversed(NAMES)), navigate=(0.1, 0.2, 0.3))
    assert a.shape == (at.ACTION_DIM,) == (50,)
    assert a.dtype == np.float32
    # mapped by name, in the articulation's order
    assert a[0] == pytest.approx(targets[NAMES[-1]])
    assert a[42] == pytest.approx(0.0)
    assert a[at.NAVIGATE].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert a[at.BASE_HEIGHT] == pytest.approx(at.STANDING_HEIGHT_M)
    assert a[-4] == a[at.BASE_HEIGHT]  # Arena's own "actions[:, -4] = 0.75"
    assert a[at.TORSO_RPY].tolist() == [0.0, 0.0, 0.0]


def test_wbc_action_refuses_mismatched_joints():
    targets = {n: 0.0 for n in NAMES}
    with pytest.raises(ContractError):
        at.wbc_action({**targets, "extra": 0.0}, NAMES)
    with pytest.raises(ContractError):
        at.wbc_action({n: 0.0 for n in NAMES[:-1]}, NAMES)
    with pytest.raises(ContractError):
        at.wbc_action(targets, NAMES[:-1])
    with pytest.raises(ContractError):
        at.wbc_action({**targets, NAMES[0]: float("nan")}, NAMES)


def test_projection_ros_convention():
    k = np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])
    ident = (0.0, 0.0, 0.0, 1.0)  # camera axes = world axes: +Z forward, +X right, +Y down
    uv = at.project_points_ros([[0, 0, 2], [1, 0, 2], [0, 1, 2], [0, 0, -1]], [0, 0, 0], ident, k)
    assert uv[0].tolist() == pytest.approx([50, 40, 2])
    assert uv[1, 0] == pytest.approx(100)  # +X -> right
    assert uv[2, 1] == pytest.approx(90)  # +Y -> down
    assert np.isnan(uv[3, 0]) and uv[3, 2] == pytest.approx(-1)
    # camera looking along world +X (rotate +90 deg about Y): a point ahead is centred
    s = np.sqrt(0.5)
    uv = at.project_points_ros([[3, 0, 0]], [0, 0, 0], (0.0, s, 0.0, s), k)
    assert uv[0].tolist() == pytest.approx([50, 40, 3])


def test_fov_and_patch():
    # Isaac Lab's PinholeCameraCfg default aperture 20.955 mm with Arena's 15 mm focal length
    assert at.horizontal_fov_deg(15.0, 20.955) == pytest.approx(69.87, abs=0.01)
    img = np.zeros((10, 10, 3), np.uint8)
    img[4:7, 4:7] = (200, 100, 50)
    assert at.patch_mean_rgb(img, 5, 5, half=1) == pytest.approx([200, 100, 50])
    assert at.patch_mean_rgb(img, 0, 5) is None
    assert at.patch_mean_rgb(img, float("nan"), 5) is None


def test_pxr_pin_survives_later_writes():
    import os

    code = (
        "import os\n"
        "from embodied_jepa.arena_transport import pin_pxr_work_thread_limit\n"
        "os.environ.pop('PXR_WORK_THREAD_LIMIT', None)\n"
        "assert pin_pxr_work_thread_limit() == '1'\n"
        "os.environ['PXR_WORK_THREAD_LIMIT'] = '16'\n"
        "assert os.environ['PXR_WORK_THREAD_LIMIT'] == '1'\n"
        "os.environ['OTHER'] = '16'\n"
        "assert os.environ['OTHER'] == '16'\n"
        "assert pin_pxr_work_thread_limit('0') is None\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=dict(os.environ))
    with pytest.raises(ContractError):
        at.pin_pxr_work_thread_limit("x", environ={})
    with pytest.raises(ContractError):  # a plain mapping cannot be pinned (no class swap)
        at.pin_pxr_work_thread_limit("1", environ={})
    with pytest.raises(ContractError):
        at.pin_pxr_work_thread_limit(None, environ={"PXR_WORK_THREAD_LIMIT": "4"})


def test_is_blank():
    assert at.is_blank(np.zeros((4, 4, 3), np.uint8))
    img = np.zeros((4, 4, 3), np.uint8)
    img[0, 0, 0] = 30
    assert not at.is_blank(img)


def test_runner_wrapper_allows_local_policies_only():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/isaac/arena_policy_runner.py"
    spec = importlib.util.spec_from_file_location("arena_policy_runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.strip_output(["--output", "x", "--a", "--output=y", "b"]) == ["--a", "b"]
    mod.check_args(["--policy_type", "zero_action", "--num_steps", "600"])
    mod.check_args(["--policy_type=replay"])
    for bad in (
        [],
        ["--policy_type", "pkg.mod.Policy"],
        ["--policy_type", "zero_action", "--remote_host", "localhost"],
        ["--policy_type", "zero_action", "--port", "5555"],
        ["--policy_type", "isaaclab_arena_gr00t.policy.X"],
    ):
        with pytest.raises(SystemExit):
            mod.check_args(bad)
