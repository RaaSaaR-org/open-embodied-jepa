"""Host-side tests of the Isaac transport's Isaac-free parts (TASK-025).

The transport itself only runs inside the Isaac container; these tests cover the pieces that
must hold everywhere: the module stays out of the core import path, the committed joint
manifest is exactly what the MuJoCo model yields, name mapping rejects mismatches, and the
camera quaternion is a proper rotation onto MuJoCo's ``onboard_rgb`` axes.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa.contracts import ContractError
from embodied_jepa.isaac_transport import (
    manifest_sha256,
    name_map,
    onboard_camera_offset,
    reset_pose,
    validate_joint_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/isaac/g1_dex3_joint_manifest_v1.json"
LEGACY_V0 = ROOT / "configs/isaac/g1_dex3_joint_manifest_v0.json"


def committed() -> dict:
    return json.loads(MANIFEST.read_text())


def test_module_imports_without_isaac_or_torch():
    code = (
        "import sys, embodied_jepa.isaac_transport; "
        "bad = [m for m in ('torch', 'warp', 'isaaclab', 'omni', 'mujoco') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_committed_manifest_is_valid_and_ordered_like_mujoco_actuators():
    manifest = validate_joint_manifest(committed())
    names = [j["name"] for j in manifest["joints"]]
    assert names[0] == "left_hip_pitch_joint" and names[-1] == "right_hand_middle_1_joint"
    q = reset_pose(manifest)
    assert q[names.index("left_elbow_joint")] == 0.08 and np.count_nonzero(q) == 2


def test_committed_manifest_matches_mujoco_model():
    pytest.importorskip("mujoco")
    from embodied_jepa.isaac_transport import joint_manifest_from_mujoco
    from embodied_jepa.simulation import MuJoCoSimulation

    sim = MuJoCoSimulation(render=False)
    try:
        assert manifest_sha256(joint_manifest_from_mujoco(sim)) == manifest_sha256(committed())
        assert np.array_equal(reset_pose(committed()), sim.data.qpos[sim.qadr])
    finally:
        sim.close()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.pop("physics_dt_s"),
        lambda m: m.update(version="other"),
        lambda m: m["joints"].pop(),
        lambda m: m["joints"][1].update(name=m["joints"][0]["name"]),
        lambda m: m["joints"][0].update(kp=float("nan")),
        lambda m: m["joints"][0].update(lower=5.0),
        lambda m: m["joints"][0].update(ctrl_min=1.0),
        lambda m: m.update(control_dt_s=0.051),
        lambda m: m["joints"][0].pop("frictionloss"),
        lambda m: m["joints"][0].update(frictionloss=-0.1),
        lambda m: m.pop("reset_elbow_rad"),
    ],
)
def test_manifest_validation_rejects(mutate):
    manifest = copy.deepcopy(committed())
    mutate(manifest)
    with pytest.raises(ContractError):
        validate_joint_manifest(manifest)


def test_legacy_v0_manifest_is_rejected_but_matches_v1_values():
    legacy = json.loads(LEGACY_V0.read_text())
    with pytest.raises(ContractError):
        validate_joint_manifest(legacy)
    for old, new in zip(legacy["joints"], committed()["joints"], strict=True):
        assert old.pop("frictionloss_not_applied") == new.pop("frictionloss")
        assert old == new


def test_name_map_is_by_name_and_strict():
    canonical = ["a", "b", "c"]
    idx = name_map(canonical, ["c", "a", "b"])
    assert [["c", "a", "b"][i] for i in idx] == canonical
    with pytest.raises(ContractError):
        name_map(canonical, ["a", "b"])
    with pytest.raises(ContractError):
        name_map(canonical, ["a", "b", "d"])
    with pytest.raises(ContractError):
        name_map(canonical, ["a", "b", "c", "c"])


def test_onboard_camera_quaternion_is_a_proper_rotation_of_mujoco_axes():
    pos, (x, y, z, w) = onboard_camera_offset()
    assert pos == (0.08, 0.0, 0.35)
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    cam_y = np.array([0.866, 0.0, 0.5]) / np.linalg.norm([0.866, 0.0, 0.5])
    np.testing.assert_allclose(r[:, 0], [0.0, -1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(r[:, 1], cam_y, atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-9)
