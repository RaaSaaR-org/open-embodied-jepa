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


def test_unknown_physics_backend_is_rejected_before_any_isaac_import():
    from embodied_jepa.isaac_transport import IsaacTransport

    with pytest.raises(ContractError, match="physics"):
        IsaacTransport(usd_path="x.usda", manifest=committed(), scene_manifest={}, physics="ode")
    assert "isaaclab" not in sys.modules


class _FakeWarpArray:
    """Stands in for a ``wp.array``: ``size`` and ``zero_()`` are all the reset uses."""

    def __init__(self, n: int):
        self.values = np.full(n, 7.0)
        self.size = n

    def zero_(self):
        self.values[:] = 0.0


def _fake_newton_transport(monkeypatch, history_fields):
    """An ``IsaacTransport`` on the Newton path with every Isaac object replaced by a fake."""
    from types import SimpleNamespace

    from embodied_jepa.isaac_transport import IsaacTransport

    manifest = committed()
    n = len(manifest["joints"])
    tr = object.__new__(IsaacTransport)
    written = {}

    class Robot:
        def write_joint_position_to_sim_index(self, position):
            written["position"] = np.asarray(position)

        def write_joint_velocity_to_sim_index(self, velocity):
            written["velocity"] = np.asarray(velocity)

        def write_data_to_sim(self):
            pass

        def reset(self):
            pass

        def update(self, dt):
            pass

    fake_torch = SimpleNamespace(
        float32=np.float32,
        as_tensor=lambda x, dtype=None, device=None: np.asarray(x, dtype=dtype),
        zeros_like=np.zeros_like,
    )
    tr.__dict__.update(
        closed=False,
        objects=False,
        newton=True,
        manifest=manifest,
        _idx=np.arange(n),
        _torch=fake_torch,
        sim=SimpleNamespace(device="cpu"),
        robot=Robot(),
        apple=None,
        plate=None,
        camera=None,
        targets=np.zeros(n),
        _nt_solver=SimpleNamespace(mjw_data=SimpleNamespace(**history_fields)),
        _bias_after_reset=None,
    )
    calls = []
    monkeypatch.setattr(tr, "_apply_effort", lambda effort: calls.append("effort"), raising=False)
    monkeypatch.setattr(
        tr, "_check_newton_joint_properties", lambda: calls.append("check"), raising=False
    )
    monkeypatch.setattr(
        tr, "_check_joint_properties", lambda: calls.append("physx_check"), raising=False
    )
    monkeypatch.setattr(tr, "_newton_reset_bias", lambda q: np.full(len(q), 0.5), raising=False)
    return tr, calls, written


def test_newton_reset_clears_mujoco_warp_history_and_sets_the_reset_bias(monkeypatch):
    from embodied_jepa.isaac_transport import NEWTON_HISTORY_FIELDS

    assert set(NEWTON_HISTORY_FIELDS) == {
        "qacc_warmstart",
        "qacc",
        "qfrc_applied",
        "xfrc_applied",
        "act",
        "ctrl",
    }
    fields = {name: _FakeWarpArray(5) for name in NEWTON_HISTORY_FIELDS}
    fields["qpos"] = _FakeWarpArray(5)  # state the reset just wrote: must not be cleared
    tr, calls, written = _fake_newton_transport(monkeypatch, fields)
    out = tr.reset(0)
    assert out["timestamp"] == 0.0
    for name in NEWTON_HISTORY_FIELDS:
        assert not fields[name].values.any(), name
    assert (fields["qpos"].values == 7.0).all()
    assert "check" in calls and "physx_check" not in calls
    q = reset_pose(committed())
    assert np.allclose(written["position"][0], q) and not written["velocity"].any()
    # The first control substep after reset uses the reset-state bias, exactly once.
    assert np.allclose(tr._bias(), 0.5) and tr._bias_after_reset is None


def test_newton_history_clear_skips_absent_or_empty_fields(monkeypatch):
    fields = {"qacc_warmstart": _FakeWarpArray(3), "act": _FakeWarpArray(0)}
    tr, _, _ = _fake_newton_transport(monkeypatch, fields)
    tr.reset(0)
    assert not fields["qacc_warmstart"].values.any()
