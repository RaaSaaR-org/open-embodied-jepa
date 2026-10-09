"""TASK-086 converter: name mapping, resampling, rotation inverse, grasp projection, cutting, rows.

Forward kinematics needs the pinned G1 assets and MuJoCo; that test is skipped when they are
absent (it renders nothing)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from embodied_jepa import real_g1_convert as rc
from embodied_jepa.embodiment import rotation_delta


def test_unitree_names_map_to_our_mjcf_names():
    names = [f"k{s}{p}" for s in ("Left", "Right") for p, _ in rc.ARM_PARTS] + [
        f"k{s}Hand{p}"
        for s in ("Left", "Right")
        for p in ("Thumb0", "Thumb1", "Thumb2", "Middle0", "Middle1", "Index0", "Index1")
    ]
    mapped = rc.map_names(names, provider="unitree")
    assert None not in mapped and len(set(mapped)) == 28
    assert mapped[0] == "left_shoulder_pitch_joint" and mapped[13] == "right_wrist_yaw_joint"
    # by name, not by position: Unitree's left hand lists middle before index
    assert mapped[17] == "left_hand_middle_0_joint" and mapped[19] == "left_hand_index_0_joint"
    assert rc.map_names(["kWaistYaw"], provider="unitree") == [None]


def test_resample_interpolates_and_propagates_invalid_frames():
    q = np.arange(10, dtype=float)[:, None]
    valid = np.ones(10, bool)
    valid[4] = False
    q20, v20, img = rc.resample(q, valid, 30.0)
    assert np.allclose(q20[:, 0], np.arange(len(q20)) * 1.5)
    assert len(q20) == 7  # s = 0, 1.5, ..., 9.0
    # s = 3.0 (exact frame 3) is valid; s = 4.5 (frames 4, 5) is not; s = 6.0 is valid
    assert not v20[3] and v20[2] and v20[4]
    assert img.tolist() == [0, 2, 3, 5, 6, 8, 9]  # floor(1.5 k + 0.5)
    q20, v20, img = rc.resample(q, valid, 20.0)
    assert np.array_equal(q20, q) and np.array_equal(v20, valid)


def test_rpy_inverts_rotation_delta():
    rng = np.random.default_rng(0)
    for _ in range(50):
        r = rng.uniform(-0.5, 0.5, 3)
        assert np.allclose(rc.rpy_from_matrix(rotation_delta(r)), r, atol=1e-9)


def test_segments_cut_at_flags_and_drop_short_runs():
    flags = np.zeros(60, bool)
    flags[[10, 45]] = True
    assert rc.segments(flags) == [(11, 45)]
    assert rc.segments(np.zeros(25, bool)) == [(0, 25)]
    assert rc.segments(np.zeros(19, bool)) == []


def test_split_rule_rounding():
    keys = [f"{i:06d}" for i in range(41)]
    s = rc.split_for("repo", keys)
    assert len(s["val"]) == 2 and len(s["test"]) == 2 and len(s["train"]) == 37
    s = rc.split_for("repo", keys[:5])
    assert len(s["val"]) == 1 and len(s["test"]) == 1 and len(s["train"]) == 3


def test_decide_rows():
    good = {
        "arm_out": 10,
        "valid_steps": 100,
        "left_median_residual": 0.1,
        "right_median_residual": 0.1,
        "left_outside_fraction": 0.1,
        "right_outside_fraction": 0.25,
    }
    assert rc.decide(good)["row"] == "R-KEEP"
    assert rc.decide(dict(good, arm_out=51))["row"] == "R-DROP-RANGE"
    assert rc.decide(dict(good, arm_out=50))["row"] == "R-KEEP"
    assert rc.decide(dict(good, right_median_residual=0.151))["row"] == "R-DROP-GRASP"
    assert rc.decide(dict(good, left_outside_fraction=0.26))["row"] == "R-DROP-GRASP"
    assert rc.decide(dict(good, arm_out=99), void="x")["row"] == "R-VOID"


def _kinematics():
    pytest.importorskip("mujoco")
    try:
        return rc.Kinematics()
    except Exception as error:  # missing pinned assets
        pytest.skip(f"G1 assets unavailable: {error}")


def test_grasp_projection_and_palm_deltas():
    kin = _kinematics()
    manifest = kin.manifest
    names = [f"{s}_hand_{f}_joint" for s in ("left", "right") for f in rc.FINGERS]
    names += ["right_elbow_joint"]
    o = np.r_[manifest["left_open_rad"], manifest["right_open_rad"]]
    c = np.r_[manifest["left_closed_rad"], manifest["right_closed_rad"]]
    q = np.stack([np.r_[o, 0.3], np.r_[c, 0.3], np.r_[(o + c) / 2, 0.3]])
    t, g, res = kin.grasp(names, q, "right")
    assert np.allclose(t, [0, 1, 0.5]) and np.allclose(g, [-1, 1, 0]) and np.allclose(res, 0)
    # a 4-frame episode at 20 Hz moving only the right elbow: left palm still, right palm moves
    elbow = np.array([0.3, 0.31, 0.32, 0.33])
    qs = np.c_[np.tile(o, (4, 1)), elbow]
    conv = rc.convert_episode(kin, names, qs, 20.0)
    assert conv.actions.shape == (3, 14)
    assert np.allclose(conv.actions[:, :6], 0, atol=1e-9)
    assert np.abs(conv.actions[:, 6:9]).max() > 0
    assert np.allclose(conv.actions[:, 12:], -1)
    assert conv.state_mask[0, kin.joint_names.index("right_elbow_joint")]
    assert not conv.state_mask[0, kin.joint_names.index("left_knee_joint")]
    # an out-of-range joint invalidates its frame and flags both neighbouring steps
    qs[2, -1] = 10.0
    conv = rc.convert_episode(kin, names, qs, 20.0)
    assert not conv.valid[2] and conv.flagged[1] and conv.flagged[2] and not conv.flagged[0]
    json.dumps(rc.episode_stats(conv))
