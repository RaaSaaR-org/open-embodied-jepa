"""e9 cross-simulator replay tooling (development, TASK-025): unit checks and the mirror's
plumbing check against MuJoCo. Not evidence of manipulation, and not a learned result."""

import json
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import apple_to_plate_v2 as v2
from embodied_jepa import isaac_e9 as ie
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
NAMES = [
    j["name"]
    for j in json.loads((ROOT / "configs/isaac/g1_dex3_joint_manifest_v1.json").read_text())[
        "joints"
    ]
]


def test_e9_is_the_frozen_task070_expert_and_seeds_are_development_only():
    assert ie.E9 == v2.GATE_EXPERT
    assert ie.check_seeds(ie.SEEDS) == tuple(range(50200, 50216))
    for bad in ((50600,), (53000,), (50199,), (50200, 50200), ()):
        with pytest.raises(ContractError):
            ie.check_seeds(bad)


def test_quaternion_matrix():
    assert np.allclose(ie.quat_xyzw_to_matrix([0, 0, 0, 1]), np.eye(3))
    c = np.cos(np.pi / 4)
    rz = ie.quat_xyzw_to_matrix([0, 0, c, c])  # 90 deg about z
    assert np.allclose(rz @ [1, 0, 0], [0, 1, 0])
    with pytest.raises(ContractError):
        ie.quat_xyzw_to_matrix([0, 0, 0, 0])


def _state(**overrides):
    s = {
        "time": 0.05,
        "q": [0.0] * 43,
        "qd": [0.0] * 43,
        "apple_pos": [0.4, -0.2, 0.77],
        "apple_quat_xyzw": [0, 0, 0, 1],
        "apple_lin_w": [0, 0, 0],
        "apple_ang_w": [0, 0, 0],
        "plate_pos": [0.48, -0.1, 0.746],
        "hand_contact": False,
        "contacts": [],
    }
    s.update(overrides)
    return s


def test_validate_state():
    out = ie.validate_state(_state(), 43)
    assert out["pre"] is None and out["q"].shape == (43,)
    pre = {"q": [0.0] * 43, "apple_pos": [0, 0, 1], "apple_quat_xyzw": [0, 0, 0, 1]}
    assert ie.validate_state(_state(pre=pre), 43)["pre"]["apple_pos"].tolist() == [0, 0, 1]
    bad = dict(_state())
    del bad["contacts"]
    for broken in (
        bad,
        _state(q=[0.0] * 42),
        _state(apple_pos=[np.nan, 0, 0]),
        _state(time=-1.0),
        _state(pre={**pre, "q": [0.0] * 3}),
    ):
        with pytest.raises(ContractError):
            ie.validate_state(broken, 43)


def test_isaac_pair_counts_uses_the_hand_rule():
    contacts = [
        {"bodies": ["apple", "right_hand_index_1_link"]},
        {"bodies": ["apple", "plate"]},
        {"bodies": ["apple", "table"]},
        {"bodies": ["plate", "right_wrist_yaw_link"]},
        {"bodies": ["left_hand_thumb_2_link", "left_hand_index_1_link"]},
        {"bodies": ["floor", "table"]},
    ]
    assert ie.isaac_pair_counts(contacts) == [1, 1, 1, 1, 1]
    assert ie.isaac_pair_counts([]) == [0] * len(ie.ISAAC_PAIR_TYPES)


def test_events_and_settle():
    n = 12
    hand = np.zeros(n, bool)
    hand[2:7] = True
    z = np.full(n, 0.77)
    z[4:7] = [0.79, 0.83, 0.80]
    z[7:] = 0.779
    pos = np.c_[np.zeros((n, 2)), z]
    lin = np.zeros((n, 3))
    lin[:9, 0] = 0.01
    plate = np.zeros(n, bool)
    plate[8:] = True
    ev = ie.events(
        hand=hand, apple_pos=pos, apple_lin=lin, plate_contact=plate, start_z=0.77, open_step=5
    )
    assert ev == {
        "grasp_step": 2,
        "lift_step": 4,
        "lifted_step": 5,
        "open_step": 5,
        "release_step": 7,
        "landing_step": 8,
        "settle_step": 9,
    }
    assert ie.settle_step(np.full((3, 3), 0.01)) is None
    assert ie.settle_step(np.zeros((3, 3))) == 0


def test_joint_groups_cover_the_manifest_and_divergence():
    groups = ie.joint_groups(NAMES)
    assert sum(len(v) for v in groups.values()) == 43
    assert {k: len(v) for k, v in groups.items()} == {
        "legs": 12,
        "waist": 3,
        "arms": 14,
        "hands": 14,
    }
    a = np.zeros((5, 43))
    b = a.copy()
    b[3, NAMES.index("right_elbow_joint")] = 0.05
    out = ie.divergence(a, b[:4], NAMES)
    assert out["common_steps"] == 4
    arms = out["groups"]["arms"]
    assert arms["max"] == pytest.approx(0.05) and arms["worst_joint"] == "right_elbow_joint"
    assert arms["first_over_0.01"] == 3 and arms["first_over_0.1"] is None
    assert out["groups"]["hands"]["max"] == 0.0
    with pytest.raises(ContractError):
        ie.joint_groups(["tail_joint"])


# ----- plumbing: the mirror over a host MuJoCo endpoint reproduces MuJoCo exactly ------------
def test_mirror_over_mujoco_reproduces_a_plain_mujoco_e9_attempt():
    pytest.importorskip("mujoco")
    from embodied_jepa import first_policy as fp
    from embodied_jepa import first_policy_runtime as rt
    from embodied_jepa import resting_expert as rx
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.simulation import MuJoCoSimulation

    class Blind(MuJoCoSimulation):  # same physics; pixels are never read by e9
        def render(self, camera="onboard_rgb"):
            return np.zeros((self.height, self.width, 3), np.uint8)

    plain = G1Embodiment(
        Blind(
            render=False,
            object_kind="apple",
            container_kind="plate",
            width=fp.IMAGE_SIZE,
            height=fp.IMAGE_SIZE,
        )
    )
    v2.apply_v2_scene(plain.model)
    endpoint = ie.MuJoCoEndpoint()
    mirror, _ = ie.make_mirror_robot(endpoint)
    # scripts/evaluate_apple.py:wide_reset(50200), the TASK-070 development reset
    reset = {
        "object_xy": [0.3238940037143524, -0.18840221016330622],
        "plate_xy": [0.47520843833476756, -0.08489034958650456],
    }
    runs = []
    for robot in (plain, mirror):
        recorder = ie.StepRecorder(robot)
        summary, arrays = rx.run_attempt(
            robot,
            rt.configured_bounds(),
            seed=50200,
            reset=reset,
            plate_offset=[0.0, 0.0],
            make_expert=lambda truth: rx.RestingPlaceExpert(truth, **ie.E9),
        )
        runs.append((summary, arrays, recorder.arrays()))
    (s0, a0, r0), (s1, a1, r1) = runs
    assert s0["complete"], s0["stop_reason"]
    assert s1["complete"], s1["stop_reason"]
    assert s0["at_rest"] == s1["at_rest"]
    assert np.array_equal(r0["rec_q"], r1["rec_q"])
    assert np.array_equal(a0["apple_pos"], a1["apple_pos"])
    assert np.array_equal(a0["counts"], a1["counts"])
    assert np.array_equal(r0["rec_targets"], r1["rec_targets"])
    # open loop: the mirror replaying the plain run's commands ends in the same state
    look = len(r0["rec_time"]) - s0["steps"]
    replay, arrays = ie.replay_open_loop(
        mirror,
        seed=50200,
        reset=reset,
        initial_q=r0["rec_initial_q"],
        targets=r0["rec_targets"],
        look_steps=look,
    )
    assert replay["at_rest"] == s0["at_rest"]
    assert np.array_equal(arrays["apple_pos"], a0["apple_pos"])
    endpoint.close()
