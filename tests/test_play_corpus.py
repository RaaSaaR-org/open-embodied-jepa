"""TASK-085 play corpus: layouts, splits, policy bounds, gate arithmetic; the simulator round trip
is graphics opt-in (``JEPA_TEST_RENDER=1``), as the other simulator tests."""

from __future__ import annotations

import os

import numpy as np
import pytest

from embodied_jepa import play_corpus as pc
from embodied_jepa.contracts import ContractError

render = pytest.mark.skipif(os.environ.get("JEPA_TEST_RENDER") != "1", reason="graphics opt-in")


def test_layouts_are_deterministic_on_the_table_and_separated():
    for seed in range(200):
        a = pc.sample_layout(np.random.default_rng([pc.POLICY_SALT, seed]))
        b = pc.sample_layout(np.random.default_rng([pc.POLICY_SALT, seed]))
        assert a == b
        assert 1 <= len(a["objects"]) <= 4
        (px0, px1), (py0, py1) = pc.PLATE_REGION
        assert px0 <= a["plate"][0] <= px1 and py0 <= a["plate"][1] <= py1
        placed = [(np.array(a["plate"]), pc.PLATE_RADIUS)]
        for name, xy in a["objects"].items():
            xy = np.array(xy)
            (x0, x1), (y0, y1) = pc.OBJECT_REGION
            assert x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1
            radius = pc.OBJECTS[name]["radius"]
            for other, r in placed:
                assert np.linalg.norm(xy - other) >= radius + r + 0.01 - 1e-12
            placed.append((xy, radius))
        pc.check_layout(a)


def test_shard_seeds_and_split_rule():
    assert pc.shard_seeds(0)[0] == 850000 and pc.shard_seeds(31)[-1] == 853199
    seeds = pc.shard_seeds(3)
    split = pc.split_assignment(seeds)
    assert len(split["val"]) == 5 and len(split["test"]) == 5 and len(split["train"]) == 90
    assert sorted(split["val"] + split["test"] + split["train"]) == seeds
    ranked = sorted(seeds, key=lambda s: pc.split_rank(s))
    assert split["val"] == sorted(ranked[:5]) and split["test"] == sorted(ranked[5:10])
    # dropping a train episode does not move val or test
    survivor = pc.split_assignment([s for s in seeds if s != split["train"][0]])
    assert survivor["val"] == split["val"] and survivor["test"] == split["test"]
    with pytest.raises(ContractError):
        pc.split_assignment(seeds[:10])


class _FakeRobot:
    manifest = {"translation_per_step_m": 0.015, "rotation_per_step_rad": 0.06}

    def __init__(self):
        self.position = np.array([0.32, -0.15, 0.07])

    def ee_pose(self, side):
        return self.position.copy(), pc.TOP_DOWN.copy()


def _truth(layout):
    position = np.zeros((4, 3))
    for i, name in enumerate(pc.OBJECT_NAMES):
        xy = layout["objects"].get(name)
        if xy is None:
            position[i] = [*pc.PARKING[name], pc.OBJECTS[name]["rest"]]
        else:
            position[i] = [*xy, pc.TABLE_TOP_Z + pc.OBJECTS[name]["rest"]]
    return {"position": position, "plate_position": np.array([*layout["plate"], 0.746])}


@pytest.mark.parametrize("mode", pc.MODES)
def test_policy_requests_are_bounded_and_one_armed(mode):
    rng = np.random.default_rng(7)
    layout = pc.sample_layout(rng)
    policy = pc.PlayPolicy(rng, mode=mode)
    robot = _FakeRobot()
    previous = -1.0
    for _ in range(400):
        action = policy.act(robot, _truth(layout))
        assert action.dtype == np.float32 and action.shape == (14,)
        assert np.all(np.abs(action) <= 1.0)
        assert np.all(action[:6] == 0) and action[12] == -1.0
        assert abs(action[13] - previous) <= pc.GRASP_RAMP + 1e-6
        previous = float(action[13])
    assert policy.skill_log
    if mode == "random":
        assert {s for s, _ in policy.skill_log} == {"wander"}


def _episode(layout, palms, positions, contact):
    n = len(palms)
    return {
        "layout": layout,
        "palms": np.c_[np.asarray(palms, float), np.zeros((n, 21))],
        "object_position": np.asarray(positions, float),
        "robot_contact": np.asarray(contact, bool),
        "grasp_contact": np.zeros((n, 4), bool),
    }


def test_gate_counts_cells_moves_and_rows():
    layout = {"plate": [0.48, -0.05], "objects": {"apple": [0.31, -0.27]}, "yaw": {}}
    start = np.array([[0.31, -0.27, 0.767], *[[*pc.PARKING[n], 0.03] for n in pc.OBJECT_NAMES[1:]]])
    moved = start.copy()
    moved[0, 0] += 0.025
    contact = np.zeros((3, 4), bool)
    contact[1, 0] = True
    palm_in = [0.241, -0.299, 0.061]  # first hand cell
    palm_out = [0.10, 0.0, 0.5]
    ep = _episode(layout, [palm_in, palm_in, palm_out], [start, moved, moved], contact)
    facts = pc.episode_facts(ep)
    assert facts["commands"] == 2 and facts["moved_any"] and facts["touched"][0]
    counts = pc.gate_counts([ep])
    assert counts["hand"].shape == (5, 7, 5) and counts["hand"][0, 0, 0] == 2
    assert counts["hand"].sum() == 2
    assert counts["starts"].shape == (4, 6) and counts["starts"][0, 0] == 1
    assert counts["contacts"][0, 0] == 1 and counts["moved"] == 1
    # moved without contact does not count; contact without 2 cm does not count
    no_touch = _episode(layout, [palm_in] * 3, [start, moved, moved], np.zeros((3, 4), bool))
    assert not pc.episode_facts(no_touch)["moved_any"]
    small = start.copy()
    small[0, 0] += 0.019
    touch_only = _episode(layout, [palm_in] * 3, [start, small, small], contact)
    assert not pc.episode_facts(touch_only)["moved_any"]
    # rows
    full = {
        "episodes": 100,
        "hand": np.full((5, 7, 5), 200),
        "starts": np.full((4, 6), 20),
        "contacts": np.full((4, 6), 10),
        "moved": 20,
    }
    assert pc.gate_row(full, total_commands=720_000)["row"] == "P1-PASS"
    assert pc.gate_row(full, total_commands=719_999)["failing"] == ["G-SIZE"]
    low = dict(full, moved=19)
    assert pc.gate_row(low, total_commands=720_000)["failing"] == ["G-MOVE"]
    hand = full["hand"].copy()
    hand.flat[:9] = 199  # 166/175 = 94.9 % < 95 %
    assert pc.gate_row(dict(full, hand=hand), total_commands=720_000)["failing"] == ["G-HAND"]
    hand.flat[8] = 200  # 167/175 = 95.4 %
    assert pc.gate_row(dict(full, hand=hand), total_commands=720_000)["row"] == "P1-PASS"
    starts = full["starts"].copy()
    starts[3, 5] = 19
    assert pc.gate_row(dict(full, starts=starts), total_commands=720_000)["failing"] == ["G-OBJ"]
    contacts = full["contacts"].copy()
    contacts.flat[:3] = 9  # 21/24 = 87.5 % < 90 %
    row = pc.gate_row(dict(full, contacts=contacts), total_commands=720_000)
    assert row["failing"] == ["G-OBJ"]
    assert pc.gate_row(full, total_commands=720_000, void="x")["row"] == "P1-VOID"


def test_import_keeps_the_core_free_of_mujoco_and_torch():
    import subprocess
    import sys

    code = (
        "import sys, embodied_jepa.play_corpus; "
        "assert 'mujoco' not in sys.modules and 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@render
def test_short_episode_round_trip(tmp_path):
    from embodied_jepa.data import DatasetStore

    robot = pc.make_play_robot()
    assert robot.sim.v2_record["apple_condim"] == 6
    ep = pc.run_episode(robot, 86099, commands=60)
    a = pc.run_episode(robot, 86099, commands=60)
    assert np.array_equal(ep["actions"], a["actions"])
    assert np.array_equal(ep["frames"], a["frames"])
    n = len(ep["actions"])
    assert ep["frames"].shape == (n + 1, 112, 112, 3) and ep["states"].shape[0] == n + 1
    assert np.all(ep["actions"][:, :6] == 0) and np.all(ep["actions"][:, 12] == -1)
    assert np.allclose(np.diff(ep["timestamps"]), 0.05)
    store = DatasetStore.create(
        tmp_path / "shard",
        fps=pc.FPS,
        state_schema=robot.state_schema,
        action_manifest=robot.manifest,
        provenance={"test": True},
    )
    metadata = pc.write_episode(store, robot, ep)
    reopened = DatasetStore(tmp_path / "shard")
    back = reopened.read_episode(pc.episode_id(86099))
    assert np.array_equal(back.actions, ep["actions"])
    row = reopened.manifest["episodes"][0]
    side = pc.load_sidecar(tmp_path / "shard", row)
    assert np.array_equal(side["palms"], ep["palms"])
    assert metadata["facts"] == pc.episode_facts(side)
