"""SO-101 S3 (TASK-099): the scripted play collector, its storage and the corpus measures.

Pure tests need NumPy only. Physics tests run headless (black frames in place of RGB) and skip
only when the pinned SO-101 assets are not fetched; the tiny corpus is collected and measured on
the CPU through the scripts' own ``main``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import so101_play_corpus as pc

ROOT = Path(__file__).resolve().parents[1]
ASSET_MANIFEST = json.loads((ROOT / "assets/so101_manifest.json").read_text())
FETCHED = all((ROOT / ASSET_MANIFEST["local_path"] / p).is_file() for p in ASSET_MANIFEST["sha256"])
HOME = np.array([0.0, -1.0, 1.0, 0.8, 0.0, 0.6])


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----- pure -----
def test_importing_the_collector_loads_no_mujoco_or_torch():
    code = (
        "import sys, embodied_jepa, embodied_jepa.so101_play_corpus; "
        "print(any(m in sys.modules for m in ('mujoco', 'torch')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_split_rule_is_deterministic_disjoint_and_complete():
    seeds = list(range(991000, 991100))
    split = pc.split_assignment(seeds, n_val=5, n_test=5)
    assert split == pc.split_assignment(list(reversed(seeds)), n_val=5, n_test=5)
    assert (len(split["val"]), len(split["test"]), len(split["train"])) == (5, 5, 90)
    flat = split["val"] + split["test"] + split["train"]
    assert sorted(flat) == seeds
    with pytest.raises(pc.ContractError):
        pc.split_assignment([1, 2], n_val=1, n_test=1)


def test_seed_ranges_do_not_overlap():
    corpus = set(pc.corpus_seed_range())
    pilot = set(range(pc.PILOT_FIRST_SEED, pc.PILOT_FIRST_SEED + 64))
    assert len(corpus) == 3000
    assert not corpus & pilot and not corpus & set(pc.DEV_SEEDS) and not pilot & set(pc.DEV_SEEDS)


def test_reset_is_deterministic_and_inside_its_regions():
    for seed in range(50):
        a = pc.sample_reset(np.random.default_rng([pc.POLICY_SALT, seed]), HOME)
        b = pc.sample_reset(np.random.default_rng([pc.POLICY_SALT, seed]), HOME)
        assert a == b
        (x0, x1), (y0, y1) = pc.CUBE_REGION
        assert x0 <= a["cube_xy"][0] <= x1 and y0 <= a["cube_xy"][1] <= y1
        assert np.linalg.norm(np.subtract(a["cube_xy"], a["target_xy"])) >= pc.MIN_CUBE_TARGET_M
        joints = np.asarray(a["joint_positions"])
        assert np.all(np.abs(joints[:5] - HOME[:5]) <= pc.HOME_JITTER_RAD)


def test_the_play_cube_region_contains_the_s2_task_boxes():
    from embodied_jepa.so101_tasks import RESETS_V0

    (x0, x1), (y0, y1) = pc.CUBE_REGION
    assert x0 <= RESETS_V0.cube_x_m[0] and RESETS_V0.cube_x_m[1] <= x1
    assert y0 <= RESETS_V0.cube_y_m[0] and RESETS_V0.cube_y_m[1] <= y1
    (tx0, tx1), (ty0, ty1) = pc.TARGET_REGION
    assert tx0 <= RESETS_V0.target_x_m[0] and RESETS_V0.target_x_m[1] <= tx1
    assert ty0 <= RESETS_V0.target_y_m[0] and RESETS_V0.target_y_m[1] <= ty1


def test_windows_count_disjoint_runs():
    mask = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]
    assert pc._windows(mask, 3) == 2
    assert pc._windows(mask, 2) == 3
    assert pc._windows([0, 0], 1) == 0


def _synthetic_episode(heights, contact):
    steps = len(heights)
    position = np.zeros((steps, 3))
    position[:, 0], position[:, 2] = 0.2, pc.REST_Z + np.asarray(heights)
    contact = np.asarray(contact, bool)
    return {
        "cube_position": position,
        "cube_velocity": np.zeros((steps, 3)),
        "cube_angular_velocity": np.zeros((steps, 3)),
        "gripper_contact": contact,
        "fixed_jaw_contact": contact,
        "moving_jaw_contact": contact,
        "target_position": np.array([0.3, 0.1, 0.0]),
        "attempts": [{"skill": "pick_lift", "failure": "none", "regrasp": 0, "lift_end_step": 25}],
    }


def test_facts_count_a_held_lift_and_a_put_down_but_not_a_transient():
    heights = [0.0] * 5 + [0.06] * 25 + [0.0] * 25
    contact = [False] * 5 + [True] * 25 + [False] * 25
    facts = pc.episode_facts(_synthetic_episode(heights, contact))
    assert facts["lift_success"] and facts["lift_windows"] == 1 and facts["picked"]
    assert facts["place_windows_anywhere"] == 1  # at rest, free, after the pick
    assert not facts["place_success"]  # not inside the target square
    assert facts["attempts"][0]["grasped"] is True
    transient = pc.episode_facts(
        _synthetic_episode(
            [0.0] * 5 + [0.06] * 5 + [0.0] * 30, [False] * 5 + [True] * 5 + [False] * 30
        )
    )
    assert transient["lift_windows"] == 0 and not transient["lift_success"]
    assert transient["attempts"][0]["grasped"] is False


# ----- physics (headless) ----------------------------------------------------------------------
@pytest.fixture
def robot():
    pytest.importorskip("mujoco")
    if not FETCHED:
        pytest.skip("pinned SO-101 assets have not been fetched")
    robot = pc.make_play_robot(image_size=8, render=False)
    robot.sim.render = lambda camera="wrist_rgb": np.zeros((8, 8, 3), np.uint8)
    yield robot
    robot.close()


def test_episode_is_deterministic_and_stores_the_executed_actions(robot):
    a = pc.run_episode(robot, 990000, commands=120)
    b = pc.run_episode(robot, 990000, commands=120)
    assert np.array_equal(a["actions"], b["actions"])
    assert np.array_equal(a["cube_position"], b["cube_position"])
    n = len(a["actions"])
    assert a["states"].shape == (n + 1, 12) and a["frames"]["wrist_rgb"].shape[0] == n + 1
    assert np.allclose(np.diff(a["timestamps"]), 1 / pc.FPS)
    assert a["actions"].dtype == np.float32 and np.all(np.abs(a["actions"]) <= 1)
    same = ~a["clipped"]
    assert np.array_equal(a["actions"][same], a["requested"][same])
    for k in range(n):
        assert np.array_equal(a["physical"][k], robot.denormalize_action(a["actions"][k]))
    # The stored arm deltas are exactly the joint targets' increments.
    increments = np.diff(a["targets"][:, :5], axis=0)
    assert np.allclose(increments, a["actions"][:, :5] * robot.scales, atol=1e-6)
    assert set(pc.PHASES[i] for i in a["phase"]) <= set(pc.PHASES)


def test_tiny_corpus_collects_seals_and_measures(tmp_path):
    pytest.importorskip("mujoco")
    pytest.importorskip("pyarrow")
    if not FETCHED:
        pytest.skip("pinned SO-101 assets have not been fetched")
    collect, measure = _script("collect_so101_play"), _script("measure_so101_play")
    out = tmp_path / "corpus"
    args = ["--kind", "debug", "--out", str(out), "--workers", "1", "--n-shards", "2"]
    args += ["--shard-size", "4", "--commands", "60", "--first-seed", "7", "--image-size", "16"]
    assert collect.main(args) == 0
    summary = json.loads((out / "corpus.json").read_text())
    assert [s["status"] for s in summary["shards"]] == ["complete", "complete"]
    assert summary["code_sha256"]["src/embodied_jepa/so101_play_corpus.py"]
    from embodied_jepa.schema_data import SchemaDatasetStore, open_dataset

    store = open_dataset(out / "shard-00")
    assert isinstance(store, SchemaDatasetStore)
    assert store.action_schema.version == "joint_delta_so101_v0"
    row = store.manifest["episodes"][0]
    episode = store.read_episode(row["episode_id"])
    side = pc.load_sidecar(out / "shard-00", row)
    assert episode.actions.shape[1] == 6 and set(episode.observations) == set(pc.CAMERAS)
    assert episode.observations["wrist_rgb"].shape[1:] == (16, 16, 3)
    assert len(side["requested"]) == len(episode.actions)
    # Never overwrites: a second run skips complete shards.
    assert collect.main(args) == 0
    report_dir = tmp_path / "measure"
    assert measure.main(["--corpus", str(out), "--out", str(report_dir), "--workers", "1"]) == 0
    report = json.loads((report_dir / "report.json").read_text())
    assert report["problems"] == []
    for key in ("episode_id", "session_id", "seed", "cube_xy"):
        assert sum(report["leakage"][key].values()) == 0
    assert report["storage"]["episodes"] == 8
    assert sum(w["episodes"] for w in report["windows"].values()) == 8
    assert (report_dir / "contact_sheet.png").is_file()
    with pytest.raises(SystemExit):
        measure.main(["--corpus", str(out), "--out", str(report_dir)])
