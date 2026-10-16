"""SO-101 S2 (TASK-098): cube lift and place tasks, their scorers and the scripted followers.

The scorer tests use synthetic per-step truth: a transient crossing, a dropped cube, a cube
still in the gripper, a cube that never came to rest and a pushed (never picked) cube must not
count. Physics tests skip only when the pinned SO-101 assets are not fetched.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from embodied_jepa import config as config_module
from embodied_jepa.config import ExperimentConfig
from embodied_jepa.registry import TASKS
from embodied_jepa.so101 import DEFAULT_ACTION_MANIFEST
from embodied_jepa.so101_tasks import (
    LIFT_V0,
    PLACE_V0,
    RESETS_V0,
    REST_Z,
    LiftThresholds,
    PlaceThresholds,
    ResetDistribution,
    SO101CubeLiftTask,
    lift_outcome,
    place_outcome,
    sample_reset,
)

ROOT = Path(__file__).resolve().parents[1]
ASSET_MANIFEST = json.loads((ROOT / "assets/so101_manifest.json").read_text())
FETCHED = all(
    (ROOT / ASSET_MANIFEST["local_path"] / relative).is_file()
    for relative in ASSET_MANIFEST["sha256"]
)
TARGET = np.array([0.20, 0.05])


# ----- synthetic trajectories ---------------------------------------------------------------
def _lift_track(heights_above_rest, contact=True):
    z = REST_Z + np.asarray(heights_above_rest, dtype=float)
    position = np.column_stack((np.full(len(z), 0.2), np.zeros(len(z)), z))
    contacts = np.full(len(z), contact) if isinstance(contact, bool) else np.asarray(contact)
    return position, contacts


def test_lift_needs_the_declared_height_for_the_whole_window():
    heights = [0.0] * 10 + [0.06] * 25 + [0.06] * 5
    position, contact = _lift_track(heights)
    out = lift_outcome(position, contact)
    assert out["success"] and out["success_step"] == 10 + LIFT_V0.hold_steps - 1
    # Exactly at the bar counts; just below never does.
    position, contact = _lift_track([LIFT_V0.lift_height_m] * 20)
    assert lift_outcome(position, contact)["success"]
    position, contact = _lift_track([LIFT_V0.lift_height_m - 1e-4] * 40)
    assert not lift_outcome(position, contact)["success"]


def test_lift_transient_crossing_and_drop_do_not_count():
    # A flick above the bar for a few steps, then back on the table.
    position, contact = _lift_track([0.0] * 5 + [0.07] * 4 + [0.0] * 40)
    out = lift_outcome(position, contact)
    assert not out["success"] and out["longest_hold_steps"] == 4
    # Held for most of the window, then dropped: the window never completes.
    position, contact = _lift_track(
        [0.0] * 5 + [0.06] * (LIFT_V0.hold_steps - 1) + [0.0] * 30,
        contact=[True] * (5 + LIFT_V0.hold_steps - 1) + [False] * 30,
    )
    assert not lift_outcome(position, contact)["success"]
    # Two holds that together exceed the window but are separated by a drop do not add up.
    position, contact = _lift_track([0.06] * 12 + [0.0] + [0.06] * 12)
    assert not lift_outcome(position, contact)["success"]


def test_lift_requires_a_gripper_contact_and_a_cube_on_the_table():
    # High without a gripper contact (e.g. thrown) does not count.
    position, _ = _lift_track([0.06] * 30)
    assert not lift_outcome(position, np.zeros(30, bool))["success"]
    loose = replace(LIFT_V0, require_gripper_contact=False)
    assert lift_outcome(position, np.zeros(30, bool), loose)["success"]
    # A cube that left the table first is lost, even if a later window would complete.
    position, contact = _lift_track([0.06] * 30)
    position[2, 0] = 0.9
    out = lift_outcome(position, contact)
    assert not out["success"] and out["lost_step"] == 2


def _place_track(xy, z, speed=0.0, spin=0.0, contact=False):
    xy = np.asarray(xy, dtype=float)
    steps = len(xy)
    position = np.column_stack((xy, np.broadcast_to(np.asarray(z, float), (steps,))))
    velocity = np.zeros((steps, 3))
    velocity[:, 0] = speed
    angular = np.zeros((steps, 3))
    angular[:, 2] = spin
    contacts = np.broadcast_to(np.asarray(contact, bool), (steps,)).copy()
    return position, velocity, angular, contacts


def _pick_then(xy_after, *, z_after=REST_Z, speed=0.0, spin=0.0, contact_after=False):
    """Six steps carried 5 cm up with a contact, then the given per-step xy."""
    carry = _place_track(np.tile(TARGET, (6, 1)), REST_Z + 0.05, contact=True)
    return _concat(carry, _place_track(xy_after, z_after, speed, spin, contact_after))


def _concat(*tracks):
    return tuple(np.concatenate(parts) for parts in zip(*tracks, strict=True))


def test_place_at_rest_inside_after_a_pick_counts_after_the_window():
    track = _pick_then(np.tile(TARGET + [0.03, -0.03], (25, 1)))
    out = place_outcome(*track, TARGET)
    assert out["picked"] and out["pick_step"] == PLACE_V0.pick_steps - 1
    assert out["success"] and out["success_step"] == 6 + PLACE_V0.rest_steps - 1
    # Just inside a corner of the square counts; 1 mm beyond an edge does not.
    edge = PLACE_V0.target_half_size_m
    corner = np.tile(TARGET + 0.999 * edge, (25, 1))
    assert place_outcome(*_pick_then(corner), TARGET)["success"]
    outside = _pick_then(np.tile(TARGET + [edge + 0.001, 0.0], (25, 1)))
    assert not place_outcome(*outside, TARGET)["success"]


def test_place_transient_crossing_does_not_count():
    # The cube slides through the square at 0.2 m/s and stops outside it.
    path = TARGET + np.column_stack((np.linspace(-0.08, 0.08, 9), np.zeros(9)))
    moving = _pick_then(path, speed=0.2)
    stopped = _place_track(np.tile(TARGET + [0.09, 0.0], (30, 1)), REST_Z)
    track = _concat(moving, stopped)
    out = place_outcome(*track, TARGET)
    assert not out["success"] and not out["final_inside"]
    # Briefly at rest inside (shorter than the window), then knocked out.
    inside = np.tile(TARGET, (PLACE_V0.rest_steps - 1, 1))
    outside = np.tile(TARGET + [0.1, 0.0], (30, 1))
    out = place_outcome(*_pick_then(np.concatenate((inside, outside))), TARGET)
    assert not out["success"] and out["longest_rest_steps"] == PLACE_V0.rest_steps - 1


def test_place_dropped_or_pushed_cube_does_not_count():
    # Dropped on the way: it ends at rest, but outside the square.
    out = place_outcome(*_pick_then(np.tile(TARGET + [0.0, 0.12], (40, 1))), TARGET)
    assert out["picked"] and not out["success"]
    # Pushed along the table into the square without a pick.
    track = _place_track(np.tile(TARGET, (40, 1)), REST_Z, contact=False)
    out = place_outcome(*track, TARGET)
    assert not out["picked"] and not out["success"]
    # A lift too short for the pick window does not make a pick.
    short = _place_track(np.tile(TARGET, (PLACE_V0.pick_steps - 1, 1)), REST_Z + 0.05, contact=True)
    rest = _place_track(np.tile(TARGET, (40, 1)), REST_Z)
    track = _concat(short, rest)
    assert not place_outcome(*track, TARGET)["picked"]


def test_place_requires_release_rest_and_no_slow_drift():
    xy = np.tile(TARGET, (30, 1))
    # Still held (gripper contact) inside the square.
    assert not place_outcome(*_pick_then(xy, contact_after=True), TARGET)["success"]
    # Held just above the table, not resting on it.
    assert not place_outcome(*_pick_then(xy, z_after=REST_Z + 0.01), TARGET)["success"]
    # Spinning or sliding faster than the bars.
    assert not place_outcome(*_pick_then(xy, speed=0.01), TARGET)["success"]
    assert not place_outcome(*_pick_then(xy, spin=0.5), TARGET)["success"]
    # Creeping 0.2 mm per step (below the speed bar as reported) drifts 4 mm over the window.
    creep = TARGET + np.column_stack((np.arange(30) * 2e-4, np.zeros(30)))
    out = place_outcome(*_pick_then(creep, speed=0.004), TARGET)
    assert not out["success"] and out["longest_rest_steps"] >= PLACE_V0.rest_steps


def test_place_lost_cube_fails_and_inputs_are_validated():
    track = _pick_then(np.tile(TARGET, (30, 1)))
    track[0][7, 2] = -0.2
    out = place_outcome(*track, TARGET)
    assert not out["success"] and out["lost_step"] == 7
    with pytest.raises(ValueError):
        place_outcome(*track, [0.2])
    bad = list(_pick_then(np.tile(TARGET, (30, 1))))
    bad[1] = bad[1][:-1]
    with pytest.raises(ValueError):
        place_outcome(*bad, TARGET)
    with pytest.raises(ValueError):
        lift_outcome(np.zeros((3, 2)), np.zeros(3, bool))


def test_thresholds_and_reset_distribution_are_validated():
    with pytest.raises(ValueError):
        LiftThresholds(lift_height_m=0.0)
    with pytest.raises(ValueError):
        LiftThresholds(hold_steps=300)
    with pytest.raises(ValueError):
        PlaceThresholds(rest_steps=0)
    with pytest.raises(ValueError):
        PlaceThresholds(lost_below_z_m=0.1)
    with pytest.raises(ValueError):
        ResetDistribution(cube_x_m=(0.3, 0.2))


def test_resets_are_declared_seeded_and_separate_per_task():
    d = RESETS_V0
    for seed in range(200):
        lift = sample_reset("so101_cube_lift_v0", seed)
        place = sample_reset("so101_cube_place_v0", seed)
        assert lift == sample_reset("so101_cube_lift_v0", seed)
        for reset in (lift, place):
            x, y = reset["cube_xy"]
            assert d.cube_x_m[0] <= x <= d.cube_x_m[1] and d.cube_y_m[0] <= y <= d.cube_y_m[1]
            assert d.cube_yaw_rad[0] <= reset["cube_yaw"] <= d.cube_yaw_rad[1]
        tx, ty = place["target_xy"]
        assert d.target_x_m[0] <= tx <= d.target_x_m[1]
        assert d.target_y_m[0] <= ty <= d.target_y_m[1]
        gap = np.subtract(place["target_xy"], place["cube_xy"])
        assert np.linalg.norm(gap) >= d.min_cube_target_m
        # The starting cube lies wholly outside the target square.
        assert np.abs(gap).max() > PLACE_V0.target_half_size_m + 0.015
        assert lift["cube_xy"] != place["cube_xy"]
    with pytest.raises(ValueError):
        sample_reset("reach", 0)
    with pytest.raises(ValueError):
        sample_reset("so101_cube_lift_v0", -1)


# ----- registration and config ---------------------------------------------------------------
def test_tasks_are_registered_lazily_and_core_import_stays_clean():
    assert TASKS._factories["so101_cube_lift_v0"] == "embodied_jepa.so101_tasks:SO101CubeLiftTask"
    assert TASKS._factories["so101_cube_place_v0"] == "embodied_jepa.so101_tasks:SO101CubePlaceTask"
    assert config_module.EMBODIMENT_TASKS["so101"] == (
        "so101_cube_lift_v0",
        "so101_cube_place_v0",
    )
    code = (
        "import sys, embodied_jepa, embodied_jepa.config, embodied_jepa.so101_tasks, "
        "embodied_jepa.so101_followers;"
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def _config(tmp_path, embodiment, schema, task):
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "native.pt").touch()
    if embodiment == "so101":
        shutil.copy(DEFAULT_ACTION_MANIFEST, tmp_path / "action.json")
    else:
        shutil.copy(ROOT / "configs/g1_sim_action.json", tmp_path / "action.json")
    raw = {
        "schema_version": 0,
        "world_model": {"backend": "native_jepa", "checkpoints": {"native_jepa": "native.pt"}},
        "embodiment": {
            "backend": embodiment,
            "transport": "mujoco",
            "action_schema": schema,
            "action_manifest": "action.json",
        },
        "planner": {"horizon": 3, "samples": 16, "elites": 4},
        "dataset": {"root": "data"},
        "task": {"name": task},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_configs_route_so101_tasks_only_to_the_so101_and_refuse_so101_configs(tmp_path):
    # An SO-101 task in an SO-101 config passes the task checks but is refused at load: the
    # benchmark runner is G1-only and there is no SO-101 entry point yet.
    for task in ("so101_cube_lift_v0", "so101_cube_place_v0"):
        with pytest.raises(ValueError, match="no SO-101 experiment entry point"):
            ExperimentConfig.load(_config(tmp_path, "so101", "joint_delta_so101_v0", task))
    with pytest.raises(ValueError, match="has no task 'apple_to_plate'"):
        ExperimentConfig.load(_config(tmp_path, "so101", "joint_delta_so101_v0", "apple_to_plate"))
    with pytest.raises(ValueError, match="needs embodiment 'so101'"):
        ExperimentConfig.load(
            _config(tmp_path, "unitree_g1_dex3", "ee_delta_grasp_v0", "so101_cube_lift_v0")
        )


def test_task_refuses_a_non_so101_embodiment():
    class G1Like:
        action_schema = object()

    with pytest.raises(ValueError, match="needs the SO-101 embodiment"):
        SO101CubeLiftTask(G1Like())


# ----- physics (headless) ----------------------------------------------------------------------
@pytest.fixture
def robot():
    pytest.importorskip("mujoco")
    if not FETCHED:
        pytest.skip("pinned SO-101 assets have not been fetched")
    from embodied_jepa.so101 import SO101Embodiment
    from embodied_jepa.so101_simulation import SO101Simulation

    sim = SO101Simulation(render=False)
    # The followers and scorers read no image; only RGB is replaced.
    sim.render = lambda camera="wrist_rgb": np.zeros((8, 8, 3), np.uint8)
    embodiment = SO101Embodiment(sim)
    yield embodiment
    embodiment.close()


def test_reset_applies_the_declared_reset_and_hold_times_out(robot):
    task = TASKS.create("so101_cube_lift_v0", embodiment=robot)
    goal = task.reset(3)
    assert goal["lift_height_m"] == LIFT_V0.lift_height_m
    truth = robot.sim.task_truth()
    np.testing.assert_allclose(truth["cube_position"][:2], task.reset_spec["cube_xy"])
    assert {"cube_angular_velocity", "gripper_contact"} <= set(truth)
    while not task.done:
        robot.observe()
        robot.execute(robot.hold_action())
        status = task.evaluate()
    assert status["termination_reason"] == "timeout" and status["steps"] == LIFT_V0.max_steps
    with pytest.raises(ValueError, match="once per control step"):
        task.evaluate()


def test_place_goal_names_the_target_and_the_marker_moves_there(robot):
    task = TASKS.create("so101_cube_place_v0", embodiment=robot)
    goal = task.reset(5)
    assert goal["target_xy"] == task.reset_spec["target_xy"]
    np.testing.assert_allclose(robot.sim.task_truth()["target_position"][:2], goal["target_xy"])


def test_scripted_followers_complete_both_tasks_on_a_development_seed(robot):
    """Development check only (privileged scripted follower; not a learned result)."""
    from embodied_jepa.so101_followers import follow_lift, follow_place
    from embodied_jepa.so101_tasks import SO101CubePlaceTask

    lift = follow_lift(robot, SO101CubeLiftTask(robot), 1000)
    assert lift["success"] and lift["termination_reason"] == "success"
    place = follow_place(robot, SO101CubePlaceTask(robot), 1000)
    assert place["success"] and place["score"]["picked"]
    # A place aim 6 cm off the target leaves the cube outside the 4 cm half-square.
    off = follow_place(robot, SO101CubePlaceTask(robot), 1000, place_offset_m=(0.0, 0.06))
    assert not off["success"] and off["score"]["picked"]


def test_repeated_scene_constants_match_the_simulation_scene():
    """so101_tasks.py repeats the scene's constants; they must not drift from the scene."""
    import xml.etree.ElementTree as ET

    from embodied_jepa import so101_simulation as scene
    from embodied_jepa import so101_tasks as tasks

    assert tasks.CUBE_HALF_SIZE_M == scene.CUBE_HALF_SIZE_M
    assert tasks.TABLE_TOP_Z == scene.TABLE_TOP_Z
    assert PLACE_V0.target_half_size_m == scene.TARGET_HALF_SIZE_M
    pytest.importorskip("mujoco")
    if not FETCHED:
        pytest.skip("pinned SO-101 assets have not been fetched")
    sim = scene.SO101Simulation(render=False)
    try:
        root = ET.fromstring(sim._scene())
    finally:
        sim.close()
    table = root.find(".//geom[@name='table']")
    cube = root.find(".//geom[@name='cube_geom']")
    (cx, cy, cz), (sx, sy, sz) = ([float(v) for v in table.get(k).split()] for k in ("pos", "size"))
    assert np.isclose(cz + sz, tasks.TABLE_TOP_Z)
    np.testing.assert_allclose(tasks.TABLE_XY, ((cx - sx, cx + sx), (cy - sy, cy + sy)))
    assert cube is not None
    assert np.allclose([float(v) for v in cube.get("size").split()], tasks.CUBE_HALF_SIZE_M)
