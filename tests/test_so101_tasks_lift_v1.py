"""TASK-101 amendment A1 (§18.2): ``so101_cube_lift_v1`` is ``so101_cube_lift_v0`` with the inert
target marker drawn per reset from the play corpus's marker distribution (salt 10107)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import so101_play_corpus as play
from embodied_jepa import so101_tasks as tasks
from embodied_jepa import so101_tasks_lift_v1 as v1

ROOT = Path(__file__).resolve().parents[1]


def test_v1_cube_draws_equal_v0_for_1000_seeds():
    for seed in range(101000, 102000):
        a, b = tasks.sample_reset(tasks.LIFT_TASK, seed), v1.sample_reset_lift_v1(seed)
        assert a["cube_xy"] == b["cube_xy"] and a["cube_yaw"] == b["cube_yaw"]
        assert a["target_xy"] == [0.30, -0.20] and b["target_xy"] != a["target_xy"]


def test_every_marker_lies_in_the_corpus_box_and_8_cm_from_its_cube():
    assert v1.MARKER_REGION is play.TARGET_REGION
    assert v1.MIN_CUBE_MARKER_M == play.MIN_CUBE_TARGET_M == 0.08
    (x0, x1), (y0, y1) = play.TARGET_REGION
    for seed in range(101000, 102000):
        spec = v1.sample_reset_lift_v1(seed)
        m, c = np.asarray(spec["target_xy"]), np.asarray(spec["cube_xy"])
        assert x0 <= m[0] <= x1 and y0 <= m[1] <= y1
        assert np.linalg.norm(m - c) >= 0.08


def test_the_marker_stream_is_fixed_by_its_salt_and_independent_of_v0s():
    assert v1.MARKER_SALT == 10107
    seed = 101600
    cube = tasks.sample_reset(tasks.LIFT_TASK, seed)["cube_xy"]
    rng = np.random.default_rng([10107, seed])
    first = None
    while first is None:
        m = np.array([rng.uniform(0.16, 0.22), rng.uniform(-0.14, 0.14)])
        if np.linalg.norm(m - np.asarray(cube)) >= 0.08:
            first = m.tolist()
    assert v1.sample_marker(seed, cube) == first
    assert v1.sample_marker(seed, cube) == v1.sample_reset_lift_v1(seed)["target_xy"]
    # not drawn from v0's stream after the cube
    rng0 = np.random.default_rng([9801, seed])
    rng0.uniform(size=3)
    assert [rng0.uniform(0.16, 0.22), rng0.uniform(-0.14, 0.14)] != first


def test_a_marker_that_cannot_be_placed_is_an_error(monkeypatch):
    monkeypatch.setattr(v1, "MIN_CUBE_MARKER_M", 1.0)
    with pytest.raises(ValueError, match="marker"):
        v1.sample_marker(1, [0.19, 0.0])


def test_the_task_class_overrides_only_name_and_reset():
    assert v1.SO101CubeLiftTaskV1.name == "so101_cube_lift_v1" == v1.LIFT_V1_TASK
    assert issubclass(v1.SO101CubeLiftTaskV1, tasks.SO101CubeLiftTask)
    own = set(vars(v1.SO101CubeLiftTaskV1)) - {"__module__", "__doc__", "__qualname__"}
    assert own <= {"name", "sample_reset", "__firstlineno__", "__static_attributes__"}


def test_the_module_imports_no_torch_or_mujoco():
    code = (
        "import sys, embodied_jepa.so101_tasks_lift_v1;"
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_pinned_task_simulation_and_config_files_keep_their_bytes():
    pins = json.loads((ROOT / "benchmarks/manifests/so101-play-v1.json").read_text())
    for rel in ("src/embodied_jepa/so101_tasks.py", "src/embodied_jepa/so101_simulation.py"):
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert got == pins["code_sha256"][rel], rel
    got = hashlib.sha256((ROOT / "src/embodied_jepa/config.py").read_bytes()).hexdigest()
    assert got == "0a6544a0230ecb7a156da03efbafb0759a494bddb85f82dbd9f6d0427bb69e57"
    assert "so101_cube_lift_v1" not in (ROOT / "src/embodied_jepa/config.py").read_text()


def test_a_v1_reset_puts_the_marker_at_the_drawn_point():
    pytest.importorskip("mujoco")
    manifest = json.loads((ROOT / "assets/so101_manifest.json").read_text())
    if not all((ROOT / manifest["local_path"] / r).is_file() for r in manifest["sha256"]):
        pytest.skip("pinned SO-101 assets have not been fetched")
    from embodied_jepa import so101_closed_loop_runtime as rt

    robot = rt.make_robot(False)
    try:
        task = rt.lift_task(robot)
        assert task.name == "so101_cube_lift_v1"
        task.reset(101900)
        truth = robot.sim.task_truth()
        spec = v1.sample_reset_lift_v1(101900)
        assert np.array_equal(truth["target_position"], [*spec["target_xy"], tasks.TABLE_TOP_Z])
    finally:
        robot.close()
