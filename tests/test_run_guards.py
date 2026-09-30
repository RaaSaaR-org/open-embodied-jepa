"""The #108 review follow-ups, fixed in ``run_guards`` for TASK-074 (TASK-073's pinned code is
unchanged)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import run_guards as rg

ROOT = Path(__file__).resolve().parents[1]
EXACT = ("success", "grasp", "at_rest", "termination_reason", "executed_steps", "chosen_sequence")
TOL = {"final_distance_cm": 1.0, "first_grasp_step": 40, "first_place_step": 40}


def _record(**over):
    base = {
        "success": True,
        "grasp": True,
        "at_rest": True,
        "termination_reason": "step_limit",
        "executed_steps": 740,
        "final_distance_cm": 3.1,
        "first_grasp_step": 265,
        "first_place_step": 613,
        "commands": np.zeros((740, 14), np.float32),
        "decisions": [{"chosen": 12}, {"chosen": 3}],
    }
    return base | over


def _cmp(a, b, image_reading):
    return rg.rerun_matches(a, b, image_reading=image_reading, exact=EXACT, tolerances=TOL)


# ----- rerun_matches ------------------------------------------------------------------------------
def test_identical_records_match_with_zero_command_difference():
    for image in (False, True):
        out = _cmp(_record(), _record(), image)
        assert out["matches"] and out["max_abs_command_difference"] == 0.0
        assert out["missing"] == [] and out["differences"] == {}


def test_a_field_missing_from_both_records_fails():
    a, b = _record(), _record()
    del a["first_place_step"], b["first_place_step"]
    for image in (False, True):
        out = _cmp(a, b, image)
        assert not out["matches"]
        assert out["missing"] == ["first_place_step"]


def test_missing_commands_fail_and_one_sided_fields_differ():
    a, b = _record(), _record()
    del a["commands"], b["commands"]
    out = _cmp(a, b, True)
    assert not out["matches"] and "commands" in out["missing"]
    assert out["max_abs_command_difference"] is None
    c = _record()
    del c["commands"]
    assert not _cmp(_record(), c, True)["matches"]
    d = _record()
    del d["grasp"]
    out = _cmp(_record(), d, True)
    assert not out["matches"] and "grasp" in out["differences"]


def test_command_difference_is_computed_and_fails_a_non_image_arm():
    b_cmd = np.zeros((740, 14), np.float32)
    b_cmd[100, 3] = 0.25
    out = _cmp(_record(), _record(commands=b_cmd), False)
    assert not out["matches"]
    assert out["max_abs_command_difference"] == pytest.approx(0.25)
    # an image-reading arm only logs it (no tolerance given) ...
    out = _cmp(_record(), _record(commands=b_cmd), True)
    assert out["matches"] and out["differences"]["max_abs_command_difference"] == 0.25
    # ... unless a command tolerance is declared
    tight = rg.rerun_matches(
        _record(),
        _record(commands=b_cmd),
        image_reading=True,
        exact=EXACT,
        tolerances=TOL | {"max_abs_command_difference": 0.1},
    )
    assert not tight["matches"]


def test_different_command_lengths_are_an_infinite_difference():
    out = _cmp(_record(), _record(commands=np.zeros((700, 14))), True)
    assert out["max_abs_command_difference"] == float("inf")


def test_image_arm_tolerances_and_exact_fields():
    assert _cmp(_record(), _record(final_distance_cm=3.9), True)["matches"]
    assert not _cmp(_record(), _record(final_distance_cm=4.2), True)["matches"]
    assert not _cmp(_record(), _record(final_distance_cm=None), True)["matches"]
    assert not _cmp(_record(), _record(success=False), True)["matches"]
    assert not _cmp(_record(), _record(decisions=[{"chosen": 12}, {"chosen": 4}]), True)["matches"]
    assert not _cmp(_record(), _record(final_distance_cm=3.2), False)["matches"]


# ----- pss_bytes: the RSS fallback -------------------------------------------------------------
def _fake_proc(tmp_path, pid, *, rss_kb, pss_kb=None, children=()):
    d = tmp_path / str(pid)
    (d / "task" / str(pid)).mkdir(parents=True)
    (d / "status").write_text(f"Name:\tx\nVmRSS:\t{rss_kb} kB\n")
    (d / "task" / str(pid) / "children").write_text(" ".join(str(c) for c in children))
    if pss_kb is not None:
        (d / "smaps_rollup").write_text(f"Rss:\t{rss_kb} kB\nPss:\t{pss_kb} kB\n")


def test_pss_is_read_when_available_and_rss_is_the_fallback(tmp_path):
    _fake_proc(tmp_path, 10, rss_kb=400, pss_kb=250, children=(11,))
    _fake_proc(tmp_path, 11, rss_kb=300)  # no smaps_rollup: unreadable PSS
    assert rg.pss_bytes(10, tmp_path) == (250 * 1024, "pss")
    assert rg.pss_bytes(11, tmp_path) == (300 * 1024, "rss")
    assert rg.pss_bytes(12, tmp_path) == (0, "gone")
    per: dict = {}
    total = rg.process_tree_memory(10, per, proc=tmp_path)
    assert total == {
        "rss": 700 * 1024,
        "pss": (250 + 300) * 1024,
        "pss_fallback_processes": 1,
    }
    assert per == {10: (400 * 1024, 250 * 1024), 11: (300 * 1024, 300 * 1024)}


def test_pss_without_a_pss_line_falls_back(tmp_path):
    _fake_proc(tmp_path, 20, rss_kb=128)
    (tmp_path / "20" / "smaps_rollup").write_text("Rss:\t128 kB\n")
    assert rg.pss_bytes(20, tmp_path) == (128 * 1024, "rss")


def test_real_proc_reads_this_process():
    """On Linux the real /proc is read; elsewhere there is no /proc and nothing is measured."""
    bytes_, source = rg.pss_bytes(os.getpid())
    if sys.platform.startswith("linux"):
        assert bytes_ > 0 and source in ("pss", "rss")
        assert rg.process_tree_memory()["rss"] > 0
    else:
        assert (bytes_, source) == (0, "gone")


# ----- BoundedPool ------------------------------------------------------------------------------
def test_pool_maps_and_closes_quickly():
    pool = rg.BoundedPool(2, rg.probe_task, initializer=rg.probe_init, join_seconds=5.0)
    try:
        out = pool.map([{"seed": i} for i in range(4)], 120.0, "probe")
        assert [o["seed"] for o in out] == [0, 1, 2, 3]
    finally:
        record = pool.close()
    assert record["terminate_returned"] is True and record["unkillable"] == []
    assert pool.close() is record  # idempotent


def test_close_is_bounded_with_workers_that_ignore_sigterm_mid_map(tmp_path):
    """Busy workers that ignore SIGTERM are SIGKILLed; close returns within its bound."""
    pool = rg.BoundedPool(
        2,
        rg.probe_task,
        initializer=rg.probe_init,
        initargs=(True,),
        join_seconds=1.0,
        terminate_seconds=5.0,
    )
    pids = set(pool.pids)
    tasks = [{"seconds": 120, "seed": i, "marker_dir": str(tmp_path)} for i in range(2)]
    pool.pool.map_async(rg.probe_task, tasks, chunksize=1)
    deadline = time.monotonic() + 60
    while len(list(tmp_path.glob("started-*"))) < 2:  # both workers inside the 120 s task
        assert time.monotonic() < deadline
        time.sleep(0.1)
    started = time.monotonic()
    record = pool.close()
    elapsed = time.monotonic() - started
    assert elapsed < 20.0, record
    assert set(record["killed"]) >= pids, (record, pids)
    assert record["unkillable"] == []
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_a_dead_worker_raises_at_once():
    import signal
    import threading

    pool = rg.BoundedPool(2, rg.probe_task, initializer=rg.probe_init, join_seconds=2.0)
    try:
        victim = next(iter(pool.pids))
        threading.Timer(1.5, lambda: os.kill(victim, signal.SIGKILL)).start()
        started = time.monotonic()
        with pytest.raises(rg.WorkerDied, match="G-worker"):
            pool.map([{"seconds": 30, "seed": i} for i in range(4)], 600.0, "probe")
        assert time.monotonic() - started < 20
    finally:
        pool.close()


def test_the_map_cap_raises():
    pool = rg.BoundedPool(1, rg.probe_task, initializer=rg.probe_init, join_seconds=2.0)
    try:
        with pytest.raises(rg.MapCapExceeded, match="G-cap"):
            pool.map([{"seconds": 30, "seed": 0}], 1.5, "probe")
    finally:
        pool.close()


# ----- TASK-073's record is untouched ------------------------------------------------------------
def test_task073_pinned_files_are_unchanged_and_run_guards_is_not_pinned():
    manifest = json.loads(
        (ROOT / "benchmarks" / "manifests" / "apple-wm-critic-v2.json").read_text()
    )
    for relative in ("src/embodied_jepa/wm_critic_v2.py", "scripts/run_wm_critic_v2.py"):
        got = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert manifest["hashes"][relative] == got, relative
    assert "src/embodied_jepa/run_guards.py" not in manifest["hashes"]


def test_run_guards_imports_no_heavy_runtime():
    import subprocess

    code = (
        "import sys, embodied_jepa.run_guards; "
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
