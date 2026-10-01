"""Host tests for ``run_tools`` (the 2026-10-02 audit's shared harness pieces, PR-C)."""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

import pytest

from embodied_jepa import run_tools as rt

ROOT = Path(__file__).resolve().parents[1]
GIB = rt.GIB


# ----- F1d: assert_local_import ------------------------------------------------------------------
def test_assert_local_import_accepts_this_checkout_and_records_the_file():
    report: dict = {}
    package = rt.assert_local_import(ROOT, report)
    assert package == (ROOT / "src" / "embodied_jepa").resolve()
    assert report["embodied_jepa_file"].endswith("src/embodied_jepa/__init__.py")
    assert report["import_root"] == str(ROOT.resolve())


def test_assert_local_import_refuses_another_checkout(tmp_path):
    (tmp_path / "src" / "embodied_jepa").mkdir(parents=True)
    (tmp_path / "src" / "embodied_jepa" / "__init__.py").write_text("")
    with pytest.raises(rt.GuardError, match="G-import"):
        rt.assert_local_import(tmp_path)


def test_assert_local_import_finds_the_root_from_the_running_script(monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(ROOT / "scripts" / "run_some_new_stage.py")])
    assert rt.assert_local_import() == (ROOT / "src" / "embodied_jepa").resolve()


# ----- F11: the first signal is recorded before the raise ----------------------------------------
@pytest.fixture
def guards():
    g = rt.install_guards()
    yield g
    g.uninstall()


def _deliver(signum):
    os.kill(os.getpid(), signum)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:  # the handler runs between bytecodes of this loop
        time.sleep(0.01)
    raise AssertionError("the signal was not delivered")


def test_a_signal_ends_the_stage_as_v_with_its_reason(guards):
    report: dict = {}
    try:
        _deliver(signal.SIGTERM)
    except BaseException as error:  # noqa: BLE001 - the runner's own except clause
        guards.void(report, error)
    assert report["outcome"] == "V"
    assert report["void_reason"] == "StageInterrupted: received SIGTERM (SIGTERM)"
    assert report["first_signal"]["signal"] == "SIGTERM"
    assert "void_surfaced_as" not in report


def test_a_signal_that_surfaces_as_another_exception_is_still_reported(guards):
    """TASK-073 main run 36793229744: the raise landed in a lock release and arrived as
    ``RuntimeError: release unlocked lock``; the void reason must still be the signal."""
    report: dict = {}
    try:
        try:
            _deliver(signal.SIGTERM)
        except BaseException:  # noqa: BLE001 - library code that swallows and re-raises
            raise RuntimeError("release unlocked lock") from None
    except BaseException as error:  # noqa: BLE001
        guards.void(report, error)
    assert report["void_reason"] == "StageInterrupted: received SIGTERM (SIGTERM)"
    assert report["void_surfaced_as"] == "RuntimeError: release unlocked lock"
    assert report["interrupted_utc"] == report["first_signal"]["utc"]


def test_signals_after_the_first_are_recorded_not_raised(guards):
    report: dict = {}
    try:
        _deliver(signal.SIGINT)
    except BaseException as error:  # noqa: BLE001
        guards.void(report, error)
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(0.05)
    guards.finish(report)
    assert report["first_signal"]["signal"] == "SIGINT"
    assert [s["signal"] for s in report["signals_during_cleanup"]] == ["SIGTERM"]


def test_finish_voids_a_stage_whose_except_clause_was_interrupted(guards):
    report: dict = {}
    try:
        try:
            _deliver(signal.SIGHUP)
        finally:
            guards.finish(report)
    except rt.StageInterrupted:
        pass
    assert report["outcome"] == "V" and "SIGHUP" in report["void_reason"]


def test_an_error_without_a_signal_is_reported_as_itself(guards):
    report: dict = {}
    try:
        raise ValueError("bad shard")
    except ValueError as error:
        guards.void(report, error)
    assert report["void_reason"] == "ValueError: bad shard" and "first_signal" not in report


# ----- the memory watch -------------------------------------------------------------------------
class _Sampler:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self, per=None):
        rss, pss = self.values.pop(0)
        if per is not None:
            per[os.getpid()] = (rss, pss)
        return {"rss": rss, "pss": pss, "pss_fallback_processes": 0}


def test_the_memory_watch_trips_once_and_its_reason_is_the_void_reason():
    watch = rt.MemoryWatch(2 * GIB, 60.0, sampler=_Sampler([(GIB, GIB), (4 * GIB, 3 * GIB)]))
    g = rt.install_guards(watch)
    report: dict = {}
    try:
        watch.sample()
        assert watch.reason is None
        try:
            watch.sample()
            time.sleep(1)
        except BaseException as error:  # noqa: BLE001
            g.void(report, error)
        g.finish(report)
    finally:
        g.uninstall()
    assert report["void_reason"].startswith("StageInterrupted: G-memory: process-tree PSS 3.00")
    assert report["memory"]["peak_tree_pss_gib"] == 3.0
    assert report["memory"]["peak_tree_rss_gib"] == 4.0


# ----- F12: budget rules ------------------------------------------------------------------------
def test_the_task073_and_task074_budget_rules_escalate_by_design():
    from embodied_jepa import lewm_planner_v2 as lp
    from embodied_jepa import wm_critic_v2 as wc

    for budget in (wc.BUDGET, lp.BUDGET):
        out = rt.check_budget(budget)
        assert not out["consistent"]
        assert out["min_cap"] == 120_000
        assert any("escalates by design" in p for p in out["problems"])


def test_a_cap_at_factor_times_calibration_is_consistent():
    out = rt.budget_rule_consistent(
        120_000,
        2.0,
        60_000,
        step=5_000,
        min_updates=10_000,
        calibration_select_every=1_000,
        selection_points=20,
        saturation_tolerance=0.01,
    )
    assert out == {"consistent": True, "problems": [], "max_request": 120_000.0, "min_cap": 120_000}


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"cap": 100_000}, "escalates by design"),
        ({"cap": 122_500, "calibration_updates": 61_000}, "not a multiple of step"),
        ({"min_updates": 200_000}, "min 200000 > cap"),
        ({"selection_points": 2}, "selection_points"),
        ({"saturation_tolerance": 1.5}, "saturation_tolerance"),
        ({"calibration_select_every": 7_000}, "multiple of its select_every"),
        ({"factor": 0.5}, "factor"),
        ({"cap": 0}, "cap must be a positive integer"),
    ],
)
def test_each_inconsistency_is_named(kwargs, fragment):
    args = {"cap": 120_000, "factor": 2.0, "calibration_updates": 60_000, "step": 5_000} | kwargs
    cap, factor, calibration = args.pop("cap"), args.pop("factor"), args.pop("calibration_updates")
    out = rt.budget_rule_consistent(cap, factor, calibration, **args)
    assert not out["consistent"]
    assert any(fragment in p for p in out["problems"]), out["problems"]


def test_the_step_rounds_the_minimum_cap_up():
    out = rt.budget_rule_consistent(125_000, 2.0, 61_000, step=5_000)
    assert out["min_cap"] == 125_000 and out["consistent"]
    assert not rt.budget_rule_consistent(120_000, 2.0, 61_000, step=5_000)["consistent"]


def test_selection_takes_the_earliest_checkpoint_within_tolerance():
    flat_tail = [[1000, 1.0], [2000, 0.5], [3000, 0.4], [4000, 0.399], [5000, 0.398]]
    assert rt.select_checkpoint(flat_tail, 0.01) == 3000
    assert not rt.last_two_triggered(flat_tail, 0.01)
    # the raw argmin (the TASK-074 last-two rule) picks the last point
    assert min(flat_tail, key=lambda p: p[1])[0] == 5000
    still_falling = [[1000, 1.0], [2000, 0.5], [3000, 0.4], [4000, 0.3], [5000, 0.2]]
    assert rt.select_checkpoint(still_falling, 0.01) == 5000
    assert rt.last_two_triggered(still_falling, 0.01)


def test_selection_rejects_bad_curves():
    with pytest.raises(ValueError):
        rt.select_checkpoint([[2000, 1.0], [1000, 0.5]], 0.01)
    with pytest.raises(ValueError):
        rt.select_checkpoint([[1000, float("nan")]], 0.01)


# ----- F13: the GPU guard -----------------------------------------------------------------------
class _FakeCuda:
    def __init__(self, free_gib, total_gib=16.0):
        self.free, self.total = free_gib * GIB, total_gib * GIB
        self.fraction = None

    def mem_get_info(self, device=0):
        return self.free, self.total

    def set_per_process_memory_fraction(self, fraction, device=0):
        self.fraction = fraction


def test_gpu_guard_records_and_caps(monkeypatch):
    monkeypatch.setenv(rt.GPU_LOCK_ENV, "1")
    cuda, report = _FakeCuda(10.0), {}
    rt.gpu_guard(report, min_free_gib=2.0, process_cap_gib=6.0, require_lock=True, cuda=cuda)
    assert report["gpu_lock_held"] and report["gpu_free_gib_at_start"] == 10.0
    assert cuda.fraction == pytest.approx(6.0 / 16.0)


def test_gpu_guard_refuses_too_little_free_memory(monkeypatch):
    monkeypatch.delenv(rt.GPU_LOCK_ENV, raising=False)
    report: dict = {}
    with pytest.raises(rt.GuardError, match="G-gpu: 5.00 GiB free < 8.0 GiB"):
        rt.gpu_guard(report, min_free_gib=2.0, process_cap_gib=6.0, cuda=_FakeCuda(5.0))
    assert report["gpu_free_gib_at_start"] == 5.0 and not report["gpu_lock_held"]


def test_gpu_guard_can_require_the_machine_lock(monkeypatch):
    monkeypatch.delenv(rt.GPU_LOCK_ENV, raising=False)
    with pytest.raises(rt.GuardError, match="gpu_run.sh"):
        rt.gpu_guard({}, min_free_gib=1.0, require_lock=True, cuda=_FakeCuda(10.0))


# ----- F15: the scale probe runs the stage's own function ----------------------------------------
class _SmokeReader:
    def __init__(self):
        self.manifest = {"splits": {"train": ["a", "b", "c"], "val": ["d"], "test": ["t"]}}
        self.reads: list[str] = []

    def episode(self, episode_id, keys=()):
        self.reads.append(episode_id)
        return {"id": episode_id, "keys": tuple(keys)}


class _Views:
    def arrays(self, episode_id, keys):
        return {"views_of": episode_id}


def test_cycled_reader_presents_the_real_sizes_and_decodes_every_slot():
    smoke = _SmokeReader()
    cycled = rt.CycledReader(smoke, {"train": 5, "val": 2}, read_splits=("train", "val"))
    assert [len(cycled.manifest["splits"][s]) for s in ("train", "val")] == [5, 2]
    assert list(cycled.source.values()) == ["a", "b", "c", "d", "a", "b", "c"]
    for slot in cycled.manifest["splits"]["train"]:
        cycled.episode(slot, keys=("apple",))
    assert smoke.reads == ["a", "b", "c", "d", "a"]
    views = rt.SlotProxy(_Views(), cycled.source)
    assert views.arrays("scale-val-1", ("x",)) == {"views_of": "c"}


class _Watch:
    measure = "pss"

    def __init__(self, peak):
        self.peak = peak

    def sample(self):
        return {"pss": GIB, "rss": GIB}


def test_the_scale_probe_calls_the_same_function_object_as_the_stage():
    calls: list = []

    def readouts_core(reader, *, smoke, probe=False):
        calls.append((readouts_core, probe, smoke))
        return {"slots": sum(len(v) for v in reader.manifest["splits"].values())}

    def stage(reader):  # the real stage calls the same core
        return readouts_core(reader, smoke=False)

    stage(_SmokeReader())
    cycled = rt.CycledReader(_SmokeReader(), {"train": 240, "val": 30})
    record = rt.scale_probe(
        readouts_core, cycled, smoke=True, watch=_Watch(8 * GIB), ceiling_gib=14, margin_gib=2
    )
    assert calls[0][0] is calls[1][0] is readouts_core
    assert [c[1] for c in calls] == [False, True]
    assert record["result"] == {"slots": 270} and record["within_margin"]
    assert record["stage_core"].endswith("readouts_core")


def test_the_scale_probe_fails_above_the_margin_and_needs_a_probe_keyword():
    def core(reader, probe=False):
        return None

    with pytest.raises(rt.GuardError, match="G-memory-margin"):
        rt.scale_probe(core, None, watch=_Watch(12.5 * GIB), ceiling_gib=14, margin_gib=2)

    def copy_of_the_steps(reader):
        return None

    with pytest.raises(TypeError, match="no probe keyword"):
        rt.scale_probe(copy_of_the_steps, None, watch=_Watch(GIB), ceiling_gib=14, margin_gib=2)


def test_run_tools_imports_without_torch_or_mujoco():
    import subprocess

    code = (
        "import sys, embodied_jepa.run_tools; "
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules"
    )
    env = os.environ | {"PYTHONPATH": str(ROOT / "src")}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_the_scale_probe_judges_its_own_peak_not_the_watchs_history():
    """An earlier 13 GiB peak (before the probe) must not fail a probe that peaks at 9 GiB."""
    values = [(13 * GIB, 13 * GIB), (GIB, GIB), (9 * GIB, 9 * GIB), (2 * GIB, 2 * GIB)]
    watch = rt.MemoryWatch(20 * GIB, 60.0, sampler=_Sampler(values))
    watch.sample()  # the history: 13 GiB

    def core(reader, probe=False):
        watch.sample()  # the probe's own peak: 9 GiB
        return "ran"

    record = rt.scale_probe(core, None, watch=watch, ceiling_gib=14, margin_gib=2)
    assert record["peak_scope"] == "probe" and record["probe_peak_tree_gib"] == 9.0
    assert record["cumulative_peak_tree_gib"] == 13.0 and record["within_margin"]
