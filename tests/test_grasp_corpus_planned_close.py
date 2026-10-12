"""TASK-091 (docs/experiments/grasp_play_corpus_v1_planned_close.md): the redefined A2 measure on
synthetic plans, A″'s pass and fail rules, the seed ranges and their disjointness from TASK-090's,
the deferral counters, the job split and recombination, the summarizer's split and VOID checks,
the frozen-file pins and the core-import check. NumPy only, except the graphics opt-in test
(``JEPA_TEST_RENDER=1``), which runs real debug episodes split and unsplit and requires their
per-reset outcomes to be bitwise equal."""

from __future__ import annotations

import importlib.util
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import grasp_corpus_wm as gc
from embodied_jepa import grasp_corpus_wm_planned_close as pcl
from embodied_jepa import zero_shot as zs
from embodied_jepa import zero_shot_grasp as zg
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
render = pytest.mark.skipif(os.environ.get("JEPA_TEST_RENDER") != "1", reason="graphics opt-in")


def _script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_t", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plan(close_at=None, *, dz=0.0, h=8):
    """A normalised [h, 14] plan: right grasp -1 until step ``close_at`` (1-based), then +1."""
    p = np.zeros((h, 14), np.float32)
    p[:, pcl.GRASP_DIM] = -1.0
    if close_at is not None:
        p[close_at - 1 :, pcl.GRASP_DIM] = 1.0
    p[:, pcl.Z_DIM] = dz
    return p


# ----- A″ (§4) ---------------------------------------------------------------------------------
def test_redefined_close_bar_reads_the_whole_plan():
    r1 = [plan(1)] * 10 + [plan(3)] * 30 + [plan(4)] * 8 + [plan(5)] * 10 + [plan(None)] * 6
    r2 = [plan(1, dz=0.5)] * 48 + [plan(1, dz=-0.5)] * 16
    a2 = pcl.a2_second(r1, r2)
    assert a2["close_le4_R1"] == {"count": 48, "meets": True}
    assert a2["close_R1_reported"] == 10  # A′'s first-command bar, reported only
    assert a2["lift_R2"] == {"count": 48, "meets": True}
    assert a2["close_by_R1"] == {"2": 10, "4": 48, "8": 58}
    assert a2["never_close_R1"] == 6
    assert a2["first_close_step_R1"][:11] == [1] * 10 + [3]
    assert a2["first_close_step_R1"][-1] is None
    assert a2["meets"] is True
    assert len(a2["mean_grasp_per_step_R1"]) == 8
    # one root fewer closing by k <= 4: the bar is missed
    r1b = list(r1)
    r1b[47] = plan(5)
    assert pcl.a2_second(r1b, r2)["close_le4_R1"] == {"count": 47, "meets": False}
    assert pcl.a2_second(r1b, r2)["meets"] is False
    # the threshold is > 0.5, not >= 0.5
    p = plan(None)
    p[2, pcl.GRASP_DIM] = 0.5
    assert pcl.first_close_step(p) is None
    p[2, pcl.GRASP_DIM] = 0.5001
    assert pcl.first_close_step(p) == 3


def test_lift_bar_is_still_the_first_command():
    r1 = [plan(1)] * 64
    late = plan(2, dz=0.5)  # closes at step 2 only: the first command does not lift
    r2 = [late] * 64
    assert pcl.a2_second(r1, r2)["lift_R2"]["count"] == 0
    r2 = [plan(1, dz=0.0)] * 64  # dz must be > 0
    assert pcl.a2_second(r1, r2)["lift_R2"]["count"] == 0
    with pytest.raises(ContractError):
        pcl.a2_second(np.zeros((4, 14)), np.zeros((4, 8, 14)))


def test_a_second_pass_and_rows():
    a1 = {"pass": True}
    good = pcl.a2_second([plan(3)] * 64, [plan(1, dz=1)] * 64)
    bad = pcl.a2_second([plan(6)] * 64, [plan(1, dz=1)] * 64)
    assert pcl.a_second_pass(a1, good) and not pcl.a_second_pass(a1, bad)
    assert not pcl.a_second_pass({"pass": False}, good)
    ok = {90100: True, 90101: True, 90102: True}
    one = {90100: False, 90101: True, 90102: True}
    two = {90100: False, 90101: False, 90102: True}
    assert pcl.a_second_row({"v2g": ok, "v1p": ok})["row"] == "A″-PASS"
    assert pcl.a_second_row({"v2g": one, "v1p": one})["row"] == "A″-PASS"  # one seed each
    r = pcl.a_second_row({"v2g": ok, "v1p": two})
    assert r["row"] == "A″-FAIL" and r["failing_arms"] == ["v1p"]
    assert pcl.a_second_row({"v2g": two, "v1p": ok})["row"] == "A″-FAIL"
    assert pcl.a_second_row({"v2g": ok, "v1p": ok}, void="x")["row"] == "A″-VOID"
    assert pcl.A2_BAR == 48 and pcl.CLOSE_STEP_MAX == 4 and pcl.A_SECOND_H == 8
    assert pcl.A_SECOND_SETTING == "step-mean" and pcl.LAMBDA == zs.LAMBDA == 2.0552


def test_first_command_check_is_bitwise():
    p = plan(1)
    assert pcl.first_command_equal(p, p[0].copy())
    q = p[0].copy()
    q[3] = np.nextafter(np.float32(0), np.float32(1))
    assert not pcl.first_command_equal(p, q)
    z = p[0].copy()
    z[0] = -0.0  # equal as floats, not bitwise
    assert not pcl.first_command_equal(p, z)


# ----- seeds (§8) ------------------------------------------------------------------------------
def test_seed_ranges_sizes_and_disjointness():
    c = pcl.COHORTS
    assert (c["a2-roots"].start, c["a2-roots"].stop) == (906000, 906064)
    assert (c["bd-grasp"].start, c["bd-grasp"].stop) == (90300, 90332)
    assert (c["bs-grasp"].start, c["bs-grasp"].stop) == (907000, 907128)
    assert set(c["debug-roots"]) | set(c["debug-grasp"]) <= set(pcl.DEBUG_SEEDS)
    assert pcl.DEBUG_SEEDS == range(90700, 90800) and pcl.BS_N == 128
    ranges = list(c.values())
    for a, b in itertools.combinations(ranges, 2):
        assert not set(a) & set(b)
    old = list(pcl.TASK090_RANGES.values()) + list(gc.COHORTS.values())
    old += list(zg.COHORTS.values()) + list(zs.COHORTS.values())
    mine = set().union(*map(set, ranges))
    for r in old:
        assert not mine & set(r)
    # TASK-090's declared but unrun BD and BS cohorts are not reused (§5.2)
    assert set(gc.COHORTS["bd-grasp"]) == set(pcl.TASK090_RANGES["bd-grasp (declared, unrun)"])
    assert set(gc.COHORTS["bs-grasp"]) == set(pcl.TASK090_RANGES["bs-grasp (declared, unrun)"])
    pcl.check_seed("a2-roots", 906063)
    for bad in (("a2-roots", 906064), ("bs-grasp", 904000), ("bd-grasp", 90200)):
        with pytest.raises(ContractError):
            pcl.check_seed(*bad)
    # no new salt or model seed
    assert gc.SALTS["shuffle"] == 9002 and gc.SALTS["probe"] == 9004
    assert gc.SALTS["bootstrap"] == 9006 and zs.SALTS["cem"] == 8803
    assert pcl.DETERMINISM == ("M-v2g", 90100, 4)


def test_runs_admitted():
    pcl.check_run("bs-grasp", "M-v2g", 90102)
    pcl.check_run("bs-grasp", "M-v1", 87100)
    pcl.check_run("bd-grasp", "M-v1p", 90100)
    pcl.check_run("bd-grasp", "ik", None)
    for bad in (
        ("bs-grasp", "M-v1", 87101),  # dropped from BS (R27.25)
        ("bd-grasp", "M-v2g", 90101),  # BD runs seed 90100 only
        ("bd-grasp", "ik", 90100),
        ("a2-roots", "ik", None),
        ("bs-grasp", "M-x", 1),
    ):
        with pytest.raises(ContractError):
            pcl.check_run(*bad)


def test_models_and_pins():
    models = pcl.a_second_models()
    assert len(models) == 11 and sum(a == "N" for a, _, _ in models) == 2
    names = {pcl.model_name(*m) for m in models}
    assert names == set(pcl.CHECKPOINT_SHA256)
    assert pcl.CHECKPOINT_SHA256["G-v2g-90100"].startswith("6ca90525")
    assert pcl.CHECKPOINT_SHA256["G-v2g-90100"].endswith("c375")
    assert pcl.CHECKPOINT_SHA256["G-v1p-90100"].startswith("1a8a2991")
    assert pcl.MOMENTS_SHA256.startswith("1e960227") and pcl.MOMENTS_SHA256.endswith("81b5c")
    assert pcl.FEATURE_REPORTS_SHA256["play-v2g"].startswith("d8afb236")


def test_frozen_files_match_their_pins():
    assert all(pcl.frozen_files_check(ROOT).values())


# ----- the job split and the cap (§5.4) --------------------------------------------------------
def test_bs_job_split():
    parts = pcl.part_ranges("bs-grasp", pcl.BS_PARTS)
    assert [(p.start, p.stop) for p in parts] == [(907000, 907064), (907064, 907128)]
    goals = [{"seed": s} for s in pcl.COHORTS["bs-grasp"]]
    a, b = pcl.split_goals(goals, 1, 2), pcl.split_goals(goals, 2, 2)
    assert [g["seed"] for g in a + b] == list(pcl.COHORTS["bs-grasp"])
    assert len(a) == len(b) == 64
    three = [pcl.split_goals(goals, i, 3) for i in (1, 2, 3)]
    assert sum(len(x) for x in three) == 128
    assert pcl.needs_parts("bs-grasp", "M-v2g") == 2
    assert pcl.needs_parts("bs-grasp", "ik") is None
    assert pcl.needs_parts("bd-grasp", "M-v2g") is None
    assert pcl.parse_part("2/2") == (2, 2)
    for bad in ("0/2", "3/2"):
        with pytest.raises(ContractError):
            pcl.parse_part(bad)
    assert pcl.JOB_CAP_SECONDS == 7200 and pcl.SPLIT_TRIGGER_SECONDS == 5400


def test_projection_and_parts_needed():
    # TASK-089: about 0.95 s per command at 6 processes, 400 commands, 64 resets -> about 4 300 s
    job = pcl.project_job_seconds(0.95, 400, 64, 6)
    assert 4000 < job < 4300
    assert pcl.parts_needed(2 * job) == 2
    assert pcl.parts_needed(2 * 6000) == 3  # a 64-reset job above 5 400 s splits further
    assert pcl.parts_needed(2 * 5400) == 2


def run_part(seeds, part, *, ok=True, **kw):
    eps = [{"seed": s, "success": bool(s % 3 == 0), "applied": [[0.0] * 14]} for s in seeds]
    out = {
        "task": pcl.TASK,
        "cohort": "bs-grasp",
        "arm": "M-v2g",
        "model_seed": 90100,
        "revision": "r",
        "goals_sha256": "g",
        "setting": "step-mean",
        "cem": zs.CEM,
        "lambda": zs.LAMBDA,
        "feature_reports_sha256": pcl.FEATURE_REPORTS_SHA256,
        "determinism_rerun": False,
        "plan_record": True,
        "model": {"checkpoint_sha256": "c"},
        "tracked_tree_dirty": False,
        "part": list(part),
        "workers": 6,
        "episodes": eps,
        "seconds": 10.0,
        "completed": ok,
        "cap_hit": not ok,
    }
    return out | kw


def test_combine_parts_checks():
    seeds = list(pcl.COHORTS["bs-grasp"])
    a, b = run_part(seeds[:64], (1, 2)), run_part(seeds[64:], (2, 2))
    c = pcl.combine_parts([b, a], seeds)
    assert [e["seed"] for e in c["episodes"]] == seeds and c["n"] == 128
    assert c["successes"] == sum(s % 3 == 0 for s in seeds) and c["seconds"] == 20.0
    assert c["combined"] and c["completed"] and not c["cap_hit"]
    with pytest.raises(ContractError):  # a part hit the cap
        pcl.combine_parts([a, run_part(seeds[64:], (2, 2), ok=False)], seeds)
    with pytest.raises(ContractError):  # a part missing
        pcl.combine_parts([a], seeds)
    with pytest.raises(ContractError):  # another model
        pcl.combine_parts([a, run_part(seeds[64:], (2, 2), model_seed=90101)], seeds)
    with pytest.raises(ContractError):  # another revision
        pcl.combine_parts([a, run_part(seeds[64:], (2, 2), revision="x")], seeds)
    with pytest.raises(ContractError):  # resets overlap
        pcl.combine_parts([a, run_part(seeds[63:127], (2, 2))], seeds)


# ----- deferral (§5.3) -------------------------------------------------------------------------
def episode(grasp, contact, phases, plan_close):
    t = len(grasp)
    applied = np.zeros((t, 14))
    applied[:, pcl.GRASP_DIM] = grasp
    log = []
    for i in range(t):
        g = [-1.0] * 8
        if plan_close[i] is not None:
            g = [-1.0] * (plan_close[i] - 1) + [1.0] * (9 - plan_close[i])
        log.append({"index": phases[i], "grasp": g})
    return {"applied": applied.tolist(), "contact": list(contact), "plan_log": log}


def test_deferral_measures():
    grasp = [-1, -1, -1, 1, -1, 1, 1, 1]
    contact = [False] * 6 + [True, True]
    phases = [0, 1, 2, 2, 2, 2, 2, 3]
    plan_close = [None, 5, 3, 1, 2, 1, 1, 1]
    m = pcl.deferral_measures(episode(grasp, contact, phases, plan_close))
    assert m["first_contact_command"] == 6 and not m["no_contact"]
    assert m["m1_closing_before_contact"] == 3  # commands 3, 5, 6 (up to the contact command)
    assert m["m1_commands_before_contact"] == 7
    assert m["grasp_phase_commands"] == 5
    assert m["m2_first_close_in_grasp_phase"] == 2  # grasp phase starts at command 2
    assert m["m3_plan_close_steps"] == [3, 1, 2, 1, 1]
    assert m["m3_share_k1"] == 0.6 and m["m3_median_k"] == 1.0
    assert m["m3_longest_run_k_gt_1"] == 1
    # no contact: counted to the end and flagged; never closing in the grasp phase
    m = pcl.deferral_measures(episode([1, -1, -1], [False] * 3, [1, 2, 2], [1, None, 4]))
    assert m["no_contact"] and m["m1_closing_before_contact"] == 1
    assert m["m2_first_close_in_grasp_phase"] is None and m["grasp_phase_reached"]
    assert m["m3_plan_close_steps"] == [None, 4] and m["m3_longest_run_k_gt_1"] == 2
    assert m["m3_median_k"] == 4.0 and m["m3_never_close"] == 1
    # never reaching the grasp phase
    m = pcl.deferral_measures(episode([-1, -1], [False, False], [0, 1], [None, None]))
    assert not m["grasp_phase_reached"] and m["m3_replans"] == 0 and m["m3_share_k1"] is None
    # a refused last request leaves one more plan than commands; more is an error
    e = episode([1, 1], [False, True], [2, 2], [1, 1])
    e["plan_log"].append({"index": 2, "grasp": [1.0] * 8})
    assert pcl.deferral_measures(e)["m3_replans"] == 2
    e["plan_log"].append({"index": 2, "grasp": [1.0] * 8})
    with pytest.raises(ContractError):
        pcl.deferral_measures(e)
    s = pcl.deferral_summary(
        [pcl.deferral_measures(episode(grasp, contact, phases, plan_close))] * 3
    )
    assert s["episodes"] == 3 and s["m3_share_k1_pooled"] == 0.6
    assert s["m2_first_close_in_grasp_phase"]["p10_25_50_75_90"][2] == 2.0


# ----- the summarizer on synthetic runs (debug cohort) -----------------------------------------
def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _goals(tmp, cohort, seeds):
    _write(
        tmp / f"goals-{cohort}.json",
        {"goals": [{"seed": s, "target": "cube"} for s in seeds], "rejections": 0},
    )


def _run(arm, seeds, succ, model_seed=None, **kw):
    eps = []
    for s, ok in zip(seeds, succ, strict=True):
        e = {
            "seed": s,
            "success": bool(ok),
            "applied": [[0.0] * 14],
            "any_grasp_contact": bool(ok),
            "max_rise": 0.1 if ok else 0.0,
            "final_subgoal": 3,
            "stop_reason": "",
            "switches": [{"steps": 1}],
            "steps": 1,
        }
        if arm in pcl.CLOSED_LOOP_ARMS:
            e["plan_seconds_mean"] = 1.0
            e["deferral"] = pcl.deferral_measures(
                {
                    "applied": [[0.0] * 14],
                    "contact": [bool(ok)],
                    "plan_log": [{"index": 2, "grasp": [1.0] * 8}],
                }  # fmt: skip
            )
        eps.append(e)
    out = {
        "task": pcl.TASK,
        "cohort": "debug-grasp",
        "arm": arm,
        "model_seed": model_seed,
        "revision": "r",
        "tracked_tree_dirty": False,
        "goals_sha256": "g",
        "setting": "step-mean",
        "cem": zs.CEM,
        "lambda": zs.LAMBDA,
        "feature_reports_sha256": None,
        "determinism_rerun": False,
        "plan_record": arm in pcl.CLOSED_LOOP_ARMS,
        "model": None,
        "workers": 6,
        "episodes": eps,
        "successes": int(sum(succ)),
        "n": len(eps),
        "seconds": 5.0,
        "completed": True,
        "cap_hit": False,
    }
    return out | kw


def test_summarizer_bs_split_and_void(tmp_path):
    s = _script("summarize_task091")
    seeds = list(range(90740, 90748))
    d = tmp_path / "debug-grasp"
    _goals(tmp_path, "debug-grasp", seeds)
    for a in ("ik", "ik-nopress", "hold", "random"):
        _write(d / f"{a}.json", _run(a, seeds, [a == "ik"] * 8))
    for arm, seed, k in (("M-v2g", 90100, 6), ("M-v1p", 90100, 1), ("M-v1", 87100, 2)):
        succ = [i < k for i in range(8)]
        p1 = _run(arm, seeds[:4], succ[:4], seed, part=[1, 2])
        p2 = _run(arm, seeds[4:], succ[4:], seed, part=[2, 2])
        _write(d / f"{arm}-{seed}.part1of2.json", p1)
        _write(d / f"{arm}-{seed}.part2of2.json", p2)
        _write(d / f"{arm}-{seed}.json", pcl.combine_parts([p1, p2], seeds))
    bd = {"bar": 0.4, "stage": "bd", "cohort": "debug-grasp", "problems": []}
    _write(tmp_path / "bd.json", bd)
    # the split checks on their own
    runs, parts = s._runs(tmp_path, "debug-grasp")
    assert set(parts) == {f"{a}.part{i}of2" for a in ("M-v2g-90100", "M-v1p-90100",
                          "M-v1-87100") for i in (1, 2)}  # fmt: skip
    assert s._split_checks(runs, parts, "debug-grasp", True) == []
    # a part that hit the cap, and a part without a combined run, are problems
    capped = dict(parts["M-v2g-90100.part2of2"], completed=False, cap_hit=True)
    bad = dict(parts, **{"M-v2g-90100.part2of2": capped})
    assert any("cap" in p for p in s._split_checks(runs, bad, "debug-grasp", True))
    orphan = dict(parts, **{"M-x-1.part1of2": parts["M-v2g-90100.part1of2"]})
    assert any("without" in p for p in s._split_checks(runs, orphan, "debug-grasp", True))
    # a BS world-model run that was not split is a problem outside debug
    unsplit = {k: dict(v, combined=False) for k, v in runs.items()}
    assert any("not combined" in p for p in s._split_checks(unsplit, {}, "bs-grasp", False))
    # the BD summary over the same runs: bar from ik's rate, deferral reported
    out = s.stage_bd(s.argparse.Namespace(runs=str(tmp_path), cohort="debug-grasp", only=None))
    assert out["ik_successes"] == 8 and out["bar"] == 0.4 and not out["problems"]
    assert out["deferral"]["M-v2g-90100"]["successes"] == 6
    # a run that did not complete under the cap makes the stage report a problem
    _write(d / "hold.json", _run("hold", seeds, [False] * 8, completed=False, cap_hit=True))
    out = s.stage_bd(s.argparse.Namespace(runs=str(tmp_path), cohort="debug-grasp", only=None))
    assert any("cap" in p for p in out["problems"])


def test_summarizer_bs_rows_three_seeds(tmp_path):
    s = _script("summarize_task091")
    seeds = list(range(90740, 90760))
    d = tmp_path / "debug-grasp"
    _goals(tmp_path, "debug-grasp", seeds)
    for a in ("ik", "ik-nopress", "hold", "random"):
        _write(d / f"{a}.json", _run(a, seeds, [a == "ik"] * 20))
    for arm, k in (("M-v2g", 18), ("M-v1p", 2)):
        for seed in gc.MODEL_SEEDS:
            succ = [i < k for i in range(20)]
            p1 = _run(arm, seeds[:10], succ[:10], seed, part=[1, 2])
            p2 = _run(arm, seeds[10:], succ[10:], seed, part=[2, 2])
            _write(d / f"{arm}-{seed}.part1of2.json", p1)
            _write(d / f"{arm}-{seed}.part2of2.json", p2)
            _write(d / f"{arm}-{seed}.json", pcl.combine_parts([p1, p2], seeds))
    _write(d / "M-v1-87100.json", _run("M-v1", seeds, [False] * 20, 87100))
    det = _run("M-v2g", seeds[:4], [True] * 4, 90100, determinism_rerun=True)
    _write(d / "M-v2g-90100-determinism.json", det)
    _write(tmp_path / "bd.json", {"bar": 0.4, "stage": "bd", "cohort": "debug-grasp"})
    ns = s.argparse.Namespace(
        runs=str(tmp_path),
        bd_summary=str(tmp_path / "bd.json"),
        cohort="debug-grasp",
        model_seeds=None,
        a_summary=None,
    )
    out = s.stage_bs(ns)
    assert out["determinism_matches"] is True and not out["problems"]
    assert out["row"] == "B-PASS" and out["row_detail"]["k_g"] == 54
    assert out["clause_fires"] is False and out["lewm_needed_claimed"] is False
    assert set(out["deferral"]) >= {"M-v2g-90100", "M-v1p-90102", "M-v1-87100"}
    # a determinism mismatch voids the row
    det2 = dict(det, episodes=[dict(e, applied=[[1.0] * 14]) for e in det["episodes"]])
    _write(d / "M-v2g-90100-determinism.json", det2)
    out = s.stage_bs(ns)
    assert out["row"] == "B-VOID" and out["determinism_matches"] is False


def test_runner_guards_without_a_gpu(monkeypatch, tmp_path):
    r = _script("run_task091")
    assert r.run_name("M-v2g", 90100, part=(1, 2)) == "M-v2g-90100.part1of2"
    assert r.run_name("ik", None) == "ik"
    assert r.run_name("M-v2g", 90100, determinism=True) == "M-v2g-90100-determinism"
    with pytest.raises(r.rt.GuardError):  # not under the GPU lock
        monkeypatch.delenv(r.rt.GPU_LOCK_ENV, raising=False)
        r.main(["run", "--cohort", "debug-grasp", "--arm", "ik", "--goals", str(tmp_path),
                "--output", str(tmp_path)])  # fmt: skip
    ns = r.argparse.Namespace(cap_seconds=60, cohort="bs-grasp")
    with pytest.raises(r.rt.GuardError):  # the cap is 7 200 s outside debug
        r._cap(ns)
    assert r._cap(r.argparse.Namespace(cap_seconds=60, cohort="debug-grasp")) == 60
    assert r._cap(r.argparse.Namespace(cap_seconds=None, cohort="bs-grasp")) == 7200
    monkeypatch.setattr(r, "LEDGER", tmp_path / "ledger")
    r.ledger_claim("goals-bs-grasp")
    with pytest.raises(r.rt.GuardError):  # a cohort started twice
        r.ledger_claim("goals-bs-grasp")


def test_cap_hit_writes_an_incomplete_report(tmp_path):
    r = _script("run_task091")
    out = tmp_path / "x.json"
    with pytest.raises(SystemExit) as stop:
        with r.Job(out, {"task": pcl.TASK}, 1):
            import time

            time.sleep(3)
    assert stop.value.code == r.EXIT_CAP
    rep = json.loads(out.read_text())
    assert rep["completed"] is False and rep["cap_hit"] is True


def test_core_import_stays_light():
    code = (
        "import sys, embodied_jepa, embodied_jepa.grasp_corpus_wm_planned_close;"
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


# ----- simulator (graphics opt-in): a split run equals the unsplit run per reset, bitwise ------
@render
def test_split_run_equals_unsplit_per_reset():
    r = _script("run_task091")
    seeds = list(pcl.COHORTS["debug-grasp"])[:3]
    goals = r._goal_worker(seeds)
    for g in goals:
        g["digest"] = zs.goal_digest(g)
    for arm in ("random", "ik"):
        whole, _ = r._run_worker((arm, None, goals, None, True))
        parts = []
        for i in (1, 2):
            out, _ = r._run_worker((arm, None, pcl.split_goals(goals, i, 2), None, True))
            parts += out
        assert [e["seed"] for e in parts] == [e["seed"] for e in whole] == seeds
        for a, b in zip(whole, parts, strict=True):
            assert a["applied"] == b["applied"] and a["success"] == b["success"], arm
            assert a["distances"] == b["distances"] and a["rise"] == b["rise"], arm
