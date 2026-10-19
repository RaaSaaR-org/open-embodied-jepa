"""TASK-101 amendment A1: the check ledger and §18.5's ordering rule (the runner refuses episodes
and the summariser refuses counts out of order), the spent cohorts, the D' projection and the
S0-NULL diagnostic Delta's measures and rows. CPU only, no simulator, no model."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import so101_closed_loop as cl

ROOT = Path(__file__).resolve().parents[1]
REV = "a" * 40
GOALS = "b" * 64


def _scripts(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        return __import__(name)
    finally:
        sys.path.pop(0)


def _entry(check, passes=True, revision=REV, utc="2026-10-20T10:00:00Z", goals=None, dirty=False):
    return cl.ledger_entry(
        check=check,
        passes=passes,
        revision=revision,
        dirty=dirty,
        finished_utc=utc,
        goals_sha256=goals,
    )


def _full_ledger(goals=GOALS):
    return [_entry(c) for c in cl.ONCE_CHECKS] + [
        _entry("renc", goals=goals, utc="2026-10-20T11:00:00Z")
    ]


def test_the_ledger_admits_only_a_complete_passing_set_at_the_revision(tmp_path):
    path = tmp_path / cl.LEDGER_FILE
    with pytest.raises(cl.GuardError, match="G-ledger"):  # no ledger at all
        cl.require_ledger(path, revision=REV, goals_sha256=GOALS)
    for e in _full_ledger():
        cl.ledger_append(path, e)
    v = cl.require_ledger(path, revision=REV, goals_sha256=GOALS)
    assert v["ok"] and v["latest_required_utc"] == "2026-10-20T11:00:00Z"
    # another revision
    with pytest.raises(cl.GuardError, match="no passing 7.2.0"):
        cl.require_ledger(path, revision="c" * 40, goals_sha256=GOALS)
    # a 7.2.2 report for another goal store
    with pytest.raises(cl.GuardError, match="7.2.2"):
        cl.require_ledger(path, revision=REV, goals_sha256="d" * 64)
    # a failing report of any check blocks every later episodes job
    cl.ledger_append(path, _entry("renc", passes=False, goals="e" * 64))
    with pytest.raises(cl.GuardError, match="failing check report"):
        cl.require_ledger(path, revision=REV, goals_sha256=GOALS)


def test_a_missing_or_dirty_check_is_not_admitted():
    for drop in cl.ONCE_CHECKS:
        entries = [e for e in _full_ledger() if e["check"] != drop]
        v = cl.ledger_verdict(entries, revision=REV, goals_sha256=GOALS)
        assert not v["ok"] and any(cl.CHECK_IDS[drop] in r for r in v["reasons"])
    dirty = [_entry(c, dirty=True) for c in cl.ONCE_CHECKS] + [_entry("renc", goals=GOALS)]
    assert not cl.ledger_verdict(dirty, revision=REV, goals_sha256=GOALS)["ok"]
    unknown = _full_ledger()
    unknown[0] = _entry(cl.ONCE_CHECKS[0], passes=None)
    assert not cl.ledger_verdict(unknown, revision=REV, goals_sha256=GOALS)["ok"]


def _goal_store(folder, cohort="Kp", task_name=cl.LIFT_TASK_NAME):
    n = 2
    arrays = {
        "seeds": np.array([101060, 101061]),
        "slots": np.arange(n),
        "frames": np.zeros((n, 4, 2, 2, 2, 3), np.uint8),
        "latents_raw": np.zeros((n, 4, 2, 3), np.float16),
        "joints": np.zeros((n, 4, 6)),
        "cube_cm": np.zeros((n, 4, 3)),
        "site_m": np.zeros((n, 4, 3)),
        "site_xmat": np.zeros((n, 4, 9)),
        "demo_actions": np.zeros((n, 5, 6), np.float32),
        "demo_lengths": np.full(n, 5),
    }
    folder.mkdir(parents=True)
    return cl.save_goal_store(folder, arrays, {"cohort": cohort, "task_name": task_name})


def _episodes_argv(goals, sha, out, cohort="Kp"):
    return [
        "episodes",
        "--cohort",
        cohort,
        "--goals",
        str(goals),
        "--goals-sha256",
        sha,
        "--arms",
        "W",
        "--configs",
        "H8-mean",
        "--output",
        str(out),
    ]


@pytest.mark.parametrize(
    "ledger",
    ["none", "failing", "other_revision", "other_goal_store"],
)
def test_episodes_are_refused_before_anything_is_built(tmp_path, monkeypatch, ledger):
    run = _scripts("run_task101")
    goals = tmp_path / "Kp-goals"
    sha = _goal_store(goals)
    monkeypatch.setattr(run.rt, "tree_dirty", lambda: False)
    monkeypatch.setattr(run.rt, "git", lambda *a: REV)
    built = []
    monkeypatch.setattr(run.rt, "run_pool", lambda *a, **k: built.append(1) or [])
    monkeypatch.setattr(run.rt, "make_robot", lambda *a, **k: built.append(1))
    path = tmp_path / cl.LEDGER_FILE
    entries = {
        "none": [],
        "failing": _full_ledger(sha) + [_entry("scene", passes=False)],
        "other_revision": [
            e | {"revision": "c" * 40} if e["check"] == "latent" else e for e in _full_ledger(sha)
        ],
        "other_goal_store": _full_ledger("f" * 64),
    }[ledger]
    for e in entries:
        cl.ledger_append(path, e)
    out = tmp_path / "Kp-episodes"
    with pytest.raises(cl.GuardError, match="G-ledger"):
        run.main(_episodes_argv(goals, sha, out))
    assert not out.exists() and not built


def test_episodes_pass_the_ledger_when_it_admits_them(tmp_path, monkeypatch):
    run = _scripts("run_task101")
    goals = tmp_path / "Kp-goals"
    sha = _goal_store(goals)
    monkeypatch.setattr(run.rt, "tree_dirty", lambda: False)
    monkeypatch.setattr(run.rt, "git", lambda *a: REV)

    class Reached(Exception):
        pass

    def stop(*a, **k):
        raise Reached

    monkeypatch.setattr(run.rt, "header", stop)  # the first step after the ledger
    for e in _full_ledger(sha):
        cl.ledger_append(tmp_path / cl.LEDGER_FILE, e)
    with pytest.raises(Reached):
        run.main(_episodes_argv(goals, sha, tmp_path / "Kp-episodes"))


def test_a_goal_store_of_another_task_is_refused(tmp_path):
    run = _scripts("run_task101")
    goals = tmp_path / "Kp-goals"
    sha = _goal_store(goals, task_name=None)
    with pytest.raises(cl.GuardError, match="not made on so101_cube_lift_v1"):
        run.main(_episodes_argv(goals, sha, tmp_path / "out"))


def test_spent_cohorts_are_refused_for_goals_and_episodes(tmp_path):
    run = _scripts("run_task101")
    for cohort in ("K", "D"):
        with pytest.raises(cl.GuardError, match="G-spent"):
            run.main(["goals", "--cohort", cohort, "--output", str(tmp_path / f"{cohort}-g")])
        with pytest.raises(cl.GuardError, match="G-spent"):
            run.main(_episodes_argv(tmp_path, "0" * 64, tmp_path / f"{cohort}-e", cohort))
        assert not (tmp_path / f"{cohort}-g").exists()


def test_the_goal_job_writes_no_episode():
    run = _scripts("run_task101")
    source = inspect.getsource(run.cmd_goals)
    assert "run_pool" not in source and "run_episode" not in source
    with pytest.raises(SystemExit):
        run.main(["goals", "--cohort", "Kp", "--output", "x", "--evidence", "8"])


def test_every_check_appends_to_the_ledger(tmp_path, monkeypatch):
    run = _scripts("run_task101")
    monkeypatch.setitem(run.CHECKS, "scene", lambda args, report: {"passes": True})
    run.main(["check", "scene", "--device", "cpu", "--output", str(tmp_path / "check-scene")])
    (entry,) = cl.ledger_read(tmp_path / cl.LEDGER_FILE)
    assert entry["check"] == "scene" and entry["id"] == "7.2.0" and entry["passes"] is True
    report = json.loads((tmp_path / "check-scene" / "report.json").read_text())
    assert entry["revision"] == report["revision"]
    assert entry["finished_utc"] == report["finished_utc"]


def test_the_summariser_refuses_an_out_of_order_cohort(tmp_path):
    summ = _scripts("summarize_task101")
    for e in _full_ledger():
        cl.ledger_append(tmp_path / cl.LEDGER_FILE, e)
    rep = {
        "cohort": "Kp",
        "arms": "W",
        "revision": REV,
        "tracked_tree_dirty": False,
        "started_utc": "2026-10-20T12:00:00Z",
    }
    assert summ.ordered_counts(tmp_path, rep, GOALS)["revision"] == REV
    for early in ("2026-10-20T10:30:00Z", "2026-10-20T11:00:00Z"):
        with pytest.raises(SystemExit, match="refusing to read counts"):
            summ.ordered_counts(tmp_path, rep | {"started_utc": early}, GOALS)
    with pytest.raises(SystemExit):
        summ.ordered_counts(tmp_path, rep | {"revision": "c" * 40}, GOALS)
    # a later failing report does not reorder what ran before it
    cl.ledger_append(tmp_path / cl.LEDGER_FILE, _entry("renc", passes=False, goals="e" * 64))
    assert summ.ordered_counts(tmp_path, rep, GOALS)


def test_the_dprime_projection_drops_the_diagnostic_first():
    fits = cl.project_dprime(
        used_gpu_h=2.8, goals_and_checks_gpu_h=0.1, seconds_per_cem_episode_wall=30.0
    )
    assert not fits["drop_diagnostic"] and not fits["over"]
    drop = cl.project_dprime(
        used_gpu_h=3.7, goals_and_checks_gpu_h=0.1, seconds_per_cem_episode_wall=30.0
    )
    assert drop["drop_diagnostic"] and not drop["over"]
    over = cl.project_dprime(
        used_gpu_h=4.0, goals_and_checks_gpu_h=0.1, seconds_per_cem_episode_wall=30.0
    )
    assert over["over"]


# ----- Delta (§18.6) --
def test_delta_states_seeds_and_windows():
    assert cl.DELTA_STATES == (0, 10, 20, 30, 40, 50, 60) and cl.SALTS["delta"] == 10108
    assert [cl.delta_subgoal(t) for t in cl.DELTA_STATES] == [0, 0, 0, 0, 1, 1, 1]
    assert cl.delta_planner_seed(101703, 50) == 10108 * 10**6 + 3050
    with pytest.raises(cl.GuardError, match="G-seed"):
        cl.delta_planner_seed(101600, 0)
    assert [t for t in cl.DELTA_STATES if cl.delta_crosses_boundary(t, 16)] == [30, 60]
    assert not any(cl.delta_crosses_boundary(t, 8) for t in cl.DELTA_STATES)
    a = cl.delta_random_rng(101700, 10).uniform(size=3)
    assert np.array_equal(a, cl.delta_random_rng(101700, 10).uniform(size=3))
    assert not np.array_equal(a, cl.delta_random_rng(101700, 20).uniform(size=3))


def test_normalised_rank_and_measures():
    assert cl.normalised_rank(0.0, [1.0, 2.0, 3.0, 4.0]) == 0.0
    assert cl.normalised_rank(5.0, [1.0, 2.0, 3.0, 4.0]) == 1.0
    assert cl.normalised_rank(2.0, [1.0, 2.0, 3.0, 4.0]) == 0.375
    rows = [
        {"t": t, "cost_demo": 1.0, "cost_cem": 2.0 if t < 40 else 0.5, "rank": r, "m3": True}
        for t, r in zip(cl.DELTA_STATES, (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6), strict=True)
    ]
    m = cl.delta_measures(rows, 16)
    assert m["all"]["states"] == 7 and m["all"]["m1"] == pytest.approx(4 / 7)
    assert m["all"]["m2"] == 0.3 and m["all"]["m3"] == 1.0
    assert m["per_subgoal"]["1"]["m1"] == 1.0 and m["per_subgoal"]["2"]["m1"] == 0.0
    assert m["without_boundary_windows"]["states"] == 5


def test_every_delta_row_in_both_directions():
    assert cl.delta_row(kept=11, m1=1.0, m2=0.0)["row"] == "Δ-VOID"
    assert cl.delta_row(kept=16, m1=1.0, m2=0.0, guard_failed=True)["row"] == "Δ-VOID"
    assert cl.delta_row(kept=16, m1=1.0, m2=0.0, over_budget=True)["row"] == "Δ-VOID"
    assert cl.delta_row(kept=12, m1=1.0, m2=0.25)["row"] == "Δ-MODEL"
    assert cl.delta_row(kept=16, m1=0.0, m2=0.40)["row"] == "Δ-MODEL"
    assert cl.delta_row(kept=16, m1=0.50, m2=0.10)["row"] == "Δ-PLANNER"
    assert cl.delta_row(kept=16, m1=0.49, m2=0.10)["row"] == "Δ-MIXED"
    assert cl.delta_row(kept=16, m1=0.9, m2=0.11)["row"] == "Δ-MIXED"
    assert cl.delta_row(kept=16, m1=0.9, m2=0.2499)["row"] == "Δ-MIXED"
