"""TASK-084 part A's decision rule (scripts/summarize_task084.py) on synthetic runs."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_task084.py"
spec = importlib.util.spec_from_file_location("summarize_task084", SCRIPT)
summ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summ)


def summary(arm, seed, revision="abc"):
    return {
        "arm": arm,
        "meta_seed": seed,
        "checkpoint_sha256": summ.CHECKPOINT_SHA256,
        "action_scale_raw": [3.7030139, 3.3012128],
        "repository_revision": revision,
        "upstream_revision": "13cf1d9",
    }


def write_runs(root: Path, up_success, ours_success, *, shift_ours_state=False, revision="abc"):
    i = 0
    for seed in summ.META_SEEDS:
        for arm, successes in (("upstream", up_success), ("ours", ours_success)):
            folder = root / f"{arm}-s{seed}"
            folder.mkdir(parents=True)
            rev = revision if (arm, seed) == ("ours", 4) else "abc"
            (folder / "summary.json").write_text(json.dumps(summary(arm, seed, rev)))
            with (folder / "episodes.jsonl").open("w") as handle:
                for ep in range(summ.EPISODES_PER_SEED):
                    k = i + ep
                    state = [float(k), 1.0]
                    if shift_ours_state and arm == "ours" and k == 0:
                        state = [99.0, 1.0]
                    row = {
                        "episode": ep,
                        "ep_seed": 1000 * seed + ep,
                        "success": bool(successes[k]),
                        "state_dist": 0.0,
                        "seconds": 1.0,
                        "init_state": state,
                        "goal_state": state,
                    }
                    handle.write(json.dumps(row) + "\n")
        i += summ.EPISODES_PER_SEED


def decide(tmp_path, up, ours, **kw):
    write_runs(tmp_path, up, ours, **kw)
    return summ.decide(summ.load(tmp_path))


def pattern(k: int, n: int = 96):
    return [j < k for j in range(n)]


def test_pass_when_both_within_ten_points(tmp_path):
    result = decide(tmp_path, pattern(67), pattern(62))  # 69.8 % and 64.6 %
    assert result["row"] == "P0-PASS"
    assert result["G-REPRO"] and result["G-PLAN"]
    assert result["discordant"] == {"upstream_only": 5, "ours_only": 0}


def test_planner_gap(tmp_path):
    result = decide(tmp_path, pattern(67), pattern(55))  # -12.5 points
    assert result["row"] == "P0-PLANNER-GAP"
    assert not result["G-PLAN-REL"]


def test_planner_gap_on_the_literal_gate(tmp_path):
    # Upstream 58/96 = 60.4 % reproduces; ours 52/96 = 54.2 % is within 10 of upstream but more
    # than 10 below the published 70.2: the plan's literal gate fails.
    result = decide(tmp_path, pattern(58), pattern(52))
    assert result["G-REPRO"] and result["G-PLAN-REL"] and not result["G-PLAN-ABS"]
    assert result["row"] == "P0-PLANNER-GAP"


def test_repro_fail_takes_precedence_over_the_planner(tmp_path):
    result = decide(tmp_path, pattern(50), pattern(50))  # 52.1 % < 60.2 %
    assert result["row"] == "P0-REPRO-FAIL"
    assert result["G-PLAN-REL"] and not result["G-PLAN-ABS"]


@pytest.mark.parametrize(
    ("k_up", "repro"), [(57, False), (58, True), (76, True), (77, False)]
)  # 59.38, 60.42, 79.17, 80.21 %
def test_repro_window_edges(tmp_path, k_up, repro):
    result = decide(tmp_path, pattern(k_up), pattern(k_up))
    assert result["G-REPRO"] is repro


def test_relative_edge(tmp_path):
    # 58/96 = 60.42 % is inside [60.2, 80.2]; ours 48/96 is -10.42 points: a gap.
    result = decide(tmp_path, pattern(58), pattern(48))
    assert result["G-REPRO"] and not result["G-PLAN"]


def test_rerun_counts_when_the_first_did_not_complete(tmp_path):
    write_runs(tmp_path, pattern(67), pattern(62))
    first, rerun = tmp_path / "ours-s2", tmp_path / "ours-s2-r2"
    first.rename(rerun)
    first.mkdir()  # the failed first attempt: no summary
    assert summ.decide(summ.load(tmp_path))["row"] == "P0-PASS"


def test_rerun_of_a_completed_run_voids(tmp_path):
    write_runs(tmp_path, pattern(67), pattern(62))
    shutil.copytree(tmp_path / "ours-s2", tmp_path / "ours-s2-r2")
    assert summ.decide(summ.load(tmp_path))["row"] == "P0-VOID"


def test_missing_run_voids_without_crashing(tmp_path):
    write_runs(tmp_path, pattern(67), pattern(62))
    (tmp_path / "upstream-s3" / "summary.json").unlink()
    assert summ.decide(summ.load(tmp_path))["row"] == "P0-VOID"


def test_mixed_revisions_void(tmp_path):
    assert decide(tmp_path, pattern(67), pattern(62), revision="other")["row"] == "P0-VOID"


def test_void_when_episodes_differ(tmp_path):
    result = decide(tmp_path, pattern(67), pattern(67), shift_ours_state=True)
    assert result["row"] == "P0-VOID"


def test_wilson_and_mcnemar():
    lo, hi = summ.wilson(67, 96)
    assert 0.59 < lo < 0.61 and 0.78 < hi < 0.80
    assert summ.mcnemar_exact(0, 0) == 1.0
    assert summ.mcnemar_exact(0, 5) == pytest.approx(0.0625)
