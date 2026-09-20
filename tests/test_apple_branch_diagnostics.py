import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "branch_diagnostics", Path(__file__).parents[1] / "scripts/diagnose_apple_branches.py"
)
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


def pairs(correct=True, visible=True, action=True):
    images = np.array(
        [np.zeros((2, 2, 3)), np.full((2, 2, 3), 255 if visible else 0)], dtype=np.uint8
    )
    costs = np.array([[0.0, 1.0], [1.0, 0.0]])
    if not correct:
        costs = costs[::-1]
    return diag.pair_metrics(
        costs,
        costs,
        costs,
        images,
        np.array([[0.0], [1.0]]),
        np.array([np.zeros((2, 14)), np.full((2, 14), float(action))]),
        np.ones(2),
    )


def test_correct_counterfactual_assignments_beat_swapped_and_persistence():
    own = pairs()[0]
    wrong = pairs(False)[0]
    assert own["ranking"] == 1 and own["own"] == 0 and own["wrong"] == 1
    assert wrong["ranking"] == 0 and wrong["own"] == 1 and wrong["wrong"] == 0
    assert diag.decision(1.0, 1.0) == (0.5, 1)
    assert diag.decision(1.0, 1.0 + 1e-8) == (0.5, 1)


@pytest.mark.parametrize("visible,action", [(False, True), (True, False), (False, False)])
def test_noninformative_pairs_retained_without_passing(visible, action):
    result = pairs(visible=visible, action=action)
    assert len(result) == 1 and not result[0]["informative"]
    report = diag.aggregate([{"session_id": "x", "pairs": result}], bootstrap=10)
    assert report["pairs"] == 1 and report["informative_roots"] == 0
    assert report["accuracy"] is None and not report["gate_passed"]


def test_root_weighting_and_grouped_bootstrap_not_pair_pseudoreplication():
    good = {"session_id": "a", "pairs": pairs() * 20}
    bad = {"session_id": "b", "pairs": pairs(False)}
    result = diag.aggregate([good, bad], bootstrap=50)
    assert result["accuracy"] == 0.5
    assert result["error_reduction"] == 0
    assert result["parent_sessions"] == 2 and result["informative_pairs"] == 21
    assert not result["gate_passed"]
    assert result == diag.aggregate([good, bad], bootstrap=50)


def test_gate_requires_coverage_and_keeps_all_ties_in_denominator():
    rows = [{"session_id": str(i % 2), "pairs": pairs()} for i in range(6)]
    assert diag.aggregate(rows, bootstrap=30)["gate_passed"]
    assert not diag.aggregate(rows[:5], bootstrap=30)["gate_passed"]
    tied = pairs()[0] | {"ranking": 0.5, "ties": 2, "own": 1.0, "wrong": 1.0}
    ties = diag.aggregate(
        [{"session_id": str(i % 2), "pairs": [tied]} for i in range(6)], bootstrap=30
    )
    assert ties["ties"] == 12 and ties["accuracy"] == 0.5
    assert not ties["gate_passed"]


def episode():
    return SimpleNamespace(
        observations={"rgb": np.zeros((3, 2, 2, 3), np.uint8)},
        robot_states=np.zeros((3, 2), np.float32),
        state_mask=np.ones((3, 2), bool),
        timestamps=np.array([1.0, 2.0, 3.0]),
    )


@pytest.mark.parametrize("field", ["image", "state", "mask", "initial_time", "relative_time"])
def test_sibling_start_and_sensor_time_must_match_exactly(field):
    a, b = episode(), episode()
    diag.same_start([a, b], 2)
    if field == "image":
        b.observations["rgb"][0, 0, 0, 0] = 1
    elif field == "state":
        b.robot_states[0, 0] = 0.01
    elif field == "mask":
        b.state_mask[0, 0] = False
    elif field == "initial_time":
        b.timestamps += 1
    else:
        b.timestamps[2] += 0.1
    with pytest.raises(ValueError, match="identical"):
        diag.same_start([a, b], 2)


def fixture():
    attempt = {
        "root_id": "p-lift",
        "split": "val",
        "phase": "lift",
        "parent_episode_id": "p",
        "session_id": "s",
        "branch": "hold",
        "episode_id": "b",
    }
    rows = [
        {"episode_id": "p", "session_id": "s"},
        {
            "episode_id": "b",
            "session_id": "s",
            "metadata": {"root_id": "p-lift", "parent_episode_id": "p", "branch": "hold"},
        },
    ]
    return {"attempts": [attempt]}, {
        "episodes": rows,
        "splits": {"train": [], "val": ["p", "b"], "test": []},
    }


def test_metadata_cohort_keeps_missing_prefixes_and_rejects_test_or_lineage_leakage():
    report, manifest = fixture()
    roots = diag.roots_from_report(report, manifest)
    assert roots[0]["attempts"][0]["stored"]
    manifest["episodes"].pop()
    assert not diag.roots_from_report(report, manifest)[0]["attempts"][0]["stored"]
    report["attempts"][0]["split"] = "test"
    with pytest.raises(ValueError, match="TRAIN/VAL"):
        diag.roots_from_report(report, manifest)
    report, manifest = fixture()
    manifest["splits"]["val"].remove("b")
    manifest["splits"]["test"].append("b")
    with pytest.raises(ValueError, match="lineage/split"):
        diag.roots_from_report(report, manifest)


def test_duplicate_or_cross_parent_sibling_identity_is_rejected():
    report, manifest = fixture()
    report["attempts"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        diag.roots_from_report(report, manifest)


def test_incomplete_run_never_retains_passing_gate():
    report = {
        "status": "completed",
        "gate_passed": True,
        "summaries": [{"gate_passed": True, "phases": {"lift": {"gate_passed": True}}}],
        "results": [{"partial": True}],
    }
    diag.invalidate(report, "timeout")
    assert report["status"] == "incomplete" and not report["gate_passed"]
    assert not report["summaries"][0]["gate_passed"]
    assert not report["summaries"][0]["phases"]["lift"]["gate_passed"]
    assert report["results"] == [{"partial": True}]


def test_supervisor_rechecks_civil_wall_after_host_suspension(monkeypatch):
    class Process:
        killed = False
        waited = False

        def poll(self):
            return None

        def kill(self):
            self.killed = True

        def wait(self):
            self.waited = True

    process = Process()
    clocks = iter((114.0, 180.0))
    sleeps = []
    monkeypatch.setattr(diag, "elapsed", lambda: next(clocks))
    monkeypatch.setattr(diag.time, "sleep", sleeps.append)
    assert diag.wait_for_worker(process)
    assert process.killed and process.waited
    assert sleeps == [0.1]
