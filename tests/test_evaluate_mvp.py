"""Interrupted evaluation must not silently shrink the planned denominator."""

import importlib.util
import json
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "scripts/evaluate_mvp.py"
spec = importlib.util.spec_from_file_location("evaluate_mvp", MODULE)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_partial_jsonl_keeps_missing_seeds_explicit(tmp_path):
    (tmp_path / "episodes.jsonl").write_text(
        json.dumps({"seed": 20000}) + "\n" + json.dumps({"seed": 20001}) + '\n{"seed":'
    )
    accounting = evaluation.observed_episodes(tmp_path)
    assert accounting["recorded_episode_seeds"] == [20000, 20001]
    assert len(accounting["unrecorded_episode_seeds"]) == 48
    assert accounting["record_errors"][0]["line"] == 3
    assert not accounting["summary_present"]


def test_launch_failure_is_recorded_for_every_planned_run(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["evaluate_mvp.py", "--output", str(tmp_path / "report")])

    def fail(*args, **kwargs):
        raise OSError("launch denied in fixture")

    monkeypatch.setattr(evaluation.subprocess, "Popen", fail)
    assert evaluation.main() == 1
    report = json.loads((tmp_path / "report/report.json").read_text())
    assert len(report["attempts"]) == 8 and not report["complete"]
    assert all(a["status"] == "orchestration_error" for a in report["attempts"])
    assert all("launch denied" in a["error"] for a in report["attempts"])
