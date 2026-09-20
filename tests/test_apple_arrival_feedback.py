"""Synthetic arrival-rule and failure-accounting tests; no model or physics execution."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_apple_arrival_feedback.py"
spec = importlib.util.spec_from_file_location("arrival_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def controller(distance=0.1, threshold=0.1):
    return SimpleNamespace(
        termination_reason=None,
        pending=None,
        goal_index=0,
        dwell=1,
        _commitment={"trace": {"plan_step": 58}, "offset": 2},
        waypoints=[SimpleNamespace(threshold=threshold)],
        _distance=lambda images, goal: distance,
    )


def test_arrival_interrupt_is_inclusive_and_does_not_advance_dwell():
    c = controller()
    event = audit.arrival_interrupt(c, SimpleNamespace(images={}))
    assert event["plan_step"] == 58 and event["commitment_offset"] == 2
    assert c._commitment is None and c.dwell == 1
    assert audit.arrival_interrupt(c, SimpleNamespace(images={})) is None
    # The intervention is applied repeatedly, including a later new commitment.
    c._commitment = {"trace": {"plan_step": 61}, "offset": 1}
    assert audit.arrival_interrupt(c, SimpleNamespace(images={}))["plan_step"] == 61


def test_outside_goal_or_unacknowledged_state_never_clears_cache():
    for c in (controller(0.10001), controller()):
        if c._distance({}, None) == 0.1:
            c.pending = ("unacknowledged",)
        original = c._commitment
        assert audit.arrival_interrupt(c, SimpleNamespace(images={})) is None
        assert c._commitment is original


def test_rule_uses_current_goal_after_advance():
    c = controller(0.1)
    c.waypoints.append(SimpleNamespace(threshold=0.01))
    c.goal_index = 1
    assert audit.arrival_interrupt(c, SimpleNamespace(images={})) is None


@pytest.mark.parametrize("durable", [False, True])
def test_finalization_distinguishes_pending_execution_from_durable_ack(tmp_path, durable):
    record = audit.planned()[0]
    record.update(status="running", advanced_past_goal2=False)
    audit.write(tmp_path / "report.json", {"records": [record, audit.planned()[1]]})
    path = tmp_path / "original/trace.jsonl"
    path.parent.mkdir()
    rows = [dict(event="pending", index=0)]
    if durable:
        rows.append(dict(event="result", index=0, executed=True))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert audit.finalize(SimpleNamespace(output=tmp_path), -9, True) == 2
    report = json.loads((tmp_path / "report.json").read_text())
    assert len(report["records"]) == 2
    row = report["records"][0]
    assert row["executed_steps"] == int(durable)
    assert row["possible_unobserved_execution"] is not durable
    assert row["advanced_past_goal2"] is None
    assert len(row["command_slots"]) == 16
    assert row["command_slots"][0]["status"] == ("accepted" if durable else "execution_uncertain")
    assert report["records"][1]["status"] == "not_started_budget"


def test_completed_negative_remains_false_not_unavailable(tmp_path):
    records = audit.planned()
    for row in records:
        row.update(status="completed", advanced_past_goal2=False)
    audit.write(tmp_path / "report.json", dict(records=records, integrity_verified_after=True))
    assert audit.finalize(SimpleNamespace(output=tmp_path), 0, False) == 0
    assert all(
        r["advanced_past_goal2"] is False
        for r in json.loads((tmp_path / "report.json").read_text())["records"]
    )


def test_gapped_acknowledgements_are_not_a_valid_prefix(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(dict(event="result", index=1, executed=True)) + "\n")
    with pytest.raises(ValueError, match="noncontiguous"):
        audit.recover_trace(path)


def test_committed_replay_mismatch_is_detected_without_inference():
    helper = SimpleNamespace(assert_trace_match=lambda *args, **kwargs: None)
    expected = dict(
        decision_kind="commitment", plan_step=58, commitment_offset=2, commitment_remaining=2
    )
    with pytest.raises(ValueError, match="commitment_offset"):
        audit.assert_match(helper, expected, expected | {"commitment_offset": 1})


def test_true_wall_elapsed_counts_suspend(monkeypatch):
    monkeypatch.setattr(audit.time, "time", lambda: 90.0)
    monkeypatch.setattr(audit.time, "monotonic", lambda: 2.0)
    assert audit.elapsed((0.0, 0.0)) == 90.0


def test_late_zero_exit_is_still_a_wall_timeout():
    child = SimpleNamespace(poll=lambda: 0, wait=lambda: 0)
    assert audit.supervise_child(child, clock=lambda: 58) == (0, True)
    assert audit.supervise_child(child, clock=lambda: 56) == (0, False)


def test_timeout_kill_race_is_preserved():
    child = SimpleNamespace(pid=123, poll=lambda: None, wait=lambda: 0)

    def already_finished(pid, signal):
        raise ProcessLookupError

    assert audit.supervise_child(child, clock=lambda: 58, kill=already_finished) == (0, True)


def test_bad_journal_preserves_incomplete_report(tmp_path):
    record = audit.planned()[0]
    record["status"] = "running"
    audit.write(tmp_path / "report.json", dict(records=[record, audit.planned()[1]]))
    path = tmp_path / "original/trace.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps(dict(event="result", index=1, executed=True)) + "\n")
    assert audit.finalize(SimpleNamespace(output=tmp_path), 2, False) == 2
    report = json.loads((tmp_path / "report.json").read_text())
    assert len(report["records"]) == 2
    assert "noncontiguous" in report["records"][0]["journal_error"]
