"""Synthetic audit logic only: no real model inference, replay or physics."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).with_name("audit_apple_control_forecasts.py")
if not SCRIPT.exists():
    SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_apple_control_forecasts.py"
spec = importlib.util.spec_from_file_location("forecast_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fake_trace():
    return dict(
        step=119,
        goal_index=5,
        goal_source_id="fixture/frame140",
        selected_origin="perturbation",
        selected_round=1,
        goal_dwell_observed=0,
        goal_image_sha256="synthetic",
        waypoint_advanced=False,
        observation_timestamp=5.95,
        execution_timestamp=6.0,
        sampled_action=[0.0] * 14,
        projected_action=[0.0] * 14,
        applied_action=[0.0] * 14,
        observed_distance=0.3,
        selected_cost=0.2,
        goal_threshold=0.06,
        rounds=[
            dict(
                selected_index=15,
                feasible_candidates=16,
                score_permutation=list(range(16)),
                candidate_costs=[0.3] + [0.25] * 14 + [0.2],
            )
            for _ in range(2)
        ],
    )


def test_recovery_uses_recorded_winner_and_lowest_feasible_demo_without_rng():
    trace = fake_trace()
    captures = []
    for _ in range(2):
        actions = (
            np.arange(16, dtype=np.float32)[:, None, None] * np.ones((16, 16, 14), np.float32) / 20
        )
        captures.append(dict(requested=actions.copy(), actions=actions, feasible=np.ones(16, bool)))
    trace["rounds"][0]["candidate_costs"][7] = 0.21
    trace["rounds"][1]["candidate_costs"][6] = 0.21
    trace["rounds"][0]["candidate_costs"][3] = None
    captures[0]["feasible"][3] = False
    state = np.random.get_state()
    found = audit.recover_candidates(trace, captures, 13)
    assert found["winner"]["round"] == 1 and found["winner"]["index"] == 15
    assert found["demonstration"]["round"] == 0 and found["demonstration"]["index"] == 7
    assert found["hold"]["index"] == 0
    assert found["winner"]["scored"].shape == (16, 14)
    np.testing.assert_array_equal(np.random.get_state()[1], state[1])
    for round in trace["rounds"]:
        round["candidate_costs"][2:15] = [None] * 13
    with pytest.raises(ValueError, match="no scored feasible demonstration"):
        audit.recover_candidates(trace, captures, 13)


def test_replay_matching_rejects_action_ranking_and_discrete_drift():
    import copy

    expected = fake_trace()
    actual = copy.deepcopy(expected)
    audit.assert_trace_match(expected, actual, acknowledged=True)
    actual["rounds"][0]["selected_index"] = 4
    with pytest.raises(ValueError, match="selected_index"):
        audit.assert_trace_match(expected, actual)
    actual = copy.deepcopy(expected)
    actual["projected_action"][0] = 1e-3
    with pytest.raises(ValueError, match="projected_action"):
        audit.assert_trace_match(expected, actual)
    actual = copy.deepcopy(expected)
    actual["applied_action"][0] = 1e-3
    with pytest.raises(ValueError, match="actual-applied"):
        audit.assert_trace_match(expected, actual, acknowledged=True)


def test_controller_snapshot_preserves_full_warm_pending_rng_and_latent_owner():
    owner = object()
    latent = SimpleNamespace(owner=owner)
    controller = SimpleNamespace(
        model=object(),
        waypoints=(),
        config=object(),
        progress_distance=lambda: None,
        _goal_latents={5: latent},
        rng=np.random.default_rng(3),
        warm=np.arange(16 * 14).reshape(16, 14),
        pending=(np.zeros(14), np.ones((16, 14)), {"cost": 3}),
        goal_index=5,
        dwell=2,
        steps=119,
    )
    saved = audit.capture_controller(controller)
    expected_rng = saved[0]["rng"].random()
    saved = audit.capture_controller(controller)
    controller.rng.random(50)
    controller.warm[:] = -5
    controller.pending[1][:] = 8
    controller.goal_index = 9
    controller.dwell = 0
    controller._goal_latents = {}
    audit.restore_controller(controller, saved)
    assert (
        controller.rng.random() == expected_rng
        and controller.goal_index == 5
        and controller.dwell == 2
    )
    assert controller.warm[0, 0] == 0 and np.all(controller.pending[1] == 1)
    assert controller._goal_latents[5].owner is owner
    # Snapshot remains reusable after another mutation.
    controller.warm[:] = 0
    audit.restore_controller(controller, saved)
    assert controller.warm[-1, -1] == 223


def test_robot_restore_includes_warmstart_targets_grasp_stop_and_refresh():
    import copy

    class MJ:
        @staticmethod
        def MjData(model):
            return SimpleNamespace()

        @staticmethod
        def mj_copyData(dest, model, source):
            vars(dest).clear()
            vars(dest).update(copy.deepcopy(vars(source)))

    sim = SimpleNamespace(
        data=SimpleNamespace(qpos=np.ones(3), qacc_warmstart=np.array([7.0])),
        targets=np.array([4.0]),
        stopped_reason="original",
    )
    robot = SimpleNamespace(
        mj=MJ, model=object(), sim=sim, _grasp=np.array([-0.4, 0.2]), _observation=None
    )

    def observe():
        robot.refreshed = True
        return "fresh"

    robot.observe = observe
    scorer = SimpleNamespace(robot=robot, stage={"grasp": True}, last_time=4.0)
    saved = audit.capture_robot(robot, scorer)
    sim.data.qacc_warmstart[:] = -8
    sim.targets[:] = -2
    robot._grasp[:] = 1
    sim.stopped_reason = "branch"
    scorer.stage["grasp"] = False
    assert audit.restore_robot(robot, scorer, saved) == "fresh"
    assert sim.data.qacc_warmstart[0] == 7 and sim.targets[0] == 4
    assert sim.stopped_reason == "original" and scorer.stage["grasp"] is True and robot.refreshed
    np.testing.assert_array_equal(robot._grasp, [-0.4, 0.2])


def test_actual_reforecast_uses_same_root_and_exact_unpadded_applied_prefix():
    root = object()
    applied = np.arange(3 * 14, dtype=np.float32).reshape(3, 14) / 100

    class Recorder:
        def predict(self, latent, actions):
            assert latent is root and actions.shape == (1, 1, 3, 14)
            np.testing.assert_array_equal(actions[0, 0], applied)
            return "recorded-prediction"

    assert audit.reforecast_actual(Recorder(), root, applied) == "recorded-prediction"
    with pytest.raises(ValueError, match="nonempty"):
        audit.reforecast_actual(Recorder(), root, [])


def test_durable_journal_and_failure_report_retain_nine_attempts(tmp_path):
    output = tmp_path / "out"
    directory = output / "root-119/winner"
    audit.save_arrays(
        directory / "frame-01.npz",
        applied=np.ones(14),
        requested=np.ones(14),
        rgb=np.zeros((8, 8, 3)),
        state=np.zeros(2),
        mask=np.ones(2, bool),
        timestamp=0.05,
    )
    records = audit.planned_records()
    records[0]["status"] = "running"
    audit.write(output / "report.json", {"records": records, "status": "running"})
    assert audit.finalize(output, -9, True) == 2
    report = json.loads((output / "report.json").read_text())
    assert len(report["records"]) == 9 and report["status"] == "incomplete"
    assert report["records"][0]["durable_prefix_frames"] == 1
    assert report["records"][0]["status"] == "interrupted"
    assert all(x["status"] == "not_started_budget" for x in report["records"][1:])
    assert not list(directory.glob("*.tmp"))


def test_json_handles_scorer_numpy_metadata_without_fabrication(tmp_path):
    path = tmp_path / "state.json"
    audit.write(path, {"initial": {"object_position": np.array([1.0, 2.0, 3.0])}})
    assert json.loads(path.read_text())["initial"]["object_position"] == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("durable", [False, True])
def test_timeout_before_frame_vs_after_durable_ack(tmp_path, durable):
    directory = tmp_path / "root-119/winner"
    audit.write(
        directory / "command.json", {"index": 1, "status": "pending", "requested": [0.0] * 14}
    )
    if durable:
        audit.save_arrays(
            directory / "frame-01.npz",
            applied=np.zeros(14),
            requested=np.zeros(14),
            rgb=np.zeros((8, 8, 3), np.uint8),
            state=np.zeros(2),
            mask=np.ones(2, bool),
            timestamp=0.05,
        )
    records = audit.planned_records()
    records[0].update(status="running", executed_steps=0)
    audit.write(tmp_path / "report.json", {"status": "running", "records": records})
    assert audit.finalize(tmp_path, -9, True) == 2
    row = json.loads((tmp_path / "report.json").read_text())["records"][0]
    assert row["executed_steps"] == int(durable)
    assert row["possible_unobserved_execution"] == (not durable)
    assert row["metrics"]["16"]["status"] == "unavailable"


def test_gapped_journal_never_counts_later_frame_as_confirmed(tmp_path):
    audit.save_arrays(tmp_path / "frame-02.npz", applied=np.zeros(14))
    count, unknown, issues = audit.durable_prefix(tmp_path)
    assert count == 0 and unknown and "gapped" in issues[0]


def test_clock_goal_identity_and_nullable_scores_are_checked():
    import copy

    trace = fake_trace()
    for key, value in [
        ("observation_timestamp", 5.96),
        ("execution_timestamp", 6.05),
        ("goal_image_sha256", "other"),
        ("waypoint_advanced", True),
    ]:
        actual = copy.deepcopy(trace)
        actual[key] = value
        with pytest.raises(ValueError, match=key):
            audit.assert_trace_match(trace, actual, acknowledged=True)
    audit.assert_score_match(
        {"success": False, "optional": None, "distance": 0.2},
        {"success": False, "optional": None, "distance": 0.2},
    )
    with pytest.raises(ValueError, match="fields"):
        audit.assert_score_match({"success": False}, {})
    with pytest.raises(ValueError, match="optional"):
        audit.assert_score_match({"optional": None}, {"optional": 0})


def test_dataclass_threshold_serialization_preserves_in_memory_type(tmp_path):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Thresholds:
        radius: float = 0.04

    thresholds = Thresholds()
    audit.write(tmp_path / "scorer.json", {"thresholds": thresholds})
    assert json.loads((tmp_path / "scorer.json").read_text())["thresholds"] == {"radius": 0.04}
    assert isinstance(thresholds, Thresholds)
