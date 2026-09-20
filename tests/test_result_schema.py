"""Reject scientifically misleading records, not merely missing top-level keys."""

import copy
from dataclasses import asdict

import pytest

from embodied_jepa.planning import CEMConfig
from embodied_jepa.result_schema import validate_results


@pytest.fixture
def records():
    planner = asdict(CEMConfig(samples=8, iterations=2, elites=2))
    planner["lower_bounds"] = list(planner["lower_bounds"])
    planner["upper_bounds"] = list(planner["upper_bounds"])
    run = dict(
        schema_version=1,
        mode="common",
        run_id="fixture",
        timestamp="2026-09-20T00:00:00Z",
        source_revision="a" * 40,
        backend="native_jepa",
        checkpoint_hash="b" * 64,
        dataset_hash="c" * 64,
        split_hash="d" * 64,
        action_hash="e" * 64,
        environment=dict(
            platform="test-host",
            python="3.12.13",
            device="cpu",
            engine="mujoco",
            mujoco="3.13.0",
            torch="2.14.0",
            numpy="2.5.3",
        ),
        planner=planner,
        task_version="tabletop_proxy_v0",
        train_seed=0,
        evaluation_seeds=[1],
        timeout_seconds=5.0,
        max_steps=2,
        model_diagnostics=None,
        model_diagnostics_reason="separate training report",
    )
    episode = dict(
        seed=1,
        score=dict(
            success=False,
            reach=False,
            grasp=None,
            transport=None,
            place=None,
            release=None,
            missing_stage_reason="reach-only task",
        ),
        termination_reason="stopped",
        executed_steps=1,
        replans=1,
        wall_seconds=0.1,
        collision_violations=None,
        collision_missing_reason="classifier unavailable",
        trace=[
            dict(
                step=0,
                executed=True,
                status="applied",
                action=[0.0] * 14,
                planning_seconds=0.01,
                control_seconds=0.02,
                candidate_evaluations=16,
                planned_cost=0.3,
                observation_timestamp=0.0,
                execution_timestamp=0.05,
            )
        ],
    )
    return run, [episode]


def test_model_and_explicit_control_records_pass(records):
    run, episodes = records
    validate_results(run, episodes)


def test_control_policy_omits_planning_measurements(records):
    run, episodes = records
    run.update(
        backend="control:hold",
        checkpoint_hash=None,
        train_seed=None,
        checkpoint_missing_reason="control policy",
        train_seed_missing_reason="control policy",
    )
    episodes[0]["replans"] = 0
    for key in ("planning_seconds", "candidate_evaluations", "planned_cost"):
        episodes[0]["trace"][0].pop(key)
    validate_results(run, episodes)


def test_in_memory_dataclass_budget_and_archived_training_reference(records):
    run, episodes = records
    run["planner"] = asdict(CEMConfig(samples=8, iterations=2, elites=2))
    run["training_report"] = {"path": "training_report.json", "sha256": "a" * 64}
    validate_results(run, episodes)
    run["training_report"]["sha256"] = "bad"
    with pytest.raises(ValueError, match="training_report.sha256"):
        validate_results(run, episodes)


@pytest.mark.parametrize(
    "path,value",
    [
        (("environment",), "fixture"),
        (("environment", "device"), "cuda"),
        (("planner",), "fixture"),
        (("planner", "samples"), True),
        (("planner", "elites"), 99),
        (("planner", "lower_bounds"), [0.0] * 13),
        (("planner", "action_penalty"), True),
        (("checkpoint_hash",), "not-a-hash"),
        (("source_revision",), "main"),
        (("timestamp",), "2026-09-20T00:00:00"),
        (("timestamp",), "yesterday"),
        (("train_seed",), False),
        (("backend",), "unknown"),
        (("evaluation_seeds",), [1, 1]),
        (("timeout_seconds",), float("nan")),
        (("schema_version",), True),
    ],
)
def test_nested_metadata_and_types_are_validated(records, path, value):
    run, episodes = records
    target = run
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_results(run, episodes)


@pytest.mark.parametrize(
    "path,value",
    [
        (("score", "success"), 1),
        (("score", "reach"), "yes"),
        (("score", "missing_stage_reason"), ""),
        (("termination_reason",), "abandoned"),
        (("termination_reason",), "success"),
        (("executed_steps",), 2),
        (("replans",), 0),
        (("wall_seconds",), -1.0),
        (("collision_missing_reason",), None),
        (("trace", 0, "action"), [2.0] * 14),
        (("trace", 0, "action"), [False] * 14),
        (("trace", 0, "planning_seconds"), float("inf")),
        (("trace", 0, "control_seconds"), 5.01),
        (("trace", 0, "candidate_evaluations"), 8),
        (("trace", 0, "executed"), None),
        (("trace", 0, "status"), "rejected"),
        (("trace", 0, "step"), 2),
    ],
)
def test_outcomes_and_trace_invariants_are_validated(records, path, value):
    run, episodes = records
    target = episodes[0]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_results(run, episodes)


def test_failed_and_uncertain_executions_remain_in_denominator(records):
    run, episodes = records
    run["evaluation_seeds"].append(2)
    failure = copy.deepcopy(episodes[0])
    failure.update(
        seed=2,
        termination_reason="runtime_error",
        executed_steps=0,
        replans=0,
        trace=[
            dict(
                step=0,
                executed=None,
                execution_uncertain=True,
                stage="execute",
                error="transport connection lost",
            )
        ],
    )
    episodes.append(failure)
    validate_results(run, episodes)
    with pytest.raises(ValueError, match="all frozen evaluation seeds"):
        validate_results(run, episodes[:1])
    with pytest.raises(ValueError, match="all frozen evaluation seeds"):
        validate_results(run, list(reversed(episodes)))


def test_historical_missing_stage_measurements_are_not_silently_upgraded(records):
    run, episodes = records
    episodes[0]["score"].pop("grasp")
    with pytest.raises(ValueError, match="missing stage grasp"):
        validate_results(run, episodes)


def test_legacy_projection_omission_stays_valid_without_rewriting_archives(records):
    run, episodes = records
    run["planner"].pop("project_candidates")
    original = copy.deepcopy(run)
    validate_results(run, episodes)
    assert run == original
    run["planner"].pop("horizon")
    with pytest.raises(ValueError, match="complete resolved"):
        validate_results(run, episodes)


def test_projected_results_require_scored_and_executed_action_agreement(records):
    run, episodes = records
    run["planner"]["project_candidates"] = True
    trace = episodes[0]["trace"][0]
    trace.update(
        sampled_action=[0.5] * 14,
        projected_action=[0.0] * 14,
        requested_action=[0.0] * 14,
        projection_seconds=0.005,
        feasible_candidates=16,
    )
    validate_results(run, episodes)
    trace["requested_action"][0] = 0.25
    with pytest.raises(ValueError, match="model-scored"):
        validate_results(run, episodes)
