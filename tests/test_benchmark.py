import copy

import pytest

from embodied_jepa.benchmark import summarize, validate_results, wilson


def records():
    episodes = [
        {
            "seed": 1,
            "score": {"success": True, "reach": True},
            "termination_reason": "success",
            "trace": [],
            "executed_steps": 0,
        },
        {
            "seed": 2,
            "score": {"success": False},
            "termination_reason": "runtime_error",
            "trace": [],
            "executed_steps": 0,
        },
    ]
    manifest = {
        key: "fixture"
        for key in (
            "run_id",
            "timestamp",
            "source_revision",
            "backend",
            "checkpoint_hash",
            "dataset_hash",
            "split_hash",
            "action_hash",
            "environment",
            "planner",
            "task_version",
        )
    }
    manifest.update(schema_version=1, mode="common", train_seed=0, evaluation_seeds=[1, 2])
    return manifest, episodes


def test_failed_episodes_remain_in_success_denominator():
    _, episodes = records()
    summary = summarize(episodes)
    assert summary["success_rate"] == 0.5
    assert summary["episodes"] == 2
    assert summary["termination_reasons"]["runtime_error"] == 1
    assert summary["plan_latency_seconds"]["p50"] is None
    assert summary["plan_latency_seconds"]["missing_reason"]
    low, high = summary["wilson95"]
    assert 0 < low < 0.5 < high < 1


def test_result_schema_requires_all_frozen_seeds_and_consistent_traces():
    manifest, episodes = records()
    validate_results(manifest, episodes)
    with pytest.raises(ValueError, match="seeds"):
        validate_results(manifest, episodes[:1])
    broken = copy.deepcopy(episodes)
    broken[0]["executed_steps"] = 1
    with pytest.raises(ValueError, match="step count"):
        validate_results(manifest, broken)
    manifest.pop("checkpoint_hash")
    with pytest.raises(ValueError, match="missing run"):
        validate_results(manifest, episodes)


def test_uncertainty_does_not_report_certainty_from_one_success():
    low, high = wilson(1, 1)
    assert low < 0.25 and high == 1
    with pytest.raises(ValueError):
        wilson(0, 0)
