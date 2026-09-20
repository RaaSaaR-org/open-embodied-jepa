import copy
from dataclasses import asdict

import pytest

from embodied_jepa.benchmark import summarize, validate_results, wilson
from embodied_jepa.planning import CEMConfig


def records():
    episodes = [
        {
            "seed": 1,
            "score": {
                "success": True,
                "reach": True,
                "grasp": None,
                "transport": None,
                "place": None,
                "release": None,
                "missing_stage_reason": "reach-only task",
            },
            "termination_reason": "success",
            "trace": [],
            "executed_steps": 0,
            "replans": 0,
            "wall_seconds": 0.1,
            "collision_violations": None,
            "collision_missing_reason": "classifier unavailable",
        },
        {
            "seed": 2,
            "score": {
                "success": False,
                "reach": False,
                "grasp": None,
                "transport": None,
                "place": None,
                "release": None,
                "missing_stage_reason": "reach-only task",
            },
            "termination_reason": "runtime_error",
            "trace": [],
            "executed_steps": 0,
            "replans": 0,
            "wall_seconds": 0.1,
            "collision_violations": None,
            "collision_missing_reason": "classifier unavailable",
        },
    ]
    manifest = dict(
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
        planner=asdict(CEMConfig()),
        task_version="tabletop_proxy_v0",
        train_seed=0,
        evaluation_seeds=[1, 2],
        timeout_seconds=5.0,
        model_diagnostics=None,
        model_diagnostics_reason="stored in fixture training report",
    )
    # JSON manifests contain lists, even when the originating dataclass uses tuples.
    for key in ("lower_bounds", "upper_bounds"):
        manifest["planner"][key] = list(manifest["planner"][key])
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
