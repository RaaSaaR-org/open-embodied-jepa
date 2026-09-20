"""Bounded synthetic runner evidence, including split isolation and budget failures."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("pyarrow")
pytest.importorskip("pandas")
pytest.importorskip("PIL")

from embodied_jepa.contracts import ContractError  # noqa: E402
from embodied_jepa.data import DatasetStore, deterministic_fixture  # noqa: E402
from embodied_jepa.training import EpisodeCache, train  # noqa: E402


@pytest.fixture
def corpus(tmp_path):
    fixture = deterministic_fixture()
    store = DatasetStore.create(
        tmp_path / "corpus",
        fps=20,
        state_schema=fixture.state_schema,
        action_manifest={"physical_scale": [0.01] * 14, "software_fixture_only": True},
        provenance={"source": "procedural unit-test fixture", "license": "Apache-2.0"},
        robot_type="software_fixture",
    )
    for index in range(6):
        store.write_episode(
            replace(
                fixture,
                episode_id=f"episode-{index}",
                session_id=f"session-{index}",
                observations={"onboard_rgb": np.roll(fixture.observations["head"], index, axis=2)},
                # This runner fixture supplies every procedural state field;
                # missing-state behavior is covered by backend contract tests.
                state_mask=np.ones_like(fixture.state_mask),
            )
        )
    store.freeze_splits(seed=3)
    return store


@pytest.mark.parametrize("backend", ["native_jepa", "leworldmodel", "sensor_wm"])
def test_training_best_latest_curves_and_no_test_decoding(corpus, tmp_path, monkeypatch, backend):
    if backend == "leworldmodel":
        pytest.importorskip("transformers")
        if not Path("third_party/le-wm/jepa.py").exists():
            pytest.skip("optional upstream source absent")
    original = DatasetStore.read_episode
    reads = []
    excluded = set(corpus.manifest["splits"]["test"] + corpus.manifest["splits"]["holdout"])

    def recorded_read(self, episode_id):
        assert episode_id not in excluded
        reads.append(episode_id)
        return original(self, episode_id)

    monkeypatch.setattr(DatasetStore, "read_episode", recorded_read)
    output = tmp_path / f"{backend}.pt"
    report = train(
        corpus.root,
        backend,
        output,
        steps=3,
        batch_size=2,
        horizon=2,
        validation_every=2,
        validation_batches=1,
        max_seconds=60,
        model_config={"hidden_dim": 32},
    )
    assert report["status"] == "completed"
    assert report["completed_steps"] == 3
    assert report["best_step"] in (0, 2, 3)
    assert report["best_checkpoint_exists"] and report["latest_checkpoint_exists"]
    assert len(reads) == len(set(reads))  # decoded once, independent of train/validation steps
    assert set(reads) == set(corpus.manifest["splits"]["train"] + corpus.manifest["splits"]["val"])
    assert not report["test_samples_loaded"]
    assert report["provenance"]["dataset_hash"] == corpus.manifest_hash
    events = [
        json.loads(line) for line in Path(report["artifacts"]["curves"]).read_text().splitlines()
    ]
    assert [event["step"] for event in events if event["kind"] == "train"] == [1, 2, 3]
    validation = [event for event in events if event["kind"] == "validation"]
    assert [event["step"] for event in validation] == [0, 2, 3]
    assert report["best_validation_mse"] == min(event["prediction_mse"] for event in validation)
    assert validation[0]["horizons"]["8"]["status"] == "unavailable"
    assert len({event["horizons"]["2"]["cohort_sha256"] for event in validation}) == 1
    import torch

    best = torch.load(output, weights_only=True)
    latest = torch.load(Path(report["artifacts"]["latest"]), weights_only=True)
    assert best["updates"] == report["best_step"]
    assert latest["updates"] == latest["metadata"]["runner_state"]["step"] == 3
    assert latest["metadata"]["runner_state"]["sampler_rng"] == report["sampler_rng"]
    with pytest.raises(FileExistsError, match="overwrite"):
        train(corpus.root, backend, output, steps=1)


def test_sampler_is_deterministic_and_memory_preflight_is_bounded(corpus):
    cache = EpisodeCache(corpus, memory_limit_bytes=16 * 1024**3)
    first = cache.sample("train", 2, 8, np.random.default_rng(7))
    second = cache.sample("train", 2, 8, np.random.default_rng(7))
    assert first.episode_ids == second.episode_ids
    np.testing.assert_array_equal(first.timestamps, second.timestamps)
    with pytest.raises(ContractError, match="memory budget"):
        EpisodeCache(corpus, memory_limit_bytes=1)


def test_optional_model_normalization_uses_every_training_transition_only(
    corpus, tmp_path, monkeypatch
):
    from embodied_jepa.models import NativeJEPA

    observed = []
    declared = []

    def fit(self, batches, *, training_episode_ids):
        declared.extend(training_episode_ids)
        for batch in batches:
            assert set(batch.episode_ids) <= set(training_episode_ids)
            observed.extend(
                (name, float(timestamp))
                for name, times in zip(batch.episode_ids, batch.timestamps, strict=True)
                for timestamp in times[:-1]
            )

    monkeypatch.setattr(NativeJEPA, "fit_normalization", fit, raising=False)
    report = train(
        corpus.root,
        "native_jepa",
        tmp_path / "normalized.pt",
        steps=1,
        batch_size=2,
        horizon=2,
        validation_batches=1,
        model_config={"hidden_dim": 32},
    )
    expected = [
        (name, float(timestamp))
        for name in corpus.manifest["splits"]["train"]
        for timestamp in corpus.read_episode(name).timestamps[:-1]
    ]
    assert observed == expected
    assert declared == corpus.manifest["splits"]["train"]
    assert report["normalization"]["transitions"] == len(expected)


def test_time_budget_preserves_partial_report_without_claiming_completion(corpus, tmp_path):
    report = train(corpus.root, "native_jepa", tmp_path / "timeout.pt", max_seconds=1e-12)
    assert report["status"] == "time_budget"
    assert report["completed_steps"] == 0
    assert report["best_step"] is None
    assert not report["best_checkpoint_exists"]
    saved = json.loads((tmp_path / "timeout.run.json").read_text())
    assert saved["status"] == "time_budget"


def test_normalization_failure_and_unwritable_checkpoint_preserve_original_report(
    corpus, tmp_path, monkeypatch
):
    from embodied_jepa.models import NativeJEPA

    def fail_fit(*args, **kwargs):
        raise ContractError("normalization failure")

    def fail_save(*args, **kwargs):
        raise ContractError("cannot save unfitted model")

    monkeypatch.setattr(NativeJEPA, "fit_normalization", fail_fit, raising=False)
    monkeypatch.setattr(NativeJEPA, "save", fail_save)
    with pytest.raises(ContractError, match="normalization failure"):
        train(corpus.root, "native_jepa", tmp_path / "unfitted.pt", horizon=2)
    saved = json.loads((tmp_path / "unfitted.run.json").read_text())
    assert saved["status"] == "failed"
    assert "normalization failure" in saved["error"]
    assert "cannot save unfitted" in saved["latest_checkpoint_error"]
    assert not saved["latest_checkpoint_exists"]


def test_unavailable_horizon_failure_is_preserved(corpus, tmp_path):
    with pytest.raises(ContractError, match="no windows"):
        train(corpus.root, "native_jepa", tmp_path / "failed.pt", horizon=8)
    saved = json.loads((tmp_path / "failed.run.json").read_text())
    assert saved["status"] == "failed"
    assert saved["completed_steps"] == 0
    assert not saved["latest_checkpoint_exists"]


def test_nonfinite_training_failure_does_not_disappear(corpus, tmp_path, monkeypatch):
    from embodied_jepa.models import NativeJEPA

    monkeypatch.setattr(NativeJEPA, "train_step", lambda *args: {"loss": float("nan")})
    with pytest.raises(ContractError, match="non-finite"):
        train(
            corpus.root,
            "native_jepa",
            tmp_path / "nonfinite.pt",
            steps=1,
            batch_size=2,
            horizon=2,
            validation_batches=1,
        )
    saved = json.loads((tmp_path / "nonfinite.run.json").read_text())
    assert saved["status"] == "failed"
    assert "non-finite" in saved["error"]
    assert saved["completed_steps"] == 0
    assert saved["best_step"] == 0


def test_unfrozen_or_leaking_splits_rejected(corpus):
    original = corpus.manifest["splits"]
    corpus.manifest["splits"] = None
    with pytest.raises(ContractError, match="sealed"):
        EpisodeCache(corpus, memory_limit_bytes=16 * 1024**3)
    corpus.manifest["splits"] = original | {"val": original["train"]}
    with pytest.raises(ContractError, match="overlap"):
        EpisodeCache(corpus, memory_limit_bytes=16 * 1024**3)


def test_noncollapsed_selector_rejects_tiny_loss_and_ranks_relative_error(
    corpus, tmp_path, monkeypatch
):
    from embodied_jepa.models import NativeJEPA

    # Initial collapsed state has the smallest raw MSE. Step two has smaller raw
    # MSE than step one but predicts worse relative to its own persistence control.
    def diagnostics(self, batch):
        mse, persistence, collapsed, std = (
            (1e-10, 1e-8, 1.0, 0.0001),
            (2.0, 4.0, 0.0, 0.8),
            (0.1, 0.05, 0.0, 0.7),
        )[self.updates]
        return {
            "prediction_mse": mse,
            "persistence_mse": persistence,
            "metric_definition_version": 2.0,
            "online_collapsed_fraction": collapsed,
            "online_latent_std_mean": std,
            "target_collapsed_fraction": collapsed,
            "target_latent_std_mean": std,
        }

    monkeypatch.setattr(NativeJEPA, "diagnostics", diagnostics)
    report = train(
        corpus.root,
        "native_jepa",
        tmp_path / "selected.pt",
        steps=2,
        batch_size=2,
        horizon=4,
        validation_every=1,
        validation_batches=1,
        selection="noncollapsed_relative",
    )
    assert report["status"] == "completed"
    assert report["best_step"] == 1
    assert report["best_selection_score"] == 0.5
    assert report["best_validation_mse"] == 2.0
    baseline, first, last = [event["selection"] for event in report["validation"]]
    assert not baseline["eligible"] and not baseline["selected"]
    assert baseline["rejection_reasons"] == [
        "online_collapsed_fraction_above_0.05",
        "online_latent_std_mean_below_0.1",
        "target_collapsed_fraction_above_0.05",
        "target_latent_std_mean_below_0.1",
    ]
    assert first["selected"]
    assert last["eligible"] and not last["selected"]
    assert last["score"] == 2.0


def test_no_eligible_checkpoint_reports_selection_failure_and_preserves_latest(
    corpus, tmp_path, monkeypatch
):
    from embodied_jepa.models import NativeJEPA

    monkeypatch.setattr(
        NativeJEPA,
        "diagnostics",
        lambda *args: {
            "prediction_mse": 1e-12,
            "persistence_mse": 0.0,
            "metric_definition_version": 2.0,
            "online_collapsed_fraction": 0.5,
            "online_latent_std_mean": 0.09,
            "target_collapsed_fraction": 0.5,
            "target_latent_std_mean": 0.09,
        },
    )
    report = train(
        corpus.root,
        "native_jepa",
        tmp_path / "ineligible.pt",
        steps=1,
        batch_size=2,
        horizon=4,
        validation_every=1,
        validation_batches=1,
        selection="noncollapsed_relative",
    )
    assert report["status"] == "selection_failed"
    assert report["best_step"] is None and report["best_selection_score"] is None
    assert not report["best_checkpoint_exists"]
    assert report["latest_checkpoint_exists"] and report["completed_steps"] == 1
    assert all(not event["selection"]["eligible"] for event in report["validation"])


def test_selector_requires_target_diversity_even_when_online_is_healthy():
    from embodied_jepa.training import selection_decision

    metrics = {
        "metric_definition_version": 2.0,
        "prediction_mse": 1e-9,
        "persistence_mse": 0.1,
        "online_collapsed_fraction": 0.0,
        "online_latent_std_mean": 1.0,
        "target_collapsed_fraction": 1.0,
        "target_latent_std_mean": 0.001,
    }
    decision = selection_decision(
        {"4": {"status": "measured", "metrics": metrics}}, "noncollapsed_relative", 4
    )
    assert not decision["eligible"]
    assert decision["rejection_reasons"] == [
        "target_collapsed_fraction_above_0.05",
        "target_latent_std_mean_below_0.1",
    ]
    legacy = metrics | {"metric_definition_version": 1.0}
    rejected = selection_decision(
        {"4": {"status": "measured", "metrics": legacy}}, "noncollapsed_relative", 4
    )
    assert not rejected["eligible"]
    assert rejected["rejection_reasons"] == ["unsupported_metric_definition_version"]


def test_budget_clock_counts_host_suspend_and_survives_clock_rollback():
    from embodied_jepa.training import BudgetReached, RunClock

    values = {"wall": 1000.0, "monotonic": 100.0, "cpu": 10.0}
    clock = RunClock(
        wall_clock=lambda: values["wall"],
        monotonic_clock=lambda: values["monotonic"],
        cpu_clock=lambda: values["cpu"],
    )
    values.update(wall=1065.0, monotonic=106.0, cpu=12.0)
    snapshot = clock.snapshot()
    assert snapshot == {
        "elapsed_seconds": 65.0,
        "wall_clock_elapsed_seconds": 65.0,
        "monotonic_elapsed_seconds": 6.0,
        "process_cpu_seconds": 2.0,
    }
    with pytest.raises(BudgetReached, match="time_budget"):
        clock.check(60)
    # A backward wall-clock correction cannot extend the monotonic deadline.
    values.update(wall=900.0, monotonic=170.0, cpu=13.0)
    assert clock.elapsed() == 70.0
    assert clock.snapshot()["wall_clock_elapsed_seconds"] == -100.0
    with pytest.raises(BudgetReached, match="time_budget"):
        clock.check(60)


def test_suspended_host_budget_exits_before_loading_data(corpus, tmp_path, monkeypatch):
    from embodied_jepa import training

    values = {"wall": 100.0, "monotonic": 0.0, "cpu": 0.0}
    clock = training.RunClock(
        wall_clock=lambda: values["wall"],
        monotonic_clock=lambda: values["monotonic"],
        cpu_clock=lambda: values["cpu"],
    )
    values["wall"] = 165.0  # Simulate sleep before the first budget check.
    monkeypatch.setattr(training, "RunClock", lambda: clock)
    report = training.train(corpus.root, "native_jepa", tmp_path / "suspended.pt", max_seconds=60)
    assert report["status"] == "time_budget" and report["completed_steps"] == 0
    assert report["elapsed_seconds"] == report["wall_clock_elapsed_seconds"] == 65.0
    assert report["monotonic_elapsed_seconds"] == report["process_cpu_seconds"] == 0.0
    assert "cache" not in report
