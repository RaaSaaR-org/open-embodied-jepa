"""Independent checks of the offline screen's leakage and comparison semantics."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "apple_goal_alignment", Path(__file__).parents[1] / "scripts/apple_goal_alignment.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def manifest():
    def original(name, session):
        return dict(
            episode_id=name,
            session_id=session,
            length=25,
            metadata={"phase_labels": [p for p in audit.PHASES for _ in range(4)]},
        )

    branch = dict(
        episode_id="p-branch",
        session_id="train-session",
        length=17,
        metadata=dict(
            parent_episode_id="p",
            phase_labels=["orient"] * 16,
            root_id="p-orient",
            branch="hold",
        ),
    )
    return dict(
        episodes=[
            original("p", "train-session"),
            branch,
            original("v", "val-session"),
            # TEST metadata deliberately has no decodable phase payload.
            dict(episode_id="t", session_id="test-session", metadata={}),
        ],
        splits=dict(train=["p", "p-branch"], val=["v"], test=["t"]),
    )


def test_sampling_uses_transition_labels_and_keeps_branch_endpoint_16():
    table = audit.sample_table(manifest())
    assert table["missing"] == []
    assert len(table["samples"]) == 50
    assert {r["episode_id"] for r in table["samples"]} == {"p", "p-branch", "v"}
    assert [r["frame"] for r in table["samples"] if r["episode_id"] == "p-branch"] == [8, 16]
    assert max(r["frame"] for r in table["samples"] if r["episode_id"] == "v") == 23
    assert table == audit.sample_table(manifest())


@pytest.mark.parametrize("fault", ["split", "session", "duplicate"])
def test_sampling_rejects_family_leakage_and_duplicate_membership(fault):
    value = manifest()
    if fault == "split":
        value["episodes"][1]["metadata"]["parent_episode_id"] = "v"
    elif fault == "session":
        value["episodes"][1]["session_id"] = "val-session"
    else:
        value["splits"]["val"].append("p")
    with pytest.raises(ValueError):
        audit.sample_table(value)


def test_fields_are_resolved_by_name_and_require_radians():
    schema = SimpleNamespace(
        names=("unrelated", *reversed(audit.FIELDS)), units=("m",) + ("rad",) * 7
    )
    assert audit.resolve_fields(schema) == tuple(range(7, 0, -1))
    schema.units = ("m",) + ("rad/s",) * 7
    with pytest.raises(ValueError):
        audit.resolve_fields(schema)


def test_weights_do_not_overweight_branch_rich_parents():
    rows = [dict(parent="a", source="original")]
    rows += [dict(parent="a", source="branch") for _ in range(8)]
    rows += [dict(parent="b", source="original") for _ in range(3)]
    weights = audit.frame_weights(rows)
    assert weights.sum() == pytest.approx(1)
    assert weights[0] == pytest.approx(0.25)
    assert weights[1:9].sum() == pytest.approx(0.25)
    assert weights[9:].sum() == pytest.approx(0.5)


def test_nearest_query_uses_images_and_deterministic_reference_ties():
    features = np.array([[0.0, 0.0], [3.0, 0.0]])
    reference = np.array([[-1.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    labels = np.array([[10.0] * 7, [20.0] * 7, [30.0] * 7])
    predicted, variance, indices = audit.nearest(features, reference, labels, chunk=1)
    np.testing.assert_array_equal(predicted, labels[[0, 2]])
    assert indices == [0, 2]
    assert np.isfinite(variance).all()


def test_pair_order_uses_measured_goal_labels_and_reports_prediction_ties():
    measured = np.array([[0.0] * 7, [1.0] * 7, [1.01] * 7])
    costs = dict(
        correct=np.array([0.0, 1.0, 2.0]), wrong=np.array([2.0, 1.0, 0.0]), tied=np.ones(3)
    )
    pairs = audit.ranking_pairs(measured, np.zeros(7), costs)
    assert pairs[0]["scores"] == {"correct": 1.0, "wrong": 0.0, "tied": 0.5}
    assert pairs[2]["eligible"] is False
    assert pairs[2]["exclusion"] == "low_arm_separation"


def test_ranking_aggregates_goals_before_roots_and_parents():
    cases = []
    for root, parent, score, count in (
        ("a1", "a", 1.0, 20),
        ("a1", "a", 0.0, 1),
        ("b1", "b", 0.0, 1),
    ):
        cases.append(
            dict(
                root_id=root,
                parent=parent,
                goal=len(cases),
                pairs=[dict(eligible=True, scores={"method": score}) for _ in range(count)],
            )
        )
    result = audit.aggregate_rankings(cases, ["a", "b", "c"])
    assert result["parents"]["method"] == {"a": 0.5, "b": 0.0}
    assert result["scores"]["method"] == pytest.approx(0.25)
    assert not result["sufficient_coverage"]


def comparisons():
    errors = {
        "mean": dict(mse=1.0, parents=dict(a=1.0, b=1.0, c=1.0)),
        "nn": dict(mse=0.5, parents=dict(a=0.5, b=0.5, c=0.5)),
    }
    ranking = dict(
        sufficient_coverage=True,
        parents=dict(pixel_pred=dict(a=0.5, b=0.5, c=0.5), nn_pred=dict(a=0.6, b=0.6, c=0.6)),
        scores=dict(pixel_pred=0.5, nn_pred=0.6),
    )
    return errors, ranking


def test_nn_can_pass_when_neural_candidate_is_absent():
    errors, ranking = comparisons()
    result = audit.route_decisions(errors, ranking, complete=True)
    assert result["nn"]["status"] == "passed"
    assert result["neural"]["status"] == "unverified"
    assert not result["neural_preferred_replacement"]


def test_one_degraded_parent_blocks_otherwise_positive_mean():
    errors, ranking = comparisons()
    ranking["parents"]["nn_pred"] = dict(a=1.0, b=1.0, c=0.49)
    assert audit.route_decisions(errors, ranking, complete=True)["nn"]["status"] == "failed"


def test_incomplete_coverage_cannot_pass_or_masquerade_as_numerical_failure():
    errors, ranking = comparisons()
    assert audit.route_decisions(errors, ranking, complete=False)["nn"]["status"] == "unverified"
    ranking["sufficient_coverage"] = False
    assert audit.route_decisions(errors, ranking, complete=True)["nn"]["status"] == "unverified"


def test_supervisor_rejects_late_zero_exit():
    child = SimpleNamespace(poll=lambda: 0, wait=lambda: 0)
    assert audit.supervise(child, budget=5, clock=lambda: 6) == (0, True)


def test_supervisor_kills_live_worker_at_deadline():
    signals = []
    child = SimpleNamespace(pid=123, poll=lambda: None, wait=lambda: -9)
    assert audit.supervise(
        child, budget=5, clock=lambda: 5, kill=lambda *args: signals.append(args)
    ) == (-9, True)
    assert signals == [(123, 9)]


def test_finalize_preserves_unverified_result_after_malformed_worker_report(tmp_path, monkeypatch):
    folder = tmp_path / "evaluate"
    folder.mkdir()
    (folder / "report.json").write_text("{unfinished")
    args = SimpleNamespace(stage="evaluate", output=tmp_path, dataset=tmp_path)
    monkeypatch.setattr(audit, "identities", lambda _: {})
    monkeypatch.setattr(audit, "verify_payloads", lambda _: None)
    monkeypatch.setattr(audit, "elapsed", lambda: 1.0)
    assert audit.finalize(args, 2, False, {}) == 2
    report = audit.json.loads((folder / "report.json").read_text())
    assert report["status"] == "incomplete"
    assert report["decisions"]["nn"]["status"] == "unverified"
    assert (folder / "seal.json").exists()


def test_finalization_integrity_work_counts_against_wall_limit(tmp_path, monkeypatch):
    folder = tmp_path / "evaluate"
    folder.mkdir()
    audit.write(folder / "report.json", dict(status="completed", integrity_verified_after=True))
    args = SimpleNamespace(stage="evaluate", output=tmp_path, dataset=tmp_path)
    now = [1.0]

    def slow_identity(_):
        now[0] = 61.0
        return {}

    monkeypatch.setattr(audit, "identities", slow_identity)
    monkeypatch.setattr(audit, "verify_payloads", lambda _: None)
    monkeypatch.setattr(audit, "elapsed", lambda: now[0])
    assert audit.finalize(args, 0, False, {}) == 2
    report = audit.json.loads((folder / "report.json").read_text())
    assert report["status"] == "incomplete"
    assert report["decisions"]["nn"]["status"] == "unverified"


def test_zero_bootstrap_denominator_is_unavailable_not_nonfinite():
    parents = ["a", "b", "c"]
    ranking = dict(
        sufficient_coverage=True,
        parents={"pixel_pred": dict.fromkeys(parents, 0.5), "nn_pred": dict.fromkeys(parents, 0.8)},
        scores={"pixel_pred": 0.5, "nn_pred": 0.8},
    )
    errors = dict(
        mean=dict(mse=2 / 3, parents=dict(a=0.0, b=1.0, c=1.0)),
        nn=dict(mse=0.1, parents=dict(a=0.0, b=0.15, c=0.15)),
    )
    result = audit.route_decisions(errors, ranking, complete=True)["nn"]
    assert result["error_reduction_ci95"] is None
    assert result["error_reduction_bootstrap_invalid_draws"] > 0
    audit.json.dumps(result, allow_nan=False)


def test_uncertainty_reports_actual_weighted_coverage():
    rows = [dict(parent="a", source="original", episode_id="a", frame=i) for i in range(2)] + [
        dict(parent="a", source="branch", episode_id="b", frame=i) for i in range(6)
    ]
    y = np.zeros((8, 7))
    variance = np.arange(1, 9)[:, None] * np.ones((8, 7))
    errors = audit.encoder_errors(rows, y, {"nn": np.ones_like(y)}, {"nn": variance}, np.ones(7))
    retained = errors["nn"]["risk_coverage"]["0.5"]["retained_by_parent"]["a"]
    assert retained["count"] == 4
    assert retained["weighted_mass"] == pytest.approx(2 / 3)


def test_neural_image_only_estimate_has_finite_bounded_variance():
    torch = pytest.importorskip("torch")
    head = audit.head_factory()
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
        head[-1].bias[7:] = 100
    mean, variance = audit.estimate_neural(head, np.zeros((2, 1728), np.float32))
    assert mean.shape == variance.shape == (2, 7)
    np.testing.assert_allclose(variance, np.exp(3), rtol=1e-6)


def test_producer_chain_detects_artifact_and_source_changes(tmp_path):
    folder = tmp_path / "prepare"
    folder.mkdir()
    identity = dict(script="fixed", dataset="fixed")
    audit.write(folder / "registration.json", dict(identities=identity))
    audit.write(folder / "report.json", dict(status="completed", integrity_verified_after=True))
    (folder / "bank.npz").write_bytes(b"canonical")
    audit.seal_stage(folder)
    args = SimpleNamespace(output=tmp_path)
    audit.verify_producer(args, "prepare", identity)
    with pytest.raises(ValueError, match="different frozen"):
        audit.verify_producer(args, "prepare", identity | {"script": "edited"})
    (folder / "bank.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed prepare"):
        audit.verify_producer(args, "prepare", identity)


def test_malformed_optional_fit_isolated(tmp_path):
    (tmp_path / "fit").mkdir()
    (tmp_path / "fit/seal.json").write_text("{broken")
    status = audit.optional_fit_status(SimpleNamespace(output=tmp_path), {})
    assert status["eligible"] is False
    assert "JSONDecodeError" in status["reason"]


def test_nn_exact_zero_spread_is_preserved():
    _, variance, _ = audit.nearest(np.zeros((1, 2)), np.zeros((8, 2)), np.ones((8, 7)))
    assert np.count_nonzero(variance) == 0


@pytest.mark.parametrize("seal_crosses_deadline", [False, True])
def test_finalization_success_and_post_seal_deadline(tmp_path, monkeypatch, seal_crosses_deadline):
    folder = tmp_path / "evaluate"
    folder.mkdir()
    audit.write(folder / "report.json", dict(status="completed", integrity_verified_after=True))
    args = SimpleNamespace(stage="evaluate", output=tmp_path, dataset=tmp_path)
    now = [1.0]
    real_seal = audit.seal_stage

    def seal(path):
        real_seal(path)
        if seal_crosses_deadline:
            now[0] = 61.0

    monkeypatch.setattr(audit, "identities", lambda _: {})
    monkeypatch.setattr(audit, "verify_payloads", lambda _: None)
    monkeypatch.setattr(audit, "elapsed", lambda: now[0])
    monkeypatch.setattr(audit, "seal_stage", seal)
    assert audit.finalize(args, 0, False, {}) == (2 if seal_crosses_deadline else 0)
    report = audit.json.loads((folder / "report.json").read_text())
    assert report["status"] == ("incomplete" if seal_crosses_deadline else "completed")
    sealed = audit.json.loads((folder / "seal.json").read_text())
    assert sealed["artifacts"]["report.json"] == audit.digest(folder / "report.json")
