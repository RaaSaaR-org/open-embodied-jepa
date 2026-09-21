"""Synthetic packaging and saved-array gate checks; never decode the real corpus."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "aligned_builder", Path(__file__).parents[1] / "scripts/build_apple_aligned_sensor.py"
)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def manifest():
    return dict(
        episodes=[
            dict(episode_id="failed", session_id="s1", length=58, metadata={"success": False}),
            dict(episode_id="ordinary", session_id="s2", length=57, metadata={}),
            dict(
                episode_id="branch",
                session_id="s1",
                length=17,
                metadata={"parent_episode_id": "failed"},
            ),
            dict(episode_id="val", session_id="s3", length=57, metadata={}),
            dict(episode_id="test", session_id="s4", metadata={}),
        ],
        splits=dict(train=["ordinary", "branch", "failed"], val=["val"], test=["test"]),
    )


def test_pair_ledger_includes_failed_original_and_complete_nonoverlapping_pairs():
    ledger = build.pair_ledger(manifest(), expected_parents=2)
    assert [r["episode_id"] for r in ledger] == ["failed", "ordinary"]
    assert all(r["pairs"] == [[0, 28], [28, 56]] for r in ledger)
    assert build.pair_ledger(manifest(), expected_parents=2) == ledger


@pytest.mark.parametrize("fault", ["overlap", "duplicate", "too_short", "same_session", "count"])
def test_pair_roster_fails_closed(fault):
    value = manifest()
    if fault == "overlap":
        value["splits"]["val"].append("failed")
    elif fault == "duplicate":
        value["splits"]["train"].append("failed")
    elif fault == "too_short":
        value["episodes"][0]["length"] = 28
    elif fault == "same_session":
        value["episodes"][1]["session_id"] = "s1"
    else:
        value["splits"]["train"].remove("ordinary")
    with pytest.raises(ValueError):
        build.pair_ledger(value, expected_parents=2)


def test_scales_are_median_of_parent_medians_with_zeros_retained():
    scales, medians = build.reduce_scales(
        [
            dict(distances=[[0, 0], [2, 4], [10, 20]]),
            dict(distances=[[8, 12]] * 7),
        ]
    )
    assert medians == [[2, 4], [8, 12]]
    assert scales == dict(visual=5.0, pose=8.0)


@pytest.mark.parametrize(
    "distances", [[[0, 0], [0, 0], [9, 9]], [[float("nan"), 1]], [[1, -1]], []]
)
def test_degenerate_calibration_has_no_floor_or_positive_pair_fallback(distances):
    with pytest.raises(ValueError):
        build.reduce_scales([dict(distances=distances)])


def test_gate_requires_positive_delta_for_every_parent():
    ranking = dict(
        sufficient_coverage=True,
        parents={
            "pixel_pred": dict(a=0.5, b=0.5, c=0.5),
            "hybrid_pred": dict(a=0.8, b=0.8, c=0.49),
        },
    )
    assert build.hybrid_gate(ranking)["status"] == "failed"
    ranking["parents"]["hybrid_pred"]["c"] = 0.6
    result = build.hybrid_gate(ranking)
    assert result["status"] == "passed"
    assert result == build.hybrid_gate(ranking)
    ranking["sufficient_coverage"] = False
    assert build.hybrid_gate(ranking)["status"] == "unverified"


def test_producer_seal_checks_nested_bundle_members_and_completion(tmp_path):
    build.write(tmp_path / "report.json", dict(status="completed", integrity_verified_after=True))
    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle" / "model.pt").write_bytes(b"weights")
    build.seal(tmp_path)
    assert build.verify_seal(tmp_path)["status"] == "completed"
    (tmp_path / "bundle" / "model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed producer"):
        build.verify_seal(tmp_path)


@pytest.mark.parametrize("crosses_deadline", [False, True])
def test_finalize_counts_post_seal_deadline_and_keeps_root_ledger(
    tmp_path, monkeypatch, crosses_deadline
):
    folder = tmp_path / "audit"
    folder.mkdir()
    args = SimpleNamespace(stage="audit", output=tmp_path)
    build.write(folder / "report.json", dict(status="completed", integrity_verified_after=True))
    monkeypatch.setattr(build, "identities", lambda _: {})
    monkeypatch.setattr(build, "verify_inputs", lambda _: None)
    monkeypatch.setattr(
        build,
        "planned_ledger",
        lambda _: dict(roots=[dict(root_id="r", status="not_started")], planned_root_count=18),
    )
    now = [1.0]
    monkeypatch.setattr(build, "elapsed", lambda: now[0])
    real = build.seal

    def seal(path):
        real(path)
        if crosses_deadline:
            now[0] = 61

    monkeypatch.setattr(build, "seal", seal)
    assert build.finalize(args, 0, False, {}) == (2 if crosses_deadline else 0)
    report = build.read(folder / "report.json")
    assert report["roots"][0]["root_id"] == "r"
    assert report["status"] == ("incomplete" if crosses_deadline else "completed")
    assert build.read(folder / "seal.json")["artifacts"]["report.json"] == build.digest(
        folder / "report.json"
    )


def test_missing_worker_report_remains_unverified_with_all_planned_roots(tmp_path, monkeypatch):
    (tmp_path / "audit").mkdir()
    args = SimpleNamespace(stage="audit", output=tmp_path)
    monkeypatch.setattr(build, "identities", lambda _: {})
    monkeypatch.setattr(build, "verify_inputs", lambda _: None)
    monkeypatch.setattr(build, "elapsed", lambda: 1)
    monkeypatch.setattr(
        build,
        "planned_ledger",
        lambda _: dict(roots=[dict(root_id=str(i), status="not_started") for i in range(18)]),
    )
    assert build.finalize(args, -9, True, {}) == 2
    report = build.read(tmp_path / "audit/report.json")
    assert len(report["roots"]) == 18
    assert report["decision"]["status"] == "unverified"


def test_float32_reference_formula_preserves_both_groups():
    # No large-dimensional weighting: means are taken within each group first.
    visual = np.asarray([0, 2], dtype=np.float32)
    pose = np.asarray([2, 2], dtype=np.float32)
    result = 0.5 * visual / np.float32(2) + 0.5 * pose / np.float32(4)
    np.testing.assert_array_equal(result, np.asarray([0.25, 0.75], dtype=np.float32))


def test_historical_stage_chain_rejects_mixed_head_without_requiring_current_old_config(
    tmp_path, monkeypatch
):
    alignment = tmp_path / "alignment"
    shared = dict(
        dataset="data",
        checkpoint="sensor",
        revision="old-revision",
        script="frozen-helper",
        **{"src/embodied_jepa/config.py": "old-config"},
    )
    reports = {
        "prepare": dict(status="completed", integrity_verified_after=True),
        "fit": dict(status="completed", integrity_verified_after=True, updates=2000),
        "evaluate": dict(
            status="completed",
            integrity_verified_after=True,
            decisions={"neural": {"status": "passed"}},
        ),
    }
    for stage in reports:
        build.write(alignment / stage / "registration.json", {"identities": shared})
    monkeypatch.setattr(
        build,
        "helper",
        lambda: SimpleNamespace(
            DATA_SHA="data", CHECKPOINT_SHA="sensor", verify_payloads=lambda _: None
        ),
    )
    monkeypatch.setattr(build, "verify_seal", lambda path: reports[path.name])
    monkeypatch.setattr(build, "digest", lambda _: "frozen-helper")
    args = SimpleNamespace(stage="prepare", dataset=tmp_path / "dataset", alignment=alignment)
    build.verify_inputs(args)  # Historical config is intentionally not today's config.
    mixed = shared | {"revision": "different-training-source"}
    build.write(alignment / "fit/registration.json", {"identities": mixed})
    with pytest.raises(ValueError, match="different historical"):
        build.verify_inputs(args)


def test_runtime_helper_matches_declared_float32_formula_and_visual_changes_cost():
    pytest.importorskip("torch")
    from embodied_jepa.models.aligned_sensor import hybrid_cost

    dv = np.asarray([0, 2, 1e-5], dtype=np.float32)
    dq = np.asarray([2, 2, 1e-3], dtype=np.float32)
    scales = {"visual": 0.17, "pose": 0.39}
    expected = np.float32(0.5) * dv / np.float32(scales["visual"]) + np.float32(
        0.5
    ) * dq / np.float32(scales["pose"])
    actual = hybrid_cost(dv, dq, scales)
    np.testing.assert_allclose(actual, expected, rtol=build.PARITY_RTOL, atol=build.PARITY_ATOL)
    assert actual[1] > actual[0]
    assert actual.dtype == np.float32
