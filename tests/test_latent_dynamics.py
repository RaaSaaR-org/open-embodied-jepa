"""TASK-065 design code (``latent_dynamics``): partition, windows, statistics, gates and rows.

Synthetic inputs only; no corpus frame is read and nothing here is a learned-dynamics or control
claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import latent_dynamics as ld
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "benchmarks" / "manifests" / "apple-latent-dynamics-v1.json").read_text()
)


def _train_sessions():
    """The 170 train sessions, from the pinned TASK-064 split lists (val + test excluded)."""
    look = json.loads((ROOT / "benchmarks/manifests/apple-look-corpus-v1.json").read_text())
    held = set(look["splits"]["val_sessions"]) | set(look["splits"]["test_sessions"])
    return [f"look-reset-{s}" for s in range(47000, 47200) if s not in held]


# ----- the manifest and the module agree ----------------------------------------------------------
def test_manifest_matches_the_module():
    assert MANIFEST["task"] == ld.TASK and MANIFEST["protocol"] == ld.PROTOCOL
    assert MANIFEST["data"]["dataset_manifest_sha256"] == ld.DATASET_MANIFEST_SHA256
    assert MANIFEST["data"]["sessions"] == ld.EXPECTED_SESSIONS
    assert MANIFEST["model"]["config"] == ld.MODEL_CONFIG
    assert MANIFEST["model"]["backend"] == ld.BACKEND == "leworldmodel"
    assert MANIFEST["gates"]["thresholds"] == ld.THRESHOLDS
    assert MANIFEST["training"]["model_seeds"] == list(ld.MODEL_SEEDS)
    assert MANIFEST["evaluation"]["gated_horizons"] == list(ld.GATED_HORIZONS) == [8, 16]
    assert list(MANIFEST["pre_declared_outcomes_in_order"]) == list(ld.ROWS)
    assert MANIFEST["abandonment_clause"]["fires_on"] == ["WM-APPLE-LOST", "WM-NO-DYNAMICS"]
    assert MANIFEST["learned_apple_to_plate_successes"] == 0
    assert MANIFEST["exemption_spent"] is False
    assert MANIFEST["data"]["test_split_decoded"] is False


def test_committed_pins_match_the_tree():
    """Every pinned file that is committed (not data/, third_party/) matches its pin.

    Except base.py, which TASK-072 changed (CUDA support): its pin must equal the bytes the
    TASK-065 checkpoints were written with, recorded in task072-checkpoint-compatibility.json.
    """
    retired = json.loads(
        (ROOT / "benchmarks/manifests/task072-checkpoint-compatibility.json").read_text()
    )["changed_files"]
    for path, want in MANIFEST["hashes"].items():
        if path.startswith(("data/", "third_party/", "outputs/", "checkpoints/")):
            continue
        if path in retired:
            assert retired[path]["sha256_before"] == want, path
            head = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            assert head == retired[path]["sha256_after"], f"{path} changed again; record it"
            continue
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == want, path


def test_model_config_turns_on_only_the_preregistered_option():
    pytest.importorskip("torch")
    from embodied_jepa.models import frozen_encoder as fe
    from embodied_jepa.models.base import VisualModel

    config = ld.MODEL_CONFIG
    assert config["frozen_encoder"] in fe.FROZEN_ENCODERS
    assert ld.BACKEND in fe.BACKENDS
    assert "frozen_encoder" not in VisualModel.defaults  # no existing model learns the key
    off = {
        "state_fusion",
        "readout_heads",
        "cameras",
        "action_chunk",
        "predictor_step_embedding",
        "multistep_tail_weight",
    }
    assert off.isdisjoint(config)
    assert config["sigreg_weight"] == 0.0 and config["latent_dim"] == 384


# ----- partition ------------------------------------------------------------------------------
def test_halves_are_pinned_and_disjoint():
    sessions = _train_sessions()
    assert len(sessions) == 170
    halves = ld.session_halves(sessions)
    assert ld.halves_sha256(halves) == MANIFEST["halves"]["sha256"]
    assert sum(v == "A" for v in halves.values()) == sum(v == "B" for v in halves.values()) == 85
    assert ld.session_halves(list(reversed(sessions))) == halves  # order-free
    assert ld.other("A") == "B" and ld.other("B") == "A"
    with pytest.raises(ContractError):
        ld.other("C")
    with pytest.raises(ContractError):
        ld.session_halves(["x", "x"])


def test_sampler_streams_differ_by_seed_and_half_and_pair_the_arms():
    states = {
        (s, h, a): tuple(ld.sampler_seed(s, h, a).generate_state(2).tolist())
        for s in ld.MODEL_SEEDS
        for h in ld.HALVES
        for a in ld.ARMS
    }
    assert len(set(states.values())) == 6  # one stream per (seed, half)
    for s in ld.MODEL_SEEDS:
        for h in ld.HALVES:
            assert states[(s, h, "W")] == states[(s, h, "N")]  # W and N see the same windows
    with pytest.raises(ContractError):
        ld.sampler_seed(0, "A", "X")


def _training_fixture():
    splits = {"a1": "train", "a2": "train", "b1": "train", "v1": "val", "t1": "test"}
    sessions_of = {"a1": "sa", "a2": "sa", "b1": "sb", "v1": "sv", "t1": "st"}
    halves = {"sa": "A", "sb": "B"}
    return splits, sessions_of, halves


def test_check_training_episodes_accepts_the_exact_halves():
    splits, sessions_of, halves = _training_fixture()
    ld.check_training_episodes({"A": ["a1", "a2"], "B": ["b1"]}, sessions_of, halves, splits)


@pytest.mark.parametrize(
    "training, message",
    [
        ({"A": ["a1", "a2", "v1"], "B": ["b1"]}, "split val"),
        ({"A": ["a1", "t1"], "B": ["b1"]}, "split test"),
        ({"A": ["a1", "a2", "b1"], "B": ["b1"]}, "not in half A"),
        ({"A": ["a1"], "B": ["b1"]}, "cover the train split"),
        ({"A": [], "B": ["b1"]}, "no training episode"),
    ],
)
def test_check_training_episodes_refuses(training, message):
    splits, sessions_of, halves = _training_fixture()
    with pytest.raises(ld.GuardError, match=message):
        ld.check_training_episodes(training, sessions_of, halves, splits)


# ----- windows ---------------------------------------------------------------------------------
def test_windows_never_cross_an_episode_and_respect_the_stride():
    w = ld.windows([20, 15, 3], horizon=16, stride=4)
    assert w.tolist() == [[0, 0], [0, 4]]
    assert ld.training_windows([17]).tolist() == [[0, 0], [0, 1]]
    assert ld.windows([2], 16, 4).shape == (0, 2)


def test_cross_session_shuffle_never_keeps_a_session():
    rng = np.random.default_rng(0)
    sessions = np.repeat(np.arange(30), rng.integers(1, 40, 30))
    perm = ld.cross_session_shuffle(sessions, np.random.default_rng(ld.SHUFFLE_SEED))
    assert sorted(perm.tolist()) == list(range(len(sessions)))
    assert not (sessions[perm] == sessions).any()
    again = ld.cross_session_shuffle(sessions, np.random.default_rng(ld.SHUFFLE_SEED))
    assert np.array_equal(perm, again)
    with pytest.raises(ContractError):
        ld.cross_session_shuffle(np.array([0, 0, 0, 1]), np.random.default_rng(0))


# ----- statistics ---------------------------------------------------------------------------------
def test_normalized_error_and_cluster_ratio():
    err = ld.normalized_sq_error(np.ones((2, 3)), np.zeros((2, 3)), np.array([1.0, 2.0, 1.0]))
    np.testing.assert_allclose(err, [0.75, 0.75])
    clusters = np.array([0, 0, 1, 2])
    num = np.array([1.0, 1.0, 2.0, 2.0])
    den = np.array([2.0, 2.0, 4.0, 4.0])
    idx = ld.cluster_bootstrap_indices(3, resamples=200)
    result = ld.cluster_ratio(num, den, clusters, idx)
    assert result["ratio"] == pytest.approx(0.5)
    assert result["ci95"] == pytest.approx([0.5, 0.5])
    assert result["clusters"] == 3 and result["windows"] == 4
    with pytest.raises(ContractError):
        ld.cluster_ratio(num, den, clusters, ld.cluster_bootstrap_indices(4, resamples=5))


def test_cluster_ratio_interval_widens_with_cluster_disagreement():
    clusters = np.repeat(np.arange(20), 5)
    rng = np.random.default_rng(1)
    den = np.ones(100)
    num = np.where(clusters % 2 == 0, 0.2, 1.8) + rng.normal(0, 0.01, 100)
    result = ld.cluster_ratio(num, den, clusters, ld.cluster_bootstrap_indices(20))
    assert result["ci95"][0] < result["ratio"] < result["ci95"][1]
    assert result["ci95"][1] - result["ci95"][0] > 0.3


def test_collapse_statistics_see_a_collapsed_prediction():
    rng = np.random.default_rng(2)
    encoded = rng.normal(size=(500, 16))
    healthy = ld.collapse_statistics(encoded * 0.9 + rng.normal(0, 0.1, encoded.shape), encoded)
    assert ld.g1_passes(healthy)
    constant = ld.collapse_statistics(np.tile(encoded.mean(0), (500, 1)), encoded)
    assert constant["predicted_collapsed_fraction"] == 1.0 and not ld.g1_passes(constant)
    low_rank = ld.collapse_statistics(np.outer(rng.normal(size=500), np.ones(16)), encoded)
    assert low_rank["effective_rank_ratio"] < 0.5 and not ld.g1_passes(low_rank)
    shrunk = ld.collapse_statistics(encoded * 0.3, encoded)
    assert shrunk["std_ratio"] == pytest.approx(0.3) and not ld.g1_passes(shrunk)


# ----- gates, both directions -------------------------------------------------------------------
def _ratio(lo, hi):
    return {"ratio": (lo + hi) / 2, "ci95": [lo, hi]}


def test_g2_g3_g4_boundaries():
    assert ld.g2_passes(_ratio(0.5, 0.8)) and not ld.g2_passes(_ratio(0.5, 0.8001))
    assert ld.g3_passes(_ratio(0.7, 0.999)) and not ld.g3_passes(_ratio(0.7, 1.0))
    assert ld.g4_passes(_ratio(1.10, 2.0), _ratio(1.5, 3.0))
    assert not ld.g4_passes(_ratio(1.0999, 2.0), _ratio(1.5, 3.0))
    assert not ld.g4_passes(_ratio(1.2, 2.0), _ratio(1.05, 3.0))


def _readable(median=0.6, ratio_hi=0.4, excess_hi=0.2):
    return {
        "median_cm": median,
        "ratio_to_B_occ": {"ratio": ratio_hi / 2, "ci95": [0.1, ratio_hi]},
        "excess_over_encoded_cm": {"difference": excess_hi / 2, "ci95": [-0.1, excess_hi]},
    }


def test_g5_needs_the_bar_and_the_margin():
    assert ld.g5_passes(_readable())
    assert ld.g5_passes(_readable(median=1.5, ratio_hi=0.6, excess_hi=0.5))
    assert not ld.g5_passes(_readable(median=1.51))
    assert not ld.g5_passes(_readable(ratio_hi=0.61))
    assert not ld.g5_passes(_readable(excess_hi=0.51))


def _horizons(fail=None, at=(8, 16)):
    out = {}
    for h in ld.GATED_HORIZONS:
        out[h] = {g: not (g == fail and h in at) for g in ld.GATES}
    return out


def test_seed_gates_need_both_horizons():
    assert ld.seed_gates(_horizons())["passes"]
    only16 = ld.seed_gates(_horizons("G2", at=(8,)))
    assert not only16["passes"] and not only16["gates"]["G2"] and not only16["dynamics"]
    apple = ld.seed_gates(_horizons("G5", at=(16,)))
    assert not apple["passes"] and apple["dynamics"]
    with pytest.raises(ContractError):
        ld.seed_gates({8: _horizons()[8]})


# ----- rows ---------------------------------------------------------------------------------------
def _seeds(*kinds):
    table = {
        "pass": {"passes": True, "dynamics": True},
        "apple": {"passes": False, "dynamics": True},
        "none": {"passes": False, "dynamics": False},
    }
    return {s: table[k] for s, k in zip(ld.MODEL_SEEDS, kinds, strict=True)}


@pytest.mark.parametrize(
    "kinds, row, fires",
    [
        (("pass", "pass", "pass"), "WM-DYNAMICS", False),
        (("pass", "pass", "apple"), "WM-UNSTABLE", False),
        (("pass", "none", "none"), "WM-UNSTABLE", False),
        (("apple", "apple", "none"), "WM-APPLE-LOST", True),
        (("apple", "apple", "apple"), "WM-APPLE-LOST", True),
        (("apple", "none", "none"), "WM-NO-DYNAMICS", True),
        (("none", "none", "none"), "WM-NO-DYNAMICS", True),
    ],
)
def test_every_row(kinds, row, fires):
    decision = ld.decide(void=False, seeds=_seeds(*kinds))
    assert decision["outcome"] == row
    assert decision["abandonment_clause_fires"] is fires


def test_void_first_and_complete_inputs():
    assert ld.decide(void=True) == {"outcome": "V", "abandonment_clause_fires": False}
    assert ld.decide(void=True, seeds=_seeds("pass", "pass", "pass"))["outcome"] == "V"
    with pytest.raises(ContractError):
        ld.decide(void=False, seeds={0: {"passes": True, "dynamics": True}})
    with pytest.raises(ContractError):
        ld.abandonment_fires("WM-OTHER")
    assert not ld.abandonment_fires("V")
