"""TASK-066 runner guards (protocol section 10), in both directions, on synthetic inputs.

The guards reused from TASK-065's runner are tested by ``tests/test_latent_dynamics_runner.py``;
this file tests what TASK-066 adds or changes, and that the reused helpers are the pinned ones.
No corpus frame is read and no model is trained; nothing here is a learned-dynamics or control
claim.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import latent_dynamics as ld
from embodied_jepa import token_dynamics as td

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_train_apple_token_dynamics", ROOT / "scripts" / "train_apple_token_dynamics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = _load()
MANIFEST = json.loads(R.MANIFEST.read_text())


def test_the_runner_and_the_reused_runner_are_pinned():
    for path in ("scripts/train_apple_token_dynamics.py", "scripts/train_apple_latent_dynamics.py"):
        want = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        assert MANIFEST["hashes"][path] == want, path


def test_the_reused_helpers_are_task065s():
    assert R.R65.__file__.endswith("train_apple_latent_dynamics.py")
    assert issubclass(R.GuardError, ld.GuardError)
    assert R.CACHE_ANCHOR_SANITY == 1e-3 and R.CACHE_LAYOUT == 16
    assert R.ALL_HORIZONS == (1, 2, 4, 8, 16, 32, 64)


# ----- G-frozen --------------------------------------------------------------------------------
def test_check_frozen_both_directions(monkeypatch):
    monkeypatch.setattr(td, "UPDATES", None)
    with pytest.raises(R.GuardError, match="G-frozen"):
        R.check_frozen()
    monkeypatch.setattr(td, "UPDATES", 10_000)
    monkeypatch.setattr(td, "PER_RUN_SECONDS", 1.0)
    monkeypatch.setattr(td, "GLOBAL_WALL_SECONDS", 1.0)
    bars = {"G1_min_effective_rank_ratio": 0.2, "G1_min_std_ratio": 0.4}
    monkeypatch.setattr(td, "THRESHOLDS", td.THRESHOLDS | bars)
    R.check_frozen()


# ----- G-anchor (tokens) -----------------------------------------------------------------------
def test_check_token_anchor_both_directions():
    pins = MANIFEST["data"]
    good = {
        "frames": pins["task064_stored_post_look_frames_sha256"],
        "P_cls": pins["task064_P_cls_feature_sha256"],
        "P_tok": pins["task064_P_tok_feature_sha256"],
    }
    R.check_token_anchor(good, pins)
    for key, message in (("frames", "frames"), ("P_cls", "P-cls"), ("P_tok", "P-tok")):
        with pytest.raises(R.GuardError, match=message):
            R.check_token_anchor(good | {key: "0" * 64}, pins)


# ----- G-cache (b): partial batches -------------------------------------------------------------
def test_partial_batch_determinism_both_directions():
    rows = np.random.default_rng(0).normal(size=(35, 8)).astype(np.float32)
    facts = R.check_partial_batches({"e": rows}, {"e": rows.copy()})
    assert facts == {"e": {"frames": 35, "final_batch": 3, "bit_identical": True}}
    moved = rows.copy()
    moved[34, 0] += np.float32(1e-6)  # a change in the partial final batch only
    with pytest.raises(R.GuardError, match="partial final batch"):
        R.check_partial_batches({"e": rows}, {"e": moved})
    with pytest.raises(R.GuardError, match="no partial-batch"):
        R.check_partial_batches({}, {})


# ----- G-cache (c): the 1e-3 anchor bound on pooled rows ------------------------------------------
def test_cache_anchor_sanity_both_directions():
    anchor = np.random.default_rng(1).normal(0, 2, (4, 6144))
    cache = anchor.copy()
    cache[3, 7] += 2.93e-5  # the size of the pre-freeze partial-batch difference
    facts = R.cache_anchor_sanity(cache, anchor, list("abcd"), ["d"])
    assert facts["differing_roots"] == ["d"] and facts["bound"] == 1e-3
    assert facts["anchor_partial_batch_roots"] == ["d"]
    assert "partial batch" in facts["explanation"]
    with pytest.raises(R.GuardError, match="G-cache"):
        R.cache_anchor_sanity(anchor + 2e-3, anchor, list("abcd"), [])


# ----- streamed statistics ------------------------------------------------------------------------
def test_chunked_moments_equal_numpy():
    x = np.random.default_rng(2).normal(3, 2, (1000, 5)).astype(np.float32)
    rows = np.arange(10, 990)
    mean, std = R.chunked_moments(x, rows, chunk=97)
    np.testing.assert_allclose(mean, x[rows].astype(np.float64).mean(0), rtol=1e-12)
    np.testing.assert_allclose(std, x[rows].astype(np.float64).std(0), rtol=1e-10)


class _Model:
    """A fake predictor: every step adds the action sum to the start latent (no torch)."""

    def predict_features(self, start, actions):
        steps = np.cumsum(actions.sum(-1), axis=1)[..., None]
        return (start[:, None, :] + steps).astype(np.float32)


def test_rollout_errors_match_the_direct_computation():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(60, 4)).astype(np.float32)
    actions = rng.uniform(-1, 1, (60, 14)).astype(np.float32)
    starts = np.array([0, 4, 30])
    scale = np.full(4, 2.0)
    shift = np.zeros(4)
    errors, moments, projected = R.rollout_errors(
        _Model(), features, actions, starts, scale, "true", None, shift
    )
    f, a = R.R65.gather(features, actions, starts, td.TRAIN_HORIZON)
    predicted = _Model().predict_features(f[:, 0], a)
    for h in R.EALL_HORIZONS:
        want = ld.normalized_sq_error(predicted[:, h - 1], f[:, h], scale)
        np.testing.assert_allclose(errors[h], want)
    for h in td.GATED_HORIZONS:
        direct = ld.collapse_statistics(predicted[:, h - 1] / scale, f[:, h] / scale)
        enc = td.Moments(shift)
        enc.add(f[:, h] / scale)
        got = td.collapse_statistics(moments[h], enc)
        assert got["std_ratio"] == pytest.approx(direct["std_ratio"])
    zero, _, none = R.rollout_errors(
        _Model(), features, actions, starts, scale, "zero", None, shift
    )
    assert none is None and projected is None
    want = ld.normalized_sq_error(f[:, 0], f[:, 8], scale)
    np.testing.assert_allclose(zero[8], want, rtol=1e-6)


def test_seed_collapse_pools_both_halves_and_releases_the_moments():
    rng = np.random.default_rng(4)
    shift = np.zeros(3)
    rows = {k: rng.normal(size=(20, 3)) for k in ("WA", "WB", "NA", "NB", "E")}

    basis = np.eye(3)
    # half A's windows come from sessions 0-1, half B's from 2-3 (each model predicts the other)
    sess = {"A": np.repeat([2, 3], 10), "B": np.repeat([0, 1], 10)}

    def moments(x):
        m = td.Moments(shift)
        m.add(x)
        return m

    def projected(x, s):
        m = td.SessionMoments(shift, basis)
        m.add(x, s)
        return m

    results = {
        (arm, 0, half): {
            "eall_moments": {h: moments(rows[arm + half]) for h in (8, 16)},
            "eall_projected": {h: projected(rows[arm + half], sess[half]) for h in (8, 16)},
        }
        for arm in ("W", "N")
        for half in ("A", "B")
    }
    enc_rows = np.concatenate([rows["E"], rows["E"]])
    enc = moments(enc_rows)
    enc_p = projected(enc_rows, np.concatenate([sess["A"], sess["B"]]))
    ctx = {
        "encoded_moments": {8: enc, 16: enc},
        "encoded_projected": {8: enc_p, 16: enc_p},
        "copy_collapse": {8: "c8", 16: "c16"},
    }
    out = R.seed_collapse(ctx, results, 0)
    want = ld.collapse_statistics(np.concatenate([rows["WA"], rows["WB"]]), enc_rows)
    for key, value in want.items():
        assert out[8]["collapse_W"][key] == pytest.approx(value, rel=1e-9), key
    w_rows = np.concatenate([rows["WA"], rows["WB"]])
    n_rows = np.concatenate([rows["NA"], rows["NB"]])
    direct = (ld.effective_rank(w_rows) - ld.effective_rank(n_rows)) / ld.effective_rank(enc_rows)
    assert out[8]["rank_W_over_N"]["difference"] == pytest.approx(direct, rel=1e-9)
    assert out[8]["rank_W_over_N"]["sessions"] == 4
    assert out[16]["collapse_copy_last"] == "c16"
    assert all("eall_moments" not in r and "eall_projected" not in r for r in results.values())


# ----- reports: V on any crash -------------------------------------------------------------------
def test_a_crash_at_preflight_writes_a_void_report(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "MANIFEST", tmp_path / "missing.json")
    report = R.run(tmp_path / "out", tmp_path / "ckpt")
    written = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["outcome"] == written["outcome"] == "V"
    assert "FileNotFoundError" in written["void_reason"]
    assert written["decision"] == {"outcome": "V", "abandonment_clause_fires": False}
    assert written["learned_apple_to_plate_successes"] == 0
    assert written["test_split_decoded"] is False


def test_a_guard_failure_mid_run_writes_a_void_report(tmp_path, monkeypatch):
    def failing(report, *args, **kwargs):
        report["stages"]["features"] = 1.0
        raise R.GuardError("G-cache: planted partial-batch difference")

    monkeypatch.setattr(R, "_run", failing)
    R.run(tmp_path / "out", tmp_path / "ckpt")
    written = json.loads((tmp_path / "out" / "report.json").read_text())
    assert written["outcome"] == "V" and "planted" in written["void_reason"]
    assert written["stages"] == {"features": 1.0}


def test_the_runner_refuses_to_overwrite(tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        R.run(tmp_path / "out", tmp_path / "ckpt")
    (tmp_path / "ckpt2").mkdir()
    with pytest.raises(FileExistsError):
        R.run(tmp_path / "out2", tmp_path / "ckpt2")
