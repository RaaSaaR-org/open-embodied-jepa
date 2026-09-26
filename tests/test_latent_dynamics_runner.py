"""TASK-065 runner guards (protocol §9), in both directions, on synthetic inputs.

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

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_train_apple_latent_dynamics", ROOT / "scripts" / "train_apple_latent_dynamics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = _load()
MANIFEST = json.loads(R.MANIFEST.read_text())


def test_the_runner_is_pinned_by_the_manifest():
    path = "scripts/train_apple_latent_dynamics.py"
    want = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    assert MANIFEST["hashes"][path] == want


# ----- G-hash ---------------------------------------------------------------------------------
def test_check_pins_both_directions(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"alpha")
    good = {"a.txt": hashlib.sha256(b"alpha").hexdigest()}
    assert R.check_pins(good, tmp_path) == good
    with pytest.raises(R.GuardError, match="G-hash"):
        R.check_pins({"a.txt": "0" * 64}, tmp_path)
    with pytest.raises(R.GuardError, match="missing"):
        R.check_pins({"b.txt": "0" * 64}, tmp_path)


def test_check_clean_both_directions():
    R.check_clean([])
    with pytest.raises(R.GuardError, match="dirty"):
        R.check_clean([" M src/x.py"])


# ----- G-device, G-data, G-split, G-folds, G-weights ---------------------------------------------
def test_check_device():
    R.check_device(True)
    with pytest.raises(R.GuardError, match="G-device"):
        R.check_device(False)


def test_check_data_both_directions():
    R.check_data(ld.DATASET_MANIFEST_SHA256, ld.EXPECTED_SESSIONS, ld.EXPECTED_EPISODES)
    with pytest.raises(R.GuardError, match="manifest"):
        R.check_data("0" * 64, ld.EXPECTED_SESSIONS, ld.EXPECTED_EPISODES)
    with pytest.raises(R.GuardError, match="sessions"):
        R.check_data(
            ld.DATASET_MANIFEST_SHA256,
            ld.EXPECTED_SESSIONS | {"test": 11},
            ld.EXPECTED_EPISODES,
        )
    with pytest.raises(R.GuardError, match="episodes"):
        R.check_data(
            ld.DATASET_MANIFEST_SHA256,
            ld.EXPECTED_SESSIONS,
            ld.EXPECTED_EPISODES | {"val": 79},
        )


def test_check_halves_and_folds_both_directions():
    halves = ld.session_halves([f"s{i}" for i in range(10)])
    R.check_halves(halves, ld.halves_sha256(halves))
    with pytest.raises(R.GuardError, match="G-split"):
        R.check_halves(halves, "0" * 64)
    R.check_folds("abc", "abc")
    with pytest.raises(R.GuardError, match="G-folds"):
        R.check_folds("abc", "abd")


def test_check_weights_both_directions():
    R.check_weights("d" * 64, "d" * 64)
    with pytest.raises(R.GuardError, match="G-weights"):
        R.check_weights("d" * 64, "e" * 64)


# ----- G-anchor, G-repro, G-cache ---------------------------------------------------------------
def test_check_anchor_both_directions():
    pins = MANIFEST["data"]
    frames, feats = (
        pins["task064_stored_post_look_frames_sha256"],
        pins["task064_P_cls_feature_sha256"],
    )
    R.check_anchor(frames, feats, pins)
    with pytest.raises(R.GuardError, match="frames"):
        R.check_anchor("0" * 64, feats, pins)
    with pytest.raises(R.GuardError, match="features"):
        R.check_anchor(frames, "0" * 64, pins)


def test_check_repro_and_cache_both_directions():
    x = np.arange(12.0).reshape(3, 4)
    R.check_repro(x, x.copy())
    with pytest.raises(R.GuardError, match="G-repro"):
        R.check_repro(x, x + 1e-12)
    assert R.check_cache(x.astype(np.float32), x) == 0.0
    with pytest.raises(R.GuardError, match="G-cache"):
        R.check_cache(x + 1e-4, x)
    with pytest.raises(R.GuardError, match="G-cache"):
        R.check_cache(x * np.nan, x)


# ----- G-labels, G-finite -----------------------------------------------------------------------
def test_check_labels_both_directions():
    R.check_look_labels([-1] * 8 + [0, 0], "e")
    with pytest.raises(R.GuardError, match="look"):
        R.check_look_labels([-1] * 7 + [0, 0, 0], "e")
    with pytest.raises(R.GuardError, match="look"):
        R.check_look_labels([-1] * 5, "e")
    assert R.check_apple_label([0.3, -0.2], [0.3, -0.2 + 5e-7], "e") <= 1e-6
    with pytest.raises(R.GuardError, match="G-labels"):
        R.check_apple_label([0.3, -0.2], [0.3, -0.2 + 2e-6], "e")


def test_check_finite():
    R.check_finite("x", np.ones(3))
    with pytest.raises(R.GuardError, match="G-finite"):
        R.check_finite("x", np.array([1.0, np.inf]))


# ----- caps --------------------------------------------------------------------------------------
def test_caps(monkeypatch):
    clock = R.Clock(cap=10.0)
    clock.check("now")
    clock.start -= 11.0
    with pytest.raises(R.GuardError, match="global wall cap"):
        clock.check("later")
    import time

    R.check_stage_cap(time.monotonic(), 5.0, "x")
    with pytest.raises(R.GuardError, match="G-cap"):
        R.check_stage_cap(time.monotonic() - 6.0, 5.0, "x")


# ----- Q-split: the reader refuses the test split -----------------------------------------
class _Store:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = {
            "episodes": [
                {"episode_id": "tr", "path": "tr.parquet", "length": 2},
                {"episode_id": "te", "path": "te.parquet", "length": 2},
            ],
            "sha256": {"tr.parquet": "0" * 64, "te.parquet": "0" * 64},
        }


def test_reader_refuses_a_test_episode_before_opening_anything(tmp_path):
    reader = R.TrainValReader(_Store(tmp_path), {"tr"})
    for call in (lambda: reader.episode("te"), lambda: reader.labels("te")):
        with pytest.raises(R.GuardError, match="Q-split"):
            call()
    assert reader.decoded == set()
    (tmp_path / "tr.parquet").write_bytes(b"not the pinned bytes")
    with pytest.raises(R.GuardError, match="hash mismatch"):
        reader.episode("tr")


# ----- reports: V on any crash, non-finite values as null -----------------------------------------
def test_finite_json_nulls_and_lists_non_finite_values(tmp_path):
    path = tmp_path / "report.json"
    R.write_report(path, {"a": float("nan"), "b": [1.0, np.float32(np.inf)], "c": {1: 2}})
    report = json.loads(path.read_text())
    assert report["a"] is None and report["b"] == [1.0, None]
    assert report["c"] == {"1": 2}
    assert report["non_finite_fields"] == ["/a=nan", "/b/1=inf"]
    with pytest.raises(FileExistsError):
        R.write_report(path, {})


def test_a_crash_at_preflight_writes_a_void_report(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "MANIFEST", tmp_path / "missing.json")
    report = R.run(tmp_path / "out", tmp_path / "ckpt")
    written = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["outcome"] == written["outcome"] == "V"
    assert "FileNotFoundError" in written["void_reason"]
    assert written["decision"] == {"outcome": "V", "abandonment_clause_fires": False}
    assert written["learned_apple_to_plate_successes"] == 0


def test_a_guard_failure_mid_run_writes_a_void_report(tmp_path, monkeypatch):
    def failing(report, *args, **kwargs):
        report["stages"]["preflight"] = 1.0
        raise R.GuardError("G-cache: planted")

    monkeypatch.setattr(R, "_run", failing)
    R.run(tmp_path / "out", tmp_path / "ckpt")
    written = json.loads((tmp_path / "out" / "report.json").read_text())
    assert written["outcome"] == "V" and "G-cache: planted" in written["void_reason"]
    assert written["stages"] == {"preflight": 1.0}
    assert written["test_split_decoded"] is False


def test_the_runner_refuses_to_overwrite(tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        R.run(tmp_path / "out", tmp_path / "ckpt")
    (tmp_path / "ckpt2").mkdir()
    with pytest.raises(FileExistsError):
        R.run(tmp_path / "out2", tmp_path / "ckpt2")


# ----- statistics helpers ------------------------------------------------------------------------
def test_windows_and_gather_stay_inside_episodes():
    table = [{"length": 20, "session": "s0"}, {"length": 10, "session": "s1"}]
    offsets = np.array([0, 20])
    starts, ep, sessions = R.window_arrays(table, offsets, [0, 1], horizon=8, stride=4)
    assert starts.tolist() == [0, 4, 8, 20] and ep.tolist() == [0, 0, 0, 1]
    assert sessions.tolist() == ["s0", "s0", "s0", "s1"]
    features = np.arange(30.0)[:, None]
    actions = np.arange(30.0)[:, None]
    f, a = R.gather(features, actions, starts, 8)
    assert f.shape == (4, 9, 1) and a.shape == (4, 8, 1)
    assert f[-1, -1, 0] == 28.0  # last window ends at the episode's last frame (index 29 unused)


def test_t1_stats_carry_the_g5_quantities():
    rng = np.random.default_rng(0)
    err = rng.uniform(0.2, 0.8, 50)
    stats = R.t1_stats(
        err, err + 1.0, err - 0.1, np.random.default_rng(1).integers(0, 50, (200, 50))
    )
    assert stats["median_cm"] == pytest.approx(np.median(err))
    assert stats["excess_over_encoded_cm"]["difference"] == pytest.approx(0.1)
    assert ld.g5_passes(stats)


def test_g4_refuses_an_undefined_resample():
    """The runner's G4 also requires no zero-denominator resample (float max would otherwise
    help a lower bound)."""
    source = (ROOT / "scripts" / "train_apple_latent_dynamics.py").read_text()
    assert 'e["wrong_over_W"]["undefined_resamples"] == 0' in source
    assert 'e["zero_over_W"]["undefined_resamples"] == 0' in source
    clusters = np.array([0, 1])
    ratio = ld.cluster_ratio(
        np.array([1.0, 1.0]), np.array([0.0, 1.0]), clusters, ld.cluster_bootstrap_indices(2)
    )
    assert ratio["undefined_resamples"] > 0
