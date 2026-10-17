"""TASK-100 (SO-101 S4) on tiny synthetic data, CPU only: contract evidence, not results.

Roots, windows, tables, the cluster bootstrap, G1's statistics and projected moments, the
readouts, the subset and grid rules, G6, the gates and the row in both directions, the feature
store and its test guard, and the schema-sized frozen-token class for both backends (a tiny
random-init DINOv2 stands in for the pinned weights).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import so101_wm as sw
from embodied_jepa import so101_wm_data as sd
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]


# ----- windows, roots, tables --------------------------------------------------------------------
def test_windows_stay_inside_episodes():
    frames, offsets = np.array([401, 20, 401]), np.array([0, 401, 421])
    starts = sw.window_starts(frames, offsets)
    assert len(starts) == 385 + 4 + 385
    episode = np.searchsorted(offsets, starts, side="right") - 1
    end = starts + sw.TRAIN_HORIZON
    assert (end <= offsets[episode] + frames[episode] - 1).all()


def test_root_rule_is_37_roots_per_episode():
    starts, ep = sw.roots(np.array([401, 401]), np.array([0, 401]))
    assert len(starts) == 74
    local = starts[:37]
    assert local[0] == 0 and local[-1] == 360 and (np.diff(local) == 10).all()
    assert (local + sw.ROOT_SPAN <= 400).all()
    assert (ep == np.repeat([0, 1], 37)).all()


def test_wrong_and_candidate_tables_use_other_episodes_and_are_fixed_by_salts():
    rng = np.random.default_rng(0)
    ep = np.repeat(np.arange(6), 5)
    wrong = sw.wrong_table(ep)
    assert (ep[wrong] != ep).all()
    assert np.array_equal(wrong, sw.wrong_table(ep))
    commands = rng.uniform(-1, 1, (30, 32, 6)).astype(np.float32)
    c = sw.candidate_sequences(commands, ep)
    seq = c["sequences"]
    assert seq.shape == (30, 16, 32, 6) and np.array_equal(seq[:, 0], commands)
    assert (np.abs(seq) <= 1).all()
    assert (ep[c["other_roots"]] != ep[:, None]).all()
    assert all(len(set(r)) == 4 for r in c["other_roots"])
    grip_only = seq[:, 9:12] - commands[:, None]
    assert np.allclose(grip_only[..., :5], 0)
    assert np.array_equal(sw.candidate_sequences(commands, ep)["sequences"], seq)
    assert c["classes"][:1] == ["executed"] and c["classes"].count("other_episode") == 4


def test_ties_count_against():
    scores = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 3.0], [2.0, 1.0, 0.5]])
    assert sw.true_rank(scores).tolist() == [0, 1, 2]


def test_jaw_targets_follow_the_rate_clip():
    out = sw.jaw_targets(
        np.array([[-1.0, -1.0, 1.0]]), np.array([1.0]), closed=-0.2, opened=1.8, rate=0.25
    )
    assert np.allclose(out, [[0.75, 0.5, 0.75]])


def test_hold_commands_keep_the_first_gripper_command():
    c = np.random.default_rng(1).uniform(-1, 1, (3, 5, 6)).astype(np.float32)
    h = sw.hold_commands(c)
    assert np.allclose(h[..., :5], 0) and np.allclose(h[..., 5], c[:, :1, 5])


# ----- the cluster bootstrap ---------------------------------------------------------------------
def test_bootstrap_medians_with_unit_weights_are_numpy_medians():
    ep = np.repeat(np.arange(4), 3)
    boot = sw.Bootstrap(ep, resamples=50)
    values = np.arange(12.0)
    boot.counts[:] = 1.0
    assert np.allclose(boot.medians(values), np.median(values))
    mask = values % 2 == 0
    assert np.allclose(boot.medians(values, mask), np.median(values[mask]))


def test_bootstrap_known_case_ratio_of_identical_episodes():
    ep = np.repeat(np.arange(10), 4)
    boot = sw.Bootstrap(ep, resamples=200)
    num = np.tile([1.0, 2.0, 3.0, 4.0], 10)
    stat = sw.ratio_of_sums(num, num / 2.0, boot, 0.95)
    assert stat["value"] == pytest.approx(2.0) and np.allclose(stat["ci"], 2.0)
    assert boot.counts.sum(1).tolist() == [10.0] * 200
    again = sw.Bootstrap(ep, resamples=200)
    assert np.array_equal(again.counts, boot.counts)


# ----- G1 ----------------------------------------------------------------------------------------
def test_collapse_statistics_and_truncation():
    rng = np.random.default_rng(2)
    enc = rng.normal(size=(60, 40))
    stats = sw.collapse_statistics(enc, enc)
    assert stats["effective_rank_ratio"] == pytest.approx(1.0)
    assert stats["std_ratio"] == pytest.approx(1.0) and stats["collapsed_fraction"] == 0.0
    t1 = sw.collapse_statistics(sw.truncate(enc, 1), enc)
    assert t1["effective_rank_ratio"] < 0.1
    flat = enc.copy()
    flat[:, :10] = 0.0
    assert sw.collapse_statistics(flat, enc)["collapsed_fraction"] == pytest.approx(0.25)


def test_projected_moments_reproduce_the_direct_projected_rank_ratios():
    rng = np.random.default_rng(3)
    ep = np.repeat(np.arange(5), 8)
    x, n, e = (rng.normal(size=(40, 30)) for _ in range(3))
    basis = np.linalg.qr(rng.normal(size=(30, 6)))[0]
    mx, mn, me = (sw.ProjectedMoments(v, ep, basis, 5) for v in (x, n, e))

    def direct(v):
        return sw.effective_rank_from_energy(sw.gram_spectrum(v @ basis))

    ones = np.ones((1, 5))
    assert mx.ranks(ones)[0] == pytest.approx(direct(x))
    boot = sw.Bootstrap(ep, resamples=20)
    out = sw.projected_rank_difference(mx, mn, me, boot, 0.975)
    assert out["value"] == pytest.approx(direct(x) / direct(e) - direct(n) / direct(e))
    counts = np.array([[2.0, 0.0, 1.0, 1.0, 1.0]])
    rows = np.concatenate([x[ep == j] for j, c in enumerate(counts[0]) for _ in range(int(c))])
    assert mx.ranks(counts)[0] == pytest.approx(direct(rows))


def test_g1_bars_and_rounding():
    bars = sw.g1_bars({8: 0.5, 16: 0.4567}, {8: 0.9, 16: 0.8})
    assert bars["B_rank"] == 0.22 and bars["B_std"] == 0.4
    floors = sw.g1_bars({8: 0.1, 16: 0.1}, {8: 0.2, 16: 0.2})
    assert floors["B_rank"] == 0.10 and floors["B_std"] == 0.25
    assert sw.ceil_above_01(0.23) == 0.24 and sw.ceil_above_01(0.2345) == 0.24


# ----- readouts ----------------------------------------------------------------------------------
def test_readouts_primal_and_dual_are_one_estimator_and_folds_are_grouped():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(10), 6)
    fold = sw.fold_assignment(groups)
    assert all(len(set(fold[groups == g])) == 1 for g in range(10))
    w = rng.normal(size=(8, 4))
    x = rng.normal(size=(60, 8))
    y = x @ w + 0.01 * rng.normal(size=(60, 4))
    primal = sw.fit_readouts(x, y, groups)
    assert primal["cube"]["form"] == "primal"
    wide = np.concatenate([x, np.zeros((60, 60))], axis=1)  # D > N: the dual form
    dual = sw.fit_readouts(wide, y, groups)
    assert dual["cube"]["form"] == "dual"
    for lab in sw.LABELS:
        assert dual[lab]["penalty"] == primal[lab]["penalty"]
        assert np.allclose(sw.apply_readout(dual[lab], wide), sw.apply_readout(primal[lab], x))
    e = sw.label_errors("cube", sw.apply_readout(primal["cube"], x), y[:, :3])
    assert e.shape == (60,) and np.median(e) < 0.1


def test_label_errors_units():
    assert sw.label_errors("cube", np.array([[3.0, 4.0, 0.0]]), np.zeros((1, 3)))[0] == 5.0
    assert sw.label_errors("jaw", np.array([[0.5]]), np.array([0.2]))[0] == pytest.approx(0.3)


# ----- subsets, grid rule, G5 bars ---------------------------------------------------------------
def test_moving_subset_rule_halves_once():
    out = sw.moving_thresholds_from_d0({"cube": 0.2, "jaw": 0.01})
    assert out["cube"]["threshold"] == 1.0 and not out["cube"]["halved"]
    assert out["jaw"]["threshold"] == 0.1 and out["jaw"]["halved"]
    m = sw.moving_mask(
        "cube", np.zeros((3, 3)), np.array([[1.0, 0, 0], [0.5, 0, 0], [0, 0, 2]]), 1.0
    )
    assert m.tolist() == [True, False, True]


def test_grid_rule_both_directions():
    assert sw.grid_rule(1.0, 0.85)["grid"] == "g8"
    assert sw.grid_rule(1.0, 0.86)["grid"] == "g4"
    assert sw.grid_rule(0.5, 0.1)["grid"] == "g4"  # g4 must exceed 0.5 cm
    assert sw.g5_bars({"cube": 0.4, "jaw": 0.02}) == {
        "B_cube": pytest.approx(0.6),
        "B_jaw": pytest.approx(0.03),
    }


# ----- G6 and the budget -------------------------------------------------------------------------
def test_g6_saturation_and_single_extension():
    flat = [[u, 1.0] for u in range(1000, 20001, 1000)]
    falling = [[u, 1.0 - 0.01 * i] for i, u in enumerate(range(1000, 20001, 1000))]
    assert sw.saturation(flat)["passes"]
    assert not sw.saturation(falling)["passes"]
    assert not sw.extension_needed({"a": flat, "b": flat})
    assert sw.extension_needed({"a": flat, "b": falling})


def test_budget_rule_lowers_u_then_escalates():
    curve = [[u, 1.0 if u >= 20000 else 2.0] for u in range(2000, 60001, 2000)]
    b = sw.budget_from_probe(curve, 0.06, used_gpu_h=6.0)
    assert b["u_star"] == 20000 and b["updates_by_rule"] == 30000
    assert not b["escalate"] and b["projection_gpu_h"]["sum"] <= 14.0
    slow = sw.budget_from_probe(curve, 0.5, used_gpu_h=6.0)
    assert slow["updates"] == 20000 and slow["escalate"]


# ----- gates and the row -------------------------------------------------------------------------
BARS = {"B_rank": 0.2, "B_std": 0.3, "B_cube": 1.0, "B_jaw": 0.1}


def _entry(good: bool) -> dict:
    s = {"value": 0.5, "ci": [0.4, 0.6 if good else 1.1]}
    return {
        "g1_stats": {"collapsed_fraction": 0.0, "effective_rank_ratio": 0.5, "std_ratio": 0.6},
        "g1_iii": {"ci": [0.01 if good else -0.01, 0.1]},
        "copy_ratio": s,
        "noaction_ratio": s,
        "sens": {"value": 1.3, "ci": [1.2 if good else 1.05, 1.4]},
        "g5": {
            lab: {
                "median": {"ci": [0.1, 0.5] if lab == "cube" else [0.01, 0.05]},
                "vs_copy": s,
                "vs_noaction": s,
            }
            for lab in sw.LABELS
        },
    }


def _gates(good_arm: str | None, *, g5: bool = True):
    gates = {}
    for arm in sw.ELIGIBLE:
        gates[arm] = {}
        for s in sw.MODEL_SEEDS:
            gates[arm][str(s)] = {}
            for h in sw.GATE_HORIZONS:
                e = _entry(arm == good_arm)
                if not g5:
                    e["g5"]["cube"]["median"]["ci"] = [2.0, 3.0]
                gates[arm][str(s)][str(h)] = sw.gates_for(e, BARS)
    return gates


def test_gates_both_directions():
    good = sw.gates_for(_entry(True), BARS)
    bad = sw.gates_for(_entry(False), BARS)
    assert good["G1_G5"] and good["G1_G4"]
    assert not bad["G1"]["iii"] and not bad["G2"] and not bad["G3"] and not bad["G4"]
    tight = dict(BARS, B_rank=0.6)
    assert not sw.gates_for(_entry(True), tight)["G1"]["ii_rank"]


def test_rows_first_match():
    seeds = [str(s) for s in sw.MODEL_SEEDS]
    g6_ok = {a: dict.fromkeys(seeds, True) for a in sw.ELIGIBLE}
    assert sw.decide(void_reasons=["x"], small_subsets=[], gates={}, g6={})["row"] == "VOID"
    assert sw.decide(void_reasons=[], small_subsets=["M"], gates={}, g6={})["row"] == "INCONCLUSIVE"
    row = sw.decide(void_reasons=[], small_subsets=[], gates=_gates("J"), g6=g6_ok)
    assert row["row"] == "WM-PASS" and row["arms"] == ["J"] and row["s5_model"] == "J"
    g6_bad = {a: dict.fromkeys(seeds, False) for a in sw.ELIGIBLE}
    assert (
        sw.decide(void_reasons=[], small_subsets=[], gates=_gates("L"), g6=g6_bad)["row"]
        == "WM-PARTIAL"
    )
    no_g5 = sw.decide(void_reasons=[], small_subsets=[], gates=_gates("L", g5=False), g6=g6_ok)
    assert no_g5["row"] == "WM-PARTIAL" and no_g5["arms"] == ["L"]
    assert (
        sw.decide(void_reasons=[], small_subsets=[], gates=_gates(None), g6=g6_ok)["row"]
        == "WM-FAIL"
    )


# ----- the feature store and the test guard -------------------------------------------------------
def _store(tmp_path, split="d0", episodes=((0, 991000), (1, 991104), (2, 991208)), frames=41):
    folder = tmp_path / "g4"
    rng = np.random.default_rng(5)
    w = sd.SplitWriter(folder, split, frames * len(episodes), "g4")
    for shard, seed in episodes:
        ep = {
            "episode_id": f"so101-play-{seed}",
            "actions": rng.uniform(-1, 1, (frames, 6)).astype(np.float32),
            "jaw": rng.normal(size=frames).astype(np.float32),
            "cube": rng.normal(size=(frames, 3)).astype(np.float32),
            "gripper_target": rng.normal(size=frames).astype(np.float32),
            "phase": np.zeros(frames, np.int8),
            "visible": np.ones((frames, 2), bool),
        }
        w.add(
            ep,
            rng.normal(size=(frames, 32, 384)).astype(np.float32),
            shard=f"shard-{shard:02d}",
            seed=seed,
        )
    files = w.close()
    (folder / "report.json").write_text(
        json.dumps({"grid": "g4", "encoder": "pretrained", "files_sha256": {split: files}})
    )
    return folder, files


def test_store_views_moments_and_test_guard(tmp_path):
    folder, files = _store(tmp_path)
    sd.verify_store_files(folder, {"d0": files}, ("d0",))
    ident = sd.Moments(np.zeros(sw.LATENT_DIM), np.ones(sw.LATENT_DIM))
    store = sd.FeatureStore(folder, "d0", moments=ident)
    fit = sd.open_view(folder, "d0-fit", ident)
    assert fit.seeds.tolist() == [991000, 991104]
    ev = sd.open_view(folder, "d0-eval", ident)
    assert ev.seeds.tolist() == [991208]
    m = sd.Moments.fit(store, fit)
    raw = store.raw(np.arange(82))
    assert np.allclose(m.mean, raw.astype(np.float64).mean(0), atol=1e-5)
    assert (m.std >= sw.FEATURE_STD_FLOOR).all()
    batch = store.gather(np.array([0, 41]), frames=17)
    assert batch["latents"].shape == (2, 17, sw.LATENT_DIM) and batch["actions"].shape == (2, 16, 6)
    with pytest.raises(ContractError, match="G-test"):
        sd.open_view(folder, "test", ident)
    with pytest.raises(FileExistsError):
        sd.SplitWriter(folder, "d0", 1, "g4")


def test_fit_sample_view_is_train_seeds_divisible_by_four(tmp_path):
    folder, _ = _store(tmp_path, split="train", episodes=((0, 991000), (0, 991001), (1, 991004)))
    ident = sd.Moments(np.zeros(sw.LATENT_DIM), np.ones(sw.LATENT_DIM))
    assert sd.open_view(folder, "fit-sample", ident).seeds.tolist() == [991000, 991004]


def test_sampler_is_the_declared_stream_and_resumes():
    a = sd.WindowSampler(np.arange(100), 100100)
    rng = np.random.default_rng(np.random.SeedSequence([10011, 100100]))
    assert np.array_equal(a.draw(), np.arange(100)[rng.integers(0, 100, 64)])
    state = a.state()
    nxt = a.draw()
    b = sd.WindowSampler(np.arange(100), 100100)
    b.restore(state)
    assert np.array_equal(b.draw(), nxt) and b.draws == 2
    assert sw.sampler_stream_for("N_L", sw.CAL_N_SEED) == sw.CAL_L_SEED
    assert sw.sampler_stream_for("N_L", 100101) == 100101


def test_projection_accumulator_matches_a_direct_fit():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(12, 128, 384)) * rng.uniform(0.5, 2.0, 384)
    acc = sd.ProjectionAccumulator()
    acc.add(x[:5])
    acc.add(x[5:])
    p = acc.fit()
    z = (x - x.mean(0)) / x.std(0)
    cov = np.einsum("npi,npj->ij", z, z) / (12 * 128)
    top = np.sort(np.linalg.eigvalsh(cov))[::-1][:96]
    assert np.allclose(np.sort(p["eigenvalues"])[::-1][:96], top, atol=1e-8)
    assert sd.project_g8(x, p).shape == (12, 128, 96)


# ----- the frozen-token class, both backends ------------------------------------------------------
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(autouse=True)
def bounded_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(int(os.environ.get("JEPA_TEST_THREADS", "4")))
    yield
    torch.set_num_threads(original)


def _tiny_dinov2():
    from transformers import Dinov2Config, Dinov2Model

    config = Dinov2Config(
        hidden_size=384,
        num_hidden_layers=1,
        num_attention_heads=6,
        mlp_ratio=1,
        patch_size=14,
        image_size=224,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        return Dinov2Model(config).float().eval()


@pytest.fixture
def sft(monkeypatch):
    from embodied_jepa.models import schema_frozen_tokens as module

    monkeypatch.setattr(
        module.SchemaFrozenTokenMixin, "_load_frozen_module", lambda s, n: _tiny_dinov2()
    )
    return module


@pytest.fixture(params=["native_jepa", "leworldmodel"])
def backend(request):
    if request.param == "leworldmodel" and not (ROOT / "third_party/le-wm/jepa.py").exists():
        pytest.skip("optional pinned LeWM source absent; run scripts/fetch_lewm.py")
    return request.param


def _model(sft, backend, *, tokens=4, width=24, seed=1, metadata=None):
    from embodied_jepa import so101_wm_runtime as runtime
    from embodied_jepa.action_schemas import JOINT_DELTA_SO101_V0

    config = {
        "frozen_encoder": "dinov2_small_224",
        "token_count": tokens,
        "token_width": width,
        "latent_dim": tokens * width,
        "camera": "cached_features",
        "max_horizon": 32,
        "hidden_dim": 16,
    }
    if backend == "leworldmodel":
        config |= {"sigreg_weight": 0.0, "predictor_heads": 2, "predictor_head_dim": 8}
    return sft.create(
        backend,
        state_schema=runtime.state_schema(),
        action_schema=JOINT_DELTA_SO101_V0,
        config=config,
        seed=seed,
        metadata=metadata or {"task": sw.TASK},
    )


def test_six_d_schema_through_the_frozen_token_class(sft, backend):
    m = _model(sft, backend)
    rng = np.random.default_rng(8)
    f = rng.normal(size=(4, 5, 96)).astype(np.float32)
    a = rng.uniform(-1, 1, (4, 4, 6)).astype(np.float32)
    before = sft.state_sha256(m)
    out = m.train_step_features(f, a)
    assert np.isfinite(out["loss"]) and sft.state_sha256(m) != before
    pred = m.predict_at(f[:, 0], rng.uniform(-1, 1, (4, 32, 6)).astype(np.float32), (1, 8, 32))
    assert pred[32].shape == (4, 96)
    with pytest.raises(ContractError):
        m.train_step_features(f, rng.uniform(-1, 1, (4, 4, 14)).astype(np.float32))
    assert m.capabilities.action_schema.dimension == 6


def test_the_backend_swap_is_one_key(sft):
    from embodied_jepa import so101_wm as s

    a, b = s.model_config("L", "g4"), s.model_config("J", "g4")
    assert {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)} == {"sigreg_weight"}
    assert s.check_arm("L")["backend"] != s.check_arm("J")["backend"]


def test_checkpoints_refuse_other_implementation_hashes(sft, backend, tmp_path, monkeypatch):
    m = _model(sft, backend)
    m.save(tmp_path / "m.pt")
    again = _model(sft, backend, seed=2)
    again.load(tmp_path / "m.pt")
    assert sft.state_sha256(again) == sft.state_sha256(m)
    cls = type(again)
    monkeypatch.setattr(cls, "implementation_sha256", property(lambda self: "0" * 64))
    with pytest.raises(ContractError, match="implementation_sha256"):
        _model(sft, backend).load(tmp_path / "m.pt")
    other = _model(sft, backend, metadata={"task": "other"})
    monkeypatch.undo()
    monkeypatch.setattr(
        sft.SchemaFrozenTokenMixin, "_load_frozen_module", lambda s, n: _tiny_dinov2()
    )
    with pytest.raises(ContractError, match="provenance"):
        other.load(tmp_path / "m.pt")


def test_implementation_hash_covers_composed_files(sft, backend):
    m = _model(sft, backend)
    from embodied_jepa.models import schema_frozen_tokens

    assert len(m.implementation_sha256) == 64
    assert Path(schema_frozen_tokens.__file__).name == "schema_frozen_tokens.py"


# ----- pins and imports --------------------------------------------------------------------------
def test_composed_files_are_byte_identical_to_their_pins():
    assert all(sw.composed_files_check(ROOT).values()), sw.composed_files_check(ROOT)


def test_g1_byte_pinned_files_match_origin_main_when_available():
    """Only additions against origin/main (when git and origin/main are available)."""
    try:
        base = subprocess.run(
            ["git", "merge-base", "HEAD", "origin/main"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--name-status", base, "--", "src", "configs"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("git or origin/main unavailable")
    changed = [line for line in diff.splitlines() if line and not line.startswith("A")]
    assert not changed, changed


def test_import_embodied_jepa_stays_torch_free():
    code = (
        "import sys, embodied_jepa, embodied_jepa.so101_wm, embodied_jepa.so101_wm_data; "
        "print('torch' in sys.modules or 'mujoco' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert out == "False"


def test_seeds_and_salts_are_declared():
    assert sw.SALTS == {
        "sampler": 10011,
        "wrong": 10012,
        "candidates": 10013,
        "bootstrap": 10014,
        "folds": 10015,
    }
    assert sw.MODEL_SEEDS == (100100, 100101, 100102)
    for seed in (sw.CAL_L_SEED, sw.CAL_N_SEED, sw.PROBE_SEED, *sw.DEBUG_SEEDS):
        assert seed in sw.DEV_SEEDS
    with pytest.raises(ContractError):
        sw.check_seed(100900)


def test_twin_seed_pairs_calibration_l_with_calibration_n_l():
    assert sw.twin_seed(sw.CAL_L_SEED) == sw.CAL_N_SEED
    assert sw.twin_seed(100101) == 100101


def test_frozen_manifest_pins_match_the_tree_and_the_module():
    from embodied_jepa import so101_wm_runtime as runtime

    manifest = json.loads(runtime.MANIFEST.read_text())
    assert manifest["status"] == "FROZEN"
    check = runtime.frozen_files_check(ROOT)
    assert check and all(check.values()), [k for k, v in check.items() if not v]
    assert manifest["composed_files_sha256"] == sw.COMPOSED_FILES_SHA256
    assert manifest["choices"] == sw.frozen_choices()
