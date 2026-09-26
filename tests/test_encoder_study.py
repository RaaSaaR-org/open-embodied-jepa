"""TASK-062 encoder study: cross-fitting, guards, floors, Holm, arm states, decision, study step.

Synthetic evidence only; these tests make no learned-representation or control claims.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_jepa import encoder_study as es
from embodied_jepa import info_ceiling as ic
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "benchmarks" / "manifests" / "apple-encoder-study-v1.json").read_text()
)


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----- the manifest and the module agree ------------------------------------------------------
def test_manifest_arms_floors_and_thresholds_match_the_module():
    arms = MANIFEST["arms_decisional_in_holm_order"]
    assert tuple(a["arm"] for a in arms) == es.ARMS
    floors = {"F-tok": "F-tok", "F-cls": "F-cls", "F-plain": "F-plain"}
    assert {a["arm"]: floors[a["floor"]] for a in arms} == es.ARM_FLOOR
    thresholds = [es.ALPHA_ONE_SIDED / (len(es.ARMS) - j) for j in range(len(es.ARMS))]
    assert np.allclose(MANIFEST["multiplicity"]["step_thresholds"], thresholds)
    budget = MANIFEST["budget"]
    assert budget["per_encoder_seconds"] == es.PER_ENCODER_SECONDS
    assert budget["global_wall_seconds"] == es.GLOBAL_WALL_SECONDS
    assert budget["encoders"] == 2 * len(es.MODELS)
    seeds = MANIFEST["seeds"]
    assert seeds["window_sampler"] == {"e_A": 6200, "e_B": 6201} and es.SAMPLER_SEED == {
        "A": 6200,
        "B": 6201,
    }
    assert seeds["rec_decoder"] == es.DECODER_SEED
    assert seeds["collapse_sample"] == es.COLLAPSE_SEED
    assert seeds["action_sensitivity_sample"] == es.SENSITIVITY_SEED
    digests = MANIFEST["training_recipe"]["seed0_init_digests"]
    assert es.SEED0_DIGEST == {
        True: digests["readout_heads_on"],
        False: digests["readout_heads_off"],
    }
    assert es.MODEL_OVERRIDES["PLAIN"] == {"readout_heads": False}
    assert es.IMAGE_SIGREG_WEIGHT == {"SIG": 1.0} and es.RECONSTRUCTION_WEIGHT == {"REC": 1.0}
    assert [r["row"] for r in MANIFEST["pre_declared_outcomes_in_order"]] == list(es.ROWS)


def test_manifest_pins_the_unchanged_probe_code_and_the_task061_report():
    hashes = MANIFEST["hashes"]
    for name in (
        "scripts/probe_observation_reprobe.py",
        "src/embodied_jepa/observation_reprobe.py",
        "scripts/probe_info_ceiling.py",
        "src/embodied_jepa/info_ceiling.py",
        "scripts/calibrate_encoder_study.py",
        "src/embodied_jepa/models/base.py",
        "src/embodied_jepa/models/lewm.py",
        "outputs/task061-observation-reprobe/run-1/report.json",
    ):
        assert name in hashes
    for name in ("scripts/probe_info_ceiling.py", "src/embodied_jepa/info_ceiling.py"):
        path = ROOT / name
        assert es.hashlib.sha256(path.read_bytes()).hexdigest() == hashes[name]


# ----- cross-fitting (§4) and G-split ---------------------------------------------------------
def synthetic_plan(n=40):
    """n roots (every 5th is val), 3 branches each, plus a test root with branches."""
    fold = ic.fold_of(n)
    roots = [
        {"episode_id": f"r{i}", "split": "val" if i % 5 == 0 else "train", "seed": 48000 + i}
        for i in range(n)
    ]
    episodes, splits = [], {"train": [], "val": [], "test": [], "holdout": []}
    for r in roots + [{"episode_id": "t0", "split": "test"}]:
        for b in range(4):
            name = r["episode_id"] if b == 0 else f"{r['episode_id']}-b{b}"
            episodes.append({"episode_id": name, "metadata": {"root_episode_id": r["episode_id"]}})
            splits[r["split"]].append(name)
    return roots, fold, {"episodes": episodes, "splits": splits}


def test_halves_follow_the_task061_folds():
    fold = np.arange(10)
    assert list(es.half_of_fold(fold)) == ["A"] * 5 + ["B"] * 5
    assert es.other("A") == "B" and es.other("B") == "A"
    with pytest.raises(ContractError):
        es.other("C")
    with pytest.raises(ContractError):
        es.half_of_fold(np.array([10]))


def test_each_encoder_trains_only_on_the_other_halfs_train_roots():
    roots, fold, manifest = synthetic_plan()
    halves = es.half_of_fold(fold)
    by_id = {r["episode_id"]: (r, h) for r, h in zip(roots, halves, strict=True)}
    for half in es.HALVES:
        chosen = es.training_roots(roots, fold, half)
        assert chosen and all(
            by_id[c][1] != half and by_id[c][0]["split"] == "train" for c in chosen
        )
        episodes = es.training_episodes(manifest, chosen)
        assert len(episodes) == 4 * len(chosen)
        assert not any(e.startswith("t0") for e in episodes)


def _episodes(roots, fold, manifest):
    return {h: es.training_episodes(manifest, es.training_roots(roots, fold, h)) for h in es.HALVES}


def test_g_split_accepts_the_cross_fitted_episodes():
    roots, fold, manifest = synthetic_plan()
    summary = es.check_training_split(manifest, roots, fold, _episodes(roots, fold, manifest))
    train_roots = sum(r["split"] == "train" for r in roots)
    assert summary["A"]["roots"] + summary["B"]["roots"] == train_roots


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("val_episode", "not a train-split episode"),
        ("test_episode", "not a train-split episode"),
        ("own_half", "a root it featurises"),
        ("drop_branch", "does not use every episode"),
        ("duplicate", "duplicate"),
        ("swap_halves", "a root it featurises"),
        ("empty", "no training episodes"),
    ],
)
def test_g_split_refuses_every_violation(mutation, message):
    roots, fold, manifest = synthetic_plan()
    episodes = _episodes(roots, fold, manifest)
    if mutation == "val_episode":
        episodes["A"] = episodes["A"] + [manifest["splits"]["val"][0]]
    elif mutation == "test_episode":
        episodes["A"] = episodes["A"] + ["t0"]
    elif mutation == "own_half":
        own = es.training_episodes(manifest, es.training_roots(roots, fold, "B"))[0]
        episodes["A"] = episodes["A"] + [own] if own not in episodes["A"] else episodes["A"]
        episodes["B"] = [e for e in episodes["B"] if e != own]
    elif mutation == "drop_branch":
        episodes["A"] = episodes["A"][:-1]
    elif mutation == "duplicate":
        episodes["A"] = episodes["A"] + episodes["A"][:1]
    elif mutation == "swap_halves":
        episodes = {"A": episodes["B"], "B": episodes["A"]}
    elif mutation == "empty":
        episodes["A"] = []
    with pytest.raises(es.GuardError, match=message):
        es.check_training_split(manifest, roots, fold, episodes)


def _features(n=60, d=12, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    y = np.stack([x[:, 0] + 0.1 * rng.normal(size=n), x[:, 1] - x[:, 2]], axis=1)
    return x, y


def test_crossfit_cv_with_one_encoder_equals_info_ceiling_nested_cv():
    x, y = _features()
    fold = ic.fold_of(len(x))
    g, diag = ic.gram(x)
    want, _fits, want_sel = ic.nested_cv(g, diag, y, fold)
    got, _fits, got_sel = es.crossfit_nested_cv({"A": (g, diag), "B": (g, diag)}, y, fold)
    assert np.array_equal(want, got)
    assert [{k: v for k, v in s.items() if k != "encoder_half"} for s in got_sel] == want_sel


def test_crossfit_cv_reads_each_fold_from_the_encoder_that_excluded_it():
    x, y = _features()
    fold = ic.fold_of(len(x))
    halves = es.half_of_fold(fold)
    g, diag = ic.gram(x)
    base, _f, _s = es.crossfit_nested_cv({"A": (g, diag), "B": (g, diag)}, y, fold)
    g2, diag2 = ic.gram(x + np.random.default_rng(9).normal(size=x.shape))
    mixed, _f, selections = es.crossfit_nested_cv({"A": (g, diag), "B": (g2, diag2)}, y, fold)
    assert np.array_equal(mixed[halves == "A"], base[halves == "A"])
    assert not np.allclose(mixed[halves == "B"], base[halves == "B"])
    assert all(s["encoder_half"] == halves[fold == s["outer_fold"]][0] for s in selections)


def test_hidden_features_equal_to_the_originals_reproduce_the_out_of_fold_predictions():
    x, y = _features()
    fold = ic.fold_of(len(x))
    grams = {h: ic.gram(x) for h in es.HALVES}
    pred, fits, _s = es.crossfit_nested_cv(grams, y, fold)
    hidden = {h: ic.cross_gram(x, x) for h in es.HALVES}
    again = es.crossfit_predict_hidden(fits, fold, hidden)
    assert np.allclose(again, pred, atol=1e-9)


def test_crossfit_secondary_predicts_every_val_root_from_the_train_roots():
    x, y = _features()
    fold = ic.fold_of(len(x))
    train = np.arange(len(x)) % 7 != 0
    grams = {h: ic.gram(x) for h in es.HALVES}
    out, selections = es.crossfit_split(grams, y, fold, train)
    assert np.isfinite(out[~train]).all() and np.isnan(out[train]).all()
    assert set(selections) <= set(es.HALVES)
    single, _r, _c = ic.split_fit(*ic.gram(x), y, train)
    assert np.allclose(out[~train], single)


# ----- floor, p-values, Holm, arm states, rows (§6-§10) ----------------------------------------
def _kept(errors, correct):
    return {"_errors_cm": np.asarray(errors, float), "T2": {"accuracy": float(np.mean(correct))}}


def test_floor_pvalue_both_directions():
    idx = ic.bootstrap_indices(190)
    rng = np.random.default_rng(1)
    floor = rng.uniform(1.0, 2.0, 190)
    assert es.floor_pvalue(floor * 0.3, floor, idx) == 0.0
    assert es.floor_pvalue(floor, floor, idx) == 1.0  # a tie counts against the arm
    assert es.floor_pvalue(floor * 1.5, floor, idx) == 1.0


def test_floor_pvalue_agrees_with_beats_random_floor_on_the_same_draws():
    rng = np.random.default_rng(2)
    floor_err = rng.uniform(0.5, 2.0, 190)
    for scale in (0.6, 0.9, 0.97, 1.0, 1.1):
        arm_err = floor_err * scale + rng.normal(0, 0.05, 190)
        arm = {"_errors_cm": arm_err, "T2": {"accuracy": 0.95}, "_correct": np.ones(190, bool)}
        flo = {"_errors_cm": floor_err, "T2": {"accuracy": 0.9}, "_correct": np.ones(190, bool)}
        verdict = ic.beats_random_floor(arm, flo)
        p = es.floor_pvalue(arm_err, floor_err, ic.bootstrap_indices(190))
        if verdict["beats_random_floor"]:
            assert p <= 0.025 + 1e-12
        if p < 0.02:
            assert verdict["median_error_difference_cm"]["ci95"][1] < 0


def _perfect_readout(n=190, seed=3):
    rng = np.random.default_rng(seed)
    xy = rng.normal(0.3, 0.02, (n, 2))
    dx = rng.choice([-0.4, 0.4], n)
    occ = rng.random(n) < 0.5
    fold = ic.fold_of(n)
    priors = ic.prior_predictions(xy, dx, occ, fold, dy=np.zeros(n))
    return xy, dx, priors


def test_arm_pvalues_need_the_floor_point_condition():
    xy, dx, priors = _perfect_readout()
    xy_pred = xy + 0.0005
    arm = ic.evaluate(xy_pred, dx, xy, dx, priors, np.ones(len(xy), bool))
    weak = ic.evaluate(xy + 0.02, -dx, xy, dx, priors, np.ones(len(xy), bool))
    good = es.arm_pvalues(xy_pred, dx, xy, dx, priors, weak, arm)
    assert good["point_conditions_hold"] and good["p"] < 0.00625 and good["p_F"] == 0.0
    same = es.arm_pvalues(xy_pred, dx, xy, dx, priors, arm, arm)
    assert not same["T2_higher_than_floor"] and same["p"] == 1.0


def test_holm_over_four_arms_both_directions():
    passed = es.holm({"A-tok": 0.001, "A-sig": 0.008, "A-rec": 0.012, "A-plain": 0.02})
    assert all(passed["rejected"].values())
    assert [s["threshold"] for s in passed["steps"]] == pytest.approx(
        [0.00625, 0.025 / 3, 0.0125, 0.025]
    )
    blocked = es.holm({"A-tok": 0.02, "A-sig": 0.0001, "A-rec": 0.01, "A-plain": 0.013})
    assert blocked["rejected"]["A-sig"]
    # A-rec fails the second step (0.01 > 0.00833), so the step-down stops there: A-tok's 0.02
    # would clear the last threshold on its own, but it is not rejected.
    assert not any(blocked["rejected"][a] for a in ("A-rec", "A-plain", "A-tok"))
    ties = es.holm({a: 0.001 for a in es.ARMS})
    assert [s["hypothesis"] for s in ties["steps"]] == list(es.ARMS)


def test_arm_states_and_pass_rule():
    assert es.arm_state(trained=False, not_collapsed=False) == "not evaluated"
    assert es.arm_state(trained=True, not_collapsed=False) == "collapse-demoted"
    assert es.arm_state(trained=True, not_collapsed=True) == "evaluated"
    keys = ("trained", "not_collapsed", "succeeds", "beats_floor", "rejected")
    assert es.passes(**dict.fromkeys(keys, True), spurious=False)
    for missing in keys:
        assert not es.passes(**(dict.fromkeys(keys, True) | {missing: False}), spurious=False)
    assert not es.passes(**dict.fromkeys(keys, True), spurious=True)


def test_collapse_gate_uses_task054_values_and_clears_e0_not_the_init():
    reference = MANIFEST["calibration"]["collapse_reference"]
    assert es.collapse_ok(reference["E0"]["image_feature"])
    assert not es.collapse_ok(reference["random"]["image_feature"])
    edge = {"collapsed_fraction": 0.05, "effective_rank": 2.0, "latent_std_mean": 0.1}
    assert es.collapse_ok(edge)
    for key, bad in (
        ("collapsed_fraction", 0.051),
        ("effective_rank", 1.99),
        ("latent_std_mean", 0.099),
    ):
        assert not es.collapse_ok(edge | {key: bad})


def test_qualifiers():
    q = es.qualifier
    assert (
        q(passes=True, succeeds=True, beats_floor=True, rejected=True, beats_prior=True) == "pass"
    )
    assert q(
        passes=False, succeeds=True, beats_floor=True, rejected=False, beats_prior=True
    ).startswith("unadjusted")
    assert q(passes=False, succeeds=True, beats_floor=False, rejected=False, beats_prior=True) == (
        "meets the bars, not the floor"
    )
    assert "collapse-demoted" in q(
        passes=False, succeeds=True, beats_floor=True, rejected=True, beats_prior=True
    )
    assert q(passes=False, succeeds=False, beats_floor=False, rejected=False, beats_prior=True) == (
        "partial information"
    )


def _arm(passes=False, state="evaluated", succeeds=False, spurious=False, p=1.0):
    return {"passes": passes, "state": state, "succeeds": succeeds, "spurious": spurious, "p": p}


@pytest.mark.parametrize(
    "arms, row",
    [
        ({"A-sig": _arm(True, p=0.004), "A-tok": _arm(True, p=0.001)}, "O-ENC-LATENT"),
        ({"A-plain": _arm(True, p=0.002)}, "O-ENC-LATENT"),
        ({"A-tok": _arm(True, p=0.001)}, "O-ENC-TOKENS"),
        ({"A-rec": _arm(state="not evaluated"), "A-tok": _arm(succeeds=True)}, "O-ENC-INCOMPLETE"),
        ({"A-tok": _arm(succeeds=True)}, "O-ENC-ARCH"),
        ({"A-sig": _arm(state="collapse-demoted", succeeds=True)}, "O-ENC-ARCH"),
        ({"A-tok": _arm(succeeds=True, spurious=True)}, "O-ENC-NONE"),
        ({}, "O-ENC-NONE"),
    ],
)
def test_decision_rows_in_order(arms, row):
    full = {a: _arm() for a in es.ARMS} | arms
    decision = es.decide(void=False, arms=full)
    assert decision["outcome"] == row
    assert es.abandonment_fires(row) == (row in ("O-ENC-ARCH", "O-ENC-NONE"))


def test_latent_choice_is_the_smallest_p_then_the_listed_order():
    full = {a: _arm() for a in es.ARMS}
    full |= {"A-rec": _arm(True, p=0.001), "A-sig": _arm(True, p=0.004)}
    assert es.decide(void=False, arms=full)["encoder_recipe"] == "A-rec"
    full |= {
        "A-rec": _arm(True, p=0.001),
        "A-sig": _arm(True, p=0.001),
        "A-plain": _arm(True, p=0.001),
    }
    assert es.decide(void=False, arms=full)["encoder_recipe"] == "A-sig"


def test_void_precedes_every_row_and_missing_arms_are_refused():
    full = {a: _arm(True, p=0.001) for a in es.ARMS}
    assert es.decide(void=True, arms=full) == {"outcome": "V"}
    with pytest.raises(ContractError):
        es.decide(void=False, arms={"A-tok": _arm()})


# ----- G-anchor (§9) ------------------------------------------------------------------------
def _anchor_views():
    view = {
        "results": {
            "all": {"T1": {"median": 0.5, "pass": True}, "T2": {"correct": 180, "n": 190}},
            "selections": {"xy": [{"family": "rbf", "lam_rel": 0.01, "inner_mse": 1e-4}]},
        },
        "per_root": [{"xy": [0.3, 0.1], "dx": 0.4, "dy": -0.2}],
    }
    return {s: copy.deepcopy(view) for s in es.ANCHOR_SOURCES}


def test_g_anchor_exact():
    comparison = es.anchor_compare(_anchor_views(), _anchor_views())
    assert comparison["verdict"] == "exact" and comparison["max_abs_delta"] == 0.0
    es.check_anchor(comparison)


def test_g_anchor_tiny_numeric_difference_is_a_caveat_not_a_void():
    new = _anchor_views()
    new["L_E0"]["per_root"][0]["dx"] += 1e-9
    comparison = es.anchor_compare(_anchor_views(), new)
    assert comparison["verdict"] == "caveat" and 0 < comparison["max_abs_delta"] <= 1e-6
    es.check_anchor(comparison)


@pytest.mark.parametrize(
    "edit",
    [
        lambda v: v["L_raw"]["results"]["all"]["T2"].__setitem__("correct", 179),
        lambda v: v["L_raw"]["results"]["all"]["T1"].__setitem__("pass", False),
        lambda v: v["L_random"]["results"]["selections"]["xy"][0].__setitem__("family", "linear"),
        lambda v: v["L_random"]["results"]["selections"]["xy"][0].__setitem__("lam_rel", 0.1),
        lambda v: v["L_E0"]["per_root"][0].__setitem__("dx", 0.4 + 1e-5),
        lambda v: v["L_E0"]["per_root"].append({"xy": [0, 0], "dx": 0, "dy": 0}),
        lambda v: v["L_E0"]["results"]["all"].pop("T1"),
    ],
)
def test_g_anchor_voids_on_any_discrete_or_large_difference(edit):
    new = _anchor_views()
    edit(new)
    comparison = es.anchor_compare(_anchor_views(), new)
    assert comparison["verdict"] == "void"
    with pytest.raises(es.GuardError, match="G-anchor"):
        es.check_anchor(comparison)


def test_g_anchor_needs_every_anchor_source():
    views = _anchor_views()
    views.pop("L_random")
    with pytest.raises(es.GuardError):
        es.anchor_compare(_anchor_views(), views)


def test_anchor_view_reads_a_task061_style_report():
    results = {"L_raw": {"all": {"x": 1}, "selections": {}, "T4_reported": {"y": 2}}}
    per_root = [{"predictions": {"L_raw": {"xy": [1, 2], "dx": 0.1, "dy": 0.2}}}]
    view = es.anchor_view(results, per_root, "L_raw")
    assert view == {
        "results": {"all": {"x": 1}, "selections": {}},
        "per_root": [{"xy": [1, 2], "dx": 0.1, "dy": 0.2}],
    }


# ----- normalisation and small helpers ----------------------------------------------------------
def test_welford_moments_match_the_masked_moments():
    rng = np.random.default_rng(5)
    states = rng.normal(size=(400, 6)).astype(np.float32)
    states[:, 5] = 3.0  # constant -> std 1
    mask = rng.random((400, 6)) < 0.9
    mean, std = es.welford_moments(states, mask)
    for j in range(5):
        values = states[mask[:, j], j].astype(np.float64)
        assert mean[j] == pytest.approx(values.mean(), abs=1e-9)
        assert std[j] == pytest.approx(values.std(), abs=1e-9)
    assert std[5] == 1.0


def test_derangement_has_no_fixed_point():
    rng = np.random.default_rng(0)
    for n in (2, 3, 10, 512):
        perm = es.derangement(n, rng)
        assert sorted(perm) == list(range(n)) and not (perm == np.arange(n)).any()
    with pytest.raises(ContractError):
        es.derangement(1, rng)


def test_summary_json_drops_private_arrays():
    assert es.summary_json({"a": 1, "_b": 2, "c": [{"_d": 3, "e": 4}]}) == {"a": 1, "c": [{"e": 4}]}


# ----- torch: G-init, the study step, the decoder ---------------------------------------------
def _lewm_source_or_skip():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not (ROOT / "third_party" / "le-wm" / "jepa.py").exists():
        pytest.skip("optional pinned LeWM source absent; run scripts/fetch_lewm.py")


def _tiny_lewm(seed=0, **overrides):
    from embodied_jepa.contracts import StateSchema
    from embodied_jepa.models import LeWM

    schema = StateSchema(("arm.q",), ("rad",), "fixture_v0")
    config = {
        "image_size": 28,
        "latent_dim": 16,
        "hidden_dim": 32,
        "encoder_depth": 1,
        "encoder_heads": 2,
        "predictor_depth": 1,
        "predictor_heads": 2,
        "predictor_head_dim": 8,
        "sigreg_projections": 16,
        "source_path": str(ROOT / "third_party" / "le-wm"),
    } | overrides
    return LeWM(schema, seed=seed, config=config, metadata={"fixture": "encoder-study"})


def _batch(size=28, seed=42):
    from embodied_jepa.contracts import SequenceBatch, StateSchema

    rng = np.random.default_rng(seed)
    schema = StateSchema(("arm.q",), ("rad",), "fixture_v0")
    return SequenceBatch(
        observations={"onboard_rgb": rng.integers(0, 256, (4, 3, size, size, 3), dtype=np.uint8)},
        robot_states=rng.normal(size=(4, 3, 1)).astype(np.float32),
        state_mask=np.ones((4, 3, 1), dtype=np.bool_),
        actions=rng.uniform(-1, 1, (4, 2, 14)).astype(np.float32),
        timestamps=np.tile(np.arange(3, dtype=np.float64), (4, 1)),
        terminated=np.zeros((4, 2), dtype=np.bool_),
        episode_ids=("a", "b", "c", "d"),
        state_schema=schema,
    )


def _params_equal(a, b):
    import torch

    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_study_step_with_both_terms_off_is_lewm_train_step_exactly():
    _lewm_source_or_skip()
    batch = _batch()
    reference, study = _tiny_lewm(), _tiny_lewm()
    assert _params_equal(reference, study)
    for _ in range(3):
        want = reference.train_step(batch)
        got = es.study_train_step(study, batch, None)
        for key in ("loss", "prediction_loss", "multistep_loss", "sigreg_loss", "gradient_norm"):
            assert got[key] == want[key]
        assert got["image_sigreg_loss"] == 0.0 and got["reconstruction_loss"] == 0.0
    assert _params_equal(reference, study)
    assert reference.updates == study.updates == 3


def test_study_step_with_image_sigreg_changes_the_update():
    _lewm_source_or_skip()
    batch = _batch()
    reference, study = _tiny_lewm(), _tiny_lewm()
    reference.train_step(batch)
    got = es.study_train_step(study, batch, None, image_sigreg_weight=1.0)
    assert got["image_sigreg_loss"] > 0
    assert not _params_equal(reference, study)


def test_study_step_reconstruction_trains_the_decoder_and_needs_one():
    _lewm_source_or_skip()
    import torch

    model = _tiny_lewm(image_size=112, patch_size=14, latent_dim=128, encoder_heads=4)
    decoder = es.make_decoder(128)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3)
    before = copy.deepcopy(decoder.state_dict())
    batch = _batch(size=112)
    with pytest.raises(ContractError, match="decoder"):
        es.study_train_step(model, batch, None, reconstruction_weight=1.0)
    metrics = es.study_train_step(
        model, batch, None, decoder=decoder, decoder_optimizer=optimizer, reconstruction_weight=1.0
    )
    assert metrics["reconstruction_loss"] > 0
    after = decoder.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)


def test_decoder_is_seeded_and_leaves_the_global_rng_alone():
    torch = pytest.importorskip("torch")
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    first = es.make_decoder(128)
    after = torch.rand(3)
    assert torch.equal(expected, after)  # the forked RNG did not move the global stream
    second = es.make_decoder(128)
    assert all(
        torch.equal(a, b)
        for a, b in zip(first.state_dict().values(), second.state_dict().values(), strict=True)
    )
    out = first(torch.zeros(2, 128))
    assert tuple(out.shape) == (2, 3, 112, 112)
    with pytest.raises(ContractError):
        es.make_decoder(128, image_size=56)


def test_g_init_both_directions():
    torch = pytest.importorskip("torch")
    module = torch.nn.Linear(3, 2)
    with pytest.raises(es.GuardError, match="G-init"):
        es.check_init(module, readout_heads=True)
    original = dict(es.SEED0_DIGEST)
    try:
        es.SEED0_DIGEST[True] = es.weights_digest(module)
        assert es.check_init(module, readout_heads=True) == es.SEED0_DIGEST[True]
    finally:
        es.SEED0_DIGEST.clear()
        es.SEED0_DIGEST.update(original)


def test_the_real_seed0_inits_have_the_pinned_digests():
    _lewm_source_or_skip()
    from embodied_jepa.config import MODELS
    from embodied_jepa.contracts import StateSchema

    runner = _load("_probe_es_init", "scripts/probe_encoder_study.py")
    try:
        from embodied_jepa.config import ExperimentConfig

        config = ExperimentConfig.load(runner.BASE.CONFIG, require_checkpoint=False)
    except Exception as error:  # noqa: BLE001 - the dataset is absent in core CI
        pytest.skip(f"apple config not loadable here: {error}")
    manifest_path = ROOT / "data" / "apple-wide-v1" / "meta" / "jepa_manifest.json"
    if not manifest_path.exists():
        pytest.skip("apple-wide-v1 absent")
    schema = StateSchema(**json.loads(manifest_path.read_text())["state_schema"])
    settings = dict(config.model_settings)
    for heads in (True, False):
        model = MODELS.create(
            "leworldmodel",
            state_schema=schema,
            device="cpu",
            seed=0,
            config=settings | {"readout_heads": heads},
        )
        es.check_init(model, readout_heads=heads)


# ----- the runner: void-safe, overrides, arm failures -----------------------------------------
def _committed_only_manifest(tmp_path, **hashes):
    committed = json.loads(json.dumps(MANIFEST))
    committed["hashes"] = hashes or {
        "configs/g1_sim_action.json": MANIFEST["hashes"]["configs/g1_sim_action.json"]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(committed))
    return path


def _runner(name):
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    return _load(name, "scripts/probe_encoder_study.py")


def test_a_failed_guard_writes_a_v_report_and_never_overwrites(tmp_path, monkeypatch):
    runner = _runner("_probe_es_t1")
    path = _committed_only_manifest(tmp_path, **{"configs/g1_sim_action.json": "0" * 64})
    monkeypatch.setattr(runner, "MANIFEST", path)
    report = runner.run(tmp_path / "out", smoke=True)
    assert report["status"] == "void" and report["outcome"] == "V"
    written = json.loads((tmp_path / "out" / "report.json").read_text())
    assert written["outcome"] == "V" and "g1_sim_action.json" in written["void_reason"]
    with pytest.raises(FileExistsError):
        runner.run(tmp_path / "out", smoke=True)


def test_a_crash_writes_a_v_report_with_a_void_reason_and_reraises(tmp_path, monkeypatch):
    runner = _runner("_probe_es_t2")
    monkeypatch.setattr(runner, "MANIFEST", _committed_only_manifest(tmp_path))

    def boom():
        raise RuntimeError("injected crash")

    monkeypatch.setattr(runner.BASE, "plan_roots", boom)
    with pytest.raises(RuntimeError, match="injected crash"):
        runner.run(tmp_path / "out", smoke=True)
    written = json.loads((tmp_path / "out" / "report.json").read_text())
    assert written["outcome"] == "V" and "injected crash" in written["void_reason"]
    assert "Traceback" in written["traceback"]


def test_a_dirty_tree_or_a_foreign_device_voids_the_gated_run(tmp_path, monkeypatch):
    runner = _runner("_probe_es_t3")
    monkeypatch.setattr(runner, "MANIFEST", _committed_only_manifest(tmp_path))
    import embodied_jepa.training as training

    monkeypatch.setattr(training, "source_identity", lambda: {"dirty": True, "revision": "x"})
    report = runner.run(tmp_path / "dirty", smoke=False)
    assert report["outcome"] == "V" and "clean tree" in report["void_reason"]
    monkeypatch.setattr(training, "source_identity", lambda: {"dirty": False, "revision": "x"})
    report = runner.run(tmp_path / "cpu", smoke=False, device="cpu")
    assert report["outcome"] == "V" and "training device" in report["void_reason"]


def test_the_wall_cap_voids():
    runner = _runner("_probe_es_t4")
    runner.BASE.Clock(3600).check("stage")
    with pytest.raises(runner.VoidRun, match="wall cap"):
        runner.BASE.Clock(-1).check("stage")


def test_smoke_only_overrides_are_refused_for_the_gated_run(tmp_path, monkeypatch):
    runner = _runner("_probe_es_t5")
    monkeypatch.setattr("sys.argv", ["x", "--output", str(tmp_path / "o"), "--device", "cpu"])
    with pytest.raises(SystemExit, match="smoke-only"):
        runner.main()


def test_non_finite_values_become_null_and_are_listed(tmp_path):
    runner = _runner("_probe_es_t6")
    runner.write_report(tmp_path / "report.json", {"a": [float("inf")], "b": {"c": np.nan}})
    written = json.loads((tmp_path / "report.json").read_text())
    assert written["a"] == [None] and written["b"]["c"] is None
    assert sorted(written["non_finite_fields"]) == ["/a/0=inf", "/b/c=nan"]


class _FakeModel:
    """The surface ``train_encoder`` touches, with a scripted ``train_step``."""

    def __init__(self, step):
        torch = pytest.importorskip("torch")
        self.config = {
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "readout_heads": True,
            "latent_dim": 4,
        }
        self.optimizer = SimpleNamespace(param_groups=[{"lr": 0.0}])
        self.metadata = {}
        self.model = SimpleNamespace(encoder=torch.nn.Linear(2, 2))
        self._step = step
        self.calls = 0

    def fit_state_normalization(self, mean, std, *, training_episode_ids):
        self.normalized = (mean, std, training_episode_ids)

    def train_step(self, sequence, readout_targets=None):
        self.calls += 1
        return self._step(self.calls)

    def save(self, path):
        Path(path).write_bytes(b"fake")


def _fake_arrays():
    from embodied_jepa.world_model_v2 import EpisodeArrays

    rng = np.random.default_rng(0)
    n = 40
    return EpisodeArrays(
        episode_ids=("e0", "e1"),
        offsets=np.array([0, 20]),
        lengths=np.array([20, 20]),
        frames={"onboard_rgb": rng.integers(0, 256, (n, 4, 4, 3), dtype=np.uint8)},
        states=rng.normal(size=(n, 1)).astype(np.float32),
        mask=np.ones((n, 1), bool),
        actions=np.zeros((n, 14), np.float32),
        timestamps=np.arange(n, dtype=np.float64),
        targets={},
        phase=np.zeros(n, np.int16),
        rows={},
    )


def _train(runner, tmp_path, monkeypatch, step, *, cap=100.0, clock_cap=100.0, steps=3):
    from embodied_jepa.contracts import StateSchema

    fake = _FakeModel(step)
    monkeypatch.setattr(runner, "build_model", lambda *a, **k: fake)
    monkeypatch.setattr(runner, "provenance", lambda *a, **k: {"p": 1})
    monkeypatch.setattr(runner.es, "check_init", lambda model, readout_heads: "init")
    store = SimpleNamespace(state_schema=StateSchema(("arm.q",), ("rad",), "fixture_v0"))
    record = runner.train_encoder(
        "R0",
        "A",
        _fake_arrays(),
        store,
        {},
        "leworldmodel",
        tmp_path / "enc",
        runner.BASE.Clock(clock_cap),
        steps=steps,
        device="cpu",
        cap_seconds=cap,
        source={},
    )
    return record, fake


def test_a_non_finite_loss_makes_the_arm_not_evaluated_without_voiding(tmp_path, monkeypatch):
    runner = _runner("_probe_es_t7")

    def step(call):
        if call == 2:
            raise ContractError("non-finite training loss")
        return {"loss": 1.0}

    record, fake = _train(runner, tmp_path, monkeypatch, step)
    assert record["status"].startswith("failed: non-finite") and record["completed_steps"] == 1
    assert fake.calls == 2


def test_the_per_encoder_cap_makes_the_arm_not_evaluated(tmp_path, monkeypatch):
    runner = _runner("_probe_es_t8")
    record, fake = _train(runner, tmp_path, monkeypatch, lambda c: {"loss": 1.0}, cap=-1.0)
    assert record["status"].startswith("failed: per-encoder cap") and fake.calls == 0


def test_the_global_wall_cap_during_training_voids(tmp_path, monkeypatch):
    runner = _runner("_probe_es_t9")
    with pytest.raises(runner.VoidRun):
        _train(runner, tmp_path, monkeypatch, lambda c: {"loss": 1.0}, clock_cap=-1.0)


def test_a_completed_encoder_records_its_steps_and_checkpoint(tmp_path, monkeypatch):
    runner = _runner("_probe_es_t10")
    record, fake = _train(runner, tmp_path, monkeypatch, lambda c: {"loss": 1.0 / c}, steps=4)
    assert record["status"] == "completed" and record["completed_steps"] == 4 and fake.calls == 4
    assert Path(record["checkpoint"]).exists() and record["init_weights_digest"] == "init"
    with pytest.raises(FileExistsError):
        _train(runner, tmp_path, monkeypatch, lambda c: {"loss": 1.0})


def test_other_contract_errors_in_training_are_crashes_not_arm_failures(tmp_path, monkeypatch):
    runner = _runner("_probe_es_t11")

    def step(call):
        raise ContractError("some other contract problem")

    with pytest.raises(ContractError, match="other contract"):
        _train(runner, tmp_path, monkeypatch, step)
