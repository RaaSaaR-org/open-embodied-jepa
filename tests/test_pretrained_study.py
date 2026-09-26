"""TASK-063 decision code and runner: guards, Holm over two arms, floors, rows, void safety.

Synthetic evidence only; these tests make no learned-representation or control claims, and no
pretrained weight is read.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import encoder_study as es
from embodied_jepa import info_ceiling as ic
from embodied_jepa import observation_reprobe as orp
from embodied_jepa import pretrained_study as ps
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "benchmarks" / "manifests" / "apple-pretrained-encoder-v1.json").read_text()
)


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----- manifest and module agree -------------------------------------------------------------
def test_manifest_arms_floors_rows_and_budget_match_the_module():
    arms = MANIFEST["arms_decisional_in_holm_order"]
    assert tuple(a["arm"] for a in arms) == ps.ARMS
    assert {a["arm"]: a["floor"] for a in arms} == ps.ARM_FLOOR
    thresholds = [ps.ALPHA_ONE_SIDED / (len(ps.ARMS) - j) for j in range(len(ps.ARMS))]
    assert MANIFEST["multiplicity"]["step_thresholds"] == pytest.approx(thresholds)
    assert [r["row"] for r in MANIFEST["pre_declared_outcomes_in_order"]] == list(ps.ROWS)
    assert MANIFEST["abandonment_clause"]["fires_on"] == [
        r for r in ps.ROWS if ps.abandonment_fires(r)
    ]
    budget = MANIFEST["budget"]
    assert budget["per_arm_seconds"] == ps.PER_ARM_SECONDS
    assert budget["global_wall_seconds"] == ps.GLOBAL_WALL_SECONDS
    assert budget["device"] == ps.DEVICE
    anchors = MANIFEST["reproduction_anchors"]
    assert tuple(anchors["task061"]["sources"]) == ps.ANCHOR_SETS["task061"]
    assert tuple(anchors["task062"]["sources"]) == ps.ANCHOR_SETS["task062"]
    assert anchors["tolerance"] == ps.ANCHOR_TOLERANCE


# ----- Holm, floors, p-values ----------------------------------------------------------------
def test_holm_over_two_arms_both_directions():
    both = ps.holm({"P-cls": 0.01, "P-tok": 0.02})
    assert both["rejected"] == {"P-cls": True, "P-tok": True}
    assert [s["threshold"] for s in both["steps"]] == pytest.approx([0.0125, 0.025])
    stop = ps.holm({"P-cls": 0.02, "P-tok": 0.013})
    assert stop["rejected"] == {"P-cls": False, "P-tok": False}  # 0.013 > 0.0125 stops it
    one = ps.holm({"P-cls": 1.0, "P-tok": 0.001})
    assert one["rejected"] == {"P-cls": False, "P-tok": True}
    ties = ps.holm({"P-tok": 0.001, "P-cls": 0.001})
    assert [s["hypothesis"] for s in ties["steps"]] == ["P-cls", "P-tok"]
    with pytest.raises(ContractError):
        ps.holm({"A-tok": 0.001})


def _perfect_readout(n=190, seed=3):
    rng = np.random.default_rng(seed)
    xy = rng.normal(0.3, 0.02, (n, 2))
    dx = rng.choice([-0.4, 0.4], n)
    occ = rng.random(n) < 0.5
    fold = ic.fold_of(n)
    priors = ic.prior_predictions(xy, dx, occ, fold, dy=np.zeros(n))
    return xy, dx, priors


def test_the_floor_condition_both_directions():
    xy, dx, priors = _perfect_readout()
    everyone = np.ones(len(xy), bool)
    arm = ic.evaluate(xy + 0.0005, dx, xy, dx, priors, everyone)
    worse = ic.evaluate(xy + 0.02, -dx, xy, dx, priors, everyone)
    assert ic.beats_random_floor(arm, worse)["beats_random_floor"]
    good = es.arm_pvalues(xy + 0.0005, dx, xy, dx, priors, worse, arm)
    assert good["p_F"] == 0.0 and good["p"] <= 0.0125
    # The floor as good as the arm: the T2 point condition fails and p is 1.
    same = es.arm_pvalues(xy + 0.0005, dx, xy, dx, priors, arm, arm)
    assert not ic.beats_random_floor(arm, arm)["beats_random_floor"] and same["p"] == 1.0


# ----- arm states, pass rule, qualifiers, rows ---------------------------------------------
def test_arm_states_and_the_per_arm_cap():
    assert ps.arm_state(evaluated=True) == "evaluated"
    assert ps.arm_state(evaluated=False) == "not evaluated"
    assert not ps.over_cap(ps.PER_ARM_SECONDS) and ps.over_cap(ps.PER_ARM_SECONDS + 0.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"evaluated": False},
        {"succeeds": False},
        {"beats_floor": False},
        {"rejected": False},
        {"spurious": True},
    ],
)
def test_every_pass_condition_is_needed(kwargs):
    full = {
        "evaluated": True,
        "succeeds": True,
        "beats_floor": True,
        "rejected": True,
        "spurious": False,
    }
    assert ps.passes(**full)
    assert not ps.passes(**(full | kwargs))


def test_qualifiers():
    q = dict(passes=False, succeeds=True, beats_floor=False, rejected=False, beats_prior=True)
    assert ps.qualifier(**q) == "meets the bars, not the floor"
    assert ps.qualifier(**(q | {"beats_floor": True})) == "unadjusted only; not a pass"
    assert ps.qualifier(**(q | {"succeeds": False})) == "partial information"


def _arm(passes=False, state="evaluated", succeeds=False, spurious=False):
    return {"passes": passes, "state": state, "succeeds": succeeds, "spurious": spurious}


@pytest.mark.parametrize(
    ("arms", "row"),
    [
        ({"P-cls": _arm(passes=True), "P-tok": _arm(passes=True)}, "O-PT-POOLED"),
        ({"P-cls": _arm(), "P-tok": _arm(passes=True)}, "O-PT-TOKENS"),
        ({"P-cls": _arm(state="not evaluated"), "P-tok": _arm(succeeds=True)}, "O-PT-INCOMPLETE"),
        ({"P-cls": _arm(), "P-tok": _arm(succeeds=True)}, "O-PT-FLOOR"),
        ({"P-cls": _arm(), "P-tok": _arm(succeeds=True, spurious=True)}, "O-PT-NONE"),
        ({"P-cls": _arm(), "P-tok": _arm()}, "O-PT-NONE"),
    ],
)
def test_decision_rows_in_order(arms, row):
    assert ps.decide(void=False, arms=arms)["outcome"] == row


def test_void_first_missing_arms_refused_and_abandonment_rows():
    arms = {"P-cls": _arm(passes=True), "P-tok": _arm(passes=True)}
    assert ps.decide(void=True, arms=arms)["outcome"] == "V"
    with pytest.raises(ContractError):
        ps.decide(void=False, arms={"P-cls": _arm()})
    assert [r for r in ps.ROWS if ps.abandonment_fires(r)] == ["O-PT-FLOOR", "O-PT-NONE"]
    assert ps.also_matching("O-PT-POOLED", arms) == ["O-PT-TOKENS"]
    assert ps.also_matching("O-PT-POOLED", arms | {"P-tok": _arm()}) == []
    assert ps.also_matching("O-PT-NONE", {"P-cls": _arm(), "P-tok": _arm()}) == []


# ----- spurious check ---------------------------------------------------------------------
def test_spurious_check_reads_hidden_features_through_the_out_of_fold_readouts():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 8))
    y = np.stack([x[:, 0], x[:, 1] - x[:, 2]], axis=1)
    fold = ic.fold_of(len(x))
    g, diag = ic.gram(x)
    pred, fits, _s = ic.nested_cv(g, diag, y, fold)
    again = orp.predict_with_fold_readouts(fits, fold, *ic.cross_gram(x, x))
    assert np.allclose(again, pred, atol=1e-9)
    blank = orp.predict_with_fold_readouts(fits, fold, *ic.cross_gram(np.zeros_like(x), x))
    assert not np.allclose(blank, pred)
    unavailable = orp.spurious_verdict(None, renderer_reproduces=False)
    assert unavailable["spurious"] and not unavailable["available"]
    assert not orp.spurious_verdict({"beats_prior": False}, renderer_reproduces=True)["spurious"]
    assert orp.spurious_verdict({"beats_prior": True}, renderer_reproduces=True)["spurious"]


# ----- guards -----------------------------------------------------------------------------------
def test_g_weights_both_directions():
    p = MANIFEST["encoder"]["pretrained_weights_digest"]
    f = MANIFEST["floors"]["weights_digest"]
    ps.check_weights(p, f, MANIFEST)
    with pytest.raises(ps.GuardError, match="pretrained"):
        ps.check_weights("0" * 64, f, MANIFEST)
    with pytest.raises(ps.GuardError, match="floor"):
        ps.check_weights(p, p, MANIFEST)


def test_g_frames_and_g_repro_both_directions():
    ps.check_frames("a" * 64, "a" * 64)
    with pytest.raises(ps.GuardError, match="G-frames"):
        ps.check_frames("a" * 64, "b" * 64)
    x = {"cls": np.ones((3, 4)), "tokens": np.zeros((3, 8))}
    ps.check_repro(x, copy.deepcopy(x))
    y = copy.deepcopy(x)
    y["cls"][0, 0] = np.nextafter(1.0, 2.0)
    with pytest.raises(ps.GuardError, match="G-repro"):
        ps.check_repro(x, y)
    with pytest.raises(ps.GuardError, match="G-repro"):
        ps.check_repro(x, {"cls": x["cls"]})


def _anchor_views(sources):
    view = {
        "results": {
            "all": {"T1": {"median": 0.5, "pass": True}, "T2": {"correct": 180, "n": 190}},
            "selections": {"xy": [{"family": "rbf", "lam_rel": 0.01, "inner_mse": 1e-4}]},
        },
        "per_root": [{"xy": [0.3, 0.1], "dx": 0.4, "dy": -0.2}],
    }
    return {s: copy.deepcopy(view) for s in sources}


@pytest.mark.parametrize("label", ["task061", "task062"])
def test_g_anchor_three_branches_for_both_anchor_sets(label):
    sources = ps.ANCHOR_SETS[label]
    exact = ps.anchor_compare(_anchor_views(sources), _anchor_views(sources), sources)
    assert exact["verdict"] == "exact"
    ps.check_anchor(exact, label)
    tiny = _anchor_views(sources)
    tiny[sources[-1]]["per_root"][0]["dx"] += 1e-9
    caveat = ps.anchor_compare(_anchor_views(sources), tiny, sources)
    assert caveat["verdict"] == "caveat"
    ps.check_anchor(caveat, label)
    for edit in (
        lambda v: v[sources[0]]["results"]["all"]["T2"].__setitem__("correct", 179),
        lambda v: v[sources[0]]["results"]["selections"]["xy"][0].__setitem__("lam_rel", 0.1),
        lambda v: v[sources[-1]]["per_root"][0].__setitem__("dx", 0.4 + 1e-5),
    ):
        bad = _anchor_views(sources)
        edit(bad)
        verdict = ps.anchor_compare(_anchor_views(sources), bad, sources)
        assert verdict["verdict"] == "void"
        with pytest.raises(ps.GuardError, match=f"G-anchor \\({label}\\)"):
            ps.check_anchor(verdict, label)
    missing = _anchor_views(sources)
    missing.pop(sources[0])
    with pytest.raises(ps.GuardError, match="missing"):
        ps.anchor_compare(_anchor_views(sources), missing, sources)


# ----- the runner: void safety, overrides, arm-level failures ----------------------------------
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
    pytest.importorskip("transformers")
    return _load(name, "scripts/probe_pretrained_encoder.py")


def test_a_failed_guard_writes_a_v_report_and_never_overwrites(tmp_path, monkeypatch):
    runner = _runner("_probe_pt_t1")
    path = _committed_only_manifest(tmp_path, **{"configs/g1_sim_action.json": "0" * 64})
    monkeypatch.setattr(runner, "MANIFEST", path)
    report = runner.run(tmp_path / "out", smoke=True)
    assert report["status"] == "void" and report["outcome"] == "V"
    written = json.loads((tmp_path / "out" / "report.json").read_text())
    assert written["outcome"] == "V" and "g1_sim_action.json" in written["void_reason"]
    assert written["learned_apple_to_plate_successes"] == 0
    with pytest.raises(FileExistsError):
        runner.run(tmp_path / "out", smoke=True)


def test_a_missing_weight_file_is_a_g_hash_void(tmp_path, monkeypatch):
    runner = _runner("_probe_pt_t2")
    name = "third_party/dinov2-small/model.safetensors"
    path = _committed_only_manifest(tmp_path, **{name: MANIFEST["hashes"][name]})
    monkeypatch.setattr(runner, "MANIFEST", path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)  # the weights are absent under tmp_path
    report = runner.run(tmp_path / "out", smoke=True)
    assert report["outcome"] == "V" and "model.safetensors" in report["void_reason"]


def test_a_crash_writes_a_v_report_with_a_void_reason_and_reraises(tmp_path, monkeypatch):
    runner = _runner("_probe_pt_t3")
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
    runner = _runner("_probe_pt_t4")
    monkeypatch.setattr(runner, "MANIFEST", _committed_only_manifest(tmp_path))
    import embodied_jepa.training as training

    monkeypatch.setattr(training, "source_identity", lambda: {"dirty": True, "revision": "x"})
    report = runner.run(tmp_path / "dirty", smoke=False)
    assert report["outcome"] == "V" and "clean tree" in report["void_reason"]
    monkeypatch.setattr(training, "source_identity", lambda: {"dirty": False, "revision": "x"})
    report = runner.run(tmp_path / "mps", smoke=False, device="mps")
    assert report["outcome"] == "V" and "device" in report["void_reason"]


def test_the_wall_cap_voids():
    runner = _runner("_probe_pt_t5")
    runner.BASE.Clock(3600).check("stage")
    with pytest.raises(runner.VoidRun, match="wall cap"):
        runner.BASE.Clock(-1).check("stage")


def test_non_finite_values_become_null_and_are_listed(tmp_path):
    runner = _runner("_probe_pt_t6")
    runner.write_report(tmp_path / "report.json", {"a": [float("inf")], "b": {"c": np.nan}})
    written = json.loads((tmp_path / "report.json").read_text())
    assert written["a"] == [None] and written["b"]["c"] is None
    assert sorted(written["non_finite_fields"]) == ["/a/0=inf", "/b/c=nan"]


def _tiny_floor(tmp_path):
    from embodied_jepa import pretrained_encoder as pe

    config = {
        "architectures": ["Dinov2Model"],
        "hidden_size": 384,
        "num_attention_heads": 6,
        "num_hidden_layers": 1,
        "patch_size": 14,
        "image_size": 518,
        "model_type": "dinov2",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return pe.random_init(tmp_path, pinned={"config.json": pe.sha256(path)})


def test_a_non_finite_feature_is_an_arm_level_failure_not_a_crash(tmp_path):
    runner = _runner("_probe_pt_t7")
    import torch

    model = _tiny_floor(tmp_path)
    frames = {"post_112": np.zeros((2, 112, 112, 3), np.uint8)}
    out, seconds, reason = runner.featurise(model, frames)
    assert reason is None and out["post_112"]["mean"].shape == (2, 384) and seconds >= 0
    with torch.no_grad():
        model.layernorm.weight.fill_(float("nan"))
    out, _seconds, reason = runner.featurise(model, frames)
    assert out is None and "non-finite" in reason


def test_token_mean_is_the_mean_over_the_patch_grid():
    runner = _runner("_probe_pt_t8")
    tokens = np.arange(2 * 256 * 384, dtype=np.float64).reshape(2, 256, 384)
    assert np.array_equal(runner.token_mean(tokens.reshape(2, -1)), tokens.mean(1))
