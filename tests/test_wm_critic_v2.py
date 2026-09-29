"""TASK-073 (``apple_wm_critic_v2``): the preregistered design, its guards and its mechanics.

NumPy tests run in the core job; torch tests import-skip without torch; simulation tests are
graphics opt-in (``JEPA_TEST_RENDER=1``) and use smoke seeds (53950-53999) only. Nothing here
simulates a K0, R, D3, corpus, S or U seed.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa import first_policy_v2_m2 as fm
from embodied_jepa import wm_critic_v2 as wc
from embodied_jepa.contracts import ContractError
from embodied_jepa.models.latent_critic import (
    CopyLast,
    LatentCritic,
    RidgeReadout,
    ZeroActions,
    grouped_folds,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "benchmarks" / "manifests" / "apple-wm-critic-v2.json").read_text())
render = pytest.mark.skipif(os.environ.get("JEPA_TEST_RENDER") != "1", reason="graphics opt-in")


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----- the frozen block, the pins and the stored cohorts ------------------------------------------
# A literal: the gated-stage authorisation PR may not change it (AUTHORISATION_SCOPE).
FROZEN_SHA256 = "cd9e8723f8afa71c9db73f8b326b20099763743798e95f3ba3369fbfa7344f83"


def test_frozen_hash_is_the_preregistered_literal():
    assert wc.frozen_sha256() == FROZEN_SHA256 == MANIFEST["frozen_sha256"]
    assert "may not change wm_critic_v2.py" in wc.AUTHORISATION_SCOPE


def test_manifest_frozen_block_equals_the_module():
    assert MANIFEST["frozen"] == wc.frozen_block()
    assert MANIFEST["frozen_sha256"] == wc.frozen_sha256()
    assert MANIFEST["gated_authorization"] is None  # the gated stage is not authorised yet


def test_pinned_files_match():
    import hashlib

    for relative, want in MANIFEST["hashes"].items():
        got = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert got == want, relative


def test_model_files_behind_the_implementation_hash_are_the_m2_pins():
    """TASK-073 edits none of the files an existing implementation hash or M2 pin covers."""
    m2 = json.loads(
        (ROOT / "benchmarks" / "manifests" / "apple-first-policy-v2-m2.json").read_text()
    )
    shared = set(m2["hashes"]) & set(MANIFEST["hashes"])
    assert len(shared) >= 30
    for relative in shared:
        assert MANIFEST["hashes"][relative] == m2["hashes"][relative], relative


def test_stored_cohorts_are_the_generated_values_and_their_digest():
    """Exact regeneration is Linux-only (macOS differs in the last ulp of a few resets), so the
    floats are compared with a tolerance, the chosen shift directions exactly (whole degrees),
    and the digest exactly against the stored JSON. The runners never regenerate: they read the
    stored values."""
    for role in wc.COHORT_ROLES:
        node = MANIFEST["cohorts"][role]
        assert node["sha256"] == wc.cohort_digest(node["values"])
        generated = wc.cohort_values(role)
        assert set(generated) == set(node["values"])
        for seed, entry in node["values"].items():
            mine = generated[seed]
            for key in ("object_xy", "plate_xy"):
                assert np.allclose(mine[key], entry[key], rtol=0, atol=1e-12)
            if role == "U":
                assert "shift_m" not in entry
                continue
            for cm, vector in entry["shift_m"].items():
                stored = round(np.degrees(np.arctan2(vector[1], vector[0]))) % 360
                again = round(np.degrees(np.arctan2(*mine["shift_m"][cm][::-1]))) % 360
                assert stored == again, (role, seed, cm)
                assert np.allclose(mine["shift_m"][cm], vector, rtol=0, atol=1e-12)
        stored = wc.stored_cohort(MANIFEST, role)
        assert tuple(stored) == wc.seeds_of(role)


def test_the_cohort_comparison_still_discriminates():
    node = MANIFEST["cohorts"]["S"]["values"]
    seed = str(wc.seeds_of("S")[0])
    moved = wc.cohort_values("S")[seed]["plate_xy"][0] + 1e-9
    assert not np.isclose(moved, node[seed]["plate_xy"][0], rtol=0, atol=1e-12)


def test_a_tampered_cohort_is_refused():
    bad = copy.deepcopy(MANIFEST)
    first = str(wc.seeds_of("S")[0])
    bad["cohorts"]["S"]["values"][first]["plate_xy"][0] += 1e-9
    with pytest.raises(wc.GuardError, match="digest"):
        wc.stored_cohort(bad, "S")


def test_wide_reset_values_equal_the_task047_generator():
    pytest.importorskip("torch")
    wide_reset = _load_script("_evaluate_apple_wm", "scripts/evaluate_apple.py").wide_reset
    for seed in (*wc.seeds_of("K0")[:3], *wc.seeds_of("S")[-3:], 53999):
        want = wide_reset(seed)
        got = wc.wide_reset_values(seed)
        assert got["object_xy"] == list(map(float, want["object_xy"]))
        assert got["plate_xy"] == list(map(float, want["plate_xy"]))


# ----- seeds --------------------------------------------------------------------------------------
def test_seed_ranges_are_fresh_disjoint_and_off_every_forbidden_range():
    wc.check_seed_ranges()
    assert wc.SEED_RANGES["K0"] == (53000, 53031) and wc.SEED_RANGES["S"] == (53800, 53863)
    for role, n in (("K0", 32), ("R", 32), ("D3", 16), ("corpus", 300), ("S", 64), ("U", 32)):
        assert len(wc.seeds_of(role)) == n
    assert "cohort_C" in wc.FORBIDDEN_RANGES and "task071_072_block" in wc.FORBIDDEN_RANGES


@pytest.mark.parametrize("seed", [45300, 45339, 52000, 52015, 51000, 50600, 20000])
def test_role_guard_refuses_cohort_c_d2_and_spent_ranges(seed):
    with pytest.raises(wc.GuardError):
        wc.check_role_seeds("smoke", (seed,), smoke=True)


def test_role_guard_is_a_whitelist_of_plain_ints_in_order():
    assert wc.check_role_seeds("S", wc.seeds_of("S")) == wc.seeds_of("S")
    with pytest.raises(wc.GuardError, match="plain ints"):
        wc.check_role_seeds("S", tuple(str(s) for s in wc.seeds_of("S")))
    with pytest.raises(wc.GuardError, match="exactly"):
        wc.check_role_seeds("S", tuple(reversed(wc.seeds_of("S"))))
    with pytest.raises(wc.GuardError, match="smoke"):
        wc.check_role_seeds("smoke", (53000,), smoke=True)
    assert wc.check_role_seeds("smoke", (53950, 53999), smoke=True) == (53950, 53999)


# ----- the condition ------------------------------------------------------------------------------
def test_expert_budgets_and_the_clock_facts():
    assert wc.EXPERT_BUDGETS == (130, 80, 45, 150, 100, 50, 30, 50, 30, 60)
    assert wc.TRANSFER_START == 405 and wc.STEADY_END == 585
    assert wc.DECISION_STEPS[0] == 304 and wc.DECISION_STEPS[-1] == 576
    assert len(wc.DECISION_STEPS) == 18
    assert set(wc.O_STAR_CM) == {t + wc.CHUNK for t in wc.DECISION_STEPS}


def _truth(plate_xy, apple_xy=(0.34, -0.18)):
    return {
        "position_frame": "world",
        "base_position_world": np.array([0.0, 0.0, 0.793]),
        "base_rotation_world": np.eye(3),
        "object_position": np.array([*apple_xy, 0.77]),
        "plate_position": np.array([*plate_xy, 0.746]),
        "container_surface_z": 0.752,
        "object_support_height": 0.027,
    }


def test_e9_pick_phases_are_invariant_under_a_plate_shift():
    from embodied_jepa import first_policy_runtime as rt
    from embodied_jepa.resting_expert import RestingPlaceExpert

    truth = _truth((0.49, -0.09))
    before = RestingPlaceExpert(truth, **fp2.EXPERT)
    for shift in ([0.0, 0.06], [0.0, -0.05], [-0.03, 0.01]):
        after = RestingPlaceExpert(rt.perturbed_truth(truth, [0.0, 0.0], shift), **fp2.EXPERT)
        for a, b in zip(
            before.phases[: wc.PICK_PHASES], after.phases[: wc.PICK_PHASES], strict=True
        ):
            assert a.name == b.name and a.grasp == b.grasp and a.commands == b.commands
            assert np.array_equal(a.target_base, b.target_base)
        assert np.array_equal(before.pick_rotation, after.pick_rotation)
        assert not np.array_equal(before.release_target, after.release_target)
    assert tuple(p.name for p in before.phases[: wc.PICK_PHASES]) == fp2.EXPERT_PHASES[:4]


def test_e9_release_is_pulled_back_on_the_reset_distribution():
    """Why the direction rule exists: e9's release x is clamped to its reach sphere."""
    for seed in wc.seeds_of("K0"):
        plate = np.asarray(wc.wide_reset_values(seed)["plate_xy"])
        assert wc.release_xy(plate)[0] < plate[0] + wc.E9_RELEASE["release_dx"] - 1e-3


def test_stored_shifts_have_the_declared_size_and_keep_e9s_release_geometry():
    for role in ("K0", "R", "D3", "S"):
        for entry in MANIFEST["cohorts"][role]["values"].values():
            plate = np.asarray(entry["plate_xy"])
            for cm, vector in entry["shift_m"].items():
                v = np.asarray(vector)
                assert np.isclose(np.linalg.norm(v), int(cm) / 100.0)
                moved = plate + v
                drift = (wc.release_xy(moved) - moved) - (wc.release_xy(plate) - plate)
                assert np.linalg.norm(drift) <= wc.ARC_TOLERANCE_M + 1e-12
                assert wc.TABLETOP["y"][0] <= moved[1] <= wc.TABLETOP["y"][1]


def test_shift_directions_are_seeded_and_shared_across_magnitudes():
    reset = wc.wide_reset_values(53000)
    assert wc.shift_vector(53000, reset, 4) == wc.shift_vector(53000, reset, 4)
    assert wc.shift_vector(53000, reset, 4) != wc.shift_vector(
        53001, wc.wide_reset_values(53001), 4
    )


# ----- the controller's choice --------------------------------------------------------------------
def test_candidates_are_the_incumbent_plus_a_5x5_grid():
    aims = wc.candidate_aims([0.5, -0.1])
    assert aims.shape == (25, 2) and wc.INCUMBENT_INDEX == 12
    assert np.array_equal(aims[12], [0.5, -0.1])
    assert np.isclose(np.abs(aims - [0.5, -0.1]).max(), 0.02)


def test_tie_rule_keeps_the_incumbent_and_takes_a_strictly_lower_cost():
    costs = np.full(25, 1.0)
    assert wc.choose(costs) == wc.INCUMBENT_INDEX
    costs[3] = 1.0 - wc.TIE_TOLERANCE_CM / 2
    assert wc.choose(costs) == wc.INCUMBENT_INDEX
    costs[3] = 0.5
    assert wc.choose(costs) == 3
    costs[3] = np.nan
    assert wc.choose(costs) == wc.INCUMBENT_INDEX
    with pytest.raises(ContractError):
        wc.choose(np.ones(24))


class _ActionBlind:
    """A 'model' that ignores actions entirely (as N does by construction)."""

    def predict_features(self, features, actions):
        return np.repeat(np.asarray(features, np.float32)[:, None] * 0.5, actions.shape[1], 1)


def _readout(d=12):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, d))
    y = x[:, :2] * 0.01
    return RidgeReadout.fit(
        x, y, np.repeat(np.arange(12), 5), lambdas=(1e-3, 1.0), folds=5, seed=7310
    )


def test_action_blind_critics_reduce_to_the_incumbent():
    r_off = _readout()
    start = np.random.default_rng(1).normal(size=12).astype(np.float32)
    chunks = np.random.default_rng(2).uniform(-0.5, 0.5, (25, 16, 14)).astype(np.float32)
    for model in (CopyLast(), ZeroActions(_ActionBlind()), _ActionBlind()):
        costs = LatentCritic(model, r_off, wc.O_STAR_CM).costs(start, chunks, 480)
        assert np.ptp(costs) == 0.0
        assert wc.choose(costs) == wc.INCUMBENT_INDEX


def test_ridge_readout_recovers_a_linear_map_in_both_forms():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 6))
    w = rng.normal(size=(6, 2))
    y = x @ w + 0.3
    groups = np.repeat(np.arange(16), 5)
    primal = RidgeReadout.fit(x, y, groups, lambdas=(1e-6, 1e-3, 1.0), folds=5, seed=1)
    dual = RidgeReadout.fit(x, y, groups, lambdas=(1e-6, 1e-3, 1.0), folds=5, seed=1, dual=True)
    assert np.allclose(primal.predict(x), y, atol=1e-3)
    assert np.allclose(dual.predict(x), primal.predict(x), atol=1e-6)
    again = RidgeReadout.from_state(primal.state())
    assert again.sha256() == primal.sha256()
    fold = grouped_folds(groups, 5, 7310)
    for g in np.unique(groups):
        assert len(set(fold[groups == g])) == 1


def test_critic_cost_is_distance_to_o_star():
    target = np.asarray(wc.O_STAR_CM[480])
    assert np.isclose(wc.critic_cost_cm(target + [3.0, 4.0], 480), 5.0)


# ----- K0, offline, D3 and gated rules ------------------------------------------------------------
def test_k0_bars_in_order_and_the_smallest_passing_size():
    good = {"B-oracle-shift": 30, "P-stale": 5, "P-truth": 22, "H-sim": 27}
    assert wc.k0_cell_passes(good)["passes"]
    assert wc.k0_cell_passes(good | {"B-oracle-shift": 27})["failed_bar"] == "B-oracle-shift"
    assert wc.k0_cell_passes(good | {"P-stale": 9})["failed_bar"] == "P-stale"
    assert wc.k0_cell_passes(good | {"P-truth": 19})["failed_bar"] == "P-truth"
    assert wc.k0_cell_passes(good | {"H-sim": 25})["failed_bar"] == "H-sim"
    assert wc.k0_next_arm({}) == "B-oracle-shift"
    assert wc.k0_next_arm({"B-oracle-shift": 20}) is None  # the first failed bar ends the cell
    assert wc.k0_next_arm({"B-oracle-shift": 30, "P-stale": 3}) == "P-truth"
    assert wc.k0_next_arm(good) is None
    cells = {(300, 3): good | {"P-stale": 12}, (300, 4): good, (300, 5): good}
    assert wc.k0_select(cells) == {"row": "K0-PASS", "shift_step": 300, "shift_cm": 4}
    assert wc.k0_select({(480, 6): good})["shift_step"] == 480
    assert wc.k0_select({(300, 3): good | {"H-sim": 22}})["row"] == "S-NO-CONDITION"


def _seed_gates(o1=True, o2=True, o2_void=False, o3=True, o3_void=False, o4=True):
    return {
        s: {
            "O1": {"passes": o1},
            "O2": {"passes": o2, "void": o2_void},
            "O3": {"passes": o3, "void": o3_void},
            "O4": {"passes": o4},
        }
        for s in wc.MODEL_SEEDS
    }


def test_offline_rows_first_match():
    ok = {"passes": True}
    assert wc.decide_offline(_seed_gates(), ok, ok)["row"] == "OFFLINE-PASS"
    no_room = wc.decide_offline(_seed_gates(), ok, wc.o0_passes({"median": 0.03}))
    assert no_room["row"] == "R-NO-HEADROOM" and no_room["fallback_authorised"]
    assert not no_room["abandonment_clause_fires"]
    assert wc.o0_passes({"median": 0.5})["passes"]
    assert wc.decide_offline(_seed_gates(o2=False), ok, {"passes": False})["row"] == "WMC-G2A"
    assert wc.decide_offline(_seed_gates(o1=False), ok, ok)["row"] == "WMC-NO-DYNAMICS"
    assert wc.decide_offline(_seed_gates(o2=False), ok, ok)["row"] == "WMC-G2A"
    assert wc.decide_offline(_seed_gates(o2_void=True), ok, ok)["row"] == "WMC-O2-VOID"
    assert wc.decide_offline(_seed_gates(o3=False), ok, ok)["row"] == "WMC-NO-RANK"
    assert wc.decide_offline(_seed_gates(o4=False), ok, ok)["row"] == "WMC-NO-RANK"
    assert wc.decide_offline(_seed_gates(o3_void=True), ok, ok)["row"] == "WMC-O3-VOID"
    assert wc.decide_offline(_seed_gates(), {"passes": False}, ok)["row"] == "WMC-PROPOSAL"
    assert wc.decide_offline(_seed_gates(o2=False), ok, ok)["abandonment_clause_fires"]
    assert wc.decide_offline(_seed_gates(o3=False), ok, ok)["abandonment_clause_fires"]
    assert not wc.decide_offline(_seed_gates(o1=False), ok, ok)["abandonment_clause_fires"]


def _ci(lo, hi, median=None):
    return {"median": median if median is not None else (lo + hi) / 2, "ci95": [lo, hi]}


def test_offline_gate_functions_in_both_directions():
    o2 = {
        "encoded_median_cm": 0.8,
        "predicted_median_cm": 1.2,
        "ratio_vs_persistence": _ci(0.5, 0.7),
        "n_ratio_vs_persistence": _ci(0.9, 1.1),
        "ratio_vs_clock_prior": _ci(0.4, 0.6),
    }
    assert wc.o2_passes(o2) == {"passes": True, "void": False}
    assert not wc.o2_passes(o2 | {"ratio_vs_persistence": _ci(0.7, 0.85)})["passes"]
    assert not wc.o2_passes(o2 | {"ratio_vs_clock_prior": _ci(0.7, 0.85)})["passes"]
    assert wc.o2_passes(o2 | {"n_ratio_vs_persistence": _ci(0.5, 0.7)})["void"]
    o3 = {
        "rho_w": _ci(0.55, 0.8),
        "margin": _ci(0.35, 0.6),
        "blind": {
            "copy-last": _ci(0, 0, 0.0),
            "N": _ci(0, 0, 0.0),
            "L-shuf": _ci(0.1, 0.3),
            "prior-distance": _ci(0.5, 0.9, 0.7),
        },
    }
    assert wc.o3_passes(o3) == {"passes": True, "void": False}  # prior-distance never voids
    assert wc.o3_passes(o3 | {"blind": o3["blind"] | {"L-shuf": _ci(0.4, 0.7, 0.5)}})["void"]
    assert not wc.o3_passes(o3 | {"margin": _ci(0.2, 0.5)})["passes"]
    o4 = {"regret_w": _ci(0.3, 0.9, 0.6), "regret_difference": _ci(-0.5, -0.1)}
    assert wc.o4_passes(o4)["passes"]
    assert not wc.o4_passes(o4 | {"regret_difference": _ci(-0.5, 0.0)})["passes"]
    assert wc.o5_passes([0.1, 0.2, 0.3])["passes"] and not wc.o5_passes([0.3, 0.4])["passes"]
    assert not wc.o5_passes([])["passes"]


def test_d3_stop_rules():
    base = {"P-reread": 5, "H-sim": 9, "H-LeWM": 9, "H-shuf": 6}
    assert wc.decide_d3(base)["row"] == "D3-GO"
    assert wc.decide_d3(base | {"H-sim": 6})["row"] == "NO-HEADROOM"
    assert wc.decide_d3(base | {"H-sim": 6})["fallback_authorised"]
    assert wc.decide_d3(base | {"H-LeWM": 7})["row"] == "WMC-DEV-STOP"


def _gated(n):
    s = {arm: np.zeros(64, bool) for arm in wc.S_ARMS}
    for arm, k in n.items():
        s[arm][:k] = True
    return s


def _harness(**kw):
    return {
        "b_hold_grasps": 0,
        "b_random_grasps": 0,
        "privileged_ok": dict.fromkeys(wc.L1_ARMS, True),
        "determinism_ok": True,
        "cuda_allocation_failed": False,
        "median_decision_seconds": 0.2,
    } | kw


def test_gated_rows_first_match_and_one_sided_mcnemar():
    assert wc.mcnemar_one_sided(10, 0) == pytest.approx(2**-10)
    assert wc.mcnemar_one_sided(0, 0) == 1.0
    n = {
        "B-oracle-shift": 60,
        "P-stale": 10,
        "P-reread": 20,
        "H-LeWM": 40,
        "H-N": 20,
        "H-shuf": 20,
        "H-rand": 20,
        "H-sim": 45,
        "H-LeWM-s1": 38,
        "H-LeWM-s2": 36,
    }
    s = _gated(n)
    u = {a: np.ones(32, bool) for a in wc.U_ARMS}
    assert wc.decide_gated(s, u, _harness())["row"] == "HYB-PASS"
    assert wc.decide_gated(_gated(n | {"B-oracle-shift": 57}), u, _harness())["row"] == "VOID"
    assert wc.decide_gated(s, u, _harness(determinism_ok=False))["row"] == "VOID"
    bad_priv = dict.fromkeys(wc.L1_ARMS, True) | {"H-N": False}
    assert wc.decide_gated(s, u, _harness(privileged_ok=bad_priv))["row"] == "VOID"
    assert wc.decide_gated(_gated(n | {"P-stale": 25}), u, _harness())["row"] == "S-VOID-CONDITION"
    assert wc.decide_gated(_gated(n | {"H-sim": 31}), u, _harness())["row"] == "H-NO-HEADROOM"
    blind = wc.decide_gated(_gated(n | {"H-shuf": 38}), u, _harness())
    assert blind["row"] == "HYB-SCENE-BLIND" and not blind["abandonment_clause_fires"]
    harm_u = u | {"H-LeWM": np.r_[np.ones(29, bool), np.zeros(3, bool)]}
    assert wc.decide_gated(s, harm_u, _harness())["row"] == "HYB-HARM"
    slow = wc.decide_gated(s, u, _harness(median_decision_seconds=0.3))
    assert slow["row"] == "HYB-SLOW" and not slow["abandonment_clause_fires"]
    assert wc.decide_gated(_gated(n | {"H-N": 35}), u, _harness())["row"] == "HYB-NO-GAIN"


def test_abandonment_and_fallback_rows():
    assert wc.CLAUSE_ROWS == ("WMC-G2A", "WMC-NO-RANK", "HYB-HARM", "HYB-NO-GAIN")
    assert set(wc.FALLBACK_ROWS) == {
        "S-NO-CONDITION",
        "R-NO-HEADROOM",
        "NO-HEADROOM",
        "H-NO-HEADROOM",
    }
    for row in wc.CLAUSE_ROWS:
        assert wc.abandonment_fires(row)
    for row in (*wc.FALLBACK_ROWS, "HYB-SCENE-BLIND", "HYB-SLOW", "VOID", "WMC-DEV-STOP"):
        assert not wc.abandonment_fires(row)
    with pytest.raises(ContractError):
        wc.abandonment_fires("NOT-A-ROW")


def test_spearman_and_cluster_statistics():
    assert wc.spearman([1, 2, 3], [2, 4, 9]) == pytest.approx(1.0)
    assert wc.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert wc.spearman([1, 1, 1], [3, 2, 1]) == 0.0
    ci = wc.cluster_median_ci(np.arange(20.0), np.repeat(np.arange(5), 4), resamples=200)
    assert ci["ci95"][0] <= ci["median"] <= ci["ci95"][1]
    r = wc.cluster_median_ratio(
        np.ones(20), 2 * np.ones(20), np.repeat(np.arange(5), 4), resamples=200
    )
    assert r["ratio"] == pytest.approx(0.5) and r["ci95"] == [0.5, 0.5]


def test_budget_rule():
    assert wc.budget_updates([6500, 7000])["updates"] == 15_000
    assert wc.budget_updates([2000])["updates"] == 10_000
    capped = wc.budget_updates([31_000])
    assert capped["updates"] == 60_000 and capped["escalate"]


def test_corpus_plan_is_fixed_and_has_the_declared_shape():
    plan = wc.corpus_plan()
    assert plan == wc.corpus_plan()
    assert [p["seed"] for p in plan] == list(wc.seeds_of("corpus"))
    splits = [p["split"] for p in plan]
    assert {k: splits.count(k) for k in ("train", "val", "test")} == wc.CORPUS_SPLITS
    assert sum(p["shifted"] for p in plan) == 225 and sum(p["misaimed"] for p in plan) == 150
    for p in plan:
        assert np.linalg.norm(p["misaim_m"]) <= wc.MISAIM_RADIUS_M + 1e-12
        assert (np.linalg.norm(p["misaim_m"]) > 0) == p["misaimed"]
    assert [p["noise_level"] for p in plan[:5]] == [0, 1, 2, 3, 0]


def test_token_grid_and_lewm_adapter_width_are_the_task066_ones():
    from embodied_jepa import token_dynamics as td
    from embodied_jepa.contracts import EE_DELTA_GRASP_V0

    assert wc.W_TOKEN_GRID == td.TOKEN_GRID == 4
    assert td.MODEL_CONFIG["max_horizon"] >= wc.CHUNK == 16
    assert EE_DELTA_GRASP_V0.dimension == 14
    source = (ROOT / "src" / "embodied_jepa" / "models" / "lewm.py").read_text()
    assert "Embedder(input_dim=14" in source  # the per-step 14-D adapter, unchanged


def test_owner_decisions_and_labels_are_recorded():
    assert wc.OWNER_DECISIONS["D7"] == "Critic first, planner fallback"
    assert wc.OWNER_DECISIONS["D1_D2"] == "Accept, diagnostic only"
    assert wc.OWNER_DECISIONS["D3_D4_D6"] == "Use the defaults"
    assert "privileged expert labels" in wc.LEARNED_LABEL and "not LeWM" in wc.LEARNED_LABEL
    assert wc.CARRIED == fm.CARRIED == "P-3"
    assert wc.P3_CHECKPOINT_SHA256.startswith("7988162d")


def test_relative_chunk_errors():
    from embodied_jepa import hybrid_selection as hs

    executed = np.ones((40, 7))
    chunks = {
        4: {"requested_free": np.ones((16, 7)).tolist(), "feasible": True},
        20: {"requested_free": (np.ones((16, 7)) * 2).tolist(), "feasible": True},
        30: {"requested_free": np.ones((16, 7)).tolist(), "feasible": True},
    }
    errors = hs.relative_chunk_errors(chunks, executed)
    assert errors == pytest.approx([0.0, 1.0])  # the chunk at 30 runs past the attempt


# ----- the runner ---------------------------------------------------------------------------------
def test_gated_stage_is_refused_without_an_authorisation_record():
    pytest.importorskip("torch")
    runner = _load_script("_run_wm_critic_v2", "scripts/run_wm_critic_v2.py")
    with pytest.raises(wc.GuardError, match="authorisation"):
        runner.stage_gated({}, {"gated_authorization": None}, None, None, None)


# ----- simulation (graphics opt-in; smoke seeds only) ---------------------------------------------
def _robot():
    from embodied_jepa import first_policy_v2_runtime as rt2

    return rt2.make_robot(), rt2.configured_bounds()


@render
def test_plate_shift_moves_the_plate_before_the_step_and_the_scorer_reads_it():
    pytest.importorskip("mujoco")
    from embodied_jepa import first_policy_v2_runtime as rt2
    from embodied_jepa import plate_shift as ps
    from embodied_jepa.policy_diagnostics import hold_controller

    robot, bounds = _robot()
    try:
        reset = wc.wide_reset_values(53998)
        truth, scorer, counter, _obs, facts = rt2.reset_and_look(robot, 53998, reset)
        hook = ps.PlateShift(robot, 3, [0.0, 0.04])
        frames = []

        class Recorder:
            def act(self, observation, step):
                frames.append(np.asarray(observation.images[fp2.CAMERA][0]).copy())
                return hold.act(observation, step)

            def advance(self, result):
                return hold.advance(result)

        hold = rt2.HarnessController(hold_controller())
        record = rt2.run_attempt(
            robot, scorer, counter, Recorder(), bounds=bounds, max_steps=6, settle_steps=0
        )
        hook.remove()
        assert hook.applied and hook.log["step"] == 3
        moved = np.asarray(truth["plate_position"][:2]) + [0.0, 0.04]
        assert np.allclose(robot.sim.data.body("plate").xpos[:2], moved)
        assert not np.array_equal(frames[2], frames[3])  # the plate moved before step 3's frame
        score = scorer.evaluate()
        apple = robot.sim.data.body("apple").xpos[:2]
        assert np.isclose(score["object_plate_distance_m"], np.linalg.norm(apple - moved))
        assert record["task_truth_in_controller"] == 0
        assert "observe" not in vars(robot)
    finally:
        robot.close()


@render
def test_branches_restore_the_live_state_bit_for_bit():
    pytest.importorskip("mujoco")
    from embodied_jepa import first_policy_v2_runtime as rt2
    from embodied_jepa import sim_selector as ss
    from embodied_jepa.policy_diagnostics import RandomController

    robot, bounds = _robot()
    lower, upper = bounds
    reset = wc.wide_reset_values(53997)

    def trajectory(branch_at):
        truth, scorer, counter, _obs, _f = rt2.reset_and_look(robot, 53997, reset)
        counter.remove()
        main = rt2.HarnessController(RandomController(1, lower, upper))
        brancher = ss.Brancher(robot)
        states = []
        for step in range(8):
            observation = robot.observe()
            if step == branch_at:
                others = [rt2.HarnessController(RandomController(k, lower, upper)) for k in (2, 3)]
                offsets, reasons = brancher.outcomes(
                    others, step, bounds, truth["plate_position"][:2], horizon=3
                )
                assert reasons == [None, None] and np.isfinite(offsets).all()
            command = np.clip(main.act(observation, step), lower, upper)
            projection = robot.project_candidates(command[None, None, None])
            main.advance(robot.execute(projection.actions[0, 0, 0]))
            states.append(robot.sim.data.qpos.copy())
        robot.stop("test")
        return np.stack(states)

    try:
        assert np.array_equal(trajectory(None), trajectory(4))
    finally:
        robot.close()


@render
def test_reread_copy_and_n_hybrids_issue_identical_commands():
    """P-reread, H-copy and H-N (an action-blind critic) are the same controller in effect."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    from embodied_jepa import first_policy_v2_model as fm2
    from embodied_jepa import first_policy_v2_runtime as rt2
    from embodied_jepa import hybrid_selection as hs
    from embodied_jepa import pretrained_encoder as pe

    torch.manual_seed(0)
    model = fm2.PolicyNet(1).eval()
    standardiser = rt2.Standardiser(np.zeros(132), np.ones(132))
    robot, bounds = _robot()
    fk = rt2.PalmFK()
    standin = hs.KinematicStandIn(bounds)
    encoder = pe.random_init()
    rng = np.random.default_rng(0)
    r_mid = RidgeReadout(
        np.zeros(98304),
        np.ones(98304),
        rng.normal(size=(98304, 2)) * 1e-6,
        np.array([0.49, -0.09]),
        1.0,
    )
    r_off = RidgeReadout(
        np.zeros(6144), np.ones(6144), rng.normal(size=(6144, 2)) * 1e-4, np.array([0.0, 0.0]), 1.0
    )
    reset = wc.wide_reset_values(53996)
    o_star = {18: (0.0, 0.0), 34: (0.0, 0.0)}  # the test's decision steps 2 and 18
    runs = {}
    try:
        for name, critic in (
            ("reread", None),
            ("copy", LatentCritic(CopyLast(), r_off, o_star)),
            ("n", LatentCritic(ZeroActions(_ActionBlind()), r_off, o_star)),
        ):
            truth, scorer, counter, _obs, _f = rt2.reset_and_look(robot, 53996, reset)
            estimates = [*truth["object_position"][:2], *truth["plate_position"][:2]]
            controller = hs.HybridController(
                fm2.predictor(model),
                standardiser,
                estimates,
                fk,
                bounds,
                variant="reread" if critic is None else "critic",
                policy=hs.P3Policy(model, standardiser, fk),
                encoder=encoder,
                r_mid=r_mid,
                critic=critic,
                standin=standin,
                decision_steps=(2, 18),
            )
            record = rt2.run_attempt(
                robot, scorer, counter, controller, bounds=bounds, max_steps=24, settle_steps=0
            )
            assert rt2.privileged_reads_ok(record)
            assert [d["chosen"] for d in controller.decisions] == [wc.INCUMBENT_INDEX] * 2
            runs[name] = record["commands"]
        assert np.array_equal(runs["reread"], runs["copy"])
        assert np.array_equal(runs["reread"], runs["n"])
    finally:
        robot.close()
        fk.close()
        standin.close()


def test_o2_moving_cohort_is_task054s():
    from embodied_jepa import world_model_v2 as wm2

    assert wc.O2["moving_threshold_m"] == wm2.MOVING_THRESHOLD_M == 0.01
    assert wc.O2["first_step"] == 416 and wc.O0["median_incumbent_regret_min_cm"] == 0.5


def test_harm_is_checked_first_and_closes_the_line():
    n = {
        "B-oracle-shift": 60,
        "P-stale": 10,
        "P-reread": 20,
        "H-LeWM": 40,
        "H-N": 20,
        "H-shuf": 20,
        "H-rand": 20,
        "H-sim": 45,
        "H-LeWM-s1": 38,
        "H-LeWM-s2": 36,
    }
    harm_u = {a: np.ones(32, bool) for a in wc.U_ARMS}
    harm_u["H-LeWM"] = np.r_[np.ones(29, bool), np.zeros(3, bool)]
    # no gain + harm
    no_gain = wc.decide_gated(_gated(n | {"H-LeWM": 21}), harm_u, _harness())
    assert no_gain["row"] == "HYB-HARM" and no_gain["abandonment_clause_fires"]
    # scene-blind + harm
    blind = wc.decide_gated(_gated(n | {"H-shuf": 38}), harm_u, _harness())
    assert blind["row"] == "HYB-HARM" and blind["abandonment_clause_fires"]
    # slow + harm
    slow = wc.decide_gated(_gated(n), harm_u, _harness(median_decision_seconds=0.3))
    assert slow["row"] == "HYB-HARM"


def test_every_row_has_a_consequence_and_the_owner_rulings_hold():
    assert set(wc.ALL_ROWS) <= set(wc.ROW_CONSEQUENCES)
    for row in wc.ALL_ROWS:
        consequence = wc.ROW_CONSEQUENCES[row]
        assert (consequence == "clause") == wc.abandonment_fires(row), row
        assert (consequence == "fallback") == (row in wc.FALLBACK_ROWS), row
    assert "R-NO-HEADROOM" in wc.FALLBACK_ROWS and not wc.abandonment_fires("HYB-SLOW")
    assert wc.OWNER_DECISIONS["harm rule"] == "Harm also closes the line (Recommended)"
    assert wc.GATED["g7_arm"] == "H-LeWM"
    assert wc.GATED["determinism_rerun_seeds"] == wc.seeds_of("S")[:4]


def test_scripts_pin_the_frozen_thread_environment():
    for name in ("run_wm_critic_v2.py", "collect_apple_shift_v2.py", "rank_wm_critic_v2.py"):
        text = (ROOT / "scripts" / name).read_text()
        pinned = text.index("os.environ.update(")
        assert pinned < text.index("import numpy") if "import numpy" in text else True
        for key, value in wc.THREAD_ENV.items():
            assert f'"{key}": "{value}"' in text, (name, key)


def test_h_shuf_foreign_latents_have_a_preregistered_fallback():
    full = {t: np.full(3, t, np.float32) for t in wc.DECISION_STEPS}
    early = {t: np.full(3, -t, np.float32) for t in wc.DECISION_STEPS if t <= 400}
    latents = [full, early, {}, full]
    out, subs = wc.shuf_latents(latents, 0)  # the next reset (1) ended after step 400
    assert set(out) == set(wc.DECISION_STEPS)
    assert np.array_equal(out[400], early[400])
    assert np.array_equal(out[576], early[400])  # its latest latent before the step
    assert {s["step"] for s in subs} == {t for t in wc.DECISION_STEPS if t > 400}
    out, subs = wc.shuf_latents(latents, 1)  # reset 2 reached nothing: reset 3 serves
    assert np.array_equal(out[304], full[304]) and all(s["reset_index"] == 3 for s in subs)
    out, subs = wc.shuf_latents(latents, 3)  # wraps to reset 0, never its own
    assert np.array_equal(out[576], full[576]) and subs == []
    with pytest.raises(wc.GuardError):
        wc.shuf_latents([full, {}], 0)


def test_o4_chooses_on_stand_in_feasibility_only():
    from embodied_jepa import wm_critic_v2_offline as off

    class Fixed:
        def __init__(self, costs):
            self.c = np.asarray(costs, float)

        def costs(self, latent, chunks, target):
            return self.c

    true = np.linspace(0.0, 2.4, 25)
    true[3] = np.nan  # this branch stopped: its truth is unknown to the closed loop
    group = {
        "seed": 1,
        "point_index": 0,
        "step": 480,
        "latent": np.zeros(3),
        "chunks": np.zeros((25, 16, 14)),
        "feasible": np.ones(25, bool),
        "true_costs_cm": true,
    }
    critic_costs = np.full(25, 5.0)
    critic_costs[3] = 0.0  # W prefers the stopped branch
    stats = off.rank_statistics(
        [group, group | {"seed": 2}],
        {"W0": Fixed(critic_costs), "N": Fixed(np.ones(25)), "copy": Fixed(np.ones(25))},
    )
    worst = np.nanmax(true) - np.nanmin(true)
    assert stats["W0"]["O4"]["regret_w"]["median"] == pytest.approx(worst)


# ----- memory and signals (K0 run-1 V, protocol §15) ----------------------------------------------
def test_memory_fields_and_worker_counts_are_frozen():
    assert wc.MEMORY["ceiling_gib"] == 12.0 and wc.MEMORY["headroom_gib"] == 4.0
    assert wc.SIM_WORKERS == 6 and wc.H_WORKERS == 4 and wc.H_WORKER_TORCH_THREADS == 4
    frozen = wc.frozen_block()
    assert frozen["memory"] == json.loads(json.dumps(wc.MEMORY))
    assert frozen["sim_workers"] == 6


HAS_PROC = Path("/proc/self/status").exists()  # the memory guard is Linux-only, like the runs


def test_process_tree_rss_counts_children():
    pytest.importorskip("torch")
    import subprocess
    import sys

    runner = _load_script("_run_wm_critic_v2_rss", "scripts/run_wm_critic_v2.py")
    if not HAS_PROC:  # macOS CI: no /proc; the runs themselves are Linux-only (G-platform)
        assert runner.process_tree_rss_bytes() == 0
        with pytest.raises(OSError):
            runner.mem_available_bytes()
        return
    alone = runner.process_tree_rss_bytes()
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; x = bytearray(200 * 2**20); time.sleep(20)"]
    )
    try:
        import time

        for _ in range(100):
            time.sleep(0.1)
            per: dict = {}
            total = runner.process_tree_rss_bytes(per=per)
            if per.get(child.pid, 0) > 150 * 2**20:
                break
        assert child.pid in per and total > alone + 150 * 2**20
    finally:
        child.kill()
    assert runner.mem_available_bytes() > 0


_SIGNAL_CHILD = """
import importlib.util, json, sys, time, types
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    "_r", Path(sys.argv[1]) / "scripts/run_wm_critic_v2.py"
)
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
out = Path(sys.argv[2])
if sys.argv[3] == "memory":  # any tree exceeds this: the guard must fire (on any platform)
    runner.process_tree_rss_bytes = lambda root=None, per=None: 10**13
def slow_preflight(report, kind, evidence):
    (out.parent / "ready").write_text("1")
    time.sleep(60)
runner.preflight = slow_preflight
args = types.SimpleNamespace(output=str(out), evidence=".", mode="preflight", smoke=False,
                             workers=None)
report = runner.run(args)
sys.exit(0 if report["outcome"] != "V" else 3)
"""


@pytest.mark.parametrize("how", ["SIGTERM", "SIGINT", "SIGHUP", "memory"])
def test_a_stop_signal_or_the_memory_ceiling_writes_the_v_report(tmp_path, how):
    pytest.importorskip("torch")
    import signal
    import subprocess
    import sys
    import time

    out = tmp_path / "run"
    child = subprocess.Popen([sys.executable, "-c", _SIGNAL_CHILD, str(ROOT), str(out), how])
    try:
        for _ in range(600):
            if (tmp_path / "ready").exists() or child.poll() is not None:
                break
            time.sleep(0.1)
        if how != "memory":
            time.sleep(1.5)  # let the memory watch take a few samples first
            child.send_signal(getattr(signal, how))
        assert child.wait(timeout=60) == 3
    finally:
        if child.poll() is None:
            child.kill()
    report = json.loads((out / "report.json").read_text())
    assert report["outcome"] == "V"
    if how == "memory":
        assert "G-memory" in report["void_reason"] and "SIGUSR1" in report["void_reason"]
    else:
        assert f"received {how}" in report["void_reason"]
    assert report["interrupted_utc"].endswith("Z")
    if HAS_PROC or how == "memory":
        assert report["memory"]["peak_tree_rss_gib"] > 0


class _FakeRenderer:
    """A robot whose sim.render returns queued frames (for the render-only retry)."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.sim = self

    def render(self):
        return self.frames.pop(0)


def test_g_frame_retries_the_render_only_and_refuses_a_persistent_mismatch():
    from embodied_jepa import wm_critic_v2_runtime as rtm

    good = np.zeros((112, 112, 3), np.uint8)
    glitch = good.copy()
    glitch[5, 5, 0] = 1  # the renderer's one-level difference
    task = {"seed": 53999, "expected_frame_sha256": rtm.frame_sha(good)}
    assert rtm.check_post_look_frame(_FakeRenderer([]), task, rtm.frame_sha(good)) == []
    retries = rtm.check_post_look_frame(_FakeRenderer([good]), task, rtm.frame_sha(glitch))
    assert retries == [rtm.frame_sha(glitch), rtm.frame_sha(good)]
    with pytest.raises(wc.GuardError, match="after 2 re-renders"):
        rtm.check_post_look_frame(_FakeRenderer([glitch, glitch]), task, rtm.frame_sha(glitch))


def test_render_majority_resolves_a_single_odd_render_and_refuses_three_different():
    pytest.importorskip("torch")
    runner = _load_script("_run_wm_critic_v2_major", "scripts/run_wm_critic_v2.py")

    class FakePool:
        def __init__(self, shas):
            self.shas = list(shas)

        def map(self, tasks, cap, what):
            return [{"seed": t["seed"], "post_look_frame_sha256": self.shas.pop(0)} for t in tasks]

    tasks = [{"kind": "frame", "seed": 53998}, {"kind": "frame", "seed": 53999}]
    chosen, odd = runner.render_majority(FakePool(["a", "b", "a", "c", "b"]), tasks, 1, "x")
    assert [c["post_look_frame_sha256"] for c in chosen] == ["a", "b"]
    assert odd == {"53999": ["b", "c", "b"]}
    with pytest.raises(wc.GuardError, match="no majority"):
        runner.render_majority(FakePool(["a", "b", "c"]), tasks[:1], 1, "x")


def test_the_corpus_stage_records_its_first_render_and_rank_uses_the_h_layout():
    text = (ROOT / "scripts" / "collect_apple_shift_v2.py").read_text()
    assert text.index('report["cohort_first_render_utc"]') < text.index("records = pool.map(tasks")
    assert "RUN.h_workers()" in (ROOT / "scripts" / "rank_wm_critic_v2.py").read_text()
