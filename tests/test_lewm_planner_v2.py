"""TASK-074 (``apple_lewm_planner_v2``): the preregistered design, its guards and its mechanics.

NumPy tests run in the core job; MuJoCo and torch tests import-skip without them; nothing here
simulates a K1, R, D3, corpus, S or U seed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import lewm_planner_v2 as lp
from embodied_jepa import wm_critic_v2 as wc
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "benchmarks" / "manifests" / "apple-lewm-planner-v2.json").read_text()
)
render = pytest.mark.skipif(os.environ.get("JEPA_TEST_RENDER") != "1", reason="graphics opt-in")
# A literal: the gated-stage authorisation PR may not change it (AUTHORISATION_SCOPE).
FROZEN_SHA256 = "2cf80f5aa54d509e3801bcb3934409407da9b6a0600e6856f319ddeda1d2e36a"


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----- the frozen block, the pins and the stored cohorts ------------------------------------------
def test_frozen_hash_is_the_preregistered_literal():
    assert lp.frozen_sha256() == FROZEN_SHA256 == MANIFEST["frozen_sha256"]
    assert "may not change lewm_planner_v2.py" in lp.AUTHORISATION_SCOPE


def test_manifest_frozen_block_equals_the_module():
    assert MANIFEST["frozen"] == lp.frozen_block()
    assert MANIFEST["gated_authorization"] is None


def test_pinned_files_match():
    for relative, want in MANIFEST["hashes"].items():
        got = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert got == want, relative
    for own in (
        "src/embodied_jepa/lewm_planner_v2.py",
        "src/embodied_jepa/place_planner.py",
        "src/embodied_jepa/run_guards.py",
        "scripts/run_lewm_planner_v2.py",
        "scripts/run_wm_critic_v2.py",
    ):
        assert own in MANIFEST["hashes"]


def test_the_protocol_document_sha_is_recorded():
    """The document is not pinned (the smoke record is added after the smokes), but its sha256
    at merge is recorded in the manifest, and the authorisation PR may not edit it."""
    doc = ROOT / "docs" / "experiments" / "apple_lewm_planner_v2.md"
    assert hashlib.sha256(doc.read_bytes()).hexdigest() == MANIFEST["protocol_document_sha256"]
    assert "apple_lewm_planner_v2.md" in lp.AUTHORISATION_SCOPE
    assert "check_authorisation" in lp.AUTHORISATION_SCOPE
    assert "tests/test_lewm_planner_v2.py" in MANIFEST["hashes"]


def test_task073_files_are_unchanged_by_this_task():
    t073 = json.loads((ROOT / "benchmarks" / "manifests" / "apple-wm-critic-v2.json").read_text())
    for relative, want in t073["hashes"].items():
        assert MANIFEST["hashes"][relative] == want, relative


def test_stored_cohorts_are_the_generated_values_and_their_digest():
    for role in lp.COHORT_ROLES:
        node = MANIFEST["cohorts"][role]
        assert lp.cohort_digest(node["values"]) == node["sha256"]
        generated = lp.cohort_values(role)
        assert set(generated) == set(node["values"])
        for seed, entry in generated.items():
            stored = node["values"][seed]
            assert stored["redraw"] == entry["redraw"]
            assert np.allclose(stored["object_xy"], entry["object_xy"], atol=1e-12)
            assert np.allclose(stored["plate_xy"], entry["plate_xy"], atol=1e-12)
            if role != "U":
                for cm in lp.SHIFT_GRID_CM:
                    assert np.allclose(
                        stored["shift_m"][str(cm)], entry["shift_m"][str(cm)], atol=1e-12
                    )
        assert len(lp.stored_cohort(MANIFEST, role)) == len(lp.seeds_of(role))


def test_a_tampered_cohort_is_refused():
    import copy

    bad = copy.deepcopy(MANIFEST)
    bad["cohorts"]["K1"]["values"]["54000"]["plate_xy"][0] += 1e-3
    with pytest.raises(lp.GuardError, match="G-cohort"):
        lp.stored_cohort(bad, "K1")


# ----- seeds --------------------------------------------------------------------------------------
def test_seed_block_is_the_owners_and_disjoint_from_everything_spent():
    lp.check_seed_ranges()
    assert lp.SEED_RANGES == {
        "K1": (54000, 54031),
        "R": (54040, 54071),
        "D3": (54100, 54115),
        "corpus": (54200, 54499),
        "S": (54500, 54563),
        "U": (54600, 54631),
    }
    assert lp.SMOKE_SEEDS == (54650, 54699)
    assert lp.DESIGN_PROBE_SEEDS == (54700, 54999)
    assert lp.FORBIDDEN_RANGES["task074_design_probes"] == (54700, 54999)
    assert lp.FORBIDDEN_RANGES["task073_block"] == (53000, 53999)


@pytest.mark.parametrize("seed", [54700, 54850, 54999, 53000, 53800, 45300, 52000])
def test_role_guard_refuses_spent_ranges(seed):
    with pytest.raises(lp.GuardError, match="forbidden"):
        lp.check_role_seeds("K1", (seed,))
    with pytest.raises(lp.GuardError, match="forbidden"):
        lp.check_role_seeds("K1", (seed,), smoke=True)


def test_role_guard_is_a_whitelist_of_plain_ints_in_order():
    assert lp.check_role_seeds("K1", lp.seeds_of("K1")) == lp.seeds_of("K1")
    with pytest.raises(lp.GuardError):
        lp.check_role_seeds("K1", tuple(reversed(lp.seeds_of("K1"))))
    with pytest.raises(lp.GuardError, match="plain ints"):
        lp.check_role_seeds("K1", (np.int64(54000),))
    with pytest.raises(lp.GuardError, match="smoke"):
        lp.check_role_seeds("K1", (54000,), smoke=True)
    assert lp.check_role_seeds("K1", (54650, 54651), smoke=True) == (54650, 54651)


# ----- the condition ------------------------------------------------------------------------------
def test_sign_convention_matches_the_scene():
    from embodied_jepa.resting_expert import RIGHT_SHOULDER_BASE

    assert RIGHT_SHOULDER_BASE[1] < 0  # the right shoulder is at -y
    assert lp.RESET_CENTERS["plate_xy"][1] < 0
    assert "-y is the robot's right" in lp.SIGN_CONVENTION
    assert lp.DIRECTION_ARC_DEG == (225, 315)


def test_stored_shifts_point_right_have_the_size_and_keep_e9s_release_geometry():
    for role in ("K1", "R", "D3", "S"):
        for seed, entry in lp.stored_cohort(MANIFEST, role).items():
            plate = np.asarray(entry["plate_xy"])
            before = wc.release_xy(plate) - plate
            for cm in lp.SHIFT_GRID_CM:
                v = np.asarray(entry["shift_m"][str(cm)])
                assert math.isclose(np.linalg.norm(v), cm / 100.0, rel_tol=1e-9)
                assert v[1] <= -cm / 100.0 * math.cos(math.radians(45)) + 1e-12, (seed, cm)
                moved = plate + v
                after = wc.release_xy(moved) - moved
                assert np.linalg.norm(after - before) <= wc.ARC_TOLERANCE_M + 1e-12


def test_condition_resets_redraw_only_when_no_right_direction_exists():
    redrawn = [s for s in lp.seeds_of("K1") if lp.condition_reset(s)["redraw"] > 0]
    assert redrawn == [54011]
    plain = lp.wide_reset_values(54011)
    assert not all(
        lp.eligible_right_directions(plain["plate_xy"], plain["object_xy"], cm / 100.0)
        for cm in lp.REDRAW_CM
    )
    r = lp.condition_reset(54011)
    for cm in lp.REDRAW_CM:
        assert lp.eligible_right_directions(r["plate_xy"], r["object_xy"], cm / 100.0)
    for key, j in (("object_xy", lp.WIDE_JITTER_M["object_xy"]), ("plate_xy", 0.02)):
        assert np.all(np.abs(np.subtract(r[key], lp.RESET_CENTERS[key])) <= j)
    assert lp.condition_reset(54000) == lp.wide_reset_values(54000) | {"redraw": 0}


def test_corpus_plan_is_fixed_and_has_the_declared_shape():
    plan = lp.corpus_plan()
    assert plan == lp.corpus_plan()
    assert [r["seed"] for r in plan] == list(lp.seeds_of("corpus"))
    splits = {k: sum(r["split"] == k for r in plan) for k in ("train", "val", "test")}
    assert splits == lp.CORPUS_SPLITS
    assert sum(r["shift"] is None for r in plan) == 75
    assert sum(r["misaimed"] for r in plan) == 150
    assert {r["shift"]["cm"] for r in plan if r["shift"]} == set(lp.CORPUS_SHIFT_CM)
    for r in plan:
        assert np.linalg.norm(r["misaim_m"]) <= lp.MISAIM_RADIUS_M + 1e-12
        if r["shift"]:
            assert r["shift"]["step"] == lp.SHIFT_STEP
            assert r["shift"]["vector"][1] < 0
    assert lp.plan_digest(plan) == MANIFEST["corpus_plan_sha256"]


# ----- the planner --------------------------------------------------------------------------------
def test_decision_steps_grid_and_clock():
    assert lp.TRANSFER_START == 405 and lp.LOWER_START == 505 and lp.PLACE_END == 725
    assert lp.DECISION_STEPS == (405, 421, 437, 453, 469, 485)
    assert len(lp.COARSE_OFFSETS_M) == 49 and len(lp.FINE_OFFSETS_M) == 25
    assert lp.CANDIDATES_PER_DECISION == 74
    assert lp.COARSE_OFFSETS_M[lp.COARSE_INCUMBENT] == (0.0, 0.0)
    assert lp.FINE_OFFSETS_M[lp.FINE_CENTRE] == (0.0, 0.0)
    coarse = lp.coarse_targets([0.49, -0.09])
    assert coarse.shape == (49, 2)
    assert np.isclose(coarse[:, 1].min(), -0.24) and np.isclose(coarse[:, 1].max(), -0.06)
    assert np.allclose(lp.fine_targets(coarse[3])[lp.FINE_CENTRE], coarse[3])


def test_tie_rule_keeps_the_incumbent():
    costs = np.full(49, 2.0)
    assert lp.choose(costs, lp.COARSE_INCUMBENT) == lp.COARSE_INCUMBENT
    costs[5] = 2.0 - 1e-7
    assert lp.choose(costs, lp.COARSE_INCUMBENT) == lp.COARSE_INCUMBENT
    costs[5] = 1.0
    assert lp.choose(costs, lp.COARSE_INCUMBENT) == 5
    costs[lp.COARSE_INCUMBENT] = np.inf
    assert lp.choose(costs, lp.COARSE_INCUMBENT) == 5
    assert lp.choose(np.full(25, np.inf), lp.FINE_CENTRE) == lp.FINE_CENTRE


def test_cost_is_the_distance_to_the_terminal_offset():
    assert lp.O_FINAL_CM == lp.O_STAR_CM[501]
    assert set(lp.cost_targets()) == {t + lp.CHUNK for t in lp.DECISION_STEPS}
    assert all(v == lp.O_FINAL_CM for v in lp.cost_targets().values())
    assert np.isclose(lp.terminal_cost_cm(np.asarray(lp.O_FINAL_CM) + [3.0, 4.0]), 5.0)
    assert set(lp.O_STAR_CM) == {t + lp.CHUNK for t in lp.DECISION_STEPS}


def test_o_star_source_is_task073s():
    assert lp.O_STAR_SOURCE == wc.O_STAR_SOURCE
    assert lp.O_STAR_SOURCE["corpus_manifest_sha256"].startswith("67c342f6")


# ----- stage rules --------------------------------------------------------------------------------
def test_k1_bars_in_order_and_the_smallest_passing_size():
    good = {"B-oracle-shift": 32, "H-handover": 32, "P-stale": 0, "P-truth": 20}
    assert lp.k1_cell_passes(good)["passes"]
    assert lp.k1_next_arm({}) == "B-oracle-shift"
    assert lp.k1_next_arm({"B-oracle-shift": 29}) is None
    assert lp.k1_next_arm({"B-oracle-shift": 30, "H-handover": 29}) is None
    assert lp.k1_next_arm({"B-oracle-shift": 30, "H-handover": 30}) == "P-stale"
    assert lp.k1_next_arm({"B-oracle-shift": 30, "H-handover": 30, "P-stale": 5}) is None
    assert not lp.k1_cell_passes(good | {"P-truth": 25})["passes"]
    assert lp.k1_cell_passes(good | {"P-truth": 24})["passes"]
    assert lp.k1_select({9: good})["shift_cm"] == 9
    assert lp.k1_select({9: good | {"P-truth": 30}, 12: good})["shift_cm"] == 12
    assert lp.k1_select({9: good | {"P-stale": 5}})["row"] == "L-NO-CONDITION"


def _gate(passes=True, void=False):
    return {"passes": passes, "void": void}


def test_offline_rows_first_match():
    def seeds(**over):
        base = {"O1": _gate(), "O2": _gate(), "O3": _gate(), "O4": _gate()}
        return {s: base | over for s in lp.MODEL_SEEDS}

    ok = {"passes": True}
    assert lp.decide_offline(seeds(), ok)["row"] == "OFFLINE-PASS"
    assert lp.decide_offline(seeds(O1=_gate(False)), ok)["row"] == "L-NO-DYNAMICS"
    assert lp.decide_offline(seeds(O2=_gate(False, True)), ok)["row"] == "L-O2-VOID"
    assert lp.decide_offline(seeds(O2=_gate(False)), ok)["row"] == "L-G2A"
    assert lp.decide_offline(seeds(O3=_gate(False, True)), ok)["row"] == "L-O3-VOID"
    assert lp.decide_offline(seeds(O4=_gate(False)), ok)["row"] == "L-NO-RANK"
    assert lp.decide_offline(seeds(), {"passes": False})["row"] == "L-PROPOSAL"
    assert lp.decide_offline(seeds(O2=_gate(False)), ok)["abandonment_clause_fires"]


def test_d3_stop_rules():
    base = {a: 0 for a in lp.D3_ARMS}
    assert lp.decide_d3(base | {"H-handover": 16, "P-truth": 13})["row"] == "L-NO-HEADROOM"
    go = base | {"H-handover": 16, "P-truth": 8, "L-plan": 12, "L-N": 9, "L-shuf": 2}
    assert lp.decide_d3(go)["row"] == "D3-GO"
    assert lp.decide_d3(go | {"L-shuf": 10})["row"] == "L-DEV-STOP"


def _s(**counts):
    out = {}
    for arm in lp.S_ARMS:
        k = counts.get(arm, 0)
        out[arm] = np.arange(lp.S_RESETS) < k
    return out


def _u(**counts):
    return {a: np.arange(lp.U_RESETS) < counts.get(a, 30) for a in lp.U_ARMS}


HARNESS = {
    "b_hold_grasps": 0,
    "b_random_grasps": 0,
    "privileged_ok": {a: True for a in lp.L1_ARMS},
    "determinism_ok": True,
    "cuda_allocation_failed": False,
    "median_decision_seconds": 0.5,
}


BASE = {
    "B-oracle-shift": 64,
    "H-handover": 64,
    "P-truth": 20,
    "P-stale": 0,
    "L-plan": 60,
    "L-plan-s1": 58,
    "L-plan-s2": 59,
    "H-twin": 62,
}


def test_gated_rows_first_match():
    base = BASE
    out = lp.decide_gated(_s(**base), _u(), HARNESS)
    assert out["row"] == "L-PASS" and out["claim"] and out["gates"]["T"]
    twin = lp.decide_gated(_s(**base | {"H-twin": 64, "L-plan": 57}), _u(), HARNESS)
    assert twin["row"] == "L-TWIN-BETTER" and not twin["claim"]
    assert twin["abandonment_clause_fires"]
    assert lp.decide_gated(_s(**base), _u(**{"L-plan": 27}), HARNESS)["row"] == "L-HARM"
    slow = HARNESS | {"median_decision_seconds": 1.2}
    assert lp.decide_gated(_s(**base), _u(), slow)["row"] == "L-SLOW"
    slow_twin = lp.decide_gated(_s(**base | {"H-twin": 64, "L-plan": 57}), _u(), slow)
    assert slow_twin["row"] == "L-TWIN-BETTER"  # L-SLOW needs T
    assert lp.decide_gated(_s(**base | {"B-oracle-shift": 59}), _u(), HARNESS)["row"] == "VOID"
    assert lp.decide_gated(_s(**base | {"P-stale": 9}), _u(), HARNESS)["row"] == (
        "S-VOID-CONDITION"
    )
    assert lp.decide_gated(_s(**base | {"H-handover": 35}), _u(), HARNESS)["row"] == (
        "L-NO-HEADROOM"
    )
    blind = base | {"L-shuf": 58}
    assert lp.decide_gated(_s(**blind), _u(), HARNESS)["row"] == "L-SCENE-BLIND"
    blind_rand = blind | {"L-rand": 58}  # L-SCENE-BLIND needs G4
    assert lp.decide_gated(_s(**blind_rand), _u(), HARNESS)["row"] == "L-NO-GAIN"
    assert lp.decide_gated(_s(**base | {"L-plan": 25}), _u(), HARNESS)["row"] == "L-NO-GAIN"
    bad = HARNESS | {"determinism_ok": False}
    assert lp.decide_gated(_s(**base), _u(), bad)["row"] == "VOID"


def test_no_claim_row_fires_with_t_failed():
    """Owner D1 read literally: only L-PASS makes the claim, and never with T failed. The draws
    keep the headroom and the blind twins fixed and put L-plan and H-twin around the 6/64
    margin, so L-TWIN-BETTER and L-PASS are both reached (dropping T from L-PASS fails this)."""
    assert lp.CLAIM_ROWS == ("L-PASS",)
    rng = np.random.default_rng(1)
    rows = []
    for _ in range(300):
        plan = int(rng.integers(40, 65))
        counts = BASE | {
            "L-plan": plan,
            "H-twin": int(np.clip(plan + rng.integers(-3, 11), 0, 64)),
            "P-truth": 20,
            "H-handover": 64,
            "L-shuf": int(rng.choice([0, 5, 58])),
        }
        u = _u(**{"L-plan": int(rng.integers(27, 33)), "P-stale": 30})
        harness = HARNESS | {"median_decision_seconds": float(rng.choice([0.5, 1.2]))}
        out = lp.decide_gated(_s(**counts), u, harness, report_resamples=50)
        rows.append(out["row"])
        if not out["gates"]["T"]:
            assert not out["claim"] and out["row"] != "L-PASS"
        if out["claim"]:
            assert all(out["gates"].values())
        if out["row"] in lp.GATED_ROWS[3:]:  # every outcome row
            assert out["claim"] or out["abandonment_clause_fires"]
    for row in ("L-PASS", "L-TWIN-BETTER", "L-SLOW", "L-HARM"):
        assert row in rows, row


def test_t_and_g5_are_reported_with_intervals_and_blocked_resets_split_p_far():
    blocked = {a: np.zeros(lp.S_RESETS, bool) for a in ("L-plan", "P-far")}
    blocked["P-far"][:10] = True
    out = lp.decide_gated(_s(**BASE | {"P-far": 30}), _u(), HARNESS | {"blocked": blocked})
    t = out["t_reported"]
    assert t["only_first"] == 0 and t["only_second"] == 2
    assert {"only_first", "only_second"} <= set(out["g5_reported"])
    # the keep mask drops the resets on which either arm's move was blocked
    clean = out["p_far_reported_excluding_blocked"]
    keep = ~blocked["P-far"]
    s = _s(**BASE | {"P-far": 30})
    want = lp.paired_one_sided(np.asarray(s["L-plan"])[keep], np.asarray(s["P-far"])[keep])
    assert clean == want and clean["difference"] == (60 - 10) - (30 - 10)
    assert t["difference"] == -2 and t["ci95"][0] <= -2 <= t["ci95"][1]
    assert "not statistical non-inferiority" in t["note"]
    assert out["g5_reported"]["n"] == lp.U_RESETS
    assert out["shift_blocked_per_arm"] == {"L-plan": 0, "P-far": 10}
    assert out["p_far_reported_excluding_blocked"] is not None


def test_every_row_has_a_consequence_and_the_clause_rows():
    assert set(lp.ALL_ROWS) <= set(lp.ROW_CONSEQUENCES)
    assert set(lp.CLAUSE_ROWS) == {
        "L-G2A",
        "L-NO-RANK",
        "L-HARM",
        "L-TWIN-BETTER",
        "L-SLOW",
        "L-SCENE-BLIND",
        "L-NO-GAIN",
    }
    for row in lp.CLAUSE_ROWS:
        assert lp.ROW_CONSEQUENCES[row].startswith("clause") and lp.abandonment_fires(row)
    assert set(lp.GATED_ROWS[3:]) - set(lp.CLAIM_ROWS) <= set(lp.CLAUSE_ROWS)
    assert not lp.abandonment_fires("L-NO-HEADROOM")
    with pytest.raises(ContractError):
        lp.abandonment_fires("HYB-PASS")


def test_d3_reports_the_twin_and_ceiling_comparisons():
    counts = {a: 0 for a in lp.D3_ARMS} | {"H-handover": 16, "L-plan": 12, "H-twin": 15}
    out = lp.decide_d3(counts)
    assert out["descriptive"] == {"L-plan_minus_H-twin": -3, "L-plan_minus_H-handover": -4}


def _record(**over):
    base = {
        "frozen_sha256": lp.frozen_sha256(),
        "go_comment_url": "https://github.com/RaaSaaR-org/open-embodied-jepa/pull/113"
        "#issuecomment-123",
        "stage": "gated",
        "authorised_utc": "2026-10-01T12:00:00Z",
        "authorised_by": lp.AUTHORISED_BY_VALUES[1],
    }
    return base | over


NOW = __import__("datetime").datetime(2026, 10, 2, tzinfo=__import__("datetime").UTC)


def test_the_authorisation_record_is_validated_against_its_schema():
    assert lp.check_authorisation(_record(), now=NOW)["stage"] == "gated"
    owner = _record(authorised_by="authorised by the task owner")
    assert lp.check_authorisation(owner, now=NOW)["authorised_by"].endswith("task owner")
    assert set(lp.AUTHORISATION_SCHEMA) == set(_record())
    other_pr = "https://github.com/RaaSaaR-org/open-embodied-jepa/pull/114#issuecomment-1"
    for bad in (
        _record(frozen_sha256="0" * 64),
        _record(go_comment_url="https://example.com/x"),
        _record(go_comment_url=other_pr),
        _record(stage="d3"),
        _record(authorised_utc="yesterday"),
        _record(authorised_utc="2026-09-30T23:59:59Z"),  # before the frozen-code smokes
        _record(authorised_utc="2026-10-03T00:00:00Z"),  # in the future
        _record(authorised_by="someone"),
        _record() | {"extra": 1},
        {"ok": True},
    ):
        with pytest.raises(lp.GuardError, match="G-authorisation"):
            lp.check_authorisation(bad, now=NOW)


def test_budget_rule():
    assert lp.budget_updates([3000, 4000])["updates"] == 10_000
    assert lp.budget_updates([12_000])["updates"] == 25_000
    assert lp.budget_updates([40_000])["escalate"]


# ----- the re-run rule (run_guards) ---------------------------------------------------------------
def _attempt(**over):
    base = {
        "success": True,
        "grasp": True,
        "at_rest": True,
        "termination_reason": "policy_complete",
        "executed_steps": 725,
        "final_distance_cm": 3.0,
        "first_grasp_step": 265,
        "first_place_step": 613,
        "commands": np.zeros((725, 7), np.float32),
        "decisions": [{"chosen": [24, 12], "target": [0.49, -0.2]}],
    }
    return base | over


def test_rerun_verdict_voids_above_a_tolerance_and_reports_above_half():
    from embodied_jepa import place_planner as pp

    same = pp.rerun_verdict(_attempt(), _attempt())
    assert not same["void"] and same["above_half_tolerance"] == []
    half = pp.rerun_verdict(_attempt(), _attempt(final_distance_cm=3.7))
    assert not half["void"]
    assert [a["field"] for a in half["above_half_tolerance"]] == ["final_distance_cm"]
    assert pp.rerun_verdict(_attempt(), _attempt(final_distance_cm=4.5))["void"]


def test_rerun_rule_is_valid_and_covers_every_field():
    from embodied_jepa.run_guards import RerunRule

    rule = lp.rerun_rule()
    assert isinstance(rule, RerunRule) and rule.image_reading
    assert "max_abs_command_difference" in rule.tolerances
    assert "L-rand" not in lp.GATED["determinism_rerun_arms"]
    assert set(lp.GATED["determinism_rerun_arms"]) <= set(lp.IMAGE_READING_ARMS)


def test_rerun_compare_tolerates_small_target_moves_and_fails_nan():
    from embodied_jepa import place_planner as pp

    assert pp.rerun_compare(_attempt(), _attempt())["matches"]
    moved = _attempt(decisions=[{"chosen": [24, 12], "target": [0.495, -0.2]}])
    out = pp.rerun_compare(_attempt(), moved)
    assert out["matches"] and pp.target_difference_cm(_attempt(), moved) == pytest.approx(0.5)
    far = _attempt(decisions=[{"chosen": [24, 12], "target": [0.51, -0.2]}])
    assert not pp.rerun_compare(_attempt(), far)["matches"]
    nan = np.zeros((725, 7), np.float32)
    nan[3, 1] = np.nan
    assert not pp.rerun_compare(_attempt(), _attempt(commands=nan))["matches"]
    assert not pp.rerun_compare(_attempt(), _attempt(success=False))["matches"]
    fewer = _attempt(decisions=[])
    assert pp.target_difference_cm(_attempt(), fewer) == float("inf")


# ----- records ------------------------------------------------------------------------------------
def test_owner_decisions_are_verbatim_and_delegations_labelled():
    assert lp.OWNER_DECISIONS["D1"] == "Beat P-3, tie readout (Recommended)"
    assert lp.OWNER_DECISIONS["D2"] == "Put-down spots (Recommended)"
    assert lp.OWNER_DECISIONS["D3_D4"] == "Right-side move, place only (Recommended)"
    assert lp.OWNER_DECISIONS["D5"] == "Yes, add it (Recommended)"
    assert lp.DELEGATED == "decided by Claude under owner delegation, 2026-09-30"
    assert "54700-54999" in lp.DESIGN_PROBES["disclosure"]
    assert "after the design probes" in lp.DESIGN_PROBES["disclosure"]
    assert lp.GATED["t_twin_margin"] == 6


def test_void_rule_names_both_boundaries():
    assert "first_outcome_utc" in lp.VOID_RULE
    assert "cohort_first_render_utc" in lp.VOID_RULE
    source = (ROOT / "scripts" / "run_lewm_planner_v2.py").read_text()
    assert source.index('report["first_outcome_utc"]') < source.index("tok_val = ")


def test_the_runner_uses_run_guards_and_pins_threads():
    source = (ROOT / "scripts" / "run_lewm_planner_v2.py").read_text()
    assert "class Pool(rg.BoundedPool)" in source
    assert "rg.process_tree_memory" in source
    assert "rerun_compare" in source
    for key, value in lp.THREAD_ENV.items():
        assert f'"{key}": "{value}"' in source
    assert lp.MEMORY["measure_key"] == "pss" and "VmRSS" in lp.MEMORY["measure"]


def test_gated_stage_is_refused_without_an_authorisation_record(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    import functools

    runner = _load_script("_run_lewm_planner_v2", "scripts/run_lewm_planner_v2.py")
    monkeypatch.setattr(
        lp, "check_authorisation", functools.partial(lp.check_authorisation, now=NOW)
    )
    with pytest.raises(lp.GuardError, match="no owner or delegated"):
        runner.stage_gated({}, {"gated_authorization": None}, tmp_path, None, None)
    with pytest.raises(lp.GuardError, match="schema"):
        runner.stage_gated({}, {"gated_authorization": {"x": 1}}, tmp_path, None, None)
    report = {}
    with pytest.raises(lp.GuardError, match="written by its authorisation PR"):
        runner.stage_gated(report, {"gated_authorization": _record()}, tmp_path, None, None)
    assert report["gated_authorization"]["stage"] == "gated"


def test_a_blocked_shift_is_a_counted_failure_through_the_runner(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    from embodied_jepa import lewm_planner_v2_runtime as lrt

    runner = _load_script("_run_lewm_planner_v2_b", "scripts/run_lewm_planner_v2.py")
    task = {"shift": {"step": 300, "vector": [0.0, -0.09]}}
    record = {"seed": 54650, "arm": "L-plan", "post_look_state": [0.0]} | lrt.blocked_record(
        task, lp.SHIFT_BLOCKED_PREFIX + " ['apple_geom']", None
    )
    assert record["termination_reason"] == "shift_blocked" and not record["success"]
    assert runner.successes([record]) == [False]
    assert runner.blocked({"L-plan": [record]}) == {"L-plan": 1}
    stripped = runner.strip([record])[0]
    assert stripped["termination_reason"] == "shift_blocked" and "commands_sha256" in stripped


def test_offline_ranking_statistics_on_a_toy_group():
    from embodied_jepa import lewm_planner_v2_offline as off

    class Critic:
        def __init__(self, sign):
            self.sign = sign

        def costs(self, start, chunks, step):  # noqa: ARG002
            return self.sign * np.asarray(chunks)[:, 0, 0]

    rng = np.random.default_rng(0)
    groups = []
    for seed in range(6):
        for step in lp.DECISION_STEPS[:2]:
            for kind, k in (("coarse", 49), ("fine", 25)):
                true = rng.uniform(0, 5, k)
                chunks = np.zeros((k, 16, 14))
                chunks[:, 0, 0] = true
                groups.append(
                    {
                        "seed": seed,
                        "step": step,
                        "kind": kind,
                        "targets": rng.normal(size=(k, 2)).tolist(),
                        "anchor": [0.0, 0.0],
                        "reading": [0.1, 0.1],
                        "latent": np.zeros(4),
                        "chunks": chunks,
                        "feasible": np.ones(k, bool),
                        "true_costs_cm": true,
                    }
                )
    stats = off.rank_statistics(groups, {"W0": Critic(1.0), "copy": Critic(0.0), "N": Critic(0.0)})
    assert stats["W0"]["O3"]["rho_w"]["median"] == pytest.approx(1.0)
    assert stats["W0"]["O4"]["regret_w"]["median"] == pytest.approx(0.0)
    assert stats["blind"]["copy-last"]["median"] == 0.0
    assert "twin-distance" in stats["reported"]


# ----- MuJoCo: the place primitive's clock and reach ----------------------------------------------
@render
def test_place_primitive_keeps_e9s_clock_and_refuses_unreachable_targets():
    pytest.importorskip("mujoco")
    from embodied_jepa import first_policy_v2_runtime as rt2
    from embodied_jepa import place_planner as pp

    fk = rt2.PalmFK()
    prim = pp.PlacePrimitive(fk, [0.34, -0.18])
    prim.retarget([0.49, -0.20], 405)
    assert (prim.expert.phase_index, prim.expert.phase_step) == (4, 0)
    prim2 = pp.PlacePrimitive(fk, [0.34, -0.18])
    prim2.retarget([0.49, -0.20], 421)
    assert (prim2.expert.phase_index, prim2.expert.phase_step) == (4, 16)
    prim3 = pp.PlacePrimitive(fk, [0.34, -0.18])
    prim3.retarget([0.49, -0.20], 505)
    assert (prim3.expert.phase_index, prim3.expert.phase_step) == (5, 0)
    prim3.expert.phase_step = 7
    prim3.retarget([0.50, -0.21], 512)  # a re-target keeps the running clock
    assert (prim3.expert.phase_index, prim3.expert.phase_step) == (5, 7)
    assert np.allclose(prim3.target, [0.50, -0.21])
    assert not pp.reachable(fk, [0.34, -0.18], [0.49, -0.80])
    with pytest.raises(ContractError):
        pp.PlacePrimitive(fk, [0.34, -0.18]).retarget([0.49, -0.2], 300)


# ----- behavioural tests added after the re-review of #113 --------------------------------------
def test_the_runtime_turns_a_blocked_move_into_a_counted_failure(monkeypatch):
    """The SHIFT_BLOCKED except path of run_attempt_task, with the simulator stubbed out."""
    from embodied_jepa import lewm_planner_v2_runtime as lrt
    from embodied_jepa import plate_shift as ps

    class Robot:
        def stop(self, reason):
            self.stopped = reason

    class Counter:
        def remove(self):
            pass

    class Hook:
        def __init__(self, *args):
            self.removed = False

        def remove(self):
            self.removed = True

    def blocked_run(*args, **kwargs):
        raise ps.GuardError(lp.SHIFT_BLOCKED_PREFIX + " ['apple_geom']")

    def other_run(*args, **kwargs):
        raise ps.GuardError("the plate is not at its reset height")

    truth = {"object_position": [0.34, -0.18, 0.8], "plate_position": [0.49, -0.09, 0.75]}
    facts = {"post_look_frame": np.zeros((2, 2, 3), np.uint8), "post_look_state": np.zeros(3)}
    monkeypatch.setitem(lrt.rtm._W, "robot", Robot())
    monkeypatch.setitem(lrt.rtm._W, "bounds", (np.zeros(14), np.ones(14)))
    monkeypatch.setattr(lrt.rt2, "reset_and_look", lambda *a: (truth, None, Counter(), None, facts))
    monkeypatch.setattr(lrt.rtm, "check_post_look_frame", lambda *a: {})
    monkeypatch.setattr(lrt.ps, "PlateShift", Hook)
    monkeypatch.setattr(lrt, "_controller", lambda task, truth: object())
    task = {
        "seed": 54650,
        "arm": "L-plan",
        "reset": {"object_xy": [0.34, -0.18], "plate_xy": [0.49, -0.09]},
        "shift": {"step": 300, "vector": [0.0, -0.09]},
    }
    monkeypatch.setattr(lrt.rt2, "run_attempt", blocked_run)
    out = lrt.run_attempt_task(task)
    assert out["termination_reason"] == "shift_blocked" and out["success"] is False
    assert out["privileged_ok"] is True and out["shift_blocked"].startswith(lp.SHIFT_BLOCKED_PREFIX)
    assert out["executed_steps"] == 300 and out["post_look_state"] == [0.0, 0.0, 0.0]
    monkeypatch.setattr(lrt.rt2, "run_attempt", other_run)  # any other guard stays a V
    with pytest.raises(ps.GuardError, match="reset height"):
        lrt.run_attempt_task(task)


def test_p_far_writes_its_first_outcome_before_its_first_fit():
    source = (ROOT / "scripts" / "run_lewm_planner_v2.py").read_text()
    start = source.index("def stage_pfar(")
    body = source[start : source.index("\ndef ", start + 10)]
    assert body.index('report["first_outcome_utc"]') < body.index('current = fit("P-far-0")')
    assert body.index('current = fit("P-far-0")') < body.index("mark_first_render(report)")


def _runner(name):
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    return _load_script(name, "scripts/run_lewm_planner_v2.py")


def test_evidence_guards_refuse_mismatched_corpora_reports_and_artefacts(tmp_path):
    runner = _runner("_run_lewm_planner_v2_guards")

    class Reader:
        def __init__(self, manifest, corpus):
            self.manifest, self.corpus = manifest, corpus

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "manifest.json").write_text("{}")
    plan = lp.corpus_plan()
    good = {
        "protocol": lp.PROTOCOL,
        "plan": plan,
        "provenance": {"smoke": False, "k1_report_sha256": "k"},
    }
    # check_corpus_kind
    runner.check_corpus_kind(Reader(good, corpus), False)
    with pytest.raises(lp.GuardError, match="smoke flag"):
        runner.check_corpus_kind(Reader(good, corpus), True)
    with pytest.raises(lp.GuardError, match="not TASK-074"):
        runner.check_corpus_kind(Reader(good | {"protocol": "x"}, corpus), False)
    # the plan digest and the K1 link
    report: dict = {}
    runner.check_corpus_provenance(report, Reader(good, corpus), MANIFEST, False, "k")
    assert report["upstream"]["corpus"]["k1_report_sha256"] == "k"
    bad_plan = good | {"plan": plan[:-1]}
    with pytest.raises(lp.GuardError, match="plan differs"):
        runner.check_corpus_provenance({}, Reader(bad_plan, corpus), MANIFEST, False, "k")
    with pytest.raises(lp.GuardError, match="after this K1"):
        runner.check_corpus_provenance({}, Reader(good, corpus), MANIFEST, False, "other")
    # the corpus stage's own plan-digest check uses the same digest
    assert lp.plan_digest(plan) == MANIFEST["corpus_plan_sha256"]
    # verify_artifacts
    f = tmp_path / "a.npz"
    f.write_bytes(b"abc")
    runner.verify_artifacts({str(f): hashlib.sha256(b"abc").hexdigest()}, "t")
    with pytest.raises(lp.GuardError, match="differs from its sha256"):
        runner.verify_artifacts({str(f): "0" * 64}, "t")
    # read_stage_report: frozen block, clean tree, outcome, kind; records what was consumed
    up = {
        "protocol": lp.PROTOCOL,
        "outcome": "K1-PASS",
        "smoke": True,
        "frozen_sha256": lp.frozen_sha256(),
        "tracked_tree_dirty": False,
        "revision": "abc",
    }

    def write(d):
        p = tmp_path / "r.json"
        p.write_text(json.dumps(d))
        return p, hashlib.sha256(p.read_bytes()).hexdigest()

    p, sha = write(up)
    into: dict = {}
    runner.read_stage_report(p, sha, "K1-PASS", True, into)
    assert into["upstream"]["K1-PASS"]["revision"] == "abc"
    for bad, match in (
        (up | {"frozen_sha256": "0" * 64}, "another frozen block"),
        (up | {"tracked_tree_dirty": True}, "clean tree"),
        (up | {"outcome": "L-NO-CONDITION"}, "recorded K1-PASS"),
        (up | {"smoke": False}, "recorded K1-PASS"),
    ):
        p, sha = write(bad)
        with pytest.raises(lp.GuardError, match=match):
            runner.read_stage_report(p, sha, "K1-PASS", True)
    with pytest.raises(lp.GuardError, match="differs from its sha256"):
        runner.read_stage_report(p, "0" * 64, "K1-PASS", True)


def test_rerun_verdict_on_every_toleranced_field():
    from embodied_jepa import place_planner as pp

    cases = {
        "first_grasp_step": (_attempt(first_grasp_step=290), _attempt(first_grasp_step=310)),
        "first_place_step": (_attempt(first_place_step=640), _attempt(first_place_step=660)),
        "target_cm_max": (
            _attempt(decisions=[{"chosen": [24, 12], "target": [0.496, -0.2]}]),
            _attempt(decisions=[{"chosen": [24, 12], "target": [0.511, -0.2]}]),
        ),
    }
    for field, (above_half, void) in cases.items():
        out = pp.rerun_verdict(_attempt(), above_half)
        assert not out["void"], field
        assert field in [a["field"] for a in out["above_half_tolerance"]], field
        assert pp.rerun_verdict(_attempt(), void)["void"], field
    cmd = np.zeros((725, 7), np.float32)
    cmd[10, 0] = 0.5
    out = pp.rerun_verdict(_attempt(), _attempt(commands=cmd))
    assert not out["void"] and out["above_half_tolerance"][0]["field"] == (
        "max_abs_command_difference"
    )
    cmd[10, 0] = 0.9
    assert pp.rerun_verdict(_attempt(), _attempt(commands=cmd))["void"]
    assert pp.rerun_verdict(_attempt(), _attempt(executed_steps=700))["void"]  # exact field


# ----- addendum A1: the train stage's memory (the run-1 V) ---------------------------------------
class _FakeReader:
    """Episodes shaped like the corpus's (frames, applied, apple, plate), with ids per split."""

    def __init__(self, splits: dict, length: int = 600):
        self.manifest = {"splits": splits}
        self.length, self.reads = length, []

    def episode(self, episode_id, keys):
        self.reads.append(episode_id)
        rng = np.random.default_rng(abs(hash(episode_id)) % 2**32)
        n = self.length
        arrays = {
            "frames": rng.integers(0, 255, (n, 112, 112, 3), dtype=np.uint8),
            "applied": rng.normal(0, 0.1, (n - 1, 14)).astype(np.float32),
            "apple": rng.normal(0, 0.1, (n, 3)).astype(np.float32),
            "plate": rng.normal(0, 0.1, (n, 3)).astype(np.float32),
        }
        return arrays, {"shift": {"step": lp.SHIFT_STEP}, "truth_xy": [0.0, 0.0, 0.3, -0.2]}


class _Clock:
    def check(self, _what):
        return None


def test_featurisation_keeps_no_view_into_a_decoded_episode(monkeypatch):
    """The run-1 V (addendum A1): ``frames[t]`` kept each far episode's whole decoded frame array
    alive (270 x about 27 MiB). Every kept frame and plate row must own its bytes, with the same
    values, and the table must match what the stage computed before the fix."""
    runner = _runner("_run_lewm_planner_v2_a1")
    from embodied_jepa import lewm_planner_v2_offline as off

    monkeypatch.setattr(
        off,
        "pooled_features",
        lambda _enc, frames, device="cpu": frames.reshape(len(frames), -1)[:, :6144].astype(
            np.float32
        ),
    )
    far = _FakeReader({"train": ["a", "b"], "val": ["c"]})
    look = _FakeReader({"train": ["l1"], "val": ["l2"]})
    sources = [("far", far, s) for s in lp.READ_SPLITS] + [
        ("look", look, s) for s in ("train", "val")
    ]
    table, plate_rows, anchor_frames, anchor_rows = runner.featurise_sources(
        sources, None, _Clock()
    )
    assert len(table.roots) == 5 and far.reads == ["a", "b", "c"]
    kept = [r[1] for rows in plate_rows.values() for r in rows]
    kept += [r[2] for rows in plate_rows.values() for r in rows]
    assert len(plate_rows["train"]) == 2 * len(lp.DECISION_STEPS)
    for array in [*kept, *anchor_frames]:
        assert array.base is None and array.flags.owndata
    # the same bytes as the decoded episode (re-decoded: the fake is deterministic per id)
    arrays, _ = far.episode("a", None)
    for (eid, frame, plate), t in zip(plate_rows["train"][:6], lp.DECISION_STEPS, strict=True):
        assert eid == "a"
        assert np.array_equal(frame, arrays["frames"][t])
        assert np.array_equal(plate, arrays["plate"][t, :2])
    first = lp.FEATURE_BAND[0]
    assert np.array_equal(np.stack(anchor_frames[:16]), arrays["frames"][first : first + 16])
    assert anchor_rows[:16] == list(range(16))


def test_the_train_stage_and_its_scale_probe_share_the_featurisation():
    source = (ROOT / "scripts" / "run_lewm_planner_v2.py").read_text()
    for name in ("def stage_train(", "def train_scale_probe("):
        start = source.index(name)
        body = source[start : source.index("\n\n\n", start)]
        assert "featurise_sources(sources, encoder, clock)" in body, name
        assert "rd.episode(" not in body and "reader.episode(" not in body, name
    start = source.index("def train_scale_probe(")
    body = source[start : source.index("\n\n\n", start)]
    assert body.index("off.train_model(") < body.index("offline_gates(")
    assert "G-memory-margin" in body


def test_the_scale_probe_decodes_every_slot_at_the_real_split_sizes():
    runner = _runner("_run_lewm_planner_v2_a1b")
    assert runner.TRAIN_SCALE_ROOTS == {"train": 240, "val": 30}
    assert runner.TRAIN_SCALE_ROOTS == {s: lp.CORPUS_SPLITS[s] for s in lp.READ_SPLITS}
    assert runner.TRAIN_SCALE_MARGIN_GIB >= 2.0 and lp.MEMORY["ceiling_gib"] == 12.0
    smoke = _FakeReader({"train": ["s1", "s2"], "val": ["s3"], "test": ["x"]}, length=20)
    cyc = runner.CycledReader(smoke, runner.TRAIN_SCALE_ROOTS)
    assert [len(cyc.manifest["splits"][s]) for s in ("train", "val")] == [240, 30]
    for slot in cyc.manifest["splits"]["val"][:4]:
        cyc.episode(slot, None)
    assert smoke.reads == ["s1", "s2", "s3", "s1"]  # decoded afresh per slot; never the test split


# ----- addendum A2: the train stage's budget cap (the run-2 ESCALATE-BUDGET) ---------------------
def test_a2_train_cap_is_80000_and_u_is_80000_for_run_2s_saturation():
    assert lp.BUDGET["cap"] == 60_000  # the frozen block is not edited
    assert lp.A2_TRAIN_BUDGET["frozen_cap"] == 60_000
    assert lp.A2_TRAIN_BUDGET["cap"] == 80_000 and lp.A2_TRAIN_BUDGET["stage"] == "train"
    rule = lp.train_budget_updates([25_000, 40_000])  # run-2's W-7410 and N-7410
    assert rule["wanted"] == 80_000 and rule["updates"] == 80_000 and not rule["escalate"]
    assert (rule["cap"], rule["frozen_cap"], rule["addendum"]) == (80_000, 60_000, "A2")
    # the same saturation under the frozen cap is run-2's ESCALATE-BUDGET
    assert lp.budget_updates([25_000, 40_000]) == {
        "max_saturation_update": 40_000,
        "wanted": 80_000.0,
        "updates": 60_000,
        "escalate": True,
    }
    # the rest of the frozen formula is unchanged under A2
    assert lp.train_budget_updates([3000, 4000])["updates"] == 10_000
    assert lp.train_budget_updates([12_000])["updates"] == 25_000


def test_a2_escalation_still_fires_above_80000():
    assert lp.train_budget_updates([40_001])["escalate"]
    assert lp.train_budget_updates([40_001])["updates"] == 80_000
    assert lp.train_budget_updates([45_000])["escalate"]
    assert not lp.train_budget_updates([40_000])["escalate"]


def test_a2_last_two_raise_is_min_2u_80000():
    assert lp.train_last_two_raise(30_000, 0) == 60_000
    assert lp.train_last_two_raise(50_000, 0) == 80_000
    assert lp.train_last_two_raise(80_000, 0) is None  # a last-two selection at U = 80 000
    assert lp.train_last_two_raise(30_000, 1) is None  # only one raise


def test_a2_calibration_must_reproduce_run_2():
    ok = lp.a2_calibration_check({"W-7410": 25_000, "N-7410": 40_000})
    assert ok["reproduces_run_2"] and ok["addendum"] == "A2"
    assert not lp.a2_calibration_check({"W-7410": 25_000, "N-7410": 41_000})["reproduces_run_2"]
    assert not lp.a2_calibration_check({"W-7410": 25_000})["reproduces_run_2"]


def test_a2_keeps_the_frozen_block_so_k1_and_corpus_reports_are_accepted(tmp_path):
    """A2 is outside the frozen block: the frozen sha is the literal, so the K1-PASS and
    CORPUS-SEALED reports recorded under it are still accepted by ``read_stage_report``."""
    assert lp.frozen_sha256() == FROZEN_SHA256
    assert "A2_TRAIN_BUDGET" not in json.dumps(lp.frozen_block())
    assert lp.frozen_block()["budget"]["cap"] == 60_000
    a2 = MANIFEST["addendum_a2"]
    assert a2["frozen_sha256_unchanged"] == FROZEN_SHA256
    assert a2["cap"] == {"frozen": 60_000, "train_stage": 80_000}
    runner = _runner("_run_lewm_planner_v2_a2")
    for outcome in ("K1-PASS", "CORPUS-SEALED"):
        recorded = a2["upstream_reports"][outcome]
        assert recorded["frozen_sha256"] == FROZEN_SHA256
        report = {
            "protocol": lp.PROTOCOL,
            "outcome": outcome,
            "smoke": False,
            "frozen_sha256": recorded["frozen_sha256"],
            "tracked_tree_dirty": False,
            "revision": recorded["revision"],
        }
        p = tmp_path / f"{outcome}.json"
        p.write_text(json.dumps(report))
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        assert runner.read_stage_report(p, sha, outcome, False)["outcome"] == outcome


def test_the_train_stage_applies_a2():
    source = (ROOT / "scripts" / "run_lewm_planner_v2.py").read_text()
    start = source.index("def stage_train(")
    body = source[start : source.index("\n\n\n", start)]
    assert 'report["addendum_a2"]' in body
    assert '{"cap": lp.A2_TRAIN_BUDGET["cap"]}' in body
    assert "lp.train_budget_updates(" in body and "lp.budget_updates(" not in body
    assert "lp.a2_calibration_check(" in body
    assert "lp.train_last_two_raise(updates, attempt)" in body
    assert body.index("reproduces_run_2") < body.index('budget["escalate"]')
