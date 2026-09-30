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
FROZEN_SHA256 = "e647b2e9fbdbeda9e8c1ef981a9f5aff5044553beeb3fb58cba1b3dc515c7b71"


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
    digest = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == MANIFEST["corpus_plan_sha256"]


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


def test_gated_rows_first_match():
    base = {
        "B-oracle-shift": 64,
        "H-handover": 64,
        "P-truth": 20,
        "P-stale": 0,
        "L-plan": 60,
        "L-plan-s1": 58,
        "L-plan-s2": 59,
        "H-twin": 62,
    }
    assert lp.decide_gated(_s(**base), _u(), HARNESS)["row"] == "L-PASS"
    assert (
        lp.decide_gated(_s(**base | {"H-twin": 64, "L-plan": 57}), _u(), HARNESS)["row"]
        == "L-PASS-TWIN-BETTER"
    )
    assert lp.decide_gated(_s(**base), _u(**{"L-plan": 27}), HARNESS)["row"] == "L-HARM"
    slow = HARNESS | {"median_decision_seconds": 1.2}
    assert lp.decide_gated(_s(**base), _u(), slow)["row"] == "L-SLOW"
    assert lp.decide_gated(_s(**base | {"B-oracle-shift": 59}), _u(), HARNESS)["row"] == "VOID"
    assert lp.decide_gated(_s(**base | {"P-stale": 9}), _u(), HARNESS)["row"] == (
        "S-VOID-CONDITION"
    )
    assert lp.decide_gated(_s(**base | {"H-handover": 35}), _u(), HARNESS)["row"] == (
        "L-NO-HEADROOM"
    )
    blind = base | {"L-shuf": 58}
    assert lp.decide_gated(_s(**blind), _u(), HARNESS)["row"] == "L-SCENE-BLIND"
    assert lp.decide_gated(_s(**base | {"L-plan": 25}), _u(), HARNESS)["row"] == "L-NO-GAIN"
    bad = HARNESS | {"determinism_ok": False}
    assert lp.decide_gated(_s(**base), _u(), bad)["row"] == "VOID"


def test_every_row_has_a_consequence_and_the_clause_rows():
    assert set(lp.ALL_ROWS) <= set(lp.ROW_CONSEQUENCES)
    assert set(lp.CLAUSE_ROWS) == {"L-G2A", "L-NO-RANK", "L-HARM", "L-NO-GAIN"}
    for row in lp.CLAUSE_ROWS:
        assert lp.ROW_CONSEQUENCES[row] == "clause" and lp.abandonment_fires(row)
    assert not lp.abandonment_fires("L-NO-HEADROOM")
    with pytest.raises(ContractError):
        lp.abandonment_fires("HYB-PASS")


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


def test_gated_stage_is_refused_without_an_authorisation_record(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    runner = _load_script("_run_lewm_planner_v2", "scripts/run_lewm_planner_v2.py")
    with pytest.raises(lp.GuardError, match="G-authorisation"):
        runner.stage_gated({}, {"gated_authorization": None}, tmp_path, None, None)
    with pytest.raises(lp.GuardError, match="G-authorisation"):
        runner.stage_gated({}, {"gated_authorization": {"x": 1}}, tmp_path, None, None)


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
