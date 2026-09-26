"""TASK-064 look-prefix corpus: the preregistered plan, look, checks, gate rule and rows.

Synthetic/unit checks of ``embodied_jepa.look_corpus`` only; they are never evidence of
manipulation or of readability. Learned Apple->Plate is still 0 successes.
"""

import importlib.util
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import info_ceiling as ic
from embodied_jepa import look_corpus as lc
from embodied_jepa import observation_reprobe as orp
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-look-corpus-v1.json"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wide = load("collect_apple_wide")


def manifest():
    return json.loads(MANIFEST.read_text())


# ----- reuse: the TASK-047/048 settings are unchanged ------------------------------------------
def test_reset_rule_matches_task047_wide_reset():
    evaluator = load("evaluate_apple")
    for seed in (47000, 47199, 47900, 47931):
        expected = evaluator.wide_reset(seed)
        got = lc.wide_reset(seed)
        assert got["object_xy"] == expected["object_xy"]
        assert got["plate_xy"] == expected["plate_xy"]


def test_collector_settings_equal_task048():
    assert lc.RESET_CENTERS == wide.RESET_CENTERS
    assert lc.WIDE_JITTER_M == wide.WIDE_JITTER_M
    assert lc.IMAGE_SIZE == wide.IMAGE_SIZE and lc.FPS == wide.FPS
    assert lc.NOISE_LEVELS == wide.NOISE_LEVELS and lc.OU_THETA == wide.OU_THETA
    assert lc.OU_SIGMA == wide.OU_SIGMA
    assert lc.BURST_PROBABILITY == wide.BURST_PROBABILITY
    assert lc.BURST_COMMANDS == wide.BURST_COMMANDS
    assert lc.AIM_OFFSET_EVERY == wide.AIM_OFFSET_EVERY and lc.AIM_OFFSET_M == wide.AIM_OFFSET_M
    assert lc.BRANCH_KINDS == wide.BRANCH_KINDS
    assert lc.BRANCHES_PER_ROOT == wide.BRANCHES_PER_ROOT
    assert (lc.BRANCH_PHASE, lc.BRANCH_END_PHASE) == (wide.BRANCH_PHASE, wide.BRANCH_END_PHASE)
    assert lc.POLICY_MAX_COMMANDS == wide.ROOT_MAX_COMMANDS
    assert np.array_equal(lc.LOWER, wide.LOWER) and np.array_equal(lc.UPPER, wide.UPPER)
    assert np.array_equal(lc.LOWER, orp.COLLECTION_LOWER)
    assert np.array_equal(lc.UPPER, orp.COLLECTION_UPPER)
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    for kind in lc.BRANCH_KINDS:
        assert lc.branch_params(kind, rng_a) == wide.branch_params(kind, rng_b)


def test_the_look_is_task061s_constant():
    assert lc.LOOK_STEPS == 8 and lc.DECISION_FRAME == 8
    assert orp.sequence_sha256(orp.look_sequence()) == lc.LOOK_SEQUENCE_SHA256
    assert lc.LOOK_SEQUENCE_SHA256 == manifest()["look_prefix"]["sequence_sha256_float32_bytes"]


def test_thresholds_are_task048s_frozen_values():
    t = lc.THRESHOLDS
    assert t["A1_root_episodes_min"] == 190 and t["A2_branch_episodes_min"] == 450
    assert t["A3_root_full_success_min"] == 80 and t["A5_grasp_phase_successes_min"] == 250
    assert t["A4_grasp_phase_failure_fraction"] == (0.20, 0.80)
    assert t["A6_right_arm_sign_fraction_min"] == 0.03
    assert t["A7_off_script_fraction_min"] == 0.50
    assert t["A8_branch_pairs_distinct_fraction_min"] == 0.90


# ----- seeds, plan, splits ----------------------------------------------------------------------
def test_seed_ranges_are_new_and_reserved_ranges_are_refused():
    seeds = set(lc.FROZEN_SEEDS)
    assert len(seeds) == 200 and not seeds & set(lc.PILOT_SEEDS)
    lc.check_seeds(lc.FROZEN_SEEDS)
    lc.check_seeds(lc.PILOT_SEEDS)
    for reserved in (
        48000,
        48199,
        48931,
        45000,
        45107,
        45200,
        45300,
        45339,
        44000,
        43000,
        42000,
        41000,
        20000,
        49100,
    ):
        with pytest.raises(ContractError, match="reserved"):
            lc.check_seeds([reserved])
        with pytest.raises(ContractError, match="reserved"):
            lc.make_plan([47000, reserved])
    assert not seeds & (ic.COHORT_C | ic.COHORT_D)
    assert not seeds & set(range(ic.SEED_RANGE[0], ic.SEED_RANGE[1] + 1))


def test_plan_structure_and_splits():
    plan = lc.make_plan(lc.FROZEN_SEEDS)
    assert plan == lc.make_plan(lc.FROZEN_SEEDS)
    assert Counter(r["split"] for r in plan["roots"]) == {"train": 170, "val": 20, "test": 10}
    ids = [r["episode_id"] for r in plan["roots"]]
    ids += [b["episode_id"] for r in plan["roots"] for b in r["branches"]]
    assert len(ids) == len(set(ids)) == 800
    kinds = Counter(b["kind"] for r in plan["roots"] for b in r["branches"])
    assert kinds == {kind: 120 for kind in lc.BRANCH_KINDS}
    assert Counter(r["noise_level"] for r in plan["roots"]) == {0: 50, 1: 50, 2: 50, 3: 50}
    aimed = [r for r in plan["roots"] if r["aim_offset_xy_m"] is not None]
    assert len(aimed) == 40
    # TASK-048 review R1 B2: branch kinds are decoupled from aim-offset roots.
    assert Counter(b["kind"] for r in aimed for b in r["branches"]) == {
        kind: 24 for kind in lc.BRANCH_KINDS
    }
    for root in plan["roots"]:
        assert root["session_id"] == f"look-reset-{root['seed']}"
        assert root["episode_id"] == f"look-{root['seed']}"
        if root["aim_offset_xy_m"] is not None:
            assert 0.015 <= np.hypot(*root["aim_offset_xy_m"]) <= 0.03
    assert plan["look"]["perturbed"] is False and plan["cameras"] == ["onboard_rgb"]


def test_splits_and_plan_hashes_are_pinned():
    plan = lc.make_plan(lc.FROZEN_SEEDS)
    pinned = manifest()["splits"]
    by_split = {
        k: sorted(r["seed"] for r in plan["roots"] if r["split"] == k) for k in ("val", "test")
    }
    assert by_split["val"] == pinned["val_sessions"] and len(by_split["val"]) == 20
    assert by_split["test"] == pinned["test_sessions"] and len(by_split["test"]) == 10
    plan_pins = manifest()["plan"]
    assert lc.plan_sha256_rounded(plan) == plan_pins["sha256_rounded_1e-9"]
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        assert lc.plan_sha256(plan) == plan_pins["sha256_macos_arm64"]


def test_read_roots_are_train_and_val_only_and_fold_hash_is_pinned():
    plan = lc.make_plan(lc.FROZEN_SEEDS)
    roots = lc.read_roots(plan)
    assert Counter(r["split"] for r in roots) == {"train": 170, "val": 20}
    assert [r["seed"] for r in roots] == sorted(r["seed"] for r in roots)
    seeds = [r["seed"] for r in roots]
    got = ic.fold_assignment_sha256(seeds, ic.fold_of(len(roots)))
    assert got == manifest()["readability_gate"]["fold_assignment_sha256"]


def test_check_read_split_both_directions():
    plan = lc.make_plan(lc.FROZEN_SEEDS)
    roots = lc.read_roots(plan)
    splits = {r["episode_id"]: r["split"] for r in plan["roots"]}
    lc.check_read_split(roots, splits)
    test_root = next(r for r in plan["roots"] if r["split"] == "test")
    with pytest.raises(lc.GuardError, match="not train or val"):
        lc.check_read_split(roots + [test_root], splits)
    moved = dict(splits)
    moved[roots[0]["episode_id"]] = "val"
    with pytest.raises(lc.GuardError, match="planned split"):
        lc.check_read_split(roots, moved)
    with pytest.raises(lc.GuardError, match="read roots"):
        lc.check_read_split(roots[1:], splits)
    # A planned train/val root that errored or never started is absent: the run is V.
    absent = {k: v for k, v in splits.items() if k != roots[5]["episode_id"]}
    with pytest.raises(lc.GuardError, match="not stored"):
        lc.check_read_split(roots, absent)


# ----- L1: the look records ----------------------------------------------------------------------
def look_record(seed, **overrides):
    record = lc.look_facts(
        applied=orp.look_sequence(),
        post_state=np.zeros(86),
        apple_xy_moves=[1e-15] * 8,
        plate_moves=[0.0] * 8,
        contacts=[False] * 8,
    )
    record["seed"] = seed
    record.update(overrides)
    return record


def test_look_records_pass_and_every_failure_is_caught():
    good = [look_record(s) for s in range(4)]
    seeds = range(4)
    assert lc.check_look_records(good, seeds)["ok"]
    assert not lc.check_look_records(good[:3], seeds)["ok"]  # a planned root without a record
    assert not lc.check_look_records([], seeds)["ok"]
    assert not lc.check_look_records(good[:3] + [look_record(9)], seeds)["ok"]  # unplanned seed
    assert not lc.check_look_records(good[:3] + [look_record(2)], seeds)["ok"]  # duplicate
    bad_cases = [
        {"applied_max_abs_minus_requested": 1e-7},
        {"post_look_state": [1e-5] + [0.0] * 85},
        {"apple_xy_move_max_m": 2e-6},
        {"plate_move_max_m": 2e-6},
        {"hand_contact": True},
        {"look_steps_checked": 7},
        {"look_commands_executed": 7},
    ]
    for override in bad_cases:
        records = good[:3] + [look_record(3, **override)]
        assert not lc.check_look_records(records, seeds)["ok"], override


def test_look_facts_measures_a_perturbed_look():
    applied = orp.look_sequence().astype(np.float64).copy()
    applied[3, 7] += 0.01
    facts = lc.look_facts(applied, np.zeros(3), [0.0] * 8, [0.0] * 8, [False] * 8)
    assert facts["applied_max_abs_minus_requested"] == pytest.approx(0.01)
    short = lc.look_facts(applied[:5], np.zeros(3), [0.0] * 5, [0.0] * 5, [False] * 5)
    assert short["applied_max_abs_minus_requested"] is None
    assert short["look_commands_executed"] == 5


# ----- acceptance checks ----------------------------------------------------------------------
def good_result():
    return {
        "root_episodes": 200,
        "branch_episodes": 590,
        "root_full_success": 120,
        "grasp_phase_failure_fraction": 0.5,
        "grasp_phase_successes": 390,
        "action_coverage": {
            "right_arm_positive_fraction_above_0_1": [0.06] * 6,
            "right_arm_negative_fraction_below_minus_0_1": [0.06] * 6,
            "off_script_fraction": 0.87,
        },
        "branch_pair_rgb_distinct_at_16": {"pairs": 500, "distinct": 500},
        "sessions_spanning_splits": 0,
        "sessions_off_plan": 0,
        "normalization_is_train_only": True,
        "split_sessions": {"train": 170, "val": 20, "test": 10},
        "audit": {"episodes_decoded": 700, "episodes_expected": 700, "all_intervals_ok": True},
        "label_errors": [],
        "privileged_labels_gated": True,
        "camera_shapes": {"onboard_rgb": [112, 112, 3]},
        "integrity": {"ok": True},
        "look": {"ok": True},
    }


FAILURES = {
    "A1_root_episodes": {"root_episodes": 189},
    "A2_branch_episodes": {"branch_episodes": 449},
    "A3_root_full_success": {"root_full_success": 79},
    "A4_grasp_phase_failure_fraction": {"grasp_phase_failure_fraction": 0.81},
    "A5_grasp_phase_successes": {"grasp_phase_successes": 249},
    "A6_right_arm_both_signs": {
        "action_coverage": {
            "right_arm_positive_fraction_above_0_1": [0.06] * 5 + [0.02],
            "right_arm_negative_fraction_below_minus_0_1": [0.06] * 6,
            "off_script_fraction": 0.87,
        }
    },
    "A7_off_script_fraction": {
        "action_coverage": {
            "right_arm_positive_fraction_above_0_1": [0.06] * 6,
            "right_arm_negative_fraction_below_minus_0_1": [0.06] * 6,
            "off_script_fraction": 0.49,
        }
    },
    "A8_branch_pairs_distinct": {"branch_pair_rgb_distinct_at_16": {"pairs": 10, "distinct": 8}},
    "A9_no_split_leakage": {"sessions_spanning_splits": 1},
    "A10_audit": {
        "audit": {"episodes_decoded": 699, "episodes_expected": 700, "all_intervals_ok": True}
    },
    "A11_labels_separated": {"privileged_labels_gated": False},
    "A12_resolution": {
        "camera_shapes": {"onboard_rgb": [112, 112, 3], "hand_crop_rgb": [112, 112, 3]}
    },
    "A13_run_integrity": {"integrity": {"ok": False}},
    "L1_look_prefix": {"look": {"ok": False}},
}


def test_every_acceptance_check_passes_and_fails():
    verdict = lc.checks(good_result())
    assert verdict["all_passed"] and set(verdict["checks"]) == set(FAILURES)
    for name, change in FAILURES.items():
        verdict = lc.checks(good_result() | change)
        assert not verdict["checks"][name], name
        assert not verdict["all_passed"]
        assert all(v for k, v in verdict["checks"].items() if k != name), name
    empty = good_result() | {"action_coverage": {}, "branch_pair_rgb_distinct_at_16": {}}
    assert not lc.checks(empty)["all_passed"]


def test_frame_interval_rule():
    assert lc.frame_interval_ok([0.05, 0.049999999999972, 0.050000000000061])
    assert not lc.frame_interval_ok([0.05, 0.1])
    assert not lc.frame_interval_ok([])


# ----- readability rule, guards, rows ------------------------------------------------------------
def test_readability_rule_needs_every_condition():
    ok = {"succeeds": True, "beats_floor": True, "p_arm": 0.001, "spurious": False, "finite": True}
    assert lc.readability_passes(**ok)
    assert lc.readability_passes(**(ok | {"p_arm": 0.025}))
    for change in (
        {"succeeds": False},
        {"beats_floor": False},
        {"p_arm": 0.0251},
        {"spurious": True},
        {"finite": False},
    ):
        assert not lc.readability_passes(**(ok | change)), change


def test_render_and_expert_guards_both_directions():
    lc.check_render([True] * 3, "frame")
    with pytest.raises(lc.GuardError, match="Q-render"):
        lc.check_render([True, False, True], "frame")
    with pytest.raises(lc.GuardError, match="Q-render"):
        lc.check_render([], "frame")
    expert = np.zeros((4, 7))
    recorded = expert.copy()
    recorded[3, 0] = 0.5  # an aim-offset root may differ
    aim = [False, False, False, True]
    assert lc.check_expert(expert, recorded, aim)["roots_compared"] == 3
    recorded[0, 0] = 1e-5
    with pytest.raises(lc.GuardError, match="Q-expert"):
        lc.check_expert(expert, recorded, aim)
    with pytest.raises(lc.GuardError, match="no root"):
        lc.check_expert(expert, recorded, [True] * 4)
    assert lc.check_apple_label([[0.3, -0.2]], [[0.3 + 1e-8, -0.2]]) <= 1e-6
    with pytest.raises(lc.GuardError, match="apple label"):
        lc.check_apple_label([[0.3, -0.2]], [[0.3 + 1e-5, -0.2]])


def test_outcome_rows_first_match():
    def decide(data, read, look=True):
        return lc.decide(void=False, data_all_passed=data, readability_passed=read, look_ok=look)

    void = lc.decide(void=True, data_all_passed=True, readability_passed=True, look_ok=True)
    assert void == {"outcome": "V"}
    accept = decide(True, True)
    assert accept["outcome"] == "C-ACCEPT" and accept["corpus_accepted"]
    assert not accept["abandonment_clause_fires"] and accept["also_matching_rows"] == []
    read = decide(True, False)
    assert read["outcome"] == "C-READ-FAIL" and read["abandonment_clause_fires"]
    assert not read["corpus_accepted"] and read["also_matching_rows"] == []
    both = decide(False, False)
    assert both["outcome"] == "C-READ-FAIL" and both["also_matching_rows"] == ["C-DATA-FAIL"]
    data = decide(False, True)
    assert data["outcome"] == "C-DATA-FAIL" and not data["abandonment_clause_fires"]
    # A failed look confounds a readability failure: C-DATA-FAIL, and the clause does not fire.
    confounded = decide(False, False, look=False)
    assert confounded["outcome"] == "C-DATA-FAIL" and not confounded["abandonment_clause_fires"]
    assert confounded["readability_failure_confounded_by_look"]
    with pytest.raises(ContractError, match="inconsistent"):
        decide(True, True, look=False)
    assert set(lc.ROWS) == {"V", "C-ACCEPT", "C-READ-FAIL", "C-DATA-FAIL"}


def test_manifest_agrees_with_the_module():
    m = manifest()
    assert m["task"] == lc.TASK and m["protocol"] == lc.PROTOCOL
    assert m["seeds"]["corpus"] == [lc.FROZEN_SEEDS[0], lc.FROZEN_SEEDS[-1]]
    assert m["seeds"]["pilot"] == [lc.PILOT_SEEDS[0], lc.PILOT_SEEDS[-1]]
    assert m["learned_apple_to_plate_successes"] == 0 and m["exemption_spent"] is False
    assert [row["row"] for row in m["pre_declared_outcomes_in_order"]] == list(lc.ROWS)
    assert m["acceptance_checks"]["thresholds"]["A3_root_full_success_min"] == 80
    budget = m["budget"]
    assert budget["global_wall_seconds"] == lc.GLOBAL_WALL_SECONDS
    assert budget["supervisor_seconds"] == lc.SUPERVISOR_SECONDS
    assert budget["worker_seconds"] == lc.WORKER_SECONDS and budget["workers"] == lc.WORKERS
    assert budget["device"] == lc.DEVICE
