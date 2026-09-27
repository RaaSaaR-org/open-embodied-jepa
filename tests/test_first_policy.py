"""TASK-067 preregistration: the frozen design module and its manifest (NumPy only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import first_policy as fp
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-first-policy-v1.json"


def test_manifest_frozen_block_equals_the_module():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["frozen"] == fp.frozen_block()
    assert manifest["results"] is None and manifest["calibration"] is None
    assert manifest["exemption_spent_cited"]["value_at_preregistration"] is False
    for key in ("R1", "R2", "R3", "R4", "R5"):
        assert manifest["owner_rulings"][key].startswith(key)
    assert manifest["owner_rulings"]["received_utc"] == fp.OWNER_RULINGS_UTC


def test_seed_ranges_are_disjoint_and_off_every_forbidden_range():
    fp.check_seed_ranges()
    seeds = [s for name in fp.SEED_RANGES for s in fp.seeds_of(name)]
    assert len(seeds) == len(set(seeds)) == 800
    assert not set(seeds) & set(fp.COHORT_C) and not set(seeds) & set(fp.COHORT_D)
    assert len(fp.seeds_of("perception_heldout")) == 128
    assert len(fp.seeds_of("calibration_C0")) == fp.C0_RESETS
    for k in (1, 2, 3):
        assert len(fp.seeds_of(f"dagger_{k}")) == fp.DAGGER_RESETS_PER_ITERATION


def test_seed_check_refuses_leaving_the_range(monkeypatch):
    monkeypatch.setitem(fp.SEED_RANGES, "dagger_3", (46672, 47000))
    with pytest.raises(fp.GuardError, match="leaves"):
        fp.check_seed_ranges()


def test_seed_check_refuses_a_pairwise_overlap(monkeypatch):
    monkeypatch.setitem(fp.SEED_RANGES, "dagger_2", (46500, 46671))
    with pytest.raises(fp.GuardError, match="overlap"):
        fp.check_seed_ranges()


def test_schedule_matches_the_collector():
    assert fp.EXPERT_POLICY_STEPS == 745
    assert [fp.scheduled_phase(s) for s in (0, 129, 130, 209, 210, 254, 255)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
    ]
    assert fp.scheduled_phase(744) == 7 and fp.scheduled_phase(799) == 7
    with pytest.raises(ContractError):
        fp.scheduled_phase(-1)


def test_schedule_matches_scripted_budgets():
    from embodied_jepa.scripted import apple_collector_policy

    truth = {
        "position_frame": "world",
        "base_position_world": np.zeros(3),
        "base_rotation_world": np.eye(3),
        "object_position": np.array([0.34, -0.18, 0.77]),
        "plate_position": np.array([0.49, -0.09, 0.75]),
        "container_surface_z": 0.76,
        "object_support_height": 0.03,
    }
    policy = apple_collector_policy(truth)
    assert tuple(p.name for p in policy.phases) == fp.PHASES
    assert tuple(p.commands for p in policy.phases) == fp.PHASE_BUDGETS


def test_policy_input_width():
    assert fp.clock_features(0).shape == (fp.POLICY_INPUTS["clock"],)
    assert sum(fp.POLICY_INPUTS.values()) == 132
    assert "Linear(132, 512)" in fp.POLICY_HEAD["layers"]


def _levels(levels, value):
    return {level: value for level in levels}


def test_c0_bars_tighten_only():
    full = fp.c0_bars(32, _levels(fp.C0_APPLE_LEVELS_CM, 30), _levels(fp.C0_PLATE_LEVELS_CM, 30))
    assert full == {
        "apple_median_cm": 0.75,
        "apple_p90_cm": 1.2,
        "plate_median_cm": 1.5,
        "plate_p90_cm": 2.5,
    }
    apple = {0.5: 32, 0.8: 28, 1.0: 20, 1.2: 31}  # a pass after a failure does not count
    plate = {1.0: 30, 1.5: 27, 2.0: 32, 2.5: 32}
    bars = fp.c0_bars(30, apple, plate)
    assert bars["apple_p90_cm"] == 0.8 and bars["apple_median_cm"] == 0.75
    assert bars["plate_p90_cm"] == 1.0 and bars["plate_median_cm"] == 1.0


def test_c0_escalates():
    good_a, good_p = _levels(fp.C0_APPLE_LEVELS_CM, 32), _levels(fp.C0_PLATE_LEVELS_CM, 32)
    assert "escalate" in fp.c0_bars(27, good_a, good_p)
    assert "escalate" in fp.c0_bars(32, {**good_a, 0.5: 27}, good_p)
    assert "escalate" in fp.c0_bars(32, good_a, {**good_p, 1.0: 27})
    with pytest.raises(ContractError):
        fp.c0_bars(32, {0.5: 32}, good_p)


def test_s0_perception():
    bars = {
        "apple_median_cm": 0.75,
        "apple_p90_cm": 1.2,
        "plate_median_cm": 1.5,
        "plate_p90_cm": 2.5,
    }
    apple = np.full(128, 0.5)
    plate = np.full(128, 1.0)
    result = fp.s0_perception(apple, plate, bars)
    assert result["apple_passes"] and result["plate_passes"]
    tail = apple.copy()
    tail[:20] = 1.5  # p90 above the bar while the median passes
    assert not fp.s0_perception(tail, plate, bars)["apple_passes"]
    with pytest.raises(ContractError):
        fp.s0_perception(apple[:10], plate[:10], bars)
    bad = apple.copy()
    bad[0] = np.nan
    with pytest.raises(fp.GuardError):
        fp.s0_perception(bad, plate, bars)


def _counts(**overrides):
    counts = {arm: {"grasp": 0, "success": 0} for arm in fp.M1_ARMS}
    counts["B-oracle"] = {"grasp": 16, "success": 16}
    for arm, (grasp, success) in overrides.items():
        counts[arm.replace("_", "-")] = {"grasp": grasp, "success": success}
    return counts


A4T = 8


def _decide(counts, **kwargs):
    return fp.decide_m1(counts, A4T, **kwargs)


def test_m1_rows():
    assert _decide(_counts(B_oracle=(16, 13)))["row"] == "V"
    assert _decide(_counts(B_hold=(1, 0)))["row"] == "V"
    passed = _decide(_counts(P_1=(3, 2), P_3=(2, 2)))
    assert passed["row"] == "M1-PASS" and passed["carried"] == "P-3"
    assert not passed["clause_fires"]
    motor = _decide(_counts(A4_look=(12, 8)))
    assert motor["row"] == "M1-MOTOR" and motor["run_F"]
    partial = _decide(_counts(A4_look=(12, 8)), f_counts={"grasp": 2, "success": 1})
    assert partial["row"] == "M1-MOTOR-F-PARTIAL" and not partial["clause_fires"]
    none = _decide(_counts(A4_look=(12, 8)), f_counts={"grasp": 2, "success": 0})
    assert none["row"] == "M1-MOTOR-F-NONE" and none["clause_fires"]
    perception = _decide(_counts(A4_look=(9, 7)))
    assert perception["row"] == "M1-PERCEPTION" and perception["clause_fires"]


def test_m1_controls_never_decide_the_row():
    # C-3, R-3, D-oracle-perc and B-replay successes change no row: M1 is about P.
    counts = _counts(C_3=(5, 4), R_3=(5, 4), D_oracle_perc=(16, 16), B_replay=(6, 5))
    assert _decide(counts)["row"] == "M1-PERCEPTION"


def test_m1_refuses_missing_or_invalid_counts():
    counts = _counts()
    del counts["R-3"]
    with pytest.raises(ContractError):
        _decide(counts)
    with pytest.raises(ContractError):
        _decide(_counts(P_0=(1, 2)))


def test_learned_arms_are_enumerated_and_labelled():
    assert set(fp.LEARNED_ARMS) == {a for a, (rung, _) in fp.ARMS.items() if rung == "L1"}
    assert fp.ARMS["F"][0] == "L2" and fp.ARMS["A4-look"][0] == "L3"
    assert set(fp.ROWS) >= fp.CLAUSE_ROWS


def test_m1_refuses_an_uncalibrated_a4_threshold():
    with pytest.raises(ContractError):
        fp.decide_m1(_counts(), 0)


def test_a4_threshold_from_c0_and_s0():
    apple = {0.5: 32, 0.8: 30, 1.0: 28, 1.2: 28}
    plate = {1.0: 32, 1.5: 32, 2.0: 30, 2.5: 28}
    # every held-out reset inside the smallest levels: predicted rate 1.0 -> expected 16 -> 8
    assert fp.a4_threshold(32, apple, plate, np.full(128, 0.3), np.full(128, 0.5)) == 8
    # half the resets beyond the largest apple level: rate halves -> threshold 4
    errors = np.r_[np.full(64, 0.3), np.full(64, 2.0)]
    assert fp.a4_threshold(32, apple, plate, errors, np.full(128, 0.5)) == 4
    # never below 1
    assert fp.a4_threshold(32, apple, plate, np.full(128, 5.0), np.full(128, 0.5)) == 1


def test_clock_has_no_rounding_noise_feature():
    features = np.stack([fp.clock_features(t) for t in range(fp.EXPERT_POLICY_STEPS)])
    assert features.std(axis=0).min() > 1e-3


def test_rows_and_rules_are_declared():
    assert "INCONCLUSIVE" in fp.ROWS and "M1-MOTOR" not in fp.ROWS
    assert fp.INTERMEDIATE_STATES == ("M1-MOTOR",)
    assert set(fp.DECLARED_EARLY_STOPS) <= set(fp.ROWS)
    assert "carried P-k" in fp.M2["stop_rule"] and "always run" in fp.M2["stop_rule"]
    assert set(fp.LADDER_ALLOWANCES_L1) == {"a", "b", "c", "d", "e"}
