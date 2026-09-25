"""TASK-057 probe runner (``scripts/diagnose_policy.py``): gate arithmetic and guards.

Each test targets a condition, not vocabulary; every guard was also checked by injecting its
violation and confirming the test fails (recorded in the Phase B PR).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_jepa import policy_diagnostics as pd
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "benchmarks" / "manifests" / "apple-policy-diagnostics-v1.json").read_text()
)
ARMS = ["A0_proprio_only", "A1_random_encoder", "A2_bc_frozen_e0", "A3_bc_finetuned_e0"]


def _runner():
    spec = importlib.util.spec_from_file_location(
        "_diagnose", ROOT / "scripts" / "diagnose_policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = _runner()


def _attempt(seed, *, grasp=False, success=False, reason="step_limit", steps=1000, first=None):
    return {
        "seed": seed,
        "grasp": grasp,
        "success": success,
        "termination_reason": reason,
        "executed_steps": steps,
        "first_command": first,
    }


def test_runner_constants_are_the_amended_protocol():
    assert R.DEVICE == "cpu", "amendment 1 pins CPU: the B2 reference ran on CPU"
    assert R.MAX_STEPS == MANIFEST["budget"]["per_attempt_step_cap"]
    assert R.ATTEMPT_WALL_SECONDS == MANIFEST["budget"]["per_attempt_cap_seconds"]
    assert R.GLOBAL_WALL_SECONDS == MANIFEST["budget"]["global_cap_seconds"]
    assert R.CONFIG.name == "apple_policy_v1.yaml"
    assert R.ARM_ORDER[0] == R.PRIMARY == "A2_bc_frozen_e0", "A2 first: its run is B2"
    assert set(R.ARMS) == set(ARMS)


def test_b3_requires_fourteen_of_sixteen():
    assert R.b3_verdict([_attempt(s, grasp=s < 14) for s in range(16)])["passed"]
    assert not R.b3_verdict([_attempt(s, grasp=s < 13) for s in range(16)])["passed"]


def test_b1_needs_oracle_and_zero_for_hold_and_random_on_grasp_and_success():
    oracle = [_attempt(s, grasp=True, success=s < 12) for s in range(16)]
    zeros = [_attempt(s) for s in range(16)]
    assert R.b1_verdict(oracle, zeros, zeros)["passed"]
    assert not R.b1_verdict(
        [_attempt(s, grasp=True, success=s < 11) for s in range(16)], zeros, zeros
    )["passed"]
    assert not R.b1_verdict(
        [_attempt(s, grasp=s < 13, success=True) for s in range(16)], zeros, zeros
    )["passed"]
    grasping_hold = zeros[:-1] + [_attempt(15, grasp=True)]
    assert not R.b1_verdict(oracle, grasping_hold, zeros)["passed"]
    assert not R.b1_verdict(oracle, zeros, grasping_hold)["passed"]
    assert not R.b1_verdict(oracle, zeros[:15], zeros)["passed"]


def test_b2_reproduces_the_reference_attempt_by_attempt():
    reference = (
        json.loads((ROOT / "outputs" / "task056-cohort-d" / "a2.json").read_text())
        if (ROOT / "outputs" / "task056-cohort-d" / "a2.json").exists()
        else {
            "attempts": [
                {
                    "seed": s,
                    "termination_reason": "step_limit",
                    "executed_steps": 1000,
                    "score": {"grasp": s == 45100},
                }
                for s in [*range(45000, 45008), *range(45100, 45108)]
            ]
        }
    )
    same = [
        _attempt(
            int(a["seed"]),
            grasp=bool(a["score"].get("grasp")),
            reason=a["termination_reason"],
            steps=a["executed_steps"],
        )
        for a in reference["attempts"]
    ]
    assert R.b2_verdict(same, reference)["passed"]
    moved = [dict(a) for a in same]
    moved[0]["executed_steps"] -= 1
    assert not R.b2_verdict(moved, reference)["passed"]
    regrasped = [dict(a) for a in same]
    regrasped[0]["grasp"] = not regrasped[0]["grasp"]
    assert not R.b2_verdict(regrasped, reference)["passed"]
    assert not R.b2_verdict(same[:-1], reference)["passed"]


def _row(seed, *, image=True, state=0.0, mask=True, output=0.0):
    return {
        "seed": seed,
        "image_identical": image,
        "state_max_abs": state,
        "mask_identical": mask,
        "output_max_abs": output,
    }


def test_d1_pipeline_fails_on_any_broken_input_or_output():
    kw = {"output_tolerance": 1e-4, "state_tolerance": 1e-6}
    good = [_row(s) for s in range(15)]
    assert not R.d1_pipeline_verdict(good, uses_image=True, **kw)["failed"]
    for broken in (
        _row(3, image=False),
        _row(3, state=1e-5),
        _row(3, mask=False),
        _row(3, output=2e-4),
        _row(3, output=float("nan")),
    ):
        rows = good[:3] + [broken] + good[4:]
        assert R.d1_pipeline_verdict(rows, uses_image=True, **kw)["failed"], broken
    # A0 consumes no image: (i) does not apply to it, (ii) and (iii) do.
    rows = good[:3] + [_row(3, image=False)] + good[4:]
    assert not R.d1_pipeline_verdict(rows, uses_image=False, **kw)["failed"]
    assert R.d1_pipeline_verdict(good[:14], uses_image=True, **kw)["failed"], "missing = failed"


def test_d1_grasp_count_uses_the_amended_cut_and_counts_missing_as_firing():
    first = lambda g: [0, 0, 0, 0, 0, 0, g]  # noqa: E731
    attempts = [_attempt(s, first=first(-1.0)) for s in range(16)]
    assert R.d1_grasp_count(attempts) == 0
    attempts[0]["first_command"] = first(-0.0092)  # the prior-only predictor fires
    attempts[1]["first_command"] = first(-0.5)  # exactly -0.5 does not
    attempts[2]["first_command"] = None  # missing counts as firing
    assert R.d1_grasp_count(attempts) == 2
    assert R.d1_grasp_count(attempts[:10]) == 2 + 6


def test_g_sub_uses_each_candidates_own_threshold_and_never_the_controls():
    thresholds = MANIFEST["gates"]["G_SUB"]["per_candidate_threshold"]
    scores = {c: thresholds[c] - 1 for c in thresholds}
    assert not R.g_sub_verdict(scores, thresholds)["passed"]
    scores["dy"] = 11  # passed the frozen 8/16, below dy's amended 12/16
    assert not R.g_sub_verdict(scores, thresholds)["passed"]
    scores["dz"] = 8
    assert R.g_sub_verdict(scores, thresholds)["passed"]
    with pytest.raises(ContractError):
        R.g_sub_verdict({**scores, "full": 16}, thresholds)
    missing = {c: v for c, v in scores.items() if c != "dz"}
    assert not R.g_sub_verdict(missing, thresholds)["passed"]


@pytest.mark.parametrize("a2_fails", [True, False])
@pytest.mark.parametrize("others_failing", [0, 2, 3])
@pytest.mark.parametrize("grasp_arms", [0, 3])
@pytest.mark.parametrize("gsub", [True, False])
@pytest.mark.parametrize("spent", [False, True])
def test_decision_table_matches_amended_clause_a(a2_fails, others_failing, grasp_arms, gsub, spent):
    others = [a for a in ARMS if a != R.PRIMARY]
    d1_failed = {a: False for a in ARMS}
    d1_failed[R.PRIMARY] = a2_fails
    for a in others[:others_failing]:
        d1_failed[a] = True
    counts = {a: (8 if i < grasp_arms else 7) for i, a in enumerate(ARMS)}
    out = R.decide(
        void=False,
        d1_failed=d1_failed,
        grasp_counts=counts,
        g_sub_passed=gsub,
        exemption_spent=spent,
    )
    failing = sum(d1_failed.values())
    clause_a = a2_fails or failing >= 3 or grasp_arms >= 3
    assert out["clause_a"] == clause_a
    if not gsub and clause_a:
        expected = "X_when_the_exemption_is_already_spent" if spent else "P_over_X_precedence"
    elif not gsub:
        expected = "X"
    else:
        expected = "P" if clause_a else "S"
    assert out["outcome"] == expected
    assert out["outcome"] in MANIFEST["pre_declared_outcomes"]
    assert out["claims_the_exemption"] == (expected == "P_over_X_precedence")
    assert (
        R.decide(
            void=True, d1_failed={}, grasp_counts={}, g_sub_passed=False, exemption_spent=False
        )["outcome"]
        == "V"
    )


def test_d2_reading_uses_the_preregistered_definition():
    tau = 0.05
    immediate = {"seed": 1, "departures": [0.1] * 20}
    gradual = {"seed": 2, "departures": [0.0] * 60 + [0.1] * 10}
    censored = {"seed": 3, "departures": [0.0] * 30 + [None] * 5}
    assert R.d2_reading([immediate], tau)["reading"] == "immediate"
    assert R.d2_reading([gradual], tau)["reading"] == "gradual"
    # Persistence 5: isolated excursions are not a departure.
    spiky = {"seed": 4, "departures": ([0.1, 0.0] * 40) + [0.1] * 5}
    assert R.d2_reading([spiky], tau)["per_seed"][0]["first_departure_step"] == 80
    out = R.d2_reading([censored], tau)
    assert out["per_seed"][0]["censored_at"] == 30 and out["reading"] == "inconclusive"


def test_d1_contrast_is_reported_against_the_frozen_thresholds():
    thresholds = MANIFEST["frozen_D1_thresholds_three_times_table_A"][R.PRIMARY]
    attempt = {
        "first_command": [0.4, 0, 0, 0, 0, 0, -1],
        "first_shadow_expert": [-0.4, 0, 0, 0, 0, 0, -1],
    }
    out = R.d1_contrast([attempt], thresholds)
    assert out["per_dimension"]["dx"]["median_abs"] == pytest.approx(0.8)
    assert not out["would_pass_frozen_rule"]
    assert "ratio" not in out["per_dimension"]["grasp"], "grasp has no D1 threshold"


def test_past_close_fraction_counts_commands_after_the_expert_close_phase():
    rows = [{"shadow_expert_phase": p} for p in ("orient", "close", "lift", "transfer", "done")]
    assert R.past_close_fraction({"trace": rows}) == pytest.approx(3 / 5)


def test_hash_verification_refuses_a_changed_input(tmp_path, monkeypatch):
    manifest = {"amendment_1": {"hashes": {"x.txt (note)": "0" * 64}}}
    (tmp_path / "x.txt").write_text("changed")
    monkeypatch.setattr(R, "ROOT", tmp_path)
    with pytest.raises(ContractError, match="sha256"):
        R.verify_hashes(manifest)


def test_val_root_whitelist_refuses_anything_but_the_fifteen_val_roots():
    pytest.importorskip("torch")
    resets = R.val_root_resets(MANIFEST)
    assert sorted(resets) == sorted(MANIFEST["gates"]["D1"]["val_root_seeds"])
    for bad in (
        [*MANIFEST["gates"]["D1"]["val_root_seeds"][:14], 45300],
        [*MANIFEST["gates"]["D1"]["val_root_seeds"][:14], 48000],
        MANIFEST["gates"]["D1"]["val_root_seeds"][:14],
    ):
        broken = json.loads(json.dumps(MANIFEST))
        broken["gates"]["D1"]["val_root_seeds"] = bad
        with pytest.raises(ContractError):
            R.val_root_resets(broken)


def test_the_runner_never_writes_the_manifest_or_spends_the_exemption():
    source = (ROOT / "scripts" / "diagnose_policy.py").read_text()
    assert "MANIFEST.write" not in source and 'exemption_spent"] =' not in source
    # Development resets come only through evaluate_policy's whitelist, over exactly COHORT_D.
    assert "seeds = tuple(r.COHORT_D)" in source and "r.cohort_resets(seeds)" in source
    with pytest.raises(ContractError):
        R.runner().cohort_resets((45300,))


class _Robot:
    def __init__(self):
        self.stopped = None

    def observe(self):
        return SimpleNamespace(images={}, state=None)

    def project_candidates(self, command):
        return SimpleNamespace(feasible=np.ones((1, 1), bool), actions=command)

    def execute(self, action):
        return SimpleNamespace(applied_action=np.asarray(action).copy(), reason=None)

    def stop(self, reason):
        self.stopped = reason


class _Stub:
    max_steps, phases = 100, ()
    done, phase, phase_index = False, "stub", 0

    def action(self, robot):  # noqa: ARG002
        a = np.zeros(14, np.float32)
        a[12:] = -1
        return a

    def advance(self, result):  # noqa: ARG002
        return None


def _clip(action, lower, upper):
    c = np.clip(action, lower, upper).astype(np.float32)
    return c, bool((c != action).any()), (c != action)[list(pd.FREE_INDICES)]


def _attempt_loop(scores, **kw):
    it = iter(scores)
    return pd.run_diagnostic_attempt(
        pd.hold_controller(),
        _Robot(),
        SimpleNamespace(evaluate=lambda: next(it)),
        configuration="none",
        clip=_clip,
        lower=np.array((0.0,) * 6 + (-0.5,) * 6 + (-1.0, -1.0), np.float32),
        upper=np.array((0.0,) * 6 + (0.5,) * 6 + (-1.0, 1.0), np.float32),
        deadline_seconds=5.0,
        shadow=pd.ShadowExpert(None, factory=lambda t: _Stub()),
        **kw,
    )


def test_every_trace_row_records_per_step_contact_height_and_drop(monkeypatch):
    monkeypatch.setattr(pd, "palm_apple_distance", lambda robot: 0.0)
    scores = [
        {"hand_contact": i == 1, "object_height_m": 0.7 + i, "dropped": False, "grasp": i == 1}
        for i in range(3)
    ]
    out = _attempt_loop(scores, max_steps=3, trace=True)
    after = [r["after"] for r in out["trace"]]
    assert [a["hand_contact"] for a in after] == [False, True, False]
    assert [a["object_height_m"] for a in after] == pytest.approx([0.7, 1.7, 2.7])
    assert after[1]["stage"] == "grasp"


def test_the_per_attempt_wall_cap_ends_an_attempt():
    scores = [{"success": False}] * 50
    out = _attempt_loop(scores, max_steps=50, trace=False, attempt_wall_seconds=0.0)
    assert out["termination_reason"] == "attempt_wall_cap" and out["executed_steps"] == 1
