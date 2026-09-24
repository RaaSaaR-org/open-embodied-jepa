"""Guards for the Apple->Plate policy protocol v1 (TASK-056).

The protocol builds every learned arm on a FROZEN feature source: the v4 E0 checkpoint
``checkpoints/task054-wm-v4/leworldmodel_baseline.pt``. ``VisualModel.load`` refuses a
checkpoint whose ``implementation_sha256`` does not match, and that hash is a digest over
``models/base.py``, the backend module, ``models/readout.py`` and ``readout_labels.py``.

So any edit to those files makes the E0 checkpoint unloadable and destroys the protocol's
feature source before a single arm is trained. The first test pins the hash. It recomputes
the digest with the standard library only, so it runs in core CI where neither torch, the
pinned LeWM source, nor the git-ignored checkpoint is available.

The remaining tests pin the protocol's two derived cohort claims -- the surviving-root counts
and the frozen evaluation cohort -- so a reviewer does not have to take the preregistration's
arithmetic on trust.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-policy-v1.json"

# benchmarks/manifests/apple-world-model-v4.json results.E0_baseline_arm_b_rerun
E0_IMPLEMENTATION_SHA256 = "4ad0a6856d847aaf724daba5e721c078ee599658d4733357ab8703eccbb15a92"


def implementation_digest(backend_module: str) -> str:
    """Reproduce VisualModel.implementation_sha256 without importing torch.

    Mirrors ``models/base.py``: for each of base.py, the backend module, readout.py and
    readout_labels.py, update with the file NAME and then its bytes, in that order.
    """
    models = ROOT / "src" / "embodied_jepa" / "models"
    digest = hashlib.sha256()
    for path in (
        models / "base.py",
        models / backend_module,
        models / "readout.py",
        models.parent / "readout_labels.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_e0_checkpoint_stays_loadable():
    """TASK-056's frozen feature source must not be invalidated by an edit.

    If this fails, someone changed models/base.py, models/lewm.py, models/readout.py or
    readout_labels.py. That is not a test to update: it means the v4 checkpoints can no
    longer be loaded and TASK-056's preregistration rests on a checkpoint that no longer
    exists. Revert the edit, or retire the protocol deliberately.
    """
    assert implementation_digest("lewm.py") == E0_IMPLEMENTATION_SHA256


def test_the_digest_is_sensitive_to_each_input_file():
    """A guard that cannot detect a change is not a guard."""
    models = ROOT / "src" / "embodied_jepa" / "models"
    names = ("base.py", "lewm.py", "readout.py")
    for name in names:
        digest = hashlib.sha256()
        for path in (
            models / "base.py",
            models / "lewm.py",
            models / "readout.py",
            models.parent / "readout_labels.py",
        ):
            digest.update(path.name.encode())
            body = path.read_bytes()
            digest.update(body + b"\n# perturbation\n" if path.name == name else body)
        assert digest.hexdigest() != E0_IMPLEMENTATION_SHA256, f"digest ignores {name}"


def test_manifest_is_wellformed_and_declares_the_frozen_feature_source():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["task"] == "TASK-056"
    assert manifest["protocol"] == "apple_policy_v1"
    assert manifest["closed_loop_started"] is False
    # Results accumulate stage by stage, so this is no longer "results is None". What must
    # hold at every stage until the third authorization is that cohort C is untouched.
    results = manifest["results"]
    if results is not None:
        assert results["cohort_C_simulated"] is False
        assert results["learned_apple_to_plate_successes"] == 0
    source = manifest["frozen_feature_source"]
    assert source["model_implementation_sha256"] == E0_IMPLEMENTATION_SHA256
    assert manifest["dataset"]["test_split_decoded"] is False


def test_surviving_root_counts_match_the_committed_collection_rule():
    """The preregistration claims 137/15/8 surviving roots; recompute it from the rule."""
    frozen_seeds = tuple(range(48000, 48200))
    order = list(frozen_seeds)
    np.random.default_rng(48).shuffle(order)
    n_val, n_test = max(1, round(0.10 * len(order))), max(1, round(0.05 * len(order)))
    split = {
        seed: ("val" if i < n_val else "test" if i < n_val + n_test else "train")
        for i, seed in enumerate(order)
    }
    aim = {seed for i, seed in enumerate(frozen_seeds) if i % 5 == 4}
    surviving = {
        name: sum(1 for s in frozen_seeds if split[s] == name and s not in aim)
        for name in ("train", "val", "test")
    }
    assert surviving == {"train": 137, "val": 15, "test": 8}

    declared = json.loads(MANIFEST.read_text())["exclusions"]
    assert declared["surviving_roots"]["train"] == 137
    assert declared["surviving_roots"]["val"] == 15
    assert declared["surviving_roots"]["test"] == 8
    assert declared["aim_offset_roots"]["total"] == len(aim) == 40
    assert declared["aim_offset_roots"]["train"] == sum(
        1 for s in frozen_seeds if split[s] == "train" and s in aim
    )


# The generator cross-check runs to this tolerance, not to exact equality. numpy's compiled
# Generator.uniform evaluates ``low + range * next_double``; whether that multiply-add contracts
# to an FMA depends on the numpy build's compiler and target, so the result can differ by one
# ULP (~1.4e-17 m here) between macOS arm64 and Linux x86-64 even though the underlying PCG64
# doubles are integer-derived and bit-exact everywhere. 1e-12 m is five orders looser than that
# ULP and ten orders tighter than the smallest physically meaningful quantity in this protocol
# (the 1.5 cm gates, the +-3 cm jitter). The repo hit the same class of problem in
# tests/test_apple_wide_collection.py, which pins a plan hash rounded to 1e-9 on every platform
# and the exact bytes only on macOS arm64.
GENERATOR_CHECK_TOLERANCE_M = 1e-12


def test_frozen_cohort_is_reproducible_and_disjoint():
    """The STORED values define cohort C; the generator is cross-checked to a tolerance.

    Two different things are asserted here and the distinction is the point:

    * **The seal** is the sha256 over the decimals stored in the manifest. It never re-runs the
      generator, so it reproduces on any platform, and it is what freezes the cohort.
    * **The cross-check** re-runs ``wide_reset`` and compares to the stored values only to prove
      the stored cohort came from the committed rule and not from a lookalike. That purpose is
      fully served at 1e-12 m, and exact equality was the wrong assertion: it passed on macOS
      arm64 and failed on Linux x86-64 by one ULP.
    """
    manifest = json.loads(MANIFEST.read_text())
    cohort = manifest["cohorts"]["C_frozen_gating"]
    centers = {"object_xy": (0.34, -0.18), "plate_xy": (0.49, -0.09)}
    jitter = {"object_xy": 0.03, "plate_xy": 0.02}

    def wide_reset(seed):
        rng = np.random.default_rng(seed)
        return {
            key: (np.array(centers[key]) + rng.uniform(-jitter[key], jitter[key], 2)).tolist()
            for key in ("object_xy", "plate_xy")
        }

    assert cohort["n"] == 40 == len(cohort["resets"])
    for seed in range(45300, 45340):
        expected = wide_reset(seed)
        stored = cohort["resets"][str(seed)]
        for key in ("object_xy", "plate_xy"):
            assert stored[key] == pytest.approx(expected[key], abs=GENERATOR_CHECK_TOLERANCE_M), (
                f"{key} of seed {seed} did not come from the committed wide_reset rule"
            )

    # The seal: a digest over the STORED decimals. json.loads -> float -> repr is idempotent
    # (CPython's shortest-round-trip repr and correctly-rounded strtod are platform-independent),
    # so this pin holds on any machine and does not depend on re-running the generator.
    blob = json.dumps(cohort["resets"], sort_keys=True, separators=(",", ":"))
    assert json.dumps(json.loads(blob), sort_keys=True, separators=(",", ":")) == blob
    assert hashlib.sha256(blob.encode()).hexdigest() == cohort["cohort_sha256"]
    assert cohort["values_are_the_definition"] is True

    consumed = (
        set(range(42000, 42032))
        | set(range(43000, 43005))
        | set(range(44000, 44020))  # reserved for TASK-034; must stay untouched
        | set(range(45000, 45008))
        | set(range(45100, 45108))
        | set(range(45200, 45208))
        | set(range(48000, 48200))
        | set(range(48900, 48932))
        | set(range(49000, 49016))
        | set(range(49100, 49132))
    )
    assert not consumed & set(cohort["seeds"])


def test_the_generator_matches_the_prior_frozen_cohorts():
    """The cohort rule must be the one the earlier ceilings used, not a lookalike.

    Same tolerance and same reasoning as the cross-check above: this one happened to pass on
    macOS arm64, but it carried the identical exact-equality exposure.
    """
    prior = json.loads(
        (ROOT / "benchmarks" / "manifests" / "apple-wide-grasp-closure-v3.json").read_text()
    )
    rng = np.random.default_rng(45200)
    obj = (np.array((0.34, -0.18)) + rng.uniform(-0.03, 0.03, 2)).tolist()
    plate = (np.array((0.49, -0.09)) + rng.uniform(-0.02, 0.02, 2)).tolist()
    assert prior["resets"]["45200"]["object_xy"] == pytest.approx(
        obj, abs=GENERATOR_CHECK_TOLERANCE_M
    )
    assert prior["resets"]["45200"]["plate_xy"] == pytest.approx(
        plate, abs=GENERATOR_CHECK_TOLERANCE_M
    )


def test_gate_thresholds_are_the_preregistered_ones():
    """Pin the numbers a later edit could quietly loosen."""
    gates = json.loads(MANIFEST.read_text())["gates"]
    assert gates["G1_full_success_min"] == 17
    assert gates["G1_n"] == 40
    assert gates["G2_minus_A0_min"] == 8
    assert gates["G3_minus_A1_min"] == 8
    assert gates["G4_grasp_min"] == 20
    assert gates["G5_hold_max"] == 0 and gates["G5_random_max"] == 0
    assert gates["G6_privileged_reads_max"] == 0


def script_gates() -> dict:
    """Read the runner's frozen thresholds without importing it.

    ``ast`` rather than an import, so this runs in the core CI job that has neither torch
    nor the data extras installed.
    """
    source = (ROOT / "scripts" / "measure_policy_preflight.py").read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "GATES" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("scripts/measure_policy_preflight.py defines no GATES mapping")


@pytest.mark.parametrize(
    ("script_key", "manifest_key"),
    [
        ("P1_close_palm_apple_median_m", "P1_close_phase_palm_apple_median_m"),
        ("P2_approach_palm_apple_median_m", "P2_orient_descend_palm_apple_median_m"),
        ("P3_orient_apple_position_median_m", "P3_orient_apple_position_median_m"),
        ("max_ratio_to_shuffled_frame_control", "max_ratio_to_shuffled_frame_control"),
    ],
)
def test_preflight_thresholds_agree_between_runner_and_manifest(script_key, manifest_key):
    """The abort rule is only binding if the runner and the manifest agree on the number."""
    manifest = json.loads(MANIFEST.read_text())["preflight"]["gates"]
    assert script_gates()[script_key] == manifest[manifest_key]


def test_preflight_thresholds_are_the_preregistered_ones():
    assert script_gates() == {
        "P1_close_palm_apple_median_m": 0.015,
        "P2_approach_palm_apple_median_m": 0.015,
        "P3_orient_apple_position_median_m": 0.020,
        "max_ratio_to_shuffled_frame_control": 0.7,
    }


def test_no_preflight_threshold_is_passable_by_a_blind_model():
    """The review's B2: P2 at 2.5 cm and P3 at 3.0 cm were passable without vision.

    A predictor that always emits the reset centre and never looks at anything has median
    apple-position error equal to the median norm of a uniform square of half-width 0.03 m
    (``wide_reset``). Every absolute pre-flight threshold must sit strictly below it, or the
    gate certifies nothing.
    """
    rng = np.random.default_rng(0)
    blind = float(np.median(np.linalg.norm(rng.uniform(-0.03, 0.03, size=(400_000, 2)), axis=1)))
    assert blind == pytest.approx(0.023937, abs=1e-4)

    gates = script_gates()
    for name, threshold in gates.items():
        if name == "max_ratio_to_shuffled_frame_control":
            continue
        assert threshold < blind, f"{name} is passable by a model that never reads the image"

    manifest = json.loads(MANIFEST.read_text())
    assert manifest["preflight"]["no_vision_control"][
        "P0b_analytic_prior_median_m"
    ] == pytest.approx(blind, abs=1e-4)
    # The same constant is recorded top-level with its derivation, so a future reader does not
    # have to rediscover why the gates sit where they do.
    assert manifest["blind_prior_baseline"]["median_error_m"] == pytest.approx(blind, abs=1e-4)
    within = float(
        (np.linalg.norm(rng.uniform(-0.03, 0.03, size=(400_000, 2)), axis=1) < 0.015).mean()
    )
    assert manifest["blind_prior_baseline"][
        "fraction_of_resets_within_1_5_cm_of_centre"
    ] == pytest.approx(within, abs=5e-3)


def test_outcome_E_has_no_branch_that_continues_the_run():
    """The near-miss band may steer the NEXT task; it may never let this one proceed.

    A binding abort with a graded reading is exactly the shape an agent under run-time
    pressure would try to reinterpret, so the manifest has to say so in terms a reviewer can
    check, and the runner has to exit non-zero for every band.
    """
    outcomes = json.loads(MANIFEST.read_text())["pre_declared_outcomes"]
    text = outcomes["E_p1_fails"].lower()
    assert "stops before training, in every sub-case" in text
    assert "no sub-case permits continuing" in text

    sub = outcomes["E_sub_cases_affect_only_the_next_task"]
    assert "never whether this one proceeds" in sub["binding_note"]
    assert set(sub) >= {"E_near", "E_clear", "binding_note", "control_reported_either_way"}

    # The runner returns 2 from a single P1-failure branch; the band only labels the message.
    source = (ROOT / "scripts" / "measure_policy_preflight.py").read_text()
    body = source.split('if not gates["P1_close_palm_apple"]["passed"]:', 1)[1]
    assert body.split("return 2")[0].count("return 0") == 0


def test_a_ratio_only_failure_is_never_labelled_a_near_miss():
    """A small absolute value with a failing ratio is the SHORTCUT case, not a near miss.

    E-near's prescribed follow-up is resolution and camera placement. The protocol's own words
    for a ratio failure are "no amount of resolution fixes that", so mislabelling one as E-near
    would point the next generation at the wrong remedy.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_preflight", ROOT / "scripts" / "measure_policy_preflight.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Absolute value well inside the near-miss band, but the control ratio failed.
    assert module.preflight_band({"value": 0.009, "ratio_passed": False}) == "E-prior"
    assert module.preflight_band({"value": 0.025, "ratio_passed": False}) == "E-prior"
    # Ratio passed: the absolute value chooses the band.
    assert module.preflight_band({"value": 0.017, "ratio_passed": True}) == "E-near"
    assert module.preflight_band({"value": 0.025, "ratio_passed": True}) == "E-clear"

    outcomes = json.loads(MANIFEST.read_text())["pre_declared_outcomes"][
        "E_sub_cases_affect_only_the_next_task"
    ]
    assert "E_prior" in outcomes
    assert "takes precedence" in outcomes["precedence_within_E"].lower()


# ---------------------------------------------------------------------------------------
# Tripwires for scripts/evaluate_policy.py, which does not exist yet.
#
# Both constraints below are recorded in benchmarks/manifests/apple-policy-v1.json as blocking
# requirements on that runner. A requirement recorded in prose depends on someone reading it;
# a requirement encoded in a test depends on nobody deleting it. The person who writes that
# runner may well never open the manifest, so these fail by themselves instead.
#
# While the file does not exist they PASS VACUOUSLY -- deliberately a bare return, not
# pytest.skip: .github/workflows/integration.yml rejects every skip whose message is not
# "graphics opt-in" or "MPS unavailable", so skipping here would fail that job. The moment the
# file appears without the constraint, CI goes red and the assertion message says why.
# ---------------------------------------------------------------------------------------
EVALUATE_POLICY = ROOT / "scripts" / "evaluate_policy.py"


def test_the_closed_loop_runner_clips_to_the_protocols_bounds_not_the_contracts():
    """ClonedPolicy.act clips to [-1, 1] because validate_actions demands it.

    The protocol's frozen right-arm bounds are +-0.5 (configs/apple_wm_v4.yaml), so a runner
    that executes act()'s output unmodified could command twice the delta every prior
    generation operated under -- silently, because the action is still contract-valid.
    """
    if not EVALUATE_POLICY.exists():
        return  # vacuously green until the runner is written
    source = EVALUATE_POLICY.read_text()
    assert any(
        marker in source
        for marker in ("lower_bounds", "upper_bounds", "_check_pinned_bounds", "project_candidates")
    ), (
        "scripts/evaluate_policy.py must apply the protocol's configured action bounds "
        "(planner.lower_bounds / planner.upper_bounds, +-0.5 on the right arm) and/or the "
        "embodiment's projection to the policy's output. ClonedPolicy.act clips only to the "
        "contract range [-1, 1], so executing its output unmodified doubles the permitted "
        "right-arm delta. See benchmarks/manifests/apple-policy-v1.json: "
        "cohorts.C_frozen_gating.runner_must_clip_to_configured_bounds."
    )


def test_the_closed_loop_runner_resets_cohort_c_from_stored_values():
    """numpy's Generator.uniform can differ by one ULP across platforms.

    A runner that recomputes the resets from ``wide_reset`` would let two machines execute
    subtly different cohorts while both passing the manifest's digest check, because that
    digest is over the STORED decimals.
    """
    if not EVALUATE_POLICY.exists():
        return  # vacuously green until the runner is written
    source = EVALUATE_POLICY.read_text()
    assert "apple-policy-v1.json" in source or "cohort" in source, (
        "scripts/evaluate_policy.py must instantiate each cohort-C reset from the STORED "
        "values in benchmarks/manifests/apple-policy-v1.json, not recompute them from "
        "wide_reset. See cohorts.C_frozen_gating.runner_must_reset_from_stored_values."
    )
    assert "wide_reset(" not in source.replace("# ", ""), (
        "scripts/evaluate_policy.py calls wide_reset() directly. Cohort C is defined by the "
        "stored manifest values; recomputing them can differ by one ULP across platforms and "
        "would let two machines run subtly different cohorts while both passing the digest "
        "check. Read the resets from the manifest instead."
    )
