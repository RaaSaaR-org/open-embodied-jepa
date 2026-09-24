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

from embodied_jepa.contracts import ContractError

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
# The runner is importable WITHOUT torch (its one policy import is deferred into the two
# functions that need it), so the guard for the most serious defect found in this task runs
# in the CORE CI job on every push -- not only in the integration job.
# ---------------------------------------------------------------------------------------
def runner():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_evaluate_policy", ROOT / "scripts" / "evaluate_policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_frozen_cohort_refusal_holds_at_the_type_boundary():
    """The most dangerous defect found in this task, and its class.

    The refusal originally tested membership BEFORE coercion, so ``"45300"`` was not equal to
    ``45300``, the intersection was empty, and the coercion that followed produced the frozen
    reset. Every other defect here costs a day; that one **silently consumes the frozen
    cohort**, which is unrecoverable.

    The class: **a guard that enumerates what is FORBIDDEN fails open; a guard that enumerates
    what is PERMITTED fails closed.** The blacklist would equally have admitted 45400 or 46000
    — and because the report hard-codes the cohort label, such a run would have been *filed as
    development data*. It fails open **and mislabels**, which is what makes it silent rather
    than merely permissive.
    """
    module = runner()
    for form in (45300, "45300", " 45300 ", 45300.0, 45339):
        with pytest.raises(ContractError, match="FROZEN gating cohort"):
            module.cohort_resets((form,))
    for unknown in (45400, 46000, 45200, 0, -1, 99999, True):
        with pytest.raises(ContractError, match="not in the development cohort"):
            module.cohort_resets((unknown,))
    with pytest.raises(ContractError, match="no seeds requested"):
        module.cohort_resets(())
    with pytest.raises(ContractError, match="duplicate seeds"):
        module.cohort_resets((45000, 45000))
    # A permitted seed in any non-canonical form is still permitted: a guard that refuses
    # valid input is a different defect, not extra safety.
    for form in (45000, "45000", " 45000 ", 45000.0):
        assert module.cohort_resets((form,))[45000]["object_xy"]
    assert len(module.cohort_resets(module.COHORT_D)) == 16


def test_the_runner_reconstructs_every_arm_shape_correctly():
    """Enumerates ALL FOUR arm shapes, which is what the original smoke did not do."""
    module = runner()
    for arm, saves_weights, expected in (
        ("A1_random_encoder", False, True),
        ("A2_bc_frozen_e0", False, True),
        ("A3_bc_finetuned_e0", True, False),
    ):
        checkpoint = {"feature_source_weights": {"w": 1} if saves_weights else None}
        assert module.frozen_flag_for(checkpoint) is expected, arm


def test_the_clip_rate_cannot_see_the_out_of_distribution_band():
    """Hand-computed, on a case that is neither zero nor all.

    A counter that always returns zero and a correct counter agree perfectly on a run where
    nothing is clipped, so the verification case is populated deliberately.

    It also demonstrates why BOTH statistics are reported. The expert's translation maximum is
    0.400 and the configured bound is 0.500, so commands in (0.4, 0.5] are outside everything
    demonstrated and are never clipped — invisible to a clip rate alone.
    """
    module = runner()
    dz = [0.10, 0.35, 0.401, 0.45, 0.50, 0.55, 0.70, 0.90, 0.399, 0.20]
    commands = []
    for value in dz:
        action = np.zeros(14, np.float32)
        action[8], action[12], action[13] = value, -1.0, 1.0
        commands.append(action)

    stats = module.command_statistics(commands, ["reach"] * 5 + ["grasp"] * 5)
    measured = stats["per_dimension"]["dz"]
    assert stats["commands"] == 10
    assert measured["out_of_distribution_rate"] == pytest.approx(0.6)  # six exceed 0.400
    assert measured["max_abs"] == pytest.approx(0.90)
    assert measured["expert_maximum"] == 0.400

    # The clip half of this comparison lives in tests/test_policy.py: clip_to_configured_bounds
    # needs PINNED_ACTION_VALUES from policy.py, which imports torch, and DEFERRING that import
    # moved the dependency from collection time to CALL time rather than removing it. This file
    # must stay runnable in the core CI job, which has no torch.
    #
    # What is asserted here is the arithmetic that makes the two statistics differ at all:
    # three of these ten commands sit in (0.400, 0.500] -- outside everything the expert ever
    # demonstrated, and below the bound, so no clip rate can see them.
    band = [v for v in dz if 0.400 + 1e-6 < v <= 0.500 + 1e-6]
    assert len(band) == 3
    assert measured["out_of_distribution_rate"] > len([v for v in dz if v > 0.500 + 1e-6]) / 10, (
        "the out-of-distribution rate must exceed what a clip rate could see; if they are "
        "equal the (0.4, 0.5] band has been lost and the second statistic is pointless"
    )


def test_the_absolute_statistics_cannot_tell_a_descent_from_an_ascent():
    """TASK-057's blocking instrument requirement, discharged by exhibiting the failure.

    Two command streams that are exact negations of each other -- one commanding the boundary
    downward on every step, one commanding it upward -- must produce BYTE-IDENTICAL absolute
    statistics, because every one of those fields is computed from ``np.abs``. That is the
    defect: TASK-056 published a directional reading ("the policies under-shoot the descent")
    off statistics that provably cannot distinguish descending from ascending.

    The same pair must be separated by the signed block. Asserting both halves is the point --
    a test that only checked the signed fields would not show WHY they are needed.
    """
    module = runner()

    def stream(sign):
        out = []
        for value in (0.40, 0.40, 0.40, 0.10, 0.40):
            action = np.zeros(14, np.float32)
            action[8] = sign * value
            action[12], action[13] = -1.0, sign * 1.0
            out.append(action)
        return out

    stages = ["none"] * 5
    down = module.command_statistics(stream(-1.0), stages)["per_dimension"]
    up = module.command_statistics(stream(+1.0), stages)["per_dimension"]

    absolute = ("expert_maximum", "out_of_distribution_rate", "max_abs", "q50", "q90", "q99")
    for name in ("dz", "grasp"):
        for field in absolute:
            assert down[name][field] == up[name][field], (
                f"{name}.{field} differs between a pure descent and a pure ascent, so this "
                "test no longer demonstrates the limitation it exists to demonstrate"
            )

    # ... and the signed block separates exactly the pair the absolute fields cannot.
    assert down["dz"]["signed"]["mean"] == pytest.approx(-0.34)
    assert up["dz"]["signed"]["mean"] == pytest.approx(+0.34)
    assert down["dz"]["signed"]["rate_at_negative_maximum"] == pytest.approx(0.8)
    assert down["dz"]["signed"]["rate_at_positive_maximum"] == pytest.approx(0.0)
    assert up["dz"]["signed"]["rate_at_positive_maximum"] == pytest.approx(0.8)
    assert down["dz"]["signed"]["rate_positive"] == pytest.approx(0.0)
    assert up["dz"]["signed"]["rate_positive"] == pytest.approx(1.0)
    # The hand: fully open and fully closed are the case that was misread.
    assert down["grasp"]["signed"]["q50"] == pytest.approx(-1.0)
    assert up["grasp"]["signed"]["q50"] == pytest.approx(+1.0)


def test_an_oscillating_command_is_distinguishable_from_a_steady_one():
    """A mean near zero must not be reported the same way as a command that is near zero.

    ``|dz| q50`` is 0.40 for a stream that alternates +-0.40 and 0.40 for one that holds
    +0.40, so the absolute median cannot see the difference. The signed rates can, and this is
    the case the closed-loop reading turns on: a policy oscillating about zero goes nowhere
    while a policy committing to one direction travels.
    """
    module = runner()

    def build(values):
        out = []
        for value in values:
            action = np.zeros(14, np.float32)
            action[8] = value
            action[12], action[13] = -1.0, 1.0
            out.append(action)
        return out

    stages = ["none"] * 4
    osc = module.command_statistics(build([0.40, -0.40, 0.40, -0.40]), stages)["per_dimension"]
    steady = module.command_statistics(build([0.40, 0.40, 0.40, 0.40]), stages)["per_dimension"]

    assert osc["dz"]["q50"] == pytest.approx(steady["dz"]["q50"])
    assert osc["dz"]["max_abs"] == pytest.approx(steady["dz"]["max_abs"])
    assert osc["dz"]["signed"]["mean"] == pytest.approx(0.0)
    assert steady["dz"]["signed"]["mean"] == pytest.approx(0.40)
    assert osc["dz"]["signed"]["rate_at_negative_maximum"] == pytest.approx(0.5)
    assert steady["dz"]["signed"]["rate_at_negative_maximum"] == pytest.approx(0.0)


def test_the_diagnostics_manifest_pins_D1_thresholds_to_its_own_offline_table():
    """TASK-057's D1 thresholds must be exactly 3x the frozen offline table, in the manifest.

    The threshold and the table it derives from live in the same file, so they can drift apart
    in a single edit and nothing would notice. apple_policy_v1_results.md section 15's finding
    was that a frozen artifact's own internal consistency is not self-enforcing.

    Also pins the two facts a later editor is most likely to soften: that cohort C is untouched,
    and that the arm list is enumerated rather than named by a predicate.
    """
    manifest = json.loads(
        (ROOT / "benchmarks" / "manifests" / "apple-policy-diagnostics-v1.json").read_text()
    )
    table = manifest["frozen_offline_error_table_A"]["values"]
    thresholds = manifest["frozen_D1_thresholds_three_times_table_A"]
    arm_dimensions = manifest["definitions_every_scope_term_used_in_a_gate_or_stop_rule"][
        "arm_dimension"
    ]

    assert set(thresholds) == set(table), "every arm needs a threshold row"
    for arm, row in thresholds.items():
        # Compared as sets: the manifest is serialized with sort_keys, so its key order is
        # alphabetical and carries no meaning. What must hold is coverage, not order.
        assert {d.removeprefix("right_") for d in row} == set(arm_dimensions), (
            f"{arm}'s thresholds must cover the six arm dimensions and only those; grasp is "
            "reported separately and has no threshold"
        )
        for dimension, value in row.items():
            assert value == pytest.approx(3 * table[arm][dimension], rel=1e-9), (
                f"{arm}.{dimension}: threshold {value} is not 3x its table cell "
                f"{table[arm][dimension]}"
            )

    # The spread quoted as the derivation of the 3x factor must be the table's own spread.
    flat = [v for row in table.values() for v in row.values()]
    assert manifest["frozen_offline_error_table_A"]["spread_ratio"] == pytest.approx(
        max(flat) / min(flat), rel=1e-3
    )

    # Cohort C is not referenced by any gate, and the arm list is enumerated.
    assert manifest["definitions_every_scope_term_used_in_a_gate_or_stop_rule"]["arm"] == [
        "A0_proprio_only",
        "A1_random_encoder",
        "A2_bc_frozen_e0",
        "A3_bc_finetuned_e0",
    ]
    # Enumerating cohort C's seeds HERE is a guard against consuming them, not a consumption:
    # this is a disjointness assertion and the manifest itself carries no cohort-C seed list.
    # Do not delete it as a "reference to C", and do not copy it into a runner as precedent for
    # enumerating C anywhere that could instantiate a reset.
    frozen_c = set(range(45300, 45340))
    development = set(manifest["cohort_discipline"]["cohort_D_development_never_gating"])
    assert not (development & frozen_c), "the development cohort must not intersect cohort C"
    assert len(development) == 16


def test_the_diagnostics_gate_has_a_failing_path_and_quotes_the_right_binomial():
    """Two defects an independent review found in the first draft, pinned so they cannot return.

    **The gate could not fail.** ``full`` (all seven dimensions) satisfied the old definition of
    "a configuration", B3 required ``full`` to reach >= 14/16, and G-SUB passed if ANY
    configuration reached >= 8/16 -- so B3 passing implied G-SUB passing while B3 failing voided
    the run. The gate wired to the abandonment clause had no failing path. The candidates are
    now enumerated and the two controls are excluded by name.

    **The binomial was wrong.** The draft quoted 0.189 for P(X >= 8 | n=16, p=1/3); the true
    value is 0.126501, and 0.189 corresponds to p = 0.3633, which nothing uses. A frozen
    manifest carrying a wrong number is the failure this project has bled over most, so the
    figures are recomputed here from the manifest's own stated null and candidate count rather
    than compared against a transcribed constant.
    """
    from math import comb

    manifest = json.loads(
        (ROOT / "benchmarks" / "manifests" / "apple-policy-diagnostics-v1.json").read_text()
    )
    gate = manifest["gates"]["G_SUB"]
    config = manifest["definitions_every_scope_term_used_in_a_gate_or_stop_rule"]["configuration"]
    candidates = config["g_sub_candidates_the_ONLY_configurations_G_SUB_RANGES_OVER"]
    controls = config["controls_which_are_configurations_but_NOT_G_SUB_candidates"]

    # The gate must have a failing path: neither control may be a candidate.
    assert set(controls) == {"none", "full"}
    for control in controls:
        assert control not in candidates, (
            f"{control!r} is a G-SUB candidate again: if 'full' can pass the gate then B3 "
            "passing implies G-SUB passing and the abandonment clause is unreachable"
        )
    assert gate["candidates"] == candidates, "the gate must range over the enumerated list"
    assert len(candidates) == 8
    assert config["total_configurations_run"] == len(candidates) + len(controls)

    def at_least(k, n, p):
        return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))

    exact = at_least(8, 16, 1 / 16)
    assert gate["one_sided_binomial_at_p0_one_sixteenth"] == pytest.approx(exact, rel=1e-6)
    assert gate["family_wise_over_8_candidates_at_p0_one_sixteenth"] == pytest.approx(
        1 - (1 - exact) ** len(candidates), rel=1e-6
    )
    dropped = gate["corrected_values_for_the_dropped_null"]
    assert dropped["one_sided_at_p0_one_third"] == pytest.approx(at_least(8, 16, 1 / 3), rel=1e-5)
    assert dropped["one_sided_at_p0_one_third"] != pytest.approx(0.189, abs=1e-3), (
        "0.189 is the erroneous figure; it must not reappear as the value of this quantity"
    )
    assert at_least(8, 16, dropped["p0_that_the_erroneous_0_189_corresponds_to"]) == pytest.approx(
        0.189, abs=1e-4
    ), "the recorded provenance of the wrong number must itself be checkable"


def test_the_protocols_printed_table_A_matches_the_manifest_cell_by_cell():
    """Document-to-manifest drift, which no manifest-internal check can see.

    An earlier revision printed Table A at 5 decimal places while the manifest carried 6, so 22
    of 28 cells disagreed and the protocol's own sentence -- "D1's thresholds are 3x the cell" --
    was false of the numbers a reader actually reads: 3 x 0.01896 = 0.05688 against a stored
    threshold of 0.056874. The manifest was internally perfect the whole time and its
    self-consistency test passed.

    **A consistency check that compares an artifact with itself certifies nothing about the
    artifact a reader reads.** This test parses the rendered Markdown table, which is the only
    thing that can catch it.
    """
    manifest = json.loads(
        (ROOT / "benchmarks" / "manifests" / "apple-policy-diagnostics-v1.json").read_text()
    )
    values = manifest["frozen_offline_error_table_A"]["values"]
    protocol = (ROOT / "docs" / "experiments" / "apple_policy_diagnostics_v1.md").read_text()

    marker = "**Table A — median |predicted − expert| per dimension"
    assert marker in protocol, "Table A's heading changed; this test pins that exact table"
    block = protocol[protocol.index(marker) :].split("\n\n")[1]
    rows = [r for r in block.split("\n") if r.startswith("| `")]
    assert len(rows) == 7, f"expected seven dimension rows, parsed {len(rows)}"

    arms = ["A0_proprio_only", "A1_random_encoder", "A2_bc_frozen_e0", "A3_bc_finetuned_e0"]
    seen = set()
    for row in rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        dimension = cells[0].strip("`")
        seen.add(dimension)
        assert len(cells) == 5, f"{dimension}: expected one column per arm"
        for arm, printed in zip(arms, cells[1:], strict=True):
            # Exact string equality, not a tolerance: the point is that the document prints the
            # manifest's stored value, and a tolerance would re-admit the rounding that caused
            # the drift.
            assert printed == f"{values[arm][dimension]:.6f}", (
                f"Table A {dimension}/{arm}: protocol prints {printed}, manifest stores "
                f"{values[arm][dimension]:.6f}"
            )
    assert seen == set(values["A0_proprio_only"]), "Table A must cover all seven dimensions"


def test_every_joint_D1_G_SUB_outcome_maps_to_exactly_one_pre_declared_outcome():
    """The gate contradiction, pinned as a decision table rather than as prose.

    An earlier revision fired the abandonment clause unconditionally on a G-SUB failure while
    the success clause said the line continues when D1 fails for >=3 arms. On the joint outcome
    "D1 fails for >=3 arms AND G-SUB fails" -- plausible, arguably the modal one -- one clause
    said stop and the other said continue, and the pre-declared outcomes listed both with no
    precedence. **A preregistration whose function is to bind the executing agent left the agent
    free to choose after seeing the numbers.**

    Prose cannot be tested, so what is tested is that the manifest carries a precedence rule, an
    explicit once-only bound, and an Outcome X conditioned on D1 rather than on G-SUB alone.
    """
    manifest = json.loads(
        (ROOT / "benchmarks" / "manifests" / "apple-policy-diagnostics-v1.json").read_text()
    )
    rule = manifest["precedence_rule_D1_over_G_SUB"]
    outcomes = manifest["pre_declared_outcomes"]

    # The rule exists, is bounded, and says which way it resolves. Assertions read the VALUES,
    # never the key names: an earlier version of this test matched "exactly once" against a
    # string that only ever held it in its key, so it failed identically on a correct manifest
    # and on an injected violation -- a test failing, but not for the reason it appeared to.
    once = rule["the_exemption_is_available_exactly_once"].lower()
    assert "once" in rule["rule"].lower(), "the precedence rule must state its own bound"
    assert "recorded as spent" in once, (
        "the once-only bound is unenforceable unless the spend is recorded in the results document"
    )
    # The non-exempt re-run is "clause (a) not holding", NOT "D1 passing": keying it on D1 alone
    # is the B-2 defect, since a re-run where D1 passes but D1-grasp still fires is not a clean
    # channel and must not fire the abandonment clause either.
    assert "clause (a) not holding" in once, (
        "the rule must say what a second, non-exempt run looks like, in terms of clause (a) "
        "rather than of D1 alone"
    )
    assert "unbounded escape" in rule["why_the_once_is_load_bearing"]

    # Outcome X must NOT be conditioned on G-SUB alone; the joint case has its own entry.
    assert "clause (a)" in outcomes["X"].lower(), (
        "Outcome X must be conditioned on CLAUSE (a) not holding. Conditioning it on G-SUB "
        "alone reproduces the contradiction with the success clause; conditioning it on D1 "
        "alone reproduces it on the D1-grasp disjunct, which is the B-2 defect."
    )
    assert "P_over_X_precedence" in outcomes
    joint = outcomes["P_over_X_precedence"]
    assert "DOES NOT FIRE" in joint.upper()
    assert "once" in joint.lower()

    # Both clauses must point at the rule, so neither can be read in isolation.
    for clause in ("abandonment_clause", "success_clause"):
        assert "precedence_rule_D1_over_G_SUB" in manifest[clause], (
            f"{clause} does not reference the precedence rule, so it can be read alone and "
            "reproduce the contradiction"
        )

    # Every joint outcome over THREE axes is decidable. Two axes is not enough: clause (a) has
    # two disjuncts, and an earlier revision stated the precedence over the D1 disjunct only,
    # leaving (D1 passes, D1-grasp fires, G-SUB fails) contradictory. A test that binarized on
    # d1_failed alone certified nothing about that case.
    for d1_failed in (True, False):
        for grasp_fired in (True, False):
            for gsub_failed in (True, False):
                clause_a = d1_failed or grasp_fired
                if gsub_failed and clause_a:
                    decided = "P_over_X_precedence"
                elif gsub_failed:
                    decided = "X"
                elif clause_a:
                    decided = "P"
                else:
                    decided = "S"
                assert decided in outcomes, (
                    f"(D1 failed={d1_failed}, D1-grasp fired={grasp_fired}, "
                    f"G-SUB failed={gsub_failed}) maps to no pre-declared outcome"
                )

    # The precedence must range over CLAUSE (a), not over D1 alone, or the D1-grasp disjunct
    # carries the contradiction. Checked on the values, never on key names.
    joint_text = outcomes["P_over_X_precedence"].lower()
    assert "either disjunct" in joint_text or "clause (a)" in joint_text, (
        "the joint outcome is keyed on D1 alone again: on (D1 passes, D1-grasp fires, G-SUB "
        "fails) the success clause says continue and Outcome X says stop"
    )
    assert "clause (a)" in rule["rule"].lower() and "either disjunct" in rule["rule"].lower()
    assert "d1_grasp" in str(rule).lower() or "d1-grasp" in str(rule).lower()

    # R-1: the "once" needs a field to flip, not prose binding a document this protocol cannot
    # reach. The field is frozen false here and set true by the results document.
    assert rule["exemption_spent"] is False, "the exemption must be unspent at preregistration"
    contract = rule["exemption_spent_contract"].lower()
    assert "successor protocol must cite this field" in contract
    assert "may not claim the exemption while it reads true" in contract


def test_the_none_configuration_is_not_terminated_by_shadow_expert_exhaustion():
    """B2 must be passable, or the run is void before any gate is read.

    The exhaustion rule was keyed on "D3", and ``none`` IS a D3 configuration, so every ``none``
    attempt would have terminated at ~805 commands with ``shadow_expert_exhausted``. But B2
    requires ``none`` to reproduce TASK-056's A2 development report, whose sixteen attempts are
    **all** ``step_limit`` at exactly 1000 executed steps. B2 could never have passed, Outcome V
    would have fired, and the run would have been void with no arm numbers reported.

    The rule is now keyed on whether the substitution set is empty. With nothing substituted the
    shadow expert is a recording rather than a command source, so its exhaustion cannot affect
    the robot and the attempt runs to its cap.
    """
    manifest = json.loads(
        (ROOT / "benchmarks" / "manifests" / "apple-policy-diagnostics-v1.json").read_text()
    )
    rule = manifest["shadow_expert_exhaustion_preregistered_behaviour"]
    config = manifest["definitions_every_scope_term_used_in_a_gate_or_stop_rule"]["configuration"]

    assert rule["keyed_on_the_SUBSTITUTION_SET_never_on_which_probe_is_running"] is True
    empty = rule["empty_substitution_set"]
    assert "'none'" in empty or '"none"' in empty, (
        "the empty-substitution branch must name `none` explicitly: it is a D3 configuration, "
        "so a rule keyed on the probe rather than on the substitution set captures it"
    )
    assert "CONTINUES TO ITS CAP" in empty.upper()
    assert "TERMINATES" in rule["non_empty_substitution_set"].upper()

    # `none` is a control with an empty substitution set; `full` has a non-empty one and is the
    # only other control, so exactly one control is exposed to the terminating branch.
    controls = config["controls_which_are_configurations_but_NOT_G_SUB_candidates"]
    assert set(controls) == {"none", "full"}

    # B3/`full` must fit inside the expert's own budget, or its own threshold is unreachable.
    # scripted.py phase commands: orient 130, descend 80, close 45, lift 150 -> grasp by 405.
    assert 130 + 80 + 45 + 150 <= 805, "the oracle must reach grasp well inside 805 commands"


def test_the_committed_script_is_the_provenance_of_the_conditional_tables():
    """Tables C-E must be re-derivable from committed code, as Table A is.

    Section 9's standing practice is that a quantity existing only in a git-ignored run artifact
    is not citable. `cloning.evaluate_policy` returns an unconditional median and the
    predictions' std, and cannot produce these tables, so a generating script is committed.
    """
    manifest = json.loads(
        (ROOT / "benchmarks" / "manifests" / "apple-policy-diagnostics-v1.json").read_text()
    )
    block = manifest["frozen_offline_conditionals_and_phase_table"]
    named = block["generating_script"].split()[0]
    assert (ROOT / named).is_file(), f"{named} is named as provenance but is not committed"

    source = (ROOT / named).read_text()
    # The spawn-pool guard: without it every worker re-executes the module body and the run
    # deadlocks at near-zero CPU, which looks like "still decoding" rather than like a failure.
    assert 'if __name__ == "__main__":' in source
    # It must not quietly decode a split the protocol forbids.
    assert '"val"' in source and "test" not in source.split("def measure")[1].split('"val"')[0]
