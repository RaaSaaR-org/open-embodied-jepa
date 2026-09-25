"""TASK-057 diagnostic run: ``apple_policy_diagnostics_v1`` as amended by amendments 1 and 2.

Runs, in the preregistered order, and stops at the first voiding control:

1. **B3** -- ``full`` (all seven free components from the shadow expert) with A2 as the
   controller, 16 development seeds. < 14/16 grasp resets voids the run.
2. **D1-pipeline** -- the 15 surviving val roots' recorded resets, zero commands executed:
   closed-loop observation and ``act`` output against the stored training frame and the arm's
   offline prediction on it.
3. **B1** -- ``scripted_oracle``, ``hold``, ``random`` on the 16 development seeds.
4. **Arms, unsubstituted** (``none``) -- A2 first, which is also **B2** (it must reproduce
   ``outputs/task056-cohort-d/a2.json``), then A0, A1, A3. Gives D1-contrast, D1-grasp and D2.
5. **D3** -- A2 under each of the eight G-SUB candidates.

Then evaluates every gate and the pre-declared outcome. **Nothing is retuned after the gates
are read, and the manifest is never written by this runner** -- ``exemption_spent`` is set, if
at all, by the results document.

CPU only (amendment 1: the B2 reference ran on CPU). Cohort C (45300-45339) and the test split
are never touched: development seeds go through ``evaluate_policy.cohort_resets`` (whitelist),
val roots through this module's own 15-seed whitelist, checked against the collection plan.

    uv run --no-sync python scripts/diagnose_policy.py --output outputs/task057-diagnostics/run
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import policy_diagnostics as pd  # noqa: E402
from embodied_jepa.contracts import ContractError  # noqa: E402

PROTOCOL = "apple_policy_diagnostics_v1"
AMENDMENTS = (1, 2)
TASK = "TASK-057"
MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-policy-diagnostics-v1.json"
CONFIG = ROOT / "configs" / "apple_policy_v1.yaml"
CHECKPOINTS = ROOT / "checkpoints" / "task056-policy-v1"
B2_REFERENCE = ROOT / "outputs" / "task056-cohort-d" / "a2.json"
ARMS = {
    "A0_proprio_only": "a0",
    "A1_random_encoder": "a1",
    "A2_bc_frozen_e0": "a2",
    "A3_bc_finetuned_e0": "a3",
}
PRIMARY = "A2_bc_frozen_e0"
#: Unsubstituted arm order: A2 first, because its run is B2 and B2 voids everything after it.
ARM_ORDER = (PRIMARY, "A0_proprio_only", "A1_random_encoder", "A3_bc_finetuned_e0")
MAX_STEPS = 1000
ATTEMPT_WALL_SECONDS = 300.0
GLOBAL_WALL_SECONDS = 21600.0
CONTROL_DEADLINE_SECONDS = 5.0
IMAGE_SIZE = 112
DEVICE = "cpu"
COHORT_C = frozenset(range(45300, 45340))


class VoidRun(Exception):
    """A voiding control failed; the run stops and reports (protocol §12)."""


class GlobalCap(Exception):
    """The 6 h global cap was reached; the run stops and reports what it has."""


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runner():
    return _load("_evaluate_policy", "scripts/evaluate_policy.py")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----- preflight -----------------------------------------------------------------------------
def frozen_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def verify_hashes(manifest) -> dict:
    """Every hashed input must equal amendment 1's record; a mismatch refuses the run."""
    recorded = manifest["amendment_1"]["hashes"]
    checked = {}
    for key, want in recorded.items():
        relative = key.split(" (")[0]
        got = sha256(ROOT / relative)
        if got != want:
            raise ContractError(f"{relative}: sha256 {got} != amendment 1's {want}")
        checked[relative] = got
    return checked


def val_root_resets(manifest) -> dict:
    """The D1-pipeline whitelist: exactly the manifest's 15 seeds, each a surviving val root."""
    from embodied_jepa.cloning import AIM_OFFSET_SEEDS

    seeds = tuple(int(s) for s in manifest["gates"]["D1"]["val_root_seeds"])
    if len(seeds) != 15 or len(set(seeds)) != 15:
        raise ContractError("D1-pipeline needs exactly 15 distinct val roots")
    if set(seeds) & COHORT_C:
        raise ContractError("a D1-pipeline seed is in the frozen cohort C")
    collector = _load("_collect_apple_wide", "scripts/collect_apple_wide.py")
    plan = {r["seed"]: r for r in collector.make_plan(collector.FROZEN_SEEDS)["roots"]}
    expected = sorted(
        s for s, r in plan.items() if r["split"] == "val" and s not in AIM_OFFSET_SEEDS
    )
    if sorted(seeds) != expected:
        raise ContractError(f"D1-pipeline seeds {sorted(seeds)} != surviving val roots {expected}")
    return {
        s: {"seed": s, "object_xy": plan[s]["object_xy"], "plate_xy": plan[s]["plate_xy"]}
        for s in seeds
    }


# ----- building controllers ------------------------------------------------------------------
def load_arm(label, config, store):
    """Rebuild an arm exactly as ``evaluate_policy.evaluate`` does, and cross-check its label."""
    import torch

    measure = _load("_measure", "scripts/measure_policy_offline_conditionals.py")
    checkpoint = CHECKPOINTS / f"{ARMS[label]}.pt"
    policy, source = measure.build_policy(checkpoint, config, store, DEVICE)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    recorded = (saved.get("metadata") or {}).get("arm")
    if recorded is not None and recorded != label:
        raise ContractError(f"{checkpoint.name} metadata arm {recorded!r} != {label!r}")
    return policy, source


def _robot():
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.simulation import MuJoCoSimulation

    return G1Embodiment(
        MuJoCoSimulation(
            object_kind="apple", container_kind="plate", width=IMAGE_SIZE, height=IMAGE_SIZE
        )
    )


def _close(robot):
    close = getattr(robot.sim, "close", None)
    if close is not None:
        close()


def run_one(make_controller, reset, configuration, *, lower, upper, trace, deadline_at):
    """One attempt, built exactly as ``evaluate_policy.evaluate`` builds one."""
    from embodied_jepa.task import AppleToPlateTask

    if time.monotonic() > deadline_at:
        raise GlobalCap("global wall cap reached before the next attempt")
    r = runner()
    robot = _robot()
    try:
        probe = robot.observe().images["onboard_rgb"]
        if probe.shape[1:3] != (IMAGE_SIZE, IMAGE_SIZE):
            raise ContractError(f"onboard render is {probe.shape[1:3]}, not {IMAGE_SIZE} px")
        robot.reset(**reset)
        scorer = AppleToPlateTask(robot)
        truth = robot.sim.task_truth()
        shadow = pd.ShadowExpert(truth)
        controller = make_controller(truth, robot, int(reset["seed"]))
        result = pd.run_diagnostic_attempt(
            controller,
            robot,
            scorer,
            configuration=configuration,
            clip=r.clip_to_configured_bounds,
            lower=lower,
            upper=upper,
            max_steps=MAX_STEPS,
            deadline_seconds=CONTROL_DEADLINE_SECONDS,
            shadow=shadow,
            trace=trace,
            attempt_wall_seconds=ATTEMPT_WALL_SECONDS,
        )
    finally:
        _close(robot)
    result["seed"] = int(reset["seed"])
    return result


# ----- gate arithmetic (pure; tested) --------------------------------------------------------
def grasp_resets(attempts) -> int:
    return sum(bool(a["grasp"]) for a in attempts)


def b1_verdict(oracle, hold, random_) -> dict:
    checks = {
        "scripted_oracle_grasp_ge_14": grasp_resets(oracle) >= 14,
        "scripted_oracle_success_ge_12": sum(bool(a["success"]) for a in oracle) >= 12,
        "hold_zero_grasp_and_success": not any(a["grasp"] or a["success"] for a in hold),
        "random_zero_grasp_and_success": not any(a["grasp"] or a["success"] for a in random_),
        "sixteen_each": len(oracle) == len(hold) == len(random_) == 16,
    }
    return {"checks": checks, "passed": all(checks.values())}


def b2_verdict(none_attempts, reference) -> dict:
    """``none`` must reproduce the TASK-056 A2 report: grasp seeds, terminations, step counts."""
    ref = {int(a["seed"]): a for a in reference["attempts"]}
    got = {int(a["seed"]): a for a in none_attempts}
    mismatches = []
    if set(ref) != set(got):
        mismatches.append({"seeds": [sorted(ref), sorted(got)]})
    for seed in sorted(set(ref) & set(got)):
        for field, theirs, mine in (
            (
                "termination_reason",
                ref[seed]["termination_reason"],
                got[seed]["termination_reason"],
            ),
            ("executed_steps", ref[seed]["executed_steps"], got[seed]["executed_steps"]),
            ("grasp", bool(ref[seed]["score"].get("grasp")), bool(got[seed]["grasp"])),
        ):
            if theirs != mine:
                mismatches.append({"seed": seed, "field": field, "reference": theirs, "run": mine})
    ref_grasp = sorted(s for s, a in ref.items() if a["score"].get("grasp"))
    run_grasp = sorted(s for s, a in got.items() if a["grasp"])
    return {
        "reference_grasp_seeds": ref_grasp,
        "run_grasp_seeds": run_grasp,
        "mismatches": mismatches,
        "passed": not mismatches and ref_grasp == run_grasp,
    }


def b3_verdict(full_attempts) -> dict:
    count = grasp_resets(full_attempts)
    return {
        "grasp_resets": count,
        "attempts": len(full_attempts),
        "passed": count >= 14 and len(full_attempts) == 16,
    }


def d1_pipeline_verdict(
    rows, *, uses_image, output_tolerance, state_tolerance, expected=15
) -> dict:
    """Arm fails iff ANY reset breaks (i) image identity, (ii) state/mask, or (iii) output."""
    failures = []
    for row in rows:
        if row is None:
            failures.append({"missing": True})
            continue
        if uses_image and not row["image_identical"]:
            failures.append({"seed": row["seed"], "check": "image"})
        if row["state_max_abs"] > state_tolerance or not row["mask_identical"]:
            failures.append({"seed": row["seed"], "check": "state"})
        output = row["output_max_abs"]
        if output is None or not np.isfinite(output) or output > output_tolerance:
            failures.append({"seed": row["seed"], "check": "output"})
    if len(rows) != expected:
        failures.append({"missing": True, "resets": len(rows)})
    return {"failed": bool(failures), "failures": failures}


def d1_grasp_count(attempts) -> int | None:
    """Seeds where the step-zero right_grasp > -0.5 (amendment 1). Missing counts as firing."""
    count = 0
    for a in attempts:
        first = a.get("first_command")
        if first is None or not np.isfinite(first[6]):
            count += 1
        elif first[6] > -0.5:
            count += 1
    return count + max(0, 16 - len(attempts))


def d1_contrast(attempts, thresholds) -> dict:
    """Reported, not decisional: per-dimension median |step-zero controller - shadow expert|."""
    rows = [
        (np.asarray(a["first_command"]), np.asarray(a["first_shadow_expert"]))
        for a in attempts
        if a.get("first_command") is not None and a.get("first_shadow_expert") is not None
    ]
    if not rows:
        return {"evaluable": False}
    diff = np.abs(np.array([c - e for c, e in rows]))
    median = np.median(diff, axis=0)
    out = {"evaluable": True, "seeds": len(rows), "per_dimension": {}}
    for i, name in enumerate(pd.FREE_NAMES):
        entry = {"median_abs": float(median[i])}
        key = f"right_{name}"
        if key in thresholds:
            entry["threshold"] = thresholds[key]
            entry["ratio"] = float(median[i] / thresholds[key])
        out["per_dimension"][name] = entry
    out["would_pass_frozen_rule"] = all(
        e["ratio"] <= 1.0 for e in out["per_dimension"].values() if "ratio" in e
    )
    return out


def d2_reading(attempts, tau) -> dict:
    firsts = []
    for a in attempts:
        departures = a["departures"]
        found = pd.first_departure_step(departures, tau)
        if found is None:
            defined = next((i for i, v in enumerate(departures) if v is None), len(departures))
            firsts.append({"seed": a["seed"], "first_departure_step": None, "censored_at": defined})
        else:
            firsts.append({"seed": a["seed"], "first_departure_step": found, "censored_at": None})
    values = [
        f["first_departure_step"] if f["censored_at"] is None else f["censored_at"] for f in firsts
    ]
    median = float(np.median(values)) if values else float("nan")
    reading = "gradual" if median >= 50 else "immediate" if median <= 5 else "inconclusive"
    return {
        "tau": tau,
        "median_first_departure_step": median,
        "reading": reading,
        "per_seed": firsts,
    }


def g_sub_verdict(scores, thresholds) -> dict:
    """Each candidate against ITS threshold; a missing candidate counts as not passing."""
    per = {}
    for candidate, tau in thresholds.items():
        score = scores.get(candidate)
        per[candidate] = {
            "grasp_resets": score,
            "threshold": tau,
            "passes": score is not None and score >= tau,
        }
    if set(scores) - set(thresholds):
        raise ContractError(f"non-candidate configurations scored: {set(scores) - set(thresholds)}")
    return {"per_candidate": per, "passed": any(v["passes"] for v in per.values())}


def decide(*, void, d1_failed, grasp_counts, g_sub_passed, exemption_spent) -> dict:
    """The amended decision table (protocol §5.1, §5.2, §7 and §13.6)."""
    if void:
        return {"outcome": "V", "clause_a": None}
    failed_arms = sum(bool(v) for v in d1_failed.values())
    fired_arms = sum(1 for v in grasp_counts.values() if v is None or v >= 8)
    d1_disjunct = bool(d1_failed[PRIMARY]) or failed_arms >= 3
    grasp_disjunct = fired_arms >= 3
    clause_a = d1_disjunct or grasp_disjunct
    if not g_sub_passed and clause_a:
        outcome = (
            "X_when_the_exemption_is_already_spent" if exemption_spent else "P_over_X_precedence"
        )
    elif not g_sub_passed:
        outcome = "X"
    elif clause_a:
        outcome = "P"
    else:
        outcome = "S"
    return {
        "outcome": outcome,
        "clause_a": clause_a,
        "d1_pipeline_disjunct": d1_disjunct,
        "d1_grasp_disjunct": grasp_disjunct,
        "d1_pipeline_failed_arms": failed_arms,
        "d1_grasp_fired_arms": fired_arms,
        "g_sub_passed": g_sub_passed,
        "exemption_spent_at_start": exemption_spent,
        "claims_the_exemption": outcome == "P_over_X_precedence",
    }


def past_close_fraction(attempt) -> float | None:
    """Share of an attempt's commands issued after the shadow expert's ``close`` phase."""
    rows = attempt.get("trace") or []
    if not rows:
        return None
    order = pd.SHADOW_EXPERT_PHASES
    past = [
        r["shadow_expert_phase"] in order
        and order.index(r["shadow_expert_phase"]) > 2
        or r["shadow_expert_phase"] == "done"
        for r in rows
    ]
    return float(np.mean(past))


# ----- D1-pipeline ---------------------------------------------------------------------------
def compare_reset(
    *,
    seed,
    live_image,
    stored_image,
    live_state,
    stored_state,
    live_mask,
    stored_mask,
    live_output,
    offline_output,
) -> dict:
    """One D1-pipeline row. The offline prediction is CLIPPED to [-1, 1] as ``act`` clips
    (amendment 1, rule (iii)); a non-finite difference is recorded as None, never NaN."""
    difference = np.abs(
        np.asarray(live_output, np.float64) - np.clip(np.asarray(offline_output, np.float64), -1, 1)
    ).max()
    return {
        "seed": int(seed),
        "image_identical": None
        if live_image is None
        else bool(np.array_equal(live_image, stored_image)),
        "state_max_abs": float(np.abs(np.asarray(live_state) - np.asarray(stored_state)).max()),
        "mask_identical": bool(np.array_equal(live_mask, stored_mask)),
        "output_max_abs": float(difference) if np.isfinite(difference) else None,
    }


def d1_pipeline(manifest, store, cameras, policies, *, resets=None, split="val", expected=15):
    """Zero executed commands: reset, observe, compare. ``resets`` defaults to the 15 val roots;
    ``--smoke`` passes train provenance roots (split "train") to show a sound pipeline passes."""
    from embodied_jepa.cloning import load_bc_split, precompute_features
    from embodied_jepa.policy import FREE_ACTION_INDICES

    measure = _load("_measure", "scripts/measure_policy_offline_conditionals.py")
    resets = val_root_resets(manifest) if resets is None else resets
    arrays, _rows, _counts, _base = load_bc_split(
        store, split, cameras, workers=8, limit=None, acknowledge=True
    )
    first = {}
    for index, episode in enumerate(arrays.episode_ids):
        parts = episode.split("-")
        if len(parts) == 2 and int(parts[1]) in resets:
            first[int(parts[1])] = int(arrays.offsets[index])
    if sorted(first) != sorted(resets):
        raise ContractError(f"a D1-pipeline root is missing from the decoded {split} split")
    order = sorted(resets)
    stored_rows = np.array([first[s] for s in order])
    observations = {}
    for seed in order:
        robot = _robot()
        try:
            probe = robot.observe().images["onboard_rgb"]
            if probe.shape[1:3] != (IMAGE_SIZE, IMAGE_SIZE):
                raise ContractError("render size")
            robot.reset(**resets[seed])
            observations[seed] = robot.observe()
        finally:
            _close(robot)
    gate = manifest["gates"]["D1"]
    report = {}
    for label, (policy, source) in policies.items():
        offline = measure.predictions(
            policy, arrays, stored_rows, precompute_features(source, arrays, stored_rows)
        )
        arm_rows = []
        for k, seed in enumerate(order):
            obs, row = observations[seed], stored_rows[k]
            live = np.asarray(policy.act(obs.images, obs.state), np.float32)
            arm_rows.append(
                compare_reset(
                    seed=seed,
                    live_image=obs.images["onboard_rgb"][0],
                    stored_image=arrays.frames["onboard_rgb"][row],
                    live_state=obs.robot_state[0],
                    stored_state=arrays.states[row],
                    live_mask=obs.state_mask[0],
                    stored_mask=arrays.mask[row],
                    live_output=live[list(FREE_ACTION_INDICES)],
                    offline_output=offline[k],
                )
            )
        report[label] = {
            "rows": arm_rows,
            **d1_pipeline_verdict(
                arm_rows,
                uses_image=bool(source.feature_dim),
                output_tolerance=gate["output_tolerance"],
                state_tolerance=gate["state_tolerance"],
                expected=expected,
            ),
        }
    return report


# ----- the run -------------------------------------------------------------------------------
def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n")


def _summary(attempts):
    return [
        {
            k: a[k]
            for k in (
                "seed",
                "termination_reason",
                "executed_steps",
                "grasp",
                "success",
                "shadow_expert_exhausted_at_step",
                "first_command",
                "first_shadow_expert",
                "clipped_commands",
            )
        }
        | {"score": a["score"]}
        for a in attempts
    ]


def execute_stages(report, *, attempts_for, controllers, d1_pipeline_fn, b2_reference):
    """The preregistered stage order (§13.6), stopping at the first voiding control.

    ``attempts_for(make, configuration, trace=, stage=)`` runs one configuration over the 16
    development seeds; ``controllers`` maps "policy" (a function of the arm label),
    "scripted_oracle", "hold" and "random" to controller factories. Returns the unsubstituted
    arm attempts, or raises ``VoidRun`` naming the control that failed.
    """
    policy = controllers["policy"]
    # 1. B3, first.
    full = attempts_for(policy(PRIMARY), "full", trace=True, stage="B3-A2-full")
    report["stages"]["B3"] = b3_verdict(full) | {"attempts": _summary(full)}
    if not report["stages"]["B3"]["passed"]:
        raise VoidRun("B3")
    # 2. D1-pipeline.
    report["stages"]["D1_pipeline"] = d1_pipeline_fn()
    # 3. B1.
    oracle = attempts_for(
        controllers["scripted_oracle"], "none", trace=False, stage="B1-scripted_oracle"
    )
    hold = attempts_for(controllers["hold"], "none", trace=False, stage="B1-hold")
    rand = attempts_for(controllers["random"], "none", trace=False, stage="B1-random")
    report["stages"]["B1"] = b1_verdict(oracle, hold, rand) | {
        "scripted_oracle": _summary(oracle),
        "hold": _summary(hold),
        "random": _summary(rand),
    }
    if not report["stages"]["B1"]["passed"]:
        raise VoidRun("B1")
    # 4. Arms unsubstituted; A2's run is B2.
    arms = {}
    for label in ARM_ORDER:
        arms[label] = attempts_for(policy(label), "none", trace=True, stage=f"none-{label}")
        if label == PRIMARY:
            report["stages"]["B2"] = b2_verdict(arms[label], b2_reference)
            if not report["stages"]["B2"]["passed"]:
                raise VoidRun("B2")
    report["stages"]["arms_none"] = {label: _summary(a) for label, a in arms.items()}
    # 5. D3.
    d3 = {}
    for candidate in pd.G_SUB_CANDIDATES:
        d3[candidate] = attempts_for(
            policy(PRIMARY), candidate, trace=True, stage=f"D3-A2-{candidate}"
        )
    report["stages"]["D3"] = {
        c: {
            "grasp_resets": grasp_resets(a),
            "full_successes": sum(bool(x["success"]) for x in a),
            "mean_past_close_fraction": _mean_or_none([past_close_fraction(x) for x in a]),
            "attempts": _summary(a),
        }
        for c, a in d3.items()
    }
    return arms


def _mean_or_none(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def evaluate_gates(report, manifest, arms):
    """Every gate and the decision, from a completed stage sequence."""
    tables = manifest["gates"]
    thresholds_d1 = manifest["frozen_D1_thresholds_three_times_table_A"]
    d1p = report["stages"]["D1_pipeline"]
    d1_failed = {label: d1p[label]["failed"] for label in ARMS}
    grasp_counts = {label: d1_grasp_count(arms[label]) for label in ARMS}
    gsub = g_sub_verdict(
        {c: report["stages"]["D3"][c]["grasp_resets"] for c in pd.G_SUB_CANDIDATES},
        tables["G_SUB"]["per_candidate_threshold"],
    )
    report["gates"] = {
        "D1_pipeline_failed": d1_failed,
        "D1_grasp_count": grasp_counts,
        "D1_grasp_signed_table": {
            label: {
                str(a["seed"]): None if a.get("first_command") is None else a["first_command"][6]
                for a in arms[label]
            }
            for label in ARMS
        },
        "D1_contrast": {label: d1_contrast(arms[label], thresholds_d1[label]) for label in ARMS},
        "D2": {
            label: d2_reading(arms[label], tables["D2"]["tau_per_arm"][label]) for label in ARMS
        },
        "G_SUB": gsub,
    }
    report["decision"] = decide(
        void=False,
        d1_failed=d1_failed,
        grasp_counts=grasp_counts,
        g_sub_passed=gsub["passed"],
        exemption_spent=manifest["precedence_rule_D1_over_G_SUB"]["exemption_spent"],
    )
    report["status"] = "completed"


def conduct(report, manifest, *, attempts_for, controllers, d1_pipeline_fn, b2_reference):
    """Stages, gates and the outcome, under amendment 2's stop rule.

    * A voiding CONTROL evaluated and failed (B1/B2/B3) is Outcome V, as in section 7.
    * A run that STOPS before every voiding control and gate is evaluated -- the global cap or
      any exception, including a crash or an interrupt -- is Outcome V with the stop reason.
      Abandonment does not fire and no gate passes or fails. The exception is re-raised after
      the outcome is recorded, so a crash still exits non-zero.
    * Only a run that completed every stage is evaluated; only there does the missing-values
      rule apply.
    """
    try:
        arms = execute_stages(
            report,
            attempts_for=attempts_for,
            controllers=controllers,
            d1_pipeline_fn=d1_pipeline_fn,
            b2_reference=b2_reference,
        )
    except VoidRun as error:
        report["decision"] = decide(
            void=True, d1_failed={}, grasp_counts={}, g_sub_passed=False, exemption_spent=None
        ) | {"void_reason": f"voiding control {error} evaluated and failed"}
        report["outcome"] = "V"
        report["status"] = f"void: voiding control {error} failed"
        return report
    except BaseException as error:
        stop(report, error)
        raise
    try:
        evaluate_gates(report, manifest, arms)
    except BaseException as error:
        stop(report, error, during="gate evaluation")
        raise
    report["outcome"] = report["decision"]["outcome"]
    return report


def stop(report, error, during="stages"):
    """Amendment 2: a run stopped before evaluation is VOID, never a bare 'stopped'.

    The stop reason is recorded BY THE RUNNER: the exception, its full traceback (archived in
    the report written with the partial results) and, for the global cap, the cap record. An
    interrupt (KeyboardInterrupt / signal) is labelled ``interrupted``: amendment 2 makes an
    agent-initiated interruption a process violation reported to the owner, not a valid void.
    """
    interrupted = isinstance(error, KeyboardInterrupt | SystemExit)
    kind = (
        "global_cap"
        if isinstance(error, GlobalCap)
        else ("interrupted" if interrupted else "exception")
    )
    reason = f"{type(error).__name__}: {error}"
    report.pop("gates", None)
    report["decision"] = decide(
        void=True, d1_failed={}, grasp_counts={}, g_sub_passed=False, exemption_spent=None
    ) | {
        "void_reason": f"stopped during {during} before every control and gate was evaluated: "
        f"{reason}"
    }
    report["outcome"] = "V"
    report["stop"] = {
        "kind": kind,
        "during": during,
        "reason": reason,
        "traceback": "".join(traceback.format_exception(error)),
        "valid_void": not interrupted,
        "note": "an interruption is a process violation reported to the owner, not a valid void"
        if interrupted
        else "stop recorded by the runner (amendment 2)",
    }
    report["status"] = (
        "void: INTERRUPTED - process violation, report to the owner (amendment 2)"
        if interrupted
        else "void: stopped before evaluation (amendment 2)"
    )


#: Load-bearing inputs amendment 1 does not hash; recorded in the report (review of PR #45).
RECORDED_INPUTS = (
    "benchmarks/manifests/apple-policy-diagnostics-v1.json",
    "docs/experiments/apple_policy_diagnostics_v1.md",
    "scripts/diagnose_policy.py",
    "scripts/evaluate_apple.py",
    "scripts/evaluate_policy.py",
    "scripts/measure_policy_offline_conditionals.py",
    "src/embodied_jepa/policy_diagnostics.py",
    "configs/g1_sim_action.json",
)


def run(output: Path):
    """The gated run. Refuses a dirty tree, a changed input, or an existing output."""
    from embodied_jepa.training import source_identity

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    started, deadline_at = time.perf_counter(), time.monotonic() + GLOBAL_WALL_SECONDS
    manifest = frozen_manifest()
    hashes = verify_hashes(manifest)
    source = source_identity()
    if source["dirty"] is not False:
        raise ContractError("the gated run requires a clean, committed tree")
    from embodied_jepa.config import ExperimentConfig
    from embodied_jepa.data import DatasetStore
    from embodied_jepa.policy import _check_pinned_bounds
    from embodied_jepa.world_model_v2 import _open

    r = runner()
    seeds = tuple(r.COHORT_D)
    resets = r.cohort_resets(seeds)  # the development whitelist; refuses cohort C
    config = ExperimentConfig.load(CONFIG)
    store = DatasetStore(config.dataset_root)
    lower = np.asarray(config.planner.lower_bounds, np.float32)
    upper = np.asarray(config.planner.upper_bounds, np.float32)
    _check_pinned_bounds(lower, upper)
    _cfg, _store, _settings, cameras = _open(CONFIG)
    report = {
        "protocol": PROTOCOL,
        "amendments": list(AMENDMENTS),
        "task": TASK,
        "source": source,
        "device": DEVICE,
        "cohort_D_seeds": list(seeds),
        "cohort_C_touched": False,
        "config": str(CONFIG.relative_to(ROOT)),
        "verified_hashes": hashes,
        "recorded_input_sha256": {p: sha256(ROOT / p) for p in RECORDED_INPUTS},
        "artifacts": str(output),
        "budget": {
            "global_wall_seconds": GLOBAL_WALL_SECONDS,
            "attempt_wall_seconds": ATTEMPT_WALL_SECONDS,
            "max_steps": MAX_STEPS,
        },
        "stages": {},
        "status": "running",
    }
    try:
        policies = {label: load_arm(label, config, store) for label in ARMS}
        report["policy_implementation_sha256"] = policies[PRIMARY][0].implementation_sha256

        def attempts_for(make, configuration, *, trace, stage):
            out = []
            for seed in seeds:
                attempt = run_one(
                    make,
                    resets[seed],
                    configuration,
                    lower=lower,
                    upper=upper,
                    trace=trace,
                    deadline_at=deadline_at,
                )
                out.append(attempt)
                _write(output / "attempts" / stage / f"{seed}.json", attempt)
            return out

        def policy(label):
            controller = pd.PolicyController(policies[label][0])
            return lambda truth, robot, seed: controller

        controllers = {
            "policy": policy,
            "scripted_oracle": lambda truth, robot, seed: pd.ScriptedOracleController(truth, robot),
            "hold": lambda truth, robot, seed: pd.hold_controller(),
            "random": lambda truth, robot, seed: pd.RandomController(seed, lower, upper),
        }
        conduct(
            report,
            manifest,
            attempts_for=attempts_for,
            controllers=controllers,
            d1_pipeline_fn=lambda: d1_pipeline(manifest, store, cameras, policies),
            b2_reference=json.loads(B2_REFERENCE.read_text()),
        )
    except GlobalCap:
        pass  # conduct() recorded outcome V with the stop reason (amendment 2)
    except BaseException as error:
        if report.get("outcome") != "V":  # e.g. a crash while loading the arms
            stop(report, error)
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        _write(output / "report.json", report)
    return report


#: Smoke reset: a surviving TRAIN root, not a cohort seed, so wiring can be exercised before the
#: pre-run review without touching the development cohort's outcome.
SMOKE_ROOT = 48000
#: The calibration's train provenance roots; disjoint from the 15 val gate roots.
SMOKE_D1_ROOTS = (48000, 48008, 48001, 48013, 48002, 48006, 48003, 48011)


def smoke(output: Path, steps: int = 20):
    """Every controller kind and both substitution branches for a few steps on a train reset."""
    from embodied_jepa.config import ExperimentConfig
    from embodied_jepa.data import DatasetStore

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest = frozen_manifest()
    verify_hashes(manifest)
    collector = _load("_collect_apple_wide", "scripts/collect_apple_wide.py")
    root = next(
        r for r in collector.make_plan(collector.FROZEN_SEEDS)["roots"] if r["seed"] == SMOKE_ROOT
    )
    if root["split"] != "train":
        raise ContractError("the smoke reset must be a train root")
    reset = {"seed": SMOKE_ROOT, "object_xy": root["object_xy"], "plate_xy": root["plate_xy"]}
    config = ExperimentConfig.load(CONFIG)
    store = DatasetStore(config.dataset_root)
    lower = np.asarray(config.planner.lower_bounds, np.float32)
    upper = np.asarray(config.planner.upper_bounds, np.float32)
    policy = pd.PolicyController(load_arm(PRIMARY, config, store)[0])
    kinds = {
        "A2-none": (lambda truth, robot, seed: policy, "none"),
        "A2-dx": (lambda truth, robot, seed: policy, "dx"),
        "A2-full": (lambda truth, robot, seed: policy, "full"),
        "oracle": (lambda truth, robot, seed: pd.ScriptedOracleController(truth, robot), "none"),
        "hold": (lambda truth, robot, seed: pd.hold_controller(), "none"),
        "random": (lambda truth, robot, seed: pd.RandomController(seed, lower, upper), "none"),
    }
    global MAX_STEPS
    saved, MAX_STEPS = MAX_STEPS, steps
    try:
        results = {
            name: run_one(
                make,
                reset,
                configuration,
                lower=lower,
                upper=upper,
                trace=True,
                deadline_at=time.monotonic() + 600,
            )
            for name, (make, configuration) in kinds.items()
        }
    finally:
        MAX_STEPS = saved
    from embodied_jepa.world_model_v2 import _open

    _cfg, _store, _settings, cameras = _open(CONFIG)
    collector_plan = {r["seed"]: r for r in collector.make_plan(collector.FROZEN_SEEDS)["roots"]}
    train_resets = {
        seed: {
            "seed": seed,
            "object_xy": collector_plan[seed]["object_xy"],
            "plate_xy": collector_plan[seed]["plate_xy"],
        }
        for seed in SMOKE_D1_ROOTS
    }
    if any(collector_plan[s]["split"] != "train" for s in SMOKE_D1_ROOTS):
        raise ContractError("the smoke D1-pipeline roots must be train roots")
    policies = {label: load_arm(label, config, store) for label in ARMS}
    pipeline = d1_pipeline(
        manifest,
        store,
        cameras,
        policies,
        resets=train_resets,
        split="train",
        expected=len(SMOKE_D1_ROOTS),
    )
    results["D1_pipeline_on_train_roots"] = pipeline
    _write(output / "smoke.json", results)
    failed = [label for label, v in pipeline.items() if v["failed"]]
    if failed:
        raise ContractError(f"D1-pipeline fails a sound pipeline on train roots for {failed}")
    return {
        k: (v["termination_reason"], v["executed_steps"])
        for k, v in results.items()
        if k != "D1_pipeline_on_train_roots"
    } | {"D1_pipeline_on_train_roots": "passed for every arm"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="wiring check on a train reset only")
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(smoke(args.output), indent=1))
        return 0
    report = run(args.output)
    print(json.dumps({"status": report["status"], "decision": report.get("decision")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
