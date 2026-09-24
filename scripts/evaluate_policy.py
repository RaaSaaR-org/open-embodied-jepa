"""Closed-loop policy evaluation on the DEVELOPMENT cohort only (TASK-056, stage 3).

This is the **reduced** runner. It does exactly what the protocol's development pre-check
needs -- instantiate a trained policy, step the embodiment, score with the frozen task, count
grasps and successes -- and nothing else. Deliberately **not** built here, each requiring its
own authorization:

* **cohort C (seeds 45300-45339).** This runner REFUSES those seeds, structurally, and a test
  asserts it. Opening the frozen cohort is a separate authorization.
* **arm A4**, the readout-driven scripted controller. Deferred, *not dropped*: A4 is what
  distinguishes "perception is inadequate in the loop" (Outcome D) from "cloning is the
  failing component" (Outcome C), so if the learned arms die on development it becomes the
  next thing built rather than the thing abandoned.
* replay capture, candidate ranking, and everything else the pre-check does not need.

Two constraints are load-bearing and are enforced here rather than trusted:

**Actions are clipped to the PROTOCOL'S CONFIGURED BOUNDS, not the contract's.**
``ClonedPolicy.act`` clips to [-1, 1] because ``validate_actions`` demands it, but the frozen
bounds in ``configs/apple_wm_v4.yaml`` are +-0.5 on the right arm. Executing a policy's raw
output would command twice the delta every prior generation operated under -- silently, since
the action is still contract-valid. Every command is clipped to the configured bounds, the
pinned components are checked against what the policy module pins, and the count of clipped
commands is reported so a policy living on the bound is visible rather than invisible.

**``wide_reset`` is IMPORTED from the collection/evaluation source, never reimplemented.**
That is the one tripwire hole a parser cannot close: a reimplementation of
``default_rng(seed).uniform(...)`` references nothing and would silently produce a different
cohort. The import is pinned by a test against the committed generator.

Simulator truth reaches only the scorer. The policy sees images and proprioception.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa.contracts import ContractError, validate_actions  # noqa: E402
from embodied_jepa.policy import PINNED_ACTION_VALUES, _check_pinned_bounds  # noqa: E402

PROTOCOL = "apple_policy_v1"
TASK = "TASK-056"

#: Development cohort: already consumed by the prior ceilings, never gating, free to re-run.
COHORT_D = tuple(range(45000, 45008)) + tuple(range(45100, 45108))
#: The frozen gating cohort. This runner refuses it; opening it is a separate authorization.
COHORT_C = tuple(range(45300, 45340))
#: Onboard render size of the corpus the policies were trained on (collect_apple_wide.py).
IMAGE_SIZE = 112


def load_wide_reset():
    """The committed reset rule, imported rather than reimplemented.

    ``scripts/evaluate_apple.py`` is not an importable module, so it is loaded by path -- the
    same way ``tests/test_apple_wide_collection.py`` loads it to pin the two copies equal.
    Reimplementing the arithmetic here is the one failure a parser cannot catch, which is why
    this is an import and why a test pins the result against the committed generator.
    """
    spec = importlib.util.spec_from_file_location(
        "_evaluate_apple_reset", ROOT / "scripts" / "evaluate_apple.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.wide_reset


def cohort_resets(seeds):
    """Reset coordinates for development seeds, from the committed generator."""
    generator = load_wide_reset()
    # Coerce BEFORE the membership test. "45300" is not equal to 45300, so testing membership
    # on the raw values lets a string seed slip past the refusal and then be coerced into a
    # frozen-cohort reset. The refusal is described as structural; this makes it so.
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ContractError(
            "no seeds requested. An empty run would report dead_on_development on zero "
            "attempts, and the protocol's stop rule keys on exactly that flag."
        )
    forbidden = sorted(set(seeds) & set(COHORT_C))
    if forbidden:
        raise ContractError(
            f"seeds {forbidden} belong to the FROZEN gating cohort C. This runner evaluates "
            "the development cohort only; opening C is a separate authorization and its "
            "resets must come from the stored manifest values, not from this generator."
        )
    # F5: a whitelist, not a blacklist of C. Any other seed would otherwise be run and then
    # filed under "D_development_never_gating", which the report hard-codes.
    outside = sorted(set(seeds) - set(COHORT_D))
    if outside:
        raise ContractError(
            f"seeds {outside} are not in the development cohort. This runner evaluates "
            "cohort D only; every other cohort needs its own authorization."
        )
    return {seed: generator(seed) for seed in seeds}


def frozen_flag_for(checkpoint):
    """Whether the encoder of this checkpoint's arm was trained frozen.

    ``cloning.train`` sets ``frozen = (encoder != "finetune")`` and saves encoder weights only
    when the source is trainable. So the flag follows the **absence** of saved weights:
    A1/A2 save none and were frozen; A3 saves them and was not.

    This is a named function rather than an inline expression because inverting it made every
    encoder arm fail to load -- three of the four, including the primary -- and the only arm
    smoked was the one that still worked. A function can be enumerated over all four arm
    shapes without a checkpoint or a simulator, which is what ``tests/test_policy.py`` does.
    """
    return checkpoint.get("feature_source_weights") is None


def clip_to_configured_bounds(action, lower, upper):
    """Clip a policy command to the protocol's frozen bounds and verify the pinned parts."""
    action = np.asarray(action, np.float32)
    if action.shape != (14,):
        raise ContractError("a policy command must be a 14-vector")
    for index, value in PINNED_ACTION_VALUES.items():
        if not np.isclose(action[index], value):
            raise ContractError(
                f"policy emitted {action[index]} for pinned component {index}; the policy "
                "module pins it and the configured bounds freeze it"
            )
    clipped = np.clip(action, lower, upper).astype(np.float32)
    return clipped, bool(np.any(clipped != action))


def run_attempt(policy, robot, scorer, *, lower, upper, max_steps, deadline_seconds):
    """One closed-loop episode. Returns the scorer's final reading plus per-step accounting."""
    reason, score, clipped_commands, control_times = "step_limit", {}, 0, []
    executed = 0
    try:
        for _ in range(max_steps):
            began = time.perf_counter()
            observation = robot.observe()
            raw = policy.act(observation.images, observation.state)
            command, was_clipped = clip_to_configured_bounds(raw, lower, upper)
            clipped_commands += int(was_clipped)
            validate_actions(command[None, None, None], ndim=4)
            # The embodiment's UNCHANGED feasibility projection, exactly as the collector's
            # commands were projected. A policy does not get a different actuation path.
            projection = robot.project_candidates(command[None, None, None])
            if not bool(projection.feasible[0, 0]):
                reason = "infeasible_command"
                break
            result = robot.execute(projection.actions[0, 0, 0])
            control_times.append(time.perf_counter() - began)
            if result.applied_action is None:
                # execute()'s reject() always sets status="stopped" and puts the diagnostic in
                # `reason`, so recording status alone loses every distinction.
                reason = result.reason or result.status
                break
            executed += 1
            score = scorer.evaluate()
            if score.get("success", False):
                reason = "success"
                break
            if control_times[-1] > deadline_seconds:
                reason = "deadline_miss"
                break
    finally:
        # Every exit path stops the robot, including an exception. G1Embodiment raises
        # ContractError("measured joint velocity limit exceeded") from inside project_candidates,
        # and without this the simulator is left actuating and the whole run dies with no report.
        robot.stop(reason)
    return {
        "termination_reason": reason,
        "executed_steps": executed,
        "clipped_commands": clipped_commands,
        "median_control_seconds": float(np.median(control_times)) if control_times else None,
        "max_control_seconds": float(np.max(control_times)) if control_times else None,
        "score": score,
    }


def evaluate(config_path, checkpoint, *, arm, seeds, output, device, max_steps, deadline):
    """Run one arm over the development cohort and write a report."""
    from embodied_jepa.config import MODELS, POLICIES, ExperimentConfig  # noqa: F401
    from embodied_jepa.data import DatasetStore
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.policy import (
        ClonedPolicy,
        FrozenEncoder,
        NoEncoder,
    )
    from embodied_jepa.simulation import MuJoCoSimulation
    from embodied_jepa.task import AppleToPlateTask
    from embodied_jepa.training import source_identity

    output = Path(output)
    if output.exists():
        raise FileExistsError("refusing to overwrite an existing evaluation report")
    output.parent.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig.load(config_path)
    store = DatasetStore(config.dataset_root)
    resets = cohort_resets(seeds)
    lower = np.asarray(config.planner.lower_bounds, np.float32)
    upper = np.asarray(config.planner.upper_bounds, np.float32)
    # The manifest names this as half the blocking requirement: the CONFIG must pin what the
    # policy module pins. Checking only the emitted action would let a widened config through.
    _check_pinned_bounds(lower, upper)

    report = {
        "format_version": 1,
        "protocol": PROTOCOL,
        "task": TASK,
        "stage": "development_precheck",
        "arm": arm,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "cohort": "D_development_never_gating",
        "cohort_seeds": [int(s) for s in seeds],
        "cohort_C_refused": True,
        "resets": {str(k): v for k, v in resets.items()},
        "configured_bounds": {"lower": lower.tolist(), "upper": upper.tolist()},
        "device": device,
        "max_steps": max_steps,
        "source": source_identity(),
        "attempts": [],
    }

    # Rebuild the policy's feature source from the checkpoint's own provenance, then let
    # ClonedPolicy.load verify it -- a mismatched arm is refused rather than silently run.
    saved = __import__("torch").load(checkpoint, map_location="cpu", weights_only=True)
    provenance = saved.get("feature_source") or {}
    if provenance.get("kind") == "none":
        source = NoEncoder()
    else:
        model = MODELS.create(
            config.backend,
            state_schema=store.state_schema,
            device=device,
            seed=0,
            config=config.model_settings,
        )
        # frozen=False DELIBERATELY for the probe, and it is load-bearing rather than
        # arbitrary. FrozenEncoder.__init__ is NOT a pure accessor when frozen=True: it calls
        # model.eval() and requires_grad_(False) on every parameter of the SHARED model. A
        # throwaway frozen instance built merely to read a digest therefore permanently freezes
        # the arm that is about to be restored as trainable, and A3's optimizer state then
        # fails to load. frozen=False constructs without mutating.
        #
        # A non-mutating digest helper in policy.py would be cleaner, and is NOT done here
        # because policy.py's implementation_sha256 is enforced by every trained checkpoint --
        # touching that file invalidates all four arms. Verified, not assumed. Recorded as owed.
        if (
            provenance.get("model_weights_sha256")
            != FrozenEncoder(model, frozen=False).weights_sha256()
        ):
            model.load(config.checkpoint)
        source = FrozenEncoder(model, frozen=frozen_flag_for(saved))
    policy = ClonedPolicy(store.state_schema, source, device=device, seed=0)
    policy.load(checkpoint)
    report["feature_source"] = source.provenance()
    # --arm is a free-form label written into the report. Mislabelling arms is precisely what
    # G2 and G3 exist to detect, so it is cross-checked against the checkpoint's own metadata.
    recorded = (saved.get("metadata") or {}).get("arm")
    if recorded is not None and recorded != arm:
        raise ContractError(
            f"--arm {arm!r} disagrees with the checkpoint's own metadata {recorded!r}"
        )
    report["checkpoint_arm_metadata"] = recorded
    report["policy_implementation_sha256"] = policy.implementation_sha256

    started = time.perf_counter()
    grasps = successes = 0
    for seed in seeds:
        robot = G1Embodiment(
            MuJoCoSimulation(
                object_kind="apple",
                container_kind="plate",
                width=IMAGE_SIZE,
                height=IMAGE_SIZE,
            )
        )
        # evaluate_apple.wide_reset's dict already carries "seed" (collect_apple_wide's does
        # not -- the test that pins the two equal compares COORDINATES, not the dict). Passing
        # it through is correct; passing seed= separately is a TypeError, which is how this was
        # found.
        robot.reset(**resets[int(seed)])
        scorer = AppleToPlateTask(robot)
        attempt = run_attempt(
            policy,
            robot,
            scorer,
            lower=lower,
            upper=upper,
            max_steps=max_steps,
            deadline_seconds=deadline,
        )
        attempt["seed"] = int(seed)
        grasps += int(bool(attempt["score"].get("grasp")))
        successes += int(bool(attempt["score"].get("success")))
        report["attempts"].append(attempt)
        robot.close() if hasattr(robot, "close") else None

    controls = [
        a["median_control_seconds"] for a in report["attempts"] if a["median_control_seconds"]
    ]
    report["result"] = {
        "attempts": len(seeds),
        "g7_reference_only": {
            "median_control_seconds": float(np.median(controls)) if controls else None,
            "gate_threshold_not_enforced_here": 0.1,
            "note": "G7 gates cohort C; this is the same quantity measured on development.",
        },
        "grasp_resets": grasps,
        "full_successes": successes,
        # The protocol's stop rule: an arm reaching grasp on 0/16 does not open cohort C.
        "dead_on_development": grasps == 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["status"] = "completed"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--max-steps", type=int, default=1000)
    # NOT the preregistered G7 threshold. G7 is a GATE on cohort C (median control <= 0.1 s,
    # zero deadline misses); this is a liveness guard for a non-gating development run, set
    # generously so a slow first step cannot end an attempt. The report records the median and
    # max control time so G7's quantity is measurable here even though it is not enforced.
    parser.add_argument("--deadline-seconds", type=float, default=5.0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=list(COHORT_D),
        help="development seeds only; cohort C is refused",
    )
    args = parser.parse_args()
    report = evaluate(
        args.config,
        args.checkpoint,
        arm=args.arm,
        seeds=tuple(args.seeds),
        output=args.output,
        device=args.device,
        max_steps=args.max_steps,
        deadline=args.deadline_seconds,
    )
    print(json.dumps(report["result"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
