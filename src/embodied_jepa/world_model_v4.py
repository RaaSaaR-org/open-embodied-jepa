"""World model v4 (TASK-054): redesign the action-conditioned prediction step.

TASK-052's four v3 arms all failed their preregistered gates, and the merged
encoder/rollout decomposition said *which* term failed. On the identical moving
validation windows the one-camera encoders improved (a directly encoded target frame
reads the palm-apple offset to 2.59 cm in arm B and 2.47 cm in arm D, against 3.26 cm
for v2), while the error the *rollout* adds on top of that -- the rollout excess --
roughly doubled to tripled, 0.39 cm in v2 against 1.06 cm (B) and 0.83 cm (D). The
error moved out of the encoder and into the action-conditioned prediction step, and
G7a (siblings' own actions beat swapped ones) failed on arms A and B.

So this protocol changes the prediction step and nothing else. The encoder, the camera
set (onboard only), the readout heads, the corpus, the splits, the budget and every
carried-over threshold are arm B's. What changes per arm is one shared, backend-agnostic
option of ``models/base.py``, each off by default so an earlier model keeps its exact
predictor and its exact loss:

- ``action_chunk``: condition each step jointly on a chunk of future actions;
- ``multistep_tail_weight``: move the multistep loss's effort towards the late steps;
- ``predictor_step_embedding``: make the rollout horizon-conditioned rather than one
  shared single-step module applied autoregressively.

What this module adds over ``world_model_v3``:

- ``decomposition_metrics``: the encoder/rollout split, promoted from the descriptive
  TASK-052 script into the gate report, so the term this task attacks is *gated*
  (G9) on the same cohort and in the same artifact as G1.
- ``evaluate_gates``: v3's thirteen gates verbatim, plus G9.

Everything else -- the train/val/test discipline, the privileged-label policy, the
opaque latents, the persistence/shuffled/sibling controls, the ranking gates -- is v2's
and v3's, unchanged. Nothing here is a closed-loop or learned-control result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from embodied_jepa.contracts import ContractError
from embodied_jepa.world_model_v2 import (
    GATE_HORIZON,
    HORIZONS,
    MOVING_THRESHOLD_M,
    _median,
    _window_actions,
    evaluate,
    predicted_readouts,
    readout_rows,
    train,
    window_starts,
)
from embodied_jepa.world_model_v3 import evaluate_gates as evaluate_v3_gates
from embodied_jepa.world_model_v3 import ranking_metrics

PROTOCOL = "apple_world_model_v4"


def decompose(model, arrays, windows, state_schema, horizon=GATE_HORIZON, per_window=None):
    """Split the palm-apple gate error into its encoder and rollout terms.

    On the gate's own moving cohort, in metres:

    - ``encoded_start``   readout of the directly encoded START observation against the
                          truth at the start frame: the encoder's own error;
    - ``encoded_target``  readout of the directly encoded TARGET observation against the
                          truth at the target frame: what a *perfect* predictor could
                          achieve with this encoder and head;
    - ``rollout``         the gate value G1 itself, recomputed here;
    - ``rollout_excess``  rollout minus encoded_target: the prediction step's own
                          contribution, and the term TASK-054 attacks;
    - ``persistence``     readout of the encoded START against the truth at the TARGET,
                          the denominator of G2a, recomputed here.

    ``rollout`` and ``persistence`` are deliberately redundant with
    ``window_metrics``: the gate function asserts they agree bit-for-bit, which is what
    proves this measurement runs on the gate's own cohort.

    This mirrors ``scripts/decompose_wm_v3_readout.py``, which produced the TASK-052
    numbers as a descriptive side measurement. Here it is part of the evaluation, so
    the decomposition and the gates always come from one checkpoint, one cohort and one
    report. ``per_window`` optionally collects the per-window error vectors, the moving
    mask and each window's episode index, so two arms can be compared by a paired
    bootstrap -- including one clustered by episode.
    """
    starts = arrays.offsets[windows[:, 0]] + windows[:, 1]
    targets = starts + horizon
    true = arrays.targets
    dropped = true["apple_dropped"][:, 0] > 0.5
    valid = ~dropped[starts] & ~dropped[targets]
    pma_start = true["palm_minus_apple"][starts]
    pma_target = true["palm_minus_apple"][targets]
    displacement = np.linalg.norm(pma_target - pma_start, axis=1)
    moving = valid & (displacement >= MOVING_THRESHOLD_M)

    # The rollout is measured over the evaluator's own action horizon and then read at
    # step ``horizon``, exactly as ``window_metrics`` does. Rolling out only ``horizon``
    # steps would not be the same computation under ``action_chunk`` > 1, where the
    # chunk of step k reaches past step ``horizon`` and would be zero-padded instead.
    actions = _window_actions(arrays, windows, max(HORIZONS))
    rollout = np.linalg.norm(
        predicted_readouts(model, arrays, starts, actions, state_schema)["palm_minus_apple"][
            :, horizon - 1
        ]
        - pma_target,
        axis=1,
    )
    at_start = readout_rows(model, arrays, starts, state_schema)["palm_minus_apple"]
    at_target = readout_rows(model, arrays, targets, state_schema)["palm_minus_apple"]
    encoded_start = np.linalg.norm(at_start - pma_start, axis=1)
    encoded_target = np.linalg.norm(at_target - pma_target, axis=1)
    persistence = np.linalg.norm(at_start - pma_target, axis=1)

    if per_window is not None:
        per_window.update(
            moving=moving,
            episode=windows[:, 0],
            rollout=rollout,
            encoded_start=encoded_start,
            encoded_target=encoded_target,
            persistence=persistence,
        )
    rollout_median = _median(rollout[moving])
    encoded_target_median = _median(encoded_target[moving])
    if rollout_median is None or encoded_target_median is None:
        raise ContractError("the decomposition needs a nonempty moving cohort")
    return {
        "horizon": horizon,
        "moving_threshold_m": MOVING_THRESHOLD_M,
        "valid_windows": int(valid.sum()),
        "moving_windows": int(moving.sum()),
        "true_displacement_median_m": _median(displacement[moving]),
        "encoded_start_median_m": _median(encoded_start[moving]),
        "encoded_target_median_m": encoded_target_median,
        "rollout_median_m": rollout_median,
        "persistence_median_m": _median(persistence[moving]),
        "rollout_excess_median_m": rollout_median - encoded_target_median,
        "encoder_share_of_rollout": encoded_target_median / rollout_median,
    }


def decomposition_metrics(model, arrays, state_schema, per_window=None):
    windows = window_starts(arrays, max(HORIZONS), stride=4)
    return {"decomposition": decompose(model, arrays, windows, state_schema, per_window=per_window)}


def extra_metrics(per_window=None):
    """v3's ranking measurements plus the gated encoder/rollout decomposition."""

    def measure(model, arrays, state_schema):
        return ranking_metrics(model, arrays, state_schema) | decomposition_metrics(
            model, arrays, state_schema, per_window=per_window
        )

    return measure


def evaluate_gates(metrics, gates):
    """v3's thirteen gates verbatim, plus G9 on the rollout excess.

    G9 is the term the decomposition named: the gate value minus what a *perfect*
    predictor could reach with this encoder and head, on the identical windows.

    Two properties of G9 are stated here because they bound what a pass would mean.

    - G9 is **not** trivially passable by a do-nothing predictor. An identity rollout
      makes the rollout error equal the persistence error, so its excess is
      ``persistence - encoded_target``; on the v3 arms that is 1.57 cm (B) and 1.51 cm
      (D), far above the threshold rather than below it.
    - G9 alone still does **not** establish that the actions are used. A predictor that
      regressed towards the cohort's centre could in principle add little error while
      conditioning on nothing. G9 is therefore read only together with G2a, G2b and
      G7a, which such a predictor fails, and never on its own. The preregistration
      records that joint reading, and that G9 can pass while G1 fails: G9 bounds the
      term this task attacks, not the absolute accuracy G1 asks for.
    """
    result = dict(evaluate_v3_gates(metrics, gates))
    result.pop("all_passed")
    decomposition = metrics["decomposition"]
    w = metrics["windows"][str(GATE_HORIZON)]
    # The decomposition must be measured on the gate's own cohort, or G9 is not
    # comparable with G1 and G2a. Two values are computed twice, once by
    # ``window_metrics`` and once by ``decompose``; they must agree exactly.
    for measured, gated, label in (
        (decomposition["rollout_median_m"], w["palm_apple_moving_median_m"], "rollout"),
        (
            decomposition["persistence_median_m"],
            w["palm_apple_moving_persistence_median_m"],
            "persistence",
        ),
        (decomposition["moving_windows"], w["palm_apple_moving_windows"], "moving windows"),
    ):
        if measured != gated:
            raise ContractError(
                f"the decomposition's {label} ({measured}) is not the gate's ({gated}): "
                "the two measurements did not run on the same cohort"
            )
    value = decomposition["rollout_excess_median_m"]
    threshold = gates["G9_rollout_excess_h8_median_m"]
    result["G9_rollout_excess_h8"] = {
        "value": value,
        "op": "<=",
        "threshold": threshold,
        "passed": bool(value is not None and value <= threshold),
    }
    result["all_passed"] = all(item["passed"] for item in result.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--protocol-manifest", type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "mps"), default="mps")
        command.add_argument("--workers", type=int, default=8)
        command.add_argument("--limit-episodes", type=int, help="smoke subsets only")
        command.add_argument("--acknowledge-privileged-training-labels", action="store_true")
        command.add_argument(
            "--require-clean", action="store_true", help="refuse a dirty or unversioned checkout"
        )
    train_parser = commands.choices["train"]
    train_parser.add_argument("--output", type=Path)
    train_parser.add_argument("--smoke-steps", type=int, help="smoke only: override steps")
    train_parser.add_argument(
        "--smoke-max-seconds", type=float, help="smoke only: override max_seconds"
    )
    train_parser.add_argument(
        "--smoke-validation-every", type=int, help="smoke only: override validation_every"
    )
    evaluate_parser = commands.choices["evaluate"]
    evaluate_parser.add_argument("--checkpoint", type=Path)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument(
        "--dump-errors",
        type=Path,
        help="also write the per-window decomposition error vectors, the moving mask "
        "and each window's episode index to this .npz, for the paired bootstrap",
    )
    args = parser.parse_args()
    manifest = json.loads(args.protocol_manifest.read_text())
    frozen = manifest["frozen"]
    common = dict(
        device=args.device,
        workers=args.workers,
        limit_episodes=args.limit_episodes,
        protocol=PROTOCOL,
        acknowledge_privileged_training_labels=args.acknowledge_privileged_training_labels,
    )
    if args.command == "train":
        budget = dict(frozen["training"])
        for key in ("steps", "max_seconds", "validation_every"):
            override = getattr(args, f"smoke_{key}")
            if override is not None:
                budget[key] = override
        report = train(
            args.config,
            output=args.output,
            steps=budget["steps"],
            batch_size=budget["batch_size"],
            horizon=budget["horizon"],
            seed=budget["seed"],
            max_seconds=budget["max_seconds"],
            validation_every=budget["validation_every"],
            selection_count=budget["selection_windows"],
            final_lr_fraction=budget["final_lr_fraction"],
            eligibility=frozen["selection_eligibility"],
            require_clean=args.require_clean,
            **common,
        )
        summary = {k: report.get(k) for k in ("status", "completed_steps", "best_step")}
    else:
        if args.dump_errors is not None and args.dump_errors.exists():
            raise FileExistsError("refusing to overwrite a per-window error dump")
        per_window = {} if args.dump_errors is not None else None
        report = evaluate(
            args.config,
            checkpoint=args.checkpoint,
            output=args.output,
            gates=frozen["gates"],
            require_clean=args.require_clean,
            gate_function=evaluate_gates,
            extra_metrics=extra_metrics(per_window),
            **common,
        )
        if per_window is not None:
            args.dump_errors.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.dump_errors, **per_window)
        summary = {"all_passed": report["gates"]["all_passed"]} | {
            k: v["passed"] for k, v in report["gates"].items() if k != "all_passed"
        }
    print(json.dumps(summary, indent=2))
    return 0 if report.get("status", "completed") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
