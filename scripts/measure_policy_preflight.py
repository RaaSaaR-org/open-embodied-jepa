"""Pre-flight measurement for the Apple->Plate policy protocol v1 (TASK-056).

This script answers the one question that decides whether TASK-056 proceeds: **does the
frozen world-model encoder localize the apple well enough, on the phases where a reactive
controller needs it?** It trains nothing and it writes no checkpoint.

Two parts, both on the VAL split only:

* **Part A, the gated measurement.** The readout error of a *directly encoded* observation
  (no rollout, no predicted latent, horizon 0), broken out by the collector's phase index.
  Gates ``P1`` (close), ``P2`` (orient+descend) and ``P3`` (orient, absolute apple
  position). This measurement has never been made; the protocol's thresholds come from
  independent sources (the measured grasp/eject boundary of
  ``apple_wide_grasp_closure_results_v3.md`` and the collector's own injected mis-aim band),
  never from this quantity.
* **Part B, the re-derivation.** ``world_model_v2.window_metrics`` on the same windows the
  v4 evaluator used, which reproduces the four quantities the protocol lists as
  ``unverified_pending_P0`` (U1-U4). Those numbers are quoted in
  ``docs/experiments/apple_policy_v1.md`` from a git-ignored run artifact; this part puts
  them in a committed report so that no framing claim rests on an uncommitted file. Part B
  calls the existing evaluator rather than reimplementing it, so it is a re-derivation and
  not a second opinion.

**P1 is a binding abort.** If it fails, this script exits non-zero, the protocol's Outcome E
fires, and no arm is trained. "P1 failed but we proceeded" is not an available outcome.
Nothing here selects a replacement plan; the executing agent stops and reports.

Privileged simulator labels are read as *scoring references only*, exactly as the world-model
evaluator reads them: they are never a model input, a controller input or a planning cost.

Usage::

    uv run --no-sync python scripts/measure_policy_preflight.py \
        --config configs/apple_wm_v4_lewm.yaml \
        --checkpoint checkpoints/task054-wm-v4/leworldmodel_baseline.pt \
        --output outputs/apple-policy-v1/preflight.json \
        --acknowledge-privileged-training-labels
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa.contracts import ContractError  # noqa: E402
from embodied_jepa.world_model_v2 import (  # noqa: E402
    HORIZONS,
    PHASE_CLOSE,
    PHASE_DESCEND,
    PHASE_LIFT,
    PHASE_ORIENT,
    _environment,
    _median,
    _open,
    _robot_state,
    _write_json,
    images,
    load_split,
    readout_rows,
    window_metrics,
    window_starts,
)

PROTOCOL = "apple_policy_v1"
TASK = "TASK-056"

# Frozen thresholds. Their provenance is in docs/experiments/apple_policy_v1.md section 5.1
# and in benchmarks/manifests/apple-policy-v1.json; neither is derived from this measurement.
GATES = {
    "P1_close_palm_apple_median_m": 0.015,
    "P2_approach_palm_apple_median_m": 0.015,
    "P3_orient_apple_position_median_m": 0.020,
    # Every gate also has to beat the no-vision control by this factor. A readout you can
    # compute equally well from someone ELSE'S FRAME is reading the prior, not the frame.
    "max_ratio_to_shuffled_frame_control": 0.7,
}
# Closed-form median error of a predictor that always emits the reset centre and never looks at
# anything: wide_reset jitters object_xy by uniform(+-0.03) per axis, so the error is the norm of
# a uniform square. 2.394 cm. P2 (was 2.5 cm) and P3 (was 3.0 cm) were passable by this model,
# which is why both were tightened to 1.5 cm = 0.63x this baseline.
BLIND_PRIOR_MEDIAN_M = 0.023937
# Upper edge of the Outcome E "near miss" band. Reading rule only: it selects the FOLLOW-UP
# task, never whether this task stops.
E_NEAR_MAX_M = 0.020
# The collector's phase order (scripted.apple_collector_policy). world_model_v2 names the
# first five; this protocol gates on the first three.
PHASE_NAMES = {
    PHASE_ORIENT: "orient",
    PHASE_DESCEND: "descend",
    PHASE_CLOSE: "close",
    PHASE_LIFT: "lift",
    4: "transfer",
    5: "release_high",
    6: "lower_open",
    7: "retreat",
}
# Roots whose scripted aim was deliberately offset 1.5-3.0 cm from the apple
# (scripts/collect_apple_wide.py: AIM_OFFSET_EVERY, FROZEN_SEEDS). Their frames are excluded
# from the pre-flight cohort for the same reason they are excluded from the BC targets: the
# script they follow is aiming somewhere the apple is not.
FROZEN_SEEDS = tuple(range(48000, 48200))
AIM_OFFSET_EVERY = 5
AIM_OFFSET_SEEDS = frozenset(
    seed
    for index, seed in enumerate(FROZEN_SEEDS)
    if index % AIM_OFFSET_EVERY == AIM_OFFSET_EVERY - 1
)


def is_excluded_episode(episode_id: str) -> bool:
    """True for branch episodes and for aim-offset roots.

    Root ids are ``wide-<seed>``; branch ids are ``wide-<seed>-b<n>-<kind>``.
    """
    parts = episode_id.split("-")
    if len(parts) != 2:
        return True  # any branch episode
    try:
        seed = int(parts[1])
    except ValueError as error:
        raise ContractError(f"unrecognized episode id {episode_id!r}") from error
    return seed in AIM_OFFSET_SEEDS


def surviving_rows(arrays) -> np.ndarray:
    """Observation rows of the surviving (non-branch, non-aim-offset) val roots."""
    keep = np.zeros(len(arrays.states), dtype=bool)
    for index, episode_id in enumerate(arrays.episode_ids):
        if is_excluded_episode(episode_id):
            continue
        start = int(arrays.offsets[index])
        keep[start : start + int(arrays.lengths[index])] = True
    return np.flatnonzero(keep)


def per_phase_errors(model, arrays, state_schema, rows) -> dict:
    """Directly encoded (horizon 0) readout error, broken out by collector phase.

    No rollout and no predicted latent: this is the encoder reading the frame in front of it,
    which is the only thing a reactive controller consumes.
    """
    dropped = arrays.targets["apple_dropped"][rows][:, 0] > 0.5
    valid_rows = rows[~dropped]
    readouts = readout_rows(model, arrays, valid_rows, state_schema)
    truth = arrays.targets
    palm_apple = np.linalg.norm(
        readouts["palm_minus_apple"] - truth["palm_minus_apple"][valid_rows], axis=1
    )
    apple_position = np.linalg.norm(
        readouts["apple_position"] - truth["apple_position"][valid_rows], axis=1
    )
    phase = arrays.phase[valid_rows]
    result = {
        "rows_considered": int(len(rows)),
        "rows_dropped_apple_off_table": int(dropped.sum()),
        "rows_scored": int(len(valid_rows)),
        "horizon": 0,
        "latent": "directly encoded observation; no rollout",
        "by_phase": {},
    }
    for index in sorted(set(int(value) for value in phase if value >= 0)):
        mask = phase == index
        result["by_phase"][PHASE_NAMES.get(index, str(index))] = {
            "phase_index": index,
            "rows": int(mask.sum()),
            "palm_apple_median_m": _median(palm_apple[mask]),
            "apple_position_median_m": _median(apple_position[mask]),
        }
    approach = np.isin(phase, (PHASE_ORIENT, PHASE_DESCEND))
    result["cohorts"] = {
        "close": {
            "rows": int((phase == PHASE_CLOSE).sum()),
            "palm_apple_median_m": _median(palm_apple[phase == PHASE_CLOSE]),
        },
        "approach_orient_descend": {
            "rows": int(approach.sum()),
            "palm_apple_median_m": _median(palm_apple[approach]),
        },
        "orient": {
            "rows": int((phase == PHASE_ORIENT).sum()),
            "apple_position_median_m": _median(apple_position[phase == PHASE_ORIENT]),
        },
        "grasp_close_and_lift": {
            "rows": int(np.isin(phase, (PHASE_CLOSE, PHASE_LIFT)).sum()),
            "palm_apple_median_m": _median(palm_apple[np.isin(phase, (PHASE_CLOSE, PHASE_LIFT))]),
        },
    }
    return result


def shuffled_frame_control(model, arrays, state_schema, rows, seed=20560, chunk=256):
    """P0a: the same readouts with the IMAGE taken from a different episode, state held.

    The direct analogue of ``held_lift_shuffled_auroc``. Frames are paired within the same
    collector phase, across different episodes, so the control keeps the phase prior (which a
    blind head can learn from proprioception) and removes only the apple's actual pixels.
    """
    episode_of = np.zeros(len(arrays.states), np.int64)
    for index in range(len(arrays.episode_ids)):
        start = int(arrays.offsets[index])
        episode_of[start : start + int(arrays.lengths[index])] = index
    rng = np.random.default_rng(seed)
    image_rows = rows.copy()
    phase = arrays.phase[rows]
    for value in sorted(set(int(v) for v in phase if v >= 0)):
        local = np.flatnonzero(phase == value)
        if len(local) < 2:
            continue
        candidates = rows[local]
        owners = episode_of[candidates]
        for offset, position in enumerate(local):
            other = candidates[owners != owners[offset]]
            if len(other):
                image_rows[position] = other[rng.integers(len(other))]
    readouts = []
    for start in range(0, len(rows), chunk):
        picture = image_rows[start : start + chunk]
        state = rows[start : start + chunk]
        latent = model.encode(images(arrays, picture), _robot_state(arrays, state, state_schema))
        readouts.append(model.readout(latent))
    joined = {n: np.concatenate([r[n] for r in readouts]) for n in readouts[0]}
    paired = int((image_rows != rows).sum())
    return joined, image_rows, paired


def preflight_band(p1: dict) -> str:
    """Which Outcome-E sub-case a P1 failure belongs to. Never affects whether the run stops.

    A RATIO-only failure is the shortcut case: the encoder is reading the prior, not the apple,
    and no amount of resolution fixes that. It must not be labelled ``E-near`` (whose follow-up
    is resolution and camera placement) just because its absolute value happens to be small.
    """
    if not p1["ratio_passed"]:
        return "E-prior"
    value = p1["value"]
    return "E-near" if value is not None and value < E_NEAR_MAX_M else "E-clear"


def control_errors(arrays, readouts, rows, paired) -> dict:
    """P0a cohort medians, on exactly the cohorts P1/P2/P3 are scored on."""
    dropped = arrays.targets["apple_dropped"][rows][:, 0] > 0.5
    valid_rows = rows[~dropped]
    keep = ~dropped
    truth = arrays.targets
    palm = np.linalg.norm(
        readouts["palm_minus_apple"][keep] - truth["palm_minus_apple"][valid_rows], axis=1
    )
    apple = np.linalg.norm(
        readouts["apple_position"][keep] - truth["apple_position"][valid_rows], axis=1
    )
    phase = arrays.phase[valid_rows]
    approach = np.isin(phase, (PHASE_ORIENT, PHASE_DESCEND))
    return {
        "method": (
            "P0a: identical readouts with the IMAGE drawn from a different episode at the same "
            "collector phase, proprioception held. The analogue of held_lift_shuffled_auroc."
        ),
        "frames_repaired_with_a_foreign_image": paired,
        "blind_prior_median_m": BLIND_PRIOR_MEDIAN_M,
        "blind_prior_note": (
            "P0b: closed-form median error of always emitting the reset centre under "
            "wide_reset's uniform(+-0.03) object jitter. P2 (was 2.5 cm) and P3 (was 3.0 cm) "
            "were passable by this model; both were tightened to 1.5 cm."
        ),
        "cohorts": {
            "P1_close_palm_apple": _median(palm[phase == PHASE_CLOSE]),
            "P2_approach_palm_apple": _median(palm[approach]),
            "P3_orient_apple_position": _median(apple[phase == PHASE_ORIENT]),
        },
    }


def evaluate_preflight_gates(measurement: dict, gates: dict) -> dict:
    """P1/P2/P3. A gate that cannot be evaluated counts as FAILED."""
    cohorts = measurement["cohorts"]
    rows = (
        (
            "P1_close_palm_apple",
            cohorts["close"]["palm_apple_median_m"],
            "P1_close_palm_apple_median_m",
        ),
        (
            "P2_approach_palm_apple",
            cohorts["approach_orient_descend"]["palm_apple_median_m"],
            "P2_approach_palm_apple_median_m",
        ),
        (
            "P3_orient_apple_position",
            cohorts["orient"]["apple_position_median_m"],
            "P3_orient_apple_position_median_m",
        ),
    )
    control = measurement.get("no_vision_control", {}).get("cohorts", {})
    ratio_max = gates["max_ratio_to_shuffled_frame_control"]
    result = {}
    for name, value, key in rows:
        threshold = gates[key]
        blind = control.get(name)
        ratio = None if not blind or value is None or blind <= 0 else value / blind
        result[name] = {
            "value": value,
            "op": "<=",
            "threshold": threshold,
            "absolute_passed": bool(value is not None and value <= threshold),
            # P0a: a readout computable from someone else's FRAME is reading the prior, not
            # the frame. A missing control counts as FAILED, never as a free pass.
            "shuffled_frame_control": blind,
            "ratio_to_control": ratio,
            "ratio_threshold": ratio_max,
            "ratio_passed": bool(ratio is not None and ratio <= ratio_max),
        }
        result[name]["passed"] = bool(
            result[name]["absolute_passed"] and result[name]["ratio_passed"]
        )
    result["all_passed"] = all(item["passed"] for item in result.values() if isinstance(item, dict))
    return result


def rederive_unverified(model, arrays, state_schema) -> dict:
    """U1-U4, by calling the v4 evaluator's own window metrics on its own window cohort."""
    windows = window_starts(arrays, max(HORIZONS), stride=4)
    measured = window_metrics(model, arrays, windows, state_schema)
    derived = {}
    for horizon in HORIZONS:
        entry = measured[str(horizon)]
        rollout = entry["palm_apple_moving_median_m"]
        persistence = entry["palm_apple_moving_persistence_median_m"]
        derived[str(horizon)] = {
            "U1_palm_apple_grasp_cohort_median_m": entry["palm_apple_grasp_cohort_median_m"],
            "U2_rollout_over_persistence_moving": (
                None if not persistence else rollout / persistence
            ),
            "U3_held_lift_shuffled_auroc": entry["held_lift_shuffled_auroc"],
            "U3_held_lift_auroc": entry["held_lift_auroc"],
            "U4_palm_apple_all_valid_windows_median_m": entry["palm_apple_median_m"],
        }
    return {
        "note": (
            "Re-derived by calling world_model_v2.window_metrics on window_starts(arrays, 16, "
            "stride=4), the same code path and the same cohort the v4 evaluator used. This is a "
            "re-derivation, not a second opinion."
        ),
        "windows": int(len(windows)),
        "by_horizon": derived,
        "raw_window_metrics": measured,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-episodes", type=int, help="smoke subsets only")
    parser.add_argument("--acknowledge-privileged-training-labels", action="store_true")
    parser.add_argument(
        "--skip-rederivation",
        action="store_true",
        help="smoke only: run part A without the U1-U4 re-derivation",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing pre-flight report")
    if args.acknowledge_privileged_training_labels is not True:
        raise ContractError(
            "the pre-flight scores against privileged simulator labels; acknowledge that use"
        )

    from embodied_jepa.config import MODELS
    from embodied_jepa.training import source_identity

    # ``_write_json`` writes through a sibling .tmp file and does not create directories.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config, store, settings, cameras = _open(args.config)
    report = {
        "format_version": 1,
        "protocol": PROTOCOL,
        "task": TASK,
        "status": "running",
        "part_a": "directly encoded (horizon 0) readout error by collector phase; gates P1/P2/P3",
        "part_b": "re-derivation of the protocol's unverified_pending_P0 items U1-U4",
        "trains_nothing": True,
        "config": str(args.config.resolve()),
        "config_resolved": config.resolved,
        "backend": config.backend,
        "checkpoint": str(args.checkpoint.resolve()),
        "cameras": list(cameras),
        "device": args.device,
        "split": "val",
        "gates_frozen_in": "benchmarks/manifests/apple-policy-v1.json",
        "thresholds": dict(GATES),
        "threshold_provenance": (
            "P1 = 1.5 cm is CARRIED from v3/v4's own G1 threshold. It is corroborated, not "
            "derived, by the grasp/eject boundary: grasp-closure v3's 24 ceiling holds sat at "
            "close-phase apple xy 0.49-1.35 cm. NOTE the corrected margin: that document's "
            "'Corrected by this verification' section retracts the 2.45-9.28 cm ejection range "
            "for the GATED failures (they were 2.70-19.10 cm; 2.45-9.12 cm are the 16 forensic "
            "TRAIN tuning closes) and records that SIX of v2's 16 gated closes crossed the "
            "1.5 cm measure, the lowest at 1.61 cm. So the gap around P1 is 1.35 -> 1.61 cm. "
            "P2 and P3 = 1.5 cm are TIGHTENED from 2.5 and 3.0 cm, which sat at or above the "
            "2.394 cm no-vision prior and were therefore passable by a model that never reads "
            "the image. No threshold is derived from the quantity being measured."
        ),
        "arguments": {
            "limit_episodes": args.limit_episodes,
            "skip_rederivation": bool(args.skip_rederivation),
            "workers": args.workers,
        },
        "privileged_label_use": (
            "scoring references only; never a model input, controller input or cost"
        ),
        "source": source_identity(),
        "environment": _environment(),
        "test_episodes_decoded": 0,
    }
    _write_json(args.output, report)

    try:
        arrays = load_split(
            store,
            "val",
            cameras,
            workers=args.workers,
            limit=args.limit_episodes,
            acknowledge_privileged_training_labels=True,
        )
        model = MODELS.create(
            config.backend,
            state_schema=store.state_schema,
            device=args.device,
            seed=0,
            config=settings,
        )
        model.load(args.checkpoint)
        report["model_implementation_sha256"] = model.implementation_sha256
        report["model_parameters"] = int(sum(p.numel() for p in model.parameters()))

        rows = surviving_rows(arrays)
        surviving_ids = [e for e in arrays.episode_ids if not is_excluded_episode(e)]
        report["cohort"] = {
            "val_episodes_decoded": len(arrays.episode_ids),
            "surviving_root_episodes": len(surviving_ids),
            "surviving_episode_ids": sorted(surviving_ids),
            "surviving_observation_rows": int(len(rows)),
            "excluded": "branch episodes and aim-offset roots (index % 5 == 4 over 48000-48199)",
        }
        if args.limit_episodes is None and len(surviving_ids) != 15:
            raise ContractError(
                f"expected 15 surviving val roots, found {len(surviving_ids)}: the pre-flight "
                "cohort does not match the preregistered exclusion table"
            )
        measurement = per_phase_errors(model, arrays, store.state_schema, rows)
        blind, image_rows, paired = shuffled_frame_control(model, arrays, store.state_schema, rows)
        if paired < 0.95 * len(rows):
            raise ContractError(
                f"the no-vision control re-paired only {paired} of {len(rows)} frames; a control "
                "that mostly keeps each frame's own image does not bound the shortcut"
            )
        measurement["no_vision_control"] = control_errors(arrays, blind, rows, paired)
        report["measurement"] = measurement
        report["gates"] = evaluate_preflight_gates(measurement, GATES)
        if not args.skip_rederivation:
            report["rederivation"] = rederive_unverified(model, arrays, store.state_schema)
        report["status"] = "completed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["elapsed_seconds"] = time.perf_counter() - started
        _write_json(args.output, report)
        raise

    report["elapsed_seconds"] = time.perf_counter() - started
    _write_json(args.output, report)

    gates = report["gates"]
    print(json.dumps({k: v for k, v in gates.items()}, indent=2))
    # Reported whatever happens: "1.7 cm at 0.4x the control" and "1.7 cm at 0.95x" are very
    # different results and only the first is worth a follow-up.
    print(
        "\nno-vision control (P0a shuffled frame / P0b analytic prior "
        f"{BLIND_PRIOR_MEDIAN_M} m):\n"
        + json.dumps(report["measurement"]["no_vision_control"]["cohorts"], indent=2)
    )
    if not gates["P1_close_palm_apple"]["passed"]:
        p1 = gates["P1_close_palm_apple"]
        value, ratio = p1["value"], p1["ratio_to_control"]
        # The band affects ONLY what the next task is. It never permits continuing.
        band = preflight_band(p1)
        print(
            f"\nP1 FAILED ({band}). value={value} ratio_to_shuffled_frame_control={ratio}. "
            "Protocol outcome E: the task stops before any training, in every sub-case. "
            "E-prior (the ratio to the no-vision control failed, whatever the absolute value) "
            "means the encoder is reading the prior, not the apple, and points the follow-up at "
            "the information-ceiling probe; E-near ([1.5, 2.0) cm with the ratio passed) points "
            "it at resolution, camera placement and close-phase coverage; E-clear (>= 2.0 cm "
            "with the ratio passed) points it at the information-ceiling probe. Report the "
            "value AND the control ratio to the coordinator, and do not select the follow-up "
            "here.",
            file=sys.stderr,
        )
        return 2
    if not gates["P2_approach_palm_apple"]["passed"]:
        print(
            "\nP2 FAILED. Recorded in pre_run_changes. It drops no arm and blocks no gate; "
            "Outcome D, if reached, is reported on PARTIAL EVIDENCE.",
            file=sys.stderr,
        )
    if not gates["P3_orient_apple_position"]["passed"]:
        print(
            "\nP3 FAILED. Arm A4 does not run. Record this as a pre-run protocol change with its "
            "reason and its P3 value; outcome D, if reached, is reported on PARTIAL EVIDENCE.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
