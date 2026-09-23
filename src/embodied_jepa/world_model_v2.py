"""World model v2 (TASK-050): cameras + proprioception + shared readout heads.

Trains any registered ``VisualModel`` backend on a sealed corpus with privileged
per-step label sidecars (``embodied_jepa.training_labels``) and evaluates the
preregistered OFFLINE gates of ``docs/experiments/apple_world_model_v2.md``.

TASK-052 generalized the runner from one camera to the model's configured camera
list and parameterized the protocol name, the gate function and the extra
measurements, so ``world_model_v3`` reuses it instead of copying it. The frozen
TASK-050 numbers belong to revision ``3b6af0b``; this file has moved on.

Rules this module enforces:

- Updates use the ``train`` split only; checkpoint selection and the offline gates
  use ``val``; ``test`` (and ``holdout``) episodes are never decoded.
- Privileged labels are loaded only with an explicit acknowledgement and are used
  only as readout-head training targets and as scoring references. Model inputs are
  the configured cameras, the proprioception and the executed actions.
- The backend comes from an ``ExperimentConfig``: swapping it is the one-line
  ``world_model.backend`` change. No function here branches on the backend.
- Evaluators obtain physical quantities only through ``model.readout`` and latent
  statistics only through ``model.latent_statistics``; latents stay opaque.

Nothing here is a closed-loop or learned-control result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embodied_jepa import readout_labels
from embodied_jepa.contracts import ContractError, RobotState, SequenceBatch, StateSchema
from embodied_jepa.data import DatasetStore
from embodied_jepa.training import BudgetReached, RunClock, json_hash, peak_rss_bytes

PROTOCOL = "apple_world_model_v2"
HORIZONS = (1, 4, 8, 16)
GATE_HORIZON = 8
SIBLING_HORIZONS = (8, 16, 32)
SIBLING_GATE_HORIZON = 16
OUTCOME_HORIZON = 64
MOVING_THRESHOLD_M = 0.01
SIBLING_DIVERGENCE_M = 0.01
APPROACH_OFFSET_M = np.array([-0.015, 0.0, 0.13], np.float32)  # object_ceiling approach
COST_FLOOR_M = 0.02
PHASE_ORIENT, PHASE_DESCEND, PHASE_CLOSE, PHASE_LIFT, PHASE_TRANSFER = range(5)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----- decoding ------------------------------------------------------------------------
def _reader(root, manifest, state_schema):
    """An episode reader over a store the parent process already verified.

    ``DatasetStore.read_episode`` re-checks the episode file's recorded hash, so a
    worker does not need to re-hash the whole corpus.
    """
    store = object.__new__(DatasetStore)
    store.root = Path(root)
    store.manifest = manifest
    store.state_schema = state_schema
    return store


_WORKER = {}


def _init_worker(root, manifest, schema_fields):
    _WORKER["store"] = _reader(root, manifest, StateSchema(**schema_fields))


def _decode(job):
    episode_id, cameras = job
    episode = _WORKER["store"].read_episode(episode_id)
    for camera in cameras:
        if camera not in episode.observations:
            raise ContractError(f"episode {episode_id} lacks camera {camera!r}")
    return episode_id, {
        "frames": {
            camera: np.ascontiguousarray(episode.observations[camera]) for camera in cameras
        },
        "states": np.asarray(episode.robot_states, np.float32),
        "mask": np.asarray(episode.state_mask, bool),
        "actions": np.asarray(episode.actions, np.float32),
        "timestamps": np.asarray(episode.timestamps, np.float64),
    }


@dataclass
class EpisodeArrays:
    """Contiguous per-split arrays; observation rows of episode e are offset[e]..+length."""

    episode_ids: tuple[str, ...]
    offsets: np.ndarray  # [E] first observation row
    lengths: np.ndarray  # [E] observations (T+1)
    frames: dict  # camera name -> [N,H,W,3] uint8 (one entry per configured camera)
    states: np.ndarray  # [N,S]
    mask: np.ndarray  # [N,S]
    actions: np.ndarray  # [N,14]; the final row of each episode is a zero placeholder
    timestamps: np.ndarray  # [N]
    targets: dict  # name -> [N,width] (privileged, training targets / scoring only)
    phase: np.ndarray  # [N] collector phase index (privileged; scoring cohorts only)
    rows: dict  # episode_id -> manifest row


def load_split(
    store,
    split,
    cameras,
    *,
    workers=8,
    limit=None,
    check_budget=lambda: None,
    acknowledge_privileged_training_labels=False,
):
    """Decode the configured cameras + proprioception + actions + readout targets.

    ``cameras`` is one camera name or a sequence of them; every decoded episode must
    carry each one.
    """
    cameras = (cameras,) if isinstance(cameras, str) else tuple(cameras)
    if not cameras or len(set(cameras)) != len(cameras):
        raise ContractError("cameras must be a nonempty sequence of distinct names")
    if split not in ("train", "val"):
        raise ContractError("world model v2 decodes only the train and val splits")
    if acknowledge_privileged_training_labels is not True:
        raise ContractError("readout targets are privileged; acknowledge training-label use")
    splits = store.manifest["splits"]
    excluded = set(splits.get("test", [])) | set(splits.get("holdout", []))
    ids = list(splits[split])
    if limit is not None:
        ids = ids[:limit]
    if excluded & set(ids):
        raise ContractError(f"{split} overlaps test/holdout")
    rows = {row["episode_id"]: row for row in store.manifest["episodes"]}
    jobs = [(name, cameras) for name in ids]
    decoded = {}
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        workers,
        initializer=_init_worker,
        initargs=(str(store.root), store.manifest, store.manifest["state_schema"]),
    ) as pool:
        for name, arrays in pool.imap_unordered(_decode, jobs, chunksize=4):
            decoded[name] = arrays
            check_budget()
    lengths = np.array([len(decoded[name]["states"]) for name in ids], np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)
    total = int(lengths.sum())
    first = decoded[ids[0]]
    frames = {
        camera: np.empty((total, *first["frames"][camera].shape[1:]), np.uint8)
        for camera in cameras
    }
    states = np.empty((total, first["states"].shape[1]), np.float32)
    mask = np.empty((total, first["mask"].shape[1]), bool)
    actions = np.zeros((total, 14), np.float32)
    timestamps = np.empty(total, np.float64)
    targets = {}
    phase = np.full(total, -1, np.int16)
    rest_cache = {}
    for index, name in enumerate(ids):
        start, count = int(offsets[index]), int(lengths[index])
        arrays = decoded.pop(name)
        for camera in cameras:
            frames[camera][start : start + count] = arrays["frames"][camera]
        states[start : start + count] = arrays["states"]
        mask[start : start + count] = arrays["mask"]
        actions[start : start + count - 1] = arrays["actions"]
        timestamps[start : start + count] = arrays["timestamps"]
        row = rows[name]
        root = row["metadata"]["root_episode_id"]
        if root not in rest_cache:
            root_labels = readout_labels.load_privileged(
                store.root, rows[root], acknowledge_privileged_training_labels=True
            )
            rest_cache[root] = float(root_labels["privileged__apple_position_world"][0, 2])
        labels = readout_labels.load_privileged(
            store.root, row, acknowledge_privileged_training_labels=True
        )
        episode_targets = readout_labels.targets(labels, rest_cache[root])
        for key, value in episode_targets.items():
            if len(value) != count:
                raise ContractError(f"label rows of {name} disagree with its observations")
            if key not in targets:
                targets[key] = np.empty((total, value.shape[1]), np.float32)
            targets[key][start : start + count] = value
        phases = np.asarray(labels["collector__phase_index"], np.int16)
        if not 0 < len(phases) <= count:
            raise ContractError(f"collector phases of {name} disagree with its observations")
        # Collector phases are recorded per command (T rows); the end row repeats.
        phase[start : start + len(phases)] = phases
        phase[start + len(phases) : start + count] = phases[-1]
    return EpisodeArrays(
        tuple(ids),
        offsets,
        lengths,
        frames,
        states,
        mask,
        actions,
        timestamps,
        targets,
        phase,
        {name: rows[name] for name in ids},
    )


# ----- batches -------------------------------------------------------------------------
def window_starts(arrays, horizon, stride=1):
    """(episode index, start) of every window of ``horizon`` transitions."""
    result = []
    for index, length in enumerate(arrays.lengths):
        transitions = int(length) - 1
        result.extend((index, start) for start in range(0, transitions - horizon + 1, stride))
    return np.asarray(result, np.int64).reshape(-1, 2)


def images(arrays, rows):
    """Every configured camera's frames at observation ``rows`` (any index shape)."""
    return {camera: value[rows] for camera, value in arrays.frames.items()}


def batch(arrays, windows, horizon, state_schema):
    """Canonical SequenceBatch + readout targets ``[B,H+1,width]`` for window starts."""
    rows = arrays.offsets[windows[:, 0]] + windows[:, 1]
    frame_index = rows[:, None] + np.arange(horizon + 1)[None]
    action_index = rows[:, None] + np.arange(horizon)[None]
    terminated = np.zeros((len(windows), horizon), bool)
    terminated[:, -1] = windows[:, 1] + horizon == arrays.lengths[windows[:, 0]] - 1
    sequence = SequenceBatch(
        observations=images(arrays, frame_index),
        robot_states=arrays.states[frame_index],
        state_mask=arrays.mask[frame_index],
        actions=arrays.actions[action_index],
        timestamps=arrays.timestamps[frame_index],
        terminated=terminated,
        episode_ids=tuple(arrays.episode_ids[i] for i in windows[:, 0]),
        state_schema=state_schema,
    )
    targets = {name: value[frame_index] for name, value in arrays.targets.items()}
    return sequence, targets


# ----- metrics -------------------------------------------------------------------------
def auroc(scores, labels):
    """Mann-Whitney AUROC with average ranks for ties; None without both classes."""
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels, bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return None
    ranks = _ranks(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _ranks(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def spearman(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _median(values):
    values = np.asarray(values, np.float64)
    return float(np.median(values)) if len(values) else None


def _robot_state(arrays, rows, state_schema):
    return RobotState(arrays.states[rows], arrays.mask[rows], arrays.timestamps[rows], state_schema)


def encode_rows(model, arrays, rows, state_schema, chunk=256):
    """Encoded latents (opaque) of observation rows, in chunks."""
    latents = []
    for start in range(0, len(rows), chunk):
        part = rows[start : start + chunk]
        latents.append(model.encode(images(arrays, part), _robot_state(arrays, part, state_schema)))
    return latents


def readout_rows(model, arrays, rows, state_schema, chunk=256):
    """Readouts of encoded observation rows: ``{name: [N,width]}``."""
    parts = [model.readout(z) for z in encode_rows(model, arrays, rows, state_schema, chunk=chunk)]
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


def predicted_readouts(model, arrays, starts, actions, state_schema, chunk=256):
    """Readouts of predicted latents for start rows and ``actions [N,H,14]``."""
    parts = []
    for start in range(0, len(starts), chunk):
        rows = starts[start : start + chunk]
        z = model.encode(images(arrays, rows), _robot_state(arrays, rows, state_schema))
        prediction = model.predict(z, np.ascontiguousarray(actions[start : start + chunk, None]))
        parts.append({k: v[:, 0] for k, v in model.readout(prediction).items()})
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


def _window_actions(arrays, windows, horizon):
    rows = arrays.offsets[windows[:, 0]] + windows[:, 1]
    return arrays.actions[rows[:, None] + np.arange(horizon)[None]]


def shuffled_pairing(windows):
    """Fixed derangement: window i takes the actions of window (i + N/2) mod N, moved
    forward until it comes from a different episode."""
    count = len(windows)
    partner = np.empty(count, np.int64)
    for i in range(count):
        j = (i + count // 2) % count
        steps = 0
        while windows[j, 0] == windows[i, 0]:
            j = (j + 1) % count
            steps += 1
            if steps > count:
                raise ContractError("shuffled control needs windows from two episodes")
        partner[i] = j
    return partner


def window_metrics(model, arrays, windows, state_schema, horizons=HORIZONS):
    """Predicted-latent readout accuracy and controls on a fixed window cohort."""
    longest = max(horizons)
    starts = arrays.offsets[windows[:, 0]] + windows[:, 1]
    actions = _window_actions(arrays, windows, longest)
    partner = shuffled_pairing(windows)
    true = arrays.targets
    predicted = predicted_readouts(model, arrays, starts, actions, state_schema)
    shuffled = predicted_readouts(model, arrays, starts, actions[partner], state_schema)
    zero = predicted_readouts(model, arrays, starts, np.zeros_like(actions), state_schema)
    encoded_start = readout_rows(model, arrays, starts, state_schema)
    dropped_true = true["apple_dropped"][:, 0] > 0.5
    result = {"windows": int(len(windows))}
    for h in horizons:
        target = starts + h
        # Scoring cohorts exclude windows whose apple is off the table at the start or
        # at the target (regression targets are undefined there; see models/readout).
        valid = ~dropped_true[starts] & ~dropped_true[target]
        pma_true = true["palm_minus_apple"][target]
        displacement = np.linalg.norm(pma_true - true["palm_minus_apple"][starts], axis=1)
        moving = valid & (displacement >= MOVING_THRESHOLD_M)

        def error(readouts, name, h=h, target=target):
            return np.linalg.norm(readouts[name][:, h - 1] - true[name][target], axis=1)

        def probability(readouts, name, h=h):
            return readouts[name][:, h - 1, 0]

        palm = error(predicted, "palm_minus_apple")
        persistence = np.linalg.norm(encoded_start["palm_minus_apple"] - pma_true, axis=1)
        phase = arrays.phase[target]
        grasp = valid & np.isin(phase, (PHASE_CLOSE, PHASE_LIFT))
        lift = valid & (phase == PHASE_LIFT)
        approach = valid & np.isin(phase, (PHASE_ORIENT, PHASE_DESCEND))
        transfer = valid & (phase == PHASE_TRANSFER)
        cost_true = np.linalg.norm(pma_true - APPROACH_OFFSET_M, axis=1)
        cost_pred = np.linalg.norm(
            predicted["palm_minus_apple"][:, h - 1] - APPROACH_OFFSET_M, axis=1
        )
        cost_rows = approach & (cost_true >= COST_FLOOR_M)
        log_ratio = np.abs(np.log(np.maximum(cost_pred, 1e-4) / np.maximum(cost_true, 1e-4)))
        plate_true = np.linalg.norm(true["apple_minus_plate"][target][:, :2], axis=1)
        plate_pred = np.linalg.norm(predicted["apple_minus_plate"][:, h - 1, :2], axis=1)
        plate_rows = transfer & (plate_true >= COST_FLOOR_M)
        plate_log_ratio = np.abs(
            np.log(np.maximum(plate_pred, 1e-4) / np.maximum(plate_true, 1e-4))
        )
        held_true = true["apple_held"][target][:, 0] > 0.5
        contact_true = true["hand_contact"][target][:, 0] > 0.5
        result[str(h)] = {
            "valid_windows": int(valid.sum()),
            "palm_apple_median_m": _median(palm[valid]),
            "palm_apple_moving_windows": int(moving.sum()),
            "palm_apple_moving_median_m": _median(palm[moving]),
            "palm_apple_moving_persistence_median_m": _median(persistence[moving]),
            "palm_apple_moving_shuffled_median_m": _median(
                error(shuffled, "palm_minus_apple")[moving]
            ),
            "palm_apple_moving_zero_action_median_m": _median(
                error(zero, "palm_minus_apple")[moving]
            ),
            "palm_apple_moving_true_displacement_median_m": _median(displacement[moving]),
            "palm_apple_grasp_cohort_median_m": _median(palm[grasp]),
            "apple_plate_median_m": _median(error(predicted, "apple_minus_plate")[valid]),
            "apple_height_grasp_windows": int(grasp.sum()),
            "apple_height_grasp_median_abs_m": _median(error(predicted, "apple_height")[grasp]),
            "held_lift_positives": int(held_true[lift].sum()),
            "held_lift_negatives": int((~held_true[lift]).sum()),
            "held_lift_auroc": auroc(probability(predicted, "apple_held")[lift], held_true[lift]),
            "held_lift_shuffled_auroc": auroc(
                probability(shuffled, "apple_held")[lift], held_true[lift]
            ),
            "held_grasp_auroc": auroc(
                probability(predicted, "apple_held")[grasp], held_true[grasp]
            ),
            "contact_grasp_auroc": auroc(
                probability(predicted, "hand_contact")[grasp], contact_true[grasp]
            ),
            "dropped_all_windows_auroc": auroc(
                probability(predicted, "apple_dropped"), dropped_true[target]
            ),
            "approach_cost_windows": int(cost_rows.sum()),
            "approach_cost_median_abs_log_ratio": _median(log_ratio[cost_rows]),
            "approach_cost_median_ratio": (
                _median(cost_pred[cost_rows]) / _median(cost_true[cost_rows])
                if cost_rows.any()
                else None
            ),
            "approach_cost_spearman": spearman(cost_pred[cost_rows], cost_true[cost_rows]),
            "transport_cost_windows": int(plate_rows.sum()),
            "transport_cost_median_abs_log_ratio": _median(plate_log_ratio[plate_rows]),
        }
    return result


def sibling_groups(arrays):
    """``root -> [(episode, start row, stop row, grasp label)]`` sharing one pre-grasp
    state: the root's continuation from its branch frame plus each of its branches."""
    groups = {}
    for index, name in enumerate(arrays.episode_ids):
        metadata = arrays.rows[name]["metadata"]
        root = metadata["root_episode_id"]
        offset = int(arrays.offsets[index])
        start = offset + (metadata["branch_frame_in_root"] if metadata["kind"] == "root" else 0)
        stop = offset + int(arrays.lengths[index])  # exclusive observation row
        grasp = bool(metadata["privileged_outcome_labels"]["stages"]["grasp"])
        groups.setdefault(root, []).append((name, start, stop, grasp))
    return groups


def sibling_metrics(model, arrays, state_schema):
    """Matched grasp-phase siblings from one pre-grasp state (root continuation and its
    branches): does the model's prediction follow the executed actions?"""
    groups = sibling_groups(arrays)
    longest = max(max(SIBLING_HORIZONS), OUTCOME_HORIZON)
    pairs = {str(h): {"own": [], "swap": [], "pred": [], "true": []} for h in SIBLING_HORIZONS}
    outcome_scores, outcome_labels = [], []
    start_mismatch = 0.0
    true = arrays.targets["palm_minus_apple"]
    dropped = arrays.targets["apple_dropped"][:, 0] > 0.5
    for root in sorted(groups):
        members = groups[root]
        if len(members) < 2:
            continue
        anchor = next((m for m in members if m[0] == root), members[0])
        for m in members:
            start_mismatch = max(start_mismatch, float(np.abs(true[m[1]] - true[anchor[1]]).max()))
        rows = np.array([anchor[1]])
        predictions = {}
        for name, start, stop, grasp in members:
            steps = min(longest, stop - 1 - start)
            if steps < min(SIBLING_HORIZONS):
                continue
            actions = arrays.actions[start : start + steps][None]
            readouts = predicted_readouts(model, arrays, rows, actions, state_schema)
            predictions[name] = (start, steps, readouts)
            if steps >= OUTCOME_HORIZON:
                outcome_scores.append(
                    float(readouts["apple_held"][0, OUTCOME_HORIZON // 2 : OUTCOME_HORIZON].max())
                )
                outcome_labels.append(grasp)
        names = sorted(predictions)
        for h in SIBLING_HORIZONS:
            record = pairs[str(h)]
            for a in names:
                for b in names:
                    if a == b:
                        continue
                    sa, na, ra = predictions[a]
                    sb, nb, rb = predictions[b]
                    if na < h or nb < h:
                        continue
                    if dropped[sa + h] or dropped[sb + h]:
                        continue
                    ta, tb = true[sa + h], true[sb + h]
                    pa, pb = ra["palm_minus_apple"][0, h - 1], rb["palm_minus_apple"][0, h - 1]
                    if np.linalg.norm(ta - tb) >= SIBLING_DIVERGENCE_M:
                        record["own"].append(float(np.linalg.norm(pa - ta)))
                        record["swap"].append(float(np.linalg.norm(pb - ta)))
                    if a < b:
                        record["pred"].append(float(np.linalg.norm(pa - pb)))
                        record["true"].append(float(np.linalg.norm(ta - tb)))
    result = {"start_state_max_label_mismatch_m": start_mismatch}
    for h, record in pairs.items():
        own, swap = np.array(record["own"]), np.array(record["swap"])
        result[h] = {
            "qualifying_ordered_pairs": int(len(own)),
            "own_beats_swapped_fraction": float((own < swap).mean()) if len(own) else None,
            "own_error_median_m": _median(own),
            "swapped_error_median_m": _median(swap),
            "unordered_pairs": int(len(record["pred"])),
            "divergence_spearman": spearman(record["pred"], record["true"]),
            "predicted_divergence_median_m": _median(record["pred"]),
            "true_divergence_median_m": _median(record["true"]),
        }
    result["outcome"] = {
        "horizon": OUTCOME_HORIZON,
        "siblings": len(outcome_labels),
        "positives": int(np.sum(outcome_labels)),
        "auroc": auroc(outcome_scores, outcome_labels),
    }
    return result


def collapse_metrics(model, arrays, state_schema, stride=4, chunk=256):
    rows = np.arange(0, len(arrays.states), stride)
    result = model.latent_statistics(encode_rows(model, arrays, rows, state_schema))
    # Descriptive: the image pathway alone, so state fusion cannot mask a collapse.
    result["image_only"] = model.image_embedding_statistics(
        [images(arrays, rows[i : i + chunk]) for i in range(0, len(rows), chunk)]
    )
    return result


# ----- gates ---------------------------------------------------------------------------
def evaluate_gates(metrics, gates):
    """Apply the preregistered thresholds (``gates`` from the frozen manifest)."""
    w = metrics["windows"][str(GATE_HORIZON)]
    s = metrics["siblings"][str(SIBLING_GATE_HORIZON)]
    c = metrics["collapse"]

    def ratio(a, b):
        return None if a is None or b is None or b <= 0 else a / b

    checks = {
        "G1_palm_apple_moving_h8": (
            w["palm_apple_moving_median_m"],
            "<=",
            gates["G1_palm_apple_moving_h8_median_m"],
        ),
        "G2a_vs_persistence_h8": (
            ratio(w["palm_apple_moving_median_m"], w["palm_apple_moving_persistence_median_m"]),
            "<=",
            gates["G2_max_ratio_to_control"],
        ),
        "G2b_vs_shuffled_actions_h8": (
            ratio(w["palm_apple_moving_median_m"], w["palm_apple_moving_shuffled_median_m"]),
            "<=",
            gates["G2_max_ratio_to_control"],
        ),
        "G3_apple_plate_h8": (w["apple_plate_median_m"], "<=", gates["G3_apple_plate_h8_median_m"]),
        "G4_apple_height_grasp_h8": (
            w["apple_height_grasp_median_abs_m"],
            "<=",
            gates["G4_apple_height_grasp_h8_median_abs_m"],
        ),
        "G5_held_auroc_lift_h8": (
            w["held_lift_auroc"]
            if min(w["held_lift_positives"], w["held_lift_negatives"])
            >= gates["G5_minimum_per_class"]
            else None,
            ">=",
            gates["G5_held_auroc_min"],
        ),
        "G6_approach_cost_calibration_h8": (
            w["approach_cost_median_abs_log_ratio"],
            "<=",
            float(np.log(gates["G6_max_cost_ratio"])),
        ),
        "G7a_sibling_own_beats_swapped_h16": (
            s["own_beats_swapped_fraction"]
            if s["qualifying_ordered_pairs"] >= gates["G7_minimum_pairs"]
            else None,
            ">=",
            gates["G7a_own_beats_swapped_min"],
        ),
        "G7b_sibling_divergence_spearman_h16": (
            s["divergence_spearman"],
            ">=",
            gates["G7b_divergence_spearman_min"],
        ),
        "G8a_collapsed_fraction": (
            c["collapsed_fraction"],
            "<=",
            gates["G8_max_collapsed_fraction"],
        ),
        "G8b_effective_rank": (c["effective_rank"], ">=", gates["G8_min_effective_rank"]),
        "G8c_latent_std_mean": (c["latent_std_mean"], ">=", gates["G8_min_latent_std_mean"]),
    }
    result = {}
    for name, (value, op, threshold) in checks.items():
        passed = value is not None and (value <= threshold if op == "<=" else value >= threshold)
        result[name] = {"value": value, "op": op, "threshold": threshold, "passed": bool(passed)}
    result["all_passed"] = all(item["passed"] for item in result.values())
    return result


# ----- training ------------------------------------------------------------------------
def selection_windows(arrays, seed, count):
    windows = window_starts(arrays, GATE_HORIZON, stride=4)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(windows), size=min(count, len(windows)), replace=False))
    return windows[chosen]


def selection_score(model, arrays, windows, state_schema, eligibility):
    measured = window_metrics(model, arrays, windows, state_schema, (GATE_HORIZON,))
    rows = np.unique(arrays.offsets[windows[:, 0]] + windows[:, 1])
    stats = model.latent_statistics(encode_rows(model, arrays, rows, state_schema))
    reasons = []
    if stats["collapsed_fraction"] > eligibility["max_collapsed_fraction"]:
        reasons.append("collapsed_fraction")
    if stats["effective_rank"] < eligibility["min_effective_rank"]:
        reasons.append("effective_rank")
    if stats["latent_std_mean"] < eligibility["min_latent_std_mean"]:
        reasons.append("latent_std_mean")
    score = measured[str(GATE_HORIZON)]["palm_apple_moving_median_m"]
    if score is None or not np.isfinite(score):
        reasons.append("no_moving_windows")
    return {
        "score": score,
        "eligible": not reasons,
        "rejection_reasons": reasons,
        "latent_statistics": stats,
        "metrics": measured[str(GATE_HORIZON)],
    }


def cosine_lr(base, step, steps, final_fraction):
    """Cosine decay from ``base`` at step 1 to ``final_fraction * base`` at ``steps``."""
    progress = 0.0 if steps <= 1 else (step - 1) / (steps - 1)
    return float(
        base * (final_fraction + (1 - final_fraction) * 0.5 * (1 + np.cos(np.pi * progress)))
    )


def selectable(step, decision, best):
    """An eligible val score that improves on the best; the untrained step-0 state is
    logged but never selectable."""
    return step > 0 and decision["eligible"] and (best is None or decision["score"] < best[1])


def _write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _environment():
    import torch

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def _open(config_path):
    from embodied_jepa.config import ExperimentConfig

    config = ExperimentConfig.load(config_path, require_checkpoint=False)
    store = DatasetStore(config.dataset_root)
    settings = dict(config.model_settings)
    cameras = settings.get("cameras") or [settings.get("camera", "onboard_rgb")]
    return config, store, settings, tuple(cameras)


def _provenance(store, source, protocol=PROTOCOL):
    return {
        "dataset_hash": store.manifest_hash,
        "split_hash": json_hash(
            {"splits": store.manifest.get("splits"), "policy": store.manifest.get("split_policy")}
        ),
        "action_hash": json_hash(store.manifest["action_manifest"]),
        "source_revision": source["revision"],
        "source_tree_hash": source["python_source_sha256"],
        "protocol": protocol,
    }


def train(
    config_path,
    *,
    output=None,
    steps,
    batch_size,
    horizon,
    seed,
    device,
    max_seconds,
    validation_every,
    selection_count,
    final_lr_fraction=1.0,
    eligibility,
    workers=8,
    limit_episodes=None,
    require_clean=False,
    protocol=PROTOCOL,
    acknowledge_privileged_training_labels=False,
):
    """Fixed-budget training; ``output`` is best-by-val, ``.latest.pt`` the last state."""
    import torch

    from embodied_jepa.config import MODELS
    from embodied_jepa.training import source_identity

    if acknowledge_privileged_training_labels is not True:
        raise ContractError("training uses privileged readout targets; acknowledge it")
    config, store, settings, cameras = _open(config_path)
    output = Path(output or config.checkpoint)
    paths = {
        "best": output,
        "latest": output.with_name(output.stem + ".latest.pt"),
        "report": output.with_name(output.stem + ".run.json"),
        "curves": output.with_name(output.stem + ".metrics.jsonl"),
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("refusing to overwrite existing training artifacts")
    output.parent.mkdir(parents=True, exist_ok=True)
    clock = RunClock()
    source = source_identity()
    if require_clean and (source["dirty"] is not False or source["revision"] == "unavailable"):
        raise ContractError("the frozen run requires a clean committed checkout")
    report = {
        "format_version": 1,
        "protocol": protocol,
        "status": "running",
        "config": str(Path(config_path).resolve()),
        "config_resolved": config.resolved,
        "backend": config.backend,
        "cameras": list(cameras),
        "device": device,
        "seed": seed,
        "budget": {
            "steps": steps,
            "batch_size": batch_size,
            "horizon": horizon,
            "max_seconds": max_seconds,
            "validation_every": validation_every,
            "selection_windows": selection_count,
            "lr_schedule": {"kind": "cosine", "final_fraction": final_lr_fraction},
        },
        "selection": {
            "split": "val",
            "metric": f"median palm-apple readout error at h={GATE_HORIZON} on predicted "
            "latents over moving selection windows (lower is better)",
            "eligibility": eligibility,
        },
        "limit_episodes": limit_episodes,
        "source": source,
        "environment": _environment(),
        "artifacts": {name: str(path.resolve()) for name, path in paths.items()},
        "test_episodes_decoded": 0,
        "privileged_label_use": "readout-head training targets and val scoring only",
        "validation": [],
    }
    _write_json(paths["report"], report)
    torch.set_num_threads(min(torch.get_num_threads(), 8))

    def check_budget():
        clock.check(max_seconds)

    model, completed, best = None, 0, None
    failed = None
    try:
        train_arrays = load_split(
            store,
            "train",
            cameras,
            workers=workers,
            limit=limit_episodes,
            check_budget=check_budget,
            acknowledge_privileged_training_labels=True,
        )
        val_arrays = load_split(
            store,
            "val",
            cameras,
            workers=workers,
            limit=limit_episodes,
            check_budget=check_budget,
            acknowledge_privileged_training_labels=True,
        )
        report["data"] = {
            "train_episodes": len(train_arrays.episode_ids),
            "train_observations": int(len(train_arrays.states)),
            "val_episodes": len(val_arrays.episode_ids),
            "val_observations": int(len(val_arrays.states)),
            "decode_seconds": clock.elapsed(),
            "peak_host_rss_bytes": peak_rss_bytes(),
        }
        metadata = _provenance(store, source, protocol)
        report["provenance"] = metadata
        model = MODELS.create(
            config.backend,
            state_schema=store.state_schema,
            device=device,
            seed=seed,
            config=settings,
            metadata=metadata,
        )
        normalization = store.manifest["normalization"]
        if normalization["fit_split"] != "train" or list(normalization["episode_ids"]) != list(
            store.manifest["splits"]["train"]
        ):
            raise ContractError("state normalization must be fitted on the train split only")
        moments = normalization["stats"]["observation.state"]
        model.fit_state_normalization(
            moments["mean"], moments["std"], training_episode_ids=normalization["episode_ids"]
        )
        report["model_config"] = model.config
        report["model_implementation_sha256"] = model.implementation_sha256
        report["model_parameters"] = int(sum(p.numel() for p in model.parameters()))
        train_windows = window_starts(train_arrays, horizon)
        chosen = selection_windows(val_arrays, seed + 10001, selection_count)
        report["data"]["train_windows"] = int(len(train_windows))
        report["data"]["selection_windows_sha256"] = hashlib.sha256(chosen.tobytes()).hexdigest()
        sampler = np.random.default_rng(seed)

        def synchronize():
            if device == "mps":
                torch.mps.synchronize()

        def validate():
            nonlocal best
            synchronize()
            begin = time.perf_counter()
            decision = selection_score(model, val_arrays, chosen, store.state_schema, eligibility)
            improved = selectable(completed, decision, best)
            event = {
                "kind": "validation",
                "step": completed,
                "elapsed_seconds": clock.elapsed(),
                "validation_seconds": time.perf_counter() - begin,
                "selected": bool(improved),
            } | decision
            report["validation"].append(event)
            with paths["curves"].open("a") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
            if improved:
                best = (completed, decision["score"])
                model.metadata["runner_state"] = {"step": completed, "selection": decision}
                model.save(paths["best"])
            _write_json(paths["report"], report)

        validate()
        base_lr = model.config["learning_rate"]
        for step in range(1, steps + 1):
            check_budget()
            for group in model.optimizer.param_groups:
                group["lr"] = cosine_lr(base_lr, step, steps, final_lr_fraction)
            picked = train_windows[sampler.integers(len(train_windows), size=batch_size)]
            sequence, targets = batch(train_arrays, picked, horizon, store.state_schema)
            synchronize()
            begin = time.perf_counter()
            metrics = model.train_step(sequence, readout_targets=targets)
            synchronize()
            if not all(np.isfinite(value) for value in metrics.values()):
                raise ContractError("non-finite training metrics")
            completed = step
            if step % 25 == 0 or step == 1:
                with paths["curves"].open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "kind": "train",
                                "step": step,
                                "metrics": metrics,
                                "update_seconds": time.perf_counter() - begin,
                                "elapsed_seconds": clock.elapsed(),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            if step % validation_every == 0 or step == steps:
                validate()
        report["status"] = "completed" if best is not None else "selection_failed"
    except BudgetReached as error:
        report["status"] = str(error)
        try:
            if (
                model is not None
                and completed
                and (not report["validation"] or report["validation"][-1]["step"] != completed)
            ):
                validate()
        except Exception as inner:  # a final validation failure must not hide the stop
            report["final_validation_error"] = f"{type(inner).__name__}: {inner}"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        failed = error
    finally:
        if model is not None:
            try:
                model.metadata["runner_state"] = {"step": completed, "latest": True}
                model.save(paths["latest"])
            except Exception as error:  # keep the report even if the last save fails
                report["latest_checkpoint_error"] = f"{type(error).__name__}: {error}"
        report.update(
            completed_steps=completed,
            best_step=None if best is None else best[0],
            best_selection_score=None if best is None else best[1],
            **clock.snapshot(),
            peak_host_rss_bytes=peak_rss_bytes(),
            best_checkpoint_exists=paths["best"].exists(),
            best_checkpoint_sha256=_sha256(paths["best"]) if paths["best"].exists() else None,
            latest_checkpoint_sha256=_sha256(paths["latest"]) if paths["latest"].exists() else None,
        )
        if device == "mps":
            report["final_mps_driver_bytes"] = torch.mps.driver_allocated_memory()
        _write_json(paths["report"], report)
    if failed is not None:
        raise failed
    return report


def evaluate(
    config_path,
    *,
    checkpoint=None,
    output,
    device,
    gates,
    workers=8,
    limit_episodes=None,
    protocol=PROTOCOL,
    gate_function=None,
    extra_metrics=None,
    require_clean=False,
    acknowledge_privileged_training_labels=False,
):
    """Offline val evaluation of a trained checkpoint against the frozen gates.

    ``gate_function`` and ``extra_metrics`` let a later protocol version reuse this
    runner with its own gate set and its own additional measurements.
    """
    gate_function = evaluate_gates if gate_function is None else gate_function
    from embodied_jepa.config import MODELS
    from embodied_jepa.training import source_identity

    if acknowledge_privileged_training_labels is not True:
        raise ContractError("evaluation scores against privileged labels; acknowledge it")
    # The checkpoint's implementation hash covers the model files, not this runner, where
    # every metric and gate lives; record the evaluator's own revision separately.
    evaluator = source_identity()
    unversioned = evaluator["dirty"] is not False or evaluator["revision"] == "unavailable"
    if require_clean and unversioned:
        raise ContractError("the frozen evaluation requires a clean committed checkout")
    output = Path(output)
    if output.exists():
        raise FileExistsError("refusing to overwrite an evaluation report")
    config, store, settings, cameras = _open(config_path)
    checkpoint = Path(checkpoint or config.checkpoint)
    clock = RunClock()
    import torch

    envelope = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = MODELS.create(
        config.backend,
        state_schema=store.state_schema,
        device=device,
        seed=envelope["seed"],
        config=settings,
        metadata=envelope["metadata"],
    )
    model.load(checkpoint)
    expected = _provenance(store, {"revision": None, "python_source_sha256": None}, protocol)
    for key in ("dataset_hash", "split_hash", "action_hash", "protocol"):
        if envelope["metadata"].get(key) != expected[key]:
            raise ContractError(f"checkpoint {key} does not match the evaluated corpus")
    arrays = load_split(
        store,
        "val",
        cameras,
        workers=workers,
        limit=limit_episodes,
        acknowledge_privileged_training_labels=True,
    )
    windows = window_starts(arrays, max(HORIZONS), stride=4)
    metrics = {
        "windows": window_metrics(model, arrays, windows, store.state_schema),
        "siblings": sibling_metrics(model, arrays, store.state_schema),
        "collapse": collapse_metrics(model, arrays, store.state_schema),
    }
    if extra_metrics is not None:
        metrics |= extra_metrics(model, arrays, store.state_schema)
    result = {
        "format_version": 1,
        "protocol": protocol,
        "split": "val",
        "test_episodes_decoded": 0,
        "backend": config.backend,
        "cameras": list(cameras),
        "device": device,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_step": envelope["metadata"].get("runner_state", {}).get("step"),
        "model_implementation_sha256": model.implementation_sha256,
        "evaluator_source": evaluator,
        "provenance": envelope["metadata"],
        "val_episodes": len(arrays.episode_ids),
        "val_observations": int(len(arrays.states)),
        "limit_episodes": limit_episodes,
        "metrics": metrics,
        "gates": gate_function(metrics, gates),
        "environment": _environment(),
        **clock.snapshot(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, result)
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
    for command in commands.choices.values():
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
    args = parser.parse_args()
    manifest = json.loads(args.protocol_manifest.read_text())
    frozen = manifest["frozen"]
    common = dict(
        device=args.device,
        workers=args.workers,
        limit_episodes=args.limit_episodes,
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
        report = evaluate(
            args.config,
            checkpoint=args.checkpoint,
            output=args.output,
            gates=frozen["gates"],
            require_clean=args.require_clean,
            **common,
        )
        summary = {"all_passed": report["gates"]["all_passed"]} | {
            k: v["passed"] for k, v in report["gates"].items() if k != "all_passed"
        }
    print(json.dumps(summary, indent=2))
    return 0 if report.get("status", "completed") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
