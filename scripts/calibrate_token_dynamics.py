"""TASK-066 pre-freeze calibration (``apple_token_dynamics_v1`` section 6), on the disjoint pilot.

It fixes two things before the freeze, from latent-space quantities only:
- the training budget, from the held-out criterion curves (``token_dynamics.budget_rule``);
- the G1 bars, from the effective-rank and std ratios of predicted against encoded latents
  (``token_dynamics.bar_rule``).

**Leakage rules (binding).** It reads only TASK-064's pilot ``pilot-d`` (reset seeds
47900-47931), which is not part of ``apple-look-v1``: no corpus episode, no corpus split and no
test split is opened. It reads frames and actions only: no label sidecar is opened, no apple
readout is fitted or evaluated, and nothing but loss curves and collapse statistics is computed.
It simulates nothing. **Learned Apple->Plate is still 0 successes.**

    uv run --no-sync python scripts/calibrate_token_dynamics.py run \\
        --output outputs/task066-calibration/run-1 \\
        --checkpoints checkpoints/task066-calibration/run-1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import latent_dynamics as ld  # noqa: E402
from embodied_jepa import token_dynamics as td  # noqa: E402


def load_task065_runner():
    """TASK-065's runner, imported read-only for its reader and helpers (its bytes are pinned)."""
    spec = importlib.util.spec_from_file_location(
        "_task065_runner", ROOT / "scripts" / "train_apple_latent_dynamics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R65 = load_task065_runner()
log = R65.log
EVAL_CHUNK = 1024
DETERMINISM_EPISODES = 2  # whole episodes re-featurised (partial final batches included)
SMOKE_UPDATES = 1000  # smoke: a mechanics check, not a calibration (its rules are not used)


def pilot_partition(sessions) -> dict:
    """``{session: "train" | "heldout"}``: sorted, permuted by ``default_rng(6600)``, first 24
    train."""
    ordered = sorted(sessions)
    if len(ordered) != td.PILOT_SESSIONS or len(set(ordered)) != len(ordered):
        raise td.GuardError(f"pilot: expected {td.PILOT_SESSIONS} distinct sessions")
    order = np.random.default_rng(td.PILOT_SPLIT_SEED).permutation(len(ordered))
    return {
        ordered[i]: ("train" if rank < td.PILOT_TRAIN_SESSIONS else "heldout")
        for rank, i in enumerate(order)
    }


def chunked_moments(features, rows, chunk=8192):
    """float64 mean and std of ``features[rows]`` in two chunked passes (no full copy)."""
    total = np.zeros(features.shape[1])
    for lo in range(0, len(rows), chunk):
        total += features[rows[lo : lo + chunk]].astype(np.float64).sum(0)
    mean = total / len(rows)
    square = np.zeros(features.shape[1])
    for lo in range(0, len(rows), chunk):
        d = features[rows[lo : lo + chunk]].astype(np.float64) - mean
        square += (d * d).sum(0)
    return mean, np.sqrt(square / len(rows))


# ----- stage 1: features -------------------------------------------------------------------------
def features_stage(output: Path, report: dict) -> None:
    import torch

    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa.data import DatasetStore
    from embodied_jepa.models.frozen_tokens import frozen_token_model

    torch.set_num_threads(td.FEATURE_THREADS)
    store = DatasetStore(ROOT / td.PILOT_DATASET)  # verifies every recorded hash
    if store.manifest_hash != td.PILOT_MANIFEST_SHA256:
        raise td.GuardError(f"pilot manifest {store.manifest_hash[:12]} is not the pinned one")
    rows = sorted(store.manifest["episodes"], key=lambda r: r["episode_id"])
    seeds = {int(r["metadata"]["reset_seed"]) for r in rows}
    if min(seeds) < 47900 or max(seeds) > 47931:
        raise td.GuardError("pilot: a reset seed lies outside 47900-47931")
    part = pilot_partition({r["session_id"] for r in rows})
    reader = R65.TrainValReader(store, {r["episode_id"] for r in rows})
    featurizer = frozen_token_model(td.BACKEND)(
        store.state_schema, device="cpu", config=td.MODEL_CONFIG
    )
    manifest = json.loads((ROOT / "benchmarks/manifests/apple-latent-dynamics-v1.json").read_text())
    R65.check_weights(
        featurizer.frozen_encoder_digest,
        manifest["model"]["frozen_encoder"]["pretrained_weights_digest"],
    )
    started = time.monotonic()
    feats, actions, table, offsets, total = [], [], [], [], 0
    first_frames = {}
    for i, row in enumerate(rows):
        frames, acts, _ = reader.episode(row["episode_id"])
        feats.append(featurizer.frozen_features(frames))
        if i < DETERMINISM_EPISODES:
            first_frames[row["episode_id"]] = frames
        actions.append(acts)
        offsets.append(total)
        total += len(frames)
        table.append(
            {
                "episode_id": row["episode_id"],
                "session": row["session_id"],
                "part": part[row["session_id"]],
                "length": int(row["length"]),
            }
        )
        if i % 20 == 0:
            log(f"pilot features {i}/{len(rows)} episodes, {total} frames")
    features = np.concatenate(feats)
    R65.check_finite("pilot features", features)
    determinism = {}
    for k, (episode_id, frames) in enumerate(first_frames.items()):
        again = featurizer.frozen_features(frames)
        cached = features[offsets[k] : offsets[k] + len(frames)]
        if not np.array_equal(again, cached):
            raise td.GuardError(f"pilot features are not deterministic on {episode_id}")
        determinism[episode_id] = {
            "frames": len(frames),
            "final_batch": len(frames) % pe.BATCH or pe.BATCH,
            "bit_identical": True,
        }
    np.save(output / "features.npy", features)
    np.save(output / "actions.npy", np.concatenate(actions))
    train_rows = np.concatenate(
        [
            np.arange(o, o + e["length"])
            for o, e in zip(offsets, table, strict=True)
            if e["part"] == "train"
        ]
    )
    mean, std = chunked_moments(features, train_rows)
    np.save(output / "scale.npy", np.maximum(std, td.METRIC_FLOOR_STD))
    np.save(output / "train_mean.npy", mean)
    (output / "table.json").write_text(json.dumps({"table": table, "offsets": offsets}))
    report["pilot"] = {
        "dataset": td.PILOT_DATASET,
        "manifest_sha256": store.manifest_hash,
        "episodes": len(rows),
        "frames": int(len(features)),
        "sessions": {p: sum(1 for v in part.values() if v == p) for p in ("train", "heldout")},
        "episodes_by_part": {
            p: sum(1 for e in table if e["part"] == p) for p in ("train", "heldout")
        },
        "reset_seeds": [min(seeds), max(seeds)],
        "labels_read": False,
        "decoded_episodes": len(reader.decoded),
    }
    report["features"] = {
        "sha256": R65.array_sha256(features),
        "seconds": time.monotonic() - started,
        "per_frame_seconds": (time.monotonic() - started) / len(features),
        "determinism_whole_episodes": determinism,
        "weights_digest": featurizer.frozen_encoder_digest,
        "latent_dim": int(features.shape[1]),
    }


# ----- stage 2: one calibration model (a subprocess) ---------------------------------------------
def heldout_windows(table, offsets, part, stride):
    members = [i for i, e in enumerate(table) if e["part"] == part]
    starts, _, _ = R65.window_arrays(table, np.asarray(offsets), members, td.TRAIN_HORIZON, stride)
    return members, starts


def criterion(model, features, actions, starts, scale, arm):
    total, count = 0.0, 0
    for lo in range(0, len(starts), EVAL_CHUNK):
        f, a = R65.gather(features, actions, starts[lo : lo + EVAL_CHUNK], td.TRAIN_HORIZON)
        if arm == "N":
            a = np.zeros_like(a)
        predicted = model.predict_features(f[:, 0], np.ascontiguousarray(a, np.float32))
        err = ld.normalized_sq_error(predicted, f[:, 1:], scale)
        total += float(err.sum())
        count += err.size
    return total / count


def reference_moments(features, starts, scale, shift):
    """Encoded targets at h and the copy-last start latents of the held-out windows (model-free)."""
    enc = {h: td.Moments(shift) for h in td.GATED_HORIZONS}
    copy = td.Moments(shift)
    for lo in range(0, len(starts), EVAL_CHUNK):
        s = starts[lo : lo + EVAL_CHUNK]
        copy.add(features[s] / scale)
        for h in td.GATED_HORIZONS:
            enc[h].add(features[s + h] / scale)
    return enc, copy


def collapse_at(model, features, actions, starts, scale, arm, shift, reference):
    """G1's quantities at h = 8 and 16 on the held-out windows: predictions (W with true actions,
    N with zero) against the encoded targets, both divided by the metric scale."""
    enc, copy = reference
    pred = {h: td.Moments(shift) for h in td.GATED_HORIZONS}
    for lo in range(0, len(starts), EVAL_CHUNK):
        f, a = R65.gather(features, actions, starts[lo : lo + EVAL_CHUNK], td.TRAIN_HORIZON)
        if arm == "N":
            a = np.zeros_like(a)
        predicted = model.predict_features(f[:, 0], np.ascontiguousarray(a, np.float32))
        for h in td.GATED_HORIZONS:
            pred[h].add(predicted[:, h - 1] / scale)
    return {
        str(h): {
            "model": td.collapse_statistics(pred[h], enc[h]),
            "copy_last": td.collapse_statistics(copy, enc[h]),
        }
        for h in td.GATED_HORIZONS
    }


def train_stage(
    output: Path, checkpoints: Path, arm: str, seed: int, updates: int = td.CAL_UPDATES
) -> None:
    import torch

    from embodied_jepa.contracts import StateSchema
    from embodied_jepa.models.frozen_tokens import frozen_token_model

    torch.set_num_threads(2)
    features = np.load(output / "features.npy", mmap_mode="r")
    features = np.ascontiguousarray(features)
    actions = np.load(output / "actions.npy")
    scale = np.load(output / "scale.npy")
    shift = np.load(output / "train_mean.npy") / scale
    meta = json.loads((output / "table.json").read_text())
    table, offsets = meta["table"], np.asarray(meta["offsets"])
    members, _ = heldout_windows(table, offsets, "train", 1)
    member_ids = [table[i]["episode_id"] for i in members]
    _, held = heldout_windows(table, offsets, "heldout", ld.EVAL_STRIDE)
    reference = reference_moments(features, held, scale, shift)
    schema = StateSchema(("unused",), ("1",), "calibration_v0")  # the model ignores state
    model = frozen_token_model(td.BACKEND)(
        schema,
        device=td.DEVICE,
        seed=seed,
        config=td.MODEL_CONFIG,
        metadata={"protocol": td.PROTOCOL, "calibration": True, "arm": arm},
    )
    frame_rows = np.concatenate(
        [np.arange(offsets[i], offsets[i] + table[i]["length"]) for i in members]
    )
    mean, std = chunked_moments(features, frame_rows)
    model.fit_frozen_feature_normalization(mean, std, training_episode_ids=member_ids)
    windows = ld.training_windows([table[i]["length"] - 1 for i in members])
    starts = offsets[np.asarray(members)[windows[:, 0]]] + windows[:, 1]
    # A calibration stream, disjoint from the gated runs' (salt 6500): salt 6600.
    rng = np.random.default_rng(np.random.SeedSequence([td.PILOT_SPLIT_SEED, seed]))
    curve, losses, ranks, best = [], [], {}, (np.inf, 0, None)
    started = time.monotonic()
    for step in range(1, updates + 1):
        pick = starts[rng.integers(0, len(starts), td.BATCH_SIZE)]
        f, a = R65.gather(features, actions, pick, td.TRAIN_HORIZON)
        if arm == "N":
            a = np.zeros_like(a)
        metrics = model.train_step_features(f, np.ascontiguousarray(a, np.float32))
        if not np.isfinite(metrics["loss"]):
            raise td.GuardError("non-finite calibration loss")
        if step % 100 == 0:
            losses.append([step, metrics["loss"]])
        if step % td.CAL_SELECT_EVERY == 0:
            value = criterion(model, features, actions, held, scale, arm)
            curve.append([step, value])
            if value < best[0]:
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best = (value, step, state)
            log(f"cal {arm}-s{seed} {step}: held-out {value:.4f}")
        if step % td.CAL_RANK_EVERY == 0:
            ranks[str(step)] = collapse_at(
                model, features, actions, held, scale, arm, shift, reference
            )
    seconds = time.monotonic() - started
    model.load_state_dict({k: v.to(td.DEVICE) for k, v in best[2].items()})
    path = checkpoints / f"{arm}-s{seed}.pt"
    model.save(path)
    record = {
        "arm": arm,
        "seed": seed,
        "training_episodes": len(member_ids),
        "training_windows": int(len(starts)),
        "heldout_windows": int(len(held)),
        "curve": curve,
        "losses": losses,
        "collapse_along_curve": ranks,
        "selected_update": int(best[1]),
        "selected_criterion": float(best[0]),
        "saturation_update": td.saturation_update(curve),
        "collapse_at_selected": collapse_at(
            model, features, actions, held, scale, arm, shift, reference
        ),
        "train_seconds": seconds,
        "updates": updates,
        "seconds_per_update_with_criteria": seconds / updates,
        "checkpoint_sha256": R65.sha256_file(path),
        "implementation_sha256": model.implementation_sha256,
        "torch": torch.__version__,
    }
    (output / f"model-{arm}-s{seed}.json").write_text(json.dumps(record, indent=1))


# ----- the driver --------------------------------------------------------------------------------
def rules(models: dict) -> dict:
    sats = [m["saturation_update"] for m in models.values()]
    rank_ratios, std_ratios = [], []
    for m in models.values():
        if m["arm"] != "W":
            continue
        for h in td.GATED_HORIZONS:
            stats = m["collapse_at_selected"][str(h)]["model"]
            rank_ratios.append(stats["effective_rank_ratio"])
            std_ratios.append(stats["std_ratio"])
    return {
        "budget": td.budget_rule(sats),
        "G1_min_effective_rank_ratio": td.bar_rule(rank_ratios, td.RANK_FLOOR),
        "G1_min_std_ratio": td.bar_rule(std_ratios, td.STD_FLOOR),
    }


def run(output: Path, checkpoints: Path, updates: int = td.CAL_UPDATES) -> dict:
    for path in (output, checkpoints):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    output.mkdir(parents=True)
    checkpoints.mkdir(parents=True)
    started = time.monotonic()
    report = {
        "protocol": td.PROTOCOL,
        "task": td.TASK,
        "status": "pre-freeze calibration (latent-space quantities only; not a row)",
        "learned_apple_to_plate_successes": 0,
        "corpus_decoded": False,
        "labels_read": False,
        "apple_readout_fitted": False,
        "updates": updates,
        "smoke": updates != td.CAL_UPDATES,
    }
    try:
        dirty = R65.tracked_tree_dirty()
        report["revision"], report["tracked_tree_dirty"] = R65.revision(), bool(dirty)
        if updates == td.CAL_UPDATES:  # a smoke may run on a dirty tree; a calibration may not
            R65.check_clean(dirty)
        import torch
        import transformers

        report["environment"] = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
        }
        R65.check_device(report["environment"]["mps_available"])
        features_stage(output, report)
        log("pilot features done; training the calibration models in parallel")
        procs = {}
        for arm, seed in td.CAL_RUNS:
            logfile = (output / f"train-{arm}-s{seed}.log").open("w")
            procs[(arm, seed)] = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "train",
                    "--output",
                    str(output),
                    "--checkpoints",
                    str(checkpoints),
                    "--arm",
                    arm,
                    "--seed",
                    str(seed),
                    "--updates",
                    str(updates),
                ],
                stdout=logfile,
                stderr=subprocess.STDOUT,
            )
        codes = {f"{a}-s{s}": p.wait() for (a, s), p in procs.items()}
        if any(codes.values()):
            raise RuntimeError(f"a calibration model failed: {codes}")
        models = {
            f"{a}-s{s}": json.loads((output / f"model-{a}-s{s}.json").read_text())
            for a, s in td.CAL_RUNS
        }
        report["models"] = models
        report["rules"] = rules(models)
        report["outcome"] = "calibrated"
    except BaseException as error:  # noqa: BLE001 -- a failed calibration is recorded
        report["outcome"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        log(f"CALIBRATION FAILED: {report['error']}")
    finally:
        report["total_seconds"] = time.monotonic() - started
        R65.write_report(output / "report.json", report)
    return report


def _load_calibration(cal: Path):
    features = np.ascontiguousarray(np.load(cal / "features.npy", mmap_mode="r"))
    meta = json.loads((cal / "table.json").read_text())
    return (
        features,
        np.load(cal / "actions.npy"),
        np.load(cal / "scale.npy"),
        np.load(cal / "train_mean.npy"),
        meta["table"],
        np.asarray(meta["offsets"]),
    )


def controls(output: Path, cal: Path, checkpoints: Path) -> dict:
    """The combined G1 on the pilot, and the synthetic collapse controls (owner ruling
    2026-09-27): W's held-out predictions truncated to their own top k principal directions.
    The combined G1 must fail k = 1, 2, 4; k = 8 is reported. Latent-space quantities only."""
    import torch

    from embodied_jepa.contracts import StateSchema
    from embodied_jepa.models.frozen_tokens import frozen_token_model

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    torch.set_num_threads(td.FEATURE_THREADS)
    started = time.monotonic()
    calibration = json.loads((cal / "report.json").read_text())
    if calibration.get("outcome") != "calibrated":
        raise td.GuardError("the calibration did not complete")
    features, actions, scale, train_mean, table, offsets = _load_calibration(cal)
    shift = train_mean / scale
    _, held = heldout_windows(table, offsets, "heldout", ld.EVAL_STRIDE)
    members, _ = heldout_windows(table, offsets, "train", 1)
    held_members = [i for i, e in enumerate(table) if e["part"] == "heldout"]
    held_sessions = R65.window_arrays(
        table, offsets, held_members, td.TRAIN_HORIZON, ld.EVAL_STRIDE
    )[2]
    train_rows = np.concatenate(
        [np.arange(offsets[i], offsets[i] + table[i]["length"]) for i in members]
    )
    train_moments = td.Moments(shift)
    for lo in range(0, len(train_rows), 8192):
        train_moments.add(features[train_rows[lo : lo + 8192]] / scale)
    basis = td.projection_basis(train_moments)
    del train_moments
    models = calibration["models"]
    ranks = [
        m["collapse_at_selected"][str(h)]["model"]["effective_rank_ratio"]
        for m in models.values()
        if m["arm"] == "W"
        for h in td.GATED_HORIZONS
    ]
    stds = [
        m["collapse_at_selected"][str(h)]["model"]["std_ratio"]
        for m in models.values()
        if m["arm"] == "W"
        for h in td.GATED_HORIZONS
    ]
    rank_bar, std_bar = td.bar_rule(ranks, td.RANK_FLOOR), td.bar_rule(stds, td.STD_FLOOR)
    thresholds = td.THRESHOLDS | {
        "G1_min_effective_rank_ratio": rank_bar["bar"],
        "G1_min_std_ratio": std_bar["bar"],
    }
    schema = StateSchema(("unused",), ("1",), "calibration_v0")

    def predictions(arm, seed):
        model = frozen_token_model(td.BACKEND)(
            schema,
            device=td.DEVICE,
            seed=seed,
            config=td.MODEL_CONFIG,
            metadata={"protocol": td.PROTOCOL, "calibration": True, "arm": arm},
        )
        model.load(checkpoints / f"{arm}-s{seed}.pt")
        out = {h: [] for h in td.GATED_HORIZONS}
        for lo in range(0, len(held), EVAL_CHUNK):
            f, a = R65.gather(features, actions, held[lo : lo + EVAL_CHUNK], td.TRAIN_HORIZON)
            if arm == "N":
                a = np.zeros_like(a)
            p = model.predict_features(f[:, 0], np.ascontiguousarray(a, np.float32))
            for h in td.GATED_HORIZONS:
                out[h].append(p[:, h - 1] / scale)
        return {h: np.concatenate(v) for h, v in out.items()}

    def moments(rows):
        m = td.Moments(shift)
        m.add(rows)
        return m

    def projected(rows):
        m = td.SessionMoments(shift, basis)
        m.add(rows, held_sessions)
        return m

    encoded = {h: features[held + h] / scale for h in td.GATED_HORIZONS}
    enc_m = {h: moments(encoded[h]) for h in td.GATED_HORIZONS}
    enc_p = {h: projected(encoded[h]) for h in td.GATED_HORIZONS}
    n_pred = predictions("N", 0)
    n_p = {h: projected(n_pred[h]) for h in td.GATED_HORIZONS}
    result = {}
    for seed in (0, 1):
        w_pred = predictions("W", seed)
        per_h = {}
        for h in td.GATED_HORIZONS:
            rows = w_pred[h]
            full = td.collapse_statistics(moments(rows), enc_m[h])
            comp = td.comparative_rank(projected(rows), n_p[h], enc_p[h])
            entry = {
                "model": {
                    "collapse": full,
                    "rank_W_over_N": comp,
                    "G1_parts": td.g1_parts(full, comp, thresholds),
                }
            }
            mean = rows.mean(0)
            centered = rows - mean
            values, vectors = np.linalg.eigh(centered.T @ centered)
            order = np.argsort(values)[::-1]
            for k in td.TRUNCATION_CONTROLS:
                v = vectors[:, order[:k]]
                truncated = mean + (centered @ v) @ v.T
                stats = td.collapse_statistics(moments(truncated), enc_m[h])
                comp_k = td.comparative_rank(projected(truncated), n_p[h], enc_p[h])
                parts = td.g1_parts(stats, comp_k, thresholds)
                entry[f"truncated_k{k}"] = {
                    "collapse": stats,
                    "rank_W_over_N": comp_k,
                    "G1_parts": parts,
                    "G1_passes": bool(all(parts.values())),
                }
            per_h[str(h)] = entry
        result[f"W-s{seed}"] = per_h
    binding = {
        f"k{k}": not any(
            result[s][str(h)][f"truncated_k{k}"]["G1_passes"]
            for s in result
            for h in td.GATED_HORIZONS
        )
        for k in td.TRUNCATION_CONTROLS
    }
    report = {
        "protocol": td.PROTOCOL,
        "status": "pre-freeze collapse controls on the pilot (latent-space only; not a row)",
        "revision": R65.revision(),
        "tracked_tree_dirty": bool(R65.tracked_tree_dirty()),
        "calibration_report_sha256": R65.sha256_file(cal / "report.json"),
        "bars": {"rank": rank_bar, "std": std_bar},
        "heldout_windows": int(len(held)),
        "heldout_sessions": int(len(np.unique(held_sessions))),
        "basis_dim": int(basis.shape[1]),
        "results": result,
        "controls_fail_G1": binding,
        "owner_condition_met": bool(binding["k1"] and binding["k2"] and binding["k4"]),
        "labels_read": False,
        "seconds": time.monotonic() - started,
    }
    R65.write_report(output / "report.json", report)
    return report


def readability(output: Path, cal: Path) -> dict:
    """The encoded 4 x 4 grid's readability on the disjoint pilot (owner ruling 2026-09-27):
    encoded latents only, never a prediction. TASK-063's probe and TASK-059's T1 bar on the
    pilot's roots at frames 8, 16 and 24 (h = 0, 8, 16), against full 16 x 16 P-tok and P-mean
    on the same roots. It reads the pilot's label sidecars (disclosed)."""
    import torch

    from embodied_jepa import info_ceiling as ic
    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa.data import DatasetStore
    from embodied_jepa.models.frozen_tokens import frozen_token_model

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    torch.set_num_threads(td.FEATURE_THREADS)
    started = time.monotonic()
    features, _, _, _, table, offsets = _load_calibration(cal)
    store = DatasetStore(ROOT / td.PILOT_DATASET)
    if store.manifest_hash != td.PILOT_MANIFEST_SHA256:
        raise td.GuardError("pilot manifest is not the pinned one")
    work = json.loads(
        (ROOT / td.PILOT_DATASET).parent.joinpath("work/collection_report.json").read_text()
    )
    flags = {int(r["seed"]): r for r in work["readability"]["per_root"]}
    rows = {r["episode_id"]: r for r in store.manifest["episodes"]}
    reader = R65.TrainValReader(store, set(rows))
    featurizer = frozen_token_model(td.BACKEND)(
        store.state_schema, device="cpu", config=td.MODEL_CONFIG
    )
    roots = sorted(
        (
            (int(rows[e["episode_id"]]["metadata"]["reset_seed"]), i)
            for i, e in enumerate(table)
            if rows[e["episode_id"]]["metadata"]["kind"] == "root"
        )
    )
    roots = [(s, i) for s, i in roots if s in flags]  # the roots with recorded occlusion flags
    frames_at = (td.DECISION_FRAME, td.DECISION_FRAME + 8, td.DECISION_FRAME + 16)
    xy, grid, tok, mean_tok, occluded, label_diff = {}, {}, {}, {}, [], 0.0
    for f in frames_at:
        xy[f], grid[f], tok[f], mean_tok[f] = [], [], [], []
    frames_needed = []
    for seed, i in roots:
        e = table[i]
        labels = reader.labels(e["episode_id"])
        R65.check_look_labels(labels["collector__phase_index"], e["episode_id"])
        apple = np.asarray(labels["privileged__apple_position_world"], np.float64)[:, :2]
        label_diff = max(
            label_diff, R65.check_apple_label(apple[0], flags[seed]["apple_xy"], e["episode_id"])
        )
        occluded.append(bool(flags[seed]["reset_occluded"]))
        frames = reader.episode(e["episode_id"])[0]
        for f in frames_at:
            xy[f].append(apple[f])
            grid[f].append(features[offsets[i] + f])
            frames_needed.append((f, frames[f]))
    for f in frames_at:
        batch = np.stack([fr for g, fr in frames_needed if g == f])
        read = pe.features(featurizer._frozen_module, batch)
        tok[f] = read["tokens"]
        mean_tok[f] = read["tokens"].reshape(len(batch), -1, td.TOKEN_WIDTH).mean(1)
    n = len(roots)
    fold = ic.fold_of(n)
    idx = ic.bootstrap_indices(n)
    occluded = np.asarray(occluded)
    results = {}
    for f in frames_at:
        truth = np.stack(xy[f])
        occ = ic.xy_error_cm(
            ic.prior_predictions(truth, np.ones(n), occluded, fold)["B_occ"], truth
        )
        entry = {}
        for name, x in (
            ("grid_4x4", np.stack(grid[f])),
            ("P_tok_16x16", tok[f]),
            ("P_mean", mean_tok[f]),
        ):
            g, diag = ic.gram(np.asarray(x, np.float64))
            pred, _, selections = ic.nested_cv(g, diag, truth, fold)
            err = ic.xy_error_cm(pred, truth)
            stats = R65.t1_stats(err, occ, err, idx)
            entry[name] = {
                "median_cm": stats["median_cm"],
                "median_ci95": stats.get("ci95"),
                "ratio_to_B_occ": stats["ratio_to_B_occ"],
                "meets_t1": td.ceiling_passes(stats),
                "selections": selections,
            }
        entry["B_occ_median_cm"] = float(np.median(occ))
        results[f"h{f - td.DECISION_FRAME}"] = entry
    gated = [f"h{h}" for h in td.GATED_HORIZONS]
    grid_ok = all(results[h]["grid_4x4"]["meets_t1"] for h in gated)
    tok_ok = all(results[h]["P_tok_16x16"]["meets_t1"] for h in gated)
    report = {
        "protocol": td.PROTOCOL,
        "status": "pre-freeze readability of ENCODED 4x4 grids on the pilot (not a row)",
        "revision": R65.revision(),
        "tracked_tree_dirty": bool(R65.tracked_tree_dirty()),
        "roots": n,
        "root_seeds": [s for s, _ in roots],
        "labels_read": "pilot-d privileged__apple_position_world and collector__phase_index "
        "(owner ruling 2026-09-27); no corpus label, no prediction",
        "apple_label_max_abs_m": label_diff,
        "results": results,
        "rule": "reconsider the pooling if the 4x4 grid misses the T1 bar at h = 8 or 16 while "
        "full P-tok meets it; if full P-tok also misses, the pilot is too small to inform",
        "grid_meets_t1_at_gated": grid_ok,
        "p_tok_meets_t1_at_gated": tok_ok,
        "verdict": "keep 4x4"
        if grid_ok
        else ("reconsider pooling" if tok_ok else "pilot too small to inform"),
        "test_split_decoded": False,
        "seconds": time.monotonic() - started,
    }
    R65.write_report(output / "report.json", report)
    return report


def anchor(output: Path) -> dict:
    """The label-free anchor pre-check on the corpus (protocol section 13): features only.

    It decodes the 190 train + val root episodes (frames only, through TASK-065's train/val-only
    reader; no label sidecar, no readout) and checks, for every one of the 190 roots -- the 14 in
    TASK-064's partial final batch included -- the anchor hashes, the pooled cache-layout rows
    against the pooled anchor rows, and the determinism of whole-episode featurisation (partial
    final batches included)."""
    import torch

    from embodied_jepa import look_corpus as lc
    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa.data import DatasetStore
    from embodied_jepa.models.frozen_tokens import frozen_token_model, pool_tokens

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    torch.set_num_threads(td.FEATURE_THREADS)
    started = time.monotonic()
    manifest = json.loads((ROOT / "benchmarks/manifests/apple-latent-dynamics-v1.json").read_text())
    store = DatasetStore(ROOT / td.DATASET)
    if store.manifest_hash != td.DATASET_MANIFEST_SHA256:
        raise td.GuardError("the corpus manifest is not the pinned one")
    splits = store.manifest["splits"]
    reader = R65.TrainValReader(store, set(splits["train"]) | set(splits["val"]))
    featurizer = frozen_token_model(td.BACKEND)(
        store.state_schema, device="cpu", config=td.MODEL_CONFIG
    )
    plan = json.loads((ROOT / "data/apple-look-v1-work/plan.json").read_text())
    roots = lc.read_roots(plan)
    task064 = json.loads((ROOT / "data/apple-look-v1-work/collection_report.json").read_text())
    pins = task064["readability"]["feature_sha256"]
    whole, first = {}, {}
    for r in roots:
        frames = reader.episode(r["episode_id"])[0]
        first[int(r["seed"])] = frames[: pe.BATCH]
        if len(whole) < DETERMINISM_EPISODES and len(frames) % pe.BATCH:
            whole[r["episode_id"]] = frames
    anchor_frames = np.stack([first[int(r["seed"])][td.DECISION_FRAME] for r in roots])
    read = pe.features(featurizer._frozen_module, anchor_frames)
    again = pe.features(featurizer._frozen_module, anchor_frames)
    hashes = {
        "frames": R65.array_sha256(anchor_frames),
        "P_cls": R65.array_sha256(np.asarray(read["cls"], np.float64).reshape(len(roots), -1)),
        "P_tok": R65.array_sha256(np.asarray(read["tokens"], np.float64).reshape(len(roots), -1)),
    }
    checks = {
        "frames_equal_task064": hashes["frames"]
        == manifest["data"]["task064_stored_post_look_frames_sha256"],
        "P_cls_equal_task064": hashes["P_cls"] == pins["P_cls"],
        "P_tok_equal_task064": hashes["P_tok"] == pins["P_tok"],
        "repro_bit_identical": bool(
            np.array_equal(read["tokens"], again["tokens"])
            and np.array_equal(read["cls"], again["cls"])
        ),
    }
    pooled_anchor = pool_tokens(read["tokens"], td.TOKEN_GRID)
    cache = np.stack(
        [featurizer.frozen_features(first[int(r["seed"])])[td.DECISION_FRAME] for r in roots]
    )
    cache_again = np.stack(
        [featurizer.frozen_features(first[int(r["seed"])])[td.DECISION_FRAME] for r in roots]
    )
    diff = np.abs(cache.astype(np.float64) - pooled_anchor)
    rel = np.linalg.norm(cache.astype(np.float64) - pooled_anchor, axis=1) / np.linalg.norm(
        pooled_anchor.astype(np.float64), axis=1
    )
    differing = [roots[k]["episode_id"] for k in np.flatnonzero(diff.max(1) > 0)]
    determinism = {}
    for episode_id, frames in whole.items():
        one, two = featurizer.frozen_features(frames), featurizer.frozen_features(frames)
        determinism[episode_id] = {
            "frames": len(frames),
            "final_batch": len(frames) % pe.BATCH,
            "bit_identical": bool(np.array_equal(one, two)),
        }
    report = {
        "protocol": td.PROTOCOL,
        "status": "pre-freeze anchor check (features only; no label, no readout, no model)",
        "revision": R65.revision(),
        "tracked_tree_dirty": bool(R65.tracked_tree_dirty()),
        "roots": len(roots),
        "anchor_batch": pe.BATCH,
        "anchor_partial_final_batch_roots": [
            r["episode_id"] for r in roots[len(roots) - len(roots) % pe.BATCH :]
        ],
        "hashes": hashes,
        "checks": checks,
        "pooled_cache_vs_pooled_anchor": {
            "roots_compared": len(roots),
            "max_abs": float(diff.max()),
            "max_relative_to_row_norm": float(rel.max()),
            "differing_roots": differing,
        },
        "cache_layout_repro_bit_identical": bool(np.array_equal(cache, cache_again)),
        "whole_episode_determinism": determinism,
        "test_split_decoded": bool(reader.decoded - reader.allowed),
        "labels_read": False,
        "seconds": time.monotonic() - started,
    }
    R65.write_report(output / "report.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "smoke"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--checkpoints", type=Path, required=True)
    a = sub.add_parser("anchor")
    a.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("controls")
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--calibration", type=Path, required=True)
    c.add_argument("--checkpoints", type=Path, required=True)
    r = sub.add_parser("readability")
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--calibration", type=Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--checkpoints", type=Path, required=True)
    t.add_argument("--arm", choices=td.ARMS, required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--updates", type=int, default=td.CAL_UPDATES)
    args = parser.parse_args(argv)
    if args.command == "train":
        train_stage(args.output, args.checkpoints, args.arm, args.seed, args.updates)
        return 0
    if args.command == "controls":
        return (
            0
            if controls(args.output, args.calibration, args.checkpoints)["owner_condition_met"]
            else 1
        )
    if args.command == "readability":
        readability(args.output, args.calibration)
        return 0
    if args.command == "anchor":
        checks = anchor(args.output)["checks"]
        return 0 if all(checks.values()) else 1
    updates = SMOKE_UPDATES if args.command == "smoke" else td.CAL_UPDATES
    report = run(args.output, args.checkpoints, updates)
    return 0 if report.get("outcome") == "calibrated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
