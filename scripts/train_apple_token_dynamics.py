"""TASK-066 runner (``apple_token_dynamics_v1``): a LeWM predictor on frozen DINOv2 patch tokens.

Protocol ``docs/experiments/apple_token_dynamics_v1.md``; design code
``src/embodied_jepa/token_dynamics.py``; model ``models/frozen_tokens.py``. It reuses TASK-065's
runner (``scripts/train_apple_latent_dynamics.py``, imported read-only, its bytes pinned) for the
train/val-only reader, the guard helpers and the window code.

A world-model test only: it trains an action-conditioned latent predictor on stored frames of the
accepted corpus ``apple-look-v1`` and measures it offline. It simulates nothing, runs no controller
and preregisters no control formulation. **Learned Apple->Plate is still 0 successes.**

Stages (any guard failure or exception is V, and ``report.json`` is still written):

1. preflight -- G-hash, clean tree, G-frozen, G-device, G-data, Q-split, G-split, G-folds,
   G-weights;
2. anchor -- G-anchor (frames, CLS and tokens) and G-repro on the 190 post-look frames;
3. labels -- G-labels and the E-post targets (train + val roots only);
4. features -- every train + val frame through the model's ``frozen_features`` (G-cache, with
   the partial-batch determinism check and the 1e-3 anchor bound);
5. training -- 12 models (W, N x seeds x halves) on MPS, each selected on val and evaluated on
   the other half; each seed's G1 moments are reduced as soon as its four models exist;
6. statistics, gates, the ceiling, the decision; G-hash again at the end.

    uv run --no-sync python scripts/train_apple_token_dynamics.py run \\
        --output outputs/task066-token-dynamics/run-1 \\
        --checkpoints checkpoints/task066-token-dynamics/run-1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import info_ceiling as ic  # noqa: E402
from embodied_jepa import latent_dynamics as ld  # noqa: E402
from embodied_jepa import look_corpus as lc  # noqa: E402
from embodied_jepa import token_dynamics as td  # noqa: E402

MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-token-dynamics-v1.json"
TASK064_REPORT = ROOT / "data" / "apple-look-v1-work" / "collection_report.json"
TASK064_PLAN = ROOT / "data" / "apple-look-v1-work" / "plan.json"


def load_task065_runner():
    """TASK-065's runner, imported read-only (its bytes are pinned by this task's manifest)."""
    spec = importlib.util.spec_from_file_location(
        "_task065_runner", ROOT / "scripts" / "train_apple_latent_dynamics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R65 = load_task065_runner()
GuardError = R65.GuardError
log = R65.log
ALL_HORIZONS = R65.ALL_HORIZONS  # E-post: 1, 2, 4, 8, 16 and the extended 32, 64
EALL_HORIZONS = td.REPORTED_HORIZONS
CACHE_ANCHOR_SANITY = R65.CACHE_ANCHOR_SANITY  # 1e-3, TASK-065's amended bound
CACHE_LAYOUT = R65.CACHE_LAYOUT  # 16: the cache featurises each episode in batches of 16
PARTIAL_BATCH_EPISODES = 2  # G-cache (b): whole episodes whose final batch is partial
EVAL_CHUNK = 2048
MOMENT_CHUNK = 8192
SMOKE = {"sessions_per_half": 8, "val_episodes": 8, "updates": 20, "select_every": 10}


# ----- guards as testable helpers (the TASK-065 ones are reused as they are) ----------------------
FROZEN_EFFECTIVE_CONFIG = {
    # the adapter defaults the protocol keeps (section 4.2); SIGReg's projection count is pinned
    # because the option validates only its type (disclosed gap, protocol section 4.1)
    "hidden_dim": 128,
    "predictor_depth": 2,
    "predictor_heads": 2,
    "predictor_head_dim": 24,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "gradient_clip": 1.0,
    "multistep_weight": 1.0,
    "sigreg_projections": 128,
}


def check_model_config(config: dict) -> None:
    """G-frozen, per model: the effective configuration is the preregistered one."""
    expected = dict(td.MODEL_CONFIG) | FROZEN_EFFECTIVE_CONFIG
    wrong = {k: config.get(k) for k, v in expected.items() if config.get(k) != v}
    if wrong:
        raise GuardError(f"G-frozen: the model configuration differs: {wrong}")


def check_frozen() -> None:
    """G-frozen: the calibrated bars, budget and caps are set."""
    try:
        td.require_frozen()
    except td.ContractError as error:
        raise GuardError(f"G-frozen: {error}") from error


def check_token_anchor(hashes: dict, manifest_data: dict) -> None:
    """G-anchor: frames, CLS and tokens of the 190 post-look frames equal TASK-064's."""
    if hashes["frames"] != manifest_data["task064_stored_post_look_frames_sha256"]:
        raise GuardError("G-anchor: the post-look frames differ from TASK-064's")
    if hashes["P_cls"] != manifest_data["task064_P_cls_feature_sha256"]:
        raise GuardError("G-anchor: the post-look P-cls features differ from TASK-064's")
    if hashes["P_tok"] != manifest_data["task064_P_tok_feature_sha256"]:
        raise GuardError("G-anchor: the post-look P-tok features differ from TASK-064's")


def check_partial_batches(cache_rows: dict, again_rows: dict) -> dict:
    """G-cache (b): whole episodes with a partial final batch re-featurise bit for bit."""
    if not cache_rows:
        raise GuardError("G-cache: no partial-batch episode was checked")
    facts = {}
    for episode_id, rows in cache_rows.items():
        if not np.array_equal(rows, again_rows[episode_id]):
            raise GuardError(f"G-cache: {episode_id} (partial final batch) is not deterministic")
        facts[episode_id] = {
            "frames": int(len(rows)),
            "final_batch": int(len(rows) % CACHE_LAYOUT),
            "bit_identical": True,
        }
    return facts


def cache_anchor_sanity(cache_rows, anchor_rows, labels, partial_roots, tol=CACHE_ANCHOR_SANITY):
    """G-cache (c): the pooled cache rows within ``tol`` of the pooled anchor rows."""
    facts = R65.cache_anchor_sanity(cache_rows, anchor_rows, labels, tol)
    facts["explanation"] = (
        "float32 CPU inference differs by batch size: TASK-064's anchor ran the 190 frames in "
        f"batches of 16, so its last {len(partial_roots)} roots were a partial batch; the cache "
        "runs every root's frame 8 inside a full 16-frame batch; pooling is a float64 mean"
    )
    facts["anchor_partial_batch_roots"] = list(partial_roots)
    return facts


# ----- memory-light moments -----------------------------------------------------------------------
def chunked_moments(features, rows, chunk=MOMENT_CHUNK):
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


# ----- per-model evaluation -----------------------------------------------------------------------
def rollout_errors(
    model, features, actions, starts, scale, mode, perm, shift, sessions=None, basis=None
):
    """Per-window normalised squared error at each E-all horizon and, for the gated horizons,
    streamed moments of the normalised predictions (G1 i-ii) and, with ``sessions`` and
    ``basis``, their per-session projected moments (G1 iii). ``mode``: true | zero | wrong."""
    errors = {h: np.empty(len(starts)) for h in EALL_HORIZONS}
    moments = projected = None  # only for the gated mode (W true, N zero): G1 reads no other
    if basis is not None:
        moments = {h: td.Moments(shift) for h in td.GATED_HORIZONS}
        projected = {h: td.SessionMoments(shift, basis) for h in td.GATED_HORIZONS}
    for lo in range(0, len(starts), EVAL_CHUNK):
        sl = slice(lo, lo + EVAL_CHUNK)
        f, a = R65.gather(features, actions, starts[sl], td.TRAIN_HORIZON)
        if mode == "zero":
            a = np.zeros_like(a)
        elif mode == "wrong":
            a = R65.gather(features, actions, starts[perm[sl]], td.TRAIN_HORIZON)[1]
        predicted = model.predict_features(f[:, 0], np.ascontiguousarray(a, np.float32))
        R65.check_finite("predictions", predicted)
        for h in EALL_HORIZONS:
            errors[h][sl] = ld.normalized_sq_error(predicted[:, h - 1], f[:, h], scale)
        if moments is not None:
            for h in td.GATED_HORIZONS:
                moments[h].add(predicted[:, h - 1] / scale)
                projected[h].add(predicted[:, h - 1] / scale, sessions[sl])
    return errors, moments, projected


def train_one(ctx, arm, seed, half, clock):
    """Train, select on val, save; then evaluate on the other half (E-all, E-post, val roots)."""
    import torch

    from embodied_jepa.models.frozen_tokens import frozen_token_model

    started = time.monotonic()
    table, features, actions, offsets = ctx["table"], ctx["features"], ctx["actions"], ctx["offs"]
    members = [i for i, e in enumerate(table) if e["half"] == half]
    member_ids = [table[i]["episode_id"] for i in members]
    model = frozen_token_model(td.BACKEND)(
        ctx["schema"],
        device=ctx["device"],
        seed=seed,
        config=td.MODEL_CONFIG,
        metadata={
            "protocol": td.PROTOCOL,
            "dataset_manifest_sha256": td.DATASET_MANIFEST_SHA256,
            "halves_sha256": ctx["halves_sha256"],
            "arm": arm,
            "half": half,
        },
    )
    check_model_config(model.config)
    frame_rows = np.concatenate(
        [np.arange(offsets[i], offsets[i] + table[i]["length"]) for i in members]
    )
    mean, std = chunked_moments(features, frame_rows)
    model.fit_frozen_feature_normalization(mean, std, training_episode_ids=member_ids)
    if model.metadata["frozen_feature_normalization_episodes_sha256"] != R65.sha256_bytes(
        json.dumps(sorted(member_ids)).encode()
    ):
        raise GuardError("G-split: normalisation episodes are not the training episodes")
    windows = ld.training_windows([table[i]["length"] - 1 for i in members])
    starts = offsets[np.asarray(members)[windows[:, 0]]] + windows[:, 1]
    rng = np.random.default_rng(td.sampler_seed(seed, half, arm))
    updates, every = ctx["updates"], ctx["select_every"]
    curve, losses, best = [], [], (np.inf, 0, None)
    for step in range(1, updates + 1):
        pick = starts[rng.integers(0, len(starts), td.BATCH_SIZE)]
        f, a = R65.gather(features, actions, pick, td.TRAIN_HORIZON)
        if arm == "N":
            a = np.zeros_like(a)
        metrics = model.train_step_features(f, np.ascontiguousarray(a, np.float32))
        if not np.isfinite(metrics["loss"]):
            raise GuardError("G-finite: non-finite training loss")
        if step % 100 == 0:
            losses.append([step, metrics["loss"]])
            R65.check_stage_cap(started, ctx["per_run_seconds"], f"model {arm}-s{seed}-{half}")
            clock.check(f"model {arm}-s{seed}-{half}")
        if step % every == 0:
            value = R65.val_criterion(
                model, features, actions, ctx["val_starts"], ctx["scale"], arm
            )
            curve.append([step, value])
            if value < best[0]:
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best = (value, step, state)
    model.load_state_dict({k: v.to(ctx["device"]) for k, v in best[2].items()})
    path = ctx["checkpoints"] / f"{arm}-s{seed}-{half}.pt"
    model.save(path)
    R65.check_stage_cap(started, ctx["per_run_seconds"], f"model {arm}-s{seed}-{half}")
    points = [u for u, _ in curve]
    record = {
        "arm": arm,
        "seed": seed,
        "half": half,
        "training_episodes": len(member_ids),
        "training_episodes_sha256": R65.sha256_bytes(json.dumps(sorted(member_ids)).encode()),
        "training_windows": int(len(starts)),
        "normalisation_sha256": R65.array_sha256(np.stack([mean, std])),
        "val_curve": curve,
        "selected_update": int(best[1]),
        "selected_val_criterion": float(best[0]),
        "selected_in_last_two_points": bool(best[1] in points[-2:]),
        "losses": losses,
        "checkpoint": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "checkpoint_sha256": R65.sha256_file(path),
        "optimizer_in_checkpoint": "final (weights are the selected update's)",
        "train_seconds": time.monotonic() - started,
        "frozen_encoder_digest": model.frozen_encoder_digest,
        "implementation_sha256": model.implementation_sha256,
        "torch": torch.__version__,
    }
    # --- evaluation of the held-out half (E-all), its roots (E-post) and the val roots ---
    ev = ctx["eval"][ld.other(half)]
    out = {"record": record}
    modes = ("true", "wrong", "zero") if arm == "W" else ("zero",)
    for mode in modes:
        gated = mode == ("true" if arm == "W" else "zero")
        errs, moments, projected = rollout_errors(
            model,
            features,
            actions,
            ev["starts"],
            ctx["scale"],
            mode,
            ev["perm"],
            ctx["shift"],
            ev["sessions"] if gated else None,
            ctx["basis"] if gated else None,
        )
        out[f"eall_{mode}"] = errs
        if gated:
            out["eall_moments"] = moments
            out["eall_projected"] = projected
    post = ctx["post"][ld.other(half)]
    for mode in modes:
        a = post["actions"]
        if mode == "zero":
            a = np.zeros_like(a)
        elif mode == "wrong":
            a = a[post["perm"]]
        predicted = model.predict_features(post["start"], np.ascontiguousarray(a, np.float32))
        R65.check_finite("E-post predictions", predicted)
        out[f"post_{mode}"] = predicted[:, [h - 1 for h in ALL_HORIZONS]]
    val = ctx["val_roots"]
    a = val["actions"] if arm == "W" else np.zeros_like(val["actions"])
    out["val_pred"] = model.predict_features(val["start"], np.ascontiguousarray(a, np.float32))[
        :, [h - 1 for h in ALL_HORIZONS]
    ]
    record["seconds_with_evaluation"] = time.monotonic() - started
    # G-cap covers the model's evaluation too (protocol section 10), unlike TASK-065's cap
    R65.check_stage_cap(started, ctx["per_run_seconds"], f"model {arm}-s{seed}-{half} evaluation")
    del model
    return out


def seed_collapse(ctx, results, seed):
    """G1 for one seed: both halves' moments pooled, against the encoded targets and copy-last.
    The moments are released afterwards, so at most one seed's are ever held."""
    out = {}
    for h in td.GATED_HORIZONS:
        w = results[("W", seed, "A")]["eall_moments"][h].merge(
            results[("W", seed, "B")]["eall_moments"][h]
        )
        n = results[("N", seed, "A")]["eall_moments"][h].merge(
            results[("N", seed, "B")]["eall_moments"][h]
        )
        enc = ctx["encoded_moments"][h]
        pw = results[("W", seed, "A")]["eall_projected"][h].merge(
            results[("W", seed, "B")]["eall_projected"][h]
        )
        pn = results[("N", seed, "A")]["eall_projected"][h].merge(
            results[("N", seed, "B")]["eall_projected"][h]
        )
        out[h] = {
            "collapse_W": td.collapse_statistics(w, enc),
            "collapse_N": td.collapse_statistics(n, enc),
            "collapse_copy_last": ctx["copy_collapse"][h],
            "rank_W_over_N": td.comparative_rank(pw, pn, ctx["encoded_projected"][h]),
        }
    for arm in td.ARMS:
        for half in td.HALVES:
            results[(arm, seed, half)].pop("eall_moments")
            results[(arm, seed, half)].pop("eall_projected")
    return out


# ----- the run -----------------------------------------------------------------------------------
def run(output: Path, checkpoints: Path, *, smoke: bool = False) -> dict:
    output, checkpoints = Path(output), Path(checkpoints)
    for path in (output, checkpoints):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    output.mkdir(parents=True)
    checkpoints.mkdir(parents=True)
    report = {
        "protocol": td.PROTOCOL,
        "task": td.TASK,
        "status": "smoke (not a row)" if smoke else "gated run",
        "learned_apple_to_plate_successes": 0,
        "exemption_spent": False,
        "control_formulation": "none preregistered or implied",
        "test_split_decoded": False,
        "stages": {},
    }
    clock = R65.Clock(td.GLOBAL_WALL_SECONDS or ld.GLOBAL_WALL_SECONDS)
    reader = None
    try:
        report |= _run(report, output, checkpoints, clock, smoke)
        reader = report.pop("_reader", None)
    except BaseException as error:  # noqa: BLE001 -- a crash is V, and the report is written
        reader = report.pop("_reader", None)
        report["outcome"] = "V"
        report["decision"] = td.decide(void=True)
        report["void_reason"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        log(f"VOID: {report['void_reason']}")
    finally:
        if reader is not None:
            report["decoded_episodes"] = len(reader.decoded)
            report["test_split_decoded"] = bool(reader.decoded - reader.allowed)
        report["total_seconds"] = clock.elapsed()
        try:
            from embodied_jepa.training import peak_rss_bytes

            report["peak_rss_bytes"] = peak_rss_bytes()
        except Exception:  # noqa: BLE001
            pass
        R65.write_report(output / "report.json", report)
        log(f"report written: outcome {report.get('outcome')}")
    return report


def _run(report, output, checkpoints, clock, smoke):
    # ----- 1. preflight: every pinned file is hashed and recorded before anything is imported --
    manifest = json.loads(MANIFEST.read_text())
    stages = report["stages"]
    report["pinned_hashes_at_preflight"] = R65.check_pins(manifest["hashes"])
    if not smoke:
        check_frozen()
    import torch

    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa.data import DatasetStore
    from embodied_jepa.models.frozen_tokens import frozen_token_model, pool_tokens

    torch.set_num_threads(td.FEATURE_THREADS)
    dirty = R65.tracked_tree_dirty()
    report["revision"], report["tracked_tree_dirty"] = R65.revision(), bool(dirty)
    if not smoke:
        R65.check_clean(dirty)
    mps = bool(torch.backends.mps.is_available())
    import transformers

    report["environment"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torch_threads": torch.get_num_threads(),
        "mps_available": mps,
    }
    report["frozen"] = {
        "updates": td.UPDATES,
        "select_every": td.SELECT_EVERY,
        "thresholds": td.THRESHOLDS,
        "per_run_seconds": td.PER_RUN_SECONDS,
        "global_wall_seconds": td.GLOBAL_WALL_SECONDS,
        "calibration": td.CALIBRATION,
    }
    R65.check_device(mps)
    store = DatasetStore(ROOT / td.DATASET)  # verifies every recorded hash
    splits = store.manifest["splits"]
    rows = {r["episode_id"]: r for r in store.manifest["episodes"]}
    sessions = {
        k: len({rows[e]["session_id"] for e in splits[k]}) for k in ("train", "val", "test")
    }
    episodes = {k: len(splits[k]) for k in ("train", "val", "test")}
    R65.check_data(store.manifest_hash, sessions, episodes)
    reader = R65.TrainValReader(store, set(splits["train"]) | set(splits["val"]))
    report["_reader"] = reader
    table, halves = R65.build_index(store, reader, splits, smoke)
    halves_sha = ld.halves_sha256(halves)
    R65.check_halves(halves, manifest["halves"]["sha256"])
    training = {h: [e["episode_id"] for e in table if e["half"] == h] for h in td.HALVES}
    split_of = {e: k for k in ("train", "val", "test") for e in splits[k]}
    sessions_of = {e: rows[e]["session_id"] for e in split_of}
    if not smoke:
        ld.check_training_episodes(training, sessions_of, halves, split_of)
    report["data"] = {
        "dataset_manifest_sha256": store.manifest_hash,
        "sessions": sessions,
        "episodes": episodes,
        "halves_sha256": halves_sha,
        "read_episodes": len(table),
        "training_episodes": {h: len(v) for h, v in training.items()},
    }
    task064 = json.loads(TASK064_REPORT.read_text())
    per_root = {int(r["seed"]): r for r in task064["readability"]["per_root"]}
    train_roots = [i for i, e in enumerate(table) if e["kind"] == "root" and e["split"] == "train"]
    val_roots = [i for i, e in enumerate(table) if e["kind"] == "root" and e["split"] == "val"]
    train_roots.sort(key=lambda i: table[i]["seed"])
    val_roots.sort(key=lambda i: table[i]["seed"])
    fold = ic.fold_of(len(train_roots))
    fold_hash = ic.fold_assignment_sha256([table[i]["seed"] for i in train_roots], fold)
    if not smoke:
        R65.check_folds(fold_hash, manifest["evaluation"]["probe"]["fold_assignment_sha256"])
    report["data"]["probe_fold_assignment_sha256"] = fold_hash
    state_schema = store.state_schema  # validated only; the models ignore proprioception
    featurizer = frozen_token_model(td.BACKEND)(state_schema, device="cpu", config=td.MODEL_CONFIG)
    R65.check_weights(
        featurizer.frozen_encoder_digest,
        manifest["model"]["frozen_encoder"]["pretrained_weights_digest"],
    )
    report["weights"] = {
        "files_sha256": pe.check_files(pe.DEFAULT_DIRECTORY),
        "digest": featurizer.frozen_encoder_digest,
    }
    stages["preflight"] = clock.elapsed()
    log("preflight passed")
    # ----- 2. anchor ---------------------------------------------------------------------------
    plan = json.loads(TASK064_PLAN.read_text())
    anchor_roots = lc.read_roots(plan)
    first_frames = {}  # each root's frames 0..15: the cache's own first batch (G-cache a)
    for r in anchor_roots:
        first_frames[int(r["seed"])] = reader.episode(r["episode_id"])[0][:CACHE_LAYOUT]
    anchor_frames = np.stack(
        [first_frames[int(r["seed"])][td.DECISION_FRAME] for r in anchor_roots]
    )
    anchor = pe.features(featurizer._frozen_module, anchor_frames)
    again = pe.features(featurizer._frozen_module, anchor_frames)
    count = len(anchor_roots)
    hashes = {
        "frames": R65.array_sha256(anchor_frames),
        "P_cls": R65.array_sha256(np.asarray(anchor["cls"], np.float64).reshape(count, -1)),
        "P_tok": R65.array_sha256(np.asarray(anchor["tokens"], np.float64).reshape(count, -1)),
    }
    check_token_anchor(hashes, manifest["data"])
    R65.check_repro(anchor["cls"], again["cls"])
    R65.check_repro(anchor["tokens"], again["tokens"])
    pooled_anchor = pool_tokens(anchor["tokens"], td.TOKEN_GRID)
    anchor_by_seed = {int(r["seed"]): pooled_anchor[k] for k, r in enumerate(anchor_roots)}
    partial = [r["episode_id"] for r in anchor_roots[count - count % pe.BATCH :]]
    report["anchor"] = {
        "hashes": hashes,
        "roots": count,
        "G_repro": "bit-identical (CLS and tokens)",
        "anchor_partial_batch_roots": partial,
    }
    del anchor, again
    stages["anchor"] = clock.elapsed()
    log("anchor passed")
    # ----- 3. labels ---------------------------------------------------------------------------
    targets, label_diff = {}, 0.0
    for i in train_roots + val_roots:
        e = table[i]
        labels = reader.labels(e["episode_id"])
        R65.check_look_labels(labels["collector__phase_index"], e["episode_id"])
        apple = np.asarray(labels["privileged__apple_position_world"], np.float64)[:, :2]
        truth = per_root[e["seed"]]["apple_xy"]
        label_diff = max(label_diff, R65.check_apple_label(apple[0], truth, e["episode_id"]))
        frames_needed = td.DECISION_FRAME + max(ALL_HORIZONS)
        if len(apple) <= frames_needed:
            raise GuardError(f"G-labels: {e['episode_id']} is shorter than frame {frames_needed}")
        targets[i] = {
            "xy": {h: apple[td.DECISION_FRAME + h] for h in (0, *ALL_HORIZONS)},
            "moved_m": {
                h: float(np.linalg.norm(apple[td.DECISION_FRAME + h] - apple[0]))
                for h in ALL_HORIZONS
            },
            "occluded": bool(per_root[e["seed"]]["reset_occluded"]),
        }
    report["labels"] = {"apple_label_max_abs_m": label_diff, "roots": len(targets)}
    stages["labels"] = clock.elapsed()
    # ----- 4. features -------------------------------------------------------------------------
    features, actions, offsets, feature_seconds = featurise(featurizer, reader, table, clock)
    np.save(output / "features.npy", features)
    read_roots = train_roots + val_roots
    cache_rows = np.stack([features[offsets[i] + td.DECISION_FRAME] for i in read_roots])
    again_rows = np.stack(
        [
            featurizer.frozen_features(first_frames[table[i]["seed"]])[td.DECISION_FRAME]
            for i in read_roots
        ]
    )
    R65.check_cache(cache_rows, again_rows)  # G-cache (a)
    partial_ids = [
        i for i, e in enumerate(table) if e["split"] == "train" and e["length"] % CACHE_LAYOUT
    ][:PARTIAL_BATCH_EPISODES]
    whole_cache, whole_again = {}, {}
    for i in partial_ids:
        frames = reader.episode(table[i]["episode_id"])[0]
        whole_cache[table[i]["episode_id"]] = features[offsets[i] : offsets[i] + len(frames)]
        whole_again[table[i]["episode_id"]] = featurizer.frozen_features(frames)
    partial_facts = check_partial_batches(whole_cache, whole_again)  # G-cache (b)
    sanity = cache_anchor_sanity(  # G-cache (c)
        cache_rows,
        np.stack([anchor_by_seed[table[i]["seed"]] for i in read_roots]),
        [table[i]["episode_id"] for i in read_roots],
        partial,
    )
    report["features"] = {
        "frames": int(len(features)),
        "latent_dim": int(features.shape[1]),
        "sha256": R65.array_sha256(features),
        "file_sha256": R65.sha256_file(output / "features.npy"),
        "seconds": feature_seconds,
        "G_cache_a": "cache path bit-identical on a second featurisation of the post-look frames",
        "G_cache_b_partial_batches": partial_facts,
        "cache_vs_anchor": sanity,
    }
    train_rows = np.concatenate(
        [
            np.arange(offsets[i], offsets[i] + e["length"])
            for i, e in enumerate(table)
            if e["split"] == "train"
        ]
    )
    train_mean, train_std = chunked_moments(features, train_rows)
    scale = np.maximum(train_std, td.METRIC_FLOOR_STD)
    shift = train_mean / scale  # a fixed moments shift (numerics only; statistics are centred)
    report["features"]["metric_scale_sha256"] = R65.array_sha256(scale)
    del featurizer
    stages["features"] = clock.elapsed()
    log("features done")
    # ----- evaluation sets ----------------------------------------------------------------------
    val_members = [i for i, e in enumerate(table) if e["split"] == "val"]
    val_starts, _, _ = R65.window_arrays(
        table, offsets, val_members, td.TRAIN_HORIZON, td.EVAL_STRIDE
    )
    evalsets, posts = {}, {}
    for half in td.HALVES:
        members = [i for i, e in enumerate(table) if e["half"] == half]
        starts, _, sess = R65.window_arrays(
            table, offsets, members, td.TRAIN_HORIZON, td.EVAL_STRIDE
        )
        perm = ld.cross_session_shuffle(sess, np.random.default_rng(td.SHUFFLE_SEED))
        evalsets[half] = {"starts": starts, "sessions": sess, "perm": perm}
        roots = [i for i in train_roots if table[i]["half"] == half]
        start = features[offsets[roots] + td.DECISION_FRAME]
        acts = np.stack(
            [
                actions[offsets[i] + td.DECISION_FRAME : offsets[i] + td.DECISION_FRAME + 64]
                for i in roots
            ]
        )
        rperm = ld.cross_session_shuffle(
            np.asarray([table[i]["session"] for i in roots]),
            np.random.default_rng(td.SHUFFLE_SEED),
        )
        posts[half] = {"roots": roots, "start": start, "actions": acts, "perm": rperm}
    # G1 (iii)'s basis: the top principal directions of every train frame's encoded latent
    # (train-only, model-free, fixed before any model is trained).
    train_moments = td.Moments(shift)
    for lo in range(0, len(train_rows), MOMENT_CHUNK):
        train_moments.add(features[train_rows[lo : lo + MOMENT_CHUNK]] / scale)
    basis = td.projection_basis(train_moments)
    report["features"]["projection_basis_sha256"] = R65.array_sha256(basis)
    del train_moments
    # G1's model-free references: the encoded targets and copy-last on every E-all window.
    all_starts = np.concatenate([evalsets[h]["starts"] for h in td.HALVES])
    all_sessions = np.concatenate([evalsets[h]["sessions"] for h in td.HALVES])
    encoded_moments = {h: td.Moments(shift) for h in td.GATED_HORIZONS}
    encoded_projected = {h: td.SessionMoments(shift, basis) for h in td.GATED_HORIZONS}
    copy_moments = td.Moments(shift)
    for lo in range(0, len(all_starts), EVAL_CHUNK):
        s = all_starts[lo : lo + EVAL_CHUNK]
        copy_moments.add(features[s] / scale)
        for h in td.GATED_HORIZONS:
            encoded_moments[h].add(features[s + h] / scale)
            encoded_projected[h].add(features[s + h] / scale, all_sessions[lo : lo + EVAL_CHUNK])
    copy_collapse = {
        h: td.collapse_statistics(copy_moments, encoded_moments[h]) for h in td.GATED_HORIZONS
    }
    del copy_moments
    val_root_start = features[offsets[val_roots] + td.DECISION_FRAME]
    val_root_actions = np.stack(
        [
            actions[offsets[i] + td.DECISION_FRAME : offsets[i] + td.DECISION_FRAME + 64]
            for i in val_roots
        ]
    )
    ctx = {
        "table": table,
        "features": features,
        "actions": actions,
        "offs": offsets,
        "schema": state_schema,
        "device": td.DEVICE,
        "halves_sha256": halves_sha,
        "checkpoints": checkpoints,
        "scale": scale,
        "shift": shift,
        "val_starts": val_starts,
        "eval": evalsets,
        "post": posts,
        "val_roots": {"start": val_root_start, "actions": val_root_actions},
        "encoded_moments": encoded_moments,
        "encoded_projected": encoded_projected,
        "basis": basis,
        "copy_collapse": copy_collapse,
        "updates": SMOKE["updates"] if smoke else td.UPDATES,
        "select_every": SMOKE["select_every"] if smoke else td.SELECT_EVERY,
        "per_run_seconds": td.PER_RUN_SECONDS or ld.PER_RUN_SECONDS,
    }
    report["evaluation_sets"] = {
        "val_windows": int(len(val_starts)),
        "eall_windows": {h: int(len(evalsets[h]["starts"])) for h in td.HALVES},
        "epost_roots": {h: len(posts[h]["roots"]) for h in td.HALVES},
        "val_roots": len(val_roots),
    }
    stages["evaluation_sets"] = clock.elapsed()
    # ----- 5. training + per-model evaluation; each seed's G1 reduced at once -------------------
    seeds = td.MODEL_SEEDS[:1] if smoke else td.MODEL_SEEDS
    results, models, collapse = {}, [], {}
    for seed in seeds:
        for arm in td.ARMS:
            for half in td.HALVES:
                log(f"training {arm}-s{seed}-{half}")
                out = train_one(ctx, arm, seed, half, clock)
                results[(arm, seed, half)] = out
                models.append(out["record"])
                log(
                    f"done {arm}-s{seed}-{half}: selected {out['record']['selected_update']}, "
                    f"{out['record']['train_seconds']:.0f} s"
                )
        collapse[seed] = seed_collapse(ctx, results, seed)
    report["models"] = models
    stages["training"] = clock.elapsed()
    # ----- 6. statistics ------------------------------------------------------------------------
    stats = statistics(ctx, results, collapse, seeds, train_roots, val_roots, targets, fold, smoke)
    R65.check_statistics_finite(stats)
    report |= stats
    stages["statistics"] = clock.elapsed()
    # ----- G-hash again, then the decision -------------------------------------------------------
    report["pinned_hashes_at_end"] = R65.check_pins(manifest["hashes"])
    if not smoke:
        R65.check_clean(R65.tracked_tree_dirty())
    clock.check("the decision")
    if smoke:
        report["outcome"] = "smoke (not a row)"
    else:
        report["decision"] = td.decide(
            void=False,
            seeds={s: report["gates"][str(s)]["seed"] for s in seeds},
            ceiling={h: report["ceiling"][str(h)]["meets_t1"] for h in td.GATED_HORIZONS},
        )
        report["outcome"] = report["decision"]["outcome"]
    report["_reader"] = reader
    return report


def featurise(model, reader, table, clock):
    started = time.monotonic()
    feats, actions, offsets = [], [], []
    total = 0
    for i, entry in enumerate(table):
        frames, acts, _ = reader.episode(entry["episode_id"])
        feats.append(model.frozen_features(frames))
        actions.append(acts)
        offsets.append(total)
        total += len(frames)
        if i % 50 == 0:
            log(f"features {i}/{len(table)} episodes, {total} frames")
            R65.check_stage_cap(started, td.FEATURE_SECONDS, "featurisation")
            clock.check("featurisation")
    R65.check_stage_cap(started, td.FEATURE_SECONDS, "featurisation")
    features = np.concatenate(feats)
    del feats
    R65.check_finite("features", features)
    return features, np.concatenate(actions), np.asarray(offsets), time.monotonic() - started


def statistics(ctx, results, collapse, seeds, train_roots, val_roots, targets, fold, smoke):
    table, features, offsets = ctx["table"], ctx["features"], ctx["offs"]
    scale = ctx["scale"]
    n = len(train_roots)
    # --- probes: fitted on the encoded grid of frame 8 + h of the train roots (10-fold CV) ---
    xy = {h: np.stack([targets[i]["xy"][h] for i in train_roots]) for h in (0, *ALL_HORIZONS)}
    if smoke:
        noise = np.random.default_rng(0)
        xy = {h: v.mean(0) + noise.normal(0, 0.017, v.shape) for h, v in xy.items()}
    occluded = np.array([targets[i]["occluded"] for i in train_roots])
    idx = ic.bootstrap_indices(n)
    encoded = {
        h: features[offsets[train_roots] + td.DECISION_FRAME + h].astype(np.float64)
        for h in (0, *ALL_HORIZONS)
    }
    probes = {}
    for h in (0, *ALL_HORIZONS):
        g, diag = ic.gram(encoded[h])
        pred, fits, selections = ic.nested_cv(g, diag, xy[h], fold)
        occ = ic.prior_predictions(xy[h], np.ones(n), occluded, fold)["B_occ"]
        probes[h] = {"pred": pred, "fits": fits, "selections": selections, "occ": occ}
    where = {}
    for half in td.HALVES:
        for k, i in enumerate(ctx["post"][half]["roots"]):
            where[i] = (half, k)
    hidx = {h: ALL_HORIZONS.index(h) for h in ALL_HORIZONS}

    def post_source(arm, seed, mode, h):
        rows = []
        for i in train_roots:
            half, k = where[i]
            trained = ld.other(half)  # the model that did not train on this root
            rows.append(results[(arm, seed, trained)][f"post_{mode}"][k, hidx[h]])
        return np.stack(rows)

    def readability(source, h, probe_h=None):
        p = probes[h if probe_h is None else probe_h]
        hat = R65.read_source(p["fits"], fold, encoded[h if probe_h is None else probe_h], source)
        err = ic.xy_error_cm(hat, xy[h])
        err_enc = ic.xy_error_cm(probes[h]["pred"], xy[h])
        err_occ = ic.xy_error_cm(probes[h]["occ"], xy[h])
        return R65.t1_stats(err, err_occ, err_enc, idx), err

    out_read, per_root_err = {"encoded": {}}, {}
    # the encoded grid at every h, h = 0 (frame 8) included (protocol section 8)
    err0 = ic.xy_error_cm(probes[0]["pred"], xy[0])
    out_read["encoded"][0] = R65.t1_stats(
        err0, ic.xy_error_cm(probes[0]["occ"], xy[0]), err0, idx
    ) | {"selections": probes[0]["selections"]}
    for h in ALL_HORIZONS:
        err_enc = ic.xy_error_cm(probes[h]["pred"], xy[h])
        err_occ = ic.xy_error_cm(probes[h]["occ"], xy[h])
        out_read["encoded"][h] = R65.t1_stats(err_enc, err_occ, err_enc, idx) | {
            "selections": probes[h]["selections"]
        }
        per_root_err[("encoded", h)] = err_enc
        stats, err = readability(encoded[0], h)
        out_read.setdefault("copy_last", {})[h] = stats
        per_root_err[("copy_last", h)] = err
    ceiling = {
        str(h): {
            "median_cm": out_read["encoded"][h]["median_cm"],
            "ratio_to_B_occ_ci95": out_read["encoded"][h]["ratio_to_B_occ"]["ci95"],
            "meets_t1": td.ceiling_passes(out_read["encoded"][h]),
        }
        for h in td.GATED_HORIZONS
    }
    gates, eall_report, post_report, val_report = {}, {}, {}, {}
    sess = np.concatenate([ctx["eval"][h]["sessions"] for h in td.HALVES])
    idx_c = ld.cluster_bootstrap_indices(len(np.unique(sess)))
    all_starts = np.concatenate([ctx["eval"][h]["starts"] for h in td.HALVES])
    copy = {
        h: ld.normalized_sq_error(features[all_starts], features[all_starts + h], scale)
        for h in EALL_HORIZONS
    }
    for seed in seeds:
        s = str(seed)

        def pooled(arm, mode, h, seed=seed):
            return np.concatenate(
                [results[(arm, seed, ld.other(half))][f"eall_{mode}"][h] for half in td.HALVES]
            )

        eall = {}
        for h in EALL_HORIZONS:
            w, nn = pooled("W", "true", h), pooled("N", "zero", h)
            wrong, zero = pooled("W", "wrong", h), pooled("W", "zero", h)
            eall[h] = {
                "mse_W": float(w.mean()),
                "mse_copy_last": float(copy[h].mean()),
                "mse_N": float(nn.mean()),
                "mse_W_wrong_actions": float(wrong.mean()),
                "mse_W_zero_actions": float(zero.mean()),
                "W_over_copy_last": ld.cluster_ratio(w, copy[h], sess, idx_c),
                "W_over_N": ld.cluster_ratio(w, nn, sess, idx_c),
                "wrong_over_W": ld.cluster_ratio(wrong, w, sess, idx_c),
                "zero_over_W": ld.cluster_ratio(zero, w, sess, idx_c),
                "N_over_copy_last": ld.cluster_ratio(nn, copy[h], sess, idx_c),
            }
            if h in td.GATED_HORIZONS:
                eall[h] |= collapse[seed][h]
        eall_report[s] = eall
        # --- E-post readability ---
        post = {}
        for h in ALL_HORIZONS:
            post[h] = {}
            for name, arm, mode in (
                ("W", "W", "true"),
                ("W_wrong_actions", "W", "wrong"),
                ("W_zero_actions", "W", "zero"),
                ("N", "N", "zero"),
            ):
                source = post_source(arm, seed, mode, h)
                stats, err = readability(source, h)
                post[h][name] = stats
                per_root_err[(f"s{seed}_{name}", h)] = err
                if name == "W":
                    post[h]["W_frame8_probe"] = readability(source, h, probe_h=0)[0]
                    w_err = err
            post[h]["W_minus_copy_last_cm"] = ic.paired_difference(
                w_err, per_root_err[("copy_last", h)], idx, ic._median
            )
            post[h]["W_minus_N_cm"] = ic.paired_difference(
                w_err, per_root_err[(f"s{seed}_N", h)], idx, ic._median
            )
        root_sessions = np.asarray([table[i]["session"] for i in train_roots])
        idx_r = ld.cluster_bootstrap_indices(len(np.unique(root_sessions)))
        for h in ALL_HORIZONS:
            target = features[offsets[train_roots] + td.DECISION_FRAME + h]
            mse = {
                name: ld.normalized_sq_error(post_source(arm, seed, mode, h), target, scale)
                for name, arm, mode in (
                    ("W", "W", "true"),
                    ("wrong", "W", "wrong"),
                    ("zero", "W", "zero"),
                    ("N", "N", "zero"),
                )
            }
            mse["copy_last"] = ld.normalized_sq_error(encoded[0], target, scale)
            post[h]["latent_mse"] = {
                "W_over_copy_last": ld.cluster_ratio(
                    mse["W"], mse["copy_last"], root_sessions, idx_r
                ),
                "W_over_N": ld.cluster_ratio(mse["W"], mse["N"], root_sessions, idx_r),
                "wrong_over_W": ld.cluster_ratio(mse["wrong"], mse["W"], root_sessions, idx_r),
                "zero_over_W": ld.cluster_ratio(mse["zero"], mse["W"], root_sessions, idx_r),
            }
        post_report[s] = post
        # --- gates ---
        horizons = {}
        for h in td.GATED_HORIZONS:
            e = eall[h]
            horizons[h] = {
                "G1": td.g1_passes(e["collapse_W"], e["rank_W_over_N"]) if not smoke else False,
                "G2": td.g2_passes(e["W_over_copy_last"]),
                "G3": td.g3_passes(e["W_over_N"]),
                "G4": R65.g4_gate(e["wrong_over_W"], e["zero_over_W"]),
                "G5": td.g5_passes(post[h]["W"]),
            }
        gates[s] = {"per_horizon": horizons, "seed": td.seed_gates(horizons)}
        if not smoke:
            gates[s]["G1_parts"] = {
                h: td.g1_parts(eall[h]["collapse_W"], eall[h]["rank_W_over_N"])
                for h in td.GATED_HORIZONS
            }
        # --- val roots, secondary (probe fitted on all train roots) ---
        val_report[s] = {}
        for h in ALL_HORIZONS:
            g, diag = ic.gram(encoded[h])
            train_index = np.arange(n)
            chosen = ic.select(g, diag, train_index, xy[h], outer=ic.FOLDS)
            readout = ic.fit_readout(
                chosen["family"], chosen["lam_rel"], g, diag, train_index, xy[h]
            )
            vxy = np.stack([targets[i]["xy"][h] for i in val_roots])
            if smoke:
                vxy = xy[h].mean(0) + np.random.default_rng(1).normal(0, 0.017, vxy.shape)
            entry = {"selection": chosen}
            sources = {
                "encoded": features[offsets[val_roots] + td.DECISION_FRAME + h],
                "copy_last": ctx["val_roots"]["start"],
            }
            for half in td.HALVES:
                sources[f"W_{half}"] = results[("W", seed, half)]["val_pred"][:, hidx[h]]
                sources[f"N_{half}"] = results[("N", seed, half)]["val_pred"][:, hidx[h]]
            for name, x in sources.items():
                g_rt, norms = ic.cross_gram(np.asarray(x, np.float64), encoded[h])
                hat = readout.predict(g_rt, norms)
                err = ic.xy_error_cm(hat, vxy)
                entry[name] = {"median_cm": float(np.median(err)), "mean_cm": float(err.mean())}
            val_report[s][h] = entry
    moved = {
        h: int(sum(targets[i]["moved_m"][h] > R65.APPLE_MOVE_M for i in train_roots + val_roots))
        for h in ALL_HORIZONS
    }
    per_root = []
    for k, i in enumerate(train_roots):
        per_root.append(
            {
                "seed": table[i]["seed"],
                "half": table[i]["half"],
                "fold": int(fold[k]),
                "occluded": bool(occluded[k]),
                "errors_cm": {
                    f"{name}@{h}": float(err[k]) for (name, h), err in per_root_err.items()
                },
            }
        )
    return {
        "eall": eall_report,
        "epost": {"encoded_and_copy_last": out_read, "per_seed": post_report},
        "ceiling": ceiling,
        "val_secondary": val_report,
        "gates": gates,
        "apple_moved_more_than_1mm_roots": moved,
        "per_root": per_root,
        "smoke_targets": "seeded noise; no real target reached a readout" if smoke else None,
        "smoke_G1": "not evaluated (calibrated bars are not read in a smoke)" if smoke else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "smoke"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--checkpoints", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.output, args.checkpoints, smoke=args.command == "smoke")
    return 0 if report.get("outcome") != "V" else 1


if __name__ == "__main__":
    raise SystemExit(main())
