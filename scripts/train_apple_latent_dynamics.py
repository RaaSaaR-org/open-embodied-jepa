"""TASK-065 runner (``apple_latent_dynamics_v1``): a LeWM predictor on frozen DINOv2 CLS latents.

Protocol ``docs/experiments/apple_latent_dynamics_v1.md``; design code
``src/embodied_jepa/latent_dynamics.py``; model ``models/frozen_encoder.py``.

A world-model test only: it trains an action-conditioned latent predictor on stored frames of the
accepted corpus ``apple-look-v1`` and measures it offline. It simulates nothing, runs no controller
and preregisters no control formulation. **Learned Apple->Plate is still 0 successes.**

Stages (any guard failure or exception is V, and ``report.json`` is still written):

1. preflight -- G-hash (every pinned file, recorded here, before any stage), clean tree,
   G-device, G-data, Q-split, G-split (halves pin), G-folds, G-weights;
2. anchor -- G-anchor and G-repro on the 190 post-look frames, as TASK-064 featurised them;
3. labels -- G-labels (the look's phase labels, the frame-0 apple label against TASK-064's reset
   truth) and the E-post targets (train + val only);
4. features -- every train + val frame through the model's ``frozen_features`` (G-cache);
5. training -- 12 models (arms W, N x seeds x halves) on MPS, each selected on val, each
   evaluated at once on E-all and E-post of the other half;
6. statistics, gates, decision; G-hash again at the end.

The test split is never decoded: ``TrainValReader`` refuses every other episode id.

    uv run --no-sync python scripts/train_apple_latent_dynamics.py run \\
        --output outputs/task065-latent-dynamics/run-1 \\
        --checkpoints checkpoints/task065-latent-dynamics/run-1
"""

from __future__ import annotations

import argparse
import hashlib
import io
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

from embodied_jepa import info_ceiling as ic  # noqa: E402
from embodied_jepa import latent_dynamics as ld  # noqa: E402
from embodied_jepa import look_corpus as lc  # noqa: E402
from embodied_jepa import observation_reprobe as orp  # noqa: E402

MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-latent-dynamics-v1.json"
TASK064_REPORT = ROOT / "data" / "apple-look-v1-work" / "collection_report.json"
TASK064_PLAN = ROOT / "data" / "apple-look-v1-work" / "plan.json"
ALL_HORIZONS = (1, 2, 4, 8, 16, 32, 64)  # E-post: reported + extended
EALL_HORIZONS = ld.REPORTED_HORIZONS  # (1, 2, 4, 8, 16)
LABEL_TOLERANCE_M = 1e-6
CACHE_TOLERANCE = 1e-5
APPLE_MOVE_M = 1e-3
EVAL_CHUNK = 2048
SMOKE = {"sessions_per_half": 8, "val_episodes": 8, "updates": 20, "select_every": 10}


class GuardError(ld.GuardError):
    """A protocol §9 guard failed (V)."""


# ----- small helpers ---------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values) -> str:
    return sha256_bytes(np.ascontiguousarray(values).tobytes())


def finite_json(value):
    """Replace every non-finite float by None and list where (a report is always writable)."""
    found = []

    def clean(item, path):
        if isinstance(item, dict):
            return {str(k): clean(v, f"{path}/{k}") for k, v in item.items()}
        if isinstance(item, list | tuple):
            return [clean(v, f"{path}/{i}") for i, v in enumerate(item)]
        if isinstance(item, np.ndarray):
            return clean(item.tolist(), path)
        if isinstance(item, np.bool_):
            return bool(item)
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, float | np.floating):
            if not np.isfinite(item):
                found.append(f"{path}={float(item)!r}")
                return None
            return float(item)
        return item

    return clean(value, ""), found


def write_report(path: Path, report: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    clean, found = finite_json(report)
    clean["non_finite_fields"] = found
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(clean, indent=1, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class Clock:
    def __init__(self, cap: float = ld.GLOBAL_WALL_SECONDS):
        self.start, self.cap = time.monotonic(), float(cap)

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def check(self, stage: str) -> None:
        if self.elapsed() > self.cap:
            raise GuardError(f"G-cap: global wall cap {self.cap} s reached before {stage}")


def check_stage_cap(started: float, cap: float, what: str) -> None:
    if time.monotonic() - started > cap:
        raise GuardError(f"G-cap: {what} exceeded its {cap} s cap")


# ----- guards as testable helpers ---------------------------------------------------------------
def check_pins(pinned: dict, root: Path = ROOT) -> dict:
    """G-hash: every pinned file's sha256, recorded; any missing or different file is V."""
    found = {}
    for relative, want in sorted(pinned.items()):
        path = root / relative
        if not path.is_file():
            raise GuardError(f"G-hash: pinned file missing: {relative}")
        got = sha256_file(path)
        if got != want:
            raise GuardError(f"G-hash: {relative} sha256 {got[:12]} != pinned {want[:12]}")
        found[relative] = got
    return found


def tracked_tree_dirty(root: Path = ROOT) -> list[str]:
    out = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    return [line for line in out.splitlines() if line.strip()]


def revision(root: Path = ROOT) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def check_clean(dirty: list[str]) -> None:
    if dirty:
        raise GuardError(f"G-hash: the tracked tree is dirty: {dirty[:5]}")


def check_device(mps_available: bool) -> None:
    if not mps_available:
        raise GuardError("G-device: MPS is unavailable")


def check_data(manifest_hash: str, sessions: dict, episodes: dict) -> None:
    if manifest_hash != ld.DATASET_MANIFEST_SHA256:
        raise GuardError(f"G-data: dataset manifest {manifest_hash[:12]} is not the pinned one")
    if sessions != ld.EXPECTED_SESSIONS:
        raise GuardError(f"G-data: sessions {sessions} != {ld.EXPECTED_SESSIONS}")
    if episodes != ld.EXPECTED_EPISODES:
        raise GuardError(f"G-data: episodes {episodes} != {ld.EXPECTED_EPISODES}")


def check_halves(halves: dict, pinned: str) -> None:
    got = ld.halves_sha256(halves)
    if got != pinned:
        raise GuardError(f"G-split: halves {got[:12]} != pinned {pinned[:12]}")


def check_folds(got: str, want: str) -> None:
    if got != want:
        raise GuardError(f"G-folds: fold hash {got[:12]} != pinned {want[:12]}")


def check_weights(digest: str, want: str) -> None:
    if digest != want:
        raise GuardError(f"G-weights: frozen encoder digest {digest[:12]} != pinned {want[:12]}")


def check_anchor(frames_sha: str, features_sha: str, pins: dict) -> None:
    if frames_sha != pins["task064_stored_post_look_frames_sha256"]:
        raise GuardError("G-anchor: the post-look frames differ from TASK-064's")
    if features_sha != pins["task064_P_cls_feature_sha256"]:
        raise GuardError("G-anchor: the post-look P-cls features differ from TASK-064's")


def check_repro(first, again) -> None:
    if not np.array_equal(first, again):
        raise GuardError("G-repro: a second featurisation is not bit-identical")


def check_cache(cache_rows, anchor_rows, tol: float = CACHE_TOLERANCE) -> float:
    diff = float(np.max(np.abs(np.asarray(cache_rows, np.float64) - anchor_rows)))
    if not diff <= tol:
        raise GuardError(f"G-cache: frame-8 cache rows differ from the anchor by {diff}")
    return diff


def check_look_labels(phases, episode_id: str) -> None:
    phases = np.asarray(phases)
    if len(phases) < lc.LOOK_STEPS or not (phases[: lc.LOOK_STEPS] == lc.LOOK_PHASE_INDEX).all():
        raise GuardError(f"G-labels: {episode_id}'s first 8 phase labels are not the look")


def check_apple_label(label_xy, truth_xy, episode_id: str, tol=LABEL_TOLERANCE_M) -> float:
    diff = float(np.max(np.abs(np.asarray(label_xy, np.float64) - np.asarray(truth_xy))))
    if not diff <= tol:
        raise GuardError(f"G-labels: {episode_id} frame-0 apple label off TASK-064 by {diff} m")
    return diff


def check_finite(name: str, values) -> None:
    if not np.isfinite(np.asarray(values)).all():
        raise GuardError(f"G-finite: non-finite {name}")


# ----- the reader: train + val only -------------------------------------------------------------
class TrainValReader:
    """Decodes only the allowed (train + val) episodes, each after its sha256 check. Any other
    episode id -- the test split -- is refused before a file is opened (Q-split)."""

    def __init__(self, store, allowed_ids):
        self.store = store
        self.allowed = frozenset(allowed_ids)
        self.rows = {r["episode_id"]: r for r in store.manifest["episodes"]}
        self.decoded: set[str] = set()

    def row(self, episode_id):
        if episode_id not in self.allowed:
            raise GuardError(f"Q-split: refusing to decode {episode_id}: not train/val")
        return self.rows[episode_id]

    def _table(self, episode_id, columns):
        row = self.row(episode_id)
        import pyarrow.parquet as pq

        path = self.store.root / row["path"]
        payload = path.read_bytes()
        if sha256_bytes(payload) != self.store.manifest["sha256"][row["path"]]:
            raise GuardError(f"G-data: episode file hash mismatch: {episode_id}")
        self.decoded.add(episode_id)
        return pq.read_table(io.BytesIO(payload), columns=columns), row

    def episode(self, episode_id, *, frames=True):
        """``(frames uint8 [L,112,112,3] or None, actions float32 [L,14], action_valid [L])``;
        the last row's action is invalid (T actions need T+1 observations)."""
        from PIL import Image

        columns = ["frame_index", "action", "action_valid"]
        if frames:
            columns.append(f"observation.images.{ld.CAMERA}")
        table, row = self._table(episode_id, columns)
        index = np.asarray(table["frame_index"].to_pylist())
        if not np.array_equal(index, np.arange(len(index))) or len(index) != row["length"]:
            raise GuardError(f"G-data: {episode_id} frame indices are not 0..L-1")
        actions = np.asarray(table["action"].to_pylist(), np.float32)
        valid = np.asarray(table["action_valid"].to_pylist(), bool)
        if valid[-1] or not valid[:-1].all():
            raise GuardError(f"G-data: {episode_id} action_valid is not all-but-last")
        actions[-1] = 0.0  # the invalid final row is never used as a transition
        images = None
        if frames:
            images = []
            for item in table[f"observation.images.{ld.CAMERA}"].to_pylist():
                with Image.open(io.BytesIO(item["bytes"])) as image:
                    images.append(np.asarray(image).copy())
            images = np.stack(images)
        return images, actions, valid

    def labels(self, episode_id):
        from embodied_jepa import training_labels

        row = self.row(episode_id)
        self.decoded.add(episode_id)
        return training_labels.load(
            self.store.root,
            row["metadata"]["training_labels"],
            groups=("collector", "privileged"),
            acknowledge_privileged_training_labels=True,
        )


# ----- statistics --------------------------------------------------------------------------------
def moments(features) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, np.float64)
    return x.mean(0), x.std(0)


def t1_stats(err, err_occ, err_enc, idx) -> dict:
    out = ic.median_ci(err, idx)
    out["median_cm"] = out["median"]
    out["ratio_to_B_occ"] = ic.paired_ratio(err, err_occ, idx, ic._median)
    out["excess_over_encoded_cm"] = ic.paired_difference(err, err_enc, idx, ic._median)
    out["B_occ_median_cm"] = float(np.median(err_occ))
    return out


def read_source(fits, fold, reference, source):
    """Each root's feature read by the fold readout that held it out (TASK-061's rule)."""
    g_new, norms = ic.cross_gram(np.asarray(source, np.float64), reference)
    return orp.predict_with_fold_readouts(fits, fold, g_new, norms)


# ----- the run -----------------------------------------------------------------------------------
def build_index(store, reader, splits, smoke: bool):
    """Episode table of the read splits: ids, split, session, half, lengths."""
    rows = {r["episode_id"]: r for r in store.manifest["episodes"]}
    train_sessions = sorted({rows[e]["session_id"] for e in splits["train"]})
    halves = ld.session_halves(train_sessions)
    episodes = sorted(splits["train"]) + sorted(splits["val"])
    if smoke:
        keep = {
            h: sorted(s for s in train_sessions if halves[s] == h)[: SMOKE["sessions_per_half"]]
            for h in ld.HALVES
        }
        kept = set(keep["A"]) | set(keep["B"])
        episodes = [e for e in sorted(splits["train"]) if rows[e]["session_id"] in kept]
        episodes += sorted(splits["val"])[: SMOKE["val_episodes"]]
    table = []
    for e in episodes:
        row = rows[e]
        split = "train" if e in set(splits["train"]) else "val"
        table.append(
            {
                "episode_id": e,
                "split": split,
                "session": row["session_id"],
                "half": halves.get(row["session_id"]) if split == "train" else None,
                "kind": row["metadata"]["kind"],
                "seed": int(row["metadata"]["reset_seed"]),
                "length": int(row["length"]),
            }
        )
    return table, halves


def featurise(model, reader, table, clock, smoke):
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
            check_stage_cap(started, ld.FEATURE_SECONDS, "featurisation")
            clock.check("featurisation")
    check_stage_cap(started, ld.FEATURE_SECONDS, "featurisation")
    features = np.concatenate(feats)
    check_finite("features", features)
    return features, np.concatenate(actions), np.asarray(offsets), time.monotonic() - started


def window_arrays(table, offsets, members, horizon, stride):
    """Global start indices, episode positions and sessions of every window of ``members``."""
    counts = [table[i]["length"] - 1 for i in members]
    w = ld.windows(counts, horizon, stride)
    ep = np.asarray(members)[w[:, 0]]
    starts = offsets[ep] + w[:, 1]
    sessions = np.asarray([table[i]["session"] for i in ep])
    return starts, ep, sessions


def gather(features, actions, starts, horizon):
    steps = np.arange(horizon + 1)
    f = features[starts[:, None] + steps]
    a = actions[starts[:, None] + steps[:-1]]
    return f, a


def rollout_errors(model, features, actions, starts, horizon, scale, horizons, action_mode, perm):
    """Per-window normalised squared error at each horizon, and W's normalised predictions at
    the gated horizons (for G1). ``action_mode``: true | zero | wrong (``perm`` of windows)."""
    errors = {h: np.empty(len(starts)) for h in horizons}
    kept = {h: np.empty((len(starts), features.shape[1]), np.float32) for h in ld.GATED_HORIZONS}
    targets_kept = {h: np.empty_like(kept[h]) for h in ld.GATED_HORIZONS}
    for lo in range(0, len(starts), EVAL_CHUNK):
        sl = slice(lo, lo + EVAL_CHUNK)
        f, a = gather(features, actions, starts[sl], horizon)
        if action_mode == "zero":
            a = np.zeros_like(a)
        elif action_mode == "wrong":
            a = gather(features, actions, starts[perm[sl]], horizon)[1]
        predicted = model.predict_features(f[:, 0], np.ascontiguousarray(a, np.float32))
        check_finite("predictions", predicted)
        for h in horizons:
            errors[h][sl] = ld.normalized_sq_error(predicted[:, h - 1], f[:, h], scale)
        for h in ld.GATED_HORIZONS:
            kept[h][sl] = predicted[:, h - 1] / scale
            targets_kept[h][sl] = f[:, h] / scale
    return errors, kept, targets_kept


def val_criterion(model, features, actions, starts, scale, arm):
    total, count = 0.0, 0
    for lo in range(0, len(starts), EVAL_CHUNK):
        f, a = gather(features, actions, starts[lo : lo + EVAL_CHUNK], ld.TRAIN_HORIZON)
        if arm == "N":
            a = np.zeros_like(a)
        predicted = model.predict_features(f[:, 0], np.ascontiguousarray(a, np.float32))
        err = ld.normalized_sq_error(predicted, f[:, 1:], scale)  # [n, 16]
        total += float(err.sum())
        count += err.size
    value = total / count
    if not np.isfinite(value):
        raise GuardError("G-finite: non-finite val criterion")
    return value


def train_one(ctx, arm, seed, half, clock):
    """Train, select on val, save; then evaluate on the other half (E-all, E-post, val roots)."""
    import torch

    from embodied_jepa.models.frozen_encoder import frozen_encoder_model

    started = time.monotonic()
    table, features, actions, offsets = ctx["table"], ctx["features"], ctx["actions"], ctx["offs"]
    members = [i for i, e in enumerate(table) if e["half"] == half]
    member_ids = [table[i]["episode_id"] for i in members]
    cls = frozen_encoder_model(ld.BACKEND)
    model = cls(
        ctx["schema"],
        device=ctx["device"],
        seed=seed,
        config=ld.MODEL_CONFIG,
        metadata={
            "protocol": ld.PROTOCOL,
            "dataset_manifest_sha256": ld.DATASET_MANIFEST_SHA256,
            "halves_sha256": ctx["halves_sha256"],
            "arm": arm,
            "half": half,
        },
    )
    frame_rows = np.concatenate(
        [np.arange(offsets[i], offsets[i] + table[i]["length"]) for i in members]
    )
    mean, std = moments(features[frame_rows])
    model.fit_frozen_feature_normalization(mean, std, training_episode_ids=member_ids)
    windows = ld.training_windows([table[i]["length"] - 1 for i in members])
    starts = offsets[np.asarray(members)[windows[:, 0]]] + windows[:, 1]
    rng = np.random.default_rng(ld.sampler_seed(seed, half, arm))
    updates, every = ctx["updates"], ctx["select_every"]
    curve, losses, best = [], [], (np.inf, 0, None)
    for step in range(1, updates + 1):
        pick = starts[rng.integers(0, len(starts), ld.BATCH_SIZE)]
        f, a = gather(features, actions, pick, ld.TRAIN_HORIZON)
        if arm == "N":
            a = np.zeros_like(a)
        metrics = model.train_step_features(f, np.ascontiguousarray(a, np.float32))
        if not np.isfinite(metrics["loss"]):
            raise GuardError("G-finite: non-finite training loss")
        if step % 100 == 0:
            losses.append([step, metrics["loss"]])
            check_stage_cap(started, ld.PER_RUN_SECONDS, f"model {arm}-s{seed}-{half}")
            clock.check(f"model {arm}-s{seed}-{half}")
        if step % every == 0:
            value = val_criterion(model, features, actions, ctx["val_starts"], ctx["scale"], arm)
            curve.append([step, value])
            if value < best[0]:
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best = (value, step, state)
    model.load_state_dict({k: v.to(ctx["device"]) for k, v in best[2].items()})
    path = ctx["checkpoints"] / f"{arm}-s{seed}-{half}.pt"
    model.save(path)
    check_stage_cap(started, ld.PER_RUN_SECONDS, f"model {arm}-s{seed}-{half}")
    record = {
        "arm": arm,
        "seed": seed,
        "half": half,
        "training_episodes": len(member_ids),
        "training_episodes_sha256": sha256_bytes(json.dumps(sorted(member_ids)).encode()),
        "training_windows": int(len(starts)),
        "normalisation_sha256": array_sha256(np.stack([mean, std])),
        "val_curve": curve,
        "selected_update": int(best[1]),
        "selected_val_criterion": float(best[0]),
        "losses": losses,
        "checkpoint": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "checkpoint_sha256": sha256_file(path),
        "optimizer_in_checkpoint": "final (weights are the selected update's)",
        "train_seconds": time.monotonic() - started,
        "frozen_encoder_digest": model.frozen_encoder_digest,
        "implementation_sha256": model.implementation_sha256,
        "torch": torch.__version__,
    }
    # --- evaluation of the held-out half (E-all), its roots (E-post) and the val roots ---
    ev = ctx["eval"][ld.other(half)]
    scale = ctx["scale"]
    out = {"record": record}
    modes = ("true", "wrong", "zero") if arm == "W" else ("zero",)
    for mode in modes:
        errs, kept, targets = rollout_errors(
            model,
            features,
            actions,
            ev["starts"],
            ld.TRAIN_HORIZON,
            scale,
            EALL_HORIZONS,
            mode,
            ev["perm"],
        )
        out[f"eall_{mode}"] = errs
        if arm == "W" and mode == "true":
            out["eall_kept"], out["eall_targets"] = kept, targets
    post = ctx["post"][ld.other(half)]
    for mode in modes:
        a = post["actions"]
        if mode == "zero":
            a = np.zeros_like(a)
        elif mode == "wrong":
            a = a[post["perm"]]
        predicted = model.predict_features(post["start"], np.ascontiguousarray(a, np.float32))
        check_finite("E-post predictions", predicted)
        out[f"post_{mode}"] = predicted[:, [h - 1 for h in ALL_HORIZONS]]
    val = ctx["val_roots"]
    a = val["actions"] if arm == "W" else np.zeros_like(val["actions"])
    out["val_pred"] = model.predict_features(val["start"], np.ascontiguousarray(a, np.float32))[
        :, [h - 1 for h in ALL_HORIZONS]
    ]
    record["seconds_with_evaluation"] = time.monotonic() - started
    del model
    return out


def run(output: Path, checkpoints: Path, *, smoke: bool = False) -> dict:
    output, checkpoints = Path(output), Path(checkpoints)
    for path in (output, checkpoints):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    output.mkdir(parents=True)
    checkpoints.mkdir(parents=True)
    clock = Clock()
    report = {
        "protocol": ld.PROTOCOL,
        "task": ld.TASK,
        "status": "smoke (not a row)" if smoke else "gated run",
        "learned_apple_to_plate_successes": 0,
        "exemption_spent": False,
        "control_formulation": "none preregistered or implied",
        "test_split_decoded": False,
        "stages": {},
    }
    reader = None
    try:
        report |= _run(report, output, checkpoints, clock, smoke)
        reader = report.pop("_reader", None)
    except BaseException as error:  # noqa: BLE001 -- a crash is V, and the report is written
        reader = report.pop("_reader", None)
        report["outcome"] = "V"
        report["decision"] = ld.decide(void=True)
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
        write_report(output / "report.json", report)
        log(f"report written: outcome {report.get('outcome')}")
    return report


def _run(report, output, checkpoints, clock, smoke):
    # ----- 1. preflight: every pinned file is hashed and recorded before anything is imported --
    manifest = json.loads(MANIFEST.read_text())
    stages = report["stages"]
    report["pinned_hashes_at_preflight"] = check_pins(manifest["hashes"])

    import torch

    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa.data import DatasetStore
    from embodied_jepa.models.frozen_encoder import frozen_encoder_model

    torch.set_num_threads(ld.FEATURE_THREADS)
    dirty = tracked_tree_dirty()
    report["revision"], report["tracked_tree_dirty"] = revision(), bool(dirty)
    if not smoke:
        check_clean(dirty)
    mps = bool(torch.backends.mps.is_available())
    report["environment"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "mps_available": mps,
    }
    import transformers

    report["environment"]["transformers"] = transformers.__version__
    check_device(mps)
    store = DatasetStore(ROOT / ld.DATASET)  # verifies every recorded hash
    splits = store.manifest["splits"]
    rows = {r["episode_id"]: r for r in store.manifest["episodes"]}
    sessions = {
        k: len({rows[e]["session_id"] for e in splits[k]}) for k in ("train", "val", "test")
    }
    episodes = {k: len(splits[k]) for k in ("train", "val", "test")}
    check_data(store.manifest_hash, sessions, episodes)
    reader = TrainValReader(store, set(splits["train"]) | set(splits["val"]))
    report["_reader"] = reader
    table, halves = build_index(store, reader, splits, smoke)
    halves_sha = ld.halves_sha256(halves)
    check_halves(halves, manifest["halves"]["sha256"])
    training = {h: [e["episode_id"] for e in table if e["half"] == h] for h in ld.HALVES}
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
        check_folds(fold_hash, manifest["evaluation"]["probe"]["fold_assignment_sha256"])
    report["data"]["probe_fold_assignment_sha256"] = fold_hash
    state_schema = store.state_schema  # validated only; the models ignore proprioception
    featurizer = frozen_encoder_model(ld.BACKEND)(
        state_schema, device="cpu", config=ld.MODEL_CONFIG
    )
    check_weights(
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
    anchor_frames = np.stack(
        [reader.episode(r["episode_id"])[0][ld.DECISION_FRAME] for r in anchor_roots]
    )
    anchor = pe.features(featurizer._frozen_module, anchor_frames)["cls"]
    again = pe.features(featurizer._frozen_module, anchor_frames)["cls"]
    frames_sha = array_sha256(anchor_frames)
    features_sha = array_sha256(np.asarray(anchor, np.float64).reshape(len(anchor), -1))
    check_anchor(frames_sha, features_sha, manifest["data"])
    check_repro(anchor, again)
    anchor_by_seed = {int(r["seed"]): anchor[k] for k, r in enumerate(anchor_roots)}
    report["anchor"] = {
        "frames_sha256": frames_sha,
        "features_sha256": features_sha,
        "roots": len(anchor_roots),
        "G_repro": "bit-identical",
    }
    stages["anchor"] = clock.elapsed()
    log("anchor passed")
    # ----- 3. labels ---------------------------------------------------------------------------
    targets, label_diff = {}, 0.0
    for i in train_roots + val_roots:
        e = table[i]
        labels = reader.labels(e["episode_id"])
        check_look_labels(labels["collector__phase_index"], e["episode_id"])
        apple = np.asarray(labels["privileged__apple_position_world"], np.float64)[:, :2]
        truth = per_root[e["seed"]]["apple_xy"]
        label_diff = max(label_diff, check_apple_label(apple[0], truth, e["episode_id"]))
        frames_needed = ld.DECISION_FRAME + max(ALL_HORIZONS)
        if len(apple) <= frames_needed:
            raise GuardError(f"G-labels: {e['episode_id']} is shorter than frame {frames_needed}")
        targets[i] = {
            "xy": {h: apple[ld.DECISION_FRAME + h] for h in (0, *ALL_HORIZONS)},
            "moved_m": {
                h: float(np.linalg.norm(apple[ld.DECISION_FRAME + h] - apple[0]))
                for h in ALL_HORIZONS
            },
            "occluded": bool(per_root[e["seed"]]["reset_occluded"]),
        }
    report["labels"] = {"apple_label_max_abs_m": label_diff, "roots": len(targets)}
    stages["labels"] = clock.elapsed()
    # ----- 4. features -------------------------------------------------------------------------
    features, actions, offsets, feature_seconds = featurise(featurizer, reader, table, clock, smoke)
    np.save(output / "features.npy", features)
    cache_rows = [features[offsets[i] + ld.DECISION_FRAME] for i in train_roots + val_roots]
    cache_anchor = [anchor_by_seed[table[i]["seed"]] for i in train_roots + val_roots]
    report["features"] = {
        "frames": int(len(features)),
        "sha256": array_sha256(features),
        "file_sha256": sha256_file(output / "features.npy"),
        "seconds": feature_seconds,
        "cache_vs_anchor_max_abs": check_cache(cache_rows, np.stack(cache_anchor)),
    }
    train_rows = np.concatenate(
        [
            np.arange(offsets[i], offsets[i] + e["length"])
            for i, e in enumerate(table)
            if e["split"] == "train"
        ]
    )
    scale = np.maximum(features[train_rows].astype(np.float64).std(0), ld.METRIC_FLOOR_STD)
    report["features"]["metric_scale_sha256"] = array_sha256(scale)
    del featurizer
    stages["features"] = clock.elapsed()
    log("features done")
    # ----- evaluation sets ----------------------------------------------------------------------
    val_members = [i for i, e in enumerate(table) if e["split"] == "val"]
    val_starts, _, _ = window_arrays(table, offsets, val_members, ld.TRAIN_HORIZON, ld.EVAL_STRIDE)
    evalsets, posts = {}, {}
    for half in ld.HALVES:
        members = [i for i, e in enumerate(table) if e["half"] == half]
        starts, ep, sess = window_arrays(table, offsets, members, ld.TRAIN_HORIZON, ld.EVAL_STRIDE)
        perm = ld.cross_session_shuffle(sess, np.random.default_rng(ld.SHUFFLE_SEED))
        evalsets[half] = {"starts": starts, "sessions": sess, "perm": perm}
        roots = [i for i in train_roots if table[i]["half"] == half]
        start = features[offsets[roots] + ld.DECISION_FRAME]
        acts = np.stack(
            [
                actions[offsets[i] + ld.DECISION_FRAME : offsets[i] + ld.DECISION_FRAME + 64]
                for i in roots
            ]
        )
        rperm = ld.cross_session_shuffle(
            np.asarray([table[i]["session"] for i in roots]), np.random.default_rng(ld.SHUFFLE_SEED)
        )
        posts[half] = {"roots": roots, "start": start, "actions": acts, "perm": rperm}
    val_root_start = features[offsets[val_roots] + ld.DECISION_FRAME]
    val_root_actions = np.stack(
        [
            actions[offsets[i] + ld.DECISION_FRAME : offsets[i] + ld.DECISION_FRAME + 64]
            for i in val_roots
        ]
    )
    ctx = {
        "table": table,
        "features": features,
        "actions": actions,
        "offs": offsets,
        "schema": state_schema,
        "device": ld.DEVICE,
        "halves_sha256": halves_sha,
        "checkpoints": checkpoints,
        "scale": scale,
        "val_starts": val_starts,
        "eval": evalsets,
        "post": posts,
        "val_roots": {"start": val_root_start, "actions": val_root_actions},
        "updates": SMOKE["updates"] if smoke else ld.UPDATES,
        "select_every": SMOKE["select_every"] if smoke else ld.SELECT_EVERY,
    }
    report["evaluation_sets"] = {
        "val_windows": int(len(val_starts)),
        "eall_windows": {h: int(len(evalsets[h]["starts"])) for h in ld.HALVES},
        "epost_roots": {h: len(posts[h]["roots"]) for h in ld.HALVES},
        "val_roots": len(val_roots),
    }
    # ----- 5. training + per-model evaluation ---------------------------------------------------
    seeds = ld.MODEL_SEEDS[:1] if smoke else ld.MODEL_SEEDS
    results, models = {}, []
    for seed in seeds:
        for arm in ld.ARMS:
            for half in ld.HALVES:
                log(f"training {arm}-s{seed}-{half}")
                out = train_one(ctx, arm, seed, half, clock)
                results[(arm, seed, half)] = out
                models.append(out["record"])
                log(
                    f"done {arm}-s{seed}-{half}: selected {out['record']['selected_update']}, "
                    f"{out['record']['train_seconds']:.0f} s"
                )
    report["models"] = models
    stages["training"] = clock.elapsed()
    # ----- 6. statistics ------------------------------------------------------------------------
    report |= statistics(ctx, results, seeds, train_roots, val_roots, targets, fold, smoke)
    stages["statistics"] = clock.elapsed()
    # ----- G-hash again, then the decision -------------------------------------------------------
    report["pinned_hashes_at_end"] = check_pins(manifest["hashes"])
    if not smoke:
        check_clean(tracked_tree_dirty())
    clock.check("the decision")
    if smoke:
        report["outcome"] = "smoke (not a row)"
    else:
        report["decision"] = ld.decide(
            void=False, seeds={s: report["gates"][str(s)]["seed"] for s in seeds}
        )
        report["outcome"] = report["decision"]["outcome"]
    report["_reader"] = reader
    return report


def statistics(ctx, results, seeds, train_roots, val_roots, targets, fold, smoke) -> dict:
    table, features, offsets = ctx["table"], ctx["features"], ctx["offs"]
    scale = ctx["scale"]
    n = len(train_roots)
    # --- probes: fitted on encoded frame 8 + h of the train roots (10-fold CV) ---
    xy = {h: np.stack([targets[i]["xy"][h] for i in train_roots]) for h in (0, *ALL_HORIZONS)}
    if smoke:
        noise = np.random.default_rng(0)
        xy = {h: v.mean(0) + noise.normal(0, 0.017, v.shape) for h, v in xy.items()}
    occluded = np.array([targets[i]["occluded"] for i in train_roots])
    idx = ic.bootstrap_indices(n)
    encoded = {
        h: features[offsets[train_roots] + ld.DECISION_FRAME + h].astype(np.float64)
        for h in (0, *ALL_HORIZONS)
    }
    probes = {}
    for h in (0, *ALL_HORIZONS):
        g, diag = ic.gram(encoded[h])
        pred, fits, selections = ic.nested_cv(g, diag, xy[h], fold)
        occ = ic.prior_predictions(xy[h], np.ones(n), occluded, fold)["B_occ"]
        probes[h] = {"pred": pred, "fits": fits, "selections": selections, "occ": occ}
    # position of every train root inside its half's E-post arrays
    where = {}
    for half in ld.HALVES:
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
        hat = read_source(p["fits"], fold, encoded[h if probe_h is None else probe_h], source)
        err = ic.xy_error_cm(hat, xy[h])
        err_enc = ic.xy_error_cm(probes[h]["pred"], xy[h])
        err_occ = ic.xy_error_cm(probes[h]["occ"], xy[h])
        return t1_stats(err, err_occ, err_enc, idx), err

    out_read, per_root_err = {"encoded": {}}, {}
    for h in ALL_HORIZONS:
        err_enc = ic.xy_error_cm(probes[h]["pred"], xy[h])
        err_occ = ic.xy_error_cm(probes[h]["occ"], xy[h])
        out_read["encoded"][h] = t1_stats(err_enc, err_occ, err_enc, idx) | {
            "selections": probes[h]["selections"]
        }
        stats, err = readability(encoded[0], h)
        out_read.setdefault("copy_last", {})[h] = stats
        per_root_err[("copy_last", h)] = err
    gates, eall_report, post_report, val_report = {}, {}, {}, {}
    # E-all pools both halves, each predicted by the other half's model; sessions are clusters.
    sess = np.concatenate([ctx["eval"][h]["sessions"] for h in ld.HALVES])
    idx_c = ld.cluster_bootstrap_indices(len(np.unique(sess)))
    copy = {}
    for h in EALL_HORIZONS:
        parts = []
        for half in ld.HALVES:
            f, _ = gather(features, ctx["actions"], ctx["eval"][half]["starts"], h)
            parts.append(ld.normalized_sq_error(f[:, 0], f[:, h], scale))
        copy[h] = np.concatenate(parts)
    for seed in seeds:
        s = str(seed)

        def pooled(arm, mode, h, seed=seed):
            return np.concatenate(
                [results[(arm, seed, ld.other(half))][f"eall_{mode}"][h] for half in ld.HALVES]
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
            if h in ld.GATED_HORIZONS:
                pred = np.concatenate(
                    [results[("W", seed, ld.other(half))]["eall_kept"][h] for half in ld.HALVES]
                )
                enc = np.concatenate(
                    [results[("W", seed, ld.other(half))]["eall_targets"][h] for half in ld.HALVES]
                )
                eall[h]["collapse_W"] = ld.collapse_statistics(pred, enc)
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
        post_report[s] = post
        # --- gates ---
        horizons = {}
        for h in ld.GATED_HORIZONS:
            e = eall[h]
            horizons[h] = {
                "G1": ld.g1_passes(e["collapse_W"]),
                "G2": ld.g2_passes(e["W_over_copy_last"]),
                "G3": ld.g3_passes(e["W_over_N"]),
                # an undefined (zero-denominator) resample counts as float max, which would
                # otherwise help a lower bound: G4 then does not pass
                "G4": ld.g4_passes(e["wrong_over_W"], e["zero_over_W"])
                and e["wrong_over_W"]["undefined_resamples"] == 0
                and e["zero_over_W"]["undefined_resamples"] == 0,
                "G5": ld.g5_passes(post[h]["W"]),
            }
        gates[s] = {"per_horizon": horizons, "seed": ld.seed_gates(horizons)}
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
                "encoded": features[offsets[val_roots] + ld.DECISION_FRAME + h],
                "copy_last": ctx["val_roots"]["start"],
            }
            for half in ld.HALVES:
                sources[f"W_{half}"] = results[("W", seed, half)]["val_pred"][:, hidx[h]]
                sources[f"N_{half}"] = results[("N", seed, half)]["val_pred"][:, hidx[h]]
            for name, x in sources.items():
                g_rt, norms = ic.cross_gram(np.asarray(x, np.float64), encoded[h])
                hat = readout.predict(g_rt, norms)
                err = ic.xy_error_cm(hat, vxy)
                entry[name] = {"median_cm": float(np.median(err)), "mean_cm": float(err.mean())}
            val_report[s][h] = entry
    moved = {
        h: int(sum(targets[i]["moved_m"][h] > APPLE_MOVE_M for i in train_roots + val_roots))
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
        "val_secondary": val_report,
        "gates": gates,
        "apple_moved_more_than_1mm_roots": moved,
        "per_root": per_root,
        "smoke_targets": "seeded noise; no real target reached a readout" if smoke else None,
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
