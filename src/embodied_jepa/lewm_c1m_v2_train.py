"""TASK-077's shared training code: contiguous per-root feature storage, a prefetching batch
sampler and the training loop with val selection (protocol §4.4-§4.5, §12, R17.18).

**The storage fix (R17.18).** The draft's cost probe fancy-indexed 64 windows of 61 frames out of
one flat ``[frames, 24 576]`` float32 table (about 0.7 GB/s). Here each split's features are one
array ``[roots, 65, 24 576]`` (frames 403-467 of a root are contiguous), so a training window
(60 transitions starting at 403-407) is one slice ``features[root, k : k + 61]`` and a batch is 16
contiguous copies into a preallocated buffer. A prefetch thread fills the next batch while the GPU
trains on the current one (a ring of three buffers; the sampler's draws happen in the thread in a
fixed order, so the batches, and with strict CUDA determinism the run, do not depend on timing).

The training loop is new shared code (the runner may not load ``scripts/train_apple_token_
dynamics.py``; ``tests/test_no_runner_imports.py``). It calls the frozen-token model's own
``train_step_features`` and ``predict_features`` unchanged, so no model file changes and no
earlier checkpoint's implementation hash changes. Selection: TASK-065's normalised MSE of the
recursive prediction over steps 1-60 from frame 405 of every val root, at evenly spaced points;
the kept checkpoint is ``run_tools.select_checkpoint(curve, 0.01)`` (the earliest point within
1 % of the minimum, not the raw argmin) and ``run_tools.last_two_triggered`` is recorded as a flag.

NumPy at import; torch is imported inside the functions that need it.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np

from embodied_jepa import lewm_c1m_v2 as lm
from embodied_jepa import run_tools as rt
from embodied_jepa.contracts import ContractError

GuardError = lm.GuardError
START_INDEX = lm.COMMIT_STEP - lm.FRAME_STEPS[0]  # 2: frame 405 in a root's 65 frames
OFFSETS = lm.WINDOW_START_STEPS[1] - lm.WINDOW_START_STEPS[0] + 1  # 5: windows start at 403-407
WINDOW_FRAMES = lm.TRAIN_HORIZON + 1  # 61
VAL_CHUNK = 50  # val roots per predict call (bounds the host copy of the predictions)
RING = 3  # prefetch buffers: the one in use, the one queued, the one being filled
SCHEMA = ("task077.none", "none", "task077_features_v0")


def sha256_array(a) -> str:
    a = np.ascontiguousarray(a)
    digest = hashlib.sha256()
    digest.update(str(a.dtype).encode())
    digest.update(json.dumps(list(a.shape)).encode())
    digest.update(memoryview(a).cast("B"))
    return digest.hexdigest()


# ----- storage ------------------------------------------------------------------------------------
class RootStore:
    """One split's per-root features ``[roots, 65, D]`` (float32, frames 403-467) and executed
    commands ``[roots, 64, 14]`` (float32, steps 403-466), with the root seeds in order."""

    def __init__(self, features, commands, roots):
        self.features, self.commands = features, commands
        self.roots = tuple(int(r) for r in roots)
        n = len(self.roots)
        if features.ndim != 3 or features.shape[:2] != (n, lm.N_FRAMES):
            raise ContractError(f"features must be [{n}, {lm.N_FRAMES}, D]")
        if commands.shape != (n, lm.N_COMMANDS, 14):
            raise ContractError(f"commands must be [{n}, {lm.N_COMMANDS}, 14]")
        if features.dtype != np.float32 or commands.dtype != np.float32:
            raise ContractError("the store holds float32")
        if not features.flags.c_contiguous:
            raise ContractError("each root's frames must be contiguous")

    @property
    def width(self) -> int:
        return int(self.features.shape[2])

    def __len__(self) -> int:
        return len(self.roots)

    def window(self, root: int, offset: int):
        """Frames ``403 + offset ... 463 + offset`` and the 60 commands between them (views)."""
        if not 0 <= int(offset) < OFFSETS:
            raise ContractError("a window starts at 403-407")
        o = int(offset)
        return self.features[root, o : o + WINDOW_FRAMES], self.commands[root, o : o + lm.HORIZON]

    def from_405(self):
        """The E60 / val windows: frame 405, its 60 executed commands, frames 406-465."""
        s = START_INDEX
        return (
            self.features[:, s],
            self.commands[:, s : s + lm.HORIZON],
            self.features[:, s + 1 : s + 1 + lm.HORIZON],
        )

    @classmethod
    def open(cls, folder, split: str, *, mmap: bool = False) -> RootStore:
        folder = Path(folder)
        mode = "r" if mmap else None
        features = np.load(folder / f"features8_{split}.npy", mmap_mode=mode)
        commands = np.load(folder / f"commands_{split}.npy")
        roots = json.loads((folder / f"roots_{split}.json").read_text())
        return cls(features, commands, roots)


def write_store(folder, split: str, features, commands, roots) -> dict:
    """Write one split's store (refuses to overwrite); returns the files' sha256."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {
        "features": folder / f"features8_{split}.npy",
        "commands": folder / f"commands_{split}.npy",
        "roots": folder / f"roots_{split}.json",
    }
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    RootStore(np.ascontiguousarray(features), commands, roots)  # validates the layout
    np.save(paths["features"], np.ascontiguousarray(features, np.float32))
    np.save(paths["commands"], np.asarray(commands, np.float32))
    paths["roots"].write_text(json.dumps([int(r) for r in roots]))
    return {k: sha256_file(p) for k, p in paths.items()}


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def moments(store: RootStore, chunk: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean and std over every kept frame of the split (float64, streamed)."""
    d = store.width
    n, s1, s2 = 0, np.zeros(d), np.zeros(d)
    for lo in range(0, len(store), chunk):
        x = np.asarray(store.features[lo : lo + chunk], np.float64).reshape(-1, d)
        n += len(x)
        s1 += x.sum(0)
        s2 += (x * x).sum(0)
    mean = s1 / n
    std = np.sqrt(np.clip(s2 / n - mean * mean, 0.0, None))
    return mean, std


def moments_file(path, chunk: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """:func:`moments` of a stored ``features8_<split>.npy`` read in plain file chunks, never
    mapped (a memory map of the 9.6 GB train store would count its touched pages in the stage's
    process-tree PSS, G-memory)."""
    with Path(path).open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        if fortran or dtype != np.float32 or len(shape) != 3:
            raise ContractError("a feature store is a C-order float32 [roots, frames, D] array")
        per = int(shape[1]) * int(shape[2])
        d = int(shape[2])
        n, s1, s2 = 0, np.zeros(d), np.zeros(d)
        for lo in range(0, int(shape[0]), chunk):
            rows = min(chunk, int(shape[0]) - lo)
            data = stream.read(rows * per * 4)
            x = np.frombuffer(data, np.float32).astype(np.float64).reshape(-1, d)
            n += len(x)
            s1 += x.sum(0)
            s2 += (x * x).sum(0)
            del data, x
    mean = s1 / n
    return mean, np.sqrt(np.clip(s2 / n - mean * mean, 0.0, None))


# ----- sampling -----------------------------------------------------------------------------------
class WindowSampler:
    """Uniform root, uniform start in 403-407, ``batch`` windows per draw, from
    ``default_rng(SeedSequence([8109, model seed]))``: W and N of a seed draw the same windows."""

    def __init__(self, n_roots: int, model_seed: int, batch: int = lm.BATCH_SIZE):
        self.n, self.batch = int(n_roots), int(batch)
        self.rng = np.random.default_rng(lm.sampler_seed(model_seed))

    def draw(self) -> tuple[np.ndarray, np.ndarray]:
        roots = self.rng.integers(0, self.n, self.batch)
        offsets = self.rng.integers(0, OFFSETS, self.batch)
        return roots, offsets


def gather_into(store: RootStore, roots, offsets, fbuf, abuf) -> None:
    for i, (r, o) in enumerate(zip(roots, offsets, strict=True)):
        f, a = store.window(int(r), int(o))
        fbuf[i] = f
        abuf[i] = a


class Prefetcher:
    """Fills the next batch in a thread while the caller trains on the current one.

    ``next()`` returns ``(features [B, 61, D], commands [B, 60, 14], roots, offsets)``; the arrays
    are ring buffers valid until the following ``next()``. Commands are zeroed for N (``zero``).
    An exception in the thread is raised by the next ``next()``."""

    def __init__(self, store: RootStore, sampler: WindowSampler, *, zero: bool = False):
        self.store, self.sampler, self.zero = store, sampler, bool(zero)
        b = sampler.batch
        self.fbuf = np.empty((RING, b, WINDOW_FRAMES, store.width), np.float32)
        self.abuf = np.empty((RING, b, lm.HORIZON, 14), np.float32)
        self.q: queue.Queue = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.fill_seconds = 0.0
        self.wait_seconds = 0.0
        self.batches = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        slot = 0
        try:
            while not self.stop.is_set():
                roots, offsets = self.sampler.draw()
                started = time.perf_counter()
                gather_into(self.store, roots, offsets, self.fbuf[slot], self.abuf[slot])
                if self.zero:
                    self.abuf[slot][:] = 0.0
                self.fill_seconds += time.perf_counter() - started
                while not self.stop.is_set():
                    try:
                        self.q.put((slot, roots, offsets), timeout=0.1)
                        break
                    except queue.Full:
                        continue
                slot = (slot + 1) % RING
        except BaseException as error:  # noqa: BLE001 - handed to the consumer
            self.q.put(("error", error, None))

    def next(self):
        started = time.perf_counter()
        slot, roots, offsets = self.q.get()
        self.wait_seconds += time.perf_counter() - started
        if slot == "error":
            raise roots
        self.batches += 1
        return self.fbuf[slot], self.abuf[slot], roots, offsets

    def close(self):
        self.stop.set()
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        self.thread.join(timeout=10)


# ----- the model ----------------------------------------------------------------------------------
def build_model(*, seed: int, device: str, metadata: dict):
    from embodied_jepa.contracts import StateSchema
    from embodied_jepa.models.frozen_tokens import frozen_token_model

    schema = StateSchema((SCHEMA[0],), (SCHEMA[1],), SCHEMA[2])
    model = frozen_token_model(lm.BACKEND)(
        schema, device=device, seed=int(seed), config=dict(lm.MODEL_CONFIG), metadata=metadata
    )
    for key, value in lm.MODEL_CONFIG.items():
        if model.config[key] != value:
            raise GuardError(f"G-config: {key} is {model.config[key]!r}, not {value!r}")
    return model


def load_model(path, *, seed: int, metadata: dict, expected_sha256: str | None = None):
    """A trained model on the CPU (every prediction a gate or the closed loop reads, §4.4)."""
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise GuardError(f"G-checkpoint: {path} differs from its recorded sha256")
    model = build_model(seed=seed, device="cpu", metadata=metadata)
    model.load(path)
    model.eval()
    return model


def model_metadata(*, arm: str, seed: int, corpus_sha256: str, moments_sha256: str) -> dict:
    return {
        "protocol": lm.PROTOCOL,
        "arm": str(arm),
        "model_seed": int(seed),
        "corpus_manifest_sha256": str(corpus_sha256),
        "normalisation_sha256": str(moments_sha256),
    }


def val_criterion(model, val: RootStore, scale, arm: str, *, chunk: int = VAL_CHUNK) -> float:
    """TASK-065's normalised MSE of the recursive prediction over steps 1-60 from frame 405 of
    every val root (N: zero commands)."""
    start, commands, targets = val.from_405()
    total, count = 0.0, 0
    for lo in range(0, len(val), chunk):
        a = np.ascontiguousarray(commands[lo : lo + chunk], np.float32)
        if arm == "N":
            a = np.zeros_like(a)
        predicted = model.predict_features(np.asarray(start[lo : lo + chunk]), a)
        d = (
            predicted.astype(np.float64) - np.asarray(targets[lo : lo + chunk], np.float64)
        ) / scale
        total += float((d * d).sum())
        count += d.size
    value = total / count
    if not np.isfinite(value):
        raise GuardError("G-finite: non-finite val criterion")
    return value


def train_model(
    *,
    arm: str,
    seed: int,
    train: RootStore,
    val: RootStore,
    mean,
    std,
    updates: int,
    select_every: int,
    device: str,
    metadata: dict,
    cap_seconds: float,
    check=None,
    log=None,
    probe: bool = False,
) -> tuple[object, dict]:
    """Train W (true commands) or N (zero commands) on the train store, select on val and return
    the model at the kept checkpoint with its record. ``check()`` is called every 100 updates
    (the stage's clock); ``probe`` marks a scale probe (same code path)."""
    import torch

    if arm not in lm.ARMS_TRAINED:
        raise ContractError(f"unknown trained arm {arm!r}")
    if updates % select_every:
        raise ContractError("updates must be a multiple of select_every")
    started = time.monotonic()
    scale = np.maximum(np.asarray(std, np.float64), lm.METRIC_FLOOR_STD)
    model = build_model(seed=seed, device=device, metadata=metadata)
    model.fit_frozen_feature_normalization(
        mean, std, training_episode_ids=tuple(str(r) for r in train.roots)
    )
    sampler = WindowSampler(len(train), seed)
    feeder = Prefetcher(train, sampler, zero=(arm == "N"))
    curve, losses, states = [], [], {}
    step_seconds = []
    sync = torch.cuda.synchronize if str(device).startswith("cuda") else (lambda: None)
    try:
        for step in range(1, int(updates) + 1):
            t0 = time.perf_counter()
            f, a, _roots, _offsets = feeder.next()
            metrics = model.train_step_features(f, a)
            if not np.isfinite(metrics["loss"]):
                raise GuardError("G-finite: non-finite training loss")
            if step % 100 == 0:
                sync()
                losses.append([step, float(metrics["loss"])])
                if time.monotonic() - started > cap_seconds:
                    raise GuardError(f"G-cap: the {arm}-{seed} job exceeded {cap_seconds} s")
                if check is not None:
                    check()
            step_seconds.append(time.perf_counter() - t0)
            if step % select_every == 0:
                value = val_criterion(model, val, scale, arm)
                curve.append([step, value])
                states[step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if log is not None:
                    log(f"{arm}-{seed}: update {step}, val {value:.6f}")
    finally:
        feeder.close()
    sync()
    kept = rt.select_checkpoint(curve, lm.SELECTION_TOLERANCE)
    model.load_state_dict({k: v.to(model.device_name) for k, v in states[kept].items()})
    seconds = np.asarray(step_seconds[10:] if len(step_seconds) > 20 else step_seconds)
    record = {
        "arm": arm,
        "seed": int(seed),
        "updates": int(updates),
        "select_every": int(select_every),
        "val_curve": curve,
        "kept_update": int(kept),
        "kept_val_criterion": float(dict((u, v) for u, v in curve)[kept]),
        "raw_argmin_update": int(min(curve, key=lambda p: p[1])[0]),
        "last_two_triggered": bool(rt.last_two_triggered(curve, lm.SELECTION_TOLERANCE)),
        "losses": losses,
        "train_roots": len(train),
        "train_roots_sha256": hashlib.sha256(json.dumps(list(train.roots)).encode()).hexdigest(),
        "seconds": time.monotonic() - started,
        "per_update_seconds": {
            "median": float(np.median(seconds)) if len(seconds) else None,
            "p95": float(np.percentile(seconds, 95)) if len(seconds) else None,
            "mean": float(np.mean(seconds)) if len(seconds) else None,
            "note": "wall time per update including the wait for the prefetched batch "
            "(selection evaluations excluded); the first 10 updates excluded when > 20",
        },
        "prefetch": {
            "batches": feeder.batches,
            "fill_seconds_total": feeder.fill_seconds,
            "wait_seconds_total": feeder.wait_seconds,
        },
        "probe": bool(probe),
        "device": str(device),
        "implementation_sha256": model.implementation_sha256,
        "frozen_encoder_digest": model.frozen_encoder_digest,
    }
    return model, record


def state_sha256(model) -> str:
    """A digest of the weights (bit-identity check of two runs of one job)."""
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


# ----- the measured gather (R17.18) --------------------------------------------------------------
def measure_gather(store: RootStore, *, seed: int, batches: int = 50) -> dict:
    """The contiguous-slice gather alone (no prefetch, no GPU): seconds per batch of 16."""
    sampler = WindowSampler(len(store), seed)
    fbuf = np.empty((lm.BATCH_SIZE, WINDOW_FRAMES, store.width), np.float32)
    abuf = np.empty((lm.BATCH_SIZE, lm.HORIZON, 14), np.float32)
    roots, offsets = sampler.draw()
    gather_into(store, roots, offsets, fbuf, abuf)  # warm
    started = time.perf_counter()
    for _ in range(int(batches)):
        roots, offsets = sampler.draw()
        gather_into(store, roots, offsets, fbuf, abuf)
    seconds = (time.perf_counter() - started) / int(batches)
    return {
        "seconds_per_batch": seconds,
        "batch_bytes": int(fbuf.nbytes + abuf.nbytes),
        "gb_per_second": (fbuf.nbytes + abuf.nbytes) / seconds / 1e9,
        "batches": int(batches),
    }
