"""TASK-100's corpus reading, featurisation and feature stores (protocol §2-§3, §8.4, §10.2).

* ``corpus_episodes``: the sealed ``so101-play-v1`` episodes by split (every shard's frozen split
  file), checked against the protocol's counts; ``verify_corpus`` checks ``corpus.json`` and every
  shard's manifest hash against TASK-099's manifest.
* ``load_episode``: one episode's frames of both cameras, the executed actions, and the labels
  (cube centre from the sidecar truth in cm, jaw angle from ``observation.state``, the gripper
  target, the phase and a colour-threshold cube visibility per camera). Labels are never model
  inputs.
* ``Featuriser``: the frozen DINOv2 ViT-S/14 (pretrained or its seed-0 random init) on 224 px
  frames (no resize), ImageNet normalisation, a fixed batch of 64 frames per camera, final patch
  tokens pooled to 4 x 4 (g4) and 8 x 8 (g8) per camera.
* ``ProjectionAccumulator`` / ``project_g8``: g8's 384 -> 96 channel projection (TASK-087's form:
  per-dimension standardisation, channel covariance averaged over cameras and grid positions,
  no whitening), fitted on the fit set.
* ``SplitWriter`` / ``FeatureStore``: one split's float16 latents ``[F, tokens, width]`` and
  labels; ``Moments`` the per-dimension standardisation of the flat 12 288-value latent
  (floor 1e-6), fitted on the fit set.

NumPy at import; torch, transformers and the dataset reader are imported where needed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_jepa import so101_wm as sw
from embodied_jepa.contracts import ContractError

SPLITS = ("train", "val", "test")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
STORE_ARRAYS = ("actions", "jaw", "cube", "gripper_target", "phase", "visible")


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_arrays(*arrays) -> str:
    digest = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        digest.update(str(a.dtype).encode())
        digest.update(json.dumps(list(a.shape)).encode())
        digest.update(memoryview(a).cast("B"))
    return digest.hexdigest()


# ----- the corpus --------------------------------------------------------------------------------
def corpus_episodes(corpus, *, check_counts: bool = True) -> dict:
    """``{split: [(shard name, episode id, seed), ...]}`` in shard order, then id order."""
    corpus = Path(corpus)
    shards = sorted(p for p in corpus.iterdir() if p.name.startswith("shard-"))
    out = {s: [] for s in SPLITS}
    for shard in shards:
        splits = json.loads((shard / "meta" / "jepa_splits.json").read_text())["episodes"]
        for split in SPLITS:
            for episode_id in sorted(splits.get(split, [])):
                out[split].append((shard.name, episode_id, int(episode_id.rsplit("-", 1)[1])))
    if check_counts:
        if len(shards) != sw.CORPUS_SHARDS:
            raise ContractError(
                f"G-corpus: expected {sw.CORPUS_SHARDS} shards, found {len(shards)}"
            )
        for split, n in sw.SPLIT_COUNTS.items():
            if len(out[split]) != n:
                raise ContractError(f"G-corpus: {split} has {len(out[split])} episodes, not {n}")
    return out


def select(episodes, *, shards=None, fit_sample: bool = False) -> list:
    out = [e for e in episodes if shards is None or sw.shard_index(e[0]) in shards]
    return [e for e in out if sw.is_fit_sample(e[2])] if fit_sample else out


def verify_corpus(corpus, *, manifest=sw.CORPUS_MANIFEST, pinned=sw.CORPUS_JSON_SHA256) -> dict:
    """§12: ``corpus.json`` against the protocol's sha256 and every shard's manifest hash
    against ``corpus.json`` and TASK-099's manifest (``SchemaDatasetStore`` also re-hashes every
    file of a shard when it opens)."""
    corpus = Path(corpus)
    got = sha256_file(corpus / "corpus.json")
    if pinned is not None and got != pinned:
        raise ContractError(f"G-corpus: corpus.json sha256 {got[:12]} is not TASK-099's")
    record = json.loads((corpus / "corpus.json").read_text())
    pins = {}
    if manifest is not None:
        pins = json.loads(Path(manifest).read_text())["shard_manifest_sha256"]
    checked = {}
    for row in record["shards"]:
        name = f"shard-{int(row['shard']):02d}"
        store = _store(corpus / name)
        if store.manifest_hash != row["manifest_hash"]:
            raise ContractError(f"G-corpus: {name} manifest differs from corpus.json")
        if pins and pins.get(name, pins.get(str(row["shard"]))) != row["manifest_hash"]:
            raise ContractError(f"G-corpus: {name} manifest differs from TASK-099's manifest")
        checked[name] = row["manifest_hash"]
    return {"corpus_json_sha256": got, "pinned": pinned is not None, "shards": checked}


_STORES: dict = {}


def _store(shard_dir):
    from embodied_jepa.schema_data import SchemaDatasetStore

    key = str(shard_dir)
    if key not in _STORES:
        _STORES[key] = SchemaDatasetStore(shard_dir)
    return _STORES[key]


def red_fraction(frames) -> np.ndarray:
    """TASK-099's colour threshold (``measure_so101_play._red_fraction``), per frame."""
    f = np.asarray(frames).astype(np.int16)
    red = (f[..., 0] > 120) & (f[..., 1] < 70) & (f[..., 2] < 70)
    return red.reshape(len(f), -1).mean(1)


VISIBLE_RED_FRACTION = 0.0002  # TASK-099: cube visible when red pixels exceed 0.02 % of a frame


def load_episode(corpus, shard: str, episode_id: str) -> dict:
    """Frames uint8 ``{camera: [T+1, 224, 224, 3]}``, actions float32 [T+1, 6] (last row 0, never
    a transition), cube centre float32 [T+1, 3] (cm, world frame), jaw float32 [T+1] (rad),
    gripper target float32 [T+1] (rad), phase int8 [T+1], visible bool [T+1, 2]."""
    store = _store(Path(corpus) / shard)
    ep = store.read_episode(episode_id)
    meta = ep.metadata
    sidecar = Path(corpus) / shard / meta["sidecar"]
    if sha256_file(sidecar) != meta["sidecar_sha256"]:
        raise ContractError(f"G-corpus: sidecar hash mismatch {episode_id}")
    with np.load(sidecar) as d:
        cube = np.asarray(d["cube_position"], np.float64) * sw.CM_PER_M
        targets = np.asarray(d["targets"], np.float32)
        phase = np.asarray(d["phase"], np.int8)
    frames = {c: ep.observations[c] for c in sw.CAMERAS}
    n = len(frames[sw.CAMERAS[0]])
    if ep.action_schema.dimension != sw.ACTION_DIM or ep.actions.shape != (n - 1, sw.ACTION_DIM):
        raise ContractError(f"G-corpus: action shape in {episode_id}")
    if cube.shape != (n, 3) or targets.shape[0] != n or ep.robot_states.shape[0] != n:
        raise ContractError(f"G-corpus: length mismatch in {episode_id}")
    if ep.state_schema.names[sw.JAW_STATE_INDEX] != "gripper.position":
        raise ContractError("G-corpus: the jaw is not observation.state[5]")
    actions = np.zeros((n, sw.ACTION_DIM), np.float32)
    actions[:-1] = ep.actions
    visible = np.stack([red_fraction(frames[c]) > VISIBLE_RED_FRACTION for c in sw.CAMERAS], 1)
    return {
        "episode_id": episode_id,
        "frames": frames,
        "actions": actions,
        "cube": cube.astype(np.float32),
        "jaw": ep.robot_states[:, sw.JAW_STATE_INDEX].astype(np.float32),
        "gripper_target": targets[:, sw.GRIPPER_CHANNEL].astype(np.float32),
        "phase": phase,
        "visible": visible,
    }


def iter_episodes(corpus, episodes, *, workers: int = 8, ahead: int | None = None):
    """Yield loaded episodes in order, decoding in a bounded process pool (TASK-087's form)."""
    jobs = [(str(corpus), shard, eid) for shard, eid, _ in episodes]
    if workers <= 1:
        for job in jobs:
            yield load_episode(*job)
        return
    import multiprocessing as mp
    from collections import deque

    ahead = 2 * int(workers) if ahead is None else int(ahead)
    with mp.get_context("spawn").Pool(workers) as pool:
        pending: deque = deque()
        queue = iter(jobs)
        for job in queue:
            pending.append(pool.apply_async(load_episode, job))
            if len(pending) >= ahead:
                break
        while pending:
            result = pending.popleft().get()
            job = next(queue, None)
            if job is not None:
                pending.append(pool.apply_async(load_episode, job))
            yield result


def episode_frames(corpus, episodes) -> list[int]:
    """Frame counts from the shard manifests (no decoding)."""
    lengths = {}
    for shard in sorted({s for s, _, _ in episodes}):
        manifest = json.loads((Path(corpus) / shard / "meta" / "jepa_manifest.json").read_text())
        for row in manifest["episodes"]:
            lengths[(shard, row["episode_id"])] = int(row["length"])
    return [lengths[(s, e)] for s, e, _ in episodes]


# ----- the frozen encoder ------------------------------------------------------------------------
def preprocess_224(frames):
    """uint8 [N, 224, 224, 3] RGB -> float32 [N, 3, 224, 224], ImageNet-normalised (no resize)."""
    import torch

    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (224, 224, 3):
        raise ContractError(f"expected uint8 [N, 224, 224, 3], got {frames.dtype} {frames.shape}")
    x = torch.from_numpy(np.array(frames, copy=True)).permute(0, 3, 1, 2).float() / 255.0
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def pool(tokens, grid: int):
    """Tokens ``[N, 256, 384]`` (torch, any device) -> float64 block means ``[N, grid^2, 384]``."""
    k = sw.NATIVE_GRID // grid
    t = tokens.double().reshape(len(tokens), grid, k, grid, k, sw.RAW_WIDTH)
    return t.mean((2, 4)).reshape(len(tokens), grid * grid, sw.RAW_WIDTH)


class Featuriser:
    """Frozen DINOv2 ViT-S/14, float32, eager attention, a fixed batch per camera; returns per
    grid ``[N, 2 * grid^2, 384]`` float32 pooled tokens, wrist first, then overview."""

    def __init__(
        self,
        encoder: str = "pretrained",
        device: str = "cuda",
        batch: int = sw.FEATURE_BATCH,
        module=None,
    ):
        import torch

        from embodied_jepa import pretrained_encoder as pe

        if encoder not in sw.ENCODERS:
            raise ContractError(f"encoder must be one of {sw.ENCODERS}")
        self.torch, self.encoder = torch, encoder
        self.device, self.batch = device, int(batch)
        if module is None:
            module = pe.load_pretrained() if encoder == "pretrained" else pe.random_init()
        self.digest = pe.weights_digest(module)
        self.model = module.to(device).eval()

    def pooled(self, frames: dict, grids=("g4", "g8")) -> dict:
        torch = self.torch
        out = {g: [] for g in grids}
        with torch.no_grad():
            for camera in sw.CAMERAS:
                x = frames[camera]
                parts = {g: [] for g in grids}
                for lo in range(0, len(x), self.batch):
                    px = preprocess_224(x[lo : lo + self.batch]).to(self.device)
                    tok = self.model(pixel_values=px).last_hidden_state[:, 1:]
                    if tok.shape[1:] != (sw.NATIVE_GRID**2, sw.RAW_WIDTH):
                        raise ContractError(f"unexpected token grid {tuple(tok.shape)}")
                    for g in grids:
                        parts[g].append(pool(tok, sw.GRIDS[g]["pool"]).float().cpu().numpy())
                for g in grids:
                    out[g].append(np.concatenate(parts[g]))
        result = {g: np.concatenate(out[g], axis=1) for g in grids}
        for value in result.values():
            if not np.isfinite(value).all():
                raise ContractError("G-finite: non-finite DINOv2 features")
        return result


def g4_latents(pooled_g4) -> np.ndarray:
    x = np.asarray(pooled_g4, np.float32)
    if x.shape[1:] != (sw.GRIDS["g4"]["tokens"], sw.RAW_WIDTH):
        raise ContractError("g4 tokens must be [N, 32, 384]")
    return x


# ----- g8's projection (pass A) ------------------------------------------------------------------
class ProjectionAccumulator:
    """Per-dimension sums and per-slot second moments of g8's pooled tokens (float64)."""

    def __init__(self):
        p, w = sw.GRIDS["g8"]["tokens"], sw.RAW_WIDTH
        self.n = 0
        self.s1 = np.zeros((p, w))
        self.s2 = np.zeros((p, w))
        self.m = np.zeros((p, w, w))

    def add(self, x):
        x = np.asarray(x, np.float64)
        if x.shape[1:] != self.s1.shape:
            raise ContractError("g8 tokens must be [N, 128, 384]")
        self.n += len(x)
        self.s1 += x.sum(0)
        self.s2 += (x * x).sum(0)
        xt = np.ascontiguousarray(x.transpose(1, 0, 2))  # [p, n, w]
        self.m += np.matmul(xt.transpose(0, 2, 1), xt)

    def fit(self, k: int = sw.GRIDS["g8"]["k"]) -> dict:
        if self.n < 2:
            raise ContractError("the fit set is empty")
        mean = self.s1 / self.n
        std = np.maximum(
            np.sqrt(np.clip(self.s2 / self.n - mean * mean, 0.0, None)), sw.FEATURE_STD_FLOOR
        )
        cov = np.zeros((sw.RAW_WIDTH, sw.RAW_WIDTH))
        for p in range(len(mean)):
            c = self.m[p] / self.n - np.outer(mean[p], mean[p])
            cov += c / np.outer(std[p], std[p])
        cov /= len(mean)
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        signs = np.sign(vectors[np.abs(vectors).argmax(0), np.arange(vectors.shape[1])])
        vectors = vectors * np.where(signs == 0, 1.0, signs)
        return {
            "mean": mean,
            "std": std,
            "components": vectors[:, :k].copy(),
            "eigenvalues": values,
            "explained": float(values[:k].sum() / values.sum()),
            "frames": int(self.n),
        }


def project_g8(pooled_g8, projection: dict) -> np.ndarray:
    """Pooled float32 [N, 128, 384] -> standardised and projected float32 [N, 128, 96]."""
    x = (np.asarray(pooled_g8, np.float64) - projection["mean"]) / projection["std"]
    return (x @ projection["components"]).astype(np.float32)


def projection_sha256(projection: dict) -> str:
    return sha256_arrays(
        *(np.asarray(projection[k], np.float64) for k in ("mean", "std", "components"))
    )


def save_projection(path, projection: dict) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    np.savez(path, **{k: np.asarray(v) for k, v in projection.items()})
    return projection_sha256(projection)


def load_projection(path, expected_sha256: str | None = None) -> dict:
    with np.load(path) as d:
        projection = {k: d[k] for k in d.files}
    for key in ("explained", "frames"):
        projection[key] = projection[key].item()
    if expected_sha256 is not None and projection_sha256(projection) != expected_sha256:
        raise ContractError("G-features: the projection differs from its recorded sha256")
    return projection


# ----- a split's store (pass B) ------------------------------------------------------------------
class SplitWriter:
    """Writes one split's arrays incrementally: latents float16 [F, tokens, width] and labels."""

    def __init__(self, folder, split: str, total_frames: int, grid: str):
        self.folder = Path(folder)
        self.split = split
        self.folder.mkdir(parents=True, exist_ok=True)
        spec = sw.GRIDS[grid]
        self.paths = {"latents": self.folder / f"latents_{split}.npy"}
        self.paths |= {k: self.folder / f"{k}_{split}.npy" for k in STORE_ARRAYS}
        self.paths["episodes"] = self.folder / f"episodes_{split}.json"
        for p in self.paths.values():
            if p.exists():
                raise FileExistsError(f"refusing to overwrite {p}")
        f = int(total_frames)
        self.latents = np.lib.format.open_memmap(
            self.paths["latents"],
            mode="w+",
            dtype=np.float16,
            shape=(f, spec["tokens"], spec["width"]),
        )
        self.arrays = {
            "actions": np.zeros((f, sw.ACTION_DIM), np.float32),
            "jaw": np.zeros(f, np.float32),
            "cube": np.zeros((f, 3), np.float32),
            "gripper_target": np.zeros(f, np.float32),
            "phase": np.zeros(f, np.int8),
            "visible": np.zeros((f, len(sw.CAMERAS)), np.bool_),
        }
        self.table: list[dict] = []
        self.offset = 0

    def add(self, episode: dict, latents, *, shard: str, seed: int):
        n = len(latents)
        lo, hi = self.offset, self.offset + n
        if hi > len(self.arrays["jaw"]):
            raise ContractError("more frames than declared")
        self.latents[lo:hi] = np.asarray(latents).astype(np.float16)
        for k in STORE_ARRAYS:
            self.arrays[k][lo:hi] = episode[k]
        self.table.append(
            {
                "episode_id": episode["episode_id"],
                "shard": shard,
                "seed": int(seed),
                "offset": lo,
                "frames": n,
            }
        )
        self.offset = hi

    def close(self) -> dict:
        if self.offset != len(self.arrays["jaw"]):
            raise ContractError(f"wrote {self.offset} of {len(self.arrays['jaw'])} frames")
        self.latents.flush()
        del self.latents
        for k, v in self.arrays.items():
            np.save(self.paths[k], v)
        self.paths["episodes"].write_text(json.dumps(self.table))
        return {k: sha256_file(p) for k, p in self.paths.items()}


def verify_store_files(folder, files_sha256: dict, splits) -> dict:
    """Re-hash a feature store's files against its feature report."""
    folder = Path(folder)
    out = {}
    for split in splits:
        for key, want in files_sha256[split].items():
            name = f"episodes_{split}.json" if key == "episodes" else f"{key}_{split}.npy"
            got = sha256_file(folder / name)
            if got != want:
                raise ContractError(f"G-features: {name} differs from the feature report")
            out[name] = got
    return out


class FeatureStore:
    """One split: latents float16 [F, tokens, width] (memory-mapped or loaded), labels and the
    episode table. Latents are returned flat and standardised by ``moments`` (float32)."""

    def __init__(self, folder, split: str, *, mmap: bool = True, moments: Moments | None = None):
        folder = Path(folder)
        self.split = split
        self.latents = np.load(folder / f"latents_{split}.npy", mmap_mode="r" if mmap else None)
        for k in STORE_ARRAYS:
            setattr(self, k, np.load(folder / f"{k}_{split}.npy"))
        self.table = json.loads((folder / f"episodes_{split}.json").read_text())
        self.frames = np.array([r["frames"] for r in self.table], np.int64)
        self.offsets = np.array([r["offset"] for r in self.table], np.int64)
        self.seeds = np.array([r["seed"] for r in self.table], np.int64)
        self.shards = np.array([sw.shard_index(r["shard"]) for r in self.table], np.int64)
        f = len(self.jaw)
        if self.latents.ndim != 3 or self.latents.shape[0] != f or int(self.frames.sum()) != f:
            raise ContractError(f"G-features: {split} is an inconsistent store")
        if self.latents.shape[1] * self.latents.shape[2] != sw.LATENT_DIM:
            raise ContractError("G-features: the latent is not 12 288 values")
        self.moments = moments

    def subset(self, episode_mask) -> FeatureView:
        return FeatureView(self, np.flatnonzero(np.asarray(episode_mask, bool)))

    def all(self) -> FeatureView:
        return FeatureView(self, np.arange(len(self.table)))

    def raw(self, index) -> np.ndarray:
        index = np.asarray(index, np.int64)
        flat = index.ravel()
        order = np.argsort(flat, kind="stable")
        values = np.empty((len(flat), sw.LATENT_DIM), np.float32)
        values[order] = np.asarray(self.latents[flat[order]], np.float32).reshape(len(flat), -1)
        return values.reshape(*index.shape, sw.LATENT_DIM)

    def standardised(self, index) -> np.ndarray:
        if self.moments is None:
            raise ContractError("set the fit-set moments before reading latents")
        return self.moments.apply(self.raw(index))

    def gather(self, starts, frames: int = sw.WINDOW_FRAMES) -> dict:
        """Windows from global start indices: standardised latents [B, frames, 12 288] float32
        and the executed actions [B, frames - 1, 6]."""
        idx = np.asarray(starts, np.int64)[:, None] + np.arange(frames)
        return {"latents": self.standardised(idx), "actions": self.actions[idx[:, :-1]]}


class FeatureView:
    """A subset of a store's episodes: their windows and roots (global indices)."""

    def __init__(self, store: FeatureStore, episodes):
        self.store = store
        self.episodes = np.asarray(episodes, np.int64)
        self.frames = store.frames[self.episodes]
        self.offsets = store.offsets[self.episodes]
        self.seeds = store.seeds[self.episodes]

    def window_starts(self, horizon: int = sw.TRAIN_HORIZON) -> np.ndarray:
        return sw.window_starts(self.frames, self.offsets, horizon)

    def roots(self):
        """(global start index, index into this view's episodes) of every root (§7.1)."""
        return sw.roots(self.frames, self.offsets)

    def frame_index(self, stride: int = 1):
        """Every ``stride``-th frame of each episode: (global index, view episode index)."""
        idx, ep = [], []
        for e, (n, o) in enumerate(zip(self.frames, self.offsets, strict=True)):
            local = np.arange(0, int(n), stride, dtype=np.int64)
            idx.append(int(o) + local)
            ep.append(np.full(len(local), e, np.int64))
        return np.concatenate(idx), np.concatenate(ep)

    def root_arrays(self, starts, horizons=sw.HORIZONS):
        """Start latents [N, 12 288], the next 32 commands [N, 32, 6] and the targets at h."""
        starts = np.asarray(starts, np.int64)
        s = self.store
        commands = s.actions[starts[:, None] + np.arange(sw.ROOT_SPAN)]
        targets = {h: s.standardised(starts + h) for h in horizons}
        return s.standardised(starts), commands, targets


# ----- standardisation of the flat latent (§3) ---------------------------------------------------
class Moments:
    """Per-dimension mean and std (floor 1e-6) of the flat latent over the fit set."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, np.float64)
        self.std = np.maximum(np.asarray(std, np.float64), sw.FEATURE_STD_FLOOR)
        self._mean32 = self.mean.astype(np.float32)
        self._scale32 = (1.0 / self.std).astype(np.float32)

    def apply(self, raw) -> np.ndarray:
        out = (np.asarray(raw, np.float32) - self._mean32) * self._scale32
        if not np.isfinite(out).all():
            raise ContractError("G-finite: non-finite standardised latent")
        return out

    @property
    def sha256(self) -> str:
        return sha256_arrays(self.mean, self.std)

    @classmethod
    def fit(cls, store: FeatureStore, view: FeatureView, chunk: int = 4096) -> Moments:
        idx, _ = view.frame_index()
        s1 = np.zeros(sw.LATENT_DIM)
        s2 = np.zeros(sw.LATENT_DIM)
        for lo in range(0, len(idx), chunk):
            x = store.raw(idx[lo : lo + chunk]).astype(np.float64)
            s1 += x.sum(0)
            s2 += (x * x).sum(0)
        n = len(idx)
        mean = s1 / n
        return cls(mean, np.sqrt(np.clip(s2 / n - mean * mean, 0.0, None)))

    def save(self, path) -> str:
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        np.savez(path, mean=self.mean, std=self.std)
        return self.sha256

    @classmethod
    def load(cls, path, expected_sha256: str | None = None) -> Moments:
        with np.load(path) as d:
            m = cls(d["mean"], d["std"])
        if expected_sha256 is not None and m.sha256 != expected_sha256:
            raise ContractError("G-features: the moments differ from their recorded sha256")
        return m


def second_moment(store: FeatureStore, view: FeatureView, chunk: int = 2048):
    """Uncentred second moment ``X^T X`` [D, D], mean and count of the standardised latents of
    every frame of ``view`` (for G1 (iii)'s basis; float64)."""
    idx, _ = view.frame_index()
    xtx = np.zeros((sw.LATENT_DIM, sw.LATENT_DIM))
    s1 = np.zeros(sw.LATENT_DIM)
    for lo in range(0, len(idx), chunk):
        x = store.standardised(idx[lo : lo + chunk]).astype(np.float64)
        xtx += x.T @ x
        s1 += x.sum(0)
    return xtx, s1 / len(idx), len(idx)


# ----- the window sampler (§5) -------------------------------------------------------------------
class WindowSampler:
    """Uniform over all valid (episode, start) pairs, with replacement, batch 64, from
    ``default_rng(SeedSequence([10011, stream]))``; ``draws`` counts batches for a resume."""

    def __init__(self, starts, stream_seed: int, batch: int = sw.BATCH):
        self.starts = np.asarray(starts, np.int64)
        self.batch = int(batch)
        self.rng = np.random.default_rng(sw.sampler_seed(stream_seed))
        self.draws = 0

    def draw(self) -> np.ndarray:
        self.draws += 1
        return self.starts[self.rng.integers(0, len(self.starts), self.batch)]

    def state(self) -> dict:
        return {"draws": self.draws, "bit_generator": self.rng.bit_generator.state}

    def restore(self, state: dict) -> None:
        self.rng.bit_generator.state = state["bit_generator"]
        self.draws = int(state["draws"])


# ----- named views (§2, §10) ---------------------------------------------------------------------
# name -> (store split, shards or None, fit sample only)
VIEWS = {
    "d0": ("d0", None, False),
    "d0-fit": ("d0", sw.D0_FIT_SHARDS, False),
    "d0-eval": ("d0", sw.D0_EVAL_SHARDS, False),
    "train": ("train", None, False),
    "fit-sample": ("train", None, True),
    "val": ("val", None, False),
    "test": ("test", None, False),
    "smoke-val": ("valsmoke", sw.SMOKE_VAL_SHARDS, False),
    "smoke-test": ("valsmoke", sw.SMOKE_TEST_SHARDS, False),
}
_OPEN: dict = {}


def open_view(folder, name: str, moments: Moments, *, allow_test: bool = False) -> FeatureView:
    """A named view of a feature store. ``test`` opens only with ``allow_test`` (the caller has
    checked the val report's recorded sha256 first, §2)."""
    if name not in VIEWS:
        raise ContractError(f"unknown view {name!r}")
    split, shards, fit = VIEWS[name]
    if split == "test" and not allow_test:
        raise ContractError("G-test: the test split opens only after the val report's sha256")
    key = (str(Path(folder).resolve()), split, moments.sha256)
    if key not in _OPEN:
        _OPEN[key] = FeatureStore(folder, split, moments=moments)
    store = _OPEN[key]
    mask = np.ones(len(store.table), bool)
    if shards is not None:
        mask &= np.isin(store.shards, list(shards))
    if fit:
        mask &= store.seeds % sw.FIT_SAMPLE_MODULUS == 0
    if not mask.any():
        raise ContractError(f"view {name!r} is empty")
    return store.subset(mask)
