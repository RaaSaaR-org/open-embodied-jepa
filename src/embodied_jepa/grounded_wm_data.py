"""TASK-087's featurisation and feature stores (protocol §2-§3).

* ``corpus_episodes``: the sealed ``play-v1`` episodes by split, from every shard's frozen split
  file, checked against the protocol's counts.
* ``load_episode``: one episode's frames, the 28-D state, actions and the right palm position (the
  sidecar is checked against its recorded sha256).
* ``Featuriser``: the frozen DINOv2 ViT-S/14 on the GPU, pooled to 4 x 4.
* Pass A (``fit_projection``) accumulates per-dimension moments and per-position second moments
  over the fit sample (train episodes whose seed is divisible by 4) and fits the projection;
  pass B (``project``) writes ``[frames, 16, 192]`` float16 per split.
* ``FeatureStore``: one split's arrays (memory-mapped latents) and its episode table.

NumPy at import; torch, transformers and the dataset reader are imported where needed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_jepa import grounded_wm as gw
from embodied_jepa.contracts import ContractError

SPLITS = ("train", "val", "test")


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
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
                seed = int(episode_id.rsplit("-", 1)[1])
                out[split].append((shard.name, episode_id, seed))
    if check_counts:
        if len(shards) != gw.CORPUS_SHARDS:
            raise ContractError(f"expected {gw.CORPUS_SHARDS} shards, found {len(shards)}")
        for split, n in gw.SPLIT_COUNTS.items():
            if len(out[split]) != n:
                raise ContractError(f"{split}: {len(out[split])} episodes, protocol says {n}")
    return out


def fit_sample(train_episodes) -> list:
    return [e for e in train_episodes if e[2] % gw.FIT_SAMPLE_MODULUS == 0]


_STORES: dict = {}


def _store(shard_dir):
    from embodied_jepa.data import DatasetStore

    key = str(shard_dir)
    if key not in _STORES:
        _STORES[key] = DatasetStore(shard_dir)
    return _STORES[key]


def state_columns(state_names) -> list[int]:
    names = list(state_names)
    return [names.index(n) for n in gw.STATE_NAMES]


def load_episode(corpus, shard: str, episode_id: str) -> dict:
    """Frames uint8 [T+1, 112, 112, 3], state float32 [T+1, 28], actions float32 [T+1, 14] (the
    last row 0, never a transition), palm float32 [T+1, 3]."""
    store = _store(Path(corpus) / shard)
    ep = store.read_episode(episode_id)
    meta = ep.metadata
    sidecar = Path(corpus) / shard / meta["sidecar"]
    if sha256_file(sidecar) != meta["sidecar_sha256"]:
        raise ContractError(f"sidecar hash mismatch: {episode_id}")
    with np.load(sidecar) as d:
        palms = np.asarray(d["palms"], np.float32)
    frames = ep.observations[gw.CAMERA]
    n = len(frames)
    if palms.shape[0] != n or ep.robot_states.shape[0] != n or ep.actions.shape[0] != n - 1:
        raise ContractError(f"length mismatch in {episode_id}")
    actions = np.zeros((n, gw.ACTION_DIM), np.float32)
    actions[:-1] = ep.actions
    return {
        "episode_id": episode_id,
        "frames": frames,
        "state": ep.robot_states[:, state_columns(ep.state_schema.names)].astype(np.float32),
        "actions": actions,
        "palm": palms[:, gw.PALM_SLICE].copy(),
    }


def _load_worker(args):
    return load_episode(*args)


def iter_episodes(corpus, episodes, *, workers: int = 8):
    """Yield loaded episodes in order, decoding in a process pool."""
    jobs = [(str(corpus), shard, eid) for shard, eid, _ in episodes]
    if workers <= 1:
        for job in jobs:
            yield load_episode(*job)
        return
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(workers) as pool:
        yield from pool.imap(_load_worker, jobs, chunksize=2)


# ----- the frozen encoder ------------------------------------------------------------------------
class Featuriser:
    """Frozen DINOv2 ViT-S/14 (pinned), float32, eager attention, fixed batch, pooled to 4 x 4."""

    def __init__(self, device: str = gw.FEATURE_DEVICE, batch: int = gw.FEATURE_BATCH):
        import torch

        from embodied_jepa import pretrained_encoder as pe

        self.torch, self.pe = torch, pe
        self.device, self.batch = device, int(batch)
        module = pe.load_pretrained()
        self.digest = pe.weights_digest(module)
        self.model = module.to(device).eval()

    def pooled(self, frames) -> np.ndarray:
        """float32 [N, 6 144] (row-major over grid row, grid column, channel)."""
        torch, pe = self.torch, self.pe
        out = []
        g, k = gw.TOKEN_GRID, pe.GRID // gw.TOKEN_GRID
        with torch.no_grad():
            for lo in range(0, len(frames), self.batch):
                px = pe.preprocess(frames[lo : lo + self.batch]).to(self.device)
                tok = self.model(pixel_values=px).last_hidden_state[:, 1:]
                tok = tok.double().reshape(len(tok), g, k, g, k, pe.WIDTH).mean((2, 4))
                out.append(tok.reshape(len(tok), -1).float().cpu().numpy())
        result = np.concatenate(out)
        if not np.isfinite(result).all():
            raise ContractError("non-finite DINOv2 features")
        return result


# ----- pass A: the projection --------------------------------------------------------------------
class ProjectionAccumulator:
    """Per-dimension sums and per-position second moments of pooled features (float64)."""

    def __init__(self):
        d, p, w = gw.RAW_DIM, gw.TOKENS, gw.TOKEN_WIDTH_RAW
        self.n = 0
        self.s1 = np.zeros(d)
        self.s2 = np.zeros(d)
        self.m = np.zeros((p, w, w))

    def add(self, x):
        x = np.asarray(x, np.float64)
        self.n += len(x)
        self.s1 += x.sum(0)
        self.s2 += (x * x).sum(0)
        t = x.reshape(len(x), gw.TOKENS, gw.TOKEN_WIDTH_RAW)
        self.m += np.einsum("npi,npj->pij", t, t)

    def fit(self, k: int = gw.K) -> dict:
        if self.n < 2:
            raise ContractError("the fit sample is empty")
        mean = self.s1 / self.n
        std = np.sqrt(np.clip(self.s2 / self.n - mean * mean, 0.0, None))
        std = np.maximum(std, gw.FEATURE_STD_FLOOR)
        mu = mean.reshape(gw.TOKENS, gw.TOKEN_WIDTH_RAW)
        sd = std.reshape(gw.TOKENS, gw.TOKEN_WIDTH_RAW)
        cov = np.zeros((gw.TOKEN_WIDTH_RAW, gw.TOKEN_WIDTH_RAW))
        for p in range(gw.TOKENS):
            c = self.m[p] / self.n - np.outer(mu[p], mu[p])
            cov += c / np.outer(sd[p], sd[p])
        cov /= gw.TOKENS
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        # a deterministic sign: the largest-magnitude entry of each component is positive
        signs = np.sign(vectors[np.abs(vectors).argmax(0), np.arange(vectors.shape[1])])
        vectors = vectors * np.where(signs == 0, 1.0, signs)
        return {
            "mean": mean.astype(np.float64),
            "std": std.astype(np.float64),
            "components": vectors[:, :k].astype(np.float64),
            "eigenvalues": values.astype(np.float64),
            "explained": float(values[:k].sum() / values.sum()),
            "frames": int(self.n),
        }


def project(pooled, projection: dict) -> np.ndarray:
    """Pooled float32 [N, 6 144] -> standardised and projected float32 [N, 16, K]."""
    x = (np.asarray(pooled, np.float64) - projection["mean"]) / projection["std"]
    t = x.reshape(len(x), gw.TOKENS, gw.TOKEN_WIDTH_RAW)
    return (t @ projection["components"]).astype(np.float32)


def projection_sha256(projection: dict) -> str:
    digest = hashlib.sha256()
    for key in ("mean", "std", "components"):
        a = np.ascontiguousarray(projection[key], np.float64)
        digest.update(key.encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


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
        raise ContractError("the projection differs from its recorded sha256")
    return projection


# ----- pass B: a split's store -------------------------------------------------------------------
class SplitWriter:
    """Writes one split's arrays incrementally (latents float16 [F, 16, K])."""

    def __init__(self, folder, split: str, total_frames: int):
        self.folder = Path(folder)
        self.split = split
        self.folder.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "latents": self.folder / f"latents_{split}.npy",
            "state": self.folder / f"state_{split}.npy",
            "actions": self.folder / f"actions_{split}.npy",
            "palm": self.folder / f"palm_{split}.npy",
            "episodes": self.folder / f"episodes_{split}.json",
        }
        for p in self.paths.values():
            if p.exists():
                raise FileExistsError(f"refusing to overwrite {p}")
        f = int(total_frames)
        self.latents = np.lib.format.open_memmap(
            self.paths["latents"], mode="w+", dtype=np.float16, shape=(f, gw.TOKENS, gw.K)
        )
        self.state = np.zeros((f, gw.STATE_DIM), np.float32)
        self.actions = np.zeros((f, gw.ACTION_DIM), np.float32)
        self.palm = np.zeros((f, 3), np.float32)
        self.table: list[dict] = []
        self.offset = 0

    def add(self, episode: dict, latents, *, shard: str, seed: int):
        n = len(latents)
        lo, hi = self.offset, self.offset + n
        if hi > len(self.state):
            raise ContractError("more frames than declared")
        self.latents[lo:hi] = latents.astype(np.float16)
        self.state[lo:hi] = episode["state"]
        self.actions[lo:hi] = episode["actions"]
        self.palm[lo:hi] = episode["palm"]
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
        if self.offset != len(self.state):
            raise ContractError(f"wrote {self.offset} of {len(self.state)} frames")
        self.latents.flush()
        del self.latents
        np.save(self.paths["state"], self.state)
        np.save(self.paths["actions"], self.actions)
        np.save(self.paths["palm"], self.palm)
        self.paths["episodes"].write_text(json.dumps(self.table))
        return {k: sha256_file(p) for k, p in self.paths.items()}


def episode_frames(corpus, episodes) -> list[int]:
    """Frame counts from the shard manifests (no decoding)."""
    lengths = {}
    for shard in sorted({s for s, _, _ in episodes}):
        manifest = json.loads((Path(corpus) / shard / "meta" / "jepa_manifest.json").read_text())
        for row in manifest["episodes"]:
            lengths[(shard, row["episode_id"])] = int(row["length"])
    return [lengths[(s, e)] for s, e, _ in episodes]


class FeatureStore:
    """One split: latents [F, 16, K] float16 (memory-mapped), state, actions, palm and the
    episode table (offsets, frame counts)."""

    def __init__(self, folder, split: str, *, mmap: bool = True):
        folder = Path(folder)
        self.split = split
        self.latents = np.load(folder / f"latents_{split}.npy", mmap_mode="r" if mmap else None)
        self.state = np.load(folder / f"state_{split}.npy")
        self.actions = np.load(folder / f"actions_{split}.npy")
        self.palm = np.load(folder / f"palm_{split}.npy")
        self.table = json.loads((folder / f"episodes_{split}.json").read_text())
        self.frames = np.array([r["frames"] for r in self.table], np.int64)
        self.offsets = np.array([r["offset"] for r in self.table], np.int64)
        f = len(self.state)
        if self.latents.shape != (f, gw.TOKENS, gw.K) or int(self.frames.sum()) != f:
            raise ContractError(f"{split}: inconsistent store")

    def moments(self):
        """Train moments of the state and palm (every frame)."""
        return (self.state.mean(0), self.state.std(0), self.palm.mean(0), self.palm.std(0))

    def gather(self, starts, frames: int):
        """Windows ``[B, frames, ...]`` from global start indices (float32)."""
        idx = np.asarray(starts, np.int64)[:, None] + np.arange(frames)
        return {
            "latents": np.asarray(self.latents[idx.ravel()], np.float32).reshape(
                len(idx), frames, gw.TOKENS, gw.K
            ),
            "state": self.state[idx],
            "actions": self.actions[idx[:, :-1]],
            "palm": self.palm[idx],
        }

    def roots(self):
        return gw.roots(self.frames, self.offsets)

    def root_arrays(self, starts, horizons=gw.HORIZONS):
        """Start latents, the next EVAL_HORIZON commands, start states and the targets at h."""
        starts = np.asarray(starts, np.int64)
        start = np.asarray(self.latents[starts], np.float32)
        cmd_idx = starts[:, None] + np.arange(gw.EVAL_HORIZON)
        commands = self.actions[cmd_idx]
        targets = {h: np.asarray(self.latents[starts + h], np.float32) for h in horizons}
        return start, commands, self.state[starts], targets


class WindowSampler:
    """Uniform over all valid (episode, start) pairs; ``default_rng(SeedSequence([8714, seed]))``,
    so every arm of a seed draws the same windows in the same order."""

    def __init__(self, store: FeatureStore, model_seed: int, batch: int = gw.BATCH):
        self.starts = gw.window_starts(store.frames, store.offsets)
        self.batch = int(batch)
        self.rng = np.random.default_rng(gw.sampler_seed(model_seed))

    def draw(self) -> np.ndarray:
        return self.starts[self.rng.integers(0, len(self.starts), self.batch)]
