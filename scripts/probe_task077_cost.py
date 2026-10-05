"""TASK-077 DRAFT, development only: the GPU cost of the 8 x 8 token predictor (no corpus).

The preregistration draft (``docs/experiments/apple_lewm_c1m_v2.md``) needs the per-update cost
and the peak GPU memory of TASK-066's token predictor at ``token_grid = 8`` to size its budget
block and wall-time caps; no task has measured it on CUDA (TASK-066 measured 0.313 s per update
on MPS, synthetic inputs, T = 16). This probe measures it on **synthetic** features and actions:
no corpus, no rendered frame, no cohort seed; the model seeds are TASK-077's declared debug seeds
(66990-66999). Nothing in it is a result, and it decides no gate.

Run it through the machine-wide GPU lock::

    scripts/gpu_run.sh --wait --min-free-gib 6 --who oej:task077-cost -- \
        uv run --no-sync python scripts/probe_task077_cost.py --output <new json>
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

DEBUG_SEEDS = (66990, 66991)
WIDTH = 384


def _revision() -> dict:
    root = Path(__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"revision": head, "tracked_tree_dirty": bool(dirty)}


def _model(grid: int, seed: int, device: str):
    from embodied_jepa.contracts import StateSchema
    from embodied_jepa.models.frozen_tokens import frozen_token_model
    from embodied_jepa.token_dynamics import MODEL_CONFIG

    config = dict(MODEL_CONFIG, token_grid=grid, latent_dim=grid * grid * WIDTH)
    schema = StateSchema(("arm.q",), ("rad",), "probe_v0")
    model = frozen_token_model("leworldmodel")(schema, device=device, seed=seed, config=config)
    dim = config["latent_dim"]
    model.fit_frozen_feature_normalization(
        np.zeros(dim), np.ones(dim), training_episode_ids=("synthetic",)
    )
    return model, dim


def time_updates(grid: int, horizon: int, batch: int, seed: int, warmup: int, timed: int) -> dict:
    import torch

    model, dim = _model(grid, seed, "cuda")
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((batch, horizon + 1, dim), dtype=np.float32)
    actions = (0.1 * rng.uniform(-1, 1, (batch, horizon, 14))).astype(np.float32)
    for _ in range(warmup):
        model.train_step_features(features, actions)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(timed):
        model.train_step_features(features, actions)
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / timed
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model
    torch.cuda.empty_cache()
    return {
        "grid": grid,
        "horizon": horizon,
        "batch": batch,
        "seconds_per_update": seconds,
        "peak_allocated_gib": peak,
        "timed_updates": timed,
        "note": "same synthetic batch every update: host gather and transfer of a fresh batch "
        "are timed separately (gather_seconds)",
    }


def time_rollout(grid: int, horizon: int, candidates: int, seed: int, repeats: int) -> dict:
    import torch

    model, dim = _model(grid, seed, "cuda")
    rng = np.random.default_rng(seed)
    start_features = rng.standard_normal((candidates, dim), dtype=np.float32)
    actions = (0.1 * rng.uniform(-1, 1, (candidates, horizon, 14))).astype(np.float32)
    model.predict_features(start_features, actions)
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(repeats):
        model.predict_features(start_features, actions)
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - begin) / repeats
    del model
    torch.cuda.empty_cache()
    return {"grid": grid, "horizon": horizon, "candidates": candidates, "seconds": seconds}


def time_gather(grid: int, horizon: int, batch: int, frames: int, seed: int, repeats: int) -> dict:
    """A fresh batch: fancy-indexing ``batch`` windows of ``horizon + 1`` frames out of a
    ``frames``-row float32 table in host memory (the cache layout of the draft)."""
    dim = grid * grid * WIDTH
    rng = np.random.default_rng(seed)
    table = np.empty((frames, dim), np.float32)
    table[:] = 0.0
    starts = rng.integers(0, frames - horizon - 1, size=(repeats, batch))
    begin = time.perf_counter()
    for row in starts:
        index = row[:, None] + np.arange(horizon + 1)[None, :]
        batch_array = table[index]
        assert batch_array.shape == (batch, horizon + 1, dim)
    seconds = (time.perf_counter() - begin) / repeats
    return {
        "grid": grid,
        "horizon": horizon,
        "batch": batch,
        "table_frames": frames,
        "table_gib": table.nbytes / 2**30,
        "gather_seconds": seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timed", type=int, default=60)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    import torch

    from embodied_jepa import devices
    from embodied_jepa.run_tools import gpu_guard

    report: dict = {"probe": "task077-cost", "synthetic": True, **_revision()}
    gpu_guard(report, min_free_gib=6.0, require_lock=True)
    report["determinism"] = devices.configure_determinism("cuda", strict=True)
    report["torch"] = torch.__version__
    report["device_name"] = torch.cuda.get_device_name(0)
    report["host"] = platform.node()
    seed_a, seed_b = DEBUG_SEEDS
    report["updates"] = [
        time_updates(4, 16, 64, seed_a, 10, args.timed),
        time_updates(8, 16, 64, seed_a, 10, args.timed),
        time_updates(8, 60, 64, seed_a, 5, max(10, args.timed // 3)),
        time_updates(8, 60, 32, seed_b, 5, max(10, args.timed // 3)),
    ]
    report["rollouts"] = [
        time_rollout(8, 60, 147, seed_a, 5),
        time_rollout(8, 60, 1, seed_a, 10),
        time_rollout(8, 60, 256, seed_a, 3),
    ]
    report["gather"] = [
        time_gather(8, 60, 64, 20_000, seed_a, 10),
        time_gather(8, 16, 64, 20_000, seed_a, 10),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
