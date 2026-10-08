"""TASK-082's training (protocol ``docs/experiments/apple_lewm_unknown_law_v2.md`` §4.4, R20.5):
TASK-077's recipe unchanged on this task's corpus, with this task's window-sampler salt (8409) and
model metadata.

TASK-077's storage (``RootStore``), prefetcher, model builder, loader and val criterion are imported
unchanged from ``lewm_c1m_v2_train``. :func:`train_model` is TASK-077's training loop **copied
verbatim** (a test checks its source against ``lewm_c1m_v2_train.train_model``): in this module the
name ``WindowSampler`` is :class:`UlWindowSampler`, whose draws come from
``default_rng(SeedSequence([8409, model seed]))``, so W and N of a seed draw the same windows in
the same order, and none is TASK-077's (salt 8109). The selection is ``run_tools
.select_checkpoint(curve, 0.01)`` with ``last_two_triggered`` recorded (no raw argmin).

NumPy at import; torch is imported inside the functions that need it.
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from embodied_jepa import lewm_c1m_v2 as lm
from embodied_jepa import lewm_c1m_v2_train as tr
from embodied_jepa import lewm_ul_v2 as ul
from embodied_jepa import run_tools as rt
from embodied_jepa.contracts import ContractError
from embodied_jepa.lewm_c1m_v2_train import (  # noqa: F401 - re-exported for the runner
    Prefetcher,
    RootStore,
    build_model,
    load_model,
    measure_gather,
    moments_file,
    sha256_file,
    state_sha256,
    val_criterion,
)

GuardError = tr.GuardError
START_INDEX = tr.START_INDEX
if (lm.ARMS_TRAINED, lm.METRIC_FLOOR_STD, lm.SELECTION_TOLERANCE, lm.BATCH_SIZE) != (
    ul.ARMS_TRAINED, ul.METRIC_FLOOR_STD, ul.SELECTION_TOLERANCE, ul.BATCH_SIZE
):  # fmt: skip
    raise ContractError("TASK-082 trains with TASK-077's recipe unchanged")


class UlWindowSampler(tr.WindowSampler):
    """TASK-077's sampler (uniform root, uniform start in 403-407, 16 windows per draw) from
    ``default_rng(SeedSequence([8409, model seed]))``."""

    def __init__(self, n_roots: int, model_seed: int, batch: int = ul.BATCH_SIZE):
        self.n, self.batch = int(n_roots), int(batch)
        self.rng = np.random.default_rng(ul.sampler_seed(model_seed))


WindowSampler = UlWindowSampler  # the name TASK-077's loop (copied below) constructs


def model_metadata(*, arm: str, seed: int, corpus_sha256: str, moments_sha256: str) -> dict:
    return {
        "protocol": ul.PROTOCOL,
        "arm": str(arm),
        "model_seed": int(seed),
        "corpus_manifest_sha256": str(corpus_sha256),
        "normalisation_sha256": str(moments_sha256),
        "sampler_salt": ul.SALTS["sampler"],
    }


# ----- TASK-077's training loop, copied verbatim (lewm_c1m_v2_train.train_model) ----------------
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
    step_seconds, selection_seconds = [], []
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
                t_sel = time.perf_counter()
                value = val_criterion(model, val, scale, arm)
                selection_seconds.append(time.perf_counter() - t_sel)
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
        "selection_seconds": selection_seconds,
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
