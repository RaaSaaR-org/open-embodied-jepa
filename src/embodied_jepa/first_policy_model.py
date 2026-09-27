"""TASK-067 policy network and its bounded trainer (torch; imported only where needed).

``POLICY_HEAD`` and ``TRAINING`` in ``first_policy`` fix the architecture and the recipe. P, C-3
and R-3 use one head; the fallback F uses one head per collector phase, chosen by
``first_policy.scheduled_phase(step)``, a scripted switch (rung L2).
"""

from __future__ import annotations

import hashlib
import io
import time

import numpy as np
import torch
from torch import nn

from embodied_jepa import first_policy as fp
from embodied_jepa.contracts import ContractError

INPUT_DIM = sum(fp.POLICY_INPUTS.values())
OUTPUT_DIM = len(fp.FREE_INDICES)
HIDDEN = 512


def head() -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(INPUT_DIM),
        nn.Linear(INPUT_DIM, HIDDEN),
        nn.SiLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.SiLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.SiLU(),
        nn.Linear(HIDDEN, OUTPUT_DIM),
    )


class PolicyNet(nn.Module):
    """``heads`` = 1 (P, C-3, R-3) or ``len(PHASES)`` (F, head chosen by the collector's clock)."""

    def __init__(self, heads: int = 1):
        super().__init__()
        if heads not in (1, len(fp.PHASES)):
            raise ContractError("a policy has one head, or one per collector phase (F)")
        self.heads = nn.ModuleList(head() for _ in range(heads))

    def forward(self, x, phase=None):
        if len(self.heads) == 1:
            return self.heads[0](x)
        if phase is None:
            raise ContractError("F needs the scheduled phase of every row")
        out = x.new_zeros((len(x), OUTPUT_DIM))
        for index, module in enumerate(self.heads):
            rows = phase == index
            if bool(rows.any()):
                out[rows] = module(x[rows])
        return out


def phases_of(steps) -> np.ndarray:
    return np.asarray([fp.scheduled_phase(int(s)) for s in np.asarray(steps).reshape(-1)])


def weights_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train(
    x,
    y,
    steps,
    val_x,
    val_y,
    val_steps,
    *,
    heads: int = 1,
    device: str = fp.TRAIN_DEVICE,
    updates: int | None = None,
    select_every: int | None = None,
    min_output_std: float | None = None,
    seed: int = fp.TRAIN_SAMPLER_SEED,
    max_seconds: float = fp.CAPS_SECONDS["per_training"],
) -> dict:
    """Train from scratch (``TRAINING``) and select on val. Inputs are already standardised.

    Returns the selected weights on CPU and the record. ``selected`` is ``None`` when no
    checkpoint was eligible (the arm then scores 0/16 on D, ``NO_ELIGIBLE_CHECKPOINT``)."""
    cfg = fp.TRAINING
    updates = int(updates or cfg["updates"])
    select_every = int(select_every or cfg["selection_every"])
    x = torch.as_tensor(np.asarray(x, np.float32))
    y = torch.as_tensor(np.asarray(y, np.float32))
    vx = torch.as_tensor(np.asarray(val_x, np.float32)).to(device)
    vy = torch.as_tensor(np.asarray(val_y, np.float32)).to(device)
    phase = torch.as_tensor(phases_of(steps))
    vphase = torch.as_tensor(phases_of(val_steps)).to(device)
    if not (len(x) == len(y) == len(phase)) or len(vx) != len(vy):
        raise ContractError("training rows do not align")
    if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise fp.GuardError("G-finite: non-finite training data")
    torch.manual_seed(fp.MODEL_SEED)
    model = PolicyNet(heads).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates, eta_min=cfg["final_learning_rate"]
    )
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    curve, best = [], None
    for update in range(1, updates + 1):
        index = torch.as_tensor(rng.integers(0, len(x), cfg["batch"]))
        pred = model(x[index].to(device), phase[index].to(device))
        loss = ((pred - y[index].to(device)) ** 2).mean()
        if not torch.isfinite(loss):
            raise fp.GuardError("G-finite: a training loss is not finite")
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        schedule.step()
        if time.monotonic() - started > max_seconds:
            raise fp.GuardError("G-cap: a training exceeded its cap")
        if update % select_every == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(vx, vphase)
                val_mse = float(((val_pred - vy) ** 2).mean())
                min_std = float(val_pred.std(dim=0).min())
            model.train()
            floor = cfg["selection_min_output_std"] if min_output_std is None else min_output_std
            eligible = min_std >= floor
            curve.append(
                {
                    "update": update,
                    "val_mse": val_mse,
                    "min_output_std": min_std,
                    "eligible": eligible,
                }
            )
            if eligible and (best is None or val_mse < best["val_mse"]):
                best = {
                    "update": update,
                    "val_mse": val_mse,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                }
    record = {
        "heads": heads,
        "updates": updates,
        "curve": curve,
        "seconds": time.monotonic() - started,
        "selected": None
        if best is None
        else {"update": best["update"], "val_mse": best["val_mse"]},
    }
    if best is None:
        return {"state": None, "record": record}
    return {"state": best["state"], "record": record}


def load(state, heads: int = 1) -> PolicyNet:
    model = PolicyNet(heads)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predictor(model: PolicyNet):
    """``(standardised [132] float32, step) -> free 7`` on CPU, one row at a time."""

    def predict(x, step):
        with torch.no_grad():
            row = torch.as_tensor(np.asarray(x, np.float32)[None])
            phase = torch.as_tensor([fp.scheduled_phase(int(step))])
            return model(row, phase)[0].numpy()

    return predict


def batch_predict(model: PolicyNet, x, steps) -> np.ndarray:
    with torch.no_grad():
        return model(
            torch.as_tensor(np.asarray(x, np.float32)), torch.as_tensor(phases_of(steps))
        ).numpy()


def state_bytes(state) -> bytes:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()
