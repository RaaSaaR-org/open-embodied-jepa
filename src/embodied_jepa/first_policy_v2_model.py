"""TASK-071 policy network and its bounded trainer (torch; imported only where needed).

v1's head (``first_policy_model.head``) and recipe (``first_policy.TRAINING``), unchanged. What
differs from v1's module is only what e9's schedule changes: F has one head per e9 phase (10),
chosen by ``first_policy_v2.scheduled_phase(step)``, a scripted switch (rung L2); and the model
and sampler seeds are TASK-071's fresh ones. ``first_policy_model`` itself is not modified.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from embodied_jepa import first_policy_model as fm
from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa.contracts import ContractError

INPUT_DIM = fm.INPUT_DIM
OUTPUT_DIM = fm.OUTPUT_DIM
weights_sha256 = fm.weights_sha256
state_bytes = fm.state_bytes


class PolicyNet(nn.Module):
    """``heads`` = 1 (P, C-3, R-3) or ``len(EXPERT_PHASES)`` (F, head chosen by e9's clock)."""

    def __init__(self, heads: int = 1):
        super().__init__()
        if heads not in (1, len(fp2.EXPERT_PHASES)):
            raise ContractError("a policy has one head, or one per e9 phase (F)")
        self.heads = nn.ModuleList(fm.head() for _ in range(heads))

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
    return np.asarray([fp2.scheduled_phase(int(s)) for s in np.asarray(steps).reshape(-1)])


def train(
    x,
    y,
    steps,
    val_x,
    val_y,
    val_steps,
    *,
    heads: int = 1,
    device: str = fp2.TRAIN_DEVICE,
    updates: int | None = None,
    select_every: int | None = None,
    min_output_std: float | None = None,
    seed: int = fp2.TRAIN_SAMPLER_SEED,
    max_seconds: float = fp2.CAPS_SECONDS["per_training"],
) -> dict:
    """Train from scratch (``TRAINING``) and select on val. Inputs are already standardised.

    As ``first_policy_model.train``, with e9's phases and TASK-071's seeds. ``selected`` is
    ``None`` when no checkpoint was eligible (``NO_ELIGIBLE_CHECKPOINT``)."""
    cfg = fp2.TRAINING
    updates = int(updates or cfg["updates"])
    select_every = int(select_every or cfg["selection_every"])
    x = torch.as_tensor(np.asarray(x, np.float32))
    y = torch.as_tensor(np.asarray(y, np.float32))
    vx = torch.as_tensor(np.asarray(val_x, np.float32)).to(device)
    vy = torch.as_tensor(np.asarray(val_y, np.float32)).to(device)
    phase = torch.as_tensor(phases_of(steps))
    vphase = torch.as_tensor(phases_of(val_steps)).to(device)
    if not (len(x) == len(y) == len(phase)) or len(vx) != len(vy) or len(x) == 0:
        raise ContractError("training rows do not align")
    if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise fp2.GuardError("G-finite: non-finite training data")
    torch.manual_seed(fp2.MODEL_SEED)
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
            raise fp2.GuardError("G-finite: a training loss is not finite")
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        schedule.step()
        if time.monotonic() - started > max_seconds:
            raise fp2.GuardError("G-cap: a training exceeded its cap")
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
            phase = torch.as_tensor([fp2.scheduled_phase(int(step))])
            return model(row, phase)[0].numpy()

    return predict


def batch_predict(model: PolicyNet, x, steps) -> np.ndarray:
    with torch.no_grad():
        return model(
            torch.as_tensor(np.asarray(x, np.float32)), torch.as_tensor(phases_of(steps))
        ).numpy()
