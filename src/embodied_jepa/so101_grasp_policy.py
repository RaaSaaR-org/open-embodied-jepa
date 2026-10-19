"""TASK-102's learned local grasp-and-lift policy G_loc
(``docs/experiments/so101_hybrid_lift_v1.md`` §5.3-§5.4): P-3's head on TASK-100's live latent
plus proprioception and a clock, its input assembly, behaviour-cloning training loop and
checkpoint selection.

Inputs per command: z, the standardised g4 latent of both cameras (12 288 values; TASK-100's
pipeline, shared with phase 1), and 35 non-latent values (``so101_hybrid.nonlatent``: the
measured joint positions and velocities, the current joint-position targets and the clock),
standardised by the training windows' moments (std floored at 1e-3) and frozen with the
checkpoint. No object truth, contact truth, goal or phase label is an input. The 6 outputs are the
normalised ``joint_delta_so101_v0`` command; ``act`` clips them to [-1, 1] before sending (the
contract validates and never clips).

Every checkpoint records this file's sha256 as its implementation hash and is refused when the
file changes. torch is imported at module import; nothing in ``import embodied_jepa`` loads this
module.
"""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from embodied_jepa import so101_hybrid as hy
from embodied_jepa.contracts import ContractError

IMPLEMENTATION_FILE = Path(__file__).resolve()


def implementation_sha256() -> str:
    return hashlib.sha256(IMPLEMENTATION_FILE.read_bytes()).hexdigest()


class GLocHead(nn.Module):
    """LayerNorm -> Linear(12 323, 512) -> SiLU -> Linear(512, 512) -> SiLU -> Linear(512, 512)
    -> SiLU -> Linear(512, 6) (P-3's head)."""

    def __init__(self, hidden: int = hy.TRAIN["hidden"]):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hy.INPUT_DIM),
            nn.Linear(hy.INPUT_DIM, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hy.OUTPUT_DIM),
        )

    def forward(self, x):
        return self.net(x)


class InputMoments:
    """The 35 non-latent inputs' mean and std (floored at 1e-3) over the training frames."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, np.float32).reshape(hy.NONLATENT_DIM)
        self.std = np.maximum(np.asarray(std, np.float32), hy.NONLATENT_STD_FLOOR).reshape(
            hy.NONLATENT_DIM
        )

    @classmethod
    def fit(cls, x) -> InputMoments:
        x = np.asarray(x, np.float64)
        return cls(x.mean(0), x.std(0))

    def apply(self, x) -> np.ndarray:
        return ((np.asarray(x, np.float32) - self.mean) / self.std).astype(np.float32)


def assemble(z, x35, moments: InputMoments) -> np.ndarray:
    """[N, 12 288] standardised latents and [N, 35] raw non-latent inputs -> [N, 12 323]."""
    z = np.asarray(z, np.float32)
    x35 = np.asarray(x35, np.float32)
    if z.ndim != 2 or z.shape[1] != hy.LATENT_DIM or x35.shape != (len(z), hy.NONLATENT_DIM):
        raise ContractError("G_loc needs z [N, 12288] and x35 [N, 35]")
    out = np.concatenate([z, moments.apply(x35)], axis=1)
    if not np.isfinite(out).all():
        raise hy.GuardError("G-finite: a non-finite G_loc input")
    return out


class GLocPolicy:
    """A trained G_loc: ``predict`` (batched, unclipped) and ``act`` (one command, clipped)."""

    def __init__(self, head: GLocHead, moments: InputMoments, device: str, meta=None):
        self.head = head.to(device).eval()
        self.moments = moments
        self.device = device
        self.meta = dict(meta or {})

    def predict(self, z, x35, chunk: int = 1024) -> np.ndarray:
        x = assemble(z, x35, self.moments)
        out = []
        with torch.no_grad():
            for lo in range(0, len(x), chunk):
                xb = torch.from_numpy(x[lo : lo + chunk]).to(self.device)
                out.append(self.head(xb).float().cpu().numpy())
        y = np.concatenate(out) if out else np.zeros((0, hy.OUTPUT_DIM), np.float32)
        if not np.isfinite(y).all():
            raise hy.GuardError("G-finite: a non-finite G_loc output")
        return y.astype(np.float32)

    def act(self, z, q, qvel, targets, t: int) -> tuple[np.ndarray, np.ndarray]:
        """One command from the live latent z [12 288] and proprioception; returns (the clipped
        command sent, the raw output)."""
        x35 = hy.nonlatent(q, qvel, targets, t)[None]
        raw = self.predict(np.asarray(z, np.float32)[None], x35)[0]
        return np.clip(raw, -1.0, 1.0).astype(np.float32), raw


# ----- checkpoints --
def save_checkpoint(path, head: GLocHead, moments: InputMoments, meta: dict) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "input_mean": moments.mean,
        "input_std": moments.std,
        "implementation_sha256": implementation_sha256(),
        "meta": dict(meta),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    path.write_bytes(buffer.getvalue())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path, device: str, expected_sha256: str | None = None) -> GLocPolicy:
    """Refused on a file-hash mismatch or when this file's implementation hash differs."""
    path = Path(path)
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and got != expected_sha256:
        raise hy.GuardError(f"G-checkpoint: {path} sha256 {got[:12]} is not the recorded one")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["implementation_sha256"] != implementation_sha256():
        raise hy.GuardError("G-checkpoint: G_loc's implementation hash differs")
    head = GLocHead()
    head.load_state_dict(payload["state_dict"])
    moments = InputMoments(payload["input_mean"], payload["input_std"])
    return GLocPolicy(head, moments, device, payload["meta"] | {"file_sha256": got})


# ----- training (§5.4) --
def cosine_lr(update: int, total: int = hy.TRAIN["updates"]) -> float:
    """3e-4 cosine to 3e-5 over the updates."""
    lo, hi = hy.TRAIN["lr_final"], hy.TRAIN["lr"]
    return lo + 0.5 * (hi - lo) * (1.0 + math.cos(math.pi * min(update, total) / total))


class Frames:
    """One split's training frames: raw float16 latents [F, 12 288] (memory or memory map), the
    raw non-latent inputs [F, 35] and the labels [F, 6]; standardised on the fly with TASK-100's
    frozen fit moments (``so101_wm_data.Moments``)."""

    def __init__(self, raw16, x35, labels, latent_moments):
        self.raw16, self.x35, self.labels = raw16, np.asarray(x35, np.float32), labels
        self.latent_moments = latent_moments
        if not (len(raw16) == len(self.x35) == len(labels)):
            raise ContractError("frames, inputs and labels differ in length")

    def __len__(self):
        return len(self.labels)

    def batch(self, index, input_moments: InputMoments) -> tuple[np.ndarray, np.ndarray]:
        index = np.asarray(index, np.int64)
        order = np.argsort(index, kind="stable")
        raw = np.empty((len(index), hy.LATENT_DIM), np.float32)
        raw[order] = np.asarray(self.raw16[index[order]], np.float32).reshape(len(index), -1)
        z = self.latent_moments.apply(raw)
        return assemble(z, self.x35[index], input_moments), np.asarray(
            self.labels[index], np.float32
        )


def evaluate(head, frames: Frames, moments: InputMoments, device: str, chunk: int = 2048):
    """Val MSE (mean over frames and the 6 channels) and each output's std over the frames."""
    head.eval()
    outs, sq, n = [], 0.0, 0
    with torch.no_grad():
        for lo in range(0, len(frames), chunk):
            idx = np.arange(lo, min(lo + chunk, len(frames)))
            x, y = frames.batch(idx, moments)
            pred = head(torch.from_numpy(x).to(device)).double()
            target = torch.from_numpy(y).to(device).double()
            sq += float(((pred - target) ** 2).sum())
            n += pred.numel()
            outs.append(pred.cpu().numpy())
    head.train()
    out = np.concatenate(outs)
    return {"val_mse": sq / n, "output_std": out.std(0).tolist()}


def train(
    i: int,
    train_frames: Frames,
    val_frames: Frames,
    moments: InputMoments,
    device: str,
    *,
    updates: int = hy.TRAIN["updates"],
    eval_every: int = hy.TRAIN["eval_every"],
    log=None,
) -> dict:
    """§5.4 for seed G-i: AdamW, lr 3e-4 cosine to 3e-5, weight decay 1e-4, gradient clip 1.0,
    batch 256 uniform over training frames, MSE over the 6 channels; initialisation and batch
    order from ``default_rng([10115, i])``; val MSE every 1 000 updates (and at 0). Returns the
    curve, the selected update's state dict and the selection."""
    rng = hy.gloc_rng(i)
    init_seed = int(rng.integers(0, 2**62))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        head = GLocHead().to(device)
    opt = torch.optim.AdamW(
        head.parameters(), lr=hy.TRAIN["lr"], weight_decay=hy.TRAIN["weight_decay"]
    )
    curve, best_state, best_key = [], None, None
    head.train()

    def point(update):
        nonlocal best_state, best_key
        p = {"update": update} | evaluate(head, val_frames, moments, device)
        curve.append(p)
        eligible = update > 0 and min(p["output_std"]) >= hy.TRAIN["min_output_std"]
        key = (p["val_mse"], update)
        if eligible and (best_key is None or key < best_key):
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if log:
            log(f"G-{i} update {update}: val MSE {p['val_mse']:.6f} eligible={eligible}")

    point(0)
    losses = []
    for update in range(1, updates + 1):
        for group in opt.param_groups:
            group["lr"] = cosine_lr(update - 1, updates)
        idx = rng.integers(0, len(train_frames), hy.TRAIN["batch"])
        x, y = train_frames.batch(idx, moments)
        pred = head(torch.from_numpy(x).to(device))
        loss = ((pred - torch.from_numpy(y).to(device)) ** 2).mean()
        if not torch.isfinite(loss):
            raise hy.GuardError("G-finite: a non-finite training loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), hy.TRAIN["grad_clip"])
        opt.step()
        losses.append(float(loss.detach()))
        if update % eval_every == 0:
            point(update)
            curve[-1]["train_loss_mean"] = float(np.mean(losses[-eval_every:]))
    selection = hy.select_gloc(curve)
    return {
        "seed": i,
        "init_seed": init_seed,
        "curve": curve,
        "selection": selection,
        "state_dict": best_state,
        "head": head,
    }
