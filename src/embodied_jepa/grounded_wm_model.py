"""TASK-087: the token predictor and its arms (protocol ``grounded_lewm_v1.md`` §4-§5).

One architecture for every arm: the pinned upstream LeWM ``ARPredictor`` (AdaLN-zero conditioned
transformer) with the 16 latent tokens of a frame as its sequence axis, attention bidirectional
within the frame (the upstream ``Attention`` with its mask fixed to non-causal, as
``models/frozen_tokens.py``), the upstream ``Embedder`` as action encoder conditioning every token,
and the upstream BatchNorm ``MLP`` as ``pred_proj`` on the 16 visual tokens. This is TASK-066's
token-predictor composition, rebuilt here so that no existing model file changes (every earlier
checkpoint's implementation hash stays valid).

Arms (``grounded_wm.ARM_SPEC``):

* ``state_token`` (S, G): a 17th token, an MLP embedding of the standardised 28-D state. In a
  roll-out the start frame's token embeds the measured state; S then carries the predictor's own
  output in the state slot (after the final LayerNorm), unsupervised; G predicts the change of the
  standardised state with ``state_head`` and re-embeds ``state + change``. No future state is read.
* ``inverse`` (G, I): an MLP reads the input latent and the predicted next latent of every step and
  predicts the 7 played action dimensions (AD-WM's action recovery), on predicted latents.
* ``fusion`` (C): an MLP embedding of the state is added to every input token (the start frame and
  every teacher-forced frame); ``readout`` (C): the right palm position from encoded and predicted
  latents.
* ``zero_actions`` (N): actions are replaced by zeros (in training and evaluation).

Torch is imported at module import; ``import embodied_jepa`` never imports this module.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import torch
from torch import nn

from embodied_jepa import grounded_wm as gw
from embodied_jepa.contracts import ContractError

LEWM_SOURCE = "third_party/le-wm"


def _bidirectional(components):
    base = components.Attention

    class WithinFrameAttention(base):
        def forward(self, x, causal=False):  # noqa: ARG002 -- always bidirectional
            return base.forward(self, x, causal=False)

    return WithinFrameAttention


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, o))


class GroundedWM(nn.Module):
    """The token predictor of one arm. Latents are ``[B, 16, K]`` (float32, the stored
    projection); states are raw 28-D and standardised inside by the train moments set with
    :meth:`set_moments`; actions are normalised ``[B, 14]``."""

    def __init__(self, arm: str, *, config: dict | None = None, source: str = LEWM_SOURCE):
        super().__init__()
        self.arm = arm
        self.flags = gw.check_arm(arm)
        self.config = dict(gw.MODEL_CONFIG) | dict(config or {})
        c = self.config
        width = int(c["width"])
        from embodied_jepa.models.lewm import load_upstream

        upstream, components = load_upstream(source)
        self.tokens = gw.TOKENS + (1 if self.flags["state_token"] else 0)
        self.predictor = components.ARPredictor(
            num_frames=self.tokens,
            input_dim=width,
            hidden_dim=width,
            output_dim=width,
            depth=int(c["depth"]),
            heads=int(c["heads"]),
            dim_head=int(c["head_dim"]),
            mlp_dim=int(c["mlp_dim"]),
            dropout=0.0,
            emb_dropout=0.0,
        )
        within = _bidirectional(components)
        for block in self.predictor.transformer.layers:
            block.attn.__class__ = within
        self.action_encoder = components.Embedder(input_dim=gw.ACTION_DIM, emb_dim=width)
        self.pred_proj = components.MLP(width, int(c["proj_hidden"]), width, norm_fn=nn.BatchNorm1d)
        if self.flags["state_token"] or self.flags["fusion"]:
            self.state_embed = _mlp(gw.STATE_DIM, int(c["state_hidden"]), width)
        if self.flags["state_head"]:
            self.state_head = _mlp(width, int(c["state_hidden"]), gw.STATE_DIM)
        flat = gw.TOKENS * width
        if self.flags["inverse"]:
            self.inverse_head = _mlp(2 * flat, int(c["inverse_hidden"]), len(gw.PLAYED_ACTION_DIMS))
        if self.flags["readout"]:
            self.readout_head = _mlp(flat, int(c["readout_hidden"]), 3)
        dim = gw.STATE_DIM
        self.register_buffer("state_mean", torch.zeros(dim))
        self.register_buffer("state_scale", torch.ones(dim))
        self.register_buffer("palm_mean", torch.zeros(3))
        self.register_buffer("palm_scale", torch.ones(3))
        self.register_buffer("moments_set", torch.tensor(False))

    # ----- normalisation ----------------------------------------------------------------------
    @torch.no_grad()
    def set_moments(self, state_mean, state_std, palm_mean, palm_std):
        dev = self.state_mean.device
        self.state_mean.copy_(torch.as_tensor(np.asarray(state_mean, np.float32), device=dev))
        scale = np.maximum(np.asarray(state_std, np.float64), gw.STATE_FLOOR_STD)
        self.state_scale.copy_(torch.as_tensor(scale.astype(np.float32), device=dev))
        self.palm_mean.copy_(torch.as_tensor(np.asarray(palm_mean, np.float32), device=dev))
        pscale = np.maximum(np.asarray(palm_std, np.float64), 1e-6).astype(np.float32)
        self.palm_scale.copy_(torch.as_tensor(pscale, device=dev))
        self.moments_set.fill_(True)

    def norm_state(self, s):
        if not bool(self.moments_set):
            raise ContractError("set the train moments before using states")
        return ((s - self.state_mean) / self.state_scale).clamp(-gw.STATE_CLIP, gw.STATE_CLIP)

    # ----- one step ---------------------------------------------------------------------------
    def step(self, z, a, state_token=None, fused=None):
        """One prediction step. ``z`` [B, 16, K]; ``a`` [B, 14]; ``state_token`` [B, K] (S, G);
        ``fused`` [B, K] added to every input token (C). Returns (next z, state-slot output)."""
        x = z if fused is None else z + fused[:, None]
        if self.flags["state_token"]:
            x = torch.cat((x, state_token[:, None]), 1)
        cond = self.action_encoder(a[:, None])  # [B, 1, K]: conditions every token of the frame
        out = self.predictor(x, cond)
        visual = out[:, : gw.TOKENS]
        b, t, d = visual.shape
        nxt = self.pred_proj(visual.reshape(b * t, d)).reshape(b, t, d)
        slot = out[:, gw.TOKENS] if self.flags["state_token"] else None
        return nxt, slot

    def _actions(self, a):
        return torch.zeros_like(a) if self.flags["zero_actions"] else a

    # ----- roll-out ---------------------------------------------------------------------------
    def rollout(self, z0, actions, s0=None):
        """Recursive roll-out from ``z0`` [B, 16, K] under ``actions`` [B, H, 14] and the raw start
        state ``s0`` [B, 28] (S, G, C). Returns predicted latents [B, H, 16, K] and, for G, the
        predicted standardised states [B, H, 28]."""
        actions = self._actions(actions)
        horizon = actions.shape[1]
        if horizon > int(self.config["max_horizon"]):
            raise ContractError("roll-out exceeds max_horizon")
        needs_state = self.flags["state_token"] or self.flags["fusion"]
        if needs_state and s0 is None:
            raise ContractError(f"arm {self.arm} needs the start state")
        sn = self.norm_state(s0) if needs_state else None
        token = self.state_embed(sn) if self.flags["state_token"] else None
        fused = self.state_embed(sn) if self.flags["fusion"] else None
        z, preds, states = z0, [], []
        for k in range(horizon):
            z, slot = self.step(z, actions[:, k], token, fused if k == 0 else None)
            preds.append(z)
            if self.flags["state_head"]:
                sn = sn + self.state_head(slot)
                states.append(sn)
                token = self.state_embed(sn)
            elif self.flags["state_token"]:
                token = slot
        out = torch.stack(preds, 1)
        return out, (torch.stack(states, 1) if states else None)

    # ----- training loss ----------------------------------------------------------------------
    def inverse(self, z_in, z_out):
        return self.inverse_head(torch.cat((z_in.flatten(1), z_out.flatten(1)), 1))

    def loss(self, z, a, s=None, palm=None):
        """§5's loss on one batch: ``z`` [B, T+1, 16, K], ``a`` [B, T, 14], ``s`` [B, T+1, 28]
        raw, ``palm`` [B, T+1, 3] raw. Returns (loss, metrics)."""
        b, t1 = z.shape[:2]
        horizon = t1 - 1
        a = self._actions(a)
        w = gw.LOSS_WEIGHTS
        sn = self.norm_state(s) if (self.flags["state_token"] or self.flags["fusion"]) else None
        # teacher-forced one-step: every frame t -> t + 1
        zt = z[:, :-1].reshape(b * horizon, gw.TOKENS, -1)
        at = a.reshape(b * horizon, -1)
        token = fused = None
        if self.flags["state_token"]:
            token = self.state_embed(sn[:, :-1].reshape(b * horizon, -1))
        if self.flags["fusion"]:
            fused = self.state_embed(sn[:, :-1].reshape(b * horizon, -1))
        one, slot = self.step(zt, at, token, fused)
        target = z[:, 1:].reshape(b * horizon, gw.TOKENS, -1)
        prediction_loss = (one - target).square().mean()
        recursive, rec_states = self.rollout(z[:, 0], a, s[:, 0] if s is not None else None)
        multistep = (recursive - z[:, 1:]).square().mean()
        loss = prediction_loss + w["multistep"] * multistep
        metrics = {"prediction_loss": prediction_loss.item(), "multistep_loss": multistep.item()}
        if self.flags["state_head"]:
            change = sn[:, 1:] - sn[:, :-1]
            tf = (self.state_head(slot) - change.reshape(b * horizon, -1)).square().mean()
            rec = (rec_states - sn[:, 1:]).square().mean()
            joint = 0.5 * (tf + rec)
            loss = loss + w["joint_change"] * joint
            metrics["joint_change_loss"] = joint.item()
        if self.flags["inverse"]:
            played = a[..., list(gw.PLAYED_ACTION_DIMS)]
            tf = (self.inverse(zt, one) - played.reshape(b * horizon, -1)).square().mean()
            inputs = torch.cat((z[:, :1], recursive[:, :-1]), 1)
            rec = (
                (
                    self.inverse(
                        inputs.reshape(b * horizon, gw.TOKENS, -1),
                        recursive.reshape(b * horizon, gw.TOKENS, -1),
                    )
                    - played.reshape(b * horizon, -1)
                )
                .square()
                .mean()
            )
            inv = 0.5 * (tf + rec)
            loss = loss + w["inverse"] * inv
            metrics["inverse_loss"] = inv.item()
        if self.flags["readout"]:
            pn = (palm - self.palm_mean) / self.palm_scale
            enc = (self.readout_head(z.flatten(2)) - pn).square().mean()
            pred = (self.readout_head(recursive.flatten(2)) - pn[:, 1:]).square().mean()
            readout = enc + pred
            loss = loss + w["readout"] * readout
            metrics["readout_loss"] = readout.item()
        metrics["loss"] = loss.item()
        return loss, metrics


def implementation_sha256() -> str:
    """sha256 over this file and the files it composes (the checkpoint's code identity)."""
    from embodied_jepa.models import lewm

    digest = hashlib.sha256()
    for path in (Path(__file__), Path(gw.__file__), Path(inspect.getfile(lewm))):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def predict_at(
    model: GroundedWM, start, commands, states, horizons, *, chunk: int = 512, device: str = "cpu"
) -> dict:
    """Evaluation roll-outs: numpy in, ``{h: float32 [N, 16, K]}`` out (eval mode, chunked)."""
    model.eval()
    hmax = max(horizons)
    out = {h: [] for h in horizons}
    for lo in range(0, len(start), chunk):
        z0 = torch.as_tensor(np.asarray(start[lo : lo + chunk], np.float32), device=device)
        a = torch.as_tensor(np.asarray(commands[lo : lo + chunk, :hmax], np.float32), device=device)
        s0 = None
        if states is not None:
            s0 = torch.as_tensor(np.asarray(states[lo : lo + chunk], np.float32), device=device)
        pred, _ = model.rollout(z0, a, s0)
        for h in horizons:
            out[h].append(pred[:, h - 1].float().cpu().numpy())
    result = {h: np.concatenate(v) for h, v in out.items()}
    for v in result.values():
        if not np.isfinite(v).all():
            raise ContractError("non-finite prediction")
    return result


@torch.no_grad()
def predict_states(
    model: GroundedWM, start, commands, states, horizons, *, chunk: int = 512, device: str = "cpu"
) -> dict:
    """G's predicted standardised states at the horizons (reported only)."""
    model.eval()
    out = {h: [] for h in horizons}
    for lo in range(0, len(start), chunk):
        z0 = torch.as_tensor(np.asarray(start[lo : lo + chunk], np.float32), device=device)
        a = torch.as_tensor(
            np.asarray(commands[lo : lo + chunk, : max(horizons)], np.float32), device=device
        )
        s0 = torch.as_tensor(np.asarray(states[lo : lo + chunk], np.float32), device=device)
        _, st = model.rollout(z0, a, s0)
        for h in horizons:
            out[h].append(st[:, h - 1].cpu().numpy())
    return {h: np.concatenate(v) for h, v in out.items()}
