"""Backend-agnostic physical readout heads over (predicted) world-model latents.

The heads are shared code: every backend that enables ``readout_heads`` gets the
same module, loss and physical decoding. They are trained on privileged simulator
labels *as targets only* (built outside the model package); at run time they read
nothing but a latent the model itself produced, so a planner can use an
object-aware cost without simulator truth and without inspecting a latent.

Regression targets are undefined once the apple has fallen off the table (it is
then outside the workspace and the camera view): their loss is masked by the
``apple_dropped`` target, which is itself a declared probability readout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from embodied_jepa.contracts import ContractError


@dataclass(frozen=True)
class ReadoutSpec:
    name: str
    width: int
    kind: str  # "regression" (physical units) or "probability"
    unit: str
    scale: float = 1.0  # regression: physical units per network unit


READOUTS = (
    ReadoutSpec("palm_minus_apple", 3, "regression", "m", 0.05),
    ReadoutSpec("apple_height", 1, "regression", "m", 0.05),
    ReadoutSpec("apple_minus_plate", 3, "regression", "m", 0.05),
    ReadoutSpec("hand_contact", 1, "probability", "1"),
    ReadoutSpec("apple_held", 1, "probability", "1"),
    ReadoutSpec("apple_dropped", 1, "probability", "1"),
)
READOUT_NAMES = tuple(spec.name for spec in READOUTS)
READOUT_VERSION = "object_readout_v2"
VALIDITY = "apple_dropped"  # regression targets count only where this target is 0


class ReadoutHeads(nn.Module):
    """LayerNorm + MLP from one latent vector to every declared readout."""

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, sum(spec.width for spec in READOUTS)),
        )

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(latent)
        result, start = {}, 0
        for spec in READOUTS:
            # Contiguous copies: some MPS kernels reject strided views.
            result[spec.name] = raw[..., start : start + spec.width].contiguous()
            start += spec.width
        return result

    def physical(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        """Regression readouts in physical units; probability readouts in [0,1]."""
        raw = self(latent)
        return {
            spec.name: raw[spec.name] * spec.scale
            if spec.kind == "regression"
            else torch.sigmoid(raw[spec.name])
            for spec in READOUTS
        }

    def loss(self, latent: torch.Tensor, targets: dict[str, torch.Tensor]):
        """Mean loss over readouts; regression smooth-L1 in scaled units, BCE otherwise.

        ``latent`` is ``[..., D]``; each target is ``[..., width]`` on the same prefix.
        """
        missing = set(READOUT_NAMES) - set(targets)
        if missing:
            raise ContractError(f"missing readout targets: {sorted(missing)}")
        raw = self(latent)
        valid = 1.0 - targets[VALIDITY].contiguous()
        terms = {}
        for spec in READOUTS:
            target = targets[spec.name].contiguous()
            if target.shape != raw[spec.name].shape:
                raise ContractError(f"readout target {spec.name} has the wrong shape")
            if spec.kind == "regression":
                error = F.smooth_l1_loss(raw[spec.name], target / spec.scale, reduction="none")
                weight = valid.expand_as(error)
                terms[spec.name] = (error * weight).sum() / weight.sum().clamp_min(1.0)
            else:
                terms[spec.name] = F.binary_cross_entropy_with_logits(raw[spec.name], target)
        return sum(terms.values()) / len(terms), terms
