"""Backend-agnostic physical readout heads over (predicted) world-model latents.

The heads are shared code: every backend that enables ``readout_heads`` gets the
same module, loss and physical decoding. They are trained on privileged simulator
labels *as targets only* (built outside the model package); at run time they read
nothing but a latent the model itself produced, so a planner can use an
object-aware cost without simulator truth and without inspecting a latent.

Regression targets are undefined once the apple has fallen off the table (it is
then outside the workspace and the camera view): their loss is masked by the
``apple_dropped`` target, which is itself a declared probability readout.

Two v3 (TASK-052) extensions, both off by default so a v2-shaped model keeps its
exact loss:

- **Auxiliary readouts.** ``apple_position`` and ``palm_position`` (both relative
  to a fixed declared workspace origin) carry weight ``auxiliary_weight``. At
  weight 0 they contribute no term and no gradient, and the loss is exactly the
  mean over the six core readouts. Above 0 they force the latent to localize the
  apple absolutely instead of only its offset from the palm.
- **Frame weighting.** A per-frame weight (supplied by the caller, derived from
  the targets) multiplies the *regression* terms only. TASK-050 measured the
  failure as concentrated in the frames where the palm-apple offset is moving,
  which are a minority of frames; the probability heads are already near-perfect
  and are left unweighted.
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
    group: str = "core"  # "core" (always trained) or "auxiliary" (weighted)


READOUTS = (
    ReadoutSpec("palm_minus_apple", 3, "regression", "m", 0.05),
    ReadoutSpec("apple_height", 1, "regression", "m", 0.05),
    ReadoutSpec("apple_minus_plate", 3, "regression", "m", 0.05),
    ReadoutSpec("hand_contact", 1, "probability", "1"),
    ReadoutSpec("apple_held", 1, "probability", "1"),
    ReadoutSpec("apple_dropped", 1, "probability", "1"),
    ReadoutSpec("apple_position", 3, "regression", "m", 0.05, "auxiliary"),
    ReadoutSpec("palm_position", 3, "regression", "m", 0.05, "auxiliary"),
)
READOUT_NAMES = tuple(spec.name for spec in READOUTS)
CORE_READOUT_NAMES = tuple(spec.name for spec in READOUTS if spec.group == "core")
READOUT_VERSION = "object_readout_v3"
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

    def loss(self, latent, targets, *, frame_weight=None, auxiliary_weight=0.0):
        """Weighted mean loss; regression smooth-L1 in scaled units, BCE otherwise.

        ``latent`` is ``[..., D]``; each target is ``[..., width]`` on the same prefix.
        ``frame_weight`` is an optional ``[..., 1]`` nonnegative per-frame weight applied
        to the regression terms only. A readout group weighted 0 contributes no term.

        The divisor is the number of **core** readouts, never the weighted total, so the
        auxiliary heads *add* supervision instead of diluting the core objective: at
        ``auxiliary_weight = 0`` the result is exactly the v2 mean over the six core
        terms, and above 0 each core term keeps the same weight it had.
        """
        if auxiliary_weight < 0:
            raise ContractError("auxiliary readout weight must be nonnegative")
        missing = set(READOUT_NAMES) - set(targets)
        if missing:
            raise ContractError(f"missing readout targets: {sorted(missing)}")
        raw = self(latent)
        valid = 1.0 - targets[VALIDITY].contiguous()
        if frame_weight is not None:
            if frame_weight.shape[:-1] != valid.shape[:-1] or frame_weight.shape[-1] != 1:
                raise ContractError("frame weight must broadcast over the readout prefix")
            valid = valid * frame_weight
        terms = {}
        for spec in READOUTS:
            group_weight = 1.0 if spec.group == "core" else auxiliary_weight
            if group_weight == 0.0:
                continue
            target = targets[spec.name].contiguous()
            if target.shape != raw[spec.name].shape:
                raise ContractError(f"readout target {spec.name} has the wrong shape")
            if spec.kind == "regression":
                error = F.smooth_l1_loss(raw[spec.name], target / spec.scale, reduction="none")
                weight = valid.expand_as(error)
                terms[spec.name] = (error * weight).sum() / weight.sum().clamp_min(1.0)
            else:
                terms[spec.name] = F.binary_cross_entropy_with_logits(raw[spec.name], target)
        weighted = sum(
            (1.0 if spec.group == "core" else auxiliary_weight) * terms[spec.name]
            for spec in READOUTS
            if spec.name in terms
        )
        return weighted / len(CORE_READOUT_NAMES), terms
