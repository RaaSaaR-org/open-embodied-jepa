"""Behaviour-cloned visuomotor policies over a FROZEN world-model encoder (TASK-056).

A policy is the control-side sibling of a planner: both turn an observation into an action
through the model contract, and the benchmark runner selects one by config. This module owns
the policy; it does **not** touch ``models/base.py`` or ``models/lewm.py``, whose
``implementation_sha256`` the v4 checkpoints enforce and on which the whole protocol's frozen
feature source depends.

The world model is used here **only** through its public surface -- ``image_features`` for the
image pathway and the frozen ``state_mean``/``state_scale`` buffers for proprioception. No
rollout, no ``predict``, no counterfactual. That is deliberate and is the protocol's whole
argument: `docs/experiments/apple_policy_v1.md` shows the model's action-conditioned prediction
has failed its gate in every generation, while its h~0 perception measured 0.49 cm on the close
phase. A policy re-observes every 50 ms and never needs a multi-step rollout.

Three feature sources, matching the protocol's arms:

* :class:`NoEncoder` -- arm **A0**, proprioception only. The protocol's most informative
  control, not a formality: the pre-flight measured a P2 control ratio of 0.682, meaning most
  of the *relative* palm-apple accuracy is available from proprioception alone, so a
  proprioception-only policy may be strong and gate G2 is what decides whether anything visual
  is happening at all.
* :class:`FrozenEncoder` -- arms **A1** (randomly initialized weights) and **A2** (the frozen
  E0 checkpoint). ``image_features`` is used rather than ``encode`` because E0 was trained with
  ``state_fusion: true``: ``encode`` would fold proprioception into the latent and make the A0
  ablation dishonest.
* A fine-tuned encoder (arm **A3**) uses the same class with ``frozen=False``.

The policy predicts only the **seven free action dimensions**. The other seven are pinned by
the protocol's frozen action bounds (left arm zero, ``left_grasp`` -1), so predicting them would
be free accuracy. The assembled 14-vector is contract-valid and is handed to the embodiment's
unchanged feasibility projection exactly as the collector's commands were.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from embodied_jepa.contracts import (
    ACTION_DIM,
    ACTION_NAMES,
    EE_DELTA_GRASP_V0,
    ContractError,
    RobotState,
    StateSchema,
)
from embodied_jepa.models.base import VisualModel

POLICY_VERSION = "cloned_bc_v1"

#: Indices of the action components a policy predicts. Everything else is pinned by the
#: protocol's frozen bounds in ``configs/apple_wm_v4.yaml``.
FREE_ACTION_INDICES = (6, 7, 8, 9, 10, 11, 13)
FREE_ACTION_NAMES = tuple(ACTION_NAMES[i] for i in FREE_ACTION_INDICES)
#: Value each pinned component takes. Left arm holds still; the left hand stays open.
PINNED_ACTION_VALUES = {i: 0.0 for i in range(6)} | {12: -1.0}
GRASP_INDEX = 13


def _check_pinned_bounds(lower, upper) -> None:
    """Refuse to build a policy against bounds that do not pin what this module pins."""
    lower = np.asarray(lower, np.float64)
    upper = np.asarray(upper, np.float64)
    if lower.shape != (ACTION_DIM,) or upper.shape != (ACTION_DIM,):
        raise ContractError("action bounds must be 14-vectors")
    for index, value in PINNED_ACTION_VALUES.items():
        if lower[index] != value or upper[index] != value:
            raise ContractError(
                f"{ACTION_NAMES[index]} is pinned to {value} by this policy but the configured "
                f"bounds allow [{lower[index]}, {upper[index]}]"
            )


def state_scale_from_moments(mean, std) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce ``VisualModel.fit_state_normalization``'s scaling, from its own constants.

    The constants are read off :class:`VisualModel` rather than copied, so they cannot drift;
    only the three lines of arithmetic are repeated, and ``tests/test_policy.py`` asserts the
    result equals a fitted model's buffers exactly. Arm A0 has no encoder to borrow buffers
    from, which is why this exists at all.
    """
    mean = np.asarray(mean, np.float64)
    std = np.asarray(std, np.float64)
    if mean.shape != std.shape or mean.ndim != 1:
        raise ContractError("state moments must be matching 1-D arrays")
    if not (np.isfinite(mean).all() and np.isfinite(std).all()) or (std < 0).any():
        raise ContractError("state moments must be finite with nonnegative std")
    constant = std < VisualModel.STATE_NOISE_STD
    scale = np.where(
        constant, VisualModel.STATE_CONSTANT_SCALE, np.maximum(std, VisualModel.STATE_FLOOR_STD)
    )
    return mean.astype(np.float32), scale.astype(np.float32)


def normalize_state(values, mask, mean, scale) -> np.ndarray:
    """Train-normalized, clipped, mask-zeroed proprioception -- ``VisualModel``'s own policy."""
    values = np.asarray(values, np.float32)
    mask = np.asarray(mask, np.bool_)
    if values.shape != mask.shape or values.shape[-1] != mean.shape[0]:
        raise ContractError("state values and mask must match the fitted moments")
    normalized = (values - mean) / scale
    clipped = np.clip(normalized, -VisualModel.STATE_CLIP, VisualModel.STATE_CLIP)
    return (clipped * mask).astype(np.float32)


class NoEncoder:
    """Arm A0's image pathway: there isn't one."""

    kind = "none"
    feature_dim = 0
    trainable = False

    def features(self, images) -> np.ndarray:  # noqa: ARG002 - signature parity
        raise ContractError("the proprioception-only arm has no image pathway")

    def parameters(self):
        return iter(())

    def provenance(self) -> dict[str, Any]:
        return {"kind": self.kind}


class FrozenEncoder:
    """A world-model encoder used as a feature source, through its public surface only.

    ``frozen=True`` (arms A1/A2) puts the model in eval mode and detaches its output, so no
    gradient reaches it and the checkpoint it came from stays authoritative. ``frozen=False``
    (arm A3) leaves it trainable.
    """

    kind = "visual_model"

    def __init__(self, model: VisualModel, *, frozen: bool = True, camera: str = "onboard_rgb"):
        if not isinstance(model, VisualModel):
            raise ContractError("a policy feature source must be a VisualModel")
        if camera not in model.camera_names:
            raise ContractError(f"model does not consume camera {camera!r}")
        self.model = model
        self.frozen = bool(frozen)
        self.camera = camera
        self.feature_dim = int(model.config["latent_dim"])
        self.trainable = not self.frozen
        if self.frozen:
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    def features(self, images) -> torch.Tensor:
        """Image-only features. **Never** ``encode``: that fuses proprioception in E0."""
        context = torch.no_grad() if self.frozen else torch.enable_grad()
        with context:
            values, prefix = self.model.image_features(images)
        if len(prefix) != 1:
            raise ContractError("policy features expect a flat batch of observations")
        return values.detach() if self.frozen else values

    def parameters(self):
        return self.model.parameters()

    def provenance(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "backend": self.model.backend,
            "frozen": self.frozen,
            "camera": self.camera,
            "latent_dim": self.feature_dim,
            "model_implementation_sha256": self.model.implementation_sha256,
            "model_parameters": int(sum(p.numel() for p in self.model.parameters())),
        }


@dataclass(frozen=True)
class PolicyDefaults:
    hidden_dim: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0


class ClonedPolicy(nn.Module):
    """An MLP head over (image features ‖ normalized proprioception) -> 7 free actions."""

    version = POLICY_VERSION
    defaults = {
        "hidden_dim": 512,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip": 1.0,
    }

    def __init__(
        self,
        state_schema: StateSchema,
        feature_source,
        *,
        device: str = "cpu",
        seed: int = 0,
        config: dict | None = None,
        metadata: dict | None = None,
    ):
        super().__init__()
        if device not in ("cpu", "mps"):
            raise ContractError("validated policy devices are cpu and mps")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ContractError("MPS was requested but is unavailable")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ContractError("seed must be a nonnegative integer")
        overrides = dict(config or {})
        unknown = overrides.keys() - self.defaults.keys()
        if unknown:
            raise ContractError(f"unknown policy config keys: {sorted(unknown)}")
        self.config = self.defaults | overrides
        if type(self.config["hidden_dim"]) is not int or self.config["hidden_dim"] < 1:
            raise ContractError("hidden_dim must be a positive integer")
        for name in ("learning_rate", "gradient_clip"):
            if not np.isfinite(self.config[name]) or self.config[name] <= 0:
                raise ContractError(f"{name} must be positive and finite")
        if not np.isfinite(self.config["weight_decay"]) or self.config["weight_decay"] < 0:
            raise ContractError("weight_decay must be nonnegative and finite")

        self.device_name = device
        self.seed = seed
        self.state_schema = state_schema
        self.features = feature_source
        self.updates = 0
        self.metadata = copy.deepcopy(dict(metadata or {}))
        json.dumps(self.metadata, allow_nan=False)  # JSON-safe provenance only

        self.register_buffer("state_mean", torch.zeros(state_schema.dimension))
        self.register_buffer("state_scale", torch.ones(state_schema.dimension))
        self.register_buffer("state_normalization_fitted", torch.tensor(False))

        width = feature_source.feature_dim + state_schema.dimension
        if width < 1:
            raise ContractError("a policy needs at least one input feature")
        self.input_dim = width
        generator = torch.Generator().manual_seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator)))
            hidden = self.config["hidden_dim"]
            self.head = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, len(FREE_ACTION_INDICES)),
            )
        self.to(device)
        trainable = list(self.head.parameters())
        if feature_source.trainable:
            trainable += [p for p in feature_source.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

    # ----- normalization -------------------------------------------------------------
    @torch.no_grad()
    def fit_state_normalization(self, mean, scale, *, training_episode_ids) -> None:
        """Freeze the train-split moments once, before any update."""
        if bool(self.state_normalization_fitted) or self.updates:
            raise ContractError("state normalization is frozen; create a new policy to refit")
        ids = tuple(training_episode_ids)
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(x, str) for x in ids):
            raise ContractError("training_episode_ids must be unique episode names")
        mean = np.asarray(mean, np.float32)
        scale = np.asarray(scale, np.float32)
        if mean.shape != (self.state_schema.dimension,) or scale.shape != mean.shape:
            raise ContractError("state moments must match the state schema")
        if not (np.isfinite(mean).all() and np.isfinite(scale).all()) or (scale <= 0).any():
            raise ContractError("state moments must be finite with positive scale")
        self.state_mean.copy_(torch.as_tensor(mean))
        self.state_scale.copy_(torch.as_tensor(scale))
        self.state_normalization_fitted.fill_(True)
        self.metadata["state_normalization_episodes_sha256"] = hashlib.sha256(
            json.dumps(sorted(ids)).encode()
        ).hexdigest()

    def normalized_state(self, values, mask) -> torch.Tensor:
        if not bool(self.state_normalization_fitted):
            raise ContractError("fit training-only state normalization before using a policy")
        array = normalize_state(
            values, mask, self.state_mean.cpu().numpy(), self.state_scale.cpu().numpy()
        )
        return torch.from_numpy(array).to(self.device_name)

    # ----- forward and control -------------------------------------------------------
    def forward(self, image_features: torch.Tensor | None, state: torch.Tensor) -> torch.Tensor:
        if image_features is None:
            if self.features.feature_dim:
                raise ContractError("this policy expects image features")
            joined = state
        else:
            if image_features.shape[0] != state.shape[0]:
                raise ContractError("image and state batch sizes disagree")
            joined = torch.cat((image_features, state), dim=-1)
        if joined.shape[-1] != self.input_dim:
            raise ContractError("policy input width does not match its head")
        return self.head(joined)

    def predict_free(self, images, robot_state: RobotState) -> np.ndarray:
        """Predicted free-action components ``[B, 7]`` for a batch of observations."""
        if not isinstance(robot_state, RobotState) or robot_state.schema != self.state_schema:
            raise ContractError("incompatible robot-state schema")
        self.eval()
        with torch.no_grad():
            visual = self.features.features(images) if self.features.feature_dim else None
            state = self.normalized_state(robot_state.values, robot_state.mask)
            free = self(visual, state)
        return free.detach().cpu().numpy().astype(np.float32)

    def act(self, images, robot_state: RobotState) -> np.ndarray:
        """One contract-valid normalized 14-D action for a single observation.

        The output is **clipped into [-1, 1]** because a regression head is not bounded and
        ``validate_actions`` rejects anything outside it. Clipping is recorded per command by
        the runner so a policy that spends its life on the bound is visible rather than silent.
        """
        free = self.predict_free(images, robot_state)
        if free.shape[0] != 1:
            raise ContractError("act handles one observation at a time")
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for index, value in PINNED_ACTION_VALUES.items():
            action[index] = value
        action[list(FREE_ACTION_INDICES)] = free[0]
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    # ----- checkpoint envelope -------------------------------------------------------
    @property
    def implementation_sha256(self) -> str:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def save(self, path) -> None:
        path = Path(path)
        if path.exists():
            raise FileExistsError("refusing to overwrite an existing policy checkpoint")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "version": self.version,
                "config": self.config,
                "state_schema": {
                    "names": list(self.state_schema.names),
                    "units": list(self.state_schema.units),
                    "version": self.state_schema.version,
                    "required": list(self.state_schema.required),
                },
                "action_schema": {
                    "version": EE_DELTA_GRASP_V0.version,
                    "names": list(EE_DELTA_GRASP_V0.names),
                },
                "free_action_indices": list(FREE_ACTION_INDICES),
                "implementation_sha256": self.implementation_sha256,
                "feature_source": self.features.provenance(),
                "weights": self.head.state_dict(),
                # The normalization buffers live on the module, not the head. Saving only the
                # head would silently drop them and a reloaded policy would normalize with the
                # identity, so they are stored explicitly and restored below.
                "state_mean": self.state_mean.cpu(),
                "state_scale": self.state_scale.cpu(),
                "state_normalization_fitted": self.state_normalization_fitted.cpu(),
                "optimizer": self.optimizer.state_dict(),
                "metadata": self.metadata,
                "seed": self.seed,
                "updates": self.updates,
            },
            path,
        )

    def load(self, path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        expected = {
            "format_version": 1,
            "version": self.version,
            "config": self.config,
            "free_action_indices": list(FREE_ACTION_INDICES),
            "implementation_sha256": self.implementation_sha256,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ContractError(f"policy checkpoint {key} is incompatible")
        self.head.load_state_dict(checkpoint["weights"], strict=True)
        for name in ("state_mean", "state_scale", "state_normalization_fitted"):
            if name not in checkpoint:
                raise ContractError(f"policy checkpoint is missing {name}")
            getattr(self, name).copy_(checkpoint[name].to(self.device_name))
        if not bool(self.state_normalization_fitted):
            raise ContractError("policy checkpoint carries unfitted state normalization")
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.metadata = checkpoint["metadata"]
        self.seed = checkpoint["seed"]
        self.updates = checkpoint["updates"]
        self.eval()
