"""Shared boundary validation and checkpoint envelope, without model-specific planning."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from embodied_jepa.contracts import (
    EE_DELTA_GRASP_V0,
    Capabilities,
    ContractError,
    RobotState,
    SequenceBatch,
    StateSchema,
    validate_actions,
)
from embodied_jepa.models import readout as readout_module


@dataclass(frozen=True)
class VisualLatent:
    """Private backend-owned carrier; the planner must never inspect this payload."""

    values: torch.Tensor
    owner: object


def _statistics(values):
    values = values.double()
    if values.shape[0] < 2:
        raise ContractError("latent statistics need at least two samples")
    std = values.std(0, unbiased=False)
    energy = torch.linalg.svdvals(values - values.mean(0)).square()
    total = energy.sum().item()
    rank = 0.0
    if total > 0:
        probabilities = energy[energy > 0] / total
        rank = (-(probabilities * probabilities.log()).sum()).exp().item()
    return {
        "samples": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        "latent_std_mean": std.mean().item(),
        "latent_std_min": std.min().item(),
        "collapsed_fraction": (std < 0.01).double().mean().item(),
        "effective_rank": rank,
    }


class VisualModel(nn.Module):
    backend = "abstract"
    defaults: dict[str, Any] = {
        "camera": "onboard_rgb",
        "image_size": 64,
        "latent_dim": 64,
        "hidden_dim": 128,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "max_horizon": 64,
        "candidate_chunk_size": 128,
        "gradient_clip": 1.0,
        # Shared (backend-agnostic) extensions, off by default so earlier models keep
        # their image-only latents. state_fusion adds a learned embedding of the
        # train-normalized proprioception to every latent; readout_heads adds the
        # declared physical readouts of embodied_jepa.models.readout.
        "state_fusion": False,
        "readout_heads": False,
        "readout_weight": 1.0,
        "readout_hidden_dim": 256,
        # Multi-camera fusion (TASK-052). ``None`` keeps the single ``camera``; a list of
        # camera names embeds each one with the backend's own encoder and sums a learned
        # per-camera projection, which is a concatenate-then-project fusion written once
        # in shared code, so the backend swap stays a one-line config change.
        "cameras": None,
        # Readout loss shaping (TASK-052), both off by default so a v2-shaped model keeps
        # its exact loss. ``readout_moving_weight`` adds that much extra weight to the
        # regression terms of frames whose true palm-apple offset has moved at least
        # ``readout_moving_threshold_m`` from the window start; ``readout_auxiliary_weight``
        # switches on the auxiliary absolute-position readouts.
        "readout_moving_weight": 0.0,
        "readout_moving_threshold_m": 0.01,
        "readout_auxiliary_weight": 0.0,
    }
    # Proprioception scaling policy (TASK-052; see fit_state_normalization).
    STATE_NOISE_STD = 1e-4  # below this a dimension never moved: treat it as constant
    STATE_FLOOR_STD = 1e-2  # TASK-050's floor, kept for every dimension above the noise
    STATE_CONSTANT_SCALE = 1.0  # constant dimensions enter at their physical unit
    STATE_CLIP = 10.0  # hard backstop on every normalized value

    def __init__(self, state_schema: StateSchema, device="cpu", seed=0, config=None, metadata=None):
        super().__init__()
        if device not in ("cpu", "mps"):
            raise ContractError("validated model devices are cpu and mps")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ContractError("MPS was requested but is unavailable")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ContractError("seed must be a nonnegative integer")
        self.device_name = device
        self.state_schema = state_schema
        self.seed = seed
        overrides = dict(config or {})
        unknown = overrides.keys() - self.defaults.keys()
        if unknown:
            raise ContractError(f"unknown {self.backend} config keys: {sorted(unknown)}")
        self.config = self.defaults | overrides
        for name in (
            "image_size",
            "latent_dim",
            "hidden_dim",
            "max_horizon",
            "candidate_chunk_size",
            "readout_hidden_dim",
        ):
            if type(self.config[name]) is not int or self.config[name] < 1:
                raise ContractError(f"{name} must be a positive integer")
        for name in ("state_fusion", "readout_heads"):
            if type(self.config[name]) is not bool:
                raise ContractError(f"{name} must be a boolean")
        for name in (
            "readout_weight",
            "readout_moving_weight",
            "readout_moving_threshold_m",
            "readout_auxiliary_weight",
        ):
            if not np.isfinite(self.config[name]) or self.config[name] < 0:
                raise ContractError(f"{name} must be nonnegative and finite")
        for name in ("learning_rate", "gradient_clip"):
            if not np.isfinite(self.config[name]) or self.config[name] <= 0:
                raise ContractError(f"{name} must be positive and finite")
        if not np.isfinite(self.config["weight_decay"]) or self.config["weight_decay"] < 0:
            raise ContractError("weight_decay must be nonnegative and finite")
        if not isinstance(self.config["camera"], str) or not self.config["camera"]:
            raise ContractError("camera must be a nonempty name")
        extra = self.config["cameras"]
        if extra is not None:
            names = list(extra)
            if (
                not names
                or len(set(names)) != len(names)
                or any(not isinstance(name, str) or not name for name in names)
            ):
                raise ContractError("cameras must be unique nonempty names")
            self.config["cameras"] = names  # list, so the checkpoint config round-trips
        self.metadata = copy.deepcopy(dict(metadata or {}))
        # Only JSON-safe provenance is accepted; no executable checkpoint objects.
        json.dumps(self.metadata, allow_nan=False)
        self.updates = 0
        self._owner = object()
        self._rng = torch.Generator().manual_seed(seed).get_state()
        self._mps_rng = None
        self.register_buffer("latent_variance", torch.ones(self.config["latent_dim"]))
        if self.config["state_fusion"]:
            dimension = state_schema.dimension
            self.register_buffer("state_mean", torch.zeros(dimension))
            self.register_buffer("state_scale", torch.ones(dimension))
            self.register_buffer("state_normalization_fitted", torch.tensor(False))

    @property
    def camera_names(self):
        """Every camera this model consumes, in fusion order."""
        return tuple(self.config["cameras"] or (self.config["camera"],))

    @property
    def fuses_cameras(self):
        """Whether the learned per-camera projection is used.

        It follows the *presence* of an explicit ``cameras`` list, not its length, so a
        one-camera and a two-camera configured model differ by exactly the extra camera's
        projection and summand. Without the key the image pathway is the backend's own
        ``embed(pixels)``, unchanged from every earlier model.
        """
        return self.config["cameras"] is not None

    @property
    def declared_readouts(self):
        """Readouts this model actually trains; untrained heads are never declared."""
        if not self.config["readout_heads"]:
            return ()
        if self.config["readout_auxiliary_weight"] > 0:
            return readout_module.READOUT_NAMES
        return readout_module.CORE_READOUT_NAMES

    @property
    def capabilities(self):
        return Capabilities(
            EE_DELTA_GRASP_V0,
            self.state_schema,
            self.config["max_horizon"],
            supported_devices=("cpu", "mps"),
            readouts=self.declared_readouts,
        )

    @contextmanager
    def rng_scope(self):
        """Isolate backend stochastic state so comparisons don't perturb each other."""
        cpu_before = torch.get_rng_state()
        mps_before = torch.mps.get_rng_state() if self.device_name == "mps" else None
        torch.set_rng_state(self._rng)
        if self.device_name == "mps":
            if self._mps_rng is None:
                torch.mps.manual_seed(self.seed)
            else:
                torch.mps.set_rng_state(self._mps_rng)
        try:
            yield
        finally:
            self._rng = torch.get_rng_state()
            torch.set_rng_state(cpu_before)
            if self.device_name == "mps":
                self._mps_rng = torch.mps.get_rng_state()
                torch.mps.set_rng_state(mps_before)

    def finish_init(self):
        # Shared modules are built after the backend's own, so a disabled extension
        # leaves the backend's initialization stream unchanged.
        with self.rng_scope():
            if self.config["state_fusion"]:
                hidden = self.config["hidden_dim"]
                self.state_encoder = nn.Sequential(
                    nn.Linear(self.state_schema.dimension, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, self.config["latent_dim"]),
                )
            if self.config["readout_heads"]:
                self.readout_head = readout_module.ReadoutHeads(
                    self.config["latent_dim"], self.config["readout_hidden_dim"]
                )
            # The camera fusion is drawn last, so adding a camera does not shift the
            # initialization of any earlier module: a one-camera and a two-camera model
            # with the same seed share their state-encoder and readout-head weights, and
            # a camera ablation differs only by the extra camera's own projection.
            if self.fuses_cameras:
                dimension = self.config["latent_dim"]
                # Concatenate-then-project, written as one projection per camera: only
                # the first carries the bias, so the fusion has no redundant offset.
                self.camera_fusion = nn.ModuleDict(
                    {
                        name: nn.Linear(dimension, dimension, bias=index == 0)
                        for index, name in enumerate(self.camera_names)
                    }
                )
        self.to(self.device_name)
        self.optimizer = torch.optim.AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        self.eval()

    def pixels(self, images, *, sequence=False, camera=None):
        name = camera or self.config["camera"]
        if name not in images:
            raise ContractError(f"missing configured camera {name!r}")
        array = images[name]
        rank = 5 if sequence else 4
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.uint8
            or array.ndim != rank
            or array.shape[-1] != 3
            or any(d < 1 for d in array.shape)
        ):
            raise ContractError("model camera requires nonempty uint8 channels-last RGB")
        prefix = array.shape[:-3]
        # Copy read-only canonical arrays; torch must not alias validated snapshots.
        tensor = torch.from_numpy(np.array(array, copy=True)).to(self.device_name)
        tensor = tensor.reshape(-1, *array.shape[-3:]).permute(0, 3, 1, 2).float() / 255
        size = self.config["image_size"]
        if tensor.shape[-2:] != (size, size):
            tensor = F.interpolate(
                tensor, (size, size), mode="bilinear", align_corners=False, antialias=True
            )
        return tensor, prefix

    def image_features(self, images, *, sequence=False, target=False):
        """Fused image embedding over every configured camera, plus the batch prefix.

        Without a configured ``cameras`` list this is exactly the backend's own
        embedding. With one, each camera is embedded by the same backend encoder and
        passed through its own learned projection before the sum; no backend code
        changes, and a one-camera list keeps its projection so that adding a camera is a
        single controlled factor.
        """
        embed = self.goal_embed if target else self.embed
        names = self.camera_names
        fused, prefix = None, None
        for name in names:
            pixels, shape = self.pixels(images, sequence=sequence, camera=name)
            if prefix is not None and shape != prefix:
                raise ContractError("configured cameras disagree on the batch shape")
            prefix = shape
            features = embed(pixels)
            if self.fuses_cameras:
                features = self.camera_fusion[name](features)
            fused = features if fused is None else fused + features
        return fused, prefix

    def check_batch(self, batch):
        if not isinstance(batch, SequenceBatch):
            raise ContractError("train/validation requires a canonical SequenceBatch")
        self.capabilities.require(
            action_schema=batch.action_schema,
            state_schema=batch.state_schema,
            horizon=batch.horizon,
            device=self.device_name,
        )

    def check_latent(self, latent, rank):
        if not isinstance(latent, VisualLatent) or latent.owner is not self._owner:
            raise ContractError("latent belongs to another model instance or checkpoint")
        if latent.values.ndim != rank or not torch.isfinite(latent.values).all():
            raise ContractError("invalid latent tensor")
        return latent.values

    @torch.no_grad()
    def encode(self, observation, robot_state: RobotState):
        self.eval()
        if not isinstance(robot_state, RobotState) or robot_state.schema != self.state_schema:
            raise ContractError("incompatible robot-state schema")
        latent, prefix = self.image_features(observation)
        if prefix[0] != robot_state.values.shape[0]:
            raise ContractError("image/state batch dimensions disagree")
        if self.config["state_fusion"]:
            latent = latent + self.state_features(robot_state.values, robot_state.mask)
        return VisualLatent(latent, self._owner)

    @torch.no_grad()
    def encode_goal(self, goal):
        self.eval()
        if self.config["state_fusion"]:
            raise ContractError(
                "state-fused latents have no image-only goal encoding; plan with declared readouts"
            )
        latent, _ = self.image_features(goal, target=True)
        return VisualLatent(latent, self._owner)

    # ----- shared state fusion and readouts ------------------------------------------
    @torch.no_grad()
    def fit_state_normalization(self, mean, std, *, training_episode_ids):
        """Freeze train-split proprioception moments once, before any update.

        TASK-050 divided every dimension by ``max(std, 0.01)``. Its review pointed out
        that a dimension which never moves in training is then amplified up to a
        hundredfold relative to a normally varying one, so a later closed-loop deviation
        enters far outside the training distribution. Two changes address that without
        throwing away dimensions that do carry signal:

        - a dimension whose train std is below ``STATE_NOISE_STD`` is treated as
          constant and scaled by ``STATE_CONSTANT_SCALE`` (its physical unit), so an
          unseen deviation enters at physical scale instead of amplified;
        - everything else keeps TASK-050's ``max(std, STATE_FLOOR_STD)``, and every
          normalized value is clipped to ``STATE_CLIP``, which bounds the remaining
          amplification instead of removing information.

        On this corpus that is 30 constant dimensions (std 1.7e-6 to 9.4e-5), 13 floored
        dimensions (1.3e-4 to 6.1e-3, which reach only 0.01-0.61 in normalized units and
        are kept precisely because muting them would delete real signal) and 43 dimensions
        scaled by their own std. The clip is inert on the train split.
        """
        if not self.config["state_fusion"]:
            raise ContractError("state normalization requires state_fusion")
        if bool(self.state_normalization_fitted) or self.updates:
            raise ContractError("state normalization is frozen; create a new model to refit")
        ids = tuple(training_episode_ids)
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(x, str) for x in ids):
            raise ContractError("training_episode_ids must be unique episode names")
        mean = np.asarray(mean, np.float64)
        std = np.asarray(std, np.float64)
        shape = (self.state_schema.dimension,)
        if mean.shape != shape or std.shape != shape:
            raise ContractError("state moments must match the state schema")
        if not (np.isfinite(mean).all() and np.isfinite(std).all()) or (std < 0).any():
            raise ContractError("state moments must be finite with nonnegative std")
        constant = std < self.STATE_NOISE_STD
        scale = np.where(constant, self.STATE_CONSTANT_SCALE, np.maximum(std, self.STATE_FLOOR_STD))
        self.state_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.state_scale.copy_(torch.as_tensor(scale, dtype=torch.float32))
        self.state_normalization_fitted.fill_(True)
        self.metadata["state_normalization_episodes_sha256"] = hashlib.sha256(
            json.dumps(sorted(ids)).encode()
        ).hexdigest()
        self.metadata["state_normalization_policy"] = {
            "noise_std": self.STATE_NOISE_STD,
            "floor_std": self.STATE_FLOOR_STD,
            "constant_scale": self.STATE_CONSTANT_SCALE,
            "clip": self.STATE_CLIP,
            "constant_dimensions": int(constant.sum()),
            "floored_dimensions": int((~constant & (std < self.STATE_FLOOR_STD)).sum()),
        }

    def state_features(self, values, mask):
        """Learned embedding of train-normalized, mask-zeroed robot state ``[...,S]``."""
        if not bool(self.state_normalization_fitted):
            raise ContractError("fit training-only state normalization before state fusion")
        values = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
        mask = torch.from_numpy(np.array(mask, dtype=np.bool_, copy=True))
        values, mask = values.to(self.device_name), mask.to(self.device_name)
        normalized = (values - self.state_mean) / self.state_scale
        return self.state_encoder(normalized.clamp(-self.STATE_CLIP, self.STATE_CLIP) * mask)

    def observe_sequence(self, batch, *, target=False):
        """Online (or target) latents ``[B,T+1,D]`` of a canonical sequence batch."""
        features, prefix = self.image_features(batch.observations, sequence=True, target=target)
        latents = features.reshape(*prefix, -1)
        if self.config["state_fusion"]:
            latents = latents + self.state_features(batch.robot_states, batch.state_mask)
        return latents

    def readout_frame_weight(self, targets):
        """Per-frame ``[B,T+1,1]`` regression weight, or None when the shaping is off.

        TASK-050 measured the readout failure as concentrated in the minority of frames
        where the true palm-apple offset has moved away from the window start. Those
        frames get ``1 + readout_moving_weight``; still frames keep weight 1. The weight
        is derived from the privileged targets, so it is disclosed as a label-derived
        training weight: it shapes the loss, never a model input.
        """
        extra = self.config["readout_moving_weight"]
        if extra <= 0:
            return None
        offset = targets["palm_minus_apple"]
        displacement = (offset - offset[:, :1]).square().sum(-1, keepdim=True).sqrt()
        moving = (displacement >= self.config["readout_moving_threshold_m"]).float()
        return 1.0 + extra * moving

    def readout_loss(self, encoded, predicted, readout_targets):
        """Shared readout loss on encoded ``[B,T+1,D]`` and predicted ``[B,T,D]`` latents.

        Targets are training-time labels only: ``{name: float32 [B,T+1,width]}``.
        """
        if not self.config["readout_heads"]:
            if readout_targets is not None:
                raise ContractError("readout targets require readout_heads")
            return encoded.new_zeros(()), {}
        if readout_targets is None:
            raise ContractError("a readout-head model trains only with readout targets")
        targets = {}
        for name in readout_module.READOUT_NAMES:
            value = readout_targets.get(name)
            if not isinstance(value, np.ndarray) or value.shape[:2] != encoded.shape[:2]:
                raise ContractError(f"readout target {name} must be an array [B,T+1,width]")
            targets[name] = torch.from_numpy(np.array(value, np.float32, copy=True)).to(
                self.device_name
            )
        weight = self.readout_frame_weight(targets)
        auxiliary = self.config["readout_auxiliary_weight"]
        encoded_loss, _ = self.readout_head.loss(
            encoded, targets, frame_weight=weight, auxiliary_weight=auxiliary
        )
        predicted_loss, predicted_terms = self.readout_head.loss(
            predicted,
            {name: value[:, 1:] for name, value in targets.items()},
            frame_weight=None if weight is None else weight[:, 1:],
            auxiliary_weight=auxiliary,
        )
        metrics = {
            "readout_encoded_loss": encoded_loss.item(),
            "readout_predicted_loss": predicted_loss.item(),
        } | {f"readout_predicted_{name}": term.item() for name, term in predicted_terms.items()}
        return self.config["readout_weight"] * (encoded_loss + predicted_loss), metrics

    @torch.no_grad()
    def latent_statistics(self, latents):
        """Model-owned collapse diagnostics over encoded [B,D] latents (evaluators never
        read latent values themselves)."""
        latents = [latents] if isinstance(latents, VisualLatent) else list(latents)
        if not latents:
            raise ContractError("latent statistics need at least one encoded latent")
        # Transfer before casting (see diagnostics): MPS->CPU float64 fusion is unsafe.
        return _statistics(torch.cat([self.check_latent(z, 2).cpu() for z in latents]))

    @torch.no_grad()
    def image_embedding_statistics(self, images):
        """The same diagnostics for the image pathway alone (before any state fusion),
        over a list of camera mappings."""
        self.eval()
        values = []
        for observation in images:
            features, _ = self.image_features(observation)
            values.append(features.cpu())
        if not values:
            raise ContractError("image statistics need at least one observation")
        return _statistics(torch.cat(values))

    @torch.no_grad()
    def readout(self, z):
        """Declared physical readouts of an encoded [B,D] or predicted [B,K,T,D] latent."""
        if not self.config["readout_heads"]:
            raise ContractError("this model declares no readouts")
        if not isinstance(z, VisualLatent) or z.values.ndim not in (2, 4):
            raise ContractError("readout requires an encoded or predicted latent")
        values = self.check_latent(z, z.values.ndim)
        self.eval()
        result = {}
        declared = self.declared_readouts
        for name, value in self.readout_head.physical(values).items():
            if name not in declared:
                continue  # an untrained auxiliary head is never exposed
            array = value.float().cpu().numpy().astype(np.float32, copy=True)
            if not np.isfinite(array).all():
                raise ContractError(f"model produced a non-finite readout {name}")
            array.setflags(write=False)
            result[name] = array
        return result

    @torch.no_grad()
    def predict(self, z, actions):
        self.eval()
        validate_actions(actions)
        initial = self.check_latent(z, 2)
        batch, candidates, horizon, dimension = actions.shape
        if batch != initial.shape[0] or horizon > self.config["max_horizon"]:
            raise ContractError("action batch/horizon is incompatible with encoded observation")
        chunk = self.config["candidate_chunk_size"]
        outputs = []
        # Chunk candidates, retaining only the compact resulting latent trajectories.
        for start in range(0, candidates, chunk):
            part = torch.from_numpy(np.array(actions[:, start : start + chunk], copy=True))
            part = part.to(self.device_name)
            count = part.shape[1]
            state = initial[:, None].expand(-1, count, -1).reshape(batch * count, -1)
            action = part.reshape(batch * count, horizon, dimension)
            outputs.append(self.rollout(state, action).reshape(batch, count, horizon, -1))
        result = torch.cat(outputs, dim=1)
        if not torch.isfinite(result).all():
            raise ContractError("model produced non-finite predicted latents")
        return VisualLatent(result, self._owner)

    def rollout(self, initial, actions):
        state = initial
        result = []
        for step in range(actions.shape[1]):
            state = self.next_embedding(state, actions[:, step])
            result.append(state)
        return torch.stack(result, dim=1)

    @torch.no_grad()
    def distance(self, predicted_z, goal_z):
        predicted = self.check_latent(predicted_z, 4)
        goal = self.check_latent(goal_z, 2)
        if goal.shape[0] != predicted.shape[0]:
            raise ContractError("goal/prediction batch dimensions disagree")
        costs = (
            (predicted - goal[:, None, None]) ** 2 / self.latent_variance.clamp_min(0.01)
        ).mean(-1)
        if not torch.isfinite(costs).all():
            raise ContractError("model produced non-finite goal distances")
        return costs.cpu().numpy().astype(np.float32, copy=True)

    def goal_embed(self, pixels):
        return self.embed(pixels)

    def sequence_tensors(self, batch):
        """Validated batch actions on the model device (images are read per camera)."""
        self.check_batch(batch)
        return torch.from_numpy(np.array(batch.actions, copy=True)).to(self.device_name)

    @torch.no_grad()
    def validation_error(self, batch):
        """Within-backend recursive latent MSE, not a cross-backend score."""
        return self.diagnostics(batch)["prediction_mse"]

    @torch.no_grad()
    def diagnostics(self, batch):
        self.eval()
        actions = self.sequence_tensors(batch)
        embeddings = self.observe_sequence(batch)
        target_embeddings = self.observe_sequence(batch, target=True)
        targets = target_embeddings[:, 1:]
        prediction = self.rollout(embeddings[:, 0], actions)
        zero_prediction = self.rollout(embeddings[:, 0], torch.zeros_like(actions))
        # Fixed permutation over B*T actions; report when it cannot change anything.
        flat = actions.reshape(-1, actions.shape[-1])
        shuffled = flat.roll(1, dims=0).reshape_as(actions)
        shuffled_prediction = self.rollout(embeddings[:, 0], shuffled)
        samples = embeddings.reshape(-1, embeddings.shape[-1])
        std = samples.std(0, unbiased=False)
        target_samples = target_embeddings.reshape(-1, target_embeddings.shape[-1])
        target_std = target_samples.std(0, unbiased=False)
        rank_available = target_samples.numel() <= 1_000_000 and min(target_samples.shape) <= 512
        effective_rank = 0.0
        if rank_available:
            # Transfer before casting: fused MPS->CPU float64 conversion can corrupt
            # values on the validated PyTorch/macOS combination.
            centered = target_samples.cpu().double()
            centered = centered - centered.mean(0)
            energy = torch.linalg.svdvals(centered).square()
            total_energy = energy.sum().item()
            if total_energy > 0:
                probabilities = energy[energy > 0] / total_energy
                effective_rank = (-(probabilities * probabilities.log()).sum()).exp().item()
        mse = (prediction - targets).square().mean()
        zero_mse = (zero_prediction - targets).square().mean()
        shuffled_mse = (shuffled_prediction - targets).square().mean()
        persistence_mse = (target_embeddings[:, :1] - targets).square().mean()
        metrics = {
            "metric_definition_version": 2.0,
            "prediction_mse": mse.item(),
            "zero_action_mse": zero_mse.item(),
            "shuffled_action_mse": shuffled_mse.item(),
            "persistence_mse": persistence_mse.item(),
            "action_effect_rms": (prediction - zero_prediction).square().mean().sqrt().item(),
            "shuffled_actions_changed": float(not torch.equal(actions, shuffled)),
            "latent_std_mean": std.mean().item(),
            "latent_std_min": std.min().item(),
            "collapsed_fraction": (std < 0.01).float().mean().item(),
            "online_latent_std_mean": std.mean().item(),
            "online_latent_std_min": std.min().item(),
            "online_collapsed_fraction": (std < 0.01).float().mean().item(),
            "target_latent_std_mean": target_std.mean().item(),
            "target_latent_std_min": target_std.min().item(),
            "target_collapsed_fraction": (target_std < 0.01).float().mean().item(),
            "target_effective_rank": effective_rank,
            "target_effective_rank_available": float(rank_available),
        }
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ContractError("non-finite model diagnostics")
        return metrics

    def optimize(self, loss, embeddings):
        if not torch.isfinite(loss):
            raise ContractError("non-finite training loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = [p for p in self.parameters() if p.grad is not None]
        if not parameters or not all(torch.isfinite(p.grad).all() for p in parameters):
            raise ContractError("missing/non-finite model gradients")
        grad = torch.nn.utils.clip_grad_norm_(
            parameters, self.config["gradient_clip"], error_if_nonfinite=True
        )
        self.optimizer.step()
        with torch.no_grad():
            variance = embeddings.detach().reshape(-1, embeddings.shape[-1]).var(0, unbiased=False)
            self.latent_variance.lerp_(variance, 0.05)
        self.updates += 1
        self._owner = object()
        return grad.item()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "format_version": 1,
            "backend": self.backend,
            "config": self.config,
            "state_schema": asdict(self.state_schema),
            "action_schema": asdict(EE_DELTA_GRASP_V0),
            "metadata": self.metadata,
            "seed": self.seed,
            "updates": self.updates,
            "weights": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng_cpu": self._rng,
            "rng_mps": self._mps_rng,
            "source_revision": self.source_revision,
            "implementation_sha256": self.implementation_sha256,
            "preprocessing": {
                "pixels": "uint8_rgb",
                "actions": "normalized_-1_1",
                "robot_state": "train_normalized_learned_state_fusion"
                if self.config["state_fusion"]
                else "ignored_visual_baseline",
                "readouts": readout_module.READOUT_VERSION
                if self.config["readout_heads"]
                else None,
            },
        }
        temporary = path.with_name(path.name + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        expected = {
            "format_version": 1,
            "backend": self.backend,
            "config": self.config,
            "state_schema": asdict(self.state_schema),
            "action_schema": asdict(EE_DELTA_GRASP_V0),
            "source_revision": self.source_revision,
            "implementation_sha256": self.implementation_sha256,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ContractError(f"checkpoint {key} is incompatible")
        for key, value in self.metadata.items():
            if checkpoint.get("metadata", {}).get(key) != value:
                raise ContractError(f"checkpoint provenance {key!r} is incompatible")
        self.load_state_dict(checkpoint["weights"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.metadata = checkpoint["metadata"]
        self.seed = checkpoint["seed"]
        self.updates = checkpoint["updates"]
        self._rng = checkpoint["rng_cpu"]
        self._mps_rng = checkpoint["rng_mps"]
        self._owner = object()  # Old latents are invalid after checkpoint replacement.
        self.eval()

    @property
    def implementation_sha256(self):
        digest = hashlib.sha256()
        for path in (
            Path(__file__),
            Path(inspect.getfile(type(self))),
            Path(readout_module.__file__),
            Path(readout_module.__file__).parents[1] / "readout_labels.py",
        ):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    @property
    def source_revision(self):
        # Identifies native implementation even in a source checkout before commit.
        return hashlib.sha256(Path(__file__).with_name("native.py").read_bytes()).hexdigest()
