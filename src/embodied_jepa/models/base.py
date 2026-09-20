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


@dataclass(frozen=True)
class VisualLatent:
    """Private backend-owned carrier; the planner must never inspect this payload."""

    values: torch.Tensor
    owner: object


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
    }

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
        ):
            if type(self.config[name]) is not int or self.config[name] < 1:
                raise ContractError(f"{name} must be a positive integer")
        for name in ("learning_rate", "gradient_clip"):
            if not np.isfinite(self.config[name]) or self.config[name] <= 0:
                raise ContractError(f"{name} must be positive and finite")
        if not np.isfinite(self.config["weight_decay"]) or self.config["weight_decay"] < 0:
            raise ContractError("weight_decay must be nonnegative and finite")
        if not isinstance(self.config["camera"], str) or not self.config["camera"]:
            raise ContractError("camera must be a nonempty name")
        self.metadata = copy.deepcopy(dict(metadata or {}))
        # Only JSON-safe provenance is accepted; no executable checkpoint objects.
        json.dumps(self.metadata, allow_nan=False)
        self.updates = 0
        self._owner = object()
        self._rng = torch.Generator().manual_seed(seed).get_state()
        self._mps_rng = None
        self.register_buffer("latent_variance", torch.ones(self.config["latent_dim"]))

    @property
    def capabilities(self):
        return Capabilities(
            EE_DELTA_GRASP_V0,
            self.state_schema,
            self.config["max_horizon"],
            supported_devices=("cpu", "mps"),
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
        self.to(self.device_name)
        self.optimizer = torch.optim.AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        self.eval()

    def pixels(self, images, *, sequence=False):
        name = self.config["camera"]
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
        pixels, prefix = self.pixels(observation)
        if prefix[0] != robot_state.values.shape[0]:
            raise ContractError("image/state batch dimensions disagree")
        return VisualLatent(self.embed(pixels), self._owner)

    @torch.no_grad()
    def encode_goal(self, goal):
        self.eval()
        pixels, _ = self.pixels(goal)
        return VisualLatent(self.goal_embed(pixels), self._owner)

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
        self.check_batch(batch)
        pixels, prefix = self.pixels(batch.observations, sequence=True)
        actions = torch.from_numpy(np.array(batch.actions, copy=True)).to(self.device_name)
        return pixels, prefix, actions

    @torch.no_grad()
    def validation_error(self, batch):
        """Within-backend recursive latent MSE, not a cross-backend score."""
        return self.diagnostics(batch)["prediction_mse"]

    @torch.no_grad()
    def diagnostics(self, batch):
        self.eval()
        pixels, prefix, actions = self.sequence_tensors(batch)
        embeddings = self.embed(pixels).reshape(*prefix, -1)
        target_embeddings = self.goal_embed(pixels).reshape(*prefix, -1)
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
                "robot_state": "ignored_visual_baseline",
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
        for path in (Path(__file__), Path(inspect.getfile(type(self)))):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    @property
    def source_revision(self):
        # Identifies native implementation even in a source checkout before commit.
        return hashlib.sha256(Path(__file__).with_name("native.py").read_bytes()).hexdigest()
