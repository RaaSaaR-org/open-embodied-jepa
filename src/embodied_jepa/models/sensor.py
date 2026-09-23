"""Task-specific learned visual/proprioceptive dynamics, not an unsupervised JEPA.

The fixed RGB grid makes visual error interpretable and cannot collapse during
training. Every predicted future proprioceptive value comes from the learned
transition. Goals contain images only. No simulator or robot kinematics is used.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from embodied_jepa.contracts import EE_DELTA_GRASP_V0, ContractError, RobotState
from embodied_jepa.models.base import VisualLatent, VisualModel


class SensorWorldModel(VisualModel):
    backend = "sensor_wm"
    defaults = VisualModel.defaults | {
        "image_size": 24,
        "hidden_dim": 256,
        "proprio_weight": 1.0,
        "multistep_weight": 1.0,
        "feature_std_floor": 0.03,
        "state_std_floor": 0.01,
        "delta_std_floor": 0.01,
    }
    preprocessing = {
        "version": "sensor_grid_v1",
        "pixels": "uint8_rgb_bilinear_antialiased_spatial_grid",
        "actions": "normalized_-1_1_executed",
        "robot_state": "train_normalized_observed_state_then_learned_recursive_prediction",
        "goal": "image_only_no_goal_robot_state",
        "history": "current_observation_only_no_hidden_episode_state",
    }

    def __init__(self, state_schema, device="cpu", seed=0, config=None, metadata=None):
        config = dict(config or {})
        size = config.get("image_size", self.defaults["image_size"])
        if type(size) is not int or size < 1:
            raise ContractError("image_size must be a positive integer")
        self.visual_dimension = 3 * size * size
        dimension = self.visual_dimension + state_schema.dimension
        if config.get("latent_dim", dimension) != dimension:
            raise ContractError("latent_dim must equal RGB grid plus robot state dimensions")
        config["latent_dim"] = dimension
        super().__init__(state_schema, device, seed, config, metadata)
        if self.config["cameras"] is not None:
            # This backend flattens one camera's pixel grid straight into its latent and
            # never routes through the shared camera fusion, so it must refuse the key
            # rather than silently train on the first camera of a multi-camera config.
            raise ContractError("sensor_wm reads a single 'camera'; it has no camera fusion")
        for name in ("proprio_weight", "feature_std_floor", "state_std_floor", "delta_std_floor"):
            if not np.isfinite(self.config[name]) or self.config[name] <= 0:
                raise ContractError(f"{name} must be positive and finite")
        if not np.isfinite(self.config["multistep_weight"]) or self.config["multistep_weight"] < 0:
            raise ContractError("multistep_weight must be nonnegative and finite")
        self.register_buffer("sensor_mean", torch.zeros(dimension))
        self.register_buffer("sensor_scale", torch.ones(dimension))
        self.register_buffer("delta_scale", torch.ones(dimension))
        self.register_buffer("normalization_fitted", torch.tensor(False))
        with self.rng_scope():
            hidden = self.config["hidden_dim"]
            self.predictor = nn.Sequential(
                nn.Linear(dimension + 14, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, dimension),
            )
            # Begin near persistence, but retain nonzero action derivatives.
            nn.init.normal_(self.predictor[-1].weight, std=0.001)
            nn.init.zeros_(self.predictor[-1].bias)
        self.finish_init()

    def _require_fitted(self):
        if not bool(self.normalization_fitted):
            raise ContractError("fit training-only normalization before using sensor_wm")

    def _raw_sequence(self, batch):
        self.check_batch(batch)
        if not batch.state_mask.all():
            raise ContractError("sensor_wm requires every proprioceptive field to be observed")
        pixels, prefix = self.pixels(batch.observations, sequence=True)
        visual = pixels.flatten(1).reshape(*prefix, self.visual_dimension)
        state = torch.from_numpy(np.array(batch.robot_states, copy=True)).to(self.device_name)
        actions = torch.from_numpy(np.array(batch.actions, copy=True)).to(self.device_name)
        return torch.cat((visual, state), -1), actions

    @torch.no_grad()
    def fit_normalization(self, batches, *, training_episode_ids):
        """Freeze moments from explicitly authorized training episodes once.

        Supply nonoverlapping windows where practical. Repeated boundary frames
        contribute with their supplied frequency; this is recorded as frame_count.
        The caller owns the sealed split; listing validation IDs as training is
        never authorized merely because this method accepts an ID list.
        """
        if bool(self.normalization_fitted) or self.updates:
            raise ContractError("normalization is frozen; create a new model to refit")
        ids = tuple(training_episode_ids)
        if (
            not ids
            or any(not isinstance(x, str) or not x for x in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ContractError("training_episode_ids must be unique nonempty episode names")
        allowed, observed = set(ids), set()
        total = square = delta_total = delta_square = None
        count, transitions = 0, 0
        for batch in batches:
            self.check_batch(batch)
            if not set(batch.episode_ids) <= allowed:
                raise ContractError("normalization received an episode outside the training split")
            observed.update(batch.episode_ids)
            raw, _ = self._raw_sequence(batch)
            values = raw.cpu().numpy().astype(np.float64)
            delta = np.diff(values, axis=1).reshape(-1, values.shape[-1])
            values = values.reshape(-1, values.shape[-1])
            if total is None:
                total = np.zeros(values.shape[-1])
                square, delta_total, delta_square = total.copy(), total.copy(), total.copy()
            total += values.sum(0)
            square += np.square(values).sum(0)
            delta_total += delta.sum(0)
            delta_square += np.square(delta).sum(0)
            count += len(values)
            transitions += len(delta)
        if observed != allowed or count < 2 or transitions < 1:
            raise ContractError("normalization must cover every declared training episode")
        mean = total / count
        floors = np.full(len(mean), self.config["state_std_floor"])
        floors[: self.visual_dimension] = self.config["feature_std_floor"]
        scale = np.maximum(np.sqrt(np.maximum(square / count - mean**2, 0)), floors)
        delta_std = np.sqrt(
            np.maximum(delta_square / transitions - (delta_total / transitions) ** 2, 0)
        )
        delta_scale = np.maximum(delta_std / scale, self.config["delta_std_floor"])
        for name, value in (
            ("sensor_mean", mean),
            ("sensor_scale", scale),
            ("delta_scale", delta_scale),
        ):
            getattr(self, name).copy_(
                torch.as_tensor(value, dtype=torch.float32, device=self.device_name)
            )
        self.normalization_fitted.fill_(True)
        self.metadata["normalization"] = {
            "method": "training_population_moments_sensor_grid_v1",
            "episode_ids": sorted(allowed),
            "frame_count": count,
            "transition_count": transitions,
        }
        self._owner = object()
        return dict(self.metadata["normalization"])

    @torch.no_grad()
    def encode(self, observation, robot_state: RobotState):
        self._require_fitted()
        if not isinstance(robot_state, RobotState) or robot_state.schema != self.state_schema:
            raise ContractError("incompatible robot-state schema")
        if not robot_state.mask.all():
            raise ContractError("sensor_wm requires every proprioceptive field to be observed")
        pixels, prefix = self.pixels(observation)
        if prefix[0] != robot_state.values.shape[0]:
            raise ContractError("image/state batch dimensions disagree")
        proprio = torch.from_numpy(np.array(robot_state.values, copy=True)).to(self.device_name)
        raw = torch.cat((pixels.flatten(1), proprio), -1)
        return VisualLatent((raw - self.sensor_mean) / self.sensor_scale, self._owner)

    @torch.no_grad()
    def encode_goal(self, goal):
        self._require_fitted()
        pixels, _ = self.pixels(goal)
        n = self.visual_dimension
        return VisualLatent(
            (pixels.flatten(1) - self.sensor_mean[:n]) / self.sensor_scale[:n], self._owner
        )

    @torch.no_grad()
    def observed_distance(self, observation_images, goal_images):
        """Measured image progress for waypoint advancement; no predicted hold action."""
        current, goal = self.encode_goal(observation_images), self.encode_goal(goal_images)
        if current.values.shape != goal.values.shape:
            raise ContractError("observed image/goal batch dimensions disagree")
        cost = (current.values - goal.values).square().mean(-1)
        if not torch.isfinite(cost).all():
            raise ContractError("non-finite observed image-goal distance")
        return cost.cpu().numpy().astype(np.float32, copy=True)

    def next_embedding(self, state, actions):
        if state.shape[-1] != self.config["latent_dim"]:
            raise ContractError(
                "prediction requires visual and proprioceptive state, not an image goal"
            )
        return state + self.predictor(torch.cat((state, actions), -1)) * self.delta_scale

    @torch.no_grad()
    def distance(self, predicted_z, goal_z):
        self._require_fitted()
        predicted, goal = self.check_latent(predicted_z, 4), self.check_latent(goal_z, 2)
        if (
            predicted.shape[-1] != self.config["latent_dim"]
            or goal.shape[-1] != self.visual_dimension
        ):
            raise ContractError("distance requires predicted sensor state and an image-only goal")
        if predicted.shape[0] != goal.shape[0]:
            raise ContractError("goal/prediction batch dimensions disagree")
        cost = (predicted[..., : self.visual_dimension] - goal[:, None, None]).square().mean(-1)
        if not torch.isfinite(cost).all():
            raise ContractError("non-finite image-goal cost")
        return cost.cpu().numpy().astype(np.float32, copy=True)

    def _sequence(self, batch):
        self._require_fitted()
        raw, actions = self._raw_sequence(batch)
        return (raw - self.sensor_mean) / self.sensor_scale, actions

    def _weighted_error(self, error):
        n = self.visual_dimension
        return (
            error[..., :n].square().mean()
            + self.config["proprio_weight"] * error[..., n:].square().mean()
        )

    def train_step(self, batch):
        self.train()
        state, actions = self._sequence(batch)
        one = self.next_embedding(state[:, :-1], actions)
        recursive = self.rollout(state[:, 0], actions)
        one_loss = self._weighted_error((one - state[:, 1:]) / self.delta_scale)
        # Normalize multistep residual errors by sqrt(horizon), retaining long-horizon pressure.
        time_scale = torch.arange(1, actions.shape[1] + 1, device=self.device_name).sqrt()[
            None, :, None
        ]
        multi_loss = self._weighted_error(
            (recursive - state[:, 1:]) / (self.delta_scale * time_scale)
        )
        loss = one_loss + self.config["multistep_weight"] * multi_loss
        grad = self.optimize(loss, state)
        self.eval()
        return {
            "loss": loss.item(),
            "prediction_loss": one_loss.item(),
            "multistep_loss": multi_loss.item(),
            "gradient_norm": grad,
            "updates": float(self.updates),
        }

    @torch.no_grad()
    def diagnostics(self, batch):
        self.eval()
        states, actions = self._sequence(batch)
        target = states[:, 1:]
        predicted = self.rollout(states[:, 0], actions)
        zero = self.rollout(states[:, 0], torch.zeros_like(actions))
        shuffled_actions = actions.reshape(-1, 14).roll(1, 0).reshape_as(actions)
        shuffled = self.rollout(states[:, 0], shuffled_actions)
        n = self.visual_dimension

        def mse(value):
            return (value[..., :n] - target[..., :n]).square().mean().item()

        result = {
            "metric_definition_version": 3.0,
            "prediction_mse": mse(predicted),
            "proprio_prediction_mse": (predicted[..., n:] - target[..., n:]).square().mean().item(),
            "persistence_mse": mse(states[:, :1]),
            "zero_action_mse": mse(zero),
            "shuffled_action_mse": mse(shuffled),
            "action_effect_rms": (predicted[..., :n] - zero[..., :n]).square().mean().sqrt().item(),
            "shuffled_actions_changed": float(not torch.equal(actions, shuffled_actions)),
        }
        if not all(np.isfinite(v) for v in result.values()):
            raise ContractError("non-finite sensor diagnostics")
        return result

    def save(self, path):
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
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
            "preprocessing": self.preprocessing,
        }
        temporary = path.with_name(path.name + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("preprocessing") != self.preprocessing:
            raise ContractError("checkpoint sensor preprocessing is incompatible")
        normalization = checkpoint.get("metadata", {}).get("normalization", {})
        if normalization.get(
            "method"
        ) != "training_population_moments_sensor_grid_v1" or not normalization.get("episode_ids"):
            raise ContractError("checkpoint lacks training normalization provenance")
        weights = checkpoint.get("weights", {})
        if not bool(weights.get("normalization_fitted", False)):
            raise ContractError("checkpoint normalization is not fitted")
        if any(not torch.isfinite(value).all() for value in weights.values()):
            raise ContractError("checkpoint contains non-finite tensors")
        for key in ("sensor_scale", "delta_scale"):
            if key not in weights or not (weights[key] > 0).all():
                raise ContractError("checkpoint contains invalid normalization scales")
        super().load(path)

    @property
    def source_revision(self):
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
