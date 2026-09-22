"""Frozen demonstration-state goal cost over an unchanged ``sensor_wm`` checkpoint.

The wrapped sensor model's learned recursive predictions are not modified. Only
the model-owned goal distance changes: planning scores the *predicted* right-arm
and right-hand joint positions against a TRAIN demonstration state goal, in the
child's frozen TRAIN-only normalization. A visual endpoint error against an
optional goal image is computed for diagnostics only; its weight is fixed at 0.
No simulator truth, object pose or task score enters this model.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from embodied_jepa.contracts import (
    EE_DELTA_GRASP_V0,
    Capabilities,
    ContractError,
    RobotState,
    StateSchema,
)
from embodied_jepa.models.sensor import SensorWorldModel

ARM_JOINTS = (
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)
HAND_JOINTS = (
    "right_hand_thumb_0",
    "right_hand_thumb_1",
    "right_hand_thumb_2",
    "right_hand_index_0",
    "right_hand_index_1",
    "right_hand_middle_0",
    "right_hand_middle_1",
)
FIELDS = tuple(f"{name}_joint.position" for name in ARM_JOINTS + HAND_JOINTS)
VISUAL_WEIGHT = 0.0
NORMALIZATION_METHOD = "training_population_moments_sensor_grid_v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def field_indices(schema):
    """Resolve the fourteen named right-arm/hand positions; never assume positions."""
    if any(schema.names.count(name) != 1 for name in FIELDS):
        raise ContractError("state goal requires fourteen unique named right-arm/hand positions")
    indices = tuple(schema.names.index(name) for name in FIELDS)
    if any(schema.units[i] != "rad" for i in indices):
        raise ContractError("state-goal joint positions must have radian units")
    return indices


@dataclass(frozen=True)
class StateGoalLatent:
    child: object
    owner: object


@dataclass(frozen=True)
class StateGoal:
    values: torch.Tensor  # normalized goal positions [B,14]
    visual: object | None
    owner: object


class StateGoalSensorWorldModel:
    """Frozen composition: unchanged sensor dynamics, demonstration-state goal cost."""

    backend = "state_goal_sensor_wm_v1"

    def __init__(self, state_schema, device="cpu", seed=0, config=None, metadata=None):
        if not isinstance(state_schema, StateSchema):
            raise ContractError("named state schema required")
        if device != "cpu":
            raise ContractError("state-goal composition is validated on CPU only")
        if type(seed) is not int or seed < 0:
            raise ContractError("nonnegative integer seed required")
        self.indices = field_indices(state_schema)
        self.state_schema = state_schema
        self.device_name = device
        self.seed = seed
        config = copy.deepcopy(config or {})
        if set(config) - {"sensor_config", "visual_weight", "version"}:
            raise ContractError("unknown state-goal config")
        self.config = {"sensor_config": {}, "visual_weight": VISUAL_WEIGHT, "version": 1} | config
        if (
            self.config["visual_weight"] != VISUAL_WEIGHT
            or self.config["version"] != 1
            or not isinstance(self.config["sensor_config"], dict)
        ):
            raise ContractError("state-goal version1 fixes the visual diagnostic weight at 0")
        self.metadata = copy.deepcopy(metadata or {})
        json.dumps(self.metadata, allow_nan=False)
        self._owner = object()
        self._sensor = None
        self._checkpoint = None
        self._diagnostics = []

    @property
    def implementation_sha256(self):
        return sha256(__file__)

    @property
    def capabilities(self):
        return Capabilities(
            EE_DELTA_GRASP_V0,
            self.state_schema,
            self.config["sensor_config"].get(
                "max_horizon", SensorWorldModel.defaults["max_horizon"]
            ),
            supported_devices=("cpu",),
        )

    def _ready(self):
        if self._sensor is None:
            raise ContractError("load a verified sensor checkpoint before inference")

    def _state(self, value, rank):
        self._ready()
        if not isinstance(value, StateGoalLatent) or value.owner is not self._owner:
            raise ContractError("latent belongs to another state-goal instance or checkpoint")
        return self._sensor.check_latent(value.child, rank)

    def _goal(self, value):
        self._ready()
        if not isinstance(value, StateGoal) or value.owner is not self._owner:
            raise ContractError("goal belongs to another state-goal instance or checkpoint")
        if (
            value.values.ndim != 2
            or value.values.shape[1] != len(FIELDS)
            or not torch.isfinite(value.values).all()
        ):
            raise ContractError("invalid normalized state goal")
        return value.values

    def _normalize(self, raw):
        offset = self._sensor.visual_dimension
        index = torch.as_tensor(self.indices, device=self.device_name) + offset
        return (raw - self._sensor.sensor_mean[index]) / self._sensor.sensor_scale[index]

    def _goal_array(self, values):
        array = np.asarray(values)
        if (
            array.dtype.kind != "f"
            or array.ndim != 2
            or array.shape[1] != len(FIELDS)
            or not len(array)
            or not np.isfinite(array).all()
        ):
            raise ContractError("state goal must be finite float[B,14] right-arm/hand radians")
        return torch.from_numpy(np.array(array, dtype=np.float32, copy=True)).to(self.device_name)

    @staticmethod
    def select(state_values, schema):
        """Extract the fourteen goal fields, in FIELDS order, from full state rows."""
        values = np.asarray(state_values)
        indices = field_indices(schema)
        if values.ndim != 2 or values.shape[1] != schema.dimension:
            raise ContractError("state rows must be [B,state_dimension]")
        return np.array(values[:, indices], dtype=np.float32, copy=True)

    @torch.no_grad()
    def encode(self, observation, robot_state):
        self._ready()
        return StateGoalLatent(self._sensor.encode(observation, robot_state), self._owner)

    @torch.no_grad()
    def encode_goal(self, goal):
        """Goal mapping: required ``state`` float[B,14]; optional diagnostic ``images``."""
        self._ready()
        if not isinstance(goal, Mapping) or "state" not in goal or set(goal) - {"state", "images"}:
            raise ContractError("state goal requires a mapping with state and optional images")
        values = self._normalize(self._goal_array(goal["state"]))
        visual = None
        if goal.get("images") is not None:
            visual = self._sensor.encode_goal(goal["images"])
            if visual.values.shape[0] != values.shape[0]:
                raise ContractError("diagnostic goal image batch differs from state goal")
        return StateGoal(values, visual, self._owner)

    @torch.no_grad()
    def predict(self, z, actions):
        self._state(z, 2)
        return StateGoalLatent(self._sensor.predict(z.child, actions), self._owner)

    @torch.no_grad()
    def distance(self, predicted_z, goal_z):
        values = self._state(predicted_z, 4)
        goal = self._goal(goal_z)
        if values.shape[0] != goal.shape[0]:
            raise ContractError("state goal/prediction batch dimensions disagree")
        index = torch.as_tensor(self.indices, device=self.device_name)
        predicted = values[..., index + self._sensor.visual_dimension]
        cost = (predicted - goal[:, None, None]).square().mean(-1)
        if not torch.isfinite(cost).all():
            raise ContractError("non-finite state-goal cost")
        if goal_z.visual is not None:
            visual = self._sensor.distance(predicted_z.child, goal_z.visual)
            # Diagnostic only: weight fixed at zero, never added to the returned cost.
            self._diagnostics.append(
                {
                    "visual_weight": VISUAL_WEIGHT,
                    "visual_endpoint_costs": visual[..., -1].reshape(-1).tolist(),
                }
            )
        return cost.cpu().numpy().astype(np.float32, copy=True)

    def pop_diagnostics(self):
        result, self._diagnostics = self._diagnostics, []
        return result

    @torch.no_grad()
    def observed_distance(self, robot_state, goal_state):
        """Measured proprioceptive progress; same normalized metric as planning cost."""
        self._ready()
        if not isinstance(robot_state, RobotState) or robot_state.schema != self.state_schema:
            raise ContractError("observed progress requires a compatible RobotState")
        if not robot_state.mask[:, list(self.indices)].all():
            raise ContractError("observed progress requires every goal field to be observed")
        current = self._normalize(self._goal_array(self.select(robot_state.values, self.schema)))
        goal = self._normalize(self._goal_array(goal_state))
        if current.shape != goal.shape:
            raise ContractError("observed state/goal batch dimensions disagree")
        cost = (current - goal).square().mean(-1)
        if not torch.isfinite(cost).all():
            raise ContractError("non-finite observed state-goal distance")
        return cost.cpu().numpy().astype(np.float32, copy=True)

    @property
    def schema(self):
        return self.state_schema

    @torch.no_grad()
    def image_distance(self, current_images, goal_images):
        """Child image metric, used only for initial-RGB demonstration retrieval."""
        self._ready()
        return self._sensor.observed_distance(current_images, goal_images)

    def train_step(self, batch):
        raise ContractError(
            "state_goal_sensor_wm_v1 is a frozen composition; training is forbidden"
        )

    def load(self, path):
        path = Path(path)
        envelope = torch.load(path, map_location="cpu", weights_only=True)
        if envelope.get("backend") != SensorWorldModel.backend:
            raise ContractError("state-goal composition wraps a sensor_wm checkpoint only")
        if envelope.get("config") != self.config["sensor_config"]:
            raise ContractError("state-goal sensor_config differs from the checkpoint")
        if envelope.get("state_schema") != asdict(self.state_schema):
            raise ContractError("checkpoint state schema differs")
        if envelope.get("action_schema") != asdict(EE_DELTA_GRASP_V0):
            raise ContractError("checkpoint action schema differs")
        normalization = envelope.get("metadata", {}).get("normalization", {})
        if normalization.get("method") != NORMALIZATION_METHOD or not normalization.get(
            "episode_ids"
        ):
            raise ContractError("checkpoint lacks TRAIN-only normalization provenance")
        for key, value in self.metadata.items():
            if envelope.get("metadata", {}).get(key) != value:
                raise ContractError(f"state-goal provenance mismatch: {key}")
        sensor = SensorWorldModel(
            self.state_schema,
            device="cpu",
            seed=envelope["seed"],
            config=self.config["sensor_config"],
            metadata=copy.deepcopy(self.metadata),
        )
        sensor.load(path)  # Existing strict loader: implementation hash, preprocessing, scales.
        if not bool(sensor.normalization_fitted):
            raise ContractError("state-goal composition requires fitted normalization")
        sensor.eval()
        sensor.requires_grad_(False)
        self._sensor = sensor
        self._checkpoint = path
        self.seed = envelope["seed"]
        self.metadata = copy.deepcopy(sensor.metadata)
        self._owner = object()
        self._diagnostics = []
        return self

    def save(self, path):
        """Copy the unchanged child checkpoint; the composition adds no weights."""
        self._ready()
        path = Path(path)
        if path.exists():
            raise FileExistsError("refusing to overwrite state-goal checkpoint")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        shutil.copyfile(self._checkpoint, temporary)
        temporary.replace(path)
        return path
