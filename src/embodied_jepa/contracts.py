"""Versioned, NumPy-only boundaries between models, data, planners and robots.

Arrays are validated without coercion: loaders/adapters must make unit, dtype,
and normalization conversions explicit. Value objects take read-only copies so
sensor buffers cannot change a validated snapshot after it has been handed off.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

ACTION_SCHEMA = "ee_delta_grasp_v0"
ACTION_NAMES = (
    "left_dx",
    "left_dy",
    "left_dz",
    "left_droll",
    "left_dpitch",
    "left_dyaw",
    "right_dx",
    "right_dy",
    "right_dz",
    "right_droll",
    "right_dpitch",
    "right_dyaw",
    "left_grasp",
    "right_grasp",
)
ACTION_DIM = len(ACTION_NAMES)
FloatArray = NDArray[np.float32]
ImageArray = NDArray[np.uint8]
Images = Mapping[str, ImageArray]
# Deliberately opaque: no planner may inspect or assume the shape of a latent.
LatentState = Any


class ContractError(ValueError):
    """A sample or component cannot satisfy a versioned runtime contract."""


def _array(value: Any, name: str, dtype: Any, ndim: int) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype):
        raise ContractError(f"{name} must be a numpy {np.dtype(dtype)} array")
    if value.ndim != ndim or any(size < 1 for size in value.shape):
        raise ContractError(f"{name} must have {ndim} nonempty axes; got {value.shape}")
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise ContractError(f"{name} must contain only finite values")
    return value


def _readonly(value: np.ndarray) -> np.ndarray:
    result = value.copy()
    result.flags.writeable = False
    return result


def _positive_integer(value: Any, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ContractError(f"{name} must be a positive integer")


def _named_strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ContractError(f"{name} must be a nonempty list or tuple of strings")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ContractError(f"{name} must contain nonempty strings")
    return tuple(value)


@dataclass(frozen=True)
class ActionSchema:
    """Normalized action identity; physical scales belong to an embodiment manifest."""

    version: str
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        names = _named_strings(self.names, "action names")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ContractError("action schema requires a version and nonempty component names")
        if len(names) != len(set(names)):
            raise ContractError("action component names must be unique")
        if self.version == ACTION_SCHEMA and names != ACTION_NAMES:
            raise ContractError("ee_delta_grasp_v0 action order is fixed")
        if self.version != ACTION_SCHEMA and not self.version.startswith("joint_delta_"):
            raise ContractError(
                "experimental joint actions require a separate joint_delta_* schema"
            )
        object.__setattr__(self, "names", names)

    @property
    def dimension(self) -> int:
        return len(self.names)


EE_DELTA_GRASP_V0 = ActionSchema(ACTION_SCHEMA, ACTION_NAMES)


def validate_actions(
    actions: np.ndarray, *, ndim: int = 4, schema: ActionSchema = EE_DELTA_GRASP_V0
) -> None:
    """Validate actions, normally [B,K,T,A], without clipping or dtype conversion."""
    _array(actions, "actions", np.float32, ndim)
    if actions.shape[-1] != schema.dimension:
        raise ContractError(f"actions last axis must have {schema.dimension} components")
    if np.any(np.abs(actions) > 1):
        raise ContractError("normalized actions must lie within [-1, 1]")


def validate_costs(costs: np.ndarray, shape: tuple[int, int, int]) -> None:
    """A backend distance must provide finite float32 costs [B,K,T]."""
    _array(costs, "costs", np.float32, 3)
    if costs.shape != shape:
        raise ContractError(f"costs must have shape {shape}; got {costs.shape}")


@dataclass(frozen=True)
class StateSchema:
    """Named state fields with units and explicit optionality; no guessed G1 ordering."""

    names: tuple[str, ...]
    units: tuple[str, ...]
    version: str
    required: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        names = _named_strings(self.names, "state names")
        units = _named_strings(self.units, "state units")
        required = tuple(self.required) if self.required else (True,) * len(names)
        if not isinstance(self.version, str) or not self.version.strip():
            raise ContractError("state schema requires a version and nonempty field names")
        if len(set(names)) != len(names):
            raise ContractError("state field names must be unique")
        if len(units) != len(names) or any(not unit for unit in units):
            raise ContractError("every state field needs an explicit unit (use '1' if unitless)")
        if len(required) != len(names) or any(type(value) is not bool for value in required):
            raise ContractError("state required flags must be one boolean per field")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "required", required)

    @property
    def dimension(self) -> int:
        return len(self.names)


def _validate_state(values: np.ndarray, mask: np.ndarray, schema: StateSchema, ndim: int) -> None:
    _array(values, "robot_state", np.float32, ndim)
    _array(mask, "state_mask", np.bool_, ndim)
    if values.shape != mask.shape or values.shape[-1] != schema.dimension:
        raise ContractError("state and mask shapes must agree with the state schema")
    if not mask[..., np.asarray(schema.required)].all():
        raise ContractError("a required robot state field is missing")


def _validate_timestamps(timestamps: np.ndarray, shape: tuple[int, ...]) -> None:
    _array(timestamps, "timestamps", np.float64, len(shape))
    if timestamps.shape != shape or np.any(timestamps < 0):
        raise ContractError(f"timestamps must be nonnegative seconds with shape {shape}")
    if len(shape) == 2 and np.any(np.diff(timestamps, axis=1) <= 0):
        raise ContractError("sequence timestamps must increase strictly within each episode")


def _images(images: Images, prefix: tuple[int, ...]) -> Images:
    if not isinstance(images, Mapping) or not images:
        raise ContractError("observations require at least one named RGB camera")
    result = {}
    for name, pixels in images.items():
        if not isinstance(name, str) or not name:
            raise ContractError("camera names must be nonempty strings")
        _array(pixels, f"camera {name}", np.uint8, len(prefix) + 3)
        if pixels.shape[: len(prefix)] != prefix or pixels.shape[-1] != 3:
            raise ContractError(f"camera {name} must have shape {prefix} + (H,W,3)")
        result[name] = _readonly(pixels)
    return MappingProxyType(result)


@dataclass(frozen=True)
class RobotState:
    """Batched state passed to a model, retaining masks, identity, and sensor time."""

    values: FloatArray
    mask: NDArray[np.bool_]
    timestamps: NDArray[np.float64]
    schema: StateSchema

    def __post_init__(self) -> None:
        _validate_state(self.values, self.mask, self.schema, 2)
        _validate_timestamps(self.timestamps, (self.values.shape[0],))
        for name in ("values", "mask", "timestamps"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


@dataclass(frozen=True)
class Observation:
    """One synchronized RGB/state snapshot per batch item, in a common clock domain."""

    images: Images
    robot_state: FloatArray
    state_mask: NDArray[np.bool_]
    timestamps: NDArray[np.float64]
    state_schema: StateSchema

    def __post_init__(self) -> None:
        _validate_state(self.robot_state, self.state_mask, self.state_schema, 2)
        batch_size = self.robot_state.shape[0]
        _validate_timestamps(self.timestamps, (batch_size,))
        object.__setattr__(self, "images", _images(self.images, (batch_size,)))
        for name in ("robot_state", "state_mask", "timestamps"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    @property
    def state(self) -> RobotState:
        return RobotState(self.robot_state, self.state_mask, self.timestamps, self.state_schema)

    def require_fresh(self, now: float, max_age: float) -> None:
        """Reject stale/future snapshots; caller supplies time from the sensor clock."""
        if not np.isfinite(now) or not np.isfinite(max_age) or now < 0 or max_age < 0:
            raise ContractError("now and max_age must be finite nonnegative seconds")
        ages = now - self.timestamps
        if np.any(ages < 0) or np.any(ages > max_age):
            raise ContractError("observation is stale or belongs to a future/different clock")


@dataclass(frozen=True)
class SequenceBatch:
    """Canonical T-transition windows: T+1 observations and T executed actions.

    Windows never cross an episode boundary. `terminated[:,t]` describes the
    transition ending at observation t+1 and may be true only at the last step.
    Values behind false state masks must still be finite; they are not measurements.
    """

    observations: Images
    robot_states: FloatArray
    state_mask: NDArray[np.bool_]
    actions: FloatArray
    timestamps: NDArray[np.float64]
    terminated: NDArray[np.bool_]
    episode_ids: tuple[str, ...]
    state_schema: StateSchema
    action_schema: ActionSchema = EE_DELTA_GRASP_V0

    def __post_init__(self) -> None:
        validate_actions(self.actions, ndim=3, schema=self.action_schema)
        batch_size, horizon, _ = self.actions.shape
        _validate_state(self.robot_states, self.state_mask, self.state_schema, 3)
        if self.robot_states.shape[:2] != (batch_size, horizon + 1):
            raise ContractError("sequence states must have shape [B,T+1,S] for actions [B,T,A]")
        _validate_timestamps(self.timestamps, (batch_size, horizon + 1))
        _array(self.terminated, "terminated", np.bool_, 2)
        if self.terminated.shape != (batch_size, horizon):
            raise ContractError("terminal markers must have shape [B,T]")
        if self.terminated[:, :-1].any():
            raise ContractError("a sequence window cannot cross an episode terminal boundary")
        ids = tuple(self.episode_ids)
        if len(ids) != batch_size or any(not isinstance(item, str) or not item for item in ids):
            raise ContractError("episode_ids must contain one nonempty string per batch item")
        object.__setattr__(self, "episode_ids", ids)
        object.__setattr__(
            self, "observations", _images(self.observations, (batch_size, horizon + 1))
        )
        for name in ("robot_states", "state_mask", "actions", "timestamps", "terminated"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    @property
    def batch_size(self) -> int:
        return self.actions.shape[0]

    @property
    def horizon(self) -> int:
        return self.actions.shape[1]

    def observation(self, step: int) -> Observation:
        if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= self.horizon:
            raise ContractError(f"observation step must be between 0 and {self.horizon}")
        return Observation(
            {name: frames[:, step] for name, frames in self.observations.items()},
            self.robot_states[:, step],
            self.state_mask[:, step],
            self.timestamps[:, step],
            self.state_schema,
        )

    def as_mapping(self) -> dict[str, Any]:
        """Return canonical field names; arrays remain read-only and schemas explicit."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_mapping(cls, batch: Mapping[str, Any]) -> SequenceBatch:
        unknown = set(batch) - set(cls.__dataclass_fields__)
        if unknown:
            raise ContractError(f"unknown canonical sequence fields: {sorted(unknown)}")
        try:
            return cls(**batch)
        except TypeError as error:
            raise ContractError(f"incomplete or invalid canonical sequence: {error}") from error


@dataclass(frozen=True)
class Capabilities:
    """Load-time negotiation; schema identity includes the ordered fields and units."""

    action_schema: ActionSchema
    state_schema: StateSchema
    max_horizon: int
    min_history: int = 1
    supported_devices: tuple[str, ...] = ("cpu",)
    # Declared physical readouts (``ReadoutWorldModel.readout``). A planner may use a
    # latent only through these names; an empty tuple means the model declares none.
    readouts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _positive_integer(self.max_horizon, "max_horizon")
        _positive_integer(self.min_history, "min_history")
        devices = tuple(self.supported_devices)
        if not devices or any(device not in ("cpu", "mps", "cuda") for device in devices):
            raise ContractError("capabilities must name supported cpu/mps/cuda devices")
        object.__setattr__(self, "supported_devices", devices)
        readouts = tuple(self.readouts)
        if any(not isinstance(name, str) or not name for name in readouts) or len(
            set(readouts)
        ) != len(readouts):
            raise ContractError("readout names must be distinct nonempty strings")
        object.__setattr__(self, "readouts", readouts)

    def require(
        self,
        *,
        action_schema: ActionSchema,
        state_schema: StateSchema,
        horizon: int,
        history: int = 1,
        device: str = "cpu",
    ) -> None:
        _positive_integer(horizon, "horizon")
        _positive_integer(history, "history")
        if action_schema != self.action_schema or state_schema != self.state_schema:
            raise ContractError("model and runtime action/state schemas are incompatible")
        if horizon > self.max_horizon or history < self.min_history:
            raise ContractError("requested horizon or observation history is unsupported")
        if device not in self.supported_devices:
            raise ContractError(f"model does not support device {device!r}")


@dataclass(frozen=True)
class ExecutionResult:
    """Transport acknowledgement; rejected/stopped requests never become transitions.

    An applied action is the normalized action that actually ran, including any
    clipping. 'stopped' reports a hold/stop, not an invented zero action (zero
    grasp means half-closed, not a hold). Times share the observation clock.
    """

    requested_action: FloatArray
    applied_action: FloatArray | None
    status: Literal["applied", "clipped", "rejected", "stopped"]
    timestamp: float
    reason: str = ""
    action_schema: ActionSchema = EE_DELTA_GRASP_V0

    def __post_init__(self) -> None:
        validate_actions(self.requested_action, ndim=1, schema=self.action_schema)
        if not np.isfinite(self.timestamp) or self.timestamp < 0:
            raise ContractError("execution timestamp must be finite nonnegative seconds")
        if self.status not in ("applied", "clipped", "rejected", "stopped"):
            raise ContractError("unknown execution status")
        if self.status in ("applied", "clipped"):
            if self.applied_action is None:
                raise ContractError("successful execution must report the applied action")
            validate_actions(self.applied_action, ndim=1, schema=self.action_schema)
            same = np.array_equal(self.requested_action, self.applied_action)
            if (self.status == "applied") != same:
                raise ContractError(
                    "execution status must distinguish unchanged and clipped actions"
                )
            object.__setattr__(self, "applied_action", _readonly(self.applied_action))
        elif self.applied_action is not None:
            raise ContractError("rejected/stopped requests cannot report an applied action")
        if self.status != "applied" and not self.reason:
            raise ContractError(
                "clipping, rejection, and stop responses require an explicit reason"
            )
        object.__setattr__(self, "requested_action", _readonly(self.requested_action))


@runtime_checkable
class WorldModel(Protocol):
    """Model-owned latents/preprocessing; no future state or goal proprioception required."""

    @property
    def capabilities(self) -> Capabilities: ...

    def encode(self, observation: Images, robot_state: RobotState) -> LatentState: ...

    def predict(self, z: LatentState, actions: FloatArray) -> LatentState:
        """Roll out [B,K,T,A] actions recursively; preserve all T predicted steps."""
        ...

    def encode_goal(self, goal: Images) -> LatentState: ...

    def distance(self, predicted_z: LatentState, goal_z: LatentState) -> FloatArray:
        """Return finite float32 [B,K,T] costs, lower being closer to the image goal."""
        ...

    def train_step(self, batch: SequenceBatch) -> Mapping[str, float]: ...

    def save(self, path: str | Path) -> None: ...

    def load(self, path: str | Path) -> None: ...


@runtime_checkable
class ReadoutWorldModel(WorldModel, Protocol):
    """A world model that also declares named physical readouts of its latents.

    ``readout`` maps an encoded ``[B,D]`` or predicted ``[B,K,T,D]`` latent to finite
    float32 arrays keyed by ``capabilities.readouts`` (trailing quantity dimension).
    It is the only sanctioned way for a planner or evaluator to derive a quantity
    from a latent; the latent payload itself stays opaque.
    """

    def readout(self, z: LatentState) -> Mapping[str, FloatArray]: ...


@runtime_checkable
class Embodiment(Protocol):
    """Robot-dependent frames, IK, scaling, safety limits and retargeting boundary."""

    @property
    def action_schema(self) -> ActionSchema: ...

    @property
    def state_schema(self) -> StateSchema: ...

    def observe(self) -> Observation:
        """Acquire a synchronized snapshot and cache it for state()."""
        ...

    def state(self) -> RobotState:
        """Return the state from the latest observe(), never a separately polled state."""
        ...

    def execute(self, normalized_action: FloatArray) -> ExecutionResult: ...

    def normalize_action(self, robot_action: FloatArray) -> FloatArray: ...

    def denormalize_action(self, action: FloatArray) -> FloatArray: ...

    def stop(self, reason: str) -> None:
        """Hold/stop without treating a zero grasp command as a safe hold."""
        ...


@runtime_checkable
class Transport(Protocol):
    """Low-level simulator/SDK boundary; commands are ordered physical joint targets.

    The embodiment maps raw transport fields into the canonical Observation.
    SDK/simulator objects never cross this boundary. reset/contact queries live
    in separate simulator/task interfaces, so hardware need not fabricate them.
    """

    def read(self) -> Mapping[str, Any]:
        """Return sensor fields with acquisition timestamps and alignment information."""
        ...

    def send_joint_targets(
        self, targets: FloatArray, *, joint_names: tuple[str, ...], deadline: float
    ) -> Mapping[str, Any]:
        """Acknowledge applied targets and time; reject stale commands before actuation."""
        ...

    def stop(self, reason: str) -> None: ...

    def close(self) -> None: ...
