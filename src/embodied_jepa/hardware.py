"""SDK2 mapping and in-memory transport rehearsal. No DDS or robot execution exists.

Message order is audited against official Unitree sources pinned in HARDWARE.md.
Joint calibration is mandatory; simulation limits/gains are never substituted.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from embodied_jepa.contracts import ContractError

SDK2_PYTHON_REVISION = "9c519023d188bfe4643d326868474878ab515ed8"
DEX3_MAPPING_REVISION = "817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b"
BODY_JOINTS = (
    *(
        f"left_{name}_joint"
        for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ),
    *(
        f"right_{name}_joint"
        for name in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ),
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    *(
        f"left_{name}_joint"
        for name in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    ),
    *(
        f"right_{name}_joint"
        for name in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    ),
)
LEFT_HAND_JOINTS = tuple(
    f"left_hand_{name}_joint"
    for name in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
)
RIGHT_HAND_JOINTS = tuple(
    f"right_hand_{name}_joint"
    for name in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
)
SDK_JOINTS = BODY_JOINTS + LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS
TOPICS = MappingProxyType(
    {
        "body_command": "rt/lowcmd",
        "body_state": "rt/lowstate",
        "left_command": "rt/dex3/left/cmd",
        "left_state": "rt/dex3/left/state",
        "right_command": "rt/dex3/right/cmd",
        "right_state": "rt/dex3/right/state",
    }
)


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ContractError(f"{name} must be a finite measured number")
    if positive and value <= 0:
        raise ContractError(f"{name} must be positive")
    return float(value)


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} is required")


def _digest(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ContractError(f"{name} must be a SHA-256 digest of a commissioned manifest")


@dataclass(frozen=True)
class JointCalibration:
    """Canonical radians to SDK radians: sdk_q = sign * canonical_q + zero_offset_rad."""

    sign: int
    zero_offset_rad: float
    lower_rad: float
    upper_rad: float
    velocity_limit_rad_s: float
    kp: float
    kd: float

    def __post_init__(self) -> None:
        if type(self.sign) is not int or self.sign not in (-1, 1):
            raise ContractError("joint sign must be explicitly calibrated +1 or -1")
        for name in (
            "zero_offset_rad",
            "lower_rad",
            "upper_rad",
            "velocity_limit_rad_s",
            "kp",
            "kd",
        ):
            _number(getattr(self, name), name)
        if (
            self.lower_rad >= self.upper_rad
            or self.velocity_limit_rad_s <= 0
            or self.kp < 0
            or self.kd < 0
        ):
            raise ContractError(
                "joint limits must be ordered, velocity positive, and gains nonnegative"
            )


@dataclass(frozen=True)
class HardwareCalibration:
    robot_serial: str
    robot_variant: str
    calibration_id: str
    source: str
    clock_domain: str
    camera_calibration_sha256: str
    action_manifest_sha256: str
    expected_mode_machine: int
    command_period_s: float
    max_state_age_s: float
    max_sensor_skew_s: float
    joints: Mapping[str, JointCalibration]

    def __post_init__(self) -> None:
        for name in ("robot_serial", "robot_variant", "calibration_id", "clock_domain"):
            _text(getattr(self, name), name)
        if self.source not in ("mock_fixture", "measured_hardware"):
            raise ContractError("source must distinguish mock_fixture from measured_hardware")
        _digest(self.camera_calibration_sha256, "camera_calibration_sha256")
        _digest(self.action_manifest_sha256, "action_manifest_sha256")
        if (
            type(self.expected_mode_machine) is not int
            or not 0 <= self.expected_mode_machine <= 255
        ):
            raise ContractError("expected_mode_machine must be the observed uint8 mode")
        for name in ("command_period_s", "max_state_age_s", "max_sensor_skew_s"):
            _number(getattr(self, name), name, positive=True)
        if not isinstance(self.joints, Mapping) or set(self.joints) != set(SDK_JOINTS):
            raise ContractError("calibration requires all 29 body and 7+7 named hand joints")
        if any(not isinstance(value, JointCalibration) for value in self.joints.values()):
            raise ContractError("every joint requires a complete JointCalibration")
        object.__setattr__(self, "joints", MappingProxyType(dict(self.joints)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> HardwareCalibration:
        fields = dict(value)
        if fields.pop("version", None) != "g1_sdk2_calibration_v0":
            raise ContractError("missing or incompatible hardware calibration version")
        try:
            fields["joints"] = {
                name: JointCalibration(**cal) for name, cal in fields["joints"].items()
            }
            return cls(**fields)
        except (KeyError, TypeError, AttributeError) as error:
            raise ContractError(
                "hardware calibration is incomplete; physical defaults are forbidden"
            ) from error


def calibration_template() -> dict[str, Any]:
    """An intentionally invalid template: no physical limits, offsets or gains guessed."""
    fields = {name: None for name in HardwareCalibration.__dataclass_fields__}
    fields["version"] = "g1_sdk2_calibration_v0"
    fields["joints"] = {
        joint: {name: None for name in JointCalibration.__dataclass_fields__}
        for joint in SDK_JOINTS
    }
    return fields


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float32
        or value.shape != (size,)
        or not np.isfinite(value).all()
    ):
        raise ContractError(f"{name} must be finite float32[{size}]")
    return value


def _joint_order(names: tuple[str, ...]) -> None:
    if len(names) != 43 or len(set(names)) != 43 or set(names) != set(SDK_JOINTS):
        raise ContractError("joint_names must identify every G1/Dex3 joint exactly once")


@dataclass(frozen=True)
class MockSensorPacket:
    """Already clock-aligned fake SDK telemetry plus RGB; not a DDS message class."""

    sequence: int
    body_q: np.ndarray
    body_dq: np.ndarray
    left_q: np.ndarray
    left_dq: np.ndarray
    right_q: np.ndarray
    right_dq: np.ndarray
    rgb: np.ndarray
    timestamps: Mapping[str, float]
    mode_machine: int
    clock_domain: str
    fault: bool = False

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise ContractError("sensor sequence must be a nonnegative integer")
        if type(self.mode_machine) is not int or not 0 <= self.mode_machine <= 255:
            raise ContractError("mode_machine must be uint8")
        if type(self.fault) is not bool:
            raise ContractError("fault must be boolean")
        _text(self.clock_domain, "clock_domain")
        for name in ("body_q", "body_dq", "left_q", "left_dq", "right_q", "right_dq"):
            value = _vector(getattr(self, name), 35 if name.startswith("body") else 7, name).copy()
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if (
            not isinstance(self.rgb, np.ndarray)
            or self.rgb.dtype != np.uint8
            or self.rgb.ndim != 3
            or self.rgb.shape[-1] != 3
            or min(self.rgb.shape) < 1
        ):
            raise ContractError("camera must supply actual uint8 RGB[H,W,3] or fail")
        pixels = self.rgb.copy()
        pixels.flags.writeable = False
        object.__setattr__(self, "rgb", pixels)
        if set(self.timestamps) != {"body", "left", "right", "camera"}:
            raise ContractError("each sensor stream requires its own acquisition timestamp")
        for value in self.timestamps.values():
            if _number(value, "sensor timestamp") < 0:
                raise ContractError("sensor timestamps must be nonnegative")
        object.__setattr__(self, "timestamps", MappingProxyType(dict(self.timestamps)))


class MockSDK2Endpoint:
    """In-memory fake endpoint with no DDS, wire serialization, or robot execution."""

    def __init__(self):
        self.connected = True
        self.packet: MockSensorPacket | None = None
        self.commands: list[dict[str, Any]] = []
        self.stops: list[str] = []

    def push(self, packet: MockSensorPacket) -> None:
        if not isinstance(packet, MockSensorPacket):
            raise ContractError("mock endpoint accepts MockSensorPacket only")
        self.packet = packet


class MockSDK2Transport:
    """Explicitly enabled in-memory mapping rehearsal; physical_execution=True always fails."""

    def __init__(
        self,
        calibration: HardwareCalibration,
        endpoint: MockSDK2Endpoint,
        *,
        joint_names: tuple[str, ...] = SDK_JOINTS,
        clock: Callable[[], float] = time.monotonic,
        physical_execution: bool = False,
    ):
        if physical_execution is not False:
            raise NotImplementedError("physical SDK2 execution is not implemented or enabled")
        if not isinstance(calibration, HardwareCalibration):
            raise ContractError("a complete hardware calibration is required")
        if type(endpoint) is not MockSDK2Endpoint:
            raise ContractError("only the built-in in-memory mock endpoint is supported")
        _joint_order(joint_names)
        self.calibration, self.endpoint = calibration, endpoint
        self.joint_names, self.clock = tuple(joint_names), clock
        self.enabled, self.closed = False, False
        self._sequence = -1
        self._consumed_sequence = -1
        self._resume_after_sequence = -1
        self._last_targets: np.ndarray | None = None
        self._last_sensor_times: np.ndarray | None = None
        self._last_command_time: float | None = None

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("mock SDK2 transport is closed")

    def stop(self, reason: str) -> None:
        _text(reason, "stop reason")
        self.enabled = False
        self._last_targets = None
        self._last_command_time = None
        self._resume_after_sequence = max(self._resume_after_sequence, self._sequence)
        self.endpoint.stops.append(reason)

    def _fail(self, reason: str) -> None:
        self.stop(reason)
        raise ContractError(reason)

    def read(self) -> dict[str, Any]:
        self._require_open()
        if not self.endpoint.connected or self.endpoint.packet is None:
            self._fail("disconnected or no sensor packet")
        packet = self.endpoint.packet
        if packet.fault:
            self._fail("body or hand fault reported")
        if packet.clock_domain != self.calibration.clock_domain:
            self._fail("sensor clock domain mismatch")
        if packet.mode_machine != self.calibration.expected_mode_machine:
            self._fail("robot mode_machine changed or differs from calibrated configuration")
        times = np.array([packet.timestamps[key] for key in ("body", "left", "right", "camera")])
        now = _number(self.clock(), "clock")
        if np.any(times > now) or np.any(now - times > self.calibration.max_state_age_s):
            self._fail("stale or future sensor timestamp")
        if np.ptp(times) > self.calibration.max_sensor_skew_s:
            self._fail("camera/body/hand streams are not synchronized")
        if packet.sequence < self._sequence or (
            self._last_sensor_times is not None and np.any(times < self._last_sensor_times)
        ):
            self._fail("sensor sequence or clock moved backwards")
        self._sequence, self._last_sensor_times = packet.sequence, times.copy()
        sdk_q = np.concatenate((packet.body_q[:29], packet.left_q, packet.right_q))
        sdk_dq = np.concatenate((packet.body_dq[:29], packet.left_dq, packet.right_dq))
        positions, velocities = {}, {}
        for index, name in enumerate(SDK_JOINTS):
            cal = self.calibration.joints[name]
            positions[name] = (sdk_q[index] - cal.zero_offset_rad) / cal.sign
            velocities[name] = sdk_dq[index] / cal.sign
            if not cal.lower_rad <= positions[name] <= cal.upper_rad:
                self._fail(f"measured joint outside calibrated range: {name}")
            if abs(velocities[name]) > cal.velocity_limit_rad_s:
                self._fail(f"measured velocity exceeds calibrated limit: {name}")
        return {
            "rgb": packet.rgb.copy(),
            "qpos": np.array([positions[n] for n in self.joint_names], dtype=np.float32),
            "qvel": np.array([velocities[n] for n in self.joint_names], dtype=np.float32),
            "joint_names": self.joint_names,
            "timestamp": float(times[0]),
            "sensor_timestamps": dict(packet.timestamps),
            "clock_domain": packet.clock_domain,
            "sequence": packet.sequence,
            "mock_only": True,
        }

    def enable_mock(self) -> None:
        """Arm only local command recording after validating fresh aligned fake telemetry."""
        raw = self.read()
        if raw["sequence"] <= self._resume_after_sequence:
            self._fail("a new sensor packet is required after stop before explicit re-enable")
        self._last_targets = raw["qpos"].copy()
        self.enabled = True

    def _envelope(self, canonical: Mapping[str, float], timestamp: float) -> dict[str, Any]:
        def motor(name, mode):
            cal = self.calibration.joints[name]
            return {
                "name": name,
                "mode": mode,
                "q": cal.sign * canonical[name] + cal.zero_offset_rad,
                "dq": 0.0,
                "tau": 0.0,
                "kp": cal.kp,
                "kd": cal.kd,
            }

        body = [motor(name, 1) for name in BODY_JOINTS]
        # LowCmd IDL has 35 slots; the six unmapped slots remain explicitly disabled.
        body.extend(
            {"name": None, "mode": 0, "q": 0.0, "dq": 0.0, "tau": 0.0, "kp": 0.0, "kd": 0.0}
            for _ in range(6)
        )
        return {
            "mock_only": True,
            "wire_serializable": False,
            "timestamp": timestamp,
            "body": {
                "topic": TOPICS["body_command"],
                "mode_pr": 0,
                "mode_machine": self.calibration.expected_mode_machine,
                "motor_cmd": body,
                "crc": None,
                "crc_required_before_real_publication": True,
            },
            "left": {
                "topic": TOPICS["left_command"],
                "motor_cmd": [motor(name, i | (1 << 4)) for i, name in enumerate(LEFT_HAND_JOINTS)],
            },
            "right": {
                "topic": TOPICS["right_command"],
                "motor_cmd": [
                    motor(name, i | (1 << 4)) for i, name in enumerate(RIGHT_HAND_JOINTS)
                ],
            },
        }

    def send_joint_targets(self, targets, *, joint_names, deadline):
        self._require_open()
        targets = _vector(targets, 43, "targets")
        _joint_order(tuple(joint_names))
        if tuple(joint_names) != self.joint_names:
            raise ContractError("targets must use the declared transport joint order")
        _number(deadline, "deadline")
        if not self.enabled:
            self._fail("mock command publication is disabled; explicit enable_mock is required")
        raw = self.read()
        now = _number(self.clock(), "clock")
        if deadline < now:
            self._fail("command deadline expired")
        if raw["sequence"] <= self._consumed_sequence:
            self._fail("sensor packet already consumed; acquire a new snapshot")
        if (
            self._last_command_time is not None
            and now - self._last_command_time < self.calibration.command_period_s - 1e-9
        ):
            self._fail("command cadence exceeds calibrated frequency")
        for index, name in enumerate(self.joint_names):
            cal = self.calibration.joints[name]
            if not cal.lower_rad <= targets[index] <= cal.upper_rad:
                self._fail(f"target outside calibrated range: {name}")
            if (
                abs(targets[index] - self._last_targets[index])
                > cal.velocity_limit_rad_s * self.calibration.command_period_s + 1e-7
            ):
                self._fail(f"target rate exceeds calibrated limit: {name}")
        envelope = self._envelope(dict(zip(self.joint_names, targets.tolist(), strict=True)), now)
        self.endpoint.commands.append(copy.deepcopy(envelope))
        self._last_targets = targets.copy()
        self._consumed_sequence = raw["sequence"]
        self._last_command_time = now
        return {
            "status": "applied",
            "targets": targets.copy(),
            "timestamp": now,
            "mock_only": True,
            "physical_execution": False,
        }

    def poll_watchdog(self) -> None:
        """Explicitly exercise timeout handling; no background real-time watchdog is claimed."""
        self.read()

    def close(self) -> None:
        if not self.closed:
            self.stop("transport closed")
            self.closed = True
