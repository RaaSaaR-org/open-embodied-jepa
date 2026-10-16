"""SO-101 embodiment adapter for ``joint_delta_so101_v0`` (SO-101 S1, TASK-097).

IK-free joint-position control. A normalized action is six components in LeRobot
``so_follower`` order: five arm joint-position deltas and one absolute gripper command.

- Arm: ``a[i] * joint_delta_per_step_rad[i]`` is added to the current joint-position *target*
  (the command history, not the measured pose) and the sum is clipped to the manifest's joint
  limits. The scales are powers of two, so the arm channels round-trip exactly in float32.
- Gripper: ``-1`` maps to ``gripper_closed_rad`` and ``+1`` to ``gripper_open_rad``; the target
  moves toward the command by at most ``gripper_max_delta_per_step_rad`` per step. A zero
  gripper command is a half-open jaw, not a hold.

A clip is never applied silently: ``execute`` returns status ``"clipped"`` with the normalized
action that actually ran and the reason, and the requested action is returned unchanged.

Physical scales, limits and the gripper mapping are ``configs/so101_sim_action.json``'s
(``so101_sim_action_v0``), never the schema's. Simulation only; nothing here talks to hardware.
NumPy only at import; the MuJoCo transport is ``so101_simulation.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from embodied_jepa.action_schemas import JOINT_DELTA_SO101_V0
from embodied_jepa.contracts import (
    ContractError,
    ExecutionResult,
    Observation,
    StateSchema,
    validate_actions,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTION_MANIFEST = ROOT / "configs/so101_sim_action.json"
ARM = JOINT_DELTA_SO101_V0.names[:-1]
STATE_SCHEMA_VERSION = "so101_proprio_v0"
# Limits may differ from the compiled model only by the manifest's decimal rounding.
LIMIT_TOLERANCE_RAD = 1e-9


def load_action_manifest(path=None) -> dict:
    """Read and validate an SO-101 action manifest (identity, scales, limits, gripper)."""
    manifest = json.loads(Path(path or DEFAULT_ACTION_MANIFEST).read_text())
    if (
        manifest.get("action_schema") != JOINT_DELTA_SO101_V0.version
        or manifest.get("calibration") != "simulation_only_not_for_hardware"
        or manifest.get("gripper_command") != "absolute_joint_position"
    ):
        raise ContractError("unsupported SO-101 action manifest identity")
    for field in (
        "control_dt_s",
        "gripper_max_delta_per_step_rad",
        "measured_joint_velocity_stop_rad_s",
        "max_snapshot_age_s",
        "max_wall_snapshot_age_s",
    ):
        value = manifest.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
            raise ContractError(f"{field} must be finite and positive")
        if not np.isfinite(value):
            raise ContractError(f"{field} must be finite and positive")
    scales = manifest.get("joint_delta_per_step_rad")
    if not isinstance(scales, dict) or tuple(scales) != ARM:
        raise ContractError(f"joint_delta_per_step_rad must name {list(ARM)} in order")
    if not all(np.isfinite(scales[n]) and scales[n] > 0 for n in ARM):
        raise ContractError("joint deltas must be finite and positive")
    limits = manifest.get("joint_limits_rad")
    if not isinstance(limits, dict) or tuple(limits) != JOINT_DELTA_SO101_V0.names:
        raise ContractError(
            f"joint_limits_rad must name {list(JOINT_DELTA_SO101_V0.names)} in order"
        )
    bounds = np.asarray([limits[n] for n in JOINT_DELTA_SO101_V0.names], dtype=float)
    if (
        bounds.shape != (6, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        raise ContractError("joint limits must be ordered finite pairs")
    closed, opened = manifest.get("gripper_closed_rad"), manifest.get("gripper_open_rad")
    if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in (closed, opened)):
        raise ContractError("gripper range must be finite")
    if not bounds[5, 0] <= closed < opened <= bounds[5, 1]:
        raise ContractError("gripper range must be ordered and within the gripper joint limits")
    home = np.asarray(manifest.get("home_rad"), dtype=float)
    if (
        home.shape != (6,)
        or not np.isfinite(home).all()
        or np.any(home < bounds[:, 0])
        or np.any(home > bounds[:, 1])
    ):
        raise ContractError("home_rad must be six finite joint positions within limits")
    return manifest


class SO101Embodiment:
    """``Embodiment`` for the simulated SO-101 with a wrist and an overview camera."""

    action_schema = JOINT_DELTA_SO101_V0

    def __init__(self, simulation, *, action_manifest=None):
        self.sim = simulation
        self.manifest = m = load_action_manifest(action_manifest)
        if not np.isclose(m["control_dt_s"], simulation.control_dt, rtol=0, atol=1e-12):
            raise ContractError("manifest and transport control timestep disagree")
        if tuple(simulation.joint_names) != JOINT_DELTA_SO101_V0.names:
            raise ContractError("transport joint order differs from joint_delta_so101_v0")
        self.limits = np.asarray(
            [m["joint_limits_rad"][n] for n in JOINT_DELTA_SO101_V0.names], dtype=float
        )
        if np.any(np.abs(self.limits - simulation.limits) > LIMIT_TOLERANCE_RAD):
            raise ContractError("manifest joint limits differ from the compiled model's")
        # The transport's exact limits govern clipping; the manifest's are the recorded copy.
        self.limits = np.asarray(simulation.limits, dtype=float).copy()
        self.scales = np.asarray([m["joint_delta_per_step_rad"][n] for n in ARM], dtype=np.float32)
        self.gripper_closed = float(m["gripper_closed_rad"])
        self.gripper_open = float(m["gripper_open_rad"])
        self.home = np.asarray(m["home_rad"], dtype=float)
        names = simulation.joint_names
        self.state_schema = StateSchema(
            tuple(f"{name}.position" for name in names)
            + tuple(f"{name}.velocity" for name in names),
            ("rad",) * len(names) + ("rad/s",) * len(names),
            STATE_SCHEMA_VERSION,
        )
        self._observation = None
        self._observation_wall = 0.0

    def reset(self, seed=0, **kwargs):
        """Reset the scene with the arm at ``home_rad`` unless joint positions are given."""
        kwargs.setdefault("joint_positions", self.home)
        truth = self.sim.reset(seed, **kwargs)
        self._observation = None
        return truth

    def observe(self):
        raw = self.sim.read()
        if tuple(raw["joint_names"]) != self.sim.joint_names:
            raise ContractError("transport joint order changed")
        values = np.concatenate((raw["qpos"], raw["qvel"])).astype(np.float32)[None]
        self._observation = Observation(
            {name: image[None] for name, image in raw["rgb"].items()},
            values,
            np.ones(values.shape, dtype=bool),
            np.array([raw["timestamp"]], dtype=np.float64),
            self.state_schema,
        )
        self._observation_wall = time.monotonic()
        return self._observation

    def state(self):
        if self._observation is None:
            raise ContractError("observe() must acquire a synchronized snapshot first")
        return self._observation.state

    def _gripper_position(self, normalized):
        width = self.gripper_open - self.gripper_closed
        return self.gripper_closed + (float(normalized) + 1.0) / 2.0 * width

    def _gripper_normalized(self, position):
        width = self.gripper_open - self.gripper_closed
        return 2.0 * (float(position) - self.gripper_closed) / width - 1.0

    def denormalize_action(self, action):
        """Normalized [6] -> physical: five arm deltas (rad) and a gripper target (rad)."""
        validate_actions(action, ndim=1, schema=JOINT_DELTA_SO101_V0)
        physical = np.empty(6, dtype=np.float32)
        physical[:5] = action[:5] * self.scales
        physical[5] = self._gripper_position(action[5])
        return physical

    def normalize_action(self, robot_action):
        action = np.asarray(robot_action)
        if action.dtype != np.float32 or action.shape != (6,) or not np.isfinite(action).all():
            raise ContractError("physical action must be finite float32[6]")
        normalized = np.empty(6, dtype=np.float32)
        normalized[:5] = action[:5] / self.scales
        normalized[5] = self._gripper_normalized(action[5])
        validate_actions(normalized, ndim=1, schema=JOINT_DELTA_SO101_V0)
        return normalized

    def hold_action(self):
        """The normalized action that keeps the current targets: zero arm deltas and the
        gripper command of the current gripper target (a zero gripper command is half-open)."""
        gripper = np.clip(self._gripper_normalized(self.sim.targets[5]), -1.0, 1.0)
        return np.array([0, 0, 0, 0, 0, gripper], dtype=np.float32)

    def preview(self, normalized_action, targets=None):
        """The joint targets an action would command from ``targets``, and its clip reasons.

        Returns ``(new_targets, applied_normalized, reasons)``; nothing is sent.
        """
        validate_actions(normalized_action, ndim=1, schema=JOINT_DELTA_SO101_V0)
        old = np.asarray(self.sim.targets if targets is None else targets, dtype=float).copy()
        applied = normalized_action.copy()
        new = old.copy()
        reasons = []
        desired = old[:5] + normalized_action[:5].astype(float) * self.scales.astype(float)
        new[:5] = np.clip(desired, self.limits[:5, 0], self.limits[:5, 1])
        for i in np.flatnonzero(new[:5] != desired):
            reasons.append(f"{ARM[i]} target clipped to joint limit")
            applied[i] = np.float32(np.clip((new[i] - old[i]) / float(self.scales[i]), -1.0, 1.0))
        requested_gripper = self._gripper_position(normalized_action[5])
        step = self.manifest["gripper_max_delta_per_step_rad"]
        gripper = float(np.clip(requested_gripper, old[5] - step, old[5] + step))
        gripper = float(np.clip(gripper, self.limits[5, 0], self.limits[5, 1]))
        new[5] = gripper
        if gripper != requested_gripper:
            reasons.append("gripper target rate-limited")
            applied[5] = np.float32(np.clip(self._gripper_normalized(gripper), -1.0, 1.0))
        return new, applied, reasons

    def execute(self, normalized_action):
        validate_actions(normalized_action, ndim=1, schema=JOINT_DELTA_SO101_V0)
        requested = normalized_action.copy()

        def reject(reason):
            self.stop(reason)
            return ExecutionResult(
                requested,
                None,
                "stopped",
                float(self.sim.data.time),
                reason,
                action_schema=JOINT_DELTA_SO101_V0,
            )

        if self._observation is None:
            return reject("no synchronized observation")
        try:
            self._observation.require_fresh(
                float(self.sim.data.time), self.manifest["max_snapshot_age_s"]
            )
        except ContractError:
            return reject("stale simulation observation")
        if not np.isclose(self._observation.timestamps[0], self.sim.data.time, rtol=0, atol=1e-9):
            return reject("observation already consumed; observe before replanning")
        if time.monotonic() - self._observation_wall > self.manifest["max_wall_snapshot_age_s"]:
            return reject("wall-clock planning deadline expired")
        if np.any(
            np.abs(self.sim.data.qvel[self.sim.vadr])
            > self.manifest["measured_joint_velocity_stop_rad_s"]
        ):
            return reject("measured joint velocity limit exceeded")
        targets, applied, reasons = self.preview(requested)
        ack = self.sim.send_joint_targets(
            targets,
            joint_names=self.sim.joint_names,
            deadline=float(self._observation.timestamps[0] + self.manifest["max_snapshot_age_s"]),
        )
        if ack["status"] != "applied":
            return reject(ack["reason"])
        same = np.array_equal(applied, requested)
        return ExecutionResult(
            requested,
            applied,
            "applied" if same else "clipped",
            ack["timestamp"],
            "" if same else "; ".join(reasons),
            action_schema=JOINT_DELTA_SO101_V0,
        )

    def close(self):
        self.sim.close()

    def stop(self, reason):
        """Hold the measured pose and require a fresh observation before resuming."""
        self.sim.stop(reason)
        self._observation = None
