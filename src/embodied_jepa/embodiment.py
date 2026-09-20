"""Named G1 arm IK and mirrored Dex3 synergy; conservative simulation-only limits."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from embodied_jepa.constraints import CandidateProjection
from embodied_jepa.contracts import (
    ACTION_SCHEMA,
    EE_DELTA_GRASP_V0,
    ContractError,
    ExecutionResult,
    Observation,
    StateSchema,
    validate_actions,
)

ROOT = Path(__file__).resolve().parents[2]
ARM = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)


def rotation_delta(rpy):
    r, p, y = rpy
    cr, cp, cy = np.cos(r), np.cos(p), np.cos(y)
    sr, sp, sy = np.sin(r), np.sin(p), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


class G1Embodiment:
    action_schema = EE_DELTA_GRASP_V0

    def __init__(self, simulation, *, action_manifest=None):
        self.sim = simulation
        self.mj, self.model = simulation.mj, simulation.model
        self.manifest = json.loads(
            Path(action_manifest or ROOT / "configs/g1_sim_action.json").read_text()
        )
        m = self.manifest
        if (
            m["action_schema"] != ACTION_SCHEMA
            or m["calibration"] != "simulation_only_not_for_hardware"
        ):
            raise ContractError("unsupported action manifest identity")
        if not np.isclose(m["control_dt_s"], simulation.control_dt):
            raise ContractError("manifest and transport control timestep disagree")
        for field in (
            "translation_per_step_m",
            "rotation_per_step_rad",
            "joint_speed_limit_rad_s",
            "measured_joint_velocity_stop_rad_s",
            "ik_damping",
            "ik_position_tolerance_m",
            "ik_rotation_tolerance_rad",
            "max_wall_snapshot_age_s",
            "max_snapshot_age_s",
        ):
            if not np.isfinite(m[field]) or m[field] <= 0:
                raise ContractError(f"{field} must be finite and positive")
        if not isinstance(m["ik_iterations"], int) or m["ik_iterations"] < 1:
            raise ContractError("ik_iterations must be a positive integer")
        self.scales = np.array(
            [*[m["translation_per_step_m"]] * 3, *[m["rotation_per_step_rad"]] * 3] * 2
            + [1.0, 1.0],
            dtype=np.float32,
        )
        names = simulation.joint_names
        self.state_schema = StateSchema(
            tuple(f"{name}.position" for name in names)
            + tuple(f"{name}.velocity" for name in names),
            ("rad",) * len(names) + ("rad/s",) * len(names),
            "g1_dex3_proprio_v0",
        )
        self.arm_ids = {
            side: np.array([self.model.joint(f"{side}_{name}_joint").id for name in ARM])
            for side in ("left", "right")
        }
        self.hand_ids = {
            side: np.array(
                [self.model.joint(f"{side}_hand_{name}_joint").id for name in m["grasp_fingers"]]
            )
            for side in ("left", "right")
        }
        self.actuator_for_joint = {int(j): i for i, j in enumerate(simulation.joint_ids)}
        self._scratch = self.mj.MjData(self.model)
        self._observation = None
        self._observation_wall = 0.0
        self._grasp = np.array([-1.0, -1.0])
        for side in ("left", "right"):
            limits = self.model.jnt_range[self.hand_ids[side]]
            for endpoint in ("open", "closed"):
                values = np.asarray(m[f"{side}_{endpoint}_rad"])
                if (
                    values.shape != (7,)
                    or not np.isfinite(values).all()
                    or np.any(values < limits[:, 0])
                    or np.any(values > limits[:, 1])
                ):
                    raise ContractError("grasp synergy violates named joint limits")
            workspace = np.asarray(m["workspace_base_m"][side])
            if (
                workspace.shape != (2, 3)
                or not np.isfinite(workspace).all()
                or np.any(workspace[0] >= workspace[1])
            ):
                raise ContractError("workspace must contain ordered finite 3D bounds")

    def reset(self, seed=0, **kwargs):
        self.sim.reset(seed, **kwargs)
        self._observation = None
        self._grasp[:] = -1
        for side in ("left", "right"):
            qadr = self.model.jnt_qposadr[self.hand_ids[side]]
            self.sim.data.qpos[qadr] = self.manifest[f"{side}_open_rad"]
        self.mj.mj_forward(self.model, self.sim.data)
        self.sim.targets[:] = self.sim.data.qpos[self.sim.qadr]
        return self.sim.task_truth()

    def observe(self):
        raw = self.sim.read()
        if tuple(raw["joint_names"]) != self.sim.joint_names:
            raise ContractError("transport joint order changed")
        values = np.concatenate((raw["qpos"], raw["qvel"])).astype(np.float32)[None]
        self._observation = Observation(
            {"onboard_rgb": raw["rgb"][None]},
            values,
            np.ones(values.shape, dtype=bool),
            np.array([raw["timestamp"]], dtype=np.float64),
            self.state_schema,
        )
        self._observation_wall = time.monotonic()
        # Preserve sensor precision and command history for repeatable planning.
        self._projection_q = raw["qpos"].copy()
        self._projection_targets = self.sim.targets.copy()
        return self._observation

    def state(self):
        if self._observation is None:
            raise ContractError("observe() must acquire a synchronized snapshot first")
        return self._observation.state

    def denormalize_action(self, action):
        validate_actions(action, ndim=1)
        physical = action * self.scales
        physical[12:] = (action[12:] + 1) / 2
        return physical.astype(np.float32)

    def normalize_action(self, robot_action):
        action = np.asarray(robot_action)
        if action.dtype != np.float32 or action.shape != (14,) or not np.isfinite(action).all():
            raise ContractError("physical action must be finite float32[14]")
        normalized = action / self.scales
        normalized[12:] = 2 * action[12:] - 1
        validate_actions(normalized, ndim=1)
        return normalized.astype(np.float32)

    def ee_pose(self, side, *, data=None):
        if side not in self.arm_ids:
            raise ContractError("side must be left or right")
        data = self.sim.data if data is None else data
        base = data.body("pelvis")
        rotation = base.xmat.reshape(3, 3)
        site = data.site(f"{side}_ee")
        return rotation.T @ (site.xpos - base.xpos), rotation.T @ site.xmat.reshape(3, 3)

    def solve_ik(self, side, position, rotation, *, data=None):
        """Solve using a scratch state; never teleport the executing simulator."""
        position, rotation = np.asarray(position), np.asarray(rotation)
        if (
            position.shape != (3,)
            or rotation.shape != (3, 3)
            or not np.isfinite(position).all()
            or not np.isfinite(rotation).all()
        ):
            return None
        if side not in self.arm_ids:
            return None
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-5
        ):
            return None
        bounds = np.asarray(self.manifest["workspace_base_m"][side])
        if np.any(position < bounds[0]) or np.any(position > bounds[1]):
            return None
        data = self.sim.data if data is None else data
        scratch = self._scratch
        scratch.qpos[:] = data.qpos
        scratch.qvel[:] = 0
        ids = self.arm_ids[side]
        qadr, vadr = self.model.jnt_qposadr[ids], self.model.jnt_dofadr[ids]
        limits = self.model.jnt_range[ids]
        site_id = self.model.site(f"{side}_ee").id
        base = data.body("pelvis")
        base_rotation = base.xmat.reshape(3, 3)
        target_position = base.xpos + base_rotation @ position
        target_rotation = base_rotation @ rotation
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(self.manifest["ik_iterations"]):
            self.mj.mj_forward(self.model, scratch)
            current_rotation = scratch.site_xmat[site_id].reshape(3, 3)
            dp = target_position - scratch.site_xpos[site_id]
            error_quaternion = np.empty(4)
            self.mj.mju_mat2Quat(error_quaternion, (target_rotation @ current_rotation.T).ravel())
            if error_quaternion[0] < 0:
                error_quaternion *= -1
            dr = np.empty(3)
            self.mj.mju_quat2Vel(dr, error_quaternion, 1.0)
            if (
                np.linalg.norm(dp) < self.manifest["ik_position_tolerance_m"]
                and np.linalg.norm(dr) < self.manifest["ik_rotation_tolerance_rad"]
            ):
                return scratch.qpos[qadr].copy()
            self.mj.mj_jacSite(self.model, scratch, jacp, jacr, site_id)
            jac = np.vstack((jacp[:, vadr], 0.3 * jacr[:, vadr]))
            error = np.concatenate((dp, 0.3 * dr))
            try:
                delta = jac.T @ np.linalg.solve(
                    jac @ jac.T + self.manifest["ik_damping"] ** 2 * np.eye(6), error
                )
            except np.linalg.LinAlgError:
                return None
            scratch.qpos[qadr] = np.clip(
                scratch.qpos[qadr] + np.clip(delta, -0.12, 0.12), limits[:, 0], limits[:, 1]
            )
        return None

    def _prepare_side(self, applied, targets, side_index, side, *, data=None):
        """Shared acceptance calculation on caller-owned action/target working copies."""
        previous_targets = targets.copy()
        max_delta = self.manifest["joint_speed_limit_rad_s"] * self.sim.control_dt
        reasons = []
        offset = side_index * 6
        position, rotation = self.ee_pose(side, data=data)
        target_position = position + applied[offset : offset + 3] * self.scales[offset : offset + 3]
        bounds = np.asarray(self.manifest["workspace_base_m"][side])
        clipped = np.clip(target_position, bounds[0], bounds[1])
        if not np.array_equal(target_position, clipped):
            applied[offset : offset + 3] = (
                (clipped - position) / self.scales[offset : offset + 3]
            ).astype(np.float32)
            if np.any(np.abs(applied[offset : offset + 3]) > 1):
                raise ContractError("current pose outside workspace")
            reasons.append(f"{side} workspace")
        target_rotation = (
            rotation_delta(applied[offset + 3 : offset + 6] * self.scales[offset + 3 : offset + 6])
            @ rotation
        )
        solution = self.solve_ik(
            side, clipped, target_rotation, **({} if data is None else {"data": data})
        )
        if solution is None:
            raise ContractError(f"{side} IK failed")
        indices = np.array([self.actuator_for_joint[int(i)] for i in self.arm_ids[side]])
        if np.any(np.abs(solution - previous_targets[indices]) > max_delta + 1e-6):
            raise ContractError(f"{side} joint rate limit")
        targets[indices] = solution
        opened = np.asarray(self.manifest[f"{side}_open_rad"])
        closed = np.asarray(self.manifest[f"{side}_closed_rad"])
        hand_indices = [self.actuator_for_joint[int(i)] for i in self.hand_ids[side]]
        previous = previous_targets[hand_indices]
        midpoint, slope = (opened + closed) / 2, (closed - opened) / 2
        lower, upper = -1.0, 1.0
        for center, derivative, prior in zip(midpoint, slope, previous, strict=True):
            if abs(derivative) < 1e-12:
                if abs(center - prior) > max_delta + 1e-6:
                    raise ContractError(f"{side} grasp has no rate-feasible synergy")
                continue
            ends = (
                (prior - max_delta - center) / derivative,
                (prior + max_delta - center) / derivative,
            )
            lower, upper = max(lower, min(ends)), min(upper, max(ends))
        if lower > upper:
            raise ContractError(f"{side} grasp has no rate-feasible synergy")
        grasp = float(np.clip(applied[12 + side_index], lower, upper))
        if grasp != float(applied[12 + side_index]):
            reasons.append(f"{side} grasp rate")
            applied[12 + side_index] = grasp
        hand_targets = opened + (grasp + 1) * 0.5 * (closed - opened)
        targets[hand_indices] = hand_targets
        return reasons

    def _snapshot_kinematics(self):
        """Robot-only kinematics consistent with the latest measured joint snapshot."""
        data = self.mj.MjData(self.model)
        data.qpos[self.sim.qadr] = self._projection_q
        self.mj.mj_forward(self.model, data)
        return data

    def project_candidates(self, requested):
        """Preview pose/hand targets using proprioception and commanded targets only.

        Backtrack each arm delta through 1, 1/2, ..., 1/64, 0. If none is
        feasible, exclude the sequence (zero delta is not a guaranteed hold).
        After step zero, assume measured joints reach accepted joint targets.
        This kinematic surrogate is replanned after every real observation;
        it is not a contact/velocity prediction or a safety guarantee.
        """
        validate_actions(requested, ndim=4)
        if requested.shape[0] != 1:
            raise ContractError("one embodiment projects batch size one")
        state = self.state()
        if not np.isclose(state.timestamps[0], self.sim.data.time, rtol=0, atol=1e-9):
            raise ContractError("projection requires a current unconsumed observation")
        count = len(self.sim.joint_names)
        if np.any(
            np.abs(state.values[0, count:]) > self.manifest["measured_joint_velocity_stop_rad_s"]
        ):
            raise ContractError("measured joint velocity limit exceeded")
        # Reset scratch non-robot coordinates to model defaults. No object pose,
        # contact, image, task evaluator, or future simulation step enters preview.
        data = self._snapshot_kinematics()
        initial_q = self._projection_q
        initial_targets = self._projection_targets
        actions = requested.copy()
        feasible = np.ones(requested.shape[:2], dtype=bool)
        for candidate in range(requested.shape[1]):
            self.mj.mj_resetData(self.model, data)
            data.qpos[self.sim.qadr] = initial_q
            targets = initial_targets.copy()
            self.mj.mj_forward(self.model, data)
            for step in range(requested.shape[2]):
                applied = actions[0, candidate, step].copy()
                for side_index, side in enumerate(("left", "right")):
                    offset = side_index * 6
                    original = applied[offset : offset + 6].copy()
                    for factor in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0):
                        trial = applied.copy()
                        trial[offset : offset + 6] = original * factor
                        trial_targets = targets.copy()
                        try:
                            self._prepare_side(trial, trial_targets, side_index, side, data=data)
                        except ContractError:
                            continue
                        applied, targets = trial, trial_targets
                        break
                    else:
                        feasible[0, candidate] = False
                        break
                if not feasible[0, candidate]:
                    break
                actions[0, candidate, step] = applied
                data.qpos[self.sim.qadr] = targets
                self.mj.mj_forward(self.model, data)
        return CandidateProjection(actions, feasible)

    def execute(self, normalized_action):
        validate_actions(normalized_action, ndim=1)
        requested = normalized_action.copy()

        def reject(reason):
            self.sim.stop(reason)
            return ExecutionResult(requested, None, "stopped", float(self.sim.data.time), reason)

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
        applied = requested.copy()
        targets = self.sim.targets.copy()
        reasons = []
        # mj_step can leave live site transforms behind the integrated qpos.
        # Preview and execution both derive targets from the observed robot q.
        data = self._snapshot_kinematics()
        try:
            for side_index, side in enumerate(("left", "right")):
                reasons.extend(self._prepare_side(applied, targets, side_index, side, data=data))
        except ContractError as error:
            return reject(str(error))
        ack = self.sim.send_joint_targets(
            targets.astype(np.float32),
            joint_names=self.sim.joint_names,
            deadline=float(self._observation.timestamps[0] + self.manifest["max_snapshot_age_s"]),
        )
        if ack["status"] != "applied":
            return reject(ack["reason"])
        self._grasp[:] = applied[12:]
        same = np.array_equal(applied, requested)
        return ExecutionResult(
            requested,
            applied,
            "applied" if same else "clipped",
            ack["timestamp"],
            "" if same else "; ".join(reasons),
        )

    def close(self):
        self.sim.close()

    def stop(self, reason):
        """Pause simulator actuation and require a fresh observation before resuming."""
        self.sim.stop(reason)
        self._observation = None
