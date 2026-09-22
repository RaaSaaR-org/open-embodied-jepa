"""Privileged scripted collector/control baseline, isolated from learned planning.

This policy consumes simulator object/container truth once when constructed.
Never provide it, its targets, or its truth inputs to a world-model planner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embodied_jepa.contracts import ContractError
from embodied_jepa.embodiment import rotation_delta


@dataclass(frozen=True)
class OraclePhase:
    name: str
    target_base: np.ndarray
    grasp: float
    commands: int


class OracleManipulationPolicy:
    """Fixed-budget right-hand top-down grasp/transfer/release collector policy.

    Call action(robot), execute the returned command, then advance(result).
    Failed requests stop the policy; successful clipped commands count normally.
    """

    def __init__(self, initial_truth):
        if initial_truth.get("position_frame") != "world":
            raise ContractError("oracle initialization requires declared world-frame task truth")
        base = np.asarray(initial_truth["base_position_world"])
        transform = np.asarray(initial_truth["base_rotation_world"]).T
        obj = transform @ (np.asarray(initial_truth["object_position"]) - base)
        container = transform @ (np.asarray(initial_truth["plate_position"]) - base)
        if not np.allclose(transform, np.eye(3), atol=1e-7):
            raise ContractError("this bounded oracle assumes the fixed upright pelvis scene")
        high = obj[2] + 0.21
        lower = (
            initial_truth["container_surface_z"]
            + initial_truth["object_support_height"]
            + 0.115
            - base[2]
        )
        transfer = np.array([container[0] - 0.03, container[1], high])
        place = np.array([container[0] - 0.03, container[1], lower])
        self.phases = (
            OraclePhase("orient", obj + [-0.03, 0, 0.13], -1.0, 130),
            OraclePhase("descend", obj + [-0.03, 0, 0.052], -1.0, 80),
            OraclePhase("close", obj + [-0.03, 0, 0.052], 1.0, 45),
            OraclePhase("lift", obj + [-0.03, 0, 0.21], 1.0, 150),
            OraclePhase("transfer", transfer, 1.0, 160),
            OraclePhase("lower", place, 1.0, 100),
            OraclePhase("release", place, -1.0, 60),
            OraclePhase("retreat", transfer, -1.0, 80),
        )
        self.rotation = rotation_delta([-np.pi / 2, 0, 0])
        self.step_count = 0
        self.phase_index = 0
        self.phase_step = 0
        self.failure = ""

    @property
    def done(self):
        return bool(self.failure) or self.phase_index == len(self.phases)

    @property
    def phase(self):
        return "done" if self.done else self.phases[self.phase_index].name

    @property
    def max_steps(self):
        return sum(phase.commands for phase in self.phases)

    def action(self, robot):
        if self.done:
            raise ContractError("scripted policy has finished")
        phase = self.phases[self.phase_index]
        position, rotation = robot.ee_pose("right")
        rotation_error = 0.5 * sum(np.cross(rotation[:, i], self.rotation[:, i]) for i in range(3))
        action = np.zeros(14, dtype=np.float32)
        action[6:9] = np.clip(
            (phase.target_base - position) / robot.manifest["translation_per_step_m"], -0.4, 0.4
        )
        action[9:12] = np.clip(rotation_error / robot.manifest["rotation_per_step_rad"], -0.5, 0.5)
        action[12:] = [-1.0, phase.grasp]
        return action

    def advance(self, result):
        if self.done:
            raise ContractError("cannot advance a finished scripted policy")
        if result.applied_action is None:
            self.failure = result.reason or "command rejected"
            return
        self.step_count += 1
        self.phase_step += 1
        if self.phase_step == self.phases[self.phase_index].commands:
            self.phase_step = 0
            self.phase_index += 1


class SlowOracleManipulationPolicy(OracleManipulationPolicy):
    """Exploratory contact controller with bounded motion and gradual closure.

    This changes only requested actions and phase duration. Runtime PD gains,
    actuator limits, collision physics, and velocity-stop thresholds stay fixed.
    Closure is a normalized synergy target, not measured contact force.
    """

    def __init__(
        self,
        initial_truth,
        *,
        closure=0.5,
        translation_limit=0.2,
        rotation_limit=0.25,
        grasp_ramp=0.03,
        duration_scale=2,
    ):
        super().__init__(initial_truth)
        if not np.isfinite(closure) or not -1 <= closure <= 1:
            raise ContractError("closure must be finite and normalized")
        for value in (translation_limit, rotation_limit, grasp_ramp):
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ContractError("motion limits and grasp ramp must lie in (0,1]")
        if type(duration_scale) is not int or not 1 <= duration_scale <= 4:
            raise ContractError("duration_scale must be an integer from 1 to 4")
        self.closure = closure
        self.translation_limit = translation_limit
        self.rotation_limit = rotation_limit
        self.grasp_ramp = grasp_ramp
        self.accepted_grasp = -1.0
        self.phases = tuple(
            OraclePhase(p.name, p.target_base.copy(), p.grasp, p.commands * duration_scale)
            for p in self.phases
        )

    def action(self, robot):
        action = super().action(robot)
        action[6:9] = np.clip(action[6:9], -self.translation_limit, self.translation_limit)
        action[9:12] = np.clip(action[9:12], -self.rotation_limit, self.rotation_limit)
        target = self.closure if self.phases[self.phase_index].grasp > 0 else -1.0
        action[13] = np.clip(
            target, self.accepted_grasp - self.grasp_ramp, self.accepted_grasp + self.grasp_ramp
        )
        return action

    def advance(self, result):
        super().advance(result)
        if result.applied_action is not None:
            self.accepted_grasp = float(result.applied_action[13])


class EarlyReleaseOracleManipulationPolicy(OracleManipulationPolicy):
    """Release over the container before sustained full-closure contact slips.

    The original reach, close and lift requests remain unchanged. The transfer
    dwell is shorter, followed by gradual opening at the high transfer pose.
    This remains a privileged initial-truth controller, not learned planning.
    """

    def __init__(self, initial_truth, *, opening_ramp=0.08):
        super().__init__(initial_truth)
        if not np.isfinite(opening_ramp) or not 0 < opening_ramp <= 1:
            raise ContractError("opening_ramp must lie in (0,1]")
        self.opening_ramp = opening_ramp
        self.accepted_grasp = -1.0
        transfer, lower, retreat = self.phases[4], self.phases[5], self.phases[7]
        self.phases = (
            *self.phases[:4],
            OraclePhase("transfer", transfer.target_base.copy(), 1.0, 60),
            OraclePhase("release_high", transfer.target_base.copy(), -1.0, 100),
            OraclePhase("lower_open", lower.target_base.copy(), -1.0, 100),
            OraclePhase("retreat", retreat.target_base.copy(), -1.0, 80),
        )

    def action(self, robot):
        action = super().action(robot)
        if self.phase_index >= 5:
            action[13] = max(-1.0, self.accepted_grasp - self.opening_ramp)
        return action

    def advance(self, result):
        super().advance(result)
        if result.applied_action is not None:
            self.accepted_grasp = float(result.applied_action[13])


def apple_collector_policy(initial_truth, *, transfer_x_shift=-0.035, palm_x_offset=0.015):
    """The privileged Apple->Plate collector policy exactly as ``scripts/collect_apple.py``
    configured it for ``data/apple-task-v1`` (early release, palm and transfer shifts).

    A scripted initial-truth controller: a feasibility reference, never a learned result.
    """
    from dataclasses import replace

    policy = EarlyReleaseOracleManipulationPolicy(initial_truth, opening_ramp=0.08)
    phases = []
    for index, phase in enumerate(policy.phases):
        target = phase.target_base.copy()
        target[0] += palm_x_offset + (transfer_x_shift if index >= 4 else 0)
        phases.append(replace(phase, target_base=target))
    policy.phases = tuple(phases)
    return policy
