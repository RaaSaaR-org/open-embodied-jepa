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
