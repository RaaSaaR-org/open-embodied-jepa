"""NON-LEARNED privileged simulator-rollout planning ceiling (TASK-044 diagnostic).

This module deliberately uses simulator truth, the complete live MuJoCo state, as
the forward model of the unchanged waypoint CEM planner. That is permitted only
because it measures a declared diagnostic ceiling ("what would the same
state-goal scaffold and planner achieve with perfect dynamics?"). Its outcomes
are never learned results.

Isolation rules:

* It is not registered in ``MODELS`` and cannot be selected by a model config.
* Constructing it requires an explicit ``acknowledge_privileged_ceiling=True``.
* The evaluator only builds it for the ``privileged_rollout`` mode, which the
  plan confines to development-stage runs that contain no learned mode.

Each candidate is scored by restoring a saved copy of the live MuJoCo state into
a separate, non-rendering twin simulator and executing the candidate's projected
actions through the unchanged ``G1Embodiment.execute`` (IK, rate limits, guards,
PD transport). The cost is the child state-goal model's own
``observed_distance``: the same TRAIN-normalized right-arm+hand metric that
measures waypoint progress. The twin never touches the live simulator.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np

from embodied_jepa import simulation as simulation_module
from embodied_jepa.contracts import (
    ContractError,
    RobotState,
    validate_actions,
)
from embodied_jepa.embodiment import G1Embodiment

PRIVILEGED_LABEL = "NON-LEARNED privileged MuJoCo-rollout planning ceiling"
PLANNING_DYNAMICS = "privileged_mujoco_rollout_v1"
# Finite penalty added to the last reached distance for any candidate step the twin
# could not execute (rejected by projection, the unchanged embodiment guards, or a
# MuJoCo instability). Normalized state-goal distances are O(0.01-1), so such a
# candidate is never preferred to an executable one.
REJECTED_ROLLOUT_COST = 1.0e3


@cache
def _blind_simulation_class(base):
    """Return the rollout-twin subclass of ``base``, built on first use.

    The twin is derived at construction time and the base is resolved through the
    ``embodied_jepa.simulation`` module attribute rather than a name bound at
    import time. Importing this module therefore never depends on import order:
    a test that replaces ``simulation.MuJoCoSimulation`` while some other module
    lazily imports this one no longer makes the class statement run against the
    replacement (which previously raised at import).
    """

    class _BlindSimulation(base):
        """Rollout twin: identical scene and transport, no rendering (images unused)."""

        def render(self, camera="onboard_rgb"):
            self._require_open()
            return np.zeros((self.height, self.width, 3), np.uint8)

    return _BlindSimulation


@dataclass(frozen=True)
class RolloutSnapshot:
    data: Any  # private MjData copy of the live simulator
    targets: np.ndarray
    body_pos: np.ndarray
    timestamp: float
    owner: object


@dataclass(frozen=True)
class RolloutGoal:
    state: np.ndarray  # raw right-arm+hand goal radians [1,14]
    owner: object


@dataclass(frozen=True)
class RolloutPrediction:
    states: np.ndarray  # full robot state rows after each executed step [1,K,H,D]
    timestamps: np.ndarray  # [1,K,H]
    valid: np.ndarray  # [1,K,H] step executed by the twin
    initial_state: np.ndarray  # snapshot state row [1,D]
    initial_timestamp: np.ndarray  # [1]
    owner: object


class PrivilegedRolloutModel:
    """Planner-facing adapter whose ``predict`` is an exact MuJoCo rollout."""

    backend = PLANNING_DYNAMICS
    label = PRIVILEGED_LABEL

    def __init__(self, metric_model, live_robot, *, acknowledge_privileged_ceiling=False):
        if acknowledge_privileged_ceiling is not True:
            raise ContractError(
                "privileged simulator rollouts are a declared non-learned diagnostic ceiling; "
                "construction requires acknowledge_privileged_ceiling=True"
            )
        for name in ("observed_distance", "capabilities"):
            if not hasattr(metric_model, name):
                raise ContractError(f"state-goal metric model lacks {name}")
        if not isinstance(live_robot, G1Embodiment):
            raise ContractError("privileged rollouts require the live G1 MuJoCo embodiment")
        self.metric = metric_model
        self.live = live_robot
        live_sim = live_robot.sim
        self.mj = live_sim.mj
        self.twin = G1Embodiment(
            _blind_simulation_class(simulation_module.MuJoCoSimulation)(
                object_kind=live_sim.object_kind,
                container_kind=live_sim.container_kind,
                width=live_sim.width,
                height=live_sim.height,
                control_dt=live_sim.control_dt,
                render=False,
            )
        )
        a, b = live_sim.model, self.twin.sim.model
        if (
            (a.nq, a.nv, a.nu, a.na, a.nbody, a.ngeom) != (b.nq, b.nv, b.nu, b.na, b.nbody, b.ngeom)
            or live_sim.joint_names != self.twin.sim.joint_names
            or a.opt.timestep != b.opt.timestep
            or live_sim.substeps != self.twin.sim.substeps
            or live_robot.manifest != self.twin.manifest
            or live_robot.state_schema != self.twin.state_schema
        ):
            raise ContractError("rollout twin differs structurally from the live simulator")
        self.state_schema = live_robot.state_schema
        self._owner = object()
        self._diagnostics = []
        self._previous_first_steps = []  # every candidate first step of the last search
        self._search_first_steps = []
        self._pending_parity = None

    @property
    def capabilities(self):
        return self.metric.capabilities

    def _live_state_row(self):
        sim = self.live.sim
        return np.concatenate((sim.data.qpos[sim.qadr], sim.data.qvel[sim.vadr])).astype(np.float32)

    def encode(self, observation, robot_state):
        """Save the live simulator state; the observation must be that exact state."""
        if not isinstance(robot_state, RobotState) or robot_state.values.shape[0] != 1:
            raise ContractError("privileged snapshot requires one RobotState row")
        sim = self.live.sim
        if float(robot_state.timestamps[0]) != float(sim.data.time) or not np.array_equal(
            robot_state.values[0], self._live_state_row()
        ):
            raise ContractError("privileged snapshot does not match the planning observation")
        data = self.mj.MjData(sim.model)
        self.mj.mj_copyData(data, sim.model, sim.data)
        current = robot_state.values[0]
        parity = None
        # Runtime conformance: the executed command was one of the previous search's
        # candidates, so the measured state must equal one simulated first step exactly.
        self._previous_first_steps, self._search_first_steps = self._search_first_steps, []
        if self._previous_first_steps:
            errors = [float(np.max(np.abs(row - current))) for row in self._previous_first_steps]
            parity = {
                "previous_search_first_step_exact_match": bool(min(errors) == 0.0),
                "previous_search_first_step_min_max_abs_error": min(errors),
            }
        self._pending_parity = parity
        return RolloutSnapshot(
            data,
            sim.targets.copy(),
            sim.model.body_pos.copy(),
            float(sim.data.time),
            self._owner,
        )

    def encode_goal(self, goal):
        if not isinstance(goal, Mapping) or "state" not in goal or set(goal) - {"state", "images"}:
            raise ContractError("privileged ceiling requires a state goal mapping")
        state = np.asarray(goal["state"])
        if (
            state.dtype.kind != "f"
            or state.ndim != 2
            or state.shape[0] != 1
            or not np.isfinite(state).all()
        ):
            raise ContractError("privileged ceiling goal must be finite float[1,D]")
        state = state.astype(np.float32, copy=True)
        state.setflags(write=False)
        return RolloutGoal(state, self._owner)

    def _restore(self, snapshot):
        twin = self.twin.sim
        twin.model.body_pos[:] = snapshot.body_pos
        self.mj.mj_copyData(twin.data, twin.model, snapshot.data)
        twin.targets[:] = snapshot.targets
        twin.stopped_reason = ""
        self.twin._observation = None
        self.twin._grasp[:] = -1

    def predict(self, z, actions):
        """Execute each candidate in the twin, step by step, as the live loop would.

        Every step passes through the twin's own mandatory ``project_candidates``
        (the live loop projects every command against the measured state) and the
        unchanged ``execute``. A step that projection or execution rejects ends that
        candidate; its remaining steps are marked invalid.
        """
        if not isinstance(z, RolloutSnapshot) or z.owner is not self._owner:
            raise ContractError("snapshot belongs to another privileged rollout instance")
        validate_actions(actions, ndim=4)
        if actions.shape[0] != 1:
            raise ContractError("privileged rollouts support batch size one")
        _, candidates, horizon, _ = actions.shape
        dimension = self.state_schema.dimension
        states = np.zeros((1, candidates, horizon, dimension), np.float32)
        stamps = np.zeros((1, candidates, horizon), np.float64)
        valid = np.zeros((1, candidates, horizon), bool)
        reasons, reprojected = {}, 0
        start = time.perf_counter()
        initial = None
        for k in range(candidates):
            self._restore(z)
            for h in range(horizon):
                observation = self.twin.observe()
                if initial is None:
                    initial = observation
                if h:
                    states[0, k, h - 1] = observation.robot_state[0]
                    stamps[0, k, h - 1] = observation.timestamps[0]
                reason = None
                try:
                    projection = self.twin.project_candidates(actions[0, k, h][None, None, None])
                    if not projection.feasible[0, 0]:
                        reason = "no feasible projection"
                    else:
                        command = projection.actions[0, 0, 0]
                        reprojected += int(not np.array_equal(command, actions[0, k, h]))
                        result = self.twin.execute(command)
                        if result.applied_action is None:
                            reason = result.reason
                except (ContractError, RuntimeError) as error:
                    reason = f"{type(error).__name__}: {error}"
                if reason is not None:
                    reasons[reason] = reasons.get(reason, 0) + 1
                    break
                valid[0, k, h] = True
            else:
                final = self.twin.observe()
                states[0, k, horizon - 1] = final.robot_state[0]
                stamps[0, k, horizon - 1] = final.timestamps[0]
        self._search_first_steps.extend(
            states[0, k, 0].copy() for k in range(candidates) if valid[0, k, 0]
        )
        diagnostic = {
            "planning_dynamics": PLANNING_DYNAMICS,
            "candidates": int(candidates),
            "horizon": int(horizon),
            "rejected_candidates": int((~valid[0, :, -1]).sum()),
            "rejection_reasons": reasons,
            "reprojected_steps": reprojected,
            "rollout_seconds": time.perf_counter() - start,
        }
        if self._pending_parity is not None:
            diagnostic |= self._pending_parity
            self._pending_parity = None
        self._diagnostics.append(diagnostic)
        for array in (states, stamps, valid):
            array.setflags(write=False)
        return RolloutPrediction(
            states,
            stamps,
            valid,
            initial.robot_state.copy(),
            initial.timestamps.copy(),
            self._owner,
        )

    def distance(self, prediction, goal):
        """State-goal cost per step; a rejected step costs the penalty plus the distance
        of the last state its candidate actually reached (so rejected sequences rank
        after every executable one, and among themselves by real progress)."""
        if not isinstance(prediction, RolloutPrediction) or prediction.owner is not self._owner:
            raise ContractError("prediction belongs to another privileged rollout instance")
        if not isinstance(goal, RolloutGoal) or goal.owner is not self._owner:
            raise ContractError("goal belongs to another privileged rollout instance")
        valid = prediction.valid
        rows = np.concatenate((prediction.initial_state, prediction.states[valid]))
        times = np.concatenate((prediction.initial_timestamp, prediction.timestamps[valid]))
        state = RobotState(
            np.ascontiguousarray(rows), np.ones(rows.shape, bool), times, self.state_schema
        )
        values = np.asarray(
            self.metric.observed_distance(state, np.repeat(goal.state, len(rows), 0)),
            dtype=np.float64,
        )
        if values.shape != (len(rows),) or not np.isfinite(values).all() or (values < 0).any():
            raise ContractError("invalid privileged rollout state-goal distance")
        reached = np.zeros(valid.shape, np.float64)
        reached[valid] = values[1:]
        costs = np.empty(valid.shape, np.float32)
        for k in range(valid.shape[1]):
            last = values[0]
            for h in range(valid.shape[2]):
                if valid[0, k, h]:
                    last = reached[0, k, h]
                    costs[0, k, h] = last
                else:
                    costs[0, k, h] = REJECTED_ROLLOUT_COST + last
        return costs

    def observed_distance(self, robot_state, goal_state):
        return self.metric.observed_distance(robot_state, goal_state)

    def pop_diagnostics(self):
        result, self._diagnostics = self._diagnostics, []
        return result

    def train_step(self, batch):
        raise ContractError("privileged simulator rollouts are not a trainable model")

    def close(self):
        self.twin.close()
