"""NON-LEARNED object-aware privileged MuJoCo-rollout ceiling (TASK-047 diagnostic).

Every earlier privileged ceiling (TASK-044/045/046) scored candidates with a cost on
the robot's own arm+hand joints only; none knew where the apple was. This module is
the first *object-aware* ceiling: it plans with exact MuJoCo rollouts (the unchanged
``PrivilegedRolloutModel`` twin machinery) and scores each candidate with a phase
cost computed from simulator object state -- the apple pose relative to the palm,
hand contact, and the apple pose relative to the plate.

Simulator truth is used here on purpose and only here: in the rollout features, the
phase costs and the phase transitions of this explicitly labelled diagnostic. It is
never a model input, never registered in ``MODELS``, and requires
``acknowledge_privileged_ceiling=True``. Its outcomes are never learned results.

Phases (the right-grasp command is scheduled per phase through the CEM bounds; the
arm motion is planned):

``approach``  palm to a pre-grasp point above the live apple, hand open.
``descend``   palm down to the grasp offset around the live apple, hand open,
              penalizing apple displacement.
``close``     hold the grasp pose while the hand closes for a fixed command count.
``lift``      raise the *apple* above its start height while in hand contact.
``transport`` move the *apple* over the plate at height.
``lower``     lower the *apple* to just above the plate surface.
``release``   hold the palm while the hand opens for a fixed command count.
``retreat``   raise the open palm.

Geometric offsets are the privileged collector's own (``scripts/collect_apple.py``:
palm 1.5 cm behind the apple centre, pre-grasp 13 cm and grasp 5.2 cm above it).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from embodied_jepa.constraints import CandidateProjection
from embodied_jepa.contracts import ContractError, ExecutionResult, Observation, validate_actions
from embodied_jepa.embodiment import rotation_delta
from embodied_jepa.privileged_rollout import (
    REJECTED_ROLLOUT_COST,
    PrivilegedRolloutModel,
    RolloutSnapshot,
)
from embodied_jepa.waypoint_planning import WaypointDecision

OBJECT_CEILING_LABEL = (
    "NON-LEARNED object-aware privileged MuJoCo-rollout ceiling (simulator object state in "
    "cost and phase transitions); not a learned result"
)
PLANNING_DYNAMICS = "privileged_mujoco_rollout_object_v1"
PHASES = ("approach", "descend", "close", "lift", "transport", "lower", "release", "retreat")
CLOSED_PHASES = ("close", "lift", "transport", "lower")
BLOCKABLE_PHASES = ("descend", "transport", "lower")
# Embodiment guard refusals that are physical stops, not software failures (as TASK-046).
GUARD_REFUSALS = ("measured joint velocity limit exceeded",)
# Collector's top-down palm orientation (scripted.OracleManipulationPolicy).
TOP_DOWN = rotation_delta([-np.pi / 2, 0, 0])


@dataclass(frozen=True)
class ObjectCeilingConfig:
    horizon: int = 6
    candidates: int = 24
    iterations: int = 2
    proposal_std: float = 0.3
    minimum_std: float = 0.05
    max_steps: int = 1000
    seed: int = 0
    arm_bound: float = 0.5  # normalized right-arm delta bound, as every earlier plan
    palm_offset: tuple[float, float] = (-0.015, 0.0)  # palm xy relative to apple centre
    approach_height_m: float = 0.13
    grasp_height_m: float = 0.052
    approach_tolerance_m: float = 0.01
    grasp_tolerance_m: float = 0.008
    # A descend/transport/lower phase also ends when it is *blocked*: its progress
    # measure (palm height, apple-plate xy distance, apple height) improved by less than
    # block_progress_m over the last block_commands commands, while near its goal
    # (descend: palm within block_xy_m of the grasp point in xy; transport: apple within
    # the scorer's plate radius; lower: always). The thumb meets the table before the
    # grasp height in every collector demonstration, so the descent is always blocked.
    block_commands: int = 10
    block_progress_m: float = 0.002
    block_xy_m: float = 0.01
    plate_radius_m: float = 0.04
    rotation_tolerance_rad: float = 0.1
    close_commands: int = 45
    lift_target_m: float = 0.15  # apple rise the lift cost asks for
    lift_done_m: float = 0.10  # apple rise (with contact) that ends the lift phase
    carry_height_m: float = 0.12  # apple rise kept during transport
    transport_tolerance_m: float = 0.015
    lower_clearance_m: float = 0.015  # apple centre above its resting height on the plate
    lower_tolerance_m: float = 0.01
    release_commands: int = 40
    retreat_height_m: float = 0.08
    phase_stall_commands: int = 200
    rotation_weight: float = 0.2
    disturbance_weight: float = 1.0
    slip_weight: float = 0.25
    contact_penalty: float = 0.05

    def __post_init__(self):
        for name in (
            "horizon",
            "candidates",
            "iterations",
            "max_steps",
            "close_commands",
            "release_commands",
            "phase_stall_commands",
            "block_commands",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ContractError(f"{name} must be a positive integer")
        if self.candidates < 3:
            raise ContractError("CEM needs hold, warm and at least one sampled candidate")
        if type(self.seed) is not int or self.seed < 0:
            raise ContractError("seed must be a nonnegative integer")
        for name, value in asdict(self).items():
            if isinstance(value, float) and (not np.isfinite(value) or value < 0):
                raise ContractError(f"{name} must be finite and nonnegative")
        if not 0 < self.arm_bound <= 1 or not 0 < self.minimum_std <= self.proposal_std:
            raise ContractError("invalid arm bound or proposal schedule")


@dataclass(frozen=True)
class ObjectRollout:
    """Per-step simulator features of every candidate rollout (world frame)."""

    palm: np.ndarray  # [K,H,3] right palm site position
    rotation_error: np.ndarray  # [K,H] top-down orientation error (rad)
    apple: np.ndarray  # [K,H,3]
    contact: np.ndarray  # [K,H] bool hand/wrist-apple contact
    dropped: np.ndarray  # [K,H] bool
    valid: np.ndarray  # [K,H] step executed by the twin
    owner: object


def rotation_error(rotation):
    """Geodesic angle (rad) between the palm orientation and ``TOP_DOWN``.

    The angle, unlike the collector's small-angle cross-product error, has no
    stationary point at 90 degrees (where the initial palm starts)."""
    cosine = (np.trace(np.asarray(rotation).T @ TOP_DOWN) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def features(robot):
    """Privileged world-frame features of ``robot``'s current simulator state."""
    truth = robot.sim.task_truth()
    base = np.asarray(truth["base_position_world"])
    rotation = np.asarray(truth["base_rotation_world"])
    position, palm_rotation = robot.ee_pose("right")
    return {
        "palm": base + rotation @ position,
        "rotation_error": rotation_error(palm_rotation),
        "apple": np.asarray(truth["object_position"], np.float64),
        "plate": np.asarray(truth["plate_position"], np.float64),
        "contact": bool(truth["hand_contact"]),
        "dropped": bool(truth["dropped"]),
        "rest_z": float(truth["container_surface_z"] + truth["object_support_height"]),
    }


class ObjectRolloutModel(PrivilegedRolloutModel):
    """Exact twin rollouts that record object-aware features per step.

    Reuses the TASK-044 twin, snapshot, restore and robot-state parity check; the
    parity check is extended to the complete MuJoCo state (apple included).
    """

    backend = PLANNING_DYNAMICS
    label = OBJECT_CEILING_LABEL

    def __init__(self, metric_model, live_robot, *, acknowledge_privileged_ceiling=False):
        super().__init__(
            metric_model,
            live_robot,
            acknowledge_privileged_ceiling=acknowledge_privileged_ceiling,
        )
        self._previous_full = []
        self._search_full = []

    def _full_state(self, sim):
        return np.concatenate((sim.data.qpos, sim.data.qvel)).copy()

    def encode(self, observation, robot_state):
        snapshot = super().encode(observation, robot_state)
        self._previous_full, self._search_full = self._search_full, []
        if self._previous_full and self._pending_parity is not None:
            current = self._full_state(self.live.sim)
            errors = [float(np.max(np.abs(row - current))) for row in self._previous_full]
            self._pending_parity |= {
                "previous_search_first_step_full_state_exact_match": bool(min(errors) == 0.0),
                "previous_search_first_step_full_state_min_max_abs_error": min(errors),
            }
        return snapshot

    def rollout(self, z, actions):
        """Execute each candidate ``[1,K,H,14]`` in the twin as the live loop would."""
        if not isinstance(z, RolloutSnapshot) or z.owner is not self._owner:
            raise ContractError("snapshot belongs to another privileged rollout instance")
        validate_actions(actions, ndim=4)
        if actions.shape[0] != 1:
            raise ContractError("privileged rollouts support batch size one")
        _, candidates, horizon, _ = actions.shape
        palm = np.zeros((candidates, horizon, 3))
        rotation = np.zeros((candidates, horizon))
        apple = np.zeros((candidates, horizon, 3))
        contact = np.zeros((candidates, horizon), bool)
        dropped = np.zeros((candidates, horizon), bool)
        valid = np.zeros((candidates, horizon), bool)
        reasons, reprojected = {}, 0
        start = time.perf_counter()
        for k in range(candidates):
            self._restore(z)
            for h in range(horizon):
                self.twin.observe()
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
                valid[k, h] = True
                row = features(self.twin)
                palm[k, h], rotation[k, h], apple[k, h] = (
                    row["palm"],
                    row["rotation_error"],
                    row["apple"],
                )
                contact[k, h], dropped[k, h] = row["contact"], row["dropped"]
                if h == 0:
                    sim = self.twin.sim
                    self._search_first_steps.append(
                        np.concatenate((sim.data.qpos[sim.qadr], sim.data.qvel[sim.vadr])).astype(
                            np.float32
                        )
                    )
                    self._search_full.append(self._full_state(sim))
        diagnostic = {
            "planning_dynamics": PLANNING_DYNAMICS,
            "candidates": int(candidates),
            "horizon": int(horizon),
            "rejected_candidates": int((~valid[:, -1]).sum()),
            "rejection_reasons": reasons,
            "reprojected_steps": reprojected,
            "rollout_seconds": time.perf_counter() - start,
        }
        if self._pending_parity is not None:
            diagnostic |= self._pending_parity
            self._pending_parity = None
        self._diagnostics.append(diagnostic)
        return ObjectRollout(palm, rotation, apple, contact, dropped, valid, self._owner)

    def predict(self, z, actions):
        raise ContractError("the object-aware ceiling scores rollouts through rollout()")

    def distance(self, prediction, goal):
        raise ContractError("the object-aware ceiling scores rollouts through rollout()")


@dataclass
class _PhaseState:
    name: str = "approach"
    commands: int = 0
    anchor: dict[str, Any] = field(default_factory=dict)


class ObjectCeilingController:
    """Call step(observation, projector), execute once, then acknowledge(result).

    NON-LEARNED: reads the live simulator's object state for its phase machine and
    scores candidates with exact twin rollouts under the phase cost.
    """

    def __init__(
        self,
        rollout_model: ObjectRolloutModel,
        live_robot,
        config: ObjectCeilingConfig,
        *,
        acknowledge_privileged_ceiling=False,
    ):
        if acknowledge_privileged_ceiling is not True:
            raise ContractError(
                "the object-aware ceiling reads simulator object state; construction requires "
                "acknowledge_privileged_ceiling=True"
            )
        if not isinstance(rollout_model, ObjectRolloutModel):
            raise ContractError("object-aware ceiling requires exact object rollouts")
        if not isinstance(config, ObjectCeilingConfig):
            raise ContractError("object-aware ceiling requires ObjectCeilingConfig")
        self.rollouts = rollout_model
        self.robot = live_robot
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.start = features(live_robot)
        self.phase = _PhaseState()
        self.phase_log = []  # (phase, first command index)
        self.steps = 0
        self.warm = None
        self.pending = None
        self.last_timestamp = -1.0
        self.last_execution_timestamp = -1.0
        self.termination_reason = None
        self._enter("approach", self.start)

    # ----- phase machine -------------------------------------------------------------
    def _grasp_target(self, apple, height):
        dx, dy = self.config.palm_offset
        return apple + np.array([dx, dy, height])

    def _enter(self, name, live):
        anchor = {"apple": live["apple"].copy(), "palm": live["palm"].copy()}
        if name in BLOCKABLE_PHASES:
            anchor["progress"] = []
        if name == "close":
            anchor["target"] = self._grasp_target(live["apple"], self.config.grasp_height_m)
        if name in ("lift", "transport", "lower"):
            anchor["relative"] = live["apple"] - live["palm"]
        if name == "lower":
            anchor["target_z"] = live["rest_z"] + self.config.lower_clearance_m
        if name == "retreat":
            anchor["target"] = live["palm"] + np.array([0, 0, self.config.retreat_height_m])
        previous = self.phase.anchor.get("ended")
        self.phase = _PhaseState(name, 0, anchor)
        self.phase_log.append(
            {"phase": name, "command": self.steps}
            | ({"previous_ended": previous} if previous else {})
        )
        self.warm = None  # a phase change changes the cost and the grasp schedule

    def _advance(self, live):
        """Apply every satisfied transition on the live state (possibly several)."""
        cfg = self.config
        while True:
            name, commands = self.phase.name, self.phase.commands
            rise = live["apple"][2] - self.start["apple"][2]
            if name == "approach":
                target = self._grasp_target(live["apple"], cfg.approach_height_m)
                done = (
                    np.linalg.norm(live["palm"] - target) < cfg.approach_tolerance_m
                    and live["rotation_error"] < cfg.rotation_tolerance_rad
                )
            elif name == "descend":
                target = self._grasp_target(live["apple"], cfg.grasp_height_m)
                done = np.linalg.norm(live["palm"] - target) < cfg.grasp_tolerance_m
                near = np.linalg.norm(live["palm"][:2] - target[:2]) < cfg.block_xy_m
            elif name == "close":
                done = commands >= cfg.close_commands
            elif name == "lift":
                done = rise >= cfg.lift_done_m and live["contact"]
            elif name == "transport":
                offset = np.linalg.norm(live["apple"][:2] - live["plate"][:2])
                done = offset < cfg.transport_tolerance_m
                near = offset < cfg.plate_radius_m
            elif name == "lower":
                done = abs(live["apple"][2] - self.phase.anchor["target_z"]) < cfg.lower_tolerance_m
                near = True
            elif name == "release":
                done = commands >= cfg.release_commands
            else:
                done = False
            if name in BLOCKABLE_PHASES and not done and near and self._blocked():
                self.phase.anchor["ended"] = "blocked"
            elif done:
                if name in BLOCKABLE_PHASES:
                    self.phase.anchor["ended"] = "reached"
            else:
                return
            self._enter(PHASES[PHASES.index(name) + 1], live)

    def _progress_measure(self, live):
        """Decreasing progress measure of a blockable phase (meters)."""
        if self.phase.name == "descend":
            return float(live["palm"][2])
        if self.phase.name == "transport":
            return float(np.linalg.norm(live["apple"][:2] - live["plate"][:2]))
        return float(live["apple"][2])

    def _blocked(self):
        cfg, history = self.config, self.phase.anchor["progress"]
        return (
            len(history) > cfg.block_commands
            and history[-1 - cfg.block_commands] - history[-1] < cfg.block_progress_m
        )

    def phase_costs(self, rollout: ObjectRollout, live):
        """Per-step phase cost ``[K,H]`` from rollout features (meters-equivalent)."""
        cfg, anchor, name = self.config, self.phase.anchor, self.phase.name
        palm, apple = rollout.palm, rollout.apple
        cost = cfg.rotation_weight * rollout.rotation_error
        if name == "approach":
            cost = cost + np.linalg.norm(
                palm - self._grasp_target(apple, cfg.approach_height_m), axis=-1
            )
        elif name == "descend":
            cost = cost + np.linalg.norm(
                palm - self._grasp_target(apple, cfg.grasp_height_m), axis=-1
            )
            cost = cost + cfg.disturbance_weight * np.linalg.norm(
                apple[..., :2] - anchor["apple"][:2], axis=-1
            )
        elif name == "close":
            cost = cost + np.linalg.norm(palm - anchor["target"], axis=-1)
            cost = cost + cfg.disturbance_weight * np.linalg.norm(
                apple[..., :2] - anchor["apple"][:2], axis=-1
            )
        elif name in ("lift", "transport", "lower"):
            slip = np.linalg.norm((apple - palm) - anchor["relative"], axis=-1)
            cost = cost + cfg.slip_weight * slip + cfg.contact_penalty * (~rollout.contact)
            start_z = self.start["apple"][2]
            if name == "lift":
                cost = cost + np.maximum(0.0, start_z + cfg.lift_target_m - apple[..., 2])
                cost = cost + np.linalg.norm(palm[..., :2] - anchor["palm"][:2], axis=-1)
            elif name == "transport":
                cost = cost + np.linalg.norm(apple[..., :2] - live["plate"][:2], axis=-1)
                cost = cost + np.maximum(0.0, start_z + cfg.carry_height_m - apple[..., 2])
            else:
                cost = cost + 2.0 * np.linalg.norm(apple[..., :2] - live["plate"][:2], axis=-1)
                cost = cost + np.abs(apple[..., 2] - anchor["target_z"])
        elif name == "release":
            cost = cost + np.linalg.norm(palm - anchor["palm"], axis=-1)
        else:
            cost = cost + np.linalg.norm(palm - anchor["target"], axis=-1)
        return cost

    def candidate_costs(self, rollout, live):
        """Mean phase cost over the horizon; a rejected step costs the penalty plus the
        cost of the last state its candidate reached (TASK-044 convention)."""
        per_step = self.phase_costs(rollout, live)
        current = float(self.phase_costs(_live_rollout(live), live)[0, 0])
        costs = np.empty(rollout.valid.shape[0])
        for k in range(len(costs)):
            last, total = current, 0.0
            for h in range(rollout.valid.shape[1]):
                if rollout.valid[k, h]:
                    last = float(per_step[k, h])
                    total += last
                else:
                    total += REJECTED_ROLLOUT_COST + last
            costs[k] = total / rollout.valid.shape[1]
        return costs

    def _bounds(self):
        cfg = self.config
        lower = np.zeros(14, np.float32)
        upper = np.zeros(14, np.float32)
        lower[6:12], upper[6:12] = -cfg.arm_bound, cfg.arm_bound
        grasp = 1.0 if self.phase.name in CLOSED_PHASES else -1.0
        lower[12:] = upper[12:] = (-1.0, grasp)
        return lower, upper

    # ----- control loop --------------------------------------------------------------
    def step(self, observation: Observation, projector) -> WaypointDecision:
        cfg = self.config
        if self.pending is not None:
            raise ContractError("acknowledge the previous execution before replanning")
        if self.termination_reason is not None:
            return WaypointDecision(None, self.summary(), self.termination_reason)
        if not isinstance(observation, Observation) or observation.timestamps.shape != (1,):
            raise ContractError("object ceiling requires one synchronized observation")
        timestamp = float(observation.timestamps[0])
        if timestamp <= self.last_timestamp or timestamp < self.last_execution_timestamp:
            raise ContractError("object ceiling requires a fresh increasing observation")
        if not callable(projector):
            raise ContractError("candidate projection is mandatory")
        if self.steps >= cfg.max_steps:
            return self._stop("step_limit")
        live = features(self.robot)
        if live["dropped"]:
            return self._stop("object_dropped")
        if self.phase.name in BLOCKABLE_PHASES:
            self.phase.anchor["progress"].append(self._progress_measure(live))
        self._advance(live)
        if self.phase.commands >= cfg.phase_stall_commands:
            return self._stop("phase_stall")
        start = time.perf_counter()
        lower, upper = self._bounds()
        hold = np.clip(np.zeros((cfg.horizon, 14), np.float32), lower, upper)
        mean = hold.copy() if self.warm is None else np.clip(self.warm, lower, upper)
        snapshot = self.rollouts.encode(observation.images, observation.state)
        best, rounds, sigma = None, [], cfg.proposal_std
        for iteration in range(cfg.iterations):
            requested = np.clip(
                mean[None] + self.rng.normal(0, sigma, (cfg.candidates, cfg.horizon, 14)),
                lower,
                upper,
            ).astype(np.float32)
            requested[0], requested[1] = hold, mean
            try:
                projection = projector(requested[None])
            except ContractError as error:
                if str(error) not in GUARD_REFUSALS:
                    raise
                return self._stop("guard_refused", {"guard_reason": str(error)})
            if not isinstance(projection, CandidateProjection) or projection.actions.shape != (
                1,
                *requested.shape,
            ):
                raise ContractError("projector changed candidate contract")
            feasible = projection.feasible[0]
            indices = np.flatnonzero(feasible)
            if not len(indices):
                raise ContractError("no feasible object-ceiling candidate sequence")
            # Only feasible candidates are rolled out (and can become parity matches).
            rollout = self.rollouts.rollout(snapshot, projection.actions[:, indices])
            costs = np.full(cfg.candidates, np.inf)
            costs[indices] = self.candidate_costs(rollout, live)
            winner = int(indices[np.argmin(costs[indices])])  # ties: lowest index
            if best is None or costs[winner] < best["cost"]:
                best = {
                    "cost": float(costs[winner]),
                    "requested": requested[winner].copy(),
                    "projected": projection.actions[0, winner].copy(),
                    "round": iteration,
                    "index": winner,
                }
            mean = best["requested"].copy()
            finite = costs[indices]
            rounds.append(
                {
                    "iteration": iteration,
                    "feasible_candidates": int(len(indices)),
                    "selected_index": winner,
                    "min_cost": float(finite.min()),
                    "cost_spread": float(np.ptp(finite)),
                    "proposal_std": sigma,
                }
            )
            sigma = max(cfg.minimum_std, sigma * 0.5)
        action = best["projected"][0].copy()
        action.setflags(write=False)
        trace = {
            "control": "object_ceiling_non_learned",
            "phase": self.phase.name,
            "phase_commands": self.phase.commands,
            "plan_step": self.steps,
            "observation_timestamp": timestamp,
            "live_palm": live["palm"].tolist(),
            "live_apple": live["apple"].tolist(),
            "live_contact": live["contact"],
            "live_rotation_error": live["rotation_error"],
            "selected_cost": best["cost"],
            "selected_round": best["round"],
            "selected_index": best["index"],
            "sampled_action": best["requested"][0].tolist(),
            "projected_action": action.tolist(),
            "rounds": rounds,
            "candidate_evaluations": cfg.candidates * cfg.iterations,
            "planning_seconds": time.perf_counter() - start,
        }
        self.last_timestamp = timestamp
        self.pending = (action, best["projected"], trace)
        return WaypointDecision(action, trace)

    def _stop(self, reason, extra=None):
        self.termination_reason = reason
        return WaypointDecision(None, self.summary() | (extra or {}), reason)

    def acknowledge(self, result: ExecutionResult) -> dict[str, Any]:
        if self.pending is None:
            raise ContractError("no object-ceiling command awaits execution acknowledgement")
        action, sequence, trace = self.pending
        if not isinstance(result, ExecutionResult) or not np.array_equal(
            result.requested_action, action
        ):
            raise ContractError("execution acknowledgement does not match projected command")
        if result.timestamp < self.last_timestamp or (
            result.applied_action is not None and result.timestamp <= self.last_timestamp
        ):
            raise ContractError("execution acknowledgement did not advance the observation clock")
        trace.update(
            status=result.status,
            reason=result.reason,
            applied_action=None
            if result.applied_action is None
            else result.applied_action.tolist(),
            execution_timestamp=result.timestamp,
        )
        self.pending = None
        self.steps += 1
        self.phase.commands += 1
        self.last_execution_timestamp = result.timestamp
        if result.applied_action is None:
            self.termination_reason = "execution_rejected"
        else:
            self.warm = np.concatenate((sequence[1:], sequence[-1:])).copy()
        return trace

    def summary(self):
        return {
            "control_label": OBJECT_CEILING_LABEL,
            "phase": self.phase.name,
            "phase_log": list(self.phase_log),
            "furthest_phase_index": PHASES.index(self.phase.name),
            "ceiling_commands": self.steps,
        }


def _live_rollout(live):
    """The live state as a one-candidate, one-step rollout (for the rejection base)."""
    return ObjectRollout(
        live["palm"][None, None],
        np.array([[live["rotation_error"]]]),
        live["apple"][None, None],
        np.array([[live["contact"]]]),
        np.array([[live["dropped"]]]),
        np.ones((1, 1), bool),
        None,
    )
