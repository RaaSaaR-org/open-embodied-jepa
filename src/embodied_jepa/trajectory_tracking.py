"""Time-indexed demonstration trajectory tracking MPC (TASK-045).

The controller tracks a TRAIN demonstration's measured right-arm+hand position
trajectory instead of discrete endpoint goals. Each CEM candidate is scored over
its whole predicted horizon: the predicted state after step ``h`` is compared
with reference frame ``t + 1 + h``, and the cost is the mean over the horizon.
The reference index ``t`` advances only by *measured* progress: it is the latest
reference frame, in a forward window, that is nearest to the observed state.

The controller uses only the generic model contract (``encode``,
``encode_goal({"state": [1,D]})``, ``predict``, ``distance`` returning
``[1,K,H]``). Any state-goal model, learned or the privileged diagnostic
ceiling, plugs into the same cost without a backend branch. The CEM sampling,
candidate labels, RNG order, std schedule and tie-breaking mirror
``WaypointController`` with commitment 1; ``waypoint_planning`` is not modified.
No task evaluator, object pose or score enters this controller.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from embodied_jepa.constraints import CandidateProjection, CandidateProjector
from embodied_jepa.contracts import (
    EE_DELTA_GRASP_V0,
    ContractError,
    ExecutionResult,
    Observation,
    RobotState,
    validate_costs,
)
from embodied_jepa.waypoint_planning import WaypointConfig, WaypointDecision


def keyframe_indices(length: int, distance: Callable[[int, int], float], threshold: float):
    """Drop dead-time frames from a demonstration reference.

    Frame 0 is kept. Frame ``i`` is kept when ``distance(i, last_kept)`` exceeds
    ``threshold``; the final frame is always kept. Frames whose state does not
    move beyond the threshold from the last kept frame carry no state change to
    track, so removing them never removes motion.
    """
    if type(length) is not int or length < 2:
        raise ContractError("a reference needs at least two frames")
    if not np.isfinite(threshold) or threshold < 0:
        raise ContractError("keyframe threshold must be finite and nonnegative")
    kept = [0]
    for index in range(1, length):
        value = float(distance(index, kept[-1]))
        if not np.isfinite(value) or value < 0:
            raise ContractError("invalid keyframe distance")
        if value > threshold:
            kept.append(index)
    if kept[-1] != length - 1:
        kept.append(length - 1)
    return kept


@dataclass(frozen=True)
class TrackingReference:
    """TRAIN demonstration reference: raw state rows at kept original frames."""

    states: np.ndarray  # [L,D] float32
    frames: tuple[int, ...]  # original demonstration frame of each reference row
    source_id: str
    source_split: str = "train"

    def __post_init__(self):
        if (
            self.source_split != "train"
            or not isinstance(self.source_id, str)
            or not self.source_id
        ):
            raise ContractError("tracking references require declared training provenance")
        states = self.states
        if (
            not isinstance(states, np.ndarray)
            or states.dtype.kind != "f"
            or states.ndim != 2
            or len(states) < 2
            or states.shape[1] < 1
            or not np.isfinite(states).all()
        ):
            raise ContractError("tracking reference must be finite float[L>=2,D]")
        frames = tuple(int(f) for f in self.frames)
        if (
            len(frames) != len(states)
            or frames[0] != 0
            or any(b <= a for a, b in zip(frames, frames[1:], strict=False))
        ):
            raise ContractError("reference frames must start at 0 and increase strictly")
        copy = states.astype(np.float32, copy=True)
        copy.setflags(write=False)
        object.__setattr__(self, "states", copy)
        object.__setattr__(self, "frames", frames)

    def __len__(self):
        return len(self.states)

    @property
    def sha256(self):
        return hashlib.sha256(
            str(self.states.shape).encode()
            + self.states.astype("<f4").tobytes()
            + np.asarray(self.frames, "<i8").tobytes()
        ).hexdigest()


@dataclass(frozen=True)
class TrackingRule:
    window: int = 16  # forward frames searched for measured progress
    stall_commands: int = 64  # sliding window of acknowledged commands
    stall_min_advance: int = 16  # reference frames required within that window

    def __post_init__(self):
        for name in ("window", "stall_commands", "stall_min_advance"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ContractError(f"{name} must be a positive integer")


class TrackingController:
    """Call step(observation, projector), execute once, then acknowledge(result).

    ``progress_distance(robot_state, goal_rows)`` returns one finite distance per
    row of ``robot_state`` against the matching goal row (the model-owned
    measured-state metric). Commitment is fixed to one command per search.
    """

    def __init__(
        self,
        model,
        reference: TrackingReference,
        *,
        progress_distance: Callable[[RobotState, np.ndarray], np.ndarray],
        config: WaypointConfig,
        rule: TrackingRule | None = None,
    ):
        if not isinstance(reference, TrackingReference):
            raise ContractError("a validated TRAIN tracking reference is required")
        if not isinstance(config, WaypointConfig) or config.commitment_steps != 1:
            raise ContractError("trajectory tracking replans after every command")
        if not callable(progress_distance):
            raise ContractError("explicit measured-state progress distance is required")
        self.model = model
        self.reference = reference
        self.config = config
        self.rule = rule or TrackingRule()
        self.progress_distance = progress_distance
        self.rng = np.random.default_rng(config.seed)
        self.index = 0
        self.index_history = []  # measured index at the search of each acknowledged command
        self.steps = 0
        self.last_timestamp = -1.0
        self.last_execution_timestamp = -1.0
        self.last_grasps = np.array(config.initial_grasps, np.float32)
        self.warm = None
        self.pending = None
        self.termination_reason = None
        self._goal_latents = {}
        # Commands issued against the final row. The demonstration spent
        # frames[-1] - frames[-2] frames after its last kept motion (settling on the
        # plate), so the tracker holds the final row for that many commands.
        self.final_commands = 0
        self.final_hold = reference.frames[-1] - reference.frames[-2]

    def _progress(self, observation):
        """Monotone measured progress over the forward window; ties go to the later frame."""
        last = len(self.reference) - 1
        window = np.arange(self.index, min(self.index + self.rule.window, last) + 1)
        state = observation.state
        rows = RobotState(
            np.repeat(state.values, len(window), 0),
            np.repeat(state.mask, len(window), 0),
            np.repeat(state.timestamps, len(window), 0),
            state.schema,
        )
        values = np.asarray(self.progress_distance(rows, self.reference.states[window]))
        if values.shape != (len(window),) or not np.isfinite(values).all() or (values < 0).any():
            raise ContractError("measured tracking distance must be finite nonnegative[window]")
        best = int(window[np.flatnonzero(values == values.min())[-1]])
        return best, values, window

    def _goal(self, frame):
        if frame not in self._goal_latents:
            self._goal_latents[frame] = self.model.encode_goal(
                {"state": self.reference.states[frame : frame + 1]}
            )
        return self._goal_latents[frame]

    def targets(self):
        last = len(self.reference) - 1
        return [min(self.index + 1 + h, last) for h in range(self.config.horizon)]

    def tracking_costs(self, prediction, targets):
        """Mean over the horizon of the model distance to each step's reference frame."""
        cfg = self.config
        per_step = np.empty((cfg.candidates, cfg.horizon), np.float64)
        cache = {}
        for h, frame in enumerate(targets):
            if frame not in cache:
                distances = self.model.distance(prediction, self._goal(frame))
                validate_costs(distances, (1, cfg.candidates, cfg.horizon))
                cache[frame] = np.asarray(distances[0], np.float64)
            per_step[:, h] = cache[frame][:, h]
        return per_step.mean(1), per_step

    def _stalled(self):
        """True once the current index gained too little over the last stall window.

        ``index_history[k]`` is the measured index when command ``k`` was planned, so
        ``index_history[-S]`` is the index ``S`` acknowledged commands ago.
        """
        n = self.rule.stall_commands
        if len(self.index_history) < n:
            return False
        return self.index - self.index_history[-n] < self.rule.stall_min_advance

    def step(self, observation: Observation, projector: CandidateProjector) -> WaypointDecision:
        if self.pending is not None:
            raise ContractError("acknowledge the previous execution before replanning")
        if self.termination_reason is not None:
            return WaypointDecision(None, {"steps": self.steps}, self.termination_reason)
        if self.steps >= self.config.max_steps:
            self.termination_reason = "step_limit"
            return WaypointDecision(None, {"steps": self.steps}, self.termination_reason)
        if not isinstance(observation, Observation) or observation.timestamps.shape != (1,):
            raise ContractError("tracking MPC requires one synchronized observation")
        timestamp = float(observation.timestamps[0])
        if timestamp <= self.last_timestamp or timestamp < self.last_execution_timestamp:
            raise ContractError("tracking MPC requires a fresh increasing observation")
        if not callable(projector):
            raise ContractError("candidate projection is mandatory")
        cfg = self.config
        self.model.capabilities.require(
            action_schema=EE_DELTA_GRASP_V0,
            state_schema=observation.state_schema,
            horizon=cfg.horizon,
            history=1,
            device=cfg.device,
        )
        start = time.perf_counter()
        previous = self.index
        self.index, window_distances, window = self._progress(observation)
        last = len(self.reference) - 1
        position = window.tolist().index(self.index)
        base = {
            "reference_index": self.index,
            "reference_frame": self.reference.frames[self.index],
            "reference_advance": self.index - previous,
            "reference_length": len(self.reference),
            "reference_source_id": self.reference.source_id,
            "tracking_distance": float(window_distances[position]),
            "observation_timestamp": timestamp,
            "steps": self.steps,
        }
        if self.index == last:
            base["final_row_commands"] = self.final_commands
            base["final_row_hold"] = self.final_hold
            if self.final_commands >= self.final_hold:
                self.termination_reason = "reference_complete"
                return WaypointDecision(None, base, self.termination_reason)
            self.final_commands += 1
        elif self._stalled():
            # Recorded failure: fewer than stall_min_advance reference frames were
            # gained over the last stall_commands acknowledged commands.
            self.termination_reason = "reference_stall"
            return WaypointDecision(
                None,
                base
                | {
                    "stall_commands": self.rule.stall_commands,
                    "stall_min_advance": self.rule.stall_min_advance,
                },
                self.termination_reason,
            )
        targets = self.targets()
        lower = np.array(cfg.lower_bounds, np.float32)
        upper = np.array(cfg.upper_bounds, np.float32)
        hold = np.zeros((cfg.horizon, 14), np.float32)
        hold[:, 12:] = self.last_grasps
        hold = np.clip(hold, lower, upper)
        mean = hold.copy() if self.warm is None else np.clip(self.warm, lower, upper)
        latent = self.model.encode(observation.images, observation.state)
        best = None
        rounds = []
        sigma = cfg.proposal_std
        for iteration in range(cfg.iterations):
            requested = np.clip(
                mean[None] + self.rng.normal(0, sigma, (cfg.candidates, cfg.horizon, 14)),
                lower,
                upper,
            ).astype(np.float32)
            labels = ["perturbation"] * cfg.candidates
            requested[0], requested[1] = hold, mean
            labels[0], labels[1] = "hold", "warm_or_search_best"
            projection_start = time.perf_counter()
            projection = projector(requested[None])
            projection_seconds = time.perf_counter() - projection_start
            if not isinstance(projection, CandidateProjection) or projection.actions.shape != (
                1,
                *requested.shape,
            ):
                raise ContractError("projector changed candidate contract")
            feasible = projection.feasible[0]
            indices = np.flatnonzero(feasible)
            if not len(indices):
                raise ContractError("no feasible tracking candidate sequence")
            score_permutation = np.arange(cfg.candidates)
            unshuffled_endpoint = None
            if cfg.ablation == "persistence":
                costs = np.full(cfg.candidates, base["tracking_distance"], np.float64)
            else:
                prediction = self.model.predict(latent, projection.actions)
                costs, per_step = self.tracking_costs(prediction, targets)
                unshuffled_endpoint = per_step[:, -1]
                if cfg.ablation == "dynamics_shuffle":
                    score_permutation[indices] = self.rng.permutation(indices)
                    costs[indices] = costs[score_permutation[indices]]
            feasible_costs = costs[indices]
            minimum = feasible_costs.min()
            ties = indices[costs[indices] == minimum]
            winner = int(self.rng.choice(ties))
            item = {
                "cost": float(costs[winner]),
                "requested": requested[winner].copy(),
                "projected": projection.actions[0, winner].copy(),
                "origin": labels[winner],
                "round": iteration,
                "index": winner,
                "endpoint": None
                if unshuffled_endpoint is None
                else float(unshuffled_endpoint[winner]),
            }
            if (
                best is None
                or item["cost"] < best["cost"]
                or (item["cost"] == best["cost"] and self.rng.integers(2))
            ):
                best = item
            mean = best["requested"].copy()
            rounds.append(
                {
                    "iteration": iteration,
                    "feasible_candidates": len(indices),
                    "candidate_costs": [
                        float(costs[i]) if feasible[i] else None for i in range(cfg.candidates)
                    ],
                    "cost_spread": float(np.ptp(feasible_costs)),
                    "cost_std": float(feasible_costs.std()),
                    "projection_seconds": projection_seconds,
                    "selected_index": winner,
                    "score_permutation": score_permutation.tolist(),
                    "proposal_std": sigma,
                }
            )
            sigma = max(cfg.minimum_std, sigma * 0.5)
        action = best["projected"][0].copy()
        action.setflags(write=False)
        trace = base | {
            "decision_kind": "search",
            "plan_step": self.steps,
            "target_reference_indices": targets,
            "target_reference_frames": [self.reference.frames[i] for i in targets],
            "reference_sha256": self.reference.sha256,
            "selected_cost_semantics": "mean_over_horizon_of_per_step_reference_distance",
            "ablation": cfg.ablation,
            "selected_cost": best["cost"],
            # The winner's own (never permuted) last-step distance; diagnostic only.
            "winner_unshuffled_endpoint_cost": best["endpoint"],
            "selected_origin": best["origin"],
            "selected_round": best["round"],
            "sampled_action": best["requested"][0].tolist(),
            "projected_action": action.tolist(),
            "rounds": rounds,
            "candidate_evaluations": cfg.candidates * cfg.iterations,
            "planning_seconds": time.perf_counter() - start,
        }
        self.last_timestamp = timestamp
        self.pending = (action, best["projected"], trace)
        return WaypointDecision(action, trace)

    def acknowledge(self, result: ExecutionResult) -> dict[str, Any]:
        if self.pending is None:
            raise ContractError("no tracking command awaits execution acknowledgement")
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
        self.steps += 1
        self.index_history.append(self.index)
        self.last_execution_timestamp = result.timestamp
        self.pending = None
        if result.applied_action is None:
            self.termination_reason = "execution_rejected"
        else:
            self.last_grasps = result.applied_action[12:].copy()
            self.warm = np.concatenate((sequence[1:], sequence[-1:])).copy()
        return trace
