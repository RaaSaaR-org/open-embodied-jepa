"""Dense RGB-waypoint MPC with opaque dynamics and explicit proposal/ablation controls.

Waypoint sources and optional action proposals must come from training data.
Progress receives images only. No task evaluator or goal proprioception enters
this controller, and a demonstration proposal is never executed without scoring.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from embodied_jepa.constraints import CandidateProjection, CandidateProjector
from embodied_jepa.contracts import (
    EE_DELTA_GRASP_V0,
    ContractError,
    ExecutionResult,
    Observation,
    validate_actions,
    validate_costs,
)


@dataclass(frozen=True)
class ImageWaypoint:
    images: Mapping[str, np.ndarray]
    threshold: float
    dwell_observations: int
    source_id: str
    source_split: str = "train"
    action_proposals: np.ndarray | None = None

    def __post_init__(self):
        if (
            self.source_split != "train"
            or not isinstance(self.source_id, str)
            or not self.source_id
        ):
            raise ContractError("waypoints require declared training-source provenance")
        if not np.isfinite(self.threshold) or self.threshold < 0:
            raise ContractError("waypoint threshold must be finite and nonnegative")
        if type(self.dwell_observations) is not int or self.dwell_observations < 1:
            raise ContractError("waypoint dwell must be a positive observation count")
        if not isinstance(self.images, Mapping) or not self.images:
            raise ContractError("waypoint requires named RGB images")
        images = {}
        for name, array in self.images.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(array, np.ndarray)
                or array.dtype != np.uint8
                or array.ndim != 4
                or array.shape[0] != 1
                or array.shape[-1] != 3
                or any(n < 1 for n in array.shape)
            ):
                raise ContractError("waypoint images must be uint8[1,H,W,3]")
            copied = array.copy()
            copied.setflags(write=False)
            images[name] = copied
        object.__setattr__(self, "images", MappingProxyType(images))
        if self.action_proposals is not None:
            validate_actions(self.action_proposals, ndim=3)
            proposals = self.action_proposals.copy()
            proposals.setflags(write=False)
            object.__setattr__(self, "action_proposals", proposals)

    @property
    def image_hash(self):
        digest = hashlib.sha256()
        for name, value in sorted(self.images.items()):
            digest.update(name.encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class WaypointConfig:
    horizon: int = 4
    candidates: int = 16
    iterations: int = 2
    max_steps: int = 1000
    commitment_steps: int = 1
    seed: int = 0
    proposal_std: float = 0.15
    minimum_std: float = 0.05
    initial_grasps: tuple[float, float] = (-1.0, -1.0)
    lower_bounds: tuple[float, ...] = (-1.0,) * 14
    upper_bounds: tuple[float, ...] = (1.0,) * 14
    ablation: str = "learned"
    device: str = "cpu"

    def __post_init__(self):
        for name in ("horizon", "candidates", "iterations", "max_steps", "commitment_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ContractError(f"{name} must be a positive integer")
        if self.commitment_steps > self.horizon:
            raise ContractError("commitment_steps cannot exceed horizon")
        if self.candidates < 3 or self.max_steps > 1000:
            raise ContractError("require at least three candidates and at most1000steps")
        if type(self.seed) is not int or self.seed < 0:
            raise ContractError("seed must be a nonnegative integer")
        for name in ("proposal_std", "minimum_std"):
            if not np.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ContractError(f"{name} must lie in (0,1]")
        if self.minimum_std > self.proposal_std:
            raise ContractError("minimum_std cannot exceed proposal_std")
        for name, count in (("lower_bounds", 14), ("upper_bounds", 14), ("initial_grasps", 2)):
            values = np.asarray(getattr(self, name), dtype=np.float32)
            if (
                values.shape != (count,)
                or not np.isfinite(values).all()
                or np.any(np.abs(values) > 1)
            ):
                raise ContractError(f"{name} must contain finite normalized values")
            object.__setattr__(self, name, tuple(float(v) for v in values))
        if np.any(np.array(self.lower_bounds) > self.upper_bounds):
            raise ContractError("action bounds are reversed")
        if self.ablation not in ("learned", "dynamics_shuffle", "persistence"):
            raise ContractError("unsupported dynamics ablation")
        if self.device not in ("cpu", "mps"):
            raise ContractError("unsupported model device")


@dataclass(frozen=True)
class WaypointDecision:
    action: np.ndarray | None
    trace: dict[str, Any]
    termination_reason: str | None = None


class WaypointController:
    """Call step(observation, projector), execute once, then acknowledge(result).

    progress_distance(current_images, goal_images) returns one finite distance.
    It is an explicit model-owned image metric; calibrate waypoint thresholds on
    training demonstrations before development trials. The callback never gets
    robot state, actions, task scores or phase labels from this controller.
    """

    def __init__(
        self,
        model,
        waypoints: Sequence[ImageWaypoint],
        *,
        progress_distance: Callable[[Mapping, Mapping], np.ndarray],
        config: WaypointConfig | None = None,
    ):
        self.model = model
        self.waypoints = tuple(waypoints)
        self.config = config or WaypointConfig()
        if not self.waypoints or any(not isinstance(w, ImageWaypoint) for w in self.waypoints):
            raise ContractError("at least one validated image waypoint is required")
        if not callable(progress_distance):
            raise ContractError("explicit observed-image progress distance is required")
        for waypoint in self.waypoints:
            proposal = waypoint.action_proposals
            if proposal is not None and (
                proposal.shape[1:] != (self.config.horizon, 14)
                or proposal.shape[0] > self.config.candidates - 2
            ):
                raise ContractError("demonstration proposals must fit the horizon/candidate budget")
        self.progress_distance = progress_distance
        self.rng = np.random.default_rng(self.config.seed)
        self.goal_index = 0
        self.dwell = 0
        self.steps = 0
        self.last_timestamp = -1.0
        self.last_execution_timestamp = -1.0
        self.last_grasps = np.array(self.config.initial_grasps, np.float32)
        self.warm = None
        self.pending = None
        self.termination_reason = None
        self._goal_latents = {}
        self._commitment = None

    def _distance(self, images, waypoint):
        values = np.asarray(self.progress_distance(images, waypoint.images))
        if values.shape != (1,) or not np.isfinite(values).all() or values[0] < 0:
            raise ContractError("observed-image distance must be finite nonnegative[1]")
        return float(values[0])

    def step(self, observation: Observation, projector: CandidateProjector) -> WaypointDecision:
        if self.pending is not None:
            raise ContractError("acknowledge the previous execution before replanning")
        if self.termination_reason is not None:
            return WaypointDecision(None, {"steps": self.steps}, self.termination_reason)
        if self.steps >= self.config.max_steps:
            self.termination_reason = "step_limit"
            return WaypointDecision(None, {"steps": self.steps}, self.termination_reason)
        if not isinstance(observation, Observation) or observation.timestamps.shape != (1,):
            raise ContractError("waypoint MPC requires one synchronized observation")
        timestamp = float(observation.timestamps[0])
        if timestamp <= self.last_timestamp or timestamp < self.last_execution_timestamp:
            raise ContractError("waypoint MPC requires a fresh increasing observation")
        if not callable(projector):
            raise ContractError("candidate projection is mandatory")
        self.model.capabilities.require(
            action_schema=EE_DELTA_GRASP_V0,
            state_schema=observation.state_schema,
            horizon=self.config.horizon,
            history=1,
            device=self.config.device,
        )
        start = time.perf_counter()
        waypoint = self.waypoints[self.goal_index]
        observed_distance = self._distance(observation.images, waypoint)
        self.dwell = self.dwell + 1 if observed_distance <= waypoint.threshold else 0
        advanced = self.dwell >= waypoint.dwell_observations
        advance_abort = None
        if advanced and self._commitment is not None:
            advance_abort = {
                "abort_reason": "waypoint_advanced",
                "plan_step": self._commitment["trace"]["plan_step"],
                "commitment_offset": self._commitment["offset"],
            }
        if advanced:
            self._commitment = None
            self.goal_index += 1
            self.dwell = 0
            if self.goal_index == len(self.waypoints):
                self.termination_reason = "waypoints_complete"
                return WaypointDecision(
                    None,
                    {
                        "goal_index": self.goal_index,
                        "observed_distance": observed_distance,
                        "waypoint_advanced": True,
                        "observation_timestamp": timestamp,
                        "cache_validation": advance_abort,
                        "commitment_remaining": 0,
                    },
                    self.termination_reason,
                )
            waypoint = self.waypoints[self.goal_index]
            observed_distance = self._distance(observation.images, waypoint)
        cfg = self.config
        cache_validation = advance_abort
        if self._commitment is not None:
            cache = self._commitment
            offset = cache["offset"]
            original = cache["projected"][offset]
            projection_start = time.perf_counter()
            refreshed = projector(original[None, None, None].copy())
            if not isinstance(refreshed, CandidateProjection) or refreshed.actions.shape != (
                1,
                1,
                1,
                14,
            ):
                raise ContractError("projector changed cached command contract")
            valid = bool(refreshed.feasible[0, 0])
            unchanged = np.array_equal(refreshed.actions[0, 0, 0], original)
            cache_validation = {
                "plan_step": cache["trace"]["plan_step"],
                "commitment_offset": offset,
                "original_scored_action": original.tolist(),
                "refreshed_projected_action": refreshed.actions[0, 0, 0].tolist(),
                "feasible": valid,
                "unchanged": unchanged,
                "projection_seconds": time.perf_counter() - projection_start,
                "abort_reason": None
                if valid and unchanged
                else ("infeasible_cached_command" if not valid else "changed_cached_command"),
            }
            if valid and unchanged:
                action = original.copy()
                action.setflags(write=False)
                trace = cache["trace"].copy()
                # The assigned cost belongs to the original search, including shuffled
                # ablation assignments. It is never a newly scored cached forecast.
                trace.update(
                    step=self.steps,
                    decision_kind="commitment",
                    commitment_offset=offset,
                    commitment_remaining=cfg.commitment_steps - offset,
                    selected_round=None,
                    rounds=[],
                    candidate_evaluations=0,
                    goal_dwell_observed=self.dwell,
                    observed_distance=observed_distance,
                    waypoint_advanced=advanced,
                    observation_timestamp=timestamp,
                    sampled_action=cache["requested"][offset].tolist(),
                    projected_action=action.tolist(),
                    cache_validation=cache_validation,
                    planning_seconds=time.perf_counter() - start,
                )
                trace.pop("selected_requested_sequence", None)
                trace.pop("selected_projected_sequence", None)
                self.last_timestamp = timestamp
                # Warm always retains the full prediction horizon. The short
                # commitment counter never determines pending sequence shape.
                self.pending = (action, self.warm.copy(), trace)
                return WaypointDecision(action, trace)
            self._commitment = None
        lower, upper = (
            np.array(cfg.lower_bounds, np.float32),
            np.array(cfg.upper_bounds, np.float32),
        )
        hold = np.zeros((cfg.horizon, 14), np.float32)
        hold[:, 12:] = self.last_grasps
        hold = np.clip(hold, lower, upper)
        mean = hold.copy() if self.warm is None else np.clip(self.warm, lower, upper)
        latent = self.model.encode(observation.images, observation.state)
        if self.goal_index not in self._goal_latents:
            self._goal_latents[self.goal_index] = self.model.encode_goal(waypoint.images)
        goal_latent = self._goal_latents[self.goal_index]
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
            if waypoint.action_proposals is not None:
                size = len(waypoint.action_proposals)
                requested[2 : 2 + size] = np.clip(waypoint.action_proposals, lower, upper)
                labels[2 : 2 + size] = ["demonstration_proposal"] * size
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
                raise ContractError("no feasible waypoint candidate sequence")
            score_permutation = np.arange(cfg.candidates)
            if cfg.ablation == "persistence":
                costs = np.full(cfg.candidates, observed_distance, np.float64)
            else:
                prediction = self.model.predict(latent, projection.actions)
                distances = self.model.distance(prediction, goal_latent)
                validate_costs(distances, (1, cfg.candidates, cfg.horizon))
                costs = distances[0, :, -1].astype(np.float64)
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
        trace = {
            "step": self.steps,
            "decision_kind": "search",
            "plan_step": self.steps,
            "plan_observation_timestamp": timestamp,
            "plan_goal_index": self.goal_index,
            "plan_goal_image_sha256": waypoint.image_hash,
            "plan_horizon": cfg.horizon,
            "commitment_offset": 0,
            "commitment_remaining": cfg.commitment_steps,
            "selected_cost_semantics": "original_search_assigned_endpoint_cost",
            "cache_validation": cache_validation,
            "goal_index": self.goal_index,
            "goal_source_id": waypoint.source_id,
            "goal_image_sha256": waypoint.image_hash,
            "goal_threshold": waypoint.threshold,
            "goal_dwell_required": waypoint.dwell_observations,
            "goal_dwell_observed": self.dwell,
            "observed_distance": observed_distance,
            "waypoint_advanced": advanced,
            "observation_timestamp": timestamp,
            "ablation": cfg.ablation,
            "selected_cost": best["cost"],
            "selected_origin": best["origin"],
            "selected_round": best["round"],
            "sampled_action": best["requested"][0].tolist(),
            "projected_action": action.tolist(),
            "rounds": rounds,
            "candidate_evaluations": cfg.candidates * cfg.iterations,
            "planning_seconds": time.perf_counter() - start,
        }
        if cfg.commitment_steps > 1:
            trace["selected_requested_sequence"] = best["requested"].tolist()
            trace["selected_projected_sequence"] = best["projected"].tolist()
            trace["plan_selected_round"] = best["round"]
            trace["plan_selected_index"] = best["index"]
            self._commitment = {
                "offset": 0,
                "requested": best["requested"].copy(),
                "projected": best["projected"].copy(),
                "trace": trace.copy(),
            }
        self.last_timestamp = timestamp
        self.pending = (action, best["projected"], trace)
        return WaypointDecision(action, trace)

    def acknowledge(self, result: ExecutionResult) -> dict[str, Any]:
        if self.pending is None:
            raise ContractError("no waypoint command awaits execution acknowledgement")
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
        self.last_execution_timestamp = result.timestamp
        self.pending = None
        if result.applied_action is None:
            self.termination_reason = "execution_rejected"
            trace["commitment_abort_reason"] = "execution_rejected"
            self._commitment = None
        else:
            self.last_grasps = result.applied_action[12:].copy()
            self.warm = np.concatenate((sequence[1:], sequence[-1:])).copy()
            # Warm starts remain proposals, including when transport clips further.
            if self._commitment is not None:
                self._commitment["offset"] += 1
                if not np.array_equal(result.applied_action, action):
                    trace["commitment_abort_reason"] = "applied_action_changed"
                    self._commitment = None
                elif self._commitment["offset"] >= self.config.commitment_steps:
                    trace["commitment_clear_reason"] = "commitment_completed"
                    self._commitment = None
        trace["commitment_remaining_after_ack"] = (
            0
            if self._commitment is None
            else self.config.commitment_steps - self._commitment["offset"]
        )
        return trace
