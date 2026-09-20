"""Model-independent bounded CEM and receding-horizon control."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from embodied_jepa.contracts import ACTION_DIM, ContractError, Observation, validate_costs


@dataclass(frozen=True)
class CEMConfig:
    horizon: int = 4
    samples: int = 64
    iterations: int = 3
    elites: int = 8
    seed: int = 0
    action_penalty: float = 0.0
    minimum_std: float = 0.05
    lower_bounds: tuple[float, ...] = (-1.0,) * ACTION_DIM
    upper_bounds: tuple[float, ...] = (1.0,) * ACTION_DIM

    def __post_init__(self):
        for name in ("horizon", "samples", "iterations", "elites"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        lower, upper = np.asarray(self.lower_bounds), np.asarray(self.upper_bounds)
        if (
            lower.shape != (ACTION_DIM,)
            or upper.shape != (ACTION_DIM,)
            or not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or np.any(lower < -1)
            or np.any(upper > 1)
            or np.any(lower > upper)
        ):
            raise ValueError("action bounds must be ordered finite normalized vectors")
        object.__setattr__(self, "lower_bounds", tuple(float(x) for x in lower))
        object.__setattr__(self, "upper_bounds", tuple(float(x) for x in upper))
        if self.elites > self.samples:
            raise ValueError("elites must not exceed samples")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not np.isfinite(self.action_penalty) or self.action_penalty < 0:
            raise ValueError("action_penalty must be finite and nonnegative")
        if not np.isfinite(self.minimum_std) or not 0 < self.minimum_std <= 1:
            raise ValueError("minimum_std must be in (0,1]")


@dataclass(frozen=True)
class Plan:
    actions: np.ndarray
    costs: np.ndarray
    elapsed_seconds: float
    evaluations: int
    config: dict[str, Any]


class CEMPlanner:
    """Latents stay opaque; adapters handle chunking and inference-only execution."""

    def __init__(self, config: CEMConfig | None = None):
        self.config = config or CEMConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def plan(self, model, latent, goal_latent, *, batch_size: int = 1) -> Plan:
        cfg = self.config
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        start = time.perf_counter()
        shape = (batch_size, cfg.horizon, ACTION_DIM)
        lower = np.asarray(cfg.lower_bounds, dtype=np.float32)
        upper = np.asarray(cfg.upper_bounds, dtype=np.float32)
        mean = (
            np.broadcast_to(np.clip(np.zeros(ACTION_DIM), lower, upper), shape)
            .astype(np.float32)
            .copy()
        )
        std = np.ones(shape, dtype=np.float32)
        best_actions = mean.copy()
        best_cost = np.full(batch_size, np.inf)
        for _ in range(cfg.iterations):
            candidates = np.clip(
                self.rng.normal(mean[:, None], std[:, None], (batch_size, cfg.samples, *shape[1:])),
                lower,
                upper,
            ).astype(np.float32)
            # Preserve the current mean and best candidate, rather than losing a good solution.
            candidates[:, 0] = mean
            if cfg.samples > 1:
                candidates[:, 1] = best_actions
            predictions = model.predict(latent, candidates)
            distances = model.distance(predictions, goal_latent)
            validate_costs(distances, (batch_size, cfg.samples, cfg.horizon))
            costs = distances[:, :, -1].astype(np.float64)
            costs += cfg.action_penalty * np.square(candidates).mean(axis=(2, 3))
            indices = np.argsort(costs, axis=1, kind="stable")[:, : cfg.elites]
            elite = candidates[np.arange(batch_size)[:, None], indices]
            current_cost = costs[np.arange(batch_size), indices[:, 0]]
            improved = current_cost < best_cost
            best_cost[improved] = current_cost[improved]
            best_actions[improved] = elite[improved, 0]
            mean = elite.mean(axis=1)
            std = np.maximum(elite.std(axis=1), cfg.minimum_std)
        return Plan(
            best_actions,
            best_cost.astype(np.float32),
            time.perf_counter() - start,
            batch_size * cfg.samples * cfg.iterations,
            asdict(cfg),
        )


class MPC:
    """Execute one action per observation; deadline misses stop without stale actuation."""

    def __init__(
        self, model, planner: CEMPlanner, *, timeout_seconds: float = 5.0, device: str = "cpu"
    ):
        if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        self.model, self.planner, self.timeout_seconds = model, planner, timeout_seconds
        self.device = device

    def run(
        self,
        embodiment,
        goal,
        *,
        max_steps: int,
        evaluate: Callable[[], dict[str, Any]] | None = None,
        record_observation: Callable[[int, Observation], None] | None = None,
    ) -> dict[str, Any]:
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        traces = []
        last_timestamp = -1.0
        reason = "step_limit"
        score = {}
        stage, step = "encode_goal", 0
        row = {}
        try:
            goal_latent = self.model.encode_goal(goal)
            for step in range(max_steps):
                start = time.perf_counter()
                stage = "observe"
                observation: Observation = embodiment.observe()
                if observation.timestamps.shape != (1,):
                    raise ContractError("MPC executes one embodiment at a time")
                if record_observation is not None:
                    record_observation(step, observation)
                timestamp = float(observation.timestamps[0])
                if timestamp <= last_timestamp:
                    reason = "stale_observation"
                    traces.append({"step": step, "executed": False, "reason": reason})
                    break
                last_timestamp = timestamp
                stage = "capabilities"
                self.model.capabilities.require(
                    action_schema=embodiment.action_schema,
                    state_schema=embodiment.state_schema,
                    horizon=self.planner.config.horizon,
                    history=1,
                    device=self.device,
                )
                stage = "plan"
                latent = self.model.encode(observation.images, observation.state)
                plan = self.planner.plan(self.model, latent, goal_latent)
                elapsed = time.perf_counter() - start
                row = {
                    "step": step,
                    "observation_timestamp": timestamp,
                    "planning_seconds": plan.elapsed_seconds,
                    "control_seconds": elapsed,
                    "candidate_evaluations": plan.evaluations,
                    "planned_cost": float(plan.costs[0]),
                }
                if elapsed > self.timeout_seconds:
                    row.update(executed=False, reason="deadline_miss")
                    traces.append(row)
                    reason = "deadline_miss"
                    break
                row["requested_action"] = plan.actions[0, 0].tolist()
                stage = "execute"
                result = embodiment.execute(plan.actions[0, 0])
                row.update(
                    executed=result.applied_action is not None,
                    status=result.status,
                    action=None
                    if result.applied_action is None
                    else result.applied_action.tolist(),
                    execution_timestamp=result.timestamp,
                    reason=result.reason,
                )
                traces.append(row)
                if result.applied_action is None:
                    reason = result.status
                    break
                if evaluate is not None:
                    stage = "evaluate"
                    score = evaluate()
                    if score.get("success", False):
                        reason = "success"
                        break
        except Exception as error:
            reason = "runtime_error"
            traces.append(
                (row if stage == "execute" else {})
                | {
                    "step": step,
                    "stage": stage,
                    "error": f"{type(error).__name__}: {error}",
                    "executed": None if stage == "execute" else False,
                    "execution_uncertain": stage == "execute",
                }
            )
        finally:
            # Even transport rejection need not imply a stop; every exit explicitly holds.
            embodiment.stop(reason)
        return {
            "termination_reason": reason,
            "trace": traces,
            "score": score,
            "replans": sum("candidate_evaluations" in t for t in traces),
            "executed_steps": sum(t["executed"] is True for t in traces),
        }
