"""CEM over any registered action schema (SO-101 S0, TASK-096).

``planning.py`` sizes ``CEMConfig`` and ``CEMPlanner`` by the 14-D ``ee_delta_grasp_v0``
(``ACTION_DIM``) and is byte-pinned by the frozen G1 protocols (TASK-091–TASK-094), so it stays
as it is. ``SchemaCEMConfig`` and ``SchemaCEMPlanner`` are the same bounded CEM with the action
dimension taken from a declared ``ActionSchema``. ``SchemaCEMPlanner.plan`` mirrors
``CEMPlanner.plan`` line for line apart from the dimension; a test holds it to the same
candidates, costs and plan as ``CEMPlanner`` on the 14-D schema with the same seed.

``create_planner`` keeps the 14-D path on the pinned ``CEMPlanner``: a G1 config never reaches
this module's planner. ``planning.MPC`` runs either planner unchanged (it reads only
``config.horizon`` and ``config.project_candidates``). NumPy only.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from embodied_jepa.action_schemas import action_schema_from_record, schema_record
from embodied_jepa.constraints import CandidateProjector
from embodied_jepa.contracts import (
    EE_DELTA_GRASP_V0,
    ActionSchema,
    ContractError,
    validate_costs,
)
from embodied_jepa.planning import CEMConfig, CEMPlanner, Plan


@dataclass(frozen=True)
class SchemaCEMConfig:
    """``CEMConfig``'s budget with bounds sized by ``action_schema`` (``None`` = [-1, 1])."""

    action_schema: ActionSchema = EE_DELTA_GRASP_V0
    horizon: int = 4
    samples: int = 64
    iterations: int = 3
    elites: int = 8
    seed: int = 0
    action_penalty: float = 0.0
    minimum_std: float = 0.05
    lower_bounds: tuple[float, ...] | None = None
    upper_bounds: tuple[float, ...] | None = None
    project_candidates: bool = False

    def __post_init__(self):
        if not isinstance(self.action_schema, ActionSchema):
            raise ValueError("action_schema must be an ActionSchema")
        try:
            action_schema_from_record(schema_record(self.action_schema))
        except ContractError as error:
            raise ValueError(f"unregistered action schema: {error}") from error
        dimension = self.action_schema.dimension
        if type(self.project_candidates) is not bool:
            raise ValueError("project_candidates must be boolean")
        for name in ("horizon", "samples", "iterations", "elites"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        lower = np.asarray(
            (-1.0,) * dimension if self.lower_bounds is None else self.lower_bounds, dtype=float
        )
        upper = np.asarray(
            (1.0,) * dimension if self.upper_bounds is None else self.upper_bounds, dtype=float
        )
        if (
            lower.shape != (dimension,)
            or upper.shape != (dimension,)
            or not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or np.any(lower < -1)
            or np.any(upper > 1)
            or np.any(lower > upper)
        ):
            raise ValueError(f"action bounds must be ordered finite normalized {dimension}-vectors")
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

    @property
    def dimension(self) -> int:
        return self.action_schema.dimension

    def budget(self) -> dict[str, Any]:
        """The schema-free fields, in ``CEMConfig``'s keyword form."""
        values = asdict(self)
        values.pop("action_schema")
        return values


class SchemaCEMPlanner(CEMPlanner):
    """``CEMPlanner`` with the action dimension taken from ``SchemaCEMConfig.action_schema``."""

    def __init__(self, config: SchemaCEMConfig | None = None):
        config = config or SchemaCEMConfig()
        if not isinstance(config, SchemaCEMConfig):
            raise ContractError("SchemaCEMPlanner requires a SchemaCEMConfig")
        self.config = config
        self.rng = np.random.default_rng(self.config.seed)

    def plan(
        self,
        model,
        latent,
        goal_latent,
        *,
        batch_size: int = 1,
        projector: CandidateProjector | None = None,
    ) -> Plan:
        # Mirrors planning.CEMPlanner.plan; the only change is ``dimension`` for ACTION_DIM.
        cfg = self.config
        dimension = cfg.dimension
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        if cfg.project_candidates and projector is None:
            raise ContractError("candidate projection requires an embodiment projector")
        start = time.perf_counter()
        projection_seconds, feasible_count = 0.0, 0
        shape = (batch_size, cfg.horizon, dimension)
        lower = np.asarray(cfg.lower_bounds, dtype=np.float32)
        upper = np.asarray(cfg.upper_bounds, dtype=np.float32)
        mean = (
            np.broadcast_to(np.clip(np.zeros(dimension), lower, upper), shape)
            .astype(np.float32)
            .copy()
        )
        std = np.ones(shape, dtype=np.float32)
        best_actions = mean.copy()
        best_requested = mean.copy()
        best_cost = np.full(batch_size, np.inf)
        for _ in range(cfg.iterations):
            candidates = np.clip(
                self.rng.normal(mean[:, None], std[:, None], (batch_size, cfg.samples, *shape[1:])),
                lower,
                upper,
            ).astype(np.float32)
            candidates[:, 0] = mean
            if cfg.samples > 1:
                candidates[:, 1] = best_requested
            requested = candidates.copy()
            feasible = np.ones((batch_size, cfg.samples), dtype=bool)
            if cfg.project_candidates:
                projection_start = time.perf_counter()
                projection = projector(requested)
                projection_seconds += time.perf_counter() - projection_start
                if projection.actions.shape != candidates.shape:
                    raise ContractError("projected candidate shape changed")
                candidates, feasible = projection.actions, projection.feasible
            feasible_count += int(feasible.sum())
            if np.any(~feasible.any(axis=1)):
                raise ContractError("no feasible candidate sequence")
            predictions = model.predict(latent, candidates)
            distances = model.distance(predictions, goal_latent)
            validate_costs(distances, (batch_size, cfg.samples, cfg.horizon))
            costs = distances[:, :, -1].astype(np.float64)
            costs += cfg.action_penalty * np.square(candidates).mean(axis=(2, 3))
            costs[~feasible] = np.inf
            for batch in range(batch_size):
                indices = np.argsort(costs[batch], kind="stable")[
                    : min(cfg.elites, int(feasible[batch].sum()))
                ]
                winner = indices[0]
                if costs[batch, winner] < best_cost[batch]:
                    best_cost[batch] = costs[batch, winner]
                    best_actions[batch] = candidates[batch, winner]
                    best_requested[batch] = requested[batch, winner]
                elite = requested[batch, indices]
                mean[batch] = elite.mean(axis=0)
                std[batch] = np.maximum(elite.std(axis=0), cfg.minimum_std)
        return Plan(
            best_actions,
            best_cost.astype(np.float32),
            time.perf_counter() - start,
            batch_size * cfg.samples * cfg.iterations,
            asdict(cfg),
            best_requested,
            projection_seconds,
            feasible_count,
        )


def create_planner(config: SchemaCEMConfig | CEMConfig) -> CEMPlanner:
    """The pinned ``CEMPlanner`` for ``ee_delta_grasp_v0``; ``SchemaCEMPlanner`` otherwise."""
    if isinstance(config, CEMConfig):
        return CEMPlanner(config)
    if not isinstance(config, SchemaCEMConfig):
        raise ContractError("planner config must be a CEMConfig or SchemaCEMConfig")
    if config.action_schema == EE_DELTA_GRASP_V0:
        return CEMPlanner(CEMConfig(**config.budget()))
    return SchemaCEMPlanner(config)
