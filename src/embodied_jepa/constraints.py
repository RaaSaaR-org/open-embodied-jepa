"""Optional embodiment-owned candidate feasibility contract (version 1)."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from embodied_jepa.contracts import ContractError, validate_actions


@dataclass(frozen=True)
class CandidateProjection:
    """Target actions [B,K,T,A] and whole-sequence feasibility [B,K].

    Only the first action is checked against measured robot state. Later steps
    use a documented command-target kinematic surrogate, never future sensors.
    Infeasible rows contain normalized placeholders and must not win planning.
    """

    actions: np.ndarray
    feasible: np.ndarray

    def __post_init__(self):
        validate_actions(self.actions, ndim=4)
        if self.feasible.dtype != np.bool_ or self.feasible.shape != self.actions.shape[:2]:
            raise ContractError("candidate feasibility must be bool[B,K]")
        for name in ("actions", "feasible"):
            value = getattr(self, name).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)


class CandidateProjector(Protocol):
    def __call__(self, requested: np.ndarray) -> CandidateProjection:
        """Preview without actuation, sensor polling, or task/object truth."""
        ...
