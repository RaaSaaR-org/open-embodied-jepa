"""Hybrid phase control: tracked approach, then open-loop demonstration close (TASK-046).

This is a NON-LEARNED diagnostic scaffold. Before the handoff it delegates every
command to an unchanged :class:`~embodied_jepa.trajectory_tracking.TrackingController`
(whatever forward model that controller was built with). At the first observation
whose *measured* reference index reaches ``handoff_row`` it stops planning and
replays the retrieved TRAIN demonstration's own recorded actions open loop,
starting at the original demonstration frame of the matched reference row, through
the same bounds check and mandatory candidate projection as ``demo_replay``.

Any grasp achieved after the handoff is produced by replayed demonstration
actions, not by a model. No task evaluator, object pose or score enters this
controller; the handoff uses only the tracker's own measured-progress rule.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from embodied_jepa.constraints import CandidateProjection
from embodied_jepa.contracts import ContractError, ExecutionResult, Observation
from embodied_jepa.trajectory_tracking import TrackingController
from embodied_jepa.waypoint_planning import WaypointDecision

HYBRID_LABEL = "NON-LEARNED hybrid phase control: tracked approach + open-loop demonstration close"


def handoff_row(first_close_row, rows_before_close):
    """Reference row at which tracking hands off: ``close row - rows_before_close``."""
    if type(first_close_row) is not int or first_close_row < 1:
        raise ContractError("hybrid handoff needs a positive first-close reference row")
    if type(rows_before_close) is not int or rows_before_close < 0:
        raise ContractError("rows before the close row must be a nonnegative integer")
    return max(0, first_close_row - rows_before_close)


class HybridPhaseController:
    """Call step(observation, projector), execute once, then acknowledge(result)."""

    def __init__(
        self,
        tracker: TrackingController,
        demonstration_actions: np.ndarray,
        *,
        handoff_row: int,
        lower_bounds,
        upper_bounds,
    ):
        if not isinstance(tracker, TrackingController):
            raise ContractError("hybrid control wraps an unchanged TrackingController")
        actions = np.asarray(demonstration_actions)
        frames = tracker.reference.frames
        if (
            actions.dtype.kind != "f"
            or actions.ndim != 2
            or actions.shape[1] != 14
            or len(actions) != frames[-1]
            or not np.isfinite(actions).all()
        ):
            raise ContractError("replay needs the reference demonstration's T finite actions")
        if type(handoff_row) is not int or not 0 <= handoff_row < len(tracker.reference) - 1:
            raise ContractError("handoff row must precede the final reference row")
        self.tracker = tracker
        self.actions = actions.astype(np.float32, copy=True)
        self.actions.setflags(write=False)
        self.handoff_row = handoff_row
        self.lower = np.asarray(lower_bounds, np.float32)
        self.upper = np.asarray(upper_bounds, np.float32)
        if self.lower.shape != (14,) or self.upper.shape != (14,):
            raise ContractError("replay bounds must be 14-dimensional")
        self.phase = "tracking"
        self.handoff = None
        self.cursor = None  # next demonstration action index to replay
        self.replayed = 0
        self.pending = None  # "tracker" or (action, trace) during replay
        self.last_timestamp = -1.0
        self.last_execution_timestamp = -1.0
        self.termination_reason = None

    def _fresh(self, observation):
        if not isinstance(observation, Observation) or observation.timestamps.shape != (1,):
            raise ContractError("hybrid control requires one synchronized observation")
        timestamp = float(observation.timestamps[0])
        last = max(self.last_timestamp, self.tracker.last_timestamp)
        executed = max(self.last_execution_timestamp, self.tracker.last_execution_timestamp)
        if timestamp <= last or timestamp < executed:
            raise ContractError("hybrid control requires a fresh increasing observation")
        return timestamp

    def step(self, observation: Observation, projector) -> WaypointDecision:
        if self.pending is not None:
            raise ContractError("acknowledge the previous execution before replanning")
        if self.termination_reason is not None:
            return WaypointDecision(None, self.summary(), self.termination_reason)
        if self.phase == "tracking":
            timestamp = self._fresh(observation)
            # The tracker's own measured-progress rule; it consumes no randomness, so
            # every pre-handoff command equals the unwrapped tracker's command.
            index, distances, window = self.tracker._progress(observation)
            if index < self.handoff_row:
                decision = self.tracker.step(observation, projector)
                decision.trace["phase"] = "tracking"
                decision.trace["handoff_row"] = self.handoff_row
                if decision.action is None:
                    self.termination_reason = decision.termination_reason
                else:
                    self.pending = "tracker"
                return decision
            position = window.tolist().index(index)
            frame = self.tracker.reference.frames[index]
            self.handoff = {
                "handoff_reference_index": int(index),
                "handoff_frame": int(frame),
                "handoff_command": int(self.tracker.steps),
                "handoff_tracking_distance": float(distances[position]),
                "handoff_observation_timestamp": timestamp,
            }
            self.phase = "replay"
            self.cursor = int(frame)  # action f turns demonstration frame f into f + 1
        else:
            timestamp = self._fresh(observation)
        if self.cursor >= len(self.actions):
            self.termination_reason = "demo_exhausted"
            return WaypointDecision(None, self.summary(), self.termination_reason)
        if not callable(projector):
            raise ContractError("candidate projection is mandatory")
        recorded = self.actions[self.cursor]
        requested = np.clip(recorded, self.lower, self.upper).astype(np.float32)
        projection = projector(requested[None, None, None])
        if not isinstance(projection, CandidateProjection) or projection.actions.shape != (
            1,
            1,
            1,
            14,
        ):
            raise ContractError("projector changed candidate contract")
        if not projection.feasible[0, 0]:
            raise ContractError("hybrid replay command has no feasible projection")
        action = projection.actions[0, 0, 0].copy()
        action.setflags(write=False)
        trace = {
            "phase": "replay",
            "control": "hybrid_demo_replay_non_learned",
            "replay_frame": self.cursor,
            "replay_command": self.replayed,
            "recorded_action": recorded.tolist(),
            "sampled_action": requested.tolist(),
            "projected_action": action.tolist(),
            "observation_timestamp": timestamp,
            "handoff_row": self.handoff_row,
        } | self.handoff
        self.last_timestamp = timestamp
        self.pending = (action, trace)
        return WaypointDecision(action, trace)

    def acknowledge(self, result: ExecutionResult) -> dict[str, Any]:
        if self.pending is None:
            raise ContractError("no hybrid command awaits execution acknowledgement")
        if self.pending == "tracker":
            self.pending = None
            trace = self.tracker.acknowledge(result)
            trace["phase"] = "tracking"
            if self.tracker.termination_reason is not None:
                self.termination_reason = self.tracker.termination_reason
            return trace
        action, trace = self.pending
        if not isinstance(result, ExecutionResult) or not np.array_equal(
            result.requested_action, action
        ):
            raise ContractError("execution acknowledgement does not match replayed command")
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
        self.last_execution_timestamp = result.timestamp
        self.cursor += 1
        self.replayed += 1
        if result.applied_action is None:
            self.termination_reason = "execution_rejected"
        return trace

    def summary(self):
        handed = self.handoff is not None
        return {
            "control_label": HYBRID_LABEL,
            "handoff_row": self.handoff_row,
            "handed_off": handed,
            "tracked_commands": int(self.tracker.steps),
            "replayed_commands": self.replayed,
            "replay_start_frame": self.handoff["handoff_frame"] if handed else None,
            "demonstration_actions": len(self.actions),
        } | (
            self.handoff
            or {
                "handoff_reference_index": None,
                "handoff_frame": None,
                "handoff_command": None,
                "handoff_tracking_distance": None,
                "handoff_observation_timestamp": None,
            }
        )
