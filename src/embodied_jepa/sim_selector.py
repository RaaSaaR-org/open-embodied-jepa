"""TASK-073: privileged branch rollouts in cloned simulator state (H-sim and the ranking truth).

``Brancher`` saves the live simulator's full ``MjData`` (``mj_copyData``) and the transport and
embodiment fields that the next command depends on, runs a candidate for a fixed number of
closed-loop commands through the exact attempt path (``observe`` -> the controller's ``act`` ->
``project_candidates`` -> ``execute``), reads the true apple-minus-plate offset, and restores
everything bit for bit. The latched scorer and the at-rest record are never called inside a
branch, and the image renderer is replaced by a blank frame there (P-3 reads no image).

Privileged by design (rung L4): ``SimSelector`` (H-sim) holds the robot handle and reads truth;
its successes are a selection ceiling, never a learned result. The ranking stage (O3, O4) uses
the same ``Brancher`` to score every candidate's true 16-command outcome. NumPy at import.
"""

from __future__ import annotations

import time

import numpy as np

from embodied_jepa import first_policy_v2_runtime as rt2
from embodied_jepa import wm_critic_v2 as wc
from embodied_jepa.contracts import ContractError


class Brancher:
    """Save / branch / restore on one live robot."""

    def __init__(self, robot):
        self.robot = robot
        self.sim = robot.sim
        self.mj, self.model = robot.mj, robot.model
        self._saved = self.mj.MjData(self.model)
        self._blank = np.zeros((self.sim.height, self.sim.width, 3), np.uint8)

    def _save(self) -> dict:
        self.mj.mj_copyData(self._saved, self.model, self.sim.data)
        r = self.robot
        return {
            "targets": self.sim.targets.copy(),
            "stopped": self.sim.stopped_reason,
            "grasp": r._grasp.copy(),
            "observation": r._observation,
            "wall": r._observation_wall,
            "projection_q": None if r._observation is None else r._projection_q.copy(),
            "projection_targets": None if r._observation is None else r._projection_targets.copy(),
        }

    def _restore(self, saved: dict) -> None:
        self.mj.mj_copyData(self.sim.data, self.model, self._saved)
        r = self.robot
        self.sim.targets[:] = saved["targets"]
        self.sim.stopped_reason = saved["stopped"]
        r._grasp[:] = saved["grasp"]
        r._observation = saved["observation"]
        r._observation_wall = saved["wall"]
        if saved["projection_q"] is not None:
            r._projection_q = saved["projection_q"].copy()
            r._projection_targets = saved["projection_targets"].copy()

    def _step(self, controller, step: int, lower, upper) -> str | None:
        """One closed-loop command in the branch; the reason it stopped, or None."""
        observation = type(self.robot).observe(self.robot)  # bypasses any harness wrapper
        command = np.clip(np.asarray(controller.act(observation, step), np.float32), lower, upper)
        try:
            projection = self.robot.project_candidates(command[None, None, None])
        except ContractError as error:
            if str(error) not in rt2.GUARD_REFUSALS:
                raise
            return "guard_refusal"
        if not bool(projection.feasible[0, 0]):
            return "infeasible_command"
        result = self.robot.execute(projection.actions[0, 0, 0])
        if result.applied_action is None:
            return result.reason or result.status or "rejected"
        controller.advance(result)
        return None

    def outcomes(self, controllers, step: int, bounds, plate_xy_at_end, horizon: int = wc.CHUNK):
        """Run each controller for ``horizon`` commands from the current state; return the true
        apple-minus-plate offsets (cm) after them (``nan`` if a branch stopped early) and the
        stop reasons. The live state is restored exactly afterwards."""
        lower, upper = bounds
        saved = self._save()
        self.sim.render = lambda camera="onboard_rgb": self._blank.copy()  # noqa: ARG005
        offsets = np.full((len(controllers), 2), np.nan)
        reasons = []
        try:
            for i, controller in enumerate(controllers):
                self._restore(saved)
                reason = None
                for j in range(horizon):
                    reason = self._step(controller, step + j, lower, upper)
                    if reason:
                        break
                reasons.append(reason)
                if reason is None:
                    apple = np.asarray(self.sim.data.body("apple").xpos[:2], np.float64)
                    offsets[i] = 100.0 * (apple - np.asarray(plate_xy_at_end, np.float64))
        finally:
            del self.sim.render
            self._restore(saved)
        return offsets, reasons


class SimSelector(rt2.LearnedController):
    """H-sim (L4): at each decision, the 25 aims around ``centre(observation, step)`` are each
    run for 16 commands in a cloned state; the aim whose true offset is nearest ``o*(t + 16)``
    is executed (tie rule as the critic's). A branch that stops early costs +inf."""

    def __init__(
        self,
        predict,
        standardiser,
        estimates,
        fk,
        bounds,
        *,
        robot,
        centre,
        plate_at,
        decision_steps=wc.DECISION_STEPS,
    ):
        super().__init__(predict, standardiser, estimates, fk, bounds)
        self.brancher = Brancher(robot)
        self.centre = centre  # (observation, step) -> incumbent plate xy
        self.plate_at = plate_at  # step -> the plate xy the harness will have at that step
        self.decision_steps = frozenset(int(s) for s in decision_steps)
        self.decisions: list[dict] = []

    def act(self, observation, step):
        step = int(step)
        if step in self.decision_steps:
            started = time.perf_counter()
            incumbent = np.asarray(self.centre(observation, step), np.float64).reshape(2)
            aims = wc.candidate_aims(incumbent)
            branches = [
                rt2.LearnedController(
                    self.predict,
                    self.standardiser,
                    np.concatenate((self.estimates[:2], aim)),
                    self.fk,
                    (self.lower, self.upper),
                )
                for aim in aims
            ]
            offsets, reasons = self.brancher.outcomes(
                branches, step, (self.lower, self.upper), self.plate_at(step + wc.CHUNK)
            )
            costs = wc.critic_cost_cm(offsets, step + wc.CHUNK)
            costs = np.where(np.isfinite(costs), costs, np.inf)
            index = wc.choose(costs)
            self.estimates[2:] = aims[index]
            self.decisions.append(
                {
                    "step": step,
                    "incumbent": incumbent.tolist(),
                    "chosen": int(index),
                    "costs_cm": [float(c) for c in costs],
                    "stopped_branches": sum(r is not None for r in reasons),
                    "seconds": time.perf_counter() - started,
                }
            )
        return super().act(observation, step)
