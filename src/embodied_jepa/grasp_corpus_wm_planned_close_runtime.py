"""TASK-091's torch side: A″'s whole A2-seed plan, and the closed-loop controller that records
each re-plan for §5.3's deferral measures without changing any executed command.

Everything is TASK-089's and TASK-090's frozen code, unchanged: the planner model is
``grasp_corpus_wm_runtime.planner_model`` (λ = 2.0552), wrapped by ``zero_shot_grasp_runtime.wrap``
(step-mean, H = 8), and the CEM planner is ``zero_shot_grasp_runtime.make_planner``. The A2 plan is
the very call ``zero_shot_grasp_runtime.first_planned`` makes, with the whole plan kept; the
controller is ``zero_shot_grasp_runtime.ChangeController`` whose planner is wrapped by a recorder
that returns the planner's plan object untouched.

Torch is imported at module import (through the frozen runtimes); ``import embodied_jepa`` never
imports this module.
"""

from __future__ import annotations

import numpy as np

from embodied_jepa import grasp_corpus_wm_planned_close as pcl
from embodied_jepa import grasp_corpus_wm_runtime as gr
from embodied_jepa import zero_shot_grasp_runtime as zgr


def a2_plan(pm, setting, seed, root_latent, root_state, goal_latent, goal_state) -> np.ndarray:
    """§4: the A2 CEM call at a root, as ``zgr.first_planned`` makes it (``zgr.wrap``,
    ``zgr.make_planner(seed, horizon)``, so CEM seed ``8803 * 10**7 + root``), keeping the whole
    normalised plan ``[H, 14]``."""
    model, horizon = zgr.wrap(pm, setting)
    plan = zgr.make_planner(seed, horizon).plan(
        model, model.encode(root_latent, root_state), model.encode_goal(goal_latent, goal_state)
    )
    return plan.actions[0].astype(np.float32)


class PlanRecorder:
    """Wraps a ``CEMPlanner``: ``plan`` calls it unchanged and returns its plan object; it keeps
    the plan's right-grasp channel per plan step. The planner's own random state is the only
    state, and it is advanced exactly as without the recorder."""

    def __init__(self, planner):
        self.planner = planner
        self.config = planner.config
        self.last_grasp = None
        self.last_first = None

    def plan(self, model, latent, goal):
        plan = self.planner.plan(model, latent, goal)
        self.last_grasp = [float(v) for v in plan.actions[0, :, pcl.GRASP_DIM]]
        self.last_first = plan.actions[0, 0].astype(np.float32).copy()
        return plan


class RecordingController(zgr.ChangeController):
    """TASK-089's G-chg controller (TASK-090's ``gr.controller``) with its planner recorded:
    ``plan_log`` gets one entry per re-plan, the subgoal index it planned for and the plan's
    right-grasp channel per step. The returned command is the frozen ``act``'s, unchanged."""

    def __init__(self, name, planner_model, encoder, seed: int, setting: str):
        super().__init__(name, planner_model, encoder, seed, setting)
        self.planner = PlanRecorder(self.planner)
        self.plan_log = []

    def act(self, robot, observation, goal, index):
        command = super().act(robot, observation, goal, index)
        if not np.array_equal(command, self.planner.last_first):
            raise RuntimeError("the executed command is not the plan's first command")
        self.plan_log.append({"index": int(index), "grasp": self.planner.last_grasp})
        return command


def controller(name, model, enc, episode_seed: int, *, record: bool = True):
    """The closed-loop world-model controller: TASK-090's ``gr.controller`` (record=False), or
    the same controller with its re-plans recorded (record=True, §5.3)."""
    if not record:
        return gr.controller(name, model, enc, episode_seed)
    return RecordingController(name, gr.planner_model(model), enc, episode_seed, gr.SETTING)
