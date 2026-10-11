"""TASK-089's torch side: A1's roll-out costs and ceiling, A2's single CEM call per root, the three
candidate changes around TASK-088's ``PlannerModel`` (protocol §6) and Part B's controller.

Nothing in TASK-088's ``zero_shot_runtime`` or in ``planning.CEMPlanner`` changes: the changes are
wrappers that keep the planner's model contract (``predict`` / ``distance``).

Torch is imported at module import; ``import embodied_jepa`` never imports this module.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from embodied_jepa import play_corpus as pc
from embodied_jepa import zero_shot as zs
from embodied_jepa import zero_shot_grasp as zg
from embodied_jepa import zero_shot_runtime as zr
from embodied_jepa.contracts import ContractError


# ----- the candidate changes (protocol §6) -------------------------------------------------------
class StepMeanModel:
    """Candidate 1: the planning cost of a candidate is the mean of its per-step cost over
    h = 1..H (returned at every step, so the planner's last-step read is that mean)."""

    def __init__(self, pm):
        self.pm = pm

    def encode(self, *a):
        return self.pm.encode(*a)

    def encode_goal(self, *a):
        return self.pm.encode_goal(*a)

    def predict(self, latent, candidates):
        return self.pm.predict(latent, candidates)

    def distance(self, predictions, goal):
        d = self.pm.distance(predictions, goal)
        return np.repeat(d.mean(-1, keepdims=True), d.shape[-1], -1).astype(np.float32)


class HeldGraspModel:
    """Candidate 3: the model is rolled out with each candidate's step-1 right grasp repeated over
    the horizon (the other dimensions unchanged); the cost is TASK-088's."""

    def __init__(self, pm):
        self.pm = pm

    def encode(self, *a):
        return self.pm.encode(*a)

    def encode_goal(self, *a):
        return self.pm.encode_goal(*a)

    def predict(self, latent, candidates):
        held = np.array(candidates, np.float32, copy=True)
        held[..., zg.GRASP_DIM] = held[..., :1, zg.GRASP_DIM]
        return self.pm.predict(latent, held)

    def distance(self, predictions, goal):
        return self.pm.distance(predictions, goal)


def wrap(pm, setting: str):
    """The planner model and horizon of a setting: TASK-088's ("base") or a candidate."""
    if setting not in zg.SETTINGS:
        raise ContractError(f"unknown setting {setting}")
    model = {"step-mean": StepMeanModel, "held": HeldGraspModel}.get(setting)
    return (model(pm) if model else pm), zg.SETTING_HORIZON[setting]


def make_planner(seed: int, horizon: int):
    from embodied_jepa.planning import CEMConfig, CEMPlanner

    cfg = zs.CEM
    return CEMPlanner(
        CEMConfig(
            horizon=horizon,
            samples=cfg["samples"],
            iterations=cfg["iterations"],
            elites=cfg["elites"],
            minimum_std=cfg["minimum_std"],
            seed=zs.cem_seed(seed),
            lower_bounds=tuple(zs.FIXED_LOWER.tolist()),
            upper_bounds=tuple(zs.FIXED_UPPER.tolist()),
        )
    )


# ----- A1 (protocol §5.2) ------------------------------------------------------------------------
def rollout_costs(pm, root_latent, root_state, goal_latent, goal_state, sequences) -> np.ndarray:
    """The planner cost of each 16-command sequence at every step: ``[n_seq, 16]``, through the
    planner's own model contract (TASK-088's ``PlannerModel``)."""
    latent = pm.encode(root_latent, root_state)
    goal = pm.encode_goal(goal_latent, goal_state)
    cand = np.asarray(sequences, np.float32)[None]
    return pm.distance(pm.predict(latent, cand), goal)[0]


def ceiling_costs(pm, frames_latents, states, goal_latent, goal_state) -> np.ndarray:
    """The same cost with the simulator as predictor: one branch's encoded real frames
    ``[16, 16, K]`` and measured raw states ``[16, 28]`` in place of the roll-out; ``[16]`` out.
    ``pm.model`` gives only the state standardisation (G's train moments)."""
    goal = pm.encode_goal(goal_latent, goal_state)
    dev = pm.device
    z = torch.as_tensor(np.asarray(frames_latents, np.float32), device=dev)[None, None]
    s = None
    if pm.w_pose:
        raw = torch.as_tensor(np.asarray(states, np.float32), device=dev)
        s = pm.model.norm_state(raw)[None, None]
    return pm.distance({"z": z, "s": s}, goal)[0, 0]


# ----- A2 (protocol §5.3) ------------------------------------------------------------------------
def first_planned(pm, setting, seed, root_latent, root_state, goal_latent, goal_state):
    """One CEM call at a root: the first planned command (the one MPC executes)."""
    model, horizon = wrap(pm, setting)
    plan = make_planner(seed, horizon).plan(
        model, model.encode(root_latent, root_state), model.encode_goal(goal_latent, goal_state)
    )
    return plan.actions[0, 0].astype(np.float32)


# ----- Part B's controller (protocol §7) ---------------------------------------------------------
class ChangeController(zr.ModelController):
    """TASK-088's ``ModelController`` with a setting's planner model and horizon."""

    def __init__(self, name, planner_model, encoder, seed: int, setting: str):
        super().__init__(name, planner_model, encoder, seed)
        self.setting = setting
        self.pm, horizon = wrap(planner_model, setting)
        self.planner = make_planner(seed, horizon)

    def act(self, robot, observation, goal, index):
        p = zs.proprio(robot, observation)
        t0 = time.perf_counter()
        z = self.encoder(observation.images[pc.CAMERA][0])[0]
        latent = self.pm.encode(z, p["state28"])
        plan = self.planner.plan(self.pm, latent, self.goals[index])
        self.last = {"cost": float(plan.costs[0]), "seconds": time.perf_counter() - t0}
        return plan.actions[0, 0].astype(np.float32)
