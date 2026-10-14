"""TASK-093's torch side: the projected CEM planner (TASK-089's planner with
``project_candidates=True`` and §4.1's projector through ``CEMPlanner``'s existing hook), Stage P's
per-root plans for both arms, P-MECH's predicted hand synergy, and the closed-loop controller.

Everything else is TASK-087's to TASK-092's frozen code, unchanged: the planner model is
``grasp_corpus_wm_runtime.planner_model`` (λ = 2.0552) wrapped by ``zero_shot_grasp_runtime.wrap``
(step-mean, H = 8); the CEM configuration is ``zero_shot_grasp_runtime.make_planner``'s with only
``project_candidates`` set; the controller is TASK-091's ``RecordingController`` (TASK-089's
``ChangeController``), whose planner is replaced by the projected one. ``planning.py`` is not
edited: the projected candidates are scored, the elites are refitted on the requests and the
projected plan is returned, as the existing code path does.

Torch is imported at module import (through the frozen runtimes); ``import embodied_jepa`` never
imports this module.
"""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np

from embodied_jepa import grasp_corpus_wm_planned_close_runtime as pr
from embodied_jepa import grasp_corpus_wm_runtime as gr
from embodied_jepa import grasp_rate_projected as rp
from embodied_jepa import zero_shot_grasp_runtime as zgr
from embodied_jepa.contracts import ContractError


def projected_planner(seed: int, horizon: int):
    """TASK-089's planner (``zgr.make_planner``) with ``project_candidates=True`` and nothing else
    changed: the same CEM seed, so its generator draws exactly the unprojected planner's
    first-iteration samples."""
    from embodied_jepa.planning import CEMPlanner

    base = zgr.make_planner(seed, horizon)
    return CEMPlanner(dataclasses.replace(base.config, project_candidates=True))


class FirstPredictRecorder:
    """Wraps a planner model: records the sha256 of the first ``predict`` call's candidates (the
    unprojected planner's first-iteration samples) and changes nothing."""

    def __init__(self, model):
        self.model = model
        self.first_sha256 = None

    def encode(self, *a):
        return self.model.encode(*a)

    def encode_goal(self, *a):
        return self.model.encode_goal(*a)

    def predict(self, latent, candidates):
        if self.first_sha256 is None:
            self.first_sha256 = hashlib.sha256(np.asarray(candidates).tobytes()).hexdigest()
        return self.model.predict(latent, candidates)

    def distance(self, predictions, goal):
        return self.model.distance(predictions, goal)


def root_plan(pm, seed, root_latent, root_state, goal_latent, goal_state, *, projector=None):
    """One CEM call at a root (Stage P), as ``pr.a2_plan`` makes it (``zgr.wrap`` with the step-mean
    setting, CEM seed ``8803 * 10**7 + root``); with ``projector`` through the projected planner.
    Returns the plan (``[H, 14]``), the requested plan and the sha256 of the first-iteration
    samples."""
    model, horizon = zgr.wrap(pm, rp.SETTING)
    rec = FirstPredictRecorder(model)
    latent, goal = rec.encode(root_latent, root_state), rec.encode_goal(goal_latent, goal_state)
    if projector is None:
        plan = zgr.make_planner(seed, horizon).plan(rec, latent, goal)
        first = rec.first_sha256
    else:
        plan = projected_planner(seed, horizon).plan(rec, latent, goal, projector=projector)
        first = projector.first_requested_sha256
    return (
        plan.actions[0].astype(np.float32),
        plan.requested_actions[0].astype(np.float32),
        first,
    )


def predicted_synergy(model, enc_root, plan, state, manifest) -> list[float]:
    """P-MECH (reported): the model's predicted right-hand synergy at plan steps 1-8 under the
    plan (TASK-092's reading: ``synergy_grasp``'s least squares on predicted joints)."""
    from embodied_jepa import grasp_rate_aware as ra
    from embodied_jepa import grasp_rate_aware_runtime as rr

    _, ps = gr.rollout_numpy(
        model,
        np.asarray(enc_root)[None],
        np.asarray(plan, np.float32)[None],
        np.asarray(state, np.float32)[None],
    )
    raw = ps[0] * model.state_scale.cpu().numpy() + model.state_mean.cpu().numpy()
    return [float(v) for v in rr.infer_grasp(raw[:, ra.HAND], manifest)]


class ProjectedPlanRecorder(pr.PlanRecorder):
    """TASK-091's recorder over the projected planner: ``plan`` passes the current projector to
    ``CEMPlanner.plan`` and keeps the plan's projected and requested right-grasp channels."""

    def __init__(self, planner):
        super().__init__(planner)
        if not planner.config.project_candidates:
            raise ContractError("the projected recorder needs project_candidates=True")
        self.projector = None
        self.last_requested = None

    def plan(self, model, latent, goal):
        if self.projector is None:
            raise ContractError("no projector for this command")
        plan = self.planner.plan(model, latent, goal, projector=self.projector)
        self.last_grasp = [float(v) for v in plan.actions[0, :, rp.GRASP_DIM]]
        self.last_requested = [float(v) for v in plan.requested_actions[0, :, rp.GRASP_DIM]]
        self.last_first = plan.actions[0, 0].astype(np.float32).copy()
        return plan


class ProjectedController(pr.RecordingController):
    """M-proj: TASK-091's recording G-chg controller whose planner is the projected one. At each
    command T₀ is read from the embodiment (its last accepted hand targets, at the observation)
    and the projector of §4.1 is passed through the existing hook; ``plan_log`` also keeps the
    requested grasp per plan step."""

    def __init__(self, name, planner_model, encoder, seed: int, setting: str = rp.SETTING):
        super().__init__(name, planner_model, encoder, seed, setting)
        horizon = self.planner.config.horizon
        self.planner = ProjectedPlanRecorder(projected_planner(seed, horizon))
        self.geometry = None

    def act(self, robot, observation, goal, index):
        if self.geometry is None:
            self.geometry = rp.RateGeometry.from_robot(robot)
        self.planner.projector = rp.GraspRateProjector(self.geometry, rp.read_t0(robot))
        try:
            command = super().act(robot, observation, goal, index)
        finally:
            self.planner.projector = None
        self.plan_log[-1]["requested_grasp"] = self.planner.last_requested
        return command


def controller(arm: str, model, enc, episode_seed: int):
    """The closed-loop world-model controller of an arm: M-proj (projected) or M-v1 (TASK-091's
    recording controller, unprojected, as TASK-091 and TASK-092 ran it)."""
    if arm == "M-proj":
        return ProjectedController(arm, gr.planner_model(model), enc, episode_seed)
    if arm == "M-v1":
        return pr.controller(arm, model, enc, episode_seed, record=True)
    raise ContractError(f"unknown arm {arm}")


def load_model(seed: int, device):
    """TASK-087's G-<seed> under TASK-088's guards, its pin and the pinned moments (TASK-092's
    ``load_v1``)."""
    from embodied_jepa import grasp_rate_aware_runtime as rr

    model, info = rr.load_v1(seed, device)
    if info["checkpoint_sha256"] != rp.CHECKPOINT_SHA256[f"G-v1-{seed}"]:
        raise ContractError(f"G-v1-{seed} is not the pinned checkpoint")
    return model, info
