"""TASK-094's torch side: M's roll-out and ceiling costs with their term decomposition, and S's
CEM call (protocol ``docs/experiments/grasp_closure_locus_v1.md`` §5.3, §6).

Everything is TASK-087's to TASK-093's frozen code, unchanged: the models are TASK-087's G-87100,
G-87101 and G-87102 under TASK-093's loader and pins; the planner model is
``grasp_corpus_wm_runtime.planner_model`` (λ = 2.0552); the step-averaged cost is
``zero_shot_grasp_runtime.wrap``'s step-mean setting; the CEM call is the one
``grasp_corpus_wm_planned_close_runtime.a2_plan`` makes (``zgr.make_planner``, CEM seed
``8803 * 10**7 + root``, unprojected), keeping the cost its best candidate received. The term
decomposition is a reported analysis of the same cost (TASK-089 A1 reported G-lat the same way),
not a cost variant; its total is checked against the planner's own ``distance``.

Torch is imported at module import (through the frozen runtimes); ``import embodied_jepa`` never
imports this module.
"""

from __future__ import annotations

import numpy as np
import torch

from embodied_jepa import grasp_closure_locus as gl
from embodied_jepa import grasp_corpus_wm_runtime as gr
from embodied_jepa import grasp_rate_projected_runtime as rpr
from embodied_jepa import zero_shot_grasp_runtime as zgr
from embodied_jepa.contracts import ContractError

TERMS = ("total", "latent", "pose", "arm", "hand")


def load_model(seed: int, device):
    """TASK-087's G-<seed> under TASK-088's guards, its pin and the pinned moments (TASK-093's
    loader)."""
    model, info = rpr.load_model(int(seed), device)
    if info["checkpoint_sha256"] != gl.CHECKPOINT_SHA256[f"G-v1-{seed}"]:
        raise ContractError(f"G-v1-{seed} is not the pinned checkpoint")
    return model, info


def planner_model(model, device: str = "cuda"):
    return gr.planner_model(model, device=device)


def encoder(device: str = "cuda"):
    return gr.encoder(device)


@torch.no_grad()
def cost_terms(pm, predictions: dict, goal: dict) -> dict:
    """The step cost per candidate and step ``[n, h]`` and its decomposition: the latent MSE, the
    pose term λ × the standardised MSE of the 14 right arm and hand joints, its arm (joints 0-6)
    and hand (joints 7-13) shares (arm + hand = pose), and the seven hand joints' shares
    ``[n, h, 7]``. ``total`` is the planner's own ``distance``; the decomposition must add up to
    it."""
    total = pm.distance(predictions, goal)[0]
    z = predictions["z"]
    lat = (z - goal["z"][:, None, None]).square().mean((-1, -2))[0]
    s = predictions["s"][..., :14]
    d2 = (s - goal["s_std"][:14]).square()[0]
    lam = float(pm.lam)
    pose = lam * d2.mean(-1)
    arm = lam * d2[..., :7].sum(-1) / 14
    hand_joints = lam * d2[..., 7:14] / 14
    out = {
        "total": total,
        "latent": lat.float().cpu().numpy(),
        "pose": pose.float().cpu().numpy(),
        "arm": arm.float().cpu().numpy(),
        "hand": hand_joints.sum(-1).float().cpu().numpy(),
        "hand_joints": hand_joints.float().cpu().numpy(),
    }
    if not np.allclose(out["latent"] + out["pose"], total, rtol=1e-4, atol=1e-5):
        raise ContractError("the cost decomposition does not add up to the planner's distance")
    return out


def rollout_terms(pm, root_latent, root_state, goal_latent, goal_state, sequences) -> dict:
    """M (i): G's roll-out from the root's encoded frame and measured state under each branch's
    applied window commands ``[n, 8, 14]`` (TASK-089 A1's roll-out), scored per step."""
    latent = pm.encode(root_latent, root_state)
    goal = pm.encode_goal(goal_latent, goal_state)
    cand = np.asarray(sequences, np.float32)[None]
    return cost_terms(pm, pm.predict(latent, cand), goal)


def ceiling_terms(pm, frame_latents, states, goal_latent, goal_state) -> dict:
    """M (ii), the ceiling: the same cost on each branch's encoded real frames ``[n, 8, 16, K]``
    and measured raw states ``[n, 8, 28]`` (TASK-089's ``ceiling_costs``, batched); ``pm.model``
    gives only the state standardisation (G's pinned train moments)."""
    goal = pm.encode_goal(goal_latent, goal_state)
    dev = pm.device
    z = torch.as_tensor(np.asarray(frame_latents, np.float32), device=dev)[None]
    raw = torch.as_tensor(np.asarray(states, np.float32), device=dev)
    s = pm.model.norm_state(raw)[None]
    return cost_terms(pm, {"z": z, "s": s}, goal)


def step_mean(terms: dict) -> dict:
    """The step-averaged cost (h = 1 ... 8) of every term, per candidate."""
    out = {k: terms[k].mean(-1) for k in TERMS}
    out["hand_joints"] = terms["hand_joints"].mean(-2)
    return out


def search_plan(pm, seed, root_latent, root_state, goal_latent, goal_state):
    """S (§6): one CEM call at a root with TASK-089's frozen, unprojected step-averaged planner,
    exactly ``pr.a2_plan``'s call. Returns the plan ``[8, 14]``, the cost its best candidate
    received and the requested plan (equal to the plan without projection)."""
    model, horizon = zgr.wrap(pm, gl.SETTING)
    plan = zgr.make_planner(int(seed), horizon).plan(
        model, model.encode(root_latent, root_state), model.encode_goal(goal_latent, goal_state)
    )
    return (
        plan.actions[0].astype(np.float32),
        float(plan.costs[0]),
        plan.requested_actions[0].astype(np.float32),
    )
