"""TASK-088's torch side: the frozen encoder, the TASK-087 world models behind the planner
contract of :class:`embodied_jepa.planning.CEMPlanner`, and the closed-loop episode.

The planner is ``embodied_jepa.planning.CEMPlanner``, unchanged (TASK-084, P0-PASS). This module
gives it the model contract it calls: ``predict(latent, candidates)`` rolls every candidate out
with the TASK-087 predictor from the current encoded frame (and, for G, the measured 28-D state),
and ``distance(predictions, goal)`` returns, per candidate and predicted step,
``w_lat * latent MSE + w_pose * lambda * standardised joint-position MSE``; the planner reads the
last step (protocol §5.1). Latents stay opaque to the planner.

Torch is imported at module import; ``import embodied_jepa`` never imports this module.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from embodied_jepa import grounded_wm as gw
from embodied_jepa import grounded_wm_data as gd
from embodied_jepa import play_corpus as pc
from embodied_jepa import zero_shot as zs
from embodied_jepa.contracts import ContractError

# TASK-087's recorded artefacts (results doc §4; feature report and val report in its evidence)
FEATURES_REPORT_SHA256 = "f5648a3529fe061dea219c26816253ea261c25ff891f043e0eb2a136e38e7eae"
PROJECTION_SHA256 = "c196fc48526123ac1b1f50849744e18232d7f09a1d7bcd719f80232a82bce33f"


class Encoder:
    """The TASK-087 latent of one frame: frozen DINOv2 ViT-S/14, 4 x 4 pooled, standardised and
    projected with TASK-087's train-fitted projection (k = 192)."""

    def __init__(self, projection_path, device="cuda"):
        self.featuriser = gd.Featuriser(device=device, batch=gw.FEATURE_BATCH)
        self.projection = gd.load_projection(projection_path, PROJECTION_SHA256)

    def __call__(self, frames) -> np.ndarray:
        frames = np.asarray(frames)
        if frames.ndim == 3:
            frames = frames[None]
        return gd.project(self.featuriser.pooled(frames), self.projection)


def load_model(path, device):
    """A TASK-087 checkpoint, refused unless T-JOB-DONE, trained on TASK-087's features, by this
    code's implementation and with its recorded checkpoint hash."""
    from embodied_jepa import grounded_wm_model as gm

    path = Path(path)
    rep = json.loads((path / "report.json").read_text())
    if rep.get("outcome") != "T-JOB-DONE" or rep.get("kind") != "job":
        raise ContractError(f"{path} is not a finished TASK-087 job")
    if rep["features_report_sha256"] != FEATURES_REPORT_SHA256:
        raise ContractError(f"{path} was trained on other features")
    if gd.sha256_file(path / "model.pt") != rep["checkpoint_sha256"]:
        raise ContractError(f"{path}/model.pt differs from its recorded sha256")
    ckpt = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    here = gm.implementation_sha256()
    if ckpt["implementation_sha256"] != here or rep["implementation_sha256"] != here:
        raise ContractError(f"{path} was trained by other code")
    model = gm.GroundedWM(rep["arm"], config=ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    info = {
        "arm": rep["arm"],
        "seed": int(rep["seed"]),
        "checkpoint_sha256": rep["checkpoint_sha256"],
        "kept_update": rep["training"]["kept_update"],
        "last_two_triggered": rep["training"]["last_two_triggered"],
    }
    return model, info


class PlannerModel:
    """The planner's model contract over one TASK-087 predictor (protocol §5.1)."""

    def __init__(self, model, *, w_lat: float, w_pose: float, lam: float, device: str):
        if w_pose > 0 and not model.flags["state_head"]:
            raise ContractError("a pose cost needs a model that predicts the state (G)")
        self.model, self.w_lat, self.w_pose, self.lam = model, w_lat, w_pose, lam
        self.device = device
        self.needs_state = model.flags["state_token"] or model.flags["fusion"]

    def encode(self, latent: np.ndarray, state28: np.ndarray) -> dict:
        z = torch.as_tensor(np.asarray(latent, np.float32), device=self.device).reshape(
            1, gw.TOKENS, gw.K
        )
        s = torch.as_tensor(np.asarray(state28, np.float32), device=self.device)[None]
        return {"z": z, "s": s}

    def encode_goal(self, latent: np.ndarray, state28: np.ndarray) -> dict:
        goal = self.encode(latent, state28)
        if self.model.flags["state_head"]:
            goal["s_std"] = self.model.norm_state(goal["s"])[0]
        return goal

    @torch.no_grad()
    def predict(self, latent: dict, candidates: np.ndarray) -> dict:
        b, n, h, _ = candidates.shape
        if b != 1:
            raise ContractError("one embodiment at a time")
        a = torch.as_tensor(candidates[0], device=self.device)
        z0 = latent["z"].expand(n, -1, -1)
        s0 = latent["s"].expand(n, -1) if self.needs_state else None
        z, states = self.model.rollout(z0, a, s0)
        return {"z": z[None], "s": None if states is None else states[None]}

    @torch.no_grad()
    def distance(self, predictions: dict, goal: dict) -> np.ndarray:
        z = predictions["z"]
        cost = torch.zeros(z.shape[:3], device=z.device)
        if self.w_lat:
            cost = cost + self.w_lat * (z - goal["z"][:, None, None]).square().mean((-1, -2))
        if self.w_pose:
            s = predictions["s"][..., :14]
            cost = cost + self.w_pose * self.lam * (s - goal["s_std"][:14]).square().mean(-1)
        out = cost.float().cpu().numpy()
        if not np.isfinite(out).all():
            raise ContractError("non-finite planning cost")
        return out


class ModelController:
    """A CEM controller over a :class:`PlannerModel`: encode the current frame, plan to the
    current subgoal, return the first action (protocol §5.1)."""

    def __init__(self, name, planner_model, encoder, seed: int):
        from embodied_jepa.planning import CEMConfig, CEMPlanner

        self.name = name
        self.pm, self.encoder = planner_model, encoder
        cfg = zs.CEM
        self.planner = CEMPlanner(
            CEMConfig(
                horizon=cfg["horizon"],
                samples=cfg["samples"],
                iterations=cfg["iterations"],
                elites=cfg["elites"],
                minimum_std=cfg["minimum_std"],
                seed=zs.cem_seed(seed),
                lower_bounds=tuple(zs.FIXED_LOWER.tolist()),
                upper_bounds=tuple(zs.FIXED_UPPER.tolist()),
            )
        )
        self.goals = None
        self.last = {}

    def set_goal(self, goal):
        latents = self.encoder(np.stack([g["frame"] for g in goal["subgoals"]]))
        self.goals = [
            self.pm.encode_goal(z, g["state28"])
            for z, g in zip(latents, goal["subgoals"], strict=True)
        ]

    def act(self, robot, observation, goal, index):
        p = zs.proprio(robot, observation)
        t0 = time.perf_counter()
        z = self.encoder(observation.images[pc.CAMERA][0])[0]
        latent = self.pm.encode(z, p["state28"])
        plan = self.planner.plan(self.pm, latent, self.goals[index])
        self.last = {"cost": float(plan.costs[0]), "seconds": time.perf_counter() - t0}
        action = plan.actions[0, 0].astype(np.float32)
        return action


def run_episode(robot, goal: dict, controller, *, record_actions=True) -> dict:
    """One closed-loop episode (protocol §5-§6). Every arm starts from the same reset and settle;
    the controller sees the observation, proprioception and the goal; the scorer reads simulator
    truth after every executed command."""
    task = goal["task"]
    zs.reset(robot, goal["layout"])
    if hasattr(controller, "set_goal"):
        controller.set_goal(goal)
    switch = zs.SubgoalSwitch(goal["subgoals"])
    sim = robot.sim
    index_obj = zs._target_index(goal["layout"]) if task == "grasp" else None
    distances, rise, contact, subgoal_index, applied, costs, seconds = [], [], [], [], [], [], []
    stop_reason, success_step = "", -1
    t_start = time.perf_counter()
    for _ in range(zs.BUDGET[task]):
        observation = robot.observe()
        p = zs.proprio(robot, observation)
        index = (
            switch.update(p["palm"], p["state28"][7:14], p["rotation"]) if task == "grasp" else 0
        )
        request = controller.act(robot, observation, goal, index)
        try:
            action = zs.execute(robot, request)
        except zs.Stopped as error:
            stop_reason = str(error)
            break
        subgoal_index.append(index)
        if record_actions:
            applied.append(np.asarray(action, np.float32))
        if hasattr(controller, "last") and controller.last:
            costs.append(controller.last["cost"])
            seconds.append(controller.last["seconds"])
        palm, _ = robot.ee_pose("right")  # scorer: simulator truth after the command
        if task == "reach":
            distances.append(float(np.linalg.norm(palm - np.asarray(goal["subgoals"][0]["palm"]))))
            done = zs.dwell_reached([d <= zs.REACH_TOLERANCE for d in distances], zs.REACH_DWELL)
        else:
            truth = sim.play_truth()
            rise.append(float(truth["position"][index_obj][2] - goal["target_z0"]))
            contact.append(bool(truth["grasp_contact"][index_obj]))
            distances.append(
                float(np.linalg.norm(palm - np.asarray(goal["subgoals"][index]["palm"])))
            )
            flags = [r >= zs.LIFT_HEIGHT and c for r, c in zip(rise, contact, strict=True)]
            done = zs.dwell_reached(flags, zs.LIFT_DWELL)
        if done >= 0:
            success_step = done
            break
    out = {
        "seed": goal["seed"],
        "task": task,
        "arm": controller.name,
        "success": success_step >= 0,
        "success_step": success_step,
        "steps": len(subgoal_index),
        "stop_reason": stop_reason,
        "distances": distances,
        "wall_seconds": time.perf_counter() - t_start,
    }
    if task == "reach":
        d = np.asarray(distances) if distances else np.array([np.inf])
        out["final_distance"] = float(d[-1])
        out["min_distance"] = float(d.min())
        out["success_at"] = {
            str(t): zs.reach_success(distances, t) for t in zs.REACH_REPORT_TOLERANCES
        }
    else:
        out["max_rise"] = float(max(rise)) if rise else 0.0
        out["any_grasp_contact"] = bool(any(contact))
        out["switches"] = switch.log
        out["final_subgoal"] = int(subgoal_index[-1]) if subgoal_index else 0
        out["rise"] = rise
        out["contact"] = contact
    if costs:
        out["plan_seconds_mean"] = float(np.mean(seconds))
        out["costs"] = costs
    if record_actions:
        out["applied"] = np.asarray(applied, np.float32).tolist()
    return out
