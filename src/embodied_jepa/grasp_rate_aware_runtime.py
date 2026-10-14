"""TASK-092's torch side: the G-req model (TASK-087's G whose loss feeds the augmented grasp input
to the predictor and the executed actions to the inverse head), the guarded G-req loader, Stage R's
roll-outs and Stage 0's two references (the inferred clip and the grasp-blind input).

TASK-087's ``GroundedWM``, TASK-090's ``grasp_corpus_wm_runtime`` and TASK-091's
``grasp_corpus_wm_planned_close_runtime`` are imported and never changed: the architecture, its
parameters and its implementation hash are TASK-087's, and the closed loop is TASK-091's recording
controller over TASK-089's step-averaged CEM.

Torch is imported at module import; ``import embodied_jepa`` never imports this module.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from embodied_jepa import grasp_corpus_wm as gc
from embodied_jepa import grasp_corpus_wm_runtime as gr
from embodied_jepa import grasp_rate_aware as ra
from embodied_jepa import grounded_wm as gw
from embodied_jepa import grounded_wm_data as gd
from embodied_jepa import grounded_wm_model as gm
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[2]


class GReqWM(gm.GroundedWM):
    """TASK-087's G (same modules, parameters and initialisation) with §4.1's loss split: the
    predictor's action input (one-step and roll-out) is ``a_in``; the inverse-dynamics target is
    the executed ``a_exec``. With ``a_in`` equal to ``a_exec`` it is TASK-087's loss, operation
    for operation. Everything except ``loss`` is inherited."""

    def loss(self, z, a_in, s=None, palm=None, a_exec=None):
        if a_exec is None:
            a_exec = a_in
        b, t1 = z.shape[:2]
        horizon = t1 - 1
        a = self._actions(a_in)
        a_target = self._actions(a_exec)
        w = gw.LOSS_WEIGHTS
        sn = self.norm_state(s) if (self.flags["state_token"] or self.flags["fusion"]) else None
        zt = z[:, :-1].reshape(b * horizon, gw.TOKENS, -1)
        at = a.reshape(b * horizon, -1)
        token = fused = None
        if self.flags["state_token"]:
            token = self.state_embed(sn[:, :-1].reshape(b * horizon, -1))
        if self.flags["fusion"]:
            fused = self.state_embed(sn[:, :-1].reshape(b * horizon, -1))
        one, slot = self.step(zt, at, token, fused)
        target = z[:, 1:].reshape(b * horizon, gw.TOKENS, -1)
        prediction_loss = (one - target).square().mean()
        recursive, rec_states = self.rollout(z[:, 0], a, s[:, 0] if s is not None else None)
        multistep = (recursive - z[:, 1:]).square().mean()
        loss = prediction_loss + w["multistep"] * multistep
        metrics = {"prediction_loss": prediction_loss.item(), "multistep_loss": multistep.item()}
        if self.flags["state_head"]:
            change = sn[:, 1:] - sn[:, :-1]
            tf = (self.state_head(slot) - change.reshape(b * horizon, -1)).square().mean()
            rec = (rec_states - sn[:, 1:]).square().mean()
            joint = 0.5 * (tf + rec)
            loss = loss + w["joint_change"] * joint
            metrics["joint_change_loss"] = joint.item()
        if self.flags["inverse"]:
            played = a_target[..., list(gw.PLAYED_ACTION_DIMS)]
            tf = (self.inverse(zt, one) - played.reshape(b * horizon, -1)).square().mean()
            inputs = torch.cat((z[:, :1], recursive[:, :-1]), 1)
            rec = (
                (
                    self.inverse(
                        inputs.reshape(b * horizon, gw.TOKENS, -1),
                        recursive.reshape(b * horizon, gw.TOKENS, -1),
                    )
                    - played.reshape(b * horizon, -1)
                )
                .square()
                .mean()
            )
            inv = 0.5 * (tf + rec)
            loss = loss + w["inverse"] * inv
            metrics["inverse_loss"] = inv.item()
        if self.flags["readout"]:
            pn = (palm - self.palm_mean) / self.palm_scale
            enc = (self.readout_head(z.flatten(2)) - pn).square().mean()
            pred = (self.readout_head(recursive.flatten(2)) - pn[:, 1:]).square().mean()
            readout = enc + pred
            loss = loss + w["readout"] * readout
            metrics["readout_loss"] = readout.item()
        metrics["loss"] = loss.item()
        return loss, metrics


# ----- paths and loaders -----------------------------------------------------------------------
def req_dir(root, seed: int, *, debug: bool = False) -> Path:
    base = "task092-debug" if debug else "task092-run"
    return Path(root) / "checkpoints" / base / f"G-req-{int(seed)}"


def load_req_model(path, device, *, debug: bool = False, root=ROOT):
    """A G-req checkpoint (§6.1), refused unless T-JOB-DONE (kind ``job``; ``debug`` admits a
    debug run), with its recorded checkpoint hash, TASK-087's implementation hash (recorded, in the
    checkpoint and here), this checkout's TASK-092 module hashes equal to the recorded ones,
    TASK-087's feature report and the pinned moments. Returns a plain ``GroundedWM`` (planning
    and roll-outs use TASK-087's forward, unchanged) and its info."""
    path = Path(path)
    rep = json.loads((path / "report.json").read_text())
    if rep.get("outcome") != "T-JOB-DONE" or rep.get("task") != ra.TASK:
        raise ContractError(f"{path} is not a finished TASK-092 job")
    if rep.get("kind") != "job" and not (debug and rep.get("kind") == "debug"):
        raise ContractError(f"{path} is a {rep.get('kind')} run")
    if rep.get("features_report_sha256") != ra.FEATURE_REPORT_SHA256:
        raise ContractError(f"{path} was trained on other features")
    if gd.sha256_file(path / "model.pt") != rep["checkpoint_sha256"]:
        raise ContractError(f"{path}/model.pt differs from its recorded sha256")
    ckpt = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    here = gm.implementation_sha256()
    if not (ckpt["implementation_sha256"] == rep["implementation_sha256"] == here):
        raise ContractError(f"{path} was trained by other code")
    if here != ra.IMPLEMENTATION_SHA256:
        raise ContractError("this checkout's TASK-087 code differs from TASK-087's")
    mods = ra.own_modules_sha256(root)
    if rep.get("modules_sha256") != mods or ckpt.get("modules_sha256") != mods:
        raise ContractError(f"{path}: TASK-092's modules differ from the ones that trained it")
    if (rep.get("p"), rep.get("augment_salt")) != (ra.AUG_P, ra.SALTS["augment"]) and not debug:
        raise ContractError(f"{path}: not §4.1's augmentation")
    model = gm.GroundedWM(rep["arm"], config=ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    buffers = gr.model_buffers(model)
    if gc.buffers_sha256(buffers) != ra.MOMENTS_SHA256:
        raise ContractError(f"{path}: not the pinned play-v1 moments")
    model.to(device).eval()
    info = {
        "family": "G-req",
        "arm": rep["arm"],
        "seed": int(rep["seed"]),
        "kind": rep["kind"],
        "checkpoint_sha256": rep["checkpoint_sha256"],
        "kept_update": rep["training"]["kept_update"],
        "last_two_triggered": rep["training"]["last_two_triggered"],
        "moments_sha256": gc.buffers_sha256(buffers),
        "modules_sha256": mods,
        "p": rep.get("p"),
    }
    return model, info


def load_v1(seed: int, device, arm: str = "G"):
    """TASK-087's checkpoint (M-v1, E (iii)'s N-87100) under TASK-088's guards and its pin."""
    model, info = gr.load_v1_model(seed, device, arm=arm)
    name = ra.model_name(f"{arm}-v1", seed)
    if info["checkpoint_sha256"] != ra.CHECKPOINT_SHA256[name]:
        raise ContractError(f"{name} is not the pinned checkpoint")
    if info["moments_sha256"] != ra.MOMENTS_SHA256:
        raise ContractError(f"{name}: not the pinned moments")
    return model, info | {"family": f"{arm}-v1"}


def load_task090(corpus: str, seed: int, device):
    """TASK-090's G models (Stage R, reported only) under its loader and TASK-091's pins."""
    model, info = gr.load_new_model(
        gr.model_dir(ROOT, "G", corpus, seed), device, feature_reports=ra.FEATURE_REPORTS_SHA256
    )
    name = f"G-{gc.CORPUS_TAG[corpus]}-{seed}"
    if info["checkpoint_sha256"] != ra.CHECKPOINT_SHA256[name]:
        raise ContractError(f"{name} is not the pinned checkpoint")
    return model, info | {"family": f"G-{gc.CORPUS_TAG[corpus]}"}


def load_family(family: str, seed: int, device, *, debug: bool = False, req_path=None):
    if family == "G-req":
        path = req_path if req_path is not None else req_dir(ROOT, seed, debug=debug)
        return load_req_model(path, device, debug=debug)
    if family == "G-v1":
        return load_v1(seed, device)
    raise ContractError(f"unknown family {family}")


# ----- roll-outs (Stage R) ---------------------------------------------------------------------
@torch.no_grad()
def rollout_at(model, start, commands, states, horizons, *, chunk=512, device="cuda"):
    """Predicted latents ``{h: [N, 16, K]}`` and standardised states ``{h: [N, 28]}`` (G) at the
    horizons, from one recursive roll-out per chunk (eval mode)."""
    model.eval()
    hmax = max(horizons)
    lat = {h: [] for h in horizons}
    sts = {h: [] for h in horizons}
    for lo in range(0, len(start), chunk):
        z0 = torch.as_tensor(np.asarray(start[lo : lo + chunk], np.float32), device=device)
        a = torch.as_tensor(np.asarray(commands[lo : lo + chunk, :hmax], np.float32), device=device)
        s0 = torch.as_tensor(np.asarray(states[lo : lo + chunk], np.float32), device=device)
        pz, ps = model.rollout(z0, a, s0)
        for h in horizons:
            lat[h].append(pz[:, h - 1].float().cpu().numpy())
            sts[h].append(ps[:, h - 1].float().cpu().numpy())
    out_l = {h: np.concatenate(v) for h, v in lat.items()}
    out_s = {h: np.concatenate(v) for h, v in sts.items()}
    for v in list(out_l.values()) + list(out_s.values()):
        if not np.isfinite(v).all():
            raise ContractError("non-finite prediction")
    return out_l, out_s


def norm_states(model, states, device="cuda") -> np.ndarray:
    with torch.no_grad():
        s = torch.as_tensor(np.asarray(states, np.float32), device=device)
        return model.norm_state(s).float().cpu().numpy()


def synergy_line(manifest: dict):
    opened = np.asarray(manifest["right_open_rad"], np.float64)
    closed = np.asarray(manifest["right_closed_rad"], np.float64)
    return opened, closed - opened


def infer_grasp(hand_q, manifest: dict) -> np.ndarray:
    """``zero_shot.synergy_grasp``'s least squares on the synergy line, vectorised ([..., 7])."""
    opened, d = synergy_line(manifest)
    t = ((np.asarray(hand_q, np.float64) - opened) @ d) / (d @ d)
    return np.clip(2 * t - 1, -1, 1)


@torch.no_grad()
def reference_rollout(model, start, commands, states, horizon, manifest, kind, *, device="cuda",
                      chunk=512):  # fmt: skip
    """Stage 0's two references (§13), G's roll-out unchanged except for the grasp input:
    ``clip`` -- each step's request clipped to +-2/11 around the grasp inferred from the
    measured hand joints (step 1) or the model's own predicted ones (later steps);
    ``blind`` -- index 13 held at the grasp inferred from the start's measured hand joints.
    Returns the standardised states at ``horizon`` [N, 28]."""
    model.eval()
    out = []
    for lo in range(0, len(start), chunk):
        z = torch.as_tensor(np.asarray(start[lo : lo + chunk], np.float32), device=device)
        s0 = np.asarray(states[lo : lo + chunk], np.float32)
        a = np.array(commands[lo : lo + chunk, :horizon], np.float32, copy=True)
        g0 = infer_grasp(s0[:, ra.HAND], manifest)
        if kind == "blind":
            a[:, :, ra.GRASP_DIM] = g0[:, None]
        sn = model.norm_state(torch.as_tensor(s0, device=device))
        token = model.state_embed(sn)
        hand = s0[:, ra.HAND].astype(np.float64)
        for k in range(horizon):
            ak = a[:, k].copy()
            if kind == "clip":
                gprev = infer_grasp(hand, manifest)
                ak[:, ra.GRASP_DIM] = np.clip(
                    ak[:, ra.GRASP_DIM], gprev - ra.RATE_LIMIT, gprev + ra.RATE_LIMIT
                )
            z, slot = model.step(z, torch.as_tensor(ak, device=device), token, None)
            sn = sn + model.state_head(slot)
            token = model.state_embed(sn)
            raw = (sn * model.state_scale + model.state_mean).float().cpu().numpy()
            hand = raw[:, ra.HAND].astype(np.float64)
        out.append(sn.float().cpu().numpy())
    return np.concatenate(out)


def g87100_report(task087=None) -> dict:
    base = gr.TASK087 if task087 is None else Path(task087)
    return json.loads((base / ra.G87100_REPORT).read_text())
