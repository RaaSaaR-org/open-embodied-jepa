"""TASK-090's torch side: the pinned moments check, the model loader for the new checkpoints
(TASK-088's guards with the new feature-report pins), featurisation of the new collections with
TASK-087's pinned projection, and the closed-loop controllers (TASK-089's step-averaged G-chg
setting).

TASK-087's model and loss code, TASK-088's ``zero_shot_runtime`` and TASK-089's
``zero_shot_grasp_runtime`` are imported and never changed.

Torch is imported at module import; ``import embodied_jepa`` never imports this module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from embodied_jepa import grasp_corpus_wm as gc
from embodied_jepa import grounded_wm_data as gd
from embodied_jepa import zero_shot as zs
from embodied_jepa import zero_shot_runtime as zr
from embodied_jepa.contracts import ContractError

# ----- TASK-087's artefacts (paths and pins) -----------------------------------------------------
TASK087 = Path(os.environ.get("TASK087_RUN", "/home/huhn/develop/emai/worktrees/task087-run"))
V1_FEATURES = TASK087 / "outputs/task087-features"
V1_CHECKPOINTS = TASK087 / "checkpoints/task087-run"
V1_FEATURES_REPORT_SHA256 = zr.FEATURES_REPORT_SHA256
PROJECTION_SHA256 = zr.PROJECTION_SHA256
ENCODER_DIGEST = "3a697b872089933d345fd71087af86e85ff31886cf1e2654763dc1884ef2af27"
IMPLEMENTATION_SHA256 = "e2ec0096c14f628c248fda7fb1a05ad2d3342fc7e20703727bcc178efd7149b1"
PIN_MODEL = "G-87100"
LAMBDA = zs.LAMBDA
SETTING = "step-mean"  # TASK-089's selected change (G-chg), held fixed (§2.2)


def new_features(root, corpus: str) -> Path:
    """The new collections' feature stores (on the data disk, through ``data/``)."""
    return Path(root) / "data" / f"task090-features-{gc.CORPUS_TAG[corpus]}"


def model_dir(root, arm: str, corpus: str, seed: int, *, debug: bool = False) -> Path:
    base = "task090-debug" if debug else "task090-run"
    return Path(root) / "checkpoints" / base / f"{arm}-{gc.CORPUS_TAG[corpus]}-{seed}"


# ----- the pinned moments (§2.2) -----------------------------------------------------------------
def checkpoint_buffers(path) -> dict:
    sd = torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=True)["state_dict"]
    return {k: sd[k].numpy() for k in gc.MOMENT_BUFFERS}


def model_buffers(model) -> dict:
    return {k: getattr(model, k).detach().cpu().numpy() for k in gc.MOMENT_BUFFERS}


def pinned_buffers(features=V1_FEATURES) -> dict:
    """The buffers ``GroundedWM.set_moments`` makes from ``play-v1`` train's moments."""
    from embodied_jepa import grounded_wm_model as gm

    model = gm.GroundedWM("G")
    model.set_moments(*gc.pinned_moments(features))
    return model_buffers(model)


def check_pinned_moments(features=V1_FEATURES, reference=None) -> dict:
    """§2.2: the buffers made from ``play-v1`` train's moments equal G-87100's bitwise."""
    reference = V1_CHECKPOINTS / PIN_MODEL if reference is None else Path(reference)
    mine, ref = pinned_buffers(features), checkpoint_buffers(reference)
    same = gc.same_buffers(mine, ref)
    out = {
        "reference": str(reference),
        "bitwise_equal": bool(same),
        "pinned_sha256": gc.buffers_sha256(mine),
        "reference_sha256": gc.buffers_sha256(ref),
    }
    if not same:
        raise ContractError("the pinned moments differ from G-87100's buffers")
    return out


# ----- loading models (§5, §7) -------------------------------------------------------------------
def load_new_model(path, device, *, feature_reports: dict, debug: bool = False):
    """A TASK-090 checkpoint, refused unless T-JOB-DONE (``kind`` job, or debug), trained by
    TASK-087's code (implementation hash), on exactly the given feature reports
    (``{name: sha256}``), with its recorded checkpoint hash and the pinned moments."""
    from embodied_jepa import grounded_wm_model as gm

    path = Path(path)
    rep = json.loads((path / "report.json").read_text())
    if rep.get("outcome") != "T-JOB-DONE":
        raise ContractError(f"{path} is not a finished TASK-090 job")
    if rep.get("kind") != "job" and not debug:
        raise ContractError(f"{path} is a {rep.get('kind')} run")
    if rep.get("task") != gc.TASK:
        raise ContractError(f"{path} is not a TASK-090 model")
    if rep["feature_reports_sha256"] != feature_reports:
        raise ContractError(f"{path} was trained on other features")
    if gd.sha256_file(path / "model.pt") != rep["checkpoint_sha256"]:
        raise ContractError(f"{path}/model.pt differs from its recorded sha256")
    ckpt = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    here = gm.implementation_sha256()
    if not (ckpt["implementation_sha256"] == rep["implementation_sha256"] == here):
        raise ContractError(f"{path} was trained by other code")
    if here != IMPLEMENTATION_SHA256:
        raise ContractError("this checkout's TASK-087 code differs from TASK-087's")
    model = gm.GroundedWM(rep["arm"], config=ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    buffers = model_buffers(model)
    if gc.buffers_sha256(buffers) != rep["moments"]["pinned_sha256"]:
        raise ContractError(f"{path}: moment buffers differ from its report")
    if not gc.same_buffers(buffers, pinned_buffers()):
        raise ContractError(f"{path}: not the pinned play-v1 moments")
    model.to(device).eval()
    info = {
        "arm": rep["arm"],
        "corpus": rep["corpus"],
        "seed": int(rep["seed"]),
        "checkpoint_sha256": rep["checkpoint_sha256"],
        "kept_update": rep["training"]["kept_update"],
        "last_two_triggered": rep["training"]["last_two_triggered"],
        "moments_sha256": gc.buffers_sha256(buffers),
    }
    return model, info


def load_v1_model(seed: int, device, arm: str = "G"):
    """TASK-087's checkpoint (M-v1) under TASK-088's guards; its buffers are the pinned ones."""
    model, info = zr.load_model(V1_CHECKPOINTS / f"{arm}-{seed}", device)
    buffers = model_buffers(model)
    if not gc.same_buffers(buffers, pinned_buffers()):
        raise ContractError(f"{arm}-{seed}: not the pinned play-v1 moments")
    info = info | {"corpus": "play-v1", "moments_sha256": gc.buffers_sha256(buffers)}
    return model, info


def planner_model(model, *, w_pose: float = 1.0, device: str = "cuda"):
    return zr.PlannerModel(model, w_lat=1.0, w_pose=w_pose, lam=LAMBDA, device=device)


def encoder(device: str = "cuda"):
    return zr.Encoder(V1_FEATURES / "projection.npz", device)


def controller(name, model, enc, episode_seed: int):
    """TASK-089's G-chg: the step-averaged cost (λ = 2.0552), H = 8, TASK-088's CEM."""
    from embodied_jepa import zero_shot_grasp_runtime as zgr

    pm = planner_model(model)
    return zgr.ChangeController(name, pm, enc, episode_seed, SETTING)


# ----- featurisation of a new collection (§2.2, pass B only) -------------------------------------
def featurise_split(corpus, episodes, split, out, featuriser, projection, workers, log=print):
    frames = gd.episode_frames(corpus, episodes)
    writer = gd.SplitWriter(out, split, sum(frames))
    for i, (episode, (shard, eid, seed)) in enumerate(
        zip(gd.iter_episodes(corpus, episodes, workers=workers), episodes, strict=True)
    ):
        if episode["episode_id"] != eid:
            raise ContractError("episode order")
        latents = gd.project(featuriser.pooled(episode["frames"]), projection)
        writer.add(episode, latents, shard=shard, seed=seed)
        if i % 200 == 0:
            log(f"{split}: {i + 1}/{len(episodes)} episodes")
    return writer.close()


def featuriser_and_projection(device="cuda"):
    featuriser = gd.Featuriser(device=device)
    if featuriser.digest != ENCODER_DIGEST:
        raise ContractError("another encoder than TASK-087's")
    projection = gd.load_projection(V1_FEATURES / "projection.npz", PROJECTION_SHA256)
    return featuriser, projection


# ----- predictions (Stage E, A′) -----------------------------------------------------------------
@torch.no_grad()
def rollout_numpy(model, z0, actions, s0, device="cuda"):
    """Roll out ``[N, H, 14]`` commands from ``[N, 16, K]`` latents (and ``[N, 28]`` raw states):
    returns predicted latents ``[N, H, 16, K]`` and, for G, standardised states ``[N, H, 28]``."""
    model.eval()
    z = torch.as_tensor(np.asarray(z0, np.float32), device=device)
    a = torch.as_tensor(np.asarray(actions, np.float32), device=device)
    s = None
    if model.flags["state_token"] or model.flags["fusion"]:
        s = torch.as_tensor(np.asarray(s0, np.float32), device=device)
    pz, ps = model.rollout(z, a, s)
    return pz.float().cpu().numpy(), None if ps is None else ps.float().cpu().numpy()


def norm_state(model, states, device="cuda") -> np.ndarray:
    with torch.no_grad():
        s = torch.as_tensor(np.asarray(states, np.float32), device=device)
        return model.norm_state(s).cpu().numpy()
