"""TASK-086: convert open real G1 + Dex3 teleoperation data to ``ee_delta_grasp_v0``.

Protocol: ``docs/experiments/real_g1_dex3_prestep.md``. Joint names are mapped by name; joints are
resampled to 20 Hz by linear interpolation; palm deltas come from our G1 MJCF's forward kinematics
of the measured state; the grasp is the projection of the 7 named Dex3 joints onto our
open-to-closed synergy. Out-of-range steps are flagged and cut, never clipped.

NumPy only at import; MuJoCo (forward kinematics), PyAV (video) and PyArrow are imported lazily.
Converted real teleoperation is task demonstration data, not play, and not a learned result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TARGET_FPS = 20
SPLIT_SALT = 8603
MIN_SEGMENT = 20
INVALID_MARGIN_RAD = 0.1
CAMERA = "real_head_rgb"
IMAGE_SIZE = 112
RANGE_BAR = 0.5  # R-RANGE: pooled fraction of steps with any arm |a| > 1
GRASP_RESIDUAL_BAR = 0.15  # rad, median RMS residual per hand
GRASP_OUTSIDE = (-0.25, 1.25)  # synergy position t
GRASP_OUTSIDE_BAR = 0.25
ARM_PARTS = (
    ("ShoulderPitch", "shoulder_pitch"),
    ("ShoulderRoll", "shoulder_roll"),
    ("ShoulderYaw", "shoulder_yaw"),
    ("Elbow", "elbow"),
    ("WristRoll", "wrist_roll"),
    ("WristPitch", "wrist_pitch"),
    ("WristYaw", "wrist_yaw"),
)
HAND_PARTS = ("Thumb0", "Thumb1", "Thumb2", "Index0", "Index1", "Middle0", "Middle1")
FINGERS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")


def unitree_name_map() -> dict[str, str]:
    """Unitree ``kRightShoulderPitch``-style names to our MJCF joint names."""
    out = {}
    for side, low in (("Left", "left"), ("Right", "right")):
        for k, ours in ARM_PARTS:
            out[f"k{side}{k}"] = f"{low}_{ours}_joint"
        for part, finger in zip(HAND_PARTS, FINGERS, strict=True):
            out[f"k{side}Hand{part}"] = f"{low}_hand_{finger}_joint"
    return out


def map_names(names, *, provider: str) -> list[str | None]:
    if provider == "unitree":
        table = unitree_name_map()
        return [table.get(n) for n in names]
    if provider == "nvidia":  # our names; legs are ignored (our pelvis is fixed)
        return [
            None if n is None or ("_hip_" in n or "_knee_" in n or "_ankle_" in n) else n
            for n in names
        ]
    raise ValueError(f"unknown provider {provider}")


# ----- resampling -----------------------------------------------------------------------------
def resample(q: np.ndarray, valid: np.ndarray, source_fps: float):
    """Linearly interpolate joints at t = k / 20; a frame is invalid if either source frame is.

    Returns (q20, valid20, image_index): image_index is the nearest source frame,
    floor(s + 0.5) with s = k * source_fps / 20."""
    n = len(q)
    q = np.where(np.isfinite(q), q, 0.0)  # invalid frames are masked below; keep NaN out
    ratio = source_fps / TARGET_FPS
    count = int(np.floor((n - 1) / ratio)) + 1
    s = np.arange(count) * ratio
    lo = np.floor(s).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (s - lo)[:, None]
    q20 = (1 - w) * q[lo] + w * q[hi]
    exact = np.isclose(s, lo)
    valid20 = valid[lo] & (valid[hi] | exact)
    image_index = np.minimum(np.floor(s + 0.5).astype(int), n - 1)
    return q20, valid20, image_index


# ----- kinematics -------------------------------------------------------------------------------
def rpy_from_matrix(m: np.ndarray) -> np.ndarray:
    """r with m = rotation_delta(r) = Rz(yaw) Ry(pitch) Rx(roll) (``embodiment.rotation_delta``)."""
    pitch = np.arcsin(np.clip(-m[..., 2, 0], -1.0, 1.0))
    roll = np.arctan2(m[..., 2, 1], m[..., 2, 2])
    yaw = np.arctan2(m[..., 1, 0], m[..., 0, 0])
    return np.stack([roll, pitch, yaw], axis=-1)


class Kinematics:
    """Our fixed-pelvis G1 MJCF: named joint positions -> palm poses in the pelvis frame."""

    def __init__(self):
        from embodied_jepa.simulation import MuJoCoSimulation

        self.sim = MuJoCoSimulation(render=False)
        self.mj, self.model = self.sim.mj, self.sim.model
        self.data = self.mj.MjData(self.model)
        self.joint_names = self.sim.joint_names  # actuated, in actuator order
        self.qadr = {n: int(self.model.joint(n).qposadr[0]) for n in self.joint_names}
        self.range = {n: self.model.joint(n).range.copy() for n in self.joint_names}
        self.site = {s: self.model.site(f"{s}_ee").id for s in ("left", "right")}
        self.pelvis = self.model.body("pelvis").id
        manifest = json.loads((ROOT / "configs/g1_sim_action.json").read_text())
        self.manifest = manifest
        self.open = {s: np.asarray(manifest[f"{s}_open_rad"]) for s in ("left", "right")}
        self.closed = {s: np.asarray(manifest[f"{s}_closed_rad"]) for s in ("left", "right")}
        if tuple(manifest["grasp_fingers"]) != FINGERS:
            raise ValueError("synergy finger order differs from the converter's")

    def invalid(self, names, q: np.ndarray) -> np.ndarray:
        """Frames where a mapped joint is non-finite or > 0.1 rad outside its MJCF range."""
        bad = ~np.isfinite(q).all(axis=1)
        for j, name in enumerate(names):
            if name is None or name not in self.range:
                continue
            lo, hi = self.range[name]
            bad |= (q[:, j] < lo - INVALID_MARGIN_RAD) | (q[:, j] > hi + INVALID_MARGIN_RAD)
        return bad

    def palms(self, names, q: np.ndarray):
        """Palm positions [T, 2, 3] and rotations [T, 2, 3, 3] (left, right) in the pelvis frame."""
        cols = [(j, self.qadr[n]) for j, n in enumerate(names) if n in self.qadr]
        pos = np.zeros((len(q), 2, 3))
        rot = np.zeros((len(q), 2, 3, 3))
        d = self.data
        for t in range(len(q)):
            d.qpos[:] = self.model.qpos0
            for j, adr in cols:
                d.qpos[adr] = q[t, j]
            self.mj.mj_kinematics(self.model, d)
            base_r = d.xmat[self.pelvis].reshape(3, 3)
            base_p = d.xpos[self.pelvis]
            for i, side in enumerate(("left", "right")):
                sid = self.site[side]
                pos[t, i] = base_r.T @ (d.site_xpos[sid] - base_p)
                rot[t, i] = base_r.T @ d.site_xmat[sid].reshape(3, 3)
        return pos, rot

    def grasp(self, names, q: np.ndarray, side: str):
        """Synergy position t, grasp 2t - 1 and the RMS residual (rad) per frame."""
        idx = [names.index(f"{side}_hand_{f}_joint") for f in FINGERS]
        h = q[:, idx]
        o, c = self.open[side], self.closed[side]
        d = c - o
        t = ((h - o) @ d) / (d @ d)
        residual = h - (o + t[:, None] * d)
        return t, 2 * t - 1, np.sqrt((residual**2).mean(axis=1))

    def state(self, names, q20: np.ndarray):
        """Our 43-joint positions and velocities (finite differences at 20 Hz) with a mask."""
        n = len(self.joint_names)
        position = np.zeros((len(q20), n))
        have = np.zeros(n, bool)
        for k, name in enumerate(self.joint_names):
            if name in names:
                position[:, k] = q20[:, names.index(name)]
                have[k] = True
        velocity = np.zeros_like(position)
        if len(q20) > 1:
            velocity[:-1] = np.diff(position, axis=0) * TARGET_FPS
            velocity[-1] = velocity[-2]
        values = np.concatenate([position, velocity], axis=1).astype(np.float32)
        mask = np.tile(np.concatenate([have, have]), (len(q20), 1))
        return values, mask


# ----- one episode ------------------------------------------------------------------------------
@dataclass
class Converted:
    actions: np.ndarray  # [K-1, 14] normalized (may lie outside [-1, 1])
    valid: np.ndarray  # [K] frame validity
    flagged: np.ndarray  # [K-1] step flags
    arm_out: np.ndarray  # [K-1] any arm |a| > 1
    grasp_out: np.ndarray  # [K-1] any grasp outside [-1, 1]
    t: np.ndarray  # [K, 2] synergy position (left, right)
    residual: np.ndarray  # [K, 2] RMS residual (rad)
    q20: np.ndarray  # [K, J] resampled source joints (columns as ``names``)
    names: list
    image_index: np.ndarray

    def state(self, kin, start: int, end: int):
        """State for frames start..end; velocities by finite differences inside the slice."""
        return kin.state(self.names, self.q20[start : end + 1])


def convert_episode(kin: Kinematics, names, q: np.ndarray, source_fps: float) -> Converted:
    """Convert one source episode's joint matrix (columns named by ``names``, ours or None)."""
    q = np.asarray(q, float)
    bad = kin.invalid(names, q)
    q20, valid, image_index = resample(q, ~bad, source_fps)
    known = [n if n is not None else "" for n in names]
    pos, rot = kin.palms(known, q20)
    tr = kin.manifest["translation_per_step_m"]
    rr = kin.manifest["rotation_per_step_rad"]
    k = len(q20)
    actions = np.zeros((max(k - 1, 0), 14))
    for i in range(2):  # left = 0, right = 1 (schema order: left 0-5, right 6-11)
        dp = (pos[1:, i] - pos[:-1, i]) / tr
        m = rot[1:, i] @ np.swapaxes(rot[:-1, i], -1, -2)
        actions[:, 6 * i : 6 * i + 3] = dp
        actions[:, 6 * i + 3 : 6 * i + 6] = rpy_from_matrix(m) / rr
    t = np.zeros((k, 2))
    residual = np.zeros((k, 2))
    for i, side in enumerate(("left", "right")):
        ti, g, res = kin.grasp(known, q20, side)
        t[:, i], residual[:, i] = ti, res
        actions[:, 12 + i] = g[1:]
    arm_out = (np.abs(actions[:, :12]) > 1).any(axis=1)
    grasp_out = (np.abs(actions[:, 12:]) > 1).any(axis=1)
    frames_ok = valid[:-1] & valid[1:]
    flagged = arm_out | grasp_out | ~frames_ok
    return Converted(
        actions, valid, flagged, arm_out, grasp_out, t, residual, q20, known, image_index
    )


def segments(flagged: np.ndarray, minimum: int = MIN_SEGMENT) -> list[tuple[int, int]]:
    """Runs [start, end) of unflagged steps with at least ``minimum`` steps."""
    out, start = [], None
    for i, f in enumerate(list(flagged) + [True]):
        if not f and start is None:
            start = i
        elif f and start is not None:
            if i - start >= minimum:
                out.append((start, i))
            start = None
    return out


# ----- statistics and the row (protocol §5) -----------------------------------------------------
def episode_stats(c: Converted) -> dict:
    ok = c.valid[:-1] & c.valid[1:]
    return {
        "steps": int(len(c.actions)),
        "valid_steps": int(ok.sum()),
        "arm_out": int((c.arm_out & ok).sum()),
        "grasp_out": int((c.grasp_out & ok).sum()),
        "invalid_frames": int((~c.valid).sum()),
        "kept_steps": int(sum(e - s for s, e in segments(c.flagged))),
    }


def decide(pooled: dict, *, void: str = "") -> dict:
    """R-VOID, R-DROP-RANGE, R-DROP-GRASP or R-KEEP (first match)."""
    if not void and (
        pooled["valid_steps"] == 0
        or not all(np.isfinite(pooled[f"{s}_median_residual"]) for s in ("left", "right"))
    ):
        void = "nothing measured"
    range_fraction = pooled["arm_out"] / max(pooled["valid_steps"], 1)
    poor = {}
    for side in ("left", "right"):
        median = pooled[f"{side}_median_residual"]
        outside = pooled[f"{side}_outside_fraction"]
        poor[side] = bool(median > GRASP_RESIDUAL_BAR or outside > GRASP_OUTSIDE_BAR)
    if void:
        row = "R-VOID"
    elif range_fraction > RANGE_BAR:
        row = "R-DROP-RANGE"
    elif any(poor.values()):
        row = "R-DROP-GRASP"
    else:
        row = "R-KEEP"
    return {"row": row, "void": void, "range_fraction": range_fraction, "grasp_poor": poor}


def split_for(repo: str, keys: list[str]) -> dict[str, list[str]]:
    ranked = sorted(
        keys, key=lambda k: hashlib.sha256(f"{SPLIT_SALT}:{repo}:{k}".encode()).digest()
    )
    n = len(ranked)
    m = max(1, n // 20)
    return {
        "val": sorted(ranked[:m]),
        "test": sorted(ranked[m : 2 * m]),
        "train": sorted(ranked[2 * m :]),
    }


def square_resize(frame: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Centre crop to a square and resize (Lanczos) to ``size``."""
    from PIL import Image

    h, w = frame.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    crop = frame[top : top + side, left : left + side]
    return np.asarray(Image.fromarray(crop).resize((size, size), Image.Resampling.LANCZOS))
