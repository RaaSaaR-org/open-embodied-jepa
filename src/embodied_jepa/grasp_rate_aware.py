"""TASK-092: a rate-aware grounded world model (G-req) -- the grasp-rate augmentation, Stage R's
roots, measures and bars, the R, A‴ and B rows, the cohorts and seeds, and the closed loop's
reported rate measures (protocol ``docs/experiments/grasp_rate_aware_wm_v1.md``).

* The augmentation (§4.1): a step is *at the limit* when its executed right grasp moved by
  +-2/11 (within 1e-4) from the previous executed grasp; with probability p = 0.5 per window
  every at-limit step's model input is replaced by a request drawn uniformly from the requests
  the embodiment's rate clip maps onto the executed grasp. The draws use their own generator,
  ``default_rng(SeedSequence([9008, m]))``; TASK-087's window sampler and initialisation are
  untouched, and with p = 0 the inputs equal the executed commands bitwise.
* Stage R (§6.2): R-roots, the saturated and random execution-identical requests (salt 9009), the
  ratio-of-sums measures, δ from the control seeds, R1's bar, R2's bars, E's checks.
* The rows of §6.4 and §7, the power recompute (§6.4) and the reported rate measures of the
  closed loop (§1.2's reading, from a run's ``plan_log``).

TASK-087's ``grounded_wm``, TASK-090's ``grasp_corpus_wm`` and TASK-091's
``grasp_corpus_wm_planned_close`` are imported and never changed. NumPy only at import. Simulator
truth never enters the augmentation, a model input or a planning cost. Nothing here is registered
or imported by ``embodied_jepa``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from embodied_jepa import grasp_corpus_wm as gc
from embodied_jepa import grasp_corpus_wm_planned_close as pcl
from embodied_jepa import grounded_wm as gw
from embodied_jepa import zero_shot as zs
from embodied_jepa import zero_shot_grasp as zg
from embodied_jepa.contracts import ContractError

TASK = "TASK-092"
PROTOCOL = "docs/experiments/grasp_rate_aware_wm_v1.md"

# ----- the grasp channel and its rate clip (§1.1, §4.1) ----------------------------------------
GRASP_DIM = 13  # the right-grasp channel of ee_delta_grasp_v0
RATE_LIMIT = 2.0 / 11.0  # joint_speed_limit_rad_s x control_dt / max slope = 2.0 x 0.05 / 0.55
AT_LIMIT_TOL = 1e-4
AUG_P = 0.5  # declared, not tuned (§4.1)
WINDOW_COMMANDS = gw.TRAIN_HORIZON  # 8


def rate_limit_from_manifest(manifest: dict) -> float:
    """The clip's half-width from ``configs/g1_sim_action.json``: the joint-rate step divided by
    the steepest synergy slope of the right hand (``embodiment._prepare_side``)."""
    opened = np.asarray(manifest["right_open_rad"], np.float64)
    closed = np.asarray(manifest["right_closed_rad"], np.float64)
    slope = np.abs((closed - opened) / 2).max()
    return float(manifest["joint_speed_limit_rad_s"] * manifest["control_dt_s"] / slope)


# ----- seeds and salts (§9) --------------------------------------------------------------------
SALTS = {"augment": 9008, "random_equivalent": 9009}
MODEL_SEEDS = gw.MODEL_SEEDS  # 87100-87102, TASK-087's, reused on purpose for pairing (§4.2)
PAIRING_SEED = gw.PROBE_SEED  # 87900, TASK-087's debug range (never a result)
PAIRING_UPDATES = 300
HISTORICAL_PAIRING_SEED = 87100
HISTORICAL_PAIRING_UPDATES = 3000  # the first selection point (§13)
DEBUG_MODEL_SEEDS = gw.DEBUG_SEEDS  # 87900-87999, Stage 0 only
DEBUG_SEEDS = range(90500, 90600)  # Stage 0 plumbing and smoke; never counted
COHORTS = {
    "debug-roots": range(90500, 90540),
    "debug-grasp": range(90540, 90600),
    "a3-roots": range(908000, 908064),
    "bd-grasp": range(90400, 90432),
    "bs-grasp": range(909000, 909128),
}
ROOT_COHORTS = ("debug-roots", "a3-roots")
GRASP_COHORTS = ("debug-grasp", "bd-grasp", "bs-grasp")
BS_N = len(COHORTS["bs-grasp"])  # 128 (§6.4's power rule)
DETERMINISM = ("M-req", 87100, 4)  # 909000-909003, inside BS (§6.4)
# Ranges of earlier tasks TASK-092 must not touch (§9)
EARLIER_RANGES = dict(pcl.TASK090_RANGES) | {f"TASK-091 {k}": v for k, v in pcl.COHORTS.items()}

# ----- arms (§5) -------------------------------------------------------------------------------
CLOSED_LOOP_ARMS = {"M-req": "G-req", "M-v1": "G-v1"}  # arm -> model family
BD_SEEDS = {"M-req": (87100,), "M-v1": (87100,)}
BS_SEEDS = {"M-req": MODEL_SEEDS, "M-v1": MODEL_SEEDS}
BASELINE_ARMS = zs.BASELINE_ARMS  # hold, random, ik, ik-nopress
JOB_CAP_SECONDS = pcl.JOB_CAP_SECONDS  # 7 200 s per GPU job
SPLIT_TRIGGER_SECONDS = pcl.SPLIT_TRIGGER_SECONDS  # 5 400 s (§13)
BS_PARTS = 2  # 909000-909063 and 909064-909127 (§6.4; Stage 0 may raise it)
LAMBDA = pcl.LAMBDA  # 2.0552, TASK-089's step-averaged cost


def check_seed(cohort: str, seed: int) -> None:
    if cohort not in COHORTS:
        raise ContractError(f"unknown cohort {cohort}")
    if int(seed) not in COHORTS[cohort]:
        raise ContractError(f"seed {seed} is not in cohort {cohort}")


def check_model_seed(seed: int, *, debug: bool = False) -> int:
    """G-req's model seeds: TASK-087's three, or its debug range (Stage 0's smoke only)."""
    seed = int(seed)
    if seed not in (DEBUG_MODEL_SEEDS if debug else MODEL_SEEDS):
        raise ContractError(f"{seed} is not a {'debug' if debug else 'model'} seed of {TASK}")
    return seed


def check_run(cohort: str, arm: str, model_seed, *, debug_req_seed: int | None = None) -> None:
    """§6.4: which (arm, model seed) a closed-loop cohort admits. A debug cohort also admits a
    debug G-req model (``debug_req_seed``, from TASK-087's debug range)."""
    if cohort not in GRASP_COHORTS:
        raise ContractError(f"{cohort} is not a closed-loop cohort")
    if arm in BASELINE_ARMS:
        if model_seed is not None:
            raise ContractError("baseline arms take no model seed")
        return
    if arm not in CLOSED_LOOP_ARMS:
        raise ContractError(f"unknown arm {arm}")
    if cohort == "debug-grasp":
        allowed = tuple(BS_SEEDS[arm])
        if arm == "M-req" and debug_req_seed is not None:
            allowed += (int(debug_req_seed),)
    elif cohort == "bd-grasp":
        allowed = BD_SEEDS[arm]
    else:
        allowed = BS_SEEDS[arm]
    if model_seed not in allowed:
        raise ContractError(f"{arm}-{model_seed} is not a run of {cohort}")


def needs_parts(cohort: str, arm: str) -> int | None:
    """BS world-model runs run as BS_PARTS jobs; everything else as one job."""
    return BS_PARTS if cohort == "bs-grasp" and arm in CLOSED_LOOP_ARMS else None


def part_ranges(cohort: str, parts: int) -> list[range]:
    seeds = COHORTS[cohort]
    n = len(seeds)
    if parts < 1 or parts > n:
        raise ContractError(f"{parts} parts of {n} resets")
    size = math.ceil(n / parts)
    return [seeds[i * size : min(n, (i + 1) * size)] for i in range(parts)]


split_goals = pcl.split_goals
parse_part = pcl.parse_part
combine_parts = pcl.combine_parts
project_job_seconds = pcl.project_job_seconds


def parts_needed(job_seconds_full: float, resets: int = BS_N, parts: int = BS_PARTS) -> int:
    return pcl.parts_needed(job_seconds_full, resets, parts)


# ----- pins (§12, §13) -------------------------------------------------------------------------
# TASK-087's checkpoints the task reads (M-v1's three G and E (iii)'s N-87100), its feature
# reports and the moments; TASK-090's six G models (reported only in Stage R).
CHECKPOINT_SHA256 = {
    "G-v1-87100": pcl.CHECKPOINT_SHA256["G-v1-87100"],
    "G-v1-87101": pcl.CHECKPOINT_SHA256["G-v1-87101"],
    "G-v1-87102": pcl.CHECKPOINT_SHA256["G-v1-87102"],
    "N-v1-87100": "52ec73a60d211a65803783ecc10601a8eb7784a5578d72eb431a3fc8762fec63",
} | {k: v for k, v in pcl.CHECKPOINT_SHA256.items() if k.startswith(("G-v2g", "G-v1p"))}
FEATURE_REPORT_SHA256 = pcl.FEATURE_REPORTS_SHA256["play-v1"]  # f5648a35...7eae
FEATURE_REPORTS_SHA256 = pcl.FEATURE_REPORTS_SHA256  # TASK-090's six G models (reported)
TEST_REPORT_SHA256 = "a6c380ff98dd922adaf589fd4fe41f40669e32244b25959fd91e2d14e42fd650"
MOMENTS_SHA256 = pcl.MOMENTS_SHA256
IMPLEMENTATION_SHA256 = "e2ec0096c14f628c248fda7fb1a05ad2d3342fc7e20703727bcc178efd7149b1"
G87100_REPORT = "checkpoints/task087-run/G-87100/report.json"  # relative to TASK-087's worktree

# Earlier tasks' frozen code that TASK-092 imports and does not edit (§13). The runner refuses to
# start when any of these differs: TASK-087's model, data and driver code (the implementation
# hash), TASK-090's frozen files (TASK-091's pins) and TASK-091's modules and scripts.
FROZEN_FILES_SHA256 = dict(pcl.FROZEN_FILES_SHA256) | {
    "src/embodied_jepa/grounded_wm.py": (
        "16ff573344489eb69f656a36c6344136d5862c663a52904dc9d5623469513444"
    ),
    "src/embodied_jepa/grounded_wm_model.py": (
        "c66e03c31ad52780bdb5c221c0a01b6f2c253ed31c6aac7a13c4769380eef1d9"
    ),
    "src/embodied_jepa/grounded_wm_data.py": (
        "e2e2a8d61fb32c408f1346557f63a0d95e5cdd43afd80db88ba22502f202f1d8"
    ),
    "src/embodied_jepa/models/lewm.py": (
        "a99e000436da627916dcc8449b43f7ea77518cc99ef82824844d82ac23b200f1"
    ),
    "scripts/train_task087.py": "1a24bca49e26b3cf135e9f9b61db88c8bdba7a5e62a69b63e90c9cad6a1fa905",
    "src/embodied_jepa/grasp_corpus_wm_planned_close.py": (
        "08a993906542046648a99e24069eef8be7ca6ea408700e875487385b0038d433"
    ),
    "src/embodied_jepa/grasp_corpus_wm_planned_close_run.py": (
        "2ab6cccaa93e2800997742790f3bd0d9e50a368c898e266b3066b3d4c985977a"
    ),
    "src/embodied_jepa/grasp_corpus_wm_planned_close_runtime.py": (
        "de2900d5ec4a28e44e4508a7dcd5bc746deabe5f99cffa78fe9f761a890fea66"
    ),
    "scripts/run_task091.py": "28dbe36b0a7b6248beda2882f3f5bf19b2fb4bf7f6176701c02e65de858f80b3",
    "scripts/summarize_task091.py": (
        "ff779b364e45fcbbf3db75b7eceae6d0d1701ba20f4a77ee95b11743e78a839c"
    ),
}
# TASK-092's own modules: a G-req checkpoint records their sha256 (§6.1) and its loader checks it
OWN_MODULES = (
    "src/embodied_jepa/grasp_rate_aware.py",
    "src/embodied_jepa/grasp_rate_aware_runtime.py",
)


def frozen_files_check(root) -> dict:
    out = {}
    for rel, pin in FROZEN_FILES_SHA256.items():
        path = Path(root) / rel
        out[rel] = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == pin
    return out


def own_modules_sha256(root) -> dict:
    return {rel: hashlib.sha256((Path(root) / rel).read_bytes()).hexdigest() for rel in OWN_MODULES}


def model_name(family: str, seed: int) -> str:
    """``G-req-87100``, ``G-v1-87100``."""
    return f"{family}-{int(seed)}"


def stage_r_models() -> list[tuple[str, int]]:
    """§6.2: the six models gated on (G-req and its paired control per seed)."""
    return [(f, s) for f in ("G-req", "G-v1") for s in MODEL_SEEDS]


def a3_models() -> list[tuple[str, int]]:
    """§6.3: the same six models, in this order."""
    return stage_r_models()


# ----- the augmentation (§4.1) -----------------------------------------------------------------
def at_limit(prev, executed, *, limit: float = RATE_LIMIT, tol: float = AT_LIMIT_TOL):
    """``prev`` [B] (the executed grasp of the command before the window; NaN when the window
    starts at an episode's first command) and ``executed`` [B, K] grasps -> (up, down) [B, K]:
    step k moved by +limit (up) or -limit (down) within ``tol`` from step k - 1."""
    prev = np.asarray(prev, np.float64)
    e = np.asarray(executed, np.float64)
    if e.ndim != 2 or prev.shape != (e.shape[0],):
        raise ContractError("at_limit needs prev [B] and executed [B, K]")
    d = np.diff(np.concatenate([prev[:, None], e], 1), axis=1)
    with np.errstate(invalid="ignore"):
        up = np.abs(d - limit) < tol
        down = np.abs(d + limit) < tol
    return up, down


def equivalent_interval(previous, up, down, *, limit: float = RATE_LIMIT):
    """The requests the rate clip maps onto an at-limit step: ``[previous + limit, 1]`` upward,
    ``[-1, previous - limit]`` downward (NaN elsewhere). ``previous`` [B, K] is the executed grasp
    of each step's predecessor."""
    previous = np.asarray(previous, np.float64)
    lo = np.where(up, previous + limit, np.where(down, -1.0, np.nan))
    hi = np.where(up, 1.0, np.where(down, previous - limit, np.nan))
    return lo, hi


def rate_clip(request, previous, *, limit: float = RATE_LIMIT):
    """The embodiment's grasp rate clip relative to the previous executed grasp, in [-1, 1]."""
    r = np.clip(np.asarray(request, np.float64), -1.0, 1.0)
    p = np.asarray(previous, np.float64)
    return np.clip(r, p - limit, p + limit)


def augmenter_seed(model_seed: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([SALTS["augment"], int(model_seed)])


def augment_actions(actions, prev, u_window, u_steps, p: float = AUG_P):
    """One batch's augmented model inputs. ``actions`` [B, K, 14] executed commands (float32),
    ``prev`` [B] the executed grasp before each window (NaN at an episode's first command),
    ``u_window`` [B] and ``u_steps`` [B, K] uniform draws. Returns (inputs, replaced [B, K],
    selected [B], at-limit [B, K]); inputs equal ``actions`` bitwise except index 13 of the
    replaced steps."""
    actions = np.asarray(actions)
    b, k = actions.shape[:2]
    grasp = actions[:, :, GRASP_DIM].astype(np.float64)
    up, down = at_limit(prev, grasp)
    previous = np.concatenate([np.asarray(prev, np.float64)[:, None], grasp[:, :-1]], 1)
    lo, hi = equivalent_interval(previous, up, down)
    selected = np.asarray(u_window, np.float64) < float(p)
    replaced = selected[:, None] & (up | down)
    draw = lo + np.asarray(u_steps, np.float64) * (hi - lo)
    out = np.array(actions, copy=True)
    if replaced.any():
        g = out[:, :, GRASP_DIM]
        g[replaced] = draw[replaced].astype(out.dtype)
        out[:, :, GRASP_DIM] = g
    if out.shape != (b, k, actions.shape[2]):
        raise ContractError("augment_actions changed the batch shape")
    return out, replaced, selected, up | down


class Augmenter:
    """§4.1's draws for one model seed: per batch, ``u_window`` [B] then ``u_steps`` [B, K] from
    ``default_rng(SeedSequence([9008, m]))``, whatever p is (so p changes which steps are
    replaced, never the stream). Counts windows and steps for the Stage T report."""

    def __init__(self, model_seed: int, p: float = AUG_P):
        if not 0.0 <= float(p) <= 1.0:
            raise ContractError("p must be in [0, 1]")
        self.p = float(p)
        self.rng = np.random.default_rng(augmenter_seed(model_seed))
        self.counts = {
            "batches": 0,
            "windows": 0,
            "windows_selected": 0,
            "windows_replaced": 0,
            "steps": 0,
            "steps_at_limit": 0,
            "steps_replaced": 0,
        }

    def __call__(self, actions, prev):
        b, k = np.asarray(actions).shape[:2]
        u_window = self.rng.random(b)
        u_steps = self.rng.random((b, k))
        out, replaced, selected, limit = augment_actions(actions, prev, u_window, u_steps, self.p)
        c = self.counts
        c["batches"] += 1
        c["windows"] += b
        c["windows_selected"] += int(selected.sum())
        c["windows_replaced"] += int(replaced.any(1).sum())
        c["steps"] += b * k
        c["steps_at_limit"] += int(limit.sum())
        c["steps_replaced"] += int(replaced.sum())
        return out


def window_prev(starts, actions_store, offsets):
    """The executed grasp of the command before each window (same episode), NaN when the window
    starts at its episode's first command."""
    starts = np.asarray(starts, np.int64)
    first = np.isin(starts, np.asarray(offsets, np.int64))
    prev = actions_store[np.maximum(starts - 1, 0), GRASP_DIM].astype(np.float64)
    return np.where(first, np.nan, prev)


# ----- Stage R (§6.2) --------------------------------------------------------------------------
R_HORIZON = 8
R_HORIZONS = (1, 4, 8)
R_LEVEL = gw.GATE_LEVEL  # 0.975
R1_FRACTION = 0.25
PREMISE_MIN_RATIO = 2.0
HAND = slice(7, 14)  # the right hand's joint positions in the 28-D state


def r_roots(starts, offsets, actions_store, horizon: int = R_HORIZON):
    """§6.2: the roots with a predecessor command in their episode and at least one at-limit
    grasp step among their ``horizon`` executed commands. Returns (mask, prev, executed
    [N, horizon, 14] float64, up, down)."""
    starts = np.asarray(starts, np.int64)
    executed = actions_store[starts[:, None] + np.arange(horizon)].astype(np.float64)
    prev = window_prev(starts, actions_store, offsets)
    up, down = at_limit(prev, executed[:, :, GRASP_DIM])
    mask = ~np.isnan(prev) & (up | down).any(1)
    return mask, prev, executed, up, down


def saturated_requests(executed, up, down):
    """§1.2: every at-limit step's request set to +-1 (executed identically)."""
    out = np.array(executed, np.float64, copy=True)
    g = out[:, :, GRASP_DIM]
    out[:, :, GRASP_DIM] = np.where(up, 1.0, np.where(down, -1.0, g))
    return out


def random_requests(executed, prev, up, down, *, salt: int = SALTS["random_equivalent"]):
    """§1.2: every at-limit step's request drawn uniformly from its equivalence set (salt 9009,
    one draw per root and step, in root order)."""
    out = np.array(executed, np.float64, copy=True)
    g = out[:, :, GRASP_DIM]
    previous = np.concatenate([np.asarray(prev, np.float64)[:, None], g[:, :-1]], 1)
    lo, hi = equivalent_interval(previous, up, down)
    u = np.random.default_rng(np.random.SeedSequence([int(salt)])).random(g.shape)
    out[:, :, GRASP_DIM] = np.where(up | down, lo + u * (hi - lo), g)
    return out


def executed_identically(requests, executed, prev, *, tol: float = AT_LIMIT_TOL) -> bool:
    """The rate clip maps every request sequence back onto the executed grasps (Stage R's
    equivalence check)."""
    g = np.asarray(executed, np.float64)[:, :, GRASP_DIM]
    previous = np.concatenate([np.asarray(prev, np.float64)[:, None], g[:, :-1]], 1)
    clipped = rate_clip(np.asarray(requests, np.float64)[:, :, GRASP_DIM], previous)
    same_other = np.array_equal(
        np.delete(np.asarray(requests, np.float64), GRASP_DIM, 2),
        np.delete(np.asarray(executed, np.float64), GRASP_DIM, 2),
    )
    return bool(same_other and np.all(np.abs(clipped - g) <= tol + 1e-9))


def hand_mse(predicted_std, measured_std) -> np.ndarray:
    """Per-root standardised right-hand joint-position MSE (state dimensions 7-13)."""
    p = np.asarray(predicted_std, np.float64)[:, HAND]
    m = np.asarray(measured_std, np.float64)[:, HAND]
    return ((p - m) ** 2).mean(1)


def ratio(num, den, weights, level: float = R_LEVEL) -> dict:
    return gw.ratio_stat(num, den, weights, level)


def delta(per_seed: dict, weights, level: float = R_LEVEL) -> dict:
    """§6.2's δ: the largest, over ordered pairs (i, j), i != j, of the upper bound of
    X(G-i) / X(G-j), minus 1 (0 if negative). ``per_seed[seed]`` per-root values."""
    bounds = {}
    for i in per_seed:
        for j in per_seed:
            if i != j:
                bounds[f"{i}/{j}"] = ratio(per_seed[i], per_seed[j], weights, level)["ci"][1]
    return {"delta": max(0.0, max(bounds.values()) - 1.0), "upper_bounds": bounds}


def r1_bar(rho_ctrl: float) -> float:
    return 1.0 + R1_FRACTION * (float(rho_ctrl) - 1.0)


def r_seed(
    *, h_exec_req, h_sat_req, h_exec_ctrl, h_sat_ctrl, l_exec_req, l_exec_ctrl,
    w_r, w_all, delta_l: float, delta_h: float,
) -> dict:  # fmt: skip
    """R1 and R2 for one seed (G-req-i against G-i, per-root values at h = 8)."""
    ctrl = ratio(h_sat_ctrl, h_exec_ctrl, w_r)
    rho_ctrl = ctrl["value"]
    bar = r1_bar(rho_ctrl)
    req = ratio(h_sat_req, h_exec_req, w_r)
    l2 = ratio(l_exec_req, l_exec_ctrl, w_all)
    h2 = ratio(h_exec_req, h_exec_ctrl, w_r)
    out = {
        "rho_ctrl": ctrl,
        "premise": bool(rho_ctrl >= PREMISE_MIN_RATIO),
        "r1_bar": bar,
        "rho_req": req,
        "r1": bool(req["ci"][1] <= bar),
        "r2_latent": l2,
        "r2_latent_bar": 1.0 + delta_l,
        "r2_hand": h2,
        "r2_hand_bar": 1.0 + delta_h,
        "r2": bool(l2["ci"][1] <= 1.0 + delta_l and h2["ci"][1] <= 1.0 + delta_h),
        "sat_req_over_exec_ctrl": ratio(h_sat_req, h_exec_ctrl, w_r),  # reported
    }
    out["r_pass"] = out["r1"] and out["r2"]
    return out


def e_seed(stats: dict, vs_n: dict | None) -> dict:
    """E's validity checks (TASK-090 §6.1's, ``gc.e_checks``) at h = 4 and 8."""
    return gc.e_checks(stats, vs_n)


def r_row(per_seed: dict, *, void: str = "") -> dict:
    """§7 after Stage R (first match): R-VOID, R-PREMISE (a control's ratio below 2), E-FAIL (a
    G-req seed fails E), R-FAIL (two or more G-req seeds fail R1 or R2), else R-PASS (Stage A‴
    runs). ``per_seed[seed]`` holds ``r_seed``'s output plus ``e`` (``e_seed``'s)."""
    premise_bad = sorted(int(s) for s, v in per_seed.items() if not v["premise"])
    e_bad = sorted(int(s) for s, v in per_seed.items() if not v["e"]["pass"])
    r_bad = sorted(int(s) for s, v in per_seed.items() if not v["r_pass"])
    if void:
        row = "R-VOID"
    elif premise_bad:
        row = "R-PREMISE"
    elif e_bad:
        row = "E-FAIL"
    elif len(r_bad) >= 2:
        row = "R-FAIL"
    else:
        row = "R-PASS"
    return {
        "row": row,
        "void_reason": void,
        "premise_failing_seeds": premise_bad,
        "e_failing_seeds": e_bad,
        "r_failing_seeds": r_bad,
        "proceeds_to_a3": row == "R-PASS",
    }


# ----- Stage A‴ (§6.3) -------------------------------------------------------------------------
A3_BAR = pcl.A2_BAR  # 48 of 64
A3_H = pcl.A_SECOND_H  # 8
A3_SETTING = pcl.A_SECOND_SETTING  # step-mean
ROOT_GRASP = {"R1": -1.0, "R2": zs.GRASP_CLOSURE}  # the demonstrator's grasp before each root


a2_third = pcl.a2_second  # close≤4@R1 and lift@R2 (TASK-091's measure, unchanged)
a3_pass = pcl.a_second_pass
first_command_equal = pcl.first_command_equal


def executed_plan_grasp(plan, root_grasp: float) -> list[float]:
    """§6.3, reported: the plan's requests passed through the rate clip from the root's grasp."""
    g = np.asarray(plan, np.float64)[:, GRASP_DIM]
    out, prev = [], float(root_grasp)
    for r in g:
        prev = float(rate_clip(r, prev))
        out.append(prev)
    return out


def a3_row(r_pass: dict, a3: dict, *, void: str = "") -> dict:
    """§7 after Stage A‴: A‴-VOID, else A‴-FAIL when two or more G-req seeds fail R1, R2 or A‴
    (a seed failing any of them counts once), else A‴-PASS (Stage BD runs)."""
    failing = sorted(int(s) for s in r_pass if not (r_pass[s] and a3.get(s, False)))
    if void:
        row = "A‴-VOID"
    elif len(failing) >= 2:
        row = "A‴-FAIL"
    else:
        row = "A‴-PASS"
    return {"row": row, "void_reason": void, "failing_seeds": failing}


# ----- the closed loop (§6.4) ------------------------------------------------------------------
P_THRESHOLD = gc.P_THRESHOLD  # 0.01
DELTA = gc.DELTA  # +0.10
SEEDS_PER_ARM = gc.SEEDS_PER_ARM
b_bar = zg.b_bar  # min(0.40, 0.9 r); None (escalate) when r < 0.30
pooled_bar = gc.pooled_bar
seed_floor = gc.seed_floor
signflip_p = gc.signflip_p
per_reset = gc.per_reset
reset_bootstrap = gc.reset_bootstrap
pooled_mcnemar = gc.pooled_mcnemar


def b_row(g_by_seed, c_by_seed, random, bar: float, *, void: str = "") -> dict:
    """§6.4's rows, first match. ``g_by_seed`` / ``c_by_seed``: M-req's and M-v1's ``[3, n]``
    success indicators on the same resets (seed i of each on reset r); ``random``: ``[n]``."""
    g = np.asarray(g_by_seed, bool)
    c = np.asarray(c_by_seed, bool)
    r = np.asarray(random, bool)
    if g.shape != c.shape or g.shape[0] != SEEDS_PER_ARM or r.shape != (g.shape[1],):
        raise ContractError("b_row needs [3, n] arms and [n] random")
    n = g.shape[1]
    gr_, cr_ = per_reset(g), per_reset(c)
    k_g, k_c = int(gr_.sum()), int(cr_.sum())
    need, floor = pooled_bar(bar, n), seed_floor(bar, n)
    seeds = [int(x) for x in g.sum(1)]
    p_c = signflip_p(gr_ - cr_)
    p_r = signflip_p(gr_ - SEEDS_PER_ARM * r.astype(np.int64))
    beats_c, beats_r = p_c < P_THRESHOLD, p_r < P_THRESHOLD
    at_bar = k_g >= need
    floor_ok = all(s >= floor for s in seeds)
    boot = reset_bootstrap(g, c)
    if void:
        name = "B-VOID"
    elif at_bar and floor_ok and beats_r and beats_c:
        name = "B-PASS"
    elif beats_c:
        name = "B-GAIN"
    elif at_bar and beats_r:
        name = "B-BAR"
    elif boot["ci"][1] < DELTA:
        name = "B-FAIL"
    else:
        name = "B-UNDER"
    return {
        "row": name,
        "void_reason": void,
        "n": n,
        "bar": bar,
        "k_g": k_g,
        "k_c": k_c,
        "pooled_bar": need,
        "seed_floor": floor,
        "seeds_req": seeds,
        "seeds_v1": [int(x) for x in c.sum(1)],
        "seeds_req_interval": [zs.exact_interval(s, n) for s in seeds],
        "seeds_v1_interval": [zs.exact_interval(int(s), n) for s in c.sum(1)],
        "p_beats_v1": p_c,
        "p_beats_random": p_r,
        "beats_v1": bool(beats_c),
        "beats_random": bool(beats_r),
        "at_pooled_bar": bool(at_bar),
        "seed_floor_met": bool(floor_ok),
        "bootstrap_rate_difference": boot,
        "pooled_mcnemar_p_reported": pooled_mcnemar(g, c),
    }


def clause_fires(row: str) -> bool:
    """§8: the abandonment clause fires on B-FAIL only."""
    return row == "B-FAIL"


deferral_measures = pcl.deferral_measures
deferral_summary = pcl.deferral_summary
OFF_SUPPORT_LIMIT = RATE_LIMIT + 1e-6
CLIP_TOL = 1e-5
START_GRASP = -1.0  # the settled open hand at every reset (TASK-088)


def rate_measures(episode: dict) -> dict:
    """§6.4, reported (§1.2's reading of a run's ``plan_log``): per command, whether the rate
    clip changed the plan's first grasp (requested against executed), and per plan step whether
    |Δg| exceeds 2/11 against the previous executed or planned grasp; with grasp-phase subsets."""
    applied = np.asarray(episode.get("applied", []), np.float64).reshape(-1, 14)
    log = list(episode.get("plan_log", []))
    t = len(applied)
    if len(log) not in (t, t + 1):
        raise ContractError(f"plan log has {len(log)} entries for {t} commands")
    prev = START_GRASP
    clipped, gp_clipped, steps, off, plans, on_plans, ep_shares = 0, 0, 0, 0, 0, 0, []
    gp_cmd = 0
    for k in range(t):
        g = np.asarray(log[k]["grasp"], np.float64)
        d = np.abs(np.diff(np.concatenate([[prev], g])))
        steps += len(d)
        off += int((d > OFF_SUPPORT_LIMIT).sum())
        plans += 1
        on_plans += int((d <= OFF_SUPPORT_LIMIT).all())
        ep_shares.append(float((d > OFF_SUPPORT_LIMIT).mean()))
        c = abs(g[0] - applied[k, GRASP_DIM]) > CLIP_TOL
        clipped += int(c)
        if log[k]["index"] == pcl.GRASP_PHASE:
            gp_cmd += 1
            gp_clipped += int(c)
        prev = applied[k, GRASP_DIM]
    return {
        "commands": t,
        "first_grasp_clipped": clipped,
        "grasp_phase_commands": gp_cmd,
        "grasp_phase_first_grasp_clipped": gp_clipped,
        "plan_steps": steps,
        "plan_steps_off_support": off,
        "plans": plans,
        "plans_wholly_on_support": on_plans,
        "episode_off_support_share": (float(np.mean(ep_shares)) if ep_shares else None),
    }


def rate_summary(measures: list[dict], successes: list[bool]) -> dict:
    """Per run: the pooled shares of §6.4's two rate measures (reported only)."""

    def share(a, b):
        tot = sum(m[b] for m in measures)
        return float(sum(m[a] for m in measures) / tot) if tot else None

    succ = [m["episode_off_support_share"] for m, s in zip(measures, successes, strict=True) if s]
    fail = [
        m["episode_off_support_share"] for m, s in zip(measures, successes, strict=True) if not s
    ]
    succ = [x for x in succ if x is not None]
    fail = [x for x in fail if x is not None]
    return {
        "grasp_phase_first_grasp_clipped_share": share(
            "grasp_phase_first_grasp_clipped", "grasp_phase_commands"
        ),
        "first_grasp_clipped_share": share("first_grasp_clipped", "commands"),
        "plan_steps_off_support_share": share("plan_steps_off_support", "plan_steps"),
        "plans_wholly_on_support_share": share("plans_wholly_on_support", "plans"),
        "episode_off_support_median_success": float(np.median(succ)) if succ else None,
        "episode_off_support_median_failure": float(np.median(fail)) if fail else None,
    }


# ----- power (§6.4) ----------------------------------------------------------------------------
POWER_ALTERNATIVE = (0.15, 0.05)
POWER_SDS = gc.POWER_SDS  # 0, 1, 2
POWER_NS = (64, 128)
POWER_TARGET = gc.POWER_TARGET  # 0.80
POWER_SIMS = 20_000


def power_table(sims: int = POWER_SIMS) -> dict:
    """TASK-090's simulation (``gc.power``) at §6.4's declared alternative."""
    rows = []
    for n in POWER_NS:
        for sd in POWER_SDS:
            rows.append(gc.power(n, sd, alt=POWER_ALTERNATIVE, sims=sims))
    return {"alternative": list(POWER_ALTERNATIVE), "sims": sims, "rows": rows}


def power_rule(rows: list[dict]) -> dict:
    """§6.4: n = 128 stands unless the power at SD 1 is below 0.80 there (then escalate)."""
    at = {(r["n"], r["sd"]): r["power"] for r in rows}
    p = at[(BS_N, 1.0)]
    return {"n": BS_N, "power_at_sd1": p, "escalate": bool(p < POWER_TARGET)}


def write_json(path, value) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=True) + "\n")
