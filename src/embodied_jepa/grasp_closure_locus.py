"""TASK-094: the grasp-closure locus diagnostic (protocol
``docs/experiments/grasp_closure_locus_v1.md``). Measurement only: no controller, no training, no
cost or planner variant, no world-model closed loop.

This module holds the NumPy part:

* **K, the physics dose** (§4): TASK-088's IK follower with its closing command held at a level c
  (``HeldIKController``), the rigidly displaced goals of the palm-offset arms, the per-episode
  plumbing checks and measures, K's paired contrasts (exact one-sided McNemar, the reset bootstrap
  with salt 9014) and K's rows.
* **M, the ranking** (§5): the root types Rs (8 of the demonstrator's 35 close commands in) and R2
  (the end of the close), the scripted hold-level branches (``build_locus_root``, run from the
  reset with the demonstrator's own arithmetic and checked bitwise), the executed-difference
  arithmetic, the discordant roots, the ceiling's and G's wins, bars and the positive control.
* **S, the search** (§6): the executed hold level of a frozen-planner plan (TASK-093's projector
  arithmetic, used only to *read* the plan), the search miss and its bar.
* The L rows, the task row, Stage 0's row, the power recompute (§8.4), the cohorts and seeds, and
  the sha256 pins of every earlier file TASK-094 imports.

TASK-087's to TASK-093's code is imported and never changed. NumPy only at import; MuJoCo is
imported lazily by :mod:`embodied_jepa.simulation`. Object truth is read only by the goal
demonstrator, the root builder, the branch outcome labels and the scorer; never by a model input,
a cost or the planner. Nothing here is registered or imported by ``embodied_jepa``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from embodied_jepa import grasp_rate_projected as rp
from embodied_jepa import play_corpus as pc
from embodied_jepa import zero_shot as zs
from embodied_jepa import zero_shot_grasp as zg
from embodied_jepa.contracts import ContractError

TASK = "TASK-094"
PROTOCOL = "docs/experiments/grasp_closure_locus_v1.md"

# ----- seeds and salts (§9) --------------------------------------------------------------------
SALTS = {"bootstrap": 9014, "reserved": 9015}
POWER_TAG = 94  # a fixed tag in the power simulation's generator key (declared at Stage 0)
MODEL_SEEDS = zs.MODEL_SEEDS  # 87100-87102, TASK-087's G (no training; G-87102 last_two_triggered)
DEBUG_SEEDS = range(94500, 94600)  # Stage 0 plumbing and smoke; never counted
COHORTS = {
    "debug-k": range(94500, 94550),
    "debug-roots": range(94550, 94600),
    "dev-k": range(94100, 94132),
    "dev-roots": range(940000, 940032),
    "gated-k": range(941000, 941128),
    "gated-roots": range(942000, 942064),
}
K_COHORTS = ("debug-k", "dev-k", "gated-k")
ROOT_COHORTS = ("debug-roots", "dev-roots", "gated-roots")
GATED_K_N = len(COHORTS["gated-k"])  # 128
GATED_ROOTS_N = len(COHORTS["gated-roots"])  # 64
DETERMINISM_K = (("K-1.0", "K-s*"), 4)  # 941000-941003, inside the gated K cohort (§9)
DETERMINISM_S = 2  # 942000-942001, inside the gated roots (§9)
EARLIER_RANGES = dict(rp.EARLIER_RANGES) | {f"TASK-093 {k}": v for k, v in rp.COHORTS.items()}


def check_seed(cohort: str, seed: int) -> None:
    if cohort not in COHORTS:
        raise ContractError(f"unknown cohort {cohort}")
    if int(seed) not in COHORTS[cohort]:
        raise ContractError(f"seed {seed} is not in cohort {cohort}")


# ----- the frozen setting (§3) -----------------------------------------------------------------
LAMBDA = zs.LAMBDA  # 2.0552, the step-averaged cost's pose weight
SETTING = "step-mean"  # TASK-089's G-chg (unprojected, as TASK-091 and TASK-093's M-v1 ran it)
HORIZON = zs.CEM["horizon"]  # 8
GRASP_DIM = zg.GRASP_DIM  # 13
RATE_LIMIT = 2.0 / 11.0  # the embodiment's executed right-grasp step per command (§5.1)
JOB_CAP_SECONDS = 1800  # per GPU job (§10.2)
STAGE0_GPU_HOURS_MAX = 0.4
PROJECTION_GPU_HOURS_MAX = 0.6
CEILING_GPU_HOURS = 1.0

# ----- pins (§10, §14) -------------------------------------------------------------------------
CHECKPOINT_SHA256 = dict(rp.CHECKPOINT_SHA256)  # G-v1-87100 ... G-v1-87102
FEATURE_REPORT_SHA256 = rp.FEATURE_REPORT_SHA256
MOMENTS_SHA256 = rp.MOMENTS_SHA256
IMPLEMENTATION_SHA256 = rp.IMPLEMENTATION_SHA256
# Earlier tasks' code that TASK-094 imports and does not edit: TASK-093's pins (TASK-087-TASK-092,
# the embodiment, its action manifest, the candidate contract, the play scene, planning.py and the
# zero-shot modules), TASK-093's own modules and scripts, and the shared run tools. The runner
# refuses to start when any of these differs.
FROZEN_FILES_SHA256 = dict(rp.FROZEN_FILES_SHA256) | {
    "src/embodied_jepa/grasp_rate_projected.py": (
        "291020d50d49e1143e416b57207b93fb36a642160ecd5b4c14d32c0ef500e293"
    ),
    "src/embodied_jepa/grasp_rate_projected_runtime.py": (
        "5eca39f0ec8cf8d6f44ae1fdad9923ad6c6ab5bd9b905bc1bbb9fe4868c1aaad"
    ),
    "scripts/run_task093.py": "d4ee90ef471da12bf14699268cd798865675f87725edeabcc62130cfa5c0b47c",
    "scripts/summarize_task093.py": (
        "fc25200e06c462e078369048ffa42f1e3ab8cfaf740e035f1da5b4b5b8b3edf6"
    ),
    "scripts/run_task093_stage.sh": (
        "1884f368b17ca8dbc854e25528feed1e45e02ce733f42cb4a9e081e46f0633f4"
    ),
    "src/embodied_jepa/run_tools.py": (
        "1b3b61f89ca50e5d12650312c374bdcd07f1eb394ffb413c19eb844eb5ff5394"
    ),
    "src/embodied_jepa/devices.py": (
        "0976ccc0fb6d3c64d8128dbc152dcb2ebd3aee2211bc3ab5e25a772518e2fb72"
    ),
}
OWN_MODULES = (
    "src/embodied_jepa/grasp_closure_locus.py",
    "src/embodied_jepa/grasp_closure_locus_runtime.py",
)


def frozen_files_check(root) -> dict:
    out = {}
    for rel, pin in FROZEN_FILES_SHA256.items():
        path = Path(root) / rel
        out[rel] = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == pin
    return out


def own_modules_sha256(root) -> dict:
    return {rel: hashlib.sha256((Path(root) / rel).read_bytes()).hexdigest() for rel in OWN_MODULES}


# ===== Part K (§4) =============================================================================
GRASP_SUBGOAL = zs.SUBGOALS.index("grasp")  # 2
LIFT_SUBGOAL = zs.SUBGOALS.index("lift")  # 3
CLOSING_SUBGOALS = (GRASP_SUBGOAL, LIFT_SUBGOAL)
OFFSET_M = 0.012
OFFSETS = {"x": (OFFSET_M, 0.0, 0.0), "y": (0.0, OFFSET_M, 0.0)}  # pelvis frame (§4.1)
S_STAR = "s*"
# arm -> (held closing command c, palm offset axis or None)
K_ARMS = {
    "K-1.0": (1.0, None),
    "K-0.9": (0.9, None),
    "K-0.8": (0.8, None),
    "K-0.7": (0.7, None),
    "K-s*": (S_STAR, None),
    "K-0.6": (0.6, None),
    "K-open": (-1.0, None),
    "K-off-x-1.0": (1.0, "x"),
    "K-off-x-s*": (S_STAR, "x"),
    "K-off-y-1.0": (1.0, "y"),
    "K-off-y-s*": (S_STAR, "y"),
}
REFERENCE_ARM = "K-1.0"
CONTRAST_ARM = "K-s*"
NULL_SECOND_ARM = "K-0.6"
OPEN_ARM = "K-open"
STAGE0_EXTRA_ARMS = ("ik",)  # Stage 0 only: TASK-088's frozen follower, for K-1.0's bitwise check
# (first, second): Δ = (k_first - k_second) / n
K_CONTRASTS = tuple(
    [(REFERENCE_ARM, a) for a in K_ARMS if a != REFERENCE_ARM]
    + [("K-s*", "K-off-x-s*"), ("K-s*", "K-off-y-s*")]
)
DELTA_K = 0.15
P_THRESHOLD = 0.01
BOOTSTRAP_RESAMPLES = 10_000
VOID_BAND = (63, 114)  # K-1.0's count of 128: 0.69 +- 0.20 (§4.3)
OPEN_MAX = 3  # K-open's count of 128 (§4.3)
SLIP_RISE = 0.01  # m


def arm_slug(arm: str) -> str:
    """File-safe arm name (``K-s*`` -> ``K-sstar``)."""
    return arm.replace("*", "star")


def arm_from_slug(slug: str) -> str:
    return slug.replace("star", "*") if slug.replace("star", "*") in K_ARMS else slug


def s_star_of(robot, goal) -> float:
    """s* (§4.1): the right-hand synergy of the goal's grasp-subgoal hand joints (the
    demonstrator's measured, object-blocked hand at the end of its close). Goal pose only."""
    return float(zs.synergy_grasp(robot, goal["subgoals"][GRASP_SUBGOAL]["state28"][7:14]))


def arm_level(arm: str, s_star: float) -> float:
    level = K_ARMS[arm][0]
    return float(s_star) if level == S_STAR else float(level)


class HeldIKController(zs.IKController):
    """TASK-088's ``IKController`` with one change of its grasp target (§4.1): where the frozen
    follower wants +1 (the subgoal's hand synergy is above 0: the grasp and lift subgoals), it
    wants the arm's level c; elsewhere -1, as frozen. Everything else -- e9's tracking law, the
    press, the 0.2 ramp -- is the frozen code's arithmetic. It reads proprioception and the goal
    pose only. ``log`` keeps, per command, the subgoal index, the requested grasp (float32) and the
    measured hand synergy (for the blocking measure)."""

    def __init__(self, arm: str):
        if arm not in K_ARMS:
            raise ContractError(f"unknown K arm {arm}")
        super().__init__(press=True)
        self.name = arm
        self.arm = arm
        self.level = None
        self.log = []

    def act(self, robot, observation, goal, index):
        if self.level is None:
            self.level = arm_level(self.arm, s_star_of(robot, goal))
        p = zs.proprio(robot, observation)
        g = goal["subgoals"][index]
        closing = zs.synergy_grasp(robot, g["state28"][7:14]) > 0
        want = self.level if closing else -1.0
        self.grasp = float(np.clip(want, self.grasp - zs.GRASP_RAMP, self.grasp + zs.GRASP_RAMP))
        target = np.asarray(g["palm"], float).copy()
        if self.press and goal["task"] == "grasp" and zs.SUBGOALS[index] == "grasp":
            target[2] -= zs.PRESS_DEPTH  # e9's press, as frozen
        action = zs.tracking_action(
            robot, p["palm"], p["rotation"], target, g["rotation"], self.grasp
        )
        self.log.append(
            {
                "index": int(index),
                "closing": bool(closing),
                "requested": float(action[GRASP_DIM]),
                "measured_synergy": float(zs.synergy_grasp(robot, p["state28"][7:14])),
            }
        )
        return action


def displaced_goal(goal: dict, axis: str) -> dict:
    """§4.1: the goal with v added to the palm of all four subgoals (float64), everything else
    the same objects."""
    v = np.asarray(OFFSETS[axis], np.float64)
    out = dict(goal)
    out["subgoals"] = [
        dict(g, palm=np.asarray(g["palm"], np.float64) + v) for g in goal["subgoals"]
    ]
    out["offset"] = v.tolist()
    return out


def displacement_ok(base: dict, off: dict, axis: str) -> bool:
    """The offset arm's goal is the K-1.0 goal displaced by exactly v: every subgoal's palm
    differs by v (float64), frames, states and rotations are bitwise the same, and so is the
    rest of the goal."""
    v = np.asarray(OFFSETS[axis], np.float64)
    if len(base["subgoals"]) != len(off["subgoals"]):
        return False
    for a, b in zip(base["subgoals"], off["subgoals"], strict=True):
        if not np.allclose(np.asarray(b["palm"]) - np.asarray(a["palm"], np.float64), v,
                           rtol=0, atol=1e-12):  # fmt: skip
            return False
        if np.asarray(b["palm"]).dtype != np.float64:
            return False
        for key in ("frame", "state28", "rotation"):
            if not np.array_equal(a[key], b[key]):
                return False
    rest = {k for k in base if k not in ("subgoals",)}
    return all(base[k] == off[k] for k in rest if k != "digest") and off.get("offset") == list(v)


def expected_requests(indices, level: float) -> np.ndarray:
    """The frozen follower's grasp requests (float32) for a sequence of subgoal indices with the
    held level on the closing subgoals: the 0.2 ramp from -1."""
    g, out = -1.0, []
    for i in indices:
        want = level if int(i) in CLOSING_SUBGOALS else -1.0
        g = float(np.clip(want, g - zs.GRASP_RAMP, g + zs.GRASP_RAMP))
        out.append(np.float32(g))
    return np.asarray(out, np.float32)


def _bits(x) -> np.ndarray:
    return np.asarray(x, np.float32).view(np.uint32)


PATH_TOLERANCE = 1e-5  # the applied right grasp against the rate-limited request path
REACH_TOLERANCE = 1e-6


def executed_path(requests) -> np.ndarray:
    """The executed right grasp the embodiment's rate limit allows for a request sequence: the
    cumulative clip at 2/11 per command from the settled open hand (-1), in grasp units (the
    steepest hand joint binds; float64)."""
    g, out = -1.0, []
    for r in np.asarray(requests, np.float64):
        g = float(np.clip(r, g - RATE_LIMIT, g + RATE_LIMIT))
        out.append(g)
    return np.asarray(out, np.float64)


def k_plumbing(log: list, applied, level: float) -> dict:
    """§4.3's plumbing check for one K episode: (a) every request is the frozen ramp towards the
    held level on the grasp and lift subgoals and towards -1 elsewhere (bitwise float32), so on
    every command of those phases after the ramp the request equals c; (b) the applied right grasp
    follows the rate-limited path of the requests (within 1e-5) and equals float32(c) bitwise on
    every command of those phases once the rate limit allows it (the path has reached c); an
    episode stopped before that has no such command; (c) the follower's closing decision is
    exactly the grasp and lift subgoals (log entries without ``closing`` are read as deciding by
    the subgoal, for re-reading recorded episodes)."""
    applied = np.asarray(applied, np.float32).reshape(-1, 14)
    t = len(applied)
    if len(log) not in (t, t + 1):  # a stopped episode logs one request it did not execute
        return {"ok": False, "reason": f"log has {len(log)} entries for {t} commands"}
    log = log[:t]
    idx = np.asarray([e["index"] for e in log], np.int64)
    req = np.asarray([e["requested"] for e in log], np.float32)
    phase = np.isin(idx, CLOSING_SUBGOALS)
    closing = np.asarray([e.get("closing", i in CLOSING_SUBGOALS) for e, i in
                          zip(log, idx, strict=True)], bool)  # fmt: skip
    c_bits = np.asarray(level, np.float32).view(np.uint32)
    exp = expected_requests(idx, level)
    ramp_ok = bool(np.array_equal(_bits(req), _bits(exp)))
    decision_ok = bool(np.array_equal(closing, phase))
    after_ramp = phase & (_bits(exp) == c_bits)
    req_at_c = int((after_ramp & (_bits(req) == c_bits)).sum())
    path = executed_path(req)
    deviation = float(np.abs(applied[:, GRASP_DIM].astype(np.float64) - path).max()) if t else 0.0
    allowed = phase & (np.abs(path - float(np.float32(level))) < REACH_TOLERANCE)
    applied_at_c = int((allowed & (_bits(applied[:, GRASP_DIM]) == c_bits)).sum())
    start = int(np.flatnonzero(phase)[0]) if phase.any() else -1
    first = int(np.flatnonzero(allowed)[0]) if allowed.any() else -1
    ok = (
        ramp_ok
        and decision_ok
        and req_at_c == int(after_ramp.sum())
        and deviation <= PATH_TOLERANCE
        and applied_at_c == int(allowed.sum())
    )
    return {
        "ok": bool(ok),
        "ramp_ok": ramp_ok,
        "decision_ok": decision_ok,
        "phase_commands": int(phase.sum()),
        "after_ramp_commands": int(after_ramp.sum()),
        "after_ramp_requests_at_c": req_at_c,
        "allowed_commands": int(allowed.sum()),
        "allowed_applied_at_c": applied_at_c,
        "path_deviation_max": deviation,
        "ramp_commands": int(np.flatnonzero(after_ramp)[0] - start) if after_ramp.any() else -1,
        "applied_reaches_c_after": first - start if first >= 0 else -1,
    }


def replumb(episode: dict) -> dict:
    """§4.3's plumbing check re-read from a recorded K episode (its requests, subgoal indices and
    applied commands), keeping the job's own closing-decision reading."""
    log = [{"index": i, "requested": r} for i, r in
           zip(episode["subgoal_index"], episode["requested"], strict=True)]  # fmt: skip
    out = k_plumbing(log, episode["applied"], episode["level"])
    job = episode.get("plumbing", {})
    out["job_decision_ok"] = bool(job.get("decision_ok", False))
    out["ok"] = bool(out["ok"] and out["job_decision_ok"])
    return out


def slipped(rise, contact) -> bool:
    """Contact lost after a rise of at least 1 cm while in grasp contact (§4.2)."""
    lifted = False
    for r, c in zip(rise, contact, strict=True):
        if r >= SLIP_RISE and c:
            lifted = True
        elif lifted and not c:
            return True
    return False


def k_episode_measures(result: dict, log: list) -> dict:
    """§4.2's per-episode measures from TASK-088's episode result and the controller's log."""
    idx = np.asarray([e["index"] for e in log], np.int64)
    syn = np.asarray([e["measured_synergy"] for e in log], np.float64)
    sel = np.isin(idx, CLOSING_SUBGOALS)
    grasp_phase = idx == GRASP_SUBGOAL
    return {
        "success": bool(result["success"]),
        "grasp_contact": bool(result.get("any_grasp_contact", False)),
        "max_rise": float(result.get("max_rise", 0.0)),
        "slip": slipped(result.get("rise", []), result.get("contact", [])),
        "stopped": bool(result.get("stop_reason")),
        "final_subgoal": int(result.get("final_subgoal", 0)),
        "switch_reasons": [s["reason"] for s in result.get("switches", [])],
        "measured_synergy_closing_median": float(np.median(syn[sel])) if sel.any() else None,
        "measured_synergy_grasp_phase_last": float(syn[grasp_phase][-1])
        if grasp_phase.any()
        else None,
    }


# ----- K statistics and rows (§4.2, §4.3) -------------------------------------------------------
def bootstrap_indices(n: int, resamples: int = BOOTSTRAP_RESAMPLES) -> np.ndarray:
    """The reset resamples shared by every K contrast: ``default_rng([9014, n])``, ``[B, n]``."""
    rng = np.random.default_rng([SALTS["bootstrap"], int(n)])
    return rng.integers(0, n, size=(resamples, n))


def paired_contrast(first, second, indices=None) -> dict:
    """Δ = (k_first - k_second) / n with the exact one-sided McNemar p ("first beats second") and
    the two-sided 95 % reset-bootstrap percentile interval (``np.percentile``, linear; the upper
    bound is the 97.5th percentile)."""
    a, b = np.asarray(first, bool), np.asarray(second, bool)
    if a.shape != b.shape or a.ndim != 1:
        raise ContractError("paired contrasts need equal-length indicator vectors")
    n = len(a)
    if indices is None:
        indices = bootstrap_indices(n)
    d = a.astype(np.int64) - b.astype(np.int64)
    means = d[indices].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    nb, nc = int((a & ~b).sum()), int((~a & b).sum())
    return {
        "n": n,
        "k_first": int(a.sum()),
        "k_second": int(b.sum()),
        "delta": float(d.mean()),
        "discordant_first": nb,
        "discordant_second": nc,
        "p_one_sided": zs.mcnemar_one_sided(nb, nc),
        "ci": [float(lo), float(hi)],
    }


def void_band(n: int) -> dict:
    """K-1.0's band and K-open's maximum for n resets: 63-114 and 3 of 128 (§4.3); at Stage 0's
    32 development resets the same proportions, rounded inwards (16-28 and 0 of 32)."""
    if n == GATED_K_N:
        return {"low": VOID_BAND[0], "high": VOID_BAND[1], "open_max": OPEN_MAX}
    return {
        "low": int(math.ceil(VOID_BAND[0] * n / GATED_K_N)),
        "high": int(math.floor(VOID_BAND[1] * n / GATED_K_N)),
        "open_max": int(math.floor(OPEN_MAX * n / GATED_K_N)),
    }


def k_contrasts(successes: dict) -> dict:
    """Every contrast of ``K_CONTRASTS`` on the arms present, on one shared resample set."""
    n = len(successes[REFERENCE_ARM])
    idx = bootstrap_indices(n)
    out = {}
    for a, b in K_CONTRASTS:
        if a in successes and b in successes:
            out[f"{a} vs {b}"] = paired_contrast(successes[a], successes[b], idx)
    return out


def k_row(successes: dict, *, plumbing_ok: bool = True, void: str = "") -> dict:
    """§4.3, first match: K-VOID, K-NULL, K-CAUSAL, K-UNCLEAR. ``successes``: arm -> bool[n] on
    the same resets (paired)."""
    n = len(successes[REFERENCE_ARM])
    band = void_band(n)
    con = k_contrasts(successes)
    main = con[f"{REFERENCE_ARM} vs {CONTRAST_ARM}"]
    second = con[f"{REFERENCE_ARM} vs {NULL_SECOND_ARM}"]
    k1 = int(np.sum(successes[REFERENCE_ARM]))
    k_open = int(np.sum(successes[OPEN_ARM]))
    reasons = []
    if not band["low"] <= k1 <= band["high"]:
        reasons.append(f"K-1.0 {k1} of {n} outside {band['low']}-{band['high']}")
    if k_open > band["open_max"]:
        reasons.append(f"K-open {k_open} of {n} above {band['open_max']}")
    if not plumbing_ok:
        reasons.append("a plumbing check failed")
    if void:
        reasons.append(void)
    null_main = main["ci"][1] < DELTA_K
    null_second = second["ci"][1] < DELTA_K
    causal = main["delta"] >= DELTA_K and main["p_one_sided"] < P_THRESHOLD
    if reasons:
        name = "K-VOID"
    elif null_main and null_second:
        name = "K-NULL"
    elif null_main:
        name = "K-UNCLEAR"  # Δ's bound below δ_K, Δ_0.6's not (§4.3)
    elif causal:
        name = "K-CAUSAL"
    else:
        name = "K-UNCLEAR"
    return {
        "row": name,
        "void_reasons": reasons,
        "n": n,
        "band": band,
        "k": {a: int(np.sum(v)) for a, v in successes.items()},
        "interval": {a: list(zs.exact_interval(int(np.sum(v)), n)) for a, v in successes.items()},
        "delta": main["delta"],
        "delta_ci": main["ci"],
        "delta_p": main["p_one_sided"],
        "delta_06": second["delta"],
        "delta_06_ci": second["ci"],
        "contrasts": con,
    }


# ===== Part M (§5) =============================================================================
ROOT_TYPES = ("Rs", "R2")
RS_CLOSE = 8  # of the demonstrator's 35 close commands (§5.1)
CLOSE = zs.GRASP_PHASES["close"]  # 35
ROOT_GOAL = {"Rs": GRASP_SUBGOAL, "R2": LIFT_SUBGOAL}  # the goal subgoal of each root type
RS_TARGET_GRASP = -1.0 + RS_CLOSE * RATE_LIMIT  # 0.4545..., Rs's last accepted grasp target
ROOT_REQUEST = {"Rs": -1.0 + RS_CLOSE * zs.GRASP_RAMP, "R2": 1.0}  # the demonstrator's request
SQUEEZE = "1.0"
CONTROL = "control"
BRANCHES_ALL = ("1.0", "0.9", "0.8", "0.7", S_STAR, CONTROL)
BRANCHES_STAGE0 = (SQUEEZE, S_STAR, CONTROL)  # §14: development M with these three only
BRANCHES_BY_COHORT = {
    "debug-roots": BRANCHES_ALL,  # the smoke runs every branch (timing, §10.2)
    "dev-roots": BRANCHES_STAGE0,
    "gated-roots": BRANCHES_ALL,
}
# root type -> [(palm target, commands)] after the root
BRANCH_SEGMENTS = {
    "Rs": (
        ("aim", CLOSE - RS_CLOSE),
        ("lift", zs.GRASP_PHASES["lift"]),
        ("lift", zs.GRASP_PHASES["hold"]),
    ),  # fmt: skip
    "R2": (("lift", zs.GRASP_PHASES["lift"]), ("lift", zs.GRASP_PHASES["hold"])),
}
BRANCH_LENGTH = {t: sum(s for _, s in segs) for t, segs in BRANCH_SEGMENTS.items()}  # 97, 70
WINDOW = HORIZON  # 8 steps stored and scored
MIN_DIFFERING = 7  # of the 8 window steps (§5.1, §7.1)
EXEC_DIFF_VOID = 8  # more than 8 of 64 roots below MIN_DIFFERING at a root type: L-VOID (§7.1)
READ_MIN_ND = 16  # a root type is read only when n_d >= 16 (§5.4)
CEILING_BAR = (3, 4)  # ĉ >= 0.75, exact (§5.5)
G_FACTOR = 0.75
POSITIVE_CONTROL = (7, 8)  # 56 / 64; 28 / 32 at Stage 0 (§5.5)
STAGE0_MIN_ND = 8  # of 32 development roots at either root type (§14)


def branch_level(name: str, s_star: float) -> float:
    if name == CONTROL:
        return -1.0
    if name == S_STAR:
        return float(s_star)
    return float(name)


def window_differences(s_star: float, root: str, level: float | None = None) -> int:
    """The executed-difference arithmetic (§5.1): how many of the 8 window steps' executed right
    grasps differ between the squeeze branch and a branch held at ``level`` (s* by default), from
    the root's demonstrator request (0.6 at Rs, +1 at R2; 0.2 ramp) and last accepted target
    (-1 + 8 x 2/11 at Rs, +1 at R2; 2/11 rate limit)."""
    level = float(s_star) if level is None else float(level)
    e0 = RS_TARGET_GRASP if root == "Rs" else 1.0
    r0 = ROOT_REQUEST[root]

    def path(c):
        r, e, out = r0, e0, []
        for _ in range(WINDOW):
            r = float(np.clip(c, r - zs.GRASP_RAMP, r + zs.GRASP_RAMP))
            e = float(np.clip(r, e - RATE_LIMIT, e + RATE_LIMIT))
            out.append(e)
        return np.asarray(out)

    return int((np.abs(path(1.0) - path(level)) > 1e-9).sum())


class _Branch:
    """TASK-089's ``_Demo`` (TASK-088's grasp demonstrator, phase by phase) driven from a root with
    a branch's held level: the same request arithmetic (the 0.2 ramp from the demonstrator's
    current request) and e9's tracking law. ``record`` keeps the applied commands, the target's
    rise and grasp contact at every step, and the onboard frame and measured state after each of
    the first 8 steps (the planning window)."""

    def __init__(self, demo):
        self.demo = demo
        self.applied, self.rise, self.contact = [], [], []
        self.frames, self.states = [], []

    def run(self, target, want, steps):
        demo, robot = self.demo, self.demo.robot
        for _ in range(steps):
            demo.grasp = float(
                np.clip(want, demo.grasp - zs.GRASP_RAMP, demo.grasp + zs.GRASP_RAMP)
            )
            obs = robot.observe()
            p = zs.proprio(robot, obs)
            request = zs.tracking_action(
                robot, p["palm"], p["rotation"], demo.target(target), demo.rotation, demo.grasp
            )
            applied = zs.execute(robot, request)
            truth = robot.sim.play_truth()
            demo.log.append(truth)
            self.applied.append(np.asarray(applied, np.float32))
            self.rise.append(float(truth["position"][demo.index][2] - demo.z0))
            self.contact.append(bool(truth["grasp_contact"][demo.index]))
            if len(self.applied) <= WINDOW:
                after = robot.observe()
                self.frames.append(after.images[pc.CAMERA][0].copy())
                self.states.append(zs.proprio(robot, after)["state28"])


def _record(robot):
    obs = robot.observe()
    return {"frame": obs.images[pc.CAMERA][0].copy(), "state28": zs.proprio(robot, obs)["state28"]}


def _to_root(robot, layout, yaw, root):
    """Re-run the demonstrator from the reset to a root: hover, open descent, and 8 (Rs) or 35 (R2)
    close commands."""
    demo = zg._Demo(robot, layout, yaw)
    ph = zs.GRASP_PHASES
    demo.run("hover", -1.0, ph["hover"])
    demo.run("aim", -1.0, ph["descend"])
    demo.run("aim", zs.GRASP_CLOSURE, RS_CLOSE if root == "Rs" else CLOSE)
    return demo


def same_record(a, b) -> bool:
    same_state = np.array_equal(a["state28"], b["state28"])
    return bool(same_state and np.array_equal(a["frame"], b["frame"]))


def build_locus_root(robot, seed: int, branches=BRANCHES_ALL) -> dict:
    """§5.1-§5.2 for one root seed: the demonstrator's draw exactly as
    ``zero_shot.make_grasp_goal`` makes it (layout, yaw, acceptance; the close observed after 8
    commands for Rs), then for each root type and branch a re-run from the reset to the root
    (checked bitwise against the draw) and the branch. A draw whose demonstrator is stopped or
    does not lift, or whose branch is stopped, is replaced by the next attempt (counted).
    Mismatches of the bitwise checks are recorded in ``problems`` (a VOID), not raised."""
    rng = np.random.default_rng([zs.SALTS["goal"], int(seed)])
    rejected = []
    ph = zs.GRASP_PHASES
    for attempt in range(zs.GOAL_ATTEMPTS):
        layout = zs.sample_layout("grasp", seed, attempt)
        yaw = float(rng.uniform(-pc.YAW_RANGE, pc.YAW_RANGE))
        try:
            demo = zg._Demo(robot, layout, yaw)
            subgoals = []
            demo.run("hover", -1.0, ph["hover"])
            subgoals.append(_record(robot))
            demo.run("aim", -1.0, ph["descend"])
            subgoals.append(_record(robot))
            demo.run("aim", zs.GRASP_CLOSURE, RS_CLOSE)
            rs = _record(robot)
            demo.run("aim", zs.GRASP_CLOSURE, CLOSE - RS_CLOSE)
            subgoals.append(_record(robot))
            demo.run("lift", zs.GRASP_CLOSURE, ph["lift"])
            demo.run("lift", zs.GRASP_CLOSURE, ph["hold"])
            subgoals.append(_record(robot))
        except zs.Stopped as error:
            rejected.append(f"stopped: {error}")
            continue
        if not zs.grasp_truth_success(demo.log, demo.index, demo.z0):
            rejected.append("demonstrator did not lift")
            continue
        s_star = float(zs.synergy_grasp(robot, subgoals[GRASP_SUBGOAL]["state28"][7:14]))
        drawn = {"Rs": rs, "R2": subgoals[GRASP_SUBGOAL]}
        out = {
            "seed": int(seed),
            "attempt": attempt,
            "layout": layout,
            "target": layout["target"],
            "yaw": yaw,
            "z0": demo.z0,
            "s_star": s_star,
            "subgoals": subgoals,
            "roots": {},
            "branches": {},
            "problems": [],
        }
        try:
            for root in ROOT_TYPES:
                for name in branches:
                    level = branch_level(name, s_star)
                    demo = _to_root(robot, layout, yaw, root)
                    here = _record(robot)
                    t0 = rp.read_t0(robot)
                    if not same_record(here, drawn[root]):
                        out["problems"].append(f"{seed} {root}/{name}: the re-run root differs")
                    out["roots"][root] = here | {"t0": t0, "request": float(demo.grasp)}
                    br = _Branch(demo)
                    checks = {}
                    for k, (target, steps) in enumerate(BRANCH_SEGMENTS[root]):
                        br.run(target, level, steps)
                        if name == SQUEEZE and root == "Rs" and k == 0:
                            checks["rs_grasp_subgoal"] = same_record(
                                _record(robot), subgoals[GRASP_SUBGOAL]
                            )
                    if name == SQUEEZE:
                        checks["lift_subgoal"] = same_record(_record(robot), subgoals[LIFT_SUBGOAL])
                        if not all(checks.values()):
                            out["problems"].append(
                                f"{seed} {root}/{name}: the squeeze branch is not the "
                                f"demonstrator's continuation {checks}"
                            )
                    out["branches"][f"{root}/{name}"] = {
                        "level": level,
                        "applied": np.stack(br.applied),
                        "frames": np.stack(br.frames),
                        "states": np.stack(br.states).astype(np.float32),
                        "rise": br.rise,
                        "contact": br.contact,
                        "lift": zs.lift_success(br.rise, br.contact),
                        "squeeze_checks": checks,
                    }
        except zs.Stopped as error:
            rejected.append(f"branch stopped: {error}")
            continue
        out["rejected"] = rejected
        return out
    raise ContractError(f"no root for seed {seed}: {rejected}")


def executed_differences(squeeze_applied, other_applied) -> dict:
    """How many of the 8 window steps' applied commands differ (any channel, bitwise), and how
    many differ in the right grasp alone."""
    a = np.asarray(squeeze_applied, np.float32)[:WINDOW]
    b = np.asarray(other_applied, np.float32)[:WINDOW]
    any_ch = (a.view(np.uint32) != b.view(np.uint32)).any(-1)
    grasp = a[:, GRASP_DIM].view(np.uint32) != b[:, GRASP_DIM].view(np.uint32)
    return {"any_channel": int(any_ch.sum()), "grasp": int(grasp.sum())}


def discordant(squeeze_lifts, other_lifts, include=None) -> np.ndarray:
    """A root is discordant when the squeeze branch lifts and the other does not (§5.4)."""
    s, o = np.asarray(squeeze_lifts, bool), np.asarray(other_lifts, bool)
    d = s & ~o
    if include is not None:
        d = d & np.asarray(include, bool)
    return d


def wins(squeeze_costs, other_costs) -> np.ndarray:
    """A win: cost(squeeze) < cost(other), strictly; ties are not wins (§5.4)."""
    return np.asarray(squeeze_costs, float) < np.asarray(other_costs, float)


def ceiling_ranks(wins_k: int, n_d: int) -> bool:
    num, den = CEILING_BAR
    return n_d > 0 and den * int(wins_k) >= num * int(n_d)


def g_bar(ceiling_rate: float, n_d: int) -> int:
    """⌈(0.5 + 0.75 (ĉ - 0.5)) n_d⌉ (§5.5)."""
    return int(math.ceil((0.5 + G_FACTOR * (ceiling_rate - 0.5)) * n_d - 1e-9))


def g_ranks(seed_wins, ceiling_wins: int, n_d: int) -> bool:
    if n_d <= 0:
        return False
    need = g_bar(ceiling_wins / n_d, n_d)
    return sum(int(w) >= need for w in seed_wins) >= 2


def positive_control_need(n: int) -> int:
    num, den = POSITIVE_CONTROL
    return int(math.ceil(num * n / den - 1e-9))


# ===== Part S (§6) =============================================================================
SQUEEZE_LEVEL = 0.9  # the plan squeezes when ℓ >= 0.9 (§1.1's top bin)
MISS_FRACTION = (1, 4)


def executed_level(geometry, plan, t0) -> np.ndarray:
    """The plan's executed right grasp at steps 1-8 from the root's last accepted hand targets T₀
    (TASK-093's cumulative clip, used only to read the plan); ℓ is the value at step 8."""
    _, g = rp.project_grasp(geometry, np.asarray(plan, np.float32), np.asarray(t0, np.float64))
    return g


def search_measures(level_8: float, plan_cost: float, squeeze_cost: float) -> dict:
    squeezes = bool(level_8 >= SQUEEZE_LEVEL)
    return {
        "level": float(level_8),
        "squeezes": squeezes,
        "miss": bool((not squeezes) and plan_cost > squeeze_cost),
        "partial_preferred": bool((not squeezes) and plan_cost <= squeeze_cost),
    }


def miss_need(n_d: int) -> int:
    num, den = MISS_FRACTION
    return int(math.ceil(num * n_d / den - 1e-9))


def search_misses(seed_misses, n_d: int) -> bool:
    if n_d <= 0:
        return False
    need = miss_need(n_d)
    return sum(int(m) >= need for m in seed_misses) >= 2


# ===== the L rows and the task row (§7) ========================================================
def type_reading(t: dict) -> dict:
    """One root type's reading: ``t`` has ``n_d``, ``ceiling_wins``, ``g_wins`` (three seeds) and
    ``s_misses`` (three seeds) on its discordant roots."""
    n_d = int(t["n_d"])
    read = n_d >= READ_MIN_ND
    c_ok = ceiling_ranks(t["ceiling_wins"], n_d)
    g_ok = g_ranks(t["g_wins"], t["ceiling_wins"], n_d)
    s_miss = search_misses(t["s_misses"], n_d)
    if not read:
        own = "unread"
    elif not c_ok:
        own = "L-COST"
    elif not g_ok:
        own = "L-MODEL"
    elif s_miss:
        own = "L-SEARCH"
    else:
        own = "L-MIXED"
    return {
        "n_d": n_d,
        "read": read,
        "ceiling_rate": (t["ceiling_wins"] / n_d) if n_d else None,
        "ceiling_ranks": c_ok,
        "g_bar": g_bar(t["ceiling_wins"] / n_d, n_d) if n_d else None,
        "g_ranks": g_ok,
        "miss_need": miss_need(n_d),
        "search_misses": s_miss,
        "own_row": own,
    }


def l_row(per_type: dict, *, void: str = "") -> dict:
    """§7.1, first match over the read root types: L-VOID, L-UNREAD, L-COST, L-MODEL, L-SEARCH,
    L-MIXED."""
    readings = {t: type_reading(v) for t, v in per_type.items()}
    read = [t for t, r in readings.items() if r["read"]]
    if void:
        name = "L-VOID"
    elif not read:
        name = "L-UNREAD"
    elif any(not readings[t]["ceiling_ranks"] for t in read):
        name = "L-COST"
    elif any(not readings[t]["g_ranks"] for t in read):
        name = "L-MODEL"
    elif any(readings[t]["search_misses"] for t in read):
        name = "L-SEARCH"
    else:
        name = "L-MIXED"
    cost_types = [t for t in read if not readings[t]["ceiling_ranks"]]
    return {"row": name, "void_reason": void, "read": read, "readings": readings,
            "l_cost_root_types": cost_types}  # fmt: skip


def task_row(k: dict, lrow: dict | None) -> str:
    """The task's row (§7.2): K's row, and after K-CAUSAL the L row."""
    if k["row"] != "K-CAUSAL":
        return k["row"]
    return f"K-CAUSAL + {lrow['row'] if lrow else 'L-VOID'}"


# ===== Stage 0 (§14) ===========================================================================
def s0_row(checks: dict, *, n_d: dict, power_ok: bool, stage0_gpu_hours: float,
           projected_gpu_hours: float) -> dict:  # fmt: skip
    """S0-ESCALATE when a degeneracy, positive-control or plumbing check fails, n_d < 8 of 32 at
    both root types, the power rule fails, Stage 0 used more than 0.4 GPU h, or the gated stage
    projects above 0.6 GPU h; otherwise S0-PASS (the protocol freezes at the merge)."""
    failing = sorted(k for k, ok in checks.items() if not ok)
    low_nd = all(int(v) < STAGE0_MIN_ND for v in n_d.values())
    over_s0 = stage0_gpu_hours > STAGE0_GPU_HOURS_MAX
    over_proj = projected_gpu_hours > PROJECTION_GPU_HOURS_MAX
    escalate = bool(failing or low_nd or not power_ok or over_s0 or over_proj)
    return {
        "row": "S0-ESCALATE" if escalate else "S0-PASS",
        "failing_checks": failing,
        "n_d": dict(n_d),
        "n_d_below_8_at_both": low_nd,
        "power_ok": bool(power_ok),
        "stage0_gpu_hours": float(stage0_gpu_hours),
        "projected_gated_gpu_hours": float(projected_gpu_hours),
        "stage0_over_budget": over_s0,
        "projection_over_budget": over_proj,
    }


# ===== power (§8.4) ============================================================================
P1 = 0.69  # K-1.0's lift rate (the pooled BS rate 221 / 320, §1.2)
POWER_SIMS = 20_000
POWER_DELTAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
POWER_D0 = (0.03, 0.08)
POWER_MODELS = ("independent", "nested")


def joint_cells(delta: float, delta06: float, d0: float, model: str = "independent") -> np.ndarray:
    """The 8 probabilities of (K-1.0, K-s*, K-0.6) lifting, indexed 4 a + 2 b + c. K-1.0 lifts with
    0.69; the discordant cells of each contrast are (Δ + d0, d0) and (Δ_0.6 + d0, d0). Given
    K-1.0's outcome, K-s* and K-0.6 are independent ("independent") or K-0.6 fails whenever K-s*
    fails and lifts only where K-s* lifts ("nested", the most correlated case; needs
    Δ_0.6 >= Δ)."""
    q1 = 1.0 - P1
    fs, f6 = (delta + d0) / P1, (delta06 + d0) / P1  # P(fail | K-1.0 lifts)
    ls = l6 = d0 / q1  # P(lift | K-1.0 fails)
    p = np.zeros(8)
    if model == "independent":
        for b in (0, 1):
            for c in (0, 1):
                p[4 + 2 * b + c] = P1 * (fs if not b else 1 - fs) * (f6 if not c else 1 - f6)
                p[2 * b + c] = q1 * (ls if b else 1 - ls) * (l6 if c else 1 - l6)
    elif model == "nested":
        if delta06 < delta:
            raise ContractError("the nested model needs Δ_0.6 >= Δ")
        p[4 + 0] = P1 * fs  # both fail
        p[4 + 2] = P1 * (f6 - fs)  # K-s* lifts, K-0.6 fails
        p[4 + 3] = P1 * (1 - f6)
        p[3] = q1 * ls  # both lift (d0 cells coincide)
        p[0] = q1 * (1 - ls)
    else:
        raise ContractError(f"unknown model {model}")
    if np.any(p < -1e-12) or abs(p.sum() - 1) > 1e-9:
        raise ContractError("impossible cell probabilities")
    return np.clip(p, 0, None) / p.clip(0, None).sum()


def k_power_cell(delta: float, d0: float, delta06: float, *, n: int = GATED_K_N,
                 sims: int = POWER_SIMS, resamples: int = BOOTSTRAP_RESAMPLES,
                 model: str = "independent") -> dict:  # fmt: skip
    """§8.4: the K row rates (K-NULL joint, K-CAUSAL, K-UNCLEAR) under ``joint_cells``, each draw
    read with K's own rule (the exact McNemar test, Δ ≥ 0.15, the reset-bootstrap percentile upper
    bounds of Δ and Δ_0.6 with ``resamples`` resamples). The generator is ``default_rng([9014,
    94, n, round(1000 Δ), round(1000 d0), round(1000 Δ_0.6), model index])``."""
    rng = np.random.default_rng(
        [SALTS["bootstrap"], POWER_TAG, int(n), int(round(1000 * delta)), int(round(1000 * d0)),
         int(round(1000 * delta06)), POWER_MODELS.index(model)]
    )  # fmt: skip
    probs = joint_cells(delta, delta06, d0, model)
    a = np.array([(i >> 2) & 1 for i in range(8)])
    b = np.array([(i >> 1) & 1 for i in range(8)])
    c = np.array([i & 1 for i in range(8)])
    rows = {"K-NULL": 0, "K-CAUSAL": 0, "K-UNCLEAR": 0}
    for _ in range(sims):
        counts = rng.multinomial(n, probs)
        boots = rng.multinomial(n, counts / n, size=resamples)
        dm = boots @ (a - b) / n
        d6 = boots @ (a - c) / n
        ub = np.percentile(dm, 97.5)
        ub6 = np.percentile(d6, 97.5)
        nb = int(counts[(a == 1) & (b == 0)].sum())
        nc = int(counts[(a == 0) & (b == 1)].sum())
        dhat = (nb - nc) / n
        if ub < DELTA_K and ub6 < DELTA_K:
            rows["K-NULL"] += 1
        elif ub >= DELTA_K and dhat >= DELTA_K and zs.mcnemar_one_sided(nb, nc) < P_THRESHOLD:
            rows["K-CAUSAL"] += 1
        else:
            rows["K-UNCLEAR"] += 1
    return {"delta": delta, "d0": d0, "delta06": delta06, "model": model, "n": n, "sims": sims,
            "resamples": resamples, "rates": {k: v / sims for k, v in rows.items()}}  # fmt: skip


def power_grid() -> list[tuple]:
    """(Δ, d0, Δ_0.6, model): every Δ and d0 with Δ_0.6 = Δ and Δ_0.6 = Δ + 0.10 (independent),
    and the rule's two cells under the nested model."""
    cells = []
    for delta in POWER_DELTAS:
        for d0 in POWER_D0:
            cells.append((delta, d0, delta, "independent"))
            cells.append((delta, d0, round(delta + 0.10, 2), "independent"))
    for d0 in POWER_D0:
        cells.append((0.0, d0, 0.0, "nested"))
        cells.append((0.30, d0, 0.30, "nested"))
    return cells


def power_rule(rows: list[dict]) -> dict:
    """§8.4: escalate when K-CAUSAL's probability at Δ = 0.30 (d0 = 0.08) or K-NULL's joint
    probability at Δ = Δ_0.6 = 0 (each d0) is below 0.95, under the independent model."""
    at = {(r["delta"], r["d0"], r["delta06"], r["model"]): r["rates"] for r in rows}
    causal = at[(0.30, 0.08, 0.30, "independent")]["K-CAUSAL"]
    null = {d0: at[(0.0, d0, 0.0, "independent")]["K-NULL"] for d0 in POWER_D0}
    ok = causal >= 0.95 and all(v >= 0.95 for v in null.values())
    return {"causal_at_0.30_d0_0.08": causal,
            "null_joint_at_0": {str(k): v for k, v in null.items()},
            "escalate": not ok}  # fmt: skip


# ----- record helpers --------------------------------------------------------------------------
def write_json(path, value) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=True) + "\n")
