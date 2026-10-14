"""TASK-093: a rate-projected planner -- the grasp-rate projector of CEM's candidates, Stage P's
executed measures, references, bar rule, degeneracy checks and rows, the B rows, the closed-loop
equivalence and rate measures, the power and row-rate recompute, the cohorts and seeds, and the
pins of every earlier file TASK-093 imports (protocol
``docs/experiments/grasp_rate_projected_planner_v1.md``).

* The projector (§4.1): each candidate's right-grasp channel (index 13) is clipped cumulatively
  onto the embodiment's rate set, step by step, from the last accepted right-hand actuator targets
  T₀, with ``embodiment._prepare_side``'s own per-joint formula (float64 bounds, the float64
  ``np.clip``, the synergy line, ``_joint_commands``' float32 cast and command-limit clip). Every
  other channel passes through bitwise. The formula is copied because ``embodiment.py`` is frozen
  and full projection is too slow (§3 rank 3); ``embodiment.py`` and ``configs/g1_sim_action.json``
  are pinned below, and Stage 0 checks the copy bitwise against ``robot.project_candidates`` and
  against executed commands.
* Stage P (§6.1-§6.2): C8@R1 and H8@R2 on the plan's executed grasp path, the references (open /
  release, random with salt 9011, closed with the vertical command -1), the bar rule, the
  degeneracy checks and the rows.
* The closed loop (§6.3): the B rows (TASK-090's sign-flip test, bootstrap and bar code), the
  per-command equivalence of requested and applied channel 13, the request-level off-support
  share, and the power and row-rate recompute (§6.4).

TASK-087's, TASK-088's, TASK-089's, TASK-090's, TASK-091's and TASK-092's code is imported and
never changed. NumPy only at import. Simulator truth never enters the projector, a model input or
a planning cost: T₀ is the embodiment's own command state. Nothing here is registered or imported
by ``embodied_jepa``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from embodied_jepa import grasp_corpus_wm as gc
from embodied_jepa import grasp_corpus_wm_planned_close as pcl
from embodied_jepa import grasp_rate_aware as ra
from embodied_jepa import zero_shot as zs
from embodied_jepa import zero_shot_grasp as zg
from embodied_jepa.constraints import CandidateProjection
from embodied_jepa.contracts import ContractError

TASK = "TASK-093"
PROTOCOL = "docs/experiments/grasp_rate_projected_planner_v1.md"

GRASP_DIM = zg.GRASP_DIM  # 13, the right-grasp channel
Z_DIM = zg.Z_DIM  # 8, the right vertical command
HORIZON = zs.CEM["horizon"]  # 8
SETTING = "step-mean"  # TASK-089's G-chg, λ = 2.0552 (§4.3)
LAMBDA = pcl.LAMBDA
RATE_LIMIT = ra.RATE_LIMIT  # 2 / 11 per command (§1.2)
SIDE = "right"

# ----- seeds and salts (§9) --------------------------------------------------------------------
SALTS = {"random_reference": 9011}
MODEL_SEEDS = zs.MODEL_SEEDS  # 87100-87102, TASK-087's G (no training)
DEBUG_SEEDS = range(91940, 92000)  # Stage 0 plumbing and smoke; never counted
COHORTS = {
    "debug-roots": range(91940, 91960),
    "debug-grasp": range(91960, 92000),
    "cal-roots": range(916000, 916064),
    "p-roots": range(917000, 917064),
    "bd-grasp": range(91900, 91932),
    "bs-grasp": range(918000, 918128),
}
ROOT_COHORTS = ("debug-roots", "cal-roots", "p-roots")
GRASP_COHORTS = ("debug-grasp", "bd-grasp", "bs-grasp")
BS_N = len(COHORTS["bs-grasp"])  # 128 (§6.4)
DETERMINISM = ("M-proj", 87100, 4)  # 918000-918003, inside BS (§6.3)
# Ranges of earlier tasks TASK-093 must not touch (§9)
EARLIER_RANGES = (
    dict(ra.EARLIER_RANGES)
    | {f"TASK-092 {k}": v for k, v in ra.COHORTS.items()}
    | {
        "TASK-092 D-RATE (spent)": range(90810, 90842),
        "TASK-092 D-RATE′": range(90850, 90882),
    }
)

# ----- arms (§5) -------------------------------------------------------------------------------
CLOSED_LOOP_ARMS = {"M-proj": True, "M-v1": False}  # arm -> candidates projected
BD_SEEDS = {"M-proj": (87100,), "M-v1": (87100,)}
BS_SEEDS = {"M-proj": MODEL_SEEDS, "M-v1": MODEL_SEEDS}
BASELINE_ARMS = zs.BASELINE_ARMS  # hold, random, ik, ik-nopress
P_ARMS = ("M-proj", "M-v1")  # Stage P's planners, per seed
JOB_CAP_SECONDS = pcl.JOB_CAP_SECONDS  # 7 200 s per GPU job
SPLIT_TRIGGER_SECONDS = pcl.SPLIT_TRIGGER_SECONDS  # 5 400 s (§10, §14)
BS_PARTS = 2  # 918000-918063 and 918064-918127 (§6.3; Stage 0 may raise it)


def check_seed(cohort: str, seed: int) -> None:
    if cohort not in COHORTS:
        raise ContractError(f"unknown cohort {cohort}")
    if int(seed) not in COHORTS[cohort]:
        raise ContractError(f"seed {seed} is not in cohort {cohort}")


def check_run(cohort: str, arm: str, model_seed) -> None:
    """§5, §6.3: which (arm, model seed) a closed-loop cohort admits."""
    if cohort not in GRASP_COHORTS:
        raise ContractError(f"{cohort} is not a closed-loop cohort")
    if arm in BASELINE_ARMS:
        if model_seed is not None:
            raise ContractError("baseline arms take no model seed")
        return
    if arm not in CLOSED_LOOP_ARMS:
        raise ContractError(f"unknown arm {arm}")
    allowed = BD_SEEDS[arm] if cohort == "bd-grasp" else BS_SEEDS[arm]
    if model_seed not in allowed:
        raise ContractError(f"{arm}-{model_seed} is not a run of {cohort}")


def needs_parts(cohort: str, arm: str) -> int | None:
    """BS world-model runs run as BS_PARTS jobs; everything else as one job."""
    return BS_PARTS if cohort == "bs-grasp" and arm in CLOSED_LOOP_ARMS else None


def part_ranges(cohort: str, parts: int) -> list[range]:
    """The contiguous reset ranges of a cohort's ``parts`` jobs (918000-918063, 918064-918127)."""
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


def model_name(arm: str, seed: int) -> str:
    """``M-proj-87100``, ``M-v1-87100``."""
    return f"{arm}-{int(seed)}"


# ----- pins (§10, §14) -------------------------------------------------------------------------
CHECKPOINT_SHA256 = {f"G-v1-{s}": ra.CHECKPOINT_SHA256[f"G-v1-{s}"] for s in MODEL_SEEDS}
FEATURE_REPORT_SHA256 = ra.FEATURE_REPORT_SHA256  # f5648a35...7eae
MOMENTS_SHA256 = ra.MOMENTS_SHA256
IMPLEMENTATION_SHA256 = ra.IMPLEMENTATION_SHA256
# Earlier tasks' code that TASK-093 imports and does not edit (§14): TASK-092's pins (TASK-087's
# model, data and driver code, TASK-090's and TASK-091's frozen files), TASK-092's modules and
# scripts, and the embodiment, its action manifest (the rate limit's source), the candidate
# contract and the play scene. The runner refuses to start when any of these differs.
FROZEN_FILES_SHA256 = dict(ra.FROZEN_FILES_SHA256) | {
    "src/embodied_jepa/grasp_rate_aware.py": (
        "61d5b9122b98a82fa482893873b157fd1ca710a42312073b8482df1da869c9f7"
    ),
    "src/embodied_jepa/grasp_rate_aware_runtime.py": (
        "e1639e7f6ce492ae6cc1732c586c876212c855a8355815a5539dd28a0ec4468d"
    ),
    "scripts/run_task092.py": "78c93858a8ce03d444604be482ea892326b01caa2951ee2a86c220cffc2d69ca",
    "scripts/summarize_task092.py": (
        "d12fe82bd36d7c0c7bf07e27d8fbc849a9cddb574ab9db251f49dafe1f014a5e"
    ),
    "src/embodied_jepa/embodiment.py": (
        "2632259a8f17db4bd1543f8d6f5155ef47a3891e3712784a42bf294c43506ce9"
    ),
    "src/embodied_jepa/constraints.py": (
        "b97f82443dedc868bc8da441ce0a737426a8cead673fd9b9aa11cc3680633144"
    ),
    "src/embodied_jepa/contracts.py": (
        "5f3afd25d608c3484d24e6aee1f77bdb1c4b487dfe4261d166048df22bd119e9"
    ),
    "src/embodied_jepa/play_corpus.py": (
        "b164507e45f56062effd51866ecefca643bc9cd88dcbcc412f220798c2a436cf"
    ),
    "src/embodied_jepa/simulation.py": (
        "9b8c9e3282e224951ce52242fcadd61e899024748396d3b96640e986d3d67739"
    ),
    "configs/g1_sim_action.json": (
        "f247effe9a9cd5e9ae3a9ad8e13ed036fa652f5a0d39f96b14c98979642c38b1"
    ),
}
OWN_MODULES = (
    "src/embodied_jepa/grasp_rate_projected.py",
    "src/embodied_jepa/grasp_rate_projected_runtime.py",
)


def frozen_files_check(root) -> dict:
    out = {}
    for rel, pin in FROZEN_FILES_SHA256.items():
        path = Path(root) / rel
        out[rel] = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == pin
    return out


def own_modules_sha256(root) -> dict:
    return {rel: hashlib.sha256((Path(root) / rel).read_bytes()).hexdigest() for rel in OWN_MODULES}


# ----- the projector (§4.1) --------------------------------------------------------------------
class RateGeometry:
    """The right hand's constants that ``_prepare_side`` uses for the grasp clip: the synergy's
    open and closed targets (the manifest), the per-command joint step ``max_delta =
    joint_speed_limit_rad_s x control_dt`` (the transport's ``control_dt``, as the embodiment
    reads it), the hand joints' MJCF ranges (float64) and their float32 command limits (rounded
    inward, as ``G1Embodiment.__init__`` makes them)."""

    def __init__(self, opened, closed, max_delta, physical, command):
        self.opened = np.asarray(opened, np.float64)
        self.closed = np.asarray(closed, np.float64)
        self.midpoint = (self.opened + self.closed) / 2
        self.slope = (self.closed - self.opened) / 2
        self.max_delta = max_delta
        self.physical = np.asarray(physical, np.float64)
        self.command = np.asarray(command, np.float32)
        if self.opened.shape != (7,) or self.closed.shape != (7,):
            raise ContractError("a hand synergy has seven joints")
        if self.physical.shape != (7, 2) or self.command.shape != (7, 2):
            raise ContractError("hand limits must be [7, 2]")
        self.moving = np.abs(self.slope) >= 1e-12

    @classmethod
    def from_robot(cls, robot, side: str = SIDE) -> RateGeometry:
        m = robot.manifest
        ids = robot.hand_ids[side]
        return cls(
            m[f"{side}_open_rad"],
            m[f"{side}_closed_rad"],
            m["joint_speed_limit_rad_s"] * robot.sim.control_dt,
            robot.model.jnt_range[ids],
            robot._command_limits[ids],
        )

    def describe(self) -> dict:
        return {
            "opened": self.opened.tolist(),
            "closed": self.closed.tolist(),
            "max_delta": float(self.max_delta),
            "physical": self.physical.tolist(),
            "command": self.command.astype(np.float64).tolist(),
            "rate_limit_steepest": float(self.max_delta / np.abs(self.slope).max()),
        }

    def bounds(self, prior):
        """``_prepare_side``'s rate-feasible grasp interval for prior targets ``[..., 7]``
        (float64): per moving joint (prior -+ max_delta - centre) / slope, intersected with
        [-1, 1]. Raises where it is empty or a fixed joint is out of reach, as the embodiment."""
        prior = np.asarray(prior, np.float64)
        md = self.max_delta
        fixed = ~self.moving
        if fixed.any() and np.any(np.abs(self.midpoint[fixed] - prior[..., fixed]) > md + 1e-6):
            raise ContractError(f"{SIDE} grasp has no rate-feasible synergy")
        c, s = self.midpoint[self.moving], self.slope[self.moving]
        p = prior[..., self.moving]
        e1 = (p - md - c) / s
        e2 = (p + md - c) / s
        lower = np.maximum(-1.0, np.minimum(e1, e2).max(-1))
        upper = np.minimum(1.0, np.maximum(e1, e2).min(-1))
        if np.any(lower > upper):
            raise ContractError(f"{SIDE} grasp has no rate-feasible synergy")
        return lower, upper

    def targets(self, grasp):
        """The hand targets of grasps ``[...]`` (float64): the synergy line, then
        ``_joint_commands`` (MJCF range check, float32 cast, command-limit clip)."""
        g = np.asarray(grasp, np.float64)[..., None]
        values = self.opened + (g + 1) * 0.5 * (self.closed - self.opened)
        if (
            not np.isfinite(values).all()
            or np.any(values < self.physical[:, 0])
            or np.any(values > self.physical[:, 1])
        ):
            raise ContractError("prepared joint targets exceed MJCF limits")
        return np.clip(values.astype(np.float32), self.command[:, 0], self.command[:, 1])


def hand_indices(robot, side: str = SIDE) -> list[int]:
    return [robot.actuator_for_joint[int(i)] for i in robot.hand_ids[side]]


def read_t0(robot, side: str = SIDE) -> np.ndarray:
    """T₀ in closed loop (§4.1): the embodiment's last accepted hand targets, as
    ``project_candidates`` reads them (``_projection_targets``, the targets at ``observe``); the
    live targets must equal them (nothing executed since the observation)."""
    held = getattr(robot, "_projection_targets", None)
    if held is None or robot._observation is None:
        raise ContractError("T₀ needs a current observation")
    idx = hand_indices(robot, side)
    t0 = np.asarray(held, np.float64)[idx].copy()
    if not np.array_equal(t0, np.asarray(robot.sim.targets, np.float64)[idx]):
        raise ContractError("the hand targets changed since the observation")
    return t0


def root_t0(geometry: RateGeometry, root_grasp: float) -> np.ndarray:
    """T₀ offline (§4.1): the synergy target of the root's grasp through ``_joint_commands``
    (-1 at R1, +1 at R2), as float64 (the embodiment's target array)."""
    return geometry.targets(np.float64(root_grasp)).astype(np.float64)


def project_grasp(geometry: RateGeometry, requested, t0):
    """The cumulative clip of §4.1 on ``requested`` ``[..., H, 14]`` from T₀ ``[7]``: returns the
    projected commands (float32, channel 13 replaced by float32(g_k), every other channel bitwise
    the request) and the float64 grasps g_k ``[..., H]``."""
    req = np.asarray(requested)
    if req.dtype != np.float32 or req.shape[-1] != 14:
        raise ContractError("requests must be float32 [..., H, 14]")
    lead, horizon = req.shape[:-2], req.shape[-2]
    prior = np.broadcast_to(np.asarray(t0, np.float64), (*lead, 7)).copy()
    out = req.copy()
    grasps = np.empty((*lead, horizon), np.float64)
    for k in range(horizon):
        lower, upper = geometry.bounds(prior)
        g = np.clip(req[..., k, GRASP_DIM].astype(np.float64), lower, upper)
        grasps[..., k] = g
        out[..., k, GRASP_DIM] = g.astype(np.float32)
        prior = geometry.targets(g).astype(np.float64)
    return out, grasps


class GraspRateProjector:
    """``CEMPlanner``'s ``projector`` hook (``CandidateProjector``): ``[B, K, H, 14]`` requests to
    a ``CandidateProjection`` whose actions carry the projected grasp; every candidate stays
    feasible (a grasp clip always has a value; an empty interval raises, as the embodiment)."""

    def __init__(self, geometry: RateGeometry, t0):
        self.geometry = geometry
        self.t0 = np.asarray(t0, np.float64).copy()
        if self.t0.shape != (7,):
            raise ContractError("T₀ has seven hand targets")
        self.calls = 0
        self.first_requested_sha256 = None

    def __call__(self, requested) -> CandidateProjection:
        requested = np.asarray(requested)
        if self.calls == 0:
            self.first_requested_sha256 = hashlib.sha256(requested.tobytes()).hexdigest()
        self.calls += 1
        actions, _ = project_grasp(self.geometry, requested, self.t0)
        return CandidateProjection(actions, np.ones(requested.shape[:2], dtype=bool))


def bits_equal(a, b) -> bool:
    a32, b32 = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return a32.shape == b32.shape and bool(np.array_equal(a32.view(np.uint32), b32.view(np.uint32)))


# ----- Stage P's measures (§6.1) ---------------------------------------------------------------
ROOT_GRASP = ra.ROOT_GRASP  # R1 -1, R2 +1: the demonstrator's grasp before each root
C8_THRESHOLD = -1.0 + 16.0 / 11.0 - 1e-6  # the maximum-rate value at step 8 (about 0.455)
H8_LEVEL = 0.5
MEASURES = ("C8", "H8")
BAR_CAP = 48
BAR_FRACTION = (3, 4)  # 0.75, exact
CONTROL_MIN = 16
P_ROOTS = 64
OFF_SUPPORT_LIMIT = RATE_LIMIT + 1e-6


def executed_path(geometry: RateGeometry, plan, root: str) -> np.ndarray:
    """The plan's executed grasp: its requests through §4.1's cumulative clip from the root's
    grasp (for a projected plan, the plan itself)."""
    t0 = root_t0(geometry, ROOT_GRASP[root])
    _, g = project_grasp(geometry, np.asarray(plan, np.float32), t0)
    return g


def c8(executed_r1) -> bool:
    """C8@R1: the executed grasp at step 8 reaches the maximum-rate value (closing at the
    maximum rate through the horizon)."""
    e = np.asarray(executed_r1, np.float64)
    return bool(e[HORIZON - 1] >= C8_THRESHOLD)


def h8(executed_r2, plan_r2) -> bool:
    """H8@R2: the executed grasp stays above 0.5 at all eight steps and the first command's right
    vertical command (dimension 8) is above 0."""
    e = np.asarray(executed_r2, np.float64)
    return bool(np.all(e[:HORIZON] > H8_LEVEL) and float(np.asarray(plan_r2)[0, Z_DIM]) > 0)


def measure_counts(geometry: RateGeometry, plans_r1, plans_r2) -> dict:
    """C8@R1 and H8@R2 per root for ``[n, H, 14]`` plans at R1 and R2."""
    c = [c8(executed_path(geometry, p, "R1")) for p in plans_r1]
    h = [h8(executed_path(geometry, p, "R2"), p) for p in plans_r2]
    return {"C8": int(sum(c)), "H8": int(sum(h)), "C8_per_root": c, "H8_per_root": h}


def in_rate_set(plan, root: str, tol: float = OFF_SUPPORT_LIMIT) -> bool:
    """Stage P's plumbing check (§6.2): every grasp step of a projected plan moves by at most
    2/11 (+ 1e-6) from the previous one, starting from the root's grasp. (Re-projecting a stored
    plan is not bitwise idempotent: the plan stores float32(g_k) while the cumulative clip carries
    the float64 g_k, so the two differ by float32 rounding, about 1e-7.)"""
    g = np.asarray(plan, np.float64)[:, GRASP_DIM]
    d = np.abs(np.diff(np.concatenate([[ROOT_GRASP[root]], g])))
    return bool(np.all(d <= tol))


def requests_beyond(plan_requested, executed) -> dict:
    """Reported: how many of a projected plan's requested grasp steps lie beyond the limit from
    the previous executed grasp (``executed``, the projected path, from the root's grasp)."""
    r = np.asarray(plan_requested, np.float64)[:, GRASP_DIM]
    return {"steps": len(r), "beyond": int((np.abs(r - np.asarray(executed)) > 1e-6).sum())}


# ----- references (§6.1) -----------------------------------------------------------------------
def _base_plan() -> np.ndarray:
    p = np.zeros((HORIZON, 14), np.float32)
    p[:, 12] = -1.0  # the left grasp, fixed at -1 (FIXED_LOWER = FIXED_UPPER there)
    return p


def open_reference(root: str) -> np.ndarray:
    """Requests -1 at every step; at R2 the vertical command +1 (open / release)."""
    p = _base_plan()
    p[:, GRASP_DIM] = -1.0
    if root == "R2":
        p[:, Z_DIM] = 1.0
    return p


def closed_down_reference() -> np.ndarray:
    """H8 only: requests +1 at every step with the vertical command -1 at R2."""
    p = _base_plan()
    p[:, GRASP_DIM] = 1.0
    p[:, Z_DIM] = -1.0
    return p


def max_rate_reference() -> np.ndarray:
    """Reported only (not a §6.1 reference): requests +1 at every step, so C8 can be met."""
    p = _base_plan()
    p[:, GRASP_DIM] = 1.0
    return p


def random_reference(seed: int) -> dict:
    """CEM's first-iteration distribution, N(0, 1) clipped to the bounds, salt 9011: one 8-step
    sequence at R1 and then one at R2 from ``default_rng([9011, root seed])``."""
    rng = np.random.default_rng([SALTS["random_reference"], int(seed)])
    out = {}
    for root in zg.ROOTS:
        x = rng.normal(0.0, 1.0, (HORIZON, 14))
        out[root] = np.clip(x, zs.FIXED_LOWER, zs.FIXED_UPPER).astype(np.float32)
    return out


def reference_counts(geometry: RateGeometry, seeds) -> dict:
    """Every reference's C8 and H8 counts on the roots ``seeds`` (no model)."""
    seeds = list(seeds)
    rand = [random_reference(s) for s in seeds]
    n = len(seeds)
    out = {
        "open": measure_counts(geometry, [open_reference("R1")] * n, [open_reference("R2")] * n),
        "random": measure_counts(geometry, [r["R1"] for r in rand], [r["R2"] for r in rand]),
        "closed_down": measure_counts(
            geometry, [closed_down_reference()] * n, [closed_down_reference()] * n
        ),
        "max_rate_reported": measure_counts(
            geometry, [max_rate_reference()] * n, [max_rate_reference()] * n
        ),
    }
    for v in out.values():
        v.pop("C8_per_root"), v.pop("H8_per_root")
    return out


# ----- the bar rule and the degeneracy checks (§6.1) -------------------------------------------
def bar_from_control(control_counts) -> int:
    """bar_m = min(48, ceil(0.75 x the median over the three M-v1 seeds' development counts))."""
    counts = [int(c) for c in control_counts]
    if len(counts) != len(MODEL_SEEDS):
        raise ContractError("the bar needs the three control seeds' counts")
    med = sorted(counts)[1]
    num, den = BAR_FRACTION
    return int(min(BAR_CAP, -(-num * med // den)))


def degeneracy(measure: str, control_counts, refs: dict) -> dict:
    """§6.1's checks for one measure: (a) open/release scores 0 (and, for H8, the closed,
    vertical -1 reference too); (b) random scores at most floor(bar / 2); (c) the control's median
    development count is at least 16."""
    if measure not in MEASURES:
        raise ContractError(f"unknown measure {measure}")
    bar = bar_from_control(control_counts)
    med = sorted(int(c) for c in control_counts)[1]
    a = refs["open"][measure] == 0
    if measure == "H8":
        a = a and refs["closed_down"][measure] == 0
    b = refs["random"][measure] <= bar // 2
    c = med >= CONTROL_MIN
    return {
        "bar": bar,
        "control_counts": [int(x) for x in control_counts],
        "control_median": med,
        "open": refs["open"][measure],
        "closed_down": refs["closed_down"][measure] if measure == "H8" else None,
        "random": refs["random"][measure],
        "random_max": bar // 2,
        "a_can_fail": bool(a),
        "b_not_by_chance": bool(b),
        "c_control_min": bool(c),
        "pass": bool(a and b and c),
    }


def s0_row(degeneracy_checks: dict, *, equivalence_ok: bool, power_ok: bool) -> dict:
    """§7: S0-ESCALATE when a degeneracy check, the projector's equivalence tests or the power
    rule fails; otherwise Stage 0 passes (S0-PASS) and the protocol freezes at the merge."""
    failing = sorted(m for m, v in degeneracy_checks.items() if not v["pass"])
    escalate = bool(failing or not equivalence_ok or not power_ok)
    return {
        "row": "S0-ESCALATE" if escalate else "S0-PASS",
        "failing_measures": failing,
        "equivalence_ok": bool(equivalence_ok),
        "power_ok": bool(power_ok),
        "bars": {m: v["bar"] for m, v in degeneracy_checks.items()},
    }


# Stage P's bars, frozen at Stage 0 (§6.1; R27.133): the rule on cal-roots 916000-916063 at
# c7cf7f4 (M-v1's C8 counts 33, 58, 51 -> ceil(0.75 x 51) = 39; H8 64, 64, 61 -> min(48, 48)).
P_BARS: dict | None = {"C8": 39, "H8": 48}


def p_gate(counts: dict, bars: dict) -> bool:
    """Stage P's gate per M-proj seed: C8@R1 >= bar_C8 and H8@R2 >= bar_H8."""
    return bool(counts["C8"] >= bars["C8"] and counts["H8"] >= bars["H8"])


def p_row(passes: dict, *, void: str = "") -> dict:
    """§6.2: P-VOID if a plumbing check failed, P-FAIL when two or more M-proj seeds fail the
    gate, else P-PASS (a seed failing alone stays in the pooled count)."""
    failing = sorted(int(s) for s, ok in passes.items() if not ok)
    if void:
        row = "P-VOID"
    elif len(failing) >= 2:
        row = "P-FAIL"
    else:
        row = "P-PASS"
    return {"row": row, "void_reason": void, "failing_seeds": failing}


# ----- the closed loop (§6.3) ------------------------------------------------------------------
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
GC_ROW_NAMES = {"B-COMP": "B-GAIN", "B-SIZE": "B-BAR"}  # TASK-090's names -> TASK-093's


def b_row(p_by_seed, c_by_seed, random, bar: float, *, void: str = "") -> dict:
    """§6.3's rows, first match. ``p_by_seed`` / ``c_by_seed``: M-proj's and M-v1's ``[3, n]``
    success indicators on the same resets (seed i of each on reset r); ``random``: ``[n]``. The
    logic is TASK-090's ``b_row`` (B-COMP named B-GAIN, B-SIZE named B-BAR)."""
    p = np.asarray(p_by_seed, bool)
    c = np.asarray(c_by_seed, bool)
    r = np.asarray(random, bool)
    if p.shape != c.shape or p.shape[0] != SEEDS_PER_ARM or r.shape != (p.shape[1],):
        raise ContractError("b_row needs [3, n] arms and [n] random")
    n = p.shape[1]
    pr_, cr_ = per_reset(p), per_reset(c)
    k_p, k_c = int(pr_.sum()), int(cr_.sum())
    need, floor = pooled_bar(bar, n), seed_floor(bar, n)
    seeds = [int(x) for x in p.sum(1)]
    p_c = signflip_p(pr_ - cr_)
    p_r = signflip_p(pr_ - SEEDS_PER_ARM * r.astype(np.int64))
    beats_c, beats_r = p_c < P_THRESHOLD, p_r < P_THRESHOLD
    at_bar = k_p >= need
    floor_ok = all(s >= floor for s in seeds)
    boot = reset_bootstrap(p, c)
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
        "k_p": k_p,
        "k_c": k_c,
        "pooled_bar": need,
        "seed_floor": floor,
        "seeds_proj": seeds,
        "seeds_v1": [int(x) for x in c.sum(1)],
        "seeds_proj_interval": [zs.exact_interval(s, n) for s in seeds],
        "seeds_v1_interval": [zs.exact_interval(int(s), n) for s in c.sum(1)],
        "p_beats_v1": p_c,
        "p_beats_random": p_r,
        "beats_v1": bool(beats_c),
        "beats_random": bool(beats_r),
        "at_pooled_bar": bool(at_bar),
        "seed_floor_met": bool(floor_ok),
        "bootstrap_rate_difference": boot,
        "pooled_mcnemar_p_reported": pooled_mcnemar(p, c),
    }


def clause_fires(row: str) -> bool:
    """§8: the abandonment clause fires on B-FAIL only."""
    return row == "B-FAIL"


deferral_measures = pcl.deferral_measures
deferral_summary = pcl.deferral_summary
rate_measures = ra.rate_measures
rate_summary = ra.rate_summary
START_GRASP = ra.START_GRASP  # -1, the settled open hand at every reset


def closed_loop_equivalence(episode: dict) -> dict:
    """§6.3's check (a B-VOID check for M-proj, §14's negative control for M-v1): for every
    executed command, whether the applied channel 13 returned by ``zs.execute`` is bitwise the
    requested one (the plan's first grasp, ``plan_log``)."""
    applied = np.asarray(episode.get("applied", []), np.float32).reshape(-1, 14)
    log = list(episode.get("plan_log", []))
    t = len(applied)
    if len(log) not in (t, t + 1):
        raise ContractError(f"plan log has {len(log)} entries for {t} commands")
    requested = np.asarray([e["grasp"][0] for e in log[:t]], np.float32)
    differ = ~(requested.view(np.uint32) == applied[:, GRASP_DIM].copy().view(np.uint32))
    phase = np.asarray([e["index"] for e in log[:t]], np.int64)
    gp = phase == pcl.GRASP_PHASE
    return {
        "commands": t,
        "channel13_differs": int(differ.sum()),
        "grasp_phase_commands": int(gp.sum()),
        "grasp_phase_channel13_differs": int((differ & gp).sum()),
    }


def request_measures(episode: dict) -> dict:
    """Reported for M-proj (§6.3): the share of the plans' *requested* grasp steps beyond the
    limit from the previous executed or planned grasp (as ``rate_measures`` reads the plans)."""
    applied = np.asarray(episode.get("applied", []), np.float64).reshape(-1, 14)
    log = list(episode.get("plan_log", []))
    t = len(applied)
    if len(log) not in (t, t + 1):
        raise ContractError(f"plan log has {len(log)} entries for {t} commands")
    prev, steps, beyond = START_GRASP, 0, 0
    for k in range(t):
        req = np.asarray(log[k].get("requested_grasp", log[k]["grasp"]), np.float64)
        plan = np.asarray(log[k]["grasp"], np.float64)
        before = np.concatenate([[prev], plan[:-1]])
        steps += len(req)
        beyond += int((np.abs(req - before) > OFF_SUPPORT_LIMIT).sum())
        prev = applied[k, GRASP_DIM]
    return {"plan_steps": steps, "requested_steps_beyond_limit": beyond}


# ----- power and row rates (§6.4) --------------------------------------------------------------
POWER_GRID = (
    ((0.15, 0.05), 128),
    ((0.12, 0.05), 128),
    ((0.10, 0.05), 128),
    ((0.20, 0.05), 128),
    ((0.15, 0.08), 128),
    ((0.10, 0.02), 128),
    ((0.10, 0.05), 192),
    ((0.12, 0.05), 192),
    ((0.10, 0.05), 256),
)
POWER_ALTERNATIVE = (0.15, 0.05)
POWER_SDS = gc.POWER_SDS  # 0, 1, 2
POWER_TARGET = gc.POWER_TARGET  # 0.80
POWER_SIMS = 20_000
ROW_GRID = ((0.10, 0.05), (0.12, 0.05), (0.15, 0.05), (0.15, 0.08))
ROW_BAR = zg.PLAN_GRASP_BAR  # 0.40, the plan's bar (the BD bar is at most this)
ROW_SEED = 93  # a fixed tag in the row simulation's generator (declared here, R27.135)


def power_rule(rows: list[dict]) -> dict:
    """§6.4: n = 128 stands unless the power at the declared alternative and SD 1 is below 0.80
    there (then disclose and escalate before the freeze)."""
    at = {
        (tuple(r["alt"]), r["n"], r["sd"]): r["power"] for r in rows
    }  # fmt: skip
    p = at[(POWER_ALTERNATIVE, BS_N, 1.0)]
    return {"n": BS_N, "power_at_sd1": p, "escalate": bool(p < POWER_TARGET)}


def row_rates(alt, n: int, sd: float, *, sims: int = POWER_SIMS, bar: float = ROW_BAR,
              seed: int = ROW_SEED) -> dict:  # fmt: skip
    """The distribution of the B rows under per-seed rates ``alt`` (M-proj, M-v1): TASK-090's
    simulation (independent per-arm, per-reset N(0, sd) logit effects shared by an arm's three
    seeds) with random at 0 successes, each draw passed through TASK-090's ``b_row`` at the bar.
    The generator is ``default_rng([9006, n, round(10 sd), seed, round(1000 p),
    round(1000 c)])``."""
    rng = np.random.default_rng(
        [gc.SALTS["bootstrap"], int(n), int(round(10 * sd)), int(seed),
         int(round(1000 * alt[0])), int(round(1000 * alt[1]))]
    )  # fmt: skip
    ap, ac = (gc._intercept_for_rate(p, sd) for p in alt)
    random = np.zeros(n, bool)
    counts: dict = {}
    for _ in range(sims):
        zp = rng.normal(0, sd, n) if sd else np.zeros(n)
        zc = rng.normal(0, sd, n) if sd else np.zeros(n)
        pp = 1 / (1 + np.exp(-(ap + zp)))
        pc_ = 1 / (1 + np.exp(-(ac + zc)))
        p = rng.random((SEEDS_PER_ARM, n)) < pp
        c = rng.random((SEEDS_PER_ARM, n)) < pc_
        name = gc.b_row(p, c, random, bar)["row"]
        name = GC_ROW_NAMES.get(name, name)
        counts[name] = counts.get(name, 0) + 1
    return {
        "alt": list(alt),
        "n": n,
        "sd": sd,
        "sims": sims,
        "bar": bar,
        "rates": {k: v / sims for k, v in sorted(counts.items())},
    }


def write_json(path, value) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=True) + "\n")
