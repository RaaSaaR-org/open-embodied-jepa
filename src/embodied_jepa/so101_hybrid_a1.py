"""TASK-102 amendment A1 (``docs/experiments/so101_hybrid_lift_v1.md`` §18): the scripted jaw
rule, the arm-only hand-over, the arm-channel CEM planner with an optional warm start, the
planner grid on development cohort Q, the per-step arm pose cost of the privileged planner check
O-pose, and the selection rule and rows of §18.4.

NumPy only at import. TASK-101's, TASK-102's and Δa's modules are imported, never edited, and
``schema_planning.py`` is subclassed, not changed. Nothing here is registered or imported by
``embodied_jepa``. Simulator truth enters only the goal demonstration, the privileged O-pose
check and the reported logs.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from embodied_jepa import so101_approach_diag as ad
from embodied_jepa import so101_closed_loop as cl
from embodied_jepa import so101_hybrid as hy
from embodied_jepa.contracts import ContractError, validate_costs
from embodied_jepa.planning import Plan
from embodied_jepa.schema_planning import SchemaCEMConfig, SchemaCEMPlanner

PROTOCOL = "docs/experiments/so101_hybrid_lift_v1.md"
SECTION = "§18 (amendment A1)"
TASK = "TASK-102"
STUDY = "A1 planner grid"
GuardError = cl.GuardError
Cohort = cl.Cohort
ROOT = Path(__file__).resolve().parents[2]

# ----- cohorts, seeds and salt (§18.3) --
COHORTS = {
    "Q": Cohort("Q", 105000, 8, (105008, 105023)),
    "debug": Cohort("debug", 105900, 10, None),  # plumbing and timing only; never read
}
Q_RESETS = 8
SALTS = {"cem": 10118}


def cohort(name: str) -> Cohort:
    if name not in COHORTS:
        raise ContractError(f"unknown cohort {name!r}")
    return COHORTS[name]


def check_seed(name: str, seed) -> int:
    if type(seed) is not int and not isinstance(seed, np.integer):
        raise GuardError(f"G-seed: {seed!r} is not an integer seed")
    seed = int(seed)
    if not cohort(name).contains(seed):
        raise GuardError(f"G-seed: {seed} is outside cohort {name}")
    return seed


def keep_cohort(name: str, demonstrate) -> dict:
    """TASK-101 §4's replacement rule on A1's cohorts: ``demonstrate(seed) -> (ok, reason)``
    over the main seeds in order; a rejected seed's slot takes the next reserve seed."""
    c = cohort(name)
    reserve = list(c.reserve_seeds)
    kept, rejected = [], []
    for seed in c.seeds:
        candidate = seed
        while True:
            ok, reason = demonstrate(candidate)
            if ok:
                kept.append(candidate)
                break
            rejected.append({"seed": candidate, "slot": len(kept), "reason": reason})
            if not reserve:
                raise GuardError(f"G-reserve: cohort {name}'s reserve is exhausted")
            candidate = reserve.pop(0)
    return {"kept": kept, "rejected": rejected, "drawn": len(kept) + len(rejected)}


def planner_seed(reset_seed: int) -> int:
    """§18.3: 10118 * 10^6 + reset seed, the same for every configuration and cost on a reset."""
    return SALTS["cem"] * 10**6 + int(reset_seed)


# ----- (i) the jaw out of the approach (§18.2) --
TAU_H = 0.08  # Stage 0's P value (R28.112), for the grid
T1 = hy.T1
JAW_OPEN_ABOVE_RAD = cl.POSE_JAW_OPEN_ABOVE_RAD


def scripted_gripper(goal_q) -> float:
    """+1 if subgoal 1's jaw is above 0.785 rad, else -1 (TASK-101's H-pose jaw law). Reads no
    measured state; the embodiment's 0.25 rad-per-command rate limit gives the opening rate."""
    g = np.asarray(goal_q, np.float64)
    if g.shape != (cl.JOINTS,) or not np.isfinite(g).all():
        raise ContractError("subgoal 1's joints must be finite [6]")
    return 1.0 if g[cl.JAW] > JAW_OPEN_ABOVE_RAD else -1.0


class ArmHandoverRule:
    """§18.2: ``hy.HandoverRule`` with the jaw condition dropped. Phase 1 ends when, for 3
    consecutive commands, the measured arm joints are within ``tau_h`` rad RMS of subgoal 1's, or
    after ``timeout`` phase-1 commands. ``update(q)`` is called once per command before acting and
    returns 1 (phase 1 acts) or 2 (phase 1 is over). The measured jaw error at the hand-over is
    recorded, never read."""

    def __init__(self, goal_q, tau_h: float = TAU_H, timeout: int = T1):
        g = np.asarray(goal_q, np.float64)
        if g.shape != (cl.JOINTS,) or not np.isfinite(g).all():
            raise ContractError("subgoal 1's joints must be finite [6]")
        if not (math.isfinite(tau_h) and 0 < tau_h <= hy.TAU_MAX):
            raise ContractError(f"tau_h must be in (0, {hy.TAU_MAX}]")
        if type(timeout) is not int or timeout < 1:
            raise ContractError("timeout must be a positive integer")
        self.goal, self.tau_h, self.timeout = g, float(tau_h), int(timeout)
        self.phase, self.near, self.on_phase1, self.command = 1, 0, 0, 0
        self.event: dict | None = None
        self.closest = math.inf
        self.longest_near = 0

    def is_near(self, q) -> bool:
        return bool(hy.arm_rms(np.asarray(q, np.float64), self.goal) <= self.tau_h)

    def update(self, q) -> int:
        if self.phase == 1:
            q = np.asarray(q, np.float64)
            rms = float(hy.arm_rms(q, self.goal))
            self.closest = min(self.closest, rms)
            self.near = self.near + 1 if rms <= self.tau_h else 0
            self.longest_near = max(self.longest_near, self.near)
            reached = self.near >= hy.HANDOVER_CONSECUTIVE
            timed_out = self.on_phase1 >= self.timeout
            if reached or timed_out:
                self.phase = 2
                self.event = {
                    "command": self.command,
                    "how": "tolerance" if reached else "timeout",
                    "phase1_commands": self.on_phase1,
                    "arm_rms_rad": rms,
                    "jaw_error_rad": float(q[cl.JAW] - self.goal[cl.JAW]),
                }
            else:
                self.on_phase1 += 1
        self.command += 1
        return self.phase


# ----- the grid (§18.3) --
FORMS = ("last", "mean")
HORIZONS = (8, 16)
EXECUTE = (1, 4)
WARM = (False, True)
REFERENCE_BUDGET = {"samples": 256, "iterations": 6, "elites": 32}
LARGER_BUDGET = {"samples": 1024, "iterations": 12, "elites": 128}
COSTS = ("pose", "J")


def config_name(cost: str, h: int, k: int, warm: bool) -> str:
    return f"{cost}-H{h}-k{k}-w{int(bool(warm))}"


def _grid() -> dict:
    out = {}
    for form in FORMS:
        for h in HORIZONS:
            for k in EXECUTE:
                for warm in WARM:
                    out[config_name(form, h, k, warm)] = {
                        "form": form,
                        "horizon": h,
                        "execute": k,
                        "warm": warm,
                        **REFERENCE_BUDGET,
                    }
    return out


CONFIGS = _grid()
BIG_SUFFIX = "@1024x12"


def declared_index(name: str) -> int:
    """§18.4 (5): k = 1 before k = 4, warm start off before on, H = 16 before H = 8, ``last``
    before ``mean``."""
    check_config(name)
    ranked = sorted(
        CONFIGS,
        key=lambda n: (
            CONFIGS[n]["execute"],
            CONFIGS[n]["warm"],
            -CONFIGS[n]["horizon"],
            CONFIGS[n]["form"] != "last",
        ),
    )
    return ranked.index(name.removesuffix(BIG_SUFFIX))


def check_config(name: str) -> dict:
    base = name.removesuffix(BIG_SUFFIX)
    if base not in CONFIGS:
        raise ContractError(f"unknown A1 configuration {name!r}")
    cfg = dict(CONFIGS[base])
    if name.endswith(BIG_SUFFIX):
        cfg |= LARGER_BUDGET
    return cfg


# ----- the arm-channel CEM planner (§18.2, §18.3) --
def planner_config(name: str, goal_q, seed: int) -> SchemaCEMConfig:
    """TASK-101's planner settings with the grid configuration's horizon and budget, and the
    gripper channel's bounds both at the scripted value (CEM searches the 5 arm channels)."""
    from embodied_jepa.action_schemas import JOINT_DELTA_SO101_V0

    cfg = check_config(name)
    g = scripted_gripper(goal_q)
    return SchemaCEMConfig(
        action_schema=JOINT_DELTA_SO101_V0,
        horizon=int(cfg["horizon"]),
        samples=int(cfg["samples"]),
        iterations=int(cfg["iterations"]),
        elites=int(cfg["elites"]),
        seed=planner_seed(seed),
        action_penalty=cl.CEM["action_penalty"],
        minimum_std=cl.CEM["minimum_std"],
        lower_bounds=(-1.0,) * cl.ARM_JOINTS + (g,),
        upper_bounds=(1.0,) * cl.ARM_JOINTS + (g,),
        project_candidates=True,
    )


def cold_mean(config: SchemaCEMConfig) -> np.ndarray:
    """float32 [H, 6]: zero arm deltas and the scripted gripper channel (CEM's cold mean)."""
    lower = np.asarray(config.lower_bounds, np.float32)
    upper = np.asarray(config.upper_bounds, np.float32)
    row = np.clip(np.zeros(config.dimension), lower, upper).astype(np.float32)
    return np.repeat(row[None], config.horizon, 0)


def shifted_mean(previous, executed: int, config: SchemaCEMConfig) -> np.ndarray:
    """§18.3's warm start: the previous plan's best requested sequence shifted by the ``executed``
    commands, its last rows filled with the cold mean."""
    p = np.asarray(previous, np.float32)
    h = config.horizon
    if p.shape != (h, config.dimension):
        raise ContractError("the previous plan must be [H, 6]")
    m = int(executed)
    if not 1 <= m <= h:
        raise ContractError("the shift must lie in 1..H")
    out = cold_mean(config)
    out[: h - m] = p[m:]
    return out


class ArmCEMPlanner(SchemaCEMPlanner):
    """``SchemaCEMPlanner`` whose ``plan`` takes an optional initial mean (the warm start). With
    ``init_mean=None`` it is ``SchemaCEMPlanner.plan`` line for line (a test holds it to the same
    candidates, costs and plan bit for bit)."""

    def plan(
        self,
        model,
        latent,
        goal_latent,
        *,
        batch_size: int = 1,
        projector=None,
        init_mean=None,
    ) -> Plan:
        cfg = self.config
        dimension = cfg.dimension
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        if cfg.project_candidates and projector is None:
            raise ContractError("candidate projection requires an embodiment projector")
        start = time.perf_counter()
        projection_seconds, feasible_count = 0.0, 0
        shape = (batch_size, cfg.horizon, dimension)
        lower = np.asarray(cfg.lower_bounds, dtype=np.float32)
        upper = np.asarray(cfg.upper_bounds, dtype=np.float32)
        if init_mean is None:
            mean = (
                np.broadcast_to(np.clip(np.zeros(dimension), lower, upper), shape)
                .astype(np.float32)
                .copy()
            )
        else:
            m0 = np.asarray(init_mean, np.float32)
            if m0.shape != shape[1:] or not np.isfinite(m0).all():
                raise ContractError("init_mean must be a finite [H, A] array")
            mean = np.broadcast_to(np.clip(m0, lower, upper), shape).astype(np.float32).copy()
        std = np.ones(shape, dtype=np.float32)
        best_actions = mean.copy()
        best_requested = mean.copy()
        best_cost = np.full(batch_size, np.inf)
        for _ in range(cfg.iterations):
            candidates = np.clip(
                self.rng.normal(mean[:, None], std[:, None], (batch_size, cfg.samples, *shape[1:])),
                lower,
                upper,
            ).astype(np.float32)
            candidates[:, 0] = mean
            if cfg.samples > 1:
                candidates[:, 1] = best_requested
            requested = candidates.copy()
            feasible = np.ones((batch_size, cfg.samples), dtype=bool)
            if cfg.project_candidates:
                projection_start = time.perf_counter()
                projection = projector(requested)
                projection_seconds += time.perf_counter() - projection_start
                if projection.actions.shape != candidates.shape:
                    raise ContractError("projected candidate shape changed")
                candidates, feasible = projection.actions, projection.feasible
            feasible_count += int(feasible.sum())
            if np.any(~feasible.any(axis=1)):
                raise ContractError("no feasible candidate sequence")
            predictions = model.predict(latent, candidates)
            distances = model.distance(predictions, goal_latent)
            validate_costs(distances, (batch_size, cfg.samples, cfg.horizon))
            costs = distances[:, :, -1].astype(np.float64)
            costs += cfg.action_penalty * np.square(candidates).mean(axis=(2, 3))
            costs[~feasible] = np.inf
            for batch in range(batch_size):
                indices = np.argsort(costs[batch], kind="stable")[
                    : min(cfg.elites, int(feasible[batch].sum()))
                ]
                winner = indices[0]
                if costs[batch, winner] < best_cost[batch]:
                    best_cost[batch] = costs[batch, winner]
                    best_actions[batch] = candidates[batch, winner]
                    best_requested[batch] = requested[batch, winner]
                elite = requested[batch, indices]
                mean[batch] = elite.mean(axis=0)
                std[batch] = np.maximum(elite.std(axis=0), cfg.minimum_std)
        return Plan(
            best_actions,
            best_cost.astype(np.float32),
            time.perf_counter() - start,
            batch_size * cfg.samples * cfg.iterations,
            asdict(cfg),
            best_requested,
            projection_seconds,
            feasible_count,
        )


# ----- O-pose's per-step cost (§18.3) --
def arm_pose_steps(q_steps, goal_q, tau_h: float = TAU_H) -> np.ndarray:
    """float64 [..., T]: (arm RMS to subgoal 1 after each command / τ_h)^2 (no jaw term)."""
    return ad.arm_rms(q_steps, goal_q) ** 2 / float(tau_h) ** 2


def form_steps(step_costs, form: str) -> np.ndarray:
    """``last``: the per-step costs as they are (the planner reads the last); ``mean``: their
    running mean over the horizon (the planner reads the mean over every command), as TASK-101's
    ``LatentCost``."""
    e = np.asarray(step_costs, np.float64)
    if form not in FORMS:
        raise ContractError(f"form must be one of {FORMS}")
    if form == "mean":
        e = np.cumsum(e, axis=-1) / np.arange(1, e.shape[-1] + 1)
    return e


# ----- the selection rule and rows (§18.4) --
ADMIT = 7  # O-pose hand-overs of 8
J_FLOOR = 4  # J hand-overs of 8
ROWS = ("A1-VOID", "A1-PLANNER-STILL-BOUND", "A1-MODEL-BOUND", "A1-CONFIG")


def admitted(pose_counts: dict) -> list[str]:
    if sorted(pose_counts) != sorted(CONFIGS):
        raise ContractError("O-pose counts need every grid configuration")
    for v in pose_counts.values():
        if not 0 <= int(v) <= Q_RESETS:
            raise ContractError("counts must lie in 0..8")
    return [n for n in CONFIGS if int(pose_counts[n]) >= ADMIT]


def order(names, pose_counts: dict, j: dict) -> list[str]:
    """§18.4's order: ``j[name]`` = {"h", "commands_median", "closest_median"}."""
    for n in names:
        if n not in j:
            raise ContractError(f"J was not measured on admitted configuration {n}")

    def key(n):
        r = j[n]
        return (
            -int(r["h"]),
            -int(pose_counts[n.removesuffix(BIG_SUFFIX)]),
            float(r["commands_median"]),
            float(r["closest_median"]),
            declared_index(n),
        )

    return sorted(names, key=key)


def needs_larger_budget(pose_counts: dict, j: dict) -> str | None:
    """§18.3 step 3: the admitted configuration to re-run at 1 024 x 12, or None."""
    names = admitted(pose_counts)
    if not names:
        return None
    ranked = order(names, pose_counts, j)
    if int(j[ranked[0]]["h"]) >= J_FLOOR:
        return None
    return ranked[0]


def row(*, void: bool, pose_counts: dict | None, j: dict | None, big: dict | None = None) -> dict:
    """First match over §18.4's rows. ``big``: {"config": name, "h", ...} from step 3, required
    exactly when ``needs_larger_budget`` names a configuration."""
    if void:
        return {"row": ROWS[0], "chosen": None, "admitted": []}
    names = admitted(pose_counts)
    if not names:
        return {"row": ROWS[1], "chosen": None, "admitted": []}
    ranked = order(names, pose_counts, j or {})
    if int(j[ranked[0]]["h"]) >= J_FLOOR:
        return {"row": ROWS[3], "chosen": ranked[0], "admitted": names, "ranked": ranked}
    want = ranked[0]
    if big is None:
        raise ContractError(f"§18.3 step 3 must run first, on {want}")
    if big["config"] != want + BIG_SUFFIX:
        raise ContractError(f"the larger budget must run on {want}")
    if int(big["h"]) >= J_FLOOR:
        return {"row": ROWS[3], "chosen": big["config"], "admitted": names, "ranked": ranked}
    return {"row": ROWS[2], "chosen": None, "admitted": names, "ranked": ranked}


# ----- budget and pins (§18.6) --
GRID_CAP_GPU_H = 3.0
PINNED = {
    "src/embodied_jepa/so101_approach_diag.py": (
        "f091d2235dc38bfc4ea3b5a54d845aa063ae61ce9d3531f54ffa02998600eb1d"
    ),
    "src/embodied_jepa/so101_approach_diag_runtime.py": (
        "38702d349bc04f1aac08c94d8a552bc4e4fd56696fbaf3649bc4be6abc5df700"
    ),
    "scripts/run_task102_approach.py": (
        "e21409823eca87aa5cd9e4f7f8440d51d0315eac78c0f68a6cc58c01bd0965f7"
    ),
    "scripts/summarize_task102_approach.py": (
        "c1fa14c899ad1f529c708a7ddf6033fb6165a81cf12e2eb4d4f522bb5e96e49b"
    ),
    "scripts/run_task102_approach_stage.sh": (
        "cd45ff6a77f63ae5c315ad451b70b5796e9d30b7d162c3e0c5b3dbd7da79bf7c"
    ),
    "docs/experiments/so101_approach_diagnostic_v1.md": (
        "76feb435f600f0ce60bd7bda1d55f2736c1f8aa1ade38ac8fe07476cd24d1f3e"
    ),
    "docs/experiments/so101_approach_diagnostic_v1_results.md": (
        "5ef2fe3d2b7d3086d4c4bb7e2e3af5175cc127803cf9f538f92baa95af9e3a0f"
    ),
    "src/embodied_jepa/schema_planning.py": (
        "bcfc76b45d4697b90188ce5b61909ec036d7059d41bebd07c76462850910796a"
    ),
}


def differing_pins(root: Path = ROOT) -> list[str]:
    return [
        rel
        for rel, pin in PINNED.items()
        if hashlib.sha256((root / rel).read_bytes()).hexdigest() != pin
    ]


def median(values) -> float:
    return ad.median(values)
