"""TASK-102 (SO-101 S5, second study): a hybrid closed loop on ``so101_cube_lift_v1`` in simulation
(``docs/experiments/so101_hybrid_lift_v1.md``) -- cohorts, seeds, salts, the hand-over rule, the
τ_h rule, the training-window rule, G_loc's non-latent inputs and clock, the arms' composition,
the configuration rule on A, statistics, the bar rule, the rows, the check ledger and the budget.

NumPy only at import. G_loc's head, training and selection are ``so101_grasp_policy`` (torch);
the episode loop, the controllers and the worker pool are ``so101_hybrid_runtime``. TASK-101's
code (``so101_closed_loop``, ``so101_closed_loop_runtime``) is imported, never edited. Nothing
here is registered or imported by ``embodied_jepa``. Simulator truth enters only the goal
demonstration, the scorer, the privileged arms (S+G's scripted approach, W+F's follower, F) and
the runner's reported logs; no learned controller reads it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from embodied_jepa import so101_closed_loop as cl
from embodied_jepa.contracts import ContractError

PROTOCOL = "docs/experiments/so101_hybrid_lift_v1.md"
TASK = "TASK-102"
MANIFEST = "benchmarks/manifests/so101-hybrid-lift-v1.json"
GuardError = cl.GuardError
Cohort = cl.Cohort

# ----- cohorts and seeds (§7, §8, §12) --
COHORTS = {
    "P": Cohort("P", 103000, 32, (103032, 103047)),
    "Pp": Cohort("Pp", 103060, 32, (103092, 103099)),  # P′, only under §7.7 row 1
    "D": Cohort("D", 103100, 64, (103164, 103199)),
    "S": Cohort("S", 103200, 128, (103400, 103463)),
    "A": Cohort("A", 103500, 32, (103532, 103547)),
    "Dp": Cohort("Dp", 103600, 64, (103664, 103699)),  # D′, only under §7.7 row 1
    "Ap": Cohort("Ap", 103700, 32, (103732, 103747)),  # A′, only under §7.7 row 1
    "debug": Cohort("debug", 103900, 100, None),  # plumbing only; never counted
}
GATED_COHORTS = ("S",)
LEDGER_COHORTS = ("P", "Pp", "A", "Ap", "D", "Dp")
COHORT_KIND = {"P": "P", "Pp": "P", "A": "A", "Ap": "A", "D": "D", "Dp": "D", "S": "S"}
LIFT_TASK_NAME = cl.LIFT_TASK_NAME  # so101_cube_lift_v1 (TASK-101's A1), unchanged


def cohort(name: str) -> Cohort:
    if name not in COHORTS:
        raise ContractError(f"unknown cohort {name!r}")
    return COHORTS[name]


def check_seed(name: str, seed) -> int:
    """§12: a seed outside its cohort (main or reserve range) is refused."""
    if type(seed) is not int and not isinstance(seed, np.integer):
        raise GuardError(f"G-seed: {seed!r} is not an integer seed")
    seed = int(seed)
    if not cohort(name).contains(seed):
        raise GuardError(f"G-seed: {seed} is outside cohort {name}")
    return seed


def keep_cohort(name: str, demonstrate) -> dict:
    """§4's replacement (TASK-101's rule) on this task's cohorts: ``demonstrate(seed) -> (ok,
    reason, record)`` over the main seeds in order; a rejected seed's slot takes the next unused
    reserve seed."""
    c = cohort(name)
    reserve = list(c.reserve_seeds)
    kept, rejected = [], []
    for seed in c.seeds:
        candidate = seed
        while True:
            ok, reason, _ = demonstrate(candidate)
            if ok:
                kept.append(candidate)
                break
            rejected.append({"seed": candidate, "slot": len(kept), "reason": reason})
            if not reserve:
                raise GuardError(f"G-reserve: cohort {name}'s reserve is exhausted")
            candidate = reserve.pop(0)
    return {"kept": kept, "rejected": rejected, "drawn": len(kept) + len(rejected)}


SALTS = {
    "cem": 10110,
    "rand": 10111,
    "offset": 10112,
    "bootstrap": 10113,
    "power": 10114,
    "gloc": 10115,
    "marker": 10107,  # part of so101_cube_lift_v1's definition (TASK-101 A1)
}


def planner_seed(reset_seed: int) -> int:
    """§5.1: 10110 * 10^6 + reset seed (the same for every CEM arm on a reset)."""
    return SALTS["cem"] * 10**6 + int(reset_seed)


def rand_rng(reset_seed: int) -> np.random.Generator:
    """§6's rand arm: ``default_rng([10111, reset seed])``."""
    return np.random.default_rng([SALTS["rand"], int(reset_seed)])


def gloc_rng(i: int) -> np.random.Generator:
    """§5.4: G-i's initialisation and batch order from ``default_rng([10115, i])``."""
    return np.random.default_rng([SALTS["gloc"], int(i)])


# ----- the models (§3) --
PRIMARY_J = cl.PRIMARY  # J-100101
TWIN_J = cl.TWIN  # N_J-100101
ARM_JOINTS, JOINTS, JAW = cl.ARM_JOINTS, cl.JOINTS, cl.JAW
LATENT_DIM = cl.LATENT_DIM

# ----- the controller (§5) --
CONFIGS = cl.CONFIGS  # H8-last, H8-mean, H16-last, H16-mean
SUBGOAL_1 = 0  # 0-based: the pre-grasp pose (after command 40)
HANDOVER_JAW_RAD = 0.15
HANDOVER_CONSECUTIVE = 3
T1 = 60  # TASK-101's timeout for subgoal 1
SCRIPTED_APPROACH_COMMANDS = cl.SUBGOAL_COMMANDS[0]  # 40: S+G hands over from command 41
MAX_COMMANDS = cl.MAX_COMMANDS  # 200


def arm_rms(q, goal_q) -> float:
    d = np.asarray(q, np.float64)[:ARM_JOINTS] - np.asarray(goal_q, np.float64)[:ARM_JOINTS]
    return float(np.sqrt(np.mean(d * d)))


class HandoverRule:
    """§5.2: phase 1 ends and G_loc takes over when, for 3 consecutive commands, the measured
    arm joints are within ``tau_h`` rad RMS of subgoal 1's and the measured jaw within 0.15 rad
    of subgoal 1's jaw, or after T₁ = 60 phase-1 commands, whichever comes first.

    ``update(q)`` is called once per command with the measured joints, before acting, and
    returns 1 (phase 1 acts on this command) or 2 (G_loc acts). It mirrors TASK-101's switching
    form: the command that observes the third consecutive near state, or the 61st command, is
    the first command of phase 2. It reads proprioception and the goal pose only. ``closest``
    is the smallest arm RMS to subgoal 1 over the states phase 1 observed (the hand-over state
    included)."""

    def __init__(self, goal_q, tau_h: float, timeout: int = T1):
        g = np.asarray(goal_q, np.float64)
        if g.shape != (JOINTS,) or not np.isfinite(g).all():
            raise ContractError("subgoal 1's joints must be finite [6]")
        if not (math.isfinite(tau_h) and 0 < tau_h <= TAU_MAX):
            raise ContractError(f"tau_h must be in (0, {TAU_MAX}]")
        self.goal, self.tau_h, self.timeout = g, float(tau_h), int(timeout)
        self.phase, self.near, self.on_phase1, self.command = 1, 0, 0, 0
        self.event: dict | None = None
        self.closest = math.inf

    def is_near(self, q) -> bool:
        q = np.asarray(q, np.float64)
        return (
            arm_rms(q, self.goal) <= self.tau_h and abs(q[JAW] - self.goal[JAW]) <= HANDOVER_JAW_RAD
        )

    def update(self, q) -> int:
        if self.phase == 1:
            q = np.asarray(q, np.float64)
            rms = arm_rms(q, self.goal)
            self.closest = min(self.closest, rms)
            self.near = self.near + 1 if self.is_near(q) else 0
            reached = self.near >= HANDOVER_CONSECUTIVE
            timed_out = self.on_phase1 >= self.timeout
            if reached or timed_out:
                self.phase = 2
                self.event = {
                    "command": self.command,
                    "how": "tolerance" if reached else "timeout",
                    "phase1_commands": self.on_phase1,
                    "arm_rms_rad": rms,
                    "jaw_error_rad": float(q[JAW] - self.goal[JAW]),
                }
            else:
                self.on_phase1 += 1
        self.command += 1
        return self.phase


# ----- τ_h (§5.2) --
ENVELOPE_LEVELS_RAD = (0.02, 0.04, 0.06, 0.08, 0.12, 0.16)  # TASK-101 A2's declared levels
ENVELOPE_PASS = 30
TAU_MAX = 0.12
P_LEVELS_RAD = (0.0, 0.04, 0.08, 0.12)
TAU_FALLBACK = 0.04
TAU_MARGIN = 3
P_RESETS = 32
GLOC_WEAK_K0 = 16


def tau_env_rule(pre_counts: dict) -> dict:
    """§5.2 step 1: τ_env is the largest envelope level L with every level up to and including L
    lifting at least 30/32 in the "pre" column; τ_max is the next lower declared level (one level
    of margin). ``pre_counts``: ``{level: successes of 32}``."""
    levels = sorted(ENVELOPE_LEVELS_RAD)
    if sorted(float(k) for k in pre_counts) != levels:
        raise ContractError(f"the envelope's pre column needs every level {levels}")
    counts = {float(k): int(v) for k, v in pre_counts.items()}
    tau_env = None
    for level in levels:
        if counts[level] < ENVELOPE_PASS:
            break
        tau_env = level
    if tau_env is None or levels.index(tau_env) == 0:
        return {"tau_env": tau_env, "tau_max": None}
    return {"tau_env": tau_env, "tau_max": levels[levels.index(tau_env) - 1]}


def p_direction(seed: int) -> np.ndarray:
    """§7.4: u = n / RMS(n), n ~ N(0, I5) from ``default_rng([10112, seed])``; one direction per
    seed for every level."""
    n = np.random.default_rng([SALTS["offset"], int(seed)]).standard_normal(ARM_JOINTS)
    return n / math.sqrt(float(np.mean(n * n)))


def p_offset(seed: int, level: float) -> np.ndarray:
    """δ = r · u (RMS exactly r over the five arm joints)."""
    if level not in P_LEVELS_RAD:
        raise ContractError(f"{level} is not a declared P level {P_LEVELS_RAD}")
    return float(level) * p_direction(seed)


def tau_h_rule(k: dict, tau_max: float = TAU_MAX) -> dict:
    """§5.2 step 2: τ_h is the largest r in {0.04, 0.08, 0.12} such that every level from 0.04 up
    to r has k(r) >= k(0) - 3; if none qualifies, τ_h = 0.04 rad (TASK-101's value). τ_h never
    exceeds τ_max. ``k``: ``{level: lifts of 32}`` for every P level."""
    if sorted(float(x) for x in k) != sorted(P_LEVELS_RAD):
        raise ContractError(f"P needs every level {P_LEVELS_RAD}")
    k = {float(x): int(v) for x, v in k.items()}
    for v in k.values():
        if not 0 <= v <= P_RESETS:
            raise ContractError("counts must lie in 0..32")
    floor = k[0.0] - TAU_MARGIN
    tau = None
    for r in P_LEVELS_RAD[1:]:
        if r > tau_max + 1e-12 or k[r] < floor:
            break
        tau = r
    fallback = tau is None
    return {
        "tau_h": TAU_FALLBACK if fallback else tau,
        "fallback": fallback,
        "k": {str(r): k[r] for r in P_LEVELS_RAD},
        "floor": floor,
        "tau_max": tau_max,
        "gloc_weak": k[0.0] < GLOC_WEAK_K0,
    }


# ----- G_loc's inputs (§5.3) --
CLOCK_PERIODS = tuple(4 * 2**i for i in range(8))  # 4 ... 512 commands, geometric
CLOCK_DIM = 1 + 2 * len(CLOCK_PERIODS)  # 17
PROPRIO_DIM = 18  # 6 positions, 6 velocities, 6 joint-position targets
NONLATENT_DIM = PROPRIO_DIM + CLOCK_DIM  # 35
INPUT_DIM = LATENT_DIM + NONLATENT_DIM  # 12 323
NONLATENT_STD_FLOOR = 1e-3
OUTPUT_DIM = JOINTS


def clock(t) -> np.ndarray:
    """t / 100 plus sin and cos at 8 geometric periods from 4 to 512 commands (P-3's form);
    float64 [..., 17]. ``t`` counts the commands since the hand-over."""
    t = np.asarray(t, np.float64)
    phase = 2.0 * np.pi * t[..., None] / np.asarray(CLOCK_PERIODS, np.float64)
    return np.concatenate([t[..., None] / 100.0, np.sin(phase), np.cos(phase)], axis=-1)


def nonlatent(q, qvel, targets, t) -> np.ndarray:
    """G_loc's 35 non-latent inputs, float32 [..., 35]: measured positions [6], velocities [6],
    the current joint-position targets [6] (all cast to float32 as the corpus stores them) and
    the clock [17]."""
    q = np.asarray(q, np.float32)
    qvel = np.asarray(qvel, np.float32)
    targets = np.asarray(targets, np.float32)
    if q.shape[-1] != JOINTS or qvel.shape[-1] != JOINTS or targets.shape[-1] != JOINTS:
        raise ContractError("q, qvel and targets need 6 values each")
    out = np.concatenate([q, qvel, targets, clock(t).astype(np.float32)], axis=-1)
    if not np.isfinite(out).all():
        raise GuardError("G-finite: a non-finite G_loc input")
    return out.astype(np.float32)


# ----- the training windows (§5.4) --
# TASK-100's pretrained g4 feature store and its report (sha256 from TASK-100's evidence
# SHA256SUMS, ./features/pretrained/report.json); the corpus manifest and corpus.json pins are
# TASK-099's (so101_wm.CORPUS_MANIFEST, so101_wm.CORPUS_JSON_SHA256).
FEATURES = cl.TASK100_FEATURES
FEATURE_REPORT_SHA256 = "afe605d5994e1cbe5b09cd2486aa42c971fc4e41dca034945c5e237b46e1508f"
CORPUS = "data/so101-play-v1"
HOLD_LIMIT = 40
SPLIT_SALT = 9902  # the corpus's frozen split (sha256("9902:<seed>"))
TRAIN_SPLITS = ("train", "val")
REFUSED_SPLIT = "test"


def check_split(split: str) -> str:
    """§5.4: the test split is never opened."""
    if split == REFUSED_SPLIT:
        raise GuardError("G-test: G_loc never opens the corpus's test split")
    if split not in TRAIN_SPLITS:
        raise ContractError(f"unknown split {split!r}")
    return split


def grasp_windows(phase, attempts, n_commands: int, *, descend: int, hold: int) -> dict:
    """§5.4's window rule. ``phase`` int [T+1] (the label of command i at index i); ``attempts``
    the corpus's per-attempt facts (``failure``, ``grasped``, ``lift_end_step``); ``n_commands``
    the episode's command count T. One window per attempt that grasped, is not ``early_close``
    and has a ``lift_end_step`` L: commands s .. L - 1, where s is the first command of the last
    contiguous ``descend`` run before L (t = 0 there), plus the ``hold`` commands that follow
    immediately from L on, at most 40. Returns ``{"windows": [...], "excluded": {...}}``."""
    phase = np.asarray(phase)
    windows, excluded = [], {"not_grasped": 0, "early_close": 0, "no_lift_end": 0, "no_descend": 0}
    for i, a in enumerate(attempts):
        lift_end = a.get("lift_end_step")
        if lift_end is None:
            excluded["no_lift_end"] += 1
            continue
        if a.get("failure") == "early_close":
            excluded["early_close"] += 1
            continue
        if a.get("grasped") is not True:
            excluded["not_grasped"] += 1
            continue
        lift_end = int(lift_end)
        if not 0 < lift_end <= n_commands:
            raise ContractError(f"lift_end_step {lift_end} outside 1..{n_commands}")
        j = lift_end - 1
        while j >= 0 and phase[j] != descend:
            j -= 1
        if j < 0:
            excluded["no_descend"] += 1
            continue
        start = j
        while start > 0 and phase[start - 1] == descend:
            start -= 1
        end, holds = lift_end, 0
        while end < n_commands and holds < HOLD_LIMIT and phase[end] == hold:
            end += 1
            holds += 1
        windows.append(
            {"attempt": i, "start": start, "lift_end": lift_end, "end": end, "holds": holds}
        )
    for a, b in zip(windows, windows[1:], strict=False):
        if b["start"] < a["end"]:
            raise ContractError("two windows of one episode overlap")
    return {"windows": windows, "excluded": excluded}


def window_task_success(cube_z_rise, gripper_contact, start: int, end: int) -> bool:
    """Reported (§7.2): the cube centre at least 5 cm up with a gripper contact on 20 consecutive
    frames inside the window's states (frames start .. end)."""
    rise = np.asarray(cube_z_rise, np.float64)[start : end + 1]
    contact = np.asarray(gripper_contact, bool)[start : end + 1]
    run = 0
    for ok in (rise >= 0.05) & contact:
        run = run + 1 if ok else 0
        if run >= 20:
            return True
    return False


# ----- G_loc's training (§5.4) --
GLOC_SEEDS = (0, 1, 2)
TRAIN = {
    "updates": 30_000,
    "lr": 3e-4,
    "lr_final": 3e-5,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "batch": 256,
    "eval_every": 1_000,
    "min_output_std": 0.02,
    "hidden": 512,
}


def select_gloc(curve: list[dict]) -> dict:
    """§5.4: the checkpoint with the lowest val MSE among the eligible ones (every output
    dimension's std over val frames >= 0.02, update > 0); ties to the earlier update.
    ``last_two_triggered`` (reported): the selected update is one of the last two evaluation
    points."""
    eligible = [
        p for p in curve if p["update"] > 0 and min(p["output_std"]) >= TRAIN["min_output_std"]
    ]
    if not eligible:
        return {"selected": None, "eligible": 0, "last_two_triggered": None}
    best = min(eligible, key=lambda p: (p["val_mse"], p["update"]))
    updates = sorted(p["update"] for p in curve)
    return {
        "selected": best["update"],
        "val_mse": best["val_mse"],
        "eligible": len(eligible),
        "last_two_triggered": best["update"] in updates[-2:],
    }


def primary_gloc(selections: dict) -> dict:
    """§5.4: the primary G_loc is the seed whose selected checkpoint has the lowest val MSE; the
    other two are replication policies. ``selections``: ``{"G-i": select_gloc(...)}``."""
    ok = {k: v for k, v in selections.items() if v.get("selected") is not None}
    if not ok:
        return {"primary": None, "replication": [], "weak": True}
    primary = min(ok, key=lambda k: (ok[k]["val_mse"], k))
    return {
        "primary": primary,
        "replication": sorted(k for k in selections if k != primary),
        "weak": False,
    }


# ----- arms (§6) --
GLOC_ARMS = ("H", "G_loc-only", "N_J+G", "S+G", "H-pose+G", "H-G1", "H-G2")
GPU_ARMS = (*GLOC_ARMS, "W+F", "H-approach")
CPU_ARMS = ("F", "rand", "hold")
ARMS = (*GPU_ARMS, *CPU_ARMS)
D_ARMS = ("H", "G_loc-only", "N_J+G", "S+G", "W+F", "F", "H-pose+G", "rand", "hold")
D_GPU_ARMS = ("H", "G_loc-only", "N_J+G", "S+G", "W+F", "H-pose+G")
REPLICATION_ARMS = ("H-G1", "H-G2")
CONTROLS = ("G_loc-only", "N_J+G", "rand", "hold")
APPROACH_ARM = "H-approach"  # A: phase 1 alone, until the hand-over fires
# arm -> (phase 1, hand-over, phase 2)
ARM_PLAN = {
    "H": ("cem:J", "rule", "gloc:primary"),
    "H-G1": ("cem:J", "rule", "gloc:replication0"),
    "H-G2": ("cem:J", "rule", "gloc:replication1"),
    "G_loc-only": (None, None, "gloc:primary"),
    "N_J+G": ("cem:N_J", "rule", "gloc:primary"),
    "S+G": ("scripted", "fixed40", "gloc:primary"),
    "W+F": ("cem:J", "rule", "follower"),
    "H-pose+G": ("pose", "rule", "gloc:primary"),
    "H-approach": ("cem:J", "rule", None),
}
EVIDENCE_FRAMES_EPISODES = 8
MCNEMAR_P = cl.MCNEMAR_P
S_RESETS = 128
D_RESETS = 64
A_RESETS = 32
NI_DELTA = 16 / 128


def check_arm(arm: str) -> str:
    if arm not in ARMS:
        raise ContractError(f"unknown arm {arm!r}")
    return arm


# ----- A's configuration rule (§7.5) --
APPROACH_MIN_A = 16


def choose_approach_config(a_counts: dict, closest_median: dict) -> dict:
    """§7.5: the configuration with the most hand-overs by tolerance on A; ties to the lower
    median closest arm RMS to subgoal 1, then TASK-101 §7.3's order (mean over last, then H = 8).
    The row is S0-APPROACH if the chosen configuration's a_A is below 16/32."""
    if sorted(a_counts) != sorted(CONFIGS) or sorted(closest_median) != sorted(CONFIGS):
        raise ContractError(f"A needs every configuration {CONFIGS}")

    def key(c):
        h, cost = cl.parse_config(c)
        return (-int(a_counts[c]), float(closest_median[c]), 0 if cost == "mean" else 1, h)

    chosen = sorted(CONFIGS, key=key)[0]
    return {
        "chosen": chosen,
        "a_A": int(a_counts[chosen]),
        "approach_fails": int(a_counts[chosen]) < APPROACH_MIN_A,
        "counts": {c: int(a_counts[c]) for c in CONFIGS},
        "closest_median": {c: float(closest_median[c]) for c in CONFIGS},
    }


def handovers_by_tolerance(summaries) -> int:
    """a_H / a_A: episodes whose hand-over fired by tolerance (within T₁ by construction)."""
    return sum(1 for s in summaries if (s.get("handover") or {}).get("how") == "tolerance")


# ----- statistics (§11) --
exact_interval = cl.exact_interval
paired = cl.paired  # exact one-sided McNemar at p < 0.01


def bootstrap_difference(first, second, *, resamples: int = 10_000):
    """§11: the reset bootstrap of the summed paired difference (salt 10113)."""
    return cl.bootstrap_difference(first, second, resamples=resamples, salt=SALTS["bootstrap"])


def mcnemar_power(p_h, p_c, coupling, *, n=S_RESETS, trials=20_000, index=0, table=None):
    """Simulated pass rate of the exact one-sided McNemar test at p < 0.01 (salt 10114)."""
    q_h, q_c = cl.discordant_probabilities(float(p_h), float(p_c), coupling)
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["power"], int(index)]))
    draws = rng.multinomial(int(n), [q_h, q_c, max(1.0 - q_h - q_c, 0.0)], size=int(trials))
    table = cl.mcnemar_pass_table(int(n)) if table is None else table
    return float(table[draws[:, 0], draws[:, 1]].mean())


bar_power = cl.bar_power


# ----- the bar (§7.6) --
CLAIM_FLOOR = 0.50
CAP_FACTOR = 0.9


def bar_rule(h_d: int, r_sg: float, r_wf: float, *, n_d: int = D_RESETS, n_s: int = S_RESETS):
    """c = 0.9 × min(r_SG, r_WF); b = min(c, max(0.50, LCB(H(D)))); the bar is ceil(b × 128).
    ``admissible`` is False when c < 0.50 (S0-CEILING)."""
    lcb = exact_interval(int(h_d), int(n_d))[0]
    c = CAP_FACTOR * min(float(r_sg), float(r_wf))
    b = min(c, max(CLAIM_FLOOR, lcb))
    bar = int(math.ceil(round(b * n_s, 9)))
    return {
        "h_d": int(h_d),
        "n_d": int(n_d),
        "r_sg": float(r_sg),
        "r_wf": float(r_wf),
        "lcb": lcb,
        "c": c,
        "b": b,
        "bar": bar,
        "admissible": c >= CLAIM_FLOOR,
        "cap_binds": c < max(CLAIM_FLOOR, lcb),
    }


def bar_table(h_values=(34, 40, 48, 56, 60, 64), r_sg: float = 1.0, r_wf: float = 1.0):
    rows = []
    for h in h_values:
        rule = bar_rule(h, r_sg, r_wf)
        p = h / D_RESETS
        rows.append(
            rule
            | {
                "power_at_hd": bar_power(p, rule["bar"]),
                "power_at_hd_minus_0.05": bar_power(max(p - 0.05, 0.0), rule["bar"]),
            }
        )
    return rows


# ----- Stage 0's rows (§7.7) --
STAGE0_ROWS = (
    "S0-VOID",
    "S0-GLOC-WEAK",
    "S0-APPROACH",
    "S0-CEILING",
    "S0-LOW",
    "S0-OVER",
    "S0-DONE",
)
SG_MIN_D = 32
AH_MIN_D = 32
WF_MIN_D = 32
LOW_POWER = 0.80


def stage0_row(
    *,
    void_reasons=(),
    gloc_eligible: bool = True,
    k0_p: int | None = None,
    a_a: int | None = None,
    sg_d: int | None = None,
    a_h: int | None = None,
    wf_d: int | None = None,
    h_d: int | None = None,
    s_fits_without_replication: bool | None = None,
) -> dict:
    """§7.7, first match. Arguments left ``None`` were not reached (a stage that a row stopped
    does not run); the first row that fires on what was read is the row."""
    out = {"void_reasons": list(void_reasons)}
    if void_reasons:
        return out | {"row": "S0-VOID"}
    if not gloc_eligible or (k0_p is not None and k0_p < GLOC_WEAK_K0):
        return out | {"row": "S0-GLOC-WEAK", "stopped_before": "A" if k0_p is not None else "P"}
    if a_a is not None and a_a < APPROACH_MIN_A:
        return out | {"row": "S0-APPROACH", "stopped_before": "D"}
    if None in (sg_d, a_h, wf_d, h_d):
        return out | {"row": None, "reason": "D has not run"}
    if sg_d < SG_MIN_D:
        return out | {"row": "S0-GLOC-WEAK"}
    if a_h < AH_MIN_D or wf_d < WF_MIN_D:
        return out | {"row": "S0-APPROACH"}
    rule = bar_rule(h_d, sg_d / D_RESETS, wf_d / D_RESETS)
    out |= {"bar": rule}
    if not rule["admissible"]:
        return out | {"row": "S0-CEILING"}
    power = bar_power(h_d / D_RESETS, rule["bar"])
    out |= {"power_at_hd": power}
    if power < LOW_POWER:
        return out | {"row": "S0-LOW"}
    if s_fits_without_replication is None:
        return out | {"row": None, "reason": "no projection"}
    if not s_fits_without_replication:
        return out | {"row": "S0-OVER"}
    return out | {"row": "S0-DONE"}


# ----- the gated rows (§9) --
GATED_ROWS = ("VOID", "HY-PASS", "HY-UNSPECIFIC", "HY-LOW", "HY-FAIL")


def gated_row(outcomes: dict | None, bar: int, *, void: bool = False) -> dict:
    """§9, first match: VOID; HY-PASS (k_H >= bar and H beats G_loc-only, N_J+G, rand and hold);
    HY-UNSPECIFIC (k_H >= bar, a test fails); HY-LOW (k_H < bar, H beats rand and hold); HY-FAIL
    otherwise."""
    if void:
        return {"row": "VOID"}
    h = np.asarray(outcomes["H"], bool)
    tests = {a: paired(h, outcomes[a]) for a in CONTROLS}
    k_h = int(h.sum())
    if k_h >= bar:
        row = "HY-PASS" if all(t["beats"] for t in tests.values()) else "HY-UNSPECIFIC"
    elif tests["rand"]["beats"] and tests["hold"]["beats"]:
        row = "HY-LOW"
    else:
        row = "HY-FAIL"
    failed = [a for a, t in tests.items() if not t["beats"]]
    return {"row": row, "k_h": k_h, "bar": int(bar), "tests": tests, "failed_tests": failed}


# ----- budget (§13) --
HARD_CAP_GPU_H = 12.0
STAGE0_CAP_GPU_H = 5.0
OVERRUN_FACTOR = 1.5
S_GPU_ARMS = ("H", "G_loc-only", "N_J+G", "S+G", "W+F", "H-pose+G", "H-G1", "H-G2")


def project_stage_s(
    *,
    lock_seconds_per_episode: dict,
    goals_gpu_h: float,
    cpu_evidence_gpu_h: float,
    used_gpu_h: float,
    resets: int = S_RESETS,
) -> dict:
    """§13: S's GPU hours = goals + every GPU arm's episodes at D's measured lock seconds per
    episode (pool throughput) + the CPU arms' evidence job; the replication arms are dropped
    first; S0-OVER if S does not fit the cap's remainder even without them."""
    remainder = HARD_CAP_GPU_H - float(used_gpu_h)

    def cost(arms):
        return (
            sum(lock_seconds_per_episode[a] * resets for a in arms) / 3600.0
            + goals_gpu_h
            + cpu_evidence_gpu_h
        )

    full = cost(S_GPU_ARMS)
    without = cost([a for a in S_GPU_ARMS if a not in REPLICATION_ARMS])
    return {
        "remainder_gpu_h": remainder,
        "with_replication_gpu_h": full,
        "without_replication_gpu_h": without,
        "fits_with_replication": full <= remainder,
        "fits_without_replication": without <= remainder,
        "drop_replication": full > remainder,
    }


# ----- the check ledger (§7.3; TASK-101 §18.5's ordering rule) --
LEDGER_FILE = "ledger.jsonl"
CHECK_IDS = {
    "latent": "7.3.1",
    "renc": "7.3.2",
    "projector": "7.3.3",
    "gloc": "7.3.4",
    "determinism": "7.3.5",
    "timing": "7.3.6",
}
ONCE_CHECKS = ("latent", "projector", "gloc", "determinism", "timing")
PER_COHORT_CHECK = "renc"
ledger_append = cl.ledger_append
ledger_read = cl.ledger_read
ordered_after_checks = cl.ordered_after_checks


def ledger_entry(**kwargs) -> dict:
    entry = cl.ledger_entry(**kwargs)
    entry["id"] = CHECK_IDS.get(kwargs["check"])
    entry["task"] = TASK
    return entry


def ledger_verdict(
    entries: list[dict], *, revision: str, goals_sha256: str, block_on_failures: bool = True
) -> dict:
    """Episodes of a ledger cohort may run only if the ledger holds a passing report at
    ``revision``, from a clean tracked tree, for every check of ``ONCE_CHECKS`` and a passing
    7.3.2 report naming the goal store the episodes read; any failing report blocks every
    episodes job (``block_on_failures``). TASK-101's verdict, with this task's checks."""
    reasons = []
    failing = [e for e in entries if e.get("passes") is False]
    if failing and block_on_failures:
        reasons.append(
            "a failing check report blocks every episodes job until a ruling: "
            + ", ".join(f"{e['check']}@{str(e['revision'])[:8]}" for e in failing)
        )

    def passing(e):
        return e.get("passes") is True and e["revision"] == revision and not e["tracked_tree_dirty"]

    required = []
    for name in ONCE_CHECKS:
        ok = [e for e in entries if e["check"] == name and passing(e)]
        if not ok:
            reasons.append(f"no passing {CHECK_IDS[name]} ({name}) report at {revision[:8]}")
        else:
            required.append(max(e["finished_utc"] for e in ok))
    renc = [
        e
        for e in entries
        if e["check"] == PER_COHORT_CHECK and passing(e) and e.get("goals_sha256") == goals_sha256
    ]
    if not renc:
        reasons.append(
            f"no passing 7.3.2 (renc) report at {revision[:8]} for goal store {goals_sha256[:12]}"
        )
    else:
        required.append(max(e["finished_utc"] for e in renc))
    return {
        "ok": not reasons,
        "reasons": reasons,
        "latest_required_utc": max(required) if required else None,
    }


def require_ledger(path, *, revision: str, goals_sha256: str) -> dict:
    """Raise a ``GuardError`` (before any controller, simulator or seed exists) unless the ledger
    admits these episodes."""
    verdict = ledger_verdict(ledger_read(path), revision=revision, goals_sha256=goals_sha256)
    if not verdict["ok"]:
        raise GuardError("G-ledger: " + "; ".join(verdict["reasons"]))
    return verdict


# ----- the stage parameters (τ_h from P, the configuration from A) --
def write_stage_parameter(path, payload: dict) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=1, default=cl._json_default))


def read_stage_parameter(path) -> dict:
    return json.loads(Path(path).read_text())
