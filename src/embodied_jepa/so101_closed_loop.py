"""TASK-101 (SO-101 S5): a first closed loop on ``so101_cube_lift_v1`` (amendment A1: v0 with the
inert target marker drawn inside the play corpus's marker range) in simulation -- cohorts, seeds,
salts, the goal store, the subgoal switching rule, the derangement, the projector, the non-learned
arms, statistics, the bar rule, the rows, Stage 0′'s check ledger (the ordering rule, §18.5) and
the S0-NULL diagnostic Delta's measures and rows (§18.6)
(``docs/experiments/so101_closed_loop_lift_v1.md``).

NumPy only at import. The torch runtime (the live featuriser, the planner-model adapters, the
episode loop and the worker pool) is ``so101_closed_loop_runtime``. Nothing here is registered or
imported by ``embodied_jepa``. Simulator truth enters only the goal demonstration, the scorer, the
privileged follower F and the runner's reported logs; no controller here reads it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embodied_jepa.contracts import ContractError

PROTOCOL = "docs/experiments/so101_closed_loop_lift_v1.md"
TASK = "TASK-101"
MANIFEST = "benchmarks/manifests/so101-closed-loop-lift-v1.json"


class GuardError(RuntimeError):
    """A §12 guard failed (a VOID)."""


# ----- cohorts and seeds (§7, §8, §12) --
@dataclass(frozen=True)
class Cohort:
    name: str
    first: int
    size: int
    reserve: tuple[int, int] | None  # inclusive range

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.first, self.first + self.size))

    @property
    def reserve_seeds(self) -> tuple[int, ...]:
        if self.reserve is None:
            return ()
        return tuple(range(self.reserve[0], self.reserve[1] + 1))

    def contains(self, seed: int) -> bool:
        return seed in self.seeds or seed in self.reserve_seeds


COHORTS = {
    "K": Cohort("K", 101010, 32, (101042, 101059)),
    "Kp": Cohort("Kp", 101060, 32, (101092, 101099)),  # K', used only under §7.6 row 1
    "D": Cohort("D", 101100, 64, (101164, 101199)),
    "S": Cohort("S", 101200, 128, (101400, 101463)),
    "Dp": Cohort("Dp", 101600, 64, (101664, 101699)),  # D', used only under §7.6 row 1
    "X": Cohort("X", 101700, 16, (101716, 101731)),  # §18.6's diagnostic cohort (Delta only)
    "debug": Cohort("debug", 101900, 100, None),  # plumbing only; never counted
}
STAGE0_COHORTS = ("K", "Kp", "D", "Dp", "X", "debug")
GATED_COHORTS = ("S",)
# A1 (§18.5): K's and D's counts were read under Stage 0's void condition, in the v0 scene; they
# are spent and are never run again (no goals, no episodes).
SPENT_COHORTS = ("K", "D")
# §18.5's ordering rule: these cohorts' episodes need the check ledger.
LEDGER_COHORTS = ("Kp", "Dp")
# The task every Stage 0' and gated job runs (A1, §18.2); a module constant, not a registry entry.
LIFT_TASK_NAME = "so101_cube_lift_v1"


def cohort(name: str) -> Cohort:
    if name not in COHORTS:
        raise ContractError(f"unknown cohort {name!r}")
    return COHORTS[name]


def check_seed(name: str, seed: int) -> int:
    """§12: a seed outside its cohort (main or reserve range) is refused."""
    if type(seed) is not int and not isinstance(seed, np.integer):
        raise GuardError(f"G-seed: {seed!r} is not an integer seed")
    seed = int(seed)
    if not cohort(name).contains(seed):
        raise GuardError(f"G-seed: {seed} is outside cohort {name}")
    return seed


SALTS = {
    "cem": 10101,
    "rand": 10102,
    "derangement": 10103,
    "bootstrap": 10104,
    "power": 10105,
    "permutation": 10106,
    "marker": 10107,  # A1: the v1 marker stream (so101_tasks_lift_v1.MARKER_SALT)
    "delta": 10108,  # A1: Delta's random candidates and planner seeds (§18.6)
}


def planner_seed(reset_seed: int) -> int:
    """§5.1: one planner per episode, seeded 10101 * 10^6 + reset seed (every CEM arm alike)."""
    return SALTS["cem"] * 10**6 + int(reset_seed)


def rand_rng(reset_seed: int) -> np.random.Generator:
    """§6's rand arm: fresh uniform commands from ``default_rng([10102, reset seed])``."""
    return np.random.default_rng([SALTS["rand"], int(reset_seed)])


def derangement(n: int, salt: int = SALTS["derangement"]) -> np.ndarray:
    """§6's W-shuf map: slot i takes the goal of slot ``out[i]``; never itself.

    A salt-fixed permutation p is read as one cycle, p[j] -> p[j + 1], so no slot maps to itself
    for n >= 2."""
    if n < 2:
        raise ContractError("a derangement needs at least two slots")
    p = np.random.default_rng(salt).permutation(n)
    out = np.empty(n, np.int64)
    out[p] = np.roll(p, -1)
    return out


# ----- the model (§3) --
TASK100_CHECKPOINTS = "checkpoints/task100-run"
TASK100_FEATURES = "outputs/task100-features-pretrained"
TASK100_READOUTS = "outputs/task100-run/readouts"
TASK100_VAL_REPORT = "outputs/task100-run/val_report.json"
VAL_REPORT_SHA256 = "b3c3eddef04da3e29a1fad952ce619af6f65714574a0e749f8f8f4c6f633aa31"
FIT_REPORT_SHA256 = "64f1b9f7d145ac95ccf933e2285de7f744ba1650520ec5ad04a3fcd0b1d8331c"
MOMENTS_SHA256 = "b303cd09ecc105473693ad317b90e209c2d1fd9d4e78ae344cdf6e52ef613b89"
IMPLEMENTATION_SHA256 = "78ba9dbe00eb1efba5ec6ecb2f99821e6817aa8e5aa7e5b7fdaaa115ea426220"
ENCODER_DIGEST = "3a697b872089933d345fd71087af86e85ff31886cf1e2654763dc1884ef2af27"
RENC_FILE = "renc_pretrained.npz"
CHECKPOINT_SHA256 = {
    "J-100100": "41fa52ad3211880ff79d6c7153f00b3a183532b1010323f7777e65bd2c18cd16",
    "J-100101": "0e04d495ef39bc1b4a44ace607a6c749c53a4df010308f747230927526180061",
    "J-100102": "0ae2b6cc099b2ce7d6158a0cb3bafceecccd02273d252759bf74e58452bff3c7",
    "N_J-100101": "a8fe58a4bd7cf7b7e39efe27f003756832f40d6c8157e6d21f696788cef90c3c",
}
KEPT_CRITERION = {
    "J-100100": 0.5314159055925411,
    "J-100101": 0.5302079930757722,
    "J-100102": 0.5338290393374114,
}
PRIMARY = "J-100101"
TWIN = "N_J-100101"


def primary_seed(criteria: dict = KEPT_CRITERION) -> str:
    """§3's rule: the J seed whose kept checkpoint has the lowest val criterion."""
    return min(criteria, key=lambda k: (criteria[k], k))


LATENT_DIM = 12_288
TOKENS, WIDTH = 32, 384
CAMERAS = ("wrist_rgb", "overview_rgb")
IMAGE_SIZE = 224

# ----- goals (§4) --
SUBGOAL_COMMANDS = (40, 70, 90, 120)
SUBGOALS = len(SUBGOAL_COMMANDS)
JOINTS = 6
ARM_JOINTS = 5
JAW = 5

# ----- the controller (§5) --
CEM = {"samples": 256, "iterations": 6, "elites": 32, "minimum_std": 0.05, "action_penalty": 0.0}
HORIZONS = (8, 16)
COSTS = ("last", "mean")
CONFIGS = tuple(f"H{h}-{c}" for h in HORIZONS for c in COSTS)
SWITCH_ARM_RMS_RAD = 0.04
SWITCH_JAW_RAD = 0.15
SWITCH_CONSECUTIVE = 3
SWITCH_TIMEOUTS = (60, 45, 30)  # subgoals 1-3; subgoal 4 is kept to the end
POSE_JAW_OPEN_ABOVE_RAD = 0.785  # H-pose: jaw +1 above mid-range, else -1
MAX_COMMANDS = 200


def parse_config(name: str) -> tuple[int, str]:
    if name not in CONFIGS:
        raise ContractError(f"unknown configuration {name!r}; expected one of {CONFIGS}")
    h, c = name[1:].split("-")
    return int(h), c


# ----- arms (§6) --
CEM_ARMS = ("W", "W-100100", "W-100102", "N", "W-shuf", "W-mean")
CPU_ARMS = ("rand", "hold", "F", "H-pose")
ARMS = (*CEM_ARMS, *CPU_ARMS)
REPLICATION_ARMS = ("W-100100", "W-100102")
D_ARMS = ("W", "N", "W-shuf", "W-mean", "rand", "hold", "F", "H-pose")
DIAGNOSTIC = "W-single"  # §7.4: W with the single final subgoal on D's first 16 resets
DIAGNOSTIC_RESETS = 16
ARM_MODEL = {
    "W": PRIMARY,
    "W-100100": "J-100100",
    "W-100102": "J-100102",
    "N": TWIN,
    "W-shuf": PRIMARY,
    "W-mean": PRIMARY,
    DIAGNOSTIC: PRIMARY,
}
EVIDENCE_FRAMES_EPISODES = 8  # §13: frames kept for the first 8 episodes of each arm and cohort
CONTROLS = ("N", "W-shuf", "W-mean", "rand", "hold")
MCNEMAR_P = 0.01
NI_DELTA = 16 / 128
S_RESETS = 128


def check_arm(arm: str) -> str:
    if arm not in ARMS and arm != DIAGNOSTIC:
        raise ContractError(f"unknown arm {arm!r}")
    return arm


# ----- subgoal switching (§5.3) --
class SubgoalSwitcher:
    """§5.3 for every arm that uses subgoals: on subgoal k (0-based here), move to k + 1 when for
    ``SWITCH_CONSECUTIVE`` consecutive commands the measured arm joints are within 0.04 rad RMS of
    subgoal k's and the jaw within 0.15 rad, or after ``SWITCH_TIMEOUTS[k]`` commands on k; the
    last subgoal is kept to the end.

    ``update(q)`` is called once per command with the measured joints, before planning, and
    returns the subgoal to plan toward on this command. At most one switch per command; a switch
    restarts the count of near commands and of commands on the new subgoal. It reads
    proprioception and the goal pose only."""

    def __init__(self, goal_joints, *, single_goal: bool = False):
        g = np.asarray(goal_joints, np.float64)
        if g.shape != (SUBGOALS, JOINTS) or not np.isfinite(g).all():
            raise ContractError("goal joints must be finite [4, 6]")
        self.goal = g
        self.k = SUBGOALS - 1 if single_goal else 0
        self.single = bool(single_goal)
        self.near = 0
        self.on_k = 0
        self.events: list[dict] = []
        self.command = 0

    def is_near(self, q, k: int) -> bool:
        q = np.asarray(q, np.float64)
        d = q[:ARM_JOINTS] - self.goal[k, :ARM_JOINTS]
        rms = float(np.sqrt(np.mean(d * d)))
        return rms <= SWITCH_ARM_RMS_RAD and abs(q[JAW] - self.goal[k, JAW]) <= SWITCH_JAW_RAD

    def update(self, q) -> int:
        if self.k < SUBGOALS - 1 and not self.single:
            self.near = self.near + 1 if self.is_near(q, self.k) else 0
            reached = self.near >= SWITCH_CONSECUTIVE
            timed_out = self.on_k >= SWITCH_TIMEOUTS[self.k]
            if reached or timed_out:
                self.events.append(
                    {
                        "command": self.command,
                        "from": self.k + 1,
                        "to": self.k + 2,
                        "how": "reached" if reached else "timeout",
                        "commands_on_subgoal": self.on_k,
                    }
                )
                self.k += 1
                self.near = 0
                self.on_k = 0
        self.on_k += 1
        self.command += 1
        return self.k


# ----- the projector (§5.2): SO101Embodiment.preview, chained, vectorised --
@dataclass(frozen=True)
class PreviewParams:
    """The embodiment's own clip parameters (limits, scales, gripper map and rate); no sensor."""

    limits: np.ndarray  # [6, 2] float64 (the transport's exact limits)
    scales: np.ndarray  # [5] float32
    gripper_closed: float
    gripper_open: float
    gripper_step: float

    @classmethod
    def of(cls, robot) -> PreviewParams:
        return cls(
            np.asarray(robot.limits, np.float64).copy(),
            np.asarray(robot.scales, np.float32).copy(),
            float(robot.gripper_closed),
            float(robot.gripper_open),
            float(robot.manifest["gripper_max_delta_per_step_rad"]),
        )


def project_sequences(requested, targets, params: PreviewParams):
    """``SO101Embodiment.preview`` chained over the horizon for every candidate at once.

    ``requested`` float32 [..., T, 6]; ``targets`` [6] the current joint-position targets.
    Returns (applied float32 [..., T, 6], final targets float64 [..., 6]). Step t starts from the
    targets step t - 1 produced. The arithmetic is ``preview``'s, operation for operation, so the
    result is bit-identical to calling ``preview`` step by step (a test checks it)."""
    a = np.asarray(requested)
    if a.dtype != np.float32 or a.shape[-1] != JOINTS or a.ndim < 2:
        raise ContractError("requested candidates must be float32 [..., T, 6]")
    lead = a.shape[:-2]
    applied = a.copy()
    old = np.broadcast_to(np.asarray(targets, np.float64), (*lead, JOINTS)).copy()
    lo, hi = params.limits[:, 0], params.limits[:, 1]
    scales = params.scales.astype(np.float64)
    width = params.gripper_open - params.gripper_closed
    for t in range(a.shape[-2]):
        step = a[..., t, :]
        desired = old[..., :ARM_JOINTS] + step[..., :ARM_JOINTS].astype(np.float64) * scales
        new_arm = np.clip(desired, lo[:ARM_JOINTS], hi[:ARM_JOINTS])
        clipped = new_arm != desired
        arm_applied = np.clip((new_arm - old[..., :ARM_JOINTS]) / scales, -1.0, 1.0).astype(
            np.float32
        )
        applied[..., t, :ARM_JOINTS] = np.where(clipped, arm_applied, step[..., :ARM_JOINTS])
        requested_g = (
            params.gripper_closed + (step[..., JAW].astype(np.float64) + 1.0) / 2.0 * width
        )
        g = np.clip(
            requested_g, old[..., JAW] - params.gripper_step, old[..., JAW] + params.gripper_step
        )
        g = np.clip(g, lo[JAW], hi[JAW])
        limited = g != requested_g
        g_applied = np.clip(2.0 * (g - params.gripper_closed) / width - 1.0, -1.0, 1.0).astype(
            np.float32
        )
        applied[..., t, JAW] = np.where(limited, g_applied, step[..., JAW])
        old = np.concatenate((new_arm, g[..., None]), axis=-1)
    return applied, old


# ----- the non-learned arms (§6) --
def pose_action(targets, goal_q, scales) -> np.ndarray:
    """H-pose: arm targets toward subgoal k's measured joints by the follower's law
    ``clip((q_k - target) / scale, +-1)``; jaw +1 if subgoal k's jaw is above 0.785 rad, else -1."""
    t = np.asarray(targets, np.float64)
    g = np.asarray(goal_q, np.float64)
    delta = (g[:ARM_JOINTS] - t[:ARM_JOINTS]) / np.asarray(scales, np.float32).astype(float)
    jaw = 1.0 if g[JAW] > POSE_JAW_OPEN_ABOVE_RAD else -1.0
    return np.concatenate((np.clip(delta, -1, 1), [jaw])).astype(np.float32)


def rand_action(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, JOINTS).astype(np.float32)


# ----- the goal store (§4) --
def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


GOAL_ARRAYS = (
    "seeds",
    "slots",
    "frames",
    "latents_raw",
    "joints",
    "cube_cm",
    "site_m",
    "site_xmat",
    "demo_actions",
    "demo_lengths",
)


def keep_cohort(name: str, demonstrate) -> dict:
    """§4's replacement: run ``demonstrate(seed) -> (ok: bool, reason: str, record)`` over the
    cohort's main seeds in order; a rejected seed's slot takes the next unused reserve seed.
    Returns kept seeds by slot and the rejections, with reasons."""
    c = cohort(name)
    reserve = list(c.reserve_seeds)
    kept, rejected, records = [], [], {}
    for seed in c.seeds:
        candidate = seed
        while True:
            ok, reason, record = demonstrate(candidate)
            records[candidate] = record
            if ok:
                kept.append(candidate)
                break
            rejected.append({"seed": candidate, "slot": len(kept), "reason": reason})
            if not reserve:
                raise GuardError(f"G-reserve: cohort {name}'s reserve is exhausted")
            candidate = reserve.pop(0)
    return {
        "kept": kept,
        "rejected": rejected,
        "records": records,
        "drawn": len(kept) + len(rejected),
    }


def demo_verdict(record: dict) -> tuple[bool, str]:
    """§4: reject a demonstration that fails, stops, or terminates or stops before command 120
    has executed (so subgoal 4 would not exist)."""
    if record.get("stopped"):
        return False, "stopped"
    if record.get("commands", 0) < SUBGOAL_COMMANDS[-1] or len(record.get("captured", [])) < 4:
        return False, f"ended before command {SUBGOAL_COMMANDS[-1]}"
    if not record.get("success"):
        return False, f"demonstration failed ({record.get('termination_reason')})"
    return True, ""


def save_goal_store(folder, arrays: dict, report: dict) -> str:
    folder = Path(folder)
    path = folder / "goals.npz"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    missing = [k for k in GOAL_ARRAYS if k not in arrays]
    if missing:
        raise ContractError(f"goal store lacks {missing}")
    np.savez(path, **{k: arrays[k] for k in GOAL_ARRAYS})
    sha = sha256_file(path)
    report = dict(report) | {"goals_sha256": sha}
    out = folder / "goals_report.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.write_text(json.dumps(report, indent=1, default=_json_default))
    return sha


def load_goal_store(folder, expected_sha256: str | None = None) -> dict:
    """§12: every arm's runner reads the goal store by hash; a mismatch is a VOID."""
    folder = Path(folder)
    path = folder / "goals.npz"
    report = json.loads((folder / "goals_report.json").read_text())
    got = sha256_file(path)
    want = expected_sha256 or report["goals_sha256"]
    if got != want or got != report["goals_sha256"]:
        raise GuardError("G-goals: the goal store differs from its recorded sha256")
    with np.load(path) as d:
        arrays = {k: d[k] for k in d.files}
    return {"arrays": arrays, "report": report, "sha256": got}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")


# ----- statistics (§11) --
def _binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    return float(sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, min(k, n) + 1)))


def exact_interval(k: int, n: int, level: float = 0.95) -> list[float]:
    """Exact (Clopper-Pearson) two-sided interval, by bisection (TASK-081's form)."""
    k, n, a = int(k), int(n), (1.0 - level) / 2.0
    if not 0 <= k <= n or n < 1:
        raise ContractError("need 0 <= k <= n, n >= 1")

    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2.0

    lower = 0.0 if k == 0 else solve(lambda p: 1.0 - _binom_cdf(k - 1, n, p) < a)
    upper = 1.0 if k == n else solve(lambda p: _binom_cdf(k, n, p) > a)
    return [lower, upper]


def mcnemar_one_sided(only_first: int, only_second: int) -> float:
    """Exact one-sided McNemar p: P(X >= only_first), X ~ Bin(only_first + only_second, 1/2)."""
    b, c = int(only_first), int(only_second)
    n = b + c
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, i) for i in range(b, n + 1)) / 2**n)


def paired(first, second) -> dict:
    a, b = np.asarray(first, bool), np.asarray(second, bool)
    if a.shape != b.shape:
        raise ContractError("paired arms need the same resets")
    only_first, only_second = int((a & ~b).sum()), int((~a & b).sum())
    p = mcnemar_one_sided(only_first, only_second)
    return {
        "difference": int(a.sum()) - int(b.sum()),
        "only_first": only_first,
        "only_second": only_second,
        "p_one_sided": p,
        "beats": bool(p < MCNEMAR_P),
    }


def bootstrap_difference(first, second, *, resamples: int = 10_000, salt: int = SALTS["bootstrap"]):
    """§11: the reset bootstrap of the summed paired difference (2.5th and 97.5th percentiles)."""
    a, b = np.asarray(first, np.int64), np.asarray(second, np.int64)
    if a.shape != b.shape:
        raise ContractError("paired arms need the same resets")
    rng = np.random.default_rng(salt)
    idx = rng.integers(0, len(a), (resamples, len(a)))
    sums = (a - b)[idx].sum(1)
    return [float(np.percentile(sums, 2.5)), float(np.percentile(sums, 97.5))]


def non_inferiority(first, second, n: int = S_RESETS) -> dict:
    lo, hi = bootstrap_difference(first, second)
    return {
        "difference": int(np.sum(first)) - int(np.sum(second)),
        "ci95": [lo, hi],
        "delta": -NI_DELTA * n,
        "not_detectably_inferior": bool(lo > -NI_DELTA * n),
    }


# ----- the bar (§7.5) and power (§11) --
def bar_rule(w_d: int, n_d: int = 64, r_f: float = 1.0, n_s: int = S_RESETS) -> dict:
    """b = min(0.9 * r_F(D), max(0.50, LCB(W(D)))); the gated bar is ceil(b * 128) of 128."""
    lcb = exact_interval(w_d, n_d)[0]
    b = min(0.9 * float(r_f), max(0.50, lcb))
    # guard against float noise at exact multiples (0.5 * 128 = 64)
    bar = int(math.ceil(round(b * n_s, 9)))
    return {"w_d": int(w_d), "n_d": int(n_d), "r_f": float(r_f), "lcb": lcb, "b": b, "bar": bar}


def bar_power(rate: float, bar: int, n: int = S_RESETS) -> float:
    """P(X >= bar), X ~ Bin(n, rate), exact."""
    return float(sum(math.comb(n, k) * rate**k * (1 - rate) ** (n - k) for k in range(bar, n + 1)))


def discordant_probabilities(p_w: float, p_c: float, coupling: str) -> tuple[float, float]:
    """(P(W only), P(control only)) per reset: ``overlap`` (nested: every control success is a W
    success when p_w >= p_c), ``independent`` or ``half`` (halfway) -- TASK-081's form."""
    nested = (max(p_w - p_c, 0.0), max(p_c - p_w, 0.0))
    independent = (p_w * (1.0 - p_c), p_c * (1.0 - p_w))
    if coupling == "overlap":
        return nested
    if coupling == "independent":
        return independent
    if coupling == "half":
        return tuple(0.5 * (x + y) for x, y in zip(nested, independent, strict=True))
    raise ContractError(f"unknown coupling {coupling!r}")


COUPLINGS = ("overlap", "half", "independent")
POWER_TRIALS = 20_000


def mcnemar_pass_table(n: int) -> np.ndarray:
    out = np.zeros((n + 1, n + 1), bool)
    for b in range(n + 1):
        for c in range(n + 1 - b):
            out[b, c] = mcnemar_one_sided(b, c) < MCNEMAR_P
    return out


def mcnemar_power(p_w, p_c, coupling, *, n=S_RESETS, trials=POWER_TRIALS, index=0, table=None):
    """Simulated pass rate of the exact one-sided McNemar test at p < 0.01 (salt 10105)."""
    q_w, q_c = discordant_probabilities(float(p_w), float(p_c), coupling)
    rng = np.random.default_rng(np.random.SeedSequence([SALTS["power"], int(index)]))
    draws = rng.multinomial(int(n), [q_w, q_c, max(1.0 - q_w - q_c, 0.0)], size=int(trials))
    table = mcnemar_pass_table(int(n)) if table is None else table
    return float(table[draws[:, 0], draws[:, 1]].mean())


def bar_table(w_values=(34, 36, 40, 48, 56, 60, 62, 64), n_d: int = 64) -> list[dict]:
    """§7.5's table, recomputed (r_F = 1)."""
    rows = []
    for w in w_values:
        rule = bar_rule(w, n_d)
        p = w / n_d
        rows.append(
            rule
            | {
                "power_at_wd": bar_power(p, rule["bar"]),
                "power_at_wd_minus_0.05": bar_power(max(p - 0.05, 0.0), rule["bar"]),
            }
        )
    return rows


# ----- K's rule (§7.3) --
def choose_config(counts: dict, eligible: dict | None = None) -> dict:
    """The most successes; ties to mean over last, then H = 8 over H = 16. A configuration whose
    projected Stage S cost does not fit the cap is not eligible. All 0/32 -> NULL (no choice)."""
    if sorted(counts) != sorted(CONFIGS):
        raise ContractError(f"K needs every configuration {CONFIGS}")
    eligible = eligible or dict.fromkeys(CONFIGS, True)
    pool = [c for c in CONFIGS if eligible.get(c)]
    null = all(int(counts[c]) == 0 for c in CONFIGS)
    if not pool:
        return {"chosen": None, "null": null, "reason": "no configuration fits the cap"}

    def key(c):
        h, cost = parse_config(c)
        return (-int(counts[c]), 0 if cost == "mean" else 1, h)

    chosen = sorted(pool, key=key)[0]
    return {"chosen": chosen, "null": null, "counts": {c: int(counts[c]) for c in CONFIGS}}


# ----- Stage 0's rows (§7.6) --
CEILING_REJECTIONS = 8
LOW_POWER = 0.80
STAGE0_ROWS = ("S0-VOID", "S0-CEILING", "S0-NULL", "S0-LOW", "S0-OVER", "S0-DONE")


def stage0_row(
    *,
    void_reasons: list,
    d_rejections: int,
    k_counts: dict,
    w_d: int,
    n_d: int = 64,
    s_fits_without_replication: bool,
) -> dict:
    """§7.6, first match."""
    rule = bar_rule(w_d, n_d)
    power = bar_power(w_d / n_d, rule["bar"])
    if void_reasons:
        row = "S0-VOID"
    elif d_rejections >= CEILING_REJECTIONS:
        row = "S0-CEILING"
    elif all(int(v) == 0 for v in k_counts.values()):
        row = "S0-NULL"
    elif power < LOW_POWER:
        row = "S0-LOW"
    elif not s_fits_without_replication:
        row = "S0-OVER"
    else:
        row = "S0-DONE"
    return {
        "row": row,
        "bar": rule,
        "power_at_wd": power,
        "void_reasons": list(void_reasons),
        "d_rejections": int(d_rejections),
    }


# ----- the gated rows (§9) --
GATED_ROWS = ("VOID", "CL-PASS", "CL-UNSPECIFIC", "CL-LOW", "CL-FAIL")


def gated_row(outcomes: dict | None, bar: int, *, void: bool = False) -> dict:
    """§9, first match: VOID; CL-PASS (k_W >= bar and W beats N, W-shuf, W-mean, rand, hold);
    CL-UNSPECIFIC (k_W >= bar, a test fails); CL-LOW (k_W < bar, W beats rand and hold);
    CL-FAIL otherwise."""
    if void:
        return {"row": "VOID"}
    w = np.asarray(outcomes["W"], bool)
    tests = {a: paired(w, outcomes[a]) for a in CONTROLS}
    k_w = int(w.sum())
    if k_w >= bar:
        row = "CL-PASS" if all(t["beats"] for t in tests.values()) else "CL-UNSPECIFIC"
    elif tests["rand"]["beats"] and tests["hold"]["beats"]:
        row = "CL-LOW"
    else:
        row = "CL-FAIL"
    return {"row": row, "k_w": k_w, "bar": int(bar), "tests": tests}


# ----- budget (§13) --
HARD_CAP_GPU_H = 20.0
STAGE0_CAP_GPU_H = 6.0
OVERRUN_FACTOR = 1.5


def project_stage_s(
    *,
    seconds_per_episode: dict,
    workers_speedup: float,
    goals_gpu_h: float,
    used_gpu_h: float,
    arms=("W", "W-100100", "W-100102", "N", "W-shuf", "W-mean"),
    resets: int = S_RESETS,
) -> dict:
    """§13: S's GPU hours = goals + every CEM arm's episodes at the measured mean seconds per
    episode (per arm, under the pool's measured throughput). Replication arms are dropped first."""
    remainder = HARD_CAP_GPU_H - used_gpu_h

    def cost(arm_list):
        total = sum(seconds_per_episode[a] * resets for a in arm_list) / workers_speedup / 3600
        return total + goals_gpu_h

    full = cost(arms)
    without = cost([a for a in arms if a not in REPLICATION_ARMS])
    return {
        "remainder_gpu_h": remainder,
        "with_replication_gpu_h": full,
        "without_replication_gpu_h": without,
        "fits_with_replication": full <= remainder,
        "fits_without_replication": without <= remainder,
        "drop_replication": full > remainder,
    }


def check_cohort_unspent(name: str) -> str:
    """A1 (§18.5): K and D are spent; neither is drawn, demonstrated or run again."""
    cohort(name)
    if name in SPENT_COHORTS:
        raise GuardError(f"G-spent: cohort {name} is spent (A1, §18.5); use K' (Kp) or D' (Dp)")
    return name


# ----- the check ledger: §18.5's ordering rule (R28.70) --
LEDGER_FILE = "ledger.jsonl"
CHECK_IDS = {
    "scene": "7.2.0",
    "latent": "7.2.1",
    "renc": "7.2.2",
    "projector": "7.2.3",
    "determinism": "7.2.4",
    "inert": "7.2.6",
}
# (ii): run once, before K′'s first episode; 7.2.2 (renc) runs on each cohort's own goal store
ONCE_CHECKS = ("scene", "latent", "projector", "determinism", "inert")
PER_COHORT_CHECK = "renc"


def ledger_entry(
    *,
    check: str,
    passes,
    revision: str,
    dirty: bool,
    finished_utc: str,
    goals_sha256=None,
    cohort_name=None,
    report_sha256=None,
) -> dict:
    return {
        "check": check,
        "id": CHECK_IDS.get(check),
        "passes": passes,
        "revision": revision,
        "tracked_tree_dirty": bool(dirty),
        "finished_utc": finished_utc,
        "goals_sha256": goals_sha256,
        "cohort": cohort_name,
        "report_sha256": report_sha256,
    }


def ledger_append(path, entry: dict) -> None:
    with Path(path).open("a") as stream:
        stream.write(json.dumps(entry, default=_json_default) + "\n")


def ledger_read(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ledger_verdict(
    entries: list[dict], *, revision: str, goals_sha256: str, block_on_failures: bool = True
) -> dict:
    """§18.5 (2): episodes may run only if the ledger holds a passing report at ``revision``, from
    a clean tracked tree, for every check of (ii) and a passing 7.2.2 report naming the goal store
    the episodes read; any failing report (of any check, at any revision) blocks every episodes
    job (``block_on_failures``; the summariser's ordering reading sets it False, since a later
    failure does not reorder what ran before it). Returns ``{"ok", "reasons",
    "latest_required_utc"}``."""
    reasons = []
    failing = [e for e in entries if e.get("passes") is False]
    if failing and block_on_failures:
        reasons.append(
            "a failing check report blocks every episodes job until a ruling: "
            + ", ".join(f"{e['check']}@{str(e['revision'])[:8]}" for e in failing)
        )

    def passing(e):
        return e.get("passes") is True and e["revision"] == revision and not e["tracked_tree_dirty"]

    required_utc = []
    for name in ONCE_CHECKS:
        ok = [e for e in entries if e["check"] == name and passing(e)]
        if not ok:
            reasons.append(f"no passing {CHECK_IDS[name]} ({name}) report at {revision[:8]}")
        else:
            required_utc.append(max(e["finished_utc"] for e in ok))
    renc = [
        e
        for e in entries
        if e["check"] == PER_COHORT_CHECK and passing(e) and e.get("goals_sha256") == goals_sha256
    ]
    if not renc:
        reasons.append(
            f"no passing 7.2.2 (renc) report at {revision[:8]} for goal store {goals_sha256[:12]}"
        )
    else:
        required_utc.append(max(e["finished_utc"] for e in renc))
    return {
        "ok": not reasons,
        "reasons": reasons,
        "latest_required_utc": max(required_utc) if required_utc else None,
    }


def require_ledger(path, *, revision: str, goals_sha256: str) -> dict:
    """Raise a ``GuardError`` (before any controller, simulator or seed exists) unless the ledger
    admits these episodes."""
    verdict = ledger_verdict(ledger_read(path), revision=revision, goals_sha256=goals_sha256)
    if not verdict["ok"]:
        raise GuardError("G-ledger: " + "; ".join(verdict["reasons"]))
    return verdict


def ordered_after_checks(episodes_started_utc: str, verdict: dict) -> bool:
    """§18.5 (3): a cohort's counts are readable only if its first episode started after the
    latest required passing check report (ISO-8601 UTC strings compare in time order)."""
    latest = verdict.get("latest_required_utc")
    return bool(verdict.get("ok")) and latest is not None and episodes_started_utc > latest


# ----- Stage 0′'s budget after K′ (§18.7, R28.73) --
def project_dprime(
    *,
    used_gpu_h: float,
    goals_and_checks_gpu_h: float,
    seconds_per_cem_episode_wall: float,
    cem_episodes: int = 4 * 64,
    diagnostic_episodes: int = DIAGNOSTIC_RESETS,
    cpu_evidence_gpu_h: float = 0.0,
    cap: float = STAGE0_CAP_GPU_H,
) -> dict:
    """D′'s projected lock time from K′'s measured wall seconds per CEM episode (pool throughput);
    the single-goal diagnostic is dropped first if the Stage 0 cap would be exceeded, then the row
    is S0-OVER."""
    base = goals_and_checks_gpu_h + cpu_evidence_gpu_h
    main = cem_episodes * seconds_per_cem_episode_wall / 3600.0
    diag = diagnostic_episodes * seconds_per_cem_episode_wall / 3600.0
    with_diag = used_gpu_h + base + main + diag
    without = used_gpu_h + base + main
    return {
        "used_gpu_h": used_gpu_h,
        "cap_gpu_h": cap,
        "cumulative_with_diagnostic_gpu_h": with_diag,
        "cumulative_without_diagnostic_gpu_h": without,
        "drop_diagnostic": with_diag > cap,
        "over": without > cap,
    }


# ----- the S0-NULL diagnostic Delta (§18.6, R28.72) --
DELTA_STATES = (0, 10, 20, 30, 40, 50, 60)
DELTA_RANDOM = 255
DELTA_MIN_KEPT = 12
DELTA_MODEL_RANK = 0.25
DELTA_PLANNER_RANK = 0.10
DELTA_PLANNER_M1 = 0.50
DELTA_ROWS = ("Δ-VOID", "Δ-MODEL", "Δ-PLANNER", "Δ-MIXED")
DELTA_CAP_GPU_H = 1.5


def delta_subgoal(t: int) -> int:
    """0-based subgoal a Delta state belongs to: subgoal 1 for t <= 30, subgoal 2 for t >= 40."""
    if t not in DELTA_STATES:
        raise ContractError(f"{t} is not a Delta state")
    return 0 if t <= 30 else 1


def delta_planner_seed(seed: int, t: int) -> int:
    """§18.6: a_cem's planner seed 10108 * 10^6 + 1 000 * (seed - 101700) + t."""
    check_seed("X", seed)
    return SALTS["delta"] * 10**6 + 1000 * (int(seed) - COHORTS["X"].first) + int(t)


def delta_random_rng(seed: int, t: int) -> np.random.Generator:
    """§18.6: the 255 random candidates of a state, from ``default_rng([10108, seed, t])``."""
    return np.random.default_rng([SALTS["delta"], int(seed), int(t)])


def delta_crosses_boundary(t: int, horizon: int) -> bool:
    """At H = 16 the windows at t = 30 and t = 60 run past their subgoal's boundary (40, 70)."""
    end = SUBGOAL_COMMANDS[delta_subgoal(t)]
    return t + horizon > end


def normalised_rank(cost_demo: float, random_costs) -> float:
    """Rank of a_demo among a_demo and the random candidates under J's cost: the fraction of
    random candidates strictly cheaper, ties counted half (0 = best, 1 = worst, chance 0.5)."""
    r = np.asarray(random_costs, np.float64)
    return float(((r < cost_demo).sum() + 0.5 * (r == cost_demo).sum()) / len(r))


def delta_measures(states: list[dict], horizon: int) -> dict:
    """m1, m2 (the rows read the full set) and m3 (reported), overall, per subgoal and without the
    boundary-crossing windows. Each state: ``t``, ``cost_demo``, ``cost_cem``, ``rank``,
    optionally ``m3`` (bool)."""

    def summary(rows):
        if not rows:
            return {"states": 0, "m1": None, "m2": None, "m3": None}
        m3 = [r["m3"] for r in rows if r.get("m3") is not None]
        return {
            "states": len(rows),
            "m1": float(np.mean([r["cost_demo"] < r["cost_cem"] for r in rows])),
            "m2": float(np.median([r["rank"] for r in rows])),
            "m3": float(np.mean(m3)) if m3 else None,
        }

    return {
        "all": summary(states),
        "per_subgoal": {
            str(k + 1): summary([s for s in states if delta_subgoal(s["t"]) == k]) for k in (0, 1)
        },
        "without_boundary_windows": summary(
            [s for s in states if not delta_crosses_boundary(s["t"], horizon)]
        ),
    }


def delta_row(*, kept: int, m1, m2, guard_failed: bool = False, over_budget: bool = False) -> dict:
    """§18.6's rows, first match: Δ-VOID (a guard failed, fewer than 12 X seeds kept a
    demonstration, or the budget would be exceeded); Δ-MODEL (m2 >= 0.25); Δ-PLANNER (m2 <= 0.10
    and m1 >= 0.50); Δ-MIXED otherwise."""
    if guard_failed or over_budget or kept < DELTA_MIN_KEPT or m1 is None or m2 is None:
        row = "Δ-VOID"
    elif m2 >= DELTA_MODEL_RANK:
        row = "Δ-MODEL"
    elif m2 <= DELTA_PLANNER_RANK and m1 >= DELTA_PLANNER_M1:
        row = "Δ-PLANNER"
    else:
        row = "Δ-MIXED"
    return {"row": row, "kept": int(kept), "m1": m1, "m2": m2}
