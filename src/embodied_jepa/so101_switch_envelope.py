"""TASK-101 amendment A2 (``docs/experiments/so101_closed_loop_lift_v1.md`` §19): the pose-error
tolerance envelope that calibrates the subgoal switching tolerance.

Development measurement, not gated, no claim. The privileged scripted follower v1
(``so101_followers``, ``FOLLOWER_V1``) runs its grasp and lift on ``so101_cube_lift_v1`` with its
arm handed over at a perturbed pose: at the pre-grasp pose (subgoal 1, after command 40) or at the
end of the descent (subgoal 2, after command 70), offset in joint space by a random direction of a
declared RMS size over the five arm joints. The lift counts per size and subgoal set the switching
tolerance τ_s by the rule in :func:`envelope_row`, declared before any count was read.

Hand-over, exactly (§19.3):

* **subgoal 1 ("pre"):** the pre-grasp phase drives to ``q_pre + δ`` instead of ``q_pre`` for its
  40 commands (jaw open). The follower then continues its own schedule unchanged: the descent's
  Cartesian waypoints, the close and the lift, re-aimed from simulator truth as always. This
  models a switch off subgoal 1 at pose error δ, after which the next phase tracks subgoal 2.
* **subgoal 2 ("carry", the reading the rule uses):** the descent's waypoint targets are offset by
  ``δ · k / 6`` (k = 1 … 6), so the arm arrives at ``q_grasp + δ`` at command 70; the close phase
  holds the arm there (``drive(q_grasp + δ, -1, 20)``), so the jaw closes at the perturbed pose; the
  lift's waypoints are the follower's own. This models a switch off subgoal 2 at pose error δ,
  after which the jaw closes where the arm is.
* **subgoal 2 ("reaim", reported only):** as "carry", but the close phase drives back to the
  follower's own ``q_grasp`` while the jaw closes.

With δ = 0 every variant executes exactly ``follow_lift``'s commands (a guard of the run).

NumPy only at import; MuJoCo comes in through the simulation object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TASK = "TASK-101 A2"
SALT = 10109
LEVELS_RAD = (0.02, 0.04, 0.06, 0.08, 0.12, 0.16)
PER_LEVEL = 32
PASS_COUNT = 30
CURRENT_TOLERANCE_RAD = 0.04
MIN_USEFUL_LEVEL_RAD = 0.06
ARM_JOINTS = 5
HANDOVER_COMMAND = {1: 40, 2: 70}
VARIANTS = (("pre", 1), ("carry", 2), ("reaim", 2))
RULE_VARIANTS = ("pre", "carry")
REPORTED_VARIANTS = ("reaim",)


@dataclass(frozen=True)
class Cohort:
    name: str
    first: int
    count: int
    reserve: tuple[int, int]

    def main(self) -> list[int]:
        return list(range(self.first, self.first + self.count))

    def spare(self) -> list[int]:
        return list(range(self.reserve[0], self.reserve[1] + 1))

    def contains(self, seed: int) -> bool:
        return seed in self.main() or seed in self.spare()


# Fresh development seeds of the lift stream (``default_rng([9801, seed])``), disjoint from every
# TASK-101 cohort (K, K', D, D', S, X, debug) and S2's seeds; unused on every remote ref
# (``git grep -w``, 2026-10-19). K'' and D'' are declared for Stage 0'' and not used here.
ENVELOPE = Cohort("E", 102000, PER_LEVEL, (102032, 102047))
K_SECOND = Cohort("K''", 102100, 32, (102132, 102139))
D_SECOND = Cohort("D''", 102200, 64, (102264, 102299))


def check_seed(seed: int) -> None:
    if type(seed) is not int or not ENVELOPE.contains(seed):
        raise ValueError(f"seed {seed!r} is not in the envelope cohort E")


def direction(seed: int, subgoal: int) -> np.ndarray:
    """The unit-RMS joint-space direction for ``seed`` and ``subgoal`` (salt 10109): a standard
    normal draw over the five arm joints, scaled so its RMS is exactly 1. The same direction is
    used at every level, so the levels of one seed differ only in size."""
    if subgoal not in HANDOVER_COMMAND:
        raise ValueError("subgoal must be 1 or 2")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    u = np.random.default_rng([SALT, seed, subgoal]).standard_normal(ARM_JOINTS)
    return u / math.sqrt(float(np.mean(u * u)))


def offset(seed: int, subgoal: int, level: float) -> np.ndarray:
    """δ: RMS exactly ``level`` rad over the five arm joints."""
    if not (math.isfinite(level) and level >= 0):
        raise ValueError("level must be finite and nonnegative")
    return float(level) * direction(seed, subgoal)


def rms(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(math.sqrt(float(np.mean(x * x))))


def perturbed_lift(robot, task, seed: int, *, variant: str, delta, config=None) -> dict:
    """Follower v1's lift on ``seed`` with the arm handed over at ``q + delta`` (module docstring).

    Returns the follower's record plus the measured arm joints after the hand-over command
    (``handover_joints``) and whether the episode ended before it."""
    from embodied_jepa import so101_followers as f
    from embodied_jepa.so101_scripted import JAW_CENTRE_OFFSET_M, _closing_axis, solve_ik

    names = dict(VARIANTS)
    if variant not in names:
        raise ValueError(f"unknown variant {variant!r}")
    subgoal = names[variant]
    cfg = f.FOLLOWER_V1 if config is None else config
    d = np.asarray(delta, dtype=float)
    if d.shape != (ARM_JOINTS,) or not np.isfinite(d).all():
        raise ValueError("delta must be five finite arm-joint offsets")

    task.reset(seed)
    run = f._Runner(robot, task)
    sim = run.sim
    record: dict = {"seed": seed, "variant": variant, "subgoal": subgoal, "delta": d.tolist()}
    handover = {"joints": None}
    count = {"n": 0}
    target_command = HANDOVER_COMMAND[subgoal]
    inner_step = run.step

    def step(action):
        try:
            inner_step(action)
        finally:
            count["n"] += 1
            if count["n"] == target_command:
                handover["joints"] = sim.data.qpos[sim.qadr].copy()

    run.step = step
    try:
        # _pick's grasp pose, with no grasp offset (follower v1, as the goal demonstration).
        truth = sim.task_truth()
        cube = truth["cube_position"]
        closing, _ = f._grasp_frame(cube, truth["cube_quaternion"])
        shift = JAW_CENTRE_OFFSET_M - cfg.fixed_jaw_clearance_m

        def aim(height):
            point = cube + np.array([0.0, 0.0, height])
            q, _ = solve_ik(sim, run.q, point, closing)
            return point + _closing_axis(sim, q) * shift

        f._site_rotation(run, aim(0.0), closing)  # as _pick (it advances nothing; kept for parity)
        pre, grasp = aim(cfg.pregrasp_height_m), aim(0.0)
        q_pre, _ = run.ik(pre, closing)
        run.drive(q_pre + d if subgoal == 1 else q_pre, 1.0, cfg.pregrasp_steps)
        # Descent: the follower's own waypoints (``_Runner.through``), offset ramped for subgoal 2.
        per = max(1, cfg.descend_steps // cfg.descend_waypoints)
        for k in range(1, cfg.descend_waypoints + 1):
            q, _ = run.ik(pre + (grasp - pre) * k / cfg.descend_waypoints, closing)
            ramp = d * k / cfg.descend_waypoints if subgoal == 2 else 0.0
            run.drive(q + ramp, 1.0, per)
        close_goal = run.q + d if variant == "carry" else run.q
        run.drive(close_goal, -1.0, cfg.close_steps)
        lifted = grasp + np.array([0.0, 0.0, cfg.lift_height_m])
        run.through(grasp, lifted, 3, -1.0, cfg.lift_steps, closing)
        run.hold()
    except f.EpisodeDone:
        pass
    out = f._finish(run, record)
    out["commands"] = count["n"]
    out["handover_joints"] = None if handover["joints"] is None else handover["joints"].tolist()
    out["ended_before_handover"] = handover["joints"] is None
    return out


# ----- the rule (§19.4), fixed before any count was read --
def level_passes(counts: dict, level: float) -> bool:
    """Both rule readings (subgoal 1 "pre" and subgoal 2 "carry") lift on ≥ 30 of 32."""
    return all(counts[(v, level)] >= PASS_COUNT for v in RULE_VARIANTS)


def envelope_row(counts: dict, kept: int = PER_LEVEL, void: str | None = None) -> dict:
    """The A2 row from ``counts[(variant, level)]`` (successes of 32 kept seeds).

    τ_s is the largest level L such that every level ≤ L passes (both readings ≥ 30/32). If that
    L is below 0.06 rad (or no level passes), τ_s stays 0.04 rad and the row is
    A2-NOT-BINDING: the tolerance is not the binding constraint; escalate (hybrid local grasp
    controller or a LeWM longer-budget retrain, the owner's choice); K'' does not run. Otherwise
    the row is A2-TAU and K'' runs with τ_s."""
    if void is not None or kept != PER_LEVEL:
        return {"row": "A2-VOID", "tau_s_rad": None, "reason": void or f"kept {kept} of 32"}
    for v in RULE_VARIANTS:
        for level in LEVELS_RAD:
            c = counts[(v, level)]
            if type(c) is not int or not 0 <= c <= PER_LEVEL:
                raise ValueError(f"count for {(v, level)} must be an integer in 0..32")
    largest = None
    for level in sorted(LEVELS_RAD):
        if not level_passes(counts, level):
            break
        largest = level
    current_ok = level_passes(counts, CURRENT_TOLERANCE_RAD)
    if largest is None or largest < MIN_USEFUL_LEVEL_RAD:
        return {
            "row": "A2-NOT-BINDING",
            "tau_s_rad": CURRENT_TOLERANCE_RAD,
            "largest_passing_rad": largest,
            "current_tolerance_passes": current_ok,
            "reason": "tolerance not the binding constraint; escalate to the owner",
        }
    return {
        "row": "A2-TAU",
        "tau_s_rad": largest,
        "largest_passing_rad": largest,
        "current_tolerance_passes": current_ok,
        "reason": "K'' runs with tau_s (§19.5)",
    }


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided interval for k of n (bisection on the binomial tail; NumPy only)."""
    if not (0 <= k <= n and n > 0):
        raise ValueError("need 0 <= k <= n, n > 0")

    def tail_ge(p):  # P(X >= k)
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))

    def tail_le(p):  # P(X <= k)
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))

    def solve(fn, target, increasing):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if (fn(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(tail_ge, alpha / 2, True)
    upper = 1.0 if k == n else solve(tail_le, alpha / 2, False)
    return lower, upper
