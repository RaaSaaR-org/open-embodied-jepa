"""TASK-089: zero-shot grasp-and-lift, an offline diagnosis (Part A) gating a closed-loop test
with one declared change (Part B), over TASK-087's world models in MuJoCo.

Protocol: ``docs/experiments/zero_shot_grasp_v1.md``. This module holds the NumPy part: cohorts,
the Part A roots and branches (fresh grasp windows recorded from TASK-088's grasp demonstrator),
A1's wins, ceiling, bars and controls, A2's measures and bars, the selection rule, Part B's bar and
the rows. The torch part is ``zero_shot_grasp_runtime``. TASK-088's frozen code
(``zero_shot``, ``zero_shot_runtime``) is imported and never changed.

Information flow (protocol §4): object truth is read only by the demonstrator, the root builder
(which records the target's rise and grasp contact for reporting) and the scorer; never by a
controller or a planning cost.

NumPy only at import; MuJoCo is imported lazily by :mod:`embodied_jepa.simulation`. Nothing here
is registered or imported by ``embodied_jepa``.
"""

from __future__ import annotations

import math

import numpy as np

from embodied_jepa import play_corpus as pc
from embodied_jepa import zero_shot as zs
from embodied_jepa.contracts import ContractError
from embodied_jepa.embodiment import rotation_delta

TASK = "TASK-089"
PROTOCOL = "docs/experiments/zero_shot_grasp_v1.md"

# ----- seeds and salts (protocol §10) ------------------------------------------------------------
SALTS = {"shuffle": 8901, "reserved": 8902}
COHORTS = {
    "debug-roots": range(89900, 89950),
    "debug-grasp": range(89950, 90000),
    "k0-roots": range(89000, 89032),
    "a-roots": range(890000, 890064),
    "bd-grasp": range(89100, 89132),
    "bs-grasp": range(891000, 891064),
}
ROOT_COHORTS = ("debug-roots", "k0-roots", "a-roots")
GRASP_COHORTS = ("debug-grasp", "bd-grasp", "bs-grasp")


def check_seed(cohort: str, seed: int) -> None:
    if cohort not in COHORTS:
        raise ContractError(f"unknown cohort {cohort}")
    if int(seed) not in COHORTS[cohort]:
        raise ContractError(f"seed {seed} is not in cohort {cohort}")


# ----- Part A: roots and branches (protocol §5.1) ------------------------------------------------
WINDOW = 16
HORIZONS = (8, 16)
ROOTS = ("R1", "R2")
# root -> {branch: (palm target, wanted grasp)}; the first branch is the true one
BRANCHES = {
    "R1": {"close": ("aim", 1.0), "open": ("aim", -1.0)},
    "R2": {"lift": ("lift", 1.0), "release": ("lift", -1.0), "stay": ("aim", 1.0)},
}
TRUE_BRANCH = {"R1": "close", "R2": "lift"}
COMPARISONS = (("R1", "close", "open"), ("R2", "lift", "release"), ("R2", "lift", "stay"))
# the subgoal a root's comparisons are scored against (TASK-088's subgoal order)
ROOT_GOAL = {"R1": 2, "R2": 3}  # R1 -> the grasp subgoal, R2 -> the lift subgoal
ROOT_STATE_SUBGOAL = {"R1": 1, "R2": 2}  # the root equals the pre-grasp / grasp subgoal's state


def comparison_name(comp) -> str:
    root, t, a = comp
    return f"{root}:{t}-vs-{a}"


class _Demo:
    """TASK-088's grasp demonstrator (``zero_shot.make_grasp_goal``), phase by phase, so the same
    trajectory can be re-run from its reset up to a root and then branched."""

    def __init__(self, robot, layout, yaw):
        self.robot = robot
        zs.reset(robot, layout)
        self.index = zs._target_index(layout)
        truth = robot.sim.play_truth()
        self.z0 = float(truth["position"][self.index][2])
        self.aim = pc.to_pelvis(truth["position"][self.index]) + pc.GRASP_OFFSET
        self.rotation = rotation_delta([0, 0, yaw]) @ zs.TOP_DOWN
        self.grasp = -1.0
        self.log = []

    def target(self, name):
        if name == "aim":
            return self.aim
        if name == "hover":
            return self.aim + [0, 0, zs.GRASP_HOVER]
        if name == "lift":
            return self.aim + [0, 0, zs.GRASP_LIFT]
        raise ContractError(name)

    def run(self, target, want, steps, record=None):
        robot = self.robot
        for _ in range(steps):
            self.grasp = float(
                np.clip(want, self.grasp - zs.GRASP_RAMP, self.grasp + zs.GRASP_RAMP)
            )
            obs = robot.observe()
            p = zs.proprio(robot, obs)
            request = zs.tracking_action(
                robot, p["palm"], p["rotation"], self.target(target), self.rotation, self.grasp
            )
            applied = zs.execute(robot, request)
            truth = robot.sim.play_truth()
            self.log.append(truth)
            if record is not None:
                after = robot.observe()
                record["actions"].append(np.asarray(applied, np.float32))
                record["frames"].append(after.images[pc.CAMERA][0].copy())
                record["states"].append(zs.proprio(robot, after)["state28"])
                record["rise"].append(float(truth["position"][self.index][2] - self.z0))
                record["contact"].append(bool(truth["grasp_contact"][self.index]))

    def to_root(self, root):
        """The demonstrator's phases up to a root: hover, descend open (R1), close (R2)."""
        phases = zs.GRASP_PHASES
        self.run("hover", -1.0, phases["hover"])
        self.run("aim", -1.0, phases["descend"])
        if root == "R2":
            self.run("aim", zs.GRASP_CLOSURE, phases["close"])


def _record(robot):
    obs = robot.observe()
    return {"frame": obs.images[pc.CAMERA][0].copy(), "state28": zs.proprio(robot, obs)["state28"]}


def build_root(robot, seed: int) -> dict:
    """Protocol §5.1: the demonstrator's draw exactly as ``zero_shot.make_grasp_goal`` makes it
    (same layout, yaw and acceptance), then the two roots and their 16-command branches, each
    re-run from the reset. A draw whose demonstrator is stopped or does not lift, or whose branch
    is stopped, is replaced by the next attempt (counted in ``rejected``)."""
    rng = np.random.default_rng([zs.SALTS["goal"], int(seed)])
    rejected = []
    for attempt in range(zs.GOAL_ATTEMPTS):
        layout = zs.sample_layout("grasp", seed, attempt)
        yaw = float(rng.uniform(-pc.YAW_RANGE, pc.YAW_RANGE))
        try:
            demo = _Demo(robot, layout, yaw)
            ph = zs.GRASP_PHASES
            subgoals = []
            demo.run("hover", -1.0, ph["hover"])
            subgoals.append(_record(robot))
            demo.run("aim", -1.0, ph["descend"])
            subgoals.append(_record(robot))
            demo.run("aim", zs.GRASP_CLOSURE, ph["close"])
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
        out = {
            "seed": int(seed),
            "attempt": attempt,
            "layout": layout,
            "target": layout["target"],
            "yaw": yaw,
            "z0": demo.z0,
            "subgoals": subgoals,
            "roots": {},
            "branches": {},
        }
        try:
            for root in ROOTS:
                for name, (target, want) in BRANCHES[root].items():
                    demo = _Demo(robot, layout, yaw)
                    demo.to_root(root)
                    here = _record(robot)
                    expect = subgoals[ROOT_STATE_SUBGOAL[root]]["state28"]
                    if not np.array_equal(here["state28"], expect):
                        raise ContractError(f"seed {seed}: the re-run root {root} differs")
                    out["roots"][root] = here
                    rec = {k: [] for k in ("actions", "frames", "states", "rise", "contact")}
                    demo.run(target, want, WINDOW, rec)
                    out["branches"][f"{root}/{name}"] = {
                        "actions": np.stack(rec["actions"]),
                        "frames": np.stack(rec["frames"]),
                        "states": np.stack(rec["states"]).astype(np.float32),
                        "rise": rec["rise"],
                        "contact": rec["contact"],
                    }
        except zs.Stopped as error:
            rejected.append(f"branch stopped: {error}")
            continue
        out["rejected"] = rejected
        return out
    raise ContractError(f"no root for seed {seed}: {rejected}")


def shuffled_deal(seed: int, comparison: int, true_actions, alt_actions):
    """A1's action-shuffled control: the 2 x 16 commands of one comparison's two branches pooled
    and dealt at random into two 16-command sequences (generator keyed by salt 8901, the root seed
    and the comparison's index 0-2)."""
    if comparison not in range(len(COMPARISONS)):
        raise ContractError(f"no comparison {comparison}")
    pool = np.concatenate([np.asarray(true_actions), np.asarray(alt_actions)])
    rng = np.random.default_rng([SALTS["shuffle"], int(seed), int(comparison)])
    order = rng.permutation(len(pool))
    return pool[order[:WINDOW]], pool[order[WINDOW:]]


# ----- A1 statistics, bars and pass rule (protocol §5.2, §5.4, §5.5) -----------------------------
GATE_CEILING = 0.75
BAR_FACTOR = 0.9
A_ROOTS = 64
COST_FAMILIES = ("latent", "latent+pose")  # P / G-lat, and G
ARM_FAMILY = {"G": "latent+pose", "P": "latent", "G-lat": "latent", "N": "latent"}
P_THRESHOLD = 0.01


def wins(true_costs, alt_costs) -> np.ndarray:
    """A root wins when the true branch's cost is strictly lower; ties are not wins."""
    return np.asarray(true_costs, float) < np.asarray(alt_costs, float)


def gated(ceiling_rate: float) -> bool:
    return ceiling_rate >= GATE_CEILING - 1e-12


def a1_bar(ceiling_rate: float) -> float:
    return 0.5 + BAR_FACTOR * (ceiling_rate - 0.5)


def bar_count(bar: float, n: int) -> int:
    return int(math.ceil(bar * n - 1e-9))


def mcnemar_one_sided(first, second) -> float:
    """Exact one-sided McNemar p that indicator ``first`` beats ``second`` (paired)."""
    a, b = np.asarray(first, bool), np.asarray(second, bool)
    return zs.mcnemar_one_sided(int((a & ~b).sum()), int((~a & b).sum()))


def k0_bars(ceiling: dict) -> dict:
    """§5.4 from A-K0's ceiling rates ``{family: {h: {comparison: rate}}}`` (h as a string, as in
    the JSON records): which comparisons are gated and their bars."""
    out = {}
    for family, by_h in ceiling.items():
        out[family] = {}
        for h, by_c in by_h.items():
            out[family][str(h)] = {
                c: {"ceiling": r, "gated": gated(r), "bar": a1_bar(r) if gated(r) else None}
                for c, r in by_c.items()
            }
    return out


def qualifying(bars: dict, h) -> bool:
    """§5.4: h qualifies when, for G's cost family, close-vs-open and at least one R2 comparison
    are gated at it."""
    names = [comparison_name(c) for c in COMPARISONS]
    b = bars["latent+pose"][str(h)]
    return bool(b[names[0]]["gated"] and (b[names[1]]["gated"] or b[names[2]]["gated"]))


def ceiling_row(bars: dict) -> bool:
    """A-CEILING (§5.4, §8): no horizon qualifies."""
    return not any(qualifying(bars, h) for h in HORIZONS)


def a1_pass(family_bars: dict, model_wins: dict, shuffled_wins: dict, n: int) -> dict:
    """§5.5 for one model at one horizon. ``family_bars``: ``{comparison: {gated, bar}}`` for the
    model's cost family at h; ``model_wins`` / ``shuffled_wins``: ``{comparison: bool[n]}``."""
    names = [comparison_name(c) for c in COMPARISONS]
    detail, ok = {}, True
    usable = family_bars[names[0]]["gated"] and (
        family_bars[names[1]]["gated"] or family_bars[names[2]]["gated"]
    )
    for c in names:
        k = int(np.sum(model_wins[c]))
        p = mcnemar_one_sided(model_wins[c], shuffled_wins[c])
        entry = {"wins": k, "n": n, "p_vs_shuffled": p, "gated": family_bars[c]["gated"]}
        if family_bars[c]["gated"]:
            need = bar_count(family_bars[c]["bar"], n)
            entry["need"] = need
            entry["meets"] = bool(k >= need and p < P_THRESHOLD)
            ok = ok and entry["meets"]
        detail[c] = entry
    return {"pass": bool(usable and ok), "usable": bool(usable), "comparisons": detail}


# ----- A2 and the selection rule (protocol §5.3, §6) ---------------------------------------------
CANDIDATES = ("step-mean", "h16", "held")  # in the selection order
SETTINGS = ("base",) + CANDIDATES  # "base" is TASK-088's setting
SETTING_HORIZON = {"base": 8, "step-mean": 8, "h16": 16, "held": 8}
A2_BAR = 48  # of 64, declared (§5.4)
GRASP_DIM, Z_DIM = 13, 8


def a2_measures(first_actions_r1, first_actions_r2) -> dict:
    """close@R1: the first planned command's right grasp > 0.5; lift@R2: grasp > 0.5 and the arm
    z command > 0."""
    r1 = np.asarray(first_actions_r1, float)
    r2 = np.asarray(first_actions_r2, float)
    return {
        "close_R1": r1[:, GRASP_DIM] > 0.5,
        "lift_R2": (r2[:, GRASP_DIM] > 0.5) & (r2[:, Z_DIM] > 0),
    }


def a2_meets(candidate: dict, base: dict) -> dict:
    out = {}
    for key in ("close_R1", "lift_R2"):
        k = int(np.sum(candidate[key]))
        p = mcnemar_one_sided(candidate[key], base[key])
        out[key] = {"count": k, "p_vs_base": p, "meets": bool(k >= A2_BAR and p < P_THRESHOLD)}
    out["meets"] = all(v["meets"] for v in out.values() if isinstance(v, dict))
    return out


def select_change(a1_by_h: dict, a2: dict) -> str | None:
    """§6: the first candidate whose horizon qualifies and A1 passes there for G-87100 (both in
    ``a1_by_h[h]``, False for a horizon that does not qualify) and whose A2 bars are met."""
    for cand in CANDIDATES:
        if a1_by_h[SETTING_HORIZON[cand]] and a2[cand]["meets"]:
            return cand
    return None


def a_row(*, void: str = "", ceiling: bool = False, a1_by_h: dict, change) -> str:
    """§8 rows 1-5 (first match). ``a1_by_h[h]`` is True only when h qualifies and A1 passes for
    G-87100 there, so A-FAIL is judged on qualifying horizons only."""
    if void:
        return "A-VOID"
    if ceiling:
        return "A-CEILING"
    if not any(a1_by_h.values()):
        return "A-FAIL"
    if change is None:
        return "A-PLAN"
    return f"A-PASS({change})"


# ----- Part B (protocol §7) ----------------------------------------------------------------------
PLAN_GRASP_BAR = 0.40
PRIMARY_SEED = 87100
REPLICATION_SEEDS = (87101, 87102)
# arm -> (TASK-087 model arm, latent weight, pose weight, uses the selected change)
MODEL_ARMS = {
    "G-chg": ("G", 1.0, 1.0, True),
    "P-chg": ("P", 1.0, 0.0, True),
    "G-base": ("G", 1.0, 1.0, False),
}
BASELINE_ARMS = zs.BASELINE_ARMS  # hold, random, ik, ik-nopress


def check_run(cohort: str, arm: str, model_seed) -> None:
    """§7.1: which (arm, model seed) a Part B cohort admits."""
    if cohort not in GRASP_COHORTS:
        raise ContractError(f"{cohort} is not a Part B cohort")
    if arm in BASELINE_ARMS:
        if model_seed is not None:
            raise ContractError("baseline arms take no model seed")
        return
    if arm not in MODEL_ARMS:
        raise ContractError(f"unknown arm {arm}")
    if model_seed == PRIMARY_SEED:
        return
    if arm == "G-chg" and model_seed in REPLICATION_SEEDS and cohort in ("bs-grasp", "debug-grasp"):
        return
    raise ContractError(f"{arm}-{model_seed} is not a run of {cohort}")


B_ESCALATE_RATE = 0.30


def b_bar(ik_successes: int, n: int):
    """§7.2: min(0.40, 0.9 x ik's Stage BD rate); None (escalate) when that rate is below 0.30."""
    rate = ik_successes / n
    if rate < B_ESCALATE_RATE:
        return None
    return min(PLAN_GRASP_BAR, 0.9 * rate)


def b_row(primary, random, bar: float, *, void: str = "") -> dict:
    """§7.2 rows (first match) on G-chg (87100); ``primary`` and ``random`` are paired success
    indicators."""
    n = len(primary)
    k = int(np.sum(primary))
    need = bar_count(bar, n)
    p = mcnemar_one_sided(primary, random)
    if void:
        name = "B-VOID"
    elif k >= need and p < P_THRESHOLD:
        name = "B-PASS"
    elif p < P_THRESHOLD:
        name = "B-LOW"
    else:
        name = "B-FAIL"
    return {"row": name, "k": k, "n": n, "need": need, "p_vs_random": p, "void_reason": void}


def clause_fires(row: str) -> bool:
    """§9: the abandonment clause fires on A-FAIL or B-FAIL."""
    return row in ("A-FAIL", "B-FAIL")


# ----- the bars (§5.4, §7.2): written by the Stage 0 and Stage BD records ------------------------
A1_BARS: dict | None = None  # set from A-K0 by the Stage 0 record, frozen with the protocol
