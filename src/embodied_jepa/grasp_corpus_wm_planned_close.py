"""TASK-091: the planned close within the plan -- TASK-090's A′ on fresh roots with A2's close bar
redefined (Stage A″), then TASK-090's closed-loop grasp-and-lift comparison on fresh resets
(protocol ``docs/experiments/grasp_play_corpus_v1_planned_close.md``).

This module holds TASK-091's cohorts and seeds (§8), the A″ measures, pass rule and rows (§4,
§6), the per-job cap and the BS job split (§5.4), the split runs' recombination, and the
receding-horizon deferral measures (§5.3, reported only). For everything else -- the A1 rule,
the bars, the sign-flip tests, the bootstrap and the B rows -- it uses TASK-090's frozen
``grasp_corpus_wm`` and TASK-089's ``zero_shot_grasp`` unchanged.

NumPy only at import. Simulator truth (the target's contact) is read only by the reported
deferral measures, after the episode; never by a model input or a planning cost. Nothing here is
registered or imported by ``embodied_jepa``.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from embodied_jepa import grasp_corpus_wm as gc
from embodied_jepa import zero_shot as zs
from embodied_jepa import zero_shot_grasp as zg
from embodied_jepa.contracts import ContractError

TASK = "TASK-091"
PROTOCOL = "docs/experiments/grasp_play_corpus_v1_planned_close.md"

# ----- seeds (§8) ------------------------------------------------------------------------------
# No new model seed and no new salt: layouts and goals 8801/8802, CEM 8803, random 8804 (TASK-088);
# A″'s shuffled deal 9002, the readability probe 9004, the reset-cluster bootstrap 9006 (TASK-090).
DEBUG_SEEDS = range(90700, 90800)  # Stage 0 plumbing and smoke; never counted
COHORTS = {
    "debug-roots": range(90700, 90740),
    "debug-grasp": range(90740, 90800),
    "a2-roots": range(906000, 906064),
    "bd-grasp": range(90300, 90332),
    "bs-grasp": range(907000, 907128),
}
ROOT_COHORTS = ("debug-roots", "a2-roots")
GRASP_COHORTS = ("debug-grasp", "bd-grasp", "bs-grasp")
# TASK-090's and D-A2's ranges, which TASK-091 must not touch (§5.2, §8): A-K0, A′, D-A2, the
# declared but never run BD and BS cohorts, TASK-090's debug range and its collections
TASK090_RANGES = {
    "k0-roots": range(90000, 90032),
    "a-roots": range(903000, 903064),
    "d-a2-roots": range(905000, 905064),
    "bd-grasp (declared, unrun)": range(90200, 90232),
    "bs-grasp (declared, unrun)": range(904000, 904128),
    "debug": range(90900, 91000),
    "play-v1p": range(910000, 913200),
    "play-v2g": range(920000, 923200),
    "debug collections": range(930000, 930400),
}
BS_N = len(COHORTS["bs-grasp"])  # 128, TASK-090's power rule (R27.22), not recomputed
DETERMINISM = ("M-v2g", 90100, 4)  # the first four BS resets, 907000-907003 (§5.2)

# the closed-loop arms and their model seeds: TASK-090's (§5.1; M-v1 87100 alone in BS, R27.25)
CLOSED_LOOP_ARMS = gc.CLOSED_LOOP_ARMS
BD_SEEDS = gc.BD_SEEDS
BS_SEEDS = gc.BS_SEEDS
MODEL_ARMS = tuple(CLOSED_LOOP_ARMS)
BASELINE_ARMS = zs.BASELINE_ARMS


def check_seed(cohort: str, seed: int) -> None:
    if cohort not in COHORTS:
        raise ContractError(f"unknown cohort {cohort}")
    if int(seed) not in COHORTS[cohort]:
        raise ContractError(f"seed {seed} is not in cohort {cohort}")


def check_run(cohort: str, arm: str, model_seed) -> None:
    """§5.1-§5.2: which (arm, model seed) a cohort admits."""
    if cohort not in GRASP_COHORTS:
        raise ContractError(f"{cohort} is not a closed-loop cohort")
    if arm in BASELINE_ARMS:
        if model_seed is not None:
            raise ContractError("baseline arms take no model seed")
        return
    if arm not in CLOSED_LOOP_ARMS:
        raise ContractError(f"unknown arm {arm}")
    allowed = BS_SEEDS if cohort in ("bs-grasp", "debug-grasp") else BD_SEEDS
    if cohort == "debug-grasp":
        allowed = {a: tuple(BS_SEEDS[a]) + tuple(BD_SEEDS[a]) for a in CLOSED_LOOP_ARMS}
    if model_seed not in allowed[arm]:
        raise ContractError(f"{arm}-{model_seed} is not a run of {cohort}")


# ----- the models (§3): TASK-090 §19.2's and TASK-087's checkpoints, pinned --------------------
CHECKPOINT_SHA256 = {
    "G-v2g-90100": "6ca905254c82ca5c643cf4a5114e1529271fb4fdf04339a49ffc9d973312c375",
    "G-v2g-90101": "21332224bd5cedf27089bb280e91ec7f82f96ac206c32da2820c4bb6c6b768ed",
    "G-v2g-90102": "129994c9c59ff72cedf329d23201a9db49f78d0c9b75d30f2713f6f7d7c76ad9",
    "G-v1p-90100": "1a8a2991d5472deb6777cfa57132d46294a3c505359f01a97864f092353263b2",
    "G-v1p-90101": "eab00cf28fc499d96de269c8f2ae6df4a22c86fe346fb39c7c3c8ce45b95cfeb",
    "G-v1p-90102": "c594eeb6065076dc85b6a3fbe3e0c1f78835f6927a89910992e29c2b06e31495",
    "N-v2g-90100": "c0ee2a02ac8d4e979ca9b64d3a3195f549803ce0b4d5ec19b19485578072a04c",
    "N-v1p-90100": "26efc846191d5c08ef8dd7884c38993fb621648e1598d26aa98874a9cce25e5e",
    "G-v1-87100": "17921a7fa15a4e310a4a99ad52f1133affa1598a824cd100fef609370a1ef273",
    "G-v1-87101": "a59f5b6434b61f0481c534c4af45fc792474171ba44af4ce9d5684089272e08d",
    "G-v1-87102": "9c797cd818db06ba8d857730bd16dda96a39674b9932df143f50c25496e56a81",
}
FEATURE_REPORTS_SHA256 = {
    "play-v1": "f5648a3529fe061dea219c26816253ea261c25ff891f043e0eb2a136e38e7eae",
    "play-v2g": "d8afb23642ef23706c7b213f9174e7d4bcdf102ca774a5fa287bcc62b404e988",
    "play-v1p": "00625a44023ad4cdb424c272598692aaa6899b51aec8b369e4d89adf64f2822e",
}
MOMENTS_SHA256 = "1e960227194ce04abb456ef3d517d070f8991ef1925eced88c0d58083ff81b5c"
LAMBDA = 2.0552
LAST_TWO_TRIGGERED = ("G-v2g-90102", "G-v1p-90100", "G-v1p-90101")  # reported (TASK-090 §2.2)

# TASK-090's frozen code at 53f0b72, which TASK-091 runs and does not edit (§12); the runner
# refuses to start if any of these files differs
FROZEN_REVISION = "53f0b72f75b61031c9bbba714b3e84c32b194925"
FROZEN_FILES_SHA256 = {
    "src/embodied_jepa/grasp_corpus_wm.py": (
        "a6fc185fd0e85923c4bdd829328200547e1e48e551c2c7ff9163cb354db5fb1d"
    ),
    "src/embodied_jepa/grasp_corpus_wm_runtime.py": (
        "2b590405dd05e1fd0474f25b31117926e7c1b6c8541c90ae043a7b73cacd295f"
    ),
    "src/embodied_jepa/zero_shot_grasp.py": (
        "cb87c91c8413d120ef9f1e93b4c1009cc13678336114ba396428915c7a252bb8"
    ),
    "src/embodied_jepa/zero_shot_grasp_runtime.py": (
        "e14cb986a70c3b5d8fabe53c165fef7c7ced6dac71c444a3f6b25b8b140bd47d"
    ),
    "src/embodied_jepa/zero_shot.py": (
        "a3fa5b03476ee934a3d9305c7d356aa78e92f401b02a2a935e5873e3f33dbfe9"
    ),
    "src/embodied_jepa/zero_shot_runtime.py": (
        "16c8ca1b25f24912c4d1c78be27b3bb921c5973bf3688bf80f2a8dec8461ac93"
    ),
    "src/embodied_jepa/planning.py": (
        "e9aea01c81f07e775a4d83116dadaebf7f723090ca4139c6289a4846d5891297"
    ),
    "scripts/run_task090.py": "79349d2694e2546e4e65856ce315d7fa8b9aa980bad385d4aa16bf8d2ad985e8",
    "scripts/summarize_task090.py": (
        "c550315544a4dcde9ae65474067e8b68f254f8ff66dada88f6d372ffc3cfbe82"
    ),
}


def frozen_files_check(root) -> dict:
    """{file: equal to its pin}; every one must hold before any stage runs."""
    out = {}
    for rel, pin in FROZEN_FILES_SHA256.items():
        path = Path(root) / rel
        out[rel] = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == pin
    return out


def model_name(arm: str, corpus: str, seed: int) -> str:
    return f"{arm}-{gc.CORPUS_TAG.get(corpus, 'v1')}-{seed}"


def a_second_models() -> list:
    """§3: the nine G and two N checkpoints of TASK-090's A′ and D-A2, in A′'s order."""
    out = [("G", c, s) for c in gc.CORPORA for s in gc.MODEL_SEEDS]
    out += [("N", c, gc.N_SEED) for c in gc.CORPORA]
    out += [("G", "play-v1", s) for s in (87100, 87101, 87102)]
    return out


# ----- Stage A″ (§4) ---------------------------------------------------------------------------
GRASP_DIM = zg.GRASP_DIM  # 13, the right-grasp channel
Z_DIM = zg.Z_DIM  # 8, right palm dz
CLOSE = 0.5
CLOSE_STEP_MAX = 4  # the redefined close bar: a close planned at plan step k <= 4
A2_BAR = gc.A2_BAR  # 48 of 64
A_SECOND_H = gc.A_PRIME_H  # A1's pass rule at h = 8
A_SECOND_SETTING = gc.A_PRIME_SETTING  # step-mean, lambda = 2.0552
REPORT_STEPS = (2, 4, 8)


def first_close_step(plan) -> int | None:
    """The first plan step k (1-based) whose right-grasp channel is > 0.5; None if none."""
    closes = np.asarray(plan, np.float64)[:, GRASP_DIM] > CLOSE
    return int(np.argmax(closes)) + 1 if closes.any() else None


def a2_second(plans_r1, plans_r2) -> dict:
    """§4's A2 on the A2-seed plans (``[n, H, 14]`` at R1 and at R2): close≤4@R1 (the redefined
    bar) and lift@R2 (kept: the first planned command at R2) each >= 48; close@R1 (the first
    command, A′'s old bar), the first closing step at R1, the share closing by k <= 2, 4, 8 and
    the mean right-grasp channel per plan step at R1 are reported."""
    p1 = np.asarray(plans_r1, np.float64)
    p2 = np.asarray(plans_r2, np.float64)
    if p1.ndim != 3 or p2.ndim != 3 or p1.shape[0] != p2.shape[0]:
        raise ContractError("a2_second needs [n, H, 14] plans at R1 and R2")
    n = p1.shape[0]
    k = [first_close_step(p) for p in p1]
    close_le = sum(1 for v in k if v is not None and v <= CLOSE_STEP_MAX)
    first = zg.a2_measures(p1[:, 0], p2[:, 0])
    lift = int(first["lift_R2"].sum())
    out = {
        "close_le4_R1": {"count": int(close_le), "meets": bool(close_le >= A2_BAR)},
        "lift_R2": {"count": lift, "meets": bool(lift >= A2_BAR)},
        "close_R1_reported": int(first["close_R1"].sum()),
        "first_close_step_R1": k,
        "close_by_R1": {
            str(s): int(sum(1 for v in k if v is not None and v <= s)) for s in REPORT_STEPS
        },
        "never_close_R1": int(sum(v is None for v in k)),
        "mean_grasp_per_step_R1": [float(v) for v in p1[:, :, GRASP_DIM].mean(0)],
        "n": n,
        "bar": A2_BAR,
        "close_step_max": CLOSE_STEP_MAX,
    }
    out["meets"] = out["close_le4_R1"]["meets"] and out["lift_R2"]["meets"]
    return out


def a_second_pass(a1_h8: dict, a2: dict) -> bool:
    """A″ for one model: TASK-089 §5.5's A1 pass rule at h = 8 and both of §4's bars."""
    return bool(a1_h8["pass"] and a2["meets"])


def a_second_row(passes: dict, *, void: str = "") -> dict:
    """``passes[arm][seed]`` for arm in v2g, v1p: A″-VOID if a check failed, else A″-FAIL when
    either arm fails on two or more of its three seeds (A′-FAIL's rule), else A″-PASS."""
    fails = {arm: sum(not v for v in by.values()) for arm, by in passes.items()}
    bad = [arm for arm, k in fails.items() if k >= 2]
    if void:
        row = "A″-VOID"
    elif bad:
        row = "A″-FAIL"
    else:
        row = "A″-PASS"
    return {"row": row, "void_reason": void, "fails": fails, "failing_arms": bad}


def first_command_equal(plan, first) -> bool:
    """§4: the A2-seed plan's first command equals ``first_planned``'s bitwise."""
    a = np.asarray(plan, np.float32)[0]
    b = np.asarray(first, np.float32)
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))


# ----- per-job cap and the BS job split (§5.4, §12) --------------------------------------------
JOB_CAP_SECONDS = 7200  # every GPU job (R27.24's training cap, extended here)
SPLIT_TRIGGER_SECONDS = 0.75 * JOB_CAP_SECONDS  # 5 400 s: Stage 0 splits further above this
BS_PARTS = 2  # each BS world-model run as jobs of 64 resets (Stage 0 may raise it, §12)


def part_ranges(cohort: str, parts: int) -> list[range]:
    """The contiguous reset ranges of a cohort's ``parts`` jobs (907000-907063, 907064-907127)."""
    seeds = COHORTS[cohort]
    n = len(seeds)
    if parts < 1 or parts > n:
        raise ContractError(f"{parts} parts of {n} resets")
    size = math.ceil(n / parts)
    return [seeds[i * size : min(n, (i + 1) * size)] for i in range(parts)]


def split_goals(goals: list, part: int, parts: int) -> list:
    """The goals of job ``part`` (1-based) of ``parts`` contiguous jobs, in the cohort's order."""
    if not 1 <= part <= parts:
        raise ContractError(f"part {part} of {parts}")
    n = len(goals)
    size = math.ceil(n / parts)
    return goals[(part - 1) * size : min(n, part * size)]


def parse_part(text: str) -> tuple[int, int]:
    i, k = (int(x) for x in str(text).split("/"))
    if not 1 <= i <= k:
        raise ContractError(f"bad part {text}")
    return i, k


def needs_parts(cohort: str, arm: str) -> int | None:
    """BS world-model runs must run as BS_PARTS jobs; everything else runs as one job."""
    return BS_PARTS if cohort == "bs-grasp" and arm in CLOSED_LOOP_ARMS else None


RUN_IDENTITY = (
    "task",
    "cohort",
    "arm",
    "model_seed",
    "revision",
    "goals_sha256",
    "setting",
    "cem",
    "lambda",
    "feature_reports_sha256",
    "determinism_rerun",
    "plan_record",
)


def combine_parts(parts: list[dict], expected_seeds) -> dict:
    """§5.4: the parts of one split run combined into one run before any summary. Every part must
    have completed under the cap, share the run's identity (arm, model, revision, goals, frozen
    settings) and the parts' resets must be disjoint and together exactly ``expected_seeds`` in
    order. Returns the combined run (episodes in seed order) or raises ContractError."""
    if not parts:
        raise ContractError("no parts")
    parts = sorted(parts, key=lambda p: p["part"][0])
    k = parts[0]["part"][1]
    if [p["part"] for p in parts] != [[i, k] for i in range(1, k + 1)]:
        raise ContractError(f"parts {[p['part'] for p in parts]} are not 1..{k} of {k}")
    for p in parts:
        if not p.get("completed") or p.get("cap_hit"):
            raise ContractError(f"part {p['part']} did not complete under the cap")
        for key in RUN_IDENTITY:
            if p.get(key) != parts[0].get(key):
                raise ContractError(f"part {p['part']} differs in {key}")
        if p.get("model") != parts[0].get("model"):
            raise ContractError(f"part {p['part']} ran another model")
        if p.get("tracked_tree_dirty") != parts[0].get("tracked_tree_dirty"):
            raise ContractError("parts differ in tree state")
    episodes = [e for p in parts for e in p["episodes"]]
    seeds = [e["seed"] for e in episodes]
    if seeds != list(expected_seeds):
        raise ContractError("the parts' resets are not exactly the cohort's, in order")
    out = {key: parts[0].get(key) for key in RUN_IDENTITY}
    out |= {
        "protocol": parts[0].get("protocol"),
        "tracked_tree_dirty": parts[0].get("tracked_tree_dirty"),
        "model": parts[0].get("model"),
        "workers": [p.get("workers") for p in parts],
        "episodes": episodes,
        "successes": int(sum(e["success"] for e in episodes)),
        "n": len(episodes),
        "seconds": float(sum(p["seconds"] for p in parts)),
        "parts": [
            {
                "part": p["part"],
                "seeds": [p["episodes"][0]["seed"], p["episodes"][-1]["seed"]],
                "seconds": p["seconds"],
                "started_utc": p.get("started_utc"),
                "finished_utc": p.get("finished_utc"),
            }
            for p in parts
        ],
        "completed": True,
        "cap_hit": False,
        "combined": True,
    }
    return out


def project_job_seconds(seconds_per_command: float, commands_per_reset: float, resets: int,
                        processes: int) -> float:  # fmt: skip
    """A job's projected wall time: its resets' planned commands spread over its processes."""
    return float(seconds_per_command * commands_per_reset * math.ceil(resets / processes))


def parts_needed(job_seconds_full: float, resets: int = BS_N, parts: int = BS_PARTS) -> int:
    """§12: the smallest number of equal jobs (at least ``parts``) whose projected seconds are at
    most 5 400, given the projected seconds of the unsplit ``resets``-reset run."""
    k = parts
    while job_seconds_full * math.ceil(resets / k) / resets > SPLIT_TRIGGER_SECONDS:
        k += 1
        if k > resets:
            raise ContractError("no split meets the trigger")
    return k


# ----- receding-horizon deferral (§5.3, reported only) -----------------------------------------
GRASP_PHASE = zs.SUBGOALS.index("grasp")  # 2


def _longest_run(flags) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def deferral_measures(episode: dict) -> dict:
    """§5.3's three measures for one world-model episode. ``episode`` holds ``applied`` [T, 14]
    (the executed commands), ``contact`` [T] (target grasp contact after each command, TASK-085's
    definition, read by the scorer) and ``plan_log`` (per re-plan: the subgoal index the
    controller planned for and the plan's right-grasp channel per plan step; at most one more
    entry than executed commands, when the last request was refused).

    1. closing commands before first contact: executed commands with right grasp > 0.5 among the
       commands issued while no contact had been observed (up to and including the command after
       which contact is first seen); with no contact, to the episode's end, flagged;
    2. first executed close in the grasp phase: the 1-based command index, counted from the
       grasp subgoal phase's first command, whose executed right grasp first exceeds 0.5 (None
       if it never does or the phase is never reached; flagged which);
    3. the planned close step on each grasp-phase re-plan: the first plan step k with right grasp
       > 0.5 (None if the plan never closes); the share with k = 1, the median k over plans
       that close, and the longest run of consecutive re-plans with k > 1 (a plan that never
       closes counts as k > 1)."""
    applied = np.asarray(episode.get("applied", []), np.float64).reshape(-1, 14)
    t = len(applied)
    contact = np.asarray(episode.get("contact", []), bool)[:t]
    log = list(episode.get("plan_log", []))
    if len(log) not in (t, t + 1):
        raise ContractError(f"plan log has {len(log)} entries for {t} commands")
    log = log[:t]
    closed = applied[:, GRASP_DIM] > CLOSE if t else np.zeros(0, bool)
    hits = np.flatnonzero(contact)
    first_contact = int(hits[0]) if len(hits) else None
    upto = t if first_contact is None else first_contact + 1
    requested = np.array([e["grasp"][0] > CLOSE for e in log], bool)
    phase = np.array([e["index"] for e in log], np.int64)
    in_grasp = np.flatnonzero(phase == GRASP_PHASE)
    first_close = None
    if len(in_grasp):
        g_closed = closed[in_grasp]
        if g_closed.any():
            first_close = int(np.argmax(g_closed)) + 1
    ks = [first_close_step(np.c_[np.zeros((len(e["grasp"]), GRASP_DIM)), e["grasp"]])
          for e in (log[i] for i in in_grasp)]  # fmt: skip
    closing = [k for k in ks if k is not None]
    return {
        "commands": t,
        "first_contact_command": first_contact,
        "no_contact": first_contact is None,
        "m1_closing_before_contact": int(closed[:upto].sum()),
        "m1_requested_closing_before_contact": int(requested[:upto].sum()),
        "m1_commands_before_contact": int(upto),
        "grasp_phase_commands": int(len(in_grasp)),
        "grasp_phase_reached": bool(len(in_grasp)),
        "m2_first_close_in_grasp_phase": first_close,
        "m3_replans": len(ks),
        "m3_plan_close_steps": ks,
        "m3_share_k1": (float(np.mean([k == 1 for k in ks])) if ks else None),
        "m3_median_k": (float(np.median(closing)) if closing else None),
        "m3_never_close": int(sum(k is None for k in ks)),
        "m3_longest_run_k_gt_1": _longest_run([k is None or k > 1 for k in ks]),
    }


def deferral_summary(measures: list[dict]) -> dict:
    """Per arm and seed: the distribution of §5.3's measures over resets (reported only)."""

    def dist(values):
        v = [x for x in values if x is not None]
        if not v:
            return {"n": 0, "none": len(values)}
        q = np.percentile(v, [10, 25, 50, 75, 90])
        return {
            "n": len(v),
            "none": len(values) - len(v),
            "mean": float(np.mean(v)),
            "p10_25_50_75_90": [float(x) for x in q],
            "max": float(np.max(v)),
        }

    return {
        "episodes": len(measures),
        "no_contact": int(sum(m["no_contact"] for m in measures)),
        "grasp_phase_not_reached": int(sum(not m["grasp_phase_reached"] for m in measures)),
        "m1_closing_before_contact": dist([m["m1_closing_before_contact"] for m in measures]),
        "m1_closing_before_contact_with_contact": dist(
            [m["m1_closing_before_contact"] for m in measures if not m["no_contact"]]
        ),
        "m2_first_close_in_grasp_phase": dist(
            [m["m2_first_close_in_grasp_phase"] for m in measures]
        ),
        "m2_never_closed_in_grasp_phase": int(
            sum(
                m["grasp_phase_reached"] and m["m2_first_close_in_grasp_phase"] is None
                for m in measures
            )
        ),
        "m3_share_k1": dist([m["m3_share_k1"] for m in measures]),
        "m3_share_k1_pooled": (
            float(
                sum(sum(k == 1 for k in m["m3_plan_close_steps"]) for m in measures)
                / max(1, sum(m["m3_replans"] for m in measures))
            )
            if any(m["m3_replans"] for m in measures)
            else None
        ),
        "m3_median_k": dist([m["m3_median_k"] for m in measures]),
        "m3_longest_run_k_gt_1": dist(
            [m["m3_longest_run_k_gt_1"] for m in measures if m["m3_replans"]]
        ),
    }


# ----- the B rows (§5.1, §6): TASK-090's, unchanged --------------------------------------------
b_row = gc.b_row
clause_fires = gc.clause_fires
b_bar = zg.b_bar
pooled_bar = gc.pooled_bar
seed_floor = gc.seed_floor
