"""TASK-072 M2 design (``apple_first_policy_v2_m2``): the gated test of the carried policy P-3 on
cohort C, under the owner's go-ahead of 2026-09-28.

Protocol ``docs/experiments/apple_first_policy_v2_m2.md``. M2 itself was preregistered by
TASK-071 (``apple_first_policy_v2.md`` §12, carrying ``apple_first_policy_v1.md`` §11); its gates
G1-G7, its rows and its arms are imported here unchanged from ``first_policy_v2.M2``. This module
fixes only what that section left to "a separate authorization":

- **which policy**: the checkpoints of TASK-072 run-1 on the Linux PC (P-3 carried by
  ``decide_m1``; C-3 and R-3 from the same run), identified by sha256;
- **which inputs**: the readouts are not stored by run-1, so they are refitted from the same
  frames and must reproduce run-1's recorded readout facts exactly (G-repro), or the run is V;
- **the cohort**: the 40 stored cohort-C resets of ``apple-policy-v1.json``, never recomputed
  (``stored_cohort``), under a whitelist of exactly those seeds;
- **G7's measurement** (``CONTROL_TIME``), the caps, the platform and the void rule.

NumPy only. **Learned Apple->Plate on the frozen benchmark is still 0 successes**; a pass here
would be "a learned policy with a DINOv2 encoder works on Apple->Plate on this cohort under
apple-to-plate-v2", not LeWM driving the robot, and not the v1 benchmark.
"""

from __future__ import annotations

import copy
import hashlib
import json
from math import comb

import numpy as np

from embodied_jepa import first_policy as fp
from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa import first_policy_v2_linux as fpl

PROTOCOL = "apple_first_policy_v2_m2"
TASK = "TASK-072"
STEP = "M2"
GuardError = fp2.GuardError

# ----- the frozen M2 definition, carried unchanged (TASK-071 §12; v1 §11) -----------------------
M2 = fp2.M2
COHORT_C = tuple(fp2.COHORT_C)  # 45300-45339
C_RESETS = len(COHORT_C)  # 40
CARRIED = "P-3"  # TASK-072 run-1's decide_m1 carried arm (report decision.carried)
LEARNED = ("P-3", "C-3", "R-3")  # rung L1; G6 applies to each
HARNESS = ("B-oracle", "B-hold", "B-random")
ARMS = (*LEARNED, "B-replay", *HARNESS)  # each runs each of the 40 resets exactly once
ROWS = ("M2-VOID", "M2-PASS", "M2-FAIL-VISION", "M2-FAIL")  # first match; V besides

# ----- the cohort: stored values, never recomputed ------------------------------------------------
COHORT_SOURCE = {
    "manifest": "benchmarks/manifests/apple-policy-v1.json",
    "key": ["cohorts", "C_frozen_gating", "resets"],
    "cohort_sha256": "4f888154c055bcbbbc0df333442e6886d8f65c2389596fe1aedde1d2cb0b533e",
    "digest": "sha256 of json.dumps(resets, sort_keys=True, separators=(',', ':'))",
}


def cohort_digest(resets: dict) -> str:
    blob = json.dumps(resets, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def stored_cohort(policy_v1_manifest: dict) -> dict[int, dict]:
    """The 40 cohort-C resets exactly as stored: ``{seed: {"object_xy", "plate_xy"}}``.

    The digest is taken over the stored decimals before anything is converted; the values are
    then passed on as the same Python floats, never recomputed from ``wide_reset``."""
    node = policy_v1_manifest
    for key in COHORT_SOURCE["key"]:
        node = node[key]
    if cohort_digest(node) != COHORT_SOURCE["cohort_sha256"]:
        raise GuardError("G-cohort: the stored cohort-C values differ from their sealed digest")
    if sorted(node) != [str(s) for s in COHORT_C]:
        raise GuardError("G-cohort: the stored cohort is not exactly seeds 45300-45339")
    out = {}
    for seed in COHORT_C:
        entry = node[str(seed)]
        if entry["seed"] != seed or set(entry) != {"seed", "object_xy", "plate_xy"}:
            raise GuardError(f"G-cohort: stored entry {seed} is malformed")
        out[seed] = {"object_xy": list(entry["object_xy"]), "plate_xy": list(entry["plate_xy"])}
    return out


def check_cohort_seeds(seeds) -> tuple[int, ...]:
    """G-seeds for M2: a whitelist of exactly cohort C, each seed a plain int, in cohort order.

    A non-int seed (for example the string "45300") is refused before any coercion."""
    seeds = tuple(seeds)
    if any(type(s) is not int for s in seeds):
        raise GuardError("G-seeds: M2 seeds must be plain ints")
    if seeds != COHORT_C:
        raise GuardError("G-seeds: M2 simulates exactly cohort C (45300-45339), once, in order")
    return seeds


# ----- the policy and its evidence: TASK-072 run-1 on the Linux PC --------------------------------
EVIDENCE = {
    "results_manifest": "benchmarks/manifests/apple-first-policy-v2-linux-results.json",
    "report": "outputs/task072-first-policy-v2-linux/run-1/report.json",
    "report_sha256": "877452840d0d03520ec8644321ef32f6b19877caa679e77d173e3d2509edec78",
    "revision": "db65816eb56dadb73664f08978cadeb6e881b35f",
    "outcome": "M1-PASS",
    "carried": CARRIED,
    "checkpoints": {
        "P-3": "7988162df0b6106db5d007e322b824a3725b74cb103ba5a8de9f552912060be8",
        "C-3": "d1820db4a1af3a63f02ed96c9a7c51d8b92d56ea7a23cf762940925f586b1289",
        "R-3": "bea0d422b951287fcad66630aa8ea837a59df441d7a6c1793e2423048e2c0fd3",
    },
    "checkpoint_dir": "checkpoints/task072-first-policy-v2-linux/run-1",
    "corpus": "data/apple-look-v2-linux/run-1",
    "corpus_manifest_sha256": "67c342f6d37f3bd1910af16139096c109fa0bf6647d87286ca26b0899c9b4f54",
    "b_replay_library_roots": 112,
}
STOP_RULE = M2["stop_rule"]


def stop_rule_runs_carried(m1_counts: dict) -> bool:
    """TASK-071 §12's stop rule: the carried P-k runs on C only if it had >= 1 grasp on D2."""
    return int(m1_counts[CARRIED]["grasp"]) >= 1


# ----- G-repro: the refitted readouts must be run-1's ---------------------------------------------
REPRODUCTION = (
    "the P and R readouts are refitted, as run-1 fitted them, on the same 426 rows: frame 0 of the "
    "170 sealed train roots of run-1's corpus plus the 256 perception-train post-look frames "
    "(51200-51455), re-rendered here",
    "each readout's selection (family, lam_rel, inner_mse) equals run-1's report exactly",
    "the P readout's S0-P errors on the 128 re-rendered held-out frames (51456-51583) equal "
    "run-1's per-reset apple and plate errors exactly (float equality after the JSON round trip)",
    "C-3's constant estimate, the mean of the cross-fitted P estimates over the train roots, "
    "equals run-1's C_mean_estimates exactly",
    "B-replay's retrieval, re-run on the 16 re-rendered D2 post-look frames (52000-52015), picks "
    "run-1's b_replay_nearest roots exactly, from the same 112-root library",
    "a mismatch in any of these is V (G-repro): nothing on cohort C is simulated, and the owner "
    "decides",
)

# ----- G7: how control time is measured -----------------------------------------------------------
CONTROL_TIME = {
    "definition": (
        "per command of the carried P-k's 40 attempts: the wall time (time.perf_counter) of the "
        "controller's act() in its worker (input assembly, palm FK, standardising, the MLP "
        "forward and command assembly; 1 torch thread); the first command of each attempt also "
        "carries that attempt's one DINOv2 forward pass plus the readout, timed in the main "
        "process at batch size 1 (6 torch threads, CPU). G7 compares the median over all of "
        "these commands with 0.100 s"
    ),
    "reported_beside": (
        "p90 and max per command; the DINOv2 forward + readout time alone (median, max); and a "
        "conservative variant with the forward pass added to every command (descriptive only)"
    ),
    "max_median_seconds": M2["G7_max_median_control_seconds"],
}


# ----- the decision (first matching row) ----------------------------------------------------------
def mcnemar_exact(only_first: int, only_second: int) -> float:
    """Exact two-sided McNemar p-value from the discordant pairs (binomial, p = 0.5)."""
    n = int(only_first) + int(only_second)
    if n == 0:
        return 1.0
    k = min(int(only_first), int(only_second))
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2.0 * tail))


def paired(first, second) -> dict:
    """Discordant pairs of two arms' per-reset successes on the identical resets."""
    a, b = np.asarray(first, bool), np.asarray(second, bool)
    if a.shape != b.shape:
        raise GuardError("paired arms need the same resets")
    only_first, only_second = int((a & ~b).sum()), int((~a & b).sum())
    return {
        "only_first": only_first,
        "only_second": only_second,
        "n_d": only_first + only_second,
        "p_exact_mcnemar": mcnemar_exact(only_first, only_second),
    }


def _vector(values, arm: str) -> np.ndarray:
    v = np.asarray(values, bool)
    if v.shape != (C_RESETS,):
        raise GuardError(f"{arm} needs one result per cohort-C reset")
    return v


def decide_m2(
    success: dict,
    grasp: dict,
    privileged_ok: dict,
    median_control_seconds: float | None,
) -> dict:
    """First-matching M2 row from per-reset counted successes and grasps (40 each per arm).

    ``success`` is the counted success of T71-R1/R2; an arm that did not run (the carried P-k
    under the stop rule) is ``None``, and every gate needing it is failed ("a gate that cannot be
    evaluated counts as failed"). ``privileged_ok`` maps each L1 arm to G-privileged's verdict."""
    s = {arm: None if success.get(arm) is None else _vector(success[arm], arm) for arm in ARMS}
    g = {arm: None if grasp.get(arm) is None else _vector(grasp[arm], arm) for arm in ARMS}
    for arm in ARMS:
        if arm != CARRIED and (s[arm] is None or g[arm] is None):
            raise GuardError(f"M2 needs {arm}'s results: the controls always run")
        if s[arm] is not None and bool((s[arm] & ~g[arm]).any()):
            raise GuardError(f"{arm}: a counted success without a grasp")
    n = {arm: None if s[arm] is None else int(s[arm].sum()) for arm in ARMS}
    p = n[CARRIED]
    gates = {
        "G1": p is not None and p >= M2["G1_min_successes"] and p > n["B-replay"],
        "G2": p is not None and p - n["C-3"] >= M2["G2_min_difference_vs_C"],
        "G3": p is not None and p - n["R-3"] >= M2["G3_min_difference_vs_R"],
        "G4": g[CARRIED] is not None and int(g[CARRIED].sum()) >= M2["G4_min_grasps"],
        "G5": int(g["B-hold"].sum()) == M2["G5_harness"]["B-hold_grasp"]
        and int(g["B-random"].sum()) == M2["G5_harness"]["B-random_grasp"]
        and n["B-oracle"] >= M2["G5_harness"]["B-oracle_min_successes"],
        "G6": all(bool(privileged_ok.get(arm, False)) for arm in LEARNED if s[arm] is not None),
        "G7": median_control_seconds is not None
        and bool(np.isfinite(median_control_seconds))
        and median_control_seconds <= M2["G7_max_median_control_seconds"],
    }
    pairs = {}
    if s[CARRIED] is not None:
        for other in ("C-3", "R-3", "B-replay"):
            pairs[f"{CARRIED} vs {other}"] = paired(s[CARRIED], s[other])
    if not gates["G5"] or not gates["G6"]:
        row = "M2-VOID"
    elif all(gates.values()):
        row = "M2-PASS"
    elif gates["G1"] and gates["G4"] and not gates["G2"]:
        row = "M2-FAIL-VISION"
    else:
        row = "M2-FAIL"
    return {
        "row": row,
        "gates": gates,
        "success_of_40": n,
        "grasp_of_40": {arm: None if g[arm] is None else int(g[arm].sum()) for arm in ARMS},
        "paired": pairs,
        "median_control_seconds": median_control_seconds,
        "carried_ran": s[CARRIED] is not None,
    }


# ----- platform, caps, void rule ------------------------------------------------------------------
PLATFORM = fpl.PLATFORM  # the Linux PC, NVIDIA EGL, as run-1
SIM_WORKERS = fpl.SIM_WORKERS  # 16, as run-1
DETERMINISM = {
    "module": "embodied_jepa.devices",
    "strict": True,
    "why": "the main process is put in run-1's strict deterministic state before the readouts are "
    "refitted, so the refit runs under the same process-wide settings (no CUDA work is done)",
}
CAPS_SECONDS = {
    "global": 7_200.0,
    "perception_collection": fp2.CAPS_SECONDS["perception_collection"],
    "per_rollout_batch": fp2.CAPS_SECONDS["per_rollout_batch"],
    "per_attempt": fp2.CAPS_SECONDS["per_attempt"],
}
VOID_RULE = (
    "a guard (G-hash, G-frozen, G-platform, G-device, G-weights, G-evidence, G-repro, G-cohort, "
    "G-seeds, G-look, G-frame, G-finite, G-cap, Q-split), a crash or a cap makes the run V: "
    "nothing is read. A V before the first cohort-C frame is rendered may be repeated after a "
    "reviewed fix. A V after that point is reported to the owner, and a repeat (run-2, cohort C "
    "simulated again) needs the owner's ruling; a second V closes M2 as INCONCLUSIVE. M2-VOID "
    "(G5 or G6) is an outcome, not a V: the run is invalid, not the arms, and it is not repeated "
    "without the owner's ruling. Nothing is re-thresholded, retrained or re-selected"
)
EXEMPTION = M2["exemption_spent_cited"]


def frozen_block() -> dict:
    """Everything this M2 step freezes, as plain JSON types; the manifest's ``frozen`` equals it."""
    return fp._plain(
        {
            "protocol": PROTOCOL,
            "task": TASK,
            "step": STEP,
            "carried_from": [fp2.PROTOCOL, fpl.PROTOCOL],
            "m2": copy.deepcopy(M2),
            "success": fp2.COUNTED_SUCCESS,
            "cohort": {"seeds": COHORT_C, "source": COHORT_SOURCE},
            "arms": ARMS,
            "learned_arms": LEARNED,
            "carried": CARRIED,
            "rows": ROWS,
            "evidence": EVIDENCE,
            "reproduction": REPRODUCTION,
            "control_time": CONTROL_TIME,
            "attempt": {
                "max_policy_steps": fp2.MAX_POLICY_STEPS,
                "settle_steps": fp2.SETTLE_STEPS,
                "random_controller_seed": fp2.RANDOM_CONTROLLER_SEED,
            },
            "platform": PLATFORM,
            "sim_workers": SIM_WORKERS,
            "determinism": DETERMINISM,
            "caps_seconds": CAPS_SECONDS,
            "void_rule": VOID_RULE,
        }
    )
