"""TASK-075 (``apple_obs_ceiling_v2``): measure the readout ceiling of each observation option
before any further world-model run.

Protocol ``docs/experiments/apple_obs_ceiling_v2.md``. This module fixes in code, before any
corpus root is re-rendered, what the protocol preregisters:
- the seeds and their guard (the tau calibration's development seeds, the smoke seeds);
- the place tolerance tau, measured on development seeds before the freeze (``TAU_MEASURED``);
- the four views and the steps rendered in each;
- the readouts, their folds, the learning curve and the statistics;
- the admission rule (c_V <= tau, beat the random-init floor and the clock prior, a clean
  plate-hidden check), the rows, their consequences and the abandonment clause;
- the downstream rule for the next world-model task's O2-type bar (c_V <= bar <= tau);
- the void rule, the memory, GPU and disk guards.

NumPy only: it imports without torch or MuJoCo.

What is carried unchanged from TASK-074 (``lewm_planner_v2``):
- the v2 scene, e9 and its schedule, P-3 and the counted success;
- the condition: a -y plate move of 9 cm at step 300 (K1-PASS), the direction rule and the
  reset re-draw rule;
- the corpus ``apple-far-shift-v2`` (sealed, manifest ``fe7ab915...``): its 240 train and 30 val
  roots are re-simulated to render the new views; its 30 test roots are never simulated,
  rendered or decoded;
- R_off's estimator (pooled 4 x 4 DINOv2 tokens -> apple-minus-plate offset; ridge, lambda grid,
  standardisation, grouped inner CV) and O2's moving windows.

**This is a measurement study.** It makes no control claim and changes no success count.
Learned Apple->Plate on the frozen benchmark is still 0 successes, and nothing here makes LeWM,
planning or behaviour cloning the project's control approach.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from embodied_jepa import first_policy as fp
from embodied_jepa import lewm_planner_v2 as lp
from embodied_jepa.contracts import ContractError

PROTOCOL = "apple_obs_ceiling_v2"
TASK = "TASK-075"
GuardError = lp.GuardError

# ----- the owner's delegation and what was decided under it ---------------------------------------
DELEGATED = "decided by Claude under owner delegation (2026-09-30), ruled 2026-10-01"
OWNER_DECISIONS = {
    "owner_delegation_verbatim": lp.OWNER_DECISIONS["owner_delegation_verbatim"],
    "owner_delegation_date": lp.OWNER_DECISIONS["owner_delegation_date"],
    "next_task_choice": "candidate (d) of the design proposal: measure the observation and "
    "readout ceilings first; views onboard 112 (reference), onboard native 224, the palm-centred "
    "hand crop and overview 224; candidate (c)'s learning curve as one factor; real-robot "
    "pretraining (b) deferred",
    "next_task_choice_by": DELEGATED,
    "lesson_carried": "TASK-074 (results §5): a bar on a readout must be calibrated from that "
    "readout's measured ceiling before the freeze; every bar here is a measured quantity",
}
DELEGATED_CHOICES = {
    "tau_condition": "tau is measured under TASK-074's condition (a -y move of 9 cm at step "
    "300, K1-PASS) with K1's H-handover arm (P-3's pick, e9's place primitive aimed at the true "
    "moved plate) displaced by a planted error",
    "tau_levels": "planted errors 0, 0.5, 1, 1.5, 2, 2.5, 3, 4 and 5 cm on 32 development resets; "
    "one direction per reset, the same at every level",
    "tau_rule": "tau is the largest level L such that every level <= L reaches >= 28/32 counted "
    "successes (the TASK-070/K1 bar family)",
    "corpus_reuse": "the reference view is apple-far-shift-v2's stored 112 px frames; the new "
    "views are rendered by re-simulating its 240 train and 30 val roots, checked against the "
    "stored arrays; the 30 test roots are excluded",
    "cross_fitting": "every readout is cross-fitted over the 270 read roots (5 outer folds by "
    "root), so c_V rests on about 200 roots instead of TASK-074's 22 val roots",
    "admission_floor_and_prior": "beating the random-init floor and the clock prior means the "
    "upper 95 % bound of the paired cluster-median error ratio is below 1.0 (strict superiority, "
    "no tuned margin); the proposal's 'margin calibrated from the reference view's development "
    "spread' is not used, so that no view's readout is seen before the freeze (deviation X-2)",
    "clock_prior": "the clock prior is the per-step median offset of the outer-training roots "
    "(cross-fitted, reads no image); TASK-074's o*(t+16) prior is reported beside it",
    "spurious_check": "the plate-hidden check gates: R(V) on the t+16 frame re-rendered with the "
    "plate hidden must NOT read the offset within tau (lower 95 % bound of its median error > "
    "tau); the apple-hidden check is reported only, because in the carry phase the apple sits in "
    "the hand, whose pose is a legitimate kinematic cue",
    "representation_row": "OBS-REPRESENTATION needs a view that passes with the full-token or "
    "raw-pixel readout (c <= tau, beats the clock prior, clean plate-hidden check) while no view "
    "is admitted with R_off",
    "view_order": "views are ordered by the size of the change to the robot: onboard 112 (none), "
    "onboard 224 (resolution), hand crop (resolution and a kinematic crop), overview 224 (an "
    "external camera); the first admitted onboard view is chosen",
    "wrist_view": "no wrist view: the scene defines none, and a scene change needs its own review",
    "learning_curve": "R_off at 25, 50 and 100 % of each outer fold's training roots; 'still "
    "falling' if the lower 95 % bound of median(e_50) - median(e_100) is > 0; it informs only the "
    "next step under OBS-NONE",
}

# ----- seeds --------------------------------------------------------------------------------------
# The repository search of 2026-10-01 (src, scripts, docs, benchmarks, tests, configs, .mc) found
# no seed use in 55000-55999 (only non-seed numbers elsewhere in 50000-59999, as TASK-068's search
# of 2026-09-28 recorded) and no use of 7501-7505 as a seed constant.
SEED_RANGES = {"tau": (55000, 55031)}  # development: spent by the Stage-0 calibration
SMOKE_SEEDS = (55050, 55099)  # smokes only; nothing from them is read
TASK_BLOCK = (55000, 55999)
FORBIDDEN_RANGES = dict(lp.FORBIDDEN_RANGES) | {"task074_block": lp.TASK_BLOCK}
SEEDS = {
    "tau_direction": 7501,
    "outer_folds": 7502,
    "inner_folds": 7503,
    "learning_curve": 7504,
    "bootstrap": 7505,
}


def seeds_of(role: str) -> tuple[int, ...]:
    low, high = SEED_RANGES[role]
    return tuple(range(low, high + 1))


def check_seed_ranges() -> None:
    spans = [*SEED_RANGES.values(), SMOKE_SEEDS]
    for i, (a, b) in enumerate(spans):
        if not (TASK_BLOCK[0] <= a <= b <= TASK_BLOCK[1]):
            raise GuardError(f"G-seeds: {a}-{b} is outside the task block")
        for c, d in spans[i + 1 :]:
            if a <= d and c <= b:
                raise GuardError("G-seeds: two ranges overlap")
        for name, (c, d) in FORBIDDEN_RANGES.items():
            if a <= d and c <= b:
                raise GuardError(f"G-seeds: {a}-{b} overlaps the forbidden range {name}")
    if len(set(SEEDS.values())) != len(SEEDS):
        raise GuardError("G-seeds: two RNG salts are equal")


def check_role_seeds(role: str, seeds, *, smoke: bool = False) -> tuple[int, ...]:
    """G-seeds: plain ints; exactly the role's seeds in order, or smoke seeds only in a smoke."""
    seeds = tuple(seeds)
    if any(type(s) is not int for s in seeds):
        raise GuardError("G-seeds: seeds must be plain ints")
    for s in seeds:
        for name, (low, high) in FORBIDDEN_RANGES.items():
            if low <= s <= high:
                raise GuardError(f"G-seeds: seed {s} lies in the forbidden range {name}")
    if smoke:
        low, high = SMOKE_SEEDS
        if any(not low <= s <= high for s in seeds) or len(set(seeds)) != len(seeds):
            raise GuardError(f"G-seeds: a smoke simulates smoke seeds {low}-{high} only")
        return seeds
    if role not in SEED_RANGES:
        raise GuardError(f"G-seeds: unknown role {role!r}")
    if seeds != seeds_of(role):
        raise GuardError(f"G-seeds: {role} simulates exactly {SEED_RANGES[role]}, once, in order")
    return seeds


# ----- carried from TASK-074 ----------------------------------------------------------------------
SCENE_VERSION = lp.SCENE_VERSION
COUNTED_SUCCESS = lp.COUNTED_SUCCESS
EXPERT = lp.EXPERT
CARRIED = lp.CARRIED  # P-3
EVIDENCE = lp.EVIDENCE
CONDITION = {
    "shift_step": lp.SHIFT_STEP,  # 300
    "shift_cm": 9,  # TASK-074 K1-PASS chose 9 cm (results §1.1)
    "direction_arc_deg": lp.DIRECTION_ARC_DEG,
    "reset": "lewm_planner_v2.condition_reset (the reset re-draw rule)",
    "shift_vector": "lewm_planner_v2.shift_vector(seed, reset, 9)",
    "k1_report_sha256": "25701293c54fd585274d7734018de7485d65f92915060ed7aa034439db8aad88",
}
SOURCE_CORPUS = {
    "name": lp.CORPUS,  # apple-far-shift-v2
    "manifest_sha256": "fe7ab9150216887f5759521dcc7fe7c7f54d2e5e8c36fecf6ed713a000b134bd",
    "plan_sha256": "41c505719a0012bd0d0dd453b39cb5230753739cef3252caa2176f334134c0b0",
    "protocol": lp.PROTOCOL,
    "frozen_sha256": "2cf80f5aa54d509e3801bcb3934409407da9b6a0600e6856f319ddeda1d2e36a",
    "revision": "9d9b03cff44848f0b43b63f9d1f560a378c672ba",
    "read_splits": ("train", "val"),
    "split_sizes": {"train": 240, "val": 30, "test": 30},
    "excluded_split": "test",
    "excluded_test_ids_sha256": "4fa7512c06a85b72fe1fe82e2538398e3c2cf55c0e9497c2116f9728304357bf",
    "excluded_rule": "the 30 test roots are never simulated, rendered or decoded "
    "(CorpusReader refuses them; the digest is sha256 of the sorted id list's JSON)",
}
# TASK-074's train stage (runs 1-3, byte-identical): R_off fitted on the train roots, read on O2's
# moving windows of the val roots (protocol A1.5, results §1.5). G-repro-off reproduces them.
TASK074_REFERENCE = {
    "encoded_median_cm": 2.8722625765232763,
    "persistence_median_cm": 4.350821813169937,
    "clock_prior_median_cm": 3.1225649505050876,
    "windows": 90,
    "roots": 22,
    "r_off_readout_sha256": "260af3f7567aadd03391edc96bd71132ce8b1fa4f791906cf6f596bbfd803157",
    "r_off_rows": 5175,
    "r_off_groups": 225,
    "baselines_json_sha256": "ff3a2e7bc5b5b9d4e625c1a969643b79c62269b6c2ce3dbc7484b2ce5ff4d07e",
    "run": "task074-run3/outputs/task074-train/run-3 (5e53ef3)",
}

# ----- the views ----------------------------------------------------------------------------------
REFERENCE_VIEW = "onboard112"
VIEWS = ("onboard112", "onboard224", "handcrop", "overview224")  # the order of change (rows)
VIEW_SPECS = {
    "onboard112": {
        "camera": "onboard_rgb",
        "render_px": 112,
        "frame_px": 112,
        "class": "onboard",
        "change": "none: apple-far-shift-v2's stored frames (the reference)",
    },
    "onboard224": {
        "camera": "onboard_rgb",
        "render_px": 224,
        "frame_px": 224,
        "class": "onboard",
        "change": "resolution: the same head camera rendered natively at 224 px",
    },
    "handcrop": {
        "camera": "onboard_rgb",
        "render_px": 320,
        "frame_px": 112,
        "class": "onboard",
        "change": "resolution and a crop: a 112 px window of a 320 px onboard render, centred on "
        "the projected right palm from robot kinematics only (hand_crop.HandCrop, TASK-048)",
    },
    "overview224": {
        "camera": "overview",
        "render_px": 224,
        "frame_px": 224,
        "class": "extra",
        "change": "an external camera (a hardware change; PRD.md:208)",
    },
}
ONBOARD_VIEWS = tuple(v for v in VIEWS if VIEW_SPECS[v]["class"] == "onboard")
EXTRA_VIEWS = tuple(v for v in VIEWS if VIEW_SPECS[v]["class"] == "extra")
RENDERED_VIEWS = tuple(v for v in VIEWS if v != REFERENCE_VIEW)
HIDDEN_KINDS = ("apple_hidden", "plate_hidden")
HIDDEN_GROUP = 5  # a geom group no scene geom uses (scripts/probe_info_ceiling.py:68)
RENDER_RULE = {
    "warm_up": "every extra renderer renders once, discarded, at the start of every root "
    "(apple_observation_reprobe_v1_results.md:271-272)",
    "hidden": "a hidden render moves every geom of the hidden body to HIDDEN_GROUP and turns "
    "that group off (not drawn, no shadow), then restores the groups; physics is untouched",
    "reference_check": "the re-simulated root's states, apple, plate, applied, base and phase "
    "arrays must equal the stored ones exactly, and its 112 px frames at the rendered steps must "
    "equal the stored frames or differ by at most one level in at most 16 pixels "
    "(lewm_planner_v2.RENDER); anything else is a V (G-repro-render)",
    "max_level_difference": lp.RENDER["max_level_difference"],
    "max_pixels": lp.RENDER["max_pixels"],
}

# ----- the steps ----------------------------------------------------------------------------------
CHUNK = lp.CHUNK  # 16
FEATURE_BAND = lp.FEATURE_BAND  # (384, 560)
TRAIN_STEPS = tuple(range(FEATURE_BAND[0], FEATURE_BAND[1] + 1, lp.READOUTS["r_off_stride"]))
DECISION_STEPS = lp.DECISION_STEPS  # 405, 421, ..., 485
EVAL_STEPS = tuple(t + CHUNK for t in DECISION_STEPS)  # 421, ..., 501
RENDER_STEPS = tuple(sorted(set(TRAIN_STEPS) | set(DECISION_STEPS) | set(EVAL_STEPS)))
HIDDEN_STEPS = EVAL_STEPS
MOVING_THRESHOLD_M = lp.O2["moving_threshold_m"]  # 0.01
ROOT_MIN_LENGTH = FEATURE_BAND[0] + CHUNK + 1  # lewm_planner_v2 featurisation's rule

# ----- the place tolerance tau (Stage 0, development) ---------------------------------------------
TAU = {
    "seeds": SEED_RANGES["tau"],
    "resets": 32,
    "levels_cm": (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0),
    "bar": 28,
    "arm": "H-handover with a planted error: P-3's pick, then e9's place primitive "
    "(place_planner.PlannerController, mode 'truth') aimed at the true moved plate plus e, at "
    "every decision step (lewm_planner_v2_runtime's H-handover path, unchanged)",
    "condition": "TASK-074's: the plate moves 9 cm to the robot's right (-y) at step 300",
    "direction": "one angle per reset, uniform on [0, 2 pi): default_rng(SeedSequence("
    "[SEEDS['tau_direction'], seed])); the same angle at every level",
    "count": "counted success (T71-R1/R2: at rest after a latched grasp and place)",
    "rule": "tau = the largest level L such that every level <= L reaches >= 28/32 counted "
    "successes; if level 0 fails, TAU-NONE (the place itself is the constraint)",
    "when": "Stage 0, before the freeze, on development seeds that are then spent",
}


def tau_direction(seed: int) -> float:
    rng = np.random.default_rng(np.random.SeedSequence([SEEDS["tau_direction"], int(seed)]))
    return float(2.0 * np.pi * rng.uniform())


def planted_error_m(seed: int, level_cm: float) -> list[float]:
    angle = tau_direction(seed)
    r = float(level_cm) / 100.0
    return [r * float(np.cos(angle)), r * float(np.sin(angle))]


def tau_from_counts(counts: dict) -> dict:
    """``counts``: level (cm) -> counted successes of 32. tau is the largest level L such that
    every level <= L reaches the bar; ``None`` (TAU-NONE) if the smallest level fails."""
    by_level = {float(k): int(v) for k, v in counts.items()}
    if sorted(by_level) != sorted(TAU["levels_cm"]):
        raise ContractError("tau needs every preregistered level")
    tau = None
    for level in sorted(by_level):
        if by_level[level] >= TAU["bar"]:
            tau = level
        else:
            break
    return {"tau_cm": tau, "row": "TAU-MEASURED" if tau is not None else "TAU-NONE"}


# Written after the Stage-0 calibration ran (protocol §2); the frozen block carries it. The run:
# ``run_obs_ceiling_v2.py tau`` at 26c64d9 (clean tree; frozen sha de3218c7... before this record
# was added), outputs/task075-tau/run-1/report.json on the Linux PC.
TAU_MEASURED: dict | None = {
    "tau_cm": 1.0,
    "row": "TAU-MEASURED",
    "counts": {
        "0.0": 31,
        "0.5": 31,
        "1.0": 28,
        "1.5": 22,
        "2.0": 23,
        "2.5": 21,
        "3.0": 15,
        "4.0": 11,
        "5.0": 8,
    },
    "at_rest": {
        "0.0": 31,
        "0.5": 31,
        "1.0": 28,
        "1.5": 22,
        "2.0": 23,
        "2.5": 21,
        "3.0": 15,
        "4.0": 11,
        "5.0": 13,
    },
    "seeds": [55000, 55031],
    "report_sha256": "08a152f14e63902e7e045f8890a000c0eedd0d7ac751af6bd879e6e512c519d3",
    "revision": "26c64d906b2777734e29021433d903080ec1b13c",
    "frozen_sha256_at_run": "de3218c787c7162069e7252dfde8ca13c59a2ef9c2f3d539fe2c47cc156caf42",
    "started_utc": "2026-10-01T12:35:57Z",
    "ended_utc": "2026-10-01T12:40:22Z",
    "load_average_at_start": [0.04638671875, 1.42724609375],
    "planted_check": "every decision aimed at plate + e (max |difference| 0.0 m); no fallback; "
    "no blocked move",
    "level_0_failure": "seed 55028: P-3's grasp failed (no latched grasp)",
}

# ----- readouts -----------------------------------------------------------------------------------
OUTER_FOLDS = 5
INNER_FOLDS = 5
LEARNING_CURVE = (0.25, 0.5, 1.0)
BOOTSTRAP_RESAMPLES = 10_000
KERNEL_FAMILIES = ("linear", "rbf")
KERNEL_LAMBDAS = tuple(10.0**k for k in range(-4, 5))  # info_ceiling.LAMBDAS
READOUTS = {
    "R_off": "RidgeReadout (latent_critic.py:42) from the pooled 4 x 4 pretrained DINOv2 "
    "patch tokens (6144-d, standardised on the fit rows) to the apple-minus-plate offset xy (m); "
    "lambda from lewm_planner_v2.READOUTS' grid by grouped inner CV; solved in the dual form, "
    "which RidgeReadout defines to give the same readout as the primal (latent_critic.py:43-45). "
    "The admission readout",
    "R_floor": "R_off on the seed-0 random-init DINOv2's pooled tokens (pretrained_encoder."
    "random_init): the floor",
    "R_full": "kernel ridge (info_ceiling's family: linear or RBF on the raw Gram, centred on "
    "the fit rows, lambda in 1e-4..1e4) on the full pretrained tokens (98 304-d); family and "
    "lambda by grouped inner CV",
    "R_pix": "the same kernel ridge on the raw uint8 pixels of the view (the information "
    "ceiling, TASK-061's family)",
    "R_proprio": "RidgeReadout (primal) from the robot state (the stored 'states' row) to the "
    "offset: reported only, a kinematic baseline",
    "clock_prior": "the per-step median offset of the outer-training roots at t + 16 (reads no "
    "image); TASK-074's o*(t + 16) is reported beside it",
    "persistence": "R_off on the frame at t, against the offset at t + 16 (reported)",
    "targets": "the offset (gating); plate xy and apple xy through R_off's estimator (reported)",
    "fit_rows": "the outer-training roots' frames at TRAIN_STEPS (every 8th band frame from 384, "
    "as TASK-074's R_off)",
    "eval_windows": "every read root's windows at the decision steps t whose true offset moves "
    "by >= 1 cm over the 16 commands (O2's moving cohort); the readout reads frame t + 16",
    "outer_folds": "5, by root: a seeded permutation of the read roots in plan order, position "
    "mod 5 (SEEDS['outer_folds'])",
    "inner_folds": "5, grouped by root (latent_critic.grouped_folds, SEEDS['inner_folds'])",
    "lambda_grid_relative": lp.READOUTS["lambda_grid_relative"],
    "kernel_lambdas": KERNEL_LAMBDAS,
    "kernel_families": KERNEL_FAMILIES,
    "learning_curve": "R_off at 25 and 50 % of each outer fold's training roots (a seeded "
    "subsample, SEEDS['learning_curve']) beside 100 %",
    "device": "DINOv2 on CUDA (strict determinism), batch 64, with a CPU anchor check per view; "
    "the fits on the CPU",
}


def outer_folds(root_ids) -> dict:
    order = np.random.default_rng(SEEDS["outer_folds"]).permutation(len(root_ids))
    fold = np.empty(len(root_ids), np.int64)
    fold[order] = np.arange(len(root_ids)) % OUTER_FOLDS
    return {r: int(f) for r, f in zip(root_ids, fold, strict=True)}


def learning_subsample(train_roots, fraction: float, outer: int) -> list:
    """A seeded subsample of an outer fold's training roots (sorted input order kept)."""
    roots = list(train_roots)
    if fraction >= 1.0:
        return roots
    take = max(INNER_FOLDS, int(round(fraction * len(roots))))
    rng = np.random.default_rng(np.random.SeedSequence([SEEDS["learning_curve"], int(outer)]))
    keep = set(rng.permutation(len(roots))[:take].tolist())
    return [r for i, r in enumerate(roots) if i in keep]


# ----- statistics ---------------------------------------------------------------------------------
def _groups(clusters):
    labels, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    return [np.flatnonzero(inverse == i) for i in range(len(labels))]


def _boot(statistic, groups, resamples, seed):
    rng = np.random.default_rng(seed)
    out = np.empty(resamples)
    for b in range(resamples):
        out[b] = statistic(
            np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        )
    return out


def cluster_median_ci(values, clusters, *, resamples: int | None = None) -> dict:
    """The median with a root-clustered bootstrap 95 % percentile interval."""
    v = np.asarray(values, np.float64)
    boot = _boot(
        lambda i: np.median(v[i]),
        _groups(clusters),
        resamples or BOOTSTRAP_RESAMPLES,
        SEEDS["bootstrap"],
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "median": float(np.median(v)),
        "ci95": [float(lo), float(hi)],
        "n": int(len(v)),
        "clusters": int(len(_groups(clusters))),
    }


def cluster_median_ratio(num, den, clusters, *, resamples: int | None = None) -> dict:
    """median(num) / median(den) over the same windows, root-clustered bootstrap 95 % interval."""
    a, b = np.asarray(num, np.float64), np.asarray(den, np.float64)

    def ratio(i):
        d = np.median(b[i])
        return np.median(a[i]) / d if d > 0 else np.finfo(np.float64).max

    boot = _boot(ratio, _groups(clusters), resamples or BOOTSTRAP_RESAMPLES, SEEDS["bootstrap"])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    base = float(np.median(b))
    return {
        "ratio": float(np.median(a) / base) if base > 0 else None,
        "ci95": [float(lo), float(hi)],
        "n": int(len(a)),
    }


def cluster_median_difference(first, second, clusters, *, resamples: int | None = None) -> dict:
    """median(first) - median(second) over the same windows, clustered bootstrap 95 % interval."""
    a, b = np.asarray(first, np.float64), np.asarray(second, np.float64)
    boot = _boot(
        lambda i: np.median(a[i]) - np.median(b[i]),
        _groups(clusters),
        resamples or BOOTSTRAP_RESAMPLES,
        SEEDS["bootstrap"],
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "difference": float(np.median(a) - np.median(b)),
        "ci95": [float(lo), float(hi)],
        "n": int(len(a)),
    }


def cluster_mean_ci(values, clusters, *, resamples: int | None = None) -> dict:
    """The mean with a root-clustered bootstrap 95 % percentile interval."""
    v = np.asarray(values, np.float64)
    boot = _boot(
        lambda i: np.mean(v[i]),
        _groups(clusters),
        resamples or BOOTSTRAP_RESAMPLES,
        SEEDS["bootstrap"],
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(np.mean(v)), "ci95": [float(lo), float(hi)], "n": int(len(v))}


def cluster_quantile_ci(values, clusters, q: float, *, resamples: int | None = None) -> dict:
    """The q-th percentile with a root-clustered bootstrap 95 % percentile interval."""
    v = np.asarray(values, np.float64)
    boot = _boot(
        lambda i: np.percentile(v[i], q),
        _groups(clusters),
        resamples or BOOTSTRAP_RESAMPLES,
        SEEDS["bootstrap"],
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "quantile": float(q),
        "value": float(np.percentile(v, q)),
        "ci95": [float(lo), float(hi)],
        "n": int(len(v)),
    }


# ----- reported-only statistics (fixed before any view number exists; they gate nothing) ----------
REPORTED_PERCENTILE = 87.5
SENSITIVITY_TAU_CM = (0.5, 1.5)
REPORTED = {
    "percentile": "the 87.5th percentile of R_off(V)'s window errors (cm), with a root-clustered "
    "interval: A1 bounds the median, so up to half the windows may err by more than tau",
    "tau_curve_prediction": "each window's R_off(V) error e (cm) is mapped to the measured tau "
    "curve's counted-success fraction (TAU_MEASURED['counts'] / 32) by linear interpolation "
    "between the levels 0, 0.5, ..., 5 cm (numpy.interp); e > 5 cm takes the 5 cm fraction 8/32, "
    "an optimistic value there. The predicted successes out of 32 are 32 x the mean of the mapped "
    "values over the windows, with a root-clustered interval. It treats each window's readout "
    "error as a constant target error in a random direction, which is what tau measured",
    "signed_mean": "the mean signed R_off(V) error (predicted minus true offset) along x and "
    "along y (cm), with root-clustered intervals; tau's tolerance depends on the direction "
    "(protocol §9b: -y errors fail earlier than +y)",
    "sensitivity": "the views that would be admitted (A1-A4 unchanged) at tau = 0.5 cm and at "
    "tau = 1.5 cm, the neighbouring levels; reported, never a row",
    "extra_with_representation": "under OBS-EXTRA, the onboard views that pass B with R_full or "
    "R_pix are listed beside the row",
}


TAU_REMEASURE = (
    "on an OBS-ONBOARD or OBS-EXTRA outcome, tau is re-measured on fresh development seeds (same "
    "arm, condition, levels, direction rule and rule) before the next world-model task freezes; "
    "its bar B uses the re-measured tau (c_V <= B <= tau_re), and if tau_re < c_V no bar is "
    "feasible and that task is not frozen"
)


def tau_curve_fraction(errors_cm, measured: dict | None = None) -> np.ndarray:
    """REPORTED['tau_curve_prediction']: the measured success fraction at each error (cm)."""
    m = measured if measured is not None else TAU_MEASURED
    levels = np.asarray(TAU["levels_cm"], np.float64)
    fraction = np.asarray([m["counts"][str(float(x))] for x in levels], np.float64) / TAU["resets"]
    return np.interp(np.asarray(errors_cm, np.float64), levels, fraction)


# ----- admission, rows and consequences -----------------------------------------------------------
ADMISSION = {
    "A1_precision": "c_V = the upper 95 % bound of the root-clustered median error of R_off(V) "
    "on the moving windows <= tau",
    "A2_floor": "the upper 95 % bound of median(e_R_off) / median(e_R_floor) < 1.0",
    "A3_clock_prior": "the upper 95 % bound of median(e_R_off) / median(e_clock) < 1.0",
    "A4_plate_hidden": "the lower 95 % bound of R_off(V)'s median error on the plate-hidden "
    "t + 16 frames > tau (without the plate, the view does not read the offset within tau)",
    "reported": "the apple-hidden error, persistence, TASK-074's o* prior, R_proprio, the plate "
    "and apple targets, the learning curve, R_full and R_pix",
}
REPRESENTATION_PASS = {
    "B1_precision": "the upper 95 % bound of the median error of R_full(V) or R_pix(V) <= tau",
    "B3_clock_prior": "its upper 95 % ratio bound against the clock prior < 1.0",
    "B4_plate_hidden": "its plate-hidden lower 95 % bound > tau",
}
RATIO_BAR = 1.0


def admitted(stats: dict, tau_cm: float) -> dict:
    """R_off admission of one view. ``stats``: ``c_upper``, ``floor_ratio_upper``,
    ``clock_ratio_upper``, ``plate_hidden_lower`` (cm and ratios)."""
    checks = {
        "A1_precision": bool(stats["c_upper"] <= tau_cm),
        "A2_floor": bool(stats["floor_ratio_upper"] < RATIO_BAR),
        "A3_clock_prior": bool(stats["clock_ratio_upper"] < RATIO_BAR),
        "A4_plate_hidden": bool(stats["plate_hidden_lower"] > tau_cm),
    }
    return {"admitted": all(checks.values()), "checks": checks}


def representation_passes(stats: dict, tau_cm: float) -> dict:
    checks = {
        "B1_precision": bool(stats["c_upper"] <= tau_cm),
        "B3_clock_prior": bool(stats["clock_ratio_upper"] < RATIO_BAR),
        "B4_plate_hidden": bool(stats["plate_hidden_lower"] > tau_cm),
    }
    return {"passes": all(checks.values()), "checks": checks}


ROWS = ("V", "TAU-NONE", "OBS-ONBOARD", "OBS-EXTRA", "OBS-REPRESENTATION", "OBS-NONE")
CLAUSE_ROWS = ("OBS-NONE",)
CLAUSE_SCOPE = (
    "a LeWM world-model task (planner or critic) for the place phase of apple-to-plate-v2 under "
    "TASK-074's condition, built on any of these four views (onboard 112, onboard 224, the hand "
    "crop, overview 224) with frozen DINOv2 features: none is preregistered without new evidence "
    "of a different kind (a new view, a new readout family or a task or condition change). The "
    "LeWM backend, the v2 task and the product goal stay open"
)
ROW_CONSEQUENCES = {
    "V": "the void rule",
    "TAU-NONE": "escalate: the place itself fails at zero planted error; nothing is rendered "
    "(found before the freeze, so the protocol is not frozen)",
    "OBS-ONBOARD": "close: the next task preregisters the LeWM place planner or critic on the "
    "chosen onboard view, with an O2-type encoded-readout bar B, c_V <= B <= tau, and a "
    "predicted-latent bar <= tau with any allowance above c_V calibrated before its freeze",
    "OBS-EXTRA": "escalate: only overview 224 is admitted; a hardware change against PRD.md:208 "
    "('Do not require additional cameras for MVP') needs the owner's ruling",
    "OBS-REPRESENTATION": "escalate: the binding factor is the pooled latent or its linear "
    "readout, not the data; a representation change for the next world model is a design "
    "question for the owner, not a silent reopening of a closed line",
    "OBS-NONE": "clause: no view reads the offset within tau, even raw pixels; if the learning "
    "curve is still falling the next step is a larger corpus (candidate (c)), otherwise a task "
    "or condition change",
}


def decide(views: dict, tau_cm: float | None, *, void: bool = False) -> dict:
    """First match. ``views[v]``: ``r_off`` (``admitted`` input), ``r_full`` and ``r_pix``
    (``representation_passes`` input)."""
    if void:
        return {"row": "V", "chosen_view": None}
    if tau_cm is None:
        return {"row": "TAU-NONE", "chosen_view": None}
    if set(views) != set(VIEWS):
        raise ContractError("the decision needs every view")
    adm = {v: admitted(views[v]["r_off"], tau_cm) for v in VIEWS}
    rep = {
        v: {k: representation_passes(views[v][k], tau_cm) for k in ("r_full", "r_pix")}
        for v in VIEWS
    }
    onboard = [v for v in ONBOARD_VIEWS if adm[v]["admitted"]]
    onboard_rep = [v for v in ONBOARD_VIEWS if any(r["passes"] for r in rep[v].values())]
    extra = [v for v in EXTRA_VIEWS if adm[v]["admitted"]]
    passing = [v for v in VIEWS if any(r["passes"] for r in rep[v].values())]
    if onboard:
        row, chosen = "OBS-ONBOARD", onboard[0]
    elif extra:
        row, chosen = "OBS-EXTRA", extra[0]
    elif passing:
        row, chosen = "OBS-REPRESENTATION", passing[0]
    else:
        row, chosen = "OBS-NONE", None
    return {
        "row": row,
        "chosen_view": chosen,
        "admitted": [v for v in VIEWS if adm[v]["admitted"]],
        "admission": adm,
        "representation": rep,
        "tau_cm": tau_cm,
        "consequence": ROW_CONSEQUENCES[row],
        "onboard_representation_passes": onboard_rep if row == "OBS-EXTRA" else None,
        "abandonment_clause_fires": row in CLAUSE_ROWS,
    }


def downstream_bar_interval(c_v_cm: float, tau_cm: float) -> dict:
    """The next world-model task's O2-type encoded-readout bar B must satisfy c_V <= B <= tau."""
    ok = bool(c_v_cm <= tau_cm)
    return {
        "lower_cm": float(c_v_cm),
        "upper_cm": float(tau_cm),
        "feasible": ok,
        "rule": "c_V <= B <= tau; the predicted-latent bar <= tau",
    }


def bar_is_valid(bar_cm: float, c_v_cm: float, tau_cm: float) -> bool:
    return bool(c_v_cm <= bar_cm <= tau_cm)


# ----- platform, guards, void rule ----------------------------------------------------------------
PLATFORM = lp.PLATFORM
SIM_WORKERS = lp.SIM_WORKERS  # 6
THREAD_ENV = lp.THREAD_ENV
QUIET_MACHINE = lp.QUIET_MACHINE  # 1- and 5-minute load <= 2.0 at the start
MEMORY = dict(lp.MEMORY) | {
    "scale_rule": "the readouts smoke's 'scale' probe runs the readouts stage's own function on "
    "smoke data presented at the real sizes (240 train + 30 val slots, each slot's source "
    "episode and views file decoded afresh); its peak must be at least SCALE_MARGIN_GIB below "
    "the ceiling before any GO (TASK-074 addendum A1's lesson)",
}
SCALE_MARGIN_GIB = 2.0
GPU = {
    "min_free_gib_after": 4.0,
    "process_cap_gib": 3.0,
    "rule": "the GPU is shared with a resident service (about 6.6 GB, never touched); a GPU stage "
    "starts only if free memory >= 4.0 GiB + the process cap, and the process is capped with "
    "torch.cuda.set_per_process_memory_fraction at 3.0 GiB, so at least 4 GiB stays free; an "
    "allocation failure is a V",
}
DISK = {
    "min_free_gib": 10.0,
    "rule": "the render stage starts only if the disk holding its output has >= 10 GiB free plus "
    "the projected store (the smoke's measured bytes per root x 270 x 1.5), and a watch voids "
    "the stage if free space falls below 10 GiB",
}
CAPS_SECONDS = {"global": 43_200.0, "per_root": 300.0, "per_attempt": 300.0}
VOID_RULE = (
    "a guard, a crash, a cap, a CUDA allocation failure or a failed reproduction makes the stage "
    "V and nothing in it is read; batches are never shrunk to fit. A V before a stage's boundary "
    "is not a spent attempt and may be repeated as-is, recorded. The boundary is "
    "cohort_first_render_utc for the render stage and first_outcome_utc for the readouts stage "
    "(written after G-repro-off and the loading of the read roots and their folds, before the "
    "first view is featurised or fitted). After the boundary a stage may be repeated once from "
    "scratch after a reviewed fix; a second V closes TASK-075 as INCONCLUSIVE. Nothing is "
    "re-thresholded, refitted or re-selected after its numbers are seen"
)
STAGES = ("tau (Stage 0, before the freeze)", "render", "readouts", "results")


def frozen_block() -> dict:
    return fp._plain(
        {
            "protocol": PROTOCOL,
            "task": TASK,
            "carried_from": [lp.PROTOCOL],
            "owner_decisions": OWNER_DECISIONS,
            "delegated": DELEGATED,
            "delegated_choices": DELEGATED_CHOICES,
            "seed_ranges": SEED_RANGES,
            "smoke_seeds": SMOKE_SEEDS,
            "task_block": TASK_BLOCK,
            "forbidden_ranges": FORBIDDEN_RANGES,
            "seeds": SEEDS,
            "scene_version": SCENE_VERSION,
            "counted_success": COUNTED_SUCCESS,
            "expert": EXPERT,
            "carried": CARRIED,
            "evidence": EVIDENCE,
            "condition": CONDITION,
            "source_corpus": SOURCE_CORPUS,
            "task074_reference": TASK074_REFERENCE,
            "views": VIEWS,
            "view_specs": VIEW_SPECS,
            "reference_view": REFERENCE_VIEW,
            "hidden_kinds": HIDDEN_KINDS,
            "hidden_group": HIDDEN_GROUP,
            "render_rule": RENDER_RULE,
            "steps": {
                "train": TRAIN_STEPS,
                "decision": DECISION_STEPS,
                "eval": EVAL_STEPS,
                "render": RENDER_STEPS,
                "hidden": HIDDEN_STEPS,
                "chunk": CHUNK,
                "band": FEATURE_BAND,
                "moving_threshold_m": MOVING_THRESHOLD_M,
                "root_min_length": ROOT_MIN_LENGTH,
            },
            "tau": TAU,
            "tau_measured": TAU_MEASURED,
            "readouts": READOUTS,
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "learning_curve": LEARNING_CURVE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "admission": ADMISSION,
            "representation_pass": REPRESENTATION_PASS,
            "ratio_bar": RATIO_BAR,
            "reported": REPORTED,
            "reported_percentile": REPORTED_PERCENTILE,
            "sensitivity_tau_cm": SENSITIVITY_TAU_CM,
            "tau_remeasure": TAU_REMEASURE,
            "rows": ROWS,
            "clause_rows": CLAUSE_ROWS,
            "clause_scope": CLAUSE_SCOPE,
            "row_consequences": ROW_CONSEQUENCES,
            "platform": PLATFORM,
            "sim_workers": SIM_WORKERS,
            "thread_env": THREAD_ENV,
            "quiet_machine": QUIET_MACHINE,
            "memory": MEMORY,
            "scale_margin_gib": SCALE_MARGIN_GIB,
            "gpu": GPU,
            "disk": DISK,
            "caps_seconds": CAPS_SECONDS,
            "void_rule": VOID_RULE,
            "stages": STAGES,
        }
    )


def frozen_sha256() -> str:
    return hashlib.sha256(json.dumps(frozen_block(), sort_keys=True).encode()).hexdigest()
