"""Decision-time observation re-probe (TASK-061): look motion, guards, Holm, decision.

Implements what ``docs/experiments/apple_observation_reprobe_v1.md`` adds to the TASK-059 probe.
Everything the two share -- readouts, folds, metrics, bars, baselines -- stays in
``embodied_jepa.info_ceiling`` and is called unchanged. NumPy only, so it imports without torch
or MuJoCo; rendering and encoders live in ``scripts/probe_observation_reprobe.py``.

**The look motion reads nothing about the reset.** ``look_sequence()`` takes no argument and
returns a constant; ``execute_look(robot)`` receives only the robot and issues that constant
through the robot's own observe/project/execute path. ``tests/test_observation_reprobe.py``
checks both, and the run's G-look guard checks the applied commands on every root.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from embodied_jepa import info_ceiling as ic
from embodied_jepa.contracts import ContractError

# ----- the look motion (protocol §3.1) --------------------------------------------------------
LOOK_STEPS = 8
_LOOK_COMMAND = np.array(
    [0.0] * 6 + [0.0, -0.4, 0.4, -0.5, -0.5, 0.5] + [-1.0, -1.0], dtype=np.float32
)
_LOOK_COMMAND.setflags(write=False)
LOOK_SEQUENCE_SHA256 = "17bfb0702a25124b44dc2565310fb947742b5f025c8211dd16d66ede15be587c"

DECISIONAL = ("L-raw", "L-E0", "LH-raw", "O-raw")  # listed order = Holm tie-break order
ARM_OF = {"L-raw": "L", "L-E0": "L", "LH-raw": "LH", "O-raw": "O"}
ALPHA_ONE_SIDED = 0.025
LOOK_STATE_TOLERANCE = 1e-6
LOOK_MOVE_TOLERANCE_M = 1e-6
PRIOR_TOLERANCE = 1e-9


class GuardError(ic.GuardError):
    """A §9 guard failed: the run is void and no outcome is read."""


def look_sequence() -> np.ndarray:
    """The fixed look: the same ``LOOK_STEPS`` x 14 float32 commands on every reset."""
    sequence = np.repeat(_LOOK_COMMAND[None], LOOK_STEPS, axis=0)
    sequence.setflags(write=False)
    return sequence


def sequence_sha256(sequence) -> str:
    return hashlib.sha256(np.ascontiguousarray(sequence, np.float32).tobytes()).hexdigest()


def execute_look(robot, after_step=None) -> np.ndarray:
    """Issue the look through the robot's own step. Returns the applied commands.

    Only ``robot.observe``, ``robot.project_candidates`` and ``robot.execute`` are called, and
    the commands come from ``look_sequence()`` alone. ``after_step(step)``, if given, is called
    after each executed command for measurement (G-look (d)); its return value is ignored, so
    it cannot influence any command. A projection that is infeasible or a command the robot
    rejects is a G-look failure."""
    applied = []
    for command in look_sequence():
        robot.observe()
        projected = robot.project_candidates(np.array(command, np.float32)[None, None, None])
        if not projected.feasible[0, 0]:
            raise GuardError("G-look: a look command's projection is infeasible")
        result = robot.execute(np.asarray(projected.actions[0, 0, 0], np.float32))
        if result.applied_action is None:
            raise GuardError(f"G-look: a look command was rejected ({result.reason})")
        applied.append(np.asarray(result.applied_action, np.float32).copy())
        if after_step is not None:
            after_step(len(applied) - 1)
    return np.stack(applied)


def check_look(
    *,
    sequence_sha: str,
    want_sha: str,
    applied,
    post_states,
    apple_xy_moves,
    plate_moves,
    contacts,
    steps_checked,
    tolerance=LOOK_STATE_TOLERANCE,
    move_tolerance=LOOK_MOVE_TOLERANCE_M,
) -> dict:
    """G-look. ``applied``: (instances, roots, steps, 14); ``post_states``: (roots, dim).

    ``apple_xy_moves``, ``plate_moves`` and ``contacts`` are per root, taken as the worst over
    the checks made after every look command; ``steps_checked`` counts those checks per root."""
    if sequence_sha != want_sha:
        raise GuardError(f"G-look: look sequence {sequence_sha} != preregistered {want_sha}")
    requested = look_sequence().astype(np.float64)
    applied = np.asarray(applied, np.float64)
    if applied.ndim != 4 or applied.shape[2:] != requested.shape or applied.shape[1] < 2:
        raise GuardError(f"G-look: applied commands have shape {applied.shape}")
    worst = float(np.abs(applied - requested).max())
    if worst != 0.0:
        raise GuardError(f"G-look: applied commands differ from the requested look by {worst}")
    states = np.asarray(post_states, np.float64)
    spread = float(np.abs(states - states[0]).max())
    if not spread <= tolerance:
        raise GuardError(f"G-look: post-look joint state varies across resets (spread {spread})")
    apple = float(np.max(np.abs(np.asarray(apple_xy_moves, np.float64))))
    plate = float(np.max(np.abs(np.asarray(plate_moves, np.float64))))
    if not apple <= move_tolerance:
        raise GuardError(f"G-look: the apple moved {apple} m in xy during the look")
    if not plate <= move_tolerance:
        raise GuardError(f"G-look: the plate moved {plate} m during the look")
    checked = np.asarray(steps_checked)
    if checked.size == 0 or not np.all(checked == LOOK_STEPS):
        raise GuardError(
            f"G-look: not every look command was checked ({sorted(set(checked.tolist()))})"
        )
    touched = int(np.sum(np.asarray(contacts, bool)))
    if touched:
        raise GuardError(f"G-look: the hand touches the apple after the look on {touched} roots")
    return {
        "applied_max_abs_minus_requested": worst,
        "post_look_state_max_abs_spread": spread,
        "apple_xy_move_max_m": apple,
        "plate_move_max_m": plate,
        "hand_contact_roots": touched,
    }


def check_priors(computed: dict, recorded: dict, what: str, *, tol=PRIOR_TOLERANCE) -> None:
    """G-prior, named: every recorded key recomputed within ``tol``."""
    try:
        ic.check_priors(computed, recorded, tol=tol)
    except ic.GuardError as error:
        raise GuardError(f"G-prior ({what}): {error}") from error


# ----- render-path validation (§5): demotes an arm, never voids ------------------------------
REQUIRED_PATH_CHECKS = {
    "L": (
        "P0_reset_112_equals_stored",
        "P1_post_112_replica",
        "P2_png_post_112",
        "P4_post_112_rerender",
    ),
    "LH": (
        "P1_post_224_replica",
        "P2_png_post_224",
        "P3_physics_224_equals_112",
        "P4_post_224_rerender",
        "P5_model_224_equals_112",
    ),
    "O": ("P1_overview_replica", "P2_png_overview", "P4_overview_rerender"),
}


def frames_identical(kept, again) -> bool:
    """P1/P2/P4 primitive: byte-for-byte equality of two frames of the same shape."""
    kept, again = np.asarray(kept), np.asarray(again)
    return bool(
        kept.shape == again.shape and kept.dtype == again.dtype and np.array_equal(kept, again)
    )


def models_agree(first: dict, second: dict) -> bool:
    """P5: the render-relevant model facts of two instances agree on every key."""
    return bool(set(first) == set(second) and all(first[k] == second[k] for k in first))


def validated_arms(counts: dict, roots: int) -> dict:
    """Arm -> True iff every applicable check held on every root."""
    out = {}
    for arm, keys in REQUIRED_PATH_CHECKS.items():
        missing = [k for k in keys if k not in counts]
        if missing:
            raise ContractError(f"render-path counts missing {missing}")
        out[arm] = all(int(counts[k]) == int(roots) for k in keys)
    return out


# ----- visibility-conditional mean with the arm's own flag (reported) --------------------------
def arm_occ_prior(xy, occluded, fold) -> np.ndarray:
    """A training fold with no root of the held root's flag falls back to its training mean."""
    xy = np.asarray(xy, np.float64)
    occluded = np.asarray(occluded, bool)
    out = np.zeros_like(xy)
    for k in range(int(np.max(fold)) + 1):
        fit, held = fold != k, fold == k
        for flag in (True, False):
            rows = held & (occluded == flag)
            if not rows.any():
                continue
            source = fit & (occluded == flag)
            out[rows] = xy[source].mean(0) if source.any() else xy[fit].mean(0)
    return out


# ----- p-values and Holm (§7) ----------------------------------------------------------------
def normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def score_test_p(correct: int, n: int, p0: float) -> float:
    """One-sided score test of H0: accuracy <= p0 -- the test the Wilson interval inverts."""
    if n <= 0:
        raise ContractError("score test needs n > 0")
    if not 0.0 < p0 < 1.0:
        return 1.0  # protocol §7: p_T2 = 1 when p0 is 0 or 1 (never passes)
    z = (correct / n - p0) / math.sqrt(p0 * (1.0 - p0) / n)
    return normal_sf(z)


def bootstrap_share_above(numerator, denominator, idx, statistic, bar) -> float:
    """Share of paired resamples whose ratio exceeds ``bar`` (undefined ratios count as above)."""
    num, den = np.asarray(numerator, np.float64), np.asarray(denominator, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = statistic(num[idx], axis=1) / statistic(den[idx], axis=1)
    above = ~np.isfinite(boot) | (boot > bar)
    return float(above.mean())


def hypothesis_pvalues(xy_pred, dx_pred, xy, dx, priors, mask) -> dict:
    """p_T1, p_T2, p_T3 and p_h on ``mask``, with the same resamples as ``ic.evaluate``."""
    mask = np.asarray(mask, bool)
    n = int(mask.sum())
    if n == 0:
        raise ContractError("empty stratum")
    idx = ic.bootstrap_indices(n)
    xy_t, dx_t = np.asarray(xy, np.float64)[mask], np.asarray(dx, np.float64)[mask]
    err = ic.xy_error_cm(np.asarray(xy_pred)[mask], xy_t)
    err_occ = ic.xy_error_cm(priors["B_occ"][mask], xy_t)
    raw = np.asarray(dx_pred, np.float64)[mask]
    sign = np.sign(dx_t)
    correct = int((np.sign(raw) == sign).sum())
    abs_err = np.abs(np.clip(raw, -ic.TRANSLATION_CLIP, ic.TRANSLATION_CLIP) - dx_t)
    const_err = np.abs(priors["B_const"][mask] - dx_t)
    p0 = float((priors["B_maj"][mask] == sign).mean())
    median = float(np.median(err))
    accuracy = correct / n
    p_t1 = bootstrap_share_above(err, err_occ, idx, ic._median, ic.T1_MAX_RATIO_UPPER)
    p_t2 = score_test_p(correct, n, p0)
    p_t3 = bootstrap_share_above(abs_err, const_err, idx, ic._mean, ic.T3_MAX_RATIO_UPPER)
    point_ok = bool(median <= ic.T1_MAX_MEDIAN_CM and accuracy >= ic.T2_MIN_ACCURACY)
    return {
        "p_T1": p_t1,
        "p_T2": p_t2,
        "p_T3": p_t3,
        "point_conditions_hold": point_ok,
        "p": max(p_t1, p_t2, p_t3) if point_ok else 1.0,
        "median_cm": median,
        "accuracy": accuracy,
        "B_maj": p0,
    }


def holm(pvalues: dict, *, alpha=ALPHA_ONE_SIDED, order=DECISIONAL) -> dict:
    """Holm step-down. ``pvalues``: hypothesis -> p. Ties broken by ``order``."""
    names = [h for h in order if h in pvalues]
    if set(names) != set(pvalues):
        raise ContractError(f"unknown hypotheses {set(pvalues) - set(order)}")
    m = len(names)
    ranked = sorted(names, key=lambda h: (float(pvalues[h]), order.index(h)))
    steps, rejecting = [], True
    for j, name in enumerate(ranked):
        threshold = alpha / (m - j)
        p = float(pvalues[name])
        rejecting = rejecting and p <= threshold
        steps.append({"hypothesis": name, "p": p, "threshold": threshold, "rejected": rejecting})
    return {
        "alpha_one_sided": alpha,
        "steps": steps,
        "rejected": {s["hypothesis"]: s["rejected"] for s in steps},
    }


# ----- spurious check (§7.4) ------------------------------------------------------------------
def predict_with_fold_readouts(fits, fold, g_new, norms) -> np.ndarray:
    """Predict row i of new features with the out-of-fold readout that held root i out.

    ``g_new``: (roots, training rows) Gram of new features against the original features."""
    rows = []
    for i in range(len(fold)):
        fit = fits[int(fold[i])]
        rows.append(fit.predict(g_new[i : i + 1, fit.stats.index], norms[i : i + 1])[0])
    return np.asarray(rows)


def spurious_verdict(hidden_eval: dict | None, *, renderer_reproduces: bool) -> dict:
    """``hidden_eval``: ``ic.evaluate`` on apple-hidden frames over all roots."""
    if not renderer_reproduces or hidden_eval is None:
        return {
            "available": False,
            "spurious": True,
            "reading": "check unavailable: the ablation renderer did not reproduce the frame",
        }
    still = bool(hidden_eval["beats_prior"])
    return {
        "available": True,
        "spurious": still,
        "reading": "cue is not the apple: spurious" if still else "apple-caused",
    }


# ----- decision (§8) --------------------------------------------------------------------------
ROWS = ("V", "O-LOOK-E0", "O-LOOK-RAW", "O-LOOK-224", "O-OVERVIEW", "O-NONE")


def passes(*, validated: bool, succeeds: bool, rejected: bool, spurious: bool) -> bool:
    return bool(validated and succeeds and rejected and not spurious)


def decide(*, void: bool, passed: dict) -> dict:
    """First matching row of §8. ``passed``: hypothesis -> bool for all four hypotheses."""
    missing = set(DECISIONAL) - set(passed)
    if missing:
        raise ContractError(f"decision needs every decisional hypothesis; missing {missing}")
    if void:
        return {"outcome": "V"}
    ok = {h: bool(passed[h]) for h in DECISIONAL}
    if ok["L-E0"]:
        row = "O-LOOK-E0"
    elif ok["L-raw"]:
        row = "O-LOOK-RAW"
    elif ok["LH-raw"]:
        row = "O-LOOK-224"
    elif ok["O-raw"]:
        row = "O-OVERVIEW"
    else:
        row = "O-NONE"
    return {
        "outcome": row,
        "passed": ok,
        "passing_hypotheses": [h for h in DECISIONAL if ok[h]],
    }
