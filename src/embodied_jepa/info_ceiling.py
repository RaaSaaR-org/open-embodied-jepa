"""Reset-frame information-ceiling probe (TASK-059): readouts, statistics, guards, decision.

Implements the numerical core of ``docs/experiments/apple_info_ceiling_v1.md``. NumPy only, so
it imports without torch or MuJoCo; rendering and encoders live in
``scripts/probe_info_ceiling.py``.

**Readouts (protocol §6).** Every source is read out by the same dual (kernel) ridge. Everything
is expressed through the Gram matrix ``G = X Xᵀ`` of the raw (uncentred) feature vectors, so a
128-d encoder feature and a 602 112-d pixel vector cost the same after one Gram pass. Centring by
the **training-part mean** is exact algebra on ``G``: row ``e``'s centred inner product with
training row ``t`` is ``G[e,t] − mean_j G[e,j] − mean_i G[i,t] + mean G[T,T]`` over training
indices only. No statistic of a held-out row enters a fit; ``tests/test_info_ceiling.py`` checks
the identity against explicit centring.

**Nothing here reads a label of a held-out row** -- the nested selection sees only the outer
training part, and the outer fold's rows are predicted once by the refit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from embodied_jepa.contracts import ContractError

FOLDS = 10
FOLD_SEED = 59
INNER_FOLDS = 5
INNER_SEED_BASE = 5900
BOOTSTRAP_SEED = 5901
BOOTSTRAP_RESAMPLES = 10_000
LAMBDAS = tuple(10.0**k for k in range(-4, 5))
FAMILIES = ("linear", "rbf")
TRANSLATION_CLIP = 0.4
SEED_RANGE = (48000, 48199)
COHORT_C = frozenset(range(45300, 45340))
COHORT_D = frozenset(range(45000, 45008)) | frozenset(range(45100, 45108))
EXPECTED_ROOTS = {"train": 170, "val": 20}
Z95 = 1.959963984540054

# Thresholds (protocol §9).
T1_MAX_MEDIAN_CM = 1.5
T1_MAX_RATIO_UPPER = 0.6
T2_MIN_ACCURACY = 0.85
T3_MAX_RATIO_UPPER = 0.6


class GuardError(ContractError):
    """A §12 guard failed: the run is void and no outcome is read."""


# ----- folds ---------------------------------------------------------------------------------
def fold_of(n: int) -> np.ndarray:
    """Outer fold per root (roots sorted by seed): seeded permutation, position mod 10."""
    order = np.random.default_rng(FOLD_SEED).permutation(n)
    fold = np.empty(n, np.int64)
    fold[order] = np.arange(n) % FOLDS
    return fold


def inner_fold_of(n: int, outer: int) -> np.ndarray:
    order = np.random.default_rng(INNER_SEED_BASE + int(outer)).permutation(n)
    fold = np.empty(n, np.int64)
    fold[order] = np.arange(n) % INNER_FOLDS
    return fold


def fold_assignment_sha256(seeds, folds) -> str:
    assignment = {str(int(s)): int(k) for s, k in zip(seeds, folds, strict=True)}
    blob = json.dumps(assignment, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


# ----- guards --------------------------------------------------------------------------------
def check_split(roots, dataset_splits, *, expected=EXPECTED_ROOTS) -> None:
    """G-split. ``roots``: dicts with ``seed``, ``split``, ``episode_id`` (sorted by seed)."""
    seeds = [int(r["seed"]) for r in roots]
    if seeds != sorted(seeds) or len(set(seeds)) != len(seeds):
        raise GuardError("roots must be distinct and sorted by seed")
    forbidden_ids = set(dataset_splits.get("test", [])) | set(dataset_splits.get("holdout", []))
    counts = {"train": 0, "val": 0}
    for root in roots:
        seed, split, episode = int(root["seed"]), root["split"], root["episode_id"]
        # Named cohorts first, so the refusal says which frozen cohort was reached for.
        if seed in COHORT_C:
            raise GuardError(f"seed {seed} is in the frozen cohort C")
        if seed in COHORT_D:
            raise GuardError(f"seed {seed} is in the development cohort D")
        if not SEED_RANGE[0] <= seed <= SEED_RANGE[1]:
            raise GuardError(f"seed {seed} is outside the wide corpus range {SEED_RANGE}")
        if split not in counts:
            raise GuardError(f"root {episode} is in split {split!r}; only train/val are allowed")
        if episode in forbidden_ids:
            raise GuardError(f"root {episode} is a test/holdout episode")
        if episode not in set(dataset_splits.get(split, [])):
            raise GuardError(f"root {episode} is not in the dataset's frozen {split} split")
        counts[split] += 1
    if counts != dict(expected):
        raise GuardError(f"root counts {counts} != preregistered {dict(expected)}")


def check_hashes(recorded: dict, actual: dict) -> None:
    """G-hash over named inputs: every recorded name present and equal."""
    for name, want in recorded.items():
        got = actual.get(name)
        if got != want:
            raise GuardError(f"{name}: sha256 {got} != preregistered {want}")


def check_render(identical, states, *, state_tolerance=1e-6) -> None:
    """G-render: 112 px re-render byte-identical on every root; reset proprio identical."""
    identical = list(identical)
    if not identical or not all(identical):
        raise GuardError(
            f"112 px re-render differs from the stored frame on {identical.count(False)} roots"
        )
    spread = float(np.asarray(states, np.float64).std(axis=0).max())
    if not spread <= state_tolerance:
        raise GuardError(f"reset proprioception varies across resets (max std {spread})")


def check_expert(expert, recorded, aim_offset, apple_label, apple_truth, *, tol=1e-6) -> None:
    """G-expert: recomputed step-0 expert == recorded label on non-aim roots; label == truth."""
    expert, recorded = np.asarray(expert, np.float64), np.asarray(recorded, np.float64)
    keep = ~np.asarray(aim_offset, bool)
    if not keep.any():
        raise GuardError("no non-aim roots to check the expert against")
    worst = float(np.abs(expert[keep] - recorded[keep]).max())
    if not worst <= tol:
        raise GuardError(f"recomputed expert differs from the recorded label by {worst}")
    drift = float(np.abs(np.asarray(apple_label) - np.asarray(apple_truth)).max())
    if not drift <= tol:
        raise GuardError(f"apple label differs from reset truth by {drift} m")


def check_priors(computed: dict, recorded: dict, *, tol=1e-9) -> None:
    """G-prior: the run's prior baselines reproduce the preregistered calibration."""
    for key, want in recorded.items():
        if key not in computed:
            raise GuardError(f"prior baseline {key} was not recomputed")
        if not abs(float(computed[key]) - float(want)) <= tol:
            raise GuardError(f"prior baseline {key}: {computed[key]} != preregistered {want}")


# ----- native equivalence (reported, not a guard) -------------------------------------------
def box_down(images, factor: int) -> np.ndarray:
    images = np.asarray(images)
    n, h, w, c = images.shape
    if h % factor or w % factor:
        raise ContractError("image size is not a multiple of the downsampling factor")
    shaped = images.reshape(n, h // factor, factor, w // factor, factor, c)
    return shaped.astype(np.float64).mean(axis=(2, 4))


def native_equivalence(stored, down, train_mask) -> dict:
    """Does the downsampled native render reproduce its own stored frame (§2.4)?

    ``equivalent`` is True when either every frame is byte-identical after rounding, or every
    downsampled frame is strictly closer (mean absolute difference) to its own stored frame than
    to any other reset's -- the weakest tolerance that still tells resets apart."""
    stored = np.asarray(stored, np.float64)
    down = np.asarray(down, np.float64)
    if stored.shape != down.shape or len(stored) < 2:
        raise ContractError("stored and downsampled frames must match and number at least 2")
    n = len(stored)
    flat_s = stored.reshape(n, -1)
    flat_d = down.reshape(n, -1)
    mae = np.stack([np.abs(flat_d[i] - flat_s).mean(axis=1) for i in range(n)])
    own = np.diag(mae).copy()
    other = np.where(np.eye(n, dtype=bool), np.inf, mae).min(axis=1)
    between = np.stack([np.abs(flat_s[i] - flat_s).mean(axis=1) for i in range(n)])
    between = np.where(np.eye(n, dtype=bool), np.inf, between).min(axis=1)
    train_mask = np.asarray(train_mask, bool)
    bias = (flat_d - flat_s)[train_mask].mean(axis=0)
    corrected = np.stack([np.abs(flat_d[i] - bias - flat_s).mean(axis=1) for i in range(n)])
    c_own = np.diag(corrected).copy()
    c_other = np.where(np.eye(n, dtype=bool), np.inf, corrected).min(axis=1)
    byte_identical = bool(np.array_equal(np.floor(down + 0.5), stored))
    identified = int((own < other).sum())
    return {
        "byte_identical_after_rounding": byte_identical,
        "own_frame_mae_levels": {
            "min": float(own.min()),
            "median": float(np.median(own)),
            "max": float(own.max()),
        },
        "own_frame_max_abs_levels": float(np.abs(down - stored).max()),
        "identifies_own_reset": identified,
        "stored_frames_nearest_other_reset_mae_levels_min": float(between.min()),
        "bias_corrected_own_mae_max": float(c_own.max()),
        "bias_corrected_identifies_own_reset": int((c_own < c_other).sum()),
        "roots": n,
        "equivalent": byte_identical or identified == n,
    }


# ----- kernel ridge from a Gram matrix -------------------------------------------------------
def gram(features, *, chunk=32) -> tuple[np.ndarray, np.ndarray]:
    """Float64 Gram matrix and squared norms of row-flattened features."""
    x = np.asarray(features).reshape(len(features), -1)
    n = len(x)
    g = np.empty((n, n), np.float64)
    for start in range(0, n, chunk):
        block = x[start : start + chunk].astype(np.float64)
        for other in range(0, n, chunk):
            g[start : start + chunk, other : other + chunk] = (
                block @ x[other : other + chunk].astype(np.float64).T
            )
    return g, np.diag(g).copy()


def cross_gram(new, reference, *, chunk=32) -> tuple[np.ndarray, np.ndarray]:
    """``new @ referenceᵀ`` and the squared norms of ``new``."""
    a = np.asarray(new).reshape(len(new), -1)
    b = np.asarray(reference).reshape(len(reference), -1)
    out = np.empty((len(a), len(b)), np.float64)
    for start in range(0, len(a), chunk):
        block = a[start : start + chunk].astype(np.float64)
        for other in range(0, len(b), chunk):
            out[start : start + chunk, other : other + chunk] = (
                block @ b[other : other + chunk].astype(np.float64).T
            )
    norms = np.einsum("ij,ij->i", a.astype(np.float64), a.astype(np.float64))
    return out, norms


@dataclass(frozen=True, eq=False)
class TrainStats:
    """What a fitted readout keeps of its training part (and nothing of any other row)."""

    index: np.ndarray
    col_mean: np.ndarray  # mean_i G[i, t] over training i, per training t
    grand_mean: float
    diag: np.ndarray
    sigma2: float


def train_stats(g, diag, index) -> TrainStats:
    index = np.asarray(index)
    g_tt = g[np.ix_(index, index)]
    d_t = diag[index]
    d2 = d_t[:, None] + d_t[None, :] - 2 * g_tt
    upper = d2[np.triu_indices(len(index), k=1)]
    sigma2 = float(np.median(np.maximum(upper, 0.0)))
    if not sigma2 > 0:
        raise ContractError("RBF bandwidth is zero: the training rows are identical")
    return TrainStats(index, g_tt.mean(axis=0), float(g_tt.mean()), d_t, sigma2)


def kernel_rows(family, g_rt, diag_r, stats: TrainStats) -> np.ndarray:
    """Kernel between arbitrary rows (their Gram against the training rows) and the training."""
    if family == "linear":
        return g_rt - g_rt.mean(axis=1, keepdims=True) - stats.col_mean[None] + stats.grand_mean
    if family == "rbf":
        d2 = diag_r[:, None] + stats.diag[None, :] - 2 * g_rt
        return np.exp(-np.maximum(d2, 0.0) / (2 * stats.sigma2))
    raise ContractError(f"unknown readout family {family!r}")


@dataclass(frozen=True, eq=False)
class Readout:
    family: str
    lam_rel: float
    stats: TrainStats
    alpha: np.ndarray
    y_mean: np.ndarray

    def predict(self, g_rt, diag_r) -> np.ndarray:
        return kernel_rows(self.family, g_rt, diag_r, self.stats) @ self.alpha + self.y_mean


def fit_readout(family, lam_rel, g, diag, index, y) -> Readout:
    stats = train_stats(g, diag, index)
    k_tt = kernel_rows(family, g[np.ix_(stats.index, stats.index)], diag[stats.index], stats)
    y_t = np.asarray(y, np.float64)[stats.index]
    y_mean = y_t.mean(axis=0)
    lam = lam_rel * float(np.mean(np.diag(k_tt)))
    alpha = np.linalg.solve(k_tt + lam * np.eye(len(k_tt)), y_t - y_mean)
    return Readout(family, float(lam_rel), stats, alpha, y_mean)


def select(g, diag, train_index, y, outer: int) -> dict:
    """Family and λ by 5-fold inner CV on the outer training part (MSE); ties -> first listed."""
    train_index = np.asarray(train_index)
    inner = inner_fold_of(len(train_index), outer)
    y = np.asarray(y, np.float64)
    sse = {(f, lam): 0.0 for f in FAMILIES for lam in LAMBDAS}
    for j in range(INNER_FOLDS):
        fit_idx, held_idx = train_index[inner != j], train_index[inner == j]
        stats = train_stats(g, diag, fit_idx)
        y_t = y[fit_idx]
        y_mean = y_t.mean(axis=0)
        for family in FAMILIES:
            k_tt = kernel_rows(family, g[np.ix_(fit_idx, fit_idx)], diag[fit_idx], stats)
            k_ht = kernel_rows(family, g[np.ix_(held_idx, fit_idx)], diag[held_idx], stats)
            scale = float(np.mean(np.diag(k_tt)))
            for lam in LAMBDAS:
                alpha = np.linalg.solve(k_tt + lam * scale * np.eye(len(k_tt)), y_t - y_mean)
                sse[(family, lam)] += float(((k_ht @ alpha + y_mean - y[held_idx]) ** 2).sum())
    best = min(sse, key=lambda key: (sse[key], FAMILIES.index(key[0]), key[1]))
    return {"family": best[0], "lam_rel": best[1], "inner_mse": sse[best] / len(train_index)}


def nested_cv(g, diag, y, fold) -> tuple[np.ndarray, list[Readout], list[dict]]:
    """Out-of-fold predictions: for each outer fold, select on its training part, refit, predict."""
    y = np.asarray(y, np.float64)
    squeeze = y.ndim == 1
    y2 = y[:, None] if squeeze else y
    predictions = np.full(y2.shape, np.nan)
    readouts, selections = [], []
    for k in range(int(fold.max()) + 1):
        train_index, held = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        chosen = select(g, diag, train_index, y2, k)
        readout = fit_readout(chosen["family"], chosen["lam_rel"], g, diag, train_index, y2)
        predictions[held] = readout.predict(g[np.ix_(held, train_index)], diag[held])
        readouts.append(readout)
        selections.append({"outer_fold": k} | chosen)
    if not np.isfinite(predictions).all():
        raise ContractError("a root received no out-of-fold prediction")
    return (predictions[:, 0] if squeeze else predictions), readouts, selections


def split_fit(g, diag, y, train_mask) -> tuple[np.ndarray, Readout, dict]:
    """Secondary estimate: select + fit on train, predict val once."""
    y = np.asarray(y, np.float64)
    squeeze = y.ndim == 1
    y2 = y[:, None] if squeeze else y
    train_index, held = np.flatnonzero(train_mask), np.flatnonzero(~np.asarray(train_mask))
    chosen = select(g, diag, train_index, y2, outer=FOLDS)  # inner seed 5910: distinct from CV
    readout = fit_readout(chosen["family"], chosen["lam_rel"], g, diag, train_index, y2)
    predicted = readout.predict(g[np.ix_(held, train_index)], diag[held])
    return (predicted[:, 0] if squeeze else predicted), readout, chosen


# ----- prior-only baselines (§2.6 / §8) ------------------------------------------------------
def prior_predictions(xy, dx, occluded, fold) -> dict:
    """Out-of-fold predictions of every prior-only baseline. Reads no observation."""
    xy, dx = np.asarray(xy, np.float64), np.asarray(dx, np.float64)
    occluded = np.asarray(occluded, bool)
    n = len(dx)
    out = {
        "B_mean": np.zeros((n, 2)),
        "B_occ": np.zeros((n, 2)),
        "B_const": np.zeros(n),
        "B_maj": np.zeros(n),
        "B_y": np.zeros(n),
    }
    for k in range(int(fold.max()) + 1):
        fit, held = fold != k, fold == k
        out["B_mean"][held] = xy[fit].mean(0)
        out["B_const"][held] = np.median(dx[fit])
        out["B_maj"][held] = 1.0 if (dx[fit] > 0).mean() >= 0.5 else -1.0
        for flag in (True, False):
            out["B_occ"][held & (occluded == flag)] = xy[fit & (occluded == flag)].mean(0)
        edges = np.quantile(xy[fit, 1], [0.2, 0.4, 0.6, 0.8])
        fit_bin, held_bin = np.digitize(xy[fit, 1], edges), np.digitize(xy[held, 1], edges)
        held_index = np.flatnonzero(held)
        for b in range(5):
            share = (dx[fit][fit_bin == b] > 0).mean()
            out["B_y"][held_index[held_bin == b]] = 1.0 if share >= 0.5 else -1.0
    return out


def prior_summary(xy, dx, occluded, fold, train_mask) -> dict:
    """The calibration's prior-baseline numbers, recomputed (G-prior compares these)."""
    p = prior_predictions(xy, dx, occluded, fold)
    occluded = np.asarray(occluded, bool)
    xy, dx = np.asarray(xy, np.float64), np.asarray(dx, np.float64)
    train = np.asarray(train_mask, bool)
    val = ~train
    val_majority = 1.0 if (dx[train] > 0).mean() >= 0.5 else -1.0
    split_numbers = {
        "val_predict_the_mean_median_cm": float(
            np.median(xy_error_cm(np.broadcast_to(xy[train].mean(0), xy[val].shape), xy[val]))
        ),
        "val_majority_dx_sign_accuracy": float((np.sign(dx[val]) == val_majority).mean()),
        "val_constant_median_dx_mae": float(np.abs(np.median(dx[train]) - dx[val]).mean()),
    }
    err = xy_error_cm(p["B_mean"], xy)
    occ_err = xy_error_cm(p["B_occ"], xy)
    sign = np.sign(dx)
    maj, yc = p["B_maj"] == sign, p["B_y"] == sign
    return {
        "cv_predict_the_mean_median_cm": float(np.median(err)),
        "cv_predict_the_mean_median_cm_occluded": float(np.median(err[occluded])),
        "cv_predict_the_mean_median_cm_visible": float(np.median(err[~occluded])),
        "cv_occlusion_conditional_mean_median_cm_occluded": float(np.median(occ_err[occluded])),
        "cv_occlusion_conditional_mean_median_cm_visible": float(np.median(occ_err[~occluded])),
        "cv_occlusion_conditional_mean_median_cm": float(np.median(occ_err)),
        "cv_y_conditional_dx_sign_accuracy": float(yc.mean()),
        "cv_y_conditional_dx_sign_accuracy_occluded": float(yc[occluded].mean()),
        "ceiling_if_visible_perfect_and_occluded_at_majority": float(
            ((~occluded).sum() + maj[occluded].sum()) / len(dx)
        ),
        "cv_y_conditional_dx_sign_accuracy_visible": float(yc[~occluded].mean()),
        "cv_constant_median_dx_mae_occluded": float(np.abs(p["B_const"] - dx)[occluded].mean()),
        "cv_constant_median_dx_mae_visible": float(np.abs(p["B_const"] - dx)[~occluded].mean()),
        "constant_plus_clip_dx_mae": float(np.abs(TRANSLATION_CLIP - dx).mean()),
        "cv_majority_dx_sign_accuracy": float(maj.mean()),
        "cv_majority_dx_sign_accuracy_occluded": float(maj[occluded].mean()),
        "cv_majority_dx_sign_accuracy_visible": float(maj[~occluded].mean()),
        "cv_constant_median_dx_mae": float(np.abs(p["B_const"] - dx).mean()),
    } | split_numbers


# ----- statistics ----------------------------------------------------------------------------
def xy_error_cm(predicted, truth) -> np.ndarray:
    return np.linalg.norm(np.asarray(predicted) - np.asarray(truth), axis=1) * 100.0


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n <= 0:
        raise ContractError("Wilson interval needs n > 0")
    p = successes / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    denominator = 1 + z * z / n
    return float((centre - half) / denominator), float((centre + half) / denominator)


def bootstrap_indices(n: int, *, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES):
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def percentile_ci(values) -> tuple[float, float]:
    lo, hi = np.percentile(np.asarray(values), [2.5, 97.5])
    return float(lo), float(hi)


def median_ci(errors, idx) -> dict:
    errors = np.asarray(errors, np.float64)
    lo, hi = percentile_ci(np.median(errors[idx], axis=1))
    return {"median": float(np.median(errors)), "mean": float(errors.mean()), "ci95": [lo, hi]}


def paired_ratio(numerator, denominator, idx, statistic) -> dict:
    """``statistic(num)/statistic(den)`` with a paired bootstrap CI (same resamples)."""
    num, den = np.asarray(numerator, np.float64), np.asarray(denominator, np.float64)
    point = float(statistic(num) / statistic(den))
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = statistic(num[idx], axis=1) / statistic(den[idx], axis=1)
    boot = np.where(np.isfinite(boot), boot, np.inf)
    lo, hi = percentile_ci(boot)
    return {"ratio": point, "ci95": [lo, hi]}


def paired_difference(a, b, idx, statistic) -> dict:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    lo, hi = percentile_ci(statistic(a[idx], axis=1) - statistic(b[idx], axis=1))
    return {"difference": float(statistic(a) - statistic(b)), "ci95": [lo, hi]}


def _median(x, axis=None):
    return np.median(x, axis=axis)


def _mean(x, axis=None):
    return np.mean(x, axis=axis)


def mcnemar_exact(correct_a, correct_b) -> dict:
    from math import comb

    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n, k = only_a + only_b, min(only_a, only_b)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return {"only_first_correct": only_a, "only_second_correct": only_b, "p_two_sided": p}


def evaluate(xy_pred, dx_pred, xy, dx, priors, mask) -> dict:
    """T1/T2/T3 on the roots in ``mask`` against the in-run prior baselines (§9)."""
    mask = np.asarray(mask, bool)
    n = int(mask.sum())
    if n == 0:
        raise ContractError("empty evaluation stratum")
    idx = bootstrap_indices(n)
    xy, dx = np.asarray(xy, np.float64)[mask], np.asarray(dx, np.float64)[mask]
    err = xy_error_cm(np.asarray(xy_pred)[mask], xy)
    err_mean = xy_error_cm(priors["B_mean"][mask], xy)
    err_occ = xy_error_cm(priors["B_occ"][mask], xy)
    raw_dx = np.asarray(dx_pred, np.float64)[mask]
    sign = np.sign(dx)
    correct = np.sign(raw_dx) == sign  # an exact 0 prediction never equals a nonzero target
    clipped = np.clip(raw_dx, -TRANSLATION_CLIP, TRANSLATION_CLIP)
    abs_err = np.abs(clipped - dx)
    const_err = np.abs(priors["B_const"][mask] - dx)
    maj = float((priors["B_maj"][mask] == sign).mean())
    y_prior = float((priors["B_y"][mask] == sign).mean())
    accuracy = float(correct.mean())
    lower, upper = wilson(int(correct.sum()), n)
    t1 = {
        **median_ci(err, idx),
        "ratio_to_B_occ": paired_ratio(err, err_occ, idx, _median),
        "ratio_to_B_mean": paired_ratio(err, err_mean, idx, _median),
        "B_occ_median_cm": float(np.median(err_occ)),
        "B_mean_median_cm": float(np.median(err_mean)),
    }
    t1["pass"] = bool(
        t1["median"] <= T1_MAX_MEDIAN_CM and t1["ratio_to_B_occ"]["ci95"][1] <= T1_MAX_RATIO_UPPER
    )
    t2 = {
        "correct": int(correct.sum()),
        "n": n,
        "accuracy": accuracy,
        "wilson95": [lower, upper],
        "B_maj_accuracy": maj,
        "B_y_accuracy": y_prior,
    }
    t2["pass"] = bool(accuracy >= T2_MIN_ACCURACY and lower > maj)
    mae_lo, mae_hi = percentile_ci(abs_err[idx].mean(axis=1))
    t3 = {
        "mae": float(abs_err.mean()),
        "ci95": [mae_lo, mae_hi],
        "B_const_mae": float(const_err.mean()),
        "ratio_to_B_const": paired_ratio(abs_err, const_err, idx, _mean),
    }
    t3["pass"] = bool(t3["ratio_to_B_const"]["ci95"][1] <= T3_MAX_RATIO_UPPER)
    beats_prior = bool(lower > max(maj, y_prior) or t1["ratio_to_B_occ"]["ci95"][1] < 1.0)
    return {
        "n": n,
        "T1": t1,
        "T2": t2,
        "T3": t3,
        "succeeds": bool(t1["pass"] and t2["pass"] and t3["pass"]),
        "beats_prior": beats_prior,
        "_errors_cm": err,
        "_correct": correct,
    }


def public(result: dict) -> dict:
    """Drop the per-root arrays kept for paired comparisons."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


def beats_random_floor(source: dict, random: dict) -> dict:
    """§9 qualifier: paired CI of median T1 error (source - random) below 0 AND higher T2."""
    n = len(source["_errors_cm"])
    diff = paired_difference(
        source["_errors_cm"], random["_errors_cm"], bootstrap_indices(n), _median
    )
    better_sign = source["T2"]["accuracy"] > random["T2"]["accuracy"]
    return {
        "median_error_difference_cm": diff,
        "T2_accuracy_higher": bool(better_sign),
        "beats_random_floor": bool(diff["ci95"][1] < 0 and better_sign),
        "mcnemar": mcnemar_exact(source["_correct"], random["_correct"]),
    }


# ----- decision (§13) ------------------------------------------------------------------------
DECISIONAL = ("E0", "A3", "random", "raw112")


def decide(*, void: bool, overall: dict, visible: dict, spurious: dict) -> dict:
    """First matching row. ``overall``/``visible``: source -> succeeds (bool). A source whose
    occluded-stratum result was found spurious (§10) is treated as not succeeding overall."""
    for table in (overall, visible):
        missing = set(DECISIONAL) - set(table)
        if missing:
            raise ContractError(f"decision needs every decisional source; missing {missing}")
    if void:
        return {"outcome": "VOID"}
    ok = {s: bool(overall[s]) and not spurious.get(s, False) for s in DECISIONAL}
    vis = {s: bool(visible[s]) for s in DECISIONAL}
    others = ("A3", "raw112", "random")
    if ok["E0"]:
        row = "O-BC"
    elif any(ok[s] for s in others):
        row = "O-ENC"
    elif vis["E0"]:
        row = "O-OCC-BC"
    elif any(vis[s] for s in others):
        row = "O-OCC-ENC"
    else:
        row = "O-OCC-NONE"
    return {"outcome": row, "overall_succeeds": ok, "visible_succeeds": vis}
