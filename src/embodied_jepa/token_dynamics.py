"""TASK-066 (``apple_token_dynamics_v1``): does an action-conditioned predictor over frozen DINOv2
patch-token latents avoid rank collapse on ``apple-look-v1`` and keep the apple readable through
multi-step prediction?

Protocol ``docs/experiments/apple_token_dynamics_v1.md``. This module fixes in code, before any
predictor is trained on the corpus, what the protocol preregisters beyond TASK-065's
``latent_dynamics`` (which it reuses unchanged: the partition, the windows, the cross-session
shuffle, the statistics and the G2-G5 gate functions): the token latent and its configuration,
the calibration rules and their recorded results, the calibrated G1 bars, the budget, the rows
and the abandonment clause. NumPy only: it imports without torch or MuJoCo.

**This is a world-model test only.** It trains an action-conditioned latent predictor and measures
it offline. It preregisters no control formulation: CEM over the world-model cost was abandoned at
TASK-054 and the behaviour-cloning line at TASK-057, and both clauses still hold. **Learned
Apple->Plate is still 0 successes.** The corpus comes from the privileged scripted collector.

The arms (per model seed, per cross-fitting half), as in TASK-065:

* ``W`` -- the pinned upstream LeWM predictor adapted to a token sequence
  (``models.frozen_tokens.frozen_token_model(BACKEND)``) on the frozen DINOv2 final patch tokens
  pooled to a 4 x 4 grid, trained on true actions;
* ``N`` -- the same model trained and evaluated with every action replaced by zero;
* copy-last -- the start latent repeated (no training).
"""

from __future__ import annotations

import math

import numpy as np

from embodied_jepa import latent_dynamics as ld
from embodied_jepa.contracts import ContractError

PROTOCOL = "apple_token_dynamics_v1"
TASK = "TASK-066"
GuardError = ld.GuardError

# ----- data: TASK-065's, unchanged -------------------------------------------------------------
DATASET = ld.DATASET
DATASET_MANIFEST_SHA256 = ld.DATASET_MANIFEST_SHA256
EXPECTED_SESSIONS = ld.EXPECTED_SESSIONS
EXPECTED_EPISODES = ld.EXPECTED_EPISODES
READ_SPLITS = ld.READ_SPLITS  # the test split is never decoded
DECISION_FRAME = ld.DECISION_FRAME
CAMERA = ld.CAMERA
HALVES = ld.HALVES

# ----- the latent and the predictor --------------------------------------------------------------
BACKEND = "leworldmodel"  # the one-line swap: "native_jepa" runs the same test on native code
FROZEN_ENCODER = "dinov2_small_tokens"
TOKEN_WIDTH = 384
NATIVE_GRID = 16  # 224 px / patch 14
TOKEN_GRID = 4  # each pooled token is the mean of a 4 x 4 block of patches
TOKENS = TOKEN_GRID * TOKEN_GRID
LATENT_DIM = TOKENS * TOKEN_WIDTH  # 6144
# Everything not listed is the adapter's declared default, as in TASK-065: hidden_dim (MLP) 128,
# predictor depth 2, 2 heads of 24, lr 3e-4, weight decay 1e-4, clip 1.0, multistep_weight 1.0,
# history one. SIGReg weight 0 (the encoder is frozen).
MODEL_CONFIG = {
    "camera": CAMERA,
    "frozen_encoder": FROZEN_ENCODER,
    "token_grid": TOKEN_GRID,
    "latent_dim": LATENT_DIM,
    "sigreg_weight": 0.0,
    "max_horizon": 64,
}
ARMS = ld.ARMS
MODEL_SEEDS = ld.MODEL_SEEDS
TRAIN_HORIZON = ld.TRAIN_HORIZON
BATCH_SIZE = ld.BATCH_SIZE
# TASK-065's sampler (default_rng(SeedSequence([6500, seed, half]))), shared by W and N: for the
# same seed and half, W and N here draw the windows TASK-065's models drew, in the same order.
sampler_seed = ld.sampler_seed
DEVICE = ld.DEVICE
FEATURE_DEVICE = ld.FEATURE_DEVICE
FEATURE_THREADS = ld.FEATURE_THREADS

# ----- calibration (protocol section 6), on the disjoint pilot, fixed before it ran -------------
PILOT_DATASET = "outputs/task064-scratch/pilot-d/dataset"  # TASK-064's pilot 47900-47931
PILOT_MANIFEST_SHA256 = "0a41d302558c8cb7aaba1b4ddc6f656438bf43b00f261e632c0c43ff54d9f50c"
PILOT_SESSIONS = 32
PILOT_SPLIT_SEED = 6600
PILOT_TRAIN_SESSIONS = 24  # the other 8 are the pilot held-out sessions
CAL_UPDATES = 30_000
CAL_SELECT_EVERY = 500
CAL_RANK_EVERY = 5_000  # collapse statistics are also recorded along the curve (reported)
CAL_RUNS = (("W", 0), ("W", 1), ("N", 0))
SATURATION_TOLERANCE = 0.01  # u_sat: first point within 1 % of the run's minimum criterion
BUDGET_FACTOR = 2.0
BUDGET_STEP = 5_000
BUDGET_MIN = 10_000
BUDGET_MAX = 30_000  # beyond it the calibration escalates to the owner (no freeze)
BAR_FACTOR = 0.5  # G1 bars: half of the recipe's own ratio on disjoint pilot data
BAR_ESCALATE_BELOW = 0.10  # a pilot rank ratio this low escalates (the recipe may be broken)
# Absolute floors (owner ruling 2026-09-27, fixed before the calibration's results were seen):
# a calibrated bar never goes below them. A rank-1 prediction scores about 1 / (encoded effective
# rank), about 0.025 at an encoded effective rank of 40; the mean predictor scores 0.
RANK_FLOOR = 0.10
STD_FLOOR = 0.25
# G1 (iii), comparative (same ruling): W keeps more rank than the no-action model N. The lower
# 95 % bound of (rank ratio W - rank ratio N) must exceed 0, from a session-clustered bootstrap
# of the effective rank in a fixed, model-free basis: the top 256 principal directions of the
# train-split encoded latents (an approximation of the full-width statistic, disclosed).
PROJECTION_DIM = 256
COMPARATIVE_MARGIN = 0.0
COMPARATIVE_RESAMPLES = 2_000
COMPARATIVE_SEED = 6603
TRUNCATION_CONTROLS = (1, 2, 4, 8)  # G1 must fail k = 1, 2, 4 (owner condition); k = 8 reported


def saturation_update(curve, tolerance: float = SATURATION_TOLERANCE) -> int:
    """First update whose held-out criterion is within ``tolerance`` (relative) of the curve's
    minimum. ``curve`` is ``[[update, criterion], ...]`` in update order."""
    points = [(int(u), float(v)) for u, v in curve]
    if not points or any(not math.isfinite(v) or v < 0 for _, v in points):
        raise ContractError("a saturation curve needs finite nonnegative criteria")
    if [u for u, _ in points] != sorted({u for u, _ in points}):
        raise ContractError("a saturation curve must be strictly increasing in updates")
    floor = min(v for _, v in points)
    return next(u for u, v in points if v <= floor * (1.0 + tolerance))


def budget_rule(saturation_updates) -> dict:
    """``U = clamp(5000 * ceil(2 * max(u_sat) / 5000), 10 000, 30 000)``; escalate when
    ``2 * max(u_sat)`` exceeds 30 000 (no freeze)."""
    values = [int(u) for u in saturation_updates]
    if not values or min(values) < 1:
        raise ContractError("the budget rule needs positive saturation updates")
    wanted = BUDGET_FACTOR * max(values)
    updates = int(BUDGET_STEP * math.ceil(wanted / BUDGET_STEP))
    return {
        "max_saturation_update": max(values),
        "wanted": wanted,
        "updates": min(max(updates, BUDGET_MIN), BUDGET_MAX),
        "escalate": bool(wanted > BUDGET_MAX),
    }


def bar_rule(pilot_ratios, floor: float) -> dict:
    """A G1 bar: ``max(floor, floor_to_0.01(0.5 * min(pilot ratios)))``; escalate when the
    reference is below 0.10."""
    values = [float(r) for r in pilot_ratios]
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ContractError("the bar rule needs finite nonnegative pilot ratios")
    reference = min(values)
    relative = math.floor(100 * BAR_FACTOR * reference + 1e-9) / 100
    return {
        "reference": reference,
        "relative": relative,
        "floor": floor,
        "bar": max(floor, relative),
        "escalate": bool(reference < BAR_ESCALATE_BELOW),
    }


# The calibration's results (protocol section 6.5), copied from its report; the freeze uses these.
CALIBRATION = {
    "report": "outputs/task066-calibration/run-1/report.json",
    "report_sha256": "a1d9fc1740cbba795a0404428e8a956de97e946a9e912d6592f188141e2f53e1",
    "revision": "7fa8183499f0b0001bbbd25674a3f18f13f0708a",
    "saturation_updates": {"W-s0": 6500, "W-s1": 7000, "N-s0": 16500},
    "budget_rule": {"wanted": 33000.0, "updates": 30000, "escalate": True},
    # the G1 references: the W calibration models' val-selected checkpoints (W-s0 at update
    # 9500, W-s1 at update 9000), pilot-held-out windows, h = 8 and 16
    "rank_ratio_reference": 0.33151133423512424,  # W-s0 at h = 16
    "std_ratio_reference": 0.7963489955650851,  # W-s1 at h = 8
    "controls_report": "outputs/task066-calibration/controls-1/report.json",
    "controls_fail_G1": {"k1": True, "k2": True, "k4": True, "k8": True},
}
# Owner ruling 2026-09-27T05:36Z on the escalation: option A, the 30 000-update ceiling (an
# under-trained N would bias G3 and G1 (iii) towards W), selection every 1500 updates.
UPDATES = 30_000
SELECT_EVERY = 1_500  # 20 selection points, as in TASK-065

# ----- evaluation: TASK-065's, unchanged ---------------------------------------------------------
EVAL_STRIDE = ld.EVAL_STRIDE
GATED_HORIZONS = ld.GATED_HORIZONS
REPORTED_HORIZONS = ld.REPORTED_HORIZONS
EXTENDED_HORIZONS = ld.EXTENDED_HORIZONS
CLUSTER_BOOTSTRAP_SEED = ld.CLUSTER_BOOTSTRAP_SEED
SHUFFLE_SEED = ld.SHUFFLE_SEED
METRIC_FLOOR_STD = ld.METRIC_FLOOR_STD

# ----- gates (protocol section 8), per seed, at each gated horizon --------------------------------
THRESHOLDS = {
    "G1_max_collapsed_fraction": ld.THRESHOLDS["G1_max_collapsed_fraction"],
    # calibrated: bar_rule on the pilot ratios, max(floor, floor_to_0.01(0.5 * reference))
    "G1_min_effective_rank_ratio": 0.16,  # reference 0.3315, floor 0.10
    "G1_min_std_ratio": 0.39,  # reference 0.7963, floor 0.25
    "G1_min_rank_ratio_margin_over_N_lower": COMPARATIVE_MARGIN,  # strictly above, lower bound
} | {k: v for k, v in ld.THRESHOLDS.items() if not k.startswith("G1")}
GATES = ld.GATES
DYNAMICS_GATES = ld.DYNAMICS_GATES
LATENT_GATES = ("G2", "G3", "G4")  # the latent-error gates, without G1

# ----- budget ------------------------------------------------------------------------------------
# Owner ruling 2026-09-27T05:36Z. Expected: about 0.095 s per update on MPS (synthetic), so
# about 2850 s of updates plus selection and evaluation per model; about 12 h in total.
PER_RUN_SECONDS = 4_500.0  # one model: 30 000 updates, 20 val evaluations, its evaluation
FEATURE_SECONDS = ld.FEATURE_SECONDS  # 5400 s
GLOBAL_WALL_SECONDS = 57_600.0  # 16 h

ROWS = (
    "V",
    "WM-TOK-DYNAMICS",
    "WM-TOK-UNSTABLE",
    "WM-TOK-CEILING",
    "WM-TOK-APPLE-LOST",
    "WM-TOK-COLLAPSE",
    "WM-TOK-NO-DYNAMICS",
)
CLAUSE_ROWS = ("WM-TOK-APPLE-LOST", "WM-TOK-COLLAPSE", "WM-TOK-NO-DYNAMICS")


# ----- collapse statistics from streamed moments -------------------------------------------------
class Moments:
    """Streamed first and second moments of ``[n, d]`` rows around a fixed ``shift``.

    Accumulators with the same shift add (``merge``), so the two cross-fitted halves pool
    exactly. ``collapse_statistics`` equals ``latent_dynamics.collapse_statistics`` on the
    concatenated rows (tested), without holding them."""

    def __init__(self, shift):
        self.shift = np.asarray(shift, np.float64).copy()
        d = len(self.shift)
        self.n = 0
        self.s1 = np.zeros(d)
        self.s2 = np.zeros((d, d))
        self._rank = None

    def add(self, rows) -> None:
        x = np.asarray(rows, np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.shift):
            raise ContractError("rows must be [n, d] with the accumulator's width")
        x = x - self.shift
        self.n += len(x)
        self.s1 += x.sum(0)
        self.s2 += x.T @ x
        self._rank = None

    def merge(self, other: Moments) -> Moments:
        if not np.array_equal(self.shift, other.shift):
            raise ContractError("only accumulators with the same shift merge")
        out = Moments(self.shift)
        out.n, out.s1, out.s2 = self.n + other.n, self.s1 + other.s1, self.s2 + other.s2
        return out

    def centered(self) -> np.ndarray:
        if self.n < 2:
            raise ContractError("collapse statistics need at least two rows")
        m = self.s1 / self.n
        return self.s2 - self.n * np.outer(m, m)

    def std(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.centered()) / self.n, 0.0, None))

    def effective_rank(self) -> float:
        """``exp`` of the entropy of the normalised covariance spectrum: the spectrum of the
        centered rows' singular values squared, as ``latent_dynamics.effective_rank`` uses."""
        if self._rank is None:
            energy = np.clip(np.linalg.eigvalsh(self.centered()), 0.0, None)
            total = energy.sum()
            if total <= 0:
                self._rank = 0.0
            else:
                p = energy[energy > 0] / total
                self._rank = float(np.exp(-(p * np.log(p)).sum()))
        return self._rank


def collapse_statistics(predicted: Moments, encoded: Moments) -> dict:
    """TASK-065's G1 quantities (``latent_dynamics.collapse_statistics``) from moments."""
    if predicted.n != encoded.n:
        raise ContractError("predicted and encoded moments must cover the same windows")
    p_std, e_std = predicted.std(), encoded.std()
    p_rank, e_rank = predicted.effective_rank(), encoded.effective_rank()
    return {
        "predicted_std_mean": float(p_std.mean()),
        "encoded_std_mean": float(e_std.mean()),
        "std_ratio": float(p_std.mean() / e_std.mean()) if e_std.mean() > 0 else 0.0,
        "predicted_collapsed_fraction": float((p_std < 0.01).mean()),
        "encoded_collapsed_fraction": float((e_std < 0.01).mean()),
        "predicted_effective_rank": p_rank,
        "encoded_effective_rank": e_rank,
        "effective_rank_ratio": p_rank / e_rank if e_rank > 0 else 0.0,
        "samples": int(predicted.n),
    }


# ----- gates -------------------------------------------------------------------------------------
def require_frozen() -> None:
    """Refuse a gated decision before the calibration's results are copied in."""
    missing = [k for k, v in THRESHOLDS.items() if v is None]
    if missing or UPDATES is None or PER_RUN_SECONDS is None or GLOBAL_WALL_SECONDS is None:
        raise ContractError(f"the protocol is not frozen: {missing or 'budget'}")


def g1_parts(stats: dict, comparative: dict, thresholds: dict | None = None) -> dict:
    """G1's parts: (i)+(ii) the calibrated rank bar (never below the 0.10 floor), the std bar
    (never below 0.25) and the collapsed fraction, on the full-width statistic; (iii) the
    comparative W-over-N rank criterion from the projected bootstrap."""
    t = THRESHOLDS if thresholds is None else thresholds
    if t["G1_min_effective_rank_ratio"] is None or t["G1_min_std_ratio"] is None:
        raise ContractError("G1's bars are calibrated; they are not set yet")
    if t["G1_min_effective_rank_ratio"] < RANK_FLOOR or t["G1_min_std_ratio"] < STD_FLOOR:
        raise ContractError("a G1 bar lies below its absolute floor")
    return {
        "collapsed_fraction": bool(
            stats["predicted_collapsed_fraction"] <= t["G1_max_collapsed_fraction"]
        ),
        "rank": bool(stats["effective_rank_ratio"] >= t["G1_min_effective_rank_ratio"]),
        "std": bool(stats["std_ratio"] >= t["G1_min_std_ratio"]),
        "rank_over_N": bool(
            comparative["undefined_resamples"] == 0
            and comparative["ci95"][0] > t["G1_min_rank_ratio_margin_over_N_lower"]
        ),
    }


def g1_passes(stats: dict, comparative: dict, thresholds: dict | None = None) -> bool:
    return bool(all(g1_parts(stats, comparative, thresholds).values()))


# ----- G1 (iii): per-session moments in a fixed projected basis, and their bootstrap ----------
def projection_basis(encoded: Moments, k: int = PROJECTION_DIM) -> np.ndarray:
    """``[d, k]``: the top ``k`` principal directions of the train-split encoded latents (fitted
    on training frames only, before any model; model-free)."""
    values, vectors = np.linalg.eigh(encoded.centered())
    order = np.argsort(values)[::-1][:k]
    return np.ascontiguousarray(vectors[:, order])


class SessionMoments:
    """Per-session first and second moments of ``(rows - shift) @ basis``."""

    def __init__(self, shift, basis):
        self.shift = np.asarray(shift, np.float64)
        self.basis = np.asarray(basis, np.float64)
        self.sessions: dict = {}

    def add(self, rows, sessions) -> None:
        y = (np.asarray(rows, np.float64) - self.shift) @ self.basis
        sessions = np.asarray(sessions)
        if len(sessions) != len(y):
            raise ContractError("one session label per row")
        for label in np.unique(sessions):
            part = y[sessions == label]
            n, s1, s2 = self.sessions.get(label, (0, 0.0, 0.0))
            self.sessions[label] = (n + len(part), s1 + part.sum(0), s2 + part.T @ part)

    def merge(self, other: SessionMoments) -> SessionMoments:
        out = SessionMoments(self.shift, self.basis)
        for source in (self, other):
            for label, (n, s1, s2) in source.sessions.items():
                m, t1, t2 = out.sessions.get(label, (0, 0.0, 0.0))
                out.sessions[label] = (m + n, t1 + s1, t2 + s2)
        return out

    def arrays(self, labels):
        n = np.array([self.sessions[s][0] for s in labels], np.float64)
        s1 = np.stack([self.sessions[s][1] for s in labels])
        s2 = np.stack([self.sessions[s][2] for s in labels]).reshape(len(labels), -1)
        return n, s1, s2


def _rank_from_sums(n, s1, s2) -> float:
    k = len(s1)
    m = s1 / n
    energy = np.clip(np.linalg.eigvalsh(s2.reshape(k, k) - n * np.outer(m, m)), 0.0, None)
    total = energy.sum()
    if not total > 0:
        return float("nan")
    p = energy[energy > 0] / total
    return float(np.exp(-(p * np.log(p)).sum()))


def comparative_rank(
    w: SessionMoments,
    n: SessionMoments,
    e: SessionMoments,
    *,
    resamples=None,
    seed=COMPARATIVE_SEED,
    block=100,
) -> dict:
    """(rank ratio W - rank ratio N) in the projected basis, with a session-clustered percentile
    interval (the same resampled sessions for all three)."""
    labels = sorted(e.sessions)
    if sorted(w.sessions) != labels or sorted(n.sessions) != labels:
        raise ContractError("W, N and the encoded targets must cover the same sessions")
    resamples = COMPARATIVE_RESAMPLES if resamples is None else resamples
    idx = np.random.default_rng(seed).integers(0, len(labels), size=(resamples, len(labels)))
    counts = np.stack([np.bincount(row, minlength=len(labels)) for row in idx]).astype(float)
    parts = {name: m.arrays(labels) for name, m in (("W", w), ("N", n), ("E", e))}

    def ranks(weights):
        out = {}
        for name, (cn, c1, c2) in parts.items():
            out[name] = [
                _rank_from_sums(wt @ cn, wt @ c1, wt @ c2) for wt in np.atleast_2d(weights)
            ]
        return {k: np.asarray(v) for k, v in out.items()}

    point = ranks(np.ones(len(labels)))
    diffs, undefined = [], 0
    for lo in range(0, resamples, block):
        r = ranks(counts[lo : lo + block])
        d = r["W"] / r["E"] - r["N"] / r["E"]
        undefined += int((~np.isfinite(d)).sum())
        diffs.append(d)
    diffs = np.concatenate(diffs)
    finite = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
    return {
        "projected_rank_W": float(point["W"][0]),
        "projected_rank_N": float(point["N"][0]),
        "projected_rank_encoded": float(point["E"][0]),
        "difference": float(point["W"][0] / point["E"][0] - point["N"][0] / point["E"][0]),
        "ci95": [float(lo), float(hi)],
        "resamples": int(resamples),
        "sessions": len(labels),
        "undefined_resamples": undefined,
        "basis_dim": int(w.basis.shape[1]),
    }


def truncated_spectrum_statistics(predicted: Moments, encoded: Moments, k: int) -> dict:
    """A synthetic collapse control: ``predicted`` projected onto its own top ``k`` principal
    directions (mean kept). Returns the full-width G1 quantities of that truncated predictor."""
    values, vectors = np.linalg.eigh(predicted.centered() / predicted.n)
    order = np.argsort(values)[::-1][:k]
    lam = np.clip(values[order], 0.0, None)
    v = vectors[:, order]
    p_std = np.sqrt(np.clip((v * v) @ lam, 0.0, None))
    p = lam[lam > 0] / lam.sum() if lam.sum() > 0 else np.array([1.0])
    p_rank = float(np.exp(-(p * np.log(p)).sum())) if lam.sum() > 0 else 0.0
    e_std, e_rank = encoded.std(), encoded.effective_rank()
    return {
        "k": int(k),
        "std_ratio": float(p_std.mean() / e_std.mean()),
        "predicted_collapsed_fraction": float((p_std < 0.01).mean()),
        "effective_rank_ratio": p_rank / e_rank if e_rank > 0 else 0.0,
        "predicted_effective_rank": p_rank,
    }


g2_passes = ld.g2_passes  # upper 95 % bound of W / copy-last <= 0.8
g3_passes = ld.g3_passes  # upper 95 % bound of W / N < 1.0
g4_passes = ld.g4_passes  # lower 95 % bounds of wrong / W and zero / W >= 1.10
g5_passes = ld.g5_passes  # the TASK-059 T1 bar and <= 0.5 cm over the encoded target


def ceiling_passes(encoded_readability: dict) -> bool:
    """The encoded grid itself meets the TASK-059 T1 bar (median <= 1.5 cm, ratio to B-occ upper
    bound <= 0.6). Not a gate: it decides whether G5 can test prediction at all (row CEILING)."""
    t = THRESHOLDS
    return bool(
        encoded_readability["median_cm"] <= t["G5_max_median_cm"]
        and encoded_readability["ratio_to_B_occ"]["ci95"][1] <= t["G5_max_ratio_to_B_occ_upper"]
    )


def seed_gates(per_horizon: dict) -> dict:
    """``per_horizon[h][gate] -> bool``: a gate passes only at every gated horizon."""
    out = ld.seed_gates(per_horizon)
    out["latent"] = bool(all(out["gates"][g] for g in LATENT_GATES))
    return out


# ----- decision (protocol section 10) ------------------------------------------------------------
def decide(*, void: bool, seeds: dict | None = None, ceiling: dict | None = None) -> dict:
    """First matching row.

    ``seeds``: ``{seed: seed_gates(...)}``. ``ceiling``: ``{h: bool}`` for the gated horizons,
    whether the encoded grid meets the T1 bar. Rows, in order: WM-TOK-DYNAMICS (every seed
    passes G1-G5); WM-TOK-UNSTABLE (one or two pass); WM-TOK-CEILING (none passes, G1-G4 pass on
    at least two seeds, and the encoded grid misses the T1 bar at a gated horizon);
    WM-TOK-APPLE-LOST (none passes, G1-G4 on at least two seeds); WM-TOK-COLLAPSE (none passes,
    G2-G4 on at least two seeds: G1 is what fails); WM-TOK-NO-DYNAMICS (otherwise)."""
    if void:
        return {"outcome": "V", "abandonment_clause_fires": False}
    if seeds is None or set(seeds) != set(MODEL_SEEDS):
        raise ContractError(f"a decision needs every model seed {MODEL_SEEDS}")
    if ceiling is None or set(ceiling) != set(GATED_HORIZONS):
        raise ContractError(f"a decision needs the ceiling at {GATED_HORIZONS}")
    passing = sorted(s for s, g in seeds.items() if g["passes"])
    dynamics = sorted(s for s, g in seeds.items() if g["dynamics"])
    latent = sorted(s for s, g in seeds.items() if g["latent"])
    ceiling_ok = bool(all(ceiling[h] for h in GATED_HORIZONS))
    if len(passing) == len(MODEL_SEEDS):
        row = "WM-TOK-DYNAMICS"
    elif passing:
        row = "WM-TOK-UNSTABLE"
    elif len(dynamics) >= 2 and not ceiling_ok:
        row = "WM-TOK-CEILING"
    elif len(dynamics) >= 2:
        row = "WM-TOK-APPLE-LOST"
    elif len(latent) >= 2:
        row = "WM-TOK-COLLAPSE"
    else:
        row = "WM-TOK-NO-DYNAMICS"
    return {
        "outcome": row,
        "passing_seeds": passing,
        "dynamics_passing_seeds": dynamics,
        "latent_gates_passing_seeds": latent,
        "ceiling_meets_t1": ceiling_ok,
        "abandonment_clause_fires": abandonment_fires(row),
    }


def abandonment_fires(outcome: str) -> bool:
    if outcome not in ROWS:
        raise ContractError(f"unknown outcome {outcome!r}")
    return outcome in CLAUSE_ROWS
