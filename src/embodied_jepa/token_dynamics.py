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


def bar_rule(pilot_ratios) -> dict:
    """A G1 bar: ``floor(100 * 0.5 * min(pilot ratios)) / 100``; escalate below 0.10."""
    values = [float(r) for r in pilot_ratios]
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ContractError("the bar rule needs finite nonnegative pilot ratios")
    reference = min(values)
    return {
        "reference": reference,
        "bar": math.floor(100 * BAR_FACTOR * reference + 1e-9) / 100,
        "escalate": bool(reference < BAR_ESCALATE_BELOW),
    }


# The calibration's results (protocol section 6.4), copied from its report; the freeze uses these.
CALIBRATION = {
    "report_sha256": None,
    "saturation_updates": None,
    "rank_ratio_reference": None,
    "std_ratio_reference": None,
}
UPDATES = None  # budget_rule(...)["updates"]
SELECT_EVERY = None  # UPDATES // 20: 20 selection points, as in TASK-065

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
    "G1_min_effective_rank_ratio": None,  # calibrated (bar_rule on the pilot rank ratios)
    "G1_min_std_ratio": None,  # calibrated (bar_rule on the pilot std ratios)
} | {k: v for k, v in ld.THRESHOLDS.items() if not k.startswith("G1")}
GATES = ld.GATES
DYNAMICS_GATES = ld.DYNAMICS_GATES
LATENT_GATES = ("G2", "G3", "G4")  # the latent-error gates, without G1

# ----- budget ------------------------------------------------------------------------------------
PER_RUN_SECONDS = None  # set from the calibrated budget and the measured update time
FEATURE_SECONDS = ld.FEATURE_SECONDS
GLOBAL_WALL_SECONDS = None

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


def g1_passes(stats: dict, thresholds: dict | None = None) -> bool:
    t = THRESHOLDS if thresholds is None else thresholds
    if t["G1_min_effective_rank_ratio"] is None or t["G1_min_std_ratio"] is None:
        raise ContractError("G1's bars are calibrated; they are not set yet")
    return bool(
        stats["predicted_collapsed_fraction"] <= t["G1_max_collapsed_fraction"]
        and stats["effective_rank_ratio"] >= t["G1_min_effective_rank_ratio"]
        and stats["std_ratio"] >= t["G1_min_std_ratio"]
    )


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
