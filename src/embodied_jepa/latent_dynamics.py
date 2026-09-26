"""TASK-065 (``apple_latent_dynamics_v1``): does a LeWM predictor on frozen DINOv2 CLS latents
learn real dynamics on ``apple-look-v1``, and does the apple stay readable through prediction?

Protocol ``docs/experiments/apple_latent_dynamics_v1.md``. This module fixes in code, before any
predictor is trained on the corpus, what the protocol preregisters: the data partition (two
cross-fitting halves of the 170 train sessions), the predictor configuration, the budget, the
evaluation windows, the statistics, the five gate families, the per-seed rule, the outcome rows
and the abandonment clause. NumPy only: it imports without torch or MuJoCo. The runner lives in
``scripts/train_apple_latent_dynamics.py``.

**This is a world-model test only.** It trains an action-conditioned latent predictor and measures
it offline. It preregisters no control formulation: CEM over the world-model cost was abandoned at
TASK-054 and the behaviour-cloning line at TASK-057, and both clauses still hold. **Learned
Apple->Plate is still 0 successes.** The corpus comes from the privileged scripted collector.

The arms (per model seed, per cross-fitting half):

* ``W`` -- the LeWM adapter's pinned upstream predictor (``ARPredictor`` + ``Embedder`` +
  ``pred_proj``, history one) on the frozen DINOv2 CLS latent
  (``models.frozen_encoder.frozen_encoder_model(BACKEND)``), trained on true actions;
* ``N`` -- the same model trained and evaluated with every action replaced by zero (the no-action
  baseline);
* copy-last -- the start latent repeated (no training).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from embodied_jepa.contracts import ContractError

PROTOCOL = "apple_latent_dynamics_v1"
TASK = "TASK-065"

# ----- data -----------------------------------------------------------------------------------
DATASET = "data/apple-look-v1"
DATASET_MANIFEST_SHA256 = "81d760d1f5a61834b4f37c3daf12ad1a7175e57fe99fe11d5fb4cecbff8edb64"
EXPECTED_SESSIONS = {"train": 170, "val": 20, "test": 10}
EXPECTED_EPISODES = {"train": 679, "val": 80, "test": 40}
READ_SPLITS = ("train", "val")  # the test split is never decoded
DECISION_FRAME = 8  # the post-look frame of every root episode (TASK-064)
CAMERA = "onboard_rgb"

# Two cross-fitting halves of the 170 train sessions (root + its branches stay together).
HALVES = ("A", "B")
HALF_SEED = 65

# ----- the predictor ---------------------------------------------------------------------------
BACKEND = "leworldmodel"  # the one-line swap: "native_jepa" runs the same test on native code
FROZEN_ENCODER = "dinov2_small_cls"
LATENT_DIM = 384
# Everything not listed is the adapter's declared default (docs/MODELS.md): hidden_dim 128,
# predictor depth 2, 2 heads of 24, lr 3e-4, weight decay 1e-4, clip 1.0, multistep_weight 1.0,
# history one. SIGReg regularises a trainable encoder's embedding; the encoder is frozen here,
# so its weight is 0.
MODEL_CONFIG = {
    "camera": CAMERA,
    "frozen_encoder": FROZEN_ENCODER,
    "latent_dim": LATENT_DIM,
    "sigreg_weight": 0.0,
    "max_horizon": 64,
}
ARMS = ("W", "N")  # W: true actions; N: every action zero, in training and in evaluation
MODEL_SEEDS = (0, 1, 2)
TRAIN_HORIZON = 16  # transitions per training window (17 frames)
BATCH_SIZE = 64
UPDATES = 10_000
SELECT_EVERY = 500  # val criterion after every 500 updates; the best checkpoint is kept
SAMPLER_SALT = 6500  # default_rng(SeedSequence([6500, seed, half index, arm index]))
DEVICE = "mps"  # training and prediction; features and statistics on CPU
FEATURE_DEVICE = "cpu"
FEATURE_THREADS = 6  # TASK-064's gate thread count, so the anchor is comparable

# ----- evaluation --------------------------------------------------------------------------------
EVAL_STRIDE = 4  # E-all and the val criterion: a window starts at every 4th frame
GATED_HORIZONS = (8, 16)
REPORTED_HORIZONS = (1, 2, 4, 8, 16)
EXTENDED_HORIZONS = (32, 64)  # beyond the training horizon; reported only
CLUSTER_BOOTSTRAP_SEED = 6501
CLUSTER_BOOTSTRAP_RESAMPLES = 10_000
SHUFFLE_SEED = 6502
METRIC_FLOOR_STD = 1e-3

# ----- gates (protocol §8), per seed, at each gated horizon ---------------------------------------
THRESHOLDS = {
    # G1 collapse (E-all): predicted latents against the encoded targets of the same windows
    "G1_max_collapsed_fraction": 0.05,
    "G1_min_effective_rank_ratio": 0.5,
    "G1_min_std_ratio": 0.5,
    # G2 beats copy-last (E-all): upper 95 % bound of MSE(W) / MSE(copy-last)
    "G2_max_ratio_upper": 0.8,
    # G3 beats no-action (E-all): upper 95 % bound of MSE(W) / MSE(N)
    "G3_max_ratio_upper_exclusive": 1.0,
    # G4 action sensitivity (E-all): lower 95 % bound of MSE(W, wrong actions) / MSE(W, true)
    "G4_min_ratio_lower": 1.10,
    # G5 apple readable through prediction (E-post, 170 cross-fitted train roots)
    "G5_max_median_cm": 1.5,  # TASK-059's T1 bar, unchanged
    "G5_max_ratio_to_B_occ_upper": 0.6,  # TASK-059's T1 bar, unchanged
    "G5_max_excess_over_encoded_upper_cm": 0.5,  # non-inferiority margin to the encoded target
}
GATES = ("G1", "G2", "G3", "G4", "G5")
DYNAMICS_GATES = ("G1", "G2", "G3", "G4")

# ----- budget ------------------------------------------------------------------------------------
PER_RUN_SECONDS = 1800.0  # one model: 10 000 updates plus its 20 val evaluations
FEATURE_SECONDS = 5400.0  # featurising every train + val frame
GLOBAL_WALL_SECONDS = 21_600.0

ROWS = ("V", "WM-DYNAMICS", "WM-UNSTABLE", "WM-APPLE-LOST", "WM-NO-DYNAMICS")


class GuardError(ContractError):
    """A protocol §9 guard failed: the run is void (V) and nothing in it is read."""


# ----- partition ---------------------------------------------------------------------------------
def session_halves(train_sessions) -> dict:
    """``{session: "A" | "B"}``: the sorted sessions, permuted by ``default_rng(65)``, the first
    half A. The same session never lands in both halves."""
    sessions = sorted(train_sessions)
    if len(set(sessions)) != len(sessions) or len(sessions) < 2:
        raise ContractError("halves need at least two distinct sessions")
    order = np.random.default_rng(HALF_SEED).permutation(len(sessions))
    cut = len(sessions) // 2
    return {sessions[i]: ("A" if rank < cut else "B") for rank, i in enumerate(order)}


def halves_sha256(halves: dict) -> str:
    blob = json.dumps(dict(sorted(halves.items())), separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def other(half: str) -> str:
    if half not in HALVES:
        raise ContractError(f"unknown half {half!r}")
    return "B" if half == "A" else "A"


def sampler_seed(seed: int, half: str, arm: str) -> np.random.SeedSequence:
    return np.random.SeedSequence([SAMPLER_SALT, int(seed), HALVES.index(half), ARMS.index(arm)])


def check_training_episodes(training: dict, sessions_of: dict, halves: dict, splits: dict) -> None:
    """G-split: each half model trains on exactly its half's train episodes, nothing else."""
    for half, episodes in training.items():
        if not episodes:
            raise GuardError(f"G-split: half {half} has no training episode")
        for episode in episodes:
            if splits.get(episode) != "train":
                raise GuardError(f"G-split: {episode} (split {splits.get(episode)}) in training")
            if halves.get(sessions_of[episode]) != half:
                raise GuardError(f"G-split: {episode} is not in half {half}")
    a, b = set(training.get("A", ())), set(training.get("B", ()))
    if a & b:
        raise GuardError("G-split: an episode trains both halves")
    expected = {e for e, s in splits.items() if s == "train"}
    if a | b != expected:
        raise GuardError("G-split: the halves do not cover the train split exactly")


# ----- windows -----------------------------------------------------------------------------------
def windows(transitions, horizon: int, stride: int) -> np.ndarray:
    """``[(episode index, start)]`` of every window of ``horizon`` transitions whose start is a
    multiple of ``stride``; ``transitions[e]`` is the number of valid actions of episode e."""
    out = [
        (e, s)
        for e, count in enumerate(transitions)
        for s in range(0, int(count) - horizon + 1, stride)
    ]
    return np.asarray(out, np.int64).reshape(-1, 2)


def training_windows(transitions, horizon: int = TRAIN_HORIZON) -> np.ndarray:
    """Every window (stride 1) a training batch samples from."""
    return windows(transitions, horizon, 1)


def cross_session_shuffle(sessions, rng) -> np.ndarray:
    """A permutation ``p`` with ``sessions[p[i]] != sessions[i]`` for every i.

    Sessions are laid out in a random order as contiguous blocks, and every position takes the
    window one maximal block length further on (cyclically), so no window keeps actions from its
    own session."""
    sessions = np.asarray(sessions)
    labels, inverse, counts = np.unique(sessions, return_inverse=True, return_counts=True)
    if len(labels) < 2 or counts.max() * 2 > len(sessions):
        raise ContractError("a cross-session shuffle needs no session holding half the windows")
    block_order = rng.permutation(len(labels))
    rank = np.empty(len(labels), np.int64)
    rank[block_order] = np.arange(len(labels))
    within = rng.permutation(len(sessions))
    layout = np.lexsort((within, rank[inverse]))  # positions -> window index
    shift = int(counts.max())
    perm = np.empty(len(sessions), np.int64)
    perm[layout] = layout[(np.arange(len(sessions)) + shift) % len(sessions)]
    if (sessions[perm] == sessions).any():
        raise ContractError("cross-session shuffle kept a session")  # unreachable by construction
    return perm


# ----- statistics --------------------------------------------------------------------------------
def normalized_sq_error(predicted, target, scale) -> np.ndarray:
    """Per-window mean over dimensions of ``((predicted - target) / scale)^2``."""
    d = (np.asarray(predicted, np.float64) - np.asarray(target, np.float64)) / np.asarray(
        scale, np.float64
    )
    return (d * d).mean(axis=-1)


def cluster_bootstrap_indices(n_clusters: int, *, seed=CLUSTER_BOOTSTRAP_SEED, resamples=None):
    resamples = CLUSTER_BOOTSTRAP_RESAMPLES if resamples is None else resamples
    return np.random.default_rng(seed).integers(0, n_clusters, size=(resamples, n_clusters))


def cluster_ratio(numerator, denominator, clusters, idx) -> dict:
    """``sum(num) / sum(den)`` over windows, with a percentile CI from resampling clusters
    (sessions) with replacement; ``idx`` indexes the sorted unique clusters."""
    num = np.asarray(numerator, np.float64)
    den = np.asarray(denominator, np.float64)
    labels, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    if idx.shape[1] != len(labels):
        raise ContractError("bootstrap indices do not match the number of clusters")
    num_c = np.bincount(inverse, weights=num, minlength=len(labels))
    den_c = np.bincount(inverse, weights=den, minlength=len(labels))
    base = float(den_c.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = num_c[idx].sum(1) / den_c[idx].sum(1)
    boot = np.where(np.isfinite(boot), boot, np.finfo(np.float64).max)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "ratio": None if base == 0.0 else float(num_c.sum() / base),
        "ci95": [float(lo), float(hi)],
        "numerator_mean": float(num.mean()),
        "denominator_mean": float(den.mean()),
        "windows": int(len(num)),
        "clusters": int(len(labels)),
    }


def effective_rank(values) -> float:
    x = np.asarray(values, np.float64)
    x = x - x.mean(0)
    energy = np.linalg.svd(x, compute_uv=False) ** 2
    total = energy.sum()
    if total <= 0:
        return 0.0
    p = energy[energy > 0] / total
    return float(np.exp(-(p * np.log(p)).sum()))


def collapse_statistics(predicted_normalized, encoded_normalized) -> dict:
    """G1's quantities on latents already divided by the metric scale."""
    p = np.asarray(predicted_normalized, np.float64)
    e = np.asarray(encoded_normalized, np.float64)
    p_std, e_std = p.std(0), e.std(0)
    p_rank, e_rank = effective_rank(p), effective_rank(e)
    return {
        "predicted_std_mean": float(p_std.mean()),
        "encoded_std_mean": float(e_std.mean()),
        "std_ratio": float(p_std.mean() / e_std.mean()) if e_std.mean() > 0 else 0.0,
        "predicted_collapsed_fraction": float((p_std < 0.01).mean()),
        "encoded_collapsed_fraction": float((e_std < 0.01).mean()),
        "predicted_effective_rank": p_rank,
        "encoded_effective_rank": e_rank,
        "effective_rank_ratio": p_rank / e_rank if e_rank > 0 else 0.0,
        "samples": int(len(p)),
    }


# ----- gates -------------------------------------------------------------------------------------
def g1_passes(stats: dict) -> bool:
    t = THRESHOLDS
    return bool(
        stats["predicted_collapsed_fraction"] <= t["G1_max_collapsed_fraction"]
        and stats["effective_rank_ratio"] >= t["G1_min_effective_rank_ratio"]
        and stats["std_ratio"] >= t["G1_min_std_ratio"]
    )


def g2_passes(ratio_w_over_copy: dict) -> bool:
    return bool(ratio_w_over_copy["ci95"][1] <= THRESHOLDS["G2_max_ratio_upper"])


def g3_passes(ratio_w_over_n: dict) -> bool:
    return bool(ratio_w_over_n["ci95"][1] < THRESHOLDS["G3_max_ratio_upper_exclusive"])


def g4_passes(ratio_shuffled_over_true: dict, ratio_zero_over_true: dict) -> bool:
    bar = THRESHOLDS["G4_min_ratio_lower"]
    return bool(
        ratio_shuffled_over_true["ci95"][0] >= bar and ratio_zero_over_true["ci95"][0] >= bar
    )


def g5_passes(readability: dict) -> bool:
    """``readability``: ``median_cm``, ``ratio_to_B_occ`` (with ``ci95``) and
    ``excess_over_encoded_cm`` (paired median difference W - encoded, with ``ci95``)."""
    t = THRESHOLDS
    return bool(
        readability["median_cm"] <= t["G5_max_median_cm"]
        and readability["ratio_to_B_occ"]["ci95"][1] <= t["G5_max_ratio_to_B_occ_upper"]
        and readability["excess_over_encoded_cm"]["ci95"][1]
        <= t["G5_max_excess_over_encoded_upper_cm"]
    )


def seed_gates(per_horizon: dict) -> dict:
    """``per_horizon[h][gate] -> bool`` for h in the gated horizons -> each gate passes only if it
    passes at every gated horizon; the seed passes only if every gate does."""
    if set(per_horizon) != set(GATED_HORIZONS):
        raise ContractError(f"gates need exactly the horizons {GATED_HORIZONS}")
    gates = {g: bool(all(per_horizon[h][g] for h in GATED_HORIZONS)) for g in GATES}
    return {
        "gates": gates,
        "dynamics": bool(all(gates[g] for g in DYNAMICS_GATES)),
        "passes": bool(all(gates.values())),
    }


# ----- decision (protocol §10) -------------------------------------------------------------------
def decide(*, void: bool, seeds: dict | None = None) -> dict:
    """First matching row.

    ``seeds``: ``{seed: seed_gates(...)}`` for every model seed. WM-DYNAMICS needs every seed to
    pass; WM-UNSTABLE is one or two passing seeds; with none passing, WM-APPLE-LOST if the
    dynamics gates G1-G4 pass on at least two seeds (the apple is what fails), otherwise
    WM-NO-DYNAMICS. The abandonment clause fires on WM-APPLE-LOST and WM-NO-DYNAMICS."""
    if void:
        return {"outcome": "V", "abandonment_clause_fires": False}
    if seeds is None or set(seeds) != set(MODEL_SEEDS):
        raise ContractError(f"a decision needs every model seed {MODEL_SEEDS}")
    passing = sorted(s for s, g in seeds.items() if g["passes"])
    dynamics = sorted(s for s, g in seeds.items() if g["dynamics"])
    if len(passing) == len(MODEL_SEEDS):
        row = "WM-DYNAMICS"
    elif passing:
        row = "WM-UNSTABLE"
    elif len(dynamics) >= 2:
        row = "WM-APPLE-LOST"
    else:
        row = "WM-NO-DYNAMICS"
    return {
        "outcome": row,
        "passing_seeds": passing,
        "dynamics_passing_seeds": dynamics,
        "abandonment_clause_fires": abandonment_fires(row),
    }


def abandonment_fires(outcome: str) -> bool:
    if outcome not in ROWS:
        raise ContractError(f"unknown outcome {outcome!r}")
    return outcome in ("WM-APPLE-LOST", "WM-NO-DYNAMICS")
