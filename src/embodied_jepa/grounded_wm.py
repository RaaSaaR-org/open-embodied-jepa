"""TASK-087 (Phase 2): robot-grounded LeWM on ``play-v1``, offline -- constants, tables,
statistics and the row (protocol ``docs/experiments/grounded_lewm_v1.md``).

NumPy only at import. The torch model is ``grounded_wm_model``; featurisation and the feature
stores are ``grounded_wm_data``. Nothing here is registered or imported by ``embodied_jepa``.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from embodied_jepa.contracts import ContractError

PROTOCOL = "docs/experiments/grounded_lewm_v1.md"
TASK = "TASK-087"

# ----- data (§2) ----------------------------------------------------------------------------------
CORPUS = "data/play-v1"
CORPUS_SHARDS = 32
# sha256 of data/play-v1/corpus.json in TASK-085's evidence (task085-run/SHA256SUMS)
CORPUS_JSON_SHA256 = "06551467725c6dba42e1fcc48576307be8f1fab3e736be5d2268a969d5cb7596"
SPLIT_COUNTS = {"train": 2874, "val": 160, "test": 160}
CAMERA = "onboard_rgb"
# The right arm's 7 and the right Dex3's 7 joints (names in g1_dex3_proprio_v0), positions then
# velocities: 28-D.
STATE_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)
STATE_NAMES = tuple(f"{j}.position" for j in STATE_JOINTS) + tuple(
    f"{j}.velocity" for j in STATE_JOINTS
)
STATE_DIM = len(STATE_NAMES)  # 28
STATE_FLOOR_STD = 0.01
STATE_CLIP = 5.0
PALM_SLICE = slice(0, 3)  # right palm position in the sidecar's ``palms`` (pelvis frame)
PLAYED_ACTION_DIMS = (6, 7, 8, 9, 10, 11, 13)  # right arm 6-D and right grasp
ACTION_DIM = 14

# ----- latent (§3) --------------------------------------------------------------------------------
TOKEN_GRID = 4
TOKENS = TOKEN_GRID * TOKEN_GRID  # 16
TOKEN_WIDTH_RAW = 384
RAW_DIM = TOKENS * TOKEN_WIDTH_RAW  # 6 144
K = 192  # kept components per token
FEATURE_STD_FLOOR = 1e-6
FIT_SAMPLE_MODULUS = 4  # train episodes whose seed is divisible by 4
FEATURE_BATCH = 64
FEATURE_DEVICE = "cuda"

# ----- arms (§4) ----------------------------------------------------------------------------------
ARMS = ("P", "S", "G", "C", "I", "N")
ELIGIBLE = ("S", "G")
ARM_SPEC = {
    # state_token: a 17th token; state_head: G's joint-change head; inverse: the ID loss;
    # fusion: C's state embedding added to the input tokens; readout: C's palm readout;
    # zero_actions: N.
    "P": {},
    "S": {"state_token": True},
    "G": {"state_token": True, "state_head": True, "inverse": True},
    "C": {"fusion": True, "readout": True},
    "I": {"inverse": True},
    "N": {"zero_actions": True},
}
ARM_FLAGS = ("state_token", "state_head", "inverse", "fusion", "readout", "zero_actions")
MODEL_CONFIG = {
    "width": K,
    "depth": 4,
    "heads": 4,
    "head_dim": 48,
    "mlp_dim": 768,
    "proj_hidden": 768,
    "state_hidden": 256,
    "inverse_hidden": 512,
    "readout_hidden": 256,
    "max_horizon": 16,
}
LOSS_WEIGHTS = {"multistep": 1.0, "joint_change": 1.0, "inverse": 1.0, "readout": 1.0}

# ----- training (§5, §6) --------------------------------------------------------------------------
TRAIN_HORIZON = 8
WINDOW_FRAMES = TRAIN_HORIZON + 1
BATCH = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
MODEL_SEEDS = (87100, 87101, 87102)
DEBUG_SEEDS = range(87900, 88000)
PROBE_SEED = 87900
PROBE_UPDATES = 60_000
PROBE_SELECT_EVERY = 2_000
SELECTIONS = 20
SELECTION_TOLERANCE = 0.01
U_MIN, U_MAX, U_STEP = 20_000, 60_000, 5_000
BUDGET_ESCALATE_HOURS = 14.0
CAP_FACTOR = 3.0

SALTS = {"wrong": 8711, "candidates": 8712, "bootstrap": 8713, "sampler": 8714}

# ----- evaluation (§7, §8) ------------------------------------------------------------------------
ROOT_STRIDE = 10
EVAL_HORIZON = 16
HORIZONS = (1, 2, 4, 8, 16)
GATE_HORIZONS = (1, 4, 8)
CANDIDATES = 16
BOOTSTRAP_RESAMPLES = 10_000
GATE_LEVEL = 0.975
REPORT_LEVEL = 0.95
TIE_TOLERANCE = 1e-6

ROWS = ("P2-VOID", "P2-PLAIN-INVALID", "P2-PASS", "P2-SHORT", "P2-FAIL")


def check_arm(arm: str) -> dict:
    if arm not in ARM_SPEC:
        raise ContractError(f"unknown arm {arm!r}; one of {ARMS}")
    return {flag: bool(ARM_SPEC[arm].get(flag, False)) for flag in ARM_FLAGS}


def check_model_seed(seed: int, *, debug: bool = False) -> int:
    seed = int(seed)
    allowed = DEBUG_SEEDS if debug else MODEL_SEEDS
    if seed not in allowed:
        kind = "debug" if debug else "model"
        raise ContractError(f"{seed} is not a {kind} seed of {TASK}")
    return seed


def sampler_seed(model_seed: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([SALTS["sampler"], int(model_seed)])


def budget_from_probe(curve, seconds_per_update: float) -> dict:
    """§6.1: u* is the first point within 1 % of the probe's minimum;
    U = max(20 000, min(60 000, ceil(1.5 u* / 5 000) * 5 000)); the cap and the escalation."""
    points = [(int(u), float(v)) for u, v in curve]
    if not points:
        raise ContractError("the probe curve is empty")
    floor = min(v for _, v in points)
    u_star = next(u for u, v in points if v <= floor + abs(floor) * SELECTION_TOLERANCE)
    raw = int(np.ceil(1.5 * u_star / U_STEP)) * U_STEP
    updates = max(U_MIN, min(U_MAX, raw))
    hours = len(ARMS) * len(MODEL_SEEDS) * updates * float(seconds_per_update) / 3600.0
    return {
        "u_star": int(u_star),
        "updates": int(updates),
        "select_every": int(updates // SELECTIONS),
        "cap_seconds": float(CAP_FACTOR * float(seconds_per_update) * updates),
        "projected_gpu_hours": float(hours),
        "escalate": bool(hours > BUDGET_ESCALATE_HOURS),
        "seconds_per_update": float(seconds_per_update),
    }


# ----- episodes, windows and roots ----------------------------------------------------------------
def window_starts(frames_per_episode, offsets, horizon: int = TRAIN_HORIZON) -> np.ndarray:
    """Global frame indices at which a window of ``horizon`` commands (``horizon + 1`` frames)
    starts and stays inside its episode."""
    out = []
    for n, o in zip(frames_per_episode, offsets, strict=True):
        if n >= horizon + 1:
            out.append(int(o) + np.arange(int(n) - horizon, dtype=np.int64))
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def roots(frames_per_episode, offsets, *, stride: int = ROOT_STRIDE, horizon: int = EVAL_HORIZON):
    """(global start index, episode index) of every root: starts 0, stride, ... with
    start + horizon <= T (T = frames - 1 commands)."""
    starts, episode = [], []
    for e, (n, o) in enumerate(zip(frames_per_episode, offsets, strict=True)):
        commands = int(n) - 1
        local = np.arange(0, commands - horizon + 1, stride, dtype=np.int64)
        starts.append(int(o) + local)
        episode.append(np.full(len(local), e, np.int64))
    return np.concatenate(starts), np.concatenate(episode)


def wrong_table(episode: np.ndarray) -> np.ndarray:
    """For each root, one root of another episode (salt 8711), drawn once."""
    rng = np.random.default_rng(SALTS["wrong"])
    n = len(episode)
    out = np.empty(n, np.int64)
    for i in range(n):
        while True:
            j = int(rng.integers(0, n))
            if episode[j] != episode[i]:
                out[i] = j
                break
    return out


def candidate_table(episode: np.ndarray, k: int = CANDIDATES) -> np.ndarray:
    """Row i: root i, then k - 1 distinct roots of other episodes (salt 8712), drawn once."""
    rng = np.random.default_rng(SALTS["candidates"])
    n = len(episode)
    table = np.empty((n, k), np.int64)
    every = np.arange(n)
    for i in range(n):
        others = every[episode != episode[i]]
        table[i, 0] = i
        table[i, 1:] = rng.choice(others, size=k - 1, replace=False)
    return table


def table_sha256(a) -> str:
    a = np.ascontiguousarray(a)
    digest = hashlib.sha256()
    digest.update(str(a.dtype).encode())
    digest.update(json.dumps(list(a.shape)).encode())
    digest.update(memoryview(a).cast("B"))
    return digest.hexdigest()


# ----- statistics ---------------------------------------------------------------------------------
def latent_error(predicted, target) -> np.ndarray:
    """Mean squared difference over every latent value, per row (float64)."""
    d = np.asarray(predicted, np.float64) - np.asarray(target, np.float64)
    return (d * d).reshape(len(d), -1).mean(1)


def true_rank(scores) -> np.ndarray:
    """Rank of column 0 among the candidates' scores (lower is better; ties count against)."""
    scores = np.asarray(scores, np.float64)
    return (scores[:, 1:] <= scores[:, :1]).sum(1)


def tie_share(commands, table, h: int) -> float:
    """Share of roots where some other candidate's first h commands equal the root's own."""
    commands = np.asarray(commands, np.float64)
    own = commands[table[:, :1], :h]
    other = commands[table[:, 1:], :h]
    same = (np.abs(other - own) <= TIE_TOLERANCE).all(axis=(2, 3))
    return float(same.any(1).mean())


def cluster_bootstrap(episode: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES) -> np.ndarray:
    """Weights ``[resamples, roots]``: each resample draws the episodes with replacement (salt
    8713); a root's weight is its episode's draw count."""
    episode = np.asarray(episode)
    ids, inverse = np.unique(episode, return_inverse=True)
    rng = np.random.default_rng(SALTS["bootstrap"])
    counts = np.empty((resamples, len(ids)), np.float64)
    for b in range(resamples):
        counts[b] = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
    return counts[:, inverse]


def _interval(boot, level: float) -> list[float]:
    alpha = (1.0 - float(level)) / 2.0
    lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
    return [float(lo), float(hi)]


def ratio_stat(num, den, weights, level: float) -> dict:
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    boot = (weights @ num) / (weights @ den)
    return {"value": float(num.sum() / den.sum()), "ci": _interval(boot, level), "level": level}


def mean_stat(values, weights, level: float) -> dict:
    values = np.asarray(values, np.float64)
    boot = (weights @ values) / weights.sum(1)
    return {"value": float(values.mean()), "ci": _interval(boot, level), "level": level}


def ratio_difference(num_x, den_x, num_p, den_p, weights, level: float) -> dict:
    """(Σ num_x / Σ den_x) − (Σ num_p / Σ den_p), paired on the same resamples."""
    a = (weights @ np.asarray(num_x, np.float64)) / (weights @ np.asarray(den_x, np.float64))
    b = (weights @ np.asarray(num_p, np.float64)) / (weights @ np.asarray(den_p, np.float64))
    value = np.sum(num_x) / np.sum(den_x) - np.sum(num_p) / np.sum(den_p)
    return {"value": float(value), "ci": _interval(a - b, level), "level": level}


def mean_difference(x, p, weights, level: float) -> dict:
    d = np.asarray(x, np.float64) - np.asarray(p, np.float64)
    boot = (weights @ d) / weights.sum(1)
    return {"value": float(d.mean()), "ci": _interval(boot, level), "level": level}


def effective_rank(x) -> float:
    """exp of the entropy of the normalised singular values of the centred rows."""
    x = np.asarray(x, np.float64).reshape(len(x), -1)
    x = x - x.mean(0)
    s = np.linalg.svd(x, compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def per_root_errors(predict, start, commands, targets, states, wrong, table, horizons=HORIZONS):
    """Per-root errors for one model.

    ``predict(start, commands, states, horizons) -> {h: [N, ...]}`` (the roll-out's prediction
    at each requested horizon). ``targets[h]`` is the encoded latent h steps after each root.
    Returns ``({h: {"true", "wrong", "zero", "copy", "rank"}}, true predictions)``."""
    hmax = max(horizons)
    commands = np.asarray(commands[:, :hmax], np.float32)
    out = {h: {} for h in horizons}
    true = predict(start, commands, states, horizons)
    bad = predict(start, commands[wrong], states, horizons)
    zero = predict(start, np.zeros_like(commands), states, horizons)
    for h in horizons:
        out[h]["true"] = latent_error(true[h], targets[h])
        out[h]["wrong"] = latent_error(bad[h], targets[h])
        out[h]["zero"] = latent_error(zero[h], targets[h])
        out[h]["copy"] = latent_error(start, targets[h])
    del bad, zero
    scores = {h: np.empty((len(start), table.shape[1])) for h in horizons}
    for h in horizons:
        scores[h][:, 0] = out[h]["true"]
    for j in range(1, table.shape[1]):
        p = predict(start, commands[table[:, j]], states, horizons)
        for h in horizons:
            scores[h][:, j] = latent_error(p[h], targets[h])
    for h in horizons:
        out[h]["rank"] = true_rank(scores[h])
    return out, true


def summarise(errors: dict, weights, level: float = REPORT_LEVEL) -> dict:
    """Per-model statistics at every horizon (§7)."""
    out = {}
    for h, e in errors.items():
        rank = np.asarray(e["rank"])
        out[str(h)] = {
            "sens": ratio_stat(e["wrong"], e["true"], weights, level),
            "zero_over_true": ratio_stat(e["zero"], e["true"], weights, level),
            "copy_ratio": ratio_stat(e["true"], e["copy"], weights, level),
            "top1": mean_stat(rank == 0, weights, level),
            "normalised_rank": mean_stat(rank / (CANDIDATES - 1), weights, level),
            "acc": float(np.sum(e["true"])),
            "sum_wrong": float(np.sum(e["wrong"])),
            "separation": float(np.sum(e["wrong"]) - np.sum(e["true"])),
            "mean_true_error": float(np.mean(e["true"])),
            "mean_copy_error": float(np.mean(e["copy"])),
        }
    return out


def compare(x: dict, p: dict, weights, level: float = GATE_LEVEL) -> dict:
    """X against P of the same seed at every horizon: Δsens, Δtop1, acc ratio, X's copy ratio."""
    out = {}
    for h in x:
        ex, ep = x[h], p[h]
        out[str(h)] = {
            "d_sens": ratio_difference(
                ex["wrong"], ex["true"], ep["wrong"], ep["true"], weights, level
            ),
            "d_top1": mean_difference(ex["rank"] == 0, ep["rank"] == 0, weights, level),
            "acc_ratio": ratio_stat(ex["true"], ep["true"], weights, level),
            "copy_ratio": ratio_stat(ex["true"], ex["copy"], weights, level),
        }
    return out


def delta_from_val(val_true: dict, weights) -> dict:
    """§6.4: δ_h = max over the six ordered pairs (i, j) of P's seeds of the upper 97.5 %
    cluster-bootstrap bound of acc_h(P_i) / acc_h(P_j) on val, minus 1 (0 if negative).
    ``val_true[seed][h]`` are P's per-root true-command errors on the val roots."""
    out = {}
    for h in GATE_HORIZONS:
        bounds = []
        for i in MODEL_SEEDS:
            for j in MODEL_SEEDS:
                if i == j:
                    continue
                stat = ratio_stat(val_true[i][h], val_true[j][h], weights, GATE_LEVEL)
                bounds.append(stat["ci"][1])
        out[str(h)] = max(0.0, max(bounds) - 1.0)
    return out


def criteria_met(comparison: dict, delta: dict, h: int) -> dict:
    c = comparison[str(h)]
    met = {
        "a_sensitivity": c["d_sens"]["ci"][0] > 0,
        "b_ranking": c["d_top1"]["ci"][0] > 0,
        "c_accuracy": c["acc_ratio"]["ci"][1] <= 1.0 + float(delta[str(h)]),
        "d_dynamics": c["copy_ratio"]["ci"][1] < 1.0,
    }
    met["all"] = all(met.values())
    return met


def plain_valid(plain: dict, plain_vs_n: dict) -> dict:
    """Row 2's checks for one seed: P's 95 % copy-ratio upper bound < 1 at h = 1, 4, 8; at h = 8
    P / N accuracy upper bound < 1 and P's sens lower bound > 1."""
    reasons = []
    for h in GATE_HORIZONS:
        if plain[str(h)]["copy_ratio"]["ci"][1] >= 1.0:
            reasons.append(f"P copy ratio at h={h}")
    if plain_vs_n["8"]["ci"][1] >= 1.0:
        reasons.append("P / N accuracy at h=8")
    if plain["8"]["sens"]["ci"][0] <= 1.0:
        reasons.append("P sens at h=8")
    return {"valid": not reasons, "reasons": reasons}


def decide(*, void_reasons, plain_checks: dict, criteria: dict) -> dict:
    """The row (first match). ``plain_checks[seed]`` from :func:`plain_valid`;
    ``criteria[arm][seed][h]`` from :func:`criteria_met` for the eligible arms."""
    if void_reasons:
        return {"row": "P2-VOID", "reasons": list(void_reasons)}
    invalid = {s: c["reasons"] for s, c in plain_checks.items() if not c["valid"]}
    if invalid:
        return {"row": "P2-PLAIN-INVALID", "reasons": invalid}
    passing = [
        arm
        for arm in ELIGIBLE
        if all(criteria[arm][str(s)][str(h)]["all"] for s in MODEL_SEEDS for h in GATE_HORIZONS)
    ]
    if passing:
        return {"row": "P2-PASS", "arms": passing}
    short = [arm for arm in ELIGIBLE if all(criteria[arm][str(s)]["1"]["all"] for s in MODEL_SEEDS)]
    if short:
        return {"row": "P2-SHORT", "arms": short}
    return {"row": "P2-FAIL", "arms": []}


def phase3_model(row: dict, d_top1_h1: dict) -> str | None:
    """§8: the passing arm; with both, the larger mean Δtop1 at h = 1 over seeds."""
    if row["row"] != "P2-PASS":
        return None
    arms = row["arms"]
    if len(arms) == 1:
        return arms[0]
    return max(arms, key=lambda a: float(np.mean([d_top1_h1[a][str(s)] for s in MODEL_SEEDS])))
