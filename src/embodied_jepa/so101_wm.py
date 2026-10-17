"""TASK-100 (SO-101 S4): an offline world-model study on ``so101-play-v1`` -- constants, roots,
tables, readouts, statistics, gates and the row (``docs/experiments/so101_world_model_v1.md``).

NumPy only at import. The torch model is ``models/schema_frozen_tokens``; featurisation and the
feature stores are ``so101_wm_data``. Nothing here is registered or imported by ``embodied_jepa``,
and simulator truth (cube position, jaw angle) enters only the readouts and subset definitions.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from embodied_jepa.contracts import ContractError

PROTOCOL = "docs/experiments/so101_world_model_v1.md"
TASK = "TASK-100"

# ----- data (§2) ----------------------------------------------------------------------------------
CORPUS = "data/so101-play-v1"
CORPUS_MANIFEST = "benchmarks/manifests/so101-play-v1.json"
CORPUS_JSON_SHA256 = "c81571de2bebe76704129d2814f39959d341ec8cba169fc310f7c22d8dd45b69"
CORPUS_SHARDS = 30
SPLIT_COUNTS = {"train": 2700, "val": 150, "test": 150}
EPISODE_COMMANDS = 400
CAMERAS = ("wrist_rgb", "overview_rgb")  # the frame latent: wrist tokens, then overview tokens
IMAGE_SIZE = 224  # DINOv2's native input; no resize
ACTION_DIM = 6  # joint_delta_so101_v0
GRIPPER_CHANNEL = 5  # absolute gripper command
JAW_STATE_INDEX = 5  # gripper.position in so101_proprio_v0
CM_PER_M = 100.0
# D0: the train episodes of shards 00-02; D0-fit (00-01) fits, D0-eval (02) is applied to (§2, §10)
D0_SHARDS = (0, 1, 2)
D0_FIT_SHARDS = (0, 1)
D0_EVAL_SHARDS = (2,)
FIT_SAMPLE_MODULUS = 4  # the gated fit sample: train episodes whose seed is divisible by 4
# Stage 0's smoke (§10.6): val episodes stand in for val and for test to exercise the code path;
# their numbers are not results and set nothing.
SMOKE_VAL_SHARDS = (0, 1, 2, 3)
SMOKE_TEST_SHARDS = (4, 5, 6, 7)

# ----- latent (§3) --------------------------------------------------------------------------------
NATIVE_GRID = 16
RAW_WIDTH = 384
GRIDS = {
    # name: pooled grid per camera, kept channels (None: no projection), tokens, token width
    "g4": {"pool": 4, "k": None, "tokens": 2 * 4 * 4, "width": RAW_WIDTH},
    "g8": {"pool": 8, "k": 96, "tokens": 2 * 8 * 8, "width": 96},
}
LATENT_DIM = 12_288  # both grids: 32 x 384 = 128 x 96
FEATURE_STD_FLOOR = 1e-6
FEATURE_BATCH = 64
ENCODERS = ("pretrained", "random_init")

# ----- arms (§4) ----------------------------------------------------------------------------------
ARMS = ("L", "J", "N_L", "N_J", "R")
ELIGIBLE = ("L", "J")
NO_ACTION_TWIN = {"L": "N_L", "J": "N_J"}
ARM_SPEC = {
    "L": {"backend": "leworldmodel", "zero_actions": False, "encoder": "pretrained"},
    "J": {"backend": "native_jepa", "zero_actions": False, "encoder": "pretrained"},
    "N_L": {"backend": "leworldmodel", "zero_actions": True, "encoder": "pretrained"},
    "N_J": {"backend": "native_jepa", "zero_actions": True, "encoder": "pretrained"},
    "R": {"backend": "leworldmodel", "zero_actions": False, "encoder": "random_init"},
}
MAX_HORIZON = 32

# ----- training (§5) ------------------------------------------------------------------------------
TRAIN_HORIZON = 16
WINDOW_FRAMES = TRAIN_HORIZON + 1
BATCH = 64
MODEL_SEEDS = (100100, 100101, 100102)
DEV_SEEDS = range(100900, 101000)  # debug, calibration and probe seeds
CAL_L_SEED = 100900
CAL_N_SEED = 100911
DEBUG_SEEDS = range(100901, 100911)
PROBE_SEED = 100920
CAL_UPDATES = 20_000
PROBE_UPDATES = 60_000
PROBE_SELECT_EVERY = 2_000
SELECTIONS = 20
SELECTION_TOLERANCE = 0.01
U_MIN, U_MAX, U_STEP = 20_000, 60_000, 5_000
CAP_FACTOR = 3.0
EXTENSION_FACTOR = 1.5
SALTS = {"sampler": 10011, "wrong": 10012, "candidates": 10013, "bootstrap": 10014, "folds": 10015}

# ----- evaluation (§6, §7) ------------------------------------------------------------------------
ROOT_STRIDE = 10
ROOT_SPAN = 32  # start + 32 <= 400
HORIZONS = (1, 4, 8, 16, 32)
GATE_HORIZONS = (8, 16)
PLAN_HORIZONS = (4, 8, 16)
CRITERION_STEPS = 16  # the val criterion: steps 1-16
BOOTSTRAP_RESAMPLES = 10_000
GATE_LEVEL = 0.975
REPORT_LEVEL = 0.95
TIE_TOLERANCE = 1e-6
CANDIDATE_CLASSES = (
    ("executed", 1),
    ("sigma_0.1", 4),
    ("sigma_0.3", 4),
    ("gripper_0.5", 3),
    ("other_episode", 4),
)
CANDIDATES = sum(n for _, n in CANDIDATE_CLASSES)  # 16
RIDGE_PENALTIES = tuple(10.0**k for k in range(-1, 6))
RIDGE_FOLDS = 5
RENC_FRAME_STRIDE = 5
LABELS = ("cube", "jaw")
PHYSICAL_COST_JAW_CM_PER_RAD = 10.0  # §9: cube error in cm plus jaw error x 1 cm / 0.1 rad

G = {
    "G1_collapse_std_factor": 0.01,
    "G1_max_collapsed_fraction": 0.05,
    "G1_rank_floor": 0.10,
    "G1_std_floor": 0.25,
    "G1_truncations": (1, 2, 4),
    "G1_projected_directions": 256,
    "G2_max_ratio": 1.0,
    "G2_reported_allocation": 0.8,
    "G3_max_ratio": 1.0,
    "G4_min_sens": 1.10,
    "G5_bar_factor": 1.5,
    "G5_max_ratio": 1.0,
    "G6_max_drop": 0.02,
}
# §7.4's moving subsets; the threshold of a subset is halved once before the freeze if its share
# of D0-eval roots x 5 550 is below 300 at h = 8; a subset under 100 test roots closes INCONCLUSIVE
MOVING_THRESHOLDS = {"cube": 1.0, "jaw": 0.2}  # cm, rad
MOVING_MIN_PROJECTED = 300
MOVING_PROJECTED_ROOTS = 5_550
MOVING_MIN_TEST_ROOTS = 100
# §10.3's grid rule
GRID_RULE_RATIO = 0.85
GRID_RULE_MIN_G4_CM = 0.5
ABSOLUTE_REFERENCES_CM = (0.5, 3.0)

# ----- budget (§8) --------------------------------------------------------------------------------
HARD_CAP_GPU_H = 20.0
STAGE0_CAP_GPU_H = 4.0
VE_ESTIMATE_GPU_H = 1.5  # §8.2's upper estimate for V, F' and E
OVERRUN_FACTOR = 1.5
DISK_FLOOR_GIB = 10.0

ROWS = ("VOID", "INCONCLUSIVE", "WM-PASS", "WM-PARTIAL", "WM-FAIL")


def check_arm(arm: str) -> dict:
    if arm not in ARM_SPEC:
        raise ContractError(f"unknown arm {arm!r}; one of {ARMS}")
    return dict(ARM_SPEC[arm])


def check_seed(seed: int, *, development: bool = False) -> int:
    seed = int(seed)
    allowed = DEV_SEEDS if development else MODEL_SEEDS
    if seed not in allowed:
        kind = "development" if development else "model"
        raise ContractError(f"{seed} is not a {kind} seed of {TASK}")
    return seed


def sampler_seed(stream_seed: int) -> np.random.SeedSequence:
    """§5: model seed m's window sampler is ``default_rng(SeedSequence([10011, m]))``, shared by
    every arm of seed m. Calibration N_L draws calibration L's windows: its stream is 100900."""
    return np.random.SeedSequence([SALTS["sampler"], int(stream_seed)])


def sampler_stream_for(arm: str, seed: int) -> int:
    """The sampler stream of a job: the model seed, except calibration N_L (seed 100911), which
    shares calibration L's stream (100900) so that both draw the same windows in the same order."""
    return CAL_L_SEED if int(seed) == CAL_N_SEED else int(seed)


def is_fit_sample(seed: int) -> bool:
    return int(seed) % FIT_SAMPLE_MODULUS == 0


def shard_index(name: str) -> int:
    return int(str(name).rsplit("-", 1)[1])


# ----- windows, roots and tables (§5, §7.1, §7.2) -------------------------------------------------
def window_starts(frames_per_episode, offsets, horizon: int = TRAIN_HORIZON) -> np.ndarray:
    """Global start indices of every window of ``horizon`` commands inside one episode."""
    out = []
    for n, o in zip(frames_per_episode, offsets, strict=True):
        if int(n) >= horizon + 1:
            out.append(int(o) + np.arange(int(n) - horizon, dtype=np.int64))
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def roots(frames_per_episode, offsets, *, stride: int = ROOT_STRIDE, span: int = ROOT_SPAN):
    """(global start index, episode index) of every root: starts 0, 10, 20, ... with
    start + 32 <= T, T = frames - 1 commands (37 per 400-command episode)."""
    starts, episode = [], []
    for e, (n, o) in enumerate(zip(frames_per_episode, offsets, strict=True)):
        local = np.arange(0, int(n) - 1 - span + 1, stride, dtype=np.int64)
        starts.append(int(o) + local)
        episode.append(np.full(len(local), e, np.int64))
    if not starts:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(starts), np.concatenate(episode)


def wrong_table(episode) -> np.ndarray:
    """For each root, one root of another episode of the same split (salt 10012), drawn once."""
    episode = np.asarray(episode)
    if len(np.unique(episode)) < 2:
        raise ContractError("wrong commands need at least two episodes")
    rng = np.random.default_rng(SALTS["wrong"])
    out = np.empty(len(episode), np.int64)
    for i in range(len(episode)):
        while True:
            j = int(rng.integers(0, len(episode)))
            if episode[j] != episode[i]:
                out[i] = j
                break
    return out


def candidate_sequences(commands, episode, *, rng_salt: int = SALTS["candidates"]) -> dict:
    """§7.2: 16 command sequences per root ``[N, 16, 32, 6]`` (float32) and the class of each
    column. Column 0 is the executed sequence; 4 add N(0, 0.1^2) and 4 N(0, 0.3^2) to every
    channel; 3 perturb only the gripper channel by N(0, 0.5^2); all clipped to [-1, 1]; 4 are the
    executed sequences of distinct roots of other episodes. Drawn once from salt 10013."""
    commands = np.asarray(commands, np.float32)
    episode = np.asarray(episode)
    n, span, width = commands.shape
    rng = np.random.default_rng(rng_salt)
    noise_small = rng.normal(0.0, 0.1, (n, 4, span, width))
    noise_large = rng.normal(0.0, 0.3, (n, 4, span, width))
    noise_grip = rng.normal(0.0, 0.5, (n, 3, span))
    others = np.empty((n, 4), np.int64)
    every = np.arange(n)
    for i in range(n):
        pool = every[episode != episode[i]]
        if len(pool) < 4:
            raise ContractError("candidates need at least four roots of other episodes")
        others[i] = rng.choice(pool, size=4, replace=False)
    out = np.empty((n, CANDIDATES, span, width), np.float32)
    out[:, 0] = commands
    base = commands[:, None].astype(np.float64)
    out[:, 1:5] = np.clip(base + noise_small, -1.0, 1.0)
    out[:, 5:9] = np.clip(base + noise_large, -1.0, 1.0)
    grip = np.repeat(base, 3, axis=1)
    grip[..., GRIPPER_CHANNEL] = np.clip(grip[..., GRIPPER_CHANNEL] + noise_grip, -1.0, 1.0)
    out[:, 9:12] = grip
    out[:, 12:16] = commands[others]
    classes = [name for name, count in CANDIDATE_CLASSES for _ in range(count)]
    return {"sequences": out, "classes": classes, "other_roots": others}


def table_sha256(a) -> str:
    a = np.ascontiguousarray(a)
    digest = hashlib.sha256()
    digest.update(str(a.dtype).encode())
    digest.update(json.dumps(list(a.shape)).encode())
    digest.update(memoryview(a).cast("B"))
    return digest.hexdigest()


def hold_commands(commands) -> np.ndarray:
    """§6: hold = arm zero, gripper at the root's first commanded value, for every step."""
    commands = np.asarray(commands, np.float32)
    out = np.zeros_like(commands)
    out[..., GRIPPER_CHANNEL] = commands[:, :1, GRIPPER_CHANNEL]
    return out


def jaw_targets(commands, start_target, *, closed: float, opened: float, rate: float):
    """The gripper target trajectory of absolute gripper commands under the rate clip (rad).
    ``commands`` [..., T] normalised gripper channel, ``start_target`` [...] in rad."""
    c = np.asarray(commands, np.float64)
    target = np.asarray(start_target, np.float64).copy()
    out = np.empty(c.shape)
    for t in range(c.shape[-1]):
        want = closed + (c[..., t] + 1.0) / 2.0 * (opened - closed)
        target = np.clip(want, target - rate, target + rate)
        target = np.clip(target, closed, opened)
        out[..., t] = target
    return out


# ----- latent statistics (§3, §7.5) ---------------------------------------------------------------
def latent_error(predicted, target) -> np.ndarray:
    """Mean squared difference over the standardised latent values, per row (float64)."""
    d = np.asarray(predicted, np.float64) - np.asarray(target, np.float64)
    return (d * d).reshape(len(d), -1).mean(1)


def true_rank(scores) -> np.ndarray:
    """How many other candidates score at least as well as column 0 (ties count against)."""
    scores = np.asarray(scores, np.float64)
    return (scores[:, 1:] <= scores[:, :1]).sum(1)


def tie_share(sequences, h: int) -> float:
    """Share of roots where some other candidate equals the executed sequence over h steps."""
    s = np.asarray(sequences, np.float64)[:, :, :h]
    same = (np.abs(s[:, 1:] - s[:, :1]) <= TIE_TOLERANCE).all(axis=(2, 3))
    return float(same.any(1).mean())


# ----- the cluster bootstrap (§7.5) ---------------------------------------------------------------
class Bootstrap:
    """Episode draw counts ``[resamples, episodes]`` (salt 10014) and each root's episode."""

    def __init__(
        self, episode, resamples: int = BOOTSTRAP_RESAMPLES, salt: int = SALTS["bootstrap"]
    ):
        episode = np.asarray(episode)
        self.ids, self.inverse = np.unique(episode, return_inverse=True)
        rng = np.random.default_rng(salt)
        e = len(self.ids)
        self.counts = np.empty((int(resamples), e), np.float64)
        for b in range(int(resamples)):
            self.counts[b] = np.bincount(rng.integers(0, e, e), minlength=e)
        self.resamples = int(resamples)

    def episode_sums(self, values, mask=None) -> np.ndarray:
        v = np.asarray(values, np.float64)
        if mask is not None:
            v = np.where(mask, v, 0.0)
        return np.bincount(self.inverse, weights=v, minlength=len(self.ids))

    def sums(self, values, mask=None) -> np.ndarray:
        return self.counts @ self.episode_sums(values, mask)

    def medians(self, values, mask=None, chunk: int = 250) -> np.ndarray:
        """Per-resample weighted median of ``values`` over the roots (in ``mask``)."""
        v = np.asarray(values, np.float64)
        e = self.inverse
        if mask is not None:
            keep = np.asarray(mask, bool)
            v, e = v[keep], e[keep]
        if len(v) == 0:
            return np.full(self.resamples, np.nan)
        order = np.argsort(v, kind="stable")
        v, e = v[order], e[order]
        out = np.empty(self.resamples)
        for lo in range(0, self.resamples, chunk):
            w = self.counts[lo : lo + chunk][:, e]
            cum = np.cumsum(w, axis=1)
            half = cum[:, -1:] / 2.0
            first = (cum < half).sum(1)
            second = (cum <= half).sum(1)
            second = np.minimum(second, len(v) - 1)
            out[lo : lo + chunk] = np.where(cum[:, -1] > 0, (v[first] + v[second]) / 2.0, np.nan)
        return out


def interval(boot, level: float) -> list[float]:
    boot = np.asarray(boot, np.float64)
    boot = boot[np.isfinite(boot)]
    if len(boot) == 0:
        return [float("nan"), float("nan")]
    alpha = (1.0 - float(level)) / 2.0
    lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
    return [float(lo), float(hi)]


def ratio_of_sums(num, den, boot: Bootstrap, level: float) -> dict:
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = boot.sums(num) / boot.sums(den)
    return {"value": float(num.sum() / den.sum()), "ci": interval(b, level), "level": level}


def median_stat(values, boot: Bootstrap, level: float, mask=None) -> dict:
    v = np.asarray(values, np.float64)
    sel = v if mask is None else v[np.asarray(mask, bool)]
    return {
        "value": float(np.median(sel)) if len(sel) else float("nan"),
        "ci": interval(boot.medians(v, mask), level),
        "level": level,
        "roots": int(len(sel)),
    }


def ratio_of_medians(num, den, boot: Bootstrap, level: float, mask=None) -> dict:
    a, b = np.asarray(num, np.float64), np.asarray(den, np.float64)
    keep = np.ones(len(a), bool) if mask is None else np.asarray(mask, bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = boot.medians(a, keep) / boot.medians(b, keep)
        value = float(np.median(a[keep]) / np.median(b[keep])) if keep.any() else float("nan")
    return {"value": value, "ci": interval(r, level), "level": level, "roots": int(keep.sum())}


def mean_stat(values, boot: Bootstrap, level: float) -> dict:
    v = np.asarray(values, np.float64)
    ones = np.ones(len(v))
    b = boot.sums(v) / boot.sums(ones)
    return {"value": float(v.mean()), "ci": interval(b, level), "level": level}


def paired_median_difference(a, b, boot: Bootstrap, level: float) -> dict:
    """median(a) - median(b) on the same resamples (§9's L - R reading)."""
    d = boot.medians(a) - boot.medians(b)
    return {"value": float(np.median(a) - np.median(b)), "ci": interval(d, level), "level": level}


# ----- G1: collapse and rank (§6) -----------------------------------------------------------------
def effective_rank_from_energy(energy) -> float:
    energy = np.clip(np.asarray(energy, np.float64), 0.0, None)
    total = float(energy.sum())
    if total <= 0:
        return 0.0
    p = energy[energy > 0] / total
    return float(np.exp(-(p * np.log(p)).sum()))


def gram_spectrum(x) -> np.ndarray:
    """Eigenvalues of the centred roots x roots Gram matrix (TASK-077's exact form; float64)."""
    x = np.asarray(x, np.float64).reshape(len(x), -1)
    xc = x - x.mean(0)
    return np.clip(np.linalg.eigvalsh(xc @ xc.T), 0.0, None)


def collapse_statistics(predicted, encoded) -> dict:
    """G1 (i) and (ii) at full width: the collapsed fraction (prediction std < 0.01 x the encoded
    std at the same frames), the std ratio (mean stds) and the effective-rank ratio."""
    p = np.asarray(predicted, np.float64).reshape(len(predicted), -1)
    e = np.asarray(encoded, np.float64).reshape(len(encoded), -1)
    p_std, e_std = p.std(0), e.std(0)
    p_rank = effective_rank_from_energy(gram_spectrum(p))
    e_rank = effective_rank_from_energy(gram_spectrum(e))
    return {
        "collapsed_fraction": float((p_std < G["G1_collapse_std_factor"] * e_std).mean()),
        "predicted_std_mean": float(p_std.mean()),
        "encoded_std_mean": float(e_std.mean()),
        "std_ratio": float(p_std.mean() / e_std.mean()) if e_std.mean() > 0 else 0.0,
        "predicted_effective_rank": p_rank,
        "encoded_effective_rank": e_rank,
        "effective_rank_ratio": p_rank / e_rank if e_rank > 0 else 0.0,
        "samples": int(len(p)),
    }


def truncate(predicted, k: int) -> np.ndarray:
    """``predicted`` projected onto its own top ``k`` principal directions (mean kept)."""
    p = np.asarray(predicted, np.float64).reshape(len(predicted), -1)
    pc = p - p.mean(0)
    values, vectors = np.linalg.eigh(pc @ pc.T)
    u = vectors[:, np.argsort(values)[::-1][: int(k)]]
    return p.mean(0) + u @ (u.T @ pc)


def fit_basis(x_t_x, mean, n: int, k: int = G["G1_projected_directions"]) -> np.ndarray:
    """G1 (iii)'s basis: the top ``k`` principal directions of the standardised encoded latents
    of the fit sample, from their accumulated uncentred second moment ``x_t_x`` [D, D], mean and
    count. Returns ``[D, k]`` (float64), with a deterministic sign."""
    mean = np.asarray(mean, np.float64)
    cov = np.asarray(x_t_x, np.float64) / n - np.outer(mean, mean)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1][:k]
    vectors = vectors[:, order]
    signs = np.sign(vectors[np.abs(vectors).argmax(0), np.arange(vectors.shape[1])])
    return vectors * np.where(signs == 0, 1.0, signs)


def basis_sha256(basis) -> str:
    return table_sha256(np.ascontiguousarray(basis, np.float64))


class ProjectedMoments:
    """Per-episode sums and cross-products of rows projected onto G1 (iii)'s basis."""

    def __init__(self, rows, episode_index, basis, episodes: int):
        z = np.asarray(rows, np.float64).reshape(len(rows), -1) @ np.asarray(basis, np.float64)
        e = np.asarray(episode_index)
        k = z.shape[1]
        self.n = np.bincount(e, minlength=episodes).astype(np.float64)
        self.s1 = np.zeros((episodes, k))
        self.s2 = np.zeros((episodes, k, k))
        for j in range(episodes):
            zj = z[e == j]
            self.s1[j] = zj.sum(0)
            self.s2[j] = zj.T @ zj

    def ranks(self, counts) -> np.ndarray:
        """Effective rank of the covariance for every row of ``counts`` [B, episodes]."""
        counts = np.atleast_2d(np.asarray(counts, np.float64))
        n = counts @ self.n
        mean = (counts @ self.s1) / n[:, None]
        k = self.s2.shape[1]
        second = (counts @ self.s2.reshape(len(self.s2), -1)).reshape(-1, k, k) / n[:, None, None]
        cov = second - mean[:, :, None] * mean[:, None, :]
        values = np.linalg.eigvalsh(cov)
        return np.array([effective_rank_from_energy(v) for v in values])


def projected_rank_difference(x, n_x, encoded, boot: Bootstrap, level: float, chunk=200) -> dict:
    """G1 (iii): rank ratio of X minus that of N_X, both against the encoded targets, each in
    the projected basis; point value on all episodes, interval over ``boot``'s resamples."""
    ones = np.ones((1, len(boot.ids)))
    point = {k: m.ranks(ones)[0] for k, m in (("X", x), ("N", n_x), ("E", encoded))}
    diffs = []
    for lo in range(0, boot.resamples, chunk):
        c = boot.counts[lo : lo + chunk]
        rx, rn, re = x.ranks(c), n_x.ranks(c), encoded.ranks(c)
        with np.errstate(divide="ignore", invalid="ignore"):
            diffs.append(rx / re - rn / re)
    d = np.concatenate(diffs)
    return {
        "rank_x": point["X"],
        "rank_n": point["N"],
        "rank_encoded": point["E"],
        "value": point["X"] / point["E"] - point["N"] / point["E"],
        "ci": interval(d, level),
        "level": level,
        "approximation": f"effective rank of a {x.s1.shape[1]}-direction projection",
    }


def floor_01(x: float) -> float:
    return float(math.floor(100.0 * float(x) + 1e-9)) / 100.0


def ceil_above_01(x: float) -> float:
    """The smallest 0.01 step strictly above ``x``."""
    return float(math.floor(100.0 * float(x) + 1e-9) + 1.0) / 100.0


def g1_bars(rank_ratios: dict, std_ratios: dict) -> dict:
    """§10.4: B_rank = max(0.10, 1/2 x calibration L's rank ratio) and B_std = max(0.25, 1/2 x
    its std ratio), each the smaller over h in {8, 16}, rounded down to 0.01."""
    half_rank = min(0.5 * float(rank_ratios[h]) for h in GATE_HORIZONS)
    half_std = min(0.5 * float(std_ratios[h]) for h in GATE_HORIZONS)
    return {
        "B_rank": max(G["G1_rank_floor"], floor_01(half_rank)),
        "B_std": max(G["G1_std_floor"], floor_01(half_std)),
        "half_rank_ratio_min": half_rank,
        "half_std_ratio_min": half_std,
    }


def g1_passes(stats: dict, comparative_lb: float, bars: dict) -> dict:
    out = {
        "i": stats["collapsed_fraction"] <= G["G1_max_collapsed_fraction"],
        "ii_rank": stats["effective_rank_ratio"] >= bars["B_rank"],
        "ii_std": stats["std_ratio"] >= bars["B_std"],
        "iii": bool(np.isfinite(comparative_lb) and comparative_lb > 0.0),
    }
    out["all"] = all(out.values())
    return out


# ----- readouts (§7.3) ----------------------------------------------------------------------------
def fold_assignment(groups, folds: int = RIDGE_FOLDS, salt: int = SALTS["folds"]) -> np.ndarray:
    """Episode-grouped folds: the unique groups in a salt-10015 permutation, dealt round-robin."""
    ids, inverse = np.unique(np.asarray(groups), return_inverse=True)
    if len(ids) < folds:
        raise ContractError(f"{len(ids)} groups cannot make {folds} folds")
    perm = np.random.default_rng(salt).permutation(len(ids))
    fold_of = np.empty(len(ids), np.int64)
    fold_of[perm] = np.arange(len(ids)) % folds
    return fold_of[inverse]


LABEL_COLUMNS = {"cube": slice(0, 3), "jaw": slice(3, 4)}


def label_matrix(cube, jaw) -> np.ndarray:
    """``[N, 4]`` float64: cube centre (cm, 3 columns) then jaw angle (rad)."""
    cube = np.asarray(cube, np.float64).reshape(-1, 3)
    jaw = np.asarray(jaw, np.float64).reshape(-1, 1)
    return np.concatenate([cube, jaw], axis=1)


def _moments(x, y, rows, chunk: int = 4096):
    """float64 ``(X^T X, sum x, X^T Y, sum y, n)`` over ``rows`` of float32/64 ``x``."""
    d, m = x.shape[1], y.shape[1]
    xtx, sx, xty, sy = np.zeros((d, d)), np.zeros(d), np.zeros((d, m)), np.zeros(m)
    rows = np.asarray(rows)
    for lo in range(0, len(rows), chunk):
        r = rows[lo : lo + chunk]
        xb = np.asarray(x[r], np.float64)
        yb = y[r]
        xtx += xb.T @ xb
        sx += xb.sum(0)
        xty += xb.T @ yb
        sy += yb.sum(0)
    return xtx, sx, xty, sy, len(rows)


def _solve_primal(stats, penalties):
    xtx, sx, xty, sy, n = stats
    mx, my = sx / n, sy / n
    a = xtx - n * np.outer(mx, mx)
    rhs = xty - n * np.outer(mx, my)
    eye = np.eye(len(mx))
    out = {}
    for lam in penalties:
        w = np.linalg.solve(a + lam * eye, rhs)
        out[lam] = (w, my - mx @ w)
    return out


def _solve_dual(x, y, penalties):
    x = np.asarray(x, np.float64)
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    k = xc @ xc.T
    eye = np.eye(len(xc))
    out = {}
    for lam in penalties:
        w = xc.T @ np.linalg.solve(k + lam * eye, yc)
        out[lam] = (w, my - mx @ w)
    return out


def fit_readouts(x, y, groups, *, penalties=RIDGE_PENALTIES, folds: int = RIDGE_FOLDS) -> dict:
    """§7.3: one ridge regression per label (float64, CPU, with an unpenalised intercept) from the
    standardised latents ``x`` [N, D] to ``y`` = :func:`label_matrix` [N, 4], each label's
    penalty chosen from ``penalties`` by ``folds``-fold episode-grouped cross-validation (salt
    10015) on its own fitting data, by the summed squared error over the label's columns (ties:
    the smaller penalty), then refitted on all the data. Primal normal equations when N >= D,
    the dual (kernel) form otherwise; the same estimator either way."""
    y = np.asarray(y, np.float64)
    n, d = x.shape[0], int(np.prod(x.shape[1:]))
    x = x.reshape(n, d)
    if y.shape != (n, 4) or not np.isfinite(y).all():
        raise ContractError("G-finite: readout labels must be finite [N, 4]")
    fold = fold_assignment(groups, folds)
    sse = {name: {lam: 0.0 for lam in penalties} for name in LABEL_COLUMNS}
    primal = n >= d
    total = _moments(x, y, np.arange(n)) if primal else None
    if primal and not np.isfinite(total[0]).all():
        raise ContractError("G-finite: non-finite readout inputs")
    for f in range(folds):
        held = np.flatnonzero(fold == f)
        train = np.flatnonzero(fold != f)
        if primal:
            part = _moments(x, y, held)
            fits = _solve_primal(tuple(t - p for t, p in zip(total, part, strict=True)), penalties)
        else:
            fits = _solve_dual(x[train], y[train], penalties)
        xh = np.asarray(x[held], np.float64)
        for lam, (w, b) in fits.items():
            r = xh @ w + b - y[held]
            for name, cols in LABEL_COLUMNS.items():
                sse[name][lam] += float((r[:, cols] ** 2).sum())
    best = {
        name: min(penalties, key=lambda lam, s=sse[name]: (s[lam], lam)) for name in LABEL_COLUMNS
    }
    chosen = sorted(set(best.values()))
    final = _solve_primal(total, chosen) if primal else _solve_dual(x, y, chosen)
    out = {}
    for name, cols in LABEL_COLUMNS.items():
        w, b = final[best[name]]
        out[name] = {
            "weights": w[:, cols].copy(),
            "intercept": b[cols].copy(),
            "penalty": float(best[name]),
            "cv_sse": {repr(float(k)): float(v) for k, v in sse[name].items()},
            "penalty_at_grid_edge": bool(best[name] in (min(penalties), max(penalties))),
            "samples": int(n),
            "groups": int(len(np.unique(groups))),
            "form": "primal" if primal else "dual",
        }
    return out


def apply_readout(fit: dict, x) -> np.ndarray:
    x = np.asarray(x, np.float64).reshape(len(x), -1)
    return x @ fit["weights"] + fit["intercept"]


def label_errors(label: str, predicted, truth) -> np.ndarray:
    """Cube: Euclidean error in cm; jaw: absolute error in rad."""
    p = np.asarray(predicted, np.float64).reshape(len(predicted), -1)
    t = np.asarray(truth, np.float64).reshape(len(truth), -1)
    if label == "cube":
        return np.linalg.norm(p - t, axis=1)
    if label == "jaw":
        return np.abs(p[:, 0] - t[:, 0])
    raise ContractError(f"unknown label {label!r}")


def readout_record(fit: dict) -> dict:
    return {k: v for k, v in fit.items() if k not in ("weights", "intercept")} | {
        "sha256": table_sha256(np.concatenate([fit["weights"].ravel(), fit["intercept"].ravel()]))
    }


def save_readouts(path, fits: dict) -> None:
    arrays = {}
    for name, fit in fits.items():
        arrays[f"{name}_weights"] = fit["weights"]
        arrays[f"{name}_intercept"] = fit["intercept"]
    np.savez(path, **arrays)


def load_readouts(path) -> dict:
    with np.load(path) as d:
        return {n: {"weights": d[f"{n}_weights"], "intercept": d[f"{n}_intercept"]} for n in LABELS}


# ----- moving subsets (§7.4) ----------------------------------------------------------------------
def moving_mask(label: str, start_truth, end_truth, threshold: float) -> np.ndarray:
    a = np.asarray(start_truth, np.float64).reshape(len(start_truth), -1)
    b = np.asarray(end_truth, np.float64).reshape(len(end_truth), -1)
    if label == "cube":
        return np.linalg.norm(b - a, axis=1) >= threshold
    return np.abs(b[:, 0] - a[:, 0]) >= threshold


def moving_thresholds_from_d0(shares_h8: dict) -> dict:
    """§7.4: a subset's threshold is halved once if its D0-eval share x 5 550 is below 300."""
    out = {}
    for label, share in shares_h8.items():
        base = MOVING_THRESHOLDS[label]
        projected = float(share) * MOVING_PROJECTED_ROOTS
        halved = projected < MOVING_MIN_PROJECTED
        out[label] = {
            "threshold": base / 2.0 if halved else base,
            "declared": base,
            "halved": bool(halved),
            "d0_eval_share_h8": float(share),
            "projected_test_roots_h8": projected,
        }
    return out


# ----- §10.3 and §10.5 ----------------------------------------------------------------------------
def grid_rule(g4_cube_median_cm: float, g8_cube_median_cm: float) -> dict:
    """g8 iff its D0-eval median cube error is at most 0.85 x g4's and g4's exceeds 0.5 cm."""
    choose_g8 = (
        g8_cube_median_cm <= GRID_RULE_RATIO * g4_cube_median_cm
        and g4_cube_median_cm > GRID_RULE_MIN_G4_CM
    )
    return {
        "grid": "g8" if choose_g8 else "g4",
        "g4_cube_median_cm": float(g4_cube_median_cm),
        "g8_cube_median_cm": float(g8_cube_median_cm),
        "ratio_g8_over_g4": float(g8_cube_median_cm / g4_cube_median_cm),
    }


def g5_bars(renc_medians: dict) -> dict:
    """B_l = 1.5 x the median error of R-enc on D0-eval's encoded frames."""
    return {f"B_{k}": G["G5_bar_factor"] * float(v) for k, v in renc_medians.items()}


# ----- G6 (§6) ------------------------------------------------------------------------------------
def saturation(curve) -> dict:
    """G6 for one model: the best val criterion over the last quarter of training (updates above
    0.75 x the final update) is not more than 2 % below the best up to 0.75 x the final update."""
    points = [(int(u), float(v)) for u, v in curve]
    if not points:
        raise ContractError("empty curve")
    final = points[-1][0]
    early = [v for u, v in points if u <= 0.75 * final]
    late = [v for u, v in points if u > 0.75 * final]
    if not early or not late:
        raise ContractError("the curve needs points on both sides of 0.75 x its final update")
    best_early, best_late = min(early), min(late)
    drop = (best_early - best_late) / abs(best_early) if best_early else 0.0
    return {
        "best_early": best_early,
        "best_late": best_late,
        "relative_drop": float(drop),
        "passes": bool(drop <= G["G6_max_drop"]),
        "final_update": final,
    }


def extension_needed(curves: dict) -> bool:
    """§6: for an eligible arm, all six models X-m and N_X-m extend if any fails G6."""
    return any(not saturation(c)["passes"] for c in curves.values())


# ----- budget (§8.1, §8.2) ------------------------------------------------------------------------
def budget_from_probe(curve, seconds_per_update: float, *, used_gpu_h: float) -> dict:
    """U = max(20 000, min(60 000, ceil(1.5 u* / 5 000) x 5 000)); then lowered in steps of
    5 000 (not below 20 000) until T + a full T' + V-E fit the cap's remainder; escalate if not."""
    points = [(int(u), float(v)) for u, v in curve]
    floor = min(v for _, v in points)
    u_star = next(u for u, v in points if v <= floor + abs(floor) * SELECTION_TOLERANCE)
    raw = int(math.ceil(1.5 * u_star / U_STEP)) * U_STEP
    updates = max(U_MIN, min(U_MAX, raw))
    spu = float(seconds_per_update)
    remainder = HARD_CAP_GPU_H - float(used_gpu_h)

    def projection(u):
        t = len(ARMS) * len(MODEL_SEEDS) * u * spu / 3600.0
        t_ext = 2 * 2 * len(MODEL_SEEDS) * (EXTENSION_FACTOR - 1.0) * u * spu / 3600.0
        return {
            "T": t,
            "T_prime_full": t_ext,
            "V_E": VE_ESTIMATE_GPU_H,
            "sum": t + t_ext + VE_ESTIMATE_GPU_H,
        }

    chosen = updates
    while projection(chosen)["sum"] > remainder and chosen > U_MIN:
        chosen -= U_STEP
    fits = projection(chosen)["sum"] <= remainder
    return {
        "u_star": int(u_star),
        "updates_by_rule": int(updates),
        "updates": int(chosen),
        "lowered": bool(chosen != updates),
        "select_every": int(chosen // SELECTIONS),
        "cap_seconds": float(CAP_FACTOR * spu * chosen),
        "seconds_per_update": spu,
        "projection_gpu_h": projection(chosen),
        "remainder_gpu_h": remainder,
        "escalate": bool(not fits),
    }


# ----- gates and the row (§6, §11) ----------------------------------------------------------------
def gates_for(entry: dict, bars: dict) -> dict:
    """G1-G5 pass flags for one eligible arm, seed and h from its evaluated quantities:
    ``entry`` holds ``g1_stats``, ``g1_iii``, ``copy_ratio``, ``noaction_ratio``, ``sens`` and
    ``g5`` ({label: {"median", "vs_copy", "vs_noaction"}}), each with gate-level intervals."""
    g1 = g1_passes(entry["g1_stats"], entry["g1_iii"]["ci"][0], bars)
    g2 = entry["copy_ratio"]["ci"][1] < G["G2_max_ratio"]
    g3 = entry["noaction_ratio"]["ci"][1] < G["G3_max_ratio"]
    g4 = entry["sens"]["ci"][0] >= G["G4_min_sens"]
    g5 = {}
    for label in LABELS:
        e = entry["g5"][label]
        g5[label] = {
            "a": bool(e["median"]["ci"][1] <= bars[f"B_{label}"]),
            "b": bool(e["vs_copy"]["ci"][1] < G["G5_max_ratio"]),
            "c": bool(e["vs_noaction"]["ci"][1] < G["G5_max_ratio"]),
        }
    g5_all = all(all(v.values()) for v in g5.values())
    return {
        "G1": g1,
        "G2": bool(g2),
        "G3": bool(g3),
        "G4": bool(g4),
        "G5": g5,
        "G5_all": bool(g5_all),
        "G1_G4": bool(g1["all"] and g2 and g3 and g4),
        "G1_G5": bool(g1["all"] and g2 and g3 and g4 and g5_all),
    }


def decide(*, void_reasons, small_subsets, gates: dict, g6: dict) -> dict:
    """§11, first match. ``gates[arm][seed][h]`` from :func:`gates_for`; ``g6[arm][seed]`` is the
    seed's G6 (X and N_X both saturated); ``small_subsets``: G5 subsets under 100 test roots."""
    if void_reasons:
        return {"row": "VOID", "reasons": list(void_reasons)}
    if small_subsets:
        return {"row": "INCONCLUSIVE", "reasons": list(small_subsets)}
    seeds = [str(s) for s in MODEL_SEEDS]
    passing = [
        arm
        for arm in ELIGIBLE
        if all(gates[arm][s][str(h)]["G1_G5"] and g6[arm][s] for s in seeds for h in GATE_HORIZONS)
    ]
    if passing:
        return {
            "row": "WM-PASS",
            "arms": passing,
            "s5_model": "L" if "L" in passing else passing[0],
        }
    partial = [arm for arm in ELIGIBLE if all(gates[arm][s]["8"]["G1_G4"] for s in seeds)]
    failures = {}
    for arm in ELIGIBLE:
        failures[arm] = {
            s: {
                "G6": not g6[arm][s],
                **{
                    f"h{h}": [k for k in ("G1_G4", "G5_all") if not gates[arm][s][str(h)][k]]
                    for h in GATE_HORIZONS
                },
            }
            for s in seeds
        }
    if partial:
        return {"row": "WM-PARTIAL", "arms": partial, "failures": failures}
    return {"row": "WM-FAIL", "arms": [], "failures": failures}


# ----- set by Stage 0 before the freeze (§10.7) ---------------------------------------------------
CHOSEN_GRID: str | None = None
BARS: dict | None = None  # B_rank, B_std, B_cube, B_jaw
MOVING: dict | None = None  # {"cube": cm, "jaw": rad} after §7.4's halving rule


def frozen_choices() -> dict:
    if CHOSEN_GRID is None or BARS is None or MOVING is None:
        raise ContractError("Stage 0 has not set the grid, the bars and the subset thresholds")
    return {"grid": CHOSEN_GRID, "bars": dict(BARS), "moving": dict(MOVING)}


def frozen_encoder_name(encoder: str) -> str:
    return {"pretrained": "dinov2_small_224", "random_init": "dinov2_small_224_random_init"}[
        encoder
    ]


def model_config(arm: str, grid: str) -> dict:
    """The run's model configuration (a module constant, TASK-066's ``MODEL_CONFIG`` precedent):
    the adapters' defaults otherwise, ``max_horizon`` 32, SIGReg off for LeWM (§4)."""
    spec, g = check_arm(arm), GRIDS[grid]
    config = {
        "frozen_encoder": frozen_encoder_name(spec["encoder"]),
        "token_count": g["tokens"],
        "token_width": g["width"],
        "latent_dim": LATENT_DIM,
        "camera": "cached_features",
        "max_horizon": MAX_HORIZON,
    }
    if spec["backend"] == "leworldmodel":
        config["sigreg_weight"] = 0.0
    return config


# ----- the existing files this study composes (byte-identical to origin/main at a277816) ----------
COMPOSED_FILES_SHA256 = {
    "src/embodied_jepa/models/frozen_tokens.py": (
        "33e8e819ad557d4fec6167482f91b5aa0c5da5a1169e7f8e6e8f050390c9a3a2"
    ),
    "src/embodied_jepa/models/frozen_encoder.py": (
        "c1a4622a0765e44fc060c5313a85e4f727f51dc361acd5de50cb6990845e6011"
    ),
    "src/embodied_jepa/models/base.py": (
        "310a509d5f8043ad3cbdf85a0568d15bb4ed2e4faeddf6e10bed63abf8ee0549"
    ),
    "src/embodied_jepa/models/lewm.py": (
        "a99e000436da627916dcc8449b43f7ea77518cc99ef82824844d82ac23b200f1"
    ),
    "src/embodied_jepa/models/native.py": (
        "95e112e13b0bf4d587fd5c35008e99b1e0a3759fb4f458c24bdd20304d0fadf3"
    ),
    "src/embodied_jepa/models/schema_backends.py": (
        "85a6e4d2158b10b7162cba5d6bd1fb598f7414875b5c648ba42ea6fa4fad7f8e"
    ),
    "src/embodied_jepa/models/readout.py": (
        "5eb6ac4734ed08f9c2531bd1f37154102bf506c487e34019fb090b51035b8bc2"
    ),
    "src/embodied_jepa/readout_labels.py": (
        "be8fb2acd26b4f5f921ba7883be2145ea189bc24c64e6bcc7931d48d9f563f1f"
    ),
    "src/embodied_jepa/pretrained_encoder.py": (
        "8b2b238ba8ead2e2e64d79f081014292d03967ebab0d88fd5bf10ffec437f27b"
    ),
    "src/embodied_jepa/contracts.py": (
        "5f3afd25d608c3484d24e6aee1f77bdb1c4b487dfe4261d166048df22bd119e9"
    ),
    "src/embodied_jepa/action_schemas.py": (
        "e83a3804baaeb4b57aecd8074a85e71cd3c70be8d2da8240e9e7df287ba74357"
    ),
    "src/embodied_jepa/schema_data.py": (
        "59b6f425570bcd614c5cb74d120ce6007dc89af28367960a7209496e6ded6708"
    ),
    "src/embodied_jepa/run_tools.py": (
        "1b3b61f89ca50e5d12650312c374bdcd07f1eb394ffb413c19eb844eb5ff5394"
    ),
    "src/embodied_jepa/devices.py": (
        "0976ccc0fb6d3c64d8128dbc152dcb2ebd3aee2211bc3ab5e25a772518e2fb72"
    ),
    "src/embodied_jepa/so101_play_corpus.py": (
        "cb8a6d0f72216f06c3d2a82cf2101a5c0ef7a1f0f732ef40e75cdcc3472463c3"
    ),
    "configs/so101_sim_action.json": (
        "c573b64ae4ae3860d3465a4f5cc334c746fece36c3dc2fcc670223f71e40016e"
    ),
}


def composed_files_check(root) -> dict:
    """§12: every composed file is byte-identical to its pin (``{path: bool}``)."""
    from pathlib import Path

    out = {}
    for rel, pin in COMPOSED_FILES_SHA256.items():
        path = Path(root) / rel
        out[rel] = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == pin
    return out
