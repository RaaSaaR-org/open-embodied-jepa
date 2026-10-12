"""TASK-090: the grasp-enriched play corpus against an equal-size control -- cohorts, Stage C's
labels and bars, the combined feature stores, Stage E's and A′'s rules, Part B's statistics, bar
and rows, and the power rule (protocol ``docs/experiments/grasp_play_corpus_v1.md``).

NumPy only at import. TASK-085's, TASK-087's, TASK-088's and TASK-089's modules are imported and
never changed. Simulator truth (the sidecars' object positions and contacts) is read only by the
Stage C labels, the reported window classes and the scorer; never by a model input or a planning
cost. Nothing here is registered or imported by ``embodied_jepa``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from embodied_jepa import grounded_wm as gw
from embodied_jepa import grounded_wm_data as gd
from embodied_jepa import play_corpus as pc
from embodied_jepa import zero_shot as zs
from embodied_jepa import zero_shot_grasp as zg
from embodied_jepa.contracts import ContractError

TASK = "TASK-090"
PROTOCOL = "docs/experiments/grasp_play_corpus_v1.md"

# ----- seeds and salts (§10) ---------------------------------------------------------------------
SALTS = {"shuffle": 9002, "policy": 9003, "probe": 9004, "split": 9005, "bootstrap": 9006}
MODEL_SEEDS = (90100, 90101, 90102)
N_SEED = 90100
DEBUG_MODEL_SEEDS = range(90900, 90910)  # plumbing and smoke only
BS_N = 128  # Stage 0's power rule (§7.2): power 0.727 < 0.80 at n = 64, SD 1 (R27.22)
COHORTS = {
    "debug-roots": range(90910, 90950),
    "debug-grasp": range(90950, 91000),
    "k0-roots": range(90000, 90032),
    "a-roots": range(903000, 903064),
    "bd-grasp": range(90200, 90232),
    "bs-grasp": range(904000, 904000 + BS_N),
}
ROOT_COHORTS = ("debug-roots", "k0-roots", "a-roots")
GRASP_COHORTS = ("debug-grasp", "bd-grasp", "bs-grasp")
CORPORA = ("play-v2g", "play-v1p")
# §7.1-§7.2: the closed-loop model arms and the seeds each cohort admits. M-v1's 87101 and 87102
# runs are dropped from BS by Stage 0 if its smoke projects BS above 14 GPU hours (§14). They
# are (R27.25): at n = 128 nine world-model runs project to about 20 GPU hours.
CLOSED_LOOP_ARMS = {"M-v2g": "play-v2g", "M-v1p": "play-v1p", "M-v1": "play-v1"}
BD_SEEDS = {"M-v2g": (90100,), "M-v1p": (90100,), "M-v1": (87100,)}
BS_M_V1_SEEDS = (87100,)
BS_SEEDS = {"M-v2g": MODEL_SEEDS, "M-v1p": MODEL_SEEDS, "M-v1": BS_M_V1_SEEDS}
# Stage T's store mode and per-job cap (§12, §14), set by Stage 0's timed smoke (R27.24): on the
# real store path, read into memory 37.0 ms per update (median), memory-mapped from a cold page
# cache 40.7 ms (mean 43.2, p90 47.3) with only a 0.6 GB part on the HDD; the cap is TASK-087's
# rule, 3 x seconds per update x U, with the smoke's median rounded up to 40 ms
TRAIN_MMAP = False
TRAIN_CAP_SECONDS = 3 * 0.040 * 60_000
CORPUS_TAG = {"play-v2g": "v2g", "play-v1p": "v1p"}


def check_seed(cohort: str, seed: int) -> None:
    if cohort not in COHORTS:
        raise ContractError(f"unknown cohort {cohort}")
    if int(seed) not in COHORTS[cohort]:
        raise ContractError(f"seed {seed} is not in cohort {cohort}")


def check_model_seed(arm: str, seed: int, *, debug: bool = False) -> None:
    """§3: G on 90100-90102, N on 90100; debug models on 90900-90909."""
    seed = int(seed)
    if arm not in ("G", "N"):
        raise ContractError(f"TASK-090 trains G and N only, not {arm}")
    if debug:
        if seed not in DEBUG_MODEL_SEEDS:
            raise ContractError(f"{seed} is not a debug model seed")
        return
    allowed = MODEL_SEEDS if arm == "G" else (N_SEED,)
    if seed not in allowed:
        raise ContractError(f"{arm}-{seed} is not a model of TASK-090")


# ----- Stage C labels (§4; the frozen label script) ----------------------------------------------
LIFT_RISE = 0.03  # m above the object's start height, with grasp contact (TASK-085)
REPORT_RISE = 0.05
NEAR = 0.08  # m, the right palm within this of an object centre ("near", §13)
WINDOW_COMMANDS = gw.TRAIN_HORIZON  # 8 commands, 9 frames (TASK-087)


def frame_labels(side: dict, on_table) -> dict:
    """Per-frame labels from one episode's sidecar arrays (``object_position`` [F, 4, 3],
    ``grasp_contact`` [F, 4]) and its on-table mask [4]: ``lift`` [F, 4] (rise from the episode's
    first frame >= 3 cm with grasp contact, on-table objects only), ``contact`` [F, 4] and
    ``rise`` [F, 4]."""
    pos = np.asarray(side["object_position"], np.float64)
    on = np.asarray(on_table, bool)
    rise = pos[:, :, 2] - pos[:1, :, 2]
    contact = np.asarray(side["grasp_contact"], bool) & on[None]
    lift = (rise >= LIFT_RISE) & contact
    return {"lift": lift, "contact": contact, "rise": rise}


def window_any(flags, commands: int = WINDOW_COMMANDS, *, last_frame: bool = True) -> np.ndarray:
    """For every window start t = 0 .. F - commands - 1 (TASK-087's ``window_starts``): whether
    any frame t .. t + commands is flagged. ``last_frame=False`` is the disclosed development
    script's convention (§13): the episode's final frame is not looked at."""
    flags = np.asarray(flags, bool)
    f = len(flags)
    n = f - commands
    if n <= 0:
        return np.zeros(0, bool)
    usable = flags if last_frame else np.r_[flags[:-1], False]
    c = np.r_[0, np.cumsum(usable.astype(np.int64))]
    return (c[np.arange(n) + commands + 1] - c[np.arange(n)]) > 0


def closed_near(side: dict, on_table) -> np.ndarray:
    """Per command: requested right grasp > 0.5 and the right palm within 8 cm of an on-table
    object's centre (reported)."""
    req = np.asarray(side["requested"], np.float64)
    t = len(req)
    palm = np.asarray(side["palms"], np.float64)[:t, :3]
    pos = np.asarray(side["object_position"], np.float64)[:t].copy()
    pos[..., 2] -= pc.PELVIS_Z
    d = np.linalg.norm(palm[:, None, :] - pos, axis=-1)
    near = (d < NEAR) & np.asarray(on_table, bool)[None]
    return (req[:, 13] > 0.5) & near.any(1)


def episode_labels(side: dict, layout, metadata: dict | None = None) -> dict:
    """One episode's Stage C counts (everything per object where §4 reports it)."""
    on = pc.on_table_mask(layout)
    lab = frame_labels(side, on)
    lift_any = lab["lift"].any(1)
    windows = window_any(lift_any)
    contact_windows = window_any(lab["contact"].any(1))
    cn = closed_near(side, on)
    # closed-near windows: any of the window's 8 commands t .. t + 7
    cn_windows = window_any(cn, WINDOW_COMMANDS - 1)
    per_object_windows = [int(window_any(lab["lift"][:, k]).sum()) for k in range(len(on))]
    rise = lab["rise"]
    max_rise = np.where(on, rise.max(0), np.nan)
    lift_states = lab["lift"]
    out = {
        "frames": int(len(lift_any)),
        "windows": int(len(windows)),
        "lift_windows": int(windows.sum()),
        "contact_windows": int(contact_windows.sum()),
        "closed_near_windows": int(cn_windows.sum()),
        "lift_frames": int(lift_any.sum()),
        "lift_states": int(lift_states.sum()),
        "lift_states_ge5": int((lift_states & (rise >= REPORT_RISE)).sum()),
        "per_object_lift_windows": per_object_windows,
        "per_object_lifted": [bool(x) for x in lift_states.any(0)],
        "max_rise": [None if np.isnan(x) else float(x) for x in max_rise],
        "on_table": [bool(x) for x in on],
    }
    if metadata is not None:
        out["mode"] = metadata.get("mode")
        out["skills"] = [s for s, _ in metadata.get("skill_log", [])]
        out["stop_reason"] = metadata.get("stop_reason", "")
        out["pick_checks"] = metadata.get("pick_checks", [])
    return out


def dev_convention_counts(side: dict) -> dict:
    """The disclosed development script's window and step counts (§13), reproduced exactly: the
    on-table mask from the first frame's positions, windows that do not look at the final frame.
    Used only to check the frozen label script against the re-run development script."""
    t = len(side["requested"])
    pos = np.asarray(side["object_position"], np.float64)
    on = (pos[0, :, 2] > pc.TABLE_TOP_Z - 0.01) & (pos[0, :, 0] > 0.1)
    lab = frame_labels(side, on)
    lift = lab["lift"][:t]
    w = window_any(np.r_[lift.any(1), False], last_frame=False)
    return {
        "steps": int(t),
        "lift_steps": int(lift.any(1).sum()),
        "windows": int(len(w)),
        "lift_windows": int(w.sum()),
    }


def share(rows) -> float:
    w = sum(r["windows"] for r in rows)
    return float(sum(r["lift_windows"] for r in rows) / w) if w else float("nan")


# Stage C's bars (§4). Declared 2.5 and 1.75; revised at the freeze by the Stage 0 record
# (R27.21) because the debug collection's point ratios fell below them (§4's revision rule):
# C1 >= 2.0 and C2 >= 1.5 = (1 + C1) / 2, the DRAFT's derivation. C3 is unchanged.
DECLARED_BARS = {"C1": 2.5, "C2": 1.75}
C1_BAR = 2.0
C2_BAR = 1.5
C3_BAND = 0.25


def stage_c(v1_train, v2g_added, v1p_added) -> dict:
    """§4 on the train episodes' label rows: C1, C2, C3 and the row (C-PASS or C-FAIL)."""
    s1, sg, sp = share(v1_train), share(v2g_added), share(v1p_added)
    c1 = sg / sp
    c2 = share(list(v1_train) + list(v2g_added)) / share(list(v1_train) + list(v1p_added))
    c3 = sp / s1
    checks = {
        "C1": {"ratio": c1, "bar": C1_BAR, "pass": bool(c1 >= C1_BAR)},
        "C2": {"ratio": c2, "bar": C2_BAR, "pass": bool(c2 >= C2_BAR)},
        "C3": {
            "ratio": c3,
            "band": [1 - C3_BAND, 1 + C3_BAND],
            "pass": bool(1 - C3_BAND <= c3 <= 1 + C3_BAND),
        },
    }
    return {
        "shares": {"play-v1": s1, "play-v2g-added": sg, "play-v1p-added": sp},
        "checks": checks,
        "row": "C-PASS" if all(c["pass"] for c in checks.values()) else "C-FAIL",
    }


def debug_ratios(v1_train, debug_grasp, debug_v0, *, resamples=10_000, seed=930000) -> dict:
    """Stage 0's debug readings (§4): C1 with play-v1 train's share as the denominator, C2
    projected to equal-size additions (each added set as many episodes as play-v1 train, at the
    debug episodes' per-episode means), and C3's direction from the v0 debug episodes; each with
    an episode-bootstrap 95 % percentile interval (both play-v1 train and the debug episodes
    resampled; generator seeded with the first debug collection seed)."""
    rng = np.random.default_rng(seed)

    def arr(rows):
        return (
            np.array([r["lift_windows"] for r in rows], np.float64),
            np.array([r["windows"] for r in rows], np.float64),
        )

    l1, w1 = arr(v1_train)
    lg, wg = arr(debug_grasp)
    lp, wp = arr(debug_v0)
    n1 = len(l1)

    def readings(a1, b1, ag, bg, ap, bp):
        s1 = a1.sum() / b1.sum()
        sg = ag.sum() / bg.sum()
        sp = ap.sum() / bp.sum()
        # projected combined sets: play-v1 train plus n1 added episodes at the debug means
        c2 = ((a1.sum() + n1 * ag.mean()) / (b1.sum() + n1 * bg.mean())) / (
            (a1.sum() + n1 * a1.mean()) / (b1.sum() + n1 * b1.mean())
        )
        return {"C1": sg / s1, "C2": c2, "C3": sp / s1}

    point = readings(l1, w1, lg, wg, lp, wp)
    boots = {k: [] for k in point}
    for _ in range(resamples):
        i1 = rng.integers(0, len(l1), len(l1))
        ig = rng.integers(0, len(lg), len(lg))
        ip = rng.integers(0, len(lp), len(lp))
        r = readings(l1[i1], w1[i1], lg[ig], wg[ig], lp[ip], wp[ip])
        for k, v in r.items():
            boots[k].append(v)
    out = {}
    for k, v in point.items():
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        out[k] = {"point": float(v), "ci95": [float(lo), float(hi)]}
    out["C1"]["bar"], out["C2"]["bar"] = C1_BAR, C2_BAR
    out["C3"]["band"] = [1 - C3_BAND, 1 + C3_BAND]
    out["revise"] = bool(point["C1"] < C1_BAR or point["C2"] < C2_BAR)
    out["shares"] = {
        "play-v1-train": float(l1.sum() / w1.sum()),
        "debug-grasp": float(lg.sum() / wg.sum()),
        "debug-v0": float(lp.sum() / wp.sum()),
    }
    out["episodes"] = {"play-v1-train": n1, "debug-grasp": len(lg), "debug-v0": len(lp)}
    out["resamples"] = resamples
    return out


# ----- combined feature stores (§2.2) ------------------------------------------------------------
class CombinedStore:
    """Several ``FeatureStore`` splits as one: latents, state, actions, palm and the episode table
    concatenated in the given order (each part's offsets shifted). With ``mmap=False`` the
    latents are read into one preallocated array part by part (no second full copy); with
    ``mmap=True`` each part stays memory-mapped and windows are gathered per part.

    The interface is ``FeatureStore``'s (``gather``, ``roots``, ``root_arrays``, ``moments``,
    ``frames``, ``offsets``, ``table``), so TASK-087's sampler and loss run on it unchanged."""

    def __init__(self, parts, *, mmap: bool = False, chunk_frames: int = 1 << 16):
        if not parts:
            raise ContractError("no parts")
        self.parts = [(str(Path(f)), s) for f, s in parts]
        self.split = "+".join(s for _, s in self.parts)
        stores = [gd.FeatureStore(f, s, mmap=True) for f, s in self.parts]
        sizes = [len(st.state) for st in stores]
        self.bounds = np.r_[0, np.cumsum(sizes)].astype(np.int64)
        self.state = np.concatenate([st.state for st in stores])
        self.actions = np.concatenate([st.actions for st in stores])
        self.palm = np.concatenate([st.palm for st in stores])
        self.table = []
        for (folder, split), st, lo in zip(self.parts, stores, self.bounds[:-1], strict=True):
            for row in st.table:
                self.table.append(
                    row | {"offset": int(row["offset"]) + int(lo), "store": folder, "split": split}
                )
        self.frames = np.array([r["frames"] for r in self.table], np.int64)
        self.offsets = np.array([r["offset"] for r in self.table], np.int64)
        self.mmap = bool(mmap)
        if self.mmap:
            self._latent_parts = [st.latents for st in stores]
            self.latents = _PartedLatents(self._latent_parts, self.bounds)
        else:
            total = int(self.bounds[-1])
            self.latents = np.empty((total, gw.TOKENS, gw.K), np.float16)
            for st, lo in zip(stores, self.bounds[:-1], strict=True):
                n = len(st.latents)
                for a in range(0, n, chunk_frames):
                    b = min(n, a + chunk_frames)
                    self.latents[lo + a : lo + b] = st.latents[a:b]
        f = int(self.bounds[-1])
        if len(self.state) != f or int(self.frames.sum()) != f:
            raise ContractError("inconsistent combined store")

    def moments(self):
        return (self.state.mean(0), self.state.std(0), self.palm.mean(0), self.palm.std(0))

    def gather(self, starts, frames: int):
        idx = np.asarray(starts, np.int64)[:, None] + np.arange(frames)
        return {
            "latents": np.asarray(self.latents[idx.ravel()], np.float32).reshape(
                len(idx), frames, gw.TOKENS, gw.K
            ),
            "state": self.state[idx],
            "actions": self.actions[idx[:, :-1]],
            "palm": self.palm[idx],
        }

    def roots(self):
        return gw.roots(self.frames, self.offsets)

    def root_arrays(self, starts, horizons=gw.HORIZONS):
        starts = np.asarray(starts, np.int64)
        start = np.asarray(self.latents[starts], np.float32)
        cmd_idx = starts[:, None] + np.arange(gw.EVAL_HORIZON)
        commands = self.actions[cmd_idx]
        targets = {h: np.asarray(self.latents[starts + h], np.float32) for h in horizons}
        return start, commands, self.state[starts], targets


class _PartedLatents:
    """Index global frames into memory-mapped parts (``mmap=True``)."""

    def __init__(self, parts, bounds):
        self.parts, self.bounds = parts, bounds
        self.shape = (int(bounds[-1]), gw.TOKENS, gw.K)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        index = np.asarray(index, np.int64)
        flat = index.ravel()
        out = np.empty((len(flat), gw.TOKENS, gw.K), np.float16)
        which = np.searchsorted(self.bounds, flat, side="right") - 1
        for p in np.unique(which):
            m = which == p
            local = flat[m] - self.bounds[p]
            order = np.argsort(local)
            got = np.asarray(self.parts[p][local[order]])
            tmp = np.empty_like(got)
            tmp[order] = got
            out[m] = tmp
        return out.reshape(*index.shape, gw.TOKENS, gw.K)


def pinned_moments(play_v1_features) -> tuple:
    """``play-v1`` train's state and palm moments (TASK-087's; §2.2), from TASK-087's train
    store, exactly as ``FeatureStore.moments`` computes them."""
    folder = Path(play_v1_features)
    state = np.load(folder / "state_train.npy")
    palm = np.load(folder / "palm_train.npy")
    return (state.mean(0), state.std(0), palm.mean(0), palm.std(0))


MOMENT_BUFFERS = ("state_mean", "state_scale", "palm_mean", "palm_scale")


def buffers_sha256(buffers: dict) -> str:
    """sha256 over the four moment buffers (float32 bytes, in a fixed order)."""
    digest = hashlib.sha256()
    for key in MOMENT_BUFFERS:
        a = np.ascontiguousarray(np.asarray(buffers[key], np.float32))
        digest.update(key.encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def same_buffers(a: dict, b: dict) -> bool:
    """Bitwise equality of the four moment buffers."""
    return all(
        np.asarray(a[k], np.float32).tobytes() == np.asarray(b[k], np.float32).tobytes()
        for k in MOMENT_BUFFERS
    )


# ----- Stage E (§6.1) ----------------------------------------------------------------------------
E_HORIZONS = (4, 8)
E_LEVEL = gw.GATE_LEVEL  # TASK-087's gate intervals (cluster bootstrap by episode)


def e_checks(stats: dict, vs_n: dict | None) -> dict:
    """§6.1 for one G model: ``stats`` is TASK-087's ``summarise`` output at E_LEVEL, ``vs_n`` its
    ``compare`` against the corpus's N twin (seed 90100 only, else None)."""
    out, ok = {}, True
    for h in E_HORIZONS:
        s = stats[str(h)]
        e = {
            "i_copy_ratio_upper": s["copy_ratio"]["ci"][1],
            "i": bool(s["copy_ratio"]["ci"][1] < 1.0),
            "ii_sens_lower": s["sens"]["ci"][0],
            "ii": bool(s["sens"]["ci"][0] > 1.0),
        }
        if vs_n is not None:
            e["iii_acc_ratio_upper"] = vs_n[str(h)]["acc_ratio"]["ci"][1]
            e["iii"] = bool(vs_n[str(h)]["acc_ratio"]["ci"][1] < 1.0)
        e["pass"] = all(v for k, v in e.items() if k in ("i", "ii", "iii"))
        ok = ok and e["pass"]
        out[str(h)] = e
    out["pass"] = ok
    return out


# ----- A-K0 and Stage A′ (§6.2) ------------------------------------------------------------------
A2_BAR = zg.A2_BAR  # 48 of 64, declared; TASK-089's comparison against the base setting dropped
A_PRIME_SETTING = "step-mean"
A_PRIME_H = 8


def a2_prime(measures: dict) -> dict:
    """§6.2: close@R1 and lift@R2 each >= 48 of 64 (no McNemar against TASK-088's setting)."""
    out = {}
    for key in ("close_R1", "lift_R2"):
        k = int(np.sum(measures[key]))
        out[key] = {"count": k, "meets": bool(k >= A2_BAR)}
    out["meets"] = all(v["meets"] for v in out.values())
    return out


def a_prime_pass(a1_h8: dict, a2: dict) -> bool:
    """A′ for one model: TASK-089 §5.5's pass rule at h = 8 and A2's two bars."""
    return bool(a1_h8["pass"] and a2["meets"])


def a_prime_row(passes: dict) -> dict:
    """``passes[arm][seed]`` for arm in v2g, v1p: A′-FAIL when either arm fails on two or more of
    its three seeds."""
    fails = {arm: sum(not v for v in by.values()) for arm, by in passes.items()}
    bad = [arm for arm, k in fails.items() if k >= 2]
    return {"row": "A′-FAIL" if bad else "A′-PASS", "fails": fails, "failing_arms": bad}


# A-K0 (Stage 0 record, R27.23): the simulator ceiling's win rates on the 32 calibration roots
# 90000-90031 (ceiling-k0-roots.json at 23d01ce, clean tree); the bars follow by TASK-089 §5.4.
K0_CEILING = {
    "latent": {
        "8": {
            "R1:close-vs-open": 0.90625,
            "R2:lift-vs-release": 0.5625,
            "R2:lift-vs-stay": 0.96875,
        },
        "16": {"R1:close-vs-open": 0.96875, "R2:lift-vs-release": 0.84375, "R2:lift-vs-stay": 1.0},
    },
    "latent+pose": {
        "8": {"R1:close-vs-open": 1.0, "R2:lift-vs-release": 1.0, "R2:lift-vs-stay": 1.0},
        "16": {"R1:close-vs-open": 1.0, "R2:lift-vs-release": 1.0, "R2:lift-vs-stay": 1.0},
    },
}
A1_BARS: dict | None = zg.k0_bars(K0_CEILING)


# ----- readability probe (§6.2, reported) --------------------------------------------------------
PROBE_FOLDS = 5
PROBE_COMPONENTS = 32
PROBE_L2 = 1.0


def probe_folds(root_seeds) -> np.ndarray:
    """Fold of each root (5 folds by root; generator keyed by salt 9004)."""
    roots = np.asarray(root_seeds)
    unique = np.unique(roots)
    rng = np.random.default_rng([SALTS["probe"], 0])
    order = rng.permutation(len(unique))
    fold_of = {int(unique[i]): k % PROBE_FOLDS for k, i in enumerate(order)}
    return np.array([fold_of[int(r)] for r in roots])


def _logistic_fit(x, y, l2=PROBE_L2, iters=50):
    n, p = x.shape
    w = np.zeros(p + 1)
    xa = np.c_[np.ones(n), x]
    reg = np.full(p + 1, l2)
    reg[0] = 0.0
    for _ in range(iters):
        z = np.clip(xa @ w, -30, 30)
        mu = 1 / (1 + np.exp(-z))
        g = xa.T @ (mu - y) + reg * w
        h = (xa * (mu * (1 - mu))[:, None]).T @ xa + np.diag(reg) + 1e-9 * np.eye(p + 1)
        step = np.linalg.solve(h, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def _features_fit(x_train):
    mu = x_train.mean(0)
    sd = x_train.std(0) + 1e-6
    z = (x_train - mu) / sd
    k = min(PROBE_COMPONENTS, z.shape[0] - 1, z.shape[1])
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    return mu, sd, vt[:k].T


def probe_scores(x, y, folds, *, fit_x=None) -> np.ndarray:
    """Cross-fitted scores: for each fold, standardise and project to at most 32 principal
    components on the other folds, fit an L2 logistic regression (penalty 1.0) there, score the
    held-out fold. ``fit_x`` (default ``x``) are the features the probe is fitted on; ``x`` the
    ones it is applied to (a model's predictions, fitted on that model's predictions)."""
    x = np.asarray(x, np.float64)
    fx = x if fit_x is None else np.asarray(fit_x, np.float64)
    y = np.asarray(y, np.float64)
    out = np.full(len(y), np.nan)
    for k in range(PROBE_FOLDS):
        test = folds == k
        train = ~test
        if test.sum() == 0:
            continue
        if len(np.unique(y[train])) < 2:
            out[test] = y[train].mean() if train.any() else 0.5
            continue
        mu, sd, comp = _features_fit(fx[train])
        w = _logistic_fit(((fx[train] - mu) / sd) @ comp, y[train])
        z = ((x[test] - mu) / sd) @ comp
        out[test] = np.c_[np.ones(len(z)), z] @ w
    return out


def auroc(scores, labels) -> float:
    s = np.asarray(scores, np.float64)
    y = np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.r_[pos, neg], kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.r_[pos, neg][order]
    # average ranks for ties
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auroc_interval(scores, labels, roots, *, resamples=2000, level=0.95) -> dict:
    """AUROC with a root-bootstrap percentile interval (generator keyed by salt 9004)."""
    roots = np.asarray(roots)
    unique = np.unique(roots)
    index = {int(r): np.flatnonzero(roots == r) for r in unique}
    rng = np.random.default_rng([SALTS["probe"], 1])
    vals = []
    for _ in range(resamples):
        pick = rng.integers(0, len(unique), len(unique))
        idx = np.concatenate([index[int(unique[i])] for i in pick])
        v = auroc(np.asarray(scores)[idx], np.asarray(labels)[idx])
        if np.isfinite(v):
            vals.append(v)
    a = (1 - level) / 2
    ci = [float(x) for x in np.percentile(vals, [100 * a, 100 * (1 - a)])] if vals else [None, None]
    return {"auroc": auroc(scores, labels), "ci": ci, "positives": int(np.sum(labels))}


# ----- Part B: bar, statistics and rows (§7.2) ---------------------------------------------------
P_THRESHOLD = 0.01
DELTA = 0.10
BOOTSTRAP_RESAMPLES = 10_000
SEEDS_PER_ARM = 3


def pooled_bar(bar: float, n: int) -> int:
    return SEEDS_PER_ARM * zs.bar_count(bar, n)


def seed_floor(bar: float, n: int) -> int:
    return int(math.ceil(zs.bar_count(bar, n) / 2 - 1e-9))


def _signflip_survival(c: tuple) -> tuple[int, np.ndarray]:
    """Null distribution of S = sum of s_r |d_r| (independent fair signs) for magnitude counts
    ``c[k]`` of |d| = k + 1: returns (offset, P(S >= offset + i)) over S's integer support."""
    pmf = np.array([1.0])
    lo = 0
    for k, count in enumerate(c, start=1):
        for _ in range(int(count)):
            new = np.zeros(len(pmf) + 2 * k)
            new[: len(pmf)] += 0.5 * pmf
            new[2 * k :] += 0.5 * pmf
            pmf = new
            lo -= k
    survival = np.cumsum(pmf[::-1])[::-1]
    return lo, survival


_SF_CACHE: dict = {}


def signflip_p(d) -> float:
    """Exact one-sided sign-flip permutation p that sum(d) > 0 under the null of symmetric
    per-reset differences: P(S >= sum(d)) with S = sum of independent sign flips of the nonzero
    d_r (enumerated by convolution, not sampled). ``d`` are integers."""
    d = np.asarray(d, np.int64)
    mags = np.abs(d[d != 0])
    if len(mags) == 0:
        return 1.0
    kmax = int(mags.max())
    c = tuple(int((mags == k).sum()) for k in range(1, kmax + 1))
    if c not in _SF_CACHE:
        _SF_CACHE[c] = _signflip_survival(c)
    lo, survival = _SF_CACHE[c]
    s = int(d.sum())
    i = s - lo
    if i <= 0:
        return 1.0
    if i >= len(survival):
        return 0.0
    return float(min(1.0, survival[i]))


def per_reset(successes_by_seed) -> np.ndarray:
    """``[seeds, n]`` success indicators -> per-reset sums (0-3)."""
    return np.asarray(successes_by_seed, np.int64).sum(0)


def reset_bootstrap(g_by_seed, c_by_seed, *, resamples=BOOTSTRAP_RESAMPLES, level=0.95) -> dict:
    """Reset-cluster bootstrap percentile interval of (k_g - k_c) / 3n (salt 9006)."""
    g, c = per_reset(g_by_seed), per_reset(c_by_seed)
    n = len(g)
    rng = np.random.default_rng([SALTS["bootstrap"]])
    draws = rng.integers(0, n, (resamples, n))
    diff = (g[draws].sum(1) - c[draws].sum(1)) / (SEEDS_PER_ARM * n)
    a = (1 - level) / 2
    lo, hi = np.percentile(diff, [100 * a, 100 * (1 - a)])
    return {"value": float((g.sum() - c.sum()) / (SEEDS_PER_ARM * n)), "ci": [float(lo), float(hi)]}


def pooled_mcnemar(g_by_seed, c_by_seed) -> float:
    """Reported only: exact one-sided McNemar over the 3n (seed, reset) pairs."""
    g = np.asarray(g_by_seed, bool).ravel()
    c = np.asarray(c_by_seed, bool).ravel()
    return zs.mcnemar_one_sided(int((g & ~c).sum()), int((~g & c).sum()))


def b_row(g_by_seed, c_by_seed, random, bar: float, *, void: str = "") -> dict:
    """§7.2's rows, first match. ``g_by_seed`` / ``c_by_seed``: M-v2g's and M-v1p's ``[3, n]``
    success indicators on the same resets (seed i of each on reset r); ``random``: ``[n]``."""
    g = np.asarray(g_by_seed, bool)
    c = np.asarray(c_by_seed, bool)
    r = np.asarray(random, bool)
    if g.shape != c.shape or g.shape[0] != SEEDS_PER_ARM or r.shape != (g.shape[1],):
        raise ContractError("b_row needs [3, n] arms and [n] random")
    n = g.shape[1]
    gr, cr = per_reset(g), per_reset(c)
    k_g, k_c = int(gr.sum()), int(cr.sum())
    need, floor = pooled_bar(bar, n), seed_floor(bar, n)
    seeds = [int(x) for x in g.sum(1)]
    p_c = signflip_p(gr - cr)
    p_r = signflip_p(gr - SEEDS_PER_ARM * r.astype(np.int64))
    beats_c, beats_r = p_c < P_THRESHOLD, p_r < P_THRESHOLD
    at_bar = k_g >= need
    floor_ok = all(s >= floor for s in seeds)
    boot = reset_bootstrap(g, c)
    if void:
        name = "B-VOID"
    elif at_bar and floor_ok and beats_r and beats_c:
        name = "B-PASS"
    elif beats_c:
        name = "B-COMP"
    elif at_bar and beats_r:
        name = "B-SIZE"  # no per-seed floor: it carries no claim
    elif boot["ci"][1] < DELTA:
        name = "B-FAIL"
    else:
        name = "B-UNDER"
    return {
        "row": name,
        "void_reason": void,
        "n": n,
        "bar": bar,
        "k_g": k_g,
        "k_c": k_c,
        "pooled_bar": need,
        "seed_floor": floor,
        "seeds_g": seeds,
        "seeds_c": [int(x) for x in c.sum(1)],
        "seeds_g_interval": [zs.exact_interval(s, n) for s in seeds],
        "p_beats_v1p": p_c,
        "p_beats_random": p_r,
        "beats_v1p": bool(beats_c),
        "beats_random": bool(beats_r),
        "at_pooled_bar": bool(at_bar),
        "seed_floor_met": bool(floor_ok),
        "bootstrap_rate_difference": boot,
        "pooled_mcnemar_p_reported": pooled_mcnemar(g, c),
    }


def clause_fires(row: str) -> bool:
    """§9: the abandonment clause fires on B-FAIL only."""
    return row == "B-FAIL"


# ----- the power rule (§7.2) ---------------------------------------------------------------------
POWER_ALTERNATIVE = (0.42, 0.25)
POWER_SDS = (0.0, 1.0, 2.0)
POWER_TARGET = 0.80


def _intercept_for_rate(rate: float, sd: float) -> float:
    """The logit intercept whose marginal rate over N(0, sd) logit effects equals ``rate``
    (Gauss-Hermite quadrature, bisection)."""
    if sd == 0:
        return math.log(rate / (1 - rate))
    x, w = np.polynomial.hermite_e.hermegauss(80)
    w = w / w.sum()

    def marginal(a):
        return float((w / (1 + np.exp(-(a + sd * x)))).sum())

    lo, hi = -20.0, 20.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if marginal(mid) < rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def power(n: int, sd: float, *, alt=POWER_ALTERNATIVE, sims: int = 20_000, seed: int = 0) -> dict:
    """Power of "M-v2g beats M-v1p" (the exact sign-flip test over per-reset sums, p < 0.01) at
    per-seed marginal success rates ``alt``, with independent per-arm, per-reset N(0, sd) logit
    effects shared by an arm's three seeds."""
    rng = np.random.default_rng([SALTS["bootstrap"], int(n), int(round(10 * sd)), seed])
    ag, ac = (_intercept_for_rate(p, sd) for p in alt)
    zg_ = rng.normal(0, sd, (sims, n)) if sd else np.zeros((sims, n))
    zc_ = rng.normal(0, sd, (sims, n)) if sd else np.zeros((sims, n))
    pg_ = 1 / (1 + np.exp(-(ag + zg_)))
    pc_ = 1 / (1 + np.exp(-(ac + zc_)))
    g = (rng.random((sims, SEEDS_PER_ARM, n)) < pg_[:, None]).sum(1)
    c = (rng.random((sims, SEEDS_PER_ARM, n)) < pc_[:, None]).sum(1)
    hits = sum(signflip_p(g[i] - c[i]) < P_THRESHOLD for i in range(sims))
    return {
        "n": n,
        "sd": sd,
        "power": hits / sims,
        "sims": sims,
        "marginal_rates": [float(pg_.mean()), float(pc_.mean())],
        "intercepts": [ag, ac],
    }


def power_rule(table: dict) -> dict:
    """§7.2: n = 64 unless the power at SD 1 is below 0.80 there; then n = 128; escalate if it
    is below 0.80 at 128 too. ``table[n][sd]`` is the power."""
    if table[64][1.0] >= POWER_TARGET:
        return {"n": 64, "escalate": False}
    return {"n": 128, "escalate": bool(table[128][1.0] < POWER_TARGET)}


# ----- record helpers ----------------------------------------------------------------------------
def write_json(path, value) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=True) + "\n")


def project_stage_c(
    v1_train, debug_grasp, *, added=2880, resamples=2000, seed=930001, bars=None
) -> dict:
    """Reported only (Stage 0): Stage C's ratios at full size, by resampling. play-v2g's added
    train episodes are drawn from the debug grasp episodes, play-v1p's from ``play-v1`` train
    (the same policy, whose code is unchanged), ``added`` each; play-v1 train is resampled too.
    Returns each ratio's median and 95 % range and the share of draws that meet the bars."""
    bars = bars or {"C1": C1_BAR, "C2": C2_BAR, "C3": C3_BAND}
    rng = np.random.default_rng(seed)
    l1 = np.array([r["lift_windows"] for r in v1_train], np.float64)
    w1 = np.array([r["windows"] for r in v1_train], np.float64)
    lg = np.array([r["lift_windows"] for r in debug_grasp], np.float64)
    wg = np.array([r["windows"] for r in debug_grasp], np.float64)
    draws = {"C1": [], "C2": [], "C3": []}
    for _ in range(resamples):
        i1 = rng.integers(0, len(l1), len(l1))
        ip = rng.integers(0, len(l1), added)
        ig = rng.integers(0, len(lg), added)
        a1, b1 = l1[i1].sum(), w1[i1].sum()
        ap, bp = l1[ip].sum(), w1[ip].sum()
        ag, bg = lg[ig].sum(), wg[ig].sum()
        draws["C1"].append((ag / bg) / (ap / bp))
        draws["C2"].append(((a1 + ag) / (b1 + bg)) / ((a1 + ap) / (b1 + bp)))
        draws["C3"].append((ap / bp) / (a1 / b1))
    out = {}
    for k, v in draws.items():
        v = np.asarray(v)
        q = np.percentile(v, [2.5, 50, 97.5])
        if k == "C3":
            meets = np.mean((v >= 1 - bars["C3"]) & (v <= 1 + bars["C3"]))
        else:
            meets = np.mean(v >= bars[k])
        out[k] = {
            "median": float(q[1]),
            "range95": [float(q[0]), float(q[2])],
            "share_meeting_bar": float(meets),
        }
    out["bars"] = bars
    out["resamples"] = resamples
    out["added"] = added
    return out
