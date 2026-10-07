"""TASK-080's offline parts (protocol ``docs/experiments/apple_lewm_c1m_v2_pred_readout.md`` §4,
§5, §9.4): the fresh corpus's store and featurisation, the readouts on predicted latents (R-S,
R-N, R-L), their evaluation, the offline aims' statistics and the echo slope.

TASK-077's offline module is imported, never edited: ``encode_tokens``, ``anchor_check``,
``load_root``, ``predict_at``, ``errors_cm``, ``collapse_statistics`` and the feature-file layout
are reused. Every ridge here is Stage O's dual ridge (the relative lambda grid, 5 inner folds
grouped by root) with this task's inner-fold salt 8205; every interval uses salt 8206.

**G-fresh**: :func:`fit_readouts` refuses any fit root of the fresh corpus (or of TASK-080's seed
block); the fresh corpus is read by Stage R's gates and reports only.

NumPy at import; torch is imported inside the functions that need it.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from embodied_jepa import lewm_c1m_v2 as lm
from embodied_jepa import lewm_c1m_v2_offline as off
from embodied_jepa import lewm_c1m_v2_train as tr
from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa import lewm_pr_v2 as pr
from embodied_jepa.contracts import ContractError

GuardError = pr.GuardError
START_INDEX = off.START_INDEX  # frame 405
R_INDEX = off.R_INDEX  # frame 465
READOUT_NAMES = ("r_s", "r_n", "r_l")


def sha256_file(path) -> str:
    return off.sha256_file(path)


def array_sha256(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ----- roots: the old corpus (fits, dry run) and the fresh one (Stage R) -------------------------
def load_roots(features, splits, *, limit: int | None = None, full405: bool = False) -> dict:
    """Per-root arrays of the given splits of a featurisation folder, in split then root order:
    the 405 and r latents (8 x 8), the true plate and palm at 405 and r, the committed aim, the
    logged 405 state, apple estimate and last grasp; with ``full405`` the full tokens at 405
    (memory-mapped, for R-plate's reading)."""
    features = Path(features)
    parts = []
    for split in splits:
        store = tr.RootStore.open(features, split, mmap=True)
        table = off.load_table(features, split)
        n = len(store) if limit is None else min(int(limit), len(store))
        if tuple(int(s) for s in table["seeds"][:n]) != store.roots[:n]:
            raise GuardError(f"G-split: the {split} table and store disagree on the roots")
        part = {
            "seeds": np.asarray(table["seeds"][:n], np.int64),
            "split": np.full(n, split),
            "start": np.asarray(store.features[:n, START_INDEX], np.float32),
            "enc_r": np.asarray(store.features[:n, R_INDEX], np.float32),
            "commands": np.asarray(store.commands[:n, START_INDEX : START_INDEX + pr.HORIZON]),
            "plate405": table["plate"][:n, START_INDEX],
            "plate_r": table["plate"][:n, R_INDEX],
            "palm405": table["palm"][:n, START_INDEX],
            "target": table["target"][:n],
            "state405": table["state405"][:n],
            "apple": table["apple"][:n],
            "last_grasp": table["last_grasp"][:n],
        }
        if "p_hat405" in table:
            part["p_hat405"] = table["p_hat405"][:n]
        if full405:
            part["full405"] = np.load(features / f"full405_{split}.npy", mmap_mode="r")[:n]
        parts.append(part)
    out = {}
    for key in parts[0]:
        if key == "full405":
            out[key] = [p[key] for p in parts]
        elif all(key in p for p in parts):
            out[key] = np.concatenate([p[key] for p in parts])
    out["n"] = int(len(out["seeds"]))
    return out


def r_plate_readings(r_plate, full405_parts) -> np.ndarray:
    """R-plate's reading at 405 from the stored full tokens (as TASK-077's offline aims did)."""
    rows = []
    for part in full405_parts:
        for i in range(len(part)):
            rows.append(np.asarray(r_plate.predict(np.asarray(part[i], np.float64))).reshape(2))
    return np.stack(rows)


def chunk_tasks(data: dict, targets=None) -> list[dict]:
    targets = data["target"] if targets is None else targets
    return [
        {
            "kind": "chunks",
            "key": i,
            "state": data["state405"][i].tolist(),
            "apple": data["apple"][i].tolist(),
            "last_grasp": data["last_grasp"][i].tolist(),
            "step": pr.COMMIT_STEP,
            "targets": [np.asarray(targets[i]).tolist()],
        }
        for i in range(data["n"])
    ]


def collect_chunks(results: list[dict], n: int) -> tuple[np.ndarray, np.ndarray]:
    got = sorted(results, key=lambda r: r["key"])
    if [r["key"] for r in got] != list(range(n)):
        raise GuardError("G-chunks: a stand-in chunk is missing")
    chunks = np.stack([np.asarray(r["chunks"][0], np.float32) for r in got])
    feasible = np.asarray([bool(r["feasible"][0]) for r in got])
    if chunks.shape != (n, pr.HORIZON, 14):
        raise GuardError(f"G-chunks: stand-in chunks {chunks.shape}")
    return chunks, feasible


# ----- roll-outs and the readouts on predicted latents (§4) --------------------------------------
def rollouts(w_model, n_model, start, chunks, mean_latent) -> dict:
    """W^s from each root's 405 latent under its stand-in chunk (S), N^s under zero commands (N),
    and W^s from L-mean's mean latent under the same chunks (L): the predicted raw 8 x 8 latent at
    r (CPU, float32, the caller's one torch thread)."""
    h = pr.HORIZON
    start = np.asarray(start, np.float32)
    chunks = np.asarray(chunks, np.float32)
    mean = np.repeat(np.asarray(mean_latent, np.float32).reshape(1, -1), len(start), 0)
    return {
        "S": off.predict_at(w_model, start, chunks, (h,))[h],
        "N": off.predict_at(n_model, start, np.zeros_like(chunks), (h,))[h],
        "L": off.predict_at(w_model, mean, chunks, (h,))[h],
    }


def ridge(x, y, groups):
    """Stage O's dual ridge with this task's inner-fold salt (8205)."""
    from embodied_jepa.models.latent_critic import RidgeReadout

    return RidgeReadout.fit(
        x,
        y,
        groups,
        lambdas=pr.LAMBDA_GRID_RELATIVE,
        folds=pr.INNER_FOLDS,
        seed=pr.SALTS["inner_folds"],
        dual=True,
    )


def fit_readouts(pred: dict, plate_r, seeds) -> dict:
    """R-S, R-N and R-L of one model seed on the fit roots (G-fresh: old roots only)."""
    pr.check_fit_roots(seeds)
    groups = np.asarray(seeds).astype(str)
    y = np.asarray(plate_r, np.float64)
    return {
        name: ridge(pred[key], y, groups) for name, key in zip(READOUT_NAMES, "SNL", strict=True)
    }


def save_readouts(folder, seed: int, readouts: dict) -> dict:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, readout in readouts.items():
        path = folder / f"{name}_{int(seed)}.npz"
        out[name] = {
            "path": str(path),
            "sha256": off.save_readout(path, readout),
            "selection": readout.selection,
        }
    return out


def load_readout(path, sha256: str):
    from embodied_jepa.models.latent_critic import RidgeReadout

    with np.load(path) as data:
        readout = RidgeReadout.from_state({k: data[k] for k in data.files})
    if readout.sha256() != sha256:
        raise GuardError(f"G-readout: {Path(path).name} differs from its recorded sha256")
    return readout


def readout_errors(readouts: dict, pred: dict, plate_r) -> dict:
    """e_S, e_N, e_L (cm) of each readout on its own model's predictions (§9.4)."""
    y = np.asarray(plate_r, np.float64)
    return {
        "e_S": off.errors_cm(readouts["r_s"].predict(pred["S"]), y),
        "e_N": off.errors_cm(readouts["r_n"].predict(pred["N"]), y),
        "e_L": off.errors_cm(readouts["r_l"].predict(pred["L"]), y),
    }


def readout_statistics(e: dict) -> dict:
    """R1-R3's statistics (salt 8206) and the reported 1.5 tau reading's inputs."""
    return {
        "e_S": pr.median_ci(e["e_S"]),
        "e_N": pr.median_ci(e["e_N"]),
        "e_L": pr.median_ci(e["e_L"]),
        "S_over_N": pr.median_ratio_ci(e["e_S"], e["e_N"]),
    }


def crossfit(pred: dict, plate_r, seeds, split) -> dict:
    """Reported only (§4.2): each readout cross-fitted over the fit roots in 5 outer folds (salt
    8204), its held-out median error by split."""
    pr.check_fit_roots(seeds)
    n = len(seeds)
    fold = pr.outer_folds(n)
    groups = np.asarray(seeds).astype(str)
    y = np.asarray(plate_r, np.float64)
    out = {}
    for name, key in zip(READOUT_NAMES, "SNL", strict=True):
        held = np.full((n, 2), np.nan)
        for k in range(pr.OUTER_FOLDS):
            fit, rows = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            held[rows] = ridge(pred[key][fit], y[fit], groups[fit]).predict(pred[key][rows])
        if not np.isfinite(held).all():
            raise GuardError("G-finite: a cross-fitted readout left a root without a prediction")
        e = off.errors_cm(held, y)
        out[name] = {s: pr.median_ci(e[np.asarray(split) == s]) for s in np.unique(split)}
        out[name]["all"] = pr.median_ci(e)
    return {"fold_sizes": np.bincount(fold).tolist(), "readouts": out}


def learning_curve(pred_fit: dict, plate_fit, seeds_fit, pred_eval: dict, plate_eval) -> dict:
    """Reported only (§4.2): each readout fitted on nested 1/4, 1/2, 3/4 and all of the fit
    roots (salt 8208), its median error on the evaluated roots."""
    pr.check_fit_roots(seeds_fit)
    subsets = pr.nested_subsets(len(seeds_fit))
    groups = np.asarray(seeds_fit).astype(str)
    y = np.asarray(plate_fit, np.float64)
    out = {}
    for name, key in zip(READOUT_NAMES, "SNL", strict=True):
        out[name] = {}
        for f, rows in subsets.items():
            model = ridge(pred_fit[key][rows], y[rows], groups[rows])
            e = off.errors_cm(model.predict(pred_eval[key]), plate_eval)
            out[name][str(f)] = {"fit_roots": int(len(rows)), "median_cm": float(np.median(e))}
    return out


# ----- the offline aims (§9.4) --------------------------------------------------------------------
def offline_aim_tasks(arm, data, p_hat, latents, tau_commit_cm, *, sysid_coef=None, keep=False):
    """One task per evaluated root for ``arm``: its logged 405 state, p̂ and the palm at 405; W and
    N start from the root's own latent, L-shuf from the next root's (cyclically), L-mean from the
    mean latent."""
    n = data["n"]
    tasks = []
    for i in range(n):
        task = {
            "kind": "offline_aim",
            "key": i,
            "arm": arm,
            "seed": int(data["seeds"][i]),
            "state": data["state405"][i].tolist(),
            "apple": data["apple"][i].tolist(),
            "last_grasp": data["last_grasp"][i].tolist(),
            "p_hat": np.asarray(p_hat[i]).tolist(),
            "h": data["palm405"][i].tolist(),
            "tau_commit_cm": float(tau_commit_cm),
        }
        if arm in ("W", "N"):
            task["start"] = np.asarray(latents[i], np.float32)
        elif arm == "L-shuf":
            task["start"] = np.asarray(latents[lm.foreign_index(i, n)], np.float32)
        elif arm == "L-mean":
            task["start"] = "mean"
        if arm == "H-sysid":
            task["sysid_coef"] = np.asarray(sysid_coef, np.float64).tolist()
        if keep and arm == "W":
            task["keep_plates"] = True
        tasks.append(task)
    return tasks


def g_star(data) -> np.ndarray:
    """The rule's fixed point g* = (p - kappa h) / (1 - kappa) from the true plate and palm."""
    return np.stack(
        [c1.fixed_point(p, h) for p, h in zip(data["plate405"], data["palm405"], strict=True)]
    )


def aim_summary(results: list[dict], data, tau_counts: dict, resets: int) -> dict:
    """An arm's offline aims: the aim error against g*, its interval (salt 8206), 87.5th
    percentile, the clip-binding fraction and the predicted count of 64 through the tau curve
    (a prediction, not a closed-loop count)."""
    got = sorted(results, key=lambda r: r["key"])
    if [r["key"] for r in got] != list(range(data["n"])):
        raise GuardError("G-aims: an offline aim is missing")
    aims = np.asarray([r["g"] for r in got], np.float64)
    errors = 100.0 * np.linalg.norm(aims - g_star(data), axis=1)
    return {
        "aim_error_cm": pr.median_ci(errors),
        "predicted_count_of_64": pr.predicted_count(errors, tau_counts, resets),
        "clip_binding_fraction": float(np.mean([bool(r["log"].get("clipped")) for r in got])),
        "fallbacks": int(sum(bool(r["log"].get("fallback_all_infeasible")) for r in got)),
        "aim_errors_cm": errors.round(6).tolist(),
    }


def echo_slopes(results: list[dict]) -> dict:
    """Reported (§5.3 point 4): per root, the 2 x 2 least-squares slope A of W's predicted plate
    p~(g) against the aim g over the feasible grid (p~ = A g + c), against kappa I (the rule's
    fixed-point form) and I (an echo of the aim)."""
    kappa = float(pr.RULE["kappa"])
    slopes = []
    for r in results:
        g = np.asarray(r.get("grid_targets", []), np.float64)
        p = np.asarray(r.get("grid_plates", []), np.float64)
        if len(g) < 3:
            continue
        x = np.hstack([g, np.ones((len(g), 1))])
        coef, *_ = np.linalg.lstsq(x, p, rcond=None)
        slopes.append(coef[:2].T)  # p = A g + c: A[i, j] = d p_i / d g_j
    if not slopes:
        return {"roots": 0}
    a = np.stack(slopes)
    eye = np.eye(2)
    return {
        "roots": int(len(a)),
        "median_matrix": np.median(a, axis=0).tolist(),
        "trace_over_2": pr.median_ci(np.trace(a, axis1=1, axis2=2) / 2.0),
        "frobenius_to_kappa_I": pr.median_ci(np.linalg.norm(a - kappa * eye, axis=(1, 2))),
        "frobenius_to_I": pr.median_ci(np.linalg.norm(a - eye, axis=(1, 2))),
        "kappa": kappa,
    }


# ----- G1-G4 on fresh roots (reported only, §9.4; TASK-077 §8.2's definitions) -----------------
def comparative_rank(w, n_, e, *, resamples: int = lm.COMPARATIVE_RESAMPLES) -> dict:
    """``lewm_c1m_v2_offline.comparative_rank``'s statistic with salt 8206."""
    grams = {k: np.asarray(x, np.float64) @ np.asarray(x, np.float64).T for k, x in
             (("W", w), ("N", n_), ("E", e))}  # fmt: skip
    m = len(e)
    rng = np.random.default_rng(pr.SALTS["bootstrap"])

    def rank(g, idx):
        sub = g[np.ix_(idx, idx)]
        j = np.full((len(idx), len(idx)), 1.0 / len(idx))
        centred = sub - j @ sub - sub @ j + j @ sub @ j
        return off.effective_rank_from_energy(np.clip(np.linalg.eigvalsh(centred), 0.0, None))

    full = np.arange(m)
    point = {k: rank(g, full) for k, g in grams.items()}
    diffs, undefined = [], 0
    for _ in range(int(resamples)):
        idx = rng.integers(0, m, m)
        r = {k: rank(g, idx) for k, g in grams.items()}
        if r["E"] <= 0:
            undefined += 1
            continue
        diffs.append(r["W"] / r["E"] - r["N"] / r["E"])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "point": point["W"] / point["E"] - point["N"] / point["E"],
        "ci95": [float(lo), float(hi)],
        "undefined_resamples": int(undefined),
        "resamples": int(resamples),
    }


def dynamics_report(w_model, n_model, store, scale, bars) -> dict:
    """G1-G4 at h = 60 on fresh roots (reported, R18.7), with the executed commands, N, copy-last
    and G4's wrong (salt 8210) and zero commands."""
    h = pr.HORIZON
    start, commands, targets = store.from_405()
    start = np.asarray(start)
    commands = np.asarray(commands, np.float32)
    perm = pr.wrong_permutation(len(store))
    zero = np.zeros_like(commands)
    pred = {
        "W_true": off.predict_at(w_model, start, commands, (h,))[h],
        "W_wrong": off.predict_at(w_model, start, commands[perm], (h,))[h],
        "W_zero": off.predict_at(w_model, start, zero, (h,))[h],
        "N": off.predict_at(n_model, start, zero, (h,))[h],
    }
    scale = np.asarray(scale, np.float64)
    target = np.asarray(targets[:, h - 1], np.float64)

    def err(p):
        d = (np.asarray(p, np.float64) - target) / scale
        return (d * d).mean(1)

    e = {k: err(v) for k, v in pred.items()} | {"copy": err(start)}
    stats = off.collapse_statistics(pred["W_true"] / scale, target / scale)
    comparative = comparative_rank(pred["W_true"] / scale, pred["N"] / scale, target / scale)
    g1 = lm.g1_passes(stats, bars, comparative)
    g = {
        "G2": pr.ratio_of_sums_ci(e["W_true"], e["copy"]),
        "G3": pr.ratio_of_sums_ci(e["W_true"], e["N"]),
        "G4_wrong": pr.ratio_of_sums_ci(e["W_wrong"], e["W_true"]),
        "G4_zero": pr.ratio_of_sums_ci(e["W_zero"], e["W_true"]),
    }
    t = lm.G_THRESHOLDS
    passes = {
        "G1": g1["passes"],
        "G2": bool(g["G2"]["ci95"][1] <= t["G2_max_ratio_upper"]),
        "G3": bool(g["G3"]["ci95"][1] < t["G3_max_ratio_upper_exclusive"]),
        "G4": bool(
            g["G4_wrong"]["ci95"][0] >= t["G4_min_ratio_lower"]
            and g["G4_zero"]["ci95"][0] >= t["G4_min_ratio_lower"]
        ),
    }
    return {"reported_only": True, "passes": passes, "G1": {"stats": stats,
            "comparative": comparative, "parts": g1}, "ratios": g}  # fmt: skip


# ----- the fresh corpus store and its featurisation (Stage C′, Stage R) --------------------------
ROOT_KEYS = (*off.ROOT_KEYS, "p_hat405")


def write_root(folder, seed: int, arrays: dict) -> str:
    """One fresh root's npz (TASK-077's layout plus p̂ at 405); refuses to overwrite."""
    path = Path(folder) / f"{int(seed)}.npz"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    missing = [k for k in ROOT_KEYS if k not in arrays]
    if missing:
        raise ContractError(f"a fresh root needs {missing}")
    shapes = {
        "frames": (lm.N_FRAMES, 112, 112, 3),
        "commands": (lm.N_COMMANDS, 14),
        "plate": (lm.N_FRAMES, 2),
        "palm": (lm.N_FRAMES, 2),
        "hidden_r": (112, 112, 3),
        "p_hat405": (2,),
    }
    for key, shape in shapes.items():
        if tuple(np.shape(arrays[key])) != shape:
            raise ContractError(f"{key} must be {shape}, not {np.shape(arrays[key])}")
    np.savez_compressed(path, **{k: np.asarray(arrays[k]) for k in ROOT_KEYS})
    return sha256_file(path)


def seal_corpus(folder, entries: dict, split: dict, provenance: dict) -> dict:
    folder = Path(folder)
    path = folder / "manifest.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    manifest = {
        "protocol": pr.PROTOCOL,
        "corpus": "apple-c1m-v2-f",
        "roots": {str(s): entries[s] for s in sorted(entries)},
        "split": {k: [int(s) for s in v if int(s) in entries] for k, v in split.items()},
        "split_declared": {k: [int(s) for s in v] for k, v in split.items()},
        "aim_from": dict(pr.AIM_FROM),
        "provenance": provenance,
    }
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return {"path": str(path), "sha256": sha256_file(path)}


def open_corpus(folder, expected_sha256: str | None) -> dict:
    path = Path(folder) / "manifest.json"
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise GuardError("G-split: the fresh corpus manifest differs from its sealed sha256")
    manifest = json.loads(path.read_text())
    if manifest.get("protocol") != pr.PROTOCOL or sorted(manifest["split"]) != sorted(pr.HALVES):
        raise GuardError("G-split: not TASK-080's fresh corpus")
    return manifest | {"_sha256": digest}


def featurise_fresh(corpus, manifest: dict, out, *, device: str, encoder=None, check=None):
    """Both halves' per-root 8 x 8 stores and side arrays, in TASK-077's featurisation layout
    (``features8_<half>.npy`` and the rest; the table adds p̂ at 405), with G-anchor on the first
    256 kept frames. Refuses to overwrite."""
    from numpy.lib.format import open_memmap

    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa.models.frozen_tokens import pool_tokens

    started = time.monotonic()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    encoder = pe.load_pretrained() if encoder is None else encoder
    encoder.to(device).eval()
    files, roots = {}, {}
    need = int(pr.FEATURE_ANCHOR["frames"])
    anchor_frames, anchor_pooled = [], []
    for half in pr.HALVES:
        seeds = [int(s) for s in manifest["split"][half]]
        n = len(seeds)
        paths = off.feature_paths(out, half)
        for path in paths.values():
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
        feats = open_memmap(paths["features"], "w+", np.float32, (n, lm.N_FRAMES, lm.LATENT_DIM))
        full405 = open_memmap(paths["full405"], "w+", np.float32, (n, off.FULL_WIDTH))
        hidden8 = np.empty((n, lm.LATENT_DIM), np.float32)
        pool4 = np.empty((n, 2, 16 * 384), np.float32)
        commands = np.empty((n, lm.N_COMMANDS, 14), np.float32)
        side = {k: [] for k in ("plate", "palm", "target", "state405", "apple", "last_grasp",
                                "p_hat405")}  # fmt: skip
        for i, seed in enumerate(seeds):
            root = off.load_root(corpus, seed, manifest["roots"][str(seed)])
            stack = np.concatenate([root["frames"], root["hidden_r"][None]])
            tokens = off.encode_tokens(encoder, stack, device=device)
            pooled = pool_tokens(tokens, lm.TOKEN_GRID)
            feats[i] = pooled[: lm.N_FRAMES]
            hidden8[i] = pooled[lm.N_FRAMES]
            pool4[i] = pool_tokens(tokens[[START_INDEX, R_INDEX]], 4)
            full405[i] = tokens[START_INDEX]
            commands[i] = root["commands"]
            for key, src in (("plate", "plate"), ("palm", "palm"), ("target", "target"),
                             ("state405", "state405"), ("apple", "apple_estimate"),
                             ("last_grasp", "last_grasp"), ("p_hat405", "p_hat405")):  # fmt: skip
                side[key].append(root[src])
            take = min(need - sum(len(a) for a in anchor_frames), lm.N_FRAMES)
            if take > 0:
                anchor_frames.append(root["frames"][:take].copy())
                anchor_pooled.append(pooled[:take].copy())
            if check is not None and i % 50 == 0:
                check()
            del tokens, pooled, stack
            if (i + 1) % off.FEATURE_REMAP_ROOTS == 0 and i + 1 < n:
                feats = off._remap(feats, paths["features"])
                full405 = off._remap(full405, paths["full405"])
        feats.flush()
        full405.flush()
        del feats, full405
        np.save(paths["commands"], commands)
        np.save(paths["hidden8"], hidden8)
        np.save(paths["pool4"], pool4)
        paths["roots"].write_text(json.dumps(seeds))
        np.savez(
            paths["table"],
            seeds=np.asarray(seeds, np.int64),
            **{k: np.asarray(v, np.float64) for k, v in side.items()},
        )
        roots[half] = n
        files[half] = {k: sha256_file(p) for k, p in paths.items()}
    anchor = None
    if anchor_frames:
        anchor = off.anchor_check(
            encoder, np.concatenate(anchor_frames), np.concatenate(anchor_pooled)
        )
    return {
        "device": str(device),
        "files_sha256": files,
        "roots": roots,
        "anchor": anchor,
        "seconds": time.monotonic() - started,
    }
