"""TASK-076 Stage O: offline plate admission on the sealed stores (no render, no simulation).

Protocol ``docs/experiments/apple_plate_twin_v2.md`` §5 (Stage O) and §6.1. Stored frames only.
NumPy at import; torch is imported by the featurisation (``obs_ceiling_v2_offline.featurise``).

* **Rows.** Each read root of ``apple-far-shift-v2`` (train and val; the test roots are never
  decoded) contributes its onboard 112 px decision frames 405-485 that lie inside its band, with
  the true plate xy at that step; and its stored plate-hidden onboard frames at 421-501 from
  ``apple-far-shift-v2-views``, with the true plate xy at that step.
* **Features.** Frozen DINOv2 ViT-S/14 full patch tokens (98 304-d; pretrained and the seed-0
  random-init floor) and the pooled 4 x 4 latent (6 144-d), on CUDA with a CPU anchor check.
* **Cross-fit.** 5 outer folds by root (salt 7602). In each, R_plate, R_plate_floor (full tokens)
  and R_plate_pool (pooled) are fitted on the other folds (dual ridge, lambda by grouped inner CV,
  salt 7603) and read the held-out roots' decision frames, and R_plate also their plate-hidden
  frames; the clock prior is the per-step median plate of the other folds.
* **The closed-loop readouts.** R_plate and R_plate_floor fitted once on every read root's
  decision frames, after the stage boundary; their sha256 is recorded.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from embodied_jepa import lewm_planner_v2 as lp
from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa.contracts import ContractError

FULL_WIDTH = 256 * 384
POOL_WIDTH = 6144
BAND = lp.FEATURE_BAND  # (384, 560): TASK-075's root rule
MIN_HI = BAND[0] + lp.CHUNK  # a root whose band ends at or before 400 is not read


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ViewsReader:
    """The sealed views store (TASK-075's format): every read verifies the file's sha256."""

    def __init__(self, path, manifest_sha256: str):
        self.path = Path(path)
        if sha256_file(self.path / "manifest.json") != manifest_sha256:
            raise lp.GuardError("G-data: the views manifest differs from its sealed hash")
        self.manifest = json.loads((self.path / "manifest.json").read_text())

    def arrays(self, episode_id: str, keys) -> dict:
        npz = self.path / "episodes" / f"{episode_id}.npz"
        if sha256_file(npz) != self.manifest["episodes"][episode_id]["npz_sha256"]:
            raise lp.GuardError(f"G-data: {episode_id}'s views differ from their sealed hash")
        with np.load(npz) as data:
            return {k: data[k] for k in keys}


def outer_folds(root_ids) -> dict:
    """5 outer folds by root: a seeded permutation (salt 7602) of the read roots in plan order,
    position mod 5."""
    order = np.random.default_rng(pt.SALTS["outer_folds"]).permutation(len(root_ids))
    fold = np.empty(len(root_ids), np.int64)
    fold[order] = np.arange(len(root_ids)) % pt.OUTER_FOLDS
    return {r: int(f) for r, f in zip(root_ids, fold, strict=True)}


def load_rows(source, views, clock=None) -> dict:
    """Every read root's decision frames and plate-hidden frames, with their labels."""
    vis_frames, vis_plate, vis_root, vis_step = [], [], [], []
    hid_frames, hid_plate, hid_root, hid_step = [], [], [], []
    roots, splits = [], {}
    for split in pt.SOURCE_CORPUS["read_splits"]:
        for episode_id in source.manifest["splits"][split]:
            if clock is not None:
                clock.check("rows")
            arrays, _meta = source.episode(episode_id, keys=("frames", "plate"))
            length = len(arrays["plate"])
            hi = min(BAND[1], length - 1)
            if hi <= MIN_HI:
                del arrays
                continue
            plate = np.asarray(arrays["plate"], np.float64)[:, :2]
            steps = [t for t in pt.DECISION_STEPS if t <= hi]
            for t in steps:
                vis_frames.append(np.asarray(arrays["frames"][t], np.uint8).copy())
                vis_plate.append(plate[t])
                vis_root.append(episode_id)
                vis_step.append(t)
            del arrays
            want = [t for t in pt.EVAL_STEPS if t <= hi]
            got = views.arrays(episode_id, ("hidden_steps", pt.VIEWS_STORE["hidden_key"]))
            stored = [int(s) for s in got["hidden_steps"]]
            if [s for s in stored if s <= hi] != want:
                raise lp.GuardError(f"G-data: {episode_id}'s hidden steps are not the frozen ones")
            for t in want:
                hid_frames.append(got[pt.VIEWS_STORE["hidden_key"]][stored.index(t)].copy())
                hid_plate.append(plate[t])
                hid_root.append(episode_id)
                hid_step.append(t)
            roots.append(episode_id)
            splits[episode_id] = split
    if not roots:
        raise lp.GuardError("G-data: no read root reaches the decision steps")
    return {
        "roots": roots,
        "splits": splits,
        "vis": {
            "frames": np.stack(vis_frames),
            "plate": np.asarray(vis_plate, np.float64),
            "root": np.asarray(vis_root),
            "step": np.asarray(vis_step, np.int64),
        },
        "hid": {
            "frames": np.stack(hid_frames),
            "plate": np.asarray(hid_plate, np.float64),
            "root": np.asarray(hid_root),
            "step": np.asarray(hid_step, np.int64),
        },
    }


def featurise_rows(rows: dict, encoder, floor, *, device: str) -> dict:
    """Full tokens and pooled latents (pretrained) of every visible and hidden row, the floor's
    full tokens of the visible rows, and the CPU anchor check."""
    from embodied_jepa import obs_ceiling_v2_offline as off

    started = time.monotonic()
    every = slice(0, FULL_WIDTH)
    pooled, full = off.featurise(encoder, rows["vis"]["frames"], device=device, columns=every)
    h_pooled, h_full = off.featurise(encoder, rows["hid"]["frames"], device=device, columns=every)
    del h_pooled
    _fp, floor_full = off.featurise(floor, rows["vis"]["frames"], device=device, columns=every)
    anchor = off.anchor_check(encoder, rows["vis"]["frames"][:32], pooled[:32])
    return {
        "full": full,
        "pooled": pooled,
        "hidden_full": h_full,
        "floor_full": floor_full,
        "anchor": anchor,
        "seconds": time.monotonic() - started,
        "frames": int(len(rows["vis"]["frames"]) + len(rows["hid"]["frames"])),
    }


def ridge(x, y, groups):
    from embodied_jepa.models.latent_critic import RidgeReadout

    return RidgeReadout.fit(
        x,
        y,
        groups,
        lambdas=pt.LAMBDA_GRID_RELATIVE,
        folds=pt.INNER_FOLDS,
        seed=pt.SALTS["inner_folds"],
        dual=True,
    )


def clock_prior(y_fit, steps_fit, steps_held) -> np.ndarray:
    """The per-step median plate of the fit rows, at each held row's step (reads no image)."""
    out = np.empty((len(steps_held), 2))
    for t in np.unique(steps_held):
        mask = steps_fit == t
        if not mask.any():
            raise ContractError(f"the clock prior has no fit rows at step {t}")
        out[steps_held == t] = np.median(y_fit[mask], axis=0)
    return out


def cross_fit(rows: dict, feats: dict, folds: dict, log=None) -> dict:
    """Out-of-fold predictions of every readout (see the module docstring)."""
    vis, hid = rows["vis"], rows["hid"]
    fold = np.asarray([folds[r] for r in vis["root"]])
    hfold = np.asarray([folds[r] for r in hid["root"]])
    y, steps = vis["plate"], vis["step"]
    pred = {k: np.full((len(y), 2), np.nan) for k in ("r_plate", "r_floor", "r_pool", "clock")}
    hidden = np.full((len(hid["plate"]), 2), np.nan)
    selections = []
    for k in range(pt.OUTER_FOLDS):
        fit, held, hheld = fold != k, fold == k, hfold == k
        if not held.any():
            continue
        groups = vis["root"][fit]
        started = time.monotonic()
        r = ridge(feats["full"][fit], y[fit], groups)
        pred["r_plate"][held] = r.predict(feats["full"][held])
        if hheld.any():
            hidden[hheld] = r.predict(feats["hidden_full"][hheld])
        f = ridge(feats["floor_full"][fit], y[fit], groups)
        pred["r_floor"][held] = f.predict(feats["floor_full"][held])
        p = ridge(feats["pooled"][fit], y[fit], groups)
        pred["r_pool"][held] = p.predict(feats["pooled"][held])
        pred["clock"][held] = clock_prior(y[fit], steps[fit], steps[held])
        selections.append(
            {
                "fold": k,
                "r_plate": r.selection,
                "r_floor": f.selection,
                "r_pool": p.selection,
                "seconds": time.monotonic() - started,
            }
        )
        if log is not None:
            log(f"outer fold {k}: {int(fit.sum())} fit rows, {int(held.sum())} held rows")
    for name, values in pred.items():
        if not np.isfinite(values).all():
            raise lp.GuardError(f"G-finite: {name} has rows without an out-of-fold prediction")
    if not np.isfinite(hidden).all():
        raise lp.GuardError("G-finite: a plate-hidden row has no out-of-fold prediction")
    errors = {k: 100.0 * np.linalg.norm(v - y, axis=1) for k, v in pred.items()}
    errors["hidden"] = 100.0 * np.linalg.norm(hidden - hid["plate"], axis=1)
    signed = 100.0 * (pred["r_plate"] - y)
    return {
        "errors": errors,
        "signed": signed,
        "clusters": vis["root"],
        "hidden_clusters": hid["root"],
        "steps": steps,
        "selections": selections,
    }


def statistics(fit: dict, tau_re_cm: float, tau_counts: dict, *, resamples=None) -> dict:
    """O1, O3, O4 and every reported number of §6.1."""
    e, roots = fit["errors"], fit["clusters"]
    r_plate = pt.cluster_median_ci(e["r_plate"], roots, resamples=resamples)
    clock = pt.cluster_median_ci(e["clock"], roots, resamples=resamples)
    hidden = pt.cluster_median_ci(e["hidden"], fit["hidden_clusters"], resamples=resamples)
    pool = pt.cluster_median_ci(e["r_pool"], roots, resamples=resamples)
    floor = pt.cluster_median_ci(e["r_floor"], roots, resamples=resamples)
    ratio = pt.cluster_median_ratio(e["r_plate"], e["clock"], roots, resamples=resamples)
    floor_ratio = pt.cluster_median_ratio(e["r_plate"], e["r_floor"], roots, resamples=resamples)
    mapped = pt.tau_curve_fraction(e["r_plate"], tau_counts) * pt.TAU["resets"]
    per_step = {
        str(int(t)): float(np.median(e["r_plate"][fit["steps"] == t]))
        for t in np.unique(fit["steps"])
    }
    gates = {
        "o1_upper": r_plate["ci95"][1],
        "o3_ratio_upper": ratio["ci95"][1],
        "o4_lower": hidden["ci95"][0],
    }
    return {
        "gates": gates,
        "decision": pt.decide_offline(gates, tau_re_cm),
        "median_cm": {
            "r_plate": r_plate,
            "clock": clock,
            "plate_hidden": hidden,
            "r_plate_pool": pool,
            "r_plate_floor": floor,
        },
        "ratio_clock": ratio,
        "reported": {
            "p87_5": pt.cluster_quantile_ci(
                e["r_plate"], roots, pt.REPORTED_PERCENTILE, resamples=resamples
            ),
            "tau_curve_predicted_successes_of_32": pt.cluster_mean_ci(
                mapped, roots, resamples=resamples
            ),
            "per_step_median_cm": per_step,
            "signed_mean_cm": {
                axis: pt.cluster_mean_ci(fit["signed"][:, i], roots, resamples=resamples)
                for i, axis in enumerate(("x", "y"))
            },
            "c_plate_cm": pool["ci95"][1],
            "r_plate_over_floor": floor_ratio,
        },
        "rows": int(len(roots)),
        "roots": int(len(np.unique(roots))),
        "hidden_rows": int(len(fit["hidden_clusters"])),
    }


def fit_closed_loop(rows: dict, feats: dict) -> dict:
    """R_plate and R_plate_floor on every read root's decision frames (the closed loop's)."""
    groups = rows["vis"]["root"]
    y = rows["vis"]["plate"]
    return {
        "r_plate": ridge(feats["full"], y, groups),
        "r_plate_floor": ridge(feats["floor_full"], y, groups),
    }


def save_readout(path, readout) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    np.savez(path, **readout.state())
    return readout.sha256()


def offline_core(
    report: dict,
    source,
    views,
    clock,
    *,
    tau_re_cm: float,
    tau_counts: dict,
    encoder,
    floor,
    device: str,
    probe: bool = False,
    log=None,
) -> dict:
    """Stage O, the stage's own function (the scale probe runs it with ``probe=True``)."""
    started = time.monotonic()
    rows = load_rows(source, views, clock)
    folds = outer_folds(rows["roots"])
    out = {
        "roots": {
            "read": len(rows["roots"]),
            "by_split": {
                s: sum(v == s for v in rows["splits"].values())
                for s in pt.SOURCE_CORPUS["read_splits"]
            },
            "rows": int(len(rows["vis"]["root"])),
            "hidden_rows": int(len(rows["hid"]["root"])),
            "fold_sizes": [sum(f == k for f in folds.values()) for k in range(pt.OUTER_FOLDS)],
            "seconds": time.monotonic() - started,
        }
    }
    if clock is not None:
        clock.check("featurisation")
    feats = featurise_rows(rows, encoder, floor, device=device)
    out["featurisation"] = {k: feats[k] for k in ("anchor", "seconds", "frames")}
    if feats["seconds"] > pt.CAPS_SECONDS["featurisation"]:
        raise lp.GuardError("G-cap: the featurisation exceeded its cap")
    if not probe:
        # Stage O's boundary (VOID_RULE): before the first fit.
        report["first_outcome_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.monotonic()
    fit = cross_fit(rows, feats, folds, log=log)
    out["cross_fit"] = {"selections": fit["selections"], "seconds": time.monotonic() - started}
    out["statistics"] = statistics(fit, tau_re_cm, tau_counts)
    started = time.monotonic()
    closed = fit_closed_loop(rows, feats)
    out["closed_loop"] = {
        name: {"sha256": r.sha256(), "selection": r.selection} for name, r in closed.items()
    } | {"seconds": time.monotonic() - started}
    return out | {"_fit": fit, "_closed": closed}
