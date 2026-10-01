"""TASK-075 offline pieces: per-view featurisation, the cross-fitted readouts and the statistics.

Protocol ``docs/experiments/apple_obs_ceiling_v2.md`` §4-§6. Stored frames only; nothing here
simulates. NumPy at import; torch and transformers are imported by the functions that need them.

* **Features.** Frozen DINOv2 ViT-S/14 (pretrained and the seed-0 random-init floor), final patch
  tokens, on CUDA in batches of 64: the pooled 4 x 4 latent (``pool_tokens``, 6144-d) and, for
  the Gram rows, the full tokens (98 304-d). A 112 px frame is resized to 224 by
  ``pretrained_encoder.preprocess`` (bicubic, as every earlier task); a 224 px frame goes in
  natively (the same scaling and ImageNet normalisation, no resize).
* **Readouts** (``oc.READOUTS``): R_off and R_floor (``RidgeReadout``, dual form), the plate and
  apple targets, R_proprio (primal), R_full and R_pix (``info_ceiling``'s kernel ridge on a
  Gram matrix, with folds grouped by root), the cross-fitted clock prior.
* **Cross-fitting.** 5 outer folds by root over the read roots: every root's windows are read by
  readouts fitted on the other folds only.
"""

from __future__ import annotations

import numpy as np

from embodied_jepa import info_ceiling as ic
from embodied_jepa import lewm_planner_v2 as lp
from embodied_jepa import obs_ceiling_v2 as oc
from embodied_jepa.contracts import ContractError

FEATURE_BATCH = 64
GRAM_BLOCK = 2048
FULL_TOKEN_PARTS = 2


# ----- features ---------------------------------------------------------------------------------
def preprocess_view(frames):
    """uint8 [N, s, s, 3] (s = 112 or 224) -> float32 [N, 3, 224, 224], ImageNet-normalised."""
    import torch

    from embodied_jepa import pretrained_encoder as pe

    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[3] != 3:
        raise ContractError(f"expected uint8 [N, s, s, 3] frames, got {frames.shape}")
    if frames.shape[1:3] == (pe.FRAME_SIZE, pe.FRAME_SIZE):
        return pe.preprocess(frames)
    if frames.shape[1:3] != (pe.INPUT_SIZE, pe.INPUT_SIZE):
        raise ContractError(f"a view frame must be 112 or 224 px, got {frames.shape[1:3]}")
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
    mean = x.new_tensor(pe.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(pe.IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def featurise(encoder, frames, *, device: str, columns=None, batch: int = FEATURE_BATCH):
    """Pooled 4 x 4 latents float32 [N, 6144] of every frame and, if ``columns`` (a slice of the
    98 304 full-token dimensions) is given, those columns of the full tokens, float32."""
    import torch

    from embodied_jepa.models.frozen_tokens import pool_tokens

    frames = np.asarray(frames)
    n = len(frames)
    pooled = np.empty((n, 6144), np.float32)
    full = None
    if columns is not None:
        width = len(range(256 * 384)[columns])
        full = np.empty((n, width), np.float32)
    encoder.to(device).eval()
    with torch.no_grad():
        for lo in range(0, n, batch):
            pixels = preprocess_view(frames[lo : lo + batch]).to(device)
            last = encoder(pixel_values=pixels).last_hidden_state[:, 1:]
            tokens = last.reshape(len(last), -1).float().cpu().numpy()
            pooled[lo : lo + len(tokens)] = pool_tokens(tokens.astype(np.float64), 4)
            if full is not None:
                full[lo : lo + len(tokens)] = tokens[:, columns]
            del tokens, last, pixels
    encoder.to("cpu")
    if not np.isfinite(pooled).all() or (full is not None and not np.isfinite(full).all()):
        raise lp.GuardError("G-finite: a feature is not finite")
    return pooled, full


def full_token_gram(encoder, frames, *, device: str, parts: int = FULL_TOKEN_PARTS):
    """Pooled latents and the float64 Gram of the full tokens, accumulated over ``parts``
    column slices of the token dimensions (memory: one slice of the full tokens is held at a
    time; the Gram is the same sum of products)."""
    width = 256 * 384
    step = -(-width // parts)
    g, pooled = None, None
    for lo in range(0, width, step):
        p, full = featurise(
            encoder, frames, device=device, columns=slice(lo, min(width, lo + step))
        )
        if pooled is None:
            pooled = p
        elif not np.array_equal(pooled, p):
            raise lp.GuardError("G-determinism: the pooled features differ between passes")
        part, _ = gram(full)
        del full
        g = part if g is None else g + part
        del part
    return pooled, g, np.diag(g).copy()


def anchor_check(
    encoder, frames, gpu_pooled, *, bound=lp.FEATURE_ANCHOR["max_abs_pooled_difference"]
):
    """G-anchor: CUDA pooled features of ``frames`` against the same code path on the CPU."""
    cpu, _ = featurise(encoder, frames, device="cpu", batch=8)
    diff = float(np.abs(cpu.astype(np.float64) - np.asarray(gpu_pooled, np.float64)).max())
    if not diff <= bound:
        raise lp.GuardError(f"G-anchor: CUDA and CPU pooled features differ by {diff} > {bound}")
    return {"frames": int(len(cpu)), "max_abs_difference": diff, "bound": bound}


def gram(x, *, block: int = GRAM_BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """Float64 ``x x^T`` of row-flattened features (any dtype), in column blocks; its diagonal."""
    x = np.asarray(x).reshape(len(x), -1)
    g = np.zeros((len(x), len(x)))
    for lo in range(0, x.shape[1], block):
        b = np.asarray(x[:, lo : lo + block], np.float64)
        g += b @ b.T
        del b
    return g, np.diag(g).copy()


# ----- readouts ---------------------------------------------------------------------------------
def ridge(x_fit, y_fit, groups, *, dual: bool):
    from embodied_jepa.models.latent_critic import RidgeReadout

    return RidgeReadout.fit(
        x_fit,
        y_fit,
        groups,
        lambdas=oc.READOUTS["lambda_grid_relative"],
        folds=oc.INNER_FOLDS,
        seed=oc.SEEDS["inner_folds"],
        dual=dual,
    )


def kernel_fit(g, diag, fit_rows, groups, y_all) -> tuple:
    """``info_ceiling``'s kernel ridge (family and lambda by inner CV) with the inner folds
    grouped by root. ``y_all`` is indexed by Gram row; only ``fit_rows`` are read."""
    from embodied_jepa.models.latent_critic import grouped_folds

    fit_rows = np.asarray(fit_rows)
    y_all = np.asarray(y_all, np.float64)
    inner = grouped_folds(groups, oc.INNER_FOLDS, oc.SEEDS["inner_folds"])
    sse = {(f, lam): 0.0 for f in oc.KERNEL_FAMILIES for lam in oc.KERNEL_LAMBDAS}
    for j in range(oc.INNER_FOLDS):
        a, h = fit_rows[inner != j], fit_rows[inner == j]
        if not len(a) or not len(h):
            raise ContractError("an inner fold is empty")
        stats = ic.train_stats(g, diag, a)
        y_a = y_all[a]
        y_mean = y_a.mean(axis=0)
        for family in oc.KERNEL_FAMILIES:
            k_aa = ic.kernel_rows(family, g[np.ix_(a, a)], diag[a], stats)
            k_ha = ic.kernel_rows(family, g[np.ix_(h, a)], diag[h], stats)
            scale = float(np.mean(np.diag(k_aa)))
            for lam in oc.KERNEL_LAMBDAS:
                alpha = np.linalg.solve(k_aa + lam * scale * np.eye(len(k_aa)), y_a - y_mean)
                sse[(family, lam)] += float(((k_ha @ alpha + y_mean - y_all[h]) ** 2).sum())
            del k_aa, k_ha
    best = min(sse, key=lambda k: (sse[k], oc.KERNEL_FAMILIES.index(k[0]), k[1]))
    readout = ic.fit_readout(best[0], best[1], g, diag, fit_rows, y_all)
    return readout, {"family": best[0], "lam_rel": best[1], "inner_mse": sse[best] / len(fit_rows)}


def kernel_predict(readout, g, diag, rows) -> np.ndarray:
    rows = np.asarray(rows)
    if not len(rows):
        return np.zeros((0, 2))
    return readout.predict(g[np.ix_(rows, readout.stats.index)], diag[rows])


# ----- the per-view data and the cross-fit -------------------------------------------------------
class ViewData:
    """One view's rows. ``roots``: dicts with ``id``, ``fold``, ``hi`` (last band step) and
    ``offset``/``plate``/``apple`` [L, 2] (m, float64) and ``states`` {step: vector}.

    ``vis`` maps (root index, step) -> visible row; ``hid[kind]`` maps (root index, step) ->
    hidden row. ``pooled``, ``floor`` are [rows, 6144]; ``pooled_hidden[kind]`` likewise.
    ``gram_index`` maps ('vis' | kind, root index, step) -> Gram row; ``g_full``/``g_pix`` are
    the Gram matrices (or None)."""

    def __init__(self, roots, vis, hid, pooled, floor, pooled_hidden, gram_index, grams):
        self.roots, self.vis, self.hid = roots, vis, hid
        self.pooled, self.floor, self.pooled_hidden = pooled, floor, pooled_hidden
        self.gram_index, self.grams = gram_index, grams


def windows(roots) -> list[tuple[int, int]]:
    """Every (root index, t) at the decision steps with t + 16 inside the root's band whose true
    offset moves by >= 1 cm over the 16 commands (O2's moving cohort)."""
    out = []
    for i, r in enumerate(roots):
        for t in oc.DECISION_STEPS:
            if t + oc.CHUNK > r["hi"]:
                continue
            move = np.linalg.norm(r["offset"][t + oc.CHUNK] - r["offset"][t])
            if move >= oc.MOVING_THRESHOLD_M:
                out.append((i, t))
    return out


def _rows(values) -> np.ndarray:
    return np.asarray(values, np.int64).reshape(-1)


def fit_steps(r) -> list[int]:
    return [s for s in oc.TRAIN_STEPS if s <= r["hi"]]


def cross_fit(data: ViewData, *, kernels: bool = True, log=None) -> dict:
    """Out-of-fold errors (cm) of every readout on every moving window of every root."""
    roots = data.roots
    wins = windows(roots)
    owner = np.asarray([i for i, _t in wins])
    folds = np.asarray([roots[i]["fold"] for i in owner])
    n = len(wins)
    truth = np.asarray([roots[i]["offset"][t + oc.CHUNK] for i, t in wins])
    truth_plate = np.asarray([roots[i]["plate"][t + oc.CHUNK] for i, t in wins])
    truth_apple = np.asarray([roots[i]["apple"][t + oc.CHUNK] for i, t in wins])
    pred = {
        k: np.full((n, 2), np.nan)
        for k in (
            "r_off",
            "r_floor",
            "persistence",
            "r_off_plate_hidden",
            "r_off_apple_hidden",
            "lc_0.25",
            "lc_0.5",
            "plate_target",
            "apple_target",
            "r_proprio",
            "clock",
            "o_star",
            "r_full",
            "r_full_plate_hidden",
            "r_full_apple_hidden",
            "r_pix",
            "r_pix_plate_hidden",
            "r_pix_apple_hidden",
        )
    }
    selections = []
    for k in range(oc.OUTER_FOLDS):
        held = np.flatnonzero(folds == k)
        fit_roots = sorted({i for i, r in enumerate(roots) if r["fold"] != k})
        eval_vis = _rows([data.vis[(owner[w], wins[w][1] + oc.CHUNK)] for w in held])
        start_vis = _rows([data.vis[(owner[w], wins[w][1])] for w in held])
        hid_rows = {
            kind: _rows([data.hid[kind][(owner[w], wins[w][1] + oc.CHUNK)] for w in held])
            for kind in oc.HIDDEN_KINDS
        }
        record = {"outer_fold": k, "fit_roots": len(fit_roots), "held_windows": int(len(held))}
        for fraction in oc.LEARNING_CURVE:
            roots_f = oc.learning_subsample(fit_roots, fraction, k)
            rows = [(i, s) for i in roots_f for s in fit_steps(roots[i])]
            vis_rows = _rows([data.vis[x] for x in rows])
            groups = [roots[i]["id"] for i, _s in rows]
            y = np.asarray([roots[i]["offset"][s] for i, s in rows])
            r_off = ridge(data.pooled[vis_rows], y, groups, dual=True)
            if fraction < 1.0:
                pred[f"lc_{fraction}"][held] = r_off.predict(data.pooled[eval_vis])
                record[f"lc_{fraction}"] = {"lam_rel": r_off.lam_rel, "rows": len(rows)}
                continue
            record["r_off"] = r_off.selection
            pred["r_off"][held] = r_off.predict(data.pooled[eval_vis])
            pred["persistence"][held] = r_off.predict(data.pooled[start_vis])
            for kind in oc.HIDDEN_KINDS:
                pred[f"r_off_{kind}"][held] = r_off.predict(
                    data.pooled_hidden[kind][hid_rows[kind]]
                )
            r_floor = ridge(data.floor[vis_rows], y, groups, dual=True)
            record["r_floor"] = r_floor.selection
            pred["r_floor"][held] = r_floor.predict(data.floor[eval_vis])
            for target, key in (("plate", "plate_target"), ("apple", "apple_target")):
                yt = np.asarray([roots[i][target][s] for i, s in rows])
                rt = ridge(data.pooled[vis_rows], yt, groups, dual=True)
                record[key] = {"lam_rel": rt.lam_rel}
                pred[key][held] = rt.predict(data.pooled[eval_vis])
            states = np.asarray([roots[i]["states"][s] for i, s in rows])
            r_prop = ridge(states, y, groups, dual=False)
            record["r_proprio"] = {"lam_rel": r_prop.lam_rel}
            pred["r_proprio"][held] = r_prop.predict(
                np.asarray([roots[owner[w]]["states"][wins[w][1] + oc.CHUNK] for w in held])
            )
            prior = {
                s: np.median(
                    np.asarray([roots[i]["offset"][s] for i in roots_f if s <= roots[i]["hi"]]),
                    axis=0,
                )
                for s in oc.EVAL_STEPS
            }
            pred["clock"][held] = np.asarray([prior[wins[w][1] + oc.CHUNK] for w in held])
            pred["o_star"][held] = (
                np.asarray([lp.O_STAR_CM[wins[w][1] + oc.CHUNK] for w in held]) / 100.0
            )
            if kernels:
                g_rows = _rows([data.gram_index[("vis", i, s)] for i, s in rows])
                g_eval = _rows(
                    [data.gram_index[("vis", owner[w], wins[w][1] + oc.CHUNK)] for w in held]
                )
                g_hid = {
                    kind: _rows(
                        [data.gram_index[(kind, owner[w], wins[w][1] + oc.CHUNK)] for w in held]
                    )
                    for kind in oc.HIDDEN_KINDS
                }
                for source in ("full", "pix"):
                    g, diag, y_all = data.grams[source]
                    readout, chosen = kernel_fit(g, diag, g_rows, groups, y_all)
                    record[f"r_{source}"] = chosen
                    pred[f"r_{source}"][held] = kernel_predict(readout, g, diag, g_eval)
                    for kind in oc.HIDDEN_KINDS:
                        pred[f"r_{source}_{kind}"][held] = kernel_predict(
                            readout, g, diag, g_hid[kind]
                        )
                    del readout
        selections.append(record)
        if log is not None:
            log(f"  outer fold {k}: {record['held_windows']} windows")
    errors = {}
    for key, p in pred.items():
        if key.startswith("r_full") or key.startswith("r_pix"):
            if not kernels:
                continue
        ref = (
            truth_plate
            if key == "plate_target"
            else truth_apple
            if key == "apple_target"
            else truth
        )
        if not np.isfinite(p).all():
            raise lp.GuardError(f"G-finite: a window received no out-of-fold {key} prediction")
        errors[key] = 100.0 * np.linalg.norm(p - ref, axis=1)
    clusters = np.asarray([roots[i]["id"] for i in owner])
    return {
        "errors": errors,
        "signed": {"r_off": 100.0 * (pred["r_off"] - truth)},
        "clusters": clusters,
        "shifted": np.asarray([bool(roots[i].get("shifted", False)) for i in owner]),
        "windows": [(roots[i]["id"], int(t)) for i, t in wins],
        "selections": selections,
    }


def view_statistics(fit: dict, *, kernels: bool = True, resamples=None) -> dict:
    """The admission and representation inputs and every reported statistic of one view."""
    e, c = fit["errors"], fit["clusters"]
    ci = {k: oc.cluster_median_ci(v, c, resamples=resamples) for k, v in e.items()}
    ratio = {
        "r_off/r_floor": oc.cluster_median_ratio(e["r_off"], e["r_floor"], c, resamples=resamples),
        "r_off/clock": oc.cluster_median_ratio(e["r_off"], e["clock"], c, resamples=resamples),
        "r_off/o_star": oc.cluster_median_ratio(e["r_off"], e["o_star"], c, resamples=resamples),
        "r_off/r_proprio": oc.cluster_median_ratio(
            e["r_off"], e["r_proprio"], c, resamples=resamples
        ),
    }
    difference = {
        "lc_0.5-r_off": oc.cluster_median_difference(
            e["lc_0.5"], e["r_off"], c, resamples=resamples
        ),
        "lc_0.25-lc_0.5": oc.cluster_median_difference(
            e["lc_0.25"], e["lc_0.5"], c, resamples=resamples
        ),
    }
    out = {
        "windows": int(len(c)),
        "roots": int(len(set(c.tolist()))),
        "median_ci": ci,
        "ratio_ci": ratio,
        "difference_ci": difference,
        "r_off": {
            "c_upper": ci["r_off"]["ci95"][1],
            "floor_ratio_upper": ratio["r_off/r_floor"]["ci95"][1],
            "clock_ratio_upper": ratio["r_off/clock"]["ci95"][1],
            "plate_hidden_lower": ci["r_off_plate_hidden"]["ci95"][0],
        },
        "learning_curve_still_falling": bool(difference["lc_0.5-r_off"]["ci95"][0] > 0),
    }
    # Reported only (oc.REPORTED): the 87.5th percentile, the tau-curve prediction, the signed
    # mean error along x and y.
    signed = np.asarray(fit.get("signed", {}).get("r_off", np.zeros((len(c), 2))), np.float64)
    predicted = oc.tau_curve_fraction(e["r_off"]) * oc.TAU["resets"] if oc.TAU_MEASURED else None
    out["reported"] = {
        "r_off_percentile": oc.cluster_quantile_ci(
            e["r_off"], c, oc.REPORTED_PERCENTILE, resamples=resamples
        ),
        "r_off_tau_curve_predicted_successes": (
            None if predicted is None else oc.cluster_mean_ci(predicted, c, resamples=resamples)
        ),
        "r_off_signed_mean_cm": {
            axis: oc.cluster_mean_ci(signed[:, j], c, resamples=resamples)
            for j, axis in enumerate(("x", "y"))
        },
    }
    # Reported only: the windows of roots whose plate moved at step 300 (tau's condition moves
    # it by 9 cm; the corpus by 3-12 cm) and of the unshifted roots.
    shifted = np.asarray(fit.get("shifted", np.zeros(len(c), bool)), bool)
    out["subsets"] = {}
    for name, mask in (("shifted", shifted), ("unshifted", ~shifted)):
        if mask.any():
            out["subsets"][name] = {
                k: oc.cluster_median_ci(e[k][mask], c[mask], resamples=resamples)
                for k in ("r_off", "r_floor", "clock")
            }
    if kernels:
        for source in ("full", "pix"):
            key = f"r_{source}"
            ratio[f"{key}/clock"] = oc.cluster_median_ratio(
                e[key], e["clock"], c, resamples=resamples
            )
            out[key] = {
                "c_upper": ci[key]["ci95"][1],
                "clock_ratio_upper": ratio[f"{key}/clock"]["ci95"][1],
                "plate_hidden_lower": ci[f"{key}_plate_hidden"]["ci95"][0],
            }
    return out
