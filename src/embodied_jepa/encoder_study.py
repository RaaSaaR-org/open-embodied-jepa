"""TASK-062 encoder study (``apple_encoder_study_v1``): cross-fitting, floors, Holm, decision.

The probe itself is TASK-061's L probe, executed by ``info_ceiling`` and ``observation_reprobe``
unchanged. What this module adds (protocol §4-§10):

* **Two-way cross-fitting.** Half A is TASK-061 folds 0-4, half B folds 5-9. Encoder ``e(h)``
  featurises every root of half ``h`` and is trained only on train-split episodes whose root is a
  train root of the *other* half. ``crossfit_nested_cv`` runs ``info_ceiling``'s nested CV with,
  for every outer fold, the Gram matrix of the encoder that excluded that fold's half.
* **The floor.** An arm must beat its seed-0 random init (``info_ceiling.beats_random_floor``);
  ``floor_pvalue`` is the share of the same paired resamples whose median error difference is >= 0.
* **Holm over four arms**, the three arm states, the pass rule, the collapse gate and the
  first-matching-row decision.
* **G-anchor** (§9): the TASK-061 anchors must reproduce bit for bit.
* **The study step** (§5.2) for A-sig and A-rec: the LeWM training step plus a SIGReg term on the
  image feature and/or a pixel-reconstruction term; with both at weight 0 it is
  ``LeWM.train_step`` exactly.

Torch is imported lazily, so ``import embodied_jepa`` stays torch-free.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from embodied_jepa import info_ceiling as ic
from embodied_jepa import observation_reprobe as orp
from embodied_jepa.contracts import ContractError

GuardError = ic.GuardError

HALVES = ("A", "B")
HALF_FOLDS = {"A": frozenset(range(0, 5)), "B": frozenset(range(5, 10))}
ARMS = ("A-tok", "A-sig", "A-rec", "A-plain")  # listed order = Holm tie-break order
LATENT_ARMS = ("A-sig", "A-rec", "A-plain")
ARM_FLOOR = {"A-tok": "F-tok", "A-sig": "F-cls", "A-rec": "F-cls", "A-plain": "F-plain"}
ARM_MODEL = {"A-tok": "R0", "A-sig": "SIG", "A-rec": "REC", "A-plain": "PLAIN"}
ARM_FEATURE = {"A-tok": "tokens", "A-sig": "probe", "A-rec": "probe", "A-plain": "probe"}
MODELS = ("R0", "SIG", "REC", "PLAIN")
MODEL_OVERRIDES = {"R0": {}, "SIG": {}, "REC": {}, "PLAIN": {"readout_heads": False}}
IMAGE_SIGREG_WEIGHT = {"SIG": 1.0}
RECONSTRUCTION_WEIGHT = {"REC": 1.0}
# Seed-0 initialisation digests (FrozenEncoder.weights_sha256's rule), calibration-v2.
SEED0_DIGEST = {
    True: "cae0ad8880bab6cdf05bc47e40b4944a17d71abe150bb212bae1a5ad2159437b",
    False: "f760fb425eea1616e90150d1789ca482dcf6df171c0820ea85c7a5232563e8e7",
}
ALPHA_ONE_SIDED = 0.025

STEPS = 15_000
BATCH_SIZE = 32
HORIZON = 16
FINAL_LR_FRACTION = 0.1
PER_ENCODER_SECONDS = 7200.0
GLOBAL_WALL_SECONDS = 64_800.0
MODEL_SEED = 0
SAMPLER_SEED = {"A": 6200, "B": 6201}
DECODER_SEED = 6210
COLLAPSE_SEED = 6220
COLLAPSE_FRAMES = 1024
SENSITIVITY_SEED = 6230
SENSITIVITY_WINDOWS = 512
SENSITIVITY_HORIZON = 8
COLLAPSE_GATE = {"max_collapsed_fraction": 0.05, "min_effective_rank": 2.0, "min_std_mean": 0.1}
ANCHOR_TOLERANCE = 1e-6
ANCHOR_SOURCES = ("L_raw", "L_E0", "L_random")


# ----- cross-fitting (§4) ---------------------------------------------------------------------
def half_of_fold(fold) -> np.ndarray:
    """``"A"``/``"B"`` per root from TASK-061's fold assignment."""
    fold = np.asarray(fold)
    if fold.min() < 0 or fold.max() >= ic.FOLDS:
        raise ContractError("fold out of range")
    return np.where(fold < 5, "A", "B")


def other(half: str) -> str:
    if half not in HALVES:
        raise ContractError(f"unknown half {half!r}")
    return "B" if half == "A" else "A"


def training_roots(roots, fold, half: str) -> list[str]:
    """Episode ids of the TRAIN roots that encoder ``e(half)`` may train on: the other half's."""
    halves = half_of_fold(fold)
    source = other(half)
    return sorted(
        r["episode_id"]
        for r, h in zip(roots, halves, strict=True)
        if h == source and r["split"] == "train"
    )


def training_episodes(manifest: dict, root_ids) -> list[str]:
    """Train-split episodes whose ``root_episode_id`` is one of ``root_ids`` (root + branches)."""
    wanted = frozenset(root_ids)
    rows = {row["episode_id"]: row for row in manifest["episodes"]}
    chosen = [
        name
        for name in manifest["splits"]["train"]
        if rows[name]["metadata"]["root_episode_id"] in wanted
    ]
    return chosen


def check_training_split(manifest: dict, roots, fold, episodes_by_half: dict) -> dict:
    """G-split additions (§11): train-split only, right half only, halves disjoint and complete."""
    rows = {row["episode_id"]: row for row in manifest["episodes"]}
    splits = manifest["splits"]
    train = frozenset(splits["train"])
    forbidden = frozenset(splits.get("val", [])) | frozenset(splits.get("test", []))
    forbidden |= frozenset(splits.get("holdout", []))
    halves = half_of_fold(fold)
    half_of_root = {r["episode_id"]: h for r, h in zip(roots, halves, strict=True)}
    train_roots = {r["episode_id"] for r in roots if r["split"] == "train"}
    used_roots = {}
    for half, episodes in episodes_by_half.items():
        if not episodes:
            raise GuardError(f"G-split: encoder e({half}) has no training episodes")
        if len(set(episodes)) != len(episodes):
            raise GuardError(f"G-split: duplicate training episodes for e({half})")
        roots_used = set()
        for name in episodes:
            if name in forbidden or name not in train:
                raise GuardError(f"G-split: {name} is not a train-split episode")
            root = rows[name]["metadata"]["root_episode_id"]
            if root not in train_roots:
                raise GuardError(f"G-split: {name}'s root {root} is not a train root")
            if half_of_root[root] == half:
                raise GuardError(f"G-split: e({half}) would train on {root}, a root it featurises")
            roots_used.add(root)
        used_roots[half] = roots_used
    if set(episodes_by_half) != set(HALVES):
        raise GuardError("G-split: need exactly the two encoders e(A) and e(B)")
    if used_roots["A"] & used_roots["B"]:
        raise GuardError("G-split: the two encoders' training roots overlap")
    if used_roots["A"] | used_roots["B"] != train_roots:
        raise GuardError("G-split: the training roots are not exactly the train roots")
    for half in HALVES:
        expected = set(training_episodes(manifest, used_roots[half]))
        if set(episodes_by_half[half]) != expected:
            raise GuardError(f"G-split: e({half}) does not use every episode of its roots")
    return {
        half: {
            "roots": len(used_roots[half]),
            "episodes": len(episodes_by_half[half]),
            "episode_ids_sha256": hashlib.sha256(
                json.dumps(sorted(episodes_by_half[half])).encode()
            ).hexdigest(),
        }
        for half in HALVES
    }


def crossfit_nested_cv(grams: dict, y, fold):
    """``info_ceiling.nested_cv`` with, for outer fold k, the Gram of the encoder that excluded
    k's half. ``grams``: half -> (g, diag) over all roots, each from that half's encoder."""
    y = np.asarray(y, np.float64)
    squeeze = y.ndim == 1
    y2 = y[:, None] if squeeze else y
    halves = half_of_fold(fold)
    predictions = np.full(y2.shape, np.nan)
    readouts, selections = [], []
    for k in range(int(fold.max()) + 1):
        half = str(halves[np.flatnonzero(fold == k)[0]])
        g, diag = grams[half]
        train_index, held = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        chosen = ic.select(g, diag, train_index, y2, k)
        readout = ic.fit_readout(chosen["family"], chosen["lam_rel"], g, diag, train_index, y2)
        predictions[held] = readout.predict(g[np.ix_(held, train_index)], diag[held])
        readouts.append(readout)
        selections.append({"outer_fold": k, "encoder_half": half} | chosen)
    if not np.isfinite(predictions).all():
        raise ContractError("a root received no out-of-fold prediction")
    return (predictions[:, 0] if squeeze else predictions), readouts, selections


def crossfit_split(grams: dict, y, fold, train_mask):
    """Secondary estimate: val roots of half h, read out from the 170 train roots with e(h)."""
    y = np.asarray(y, np.float64)
    squeeze = y.ndim == 1
    y2 = y[:, None] if squeeze else y
    halves = half_of_fold(fold)
    train_mask = np.asarray(train_mask, bool)
    out = np.full(y2.shape, np.nan)
    selections = {}
    for half in HALVES:
        g, diag = grams[half]
        held = np.flatnonzero(~train_mask & (halves == half))
        if not len(held):
            continue
        train_index = np.flatnonzero(train_mask)
        chosen = ic.select(g, diag, train_index, y2, outer=ic.FOLDS)
        readout = ic.fit_readout(chosen["family"], chosen["lam_rel"], g, diag, train_index, y2)
        out[held] = readout.predict(g[np.ix_(held, train_index)], diag[held])
        selections[half] = chosen
    return (out[:, 0] if squeeze else out), selections


def evaluate_crossfit_source(grams, xy, dx, dy, fold, priors, masks, train_mask, split_priors):
    """``probe_info_ceiling.evaluate_source`` for cross-fitted features (same outputs)."""
    xy_pred, xy_fits, xy_sel = crossfit_nested_cv(grams, xy, fold)
    dx_pred, dx_fits, dx_sel = crossfit_nested_cv(grams, dx, fold)
    dy_pred, _dy_fits, dy_sel = crossfit_nested_cv(grams, dy, fold)
    result = {"selections": {"xy": xy_sel, "dx": dx_sel, "dy": dy_sel}}
    kept = {}
    for stratum, mask in masks.items():
        evaluation = ic.evaluate(xy_pred, dx_pred, xy, dx, priors, mask)
        kept[stratum] = evaluation
        result[stratum] = ic.public(evaluation)
    sx, s_sel_xy = crossfit_split(grams, xy, fold, train_mask)
    sd, s_sel_dx = crossfit_split(grams, dx, fold, train_mask)
    val = ~np.asarray(train_mask, bool)
    full_xy, full_dx = np.zeros_like(np.asarray(xy, np.float64)), np.zeros(len(dx))
    full_xy[val], full_dx[val] = sx[val], sd[val]
    result["secondary_train_to_val"] = ic.public(
        ic.evaluate(full_xy, full_dx, xy, dx, split_priors, val)
    ) | {"selections": {"xy": s_sel_xy, "dx": s_sel_dx}}
    predictions = {
        "xy": xy_pred,
        "dx": dx_pred,
        "dy": dy_pred,
        "xy_fits": xy_fits,
        "dx_fits": dx_fits,
    }
    return result, kept, predictions


def crossfit_predict_hidden(fits, fold, hidden_grams: dict) -> np.ndarray:
    """Spurious check: root i's hidden features (from its own held-out encoder) through the
    readout that held it out. ``hidden_grams``: half -> (hidden @ that encoder's features.T,
    hidden norms), over all roots."""
    halves = half_of_fold(fold)
    rows = []
    for i in range(len(fold)):
        fit = fits[int(fold[i])]
        g_new, norms = hidden_grams[str(halves[i])]
        rows.append(fit.predict(g_new[i : i + 1, fit.stats.index], norms[i : i + 1])[0])
    return np.asarray(rows)


# ----- floor, p-values, Holm, pass rule (§6-§7) -----------------------------------------------
def floor_pvalue(arm_errors, floor_errors, idx) -> float:
    """p_F: share of paired resamples whose median error difference (arm - floor) is >= 0."""
    a, b = np.asarray(arm_errors, np.float64), np.asarray(floor_errors, np.float64)
    diff = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    return float((diff >= 0).mean())


def arm_pvalues(xy_pred, dx_pred, xy, dx, priors, floor_kept: dict, arm_kept: dict) -> dict:
    """p_arm = max(p_T1, p_T2, p_T3, p_F), all roots; 1 if a point condition fails."""
    everyone = np.ones(len(xy), bool)
    base = orp.hypothesis_pvalues(xy_pred, dx_pred, xy, dx, priors, everyone)
    idx = ic.bootstrap_indices(len(xy))
    p_f = floor_pvalue(arm_kept["_errors_cm"], floor_kept["_errors_cm"], idx)
    t2_higher = bool(arm_kept["T2"]["accuracy"] > floor_kept["T2"]["accuracy"])
    point_ok = bool(base["point_conditions_hold"] and t2_higher)
    p = max(base["p_T1"], base["p_T2"], base["p_T3"], p_f) if point_ok else 1.0
    return base | {
        "p_F": p_f,
        "T2_higher_than_floor": t2_higher,
        "point_conditions_hold": point_ok,
        "p": p,
    }


def holm(pvalues: dict) -> dict:
    return orp.holm(pvalues, alpha=ALPHA_ONE_SIDED, order=ARMS)


def collapse_ok(stats: dict) -> bool:
    return bool(
        stats["collapsed_fraction"] <= COLLAPSE_GATE["max_collapsed_fraction"]
        and stats["effective_rank"] >= COLLAPSE_GATE["min_effective_rank"]
        and stats["latent_std_mean"] >= COLLAPSE_GATE["min_std_mean"]
    )


def check_init(model, *, readout_heads: bool) -> str:
    """G-init: the model about to be trained (or read as a floor) is its seed-0 init."""
    digest = weights_digest(model)
    if digest != SEED0_DIGEST[bool(readout_heads)]:
        raise GuardError(f"G-init: initial weights digest {digest[:12]} is not the seed-0 init")
    return digest


def arm_state(*, trained: bool, not_collapsed: bool) -> str:
    """§7: 'not evaluated' (mechanical failure), 'collapse-demoted' or 'evaluated'."""
    if not trained:
        return "not evaluated"
    return "evaluated" if not_collapsed else "collapse-demoted"


def passes(
    *,
    trained: bool,
    not_collapsed: bool,
    succeeds: bool,
    beats_floor: bool,
    rejected: bool,
    spurious: bool,
) -> bool:
    return bool(
        trained and not_collapsed and succeeds and beats_floor and rejected and not spurious
    )


def qualifier(*, passes: bool, succeeds: bool, beats_floor: bool, rejected: bool, beats_prior):
    if passes:
        return "pass"
    if succeeds and beats_floor and not rejected:
        return "unadjusted only; not a pass"
    if succeeds and not beats_floor:
        return "meets the bars, not the floor"
    if succeeds:
        return "meets the bars, floor and Holm, but collapse-demoted or spurious; not a pass"
    if beats_prior:
        return "partial information"
    return None


ROWS = ("V", "O-ENC-LATENT", "O-ENC-TOKENS", "O-ENC-INCOMPLETE", "O-ENC-ARCH", "O-ENC-NONE")


def decide(*, void: bool, arms: dict) -> dict:
    """First matching row (§10). ``arms``: arm -> {"passes", "state", "succeeds", "spurious",
    "p"}."""
    missing = set(ARMS) - set(arms)
    if missing:
        raise ContractError(f"decision needs every arm; missing {missing}")
    if void:
        return {"outcome": "V"}
    passed = {a: bool(arms[a]["passes"]) for a in ARMS}
    latent = [a for a in LATENT_ARMS if passed[a]]
    if latent:
        chosen = min(latent, key=lambda a: (float(arms[a]["p"]), ARMS.index(a)))
        return {"outcome": "O-ENC-LATENT", "encoder_recipe": chosen, "passed": passed}
    if passed["A-tok"]:
        return {"outcome": "O-ENC-TOKENS", "encoder_recipe": "A-tok", "passed": passed}
    unevaluated = [a for a in ARMS if arms[a]["state"] == "not evaluated"]
    if unevaluated:
        return {"outcome": "O-ENC-INCOMPLETE", "not_evaluated": unevaluated, "passed": passed}
    exposed = [a for a in ARMS if arms[a]["succeeds"] and not arms[a]["spurious"]]
    if exposed:
        return {"outcome": "O-ENC-ARCH", "succeeding_arms": exposed, "passed": passed}
    return {"outcome": "O-ENC-NONE", "passed": passed}


def abandonment_fires(outcome: str) -> bool:
    return outcome in ("O-ENC-ARCH", "O-ENC-NONE")


# ----- G-anchor (§9) --------------------------------------------------------------------------
def _walk(ref, new, path, out):
    """Collect (path, kind, ref, new) differences; kind is 'numeric' or 'discrete'."""
    if isinstance(ref, dict) and isinstance(new, dict):
        for key in sorted(set(ref) | set(new)):
            if key not in ref or key not in new:
                out.append((f"{path}.{key}", "discrete", ref.get(key), new.get(key)))
            else:
                _walk(ref[key], new[key], f"{path}.{key}", out)
        return
    if isinstance(ref, list | tuple) and isinstance(new, list | tuple):
        if len(ref) != len(new):
            out.append((path, "discrete", len(ref), len(new)))
            return
        for i, (a, b) in enumerate(zip(ref, new, strict=True)):
            _walk(a, b, f"{path}[{i}]", out)
        return
    if isinstance(ref, bool) or isinstance(new, bool) or ref is None or new is None:
        if ref != new:
            out.append((path, "discrete", ref, new))
        return
    if isinstance(ref, int) and isinstance(new, int):
        if ref != new:
            out.append((path, "discrete", ref, new))
        return
    if isinstance(ref, int | float) and isinstance(new, int | float):
        if float(ref) != float(new):
            out.append((path, "numeric", float(ref), float(new)))
        return
    if ref != new:
        out.append((path, "discrete", ref, new))


def _jsonable(value):
    return json.loads(json.dumps(value, default=lambda o: np.asarray(o).tolist()))


# Float fields whose change is discrete by nature (a selected grid value). Integers (counts,
# folds) and strings (families) are compared as discrete values already.
_DISCRETE_NUMERIC = (".lam_rel",)


def anchor_compare(reference: dict, recomputed: dict) -> dict:
    """G-anchor verdict. ``reference``/``recomputed``: source -> {"results": the fields that
    ``evaluate_source`` + masks produce, "per_root": [{xy, dx, dy}, ...]}."""
    diffs = []
    for source in ANCHOR_SOURCES:
        if source not in reference or source not in recomputed:
            raise GuardError(f"G-anchor: missing anchor source {source}")
        _walk(_jsonable(reference[source]), _jsonable(recomputed[source]), source, diffs)
    numeric = [d for d in diffs if d[1] == "numeric"]
    discrete = [d for d in diffs if d[1] == "discrete"]
    selection_like = [d for d in numeric if any(key in d[0] for key in _DISCRETE_NUMERIC)]
    max_delta = max((abs(d[2] - d[3]) for d in numeric), default=0.0)
    if discrete or selection_like or max_delta > ANCHOR_TOLERANCE:
        verdict = "void"
    elif numeric:
        verdict = "caveat"
    else:
        verdict = "exact"
    return {
        "verdict": verdict,
        "max_abs_delta": max_delta,
        "numeric_differences": len(numeric),
        "discrete_differences": len(discrete) + len(selection_like),
        "examples": [list(map(str, d)) for d in (discrete + selection_like + numeric)[:20]],
    }


def check_anchor(comparison: dict) -> None:
    if comparison["verdict"] == "void":
        raise GuardError(
            "G-anchor: TASK-061's anchors did not reproduce "
            f"(max |delta| {comparison['max_abs_delta']}, "
            f"{comparison['discrete_differences']} discrete differences)"
        )


def anchor_view(results: dict, per_root: list, source: str) -> dict:
    """The compared part of a TASK-061-style report for one source."""
    keep = {k: v for k, v in results[source].items() if k in _ANCHOR_RESULT_KEYS}
    return {
        "results": keep,
        "per_root": [
            {k: row["predictions"][source][k] for k in ("xy", "dx", "dy")} for row in per_root
        ],
    }


_ANCHOR_RESULT_KEYS = (
    "all",
    "reset_occluded",
    "reset_visible",
    "secondary_train_to_val",
    "selections",
)


# ----- state normalisation over a training half (§4) -----------------------------------------
def welford_moments(states, masks) -> tuple[np.ndarray, np.ndarray]:
    """``DatasetStore.fit_normalization``'s per-field Welford mean and std, row by row."""
    states, masks = np.asarray(states), np.asarray(masks, bool)
    count = np.zeros(states.shape[1], dtype=np.int64)
    mean = np.zeros(states.shape[1], np.float64)
    m2 = np.zeros(states.shape[1], np.float64)
    for values, mask in zip(states, masks, strict=True):
        count[mask] += 1
        delta = values[mask] - mean[mask]
        mean[mask] += delta / count[mask]
        m2[mask] += delta * (values[mask] - mean[mask])
    std = np.sqrt(m2 / np.maximum(count, 1))
    std[std < 1e-6] = 1
    return mean, std


# ----- A-rec: the decoder and the training step (§5.2) ----------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_decoder(latent_dim: int = 128, image_size: int = 112, seed: int = DECODER_SEED):
    """Linear -> 7x7x256 -> four ConvTranspose2d(k4, s2, p1) -> 3 x 112 x 112, GELU between.

    Built under ``torch.manual_seed(seed)`` inside a forked RNG, so no other stream moves."""
    import torch
    from torch import nn

    if image_size != 112:
        raise ContractError("the A-rec decoder is defined for 112 px")

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(latent_dim, 256 * 7 * 7)
            self.body = nn.Sequential(
                nn.GELU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.GELU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.GELU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.GELU(),
                nn.ConvTranspose2d(32, 3, 4, 2, 1),
            )

        def forward(self, z):
            return self.body(self.project(z).reshape(-1, 256, 7, 7))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Decoder()


def normalized_pixels(model, images):
    """The ImageNet-normalised pixels LeWM's ``embed`` sees, for a flat batch of frames."""
    pixels, _ = model.pixels(images, camera=model.camera_names[0])
    mean = pixels.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = pixels.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (pixels - mean) / std


def study_train_step(
    model,
    batch,
    readout_targets,
    *,
    image_sigreg_weight: float = 0.0,
    decoder=None,
    decoder_optimizer=None,
    reconstruction_weight: float = 0.0,
):
    """``LeWM.train_step`` plus two optional terms on the image feature (before state fusion):

    * ``image_sigreg_weight`` x the model's own SIGReg over the window's image features (A-sig);
    * ``reconstruction_weight`` x the pixel MSE of frame 0 of every window, reconstructed from its
      image feature by ``decoder`` (A-rec).

    The LeWM part is ``LeWM.train_step`` line for line, with ``observe_sequence`` inlined so the
    image features are available, and ``VisualModel.optimize`` inlined so a decoder shares the
    gradient clip. With both weights 0 the update is exactly ``LeWM.train_step``'s: no extra
    random draw is made, the decoder (if any) gets no gradient and is not stepped."""
    if reconstruction_weight > 0 and (decoder is None or decoder_optimizer is None):
        raise ContractError("a reconstruction term needs a decoder and its optimizer")
    model.check_batch(batch)
    if batch.batch_size < 2:
        raise ContractError("LeWM BatchNorm requires at least two sequences during training")
    model.train()
    if decoder is not None:
        decoder.train()
    with model.rng_scope():
        actions = model.sequence_tensors(batch)
        features, prefix = model.image_features(batch.observations, sequence=True)
        image_latents = features.reshape(*prefix, -1)
        embeddings = image_latents
        if model.config["state_fusion"]:
            embeddings = embeddings + model.state_features(batch.robot_states, batch.state_mask)
        one_step = model.next_embedding(embeddings[:, :-1], actions)
        prediction_loss = (one_step - embeddings[:, 1:]).square().mean()
        recursive = model.rollout(embeddings[:, 0], actions)
        multistep = model.multistep_loss(recursive, embeddings[:, 1:])
        sigreg = model.sigreg(embeddings.transpose(0, 1))
        loss = (
            prediction_loss
            + model.config["multistep_weight"] * multistep
            + model.config["sigreg_weight"] * sigreg
        )
        readout_loss, readout_metrics = model.readout_loss(embeddings, recursive, readout_targets)
        loss = loss + readout_loss
        image_sigreg = loss.new_zeros(())
        if image_sigreg_weight > 0:
            image_sigreg = model.sigreg(image_latents.transpose(0, 1))
            loss = loss + image_sigreg_weight * image_sigreg
        reconstruction = loss.new_zeros(())
        if reconstruction_weight > 0:
            first = {k: v[:, 0] for k, v in batch.observations.items()}
            target = normalized_pixels(model, first)
            reconstruction = (decoder(image_latents[:, 0]) - target).square().mean()
            loss = loss + reconstruction_weight * reconstruction
        grad_norm = _optimize(
            model, decoder, decoder_optimizer, loss, embeddings, reconstruction_weight > 0
        )
    model.eval()
    if decoder is not None:
        decoder.eval()
    return {
        "loss": loss.item(),
        "prediction_loss": prediction_loss.item(),
        "multistep_loss": multistep.item(),
        "sigreg_loss": sigreg.item(),
        "image_sigreg_loss": image_sigreg.item(),
        "reconstruction_loss": reconstruction.item(),
        "gradient_norm": grad_norm,
        "updates": float(model.updates),
    } | readout_metrics


def _optimize(model, decoder, decoder_optimizer, loss, embeddings, with_decoder: bool):
    """``VisualModel.optimize`` with the decoder's parameters inside the same clip."""
    import torch

    if not torch.isfinite(loss):
        raise ContractError("non-finite training loss")
    model.optimizer.zero_grad(set_to_none=True)
    if decoder_optimizer is not None:
        decoder_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = [p for p in model.parameters() if p.grad is not None]
    if with_decoder:
        parameters += [p for p in decoder.parameters() if p.grad is not None]
    if not parameters or not all(torch.isfinite(p.grad).all() for p in parameters):
        raise ContractError("missing/non-finite model gradients")
    grad = torch.nn.utils.clip_grad_norm_(
        parameters, model.config["gradient_clip"], error_if_nonfinite=True
    )
    model.optimizer.step()
    if with_decoder:
        decoder_optimizer.step()
    with torch.no_grad():
        variance = embeddings.detach().reshape(-1, embeddings.shape[-1]).var(0, unbiased=False)
        model.latent_variance.lerp_(variance, 0.05)
    model.updates += 1
    model._owner = object()
    return grad.item()


# ----- diagnostics (§8) -----------------------------------------------------------------------
def derangement(n: int, rng) -> np.ndarray:
    """A permutation with no fixed point (seeded)."""
    if n < 2:
        raise ContractError("a derangement needs at least two items")
    while True:
        perm = rng.permutation(n)
        if not (perm == np.arange(n)).any():
            return perm


def weights_digest(module) -> str:
    """sha256 over a module's state dict in key order (``FrozenEncoder.weights_sha256``'s rule)."""
    digest = hashlib.sha256()
    state = module.state_dict()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def summary_json(value: Any):
    """Drop per-root arrays (keys starting with ``_``) recursively."""
    if isinstance(value, dict):
        return {k: summary_json(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, list):
        return [summary_json(v) for v in value]
    return value
