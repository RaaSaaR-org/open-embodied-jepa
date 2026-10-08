"""TASK-082's offline parts (protocol ``docs/experiments/apple_lewm_unknown_law_v2.md`` §4.3, §5,
§6.2, §9.1, §9.4): the corpus ``apple-ul-v2``'s store and featurisation, Stage O (admission and
the train-only fits, the learned tier included) and Stage G's pieces (the readouts on predicted
latents, the dynamics gates, the offline aims against the logged ceiling labels).

TASK-077's and TASK-080's offline modules are imported, never edited: ``encode_tokens``,
``anchor_check``, ``load_root``, ``predict_at``, ``errors_cm``, ``collapse_statistics``,
``effective_rank_from_energy``, ``save_readout``, the feature-file layout and check, the roll-outs
and the readout errors are reused. Every ridge here is Stage O's dual ridge (the relative lambda
grid, 5 inner folds grouped by root) with this task's fold salt 8410, sub-key 2; every interval
uses salt 8411; the learned tier's CV uses salt 8410, sub-key 3 (the design note's
``plate_law_dev.krr_fit_fixed`` and ``krr_predict``, pinned, on this task's folds).

**G-split**: Stage O fits on train only (its admission is cross-fitted over train + val, TASK-077's
declared exception); Stage G's readouts fit on train + val only and refuse a gate-P root.

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
from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa import lewm_pr_v2_offline as pof
from embodied_jepa import lewm_ul_v2 as ul
from embodied_jepa import lewm_ul_v2_train as ult
from embodied_jepa import plate_law_dev as pl
from embodied_jepa.contracts import ContractError

GuardError = ul.GuardError
START_INDEX = off.START_INDEX  # frame 405
R_INDEX = off.R_INDEX  # frame 465
FULL_WIDTH = off.FULL_WIDTH
READOUT_NAMES = ("r_s", "r_n", "r_l")
sha256_file = off.sha256_file
errors_cm = off.errors_cm
save_readout = off.save_readout
feature_paths = off.feature_paths
verify_feature_files = off.verify_feature_files
load_table = off.load_table
load_root = off.load_root
predict_at = off.predict_at


def array_sha256(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def moments_sha256(mean, std) -> str:
    return off.moments_sha256(mean, std)


# ----- the corpus store (Stage C) ----------------------------------------------------------------
ROOT_KEYS = (*off.ROOT_KEYS, "p_hat405", "label_aim", "label_converged")


def write_root(folder, seed: int, arrays: dict) -> str:
    """One root's npz (TASK-077's layout plus p-hat at 405 and the ceiling label); refuses to
    overwrite; returns its sha256."""
    path = Path(folder) / f"{int(seed)}.npz"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    missing = [k for k in ROOT_KEYS if k not in arrays]
    if missing:
        raise ContractError(f"a corpus root needs {missing}")
    shapes = {
        "frames": (ul.N_FRAMES, 112, 112, 3),
        "commands": (ul.N_COMMANDS, 14),
        "plate": (ul.N_FRAMES, 2),
        "palm": (ul.N_FRAMES, 2),
        "hidden_r": (112, 112, 3),
        "p_hat405": (2,),
        "label_aim": (2,),
        "label_converged": (),
    }
    for key, shape in shapes.items():
        if tuple(np.shape(arrays[key])) != shape:
            raise ContractError(f"{key} must be {shape}, not {np.shape(arrays[key])}")
    if not np.isfinite(np.asarray(arrays["label_aim"], np.float64)).all():
        raise ContractError("a kept root carries its ceiling label")
    np.savez_compressed(path, **{k: np.asarray(arrays[k]) for k in ROOT_KEYS})
    return sha256_file(path)


def seal_corpus(folder, entries: dict, split: dict, provenance: dict) -> dict:
    """The sealed manifest: every kept root's sha256 and split, the aim construction per split,
    the provenance; refuses to overwrite."""
    folder = Path(folder)
    path = folder / "manifest.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    manifest = {
        "protocol": ul.PROTOCOL,
        "corpus": ul.CORPUS_NAME,
        "law": dict(ul.LAW),
        "roots": {str(s): entries[s] for s in sorted(entries)},
        "split": {k: [int(s) for s in v if int(s) in entries] for k, v in split.items()},
        "split_declared": {k: [int(s) for s in v] for k, v in split.items()},
        "aim_from": dict(ul.AIM_FROM),
        "provenance": provenance,
    }
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return {"path": str(path), "sha256": sha256_file(path)}


def open_corpus(folder, expected_sha256: str | None) -> dict:
    """The sealed manifest, checked against its recorded sha256 (G-split: every later stage)."""
    path = Path(folder) / "manifest.json"
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise GuardError("G-split: the corpus manifest differs from its sealed sha256")
    manifest = json.loads(path.read_text())
    if manifest.get("protocol") != ul.PROTOCOL or sorted(manifest["split"]) != sorted(ul.SPLITS):
        raise GuardError(f"G-split: not {ul.CORPUS_NAME}")
    if manifest.get("law") != json.loads(json.dumps(ul.LAW)):
        raise GuardError("G-law: the corpus was not collected under U-sat at the frozen parameters")
    return manifest | {"_sha256": digest}


# ----- featurisation (Stage O, GPU) ---------------------------------------------------------------
SIDE_KEYS = {  # table key -> root key
    "plate": "plate",
    "palm": "palm",
    "target": "target",
    "state405": "state405",
    "apple": "apple_estimate",
    "last_grasp": "last_grasp",
    "p_hat405": "p_hat405",
    "label_aim": "label_aim",
    "label_converged": "label_converged",
}


def featurise(corpus, manifest: dict, out, *, device: str, encoder=None, check=None) -> dict:
    """Every split's per-root 8 x 8 store and side arrays in TASK-077's featurisation layout
    (``features8_<split>.npy`` and the rest; the table adds p-hat at 405 and the label), with
    G-anchor on the first 256 kept frames. The stores are re-mapped every 32 roots (R17.31).
    Refuses to overwrite."""
    from numpy.lib.format import open_memmap

    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa.models.frozen_tokens import pool_tokens

    started = time.monotonic()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    encoder = pe.load_pretrained() if encoder is None else encoder
    encoder.to(device).eval()
    files, roots = {}, {}
    need = int(ul.FEATURE_ANCHOR["frames"])
    anchor_frames, anchor_pooled = [], []
    for split in ul.SPLITS:
        seeds = [int(s) for s in manifest["split"][split]]
        n = len(seeds)
        paths = feature_paths(out, split)
        for path in paths.values():
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
        feats = open_memmap(paths["features"], "w+", np.float32, (n, ul.N_FRAMES, ul.LATENT_DIM))
        full405 = open_memmap(paths["full405"], "w+", np.float32, (n, FULL_WIDTH))
        hidden8 = np.empty((n, ul.LATENT_DIM), np.float32)
        pool4 = np.empty((n, 2, 16 * 384), np.float32)
        commands = np.empty((n, ul.N_COMMANDS, 14), np.float32)
        side = {k: [] for k in SIDE_KEYS}
        for i, seed in enumerate(seeds):
            root = load_root(corpus, seed, manifest["roots"][str(seed)])
            stack = np.concatenate([root["frames"], root["hidden_r"][None]])
            tokens = off.encode_tokens(encoder, stack, device=device)
            pooled = pool_tokens(tokens, ul.TOKEN_GRID)
            feats[i] = pooled[: ul.N_FRAMES]
            hidden8[i] = pooled[ul.N_FRAMES]
            pool4[i] = pool_tokens(tokens[[START_INDEX, R_INDEX]], 4)
            full405[i] = tokens[START_INDEX]
            commands[i] = root["commands"]
            for key, src in SIDE_KEYS.items():
                side[key].append(root[src])
            take = min(need - sum(len(a) for a in anchor_frames), ul.N_FRAMES)
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
        roots[split] = n
        files[split] = {k: sha256_file(p) for k, p in paths.items()}
    gpu_seconds = time.monotonic() - started
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
        "featurisation_seconds": gpu_seconds,
    }


# ----- ridges and the learned tier ---------------------------------------------------------------
def ridge(x, y, groups):
    """Stage O's dual ridge with this task's inner-fold seed (salt 8410, sub-key 2)."""
    from embodied_jepa.models.latent_critic import RidgeReadout

    return RidgeReadout.fit(
        x,
        y,
        groups,
        lambdas=ul.LAMBDA_GRID_RELATIVE,
        folds=ul.INNER_FOLDS,
        seed=ul.fold_seed("inner"),
        dual=True,
    )


def krr_cv_fit(x, y) -> dict:
    """An RBF kernel ridge of ``y`` on standardised ``x``: length and lambda by 5-fold CV over the
    design note's grids (median error; folds salt 8410, sub-key 3), then refitted on all rows
    (``plate_law_dev.krr_fit_fixed``)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    fold = ul.learned_tier_folds(len(x))
    best = None
    for length in ul.KRR_LENGTHS:
        for lam in ul.KRR_LAMBDAS:
            err = np.empty(len(x))
            for k in range(ul.OUTER_FOLDS):
                fit, held = fold != k, fold == k
                m = pl.krr_fit_fixed(x[fit], y[fit], length, lam)
                err[held] = np.linalg.norm(pl.krr_predict(m, x[held]) - y[held], axis=1)
            score = float(np.median(err))
            if best is None or score < best[0]:
                best = (score, length, lam)
    model = pl.krr_fit_fixed(x, y, best[1], best[2])
    model["cv_median_m"] = best[0]
    return model


def krr_sysid_fit(p_hat, h, g, plate_r) -> dict:
    """H-sysid-krr: plate(r) - p-hat on standardised (p-hat, h, g)."""
    return krr_cv_fit(pl.krr_features(p_hat, h, g), np.asarray(plate_r) - np.asarray(p_hat))


def p_aim_fit(p_hat, h, label) -> dict:
    """P-aim: the ceiling's logged aim, as g - p-hat, on standardised (p-hat, h)."""
    x = np.concatenate([np.asarray(p_hat, np.float64), np.asarray(h, np.float64)], axis=1)
    return krr_cv_fit(x, np.asarray(label, np.float64) - np.asarray(p_hat, np.float64))


def kappa_fit(p_hat, h, g, plate_r) -> float:
    """H-rule-fit's gain: least squares of plate(r) - p-hat on g - h (one scalar)."""
    u = np.asarray(g, np.float64) - np.asarray(h, np.float64)
    v = np.asarray(plate_r, np.float64) - np.asarray(p_hat, np.float64)
    return float((u * v).sum() / (u * u).sum())


def model_json(model: dict) -> dict:
    return pl.model_to_json(model)


def json_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# ----- Stage O: admission and the train-only fits (CPU) ------------------------------------------
def _tv(feat, tables, stores, splits=ul.FIT_SPLITS) -> dict:
    return {
        "seeds": np.concatenate([tables[s]["seeds"] for s in splits]),
        "x_r": np.concatenate([np.asarray(stores[s].features[:, R_INDEX]) for s in splits]),
        "hidden": np.concatenate([np.load(Path(feat) / f"hidden8_r_{s}.npy") for s in splits]),
        "pool4_r": np.concatenate(
            [np.load(Path(feat) / f"pool4_405_r_{s}.npy")[:, 1] for s in splits]
        ),
        "y_r": np.concatenate([tables[s]["plate"][:, R_INDEX] for s in splits]),
        "y405": np.concatenate([tables[s]["plate"][:, START_INDEX] for s in splits]),
        "h405": np.concatenate([tables[s]["palm"][:, START_INDEX] for s in splits]),
        "g": np.concatenate([tables[s]["target"] for s in splits]),
        "p_hat": np.concatenate([tables[s]["p_hat405"] for s in splits]),
        "label": np.concatenate([tables[s]["label_aim"] for s in splits]),
        "split": np.concatenate([np.full(len(tables[s]["seeds"]), s) for s in splits]),
    }


def readouts_core(feat, out, *, tau_commit_cm, r_plate, check=None) -> dict:
    """Stage O's CPU part (§9.1): O1, O3, O4 and the learning curve cross-fitted over train + val
    (5 outer folds, salt 8410 sub-key 1); the reported readings (R-plate's error at 405 on this
    corpus, the design note's §4.2 table at full size, P-aim's aim error); then the train-only
    fits every later stage reads. ``r_plate``: the carried R-plate (only its reported reading of
    the stored tokens uses it; every fit reads the logged CPU p-hat)."""
    started = time.monotonic()
    feat, out = Path(feat), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tables = {s: load_table(feat, s) for s in ul.FIT_SPLITS}  # gate-P: Stage G only
    stores = {s: ult.RootStore.open(feat, s, mmap=True) for s in ul.FIT_SPLITS}
    tv = _tv(feat, tables, stores)
    n = len(tv["seeds"])
    roots = tv["seeds"].astype(str)
    fold = ul.outer_folds(n)
    yr = tv["y_r"]
    pv, ph, p4, prior = (np.full((n, 2), np.nan) for _ in range(4))
    lin, krr, p_aim = (np.full((n, 2), np.nan) for _ in range(3))
    curve = {f: np.full((n, 2), np.nan) for f in ul.FRACTIONS}
    sizes = {f: [] for f in ul.FRACTIONS}
    selections = []
    for k in range(ul.OUTER_FOLDS):
        fit, held = np.flatnonzero(fold != k), fold == k
        subsets = ul.nested_subsets(fit, k)
        if not np.array_equal(subsets[1.0], fit):
            raise GuardError("G-curve: the full-data subset is not the fold's fit rows")
        for f in ul.FRACTIONS:
            rows = subsets[f]
            model = ridge(tv["x_r"][rows], yr[rows], roots[rows])
            curve[f][held] = model.predict(tv["x_r"][held])
            sizes[f].append(int(len(rows)))
            if f == 1.0:
                pv[held] = curve[f][held]
                ph[held] = model.predict(tv["hidden"][held])
                selections.append({"what": "R8", "fold": k, "selection": model.selection})
        m4 = ridge(tv["pool4_r"][fit], yr[fit], roots[fit])
        p4[held] = m4.predict(tv["pool4_r"][held])
        prior[held] = yr[fit].mean(axis=0)
        coef = c1.sysid_fit(tv["p_hat"][fit], tv["h405"][fit], tv["g"][fit], yr[fit])
        lin[held] = c1.sysid_predict(coef, tv["p_hat"][held], tv["h405"][held], tv["g"][held])
        km = krr_sysid_fit(tv["p_hat"][fit], tv["h405"][fit], tv["g"][fit], yr[fit])
        krr[held] = pl.krr_plate(km, tv["p_hat"][held], tv["h405"][held], tv["g"][held])
        pm = p_aim_fit(tv["p_hat"][fit], tv["h405"][fit], tv["label"][fit])
        x_held = np.concatenate([tv["p_hat"][held], tv["h405"][held]], axis=1)
        p_aim[held] = tv["p_hat"][held] + pl.krr_predict(pm, x_held)
        if check is not None:
            check()
    for name, values in {"visible": pv, "hidden": ph, "pool4": p4, "prior": prior, "linear": lin,
                         "krr": krr, "p_aim": p_aim,
                         **{str(f): v for f, v in curve.items()}}.items():  # fmt: skip
        if not np.isfinite(values).all():
            raise GuardError(f"G-finite: {name} lacks an out-of-fold prediction")
    ev, eh, ep = errors_cm(pv, yr), errors_cm(ph, yr), errors_cm(prior, yr)
    e_curve = {f: errors_cm(curve[f], yr) for f in ul.FRACTIONS}
    guard = ul.median_difference_ci(e_curve[0.75], e_curve[1.0])
    c_plate = ul.median_ci(ev)
    prior_ratio = ul.median_ratio_ci(ev, ep)
    hidden = ul.median_ci(eh)
    falling = bool(guard["ci95"][0] > 0.0)
    kappa = ul.H_RULE_LAW["kappa"]
    open_form = tv["p_hat"] + kappa * (tv["g"] - tv["h405"])
    stored = np.concatenate([
        r_plate.predict(np.asarray(part[lo : lo + 128], np.float64))
        for part in (np.load(feat / f"full405_{s}.npy", mmap_mode="r") for s in ul.FIT_SPLITS)
        for lo in range(0, len(part), 128)
    ]).reshape(-1, 2)  # fmt: skip
    admission = {
        "c_plate": c_plate,
        "prior_ratio": prior_ratio,
        "prior": ul.median_ci(ep),
        "hidden": hidden,
        "learning_curve": {
            "fit_rows_per_fold": {str(f): sizes[f] for f in ul.FRACTIONS},
            "median_cm": {str(f): float(np.median(e_curve[f])) for f in ul.FRACTIONS},
            "p87_5_cm": {str(f): float(np.percentile(e_curve[f], 87.5)) for f in ul.FRACTIONS},
            "guard_3_4_minus_all": guard,
            "still_falling": falling,
        },
        "reported": {
            "pool4_at_r": ul.median_ci(errors_cm(p4, yr)),
            "r_plate_405_logged_cpu": ul.median_ci(errors_cm(tv["p_hat"], tv["y405"])),
            "r_plate_405_stored_tokens": ul.median_ci(errors_cm(stored, tv["y405"])),
            "p_hat_logged_vs_stored_max_cm": float(errors_cm(stored, tv["p_hat"]).max()),
            "sysid_linear_at_r": ul.median_ci(errors_cm(lin, yr)),
            "sysid_krr_at_r": ul.median_ci(errors_cm(krr, yr)),
            "c1m_rule_open_form_at_r": ul.median_ci(errors_cm(open_form, yr)),
            "p_aim_vs_label": ul.median_ci(errors_cm(p_aim, tv["label"])),
            "plate_motion_405_to_r_cm": ul.median_ci(errors_cm(tv["y405"], yr)),
            "note": "out-of-fold over train + val, the outer folds of O1-O4; the learned tier's "
            "length and lambda chosen inside each fit (salt 8410, sub-key 3)",
        },
    }
    decision = (
        ul.decide_o(
            c_plate=c_plate,
            prior_ratio=prior_ratio,
            hidden=hidden,
            tau_commit_cm=tau_commit_cm,
            falling=falling,
        )  # fmt: skip
        if not ul.is_missing(tau_commit_cm)
        else ul.NOT_EVALUATED
    )
    fits = downstream_fits(feat, out, tables["train"], stores["train"], check=check)
    return {
        "rows": n,
        "folds": fold.tolist(),
        "tau_commit_cm": tau_commit_cm,
        "admission": admission,
        "decision": decision,
        "selections": selections,
        "fits": fits,
        "seconds": time.monotonic() - started,
    }


def downstream_fits(feat, out, table, train_store, *, check=None) -> dict:
    """The train-only fits every later stage reads (G-split): R8 (8 x 8 at r -> plate at r), the
    normalisation moments over every kept train frame, L-mean's mean 405 latent, H-sysid's linear
    coefficients, H-rule-fit's kappa, H-sysid-krr and P-aim (each on the logged CPU p-hat)."""
    seeds = table["seeds"].astype(str)
    y405, yr = table["plate"][:, START_INDEX], table["plate"][:, R_INDEX]
    h405, g = table["palm"][:, START_INDEX], table["target"]
    p_hat, label = table["p_hat405"], table["label_aim"]
    x_r = np.asarray(train_store.features[:, R_INDEX])
    r8 = ridge(x_r, yr, seeds)
    coef = c1.sysid_fit(p_hat, h405, g, yr)
    kappa = kappa_fit(p_hat, h405, g, yr)
    krr = model_json(krr_sysid_fit(p_hat, h405, g, yr))
    p_aim = model_json(p_aim_fit(p_hat, h405, label))
    if check is not None:
        check()
    mean, std = ult.moments_file(Path(feat) / "features8_train.npy")
    mean_latent = np.asarray(train_store.features[:, START_INDEX], np.float64).mean(0)
    out = Path(out)
    paths = {
        "r8": out / "r8.npz",
        "sysid": out / "sysid.json",
        "rule_fit": out / "rule_fit.json",
        "krr": out / "krr.json",
        "p_aim": out / "p_aim.json",
        "moments": out / "moments.npz",
        "mean_latent": out / "mean_latent.npy",
    }
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    record = {"r8": {"path": str(paths["r8"]), "sha256": save_readout(paths["r8"], r8)}}
    for name, obj in (("sysid", {"coef": coef.tolist()}), ("rule_fit", {"kappa": kappa}),
                      ("krr", krr), ("p_aim", p_aim)):  # fmt: skip
        paths[name].write_text(json.dumps(obj, sort_keys=True))
        record[name] = {"path": str(paths[name]), "sha256": json_sha256(obj)}
    np.savez(paths["moments"], mean=mean, std=std)
    np.save(paths["mean_latent"], mean_latent.astype(np.float32))
    latent32 = np.ascontiguousarray(mean_latent.astype(np.float32))
    record |= {
        "moments": {"path": str(paths["moments"]), "sha256": moments_sha256(mean, std)},
        "mean_latent": {"path": str(paths["mean_latent"]), "sha256": array_sha256(latent32)},
        "sysid_coef": coef.tolist(),
        "kappa_fit": kappa,
        "krr_selection": {"length": krr["length"], "lambda": krr["lambda"],
                          "cv_median_cm": 100.0 * krr["cv_median_m"]},
        "p_aim_selection": {"length": p_aim["length"], "lambda": p_aim["lambda"],
                            "cv_median_cm": 100.0 * p_aim["cv_median_m"]},
        "selections": {"r8": r8.selection},
        "train_roots": int(len(seeds)),
        "train_reading_error_405": ul.median_ci(errors_cm(p_hat, y405)),
    }  # fmt: skip
    return record


def load_json_fit(record: dict) -> dict:
    """A JSON fit (sysid, rule_fit, krr, p_aim), checked against its recorded sha256."""
    obj = json.loads(Path(record["path"]).read_text())
    if json_sha256(obj) != record["sha256"]:
        raise GuardError(f"G-readout: {Path(record['path']).name} differs from its sha256")
    return obj


# ----- Stage G: roots, roll-outs and readouts on predicted latents --------------------------------
def load_roots(features, splits, *, limit: int | None = None, full405: bool = False) -> dict:
    """TASK-080's ``load_roots`` plus the ceiling label and its convergence flag."""
    data = pof.load_roots(features, splits, limit=limit, full405=full405)
    labels, converged = [], []
    for split in splits:
        table = load_table(features, split)
        n = len(table["seeds"]) if limit is None else min(int(limit), len(table["seeds"]))
        labels.append(table["label_aim"][:n])
        converged.append(table["label_converged"][:n].astype(bool))
    data["label"] = np.concatenate(labels)
    data["label_converged"] = np.concatenate(converged)
    return data


def check_fit_roots(seeds, split: dict) -> None:
    """G-split: a Stage G readout is fitted on train + val roots only, never on gate-P."""
    gate = {int(s) for s in split["gate_p"]}
    allowed = {int(s) for k in ul.FIT_SPLITS for s in split[k]}
    bad = [int(s) for s in seeds if int(s) in gate or int(s) not in allowed]
    if bad:
        raise GuardError(f"G-split: a readout fit reads roots outside train + val: {bad[:5]}")


rollouts = pof.rollouts  # W^s (stand-in), N^s (zero), W^s from the mean latent (stand-in)
readout_errors = pof.readout_errors  # e_S, e_N, e_L in cm


def fit_readouts(pred: dict, plate_r, seeds, split: dict) -> dict:
    """R-S, R-N and R-L of one model seed on the fit roots (train + val only)."""
    check_fit_roots(seeds, split)
    groups = np.asarray(seeds).astype(str)
    y = np.asarray(plate_r, np.float64)
    return {
        name: ridge(pred[key], y, groups) for name, key in zip(READOUT_NAMES, "SNL", strict=True)
    }


def save_readouts(folder, seed: int, readouts: dict) -> dict:
    return pof.save_readouts(folder, seed, readouts)


def load_readout(path, sha256: str):
    return pof.load_readout(path, sha256)


def readout_statistics(e: dict) -> dict:
    """R1-R3's statistics (salt 8411)."""
    return {
        "e_S": ul.median_ci(e["e_S"]),
        "e_N": ul.median_ci(e["e_N"]),
        "e_L": ul.median_ci(e["e_L"]),
        "S_over_N": ul.median_ratio_ci(e["e_S"], e["e_N"]),
    }


def crossfit(pred: dict, plate_r, seeds, split_of, split: dict) -> dict:
    """Reported only: each readout cross-fitted over the fit roots in 5 outer folds (salt 8410,
    sub-key 1), its held-out median error by split."""
    check_fit_roots(seeds, split)
    n = len(seeds)
    fold = ul.outer_folds(n)
    groups = np.asarray(seeds).astype(str)
    y = np.asarray(plate_r, np.float64)
    out = {}
    for name, key in zip(READOUT_NAMES, "SNL", strict=True):
        held = np.full((n, 2), np.nan)
        for k in range(ul.OUTER_FOLDS):
            fit, rows = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            held[rows] = ridge(pred[key][fit], y[fit], groups[fit]).predict(pred[key][rows])
        if not np.isfinite(held).all():
            raise GuardError("G-finite: a cross-fitted readout left a root without a prediction")
        e = errors_cm(held, y)
        out[name] = {str(s): ul.median_ci(e[np.asarray(split_of) == s])
                     for s in np.unique(split_of)}  # fmt: skip
        out[name]["all"] = ul.median_ci(e)
    return {"fold_sizes": np.bincount(fold).tolist(), "readouts": out}


def learning_curve(pred_fit: dict, plate_fit, seeds_fit, pred_eval: dict, plate_eval,
                   split: dict) -> dict:  # fmt: skip
    """Reported only: each readout fitted on nested 1/4, 1/2, 3/4 and all of the fit roots (salt
    8410, sub-key 4), its median error on gate-P."""
    check_fit_roots(seeds_fit, split)
    subsets = ul.nested_subsets(np.arange(len(seeds_fit)), 0)
    groups = np.asarray(seeds_fit).astype(str)
    y = np.asarray(plate_fit, np.float64)
    out = {}
    for name, key in zip(READOUT_NAMES, "SNL", strict=True):
        out[name] = {}
        for f, rows in subsets.items():
            model = ridge(pred_fit[key][rows], y[rows], groups[rows])
            e = errors_cm(model.predict(pred_eval[key]), plate_eval)
            out[name][str(f)] = {"fit_roots": int(len(rows)), "median_cm": float(np.median(e))}
    return out


# ----- Stage G: the dynamics gates on gate-P (TASK-077 §8.2's definitions) ------------------------
def comparative_rank(w, n_, e, *, resamples: int = ul.COMPARATIVE_RESAMPLES) -> dict:
    """G1 (iii): (rank ratio W - rank ratio N), both against the encoded rank of the same resample
    of gate-P roots (``lewm_c1m_v2_offline.comparative_rank``'s statistic, salt 8411)."""
    grams = {k: np.asarray(x, np.float64) @ np.asarray(x, np.float64).T for k, x in
             (("W", w), ("N", n_), ("E", e))}  # fmt: skip
    m = len(e)
    rng = np.random.default_rng(ul.SALTS["bootstrap"])

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
    lo, hi = np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan)
    return {
        "rank_W": point["W"],
        "rank_N": point["N"],
        "rank_encoded": point["E"],
        "difference": point["W"] / point["E"] - point["N"] / point["E"],
        "ci95": [float(lo), float(hi)],
        "resamples": int(resamples),
        "undefined_resamples": int(undefined),
    }


def dynamics(w_model, n_model, store, scale, *, bars=None, horizons=None) -> dict:
    """G1-G4 of one seed on E60 (one window per gate-P root from its encoded 405 frame under its
    executed commands) at h = 60, and every gate at the reported horizons (16, 30); copy-last's
    and N's errors beside them. G1's bars are carried (TASK-077's calibration)."""
    bars = ul.G1_BARS if bars is None else bars
    h60 = ul.HORIZON
    horizons = (*ul.REPORTED_HORIZONS, h60) if horizons is None else tuple(horizons)
    start, commands, targets = store.from_405()
    start = np.asarray(start)
    commands = np.asarray(commands, np.float32)
    perm = ul.wrong_permutation(len(store))
    zero = np.zeros_like(commands)
    pred = {
        "W_true": predict_at(w_model, start, commands, horizons),
        "W_wrong": predict_at(w_model, start, commands[perm], horizons),
        "W_zero": predict_at(w_model, start, zero, horizons),
        "N": predict_at(n_model, start, zero, horizons),
    }
    scale = np.asarray(scale, np.float64)

    def err(p, h):
        d = (np.asarray(p, np.float64) - np.asarray(targets[:, h - 1], np.float64)) / scale
        return (d * d).mean(1)

    per_h = {}
    for h in horizons:
        e = {k: err(v[h], h) for k, v in pred.items()}
        e["copy"] = err(start, h)
        per_h[h] = {
            "G2": ul.ratio_of_sums_ci(e["W_true"], e["copy"]),
            "G3": ul.ratio_of_sums_ci(e["W_true"], e["N"]),
            "G4_wrong": ul.ratio_of_sums_ci(e["W_wrong"], e["W_true"]),
            "G4_zero": ul.ratio_of_sums_ci(e["W_zero"], e["W_true"]),
            "mse": {k: float(v.mean()) for k, v in e.items()},
        }
    encoded = np.asarray(targets[:, h60 - 1], np.float64) / scale
    w_norm = np.asarray(pred["W_true"][h60], np.float64) / scale
    n_norm = np.asarray(pred["N"][h60], np.float64) / scale
    stats = off.collapse_statistics(w_norm, encoded)
    comparative = comparative_rank(w_norm, n_norm, encoded)
    g1 = lm.g1_passes(stats, bars, comparative)
    g = per_h[h60] | {"G1": {"stats": stats, "comparative": comparative, "parts": g1}}
    passes = ul.dynamics_passes(g)
    reported = {}
    for h in ul.REPORTED_HORIZONS:
        if h in per_h:
            reported[str(h)] = {
                k: per_h[h][k] for k in ("G2", "G3", "G4_wrong", "G4_zero", "mse")
            } | {"passes_g2_g4_form": {
                "G2": bool(per_h[h]["G2"]["ci95"][1] <= ul.G_THRESHOLDS["G2_max_ratio_upper"]),
                "G3": bool(per_h[h]["G3"]["ci95"][1]
                           < ul.G_THRESHOLDS["G3_max_ratio_upper_exclusive"]),
                "G4": bool(per_h[h]["G4_wrong"]["ci95"][0] >= ul.G_THRESHOLDS["G4_min_ratio_lower"]
                           and per_h[h]["G4_zero"]["ci95"][0]
                           >= ul.G_THRESHOLDS["G4_min_ratio_lower"]),
            }}  # fmt: skip
    return {
        "passes": passes,
        "G1": g["G1"],
        "at_60": {k: per_h[h60][k] for k in ("G2", "G3", "G4_wrong", "G4_zero", "mse")},
        "reported_horizons": reported,
        "bars": bars,
        "bars_label": "carried (TASK-077's calibration on C1-M's val roots)",
    }


# ----- Stage G: the offline aims (§9.4) -----------------------------------------------------------
def offline_aim_tasks(arm, data, latents, tau_commit_cm, *, fits: dict, keep=False):
    """One task per gate-P root for ``arm``: its logged 405 state, its logged p-hat and the palm at
    405; W, N and the spread arms start from the root's own latent, L-shuf from the next root's
    (cyclically), L-mean from the mean latent. ``fits``: sysid_coef, kappa_fit, krr, p_aim."""
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
            "p_hat": np.asarray(data["p_hat405"][i]).tolist(),
            "h": data["palm405"][i].tolist(),
            "tau_commit_cm": float(tau_commit_cm),
        }
        if arm in ("W", "N") or ul.spread_seed(arm) is not None:
            task["start"] = np.asarray(latents[i], np.float32)
        elif arm == "L-shuf":
            task["start"] = np.asarray(latents[(i + 1) % n], np.float32)
        elif arm == "L-mean":
            task["start"] = "mean"
        if arm == "H-sysid":
            task["sysid_coef"] = fits["sysid_coef"]
        if arm == "H-rule-fit":
            task["kappa_fit"] = float(fits["kappa_fit"])
        if arm == "H-sysid-krr":
            task["krr"] = fits["krr"]
        if arm == "P-aim":
            task["p_aim"] = fits["p_aim"]
        if keep and arm == "W":
            task["keep_plates"] = True
        tasks.append(task)
    return tasks


def aim_summary(results: list[dict], data, tau_counts: dict, resets: int) -> dict:
    """An arm's offline aims: the aim error against the root's logged ceiling label, its interval
    (salt 8411), 87.5th percentile and maximum, the clip-binding fraction and the predicted count
    of 128 through K0's tau curve (a prediction, never a closed-loop count)."""
    got = sorted(results, key=lambda r: r["key"])
    if [r["key"] for r in got] != list(range(data["n"])):
        raise GuardError("G-aims: an offline aim is missing")
    aims = np.asarray([r["g"] for r in got], np.float64)
    errors = 100.0 * np.linalg.norm(aims - np.asarray(data["label"], np.float64), axis=1)
    stats = ul.median_ci(errors) | {"max": float(errors.max())}
    return {
        "aim_error_vs_label_cm": stats,
        "predicted_count_of_128": ul.predicted_count(errors, tau_counts, resets),
        "clip_binding_fraction": float(np.mean([bool(r["log"].get("clipped")) for r in got])),
        "fallbacks": int(sum(bool(r["log"].get("fallback_all_infeasible")) for r in got)),
        "aim_errors_cm": errors.round(6).tolist(),
    }


def echo_slopes(results: list[dict], data) -> dict:
    """Reported (§9.4): per root, the 2 x 2 least-squares slope A of W's predicted plate against
    the aim over the feasible grid (p~ = A g + c), beside U-sat's local Jacobian of the plate's
    displacement at the root's ceiling label, evaluated at the palm travel d = label - h (an
    approximation, stated: it takes the palm to end at the aim, while the law reads the palm at
    t - L along the place primitive's path)."""
    slopes, laws = [], []
    by_key = {r["key"]: r for r in results}
    for i in range(data["n"]):
        r = by_key.get(i)
        if r is None:
            continue
        g = np.asarray(r.get("grid_targets", []), np.float64)
        p = np.asarray(r.get("grid_plates", []), np.float64)
        if len(g) < 3:
            continue
        x = np.hstack([g, np.ones((len(g), 1))])
        coef, *_ = np.linalg.lstsq(x, p, rcond=None)
        slopes.append(coef[:2].T)
        d = np.asarray(data["label"][i], np.float64) - np.asarray(data["palm405"][i], np.float64)
        laws.append(ul.law_jacobian(d))
    if not slopes:
        return {"roots": 0}
    a, j = np.stack(slopes), np.stack(laws)
    return {
        "roots": int(len(a)),
        "median_matrix": np.median(a, axis=0).tolist(),
        "law_jacobian_median": np.median(j, axis=0).tolist(),
        "frobenius_to_law_jacobian": ul.median_ci(np.linalg.norm(a - j, axis=(1, 2))),
        "frobenius_to_I": ul.median_ci(np.linalg.norm(a - np.eye(2), axis=(1, 2))),
    }
