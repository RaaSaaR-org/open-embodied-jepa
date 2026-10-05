"""TASK-077's offline stages: the corpus store, Stage O (featurisation and admission readouts) and
Stage G (the 8 x 8 dynamics gates at h = 60). Protocol ``docs/experiments/apple_lewm_c1m_v2.md``
§4.3, §7 steps 5 and 7, §8.1-§8.2.

* **The corpus store** (Stage C): one compressed npz per root (its 65 onboard frames, 64 executed
  commands, the plate and palm at every kept step, the plate-hidden render at r, the 405 state)
  and a sealed manifest with every root's sha256 and split.
* **Featurisation** (Stage O, CUDA through ``scripts/gpu_run.sh``): every kept frame through the
  frozen DINOv2, pooled to 8 x 8 and written per split as contiguous per-root arrays
  ``[roots, 65, 24 576]`` (``lewm_c1m_v2_train.RootStore``); the 4 x 4 pool at 405 and r
  (reported), the full tokens at 405 (R-plate), the 8 x 8 pool of the plate-hidden render at r;
  a CPU anchor check (G-anchor).
* **Admission and the downstream fits** (Stage O, CPU): O1, O3, O4 and the learning curve,
  cross-fitted over train + val roots (5 outer folds, salt 8104; the declared G-split exception),
  then the train-only fits every later stage reads: R-plate, R8, H-sysid, the normalisation
  moments and L-mean's mean latent.
* **Stage G** (CPU, one torch thread): E60, one window per gate root, every gate on every seed.

NumPy at import; torch is imported inside the functions that need it.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from embodied_jepa import lewm_c1m_v2 as lm
from embodied_jepa import lewm_c1m_v2_train as tr
from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa.contracts import ContractError

GuardError = lm.GuardError
FULL_WIDTH = 256 * 384
FEATURE_BATCH = 66  # one root's 65 frames and its hidden render
FEATURE_REMAP_ROOTS = 32  # roots between re-mappings of the stores (R17.31: bounds their PSS)
R_INDEX = lm.READ_STEP - lm.FRAME_STEPS[0]  # 62
START_INDEX = tr.START_INDEX  # 2


def sha256_file(path) -> str:
    return tr.sha256_file(path)


# ----- the corpus store (Stage C) ----------------------------------------------------------------
ROOT_KEYS = (
    "frames",
    "commands",
    "plate",
    "palm",
    "hidden_r",
    "target",
    "state405",
    "apple_estimate",
    "last_grasp",
)


def write_root(folder, seed: int, arrays: dict) -> str:
    """One root's npz (compressed); refuses to overwrite; returns its sha256."""
    path = Path(folder) / f"{int(seed)}.npz"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    missing = [k for k in ROOT_KEYS if k not in arrays]
    if missing:
        raise ContractError(f"a corpus root needs {missing}")
    shapes = {
        "frames": (lm.N_FRAMES, 112, 112, 3),
        "commands": (lm.N_COMMANDS, 14),
        "plate": (lm.N_FRAMES, 2),
        "palm": (lm.N_FRAMES, 2),
        "hidden_r": (112, 112, 3),
    }
    for key, shape in shapes.items():
        if tuple(np.shape(arrays[key])) != shape:
            raise ContractError(f"{key} must be {shape}, not {np.shape(arrays[key])}")
    np.savez_compressed(path, **{k: np.asarray(arrays[k]) for k in ROOT_KEYS})
    return sha256_file(path)


def seal_corpus(folder, entries: dict, split: dict, provenance: dict) -> dict:
    """The sealed manifest: every kept root's sha256 and split, the exclusions, the provenance."""
    folder = Path(folder)
    path = folder / "manifest.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    kept = {str(s): entries[s] for s in sorted(entries)}
    manifest = {
        "protocol": lm.PROTOCOL,
        "roots": kept,
        "split": {
            name: [int(s) for s in seeds if int(s) in entries] for name, seeds in split.items()
        },
        "split_declared": {name: [int(s) for s in seeds] for name, seeds in split.items()},
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
    return json.loads(path.read_text()) | {"_sha256": digest}


def load_root(folder, seed: int, expected_sha256: str | None = None) -> dict:
    path = Path(folder) / f"{int(seed)}.npz"
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise GuardError(f"G-hash: corpus root {seed} differs from its sealed sha256")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


# ----- featurisation (Stage O, GPU) ------------------------------------------------------------
def encode_tokens(encoder, frames, *, device: str, batch: int = FEATURE_BATCH) -> np.ndarray:
    """Full DINOv2 tokens float32 [N, 98 304] of uint8 frames (``pretrained_encoder``'s input
    handling, the closed loop's read-out point)."""
    import torch

    from embodied_jepa import pretrained_encoder as pe

    frames = np.asarray(frames)
    out = np.empty((len(frames), FULL_WIDTH), np.float32)
    with torch.no_grad():
        for lo in range(0, len(frames), batch):
            pixels = pe.preprocess(frames[lo : lo + batch]).to(device)
            last = encoder(pixel_values=pixels).last_hidden_state[:, 1:]
            out[lo : lo + len(last)] = last.reshape(len(last), -1).float().cpu().numpy()
    if not np.isfinite(out).all():
        raise GuardError("G-finite: a feature is not finite")
    return out


def anchor_check(encoder, frames, device_pooled, *, bound=None) -> dict:
    """G-anchor: the device's 8 x 8 pooled features of ``frames`` against the CPU path."""
    from embodied_jepa.models.frozen_tokens import pool_tokens

    bound = lm.FEATURE_ANCHOR["max_abs_pooled_difference"] if bound is None else bound
    encoder.to("cpu")
    cpu = pool_tokens(encode_tokens(encoder, frames, device="cpu", batch=8), lm.TOKEN_GRID)
    diff = float(np.abs(cpu.astype(np.float64) - np.asarray(device_pooled, np.float64)).max())
    if not diff <= bound:
        raise GuardError(f"G-anchor: device and CPU pooled features differ by {diff} > {bound}")
    return {"frames": int(len(cpu)), "max_abs_difference": diff, "bound": bound}


def featurise_corpus(
    corpus,
    manifest: dict,
    out,
    *,
    device: str,
    encoder=None,
    check=None,
    probe: bool = False,
    remap_every: int | None = FEATURE_REMAP_ROOTS,
) -> dict:
    """Every split's per-root 8 x 8 store plus the side arrays (refuses to overwrite).

    The two memory-mapped stores are re-mapped every ``remap_every`` roots (R17.31, Erratum
    2026-10-05): the written pages of a mapping count in the process's PSS until it is unmapped,
    so one mapping held over the train split grows to its whole 9.6 GB file. Re-mapping writes the
    same bytes to the same offsets (the header is written once, at creation). ``None`` keeps one
    mapping per split, as at ``862d63c`` (development probes only)."""
    from numpy.lib.format import open_memmap

    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa.models.frozen_tokens import pool_tokens

    started = time.monotonic()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    encoder = pe.load_pretrained() if encoder is None else encoder
    encoder.to(device).eval()
    files, anchor = {}, None
    # G-anchor's frames (R17.39): the first FEATURE_ANCHOR["frames"] (256) kept frames, in split
    # and root order, as the frozen block declares (32 before, as TASK-075/076 checked)
    anchor_need = int(lm.FEATURE_ANCHOR["frames"])
    anchor_frames, anchor_pooled = [], []
    table = {}
    for split in ("train", "val", "gate"):
        seeds = [int(s) for s in manifest["split"][split]]
        n = len(seeds)
        paths = feature_paths(out, split)
        for path in paths.values():
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}")
        feats = open_memmap(paths["features"], "w+", np.float32, (n, lm.N_FRAMES, lm.LATENT_DIM))
        hidden8 = np.empty((n, lm.LATENT_DIM), np.float32)
        pool4 = np.empty((n, 2, 16 * 384), np.float32)
        full405 = open_memmap(paths["full405"], "w+", np.float32, (n, FULL_WIDTH))
        commands = np.empty((n, lm.N_COMMANDS, 14), np.float32)
        side = {k: [] for k in ("plate", "palm", "target", "state405", "apple", "last_grasp")}
        for i, seed in enumerate(seeds):
            root = load_root(corpus, seed, manifest["roots"][str(seed)])
            stack = np.concatenate([root["frames"], root["hidden_r"][None]])
            tokens = encode_tokens(encoder, stack, device=device)
            pooled = pool_tokens(tokens, lm.TOKEN_GRID)
            feats[i] = pooled[: lm.N_FRAMES]
            hidden8[i] = pooled[lm.N_FRAMES]
            pool4[i] = pool_tokens(tokens[[START_INDEX, R_INDEX]], 4)
            full405[i] = tokens[START_INDEX]
            commands[i] = root["commands"]
            side["plate"].append(root["plate"])
            side["palm"].append(root["palm"])
            side["target"].append(root["target"])
            side["state405"].append(root["state405"])
            side["apple"].append(root["apple_estimate"])
            side["last_grasp"].append(root["last_grasp"])
            take = min(anchor_need - sum(len(a) for a in anchor_frames), lm.N_FRAMES)
            if take > 0:
                anchor_frames.append(root["frames"][:take].copy())
                anchor_pooled.append(pooled[:take].copy())
            if check is not None and i % 50 == 0:
                check()
            del tokens, pooled, stack
            if remap_every is not None and (i + 1) % remap_every == 0 and i + 1 < n:
                feats = _remap(feats, paths["features"])
                full405 = _remap(full405, paths["full405"])
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
        table[split] = n
        files[split] = {k: sha256_file(p) for k, p in paths.items()}
    gpu_seconds = time.monotonic() - started
    if anchor_frames:
        anchor = anchor_check(encoder, np.concatenate(anchor_frames), np.concatenate(anchor_pooled))
    return {
        "device": str(device),
        "files_sha256": files,
        "roots": table,
        "anchor": anchor,
        "seconds": time.monotonic() - started,
        "featurisation_seconds": gpu_seconds,
        "probe": bool(probe),
    }


def _remap(store, path):
    """Flush and unmap a store, and map it again (its written pages leave the process's PSS)."""
    store.flush()
    del store
    return np.load(path, mmap_mode="r+")


FEATURE_FILES = {  # every file a split's featurisation writes (its sha256 is in the report)
    "features": "features8_{split}.npy",
    "commands": "commands_{split}.npy",
    "roots": "roots_{split}.json",
    "hidden8": "hidden8_r_{split}.npy",
    "pool4": "pool4_405_r_{split}.npy",
    "full405": "full405_{split}.npy",
    "table": "table_{split}.npz",
}


def feature_paths(folder, split: str) -> dict:
    return {k: Path(folder) / v.format(split=split) for k, v in FEATURE_FILES.items()}


def verify_feature_files(folder, files_sha256: dict, splits) -> dict:
    """G-split (the #143 approval's note 1): before a stage reads a split's featurisation, every
    file of that split is hashed and compared with the featurise report's ``files_sha256``. A
    split the report lacks, a missing file or a different sha256 is a GuardError (V)."""
    started = time.monotonic()
    checked, size = {}, 0
    for split in splits:
        want = files_sha256.get(split)
        if not isinstance(want, dict) or set(want) != set(FEATURE_FILES):
            raise GuardError(f"G-split: the featurise report records no complete {split} split")
        for key, path in feature_paths(folder, split).items():
            if not path.is_file():
                raise GuardError(f"G-split: the feature file {path.name} is missing")
            got = sha256_file(path)
            if got != want[key]:
                raise GuardError(
                    f"G-split: {path.name} sha256 {got[:12]} differs from the featurise report's "
                    f"{str(want[key])[:12]}"
                )
            checked[f"{split}/{key}"] = got
            size += path.stat().st_size
    return {
        "splits": list(splits),
        "files": len(checked),
        "bytes": int(size),
        "seconds": time.monotonic() - started,
        "ok": True,
    }


def load_table(folder, split: str) -> dict:
    with np.load(Path(folder) / f"table_{split}.npz") as data:
        return {k: data[k] for k in data.files}


# ----- Stage O: admission and the downstream fits (CPU) -----------------------------------------
def ridge(x, y, groups):
    from embodied_jepa.models.latent_critic import RidgeReadout

    return RidgeReadout.fit(
        x,
        y,
        groups,
        lambdas=lm.LAMBDA_GRID_RELATIVE,
        folds=lm.INNER_FOLDS,
        seed=lm.SALTS["inner_folds"],
        dual=True,
    )


def save_readout(path, readout) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    np.savez(path, **readout.state())
    return readout.sha256()


def errors_cm(pred, y) -> np.ndarray:
    return 100.0 * np.linalg.norm(np.asarray(pred) - np.asarray(y), axis=1)


def readouts_core(feat, out, *, tau_commit_cm, check=None, probe: bool = False) -> dict:
    """Stage O's CPU part. ``feat``: the featurisation folder. Returns the report section and the
    paths of every downstream fit (refuses to overwrite)."""
    started = time.monotonic()
    feat, out = Path(feat), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tables = {s: load_table(feat, s) for s in ("train", "val")}  # the gate split: Stage G only
    stores = {s: tr.RootStore.open(feat, s, mmap=True) for s in ("train", "val")}
    tv = {
        "seeds": np.concatenate([tables[s]["seeds"] for s in ("train", "val")]),
        "x_r": np.concatenate(
            [np.asarray(stores[s].features[:, R_INDEX]) for s in ("train", "val")]
        ),
        "hidden": np.concatenate([np.load(feat / f"hidden8_r_{s}.npy") for s in ("train", "val")]),
        "pool4_r": np.concatenate(
            [np.load(feat / f"pool4_405_r_{s}.npy")[:, 1] for s in ("train", "val")]
        ),
        "y_r": np.concatenate([tables[s]["plate"][:, R_INDEX] for s in ("train", "val")]),
        "y405": np.concatenate([tables[s]["plate"][:, START_INDEX] for s in ("train", "val")]),
        "h405": np.concatenate([tables[s]["palm"][:, START_INDEX] for s in ("train", "val")]),
        "g": np.concatenate([tables[s]["target"] for s in ("train", "val")]),
    }
    full405 = np.concatenate(
        [np.load(feat / f"full405_{s}.npy", mmap_mode="r") for s in ("train", "val")]
    )
    n = len(tv["seeds"])
    roots = tv["seeds"].astype(str)
    fold = lm.outer_folds(n)
    yr = tv["y_r"]
    pv, ph, p4, prior = (np.full((n, 2), np.nan) for _ in range(4))
    p405 = np.full((n, 2), np.nan)
    sysid = np.full((n, 2), np.nan)
    curve = {f: np.full((n, 2), np.nan) for f in lm.FRACTIONS}
    sizes = {f: [] for f in lm.FRACTIONS}
    selections = []
    for k in range(lm.OUTER_FOLDS):
        fit, held = np.flatnonzero(fold != k), fold == k
        subsets = lm.nested_subsets(fit, k)
        if not np.array_equal(subsets[1.0], fit):
            raise GuardError("G-curve: the full-data subset is not the fold's fit rows")
        for f in lm.FRACTIONS:
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
        rp = ridge(np.asarray(full405[fit]), tv["y405"][fit], roots[fit])
        p405[held] = rp.predict(np.asarray(full405[held]))
        coef = c1.sysid_fit(tv["y405"][fit], tv["h405"][fit], tv["g"][fit], yr[fit])
        sysid[held] = c1.sysid_predict(coef, p405[held], tv["h405"][held], tv["g"][held])
        if check is not None:
            check()
    for name, values in {
        "visible": pv,
        "hidden": ph,
        "pool4": p4,
        "prior": prior,
        "p405": p405,
        "sysid": sysid,
        **{str(f): v for f, v in curve.items()},
    }.items():
        if not np.isfinite(values).all():
            raise GuardError(f"G-finite: {name} lacks an out-of-fold prediction")
    ev, eh, ep = errors_cm(pv, yr), errors_cm(ph, yr), errors_cm(prior, yr)
    e_curve = {f: errors_cm(curve[f], yr) for f in lm.FRACTIONS}
    guard = lm.median_difference_ci(e_curve[0.75], e_curve[1.0])
    c_plate = lm.median_ci(ev)
    prior_ratio = lm.median_ratio_ci(ev, ep)
    hidden = lm.median_ci(eh)
    falling = bool(guard["ci95"][0] > 0.0)
    admission = {
        "c_plate": c_plate,
        "prior_ratio": prior_ratio,
        "prior": lm.median_ci(ep),
        "hidden": hidden,
        "learning_curve": {
            "fit_rows_per_fold": {str(f): sizes[f] for f in lm.FRACTIONS},
            "median_cm": {str(f): float(np.median(e_curve[f])) for f in lm.FRACTIONS},
            "p87_5_cm": {str(f): float(np.percentile(e_curve[f], 87.5)) for f in lm.FRACTIONS},
            "guard_3_4_minus_all": guard,
            "still_falling": falling,
        },
        "reported": {
            "pool4_at_r": lm.median_ci(errors_cm(p4, yr)),
            "r_plate_405": lm.median_ci(errors_cm(p405, tv["y405"])),
            "sysid_at_r": lm.median_ci(errors_cm(sysid, yr)),
            "plate_motion_405_to_r_cm": lm.median_ci(errors_cm(tv["y405"], yr)),
        },
    }
    decision = (
        lm.decide_o(
            c_plate=c_plate,
            prior_ratio=prior_ratio,
            hidden=hidden,
            tau_commit_cm=tau_commit_cm,
            falling=falling,
        )
        if not lm.is_missing(tau_commit_cm)
        else lm.NOT_EVALUATED
    )
    fits = downstream_fits(feat, out, tables, stores["train"], check=check)
    return {
        "rows": n,
        "folds": fold.tolist(),
        "tau_commit_cm": tau_commit_cm,
        "admission": admission,
        "decision": decision,
        "selections": selections,
        "fits": fits,
        "seconds": time.monotonic() - started,
        "probe": bool(probe),
    }


def downstream_fits(feat, out, tables, train_store, *, check=None) -> dict:
    """The train-only fits every later stage reads (G-split): R-plate (full tokens at 405 ->
    plate at 405), R8 (8 x 8 at r -> plate at r), H-sysid (out-of-fold readings on train), the
    normalisation moments over every kept train frame and L-mean's mean 405 latent."""
    t = tables["train"]
    seeds = t["seeds"].astype(str)
    full = np.asarray(np.load(Path(feat) / "full405_train.npy", mmap_mode="r"))
    y405, yr = t["plate"][:, START_INDEX], t["plate"][:, R_INDEX]
    h405, g = t["palm"][:, START_INDEX], t["target"]
    r_plate = ridge(full, y405, seeds)
    x_r = np.asarray(train_store.features[:, R_INDEX])
    r8 = ridge(x_r, yr, seeds)
    fold = lm.outer_folds(len(seeds))
    reading = np.full((len(seeds), 2), np.nan)
    for k in range(lm.OUTER_FOLDS):
        fit, held = fold != k, fold == k
        reading[held] = ridge(full[fit], y405[fit], seeds[fit]).predict(full[held])
    coef = c1.sysid_fit(reading, h405, g, yr)
    del full
    if check is not None:
        check()
    mean, std = tr.moments_file(Path(feat) / "features8_train.npy")
    mean_latent = np.asarray(train_store.features[:, START_INDEX], np.float64).mean(0)
    out = Path(out)
    paths = {
        "r_plate": out / "r_plate.npz",
        "r8": out / "r8.npz",
        "sysid": out / "sysid.json",
        "moments": out / "moments.npz",
        "mean_latent": out / "mean_latent.npy",
    }
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    record = {
        "r_plate": {
            "path": str(paths["r_plate"]),
            "sha256": save_readout(paths["r_plate"], r_plate),
        },
        "r8": {"path": str(paths["r8"]), "sha256": save_readout(paths["r8"], r8)},
    }
    paths["sysid"].write_text(json.dumps({"coef": coef.tolist()}))
    np.savez(paths["moments"], mean=mean, std=std)
    np.save(paths["mean_latent"], mean_latent.astype(np.float32))
    latent32 = np.ascontiguousarray(mean_latent.astype(np.float32))
    record |= {
        "sysid": {"path": str(paths["sysid"]), "coef": coef.tolist()},
        "moments": {"path": str(paths["moments"]), "sha256": moments_sha256(mean, std)},
        "mean_latent": {
            "path": str(paths["mean_latent"]),
            "sha256": hashlib.sha256(latent32.tobytes()).hexdigest(),
        },
        "selections": {"r_plate": r_plate.selection, "r8": r8.selection},
        "train_roots": int(len(seeds)),
        "train_reading_error_405": lm.median_ci(errors_cm(reading, y405)),
    }
    return record


def moments_sha256(mean, std) -> str:
    return tr.sha256_array(np.stack([np.asarray(mean, np.float64), np.asarray(std, np.float64)]))


# ----- Stage G: collapse statistics via the Gram matrix ------------------------------------------
def _spectrum(x) -> np.ndarray:
    """Nonzero covariance energies of rows ``x`` [n, D] from the centred n x n Gram (exact)."""
    x = np.asarray(x, np.float64)
    xc = x - x.mean(0)
    return np.clip(np.linalg.eigvalsh(xc @ xc.T), 0.0, None)


def effective_rank_from_energy(energy) -> float:
    total = float(np.sum(energy))
    if total <= 0:
        return 0.0
    p = energy[energy > 0] / total
    return float(np.exp(-(p * np.log(p)).sum()))


def collapse_statistics(predicted, encoded) -> dict:
    """G1's quantities on latents already divided by the metric scale, the ranks from the Gram."""
    p, e = np.asarray(predicted, np.float64), np.asarray(encoded, np.float64)
    p_std, e_std = p.std(0), e.std(0)
    p_rank = effective_rank_from_energy(_spectrum(p))
    e_rank = effective_rank_from_energy(_spectrum(e))
    return {
        "predicted_std_mean": float(p_std.mean()),
        "encoded_std_mean": float(e_std.mean()),
        "std_ratio": float(p_std.mean() / e_std.mean()) if e_std.mean() > 0 else 0.0,
        "predicted_collapsed_fraction": float((p_std < lm.COLLAPSE_STD).mean()),
        "encoded_collapsed_fraction": float((e_std < lm.COLLAPSE_STD).mean()),
        "predicted_effective_rank": p_rank,
        "encoded_effective_rank": e_rank,
        "effective_rank_ratio": p_rank / e_rank if e_rank > 0 else 0.0,
        "samples": int(len(p)),
    }


def truncated_statistics(predicted, encoded, k: int) -> dict:
    """``predicted`` projected onto its own top ``k`` principal directions (mean kept)."""
    p = np.asarray(predicted, np.float64)
    pc = p - p.mean(0)
    values, vectors = np.linalg.eigh(pc @ pc.T)
    order = np.argsort(values)[::-1][: int(k)]
    lam = np.clip(values[order], 0.0, None)
    u = vectors[:, order]
    truncated = p.mean(0) + u @ (u.T @ pc)
    stats = collapse_statistics(truncated, encoded)
    stats["k"] = int(k)
    stats["energies"] = lam.tolist()
    return stats


def comparative_rank(w, n_, e, *, resamples: int = lm.COMPARATIVE_RESAMPLES) -> dict:
    """G1 (iii): (rank ratio W - rank ratio N), both against the encoded rank of the same
    resample of gate roots (2 000 resamples, salt 8106)."""
    grams = {}
    for name, x in (("W", w), ("N", n_), ("E", e)):
        x = np.asarray(x, np.float64)
        grams[name] = x @ x.T
    m = len(e)
    rng = np.random.default_rng(lm.SALTS["bootstrap"])

    def rank(g, idx):
        sub = g[np.ix_(idx, idx)]
        j = np.full((len(idx), len(idx)), 1.0 / len(idx))
        centred = sub - j @ sub - sub @ j + j @ sub @ j
        return effective_rank_from_energy(np.clip(np.linalg.eigvalsh(centred), 0.0, None))

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


def calibrate_g1(predicted, encoded) -> dict:
    """G1's bars from the calibration W on val at h = 60, and the truncation controls: the
    combined G1 (i)+(ii) must fail rank-1, 2 and 4 truncations, or the rank bar is raised to
    the next 0.01 above every passing truncation's ratio (disclosed)."""
    stats = collapse_statistics(predicted, encoded)
    bars = lm.g1_bars(stats["effective_rank_ratio"], stats["std_ratio"])
    controls, raised = {}, False
    for k in lm.TRUNCATION_CONTROLS:
        t = truncated_statistics(predicted, encoded, k)
        passes = (
            t["predicted_collapsed_fraction"] <= lm.G_THRESHOLDS["G1_max_collapsed_fraction"]
            and t["effective_rank_ratio"] >= bars["rank"]["bar"]
            and t["std_ratio"] >= bars["std"]["bar"]
        )
        controls[str(k)] = {"stats": t, "passes_g1_i_ii": bool(passes)}
    passing = [c["stats"]["effective_rank_ratio"] for c in controls.values() if c["passes_g1_i_ii"]]
    if passing:
        bars["rank"]["bar"] = math_ceil_01(max(passing))
        raised = True
    return {"stats": stats, "bars": bars, "truncation_controls": controls, "bar_raised": raised}


def math_ceil_01(x: float) -> float:
    return float(np.floor(100.0 * float(x) + 1e-9) + 1.0) / 100.0


# ----- Stage G: E60 on the gate split ------------------------------------------------------------
def predict_at(model, start, commands, horizons, *, chunk: int = 25) -> dict:
    """Predicted raw latents at each horizon in ``horizons`` (CPU), [n, D] each."""
    out = {h: np.empty((len(start), start.shape[1]), np.float32) for h in horizons}
    for lo in range(0, len(start), chunk):
        a = np.ascontiguousarray(commands[lo : lo + chunk], np.float32)
        pred = model.predict_features(np.asarray(start[lo : lo + chunk], np.float32), a)
        for h in horizons:
            out[h][lo : lo + len(a)] = pred[:, h - 1]
    return out


def seed_gates(
    *,
    w_model,
    n_model,
    gate,
    table,
    scale,
    r8,
    bars,
    tau_commit_cm,
    standin_commands=None,
) -> dict:
    """Every gate of one seed on E60 (and the reported horizons)."""
    horizons = (*lm.REPORTED_HORIZONS, lm.HORIZON)
    start, commands, targets = gate.from_405()
    start = np.asarray(start)
    commands = np.asarray(commands, np.float32)
    perm = lm.wrong_permutation(len(gate))
    zero = np.zeros_like(commands)
    pred = {
        "W_true": predict_at(w_model, start, commands, horizons),
        "W_wrong": predict_at(w_model, start, commands[perm], horizons),
        "W_zero": predict_at(w_model, start, zero, horizons),
        "N": predict_at(n_model, start, zero, horizons),
    }
    if standin_commands is not None:
        pred["W_standin"] = predict_at(w_model, start, standin_commands, (lm.HORIZON,))
    scale = np.asarray(scale, np.float64)

    def err(p, h):
        target = np.asarray(targets[:, h - 1], np.float64)
        d = (np.asarray(p, np.float64) - target) / scale
        return (d * d).mean(1)

    per_h = {}
    for h in horizons:
        e = {k: err(v[h], h) for k, v in pred.items() if h in v}
        e["copy"] = err(start, h)
        per_h[h] = {
            "G2": lm.ratio_of_sums_ci(e["W_true"], e["copy"]),
            "G3": lm.ratio_of_sums_ci(e["W_true"], e["N"]),
            "G4_wrong": lm.ratio_of_sums_ci(e["W_wrong"], e["W_true"]),
            "G4_zero": lm.ratio_of_sums_ci(e["W_zero"], e["W_true"]),
            "mse": {k: float(v.mean()) for k, v in e.items()},
        }
    h = lm.HORIZON
    encoded = np.asarray(targets[:, h - 1], np.float64) / scale
    w_norm = np.asarray(pred["W_true"][h], np.float64) / scale
    n_norm = np.asarray(pred["N"][h], np.float64) / scale
    stats = collapse_statistics(w_norm, encoded)
    comparative = comparative_rank(w_norm, n_norm, encoded)
    g1 = lm.g1_passes(stats, bars, comparative)
    plate_r = table["plate"][:, R_INDEX]
    e_w = errors_cm(r8.predict(pred["W_true"][h]), plate_r)
    e_n = errors_cm(r8.predict(pred["N"][h]), plate_r)
    e_enc = errors_cm(r8.predict(np.asarray(targets[:, h - 1])), plate_r)
    g5a = lm.median_ci(e_w)
    g5b = lm.median_ratio_ci(e_w, e_n)
    g = per_h[h]
    gates = {
        "G1": g1["passes"],
        "G2": bool(g["G2"]["ci95"][1] <= lm.G_THRESHOLDS["G2_max_ratio_upper"]),
        "G3": bool(g["G3"]["ci95"][1] < lm.G_THRESHOLDS["G3_max_ratio_upper_exclusive"]),
        "G4": bool(
            g["G4_wrong"]["ci95"][0] >= lm.G_THRESHOLDS["G4_min_ratio_lower"]
            and g["G4_zero"]["ci95"][0] >= lm.G_THRESHOLDS["G4_min_ratio_lower"]
        ),
        "G5": bool(
            g5a["ci95"][1] <= float(tau_commit_cm)
            and g5b["ci95"][1] < lm.G_THRESHOLDS["G5_ratio_upper_exclusive"]
        ),
    }
    reported = {
        "g5_tau_scaled_cm": float(tau_commit_cm) * (1.0 - lm.RULE["kappa"]),
        "encoded_readout_gate": lm.median_ci(e_enc),
        "w_minus_encoded": lm.median_difference_ci(e_w, e_enc),
        "constant_prior": "Stage O (train mean of plate(r))",
    }
    if "W_standin" in pred:
        e_s = errors_cm(r8.predict(pred["W_standin"][h]), plate_r)
        reported["g5_standin"] = lm.median_ci(e_s)
        reported["g5_standin_minus_executed"] = lm.median_difference_ci(e_s, e_w)
    return {
        "gates": gates,
        "G1": {"stats": stats, "comparative": comparative, "parts": g1},
        "G5": {"a": g5a, "b": g5b, "e_N": lm.median_ci(e_n)},
        "per_horizon": {str(k): v for k, v in per_h.items()},
        "reported": reported,
    }
