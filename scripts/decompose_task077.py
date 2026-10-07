"""TASK-077 decomposition record (development; R17.51, R17.53, R17.54).

Record: ``docs/experiments/apple_lewm_c1m_v2_decomposition.md`` (§1 declares everything this
script computes and the decision rule it applies). CPU only; the six kept Stage T checkpoints as
they are; the **train and val** splits of ``apple-c1m-v2`` only (the gate split is never opened);
no training, no collection, no closed loop. It gates no claim and changes no TASK-077 row.

One invocation writes ``<output>/report.json`` (refuses an existing output) and stops with an
error, and no row, on any failed check.
"""

from __future__ import annotations

import os

THREAD_ENV = {
    "MKL_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "6",
    "MKL_NUM_THREADS": "6",
    "OPENBLAS_NUM_THREADS": "16",
}
os.environ.update(THREAD_ENV)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import first_policy_v2_linux as fpl  # noqa: E402
from embodied_jepa import lewm_c1m_v2 as lm  # noqa: E402
from embodied_jepa import lewm_c1m_v2_offline as off  # noqa: E402
from embodied_jepa import lewm_c1m_v2_train as tr  # noqa: E402
from embodied_jepa import plate_twin_v2_harness as hz  # noqa: E402
from embodied_jepa import run_guards as rg  # noqa: E402
from embodied_jepa import run_tools as rt  # noqa: E402

fpl.configure_headless()
GIB = 2**30
SPLITS = ("train", "val")
HORIZONS = (16, 30, 60)
TAU_COMMIT_CM = 1.0  # K0 (protocol §7.1); checked against the frozen block below
CORPUS_SHA256 = "ad8974b2a8b560bb974c6e0b4f90bd3f1fc79a535ebe46ef6c409bde7e4343fb"
MEMORY_CEILING_GIB = 12.0
CHUNK_CAP_SECONDS = 3600.0
G1_RESAMPLES = lm.COMPARATIVE_RESAMPLES


class CheckFailed(RuntimeError):
    pass


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc()}] {msg}", flush=True)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def no_gate(path) -> Path:
    path = Path(path)
    if "gate" in path.name:
        raise CheckFailed(f"the gate split is never opened: {path}")
    return path


def git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


# ----- inputs -----------------------------------------------------------------------------------
def check_inputs(args, report: dict) -> dict:
    features = Path(args.features)
    feat_report = json.loads((features.parent / "report.json").read_text())
    if feat_report.get("outcome") != "FEATURISED" or feat_report.get("debug"):
        raise CheckFailed("the featurise report is not a real FEATURISED report")
    if feat_report.get("corpus_sha256") != CORPUS_SHA256:
        raise CheckFailed("the featurise report was made from another corpus")
    files = feat_report["fields"]["featurisation"]["files_sha256"]
    verified = off.verify_feature_files(features, files, SPLITS)
    fits = Path(args.fits)
    ro = json.loads((fits.parent / "report.json").read_text())
    if ro.get("outcome") != "O-PASS" or ro.get("corpus_sha256") != CORPUS_SHA256:
        raise CheckFailed("the readouts report is not the O-PASS report of this corpus")
    rec = ro["fields"]["fits"]
    with np.load(fits / "moments.npz") as d:
        mean, std = d["mean"], d["std"]
    if off.moments_sha256(mean, std) != rec["moments"]["sha256"]:
        raise CheckFailed("the moments differ from Stage O's sha256")
    from embodied_jepa.models.latent_critic import RidgeReadout

    with np.load(fits / "r8.npz") as d:
        r8 = RidgeReadout.from_state({k: d[k] for k in d.files})
    if r8.sha256() != rec["r8"]["sha256"]:
        raise CheckFailed("R8 differs from Stage O's sha256")
    jobs = {}
    for path in args.models:
        path = Path(path)
        job_report = json.loads(path.read_text())
        if job_report.get("outcome") != "T-JOB-DONE" or job_report.get("debug"):
            raise CheckFailed(f"{path} is not a real T-JOB-DONE report")
        record = dict(job_report["fields"]["record"])
        meta = record["metadata"]
        if meta.get("corpus_manifest_sha256") != CORPUS_SHA256:
            raise CheckFailed(f"{path}: another corpus")
        if meta.get("normalisation_sha256") != rec["moments"]["sha256"]:
            raise CheckFailed(f"{path}: other moments")
        record["checkpoint"] = str(path.parent / Path(record["checkpoint"]).name)
        record["report_sha256"] = sha256_file(path)
        record["report"] = str(path.resolve())
        jobs[(record["arm"], int(record["seed"]))] = record
    want = {(a, s) for a in ("W", "N") for s in lm.MODEL_SEEDS}
    if set(jobs) != want:
        raise CheckFailed(f"need W and N of {lm.MODEL_SEEDS}, got {sorted(jobs)}")
    evidence = hz.check_evidence(Path(args.evidence))
    if lm.K0_MEASURED is None or float(lm.K0_MEASURED["tau_commit_cm"]) != TAU_COMMIT_CM:
        raise CheckFailed("tau_commit differs from K0's frozen value")
    report["inputs"] = {
        "features": str(features.resolve()),
        "featurise_report_sha256": sha256_file(features.parent / "report.json"),
        "feature_files_verified": verified,
        "fits": str(fits.resolve()),
        "readouts_report_sha256": sha256_file(fits.parent / "report.json"),
        "moments_content_sha256": rec["moments"]["sha256"],
        "r8_content_sha256": rec["r8"]["sha256"],
        "corpus_sha256": CORPUS_SHA256,
        "models": {
            f"{a}-{s}": {
                k: j.get(k)
                for k in (
                    "report",
                    "report_sha256",
                    "checkpoint",
                    "checkpoint_sha256",
                    "kept_update",
                    "last_two_triggered",
                )
            }
            for (a, s), j in sorted(jobs.items())
        },
        "evidence": {"root": str(Path(args.evidence).resolve()), "sha256": evidence["sha256"]},
        "tau_commit_cm": TAU_COMMIT_CM,
    }
    return {"r8": r8, "scale": np.maximum(std, lm.METRIC_FLOOR_STD), "jobs": jobs}


def load_splits(features: Path, limit: int | None) -> dict:
    stores = {s: tr.RootStore.open(no_gate(features), s, mmap=True) for s in SPLITS}
    tables = {s: off.load_table(features, s) for s in SPLITS}
    idx = {
        s: np.arange(len(stores[s]) if limit is None else min(limit, len(stores[s])))
        for s in SPLITS
    }
    S = tr.START_INDEX

    def cat(fn):
        return np.concatenate([fn(s)[idx[s]] for s in SPLITS])

    data = {
        "seeds": cat(lambda s: tables[s]["seeds"]),
        "split": np.concatenate([np.full(len(idx[s]), s) for s in SPLITS]),
        "start": cat(lambda s: np.asarray(stores[s].features[:, S])),
        "commands": cat(lambda s: np.asarray(stores[s].commands[:, S : S + lm.HORIZON])),
        "plate": cat(lambda s: tables[s]["plate"]),
        "target": cat(lambda s: tables[s]["target"]),
        "state405": cat(lambda s: tables[s]["state405"]),
        "apple": cat(lambda s: tables[s]["apple"]),
        "last_grasp": cat(lambda s: tables[s]["last_grasp"]),
        "encoded": {
            h: cat(lambda s, h=h: np.asarray(stores[s].features[:, S + h])) for h in HORIZONS
        },
    }
    return data


# ----- measurements -----------------------------------------------------------------------------
def standin_chunks(data: dict, args, report: dict) -> np.ndarray:
    from embodied_jepa import lewm_c1m_v2_runtime as wrt

    config = {"p3_checkpoint": hz.p3_checkpoint(Path(args.evidence)), "torch_threads": 1}
    pool = rg.BoundedPool(
        int(args.workers), wrt.run_task, initializer=wrt.worker_init, initargs=(config,)
    )
    try:
        tasks = [
            {
                "kind": "chunks",
                "key": i,
                "state": data["state405"][i].tolist(),
                "apple": data["apple"][i].tolist(),
                "last_grasp": data["last_grasp"][i].tolist(),
                "step": lm.COMMIT_STEP,
                "targets": [np.asarray(data["target"][i]).tolist()],
            }
            for i in range(len(data["seeds"]))
        ]
        started = time.monotonic()
        got = sorted(pool.map(tasks, CHUNK_CAP_SECONDS, "stand-in chunks"), key=lambda r: r["key"])
        seconds = time.monotonic() - started
    finally:
        report["pool_close"] = pool.close()
    chunks = np.stack([np.asarray(r["chunks"][0], np.float32) for r in got])
    feasible = np.asarray([bool(r["feasible"][0]) for r in got])
    report["standin"] = {
        "roots": int(len(got)),
        "infeasible": int((~feasible).sum()),
        "infeasible_seeds": [int(s) for s in data["seeds"][~feasible]],
        "seconds": seconds,
        "chunks_sha256": hashlib.sha256(np.ascontiguousarray(chunks).tobytes()).hexdigest(),
    }
    if chunks.shape != data["commands"].shape:
        raise CheckFailed(f"stand-in chunks {chunks.shape} != commands {data['commands'].shape}")
    return chunks


def crossfit(x, y, groups, fold, evals=()) -> tuple[np.ndarray, list[np.ndarray]]:
    """Held-out predictions of a ridge fitted per outer fold on ``x``; also each extra array of
    ``evals`` predicted on its held-out rows by the same fold's fit."""
    out = np.full((len(y), 2), np.nan)
    extra = [np.full((len(y), 2), np.nan) for _ in evals]
    for k in range(lm.OUTER_FOLDS):
        fit, held = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        model = off.ridge(x[fit], y[fit], groups[fit])
        out[held] = model.predict(x[held])
        for e, arr in zip(extra, evals, strict=True):
            e[held] = model.predict(arr[held])
    for a in (out, *extra):
        if not np.isfinite(a).all():
            raise CheckFailed("a cross-fitted readout left a row without a prediction")
    return out, extra


def summarise(errors: np.ndarray, data: dict) -> dict:
    return {s: lm.median_ci(errors[data["split"] == s]) for s in SPLITS}


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def spearman_ci(a, b) -> dict:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    idx = lm.bootstrap_index(len(a))
    boot = np.array([spearman(a[i], b[i]) for i in idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"rho": spearman(a, b), "ci95": [float(lo), float(hi)], "n": int(len(a))}


def by_quartile(m, columns: dict) -> dict:
    edges = np.quantile(m, [0.25, 0.5, 0.75])
    q = np.searchsorted(edges, m, side="right")
    out = {"edges": edges.tolist(), "quartiles": []}
    for k in range(4):
        sel = q == k
        out["quartiles"].append(
            {"n": int(sel.sum()), "m_median": float(np.median(m[sel]))}
            | {name: float(np.median(v[sel])) for name, v in columns.items()}
        )
    return out


def comparative_with_centre(w, n_, e, resamples: int = G1_RESAMPLES) -> dict:
    """``lewm_c1m_v2_offline.comparative_rank``'s statistic and resampling (same salt and draw
    order), also returning the resamples' mean and median (reported only)."""
    grams = {
        k: np.asarray(x, np.float64) @ np.asarray(x, np.float64).T
        for k, x in (("W", w), ("N", n_), ("E", e))
    }
    m = len(e)
    rng = np.random.default_rng(lm.SALTS["bootstrap"])

    def rank(g, idx):
        sub = g[np.ix_(idx, idx)]
        j = np.full((len(idx), len(idx)), 1.0 / len(idx))
        centred = sub - j @ sub - sub @ j + j @ sub @ j
        return off.effective_rank_from_energy(np.clip(np.linalg.eigvalsh(centred), 0.0, None))

    full = np.arange(m)
    point = {k: rank(g, full) for k, g in grams.items()}
    diffs, ranks = [], {k: [] for k in grams}
    for _ in range(int(resamples)):
        idx = rng.integers(0, m, m)
        r = {k: rank(g, idx) for k, g in grams.items()}
        for k in r:
            ranks[k].append(r[k])
        diffs.append(r["W"] / r["E"] - r["N"] / r["E"])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "point": point["W"] / point["E"] - point["N"] / point["E"],
        "ci95": [float(lo), float(hi)],
        "resample_mean": float(np.mean(diffs)),
        "resample_median": float(np.median(diffs)),
        "ranks_point": point,
        "ranks_resample_mean": {k: float(np.mean(v)) for k, v in ranks.items()},
        "resamples": int(resamples),
    }


def ub(stat: dict) -> float:
    return float(stat["ci95"][1])


def decide(results: dict) -> dict:
    """§1.5's rule (first match), on the val roots, all three seeds."""
    tau = TAU_COMMIT_CM
    seeds = [str(s) for s in lm.MODEL_SEEDS]
    e60 = results["encoded"]["60"]["val"]
    if ub(e60) > tau:
        return {"row": "D-VOID-CEILING", "sub": None}

    def holds(kind: str, h: str) -> list[bool]:
        out = []
        for s in seeds:
            r = results["seeds"][s][h]
            out.append(ub(r[kind]["val"]) <= tau and ub(r[f"{kind}_over_N"]) < 1.0)
        return out

    checks = {
        "S_60": holds("S", "60"),
        "X_60": holds("X", "60"),
        "X_16": holds("X", "16"),
        "X_30": holds("X", "30"),
    }
    if all(checks["S_60"]):
        row, sub = "D-READOUT", None
    elif all(checks["X_60"]):
        row, sub = "D-COMMAND", None
    else:
        row = "D-TASK"
        sub = "D-TASK/H" if all(checks["X_16"]) or all(checks["X_30"]) else "D-TASK/R"
    return {"row": row, "sub": sub, "checks_per_seed": checks, "tau_commit_cm": tau}


def run(args) -> dict:
    report: dict = {
        "record": "TASK-077 decomposition (development; R17.51, R17.53)",
        "document": "docs/experiments/apple_lewm_c1m_v2_decomposition.md",
        "started_utc": utc(),
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "script_sha256": sha256_file(Path(__file__)),
        "argv": sys.argv,
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV},
        "torch_threads": 1,
        "load_average_at_start": list(os.getloadavg()),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "limit_roots_per_split": args.limit,
        "salts": {
            "outer_folds": lm.SALTS["outer_folds"],
            "inner_folds": lm.SALTS["inner_folds"],
            "bootstrap": lm.SALTS["bootstrap"],
        },
        "model_seeds": list(lm.MODEL_SEEDS),
        "horizons": list(HORIZONS),
    }
    if report["tracked_tree_dirty"] and not args.allow_dirty:
        raise CheckFailed("the tracked tree is dirty")
    import torch

    torch.set_num_threads(1)
    report["torch"] = torch.__version__
    watch = rt.MemoryWatch(int(MEMORY_CEILING_GIB * GIB), 2.0, measure="pss")

    def on_usr1(signum, frame):
        raise CheckFailed(f"memory: {watch.reason}")

    signal.signal(signal.SIGUSR1, on_usr1)
    watch.start()
    t0 = time.monotonic()
    try:
        ctx = check_inputs(args, report)
        data = load_splits(Path(args.features), args.limit)
        n = len(data["seeds"])
        report["roots"] = {s: int((data["split"] == s).sum()) for s in SPLITS}
        fold = lm.outer_folds(n)
        groups = data["seeds"].astype(str)
        report["outer_fold_sizes"] = np.bincount(fold).tolist()
        log(f"inputs checked; {n} roots")
        standin = standin_chunks(data, args, report)
        log("stand-in chunks done")
        diff = standin.astype(np.float64) - data["commands"].astype(np.float64)
        mismatch = np.sqrt((diff**2).mean(axis=(1, 2)))
        report["mismatch"] = {
            "rms_per_root": {s: lm.median_ci(mismatch[data["split"] == s]) for s in SPLITS},
            "rms_per_dim_all_roots": np.sqrt((diff**2).mean(axis=(0, 1))).tolist(),
            "executed_std_per_dim": data["commands"].reshape(-1, 14).std(0).tolist(),
            "standin_std_per_dim": standin.reshape(-1, 14).std(0).tolist(),
        }
        plate = {h: data["plate"][:, tr.START_INDEX + h] for h in HORIZONS}
        results: dict = {"encoded": {}, "seeds": {}}
        enc_err = {}
        for h in HORIZONS:
            pred, _ = crossfit(data["encoded"][h], plate[h], groups, fold)
            enc_err[h] = off.errors_cm(pred, plate[h])
            results["encoded"][str(h)] = summarise(enc_err[h], data)
        log("encoded readouts done")
        scale = ctx["scale"]
        val = data["split"] == "val"
        for s in lm.MODEL_SEEDS:
            t_seed = time.monotonic()
            models = {}
            for arm in ("W", "N"):
                job = ctx["jobs"][(arm, s)]
                models[arm] = tr.load_model(
                    job["checkpoint"],
                    seed=s,
                    metadata=job["metadata"],
                    expected_sha256=job["checkpoint_sha256"],
                )
            zero = np.zeros_like(data["commands"])
            pred = {
                "X": off.predict_at(models["W"], data["start"], data["commands"], HORIZONS),
                "S": off.predict_at(models["W"], data["start"], standin, HORIZONS),
                "N": off.predict_at(models["N"], data["start"], zero, HORIZONS),
            }
            del models
            log(f"seed {s}: roll-outs done")
            per = {}
            for h in HORIZONS:
                y = plate[h]
                px, (pxs,) = crossfit(pred["X"][h], y, groups, fold, evals=(pred["S"][h],))
                ps, _ = crossfit(pred["S"][h], y, groups, fold)
                pn, _ = crossfit(pred["N"][h], y, groups, fold)
                e = {
                    "X": off.errors_cm(px, y),
                    "S": off.errors_cm(ps, y),
                    "N": off.errors_cm(pn, y),
                    "XS": off.errors_cm(pxs, y),
                }
                r = {k: summarise(v, data) for k, v in e.items()}
                r |= {
                    "X_over_N": lm.median_ratio_ci(e["X"][val], e["N"][val]),
                    "S_over_N": lm.median_ratio_ci(e["S"][val], e["N"][val]),
                    "X_minus_E": lm.median_difference_ci(e["X"][val], enc_err[h][val]),
                    "S_minus_X": lm.median_difference_ci(e["S"][val], e["X"][val]),
                    "XS_minus_X": lm.median_difference_ci(e["XS"][val], e["X"][val]),
                }
                if h == lm.HORIZON:
                    fx = off.errors_cm(ctx["r8"].predict(pred["X"][h]), y)
                    fs = off.errors_cm(ctx["r8"].predict(pred["S"][h]), y)
                    r["frozen_R8_X"] = summarise(fx, data)
                    r["frozen_R8_S"] = summarise(fs, data)
                    cols = {
                        "e_X": e["X"],
                        "e_S": e["S"],
                        "d_refit": e["S"] - e["X"],
                        "d_frozen": fs - fx,
                    }
                    r["mismatch_cost"] = {
                        sub: {
                            "spearman_m_d_refit": spearman_ci(mismatch[sel], cols["d_refit"][sel]),
                            "spearman_m_d_frozen": spearman_ci(
                                mismatch[sel], cols["d_frozen"][sel]
                            ),
                            "by_quartile": by_quartile(
                                mismatch[sel], {k: v[sel] for k, v in cols.items()}
                            ),
                        }
                        for sub, sel in (("val", val), ("all", np.ones(n, bool)))
                    }
                    r["g1_iii_val"] = comparative_with_centre(
                        pred["X"][h][val] / scale,
                        pred["N"][h][val] / scale,
                        data["encoded"][h][val] / scale,
                    )
                per[str(h)] = r
            results["seeds"][str(s)] = per
            del pred
            log(f"seed {s}: readouts done ({time.monotonic() - t_seed:.0f} s)")
        report["results"] = results
        report["decision"] = decide(results)
        report["revision_at_end"] = git("rev-parse", "HEAD")
        report["outcome"] = report["decision"]["row"]
    finally:
        watch.stop()
        report["memory"] = watch.summary()
        report["total_seconds"] = time.monotonic() - t0
        report["ended_utc"] = utc()
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features", required=True)
    p.add_argument("--fits", required=True)
    p.add_argument("--models", nargs=6, required=True)
    p.add_argument("--evidence", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="smoke only: roots per split")
    p.add_argument("--allow-dirty", action="store_true", help="smoke only")
    args = p.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    for path in (args.features, args.fits, *args.models):
        no_gate(path)
    out.mkdir(parents=True)
    try:
        report = run(args)
    except Exception as error:  # a failed check: no row
        (out / "failed.json").write_text(json.dumps({"error": repr(error), "at": utc()}))
        raise
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=float))
    log(f"report written: {report['outcome']} {report['decision'].get('sub')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
