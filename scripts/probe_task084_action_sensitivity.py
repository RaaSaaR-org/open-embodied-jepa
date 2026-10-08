"""TASK-084 (Phase 0) part B: action-sensitivity baseline of the existing LeWM checkpoints.

Protocol: ``docs/experiments/jepa_wms_pusht_calibration.md`` §5. Reported only; it gates nothing.
CPU only; TASK-077's three kept W checkpoints (model seeds 66800-66802, the ones TASK-080/081/083
ran in closed loop) as they are; the **val** split of ``apple-c1m-v2`` only (the gate split is
never opened; the train split is not read). No training, no collection, no closed loop.

For each seed and each horizon h it measures, from frame 405 of every val root:

- ``wrong_over_true`` and ``zero_over_true``: TASK-077's G4 statistic (ratio of summed normalised
  MSE to the encoded frame at h, root-bootstrapped, salt 8106) at short horizons, with the wrong
  commands a derangement of the val roots (salt 8110, as G4).
- ``ranking``: among K = 16 candidate command sequences (the root's own executed commands and 15
  other val roots' commands, drawn once with salt 8601), the rank of the true one when every
  candidate is scored by the normalised squared distance of its predicted latent at h to the
  encoded frame at h; top-1 accuracy (chance 1/16) and mean normalised rank (0 best, 1 worst;
  chance 0.5), root-bootstrapped (salt 8106).

One invocation writes ``<output>/report.json`` and refuses an existing output.
"""

from __future__ import annotations

import os

THREAD_ENV = {"OMP_NUM_THREADS": "6", "MKL_NUM_THREADS": "6", "OPENBLAS_NUM_THREADS": "6"}
os.environ.update(THREAD_ENV)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import lewm_c1m_v2 as lm  # noqa: E402
from embodied_jepa import lewm_c1m_v2_offline as off  # noqa: E402
from embodied_jepa import lewm_c1m_v2_train as tr  # noqa: E402

CORPUS_SHA256 = "ad8974b2a8b560bb974c6e0b4f90bd3f1fc79a535ebe46ef6c409bde7e4343fb"
SPLIT = "val"
HORIZONS = (1, 2, 4, 8, 16, 30, 60)
CANDIDATES = 16
CANDIDATE_SALT = 8601
MODEL_SEEDS = (66800, 66801, 66802)


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc()}] {msg}", flush=True)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def candidate_table(n: int, k: int = CANDIDATES) -> np.ndarray:
    """Row i: root i first, then k - 1 distinct other roots (salt 8601), drawn once."""
    rng = np.random.default_rng(CANDIDATE_SALT)
    table = np.empty((n, k), np.int64)
    for i in range(n):
        others = np.delete(np.arange(n), i)
        table[i, 0] = i
        table[i, 1:] = rng.choice(others, size=k - 1, replace=False)
    return table


def mean_ci(values) -> dict:
    values = np.asarray(values, np.float64)
    idx = lm.bootstrap_index(len(values))
    boot = values[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(values.mean()), "ci95": [float(lo), float(hi)], "n": int(len(values))}


def check_inputs(args) -> dict:
    features = Path(args.features)
    if "gate" in str(features):
        raise SystemExit("the gate split is never opened")
    feat_report = json.loads((features.parent / "report.json").read_text())
    if feat_report.get("outcome") != "FEATURISED" or feat_report.get("debug"):
        raise SystemExit("the featurise report is not a real FEATURISED report")
    if feat_report.get("corpus_sha256") != CORPUS_SHA256:
        raise SystemExit("the featurise report was made from another corpus")
    verified = off.verify_feature_files(
        features, feat_report["fields"]["featurisation"]["files_sha256"], (SPLIT,)
    )
    fits = Path(args.fits)
    ro = json.loads((fits.parent / "report.json").read_text())
    if ro.get("outcome") != "O-PASS" or ro.get("corpus_sha256") != CORPUS_SHA256:
        raise SystemExit("the readouts report is not the O-PASS report of this corpus")
    with np.load(fits / "moments.npz") as d:
        mean, std = d["mean"], d["std"]
    moments = ro["fields"]["fits"]["moments"]["sha256"]
    if off.moments_sha256(mean, std) != moments:
        raise SystemExit("the moments differ from Stage O's sha256")
    jobs = {}
    for path in args.models:
        path = Path(path)
        job = json.loads(path.read_text())
        if job.get("outcome") != "T-JOB-DONE" or job.get("debug"):
            raise SystemExit(f"{path} is not a real T-JOB-DONE report")
        record = dict(job["fields"]["record"])
        if record["arm"] != "W":
            raise SystemExit(f"{path} is not a W model")
        meta = record["metadata"]
        if meta.get("corpus_manifest_sha256") != CORPUS_SHA256:
            raise SystemExit(f"{path}: another corpus")
        if meta.get("normalisation_sha256") != moments:
            raise SystemExit(f"{path}: other moments")
        record["checkpoint"] = str(path.parent / Path(record["checkpoint"]).name)
        record["report"] = str(path.resolve())
        record["report_sha256"] = sha256_file(path)
        jobs[int(record["seed"])] = record
    if tuple(sorted(jobs)) != MODEL_SEEDS:
        raise SystemExit(f"need W of {MODEL_SEEDS}, got {sorted(jobs)}")
    return {
        "features": features,
        "verified": verified,
        "scale": np.maximum(np.asarray(std, np.float64), lm.METRIC_FLOOR_STD),
        "moments_sha256": moments,
        "jobs": jobs,
    }


def measure(model, start, commands, targets, scale, perm, table) -> dict:
    n = len(start)
    zero = np.zeros_like(commands)

    def err(pred, h):
        d = (np.asarray(pred, np.float64) - np.asarray(targets[:, h - 1], np.float64)) / scale
        return (d * d).mean(1)

    pred = {
        "true": off.predict_at(model, start, commands, HORIZONS),
        "wrong": off.predict_at(model, start, commands[perm], HORIZONS),
        "zero": off.predict_at(model, start, zero, HORIZONS),
    }
    # Candidate j of every root: start of root i, commands of root table[i, j].
    cand = np.empty((len(HORIZONS), n, CANDIDATES), np.float64)
    for j in range(CANDIDATES):
        p = off.predict_at(model, start, commands[table[:, j]], HORIZONS)
        for a, h in enumerate(HORIZONS):
            cand[a, :, j] = err(p[h], h)
    out = {}
    for a, h in enumerate(HORIZONS):
        e = {k: err(v[h], h) for k, v in pred.items()}
        e["copy"] = err(start, h)
        scores = cand[a]
        # Rank of the true candidate (column 0); ties count against it (stable worst case).
        rank = (scores[:, 1:] <= scores[:, :1]).sum(1)
        out[str(h)] = {
            "wrong_over_true": lm.ratio_of_sums_ci(e["wrong"], e["true"]),
            "zero_over_true": lm.ratio_of_sums_ci(e["zero"], e["true"]),
            "true_over_copy": lm.ratio_of_sums_ci(e["true"], e["copy"]),
            "ranking": {
                "top1": mean_ci(rank == 0),
                "normalised_rank": mean_ci(rank / (CANDIDATES - 1)),
                "chance_top1": 1.0 / CANDIDATES,
                "chance_normalised_rank": 0.5,
            },
            "mse": {k: float(v.mean()) for k, v in e.items()},
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", required=True)
    parser.add_argument("--fits", required=True)
    parser.add_argument("--models", nargs=3, required=True, help="the three W T-JOB reports")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None, help="smoke only: first N val roots")
    args = parser.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    import torch

    torch.set_num_threads(6)
    report = {
        "task": "TASK-084 part B (reported only)",
        "document": "docs/experiments/jepa_wms_pusht_calibration.md",
        "started_utc": utc(),
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "script_sha256": sha256_file(Path(__file__)),
        "argv": sys.argv,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "split": SPLIT,
        "horizons": list(HORIZONS),
        "candidates": CANDIDATES,
        "salts": {
            "candidates": CANDIDATE_SALT,
            "wrong": lm.SALTS["wrong_commands"],
            "bootstrap": lm.SALTS["bootstrap"],
        },
        "limit": args.limit,
    }
    t0 = time.monotonic()
    ctx = check_inputs(args)
    store = tr.RootStore.open(ctx["features"], SPLIT, mmap=True)
    start, commands, targets = store.from_405()
    n = len(store) if args.limit is None else min(args.limit, len(store))
    start = np.asarray(start[:n], np.float32)
    commands = np.asarray(commands[:n], np.float32)
    targets = np.asarray(targets[:n], np.float32)
    perm = lm.wrong_permutation(n)
    table = candidate_table(n)
    report["inputs"] = {
        "features": str(ctx["features"].resolve()),
        "feature_files_verified": ctx["verified"],
        "corpus_sha256": CORPUS_SHA256,
        "moments_sha256": ctx["moments_sha256"],
        "roots": int(n),
        "candidate_table_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
        "models": {
            str(s): {
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
            for s, j in sorted(ctx["jobs"].items())
        },
    }
    report["seeds"] = {}
    for s in MODEL_SEEDS:
        job = ctx["jobs"][s]
        model = tr.load_model(
            job["checkpoint"],
            seed=s,
            metadata=job["metadata"],
            expected_sha256=job["checkpoint_sha256"],
        )
        t = time.monotonic()
        report["seeds"][str(s)] = measure(
            model, start, commands, targets, ctx["scale"], perm, table
        )
        log(f"seed {s} done ({time.monotonic() - t:.0f} s)")
        del model
    report["total_seconds"] = time.monotonic() - t0
    report["ended_utc"] = utc()
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=float))
    log(f"report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
