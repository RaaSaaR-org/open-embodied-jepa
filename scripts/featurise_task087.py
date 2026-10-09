"""TASK-087 Stages F and F': featurise play-v1 with frozen DINOv2 (protocol §3, §12).

``--stage trainval`` (F): pass A featurises the fit sample (train episodes whose seed is divisible
by 4) and fits the standardisation and the k = 192 projection; pass B featurises every train and
val episode and writes the projected float16 latents. ``--stage test`` (F'): only after the val
report exists and its sha256 is given; featurises the test split with the frozen projection.
One GPU job (``scripts/gpu_run.sh``); refuses an existing output; stops when ``/`` falls below
10 GiB free.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import grounded_wm as gw  # noqa: E402
from embodied_jepa import grounded_wm_data as gd  # noqa: E402
from embodied_jepa import run_tools as rt  # noqa: E402

DISK_FLOOR_GIB = 10.0
BYTES_PER_FRAME = gw.TOKENS * gw.K * 2 + (gw.STATE_DIM + gw.ACTION_DIM + 3) * 4


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc()}] {msg}", flush=True)


def git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def free_gib(path) -> float:
    return shutil.disk_usage(path).free / 2**30


def check_disk(path, need_bytes: float = 0.0) -> None:
    free = free_gib(path)
    if free - need_bytes / 2**30 < DISK_FLOOR_GIB:
        raise rt.GuardError(
            f"G-disk: {free:.1f} GiB free, {need_bytes / 2**30:.1f} GiB needed, floor 10 GiB"
        )


def write_split(corpus, episodes, split, writer_dir, featuriser, projection, workers, delta_acc):
    frames = gd.episode_frames(corpus, episodes)
    writer = gd.SplitWriter(writer_dir, split, sum(frames))
    t0 = time.monotonic()
    for i, (episode, (shard, eid, seed)) in enumerate(
        zip(gd.iter_episodes(corpus, episodes, workers=workers), episodes, strict=True)
    ):
        if episode["episode_id"] != eid:
            raise rt.GuardError("G-order: episode order")
        pooled = featuriser.pooled(episode["frames"])
        latents = gd.project(pooled, projection)
        if delta_acc is not None and seed % gw.FIT_SAMPLE_MODULUS == 0:
            z = (pooled.astype(np.float64) - projection["mean"]) / projection["std"]
            d_full = np.diff(z, axis=0)
            d_kept = np.diff(latents.astype(np.float64), axis=0)
            delta_acc[0] += float((d_kept * d_kept).sum())
            delta_acc[1] += float((d_full * d_full).sum())
        writer.add(episode, latents, shard=shard, seed=seed)
        if i % 100 == 0:
            check_disk(writer_dir)
            log(f"{split}: {i + 1}/{len(episodes)} episodes, {time.monotonic() - t0:.0f} s")
    return writer.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default=gw.CORPUS)
    parser.add_argument("--stage", required=True, choices=("trainval", "test"))
    parser.add_argument("--output", required=True, help="the feature store folder")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--debug", type=int, default=None, help="smoke: N episodes per split")
    parser.add_argument("--val-report", default=None)
    parser.add_argument("--val-report-sha256", default=None)
    args = parser.parse_args(argv)
    out = Path(args.output)
    report = {
        "task": gw.TASK,
        "protocol": gw.PROTOCOL,
        "stage": args.stage,
        "debug": args.debug,
        "started_utc": utc(),
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "argv": sys.argv,
        "platform": platform.platform(),
    }
    rt.assert_local_import(ROOT, report)
    rt.gpu_guard(report, min_free_gib=4.0, require_lock=args.debug is None)
    corpus = Path(args.corpus)
    report["corpus_json_sha256"] = gd.sha256_file(corpus / "corpus.json")
    episodes = gd.corpus_episodes(corpus, check_counts=args.debug is None)
    t0 = time.monotonic()
    if args.stage == "trainval":
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
        train, val = episodes["train"], episodes["val"]
        fit = gd.fit_sample(train)
        if args.debug is not None:
            fit, train, val = fit[: args.debug], fit[: args.debug], val[: max(2, args.debug // 4)]
        need = sum(gd.episode_frames(corpus, train)) + sum(gd.episode_frames(corpus, val))
        check_disk(corpus.parent, need * BYTES_PER_FRAME)
        report["disk_free_gib_before"] = free_gib(corpus.parent)
        out.mkdir(parents=True)
        featuriser = gd.Featuriser()
        report["encoder_digest"] = featuriser.digest
        log(f"pass A: {len(fit)} fit-sample episodes")
        acc = gd.ProjectionAccumulator()
        for i, episode in enumerate(gd.iter_episodes(corpus, fit, workers=args.workers)):
            acc.add(featuriser.pooled(episode["frames"]))
            if i % 100 == 0:
                log(f"pass A: {i + 1}/{len(fit)}, {time.monotonic() - t0:.0f} s")
        projection = acc.fit()
        report["projection"] = {
            "sha256": gd.save_projection(out / "projection.npz", projection),
            "fit_episodes": len(fit),
            "fit_frames": projection["frames"],
            "explained_variance": projection["explained"],
            "file_sha256": gd.sha256_file(out / "projection.npz"),
        }
        log(f"projection: {projection['explained']:.4f} of the variance kept")
        report["pass_a_seconds"] = time.monotonic() - t0
        files, delta = {}, [0.0, 0.0]
        for split, eps in (("train", train), ("val", val)):
            files[split] = write_split(
                corpus,
                eps,
                split,
                out,
                featuriser,
                projection,
                args.workers,
                delta if split == "train" else None,
            )
        report["projection"]["change_variance_kept"] = delta[0] / max(delta[1], 1e-30)
        report["files_sha256"] = files
        report["episodes"] = {"train": len(train), "val": len(val), "fit": len(fit)}
    else:
        if args.val_report is None or args.val_report_sha256 is None:
            raise rt.GuardError("G-test: the test split opens only after the val report")
        val_report = Path(args.val_report)
        if gd.sha256_file(val_report) != args.val_report_sha256:
            raise rt.GuardError("G-test: the val report differs from its recorded sha256")
        vr = json.loads(val_report.read_text())
        if vr.get("outcome") != "VAL-DONE" or (vr.get("debug") and args.debug is None):
            raise rt.GuardError("G-test: not a real VAL-DONE report")
        feat_report = json.loads((out / "report.json").read_text())
        projection = gd.load_projection(out / "projection.npz", feat_report["projection"]["sha256"])
        test = episodes["test"] if args.debug is None else episodes["test"][: args.debug]
        check_disk(corpus.parent, sum(gd.episode_frames(corpus, test)) * BYTES_PER_FRAME)
        featuriser = gd.Featuriser()
        if featuriser.digest != feat_report["encoder_digest"]:
            raise rt.GuardError("G-encoder: another encoder than Stage F's")
        report["val_report_sha256"] = args.val_report_sha256
        report["projection_sha256"] = feat_report["projection"]["sha256"]
        report["files_sha256"] = {
            "test": write_split(
                corpus, test, "test", out, featuriser, projection, args.workers, None
            )
        }
        report["episodes"] = {"test": len(test)}
    report["seconds"] = time.monotonic() - t0
    report["disk_free_gib_after"] = free_gib(corpus.parent)
    report["finished_utc"] = utc()
    report["outcome"] = "F-DONE" if args.stage == "trainval" else "FT-DONE"
    name = "report.json" if args.stage == "trainval" else "report_test.json"
    path = out / name
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.write_text(json.dumps(report, indent=1))
    log(f"{report['outcome']} in {report['seconds']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
