"""TASK-100: featurise ``so101-play-v1`` with frozen DINOv2, and fit the standardisation
(protocol §3, §8.4, §10.2).

Stages:
  stage0    (GPU) D0 (train episodes of shards 00-02) as split ``d0`` and the smoke's val episodes
            (shards 00-07) as split ``valsmoke``, for the given grids, with the probe of §10.2:
            throughput, GPU against CPU agreement on 64 frames, two featurisations of one
            episode, and (g8) the variance kept at k = 96. g8's projection is fitted on D0-fit.
  trainval  (GPU) Stage F: train and val at the frozen grid; g8's projection on the fit sample.
  test      (GPU) Stage F': the test split, only after the val report's recorded sha256.
  fit       (CPU) the per-dimension standardisation of the flat latent on a fit view (D0-fit at
            Stage 0, the fit sample at Stage F) and, for the pretrained encoder, G1 (iii)'s
            256-direction basis; writes ``moments.npz``, ``basis.npz`` and ``report_fit.json``.
Every output is new (refuses to overwrite); the GPU stages run under ``scripts/gpu_run.sh`` and
stop when ``/`` (or the output's disk) would fall below 10 GiB free.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import run_tools as rt  # noqa: E402
from embodied_jepa import so101_wm as sw  # noqa: E402
from embodied_jepa import so101_wm_data as sd  # noqa: E402
from embodied_jepa import so101_wm_runtime as runtime  # noqa: E402

BYTES_PER_FRAME = sw.LATENT_DIM * 2 + (sw.ACTION_DIM + 3 + 1 + 1) * 4 + 1 + 2


def free_gib(path) -> float:
    return shutil.disk_usage(path).free / 2**30


def check_disk(path, need_bytes: float = 0.0) -> None:
    for where in {Path(path).resolve(), Path("/")}:
        free = free_gib(where)
        if free - need_bytes / 2**30 < sw.DISK_FLOOR_GIB:
            raise rt.GuardError(
                f"G-disk: {where}: {free:.1f} GiB free, need {need_bytes / 2**30:.1f}"
            )


def fit_projection(corpus, fit_episodes, featuriser, workers) -> dict:
    acc = sd.ProjectionAccumulator()
    t0 = time.monotonic()
    for i, episode in enumerate(sd.iter_episodes(corpus, fit_episodes, workers=workers)):
        acc.add(featuriser.pooled(episode["frames"], grids=("g8",))["g8"])
        if i % 50 == 0:
            runtime.log(f"pass A: {i + 1}/{len(fit_episodes)}, {time.monotonic() - t0:.0f} s")
    return acc.fit()


def write_splits(corpus, plan: dict, folders: dict, featuriser, projection, workers) -> dict:
    """Pass B: every split of ``plan`` ({split: episodes}) into each grid folder."""
    files = {g: {} for g in folders}
    timing = {"images": 0, "gpu_seconds": 0.0, "wall_seconds": 0.0}
    t_wall = time.monotonic()
    for split, episodes in plan.items():
        frames = sd.episode_frames(corpus, episodes)
        writers = {g: sd.SplitWriter(f, split, sum(frames), g) for g, f in folders.items()}
        t0 = time.monotonic()
        for i, (episode, (shard, eid, seed)) in enumerate(
            zip(sd.iter_episodes(corpus, episodes, workers=workers), episodes, strict=True)
        ):
            if episode["episode_id"] != eid:
                raise rt.GuardError("G-order: episode order")
            t_gpu = time.monotonic()
            pooled = featuriser.pooled(episode["frames"], grids=tuple(folders))
            timing["gpu_seconds"] += time.monotonic() - t_gpu
            timing["images"] += sum(len(v) for v in episode["frames"].values())
            for g, w in writers.items():
                latents = (
                    sd.g4_latents(pooled[g]) if g == "g4" else sd.project_g8(pooled[g], projection)
                )
                w.add(episode, latents, shard=shard, seed=seed)
            if i % 50 == 0:
                for f in folders.values():
                    check_disk(f)
                runtime.log(
                    f"{split}: {i + 1}/{len(episodes)} episodes, {time.monotonic() - t0:.0f} s"
                )
        for g, w in writers.items():
            files[g][split] = w.close()
    timing["wall_seconds"] = time.monotonic() - t_wall
    timing["images_per_second_gpu"] = timing["images"] / max(timing["gpu_seconds"], 1e-9)
    timing["images_per_second_wall"] = timing["images"] / max(timing["wall_seconds"], 1e-9)
    return {"files": files, "timing": timing}


def probe(corpus, episode, featuriser, encoder) -> dict:
    """§10.2: two GPU featurisations of one episode bit-identical; GPU against CPU on 64 frames."""
    loaded = sd.load_episode(corpus, episode[0], episode[1])
    a = featuriser.pooled(loaded["frames"])
    b = featuriser.pooled(loaded["frames"])
    identical = {g: bool(np.array_equal(a[g], b[g])) for g in a}
    cpu = sd.Featuriser(encoder, device="cpu")
    if cpu.digest != featuriser.digest:
        raise rt.GuardError("G-encoder: the CPU and GPU encoders differ")
    first = {c: v[:64] for c, v in loaded["frames"].items()}
    on_cpu = cpu.pooled(first)
    agreement = {}
    for g in a:
        rows = np.concatenate([a[g][:64]], axis=0)
        d = np.abs(rows.astype(np.float64) - on_cpu[g].astype(np.float64))
        agreement[g] = {
            "max_abs": float(d.max()),
            "mean_abs": float(d.mean()),
            "max_abs_relative_to_std": float(d.max() / rows.std()),
        }
    return {
        "episode": episode[1],
        "gpu_twice_bit_identical": identical,
        "gpu_vs_cpu_64_frames": agreement,
    }


def stage_fit(args) -> int:
    feat = Path(args.features)
    report = runtime.header("fit", features=str(feat), view=args.view)
    store_report = json.loads((feat / "report.json").read_text())
    for name in ("moments.npz", "basis.npz", "report_fit.json"):
        if (feat / name).exists():
            raise SystemExit(f"refusing to overwrite {feat / name}")
    split = sd.VIEWS[args.view][0]
    report["store_files_verified"] = sd.verify_store_files(
        feat, store_report["files_sha256"], (split,)
    )
    identity = sd.Moments(np.zeros(sw.LATENT_DIM), np.ones(sw.LATENT_DIM))
    store = sd.FeatureStore(feat, split, moments=identity)
    _, _, fit_only = sd.VIEWS[args.view]
    mask = np.ones(len(store.table), bool)
    if sd.VIEWS[args.view][1] is not None:
        mask &= np.isin(store.shards, list(sd.VIEWS[args.view][1]))
    if fit_only:
        mask &= store.seeds % sw.FIT_SAMPLE_MODULUS == 0
    view = store.subset(mask)
    t0 = time.monotonic()
    moments = sd.Moments.fit(store, view)
    report["moments_sha256"] = moments.save(feat / "moments.npz")
    report["moments_file_sha256"] = sd.sha256_file(feat / "moments.npz")
    report["fit_episodes"] = int(len(view.episodes))
    report["fit_frames"] = int(view.frames.sum())
    report["dimensions_at_std_floor"] = int((moments.std <= sw.FEATURE_STD_FLOOR).sum())
    report["grid"], report["encoder"] = store_report["grid"], store_report["encoder"]
    if store_report["encoder"] == "pretrained":
        store.moments = moments
        xtx, mean, n = sd.second_moment(store, view)
        basis = sw.fit_basis(xtx, mean, n)
        np.savez(feat / "basis.npz", basis=basis)
        report["basis_sha256"] = sw.basis_sha256(basis)
        report["basis_file_sha256"] = sd.sha256_file(feat / "basis.npz")
        cov = xtx / n - np.outer(mean, mean)
        total = float(np.trace(cov))
        report["basis_variance_kept"] = float(np.einsum("ij,ij->", basis, cov @ basis) / total)
    report["seconds"] = time.monotonic() - t0
    report["finished_utc"] = runtime.utc()
    report["outcome"] = "FIT-DONE"
    runtime.write_json(feat / "report_fit.json", report)
    runtime.log(f"fit on {args.view}: {report['fit_frames']} frames, {report['seconds']:.0f} s")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("stage0", "trainval", "test", "fit"))
    parser.add_argument("--corpus", default=sw.CORPUS)
    parser.add_argument("--encoder", choices=sw.ENCODERS, default="pretrained")
    parser.add_argument("--grids", default="g4,g8", help="stage0: the grids to write")
    parser.add_argument("--output", help="the feature folder (stage0: the encoder's folder)")
    parser.add_argument("--features", help="fit: the feature folder")
    parser.add_argument("--view", default="fit-sample", help="fit: the fit view")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--val-report", default=None)
    parser.add_argument("--val-report-sha256", default=None)
    args = parser.parse_args(argv)
    if args.stage == "fit":
        return stage_fit(args)
    import torch

    from embodied_jepa import devices

    report = runtime.header(f"features-{args.stage}", torch=torch.__version__, encoder=args.encoder)
    strict = args.stage in ("trainval", "test")
    runtime.require_clean(report, strict)
    rt.gpu_guard(report, min_free_gib=4.0, require_lock=True)
    report["determinism"] = devices.configure_determinism("cuda", strict=True)
    report["accelerator"] = devices.accelerator_info("cuda")
    corpus = Path(args.corpus)
    report["corpus_check"] = sd.verify_corpus(corpus)
    episodes = sd.corpus_episodes(corpus)
    out = Path(args.output)
    t0 = time.monotonic()
    featuriser = sd.Featuriser(args.encoder, device="cuda")
    report["encoder_digest"] = featuriser.digest
    report["feature_batch"] = featuriser.batch
    if args.stage == "stage0":
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
        grids = tuple(g for g in args.grids.split(",") if g)
        if not grids or any(g not in sw.GRIDS for g in grids):
            raise SystemExit(f"grids must be among {sorted(sw.GRIDS)}")
        d0 = sd.select(episodes["train"], shards=sw.D0_SHARDS)
        fit = sd.select(d0, shards=sw.D0_FIT_SHARDS)
        smoke = sd.select(episodes["val"], shards=sw.SMOKE_VAL_SHARDS + sw.SMOKE_TEST_SHARDS)
        plan = {"d0": d0, "valsmoke": smoke}
        need = sum(sum(sd.episode_frames(corpus, e)) for e in plan.values()) * BYTES_PER_FRAME
        out.mkdir(parents=True)
        check_disk(out, need * len(grids))
        report["probe"] = probe(corpus, d0[0], featuriser, args.encoder)
        runtime.log(f"probe: {report['probe']}")
        projection = None
        folders = {g: out / g for g in grids}
        if "g8" in grids:
            projection = fit_projection(corpus, fit, featuriser, args.workers)
        result = write_splits(corpus, plan, folders, featuriser, projection, args.workers)
        report["timing"] = result["timing"]
        report["episodes"] = {k: len(v) for k, v in plan.items()} | {"projection_fit": len(fit)}
        for g, folder in folders.items():
            sub = {
                "task": sw.TASK,
                "stage": "stage0",
                "grid": g,
                "encoder": args.encoder,
                "encoder_digest": featuriser.digest,
                "files_sha256": result["files"][g],
                "episodes": report["episodes"],
                "revision": report["revision"],
            }
            if g == "g8":
                sub["projection"] = {
                    "sha256": sd.save_projection(folder / "projection.npz", projection),
                    "fit_view": "d0-fit",
                    "fit_frames": projection["frames"],
                    "explained_variance_k96": projection["explained"],
                }
            runtime.write_json(folder / "report.json", sub)
        report["outcome"] = "F0-DONE"
        name = out / "report.json"
    elif args.stage == "trainval":
        grid = sw.frozen_choices()["grid"]
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
        train, val = episodes["train"], episodes["val"]
        fit = sd.select(train, fit_sample=True)
        need = sum(sum(sd.episode_frames(corpus, e)) for e in (train, val)) * BYTES_PER_FRAME
        out.mkdir(parents=True)
        check_disk(out, need)
        projection = fit_projection(corpus, fit, featuriser, args.workers) if grid == "g8" else None
        result = write_splits(
            corpus, {"train": train, "val": val}, {grid: out}, featuriser, projection, args.workers
        )
        report |= {"grid": grid, "files_sha256": result["files"][grid], "timing": result["timing"]}
        report["episodes"] = {"train": len(train), "val": len(val), "fit_sample": len(fit)}
        if projection is not None:
            report["projection"] = {
                "sha256": sd.save_projection(out / "projection.npz", projection),
                "fit_view": "fit-sample",
                "fit_frames": projection["frames"],
                "explained_variance_k96": projection["explained"],
            }
        report["outcome"] = "F-DONE"
        name = out / "report.json"
    else:
        if args.val_report is None or args.val_report_sha256 is None:
            raise rt.GuardError("G-test: the test split opens only after the val report")
        if sd.sha256_file(args.val_report) != args.val_report_sha256:
            raise rt.GuardError("G-test: the val report differs from its recorded sha256")
        vr = json.loads(Path(args.val_report).read_text())
        if vr.get("outcome") != "VAL-DONE" or vr.get("purpose") != "val":
            raise rt.GuardError("G-test: not a gated VAL-DONE report")
        feat_report = json.loads((out / "report.json").read_text())
        if featuriser.digest != feat_report["encoder_digest"]:
            raise rt.GuardError("G-encoder: another encoder than Stage F's")
        grid = feat_report["grid"]
        projection = None
        if grid == "g8":
            projection = sd.load_projection(
                out / "projection.npz", feat_report["projection"]["sha256"]
            )
        test = episodes["test"]
        check_disk(out, sum(sd.episode_frames(corpus, test)) * BYTES_PER_FRAME)
        result = write_splits(
            corpus, {"test": test}, {grid: out}, featuriser, projection, args.workers
        )
        report |= {"grid": grid, "files_sha256": result["files"][grid], "timing": result["timing"]}
        report["val_report_sha256"] = args.val_report_sha256
        report["episodes"] = {"test": len(test)}
        report["outcome"] = "FT-DONE"
        name = out / "report_test.json"
    runtime.require_clean(report, strict)
    report["seconds"] = time.monotonic() - t0
    report["finished_utc"] = runtime.utc()
    report["memory"] = devices.memory_report("cuda")
    runtime.write_json(name, report)
    runtime.log(f"{report['outcome']} in {report['seconds']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
