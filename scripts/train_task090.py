"""TASK-090 Stage T: train one model (G or N, one seed) on ``play-v1`` train plus one new
collection's train split, select on the common val windows (protocol
docs/experiments/grasp_play_corpus_v1.md §2.2, §5).

TASK-087's model, loss, sampler, optimiser and selection rule, unchanged; the train store is the
combined store (``grasp_corpus_wm.CombinedStore``), read into memory (``--no-mmap``, the default)
or memory-mapped (``--mmap``); the state and palm moments are ``play-v1`` train's (checked bitwise
against G-87100's buffers). One invocation = one job (one ``scripts/gpu_run.sh`` lock). Writes
``<output>/model.pt`` and ``<output>/report.json``; refuses an existing output.

``--debug`` (smoke only): debug model seeds, a debug collection's features in place of the new
corpus's (``--features-dir``), any update count, and ``--time-only`` for the timed smoke.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import grasp_corpus_wm as gc  # noqa: E402
from embodied_jepa import grounded_wm as gw  # noqa: E402
from embodied_jepa import grounded_wm_data as gd  # noqa: E402
from embodied_jepa import run_tools as rt  # noqa: E402

UPDATES = 60_000  # §2.2
SELECT_EVERY = 3_000  # TASK-087's (U / 20)


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc()}] {msg}", flush=True)


def git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def val_criterion(model, val, roots, device) -> float:
    """TASK-087 §6.2: mean latent error of the recursive roll-out at steps 1-8 from every val
    root (here the common val windows)."""
    from embodied_jepa import grounded_wm_model as gm

    start, commands, states, _ = val.root_arrays(roots, horizons=(1,))
    horizons = tuple(range(1, gw.TRAIN_HORIZON + 1))
    pred = gm.predict_at(model, start, commands, states, horizons, chunk=1024, device=device)
    total = 0.0
    for h in horizons:
        target = np.asarray(val.latents[roots + h], np.float32)
        total += float(gw.latent_error(pred[h], target).mean())
    value = total / len(horizons)
    if not np.isfinite(value):
        raise rt.GuardError("G-finite: non-finite val criterion")
    return value


def feature_report(folder: Path, *, debug: bool) -> tuple[dict, str]:
    rep = json.loads((folder / "report.json").read_text())
    if rep.get("outcome") != "F-DONE" or rep.get("task") != gc.TASK:
        raise rt.GuardError(f"G-features: {folder} is not a TASK-090 F-DONE store")
    if bool(rep.get("debug")) != debug:
        raise rt.GuardError(f"G-features: {folder} debug={rep.get('debug')}")
    return rep, gd.sha256_file(folder / "report.json")


def check_frozen_knobs(mmap: bool, cap_seconds: float) -> None:
    """Stage T's store mode and per-job cap are Stage 0's (R27.24); non-debug runs use exactly
    them."""
    if bool(mmap) != gc.TRAIN_MMAP:
        raise SystemExit(f"Stage T uses mmap={gc.TRAIN_MMAP} (R27.24)")
    if float(cap_seconds) != float(gc.TRAIN_CAP_SECONDS):
        raise SystemExit(f"Stage T's cap is {gc.TRAIN_CAP_SECONDS:.0f} s (R27.24)")


def evict(paths) -> None:
    """Drop the files' pages from the page cache (cold reads for the timed smoke)."""
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", required=True, choices=gc.CORPORA)
    parser.add_argument("--arm", required=True, choices=("G", "N"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--select-every", type=int, default=SELECT_EVERY)
    parser.add_argument("--cap-seconds", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mmap", action="store_true", help="memory-map the train store")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--features-dir", nargs=3, metavar=("TRAIN_NEW", "VAL_V2G", "VAL_V1P"))
    parser.add_argument("--evict", action="store_true", help="debug: cold page cache first")
    parser.add_argument("--time-only", action="store_true", help="debug: no selection")
    args = parser.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    gc.check_model_seed(args.arm, args.seed, debug=args.debug)
    if not args.debug:
        if (args.updates, args.select_every) != (UPDATES, SELECT_EVERY):
            raise SystemExit("Stage T uses U = 60 000 and TASK-087's selection points (§2.2)")
        if args.features_dir or args.evict or args.time_only:
            raise SystemExit("--features-dir, --evict and --time-only are debug-only")
        check_frozen_knobs(args.mmap, args.cap_seconds)
    if args.updates % args.select_every:
        raise SystemExit("updates must be a multiple of select_every")
    import torch

    from embodied_jepa import devices
    from embodied_jepa import grasp_corpus_wm_runtime as gr
    from embodied_jepa import grounded_wm_model as gm

    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    if not args.debug and dirty:
        raise rt.GuardError("G-clean: Stage T runs from a clean tracked tree")
    report = {
        "task": gc.TASK,
        "protocol": gc.PROTOCOL,
        "kind": "debug" if args.debug else "job",
        "started_utc": utc(),
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": dirty,
        "argv": sys.argv,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "arm": args.arm,
        "corpus": args.corpus,
        "seed": args.seed,
        "mmap": bool(args.mmap),
    }
    rt.assert_local_import(ROOT, report)
    if args.device == "cuda":
        rt.gpu_guard(report, min_free_gib=4.0, require_lock=True)
        report["determinism"] = devices.configure_determinism("cuda", strict=True)
        report["accelerator"] = devices.accelerator_info("cuda")
    out.mkdir(parents=True)
    (out / "started.json").write_text(json.dumps({"started_utc": report["started_utc"]}))

    # ----- stores and their pins -----
    v1 = gr.V1_FEATURES
    if gd.sha256_file(v1 / "report.json") != gr.V1_FEATURES_REPORT_SHA256:
        raise rt.GuardError("G-features: play-v1's feature report differs from TASK-087's")
    v1_report = json.loads((v1 / "report.json").read_text())
    if args.features_dir:
        train_new, val_v2g, val_v1p = (Path(p) for p in args.features_dir)
    else:
        train_new = gr.new_features(ROOT, args.corpus)
        val_v2g = gr.new_features(ROOT, "play-v2g")
        val_v1p = gr.new_features(ROOT, "play-v1p")
    reports = {"play-v1": gr.V1_FEATURES_REPORT_SHA256}
    new_reports = {}
    for name, folder in (("play-v2g", val_v2g), ("play-v1p", val_v1p)):
        rep, sha = feature_report(folder, debug=args.debug)
        reports[name] = sha
        new_reports[name] = (folder, rep)
    report["feature_reports_sha256"] = reports
    if train_new.resolve() != new_reports[args.corpus][0].resolve():
        raise rt.GuardError("G-features: the train store is not the corpus's feature store")
    t0 = time.monotonic()
    verified = gd.verify_store_files(v1, v1_report["files_sha256"], ("train", "val"))
    for name, (folder, rep) in new_reports.items():
        splits = ("train", "val") if name == args.corpus else ("val",)
        verified |= {
            f"{name}/{k}": v
            for k, v in gd.verify_store_files(folder, rep["files_sha256"], splits).items()
        }
    report["store_files_verified"] = verified
    report["verify_seconds"] = time.monotonic() - t0
    if args.evict:
        evict(
            [v1 / "latents_train.npy", train_new / "latents_train.npy"]
            + [p / "latents_val.npy" for p in (v1, val_v2g, val_v1p)]
        )
    t0 = time.monotonic()
    train = gc.CombinedStore([(v1, "train"), (train_new, "train")], mmap=args.mmap)
    val = gc.CombinedStore([(v1, "val"), (val_v2g, "val"), (val_v1p, "val")], mmap=False)
    report["load_seconds"] = time.monotonic() - t0
    val_roots, _ = val.roots()
    pinned = gc.pinned_moments(v1)
    sampler = gd.WindowSampler(train, args.seed)
    report["data"] = {
        "train_frames": int(len(train.state)),
        "train_episodes": len(train.table),
        "train_windows": int(len(sampler.starts)),
        "val_episodes": len(val.table),
        "val_roots": int(len(val_roots)),
        "parts": train.parts,
        "val_parts": val.parts,
    }
    combined = train.moments()
    report["moments"] = {
        "combined_set_reported": {
            k: np.asarray(v, np.float64).tolist()
            for k, v in zip(
                ("state_mean", "state_std", "palm_mean", "palm_std"), combined, strict=True
            )
        }
    }
    torch.manual_seed(args.seed)
    model = gm.GroundedWM(args.arm).to(args.device)
    model.set_moments(*pinned)
    buffers = gr.model_buffers(model)
    ref = gr.checkpoint_buffers(gr.V1_CHECKPOINTS / gr.PIN_MODEL)
    if not gc.same_buffers(buffers, ref):
        raise rt.GuardError("G-moments: the pinned moments differ from G-87100's buffers")
    report["moments"] |= {
        "pinned_sha256": gc.buffers_sha256(buffers),
        "g87100_sha256": gc.buffers_sha256(ref),
        "bitwise_equal_g87100": True,
    }
    report["parameters"] = int(sum(p.numel() for p in model.parameters()))
    report["implementation_sha256"] = gm.implementation_sha256()
    if report["implementation_sha256"] != gr.IMPLEMENTATION_SHA256:
        raise rt.GuardError("G-implementation: TASK-087's code changed")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=gw.LEARNING_RATE, weight_decay=gw.WEIGHT_DECAY
    )
    flags = gw.check_arm(args.arm)
    needs_state = flags["state_token"] or flags["fusion"]
    sync = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)
    curve, states, losses, step_seconds = [], {}, [], []
    started = time.monotonic()
    log(f"{args.arm}-{args.corpus}-{args.seed}: {args.updates} updates, mmap={args.mmap}")
    for step in range(1, args.updates + 1):
        t0 = time.perf_counter()
        batch = train.gather(sampler.draw(), gw.WINDOW_FRAMES)
        z = torch.as_tensor(batch["latents"], device=args.device)
        a = torch.as_tensor(batch["actions"], device=args.device)
        s = torch.as_tensor(batch["state"], device=args.device) if needs_state else None
        palm = torch.as_tensor(batch["palm"], device=args.device) if flags["readout"] else None
        model.train()
        loss, metrics = model.loss(z, a, s, palm)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gw.GRADIENT_CLIP)
        optimizer.step()
        if step % 100 == 0:
            sync()
            if not np.isfinite(metrics["loss"]):
                raise rt.GuardError("G-finite: non-finite training loss")
            losses.append([step, metrics])
            if time.monotonic() - started > args.cap_seconds:
                raise rt.GuardError(f"G-cap: exceeded {args.cap_seconds} s")
        step_seconds.append(time.perf_counter() - t0)
        if step % args.select_every == 0 and not args.time_only:
            value = val_criterion(model, val, val_roots, args.device)
            curve.append([step, value])
            states[step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log(f"update {step}, val {value:.6f}, loss {metrics['loss']:.5f}")
    sync()
    sec = np.asarray(step_seconds[10:] if len(step_seconds) > 20 else step_seconds)
    timing = {
        "seconds": time.monotonic() - started,
        "seconds_per_update_median": float(np.median(sec)),
        "seconds_per_update_mean": float(np.mean(sec)),
        "seconds_per_update_p90": float(np.percentile(sec, 90)),
    }
    if args.time_only:
        report["training"] = {"updates": args.updates, "time_only": True, **timing}
        report["finished_utc"] = utc()
        report["outcome"] = "T-TIMED"
        (out / "report.json").write_text(json.dumps(report, indent=1))
        log(json.dumps(timing))
        return 0
    kept = rt.select_checkpoint(curve, gw.SELECTION_TOLERANCE)
    model.load_state_dict(states[kept])
    if not gc.same_buffers(gr.model_buffers(model), ref):
        raise rt.GuardError("G-moments: the kept checkpoint's buffers changed")
    report["training"] = {
        "updates": args.updates,
        "select_every": args.select_every,
        "val_curve": curve,
        "kept_update": int(kept),
        "kept_val_criterion": float(dict(curve)[kept]),
        "raw_argmin_update": int(min(curve, key=lambda p: p[1])[0]),
        "last_two_triggered": bool(rt.last_two_triggered(curve, gw.SELECTION_TOLERANCE)),
        "losses": losses,
        **timing,
    }
    if args.device == "cuda":
        report["memory"] = devices.memory_report("cuda")
    checkpoint = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "arm": args.arm,
        "corpus": args.corpus,
        "seed": args.seed,
        "config": model.config,
        "implementation_sha256": report["implementation_sha256"],
        "feature_reports_sha256": reports,
        "kept_update": int(kept),
    }
    torch.save(checkpoint, out / "model.pt")
    report["checkpoint_sha256"] = gd.sha256_file(out / "model.pt")
    report["state_sha256"] = gm.state_sha256(model)
    report["finished_utc"] = utc()
    report["outcome"] = "T-JOB-DONE"
    (out / "report.json").write_text(json.dumps(report, indent=1))
    log(f"kept {kept}, last_two {report['training']['last_two_triggered']}, done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
