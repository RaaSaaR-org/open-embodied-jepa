"""TASK-087: train one arm and seed on the play-v1 feature store; select on val (protocol §5-§6).

One invocation = one job (one ``scripts/gpu_run.sh`` lock). Writes ``<output>/model.pt`` (the kept
checkpoint) and ``<output>/report.json``; refuses an existing output. ``--probe`` is the budget
probe of §6.1 (P, debug seed 87900): it also writes the budget derived by the rule.
"""

from __future__ import annotations

import argparse
import json
import platform
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


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc()}] {msg}", flush=True)


def git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def val_criterion(model, val: gd.FeatureStore, roots, device) -> float:
    """§6.2: mean latent error of the recursive roll-out at steps 1-8 from every val root."""
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", required=True)
    parser.add_argument("--arm", required=True, choices=gw.ARMS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--select-every", type=int, required=True)
    parser.add_argument("--cap-seconds", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--debug", action="store_true", help="debug seeds (smoke, probe)")
    parser.add_argument("--probe", action="store_true", help="the budget probe of §6.1")
    parser.add_argument("--expect-features-report-sha256", default=None)
    args = parser.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    if out.name.endswith("-r2"):
        first = out.with_name(out.name[: -len("-r2")])
        if (first / "report.json").exists() or not (first / "started.json").exists():
            raise SystemExit("an -r2 run is allowed only after an incomplete first run")
    gw.check_model_seed(args.seed, debug=args.debug)
    if args.probe and (args.arm != "P" or args.seed != gw.PROBE_SEED):
        raise SystemExit("the probe is P with seed 87900")
    if args.updates % args.select_every:
        raise SystemExit("updates must be a multiple of select_every")
    strict = not args.debug or args.probe  # the probe runs under the lock on the real store
    import torch

    from embodied_jepa import devices
    from embodied_jepa import grounded_wm_model as gm

    report = {
        "task": gw.TASK,
        "protocol": gw.PROTOCOL,
        "kind": "probe" if args.probe else ("debug" if args.debug else "job"),
        "started_utc": utc(),
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "argv": sys.argv,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "arm": args.arm,
        "seed": args.seed,
    }
    rt.assert_local_import(ROOT, report)
    if args.device == "cuda":
        rt.gpu_guard(report, min_free_gib=4.0, require_lock=strict)
        report["determinism"] = devices.configure_determinism("cuda", strict=True)
        report["accelerator"] = devices.accelerator_info("cuda")
    out.mkdir(parents=True)
    (out / "started.json").write_text(json.dumps({"started_utc": report["started_utc"]}))
    feat_dir = Path(args.features)
    feat_report_path = feat_dir / "report.json"
    feat_report = json.loads(feat_report_path.read_text())
    report["features_report_sha256"] = gd.sha256_file(feat_report_path)
    if args.expect_features_report_sha256 and (
        report["features_report_sha256"] != args.expect_features_report_sha256
    ):
        raise rt.GuardError("G-features: the feature report differs from the recorded one")
    if strict and (feat_report.get("debug") or feat_report.get("outcome") != "F-DONE"):
        raise rt.GuardError("G-features: not a real F-DONE feature store")
    if strict:
        report["store_files_verified"] = gd.verify_store_files(
            feat_dir, feat_report["files_sha256"], ("train", "val")
        )
    train = gd.FeatureStore(feat_dir, "train")
    val = gd.FeatureStore(feat_dir, "val")
    val_roots, _ = val.roots()
    state_mean, state_std, palm_mean, palm_std = train.moments()
    sampler = gd.WindowSampler(train, args.seed)
    report["data"] = {
        "train_frames": int(len(train.state)),
        "train_episodes": len(train.table),
        "train_windows": int(len(sampler.starts)),
        "val_roots": int(len(val_roots)),
    }
    torch.manual_seed(args.seed)
    model = gm.GroundedWM(args.arm).to(args.device)
    model.set_moments(state_mean, state_std, palm_mean, palm_std)
    report["parameters"] = int(sum(p.numel() for p in model.parameters()))
    report["implementation_sha256"] = gm.implementation_sha256()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=gw.LEARNING_RATE, weight_decay=gw.WEIGHT_DECAY
    )
    flags = gw.check_arm(args.arm)
    needs_state = flags["state_token"] or flags["fusion"]
    sync = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)
    curve, states, losses, step_seconds = [], {}, [], []
    started = time.monotonic()
    log(f"{args.arm}-{args.seed}: {args.updates} updates, {report['parameters']} parameters")
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
        if step % args.select_every == 0:
            value = val_criterion(model, val, val_roots, args.device)
            curve.append([step, value])
            states[step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log(
                f"{args.arm}-{args.seed}: update {step}, val {value:.6f}, "
                f"loss {metrics['loss']:.5f}"
            )
    sync()
    kept = rt.select_checkpoint(curve, gw.SELECTION_TOLERANCE)
    model.load_state_dict(states[kept])
    sec = np.asarray(step_seconds[10:] if len(step_seconds) > 20 else step_seconds)
    report["training"] = {
        "updates": args.updates,
        "select_every": args.select_every,
        "val_curve": curve,
        "kept_update": int(kept),
        "kept_val_criterion": float(dict(curve)[kept]),
        "raw_argmin_update": int(min(curve, key=lambda p: p[1])[0]),
        "last_two_triggered": bool(rt.last_two_triggered(curve, gw.SELECTION_TOLERANCE)),
        "losses": losses,
        "seconds": time.monotonic() - started,
        "seconds_per_update_median": float(np.median(sec)),
        "seconds_per_update_mean": float(np.mean(sec)),
    }
    if args.device == "cuda":
        report["memory"] = devices.memory_report("cuda")
    if args.probe:
        report["budget"] = gw.budget_from_probe(curve, float(np.median(sec)))
        copy = val.root_arrays(val_roots, horizons=(1,))
        from embodied_jepa import grounded_wm_model as gm2

        pred = gm2.predict_at(model, copy[0], copy[1], copy[2], (1,), device=args.device)
        true1 = gw.latent_error(pred[1], copy[3][1])
        copy1 = gw.latent_error(copy[0], copy[3][1])
        report["probe_copy_ratio_h1_val"] = float(true1.sum() / copy1.sum())
    checkpoint = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "arm": args.arm,
        "seed": args.seed,
        "config": model.config,
        "implementation_sha256": report["implementation_sha256"],
        "features_report_sha256": report["features_report_sha256"],
        "kept_update": int(kept),
    }
    torch.save(checkpoint, out / "model.pt")
    report["checkpoint_sha256"] = gd.sha256_file(out / "model.pt")
    report["state_sha256"] = gm.state_sha256(model)
    report["finished_utc"] = utc()
    report["outcome"] = "T-JOB-DONE"
    (out / "report.json").write_text(json.dumps(report, indent=1))
    log(f"{args.arm}-{args.seed}: kept {kept}, done in {report['training']['seconds']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
