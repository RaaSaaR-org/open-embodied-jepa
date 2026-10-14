"""TASK-092 Stage T: train one G-req model (one seed) on ``play-v1`` train with §4.1's grasp-rate
augmentation; select on ``play-v1`` val under the executed commands (protocol
docs/experiments/grasp_rate_aware_wm_v1.md §4.2, §6.1).

TASK-087's model, sampler (salt 8714), initialisation (``torch.manual_seed(m)``), optimiser,
budget, selection rule and val criterion, unchanged; the train store is TASK-087's, held in memory;
the moments are ``play-v1`` train's, checked bitwise against G-87100's buffers. The predictor's
grasp input is augmented with p = 0.5 (salt 9008, its own generator); the targets are the recorded
ones. One invocation = one job (one ``scripts/gpu_run.sh`` lock). Writes ``<output>/model.pt`` and
``<output>/report.json``; refuses an existing output.

Kinds: ``job`` (Stage T: a model seed, p = 0.5, U = 60 000, selection every 3 000, the 7 200 s
cap, a clean tree); ``--pairing`` (Stage 0, §13: model seed 87100 with p = 0 up to the first
selection point, 3 000 updates; its val criterion must equal G-87100's recorded one; no
checkpoint); ``--debug`` (Stage 0: TASK-087's debug seeds, any p, U and selection interval).
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

from embodied_jepa import grasp_corpus_wm as gc  # noqa: E402
from embodied_jepa import grasp_rate_aware as ra  # noqa: E402
from embodied_jepa import grounded_wm as gw  # noqa: E402
from embodied_jepa import grounded_wm_data as gd  # noqa: E402
from embodied_jepa import run_tools as rt  # noqa: E402

UPDATES = 60_000  # TASK-087's U (§4.2)
SELECT_EVERY = 3_000


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc()}] {msg}", flush=True)


def git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def val_criterion(model, val, roots, device) -> float:
    """TASK-087 §6.2 (``train_task087.val_criterion``, copied): the mean latent error of the
    recursive roll-out at steps 1-8 from every val root, under the executed commands."""
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


def check_kind(args) -> str:
    if args.pairing and args.debug:
        raise SystemExit("--pairing and --debug are exclusive")
    if args.pairing:
        if (args.seed, args.p) != (ra.HISTORICAL_PAIRING_SEED, 0.0):
            raise SystemExit("the historical pairing check is seed 87100 with p = 0 (§13)")
        if (args.updates, args.select_every) != (ra.HISTORICAL_PAIRING_UPDATES, SELECT_EVERY):
            raise SystemExit("the historical pairing check runs to the first selection point")
        return "pairing"
    if args.debug:
        ra.check_model_seed(args.seed, debug=True)
        return "debug"
    ra.check_model_seed(args.seed)
    if (args.updates, args.select_every, args.p) != (UPDATES, SELECT_EVERY, ra.AUG_P):
        raise SystemExit("Stage T uses U = 60 000, selection every 3 000 and p = 0.5 (§4)")
    if args.cap_seconds != ra.JOB_CAP_SECONDS or args.mmap:
        raise SystemExit("Stage T uses the 7 200 s cap and the in-memory store (§4.2, §6.1)")
    if args.time_only:
        raise SystemExit("--time-only is debug-only")
    return "job"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--select-every", type=int, default=SELECT_EVERY)
    parser.add_argument("--p", type=float, default=ra.AUG_P)
    parser.add_argument("--cap-seconds", type=float, default=float(ra.JOB_CAP_SECONDS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--mmap", action="store_true", help="debug: memory-map the train store")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pairing", action="store_true")
    parser.add_argument("--time-only", action="store_true", help="debug: no selection")
    args = parser.parse_args(argv)
    kind = check_kind(args)
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    if args.updates % args.select_every:
        raise SystemExit("updates must be a multiple of select_every")
    import torch

    from embodied_jepa import devices
    from embodied_jepa import grasp_corpus_wm_runtime as gr
    from embodied_jepa import grasp_rate_aware_runtime as rr
    from embodied_jepa import grounded_wm_model as gm

    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    if kind == "job" and dirty:
        raise rt.GuardError("G-clean: Stage T runs from a clean tracked tree")
    frozen = ra.frozen_files_check(ROOT)
    if not all(frozen.values()):
        raise rt.GuardError(f"G-frozen: {[k for k, v in frozen.items() if not v]}")
    report = {
        "task": ra.TASK,
        "protocol": ra.PROTOCOL,
        "kind": kind,
        "started_utc": utc(),
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": dirty,
        "argv": sys.argv,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "arm": "G",
        "seed": args.seed,
        "p": args.p,
        "augment_salt": ra.SALTS["augment"],
        "rate_limit": ra.RATE_LIMIT,
        "at_limit_tol": ra.AT_LIMIT_TOL,
        "mmap": bool(args.mmap),
        "frozen_files_match": True,
        "modules_sha256": ra.own_modules_sha256(ROOT),
    }
    rt.assert_local_import(ROOT, report)
    if args.device == "cuda":
        rt.gpu_guard(report, min_free_gib=4.0, require_lock=True)
        report["determinism"] = devices.configure_determinism("cuda", strict=True)
        report["accelerator"] = devices.accelerator_info("cuda")
    out.mkdir(parents=True)
    (out / "started.json").write_text(json.dumps({"started_utc": report["started_utc"]}))

    feat = gr.V1_FEATURES
    report["features_report_sha256"] = gd.sha256_file(feat / "report.json")
    if report["features_report_sha256"] != ra.FEATURE_REPORT_SHA256:
        raise rt.GuardError("G-features: play-v1's feature report differs from TASK-087's")
    feat_report = json.loads((feat / "report.json").read_text())
    t0 = time.monotonic()
    report["store_files_verified"] = gd.verify_store_files(
        feat, feat_report["files_sha256"], ("train", "val")
    )
    report["verify_seconds"] = time.monotonic() - t0
    t0 = time.monotonic()
    train = gd.FeatureStore(feat, "train", mmap=args.mmap)
    val = gd.FeatureStore(feat, "val", mmap=args.mmap)
    report["load_seconds"] = time.monotonic() - t0
    val_roots, _ = val.roots()
    state_mean, state_std, palm_mean, palm_std = train.moments()
    sampler = gd.WindowSampler(train, args.seed)
    augment = ra.Augmenter(args.seed, args.p)
    report["data"] = {
        "train_frames": int(len(train.state)),
        "train_episodes": len(train.table),
        "train_windows": int(len(sampler.starts)),
        "val_roots": int(len(val_roots)),
    }
    torch.manual_seed(args.seed)
    model = rr.GReqWM("G").to(args.device)
    model.set_moments(state_mean, state_std, palm_mean, palm_std)
    buffers = gr.model_buffers(model)
    ref = gr.checkpoint_buffers(gr.V1_CHECKPOINTS / gr.PIN_MODEL)
    if not gc.same_buffers(buffers, ref) or gc.buffers_sha256(buffers) != ra.MOMENTS_SHA256:
        raise rt.GuardError("G-moments: play-v1 train's moments differ from G-87100's buffers")
    report["moments"] = {
        "sha256": gc.buffers_sha256(buffers),
        "g87100_sha256": gc.buffers_sha256(ref),
        "bitwise_equal_g87100": True,
    }
    report["parameters"] = int(sum(p.numel() for p in model.parameters()))
    report["implementation_sha256"] = gm.implementation_sha256()
    if report["implementation_sha256"] != ra.IMPLEMENTATION_SHA256:
        raise rt.GuardError("G-implementation: TASK-087's code changed")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=gw.LEARNING_RATE, weight_decay=gw.WEIGHT_DECAY
    )
    sync = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)
    curve, states, losses, step_seconds = [], {}, [], []
    started = time.monotonic()
    log(f"G-req-{args.seed} ({kind}): {args.updates} updates, p = {args.p}")
    for step in range(1, args.updates + 1):
        t0 = time.perf_counter()
        starts = sampler.draw()
        batch = train.gather(starts, gw.WINDOW_FRAMES)
        prev = ra.window_prev(starts, train.actions, train.offsets)
        a_in = augment(batch["actions"], prev)
        z = torch.as_tensor(batch["latents"], device=args.device)
        a = torch.as_tensor(a_in, device=args.device)
        a_exec = torch.as_tensor(batch["actions"], device=args.device)
        s = torch.as_tensor(batch["state"], device=args.device)
        model.train()
        loss, metrics = model.loss(z, a, s, None, a_exec)
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
            log(f"update {step}, val {value!r}, loss {metrics['loss']:.5f}")
    sync()
    sec = np.asarray(step_seconds[10:] if len(step_seconds) > 20 else step_seconds)
    timing = {
        "seconds": time.monotonic() - started,
        "seconds_per_update_median": float(np.median(sec)),
        "seconds_per_update_mean": float(np.mean(sec)),
        "seconds_per_update_p90": float(np.percentile(sec, 90)),
    }
    report["augmentation"] = dict(augment.counts)
    if args.device == "cuda":
        report["memory"] = devices.memory_report("cuda")
    if args.time_only:
        report["training"] = {"updates": args.updates, "time_only": True, **timing}
        report["finished_utc"] = utc()
        report["outcome"] = "T-TIMED"
        (out / "report.json").write_text(json.dumps(report, indent=1))
        return 0
    if kind == "pairing":
        recorded = rr.g87100_report()
        want = recorded["training"]["val_curve"][0]
        got = curve[0]
        report["training"] = {"updates": args.updates, "val_curve": curve, **timing}
        report["pairing"] = {
            "g87100_report_sha256": gd.sha256_file(gr.V1_CHECKPOINTS / "G-87100/report.json"),
            "recorded_val_at_3000": want,
            "this_val_at_3000": got,
            "bitwise_equal": bool(
                int(want[0]) == int(got[0])
                and np.float64(want[1]).tobytes() == np.float64(got[1]).tobytes()
            ),
            "state_sha256_at_3000": gm.state_sha256(model),
        }
        report["finished_utc"] = utc()
        report["outcome"] = "T-PAIRING-DONE"
        (out / "report.json").write_text(json.dumps(report, indent=1))
        log(f"pairing: recorded {want}, this {got}, bitwise {report['pairing']['bitwise_equal']}")
        return 0 if report["pairing"]["bitwise_equal"] else 1
    kept = rt.select_checkpoint(curve, gw.SELECTION_TOLERANCE)
    model.load_state_dict(states[kept])
    if gc.buffers_sha256(gr.model_buffers(model)) != ra.MOMENTS_SHA256:
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
    checkpoint = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "arm": "G",
        "family": "G-req",
        "seed": args.seed,
        "config": model.config,
        "implementation_sha256": report["implementation_sha256"],
        "modules_sha256": report["modules_sha256"],
        "features_report_sha256": report["features_report_sha256"],
        "p": args.p,
        "kept_update": int(kept),
    }
    torch.save(checkpoint, out / "model.pt")
    report["checkpoint_sha256"] = gd.sha256_file(out / "model.pt")
    report["state_sha256"] = gm.state_sha256(model)
    report["finished_utc"] = utc()
    report["outcome"] = "T-JOB-DONE"
    (out / "report.json").write_text(json.dumps(report, indent=1))
    log(f"G-req-{args.seed}: kept {kept}, last_two {report['training']['last_two_triggered']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
