"""TASK-100: train one arm and seed on a feature store (protocol §5, §6 G6, §8.1, §10.4, §10.6).

One invocation = one job (one ``scripts/gpu_run.sh`` lock). Writes ``<output>/model.pt`` (the
kept checkpoint), ``final.pt`` (the final full state: weights, optimiser, RNG), every selection
point's weights (``selection_states.pt``), the sampler state and ``report.json``; refuses an
existing output.

Kinds:
  job          a gated job (model seed): trains on ``train``, selects on ``val``
  probe        §8.1's budget probe (L, seed 100920, 60 000 updates, select every 2 000 on val)
  calibration  §10.4's calibration L / N_L (seeds 100900 / 100911, 20 000 updates on D0-fit;
               keeps the final state; the D0-eval criterion curve is recorded, not used)
  debug        §10.6's smoke (debug seeds; any train and selection views)
``--resume <job dir> --extend-to N`` is G6's one extension (Stage T'): the job's final state and
sampler stream continue to N updates, and selection is redone over the whole curve.
"""

from __future__ import annotations

import argparse
import json
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

KIND_VIEWS = {
    "job": ("train", "val"),
    "probe": ("train", "val"),
    "calibration": ("d0-fit", "d0-eval"),
}


def check_kind(args) -> None:
    development = args.kind != "job"
    sw.check_seed(args.seed, development=development)
    if args.kind == "probe" and (args.arm != "L" or args.seed != sw.PROBE_SEED):
        raise SystemExit("the probe is L with seed 100920")
    if args.kind == "calibration":
        want = {"L": sw.CAL_L_SEED, "N_L": sw.CAL_N_SEED}
        if want.get(args.arm) != args.seed or args.updates != sw.CAL_UPDATES:
            raise SystemExit("calibration is L-100900 or N_L-100911 for 20 000 updates")
    if args.kind == "debug" and args.seed not in sw.DEBUG_SEEDS:
        raise SystemExit("debug jobs use the debug seeds 100901-100910")
    if args.kind in KIND_VIEWS and (args.train_view, args.select_view) != KIND_VIEWS[args.kind]:
        raise SystemExit(
            f"{args.kind} trains on {KIND_VIEWS[args.kind][0]} and selects on "
            f"{KIND_VIEWS[args.kind][1]}"
        )
    if args.updates % args.select_every:
        raise SystemExit("updates must be a multiple of select_every")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", required=True, help="the arm's encoder's feature folder")
    parser.add_argument("--grid", required=True, choices=sorted(sw.GRIDS))
    parser.add_argument("--arm", required=True, choices=sw.ARMS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--kind", required=True, choices=("job", "probe", "calibration", "debug"))
    parser.add_argument("--train-view", default="train")
    parser.add_argument("--select-view", default="val")
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--select-every", type=int, required=True)
    parser.add_argument("--cap-seconds", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None, help="G6's extension: the job to continue")
    parser.add_argument("--extend-to", type=int, default=None)
    parser.add_argument("--expect-fit-report-sha256", default=None)
    parser.add_argument("--used-gpu-h", type=float, default=None, help="probe: GPU h used before B")
    args = parser.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    if out.name.endswith("-r2"):
        first = out.with_name(out.name[: -len("-r2")])
        if (first / "report.json").exists() or not (first / "started.json").exists():
            raise SystemExit("an -r2 run is allowed only after an incomplete first run")
    check_kind(args)
    strict = args.kind in ("job", "probe")
    if strict and args.grid != sw.CHOSEN_GRID:
        raise SystemExit(f"gated jobs use the frozen grid {sw.CHOSEN_GRID}")
    import torch

    from embodied_jepa import devices
    from embodied_jepa.models import schema_frozen_tokens as sft

    report = runtime.header(args.kind, torch=torch.__version__, arm=args.arm, seed=args.seed)
    report |= {"grid": args.grid, "train_view": args.train_view, "select_view": args.select_view}
    runtime.require_clean(report, strict)
    if args.device == "cuda":
        rt.gpu_guard(report, min_free_gib=4.0, require_lock=True)
        report["determinism"] = devices.configure_determinism("cuda", strict=True)
        report["accelerator"] = devices.accelerator_info("cuda")
    spec = sw.check_arm(args.arm)
    feat = Path(args.features)
    moments, fit = runtime.load_fit(feat, args.expect_fit_report_sha256)
    report["fit_report_sha256"] = fit["report_sha256"]
    report["moments_sha256"] = moments.sha256
    if fit["grid"] != args.grid or fit["encoder"] != spec["encoder"]:
        raise rt.GuardError("G-features: the feature folder's grid or encoder is not the arm's")
    splits = {sd.VIEWS[v][0] for v in (args.train_view, args.select_view)}
    report["store_files_verified"] = runtime.verify_features(feat, sorted(splits))
    train = sd.open_view(feat, args.train_view, moments)
    select = sd.open_view(feat, args.select_view, moments)
    sampler = sd.WindowSampler(train.window_starts(), sw.sampler_stream_for(args.arm, args.seed))
    report["data"] = {
        "train_episodes": int(len(train.episodes)),
        "train_windows": int(len(sampler.starts)),
        "select_roots": int(len(select.roots()[0])),
        "sampler_stream": sw.sampler_stream_for(args.arm, args.seed),
    }
    metadata = {
        "task": sw.TASK,
        "arm": args.arm,
        "grid": args.grid,
        "encoder": spec["encoder"],
        "feature_fit_report_sha256": fit["report_sha256"],
        "moments_sha256": moments.sha256,
    }
    model = runtime.build_model(args.arm, args.seed, args.grid, metadata, args.device)
    curve, states, losses, step_seconds = [], {}, [], []
    start_update = 0
    total = args.updates
    if args.resume:
        prior_dir = Path(args.resume)
        prior = json.loads((prior_dir / "report.json").read_text())
        if (prior["arm"], prior["seed"], prior["grid"]) != (args.arm, args.seed, args.grid):
            raise SystemExit("the extension must continue the same arm, seed and grid")
        if args.extend_to is None or args.extend_to <= prior["training"]["updates"]:
            raise SystemExit("--extend-to must exceed the job's updates")
        model.load(prior_dir / "final.pt")
        sampler.restore(prior["sampler_state"])
        curve = [list(p) for p in prior["training"]["val_curve"]]
        states = torch.load(prior_dir / "selection_states.pt", weights_only=True)
        start_update = int(prior["training"]["updates"])
        total = int(args.extend_to)
        report["resumed_from"] = {
            "job": str(prior_dir),
            "report_sha256": sd.sha256_file(prior_dir / "report.json"),
        }
        if (total - start_update) % args.select_every:
            raise SystemExit("the extension must be a multiple of select_every")
    report["metadata"] = model.metadata
    report["parameters"] = sft.parameter_count(model)
    report["implementation_sha256"] = model.implementation_sha256
    report["config"] = model.config
    out.mkdir(parents=True)
    (out / "started.json").write_text(json.dumps({"started_utc": report["started_utc"]}))
    sync = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)
    zero = spec["zero_actions"]
    started = time.monotonic()
    runtime.log(
        f"{args.arm}-{args.seed}: updates {start_update}..{total}, "
        f"{report['parameters']} parameters"
    )
    for step in range(start_update + 1, total + 1):
        t0 = time.perf_counter()
        batch = train.store.gather(sampler.draw())
        actions = np.zeros_like(batch["actions"]) if zero else batch["actions"]
        metrics = model.train_step_features(batch["latents"], actions)
        if step % 100 == 0:
            if not np.isfinite(metrics["loss"]):
                raise rt.GuardError("G-finite: non-finite training loss")
            losses.append([step, {k: float(v) for k, v in metrics.items()}])
            if time.monotonic() - started > args.cap_seconds:
                raise rt.GuardError(f"G-cap: exceeded {args.cap_seconds} s")
        step_seconds.append(time.perf_counter() - t0)
        if step % args.select_every == 0:
            sync()
            value = runtime.criterion(model, select, args.arm)
            curve.append([step, value])
            states[step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            runtime.log(
                f"{args.arm}-{args.seed}: update {step}, criterion {value:.6f}, "
                f"loss {metrics['loss']:.5f}"
            )
    sync()
    model.save(out / "final.pt")
    torch.save(states, out / "selection_states.pt")
    final_state = sft.state_sha256(model)
    if args.kind == "calibration":
        kept = total  # §10.4: Stage 0 does not open val; the final state is kept (disclosed)
    else:
        kept = rt.select_checkpoint(curve, sw.SELECTION_TOLERANCE)
    model.load_state_dict(states[kept])
    model.save(out / "model.pt")
    sec = np.asarray(step_seconds[10:] if len(step_seconds) > 20 else step_seconds)
    report["sampler_state"] = sampler.state()
    report["training"] = {
        "updates": int(total),
        "select_every": int(args.select_every),
        "val_curve": curve,
        "kept_update": int(kept),
        "kept_criterion": float(dict((int(u), v) for u, v in curve)[kept]),
        "raw_argmin_update": int(min(curve, key=lambda p: p[1])[0]),
        "last_two_triggered": bool(rt.last_two_triggered(curve, sw.SELECTION_TOLERANCE)),
        "losses": losses,
        "seconds": time.monotonic() - started,
        "seconds_per_update_median": float(np.median(sec)),
        "seconds_per_update_mean": float(np.mean(sec)),
    }
    if len(curve) >= 2:
        try:
            report["training"]["g6"] = sw.saturation(curve)
        except Exception as error:  # noqa: BLE001 -- recorded, decided at scoring
            report["training"]["g6"] = {"error": str(error)}
    if args.device == "cuda":
        report["memory"] = devices.memory_report("cuda")
    if args.kind == "probe":
        if args.used_gpu_h is None:
            raise SystemExit("the probe needs --used-gpu-h (the ledger before Stage B)")
        report["budget"] = sw.budget_from_probe(
            curve,
            float(np.median(sec)),
            used_gpu_h=args.used_gpu_h + report["training"]["seconds"] / 3600,
        )
    report["checkpoint_sha256"] = sd.sha256_file(out / "model.pt")
    report["final_sha256"] = sd.sha256_file(out / "final.pt")
    report["state_sha256"] = sft.state_sha256(model)
    report["final_state_sha256"] = final_state
    runtime.require_clean(report, strict)
    report["finished_utc"] = runtime.utc()
    report["outcome"] = "T-JOB-DONE"
    runtime.write_json(out / "report.json", report)
    runtime.log(
        f"{args.arm}-{args.seed}: kept {kept}, done in {report['training']['seconds']:.0f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
