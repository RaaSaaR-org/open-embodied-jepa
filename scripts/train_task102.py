"""TASK-102's G_loc training (protocol docs/experiments/so101_hybrid_lift_v1.md §5.4, §7.2).

Subcommands (each refuses an existing output and a dirty tracked tree):

* ``windows`` (CPU, no GPU lock): the corpus pins and TASK-100's feature report and train/val
  files (the test split is refused before any file is opened), §5.4's window rule on every train
  and val episode, the store-corpus alignment guard on every window, and a cache of the window
  frames' raw float16 latents gathered from the store (nothing is featurised again).
* ``train --seed i`` (one GPU job per seed, under ``scripts/gpu_run.sh --wait --board``): G-i,
  30 000 AdamW updates, val MSE every 1 000, selection by the eligibility rule; writes the
  selected checkpoint ``model.pt`` and ``report.json``.
* ``select`` (CPU): the primary G_loc (the lowest selected val MSE) and the replication seeds,
  with every checkpoint's path and sha256, into ``gloc.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from embodied_jepa import so101_hybrid as hy
from embodied_jepa import so101_hybrid_runtime as hr


def refuse_dirty(report: dict) -> None:
    if report["tracked_tree_dirty"]:
        raise hy.GuardError("G-tree: the tracked tree is dirty")


def cmd_windows(args) -> int:
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    report = hr.header("windows")
    refuse_dirty(report)
    hr.check_pins(report)
    shared = hr.shared_root()
    features = shared / hy.FEATURES
    corpus = shared / hy.CORPUS
    t0 = time.monotonic()
    report["corpus_check"] = hr.corpus_check(corpus)
    hr.log("corpus pins ok")
    report["feature_check"] = hr.feature_check(features, hy.TRAIN_SPLITS)
    hr.log("feature files ok")
    out.mkdir(parents=True)
    report["splits"] = {}
    for split in hy.TRAIN_SPLITS:
        report["splits"][split] = hr.build_split(split, features, corpus, out)
        r = report["splits"][split]
        hr.log(f"{split}: {r['windows']} windows, {r['frames']} frames, excluded {r['excluded']}")
    train = {w["seed"] for w in json.loads((out / "windows_train.json").read_text())}
    val = {w["seed"] for w in json.loads((out / "windows_val.json").read_text())}
    if train & val:
        raise hy.GuardError("G-split: train and val windows share an episode")
    report["shared_episodes"] = 0
    report["seconds"] = time.monotonic() - t0
    report["finished_utc"] = hr.utc()
    hr.write_json(out / "windows_report.json", report)
    return 0


def cmd_train(args) -> int:
    import numpy as np
    import torch

    from embodied_jepa import devices
    from embodied_jepa import run_tools as tools
    from embodied_jepa import so101_grasp_policy as gp

    if args.seed not in hy.GLOC_SEEDS:
        raise hy.GuardError(f"G-seed: G_loc seeds are {hy.GLOC_SEEDS}")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    report = hr.header("train", seed=args.seed, windows=args.windows, device=args.device)
    refuse_dirty(report)
    hr.check_pins(report)
    if args.device == "cuda":
        tools.gpu_guard(report, min_free_gib=4.0, require_lock=True)
        report["determinism"] = devices.configure_determinism("cuda", strict=True)
        report["accelerator"] = devices.accelerator_info("cuda")
    out.mkdir(parents=True)
    t0 = time.monotonic()
    windows = Path(args.windows)
    wreport = json.loads((windows / "windows_report.json").read_text())
    report["windows_report_sha256"] = hr.sha256_file(windows / "windows_report.json")
    report["windows_revision"] = wreport["revision"]
    moments = hr.rt.load_moments()
    train = hr.load_split_frames(windows, "train", moments, in_memory=True)
    val = hr.load_split_frames(windows, "val", moments, in_memory=True)
    report["frames"] = {"train": len(train), "val": len(val)}
    report["load_seconds"] = time.monotonic() - t0
    hr.log(f"G-{args.seed}: {len(train)} train frames, {len(val)} val frames loaded")
    input_moments = gp.InputMoments.fit(train.x35)
    result = gp.train(
        args.seed,
        train,
        val,
        input_moments,
        args.device,
        updates=args.updates,
        log=hr.log,
    )
    sel = result["selection"]
    report |= {
        "implementation_sha256": gp.implementation_sha256(),
        "init_seed": result["init_seed"],
        "curve": result["curve"],
        "selection": sel,
        "input_moments": {"mean": input_moments.mean, "std": input_moments.std},
        "updates": args.updates,
        "train_seconds": time.monotonic() - t0,
    }
    if sel["selected"] is not None:
        head = gp.GLocHead()
        head.load_state_dict(result["state_dict"])
        report["checkpoint_sha256"] = gp.save_checkpoint(
            out / "model.pt",
            head,
            input_moments,
            {
                "task": hy.TASK,
                "name": f"G-{args.seed}",
                "seed": args.seed,
                "update": sel["selected"],
                "val_mse": sel["val_mse"],
                "windows_report_sha256": report["windows_report_sha256"],
                "latent_moments_sha256": moments.sha256,
                "revision": report["revision"],
            },
        )
        # §7.3.4's offline reference: the selected policy's batched predictions on val
        policy = gp.load_checkpoint(out / "model.pt", args.device, report["checkpoint_sha256"])
        check = gp.evaluate(policy.head, val, input_moments, args.device)
        report["reloaded_val_mse"] = check["val_mse"]
        if not np.isclose(check["val_mse"], sel["val_mse"], rtol=1e-6, atol=0):
            raise hy.GuardError("G-checkpoint: the reloaded checkpoint's val MSE differs")
    if args.device == "cuda":
        report["cuda_max_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    if hr.tree_dirty():
        raise hy.GuardError("G-tree: the tracked tree became dirty")
    report["seconds"] = time.monotonic() - t0
    report["finished_utc"] = hr.utc()
    hr.write_json(out / "report.json", report)
    hr.log(f"G-{args.seed}: selected {sel}")
    return 0


def cmd_select(args) -> int:
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    selections, checkpoints = {}, {}
    for run in args.runs:
        rep = json.loads((Path(run) / "report.json").read_text())
        name = f"G-{rep['seed']}"
        selections[name] = rep["selection"]
        if rep["selection"]["selected"] is not None:
            path = Path(run).resolve() / "model.pt"
            if hr.sha256_file(path) != rep["checkpoint_sha256"]:
                raise hy.GuardError(f"G-checkpoint: {path} differs from its report")
            checkpoints[name] = {
                "path": str(path),
                "sha256": rep["checkpoint_sha256"],
                "report_sha256": hr.sha256_file(Path(run) / "report.json"),
                "implementation_sha256": rep["implementation_sha256"],
            }
    if sorted(selections) != [f"G-{i}" for i in hy.GLOC_SEEDS]:
        raise hy.GuardError("select needs every G_loc seed's report")
    choice = hy.primary_gloc(selections)
    payload = {
        "task": hy.TASK,
        "primary": choice["primary"],
        "replication": choice["replication"],
        "weak": choice["weak"],
        "selections": selections,
        "checkpoints": checkpoints,
        "written_utc": hr.utc(),
    }
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("windows")
    w.add_argument("--output", required=True)
    t = sub.add_parser("train")
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--windows", required=True)
    t.add_argument("--device", default="cuda")
    t.add_argument("--updates", type=int, default=hy.TRAIN["updates"])
    t.add_argument("--output", required=True)
    s = sub.add_parser("select")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--output", required=True)
    args = p.parse_args(argv)
    return {"windows": cmd_windows, "train": cmd_train, "select": cmd_select}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
