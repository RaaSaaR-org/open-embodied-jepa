"""TASK-102 Δa's runner (``docs/experiments/so101_approach_diagnostic_v1.md``).

Subcommands (each refuses an existing output):

* ``goals``     -- cohort L's (or debug's) goal demonstrations and goal store (GPU).
* ``lambda0``   -- §2: Stage 0's A logs re-read (CPU, no lock, no simulation).
* ``landscape`` -- §3 (Λ1, Λ3): every kept slot's landscape in a pool of workers (GPU).
* ``approach``  -- §4 (Λ2): phase 1 alone on L's slots 0-7 for the named arms (GPU for J-* and
  O-img; ``--device none`` for O-pose, which renders nothing).

Every job records the revision, the tree state, TASK-100's frozen-files check and TASK-101's and
TASK-102's pins (unchanged files), and refuses a seed outside its cohort.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from embodied_jepa import so101_approach_diag as ad
from embodied_jepa import so101_approach_diag_runtime as ar
from embodied_jepa import so101_closed_loop as cl
from embodied_jepa import so101_hybrid_runtime as hr


def gpu_prelude(report: dict, device: str) -> None:
    from embodied_jepa import devices
    from embodied_jepa import run_tools as tools

    if device == "cuda":
        tools.gpu_guard(report, min_free_gib=4.0, require_lock=True)
        report["determinism"] = devices.configure_determinism("cuda", strict=True)
        report["accelerator"] = devices.accelerator_info("cuda")


def finish(report: dict, out: Path, t0: float) -> None:
    report["wall_seconds"] = time.monotonic() - t0
    hr.check_pins(report)
    report["tracked_tree_dirty_at_end"] = ar.tree_dirty()
    report["revision_at_end"] = ar.git("rev-parse", "HEAD")
    report["finished_utc"] = ar.utc()
    ar.dump(out, report)


def load_goals(args) -> dict:
    goals = cl.load_goal_store(args.goals, args.goals_sha256)
    cohort = goals["report"]["cohort"]
    for s in goals["arrays"]["seeds"]:
        ad.check_seed(cohort, int(s))
    return goals | {"cohort": cohort}


def cmd_goals(args) -> int:
    t0 = time.monotonic()
    report = ar.header("goals-job", cohort=args.cohort, device=args.device)
    hr.check_pins(report)
    gpu_prelude(report, args.device)
    report |= ar.build_goal_store(args.cohort, args.device, args.output, limit=args.limit)
    finish(report, Path(args.output) / "job.json", t0)
    return 0


def cmd_lambda0(args) -> int:
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    t0 = time.monotonic()
    report = ar.header("lambda0", stage0=args.stage0)
    hr.check_pins(report)
    report["lambda0"] = ar.lambda0(args.stage0)
    finish(report, out, t0)
    return 0


def _slots(text, n) -> list[int]:
    if not text:
        return list(range(n))
    lo, hi = (int(x) for x in text.split("-"))
    return list(range(lo, hi + 1))


def _pool(args, jobs, out: Path, report: dict, goals: dict) -> None:
    progress = out / "progress.jsonl"

    def on_result(r):
        with progress.open("a") as f:
            f.write(json.dumps(r, default=cl._json_default) + "\n")
        ar.log(json.dumps({k: r.get(k) for k in ("arm", "seed", "termination", "handover",
                                                  "closest_arm_rms", "commands", "seconds",
                                                  "episode_seconds")}))  # fmt: skip

    report["started_jobs_utc"] = ar.utc()
    report["results"] = ar.run_pool(
        jobs,
        goals_dir=args.goals,
        goals_sha256=goals["sha256"],
        device=args.device,
        workers=args.workers,
        on_result=on_result,
    )


def cmd_landscape(args) -> int:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    goals = load_goals(args)
    report = ar.header("landscape", goals_sha256=goals["sha256"], cohort=goals["cohort"],
                       workers=args.workers, device=args.device)  # fmt: skip
    hr.check_pins(report)
    gpu_prelude(report, args.device)
    seeds = [int(s) for s in goals["arrays"]["seeds"]]
    slots = _slots(args.slots, len(seeds))
    jobs = [{"kind": "landscape", "seed": seeds[i], "slot": i, "output": str(out)} for i in slots]
    _pool(args, jobs, out, report, goals)
    finish(report, out / "report.json", t0)
    return 0


def cmd_approach(args) -> int:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    goals = load_goals(args)
    for arm in args.arms:
        cfg = ad.check_arm(arm)
        if (cfg["cost"] == "pose") != (args.device == "none"):
            raise ad.GuardError("O-pose runs with --device none, every other arm on a GPU")
    report = ar.header("approach", goals_sha256=goals["sha256"], cohort=goals["cohort"],
                       arms=args.arms, workers=args.workers, device=args.device,
                       arm_settings={a: ad.ARMS[a] for a in args.arms})  # fmt: skip
    hr.check_pins(report)
    gpu_prelude(report, args.device)
    seeds = [int(s) for s in goals["arrays"]["seeds"]]
    slots = _slots(args.slots, min(ad.L2_SLOTS, len(seeds)))
    jobs = []
    for arm in args.arms:
        (out / arm).mkdir()
        jobs += [
            {
                "kind": "approach",
                "arm": arm,
                "seed": seeds[i],
                "slot": i,
                "output": str(out / arm),
                "timeout": args.debug_timeout or ar.hy.T1,
            }
            for i in slots
        ]
    _pool(args, jobs, out, report, goals)
    finish(report, out / "report.json", t0)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("goals")
    g.add_argument("--cohort", required=True, choices=sorted(ad.COHORTS))
    g.add_argument("--output", required=True)
    g.add_argument("--device", default="cuda")
    g.add_argument("--limit", type=int)
    z = sub.add_parser("lambda0")
    z.add_argument("--stage0", required=True)
    z.add_argument("--output", required=True)
    for name in ("landscape", "approach"):
        s = sub.add_parser(name)
        s.add_argument("--goals", required=True)
        s.add_argument("--goals-sha256", required=True)
        s.add_argument("--output", required=True)
        s.add_argument("--device", default="cuda")
        s.add_argument("--workers", type=int, default=6)
        s.add_argument("--slots")
        if name == "approach":
            s.add_argument("--arms", nargs="+", required=True, choices=sorted(ad.ARMS))
            s.add_argument("--debug-timeout", type=int, help="debug seeds only (timing)")
    args = p.parse_args(argv)
    return {
        "goals": cmd_goals,
        "lambda0": cmd_lambda0,
        "landscape": cmd_landscape,
        "approach": cmd_approach,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
