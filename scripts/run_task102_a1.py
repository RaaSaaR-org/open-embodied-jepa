"""TASK-102 A1's runner (``docs/experiments/so101_hybrid_lift_v1.md`` §18.3).

Subcommands (each refuses an existing output):

* ``goals`` -- cohort Q's (or debug's) goal demonstrations and goal store (GPU).
* ``grid``  -- phase 1 alone on Q's resets for the named grid configurations and one cost
  (``--cost pose`` with ``--device none`` on the CPU; ``--cost J`` on a GPU).

Every job records the revision, the tree state and every pin (TASK-100, TASK-101, TASK-102's
CANDIDATE manifest, Δa's files, ``schema_planning.py``) at its start and end, and refuses a seed
outside its cohort. A job on cohort Q needs a clean tracked tree.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from embodied_jepa import so101_closed_loop as cl
from embodied_jepa import so101_hybrid_a1 as a1
from embodied_jepa import so101_hybrid_a1_runtime as r1


def gpu_prelude(report: dict, device: str) -> None:
    from embodied_jepa import devices
    from embodied_jepa import run_tools as tools

    if device == "cuda":
        tools.gpu_guard(report, min_free_gib=4.0, require_lock=True)
        report["determinism"] = devices.configure_determinism("cuda", strict=True)
        report["accelerator"] = devices.accelerator_info("cuda")


def finish(report: dict, out: Path, t0: float, counted: bool) -> None:
    report["wall_seconds"] = time.monotonic() - t0
    end: dict = {}
    r1.check_pins(end, counted=counted)
    report["pins_at_end"] = end
    report["tracked_tree_dirty_at_end"] = r1.tree_dirty()
    report["revision_at_end"] = r1.git("rev-parse", "HEAD")
    if counted and report["revision_at_end"] != report.get("revision"):
        raise a1.GuardError("G-tree: the revision changed during the job")
    report["finished_utc"] = r1.utc()
    r1.dump(out, report)


def load_goals(args) -> dict:
    goals = cl.load_goal_store(args.goals, args.goals_sha256)
    cohort = goals["report"]["cohort"]
    for s in goals["arrays"]["seeds"]:
        a1.check_seed(cohort, int(s))
    return goals | {"cohort": cohort}


def cmd_goals(args) -> int:
    t0 = time.monotonic()
    counted = args.cohort == "Q"
    report = r1.header("goals-job", cohort=args.cohort, device=args.device)
    r1.check_pins(report, counted=counted)
    gpu_prelude(report, args.device)
    report |= r1.build_goal_store(args.cohort, args.device, args.output, limit=args.limit)
    finish(report, Path(args.output) / "job.json", t0, counted)
    return 0


def cmd_grid(args) -> int:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    goals = load_goals(args)
    counted = goals["cohort"] == "Q"
    if (args.cost == "pose") != (args.device == "none"):
        raise a1.GuardError("O-pose runs with --device none, J on a GPU")
    if counted and args.debug_timeout:
        raise a1.GuardError("a shortened timeout is for debug seeds only")
    configs = list(args.configs)
    for name in configs:
        a1.check_config(name)
        if name.endswith(a1.BIG_SUFFIX) and args.cost != "J":
            raise a1.GuardError("the larger budget is for J only (§18.3 step 3)")
    report = r1.header(
        "grid",
        goals_sha256=goals["sha256"],
        cohort=goals["cohort"],
        cost=args.cost,
        configs=configs,
        settings={n: a1.check_config(n) for n in configs},
        workers=args.workers,
        device=args.device,
        tau_h=a1.TAU_H,
    )
    r1.check_pins(report, counted=counted)
    gpu_prelude(report, args.device)
    seeds = [int(s) for s in goals["arrays"]["seeds"]]
    n = min(a1.Q_RESETS, len(seeds))
    if counted and n != a1.Q_RESETS:
        raise a1.GuardError("G-reserve: Q needs 8 kept resets")
    jobs = []
    for name in configs:
        (out / name).mkdir()
        jobs += [
            {
                "config": name,
                "cost": args.cost,
                "seed": seeds[i],
                "slot": i,
                "output": str(out / name),
                "timeout": args.debug_timeout or a1.T1,
            }
            for i in range(n)
        ]
    progress = out / "progress.jsonl"

    def on_result(r):
        with progress.open("a") as f:
            f.write(json.dumps(r, default=cl._json_default) + "\n")
        r1.log(json.dumps({k: r.get(k) for k in ("config", "cost", "seed", "handover",
                                                  "phase1_commands", "closest_arm_rms",
                                                  "episode_seconds")}))  # fmt: skip

    report["started_jobs_utc"] = r1.utc()
    report["results"] = r1.run_pool(
        jobs,
        goals_dir=args.goals,
        goals_sha256=goals["sha256"],
        device=args.device,
        workers=args.workers,
        on_result=on_result,
    )
    finish(report, out / "report.json", t0, counted)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("goals")
    g.add_argument("--cohort", required=True, choices=sorted(a1.COHORTS))
    g.add_argument("--output", required=True)
    g.add_argument("--device", default="cuda")
    g.add_argument("--limit", type=int)
    s = sub.add_parser("grid")
    s.add_argument("--goals", required=True)
    s.add_argument("--goals-sha256", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--cost", required=True, choices=a1.COSTS)
    s.add_argument("--configs", nargs="+", required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--debug-timeout", type=int, help="debug seeds only (timing)")
    args = p.parse_args(argv)
    return {"goals": cmd_goals, "grid": cmd_grid}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
