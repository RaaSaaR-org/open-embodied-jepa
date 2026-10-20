"""TASK-102 A1's summary (``docs/experiments/so101_hybrid_lift_v1.md`` §18.3-§18.4): the grid's
readings per configuration and cost from the jobs' reports, the GPU hours from the lock log, the
selection and the row. With ``--partial`` it summarises what has run (for deciding step 2 and
step 3) without the row's completeness checks.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

from embodied_jepa import so101_closed_loop as cl
from embodied_jepa import so101_hybrid_a1 as a1

GPU_LOG = Path.home() / ".local/state/gpu/oej-gpu_run.log"
WHO = "oej:task102-a1"


def _num(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def gpu_hours(log=GPU_LOG) -> dict:
    jobs, seconds = [], 0
    if Path(log).exists():
        for line in Path(log).read_text().splitlines():
            pattern = rf"(\S+) end: ({re.escape(WHO)}) pid (\d+) status (\d+) after (\d+) s"
            m = re.match(pattern, line)
            if m:
                jobs.append({"utc": m[1], "who": m[2], "status": int(m[4]), "seconds": int(m[5])})
                seconds += int(m[5])
    return {"jobs": jobs, "count": len(jobs), "gpu_hours": seconds / 3600.0}


def _stats(rows: list[dict]) -> dict:
    h = sum(1 for r in rows if r["by_tolerance"])
    n = len(rows)
    closest = [r["closest_arm_rms"] for r in rows]
    cmds = [r["phase1_commands"] if r["by_tolerance"] else a1.T1 for r in rows]
    ho = [r for r in rows if r["by_tolerance"]]
    spc = [r["seconds_per_command"] for r in rows if r["seconds_per_command"] is not None]
    return {
        "episodes": n,
        "h": h,
        "interval": cl.exact_interval(h, n) if n else None,
        "closest_median": float(np.median(closest)),
        "closest_min": float(np.min(closest)),
        "closest_max": float(np.max(closest)),
        "commands_median": float(np.median(cmds)),
        "handover_arm_rms_median": _num(np.median([r["handover_arm_rms"] for r in ho]))
        if ho
        else None,
        "handover_jaw_error_median": _num(np.median([r["handover_jaw_error"] for r in ho]))
        if ho
        else None,
        "handover_jaw_within_0.15": sum(1 for r in ho if abs(r["handover_jaw_error"]) <= 0.15),
        "jaw_error_at_closest_median": float(np.median([r["jaw_error_at_closest"] for r in rows])),
        "stopped": sum(1 for r in rows if str(r["termination"]).startswith("stopped")),
        "seconds_per_command_median": float(np.median(spc)) if spc else None,
        "plan_seconds_median": _num(
            np.median([r["plan_seconds_median"] for r in rows if r["plan_seconds_median"]])
        ),
        "rollout_checks": sum(r["rollout_checks"] for r in rows),
        "oracle_over_velocity": sum(r["oracle_over_velocity"] for r in rows),
        "per_seed": {
            str(r["seed"]): {
                "handover": r["handover"],
                "commands": r["phase1_commands"],
                "closest": r["closest_arm_rms"],
            }
            for r in rows
        },
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--partial", action="store_true")
    args = p.parse_args(argv)
    root = Path(args.root)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    goals = json.loads((root / "Q-goals" / "goals_report.json").read_text())
    q_sha = goals["goals_sha256"]
    seen: dict = {}
    jobs, revisions, void = [], set(), []
    for rep_path in sorted(root.glob("*/report.json")):
        rep = json.loads(rep_path.read_text())
        if rep.get("kind") != "grid" or rep.get("cohort") != "Q":
            continue
        if rep["goals_sha256"] != q_sha:
            void.append(f"{rep_path.parent.name}: another goal store")
        revisions.add(rep["revision"])
        if rep.get("tracked_tree_dirty") or rep.get("tracked_tree_dirty_at_end"):
            void.append(f"{rep_path.parent.name}: dirty tree")
        if not rep.get("a1_pins_ok") or rep["pins_at_end"].get("a1_pins_ok") is not True:
            void.append(f"{rep_path.parent.name}: pins")
        jobs.append({"job": rep_path.parent.name, "cost": rep["cost"], "configs": rep["configs"],
                     "revision": rep["revision"], "wall_seconds": rep["wall_seconds"]})  # fmt: skip
        for r in rep["results"]:
            key = (r["cost"], r["config"])
            seen.setdefault(key, []).append(r)
    if len(revisions) > 1:
        void.append(f"jobs at several revisions: {sorted(revisions)}")
    for (cost, name), rows in seen.items():
        if len(rows) != a1.Q_RESETS or len({r["seed"] for r in rows}) != a1.Q_RESETS:
            void.append(f"{cost} {name}: {len(rows)} episodes (a cohort run twice or incomplete)")
    stats = {f"{c}/{n}": _stats(rows) for (c, n), rows in sorted(seen.items())}
    gpu = gpu_hours()
    if gpu["gpu_hours"] > a1.GRID_CAP_GPU_H:
        void.append(f"GPU {gpu['gpu_hours']:.3f} h above the cap")
    if any(j["status"] for j in gpu["jobs"]):
        void.append("a GPU job exited non-zero")
    pose = {n: stats[f"pose/{n}"]["h"] for n in a1.CONFIGS if f"pose/{n}" in stats}
    result: dict = {"goals_sha256": q_sha, "kept": goals["kept"], "rejected": goals["rejected"],
                    "revisions": sorted(revisions), "jobs": jobs, "gpu": gpu, "void": void,
                    "stats": stats}  # fmt: skip
    if len(pose) == len(a1.CONFIGS):
        admitted = a1.admitted(pose)
        result["admitted"] = admitted
        j = {
            n: {k: stats[f"J/{n}"][k] for k in ("h", "commands_median", "closest_median")}
            for n in admitted + [x + a1.BIG_SUFFIX for x in admitted]
            if f"J/{n}" in stats
        }
        missing = [n for n in admitted if n not in j]
        result["j_missing"] = missing
        if not missing:
            result["needs_larger_budget"] = a1.needs_larger_budget(pose, j)
            big = None
            if result["needs_larger_budget"]:
                bn = result["needs_larger_budget"] + a1.BIG_SUFFIX
                if bn in j:
                    big = {"config": bn, "h": j[bn]["h"]}
            try:
                result["row"] = a1.row(void=bool(void), pose_counts=pose, j=j, big=big)
            except Exception as error:  # noqa: BLE001 -- step 3 not run yet
                result["row"] = {"pending": str(error)}
        elif not admitted:
            result["row"] = a1.row(void=bool(void), pose_counts=pose, j={})
    elif not args.partial:
        raise SystemExit(f"O-pose ran on {len(pose)} of {len(a1.CONFIGS)} configurations")
    out.write_text(json.dumps(result, indent=2, default=cl._json_default) + "\n")
    brief = {"row": result.get("row"), "admitted": result.get("admitted"), "void": void,
             "gpu_hours": gpu["gpu_hours"],
             "h": {k: v["h"] for k, v in stats.items()}}  # fmt: skip
    print(json.dumps(brief, default=cl._json_default))
    return 0


if __name__ == "__main__":
    sys.exit(main())
