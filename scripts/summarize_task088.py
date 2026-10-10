"""TASK-088: summarise a stage's runs (protocol §7, §8, §10).

``--stage k0``: ik / hold / random on the calibration cohorts and the bars by §7's rule.
``--stage dev`` / ``gated``: every run of the stage, counts with exact 95 % intervals, the
reported-only quantities and, for ``gated``, the row (§8) with the frozen bars ``zero_shot.BARS``.
Writes ``<runs>/summary-<stage>.json`` and prints a table. Reads only the run files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import zero_shot as zs  # noqa: E402

STAGE_COHORTS = {
    "k0": ("k0-reach", "k0-grasp"),
    "dev": ("dev-reach", "dev-grasp"),
    "gated": ("gated-reach", "gated-grasp"),
    "debug": ("debug-reach", "debug-grasp"),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def arm_summary(run: dict) -> dict:
    eps = run["episodes"]
    k, n = int(sum(e["success"] for e in eps)), len(eps)
    lo, hi = zs.exact_interval(k, n)
    out = {
        "successes": k,
        "n": n,
        "rate": k / n if n else None,
        "interval95": [lo, hi],
        "stopped": int(sum(bool(e["stop_reason"]) for e in eps)),
        "stop_reasons": sorted({e["stop_reason"][:60] for e in eps if e["stop_reason"]}),
        "steps_mean": float(np.mean([e["steps"] for e in eps])),
        "revision": run["revision"],
        "tracked_tree_dirty": run["tracked_tree_dirty"],
        "goals_sha256": run["goals_sha256"],
        "model": run.get("model"),
    }
    if eps and "plan_seconds_mean" in eps[0]:
        out["plan_seconds_mean"] = float(np.mean([e["plan_seconds_mean"] for e in eps]))
    if eps and eps[0]["task"] == "reach":
        for t in zs.REACH_REPORT_TOLERANCES:
            out[f"success_at_{t}"] = int(sum(e["success_at"][str(t)] for e in eps))
        out["final_distance_median"] = float(np.median([e["final_distance"] for e in eps]))
        out["min_distance_median"] = float(np.median([e["min_distance"] for e in eps]))
    else:
        out["max_rise_median"] = float(np.median([e["max_rise"] for e in eps]))
        out["any_grasp_contact"] = int(sum(e["any_grasp_contact"] for e in eps))
        reached = {}
        for e in eps:
            for s in e["switches"]:
                key = f"{zs.SUBGOALS[s['from']]}:{s['reason']}"
                reached[key] = reached.get(key, 0) + 1
        out["switches"] = reached
    return out


def per_object(run: dict, goals: dict) -> dict:
    out = {}
    for e in run["episodes"]:
        obj = goals[e["seed"]]["target"]
        k, n = out.get(obj, (0, 0))
        out[obj] = (k + int(e["success"]), n + 1)
    return {o: {"successes": k, "n": n} for o, (k, n) in sorted(out.items())}


def paired(a: dict, b: dict) -> dict:
    sa = {e["seed"]: e["success"] for e in a["episodes"]}
    sb = {e["seed"]: e["success"] for e in b["episodes"]}
    if set(sa) != set(sb):
        raise SystemExit("paired runs differ in their resets")
    b_only = sum(1 for s in sa if sa[s] and not sb[s])
    c_only = sum(1 for s in sa if sb[s] and not sa[s])
    return {
        "first_only": b_only,
        "second_only": c_only,
        "two_sided_p": min(
            1.0, 2 * min(zs.mcnemar_one_sided(b_only, c_only), zs.mcnemar_one_sided(c_only, b_only))
        ),
        "first_better_one_sided_p": zs.mcnemar_one_sided(b_only, c_only),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--goals", required=True)
    parser.add_argument("--stage", required=True, choices=sorted(STAGE_COHORTS))
    args = parser.parse_args(argv)
    runs = Path(args.runs)
    summary = {"task": zs.TASK, "stage": args.stage, "cohorts": {}}
    for cohort in STAGE_COHORTS[args.stage]:
        folder = runs / cohort
        if not folder.exists():
            continue
        greport = load(Path(args.goals) / f"goals-{cohort}.json")
        goals = {g["seed"]: g for g in greport["goals"]}
        files = {p.stem: load(p) for p in sorted(folder.glob("*.json"))}
        shas = {r["goals_sha256"] for r in files.values()}
        if len(shas) != 1:
            raise SystemExit(f"{cohort}: runs used different goal files")
        c = {
            "goals_rejected_draws": sum(len(g["rejected"]) for g in greport["goals"]),
            "arms": {name: arm_summary(r) for name, r in files.items()},
        }
        if zs.cohort_task(cohort) == "grasp":
            c["targets"] = {
                o: sum(g["target"] == o for g in goals.values()) for o in zs.GRASP_OBJECTS
            }
            c["per_object"] = {name: per_object(r, goals) for name, r in files.items()}
        pairs = {}
        for first, second in (
            ("P-87100", "G-87100"),
            ("P-87100", "random"),
            ("G-87100", "random"),
            ("ik", "P-87100"),
            ("ik", "G-87100"),
            ("ik", "ik-nopress"),
        ):
            if first in files and second in files:
                pairs[f"{first} vs {second}"] = paired(files[first], files[second])
        c["paired"] = pairs
        summary["cohorts"][cohort] = c
    if args.stage == "k0":
        bars = {}
        for cohort in STAGE_COHORTS["k0"]:
            ik = summary["cohorts"][cohort]["arms"]["ik"]
            task = zs.cohort_task(cohort)
            rate = ik["successes"] / ik["n"]
            bars[task] = {
                "ik_successes": ik["successes"],
                "n": ik["n"],
                "set": rate >= 0.5,
                "bar": zs.bar_from_ceiling(task, ik["successes"], ik["n"]) if rate >= 0.5 else None,
            }
            if bars[task]["set"]:
                bars[task]["gated_count"] = zs.bar_count(bars[task]["bar"], 64)
        summary["bars"] = bars
    if args.stage == "gated":
        res = {}
        for cohort in STAGE_COHORTS["gated"]:
            task = zs.cohort_task(cohort)
            files = {p.stem: load(p) for p in sorted((runs / cohort).glob("*.json"))}
            res[task] = {
                "P": [e["success"] for e in files["P-87100"]["episodes"]],
                "G": [e["success"] for e in files["G-87100"]["episodes"]],
            }
        summary["row"] = zs.row(res, zs.BARS)
    out = runs / f"summary-{args.stage}.json"
    out.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    for cohort, c in summary["cohorts"].items():
        print(f"== {cohort} (rejected goal draws {c['goals_rejected_draws']})")
        for name, a in c["arms"].items():
            lo, hi = a["interval95"]
            extra = a.get("plan_seconds_mean")
            print(
                f"  {name:10s} {a['successes']:3d}/{a['n']:<3d} [{lo:.3f}, {hi:.3f}] "
                f"stopped {a['stopped']}" + (f" plan {extra:.2f}s" if extra else "")
            )
    if "bars" in summary:
        print("bars:", json.dumps(summary["bars"]))
    if "row" in summary:
        print("row:", json.dumps(summary["row"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
