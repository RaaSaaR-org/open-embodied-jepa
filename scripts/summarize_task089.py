"""TASK-089: summarise a stage (protocol §5.4, §5.5, §7.2, §8).

* ``--stage k0 --runs DIR``: A-K0's ceiling, the gated comparisons and bars, A-CEILING.
* ``--stage a --runs DIR``: Stage A's A1 (per arm and horizon) and A2 (per setting), the selection
  rule and the row (A-VOID / A-FAIL / A-PLAN / A-PASS(k)); uses the frozen ``A1_BARS``.
* ``--stage bd --runs DIR``: Stage BD's counts and the bar from ik's rate (§7.2).
* ``--stage bs --runs DIR --bd-summary FILE``: Stage BS's counts, the row and the reported tests.

Writes ``DIR/summary-<stage>.json`` (refuses an existing one) and prints it.
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
from embodied_jepa import zero_shot_grasp as zg  # noqa: E402


def load(path):
    return json.loads(Path(path).read_text())


def interval(k, n):
    lo, hi = zs.exact_interval(int(k), int(n))
    return [round(lo, 3), round(hi, 3)]


def common_checks(reports, cohort, n_expected):
    """Every report at one revision, from a clean tree (non-debug), on the cohort's seeds."""
    problems = []
    revisions = {r["revision"] for r in reports}
    if len(revisions) != 1:
        problems.append(f"revisions differ: {sorted(revisions)}")
    if not cohort.startswith("debug") and any(r["tracked_tree_dirty"] for r in reports):
        problems.append("a run was made from a dirty tree")
    for r in reports:
        if r.get("n", n_expected) != n_expected:
            problems.append(f"{r.get('arm', 'report')}: n = {r.get('n')}")
    return problems


def stage_k0(args):
    rep = load(Path(args.runs) / "ceiling-k0-roots.json")
    roots = load(Path(args.runs) / "roots-k0-roots.json")
    problems = common_checks([rep, roots], "k0-roots", len(zg.COHORTS["k0-roots"]))
    bars = rep["bars"]
    return {
        "stage": "k0",
        "cohort": "k0-roots",
        "n": rep["n"],
        "rates": rep["rates"],
        "bars": bars,
        "qualifying": {str(h): zg.qualifying(bars, h) for h in zg.HORIZONS},
        "a_ceiling": zg.ceiling_row(bars),
        "rejected_draws": roots["rejections"],
        "targets": _count([r["target"] for r in roots["roots"]]),
        "problems": problems,
        "revision": rep["revision"],
    }


def _count(items):
    out = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


def stage_a(args):
    if zg.A1_BARS is None:
        raise SystemExit("A1_BARS is not frozen (Stage 0 record)")
    d, cohort = Path(args.runs), args.cohort or "a-roots"
    a1, a2f = load(d / f"a1-{cohort}.json"), load(d / f"a2-{cohort}.json")
    roots = load(d / f"roots-{cohort}.json")
    n = a1["n"] if cohort.startswith("debug") else len(zg.COHORTS[cohort])
    problems = common_checks([a1, a2f, roots], cohort, n)
    if not cohort.startswith("debug") and [r["seed"] for r in roots["roots"]] != list(
        zg.COHORTS[cohort]
    ):
        problems.append("roots are not exactly the cohort's seeds")
    for f in (a1, a2f):
        if f["roots_sha256"] != a1["roots_sha256"]:
            problems.append("A1 and A2 read different roots")
    bars = zg.A1_BARS
    arms = {}
    for name, res in a1["arms"].items():
        arm = name.rsplit("-", 1)[0]
        family = zg.ARM_FAMILY[arm]
        arms[name] = {"family": family}
        for h in zg.HORIZONS:
            fb = bars[family][str(h)]
            out = zg.a1_pass(fb, res["model"][str(h)], res["shuffled"][str(h)], n)
            out["shuffled_wins"] = {c: int(sum(v)) for c, v in res["shuffled"][str(h)].items()}
            for entry in out["comparisons"].values():
                entry["interval"] = interval(entry["wins"], n)
            out["qualifying"] = zg.qualifying(bars, h)
            arms[name][str(h)] = out
        arms[name]["step_mean_wins"] = {c: int(sum(v)) for c, v in res["step_mean"].items()}
    ceiling = {
        f: {h: {c: int(sum(v)) for c, v in by.items()} for h, by in byh.items()}
        for f, byh in a1["ceiling"].items()
    }
    g = arms["G-87100"]
    a1_by_h = {h: bool(zg.qualifying(bars, h) and g[str(h)]["pass"]) for h in zg.HORIZONS}
    a2 = {}
    for model, by_setting in a2f["a2"].items():
        a2[model] = {}
        base = {k: np.asarray(by_setting["base"][k]) for k in ("close_R1", "lift_R2")}
        for setting, v in by_setting.items():
            m = {k: np.asarray(v[k]) for k in ("close_R1", "lift_R2")}
            entry = {k: int(m[k].sum()) for k in m}
            if setting != "base":
                entry |= zg.a2_meets(m, base)
            a2[model][setting] = entry
    change = zg.select_change(a1_by_h, a2["G"])
    row = zg.a_row(void="; ".join(problems), a1_by_h=a1_by_h, change=change)
    rise = {}
    for key in ("R1/close", "R1/open", "R2/lift", "R2/release", "R2/stay"):
        rs = [r["branches"][key]["max_rise"] for r in roots["roots"]]
        rise[key] = {
            "median_max_rise_m": float(np.median(rs)),
            "lifted_5cm": int(np.sum(np.asarray(rs) >= zs.LIFT_HEIGHT)),
        }
    return {
        "stage": "a",
        "cohort": cohort,
        "n": n,
        "row": row,
        "selected_change": change,
        "clause_fires": zg.clause_fires(row),
        "a1_by_h_G87100": {str(h): v for h, v in a1_by_h.items()},
        "a1": arms,
        "ceiling_wins_reported": ceiling,
        "a2": a2,
        "branch_rise": rise,
        "rejected_draws": roots["rejections"],
        "targets": _count([r["target"] for r in roots["roots"]]),
        "problems": problems,
        "revision": a1["revision"],
        "a1_bars": bars,
    }


def _runs(d, cohort):
    out = {}
    for f in sorted((Path(d) / cohort).glob("*.json")):
        out[f.stem] = load(f)
    return out


def _arm_table(runs, n):
    table = {}
    for name, r in runs.items():
        eps = r["episodes"]
        k = int(sum(e["success"] for e in eps))
        table[name] = {
            "successes": k,
            "n": len(eps),
            "interval": interval(k, len(eps)),
            "grasp_contact": int(sum(e["any_grasp_contact"] for e in eps)),
            "max_rise_median_m": float(np.median([e["max_rise"] for e in eps])),
            "stopped": int(sum(bool(e["stop_reason"]) for e in eps)),
            "plan_seconds_mean": float(
                np.mean([e["plan_seconds_mean"] for e in eps if "plan_seconds_mean" in e] or [0.0])
            ),
            "grasp_phase_share_closed": _phase_share(eps),
        }
    return table


def _phase_share(eps):
    """Share of executed grasp commands above 0.5 per subgoal phase, pooled over episodes."""
    pooled = {k: [] for k in range(4)}
    for e in eps:
        a = np.asarray(e["applied"])
        if not len(a):
            continue
        seq = []
        for i, s in enumerate(e["switches"]):
            seq += [i] * s["steps"]
        seq += [len(e["switches"])] * (len(a) - len(seq))
        seq = np.asarray(seq[: len(a)])
        for k in range(4):
            pooled[k] += list(a[seq == k, zg.GRASP_DIM])
    return {
        zs.SUBGOALS[k]: (float(np.mean(np.asarray(v) > 0.5)) if v else None)
        for k, v in pooled.items()
    }


def _targets(d, cohort):
    g = load(Path(d) / f"goals-{cohort}.json")
    return {x["seed"]: x["target"] for x in g["goals"]}, g


def stage_bd(args):
    cohort = args.cohort or "bd-grasp"
    runs = _runs(args.runs, cohort)
    n = runs["ik"]["n"] if cohort.startswith("debug") else len(zg.COHORTS[cohort])
    problems = common_checks(list(runs.values()), cohort, n)
    ik = runs["ik"]["successes"]
    bar = zg.b_bar(ik, n)
    targets, goals = _targets(args.runs, cohort)
    return {
        "stage": "bd",
        "cohort": cohort,
        "n": n,
        "arms": _arm_table(runs, n),
        "ik_successes": ik,
        "bar": bar,
        "bar_count_of_64": None if bar is None else zg.bar_count(bar, 64),
        "escalate": bar is None,
        "change": sorted({r.get("change") for r in runs.values()}),
        "targets": _count(targets.values()),
        "rejected_draws": goals["rejections"],
        "problems": problems,
        "revision": next(iter(runs.values()))["revision"],
    }


def stage_bs(args):
    runs = _runs(args.runs, "bs-grasp")
    n = len(zg.COHORTS["bs-grasp"])
    bd = load(args.bd_summary)
    problems = common_checks(list(runs.values()), "bs-grasp", n)
    need = {"G-chg-87100", "P-chg-87100", "G-base-87100", "ik", "ik-nopress", "hold", "random"}
    missing = need - set(runs)
    if missing:
        problems.append(f"missing runs: {sorted(missing)}")
    if bd["bar"] is None:
        problems.append("Stage BD set no bar")
    if len({r.get("change") for r in runs.values()}) != 1:
        problems.append("runs used different changes")
    for r in runs.values():
        if [e["seed"] for e in r["episodes"]] != list(zg.COHORTS["bs-grasp"]):
            problems.append(f"{r['arm']}: not the cohort's seeds")
    succ = {k: np.asarray([e["success"] for e in r["episodes"]]) for k, r in runs.items()}
    out = zg.b_row(succ["G-chg-87100"], succ["random"], bd["bar"] or 1.0, void="; ".join(problems))
    targets, goals = _targets(args.runs, "bs-grasp")
    per_obj = {}
    for name, r in runs.items():
        per_obj[name] = {}
        for e in r["episodes"]:
            t = targets[e["seed"]]
            per_obj[name].setdefault(t, [0, 0])
            per_obj[name][t][0] += int(e["success"])
            per_obj[name][t][1] += 1

    def mc(a, b):
        a, b = succ[a], succ[b]
        return {
            "discordant_for": int((a & ~b).sum()),
            "discordant_against": int((~a & b).sum()),
            "p_one_sided": zg.mcnemar_one_sided(a, b),
        }

    tests = {
        "G-chg vs G-base": mc("G-chg-87100", "G-base-87100"),
        "G-chg vs random": mc("G-chg-87100", "random"),
        "P-chg vs G-chg": mc("P-chg-87100", "G-chg-87100"),
        "G-chg vs P-chg": mc("G-chg-87100", "P-chg-87100"),
        "ik vs G-chg": mc("ik", "G-chg-87100"),
    }
    return {
        "stage": "bs",
        "cohort": "bs-grasp",
        "n": n,
        "row": out["row"],
        "row_detail": out,
        "clause_fires": zg.clause_fires(out["row"]),
        "bar": bd["bar"],
        "bd_summary_revision": bd["revision"],
        "change": sorted({r.get("change") for r in runs.values()}),
        "arms": _arm_table(runs, n),
        "tests": tests,
        "per_object": per_obj,
        "targets": _count(targets.values()),
        "rejected_draws": goals["rejections"],
        "problems": problems,
        "revision": next(iter(runs.values()))["revision"],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=("k0", "a", "bd", "bs"))
    p.add_argument("--runs", required=True)
    p.add_argument("--bd-summary")
    p.add_argument("--cohort", help="a debug cohort, for smokes (stages a and bd)")
    args = p.parse_args(argv)
    fn = {"k0": stage_k0, "a": stage_a, "bd": stage_bd, "bs": stage_bs}[args.stage]
    summary = fn(args)
    tag = f"{args.stage}-{args.cohort}" if args.cohort else args.stage
    out = Path(args.runs) / f"summary-{tag}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.write_text(json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(summary, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
