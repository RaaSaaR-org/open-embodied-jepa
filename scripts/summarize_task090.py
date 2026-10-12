"""TASK-090: summarise a stage (protocol docs/experiments/grasp_play_corpus_v1.md §4-§8).

* ``--stage k0 --runs DIR``: A-K0's ceiling, gating, bars and A-CEILING (h = 8 must qualify).
* ``--stage t --runs DIR``: Stage T's eight reports and their checks (T-VOID).
* ``--stage e --runs DIR`` / ``--stage a --runs DIR``: Stage E's and Stage A′'s rows, with the
  revision and clean-tree checks.
* ``--stage bd --runs DIR``: Stage BD's counts and the bar from ik's rate (§7.2).
* ``--stage bs --runs DIR --bd-summary FILE``: Stage BS's row, the reported tests and the
  determinism re-run.

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

from embodied_jepa import grasp_corpus_wm as gc  # noqa: E402
from embodied_jepa import zero_shot as zs  # noqa: E402
from embodied_jepa import zero_shot_grasp as zg  # noqa: E402

IMPLEMENTATION_SHA256 = "e2ec0096c14f628c248fda7fb1a05ad2d3342fc7e20703727bcc178efd7149b1"
MODEL_ARMS = ("M-v2g", "M-v1p", "M-v1")


def load(path):
    return json.loads(Path(path).read_text())


def interval(k, n):
    lo, hi = zs.exact_interval(int(k), int(n))
    return [round(lo, 3), round(hi, 3)]


def common_checks(reports, cohort, n_expected=None):
    problems = []
    revisions = {r["revision"] for r in reports}
    if len(revisions) != 1:
        problems.append(f"revisions differ: {sorted(revisions)}")
    if not str(cohort).startswith("debug") and any(r["tracked_tree_dirty"] for r in reports):
        problems.append("a run was made from a dirty tree")
    if n_expected is not None:
        for r in reports:
            if r.get("n", n_expected) != n_expected:
                problems.append(f"{r.get('arm', 'report')}: n = {r.get('n')}")
    return problems


def _count(items):
    out = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


def stage_k0(args):
    rep = load(Path(args.runs) / "ceiling-k0-roots.json")
    roots = load(Path(args.runs) / "roots-k0-roots.json")
    problems = common_checks([rep, roots], "k0-roots", len(gc.COHORTS["k0-roots"]))
    bars = rep["bars"]
    return {
        "stage": "k0",
        "cohort": "k0-roots",
        "n": rep["n"],
        "rates": rep["rates"],
        "bars": bars,
        "qualifying": {str(h): zg.qualifying(bars, h) for h in zg.HORIZONS},
        "row": "A-CEILING" if not zg.qualifying(bars, 8) else "A-K0-PASS",
        "rejected_draws": roots["rejections"],
        "targets": _count([r["target"] for r in roots["roots"]]),
        "problems": problems,
        "revision": rep["revision"],
    }


def stage_t(args):
    runs = Path(args.runs)
    expected = [("G", c, s) for c in gc.CORPORA for s in gc.MODEL_SEEDS]
    expected += [("N", c, gc.N_SEED) for c in gc.CORPORA]
    reports, problems, models = [], [], {}
    for arm, corpus, seed in expected:
        name = f"{arm}-{gc.CORPUS_TAG[corpus]}-{seed}"
        folder = runs / name
        if (runs / f"{name}-r2").exists():
            problems.append(f"{name}: run twice")
        if not (folder / "report.json").exists():
            problems.append(f"{name}: no report")
            continue
        rep = load(folder / "report.json")
        reports.append(rep)
        checks = {
            "outcome": rep.get("outcome") == "T-JOB-DONE" and rep.get("kind") == "job",
            "implementation": rep["implementation_sha256"] == IMPLEMENTATION_SHA256,
            "moments": rep["moments"]["bitwise_equal_g87100"]
            and rep["moments"]["pinned_sha256"] == rep["moments"]["g87100_sha256"],
            "identity": (rep["arm"], rep["corpus"], int(rep["seed"])) == (arm, corpus, seed),
            "updates": rep["training"]["updates"] == 60_000,
        }
        if not all(checks.values()):
            problems.append(f"{name}: failed {[k for k, v in checks.items() if not v]}")
        models[name] = {
            "checkpoint_sha256": rep["checkpoint_sha256"],
            "kept_update": rep["training"]["kept_update"],
            "last_two_triggered": rep["training"]["last_two_triggered"],
            "kept_val_criterion": rep["training"]["kept_val_criterion"],
            "seconds": rep["training"]["seconds"],
            "seconds_per_update_median": rep["training"]["seconds_per_update_median"],
            "mmap": rep["mmap"],
            "moments_sha256": rep["moments"]["pinned_sha256"],
            "feature_reports_sha256": rep["feature_reports_sha256"],
            "checks": checks,
        }
    problems += common_checks(reports, "stage-t")
    if len({json.dumps(r["feature_reports_sha256"], sort_keys=True) for r in reports}) > 1:
        problems.append("the jobs used different feature stores")
    return {
        "stage": "t",
        "row": "T-VOID" if problems else "T-DONE",
        "models": models,
        "problems": problems,
        "revision": reports[0]["revision"] if reports else None,
    }


def stage_e(args):
    rep = load(Path(args.runs) / "stage-e.json")
    problems = common_checks([rep], "stage-e")
    return {
        "stage": "e",
        "row": "E-VOID" if problems else rep["row"],
        "checks": rep["checks"],
        "lambda_reported": rep["lambda_reported"],
        "problems": problems,
        "revision": rep["revision"],
    }


def stage_a(args):
    rep = load(Path(args.runs) / "a-prime-a-roots.json")
    roots = load(Path(args.runs) / "roots-a-roots.json")
    problems = common_checks([rep, roots], "a-roots", len(gc.COHORTS["a-roots"]))
    return {
        "stage": "a",
        "row": "A′-VOID" if problems else rep["a_prime"]["row"],
        "a_prime": rep["a_prime"],
        "per_model": {
            k: {
                "a1_h8_pass": v.get("a1_h8", {}).get("pass"),
                "a2": {kk: v["a2"][kk]["count"] for kk in ("close_R1", "lift_R2")}
                if "a2" in v
                else None,
                "a_prime_pass": v.get("a_prime_pass"),
            }
            for k, v in rep["results"].items()
        },
        "probe": rep["probe"],
        "problems": problems,
        "revision": rep["revision"],
    }


def _runs(d, cohort):
    return {f.stem: load(f) for f in sorted((Path(d) / cohort).glob("*.json"))}


def _phase_share(eps):
    """TASK-089's executed grasp share per subgoal phase (its review-corrected alignment)."""
    pooled = {k: [] for k in range(4)}
    for e in eps:
        a = np.asarray(e["applied"])
        if not len(a):
            continue
        seq = []
        for i, s in enumerate(e["switches"]):
            seq += [i] * s["steps"]
        seq = seq[1:]
        seq += [len(e["switches"])] * (len(a) - len(seq))
        seq = np.asarray(seq[: len(a)])
        for k in range(4):
            pooled[k] += list(a[seq == k, zg.GRASP_DIM])
    return {
        zs.SUBGOALS[k]: (float(np.mean(np.asarray(v) > 0.5)) if v else None)
        for k, v in pooled.items()
    }


def _arm_table(runs):
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
            "final_subgoal": _count([e["final_subgoal"] for e in eps]),
            "stopped": int(sum(bool(e["stop_reason"]) for e in eps)),
            "plan_seconds_mean": float(
                np.mean([e["plan_seconds_mean"] for e in eps if "plan_seconds_mean" in e] or [0.0])
            ),
            "wall_seconds": float(r["seconds"]),
            "grasp_phase_share_closed": _phase_share(eps),
        }
    return table


def stage_bd(args):
    cohort = args.cohort or "bd-grasp"
    runs = {k: v for k, v in _runs(args.runs, cohort).items() if not k.endswith("determinism")}
    n = runs["ik"]["n"] if cohort.startswith("debug") else len(gc.COHORTS[cohort])
    problems = common_checks(list(runs.values()), cohort, n)
    ik = runs["ik"]["successes"]
    bar = zg.b_bar(ik, n)
    g = load(Path(args.runs) / f"goals-{cohort}.json")
    return {
        "stage": "bd",
        "cohort": cohort,
        "n": n,
        "arms": _arm_table(runs),
        "ik_successes": ik,
        "bar": bar,
        "pooled_bar_at_bs": None if bar is None else gc.pooled_bar(bar, gc.BS_N),
        "seed_floor_at_bs": None if bar is None else gc.seed_floor(bar, gc.BS_N),
        "escalate": bar is None,
        "targets": _count([x["target"] for x in g["goals"]]),
        "rejected_draws": g["rejections"],
        "problems": problems,
        "revision": next(iter(runs.values()))["revision"],
    }


def stage_bs(args):
    cohort = args.cohort or "bs-grasp"
    debug = cohort.startswith("debug")
    allr = _runs(args.runs, cohort)
    det = {k: v for k, v in allr.items() if k.endswith("determinism")}
    runs = {k: v for k, v in allr.items() if k not in det}
    n = runs["ik"]["n"] if debug else len(gc.COHORTS[cohort])
    bd = load(args.bd_summary)
    problems = common_checks(list(allr.values()), cohort, None)
    for r in runs.values():
        if r["n"] != n:
            problems.append(f"{r['arm']}: n = {r['n']}")
    if not debug:
        if bd.get("cohort") != "bd-grasp" or bd.get("problems") or bd.get("stage") != "bd":
            problems.append("the BD summary is not a clean Stage BD summary")
        if runs and bd.get("revision") != next(iter(runs.values()))["revision"]:
            problems.append("Stage BD ran at another revision")
        for r in runs.values():
            if [e["seed"] for e in r["episodes"]] != list(gc.COHORTS[cohort]):
                problems.append(f"{r['arm']}: not the cohort's seeds")
    seeds = args.model_seeds or list(gc.MODEL_SEEDS)
    v1_seeds = list(gc.BS_M_V1_SEEDS)
    need = {f"M-v2g-{s}" for s in seeds} | {f"M-v1p-{s}" for s in seeds}
    need |= {f"M-v1-{s}" for s in v1_seeds} | {"ik", "ik-nopress", "hold", "random"}
    missing = need - set(runs)
    if missing:
        problems.append(f"missing runs: {sorted(missing)}")
    if bd["bar"] is None:
        problems.append("Stage BD set no bar")
    # the determinism re-run: the first four resets of M-v2g-90100 must match bitwise (§7.2)
    det_ok = None
    key = f"M-v2g-{seeds[0]}"
    if det:
        d = next(iter(det.values()))
        first = {e["seed"]: e for e in runs.get(key, {"episodes": []})["episodes"]}
        det_ok = (
            all(
                e["applied"] == first.get(e["seed"], {}).get("applied")
                and e["success"] == first.get(e["seed"], {}).get("success")
                for e in d["episodes"]
            )
            and len(d["episodes"]) == 4
        )
        if not det_ok:
            problems.append("the determinism re-run does not match")
    elif not debug:
        problems.append("no determinism re-run")
    if problems and missing:
        return {"stage": "bs", "row": "B-VOID", "problems": problems}
    succ = {k: np.asarray([e["success"] for e in r["episodes"]], bool) for k, r in runs.items()}
    g = np.stack([succ[f"M-v2g-{s}"] for s in seeds])
    c = np.stack([succ[f"M-v1p-{s}"] for s in seeds])
    out = gc.b_row(g, c, succ["random"], bd["bar"] or 1.0, void="; ".join(problems))
    reported = {}
    if all(f"M-v1-{s}" in succ for s in v1_seeds):
        # M-v1's seeds (3, or 87100 alone when Stage 0 dropped two) scaled to three per reset
        v1 = np.stack([succ[f"M-v1-{s}"] for s in v1_seeds])
        scale = 3 // len(v1_seeds)
        for a_name, a in (("M-v2g", g), ("M-v1p", c)):
            d_r = gc.per_reset(a) - scale * gc.per_reset(v1)
            reported[f"{a_name} vs M-v1"] = {
                "m_v1_seeds": v1_seeds,
                "p_signflip": gc.signflip_p(d_r),
                "bootstrap": gc.reset_bootstrap(a, np.repeat(v1, scale, 0))
                if len(v1_seeds) in (1, 3)
                else None,
            }
    reported["ik vs M-v2g (per reset, ik x 3)"] = {
        "p_signflip": gc.signflip_p(3 * succ["ik"].astype(int) - gc.per_reset(g))
    }
    gl = load(Path(args.runs) / f"goals-{cohort}.json")
    targets = {x["seed"]: x["target"] for x in gl["goals"]}
    per_obj = {}
    for name, r in runs.items():
        per_obj[name] = {}
        for e in r["episodes"]:
            t = targets[e["seed"]]
            per_obj[name].setdefault(t, [0, 0])
            per_obj[name][t][0] += int(e["success"])
            per_obj[name][t][1] += 1
    return {
        "stage": "bs",
        "cohort": cohort,
        "n": n,
        "row": out["row"],
        "row_detail": out,
        "clause_fires": gc.clause_fires(out["row"]),
        "bar": bd["bar"],
        "determinism_matches": det_ok,
        "arms": _arm_table(runs),
        "reported_tests": reported,
        "per_object": per_obj,
        "targets": _count(targets.values()),
        "rejected_draws": gl["rejections"],
        "problems": problems,
        "revision": next(iter(runs.values()))["revision"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("k0", "t", "e", "a", "bd", "bs"))
    parser.add_argument("--runs", required=True)
    parser.add_argument("--bd-summary")
    parser.add_argument("--cohort", help="debug cohorts only (smoke)")
    parser.add_argument("--model-seeds", type=int, nargs="+", help="debug only")
    args = parser.parse_args(argv)
    if args.cohort and not args.cohort.startswith("debug"):
        raise SystemExit("--cohort is for debug cohorts only")
    if args.model_seeds and not args.cohort:
        raise SystemExit("--model-seeds is for debug cohorts only")
    fn = {
        "k0": stage_k0,
        "t": stage_t,
        "e": stage_e,
        "a": stage_a,
        "bd": stage_bd,
        "bs": stage_bs,
    }[args.stage]
    if args.stage == "bs" and not args.bd_summary:
        raise SystemExit("Stage BS needs the Stage BD summary")
    summary = fn(args)
    name = f"summary-{args.stage}{'-' + args.cohort if args.cohort else ''}.json"
    path = Path(args.runs) / name
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.write_text(json.dumps(summary, indent=1, sort_keys=True, allow_nan=True) + "\n")
    print(json.dumps({k: summary.get(k) for k in ("stage", "row", "bar", "problems")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
