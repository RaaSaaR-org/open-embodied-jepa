"""TASK-093: summarise a stage (protocol docs/experiments/grasp_rate_projected_planner_v1.md).

* ``--stage 0 --runs DIR``: Stage 0's row (S0-ESCALATE or S0-PASS) from §6.1's calibration on
  cal-roots (``plans-cal-roots.json``: the bars and the degeneracy checks), the projector's
  equivalence tests (``equivalence.json``; ``smoke-checks.json``; the debug closed loop's
  request-vs-applied channel 13, M-proj none and M-v1 at least one) and the power rule
  (``power.json``), with the row rates (``rows.json``, reported).
* ``--stage p --runs DIR --s0-summary FILE``: Stage P's row (P-VOID, P-FAIL or P-PASS).
* ``--stage bd --runs DIR --p-summary FILE``: Stage BD's counts, the bar from ik's rate, the
  deferral, rate, request and equivalence measures (reported; an M-proj mismatch is a defect).
* ``--stage bs --runs DIR --bd-summary FILE --p-summary FILE``: Stage BS's row (§6.3, first match),
  the reported tests, the determinism re-run and the job split's checks.
* ``--stage timing --runs DIR --cohort debug-grasp``: Stage 0's per-command timing and the
  projected job times and split.

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

from embodied_jepa import grasp_rate_projected as rp  # noqa: E402
from embodied_jepa import zero_shot as zs  # noqa: E402
from embodied_jepa import zero_shot_grasp as zg  # noqa: E402


def load(path):
    return json.loads(Path(path).read_text())


def interval(k, n):
    lo, hi = zs.exact_interval(int(k), int(n))
    return [round(lo, 3), round(hi, 3)]


def _count(items):
    out = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


def common_checks(reports, cohort, n_expected=None):
    problems = []
    revisions = {r["revision"] for r in reports}
    if len(revisions) != 1:
        problems.append(f"revisions differ: {sorted(revisions)}")
    debug = str(cohort).startswith("debug")
    for r in reports:
        name = r.get("arm", "report")
        if not debug and r.get("tracked_tree_dirty"):
            problems.append(f"{name}: made from a dirty tree")
        if r.get("task") != rp.TASK:
            problems.append(f"{name}: not a TASK-093 report")
        if not r.get("completed") or r.get("cap_hit"):
            problems.append(f"{name}: did not complete under the cap")
        if r.get("frozen_files_match") is False:
            problems.append(f"{name}: frozen files differed")
        if n_expected is not None and r.get("n", n_expected) != n_expected:
            problems.append(f"{name}: n = {r.get('n')}")
    return problems


def _runs(d, cohort):
    runs, parts = {}, {}
    for f in sorted((Path(d) / cohort).glob("*.json")):
        (parts if ".part" in f.stem else runs)[f.stem] = load(f)
    return runs, parts


def _per_seed_plans(rep):
    out = {}
    for name, v in rep["results"].items():
        out[name] = {
            arm: {
                "C8": v[arm]["C8"],
                "H8": v[arm]["H8"],
                "p_mech_mean_abs_gap_R1": round(v[arm]["p_mech_mean_abs_gap_R1"], 4),
                "close_R1_request_terms": v[arm]["close_R1_request_terms"],
                "lift_R2_request_terms": v[arm]["lift_R2_request_terms"],
                "mean_executed_R1": np.asarray(v[arm]["executed_R1"]).mean(0).round(3).tolist(),
                "mean_executed_R2": np.asarray(v[arm]["executed_R2"]).mean(0).round(3).tolist(),
                "mean_predicted_synergy_R1": np.asarray(v[arm]["predicted_synergy_R1"])
                .mean(0)
                .round(3)
                .tolist(),
            }
            for arm in rp.P_ARMS
        }
        out[name]["M-proj"]["requests_beyond_limit"] = v["M-proj"]["requests_beyond_limit"]
        out[name]["last_two_triggered"] = v["model"].get("last_two_triggered")
        if "p_gate" in v:
            out[name]["p_gate"] = v["p_gate"]
    return out


def _closed_loop_equivalence(runs):
    """§14: in the debug closed loop, M-proj's applied channel 13 equals the request on every
    command; M-v1 (the negative control) differs on at least one."""
    out = {}
    for name, r in runs.items():
        if r.get("arm") not in rp.CLOSED_LOOP_ARMS:
            continue
        eps = r["episodes"]
        out[name] = {
            "arm": r["arm"],
            "commands": int(sum(e["equivalence"]["commands"] for e in eps)),
            "channel13_differs": int(sum(e["equivalence"]["channel13_differs"] for e in eps)),
            "grasp_phase_commands": int(sum(e["equivalence"]["grasp_phase_commands"] for e in eps)),
            "grasp_phase_channel13_differs": int(
                sum(e["equivalence"]["grasp_phase_channel13_differs"] for e in eps)
            ),
        }
    proj = [v for v in out.values() if v["arm"] == "M-proj"]
    ctrl = [v for v in out.values() if v["arm"] == "M-v1"]
    ok = bool(
        proj
        and ctrl
        and all(v["channel13_differs"] == 0 and v["commands"] > 0 for v in proj)
        and any(v["channel13_differs"] > 0 for v in ctrl)
    )
    return {"runs": out, "pass": ok}


# ----- Stage 0 (§6.1, §6.4, §14) ---------------------------------------------------------------
def stage_0(args):
    d = Path(args.runs)
    cal = load(d / "plans-cal-roots.json")
    roots = load(d / "roots-cal-roots.json")
    problems = common_checks([cal, roots], "cal-roots", len(rp.COHORTS["cal-roots"]))
    problems += cal.get("problems", [])
    eq = load(d / "equivalence.json")
    smoke = load(d / "smoke-checks.json")
    power = load(d / "power.json")
    rows = load(d / "rows.json")
    runs, _ = _runs(d, "debug-grasp")
    cl = _closed_loop_equivalence(runs)
    equivalence_ok = bool(eq.get("pass") and smoke.get("pass") and cl["pass"])
    rule = rp.power_rule(power["rows"])
    power_ok = not rule["escalate"] and power.get("sims") == rp.POWER_SIMS
    calib = cal["calibration"]
    row = rp.s0_row(calib, equivalence_ok=equivalence_ok and not problems, power_ok=power_ok)
    return {
        "stage": "0",
        "row": row["row"],
        "s0_row": row,
        "bars": row["bars"],
        "calibration": calib,
        "references": cal["references"],
        "per_seed": _per_seed_plans(cal),
        "equivalence": {
            "embodiment": {
                k: eq.get(k)
                for k in (
                    "states",
                    "compared_steps",
                    "clipped_steps",
                    "mismatches",
                    "nonzero_feasible_sequences",
                    "other_channels_equal",
                    "pass",
                )
            },  # fmt: skip
            "smoke_checks": {"pass": smoke.get("pass"), "rows": len(smoke.get("rows", []))},
            "closed_loop": cl,
        },
        "power": {"rule": rule, "rows": power["rows"], "sims": power.get("sims")},
        "row_rates": rows["rows"],
        "targets": _count([r["target"] for r in roots["roots"]]),
        "rejected_draws": roots.get("rejections"),
        "geometry": cal.get("geometry"),
        "problems": problems,
        "revision": cal.get("revision"),
    }


# ----- Stage P (§6.2) --------------------------------------------------------------------------
def stage_p(args):
    cohort = args.cohort or "p-roots"
    d = Path(args.runs)
    rep = load(d / f"plans-{cohort}.json")
    roots = load(d / f"roots-{cohort}.json")
    n = None if cohort.startswith("debug") else len(rp.COHORTS[cohort])
    problems = common_checks([rep, roots], cohort, n)
    problems += rep.get("problems", [])
    if rep.get("bars") != rp.P_BARS:
        problems.append("Stage P's bars are not Stage 0's frozen ones")
    if not cohort.startswith("debug"):
        s0 = load(args.s0_summary)
        if s0.get("row") != "S0-PASS" or s0.get("bars") != rp.P_BARS:
            problems.append("Stage 0 did not pass with these bars")
    for name, v in rep.get("results", {}).items():
        if v["model"]["checkpoint_sha256"] != rp.CHECKPOINT_SHA256.get(name):
            problems.append(f"{name}: not the pinned checkpoint")
    passes = {s: bool(rep["results"][f"G-v1-{s}"].get("p_gate")) for s in rp.MODEL_SEEDS}
    row = rp.p_row(passes, void="; ".join(problems))
    return {
        "stage": "p",
        "cohort": cohort,
        "n": rep.get("n"),
        "row": row["row"],
        "p_row": row,
        "bars": rep.get("bars"),
        "clause_fires": False,
        "per_seed": _per_seed_plans(rep),
        "references": rep.get("references"),
        "targets": _count([r["target"] for r in roots["roots"]]),
        "rejected_draws": roots.get("rejections"),
        "problems": problems,
        "revision": rep.get("revision"),
    }


# ----- the closed loop (§6.3) ------------------------------------------------------------------
def _phase_share(eps):
    """TASK-089's executed grasp share per subgoal phase (TASK-092's summarizer, copied)."""
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
        if r.get("determinism_rerun"):
            continue
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


def _measures(runs):
    out = {}
    for name, r in runs.items():
        if r.get("arm") not in rp.CLOSED_LOOP_ARMS or r.get("determinism_rerun"):
            continue
        eps = r["episodes"]
        succ = [bool(e["success"]) for e in eps]
        rq = [e["requests"] for e in eps]
        steps = sum(m["plan_steps"] for m in rq)
        out[name] = {
            "deferral": rp.deferral_summary([e["deferral"] for e in eps]),
            "rate": rp.rate_summary([e["rate"] for e in eps], succ),
            "requested_steps_beyond_limit_share": (
                sum(m["requested_steps_beyond_limit"] for m in rq) / steps if steps else None
            ),
            "equivalence": _closed_loop_equivalence({name: r})["runs"][name],
        }
    return out


def _split_checks(runs, parts, cohort, debug):
    problems = []
    for name, r in runs.items():
        if r.get("arm") not in rp.CLOSED_LOOP_ARMS or r.get("determinism_rerun"):
            continue
        mine = {k: v for k, v in parts.items() if k.split(".part")[0] == name}
        need = rp.needs_parts(cohort, r["arm"])
        if need and not r.get("combined"):
            problems.append(f"{name}: not combined from its {need} jobs")
        if r.get("combined"):
            if not debug and len(r["parts"]) != rp.BS_PARTS:
                problems.append(f"{name}: {len(r['parts'])} parts, not {rp.BS_PARTS}")
            if len(mine) != len(r["parts"]):
                problems.append(f"{name}: part files and combined parts differ")
            try:
                eps = rp.combine_parts(list(mine.values()), [e["seed"] for e in r["episodes"]])
                if eps["episodes"] != r["episodes"]:
                    problems.append(f"{name}: combined episodes differ from its parts")
            except Exception as error:  # noqa: BLE001
                problems.append(f"{name}: {error}")
    for k, p in parts.items():
        if k.split(".part")[0] not in runs:
            problems.append(f"{k}: a part without its combined run")
        if p.get("cap_hit"):
            problems.append(f"{k}: hit the 7 200 s cap")
    return problems


def _equivalence_problems(runs):
    """§6.3's B-VOID check: no M-proj command whose applied channel 13 differs from the request."""
    problems = []
    for name, r in runs.items():
        if r.get("arm") == "M-proj":
            k = sum(e["equivalence"]["channel13_differs"] for e in r["episodes"])
            if k:
                problems.append(f"{name}: {k} commands' applied channel 13 differs")
    return problems


def prior_checks(args, runs) -> list[str]:
    """Non-debug BD and BS: Stage P ended P-PASS cleanly at the same revision."""
    if not args.p_summary:
        return ["no Stage P summary"]
    ps = load(args.p_summary)
    problems = []
    if ps.get("stage") != "p" or ps.get("row") != "P-PASS" or ps.get("problems"):
        problems.append("Stage P did not end P-PASS cleanly")
    revs = {r["revision"] for r in runs.values()}
    if revs and ps.get("revision") not in revs:
        problems.append("Stage P ran at another revision")
    return problems


def stage_bd(args):
    cohort = args.cohort or "bd-grasp"
    debug = cohort.startswith("debug")
    allr, parts = _runs(args.runs, cohort)
    runs = {k: v for k, v in allr.items() if not v.get("determinism_rerun")}
    if debug and args.only:
        runs = {k: v for k, v in runs.items() if k in args.only}
    n = runs["ik"]["n"] if debug else len(rp.COHORTS[cohort])
    problems = common_checks(list(runs.values()), cohort, n)
    problems += _split_checks(runs, parts, cohort, debug)
    problems += _equivalence_problems(runs)
    if not debug:
        for r in runs.values():
            if [e["seed"] for e in r["episodes"]] != list(rp.COHORTS[cohort]):
                problems.append(f"{r['arm']}: not the cohort's seeds")
        need = {f"{a}-{s}" for a in rp.CLOSED_LOOP_ARMS for s in rp.BD_SEEDS[a]}
        need |= set(rp.BASELINE_ARMS)
        if need - set(runs):
            problems.append(f"missing runs: {sorted(need - set(runs))}")
        problems += prior_checks(args, runs)
    ik = runs["ik"]["successes"]
    bar = rp.b_bar(ik, n)
    g = load(Path(args.runs) / f"goals-{cohort}.json")
    return {
        "stage": "bd",
        "cohort": cohort,
        "n": n,
        "arms": _arm_table(runs),
        "measures": _measures(runs),
        "ik_successes": ik,
        "bar": bar,
        "pooled_bar_at_bs": None if bar is None else rp.pooled_bar(bar, rp.BS_N),
        "seed_floor_at_bs": None if bar is None else rp.seed_floor(bar, rp.BS_N),
        "escalate": bar is None,
        "row": "BD-ESCALATE" if bar is None else None,
        "targets": _count([x["target"] for x in g["goals"]]),
        "rejected_draws": g["rejections"],
        "problems": problems,
        "revision": next(iter(runs.values()))["revision"],
    }


def stage_bs(args):
    cohort = args.cohort or "bs-grasp"
    debug = cohort.startswith("debug")
    allr, parts = _runs(args.runs, cohort)
    det = {k: v for k, v in allr.items() if v.get("determinism_rerun")}
    runs = {k: v for k, v in allr.items() if k not in det}
    n = runs["ik"]["n"] if debug else len(rp.COHORTS[cohort])
    bd = load(args.bd_summary)
    problems = common_checks(list(allr.values()), cohort, None)
    problems += _split_checks(runs, parts, cohort, debug)
    problems += _equivalence_problems(runs)
    for r in runs.values():
        if r["n"] != n:
            problems.append(f"{r['arm']}: n = {r['n']}")
    if not debug:
        if bd.get("cohort") != "bd-grasp" or bd.get("problems") or bd.get("stage") != "bd":
            problems.append("the BD summary is not a clean Stage BD summary")
        if runs and bd.get("revision") != next(iter(runs.values()))["revision"]:
            problems.append("Stage BD ran at another revision")
        for r in runs.values():
            if [e["seed"] for e in r["episodes"]] != list(rp.COHORTS[cohort]):
                problems.append(f"{r['arm']}: not the cohort's seeds")
        problems += prior_checks(args, allr)
    seeds = args.model_seeds or list(rp.MODEL_SEEDS)
    need = {f"M-proj-{s}" for s in seeds} | {f"M-v1-{s}" for s in seeds}
    need |= set(rp.BASELINE_ARMS)
    missing = need - set(runs)
    if missing:
        problems.append(f"missing runs: {sorted(missing)}")
    if bd["bar"] is None:
        problems.append("Stage BD set no bar")
    det_ok = None
    key = f"M-proj-{seeds[0]}"
    if det:
        d = next(iter(det.values()))
        first = {e["seed"]: e for e in runs.get(key, {"episodes": []})["episodes"]}
        det_ok = (
            all(
                e["applied"] == first.get(e["seed"], {}).get("applied")
                and e["success"] == first.get(e["seed"], {}).get("success")
                for e in d["episodes"]
            )
            and len(d["episodes"]) == rp.DETERMINISM[2]
            and [e["seed"] for e in d["episodes"]]
            == [e["seed"] for e in runs.get(key, {"episodes": []})["episodes"]][:4]
        )
        if not det_ok:
            problems.append("the determinism re-run does not match")
    elif not debug:
        problems.append("no determinism re-run")
    if problems and missing:
        return {"stage": "bs", "row": "B-VOID", "problems": problems}
    succ = {k: np.asarray([e["success"] for e in r["episodes"]], bool) for k, r in runs.items()}
    if debug and len(seeds) < 3:  # the debug smoke: one seed, repeated, for the plumbing only
        seeds = (seeds * 3)[:3]
    p = np.stack([succ[f"M-proj-{s}"] for s in seeds])
    c = np.stack([succ[f"M-v1-{s}"] for s in seeds])
    out = rp.b_row(p, c, succ["random"], bd["bar"] or 1.0, void="; ".join(problems))
    reported = {
        "ik vs M-proj (per reset, ik x 3)": {
            "p_signflip": rp.signflip_p(3 * succ["ik"].astype(int) - rp.per_reset(p))
        }
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
    beside = {}
    if args.p_summary:
        ps = load(args.p_summary)
        beside["stage_p"] = {"row": ps.get("row"), "per_seed": ps.get("per_seed")}
    return {
        "stage": "bs",
        "cohort": cohort,
        "n": n,
        "row": out["row"],
        "row_detail": out,
        "clause_fires": rp.clause_fires(out["row"]),
        "lewm_needed_claimed": False,
        "bar": bd["bar"],
        "determinism_matches": det_ok,
        "arms": _arm_table(runs),
        "measures": _measures(runs),
        "beside_each_seed": beside,
        "reported_tests": reported,
        "per_object": per_obj,
        "targets": _count(targets.values()),
        "rejected_draws": gl["rejections"],
        "problems": problems,
        "revision": next(iter(runs.values()))["revision"],
    }


# ----- Stage 0's timing (§10, §14) -------------------------------------------------------------
def stage_timing(args):
    cohort = args.cohort
    runs, parts = _runs(args.runs, cohort)
    runs |= parts
    rows = {}
    for name, r in runs.items():
        if not r.get("completed"):
            continue
        eps = r["episodes"]
        commands = sum(e["steps"] for e in eps)
        procs = min(r["workers"], len(eps)) if isinstance(r["workers"], int) else None
        if not procs or not commands:
            continue
        per_command_wall = r["seconds"] * procs / commands
        row = {
            "arm": r["arm"],
            "processes": procs,
            "episodes": len(eps),
            "commands": commands,
            "seconds": r["seconds"],
            "wall_seconds_per_command_per_process": per_command_wall,
        }
        if r.get("arm") in rp.CLOSED_LOOP_ARMS:
            full = zs.BUDGET["grasp"]
            job64 = rp.project_job_seconds(per_command_wall, full, 64, 6)
            run128 = rp.project_job_seconds(per_command_wall, full, rp.BS_N, 6)
            row |= {
                "plan_seconds_per_command": float(
                    np.mean([e["plan_seconds_mean"] for e in eps if "plan_seconds_mean" in e])
                ),
                "projected_64_reset_job_seconds": job64,
                "projected_128_reset_run_seconds": run128,
                "parts_needed": rp.parts_needed(run128),
            }
        rows[name] = row
    wm = [v for v in rows.values() if "projected_64_reset_job_seconds" in v]
    worst = max((v["projected_64_reset_job_seconds"] for v in wm), default=None)
    return {
        "stage": "timing",
        "cohort": cohort,
        "runs": rows,
        "worst_projected_64_reset_job_seconds": worst,
        "split_trigger_seconds": rp.SPLIT_TRIGGER_SECONDS,
        "bs_parts": max([rp.BS_PARTS] + [v["parts_needed"] for v in wm]),
        "note": "debug timing; projections assume every episode runs the full 400 commands",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("0", "p", "bd", "bs", "timing"))
    parser.add_argument("--runs", required=True)
    parser.add_argument("--s0-summary")
    parser.add_argument("--p-summary")
    parser.add_argument("--bd-summary")
    parser.add_argument("--cohort", help="debug cohorts only (smoke)")
    parser.add_argument("--model-seeds", type=int, nargs="+", help="debug only")
    parser.add_argument("--only", nargs="+", help="debug only: the runs to summarise")
    parser.add_argument("--name", help="debug only: the summary's file name suffix")
    args = parser.parse_args(argv)
    if args.cohort and not args.cohort.startswith("debug"):
        raise SystemExit("--cohort is for debug cohorts only")
    if (args.model_seeds or args.only or args.name) and not args.cohort:
        raise SystemExit("--model-seeds, --only and --name are for debug runs only")
    if args.stage == "timing" and not args.cohort:
        raise SystemExit("timing is Stage 0's, on a debug cohort")
    if args.stage == "p" and not args.cohort and not args.s0_summary:
        raise SystemExit("Stage P needs Stage 0's summary")
    if args.stage == "bs" and not args.bd_summary:
        raise SystemExit("Stage BS needs the Stage BD summary")
    if args.stage in ("bd", "bs") and not args.cohort and not args.p_summary:
        raise SystemExit("Stages BD and BS need Stage P's summary (§7 order)")
    fn = {"0": stage_0, "p": stage_p, "bd": stage_bd, "bs": stage_bs,
          "timing": stage_timing}[args.stage]  # fmt: skip
    summary = fn(args)
    suffix = args.name or (args.cohort or "")
    name = f"summary-{args.stage}{'-' + suffix if suffix else ''}.json"
    path = Path(args.runs) / name
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.write_text(json.dumps(summary, indent=1, sort_keys=True, allow_nan=True) + "\n")
    print(json.dumps({k: summary.get(k) for k in ("stage", "row", "bar", "bars", "problems")},
                     indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
