"""TASK-092: summarise a stage (protocol docs/experiments/grasp_rate_aware_wm_v1.md).

* ``--stage t --runs DIR``: Stage T's three jobs and their checks (T-VOID or T-DONE).
* ``--stage r --runs DIR``: Stage R's row (R-VOID, R-PREMISE, E-FAIL, R-FAIL or R-PASS) from
  ``stage-r.json``, with every seed's R1, R2 and E.
* ``--stage a --runs DIR --r-summary FILE``: Stage A‴'s row (A‴-VOID, A‴-FAIL or A‴-PASS),
  counting a G-req seed that fails R1, R2 or A‴ once.
* ``--stage bd --runs DIR``: Stage BD's counts, the bar from ik's rate, the deferral and rate
  measures (reported).
* ``--stage bs --runs DIR --bd-summary FILE --r-summary FILE --a-summary FILE``: Stage BS's row
  (§6.4, first match), the reported tests, the determinism re-run and the job split's checks.
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

from embodied_jepa import grasp_corpus_wm as gc  # noqa: E402
from embodied_jepa import grasp_rate_aware as ra  # noqa: E402
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


def _phase_share(eps):
    """TASK-089's executed grasp share per subgoal phase (TASK-091's summarizer, copied)."""
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
        if r.get("task") != ra.TASK:
            problems.append(f"{name}: not a TASK-092 report")
        if not r.get("completed") or r.get("cap_hit"):
            problems.append(f"{name}: did not complete under the cap")
        if r.get("frozen_files_match") is False:
            problems.append(f"{name}: frozen files differed")
        if n_expected is not None and r.get("n", n_expected) != n_expected:
            problems.append(f"{name}: n = {r.get('n')}")
    return problems


# ----- Stage T (§6.1) --------------------------------------------------------------------------
def stage_t(args):
    problems, rows = [], {}
    for seed in ra.MODEL_SEEDS:
        folder = Path(args.runs) / f"G-req-{seed}"
        if not (folder / "report.json").exists():
            problems.append(f"G-req-{seed}: no report (incomplete or not run)")
            continue
        r = load(folder / "report.json")
        bad = []
        if r.get("outcome") != "T-JOB-DONE" or r.get("kind") != "job":
            bad.append("not a finished job")
        if r.get("tracked_tree_dirty"):
            bad.append("dirty tree")
        if r.get("implementation_sha256") != ra.IMPLEMENTATION_SHA256:
            bad.append("implementation hash")
        if r.get("features_report_sha256") != ra.FEATURE_REPORT_SHA256:
            bad.append("feature report")
        if r.get("moments", {}).get("sha256") != ra.MOMENTS_SHA256:
            bad.append("moments")
        if (r.get("p"), r.get("augment_salt")) != (ra.AUG_P, ra.SALTS["augment"]):
            bad.append("augmentation")
        if r.get("training", {}).get("seconds", 0) > ra.JOB_CAP_SECONDS:
            bad.append("cap")
        problems += [f"G-req-{seed}: {b}" for b in bad]
        t = r.get("training", {})
        rows[f"G-req-{seed}"] = {
            "checkpoint_sha256": r.get("checkpoint_sha256"),
            "modules_sha256": r.get("modules_sha256"),
            "kept_update": t.get("kept_update"),
            "kept_val_criterion": t.get("kept_val_criterion"),
            "last_two_triggered": t.get("last_two_triggered"),
            "seconds": t.get("seconds"),
            "augmentation": r.get("augmentation"),
            "revision": r.get("revision"),
        }
    revisions = {v["revision"] for v in rows.values()}
    if len(revisions) > 1:
        problems.append(f"revisions differ: {sorted(revisions)}")
    return {"stage": "t", "row": "T-VOID" if problems else "T-DONE", "models": rows,
            "problems": problems}  # fmt: skip


# ----- Stage R (§6.2) --------------------------------------------------------------------------
def stage_r(args):
    rep = load(Path(args.runs) / (args.r_file or "stage-r.json"))
    problems = common_checks([rep], rep.get("cohort"))
    debug = bool(rep.get("debug"))
    if not debug and rep.get("split") != "test":
        problems.append("Stage R did not read play-v1 test")
    for name, info in rep.get("models", {}).items():
        pin = ra.CHECKPOINT_SHA256.get(name)
        if pin is not None and info["checkpoint_sha256"] != pin:
            problems.append(f"{name}: not the pinned checkpoint")
        if info.get("moments_sha256") not in (None, ra.MOMENTS_SHA256):
            problems.append(f"{name}: moments")
    row = rep["row"]
    if problems and row["row"] != "R-VOID":
        row = dict(row, row="R-VOID", void_reason="; ".join(problems))
    per_seed = {}
    for s, v in rep["per_seed"].items():
        per_seed[s] = {
            "rho_ctrl": round(v["rho_ctrl"]["value"], 3),
            "rho_ctrl_ci": [round(x, 3) for x in v["rho_ctrl"]["ci"]],
            "r1_bar": round(v["r1_bar"], 3),
            "rho_req": round(v["rho_req"]["value"], 3),
            "rho_req_upper": round(v["rho_req"]["ci"][1], 3),
            "r1": v["r1"],
            "r2_latent_upper": round(v["r2_latent"]["ci"][1], 4),
            "r2_latent_bar": round(v["r2_latent_bar"], 4),
            "r2_hand_upper": round(v["r2_hand"]["ci"][1], 3),
            "r2_hand_bar": round(v["r2_hand_bar"], 3),
            "r2": v["r2"],
            "e": v["e"],
            "sat_req_over_exec_ctrl": round(v["sat_req_over_exec_ctrl"]["value"], 3),
            "r_pass": v["r_pass"],
        }
    return {
        "stage": "r",
        "split": rep.get("split"),
        "debug": debug,
        "row": row["row"],
        "r_row": row,
        "clause_fires": False,
        "delta_L": rep["delta_L"]["delta"],
        "delta_H": rep["delta_H"]["delta"],
        "counts": rep["counts"],
        "per_seed": per_seed,
        "r_pass": {s: v["r_pass"] for s, v in rep["per_seed"].items()},
        "problems": problems,
        "revision": rep.get("revision"),
    }


# ----- Stage A‴ (§6.3) -------------------------------------------------------------------------
def _wins(v, fam, h, kind="model"):
    return {c: int(sum(w)) for c, w in v[fam][kind][h].items()}


def stage_a(args):
    cohort = args.cohort or "a3-roots"
    rep = load(Path(args.runs) / f"a-third-{cohort}.json")
    roots = load(Path(args.runs) / f"roots-{cohort}.json")
    r_sum = load(args.r_summary)
    n = None if cohort.startswith("debug") else len(ra.COHORTS[cohort])
    problems = common_checks([rep, roots], cohort, n)
    if rep.get("first_command_mismatches"):
        problems.append("a plan's first command differs from first_planned")
    if rep.get("a1_bars") != gc.A1_BARS:
        problems.append("A1's bars are not A-K0's frozen ones")
    if len(rep.get("results", {})) != len(ra.a3_models()):
        problems.append("not every model of §5 ran")
    if not cohort.startswith("debug"):
        if r_sum.get("row") != "R-PASS" or r_sum.get("problems"):
            problems.append("Stage R did not pass cleanly")
        if r_sum.get("revision") != rep.get("revision"):
            problems.append("Stage R ran at another revision")
    for name, v in rep.get("results", {}).items():
        pin = ra.CHECKPOINT_SHA256.get(name)
        if pin is not None and v["model"]["checkpoint_sha256"] != pin:
            problems.append(f"{name}: not the pinned checkpoint")
    a3 = {}
    for s in ra.MODEL_SEEDS:
        v = rep.get("results", {}).get(ra.model_name("G-req", s))
        a3[s] = bool(v and v["a3_pass"])
    r_pass = {int(s): bool(v) for s, v in r_sum.get("r_pass", {}).items()}
    row = ra.a3_row(r_pass, a3, void="; ".join(problems))
    per_model = {}
    for name, v in rep.get("results", {}).items():
        a2 = v["a2"]
        exe = {k: np.asarray(x).mean(0).round(3).tolist() for k, x in
               v["executed_plan_grasp"].items()}  # fmt: skip
        pred = {k: np.asarray(x).mean(0).round(3).tolist() for k, x in
                v["predicted_hand_synergy"].items()}  # fmt: skip
        per_model[name] = {
            "a1_h8_pass": v["a1_h8"]["pass"],
            "a1_h8_wins": _wins(v, "latent+pose", "8"),
            "a1_h8_shuffled": _wins(v, "latent+pose", "8", "shuffled"),
            "a1_h16_wins": _wins(v, "latent+pose", "16"),
            "latent_wins_h8": _wins(v, "latent", "8"),
            "latent_wins_h16": _wins(v, "latent", "16"),
            "close_le4_R1": a2["close_le4_R1"]["count"],
            "lift_R2": a2["lift_R2"]["count"],
            "close_R1_reported": a2["close_R1_reported"],
            "close_by_R1": a2["close_by_R1"],
            "never_close_R1": a2["never_close_R1"],
            "a3_pass": v["a3_pass"],
            "last_two_triggered": v["model"].get("last_two_triggered"),
            "probe": v.get("probe"),
            "mean_executed_plan_grasp": exe,
            "mean_predicted_hand_synergy": pred,
        }
    ceiling = {
        fam: {h: {c: int(sum(w)) for c, w in by_c.items()} for h, by_c in by_h.items()}
        for fam, by_h in rep.get("ceiling", {}).items()
    }
    return {
        "stage": "a",
        "cohort": cohort,
        "n": rep.get("n"),
        "row": row["row"],
        "a3_row": row,
        "a3_pass": {str(k): v for k, v in a3.items()},
        "r_pass": {str(k): v for k, v in r_pass.items()},
        "clause_fires": False,
        "per_model": per_model,
        "ceiling_wins": ceiling,
        "probe": rep.get("probe"),
        "probe_positives": rep.get("probe_positives"),
        "lift_branch_rise_ge5_16": rep.get("lift_branch_rise_ge5_16"),
        "rejected_draws": roots.get("rejections"),
        "targets": _count([r["target"] for r in roots["roots"]]),
        "problems": problems,
        "revision": rep.get("revision"),
    }


# ----- the closed loop (§6.4) ------------------------------------------------------------------
def _runs(d, cohort):
    runs, parts = {}, {}
    for f in sorted((Path(d) / cohort).glob("*.json")):
        (parts if ".part" in f.stem else runs)[f.stem] = load(f)
    return runs, parts


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


def _deferral_and_rate(runs):
    out = {}
    for name, r in runs.items():
        if r.get("arm") not in ra.CLOSED_LOOP_ARMS or r.get("determinism_rerun"):
            continue
        eps = r["episodes"]
        ms = [e["deferral"] for e in eps if "deferral" in e]
        rm = [e["rate"] for e in eps if "rate" in e]
        if len(ms) != len(eps) or len(rm) != len(eps):
            out[name] = {"missing": len(eps) - min(len(ms), len(rm))}
            continue
        succ = [bool(e["success"]) for e in eps]
        out[name] = {
            "deferral": ra.deferral_summary(ms),
            "rate": ra.rate_summary(rm, succ),
        }
    return out


def _split_checks(runs, parts, cohort, debug):
    problems = []
    for name, r in runs.items():
        if r.get("arm") not in ra.CLOSED_LOOP_ARMS or r.get("determinism_rerun"):
            continue
        mine = {k: v for k, v in parts.items() if k.split(".part")[0] == name}
        need = ra.needs_parts(cohort, r["arm"])
        if need and not r.get("combined"):
            problems.append(f"{name}: not combined from its {need} jobs")
        if r.get("combined"):
            if not debug and len(r["parts"]) != ra.BS_PARTS:
                problems.append(f"{name}: {len(r['parts'])} parts, not {ra.BS_PARTS}")
            if len(mine) != len(r["parts"]):
                problems.append(f"{name}: part files and combined parts differ")
            try:
                eps = ra.combine_parts(list(mine.values()), [e["seed"] for e in r["episodes"]])
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


def stage_bd(args):
    cohort = args.cohort or "bd-grasp"
    debug = cohort.startswith("debug")
    allr, parts = _runs(args.runs, cohort)
    runs = {k: v for k, v in allr.items() if not v.get("determinism_rerun")}
    if debug and args.only:
        runs = {k: v for k, v in runs.items() if k in args.only}
    n = runs["ik"]["n"] if debug else len(ra.COHORTS[cohort])
    problems = common_checks(list(runs.values()), cohort, n)
    problems += _split_checks(runs, parts, cohort, debug)
    if not debug:
        for r in runs.values():
            if [e["seed"] for e in r["episodes"]] != list(ra.COHORTS[cohort]):
                problems.append(f"{r['arm']}: not the cohort's seeds")
        need = {f"{a}-{s}" for a in ra.CLOSED_LOOP_ARMS for s in ra.BD_SEEDS[a]}
        need |= set(ra.BASELINE_ARMS)
        if need - set(runs):
            problems.append(f"missing runs: {sorted(need - set(runs))}")
    ik = runs["ik"]["successes"]
    bar = ra.b_bar(ik, n)
    g = load(Path(args.runs) / f"goals-{cohort}.json")
    return {
        "stage": "bd",
        "cohort": cohort,
        "n": n,
        "arms": _arm_table(runs),
        "deferral_and_rate": _deferral_and_rate(runs),
        "ik_successes": ik,
        "bar": bar,
        "pooled_bar_at_bs": None if bar is None else ra.pooled_bar(bar, ra.BS_N),
        "seed_floor_at_bs": None if bar is None else ra.seed_floor(bar, ra.BS_N),
        "escalate": bar is None,
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
    n = runs["ik"]["n"] if debug else len(ra.COHORTS[cohort])
    bd = load(args.bd_summary)
    problems = common_checks(list(allr.values()), cohort, None)
    problems += _split_checks(runs, parts, cohort, debug)
    for r in runs.values():
        if r["n"] != n:
            problems.append(f"{r['arm']}: n = {r['n']}")
    if not debug:
        if bd.get("cohort") != "bd-grasp" or bd.get("problems") or bd.get("stage") != "bd":
            problems.append("the BD summary is not a clean Stage BD summary")
        if runs and bd.get("revision") != next(iter(runs.values()))["revision"]:
            problems.append("Stage BD ran at another revision")
        for r in runs.values():
            if [e["seed"] for e in r["episodes"]] != list(ra.COHORTS[cohort]):
                problems.append(f"{r['arm']}: not the cohort's seeds")
    seeds = args.model_seeds or list(ra.MODEL_SEEDS)
    need = {f"M-req-{s}" for s in seeds} | {f"M-v1-{s}" for s in ra.MODEL_SEEDS}
    need |= set(ra.BASELINE_ARMS)
    missing = need - set(runs)
    if missing:
        problems.append(f"missing runs: {sorted(missing)}")
    if bd["bar"] is None:
        problems.append("Stage BD set no bar")
    det_ok = None
    key = f"M-req-{seeds[0]}"
    if det:
        d = next(iter(det.values()))
        first = {e["seed"]: e for e in runs.get(key, {"episodes": []})["episodes"]}
        det_ok = (
            all(
                e["applied"] == first.get(e["seed"], {}).get("applied")
                and e["success"] == first.get(e["seed"], {}).get("success")
                for e in d["episodes"]
            )
            and len(d["episodes"]) == ra.DETERMINISM[2]
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
    g = np.stack([succ[f"M-req-{s}"] for s in seeds])
    c = np.stack([succ[f"M-v1-{s}"] for s in ra.MODEL_SEEDS])
    out = ra.b_row(g, c, succ["random"], bd["bar"] or 1.0, void="; ".join(problems))
    reported = {
        "ik vs M-req (per reset, ik x 3)": {
            "p_signflip": ra.signflip_p(3 * succ["ik"].astype(int) - ra.per_reset(g))
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
    if args.r_summary:
        rs = load(args.r_summary)
        beside["stage_r"] = {"row": rs.get("row"), "per_seed": rs.get("per_seed")}
    if args.a_summary:
        asum = load(args.a_summary)
        beside["stage_a3"] = {
            "row": asum.get("row"),
            "per_seed": {
                str(s): asum["per_model"].get(ra.model_name("G-req", s), {}).get("a3_pass")
                for s in ra.MODEL_SEEDS
            },
        }
    return {
        "stage": "bs",
        "cohort": cohort,
        "n": n,
        "row": out["row"],
        "row_detail": out,
        "clause_fires": ra.clause_fires(out["row"]),
        "lewm_needed_claimed": False,
        "bar": bd["bar"],
        "determinism_matches": det_ok,
        "arms": _arm_table(runs),
        "deferral_and_rate": _deferral_and_rate(runs),
        "beside_each_seed": beside,
        "reported_tests": reported,
        "per_object": per_obj,
        "targets": _count(targets.values()),
        "rejected_draws": gl["rejections"],
        "problems": problems,
        "revision": next(iter(runs.values()))["revision"],
    }


# ----- Stage 0's timing (§13) ------------------------------------------------------------------
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
        if r.get("arm") in ra.CLOSED_LOOP_ARMS:
            full = zs.BUDGET["grasp"]
            job64 = ra.project_job_seconds(per_command_wall, full, 64, 6)
            run128 = ra.project_job_seconds(per_command_wall, full, ra.BS_N, 6)
            row |= {
                "plan_seconds_per_command": float(
                    np.mean([e["plan_seconds_mean"] for e in eps if "plan_seconds_mean" in e])
                ),
                "projected_64_reset_job_seconds": job64,
                "projected_128_reset_run_seconds": run128,
                "parts_needed": ra.parts_needed(run128),
            }
        rows[name] = row
    wm = [v for v in rows.values() if "projected_64_reset_job_seconds" in v]
    worst = max((v["projected_64_reset_job_seconds"] for v in wm), default=None)
    return {
        "stage": "timing",
        "cohort": cohort,
        "runs": rows,
        "worst_projected_64_reset_job_seconds": worst,
        "split_trigger_seconds": ra.SPLIT_TRIGGER_SECONDS,
        "bs_parts": max([ra.BS_PARTS] + [v["parts_needed"] for v in wm]),
        "note": "debug timing; projections assume every episode runs the full 400 commands",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("t", "r", "a", "bd", "bs", "timing"))
    parser.add_argument("--runs", required=True)
    parser.add_argument("--r-file", help="Stage R: the report's file name (default stage-r.json)")
    parser.add_argument("--r-summary")
    parser.add_argument("--a-summary")
    parser.add_argument("--bd-summary")
    parser.add_argument("--cohort", help="debug cohorts only (smoke)")
    parser.add_argument("--model-seeds", type=int, nargs="+", help="debug only")
    parser.add_argument("--only", nargs="+", help="debug only: the runs to summarise")
    parser.add_argument("--name", help="debug only: the summary's file name suffix")
    args = parser.parse_args(argv)
    if args.cohort and not args.cohort.startswith("debug"):
        raise SystemExit("--cohort is for debug cohorts only")
    if (args.model_seeds or args.only or args.name) and not (args.cohort or args.r_file):
        raise SystemExit("--model-seeds, --only and --name are for debug runs only")
    if args.stage == "timing" and not args.cohort:
        raise SystemExit("timing is Stage 0's, on a debug cohort")
    if args.stage == "a" and not args.r_summary:
        raise SystemExit("Stage A‴ needs Stage R's summary (§7: R and A‴ failures count together)")
    if args.stage == "bs" and not args.bd_summary:
        raise SystemExit("Stage BS needs the Stage BD summary")
    if args.stage == "bs" and not args.cohort and not (args.a_summary and args.r_summary):
        raise SystemExit("Stage BS needs Stage R's and A‴'s summaries (reported beside each seed)")
    fn = {"t": stage_t, "r": stage_r, "a": stage_a, "bd": stage_bd, "bs": stage_bs,
          "timing": stage_timing}[args.stage]  # fmt: skip
    summary = fn(args)
    suffix = args.name or (args.cohort or "")
    name = f"summary-{args.stage}{'-' + suffix if suffix else ''}.json"
    path = Path(args.runs) / name
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.write_text(json.dumps(summary, indent=1, sort_keys=True, allow_nan=True) + "\n")
    print(json.dumps({k: summary.get(k) for k in ("stage", "row", "bar", "problems")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
