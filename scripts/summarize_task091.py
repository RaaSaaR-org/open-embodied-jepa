"""TASK-091: summarise a stage (protocol docs/experiments/grasp_play_corpus_v1_planned_close.md).

* ``--stage a --runs DIR``: Stage A″'s row (A″-VOID, A″-FAIL or A″-PASS) with its checks.
* ``--stage bd --runs DIR``: Stage BD's counts, the bar from ik's rate (TASK-090 §7.2) and the
  deferral measures (§5.3, reported).
* ``--stage bs --runs DIR --bd-summary FILE``: Stage BS's row (TASK-090's rows, first match), the
  reported tests, the determinism re-run, the job split's checks and the deferral measures.
* ``--stage timing --runs DIR --cohort debug-grasp``: Stage 0's per-command timing and the
  projected job times and split (§5.4, §12).

Writes ``DIR/summary-<stage>.json`` (refuses an existing one) and prints it. A run file that did
not complete (a cap hit), a BS world-model run that is not the combination of its parts, or a
part that is missing makes the stage VOID.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import grasp_corpus_wm as gc  # noqa: E402
from embodied_jepa import grasp_corpus_wm_planned_close as pcl  # noqa: E402
from embodied_jepa import zero_shot as zs  # noqa: E402


def _frozen_summarizer():
    bad = [k for k, ok in pcl.frozen_files_check(ROOT).items() if not ok]
    if bad:
        raise SystemExit(f"these frozen files differ from their pins: {bad}")
    spec = importlib.util.spec_from_file_location(
        "summarize_task090_frozen", ROOT / "scripts" / "summarize_task090.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s090 = _frozen_summarizer()
load, interval, _count = s090.load, s090.interval, s090._count


def common_checks(reports, cohort, n_expected=None):
    problems = s090.common_checks(reports, cohort, n_expected)
    for r in reports:
        if r.get("task") != pcl.TASK:
            problems.append(f"{r.get('arm', 'report')}: not a TASK-091 report")
        if not r.get("completed") or r.get("cap_hit"):
            problems.append(f"{r.get('arm', 'report')}: did not complete under the cap")
        if r.get("frozen_files_match") is False:
            problems.append(f"{r.get('arm', 'report')}: frozen files differed")
    return problems


# ----- Stage A″ (§4) ---------------------------------------------------------------------------
def stage_a(args):
    cohort = args.cohort or "a2-roots"
    rep = load(Path(args.runs) / f"a-second-{cohort}.json")
    roots = load(Path(args.runs) / f"roots-{cohort}.json")
    n = None if cohort.startswith("debug") else len(pcl.COHORTS[cohort])
    problems = common_checks([rep, roots], cohort, n)
    if rep.get("first_command_mismatches"):
        problems.append("a plan's first command differs from first_planned")
    if rep.get("feature_reports_sha256") != pcl.FEATURE_REPORTS_SHA256:
        problems.append("feature reports are not the pinned ones")
    if rep.get("a1_bars") != gc.A1_BARS:
        problems.append("A1's bars are not A-K0's frozen ones")
    for name, v in rep.get("results", {}).items():
        if v["model"]["checkpoint_sha256"] != pcl.CHECKPOINT_SHA256.get(name):
            problems.append(f"{name}: not the pinned checkpoint")
    if len(rep.get("results", {})) != len(pcl.a_second_models()):
        problems.append("not every model of §3 ran")
    passes = {}
    if not problems:
        passes = {
            tag: {s: rep["results"][f"G-{tag}-{s}"]["a_second_pass"] for s in gc.MODEL_SEEDS}
            for tag in ("v2g", "v1p")
        }
    row = pcl.a_second_row(passes, void="; ".join(problems))
    per_model = {}
    for name, v in rep.get("results", {}).items():
        if "a2" not in v:
            per_model[name] = {"latent_wins_h8": _wins(v, "latent", "8")}
            continue
        a2 = v["a2"]
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
            "first_close_step_R1": _count([str(k) for k in a2["first_close_step_R1"]]),
            "mean_grasp_per_step_R1": [round(x, 3) for x in a2["mean_grasp_per_step_R1"]],
            "a_second_pass": v["a_second_pass"],
            "last_two_triggered": v["model"].get("last_two_triggered"),
            "probe": v.get("probe"),
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
        "a_second": row,
        "clause_fires": False,  # §7: A″ fires nothing
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


def _wins(v, fam, h, kind="model"):
    return {c: int(sum(w)) for c, w in v[fam][kind][h].items()}


# ----- the closed loop (§5) --------------------------------------------------------------------
def _runs(d, cohort):
    """The cohort's run files: combined or unsplit runs by name, and the part files apart."""
    runs, parts = {}, {}
    for f in sorted((Path(d) / cohort).glob("*.json")):
        if ".part" in f.stem:
            parts[f.stem] = load(f)
        else:
            runs[f.stem] = load(f)
    return runs, parts


def _deferral(runs):
    out = {}
    for name, r in runs.items():
        if r.get("arm") not in pcl.CLOSED_LOOP_ARMS:
            continue
        ms = [e["deferral"] for e in r["episodes"] if "deferral" in e]
        if len(ms) != len(r["episodes"]):
            out[name] = {"missing": len(r["episodes"]) - len(ms)}
            continue
        succ = [e["success"] for e in r["episodes"]]
        out[name] = pcl.deferral_summary(ms) | {
            "successes": int(sum(succ)),
            "per_reset_m3_share_k1": [m["m3_share_k1"] for m in ms],
            "per_reset_m3_longest_run": [m["m3_longest_run_k_gt_1"] for m in ms],
            "per_reset_m2": [m["m2_first_close_in_grasp_phase"] for m in ms],
            "per_reset_m1": [m["m1_closing_before_contact"] for m in ms],
        }
    return out


def _split_checks(runs, parts, cohort, debug):
    """§5.4: every BS world-model run is the combination of exactly its parts; no part hit the
    cap; no part exists without its combined run."""
    problems = []
    for name, r in runs.items():
        if r.get("arm") not in pcl.CLOSED_LOOP_ARMS or r.get("determinism_rerun"):
            continue
        mine = {k: v for k, v in parts.items() if k.split(".part")[0] == name}
        need = pcl.needs_parts(cohort, r["arm"])
        if need and not r.get("combined"):
            problems.append(f"{name}: not combined from its {need} jobs")
        if r.get("combined"):
            if not debug and len(r["parts"]) != pcl.BS_PARTS:
                problems.append(f"{name}: {len(r['parts'])} parts, not {pcl.BS_PARTS}")
            if len(mine) != len(r["parts"]):
                problems.append(f"{name}: part files and combined parts differ")
            for k, p in mine.items():
                if not p.get("completed") or p.get("cap_hit"):
                    problems.append(f"{k}: did not complete under the cap")
            try:
                eps = pcl.combine_parts(list(mine.values()), [e["seed"] for e in r["episodes"]])
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


def _arm_table(runs):
    return s090._arm_table(
        {k: v for k, v in runs.items() if not v.get("determinism_rerun")}
    )  # fmt: skip


def stage_bd(args):
    cohort = args.cohort or "bd-grasp"
    debug = cohort.startswith("debug")
    allr, parts = _runs(args.runs, cohort)
    runs = {k: v for k, v in allr.items() if not v.get("determinism_rerun")}
    if debug and args.only:
        runs = {k: v for k, v in runs.items() if k in args.only}
    n = runs["ik"]["n"] if debug else len(pcl.COHORTS[cohort])
    problems = common_checks(list(runs.values()), cohort, n)
    problems += _split_checks(runs, parts, cohort, debug)
    if not debug:
        for r in runs.values():
            if [e["seed"] for e in r["episodes"]] != list(pcl.COHORTS[cohort]):
                problems.append(f"{r['arm']}: not the cohort's seeds")
        need = {f"{a}-{s}" for a in pcl.CLOSED_LOOP_ARMS for s in pcl.BD_SEEDS[a]}
        need |= set(pcl.BASELINE_ARMS)
        if need - set(runs):
            problems.append(f"missing runs: {sorted(need - set(runs))}")
    ik = runs["ik"]["successes"]
    bar = pcl.b_bar(ik, n)
    g = load(Path(args.runs) / f"goals-{cohort}.json")
    return {
        "stage": "bd",
        "cohort": cohort,
        "n": n,
        "arms": _arm_table(runs),
        "deferral": _deferral(runs),
        "ik_successes": ik,
        "bar": bar,
        "pooled_bar_at_bs": None if bar is None else pcl.pooled_bar(bar, pcl.BS_N),
        "seed_floor_at_bs": None if bar is None else pcl.seed_floor(bar, pcl.BS_N),
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
    n = runs["ik"]["n"] if debug else len(pcl.COHORTS[cohort])
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
            if [e["seed"] for e in r["episodes"]] != list(pcl.COHORTS[cohort]):
                problems.append(f"{r['arm']}: not the cohort's seeds")
    seeds = args.model_seeds or list(gc.MODEL_SEEDS)
    v1_seeds = list(pcl.BS_SEEDS["M-v1"])
    need = {f"M-v2g-{s}" for s in seeds} | {f"M-v1p-{s}" for s in seeds}
    need |= {f"M-v1-{s}" for s in v1_seeds} | set(pcl.BASELINE_ARMS)
    missing = need - set(runs)
    if missing:
        problems.append(f"missing runs: {sorted(missing)}")
    if bd["bar"] is None:
        problems.append("Stage BD set no bar")
    # the determinism re-run: the first four resets of M-v2g-90100 must match bitwise (§5.1)
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
            and len(d["episodes"]) == pcl.DETERMINISM[2]
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
    g = np.stack([succ[f"M-v2g-{s}"] for s in seeds])
    c = np.stack([succ[f"M-v1p-{s}"] for s in seeds])
    out = pcl.b_row(g, c, succ["random"], bd["bar"] or 1.0, void="; ".join(problems))
    reported = {}
    if all(f"M-v1-{s}" in succ for s in v1_seeds):
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
        "clause_fires": pcl.clause_fires(out["row"]),
        "lewm_needed_claimed": False,
        "bar": bd["bar"],
        "determinism_matches": det_ok,
        "arms": _arm_table(runs),
        "deferral": _deferral(runs),
        "a_second_per_seed": _a_per_seed(args.a_summary, seeds),
        "reported_tests": reported,
        "per_object": per_obj,
        "targets": _count(targets.values()),
        "rejected_draws": gl["rejections"],
        "problems": problems,
        "revision": next(iter(runs.values()))["revision"],
    }


def _a_per_seed(path, seeds):
    """§4: each gated seed's A″ result, reported beside its closed-loop count."""
    if not path:
        return None
    a = load(path)
    out = {}
    for tag in ("v2g", "v1p"):
        for s in seeds:
            m = a["per_model"].get(f"G-{tag}-{s}", {})
            out[f"M-{tag}-{s}"] = {
                "a_second_pass": m.get("a_second_pass"),
                "close_le4_R1": m.get("close_le4_R1"),
                "close_R1_reported": m.get("close_R1_reported"),
                "lift_R2": m.get("lift_R2"),
            }
    return {"row": a.get("row"), "per_seed": out}


# ----- Stage 0's timing and the job split (§5.4, §12) ------------------------------------------
def stage_timing(args):
    cohort = args.cohort
    runs, parts = _runs(args.runs, cohort)
    runs |= parts
    rows = {}
    for name, r in runs.items():
        if r.get("arm") not in pcl.CLOSED_LOOP_ARMS or not r.get("completed"):
            continue
        eps = r["episodes"]
        commands = sum(e["steps"] for e in eps)
        procs = min(r["workers"], len(eps)) if isinstance(r["workers"], int) else None
        if not procs or not commands:
            continue
        per_command_wall = r["seconds"] * procs / commands
        plan = float(np.mean([e["plan_seconds_mean"] for e in eps if "plan_seconds_mean" in e]))
        full = zs.BUDGET["grasp"]
        # a 64-reset job at 6 processes, every episode at the full 400-command budget (an upper
        # bound: successes end early), and the unsplit 128-reset run
        job64 = pcl.project_job_seconds(per_command_wall, full, 64, 6)
        run128 = pcl.project_job_seconds(per_command_wall, full, pcl.BS_N, 6)
        rows[name] = {
            "processes": procs,
            "episodes": len(eps),
            "commands": commands,
            "seconds": r["seconds"],
            "wall_seconds_per_command_per_process": per_command_wall,
            "plan_seconds_per_command": plan,
            "projected_64_reset_job_seconds": job64,
            "projected_128_reset_run_seconds": run128,
            "parts_needed": pcl.parts_needed(run128),
        }
    worst = max((v["projected_64_reset_job_seconds"] for v in rows.values()), default=None)
    return {
        "stage": "timing",
        "cohort": cohort,
        "runs": rows,
        "worst_projected_64_reset_job_seconds": worst,
        "split_trigger_seconds": pcl.SPLIT_TRIGGER_SECONDS,
        "bs_parts": max([pcl.BS_PARTS] + [v["parts_needed"] for v in rows.values()]),
        "note": "debug timing; projections assume every episode runs the full 400 commands",
        "basis_task089": "TASK-089's 64-reset world-model runs took 3 623-4 321 s (R27.25)",
        "projected_bs_gpu_hours": None if worst is None else round(7 * 2 * worst / 3600 + 0.25, 1),
        "seconds_per_part_job": None if worst is None else math.ceil(worst),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("a", "bd", "bs", "timing"))
    parser.add_argument("--runs", required=True)
    parser.add_argument("--bd-summary")
    parser.add_argument("--a-summary", help="Stage BS: Stage A″'s summary (per-seed results)")
    parser.add_argument("--cohort", help="debug cohorts only (smoke)")
    parser.add_argument("--model-seeds", type=int, nargs="+", help="debug only")
    parser.add_argument("--only", nargs="+", help="debug only: the runs to summarise")
    args = parser.parse_args(argv)
    if args.cohort and not args.cohort.startswith("debug"):
        raise SystemExit("--cohort is for debug cohorts only")
    if (args.model_seeds or args.only) and not args.cohort:
        raise SystemExit("--model-seeds and --only are for debug cohorts only")
    if args.stage == "timing" and not args.cohort:
        raise SystemExit("timing is Stage 0's, on a debug cohort")
    if args.stage == "bs" and not args.bd_summary:
        raise SystemExit("Stage BS needs the Stage BD summary")
    if args.stage == "bs" and not args.cohort and not args.a_summary:
        raise SystemExit("Stage BS needs Stage A″'s summary (§4: per-seed results reported)")
    fn = {"a": stage_a, "bd": stage_bd, "bs": stage_bs, "timing": stage_timing}[args.stage]
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
