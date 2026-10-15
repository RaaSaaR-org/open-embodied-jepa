"""TASK-094: summarise a stage (protocol docs/experiments/grasp_closure_locus_v1.md).

* ``--stage 0 --runs DIR --debug-runs DIR --gpu-log FILE``: Stage 0's row (S0-ESCALATE or S0-PASS,
  §14): the hash checks (``verify-task094.json``), the seed check (``seeds.json``), development K
  on 94100-94131 (every arm's plumbing, K-1.0's and K-open's counts against the band's Stage 0
  proportions, K-1.0 bitwise the frozen ``ik``; every count reported), development M and S on
  940000-940031 (the bitwise root re-runs and the squeeze's reproduction, the executed-difference
  check, the positive control at 28 / 32 per root type, n_d at each root type, G's wins and S's
  measures, reported), the power rule (``power.json``), the debug smoke, Stage 0's GPU time from
  ``gpu_run``'s log and the gated stage's projection.
* ``--stage gated --runs DIR --s0-summary FILE``: the gated stage's rows: K's row (§4.3), M's and
  S's readings per root type, the L row (§7.1, read only after K-CAUSAL) and the task row, with
  every K arm, the contrasts, the determinism re-runs and the term decomposition.

Writes ``DIR/summary-<stage>.json`` (refuses an existing one) and prints it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import grasp_closure_locus as gl  # noqa: E402
from embodied_jepa import zero_shot as zs  # noqa: E402


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
    revisions = {r.get("revision") for r in reports}
    if len(revisions) != 1:
        problems.append(f"revisions differ: {sorted(map(str, revisions))}")
    debug = str(cohort).startswith("debug")
    for r in reports:
        name = r.get("arm") or r.get("argv", ["?", "?"])[1:2]
        if not debug and r.get("tracked_tree_dirty"):
            problems.append(f"{name}: made from a dirty tree")
        if r.get("task") != gl.TASK:
            problems.append(f"{name}: not a TASK-094 report")
        if not r.get("completed") or r.get("cap_hit"):
            problems.append(f"{name}: did not complete under the cap")
        if r.get("frozen_files_match") is False:
            problems.append(f"{name}: frozen files differed")
        if n_expected is not None and r.get("n", n_expected) != n_expected:
            problems.append(f"{name}: n = {r.get('n')}")
    return problems


# ----- K (§4) ----------------------------------------------------------------------------------
def k_runs(d, cohort):
    runs, det = {}, {}
    for f in sorted((Path(d) / cohort).glob("*.json")):
        rep = load(f)
        (det if rep.get("determinism_rerun") else runs)[rep["arm"]] = rep
    return runs, det


def _median(values):
    v = [x for x in values if x is not None]
    return float(np.median(v)) if v else None


def arm_table(runs):
    table = {}
    for arm, r in runs.items():
        eps = r["episodes"]
        n = len(eps)
        k = int(sum(e["success"] for e in eps))
        row = {
            "successes": k,
            "n": n,
            "interval": interval(k, n),
            "grasp_contact": int(sum(e["any_grasp_contact"] for e in eps)),
            "max_rise_median_m": float(np.median([e["max_rise"] for e in eps])),
            "stopped": int(sum(bool(e["stop_reason"]) for e in eps)),
            "final_subgoal": _count([e["final_subgoal"] for e in eps]),
            "switch_reasons": _count(
                [f"{i}:{s['reason']}" for e in eps for i, s in enumerate(e["switches"])]
            ),
        }
        if all("measures" in e for e in eps):
            m = [e["measures"] for e in eps]
            row |= {
                "slip": int(sum(x["slip"] for x in m)),
                "measured_synergy_closing_median": _median(
                    [x["measured_synergy_closing_median"] for x in m]
                ),
                "measured_synergy_grasp_phase_last_median": _median(
                    [x["measured_synergy_grasp_phase_last"] for x in m]
                ),
                "level_median": float(np.median([e["level"] for e in eps])),
                "plumbing_ok": bool(all(gl.replumb(e)["ok"] for e in eps)),
                "plumbing_job_ok": bool(all(e["plumbing"]["ok"] for e in eps)),
                "applied_reaches_c_after_max": int(
                    max(gl.replumb(e)["applied_reaches_c_after"] for e in eps)
                ),
                "path_deviation_max": float(max(gl.replumb(e)["path_deviation_max"] for e in eps)),
            }
            if any(e.get("displacement_ok") is not None for e in eps):
                row["displacement_ok"] = bool(all(e["displacement_ok"] for e in eps))
        table[arm] = row
    return table


def k_successes(runs, seeds):
    out = {}
    for arm, r in runs.items():
        by = {e["seed"]: bool(e["success"]) for e in r["episodes"]}
        out[arm] = [by[s] for s in seeds]
    return out


def same_applied(a, b) -> bool:
    x, y = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return x.shape == y.shape and bool(np.array_equal(x.view(np.uint32), y.view(np.uint32)))


def k_determinism(runs, det):
    out = {}
    for arm, r in det.items():
        main = {e["seed"]: e for e in runs[arm]["episodes"]} if arm in runs else {}
        out[arm] = {
            str(e["seed"]): bool(
                e["seed"] in main
                and same_applied(e["applied"], main[e["seed"]]["applied"])
                and e["success"] == main[e["seed"]]["success"]
            )
            for e in r["episodes"]
        }
    return out


def k_summary(d, cohort, *, expect_n=None):
    runs, det = k_runs(d, cohort)
    goals = load(Path(d) / f"goals-{cohort}.json")
    seeds = [g["seed"] for g in goals["goals"]]
    problems = common_checks(list(runs.values()) + list(det.values()) + [goals], cohort, expect_n)
    for arm, r in runs.items():
        if r.get("goals_sha256") is None:
            problems.append(f"{arm}: no goal digest record")
        if [e["seed"] for e in r["episodes"]] != seeds:
            problems.append(f"{arm}: not the cohort's goals in order")
    missing = [a for a in gl.K_ARMS if a not in runs]
    # §4.3's plumbing check re-read from the recorded episodes (the summarizer's reading; the
    # job's own reading is reported beside it)
    plumbing = {a: all(gl.replumb(e)["ok"] for e in r["episodes"])
                for a, r in runs.items() if a in gl.K_ARMS}  # fmt: skip
    plumbing_job = {a: r.get("plumbing_ok") for a, r in runs.items() if a in gl.K_ARMS}
    displaced = {a: r.get("displacement_ok") for a, r in runs.items()
                 if a in gl.K_ARMS and gl.K_ARMS[a][1]}  # fmt: skip
    out = {
        "cohort": cohort,
        "n": len(seeds),
        "arms": arm_table(runs),
        "missing_arms": missing,
        "plumbing": plumbing,
        "plumbing_job": plumbing_job,
        "displacement": displaced,
        "determinism": k_determinism(runs, det),
        "s_star": {
            "median": float(np.median([e["s_star"] for e in runs[gl.REFERENCE_ARM]["episodes"]]))
            if gl.REFERENCE_ARM in runs
            else None,
            "p10_p90": np.percentile(
                [e["s_star"] for e in runs[gl.REFERENCE_ARM]["episodes"]], [10, 90]
            ).tolist()
            if gl.REFERENCE_ARM in runs
            else None,
        },
        "rejected_draws": goals.get("rejections"),
        "targets": _count([g["target"] for g in goals["goals"]]),
        "problems": problems,
        "revision": goals.get("revision"),
    }
    if "ik" in runs and gl.REFERENCE_ARM in runs:
        a = {e["seed"]: e for e in runs["ik"]["episodes"]}
        b = {e["seed"]: e for e in runs[gl.REFERENCE_ARM]["episodes"]}
        out["k1_equals_frozen_ik"] = bool(
            all(same_applied(a[s]["applied"], b[s]["applied"]) for s in seeds)
        )
    if not missing:
        plumbing_ok = all(plumbing.values()) and all(v is not False for v in displaced.values())
        det_ok = all(all(v.values()) for v in out["determinism"].values())
        void = "; ".join(problems + ([] if det_ok else ["a determinism re-run differs"]))
        out["k_row"] = gl.k_row(k_successes(runs, seeds), plumbing_ok=plumbing_ok, void=void)
    return out


# ----- M and S (§5, §6) ------------------------------------------------------------------------
def _branch_rows(roots_rep):
    return {r["seed"]: r for r in roots_rep["roots"]}


def ms_summary(d, cohort, *, expect_n=None):
    d = Path(d)
    roots = load(d / f"roots-{cohort}.json")
    rank = load(d / f"rank-{cohort}.json") if (d / f"rank-{cohort}.json").exists() else None
    search = load(d / f"search-{cohort}.json") if (d / f"search-{cohort}.json").exists() else None
    reports = [x for x in (roots, rank, search) if x is not None]
    problems = common_checks(reports, cohort, expect_n)
    problems += roots.get("problems", [])
    for x in (rank, search):
        if x is not None:
            problems += x.get("problems", [])
            if x.get("roots_sha256") is None:
                problems.append("a ranking or search without its roots' digest")
    rows = _branch_rows(roots)
    seeds = [r["seed"] for r in roots["roots"]]
    n = len(seeds)
    branches = roots["branches"]
    per_type, readings_in, details = {}, {}, {}
    for t in gl.ROOT_TYPES:
        lift = {b: np.array([rows[s]["branches"][f"{t}/{b}"]["lift"] for s in seeds])
                for b in branches}  # fmt: skip
        differs = np.array(
            [rows[s]["branches"][f"{t}/{gl.S_STAR}"]["window_differs"]["any_channel"]
             for s in seeds]
        )  # fmt: skip
        include = differs >= gl.MIN_DIFFERING
        excluded = int((~include).sum())
        disc = gl.discordant(lift[gl.SQUEEZE], lift[gl.S_STAR], include)
        n_d = int(disc.sum())
        entry = {
            "n": n,
            "lift_rate": {b: int(v.sum()) for b, v in lift.items()},
            "window_differs_any_channel": _count(differs.tolist()),
            "excluded_below_7": excluded,
            "n_d": n_d,
            "discordant_roots": [s for s, x in zip(seeds, disc, strict=True) if x],
        }
        readings = {"n_d": n_d, "ceiling_wins": 0, "g_wins": [0, 0, 0], "s_misses": [0, 0, 0]}
        if rank is not None:
            ceil = rank["ceiling"]
            cost = {b: np.array([ceil[f"{s}/{t}"][b]["total"] for s in seeds]) for b in branches}
            w = gl.wins(cost[gl.SQUEEZE], cost[gl.S_STAR])
            pc = gl.wins(cost[gl.SQUEEZE], cost[gl.CONTROL])
            entry["ceiling_wins"] = int(w[disc].sum())
            entry["ceiling_rate"] = (entry["ceiling_wins"] / n_d) if n_d else None
            entry["ceiling_interval"] = interval(entry["ceiling_wins"], n_d) if n_d else None
            entry["positive_control"] = int(pc.sum())
            entry["positive_control_need"] = gl.positive_control_need(n)
            entry["ceiling_wins_all_roots"] = int(w.sum())
            g_wins, g_costs = [], {}
            for seed in gl.MODEL_SEEDS:
                res = rank["results"][f"G-v1-{seed}"]["roots"]
                gc_ = {b: np.array([res[f"{s}/{t}"][b]["total"] for s in seeds]) for b in branches}
                g_costs[seed] = gc_
                g_wins.append(int(gl.wins(gc_[gl.SQUEEZE], gc_[gl.S_STAR])[disc].sum()))
            entry["g_wins"] = dict(zip(map(str, gl.MODEL_SEEDS), g_wins, strict=True))
            other = {}
            for b in branches:
                if b in (gl.SQUEEZE, gl.S_STAR, gl.CONTROL):
                    continue
                db = gl.discordant(lift[gl.SQUEEZE], lift[b])
                other[b] = {
                    "n_d": int(db.sum()),
                    "ceiling_wins": int(gl.wins(cost[gl.SQUEEZE], cost[b])[db].sum()),
                    "g_wins": [
                        int(gl.wins(g_costs[s_][gl.SQUEEZE], g_costs[s_][b])[db].sum())
                        for s_ in gl.MODEL_SEEDS
                    ],  # fmt: skip
                }
            entry["other_levels"] = other
            entry["terms_discordant_median"] = _terms(rank, t, seeds, disc, branches)
            readings["ceiling_wins"] = entry["ceiling_wins"]
            readings["g_wins"] = g_wins
            if search is not None:
                misses, s_rep = [], {}
                for seed in gl.MODEL_SEEDS:
                    sres = search["results"][f"G-v1-{seed}"]["roots"]
                    meas = [
                        gl.search_measures(sres[f"{s}/{t}"]["level"], sres[f"{s}/{t}"]["cost"],
                                           g_costs[seed][gl.SQUEEZE][i])
                        for i, s in enumerate(seeds)
                    ]  # fmt: skip
                    m_d = [m for m, x in zip(meas, disc, strict=True) if x]
                    misses.append(int(sum(m["miss"] for m in m_d)))
                    levels = np.array([m["level"] for m in meas])
                    s_rep[str(seed)] = {
                        "misses_discordant": misses[-1],
                        "squeezes_all": int(sum(m["squeezes"] for m in meas)),
                        "squeezes_discordant": int(sum(m["squeezes"] for m in m_d)),
                        "partial_preferred_all": int(sum(m["partial_preferred"] for m in meas)),
                        "partial_preferred_discordant": int(
                            sum(m["partial_preferred"] for m in m_d)
                        ),
                        "misses_all": int(sum(m["miss"] for m in meas)),
                        "level_quantiles": np.percentile(levels, [10, 50, 90]).round(3).tolist(),
                        "first_requested_median": float(np.median(
                            [sres[f"{s}/{t}"]["first_requested_grasp"] for s in seeds])),
                        "first_executed_median": float(np.median(
                            [sres[f"{s}/{t}"]["first_executed_grasp"] for s in seeds])),
                    }  # fmt: skip
                entry["search"] = s_rep
                readings["s_misses"] = misses
        per_type[t] = entry
        readings_in[t] = readings
        details[t] = gl.type_reading(readings)
    if search is not None:
        det = {
            k: v for seed in gl.MODEL_SEEDS
            for k, v in search["results"][f"G-v1-{seed}"]["determinism"].items()
        }  # fmt: skip
    else:
        det = {}
    return {
        "cohort": cohort,
        "n": n,
        "branches": branches,
        "per_type": per_type,
        "readings": readings_in,
        "type_rows": details,
        "s_determinism_ok": bool(det) and all(det.values()),
        "rejected_draws": roots.get("rejections"),
        "targets": _count([r["target"] for r in roots["roots"]]),
        "s_star_median": float(np.median([r["s_star"] for r in roots["roots"]])),
        "ceiling_identical_across_seeds": rank.get("ceiling_identical_across_seeds")
        if rank
        else None,
        "problems": problems,
        "revision": roots.get("revision"),
    }


def _terms(rank, t, seeds, disc, branches):
    """Reported (§5.3): the median, over the discordant roots, of each term of the squeeze and
    s* branches, on the ceiling and per G seed."""
    out = {}
    idx = [s for s, x in zip(seeds, disc, strict=True) if x]
    if not idx:
        return out
    sources = {"ceiling": rank["ceiling"]} | {
        str(s): rank["results"][f"G-v1-{s}"]["roots"] for s in gl.MODEL_SEEDS
    }
    for name, src in sources.items():
        out[name] = {}
        for b in (gl.SQUEEZE, gl.S_STAR):
            out[name][b] = {
                term: float(np.median([src[f"{s}/{t}"][b][term] for s in idx]))
                for term in ("total", "latent", "pose", "arm", "hand", "total_at_h8")
            }
            out[name][b]["hand_joints"] = (
                np.median([src[f"{s}/{t}"][b]["hand_joints"] for s in idx], 0).round(5).tolist()
            )
    return out


# ----- GPU time (§10.2) ------------------------------------------------------------------------
START = re.compile(r"^(\S+) start: (\S+) pid (\d+) cwd \S+ rev \S+: (.*); gpu ")
END = re.compile(r"^(\S+) end: (\S+) pid (\d+) status (\S+) after (\d+) s")


def gpu_jobs(path, prefix):
    """``gpu_run``'s log: every job whose holder name starts with ``prefix``, with its command
    and duration."""
    starts, jobs = {}, []
    for line in Path(path).read_text().splitlines():
        m = START.match(line)
        if m and m.group(2).startswith(prefix):
            starts[m.group(3)] = {"utc": m.group(1), "who": m.group(2), "command": m.group(4)}
            continue
        m = END.match(line)
        if m and m.group(2).startswith(prefix):
            job = starts.pop(m.group(3), {"who": m.group(2), "command": "?"})
            jobs.append(job | {"status": m.group(4), "seconds": int(m.group(5))})
    return jobs


def _job_kind(command):
    m = re.search(r"run_task094\.py (\S+)(.*)", command)
    if not m:
        return None, None, None
    cohort = re.search(r"--cohort (\S+)", m.group(2))
    arm = re.search(r"--arm (\S+)", m.group(2))
    return m.group(1), cohort.group(1) if cohort else None, arm.group(1) if arm else None


def projection(jobs) -> dict:
    """§10.2: the gated stage projected from Stage 0's development jobs (their whole durations,
    start-up included): K's goals and every arm x 4 (32 -> 128 resets), each determinism re-run as
    one development arm job; M's roots and ranking x 4 (32 -> 64 roots, 3 -> 6 branches); S x 2
    (32 -> 64 roots)."""
    dev = {}
    for j in jobs:
        kind, cohort, arm = _job_kind(j["command"])
        # a job that exited VOID (3) ran to completion; only its checks differed
        if cohort in ("dev-k", "dev-roots") and str(j["status"]) in ("0", "3"):
            dev[(kind, arm)] = j["seconds"]
    k_arms = [dev.get(("k-run", gl.arm_slug(a)), dev.get(("k-run", a))) for a in gl.K_ARMS]
    if None in k_arms or ("goals", None) not in dev:
        return {"complete": False, "dev_jobs": {f"{k}:{a}": v for (k, a), v in dev.items()}}
    parts = {
        "goals": 4 * dev[("goals", None)],
        "k_arms": 4 * sum(k_arms),
        "k_determinism": 2 * dev.get(("k-run", "K-1.0"), max(k_arms)),
        "roots": 4 * dev[("roots", None)],
        "rank": 4 * dev[("rank", None)],
        "search": 2 * dev[("search", None)],
    }
    total = float(sum(parts.values()))
    return {"complete": True, "seconds": parts, "total_seconds": total, "gpu_hours": total / 3600,
            "dev_jobs": {f"{k}:{a}": v for (k, a), v in dev.items()}}  # fmt: skip


# ----- Stage 0 (§14) ---------------------------------------------------------------------------
def stage_0(args):
    d = Path(args.runs)
    verify = load(d / "verify-task094.json")
    seeds = load(d / "seeds.json")
    power = load(d / "power.json")
    k = k_summary(d, "dev-k", expect_n=None)
    m = ms_summary(d, "dev-roots", expect_n=len(gl.COHORTS["dev-roots"]))
    band = gl.void_band(k["n"])
    a = k["arms"]
    jobs = gpu_jobs(args.gpu_log, args.who_prefix)
    s0_seconds = float(sum(j["seconds"] for j in jobs))
    proj = projection(jobs)
    checks = {
        "verify": bool(verify.get("all_pass")),
        "seeds_no_hits_elsewhere": not seeds.get("hits_elsewhere")
        and not seeds.get("evidence_hits_in_seed_fields"),
        "k_all_arms_ran": not k["missing_arms"],
        "k_plumbing": all(bool(v) for v in k["plumbing"].values()),
        "k_displacement": all(v is True for v in k["displacement"].values()),
        "k1_equals_frozen_ik": bool(k.get("k1_equals_frozen_ik")),
        "k1_in_band": band["low"] <= a[gl.REFERENCE_ARM]["successes"] <= band["high"],
        "k_open_degeneracy": a[gl.OPEN_ARM]["successes"] <= band["open_max"],
        "k_reports": not k["problems"],
        "m_roots_and_squeeze_bitwise": not [p for p in m["problems"]],
        "m_executed_difference": all(
            v["excluded_below_7"] <= len(gl.COHORTS["dev-roots"]) * gl.EXEC_DIFF_VOID
            // gl.GATED_ROOTS_N for v in m["per_type"].values()
        ),
        "m_positive_control": all(
            v.get("positive_control", -1) >= v.get("positive_control_need", 10**9)
            for v in m["per_type"].values()
        ),
        "m_ceiling_identical_across_seeds": bool(m["ceiling_identical_across_seeds"]),
        "s_determinism": bool(m["s_determinism_ok"]),
        "projection_complete": bool(proj["complete"]),
    }  # fmt: skip
    debug = {}
    if args.debug_runs:
        dd = Path(args.debug_runs)
        if (dd / "goals-debug-k.json").exists():
            dk = k_summary(dd, "debug-k")
            debug["k"] = {kk: dk[kk] for kk in ("arms", "plumbing", "displacement", "determinism",
                                                 "problems", "missing_arms")}  # fmt: skip
            debug["k"]["k1_equals_frozen_ik"] = dk.get("k1_equals_frozen_ik")
            debug["k"]["k_row_smoke"] = dk.get("k_row", {}).get("row")
        if (dd / "roots-debug-roots.json").exists():
            dm = ms_summary(dd, "debug-roots")
            debug["m"] = {kk: dm[kk] for kk in ("per_type", "type_rows", "s_determinism_ok",
                                                 "problems")}  # fmt: skip
    rule = power["rule"]
    power_ok = (not rule["escalate"]) and power.get("sims") == gl.POWER_SIMS
    nd = {t: v["n_d"] for t, v in m["per_type"].items()}
    row = gl.s0_row(
        checks,
        n_d=nd,
        power_ok=power_ok,
        stage0_gpu_hours=s0_seconds / 3600,
        projected_gpu_hours=proj.get("gpu_hours", float("inf")),
    )
    return {
        "stage": "0",
        "row": row["row"],
        "s0_row": row,
        "checks": checks,
        "k_dev": k,
        "ms_dev": m,
        "power": {"rule": rule, "rows": power["rows"], "sims": power.get("sims")},
        "seeds": {
            "branches": seeds.get("branches"),
            "local_branches": seeds.get("local_branches"),
            "hits_elsewhere": seeds.get("hits_elsewhere"),
            "evidence_hits": [(h["file"], h["token"]) for h in seeds.get("evidence_hits", [])],
            "evidence_hits_in_seed_fields": seeds.get("evidence_hits_in_seed_fields"),
        },
        "verify": verify.get("checks"),
        "last_two_triggered": {
            k_: v["last_two_triggered"] for k_, v in verify.get("task087_checkpoints", {}).items()
        },
        "debug": debug,
        "gpu": {
            "jobs": jobs,
            "stage0_seconds": s0_seconds,
            "stage0_gpu_hours": s0_seconds / 3600,
            "projection": proj,
        },  # fmt: skip
        "revision": verify.get("revision"),
    }


# ----- the gated stage (§4.3, §7) --------------------------------------------------------------
def stage_gated(args):
    d = Path(args.runs)
    s0 = load(args.s0_summary)
    problems = []
    if s0.get("row") != "S0-PASS":
        problems.append("Stage 0 did not pass")
    budget_stop = (d / "BUDGET-STOP").exists()
    k = k_summary(d, "gated-k", expect_n=gl.GATED_K_N)
    if k.get("k_row") is None:
        k["k_row"] = {"row": "K-VOID", "void_reasons": ["not every K arm ran"]}
    if budget_stop:
        k["k_row"] = dict(k["k_row"], row="K-VOID")
        k["k_row"].setdefault("void_reasons", []).append("the 1.0 GPU h ceiling (§10.2)")
    if problems:
        k["k_row"] = dict(k["k_row"], row="K-VOID")
        k["k_row"].setdefault("void_reasons", []).extend(problems)
    want_det = set(gl.DETERMINISM_K[0])
    if set(k["determinism"]) != want_det and k["k_row"]["row"] != "K-VOID":
        k["k_row"] = dict(k["k_row"], row="K-VOID")
        k["k_row"].setdefault("void_reasons", []).append("a K determinism re-run is missing")
    m = None
    lrow = None
    if (d / "roots-gated-roots.json").exists():
        m = ms_summary(d, "gated-roots", expect_n=gl.GATED_ROOTS_N)
        void = list(m["problems"])
        for t, v in m["per_type"].items():
            if v["excluded_below_7"] > gl.EXEC_DIFF_VOID:
                void.append(f"{t}: {v['excluded_below_7']} roots differ at fewer than 7 steps")
            if v.get("positive_control", -1) < v.get("positive_control_need", 10**9):
                void.append(f"{t}: the positive control failed")
        if not m["s_determinism_ok"]:
            void.append("S's determinism re-run differs or is missing")
        if not m["ceiling_identical_across_seeds"]:
            void.append("the ceiling differs between seeds")
        if budget_stop:
            void.append("the 1.0 GPU h ceiling (§10.2)")
        lrow = gl.l_row(m["readings"], void="; ".join(void))
    task = gl.task_row(k["k_row"], lrow)
    return {
        "stage": "gated",
        "row": task,
        "k_row": k["k_row"],
        "l_row": lrow,
        "l_row_read": k["k_row"]["row"] == "K-CAUSAL",
        "k": k,
        "ms": m,
        "budget_stop": budget_stop,
        "s0_revision": s0.get("revision"),
        "revision": k.get("revision"),
        "clause_fires": False,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=("0", "gated"))
    p.add_argument("--runs", required=True)
    p.add_argument("--debug-runs")
    p.add_argument("--gpu-log", default=str(Path.home() / ".local/state/gpu/oej-gpu_run.log"))
    p.add_argument("--who-prefix", default="oej:task094-")
    p.add_argument("--s0-summary")
    args = p.parse_args(argv)
    if args.stage == "gated" and not args.s0_summary:
        p.error("--stage gated needs --s0-summary")
    out = Path(args.runs) / f"summary-{args.stage}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    summary = stage_0(args) if args.stage == "0" else stage_gated(args)
    gl.write_json(out, summary)
    print(json.dumps({k: summary[k] for k in ("stage", "row")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
