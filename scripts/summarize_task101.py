"""TASK-101's Stage 0 summary (protocol docs/experiments/so101_closed_loop_lift_v1.md §7, §10,
§11, §13): K's counts and the configuration rule, D's counts with exact intervals, the paired
tests, the per-phase decomposition, the bar, the recomputed power tables, the timing, the GPU
hours and the Stage S projection, and the Stage 0 row. NumPy only; reads the stage outputs and
the GPU-lock log; writes ``stage0_summary.json`` and ``stage0_summary.md`` (refuses to
overwrite).

Usage: summarize_task101.py --stage0 <outputs/task101-stage0> --output <dir>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from embodied_jepa import so101_closed_loop as cl

GPU_LOG = Path.home() / ".local/state/gpu/oej-gpu_run.log"
CHECKS = ("latent", "adapter", "ties", "determinism", "render", "renc", "projector")


def load(path) -> dict:
    return json.loads(Path(path).read_text())


def gpu_hours(log=GPU_LOG, prefix="oej:task101") -> dict:
    """Lock time of every GPU job whose holder name starts with ``prefix`` (§13: GPU hours are
    lock time)."""
    jobs = []
    pattern = re.compile(r"^(\S+) end: (\S+) pid \d+ status (-?\d+) after (\d+) s")
    for line in Path(log).read_text().splitlines():
        m = pattern.match(line)
        if m and m.group(2).startswith(prefix):
            jobs.append(
                {
                    "utc": m.group(1),
                    "who": m.group(2),
                    "status": int(m.group(3)),
                    "seconds": int(m.group(4)),
                }
            )
    return {"jobs": jobs, "gpu_h": sum(j["seconds"] for j in jobs) / 3600.0}


def episodes(report: dict) -> list[dict]:
    return report["episodes"]


def by_arm(eps: list[dict]) -> dict:
    out: dict = {}
    for e in eps:
        key = e["arm"] if e.get("config") is None else f"{e['arm']}@{e['config']}"
        out.setdefault(key, []).append(e)
    for v in out.values():
        v.sort(key=lambda e: e["slot"])
    return out


def count_row(eps: list[dict]) -> dict:
    k, n = sum(e["success"] for e in eps), len(eps)
    return {"k": int(k), "n": n, "ci95": cl.exact_interval(k, n) if n else None}


def outcome_vector(eps: list[dict], n: int) -> np.ndarray:
    v = np.zeros(n, bool)
    for e in eps:
        v[e["slot"]] = e["success"]
    return v


def plan_seconds(folder: Path) -> np.ndarray:
    vals = []
    for f in sorted(folder.glob("*.npz")):
        with np.load(f) as d:
            vals.append(d["plan_seconds"])
    return np.concatenate(vals) if vals else np.zeros(0)


def step_seconds_max(folder: Path) -> float:
    m = 0.0
    for f in sorted(folder.glob("*.npz")):
        with np.load(f) as d:
            m = max(m, float(d["step_seconds"].max()))
    return m


# ----- per-phase decomposition (§10, reported only) ------------------------------------------
ENVELOPE_MM = {"closing": (-20.0, 5.0), "along": (-10.0, 10.0), "up": (None, 10.0)}


def grasp_offset(site, cube, demo_site, demo_cube, demo_xmat) -> np.ndarray:
    """(closing, along, up) in mm of the fingertip-to-cube offset minus the demonstration's,
    in the demonstration's grasp frame (closing = its site +z, along = its site +y, up = +z)."""
    r = np.asarray(demo_xmat, float).reshape(3, 3)
    d = (np.asarray(site) - np.asarray(cube)) - (np.asarray(demo_site) - np.asarray(demo_cube))
    return 1000.0 * np.array([d @ r[:, 2], d @ r[:, 1], d[2]])


def inside_envelope(off) -> bool:
    c, a, u = off
    return bool(-20.0 <= c <= 5.0 and abs(a) <= 10.0 and u <= 10.0)


def decompose(paths, eps: list[dict], goals: dict) -> dict:
    """``paths``: the episode files, in the order of ``eps``."""
    a = goals["arrays"]
    reached = {k: {"reached": 0, "timeout": 0, "never": 0} for k in (1, 2, 3)}
    offsets_start, offsets_end, inside_start, inside_end = [], [], 0, 0
    contact_any = both = lost = stopped = clipped = 0
    rises = []
    for path, e in zip(paths, eps, strict=True):
        hows = {ev["from"]: ev["how"] for ev in e.get("switch_events") or []}
        for k in (1, 2, 3):
            reached[k][hows.get(k, "never")] += 1
        lost += e["termination"] == "cube_lost"
        stopped += bool(e.get("stopped"))
        clipped += int(e.get("clipped_commands") or 0)
        rises.append(e.get("max_lift_m") or 0.0)
        with np.load(path) as d:
            contacts = d["contacts"]
            contact_any += bool(contacts[:, 2].any())
            both += bool((contacts[:, 0] & contacts[:, 1]).any())
            sub = d["subgoal"]
            on3 = np.flatnonzero(sub == 2)
            if len(on3):
                slot = e["slot"]
                cube_cm = a["cube_cm"][slot] / 100.0
                t0, t1 = on3[0], on3[-1]
                before = max(t0 - 1, 0)
                o0 = grasp_offset(
                    d["site"][before],
                    d["cube"][before],
                    a["site_m"][slot, 1],
                    cube_cm[1],
                    a["site_xmat"][slot, 1],
                )
                o1 = grasp_offset(
                    d["site"][t1],
                    d["cube"][t1],
                    a["site_m"][slot, 2],
                    cube_cm[2],
                    a["site_xmat"][slot, 2],
                )
                offsets_start.append(o0)
                offsets_end.append(o1)
                inside_start += inside_envelope(o0)
                inside_end += inside_envelope(o1)

    def med(x):
        return np.median(np.abs(np.asarray(x)), 0).round(1).tolist() if x else None

    return {
        "episodes": len(eps),
        "subgoals": {str(k): v for k, v in reached.items()},
        "reached_subgoal_3": len(offsets_start),
        "grasp_offset_start_median_abs_mm": med(offsets_start),
        "grasp_offset_end_median_abs_mm": med(offsets_end),
        "inside_envelope_at_start": inside_start,
        "inside_envelope_at_end": inside_end,
        "any_gripper_contact": contact_any,
        "both_jaw_contact": both,
        "max_rise_median_m": float(np.median(rises)) if rises else None,
        "max_rise_ge_5cm": int(sum(r >= 0.05 for r in rises)),
        "lost": lost,
        "stopped": stopped,
        "clipped_commands": clipped,
    }


# ----- the summary ---------------------------------------------------------------------------
def summarize(s0: Path) -> dict:
    out: dict = {"task": cl.TASK, "protocol": cl.PROTOCOL}
    out["checks"] = {}
    for c in CHECKS:
        path = s0 / f"check-{c}" / "report.json"
        if path.exists():
            r = load(path)
            out["checks"][c] = {
                "passes": r["result"].get("passes"),
                "revision": r["revision"],
                "result": r["result"],
            }
    # K (§7.3)
    k_goals = cl.load_goal_store(s0 / "K-goals")
    k_rep = load(s0 / "K-episodes" / "report.json")
    k_arms = by_arm(episodes(k_rep))
    k_counts = {c: count_row(k_arms[f"W@{c}"])["k"] for c in cl.CONFIGS}
    eps_sec = {
        c: float(np.mean([e["episode_seconds"] for e in k_arms[f"W@{c}"]])) for c in cl.CONFIGS
    }
    k_total_ep_sec = sum(e["episode_seconds"] for e in episodes(k_rep))
    k_factor = k_rep["wall_seconds"] / k_total_ep_sec  # wall per episode-second in the pool
    k_timing = {}
    for c in cl.CONFIGS:
        ps = plan_seconds(s0 / "K-episodes" / f"W@{c}")
        k_timing[c] = {
            "plan_seconds_median": float(np.median(ps)),
            "plan_seconds_p95": float(np.percentile(ps, 95)),
            "step_seconds_max": step_seconds_max(s0 / "K-episodes" / f"W@{c}"),
            "episode_seconds_mean": eps_sec[c],
            "projected_S_cem_gpu_h": k_factor * 6 * cl.S_RESETS * eps_sec[c] / 3600.0,
        }
    eligible = {
        c: k_timing[c]["projected_S_cem_gpu_h"] <= cl.HARD_CAP_GPU_H - cl.STAGE0_CAP_GPU_H
        for c in cl.CONFIGS
    }
    choice = cl.choose_config(k_counts, eligible)
    out["K"] = {
        "goals_sha256": k_goals["sha256"],
        "kept": len(k_goals["arrays"]["seeds"]),
        "rejected": k_goals["report"]["rejected"],
        "counts": {c: count_row(k_arms[f"W@{c}"]) for c in cl.CONFIGS},
        "timing": k_timing,
        "workers": k_rep["workers"],
        "wall_seconds": k_rep["wall_seconds"],
        "eligible": eligible,
        "choice": choice,
        "revision": k_rep["revision"],
    }
    # D (§7.4)
    d_path = s0 / "D-goals"
    gpu = gpu_hours()
    out["gpu"] = {"stage0_gpu_h": gpu["gpu_h"], "jobs": len(gpu["jobs"])}
    if not (s0 / "D-cem" / "report.json").exists():
        # Stage 0 stopped before D's closed loop (a VOID raised before W's D counts were read)
        out["D"] = None
        void = [c for c, v in out["checks"].items() if v["passes"] is False]
        if (d_path / "goals_report.json").exists():
            d_goals = cl.load_goal_store(d_path)
            ev = load(d_path / "cpu-evidence" / "report.json")["episodes"]
            out["D_goals"] = {
                "goals_sha256": d_goals["sha256"],
                "kept": len(d_goals["arrays"]["seeds"]),
                "drawn": d_goals["report"]["drawn"],
                "rejected": d_goals["report"]["rejected"],
                "cpu_evidence_counts": {
                    a: count_row([e for e in ev if e["arm"] == a]) for a in cl.CPU_ARMS
                },
                "F_equals_demonstration": all(
                    e.get("equals_demonstration") for e in ev if e["arm"] == "F"
                ),
            }
        out["row"] = {
            "row": "S0-VOID" if void else None,
            "void_reasons": void,
            "stopped_before": "D's closed loop",
        }
        return out
    d_goals = cl.load_goal_store(d_path)
    n = len(d_goals["arrays"]["seeds"])
    cem = load(s0 / "D-cem" / "report.json")
    cpu = load(s0 / "D-cpu" / "report.json")
    ev = load(d_path / "cpu-evidence" / "report.json")
    single = load(s0 / "D-single" / "report.json")
    cfg = cem["configs"]
    arms = by_arm(episodes(cem) + episodes(cpu) + ev["episodes"] + episodes(single))
    name = {a: (f"{a}@{cfg}" if a in cl.CEM_ARMS else a) for a in cl.D_ARMS}
    vec = {a: outcome_vector(arms[name[a]], n) for a in cl.D_ARMS}
    for a in cl.D_ARMS:
        if len(arms[name[a]]) != n:
            raise SystemExit(f"D: {a} has {len(arms[name[a]])} episodes, not {n}")
    counts = {a: count_row(arms[name[a]]) for a in cl.D_ARMS}
    w_d, r_f = counts["W"]["k"], counts["F"]["k"] / n
    bar = cl.bar_rule(w_d, n, r_f)
    pairs = {a: cl.paired(vec["W"], vec[a]) for a in cl.D_ARMS if a != "W"}
    pairs_vs_rand = {a: cl.paired(vec[a], vec["rand"]) for a in ("N", "W-shuf", "W-mean")}
    folders = {a: (s0 / "D-cem" / name[a]) if a in cl.CEM_ARMS else None for a in cl.D_ARMS}

    def where(arm, e):
        if arm in cl.CEM_ARMS:
            return s0 / "D-cem" / name[arm] / e["file"]
        first = d_path / "cpu-evidence" / arm / e["file"]
        return first if e["slot"] < cl.EVIDENCE_FRAMES_EPISODES else s0 / "D-cpu" / arm / e["file"]

    decomposition = {
        a: decompose([where(a, e) for e in arms[name[a]]], arms[name[a]], d_goals)
        for a in ("W", "N", "W-shuf", "W-mean", "H-pose")
    }
    single_eps = arms[f"{cl.DIAGNOSTIC}@{cfg}"]
    single_w = [e for e in arms[name["W"]] if e["slot"] < cl.DIAGNOSTIC_RESETS]
    # power (§11): the bar by exact sums, McNemar by simulation from D's rates
    rate_w = w_d / n
    table = cl.mcnemar_pass_table(cl.S_RESETS)
    power = {
        "bar_at_wd": cl.bar_power(rate_w, bar["bar"]),
        "bar_at_wd_minus_0.05": cl.bar_power(max(rate_w - 0.05, 0.0), bar["bar"]),
        "bar_table": cl.bar_table(),
        "mcnemar": {},
    }
    for i, a in enumerate(cl.CONTROLS):
        p_c = counts[a]["k"] / n
        power["mcnemar"][a] = {
            "p_w": rate_w,
            "p_control": p_c,
            **{
                cp: cl.mcnemar_power(rate_w, p_c, cp, index=10 * i + j, table=table)
                for j, cp in enumerate(cl.COUPLINGS)
            },
        }
    # timing and GPU (§7.2.5, §13)
    d_timing = {}
    for a in ("W", "N", "W-shuf", "W-mean"):
        ps = plan_seconds(folders[a])
        d_timing[a] = {
            "plan_seconds_median": float(np.median(ps)),
            "plan_seconds_p95": float(np.percentile(ps, 95)),
            "step_seconds_max": step_seconds_max(folders[a]),
            "episode_seconds_mean": float(np.mean([e["episode_seconds"] for e in arms[name[a]]])),
        }
    goals_lock = [j for j in gpu["jobs"] if j["who"] == "oej:task101"]  # informative only
    d_goals_seconds = load(d_path / "job.json")["seconds_total"]
    cem_ep_sec = {a: d_timing[a]["episode_seconds_mean"] for a in ("W", "N", "W-shuf", "W-mean")}
    cem_ep_sec |= {"W-100100": cem_ep_sec["W"], "W-100102": cem_ep_sec["W"]}
    d_cem_episode_seconds = sum(e["episode_seconds"] for e in episodes(cem))
    speedup = d_cem_episode_seconds / cem["wall_seconds"]
    projection = cl.project_stage_s(
        seconds_per_episode=cem_ep_sec,
        workers_speedup=speedup,
        goals_gpu_h=2.0 * d_goals_seconds / 3600.0,
        used_gpu_h=gpu["gpu_h"],
    )
    void = [c for c, v in out["checks"].items() if v["passes"] is False]
    row = cl.stage0_row(
        void_reasons=void,
        d_rejections=len(d_goals["report"]["rejected"]),
        k_counts=k_counts,
        w_d=w_d,
        n_d=n,
        s_fits_without_replication=projection["fits_without_replication"],
    )
    out["D"] = {
        "goals_sha256": d_goals["sha256"],
        "kept": n,
        "rejected": d_goals["report"]["rejected"],
        "drawn": d_goals["report"]["drawn"],
        "config": cfg,
        "counts": counts,
        "pairs_w_minus": pairs,
        "pairs_vs_rand": pairs_vs_rand,
        "non_inferiority_vs_H_pose": cl.non_inferiority(vec["W"], vec["H-pose"], n),
        "decomposition": decomposition,
        "single_goal_diagnostic": {
            "W-single": count_row(single_eps),
            "W_same_resets": count_row(single_w),
        },
        "timing": d_timing,
        "workers": cem["workers"],
        "wall_seconds": cem["wall_seconds"],
        "revision": cem["revision"],
        "F_equals_demonstration": all(e.get("equals_demonstration") for e in arms["F"]),
    }
    out["bar"] = bar
    out["power"] = power
    out["gpu"] = {
        "stage0_gpu_h": gpu["gpu_h"],
        "jobs": len(gpu["jobs"]),
        "goals_jobs": len(goals_lock),
    }
    out["projection"] = projection | {
        "pool_speedup": speedup,
        "per_arm_episode_seconds": cem_ep_sec,
    }
    out["row"] = row
    return out


def markdown(s: dict) -> str:
    lines = [f"# {cl.TASK} Stage 0 summary", ""]
    lines += ["## Checks", "", "| check | passes |", "|---|---|"]
    lines += [f"| {c} | {v['passes']} |" for c, v in s["checks"].items()]
    k = s["K"]
    lines += [
        "",
        "## K (calibration, 32 resets)",
        "",
        "| configuration | W successes | exact 95 % | plan s median | p95 | projected S GPU h |",
        "|---|---|---|---|---|---|",
    ]
    for c in cl.CONFIGS:
        r, t = k["counts"][c], k["timing"][c]
        lines.append(
            f"| {c} | {r['k']}/{r['n']} | {r['ci95'][0]:.3f}–{r['ci95'][1]:.3f} | "
            f"{t['plan_seconds_median']:.3f} | {t['plan_seconds_p95']:.3f} | "
            f"{t['projected_S_cem_gpu_h']:.2f} |"
        )
    lines += ["", f"Chosen: **{k['choice']['chosen']}** (null: {k['choice']['null']})"]
    if s.get("D") is None:
        lines += [
            "",
            f"Row: **{s['row']['row']}** ({', '.join(s['row']['void_reasons'])}); "
            f"stopped before D's closed loop; Stage 0 GPU h {s['gpu']['stage0_gpu_h']:.2f}",
        ]
    d = s.get("D")
    if d:
        lines += [
            "",
            f"## D (development, {d['kept']} resets, {d['config']})",
            "",
            "| arm | successes | exact 95 % | W − arm | W only / arm only | McNemar p |",
            "|---|---|---|---|---|---|",
        ]
        for a, r in d["counts"].items():
            p = d["pairs_w_minus"].get(a)
            extra = (
                (
                    f"{p['difference']:+d} | {p['only_first']}/{p['only_second']} | "
                    f"{p['p_one_sided']:.2g}"
                )
                if p
                else " | | "
            )
            lines.append(
                f"| {a} | {r['k']}/{r['n']} | {r['ci95'][0]:.3f}–{r['ci95'][1]:.3f} | {extra} |"
            )
        b = s["bar"]
        lines += [
            "",
            f"Bar: b = {b['b']:.4f} (LCB {b['lcb']:.4f}), **{b['bar']}/128**; power at "
            f"W(D) {s['power']['bar_at_wd']:.3f}",
            "",
            f"Row: **{s['row']['row']}**; Stage 0 GPU h {s['gpu']['stage0_gpu_h']:.2f}; "
            f"projected S {s['projection']['with_replication_gpu_h']:.2f} GPU h "
            f"(without replication "
            f"{s['projection']['without_replication_gpu_h']:.2f})",
        ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage0", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    out = Path(args.output)
    for name in ("stage0_summary.json", "stage0_summary.md"):
        if (out / name).exists():
            raise FileExistsError(f"refusing to overwrite {out / name}")
    out.mkdir(parents=True, exist_ok=True)
    s = summarize(Path(args.stage0))
    (out / "stage0_summary.json").write_text(json.dumps(s, indent=1, default=cl._json_default))
    (out / "stage0_summary.md").write_text(markdown(s))
    print(markdown(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
