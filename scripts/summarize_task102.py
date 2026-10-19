"""TASK-102's Stage 0 summariser (protocol docs/experiments/so101_hybrid_lift_v1.md §5.2, §7, §10,
§11, §13). It reads job reports and per-episode logs only and applies the declared rules:

* ``tau --p <P-episodes> --output tau.json``: τ_env and τ_max from TASK-101 A2's recorded
  envelope ("pre" column), k(r) on P with exact intervals and achieved hand-over RMS, the G-zero
  guard, the ordering reading, and τ_h by §5.2 step 2 (S0-GLOC-WEAK if k(0) < 16).
* ``config --a <A-episodes> --tau tau.json --output config.json``: a_A per configuration, the
  commands used, the closest and hand-over arm RMS, and §7.5's rule.
* ``stage0 --s0 <stage dir> --output summary.json``: everything the Stage 0 record states:
  windows and training, the check ledger and ordering, P, A, D (every arm with its interval,
  a_H, W+F split by hand-over type and RMS, the per-phase decomposition), the bar, the power
  tables, timing, GPU hours and the projection, and the Stage 0 row.
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
from embodied_jepa import so101_hybrid as hy

ENVELOPE = Path.home() / "develop/emai/evidence/task101-envelope/envelope.json"
ENVELOPE_SHA256 = None  # read and reported (the record's SHA256SUMS lists it)
GPU_LOG = Path.home() / ".local/state/gpu/oej-gpu_run.log"
WHO = "oej:task102-s0"


def load_report(folder) -> dict:
    return json.loads((Path(folder) / "report.json").read_text())


def interval(k, n):
    lo, hi = hy.exact_interval(int(k), int(n))
    return [round(lo, 4), round(hi, 4)]


def count(k, n) -> dict:
    return {"k": int(k), "n": int(n), "rate": k / n if n else None, "ci95": interval(k, n)}


def ledger_ordering(stage: Path, report: dict) -> dict:
    """§7.3 / TASK-101 §18.5 (3): the cohort's episodes started after its latest required check."""
    entries = hy.ledger_read(stage / hy.LEDGER_FILE)
    verdict = hy.ledger_verdict(
        entries,
        revision=report["revision"],
        goals_sha256=report["goals_sha256"],
        block_on_failures=False,
    )
    started = report.get("episodes_started_utc") or report["started_utc"]
    return {
        "verdict_ok": verdict["ok"],
        "reasons": verdict["reasons"],
        "latest_required_utc": verdict["latest_required_utc"],
        "episodes_started_utc": started,
        "ordered": hy.ordered_after_checks(started, verdict),
    }


def envelope_tau() -> dict:
    env = json.loads(ENVELOPE.read_text())
    pre = {
        row["level_rad"]: row["successes"]
        for row in env["table"]
        if row["variant"] == "pre" and row["of"] == 32
    }
    return hy.tau_env_rule(pre) | {
        "pre_counts": {str(k): v for k, v in sorted(pre.items())},
        "envelope_sha256": cl.sha256_file(ENVELOPE),
    }


# ----- tau (§5.2 step 2, §7.4) --
def cmd_tau(args) -> int:
    folder = Path(args.p)
    rep = load_report(folder)
    if rep["cohort"] not in ("P", "Pp"):
        raise hy.GuardError("tau reads P's (or P′'s) episodes")
    eps = rep["episodes"]
    order = ledger_ordering(folder.parent, rep)
    void = []
    if not order["ordered"]:
        void.append("P's episodes started before a required check")
    if not all(e["guards"].get("g_zero_approach", True) for e in eps if e["level"] == 0.0):
        void.append("G-zero failed on P's r = 0 approach")
    if len(eps) != 4 * hy.P_RESETS:
        void.append(f"P ran {len(eps)} episodes, not 128")
    k, levels = {}, {}
    for r in hy.P_LEVELS_RAD:
        rows = [e for e in eps if e["level"] == r]
        k[r] = sum(e["success"] for e in rows)
        rms = [e["handover"]["arm_rms_rad"] for e in rows if e.get("handover")]
        levels[str(r)] = count(k[r], len(rows)) | {
            "handover_rms_median_rad": float(np.median(rms)) if rms else None,
            "handover_rms_range_rad": [float(min(rms)), float(max(rms))] if rms else None,
            "terminations": _tally(e["termination"] for e in rows),
            "stopped": sum(e["stopped"] for e in rows),
        }
    env = envelope_tau()
    rule = hy.tau_h_rule(k, tau_max=env["tau_max"])
    out = {
        "task": hy.TASK,
        "cohort": rep["cohort"],
        "p_report_sha256": cl.sha256_file(folder / "report.json"),
        "revision": rep["revision"],
        "envelope": env,
        "levels": levels,
        "rule": rule,
        "tau_h": rule["tau_h"],
        "gloc_weak": rule["gloc_weak"],
        "ordering": order,
        "void_reasons": void,
    }
    if void:
        out["tau_h"] = None
    hy.write_stage_parameter(args.output, out)
    print(json.dumps(out, indent=1, default=cl._json_default))
    return 0


def _tally(values) -> dict:
    out: dict = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


# ----- config (§7.5) --
def cmd_config(args) -> int:
    folder = Path(args.a)
    rep = load_report(folder)
    if rep["cohort"] not in ("A", "Ap"):
        raise hy.GuardError("config reads A's (or A′'s) episodes")
    tau = hy.read_stage_parameter(args.tau)
    if rep["stage_parameters"]["tau_h"] != tau["tau_h"]:
        raise hy.GuardError("A ran with a τ_h other than P's")
    order = ledger_ordering(folder.parent, rep)
    eps = rep["episodes"]
    void = [] if order["ordered"] else ["A's episodes started before a required check"]
    if len(eps) != 4 * hy.A_RESETS:
        void.append(f"A ran {len(eps)} episodes, not 128")
    per, a_counts, closest = {}, {}, {}
    for c in hy.CONFIGS:
        rows = [e for e in eps if e["config"] == c]
        a_counts[c] = hy.handovers_by_tolerance(rows)
        cl_rms = [e["closest_rms_to_subgoal1"] for e in rows]
        ho = [e["handover"]["arm_rms_rad"] for e in rows if e.get("handover")]
        closest[c] = float(np.median(cl_rms))
        per[c] = count(a_counts[c], len(rows)) | {
            "commands_median": float(np.median([e["commands"] for e in rows])),
            "closest_rms_median_rad": closest[c],
            "closest_rms_min_rad": float(min(cl_rms)),
            "handover_rms_median_rad": float(np.median(ho)) if ho else None,
            "handover_rms_min_rad": float(min(ho)) if ho else None,
            "timeouts": sum((e.get("handover") or {}).get("how") == "timeout" for e in rows),
            "stopped": sum(e["stopped"] for e in rows),
        }
    rule = hy.choose_approach_config(a_counts, closest)
    out = {
        "task": hy.TASK,
        "cohort": rep["cohort"],
        "a_report_sha256": cl.sha256_file(folder / "report.json"),
        "tau_h": tau["tau_h"],
        "configs": per,
        "rule": rule,
        "chosen": None if void else rule["chosen"],
        "approach_fails": rule["approach_fails"],
        "ordering": order,
        "void_reasons": void,
    }
    hy.write_stage_parameter(args.output, out)
    print(json.dumps(out, indent=1, default=cl._json_default))
    return 0


# ----- stage0 (§7.7) --
def gpu_hours(log=GPU_LOG, who=WHO) -> dict:
    jobs = []
    if log.exists():
        for line in log.read_text().splitlines():
            m = re.match(rf"(\S+) end: {re.escape(who)} pid (\d+) status (\d+) after (\d+) s", line)
            if m:
                jobs.append(
                    {"end_utc": m[1], "pid": int(m[2]), "status": int(m[3]), "s": int(m[4])}
                )
    return {"jobs": len(jobs), "gpu_h": sum(j["s"] for j in jobs) / 3600.0, "list": jobs}


def episodes_of(folders) -> list[dict]:
    out = []
    for f in folders:
        rep = load_report(f)
        for e in rep["episodes"]:
            out.append(e | {"_folder": str(f)})
    return out


def grasp_frame_offsets(log_path) -> dict:
    """§10's per-phase decomposition from one episode log: the jaw site's offset from the cube
    centre in the cube's grasp frame (closing, along, up) at the first phase-2 command with the
    jaw command below 0 and at the first both-jaw contact; the arm RMS motion between them; the
    maximum rise."""
    from embodied_jepa.so101_followers import _grasp_frame
    from embodied_jepa.so101_tasks import REST_Z

    with np.load(log_path) as x:
        phase, req, q = x["phase"], x["requested"], x["q"]
        cube, quat, site, contacts = x["cube"], x["quat"], x["site"], x["contacts"]

    def offset(i):
        closing, along = _grasp_frame(cube[i], quat[i])
        d = site[i] - cube[i]
        return [float(d @ closing), float(d @ along), float(d[2])]

    p2 = np.flatnonzero(phase == 2)
    closing_idx = [i for i in p2 if req[i, 5] < 0]
    both = np.flatnonzero(contacts[:, 0] & contacts[:, 1])
    first_close = int(closing_idx[0]) if closing_idx else None
    first_both = int(both[0]) if len(both) else None
    motion = None
    if first_close is not None and first_both is not None and first_both >= first_close:
        motion = hy.arm_rms(q[first_both], q[first_close])
    return {
        "first_close_step": first_close,
        "offset_at_close_m": offset(first_close) if first_close is not None else None,
        "first_both_contact_step": first_both,
        "offset_at_contact_m": offset(first_both) if first_both is not None else None,
        "arm_motion_close_to_contact_rad": motion,
        "max_rise_m": float(cube[:, 2].max() - REST_Z),
    }


def arm_summary(rows) -> dict:
    k = sum(e["success"] for e in rows)
    out = count(k, len(rows)) | {
        "terminations": _tally(e["termination"] for e in rows),
        "stopped": sum(bool(e.get("stopped")) for e in rows),
        "clipped_commands": int(sum(e.get("clipped_commands", 0) for e in rows)),
    }
    ho = [e.get("handover") for e in rows if e.get("handover")]
    if ho:
        tol = [h for h in ho if h["how"] == "tolerance"]
        out["handover"] = {
            "by_tolerance": len(tol),
            "by_timeout": sum(h["how"] == "timeout" for h in ho),
            "fixed": sum(h["how"] == "fixed" for h in ho),
            "phase1_commands_median": float(np.median([h["phase1_commands"] for h in ho])),
            "arm_rms_median_rad": float(np.median([h["arm_rms_rad"] for h in ho])),
            "jaw_error_median_rad": float(np.median([abs(h["jaw_error_rad"]) for h in ho])),
        }
    if any(e.get("closest_rms_to_subgoal1") is not None for e in rows):
        c = [e["closest_rms_to_subgoal1"] for e in rows if e.get("closest_rms_to_subgoal1")]
        out["closest_rms_median_rad"] = float(np.median(c))
    return out


def wf_split(rows) -> dict:
    """W+F split by hand-over type and by hand-over RMS (§6, §7.5)."""
    out = {}
    for how in ("tolerance", "timeout"):
        sub = [e for e in rows if (e.get("handover") or {}).get("how") == how]
        out[how] = count(sum(e["success"] for e in sub), len(sub)) if sub else {"n": 0}
    bins = [(0.0, 0.04), (0.04, 0.08), (0.08, 0.12), (0.12, 0.2), (0.2, 0.4), (0.4, math.inf)]
    out["by_handover_rms"] = []
    for lo, hi in bins:
        sub = [e for e in rows if e.get("handover") and lo <= e["handover"]["arm_rms_rad"] < hi]
        out["by_handover_rms"].append(
            {"rms_rad": [lo, hi], "n": len(sub), "k": sum(e["success"] for e in sub)}
        )
    return out


def power_tables(rates: dict, bar: int | None) -> dict:
    out = {"bar_power_at_hd": None, "mcnemar": {}}
    p_h = rates["H"]
    if bar is not None:
        out["bar_power_at_hd"] = hy.bar_power(p_h, bar)
        out["bar_power_at_hd_minus_0.05"] = hy.bar_power(max(p_h - 0.05, 0), bar)
    table = cl.mcnemar_pass_table(hy.S_RESETS)
    for i, arm in enumerate(hy.CONTROLS):
        out["mcnemar"][arm] = {
            c: hy.mcnemar_power(p_h, rates[arm], c, index=10 * i + j, table=table)
            for j, c in enumerate(cl.COUPLINGS)
        }
    return out


def cmd_stage0(args) -> int:
    s0 = Path(args.s0)
    out: dict = {"task": hy.TASK, "stage": str(s0)}
    void = []
    # windows and training (§7.2)
    win = json.loads((s0 / "gloc-windows/windows_report.json").read_text())
    out["windows"] = {
        k: {x: v[x] for x in v if x != "files_sha256"} for k, v in win["splits"].items()
    } | {"revision": win["revision"]}
    sel = json.loads((s0 / "gloc.json").read_text())
    train = {}
    for name, entry in sel["checkpoints"].items():
        rep = json.loads((Path(entry["path"]).parent / "report.json").read_text())
        train[name] = {
            "selection": rep["selection"],
            "frames": rep["frames"],
            "checkpoint_sha256": entry["sha256"],
            "seconds": rep["seconds"],
            "revision": rep["revision"],
            "final_val_mse": rep["curve"][-1]["val_mse"],
        }
    out["gloc"] = {"primary": sel["primary"], "replication": sel["replication"], "seeds": train}
    eligible = not sel["weak"]
    # the ledger
    out["ledger"] = hy.ledger_read(s0 / hy.LEDGER_FILE)
    failing = [e for e in out["ledger"] if e.get("passes") is False]
    if failing:
        void.append(f"failing checks: {[e['check'] for e in failing]}")
    # P
    tau = hy.read_stage_parameter(s0 / "tau.json")
    out["P"] = tau
    void += tau["void_reasons"]
    k0 = tau["rule"]["k"]["0.0"]
    row_args = {"gloc_eligible": eligible, "k0_p": k0}
    # A
    if (s0 / "config.json").exists():
        cfg = hy.read_stage_parameter(s0 / "config.json")
        out["A"] = cfg
        void += cfg["void_reasons"]
        row_args["a_a"] = cfg["rule"]["a_A"]
    # D
    d_folders = sorted(p for p in s0.glob("D-*") if (p / "report.json").exists())
    if d_folders:
        eps = episodes_of(d_folders)
        reps = {f.name: load_report(f) for f in d_folders}
        for name, rep in reps.items():
            o = ledger_ordering(s0, rep)
            if not o["ordered"]:
                void.append(f"{name} started before a required check")
            sp = rep["stage_parameters"]
            if sp["tau_h"] != tau["tau_h"] or sp["config"] != out["A"]["chosen"]:
                void.append(f"{name} ran with stage parameters other than P's and A's")
        if any(not all(e.get("guards", {}).values()) for e in eps):
            void.append("a G-zero guard failed on D")
        by_arm = {
            a: sorted((e for e in eps if e["arm"] == a), key=lambda e: e["slot"]) for a in hy.D_ARMS
        }
        for a, rows in by_arm.items():
            if len(rows) != hy.D_RESETS or [e["slot"] for e in rows] != list(range(hy.D_RESETS)):
                void.append(f"D's {a} has {len(rows)} episodes, not one per slot")
        if any(e["arm"] == "F" and not e.get("equals_demonstration") for e in eps):
            void.append("F diverged from the goal demonstration")
        out["D"] = {a: arm_summary(rows) for a, rows in by_arm.items() if rows}
        k = {a: out["D"][a]["k"] for a in out["D"]}
        a_h = hy.handovers_by_tolerance(by_arm["H"])
        out["D_a_H"] = count(a_h, hy.D_RESETS)
        out["D_WF_split"] = wf_split(by_arm["W+F"])
        out["D_H_split"] = wf_split(by_arm["H"])
        outcomes = {a: np.array([e["success"] for e in rows], bool) for a, rows in by_arm.items()}
        out["D_paired_against_H"] = {
            a: hy.paired(outcomes["H"], outcomes[a]) for a in by_arm if a != "H"
        }
        out["D_reported"] = {
            "H_vs_H-pose+G_bootstrap": hy.bootstrap_difference(outcomes["H"], outcomes["H-pose+G"]),
            "H_vs_F": {
                "difference": int(outcomes["H"].sum() - outcomes["F"].sum()),
                "ci95": hy.bootstrap_difference(outcomes["H"], outcomes["F"]),
            },
        }
        # decomposition (§10) for every G_loc arm
        decomposition = {}
        for a in ("H", "S+G", "N_J+G", "G_loc-only", "H-pose+G", "W+F"):
            rows = []
            for e in by_arm[a]:
                path = Path(e["_folder"]) / (f"{a}@{e['config']}" if e["config"] else a) / e["file"]
                rows.append(grasp_frame_offsets(path) | {"success": e["success"]})
            decomposition[a] = _decomposition_summary(rows)
        out["D_decomposition"] = decomposition
        row_args |= {
            "sg_d": k["S+G"],
            "a_h": a_h,
            "wf_d": k["W+F"],
            "h_d": k["H"],
        }
        rates = {a: k[a] / hy.D_RESETS for a in k}
        bar = hy.bar_rule(k["H"], rates["S+G"], rates["W+F"])
        out["bar"] = bar
        out["bar_table_with_cap"] = hy.bar_table(r_sg=rates["S+G"], r_wf=rates["W+F"])
        out["power"] = power_tables(rates, bar["bar"] if bar["admissible"] else None)
        # the projection (§13): D's measured lock seconds per episode
        gh = gpu_hours()
        out["gpu"] = {k_: v for k_, v in gh.items() if k_ != "list"} | {"jobs_list": gh["list"]}
        lock = {}
        for rep in reps.values():
            arms = rep["arms"].split(",")
            n = len(rep["episodes"])
            if rep["device"] == "cuda" and n:
                weight = {
                    a: sum(e["episode_seconds"] for e in rep["episodes"] if e["arm"] == a)
                    for a in arms
                }
                total = sum(weight.values())
                for a in arms:
                    eps_a = sum(e["arm"] == a for e in rep["episodes"])
                    lock[a] = rep["wall_seconds"] * weight[a] / total / max(eps_a, 1)
        lock["H-G1"] = lock["H-G2"] = lock["H"]
        goals_h = args.goals_gpu_h
        cpu_ev = reps.get("D-cpu-ev", {}).get("wall_seconds", 0.0) / 3600.0
        out["projection"] = hy.project_stage_s(
            lock_seconds_per_episode=lock,
            goals_gpu_h=goals_h,
            cpu_evidence_gpu_h=cpu_ev,
            used_gpu_h=gh["gpu_h"],
        ) | {"lock_seconds_per_episode": lock}
        row_args["s_fits_without_replication"] = out["projection"]["fits_without_replication"]
    if "gpu" not in out:
        gh = gpu_hours()
        out["gpu"] = {k_: v for k_, v in gh.items() if k_ != "list"} | {"jobs_list": gh["list"]}
    out["row"] = hy.stage0_row(void_reasons=void, **row_args)
    out["void_reasons"] = void
    Path(args.output).write_text(json.dumps(out, indent=1, default=cl._json_default))
    print(json.dumps({k_: out[k_] for k_ in ("row",)}, indent=1, default=cl._json_default))
    return 0


def _decomposition_summary(rows) -> dict:
    def med(key, i=None, sel=None):
        vals = [
            (r[key] if i is None else r[key][i])
            for r in rows
            if r[key] is not None and (sel is None or sel(r))
        ]
        return float(np.median(vals)) if vals else None

    return {
        "closed": sum(r["first_close_step"] is not None for r in rows),
        "both_jaw_contact": sum(r["first_both_contact_step"] is not None for r in rows),
        "offset_at_close_median_m": [med("offset_at_close_m", i) for i in range(3)],
        "offset_at_close_median_m_successes": [
            med("offset_at_close_m", i, lambda r: r["success"]) for i in range(3)
        ],
        "offset_at_close_median_m_failures": [
            med("offset_at_close_m", i, lambda r: not r["success"]) for i in range(3)
        ],
        "arm_motion_close_to_contact_median_rad": med("arm_motion_close_to_contact_rad"),
        "max_rise_median_m": med("max_rise_m"),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tau")
    t.add_argument("--p", required=True)
    t.add_argument("--output", required=True)
    c = sub.add_parser("config")
    c.add_argument("--a", required=True)
    c.add_argument("--tau", required=True)
    c.add_argument("--output", required=True)
    s = sub.add_parser("stage0")
    s.add_argument("--s0", required=True)
    s.add_argument("--goals-gpu-h", type=float, default=0.1)
    s.add_argument("--output", required=True)
    args = p.parse_args(argv)
    return {"tau": cmd_tau, "config": cmd_config, "stage0": cmd_stage0}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
