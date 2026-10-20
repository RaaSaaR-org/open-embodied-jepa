"""TASK-102 Δa's summariser (``docs/experiments/so101_approach_diagnostic_v1.md`` §3.4, §4, §6,
§7): the readings of Λ0-Λ3 from the jobs' outputs, the GPU hours from the lock log, and the row
(``so101_approach_diag.row``, first match). Reported-only readings that no row reads are marked
``reported_only``; those added after the declaration are also marked ``post_hoc``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

from embodied_jepa import so101_approach_diag as ad
from embodied_jepa import so101_closed_loop as cl

GPU_LOG = Path.home() / ".local/state/gpu/oej-gpu_run.log"
WHO = "oej:task102-approach-diag"
L2_DIRS = {"J-ref": "approach-a", "O-img": "approach-a", "J-big16": "approach-b",
           "J-big32": "approach-c", "O-pose": "approach-pose"}  # fmt: skip


def _num(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


def gpu_hours(log=GPU_LOG) -> dict:
    jobs, seconds = [], 0
    for line in Path(log).read_text().splitlines():
        m = re.match(
            rf"(\S+) end: ({re.escape(WHO)}\S*) pid (\d+) status (\d+) after (\d+) s", line
        )
        if m:
            jobs.append({"utc": m[1], "who": m[2], "status": int(m[4]), "seconds": int(m[5])})
            seconds += int(m[5])
    return {"jobs": jobs, "count": len(jobs), "gpu_hours": seconds / 3600.0}


def kept(root: Path) -> dict:
    rep = json.loads((root / "L-goals/goals_report.json").read_text())
    return {"kept": rep["kept"], "rejected": rep["rejected"], "sha256": rep["goals_sha256"]}


# ----- Λ1 / Λ3 --
def _costs(c_true, c_hold):
    """{name: [states]} for every cost of §3.2 at one family's states."""
    out = {}
    for p, part in enumerate(ad.PARTS):
        out[f"true/{part}"] = c_true[:, p]
        for h in ad.HORIZONS:
            for f in ad.FORMS:
                out[f"J-hold H{h}-{f}/{part}"] = ad.form_cost(c_hold[:, :, p], f, h)
    return out


def landscape(root: Path) -> dict:
    folder = root / "landscape"
    files = sorted(folder.glob("*.npz"))
    per_cost_line: dict = {}
    rho_line: dict = {}
    rho_demo: dict = {}
    excluded = 0
    cand_rows = []
    seeds_info = []
    for path in files:
        seed = int(path.stem)
        with np.load(path) as d:
            z = {k: d[k] for k in d.files if "frames" not in k}
        ok = z["line_cube_moved_m"] <= ad.CUBE_MOVED_M
        excluded += int((~ok).sum())
        seeds_info.append(
            {
                "seed": seed,
                "d0": float(z["d0"]),
                "line_d": z["line_d"].tolist(),
                "demo_d": z["demo_d"].tolist(),
                "demo_jaw_err": z["demo_jaw_err"].tolist(),
                "line_jaw_err": z["line_jaw_err"].tolist(),
                "cube_moved_max_m": float(z["line_cube_moved_m"].max()),
            }
        )
        line = _costs(z["line_c_true"], z["line_c_hold"])
        demo = _costs(z["demo_c_true"], z["demo_c_hold"])
        for name, c in line.items():
            per_cost_line.setdefault(name, {})[seed] = {
                float(n): float(v) for n, v, k in zip(z["line_nominal"], c, ok, strict=True)
                if k and n >= 0
            }  # fmt: skip
            rho_line.setdefault(name, []).append(ad.spearman(c[ok], z["line_d"][ok]))
            rho_demo.setdefault(name, []).append(ad.spearman(demo[name], z["demo_d"]))
        for s, nominal in enumerate(z["cand_nominal"]):
            d_end = z["cand_d_end"][s]
            for h in ad.HORIZONS:
                for f in ad.FORMS:
                    for p, part in enumerate(ad.PARTS):
                        t = ad.form_cost(z["cand_true_err"][s, :, :, p], f, h)
                        j = ad.form_cost(z["cand_j_err"][s, :, :, p], f, h)
                        de = d_end[:, h - 1]
                        cand_rows.append(
                            {
                                "seed": seed,
                                "nominal": "d0" if nominal < 0 else float(nominal),
                                "config": f"H{h}-{f}",
                                "part": part,
                                "rho_true": ad.spearman(t, de),
                                "rho_J": ad.spearman(j, de),
                                "rho_JT": ad.spearman(j, t),
                                "regret_true": float(de[np.argmin(t)] - de.min()),
                                "regret_J": float(de[np.argmin(j)] - de.min()),
                                "approach_rank_true": int(np.sum(t < t[0])),
                                "approach_rank_J": int(np.sum(j < j[0])),
                                "d_start": float(
                                    z["line_d"][list(z["line_nominal"]).index(nominal)]
                                ),
                                "d_end_approach": float(de[0]),
                                "d_end_min": float(de.min()),
                            }
                        )
    bands = {}
    for name, per_seed in per_cost_line.items():
        b = ad.band_ordering(per_seed)
        low = ad.band_ordering(per_seed, ad.LOW_BANDS)
        bands[name] = {
            "d_flat": ad.d_flat(b),
            "bands": b,
            "low_bands": low,
            "rho_line_median": ad.median(rho_line[name]),
            "rho_demo_median": ad.median(rho_demo[name]),
        }
    cand = {}
    for cfg in (f"H{h}-{f}" for h in ad.HORIZONS for f in ad.FORMS):
        for part in ad.PARTS:
            rows = [r for r in cand_rows if r["config"] == cfg and r["part"] == part]
            entry = {"by_state": {}}
            for nominal in ad.CANDIDATE_STATES:
                rs = [r for r in rows if r["nominal"] == nominal]
                entry["by_state"][str(nominal)] = {
                    k: ad.median(r[k] for r in rs)
                    for k in (
                        "rho_true",
                        "rho_J",
                        "rho_JT",
                        "regret_true",
                        "regret_J",
                        "approach_rank_true",
                        "approach_rank_J",
                        "d_start",
                        "d_end_approach",
                        "d_end_min",
                    )  # fmt: skip
                } | {"readings": len(rs)}
            far = [r for r in rows if r["nominal"] in ad.FAR_STATES]
            entry["far"] = {
                "readings": len(far),
                "rho_true": ad.median(r["rho_true"] for r in far),
                "rho_J": ad.median(r["rho_J"] for r in far),
                "rho_JT": ad.median(r["rho_JT"] for r in far),
            }
            entry["far"]["delta_rho"] = entry["far"]["rho_true"] - entry["far"]["rho_J"]
            cand[f"{cfg}/{part}"] = entry
    return {
        "seeds": len(files),
        "excluded_states": excluded,
        "seeds_info": seeds_info,
        "costs": bands,
        "candidates": cand,
        "delta_rho": cand["H16-last/full"]["far"]["delta_rho"],
    }


# ----- Λ2 --
def _runs(arr, cond) -> int:
    best = run = 0
    for x in cond:
        run = run + 1 if x else 0
        best = max(best, run)
    return best


def approach(root: Path, goals) -> dict:
    a = goals
    out = {}
    for arm, sub in L2_DIRS.items():
        rep_path = root / sub / "report.json"
        if not rep_path.exists():
            out[arm] = None
            continue
        rep = json.loads(rep_path.read_text())
        res = [r for r in rep["results"] if r["arm"] == arm]
        eps = []
        for r in res:
            with np.load(root / sub / arm / r["file"]) as d:
                q = d["q"]
                c_true = d["c_true"]
            slot = int(r["slot"])
            q1 = a["joints"][slot, 0]
            rms = ad.arm_rms(q, q1)
            jaw = q[:, cl.JAW] - q1[cl.JAW]
            eps.append(
                r
                | {
                    "post_hoc_arm_only_run": _runs(rms, rms <= ad.TAU_H),
                    "post_hoc_jaw_ever_within": bool(np.any(np.abs(jaw) <= ad.JAW_TOL)),
                    "post_hoc_jaw_median_abs": float(np.median(np.abs(jaw))),
                    "post_hoc_c_true_first": _num(float(c_true[0, 0])),
                    "post_hoc_c_true_min": _num(float(np.nanmin(c_true[:, 0])))
                    if np.isfinite(c_true[:, 0]).any()
                    else None,
                }
            )
        n = len(eps)
        h = sum(1 for e in eps if e["by_tolerance"])
        out[arm] = {
            "episodes": n,
            "h": h,
            "exact_95": cl.exact_interval(h, n) if hasattr(cl, "exact_interval") else None,
            "closest_arm_rms_median": ad.median(e["closest_arm_rms"] for e in eps),
            "closest_arm_rms_min": min(e["closest_arm_rms"] for e in eps),
            "closest_arm_rms_max": max(e["closest_arm_rms"] for e in eps),
            "jaw_error_at_closest_median": ad.median(e["jaw_error_at_closest"] for e in eps),
            "commands_median": ad.median(e["commands"] for e in eps),
            "plan_seconds_median": ad.median(e["plan_seconds_median"] for e in eps),
            "oracle_over_velocity": sum(e["oracle_over_velocity"] for e in eps),
            "rollout_checks": sum(e["rollout_checks"] for e in eps),
            "post_hoc_arm_only_handovers": sum(1 for e in eps if e["post_hoc_arm_only_run"] >= 3),
            "post_hoc_jaw_ever_within": sum(1 for e in eps if e["post_hoc_jaw_ever_within"]),
            "post_hoc_jaw_median_abs": ad.median(e["post_hoc_jaw_median_abs"] for e in eps),
            "revision": rep["revision"],
            "tracked_tree_dirty": rep["tracked_tree_dirty"],
            "wall_seconds": rep["wall_seconds"],
            "episodes_detail": eps,
        }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    root = Path(args.root)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    k = kept(root)
    goals = cl.load_goal_store(root / "L-goals", k["sha256"])["arrays"]
    lam1 = landscape(root)
    lam2 = approach(root, goals)
    gpu = gpu_hours()
    complete = all(v is not None and v["episodes"] == ad.L2_SLOTS for v in lam2.values())
    void_reasons = []
    if len(k["kept"]) < ad.MIN_KEPT:
        void_reasons.append("fewer than 12 kept")
    if lam1["seeds"] != len(k["kept"]):
        void_reasons.append("landscape incomplete")
    if not complete:
        void_reasons.append("an approach arm incomplete")
    if gpu["gpu_hours"] > 2.0:
        void_reasons.append("over the 2 GPU h cap")
    for v in lam2.values():
        if v and v["tracked_tree_dirty"]:
            void_reasons.append("a job ran from a dirty tree")
    h = {arm: (v["h"] if v else 0) for arm, v in lam2.items()}
    verdict = ad.row(void=bool(void_reasons), h=h, delta_rho=lam1["delta_rho"])
    summary = {
        "study": ad.STUDY,
        "protocol": ad.PROTOCOL,
        "goals": k,
        "lambda0": json.loads((root / "lambda0.json").read_text())["lambda0"]["configs"]
        if (root / "lambda0.json").exists()
        else None,
        "lambda1": lam1,
        "lambda2": lam2,
        "gpu": gpu,
        "void_reasons": void_reasons,
        "h": h,
        "delta_rho": lam1["delta_rho"],
        "row": verdict,
    }
    for v in (summary["lambda0"] or {}).values():
        v.pop("episodes_detail", None)
    out.write_text(json.dumps(summary, indent=1, default=cl._json_default))
    brief = {"row": verdict, "h": h, "delta_rho": lam1["delta_rho"], "void": void_reasons}
    print(json.dumps(brief | {"gpu": gpu["gpu_hours"]}, default=cl._json_default))
    return 0


if __name__ == "__main__":
    sys.exit(main())
