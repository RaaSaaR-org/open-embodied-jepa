"""The C1 feasibility record's runner (development only, before any protocol; no world model).

Design note ``docs/experiments/apple_lewm_next_v2_design.md`` §4.1 (C1-F1 to C1-F6); record
``docs/experiments/apple_lewm_next_v2_c1_feasibility.md``; constants and rules
``src/embodied_jepa/lewm_next_c1.py``; workers ``src/embodied_jepa/lewm_next_c1_runtime.py`` (on
TASK-076's worker, hook and arms, unchanged). Guards from ``embodied_jepa.run_tools`` and TASK-076's
harness module; this runner loads no other script.

One invocation runs every stage in the declared order and writes ``<output>/report.json`` (it
refuses to overwrite an existing output):

1. P-3's post-look estimates by G-repro (TASK-072 run-1's readout, refitted exactly).
2. **ceiling**: H-final(commit) on cohort F (57000-57031): C1-F1, C1-F2's ceiling, C1-F6.
3. **reach**: e9's committed aim at a in {-0.5, -0.4, -0.3, -0.2}, b in {-3, 0, +3} cm on F:
   C1-F2's a_lo.
4. **proxies**: H-now, N-proxy, shuf-proxy, mean-proxy (and the two reported-only
   offset-consistent variants) on F: C1-F3.
5. **corpus**: 256 roots (57100-57355), e9's aim committed uniformly over the box, the onboard
   frames at 405 and r and the plate-hidden render at r.
6. **offline** (CPU, no GPU): frozen DINOv2 features; cross-fitted R-plate-pool at r and on the
   plate-hidden renders at r (C1-F5); R-plate at 405; the closed-loop smoke R-plate; H-sysid.
7. **comparators**: H-rule, H-sysid (on the reading) and their true-plate versions, and
   H-now-reaim (reported only), on F: C1-F4.

``--debug`` simulates debug seeds 57900-57999 only, at small sizes; nothing in it is read.
"""

from __future__ import annotations

import os

os.environ.update(
    {
        "MKL_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": "6",
        "MKL_NUM_THREADS": "6",
        "OPENBLAS_NUM_THREADS": "16",
    }
)

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import devices  # noqa: E402
from embodied_jepa import first_policy_v2_linux as fpl  # noqa: E402
from embodied_jepa import lewm_next_c1 as c1  # noqa: E402
from embodied_jepa import lewm_planner_v2 as lp  # noqa: E402
from embodied_jepa import plate_twin_v2 as pt  # noqa: E402
from embodied_jepa import plate_twin_v2_harness as hz  # noqa: E402
from embodied_jepa import run_guards as rg  # noqa: E402
from embodied_jepa import run_tools as rt  # noqa: E402

TASK076_MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-plate-twin-v2.json"
fpl.configure_headless()
GIB = 2**30
log = hz.log
GLOBAL_CAP_SECONDS = 4 * 3600.0
MAP_CAP_SECONDS = 3600.0
DEBUG = {
    "F": (57900, 57905),  # 6 resets
    "corpus": (57910, 57939),  # 30 roots
    "reach_min_fraction": c1.REACH_MIN / c1.F_RESETS,
    "fallback_a_lo": -0.2,  # debug only, if the rule finds none on 6 resets
    "fallback_r": 485,  # debug only
}


class Pool(rg.BoundedPool):
    """``run_guards.BoundedPool`` on the C1 worker; a dead worker or a cap is a GuardError; every
    map runs TASK-071's G-look check (as TASK-076's harness pool)."""

    def __init__(self, workers: int, config: dict):
        from embodied_jepa import lewm_next_c1_runtime as crt

        super().__init__(workers, crt.run_task, initializer=crt.worker_init, initargs=(config,))
        self.look_reference = None

    def map(self, tasks: list[dict], cap: float, what: str) -> list[dict]:
        try:
            out = super().map(tasks, cap, what)
        except (rg.WorkerDied, rg.MapCapExceeded) as error:
            raise lp.GuardError(str(error)) from error
        self.look_reference = hz.check_look_states(out, self.look_reference)
        return out


# ----- preflight ----------------------------------------------------------------------------------
def preflight(report: dict, args) -> dict:
    import mujoco
    import torch

    from embodied_jepa import pretrained_encoder as pe

    rt.assert_local_import(ROOT, report)
    manifest = json.loads(TASK076_MANIFEST.read_text())
    report["task076_pins_at_preflight"] = hz.check_pins(manifest["hashes"])  # TASK-076's code
    dirty = hz.tracked_tree_dirty()
    report["revision"], report["tracked_tree_dirty"] = hz.revision(), bool(dirty)
    if not args.debug:
        hz.check_clean(dirty)
    report["platform"] = fpl.check_platform()
    report["thread_env"] = {k: os.environ.get(k) for k in pt.THREAD_ENV}
    if report["thread_env"] != pt.THREAD_ENV:
        raise lp.GuardError(f"G-threads: {report['thread_env']} is not {pt.THREAD_ENV}")
    report["load_average_at_start"] = list(os.getloadavg())
    available = hz.mem_available_bytes()
    report["mem_available_at_start_gib"] = available / GIB
    if available < (pt.MEMORY["ceiling_gib"] + pt.MEMORY["headroom_gib"]) * GIB:
        raise lp.GuardError(f"G-memory: MemAvailable {available / GIB:.1f} GiB is too low")
    torch.set_num_threads(6)
    devices.configure_determinism("cuda", strict=True)  # no CUDA context is created
    if mujoco.__version__ != manifest["mujoco_version"]:
        raise lp.GuardError(f"G-hash: MuJoCo {mujoco.__version__} is not the pinned version")
    report["environment"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
    }
    digests = {
        "pretrained": pe.weights_digest(pe.load_pretrained()),
        "floor": pe.weights_digest(pe.random_init()),
    }
    if digests != manifest["encoder_digests"]:
        raise lp.GuardError("G-weights: an encoder digest differs from TASK-076's pin")
    report["encoder_digests"] = digests
    c1.check_seed_ranges()
    found = hz.check_evidence(Path(args.evidence))
    report["evidence"] = {"root": str(args.evidence), "sha256": found["sha256"]}
    report["_run1"] = found["run1"]
    return manifest


# ----- helpers ------------------------------------------------------------------------------------
def seeds_for(role: str, args) -> tuple[int, ...]:
    if args.debug:
        low, high = DEBUG[role]
        return c1.check_seeds(role, tuple(range(low, high + 1)), debug=True)
    return c1.check_seeds(role, c1.seeds_of(role))


def attempt_tasks(arm: str, seeds, resets, est, per_seed=None, **extra) -> list[dict]:
    tasks = []
    for s in seeds:
        task = {
            "kind": "attempt",
            "arm": arm,
            "seed": s,
            "reset": resets[s],
            "estimates": est[s]["estimates"],
            "expected_frame_sha256": est[s]["frame_sha256"],
            "expected_state_sha256": est[s]["state_sha256"],
            "wall_seconds": pt.CAPS_SECONDS["per_attempt"],
        }
        tasks.append(task | (per_seed(s) if per_seed else {}) | extra)
    return tasks


def run_arm(pool, tasks, what: str) -> list[dict]:
    records = pool.map(tasks, MAP_CAP_SECONDS, what)
    arm = tasks[0]["arm"]
    if arm not in c1.PRIVILEGED_ARMS:
        for r in records:
            if r.get("blocked") is None and not r.get("privileged_ok", False):
                raise lp.GuardError(f"G-privileged: {arm} seed {r['seed']} read task truth")
    log(f"{what}: {sum(bool(r['success']) for r in records)}/{len(records)}")
    return records


def successes(records) -> list[bool]:
    return [bool(r["success"]) for r in records]


def lean(records, *, keep_path: bool = False) -> list[dict]:
    """Per-attempt facts for the report: commands hashed, frames and (unless kept) paths dropped."""
    out = []
    for item in hz.strip(records):
        item = dict(item)
        item.pop("captured", None)
        if not keep_path and "kpred" in item:
            item["kpred"] = {k: v for k, v in item["kpred"].items() if k != "plate_path"}
        out.append(item)
    return out


def arm_summary(records) -> dict:
    misses = [
        r["commit"]["landing_miss_cm"]
        for r in records
        if r.get("commit") and r["commit"].get("landing_miss_cm") is not None
    ]
    a_values = [r["commit"]["a_true"] for r in records if r.get("commit")]
    return {
        "count": sum(successes(records)),
        "n": len(records),
        "per_reset": successes(records),
        "refused": sum(r["blocked"] is not None for r in records),
        "fallbacks": sum(bool(r.get("commit", {}).get("fallback")) for r in records),
        "landing_miss_cm": {
            "median": float(np.median(misses)) if misses else None,
            "p87_5": float(np.percentile(misses, 87.5)) if misses else None,
            "max": float(np.max(misses)) if misses else None,
        },
        "a_true": {
            "median": float(np.median(a_values)) if a_values else None,
            "min": float(np.min(a_values)) if a_values else None,
            "max": float(np.max(a_values)) if a_values else None,
        },
        "seconds": {
            "median": float(np.median([r["seconds"] for r in records])),
            "max": float(np.max([r["seconds"] for r in records])),
        },
    }


# ----- the stages ---------------------------------------------------------------------------------
def stage_ceiling(report, pool, seeds, resets, est):
    records = run_arm(pool, attempt_tasks("H-final", seeds, resets, est), "ceiling H-final(commit)")
    s1 = c1.RULE["s1"]
    distances, remaining, speeds, m2 = [], [], [], []
    for r in records:
        k = r["kpred"]
        path = {int(t): np.asarray(v) for t, v in k["plate_path"].items()}
        if r["blocked"] is not None or s1 not in path:
            distances.append(None)
            continue
        distances.append({t: 100.0 * float(np.linalg.norm(v - path[s1])) for t, v in path.items()})
        remaining.append(k["remaining_405_cm"])
    for r in records:
        k = r["kpred"]
        if k.get("palm_speed_405_cm") is not None:
            speeds.append(k["palm_speed_405_cm"])
        if k.get("m2_405_cm") is not None:
            m2.append(k["m2_405_cm"])
    r_step = c1.read_step(distances)
    looks = [d.get("lookahead") for r in records for d in r["decisions"] if d.get("lookahead")]
    summary = arm_summary(records)
    out = {
        "seeds": list(seeds),
        "arm": summary,
        "remaining_405_cm": remaining,
        "palm_speed_405_cm": speeds,
        "m2_405_cm": {
            "median": float(np.median(m2)) if m2 else None,
            "max": float(np.max(m2)) if m2 else None,
        },
        "within_0_1_cm_count_by_step": {
            str(t): sum(1 for d in distances if d is not None and d.get(t, math.inf) <= 0.1)
            for t in range(c1.RULE["s0"], s1 + 1)
        },
        "lookahead": {
            "decisions": len(looks),
            "converged": sum(bool(x["converged"]) for x in looks),
            "iterations_max": max((len(x["iterations"]) for x in looks), default=0),
        },
        "f1": c1.decide_f1(remaining, r_step, speeds),
        "f6": c1.decide_f6([r["seconds"] for r in records]),
        "attempts": lean(records, keep_path=True),
    }
    report["stages"]["ceiling"] = out
    return records, r_step


def stage_reach(report, pool, seeds, resets, est, args):
    complete, per_level = {}, {}
    for a in c1.A_LEVELS:
        ok = np.ones(len(seeds), bool)
        detail = {}
        for b in c1.REACH_B_M:
            tasks = attempt_tasks("reach", seeds, resets, est, a=a, b_m=b)
            records = run_arm(pool, tasks, f"reach a={a} b={b * 100:+.0f} cm")
            done = np.array(
                [
                    r["blocked"] is None
                    and not r.get("commit", {}).get("fallback", True)
                    and r["executed_steps"] is not None
                    and int(r["executed_steps"]) >= c1.TRANSFER_END
                    for r in records
                ]
            )
            ok &= done
            detail[f"{b:+.2f}"] = {
                "complete": int(done.sum()),
                "success": sum(successes(records)),
                "incomplete_seeds": [s for s, d in zip(seeds, done, strict=True) if not d],
                "terminations": sorted({str(r["termination_reason"]) for r in records}),
            }
        complete[a] = int(ok.sum())
        per_level[str(a)] = {"complete_all_three": int(ok.sum()), "by_b": detail}
    if args.debug:
        need = math.ceil(DEBUG["reach_min_fraction"] * len(seeds) - 1e-12)
        a_lo = None
        for level in sorted(c1.A_LEVELS, reverse=True):
            if complete[level] >= need:
                a_lo = level
            else:
                break
    else:
        a_lo = c1.decide_a_lo(complete)
    report["stages"]["reach"] = {"levels": per_level, "a_lo": a_lo}
    return a_lo


def stage_proxies(report, pool, seeds, resets, est, a_lo, ceiling_records):
    n = len(seeds)
    final_target = {r["seed"]: r["commit"]["target"] for r in ceiling_records if r.get("commit")}
    true_plate = {s: resets[s]["plate_xy"] for s in seeds}
    arms = {}
    for arm in c1.PROXY_ARMS:
        per = None
        if arm == "shuf-proxy":

            def per(s):
                return {"foreign_plate": true_plate[seeds[c1.foreign_index(seeds.index(s), n)]]}

        arms[arm] = run_arm(pool, attempt_tasks(arm, seeds, resets, est, per, a_lo=a_lo), arm)
    # reported only (R14.7): the offset-consistent variants, from H-final(commit)'s own aim
    ceiling_palm = {r["seed"]: np.asarray(r["commit"]["h"]) for r in ceiling_records}
    for name, base in (("shuf-proxy-offset", "shuf"), ("mean-proxy-offset", "mean")):

        def per(s, base=base):
            p = np.asarray(true_plate[s])
            other = (
                np.asarray(true_plate[seeds[c1.foreign_index(seeds.index(s), n)]])
                if base == "shuf"
                else np.asarray(c1.PLATE_MEAN)
            )
            g = np.asarray(final_target[s]) + (other - p) / (1.0 - c1.RULE["kappa"])
            a, b = c1.to_box(g, p, ceiling_palm[s])
            a = min(max(a, a_lo), c1.A_HI)
            b = min(max(b, -c1.B_HALF_M), c1.B_HALF_M)
            return {"a": a, "b_m": b}

        tasks = attempt_tasks("reach", seeds, resets, est, per)
        for t in tasks:
            t["variant"] = name
        arms[name] = run_arm(pool, tasks, name)
    ceiling = successes(ceiling_records)
    f3 = c1.decide_f3(ceiling, {a: successes(arms[a]) for a in c1.PROXY_ARMS})
    reported = {
        name: c1.paired_interval(ceiling, successes(arms[name]))
        | {"count": sum(successes(arms[name]))}
        for name in ("shuf-proxy-offset", "mean-proxy-offset")
    }
    report["stages"]["proxies"] = {
        "a_lo": a_lo,
        "a_n": c1.n_proxy_a(a_lo),
        "arms": {a: arm_summary(r) for a, r in arms.items()},
        "f3": f3,
        "offset_variants_reported_only": reported,
        "attempts": {a: lean(r) for a, r in arms.items()},
    }
    return f3


def stage_corpus(report, pool, seeds, resets, est, a_lo, r_step, folder: Path):
    draws = {s: c1.corpus_aim(s, a_lo) for s in seeds}
    tasks = attempt_tasks(
        "collect",
        seeds,
        resets,
        est,
        lambda s: {"a": draws[s][0], "b_m": draws[s][1]},
        capture=[c1.COMMIT_STEP, int(r_step)],
    )
    records = run_arm(pool, tasks, "corpus")
    rows = []
    for r in records:
        cap = r["captured"]
        if r.get("commit") is None or str(c1.COMMIT_STEP) not in cap or str(r_step) not in cap:
            continue
        rows.append(r)
    s1 = c1.RULE["s1"]
    arrays = {
        "seeds": np.asarray([r["seed"] for r in rows], np.int64),
        "a": np.asarray([draws[r["seed"]][0] for r in rows]),
        "b_m": np.asarray([draws[r["seed"]][1] for r in rows]),
        "p405": np.asarray([r["commit"]["p"] for r in rows]),
        "h405": np.asarray([r["commit"]["h"] for r in rows]),
        "target": np.asarray([r["commit"]["target"] for r in rows]),
        "fallback": np.asarray([r["commit"]["fallback"] for r in rows]),
        "plate_r": np.asarray([r["captured"][str(r_step)]["plate"] for r in rows]),
        "plate_s1": np.asarray(
            [r["kpred"]["plate_path"].get(str(s1), [np.nan, np.nan]) for r in rows]
        ),
        "success": np.asarray([bool(r["success"]) for r in rows]),
        "frames405": np.stack([r["captured"][str(c1.COMMIT_STEP)]["visible"] for r in rows]),
        "frames_r": np.stack([r["captured"][str(r_step)]["visible"] for r in rows]),
        "hidden_r": np.stack([r["captured"][str(r_step)]["hidden"] for r in rows]),
        "r": np.asarray(int(r_step)),
    }
    path = folder / "corpus.npz"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    np.savez(path, **arrays)
    report["stages"]["corpus"] = {
        "seeds": [int(seeds[0]), int(seeds[-1])],
        "roots": len(records),
        "rows": len(rows),
        "excluded": sorted(set(seeds) - {r["seed"] for r in rows}),
        "count": sum(successes(records)),
        "fallbacks": int(arrays["fallback"].sum()),
        "path": str(path),
        "sha256": hz.sha256_file(path),
        "attempts": lean(records),
    }
    return arrays


def stage_offline(report, arrays: dict, folder: Path):
    """CPU only: frozen DINOv2 features, the cross-fits of C1-F5, the closed-loop smoke R-plate and
    H-sysid. No GPU and no CUDA context."""
    import time

    from embodied_jepa import obs_ceiling_v2_offline as ooff
    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa.models.latent_critic import RidgeReadout
    from embodied_jepa.plate_twin_v2_offline import FULL_WIDTH, save_readout

    started = time.monotonic()
    encoder = pe.load_pretrained()
    pooled405, full405 = ooff.featurise(
        encoder, arrays["frames405"], device="cpu", columns=slice(0, FULL_WIDTH)
    )
    pooled_r, _ = ooff.featurise(encoder, arrays["frames_r"], device="cpu")
    hidden_r, _ = ooff.featurise(encoder, arrays["hidden_r"], device="cpu")
    feat_seconds = time.monotonic() - started
    n = len(arrays["seeds"])
    roots = arrays["seeds"].astype(str)
    fold = c1.outer_folds(n)
    y405, yr = arrays["p405"], arrays["plate_r"]

    def ridge(x, y, groups):
        return RidgeReadout.fit(
            x,
            y,
            groups,
            lambdas=c1.LAMBDA_GRID_RELATIVE,
            folds=c1.INNER_FOLDS,
            seed=c1.SALTS["inner_folds"],
            dual=True,
        )

    pred = {k: np.full((n, 2), np.nan) for k in ("pool_r", "hidden_r", "full405", "prior_r")}
    sysid_oof = {k: np.full((n, 2), np.nan) for k in ("true", "reading")}
    selections = []
    for k in range(c1.OUTER_FOLDS):
        fit, held = fold != k, fold == k
        pool_readout = ridge(pooled_r[fit], yr[fit], roots[fit])
        pred["pool_r"][held] = pool_readout.predict(pooled_r[held])
        pred["hidden_r"][held] = pool_readout.predict(hidden_r[held])
        full_readout = ridge(full405[fit], y405[fit], roots[fit])
        pred["full405"][held] = full_readout.predict(full405[held])
        pred["prior_r"][held] = yr[fit].mean(axis=0)
        selections.append(
            {"fold": k, "pool_r": pool_readout.selection, "full405": full_readout.selection}
        )
    for k in range(c1.OUTER_FOLDS):  # H-sysid's cross-fitted residuals (reported)
        fit, held = fold != k, fold == k
        coef = c1.sysid_fit(y405[fit], arrays["h405"][fit], arrays["target"][fit], yr[fit])
        sysid_oof["true"][held] = c1.sysid_predict(
            coef, y405[held], arrays["h405"][held], arrays["target"][held]
        )
        coef = c1.sysid_fit(
            pred["full405"][fit], arrays["h405"][fit], arrays["target"][fit], yr[fit]
        )
        sysid_oof["reading"][held] = c1.sysid_predict(
            coef, pred["full405"][held], arrays["h405"][held], arrays["target"][held]
        )
    for name, values in (pred | {f"sysid_{k}": v for k, v in sysid_oof.items()}).items():
        if not np.isfinite(values).all():
            raise lp.GuardError(f"G-finite: {name} has rows without an out-of-fold prediction")
    err = {
        "pool_r": 100.0 * np.linalg.norm(pred["pool_r"] - yr, axis=1),
        "hidden_r": 100.0 * np.linalg.norm(pred["hidden_r"] - yr, axis=1),
        "full405": 100.0 * np.linalg.norm(pred["full405"] - y405, axis=1),
        "prior_r": 100.0 * np.linalg.norm(pred["prior_r"] - yr, axis=1),
        "sysid_true": 100.0 * np.linalg.norm(sysid_oof["true"] - yr, axis=1),
        "sysid_reading": 100.0 * np.linalg.norm(sysid_oof["reading"] - yr, axis=1),
    }
    stats = {k: c1.median_ci(v) | {"p87_5": float(np.percentile(v, 87.5))} for k, v in err.items()}
    f5 = c1.decide_f5(stats["pool_r"], stats["hidden_r"])
    closed = ridge(full405, y405, roots)
    r_plate_path = folder / "r_plate.npz"
    r_plate_sha = save_readout(r_plate_path, closed)
    coef_true = c1.sysid_fit(y405, arrays["h405"], arrays["target"], yr)
    coef_read = c1.sysid_fit(pred["full405"], arrays["h405"], arrays["target"], yr)
    errors_path = folder / "errors.npz"
    np.savez(
        errors_path, roots=arrays["seeds"], fold=fold, **{f"e__{k}": v for k, v in err.items()}
    )
    report["stages"]["offline"] = {
        "device": "cpu",
        "featurisation_seconds": feat_seconds,
        "rows": n,
        "folds": fold.tolist(),
        "selections": selections,
        "statistics": stats,
        "f5": f5,
        "plate_motion_at_r_cm": c1.median_ci(100.0 * np.linalg.norm(yr - y405, axis=1)),
        "r_plate": {
            "path": str(r_plate_path),
            "sha256": r_plate_sha,
            "selection": closed.selection,
        },
        "sysid": {"coef_true": coef_true.tolist(), "coef_reading": coef_read.tolist()},
        "errors": {"path": str(errors_path), "sha256": hz.sha256_file(errors_path)},
        "seconds": time.monotonic() - started,
    }
    return f5, {"r_plate": str(r_plate_path), "r_plate_sha256": r_plate_sha}, coef_true, coef_read


def stage_comparators(report, pool, seeds, resets, est, a_lo, coef_true, coef_read, ceiling):
    arms = {}
    for arm in c1.COMPARATOR_ARMS:
        extra = {"a_lo": a_lo}
        if arm == "H-sysid":
            extra["sysid_coef"] = coef_read.tolist()
        elif arm == "H-sysid-true":
            extra["sysid_coef"] = coef_true.tolist()
        arms[arm] = run_arm(pool, attempt_tasks(arm, seeds, resets, est, **extra), arm)
    readings = [
        {"seed": r["seed"], "reading": r["decisions"][0].get("reading"), "true": r["commit"]["p"]}
        for arm in ("H-rule", "H-sysid")
        for r in arms[arm]
        if r["decisions"] and r.get("commit")
    ]
    reading_err = [
        100.0 * float(np.linalg.norm(np.asarray(x["reading"]) - np.asarray(x["true"])))
        for x in readings
        if x["reading"] is not None
    ]
    counts = {a: sum(successes(r)) for a, r in arms.items()}
    best = max(("H-rule", "H-sysid"), key=lambda a: counts[a])
    report["stages"]["comparators"] = {
        "arms": {a: arm_summary(r) for a, r in arms.items()},
        "paired_vs_ceiling": {
            a: c1.paired_interval(ceiling, successes(r)) for a, r in arms.items()
        },
        "reading_error_405_cm": c1.median_ci(reading_err) if reading_err else None,
        "best_on_reading": {
            "arm": best,
            "count": counts[best],
            "tie": counts["H-rule"] == counts["H-sysid"],
        },
        "f4": "reported, no bar: the better of H-rule and H-sysid on the reading is the protocol's "
        "non-inferiority comparator",
        "attempts": {a: lean(r) for a, r in arms.items()},
    }


# ----- the run ------------------------------------------------------------------------------------
def run(args) -> dict:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    report = {
        "record": c1.RECORD,
        "design_note": c1.DESIGN_NOTE,
        "debug": bool(args.debug),
        "outcome": None,
        "stages": {},
        "argv": sys.argv[1:],
        "paths": {"output": str(output.resolve()), "evidence": str(Path(args.evidence).resolve())},
        "started_utc": hz.utc(),
        "no_world_model": True,
        "gpu": "none: every stage, featurisation included, runs on the CPU",
        "salts": c1.SALTS,
        "seed_ranges": c1.SEED_RANGES if not args.debug else DEBUG,
    }
    clock = hz.Clock(GLOBAL_CAP_SECONDS)
    watch = rt.MemoryWatch(
        pt.MEMORY["ceiling_gib"] * GIB, pt.MEMORY["sample_seconds"], measure=pt.MEMORY["measure"]
    )
    guards = rt.install_guards(watch=watch, log=log)
    report["_watch"] = watch
    pool = None
    try:
        manifest = preflight(report, args)
        f_seeds, k_seeds = seeds_for("F", args), seeds_for("corpus", args)
        resets = {s: c1.reset_of(s) for s in (*f_seeds, *k_seeds)}
        report["resets_digest"] = lp.plan_digest({str(s): v for s, v in resets.items()})
        config = {
            "p3_checkpoint": hz.p3_checkpoint(Path(args.evidence)),
            "torch_threads": pt.WORKER_TORCH_THREADS,
        }
        workers = int(args.workers or pt.SIM_WORKERS)
        report["workers"] = workers
        pool = Pool(workers, config)
        p_readout, encoder = hz.refit_p_readout(report, pool, Path(args.evidence), report["_run1"])
        report["first_render_utc"] = hz.utc()
        est = hz.cohort_estimates(pool, p_readout, encoder, (*f_seeds, *k_seeds), resets, 1800.0)
        report["stages"]["render_disagreements"] = est.pop("_render_disagreements", {})
        del p_readout
        clock.check("ceiling")
        ceiling_records, r_step = stage_ceiling(report, pool, f_seeds, resets, est)
        clock.check("reach")
        a_lo = stage_reach(report, pool, f_seeds, resets, est, args)
        ceiling = successes(ceiling_records)
        f1 = report["stages"]["ceiling"]["f1"]
        f2 = c1.decide_f2(sum(ceiling), report["stages"]["ceiling"]["arm"]["refused"], a_lo)
        report["stages"]["f2"] = f2
        if args.debug:
            a_lo = a_lo if a_lo is not None else DEBUG["fallback_a_lo"]
            r_step = r_step if r_step is not None else DEBUG["fallback_r"]
        f3 = f5 = None
        if a_lo is None or r_step is None:
            report["stopped"] = "a_lo or r is undefined: the stages that need them did not run"
        else:
            clock.check("proxies")
            f3 = stage_proxies(report, pool, f_seeds, resets, est, a_lo, ceiling_records)
            clock.check("corpus")
            arrays = stage_corpus(report, pool, k_seeds, resets, est, a_lo, r_step, output)
            report["pool_close_1"] = pool.close()
            pool = None
            clock.check("offline")
            f5, readout_config, coef_true, coef_read = stage_offline(report, arrays, output)
            del arrays
            clock.check("comparators")
            pool = Pool(workers, config | readout_config)
            stage_comparators(
                report, pool, f_seeds, resets, est, a_lo, coef_true, coef_read, ceiling
            )
        decision = c1.verdict(f1, f2, f3, f5)
        report["checks"] = {
            "C1-F1": f1,
            "C1-F2": f2,
            "C1-F3": None if f3 is None else {"passes": f3["passes"]},
            "C1-F5": None
            if f5 is None
            else {k: f5[k] for k in ("readout_ok", "hidden_ok", "passes")},
            "C1-F6": report["stages"]["ceiling"]["f6"],
        }
        report["verdict"] = decision
        report["pinned_at_end"] = hz.check_pins(manifest["hashes"])
        report["revision_at_end"] = hz.revision()
        if not args.debug and hz.tracked_tree_dirty():
            raise lp.GuardError("G-hash: the tracked tree changed during the run")
        report["outcome"] = "C1-DEBUG" if args.debug else decision["row"]
    except BaseException as error:  # noqa: BLE001 - every failure, signal included, is V
        guards.void(report, error)
    finally:
        guards.finish(report, pool)
        if pool is not None:
            report["pool_close"] = pool.close_record
        report.pop("_watch", None)
        report.pop("_run1", None)
        report["ended_utc"] = hz.utc()
        report["total_seconds"] = clock.elapsed()
        hz.write_report(output / "report.json", report)
        log(f"report written: outcome {report.get('outcome')}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence", required=True, help="TASK-072 run-1's evidence root")
    parser.add_argument("--debug", action="store_true", help="debug seeds only; nothing is read")
    parser.add_argument("--workers", type=int, default=None, help="debug only (1-6)")
    args = parser.parse_args(argv)
    if args.workers is not None and (not args.debug or not 1 <= args.workers <= 6):
        parser.error("--workers is for debug runs only, from 1 to 6")
    report = run(args)
    return 0 if report.get("outcome") not in (None, "V") else 1


if __name__ == "__main__":
    sys.exit(main())
