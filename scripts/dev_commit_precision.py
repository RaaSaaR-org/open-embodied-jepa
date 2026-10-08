"""TASK-081 design note's development feasibility check (not gated; nothing is frozen).

``docs/experiments/apple_lewm_commit_precision_v2_design.md`` §4. On a new development seed
block (70000-70063, disjoint from every TASK-076/077/080 range and from every forbidden range
of TASK-080's frozen block), under TASK-080's condition C1-M with TASK-080's artifacts unchanged
(W-66800 and R-S from Stage R, R-plate, P-3), CPU only:

* W is run in closed loop three times per reset, executing the ``frozen`` aim (TASK-080's
  controller), the ``damped`` aim and the ``affine_local`` aim (``commit_precision_dev``); every
  run logs every variant's aim at 405 from the same roll-outs;
* H-final(commit) (the privileged look-ahead, not learned) gives each reset's reference aim, so
  each variant's aim error is measured against where the plate really lands;
* H-rule (the hand-written rule given the plate law, not learned) is the TASK-080 comparator.

Variants, step size, cap, seeds and arms were fixed in ``commit_precision_dev`` and here before
this script was first run. It reuses TASK-080's runner as a library (its preflight pieces, the
artifact checks by sha256, the cohort estimates); it never runs a TASK-080 stage and simulates no
seed of K, D, S or F. One invocation writes ``<output>/report.json`` and refuses an existing one.
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

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import commit_precision_dev as cp  # noqa: E402
from embodied_jepa import lewm_c1m_v2 as lm  # noqa: E402
from embodied_jepa import lewm_planner_v2 as lp  # noqa: E402
from embodied_jepa import lewm_pr_v2 as pr  # noqa: E402
from embodied_jepa import plate_twin_v2_harness as hz  # noqa: E402
from embodied_jepa import run_guards as rg  # noqa: E402

DEV_SEEDS = (70000, 70063)  # development only (design note §4.1)
DEV_BLOCK = (70000, 71999)  # the design note's proposed TASK-081 block; 70000-70099 is development
RUNS = (("W", "frozen"), ("W", "damped"), ("W", "affine_local"), ("H-final", None),
        ("H-rule", None))  # fmt: skip
MAP_CAP_SECONDS = 3600.0


def _runner():
    spec = importlib.util.spec_from_file_location("run_lewm_pr_v2", ROOT / "scripts" /
                                                  "run_lewm_pr_v2.py")  # fmt: skip
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_dev_seeds(seeds) -> None:
    """Development seeds lie in 70000-70099 and in no forbidden or TASK-080 range."""
    spans = dict(pr.FORBIDDEN_RANGES) | dict(pr.SEED_RANGES) | {"task080_block": pr.SEED_BLOCK}
    for s in seeds:
        if not 70000 <= s <= 70099:
            raise lp.GuardError(f"development seed {s} leaves 70000-70099")
        for name, (low, high) in spans.items():
            if low <= s <= high:
                raise lp.GuardError(f"development seed {s} lies in {name}")


class DevPool(rg.BoundedPool):
    def __init__(self, workers: int, config: dict):
        super().__init__(workers, cp.run_task, initializer=cp.worker_init, initargs=(config,))
        self.look_reference = None

    def map(self, tasks, cap, what):
        try:
            out = super().map(tasks, cap, what)
        except (rg.WorkerDied, rg.MapCapExceeded) as error:
            raise lp.GuardError(str(error)) from error
        self.look_reference = hz.check_look_states(out, self.look_reference)
        return out


# ----- the summary (design note §4.3) ---------------------------------------------------------
def _median_ci(values) -> dict:
    v = np.asarray(values, np.float64)
    rng = np.random.default_rng(np.random.SeedSequence([cp.POWER_SALT, 1, len(v)]))
    boot = np.median(v[rng.integers(0, len(v), (10_000, len(v)))], axis=1)
    return {
        "median": float(np.median(v)),
        "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        "p87_5": float(np.percentile(v, 87.5)),
        "max": float(v.max()),
    }


def analyse(report: dict) -> dict:
    """Every variant's aim error against H-final(commit)'s aim on the same reset, its predicted
    count through TASK-080's pooled tau curve, convergence, and the closed-loop counts."""
    runs = report["runs"]
    counts = dict(pr.K0_PRIME_MEASURED["pooled"]["counts_pooled"])
    ref = {a["seed"]: np.asarray(a["commit"]["target"], np.float64) for a in runs["H-final"]}
    seeds = [a["seed"] for a in runs["W:frozen"]]
    logs = {}
    for name in ("W:frozen", "W:damped", "W:affine_local"):
        for a in runs[name]:
            v = a["decisions"][0]["world_model"]["variants"]
            if name == "W:frozen":
                logs[a["seed"]] = v
            elif json.dumps(v, sort_keys=True) != json.dumps(logs[a["seed"]], sort_keys=True):
                raise lp.GuardError(f"the variants at 405 differ between runs on {a['seed']}")
    out = {"resets": len(seeds), "variants": {}, "closed_loop": {}}
    for v in cp.VARIANTS:
        err = [100.0 * float(np.linalg.norm(np.asarray(logs[s][v]["g"]) - ref[s])) for s in seeds]
        conv = [logs[s][v].get("converged") for s in seeds]
        entry = {
            "aim_error_vs_hfinal_cm": _median_ci(err),
            "predicted_count_of_64": pr.predicted_count(err, counts, pr.POOLED_RESETS),
            "rollouts_median": float(np.median([logs[s][v]["rollouts"] for s in seeds])),
            "errors_cm": err,
        }
        if conv[0] is not None:
            entry["not_converged"] = int(sum(not c for c in conv))
        out["variants"][v] = entry
    capped = [s for s in seeds if not logs[s]["frozen"]["converged"]]
    out["frozen_capped_seeds"] = capped
    for v in cp.VARIANTS:
        e = dict(zip(seeds, out["variants"][v]["errors_cm"], strict=True))
        out["variants"][v]["on_frozen_capped"] = (
            _median_ci([e[s] for s in capped]) if capped else None
        )
        out["variants"][v]["on_frozen_converged"] = _median_ci(
            [e[s] for s in seeds if s not in capped]
        )
    success = {n: [bool(a["success"]) for a in r] for n, r in runs.items()}
    for n, r in runs.items():
        miss = [
            a["commit"]["landing_miss_cm"] for a in r if "landing_miss_cm" in a.get("commit", {})
        ]
        out["closed_loop"][n] = {"count": int(sum(success[n])), "n": len(r),
                                 "landing_miss_cm": _median_ci(miss)}  # fmt: skip
        if n.startswith("W:"):
            out["closed_loop"][n]["minus_h_rule"] = pr.paired_interval(
                success[n], success["H-rule"]
            )
            out["closed_loop"][n]["on_frozen_capped"] = int(
                sum(a["success"] for a in r if a["seed"] in capped)
            )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--analyse", help="summarise an existing report.json and exit")
    ap.add_argument("--output")
    ap.add_argument("--evidence")
    ap.add_argument("--old-features")
    ap.add_argument("--old-fits")
    ap.add_argument("--models", nargs=6)
    ap.add_argument("--stage-r")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--first", type=int, default=DEV_SEEDS[0])
    ap.add_argument("--last", type=int, default=DEV_SEEDS[1])
    args = ap.parse_args(argv)
    if args.analyse:
        summary = analyse(json.loads(Path(args.analyse).read_text()))
        target = Path(args.analyse).with_name("summary.json")
        if target.exists():
            raise SystemExit(f"refusing: {target} exists")
        target.write_text(json.dumps(summary, indent=1))
        print(f"wrote {target}")
        return 0
    for name in ("output", "evidence", "old_features", "old_fits", "models", "stage_r"):
        if getattr(args, name) is None:
            ap.error(f"--{name.replace('_', '-')} is required for a run")
    out_dir = Path(args.output)
    if out_dir.exists():
        raise SystemExit(f"refusing: {out_dir} exists")
    seeds = tuple(range(args.first, args.last + 1))
    check_dev_seeds(seeds)
    rn = _runner()
    started = time.monotonic()
    report = {
        "what": "TASK-081 design note development feasibility (not gated)",
        "revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True
        ).stdout.strip(),  # fmt: skip
        "tracked_tree_dirty": bool(hz.tracked_tree_dirty()),
        "seeds": [seeds[0], seeds[-1]],
        "variants": list(cp.VARIANTS),
        "executed": [list(r) for r in RUNS],
        "damping": cp.DAMPING,
        "stages": {},
        "argv": sys.argv[1:] if argv is None else list(argv),
    }
    manifest = json.loads((ROOT / lm.TASK076_MANIFEST).read_text())
    rn.sim_preflight(report, args, manifest)
    chain = rn.old_chain(args, report, ())
    stage_r = rn.completed(args.stage_r, ("R-PASS",), False)
    saved = stage_r["fields"]["fits"]["seeds"][str(pr.PRIMARY_SEED)]["readouts"]
    folder = Path(args.stage_r).parent / "fits"
    readouts = {k: v | {"path": str(folder / Path(v["path"]).name)} for k, v in saved.items()}
    config = rn.worker_config(chain, readouts=readouts, with_r8=True)
    config = {"torch_threads": pr.WORKER_TORCH_THREADS,
              "p3_checkpoint": hz.p3_checkpoint(Path(args.evidence))} | config  # fmt: skip
    pool = DevPool(int(args.workers), config)
    try:
        p_readout, encoder = hz.refit_p_readout(report, pool, Path(args.evidence), report["_run1"])
        resets = {s: pr.reset_of(s) for s in seeds}
        est = rn.cohort_estimates(pool, p_readout, encoder, seeds, resets, 1800.0)
        report["render_disagreements"] = est.pop("_render_disagreements", {})
        common = {"tau_commit_cm": float(pr.K0_PRIME_MEASURED["tau_commit_cm"]), "a_lo": pr.A_LO}
        runs = {}
        for arm, variant in RUNS:
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
                    "wall_seconds": pr.CAPS_SECONDS["per_attempt"],
                    "move_offset": pr.move_offset(s).tolist(),
                } | common
                if variant is not None:
                    task["variant"] = variant
                tasks.append(task)
            name = arm if variant is None else f"{arm}:{variant}"
            t0 = time.monotonic()
            records = pool.map(tasks, MAP_CAP_SECONDS, name)
            if arm not in pr.PRIVILEGED_ARMS:
                for r in records:
                    if r.get("blocked") is None and (
                        not r.get("privileged_ok", False) or r.get("task_truth_in_controller", 1)
                    ):
                        raise lp.GuardError(f"G-privileged: {name} seed {r['seed']} read truth")
            runs[name] = rn.lean(records)
            print(f"{name}: {sum(bool(r['success']) for r in records)}/{len(records)} "
                  f"({time.monotonic() - t0:.0f} s)", flush=True)  # fmt: skip
        report["runs"] = runs
    finally:
        report["pool_close"] = pool.close()
    report["seconds"] = time.monotonic() - started
    report.pop("_run1", None)
    out_dir.mkdir(parents=True)
    (out_dir / "report.json").write_text(json.dumps(report, default=float))
    print(f"wrote {out_dir / 'report.json'} in {report['seconds']:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
