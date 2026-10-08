"""TASK-082 design note's development feasibility check (not gated; no world model).

``docs/experiments/apple_lewm_unknown_law_v2_design.md`` §4. On a new development block (72000-
72355 of TASK-082's block 72000-74999, R19.24), CPU only, under C1-M's condition with only the
plate law after the commit step changed (``plate_law_dev``), for each law in ``c1m`` (the
reference), ``sat`` and ``play``, on the same 32 check resets (72000-72031):

* **H-final(commit)**, the privileged look-ahead ceiling (bar of interest: >= 30/32);
* **H-now**, the true plate at 405 (privileged; shows that the plate still has to be predicted);
* **H-rule-true**, the hand-written rule handed C1-M's law (kappa = -0.5) and the true plate;
  under ``sat`` and ``play`` that law is wrong, which is the point;
* a **256-root corpus** (72100-72355, uniform box aims, the same roots under every law) and, fitted
  on it with the true plate, **H-sysid-true** (TASK-080's linear form) and **H-sysid-krr-true**
  (an RBF kernel ridge on the same inputs), each run in closed loop on the check resets, plus
  their cross-fitted offline errors on the corpus.

Every arm reads the true plate, so each count is an optimistic bound for an arm that reads the
plate from the image. The laws' parameters, the arms, the seeds and the salts were fixed in
``plate_law_dev`` before any seed of the block was simulated. One invocation writes
``<output>/report.json`` and refuses an existing output.
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
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import first_policy_v2_linux as fpl  # noqa: E402
from embodied_jepa import lewm_next_c1 as c1  # noqa: E402
from embodied_jepa import lewm_planner_v2 as lp  # noqa: E402
from embodied_jepa import plate_law_dev as pl  # noqa: E402
from embodied_jepa import plate_twin_v2 as pt  # noqa: E402
from embodied_jepa import plate_twin_v2_harness as hz  # noqa: E402
from embodied_jepa import run_guards as rg  # noqa: E402

fpl.configure_headless()
MAP_CAP_SECONDS = 3600.0
LAW_ORDER = ("c1m", "sat", "play")
CHECK_ARMS = ("H-final", "H-now", "H-rule-true")
FIT_ARMS = ("H-sysid-true", "H-sysid-krr-true")


class DevPool(rg.BoundedPool):
    def __init__(self, workers: int, config: dict):
        from embodied_jepa import plate_law_dev_runtime as drt

        super().__init__(workers, drt.run_task, initializer=drt.worker_init, initargs=(config,))
        self.look_reference = None

    def map(self, tasks, cap, what):
        try:
            out = super().map(tasks, cap, what)
        except (rg.WorkerDied, rg.MapCapExceeded) as error:
            raise lp.GuardError(str(error)) from error
        self.look_reference = hz.check_look_states(out, self.look_reference)
        return out


def run_arm(pool, arm, law, seeds, resets, est, extra=None, per_seed=None) -> list[dict]:
    from embodied_jepa import plate_law_dev_runtime as drt

    tasks = []
    for s in seeds:
        task = {
            "kind": "attempt",
            "arm": arm,
            "law": law,
            "seed": s,
            "reset": resets[s],
            "estimates": est[s]["estimates"],
            "expected_frame_sha256": est[s]["frame_sha256"],
            "expected_state_sha256": est[s]["state_sha256"],
            "wall_seconds": pt.CAPS_SECONDS["per_attempt"],
            "move_offset": pl.move_offset(s).tolist(),
            "a_lo": pl.A_LO,
        }
        tasks.append(task | (extra or {}) | (per_seed(s) if per_seed else {}))
    t0 = time.monotonic()
    records = pool.map(tasks, MAP_CAP_SECONDS, f"{law}:{arm}")
    if arm not in drt.PRIVILEGED_ARMS:
        for r in records:
            if r.get("blocked") is None and not r.get("privileged_ok", False):
                raise lp.GuardError(f"G-privileged: {arm} seed {r['seed']} read task truth")
    for r in records:
        r.pop("post_look_state", None)
    n = sum(bool(r["success"]) for r in records)
    print(f"{law}:{arm}: {n}/{len(records)} ({time.monotonic() - t0:.0f} s)", flush=True)
    return records


def arm_summary(records, reference=None) -> dict:
    out = {
        "count": int(sum(bool(r["success"]) for r in records)),
        "n": len(records),
        "per_reset": [bool(r["success"]) for r in records],
        "refused": int(sum(r.get("blocked") is not None for r in records)),
        "fallbacks": int(sum(bool(r.get("commit", {}).get("fallback")) for r in records)),
        "seconds_median": float(np.median([r["seconds"] for r in records])),
        "seconds_max": float(np.max([r["seconds"] for r in records])),
    }
    misses = [r["commit"]["landing_miss_cm"] for r in records
              if r.get("commit") and r["commit"].get("landing_miss_cm") is not None]  # fmt: skip
    if misses:
        out["landing_miss_cm"] = pl.median_ci(misses)
    motion = [r["commit"]["plate_motion_cm"] for r in records
              if r.get("commit") and r["commit"].get("plate_motion_cm") is not None]  # fmt: skip
    if motion:
        out["plate_motion_405_to_s1_cm"] = pl.median_ci(motion)
    conv = [r["commit"].get("lookahead_converged") for r in records if r.get("commit")]
    if any(c is not None for c in conv):
        out["lookahead_converged"] = int(sum(bool(c) for c in conv))
    if reference is not None:
        ref = {r["seed"]: r["commit"]["target"] for r in reference if r.get("commit")}
        err = [100.0 * float(np.linalg.norm(np.asarray(r["commit"]["target"]) -
                                            np.asarray(ref[r["seed"]])))
               for r in records if r.get("commit") and r["seed"] in ref]  # fmt: skip
        if err:
            out["aim_error_vs_ceiling_cm"] = pl.median_ci(err)
        out["paired_vs_ceiling"] = pl.paired_interval(
            [bool(r["success"]) for r in reference], out["per_reset"]
        )
    return out


def run(args) -> dict:
    out_dir = Path(args.output)
    if out_dir.exists():
        raise SystemExit(f"refusing: {out_dir} exists")
    pl.check_seed_ranges()
    f_seeds = pl.seeds_of("F", debug=args.debug)
    c_seeds = pl.seeds_of("corpus", debug=args.debug)
    pl.check_seeds(f_seeds + c_seeds, debug=args.debug)
    dirty = hz.tracked_tree_dirty()
    if dirty and not args.debug:
        raise SystemExit("refusing: the tracked tree is dirty (commit the code first)")
    started = time.monotonic()
    report = {
        "what": "TASK-082 design note development feasibility (not gated, no world model)",
        "revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True
        ).stdout.strip(),  # fmt: skip
        "tracked_tree_dirty": bool(dirty),
        "debug": bool(args.debug),
        "laws": {k: pl.LAWS[k] for k in args.laws},
        "seeds": {"F": [f_seeds[0], f_seeds[-1]], "corpus": [c_seeds[0], c_seeds[-1]]},
        "salts": pl.SALTS,
        "move": {"family": pl.MOVE_FAMILY, "rho_cm": pl.MOVE_RHO_CM},
        "a_lo": pl.A_LO,
        "argv": sys.argv[1:],
        "gpu": "none",
        "laws_out": {},
        "stages": {},
    }
    found = hz.check_evidence(Path(args.evidence))
    report["evidence_sha256"] = found["sha256"]
    config = {"torch_threads": pt.WORKER_TORCH_THREADS,
              "p3_checkpoint": hz.p3_checkpoint(Path(args.evidence))}  # fmt: skip
    pool = DevPool(int(args.workers), config)
    try:
        p_readout, encoder = hz.refit_p_readout(report, pool, Path(args.evidence), found["run1"])
        seeds = f_seeds + c_seeds
        resets = {s: pl.reset_of(s) for s in seeds}
        est = hz.cohort_estimates(pool, p_readout, encoder, seeds, resets, 1800.0)
        report["render_disagreements"] = est.pop("_render_disagreements", {})
        print(f"estimates ready ({time.monotonic() - started:.0f} s)", flush=True)
        for law in args.laws:
            law_out = {"arms": {}}
            runs = {arm: run_arm(pool, arm, law, f_seeds, resets, est) for arm in CHECK_ARMS}
            draws = {s: pl.corpus_aim(s) for s in c_seeds}
            corpus = run_arm(
                pool, "collect", law, c_seeds, resets, est,
                per_seed=lambda s, d=draws: {"a": d[s][0], "b_m": d[s][1]},
            )  # fmt: skip
            rows = [r for r in corpus
                    if r.get("commit") and r["commit"].get("plate_s1") is not None]  # fmt: skip
            p = np.asarray([r["commit"]["p"] for r in rows])
            h = np.asarray([r["commit"]["h"] for r in rows])
            g = np.asarray([r["commit"]["target"] for r in rows])
            y = np.asarray([r["commit"]["plate_s1"] for r in rows])
            coef = c1.sysid_fit(p, h, g, y)
            krr = pl.krr_fit(p, h, g, y)
            offline = pl.cross_fitted_errors(p, h, g, y)
            law_out["corpus"] = {
                "roots": len(corpus),
                "rows": len(rows),
                "successes": int(sum(bool(r["success"]) for r in corpus)),
                "plate_motion_405_to_s1_cm": pl.median_ci(100 * np.linalg.norm(y - p, axis=1)),
                "palm_travel_cm": pl.median_ci(100 * np.linalg.norm(g - h, axis=1)),
                "offline_error_cm": {k: pl.median_ci(v) for k, v in offline.items()},
                "krr": {
                    "length": krr["length"],
                    "lambda": krr["lambda"],
                    "cv_median_cm": 100 * krr["cv_median_m"],
                },  # fmt: skip
                "linear_coef": coef.tolist(),
            }
            runs["H-sysid-true"] = run_arm(pool, "H-sysid-true", law, f_seeds, resets, est,
                                           extra={"sysid_coef": coef.tolist()})  # fmt: skip
            runs["H-sysid-krr-true"] = run_arm(
                pool, "H-sysid-krr-true", law, f_seeds, resets, est,
                extra={"krr": pl.model_to_json(krr)},
            )  # fmt: skip
            ceiling = runs["H-final"]
            for arm, records in runs.items():
                law_out["arms"][arm] = arm_summary(records, None if arm == "H-final" else ceiling)
            law_out["attempts"] = {a: [{k: v for k, v in r.items()} for r in rec]
                                   for a, rec in runs.items()}  # fmt: skip
            law_out["corpus_attempts"] = [
                {k: r.get(k) for k in ("seed", "success", "commit", "termination_reason")}
                for r in corpus
            ]
            report["laws_out"][law] = law_out
    finally:
        report["pool_close"] = pool.close()
    report["seconds"] = time.monotonic() - started
    out_dir.mkdir(parents=True)
    (out_dir / "report.json").write_text(json.dumps(report, default=float))
    print(f"wrote {out_dir / 'report.json'} in {report['seconds']:.0f} s", flush=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", required=True)
    ap.add_argument("--evidence", required=True, help="TASK-072 run-1's evidence root")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--debug", action="store_true", help="debug seeds only; nothing is read")
    ap.add_argument("--laws", nargs="+", default=list(LAW_ORDER), choices=LAW_ORDER)
    args = ap.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
