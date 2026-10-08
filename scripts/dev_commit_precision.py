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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--old-features", required=True)
    ap.add_argument("--old-fits", required=True)
    ap.add_argument("--models", nargs=6, required=True)
    ap.add_argument("--stage-r", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--first", type=int, default=DEV_SEEDS[0])
    ap.add_argument("--last", type=int, default=DEV_SEEDS[1])
    args = ap.parse_args(argv)
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
