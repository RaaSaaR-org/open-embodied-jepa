"""TASK-073 ranking stage (O3, O4) on cohort R (53040-53071), after the train stage.

Protocol ``docs/experiments/apple_wm_critic_v2.md`` §6.3-§6.4. Each R reset runs P-reread with
the plate shift of K0's choice; at each of the five R points the 25 aims around R-mid's
incumbent are (a) run for 16 closed-loop commands in cloned simulator state (the true outcome,
privileged, scoring only) and (b) turned into stand-in chunks that every ranker scores: the three
W seeds, the blind rankers copy-last, N, L-shuf (the start latent of reset (i + 1) mod n at the
same point) and prior-distance (the aim's distance from the incumbent). Cohort R is never fitted
on. The blind rankers' results are written to the report before any W ranker is scored.

    uv run --no-sync python scripts/rank_wm_critic_v2.py \\
        --output outputs/task073-rank/run-1 --train-report <train report> --train-sha256 <sha> \\
        --evidence /home/huhn/develop/emai/worktrees/task072-run
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import wm_critic_v2 as wc  # noqa: E402


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN = _load("_task073_runner", "scripts/run_wm_critic_v2.py")
R65 = RUN.R65


def load_critics(critic_files: dict, schema) -> dict:
    """The critics the rankers use, loaded in the main process on the CPU."""
    from embodied_jepa import token_dynamics as td
    from embodied_jepa.models.frozen_tokens import frozen_token_model
    from embodied_jepa.models.latent_critic import CopyLast, LatentCritic, RidgeReadout, ZeroActions

    with np.load(critic_files["r_off"]) as data:
        r_off = RidgeReadout.from_state({k: data[k] for k in data.files})

    def model(path):
        m = frozen_token_model(td.BACKEND)(schema, device="cpu", seed=0, config=td.MODEL_CONFIG)
        m.load(path)
        m.eval()
        return m

    critics = {
        name: LatentCritic(model(path), r_off, wc.O_STAR_CM)
        for name, path in critic_files["models"].items()
        if name.startswith("W")
    }
    critics["N"] = LatentCritic(
        ZeroActions(model(critic_files["models"]["N"])), r_off, wc.O_STAR_CM
    )
    critics["copy"] = LatentCritic(CopyLast(), r_off, wc.O_STAR_CM)
    return critics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-report", required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--k0-report", required=True)
    parser.add_argument("--k0-sha256", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="cohort R stand-ins on smoke seeds; mechanics only, nothing is read",
    )
    args = parser.parse_args(argv)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    evidence = Path(args.evidence).resolve()
    report = {
        "protocol": wc.PROTOCOL,
        "task": wc.TASK,
        "mode": "rank",
        "smoke": bool(args.smoke),
        "outcome": None,
        "stages": {},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    clock = R65.Clock(wc.CAPS_SECONDS["global"])
    try:
        kind = "smoke" if args.smoke else "rank"
        manifest = RUN.preflight(report, kind, evidence)
        train = RUN.read_stage_report(
            args.train_report, args.train_sha256, "TRAIN-COMPLETE", args.smoke
        )["stages"]
        k0 = RUN.read_stage_report(args.k0_report, args.k0_sha256, "K0-PASS", args.smoke)
        selection = k0["stages"]["k0"]["selection"]
        step, cm = int(selection["shift_step"]), int(selection["shift_cm"])
        resets, seeds = RUN.cohort(manifest, "R", args.smoke)
        critic_files = train["critic"]
        pool = RUN.Pool(
            wc.SIM_WORKERS,
            {
                "p3_checkpoint": RUN.p3_checkpoint(evidence),
                "torch_threads": 1,
                "r_mid": critic_files["r_mid"],
            },
        )
        report["_pool"] = pool
        readout, encoder = RUN.refit_p_readout(report, pool, evidence, report["_run1"])
        report["cohort_first_render_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        est = RUN.cohort_estimates(pool, readout, encoder, seeds, resets, 1800.0)
        points = wc.R_POINTS[step]
        tasks = [
            {
                "kind": "rank",
                "seed": s,
                "reset": {k: resets[s][k] for k in ("object_xy", "plate_xy")},
                "shift": {"step": step, "vector": resets[s]["shift_m"][str(cm)]},
                "estimates": est[s]["estimates"],
                "points": points,
            }
            for s in seeds
        ]
        results = pool.map(tasks, 7200.0, "rank")
        groups = []
        for r in results:
            if r["post_look_frame_sha256"] != est[r["seed"]]["frame_sha256"]:
                raise wc.GuardError(f"G-frame: seed {r['seed']} post-look frame differs")
            for g in r["groups"]:
                groups.append(g | {"seed": r["seed"], "point_index": points.index(g["step"])})
        complete = {int(p): sum(g["step"] == p for g in groups) for p in points}
        from embodied_jepa.first_policy_v2_runtime import make_robot

        schema = make_robot().state_schema
        critics = load_critics(critic_files, schema)
        from embodied_jepa import wm_critic_v2_offline as off

        blind_only = off.rank_statistics(
            groups, {k: v for k, v in critics.items() if k in ("copy", "N")}
        )
        report["stages"]["blind_rankers"] = blind_only  # written before any W ranker is scored
        R65.write_report(out / "blind.json", blind_only)
        stats = off.rank_statistics(groups, critics)
        RUN.end_checks(report, manifest, kind)
        report["stages"]["rank"] = {
            "shift_step": step,
            "shift_cm": cm,
            "points": list(points),
            "groups_per_point": complete,
            "statistics": stats,
        }
        report["outcome"] = "RANK-COMPLETE"
    except Exception as error:  # noqa: BLE001 - every failure is V
        report["outcome"] = "V"
        report["void_reason"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        RUN.log(f"VOID: {report['void_reason']}")
    finally:
        pool = report.pop("_pool", None)
        if pool is not None:
            pool.close()
        report.pop("_run1", None)
        report["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["total_seconds"] = clock.elapsed()
        R65.write_report(out / "report.json", report)
    return 0 if report["outcome"] != "V" else 1


if __name__ == "__main__":
    sys.exit(main())
