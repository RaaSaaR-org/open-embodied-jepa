"""TASK-073 corpus: collect ``apple-shift-v2`` (300 e9 roots, 53400-53699) after K0 passes.

Protocol ``docs/experiments/apple_wm_critic_v2.md`` §5.1. One root per corpus seed, from the
plan fixed in ``wm_critic_v2.corpus_plan``: e9 built from the post-shift truth plus the root's
mis-aim (never used for behaviour cloning), TASK-048's noise on its commands, the plate shift at
K0's step and |d| on the 75 % shifted roots, then the task's settle. The episode schema is
``apple-look-v2``'s plus the plate trajectory (``plate``, a privileged label). The test split
is written and sealed but never decoded by any TASK-073 stage.

Privileged scripted collector by design; nothing here is a learned result.

    uv run --no-sync python scripts/collect_apple_shift_v2.py \\
        --output data/apple-shift-v2/run-1 --report outputs/task073-corpus/run-1 \\
        --k0-report outputs/task073-k0/run-1/report.json --k0-sha256 <sha256> \\
        --evidence /home/huhn/develop/emai/worktrees/task072-run
"""

from __future__ import annotations

import os

# Owner ruling 2026-09-29: pin BLAS/MKL threading before NumPy or torch is imported (they read
# these at load time); preflight's G-threads checks they equal wm_critic_v2.THREAD_ENV.
os.environ.update(
    {
        "MKL_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": "6",
        "MKL_NUM_THREADS": "6",
        "OPENBLAS_NUM_THREADS": "16",
    }
)

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import first_policy_v2_runtime as rt2  # noqa: E402
from embodied_jepa import wm_critic_v2 as wc  # noqa: E402


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN = _load("_task073_runner", "scripts/run_wm_critic_v2.py")
R65 = RUN.R65
log = RUN.log


def collect_tasks(
    plan, folder: Path, shift_step: int | None, shift_cm: int | None, *, max_steps=None
) -> list[dict]:
    tasks = []
    for root in plan:
        reset = wc.wide_reset_values(root["seed"])
        shift = None
        if root["shifted"]:
            shift = {
                "step": int(shift_step),
                "vector": wc.shift_vector(root["seed"], reset, int(shift_cm)),
            }
        task = {"kind": "collect", "reset": reset, "shift": shift, "folder": str(folder)} | root
        if max_steps is not None:
            task["max_steps"] = int(max_steps)
        tasks.append(task)
    return tasks


def smoke_plan() -> list[dict]:
    """The frozen plan's first 12 roots, moved onto smoke seeds 53962-53973 (8 / 2 / 2)."""
    plan = []
    seeds = range(RUN.SMOKE_CORPUS[0], RUN.SMOKE_CORPUS[1] + 1)
    for i, (root, seed) in enumerate(zip(wc.corpus_plan()[:12], seeds, strict=True)):
        split = "train" if i < 8 else "val" if i < 10 else "test"
        plan.append(
            root
            | {
                "seed": seed,
                "episode_id": f"shift2-smoke-{seed}",
                "session_id": f"shift2-smoke-reset-{seed}",
                "split": split,
            }
        )
    return plan


def seal(corpus: Path, plan: list[dict], episodes: dict, provenance: dict) -> str:
    path = corpus / "manifest.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    splits = {k: [] for k in ("train", "val", "test")}
    for root in plan:
        splits[root["split"]].append(root["episode_id"])
    manifest = {
        "corpus": wc.CORPUS,
        "protocol": wc.PROTOCOL,
        "task": wc.TASK,
        "privileged_scripted_collector": True,
        "learned_control": False,
        "scene_version": wc.SCENE_VERSION,
        "expert": {"class": "resting_expert.RestingPlaceExpert", "kwargs": wc.EXPERT},
        "episode_arrays": list(RUN.rtm.EPISODE_ARRAYS),
        "plan": plan,
        "splits": splits,
        "episodes": episodes,
        "provenance": provenance,
    }
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return rt2.sha256_file(path)


def summary(records: list[dict]) -> dict:
    def block(rows):
        return {
            "roots": len(rows),
            "complete": sum(r["complete"] for r in rows),
            "counted_success": sum(r["success"] for r in rows),
            "at_rest": sum(r["at_rest"] for r in rows),
            "off_plate": sum(r["off_plate"] for r in rows),
            "grasp": sum(r["grasp_before_settle"] for r in rows),
            "hand_contact_any": sum(r["hand_contact_any"] for r in rows),
            "shift_applied": sum(r["shift_applied"] for r in rows),
        }

    out = {"all": block(records)}
    for key in ("split", "shifted", "misaimed", "noise_level"):
        for value in sorted({r[key] for r in records}, key=str):
            out[f"{key}={value}"] = block([r for r in records if r[key] == value])
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, help="the corpus directory (new)")
    parser.add_argument("--report", required=True, help="the collection report directory (new)")
    parser.add_argument("--k0-report", required=True)
    parser.add_argument("--k0-sha256", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="12 smoke-seed roots (53962-53973); mechanics only, nothing is read",
    )
    args = parser.parse_args(argv)
    corpus, out = Path(args.output), Path(args.report)
    for path in (corpus, out):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    out.mkdir(parents=True)
    report = {
        "protocol": wc.PROTOCOL,
        "task": wc.TASK,
        "mode": "corpus",
        "smoke": bool(args.smoke),
        "outcome": None,
        "stages": {},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    clock = R65.Clock(wc.CAPS_SECONDS["global"])
    watch = RUN.install_guards()
    pool = None
    try:
        kind = "smoke" if args.smoke else "corpus"
        manifest = RUN.preflight(report, kind, Path(args.evidence).resolve())
        k0 = RUN.read_stage_report(args.k0_report, args.k0_sha256, "K0-PASS", args.smoke)
        selection = k0["stages"]["k0"]["selection"]
        plan = wc.corpus_plan() if not args.smoke else smoke_plan()
        wc.check_role_seeds("corpus", tuple(r["seed"] for r in plan), smoke=args.smoke)
        (corpus / "episodes").mkdir(parents=True)
        pool = RUN.Pool(
            RUN.sim_workers(),
            {"p3_checkpoint": RUN.p3_checkpoint(Path(args.evidence).resolve()), "torch_threads": 1},
        )
        tasks = collect_tasks(
            plan, corpus / "episodes", selection["shift_step"], selection["shift_cm"]
        )
        # N-d: the corpus stage's pre-render boundary (protocol §7.1, §15)
        report["cohort_first_render_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        records = pool.map(tasks, 3 * 3600.0, "corpus")
        episodes = {
            r["episode_id"]: {"npz_sha256": r["npz_sha256"], "meta_sha256": r["meta_sha256"]}
            for r in records
        }
        sha = seal(
            corpus,
            plan,
            episodes,
            {
                "revision": report["revision"],
                "k0_report_sha256": args.k0_sha256,
                "smoke": bool(args.smoke),
                "shift_step": selection["shift_step"],
                "shift_cm": selection["shift_cm"],
            },
        )
        RUN.end_checks(report, manifest, kind)
        report["stages"]["corpus"] = {
            "manifest_sha256": sha,
            "summary": summary(records),
            "path": str(corpus),
        }
        report["outcome"] = "CORPUS-SEALED"
    except BaseException as error:  # noqa: BLE001 - every failure, signal included, is V
        RUN.void(report, error)
    finally:
        if pool is not None:
            pool.close()
        watch.stop()
        report["memory"] = watch.summary()
        report.pop("_run1", None)
        report["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["total_seconds"] = clock.elapsed()
        R65.write_report(out / "report.json", report)
    return 0 if report["outcome"] != "V" else 1


if __name__ == "__main__":
    sys.exit(main())
