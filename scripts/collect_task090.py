"""TASK-090: collect play-v2g's or play-v1p's added episodes, or Stage 0's debug collection
(protocol docs/experiments/grasp_play_corpus_v1.md §2, §4, §10, §14).

Shards of 100 seeds run in parallel processes; each shard is its own ``DatasetStore`` under
``<out>/shard-SS`` with a sidecar per episode, sealed by the per-shard split rule (salt 9005; 5 val
and 5 test per shard). Never overwrites: a complete shard is skipped, an incomplete one refused.
Run under ``scripts/gpu_run.sh --wait --min-free-gib 8 --board`` (frames are rendered with EGL).

    MUJOCO_GL=egl scripts/gpu_run.sh --wait --min-free-gib 8 --board -- \\
        .venv/bin/python scripts/collect_task090.py --corpus play-v2g --out data/play-v2g

``--corpus debug-grasp`` (930000-930299) and ``debug-v0`` (930300-930399) are Stage 0's debug
collection; the corpus runs need a clean tracked tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import play_corpus as pc  # noqa: E402
from embodied_jepa import play_corpus_grasp as pg  # noqa: E402

DISK_FLOOR_GIB = 10.0
CHOICES = {
    "play-v2g": ("grasp", False),
    "play-v1p": ("v0", False),
    "debug-grasp": ("grasp", True),
    "debug-v0": ("v0", True),
}


def git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def renderer_string() -> str:
    try:
        from OpenGL import GL

        return " / ".join(
            GL.glGetString(x).decode() for x in (GL.GL_VENDOR, GL.GL_RENDERER, GL.GL_VERSION)
        )
    except Exception as error:  # pragma: no cover - diagnostic only
        return f"unavailable: {error!r}"


def run_shard(job: dict) -> dict:
    report = _run_shard(job)
    path = Path(job["out"]) / "reports" / f"shard-{job['shard']:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if report["status"] not in ("exists-complete", "exists-excluded") and not path.exists():
        pc.write_json(path, report)
    return report


def _run_shard(job: dict) -> dict:
    from embodied_jepa.data import DatasetStore

    shard, seeds, out, policy = job["shard"], job["seeds"], Path(job["out"]), job["policy"]
    root = out / f"shard-{shard:02d}"
    report = {"shard": shard, "seeds": [seeds[0], seeds[-1]], "policy": policy, "episodes": []}
    if root.exists():
        if (root / "EXCLUDED.json").exists():
            report["status"] = "exists-excluded"
            return report
        manifest = json.loads((root / "meta/jepa_manifest.json").read_text())
        if manifest.get("splits") is not None:
            existing = DatasetStore(root)
            report.update(
                status="exists-complete",
                stored=len(existing.manifest["episodes"]),
                manifest_hash=existing.manifest_hash,
                splits={k: len(v) for k, v in existing.manifest["splits"].items()},
            )
            return report
        raise FileExistsError(f"incomplete shard exists, inspect before recovery: {root}")
    free = pc.free_gib(out)
    if free < DISK_FLOOR_GIB:
        report.update(status="skipped-disk", free_gib=free)
        return report
    started = time.time()
    robot = pc.make_play_robot()
    report["renderer"] = renderer_string()
    version = pg.POLICY_VERSION if policy == "grasp" else pc.POLICY_VERSION
    store = DatasetStore.create(
        root,
        fps=pc.FPS,
        state_schema=robot.state_schema,
        action_manifest=robot.manifest,
        provenance={
            "task": pg.TASK,
            "protocol": pg.PROTOCOL,
            "corpus": job["corpus"],
            "scene": pc.SCENE_VERSION,
            "policy": version,
            "collector": "privileged scripted play mixture; not a learned result",
            "revision": job["revision"],
            "shard": shard,
            "policy_salt": pg.policy_salt(policy),
            "split_salt": pg.SPLIT_SALT,
            "renderer": report["renderer"],
            "license": "Apache-2.0 (project-generated simulation data)",
        },
    )
    stored = []
    for seed in seeds:
        t0 = time.time()
        row = {"seed": seed}
        try:
            ep = pg.run_episode(robot, seed, policy=policy)
        except Exception as error:  # discards the episode (TASK-085 §5); type recorded
            row.update(
                status="error",
                error=repr(error),
                error_type=type(error).__name__,
                seconds=time.time() - t0,
            )
            report["episodes"].append(row)
            continue
        commands = len(ep["actions"])
        row.update(
            commands=commands,
            mode=ep["mode"],
            stop_reason=ep["stop_reason"],
            pick_checks=ep["pick_checks"],
        )
        if commands < pc.MIN_STORED_COMMANDS:
            row["status"] = "discarded-short"
        else:
            metadata = pg.write_episode(store, robot, ep)
            row.update(status="stored", facts=metadata["facts"])
            stored.append(seed)
        row["seconds"] = time.time() - t0
        report["episodes"].append(row)
    if len(stored) < pc.VAL_PER_SHARD + pc.TEST_PER_SHARD + 1:
        pc.write_json(root / "EXCLUDED.json", {"stored": len(stored), "reason": "too few"})
        report.update(status="excluded-unsealed", stored=len(stored), seconds=time.time() - started)
        return report
    split = pg.split_assignment(stored)
    store.freeze_split_assignments(
        {name: [pc.episode_id(s) for s in split[name]] for name in ("train", "val", "test")}
        | {"holdout": []},
        provenance={
            "rule": f"per shard, rank by sha256('{pg.SPLIT_SALT}:<seed>'): lowest "
            f"{pc.VAL_PER_SHARD} val, next {pc.TEST_PER_SHARD} test, rest train (TASK-090 §2.1)",
            "split_salt": pg.SPLIT_SALT,
        },
        heldout_combinations=(),
    )
    report.update(
        status="complete",
        stored=len(stored),
        manifest_hash=store.manifest_hash,
        splits={k: len(v) for k, v in split.items()},
        seconds=time.time() - started,
    )
    robot.close()
    return report


def shard_jobs(corpus: str, revision: str, out: str) -> list[dict]:
    policy, debug = CHOICES[corpus]
    if debug:
        seeds = list(pg.DEBUG_COLLECTION[policy])
    else:
        seeds = list(pg.corpus_seeds(corpus))
    jobs = []
    for shard in range(len(seeds) // pc.SHARD_SIZE):
        chunk = seeds[pc.SHARD_SIZE * shard : pc.SHARD_SIZE * (shard + 1)]
        for s in chunk:
            pg.check_collection_seed(policy, s, debug=debug)
        jobs.append(
            {
                "corpus": corpus,
                "policy": policy,
                "shard": shard,
                "seeds": chunk,
                "out": out,
                "revision": revision,
            }
        )
    return jobs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, choices=sorted(CHOICES))
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--evidence", default=None)
    args = parser.parse_args(argv)

    if os.environ.get("MUJOCO_GL") != "egl":
        raise SystemExit("set MUJOCO_GL=egl")
    if os.environ.get("GPU_RUN_LOCKED") != "1":
        raise SystemExit("collection renders on the GPU: run it under scripts/gpu_run.sh")
    _, debug = CHOICES[args.corpus]
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if not debug and dirty:
        raise SystemExit("a corpus run needs a clean tracked tree")
    revision = git("rev-parse", "HEAD")
    jobs = shard_jobs(args.corpus, revision, args.out)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        reports = list(pool.imap_unordered(run_shard, jobs))
    reports.sort(key=lambda r: r["shard"])
    summary = {
        "task": pg.TASK,
        "protocol": pg.PROTOCOL,
        "corpus": args.corpus,
        "revision": revision,
        "dirty": bool(dirty),
        "debug": debug,
        "host": platform.node(),
        "python": platform.python_version(),
        "args": vars(args),
        "seconds": time.time() - started,
        "renderers": sorted({r.get("renderer", "") for r in reports} - {""}),
        "policy_salt": pg.policy_salt(CHOICES[args.corpus][0]),
        "split_salt": pg.SPLIT_SALT,
        "shards": [
            {
                **{
                    k: r.get(k)
                    for k in ("shard", "status", "stored", "manifest_hash", "splits", "seeds")
                },
                "discarded_short": sum(
                    e.get("status") == "discarded-short" for e in r.get("episodes", [])
                ),
                "errors": sum(e.get("status") == "error" for e in r.get("episodes", [])),
                "error_types": sorted(
                    {e["error_type"] for e in r.get("episodes", []) if "error_type" in e}
                ),
            }
            for r in reports
        ],
    }
    path = pc.corpus_manifest_path(out)
    if path.exists():
        previous = json.loads(path.read_text())
        summary["previous_runs"] = previous.get("previous_runs", []) + [
            {k: previous.get(k) for k in ("revision", "seconds", "shards")}
        ]
    pc.write_json(path, summary)
    if args.evidence:
        evidence = Path(args.evidence)
        evidence.mkdir(parents=True, exist_ok=True)
        for source in [path, *sorted((out / "reports").glob("*.json"))]:
            (evidence / source.name).write_bytes(source.read_bytes())
        lines = [
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}"
            for p in sorted(evidence.iterdir())
            if p.name != "SHA256SUMS" and p.is_file()
        ]
        (evidence / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    complete = sum(r.get("status") in ("complete", "exists-complete") for r in reports)
    print(json.dumps({"seconds": round(summary["seconds"], 1), "complete": complete}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
