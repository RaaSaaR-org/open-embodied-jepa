"""SO-101 S3 (TASK-099): collect the SO-101 play corpus (development data, no gate, no claim).

Shards run in parallel processes (spawn); each shard is its own ``SchemaDatasetStore`` (LeRobot
v3 PNG-in-Parquet, ``joint_delta_so101_v0``) under ``<out>/shard-SS`` with one sidecar per
episode, sealed with its train/val/test split. Never overwrites: an existing complete shard is
skipped, an incomplete one is refused. Seeds are per-episode and deterministic
(``numpy.random.default_rng([9901, seed])``); a shard's seeds are a fixed range.

Two kinds of run, both rendering both cameras with NVIDIA EGL, so both go through the GPU lock:

    MUJOCO_GL=egl scripts/gpu_run.sh --wait --board -- \\
        .venv/bin/python scripts/collect_so101_play.py --kind pilot --out data/so101-play-pilot-v1
    MUJOCO_GL=egl scripts/gpu_run.sh --wait --board -- \\
        .venv/bin/python scripts/collect_so101_play.py --kind corpus --out data/so101-play-v1

``pilot``: seeds 990100-990163 in 2 shards of 32 (2 val, 2 test each). ``corpus``
(``so101-play-v1``): seeds 991000-993999 in 30 shards of 100 (5 val, 5 test each). Both need a
clean tracked tree. ``--kind debug`` (tests) runs any seeds without rendering (black frames).
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

from embodied_jepa import so101_play_corpus as pc  # noqa: E402

KINDS = {
    # kind -> (first seed, shard size, shards, n_val, n_test)
    "pilot": (pc.PILOT_FIRST_SEED, 32, 2, 2, 2),
    "corpus": (pc.CORPUS_FIRST_SEED, pc.SHARD_SIZE, pc.CORPUS_SHARDS, 5, 5),
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


def code_hashes() -> dict:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in pc.CODE_FILES
        if (ROOT / name).is_file()
    }


def make_robot(job):
    if job["render"]:
        return pc.make_play_robot(image_size=job["image_size"])
    import numpy as np

    robot = pc.make_play_robot(image_size=job["image_size"], render=False)
    size = job["image_size"]
    robot.sim.render = lambda camera="wrist_rgb": np.zeros((size, size, 3), np.uint8)
    return robot


def run_shard(job: dict) -> dict:
    report = _run_shard(job)
    path = Path(job["out"]) / "reports" / f"shard-{job['shard']:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if report["status"] not in ("exists-complete", "exists-excluded") and not path.exists():
        pc.write_json(path, report)
    return report


def _run_shard(job: dict) -> dict:
    from embodied_jepa.schema_data import SchemaDatasetStore

    shard, seeds, out = job["shard"], job["seeds"], Path(job["out"])
    root = out / f"shard-{shard:02d}"
    report = {"shard": shard, "seeds": [seeds[0], seeds[-1]], "episodes": []}
    if root.exists():
        if (root / "EXCLUDED.json").exists():
            report["status"] = "exists-excluded"
            return report
        manifest = json.loads((root / "meta/jepa_manifest.json").read_text())
        if manifest.get("splits") is not None:
            existing = SchemaDatasetStore(root)
            report.update(
                status="exists-complete",
                stored=len(existing.manifest["episodes"]),
                manifest_hash=existing.manifest_hash,
                splits={k: len(v) for k, v in existing.manifest["splits"].items()},
            )
            return report
        raise FileExistsError(f"incomplete shard exists, inspect before recovery: {root}")
    free = pc.free_gib(out)
    if free < job["disk_floor_gib"]:
        report.update(status="skipped-disk", free_gib=free)
        return report
    started = time.time()
    robot = make_robot(job)
    report["renderer"] = renderer_string() if job["render"] else "disabled (debug)"
    store = pc.create_store(
        root,
        robot,
        {
            "task": pc.TASK,
            "doc": pc.DOC,
            "kind": job["kind"],
            "scene": pc.SCENE_VERSION,
            "policy": pc.POLICY_VERSION,
            "collector": "privileged scripted play mixture; not a learned result",
            "revision": job["revision"],
            "code_sha256": job["code_sha256"],
            "shard": shard,
            "policy_salt": pc.POLICY_SALT,
            "split_salt": pc.SPLIT_SALT,
            "image_size": job["image_size"],
            "cameras": list(pc.CAMERAS),
            "renderer": report["renderer"],
            "license": "Apache-2.0 (project-generated simulation data; Menagerie SO-101 model "
            "Apache-2.0)",
        },
    )
    stored = []
    for seed in seeds:
        t0 = time.time()
        row = {"seed": seed}
        try:
            ep = pc.run_episode(robot, seed, commands=job["commands"])
        except Exception as error:  # the episode is discarded; its type is recorded
            row.update(
                status="error",
                error=repr(error),
                error_type=type(error).__name__,
                seconds=time.time() - t0,
            )
            report["episodes"].append(row)
            continue
        commands = len(ep["actions"])
        row.update(commands=commands, mode=ep["mode"], stop_reason=ep["stop_reason"])
        row["collect_seconds"] = time.time() - t0
        if commands < job["min_commands"]:
            row["status"] = "discarded-short"
        else:
            metadata = pc.write_episode(store, robot, ep)
            row.update(status="stored", facts=metadata["facts"])
            stored.append(seed)
        row["seconds"] = time.time() - t0
        report["episodes"].append(row)
    robot.close()
    if len(stored) < job["n_val"] + job["n_test"] + 1:
        pc.write_json(root / "EXCLUDED.json", {"stored": len(stored), "reason": "too few"})
        report.update(status="excluded-unsealed", stored=len(stored), seconds=time.time() - started)
        return report
    split = pc.split_assignment(stored, n_val=job["n_val"], n_test=job["n_test"])
    store.freeze_split_assignments(
        {name: [pc.episode_id(s) for s in split[name]] for name in ("train", "val", "test")}
        | {"holdout": []},
        provenance={
            "rule": f"per shard, rank by sha256('{pc.SPLIT_SALT}:<seed>'): lowest "
            f"{job['n_val']} val, next {job['n_test']} test, rest train; one episode per session",
            "split_salt": pc.SPLIT_SALT,
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
    return report


def jobs_for(args, revision, hashes) -> list[dict]:
    if args.kind == "debug":
        first, size, shards, n_val, n_test = (
            args.first_seed,
            args.shard_size,
            args.n_shards,
            1,
            1,
        )
    else:
        first, size, shards, n_val, n_test = KINDS[args.kind]
    selected = args.shards if args.shards is not None else list(range(shards))
    jobs = []
    for shard in selected:
        if not 0 <= shard < shards:
            raise SystemExit(f"shard {shard} outside 0..{shards - 1}")
        seeds = pc.shard_seeds(shard, first=first, size=size)
        if args.kind != "debug" and set(seeds) & set(pc.DEV_SEEDS):
            raise SystemExit("development seeds are never stored")
        jobs.append(
            {
                "kind": args.kind,
                "shard": shard,
                "seeds": seeds,
                "out": args.out,
                "commands": args.commands if args.kind == "debug" else pc.EPISODE_COMMANDS,
                "min_commands": min(pc.MIN_STORED_COMMANDS, args.commands),
                "n_val": n_val,
                "n_test": n_test,
                "disk_floor_gib": 0.0 if args.kind == "debug" else pc.DISK_FLOOR_GIB,
                "render": args.kind != "debug",
                "image_size": args.image_size if args.kind == "debug" else pc.IMAGE_SIZE,
                "revision": revision,
                "code_sha256": hashes,
            }
        )
    return jobs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("pilot", "corpus", "debug"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--shards", type=int, nargs="*", default=None)
    # debug only:
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=4)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--commands", type=int, default=pc.EPISODE_COMMANDS)
    parser.add_argument("--image-size", type=int, default=16)
    args = parser.parse_args(argv)

    dirty = git("status", "--porcelain", "--untracked-files=no")
    revision = git("rev-parse", "HEAD")
    if args.kind != "debug":
        if os.environ.get("MUJOCO_GL") != "egl":
            raise SystemExit("set MUJOCO_GL=egl")
        if os.environ.get("GPU_RUN_LOCKED") != "1":
            raise SystemExit("this run renders on the GPU: run it under scripts/gpu_run.sh")
        if dirty:
            raise SystemExit("a pilot or corpus run needs a clean tracked tree")
    hashes = code_hashes()
    jobs = jobs_for(args, revision, hashes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.workers <= 1:
        reports = [run_shard(job) for job in jobs]
    else:
        with get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
            reports = list(pool.imap_unordered(run_shard, jobs))
    reports.sort(key=lambda r: r["shard"])
    summary = {
        "task": pc.TASK,
        "doc": pc.DOC,
        "kind": args.kind,
        "revision": revision,
        "dirty": bool(dirty),
        "code_sha256": hashes,
        "host": platform.node(),
        "python": platform.python_version(),
        "args": vars(args),
        "workers": args.workers,
        "seconds": time.time() - started,
        "renderers": sorted({r.get("renderer", "") for r in reports} - {""}),
        "policy_salt": pc.POLICY_SALT,
        "split_salt": pc.SPLIT_SALT,
        "image_size": jobs[0]["image_size"] if jobs else None,
        "commands": jobs[0]["commands"] if jobs else None,
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
                "seconds": r.get("seconds"),
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
    complete = sum(r.get("status") in ("complete", "exists-complete") for r in reports)
    print(json.dumps({"seconds": round(summary["seconds"], 1), "complete_shards": complete}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
