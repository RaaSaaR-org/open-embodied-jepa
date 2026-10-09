"""TASK-085: collect the play corpus ``play-v1`` (protocol docs/experiments/play_corpus_v1.md).

Shards run in parallel processes; each shard is its own ``DatasetStore`` under ``<out>/shard-SS``
with a sidecar per episode. Never overwrites: an existing shard directory is skipped if it is
complete (splits frozen) and refused otherwise. Run the whole job under
``scripts/gpu_run.sh --wait --min-free-gib 8 --board`` (frames are rendered with NVIDIA EGL).

    MUJOCO_GL=egl scripts/gpu_run.sh --wait --min-free-gib 8 --board -- \
        .venv/bin/python scripts/collect_play_corpus.py --out data/play-v1 --workers 12
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
    from embodied_jepa.data import DatasetStore

    shard, seeds, out = job["shard"], job["seeds"], Path(job["out"])
    root = out / f"shard-{shard:02d}"
    report = {"shard": shard, "seeds": [seeds[0], seeds[-1]], "episodes": []}
    if root.exists():
        if (root / "EXCLUDED.json").exists():
            report["status"] = "exists-excluded"
            return report
        manifest = json.loads((root / "meta/jepa_manifest.json").read_text())
        if manifest.get("splits") is not None:
            report["status"] = "exists-complete"
            return report
        raise FileExistsError(f"incomplete shard exists, inspect before recovery: {root}")
    free = pc.free_gib(out)
    if free < job["disk_floor_gib"]:
        report.update(status="skipped-disk", free_gib=free)
        return report
    started = time.time()
    robot = pc.make_play_robot()
    report["renderer"] = renderer_string()
    store = DatasetStore.create(
        root,
        fps=pc.FPS,
        state_schema=robot.state_schema,
        action_manifest=robot.manifest,
        provenance={
            "task": "TASK-085",
            "protocol": "docs/experiments/play_corpus_v1.md",
            "scene": pc.SCENE_VERSION,
            "policy": pc.POLICY_VERSION,
            "collector": "privileged scripted play mixture; not a learned result",
            "revision": job["revision"],
            "shard": shard,
            "policy_salt": pc.POLICY_SALT,
            "split_salt": pc.SPLIT_SALT,
            "renderer": report["renderer"],
            "license": "Apache-2.0 (project-generated simulation data)",
        },
    )
    stored = []
    for seed in seeds:
        t0 = time.time()
        row = {"seed": seed}
        try:
            ep = pc.run_episode(robot, seed, commands=job["commands"])
        except Exception as error:  # a MuJoCo error discards the episode (protocol §5)
            row.update(status="error", error=repr(error), seconds=time.time() - t0)
            report["episodes"].append(row)
            continue
        commands = len(ep["actions"])
        row.update(commands=commands, mode=ep["mode"], stop_reason=ep["stop_reason"])
        if commands < job["min_commands"]:
            row["status"] = "discarded-short"
        else:
            metadata = pc.write_episode(store, robot, ep)
            row.update(status="stored", facts=metadata["facts"])
            row["reduced_fraction"] = float(ep["reduced"].mean()) if commands else 0.0
            stored.append(seed)
        row["seconds"] = time.time() - t0
        report["episodes"].append(row)
    if len(stored) < job["n_val"] + job["n_test"] + 1:
        # protocol §7: a shard too small to split is left unsealed, recorded and excluded
        pc.write_json(root / "EXCLUDED.json", {"stored": len(stored), "reason": "too few"})
        report.update(status="excluded-unsealed", stored=len(stored), seconds=time.time() - started)
        return report
    split = pc.split_assignment(stored, n_val=job["n_val"], n_test=job["n_test"])
    store.freeze_split_assignments(
        {name: [pc.episode_id(s) for s in split[name]] for name in ("train", "val", "test")}
        | {"holdout": []},
        provenance={
            "rule": f"per shard, rank by sha256('{pc.SPLIT_SALT}:<seed>'): lowest "
            f"{job['n_val']} val, next {job['n_test']} test, rest train (protocol §7)",
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--shards", type=int, nargs="*", default=None)
    parser.add_argument("--first-seed", type=int, default=pc.CORPUS_FIRST_SEED)
    parser.add_argument("--shard-size", type=int, default=pc.SHARD_SIZE)
    parser.add_argument("--n-shards", type=int, default=pc.SHARDS)
    parser.add_argument("--commands", type=int, default=pc.EPISODE_COMMANDS)
    parser.add_argument("--n-val", type=int, default=pc.VAL_PER_SHARD)
    parser.add_argument("--n-test", type=int, default=pc.TEST_PER_SHARD)
    parser.add_argument("--debug", action="store_true", help="debug seeds 86000-86099 only")
    parser.add_argument("--evidence", default=None)
    args = parser.parse_args(argv)

    if os.environ.get("MUJOCO_GL") != "egl":
        raise SystemExit("set MUJOCO_GL=egl (protocol §13)")
    if not args.debug and os.environ.get("GPU_RUN_LOCKED") != "1":
        raise SystemExit("the corpus run renders on the GPU: run it under scripts/gpu_run.sh")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    revision = git("rev-parse", "HEAD")
    shards = args.shards if args.shards is not None else list(range(args.n_shards))
    jobs = []
    for shard in shards:
        seeds = pc.shard_seeds(shard, first=args.first_seed, size=args.shard_size)
        allowed = pc.DEBUG_SEEDS if args.debug else range(850000, 853200)
        if not all(s in allowed for s in seeds):
            raise SystemExit(f"shard {shard} seeds {seeds[0]}-{seeds[-1]} outside {allowed}")
        jobs.append(
            {
                "shard": shard,
                "seeds": seeds,
                "out": args.out,
                "commands": args.commands,
                "min_commands": pc.MIN_STORED_COMMANDS,
                "n_val": args.n_val,
                "n_test": args.n_test,
                "disk_floor_gib": pc.DISK_FLOOR_GIB,
                "revision": revision,
            }
        )
    if args.debug and (args.n_val, args.n_test) != (1, 1):
        raise SystemExit("debug shards use 1 val and 1 test (protocol §7, §12)")
    if not args.debug:
        fixed = (args.first_seed, args.shard_size, args.commands, args.n_val, args.n_test)
        if fixed != (pc.CORPUS_FIRST_SEED, pc.SHARD_SIZE, pc.EPISODE_COMMANDS, 5, 5):
            raise SystemExit("the corpus run uses the protocol's seeds, sizes and splits")
        if dirty:
            raise SystemExit("the corpus run needs a clean tracked tree")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        reports = list(pool.imap_unordered(run_shard, jobs))
    reports.sort(key=lambda r: r["shard"])
    reports_dir = out / "reports"
    reports_dir.mkdir(exist_ok=True)
    for report in reports:
        path = reports_dir / f"shard-{report['shard']:02d}.json"
        if not path.exists():
            pc.write_json(path, report)
    summary = {
        "task": "TASK-085",
        "protocol": "docs/experiments/play_corpus_v1.md",
        "revision": revision,
        "dirty": bool(dirty),
        "debug": args.debug,
        "host": platform.node(),
        "python": platform.python_version(),
        "args": vars(args),
        "seconds": time.time() - started,
        "renderers": sorted({r.get("renderer", "") for r in reports} - {""}),
        "shards": [
            {k: r.get(k) for k in ("shard", "status", "stored", "manifest_hash", "splits", "seeds")}
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
        for source in [path, *sorted(reports_dir.glob("*.json"))]:
            target = evidence / source.name
            target.write_bytes(source.read_bytes())
        lines = [
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}"
            for p in sorted(evidence.iterdir())
            if p.name != "SHA256SUMS" and p.is_file()
        ]
        (evidence / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    stats = {s: sum(1 for r in reports if r.get("status") == s) for s in {"complete"}}
    print(json.dumps({"seconds": round(summary["seconds"], 1), **stats}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
