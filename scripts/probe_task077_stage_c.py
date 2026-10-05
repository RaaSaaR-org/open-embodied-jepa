"""Stage C scale probe (development; debug seeds 66900-66999 only; nothing is read as a result).
Runs the committed runner's own setup, cohort estimates and 'collect' attempts at this worktree's
committed revision, timing each part, and writes each root with the stage's own write_root
(removed afterwards). R17.28. Run from the worktree root:

    MUJOCO_GL=egl uv run --no-sync python scripts/probe_task077_stage_c.py outputs/<new dir>

It ran at 86985a2 from an uncommitted copy (sha256 6915023f...de3) whose code is this file's
before formatting and lint fixes; outputs/task077-cprobe-4/report.json (sha256 3ae3486f...7d28).
"""

import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
spec = importlib.util.spec_from_file_location("_run", ROOT / "scripts/run_lewm_c1m_v2.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
lm, hz, off, rt, np = run.lm, run.hz, run.off, run.rt, run.np


def main():
    out_dir = ROOT / sys.argv[1]
    out_dir.mkdir(parents=True)  # refuses an existing folder
    args = run.build_parser().parse_args(
        [
            "corpus",
            "--debug",
            "--output",
            str(out_dir / "unused"),
            "--evidence",
            "/home/huhn/develop/emai/worktrees/task076-evidence",
        ]
    )
    watch = rt.MemoryWatch(
        12 * 2**30, 0.5, measure="pss", disk_path=ROOT, min_disk_free_bytes=10 * 2**30
    )
    watch.start()
    report = {
        "stages": {},
        "revision": hz.revision(),
        "dirty": bool(hz.tracked_tree_dirty()),
        "load_start": os.getloadavg(),
    }
    manifest = json.loads((ROOT / lm.TASK076_MANIFEST).read_text())
    t = time.monotonic()
    pool, co = run.sim_setup(report, args, manifest, {})
    report["setup_s"] = time.monotonic() - t
    seeds = tuple(range(66900, 67000))
    lm.check_seeds("corpus", seeds, debug=True)
    resets = {s: lm.reset_of(s) for s in seeds}
    try:
        t = time.monotonic()
        est = hz.cohort_estimates(pool, co.p_readout, co.encoder, seeds, resets, 1800.0)
        report["estimates_s"] = time.monotonic() - t
        report["render_disagreements"] = est.pop("_render_disagreements", {})
        co.resets |= resets
        co.est |= est
        folder = out_dir / "corpus"
        folder.mkdir()
        chunks, per_root, sizes, excluded = [], [], [], 0
        for lo in range(0, len(seeds), run.CORPUS_CHUNK):
            part = seeds[lo : lo + run.CORPUS_CHUNK]
            tasks = run.attempt_tasks(
                "collect",
                part,
                co,
                lambda s: dict(zip(("a", "b_m"), lm.corpus_aim(s), strict=True)),
            )
            t = time.monotonic()
            recs = pool.map(tasks, run.MAP_CAP_SECONDS, "collect")
            chunks.append({"roots": len(part), "map_s": time.monotonic() - t})
            t = time.monotonic()
            for r in recs:
                per_root.append(r.get("log", {}).get("seconds") or r.get("seconds"))
                arrays = r.get("corpus") or {"complete": False}
                if r["blocked"] is not None or not arrays.get("complete"):
                    excluded += 1
                    continue
                off.write_root(folder, r["seed"], arrays)
                sizes.append((folder / f"{r['seed']}.npz").stat().st_size)
            chunks[-1]["write_s"] = time.monotonic() - t
        report |= {
            "chunks": chunks,
            "excluded": excluded,
            "per_root_s": [x for x in per_root if x is not None],
            "root_bytes": {"median": float(np.median(sizes)), "max": int(max(sizes))},
        }
    finally:
        report["pool_close"] = pool.close()
        watch.stop() if hasattr(watch, "stop") else None
        report["peak_tree_pss_gib"] = watch.peak / 2**30
        report["load_end"] = os.getloadavg()
        report.pop("_run1", None)
        report.pop("_watch", None)
        (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=str))
        shutil.rmtree(out_dir / "corpus", ignore_errors=True)
    print(
        json.dumps(
            {
                k: report.get(k)
                for k in (
                    "revision",
                    "dirty",
                    "setup_s",
                    "estimates_s",
                    "chunks",
                    "excluded",
                    "root_bytes",
                    "peak_tree_pss_gib",
                    "load_start",
                    "load_end",
                )
            },
            default=str,
        )
    )
    pr = report.get("per_root_s") or []
    if pr:
        print("per_root_s n", len(pr), "median", float(np.median(pr)), "max", max(pr))


if __name__ == "__main__":
    main()
