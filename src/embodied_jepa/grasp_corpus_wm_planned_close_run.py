"""TASK-091's shared runner code: the report, guards and helpers of TASK-090's frozen runner
(``scripts/run_task090.py`` at 53f0b72), copied here so that ``scripts/run_task091.py`` loads no
other script (the 2026-10-02 audit, F16). The copies are the frozen runner's code with ``ROOT``
passed in; ``tests/test_grasp_corpus_planned_close.py`` checks the shuffled deal against the
frozen runner's.

NumPy only at import; torch is imported inside the functions that need it. Nothing here is
registered or imported by ``embodied_jepa``.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from embodied_jepa import grasp_corpus_wm as gc
from embodied_jepa import grasp_corpus_wm_planned_close as pcl
from embodied_jepa import grounded_wm as gw
from embodied_jepa import grounded_wm_data as gd
from embodied_jepa import run_tools as rt
from embodied_jepa import zero_shot_grasp as zg

DISK_FLOOR_GIB = 10.0
DATA_DISK = Path("/mnt/data/emai/projects/open-embodied-jepa")
CORPUS_DIRS = {"play-v1": "data/play-v1", "play-v2g": "data/play-v2g", "play-v1p": "data/play-v1p"}


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc()}] {msg}", flush=True)


def git(root, *args) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def sha(path) -> str:
    return gd.sha256_file(path)


def write_json(path, value) -> None:
    path = Path(path)
    if path.exists():
        raise rt.GuardError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=True) + "\n")


def base_report(root, cohort=None, *, clean=False, gpu=True) -> dict:
    """TASK-090's report and guards (revision, clean tracked tree for non-debug cohorts, local
    import, disk floor, GPU lock) with TASK-091's task, protocol and frozen-file check."""
    root = Path(root)
    frozen = pcl.frozen_files_check(root)
    if not all(frozen.values()):
        bad = [k for k, ok in frozen.items() if not ok]
        raise rt.GuardError(f"G-frozen: these frozen files differ from their pins: {bad}")
    dirty = bool(git(root, "status", "--porcelain", "--untracked-files=no"))
    report = {
        "task": pcl.TASK,
        "protocol": pcl.PROTOCOL,
        "started_utc": utc(),
        "revision": git(root, "rev-parse", "HEAD"),
        "tracked_tree_dirty": dirty,
        "argv": sys.argv,
        "platform": platform.platform(),
        "cohort": cohort,
        "frozen_files_match": True,
        "job_cap_seconds": pcl.JOB_CAP_SECONDS,
    }
    rt.assert_local_import(root, report)
    if (clean or (cohort is not None and not cohort.startswith("debug"))) and dirty:
        raise rt.GuardError("G-clean: non-debug stages run from a clean tracked tree")
    free = min(shutil.disk_usage("/").free, shutil.disk_usage(root / "data").free) / 2**30
    report["disk_free_gib"] = free
    if free < DISK_FLOOR_GIB:
        raise rt.GuardError(f"G-disk: {free:.1f} GiB free < {DISK_FLOOR_GIB}")
    if gpu:
        if os.environ.get(rt.GPU_LOCK_ENV) != "1":
            raise rt.GuardError("G-gpu-lock: run through scripts/gpu_run.sh")
        report["gpu_lock_held"] = True
    report["mujoco_gl"] = os.environ.get("MUJOCO_GL")
    return report


# ----- TASK-090 §12's verify (``run_task090.py verify``), unchanged in substance ----------------
def verify_task090(root, main_checkout, task087_evidence) -> dict:
    """The data-disk links, ``play-v1`` re-hashed, TASK-087's feature report, store files and
    projection, its 18 checkpoints against their reports and evidence copies, the pinned moments
    against G-87100's buffers and TASK-087's implementation hash. Returns the report (``checks``,
    ``all_pass``)."""
    import torch  # noqa: F401  (checkpoint buffers)

    from embodied_jepa import grasp_corpus_wm_runtime as gr
    from embodied_jepa import grounded_wm_model as gm

    root = Path(root)
    report, checks, links = {}, {}, {}
    for name in ("data", "checkpoints", "outputs"):
        for base in (root, Path(main_checkout)):
            p = base / name
            if not p.is_symlink():
                continue
            target = p.resolve()
            links[str(p)] = {"target": str(target), "on_data_disk": DATA_DISK in target.parents}
    report["links"] = links
    dev = subprocess.run(
        ["findmnt", "-n", "-o", "SOURCE", "--target", str(DATA_DISK)],
        capture_output=True,
        text=True,
    ).stdout.strip()
    block = Path(dev).name.rstrip("0123456789")
    rota = Path(f"/sys/block/{block}/queue/rotational")
    report["data_disk"] = {
        "device": dev,
        "rotational": rota.read_text().strip() == "1" if rota.exists() else None,
    }
    checks["links_on_data_disk"] = bool(links) and all(v["on_data_disk"] for v in links.values())
    t0 = time.monotonic()
    report["play_v1"] = gd.verify_corpus(root / CORPUS_DIRS["play-v1"])
    report["play_v1_seconds"] = time.monotonic() - t0
    checks["play_v1"] = True
    feat = gr.V1_FEATURES
    checks["feature_report"] = sha(feat / "report.json") == gr.V1_FEATURES_REPORT_SHA256
    frep = json.loads((feat / "report.json").read_text())
    report["store_files"] = gd.verify_store_files(feat, frep["files_sha256"], ("train", "val"))
    checks["store_files"] = True
    checks["projection_file"] = sha(feat / "projection.npz") == frep["projection"]["file_sha256"]
    gd.load_projection(feat / "projection.npz", gr.PROJECTION_SHA256)
    checks["projection_arrays"] = True
    evidence = Path(task087_evidence)
    ckpts = {}
    for seed in (87100, 87101, 87102):
        for arm in gw.ARMS:
            name = f"{arm}-{seed}"
            folder = gr.V1_CHECKPOINTS / name
            rep = json.loads((folder / "report.json").read_text())
            ev = evidence / f"train_report_{name}.json"
            row = {
                "model_pt": sha(folder / "model.pt"),
                "recorded": rep["checkpoint_sha256"],
                "report_equals_evidence": sha(folder / "report.json") == sha(ev),
            }
            row["ok"] = row["model_pt"] == row["recorded"] and row["report_equals_evidence"]
            ckpts[name] = row
    report["checkpoints"] = ckpts
    checks["checkpoints"] = all(r["ok"] for r in ckpts.values()) and len(ckpts) == 18
    report["moments"] = gr.check_pinned_moments()
    checks["moments_bitwise"] = report["moments"]["bitwise_equal"]
    checks["implementation_hash"] = gm.implementation_sha256() == gr.IMPLEMENTATION_SHA256
    report["checks"] = checks
    report["all_pass"] = all(checks.values())
    return report


# ----- A′'s helpers (``run_task090.py``), unchanged ------------------------------------------
def feature_pins(root) -> dict:
    """The three feature reports' current sha256 (``play-v1`` pinned by TASK-088's runtime)."""
    from embodied_jepa import grasp_corpus_wm_runtime as gr

    pins = {"play-v1": gr.V1_FEATURES_REPORT_SHA256}
    for corpus in gc.CORPORA:
        pins[corpus] = sha(gr.new_features(root, corpus) / "report.json")
    return pins


def encoded(enc_fn, data, rep) -> dict:
    enc = {}
    for r in rep["roots"]:
        s = r["seed"]
        for root in zg.ROOTS:
            enc[f"{s}/{root}"] = enc_fn(data[f"{s}/{root}/frame"])[0]
            enc[f"{s}/goal/{root}"] = enc_fn(data[f"{s}/subgoal{zg.ROOT_GOAL[root]}/frame"])[0]
            for name in zg.BRANCHES[root]:
                enc[f"{s}/{root}/{name}"] = enc_fn(data[f"{s}/{root}/{name}/frames"])
    return enc


def ceiling(pm_by_family, enc, data, rep) -> dict:
    """{family: {h: {comparison: [win per root]}}} with the simulator as predictor (TASK-089)."""
    from embodied_jepa import zero_shot_grasp_runtime as zgr

    out = {}
    for family, pm in pm_by_family.items():
        out[family] = {
            str(h): {zg.comparison_name(c): [] for c in zg.COMPARISONS} for h in zg.HORIZONS
        }
        for r in rep["roots"]:
            s = r["seed"]
            for comp in zg.COMPARISONS:
                root, t, a = comp
                gl, gs = enc[f"{s}/goal/{root}"], data[f"{s}/subgoal{zg.ROOT_GOAL[root]}/state28"]
                costs = {
                    b: zgr.ceiling_costs(
                        pm, enc[f"{s}/{root}/{b}"], data[f"{s}/{root}/{b}/states"], gl, gs
                    )
                    for b in (t, a)
                }
                for h in zg.HORIZONS:
                    w = bool(zg.wins(costs[t][h - 1], costs[a][h - 1]))
                    out[family][str(h)][zg.comparison_name(comp)].append(w)
    return out


def pin_model_pms():
    """The ceiling's cost: G-87100's buffers give the state standardisation, and they are the
    pinned ``play-v1`` moments (checked bitwise); the model's weights are not used."""
    from embodied_jepa import grasp_corpus_wm_runtime as gr

    model, info = gr.load_v1_model(87100, "cuda")
    return {
        "latent": gr.planner_model(model, w_pose=0.0),
        "latent+pose": gr.planner_model(model, w_pose=1.0),
    }, info


def shuffled_deal(seed: int, comparison: int, true_actions, alt_actions):
    """TASK-089's A1 shuffled control with A′'s salt 9002."""
    pool = np.concatenate([np.asarray(true_actions), np.asarray(alt_actions)])
    rng = np.random.default_rng([gc.SALTS["shuffle"], int(seed), int(comparison)])
    order = rng.permutation(len(pool))
    return pool[order[: zg.WINDOW]], pool[order[zg.WINDOW :]]
