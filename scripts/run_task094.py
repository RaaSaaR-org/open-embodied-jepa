"""TASK-094: the grasp-closure locus diagnostic's jobs (protocol
docs/experiments/grasp_closure_locus_v1.md). Measurement only: no controller, no training, no
world-model closed loop. Subcommands; every GPU use (MuJoCo rendering, encoding, roll-outs, CEM)
is one job under ``scripts/gpu_run.sh --wait --min-free-gib 8 --board`` and stops at the 1 800 s
cap:

* ``verify --output DIR``: §14's hash checks -- TASK-087's three G checkpoints against their pins,
  reports and evidence copies, the feature report and store files, the projection, the moments,
  TASK-087's implementation hash, λ, the rate limit, CEM and every frozen file (CPU).
* ``seeds --output FILE``: §9's uniqueness check at this revision, over every remote branch and
  the evidence (CPU).
* ``power --output FILE``: §8.4's K table with 20 000 simulations, K-NULL's joint condition
  included (CPU).
* ``goals --cohort``: TASK-088's grasp goals (``zero_shot.make_grasp_goal``) for a K cohort.
* ``k-run --cohort --arm``: one K arm (§4.1) on a K cohort's goals, with the per-episode plumbing
  checks; ``--determinism`` re-runs K-1.0 or K-s* on the first four resets.
* ``roots --cohort``: M's roots and branches (§5.1-§5.2) for a root cohort, re-run from the reset
  and checked bitwise.
* ``rank --cohort``: M's costs (§5.3) -- G-87100-87102's roll-outs and the ceiling on real frames,
  with the term decomposition.
* ``search --cohort``: S's CEM calls (§6), one per root type, root and seed, with ℓ (the median
  executed right grasp over window steps 3-7) and the search miss's comparator (G-i's cost of the
  plan and of the plan with the squeeze branch's right-grasp requests; R27.183 (3)-(4)), and the
  determinism re-run on the first two roots.
* ``substitute --cohort dev-roots``: Stage 0′ (§17; R27.183 (5)): the comparator's costs for the
  stored development plans of Stage 0's search on 940000-940031 (no CEM call; reported only).

TASK-087's to TASK-093's frozen code is imported, never edited, and checked against its pins
before anything runs (``grasp_closure_locus.FROZEN_FILES_SHA256``). Every job must start in this
checkout's root with ``third_party/le-wm`` present (R27.107) and loads and checks its goals or roots
and its models before any ledger claim. Non-debug cohorts need a clean tracked tree; every report
records the revision; every file is refused if it exists, and a non-debug job is refused if the
ledger (``outputs/task094-ledger``) shows it was started before (a cohort run twice is VOID).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import types
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import grasp_closure_locus as gl  # noqa: E402
from embodied_jepa import grounded_wm_data as gd  # noqa: E402
from embodied_jepa import run_tools as rt  # noqa: E402
from embodied_jepa import zero_shot as zs  # noqa: E402

LEDGER = ROOT / "outputs" / "task094-ledger"
LEWM_SOURCE = ROOT / "third_party" / "le-wm"
EXIT_CAP = 4
EXIT_VOID = 3
DISK_FLOOR_GIB = 10.0

sha = gd.sha256_file


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc()}] {msg}", flush=True)


def git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def write_json(path, value) -> None:
    path = Path(path)
    if path.exists():
        raise rt.GuardError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=True) + "\n")


class CapHit(Exception):
    pass


def _alarm(signum, frame):
    raise CapHit(f"the per-job cap of {gl.JOB_CAP_SECONDS} s was reached")


def tree_guard() -> dict:
    """R27.107's lesson: the job runs in this checkout's root, where ``third_party/le-wm`` is."""
    cwd = Path.cwd().resolve()
    if cwd != ROOT.resolve():
        raise rt.GuardError(f"G-cwd: run from {ROOT}, not {cwd}")
    if not LEWM_SOURCE.is_dir():
        raise rt.GuardError(f"G-lewm: {LEWM_SOURCE} is missing")
    return {"cwd": str(cwd), "lewm_source": str(LEWM_SOURCE.resolve())}


def debug_cohort(cohort) -> bool:
    return cohort is not None and str(cohort).startswith("debug")


def base_report(cohort=None, *, gpu=True) -> dict:
    """TASK-093's report and guards with TASK-094's task, protocol, frozen-file check and the
    working-directory guard."""
    frozen = gl.frozen_files_check(ROOT)
    if not all(frozen.values()):
        bad = [k for k, ok in frozen.items() if not ok]
        raise rt.GuardError(f"G-frozen: these frozen files differ from their pins: {bad}")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    report = {
        "task": gl.TASK,
        "protocol": gl.PROTOCOL,
        "started_utc": utc(),
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": dirty,
        "argv": sys.argv,
        "platform": platform.platform(),
        "cohort": cohort,
        "frozen_files_match": True,
        "modules_sha256": gl.own_modules_sha256(ROOT),
        "job_cap_seconds": gl.JOB_CAP_SECONDS,
        "tree": tree_guard(),
    }
    rt.assert_local_import(ROOT, report)
    if cohort is not None and not debug_cohort(cohort) and dirty:
        raise rt.GuardError("G-clean: non-debug cohorts run from a clean tracked tree")
    free = min(shutil.disk_usage("/").free, shutil.disk_usage(ROOT / "outputs").free) / 2**30
    report["disk_free_gib"] = free
    if free < DISK_FLOOR_GIB:
        raise rt.GuardError(f"G-disk: {free:.1f} GiB free < {DISK_FLOOR_GIB}")
    if gpu:
        if os.environ.get(rt.GPU_LOCK_ENV) != "1":
            raise rt.GuardError("G-gpu-lock: run through scripts/gpu_run.sh")
        report["gpu_lock_held"] = True
    report["mujoco_gl"] = os.environ.get("MUJOCO_GL")
    return report


def ledger_claim(key: str) -> None:
    """A non-debug cohort's goals, arm run, roots, ranking or search may start only once."""
    LEDGER.mkdir(parents=True, exist_ok=True)
    path = LEDGER / key
    try:
        with open(path, "x") as f:
            f.write(json.dumps({"utc": utc(), "revision": git("rev-parse", "HEAD")}) + "\n")
    except FileExistsError as error:
        raise rt.GuardError(f"G-twice: {key} was started before ({path})") from error


def _load_model(seed: int) -> dict:
    from embodied_jepa import grasp_closure_locus_runtime as lr

    _, info = lr.load_model(int(seed), "cpu")
    return info


def _load_encoder() -> dict:
    """TASK-087's frozen DINOv2 encoder and its pinned projection (on the CPU)."""
    from embodied_jepa import grasp_corpus_wm_runtime as gr

    enc = gr.encoder("cpu")
    digest = getattr(enc.featuriser, "digest", None)
    if digest != gr.ENCODER_DIGEST:
        raise rt.GuardError("G-encoder: not TASK-087's DINOv2 encoder")
    return {"featuriser_digest": digest}


def preflight(seeds) -> dict:
    """Load every model a job needs -- the G checkpoints and the DINOv2 encoder with its
    projection -- on the CPU before any ledger claim (R27.109)."""
    out = {f"G-v1-{s}": _load_model(s)["checkpoint_sha256"] for s in seeds}
    out["encoder"] = _load_encoder()
    return out


class Job:
    """One capped job: the report it writes on a cap hit (never over an existing file)."""

    def __init__(self, out, report, cap):
        self.out, self.report, self.cap = Path(out), report, cap

    def __enter__(self):
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(self.cap))
        self.t0 = time.time()
        return self

    def __exit__(self, kind, error, tb):
        signal.alarm(0)
        if kind is CapHit:
            self.report |= {
                "completed": False,
                "cap_hit": True,
                "seconds": time.time() - self.t0,
                "finished_utc": utc(),
                "error": str(error),
            }
            if not self.out.exists():
                write_json(self.out, self.report)
            log(f"CAP-HIT after {self.report['seconds']:.0f} s: {self.out} did not complete")
            raise SystemExit(EXIT_CAP)
        return False


def _cap(args) -> int:
    cap = getattr(args, "cap_seconds", None) or gl.JOB_CAP_SECONDS
    if cap != gl.JOB_CAP_SECONDS and not debug_cohort(getattr(args, "cohort", "debug")):
        raise rt.GuardError("the per-job cap is 1 800 s outside debug cohorts")
    return int(cap)


def _limit(args, items):
    if getattr(args, "limit", None):
        if not debug_cohort(args.cohort):
            raise rt.GuardError("--limit is for debug cohorts only")
        return items[: args.limit]
    return items


def _manifest() -> dict:
    return json.loads((ROOT / "configs" / "g1_sim_action.json").read_text())


def _lambda_check():
    from embodied_jepa import grasp_corpus_wm_runtime as gr

    if not (zs.LAMBDA == gr.LAMBDA == gl.LAMBDA == 2.0552) or gr.SETTING != gl.SETTING:
        raise rt.GuardError("G-lambda: λ or the setting is not TASK-089's")


# ----- §14: verify -----------------------------------------------------------------------------
def cmd_verify(args) -> int:
    from embodied_jepa import grasp_corpus_wm_runtime as gr
    from embodied_jepa import grasp_rate_aware as ra
    from embodied_jepa import grounded_wm_model as gm

    out = Path(args.output) / "verify-task094.json"
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    report = base_report(gpu=False)
    checks = {}
    feat = gr.V1_FEATURES
    frep = json.loads((feat / "report.json").read_text())
    checks["feature_report"] = sha(feat / "report.json") == gl.FEATURE_REPORT_SHA256
    t0 = time.monotonic()
    report["store_files"] = gd.verify_store_files(feat, frep["files_sha256"], ("train", "val"))
    report["store_seconds"] = time.monotonic() - t0
    checks["store_files"] = True
    checks["projection_file"] = sha(feat / "projection.npz") == frep["projection"]["file_sha256"]
    gd.load_projection(feat / "projection.npz", gr.PROJECTION_SHA256)
    checks["projection_arrays"] = True
    ckpts = {}
    ev = Path(args.task087_evidence)
    for seed in gl.MODEL_SEEDS:
        folder = gr.V1_CHECKPOINTS / f"G-{seed}"
        rep = json.loads((folder / "report.json").read_text())
        row = {
            "model_pt": sha(folder / "model.pt"),
            "recorded": rep["checkpoint_sha256"],
            "pinned": gl.CHECKPOINT_SHA256[f"G-v1-{seed}"],
            "report_equals_evidence": sha(folder / "report.json")
            == sha(ev / f"train_report_G-{seed}.json"),
            "kind": rep.get("kind"),
            "outcome": rep.get("outcome"),
            "last_two_triggered": rep["training"]["last_two_triggered"],
        }
        row["ok"] = (
            row["model_pt"] == row["recorded"] == row["pinned"]
            and row["report_equals_evidence"]
            and row["kind"] == "job"
            and row["outcome"] == "T-JOB-DONE"
        )
        ckpts[f"G-v1-{seed}"] = row
    report["task087_checkpoints"] = ckpts
    checks["task087_checkpoints"] = all(r["ok"] for r in ckpts.values())
    report["moments"] = gr.check_pinned_moments()
    checks["moments_pin"] = report["moments"]["pinned_sha256"] == gl.MOMENTS_SHA256
    checks["implementation_hash"] = gm.implementation_sha256() == gl.IMPLEMENTATION_SHA256
    checks["lambda"] = zs.LAMBDA == gl.LAMBDA == gr.LAMBDA == 2.0552
    checks["setting"] = gr.SETTING == gl.SETTING == "step-mean"
    checks["rate_limit"] = abs(ra.rate_limit_from_manifest(_manifest()) - gl.RATE_LIMIT) < 1e-12
    checks["cem"] = zs.CEM == {
        "horizon": 8, "samples": 300, "iterations": 10, "elites": 30, "minimum_std": 0.05,
    }  # fmt: skip
    checks["levels_and_offsets"] = (
        [gl.K_ARMS[a] for a in gl.K_ARMS]
        == [(1.0, None), (0.9, None), (0.8, None), (0.7, None), ("s*", None), (0.6, None),
            (-1.0, None), (1.0, "x"), ("s*", "x"), (1.0, "y"), ("s*", "y")]
        and gl.OFFSETS == {"x": (0.012, 0.0, 0.0), "y": (0.0, 0.012, 0.0)}
        and gl.BRANCHES_ALL == ("1.0", "0.9", "0.8", "0.7", "s*", "control")
        and gl.RS_CLOSE == 8
    )  # fmt: skip
    report["frozen_files"] = gl.frozen_files_check(ROOT)
    checks["frozen_files"] = all(report["frozen_files"].values())
    report["checks"] = checks
    report["all_pass"] = all(checks.values())
    report["finished_utc"] = utc()
    write_json(out, report)
    log(json.dumps(checks))
    return 0 if report["all_pass"] else 1


# ----- §9: the seed uniqueness check -----------------------------------------------------------
SEED_PATTERN = re.compile(
    r"(?<![0-9A-Za-z.])(94[0-26][0-9]{3}|941[0-9]{2}|945[0-9]{2}|901[45])(?![0-9A-Za-z])"
)
CHECK_RANGES = {
    "dev-k": range(94100, 94132),
    "debug": range(94500, 94600),
    "dev-roots": range(940000, 940032),
    "gated-k": range(941000, 941128),
    "gated-roots": range(946000, 946128),
    "retired-roots": range(942000, 942064),
    "salt 9014": range(9014, 9015),
    "salt 9015": range(9015, 9016),
}
OWN_MARKERS = ("task094", "TASK-094", "grasp_closure_locus", "grasp-closure-locus")


def _in_ranges(number: int) -> list[str]:
    return [k for k, r in CHECK_RANGES.items() if number in r]


def cmd_seeds(args) -> int:
    report = base_report(gpu=False)
    out = Path(args.output)
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    subprocess.run(["git", "fetch", "--quiet", "--all"], cwd=ROOT, check=False)
    branches = git("for-each-ref", "--format=%(refname:short)", "refs/remotes").split()
    branches = [b for b in branches if not b.endswith("/HEAD")]
    locals_ = git("for-each-ref", "--format=%(refname:short)", "refs/heads").split()
    paths = ["src", "scripts", "docs", ".mc", "tests", "benchmarks", "configs"]
    hits = {}
    pattern = r"94[0-26][0-9]{3}|941[0-9]{2}|945[0-9]{2}|901[45]"
    for branch in branches + locals_ + ["HEAD"]:
        res = subprocess.run(
            ["git", "grep", "-n", "-I", "-E", pattern, branch, "--"] + paths,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        for line in res.stdout.splitlines():
            parts = line.split(":", 3)
            if len(parts) < 4:
                continue
            _, file, _, text = parts
            for m in SEED_PATTERN.finditer(text):
                if _in_ranges(int(m.group(1))):
                    hits.setdefault(file, {}).setdefault(m.group(1), set()).add(branch)
    classified = {"own": {}, "other": {}}
    for file, nums in hits.items():
        own = any(m in file for m in OWN_MARKERS) or file in (
            "docs/DECISIONS.md",
            "CLAUDE.md",
            "docs/experiments/README.md",
        )
        classified["own" if own else "other"][file] = {n: sorted(b) for n, b in nums.items()}
    ev_hits = []
    exts = {".json", ".txt", ".log", ".md", ".csv", ".yaml", ".py", ".sh"}
    for path in sorted(Path(args.evidence).rglob("*")):
        if not path.is_file() or path.suffix not in exts or "task094" in str(path):
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for m in SEED_PATTERN.finditer(text):
            if _in_ranges(int(m.group(1))):
                ctx = text[max(0, m.start() - 40) : m.end() + 40].replace("\n", " ")
                ev_hits.append({"file": str(path), "token": m.group(1), "context": ctx})
    seed_fields = [
        h for h in ev_hits if re.search(r'"(seed|reset|root|seeds|salt)"\s*:', h["context"])
    ]
    report |= {
        "branches": len(branches),
        "local_branches": len(locals_),
        "paths": paths,
        "ranges": {k: [r.start, r.stop - 1] for k, r in CHECK_RANGES.items()},
        "hits_in_task094_files": classified["own"],
        "hits_elsewhere": classified["other"],
        "evidence_dir": str(args.evidence),
        "evidence_hits": ev_hits,
        "evidence_hits_in_seed_fields": seed_fields,
        "finished_utc": utc(),
    }
    write_json(out, report)
    log(
        f"{len(branches)} remote and {len(locals_)} local branches; hits outside TASK-094's files: "
        f"{len(classified['other'])} files; evidence tokens: {len(ev_hits)} "
        f"({len(seed_fields)} in seed fields)"
    )
    return 0


# ----- §8.4: power (CPU) -----------------------------------------------------------------------
def _power_cell(cell):
    delta, d0, delta06, model, sims = cell
    return gl.k_power_cell(delta, d0, delta06, model=model, sims=sims)


def cmd_power(args) -> int:
    report = base_report(gpu=False)
    out = Path(args.output)
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    t0 = time.monotonic()
    cells = [(*c, args.sims) for c in gl.power_grid()]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(_power_cell, cells)
    report |= {
        "rows": rows,
        "sims": args.sims,
        "resamples": gl.BOOTSTRAP_RESAMPLES,
        "p1": gl.P1,
        "rule": gl.power_rule(rows),
        "seconds": time.monotonic() - t0,
        "finished_utc": utc(),
    }
    write_json(out, report)
    for r in rows:
        log(f"Δ {r['delta']} d0 {r['d0']} Δ0.6 {r['delta06']} {r['model']}: {r['rates']}")
    log(json.dumps(report["rule"]))
    return 0


# ----- K: goals and arm runs (§4) --------------------------------------------------------------
def _goal_worker(seeds):
    from embodied_jepa import play_corpus as pc

    robot = pc.make_play_robot()
    out = [zs.make_grasp_goal(robot, s) for s in seeds]
    robot.close()
    return out


def cmd_goals(args) -> int:
    if args.cohort not in gl.K_COHORTS:
        raise rt.GuardError(f"{args.cohort} is not a K cohort")
    report = base_report(args.cohort)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    jpath, npath = out / f"goals-{args.cohort}.json", out / f"goals-{args.cohort}.npz"
    for p in (jpath, npath):
        if p.exists():
            raise rt.GuardError(f"refusing to overwrite {p}")
    seeds = _limit(args, list(gl.COHORTS[args.cohort]))
    if not debug_cohort(args.cohort):
        ledger_claim(f"goals-{args.cohort}")
    with Job(jpath, report, _cap(args)) as job:
        chunks = [seeds[i :: args.workers] for i in range(args.workers)]
        with mp.get_context("spawn").Pool(args.workers) as pool:
            parts = pool.map(_goal_worker, [c for c in chunks if c])
        goals = sorted((g for p in parts for g in p), key=lambda g: g["seed"])
        arrays, meta = {}, []
        for g in goals:
            row = {k: v for k, v in g.items() if k != "subgoals"}
            row["subgoals"] = []
            for k, s in enumerate(g["subgoals"]):
                arrays[f"{g['seed']}_{k}_frame"] = s["frame"]
                row["subgoals"].append(
                    {
                        "state28": s["state28"].tolist(),
                        "palm": s["palm"].tolist(),
                        "rotation": s["rotation"].tolist(),
                    }
                )
            row["digest"] = zs.goal_digest(g)
            meta.append(row)
        np.savez_compressed(npath, **arrays)
        report |= {
            "seconds": time.time() - job.t0,
            "goals": meta,
            "frames_sha256": sha(npath),
            "rejections": sum(len(g["rejected"]) for g in goals),
            "completed": True,
            "cap_hit": False,
            "finished_utc": utc(),
        }
        write_json(jpath, report)
    log(f"{len(goals)} goals, {report['rejections']} rejected draws, {report['seconds']:.0f} s")
    return 0


def load_goals(folder, cohort, revision=None):
    folder = Path(folder)
    rep = json.loads((folder / f"goals-{cohort}.json").read_text())
    if not rep.get("completed"):
        raise rt.GuardError("G-goals: the goals job did not complete")
    npath = folder / f"goals-{cohort}.npz"
    if sha(npath) != rep["frames_sha256"]:
        raise rt.GuardError("G-goals: frames differ from the goal report")
    goals = []
    with np.load(npath) as frames:
        for row in rep["goals"]:
            gl.check_seed(cohort, row["seed"])
            g = dict(row)
            g["subgoals"] = [
                {
                    "frame": frames[f"{row['seed']}_{k}_frame"],
                    "state28": np.asarray(s["state28"], np.float32),
                    "palm": np.asarray(s["palm"], np.float32),
                    "rotation": np.asarray(s["rotation"], np.float32),
                }
                for k, s in enumerate(row["subgoals"])
            ]
            if zs.goal_digest(g) != row["digest"]:
                raise rt.GuardError(f"G-goals: seed {row['seed']} differs from its digest")
            goals.append(g)
    if not debug_cohort(cohort):
        if [g["seed"] for g in goals] != list(gl.COHORTS[cohort]):
            raise rt.GuardError("G-goals: the cohort's seeds are not exactly its goals")
        if revision is not None and rep["revision"] != revision:
            raise rt.GuardError("G-goals: the goals were made at another revision")
    return goals, sha(folder / f"goals-{cohort}.json")


def _k_worker(job):
    arm, goals = job
    from embodied_jepa import play_corpus as pc
    from embodied_jepa import zero_shot_runtime as zr

    robot = pc.make_play_robot()
    out = []
    for goal in goals:
        s_star = gl.s_star_of(robot, goal)
        run_goal = goal
        displaced = None
        if arm in gl.K_ARMS and gl.K_ARMS[arm][1] is not None:
            axis = gl.K_ARMS[arm][1]
            run_goal = gl.displaced_goal(goal, axis)
            displaced = gl.displacement_ok(goal, run_goal, axis)
        if arm == "ik":
            controller = zs.IKController()
        else:
            controller = gl.HeldIKController(arm)
        result = zr.run_episode(robot, run_goal, controller)
        episode = {
            "seed": goal["seed"],
            "arm": arm,
            "goal_digest": goal["digest"],
            "s_star": s_star,
            "success": bool(result["success"]),
            "success_step": result["success_step"],
            "steps": result["steps"],
            "stop_reason": result["stop_reason"],
            "max_rise": result["max_rise"],
            "any_grasp_contact": result["any_grasp_contact"],
            "switches": result["switches"],
            "final_subgoal": result["final_subgoal"],
            "rise": result["rise"],
            "contact": result["contact"],
            "distances": result["distances"],
            "applied": result["applied"],
            "wall_seconds": result["wall_seconds"],
        }
        if arm != "ik":
            level = controller.level
            episode |= {
                "level": level,
                "displacement_ok": displaced,
                "requested": [e["requested"] for e in controller.log],
                "subgoal_index": [e["index"] for e in controller.log],
                "measured_synergy": [e["measured_synergy"] for e in controller.log],
                "plumbing": gl.k_plumbing(controller.log, result["applied"], level),
                "measures": gl.k_episode_measures(result, controller.log),
            }
        out.append(episode)
        log(f"{arm} {goal['seed']} success={episode['success']} steps={episode['steps']} "
            f"{episode['stop_reason']}")  # fmt: skip
    robot.close()
    return out


def k_run_name(arm, *, determinism=False) -> str:
    return gl.arm_slug(arm) + ("-determinism" if determinism else "")


def cmd_k_run(args) -> int:
    if args.cohort not in gl.K_COHORTS:
        raise rt.GuardError(f"{args.cohort} is not a K cohort")
    debug = debug_cohort(args.cohort)
    arm = gl.arm_from_slug(args.arm)
    if arm not in gl.K_ARMS and not (arm in gl.STAGE0_EXTRA_ARMS and args.cohort != "gated-k"):
        raise rt.GuardError(f"{arm} is not a K arm of {args.cohort}")
    report = base_report(args.cohort)
    goals, goals_sha = load_goals(args.goals, args.cohort, None if debug else report["revision"])
    goals = _limit(args, goals)
    if args.determinism:
        if arm not in gl.DETERMINISM_K[0] or args.cohort == "dev-k":
            raise rt.GuardError("the determinism re-run is K-1.0 or K-s* on the gated K cohort")
        goals = goals[: gl.DETERMINISM_K[1]]
    name = k_run_name(arm, determinism=args.determinism)
    out = Path(args.output) / args.cohort / f"{name}{args.suffix or ''}.json"
    if args.suffix and not debug:
        raise rt.GuardError("--suffix is for debug cohorts only")
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    report |= {
        "arm": arm,
        "goals_sha256": goals_sha,
        "workers": args.workers,
        "determinism_rerun": bool(args.determinism),
        "seeds": [g["seed"] for g in goals],
        "level_spec": gl.K_ARMS[arm][0] if arm in gl.K_ARMS else "ik (frozen)",
        "offset": gl.OFFSETS.get(gl.K_ARMS[arm][1]) if arm in gl.K_ARMS else None,
    }
    if not debug:
        ledger_claim(f"k-{args.cohort}-{name}")
    with Job(out, report, _cap(args)) as job:
        workers = min(args.workers, len(goals))
        jobs = [(arm, goals[i::workers]) for i in range(workers)]
        with mp.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(_k_worker, [j for j in jobs if j[1]])
        episodes = sorted((e for p in parts for e in p), key=lambda e: e["seed"])
        plumbing = [e["plumbing"]["ok"] for e in episodes if "plumbing" in e]
        displaced = [e["displacement_ok"] for e in episodes if e.get("displacement_ok") is not None]
        report |= {
            "episodes": episodes,
            "successes": int(sum(e["success"] for e in episodes)),
            "n": len(episodes),
            "plumbing_ok": bool(all(plumbing)),
            "plumbing_failures": [
                e["seed"] for e in episodes if "plumbing" in e and not e["plumbing"]["ok"]
            ],  # fmt: skip
            "displacement_ok": bool(all(displaced)) if displaced else None,
            "seconds": time.time() - job.t0,
            "completed": True,
            "cap_hit": False,
            "finished_utc": utc(),
        }
        write_json(out, report)
    log(f"{name} on {args.cohort}: {report['successes']}/{report['n']} in "
        f"{report['seconds']:.0f} s; plumbing {report['plumbing_ok']}")  # fmt: skip
    if not report["plumbing_ok"] or report["displacement_ok"] is False:
        log("K-VOID check: a plumbing or displacement check failed")
        return EXIT_VOID
    return 0


# ----- M: roots and branches (§5.1-§5.2) -------------------------------------------------------
def _roots_worker(job):
    seeds, branches = job
    from embodied_jepa import play_corpus as pc

    robot = pc.make_play_robot()
    out = []
    for s in seeds:
        out.append(gl.build_locus_root(robot, s, branches))
        log(f"root {s}: {len(out[-1]['rejected'])} rejected, problems {out[-1]['problems']}")
    robot.close()
    return out


def branch_key(seed, root, name) -> str:
    return f"{seed}/{root}/{gl.arm_slug(name)}"


def cmd_roots(args) -> int:
    if args.cohort not in gl.ROOT_COHORTS:
        raise rt.GuardError(f"{args.cohort} is not a root cohort")
    report = base_report(args.cohort)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    jpath, npath = out / f"roots-{args.cohort}.json", out / f"roots-{args.cohort}.npz"
    for p in (jpath, npath):
        if p.exists():
            raise rt.GuardError(f"refusing to overwrite {p}")
    seeds = _limit(args, list(gl.COHORTS[args.cohort]))
    for s in seeds:  # refuses the retired root seeds 942000-942063 (R27.183 (1))
        gl.check_seed(args.cohort, s)
    branches = gl.BRANCHES_BY_COHORT[args.cohort]
    if not debug_cohort(args.cohort):
        ledger_claim(f"roots-{args.cohort}")
    with Job(jpath, report, _cap(args)) as job:
        workers = min(args.workers, len(seeds))
        chunks = [(seeds[i::workers], branches) for i in range(workers)]
        with mp.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(_roots_worker, [c for c in chunks if c[0]])
        roots = sorted((r for p in parts for r in p), key=lambda r: r["seed"])
        robot = types.SimpleNamespace(manifest=_manifest())  # synergy_grasp reads the manifest
        arrays, meta, problems = {}, [], []
        for r in roots:
            s = r["seed"]
            keys = ("seed", "attempt", "layout", "target", "yaw", "z0", "s_star", "rejected")
            row = {k: r[k] for k in keys}
            problems += r["problems"]
            for k, g in enumerate(r["subgoals"]):
                arrays[f"{s}/subgoal{k}/frame"] = g["frame"]
                arrays[f"{s}/subgoal{k}/state28"] = g["state28"]
            row["roots"] = {}
            for root, rec in r["roots"].items():
                arrays[f"{s}/{root}/frame"] = rec["frame"]
                arrays[f"{s}/{root}/state28"] = rec["state28"]
                arrays[f"{s}/{root}/t0"] = rec["t0"]
                row["roots"][root] = {
                    "request": rec["request"],
                    "t0_synergy": float(zs.synergy_grasp(robot, rec["t0"])),
                }
            row["branches"] = {}
            for key, b in r["branches"].items():
                root, name = key.split("/")
                bk = branch_key(s, root, name)
                for f in ("applied", "frames", "states"):
                    arrays[f"{bk}/{f}"] = b[f]
                arrays[f"{bk}/rise"] = np.asarray(b["rise"], np.float64)
                arrays[f"{bk}/contact"] = np.asarray(b["contact"], bool)
                sq = r["branches"][f"{root}/{gl.SQUEEZE}"]["applied"]
                row["branches"][key] = {
                    "level": b["level"],
                    "lift": bool(b["lift"]),
                    "max_rise": float(max(b["rise"])),
                    "any_contact": bool(any(b["contact"])),
                    "commands": len(b["rise"]),
                    "window_differs": gl.executed_differences(sq, b["applied"]),
                    "squeeze_checks": b["squeeze_checks"],
                }
            meta.append(row)
        np.savez_compressed(npath, **arrays)
        report |= {
            "branches": list(branches),
            "seconds": time.time() - job.t0,
            "roots": meta,
            "arrays_sha256": sha(npath),
            "rejections": sum(len(r["rejected"]) for r in roots),
            "problems": problems,
            "completed": True,
            "cap_hit": False,
            "finished_utc": utc(),
        }
        write_json(jpath, report)
    log(f"{len(roots)} roots, {report['rejections']} rejected draws, {len(problems)} problems, "
        f"{report['seconds']:.0f} s")  # fmt: skip
    return EXIT_VOID if problems else 0


def load_roots(folder, cohort, revision):
    """A roots job's report and arrays, checked against each other and the cohort; ``revision``
    None (Stage 0′'s ``substitute`` on Stage 0's stored roots) skips the revision check only."""
    folder = Path(folder)
    rep = json.loads((folder / f"roots-{cohort}.json").read_text())
    if not rep.get("completed"):
        raise rt.GuardError("G-roots: the roots job did not complete")
    npath = folder / f"roots-{cohort}.npz"
    if sha(npath) != rep["arrays_sha256"]:
        raise rt.GuardError("G-roots: arrays differ from the roots report")
    seeds = [r["seed"] for r in rep["roots"]]
    for s in seeds:
        gl.check_seed(cohort, s)
    if not debug_cohort(cohort):
        if seeds != list(gl.COHORTS[cohort]):
            raise rt.GuardError("G-roots: the cohort's seeds are not exactly its roots")
        if revision is not None and rep["revision"] != revision:
            raise rt.GuardError("G-roots: the roots were made at another revision")
    return rep, np.load(npath), sha(folder / f"roots-{cohort}.json")


def _encode_roots(enc_fn, rep, data) -> dict:
    """Every frame M and S score, encoded once: roots, goals and branch windows."""
    out = {}
    for r in rep["roots"]:
        s = r["seed"]
        for root in gl.ROOT_TYPES:
            out[f"{s}/{root}"] = enc_fn(data[f"{s}/{root}/frame"])[0]
            k = gl.ROOT_GOAL[root]
            out[f"{s}/goal/{root}"] = enc_fn(data[f"{s}/subgoal{k}/frame"])[0]
    return out


def _round(x, nd=6):
    return np.round(np.asarray(x, np.float64), nd).tolist()


# ----- M: the costs (§5.3) ---------------------------------------------------------------------
def cmd_rank(args) -> int:
    if args.cohort not in gl.ROOT_COHORTS:
        raise rt.GuardError(f"{args.cohort} is not a root cohort")
    debug = debug_cohort(args.cohort)
    report = base_report(args.cohort)
    out = Path(args.output) / f"rank-{args.cohort}.json"
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    _lambda_check()
    rep, data, roots_sha = load_roots(args.roots, args.cohort, report["revision"])
    if args.limit:
        if not debug:
            raise rt.GuardError("--limit is for debug cohorts only")
        rep = dict(rep, roots=rep["roots"][: args.limit])
    branches = rep["branches"]
    report["preflight"] = preflight(gl.MODEL_SEEDS)
    if not debug:  # the claim comes last: after every guard, the roots and the model loads
        ledger_claim(f"rank-{args.cohort}")
    with Job(out, report, _cap(args)) as job:
        from embodied_jepa import devices
        from embodied_jepa import grasp_closure_locus_runtime as lr

        devices.configure_determinism("cuda", strict=False)
        enc_fn = lr.encoder("cuda")
        enc = _encode_roots(enc_fn, rep, data)
        win = {}
        for r in rep["roots"]:
            s = r["seed"]
            for root in gl.ROOT_TYPES:
                frames = np.concatenate(
                    [data[f"{branch_key(s, root, b)}/frames"] for b in branches]
                )
                win[f"{s}/{root}"] = enc_fn(frames).reshape(len(branches), gl.WINDOW,
                                                            *enc[f"{s}/{root}"].shape)  # fmt: skip
        results, ceilings = {}, {}
        for seed in gl.MODEL_SEEDS:
            model, info = lr.load_model(seed, "cuda")
            pm = lr.planner_model(model)
            res = {"model": info, "roots": {}}
            ceil = {}
            for r in rep["roots"]:
                s = r["seed"]
                for root in gl.ROOT_TYPES:
                    k = gl.ROOT_GOAL[root]
                    seqs = np.stack(
                        [data[f"{branch_key(s, root, b)}/applied"][: gl.WINDOW] for b in branches]
                    )
                    states = np.stack([data[f"{branch_key(s, root, b)}/states"] for b in branches])
                    goal = (enc[f"{s}/goal/{root}"], data[f"{s}/subgoal{k}/state28"])
                    g_terms = lr.rollout_terms(
                        pm, enc[f"{s}/{root}"], data[f"{s}/{root}/state28"], *goal, seqs
                    )
                    c_terms = lr.ceiling_terms(pm, win[f"{s}/{root}"], states, *goal)
                    entry = {}
                    for which, terms in (("g", g_terms), ("ceiling", c_terms)):
                        mean = lr.step_mean(terms)
                        entry[which] = {
                            b: {
                                **{t: float(mean[t][i]) for t in lr.TERMS},
                                "hand_joints": _round(mean["hand_joints"][i]),
                                "total_at_h8": float(terms["total"][i, -1]),
                            }
                            for i, b in enumerate(branches)
                        }
                    res["roots"][f"{s}/{root}"] = entry["g"]
                    ceil[f"{s}/{root}"] = entry["ceiling"]
            results[f"G-v1-{seed}"] = res
            ceilings[f"G-v1-{seed}"] = ceil
            log(f"G-v1-{seed}: {len(res['roots'])} root costs")
            del model, pm
        first = ceilings[f"G-v1-{gl.MODEL_SEEDS[0]}"]
        same = all(c == first for c in ceilings.values())
        problems = [] if same else ["the ceiling differs between the seeds' standardisations"]
        report |= {
            "roots_sha256": roots_sha,
            "n": len(rep["roots"]),
            "branches": branches,
            "results": results,
            "ceiling": first,
            "ceiling_identical_across_seeds": same,
            "problems": problems,
            "seconds": time.time() - job.t0,
            "completed": True,
            "cap_hit": False,
            "finished_utc": utc(),
        }
        write_json(out, report)
    log(f"rank on {args.cohort}: {report['n']} roots, {report['seconds']:.0f} s")
    return EXIT_VOID if problems else 0


def _squeeze_request_deviation(geometry, data, s, root) -> float:
    """The comparator's plumbing check: the squeeze branch's right-grasp requests that S
    substitutes (``gl.squeeze_requests``), executed through the cumulative clip from the root's
    T₀, must give the squeeze branch's recorded applied right grasp over the window."""
    plan = np.zeros((gl.WINDOW, 14), np.float32)
    plan[:, gl.GRASP_DIM] = gl.squeeze_requests(root)
    path = gl.executed_level(geometry, plan, data[f"{s}/{root}/t0"])
    applied = np.asarray(data[f"{branch_key(s, root, gl.SQUEEZE)}/applied"], np.float64)
    return float(np.abs(path - applied[: gl.WINDOW, gl.GRASP_DIM]).max())


# ----- S: the search (§6) ----------------------------------------------------------------------
def cmd_search(args) -> int:
    if args.cohort not in gl.ROOT_COHORTS:
        raise rt.GuardError(f"{args.cohort} is not a root cohort")
    debug = debug_cohort(args.cohort)
    if args.check_a2 and not debug:
        raise rt.GuardError("--check-a2 is for debug cohorts only")
    report = base_report(args.cohort)
    out = Path(args.output) / f"search-{args.cohort}.json"
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    _lambda_check()
    rep, data, roots_sha = load_roots(args.roots, args.cohort, report["revision"])
    if args.limit:
        if not debug:
            raise rt.GuardError("--limit is for debug cohorts only")
        rep = dict(rep, roots=rep["roots"][: args.limit])
    report["preflight"] = preflight(gl.MODEL_SEEDS)
    if not debug:
        ledger_claim(f"search-{args.cohort}")
    with Job(out, report, _cap(args)) as job:
        from embodied_jepa import devices
        from embodied_jepa import grasp_closure_locus_runtime as lr
        from embodied_jepa import grasp_corpus_wm_planned_close_runtime as pr
        from embodied_jepa import grasp_rate_projected as rp
        from embodied_jepa import play_corpus as pc

        devices.configure_determinism("cuda", strict=False)
        robot = pc.make_play_robot()
        geometry = rp.RateGeometry.from_robot(robot)
        robot.close()
        report["geometry"] = geometry.describe()
        enc = _encode_roots(lr.encoder("cuda"), rep, data)
        results, problems = {}, []
        det_roots = [r["seed"] for r in rep["roots"][: gl.DETERMINISM_S]]
        for seed in gl.MODEL_SEEDS:
            model, info = lr.load_model(seed, "cuda")
            pm = lr.planner_model(model)
            res = {"model": info, "roots": {}, "determinism": {}}
            for r in rep["roots"]:
                s = r["seed"]
                for root in gl.ROOT_TYPES:
                    k = gl.ROOT_GOAL[root]
                    xin = (enc[f"{s}/{root}"], data[f"{s}/{root}/state28"],
                           enc[f"{s}/goal/{root}"], data[f"{s}/subgoal{k}/state28"])  # fmt: skip
                    plan, cost, req = lr.search_plan(pm, s, *xin)
                    t0 = data[f"{s}/{root}/t0"]
                    path = gl.executed_level(geometry, plan, t0)
                    sub = lr.substitution_costs(pm, *xin, plan, root)
                    res["roots"][f"{s}/{root}"] = {
                        # reported; Stage 0′ checked it on the development roots
                        "squeeze_request_path_deviation": _squeeze_request_deviation(
                            geometry, data, s, root
                        ),  # fmt: skip
                        "plan": plan.tolist(),
                        "cost": cost,
                        "requests_equal_plan": bool(rp.bits_equal(req, plan)),
                        "executed": path.tolist(),
                        "level": gl.window_level(path),
                        "level_step8": float(path[gl.HORIZON - 1]),
                        "first_requested_grasp": float(plan[0, gl.GRASP_DIM]),
                        "first_executed_grasp": float(path[0]),
                        "plan_cost": sub["plan"]["total"],
                        "substituted_cost": sub["substituted"]["total"],
                        "substitution_terms": sub,
                    }
                    if args.check_a2:
                        ref = pr.a2_plan(pm, gl.SETTING, s, *xin)
                        if not rp.bits_equal(plan, ref):
                            problems.append(f"G-v1-{seed} root {s} {root}: plan != a2_plan")
                    if s in det_roots:
                        again, cost2, _ = lr.search_plan(pm, s, *xin)
                        sub2 = lr.substitution_costs(pm, *xin, again, root)
                        same = bool(rp.bits_equal(again, plan) and cost2 == cost
                                    and sub2 == sub)  # fmt: skip
                        res["determinism"][f"{s}/{root}"] = same
                        if not same:
                            problems.append(f"G-v1-{seed} root {s} {root}: re-run differs")
            results[f"G-v1-{seed}"] = res
            log(f"G-v1-{seed}: {len(res['roots'])} plans")
            del model, pm
        report |= {
            "roots_sha256": roots_sha,
            "n": len(rep["roots"]),
            "results": results,
            "determinism_roots": det_roots,
            "problems": problems,
            "seconds": time.time() - job.t0,
            "completed": True,
            "cap_hit": False,
            "finished_utc": utc(),
        }
        write_json(out, report)
    log(f"search on {args.cohort}: {report['n']} roots, {len(problems)} problems, "
        f"{report['seconds']:.0f} s")  # fmt: skip
    return EXIT_VOID if problems else 0


# ----- Stage 0′: the comparator on the stored development plans (§17; R27.183 (5)) ------------
def cmd_substitute(args) -> int:
    """G-i's step-averaged cost of each stored development plan and of the same plan with the
    squeeze branch's right-grasp requests (one roll-out call per root, type and seed), with ℓ
    recomputed. Reported only; it sets nothing. No CEM call: the plans are Stage 0's."""
    if args.cohort != "dev-roots":
        raise rt.GuardError("substitute is Stage 0′'s job on the development roots only")
    report = base_report(args.cohort)
    out = Path(args.output) / f"substitute-{args.cohort}.json"
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    _lambda_check()
    rep, data, roots_sha = load_roots(args.roots, args.cohort, None)
    search_path = Path(args.search)
    search = json.loads(search_path.read_text())
    if search.get("roots_sha256") != roots_sha or not search.get("completed"):
        raise rt.GuardError("G-search: the stored search is not over these roots, or incomplete")
    if search.get("task") != gl.TASK or search.get("cohort") != args.cohort:
        raise rt.GuardError("G-search: not TASK-094's development search")
    report["preflight"] = preflight(gl.MODEL_SEEDS)
    ledger_claim(f"substitute-{args.cohort}")
    with Job(out, report, _cap(args)) as job:
        from embodied_jepa import devices
        from embodied_jepa import grasp_closure_locus_runtime as lr

        devices.configure_determinism("cuda", strict=False)
        from embodied_jepa import grasp_rate_projected as rp
        from embodied_jepa import play_corpus as pc

        robot = pc.make_play_robot()
        geometry = rp.RateGeometry.from_robot(robot)
        robot.close()
        enc = _encode_roots(lr.encoder("cuda"), rep, data)
        results, problems = {}, []
        for seed in gl.MODEL_SEEDS:
            model, info = lr.load_model(seed, "cuda")
            pm = lr.planner_model(model)
            stored = search["results"][f"G-v1-{seed}"]["roots"]
            res = {"model": info, "roots": {}}
            for r in rep["roots"]:
                s = r["seed"]
                for root in gl.ROOT_TYPES:
                    k = gl.ROOT_GOAL[root]
                    xin = (enc[f"{s}/{root}"], data[f"{s}/{root}/state28"],
                           enc[f"{s}/goal/{root}"], data[f"{s}/subgoal{k}/state28"])  # fmt: skip
                    st = stored[f"{s}/{root}"]
                    plan = np.asarray(st["plan"], np.float32)
                    sub = lr.substitution_costs(pm, *xin, plan, root)
                    path = np.asarray(st["executed"], np.float64)
                    dev_sq = _squeeze_request_deviation(geometry, data, s, root)
                    if dev_sq > gl.PATH_TOLERANCE:
                        problems.append(f"{s} {root}: squeeze requests off its applied grasp")
                    res["roots"][f"{s}/{root}"] = {
                        "squeeze_request_path_deviation": dev_sq,
                        "cem_cost": st["cost"],
                        "plan_cost": sub["plan"]["total"],
                        "substituted_cost": sub["substituted"]["total"],
                        "substitution_terms": sub,
                        "level": gl.window_level(path),
                        "level_step8": float(path[gl.HORIZON - 1]),
                    }
            results[f"G-v1-{seed}"] = res
            log(f"G-v1-{seed}: {len(res['roots'])} substituted costs")
            del model, pm
        report |= {
            "roots_sha256": roots_sha,
            "roots_revision": rep.get("revision"),
            "search_sha256": sha(search_path),
            "search_revision": search.get("revision"),
            "n": len(rep["roots"]),
            "results": results,
            "problems": problems,
            "seconds": time.time() - job.t0,
            "completed": True,
            "cap_hit": False,
            "finished_utc": utc(),
        }
        write_json(out, report)
    log(f"substitute on {args.cohort}: {report['n']} roots, {report['seconds']:.0f} s")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--output", required=True)
    v.add_argument(
        "--task087-evidence", default=str(Path.home() / "develop/emai/evidence/task087-val")
    )
    se = sub.add_parser("seeds")
    se.add_argument("--output", required=True)
    se.add_argument("--evidence", default=str(Path.home() / "develop/emai/evidence"))
    po = sub.add_parser("power")
    po.add_argument("--output", required=True)
    po.add_argument("--sims", type=int, default=gl.POWER_SIMS)
    po.add_argument("--workers", type=int, default=12)
    g = sub.add_parser("goals")
    g.add_argument("--cohort", required=True, choices=gl.K_COHORTS)
    g.add_argument("--output", required=True)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--limit", type=int)
    g.add_argument("--cap-seconds", type=int)
    k = sub.add_parser("k-run")
    k.add_argument("--cohort", required=True, choices=gl.K_COHORTS)
    k.add_argument("--arm", required=True)
    k.add_argument("--goals", required=True)
    k.add_argument("--output", required=True)
    k.add_argument("--workers", type=int, default=8)
    k.add_argument("--limit", type=int)
    k.add_argument("--determinism", action="store_true")
    k.add_argument("--suffix", help="debug: an output name suffix")
    k.add_argument("--cap-seconds", type=int)
    ro = sub.add_parser("roots")
    ro.add_argument("--cohort", required=True, choices=gl.ROOT_COHORTS)
    ro.add_argument("--output", required=True)
    ro.add_argument("--workers", type=int, default=8)
    ro.add_argument("--limit", type=int)
    ro.add_argument("--cap-seconds", type=int)
    for name in ("rank", "search"):
        p = sub.add_parser(name)
        p.add_argument("--cohort", required=True, choices=gl.ROOT_COHORTS)
        p.add_argument("--roots", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--limit", type=int)
        p.add_argument("--cap-seconds", type=int)
        if name == "search":
            p.add_argument("--check-a2", action="store_true", help="debug: plan == pr.a2_plan")
    su = sub.add_parser("substitute")
    su.add_argument("--cohort", required=True, choices=("dev-roots",))
    su.add_argument("--roots", required=True)
    su.add_argument("--search", required=True)
    su.add_argument("--output", required=True)
    su.add_argument("--cap-seconds", type=int)
    args = parser.parse_args(argv)
    return {
        "verify": cmd_verify,
        "seeds": cmd_seeds,
        "power": cmd_power,
        "goals": cmd_goals,
        "k-run": cmd_k_run,
        "roots": cmd_roots,
        "rank": cmd_rank,
        "search": cmd_search,
        "substitute": cmd_substitute,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
