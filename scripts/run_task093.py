"""TASK-093: Stage 0's checks and calibration, Stage P and the closed loop (protocol
docs/experiments/grasp_rate_projected_planner_v1.md). Subcommands; every GPU use (models, MuJoCo
rendering) is one job under ``scripts/gpu_run.sh --wait --min-free-gib 8 --board`` and stops at
the 7 200 s cap:

* ``verify --output DIR``: §14's hash checks -- TASK-087's three G checkpoints against their pins,
  reports and evidence copies, the feature report and store files, the projection, the moments,
  TASK-087's implementation hash, λ, the rate limit and every frozen file (CPU).
* ``seeds --output FILE``: §9's uniqueness check at this revision, over every remote branch and
  the evidence (CPU).
* ``power --output FILE``, ``rows --output FILE``: §6.4's power table and B-row rates with 20 000
  simulations (CPU).
* ``equivalence --output FILE``: §14's projector test against ``robot.project_candidates`` on
  random request sequences from debug states (zero and non-zero arm deltas), bitwise.
* ``roots --cohort``: TASK-089's roots (``zero_shot_grasp.build_root``) for cal-roots, p-roots or
  debug-roots.
* ``plans --cohort``: one CEM call per root at R1 and R2 for M-proj and M-v1 on each seed, the
  executed measures, the references and the plumbing checks: the calibration of §6.1 on
  cal-roots, Stage P on p-roots (§6.2), the smoke on debug-roots.
* ``smoke-checks --output FILE``: §14's read-only-array check (``torch.as_tensor`` on the
  projection's read-only arrays) and the identity-projector pairing on a real model.
* ``goals``, ``run``, ``combine``, ``video``: the closed loop (§6.3).

TASK-087's to TASK-092's frozen code is imported, never edited, and checked against its pins
before anything runs (``grasp_rate_projected.FROZEN_FILES_SHA256``). Every job must start in this
checkout's root with ``third_party/le-wm`` present (the loader resolves it from the working
directory; R27.107) and loads its models before any ledger claim. Non-debug stages need a clean
tracked tree; every report records the revision; every file is refused if it exists, and a
non-debug root set, plan set, goal set or run is refused if the ledger (``outputs/task093-ledger``)
shows it was started before (§7: a cohort run twice is VOID).
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
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import grasp_corpus_wm as gc  # noqa: E402
from embodied_jepa import grasp_rate_projected as rp  # noqa: E402
from embodied_jepa import grounded_wm_data as gd  # noqa: E402
from embodied_jepa import run_tools as rt  # noqa: E402
from embodied_jepa import zero_shot as zs  # noqa: E402
from embodied_jepa import zero_shot_grasp as zg  # noqa: E402

LEDGER = ROOT / "outputs" / "task093-ledger"
LEWM_SOURCE = ROOT / "third_party" / "le-wm"
EXIT_CAP = 4
EXIT_VOID = 3
DISK_FLOOR_GIB = 10.0


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc()}] {msg}", flush=True)


def git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


sha = gd.sha256_file


def write_json(path, value) -> None:
    path = Path(path)
    if path.exists():
        raise rt.GuardError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=True) + "\n")


class CapHit(Exception):
    pass


def _alarm(signum, frame):
    raise CapHit(f"the per-job cap of {rp.JOB_CAP_SECONDS} s was reached")


def tree_guard() -> dict:
    """R27.107's lesson: the job runs in this checkout's root, where ``third_party/le-wm`` is."""
    cwd = Path.cwd().resolve()
    if cwd != ROOT.resolve():
        raise rt.GuardError(f"G-cwd: run from {ROOT}, not {cwd}")
    if not LEWM_SOURCE.is_dir():
        raise rt.GuardError(f"G-lewm: {LEWM_SOURCE} is missing")
    return {"cwd": str(cwd), "lewm_source": str(LEWM_SOURCE.resolve())}


def base_report(cohort=None, *, clean=False, gpu=True) -> dict:
    """TASK-092's report and guards with TASK-093's task, protocol, frozen-file check and the
    working-directory guard."""
    frozen = rp.frozen_files_check(ROOT)
    if not all(frozen.values()):
        bad = [k for k, ok in frozen.items() if not ok]
        raise rt.GuardError(f"G-frozen: these frozen files differ from their pins: {bad}")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    report = {
        "task": rp.TASK,
        "protocol": rp.PROTOCOL,
        "started_utc": utc(),
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": dirty,
        "argv": sys.argv,
        "platform": platform.platform(),
        "cohort": cohort,
        "frozen_files_match": True,
        "modules_sha256": rp.own_modules_sha256(ROOT),
        "job_cap_seconds": rp.JOB_CAP_SECONDS,
        "tree": tree_guard(),
    }
    rt.assert_local_import(ROOT, report)
    if (clean or (cohort is not None and not cohort.startswith("debug"))) and dirty:
        raise rt.GuardError("G-clean: non-debug stages run from a clean tracked tree")
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
    """A non-debug cohort's roots, plans, goals or run may start only once (§7)."""
    LEDGER.mkdir(parents=True, exist_ok=True)
    path = LEDGER / key
    try:
        with open(path, "x") as f:
            f.write(json.dumps({"utc": utc(), "revision": git("rev-parse", "HEAD")}) + "\n")
    except FileExistsError as error:
        raise rt.GuardError(f"G-twice: {key} was started before ({path})") from error


def preflight(seeds) -> dict:
    """Load every model a job needs (on the CPU) before any ledger claim (R27.109)."""
    from embodied_jepa import grasp_rate_projected_runtime as rr

    out = {}
    for s in seeds:
        _, info = rr.load_model(int(s), "cpu")
        out[f"G-v1-{s}"] = info["checkpoint_sha256"]
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
    cap = getattr(args, "cap_seconds", None) or rp.JOB_CAP_SECONDS
    if cap != rp.JOB_CAP_SECONDS and not str(getattr(args, "cohort", "debug")).startswith("debug"):
        raise rt.GuardError("the per-job cap is 7 200 s outside debug cohorts")
    return int(cap)


def _manifest() -> dict:
    return json.loads((ROOT / "configs" / "g1_sim_action.json").read_text())


def _lambda_check():
    from embodied_jepa import grasp_corpus_wm_runtime as gr

    if not (zs.LAMBDA == gr.LAMBDA == rp.LAMBDA) or gr.SETTING != rp.SETTING:
        raise rt.GuardError("G-lambda: λ or the setting is not TASK-089's")


# ----- §14: verify -----------------------------------------------------------------------------
def cmd_verify(args) -> int:
    from embodied_jepa import grasp_corpus_wm_runtime as gr
    from embodied_jepa import grasp_rate_aware as ra
    from embodied_jepa import grounded_wm_model as gm

    out = Path(args.output)
    path = out / "verify-task093.json"
    if path.exists():
        raise rt.GuardError(f"refusing to overwrite {path}")
    report = base_report(gpu=False)
    checks = {}
    feat = gr.V1_FEATURES
    frep = json.loads((feat / "report.json").read_text())
    checks["feature_report"] = sha(feat / "report.json") == rp.FEATURE_REPORT_SHA256
    t0 = time.monotonic()
    report["store_files"] = gd.verify_store_files(feat, frep["files_sha256"], ("train", "val"))
    report["store_seconds"] = time.monotonic() - t0
    checks["store_files"] = True
    checks["projection_file"] = sha(feat / "projection.npz") == frep["projection"]["file_sha256"]
    gd.load_projection(feat / "projection.npz", gr.PROJECTION_SHA256)
    checks["projection_arrays"] = True
    ckpts = {}
    ev = Path(args.task087_evidence)
    for seed in rp.MODEL_SEEDS:
        folder = gr.V1_CHECKPOINTS / f"G-{seed}"
        rep = json.loads((folder / "report.json").read_text())
        row = {
            "model_pt": sha(folder / "model.pt"),
            "recorded": rep["checkpoint_sha256"],
            "pinned": rp.CHECKPOINT_SHA256[f"G-v1-{seed}"],
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
    checks["moments_pin"] = report["moments"]["pinned_sha256"] == rp.MOMENTS_SHA256
    checks["implementation_hash"] = gm.implementation_sha256() == rp.IMPLEMENTATION_SHA256
    checks["lambda"] = zs.LAMBDA == rp.LAMBDA == gr.LAMBDA == 2.0552
    checks["rate_limit"] = abs(ra.rate_limit_from_manifest(_manifest()) - 2 / 11) < 1e-12
    checks["cem"] = zs.CEM == {
        "horizon": 8, "samples": 300, "iterations": 10, "elites": 30, "minimum_std": 0.05,
    }  # fmt: skip
    report["frozen_files"] = rp.frozen_files_check(ROOT)
    checks["frozen_files"] = all(report["frozen_files"].values())
    report["checks"] = checks
    report["all_pass"] = all(checks.values())
    report["finished_utc"] = utc()
    write_json(path, report)
    log(json.dumps(checks))
    return 0 if report["all_pass"] else 1


# ----- §9: the seed uniqueness check -----------------------------------------------------------
SEED_PATTERN = re.compile(r"(?<![0-9A-Za-z.])(91[678][0-9]{3}|919[0-9]{2}|9011)(?![0-9A-Za-z])")
CHECK_RANGES = {
    "cal-roots": range(916000, 916064),
    "p-roots": range(917000, 917064),
    "bs-grasp": range(918000, 918128),
    "bd-grasp": range(91900, 91932),
    "debug": range(91940, 92000),
    "declared span 916000-918999": range(916000, 919000),
    "declared span 91900-91999": range(91900, 92000),
    "salt 9011": range(9011, 9012),
}
OWN_MARKERS = ("task093", "TASK-093", "grasp_rate_projected")


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
    paths = ["src", "scripts", "docs", ".mc", "tests", "benchmarks", "configs"]
    hits = {}
    for branch in branches + ["HEAD"]:
        res = subprocess.run(
            ["git", "grep", "-n", "-I", "-E", r"91[678][0-9]{3}|919[0-9]{2}|9011", branch, "--"]
            + paths,
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
        if not path.is_file() or path.suffix not in exts or "task093" in str(path):
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
        "paths": paths,
        "ranges": {k: [r.start, r.stop - 1] for k, r in CHECK_RANGES.items()},
        "hits_in_task093_files": classified["own"],
        "hits_elsewhere": classified["other"],
        "evidence_dir": str(args.evidence),
        "evidence_hits": ev_hits,
        "evidence_hits_in_seed_fields": seed_fields,
        "finished_utc": utc(),
    }
    write_json(out, report)
    log(
        f"{len(branches)} branches; hits outside TASK-093's files: {len(classified['other'])} "
        f"files; evidence tokens: {len(ev_hits)} ({len(seed_fields)} in seed fields)"
    )
    return 0


# ----- §6.4: power and row rates (CPU) ---------------------------------------------------------
def _power_cell(cell):
    alt, n, sd, sims = cell
    r = gc.power(n, sd, alt=alt, sims=sims)
    return {"alt": list(alt), "n": n, "sd": sd, "power": r["power"], "sims": sims,
            "marginal_rates": r["marginal_rates"]}  # fmt: skip


def cmd_power(args) -> int:
    report = base_report(gpu=False)
    t0 = time.monotonic()
    cells = [(alt, n, sd, args.sims) for alt, n in rp.POWER_GRID for sd in rp.POWER_SDS]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(_power_cell, cells)
    report |= {
        "rows": rows,
        "sims": args.sims,
        "alternative": list(rp.POWER_ALTERNATIVE),
        "rule": rp.power_rule(rows),
        "seconds": time.monotonic() - t0,
        "finished_utc": utc(),
    }
    write_json(args.output, report)
    for r in rows:
        log(f"{r['alt']} n {r['n']} SD {r['sd']}: power {r['power']:.4f}")
    log(json.dumps(report["rule"]))
    return 0


def _row_cell(cell):
    alt, n, sd, sims = cell
    return rp.row_rates(alt, n, sd, sims=sims)


def cmd_rows(args) -> int:
    report = base_report(gpu=False)
    t0 = time.monotonic()
    cells = [(alt, 128, sd, args.sims) for alt in rp.ROW_GRID for sd in rp.POWER_SDS]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(_row_cell, cells)
    report |= {
        "rows": rows,
        "sims": args.sims,
        "bar": rp.ROW_BAR,
        "random_rate": 0.0,
        "seconds": time.monotonic() - t0,
        "finished_utc": utc(),
    }
    write_json(args.output, report)
    for r in rows:
        log(f"{r['alt']} SD {r['sd']}: {r['rates']}")
    return 0


# ----- §14: the projector against the embodiment -----------------------------------------------
EQ_SEEDS = (91940, 91941, 91942, 91943)  # debug layouts
EQ_COMMANDS = 60
EQ_EVERY = 6
EQ_CANDIDATES = 24


def _eq_requests(rng, nonzero: bool) -> np.ndarray:
    """Random request sequences [1, K, 8, 14] within TASK-088's bounds: the right grasp a mix of
    uniform, saturated and near-previous values; the right arm zero or N(0, 0.5)."""
    k, h = EQ_CANDIDATES, rp.HORIZON
    req = np.zeros((1, k, h, 14), np.float32)
    req[..., 12] = -1.0
    kind = rng.integers(0, 3, (k, h))
    g = np.where(kind == 0, rng.uniform(-1, 1, (k, h)), np.where(kind == 1, rng.choice(
        [-1.0, 1.0], (k, h)), rng.normal(0, 0.15, (k, h))))  # fmt: skip
    req[0, :, :, 13] = np.clip(g, -1, 1)
    if nonzero:
        req[0, :, :, 6:12] = np.clip(rng.normal(0, 0.5, (k, h, 6)), -1, 1)
    return np.clip(req, zs.FIXED_LOWER, zs.FIXED_UPPER).astype(np.float32)


def cmd_equivalence(args) -> int:
    """§14: on random request sequences from debug states, with zero and with non-zero right-arm
    deltas, the projector's grasp channel equals ``robot.project_candidates``' at every step,
    bitwise in float32 (with non-zero deltas on the sequences the embodiment marks feasible)."""
    from embodied_jepa import play_corpus as pc

    report = base_report("debug-roots")
    out = Path(args.output)
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    rows = []
    with Job(out, report, _cap(args)) as job:
        robot = pc.make_play_robot()
        geometry = rp.RateGeometry.from_robot(robot)
        report["geometry"] = geometry.describe()
        rng = np.random.default_rng([rp.SALTS["random_reference"], 914])
        for seed in EQ_SEEDS:
            zs.reset(robot, zs.sample_layout("grasp", seed, 0))
            want = -1.0
            for step in range(EQ_COMMANDS):
                obs = robot.observe()
                if step % EQ_EVERY == 0:
                    t0 = rp.read_t0(robot)
                    for nonzero in (False, True):
                        req = _eq_requests(rng, nonzero)
                        emb = robot.project_candidates(req)
                        mine = rp.GraspRateProjector(geometry, t0)(req)
                        feas = emb.feasible[0]
                        a = emb.actions[0, feas, :, 13]
                        b = mine.actions[0, feas, :, 13]
                        rows.append({
                            "seed": seed, "step": step, "nonzero_arm": nonzero,
                            "t0_grasp": zs.synergy_grasp(robot, t0),
                            "sequences": int(req.shape[1]), "feasible": int(feas.sum()),
                            "compared_steps": int(a.size),
                            "mismatches": int((a.view(np.uint32) != b.view(np.uint32)).sum()),
                            "clipped_steps": int((b != req[0, feas, :, 13]).sum()),
                            "other_channels_equal": rp.bits_equal(
                                np.delete(mine.actions, 13, -1), np.delete(req, 13, -1)),
                            "arm_deltas_nonzero": int(
                                (np.abs(req[0, feas, :, 6:12]) > 0).any(-1).sum()),
                        })  # fmt: skip
                if step % 15 == 0:
                    want = float(rng.choice([-1.0, 1.0, rng.uniform(-1, 1)]))
                p = zs.proprio(robot, obs)
                target = p["palm"] + rng.normal(0, 0.01, 3)
                request = zs.tracking_action(robot, p["palm"], p["rotation"], target,
                                             p["rotation"], want)  # fmt: skip
                try:
                    zs.execute(robot, request)
                except zs.Stopped as error:
                    log(f"seed {seed} stopped at {step}: {error}")
                    break
        robot.close()
        mism = sum(r["mismatches"] for r in rows)
        report |= {
            "rows": rows,
            "states": len(rows) // 2,
            "compared_steps": sum(r["compared_steps"] for r in rows),
            "clipped_steps": sum(r["clipped_steps"] for r in rows),
            "mismatches": mism,
            "other_channels_equal": all(r["other_channels_equal"] for r in rows),
            "nonzero_feasible_sequences": sum(r["feasible"] for r in rows if r["nonzero_arm"]),
            "pass": bool(
                mism == 0
                and rows
                and all(r["other_channels_equal"] for r in rows)
                and sum(r["clipped_steps"] for r in rows) > 0
                and any(r["nonzero_arm"] and r["arm_deltas_nonzero"] for r in rows)
            ),
            "seconds": time.time() - job.t0,
            "completed": True,
            "cap_hit": False,
            "finished_utc": utc(),
        }
        write_json(out, report)
    log(f"equivalence: {report['compared_steps']} steps, {report['clipped_steps']} clipped, "
        f"{report['mismatches']} mismatches, pass {report['pass']}")  # fmt: skip
    return 0 if report["pass"] else EXIT_VOID


# ----- roots (§6.1, §6.2) ----------------------------------------------------------------------
def _roots_worker(seeds):
    from embodied_jepa import play_corpus as pc

    robot = pc.make_play_robot()
    out = [zg.build_root(robot, s) for s in seeds]
    robot.close()
    return out


def _limit(args, seeds):
    if args.limit and not args.cohort.startswith("debug"):
        raise rt.GuardError("--limit is for debug cohorts only")
    return seeds[: args.limit or None]


def cmd_roots(args) -> int:
    if args.cohort not in rp.ROOT_COHORTS:
        raise rt.GuardError(f"{args.cohort} is not a root cohort")
    report = base_report(args.cohort)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    jpath, npath = out / f"roots-{args.cohort}.json", out / f"roots-{args.cohort}.npz"
    for p in (jpath, npath):
        if p.exists():
            raise rt.GuardError(f"refusing to overwrite {p}")
    seeds = _limit(args, list(rp.COHORTS[args.cohort]))
    if not args.cohort.startswith("debug"):
        ledger_claim(f"roots-{args.cohort}")
    with Job(jpath, report, _cap(args)) as job:
        chunks = [seeds[i :: args.workers] for i in range(args.workers)]
        with mp.get_context("spawn").Pool(args.workers) as pool:
            parts = pool.map(_roots_worker, [c for c in chunks if c])
        roots = sorted((r for p in parts for r in p), key=lambda r: r["seed"])
        arrays, meta = {}, []
        for r in roots:
            s = r["seed"]
            keys = ("seed", "attempt", "layout", "target", "yaw", "z0", "rejected")
            row = {k: r[k] for k in keys}
            for k, g in enumerate(r["subgoals"]):
                arrays[f"{s}/subgoal{k}/frame"] = g["frame"]
                arrays[f"{s}/subgoal{k}/state28"] = g["state28"]
            for root, rec in r["roots"].items():
                arrays[f"{s}/{root}/frame"] = rec["frame"]
                arrays[f"{s}/{root}/state28"] = rec["state28"]
            meta.append(row)
        np.savez_compressed(npath, **arrays)
        report |= {
            "seconds": time.time() - job.t0,
            "roots": meta,
            "arrays_sha256": sha(npath),
            "rejections": sum(len(r["rejected"]) for r in roots),
            "completed": True,
            "cap_hit": False,
            "finished_utc": utc(),
        }
        write_json(jpath, report)
    log(f"{len(roots)} roots, {report['rejections']} rejected draws, {report['seconds']:.0f} s")
    return 0


def load_roots(folder, cohort, revision):
    folder = Path(folder)
    rep = json.loads((folder / f"roots-{cohort}.json").read_text())
    if not rep.get("completed"):
        raise rt.GuardError("G-roots: the roots job did not complete")
    npath = folder / f"roots-{cohort}.npz"
    if sha(npath) != rep["arrays_sha256"]:
        raise rt.GuardError("G-roots: arrays differ from the roots report")
    seeds = [r["seed"] for r in rep["roots"]]
    for s in seeds:
        rp.check_seed(cohort, s)
    if not cohort.startswith("debug"):
        if seeds != list(rp.COHORTS[cohort]):
            raise rt.GuardError("G-roots: the cohort's seeds are not exactly its roots")
        if rep["revision"] != revision:
            raise rt.GuardError("G-roots: the roots were made at another revision")
    return rep, np.load(npath), sha(folder / f"roots-{cohort}.json")


# ----- plans: §6.1's calibration, Stage P and the smoke ----------------------------------------
def cmd_plans(args) -> int:
    """One CEM call per root at R1 and at R2 with the A2 seed for M-proj and M-v1 on each seed:
    the executed measures C8@R1 and H8@R2, the plumbing checks (every M-proj plan in the rate set
    and its own projection; both arms' first-iteration samples identical; every M-v1 plan equal
    to ``pr.a2_plan``'s bitwise), P-MECH, the requests beyond the limit and TASK-089's request
    measures (reported), and the references. On cal-roots this is §6.1's calibration (bars and
    degeneracy checks); on p-roots it is Stage P's gate with the frozen bars."""
    from embodied_jepa import devices
    from embodied_jepa import grasp_corpus_wm_planned_close_runtime as pr
    from embodied_jepa import grasp_corpus_wm_runtime as gr
    from embodied_jepa import grasp_rate_projected_runtime as rr
    from embodied_jepa import play_corpus as pc

    if args.cohort not in rp.ROOT_COHORTS:
        raise rt.GuardError(f"{args.cohort} is not a root cohort")
    debug = args.cohort.startswith("debug")
    if args.stage_p_smoke and not debug:
        raise rt.GuardError("--stage-p-smoke is for debug cohorts only")
    stage_p = args.cohort == "p-roots" or bool(args.stage_p_smoke)
    if stage_p and rp.P_BARS is None:
        raise rt.GuardError("G-bars: Stage P needs the bars frozen at Stage 0")
    report = base_report(args.cohort)
    out = Path(args.output) / f"plans-{args.cohort}.json"
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    _lambda_check()
    manifest = _manifest()
    report["preflight"] = preflight(rp.MODEL_SEEDS)
    if not debug:
        ledger_claim(f"plans-{args.cohort}")
    with Job(out, report, _cap(args)) as job:
        devices.configure_determinism("cuda", strict=False)
        rep, data, roots_sha = load_roots(args.roots, args.cohort, report["revision"])
        if args.limit:
            if not debug:
                raise rt.GuardError("--limit is for debug cohorts only")
            rep = dict(rep, roots=rep["roots"][: args.limit])
        robot = pc.make_play_robot()
        geometry = rp.RateGeometry.from_robot(robot)
        robot.close()
        report["geometry"] = geometry.describe()
        enc_fn = gr.encoder("cuda")
        enc = {}
        for r in rep["roots"]:
            s = r["seed"]
            for root in zg.ROOTS:
                enc[f"{s}/{root}"] = enc_fn(data[f"{s}/{root}/frame"])[0]
                enc[f"{s}/goal/{root}"] = enc_fn(data[f"{s}/subgoal{zg.ROOT_GOAL[root]}/frame"])[0]
        seeds_r = [r["seed"] for r in rep["roots"]]
        problems, results = [], {}
        for seed in rp.MODEL_SEEDS:
            model, info = rr.load_model(seed, "cuda")
            pm = gr.planner_model(model)
            res = {arm: {"plans": {}, "requested": {}, "first_sha": {}} for arm in rp.P_ARMS}
            for arm in rp.P_ARMS:
                for root in zg.ROOTS:
                    res[arm]["plans"][root], res[arm]["requested"][root] = [], []
                    res[arm]["first_sha"][root] = []
            mech = {arm: [] for arm in rp.P_ARMS}
            pred_r1 = {arm: [] for arm in rp.P_ARMS}
            beyond = {root: [] for root in zg.ROOTS}
            for r in rep["roots"]:
                s = r["seed"]
                for root in zg.ROOTS:
                    xin = (
                        enc[f"{s}/{root}"],
                        data[f"{s}/{root}/state28"],
                        enc[f"{s}/goal/{root}"],
                        data[f"{s}/subgoal{zg.ROOT_GOAL[root]}/state28"],
                    )
                    t0 = rp.root_t0(geometry, rp.ROOT_GRASP[root])
                    proj = rp.GraspRateProjector(geometry, t0)
                    plan_p, req_p, first_p = rr.root_plan(pm, s, *xin, projector=proj)
                    plan_c, req_c, first_c = rr.root_plan(pm, s, *xin)
                    ref = pr.a2_plan(pm, rp.SETTING, s, *xin)
                    if not rp.bits_equal(plan_c, ref):
                        problems.append(f"G-v1-{seed} root {s} {root}: M-v1 plan != a2_plan")
                    if first_p != first_c:
                        problems.append(f"G-v1-{seed} root {s} {root}: first samples differ")
                    if not rp.in_rate_set(geometry, plan_p, root):
                        problems.append(f"G-v1-{seed} root {s} {root}: M-proj off the rate set")
                    for arm, plan, req, first in (("M-proj", plan_p, req_p, first_p),
                                                  ("M-v1", plan_c, req_c, first_c)):  # fmt: skip
                        res[arm]["plans"][root].append(plan.tolist())
                        res[arm]["requested"][root].append(req.tolist())
                        res[arm]["first_sha"][root].append(first)
                    beyond[root].append(rp.requests_beyond(req_p, plan_p[:, rp.GRASP_DIM]))
                    if root == "R1":
                        for arm, plan in (("M-proj", plan_p), ("M-v1", plan_c)):
                            pred = rr.predicted_synergy(model, xin[0], plan, xin[1], manifest)
                            exe = rp.executed_path(geometry, plan, "R1")
                            pred_r1[arm].append(pred)
                            mech[arm].append(float(np.mean(np.abs(np.asarray(pred) - exe))))
            for arm in rp.P_ARMS:
                p1 = np.asarray(res[arm]["plans"]["R1"], np.float32)
                p2 = np.asarray(res[arm]["plans"]["R2"], np.float32)
                counts = rp.measure_counts(geometry, p1, p2)
                first = zg.a2_measures(p1[:, 0], p2[:, 0])
                res[arm] |= {
                    "C8": counts["C8"],
                    "H8": counts["H8"],
                    "C8_per_root": counts["C8_per_root"],
                    "H8_per_root": counts["H8_per_root"],
                    "executed_R1": [rp.executed_path(geometry, p, "R1").tolist() for p in p1],
                    "executed_R2": [rp.executed_path(geometry, p, "R2").tolist() for p in p2],
                    "p_mech_mean_abs_gap_R1": float(np.mean(mech[arm])),
                    "p_mech_per_root": mech[arm],
                    "predicted_synergy_R1": pred_r1[arm],
                    "close_R1_request_terms": int(first["close_R1"].sum()),
                    "lift_R2_request_terms": int(first["lift_R2"].sum()),
                }
            res["M-proj"]["requests_beyond_limit"] = {
                root: {
                    "steps": int(sum(b["steps"] for b in v)),
                    "beyond": int(sum(b["beyond"] for b in v)),
                }
                for root, v in beyond.items()
            }
            res["model"] = info
            if stage_p:
                res["p_gate"] = rp.p_gate(res["M-proj"], rp.P_BARS)
            log(
                f"G-v1-{seed}: M-proj C8 {res['M-proj']['C8']} H8 {res['M-proj']['H8']}; "
                f"M-v1 C8 {res['M-v1']['C8']} H8 {res['M-v1']['H8']}"
            )
            results[f"G-v1-{seed}"] = res
            del model
        refs = rp.reference_counts(geometry, seeds_r)
        report |= {
            "roots_sha256": roots_sha,
            "n": len(rep["roots"]),
            "results": results,
            "references": refs,
            "problems": problems,
            "seconds": time.time() - job.t0,
            "completed": True,
            "cap_hit": False,
            "finished_utc": utc(),
        }
        control = {m: [results[f"G-v1-{s}"]["M-v1"][m] for s in rp.MODEL_SEEDS]
                   for m in rp.MEASURES}  # fmt: skip
        if not stage_p:  # §6.1's calibration (debug-roots: the smoke's reading, not a bar)
            report["calibration"] = {m: rp.degeneracy(m, control[m], refs) for m in rp.MEASURES}
        else:
            passes = {s: results[f"G-v1-{s}"]["p_gate"] for s in rp.MODEL_SEEDS}
            report["bars"] = rp.P_BARS
            report["p_row"] = rp.p_row(passes, void="; ".join(problems))
        write_json(out, report)
    bars = {m: v["bar"] for m, v in report.get("calibration", {}).items()}
    log(f"plans on {args.cohort}: {len(problems)} problems; bars {bars or report.get('bars')}")
    return EXIT_VOID if problems else 0


# ----- §14: the read-only arrays and the identity-projector pairing on a real model ------------
def cmd_smoke_checks(args) -> int:
    import warnings

    import torch

    from embodied_jepa import devices
    from embodied_jepa import grasp_corpus_wm_runtime as gr
    from embodied_jepa import grasp_rate_projected_runtime as rr
    from embodied_jepa import zero_shot_grasp_runtime as zgr
    from embodied_jepa.constraints import CandidateProjection

    report = base_report("debug-roots")
    out = Path(args.output)
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    with Job(out, report, _cap(args)) as job:
        devices.configure_determinism("cuda", strict=False)
        rep, data, _ = load_roots(args.roots, "debug-roots", report["revision"])
        model, info = rr.load_model(87100, "cuda")
        pm = gr.planner_model(model)
        enc_fn = gr.encoder("cuda")
        rows = []
        for r in rep["roots"][: args.limit]:
            s = r["seed"]
            for root in zg.ROOTS:
                xin = (
                    enc_fn(data[f"{s}/{root}/frame"])[0],
                    data[f"{s}/{root}/state28"],
                    enc_fn(data[f"{s}/subgoal{zg.ROOT_GOAL[root]}/frame"])[0],
                    data[f"{s}/subgoal{zg.ROOT_GOAL[root]}/state28"],
                )
                model_s, _ = zgr.wrap(pm, rp.SETTING)
                lat, goal = model_s.encode(*xin[:2]), model_s.encode_goal(*xin[2:])
                rng = np.random.default_rng([rp.SALTS["random_reference"], s, 1])
                cand = np.clip(rng.normal(0, 1, (1, 300, 8, 14)), zs.FIXED_LOWER,
                               zs.FIXED_UPPER).astype(np.float32)  # fmt: skip
                ro = CandidateProjection(cand, np.ones((1, 300), bool)).actions
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    d_ro = model_s.distance(model_s.predict(lat, ro), goal)
                d_rw = model_s.distance(model_s.predict(lat, cand.copy()), goal)

                def ident(req):
                    return CandidateProjection(np.asarray(req), np.ones(req.shape[:2], bool))

                ident.first_requested_sha256 = None
                planner_i = rr.projected_planner(s, 8)
                plan_i = planner_i.plan(model_s, lat, goal, projector=ident)
                plan_u = zgr.make_planner(s, 8).plan(model_s, lat, goal)
                rows.append({
                    "seed": s, "root": root,
                    "readonly_writeable": bool(ro.flags.writeable),
                    "readonly_cost_bitwise_equal": rp.bits_equal(d_ro, d_rw),
                    "readonly_warnings": sorted({str(w.message)[:120] for w in caught}),
                    "identity_plan_bitwise_equal": rp.bits_equal(plan_i.actions, plan_u.actions),
                    "identity_cost_equal": bool(np.array_equal(plan_i.costs, plan_u.costs)),
                })  # fmt: skip
        report |= {
            "model": info,
            "rows": rows,
            "torch": torch.__version__,
            "pass": all(
                r["readonly_cost_bitwise_equal"]
                and r["identity_plan_bitwise_equal"]
                and r["identity_cost_equal"]
                and not r["readonly_writeable"]
                for r in rows
            ),  # fmt: skip
            "seconds": time.time() - job.t0,
            "completed": True,
            "cap_hit": False,
            "finished_utc": utc(),
        }
        write_json(out, report)
    log(f"smoke checks: pass {report['pass']} on {len(rows)} root calls")
    return 0 if report["pass"] else EXIT_VOID


# ----- §6.3: goals, runs, combine, video -------------------------------------------------------
def _goal_worker(seeds):
    from embodied_jepa import play_corpus as pc

    robot = pc.make_play_robot()
    out = [zs.make_grasp_goal(robot, s) for s in seeds]
    robot.close()
    return out


def cmd_goals(args) -> int:
    if args.cohort not in rp.GRASP_COHORTS:
        raise rt.GuardError(f"{args.cohort} is not a closed-loop cohort")
    report = base_report(args.cohort)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    jpath, npath = out / f"goals-{args.cohort}.json", out / f"goals-{args.cohort}.npz"
    for p in (jpath, npath):
        if p.exists():
            raise rt.GuardError(f"refusing to overwrite {p}")
    seeds = _limit(args, list(rp.COHORTS[args.cohort]))
    if not args.cohort.startswith("debug"):
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
            rp.check_seed(cohort, row["seed"])
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
    if not cohort.startswith("debug"):
        if [g["seed"] for g in goals] != list(rp.COHORTS[cohort]):
            raise rt.GuardError("G-goals: the cohort's seeds are not exactly its goals")
        if revision is not None and rep["revision"] != revision:
            raise rt.GuardError("G-goals: the goals were made at another revision")
    return goals, sha(folder / f"goals-{cohort}.json")


def _controller(arm, model_seed, episode_seed, cache):
    if arm == "hold":
        return zs.HoldController()
    if arm == "random":
        return zs.RandomController(episode_seed)
    if arm == "ik":
        return zs.IKController()
    if arm == "ik-nopress":
        return zs.IKController(press=False)
    from embodied_jepa import grasp_corpus_wm_runtime as gr
    from embodied_jepa import grasp_rate_projected_runtime as rr

    if "model" not in cache:
        from embodied_jepa import devices

        devices.configure_determinism("cuda", strict=False)
        cache["model"], cache["info"] = rr.load_model(model_seed, "cuda")
        cache["encoder"] = gr.encoder("cuda")
    return rr.controller(arm, cache["model"], cache["encoder"], episode_seed)


def _run_worker(job):
    arm, model_seed, goals = job
    from embodied_jepa import play_corpus as pc
    from embodied_jepa import zero_shot_runtime as zr

    robot = pc.make_play_robot()
    cache, out = {}, []
    for goal in goals:
        controller = _controller(arm, model_seed, goal["seed"], cache)
        result = zr.run_episode(robot, goal, controller)
        result["goal_digest"] = goal["digest"]
        if hasattr(controller, "setting"):
            result["setting"] = controller.setting
        if hasattr(controller, "plan_log"):
            result["plan_log"] = controller.plan_log
            result["projected"] = bool(rp.CLOSED_LOOP_ARMS[arm])
            result["deferral"] = rp.deferral_measures(result)
            result["rate"] = rp.rate_measures(result)
            result["equivalence"] = rp.closed_loop_equivalence(result)
            result["requests"] = rp.request_measures(result)
        out.append(result)
        log(
            f"{arm} {goal['seed']} success={result['success']} steps={result['steps']} "
            f"{result['stop_reason']}"
        )
    robot.close()
    return out, cache.get("info")


def run_name(arm, model_seed, *, determinism=False, part=None) -> str:
    name = f"{arm}-{model_seed}" if arm in rp.CLOSED_LOOP_ARMS else arm
    if determinism:
        name += "-determinism"
    if part is not None:
        name += f".part{part[0]}of{part[1]}"
    return name


def cmd_run(args) -> int:
    report = base_report(args.cohort)
    debug = args.cohort.startswith("debug")
    try:
        rp.check_run(args.cohort, args.arm, args.model_seed)
    except Exception as error:
        raise rt.GuardError(str(error)) from error
    _lambda_check()
    goals, goals_sha = load_goals(args.goals, args.cohort, None if debug else report["revision"])
    if args.limit:
        if not debug:
            raise rt.GuardError("--limit is for debug cohorts only")
        goals = goals[: args.limit]
    if args.determinism:
        if args.cohort not in ("bs-grasp", "debug-grasp") or (
            (args.arm, args.model_seed) != rp.DETERMINISM[:2] and not debug
        ):
            raise rt.GuardError("the determinism re-run is M-proj-87100 on bs-grasp (§6.3)")
        goals = goals[: rp.DETERMINISM[2]]
    part = None
    need = rp.needs_parts(args.cohort, args.arm)
    if args.part:
        part = rp.parse_part(args.part)
        if args.determinism:
            raise rt.GuardError("the determinism re-run is one job")
        if not debug and (need is None or part[1] != need):
            raise rt.GuardError(f"{args.arm} on {args.cohort} runs as {need or 1} job(s)")
        goals = rp.split_goals(goals, *part)
    elif need is not None and not args.determinism:
        raise rt.GuardError(f"{args.arm} on {args.cohort} runs as {need} jobs: --part")
    name = run_name(args.arm, args.model_seed, determinism=args.determinism, part=part)
    if args.suffix and not debug:
        raise rt.GuardError("--suffix is for debug cohorts only")
    out = Path(args.output) / args.cohort / f"{name}{args.suffix or ''}.json"
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    report |= {
        "arm": args.arm,
        "model_seed": args.model_seed,
        "goals_sha256": goals_sha,
        "setting": rp.SETTING,
        "cem": zs.CEM,
        "lambda": zs.LAMBDA,
        "projected": bool(rp.CLOSED_LOOP_ARMS.get(args.arm, False)),
        "workers": args.workers,
        "feature_reports_sha256": rp.FEATURE_REPORT_SHA256,
        "determinism_rerun": bool(args.determinism),
        "part": list(part) if part else None,
        "plan_record": bool(args.arm in rp.CLOSED_LOOP_ARMS),
        "seeds": [g["seed"] for g in goals],
    }
    if args.arm in rp.CLOSED_LOOP_ARMS:
        report["preflight"] = preflight([args.model_seed])
    if not debug:
        ledger_claim(f"run-{args.cohort}-{name}")
    with Job(out, report, _cap(args)) as job:
        workers = min(args.workers, len(goals))
        jobs = [(args.arm, args.model_seed, goals[i::workers]) for i in range(workers)]
        with mp.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(_run_worker, [j for j in jobs if j[2]])
        episodes = sorted((e for p, _ in parts for e in p), key=lambda e: e["seed"])
        infos = [i for _, i in parts if i]
        report |= {
            "model": infos[0] if infos else None,
            "episodes": episodes,
            "successes": int(sum(e["success"] for e in episodes)),
            "n": len(episodes),
            "seconds": time.time() - job.t0,
            "completed": True,
            "cap_hit": False,
            "finished_utc": utc(),
        }
        if args.arm in rp.CLOSED_LOOP_ARMS:
            report["channel13_differs"] = int(
                sum(e["equivalence"]["channel13_differs"] for e in episodes)
            )
        write_json(out, report)
    k, n, sec = report["successes"], report["n"], report["seconds"]
    log(f"{name} on {args.cohort}: {k}/{n} in {sec:.0f} s")
    if args.arm == "M-proj" and report["channel13_differs"]:
        log("B-VOID check: an applied channel 13 differs from the projected request")
        return EXIT_VOID
    return 0


def cmd_combine(args) -> int:
    """A split run's parts combined into one run (CPU), before any summary."""
    folder = Path(args.runs) / args.cohort
    name = run_name(args.arm, args.model_seed)
    out = folder / f"{name}.json"
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    files = sorted(folder.glob(f"{name}.part*of*.json"))
    parts = [json.loads(f.read_text()) for f in files]
    if args.cohort.startswith("debug"):
        goals, _ = load_goals(args.runs, args.cohort)
        expected = [g["seed"] for g in goals][: sum(len(p["episodes"]) for p in parts)]
    else:
        expected = list(rp.COHORTS[args.cohort])
    combined = rp.combine_parts(parts, expected)
    combined |= {
        "projected": parts[0].get("projected"),
        "channel13_differs": int(sum(p.get("channel13_differs", 0) for p in parts)),
        "part_files": {f.name: sha(f) for f in files},
        "combined_utc": utc(),
        "combined_revision": git("rev-parse", "HEAD"),
    }
    write_json(out, combined)
    log(f"{name}: {len(parts)} parts, {combined['successes']}/{combined['n']}")
    return 0


def cmd_video(args) -> int:
    """TASK-092's MP4 replay (its code, copied; TASK-093's label): replay one episode's applied
    commands from its reset (checked against the run's palm distances) and write an MP4."""
    from PIL import Image, ImageDraw

    from embodied_jepa import play_corpus as pc

    base_report(gpu=True)
    result = json.loads(Path(args.result).read_text())
    goals, _ = load_goals(args.goals, result["cohort"])
    goal = next(g for g in goals if g["seed"] == args.seed)
    episode = next(e for e in result["episodes"] if e["seed"] == args.seed)
    robot = pc.make_play_robot()
    sim = robot.sim
    zs.reset(robot, goal["layout"])
    import mujoco

    big = mujoco.Renderer(sim.model, height=360, width=480)
    verdict = "SUCCESS" if episode["success"] else "FAILURE"
    name = result["arm"] + (f"-{result['model_seed']}" if result.get("model_seed") else "")
    label = f"TASK-093 grasp | arm {name} (step-mean) | reset {goal['seed']} | {verdict} (sim)"
    frames = []

    def snap(index, step):
        big.update_scene(sim.data, camera="overview")
        over = big.render()
        onboard = np.kron(sim.render(), np.ones((2, 2, 1), np.uint8)).astype(np.uint8)[:, :224]
        goal_img = np.kron(goal["subgoals"][index]["frame"], np.ones((2, 2, 1), np.uint8))
        side = np.concatenate([onboard[:180], goal_img[:180]], 0)
        canvas = np.zeros((400, 480 + 224, 3), np.uint8)
        canvas[40:400, :480] = over
        canvas[40:400, 480:] = side
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        draw.text((6, 4), label, fill=(255, 255, 255))
        draw.text(
            (6, 20),
            f"step {step}  onboard (top right), {zs.SUBGOALS[index]} subgoal image (bottom right)",
            fill=(200, 200, 200),
        )
        frames.append(np.asarray(image))

    switch = zs.SubgoalSwitch(goal["subgoals"])
    replayed, index = [], 0
    for k, action in enumerate(episode["applied"]):
        obs = robot.observe()
        p = zs.proprio(robot, obs)
        index = switch.update(p["palm"], p["state28"][7:14], p["rotation"])
        snap(index, k)
        res = robot.execute(np.asarray(action, np.float32))
        if res.status not in ("applied", "clipped"):
            log(f"replay stopped at {k}: {res.reason}")
            break
        palm, _ = robot.ee_pose("right")
        replayed.append(float(np.linalg.norm(palm - goal["subgoals"][index]["palm"])))
    snap(index, len(episode["applied"]))
    frames += [frames[-1]] * 20
    ok = np.allclose(replayed, episode["distances"][: len(replayed)], atol=1e-4)
    log(f"replay matches the run's palm distances: {ok}")
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
           f"{w}x{h}", "-r", "20", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           args.output]  # fmt: skip
    subprocess.run(cmd, input=np.stack(frames).tobytes(), check=True)
    meta = {
        "label": label,
        "result_file": args.result,
        "result_sha256": sha(args.result),
        "seed": args.seed,
        "replay_matches": bool(ok),
        "frames": len(frames),
    }
    Path(args.output + ".json").write_text(json.dumps(meta, indent=1) + "\n")
    log(f"wrote {args.output} ({len(frames)} frames)")
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
    for name in ("power", "rows"):
        po = sub.add_parser(name)
        po.add_argument("--output", required=True)
        po.add_argument("--sims", type=int, default=rp.POWER_SIMS)
        po.add_argument("--workers", type=int, default=12)
    eq = sub.add_parser("equivalence")
    eq.add_argument("--output", required=True)
    eq.add_argument("--cap-seconds", type=int)
    ro = sub.add_parser("roots")
    ro.add_argument("--cohort", required=True, choices=rp.ROOT_COHORTS)
    ro.add_argument("--output", required=True)
    ro.add_argument("--workers", type=int, default=8)
    ro.add_argument("--limit", type=int)
    ro.add_argument("--cap-seconds", type=int)
    pl = sub.add_parser("plans")
    pl.add_argument("--cohort", required=True, choices=rp.ROOT_COHORTS)
    pl.add_argument("--roots", required=True)
    pl.add_argument("--output", required=True)
    pl.add_argument("--limit", type=int)
    pl.add_argument("--stage-p-smoke", action="store_true", help="debug: Stage P's gate and row")
    pl.add_argument("--cap-seconds", type=int)
    sc = sub.add_parser("smoke-checks")
    sc.add_argument("--roots", required=True)
    sc.add_argument("--output", required=True)
    sc.add_argument("--limit", type=int, default=4)
    sc.add_argument("--cap-seconds", type=int)
    g = sub.add_parser("goals")
    g.add_argument("--cohort", required=True, choices=rp.GRASP_COHORTS)
    g.add_argument("--output", required=True)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--limit", type=int)
    g.add_argument("--cap-seconds", type=int)
    r = sub.add_parser("run")
    r.add_argument("--cohort", required=True, choices=rp.GRASP_COHORTS)
    r.add_argument("--arm", required=True, choices=sorted(rp.CLOSED_LOOP_ARMS) + list(
        rp.BASELINE_ARMS))  # fmt: skip
    r.add_argument("--model-seed", type=int)
    r.add_argument("--goals", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--part", help="i/k: job i of the run's k jobs (BS world-model runs)")
    r.add_argument("--limit", type=int)
    r.add_argument("--determinism", action="store_true")
    r.add_argument("--suffix", help="debug: an output name suffix")
    r.add_argument("--cap-seconds", type=int)
    c = sub.add_parser("combine")
    c.add_argument("--cohort", required=True, choices=rp.GRASP_COHORTS)
    c.add_argument("--arm", required=True, choices=sorted(rp.CLOSED_LOOP_ARMS))
    c.add_argument("--model-seed", type=int, required=True)
    c.add_argument("--runs", required=True)
    vi = sub.add_parser("video")
    vi.add_argument("--result", required=True)
    vi.add_argument("--goals", required=True)
    vi.add_argument("--seed", type=int, required=True)
    vi.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    return {
        "verify": cmd_verify,
        "seeds": cmd_seeds,
        "power": cmd_power,
        "rows": cmd_rows,
        "equivalence": cmd_equivalence,
        "roots": cmd_roots,
        "plans": cmd_plans,
        "smoke-checks": cmd_smoke_checks,
        "goals": cmd_goals,
        "run": cmd_run,
        "combine": cmd_combine,
        "video": cmd_video,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
