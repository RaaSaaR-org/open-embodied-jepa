"""TASK-092: Stage 0's checks, Stage R, Stage A‴ and the closed loop (protocol
docs/experiments/grasp_rate_aware_wm_v1.md). Subcommands; every GPU use is one job under
``scripts/gpu_run.sh --wait --min-free-gib 8 --board`` and stops at the 7 200 s cap:

* ``verify --output DIR``: §13's hash checks -- TASK-087's three G checkpoints and N-87100 against
  their pins, its feature reports (train/val and test), store files and projection, the moments,
  TASK-087's implementation hash, the frozen files of TASK-087, TASK-090 and TASK-091, λ (CPU).
* ``seeds --output FILE``: §9's uniqueness check at this revision, over every remote branch and
  the evidence (CPU).
* ``power --output FILE``: §6.4's power simulation with 20 000 simulations (CPU).
* ``pairing-batch --output FILE``: §13's first-batch check: with p = 0.5 the first batch of model
  seed 87900 differs from the executed one only in index 13 of at-limit steps, and with p = 0 not
  at all; the rate clip maps every drawn request back onto the executed grasp (CPU).
* ``references --output FILE``: §13's references on ``play-v1`` val (inferred clip, grasp-blind,
  R2's δ on val).
* ``stage-r --output FILE``: Stage R on ``play-v1`` test (§6.2); ``--debug-req`` runs it on val
  with a debug G-req model (Stage 0's smoke; never a result).
* ``roots``, ``a-third``: Stage A‴ (§6.3) on 908000-908063.
* ``goals``, ``run``, ``combine``, ``video``: the closed loop (§6.4).

TASK-087's, TASK-090's and TASK-091's frozen code is imported, never edited, and checked against
its pins before anything runs (``grasp_rate_aware.FROZEN_FILES_SHA256``). Non-debug stages need a
clean tracked tree; every report records the revision; every file is refused if it exists, and a
non-debug root set, goal set or run is refused if the ledger (``outputs/task092-ledger``) shows it
was started before (§7: a cohort run twice is VOID).
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
from embodied_jepa import grasp_corpus_wm_planned_close_run as prun  # noqa: E402
from embodied_jepa import grasp_rate_aware as ra  # noqa: E402
from embodied_jepa import grounded_wm as gw  # noqa: E402
from embodied_jepa import grounded_wm_data as gd  # noqa: E402
from embodied_jepa import run_tools as rt  # noqa: E402
from embodied_jepa import zero_shot as zs  # noqa: E402
from embodied_jepa import zero_shot_grasp as zg  # noqa: E402

LEDGER = ROOT / "outputs" / "task092-ledger"
EXIT_CAP = 4
EXIT_VOID = 3
DISK_FLOOR_GIB = 10.0
STAGE_R_COUNTS = {"roots": 5634, "episodes": 160, "r_roots": 821, "r_root_episodes": 154}


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
    raise CapHit(f"the per-job cap of {ra.JOB_CAP_SECONDS} s was reached")


def base_report(cohort=None, *, clean=False, gpu=True) -> dict:
    """TASK-091's report and guards (``grasp_corpus_wm_planned_close_run.base_report``) with
    TASK-092's task, protocol and frozen-file check."""
    frozen = ra.frozen_files_check(ROOT)
    if not all(frozen.values()):
        bad = [k for k, ok in frozen.items() if not ok]
        raise rt.GuardError(f"G-frozen: these frozen files differ from their pins: {bad}")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    report = {
        "task": ra.TASK,
        "protocol": ra.PROTOCOL,
        "started_utc": utc(),
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": dirty,
        "argv": sys.argv,
        "platform": platform.platform(),
        "cohort": cohort,
        "frozen_files_match": True,
        "modules_sha256": ra.own_modules_sha256(ROOT),
        "job_cap_seconds": ra.JOB_CAP_SECONDS,
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
    """A non-debug cohort's roots, goals or run may start only once (§7)."""
    LEDGER.mkdir(parents=True, exist_ok=True)
    path = LEDGER / key
    try:
        with open(path, "x") as f:
            f.write(json.dumps({"utc": utc(), "revision": git("rev-parse", "HEAD")}) + "\n")
    except FileExistsError as error:
        raise rt.GuardError(f"G-twice: {key} was started before ({path})") from error


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
    cap = getattr(args, "cap_seconds", None) or ra.JOB_CAP_SECONDS
    if cap != ra.JOB_CAP_SECONDS and not str(getattr(args, "cohort", "debug")).startswith("debug"):
        raise rt.GuardError("the per-job cap is 7 200 s outside debug cohorts")
    return int(cap)


def _manifest() -> dict:
    return json.loads((ROOT / "configs" / "g1_sim_action.json").read_text())


# ----- §13: verify -----------------------------------------------------------------------------
def cmd_verify(args) -> int:
    from embodied_jepa import grasp_corpus_wm_runtime as gr
    from embodied_jepa import grounded_wm_model as gm

    out = Path(args.output)
    path = out / "verify-task092.json"
    if path.exists():
        raise rt.GuardError(f"refusing to overwrite {path}")
    report = base_report(gpu=False)
    checks = {}
    feat = gr.V1_FEATURES
    frep = json.loads((feat / "report.json").read_text())
    checks["feature_report"] = sha(feat / "report.json") == ra.FEATURE_REPORT_SHA256
    t0 = time.monotonic()
    report["store_files"] = gd.verify_store_files(feat, frep["files_sha256"], ("train", "val"))
    trep = json.loads((feat / "report_test.json").read_text())
    checks["test_report"] = sha(feat / "report_test.json") == ra.TEST_REPORT_SHA256
    report["test_store_files"] = gd.verify_store_files(feat, trep["files_sha256"], ("test",))
    report["store_seconds"] = time.monotonic() - t0
    checks["store_files"] = True
    checks["projection_file"] = sha(feat / "projection.npz") == frep["projection"]["file_sha256"]
    gd.load_projection(feat / "projection.npz", gr.PROJECTION_SHA256)
    checks["projection_arrays"] = True
    ckpts = {}
    ev = Path(args.task087_evidence)
    for arm, seed in [("G", 87100), ("G", 87101), ("G", 87102), ("N", 87100)]:
        folder = gr.V1_CHECKPOINTS / f"{arm}-{seed}"
        rep = json.loads((folder / "report.json").read_text())
        row = {
            "model_pt": sha(folder / "model.pt"),
            "recorded": rep["checkpoint_sha256"],
            "pinned": ra.CHECKPOINT_SHA256[f"{arm}-v1-{seed}"],
            "report_equals_evidence": sha(folder / "report.json")
            == sha(ev / f"train_report_{arm}-{seed}.json"),
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
        ckpts[f"{arm}-v1-{seed}"] = row
    report["task087_checkpoints"] = ckpts
    checks["task087_checkpoints"] = all(r["ok"] for r in ckpts.values())
    rows090 = {}
    for name, pin in ra.CHECKPOINT_SHA256.items():
        if not name.startswith(("G-v2g", "G-v1p")):
            continue
        tag, seed = name.split("-")[1], int(name.split("-")[2])
        corpus = {"v2g": "play-v2g", "v1p": "play-v1p"}[tag]
        rows090[name] = sha(gr.model_dir(ROOT, "G", corpus, seed) / "model.pt") == pin
    report["task090_checkpoints"] = rows090
    checks["task090_checkpoints"] = len(rows090) == 6 and all(rows090.values())
    feats = {"play-v1": sha(feat / "report.json")}
    for corpus in gc.CORPORA:
        feats[corpus] = sha(gr.new_features(ROOT, corpus) / "report.json")
    checks["task090_feature_reports"] = feats == ra.FEATURE_REPORTS_SHA256
    report["moments"] = gr.check_pinned_moments()
    checks["moments_pin"] = report["moments"]["pinned_sha256"] == ra.MOMENTS_SHA256
    checks["implementation_hash"] = gm.implementation_sha256() == ra.IMPLEMENTATION_SHA256
    checks["lambda"] = zs.LAMBDA == ra.LAMBDA == gr.LAMBDA
    checks["rate_limit"] = abs(ra.rate_limit_from_manifest(_manifest()) - ra.RATE_LIMIT) < 1e-12
    report["frozen_files"] = ra.frozen_files_check(ROOT)
    checks["frozen_files"] = all(report["frozen_files"].values())
    g87100 = json.loads((gr.V1_CHECKPOINTS / "G-87100" / "report.json").read_text())
    report["g87100_val_at_3000"] = g87100["training"]["val_curve"][0]
    report["checks"] = checks
    report["all_pass"] = all(checks.values())
    report["finished_utc"] = utc()
    write_json(path, report)
    log(json.dumps(checks))
    return 0 if report["all_pass"] else 1


# ----- §9: the seed uniqueness check -----------------------------------------------------------
SEED_PATTERN = re.compile(
    r"(?<![0-9A-Za-z.])(90[89][0-9]{3}|90[45][0-9]{2}|900[89])(?![0-9A-Za-z])"
)
CHECK_RANGES = {
    "a3-roots": range(908000, 908064),
    "bd-grasp": range(90400, 90432),
    "bs-grasp": range(909000, 909128),
    "debug": range(90500, 90600),
    "declared span 908000-909999": range(908000, 910000),
    "declared span 90400-90599": range(90400, 90600),
    "salts 9008-9009": range(9008, 9010),
}
OWN_MARKERS = ("task092", "TASK-092", "grasp_rate_aware")


def _in_ranges(number: int) -> list[str]:
    return [k for k, r in CHECK_RANGES.items() if number in r]


def cmd_seeds(args) -> int:
    report = base_report(gpu=False)
    out = Path(args.output)
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    subprocess.run(["git", "fetch", "--quiet", "origin"], cwd=ROOT, check=False)
    branches = git("for-each-ref", "--format=%(refname:short)", "refs/remotes").split()
    branches = [b for b in branches if not b.endswith("/HEAD")]
    paths = ["src", "scripts", "docs", ".mc", "tests", "benchmarks", "configs"]
    hits = {}
    for branch in branches + ["HEAD"]:
        res = subprocess.run(
            ["git", "grep", "-n", "-I", "-E", r"90[89][0-9]{3}|90[45][0-9]{2}|900[89]", branch,
             "--"] + paths,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )  # fmt: skip
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
        own = any(m in file for m in OWN_MARKERS) or file == "docs/DECISIONS.md"
        classified["own" if own else "other"][file] = {n: sorted(b) for n, b in nums.items()}
    ev_hits = []
    exts = {".json", ".txt", ".log", ".md", ".csv", ".yaml", ".py", ".sh"}
    for path in sorted(Path(args.evidence).rglob("*")):
        if not path.is_file() or path.suffix not in exts or "task092" in str(path):
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for m in SEED_PATTERN.finditer(text):
            if _in_ranges(int(m.group(1))):
                ctx = text[max(0, m.start() - 40) : m.end() + 40].replace("\n", " ")
                ev_hits.append({"file": str(path), "token": m.group(1), "context": ctx})
    seed_fields = [h for h in ev_hits if re.search(r'"(seed|reset|root|seeds)"\s*:', h["context"])]
    report |= {
        "branches": len(branches),
        "paths": paths,
        "ranges": {k: [r.start, r.stop - 1] for k, r in CHECK_RANGES.items()},
        "hits_in_task092_files": classified["own"],
        "hits_elsewhere": classified["other"],
        "evidence_dir": str(args.evidence),
        "evidence_hits": ev_hits,
        "evidence_hits_in_seed_fields": seed_fields,
        "finished_utc": utc(),
    }
    write_json(out, report)
    log(
        f"{len(branches)} branches; hits outside TASK-092's files: {len(classified['other'])} "
        f"files; evidence tokens: {len(ev_hits)} ({len(seed_fields)} in seed fields)"
    )
    return 0


# ----- §6.4: power -----------------------------------------------------------------------------
def cmd_power(args) -> int:
    report = base_report(gpu=False)
    t0 = time.monotonic()
    table = ra.power_table(sims=args.sims)
    report |= table
    report["rule"] = ra.power_rule(table["rows"])
    report["seconds"] = time.monotonic() - t0
    report["finished_utc"] = utc()
    write_json(args.output, report)
    for r in table["rows"]:
        log(f"n {r['n']} SD {r['sd']}: power {r['power']:.4f}")
    log(json.dumps(report["rule"]))
    return 0


# ----- §13: the first-batch pairing check (CPU) ------------------------------------------------
def cmd_pairing_batch(args) -> int:
    from embodied_jepa import grasp_corpus_wm_runtime as gr

    report = base_report(gpu=False)
    feat = gr.V1_FEATURES
    if sha(feat / "report.json") != ra.FEATURE_REPORT_SHA256:
        raise rt.GuardError("G-features")
    train = gd.FeatureStore(feat, "train", mmap=True)
    out = {}
    for p in (0.0, ra.AUG_P):
        sampler = gd.WindowSampler(train, ra.PAIRING_SEED)
        aug = ra.Augmenter(ra.PAIRING_SEED, p)
        starts = sampler.draw()
        batch = train.gather(starts, gw.WINDOW_FRAMES)
        prev = ra.window_prev(starts, train.actions, train.offsets)
        a_in = aug(batch["actions"], prev)
        exe = batch["actions"]
        diff = a_in != exe
        other = diff.copy()
        other[:, :, ra.GRASP_DIM] = False
        grasp = exe[:, :, ra.GRASP_DIM].astype(np.float64)
        up, down = ra.at_limit(prev, grasp)
        previous = np.concatenate([prev[:, None], grasp[:, :-1]], 1)
        clipped = ra.rate_clip(a_in[:, :, ra.GRASP_DIM], previous)
        clip_err = np.where(np.isnan(previous), 0.0, np.abs(clipped - grasp))
        out[str(p)] = {
            "windows": int(len(starts)),
            "starts_sha256": gw.table_sha256(starts),
            "bitwise_equal": bool(np.array_equal(a_in.view(np.uint32), exe.view(np.uint32))),
            "changed_entries": int(diff.sum()),
            "changed_outside_index_13": int(other.sum()),
            "changed_steps_not_at_limit": int((diff[:, :, ra.GRASP_DIM] & ~(up | down)).sum()),
            "at_limit_steps": int((up | down).sum()),
            "windows_selected": aug.counts["windows_selected"],
            "steps_replaced": aug.counts["steps_replaced"],
            "max_clip_error_replaced": float(clip_err[diff[:, :, ra.GRASP_DIM]].max())
            if diff.any()
            else 0.0,
        }
    ok = (
        out["0.0"]["bitwise_equal"]
        and out[str(ra.AUG_P)]["changed_outside_index_13"] == 0
        and out[str(ra.AUG_P)]["changed_steps_not_at_limit"] == 0
        and out[str(ra.AUG_P)]["max_clip_error_replaced"] <= ra.AT_LIMIT_TOL
        and out["0.0"]["starts_sha256"] == out[str(ra.AUG_P)]["starts_sha256"]
    )
    # the whole train split: at-limit steps, and that no executed step exceeds the limit (§1.2)
    starts = gd.WindowSampler(train, ra.PAIRING_SEED).starts
    d = np.abs(
        np.concatenate(
            [
                np.diff(train.actions[o : o + n - 1, ra.GRASP_DIM].astype(np.float64))
                for o, n in zip(train.offsets, train.frames, strict=True)
            ]
        )
    )  # T commands per episode (its last frame has no command)
    report |= {
        "seed": ra.PAIRING_SEED,
        "batches": out,
        "pass": bool(ok),
        "train_windows": int(len(starts)),
        "train_max_abs_grasp_step": float(d.max()),
        "train_steps_at_limit_share": float((np.abs(d - ra.RATE_LIMIT) < ra.AT_LIMIT_TOL).mean()),
        "finished_utc": utc(),
    }
    write_json(args.output, report)
    log(json.dumps({"pass": ok, **{k: v["steps_replaced"] for k, v in out.items()}}))
    return 0 if ok else 1


# ----- §6.2 and §13: Stage R and the references ------------------------------------------------
def _store(split: str):
    from embodied_jepa import grasp_corpus_wm_runtime as gr

    feat = gr.V1_FEATURES
    if sha(feat / "report.json") != ra.FEATURE_REPORT_SHA256:
        raise rt.GuardError("G-features: play-v1's feature report is not TASK-087's")
    if split == "test":
        if sha(feat / "report_test.json") != ra.TEST_REPORT_SHA256:
            raise rt.GuardError("G-features: the test report is not TASK-087's")
        files = json.loads((feat / "report_test.json").read_text())["files_sha256"]
    else:
        files = json.loads((feat / "report.json").read_text())["files_sha256"]
    verified = gd.verify_store_files(feat, files, (split,))
    return gd.FeatureStore(feat, split, mmap=False), verified


def _r_inputs(store):
    roots, episode = store.roots()
    mask, prev, executed, up, down = ra.r_roots(roots, store.offsets, store.actions)
    start, commands, states, targets = store.root_arrays(roots, horizons=ra.R_HORIZONS)
    sat = ra.saturated_requests(executed[mask], up[mask], down[mask])
    rnd = ra.random_requests(executed[mask], prev[mask], up[mask], down[mask])
    checks = {
        "sat_executed_identically": ra.executed_identically(sat, executed[mask], prev[mask]),
        "rnd_executed_identically": ra.executed_identically(rnd, executed[mask], prev[mask]),
        "executed_equal_root_commands": bool(
            np.array_equal(executed.astype(np.float32), commands[:, : ra.R_HORIZON])
        ),
    }
    meas = {h: store.state[roots + h] for h in ra.R_HORIZONS}
    return {
        "roots": roots,
        "episode": episode,
        "mask": mask,
        "prev": prev,
        "start": start,
        "commands": commands,
        "states": states,
        "targets": targets,
        "sat": sat.astype(np.float32),
        "rnd": rnd.astype(np.float32),
        "meas": meas,
        "checks": checks,
    }


def _model_measures(model, x, device, *, e_wrong=None):
    """Per-root measures of one model: L_exec (all roots), H_x and L_x (R-roots), and for E
    the wrong-command errors (all roots)."""
    from embodied_jepa import grasp_rate_aware_runtime as rr

    m = x["mask"]
    hs = ra.R_HORIZONS
    lat, st = rr.rollout_at(model, x["start"], x["commands"], x["states"], hs, device=device)
    out = {"L_exec_all": {}, "H": {"exec": {}, "sat": {}, "rnd": {}}, "L_R": {}}
    for h in hs:
        out["L_exec_all"][h] = gw.latent_error(lat[h], x["targets"][h])
        meas = rr.norm_states(model, x["meas"][h][m], device)
        out["H"]["exec"][h] = ra.hand_mse(st[h][m], meas)
        out["L_R"].setdefault("exec", {})[h] = out["L_exec_all"][h][m]
    for kind in ("sat", "rnd"):
        lat_k, st_k = rr.rollout_at(
            model, x["start"][m], x[kind], x["states"][m], hs, device=device
        )
        for h in hs:
            meas = rr.norm_states(model, x["meas"][h][m], device)
            out["H"][kind][h] = ra.hand_mse(st_k[h], meas)
            out["L_R"].setdefault(kind, {})[h] = gw.latent_error(lat_k[h], x["targets"][h][m])
    if e_wrong is not None:
        lw, _ = rr.rollout_at(
            model, x["start"], x["commands"][e_wrong], x["states"], gc.E_HORIZONS, device=device
        )
        out["L_wrong_all"] = {h: gw.latent_error(lw[h], x["targets"][h]) for h in gc.E_HORIZONS}
    return out


def _ratios(meas, w_all, w_r):
    """Reported ratios for one model, per horizon."""
    rep = {}
    for h in ra.R_HORIZONS:
        H, L = meas["H"], meas["L_R"]
        rep[str(h)] = {
            "H_exec_mean": float(H["exec"][h].mean()),
            "H_sat_mean": float(H["sat"][h].mean()),
            "H_rnd_mean": float(H["rnd"][h].mean()),
            "H_sat_over_exec": ra.ratio(H["sat"][h], H["exec"][h], w_r),
            "H_rnd_over_exec": ra.ratio(H["rnd"][h], H["exec"][h], w_r),
            "L_sat_over_exec_R": ra.ratio(L["sat"][h], L["exec"][h], w_r),
            "L_rnd_over_exec_R": ra.ratio(L["rnd"][h], L["exec"][h], w_r),
            "L_exec_mean_all": float(meas["L_exec_all"][h].mean()),
        }
    return rep


def cmd_stage_r(args) -> int:
    from embodied_jepa import devices
    from embodied_jepa import grasp_rate_aware_runtime as rr

    debug = args.debug_req is not None
    split = "val" if debug else "test"
    report = base_report("debug-r" if debug else "stage-r")
    out_path = Path(args.output)
    if out_path.exists():
        raise rt.GuardError(f"refusing to overwrite {out_path}")
    report["determinism"] = devices.configure_determinism("cuda", strict=True)
    if not debug:
        ledger_claim("stage-r")
    with Job(out_path, report, ra.JOB_CAP_SECONDS) as job:
        store, verified = _store(split)
        x = _r_inputs(store)
        roots, episode, mask = x["roots"], x["episode"], x["mask"]
        counts = {
            "roots": int(len(roots)),
            "episodes": int(len(store.table)),
            "r_roots": int(mask.sum()),
            "r_root_episodes": int(len(np.unique(episode[mask]))),
        }
        void = []
        if not all(x["checks"].values()):
            void.append(f"equivalence check failed: {x['checks']}")
        if not debug and counts != STAGE_R_COUNTS:
            void.append(f"root counts {counts} are not §6.2's {STAGE_R_COUNTS}")
        wrong = gw.wrong_table(episode)
        w_all = gw.cluster_bootstrap(episode)
        w_r = gw.cluster_bootstrap(episode[mask])
        meas, infos = {}, {}
        models = [(f, s) for f, s in ra.stage_r_models()]
        for family, seed in models:
            model, info = rr.load_family(
                family, seed, "cuda", debug=debug, req_path=args.debug_req if debug else None
            )
            meas[(family, seed)] = _model_measures(model, x, "cuda", e_wrong=wrong)
            infos[ra.model_name(family, seed)] = info
            hh = meas[(family, seed)]["H"]
            log(f"{family}-{seed}: H8 sat/exec {hh['sat'][8].sum() / hh['exec'][8].sum():.3f}")
            del model
        n_model, n_info = rr.load_v1(87100, "cuda", arm="N")
        infos["N-v1-87100"] = n_info
        ln, _ = rr.rollout_at(
            n_model, x["start"], x["commands"], x["states"], gc.E_HORIZONS, device="cuda"
        )
        n_err = {h: gw.latent_error(ln[h], x["targets"][h]) for h in gc.E_HORIZONS}
        del n_model
        reported = {}
        if not args.skip_task090:
            for corpus in gc.CORPORA:
                for seed in gc.MODEL_SEEDS:
                    model, info = rr.load_task090(corpus, seed, "cuda")
                    name = f"G-{gc.CORPUS_TAG[corpus]}-{seed}"
                    reported[name] = {"model": info, "ratios": _ratios(
                        _model_measures(model, x, "cuda"), w_all, w_r)}  # fmt: skip
                    del model
        h8 = ra.R_HORIZON
        ctrl_l = {s: meas[("G-v1", s)]["L_exec_all"][h8] for s in ra.MODEL_SEEDS}
        ctrl_h = {s: meas[("G-v1", s)]["H"]["exec"][h8] for s in ra.MODEL_SEEDS}
        d_l, d_h = ra.delta(ctrl_l, w_all), ra.delta(ctrl_h, w_r)
        per_seed = {}
        copy = {h: gw.latent_error(x["start"], x["targets"][h]) for h in gc.E_HORIZONS}
        for s in ra.MODEL_SEEDS:
            q, c = meas[("G-req", s)], meas[("G-v1", s)]
            row = ra.r_seed(
                h_exec_req=q["H"]["exec"][h8], h_sat_req=q["H"]["sat"][h8],
                h_exec_ctrl=c["H"]["exec"][h8], h_sat_ctrl=c["H"]["sat"][h8],
                l_exec_req=q["L_exec_all"][h8], l_exec_ctrl=c["L_exec_all"][h8],
                w_r=w_r, w_all=w_all, delta_l=d_l["delta"], delta_h=d_h["delta"],
            )  # fmt: skip
            stats = {
                str(h): {
                    "copy_ratio": gw.ratio_stat(q["L_exec_all"][h], copy[h], w_all, gc.E_LEVEL),
                    "sens": gw.ratio_stat(q["L_wrong_all"][h], q["L_exec_all"][h], w_all,
                                          gc.E_LEVEL),
                }
                for h in gc.E_HORIZONS
            }  # fmt: skip
            vs_n = None
            if s == 87100:
                vs_n = {
                    str(h): {
                        "acc_ratio": gw.ratio_stat(q["L_exec_all"][h], n_err[h], w_all, gc.E_LEVEL)
                    }
                    for h in gc.E_HORIZONS
                }
            row["e"] = ra.e_seed(stats, vs_n)
            row["e_stats"] = stats
            row["e_vs_n"] = vs_n
            per_seed[s] = row
        ratios = {
            ra.model_name(f, s): _ratios(meas[(f, s)], w_all, w_r) for f, s in models
        }  # fmt: skip
        row = ra.r_row(per_seed, void="; ".join(void))
        arrays = {}
        for (f, s), mm in meas.items():
            for kind in ("exec", "sat", "rnd"):
                arrays[f"{f}-{s}/H_{kind}_h8"] = mm["H"][kind][h8]
            arrays[f"{f}-{s}/L_exec_all_h8"] = mm["L_exec_all"][h8]
        npz = out_path.with_suffix(".npz")
        if npz.exists():
            raise rt.GuardError(f"refusing to overwrite {npz}")
        np.savez_compressed(npz, roots=roots, episode=episode, mask=mask, **arrays)
        report |= {
            "split": split,
            "debug": debug,
            "store_files_verified": verified,
            "counts": counts,
            "equivalence_checks": x["checks"],
            "models": infos,
            "delta_L": d_l,
            "delta_H": d_h,
            "per_seed": {str(k): v for k, v in per_seed.items()},
            "ratios": ratios,
            "reported_task090": reported,
            "row": row,
            "per_root_npz": npz.name,
            "per_root_npz_sha256": sha(npz),
            "seconds": time.time() - job.t0,
            "completed": True,
            "cap_hit": False,
            "finished_utc": utc(),
        }
        write_json(out_path, report)
    log(f"Stage R ({split}): {row['row']}")
    return EXIT_VOID if row["row"] == "R-VOID" else 0


def cmd_references(args) -> int:
    """§13: on the val R-roots, for each control G-i: the bare ratio and R1's bar on val, the
    inferred-clip and grasp-blind references (H_exec, H_sat, their ratio and H_exec / H_exec of
    the bare control with intervals, against 1 + δ_H on val), and R2's δ on val; plus how well
    the inferred grasp recovers the executed one on val frames."""
    from embodied_jepa import devices
    from embodied_jepa import grasp_rate_aware_runtime as rr

    report = base_report("references")
    out_path = Path(args.output)
    if out_path.exists():
        raise rt.GuardError(f"refusing to overwrite {out_path}")
    report["determinism"] = devices.configure_determinism("cuda", strict=True)
    manifest = _manifest()
    with Job(out_path, report, ra.JOB_CAP_SECONDS) as job:
        store, verified = _store("val")
        x = _r_inputs(store)
        m, ep = x["mask"], x["episode"]
        w_all, w_r = gw.cluster_bootstrap(ep), gw.cluster_bootstrap(ep[m])
        fr = np.arange(len(store.state))
        has_prev = ~np.isin(fr, store.offsets)
        gi = rr.infer_grasp(store.state[fr[has_prev]][:, ra.HAND], manifest)
        ge = store.actions[fr[has_prev] - 1, ra.GRASP_DIM]
        err = np.abs(gi - ge)
        recovery = {
            "median_abs_error": float(np.median(err)),
            "p90_abs_error": float(np.quantile(err, 0.9)),
            "share_within_limit": float((err <= ra.RATE_LIMIT).mean()),
        }
        h = ra.R_HORIZON
        execs = x["commands"][m, :h]
        meas_l, meas_h, rows = {}, {}, {}
        for seed in ra.MODEL_SEEDS:
            model, info = rr.load_v1(seed, "cuda")
            meas = _model_measures(model, x, "cuda")
            meas_l[seed] = meas["L_exec_all"][h]
            meas_h[seed] = meas["H"]["exec"][h]
            target = rr.norm_states(model, x["meas"][h][m], "cuda")
            bare = ra.ratio(meas["H"]["sat"][h], meas["H"]["exec"][h], w_r)
            row = {"model": info, "bare": {
                "H_exec_mean": float(meas["H"]["exec"][h].mean()),
                "H_sat_mean": float(meas["H"]["sat"][h].mean()),
                "rho_ctrl": bare, "r1_bar_val": ra.r1_bar(bare["value"])}}  # fmt: skip
            for kind in ("clip", "blind"):
                hx = ra.hand_mse(
                    rr.reference_rollout(
                        model, x["start"][m], execs, x["states"][m], h, manifest, kind
                    ),
                    target,
                )
                hs = ra.hand_mse(
                    rr.reference_rollout(
                        model, x["start"][m], x["sat"], x["states"][m], h, manifest, kind
                    ),
                    target,
                )
                ratio = ra.ratio(hs, hx, w_r)  # fmt: skip
                row[kind] = {
                    "H_exec_mean": float(hx.mean()),
                    "H_sat_mean": float(hs.mean()),
                    "ratio": ratio,
                    "ratio_upper_le_r1_bar": bool(ratio["ci"][1] <= row["bare"]["r1_bar_val"]),
                    "H_exec_over_bare": ra.ratio(hx, meas["H"]["exec"][h], w_r),
                }
            rows[seed] = row
            log(f"G-{seed}: rho {bare['value']:.3f}, clip {row['clip']['ratio']['value']:.3f}, "
                f"blind/bare {row['blind']['H_exec_over_bare']['value']:.2f}")  # fmt: skip
            del model
        d_l, d_h = ra.delta(meas_l, w_all), ra.delta(meas_h, w_r)
        for row in rows.values():
            for kind in ("clip", "blind"):
                r = row[kind]["H_exec_over_bare"]
                row[kind]["r2_excludes"] = bool(r["ci"][0] > 1.0 + d_h["delta"])
        report |= {
            "split": "val",
            "store_files_verified": verified,
            "counts": {
                "roots": int(len(x["roots"])),
                "r_roots": int(m.sum()),
                "r_root_episodes": int(len(np.unique(ep[m]))),
            },  # fmt: skip
            "equivalence_checks": x["checks"],
            "inferred_grasp_recovery_val_frames": recovery,
            "delta_L_val": d_l,
            "delta_H_val": d_h,
            "controls": {str(k): v for k, v in rows.items()},
            "seconds": time.time() - job.t0,
            "completed": True,
            "cap_hit": False,
            "finished_utc": utc(),
        }
        write_json(out_path, report)
    log(f"δ_L {d_l['delta']:.4f}, δ_H {d_h['delta']:.4f}")
    return 0


# ----- §6.3: roots and Stage A‴ ----------------------------------------------------------------
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
    if args.cohort not in ra.ROOT_COHORTS:
        raise rt.GuardError(f"{args.cohort} is not a root cohort")
    report = base_report(args.cohort)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    jpath, npath = out / f"roots-{args.cohort}.json", out / f"roots-{args.cohort}.npz"
    for p in (jpath, npath):
        if p.exists():
            raise rt.GuardError(f"refusing to overwrite {p}")
    seeds = _limit(args, list(ra.COHORTS[args.cohort]))
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
            row["branches"] = {}
            for key, b in r["branches"].items():
                for k in ("actions", "frames", "states"):
                    arrays[f"{s}/{key}/{k}"] = b[k]
                row["branches"][key] = {
                    "rise": b["rise"],
                    "contact": b["contact"],
                    "max_rise": float(max(b["rise"])),
                }
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
        ra.check_seed(cohort, s)
    if not cohort.startswith("debug"):
        if seeds != list(ra.COHORTS[cohort]):
            raise rt.GuardError("G-roots: the cohort's seeds are not exactly its roots")
        if rep["revision"] != revision:
            raise rt.GuardError("G-roots: the roots were made at another revision")
    return rep, np.load(npath), sha(folder / f"roots-{cohort}.json")


def _lambda_check():
    from embodied_jepa import grasp_corpus_wm_runtime as gr

    if not (zs.LAMBDA == gr.LAMBDA == ra.LAMBDA) or ra.A3_SETTING != "step-mean":
        raise rt.GuardError("G-lambda: λ or the A2 setting is not TASK-089's")


def _predicted_grasp(model, enc_root, plan, state, manifest):
    """§6.3, reported: the model's predicted hand synergy at plan steps 1-8 under the plan."""
    from embodied_jepa import grasp_corpus_wm_runtime as gr
    from embodied_jepa import grasp_rate_aware_runtime as rr

    _, ps = gr.rollout_numpy(model, enc_root[None], np.asarray(plan, np.float32)[None],
                             np.asarray(state, np.float32)[None])  # fmt: skip
    raw = ps[0] * model.state_scale.cpu().numpy() + model.state_mean.cpu().numpy()
    return [float(v) for v in rr.infer_grasp(raw[:, ra.HAND], manifest)]


def cmd_a_third(args) -> int:
    """§6.3: TASK-091's Stage A″ for the six models of §5 (A1 at h = 8 and 16 with G's cost,
    the action-shuffled control with salt 9002, the readability probe with salt 9004, latent-only
    wins, close≤4@R1 and lift@R2 from the whole A2-seed plan, every plan's first command checked
    bitwise against ``first_planned``), plus the plan's executed grasp beside the predicted hand
    synergy (reported)."""
    from embodied_jepa import devices
    from embodied_jepa import grasp_corpus_wm_planned_close_runtime as pr
    from embodied_jepa import grasp_corpus_wm_runtime as gr
    from embodied_jepa import grasp_rate_aware_runtime as rr
    from embodied_jepa import zero_shot_grasp_runtime as zgr

    if args.cohort not in ra.ROOT_COHORTS:
        raise rt.GuardError(f"{args.cohort} is not a root cohort")
    debug = args.cohort.startswith("debug")
    if args.debug_req and not debug:
        raise rt.GuardError("--debug-req is for debug cohorts only")
    report = base_report(args.cohort)
    out = Path(args.output) / f"a-third-{args.cohort}.json"
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    _lambda_check()
    bars = gc.A1_BARS
    if bars is None:
        raise rt.GuardError("G-k0: A-K0's frozen A1_BARS are missing")
    report["a1_bars"] = bars
    manifest = _manifest()
    if not debug:
        ledger_claim(f"a-third-{args.cohort}")
    with Job(out, report, _cap(args)):
        devices.configure_determinism("cuda", strict=False)
        rep, data, roots_sha = load_roots(args.roots, args.cohort, report["revision"])
        if args.limit:
            if not debug:
                raise rt.GuardError("--limit is for debug cohorts only")
            rep = dict(rep, roots=rep["roots"][: args.limit])
        enc = prun.encoded(gr.encoder("cuda"), data, rep)
        pms, _ = prun.pin_model_pms()
        report["ceiling"] = prun.ceiling(pms, enc, data, rep)
        seeds_r = [r["seed"] for r in rep["roots"]]
        folds = gc.probe_folds(np.repeat(seeds_r, 5))
        branches = [("R1", "close"), ("R1", "open"), ("R2", "lift"), ("R2", "release"),
                    ("R2", "stay")]  # fmt: skip
        labels, ceil_x, floor_x = {}, {}, {}
        pin_model = pms["latent+pose"].model
        for h in zg.HORIZONS:
            y, cx, fx = [], [], []
            for r in rep["roots"]:
                s = r["seed"]
                for root, name in branches:
                    b = r["branches"][f"{root}/{name}"]
                    y.append(bool(b["rise"][h - 1] >= gc.LIFT_RISE and b["contact"][h - 1]))
                    st = gr.norm_state(pin_model, data[f"{s}/{root}/{name}/states"][h - 1][None])
                    cx.append(np.r_[enc[f"{s}/{root}/{name}"][h - 1].ravel(), st[0]])
                    fx.append(st[0])
            labels[h], ceil_x[h], floor_x[h] = np.array(y), np.array(cx), np.array(fx)
        probe = {}
        for h in zg.HORIZONS:
            roots_of = np.repeat(seeds_r, len(branches))
            probe[str(h)] = {
                "ceiling": gc.auroc_interval(
                    gc.probe_scores(ceil_x[h], labels[h], folds), labels[h], roots_of
                ),
                "proprio_floor": gc.auroc_interval(
                    gc.probe_scores(floor_x[h], labels[h], folds), labels[h], roots_of
                ),
            }
        family_bars = bars["latent+pose"][str(ra.A3_H)]
        results, mismatches = {}, []
        for family, seed in ra.a3_models():
            model, info = rr.load_family(
                family, seed, "cuda", debug=debug,
                req_path=args.debug_req if (debug and family == "G-req") else None,
            )  # fmt: skip
            name = ra.model_name(family, seed)
            fams = {
                "latent": gr.planner_model(model, w_pose=0.0),
                "latent+pose": gr.planner_model(model, w_pose=1.0),
            }
            res = {"model": info}
            for fam, pm in fams.items():
                w = {
                    k: {
                        str(h): {zg.comparison_name(c): [] for c in zg.COMPARISONS}
                        for h in zg.HORIZONS
                    }
                    for k in ("model", "shuffled")
                }
                for r in rep["roots"]:
                    s = r["seed"]
                    for ci, comp in enumerate(zg.COMPARISONS):
                        root, t, a = comp
                        rl, rs = enc[f"{s}/{root}"], data[f"{s}/{root}/state28"]
                        gl = enc[f"{s}/goal/{root}"]
                        gs = data[f"{s}/subgoal{zg.ROOT_GOAL[root]}/state28"]
                        ta, aa = data[f"{s}/{root}/{t}/actions"], data[f"{s}/{root}/{a}/actions"]
                        d1, d2 = prun.shuffled_deal(s, ci, ta, aa)
                        costs = zgr.rollout_costs(pm, rl, rs, gl, gs, np.stack([ta, aa, d1, d2]))
                        c = zg.comparison_name(comp)
                        for h in zg.HORIZONS:
                            w["model"][str(h)][c].append(
                                bool(zg.wins(costs[0, h - 1], costs[1, h - 1]))
                            )
                            w["shuffled"][str(h)][c].append(
                                bool(zg.wins(costs[2, h - 1], costs[3, h - 1]))
                            )
                res[fam] = w
            n = len(rep["roots"])
            res["a1_h8"] = zg.a1_pass(
                family_bars,
                res["latent+pose"]["model"][str(ra.A3_H)],
                res["latent+pose"]["shuffled"][str(ra.A3_H)],
                n,
            )
            plans = {root: [] for root in zg.ROOTS}
            executed = {root: [] for root in zg.ROOTS}
            predicted = {root: [] for root in zg.ROOTS}
            pm = fams["latent+pose"]
            for r in rep["roots"]:
                s = r["seed"]
                for root in zg.ROOTS:
                    xin = (
                        enc[f"{s}/{root}"],
                        data[f"{s}/{root}/state28"],
                        enc[f"{s}/goal/{root}"],
                        data[f"{s}/subgoal{zg.ROOT_GOAL[root]}/state28"],
                    )
                    plan = pr.a2_plan(pm, ra.A3_SETTING, s, *xin)
                    first = zgr.first_planned(pm, ra.A3_SETTING, s, *xin)
                    if not ra.first_command_equal(plan, first):
                        mismatches.append([name, s, root])
                    plans[root].append(plan.tolist())
                    executed[root].append(ra.executed_plan_grasp(plan, ra.ROOT_GRASP[root]))
                    predicted[root].append(_predicted_grasp(model, xin[0], plan, xin[1], manifest))
            res["a2"] = ra.a2_third(plans["R1"], plans["R2"]) | {
                "plans_R1": plans["R1"],
                "plans_R2": plans["R2"],
            }
            res["a3_pass"] = ra.a3_pass(res["a1_h8"], res["a2"])
            res["executed_plan_grasp"] = executed
            res["predicted_hand_synergy"] = predicted
            res["probe"] = {}
            for h in zg.HORIZONS:
                px = []
                for r in rep["roots"]:
                    s = r["seed"]
                    for root, bname in branches:
                        pz, ps = gr.rollout_numpy(
                            model,
                            enc[f"{s}/{root}"][None],
                            data[f"{s}/{root}/{bname}/actions"][None],
                            data[f"{s}/{root}/state28"][None],
                        )
                        px.append(np.r_[pz[0, h - 1].ravel(), ps[0, h - 1]])
                px = np.array(px)
                roots_of = np.repeat(seeds_r, len(branches))
                res["probe"][str(h)] = gc.auroc_interval(
                    gc.probe_scores(px, labels[h], folds), labels[h], roots_of
                )
            a2 = res["a2"]
            log(
                f"{name}: A1 h8 pass {res['a1_h8']['pass']}, close≤4@R1 "
                f"{a2['close_le4_R1']['count']}, lift@R2 {a2['lift_R2']['count']}, "
                f"A‴ {res['a3_pass']}"
            )
            results[name] = res
            del model
            if mismatches:
                break  # A″'s rule: stop at the first mismatch
        report |= {
            "roots_sha256": roots_sha,
            "n": len(rep["roots"]),
            "results": results,
            "probe": probe,
            "probe_positives": {str(h): int(labels[h].sum()) for h in zg.HORIZONS},
            "lift_branch_rise_ge5_16": sum(
                max(r["branches"]["R2/lift"]["rise"]) >= gc.REPORT_RISE for r in rep["roots"]
            ),
            "first_command_mismatches": mismatches,
            "a3_pass_req": {
                str(s): results.get(ra.model_name("G-req", s), {}).get("a3_pass")
                for s in ra.MODEL_SEEDS
            },
            "completed": True,
            "cap_hit": False,
            "finished_utc": utc(),
        }
        write_json(out, report)
    log(f"Stage A‴: {report['a3_pass_req']}")
    return EXIT_VOID if mismatches else 0


# ----- §6.4: goals, runs, combine, video -------------------------------------------------------
def _goal_worker(seeds):
    from embodied_jepa import play_corpus as pc

    robot = pc.make_play_robot()
    out = [zs.make_grasp_goal(robot, s) for s in seeds]
    robot.close()
    return out


def cmd_goals(args) -> int:
    if args.cohort not in ra.GRASP_COHORTS:
        raise rt.GuardError(f"{args.cohort} is not a closed-loop cohort")
    report = base_report(args.cohort)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    jpath, npath = out / f"goals-{args.cohort}.json", out / f"goals-{args.cohort}.npz"
    for p in (jpath, npath):
        if p.exists():
            raise rt.GuardError(f"refusing to overwrite {p}")
    seeds = _limit(args, list(ra.COHORTS[args.cohort]))
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
            ra.check_seed(cohort, row["seed"])
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
        if [g["seed"] for g in goals] != list(ra.COHORTS[cohort]):
            raise rt.GuardError("G-goals: the cohort's seeds are not exactly its goals")
        if revision is not None and rep["revision"] != revision:
            raise rt.GuardError("G-goals: the goals were made at another revision")
    return goals, sha(folder / f"goals-{cohort}.json")


def _controller(arm, model_seed, episode_seed, cache, debug_req):
    if arm == "hold":
        return zs.HoldController()
    if arm == "random":
        return zs.RandomController(episode_seed)
    if arm == "ik":
        return zs.IKController()
    if arm == "ik-nopress":
        return zs.IKController(press=False)
    from embodied_jepa import grasp_corpus_wm_planned_close_runtime as pr
    from embodied_jepa import grasp_corpus_wm_runtime as gr
    from embodied_jepa import grasp_rate_aware_runtime as rr

    if "model" not in cache:
        from embodied_jepa import devices

        devices.configure_determinism("cuda", strict=False)
        family = ra.CLOSED_LOOP_ARMS[arm]
        debug = debug_req is not None and family == "G-req"
        cache["model"], cache["info"] = rr.load_family(
            family, model_seed, "cuda", debug=debug, req_path=debug_req if debug else None
        )
        cache["encoder"] = gr.encoder("cuda")
    return pr.controller(arm, cache["model"], cache["encoder"], episode_seed, record=True)


def _run_worker(job):
    arm, model_seed, goals, debug_req = job
    from embodied_jepa import play_corpus as pc
    from embodied_jepa import zero_shot_runtime as zr

    robot = pc.make_play_robot()
    cache, out = {}, []
    for goal in goals:
        controller = _controller(arm, model_seed, goal["seed"], cache, debug_req)
        result = zr.run_episode(robot, goal, controller)
        result["goal_digest"] = goal["digest"]
        if hasattr(controller, "setting"):
            result["setting"] = controller.setting
        if hasattr(controller, "plan_log"):
            result["plan_log"] = controller.plan_log
            result["deferral"] = ra.deferral_measures(result)
            result["rate"] = ra.rate_measures(result)
        out.append(result)
        log(
            f"{arm} {goal['seed']} success={result['success']} steps={result['steps']} "
            f"{result['stop_reason']}"
        )
    robot.close()
    return out, cache.get("info")


def run_name(arm, model_seed, *, determinism=False, part=None) -> str:
    name = f"{arm}-{model_seed}" if arm in ra.CLOSED_LOOP_ARMS else arm
    if determinism:
        name += "-determinism"
    if part is not None:
        name += f".part{part[0]}of{part[1]}"
    return name


def cmd_run(args) -> int:
    report = base_report(args.cohort)
    debug = args.cohort.startswith("debug")
    if args.debug_req and not debug:
        raise rt.GuardError("--debug-req is for debug cohorts only")
    debug_seed = None
    if args.debug_req:
        debug_seed = int(json.loads((Path(args.debug_req) / "report.json").read_text())["seed"])
        ra.check_model_seed(debug_seed, debug=True)
    try:
        ra.check_run(args.cohort, args.arm, args.model_seed, debug_req_seed=debug_seed)
    except Exception as error:
        raise rt.GuardError(str(error)) from error
    debug_req = args.debug_req if (args.arm == "M-req" and args.model_seed == debug_seed) else None
    _lambda_check()
    goals, goals_sha = load_goals(args.goals, args.cohort, None if debug else report["revision"])
    if args.limit:
        if not debug:
            raise rt.GuardError("--limit is for debug cohorts only")
        goals = goals[: args.limit]
    if args.determinism:
        if args.cohort not in ("bs-grasp", "debug-grasp") or (
            (args.arm, args.model_seed) != ra.DETERMINISM[:2] and not debug
        ):
            raise rt.GuardError("the determinism re-run is M-req-87100 on bs-grasp (§6.4)")
        goals = goals[: ra.DETERMINISM[2]]
    part = None
    need = ra.needs_parts(args.cohort, args.arm)
    if args.part:
        part = ra.parse_part(args.part)
        if args.determinism:
            raise rt.GuardError("the determinism re-run is one job")
        if not debug and (need is None or part[1] != need):
            raise rt.GuardError(f"{args.arm} on {args.cohort} runs as {need or 1} job(s)")
        goals = ra.split_goals(goals, *part)
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
        "setting": "step-mean",
        "cem": zs.CEM,
        "lambda": zs.LAMBDA,
        "workers": args.workers,
        "feature_reports_sha256": ra.FEATURE_REPORT_SHA256,
        "determinism_rerun": bool(args.determinism),
        "part": list(part) if part else None,
        "plan_record": bool(args.arm in ra.CLOSED_LOOP_ARMS),
        "debug_req": str(args.debug_req) if debug_req else None,
        "seeds": [g["seed"] for g in goals],
    }
    if not debug:
        ledger_claim(f"run-{args.cohort}-{name}")
    with Job(out, report, _cap(args)) as job:
        workers = min(args.workers, len(goals))
        jobs = [(args.arm, args.model_seed, goals[i::workers], debug_req) for i in range(workers)]
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
        write_json(out, report)
    k, n, sec = report["successes"], report["n"], report["seconds"]
    log(f"{name} on {args.cohort}: {k}/{n} in {sec:.0f} s")
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
        expected = list(ra.COHORTS[args.cohort])
    combined = ra.combine_parts(parts, expected)
    combined |= {
        "part_files": {f.name: sha(f) for f in files},
        "combined_utc": utc(),
        "combined_revision": git("rev-parse", "HEAD"),
    }
    write_json(out, combined)
    log(f"{name}: {len(parts)} parts, {combined['successes']}/{combined['n']}")
    return 0


def cmd_video(args) -> int:
    """TASK-091's MP4 replay (its code, copied; TASK-092's label): replay one episode's applied
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
    label = f"TASK-092 grasp | arm {name} (step-mean) | reset {goal['seed']} | {verdict} (sim)"
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
    po = sub.add_parser("power")
    po.add_argument("--output", required=True)
    po.add_argument("--sims", type=int, default=ra.POWER_SIMS)
    pb = sub.add_parser("pairing-batch")
    pb.add_argument("--output", required=True)
    rf = sub.add_parser("references")
    rf.add_argument("--output", required=True)
    sr = sub.add_parser("stage-r")
    sr.add_argument("--output", required=True)
    sr.add_argument("--debug-req", help="debug: a debug G-req run (val split; never a result)")
    sr.add_argument("--skip-task090", action="store_true", help="debug only")
    ro = sub.add_parser("roots")
    ro.add_argument("--cohort", required=True, choices=ra.ROOT_COHORTS)
    ro.add_argument("--output", required=True)
    ro.add_argument("--workers", type=int, default=8)
    ro.add_argument("--limit", type=int)
    ro.add_argument("--cap-seconds", type=int)
    a = sub.add_parser("a-third")
    a.add_argument("--cohort", required=True, choices=ra.ROOT_COHORTS)
    a.add_argument("--roots", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--limit", type=int)
    a.add_argument("--debug-req")
    a.add_argument("--cap-seconds", type=int)
    g = sub.add_parser("goals")
    g.add_argument("--cohort", required=True, choices=ra.GRASP_COHORTS)
    g.add_argument("--output", required=True)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--limit", type=int)
    g.add_argument("--cap-seconds", type=int)
    r = sub.add_parser("run")
    r.add_argument("--cohort", required=True, choices=ra.GRASP_COHORTS)
    r.add_argument("--arm", required=True, choices=sorted(ra.CLOSED_LOOP_ARMS) + list(
        ra.BASELINE_ARMS))  # fmt: skip
    r.add_argument("--model-seed", type=int)
    r.add_argument("--goals", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--part", help="i/k: job i of the run's k jobs (BS world-model runs)")
    r.add_argument("--limit", type=int)
    r.add_argument("--determinism", action="store_true")
    r.add_argument("--debug-req", help="debug: a debug G-req run for M-req")
    r.add_argument("--suffix", help="debug: an output name suffix")
    r.add_argument("--cap-seconds", type=int)
    c = sub.add_parser("combine")
    c.add_argument("--cohort", required=True, choices=ra.GRASP_COHORTS)
    c.add_argument("--arm", required=True, choices=sorted(ra.CLOSED_LOOP_ARMS))
    c.add_argument("--model-seed", type=int, required=True)
    c.add_argument("--runs", required=True)
    vi = sub.add_parser("video")
    vi.add_argument("--result", required=True)
    vi.add_argument("--goals", required=True)
    vi.add_argument("--seed", type=int, required=True)
    vi.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if getattr(args, "skip_task090", False) and not getattr(args, "debug_req", None):
        raise SystemExit("--skip-task090 is debug-only")
    return {
        "verify": cmd_verify,
        "seeds": cmd_seeds,
        "power": cmd_power,
        "pairing-batch": cmd_pairing_batch,
        "references": cmd_references,
        "stage-r": cmd_stage_r,
        "roots": cmd_roots,
        "a-third": cmd_a_third,
        "goals": cmd_goals,
        "run": cmd_run,
        "combine": cmd_combine,
        "video": cmd_video,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
