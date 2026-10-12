"""TASK-091: Stage 0's checks, Stage A″ and the closed loop (protocol
docs/experiments/grasp_play_corpus_v1_planned_close.md). Subcommands; every GPU use is one job
under ``scripts/gpu_run.sh --wait --min-free-gib 8 --board`` and stops at the 7 200 s cap (§5.4):

* ``verify --output DIR``: §12's hash checks -- TASK-090's own check (``run_task090.py verify``:
  the data-disk links, ``play-v1``, TASK-087's feature report, store files, projection, 18
  checkpoints and the pinned moments) plus TASK-090's eight checkpoints against §19.2's pins and
  its evidence reports, the three feature reports, TASK-087's G pins, λ and the frozen files (CPU).
* ``seeds --output FILE``: §8's uniqueness check at this revision, over every remote branch and
  the evidence (CPU).
* ``roots --cohort C --output DIR``: A″'s roots (TASK-089's sampler, ``zg.build_root``).
* ``a-second --cohort C --roots DIR --output DIR``: Stage A″ (§4) for every model of §3.
* ``goals --cohort C --output DIR``; ``run --cohort C --arm A [--model-seed S] [--part i/k]``;
  ``combine --cohort C --arm A --model-seed S --runs DIR``;
  ``video --result FILE --goals DIR --seed N --output FILE.mp4``: the closed loop (§5).

TASK-090's frozen runner and modules are imported, never edited, and checked against their pins
before anything runs. Non-debug cohorts need a clean tracked tree; every report records the
revision; every file is refused if it exists, and a non-debug cohort's roots, goals or run is
refused if the ledger (``outputs/task091-ledger``) shows it was started before (§6: a root set or
cohort run twice is VOID).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import grasp_corpus_wm as gc  # noqa: E402
from embodied_jepa import grasp_corpus_wm_planned_close as pcl  # noqa: E402
from embodied_jepa import run_tools as rt  # noqa: E402
from embodied_jepa import zero_shot as zs  # noqa: E402
from embodied_jepa import zero_shot_grasp as zg  # noqa: E402


def _frozen_runner():
    """TASK-090's frozen ``scripts/run_task090.py`` of this checkout, as a module (its guards,
    report and helpers); checked against its pin first."""
    bad = [k for k, ok in pcl.frozen_files_check(ROOT).items() if not ok]
    if bad:
        raise rt.GuardError(f"G-frozen: these frozen files differ from their pins: {bad}")
    spec = importlib.util.spec_from_file_location(
        "run_task090_frozen", ROOT / "scripts" / "run_task090.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r090 = _frozen_runner()
utc, log, sha, git = r090.utc, r090.log, r090.sha, r090.git
LEDGER = ROOT / "outputs" / "task091-ledger"
EXIT_CAP = 4
EXIT_VOID = 3


class CapHit(Exception):
    pass


def _alarm(signum, frame):
    raise CapHit(f"the per-job cap of {pcl.JOB_CAP_SECONDS} s was reached")


def base_report(cohort=None, *, clean=False, gpu=True) -> dict:
    """TASK-090's ``base_report`` (revision, clean-tree, local-import, disk and GPU-lock guards)
    with TASK-091's task, protocol and frozen-file check."""
    report = r090.base_report(cohort, clean=clean, gpu=gpu)
    report |= {
        "task": pcl.TASK,
        "protocol": pcl.PROTOCOL,
        "frozen_files_match": all(pcl.frozen_files_check(ROOT).values()),
        "job_cap_seconds": pcl.JOB_CAP_SECONDS,
    }
    return report


def write_json(path, value) -> None:
    r090.write_json(path, value)


def ledger_claim(key: str) -> None:
    """A non-debug cohort's roots, goals or run may start only once (§6)."""
    LEDGER.mkdir(parents=True, exist_ok=True)
    path = LEDGER / key
    try:
        with open(path, "x") as f:
            f.write(json.dumps({"utc": utc(), "revision": git("rev-parse", "HEAD")}) + "\n")
    except FileExistsError as error:
        raise rt.GuardError(f"G-twice: {key} was started before ({path})") from error


class Job:
    """One capped job: the report it writes on a cap hit or a guard failure (never over an
    existing file)."""

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
    cap = getattr(args, "cap_seconds", None) or pcl.JOB_CAP_SECONDS
    if cap != pcl.JOB_CAP_SECONDS and not str(getattr(args, "cohort", "debug")).startswith("debug"):
        raise rt.GuardError("the per-job cap is 7 200 s outside debug cohorts (§5.4)")
    return int(cap)


# ----- §12: verify -----------------------------------------------------------------------------
def cmd_verify(args) -> int:
    from embodied_jepa import grasp_corpus_wm_runtime as gr
    from embodied_jepa import grounded_wm_model as gm

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "verify-task091.json"
    if path.exists():
        raise rt.GuardError(f"refusing to overwrite {path}")
    report = base_report(gpu=False)
    # TASK-090's own check, unchanged (it writes its own report)
    t0 = time.monotonic()
    code = r090.cmd_verify(
        argparse.Namespace(
            output=str(out / "verify-task090-checks.json"),
            main_checkout=args.main_checkout,
            task087_evidence=args.task087_evidence,
        )
    )
    report["task090_verify"] = {
        "exit": code,
        "file": "verify-task090-checks.json",
        "sha256": sha(out / "verify-task090-checks.json"),
        "seconds": time.monotonic() - t0,
    }
    checks = {"task090_verify": code == 0}
    # TASK-090's eight checkpoints: model.pt, its report and §19.2's pins, the evidence copy
    ev = Path(args.task090_evidence)
    rows = {}
    for arm, corpus, seed in [m for m in pcl.a_second_models() if m[1] != "play-v1"]:
        name = pcl.model_name(arm, corpus, seed)
        folder = gr.model_dir(ROOT, arm, corpus, seed)
        rep = json.loads((folder / "report.json").read_text())
        row = {
            "model_pt": sha(folder / "model.pt"),
            "recorded": rep["checkpoint_sha256"],
            "pinned": pcl.CHECKPOINT_SHA256[name],
            "report_equals_evidence": sha(folder / "report.json")
            == sha(ev / "t" / name / "report.json"),
            "feature_reports": rep["feature_reports_sha256"] == pcl.FEATURE_REPORTS_SHA256,
            "moments": rep["moments"]["pinned_sha256"] == pcl.MOMENTS_SHA256,
            "last_two_triggered": rep["training"]["last_two_triggered"],
        }
        row["ok"] = (
            row["model_pt"] == row["recorded"] == row["pinned"]
            and row["report_equals_evidence"]
            and row["feature_reports"]
            and row["moments"]
        )
        rows[name] = row
    report["task090_checkpoints"] = rows
    checks["task090_checkpoints"] = len(rows) == 8 and all(r["ok"] for r in rows.values())
    checks["last_two_triggered"] = sorted(
        k for k, r in rows.items() if r["last_two_triggered"]
    ) == sorted(pcl.LAST_TWO_TRIGGERED)
    # TASK-087's three G checkpoints against the pins
    v1 = {}
    for seed in (87100, 87101, 87102):
        folder = gr.V1_CHECKPOINTS / f"G-{seed}"
        v1[f"G-v1-{seed}"] = sha(folder / "model.pt") == pcl.CHECKPOINT_SHA256[f"G-v1-{seed}"]
    report["task087_checkpoints"] = v1
    checks["task087_checkpoints"] = all(v1.values())
    # the three feature reports (§3) and the projection
    feats = {"play-v1": sha(gr.V1_FEATURES / "report.json")}
    for corpus in gc.CORPORA:
        feats[corpus] = sha(gr.new_features(ROOT, corpus) / "report.json")
    report["feature_reports_sha256"] = feats
    checks["feature_reports"] = feats == pcl.FEATURE_REPORTS_SHA256
    checks["moments_pin"] = gr.check_pinned_moments()["pinned_sha256"] == pcl.MOMENTS_SHA256
    checks["implementation_hash"] = gm.implementation_sha256() == gr.IMPLEMENTATION_SHA256
    checks["lambda"] = zs.LAMBDA == pcl.LAMBDA == gr.LAMBDA
    report["frozen_files"] = pcl.frozen_files_check(ROOT)
    checks["frozen_files"] = all(report["frozen_files"].values())
    report["checks"] = checks
    report["all_pass"] = all(checks.values())
    report["finished_utc"] = utc()
    write_json(path, report)
    log(json.dumps(checks))
    return 0 if report["all_pass"] else 1


# ----- §8: the seed uniqueness check -----------------------------------------------------------
SEED_PATTERN = re.compile(r"(?<![0-9A-Za-z.])(90[37][0-9]{2}|90[67][0-9]{3})(?![0-9A-Za-z])")
CHECK_RANGES = {
    "a2-roots": range(906000, 906064),
    "bd-grasp": range(90300, 90332),
    "bs-grasp": range(907000, 907128),
    "debug": range(90700, 90800),
}
OWN_FILES = (
    "docs/experiments/grasp_play_corpus_v1_planned_close.md",
    "src/embodied_jepa/grasp_corpus_wm_planned_close.py",
    "src/embodied_jepa/grasp_corpus_wm_planned_close_runtime.py",
    "scripts/run_task091.py",
    "scripts/summarize_task091.py",
    "scripts/run_task091_stage.sh",
    "tests/test_grasp_corpus_planned_close.py",
)


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
            ["git", "grep", "-n", "-I", "-E", r"90[37][0-9]{2}|90[67][0-9]{3}", branch, "--"]
            + paths,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        for line in res.stdout.splitlines():
            parts = line.split(":", 3)
            if len(parts) < 4:
                continue
            _, file, lineno, text = parts
            for m in SEED_PATTERN.finditer(text):
                where = _in_ranges(int(m.group(1)))
                if where:
                    hits.setdefault(file, {}).setdefault(m.group(1), set()).add(branch)
    classified = {"own": {}, "other": {}}
    for file, nums in hits.items():
        own = file in OWN_FILES or "TASK-091" in file or file == "docs/DECISIONS.md"
        kind = "own" if own else "other"
        classified[kind][file] = {n: sorted(b) for n, b in nums.items()}
    # the evidence: tokens in the ranges, with their context, for reading
    ev_hits = []
    ev = Path(args.evidence)
    exts = {".json", ".txt", ".log", ".md", ".csv", ".yaml", ".py", ".sh"}
    for path in sorted(ev.rglob("*")):
        if not path.is_file() or path.suffix not in exts or "task091" in str(path):
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for m in SEED_PATTERN.finditer(text):
            if _in_ranges(int(m.group(1))):
                ctx = text[max(0, m.start() - 40) : m.end() + 40].replace("\n", " ")
                ev_hits.append({"file": str(path), "token": m.group(1), "context": ctx})
    report |= {
        "branches": len(branches),
        "paths": paths,
        "ranges": {k: [r.start, r.stop - 1] for k, r in CHECK_RANGES.items()},
        "hits_in_task091_files": classified["own"],
        "hits_elsewhere": classified["other"],
        "evidence_dir": str(ev),
        "evidence_hits": ev_hits,
        "finished_utc": utc(),
    }
    write_json(out, report)
    log(
        f"{len(branches)} branches; hits outside TASK-091's files: {len(classified['other'])} "
        f"files; evidence tokens: {len(ev_hits)}"
    )
    return 0


# ----- §4: roots and Stage A″ ------------------------------------------------------------------
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
    if args.cohort not in pcl.ROOT_COHORTS:
        raise rt.GuardError(f"{args.cohort} is not a root cohort")
    report = base_report(args.cohort)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    jpath, npath = out / f"roots-{args.cohort}.json", out / f"roots-{args.cohort}.npz"
    for p in (jpath, npath):
        if p.exists():
            raise rt.GuardError(f"refusing to overwrite {p}")
    seeds = _limit(args, list(pcl.COHORTS[args.cohort]))
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
        pcl.check_seed(cohort, s)
    if not cohort.startswith("debug"):
        if seeds != list(pcl.COHORTS[cohort]):
            raise rt.GuardError("G-roots: the cohort's seeds are not exactly its roots")
        if rep["revision"] != revision:
            raise rt.GuardError("G-roots: the roots were made at another revision")
    return rep, np.load(npath), sha(folder / f"roots-{cohort}.json")


def _feature_pins() -> dict:
    """The real feature reports (every TASK-091 cohort, debug included, uses the real models)."""
    pins = r090._feature_pins(None)
    if pins != pcl.FEATURE_REPORTS_SHA256:
        raise rt.GuardError(f"G-features: feature reports {pins} are not the pinned ones")
    return pins


def load_model(arm, corpus, seed, pins):
    """A checkpoint of §3 under TASK-090's loader guards, plus its §19.2 / TASK-087 pin."""
    from embodied_jepa import grasp_corpus_wm_runtime as gr

    if corpus == "play-v1":
        model, info = gr.load_v1_model(seed, "cuda", arm=arm)
    else:
        model, info = gr.load_new_model(
            gr.model_dir(ROOT, arm, corpus, seed), "cuda", feature_reports=pins
        )
    name = pcl.model_name(arm, corpus, seed)
    if info["checkpoint_sha256"] != pcl.CHECKPOINT_SHA256[name]:
        raise rt.GuardError(f"G-checkpoint: {name} is not the pinned checkpoint")
    if info["moments_sha256"] != pcl.MOMENTS_SHA256:
        raise rt.GuardError(f"G-moments: {name}")
    return model, info


def _lambda_check():
    from embodied_jepa import grasp_corpus_wm_runtime as gr

    if not (zs.LAMBDA == gr.LAMBDA == pcl.LAMBDA) or gc.A_PRIME_SETTING != "step-mean":
        raise rt.GuardError("G-lambda: λ or the A2 setting is not TASK-090's")


def cmd_a_second(args) -> int:
    """§4: TASK-090 §6.2's Stage A′ (A1 at h = 8 and 16 with G's cost, the action-shuffled
    control with salt 9002, the readability probe with salt 9004, latent-only wins, the lift
    branch's rise, the ceiling on these roots), with A2's close bar read from the whole A2-seed
    plan (close≤4@R1) and every plan's first command checked bitwise against ``first_planned``."""
    from embodied_jepa import devices
    from embodied_jepa import grasp_corpus_wm_planned_close_runtime as pr
    from embodied_jepa import grasp_corpus_wm_runtime as gr
    from embodied_jepa import zero_shot_grasp_runtime as zgr

    if args.cohort not in pcl.ROOT_COHORTS:
        raise rt.GuardError(f"{args.cohort} is not a root cohort")
    report = base_report(args.cohort)
    out = Path(args.output) / f"a-second-{args.cohort}.json"
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    _lambda_check()
    bars = gc.A1_BARS
    if bars is None:
        raise rt.GuardError("G-k0: A-K0's frozen A1_BARS are missing")
    report["a1_bars"] = bars
    if not args.cohort.startswith("debug"):
        ledger_claim(f"a-second-{args.cohort}")
    with Job(out, report, _cap(args)):
        devices.configure_determinism("cuda", strict=False)
        rep, data, roots_sha = load_roots(args.roots, args.cohort, report["revision"])
        if args.limit:
            if not args.cohort.startswith("debug"):
                raise rt.GuardError("--limit is for debug cohorts only")
            rep = dict(rep, roots=rep["roots"][: args.limit])
        enc = r090._encoded(gr.encoder("cuda"), data, rep)
        pins = _feature_pins()
        report["feature_reports_sha256"] = pins
        pms, _ = r090._pin_model_pms()
        report["ceiling"] = r090._ceiling(pms, enc, data, rep)
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
        family_bars = bars["latent+pose"][str(pcl.A_SECOND_H)]
        results, mismatches = {}, []
        for arm, corpus, seed in pcl.a_second_models():
            model, info = load_model(arm, corpus, seed, pins)
            name = pcl.model_name(arm, corpus, seed)
            fams = {"latent": gr.planner_model(model, w_pose=0.0)}
            if arm == "G":
                fams["latent+pose"] = gr.planner_model(model, w_pose=1.0)
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
                        d1, d2 = r090.shuffled_deal(s, ci, ta, aa)
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
            if arm == "G":
                n = len(rep["roots"])
                res["a1_h8"] = zg.a1_pass(
                    family_bars,
                    res["latent+pose"]["model"][str(pcl.A_SECOND_H)],
                    res["latent+pose"]["shuffled"][str(pcl.A_SECOND_H)],
                    n,
                )
                plans = {root: [] for root in zg.ROOTS}
                pm = fams["latent+pose"]
                for r in rep["roots"]:
                    s = r["seed"]
                    for root in zg.ROOTS:
                        x = (
                            enc[f"{s}/{root}"],
                            data[f"{s}/{root}/state28"],
                            enc[f"{s}/goal/{root}"],
                            data[f"{s}/subgoal{zg.ROOT_GOAL[root]}/state28"],
                        )
                        plan = pr.a2_plan(pm, pcl.A_SECOND_SETTING, s, *x)
                        first = zgr.first_planned(pm, pcl.A_SECOND_SETTING, s, *x)
                        if not pcl.first_command_equal(plan, first):
                            mismatches.append([name, s, root])
                        plans[root].append(plan.tolist())
                res["a2"] = pcl.a2_second(plans["R1"], plans["R2"]) | {
                    "plans_R1": plans["R1"],
                    "plans_R2": plans["R2"],
                }
                res["a_second_pass"] = pcl.a_second_pass(res["a1_h8"], res["a2"])
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
                    f"close@R1 (reported) {a2['close_R1_reported']}, A″ {res['a_second_pass']}"
                )
            results[name] = res
            del model
            if mismatches:
                break  # §4: A″ stops at the first mismatch
        void = ""
        if mismatches:
            void = f"first command differs from first_planned on {mismatches[:5]}"
        passes = {}
        if not mismatches:
            passes = {
                tag: {s: results[f"G-{tag}-{s}"]["a_second_pass"] for s in gc.MODEL_SEEDS}
                for tag in ("v2g", "v1p")
            }
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
            "a_second": pcl.a_second_row(passes, void=void),
            "completed": True,
            "cap_hit": False,
            "finished_utc": utc(),
        }
        write_json(out, report)
    log(f"Stage A″: {report['a_second']}")
    return EXIT_VOID if mismatches else 0


# ----- §5: goals, runs, combine, video ---------------------------------------------------------
def _goal_worker(seeds):
    from embodied_jepa import play_corpus as pc

    robot = pc.make_play_robot()
    out = [zs.make_grasp_goal(robot, s) for s in seeds]
    robot.close()
    return out


def cmd_goals(args) -> int:
    if args.cohort not in pcl.GRASP_COHORTS:
        raise rt.GuardError(f"{args.cohort} is not a closed-loop cohort")
    report = base_report(args.cohort)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    jpath, npath = out / f"goals-{args.cohort}.json", out / f"goals-{args.cohort}.npz"
    for p in (jpath, npath):
        if p.exists():
            raise rt.GuardError(f"refusing to overwrite {p}")
    seeds = _limit(args, list(pcl.COHORTS[args.cohort]))
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
            pcl.check_seed(cohort, row["seed"])
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
        if [g["seed"] for g in goals] != list(pcl.COHORTS[cohort]):
            raise rt.GuardError("G-goals: the cohort's seeds are not exactly its goals")
        if revision is not None and rep["revision"] != revision:
            raise rt.GuardError("G-goals: the goals were made at another revision")
    return goals, sha(folder / f"goals-{cohort}.json")


def _controller(arm, model_seed, episode_seed, cache, record):
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

    if "model" not in cache:
        from embodied_jepa import devices

        devices.configure_determinism("cuda", strict=False)
        cache["model"], cache["info"] = load_model(
            "G", pcl.CLOSED_LOOP_ARMS[arm], model_seed, cache["pins"]
        )
        cache["encoder"] = gr.encoder("cuda")
    return pr.controller(arm, cache["model"], cache["encoder"], episode_seed, record=record)


def _run_worker(job):
    arm, model_seed, goals, pins, record = job
    from embodied_jepa import play_corpus as pc
    from embodied_jepa import zero_shot_runtime as zr

    robot = pc.make_play_robot()
    cache, out = {"pins": pins}, []
    for goal in goals:
        controller = _controller(arm, model_seed, goal["seed"], cache, record)
        result = zr.run_episode(robot, goal, controller)
        result["goal_digest"] = goal["digest"]
        if hasattr(controller, "setting"):
            result["setting"] = controller.setting
        if hasattr(controller, "plan_log"):
            result["plan_log"] = controller.plan_log
            result["deferral"] = pcl.deferral_measures(result)
        out.append(result)
        log(
            f"{arm} {goal['seed']} success={result['success']} steps={result['steps']} "
            f"{result['stop_reason']}"
        )
    robot.close()
    return out, cache.get("info")


def run_name(arm, model_seed, *, determinism=False, part=None) -> str:
    name = f"{arm}-{model_seed}" if arm in pcl.CLOSED_LOOP_ARMS else arm
    if determinism:
        name += "-determinism"
    if part is not None:
        name += f".part{part[0]}of{part[1]}"
    return name


def cmd_run(args) -> int:
    report = base_report(args.cohort)
    debug = args.cohort.startswith("debug")
    try:
        pcl.check_run(args.cohort, args.arm, args.model_seed)
    except Exception as error:
        raise rt.GuardError(str(error)) from error
    _lambda_check()
    goals, goals_sha = load_goals(args.goals, args.cohort, None if debug else report["revision"])
    if args.limit:
        if not debug:
            raise rt.GuardError("--limit is for debug cohorts only")
        goals = goals[: args.limit]
    if args.determinism:
        if (
            args.cohort not in ("bs-grasp", "debug-grasp")
            or (args.arm, args.model_seed) != (pcl.DETERMINISM[:2])
        ):
            raise rt.GuardError("the determinism re-run is M-v2g-90100 on bs-grasp (§5.1)")
        goals = goals[: pcl.DETERMINISM[2]]
    part = None
    need = pcl.needs_parts(args.cohort, args.arm)
    if args.part:
        part = pcl.parse_part(args.part)
        if args.determinism:
            raise rt.GuardError("the determinism re-run is one job")
        if not debug and (need is None or part[1] != need):
            raise rt.GuardError(f"{args.arm} on {args.cohort} runs as {need or 1} job(s)")
        goals = pcl.split_goals(goals, *part)
    elif need is not None and not args.determinism:
        raise rt.GuardError(f"{args.arm} on {args.cohort} runs as {need} jobs (§5.4): --part")
    record = not args.no_plan_record
    if not record and not debug:
        raise rt.GuardError("--no-plan-record is for debug cohorts only")
    name = run_name(args.arm, args.model_seed, determinism=args.determinism, part=part)
    if args.suffix and not debug:
        raise rt.GuardError("--suffix is for debug cohorts only")
    out = Path(args.output) / args.cohort / f"{name}{args.suffix or ''}.json"
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    pins = _feature_pins() if args.arm in pcl.CLOSED_LOOP_ARMS else None
    report |= {
        "arm": args.arm,
        "model_seed": args.model_seed,
        "goals_sha256": goals_sha,
        "setting": "step-mean",
        "cem": zs.CEM,
        "lambda": zs.LAMBDA,
        "workers": args.workers,
        "feature_reports_sha256": pins,
        "determinism_rerun": bool(args.determinism),
        "part": list(part) if part else None,
        "plan_record": bool(record and args.arm in pcl.CLOSED_LOOP_ARMS),
        "seeds": [g["seed"] for g in goals],
    }
    if not debug:
        ledger_claim(f"run-{args.cohort}-{name}")
    with Job(out, report, _cap(args)) as job:
        workers = min(args.workers, len(goals))
        jobs = [
            (args.arm, args.model_seed, goals[i::workers], pins, record) for i in range(workers)
        ]
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
    """§5.4: a split run's parts combined into one run (CPU), before any summary."""
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
        expected = list(pcl.COHORTS[args.cohort])
    combined = pcl.combine_parts(parts, expected)
    combined |= {
        "part_files": {f.name: sha(f) for f in files},
        "combined_utc": utc(),
        "combined_revision": git("rev-parse", "HEAD"),
    }
    write_json(out, combined)
    log(f"{name}: {len(parts)} parts, {combined['successes']}/{combined['n']}")
    return 0


def cmd_video(args) -> int:
    """TASK-090's MP4 replay (its code, copied; TASK-091's goals and label): replay one episode's
    applied commands from its reset (checked against the run's palm distances) and write an MP4:
    the overview camera, the onboard frame and the current subgoal."""
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
    label = f"TASK-091 grasp | arm {name} (step-mean) | reset {goal['seed']} | {verdict} (sim)"
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
    v.add_argument("--main-checkout", default="/home/huhn/develop/emai/open-embodied-jepa")
    v.add_argument(
        "--task087-evidence", default=str(Path.home() / "develop/emai/evidence/task087-val")
    )
    v.add_argument(
        "--task090-evidence", default=str(Path.home() / "develop/emai/evidence/task090-t")
    )
    se = sub.add_parser("seeds")
    se.add_argument("--output", required=True)
    se.add_argument("--evidence", default=str(Path.home() / "develop/emai/evidence"))
    ro = sub.add_parser("roots")
    ro.add_argument("--cohort", required=True, choices=pcl.ROOT_COHORTS)
    ro.add_argument("--output", required=True)
    ro.add_argument("--workers", type=int, default=8)
    ro.add_argument("--limit", type=int)
    ro.add_argument("--cap-seconds", type=int)
    a = sub.add_parser("a-second")
    a.add_argument("--cohort", required=True, choices=pcl.ROOT_COHORTS)
    a.add_argument("--roots", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--limit", type=int)
    a.add_argument("--cap-seconds", type=int)
    g = sub.add_parser("goals")
    g.add_argument("--cohort", required=True, choices=pcl.GRASP_COHORTS)
    g.add_argument("--output", required=True)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--limit", type=int)
    g.add_argument("--cap-seconds", type=int)
    r = sub.add_parser("run")
    r.add_argument("--cohort", required=True, choices=pcl.GRASP_COHORTS)
    r.add_argument("--arm", required=True, choices=sorted(pcl.CLOSED_LOOP_ARMS) + list(
        pcl.BASELINE_ARMS))  # fmt: skip
    r.add_argument("--model-seed", type=int)
    r.add_argument("--goals", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--part", help="i/k: job i of the run's k jobs (BS world-model runs, §5.4)")
    r.add_argument("--limit", type=int)
    r.add_argument("--determinism", action="store_true")
    r.add_argument("--no-plan-record", action="store_true", help="debug: TASK-090's controller")
    r.add_argument("--suffix", help="debug: an output name suffix")
    r.add_argument("--cap-seconds", type=int)
    c = sub.add_parser("combine")
    c.add_argument("--cohort", required=True, choices=pcl.GRASP_COHORTS)
    c.add_argument("--arm", required=True, choices=sorted(pcl.CLOSED_LOOP_ARMS))
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
        "roots": cmd_roots,
        "a-second": cmd_a_second,
        "goals": cmd_goals,
        "run": cmd_run,
        "combine": cmd_combine,
        "video": cmd_video,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
