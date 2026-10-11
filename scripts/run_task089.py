"""TASK-089: zero-shot grasp-and-lift, Part A (offline) and Part B (closed loop).

Protocol: docs/experiments/zero_shot_grasp_v1.md. Subcommands (each one GPU job, run under
``scripts/gpu_run.sh --wait --min-free-gib 8 --board``):

* ``lambda --output DIR``: the pose weight lambda on TASK-087's val store (checked = 2.0552).
* ``roots --cohort C --output DIR``: Part A's roots and branches (``DIR/roots-C.{json,npz}``).
* ``ceiling --cohort C --roots DIR --output DIR``: A1's simulator ceiling (``DIR/ceiling-C.json``).
* ``a1 --cohort C --roots DIR --output DIR``: A1 for every model arm, the shuffled control and the
  ceiling (``DIR/a1-C.json``); not on the calibration cohort.
* ``a2 --cohort C --roots DIR --output DIR``: A2's first planned commands (``DIR/a2-C.json``).
* ``goals --cohort C --output DIR``: Part B goals by TASK-088's grasp demonstrator.
* ``run --cohort C --arm A [--model-seed S] --change-file F --goals DIR --output DIR``: one Part B
  arm; the change is read from Stage A's summary (it must be A-PASS(k)).
* ``video --result FILE --goals DIR --seed N --output FILE.mp4``: replay one Part B episode.

Non-debug cohorts need a clean tracked tree; every report records the revision; every file is
refused if it exists.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import run_tools as rt  # noqa: E402
from embodied_jepa import zero_shot as zs  # noqa: E402
from embodied_jepa import zero_shot_grasp as zg  # noqa: E402

TASK087 = Path(os.environ.get("TASK087_RUN", "/home/huhn/develop/emai/worktrees/task087-run"))
FEATURES = TASK087 / "outputs/task087-features"
CHECKPOINTS = TASK087 / "checkpoints/task087-run"
DISK_FLOOR_GIB = 10.0
A1_ARMS = (("G", 87100), ("P", 87100), ("G-lat", 87100), ("N", 87100)) + tuple(
    (a, s) for s in zg.REPLICATION_SEEDS for a in ("G", "P")
)
A2_MODELS = (("G", 1.0, 1.0), ("P", 1.0, 0.0))


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc()}] {msg}", flush=True)


def git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def base_report(cohort=None) -> dict:
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    report = {
        "task": zg.TASK,
        "protocol": zg.PROTOCOL,
        "started_utc": utc(),
        "revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": dirty,
        "argv": sys.argv,
        "platform": platform.platform(),
        "cohort": cohort,
    }
    rt.assert_local_import(ROOT, report)
    if cohort is not None and not cohort.startswith("debug") and dirty:
        raise rt.GuardError("G-clean: non-debug cohorts run from a clean tracked tree")
    free = shutil.disk_usage("/").free / 2**30
    report["disk_free_gib"] = free
    if free < DISK_FLOOR_GIB:
        raise rt.GuardError(f"G-disk: {free:.1f} GiB free < {DISK_FLOOR_GIB}")
    if os.environ.get(rt.GPU_LOCK_ENV) != "1":
        raise rt.GuardError("G-gpu-lock: run through scripts/gpu_run.sh")
    report["gpu_lock_held"] = True
    report["mujoco_gl"] = os.environ.get("MUJOCO_GL")
    return report


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n")


def sha256(path) -> str:
    from embodied_jepa.grounded_wm_data import sha256_file

    return sha256_file(path)


def refuse_existing(*paths):
    for p in paths:
        if Path(p).exists():
            raise rt.GuardError(f"refusing to overwrite {p}")


def same_revision(report, path):
    other = json.loads(Path(path).read_text())["revision"]
    if not report["cohort"].startswith("debug") and other != report["revision"]:
        raise rt.GuardError(f"G-revision: {path} was made at another revision")


# ----- lambda ------------------------------------------------------------------------------------
def cmd_lambda(args) -> int:
    import torch

    report = base_report()
    out = Path(args.output) / "lambda.json"
    refuse_existing(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lat = np.load(FEATURES / "latents_val.npy", mmap_mode="r")
    st = np.load(FEATURES / "state_val.npy")
    eps = json.loads((FEATURES / "episodes_val.json").read_text())
    report["per_seed"] = {}
    for seed in zs.MODEL_SEEDS:
        sd = torch.load(CHECKPOINTS / f"G-{seed}/model.pt", map_location="cpu", weights_only=True)
        report["per_seed"][str(seed)] = zs.pose_lambda(
            lat,
            st,
            eps,
            sd["state_dict"]["state_mean"].numpy(),
            sd["state_dict"]["state_scale"].numpy(),
        )
        if abs(report["per_seed"][str(seed)]["lambda"] - zs.LAMBDA) > 1e-4:
            raise rt.GuardError("G-lambda: differs from TASK-088's frozen 2.0552")
    report["finished_utc"] = utc()
    write_json(out, report)
    log(json.dumps(report["per_seed"]))
    return 0


# ----- Part A: roots -----------------------------------------------------------------------------
def _roots_worker(seeds):
    from embodied_jepa import play_corpus as pc

    robot = pc.make_play_robot()
    out = [zg.build_root(robot, s) for s in seeds]
    robot.close()
    return out


def cmd_roots(args) -> int:
    if args.cohort not in zg.ROOT_COHORTS:
        raise rt.GuardError(f"{args.cohort} is not a root cohort")
    report = base_report(args.cohort)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    jpath, npath = out / f"roots-{args.cohort}.json", out / f"roots-{args.cohort}.npz"
    refuse_existing(jpath, npath)
    seeds = list(zg.COHORTS[args.cohort])[: args.limit or None]
    if args.limit and not args.cohort.startswith("debug"):
        raise rt.GuardError("--limit is for debug cohorts only")
    t0 = time.time()
    chunks = [seeds[i :: args.workers] for i in range(args.workers)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        parts = pool.map(_roots_worker, [c for c in chunks if c])
    roots = sorted((r for p in parts for r in p), key=lambda r: r["seed"])
    arrays, meta = {}, []
    for r in roots:
        s = r["seed"]
        row = {k: r[k] for k in ("seed", "attempt", "layout", "target", "yaw", "z0", "rejected")}
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
                "final_grasp_mean": float(np.mean(b["states"][-1][7:14])),
            }
        meta.append(row)
    np.savez_compressed(npath, **arrays)
    report |= {
        "seconds": time.time() - t0,
        "roots": meta,
        "arrays_sha256": sha256(npath),
        "rejections": sum(len(r["rejected"]) for r in roots),
        "finished_utc": utc(),
    }
    write_json(jpath, report)
    log(f"{len(roots)} roots, {report['rejections']} rejected draws, {report['seconds']:.0f} s")
    return 0


def load_roots(folder, cohort):
    folder = Path(folder)
    rep = json.loads((folder / f"roots-{cohort}.json").read_text())
    npath = folder / f"roots-{cohort}.npz"
    if sha256(npath) != rep["arrays_sha256"]:
        raise rt.GuardError("G-roots: arrays differ from the roots report")
    seeds = [r["seed"] for r in rep["roots"]]
    for s in seeds:
        zg.check_seed(cohort, s)
    if not cohort.startswith("debug") and seeds != list(zg.COHORTS[cohort]):
        raise rt.GuardError("G-roots: the cohort's seeds are not exactly its roots")
    data = np.load(npath)
    return rep, data, sha256(folder / f"roots-{cohort}.json")


# ----- Part A: costs -----------------------------------------------------------------------------
def _models(arms):
    from embodied_jepa import devices
    from embodied_jepa import zero_shot_runtime as zr

    devices.configure_determinism("cuda", strict=False)
    models = {}
    for arm, seed in arms:
        model_arm = "G" if arm == "G-lat" else arm
        if (model_arm, seed) not in models:
            models[(model_arm, seed)] = zr.load_model(CHECKPOINTS / f"{model_arm}-{seed}", "cuda")
    encoder = zr.Encoder(FEATURES / "projection.npz", "cuda")
    return models, encoder


def _pm(models, arm, seed):
    from embodied_jepa import zero_shot_runtime as zr

    family = zg.ARM_FAMILY[arm]
    model = models[("G" if arm == "G-lat" else arm, seed)][0]
    w_pose = 1.0 if family == "latent+pose" else 0.0
    return zr.PlannerModel(model, w_lat=1.0, w_pose=w_pose, lam=zs.LAMBDA, device="cuda")


def _encoded(encoder, data, rep):
    """Encode every root, goal and branch frame once."""
    enc = {}
    for r in rep["roots"]:
        s = r["seed"]
        for root in zg.ROOTS:
            enc[f"{s}/{root}"] = encoder(data[f"{s}/{root}/frame"])[0]
            enc[f"{s}/goal/{root}"] = encoder(data[f"{s}/subgoal{zg.ROOT_GOAL[root]}/frame"])[0]
            for name in zg.BRANCHES[root]:
                enc[f"{s}/{root}/{name}"] = encoder(data[f"{s}/{root}/{name}/frames"])
    return enc


def _ceiling(models, enc, data, rep) -> dict:
    """{family: {h: {comparison: [win per root]}}} with the simulator as predictor."""
    from embodied_jepa import zero_shot_grasp_runtime as zgr

    out = {}
    for family in zg.COST_FAMILIES:
        pm = _pm(models, "G" if family == "latent+pose" else "G-lat", 87100)
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


def cmd_ceiling(args) -> int:
    report = base_report(args.cohort)
    out = Path(args.output) / f"ceiling-{args.cohort}.json"
    refuse_existing(out)
    same_revision(report, Path(args.roots) / f"roots-{args.cohort}.json")
    rep, data, roots_sha = load_roots(args.roots, args.cohort)
    models, encoder = _models([("G", 87100)])
    enc = _encoded(encoder, data, rep)
    wins = _ceiling(models, enc, data, rep)
    n = len(rep["roots"])
    rates = {
        f: {h: {c: float(np.mean(v)) for c, v in by_c.items()} for h, by_c in by_h.items()}
        for f, by_h in wins.items()
    }
    report |= {
        "roots_sha256": roots_sha,
        "n": n,
        "wins": wins,
        "rates": rates,
        "bars": zg.k0_bars(rates),
        "finished_utc": utc(),
    }
    report["a_ceiling_row"] = zg.ceiling_row(report["bars"])
    write_json(out, report)
    log(json.dumps(rates))
    return 0


def cmd_a1(args) -> int:
    from embodied_jepa import zero_shot_grasp_runtime as zgr

    if args.cohort == "k0-roots":
        raise rt.GuardError("A-K0 computes the simulator ceiling only (§5.4)")
    report = base_report(args.cohort)
    out = Path(args.output) / f"a1-{args.cohort}.json"
    refuse_existing(out)
    same_revision(report, Path(args.roots) / f"roots-{args.cohort}.json")
    rep, data, roots_sha = load_roots(args.roots, args.cohort)
    models, encoder = _models(A1_ARMS)
    enc = _encoded(encoder, data, rep)
    report["models"] = {f"{a}-{s}": m[1] for (a, s), m in models.items()}
    report["ceiling"] = _ceiling(models, enc, data, rep)
    arms = {}
    for arm, seed in A1_ARMS:
        pm = _pm(models, arm, seed)
        res = {
            kind: {str(h): {zg.comparison_name(c): [] for c in zg.COMPARISONS} for h in zg.HORIZONS}
            for kind in ("model", "shuffled")
        }
        for r in rep["roots"]:
            s = r["seed"]
            for comp in zg.COMPARISONS:
                root, t, a = comp
                rl, rs = enc[f"{s}/{root}"], data[f"{s}/{root}/state28"]
                gl, gs = enc[f"{s}/goal/{root}"], data[f"{s}/subgoal{zg.ROOT_GOAL[root]}/state28"]
                ta, aa = data[f"{s}/{root}/{t}/actions"], data[f"{s}/{root}/{a}/actions"]
                d1, d2 = zg.shuffled_deal(s, root, ta, aa)
                costs = zgr.rollout_costs(pm, rl, rs, gl, gs, np.stack([ta, aa, d1, d2]))
                for h in zg.HORIZONS:
                    c = zg.comparison_name(comp)
                    res["model"][str(h)][c].append(bool(zg.wins(costs[0, h - 1], costs[1, h - 1])))
                    res["shuffled"][str(h)][c].append(
                        bool(zg.wins(costs[2, h - 1], costs[3, h - 1]))
                    )
        arms[f"{arm}-{seed}"] = res
    report |= {
        "roots_sha256": roots_sha,
        "n": len(rep["roots"]),
        "arms": arms,
        "lambda": zs.LAMBDA,
        "finished_utc": utc(),
    }
    write_json(out, report)
    for name, res in arms.items():
        log(
            name
            + " "
            + json.dumps(
                {h: {c: int(sum(v)) for c, v in by.items()} for h, by in res["model"].items()}
            )
        )
    return 0


def cmd_a2(args) -> int:
    from embodied_jepa import zero_shot_grasp_runtime as zgr

    if args.cohort == "k0-roots":
        raise rt.GuardError("A-K0 computes the simulator ceiling only (§5.4)")
    report = base_report(args.cohort)
    out = Path(args.output) / f"a2-{args.cohort}.json"
    refuse_existing(out)
    same_revision(report, Path(args.roots) / f"roots-{args.cohort}.json")
    rep, data, roots_sha = load_roots(args.roots, args.cohort)
    models, encoder = _models([("G", 87100), ("P", 87100)])
    enc = _encoded(encoder, data, rep)
    res = {}
    t0 = time.time()
    for arm, _, _ in A2_MODELS:
        pm = _pm(models, arm, 87100)
        res[arm] = {}
        for setting in zg.SETTINGS:
            firsts = {root: [] for root in zg.ROOTS}
            for r in rep["roots"]:
                s = r["seed"]
                for root in zg.ROOTS:
                    firsts[root].append(
                        zgr.first_planned(
                            pm,
                            setting,
                            s,
                            enc[f"{s}/{root}"],
                            data[f"{s}/{root}/state28"],
                            enc[f"{s}/goal/{root}"],
                            data[f"{s}/subgoal{zg.ROOT_GOAL[root]}/state28"],
                        ).tolist()
                    )
            m = zg.a2_measures(firsts["R1"], firsts["R2"])
            res[arm][setting] = {
                "first_R1": firsts["R1"],
                "first_R2": firsts["R2"],
                "close_R1": [bool(x) for x in m["close_R1"]],
                "lift_R2": [bool(x) for x in m["lift_R2"]],
            }
            log(
                f"{arm} {setting}: close@R1 {int(m['close_R1'].sum())} lift@R2 "
                f"{int(m['lift_R2'].sum())} of {len(rep['roots'])}"
            )
    report |= {
        "roots_sha256": roots_sha,
        "n": len(rep["roots"]),
        "a2": res,
        "seconds": time.time() - t0,
        "finished_utc": utc(),
    }
    write_json(out, report)
    return 0


# ----- Part B: goals, runs, video ----------------------------------------------------------------
def _goal_worker(seeds):
    from embodied_jepa import play_corpus as pc

    robot = pc.make_play_robot()
    out = [zs.make_grasp_goal(robot, s) for s in seeds]
    robot.close()
    return out


def cmd_goals(args) -> int:
    if args.cohort not in zg.GRASP_COHORTS:
        raise rt.GuardError(f"{args.cohort} is not a Part B cohort")
    report = base_report(args.cohort)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    jpath, npath = out / f"goals-{args.cohort}.json", out / f"goals-{args.cohort}.npz"
    refuse_existing(jpath, npath)
    seeds = list(zg.COHORTS[args.cohort])
    chunks = [seeds[i :: args.workers] for i in range(args.workers)]
    t0 = time.time()
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
        "seconds": time.time() - t0,
        "goals": meta,
        "frames_sha256": sha256(npath),
        "rejections": sum(len(g["rejected"]) for g in goals),
        "finished_utc": utc(),
    }
    write_json(jpath, report)
    log(f"{len(goals)} goals, {report['rejections']} rejected draws, {report['seconds']:.0f} s")
    return 0


def load_goals(folder, cohort):
    folder = Path(folder)
    rep = json.loads((folder / f"goals-{cohort}.json").read_text())
    npath = folder / f"goals-{cohort}.npz"
    if sha256(npath) != rep["frames_sha256"]:
        raise rt.GuardError("G-goals: frames differ from the goal report")
    goals = []
    with np.load(npath) as frames:
        for row in rep["goals"]:
            zg.check_seed(cohort, row["seed"])
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
    if [g["seed"] for g in goals] != list(zg.COHORTS[cohort]):
        raise rt.GuardError("G-goals: the cohort's seeds are not exactly its goals")
    return goals, sha256(folder / f"goals-{cohort}.json"), rep["revision"]


def selected_change(args, cohort):
    """Part B's change: from Stage A's summary (A-PASS(k)); a debug cohort may name a setting."""
    if args.change_file:
        summary = json.loads(Path(args.change_file).read_text())
        row = summary["row"]
        if not row.startswith("A-PASS(") or summary.get("cohort") != "a-roots":
            raise rt.GuardError(f"G-change: Part B needs A-PASS on Stage A, not {row}")
        return summary["selected_change"], sha256(args.change_file)
    if cohort.startswith("debug") and args.setting:
        return args.setting, None
    raise rt.GuardError("G-change: give Stage A's summary (--change-file)")


def _controller(arm, model_seed, episode_seed, cache, change):
    if arm == "hold":
        return zs.HoldController()
    if arm == "random":
        return zs.RandomController(episode_seed)
    if arm == "ik":
        return zs.IKController()
    if arm == "ik-nopress":
        return zs.IKController(press=False)
    from embodied_jepa import zero_shot_grasp_runtime as zgr
    from embodied_jepa import zero_shot_runtime as zr

    model_arm, w_lat, w_pose, uses_change = zg.MODEL_ARMS[arm]
    if "model" not in cache:
        from embodied_jepa import devices

        devices.configure_determinism("cuda", strict=False)
        cache["model"], cache["info"] = zr.load_model(
            CHECKPOINTS / f"{model_arm}-{model_seed}", "cuda"
        )
        cache["encoder"] = zr.Encoder(FEATURES / "projection.npz", "cuda")
    pm = zr.PlannerModel(cache["model"], w_lat=w_lat, w_pose=w_pose, lam=zs.LAMBDA, device="cuda")
    setting = change if uses_change else "base"
    return zgr.ChangeController(arm, pm, cache["encoder"], episode_seed, setting)


def _run_worker(job):
    arm, model_seed, goals, change = job
    from embodied_jepa import play_corpus as pc
    from embodied_jepa import zero_shot_runtime as zr

    robot = pc.make_play_robot()
    cache, out = {}, []
    for goal in goals:
        controller = _controller(arm, model_seed, goal["seed"], cache, change)
        result = zr.run_episode(robot, goal, controller)
        result["goal_digest"] = goal["digest"]
        if hasattr(controller, "setting"):
            result["setting"] = controller.setting
        out.append(result)
        log(
            f"{arm} {goal['seed']} success={result['success']} steps={result['steps']} "
            f"{result['stop_reason']}"
        )
    robot.close()
    return out, cache.get("info")


def cmd_run(args) -> int:
    report = base_report(args.cohort)
    zg.check_run(args.cohort, args.arm, args.model_seed)
    name = f"{args.arm}-{args.model_seed}" if args.arm in zg.MODEL_ARMS else args.arm
    out = Path(args.output) / args.cohort / f"{name}.json"
    refuse_existing(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    change, change_sha = selected_change(args, args.cohort)
    if change not in zg.CANDIDATES:
        raise rt.GuardError(f"G-change: {change} is not a candidate")
    goals, goals_sha, grev = load_goals(args.goals, args.cohort)
    if not args.cohort.startswith("debug") and grev != report["revision"]:
        raise rt.GuardError("G-goals: the goals were made at another revision")
    if args.limit:
        if not args.cohort.startswith("debug"):
            raise rt.GuardError("--limit is for debug cohorts only")
        goals = goals[: args.limit]
    report |= {
        "arm": args.arm,
        "model_seed": args.model_seed,
        "goals_sha256": goals_sha,
        "change": change,
        "change_file_sha256": change_sha,
        "cem": zs.CEM,
        "lambda": zs.LAMBDA,
        "workers": args.workers,
    }
    t0 = time.time()
    jobs = [
        (args.arm, args.model_seed, goals[i :: args.workers], change) for i in range(args.workers)
    ]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        parts = pool.map(_run_worker, [j for j in jobs if j[2]])
    episodes = sorted((e for p, _ in parts for e in p), key=lambda e: e["seed"])
    infos = [i for _, i in parts if i]
    report |= {
        "model": infos[0] if infos else None,
        "episodes": episodes,
        "successes": int(sum(e["success"] for e in episodes)),
        "n": len(episodes),
        "seconds": time.time() - t0,
        "finished_utc": utc(),
    }
    write_json(out, report)
    log(
        f"{name} on {args.cohort}: {report['successes']}/{report['n']} in {report['seconds']:.0f} s"
    )
    return 0


def cmd_video(args) -> int:
    """Replay one Part B episode's applied commands from its reset (checked against the run's palm
    distances) and write an MP4: the overview camera, the onboard frame and the current subgoal."""
    from PIL import Image, ImageDraw

    from embodied_jepa import play_corpus as pc

    base_report()
    result = json.loads(Path(args.result).read_text())
    goals, _, _ = load_goals(args.goals, result["cohort"])
    goal = next(g for g in goals if g["seed"] == args.seed)
    episode = next(e for e in result["episodes"] if e["seed"] == args.seed)
    robot = pc.make_play_robot()
    sim = robot.sim
    zs.reset(robot, goal["layout"])
    import mujoco

    big = mujoco.Renderer(sim.model, height=360, width=480)
    verdict = "SUCCESS" if episode["success"] else "FAILURE"
    name = result["arm"] + (f"-{result['model_seed']}" if result.get("model_seed") else "")
    label = (
        f"TASK-089 grasp | arm {name} ({result['change']}) | reset {goal['seed']} | {verdict} (sim)"
    )
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
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        "20",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        args.output,
    ]
    subprocess.run(cmd, input=np.stack(frames).tobytes(), check=True)
    Path(args.output + ".json").write_text(
        json.dumps(
            {
                "label": label,
                "result_file": args.result,
                "result_sha256": sha256(args.result),
                "seed": args.seed,
                "replay_matches": bool(ok),
                "frames": len(frames),
            },
            indent=1,
        )
        + "\n"
    )
    log(f"wrote {args.output} ({len(frames)} frames)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    la = sub.add_parser("lambda")
    la.add_argument("--output", required=True)
    ro = sub.add_parser("roots")
    ro.add_argument("--cohort", required=True, choices=zg.ROOT_COHORTS)
    ro.add_argument("--output", required=True)
    ro.add_argument("--workers", type=int, default=8)
    ro.add_argument("--limit", type=int)
    for name in ("ceiling", "a1", "a2"):
        p = sub.add_parser(name)
        p.add_argument("--cohort", required=True, choices=zg.ROOT_COHORTS)
        p.add_argument("--roots", required=True)
        p.add_argument("--output", required=True)
    g = sub.add_parser("goals")
    g.add_argument("--cohort", required=True, choices=zg.GRASP_COHORTS)
    g.add_argument("--output", required=True)
    g.add_argument("--workers", type=int, default=8)
    r = sub.add_parser("run")
    r.add_argument("--cohort", required=True, choices=zg.GRASP_COHORTS)
    r.add_argument("--arm", required=True, choices=sorted(zg.MODEL_ARMS) + list(zg.BASELINE_ARMS))
    r.add_argument("--model-seed", type=int)
    r.add_argument("--change-file")
    r.add_argument("--setting", choices=zg.CANDIDATES, help="debug cohorts only")
    r.add_argument("--goals", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--limit", type=int)
    v = sub.add_parser("video")
    v.add_argument("--result", required=True)
    v.add_argument("--goals", required=True)
    v.add_argument("--seed", type=int, required=True)
    v.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    return {
        "lambda": cmd_lambda,
        "roots": cmd_roots,
        "ceiling": cmd_ceiling,
        "a1": cmd_a1,
        "a2": cmd_a2,
        "goals": cmd_goals,
        "run": cmd_run,
        "video": cmd_video,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
