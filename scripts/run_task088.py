"""TASK-088 (Phase 3): zero-shot reach and grasp-and-lift by CEM over the TASK-087 world models.

Protocol: docs/experiments/zero_shot_reach_grasp_v1.md. Subcommands (each one GPU job, run under
``scripts/gpu_run.sh --wait --min-free-gib 8 --board``):

* ``goals --cohort C --output DIR``: the task's demonstrators make every goal of cohort C
  (``DIR/goals-C.json`` + ``DIR/goals-C.npz``); refuses an existing file.
* ``lambda --output DIR``: the pose weight lambda (§5.1) on TASK-087's val store.
* ``run --cohort C --arm A [--model-seed S] --goals DIR --output DIR``: one arm on cohort C
  (``DIR/C/<arm>[-<seed>].json``); refuses an existing file; checks the goals' digests.
* ``video --result FILE --goals DIR --seed N --output FILE.mp4``: replay one episode's applied
  commands from its reset and render it from the overview and onboard cameras.

Non-debug cohorts need a clean tracked tree; every report records the revision.
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

TASK087 = Path(os.environ.get("TASK087_RUN", "/home/huhn/develop/emai/worktrees/task087-run"))
FEATURES = TASK087 / "outputs/task087-features"
CHECKPOINTS = TASK087 / "checkpoints/task087-run"
DISK_FLOOR_GIB = 10.0


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{utc()}] {msg}", flush=True)


def git(*args) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def base_report(args, cohort=None) -> dict:
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    report = {
        "task": zs.TASK,
        "protocol": zs.PROTOCOL,
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


# ----- goals -------------------------------------------------------------------------------------
def _goal_worker(args):
    cohort, seeds = args
    from embodied_jepa import play_corpus as pc

    task = zs.cohort_task(cohort)
    robot = pc.make_play_robot()
    out = []
    for seed in seeds:
        out.append(zs.make_goal(robot, task, seed))
    robot.close()
    return out


def cmd_goals(args) -> int:
    report = base_report(args, args.cohort)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    jpath, npath = out / f"goals-{args.cohort}.json", out / f"goals-{args.cohort}.npz"
    if jpath.exists() or npath.exists():
        raise rt.GuardError(f"refusing to overwrite {jpath}")
    seeds = list(zs.COHORTS[args.cohort])
    chunks = [seeds[i :: args.workers] for i in range(args.workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        parts = pool.map(_goal_worker, [(args.cohort, c) for c in chunks if c])
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


def load_goals(folder, cohort) -> list[dict]:
    folder = Path(folder)
    rep = json.loads((folder / f"goals-{cohort}.json").read_text())
    npath = folder / f"goals-{cohort}.npz"
    if sha256(npath) != rep["frames_sha256"]:
        raise rt.GuardError("G-goals: frames differ from the goal report")
    goals = []
    with np.load(npath) as frames:
        for row in rep["goals"]:
            zs.check_seed(cohort, row["seed"])
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
    if [g["seed"] for g in goals] != list(zs.COHORTS[cohort]):
        raise rt.GuardError("G-goals: the cohort's seeds are not exactly its goals")
    return goals, sha256(folder / f"goals-{cohort}.json")


# ----- lambda ------------------------------------------------------------------------------------
def cmd_lambda(args) -> int:
    import torch

    report = base_report(args)
    out = Path(args.output) / "lambda.json"
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    lat = np.load(FEATURES / "latents_val.npy", mmap_mode="r")
    st = np.load(FEATURES / "state_val.npy")
    eps = json.loads((FEATURES / "episodes_val.json").read_text())
    report["files_sha256"] = {
        n: sha256(FEATURES / n) for n in ("latents_val.npy", "state_val.npy", "episodes_val.json")
    }
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
    report["finished_utc"] = utc()
    write_json(out, report)
    log(json.dumps(report["per_seed"]))
    return 0


# ----- run ---------------------------------------------------------------------------------------
def _controller(arm, model_seed, episode_seed, cache):
    if arm == "hold":
        return zs.HoldController()
    if arm == "random":
        return zs.RandomController(episode_seed)
    if arm == "ik":
        return zs.IKController()
    if arm == "ik-nopress":
        return zs.IKController(press=False)
    from embodied_jepa import zero_shot_runtime as zr

    model_arm, w_lat, w_pose = zs.MODEL_ARMS[arm]
    if "model" not in cache:
        device = "cuda"
        from embodied_jepa import devices

        devices.configure_determinism(device, strict=False)
        model, info = zr.load_model(CHECKPOINTS / f"{model_arm}-{model_seed}", device)
        cache["model"], cache["info"] = model, info
        cache["encoder"] = zr.Encoder(FEATURES / "projection.npz", device)
    pm = zr.PlannerModel(
        cache["model"], w_lat=w_lat, w_pose=w_pose, lam=cache["lambda"], device="cuda"
    )
    return zr.ModelController(arm, pm, cache["encoder"], episode_seed)


def _run_worker(job):
    arm, model_seed, goals, lam = job
    from embodied_jepa import play_corpus as pc
    from embodied_jepa import zero_shot_runtime as zr

    robot = pc.make_play_robot()
    cache = {"lambda": lam}
    out = []
    for goal in goals:
        controller = _controller(arm, model_seed, goal["seed"], cache)
        result = zr.run_episode(robot, goal, controller)
        result["goal_digest"] = goal["digest"]
        out.append(result)
        print(
            f"[{utc()}] {arm} {goal['seed']} success={result['success']} steps={result['steps']}"
            f" {result['stop_reason']}",
            flush=True,
        )
    robot.close()
    return out, cache.get("info")


def cmd_run(args) -> int:
    report = base_report(args, args.cohort)
    zs.check_run(args.cohort, args.arm, args.model_seed)
    name = f"{args.arm}-{args.model_seed}" if args.arm in zs.MODEL_ARMS else args.arm
    out = Path(args.output) / args.cohort / f"{name}.json"
    if out.exists():
        raise rt.GuardError(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    goals, goals_sha = load_goals(args.goals, args.cohort)
    if args.limit:
        if not args.cohort.startswith("debug"):
            raise rt.GuardError("--limit is for debug cohorts only")
        goals = goals[: args.limit]
    lam = None
    if args.arm in zs.MODEL_ARMS and zs.MODEL_ARMS[args.arm][2] > 0:
        lrep = json.loads((Path(args.lambda_file)).read_text())
        lam = float(lrep["per_seed"][str(args.model_seed)]["lambda"])
        report["lambda"] = lam
        report["lambda_file_sha256"] = sha256(args.lambda_file)
    report |= {
        "arm": args.arm,
        "model_seed": args.model_seed,
        "goals_sha256": goals_sha,
        "cem": zs.CEM,
        "workers": args.workers,
    }
    t0 = time.time()
    jobs = [(args.arm, args.model_seed, goals[i :: args.workers], lam) for i in range(args.workers)]
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


# ----- video -------------------------------------------------------------------------------------
def cmd_video(args) -> int:
    """Replay one episode's applied commands from its reset (MuJoCo is deterministic for the same
    commands; the replayed palm distances are checked against the run's) and write an MP4: the
    overview camera, the onboard frame and the current (sub)goal image, with a label."""
    from PIL import Image, ImageDraw

    from embodied_jepa import play_corpus as pc

    base_report(args)
    result = json.loads(Path(args.result).read_text())
    cohort = result["cohort"]
    goals, _ = load_goals(args.goals, cohort)
    goal = next(g for g in goals if g["seed"] == args.seed)
    episode = next(e for e in result["episodes"] if e["seed"] == args.seed)
    robot = pc.make_play_robot()
    sim = robot.sim
    zs.reset(robot, goal["layout"])
    import mujoco

    big = mujoco.Renderer(sim.model, height=360, width=480)
    verdict = "SUCCESS" if episode["success"] else "FAILURE"
    name = result["arm"] + (f"-{result['model_seed']}" if result.get("model_seed") else "")
    label = f"TASK-088 {goal['task']} | arm {name} | reset {goal['seed']} | {verdict} (sim)"
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
        sub = zs.SUBGOALS[index] if goal["task"] == "grasp" else "goal"
        draw.text(
            (6, 20),
            f"step {step}  current: onboard (top right), {sub} image (bottom right)",
            fill=(200, 200, 200),
        )
        frames.append(np.asarray(image))

    switch = zs.SubgoalSwitch(goal["subgoals"])
    replayed = []
    index = 0
    for k, action in enumerate(episode["applied"]):
        obs = robot.observe()
        p = zs.proprio(robot, obs)
        if goal["task"] == "grasp":
            index = switch.update(p["palm"], p["state28"][7:14], p["rotation"])
        snap(index, k)
        out = robot.execute(np.asarray(action, np.float32))
        if out.status not in ("applied", "clipped"):
            log(f"replay stopped at {k}: {out.reason}")
            break
        palm, _ = robot.ee_pose("right")
        replayed.append(float(np.linalg.norm(palm - goal["subgoals"][index]["palm"])))
    snap(index, len(episode["applied"]))
    for _ in range(20):
        frames.append(frames[-1])
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
    g = sub.add_parser("goals")
    g.add_argument("--cohort", required=True, choices=sorted(zs.COHORTS))
    g.add_argument("--output", required=True)
    g.add_argument("--workers", type=int, default=8)
    la = sub.add_parser("lambda")
    la.add_argument("--output", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cohort", required=True, choices=sorted(zs.COHORTS))
    r.add_argument("--arm", required=True, choices=sorted(zs.MODEL_ARMS) + list(zs.BASELINE_ARMS))
    r.add_argument("--model-seed", type=int)
    r.add_argument("--goals", required=True)
    r.add_argument("--lambda-file")
    r.add_argument("--output", required=True)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--limit", type=int)
    v = sub.add_parser("video")
    v.add_argument("--result", required=True)
    v.add_argument("--goals", required=True)
    v.add_argument("--seed", type=int, required=True)
    v.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    return {"goals": cmd_goals, "lambda": cmd_lambda, "run": cmd_run, "video": cmd_video}[
        args.command
    ](args)


if __name__ == "__main__":
    raise SystemExit(main())
