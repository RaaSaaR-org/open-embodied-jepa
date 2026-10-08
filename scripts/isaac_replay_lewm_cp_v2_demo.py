"""TASK-081 demo in Isaac Sim: a kinematic replay of MuJoCo episodes. Illustration only.

Not evidence, not a gated run, and **not an Isaac run of the controller**. The LeWM controller
(W) was trained and evaluated on MuJoCo images; running it closed-loop in Isaac would be a
different, unvalidated experiment. This script instead re-runs the MuJoCo attempts of the demo
video (``scripts/render_lewm_cp_v2_demo.py``, debug seeds only) and records their states, and a
container script (``scripts/isaac/replay_kinematic_arena.py``) poses the converted G1 + Dex3 USD,
an apple and a plate in the GR00T tutorial's Arena shelf scene from those states, frame by frame,
with no physics, and renders a third-person camera. Isaac's physics is not exercised: every pose
in the Isaac frames is MuJoCo's.

Subcommands:

- ``dump``: re-runs the demo's clips (W on 71921; W, N and H-rule on 71920, the clips its fixed
  rule selected) with TASK-081's worker, unchanged, through the demo script's render hooks, and
  writes per clip ``<clip>.npz``: the world pose of every MuJoCo body at every control step
  (``xpos``, ``xquat`` wxyz, with ``body_names`` and ``body_parent``), the apple and plate poses,
  the committed aim, the plate's final position (the scorer's 4 cm disc centre), the 112 px
  onboard frames and the MuJoCo third-person frames at the rendered steps, and the outcome. No
  screen is run; the clips are the demo's. ``report.json`` checks each outcome against the demo's.
- ``compose``: reads the dump and the Isaac frames and writes the captioned MP4s (Isaac only,
  and MuJoCo | Isaac side by side) with system ``ffmpeg``, plus ``SHA256SUMS``.
"""

from __future__ import annotations

import os

os.environ.update(
    {
        "MKL_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": "6",
        "MKL_NUM_THREADS": "6",
        "OPENBLAS_NUM_THREADS": "16",
    }
)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CLIPS = (("W", 71921), ("W", 71920), ("N", 71920), ("H-rule", 71920))
MIN_DISK_GIB = 10.0
LABEL = (
    "Isaac Sim kinematic replay of MuJoCo episodes, no Isaac physics - "
    "illustration only, not evidence"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def demo_module():
    return _load("render_lewm_cp_v2_demo", ROOT / "scripts" / "render_lewm_cp_v2_demo.py")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rendered_steps(t: int, stride: int, keys) -> bool:
    return t % stride == 0 or t in keys


# ----- dump ---------------------------------------------------------------------------------------
def make_logger(demo):
    class StateLogger(demo.Recorder):
        """Records every control step's MuJoCo state; renders the MuJoCo view on rendered steps."""

        def __init__(self, robot, *, arm, seed, stride, camera, keys):
            super().__init__(robot, None, None, arm=arm, seed=seed, landing=None,
                             stride=stride, camera=camera)  # fmt: skip
            self.keys = tuple(keys)
            self.rows: dict[str, list] = {k: [] for k in ("t", "xpos", "xquat", "aim")}
            self.views: list[tuple[int, np.ndarray, np.ndarray]] = []

        def observed(self, t: int, observation) -> None:
            from embodied_jepa import first_policy_v2 as fp2

            data = self.robot.sim.data
            aim = self.aim()
            self.rows["t"].append(int(t))
            self.rows["xpos"].append(np.array(data.xpos, np.float64))
            self.rows["xquat"].append(np.array(data.xquat, np.float64))
            self.rows["aim"].append(np.full(2, np.nan) if aim is None else aim[:2].copy())
            if rendered_steps(t, self.stride, self.keys):
                onboard = np.asarray(observation.images[fp2.CAMERA][0], np.uint8).copy()
                self.views.append((int(t), self.render(t), onboard))

    return StateLogger


def dump(args) -> int:
    from embodied_jepa import lewm_c1m_v2 as lm
    from embodied_jepa import lewm_cp_v2 as cpv
    from embodied_jepa import plate_shift as ps

    out_dir = Path(args.output)
    if out_dir.exists():
        raise SystemExit(f"refusing: {out_dir} exists")
    out_dir.mkdir(parents=True)
    if shutil.disk_usage(out_dir).free / 2**30 < MIN_DISK_GIB:
        raise SystemExit("refusing: under 10 GiB free")
    demo = demo_module()
    seeds = sorted({s for _, s in CLIPS})
    demo.check_debug_seeds(seeds)
    rn = demo._runner()
    from embodied_jepa import lewm_cp_v2_runtime as cpr
    from embodied_jepa import plate_twin_v2_harness as hz
    from embodied_jepa import wm_critic_v2_runtime as rtm

    started = time.monotonic()
    report = {
        "what": "TASK-081 demo, Isaac kinematic replay: MuJoCo state dump (illustration only)",
        "revision": hz.revision(),
        "tracked_tree_dirty": bool(hz.tracked_tree_dirty()),
        "argv": sys.argv[1:],
        "clips_requested": [list(c) for c in CLIPS],
        "stages": {},
    }
    manifest = json.loads((ROOT / lm.TASK076_MANIFEST).read_text())
    rn.sim_preflight(report, args, manifest)
    chain = rn.old_chain(args, report, ())
    readouts = rn.stage_r_readouts(args, report)
    config = rn.worker_config(chain, readouts=readouts, with_r8=True)
    config = {"torch_threads": cpv.WORKER_TORCH_THREADS,
              "p3_checkpoint": hz.p3_checkpoint(Path(args.evidence))} | config  # fmt: skip
    common = {"tau_commit_cm": cpv.TAU_COMMIT_CM, "a_lo": cpv.A_LO}
    pool = rn.Pool(int(args.workers), config)
    try:
        p_readout, encoder = hz.refit_p_readout(report, pool, Path(args.evidence), report["_run1"])
        resets = {s: cpv.reset_of(s) for s in seeds}
        est = rn.cohort_estimates(pool, p_readout, encoder, seeds, resets, 1800.0)
        est.pop("_render_disagreements", None)
    finally:
        report["pool_close"] = pool.close()
    report.pop("_run1", None)
    co = SimpleNamespace(resets=resets, est=est)

    cpr.worker_init(config)
    box: dict = {}
    demo.install_hooks(box)
    robot = rtm._W["robot"]
    mj, model = robot.mj, robot.model
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = args.lookat
    camera.distance, camera.azimuth, camera.elevation = args.distance, args.azimuth, args.elevation
    names = [model.body(i).name for i in range(model.nbody)]
    expected = {}
    if args.demo_report:
        prior = json.loads(Path(args.demo_report).read_text())
        expected = {(c["arm"], c["seed"]): c for c in prior.get("clips", [])}
    keys = (lm.MOVE_STEP, lm.COMMIT_STEP + 1)
    Logger = make_logger(demo)
    report["clips"] = []
    for k, (arm, seed) in enumerate(CLIPS, start=1):
        name = f"{k:02d}_{arm}_{seed}"
        logger = Logger(robot, arm=arm, seed=seed, stride=args.stride, camera=camera, keys=keys)
        box["recorder"] = logger
        t0 = time.monotonic()
        try:
            record = cpr.run_attempt_task(rn.attempt_tasks(arm, [seed], co, **common)[0])
        finally:
            box["recorder"] = None
        landing = demo.landing_of(record)
        target = (record.get("commit") or {}).get("target")
        steps = np.asarray(logger.rows["t"], np.int64)
        view_t = np.asarray([v[0] for v in logger.views], np.int64)
        np.savez_compressed(
            out_dir / f"{name}.npz",
            t=steps,
            xpos=np.stack(logger.rows["xpos"]),
            xquat=np.stack(logger.rows["xquat"]),
            aim=np.stack(logger.rows["aim"]),
            body_names=np.asarray(names),
            body_parent=np.asarray(model.body_parentid, np.int64),
            apple_body=np.int64(model.body("apple").id),
            plate_body=np.int64(model.body(ps.PLATE_BODY).id),
            landing=np.full(2, np.nan) if landing is None else np.asarray(landing, np.float64),
            target=np.full(2, np.nan) if target is None else np.asarray(target, np.float64),
            view_t=view_t,
            mujoco_view=np.stack([v[1] for v in logger.views]),
            onboard=np.stack([v[2] for v in logger.views]),
            success=np.bool_(record["success"]),
            final_distance_cm=np.float64(record.get("final_distance_cm") or np.nan),
            arm=np.str_(arm),
            seed=np.int64(seed),
            move_step=np.int64(lm.MOVE_STEP),
            commit_step=np.int64(lm.COMMIT_STEP),
            settle_step=np.int64(lm.RULE["s1"]),
        )  # fmt: skip
        prior = expected.get((arm, seed))
        entry = {
            "name": name, "arm": arm, "seed": seed, "success": bool(record["success"]),
            "final_distance_cm": record.get("final_distance_cm"), "target": target,
            "steps": int(steps.size), "first_step": int(steps[0]), "last_step": int(steps[-1]),
            "rendered_steps": int(view_t.size), "seconds_wall": time.monotonic() - t0,
            "matches_demo_render": None if prior is None else bool(
                prior["success"] == bool(record["success"])
                and abs(prior["final_distance_cm"] - record["final_distance_cm"]) < 1e-9),
        }  # fmt: skip
        report["clips"].append(entry)
        print(f"{name}: success={entry['success']} steps={entry['steps']} "
              f"matches_demo={entry['matches_demo_render']}", flush=True)  # fmt: skip
    report["seconds"] = time.monotonic() - started
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float))
    sums = [f"{sha256_file(p)}  {p.name}\n" for p in sorted(out_dir.iterdir()) if p.is_file()]
    (out_dir / "SHA256SUMS").write_text("".join(sums))
    return 0


# ----- compose ------------------------------------------------------------------------------------
def compose(args) -> int:
    from PIL import Image, ImageDraw

    demo = demo_module()
    dump_dir, isaac_dir, out_dir = Path(args.dump), Path(args.isaac), Path(args.output)
    if out_dir.exists():
        raise SystemExit(f"refusing: {out_dir} exists")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("system ffmpeg is required")
    out_dir.mkdir(parents=True)
    if shutil.disk_usage(out_dir).free / 2**30 < MIN_DISK_GIB:
        raise SystemExit("refusing: under 10 GiB free")
    composer = demo.Composer()
    vw, vh = demo.VIEW_W, demo.VIEW_H
    fps = int(args.fps)
    isaac_report = json.loads((isaac_dir / "run" / "report.json").read_text())

    def label(img: Image.Image, text: str, y: int, colour=(255, 210, 120)) -> None:
        draw = ImageDraw.Draw(img)
        w = draw.textlength(text, font=composer.small)
        draw.rectangle((0, y, img.width, y + 22), fill=(0, 0, 0))
        draw.text(((img.width - w) / 2, y + 3), text, font=composer.small, fill=colour)

    def isaac_frame(view, onboard, lines, banner=None) -> np.ndarray:
        frame = Image.fromarray(composer.frame(view, onboard, lines, banner))
        label(frame, LABEL, vh - 22)
        return np.asarray(frame)

    def pair_frame(mj_view, isaac_view, lines, banner=None) -> np.ndarray:
        canvas = Image.new("RGB", (2 * vw, vh + 66), (24, 26, 30))
        canvas.paste(Image.fromarray(mj_view), (0, 44))
        canvas.paste(Image.fromarray(isaac_view), (vw, 44))
        draw = ImageDraw.Draw(canvas)
        if banner is not None:
            draw.rectangle((0, 0, 2 * vw, 44), fill=banner[1])
            draw.text((12, 8), banner[0], font=composer.big, fill=(255, 255, 255))
        else:
            head = "  |  ".join(t for t, _ in lines[:1] + lines[3:5])
            draw.text((12, 4), head, font=composer.bold, fill=(235, 235, 235))
            phase = " ".join(t for t, s in lines[6:] if t and s == "normal")
            draw.text((12, 24), phase, font=composer.normal, fill=(200, 200, 200))
        draw.text((8, 48), "MuJoCo (the episode as run)", font=composer.normal,
                  fill=(255, 255, 255))  # fmt: skip
        draw.text((vw + 8, 48), "Isaac Sim (the same states, posed; no physics)",
                  font=composer.normal, fill=(255, 255, 255))  # fmt: skip
        label(canvas, LABEL, vh + 44)
        return np.asarray(canvas)

    def card() -> np.ndarray:
        return composer.card([
            ("TASK-081 demo, replayed in Isaac Sim", "big"),
            ("Isaac Sim kinematic replay of MuJoCo episodes -", "bold"),
            ("not an Isaac run of the controller.", "bold"),
            ("", "normal"),
            ("Each attempt ran in MuJoCo (apple-to-plate-v2, condition C1-M, debug", "normal"),
            ("seeds). Every frame here poses the G1 + Dex3 (our MJCF converted to USD),", "normal"),
            ("an apple and a plate in the GR00T tutorial's Arena shelf scene at", "normal"),
            ("MuJoCo's recorded joint and object poses. Isaac's physics never runs;", "normal"),
            ("the controller never saw these images. Apple and plate meshes are", "normal"),
            ("Arena's, scaled to our sizes; the Arena plate is not our blue plate.", "normal"),
            ("", "normal"),
            ("ILLUSTRATION ONLY: debug seeds, clips chosen by a fixed rule; not", "bold"),
            ("evidence. Gated result and caveats:", "normal"),
            ("docs/experiments/apple_lewm_commit_precision_v2_results.md", "normal"),
        ])  # fmt: skip

    report = {"what": "TASK-081 Isaac kinematic replay video (illustration only)",
              "argv": sys.argv[1:], "isaac_run": str(isaac_dir), "dump": str(dump_dir),
              "clips": []}  # fmt: skip
    parts = {"isaac": [], "pair": []}
    intro = card()
    intro_pair = np.asarray(Image.fromarray(intro).resize((2 * vw, vh + 66)))
    for kind, frame in (("isaac", intro), ("pair", intro_pair)):
        path = out_dir / f"00_intro_{kind}.mp4"
        video = demo.Video(path, (frame.shape[1], frame.shape[0]), fps)
        video.write(frame, 6 * fps)
        video.close()
        parts[kind].append(path)
    for clip in isaac_report["clips"]:
        name = clip["name"]
        z = np.load(dump_dir / f"{name}.npz")
        arm, seed = str(z["arm"]), int(z["seed"])
        recorder = SimpleNamespace(arm=arm, seed=seed)
        lines_of = demo.Recorder.lines.__get__(recorder)
        move, commit = int(z["move_step"]), int(z["commit_step"])
        frames_dir = isaac_dir / "run" / "frames" / name
        videos = {
            "isaac": demo.Video(out_dir / f"{name}_isaac.mp4", composer.size, fps),
            "pair": demo.Video(out_dir / f"{name}_pair.mp4", (2 * vw, vh + 66), fps),
        }
        last = None
        for i, t in enumerate(z["view_t"].tolist()):
            iv = np.asarray(Image.open(frames_dir / f"{t:05d}.png").convert("RGB"))
            mv, ob = z["mujoco_view"][i], z["onboard"][i]
            hold = 30 if t in (move, commit + 1) else 1
            videos["isaac"].write(isaac_frame(iv, ob, lines_of(t)), hold)
            videos["pair"].write(pair_frame(mv, iv, lines_of(t)), hold)
            last = (t, iv, mv, ob)
        success, distance = bool(z["success"]), float(z["final_distance_cm"])
        if success:
            banner = (f"SUCCESS in MuJoCo: at rest, {distance:.1f} cm off centre", (30, 130, 60))
        else:
            banner = (f"MISS in MuJoCo: {distance:.1f} cm off the plate centre", (170, 40, 40))
        t, iv, mv, ob = last
        videos["isaac"].write(isaac_frame(iv, ob, lines_of(t), banner), int(2.5 * fps))
        videos["pair"].write(pair_frame(mv, iv, lines_of(t), banner), int(2.5 * fps))
        for kind, video in videos.items():
            video.close()
            parts[kind].append(video.path)
        report["clips"].append({"name": name, "arm": arm, "seed": seed, "success": success,
                                "final_distance_cm": distance,
                                "frames": videos["isaac"].frames})  # fmt: skip
    for kind, files in parts.items():
        listing = out_dir / f"concat_{kind}.txt"
        listing.write_text("".join(f"file '{p.name}'\n" for p in files))
        final = out_dir / f"task081_lewm_isaac_replay_{kind}.mp4"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c", "copy", str(final)],
            check=True, cwd=out_dir,
        )  # fmt: skip
        report[f"video_{kind}"] = final.name
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float))
    sums = [f"{sha256_file(p)}  {p.name}\n" for p in sorted(out_dir.iterdir()) if p.is_file()]
    (out_dir / "SHA256SUMS").write_text("".join(sums))
    print(f"wrote {out_dir}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dump", help="re-run the demo's clips in MuJoCo and record their states")
    d.add_argument("--output", required=True)
    d.add_argument("--evidence", required=True)
    d.add_argument("--old-features", required=True)
    d.add_argument("--old-fits", required=True)
    d.add_argument("--models", nargs=6, required=True)
    d.add_argument("--stage-r", required=True)
    d.add_argument("--demo-report", help="the demo video's report.json, to check outcomes")
    d.add_argument("--workers", type=int, default=2)
    d.add_argument("--stride", type=int, default=3, help="render every k-th step")
    d.add_argument("--lookat", type=float, nargs=3, default=(0.42, -0.14, 0.80))
    d.add_argument("--distance", type=float, default=0.95)
    d.add_argument("--azimuth", type=float, default=165.0)
    d.add_argument("--elevation", type=float, default=-30.0)
    c = sub.add_parser("compose", help="caption the Isaac frames and encode the MP4s")
    c.add_argument("--dump", required=True)
    c.add_argument("--isaac", required=True, help="the run_isaac.sh output directory")
    c.add_argument("--output", required=True)
    c.add_argument("--fps", type=int, default=30)
    args = ap.parse_args(argv)
    return dump(args) if args.command == "dump" else compose(args)


if __name__ == "__main__":
    raise SystemExit(main())
