"""TASK-081 demo video: illustration only, not evidence, not a gated run.

Renders a few closed-loop attempts of TASK-081's arms (``docs/experiments/
apple_lewm_commit_precision_v2.md``) on **debug seeds** (71900-71999, whose outcomes the protocol
never reads) to an MP4 for watching. Nothing here is a result: the seeds are few, chosen by the
rule below after a screen, and the counts printed are not evidence of anything.

It reuses TASK-081's runner as a library (its artifact checks by sha256, the worker configuration,
the P readout refit and the cohort estimates; ``scripts/run_lewm_cp_v2.py`` is loaded, never
edited, and runs no stage) and TASK-081's worker (``lewm_cp_v2_runtime``) unchanged:

1. **screen**: W, N and H-rule run in closed loop on ``--seeds`` in a worker pool, no rendering;
2. **clip selection** (fixed here before the first run): seed A is the first screened seed where
   W succeeds and N misses; seed B the first other seed where W succeeds. Clips: W on B, then
   W, N and H-rule on A (fewer when no such seed exists);
3. **render**: each clip re-runs the same attempt in this process with TASK-081's worker. Only
   two classes are swapped in this process for the render (``CorpusMotion`` and ``CommandLog`` of
   ``lewm_c1m_v2_runtime``, which TASK-080's worker instantiates per attempt): the subclasses
   render an extra third-person view after each observation and read the controller's committed
   aim; they change no command, no plate motion and no observation the controller sees.

Markers: green, the place aim the arm committed at step 405; orange disc, the scorer's 4 cm radius
around where the plate stopped (step 525, from the screen run of the same attempt). The right
panel shows the onboard 112 px frame the controllers read. Frames go to system ``ffmpeg``.
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

from embodied_jepa import lewm_c1m_v2 as lm  # noqa: E402
from embodied_jepa import lewm_cp_v2 as cpv  # noqa: E402
from embodied_jepa import lewm_planner_v2 as lp  # noqa: E402

ARMS = ("W", "N", "H-rule")
ARM_TEXT = {
    "W": (
        "W: LeWM token predictor",
        "model seed 66800, frozen DINOv2,",
        "R-S readout, affine_local",
    ),
    "N": ("N: W's action-blind twin", "control: the predictor's action", "input is zeroed"),
    "H-rule": ("H-rule: hand-written rule", "not learned; given the", "simulator's plate law"),
}
VIEW_W, VIEW_H, PANEL_W, ONBOARD_PX = 640, 480, 256, 224
MIN_DISK_GIB = 10.0
SCORER_RADIUS_M = 0.04
MAP_CAP_SECONDS = 3600.0
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _runner():
    spec = importlib.util.spec_from_file_location(
        "run_lewm_cp_v2", ROOT / "scripts" / "run_lewm_cp_v2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_debug_seeds(seeds) -> None:
    low, high = cpv.DEBUG_SEEDS
    for s in seeds:
        if not low <= s <= high:
            raise lp.GuardError(f"seed {s} is outside TASK-081's debug block {low}-{high}")


def select_clips(screen: dict, seeds) -> list[tuple[str, int]]:
    """The fixed rule: A = first seed with W success and N miss; B = first other W success."""
    ok = {arm: {r["seed"]: bool(r["success"]) for r in screen[arm]} for arm in ARMS}
    a = next((s for s in seeds if ok["W"][s] and not ok["N"][s]), None)
    b = next((s for s in seeds if ok["W"][s] and s != a), None)
    clips = [] if b is None else [("W", b)]
    if a is not None:
        clips += [("W", a), ("N", a), ("H-rule", a)]
    return clips


def phase_of(t: int) -> str:
    if t < lm.MOVE_STEP:
        return "P-3 (learned policy) picks the apple"
    if t < lm.COMMIT_STEP:
        return "the plate has moved (simulation-only, step 300)"
    if t < lm.COMMIT_STEP + 20:
        return "step 405: the place aim is committed"
    if t <= lm.RULE["s1"]:
        return "e9's scripted place; the plate reacts to the hand"
    return "release and settle"


# ----- frame composition ------------------------------------------------------------------------
class Composer:
    def __init__(self):
        from PIL import ImageFont

        def font(path, size):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                return ImageFont.load_default(size=size)

        self.small, self.normal, self.bold = font(FONT, 13), font(FONT, 15), font(FONT_BOLD, 17)
        self.big = font(FONT_BOLD, 26)
        self.size = (VIEW_W + PANEL_W, VIEW_H)

    def frame(self, view, onboard, lines, banner=None) -> np.ndarray:
        from PIL import Image, ImageDraw

        canvas = Image.new("RGB", self.size, (24, 26, 30))
        canvas.paste(Image.fromarray(view), (0, 0))
        board = Image.fromarray(onboard).resize((ONBOARD_PX, ONBOARD_PX), Image.NEAREST)
        x0 = VIEW_W + (PANEL_W - ONBOARD_PX) // 2
        canvas.paste(board, (x0, 30))
        draw = ImageDraw.Draw(canvas)
        draw.text((x0, 8), "onboard camera (112 px)", font=self.small, fill=(200, 200, 200))
        y = 30 + ONBOARD_PX + 12
        for text, style in lines:
            f = {"bold": self.bold, "small": self.small}.get(style, self.normal)
            draw.text((VIEW_W + 12, y), text, font=f, fill=(235, 235, 235))
            y += 22 if style == "bold" else 19
        draw.rectangle((0, VIEW_H - 22, VIEW_W, VIEW_H), fill=(0, 0, 0))
        draw.text(
            (8, VIEW_H - 19),
            "illustration only - debug seed, not evidence, not a gated run",
            font=self.small,
            fill=(255, 210, 120),
        )
        if banner is not None:
            text, colour = banner
            draw.rectangle((0, 0, VIEW_W + PANEL_W, 44), fill=colour)
            draw.text((12, 8), text, font=self.big, fill=(255, 255, 255))
        return np.asarray(canvas, np.uint8)

    def card(self, lines) -> np.ndarray:
        from PIL import Image, ImageDraw

        canvas = Image.new("RGB", self.size, (24, 26, 30))
        draw = ImageDraw.Draw(canvas)
        y = 60
        for text, style in lines:
            f = {"big": self.big, "bold": self.bold, "small": self.small}.get(style, self.normal)
            draw.text((40, y), text, font=f, fill=(235, 235, 235))
            y += 40 if style == "big" else 26
        return np.asarray(canvas, np.uint8)


class Video:
    """Raw RGB frames piped to system ffmpeg (H.264, yuv420p)."""

    def __init__(self, path: Path, size, fps: int):
        self.path = path
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )  # fmt: skip
        self.frames = 0

    def write(self, frame: np.ndarray, repeat: int = 1) -> None:
        data = np.ascontiguousarray(frame, np.uint8).tobytes()
        for _ in range(repeat):
            self.proc.stdin.write(data)
            self.frames += 1

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path}")


# ----- the render hooks (this process only) ------------------------------------------------------
class Recorder:
    """Per clip: renders the third-person view after each kept observation, with the markers."""

    def __init__(self, robot, composer, video, *, arm, seed, landing, stride, camera):
        self.robot, self.composer, self.video = robot, composer, video
        self.arm, self.seed, self.landing = arm, int(seed), landing
        self.stride, self.camera = int(stride), camera
        self.log = None
        self.last = None
        mj = robot.mj
        self.renderer = mj.Renderer(robot.model, height=VIEW_H, width=VIEW_W)

    def aim(self):
        decisions = getattr(getattr(self.log, "inner", None), "decisions", None) or []
        return None if not decisions else np.asarray(decisions[0]["target"], np.float64)

    def _marker(self, geom_type, size, pos, rgba):
        mj, scene = self.robot.mj, self.renderer.scene
        if scene.ngeom >= scene.maxgeom:
            return
        mj.mjv_initGeom(
            scene.geoms[scene.ngeom], geom_type, np.asarray(size, np.float64),
            np.asarray(pos, np.float64), np.eye(3).flatten(), np.asarray(rgba, np.float32),
        )  # fmt: skip
        scene.ngeom += 1

    def render(self, t: int) -> np.ndarray:
        from embodied_jepa import plate_shift as ps

        mj, data = self.robot.mj, self.robot.sim.data
        self.renderer.update_scene(data, camera=self.camera)
        z = float(data.body(ps.PLATE_BODY).xpos[2])
        if self.landing is not None and t >= lm.COMMIT_STEP:
            self._marker(mj.mjtGeom.mjGEOM_CYLINDER, [SCORER_RADIUS_M, 0.0015, 0],
                         [*self.landing, z + 0.03], [1.0, 0.55, 0.0, 0.45])  # fmt: skip
        g = self.aim()
        if g is not None:
            self._marker(mj.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0], [*g, z + 0.14],
                         [0.1, 0.95, 0.2, 1.0])  # fmt: skip
            self._marker(mj.mjtGeom.mjGEOM_CYLINDER, [0.0025, 0.07, 0], [*g, z + 0.07],
                         [0.1, 0.95, 0.2, 0.8])  # fmt: skip
        return self.renderer.render().copy()

    def lines(self, t: int):
        title, sub, sub2 = ARM_TEXT[self.arm]
        out = [(title, "bold"), (sub, "small"), (sub2, "small"),
               (f"reset seed {self.seed} (debug)", "normal"), (f"step {t}", "normal"),
               ("", "small")]  # fmt: skip
        phase = phase_of(t)
        words, line = phase.split(), ""
        for w in words:  # wrap at about 30 characters
            if len(line) + len(w) > 30:
                out.append((line.strip(), "normal"))
                line = ""
            line += w + " "
        out.append((line.strip(), "normal"))
        out += [
            ("", "small"),
            ("green: committed place aim", "small"),
            ("orange: 4 cm around the plate's", "small"),
            ("final position", "small"),
        ]
        return out  # fmt: skip

    def observed(self, t: int, observation) -> None:
        from embodied_jepa import first_policy_v2 as fp2

        onboard = np.asarray(observation.images[fp2.CAMERA][0], np.uint8)
        key = t in (lm.MOVE_STEP, lm.COMMIT_STEP + 1)
        if t % self.stride and not key:
            return
        frame = self.composer.frame(self.render(t), onboard, self.lines(t))
        hold = 1
        if t == lm.MOVE_STEP or t == lm.COMMIT_STEP + 1:  # pause on the move and the commit
            hold = 30
        self.video.write(frame, hold)
        self.last = (t, onboard)


def install_hooks(recorder_box: dict) -> None:
    from embodied_jepa import lewm_c1m_v2_runtime as wrt

    base_motion, base_log = wrt.CorpusMotion, wrt.CommandLog

    class DemoMotion(base_motion):
        def _observe(self):
            t = int(self.state["calls"])
            observation = super()._observe()
            recorder = recorder_box.get("recorder")
            if recorder is not None:
                recorder.observed(t, observation)
            return observation

    class DemoLog(base_log):
        def __init__(self, inner):
            super().__init__(inner)
            recorder = recorder_box.get("recorder")
            if recorder is not None:
                recorder.log = self

    wrt.CorpusMotion, wrt.CommandLog = DemoMotion, DemoLog


# ----- main ---------------------------------------------------------------------------------------
def landing_of(record: dict):
    path = (record.get("kpred") or {}).get("plate_path") or {}
    xy = path.get(str(lm.RULE["s1"]))
    return None if xy is None else np.asarray(xy, np.float64)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--old-features", required=True)
    ap.add_argument("--old-fits", required=True)
    ap.add_argument("--models", nargs=6, required=True)
    ap.add_argument("--stage-r", required=True)
    ap.add_argument("--first", type=int, default=71920)
    ap.add_argument("--last", type=int, default=71931)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stride", type=int, default=3, help="render every k-th step")
    ap.add_argument("--fps", type=int, default=30)
    # free camera: a third-person view of the table from the front right
    ap.add_argument("--lookat", type=float, nargs=3, default=(0.42, -0.14, 0.80))
    ap.add_argument("--distance", type=float, default=0.95)
    ap.add_argument("--azimuth", type=float, default=165.0)
    ap.add_argument("--elevation", type=float, default=-30.0)
    args = ap.parse_args(argv)
    out_dir = Path(args.output)
    if out_dir.exists():
        raise SystemExit(f"refusing: {out_dir} exists")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("system ffmpeg is required")
    seeds = tuple(range(args.first, args.last + 1))
    check_debug_seeds(seeds)
    out_dir.mkdir(parents=True)
    if shutil.disk_usage(out_dir).free / 2**30 < MIN_DISK_GIB:
        raise SystemExit("refusing: under 10 GiB free")
    rn = _runner()
    from embodied_jepa import lewm_cp_v2_runtime as cpr
    from embodied_jepa import plate_twin_v2_harness as hz
    from embodied_jepa import wm_critic_v2_runtime as rtm

    started = time.monotonic()
    report = {
        "what": "TASK-081 demo video: illustration only, not evidence, not a gated run",
        "revision": hz.revision(),
        "tracked_tree_dirty": bool(hz.tracked_tree_dirty()),
        "argv": sys.argv[1:] if argv is None else list(argv),
        "seeds": list(seeds),
        "selection_rule": select_clips.__doc__,
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
        co = SimpleNamespace(resets=resets, est=est)
        screen = {}
        for arm in ARMS:
            screen[arm] = rn.run_arm(pool, rn.attempt_tasks(arm, seeds, co, **common),
                                     f"screen {arm}")  # fmt: skip
    finally:
        report["pool_close"] = pool.close()
    report.pop("_run1", None)
    report["screen"] = {
        arm: [
            {
                "seed": r["seed"],
                "success": bool(r["success"]),
                "final_distance_cm": r.get("final_distance_cm"),
                "target": (r.get("commit") or {}).get("target"),
                "landing": None if landing_of(r) is None else landing_of(r).tolist(),
            }
            for r in recs
        ]
        for arm, recs in screen.items()
    }
    clips = select_clips(screen, seeds)
    report["clips"] = []
    if not clips:
        (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float))
        raise SystemExit("no seed met the selection rule; see report.json")

    cpr.worker_init(config)  # the same worker, in this process, for the render
    box: dict = {}
    install_hooks(box)
    robot = rtm._W["robot"]
    camera = robot.mj.MjvCamera()
    camera.type = robot.mj.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = args.lookat
    camera.distance, camera.azimuth, camera.elevation = args.distance, args.azimuth, args.elevation
    composer = Composer()
    by_key = {(arm, r["seed"]): r for arm, recs in screen.items() for r in recs}
    parts = []
    intro = out_dir / "00_intro.mp4"
    video = Video(intro, composer.size, args.fps)
    video.write(composer.card([
        ("TASK-081 demo: LeWM picks the place aim", "big"),
        ("apple-to-plate-v2, condition C1-M, MuJoCo, simulation only", "normal"),
        ("", "normal"),
        ("P-3 (learned behaviour cloning) picks the apple; the plate then moves", "normal"),
        ("(simulation-only); at step 405 one place aim is committed; e9's scripted", "normal"),
        ("place puts the apple there while the plate reacts to the hand.", "normal"),
        ("", "normal"),
        ("W chooses that aim with a LeWM token predictor on frozen DINOv2 features.", "normal"),
        ("N (its action-blind twin) and H-rule (hand-written, given the plate law)", "normal"),
        ("are shown on the same reset for contrast.", "normal"),
        ("", "normal"),
        ("ILLUSTRATION ONLY: debug seeds 71900-71999, clips chosen by a fixed rule", "bold"),
        ("after a screen; not evidence. For the gated result and its caveats see", "normal"),
        ("docs/experiments/apple_lewm_commit_precision_v2_results.md", "normal"),
    ]), 5 * args.fps)  # fmt: skip
    video.close()
    parts.append(intro)
    for k, (arm, seed) in enumerate(clips, start=1):
        screened = by_key[(arm, seed)]
        path = out_dir / f"{k:02d}_{arm}_{seed}.mp4"
        video = Video(path, composer.size, args.fps)
        recorder = Recorder(robot, composer, video, arm=arm, seed=seed,
                            landing=landing_of(screened), stride=args.stride,
                            camera=camera)  # fmt: skip
        box["recorder"] = recorder
        t0 = time.monotonic()
        try:
            record = cpr.run_attempt_task(rn.attempt_tasks(arm, [seed], co, **common)[0])
        finally:
            box["recorder"] = None
        success = bool(record["success"])
        distance = record.get("final_distance_cm")
        if success:
            d = "" if distance is None else f", {distance:.1f} cm off centre"
            banner = (f"SUCCESS: at rest on the plate{d}", (30, 130, 60))
        else:
            d = "" if distance is None else f": {distance:.1f} cm off the plate centre"
            banner = (f"MISS{d}", (170, 40, 40))
        if recorder.last is not None:
            t, onboard = recorder.last
            final = composer.frame(recorder.render(t), onboard, recorder.lines(t), banner)
            video.write(final, int(2.5 * args.fps))
        video.close()
        parts.append(path)
        agrees = success == bool(screened["success"]) and (
            (record.get("commit") or {}).get("target") == (screened.get("commit") or {}).get(
                "target")
        )  # fmt: skip
        report["clips"].append({
            "file": path.name, "arm": arm, "seed": seed, "success": success,
            "final_distance_cm": distance, "frames": video.frames,
            "seconds_video": video.frames / args.fps, "seconds_wall": time.monotonic() - t0,
            "matches_screen_run": bool(agrees),
        })  # fmt: skip
        print(f"clip {path.name}: success={success} ({video.frames} frames)", flush=True)
    listing = out_dir / "concat.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in parts))
    final = out_dir / "task081_lewm_demo.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c", "copy", str(final)],
        check=True, cwd=out_dir,
    )  # fmt: skip
    report["video"] = final.name
    report["seconds"] = time.monotonic() - started
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float))
    sums = [f"{sha256_file(p)}  {p.name}\n" for p in sorted(out_dir.iterdir()) if p.is_file()]
    (out_dir / "SHA256SUMS").write_text("".join(sums))
    print(f"wrote {final} in {report['seconds']:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
