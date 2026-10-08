"""TASK-081's controller in closed loop in Isaac Sim: development only, not evidence.

Host side of ``scripts/isaac/lewm_cp_server_isaac.py`` (container). **Not a gated run, not
evidence, changes no count.** Debug seeds only (71900-71999, whose outcomes TASK-081 never reads).

What runs where:

* **Isaac** (the container) steps the physics (PhysX or Isaac Lab's Newton backend, per the
  server's ``--physics``) of our G1 + Dex3 (the MJCF conversion) with the v2 table, apple and
  plate, and renders the onboard 112 px frame with the pinned camera manifest (same mount, pose
  and intrinsics as MuJoCo's ``onboard_rgb``; path traced, so the shading differs: ISAAC_V2_SCENE
  §1). Every image any controller reads is Isaac's.
* **The host** runs TASK-081's worker unchanged (``lewm_cp_v2_runtime.run_attempt_task``: P-3's
  pick, the C1-M plate law in ``CorpusMotion``, W's or H-rule's single aim at 405, e9's scripted
  place) on an ``isaac_e9.MirrorSimulation`` whose physics is Isaac's (``RenderMirror``). The
  plate law teleports the plate in the mirror; the mirror forwards every plate move to Isaac
  before the next render or physics step. ``render()`` returns Isaac's frame.

Two things in TASK-081's worker cannot hold in Isaac and are replaced in this process only:

1. ``wm_critic_v2_runtime.check_post_look_frame`` compares the post-look frame with MuJoCo's
   sha256; here it instead computes P-3's post-look estimates from **Isaac's** post-look frame
   (the M2 P readout, refitted as in the demo) and puts them into the task. For the domain-gap
   record it also computes them from MuJoCo's rendering of the same Isaac state (never used).
2. ``wall_seconds`` is raised (Isaac is slower than MuJoCo).

Step A (``--step-a-seeds``): reset + look per seed, then the C1-M move teleported at the post-look
state; R-plate's reading and P-3's estimates on Isaac's frame and on MuJoCo's rendering of the
same state, against the true positions. Step B (``--seeds``): closed-loop attempts of ``--arms``
with a captioned video; at 405 also W's aim and R-plate's reading on MuJoCo's rendering of the
same state (counterfactual, logged, never used). ``--pick e9`` replaces P-3's pick by e9's
privileged scripted pick (a labelled variant). ``--sham`` runs the same client against a host
MuJoCo endpoint (a plumbing check: it must reproduce plain MuJoCo).
"""

from __future__ import annotations

import os

for _key, _value in {  # overridable: the machine may be shared (BLAS oversubscription)
    "MKL_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "6",
    "MKL_NUM_THREADS": "6",
    "OPENBLAS_NUM_THREADS": "16",
}.items():
    os.environ.setdefault(_key, _value)

import argparse  # noqa: E402
import dataclasses  # noqa: E402
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

from embodied_jepa import isaac_e9 as ie  # noqa: E402
from embodied_jepa import lewm_c1m_v2 as lm  # noqa: E402
from embodied_jepa import lewm_cp_v2 as cpv  # noqa: E402
from embodied_jepa.contracts import ContractError  # noqa: E402
from embodied_jepa.simulation import MuJoCoSimulation  # noqa: E402

LABEL = (
    "Isaac Sim closed-loop, development only, not evidence; LeWM chooses only the place aim;"
    " trained on MuJoCo images"
)
ARM_TEXT = {
    "W": ("W: LeWM token predictor", "seed 66800, frozen DINOv2, R-S, affine_local"),
    "H-rule": ("H-rule: hand-written rule", "not learned; given the plate law"),
    "N": ("N: W's action-blind twin", "control"),
}
VIEW_W, VIEW_H, SIDE_W, BAR_H = 640, 480, 480, 56
BOARD = 224
MIN_DISK_GIB = 10.0
SCORER_RADIUS_M = 0.04
WALL_SECONDS = 3600.0
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
VARIANT = {
    ("p3", "isaac"): "pick: P-3 (learned), estimates from Isaac's frame",
    ("p3", "mujoco"): "DIAGNOSTIC: P-3's estimates from MuJoCo's render",
    ("e9", "isaac"): "VARIANT: e9's scripted privileged pick",
    ("e9", "mujoco"): "VARIANT: e9's scripted privileged pick",
}
THIRD = {"eye": (1.2147, -0.3529, 1.275), "target": (0.42, -0.14, 0.80), "fovy": 45.0}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log(msg: str) -> None:
    print(f"[isaac-cp {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----- endpoints ----------------------------------------------------------------------------------
class SocketEndpoint:
    """The mirror endpoint over the container server's Unix socket (``e9_replay``'s, plus
    ``call`` for ``set_plate`` and ``render``)."""

    def __init__(self, path: Path, timeout_s: float):
        from multiprocessing.connection import Client

        deadline = time.monotonic() + timeout_s
        key_path = path.parent / "authkey"
        while not (path.exists() and key_path.exists()):
            if time.monotonic() > deadline:
                raise SystemExit(f"no server socket at {path} after {timeout_s:.0f} s")
            time.sleep(2.0)
        self.conn = Client(str(path), family="AF_UNIX", authkey=bytes.fromhex(key_path.read_text()))
        self.hello = self.call("hello")

    def call(self, cmd: str, **kwargs) -> dict:
        self.conn.send({"cmd": cmd, **kwargs})
        reply = self.conn.recv()
        if not reply.get("ok"):
            raise RuntimeError(f"server {cmd}: {reply.get('error')}\n{reply.get('traceback')}")
        return reply

    def reset(self, **kwargs) -> dict:
        return self.call("reset", **kwargs)["state"]

    def step(self, targets, deadline) -> dict:
        reply = self.call("step", targets=targets, deadline=deadline)
        return {"ack": reply["ack"], "state": reply["state"]}

    def stop(self, reason) -> dict:
        return self.call("stop", reason=reason)["state"]

    def close(self) -> None:
        try:
            self.call("close")
        finally:
            self.conn.close()


class ShamEndpoint(ie.MuJoCoEndpoint):
    """``isaac_e9.MuJoCoEndpoint`` with rendering (112 px, as ``make_v2_robot``) and the plate
    teleport: with it the mirror must reproduce a plain MuJoCo attempt (plumbing check)."""

    def __init__(self):
        from embodied_jepa import apple_to_plate_v2 as v2
        from embodied_jepa import first_policy as fp

        self.sim = MuJoCoSimulation(
            object_kind="apple", container_kind="plate", width=fp.IMAGE_SIZE, height=fp.IMAGE_SIZE
        )
        self.scene = v2.apply_v2_scene(self.sim.model)
        self.sim.render()
        self.hello = {"physics": "sham-mujoco"}

    def call(self, cmd: str, **kwargs) -> dict:
        if cmd == "set_plate":
            self.sim.model.body("plate").pos[:] = kwargs["xyz"]
            self.sim.mj.mj_forward(self.sim.model, self.sim.data)
            return {"plate_pos": self.sim.model.body("plate").pos.tolist()}
        if cmd == "render":
            rgb = self.sim.render()
            out = {"rgb": rgb.tobytes(), "shape": list(rgb.shape)}
            if kwargs.get("third"):
                out |= {"third": np.zeros((VIEW_H, VIEW_W, 3), np.uint8).tobytes(),
                        "third_shape": [VIEW_H, VIEW_W, 3]}  # fmt: skip
            return out
        if cmd in ("info", "close"):
            return {}
        raise ValueError(cmd)

    def close(self) -> None:
        return None


# ----- the mirror ---------------------------------------------------------------------------------
class RenderMirror(ie.MirrorSimulation):
    """``MirrorSimulation`` whose onboard frames are the remote simulator's and whose plate moves
    are forwarded to it. ``mujoco_render()`` is MuJoCo's rendering of the same mirrored state
    (domain-gap record and video only; no controller reads it)."""

    def __init__(self, endpoint, **kwargs):
        self.want_third = False
        self.last_third = None
        self.last_mujoco = None
        self.renders = 0
        self.plate_pushes = 0
        super().__init__(endpoint, render=True, **kwargs)

    def _push_plate(self) -> None:
        if self._remote is None:
            return
        local = np.asarray(self.model.body("plate").pos, np.float64).copy()
        if np.abs(local - self._remote["plate_pos"]).max() <= 1e-12:
            return
        got = np.asarray(self.endpoint.call("set_plate", xyz=local.tolist())["plate_pos"], float)
        if np.abs(got - local).max() > 1e-5:
            raise ContractError(f"remote plate at {got}, not {local}")
        self._remote["plate_pos"] = local
        self.plate_pushes += 1

    def mujoco_render(self) -> np.ndarray:
        return MuJoCoSimulation.render(self, "onboard_rgb")

    def render(self, camera="onboard_rgb"):
        if camera != "onboard_rgb":
            raise ContractError("only the onboard camera is rendered remotely")
        self._sync()
        self._push_plate()
        third = bool(self.want_third)
        reply = self.endpoint.call("render", third=third)
        rgb = np.frombuffer(reply["rgb"], np.uint8).reshape(reply["shape"]).copy()
        if rgb.shape != (self.height, self.width, 3):
            raise ContractError(f"remote frame {rgb.shape} is not {self.height}x{self.width}")
        if third:
            self.last_third = np.frombuffer(reply["third"], np.uint8).reshape(reply["third_shape"])
            self.last_mujoco = self.mujoco_render()
        self.renders += 1
        return rgb

    def send_joint_targets(self, targets, *, joint_names, deadline):
        self._sync()
        self._push_plate()
        return super().send_joint_targets(targets, joint_names=joint_names, deadline=deadline)


def make_robot(endpoint):
    """``G1Embodiment`` over a v2 ``RenderMirror`` (112 px), as ``rt2.make_robot`` builds it."""
    from embodied_jepa import apple_to_plate_v2 as v2
    from embodied_jepa import first_policy as fp
    from embodied_jepa import first_policy_v2 as fp2
    from embodied_jepa.embodiment import G1Embodiment

    sim = RenderMirror(
        endpoint,
        object_kind="apple",
        container_kind="plate",
        width=fp.IMAGE_SIZE,
        height=fp.IMAGE_SIZE,
    )
    record = v2.apply_v2_scene(sim.model)
    if record["scene_version"] != fp2.SCENE_VERSION:
        raise ContractError(f"not apple-to-plate-v2: {record}")
    robot = G1Embodiment(sim)
    robot.sim.render()
    return robot


# ----- readings for the domain gap ----------------------------------------------------------------
class Readers:
    def __init__(self, p_readout, encoder):
        self.p_readout, self.encoder = p_readout, encoder

    def estimates(self, frame) -> np.ndarray:
        """P-3's post-look estimates from one frame (the runner's streamed estimate, one row)."""
        from embodied_jepa import first_policy_perception as fpp

        tokens = fpp.featurise(self.encoder, np.asarray(frame, np.uint8)[None])
        return np.asarray(RN.streamed_estimates(self.p_readout, [tokens])[0], np.float64)

    @staticmethod
    def r_plate(frame) -> np.ndarray:
        from embodied_jepa import lewm_c1m_v2_runtime as wrt
        from embodied_jepa import plate_twin_v2_runtime as ptr

        tokens, _pooled = wrt.encode(frame)
        return np.asarray(ptr._readout("r_plate").predict(tokens), np.float64).reshape(2)


def cm(a, b) -> float:
    return 100.0 * float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def gap_row(readers, robot, frame_isaac, *, estimates: bool) -> dict:
    sim = robot.sim
    frame_mj = sim.mujoco_render()
    plate = np.asarray(sim.model.body("plate").pos[:2], float)
    apple = np.asarray(sim.data.body("apple").xpos[:2], float)
    row = {
        "plate_true": plate.tolist(),
        "apple_true": apple.tolist(),
        "mad_isaac_vs_mujoco": float(np.abs(frame_isaac.astype(float) - frame_mj).mean()),
    }
    for name, frame in (("isaac", frame_isaac), ("mujoco", frame_mj)):
        p = readers.r_plate(frame)
        row[f"r_plate_{name}"] = p.tolist()
        row[f"r_plate_err_cm_{name}"] = cm(p, plate)
        if estimates:
            e = readers.estimates(frame)
            row[f"p3_estimates_{name}"] = e.tolist()
            row[f"p3_apple_err_cm_{name}"] = cm(e[:2], apple)
            row[f"p3_plate_err_cm_{name}"] = cm(e[2:4], plate)
    return row, frame_mj


def save_png(path: Path, frame) -> None:
    from PIL import Image

    Image.fromarray(np.asarray(frame, np.uint8)).save(path)


def step_a(robot, readers, seeds, out_dir: Path) -> dict:
    """Reset + look per seed (post-look state), then the C1-M move teleported at that state."""
    from embodied_jepa import first_policy_v2_runtime as rt2
    from embodied_jepa import lewm_pr_v2 as pr
    from embodied_jepa import plate_shift as ps

    frames = out_dir / "step_a_frames"
    frames.mkdir()
    rows = []
    for seed in seeds:
        reset = cpv.reset_of(seed)
        _truth, _scorer, counter, obs, _facts = rt2.reset_and_look(robot, seed, reset)
        counter.remove()
        frame = np.asarray(obs.images["onboard_rgb"][0], np.uint8)
        row, mj = gap_row(readers, robot, frame, estimates=True)
        row |= {"seed": seed, "state": "post_look"}
        rows.append(row)
        save_png(frames / f"{seed}_post_look_isaac.png", frame)
        save_png(frames / f"{seed}_post_look_mujoco.png", mj)
        ps.move_plate(robot.sim, pr.move_offset(seed))
        frame2 = robot.sim.render()
        row2, mj2 = gap_row(readers, robot, frame2, estimates=False)
        row2 |= {"seed": seed, "state": "post_look_plate_moved"}
        rows.append(row2)
        save_png(frames / f"{seed}_moved_isaac.png", frame2)
        save_png(frames / f"{seed}_moved_mujoco.png", mj2)
        robot.stop("step_a")
        log(
            f"step A {seed}: R-plate {row['r_plate_err_cm_isaac']:.2f} cm (Isaac) / "
            f"{row['r_plate_err_cm_mujoco']:.2f} cm (MuJoCo render); moved "
            f"{row2['r_plate_err_cm_isaac']:.2f} / {row2['r_plate_err_cm_mujoco']:.2f}; "
            f"P-3 apple {row['p3_apple_err_cm_isaac']:.2f} / {row['p3_apple_err_cm_mujoco']:.2f}"
        )
    return {"rows": rows, "summary": summarise_gap(rows)}


def summarise_gap(rows) -> dict:
    out = {}
    keys = sorted({k for r in rows for k in r if k.endswith(("_isaac", "_mujoco")) and "err" in k})
    keys.append("mad_isaac_vs_mujoco")
    for k in keys:
        v = np.asarray([r[k] for r in rows if k in r], float)
        if len(v):
            out[k] = {"n": len(v), "median": float(np.median(v)), "max": float(v.max()),
                      "min": float(v.min())}  # fmt: skip
    return out


# ----- video --------------------------------------------------------------------------------------
def project(points, size=(VIEW_W, VIEW_H)):
    eye, target = np.asarray(THIRD["eye"], float), np.asarray(THIRD["target"], float)
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    w, h = size
    f = (h / 2.0) / np.tan(np.deg2rad(THIRD["fovy"]) / 2.0)
    d = np.asarray(points, float) - eye
    z = d @ fwd
    return np.stack([w / 2 + f * (d @ right) / z, h / 2 - f * (d @ up) / z], axis=1)


class Composer:
    def __init__(self):
        from PIL import ImageFont

        def font(path, size):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                return ImageFont.load_default(size=size)

        self.small, self.normal, self.bold = font(FONT, 13), font(FONT, 15), font(FONT_BOLD, 17)
        self.big = font(FONT_BOLD, 24)
        self.size = (VIEW_W + SIDE_W, VIEW_H + BAR_H)

    def frame(self, view, onboard, onboard_mj, lines, *, aim=None, plate=None, plate_z=0.746,
              banner=None, third_ok=True) -> np.ndarray:  # fmt: skip
        from PIL import Image, ImageDraw

        canvas = Image.new("RGB", self.size, (24, 26, 30))
        view_img = Image.fromarray(np.asarray(view, np.uint8))
        draw_v = ImageDraw.Draw(view_img)
        if third_ok and plate is not None:
            ring = [[plate[0] + SCORER_RADIUS_M * np.cos(a), plate[1] + SCORER_RADIUS_M * np.sin(a),
                     plate_z + 0.012] for a in np.linspace(0, 2 * np.pi, 49)]  # fmt: skip
            uv = project(ring)
            draw_v.line([tuple(p) for p in uv], fill=(255, 140, 0), width=2)
        if third_ok and aim is not None:
            stem = project([[aim[0], aim[1], plate_z + 0.012], [aim[0], aim[1], plate_z + 0.14]])
            draw_v.line([tuple(stem[0]), tuple(stem[1])], fill=(30, 240, 60), width=3)
            x, y = stem[1]
            draw_v.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(30, 240, 60))
        canvas.paste(view_img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), "Isaac Sim (Isaac physics + Isaac render)", font=self.small,
                  fill=(255, 255, 255))  # fmt: skip
        x0 = VIEW_W + 12
        draw.text((x0, 4), "Isaac onboard 112 px", font=self.small, fill=(220, 220, 220))
        draw.text((x0, 18), "(what the controllers read)", font=self.small, fill=(220, 220, 220))
        board = Image.fromarray(np.asarray(onboard, np.uint8)).resize((BOARD, BOARD), Image.NEAREST)
        canvas.paste(board, (x0, 36))
        x1 = x0 + BOARD + 8
        draw.text((x1, 4), "MuJoCo render, same state", font=self.small, fill=(160, 160, 160))
        draw.text((x1, 18), "(not read; domain gap)", font=self.small, fill=(160, 160, 160))
        if onboard_mj is not None:
            mj = Image.fromarray(np.asarray(onboard_mj, np.uint8)).resize(
                (BOARD, BOARD), Image.NEAREST
            )
            canvas.paste(mj, (x1, 36))
        y = 36 + BOARD + 10
        for text, style in lines:
            f = {"bold": self.bold, "small": self.small}.get(style, self.normal)
            draw.text((x0, y), text, font=f, fill=(235, 235, 235))
            y += 21 if style == "bold" else 18
        draw.rectangle((0, VIEW_H, self.size[0], self.size[1]), fill=(0, 0, 0))
        words, line, rows = LABEL.split(" "), "", []
        for w in words:
            if len(line) + len(w) > 95:
                rows.append(line.strip())
                line = ""
            line += w + " "
        rows.append(line.strip())
        for k, text in enumerate(rows[:2]):
            draw.text((10, VIEW_H + 8 + 20 * k), text, font=self.normal, fill=(255, 210, 120))
        if banner is not None:
            text, colour = banner
            draw.rectangle((0, 0, VIEW_W, 40), fill=colour)
            draw.text((10, 10), text, font=self.bold, fill=(255, 255, 255))
        return np.asarray(canvas, np.uint8)

    def card(self, lines) -> np.ndarray:
        from PIL import Image, ImageDraw

        canvas = Image.new("RGB", self.size, (24, 26, 30))
        draw = ImageDraw.Draw(canvas)
        y = 50
        for text, style in lines:
            f = {"big": self.big, "bold": self.bold, "small": self.small}.get(style, self.normal)
            draw.text((40, y), text, font=f, fill=(235, 235, 235))
            y += 38 if style == "big" else 25
        return np.asarray(canvas, np.uint8)


def phase_of(t: int, pick: str) -> str:
    if t < lm.MOVE_STEP:
        return ("P-3 (learned BC policy)" if pick == "p3" else "e9 (SCRIPTED, privileged)") + (
            " picks the apple"
        )
    if t < lm.COMMIT_STEP:
        return "the plate has moved (simulation-only C1-M move, step 300)"
    if t < lm.COMMIT_STEP + 20:
        return "step 405: the single place aim is committed"
    if t <= lm.RULE["s1"]:
        return "e9's scripted place; the plate reacts to the hand"
    return "release and settle"


class Recorder:
    def __init__(self, robot, composer, video, *, arm, seed, stride, pick, physics, variant=""):
        self.variant = variant
        self.robot, self.composer, self.video = robot, composer, video
        self.arm, self.seed, self.stride, self.pick = arm, int(seed), int(stride), pick
        self.physics = physics
        self.log = None
        self.last = None
        self.gap405 = None

    def aim(self):
        decisions = getattr(getattr(self.log, "inner", None), "decisions", None) or []
        return None if not decisions else np.asarray(decisions[0]["target"], np.float64)

    def wants(self, t: int) -> bool:
        return t % self.stride == 0 or t in (lm.MOVE_STEP, lm.COMMIT_STEP + 1)

    def lines(self, t: int):
        sim = self.robot.sim
        plate = np.asarray(sim.model.body("plate").pos[:2], float)
        title, sub = ARM_TEXT.get(self.arm, (self.arm, ""))
        out = [(title, "bold"), (sub, "small"), (self.variant, "small"),
               (f"debug seed {self.seed}, step {t}, physics: {self.physics}", "small")]  # fmt: skip
        words, line = phase_of(t, self.pick).split(), ""
        for w in words:
            if len(line) + len(w) > 52:
                out.append((line.strip(), "normal"))
                line = ""
            line += w + " "
        out.append((line.strip(), "normal"))
        out.append((f"true plate (x, y): ({plate[0]:.3f}, {plate[1]:.3f}) m", "small"))
        g = self.aim()
        if g is not None:
            out.append((f"committed aim (x, y): ({g[0]:.3f}, {g[1]:.3f}) m", "small"))
        if self.gap405 is not None:
            out.append((f"R-plate at 405: {self.gap405['r_plate_err_cm_isaac']:.1f} cm off "
                        f"(MuJoCo render {self.gap405['r_plate_err_cm_mujoco']:.1f})",
                        "small"))  # fmt: skip
        out.append(("green: committed aim; orange: 4 cm around the plate", "small"))
        return out

    def compose(self, t, onboard, banner=None):
        sim = self.robot.sim
        plate = np.asarray(sim.model.body("plate").pos, float)
        return self.composer.frame(
            sim.last_third, onboard, sim.last_mujoco, self.lines(t), aim=self.aim(),
            plate=plate[:2] if t >= lm.COMMIT_STEP else None, plate_z=float(plate[2]),
            banner=banner, third_ok=sim.last_third is not None and sim.last_third.any(),
        )  # fmt: skip

    def observed(self, t: int, observation) -> None:
        onboard = np.asarray(observation.images["onboard_rgb"][0], np.uint8)
        if not self.wants(t):
            return
        hold = 30 if t in (lm.MOVE_STEP, lm.COMMIT_STEP + 1) else 1
        self.video.write(self.compose(t, onboard), hold)
        self.last = (t, onboard)


def install_hooks(box: dict, readers) -> None:
    """Per attempt: the video, the third-person render request, and the 405 domain-gap record."""
    from embodied_jepa import lewm_c1m_v2_runtime as wrt
    from embodied_jepa import lewm_pr_v2_runtime as prt

    base_motion, base_log = wrt.CorpusMotion, wrt.CommandLog

    class Motion(base_motion):
        def _observe(self):
            t = int(self.state["calls"])
            recorder = box.get("recorder")
            sim = self.robot.sim
            sim.want_third = recorder is not None and recorder.wants(t)
            try:
                observation = super()._observe()
            finally:
                sim.want_third = False
            if t == lm.COMMIT_STEP:
                frame = np.asarray(observation.images["onboard_rgb"][0], np.uint8)
                row, mj = gap_row(readers, self.robot, frame, estimates=False)
                box["gap405"] = row
                box["frame405_mujoco"] = mj
                if recorder is not None:
                    recorder.gap405 = row
            if recorder is not None:
                recorder.observed(t, observation)
            return observation

    class Log(base_log):
        def __init__(self, inner):
            super().__init__(inner)
            recorder = box.get("recorder")
            if recorder is not None:
                recorder.log = self

    wrt.CorpusMotion, wrt.CommandLog = Motion, Log

    original = prt.PredReadoutAim.__call__

    def counterfactual(self, ctl, observation, step, record):
        """W's aim on MuJoCo's rendering of the same 405 state (logged, never used)."""
        mj = box.get("frame405_mujoco")
        if mj is not None and self.arm != "L-rand":
            images = dict(observation.images)
            images["onboard_rgb"] = np.asarray(mj, np.uint8)[None]
            obs_mj = dataclasses.replace(observation, images=images)
            scratch: dict = {}
            readings = list(self.readings)
            g_mj = original(self, ctl, obs_mj, step, scratch)
            self.readings[:] = readings
            box["counterfactual"] = {
                "target_on_mujoco_frame": np.asarray(g_mj, float).tolist(),
                "reading_on_mujoco_frame": scratch.get("reading"),
            }
        return original(self, ctl, observation, step, record)

    prt.PredReadoutAim.__call__ = counterfactual


def install_estimates(box: dict, readers, source: str = "isaac") -> None:
    """Replace the post-look frame check by P-3's estimates from Isaac's post-look frame."""
    from embodied_jepa import first_policy_v2_runtime as rt2
    from embodied_jepa import wm_critic_v2_runtime as rtm

    def post_look(robot, task, frame):
        frame = np.asarray(frame, np.uint8)
        row, _mj = gap_row(readers, robot, frame, estimates=True)
        task["estimates"] = row[f"p3_estimates_{source}"]  # "mujoco": a labelled diagnostic
        box["estimates_used"] = list(task["estimates"])
        box["gap_post_look"] = row
        return {"isaac_post_look": True}

    rtm.check_post_look_frame = post_look
    original_reset = rt2.reset_and_look

    def reset_and_look(robot, seed, reset):
        out = original_reset(robot, seed, reset)
        box["truth"] = out[0]
        return out

    rt2.reset_and_look = reset_and_look


def install_e9_pick(box: dict) -> None:
    """``--pick e9``: e9's privileged scripted pick in place of P-3 before 405 (labelled)."""
    from embodied_jepa import first_policy_v2_runtime as rt2
    from embodied_jepa import lewm_next_c1_runtime as c1rt
    from embodied_jepa import plate_twin_v2 as pt

    act0, adv0 = c1rt.CommitController.act, c1rt.CommitController.advance

    def act(self, observation, step):
        if int(step) < pt.TRANSFER_START:
            if getattr(self, "_e9", None) is None:
                self._e9 = rt2.ExpertController(box["truth"], box["robot"])
            state = rt2.state_of(observation)
            if int(step) >= 397:
                self.palm[int(step)] = self.fk.pose9(state)[:2].copy()
            return self._e9.act(observation, step)
        return act0(self, observation, step)

    def advance(self, result):
        e9 = getattr(self, "_e9", None)
        if e9 is not None and not e9.expert.done:
            e9.advance(result)
        return adv0(self, result)

    c1rt.CommitController.act, c1rt.CommitController.advance = act, advance


class Video:
    def __init__(self, path: Path, size, fps: int):
        self.path = path
        self.proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
             "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE,
        )  # fmt: skip
        self.frames = 0

    def write(self, frame, repeat: int = 1) -> None:
        data = np.ascontiguousarray(frame, np.uint8).tobytes()
        for _ in range(repeat):
            self.proc.stdin.write(data)
            self.frames += 1

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path}")


def lean_record(record: dict) -> dict:
    keep = ("seed", "arm", "success", "at_rest", "latched_success", "final_distance_cm",
            "complete", "termination_reason", "grasp", "first_grasp_step", "first_place_step",
            "executed_steps", "privileged_ok", "task_truth_in_controller", "blocked", "commit",
            "seconds", "solver")  # fmt: skip
    out = {k: record.get(k) for k in keep}
    kp = record.get("kpred") or {}
    out["plate_path_s1"] = (kp.get("plate_path") or {}).get(str(lm.RULE["s1"]))
    out["move"] = kp.get("move")
    decisions = record.get("decisions") or []
    if decisions:
        d = decisions[0]
        out["decision"] = {k: d.get(k) for k in ("target", "reading", "fallback", "box")}
        wm = d.get("world_model") or {}
        out["decision"]["world_model"] = {k: wm.get(k) for k in (
            "solver", "grid_best", "fallback_grid_argmin", "fallback_reason", "clipped",
            "residual_cm", "feasible", "seconds")}  # fmt: skip
    return out


RN = None


def main(argv=None) -> int:
    global RN
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", required=True)
    ap.add_argument("--socket", default=None, help="the server's cp.sock")
    ap.add_argument("--sham", action="store_true", help="host MuJoCo endpoint (plumbing check)")
    ap.add_argument("--p-readout-cache", default=None, help="pickle of the refitted P readout")
    ap.add_argument("--refit-only", action="store_true", help="refit, write the cache, exit")
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--old-features", required=True)
    ap.add_argument("--old-fits", required=True)
    ap.add_argument("--models", nargs=6, required=True)
    ap.add_argument("--stage-r", required=True)
    ap.add_argument("--step-a-seeds", type=int, nargs="*", default=[])
    ap.add_argument("--seeds", type=int, nargs="*", default=[])
    ap.add_argument("--arms", nargs="+", default=["W", "H-rule"])
    ap.add_argument("--pick", choices=("p3", "e9"), default="p3")
    ap.add_argument(
        "--estimates-source",
        choices=("isaac", "mujoco"),
        default="isaac",
        help="P-3's post-look estimates from Isaac's frame, or (diagnostic) from MuJoCo's "
        "rendering of the same Isaac state",
    )
    ap.add_argument("--physics-label", default="Isaac")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--socket-timeout", type=float, default=1800.0)
    args = ap.parse_args(argv)
    out_dir = Path(args.output)
    if out_dir.exists():
        raise SystemExit(f"refusing: {out_dir} exists")
    if not args.refit_only and (args.socket is None) == (not args.sham):
        raise SystemExit("give exactly one of --socket and --sham")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("system ffmpeg is required")
    seeds_all = sorted(set(args.seeds) | set(args.step_a_seeds))
    low, high = cpv.DEBUG_SEEDS
    if any(not low <= s <= high for s in seeds_all):
        raise SystemExit(f"debug seeds only ({low}-{high})")
    out_dir.mkdir(parents=True)
    if shutil.disk_usage(out_dir).free / 2**30 < MIN_DISK_GIB:
        raise SystemExit("refusing: under 10 GiB free")
    demo = _load("render_lewm_cp_v2_demo", ROOT / "scripts" / "render_lewm_cp_v2_demo.py")
    RN = rn = demo._runner()
    from embodied_jepa import lewm_cp_v2_runtime as cpr
    from embodied_jepa import plate_twin_v2_harness as hz
    from embodied_jepa import wm_critic_v2_runtime as rtm

    started = time.monotonic()
    report = {
        "what": "TASK-081 controller closed loop in Isaac Sim: development only, not evidence",
        "label": LABEL,
        "revision": hz.revision(),
        "tracked_tree_dirty": bool(hz.tracked_tree_dirty()),
        "argv": sys.argv[1:] if argv is None else list(argv),
        "pick": args.pick,
        "estimates_source": args.estimates_source,
        "stages": {},
    }
    manifest = json.loads((ROOT / lm.TASK076_MANIFEST).read_text())
    rn.sim_preflight(report, args, manifest)
    chain = rn.old_chain(args, report, ())
    readouts = rn.stage_r_readouts(args, report)
    config = rn.worker_config(chain, readouts=readouts, with_r8=True)
    config = {"torch_threads": cpv.WORKER_TORCH_THREADS,
              "p3_checkpoint": hz.p3_checkpoint(Path(args.evidence))} | config  # fmt: skip
    import pickle

    cache = None if args.p_readout_cache is None else Path(args.p_readout_cache)
    if cache is not None and cache.exists():
        from embodied_jepa import pretrained_encoder as pe

        saved = pickle.loads(cache.read_bytes())
        p_readout, encoder = saved["p_readout"], pe.load_pretrained()
        report["p_readout_cache"] = {"path": str(cache), "sha256": demo.sha256_file(cache),
                                     "refit_report": saved["refit_report"]}  # fmt: skip
        log(f"P readout loaded from {cache}")
    else:
        pool = rn.Pool(int(args.workers), config)
        try:
            p_readout, encoder = hz.refit_p_readout(
                report, pool, Path(args.evidence), report["_run1"]
            )
        finally:
            report["pool_close"] = pool.close()
        report.pop("_run1", None)
        log(f"P readout refitted ({time.monotonic() - started:.0f} s)")
        if cache is not None:
            refit_report = json.loads(json.dumps(report.get("stages", {}), default=float))
            cache.write_bytes(pickle.dumps({"p_readout": p_readout, "refit_report": refit_report}))
    if args.refit_only:
        (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float))
        return 0
    cpr.worker_init(config)
    readers = Readers(p_readout, encoder)
    endpoint = (
        ShamEndpoint() if args.sham else SocketEndpoint(Path(args.socket), args.socket_timeout)
    )
    report["server"] = endpoint.hello
    robot = make_robot(endpoint)
    rtm._W["robot"] = robot
    box: dict = {"robot": robot}
    install_estimates(box, readers, args.estimates_source)
    install_hooks(box, readers)
    if args.pick == "e9":
        install_e9_pick(box)

    def save():
        (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float))

    if args.step_a_seeds:
        t0 = time.monotonic()
        report["step_a"] = step_a(robot, readers, args.step_a_seeds, out_dir)
        report["step_a"]["seconds"] = time.monotonic() - t0
        save()
        log(f"step A summary: {json.dumps(report['step_a']['summary'])}")
    composer = Composer()
    parts = []
    if args.seeds and not args.no_video:
        intro = out_dir / "00_intro.mp4"
        video = Video(intro, composer.size, args.fps)
        video.write(composer.card([
            ("TASK-081's controller, closed loop in Isaac Sim", "big"),
            ("Development only - NOT evidence, not a gated run, debug seeds 71900-71999", "bold"),
            ("", "normal"),
            (f"Isaac steps the physics ({args.physics_label}) and renders the onboard", "normal"),
            ("112 px camera at MuJoCo's camera pose and intrinsics; every image a", "normal"),
            ("controller reads", "normal"),
            ("is Isaac's. The models were trained on MuJoCo images only.", "normal"),
            ("", "normal"),
            ("P-3 (learned BC policy) picks the apple" if args.pick == "p3" else
             "e9's SCRIPTED privileged pick (variant) picks the apple", "normal"),
            ("then the plate moves (simulation-only C1-M); at step 405 one place aim is", "normal"),
            ("committed; e9's scripted place puts the apple there.", "normal"),
            ("W: LeWM token predictor chooses that aim (it chooses only the place aim).", "normal"),
            ("H-rule: hand-written, not learned, given the plate law (contrast).", "normal"),
            ("", "normal"),
            ("Right: Isaac's onboard frame (read) and MuJoCo's rendering of the same", "small"),
            ("Isaac state (not read, shown for the domain gap).", "small"),
            ("", "normal"),
            ("Variant: " + VARIANT[(args.pick, args.estimates_source)], "bold"),
        ]), 6 * args.fps)  # fmt: skip
        video.close()
        parts.append(intro)
    report["attempts"] = []
    for seed in args.seeds:
        for arm in args.arms:
            task = rn.attempt_tasks(
                arm, [seed],
                SimpleNamespace(resets={seed: cpv.reset_of(seed)},
                                est={seed: {"estimates": [0.0] * 4, "frame_sha256": None,
                                            "state_sha256": None}}),
                tau_commit_cm=cpv.TAU_COMMIT_CM, a_lo=cpv.A_LO,
            )[0]  # fmt: skip
            task["wall_seconds"] = WALL_SECONDS
            for key in ("gap405", "frame405_mujoco", "counterfactual", "gap_post_look",
                        "estimates_used"):  # fmt: skip
                box.pop(key, None)
            video = recorder = None
            name = f"{len(parts):02d}_{arm}_{seed}"
            if not args.no_video:
                video = Video(out_dir / f"{name}.mp4", composer.size, args.fps)
                variant = VARIANT[(args.pick, args.estimates_source)]
                recorder = Recorder(robot, composer, video, arm=arm, seed=seed,
                                    stride=args.stride, pick=args.pick,
                                    physics=args.physics_label, variant=variant)  # fmt: skip
            box["recorder"] = recorder
            t0 = time.monotonic()
            error = None
            try:
                record = cpr.run_attempt_task(task)
            except Exception as exc:  # noqa: BLE001 - a development run records and continues
                import traceback

                error = f"{type(exc).__name__}: {exc}"
                record = {"seed": seed, "arm": arm, "success": False, "error": error,
                          "traceback": traceback.format_exc()}  # fmt: skip
                log(f"{arm} {seed}: ERROR {error}")
                try:
                    robot.stop("error")
                except Exception:  # noqa: BLE001
                    pass
            finally:
                box["recorder"] = None
            row = lean_record(record) | {
                "error": error,
                "wall_seconds": time.monotonic() - t0,
                "gap_post_look": box.get("gap_post_look"),
                "gap405": box.get("gap405"),
                "counterfactual_405": box.get("counterfactual"),
                "estimates_used": box.get("estimates_used"),
                "renders": robot.sim.renders,
                "plate_pushes": robot.sim.plate_pushes,
            }
            if error is not None:
                row["traceback"] = record.get("traceback")
            report["attempts"].append(row)
            success = bool(record.get("success"))
            d = record.get("final_distance_cm")
            log(f"{arm} {seed}: success={success} final={d} lifted_grasp={record.get('grasp')} "
                f"({row['wall_seconds']:.0f} s)")  # fmt: skip
            if video is not None:
                if error is not None:
                    banner = ("ERROR: attempt stopped", (120, 120, 120))
                elif success:
                    dd = "" if d is None else f", {d:.1f} cm off centre"
                    banner = (f"AT REST ON THE PLATE{dd} (Isaac state)", (30, 130, 60))
                else:
                    dd = "" if d is None else f": {d:.1f} cm off the plate centre"
                    why = record.get("termination_reason")
                    if d is None and why:
                        dd = f": stopped at step {record.get('executed_steps')} ({why})"
                    banner = (f"MISS{dd} (Isaac state)", (170, 40, 40))
                if recorder.last is not None:
                    robot.sim.want_third = True
                    try:
                        onboard = robot.sim.render()
                    finally:
                        robot.sim.want_third = False
                    t_last = recorder.last[0]
                    video.write(recorder.compose(t_last, onboard, banner), int(2.5 * args.fps))
                video.close()
                parts.append(video.path)
            save()
    if parts:
        listing = out_dir / "concat.txt"
        listing.write_text("".join(f"file '{p.name}'\n" for p in parts))
        final = out_dir / "task081_isaac_closedloop.mp4"
        cmd = ["ffmpeg", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
               "-c", "copy", str(final)]  # fmt: skip
        subprocess.run(cmd, check=True, cwd=out_dir)
        report["video"] = final.name
    try:
        report["server_timing"] = endpoint.call("info")
    except Exception as exc:  # noqa: BLE001
        report["server_timing"] = f"unavailable: {exc}"
    endpoint.close()
    report["seconds"] = time.monotonic() - started
    save()
    sums = [f"{demo.sha256_file(p)}  {p.relative_to(out_dir)}\n"
            for p in sorted(out_dir.rglob("*")) if p.is_file()]  # fmt: skip
    (out_dir / "SHA256SUMS").write_text("".join(sums))
    log(f"done in {report['seconds']:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
