"""Isaac Sim bring-up spike: fixed-root Unitree G1 + dual Dex3, one onboard RGB frame.

Development spike only (not a gated experiment, not a manipulation or policy result).
Runs INSIDE the ``isaaclab_arena`` container with Isaac Sim's Python; it is not part of
the ``embodied_jepa`` package and adds nothing to ``uv.lock``. See
``docs/ISAAC_BRINGUP_SPIKE.md`` and ``scripts/isaac/run_bringup.sh``.

It (1) spawns the Arena G1 USD (29 body DoF + two 7-DoF Dex3 hands) with the root link
fixed, (2) steps physics, (3) dumps joint names/limits for comparison with the MuJoCo
model, (4) renders one RGB frame from a camera placed like the project's MuJoCo
``onboard_rgb`` camera, and (5) records step time and GPU memory.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import time
import zlib
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--robot",
    choices=("arena_g1_dex3", "lab_g1_29dof"),
    default="arena_g1_dex3",
    help="arena_g1_dex3: Arena G1_CFG (g1_29dof_with_hand_rev_1_0.usd); "
    "lab_g1_29dof: Isaac Lab G1_29DOF_CFG (g1.usd)",
)
parser.add_argument("--steps", type=int, default=1000, help="physics-only steps to time")
parser.add_argument("--render_steps", type=int, default=50, help="steps with camera render")
parser.add_argument("--dt", type=float, default=0.002, help="physics dt (MuJoCo uses 0.002)")
parser.add_argument("--width", type=int, default=224)
parser.add_argument("--height", type=int, default=224)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - diagnostic only
        return None


if args.output.exists():
    raise SystemExit(f"refusing to overwrite {args.output}")
args.output.mkdir(parents=True)
memory = {"before_app_launch_mib": gpu_used_mib()}
t_launch = time.perf_counter()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
launch_s = time.perf_counter() - t_launch
memory["after_app_launch_mib"] = gpu_used_mib()

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Minimal dependency-free RGB8 PNG writer."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def quat_xyzw_from_matrix(m: np.ndarray) -> tuple[float, float, float, float]:
    tr = np.trace(m)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, x = 0.25 * s, (m[2, 1] - m[1, 2]) / s
        y, z = (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
        q = np.zeros(4)
        q[i] = 0.25 * s
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
        w = (m[k, j] - m[j, k]) / s
        x, y, z = q[:3]
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return tuple(float(v) for v in q)


def mujoco_onboard_camera_offset():
    """The project's MuJoCo camera: torso_link, pos .08 0 .35, xyaxes 0 -1 0 .866 0 .5, fovy 75.

    MuJoCo and USD/OpenGL cameras share the frame convention (look along -Z, +Y up), so
    the rotation is the matrix whose columns are the MuJoCo camera x, y, z axes.
    """
    x = np.array([0.0, -1.0, 0.0])
    y = np.array([0.866, 0.0, 0.5])
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return (0.08, 0.0, 0.35), quat_xyzw_from_matrix(np.stack([x, y, z], axis=1))


def robot_cfg():
    if args.robot == "arena_g1_dex3":
        from isaaclab_arena.embodiments.g1.g1 import G1_CFG

        cfg = G1_CFG.copy()
    else:
        from isaaclab_assets import G1_29DOF_CFG

        cfg = G1_29DOF_CFG.copy()
    cfg.prim_path = "/World/Robot"
    cfg.spawn.articulation_props.fix_root_link = True
    # MuJoCo model: pelvis at (0, 0, 0.793), identity orientation (xyzw).
    cfg.init_state.pos = (0.0, 0.0, 0.793)
    cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)
    return cfg


def main() -> None:
    record: dict = {
        "robot_variant": args.robot,
        "physics_dt_s": args.dt,
        "device": args.device,
        "isaac_sim_version": Path("/isaac-sim/VERSION").read_text().strip()
        if Path("/isaac-sim/VERSION").exists()
        else None,
        "app_launch_s": launch_s,
    }
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args.dt, device=args.device))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light.func("/World/light", light)
    table = sim_utils.CuboidCfg(
        size=(0.64, 0.90, 0.08),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.4, 0.25)),
    )
    table.func("/World/table", table, translation=(0.45, 0.0, 0.70))

    cfg = robot_cfg()
    record["usd_path"] = cfg.spawn.usd_path
    robot = Articulation(cfg=cfg)

    pos, rot = mujoco_onboard_camera_offset()
    focal = 20.955 / (2.0 * np.tan(np.deg2rad(75.0) / 2.0))  # vertical fov 75 deg, square
    cam_cfg = CameraCfg(
        prim_path="/World/Robot/torso_link/onboard_rgb",
        update_period=0.0,
        height=args.height,
        width=args.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(focal),
            horizontal_aperture=20.955,
            vertical_aperture=20.955 * args.height / args.width,
            clipping_range=(0.01, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="opengl"),
    )
    camera = Camera(cfg=cam_cfg)
    record["camera"] = {
        "parent": "torso_link",
        "pos": pos,
        "rot_xyzw_opengl": rot,
        "focal_length": float(focal),
        "width": args.width,
        "height": args.height,
        "matches": "MuJoCo onboard_rgb (simulation.py) pose and fovy=75; not calibrated",
    }

    t0 = time.perf_counter()
    sim.reset()
    record["sim_reset_s"] = time.perf_counter() - t0
    memory["after_sim_reset_mib"] = gpu_used_mib()

    names = list(robot.joint_names)
    limits = wp.to_torch(robot.data.joint_pos_limits)[0].cpu().numpy()
    vel = wp.to_torch(robot.data.joint_vel_limits)[0].cpu().numpy()
    eff = wp.to_torch(robot.data.joint_effort_limits)[0].cpu().numpy()
    record["num_joints"] = len(names)
    record["body_names"] = list(robot.body_names)
    record["fixed_base"] = bool(robot.is_fixed_base)
    joints = [
        {
            "index": i,
            "name": n,
            "lower": float(limits[i, 0]),
            "upper": float(limits[i, 1]),
            "velocity_limit": float(vel[i]),
            "effort_limit": float(eff[i]),
        }
        for i, n in enumerate(names)
    ]
    (args.output / "isaac_joints.json").write_text(
        json.dumps({"robot_variant": args.robot, "joints": joints}, indent=2)
    )

    # Hold the default pose with the configured PD actuators.
    hold = wp.to_torch(robot.data.default_joint_pos).clone()
    root0 = wp.to_torch(robot.data.root_pos_w)[0].cpu().numpy().copy()

    def step(render: bool) -> None:
        robot.set_joint_position_target_index(target=hold)
        robot.write_data_to_sim()
        sim.step(render=render)
        robot.update(args.dt)

    for _ in range(20):  # warm-up
        step(False)
    sync = torch.cuda.synchronize if "cuda" in str(sim.device) else (lambda: None)
    sync()
    times = []
    for _ in range(args.steps):
        t = time.perf_counter()
        step(False)
        sync()
        times.append(time.perf_counter() - t)
    phys = np.array(times)
    memory["after_physics_mib"] = gpu_used_mib()

    rtimes = []
    for _ in range(args.render_steps):
        t = time.perf_counter()
        step(True)
        camera.update(args.dt)
        rgb = camera.data.output["rgb"]
        sync()
        rtimes.append(time.perf_counter() - t)
    rend = np.array(rtimes)
    memory["after_render_mib"] = gpu_used_mib()
    memory["torch_max_allocated_mib"] = (
        torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
    )

    img = rgb[0, ..., :3].detach().cpu().numpy().astype(np.uint8)
    write_png(args.output / "onboard_rgb.png", img)
    root1 = wp.to_torch(robot.data.root_pos_w)[0].cpu().numpy()
    qpos = wp.to_torch(robot.data.joint_pos)[0].cpu().numpy()

    record.update(
        {
            "root_drift_m": float(np.linalg.norm(root1 - root0)),
            "joint_pos_finite": bool(np.isfinite(qpos).all()),
            "max_abs_hold_error_rad": float(np.abs(qpos - hold[0].cpu().numpy()).max()),
            "physics_step_ms": {
                "n": int(phys.size),
                "mean": 1e3 * float(phys.mean()),
                "median": 1e3 * float(np.median(phys)),
                "p95": 1e3 * float(np.percentile(phys, 95)),
            },
            "render_step_ms": {
                "n": int(rend.size),
                "first": 1e3 * float(rend[0]),
                "median_excl_first": 1e3 * float(np.median(rend[1:])) if rend.size > 1 else None,
                "p95_excl_first": 1e3 * float(np.percentile(rend[1:], 95))
                if rend.size > 1
                else None,
            },
            "rgb_shape": list(img.shape),
            "rgb_mean": float(img.mean()),
            "rgb_std": float(img.std()),
            "gpu_memory_device_used_mib": memory,
            "note": "device-wide nvidia-smi figures include other processes (GR00T server)",
            "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES",)},
        }
    )
    (args.output / "bringup.json").write_text(json.dumps(record, indent=2))
    print("BRINGUP_RESULT " + json.dumps({k: v for k, v in record.items() if k != "body_names"}))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
