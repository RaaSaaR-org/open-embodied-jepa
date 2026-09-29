"""Isaac side of the IsaacTransport / MuJoCo transport parity check (development, TASK-025).

Runs INSIDE the ``isaaclab_arena`` container via ``scripts/isaac/run_isaac.sh``. Not a gated
experiment and not a manipulation result.

It builds ``IsaacTransport`` on the converted USD with the committed scene and camera manifests
(by default with the v2 apple and plate at the seed-0 reset layout), then:

1. replays a ``parity_targets.trajectory`` variant (default ``free_space_v1``) twice from
   ``reset()`` with a ``read()`` (state + onboard RGB) after every 0.05 s interval, recording
   joint state, episode time, PhysX contacts (last physics step and any substep of the
   interval), the apple position, wall time per interval (CUDA-synchronised) and per render;
   RGB frames and instance segmentation are saved at reads 0, 30, 50, 70 and 90 of both
   replays (the second replay checks image determinism);
2. checks render freshness: a read right after ``reset()`` must show the reset state, not the
   last pre-reset frame, and a second render of the same state must equal the first;
3. checks rejections: expired deadline, out-of-limit target, wrong joint order, apple reset
   off the table or overlapping the plate;
4. closes the transport and checks that further use fails.

Output (``--output``): ``isaac_trace.npz``, ``frames/*.png``, ``frames/*_seg.npy`` and
``isaac_parity.json``; ``parity_mujoco.py`` on the host replays the same targets in MuJoCo and
compares.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

from isaaclab.app import AppLauncher

# Development diagnostic: two of ten Newton runs hung (one CPU core busy, nothing logged)
# before the model was built. Dump every thread's Python stack every 5 min to the log.
faulthandler.dump_traceback_later(300, repeat=True)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--usd", required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--action_manifest", type=Path, required=True)
parser.add_argument("--scene", type=Path, required=True, help="v2 scene manifest")
parser.add_argument("--camera", type=Path, required=True, help="onboard camera manifest")
parser.add_argument("--trajectory", default="free_space_v1")
parser.add_argument("--no_objects", action="store_true", help="robot, floor and table only")
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--joint_friction", choices=("frictionloss", "none"), default="frictionloss")
parser.add_argument("--physics", choices=("physx", "newton"), default="physx")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True
if args.output.exists():
    raise SystemExit(f"refusing to overwrite {args.output}")
args.output.mkdir(parents=True)
(args.output / "frames").mkdir()


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


memory = {"before_app_launch_mib": gpu_used_mib()}
t_launch = time.perf_counter()
app = AppLauncher(args).app
launch_s = time.perf_counter() - t_launch

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/oej/src")  # run_isaac.sh mounts src there; the entrypoint drops PYTHONPATH
from parity_targets import VARIANTS, trajectory  # noqa: E402

from embodied_jepa.contracts import ContractError  # noqa: E402
from embodied_jepa.isaac_scene import (  # noqa: E402
    canonical_sha256,
    check_render_freshness,
    usd_canonical_hash,
)
from embodied_jepa.isaac_transport import (  # noqa: E402
    IsaacTransport,
    manifest_sha256,
    reset_pose,
)

FRAMES = (0, 30, 50, 70, 90)  # read index: 0 = after reset, k = after interval k


def write_png(path: Path, rgb: np.ndarray) -> None:
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


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def expect(exc, fn) -> str:
    try:
        fn()
    except exc as e:  # noqa: PERF203
        return f"raised {type(e).__name__}: {e}"
    return "did not raise"


def save_frame(tr, obs, rep: int, k: int) -> None:
    write_png(args.output / "frames" / f"isaac_r{rep}_{k:03d}.png", obs["rgb"])
    seg, labels = tr.segmentation()
    np.save(args.output / "frames" / f"isaac_r{rep}_{k:03d}_seg.npy", seg)
    (args.output / "frames" / f"isaac_r{rep}_{k:03d}_seg.json").write_text(json.dumps(labels))


def main() -> None:
    manifest = json.loads(args.manifest.read_text())
    scene = json.loads(args.scene.read_text())
    camera = json.loads(args.camera.read_text())
    if args.trajectory not in VARIANTS:
        raise SystemExit(f"unknown trajectory {args.trajectory}")
    objects = not args.no_objects
    t0 = time.perf_counter()
    tr = IsaacTransport(
        usd_path=args.usd,
        manifest=manifest,
        scene_manifest=scene,
        camera_manifest=camera,
        objects=objects,
        render=True,
        record_contacts=True,
        joint_friction=args.joint_friction,
        physics=args.physics,
    )
    build_s = time.perf_counter() - t0
    memory["after_transport_build_mib"] = gpu_used_mib()
    names = list(tr.joint_names)
    targets = trajectory(names, reset_pose(manifest), args.action_manifest, args.trajectory)
    record: dict = {
        "isaac_sim_version": Path("/isaac-sim/VERSION").read_text().strip(),
        "physics_backend": tr.physics,
        "newton_model": tr.newton_model,
        "usd": args.usd,
        "manifest_sha256": manifest_sha256(manifest),
        "scene_manifest_sha256": canonical_sha256(scene),
        "usd_canonical_tree_sha256": usd_canonical_hash(args.usd),
        "camera_manifest_sha256": canonical_sha256(camera),
        "trajectory": args.trajectory,
        "objects": objects,
        "reset": {"seed": 0, "object_xy": None, "plate_xy": None},
        "isaac_joint_order": list(tr.isaac_joint_names),
        "physics_dt_s": tr.physics_dt,
        "control_dt_s": tr.control_dt,
        "substeps": tr.substeps,
        "rgb_size": [tr.width, tr.height],
        "render_settings_read_back": tr.render_settings,
        "app_launch_s": launch_s,
        "transport_build_s": build_s,
        "device": str(tr.sim.device),
        "joint_friction": args.joint_friction,
        "physx_joint_friction": tr.physx_joint_friction,
        "object_properties": tr.object_properties,
        "contact_api_error": tr.contact_api_error,
    }
    qpos, qvel, times, step_ms, render_ms, apple = [], [], [], [], [], []
    contacts_last, contacts_interval = [], []
    for rep in range(args.repeats):
        tr.reset(0)
        obs = tr.read()
        rq, rv, rt, sm, rm = [obs["qpos"]], [obs["qvel"]], [obs["timestamp"]], [], []
        ra = [tr.task_truth()["object_position"].tolist() if objects else None]
        cl, ci = [tr.contacts()["last_step"]], [tr.contacts()["interval_pairs"]]
        if rep == 0:
            first_reset = obs["rgb"].copy()
        if rep < 2:
            save_frame(tr, obs, rep, 0)
        for k, target in enumerate(targets, start=1):
            sync()
            a = time.perf_counter()
            ack = tr.send_joint_targets(target, joint_names=names, deadline=tr.time + 0.05)
            sync()
            b = time.perf_counter()
            obs = tr.read()
            sync()
            c = time.perf_counter()
            if ack["status"] != "applied":
                raise RuntimeError(f"interval {k}: {ack}")
            sm.append(1e3 * (b - a))
            rm.append(1e3 * (c - b))
            rq.append(obs["qpos"])
            rv.append(obs["qvel"])
            rt.append(obs["timestamp"])
            ra.append(tr.task_truth()["object_position"].tolist() if objects else None)
            cl.append(tr.contacts()["last_step"])
            ci.append(tr.contacts()["interval_pairs"])
            if rep < 2 and k in FRAMES:
                save_frame(tr, obs, rep, k)
        qpos.append(rq)
        qvel.append(rv)
        times.append(rt)
        step_ms.append(sm)
        render_ms.append(rm)
        apple.append(ra)
        contacts_last.append(cl)
        contacts_interval.append(ci)
    memory["after_trajectories_mib"] = gpu_used_mib()
    memory["torch_max_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20

    # Render freshness. (a) Move, render twice: the renders must be equal (deterministic).
    tr.reset(0)
    tr.send_joint_targets(targets[40], joint_names=names, deadline=tr.time + 0.05)
    moved_1, moved_2 = tr.render(), tr.render()
    # (b) Reset without stepping: the read must show the reset state, i.e. equal the reset
    # frame of the first replay and differ from the moved frame.
    tr.reset(0)
    after_reset = tr.read()["rgb"]

    def d(x, y):
        return {
            "max_abs_diff": int(np.abs(x.astype(int) - y.astype(int)).max()),
            "mean_abs_diff": float(np.abs(x.astype(int) - y.astype(int)).mean()),
        }

    record["freshness"] = {
        "second_render_of_moved_state": d(moved_1, moved_2),
        "post_reset_read_vs_first_reset_frame": d(after_reset, first_reset),
        "post_reset_read_vs_last_pre_reset_frame": d(after_reset, moved_2),
    }
    check_render_freshness(record["freshness"])

    # Rejections.
    tr.reset(0)
    expired = tr.send_joint_targets(targets[0], joint_names=names, deadline=tr.time - 1.0)
    bad = targets[0].copy()
    bad[names.index("left_elbow_joint")] = 10.0
    record["rejections"] = {
        "expired_deadline": expired,
        "out_of_limit": expect(
            ContractError, lambda: tr.send_joint_targets(bad, joint_names=names, deadline=1e9)
        ),
        "wrong_order": expect(
            ContractError,
            lambda: tr.send_joint_targets(
                targets[0], joint_names=list(reversed(names)), deadline=1e9
            ),
        ),
        "object_xy_off_table": expect(ContractError, lambda: tr.reset(object_xy=(0.9, -0.2))),
        "object_overlapping_plate": expect(
            ContractError, lambda: tr.reset(object_xy=(0.48, -0.12), plate_xy=(0.48, -0.10))
        ),
    }
    tr.close()
    record["after_close"] = {
        "read": expect(RuntimeError, tr.read),
        "send": expect(
            RuntimeError,
            lambda: tr.send_joint_targets(targets[0], joint_names=names, deadline=1e9),
        ),
        "task_truth": expect(RuntimeError, tr.task_truth),
    }

    step = np.array(step_ms)
    rend = np.array(render_ms)
    record.update(
        {
            "intervals": int(targets.shape[0]),
            "repeats": args.repeats,
            "send_joint_targets_ms": {
                "median": float(np.median(step)),
                "mean": float(step.mean()),
                "p95": float(np.percentile(step, 95)),
                "max": float(step.max()),
                "per_physics_substep_median": float(np.median(step)) / tr.substeps,
            },
            "read_with_render_ms": {
                "median": float(np.median(rend)),
                "p95": float(np.percentile(rend, 95)),
                "max": float(rend.max()),
            },
            "repeat_max_abs_qpos_diff": float(np.abs(np.array(qpos[0]) - np.array(qpos[-1])).max())
            if args.repeats > 1
            else None,
            "time_monotonic": bool(all(np.all(np.diff(t) > 0) for t in times)),
            "final_time_s": float(times[0][-1]),
            "all_finite": bool(
                np.isfinite(np.array(qpos)).all() and np.isfinite(np.array(qvel)).all()
            ),
            "isaac_contacts_last_step_per_read": contacts_last[0],
            "isaac_contacts_any_substep_per_read": contacts_interval[0],
            "apple_position_per_read": apple[0],
            "gpu_memory_device_used_mib": memory,
            "note": "device-wide nvidia-smi figures include other processes (GR00T server)",
        }
    )
    np.savez_compressed(
        args.output / "isaac_trace.npz",
        joint_names=np.array(names),
        targets=targets,
        qpos=np.array(qpos),
        qvel=np.array(qvel),
        time=np.array(times),
        send_ms=step,
        read_ms=rend,
    )
    (args.output / "isaac_parity.json").write_text(json.dumps(record, indent=2, default=str))
    summary = {
        k: record[k] for k in ("send_joint_targets_ms", "repeat_max_abs_qpos_diff", "freshness")
    }
    print("PARITY_ISAAC " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Kit's app.close() can end the process before Python prints the traceback.
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        app.close()
