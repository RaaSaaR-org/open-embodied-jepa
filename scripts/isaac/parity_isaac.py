"""Isaac side of the IsaacTransport / MuJoCo transport parity check (development, TASK-025).

Runs INSIDE the ``isaaclab_arena`` container via ``scripts/isaac/run_isaac.sh``. Not a gated
experiment and not a manipulation result: robot + table only, no apple or plate.

It builds ``IsaacTransport`` on the converted USD, then:

1. replays the ``parity_targets.trajectory`` joint targets twice from ``reset()`` with a
   ``read()`` (state + onboard RGB) after every 0.05 s interval, recording joint state,
   episode time, wall time per interval (CUDA-synchronised) and per render;
2. checks render staleness (a second render of the same state must equal the first);
3. checks rejections: expired deadline, out-of-limit target, wrong joint order;
4. closes the transport and checks that further use fails.

Output (``--output``): ``isaac_trace.npz``, ``frames/*.png`` and ``isaac_parity.json``;
``parity_mujoco.py`` on the host replays the same targets in MuJoCo and compares.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--usd", required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--action_manifest", type=Path, required=True)
parser.add_argument("--size", type=int, default=112, help="onboard RGB width = height")
parser.add_argument("--repeats", type=int, default=2)
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
from parity_targets import trajectory  # noqa: E402

from embodied_jepa.contracts import ContractError  # noqa: E402
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


def main() -> None:
    manifest = json.loads(args.manifest.read_text())
    t0 = time.perf_counter()
    tr = IsaacTransport(
        usd_path=args.usd, manifest=manifest, width=args.size, height=args.size, render=True
    )
    build_s = time.perf_counter() - t0
    memory["after_transport_build_mib"] = gpu_used_mib()
    names = list(tr.joint_names)
    targets = trajectory(names, reset_pose(manifest), args.action_manifest)
    record: dict = {
        "isaac_sim_version": Path("/isaac-sim/VERSION").read_text().strip(),
        "usd": args.usd,
        "manifest_sha256": manifest_sha256(manifest),
        "isaac_joint_order": list(tr.isaac_joint_names),
        "physics_dt_s": tr.physics_dt,
        "control_dt_s": tr.control_dt,
        "substeps": tr.substeps,
        "rgb_size": args.size,
        "app_launch_s": launch_s,
        "transport_build_s": build_s,
        "device": str(tr.sim.device),
    }
    qpos, qvel, times, step_ms, render_ms = [], [], [], [], []
    for rep in range(args.repeats):
        tr.reset()
        obs = tr.read()
        rq, rv, rt, sm, rm = [obs["qpos"]], [obs["qvel"]], [obs["timestamp"]], [], []
        if rep == 0:
            write_png(args.output / "frames" / "isaac_000.png", obs["rgb"])
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
            if rep == 0 and k in FRAMES:
                write_png(args.output / "frames" / f"isaac_{k:03d}.png", obs["rgb"])
        qpos.append(rq)
        qvel.append(rv)
        times.append(rt)
        step_ms.append(sm)
        render_ms.append(rm)
    memory["after_trajectories_mib"] = gpu_used_mib()
    memory["torch_max_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20

    # Render staleness: two renders of the same (just-moved) state.
    tr.reset()
    tr.send_joint_targets(targets[40], joint_names=names, deadline=tr.time + 0.05)
    r1, r2 = tr.render(), tr.render()
    record["staleness"] = {
        "second_render_max_abs_diff": int(np.abs(r1.astype(int) - r2.astype(int)).max()),
        "second_render_mean_abs_diff": float(np.abs(r1.astype(int) - r2.astype(int)).mean()),
    }

    # Rejections.
    tr.reset()
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
        "object_xy_reset": expect(ContractError, lambda: tr.reset(object_xy=(0.4, -0.2))),
    }
    tr.close()
    record["after_close"] = {
        "read": expect(RuntimeError, tr.read),
        "send": expect(
            RuntimeError,
            lambda: tr.send_joint_targets(targets[0], joint_names=names, deadline=1e9),
        ),
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
    summary = {k: record[k] for k in ("send_joint_targets_ms", "repeat_max_abs_qpos_diff")}
    print("PARITY_ISAAC " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
