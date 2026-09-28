"""MuJoCo side of the IsaacTransport / MuJoCo transport parity check, and the comparison.

Development tooling (TASK-025), not a gated experiment and not a manipulation result. Run on
the host after ``parity_isaac.py``:

    uv run --no-sync python scripts/isaac/parity_mujoco.py \\
        --isaac outputs/<isaac-run>/run --output outputs/<isaac-run>/parity

It replays exactly the joint targets recorded in ``isaac_trace.npz`` through
``MuJoCoSimulation.send_joint_targets`` (same manifest, 112 px ``onboard_rgb``), with the
apple and plate made invisible and non-colliding so both simulators hold robot + table only,
and compares realised joint motion, tracking error, timing and onboard frames.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from embodied_jepa.isaac_transport import (
    joint_manifest_from_mujoco,
    manifest_sha256,
    reset_pose,
)
from embodied_jepa.simulation import MuJoCoSimulation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_joints import group  # noqa: E402
from parity_targets import trajectory  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FRAMES = (0, 30, 50, 70, 90)


def hide_objects(sim: MuJoCoSimulation) -> None:
    model = sim.model
    for g in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[g])).name
        if body in ("apple", "plate"):
            model.geom_rgba[g, 3] = 0.0
            model.geom_contype[g] = 0
            model.geom_conaffinity[g] = 0


def stats(x: np.ndarray) -> dict:
    return {
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", type=Path, required=True, help="parity_isaac.py output dir")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "configs/isaac/g1_dex3_joint_manifest_v0.json"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    trace = np.load(args.isaac / "isaac_trace.npz")
    isaac_record = json.loads((args.isaac / "isaac_parity.json").read_text())
    names = [str(n) for n in trace["joint_names"]]
    targets = trace["targets"]
    size = int(isaac_record["rgb_size"])

    sim = MuJoCoSimulation(width=size, height=size, render=True)
    committed = json.loads(args.manifest.read_text())
    regenerated = joint_manifest_from_mujoco(sim)
    if manifest_sha256(committed) != manifest_sha256(regenerated):
        raise SystemExit("committed joint manifest differs from the MuJoCo model")
    if manifest_sha256(committed) != isaac_record["manifest_sha256"]:
        raise SystemExit("Isaac run used a different joint manifest")
    if list(sim.joint_names) != names:
        raise SystemExit("joint order differs")
    hide_objects(sim)
    # Objects far from the arms' reach and invisible/non-colliding: robot + table only.
    sim.reset(0, object_xy=[0.63, 0.30], plate_xy=[0.60, -0.24])
    sim.render()  # warm the renderer; the first MuJoCo render is discarded elsewhere too
    if not np.array_equal(reset_pose(committed), sim.data.qpos[sim.qadr]):
        raise SystemExit("manifest reset pose differs from MuJoCoSimulation.reset")
    regenerated_targets = trajectory(
        names, reset_pose(committed), ROOT / "configs/g1_sim_action.json"
    )
    targets_equal = bool(np.array_equal(regenerated_targets, targets))

    qpos, qvel, times, step_ms, render_ms = [], [], [], [], []
    frames = {}
    obs = sim.read()
    qpos.append(obs["qpos"])
    qvel.append(obs["qvel"])
    times.append(obs["timestamp"])
    frames[0] = obs["rgb"]
    for k, target in enumerate(targets, start=1):
        a = time.perf_counter()
        ack = sim.send_joint_targets(target, joint_names=names, deadline=sim.data.time + 0.05)
        b = time.perf_counter()
        obs = sim.read()
        c = time.perf_counter()
        if ack["status"] != "applied":
            raise RuntimeError(f"interval {k}: {ack}")
        step_ms.append(1e3 * (b - a))
        render_ms.append(1e3 * (c - b))
        qpos.append(obs["qpos"])
        qvel.append(obs["qvel"])
        times.append(obs["timestamp"])
        if k in FRAMES:
            frames[k] = obs["rgb"]
    sim.close()
    mq, mv, mt = np.array(qpos), np.array(qvel), np.array(times)
    iq, it = trace["qpos"][0], trace["time"][0]

    # Joint motion: Isaac vs MuJoCo, and each vs its commanded target (after each interval).
    diff = np.abs(iq - mq)  # [91, 43]
    groups = sorted({group(n) for n in names})
    by_group = {
        g: {
            "isaac_vs_mujoco_rad": stats(diff[:, [group(n) == g for n in names]]),
            "mujoco_tracking_rad": stats(
                np.abs(mq[1:] - targets)[:, [group(n) == g for n in names]]
            ),
            "isaac_tracking_rad": stats(
                np.abs(iq[1:] - targets)[:, [group(n) == g for n in names]]
            ),
        }
        for g in groups
    }
    worst = np.unravel_index(np.argmax(diff), diff.shape)
    phases = {
        "hold 0-10": (0, 11),
        "arms 10-50": (11, 51),
        "hands 50-70": (51, 71),
        "return 70-90": (71, 91),
    }
    by_phase = {p: float(diff[a:b].max()) for p, (a, b) in phases.items()}
    per_joint_max = {n: float(diff[:, i].max()) for i, n in enumerate(names)}

    from PIL import Image

    args.output.mkdir(parents=True)
    image_rows = {}
    for k, rgb in frames.items():
        isaac_png = args.isaac / "frames" / f"isaac_{k:03d}.png"
        isaac_rgb = np.asarray(Image.open(isaac_png).convert("RGB"))
        d = np.abs(isaac_rgb.astype(int) - rgb.astype(int))
        image_rows[k] = {"mean_abs_diff": float(d.mean()), "max_abs_diff": int(d.max())}
        Image.fromarray(np.concatenate([rgb, isaac_rgb], axis=1)).save(
            args.output / f"mujoco_left_isaac_right_{k:03d}.png"
        )

    report = {
        "question": "Does IsaacTransport on the converted USD realise the same joint motion "
        "as MuJoCoSimulation for identical joint targets? (robot + table only)",
        "isaac_run": str(args.isaac),
        "manifest_sha256": manifest_sha256(committed),
        "targets_regenerated_equal": targets_equal,
        "intervals": int(targets.shape[0]),
        "episode_time_equal": bool(np.allclose(it, mt, atol=1e-9)),
        "joint_abs_diff_by_group": by_group,
        "joint_abs_diff_max_by_phase_rad": by_phase,
        "worst": {
            "joint": names[int(worst[1])],
            "read_index": int(worst[0]),
            "rad": float(diff[worst]),
        },
        "final_pose_abs_diff_max_rad": float(diff[-1].max()),
        "per_joint_max_abs_diff_rad": per_joint_max,
        "isaac_repeat_max_abs_qpos_diff": isaac_record["repeat_max_abs_qpos_diff"],
        "timing_ms": {
            "mujoco_send_joint_targets": stats(np.array(step_ms)),
            "mujoco_read_with_render": stats(np.array(render_ms)),
            "isaac_send_joint_targets": isaac_record["send_joint_targets_ms"],
            "isaac_read_with_render": isaac_record["read_with_render_ms"],
        },
        "images_mean_abs_diff_uint8": image_rows,
        "isaac_staleness": isaac_record["staleness"],
        "isaac_rejections": isaac_record["rejections"],
        "isaac_after_close": isaac_record["after_close"],
        "caveats": [
            "MuJoCo frictionloss (0.1-0.2 N m) is not applied in Isaac (no equivalent)",
            "MuJoCo self-collision is on, Isaac self-collision is off (importer default)",
            "different engines, integrators (implicitfast vs PhysX TGS) and renderers",
            "robot + table only; no apple/plate; no contact parity",
        ],
    }
    np.savez_compressed(args.output / "mujoco_trace.npz", qpos=mq, qvel=mv, time=mt)
    (args.output / "parity_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "targets_regenerated_equal",
                    "episode_time_equal",
                    "joint_abs_diff_max_by_phase_rad",
                    "worst",
                    "final_pose_abs_diff_max_rad",
                    "isaac_repeat_max_abs_qpos_diff",
                    "timing_ms",
                    "images_mean_abs_diff_uint8",
                    "isaac_staleness",
                )
            },
            indent=2,
        )
    )
    for g, row in by_group.items():
        print(
            f"{g:6s} isaac-vs-mujoco max {row['isaac_vs_mujoco_rad']['max']:.4f} "
            f"median {row['isaac_vs_mujoco_rad']['median']:.4f} | tracking max "
            f"mj {row['mujoco_tracking_rad']['max']:.4f} "
            f"isaac {row['isaac_tracking_rad']['max']:.4f}"
        )


if __name__ == "__main__":
    main()
