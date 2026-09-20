"""Bounded native macOS G1 + dual Dex3 asset, physics, RGB, and viewer probe.

Run with .venv/bin/python; the optional --viewer mode requires mjpython on macOS.
This scene establishes asset feasibility, not calibrated hardware or grasp success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "third_party/unitree_mujoco/unitree_robots/g1"
FINGERS = ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")


def probe_xml() -> str:
    """Compose a fixed-base asset probe; preserve every articulated body and motor."""
    tree = ET.parse(ASSETS / "g1_29dof_with_hand.xml")
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(ASSETS / "meshes"))
    pelvis = root.find(".//body[@name='pelvis']")
    pelvis.remove(pelvis.find("joint[@type='free']"))
    ET.SubElement(root, "option", timestep="0.002")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 .05", rgba=".3 .35 .4 1")
    ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", directional="true")
    ET.SubElement(
        world, "camera", name="overview", pos="2 -2 1.6", xyaxes=".707 .707 0 -.25 .25 .935"
    )
    torso = root.find(".//body[@name='torso_link']")
    # +X is robot forward. These are explicit simulated extrinsics, not a calibration.
    ET.SubElement(
        torso, "camera", name="onboard_rgb", pos=".08 0 .35", xyaxes="0 -1 0 .5 0 .866", fovy="70"
    )
    for side in ("left", "right"):
        wrist = root.find(f".//body[@name='{side}_wrist_yaw_link']")
        ET.SubElement(wrist, "site", name=f"{side}_ee", pos=".12 0 0", size=".006")
    return ET.tostring(root, encoding="unicode")


def inventory(model: mujoco.MjModel) -> dict:
    return {
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "nbody": model.nbody,
        "ngeom": model.ngeom,
        "nmesh": model.nmesh,
        "joints": [
            {
                "name": model.joint(i).name,
                "qpos_address": int(model.jnt_qposadr[i]),
                "dof_address": int(model.jnt_dofadr[i]),
                "limited": bool(model.jnt_limited[i]),
                "range_rad": model.jnt_range[i].tolist(),
            }
            for i in range(model.njnt)
        ],
        "actuators": [
            {
                "name": model.actuator(i).name,
                "joint": model.joint(int(model.actuator_trnid[i, 0])).name,
                "torque_range_nm": model.actuator_ctrlrange[i].tolist(),
            }
            for i in range(model.nu)
        ],
        "bodies": [
            {
                "name": model.body(i).name,
                "mass_kg": float(model.body_mass[i]),
                "principal_inertia": model.body_inertia[i].tolist(),
                "collision_geom_count": int(
                    np.sum(
                        (model.geom_bodyid == i)
                        & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
                    )
                ),
            }
            for i in range(1, model.nbody)
        ],
        "cameras": [model.camera(i).name for i in range(model.ncam)],
        "sites": [model.site(i).name for i in range(model.nsite)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/mujoco_probe")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.frames < 1:
        parser.error("steps and frames must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "assets/manifest.json").read_text())
    for relative, expected in manifest["sha256"].items():
        path = ROOT / manifest["local_path"] / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Asset hash mismatch: {path}; run scripts/fetch_assets.py")
    source = mujoco.MjModel.from_xml_path(str(ASSETS / "scene_29dof_with_hand.xml"))
    xml = probe_xml()
    (args.output / "probe.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    assert model.nu == 43 and model.nq == 43
    for side in ("left", "right"):
        for finger in FINGERS:
            joint = model.joint(f"{side}_hand_{finger}_joint")
            assert model.jnt_limited[joint.id]
            assert np.any(model.actuator_trnid[:, 0] == joint.id)
            body = int(model.jnt_bodyid[joint.id])
            assert model.body_mass[body] > 0 and np.all(model.body_inertia[body] > 0)
            assert np.any((model.geom_bodyid == body) & (model.geom_contype != 0))
    qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    vadr = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    hand = np.array(["hand_" in model.actuator(i).name for i in range(model.nu)])
    kp, kd = np.where(hand, 2.0, 60.0), np.where(hand, 0.1, 3.0)

    def reset() -> None:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

    def step() -> None:
        torque = -kp * data.qpos[qadr] - kd * data.qvel[vadr]
        data.ctrl[:] = np.clip(
            torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
        )
        mujoco.mj_step(model, data)

    traces = []
    start = time.perf_counter()
    for _ in range(2):
        reset()
        trace = []
        for _ in range(args.steps):
            step()
            trace.append(np.concatenate((data.qpos, data.qvel)).copy())
        traces.append(np.asarray(trace))
    elapsed = time.perf_counter() - start
    assert np.isfinite(traces).all()
    np.testing.assert_array_equal(traces[0], traces[1])
    warnings = {str(i): int(w.number) for i, w in enumerate(data.warning) if w.number}
    if warnings:
        raise RuntimeError(f"MuJoCo warnings: {warnings}")
    reset()
    render_start = time.perf_counter()
    with mujoco.Renderer(model, height=240, width=320) as renderer:
        renderer.update_scene(data, camera="overview")
        overview = renderer.render().copy()
        Image.fromarray(overview).save(args.output / "overview.png")
        frames = []
        for _ in range(args.frames):
            renderer.update_scene(data, camera="onboard_rgb")
            frames.append(renderer.render().copy())
        rgb = frames[0]
        assert rgb.dtype == np.uint8 and rgb.shape == (240, 320, 3) and rgb.std() > 1
        for frame in frames[1:]:
            np.testing.assert_array_equal(rgb, frame)
        Image.fromarray(rgb).save(args.output / "onboard_rgb.png")
    render_seconds = time.perf_counter() - render_start
    viewer_result = "not_requested"
    if args.viewer:
        from mujoco import viewer as viewer_module

        with viewer_module.launch_passive(model, data) as viewer:
            start = time.monotonic()
            sync_count = 0
            while viewer.is_running() and time.monotonic() - start < 1.5:
                step()
                viewer.sync()
                sync_count += 1
                time.sleep(0.01)
        assert sync_count > 0, "Viewer closed before its first sync"
        viewer_result = {"sync_count": sync_count, "closed": True, "bounded_seconds": 1.5}
    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
        },
        "asset_revision": manifest["revision"],
        "project_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "probe_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "source": inventory(source),
        "fixed_base_probe": inventory(model),
        "physics": {
            "steps_per_repeat": args.steps,
            "repeats": 2,
            "timestep_s": model.opt.timestep,
            "steps_per_second_including_trace_copy": 2 * args.steps / elapsed,
            "deterministic_trace_exact": True,
            "finite_state": True,
            "warnings": warnings,
        },
        "render": {
            "shape_hwc": list(rgb.shape),
            "frames": args.frames,
            "frames_per_second_including_setup_and_png": args.frames / render_seconds,
            "stationary_frames_exact": True,
            "std": float(rgb.std()),
        },
        "viewer": viewer_result,
        "limitations": [
            "Fixed pelvis; no locomotion or balance control.",
            "Camera and EE extrinsics are simulation choices, not physical calibration.",
            "Upstream 29-DOF G1 + Dex3-1 is not an EDU4 serial-specific calibrated model.",
            "Collision and mass presence do not validate contact accuracy or grasp capability.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in ("environment", "physics", "render", "viewer")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
