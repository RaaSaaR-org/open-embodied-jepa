"""Development probe for choosing the Isaac render settings of ``onboard_rgb`` (TASK-025).

Runs INSIDE the ``isaaclab_arena`` container via ``scripts/isaac/run_isaac.sh``. Not a gated
experiment and not a parity result: it renders the seed-0 reset scene (robot at its reset
pose, v2 apple and plate at the ``MuJoCoSimulation.reset`` defaults) under a grid of renderer
settings and light intensities and saves two consecutive renders per variant, the instance
segmentation and the renderer's setting trees. The frames were scored on the host against
MuJoCo's frame of the same state (mean absolute difference and render-to-render change); the
settings chosen are pinned in ``isaac_scene.ISAAC_RENDER``. The choice used only this reset
frame; the image-parity check reports frames the choice never saw.

The committed grid is the last of four (``outputs/isaac-v2-render-probe-{1..4}``, 2026-09-29,
uncommitted development code): probe 1 found that renders did not update at all (Isaac Lab 3
renders at most once per physics step, fixed in ``IsaacTransport._pump_render``), probe 2
compared real-time modes at too-low light intensities, probe 3 compared real-time (off, FXAA,
DLAA x8) and path-traced modes at 300/1000/3000, and probe 4 (this grid) swept the path
tracer's light intensity, sRGB gamma and tonemapper.

It also drops the apple 5 cm onto the table once and records the contacts, as a smoke check of
the transport's contact path.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--usd", required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--scene", type=Path, required=True)
parser.add_argument("--camera", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True
if args.output.exists():
    raise SystemExit(f"refusing to overwrite {args.output}")
args.output.mkdir(parents=True)
app = AppLauncher(args).app

import carb  # noqa: E402
import numpy as np  # noqa: E402
from pxr import UsdLux  # noqa: E402

sys.path.insert(0, "/oej/src")
from embodied_jepa.isaac_transport import IsaacTransport, reset_pose  # noqa: E402

BASE = {
    "/rtx/directLighting/sampledLighting/enabled": False,
    "/rtx/shadows/enabled": True,
    "/rtx/ambientOcclusion/enabled": False,
    "/rtx/indirectDiffuse/enabled": False,
    "/rtx/reflections/enabled": False,
    "/rtx/translucency/enabled": False,
    "/rtx-transient/dldenoiser/enabled": False,
}


def dump(settings, root: str):
    try:
        return settings.get_settings_dictionary(root).get_dict()
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def main() -> None:
    manifest = json.loads(args.manifest.read_text())
    camera = json.loads(args.camera.read_text())
    camera["isaac_render"]["lights"] = [
        {
            "type": "distant",
            "path": "/World/light_top",
            "intensity": 1.0,
            "angle_deg": 0.0,
            "color": [1, 1, 1],
            "shadow": True,
            "direction": [0, 0, -1],
        },
        {
            "type": "distant",
            "path": "{camera}/headlight",
            "intensity": 1.0,
            "angle_deg": 0.0,
            "color": [1, 1, 1],
            "shadow": False,
        },
        {
            "type": "dome",
            "path": "/World/dome",
            "intensity": 0.0,
            "color": [1, 1, 1],
            "shadow": False,
        },
    ]
    tr = IsaacTransport(
        usd_path=args.usd,
        manifest=manifest,
        scene_manifest=json.loads(args.scene.read_text()),
        camera_manifest=camera,
        objects=True,
        render=True,
        record_contacts=True,
    )
    settings = carb.settings.get_settings()
    stage = tr._sim_utils.get_current_stage()
    top = UsdLux.DistantLight(stage.GetPrimAtPath("/World/light_top"))
    head = UsdLux.DistantLight(stage.GetPrimAtPath(f"{tr.camera_prim_path}/headlight"))
    dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/dome"))
    record = {
        "settings_before": {
            k: dump(settings, k) for k in ("/rtx/post/aa", "/rtx/rendermode", "/rtx/pathtracing")
        },
        "variants": [],
    }
    tr.reset(0)
    import omni.replicator.core as rep

    modes = {"pt_64": ("pathtraced", 64, {"/rtx/pathtracing/spp": 64}, 1)}
    variants = []
    for scale, gamma, op in itertools.product((150.0, 200.0, 300.0, 450.0), (True, False), (6, 1)):
        variants.append(
            {
                "name": f"pt_64_i{scale:g}_g{int(gamma)}_op{op}",
                "mode": "pt_64",
                "top": scale,
                "head": scale * 0.4 / 0.7,
                "dome": 0.0,
                "ambient": 1.0,
                "extra": {
                    "/rtx/post/tonemap/enableSrgbToGamma": gamma,
                    "/rtx/post/tonemap/op": op,
                },
            }
        )
    for v in variants:
        kind, aa, extra, renders = modes[v["mode"]]
        for key, value in {**BASE, **extra}.items():
            settings.set(key, value)
        settings.set("/rtx/sceneDb/ambientLightIntensity", float(v["ambient"]))
        for key, value in v.get("extra", {}).items():
            settings.set(key, value)
        if kind == "realtime":
            rep.settings.set_render_rtx_realtime(antialiasing=aa)
        else:
            rep.settings.set_render_pathtraced(samples_per_pixel=aa)
        top.GetIntensityAttr().Set(float(v["top"]))
        head.GetIntensityAttr().Set(float(v["head"]))
        dome.GetIntensityAttr().Set(float(v["dome"]))
        tr.renders_per_frame = 4
        tr.render()  # settle the new settings
        tr.renders_per_frame = renders
        t0 = time.perf_counter()
        a = tr.render()
        ms = 1e3 * (time.perf_counter() - t0)
        b = tr.render()
        np.save(args.output / f"frame_{v['name']}.npy", np.stack([a, b]))
        record["variants"].append(
            dict(
                v,
                render_ms=ms,
                renders_per_frame=renders,
                second_render_max_abs_diff=int(np.abs(a.astype(int) - b.astype(int)).max()),
                second_render_mean_abs_diff=float(np.abs(a.astype(int) - b.astype(int)).mean()),
                mean=float(a.mean()),
            )
        )
        print("VARIANT", v["name"], float(a.mean()), ms, flush=True)
    seg, labels = tr.segmentation()
    np.save(args.output / "segmentation.npy", seg)
    record["segmentation_labels"] = labels
    record["settings_after"] = {
        k: dump(settings, k) for k in ("/rtx/post/tonemap", "/rtx/pathtracing", "/rtx/rendermode")
    }
    record["object_properties"] = tr.object_properties

    try:
        contact_smoke(tr, manifest, record)
    except Exception:  # noqa: BLE001 - recorded
        import traceback

        record["contact_smoke_error"] = traceback.format_exc()
    (args.output / "probe.json").write_text(json.dumps(record, indent=2, default=str))
    tr.close()


def contact_smoke(tr, manifest, record) -> None:
    tr.reset(0)
    hold = reset_pose(manifest)
    truth0 = tr.task_truth()
    tr.set_object_state([0.40, -0.26, 0.767 + 0.05])
    log = []
    for _ in range(20):
        tr.send_joint_targets(hold, joint_names=tr.joint_names, deadline=tr.time + 0.05)
        t = tr.task_truth()
        log.append(
            {
                "time": tr.time,
                "z": float(t["object_position"][2]),
                "contacts": tr.contacts()["interval_pairs"],
                "last_step": tr.contacts()["last_step"],
            }
        )
    record["contact_smoke"] = {
        "truth_after_reset": {k: str(v) for k, v in truth0.items()},
        "log": log,
        "api_error": tr.contact_api_error,
    }


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
