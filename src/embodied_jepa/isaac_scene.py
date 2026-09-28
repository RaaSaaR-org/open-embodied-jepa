"""Simulator-neutral description of the ``apple-to-plate-v2`` scene and onboard camera for Isaac.

Development tooling for TASK-025. NumPy only at import: it is not part of the core import path
and imports no Isaac, torch or MuJoCo module. ``MuJoCoSimulation`` stays the single authority:

* ``scene_manifest_from_mujoco`` reads the compiled v2 scene (table, floor, apple, plate and its
  16 rim capsules) from a ``MuJoCoSimulation`` switched to v2 by
  ``apple_to_plate_v2.apply_v2_scene``; ``configs/isaac/apple_to_plate_v2_scene_v1.json`` is its
  committed output, checked by ``tests/test_isaac_scene.py``;
* ``camera_manifest_from_mujoco`` reads ``onboard_rgb`` (parent body, pose, fovy, clip planes,
  resolution) and MuJoCo's lights; the committed camera manifest adds the pinned Isaac render
  settings and the declared image-parity metric;
* ``reset_layout`` and ``task_truth_from_state`` reproduce ``MuJoCoSimulation.reset``'s
  apple/plate placement rules and ``MuJoCoSimulation.task_truth``'s scoring flags for the apple
  and plate kinds, so the Isaac transport places and scores objects by the same rules.

``task_truth_from_state`` is evaluator-only simulator truth: it must never reach a model input,
a controller or a planning cost (CLAUDE.md research-evidence rules).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from embodied_jepa.contracts import ContractError

SCENE_MANIFEST_VERSION = "apple_to_plate_v2_isaac_scene_v1"
CAMERA_MANIFEST_VERSION = "onboard_camera_manifest_v1"

# MuJoCoSimulation.reset / task_truth constants for object_kind="apple", container_kind="plate".
TABLE_XY_BOUNDS = ((0.18, 0.65), (-0.32, 0.32))
APPLE_RESET_Z_ON_TABLE = 0.743  # + support height; MuJoCoSimulation.reset
PLATE_BODY_Z = 0.746
CONTAINER_SURFACE_Z = 0.752
OBJECT_SUPPORT_HEIGHT = 0.027
PLATE_INTERIOR_M = 0.063
PLATE_ENVELOPE_M = 0.071
DEFAULT_PLATE_XY = (0.48, -0.10)
LIFTED_Z, DROPPED_Z = 0.82, 0.70
PLACED_RADIUS_M, PLACED_Z_TOL_M, PLACED_SPEED_M_S = 0.04, 0.012, 0.1


def canonical_sha256(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def _f(x) -> list[float]:
    return [float(v) for v in np.asarray(x, dtype=float).ravel()]


def _quat_rotate(q_wxyz, v) -> np.ndarray:
    w, x, y, z = q_wxyz
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return r @ np.asarray(v, dtype=float)


def _geom(model, name: str) -> dict:
    g = model.geom(name)
    gid = int(g.id)
    return {
        "name": name,
        "friction": _f(model.geom_friction[gid]),
        "condim": int(model.geom_condim[gid]),
        "solref": _f(model.geom_solref[gid]),
        "solimp": _f(model.geom_solimp[gid]),
        "margin": float(model.geom_margin[gid]),
        "rgba": _f(model.geom_rgba[gid]),
    }


def scene_manifest_from_mujoco(sim) -> dict:
    """The v2 scene objects as MuJoCo compiled them. ``sim`` must already be switched to v2."""
    model = sim.model
    apple = model.geom("apple_geom")
    if int(model.geom_condim[apple.id]) != 6:
        raise ContractError("scene_manifest_from_mujoco needs the v2 scene (apple condim 6)")
    if sim.object_kind != "apple" or sim.container_kind != "plate":
        raise ContractError("only the apple/plate scene is described")
    table = model.geom("table")
    floor = model.geom("floor")
    base = model.geom("plate_base")
    rims = []
    for i in range(16):
        r = model.geom(f"plate_rim_{i}")
        pos = np.asarray(model.geom_pos[r.id], dtype=float)
        half = float(model.geom_size[r.id][1])
        axis = _quat_rotate(model.geom_quat[r.id], [0.0, 0.0, 1.0])
        rims.append(
            {
                "from": _f(pos - half * axis),
                "to": _f(pos + half * axis),
                "radius": float(model.geom_size[r.id][0]),
            }
        )
    rim_geom = _geom(model, "plate_rim_0")
    return {
        "version": SCENE_MANIFEST_VERSION,
        "scene_version": "apple_to_plate_v2",
        "units": "m, kg, s; world frame +X forward, +Y left, +Z up (MuJoCoSimulation world)",
        "mujoco_option": {
            "timestep_s": float(model.opt.timestep),
            "gravity": _f(model.opt.gravity),
            "integrator": int(model.opt.integrator),
            "cone": int(model.opt.cone),
            "impratio": float(model.opt.impratio),
            "noslip_iterations": int(model.opt.noslip_iterations),
            "contact_friction_combination": "per-component max of the two geoms; condim max",
        },
        "floor": {
            **_geom(model, "floor"),
            "type": "plane",
            "z": float(model.geom_pos[floor.id][2]),
        },
        "table": {
            **_geom(model, "table"),
            "type": "box",
            "pos": _f(model.geom_pos[table.id]),
            "half_size": _f(model.geom_size[table.id]),
        },
        "apple": {
            **_geom(model, "apple_geom"),
            "type": "sphere",
            "radius": float(model.geom_size[apple.id][0]),
            "mass": float(model.body("apple").mass[0]),
            "inertia_diag": _f(model.body("apple").inertia),
            "free_joint": "apple_free",
        },
        "plate": {
            "body": "static (no joint); MuJoCoSimulation.reset moves model.body('plate').pos",
            "body_z": float(model.body("plate").pos[2]),
            "base": {
                **_geom(model, "plate_base"),
                "type": "cylinder",
                "radius": float(model.geom_size[base.id][0]),
                "half_height": float(model.geom_size[base.id][1]),
                "pos": _f(model.geom_pos[base.id]),
            },
            "rim": {
                "friction": rim_geom["friction"],
                "condim": rim_geom["condim"],
                "rgba": rim_geom["rgba"],
                "type": "capsule",
                "capsules": rims,
            },
        },
        "task_constants": {
            "table_top_z": float(model.geom_pos[table.id][2] + model.geom_size[table.id][2]),
            "apple_reset_z_on_table": APPLE_RESET_Z_ON_TABLE + sim.object_support_height,
            "container_surface_z": float(sim.container_surface_z),
            "object_support_height": float(sim.object_support_height),
        },
    }


def validate_scene_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict) or manifest.get("version") != SCENE_MANIFEST_VERSION:
        raise ContractError("unsupported or missing scene manifest version")
    for key in ("floor", "table", "apple", "plate", "task_constants", "isaac_physx"):
        if key not in manifest:
            raise ContractError(f"scene manifest is missing {key!r}")
    apple = manifest["apple"]
    if apple["condim"] != 6 or not np.allclose(apple["friction"], [1.0, 0.01, 0.001]):
        raise ContractError("scene manifest does not describe the v2 apple contact")
    if len(manifest["plate"]["rim"]["capsules"]) != 16:
        raise ContractError("the plate rim has 16 capsules")
    values = [apple["radius"], apple["mass"], *apple["inertia_diag"]]
    if not (np.isfinite(values).all() and min(values) > 0):
        raise ContractError("apple geometry and mass must be positive and finite")
    return manifest


def camera_manifest_from_mujoco(sim, camera: str = "onboard_rgb") -> dict:
    """``onboard_rgb`` geometry and MuJoCo's lighting, as the compiled model has them."""
    model = sim.model
    cam = model.camera(camera)
    body = model.body(int(cam.bodyid[0])).name
    extent = float(model.stat.extent)
    quat = np.asarray(cam.quat, dtype=float)
    return {
        "name": camera,
        "parent_body": body,
        "pos_m": _f(cam.pos),
        "quat_wxyz": _f(quat),
        "fovy_deg": float(cam.fovy[0]),
        "width": int(sim.width),
        "height": int(sim.height),
        "znear_m": float(model.vis.map.znear) * extent,
        "zfar_m": float(model.vis.map.zfar) * extent,
        "frame_convention": "looks along -Z, +Y up (MuJoCo and USD/OpenGL cameras agree)",
        "output": "RGB uint8 HWC",
        "mujoco_lighting": {
            "lights": [
                {
                    "directional": bool(model.light_type[i] == 1)
                    if hasattr(model, "light_type")
                    else bool(model.light_directional[i]),
                    "dir": _f(model.light_dir[i]),
                    "diffuse": _f(model.light_diffuse[i]),
                    "ambient": _f(model.light_ambient[i]),
                    "specular": _f(model.light_specular[i]),
                    "castshadow": bool(model.light_castshadow[i]),
                }
                for i in range(model.nlight)
            ],
            "headlight": {
                "active": bool(model.vis.headlight.active),
                "ambient": _f(model.vis.headlight.ambient),
                "diffuse": _f(model.vis.headlight.diffuse),
                "specular": _f(model.vis.headlight.specular),
            },
            "colour_pipeline": "fixed-function OpenGL: rgba * light, clamped, no tonemap/gamma",
        },
    }


def validate_camera_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict) or manifest.get("version") != CAMERA_MANIFEST_VERSION:
        raise ContractError("unsupported or missing camera manifest version")
    for key in ("camera", "isaac_render", "image_parity"):
        if key not in manifest:
            raise ContractError(f"camera manifest is missing {key!r}")
    cam = manifest["camera"]
    q = np.asarray(cam["quat_wxyz"], dtype=float)
    if q.shape != (4,) or not np.isclose(np.linalg.norm(q), 1.0, atol=1e-6):
        raise ContractError("camera quaternion must be a unit quaternion")
    for key in ("width", "height"):
        if type(cam[key]) is not int or cam[key] < 1:
            raise ContractError("camera resolution must be positive integers")
    if not 0 < cam["fovy_deg"] < 180 or not 0 < cam["znear_m"] < cam["zfar_m"]:
        raise ContractError("camera fovy/clip planes are inconsistent")
    return manifest


def camera_quat_xyzw(manifest: dict) -> tuple[float, float, float, float]:
    """The ``onboard_rgb`` rotation in its parent body frame, in Isaac Lab 3's (x, y, z, w)."""
    w, x, y, z = manifest["camera"]["quat_wxyz"]
    return (float(x), float(y), float(z), float(w))


def pinhole_from_fovy(fovy_deg: float, width: int, height: int, aperture_mm: float = 20.955):
    """Focal length and apertures (mm) giving MuJoCo's vertical fovy with square pixels."""
    vertical = aperture_mm * height / width
    focal = vertical / (2.0 * np.tan(np.deg2rad(fovy_deg) / 2.0))
    return float(focal), float(aperture_mm), float(vertical)


def reset_layout(seed=0, *, object_xy=None, plate_xy=None, object_on_container=False):
    """``MuJoCoSimulation.reset``'s apple/plate placement: (apple xyz, plate xyz), world frame.

    Same defaults (seeded apple jitter around (0.40, -0.26), plate (0.48, -0.10)), the same
    tabletop bounds and the same overlap/fit checks, raising ``ContractError`` where it raises.
    """
    if type(object_on_container) is not bool:
        raise ContractError("object_on_container must be an explicit boolean")
    rng = np.random.default_rng(seed)
    plate = np.asarray(plate_xy if plate_xy is not None else DEFAULT_PLATE_XY, dtype=float)
    if object_xy is None and object_on_container:
        object_xy = plate.copy()
    obj = np.asarray(
        object_xy
        if object_xy is not None
        else [0.40 + rng.uniform(-0.02, 0.02), -0.26 + rng.uniform(-0.02, 0.02)],
        dtype=float,
    )
    (x0, x1), (y0, y1) = TABLE_XY_BOUNDS
    for name, xy in (("object", obj), ("plate", plate)):
        if (
            xy.shape != (2,)
            or not np.isfinite(xy).all()
            or not (x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1)
        ):
            raise ContractError(f"{name} reset xy must lie on the tabletop")
    offset = np.abs(obj - plate)
    if object_on_container:
        if float(np.linalg.norm(offset)) + OBJECT_SUPPORT_HEIGHT > PLATE_INTERIOR_M - 0.001:
            raise ContractError(
                "supported placement must fit inside container without rim penetration"
            )
        z = CONTAINER_SURFACE_Z + OBJECT_SUPPORT_HEIGHT + 0.001
    else:
        separation = np.linalg.norm(offset) - PLATE_ENVELOPE_M - OBJECT_SUPPORT_HEIGHT
        if separation < 0.005:
            raise ContractError(
                "object/container reset overlaps collision envelopes; separate them "
                "or use object_on_container=True for a supported goal"
            )
        z = APPLE_RESET_Z_ON_TABLE + OBJECT_SUPPORT_HEIGHT
    return np.array([*obj, z]), np.array([*plate, PLATE_BODY_Z])


def task_truth_from_state(
    *,
    object_position,
    object_velocity,
    plate_position,
    hand_contact: bool,
    base_position,
    base_rotation,
    timestamp: float,
) -> dict:
    """``MuJoCoSimulation.task_truth``'s dict for the apple/plate kinds. Evaluator-only."""
    apple = np.asarray(object_position, dtype=float).copy()
    plate = np.asarray(plate_position, dtype=float).copy()
    velocity = np.asarray(object_velocity, dtype=float).copy()
    return {
        "position_frame": "world",
        "object_kind": "apple",
        "container_kind": "plate",
        "object_support_height": OBJECT_SUPPORT_HEIGHT,
        "container_surface_z": CONTAINER_SURFACE_Z,
        "base_position_world": np.asarray(base_position, dtype=float).copy(),
        "base_rotation_world": np.asarray(base_rotation, dtype=float).copy(),
        "object_position": apple,
        "plate_position": plate,
        "object_velocity": velocity,
        "hand_contact": bool(hand_contact),
        "lifted": bool(apple[2] > LIFTED_Z),
        "placed": bool(
            np.linalg.norm(apple[:2] - plate[:2]) < PLACED_RADIUS_M
            and abs(apple[2] - (CONTAINER_SURFACE_Z + OBJECT_SUPPORT_HEIGHT)) < PLACED_Z_TOL_M
            and np.linalg.norm(velocity) < PLACED_SPEED_M_S
            and not hand_contact
        ),
        "dropped": bool(apple[2] < DROPPED_Z),
        "timestamp": float(timestamp),
    }


def is_hand_body(name: str) -> bool:
    """``MuJoCoSimulation.task_truth``'s hand-contact rule: hand or wrist links."""
    return "hand_" in name or "wrist_" in name


def settle_time(times, speeds, *, threshold: float = 0.001) -> float | None:
    """First time after which the speed stays at or below ``threshold`` to the end, or None."""
    speeds = np.asarray(speeds, dtype=float)
    above = np.flatnonzero(speeds > threshold)
    if above.size == 0:
        return float(times[0])
    if above[-1] == speeds.size - 1:
        return None
    return float(times[above[-1] + 1])


# ----- the PhysX mapping (hand-written; committed into the scene manifest) ---------------------
# MuJoCo combines contact friction per component by max; PhysX's "max" combine mode is the same
# rule for its one sliding coefficient. MuJoCo has one sliding coefficient; PhysX's static and
# dynamic coefficients are both set to it. Restitution 0: MuJoCo's default contact (solref
# 0.02 s, damping ratio 1) is critically damped and does not bounce.
_MAT_1 = {
    "static_friction": 1.0,
    "dynamic_friction": 1.0,
    "restitution": 0.0,
    "friction_combine_mode": "max",
    "restitution_combine_mode": "max",
}
ISAAC_PHYSX = {
    "default_material": dict(_MAT_1),
    "floor_material": dict(_MAT_1),
    "table_material": dict(_MAT_1),
    "apple": {
        "material": dict(_MAT_1),
        # MuJoCo condim 6 torsional coefficient 0.01 m bounds the spin torque by 0.01 N;
        # PhysX bounds it by mu N r with the patch radius r, so r = 0.01 m / mu (mu = 1).
        "torsional_patch_radius": 0.01,
        "min_torsional_patch_radius": 0.01,
        "rolling_friction": None,  # PhysX rigid bodies have no rolling friction
        "linear_damping": 0.0,  # PhysX default 0; MuJoCo has no body damping
        "angular_damping": 0.0,  # PhysX default 0.05; MuJoCo has none
        "sleep_threshold": 0.0,  # MuJoCo does not sleep bodies; PhysX would
        "max_depenetration_velocity": 1.0,
        "rest_offset": 0.0,
        "contact_offset": None,  # PhysX default (auto)
    },
    "plate": {
        "body": "kinematic rigid body (MuJoCo: static body moved at reset)",
        "base_material": dict(_MAT_1),
        "rim_material": dict(_MAT_1),
        "base_collision": "convex hull of the USD cylinder (Isaac Lab disables PhysX custom "
        "cylinder geometry)",
        "rim_collision": "native PhysX capsules",
    },
    "solver": "PhysX TGS, GPU pipeline, dt 0.002 s, default 4 position / 1 velocity iterations",
}

# Parameters PhysX cannot express as MuJoCo does. Reported with every run.
UNMATCHED = [
    "rolling friction (MuJoCo condim 6, 0.001 m on the apple): PhysX rigid bodies have none; "
    "not substituted (angular damping would be viscous, not Coulomb, and act in free flight)",
    "torsional friction: MuJoCo 0.01 m as a Coulomb torque bound 0.01 N; PhysX torsional "
    "patch radius 0.01 m (bound mu N r), nominally equal, different algorithm; the PhysX "
    "per-pair combination of patch radii is assumed to be max (set on the apple only)",
    "contact softness: MuJoCo solref (0.02 s, 1) / solimp (0.9, 0.95, 0.001) soft contacts "
    "allow ~mm penetration; PhysX contacts are rigid with contact/rest offsets",
    "friction cone: MuJoCo pyramidal (cone 0) with one coefficient; PhysX patch friction "
    "with static and dynamic coefficients (both set to 1)",
    "plate base: MuJoCo analytic cylinder; PhysX convex hull of the USD cylinder",
    "robot collision: MuJoCo self-collision on; Isaac self-collision off (importer default)",
    "robot geoms: MuJoCo friction (1, 0.005, 0.0001), condim 3; PhysX friction from the "
    "converter's bindings or the scene default material (read back in the run record)",
    "integrator: MuJoCo implicitfast vs PhysX TGS",
]

# ----- the pinned Isaac render settings (hand-written; committed into the camera manifest) ------
# Chosen in development render probes (outputs/isaac-v2-render-probe-{1..4}, 2026-09-29) on ONE
# frame only: the seed-0 reset scene (robot reset pose, v2 apple and plate at their defaults).
# The real-time path tracer (RT2) was grainy at 112 px with anti-aliasing off (mean |frame -
# next render of the same state| 10-19 /255) and DLAA still changed between renders (1.6-2.3);
# the path tracer at 64 samples per pixel in one frame, OptiX-denoised, rendered the same state
# identically twice. Light intensity (top 300, headlight 300 * 0.4 / 0.7 as MuJoCo's diffuse
# ratio) was the lowest mean absolute difference to MuJoCo's frame of the grid 150/200/300/450
# x gamma on/off x tonemapper 6/1. The ambient light is ignored by the path tracer.
_TOP_INTENSITY = 300.0
ISAAC_RENDER = {
    "status": "pinned",
    "renderer": "RTX path tracer, 64 samples per pixel in one frame, OptiX denoiser",
    "carb_settings": {
        "/rtx/rendermode": "PathTracing",
        "/rtx/pathtracing/spp": 64,
        "/rtx/pathtracing/totalSpp": 64,
        "/rtx/pathtracing/maxBounces": 3,
        "/rtx/pathtracing/optixDenoiser/enabled": True,
        "/rtx/post/tonemap/op": 6,
        "/rtx/post/tonemap/enableSrgbToGamma": True,
        "/rtx/post/histogram/enabled": False,
        "/rtx-transient/dlssg/enabled": False,
    },
    "warmup_renders": 1,
    "renders_per_frame": 1,
    "data_types": ["rgb", "instance_id_segmentation_fast"],
    "lights": [
        {
            "type": "distant",
            "path": "/World/light_top",
            "intensity": _TOP_INTENSITY,
            "angle_deg": 0.0,
            "color": [1.0, 1.0, 1.0],
            "shadow": True,
            "direction": [0.0, 0.0, -1.0],
            "mujoco": "light dir 0 0 -1, diffuse 0.7, castshadow",
        },
        {
            "type": "distant",
            "path": "{camera}/headlight",
            "intensity": _TOP_INTENSITY * 0.4 / 0.7,
            "angle_deg": 0.0,
            "color": [1.0, 1.0, 1.0],
            "shadow": False,
            "mujoco": "headlight diffuse 0.4 (ambient 0.1 has no path-traced equivalent)",
        },
    ],
    "floor": "a 4 m x 4 m x 2 cm box with its top at z = 0 (MuJoCo: plane), MuJoCo rgba",
    "materials": "UsdPreviewSurface, MuJoCo rgb as diffuse, roughness 0.5, metallic 0",
}

# The declared image-parity metric. Fixed before the first image-parity run; the render
# settings above were chosen on the seed-0 reset frame, so read 0 of a parity run (the same
# robot pose) is reported separately from the frames the choice never saw.
IMAGE_PARITY = {
    "reference": "MuJoCoSimulation onboard_rgb at the same resolution and joint state, v2 scene",
    "held_out_reads": [30, 50, 70, 90],
    "calibration_read": 0,
    "metrics": {
        "mad": "mean |isaac - mujoco| over all pixels and RGB channels, uint8 units",
        "psnr_db": "10 log10(255^2 / mean squared difference)",
        "ssim_luma": "SSIM of Rec.601 luma, 7x7 uniform window, K1 0.01, K2 0.03",
        "mask_iou": "per-class IoU of segmentation masks: robot, table, apple, plate, "
        "floor_or_background",
        "centroid_px": "apple and plate mask centroid distance, pixels",
    },
    "thresholds": {
        "geometry": {
            "robot_iou_min": 0.90,
            "table_iou_min": 0.90,
            "apple_iou_min": 0.80,
            "plate_iou_min": 0.80,
            "centroid_px_max": 1.0,
        },
        "photometric": {"mad_max": 15.0, "ssim_luma_min": 0.80},
    },
    "verdict_rule": "geometry PASS iff every held-out read meets every geometry bar; "
    "photometric PASS iff every held-out read meets both photometric bars",
}


def build_scene_manifest(sim) -> dict:
    """The committed scene manifest: MuJoCo-derived scene plus the PhysX mapping."""
    manifest = scene_manifest_from_mujoco(sim)
    manifest["isaac_physx"] = ISAAC_PHYSX
    manifest["unmatched_in_physx"] = UNMATCHED
    return validate_scene_manifest(manifest)


def build_camera_manifest(sim) -> dict:
    """The committed camera manifest: MuJoCo-derived camera plus the pinned Isaac rendering."""
    return validate_camera_manifest(
        {
            "version": CAMERA_MANIFEST_VERSION,
            "camera": camera_manifest_from_mujoco(sim),
            "isaac_render": ISAAC_RENDER,
            "image_parity": IMAGE_PARITY,
        }
    )
