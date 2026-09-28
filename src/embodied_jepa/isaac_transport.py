"""Minimal Isaac Sim transport for the fixed-pelvis G1 + dual Dex3 (development, TASK-025).

This module is importable without Isaac: every Isaac Sim / Isaac Lab / torch / warp import
is inside ``IsaacTransport``. The transport itself only runs inside the ``isaaclab_arena``
container (Isaac Sim 6.0.0-rc.22, Isaac Lab 3.0.0) after the caller has started the Kit app
(``isaaclab.app.AppLauncher``); no Isaac dependency is declared in ``pyproject.toml`` or
``uv.lock``. See ``docs/ISAAC_MJCF_TRANSPORT.md``.

Scope, as in ``docs/ISAAC_PORT.md``: ``read`` / ``send_joint_targets`` / ``stop`` / ``reset``
/ ``close``, plus the evaluator-only ``task_truth`` and ``contacts``. The robot is the USD
converted from the project's pinned MJCF (``scripts/isaac/convert_mjcf_to_usd.py``). The floor
and table always exist; with ``objects=True`` the ``apple-to-plate-v2`` apple and plate are
added from ``configs/isaac/apple_to_plate_v2_scene_v1.json`` and ``reset(object_xy=...,
plate_xy=...)`` places them by ``MuJoCoSimulation.reset``'s rules (``isaac_scene``). The
onboard camera and the pinned render settings come from ``configs/isaac/onboard_camera_v1.json``.
PhysX cannot express every MuJoCo contact parameter; see ``docs/ISAAC_V2_SCENE.md``.

Control mirrors ``MuJoCoSimulation.send_joint_targets``: per physics substep, linearly
interpolated targets, ``kp (q* - q) - kd qd + bias`` with the simulator's own gravity and
Coriolis compensation, clipped to the MJCF actuator ``ctrlrange``, applied as joint efforts.
MuJoCo's passive joint damping is applied as a zero-stiffness PhysX drive. MuJoCo's
``frictionloss`` (a Coulomb torque in N m) is applied by default
(``joint_friction="frictionloss"``) as PhysX's static and dynamic joint friction *efforts*,
the closest available PhysX model but not the same algorithm; ``joint_friction="none"``
leaves joint friction out. Either way the converter's load-proportional PhysX friction
coefficient is zeroed. Damping and friction are read back after every reset. Joints are
mapped by name; the canonical order is the MJCF actuator order that
``MuJoCoSimulation.joint_names`` uses.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_jepa.contracts import ContractError

# v1 renamed the per-joint key ``frictionloss_not_applied`` (v0, a misnomer: the default mode
# applies it) to ``frictionloss`` and made it required. The v0 file stays committed because
# the recorded parity runs of 2026-09-28 reference its hash.
JOINT_MANIFEST_VERSION = "g1_dex3_joint_manifest_v1"
_MANIFEST_KEYS = (
    "version",
    "source_mjcf_sha256",
    "physics_dt_s",
    "control_dt_s",
    "pelvis_pos_m",
    "reset_elbow_rad",
    "joints",
)
_JOINT_KEYS = (
    "name",
    "lower",
    "upper",
    "ctrl_min",
    "ctrl_max",
    "kp",
    "kd",
    "damping",
    "frictionloss",
)
_RESET_ELBOW_RAD = 0.08  # MuJoCoSimulation.reset: slight elbow flexion


def joint_manifest_from_mujoco(sim) -> dict:
    """Name-addressed joint manifest from a ``MuJoCoSimulation`` (the single authority)."""
    model = sim.model
    mjcf = Path(sim.asset_root) / "unitree_robots/g1/g1_29dof_with_hand.xml"
    joints = []
    for a, (name, jid) in enumerate(zip(sim.joint_names, sim.joint_ids, strict=True)):
        jid = int(jid)
        joints.append(
            {
                "name": name,
                "lower": float(model.jnt_range[jid, 0]),
                "upper": float(model.jnt_range[jid, 1]),
                "ctrl_min": float(model.actuator_ctrlrange[a, 0]),
                "ctrl_max": float(model.actuator_ctrlrange[a, 1]),
                "kp": float(sim.kp[a]),
                "kd": float(sim.kd[a]),
                "damping": float(model.dof_damping[int(model.jnt_dofadr[jid])]),
                "frictionloss": float(model.dof_frictionloss[int(model.jnt_dofadr[jid])]),
            }
        )
    return {
        "version": JOINT_MANIFEST_VERSION,
        "source_mjcf_sha256": hashlib.sha256(mjcf.read_bytes()).hexdigest(),
        "physics_dt_s": float(model.opt.timestep),
        "control_dt_s": float(sim.control_dt),
        "pelvis_pos_m": [float(v) for v in model.body("pelvis").pos],
        "reset_elbow_rad": _RESET_ELBOW_RAD,
        "joints": joints,
    }


def reset_pose(manifest: dict) -> np.ndarray:
    """MuJoCoSimulation's reset joint pose in the manifest's canonical order."""
    names = [j["name"] for j in manifest["joints"]]
    q = np.zeros(len(names))
    for side in ("left", "right"):
        q[names.index(f"{side}_elbow_joint")] = manifest["reset_elbow_rad"]
    return q


def manifest_sha256(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def validate_joint_manifest(manifest: dict) -> dict:
    """Reject a manifest with missing fields, duplicates or non-finite/inconsistent values."""
    if not isinstance(manifest, dict) or any(k not in manifest for k in _MANIFEST_KEYS):
        raise ContractError("joint manifest is missing required fields")
    if manifest["version"] != JOINT_MANIFEST_VERSION:
        raise ContractError(f"unsupported joint manifest version {manifest['version']!r}")
    joints = manifest["joints"]
    names = [j.get("name") for j in joints]
    if len(joints) != 43 or len(set(names)) != 43:
        raise ContractError("joint manifest must name 43 distinct robot joints")
    for j in joints:
        if any(k not in j for k in _JOINT_KEYS):
            raise ContractError(f"joint {j.get('name')!r} is missing fields")
        values = np.array([j[k] for k in _JOINT_KEYS[1:]], dtype=float)
        if not np.isfinite(values).all():
            raise ContractError(f"joint {j['name']!r} has non-finite values")
        if not (j["lower"] < j["upper"] and j["ctrl_min"] < 0 < j["ctrl_max"]):
            raise ContractError(f"joint {j['name']!r} has inconsistent limits")
        if j["kp"] < 0 or j["kd"] < 0 or j["damping"] < 0 or j["frictionloss"] < 0:
            raise ContractError(f"joint {j['name']!r} has negative gains")
    if not np.isfinite(float(manifest["reset_elbow_rad"])):
        raise ContractError("reset_elbow_rad must be finite")
    dt, control = float(manifest["physics_dt_s"]), float(manifest["control_dt_s"])
    substeps = round(control / dt)
    if dt <= 0 or substeps < 1 or not np.isclose(substeps * dt, control):
        raise ContractError("control_dt must be a positive multiple of physics_dt")
    return manifest


def name_map(canonical: list[str] | tuple[str, ...], simulator: list[str]) -> np.ndarray:
    """Indices ``idx`` with ``simulator[idx[i]] == canonical[i]``; the sets must be equal."""
    if len(set(simulator)) != len(simulator) or set(canonical) != set(simulator):
        missing = sorted(set(canonical) - set(simulator))
        extra = sorted(set(simulator) - set(canonical))
        raise ContractError(f"joint names differ: missing {missing}, unexpected {extra}")
    lookup = {n: i for i, n in enumerate(simulator)}
    return np.array([lookup[n] for n in canonical], dtype=np.int64)


def onboard_camera_offset() -> tuple[tuple[float, float, float], tuple[float, ...]]:
    """MuJoCo ``onboard_rgb`` on torso_link: pos .08 0 .35, xyaxes 0 -1 0 .866 0 .5.

    MuJoCo and USD/OpenGL cameras share the frame convention (look along -Z, +Y up), so the
    rotation is the matrix whose columns are the MuJoCo camera x, y, z axes. Returns the
    position and the quaternion in Isaac Lab 3's (x, y, z, w) order.
    """
    x = np.array([0.0, -1.0, 0.0])
    y = np.array([0.866, 0.0, 0.5])
    y /= np.linalg.norm(y)
    m = np.stack([x, y, np.cross(x, y)], axis=1)
    w = 0.5 * np.sqrt(max(1.0 + np.trace(m), 0.0))
    q = np.array(
        [
            (m[2, 1] - m[1, 2]) / (4 * w),
            (m[0, 2] - m[2, 0]) / (4 * w),
            (m[1, 0] - m[0, 1]) / (4 * w),
            w,
        ]
    )
    q /= np.linalg.norm(q)
    return (0.08, 0.0, 0.35), tuple(float(v) for v in q)


class IsaacTransport:
    """Fixed-pelvis G1 + Dex3 in Isaac Sim with the MuJoCo transport's control semantics.

    ``scene_manifest`` (``configs/isaac/apple_to_plate_v2_scene_v1.json``) gives the floor, the
    table and, with ``objects=True``, the v2 apple and plate with their PhysX mapping.
    ``camera_manifest`` (``configs/isaac/onboard_camera_v1.json``) gives ``onboard_rgb`` and the
    pinned render settings; it is required when ``render=True``. ``read()`` never contains
    object state; object and contact truth is only available from ``task_truth()`` and
    ``contacts()``, which are evaluator-only.
    """

    clock_domain = "isaac_episode_sim_time"

    def __init__(
        self,
        *,
        usd_path: str,
        manifest: dict,
        scene_manifest: dict,
        camera_manifest: dict | None = None,
        objects: bool = False,
        render: bool = True,
        record_contacts: bool = False,
        device: str = "cuda:0",
        joint_friction: str = "frictionloss",
    ):
        from embodied_jepa import isaac_scene as scene

        self.manifest = validate_joint_manifest(manifest)
        self.scene = scene.validate_scene_manifest(scene_manifest)
        if render and camera_manifest is None:
            raise ContractError("rendering needs a camera manifest")
        self.camera_manifest = (
            scene.validate_camera_manifest(camera_manifest) if camera_manifest else None
        )
        if joint_friction not in ("frictionloss", "none"):
            raise ContractError("joint_friction must be 'frictionloss' or 'none'")
        self.joint_friction = joint_friction
        self.objects = bool(objects)
        # Object truth (hand contact) needs contact reports whenever objects exist.
        self.record_contacts = bool(record_contacts or objects)
        import isaaclab.sim as sim_utils
        import torch
        import warp as wp
        from isaaclab.actuators import IdealPDActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg

        self._torch, self._wp, self._sim_utils = torch, wp, sim_utils
        joints = manifest["joints"]
        self.joint_names = tuple(j["name"] for j in joints)
        self.lower = np.array([j["lower"] for j in joints])
        self.upper = np.array([j["upper"] for j in joints])
        self.ctrl_min = np.array([j["ctrl_min"] for j in joints])
        self.ctrl_max = np.array([j["ctrl_max"] for j in joints])
        self.physics_dt = float(manifest["physics_dt_s"])
        self.control_dt = float(manifest["control_dt_s"])
        self.substeps = round(self.control_dt / self.physics_dt)
        self.render_enabled = render
        self.closed = False
        self.stopped_reason = ""
        physx = self.scene["isaac_physx"]
        default = physx["default_material"]
        if self.camera_manifest is not None:
            r = self.camera_manifest["isaac_render"]
            self.width = int(self.camera_manifest["camera"]["width"])
            self.height = int(self.camera_manifest["camera"]["height"])
            self.warmup_renders = int(r["warmup_renders"])
            self.renders_per_frame = int(r["renders_per_frame"])
        cfg = sim_utils.SimulationCfg(
            dt=self.physics_dt,
            device=device,
            gravity=tuple(self.scene["mujoco_option"]["gravity"]),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=default["static_friction"],
                dynamic_friction=default["dynamic_friction"],
                restitution=default["restitution"],
                friction_combine_mode=default["friction_combine_mode"],
                restitution_combine_mode=default["restitution_combine_mode"],
            ),
        )
        self.sim = sim_utils.SimulationContext(cfg)
        self.render_settings = self._apply_render_settings() if self.camera_manifest else {}
        if self.record_contacts:
            # Isaac Lab turns PhysX contact processing off unless a contact sensor asks for it.
            self.sim.set_setting("/physics/disableContactProcessing", False)
        self._spawn_static_scene()
        effort = {j["name"]: max(-j["ctrl_min"], j["ctrl_max"]) for j in joints}
        robot_cfg = ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True),
            ),
            # The converted USD keeps the MJCF pelvis pos on the pelvis prim: no spawn offset.
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0, 0, 0, 1)),
            actuators={
                "all": IdealPDActuatorCfg(
                    joint_names_expr=[".*"], stiffness=0.0, damping=0.0, effort_limit=effort
                )
            },
        )
        self.robot = Articulation(robot_cfg)
        if self.record_contacts:
            self._activate_link_contact_reports()
        self.apple = self.plate = None
        if self.objects:
            self._spawn_objects()
        self.camera = None
        if render:
            self._spawn_camera_and_lights()
        self.sim.reset()
        self.isaac_joint_names = tuple(self.robot.joint_names)
        self._idx = name_map(self.joint_names, list(self.isaac_joint_names))
        lim = self._np(self.robot.data.joint_pos_limits)[0][self._idx]
        if np.abs(lim - np.stack([self.lower, self.upper], axis=1)).max() > 1e-4:
            raise ContractError("Isaac joint limits differ from the joint manifest")
        # MuJoCo passive damping as a zero-stiffness PhysX drive (implicit, like MuJoCo's).
        self._damping = np.array([j["damping"] for j in joints])
        damping = np.zeros(len(self._idx))
        damping[self._idx] = self._damping
        self.robot.write_joint_damping_to_sim_index(
            damping=torch.as_tensor(damping, dtype=torch.float32, device=self.sim.device)[None]
        )
        self._set_joint_friction(joints)
        self._kp = np.array([j["kp"] for j in joints])
        self._kd = np.array([j["kd"] for j in joints])
        self.targets = np.zeros(len(joints))
        self.time = 0.0
        self._last_contacts: list[dict] = []
        self._interval_pairs: set[tuple[str, str]] = set()
        self.contact_api_error = None
        self._contact_views = []
        if self.record_contacts:
            try:
                self._make_contact_views()
            except Exception as exc:  # noqa: BLE001 - recorded; contacts then unavailable
                self.contact_api_error = f"{type(exc).__name__}: {exc}"
        self.object_properties = self._read_object_properties() if self.objects else None
        self.reset()

    # ------------------------------------------------------------------ scene construction
    def _material(self, spec: dict):
        return self._sim_utils.RigidBodyMaterialCfg(
            static_friction=spec["static_friction"],
            dynamic_friction=spec["dynamic_friction"],
            restitution=spec["restitution"],
            friction_combine_mode=spec["friction_combine_mode"],
            restitution_combine_mode=spec["restitution_combine_mode"],
        )

    def _preview(self, rgba):
        return self._sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(float(v) for v in rgba[:3]), roughness=0.5, metallic=0.0
        )

    def _spawn_static_scene(self) -> None:
        """Floor (a thin box, top at z = 0) and the table as kinematic (immovable) bodies.

        Kinematic rather than static so that PhysX contact views can name them as filters."""
        su, physx = self._sim_utils, self.scene["isaac_physx"]
        fixed = su.RigidBodyPropertiesCfg(kinematic_enabled=True, sleep_threshold=0.0)
        floor, table = self.scene["floor"], self.scene["table"]
        floor_cfg = su.CuboidCfg(
            size=(4.0, 4.0, 0.02),
            rigid_props=fixed,
            collision_props=su.CollisionPropertiesCfg(),
            physics_material=self._material(physx["floor_material"]),
            visual_material=self._preview(floor["rgba"]),
        )
        floor_cfg.func("/World/floor", floor_cfg, translation=(0.0, 0.0, floor["z"] - 0.01))
        table_cfg = su.CuboidCfg(
            size=tuple(2.0 * float(v) for v in table["half_size"]),
            rigid_props=fixed,
            collision_props=su.CollisionPropertiesCfg(),
            physics_material=self._material(physx["table_material"]),
            visual_material=self._preview(table["rgba"]),
        )
        table_cfg.func("/World/table", table_cfg, translation=tuple(table["pos"]))

    def _spawn_objects(self) -> None:
        """The v2 apple (dynamic sphere) and plate (kinematic cylinder + 16 rim capsules)."""
        su = self._sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab.sim import schemas

        physx, apple = self.scene["isaac_physx"], self.scene["apple"]
        a = physx["apple"]
        apple_cfg = su.SphereCfg(
            radius=float(apple["radius"]),
            rigid_props=su.RigidBodyPropertiesCfg(
                linear_damping=a["linear_damping"],
                angular_damping=a["angular_damping"],
                sleep_threshold=a["sleep_threshold"],
                max_depenetration_velocity=a["max_depenetration_velocity"],
            ),
            mass_props=su.MassPropertiesCfg(mass=float(apple["mass"])),
            collision_props=su.CollisionPropertiesCfg(
                torsional_patch_radius=a["torsional_patch_radius"],
                min_torsional_patch_radius=a["min_torsional_patch_radius"],
                rest_offset=a["rest_offset"],
            ),
            physics_material=self._material(a["material"]),
            visual_material=self._preview(apple["rgba"]),
            activate_contact_sensors=True,
        )
        self.apple = RigidObject(
            RigidObjectCfg(
                prim_path="/World/Apple",
                spawn=apple_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.40, -0.26, 0.77)),
            )
        )
        plate = self.scene["plate"]
        p = physx["plate"]
        stage = su.get_current_stage()
        from pxr import UsdGeom

        UsdGeom.Xform.Define(stage, "/World/Plate")
        base = plate["base"]
        base_cfg = su.CylinderCfg(
            radius=float(base["radius"]),
            height=2.0 * float(base["half_height"]),
            collision_props=su.CollisionPropertiesCfg(),
            physics_material=self._material(p["base_material"]),
            visual_material=self._preview(base["rgba"]),
        )
        base_cfg.func("/World/Plate/base", base_cfg, translation=tuple(base["pos"]))
        rim = plate["rim"]
        rim_material = self._material(p["rim_material"])
        for i, cap in enumerate(rim["capsules"]):
            a0, a1 = np.asarray(cap["from"]), np.asarray(cap["to"])
            d = a1 - a0
            length = float(np.linalg.norm(d))
            q = _quat_z_to(d / length)
            cap_cfg = su.CapsuleCfg(
                radius=float(cap["radius"]),
                height=length,
                axis="Z",
                collision_props=su.CollisionPropertiesCfg(),
                physics_material=rim_material,
                visual_material=self._preview(rim["rgba"]),
            )
            cap_cfg.func(
                f"/World/Plate/rim_{i:02d}",
                cap_cfg,
                translation=tuple(float(v) for v in (a0 + a1) / 2),
                orientation=q,
            )
        schemas.define_rigid_body_properties(
            "/World/Plate", su.RigidBodyPropertiesCfg(kinematic_enabled=True, sleep_threshold=0.0)
        )
        schemas.activate_contact_sensors("/World/Plate", threshold=0.0)
        x, y = 0.48, -0.10
        self.plate = RigidObject(
            RigidObjectCfg(
                prim_path="/World/Plate",
                spawn=None,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, plate["body_z"])),
            )
        )

    def _spawn_camera_and_lights(self) -> None:
        from isaaclab.sensors.camera import Camera, CameraCfg
        from pxr import Gf, UsdLux

        from embodied_jepa import isaac_scene as scene

        su = self._sim_utils
        cm = self.camera_manifest
        cam = cm["camera"]
        stage = su.get_current_stage()
        parent = next(
            (
                p.GetPath()
                for p in stage.Traverse()
                if p.GetName() == cam["parent_body"] and str(p.GetPath()).startswith("/World/Robot")
            ),
            None,
        )
        if parent is None:
            raise ContractError(f"converted USD has no {cam['parent_body']} for the camera")
        focal, h_ap, v_ap = scene.pinhole_from_fovy(cam["fovy_deg"], self.width, self.height)
        self.camera_prim_path = f"{parent}/{cam['name']}"
        self.camera = Camera(
            CameraCfg(
                prim_path=self.camera_prim_path,
                update_period=0.0,
                height=self.height,
                width=self.width,
                data_types=list(cm["isaac_render"]["data_types"]),
                colorize_instance_id_segmentation=False,
                spawn=su.PinholeCameraCfg(
                    focal_length=focal,
                    horizontal_aperture=h_ap,
                    vertical_aperture=v_ap,
                    clipping_range=(cam["znear_m"], cam["zfar_m"]),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(cam["pos_m"]),
                    rot=scene.camera_quat_xyzw(cm),
                    convention="opengl",
                ),
            )
        )
        for light in cm["isaac_render"]["lights"]:
            path = light["path"].replace("{camera}", self.camera_prim_path)
            if light["type"] == "distant":
                prim = UsdLux.DistantLight.Define(stage, path)
                prim.CreateAngleAttr(float(light["angle_deg"]))
            elif light["type"] == "dome":
                prim = UsdLux.DomeLight.Define(stage, path)
            else:
                raise ContractError(f"unsupported light type {light['type']!r}")
            prim.CreateIntensityAttr(float(light["intensity"]))
            prim.CreateColorAttr(Gf.Vec3f(*[float(v) for v in light["color"]]))
            UsdLux.ShadowAPI.Apply(prim.GetPrim()).CreateShadowEnableAttr(bool(light["shadow"]))
            if "direction" in light:
                _aim_distant_light(prim.GetPrim(), light["direction"])

    def _apply_render_settings(self) -> dict:
        """Set the manifest's renderer settings and read every one back."""
        wanted = self.camera_manifest["isaac_render"]["carb_settings"]
        got = {}
        for key, value in wanted.items():
            self.sim.set_setting(key, value)
            got[key] = self.sim.get_setting(key)
            if got[key] != value:
                raise ContractError(f"render setting {key} read back {got[key]!r}, not {value!r}")
        return got

    def _activate_link_contact_reports(self) -> None:
        """PhysX contact reports on every robot link.

        Isaac Lab's ``activate_contact_sensors`` stops at the first rigid body it meets, and
        the converted USD nests every link under its parent link, so it would only reach the
        pelvis. Threshold 0: every contact is reported. Report-only; no physical effect."""
        from pxr import PhysxSchema, UsdPhysics

        stage = self._sim_utils.get_current_stage()
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith("/World/Robot/") and prim.HasAPI(
                UsdPhysics.RigidBodyAPI
            ):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)

    def _read_object_properties(self) -> dict:
        """What PhysX actually holds for the apple and plate, read back after ``sim.reset``."""
        view = self.apple.root_view
        out = {
            "apple_mass": self._np(view.get_masses()).ravel().tolist(),
            "apple_inertia": self._np(view.get_inertias()).ravel().tolist(),
        }
        try:
            mats = self._np(view.get_material_properties())
            out["apple_material_static_dynamic_restitution"] = mats.reshape(-1).tolist()
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            out["apple_material_error"] = f"{type(exc).__name__}: {exc}"
        stage = self._sim_utils.get_current_stage()
        attrs = {}
        for path in ("/World/Apple", "/World/Apple/geometry/mesh", "/World/Plate"):
            prim = stage.GetPrimAtPath(path)
            attrs[path] = {
                a.GetName(): _jsonable(a.Get())
                for a in prim.GetAttributes()
                if a.GetName().startswith(("physx", "physics:"))
            }
        out["usd_physics_attributes"] = attrs
        return out

    # ------------------------------------------------------------------ helpers
    def _set_joint_friction(self, joints) -> None:
        """Replace the converter's joint friction with MuJoCo's frictionloss (or nothing).

        The importer copies MuJoCo ``frictionloss`` (a Coulomb torque in N m) into PhysX's
        deprecated, load-proportional joint friction *coefficient*, a different model that
        made the arms creep in development runs. That coefficient is zeroed. With
        ``"frictionloss"`` PhysX's static and dynamic friction *efforts* (N m) are set to the
        MuJoCo value and viscous friction to 0; with ``"none"`` all three are 0. Read back.
        """
        wp, view = self._wp, self.robot.root_view
        n = len(self._idx)
        coeff = view.get_dof_friction_coefficients()
        props = view.get_dof_friction_properties()
        fl = np.zeros(n)
        if self.joint_friction == "frictionloss":
            fl[self._idx] = [j["frictionloss"] for j in joints]
        new_props = np.stack([fl, fl, np.zeros(n)], axis=1)[None].astype(np.float32)
        self._friction_props = new_props[0]
        indices = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=coeff.device)
        view.set_dof_friction_coefficients(
            wp.array(np.zeros((1, n), dtype=np.float32), dtype=wp.float32, device=coeff.device),
            indices,
        )
        view.set_dof_friction_properties(
            wp.array(new_props, dtype=wp.float32, device=props.device), indices
        )
        self._check_joint_properties()

    def _check_joint_properties(self) -> None:
        """Read PhysX damping and friction back; raise if they differ from what was set."""
        view = self.robot.root_view
        got_damping = self._np(view.get_dof_dampings())[0][self._idx]
        if np.abs(got_damping - self._damping).max() > 1e-6:
            raise ContractError("PhysX drive damping read-back differs from the manifest")
        got_coeff = self._np(view.get_dof_friction_coefficients())[0]
        got_props = self._np(view.get_dof_friction_properties())[0]
        if np.abs(got_coeff).max() > 0 or np.abs(got_props - self._friction_props).max() > 1e-6:
            raise ContractError("PhysX joint friction read-back differs from what was set")
        self.physx_joint_friction = {
            "legacy_coefficient": got_coeff[self._idx].tolist(),
            "static_dynamic_viscous": got_props[self._idx].tolist(),
        }

    def _np(self, x) -> np.ndarray:
        t = x if isinstance(x, self._torch.Tensor) else self._wp.to_torch(x)
        return t.detach().cpu().numpy()

    def _require_open(self):
        if self.closed:
            raise RuntimeError("simulation is closed")

    def _state(self) -> tuple[np.ndarray, np.ndarray]:
        q = self._np(self.robot.data.joint_pos)[0][self._idx].astype(float)
        qd = self._np(self.robot.data.joint_vel)[0][self._idx].astype(float)
        return q, qd

    def _bias(self) -> np.ndarray:
        view = self.robot.root_view
        g = self._np(view.get_gravity_compensation_forces())[0]
        c = self._np(view.get_coriolis_and_centrifugal_compensation_forces())[0]
        return (g + c)[self._idx].astype(float)

    def _apply_effort(self, effort: np.ndarray) -> None:
        isaac = np.zeros(len(self._idx))
        isaac[self._idx] = effort
        self.robot.set_joint_effort_target_index(
            target=self._torch.as_tensor(isaac, dtype=self._torch.float32, device=self.sim.device)[
                None
            ]
        )

    def _make_contact_views(self) -> None:
        """PhysX tensor contact views: robot links and the apple against the scene bodies.

        Contacts are read as per-pair force matrices (N) after every physics step. Robot
        self-contacts are not reported (Isaac self-collision is off)."""
        from isaaclab_physx.physics import PhysxManager
        from pxr import UsdPhysics

        stage = self._sim_utils.get_current_stage()
        links = [
            str(p.GetPath())
            for p in stage.Traverse()
            if str(p.GetPath()).startswith("/World/Robot/") and p.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        self.contact_link_paths = links
        scene = ["/World/table", "/World/floor"]
        if self.objects:
            scene += ["/World/Apple", "/World/Plate"]
        view = PhysxManager.get_physics_sim_view()
        self._contact_views.append(
            (
                view.create_rigid_contact_view(
                    links, filter_patterns=[list(scene)] * len(links), max_contact_data_count=0
                ),
                links,
                scene,
            )
        )
        if self.objects:
            apple_filters = ["/World/table", "/World/floor", "/World/Plate", *links]
            self._contact_views.append(
                (
                    view.create_rigid_contact_view(
                        "/World/Apple", filter_patterns=apple_filters, max_contact_data_count=0
                    ),
                    ["/World/Apple"],
                    apple_filters,
                )
            )

    def _collect_contacts(self) -> None:
        """Body pairs in contact at the last physics step, with the pair force (N)."""
        found: dict[tuple[str, str], float] = {}
        for view, sensors, filters in self._contact_views:
            matrix = self._np(view.get_contact_force_matrix(dt=self.physics_dt))
            matrix = matrix.reshape(len(sensors), len(filters), 3)
            force = np.linalg.norm(matrix, axis=-1)
            for i, j in zip(*np.nonzero(force > 0.0), strict=True):
                pair = tuple(sorted((_body_label(sensors[i]), _body_label(filters[j]))))
                found[pair] = max(found.get(pair, 0.0), float(force[i, j]))
        self._last_contacts = [
            {"bodies": list(pair), "force_n": f} for pair, f in sorted(found.items())
        ]
        self._interval_pairs.update(found)

    def _step_physics(self) -> None:
        self.robot.write_data_to_sim()
        self.sim.step(render=False)
        self.robot.update(self.physics_dt)
        for obj in (self.apple, self.plate):
            if obj is not None:
                obj.update(self.physics_dt)
        self.time = round(self.time + self.physics_dt, 9)
        if self._contact_views:
            self._collect_contacts()

    # ------------------------------------------------------------------ transport API
    def reset(self, seed=0, *, object_xy=None, plate_xy=None, object_on_container=False):
        """Reset to MuJoCoSimulation's initial pose and, with objects, its apple/plate layout.

        Placement follows ``MuJoCoSimulation.reset`` (``isaac_scene.reset_layout``): the same
        seeded default apple position, plate default, tabletop bounds and overlap checks. The
        episode clock restarts. The return value carries no object state (see ``task_truth``).
        """
        self._require_open()
        if not self.objects and (object_xy is not None or plate_xy is not None):
            raise ContractError("this Isaac transport was built without the apple and plate")
        from embodied_jepa import isaac_scene as scene

        torch = self._torch
        if self.objects:
            apple_pos, plate_pos = scene.reset_layout(
                seed,
                object_xy=object_xy,
                plate_xy=plate_xy,
                object_on_container=object_on_container,
            )
            self._write_object(self.apple, apple_pos)
            self._write_object(self.plate, plate_pos)
        q = reset_pose(self.manifest)
        isaac_q = np.zeros(len(self._idx))
        isaac_q[self._idx] = q
        qt = torch.as_tensor(isaac_q, dtype=torch.float32, device=self.sim.device)[None]
        self.robot.write_joint_position_to_sim_index(position=qt)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(qt))
        self._apply_effort(np.zeros(len(self._idx)))
        self.robot.write_data_to_sim()
        self.robot.reset()
        self.robot.update(0.0)
        for obj in (self.apple, self.plate):
            if obj is not None:
                obj.reset()
                obj.update(0.0)
        self._check_joint_properties()  # robot.reset() must not undo damping/friction
        self.targets[:] = q
        self.time = 0.0
        self.stopped_reason = ""
        self._last_contacts = []
        self._interval_pairs = set()
        if self.camera is not None:
            for _ in range(self.warmup_renders):  # discard render-pipeline warm-up frames
                self.render()
        return {"timestamp": self.time, "clock": self.clock_domain}

    def _write_object(self, obj, pos, lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)):
        torch = self._torch
        pose = torch.tensor(
            [[*[float(v) for v in pos], 0.0, 0.0, 0.0, 1.0]], device=self.sim.device
        )
        obj.write_root_pose_to_sim_index(root_pose=pose)
        vel = torch.tensor([[*lin_vel, *ang_vel]], dtype=torch.float32, device=self.sim.device)
        obj.write_root_velocity_to_sim_index(root_velocity=vel)

    def set_object_state(self, pos, lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)) -> None:
        """Scripted-check harness only: teleport the apple (drop tests). Not a controller API."""
        self._require_open()
        if not self.objects:
            raise ContractError("no apple in this scene")
        self._write_object(self.apple, pos, lin_vel, ang_vel)
        self.apple.update(0.0)

    def _pump_render(self) -> None:
        """Render the current physics state ``renders_per_frame`` times.

        Isaac Lab 3 renders at most once per physics step (``ensure_isaac_rtx_render_update``
        dedups on the step count), so without this a render after ``reset()`` or a second
        render of the same step returns the previous frame unchanged. This syncs physics to
        Fabric, pumps Kit's update loop with physics stepping paused, and marks the step as
        rendered so the camera does not pump again."""
        import omni.kit.app
        from isaaclab_physx.renderers import isaac_rtx_renderer_utils as rtx

        self.sim.physics_manager.forward()
        self.sim.set_setting("/app/player/playSimulations", False)
        try:
            for _ in range(self.renders_per_frame):
                omni.kit.app.get_app().update()
        finally:
            self.sim.set_setting("/app/player/playSimulations", True)
        rtx._last_render_update_key = (id(self.sim), self.sim._physics_step_count)

    def render(self) -> np.ndarray:
        self._require_open()
        if self.camera is None:
            raise RuntimeError("RGB rendering was explicitly disabled")
        self._pump_render()
        self.camera.update(0.0, force_recompute=True)
        rgb = self.camera.data.output["rgb"][0, ..., :3]
        return self._np(rgb).astype(np.uint8)

    def segmentation(self) -> tuple[np.ndarray, dict]:
        """Instance-id segmentation of the last rendered frame and its id -> prim path map.

        Diagnostic for the camera-geometry parity check only (no observation uses it)."""
        self._require_open()
        seg = self._np(self.camera.data.output["instance_id_segmentation_fast"])[0, ..., 0]
        info = self.camera.data.info[0]["instance_id_segmentation_fast"]
        return seg.astype(np.int64), {int(k): str(v) for k, v in info["idToLabels"].items()}

    def read(self) -> dict:
        self._require_open()
        q, qd = self._state()
        rgb = self.render() if self.render_enabled else None
        return {
            "rgb": rgb,
            "qpos": q,
            "qvel": qd,
            "joint_names": self.joint_names,
            "timestamp": self.time,
            "rgb_timestamp": self.time if rgb is not None else None,
            "clock": self.clock_domain,
            "sensor_valid": bool(np.isfinite(q).all() and np.isfinite(qd).all()),
        }

    def send_joint_targets(self, targets, *, joint_names, deadline):
        self._require_open()
        targets = np.asarray(targets, dtype=float)
        if (
            tuple(joint_names) != self.joint_names
            or targets.shape != self.targets.shape
            or not np.isfinite(targets).all()
        ):
            raise ContractError("joint targets must match named robot actuator order and be finite")
        if not np.isfinite(deadline) or deadline < self.time:
            self.stop("command deadline expired")
            return {"status": "rejected", "timestamp": self.time, "reason": self.stopped_reason}
        if np.any(targets < self.lower) or np.any(targets > self.upper):
            raise ContractError("joint targets exceed MJCF limits")
        old = self.targets.copy()
        self.targets[:] = targets
        self._interval_pairs = set()
        for k in range(self.substeps):
            interpolated = old + (targets - old) * ((k + 1) / self.substeps)
            q, qd = self._state()
            torque = self._kp * (interpolated - q) - self._kd * qd + self._bias()
            self._apply_effort(np.clip(torque, self.ctrl_min, self.ctrl_max))
            self._step_physics()
            if not np.isfinite(self._state()[0]).all():
                self.stop("nonfinite state")
                raise RuntimeError(self.stopped_reason)
        self.stopped_reason = ""
        return {"status": "applied", "timestamp": self.time, "targets": self.targets.copy()}

    def stop(self, reason):
        """Hold the current (limit-clipped) pose as target and zero the effort command."""
        self._require_open()
        q, _ = self._state()
        self.targets[:] = np.clip(q, self.lower, self.upper)
        self._apply_effort(np.zeros(len(self._idx)))
        self.stopped_reason = str(reason)

    # ------------------------------------------------------------------ evaluator-only truth
    def contacts(self) -> dict:
        """Evaluator/diagnostic only: PhysX contacts at the last physics step, and the body
        pairs seen at any substep of the last control interval. Never an observation."""
        self._require_open()
        return {
            "api_error": self.contact_api_error,
            "last_step": [dict(c) for c in self._last_contacts],
            "interval_pairs": [list(p) for p in sorted(self._interval_pairs)],
        }

    def task_truth(self) -> dict:
        """Evaluator-only simulator truth, as ``MuJoCoSimulation.task_truth`` defines it.

        Never part of ``read()``; it must not reach a model input, controller or planning cost.
        """
        self._require_open()
        if not self.objects:
            raise ContractError("no apple/plate in this scene")
        from embodied_jepa import isaac_scene as scene

        if self.contact_api_error is not None:
            raise RuntimeError(f"contact reports unavailable: {self.contact_api_error}")
        apple = self._np(self.apple.data.root_link_pose_w).reshape(-1)[:3]
        vel = self._np(self.apple.data.root_com_vel_w).reshape(-1)[:3]
        plate = self._np(self.plate.data.root_link_pose_w).reshape(-1)[:3]
        root = self._np(self.robot.data.root_link_pose_w).reshape(-1)
        hand = any(
            "apple" in c["bodies"] and any(scene.is_hand_body(b) for b in c["bodies"])
            for c in self._last_contacts
        )
        return scene.task_truth_from_state(
            object_position=apple,
            object_velocity=vel,
            plate_position=plate,
            hand_contact=hand,
            base_position=root[:3],
            base_rotation=_quat_xyzw_to_matrix(root[3:7]),
            timestamp=self.time,
        )

    def close(self):
        if self.closed:
            return
        self.camera = None
        self.robot = self.apple = self.plate = None
        self.sim.clear_instance()
        self.sim = None
        self.closed = True


def _body_label(path: str) -> str:
    """An actor prim path as a MuJoCo-comparable label: ``floor``, ``table``, ``apple``,
    ``plate`` or the robot link name (the last path component under ``/World/Robot``)."""
    for prefix, label in (
        ("/World/floor", "floor"),
        ("/World/table", "table"),
        ("/World/Apple", "apple"),
        ("/World/Plate", "plate"),
    ):
        if path == prefix or path.startswith(prefix + "/"):
            return label
    return path.rsplit("/", 1)[-1]


def _quat_z_to(d) -> tuple[float, float, float, float]:
    """(x, y, z, w) rotating +Z onto the unit vector ``d``."""
    z = np.array([0.0, 0.0, 1.0])
    d = np.asarray(d, dtype=float)
    axis = np.cross(z, d)
    s, c = np.linalg.norm(axis), float(np.dot(z, d))
    if s < 1e-12:
        return (0.0, 0.0, 0.0, 1.0) if c > 0 else (1.0, 0.0, 0.0, 0.0)
    angle = np.arctan2(s, c)
    axis = axis / s * np.sin(angle / 2)
    return (float(axis[0]), float(axis[1]), float(axis[2]), float(np.cos(angle / 2)))


def _quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _aim_distant_light(prim, direction) -> None:
    """Orient a UsdLux distant light (which shines along its local -Z) along ``direction``."""
    from pxr import Gf, UsdGeom

    d = np.asarray(direction, dtype=float)
    x, y, z, w = _quat_z_to(-d / np.linalg.norm(d))
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))


def _jsonable(v):
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    try:
        return [float(x) for x in v]
    except TypeError:
        return str(v)
