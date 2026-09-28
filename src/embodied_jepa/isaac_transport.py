"""Minimal Isaac Sim transport for the fixed-pelvis G1 + dual Dex3 (development, TASK-025).

This module is importable without Isaac: every Isaac Sim / Isaac Lab / torch / warp import
is inside ``IsaacTransport``. The transport itself only runs inside the ``isaaclab_arena``
container (Isaac Sim 6.0.0-rc.22, Isaac Lab 3.0.0) after the caller has started the Kit app
(``isaaclab.app.AppLauncher``); no Isaac dependency is declared in ``pyproject.toml`` or
``uv.lock``. See ``docs/ISAAC_MJCF_TRANSPORT.md``.

Scope, as in ``docs/ISAAC_PORT.md``: ``read`` / ``send_joint_targets`` / ``stop`` / ``reset``
/ ``close`` over the robot only. The table is present; the apple and plate are not, so
``reset`` rejects object/plate coordinates and there is no ``task_truth``. The robot is the
USD converted from the project's pinned MJCF (``scripts/isaac/convert_mjcf_to_usd.py``).

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
    """Fixed-pelvis G1 + Dex3 in Isaac Sim with the MuJoCo transport's control semantics."""

    clock_domain = "isaac_episode_sim_time"

    def __init__(
        self,
        *,
        usd_path: str,
        manifest: dict,
        width: int = 96,
        height: int = 96,
        render: bool = True,
        device: str = "cuda:0",
        warmup_renders: int = 3,
        joint_friction: str = "frictionloss",
    ):
        self.manifest = validate_joint_manifest(manifest)
        if joint_friction not in ("frictionloss", "none"):
            raise ContractError("joint_friction must be 'frictionloss' or 'none'")
        self.joint_friction = joint_friction
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in (width, height)):
            raise ContractError("RGB dimensions must be positive integers")
        import isaaclab.sim as sim_utils
        import torch
        import warp as wp
        from isaaclab.actuators import IdealPDActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg

        self._torch, self._wp = torch, wp
        joints = manifest["joints"]
        self.joint_names = tuple(j["name"] for j in joints)
        self.lower = np.array([j["lower"] for j in joints])
        self.upper = np.array([j["upper"] for j in joints])
        self.ctrl_min = np.array([j["ctrl_min"] for j in joints])
        self.ctrl_max = np.array([j["ctrl_max"] for j in joints])
        self.physics_dt = float(manifest["physics_dt_s"])
        self.control_dt = float(manifest["control_dt_s"])
        self.substeps = round(self.control_dt / self.physics_dt)
        self.width, self.height = width, height
        self.render_enabled = render
        self.warmup_renders = int(warmup_renders)
        self.closed = False
        self.stopped_reason = ""

        cfg = sim_utils.SimulationCfg(
            dt=self.physics_dt,
            device=device,
            render=sim_utils.RenderCfg(antialiasing_mode="Off", enable_dlssg=False),
        )
        self.sim = sim_utils.SimulationContext(cfg)
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/ground", ground)
        light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
        light.func("/World/light", light)
        table = sim_utils.CuboidCfg(  # MuJoCoSimulation: box pos .45 0 .70, half-size .32 .45 .04
            size=(0.64, 0.90, 0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.4, 0.25)),
        )
        table.func("/World/table", table, translation=(0.45, 0.0, 0.70))
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
        self.camera = None
        if render:
            from isaaclab.sensors.camera import Camera, CameraCfg

            torso = next(
                (
                    p.GetPath()
                    for p in sim_utils.get_current_stage().Traverse()
                    if p.GetName() == "torso_link" and str(p.GetPath()).startswith("/World/Robot")
                ),
                None,
            )
            if torso is None:
                raise ContractError("converted USD has no torso_link for the onboard camera")
            pos, rot = onboard_camera_offset()
            aperture = 20.955
            focal = aperture / (2.0 * np.tan(np.deg2rad(75.0) / 2.0))  # MuJoCo fovy 75, square
            self.camera = Camera(
                CameraCfg(
                    prim_path=f"{torso}/onboard_rgb",
                    update_period=0.0,
                    height=height,
                    width=width,
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=float(focal),
                        horizontal_aperture=aperture,
                        vertical_aperture=aperture * height / width,
                        clipping_range=(0.01, 10.0),
                    ),
                    offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="opengl"),
                )
            )
        self.sim.reset()
        self.isaac_joint_names = tuple(self.robot.joint_names)
        self._idx = name_map(self.joint_names, list(self.isaac_joint_names))
        self._idx_t = torch.as_tensor(self._idx, device=self.sim.device)
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
        self.reset()

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

    # ------------------------------------------------------------------ transport API
    def reset(self, seed=0, *, object_xy=None, plate_xy=None):
        """Robot-only reset to MuJoCoSimulation's initial pose; the episode clock restarts."""
        self._require_open()
        if object_xy is not None or plate_xy is not None:
            raise ContractError("the Isaac transport has no apple/plate yet")
        del seed  # the robot reset is deterministic; no scene randomisation exists yet
        torch = self._torch
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
        self._check_joint_properties()  # robot.reset() must not undo damping/friction
        self.targets[:] = q
        self.time = 0.0
        self.stopped_reason = ""
        if self.camera is not None:
            for _ in range(self.warmup_renders):  # discard render-pipeline warm-up frames
                self.sim.render()
                self.camera.update(0.0, force_recompute=True)
        return {"timestamp": self.time, "clock": self.clock_domain}

    def render(self) -> np.ndarray:
        self._require_open()
        if self.camera is None:
            raise RuntimeError("RGB rendering was explicitly disabled")
        self.sim.render()
        self.camera.update(0.0, force_recompute=True)
        rgb = self.camera.data.output["rgb"][0, ..., :3]
        return self._np(rgb).astype(np.uint8)

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
        for k in range(self.substeps):
            interpolated = old + (targets - old) * ((k + 1) / self.substeps)
            q, qd = self._state()
            torque = self._kp * (interpolated - q) - self._kd * qd + self._bias()
            self._apply_effort(np.clip(torque, self.ctrl_min, self.ctrl_max))
            self.robot.write_data_to_sim()
            self.sim.step(render=False)
            self.robot.update(self.physics_dt)
            self.time = round(self.time + self.physics_dt, 9)
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

    def close(self):
        if self.closed:
            return
        self.camera = None
        self.robot = None
        self.sim.clear_instance()
        self.sim = None
        self.closed = True
