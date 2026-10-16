"""SO-101 tabletop MuJoCo transport (SO-101 S1, TASK-097); importable without MuJoCo.

The arm is MuJoCo Menagerie's ``robotstudio_so101/so101.xml`` at the revision pinned in
``assets/so101_manifest.json`` (fetched by ``scripts/fetch_so101.py``), loaded unchanged and
extended here, in memory, with a table, one free 3 cm cube, a flat target marker, lights and a
fixed overview camera. The wrist camera is Menagerie's own ``wrist_cam`` on its ``camera_mount``
body. The robot base sits on the table top at the world origin (table top z = 0); +x points
away from the base, the arm's straight-out direction at zero joint angles.

Commands are joint-position targets for Menagerie's six position actuators (its ``sts3215``
servo model), interpolated linearly across the physics substeps of one control step. Simulator
truth (cube and target poses, contacts) is ``task_truth()`` only: it never enters ``read()``.

This module is opt-in: nothing in ``import embodied_jepa`` loads it, ``mujoco`` is imported
lazily in the constructor, and the G1 transport ``simulation.py`` is untouched.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[2]
ASSET_MANIFEST = ROOT / "assets/so101_manifest.json"
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
CAMERAS = ("wrist_rgb", "overview_rgb")
# MJCF camera names behind the canonical image keys.
CAMERA_SOURCES = {"wrist_rgb": "wrist_cam", "overview_rgb": "overview"}
CUBE_HALF_SIZE_M = 0.015
CUBE_MASS_KG = 0.02
TARGET_HALF_SIZE_M = 0.04
TABLE_TOP_Z = 0.0
# Development default reset region (world xy, metres). Not a task distribution: S2's task
# modules (TASK-098) declare their own resets and pass them in explicitly.
DEV_CUBE_REGION = ((0.17, 0.27), (-0.08, 0.08))
DEV_TARGET_REGION = ((0.15, 0.27), (-0.15, 0.15))
TABLETOP = ((0.08, 0.40), (-0.25, 0.25))


def default_asset_root() -> Path:
    manifest = json.loads(ASSET_MANIFEST.read_text())
    return ROOT / manifest["local_path"]


def assets_fetched() -> bool:
    """Whether the pinned model's files exist (hashes are checked by the constructor)."""
    manifest = json.loads(ASSET_MANIFEST.read_text())
    root = ROOT / manifest["local_path"]
    return all((root / relative).is_file() for relative in manifest["sha256"])


def _verify_assets(root: Path) -> dict:
    manifest = json.loads(ASSET_MANIFEST.read_text())
    for relative, expected in manifest["sha256"].items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ContractError(f"Missing or modified asset {path}; run scripts/fetch_so101.py")
    return manifest


class SO101Simulation:
    """Fixed-base SO-101 on a table with a free cube and a flat target marker."""

    def __init__(self, *, asset_root=None, width=224, height=224, control_dt=0.05, render=True):
        self.mj = importlib.import_module("mujoco")
        self.asset_root = Path(asset_root) if asset_root is not None else default_asset_root()
        self.asset_manifest = _verify_assets(self.asset_root)
        if not np.isfinite(control_dt) or control_dt <= 0:
            raise ContractError("control_dt must be finite and positive")
        self.control_dt = float(control_dt)
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in (width, height)):
            raise ContractError("RGB dimensions must be positive integers")
        self.width, self.height = width, height
        self.render_enabled = render
        self.renderer = None
        self.closed = False
        self.model = self.mj.MjModel.from_xml_string(self._scene())
        self.data = self.mj.MjData(self.model)
        self.substeps = round(control_dt / self.model.opt.timestep)
        if self.substeps < 1 or not np.isclose(self.substeps * self.model.opt.timestep, control_dt):
            raise ContractError("control_dt must be a positive multiple of the physics timestep")
        self.joint_ids = self.model.actuator_trnid[:, 0].copy()
        self.joint_names = tuple(self.model.joint(int(i)).name for i in self.joint_ids)
        if self.joint_names != JOINT_NAMES:
            raise ContractError(f"unexpected SO-101 actuator order {self.joint_names}")
        self.qadr = self.model.jnt_qposadr[self.joint_ids]
        self.vadr = self.model.jnt_dofadr[self.joint_ids]
        # A target must satisfy both the joint range and the actuator's control range.
        self.limits = np.stack(
            (
                np.maximum(
                    self.model.jnt_range[self.joint_ids, 0], self.model.actuator_ctrlrange[:, 0]
                ),
                np.minimum(
                    self.model.jnt_range[self.joint_ids, 1], self.model.actuator_ctrlrange[:, 1]
                ),
            ),
            axis=1,
        )
        self.targets = np.zeros(self.model.nu)
        self.stopped_reason = ""
        self.reset()

    def _scene(self) -> str:
        directory = self.asset_root / self.asset_manifest["sparse_path"]
        root = ET.parse(directory / "so101.xml").getroot()
        root.find("compiler").set("meshdir", str((directory / "assets").resolve()))
        root.set("model", "so101_tabletop")
        visual = root.find("visual")
        ET.SubElement(
            visual, "headlight", diffuse=".5 .5 .5", ambient=".35 .35 .35", specular="0 0 0"
        )
        # The offscreen buffer is at least 640 px and always fits the requested RGB size.
        ET.SubElement(
            visual,
            "global",
            offwidth=str(max(640, self.width)),
            offheight=str(max(640, self.height)),
        )
        ET.SubElement(
            root.find("asset"),
            "texture",
            name="sky",
            type="skybox",
            builtin="gradient",
            rgb1=".75 .78 .82",
            rgb2=".45 .48 .52",
            width="64",
            height="384",
        )
        world = root.find("worldbody")
        ET.SubElement(world, "light", pos=".2 -.3 1.2", dir="0 .2 -1", directional="true")
        ET.SubElement(
            world,
            "geom",
            name="floor",
            type="plane",
            pos="0 0 -.75",
            size="2 2 .05",
            rgba=".3 .32 .35 1",
        )
        ET.SubElement(
            world,
            "geom",
            name="table",
            type="box",
            pos=".15 0 -.02",
            size=".35 .40 .02",
            rgba=".62 .52 .4 1",
            friction="1 .005 .0001",
        )
        target = ET.SubElement(world, "body", name="target", pos=".22 .12 0")
        h = TARGET_HALF_SIZE_M
        ET.SubElement(
            target,
            "geom",
            name="target_marker",
            type="box",
            pos="0 0 .0005",
            size=f"{h} {h} .0005",
            rgba=".15 .35 .85 1",
            contype="0",
            conaffinity="0",
        )
        cube = ET.SubElement(world, "body", name="cube", pos=f".22 0 {CUBE_HALF_SIZE_M}")
        ET.SubElement(cube, "freejoint", name="cube_free")
        c = CUBE_HALF_SIZE_M
        ET.SubElement(
            cube,
            "geom",
            name="cube_geom",
            type="box",
            size=f"{c} {c} {c}",
            mass=str(CUBE_MASS_KG),
            friction="1 .005 .0001",
            condim="4",
            rgba=".85 .12 .1 1",
        )
        # The overview camera looks at a fixed point above the workspace from the front-right.
        ET.SubElement(world, "body", name="workspace_center", pos=".14 .02 .05")
        ET.SubElement(
            world,
            "camera",
            name="overview",
            mode="targetbody",
            target="workspace_center",
            pos=".52 -.14 .34",
            fovy="55",
        )
        return ET.tostring(root, encoding="unicode")

    def _require_open(self):
        if self.closed:
            raise RuntimeError("simulation is closed")

    @staticmethod
    def _check_xy(name, xy, region):
        (x0, x1), (y0, y1) = region
        if (
            xy.shape != (2,)
            or not np.isfinite(xy).all()
            or not (x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1)
        ):
            raise ContractError(f"{name} reset xy must lie in {region}")

    def reset(self, seed=0, *, cube_xy=None, cube_yaw=None, target_xy=None, joint_positions=None):
        """Reset the scene; xy are world metres on the table top, yaw is radians about +z.

        Unspecified poses are drawn from the development default region with ``seed``; a
        task passes its own declared reset explicitly.
        """
        self._require_open()
        rng = np.random.default_rng(seed)
        if cube_xy is None:
            cube_xy = [rng.uniform(*DEV_CUBE_REGION[0]), rng.uniform(*DEV_CUBE_REGION[1])]
        if cube_yaw is None:
            cube_yaw = rng.uniform(-np.pi / 4, np.pi / 4)
        cube_xy = np.asarray(cube_xy, dtype=float)
        self._check_xy("cube", cube_xy, TABLETOP)
        if not np.isfinite(cube_yaw):
            raise ContractError("cube yaw must be finite")
        if target_xy is None:
            for _ in range(100):
                target_xy = np.array(
                    [rng.uniform(*DEV_TARGET_REGION[0]), rng.uniform(*DEV_TARGET_REGION[1])]
                )
                if np.linalg.norm(target_xy - cube_xy) >= 0.09:
                    break
            else:
                raise ContractError("could not draw a target away from the cube")
        target_xy = np.asarray(target_xy, dtype=float)
        self._check_xy("target", target_xy, TABLETOP)
        self.mj.mj_resetData(self.model, self.data)
        if joint_positions is not None:
            q = np.asarray(joint_positions, dtype=float)
            if (
                q.shape != (len(JOINT_NAMES),)
                or not np.isfinite(q).all()
                or np.any(q < self.limits[:, 0])
                or np.any(q > self.limits[:, 1])
            ):
                raise ContractError("reset joint positions must be finite and within limits")
            self.data.qpos[self.qadr] = q
        cube_q = int(self.model.joint("cube_free").qposadr[0])
        self.data.qpos[cube_q : cube_q + 3] = [*cube_xy, TABLE_TOP_Z + CUBE_HALF_SIZE_M]
        self.data.qpos[cube_q + 3 : cube_q + 7] = [np.cos(cube_yaw / 2), 0, 0, np.sin(cube_yaw / 2)]
        self.model.body("target").pos[:] = [*target_xy, TABLE_TOP_Z]
        self.mj.mj_forward(self.model, self.data)
        self.targets[:] = self.data.qpos[self.qadr]
        self.data.ctrl[:] = self.targets
        self.stopped_reason = ""
        return self.task_truth()

    def render(self, camera="wrist_rgb"):
        self._require_open()
        if not self.render_enabled:
            raise RuntimeError("RGB rendering was explicitly disabled")
        if camera not in CAMERA_SOURCES:
            raise ContractError(f"unknown camera {camera!r}; expected one of {CAMERAS}")
        if self.renderer is None:
            self.renderer = self.mj.Renderer(self.model, height=self.height, width=self.width)
        self.renderer.update_scene(self.data, camera=CAMERA_SOURCES[camera])
        return self.renderer.render().copy()

    def read(self):
        """Both cameras and the joint state, all from the same simulator instant."""
        self._require_open()
        return {
            "rgb": {camera: self.render(camera) for camera in CAMERAS},
            "qpos": self.data.qpos[self.qadr].copy(),
            "qvel": self.data.qvel[self.vadr].copy(),
            "joint_names": self.joint_names,
            "timestamp": float(self.data.time),
        }

    def send_joint_targets(self, targets, *, joint_names, deadline):
        self._require_open()
        targets = np.asarray(targets)
        if (
            tuple(joint_names) != self.joint_names
            or targets.shape != self.targets.shape
            or not np.isfinite(targets).all()
        ):
            raise ContractError("joint targets must match named robot actuator order and be finite")
        if not np.isfinite(deadline) or deadline < self.data.time:
            self.stop("command deadline expired")
            return {
                "status": "rejected",
                "timestamp": float(self.data.time),
                "reason": self.stopped_reason,
            }
        if np.any(targets < self.limits[:, 0]) or np.any(targets > self.limits[:, 1]):
            raise ContractError("joint targets exceed MJCF limits")
        old = self.targets.copy()
        self.targets[:] = targets
        for k in range(self.substeps):
            self.data.ctrl[:] = old + (targets - old) * ((k + 1) / self.substeps)
            self.mj.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or any(w.number for w in self.data.warning):
                self.stop("nonfinite state or MuJoCo warning")
                raise RuntimeError(self.stopped_reason)
        self.stopped_reason = ""
        return {
            "status": "applied",
            "timestamp": float(self.data.time),
            "targets": self.targets.copy(),
        }

    def stop(self, reason):
        """Hold the measured pose: targets become the clipped measured joint positions."""
        self.targets[:] = np.clip(self.data.qpos[self.qadr], self.limits[:, 0], self.limits[:, 1])
        self.data.ctrl[:] = self.targets
        self.stopped_reason = str(reason)

    def gripper_contacts(self):
        """Names of gripper collision geoms touching the cube (simulator truth)."""
        cube = self.model.geom("cube_geom").id
        jaws = set()
        for contact in self.data.contact[: self.data.ncon]:
            if cube in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == cube else contact.geom1
                body = self.model.body(int(self.model.geom_bodyid[other])).name
                if body in ("gripper", "moving_jaw_so101_v1", "camera_mount"):
                    jaws.add(body)
        return jaws

    def task_truth(self):
        """Evaluator/collector-only simulator truth; never part of the canonical observation."""
        self._require_open()
        contacts = self.gripper_contacts()
        return {
            "position_frame": "world",
            "table_top_z": TABLE_TOP_Z,
            "cube_half_size": CUBE_HALF_SIZE_M,
            "cube_position": self.data.body("cube").xpos.copy(),
            "cube_quaternion": self.data.body("cube").xquat.copy(),
            "cube_velocity": self.data.joint("cube_free").qvel[:3].copy(),
            # S2 (TASK-098): the at-rest scorer also bounds the cube's spin.
            "cube_angular_velocity": self.data.joint("cube_free").qvel[3:6].copy(),
            "target_position": self.data.body("target").xpos.copy(),
            "gripper_site_position": self.data.site("gripperframe").xpos.copy(),
            "fixed_jaw_contact": "gripper" in contacts,
            "moving_jaw_contact": "moving_jaw_so101_v1" in contacts,
            # Any gripper collision geom (either jaw or the camera mount) touching the cube.
            "gripper_contact": bool(contacts),
            "timestamp": float(self.data.time),
        }

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.closed = True
