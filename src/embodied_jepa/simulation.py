"""Native MuJoCo transport and scoring truth; importable without MuJoCo installed."""

from __future__ import annotations

import hashlib
import importlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[2]


class MuJoCoSimulation:
    """Fixed-pelvis G1/Dex3 tabletop with explicitly synthetic apple/plate proxies."""

    def __init__(
        self,
        *,
        asset_root=None,
        width=96,
        height=96,
        control_dt=0.05,
        render=True,
        object_kind="apple",
        container_kind="plate",
    ):
        if object_kind not in ("apple", "cube", "banana") or container_kind not in (
            "plate",
            "bowl",
            "target",
        ):
            raise ContractError("unsupported procedural object/container kind")
        self.object_kind, self.container_kind = object_kind, container_kind
        self.object_support_height = {"apple": 0.027, "cube": 0.023, "banana": 0.018}[object_kind]
        self.mj = importlib.import_module("mujoco")
        self.asset_root = Path(asset_root or ROOT / "third_party/unitree_mujoco")
        manifest = json.loads((ROOT / "assets/manifest.json").read_text())
        for relative, expected in manifest["sha256"].items():
            path = self.asset_root / relative
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ContractError(
                    f"Missing or modified asset {path}; run scripts/fetch_assets.py"
                )
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
        self.qadr = self.model.jnt_qposadr[self.joint_ids]
        self.vadr = self.model.jnt_dofadr[self.joint_ids]
        hand = np.array(["hand_" in name for name in self.joint_names])
        self.kp = np.where(hand, 4.0, 100.0)
        self.kd = np.where(hand, 0.2, 5.0)
        self.targets = np.zeros(self.model.nu)
        self.stopped_reason = ""
        self.reset()

    def _scene(self):
        directory = self.asset_root / "unitree_robots/g1"
        root = ET.parse(directory / "g1_29dof_with_hand.xml").getroot()
        root.find("compiler").set("meshdir", str((directory / "meshes").resolve()))
        pelvis = root.find(".//body[@name='pelvis']")
        pelvis.remove(pelvis.find("joint[@type='free']"))
        ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
        world = root.find("worldbody")
        ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", directional="true")
        ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 .05", rgba=".3 .35 .4 1")
        ET.SubElement(
            world,
            "geom",
            name="table",
            type="box",
            pos=".45 0 .70",
            size=".32 .45 .04",
            rgba=".55 .4 .25 1",
        )
        ET.SubElement(
            world, "camera", name="overview", pos="1.5 -1.5 1.5", xyaxes=".707 .707 0 -.3 .3 .9"
        )
        torso = root.find(".//body[@name='torso_link']")
        ET.SubElement(
            torso,
            "camera",
            name="onboard_rgb",
            pos=".08 0 .35",
            xyaxes="0 -1 0 .866 0 .5",
            fovy="75",
        )
        for side in ("left", "right"):
            wrist = root.find(f".//body[@name='{side}_wrist_yaw_link']")
            ET.SubElement(
                wrist, "site", name=f"{side}_ee", pos=".12 0 0", size=".003", rgba="0 0 0 0"
            )
        apple = ET.SubElement(world, "body", name="apple", pos=".35 -.16 .768")
        ET.SubElement(apple, "freejoint", name="apple_free")
        shape = {
            "apple": {"type": "sphere", "size": ".027", "rgba": ".85 .07 .04 1"},
            "cube": {"type": "box", "size": ".023 .023 .023", "rgba": ".1 .75 .2 1"},
            "banana": {
                "type": "capsule",
                "size": ".018 .035",
                "quat": ".70710678 0 .70710678 0",
                "rgba": ".95 .8 .06 1",
            },
        }[self.object_kind]
        ET.SubElement(apple, "geom", name="apple_geom", mass=".08", friction="1 .01 .001", **shape)
        plate = ET.SubElement(world, "body", name="plate", pos=".47 -.15 .746")
        color = {"plate": ".15 .35 .85 1", "bowl": ".7 .2 .7 1", "target": ".15 .75 .7 1"}[
            self.container_kind
        ]
        if self.container_kind == "target":
            ET.SubElement(
                plate, "geom", name="plate_base", type="box", size=".07 .07 .002", rgba=color
            )
            self.container_surface_z = 0.748
        else:
            ET.SubElement(
                plate,
                "geom",
                name="plate_base",
                type="cylinder",
                size=".07 .006",
                rgba=color,
                friction="1 .01 .001",
            )
            self.container_surface_z = 0.752
            rim_height = 0.009 if self.container_kind == "plate" else 0.025
            for i in range(16):
                a, b = 2 * np.pi * i / 16, 2 * np.pi * (i + 1) / 16
                xyz = [
                    0.067 * np.cos(a),
                    0.067 * np.sin(a),
                    rim_height,
                    0.067 * np.cos(b),
                    0.067 * np.sin(b),
                    rim_height,
                ]
                ET.SubElement(
                    plate,
                    "geom",
                    name=f"plate_rim_{i}",
                    type="capsule",
                    fromto=" ".join(map(str, xyz)),
                    size=".004" if self.container_kind == "plate" else ".012",
                    rgba=color,
                )
        return ET.tostring(root, encoding="unicode")

    def _require_open(self):
        if self.closed:
            raise RuntimeError("simulation is closed")

    def reset(
        self,
        seed=0,
        *,
        object_xy=None,
        plate_xy=None,
        object_kind=None,
        container_kind=None,
        object_on_container=False,
    ):
        """Reset robot/scene; xy coordinates are world-frame meters on the tabletop."""
        self._require_open()
        if object_kind not in (None, self.object_kind) or container_kind not in (
            None,
            self.container_kind,
        ):
            raise ContractError(
                "geometry kind is fixed per simulator; construct a new scene for another pair"
            )
        if type(object_on_container) is not bool:
            raise ContractError("object_on_container must be an explicit boolean")
        rng = np.random.default_rng(seed)
        plate_xy = np.asarray(plate_xy if plate_xy is not None else [0.48, -0.10], dtype=float)
        if object_xy is None and object_on_container:
            object_xy = plate_xy.copy()
        object_xy = np.asarray(
            object_xy
            if object_xy is not None
            else [0.40 + rng.uniform(-0.02, 0.02), -0.26 + rng.uniform(-0.02, 0.02)],
            dtype=float,
        )
        for name, xy in (("object", object_xy), ("plate", plate_xy)):
            if (
                xy.shape != (2,)
                or not np.isfinite(xy).all()
                or not (0.18 <= xy[0] <= 0.65 and -0.32 <= xy[1] <= 0.32)
            ):
                raise ContractError(f"{name} reset xy must lie on the tabletop")
        # Conservative XY collision envelopes: compiled reset orientations are fixed.
        object_radius = {"apple": 0.027, "cube": np.sqrt(2) * 0.023, "banana": 0.053}[
            self.object_kind
        ]
        offset = np.abs(object_xy - plate_xy)
        if object_on_container:
            interior = (
                0.07
                if self.container_kind == "target"
                else (0.063 if self.container_kind == "plate" else 0.055)
            )
            extent = (
                float(offset.max())
                if self.container_kind == "target"
                else float(np.linalg.norm(offset))
            )
            if extent + object_radius > interior - 0.001:
                raise ContractError(
                    "supported placement must fit inside container without rim penetration"
                )
            object_z = self.container_surface_z + self.object_support_height + 0.001
        else:
            if self.container_kind == "target":
                separation = np.linalg.norm(np.maximum(offset - 0.07, 0)) - object_radius
            else:
                container_radius = 0.071 if self.container_kind == "plate" else 0.079
                separation = np.linalg.norm(offset) - container_radius - object_radius
            if separation < 0.005:
                raise ContractError(
                    "object/container reset overlaps collision envelopes; separate them "
                    "or use object_on_container=True for a supported goal"
                )
            object_z = 0.743 + self.object_support_height
        self.mj.mj_resetData(self.model, self.data)
        apple_q = int(self.model.joint("apple_free").qposadr[0])
        self.data.qpos[apple_q : apple_q + 3] = [*object_xy, object_z]
        self.model.body("plate").pos[:] = [*plate_xy, 0.746]
        # Slight arm flexion keeps the initial Jacobian away from straight-arm singularity.
        for side in ("left", "right"):
            self.data.joint(f"{side}_elbow_joint").qpos[0] = 0.08
        self.mj.mj_forward(self.model, self.data)
        self.targets[:] = self.data.qpos[self.qadr]
        self.stopped_reason = ""
        return self.task_truth()

    def render(self, camera="onboard_rgb"):
        self._require_open()
        if not self.render_enabled:
            raise RuntimeError("RGB rendering was explicitly disabled")
        if self.renderer is None:
            self.renderer = self.mj.Renderer(self.model, height=self.height, width=self.width)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def read(self):
        self._require_open()
        return {
            "rgb": self.render(),
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
        limits = self.model.jnt_range[self.joint_ids]
        if np.any(targets < limits[:, 0]) or np.any(targets > limits[:, 1]):
            raise ContractError("joint targets exceed MJCF limits")
        old = self.targets.copy()
        self.targets[:] = targets
        for k in range(self.substeps):
            interpolated = old + (targets - old) * ((k + 1) / self.substeps)
            torque = (
                self.kp * (interpolated - self.data.qpos[self.qadr])
                - self.kd * self.data.qvel[self.vadr]
            )
            torque += self.data.qfrc_bias[self.vadr]
            self.data.ctrl[:] = np.clip(
                torque, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
            )
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
        limits = self.model.jnt_range[self.joint_ids]
        self.targets[:] = np.clip(self.data.qpos[self.qadr], limits[:, 0], limits[:, 1])
        self.data.ctrl[:] = 0
        self.stopped_reason = str(reason)

    def task_truth(self):
        """Evaluator/collector-only simulator truth: never included in canonical observation."""
        self._require_open()
        apple = self.data.body("apple").xpos.copy()
        plate = self.data.body("plate").xpos.copy()
        apple_id = self.model.geom("apple_geom").id
        hand_contact = False
        for contact in self.data.contact[: self.data.ncon]:
            if apple_id in (contact.geom1, contact.geom2):
                other = contact.geom2 if contact.geom1 == apple_id else contact.geom1
                body_name = self.model.body(int(self.model.geom_bodyid[other])).name
                hand_contact |= "hand_" in body_name or "wrist_" in body_name
        velocity = self.data.joint("apple_free").qvel[:3].copy()
        return {
            "position_frame": "world",
            "object_kind": self.object_kind,
            "container_kind": self.container_kind,
            "object_support_height": self.object_support_height,
            "container_surface_z": self.container_surface_z,
            "base_position_world": self.data.body("pelvis").xpos.copy(),
            "base_rotation_world": self.data.body("pelvis").xmat.reshape(3, 3).copy(),
            "object_position": apple,
            "plate_position": plate,
            "object_velocity": velocity,
            "hand_contact": bool(hand_contact),
            "lifted": bool(apple[2] > 0.82),
            "placed": bool(
                np.linalg.norm(apple[:2] - plate[:2]) < 0.04
                and abs(apple[2] - (self.container_surface_z + self.object_support_height)) < 0.012
                and np.linalg.norm(velocity) < 0.1
                and not hand_contact
            ),
            "dropped": bool(apple[2] < 0.70),
            "timestamp": float(self.data.time),
        }

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.closed = True
