"""Robot-only hand-region crop of the onboard camera (TASK-048).

The crop window is located purely from robot kinematics: the right palm site
(``right_ee``, on the wrist-yaw link) and the torso-mounted ``onboard_rgb``
camera are both rigidly attached to robot links, so their poses are functions
of the robot's own joint state. No object, container, contact or task
quantity is read. The crop is therefore an allowed model input, like the
onboard image it is cut from.

It uses the same simulator kinematics buffers that the renderer draws, so the
window is consistent with the pixels. On hardware the same window would come
from forward kinematics of measured joints and the camera calibration.

``mujoco`` is imported lazily through the embodiment's simulation handle.
"""

from __future__ import annotations

import numpy as np

from embodied_jepa.contracts import ContractError

CAMERA = "onboard_rgb"
SITE = "right_ee"
CROP_CAMERA_NAME = "hand_crop_rgb"


class HandCrop:
    """Render the onboard camera at ``render_size`` and cut a ``crop_size`` window
    centred on the projected right palm, shifted to stay inside the image."""

    def __init__(self, robot, *, render_size=320, crop_size=112):
        for name, value in (("render_size", render_size), ("crop_size", crop_size)):
            if type(value) is not int or value < 1:
                raise ContractError(f"{name} must be a positive integer")
        if crop_size > render_size:
            raise ContractError("crop must fit inside the rendered image")
        self.robot = robot
        self.render_size, self.crop_size = render_size, crop_size
        mj = robot.mj
        self.camera_id = mj.mj_name2id(robot.model, mj.mjtObj.mjOBJ_CAMERA, CAMERA)
        if self.camera_id < 0:
            raise ContractError(f"missing camera {CAMERA}")
        fovy = float(robot.model.cam_fovy[self.camera_id])
        self.focal_px = (render_size / 2) / np.tan(np.deg2rad(fovy) / 2)
        self.renderer = mj.Renderer(robot.model, height=render_size, width=render_size)

    def window(self, data=None):
        """Top-left (row, col) of the crop from palm/camera link poses only."""
        data = self.robot.sim.data if data is None else data
        palm = np.asarray(data.site(SITE).xpos, dtype=float)
        origin = np.asarray(data.cam_xpos[self.camera_id], dtype=float)
        rotation = np.asarray(data.cam_xmat[self.camera_id], dtype=float).reshape(3, 3)
        local = rotation.T @ (palm - origin)  # MuJoCo camera looks along -z, y up
        half, size = self.render_size / 2, self.crop_size
        if local[2] >= -1e-6:
            col, row = half, half  # behind the camera: fall back to the image centre
        else:
            col = half + self.focal_px * local[0] / -local[2]
            row = half - self.focal_px * local[1] / -local[2]
        top = int(np.clip(round(row - size / 2), 0, self.render_size - size))
        left = int(np.clip(round(col - size / 2), 0, self.render_size - size))
        return top, left

    def capture(self):
        data = self.robot.sim.data
        self.renderer.update_scene(data, camera=CAMERA)
        image = self.renderer.render()
        top, left = self.window(data)
        crop = image[top : top + self.crop_size, left : left + self.crop_size].copy()
        return crop, np.array([top, left], dtype=np.int16)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
