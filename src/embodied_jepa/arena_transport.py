"""Opt-in adapter for the Isaac Lab-Arena G1 static pick-and-place scene (development spike).

The scene is the one NVIDIA's GR00T end-to-end tutorial evaluates in ("Sim Evaluation"):
environment ``galileo_g1_static_pick_and_place``, object ``apple_01_objaverse_robolab``,
destination ``clay_plates_hot3d_robolab``, embodiment ``g1_wbc_agile_joint``, IsaacLab-Arena
``release/0.2.1``. See ``docs/ARENA.md``.

NumPy only at import. ``ArenaScene`` imports Isaac Lab / Arena inside its constructor and only
works inside the Arena container (``scripts/isaac/run_isaac.sh``). Nothing in the package
imports this module. Nothing here is a learned, policy or task result: the only policies are a
hold command and scripted teleports of the apple.

The action of ``g1_wbc_agile_joint`` is 50-D (Arena ``G1DecoupledWBCJointAction``):
``[0:43]`` absolute joint-position targets in the articulation's own (breadth-first) joint order,
of which the WBC uses only the upper body (arms, wrists, Dex3; legs and waist come from the AGILE
ONNX lower-body policy), then ``navigate_cmd`` (vx, vy, wz), ``base_height_command`` (pelvis
height, m) and ``torso_orientation_rpy_cmd`` (3; ignored by the AGILE backend).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from embodied_jepa.contracts import ContractError

ARENA_REF = "release/0.2.1 @ 8b4a3a47fc53de23e8205089d71109a2e2348acd"
TUTORIAL = {
    "environment": "galileo_g1_static_pick_and_place",
    "object": "apple_01_objaverse_robolab",
    "destination": "clay_plates_hot3d_robolab",
    "embodiment": "g1_wbc_agile_joint",
    "num_steps": 600,
    "episode_length_s": 6.0,
    "force_threshold_n": 0.5,
    "velocity_threshold_m_s": 0.1,
}

G1_NUM_JOINTS = 43
NAVIGATE = slice(43, 46)
BASE_HEIGHT = 43 + 3
TORSO_RPY = slice(47, 50)
ACTION_DIM = 50
STANDING_HEIGHT_M = 0.75  # Arena's own standing-idle value (actions[:, -4] = 0.75 in its tests)


def pin_pxr_work_thread_limit(value: str | None = None, environ=None) -> str | None:
    """Pin ``PXR_WORK_THREAD_LIMIT`` in ``os.environ`` before Kit starts (PR #112's workaround).

    Kit overwrites the variable during start-up, so a value passed in from outside has no effect
    on its own (docs/ISAAC_E9_REPLAY.md §4). As in ``scripts/isaac/e9_server_isaac.py``, later
    writes of that one key are replaced by the pinned value. ``value`` defaults to an inherited
    positive integer, else ``"1"``; ``"0"`` means no pin (it does not undo an earlier pin).
    Returns the pinned value or None. ``environ`` must be ``os.environ`` or another
    ``os._Environ`` (the pin swaps its class); anything else, e.g. a plain dict, is refused.
    """
    import os

    env = os.environ if environ is None else environ
    if not isinstance(env, os._Environ):
        raise ContractError("environ must be os.environ (an os._Environ), not a plain mapping")
    if value is None:
        value = env.get("PXR_WORK_THREAD_LIMIT", "").strip() or "1"
    if not value.isdigit():
        raise ContractError(f"PXR_WORK_THREAD_LIMIT={value!r} is not an integer >= 0")
    if int(value) == 0:
        return None
    pin = str(int(value))
    if not getattr(type(env), "_oej_pxr_pin", None):

        class _Pinned(type(env)):
            _oej_pxr_pin = pin

            def __setitem__(self, key, val):
                if key == "PXR_WORK_THREAD_LIMIT" and val != self._oej_pxr_pin:
                    os.write(2, f"[arena] replaced write {key}={val!r} by {pin}\n".encode())
                    val = self._oej_pxr_pin
                super().__setitem__(key, val)

        env.__class__ = _Pinned
    type(env)._oej_pxr_pin = pin
    env["PXR_WORK_THREAD_LIMIT"] = pin
    return pin


def tutorial_argv(
    *,
    headless: bool = True,
    enable_cameras: bool = True,
    num_envs: int = 1,
    environment: str = TUTORIAL["environment"],
    external_environment: str | None = None,
):
    """The Arena CLI arguments of the tutorial's evaluation command, minus the GR00T policy.

    ``--viz kit`` (an on-screen viewer) is replaced by ``--headless``. ``environment`` and
    ``external_environment`` (Arena's ``module:Class`` for an environment outside Arena, e.g.
    ``scripts/isaac/arena_layout_env.py``) select a variant; the defaults are the tutorial's.
    """
    argv = []
    if headless:
        argv.append("--headless")
    if enable_cameras:
        argv.append("--enable_cameras")
    if num_envs != 1:
        argv += ["--num_envs", str(int(num_envs))]
    if external_environment is not None:
        argv += ["--external_environment_class_path", str(external_environment)]
    argv += [
        str(environment),
        "--object",
        TUTORIAL["object"],
        "--destination",
        TUTORIAL["destination"],
        "--embodiment",
        TUTORIAL["embodiment"],
    ]
    return argv


def wbc_action(
    targets: Mapping[str, float],
    sim_joint_names: Sequence[str],
    *,
    base_height: float = STANDING_HEIGHT_M,
    navigate: Sequence[float] = (0.0, 0.0, 0.0),
    torso_rpy: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Build one 50-D ``g1_wbc_agile_joint`` action from joint targets keyed by joint name.

    ``targets`` must name every one of the 43 articulation joints exactly once (map by name: the
    Arena articulation orders joints breadth-first, not in MJCF order). Values are not clipped.
    """
    names = list(sim_joint_names)
    if len(names) != G1_NUM_JOINTS or len(set(names)) != G1_NUM_JOINTS:
        raise ContractError(f"expected {G1_NUM_JOINTS} distinct articulation joint names")
    missing = [n for n in names if n not in targets]
    extra = [n for n in targets if n not in set(names)]
    if missing or extra:
        raise ContractError(f"joint targets do not match the articulation: {missing=} {extra=}")
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[:G1_NUM_JOINTS] = [float(targets[n]) for n in names]
    nav = np.asarray(navigate, dtype=float)
    rpy = np.asarray(torso_rpy, dtype=float)
    if nav.shape != (3,) or rpy.shape != (3,):
        raise ContractError("navigate and torso_rpy are 3-vectors")
    action[NAVIGATE] = nav
    action[BASE_HEIGHT] = float(base_height)
    action[TORSO_RPY] = rpy
    if not np.all(np.isfinite(action)):
        raise ContractError("non-finite action")
    return action


def quat_xyzw_to_matrix(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if not np.isfinite(n) or n < 1e-9:
        raise ContractError("zero or non-finite quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def project_points_ros(points_w, cam_pos_w, cam_quat_w_ros_xyzw, intrinsics) -> np.ndarray:
    """Pixel (u, v) and depth of world points for a camera in ROS optical convention.

    ROS optical frame: +Z forward, +X right, +Y down (Isaac Lab ``CameraData.quat_w_ros``,
    xyzw). Returns an (N, 3) array ``[u, v, depth]``; points behind the camera get NaN pixels.
    """
    p = np.atleast_2d(np.asarray(points_w, dtype=float))
    r = quat_xyzw_to_matrix(cam_quat_w_ros_xyzw)
    k = np.asarray(intrinsics, dtype=float).reshape(3, 3)
    pc = (p - np.asarray(cam_pos_w, dtype=float)) @ r  # world -> camera: R^T (p - t)
    out = np.full((len(p), 3), np.nan)
    out[:, 2] = pc[:, 2]
    front = pc[:, 2] > 1e-9
    uvw = pc[front] @ k.T
    out[front, 0] = uvw[:, 0] / uvw[:, 2]
    out[front, 1] = uvw[:, 1] / uvw[:, 2]
    return out


def horizontal_fov_deg(focal_length: float, horizontal_aperture: float) -> float:
    """Pinhole horizontal field of view from USD focal length and aperture (same units)."""
    return float(np.degrees(2.0 * np.arctan(0.5 * horizontal_aperture / focal_length)))


def is_blank(image: np.ndarray, max_level: int = 2) -> bool:
    """True for an all-black (or near-black) frame, e.g. a camera that has not rendered yet."""
    return int(np.asarray(image).max(initial=0)) <= max_level


def patch_mean_rgb(image: np.ndarray, u: float, v: float, half: int = 2) -> list[float] | None:
    """Mean RGB of a (2*half+1)^2 patch around pixel (u, v); None if it leaves the image."""
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] < 3 or not (np.isfinite(u) and np.isfinite(v)):
        return None
    c, r = int(round(u)), int(round(v))
    if r - half < 0 or c - half < 0 or r + half >= img.shape[0] or c + half >= img.shape[1]:
        return None
    patch = img[r - half : r + half + 1, c - half : c + half + 1, :3].astype(float)
    return patch.reshape(-1, 3).mean(axis=0).tolist()


class ArenaScene:
    """The tutorial scene, built exactly as ``policy_runner.py`` builds it, with a local policy.

    Container-only. Construct once per process (it launches Kit). ``step`` takes a 50-D action
    (``wbc_action``) and returns the termination flags; ``frames`` returns every scene camera's
    RGB; ``truth`` is evaluator-only (apple/plate state, never a policy input).
    """

    def __init__(
        self,
        extra_argv: Sequence[str] = (),
        *,
        num_envs: int = 1,
        enable_cameras: bool = True,
        environment: str = TUTORIAL["environment"],
        external_environment: str | None = None,
        hold_terminations: bool = False,
    ):
        import sys

        from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
        from isaaclab_arena.utils.isaaclab_utils.simulation_app import get_app_launcher

        argv = tutorial_argv(
            num_envs=num_envs,
            enable_cameras=enable_cameras,
            environment=environment,
            external_environment=external_environment,
        ) + list(extra_argv)
        sys.argv = [sys.argv[0], *argv]  # the Arena environment parser reads sys.argv
        parser = get_isaaclab_arena_cli_parser()
        args, _ = parser.parse_known_args(argv)
        self.app_launcher = get_app_launcher(args)

        import torch
        import warp as wp
        from isaaclab_arena_environments.cli import (
            get_arena_builder_from_cli,
            get_isaaclab_arena_environments_cli_parser,
        )

        self._torch, self._wp = torch, wp
        parser = get_isaaclab_arena_environments_cli_parser(parser)
        self.args = parser.parse_args(argv)
        builder = get_arena_builder_from_cli(self.args)
        self.env, self.cfg = builder.make_registered_and_return_cfg(render_mode="rgb_array")
        self.unwrapped = self.env.unwrapped
        self.arena_env = self.unwrapped.cfg.isaaclab_arena_env
        self.robot = self.unwrapped.scene["robot"]
        self.apple = self.unwrapped.scene[self.arena_env.task.pick_up_object.name]
        self.plate = self.unwrapped.scene[self.arena_env.task.destination_location.name]
        self.joint_names = list(self.robot.data.joint_names)
        self.success_inputs: dict = {}
        self.held_terms: dict[str, bool] = {}
        self._wrap_success_term()
        if hold_terminations:
            self._hold_terminations()

    def _wrap_success_term(self) -> None:
        """Record the success term's own inputs when it is evaluated, before any auto-reset.

        Isaac Lab resets a terminated env inside ``step()``, so state read after ``step()`` on a
        terminating step is the next episode's. The wrapper stores the apple-plate force and the
        apple speed exactly as ``object_on_destination`` saw them (env 0); report-only.
        """
        tm = self.unwrapped.termination_manager
        if "success" not in tm.active_terms:
            return
        cfg = tm.get_term_cfg("success")
        inner = cfg.func

        def recorded(env, **params):
            value = inner(env, **params)
            try:
                sensor = env.unwrapped.scene[params["contact_sensor_cfg"].name]
                obj = env.unwrapped.scene[params["object_cfg"].name]
                force = self.np(sensor.data.force_matrix_w).reshape(env.unwrapped.num_envs, -1, 3)
                vel = self.np(obj.data.root_lin_vel_w)
                self.success_inputs = {
                    "value": bool(self.np(value)[0]),
                    "apple_plate_force_n": float(np.linalg.norm(force[0], axis=-1).max()),
                    "apple_speed_m_s": float(np.linalg.norm(vel[0])),
                    "apple_pos_local": (
                        self.np(obj.data.root_pos_w)[0]
                        - self.np(env.unwrapped.scene.env_origins)[0]
                    ).tolist(),
                }
            except Exception as exc:  # noqa: BLE001 - report-only
                self.success_inputs = {"error": f"{type(exc).__name__}: {exc}"}
            return value

        cfg.func = recorded

    def _hold_terminations(self) -> None:
        """Evaluate every termination term as usual, record its value (env 0), return False.

        For runs longer than the task's episode (an e9 attempt is about 40 s): the env never
        terminates or auto-resets, and the caller reads ``held_terms`` after each step (also
        returned by ``step``). The success term's own inputs are still recorded."""
        torch = self._torch
        tm = self.unwrapped.termination_manager
        for name in tm.active_terms:
            cfg = tm.get_term_cfg(name)
            inner = cfg.func

            def held(env, _inner=inner, _name=name, **params):
                value = _inner(env, **params)
                self.held_terms[_name] = bool(self.np(value)[0])
                return torch.zeros_like(value)

            cfg.func = held

    # ---------------------------------------------------------------- helpers
    def np(self, x) -> np.ndarray:
        t = x if isinstance(x, self._torch.Tensor) else self._wp.to_torch(x)
        return t.detach().cpu().numpy()

    def cameras(self) -> dict:
        from isaaclab.sensors import Camera, TiledCamera

        return {
            name: s
            for name, s in self.unwrapped.scene.sensors.items()
            if isinstance(s, (Camera, TiledCamera))
        }

    # ---------------------------------------------------------------- control
    def reset(self, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        return obs

    def hold_targets(self) -> dict[str, float]:
        """The articulation's default joint positions (the env's open-arm initial pose)."""
        q = self.np(self.robot.data.default_joint_pos)[0]
        return dict(zip(self.joint_names, q.tolist(), strict=True))

    def step(self, action: np.ndarray) -> dict:
        torch = self._torch
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (ACTION_DIM,):
            raise ContractError(f"expected a ({ACTION_DIM},) action")
        batch = torch.as_tensor(a, device=self.unwrapped.device).repeat(self.unwrapped.num_envs, 1)
        with torch.inference_mode():
            obs, _, terminated, truncated, _ = self.env.step(batch)
        tm = self.unwrapped.termination_manager
        terms = {n: bool(self.np(tm.get_term(n))[0]) for n in tm.active_terms}
        return {
            "terminated": bool(self.np(terminated)[0]),
            "truncated": bool(self.np(truncated)[0]),
            "terms": terms,
            "held_terms": dict(self.held_terms),
            "success_inputs": dict(self.success_inputs),  # pre-reset, from the success term
            "obs": obs,
        }

    def frames(self) -> dict[str, np.ndarray]:
        out = {}
        for name, cam in self.cameras().items():
            rgb = cam.data.output.get("rgb")
            if rgb is not None:
                out[name] = self.np(rgb)[0][..., :3].astype(np.uint8)
        return out

    def camera_pose(self, name: str) -> dict:
        """Live camera pose (env-local position, ROS xyzw quaternion) and intrinsics."""
        cam = self.cameras()[name]
        origin = self.np(self.unwrapped.scene.env_origins)[0]
        return {
            "pos_w": self.np(cam.data.pos_w)[0].tolist(),
            "pos_local": (self.np(cam.data.pos_w)[0] - origin).tolist(),
            "quat_w_ros_xyzw": self.np(cam.data.quat_w_ros)[0].tolist(),
            "intrinsics": self.np(cam.data.intrinsic_matrices)[0].tolist(),
        }

    def teleport_apple(self, xyz_local, quat_xyzw=(0.0, 0.0, 0.0, 1.0)) -> None:
        """Scripted-check harness only (the success-path check). Not a controller API."""
        from isaaclab_arena.utils.pose import Pose

        with self._torch.inference_mode():
            self.arena_env.task.pick_up_object.set_object_pose(
                self.env,
                Pose(position_xyz=tuple(map(float, xyz_local)), rotation_xyzw=tuple(quat_xyzw)),
            )

    def set_object_pose(self, role: str, xyz_local, quat_xyzw=(0.0, 0.0, 0.0, 1.0)) -> None:
        """Teleport the apple (``role="apple"``) or plate (``"plate"``) with zero velocity.

        Harness only (scene layout at reset): not a controller API."""
        from isaaclab_arena.utils.pose import Pose

        asset = {
            "apple": self.arena_env.task.pick_up_object,
            "plate": self.arena_env.task.destination_location,
        }[role]
        with self._torch.inference_mode():
            asset.set_object_pose(
                self.env,
                Pose(position_xyz=tuple(map(float, xyz_local)), rotation_xyzw=tuple(quat_xyzw)),
            )

    def raw_state(self, *, bodies: bool = False) -> dict:
        """Evaluator-only raw state of env 0 as NumPy (env-local positions, xyzw quaternions).

        Robot joints in the articulation's own order (``joint_names``), the root (pelvis) pose
        and velocity, the apple's link pose, centre-of-mass pose and velocity, the plate's pose
        and velocity, the apple-plate force (the success term's sensor) and, if the scene has
        ``oej_apple_hand_contact``, the apple-hand force. ``bodies`` adds every robot link's
        pose (``robot.data.body_names`` order)."""
        origin = self.np(self.unwrapped.scene.env_origins)[0]
        r = self.robot.data

        def local(pose):
            pose = np.asarray(pose, float).copy()
            pose[..., :3] -= origin
            return pose

        apple, plate = self.apple.data, self.plate.data
        plate_sensor = self.unwrapped.scene["pick_up_object_contact_sensor"]
        out = {
            "q": self.np(r.joint_pos)[0].astype(float),
            "qd": self.np(r.joint_vel)[0].astype(float),
            "q_target": self.np(r.joint_pos_target)[0].astype(float),
            "pelvis_pose": local(self.np(r.root_link_pose_w)[0]),
            "pelvis_vel": self.np(r.root_link_vel_w)[0].astype(float),
            "apple_pose": local(self.np(apple.root_link_pose_w)[0]),
            "apple_com_pose": local(self.np(apple.root_com_pose_w)[0]),
            "apple_com_vel": self.np(apple.root_com_vel_w)[0].astype(float),
            "plate_pose": local(self.np(plate.root_link_pose_w)[0]),
            "plate_vel": self.np(plate.root_com_vel_w)[0].astype(float),
            "apple_plate_force_n": float(
                np.linalg.norm(
                    self.np(plate_sensor.data.force_matrix_w).reshape(-1, 3), axis=-1
                ).max()
            ),
        }
        if "oej_apple_hand_contact" in self.unwrapped.scene.keys():
            hand = self.unwrapped.scene["oej_apple_hand_contact"]
            force = np.linalg.norm(self.np(hand.data.force_matrix_w).reshape(-1, 3), axis=-1)
            out["apple_hand_force_n"] = float(force.max())
        if bodies:
            out["body_pose"] = local(self.np(r.body_link_pose_w)[0])
        return out

    def truth(self) -> dict:
        """Evaluator-only state: never a policy input."""
        origin = self.np(self.unwrapped.scene.env_origins)[0]
        ap = self.np(self.apple.data.root_pos_w)[0]
        pp = self.np(self.plate.data.root_pos_w)[0]
        sensor = self.unwrapped.scene["pick_up_object_contact_sensor"]
        force = self.np(sensor.data.force_matrix_w).reshape(-1, 3)
        return {
            "apple_pos_local": (ap - origin).tolist(),
            "apple_lin_vel": self.np(self.apple.data.root_lin_vel_w)[0].tolist(),
            "plate_pos_local": (pp - origin).tolist(),
            "apple_plate_force_n": float(np.linalg.norm(force, axis=-1).max()),
            "pelvis_pos_local": (self.np(self.robot.data.root_pos_w)[0] - origin).tolist(),
        }

    def metrics(self) -> dict | None:
        from isaaclab_arena.metrics.metrics import compute_metrics

        if getattr(self.unwrapped.cfg, "metrics", None) is None:
            return None
        return compute_metrics(self.unwrapped)

    def close(self) -> None:
        self.env.close()
