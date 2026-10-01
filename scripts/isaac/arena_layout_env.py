"""A variant of Arena's ``galileo_g1_static_pick_and_place`` env for the e9 cross-sim check.

Container-only (imports Isaac Lab / Arena at module import). Loaded by Arena's own
``--external_environment_class_path arena_layout_env:OejG1StaticEnvironment`` with
``/oej/scripts/isaac`` on ``sys.path`` (``scripts/isaac/arena_e9_server.py``). Development
only (docs/ARENA.md §7): not a gated run, not a learned or policy result.

It keeps the tutorial's robot USD, embodiment (``g1_wbc_agile_joint``, AGILE WBC), background,
shelf support, apple, plate, finger friction and success term, and adds, all opt-in:

- ``--oej_hand_sensor``: a second contact sensor on the apple, filtered to the robot's wrist and
  hand links (the evaluator's apple-hand flag; read only, it changes no physics);
- ``--oej_robot_xyz X Y Z``: the robot's initial root position (the tutorial's is 0.25 0.08 0);
- ``--oej_platform_top_z Z``: a static, invisible box under the robot's feet with its top at
  ``Z`` (env-local), so the robot stands higher relative to the shelf (table-height matching);
- ``--oej_object_xyz`` / ``--oej_plate_xyz``: the apple's and plate's spawn positions (the
  e9 server teleports both after every reset anyway);
- ``--oej_episode_s``: the episode length (the tutorial's 6 s would time out an e9 attempt);
- ``--oej_hand_net_sensor``: a contact sensor on the right hand's palm and finger links with no
  filter (net contact force on each link; read only). With the apple and plate out of reach it
  reads hand-shelf contact (the shelf-press probe, docs/ARENA.md §8).

Terminations are left as the task defines them; the server holds them (records each term's
value and returns False) through ``arena_transport.ArenaScene(hold_terminations=True)``.
"""

from __future__ import annotations

import argparse

from isaaclab_arena_environments.galileo_g1_static_pick_and_place_environment import (
    GalileoG1StaticPickAndPlaceEnvironment,
)

HAND_LINKS = tuple(
    f"{side}_{link}"
    for side in ("left", "right")
    for link in (
        "wrist_yaw_link",
        "hand_palm_link",
        "hand_thumb_0_link",
        "hand_thumb_1_link",
        "hand_thumb_2_link",
        "hand_index_0_link",
        "hand_index_1_link",
        "hand_middle_0_link",
        "hand_middle_1_link",
    )
)
PLATFORM_SIZE_XY = (0.50, 0.70)
PLATFORM_DEPTH = 0.30  # the box reaches below the floor; static colliders do not interact


class OejG1StaticEnvironment(GalileoG1StaticPickAndPlaceEnvironment):
    name: str = "oej_g1_static_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab import sim as sim_utils
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.utils.pose import Pose

        env = super().get_env(args_cli)
        robot_xyz = tuple(args_cli.oej_robot_xyz) if args_cli.oej_robot_xyz else None
        if robot_xyz is not None:
            env.embodiment.set_initial_pose(
                Pose(position_xyz=robot_xyz, rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
            )
        if args_cli.oej_platform_top_z is not None:
            if robot_xyz is None:
                raise ValueError("--oej_platform_top_z needs --oej_robot_xyz")
            top = float(args_cli.oej_platform_top_z)

            class RobotPlatform(Object):
                def __init__(self):
                    self.spawner_cfg = sim_utils.CuboidCfg(
                        size=(*PLATFORM_SIZE_XY, PLATFORM_DEPTH),
                        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
                        visible=False,
                    )
                    super().__init__(
                        name="oej_robot_platform",
                        prim_path="{ENV_REGEX_NS}/oej_robot_platform",
                        object_type=ObjectType.SPAWNER,
                        initial_pose=Pose(
                            position_xyz=(robot_xyz[0], robot_xyz[1], top - PLATFORM_DEPTH / 2),
                            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
                        ),
                        tags=["background", "procedural"],
                    )

            env.scene.add_asset(RobotPlatform())
        for flag, asset in (
            ("oej_object_xyz", env.task.pick_up_object),
            ("oej_plate_xyz", env.task.destination_location),
        ):
            xyz = getattr(args_cli, flag)
            if xyz:
                asset.set_initial_pose(
                    Pose(position_xyz=tuple(xyz), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
                )
        inner = env.env_cfg_callback
        episode_s = args_cli.oej_episode_s
        hand_sensor = args_cli.oej_hand_sensor
        hand_net_sensor = args_cli.oej_hand_net_sensor

        def callback(env_cfg):
            env_cfg = inner(env_cfg) if inner is not None else env_cfg
            if episode_s is not None:
                env_cfg.episode_length_s = float(episode_s)
            if hand_sensor:
                from isaaclab.sensors import ContactSensorCfg

                plate_sensor = env_cfg.scene.pick_up_object_contact_sensor
                env_cfg.scene.oej_apple_hand_contact = ContactSensorCfg(
                    prim_path=plate_sensor.prim_path,
                    filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Robot/{n}" for n in HAND_LINKS],
                )
            if hand_net_sensor:
                from isaaclab.sensors import ContactSensorCfg

                env_cfg.scene.oej_right_hand_net_contact = ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_hand_.*_link"
                )
            return env_cfg

        env.env_cfg_callback = callback
        return env

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        GalileoG1StaticPickAndPlaceEnvironment.add_cli_args(parser)
        parser.add_argument("--oej_robot_xyz", type=float, nargs=3, default=None)
        parser.add_argument("--oej_platform_top_z", type=float, default=None)
        parser.add_argument("--oej_object_xyz", type=float, nargs=3, default=None)
        parser.add_argument("--oej_plate_xyz", type=float, nargs=3, default=None)
        parser.add_argument("--oej_episode_s", type=float, default=None)
        parser.add_argument("--oej_hand_sensor", action="store_true")
        parser.add_argument("--oej_hand_net_sensor", action="store_true")
