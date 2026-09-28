"""Dump what Isaac Sim actually simulates for a converted G1 + Dex3 USD.

Development tooling (not a gated experiment). Runs INSIDE the ``isaaclab_arena`` container
(``scripts/isaac/run_isaac.sh audit_usd.py <out> --usd /oej/usd/...``). It spawns the USD as a
fixed-root Isaac Lab articulation (no stepping) and writes ``isaac_model.json``:

* joints: PhysX-parsed limits, drive stiffness/damping, armature, friction, effort/velocity
  limits, plus the USD joint frames (axis token, localPos/Rot 0/1) for axis and
  parent-to-child transform checks;
* bodies: PhysX masses, inertia tensors and centres of mass (body frame), and the number of
  collision shapes below each body with their collision approximation;
* forward kinematics: world link poses at the zero pose and at a few seeded random poses
  (joint positions listed in Isaac's joint order and by name), so the host can compare FK.

``compare_joints.py --isaac-model`` compares this dump with the MuJoCo model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--usd", required=True, help="main .usda of the converted asset")
parser.add_argument("--fk_poses", type=int, default=3, help="seeded random FK poses")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.output.exists():
    raise SystemExit(f"refusing to overwrite {args.output}")
args.output.mkdir(parents=True)
app = AppLauncher(args).app

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab.actuators import IdealPDActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402


def t(x) -> np.ndarray:
    return (
        wp.to_torch(x).detach().cpu().numpy()
        if not isinstance(x, torch.Tensor)
        else (x.detach().cpu().numpy())
    )


def quat_wxyz(q) -> list[float]:
    return [float(q.GetReal()), *(float(v) for v in q.GetImaginary())]


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.002, device=args.device))
    cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args.usd,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True),
        ),
        # The converted USD keeps the MJCF pelvis pos (0, 0, 0.793) on the pelvis prim, so the
        # spawn offset is zero; a 0.793 offset here would put the pelvis at 1.586 m.
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0, 0, 0, 1)),
        actuators={"all": IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
    )
    robot = Articulation(cfg)
    stage = sim_utils.get_current_stage()
    usd_joints = {}
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith("/World/Robot"):
            continue
        if prim.IsA(UsdPhysics.RevoluteJoint):
            j = UsdPhysics.RevoluteJoint(prim)
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            usd_joints[prim.GetName()] = {
                "path": str(prim.GetPath()),
                "axis": j.GetAxisAttr().Get(),
                "local_pos0": list(j.GetLocalPos0Attr().Get()),
                "local_rot0_wxyz": quat_wxyz(j.GetLocalRot0Attr().Get()),
                "local_pos1": list(j.GetLocalPos1Attr().Get()),
                "local_rot1_wxyz": quat_wxyz(j.GetLocalRot1Attr().Get()),
                "body0": str(j.GetBody0Rel().GetTargets()[0]).rsplit("/", 1)[-1],
                "body1": str(j.GetBody1Rel().GetTargets()[0]).rsplit("/", 1)[-1],
                "usd_lower_deg": j.GetLowerLimitAttr().Get(),
                "usd_upper_deg": j.GetUpperLimitAttr().Get(),
                "usd_drive_stiffness": drive.GetStiffnessAttr().Get() if drive else None,
                "usd_drive_damping": drive.GetDampingAttr().Get() if drive else None,
                "usd_drive_max_force": drive.GetMaxForceAttr().Get() if drive else None,
                "usd_physx_armature": prim.GetAttribute("physxJoint:armature").Get(),
                "usd_physx_joint_friction": prim.GetAttribute("physxJoint:jointFriction").Get(),
                "usd_mjc_damping": prim.GetAttribute("mjc:damping").Get(),
            }
        elif prim.IsA(UsdPhysics.Joint) and not prim.IsA(UsdPhysics.RevoluteJoint):
            usd_joints.setdefault("_other_joints", []).append(
                {"path": str(prim.GetPath()), "type": prim.GetTypeName()}
            )

    sim.reset()
    names = list(robot.joint_names)
    bodies = list(robot.body_names)
    lim = t(robot.data.joint_pos_limits)[0]
    per_joint = {
        "stiffness": t(robot.data.joint_stiffness)[0],
        "damping": t(robot.data.joint_damping)[0],
        "armature": t(robot.data.joint_armature)[0],
        "friction_coeff": t(robot.data.joint_friction_coeff)[0],
        "dynamic_friction_coeff": t(robot.data.joint_dynamic_friction_coeff)[0],
        "viscous_friction_coeff": t(robot.data.joint_viscous_friction_coeff)[0],
        "velocity_limit": t(robot.data.joint_vel_limits)[0],
        "effort_limit": t(robot.data.joint_effort_limits)[0],
    }
    # PhysX drive values after Isaac Lab's explicit (IdealPD, kp = kd = 0) actuator init;
    # the USD-authored drive values are in row["usd"].
    view = robot.root_view
    physx_gains = {
        "physx_stiffness_after_init": t(view.get_dof_stiffnesses())[0],
        "physx_damping_after_init": t(view.get_dof_dampings())[0],
        "physx_max_force_after_init": t(view.get_dof_max_forces())[0],
    }
    joints = []
    for i, n in enumerate(names):
        row = {"index": i, "name": n, "lower": float(lim[i, 0]), "upper": float(lim[i, 1])}
        row.update({k: float(v[i]) for k, v in per_joint.items()})
        row.update({k: float(v[i]) for k, v in physx_gains.items()})
        row["usd"] = usd_joints.get(n)
        joints.append(row)

    masses = t(view.get_masses())[0]
    inertias = t(view.get_inertias())[0].reshape(-1, 3, 3)
    coms = t(view.get_coms())[0]  # pos xyz + quat xyzw, body frame
    shapes = {b: [] for b in bodies}
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        p = str(prim.GetPath())
        if not p.startswith("/World/Robot") or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        owner = prim.GetParent()  # nearest rigid-body ancestor (mesh prims may share names)
        while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        owner = owner.GetName() if owner else None
        approx = (
            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            else None
        )
        if owner in shapes:
            shapes[owner].append({"type": prim.GetTypeName(), "approximation": approx})
    body_rows = [
        {
            "name": b,
            "mass": float(masses[i]),
            "inertia_body": inertias[i].tolist(),
            "com_pos_body": coms[i, :3].tolist(),
            "com_quat_xyzw_body": coms[i, 3:].tolist(),
            "collision_shapes": shapes[b],
        }
        for i, b in enumerate(bodies)
    ]

    # Forward kinematics at the zero pose and seeded random poses inside the limits.
    rng = np.random.default_rng(20260928)
    poses = [np.zeros(len(names))]
    for _ in range(args.fk_poses):
        poses.append(rng.uniform(lim[:, 0], lim[:, 1]))
    psv = robot.data._physics_sim_view  # noqa: SLF001 - FK without stepping
    fk = []
    for q in poses:
        qt = torch.tensor(q, dtype=torch.float32, device=sim.device)[None]
        robot.write_joint_position_to_sim_index(position=qt)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(qt))
        psv.update_articulations_kinematic()
        tf = t(view.get_link_transforms())[0]
        fk.append(
            {
                "joint_pos_by_name": {n: float(v) for n, v in zip(names, q, strict=True)},
                "links": {b: tf[i].tolist() for i, b in enumerate(bodies)},  # xyz + qxyzw
            }
        )

    record = {
        "usd": args.usd,
        "isaac_sim_version": Path("/isaac-sim/VERSION").read_text().strip(),
        "fixed_base": bool(robot.is_fixed_base),
        "num_joints": len(names),
        "num_bodies": len(bodies),
        "joint_order": names,
        "body_order": bodies,
        "joints": joints,
        "other_usd_joints": usd_joints.get("_other_joints", []),
        "bodies": body_rows,
        "fk": fk,
        "root_pose_w": t(robot.data.root_link_pose_w)[0].tolist(),
        "notes": "limits in rad; link transforms are actor frames, xyz + quat xyzw, world",
    }
    (args.output / "isaac_model.json").write_text(json.dumps(record, indent=2))
    summary = {"num_joints": len(names), "bodies": len(bodies)}
    print("AUDIT_RESULT " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
