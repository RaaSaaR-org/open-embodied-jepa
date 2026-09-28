"""Compare an Isaac Sim G1+Dex3 dump with the project's MuJoCo embodiment.

Development helper (not a gated experiment). Run on the host with the project environment.
Two inputs are supported:

* ``--isaac`` — the joint dump of ``bringup_g1_dex3.py`` (names, counts and limits only), as
  used by the bring-up spike on NVIDIA's USD:

      uv run --no-sync python scripts/isaac/compare_joints.py \\
          --isaac outputs/isaac-bringup-spike-<n>/run/isaac_joints.json \\
          --output outputs/isaac-bringup-spike-<n>/joint_comparison.json

* ``--isaac-model`` — the full dump of ``audit_usd.py`` on a USD converted from the pinned
  MJCF. It additionally compares joint axes, joint frames (child-in-parent transforms), body
  masses, centres of mass and inertia tensors, armature, passive damping, friction, effort
  limits, drive gains, collision-shape counts and forward kinematics at seeded poses:

      uv run --no-sync python scripts/isaac/compare_joints.py \\
          --isaac-model outputs/<audit-run>/run/isaac_model.json \\
          --output outputs/<audit-run>/model_comparison.json

The MuJoCo reference is ``MuJoCoSimulation`` (the pinned MJCF with the pelvis free joint
removed, i.e. the model every project experiment uses). Every mismatch is listed; nothing is
tuned to make the two agree.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def group(name: str) -> str:
    if "_hand_" in name:
        return "dex3"
    if any(k in name for k in ("shoulder", "elbow", "wrist")):
        return "arm"
    if name.startswith("waist"):
        return "waist"
    return "leg"


def mujoco_sim():
    from embodied_jepa.simulation import MuJoCoSimulation

    return MuJoCoSimulation(render=False)


def mujoco_joints(sim) -> dict[str, dict]:
    model = sim.model
    out = {}
    for order, (name, jid) in enumerate(zip(sim.joint_names, sim.joint_ids, strict=True)):
        lo, hi = (float(v) for v in model.jnt_range[int(jid)])
        out[name] = {"order": order, "lower": lo, "upper": hi}
    return out


# ---------------------------------------------------------------- rotation helpers (wxyz)
def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_angle(a: np.ndarray, b: np.ndarray) -> float:
    c = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(math.acos(max(-1.0, min(1.0, c))))


def xyzw_to_wxyz(q) -> list[float]:
    return [q[3], q[0], q[1], q[2]]


AXES = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}


# ---------------------------------------------------------------- legacy joint-limit mode
def compare_limits(isaac: dict, mj: dict, tol: float) -> dict:
    only_mj = sorted(set(mj) - set(isaac))
    only_isaac = sorted(set(isaac) - set(mj))
    rows = []
    for name in sorted(set(mj) & set(isaac), key=lambda n: mj[n]["order"]):
        a, b = mj[name], isaac[name]
        dl, du = b["lower"] - a["lower"], b["upper"] - a["upper"]
        rows.append(
            {
                "name": name,
                "group": group(name),
                "mujoco": [a["lower"], a["upper"]],
                "isaac": [b["lower"], b["upper"]],
                "limit_match": abs(dl) <= tol and abs(du) <= tol,
                "max_abs_limit_diff_rad": max(abs(dl), abs(du)),
            }
        )
    counts = {
        g: {"mujoco": sum(group(n) == g for n in mj), "isaac": sum(group(n) == g for n in isaac)}
        for g in ("leg", "waist", "arm", "dex3")
    }
    mismatched = [r for r in rows if not r["limit_match"]]
    assert all(math.isfinite(r["max_abs_limit_diff_rad"]) for r in rows)
    return {
        "mujoco_joint_count": len(mj),
        "isaac_joint_count": len(isaac),
        "counts_by_group": counts,
        "only_in_mujoco": only_mj,
        "only_in_isaac": only_isaac,
        "common": len(rows),
        "limit_mismatches": [
            {k: r[k] for k in ("name", "group", "mujoco", "isaac", "max_abs_limit_diff_rad")}
            for r in mismatched
        ],
        "arm_dex3_all_names_and_limits_match": not any(
            group(n) in ("arm", "dex3") for n in only_mj + only_isaac
        )
        and not any(r["group"] in ("arm", "dex3") for r in mismatched),
        "tolerance_rad": tol,
        "rows": rows,
    }


# ---------------------------------------------------------------- full model mode
TOL = {
    "limit_rad": 1e-4,  # USD stores degrees as float32
    "axis_rad": 1e-4,
    "pos_m": 1e-5,
    "rot_rad": 1e-4,
    "mass_rel": 1e-5,
    "inertia_rel": 1e-4,
    "param_abs": 1e-6,
    "fk_pos_m": 1e-4,
    "fk_rot_rad": 1e-3,
}

PARAM_NOTES = {
    "friction": "MuJoCo frictionloss [N m] vs PhysX joint friction coefficient [-] after "
    "Isaac Lab actuator init; different semantics",
    "usd_joint_friction_attr": "MuJoCo frictionloss [N m] vs USD physxJoint:jointFriction "
    "(a PhysX coefficient [-]); the number is copied, the semantics differ",
    "passive_damping": "MuJoCo dof_damping [N m s/rad] vs PhysX drive damping",
    "effort_limit": "MuJoCo actuator ctrlrange [N m] vs PhysX max joint force",
    "position_gain_in_model": "MuJoCo motor (no servo) vs USD drive stiffness",
    "velocity_gain_in_model": "MuJoCo motor (no servo) vs USD drive damping",
}


class Mismatches(list):
    def flag(self, kind, name, mujoco, isaac, detail=""):
        self.append(
            {"kind": kind, "name": name, "mujoco": mujoco, "isaac": isaac, "detail": detail}
        )


def compare_joint(model, name: str, ij: dict, ctrl: dict, out: Mismatches) -> dict:
    jid = model.joint(name).id
    dof = int(model.jnt_dofadr[jid])
    body = int(model.jnt_bodyid[jid])
    usd = ij["usd"]
    row = {"name": name, "group": group(name)}
    lo, hi = (float(v) for v in model.jnt_range[jid])
    dlim = max(abs(ij["lower"] - lo), abs(ij["upper"] - hi))
    row["limits"] = {"mujoco": [lo, hi], "isaac": [ij["lower"], ij["upper"]], "diff": dlim}
    if dlim > TOL["limit_rad"]:
        out.flag("joint_limit", name, [lo, hi], [ij["lower"], ij["upper"]], f"max {dlim:.3g}")
    # Joint axis in the child-body frame.
    mj_axis = np.asarray(model.jnt_axis[jid], dtype=float)
    is_axis = quat_to_mat(usd["local_rot1_wxyz"]) @ AXES[usd["axis"]]
    ang = float(math.acos(max(-1.0, min(1.0, float(mj_axis @ is_axis)))))
    row["axis"] = {"mujoco": mj_axis.tolist(), "isaac": is_axis.tolist(), "angle_rad": ang}
    if ang > TOL["axis_rad"]:
        out.flag("joint_axis", name, mj_axis.tolist(), is_axis.tolist(), f"{ang:.3g} rad")
    # Joint anchor in the child frame.
    danchor = float(np.linalg.norm(np.asarray(usd["local_pos1"]) - model.jnt_pos[jid]))
    row["anchor_diff_m"] = danchor
    if danchor > TOL["pos_m"]:
        out.flag("joint_anchor", name, model.jnt_pos[jid].tolist(), usd["local_pos1"])
    # Child-in-parent transform T0 * inv(T1) against MuJoCo body_pos/body_quat.
    r0, r1 = quat_to_mat(usd["local_rot0_wxyz"]), quat_to_mat(usd["local_rot1_wxyz"])
    r_pc = r0 @ r1.T
    p_pc = np.asarray(usd["local_pos0"]) - r_pc @ np.asarray(usd["local_pos1"])
    dpos = float(np.linalg.norm(p_pc - model.body_pos[body]))
    drot = rot_angle(r_pc, quat_to_mat(model.body_quat[body]))
    row["child_in_parent"] = {"pos_diff_m": dpos, "rot_diff_rad": drot}
    parent = model.body(int(model.body_parentid[body])).name
    if [usd["body0"], usd["body1"]] != [parent, model.body(body).name]:
        child = model.body(body).name
        out.flag("joint_bodies", name, [parent, child], [usd["body0"], usd["body1"]])
    if dpos > TOL["pos_m"] or drot > TOL["rot_rad"]:
        out.flag("joint_frame", name, "", "", f"pos {dpos:.3g} m, rot {drot:.3g} rad")
    params = {
        "armature": (float(model.dof_armature[dof]), ij["armature"]),
        "passive_damping": (float(model.dof_damping[dof]), ij["physx_damping_after_init"]),
        "friction": (float(model.dof_frictionloss[dof]), ij["friction_coeff"]),
        "usd_joint_friction_attr": (
            float(model.dof_frictionloss[dof]),
            usd["usd_physx_joint_friction"],
        ),
        "effort_limit": (float(max(abs(v) for v in ctrl[name])), ij["effort_limit"]),
        "position_gain_in_model": (0.0, usd["usd_drive_stiffness"]),
        "velocity_gain_in_model": (0.0, usd["usd_drive_damping"]),
    }
    row["params"] = {k: {"mujoco": a, "isaac": b} for k, (a, b) in params.items()}
    for k, (a, b) in params.items():
        if b is None or not math.isfinite(b) or abs(a - b) > TOL["param_abs"] * max(1, abs(a)):
            out.flag(f"joint_{k}", name, a, b, PARAM_NOTES.get(k, ""))
    return row


def compare_body(model, name: str, ib: dict, out: Mismatches) -> dict:
    bid = model.body(name).id
    m_mj, m_is = float(model.body_mass[bid]), ib["mass"]
    r = quat_to_mat(model.body_iquat[bid])
    i_mj = r @ np.diag(model.body_inertia[bid]) @ r.T
    i_is = np.asarray(ib["inertia_body"])
    # Isaac reports the tensor about the COM in the body frame; compare the full tensor and
    # the principal moments (orientation-free).
    full_rel = float(np.linalg.norm(i_is - i_mj) / np.linalg.norm(i_mj))
    principal = np.sort(np.linalg.eigvalsh(i_is)) - np.sort(model.body_inertia[bid])
    eig_rel = float(np.max(np.abs(principal)) / np.max(model.body_inertia[bid]))
    dcom = float(np.linalg.norm(np.asarray(ib["com_pos_body"]) - model.body_ipos[bid]))
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
    mj_coll = sum(1 for g in geoms if model.geom_contype[g] or model.geom_conaffinity[g])
    is_coll = len(ib["collision_shapes"])
    if abs(m_is - m_mj) > TOL["mass_rel"] * max(m_mj, 1e-9):
        out.flag("body_mass", name, m_mj, m_is)
    if dcom > TOL["pos_m"]:
        out.flag("body_com", name, model.body_ipos[bid].tolist(), ib["com_pos_body"])
    if full_rel > TOL["inertia_rel"]:
        out.flag("body_inertia", name, "", "", f"full rel {full_rel:.3g}, principal {eig_rel:.3g}")
    if mj_coll != is_coll:
        out.flag("body_collision_shape_count", name, mj_coll, is_coll)
    return {
        "name": name,
        "mass": {"mujoco": m_mj, "isaac": m_is},
        "com_diff_m": dcom,
        "inertia_full_rel_diff": full_rel,
        "inertia_principal_rel_diff": eig_rel,
        "collision_shapes": {
            "mujoco": mj_coll,
            "isaac": is_coll,
            "isaac_detail": ib["collision_shapes"],
        },
    }


def compare_fk(sim, dump: dict, robot_bodies: list[str], out: Mismatches) -> list[dict]:
    model, data = sim.model, sim.data
    qadr = {n: int(model.jnt_qposadr[model.joint(n).id]) for n in sim.joint_names}
    rows = []
    for k, pose in enumerate(dump["fk"]):
        data.qpos[:] = model.qpos0
        for n, v in pose["joint_pos_by_name"].items():
            data.qpos[qadr[n]] = v
        sim.mj.mj_kinematics(model, data)
        worst_p, worst_r, worst_b = 0.0, 0.0, None
        for b, tf in pose["links"].items():
            if b not in robot_bodies:
                continue
            bid = model.body(b).id
            dp = float(np.linalg.norm(np.asarray(tf[:3]) - data.xpos[bid]))
            dr = rot_angle(quat_to_mat(xyzw_to_wxyz(tf[3:])), data.xmat[bid].reshape(3, 3))
            if dp > worst_p:
                worst_p, worst_b = dp, b
            worst_r = max(worst_r, dr)
        rows.append(
            {
                "pose": k,
                "max_pos_diff_m": worst_p,
                "worst_body": worst_b,
                "max_rot_diff_rad": worst_r,
            }
        )
        if worst_p > TOL["fk_pos_m"] or worst_r > TOL["fk_rot_rad"]:
            detail = f"pos {worst_p:.3g} m, rot {worst_r:.3g} rad"
            out.flag("forward_kinematics", f"pose {k}", "", "", detail)
    sim.reset()
    return rows


def compare_model(dump: dict, sim) -> dict:
    model = sim.model
    names = list(sim.joint_names)
    isaac_joints = {j["name"]: j for j in dump["joints"]}
    isaac_bodies = {b["name"]: b for b in dump["bodies"]}
    out = Mismatches()
    for n in sorted(set(names) - set(isaac_joints)):
        out.flag("joint_missing_in_isaac", n, "present", "absent")
    for n in sorted(set(isaac_joints) - set(names)):
        out.flag("joint_missing_in_mujoco", n, "absent", "present")
    ctrl = {
        model.joint(int(model.actuator_trnid[a, 0])).name: model.actuator_ctrlrange[a]
        for a in range(model.nu)
    }
    joint_rows = [
        compare_joint(model, n, isaac_joints[n], ctrl, out) for n in names if n in isaac_joints
    ]
    robot_bodies = [
        model.body(i).name
        for i in range(1, model.nbody)
        if model.body(i).name not in ("apple", "plate")
    ]
    for n in sorted(set(robot_bodies) - set(isaac_bodies)):
        out.flag("body_missing_in_isaac", n, "present", "absent")
    for n in sorted(set(isaac_bodies) - set(robot_bodies)):
        out.flag("body_missing_in_mujoco", n, "absent", "present")
    body_rows = [
        compare_body(model, n, isaac_bodies[n], out) for n in robot_bodies if n in isaac_bodies
    ]
    fk_rows = compare_fk(sim, dump, robot_bodies, out)
    out.flag(
        "self_collision",
        "articulation",
        "collision geoms collide except parent-child pairs (contype/conaffinity 1)",
        "disabled (importer allow_self_collision=False)",
    )
    kinds: dict[str, int] = {}
    for m in out:
        kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
    return {
        "usd": dump["usd"],
        "fixed_base": dump["fixed_base"],
        "mujoco_joint_count": len(names),
        "isaac_joint_count": dump["num_joints"],
        "joint_order_equal": dump["joint_order"] == names,
        "mujoco_body_count": len(robot_bodies),
        "isaac_body_count": dump["num_bodies"],
        "total_mass": {
            "mujoco": float(sum(model.body_mass[model.body(b).id] for b in robot_bodies)),
            "isaac": float(sum(b["mass"] for b in dump["bodies"])),
        },
        "tolerances": TOL,
        "mismatch_counts_by_kind": kinds,
        "mismatches": list(out),
        "fk": fk_rows,
        "joints": joint_rows,
        "bodies": body_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--isaac", type=Path, help="bringup_g1_dex3.py isaac_joints.json")
    src.add_argument("--isaac-model", type=Path, help="audit_usd.py isaac_model.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tol", type=float, default=1e-3, help="limit tolerance [rad] (--isaac)")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    sim = mujoco_sim()
    if args.isaac:
        isaac = {j["name"]: j for j in json.loads(args.isaac.read_text())["joints"]}
        summary = compare_limits(isaac, mujoco_joints(sim), args.tol)
        printable = {k: v for k, v in summary.items() if k != "rows"}
        lines = [
            f"  {m['name']:32s} mj=[{m['mujoco'][0]:+.4f},{m['mujoco'][1]:+.4f}] "
            f"isaac=[{m['isaac'][0]:+.4f},{m['isaac'][1]:+.4f}]"
            for m in summary["limit_mismatches"]
        ]
    else:
        summary = compare_model(json.loads(args.isaac_model.read_text()), sim)
        skip = ("joints", "bodies", "mismatches")
        printable = {k: v for k, v in summary.items() if k not in skip}
        lines = [
            f"  {m['kind']:30s} {m['name']:28s} mj={m['mujoco']!s:.40} "
            f"isaac={m['isaac']!s:.40} {m['detail']}"
            for m in summary["mismatches"]
        ]
    sim.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(printable, indent=2, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
