"""Compare the MuJoCo model that Isaac Lab's Newton backend compiled with the host MuJoCo model.

Development, TASK-025 (Newton spike). Run on the host after an Isaac run made with
``--physics newton``:

    uv run --no-sync python scripts/isaac/audit_newton_model.py \\
        --isaac outputs/<isaac-run>/run --output outputs/<isaac-run>/newton_audit

``IsaacTransport(physics="newton")`` records ``newton_model``: MuJoCo-Warp's solver options and,
per geom, joint and body, what the compiled model holds (read back from ``mjw_model`` after
``sim.reset``). This script compares that record with ``MuJoCoSimulation`` switched to v2
(``apply_v2_scene``): solver options, per-role geom contact parameters (friction triple,
condim, solref, solimp, margin), collision filtering, per-joint damping / armature /
frictionloss / range / limit solref and solimp, and body masses and inertias. It is a
parameter audit, not a behaviour check, and nothing here is a learned or task result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from embodied_jepa.apple_to_plate_v2 import apply_v2_scene
from embodied_jepa.simulation import MuJoCoSimulation

TOL = 1e-6
RTOL = 1e-6  # MuJoCo-Warp holds the model in float32


def _close(a, b) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=RTOL, atol=1e-9))


OPT_KEYS = (
    "timestep",
    "integrator",
    "cone",
    "solver",
    "iterations",
    "ls_iterations",
    "tolerance",
    "ls_tolerance",
    "impratio",
)


def _self_collides(masks) -> bool:
    return any((a[0] & b[1]) or (b[0] & a[1]) for i, a in enumerate(masks) for b in masks[i + 1 :])


def host_role(model, g: int) -> str:
    name = model.geom(g).name
    body = model.body(int(model.geom_bodyid[g])).name
    if name in ("floor", "table"):
        return name
    if body == "apple":
        return "apple"
    if name == "plate_base":
        return "plate_base"
    if name.startswith("plate_rim"):
        return "plate_rim"
    if body == "world":
        return "world_other"
    return "robot"


def _unique_rows(rows) -> list[list[float]]:
    if not rows:
        return []
    arr = np.round(np.asarray(rows, dtype=float), 6)
    return np.unique(arr, axis=0).tolist()


def compare_geoms(model, newton: dict) -> dict:
    out = {}
    roles = ("floor", "table", "apple", "plate_base", "plate_rim", "robot")
    for role in roles:
        host = [
            g
            for g in range(model.ngeom)
            if host_role(model, g) == role and (model.geom_contype[g] or model.geom_conaffinity[g])
        ]
        nt = [
            g for g in newton["geoms"] if g["role"] == role and (g["contype"] or g["conaffinity"])
        ]
        row = {"colliding_geoms": {"mujoco": len(host), "newton": len(nt)}}
        for key, width in (
            ("friction", 3),
            ("condim", 1),
            ("solref", 2),
            ("solimp", 5),
            ("margin", 1),
        ):
            h = _unique_rows(
                [np.atleast_1d(getattr(model, f"geom_{key}")[g])[:width] for g in host]
            )
            n = _unique_rows([np.atleast_1d(g[key])[:width] for g in nt])
            row[key] = {"mujoco": h, "newton": n, "match": _close(h, n)}
        h_types = sorted(int(model.geom_type[g]) for g in host)
        n_types = sorted(g["type"] for g in nt)
        row["types"] = {"mujoco": h_types, "newton": n_types, "match": h_types == n_types}
        row["gap_newton"] = sorted({g["gap"] for g in nt})
        out[role] = row
    return out


def compare_joints(sim, newton: dict) -> dict:
    model = sim.model
    rows, worst = {}, {}
    for name, jid in zip(sim.joint_names, sim.joint_ids, strict=True):
        jid, dof = int(jid), int(model.jnt_dofadr[int(jid)])
        nt = newton["joints"][name]
        host = {
            "damping": float(model.dof_damping[dof]),
            "armature": float(model.dof_armature[dof]),
            "frictionloss": float(model.dof_frictionloss[dof]),
            "range": model.jnt_range[jid].tolist(),
            "limited": bool(model.jnt_limited[jid]),
            "solref_limit": model.jnt_solref[jid].tolist(),
            "solimp_limit": model.jnt_solimp[jid].tolist(),
        }
        diff = {}
        for key, h in host.items():
            if key == "limited":
                diff[key] = 0.0 if h == nt[key] else 1.0
            else:
                diff[key] = float(np.abs(np.asarray(h) - np.asarray(nt[key])).max())
            worst[key] = max(worst.get(key, 0.0), diff[key])
        rows[name] = {"mujoco": host, "newton": nt, "abs_diff": diff}
    match = {k: v <= TOL * max(1.0, _scale(rows, k)) for k, v in worst.items()}
    return {"max_abs_diff": worst, "match": match, "rows": rows}


def _scale(rows: dict, key: str) -> float:
    """Largest magnitude of a joint quantity (joint ranges reach ~3 rad; float32 rounding)."""
    return max(
        float(np.abs(np.asarray(r["mujoco"][key], dtype=float)).max()) for r in rows.values()
    )


def compare_bodies(model, newton: dict) -> dict:
    rows, mass_diff, inertia_rel = {}, 0.0, 0.0
    for label, nt in newton["bodies"].items():
        if label in ("world", "table", "floor"):
            continue
        try:
            b = model.body(label)
        except KeyError:
            rows[label] = {"newton": nt, "mujoco": None}
            continue
        h_mass, h_inertia = float(model.body_mass[b.id]), model.body_inertia[b.id]
        dm = abs(h_mass - nt["mass"])
        # Principal inertias; the order of the principal axes can differ between compilers.
        di = float(
            np.abs(np.sort(h_inertia) - np.sort(nt["inertia"])).max() / max(h_inertia.max(), 1e-12)
        )
        mass_diff, inertia_rel = max(mass_diff, dm), max(inertia_rel, di)
        rows[label] = {
            "mujoco": {"mass": h_mass, "inertia": h_inertia.tolist()},
            "newton": nt,
            "mass_abs_diff": dm,
            "inertia_rel_diff": di,
        }
    return {"max_mass_abs_diff_kg": mass_diff, "max_inertia_rel_diff": inertia_rel, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", type=Path, required=True, help="run dir with the record")
    parser.add_argument("--record", default=None, help="record file name inside --isaac")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    names = [args.record] if args.record else ["isaac_scripted.json", "isaac_parity.json"]
    path = next((args.isaac / n for n in names if (args.isaac / n).exists()), None)
    if path is None:
        raise SystemExit(f"no run record in {args.isaac}")
    record = json.loads(path.read_text())
    if record.get("physics_backend") != "newton" or not record.get("newton_model"):
        raise SystemExit("this run was not made with the Newton backend")
    newton = record["newton_model"]
    sim = MuJoCoSimulation(render=False)
    apply_v2_scene(sim.model)
    model = sim.model
    opt = {}
    for key in OPT_KEYS:
        h = getattr(model.opt, key)
        n = newton["opt"].get(key)
        opt[key] = {
            "mujoco": float(h),
            "newton": n,
            "match": n is not None and _close(float(h), float(n)),
        }
    grav = np.asarray(newton["opt"]["gravity"], dtype=float)
    opt["gravity"] = {
        "mujoco": model.opt.gravity.tolist(),
        "newton": grav.tolist(),
        "match": bool(np.allclose(model.opt.gravity, grav)),
    }
    report = {
        "question": "Does the MuJoCo model compiled by Isaac Lab's Newton backend hold the host "
        "MuJoCo v2 model's contact, joint and body parameters? (parameter audit only)",
        "isaac_record": str(path),
        "versions": {
            "host_mujoco": __import__("mujoco").__version__,
            "newton": newton.get("newton_version"),
            "mujoco_warp": newton.get("mujoco_warp_version"),
            "container_mujoco": newton.get("mujoco_version"),
        },
        "sizes": {
            "mujoco": {"nv": model.nv, "nu": model.nu, "nexclude": model.nexclude},
            "newton": {k: newton[k] for k in ("nv", "nu", "nexclude", "npair")},
        },
        "opt": opt,
        "robot_self_collision_by_bitmask": {
            "mujoco": _self_collides(
                [
                    (int(model.geom_contype[g]), int(model.geom_conaffinity[g]))
                    for g in range(model.ngeom)
                    if host_role(model, g) == "robot"
                    and (model.geom_contype[g] or model.geom_conaffinity[g])
                ]
            ),
            "newton": _self_collides(
                [
                    (g["contype"], g["conaffinity"])
                    for g in newton["geoms"]
                    if g["role"] == "robot" and (g["contype"] or g["conaffinity"])
                ]
            ),
            "note": "whether any two colliding robot geoms pass the contype/conaffinity test "
            "(MuJoCo additionally skips parent-child pairs and explicit excludes)",
        },
        "geoms": compare_geoms(model, newton),
        "joints": compare_joints(sim, newton),
        "bodies": compare_bodies(model, newton),
    }
    sim.close()
    args.output.mkdir(parents=True)
    (args.output / "newton_audit.json").write_text(json.dumps(report, indent=2))
    print("opt:", {k: v["match"] for k, v in opt.items()})
    for role, row in report["geoms"].items():
        print(
            f"geoms {role:10s}",
            row["colliding_geoms"],
            {k: row[k]["match"] for k in ("friction", "condim", "solref", "solimp", "margin")},
        )
    print("joints max abs diff:", report["joints"]["max_abs_diff"])
    b = report["bodies"]
    print("bodies:", b["max_mass_abs_diff_kg"], b["max_inertia_rel_diff"])


if __name__ == "__main__":
    main()
