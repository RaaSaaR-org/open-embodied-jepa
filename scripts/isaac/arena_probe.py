"""Bring up the GR00T tutorial's Arena scene with a local hold policy and record its facts.

Development spike (docs/ARENA.md); container-only, via
``scripts/isaac/run_isaac.sh arena_probe.py outputs/<new> [--hold_steps N]``. No GR00T server
is started or contacted. Writes into ``--output``:

- ``facts.json``: sim/control rates, action space, robot (base, joints, limits, gains), cameras
  (prim, parent, offset, resolution, intrinsics, world pose), apple and plate (USD bounds,
  PhysX mass/inertia/material, USD materials and shader inputs), shelf, reset events,
  termination terms;
- ``frames/<camera>_<tag>.png`` for every scene camera (the env's rgb_array viewer renders black
  when headless, so it is not saved);
- ``episodes.json``: episode A (hold until the episode ends), episode B (Arena's own
  success-path check, but dropped from 1 cm above the plate's top instead of Arena's test offset,
  which interpenetrates: hold 50 steps, teleport, hold until a term fires), and the Arena metrics
  over both.

Scripted and teleported behaviour only; nothing here is a learned or policy result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, "/oej/src")
from embodied_jepa import arena_transport as at  # noqa: E402

PXR_PIN = at.pin_pxr_work_thread_limit()  # before any Kit import (PR #112)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float | int | str | bool) or x is None:
        return x
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
    except ImportError:
        pass
    return repr(x)


def section(facts: dict, name: str, fn):
    t0 = time.monotonic()
    try:
        facts[name] = fn()
    except Exception as exc:  # noqa: BLE001 - diagnostic probe: record and continue
        facts[name] = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
    print(f"[arena_probe] section {name} ({time.monotonic() - t0:.1f}s)", flush=True)


def save_png(path, img):
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(img)).save(path)


ATTR_PREFIXES = ("physics:", "physx", "xformOp:scale", "primvars:displayColor")


def usd_facts(prim_root: str, max_prims: int = 60) -> dict:
    """Bounds, physics attributes and bound materials (with shader inputs) under a prim."""
    import omni.usd
    from pxr import Usd, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_root)
    if not root.IsValid():
        return {"error": f"no prim {prim_root}"}
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, "render"])
    box = cache.ComputeWorldBound(root).ComputeAlignedRange()
    out = {
        "world_aabb_min": list(box.GetMin()),
        "world_aabb_max": list(box.GetMax()),
        "world_aabb_size": list(box.GetSize()),
        "prims": {},
        "materials": {},
    }
    for i, prim in enumerate(Usd.PrimRange(root)):
        if i >= max_prims:
            out["truncated_at"] = max_prims
            break
        attrs = {
            a.GetName(): jsonable(a.Get())
            for a in prim.GetAttributes()
            if a.GetName().startswith(
                ("physics:", "physx", "xformOp:scale", "primvars:displayColor")
            )
            and a.Get() is not None
        }
        info = {"type": prim.GetTypeName(), "apis": [str(s) for s in prim.GetAppliedSchemas()]}
        if attrs:
            info["attrs"] = attrs
        if prim.IsA(UsdGeom.Gprim):
            mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if mat:
                mpath = str(mat.GetPath())
                info["material"] = mpath
                if mpath not in out["materials"]:
                    shaders = {}
                    for sp in Usd.PrimRange(mat.GetPrim()):
                        sh = UsdShade.Shader(sp)
                        if not sh:
                            continue
                        ins = {}
                        for inp in sh.GetInputs():
                            v = inp.Get()
                            if v is not None:
                                ins[inp.GetBaseName()] = jsonable(
                                    v.path if hasattr(v, "path") else v
                                )
                        src = sp.GetAttribute("info:mdl:sourceAsset")
                        shaders[str(sp.GetPath())] = {
                            "id": jsonable(sh.GetIdAttr().Get()),
                            "mdl": jsonable(src.Get().path if src and src.Get() else None),
                            "inputs": ins,
                        }
                    out["materials"][mpath] = shaders
        out["prims"][str(prim.GetPath())] = info
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--hold_steps", type=int, default=320)
    p.add_argument("--settle_steps", type=int, default=50)
    p.add_argument("--success_steps", type=int, default=100)
    p.add_argument("--drop_clearance", type=float, default=0.01)
    own, rest = p.parse_known_args()
    os.makedirs(os.path.join(own.output, "frames"), exist_ok=True)
    out = own.output
    t_start = time.monotonic()

    scene = at.ArenaScene(rest)
    u = scene.unwrapped
    print(f"[arena_probe] scene built after {time.monotonic() - t_start:.1f}s", flush=True)
    facts: dict = {"arena_ref": at.ARENA_REF, "tutorial": at.TUTORIAL, "argv": sys.argv}
    try:
        from pxr import Work

        facts["pxr_work"] = {"pinned": PXR_PIN, "concurrency_limit": Work.GetConcurrencyLimit()}
    except Exception as exc:  # noqa: BLE001
        facts["pxr_work"] = {"pinned": PXR_PIN, "error": repr(exc)}

    def versions():
        import importlib.metadata as md

        import omni.kit.app

        v = {"isaac_sim_kit": omni.kit.app.get_app().get_app_version()}
        for pkg in ("isaaclab", "isaaclab_arena", "torch", "warp-lang"):
            try:
                v[pkg] = md.version(pkg)
            except Exception as exc:  # noqa: BLE001
                v[pkg] = f"? {exc}"
        return v

    section(facts, "versions", versions)

    def timing():
        return {
            "sim_dt_s": float(u.cfg.sim.dt),
            "decimation": int(u.cfg.decimation),
            "render_interval": int(u.cfg.sim.render_interval),
            "step_dt_s": float(u.step_dt),
            "control_hz": 1.0 / float(u.step_dt),
            "episode_length_s": float(u.cfg.episode_length_s),
            "max_episode_length_steps": int(u.max_episode_length),
            "num_rerenders_on_reset": getattr(u.cfg, "num_rerenders_on_reset", None),
            "physics_device": str(u.device),
            "physx": jsonable(
                {
                    k: getattr(u.cfg.sim.physx, k)
                    for k in dir(u.cfg.sim.physx)
                    if not k.startswith("_") and not callable(getattr(u.cfg.sim.physx, k))
                }
            )
            if hasattr(u.cfg.sim, "physx")
            else None,
        }

    section(facts, "timing", timing)

    def action_space():
        am = u.action_manager
        terms = {}
        for name in am.active_terms:
            t = am.get_term(name)
            terms[name] = {
                "class": type(t).__name__,
                "action_dim": int(t.action_dim),
                "wbc_version": getattr(t.cfg, "wbc_version", None),
                "joint_names": list(getattr(t, "_joint_names", [])),
                "upper_body_wbc_indices": jsonable(
                    t.robot_model.get_joint_group_indices("upper_body")
                )
                if hasattr(t, "robot_model")
                else None,
                "lower_body_wbc_indices": jsonable(
                    t.robot_model.get_joint_group_indices("lower_body")
                )
                if hasattr(t, "robot_model")
                else None,
                "wbc_joint_order": getattr(t, "wbc_g1_joints_order", None),
            }
        return {
            "single_action_space_shape": list(u.single_action_space.shape),
            "total_action_dim": int(am.total_action_dim),
            "terms": terms,
        }

    section(facts, "action_space", action_space)

    def observation_space():
        out = {}
        for group, space in u.single_observation_space.spaces.items():
            if hasattr(space, "spaces"):
                out[group] = {k: list(v.shape) for k, v in space.spaces.items()}
            else:
                out[group] = list(space.shape)
        return out

    section(facts, "observation_space", observation_space)
    scene.reset(seed=42)

    def robot():
        r = scene.robot
        names = scene.joint_names
        limits = scene.np(r.data.joint_pos_limits)[0]
        d = {
            "num_joints": len(names),
            "joint_names_sim_order": names,
            "is_fixed_base": bool(r.is_fixed_base),
            "usd_path": str(r.cfg.spawn.usd_path),
            "init_state_pos": list(r.cfg.init_state.pos),
            "init_state_rot_xyzw": list(r.cfg.init_state.rot),
            "default_joint_pos": dict(
                zip(names, scene.np(r.data.default_joint_pos)[0].tolist(), strict=True)
            ),
            "joint_pos_limits": {n: limits[i].tolist() for i, n in enumerate(names)},
            "root_pos_local_after_reset": scene.truth()["pelvis_pos_local"],
            "body_names": list(r.data.body_names),
            "actuators": {
                k: {
                    "class": type(a).__name__,
                    "joint_names": list(a.joint_names),
                    "stiffness": scene.np(a.stiffness)[0].tolist(),
                    "damping": scene.np(a.damping)[0].tolist(),
                    "effort_limit": scene.np(a.effort_limit)[0].tolist(),
                }
                for k, a in r.actuators.items()
            },
        }
        hand = [n for n in names if "hand" in n]
        d["hand_joints"] = hand
        return d

    section(facts, "robot", robot)

    def body_pose(name):
        r = scene.robot
        i = list(r.data.body_names).index(name)
        origin = scene.np(u.scene.env_origins)[0]
        return {
            "pos_local": (scene.np(r.data.body_pos_w)[0][i] - origin).tolist(),
            "quat_w_xyzw": scene.np(r.data.body_quat_w)[0][i].tolist(),
        }

    def cameras():
        out = {}
        for name, cam in scene.cameras().items():
            cfg = cam.cfg
            origin = scene.np(u.scene.env_origins)[0]
            spawn = cfg.spawn
            k = scene.np(cam.data.intrinsic_matrices)[0]
            out[name] = {
                "class": type(cam).__name__,
                "prim_path": cfg.prim_path,
                "height": int(cfg.height),
                "width": int(cfg.width),
                "data_types": list(cfg.data_types),
                "update_period": cfg.update_period,
                "focal_length": getattr(spawn, "focal_length", None),
                "horizontal_aperture": getattr(spawn, "horizontal_aperture", None),
                "vertical_aperture": getattr(spawn, "vertical_aperture", None),
                "clipping_range": list(getattr(spawn, "clipping_range", []) or []),
                "hfov_deg": at.horizontal_fov_deg(spawn.focal_length, spawn.horizontal_aperture)
                if getattr(spawn, "horizontal_aperture", None)
                else None,
                "vfov_deg_from_K": float(np.degrees(2 * np.arctan(0.5 * cfg.height / k[1, 1]))),
                "hfov_deg_from_K": float(np.degrees(2 * np.arctan(0.5 * cfg.width / k[0, 0]))),
                "offset_pos": list(cfg.offset.pos),
                "offset_rot_xyzw": list(cfg.offset.rot),
                "offset_convention": cfg.offset.convention,
                "intrinsics": k.tolist(),
                "pos_local": (scene.np(cam.data.pos_w)[0] - origin).tolist(),
                "quat_w_ros_xyzw": scene.np(cam.data.quat_w_ros)[0].tolist(),
            }
        for b in ("pelvis", "torso_link", "head_link"):
            try:
                out[f"_body_{b}"] = body_pose(b)
            except ValueError:
                out[f"_body_{b}"] = "absent"
        return out

    def objects():
        out = {}
        origin = scene.np(u.scene.env_origins)[0]
        for role, obj, asset in (
            ("apple", scene.apple, scene.arena_env.task.pick_up_object),
            ("plate", scene.plate, scene.arena_env.task.destination_location),
        ):
            view = obj.root_view if hasattr(obj, "root_view") else obj.root_physx_view
            d = {
                "asset_name": asset.name,
                "usd_path": str(getattr(asset, "usd_path", None)),
                "scale": jsonable(getattr(asset, "scale", None)),
                "prim_path": obj.cfg.prim_path,
                "pos_local": (scene.np(obj.data.root_pos_w)[0] - origin).tolist(),
                "quat_w_xyzw": scene.np(obj.data.root_quat_w)[0].tolist(),
                "mass_kg": scene.np(view.get_masses()).ravel().tolist(),
                "inertia": scene.np(view.get_inertias()).ravel().tolist(),
            }
            try:
                d["material_static_dynamic_restitution_per_shape"] = (
                    scene.np(view.get_material_properties()).reshape(-1, 3).tolist()
                )
            except Exception as exc:  # noqa: BLE001
                d["material_error"] = f"{type(exc).__name__}: {exc}"
            d["usd"] = usd_facts(
                obj.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace(
                    "env_.*", "env_0"
                )
            )
            out[role] = d
        return out

    def task_and_events():
        tm = u.termination_manager
        em = u.event_manager
        terms = {}
        for n in tm.active_terms:
            c = tm.get_term_cfg(n)
            terms[n] = {
                "func": getattr(c.func, "__name__", repr(c.func)),
                "params": jsonable(
                    {
                        k: repr(v) if not isinstance(v, int | float) else v
                        for k, v in c.params.items()
                    }
                ),
                "time_out": bool(c.time_out),
            }
        events = {}
        for mode in em.available_modes:
            events[mode] = {}
            for n in em.active_terms[mode]:
                c = em.get_term_cfg(n)
                events[mode][n] = {
                    "func": getattr(c.func, "__name__", repr(c.func)),
                    "params": jsonable({k: repr(v) for k, v in c.params.items()}),
                }
        sensor = u.scene["pick_up_object_contact_sensor"]
        return {
            "termination_terms": terms,
            "events": events,
            "contact_sensor": {
                "prim_path": sensor.cfg.prim_path,
                "filter_prim_paths_expr": list(sensor.cfg.filter_prim_paths_expr),
                "update_period": sensor.cfg.update_period,
                "history_length": sensor.cfg.history_length,
            },
            "task_description": scene.arena_env.task.get_task_description(),
            "metrics": [type(m).__name__ for m in (u.cfg.metrics or [])],
        }

    def background():
        bg = scene.arena_env.task.background_scene
        d = {
            "name": bg.name,
            "usd_path": str(getattr(bg, "usd_path", None)),
            "object_min_z": getattr(bg, "object_min_z", None),
        }
        d["shelf_support_usd"] = usd_facts("/World/envs/env_0/static_pick_place_shelf_support", 5)
        d["background_usd_bounds"] = {
            k: v
            for k, v in usd_facts("/World/envs/env_0/galileo_locomanip", 1).items()
            if k.startswith("world_aabb")
        }
        d["env_origin"] = scene.np(u.scene.env_origins)[0].tolist()
        d["scene_entities"] = sorted(u.scene.keys())
        return d

    section(facts, "cameras", cameras)
    section(facts, "objects", objects)
    section(facts, "task_and_events", task_and_events)
    section(facts, "background", background)

    def write_frames(tag):
        fr = scene.frames()
        for name, img in fr.items():
            save_png(os.path.join(out, "frames", f"{name}_{tag}.png"), img)
        return fr

    # ------------------------------------------------------------ episode A: hold
    hold = at.wbc_action(scene.hold_targets(), scene.joint_names)
    episodes: dict = {"hold_action": hold.tolist()}
    log = []
    first = None
    step_times = []
    for k in range(own.hold_steps):
        t0 = time.monotonic()
        res = scene.step(hold)
        step_times.append(time.monotonic() - t0)
        tr = scene.truth()
        log.append(
            {
                "step": k,
                "terminated": res["terminated"],
                "truncated": res["truncated"],
                "terms": res["terms"],
                **tr,
            }
        )
        if k == 0:
            fr = write_frames("hold_step000")
            section(facts, "plate_colour", lambda fr=fr: plate_colour(scene, fr, facts))
        if k in (100, 200):
            write_frames(f"hold_step{k:03d}")
        if res["terminated"] or res["truncated"]:
            first = k
            # The env auto-resets inside step(); this frame is the post-reset one.
            write_frames(f"hold_after_end_step{k:03d}")
            break
    episodes["A_hold"] = {
        "ended_at_step": first,
        "log": log,
        "step_wall_s_median": float(np.median(step_times)),
        "step_wall_s_p95": float(np.percentile(step_times, 95)),
    }
    print(f"[arena_probe] episode A ended at step {first}", flush=True)

    # ------------------------------------------------------------ episode B: success path
    log = []
    for _ in range(own.settle_steps):
        scene.step(hold)
    plate = scene.truth()["plate_pos_local"]
    origin_z = float(scene.np(u.scene.env_origins)[0][2])
    plate_top = usd_facts(scene.plate.cfg.prim_path.replace("env_.*", "env_0"), 1)
    plate_top_z = float(plate_top["world_aabb_max"][2]) - origin_z
    a0 = facts["objects"]["apple"]
    apple_origin_above_bottom = float(a0["pos_local"][2]) - (
        float(a0["usd"]["world_aabb_min"][2]) - origin_z
    )
    target = (plate[0], plate[1], plate_top_z + apple_origin_above_bottom + own.drop_clearance)
    scene.teleport_apple(target)
    fired = None
    for k in range(own.success_steps):
        res = scene.step(hold)
        log.append(
            {
                "step": k,
                "terminated": res["terminated"],
                "truncated": res["truncated"],
                "terms": res["terms"],
                **scene.truth(),
            }
        )
        if k in (0, 10, 25):
            write_frames(f"drop_step{k:03d}")
        if res["terminated"] or res["truncated"]:
            fired = k
            break
    episodes["B_success_path"] = {
        "teleport_target_local": list(target),
        "plate_top_z_local": plate_top_z,
        "apple_origin_above_bottom_m": apple_origin_above_bottom,
        "drop_clearance_m": own.drop_clearance,
        "ended_at_step_after_teleport": fired,
        "log": log,
    }
    print(f"[arena_probe] episode B ended at step {fired}", flush=True)
    try:
        episodes["metrics"] = jsonable(scene.metrics())
    except Exception as exc:  # noqa: BLE001
        episodes["metrics"] = {"error": f"{type(exc).__name__}: {exc}"}
    print(f"[arena_probe] metrics {episodes['metrics']}", flush=True)
    facts["wall_s_total"] = time.monotonic() - t_start

    with open(os.path.join(out, "facts.json"), "w") as f:
        json.dump(jsonable(facts), f, indent=1)
    with open(os.path.join(out, "episodes.json"), "w") as f:
        json.dump(jsonable(episodes), f, indent=1)
    print("[arena_probe] done", flush=True)
    sys.stdout.flush()
    os._exit(0)  # Kit shutdown can hang (Arena force-exits its test subprocesses the same way)


def plate_colour(scene, frames, facts):
    """Sample rendered colour at the projected plate centre, a ring inside it, and the apple."""
    out = {}
    plate = facts.get("objects", {}).get("plate", {})
    apple = facts.get("objects", {}).get("apple", {})
    origin = scene.np(scene.unwrapped.scene.env_origins)[0]
    for name, cam in facts.get("cameras", {}).items():
        if name.startswith("_") or name not in frames or "intrinsics" not in cam:
            continue
        img = frames[name]
        pos_w = np.asarray(cam["pos_local"]) + origin
        res = {}
        for role, d in (("plate", plate), ("apple", apple)):
            usd = d.get("usd", {})
            if "world_aabb_min" not in usd:
                continue
            lo, hi = np.asarray(usd["world_aabb_min"]), np.asarray(usd["world_aabb_max"])
            c = (lo + hi) / 2
            top = np.array(
                [c[0], c[1], hi[2] if role == "apple" else lo[2] + 0.3 * (hi[2] - lo[2])]
            )
            r = 0.25 * min(hi[0] - lo[0], hi[1] - lo[1])
            pts = [top] + [
                top + r * np.array([np.cos(a), np.sin(a), 0.0])
                for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)
            ]
            uv = at.project_points_ros(pts, pos_w, cam["quat_w_ros_xyzw"], cam["intrinsics"])
            samples = [at.patch_mean_rgb(img, x, y) for x, y, _ in uv]
            res[role] = {"pixels_uvz": uv.tolist(), "rgb_samples": samples}
        out[name] = res
    return out


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
