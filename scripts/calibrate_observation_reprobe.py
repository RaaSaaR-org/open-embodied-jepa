"""TASK-061 pre-freeze calibration: rendering facts and prior-only baselines. Nothing is fitted.

Produces the numbers that ``docs/experiments/apple_observation_reprobe_v1.md`` uses to choose
the look motion, to validate each arm's render path, and to justify thresholds, before any readout
exists. It fits **no** readout on any image or encoder feature. Its only "models" are prior-only
predictors (a training-fold mean, a majority sign, a constant), and they read no observation.

Scope: the 190 train + val roots of ``data/apple-wide-v1``, filtered from the frozen collection
plan and checked against the dataset manifest's frozen splits (as in TASK-059). No
``DatasetStore`` is constructed and no test episode is opened.

It measures:

* **Look-motion candidates (visibility only).** Five reset-independent right-arm commands, each
  repeated for k = 1..10 steps through the collector's own step (clip to the collection bounds,
  project, execute). For each (variant, k) it counts the roots whose apple has >= 1 segmentation
  pixel in the 112 px onboard camera. The pre-declared rule in ``choose_look`` picks the look.
* **The chosen look, in detail.** Applied commands on every root; the robot state after the look
  (must be identical across resets); apple and plate displacement during the look; the expert's
  command from the post-look state (the new targets).
* **Apple pixels in every arm's decision frame** (segmentation, analysis only).
* **Render-path validation per arm.** Each arm's frame is rendered by the simulation's own
  observation path (``G1Embodiment.observe`` for onboard frames, ``MuJoCoSimulation.render`` for
  ``overview``). It is compared byte-for-byte with the frame of an independent replica instance
  that performed the same reset and look, after the dataset's PNG encode/decode. The 224 px
  instance's physics must equal the 112 px instance's exactly.
* **Prior-only baselines** for the reset targets (must equal TASK-059's manifest) and for the
  post-look targets, plus each arm's visibility-conditional mean.

    uv run --no-sync python scripts/calibrate_observation_reprobe.py \
        --output outputs/task061-observation-reprobe/calibration-v4.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import info_ceiling as ic  # noqa: E402

STORE = ROOT / "data" / "apple-wide-v1"
TASK059_MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-info-ceiling-v1.json"
SMALL, LARGE, NATIVE = 112, 224, 448
MAX_LOOK_STEPS = 10
FREE = [6, 7, 8, 9, 10, 11, 13]

# Right-arm (dx, dy, dz, droll, dpitch, dyaw). Left arm 0, both grasps -1 (open). Each is a
# constant: it reads nothing about the reset. ``expert_no_dx`` is the expert's own step-0 command
# with its reset-dependent components fixed: dx = 0 and dy at its mode (-0.4).
LOOK_VARIANTS = {
    "expert_no_dx": (0.0, -0.4, 0.4, -0.5, -0.5, 0.5),
    "raise": (0.0, 0.0, 0.4, 0.0, 0.0, 0.0),
    "raise_out": (0.0, -0.4, 0.4, 0.0, 0.0, 0.0),
    "raise_back": (-0.4, 0.0, 0.4, 0.0, 0.0, 0.0),
    "out": (0.0, -0.4, 0.0, 0.0, 0.0, 0.0),
}


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COLLECTOR = _load("_collect_apple_wide", "scripts/collect_apple_wide.py")


def look_command(variant: str) -> np.ndarray:
    command = np.zeros(14, np.float32)
    command[6:12] = LOOK_VARIANTS[variant]
    command[12:] = -1.0
    return np.clip(command, COLLECTOR.LOWER, COLLECTOR.UPPER).astype(np.float32)


def look_sequence(variant: str, steps: int) -> np.ndarray:
    return np.repeat(look_command(variant)[None], steps, axis=0)


def choose_look(visible_by_variant: dict, total: int) -> dict:
    """Pre-declared rule. For each variant, k* = the smallest k (1..10) at which the apple is
    visible on every root. Choose the variant with the smallest k*; ties go to the listed order.
    If no variant reaches every root, choose the (variant, k) with the most visible roots, then
    the smallest k, then the listed order."""
    order = list(LOOK_VARIANTS)
    full = []
    for name in order:
        counts = visible_by_variant[name]
        ks = [k for k in range(1, MAX_LOOK_STEPS + 1) if counts[k] == total]
        if ks:
            full.append((ks[0], order.index(name), name))
    if full:
        k, _, name = min(full)
        return {"variant": name, "steps": k, "rule": "smallest k with every root visible"}
    best = max(
        (visible_by_variant[n][k], -k, -order.index(n), n)
        for n in order
        for k in range(1, MAX_LOOK_STEPS + 1)
    )
    return {"variant": best[3], "steps": -best[1], "rule": "no variant reaches every root"}


def roots_train_val():
    plan = COLLECTOR.make_plan(COLLECTOR.FROZEN_SEEDS)["roots"]
    roots = sorted((r for r in plan if r["split"] in ("train", "val")), key=lambda r: r["seed"])
    if len(roots) != 190:
        raise RuntimeError(f"expected 190 train+val roots, found {len(roots)}")
    return roots


def first_frame(manifest, episode_id):
    import pyarrow.parquet as pq
    from PIL import Image

    row = next(r for r in manifest["episodes"] if r["episode_id"] == episode_id)
    path = STORE / row["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"][row["path"]]:
        raise RuntimeError(f"episode hash mismatch: {episode_id}")
    table = pq.read_table(path, columns=["observation.images.onboard_rgb", "frame_index"])
    if table["frame_index"][0].as_py() != 0:
        raise RuntimeError("first stored row is not frame 0")
    item = table["observation.images.onboard_rgb"][0].as_py()
    with Image.open(io.BytesIO(item["bytes"])) as image:
        return np.asarray(image).copy(), row


def png_roundtrip(frame: np.ndarray) -> np.ndarray:
    """The dataset's storage encoding (``data.py``: PIL PNG), decoded again."""
    from PIL import Image

    stream = io.BytesIO()
    Image.fromarray(frame).save(stream, format="PNG")
    with Image.open(io.BytesIO(stream.getvalue())) as image:
        return np.asarray(image).copy()


def warm_up(renderer, data, camera="onboard_rgb"):
    """One discarded render. A fresh MuJoCo renderer's FIRST onboard render at 224 px differs
    from every later render of the same state (1 pixel, 1 level); from the second render on it
    is stable. Every renderer is warmed up once before any frame is kept."""
    renderer.update_scene(data, camera=camera)
    renderer.render()


def make_robot(size):
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.simulation import MuJoCoSimulation

    robot = G1Embodiment(
        MuJoCoSimulation(object_kind="apple", container_kind="plate", width=size, height=size)
    )
    robot.sim.render()  # creates the simulation's own renderer and warms it up
    return robot


class Segmenter:
    """Apple pixel counts by segmentation render (privileged; analysis only)."""

    def __init__(self, robot, sizes):
        mj = robot.mj
        self.robot, self.mj = robot, mj
        body = mj.mj_name2id(robot.model, mj.mjtObj.mjOBJ_BODY, "apple")
        self.geoms = [g for g in range(robot.model.ngeom) if robot.model.geom_bodyid[g] == body]
        self.renderers = {}
        for n in sizes:
            renderer = mj.Renderer(robot.model, height=n, width=n)
            renderer.enable_segmentation_rendering()
            warm_up(renderer, robot.sim.data)
            self.renderers[n] = renderer

    def count(self, size, camera="onboard_rgb"):
        renderer = self.renderers[size]
        renderer.update_scene(self.robot.sim.data, camera=camera)
        seg = renderer.render()
        hit = np.isin(seg[..., 0], self.geoms) & (seg[..., 1] == int(self.mj.mjtObj.mjOBJ_GEOM))
        return int(hit.sum())

    def close(self):
        for renderer in self.renderers.values():
            renderer.close()


def execute_look(robot, sequence):
    """Execute a fixed command sequence through the collector's step. Returns applied commands.

    Reads nothing about the reset: the sequence is fixed before the call, and only the robot's
    own observe/project/execute path runs."""
    applied = []
    for command in sequence:
        robot.observe()
        projected = robot.project_candidates(np.asarray(command, np.float32)[None, None, None])
        if not projected.feasible[0, 0]:
            raise RuntimeError("look command projection infeasible")
        result = robot.execute(projected.actions[0, 0, 0].astype(np.float32))
        if result.applied_action is None:
            raise RuntimeError(f"look command rejected: {result.reason}")
        applied.append(result.applied_action.copy())
    return np.asarray(applied)


def look_candidates(roots):
    robot = make_robot(SMALL)
    seg = Segmenter(robot, (SMALL,))
    table = {}
    try:
        for name in LOOK_VARIANTS:
            counts = np.zeros((len(roots), MAX_LOOK_STEPS + 1), int)
            for i, root in enumerate(roots):
                robot.reset(
                    seed=root["seed"], object_xy=root["object_xy"], plate_xy=root["plate_xy"]
                )
                counts[i, 0] = seg.count(SMALL)
                for t, command in enumerate(look_sequence(name, MAX_LOOK_STEPS)):
                    execute_look(robot, command[None])
                    counts[i, t + 1] = seg.count(SMALL)
            visible = (counts > 0).sum(axis=0)
            table[name] = {
                "command_right_arm": list(LOOK_VARIANTS[name]),
                "visible_roots_by_k": visible.tolist(),
                "median_visible_pixels_by_k": [
                    float(np.median(counts[:, k][counts[:, k] > 0])) if visible[k] else 0.0
                    for k in range(MAX_LOOK_STEPS + 1)
                ],
                "min_pixels_by_k": counts.min(axis=0).tolist(),
            }
    finally:
        seg.close()
        robot.sim.close()
    return table


def hold_control(roots, steps):
    """Apple motion when the arm holds still for the same number of steps (zero right-arm
    command, grasps open): separates gravity settling from anything the look does."""
    hold = np.zeros((steps, 14), np.float32)
    hold[:, 12:] = -1.0
    robot = make_robot(SMALL)
    moves = []
    try:
        for root in roots:
            truth = robot.reset(
                seed=root["seed"], object_xy=root["object_xy"], plate_xy=root["plate_xy"]
            )
            before = truth["object_position"].copy()
            execute_look(robot, hold)
            moves.append(robot.sim.task_truth()["object_position"] - before)
    finally:
        robot.sim.close()
    return np.asarray(moves)


def model_settings(robot) -> dict:
    """Render-relevant model facts: the MJCF the instance was built from, offscreen size and
    anti-aliasing samples. The 112 and 224 px instances must agree on all of them."""
    model = robot.model
    return {
        "scene_sha256": hashlib.sha256(robot.sim._scene().encode()).hexdigest(),
        "offwidth": int(model.vis.global_.offwidth),
        "offheight": int(model.vis.global_.offheight),
        "offsamples": int(model.vis.quality.offsamples),
        "shadowsize": int(model.vis.quality.shadowsize),
        "render_size": [int(robot.sim.width), int(robot.sim.height)],
    }


def state_vector(robot):
    sim = robot.sim
    return np.concatenate((sim.data.qpos[sim.qadr], sim.data.qvel[sim.vadr])).astype(np.float64)


def measure_arms(roots, dataset, sequence):
    """Reset and post-look measurements, with a replica per resolution for path validation."""
    from embodied_jepa.scripted import apple_collector_policy

    robots = {
        "112": make_robot(SMALL),
        "112_replica": make_robot(SMALL),
        "224": make_robot(LARGE),
        "224_replica": make_robot(LARGE),
    }
    seg = Segmenter(robots["112"], (SMALL, LARGE, NATIVE))
    native = {
        k: robots[k].mj.Renderer(robots[k].model, height=NATIVE, width=NATIVE)
        for k in ("112", "112_replica")
    }
    for k, renderer in native.items():
        warm_up(renderer, robots[k].sim.data)
    rows = []
    try:
        for root in roots:
            truths = {
                k: r.reset(
                    seed=root["seed"], object_xy=root["object_xy"], plate_xy=root["plate_xy"]
                )
                for k, r in robots.items()
            }
            truth = truths["112"]
            obs = {k: r.observe().images["onboard_rgb"][0].copy() for k, r in robots.items()}
            stored, _row = first_frame(dataset, root["episode_id"])
            overview = {
                k: robots[k].sim.render(camera="overview").copy() for k in ("112", "112_replica")
            }
            overview_again = robots["112"].sim.render(camera="overview").copy()
            n448 = {}
            for k, renderer in native.items():
                renderer.update_scene(robots[k].sim.data, camera="onboard_rgb")
                n448[k] = renderer.render().copy()
            r448_u8 = np.floor(ic.box_down(n448["112"][None], NATIVE // SMALL)[0] + 0.5).astype(
                np.uint8
            )
            r448_u8_rep = np.floor(
                ic.box_down(n448["112_replica"][None], NATIVE // SMALL)[0] + 0.5
            ).astype(np.uint8)
            expert0 = apple_collector_policy(truth).action(robots["112"])
            reset_state = {k: state_vector(r) for k, r in robots.items()}
            reset_pixels = {
                "onboard_112": seg.count(SMALL),
                "onboard_224": seg.count(LARGE),
                "onboard_448": seg.count(NATIVE),
                "overview_112": seg.count(SMALL, camera="overview"),
            }
            applied = {k: execute_look(r, sequence) for k, r in robots.items()}
            post_obs = {k: r.observe().images["onboard_rgb"][0].copy() for k, r in robots.items()}
            rerender = {k: robots[k].sim.render().copy() for k in ("112", "224")}
            post_truth = robots["112"].sim.task_truth()
            post_state = {k: state_vector(r) for k, r in robots.items()}
            expert_post = apple_collector_policy(truth).action(robots["112"])
            expert_post_224 = apple_collector_policy(truths["224"]).action(robots["224"])
            post_pixels = {"onboard_112": seg.count(SMALL), "onboard_224": seg.count(LARGE)}
            palm, _ = robots["112"].ee_pose("right")
            same = np.array_equal
            rows.append(
                {
                    "seed": int(root["seed"]),
                    "split": root["split"],
                    "apple_xy": [float(v) for v in truth["object_position"][:2]],
                    "reset_pixels": reset_pixels,
                    "post_look_pixels": post_pixels,
                    "expert_step0": [float(v) for v in expert0[FREE]],
                    "expert_post_look": [float(v) for v in expert_post[FREE]],
                    "expert_post_look_224_equal": bool(same(expert_post, expert_post_224)),
                    "applied_look": applied["112"],
                    "applied_equal_across_instances": all(
                        same(applied["112"], a) for a in applied.values()
                    ),
                    "reset_state": reset_state["112"],
                    "post_state": post_state["112"],
                    "physics_224_equals_112": bool(
                        same(reset_state["112"], reset_state["224"])
                        and same(post_state["112"], post_state["224"])
                        and same(reset_state["224"], reset_state["224_replica"])
                        and same(post_state["224"], post_state["224_replica"])
                    ),
                    "apple_xy_displacement_m": float(
                        np.abs(
                            post_truth["object_position"][:2] - truth["object_position"][:2]
                        ).max()
                    ),
                    "apple_move_m": [
                        float(v) for v in post_truth["object_position"] - truth["object_position"]
                    ],
                    "apple_z_settle_m": float(
                        post_truth["object_position"][2] - truth["object_position"][2]
                    ),
                    "plate_displacement_m": float(
                        np.abs(post_truth["plate_position"] - truth["plate_position"]).max()
                    ),
                    "hand_contact_after_look": bool(post_truth["hand_contact"]),
                    "palm_post_look": [float(v) for v in palm],
                    "path": {
                        "reset_112_equals_stored": bool(same(obs["112"], stored)),
                        "reset_112_replica": bool(same(obs["112"], obs["112_replica"])),
                        "reset_224_replica": bool(same(obs["224"], obs["224_replica"])),
                        "reset_overview_replica": bool(
                            same(overview["112"], overview["112_replica"])
                        ),
                        "post_112_replica": bool(same(post_obs["112"], post_obs["112_replica"])),
                        "post_224_replica": bool(same(post_obs["224"], post_obs["224_replica"])),
                        "r448_u8_replica": bool(same(r448_u8, r448_u8_rep)),
                        "r448_u8_equals_stored": bool(same(r448_u8, stored)),
                        "png_roundtrip": bool(
                            all(
                                same(png_roundtrip(f), f)
                                for f in (
                                    post_obs["112"],
                                    post_obs["224"],
                                    obs["224"],
                                    overview["112"],
                                    r448_u8,
                                )
                            )
                        ),
                        "post_112_differs_from_reset": bool(not same(post_obs["112"], obs["112"])),
                        "P4_post_112_rerender": bool(same(rerender["112"], post_obs["112"])),
                        "P4_post_224_rerender": bool(same(rerender["224"], post_obs["224"])),
                        "P4_overview_rerender": bool(same(overview_again, overview["112"])),
                    },
                }
            )
    finally:
        seg.close()
        for renderer in native.values():
            renderer.close()
        for robot in robots.values():
            robot.sim.close()
    return rows


def arm_occ_prior(xy, occluded, fold) -> np.ndarray:
    """Visibility-conditional mean for an arm's own flag. A training fold with no root of the
    held root's flag falls back to the training-fold mean (B-mean)."""
    xy = np.asarray(xy, np.float64)
    occluded = np.asarray(occluded, bool)
    out = np.zeros_like(xy)
    for k in range(int(fold.max()) + 1):
        fit, held = fold != k, fold == k
        for flag in (True, False):
            rows = held & (occluded == flag)
            if not rows.any():
                continue
            source = fit & (occluded == flag)
            out[rows] = xy[source].mean(0) if source.any() else xy[fit].mean(0)
    return out


def quantiles(values):
    return np.percentile(np.asarray(values), [0, 25, 50, 75, 90, 100]).tolist()


def pixel_summary(values):
    values = np.asarray(values)
    return {
        "zero": int((values == 0).sum()),
        "visible": int((values > 0).sum()),
        "quantiles_min_p25_p50_p75_p90_max": quantiles(values),
        "visible_median": float(np.median(values[values > 0])) if (values > 0).any() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    started = time.perf_counter()
    dataset = json.loads((STORE / "meta" / "jepa_manifest.json").read_text())
    roots = roots_train_val()
    ic.check_split(roots, dataset["splits"])
    fold = ic.fold_of(len(roots))
    candidates = look_candidates(roots)
    chosen = choose_look({n: c["visible_roots_by_k"] for n, c in candidates.items()}, len(roots))
    sequence = look_sequence(chosen["variant"], chosen["steps"])
    rows = measure_arms(roots, dataset, sequence)
    settings = {}
    for size in (SMALL, LARGE):
        probe = make_robot(size)
        settings[str(size)] = model_settings(probe)
        probe.sim.close()
    hold_moves = hold_control(roots, chosen["steps"])
    look_moves = np.array([r["apple_move_m"] for r in rows])

    xy = np.array([r["apple_xy"] for r in rows])
    train = np.array([r["split"] == "train" for r in rows])
    dx0 = np.array([r["expert_step0"][0] for r in rows])
    post = np.array([r["expert_post_look"] for r in rows])
    dx_post = post[:, 0]
    reset_px = {k: np.array([r["reset_pixels"][k] for r in rows]) for k in rows[0]["reset_pixels"]}
    post_px = {
        k: np.array([r["post_look_pixels"][k] for r in rows]) for k in rows[0]["post_look_pixels"]
    }
    occluded059 = reset_px["onboard_112"] == 0
    applied = np.stack([r["applied_look"] for r in rows])
    reset_states = np.stack([r["reset_state"] for r in rows])
    post_states = np.stack([r["post_state"] for r in rows])
    task059 = json.loads(TASK059_MANIFEST.read_text())["calibration"]["values"]["prior_baselines"]
    reset_priors = ic.prior_summary(xy, dx0, occluded059, fold, train)
    post_priors = ic.prior_summary(xy, dx_post, occluded059, fold, train)
    arm_flags = {
        "L_onboard112_look": post_px["onboard_112"] == 0,
        "LH_onboard224_look": post_px["onboard_224"] == 0,
        "O_overview112_reset": reset_px["overview_112"] == 0,
        "H_onboard224_reset": reset_px["onboard_224"] == 0,
    }
    arm_occ = {}
    for arm, flag in arm_flags.items():
        err = ic.xy_error_cm(arm_occ_prior(xy, flag, fold), xy)
        arm_occ[arm] = {
            "occluded_roots": int(flag.sum()),
            "B_occ_arm_median_cm": float(np.median(err)),
            "B_occ_arm_median_cm_arm_visible": float(np.median(err[~flag])),
            "B_occ_arm_median_cm_arm_occluded": float(np.median(err[flag])) if flag.any() else None,
        }
    path_keys = rows[0]["path"].keys()
    names = ("right_dx", "right_dy", "right_dz", "right_droll", "right_dpitch", "right_dyaw")
    report = {
        "task": "TASK-061",
        "fits_nothing": True,
        "roots": {"train": int(train.sum()), "val": int((~train).sum())},
        "fold_assignment_sha256": ic.fold_assignment_sha256([r["seed"] for r in rows], fold),
        "look_candidates": candidates,
        "look_chosen": chosen
        | {
            "command": look_command(chosen["variant"]).tolist(),
            "sequence_sha256": hashlib.sha256(
                np.ascontiguousarray(sequence, np.float32).tobytes()
            ).hexdigest(),
        },
        "look_facts": {
            "applied_max_abs_spread_across_roots": float(np.abs(applied - applied[0]).max()),
            "applied_max_abs_minus_requested": float(np.abs(applied - sequence[None]).max()),
            "applied_equal_across_instances": int(
                sum(r["applied_equal_across_instances"] for r in rows)
            ),
            "reset_state_max_std": float(reset_states.std(axis=0).max()),
            "post_look_state_max_std": float(post_states.std(axis=0).max()),
            "post_look_state_max_abs_spread": float(np.abs(post_states - post_states[0]).max()),
            "apple_xy_displacement_max_m": max(r["apple_xy_displacement_m"] for r in rows),
            "apple_z_settle_min_max_m": [
                min(r["apple_z_settle_m"] for r in rows),
                max(r["apple_z_settle_m"] for r in rows),
            ],
            "hold_control_apple_xy_displacement_max_m": float(np.abs(hold_moves[:, :2]).max()),
            "hold_control_apple_z_settle_min_max_m": [
                float(hold_moves[:, 2].min()),
                float(hold_moves[:, 2].max()),
            ],
            "apple_move_look_minus_hold_max_abs_m": float(np.abs(look_moves - hold_moves).max()),
            "plate_displacement_max_m": max(r["plate_displacement_m"] for r in rows),
            "hand_contact_after_look": int(sum(r["hand_contact_after_look"] for r in rows)),
            "palm_post_look_first_root": rows[0]["palm_post_look"],
            "expert_post_look_224_equal": int(sum(r["expert_post_look_224_equal"] for r in rows)),
        },
        "targets_post_look": {
            "per_dimension_min_mean_max": {
                n: [float(post[:, i].min()), float(post[:, i].mean()), float(post[:, i].max())]
                for i, n in enumerate(names)
            },
            "dx_positive_count": int((dx_post > 0).sum()),
            "dx_zero_count": int((dx_post == 0).sum()),
            "dx_saturated_fraction": float((np.abs(dx_post) >= 0.4 - 1e-6).mean()),
            "dx_sign_equals_reset_sign": int((np.sign(dx_post) == np.sign(dx0)).sum()),
            "dx_sign_boundary_apple_x_m": [
                float(xy[dx_post > 0, 0].min()) if (dx_post > 0).any() else None,
                float(xy[dx_post < 0, 0].max()) if (dx_post < 0).any() else None,
            ],
            "sign_perfect_clip_dx_mae": float(np.abs(np.sign(dx_post) * 0.4 - dx_post).mean()),
        },
        "apple_pixels": {
            "reset": {k: pixel_summary(v) for k, v in reset_px.items()},
            "post_look": {k: pixel_summary(v) for k, v in post_px.items()},
            "post_look_112_visible_on_task059_occluded_roots": int(
                ((post_px["onboard_112"] > 0) & occluded059).sum()
            ),
        },
        "render_path_validation": {k: int(sum(r["path"][k] for r in rows)) for k in path_keys}
        | {
            "physics_224_equals_112": int(sum(r["physics_224_equals_112"] for r in rows)),
            "roots": len(rows),
            "model_settings": settings,
            "P5_model_224_equals_112_except_render_size": all(
                settings["224"][k] == settings["112"][k]
                for k in settings["112"]
                if k != "render_size"
            ),
        },
        "prior_baselines": {
            "reset_targets": reset_priors,
            "reset_targets_max_abs_vs_task059_manifest": max(
                abs(reset_priors[k] - task059[k]) for k in task059
            ),
            "post_look_targets": post_priors,
            "arm_visibility_conditional_mean": arm_occ,
        },
        "holm": {
            "alpha_one_sided": 0.025,
            "hypotheses": 4,
            "step_thresholds": [0.025 / (4 - j) for j in range(4)],
        },
        "wilson_95_lower_reference": {
            "162_of_190": ic.wilson(162, 190)[0],
            "19_of_20": ic.wilson(19, 20)[0],
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(json.dumps({k: report[k] for k in ("look_chosen", "look_facts")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
