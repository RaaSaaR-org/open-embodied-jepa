"""TASK-059 pre-freeze calibration: rendering facts and prior-only baselines. Nothing is fitted.

Produces every number that ``docs/experiments/apple_info_ceiling_v1.md`` uses to set or justify
a threshold before any readout exists. It fits **no** readout on any image or encoder feature;
the only "models" here are prior-only predictors (train-fold mean, majority sign), which read no
observation at all.

Over the 190 train + val roots of ``data/apple-wide-v1`` it measures the list below. The roots are
filtered from the collection plan and then checked: the run refuses unless every root's split
agrees with the dataset manifest's frozen split, no root is a test/holdout episode, and every seed
lies in 48000-48199. No test episode is decoded (the dataset manifest is read as JSON; no
``DatasetStore`` is constructed, so not even a hash pass touches test files).

* whether a 112 px re-render from the recorded reset coordinates is byte-identical to the stored
  training frame, and whether the reset proprioception is identical across resets;
* the apple's visible pixel count at reset, by segmentation render, at 112 px and at 448 px
  (privileged, analysis only -- never a readout input);
* how closely a 4x4 box-downsample of the 448 px render reproduces the stored 112 px frame, and
  whether it identifies its own reset among the 190;
* the scripted expert's step-zero command, recomputed from the reset, against the recorded label;
* which body occludes the apple on occluded resets (segmentation at the apple centre's
  projection, 448 px);
* the prior-only baselines: predict-the-mean apple xy, the majority dx sign, the constant dx.

    uv run --no-sync python scripts/calibrate_info_ceiling.py \
        --output outputs/task059-info-ceiling/calibration.json
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

STORE = ROOT / "data" / "apple-wide-v1"
IMAGE_SIZE = 112
NATIVE_SIZE = 448
FOLDS = 10
FOLD_SEED = 59
TRANSLATION_CLIP = 0.4


def _collector():
    spec = importlib.util.spec_from_file_location(
        "_collect_apple_wide", ROOT / "scripts" / "collect_apple_wide.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def roots_train_val():
    """The 190 train + val roots of the frozen collection plan, sorted by seed."""
    collector = _collector()
    plan = collector.make_plan(collector.FROZEN_SEEDS)["roots"]
    roots = sorted((r for r in plan if r["split"] in ("train", "val")), key=lambda r: r["seed"])
    if len(roots) != 190:
        raise RuntimeError(f"expected 190 train+val roots, found {len(roots)}")
    return roots


def check_roots(roots, manifest) -> None:
    splits = manifest["splits"]
    forbidden = set(splits.get("test", [])) | set(splits.get("holdout", []))
    for root in roots:
        if not 48000 <= root["seed"] <= 48199:
            raise RuntimeError(f"seed {root['seed']} outside 48000-48199")
        if root["episode_id"] in forbidden:
            raise RuntimeError(f"{root['episode_id']} is a test/holdout episode")
        if root["episode_id"] not in set(splits[root["split"]]):
            raise RuntimeError(
                f"{root['episode_id']} is not in the dataset's {root['split']} split"
            )


def fold_of(n: int) -> np.ndarray:
    """Outer fold per root (roots sorted by seed): a seeded permutation, then index mod 10."""
    order = np.random.default_rng(FOLD_SEED).permutation(n)
    fold = np.empty(n, np.int64)
    fold[order] = np.arange(n) % FOLDS
    return fold


def first_frame(manifest, episode_id):
    """The stored reset frame, after checking the episode file's recorded sha256."""
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


def apple_pixels(robot, renderer, camera="onboard_rgb") -> int:
    mj = robot.mj
    body = mj.mj_name2id(robot.model, mj.mjtObj.mjOBJ_BODY, "apple")
    geoms = [g for g in range(robot.model.ngeom) if robot.model.geom_bodyid[g] == body]
    renderer.update_scene(robot.sim.data, camera=camera)
    seg = renderer.render()
    return int((np.isin(seg[..., 0], geoms) & (seg[..., 1] == int(mj.mjtObj.mjOBJ_GEOM))).sum())


def body_at_point(robot, renderer, camera, point):
    """Name of the body drawn at a world point's projection in the onboard camera (448 px)."""
    data = robot.sim.data
    rotation = data.cam_xmat[camera].reshape(3, 3)
    local = rotation.T @ (np.asarray(point, float) - data.cam_xpos[camera])
    if local[2] >= 0:
        return None
    focal = (NATIVE_SIZE / 2) / np.tan(np.deg2rad(float(robot.model.cam_fovy[camera])) / 2)
    col = int(np.floor(NATIVE_SIZE / 2 + focal * local[0] / -local[2]))
    row = int(np.floor(NATIVE_SIZE / 2 - focal * local[1] / -local[2]))
    if not (0 <= row < NATIVE_SIZE and 0 <= col < NATIVE_SIZE):
        return None
    renderer.update_scene(data, camera="onboard_rgb")
    object_id, object_type = renderer.render()[row, col]
    if object_type != int(robot.mj.mjtObj.mjOBJ_GEOM) or object_id < 0:
        return None
    return robot.model.body(int(robot.model.geom_bodyid[object_id])).name


def box_down(image, factor):
    h, w, c = image.shape
    return (
        image.reshape(h // factor, factor, w // factor, factor, c).astype(np.float64).mean((1, 3))
    )


def measure_roots():
    from embodied_jepa import readout_labels
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.scripted import apple_collector_policy
    from embodied_jepa.simulation import MuJoCoSimulation

    manifest = json.loads((STORE / "meta" / "jepa_manifest.json").read_text())
    robot = G1Embodiment(
        MuJoCoSimulation(
            object_kind="apple", container_kind="plate", width=IMAGE_SIZE, height=IMAGE_SIZE
        )
    )
    seg = {n: robot.mj.Renderer(robot.model, height=n, width=n) for n in (IMAGE_SIZE, NATIVE_SIZE)}
    for renderer in seg.values():
        renderer.enable_segmentation_rendering()
    native = robot.mj.Renderer(robot.model, height=NATIVE_SIZE, width=NATIVE_SIZE)
    rows, stored_frames, down_frames, states = [], [], [], []
    roots = roots_train_val()
    check_roots(roots, manifest)
    camera = robot.mj.mj_name2id(robot.model, robot.mj.mjtObj.mjOBJ_CAMERA, "onboard_rgb")
    for root in roots:
        truth = robot.reset(
            seed=root["seed"], object_xy=root["object_xy"], plate_xy=root["plate_xy"]
        )
        observation = robot.observe()
        stored, row = first_frame(manifest, root["episode_id"])
        native.update_scene(robot.sim.data, camera="onboard_rgb")
        down = box_down(native.render(), NATIVE_SIZE // IMAGE_SIZE)
        expert = apple_collector_policy(truth).action(robot)
        labels = readout_labels.load_privileged(
            STORE, row, acknowledge_privileged_training_labels=True
        )
        rows.append(
            {
                "seed": int(root["seed"]),
                "split": root["split"],
                "aim_offset": root["aim_offset_xy_m"] is not None,
                "rerender_112_identical": bool(
                    np.array_equal(observation.images["onboard_rgb"][0], stored)
                ),
                "apple_pixels_112": apple_pixels(robot, seg[IMAGE_SIZE]),
                "apple_pixels_448": apple_pixels(robot, seg[NATIVE_SIZE]),
                "body_at_apple_centre_448": body_at_point(
                    robot, seg[NATIVE_SIZE], camera, truth["object_position"]
                ),
                "apple_xy": [float(v) for v in truth["object_position"][:2]],
                "apple_xy_label_max_abs": float(
                    np.abs(
                        np.asarray(labels["privileged__apple_position_world"][0][:2], np.float64)
                        - np.asarray(truth["object_position"][:2])
                    ).max()
                ),
                "expert_step0_free": [float(v) for v in expert[[6, 7, 8, 9, 10, 11, 13]]],
                "recorded_step0_free": [
                    float(v)
                    for v in np.asarray(labels["collector__base_action"][0])[
                        [6, 7, 8, 9, 10, 11, 13]
                    ]
                ],
            }
        )
        stored_frames.append(stored)
        down_frames.append(down)
        states.append(observation.robot_state[0].astype(np.float64))
    for renderer in (*seg.values(), native):
        renderer.close()
    robot.sim.close()
    return rows, np.stack(stored_frames), np.stack(down_frames), np.stack(states)


def equivalence(stored, down, train):
    stored = stored.astype(np.float64)
    mae = np.abs(down[:, None] - stored[None]).mean(axis=(2, 3, 4))
    own = np.diag(mae).copy()
    off = np.where(np.eye(len(mae), dtype=bool), np.inf, mae)
    between = np.abs(stored[:, None] - stored[None]).mean(axis=(2, 3, 4))
    between = np.where(np.eye(len(between), dtype=bool), np.inf, between)
    over = np.abs(down - stored).max(axis=3) > 8
    level = stored.max(axis=3)
    edge = np.zeros_like(over)
    centre = level[:, 1:-1, 1:-1]
    edge[:, 1:-1, 1:-1] = (
        np.stack(
            [
                np.abs(centre - level[:, 1:-1, :-2]),
                np.abs(centre - level[:, 1:-1, 2:]),
                np.abs(centre - level[:, :-2, 1:-1]),
                np.abs(centre - level[:, 2:, 1:-1]),
            ]
        ).max(axis=0)
        > 8
    )
    bias = (down - stored)[train].mean(axis=0)
    corrected = np.abs((down - bias)[:, None] - stored[None]).mean(axis=(2, 3, 4))
    c_own = np.diag(corrected).copy()
    c_off = np.where(np.eye(len(corrected), dtype=bool), np.inf, corrected)
    return {
        "native_size": NATIVE_SIZE,
        "downsample": "4x4 box mean, float64, no rounding",
        "own_frame_mae_levels": {"min": own.min(), "median": np.median(own), "max": own.max()},
        "own_frame_max_abs_levels": float(np.abs(down - stored).max()),
        "fraction_pixels_over_8_levels": float(over.mean()),
        "share_of_over_8_pixels_at_edges": float(edge[over].mean()),
        "edge_definition": "stored-frame max-channel level differs by > 8 from a 4-neighbour",
        "identifies_own_reset": int((own < off.min(axis=1)).sum()),
        "stored_frames_nearest_other_reset_mae_levels": {
            "min": between.min(),
            "median": float(np.median(between.min(axis=1))),
        },
        "bias_corrected": {
            "bias_estimated_on": "train roots only",
            "own_frame_mae_levels_min": c_own.min(),
            "own_frame_mae_levels_median": float(np.median(c_own)),
            "own_frame_mae_levels_max": c_own.max(),
            "identifies_own_reset": int((c_own < c_off.min(axis=1)).sum()),
        },
    }


def wilson_lower(successes: int, n: int, z: float = 1.959963984540054) -> float:
    p = successes / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((centre - half) / (1 + z * z / n))


def prior_baselines(xy, dx, occluded, fold, train):
    n = len(dx)
    mean_pred = np.zeros((n, 2))
    dx_median = np.zeros(n)
    majority = np.zeros(n)
    stratum_mean = np.zeros((n, 2))
    for k in range(FOLDS):
        fit, held = fold != k, fold == k
        mean_pred[held] = xy[fit].mean(0)
        dx_median[held] = np.median(dx[fit])
        majority[held] = 1.0 if (dx[fit] > 0).mean() >= 0.5 else -1.0
        for flag in (True, False):
            stratum_mean[held & (occluded == flag)] = xy[fit & (occluded == flag)].mean(0)
    # y-conditional sign prior: occlusion tracks the apple's y, so a readout that only learns
    # "where along y the hand hides the apple" could exploit P(dx > 0 | y). Privileged baseline:
    # majority sign within 5 quantile bins of the TRUE apple y, bins fit on the training folds.
    y_prior = np.zeros(n)
    for k in range(FOLDS):
        fit, held = fold != k, fold == k
        edges = np.quantile(xy[fit, 1], [0.2, 0.4, 0.6, 0.8])
        fit_bin, held_bin = np.digitize(xy[fit, 1], edges), np.digitize(xy[held, 1], edges)
        held_index = np.flatnonzero(held)
        for b in range(5):
            share = (dx[fit][fit_bin == b] > 0).mean()
            y_prior[held_index[held_bin == b]] = 1.0 if share >= 0.5 else -1.0
    error_cm = np.linalg.norm(mean_pred - xy, axis=1) * 100
    stratum_cm = np.linalg.norm(stratum_mean - xy, axis=1) * 100
    sign = np.sign(dx)
    val = ~train
    val_mean = xy[train].mean(0)
    return {
        "cv_predict_the_mean_median_cm": float(np.median(error_cm)),
        "cv_predict_the_mean_median_cm_occluded": float(np.median(error_cm[occluded])),
        "cv_predict_the_mean_median_cm_visible": float(np.median(error_cm[~occluded])),
        "cv_occlusion_conditional_mean_median_cm_occluded": float(np.median(stratum_cm[occluded])),
        "cv_occlusion_conditional_mean_median_cm_visible": float(np.median(stratum_cm[~occluded])),
        "cv_occlusion_conditional_mean_median_cm": float(np.median(stratum_cm)),
        "cv_y_conditional_dx_sign_accuracy": float((y_prior == sign).mean()),
        "cv_y_conditional_dx_sign_accuracy_occluded": float((y_prior == sign)[occluded].mean()),
        "ceiling_if_visible_perfect_and_occluded_at_majority": float(
            ((~occluded).sum() + (majority == sign)[occluded].sum()) / n
        ),
        "cv_y_conditional_dx_sign_accuracy_visible": float((y_prior == sign)[~occluded].mean()),
        "cv_constant_median_dx_mae_occluded": float(np.abs(dx_median - dx)[occluded].mean()),
        "cv_constant_median_dx_mae_visible": float(np.abs(dx_median - dx)[~occluded].mean()),
        "constant_plus_clip_dx_mae": float(np.abs(TRANSLATION_CLIP - dx).mean()),
        "cv_majority_dx_sign_accuracy": float((majority == sign).mean()),
        "cv_majority_dx_sign_accuracy_occluded": float((majority == sign)[occluded].mean()),
        "cv_majority_dx_sign_accuracy_visible": float((majority == sign)[~occluded].mean()),
        "cv_constant_median_dx_mae": float(np.abs(dx_median - dx).mean()),
        "val_predict_the_mean_median_cm": float(
            np.median(np.linalg.norm(val_mean - xy[val], axis=1) * 100)
        ),
        "val_majority_dx_sign_accuracy": float(
            (sign[val] == (1.0 if (dx[train] > 0).mean() >= 0.5 else -1.0)).mean()
        ),
        "val_constant_median_dx_mae": float(np.abs(np.median(dx[train]) - dx[val]).mean()),
    }


def occluders(rows, occluded) -> dict:
    names = [str(r["body_at_apple_centre_448"]) for r, o in zip(rows, occluded, strict=True) if o]
    return {name: names.count(name) for name in sorted(set(names))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    started = time.perf_counter()
    rows, stored, down, states = measure_roots()
    train = np.array([r["split"] == "train" for r in rows])
    aim = np.array([r["aim_offset"] for r in rows])
    xy = np.array([r["apple_xy"] for r in rows])
    expert = np.array([r["expert_step0_free"] for r in rows])
    recorded = np.array([r["recorded_step0_free"] for r in rows])
    dx = expert[:, 0]
    pixels = np.array([r["apple_pixels_112"] for r in rows])
    pixels_448 = np.array([r["apple_pixels_448"] for r in rows])
    occluded = pixels == 0
    names = (
        "right_dx",
        "right_dy",
        "right_dz",
        "right_droll",
        "right_dpitch",
        "right_dyaw",
        "right_grasp",
    )
    report = {
        "task": "TASK-059",
        "fits_nothing": True,
        "roots": {"train": int(train.sum()), "val": int((~train).sum())},
        "rerender_112_byte_identical": int(sum(r["rerender_112_identical"] for r in rows)),
        "reset_state_max_std": float(states.std(axis=0).max()),
        "apple_label_equals_reset_truth_max_abs_m": max(r["apple_xy_label_max_abs"] for r in rows),
        "apple_visible_pixels_at_reset": {
            "occluded_112": int(occluded.sum()),
            "occluded_448": int((pixels_448 == 0).sum()),
            "occluded_112_val": int((occluded & ~train).sum()),
            "quantiles_112_p25_p50_p75_p90_max": np.percentile(
                pixels, [25, 50, 75, 90, 100]
            ).tolist(),
            "quantiles_448_p25_p50_p75_p90_max": np.percentile(
                pixels_448, [25, 50, 75, 90, 100]
            ).tolist(),
            "mean_apple_y_occluded": float(xy[occluded, 1].mean()),
            "mean_apple_y_visible": float(xy[~occluded, 1].mean()),
            "mean_apple_x_occluded": float(xy[occluded, 0].mean()),
            "mean_apple_x_visible": float(xy[~occluded, 0].mean()),
            "occluded_fraction_dx_positive": float(occluded[dx > 0].mean()),
            "occluded_fraction_dx_negative": float(occluded[dx < 0].mean()),
            "body_at_apple_centre_on_occluded_448": occluders(rows, occluded),
        },
        "expert_step0": {
            "recomputed_equals_recorded_on_non_aim_roots_max_abs": float(
                np.abs(expert[~aim] - recorded[~aim]).max()
            ),
            "recomputed_minus_recorded_on_aim_roots_max_abs": float(
                np.abs(expert[aim] - recorded[aim]).max()
            ),
            "per_dimension_min_mean_max": {
                n: [
                    float(expert[:, i].min()),
                    float(expert[:, i].mean()),
                    float(expert[:, i].max()),
                ]
                for i, n in enumerate(names)
            },
            "dx_positive_fraction": float((dx > 0).mean()),
            "dx_positive_count": int((dx > 0).sum()),
            "dx_zero_count": int((dx == 0).sum()),
            "dx_saturated_fraction": float((np.abs(dx) >= TRANSLATION_CLIP - 1e-6).mean()),
            "dx_sign_boundary_apple_x_m": [float(xy[dx > 0, 0].min()), float(xy[dx < 0, 0].max())],
            "sign_perfect_clip_dx_mae": float(np.abs(np.sign(dx) * TRANSLATION_CLIP - dx).mean()),
        },
        "native_equivalence": equivalence(stored, down, train),
        "prior_baselines": prior_baselines(xy, dx, occluded, fold_of(len(rows)), train),
        "wilson_95_lower_reference": {
            "162_of_190": wilson_lower(162, 190),
            "68_of_79": wilson_lower(68, 79),
            "19_of_20": wilson_lower(19, 20),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(json.dumps(report, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
