"""TASK-059 gated run: the reset-frame information-ceiling probe (``apple_info_ceiling_v1``).

Order (protocol §12 first, then §5-§11, then §13):

1. **Guards** -- G-hash (inputs, episode files, encoder weight digests), G-split (exactly the
   190 train+val roots; test ids, cohort C, cohort D and foreign seeds refused), G-render (112 px
   re-render byte-identical on 190/190, reset proprioception identical), G-expert, G-prior. Any
   failure writes a VOID report and stops.
2. **Sources** -- E0 / A3 / random ``image_features`` of the stored frame, raw 112 px, raw 448 px
   (reported only), and 448 px box-downsampled to 112 (reported only).
3. **Readouts** -- nested 10-fold CV (primary), train->val (secondary), from
   ``embodied_jepa.info_ceiling``.
4. **Evaluation** on all / occluded / visible roots; the random-floor qualifier; the §10
   apple-hidden / shadow-off explanation check if any decisional source beats the prior on
   occluded resets.
5. **Descriptive** -- visibility over time along the nominal expert (train roots only), other
   cameras at reset, frame near-identity, native equivalence.
6. **Decision** (§13).

CPU only. The manifest is never written. The test split is never decoded: the frame reader
refuses any episode id that is not one of the 190 checked roots. (Loading the encoders builds a
``DatasetStore``, whose ``verify()`` reads every episode file's bytes only to check its sha256.)

    uv run --no-sync python scripts/probe_info_ceiling.py \
        --output outputs/task059-info-ceiling/run-1

``--smoke`` checks the wiring on 24 roots (20 train, 4 val) with the readout TARGETS REPLACED by
seeded noise, so it reveals nothing about the real readout; it skips G-prior (a full-cohort
number) and runs the descriptive rollout on 2 roots for 20 steps.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import info_ceiling as ic  # noqa: E402
from embodied_jepa.contracts import ContractError  # noqa: E402

PROTOCOL = "apple_info_ceiling_v1"
TASK = "TASK-059"
MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-info-ceiling-v1.json"
CONFIG = ROOT / "configs" / "apple_policy_v1.yaml"
STORE = ROOT / "data" / "apple-wide-v1"
CHECKPOINTS = ROOT / "checkpoints" / "task056-policy-v1"
ENCODER_ARMS = {"E0": "a2", "A3": "a3", "random": "a1"}
MANIFEST_SOURCE_KEYS = {"E0": "i_E0", "A3": "ii_A3", "random": "iii_random"}
IMAGE_SIZE = 112
NATIVE_SIZE = 448
CROP_RENDER_SIZE = 320
DEVICE = "cpu"
GLOBAL_WALL_SECONDS = 7200.0
DESCRIPTIVE_STEPS = 210  # orient (130) + descend (80)
SATURATION_BAND_M = 0.4 * 0.015  # dx leaves the +-0.4 clip inside 6 mm
HIDDEN_GROUP = 5
FREE = [6, 7, 8, 9, 10, 11, 13]


class VoidRun(Exception):
    """A §12 guard failed."""


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n")


class Clock:
    def __init__(self, cap):
        self.start, self.cap = time.monotonic(), cap

    def elapsed(self):
        return time.monotonic() - self.start

    def check(self, stage):
        if self.elapsed() > self.cap:
            raise VoidRun(f"global wall cap {self.cap} s reached before {stage}")


# ----- roots and frames ----------------------------------------------------------------------
def plan_roots():
    collector = _load("_collect_apple_wide", "scripts/collect_apple_wide.py")
    plan = collector.make_plan(collector.FROZEN_SEEDS)["roots"]
    roots = sorted((r for r in plan if r["split"] in ("train", "val")), key=lambda r: r["seed"])
    return roots, collector


class FrameReader:
    """Reads frame 0 of an ALLOWED episode only, after its sha256 check (G-hash)."""

    def __init__(self, manifest, allowed_ids):
        self.manifest = manifest
        self.allowed = frozenset(allowed_ids)
        self.rows = {r["episode_id"]: r for r in manifest["episodes"]}

    def row(self, episode_id):
        if episode_id not in self.allowed:
            raise ic.GuardError(f"refusing to open episode {episode_id}: not a checked root")
        return self.rows[episode_id]

    def first_frame(self, episode_id, store_root=STORE):
        import pyarrow.parquet as pq
        from PIL import Image

        row = self.row(episode_id)
        path = Path(store_root) / row["path"]
        if sha256(path) != self.manifest["sha256"][row["path"]]:
            raise ic.GuardError(f"episode file hash mismatch: {episode_id}")
        table = pq.read_table(path, columns=["observation.images.onboard_rgb", "frame_index"])
        if table["frame_index"][0].as_py() != 0:
            raise ic.GuardError(f"first stored row of {episode_id} is not frame 0")
        item = table["observation.images.onboard_rgb"][0].as_py()
        with Image.open(io.BytesIO(item["bytes"])) as image:
            return np.asarray(image).copy()


# ----- simulation ----------------------------------------------------------------------------
def make_robot():
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.simulation import MuJoCoSimulation

    return G1Embodiment(
        MuJoCoSimulation(
            object_kind="apple", container_kind="plate", width=IMAGE_SIZE, height=IMAGE_SIZE
        )
    )


class Renders:
    """Segmentation, native RGB, and ablated 112 px renderers on the robot's model."""

    def __init__(self, robot):
        mj = robot.mj
        self.robot, self.mj = robot, mj
        model = robot.model
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "apple")
        self.apple_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
        self.seg = {}
        for n in (IMAGE_SIZE, NATIVE_SIZE, CROP_RENDER_SIZE):
            renderer = mj.Renderer(model, height=n, width=n)
            renderer.enable_segmentation_rendering()
            self.seg[n] = renderer
        self.native = mj.Renderer(model, height=NATIVE_SIZE, width=NATIVE_SIZE)
        self.small = mj.Renderer(model, height=IMAGE_SIZE, width=IMAGE_SIZE)
        self.camera_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "onboard_rgb")

    def apple_pixels(self, size, camera="onboard_rgb", window=None):
        renderer = self.seg[size]
        renderer.update_scene(self.robot.sim.data, camera=camera)
        seg = renderer.render()
        if window is not None:
            top, left = window
            seg = seg[top : top + IMAGE_SIZE, left : left + IMAGE_SIZE]
        hit = np.isin(seg[..., 0], self.apple_geoms) & (
            seg[..., 1] == int(self.mj.mjtObj.mjOBJ_GEOM)
        )
        return int(hit.sum())

    def native_rgb(self):
        self.native.update_scene(self.robot.sim.data, camera="onboard_rgb")
        return self.native.render().copy()

    def small_rgb(self, *, hide_apple=False, shadows=True):
        """A fresh 112 px render, optionally with the apple hidden or shadows off (§10)."""
        model = self.robot.model
        option = self.mj.MjvOption()
        saved = [int(model.geom_group[g]) for g in self.apple_geoms]
        try:
            if hide_apple:
                for g in self.apple_geoms:
                    model.geom_group[g] = HIDDEN_GROUP
                option.geomgroup[HIDDEN_GROUP] = 0
            self.small.update_scene(self.robot.sim.data, camera="onboard_rgb", scene_option=option)
            self.small.scene.flags[self.mj.mjtRndFlag.mjRND_SHADOW] = bool(shadows)
            image = self.small.render().copy()
        finally:
            for g, group in zip(self.apple_geoms, saved, strict=True):
                model.geom_group[g] = group
            self.small.scene.flags[self.mj.mjtRndFlag.mjRND_SHADOW] = True
        return image

    def close(self):
        for renderer in (*self.seg.values(), self.native, self.small):
            renderer.close()


def crop_window(robot):
    from embodied_jepa.hand_crop import HandCrop

    crop = HandCrop(robot, render_size=CROP_RENDER_SIZE, crop_size=IMAGE_SIZE)
    try:
        return crop.window()
    finally:
        crop.close()


def measure_resets(roots, reader, *, clock):
    """Stored frames, re-render checks, visibility, native renders, ablations, targets."""
    from embodied_jepa import readout_labels
    from embodied_jepa.scripted import apple_collector_policy

    robot = make_robot()
    renders = Renders(robot)
    out = {k: [] for k in ("stored", "native", "hidden", "shadow_off", "fresh_small", "states")}
    rows = []
    try:
        for root in roots:
            clock.check("reset measurement")
            truth = robot.reset(
                seed=root["seed"], object_xy=root["object_xy"], plate_xy=root["plate_xy"]
            )
            observation = robot.observe()
            stored = reader.first_frame(root["episode_id"])
            labels = readout_labels.load_privileged(
                STORE, reader.row(root["episode_id"]), acknowledge_privileged_training_labels=True
            )
            expert = apple_collector_policy(truth).action(robot)
            window = crop_window(robot)
            rows.append(
                {
                    "seed": int(root["seed"]),
                    "split": root["split"],
                    "episode_id": root["episode_id"],
                    "aim_offset": root["aim_offset_xy_m"] is not None,
                    "rerender_identical": bool(
                        np.array_equal(observation.images["onboard_rgb"][0], stored)
                    ),
                    "apple_pixels": {
                        "onboard_112": renders.apple_pixels(IMAGE_SIZE),
                        "onboard_448": renders.apple_pixels(NATIVE_SIZE),
                        "overview_112": renders.apple_pixels(IMAGE_SIZE, camera="overview"),
                        "overview_448": renders.apple_pixels(NATIVE_SIZE, camera="overview"),
                        "hand_crop_window": renders.apple_pixels(CROP_RENDER_SIZE, window=window),
                    },
                    "apple_xy_truth": [float(v) for v in truth["object_position"][:2]],
                    "apple_xy_label": [
                        float(v) for v in labels["privileged__apple_position_world"][0][:2]
                    ],
                    "expert_step0": [float(v) for v in expert[FREE]],
                    "recorded_step0": [
                        float(v) for v in np.asarray(labels["collector__base_action"][0])[FREE]
                    ],
                    "reset": {"object_xy": root["object_xy"], "plate_xy": root["plate_xy"]},
                }
            )
            out["stored"].append(stored)
            out["native"].append(renders.native_rgb())
            out["fresh_small"].append(renders.small_rgb())
            out["hidden"].append(renders.small_rgb(hide_apple=True))
            out["shadow_off"].append(renders.small_rgb(shadows=False))
            out["states"].append(observation.robot_state[0].astype(np.float64))
    finally:
        renders.close()
        robot.sim.close()
    return rows, {k: np.stack(v) for k, v in out.items()}


def derived_dx(rows, xy_pred):
    """Reported only: the expert's step-0 dx if the apple were at the PREDICTED xy."""
    from embodied_jepa.scripted import apple_collector_policy

    robot = make_robot()
    try:
        values = []
        for row, xy in zip(rows, xy_pred, strict=True):
            robot.reset(seed=row["seed"], **row["reset"])
            truth = robot.sim.task_truth()
            truth["object_position"] = np.array([xy[0], xy[1], truth["object_position"][2]])
            values.append(float(apple_collector_policy(truth).action(robot)[6]))
        return np.asarray(values)
    finally:
        robot.sim.close()


# ----- descriptive: visibility over time (§11.1) ---------------------------------------------
def visibility_over_time(roots, collector, *, steps, clock):
    from embodied_jepa.scripted import apple_collector_policy

    robot = make_robot()
    renders = Renders(robot)
    results = []
    try:
        for root in roots:
            if root["split"] != "train":
                raise ic.GuardError("visibility over time runs on train roots only")
            clock.check("visibility over time")
            truth = robot.reset(
                seed=root["seed"], object_xy=root["object_xy"], plate_xy=root["plate_xy"]
            )
            expert = apple_collector_policy(truth)
            target_x = float(expert.phases[0].target_base[0])
            dx0 = None
            first = {1: None, 10: None}
            at_first = None
            x_in_band = None
            stop = None
            pixels = [renders.apple_pixels(IMAGE_SIZE)]
            for t in range(steps):
                robot.observe()  # projection reads the synchronized snapshot, as the collector's
                position, _ = robot.ee_pose("right")
                offset = target_x - float(position[0])
                if x_in_band is None and abs(offset) <= SATURATION_BAND_M:
                    x_in_band = t
                command = expert.action(robot)
                if dx0 is None:
                    dx0 = float(command[6])
                for threshold in (1, 10):
                    if first[threshold] is None and pixels[-1] >= threshold:
                        first[threshold] = t
                        if threshold == 1:
                            at_first = {
                                "x_offset_to_orient_target_m": offset,
                                "outside_saturation_band": bool(abs(offset) > SATURATION_BAND_M),
                                "dx": float(command[6]),
                                "dx_sign_equals_step0": bool(np.sign(command[6]) == np.sign(dx0)),
                            }
                requested = np.clip(command, collector.LOWER, collector.UPPER).astype(np.float32)
                try:
                    projected = robot.project_candidates(requested[None, None, None])
                except ContractError as error:
                    stop = f"guard: {error}"
                    break
                if not projected.feasible[0, 0]:
                    stop = "projection_infeasible"
                    break
                result = robot.execute(projected.actions[0, 0, 0].astype(np.float32))
                if result.applied_action is None:
                    stop = f"rejected: {result.reason}"
                    break
                expert.advance(result)
                pixels.append(renders.apple_pixels(IMAGE_SIZE))
            results.append(
                {
                    "seed": int(root["seed"]),
                    "first_step_ge_1_px": first[1],
                    "first_step_ge_10_px": first[10],
                    "at_first_visible": at_first,
                    "first_step_x_offset_in_band": x_in_band,
                    "stop": stop,
                    "steps_run": len(pixels) - 1,
                    "max_pixels": int(max(pixels)),
                }
            )
    finally:
        renders.close()
        robot.sim.close()
    return results


def summarize_visibility(results):
    firsts = [r["first_step_ge_1_px"] for r in results]
    seen = [f for f in firsts if f is not None]
    at = [r["at_first_visible"] for r in results if r["at_first_visible"] is not None]
    band = [r["first_step_x_offset_in_band"] for r in results]
    before_band = [
        r["first_step_ge_1_px"] < r["first_step_x_offset_in_band"]
        for r in results
        if r["first_step_ge_1_px"] is not None and r["first_step_x_offset_in_band"] is not None
    ]
    quant = lambda v: np.percentile(v, [0, 25, 50, 75, 100]).tolist() if v else None  # noqa: E731
    return {
        "roots": len(results),
        "visible_at_reset": sum(f == 0 for f in firsts),
        "never_visible_within_steps": sum(f is None for f in firsts),
        "first_step_ge_1_px_quantiles_min_p25_p50_p75_max": quant(seen),
        "first_step_ge_10_px_quantiles": quant(
            [r["first_step_ge_10_px"] for r in results if r["first_step_ge_10_px"] is not None]
        ),
        "at_first_visible_outside_saturation_band": sum(a["outside_saturation_band"] for a in at),
        "at_first_visible_dx_sign_equals_step0": sum(a["dx_sign_equals_step0"] for a in at),
        "at_first_visible_abs_x_offset_m_quantiles": quant(
            [abs(a["x_offset_to_orient_target_m"]) for a in at]
        ),
        "first_step_x_offset_in_band_quantiles": quant([b for b in band if b is not None]),
        "never_in_band": sum(b is None for b in band),
        "first_visible_before_x_enters_band": int(sum(before_band)),
        "comparable_roots": len(before_band),
        "stops": sorted({r["stop"] for r in results if r["stop"]}),
    }


# ----- features ------------------------------------------------------------------------------
def encoder_features(frames_by_name, manifest, *, smoke=False):
    """E0 / A3 / random image features, after checking each encoder's weight digest."""
    import torch

    measure = _load("_measure", "scripts/measure_policy_offline_conditionals.py")
    from embodied_jepa.world_model_v2 import _open

    config, store, _settings, cameras = _open(CONFIG)
    if tuple(cameras) != ("onboard_rgb",):
        raise ic.GuardError(f"config cameras {cameras} != ('onboard_rgb',)")
    features, digests = {}, {}
    for name, arm in ENCODER_ARMS.items():
        _policy, source = measure.build_policy(CHECKPOINTS / f"{arm}.pt", config, store, DEVICE)
        digest = source.weights_sha256()
        digests[name] = digest
        want = manifest["sources"][MANIFEST_SOURCE_KEYS[name]]["encoder_weights_sha256"]
        ic.check_encoder_digest(name, digest, want)
        source.model.eval()
        for frames_name, frames in frames_by_name.items():
            out = []
            with torch.no_grad():
                for start in range(0, len(frames), 32):
                    batch = {"onboard_rgb": frames[start : start + 32]}
                    out.append(source.features(batch, inference=True).cpu().numpy())
            features[(name, frames_name)] = np.concatenate(out).astype(np.float64)
    return features, digests


# ----- the run -------------------------------------------------------------------------------
def environment():
    import mujoco
    import torch

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "device": DEVICE,
    }


def evaluate_source(name, g, diag, xy, dx, dy, fold, priors, masks, train_mask, split_priors):
    xy_pred, xy_fits, xy_sel = ic.nested_cv(g, diag, xy, fold)
    dx_pred, dx_fits, dx_sel = ic.nested_cv(g, diag, dx, fold)
    dy_pred, _dy_fits, dy_sel = ic.nested_cv(g, diag, dy, fold)  # T4, reported only
    result = {"selections": {"xy": xy_sel, "dx": dx_sel, "dy": dy_sel}}
    kept = {}
    for stratum, mask in masks.items():
        evaluation = ic.evaluate(xy_pred, dx_pred, xy, dx, priors, mask)
        kept[stratum] = evaluation
        result[stratum] = ic.public(evaluation)
    sx, _, s_sel_xy = ic.split_fit(g, diag, xy, train_mask)
    sd, _, s_sel_dx = ic.split_fit(g, diag, dx, train_mask)
    val = ~train_mask
    full_xy, full_dx = np.zeros_like(xy), np.zeros_like(dx)
    full_xy[val], full_dx[val] = sx, sd
    result["secondary_train_to_val"] = ic.public(
        ic.evaluate(full_xy, full_dx, xy, dx, split_priors, val)
    ) | {"selections": {"xy": s_sel_xy, "dx": s_sel_dx}}
    predictions = {
        "xy": xy_pred,
        "dx": dx_pred,
        "dy": dy_pred,
        "xy_fits": xy_fits,
        "dx_fits": dx_fits,
    }
    return result, kept, predictions


def explanation_check(
    name,
    predictions,
    g_train_features,
    ablated,
    occluded_index,
    xy,
    dx,
    priors,
    occluded_mask,
    fold,
):
    """§10: apply each occluded root's own out-of-fold readouts to its ablated re-renders."""
    report = {}
    for label, features in ablated.items():
        g_new, norms = ic.cross_gram(features, g_train_features)
        xy_hat, dx_hat = predictions["xy"].copy(), predictions["dx"].copy()
        for row, root in enumerate(occluded_index):
            k = int(fold[root])
            xy_fit, dx_fit = predictions["xy_fits"][k], predictions["dx_fits"][k]
            g_rt = g_new[row : row + 1, xy_fit.stats.index]
            xy_hat[root] = xy_fit.predict(g_rt, norms[row : row + 1])[0]
            g_rt = g_new[row : row + 1, dx_fit.stats.index]
            dx_hat[root] = dx_fit.predict(g_rt, norms[row : row + 1])[0, 0]
        report[label] = ic.public(ic.evaluate(xy_hat, dx_hat, xy, dx, priors, occluded_mask))
    still = report["apple_hidden"]["beats_prior"]
    report["spurious"] = bool(still)
    # Owner ruling (pre-run, recorded in PR #49): scene-wide shadow-off changes ~59 % of every
    # frame, so the shadow-vs-residual sub-label is not interpretable and is never assigned.
    # The shadow-off numbers are kept, marked non-decisional.
    report["shadow_off"]["non_decisional"] = True
    report["reading"] = (
        "cue is not the apple: spurious"
        if still
        else "apple-caused: believed (shadow-vs-residual sub-label not interpretable)"
    )
    return report


def run(output: Path, *, smoke: bool = False) -> dict:
    from embodied_jepa.training import peak_rss_bytes, source_identity

    clock = Clock(GLOBAL_WALL_SECONDS)
    report = {
        "protocol": PROTOCOL,
        "task": TASK,
        "smoke": smoke,
        "source": source_identity(),
        "environment": environment(),
        "status": "running",
    }
    manifest = json.loads(MANIFEST.read_text())
    try:
        # --- G-hash (inputs) ---
        actual = {name: sha256(ROOT / name) for name in manifest["hashes"]}
        ic.check_hashes(manifest["hashes"], actual)
        report["hashes"] = actual
        if not smoke:
            ic.check_clean_tree(report["source"]["dirty"])
        # --- G-split ---
        roots, collector = plan_roots()
        dataset = json.loads((STORE / "meta" / "jepa_manifest.json").read_text())
        expected = ic.EXPECTED_ROOTS
        if smoke:
            train_r = [r for r in roots if r["split"] == "train"][:20]
            val_r = [r for r in roots if r["split"] == "val"][:4]
            roots = sorted(train_r + val_r, key=lambda r: r["seed"])
            expected = {"train": 20, "val": 4}
        ic.check_split(roots, dataset["splits"], expected=expected)
        seeds = [r["seed"] for r in roots]
        fold = ic.fold_of(len(roots))
        fold_hash = ic.fold_assignment_sha256(seeds, fold)
        if not smoke:
            ic.check_fold_hash(fold_hash, manifest["split"]["fold_assignment_sha256"])
        report["fold_assignment_sha256"] = fold_hash
        reader = FrameReader(dataset, [r["episode_id"] for r in roots])
        # --- measurements ---
        rows, frames = measure_resets(roots, reader, clock=clock)
        # --- G-render, G-expert ---
        ic.check_render([r["rerender_identical"] for r in rows], frames["states"])
        ic.check_frames_identical(
            frames["fresh_small"], frames["stored"], "the ablation renderer vs stored frames"
        )
        ic.check_expert(
            [r["expert_step0"] for r in rows],
            [r["recorded_step0"] for r in rows],
            [r["aim_offset"] for r in rows],
            [r["apple_xy_label"] for r in rows],
            [r["apple_xy_truth"] for r in rows],
        )
        xy = np.array([r["apple_xy_truth"] for r in rows], np.float64)
        dx = np.array([r["expert_step0"][0] for r in rows], np.float64)
        dy = np.array([r["expert_step0"][1] for r in rows], np.float64)
        train_mask = np.array([r["split"] == "train" for r in rows])
        pixels = np.array([r["apple_pixels"]["onboard_112"] for r in rows])
        occluded = pixels == 0
        # --- G-prior ---
        summary = ic.prior_summary(xy, dx, occluded, fold, train_mask)
        if not smoke:
            ic.check_priors(summary, manifest["calibration"]["values"]["prior_baselines"])
        report["prior_baselines"] = summary
        report["guards"] = "passed"
        if smoke:  # hide every real target from the readouts
            noise = np.random.default_rng(0)
            xy = xy.mean(0) + noise.normal(0, 0.017, xy.shape)
            dx = np.clip(noise.normal(0.1, 0.4, dx.shape), -0.4, 0.4)
            dx[dx == 0] = 0.1
            dy = noise.normal(-0.39, 0.02, dy.shape)
            report["smoke_targets"] = "seeded noise; no real target reached a readout"
        clock.check("features")
        # --- sources ---
        features, digests = encoder_features(
            {
                "stored": frames["stored"],
                "hidden": frames["hidden"],
                "shadow_off": frames["shadow_off"],
            },
            manifest,
        )
        report["encoder_weights_sha256"] = digests
        sources = {
            "E0": features[("E0", "stored")],
            "A3": features[("A3", "stored")],
            "random": features[("random", "stored")],
            "raw112": frames["stored"].reshape(len(rows), -1) / 255.0,
            "raw448": frames["native"].reshape(len(rows), -1) / 255.0,
            "raw448down": ic.box_down(frames["native"], NATIVE_SIZE // IMAGE_SIZE).reshape(
                len(rows), -1
            )
            / 255.0,
        }
        ablated_sources = {
            name: {
                "apple_hidden": (
                    features[(name, "hidden")]
                    if name in ENCODER_ARMS
                    else frames["hidden"].reshape(len(rows), -1) / 255.0
                ),
                "shadow_off": (
                    features[(name, "shadow_off")]
                    if name in ENCODER_ARMS
                    else frames["shadow_off"].reshape(len(rows), -1) / 255.0
                ),
            }
            for name in ic.DECISIONAL
        }
        priors = ic.prior_predictions(xy, dx, occluded, fold, dy=dy)
        split_fold = np.where(train_mask, 1, 0)
        split_priors = ic.prior_predictions(xy, dx, occluded, split_fold, dy=dy)
        masks = {"all": np.ones(len(rows), bool), "occluded": occluded, "visible": ~occluded}
        results, kept, preds = {}, {}, {}
        for name, x in sources.items():
            clock.check(f"readout {name}")
            g, diag = ic.gram(x)
            results[name], kept[name], preds[name] = evaluate_source(
                name, g, diag, xy, dx, dy, fold, priors, masks, train_mask, split_priors
            )
            results[name]["decisional"] = name in ic.DECISIONAL
            results[name]["feature_dim"] = int(x.shape[1])
        # --- random-floor qualifier ---
        for name in ("E0", "A3"):
            results[name]["random_floor"] = {
                stratum: ic.beats_random_floor(kept[name][stratum], kept["random"][stratum])
                for stratum in ("all", "visible")
            }
        # --- §7 pairwise comparisons of decisional sources ---
        pairs = {}
        for i, first in enumerate(ic.DECISIONAL):
            for second in ic.DECISIONAL[i + 1 :]:
                pairs[f"{first}_minus_{second}"] = {
                    stratum: ic.compare_sources(kept[first][stratum], kept[second][stratum])
                    for stratum in ("all", "visible")
                }
        report["pairwise_decisional"] = pairs
        # --- T4, reported only: dy, apple x alone, derived dx sign, per stratum ---
        for name in sources:
            derived = derived_dx(rows, preds[name]["xy"])
            results[name]["T4_reported"] = {
                stratum: ic.evaluate_reported(
                    preds[name]["xy"], preds[name]["dy"], derived, xy, dy, dx, priors, mask
                )
                for stratum, mask in masks.items()
            }
        # --- §10 explanation check ---
        occluded_index = np.flatnonzero(occluded)
        triggered = [s for s in ic.DECISIONAL if results[s]["occluded"]["beats_prior"]]
        explanation = {"triggered_by": triggered}
        spurious = {}
        for name in triggered:
            explanation[name] = explanation_check(
                name,
                preds[name],
                sources[name],
                {k: v[occluded_index] for k, v in ablated_sources[name].items()},
                occluded_index,
                xy,
                dx,
                priors,
                occluded,
                fold,
            )
            spurious[name] = explanation[name]["spurious"]
        if not triggered:
            explanation["not_run"] = "no decisional source beats the prior on occluded resets"
        report["explanation_check"] = explanation
        # --- decision ---
        report["decision"] = ic.decide(
            void=False,
            overall={s: results[s]["all"]["succeeds"] for s in ic.DECISIONAL},
            visible={s: results[s]["visible"]["succeeds"] for s in ic.DECISIONAL},
            spurious=spurious,
        )
        report["results"] = results
        # --- descriptive ---
        clock.check("descriptive")
        train_roots = [r for r in roots if r["split"] == "train"]
        steps = 20 if smoke else DESCRIPTIVE_STEPS
        visibility = visibility_over_time(
            train_roots[:2] if smoke else train_roots, collector, steps=steps, clock=clock
        )
        occluded_stored = frames["stored"][occluded].astype(np.float64)
        pair_mae, pair_px = [], []
        for i in range(len(occluded_stored)):
            for j in range(i + 1, len(occluded_stored)):
                d = np.abs(occluded_stored[i] - occluded_stored[j])
                pair_mae.append(d.mean())
                pair_px.append(int((d.max(axis=2) > 8).sum()))
        report["descriptive"] = {
            "visibility_over_time": summarize_visibility(visibility),
            "visibility_over_time_per_root": visibility,
            "apple_pixels_at_reset": {
                key: {
                    "zero": int(sum(r["apple_pixels"][key] == 0 for r in rows)),
                    "quantiles_p25_p50_p75_p90_max": np.percentile(
                        [r["apple_pixels"][key] for r in rows], [25, 50, 75, 90, 100]
                    ).tolist(),
                }
                for key in rows[0]["apple_pixels"]
            },
            "occluded_frame_pairs": {
                "pairs": len(pair_mae),
                "mae_levels_min": float(min(pair_mae)) if pair_mae else None,
                "mae_levels_median": float(np.median(pair_mae)) if pair_mae else None,
                "pixels_over_8_levels_min": int(min(pair_px)) if pair_px else None,
                "pixels_over_8_levels_median": float(np.median(pair_px)) if pair_px else None,
            },
            "native_equivalence": ic.native_equivalence(
                frames["stored"],
                ic.box_down(frames["native"], NATIVE_SIZE // IMAGE_SIZE),
                train_mask,
            ),
        }
        report["per_root"] = [
            {
                "seed": r["seed"],
                "split": r["split"],
                "fold": int(fold[i]),
                "occluded": bool(occluded[i]),
                "apple_pixels": r["apple_pixels"],
                "apple_xy": r["apple_xy_truth"],
                "expert_dx": r["expert_step0"][0],
                "expert_dy": r["expert_step0"][1],
                "predictions": {
                    s: {
                        "xy": preds[s]["xy"][i].tolist(),
                        "dx": float(preds[s]["dx"][i]),
                        "dy": float(preds[s]["dy"][i]),
                    }
                    for s in preds
                },
            }
            for i, r in enumerate(rows)
        ]
        report["status"] = "complete"
    except (ic.GuardError, VoidRun) as error:
        report["status"] = "void"
        report["decision"] = {"outcome": "VOID", "reason": str(error)}
    except Exception as error:  # noqa: BLE001 - record, then re-raise after writing
        report["status"] = "error"
        report["error"] = "".join(traceback.format_exception(error))
        report["elapsed_seconds"] = clock.elapsed()
        report["peak_rss_bytes"] = peak_rss_bytes()
        _write(output / "report.json", report)
        raise
    report["elapsed_seconds"] = clock.elapsed()
    report["peak_rss_bytes"] = peak_rss_bytes()
    _write(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = run(args.output, smoke=args.smoke)
    print(json.dumps({k: report.get(k) for k in ("status", "decision", "elapsed_seconds")}))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
