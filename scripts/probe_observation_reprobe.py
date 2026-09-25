"""TASK-061 gated run: the decision-time observation re-probe (``apple_observation_reprobe_v1``).

Order (protocol §9 guards first, then §3-§7, then §8):

1. **Guards** -- G-hash (inputs incl. the unchanged TASK-059 probe code; every episode file read;
   encoder weight digests; clean tree), G-split (190 train+val roots; test ids, cohorts C/D and
   foreign seeds refused; fold hash), G-render (112 px reset re-render byte-identical 190/190,
   reset proprioception identical), G-expert, G-look (sequence hash; applied == requested on
   every root in every instance; post-look state identical; apple xy / plate unmoved; no
   contact), G-prior (reset targets vs TASK-059's manifest, post-look targets vs this manifest).
   Any failure, crash or wall-cap stop writes a report with outcome ``V`` and a ``void_reason``.
2. **Frames** -- four simulation instances (112, 112 replica, 224, 224 replica) run the same
   reset and look. Render-path validation (§5) demotes an arm; it never voids.
3. **Sources** -- anchor raw-112 (stored reset frame), L raw/E0/A3/random (112 px post-look),
   LH raw (224 px post-look), O raw/E0 (overview 112 px at reset), H raw (224 px at reset),
   R float/uint8 (448 px 4x4 box mean at reset).
4. **Readouts** -- ``info_ceiling`` nested CV and evaluation, unchanged, via TASK-059's
   ``evaluate_source``.
5. **Holm** over L-raw, L-E0, LH-raw, O-raw; the apple-hidden spurious check on all four.
6. **Decision** (§8).

CPU only. The manifests are never written. The test split is never decoded: the frame reader
refuses any episode that is not a checked root. (Loading the encoders builds a ``DatasetStore``,
whose ``verify()`` reads every episode file's bytes only to check its sha256.)

    uv run --no-sync python scripts/probe_observation_reprobe.py \
        --output outputs/task061-observation-reprobe/run-1

``--smoke`` checks the wiring on 24 roots (20 train, 4 val) with the readout TARGETS REPLACED by
seeded noise, so it reveals nothing about the real readouts; it skips G-prior, the fold hash and
the clean-tree check (full-cohort or gated-run facts).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import info_ceiling as ic  # noqa: E402
from embodied_jepa import observation_reprobe as orp  # noqa: E402

PROTOCOL = "apple_observation_reprobe_v1"
TASK = "TASK-061"
MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-observation-reprobe-v1.json"
TASK059_MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-info-ceiling-v1.json"
STORE = ROOT / "data" / "apple-wide-v1"
SMALL, LARGE, NATIVE = 112, 224, 448
GLOBAL_WALL_SECONDS = 10800.0
FREE = [6, 7, 8, 9, 10, 11, 13]
INSTANCES = ("112", "112_replica", "224", "224_replica")


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# TASK-059's runner, loaded unchanged (G-hash pins its bytes): frame reader, encoder features,
# evaluate_source, derived_dx, clock, void-safe report writer.
BASE = _load("_probe_info_ceiling", "scripts/probe_info_ceiling.py")
VoidRun = BASE.VoidRun


def plan_roots():
    return BASE.plan_roots()


def sha256(path) -> str:
    return BASE.sha256(path)


def write_report(path: Path, report: dict) -> None:
    BASE.write_report(path, report)


# ----- simulation ----------------------------------------------------------------------------
def warm_up(renderer, data, camera="onboard_rgb"):
    """One discarded render (protocol §5): a fresh renderer's first 224 px render can differ
    from its later renders. Applied to every renderer of every arm."""
    renderer.update_scene(data, camera=camera)
    renderer.render()


def make_robot(size):
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.simulation import MuJoCoSimulation

    robot = G1Embodiment(
        MuJoCoSimulation(object_kind="apple", container_kind="plate", width=size, height=size)
    )
    robot.sim.render()  # creates the simulation's own renderer and discards its first render
    return robot


def model_settings(robot) -> dict:
    """P5: render-relevant model facts; the 112 and 224 px instances may differ only in size."""
    import hashlib

    model = robot.model
    return {
        "scene_sha256": hashlib.sha256(robot.sim._scene().encode()).hexdigest(),
        "offwidth": int(model.vis.global_.offwidth),
        "offheight": int(model.vis.global_.offheight),
        "offsamples": int(model.vis.quality.offsamples),
        "shadowsize": int(model.vis.quality.shadowsize),
    }


def png_roundtrip(frame: np.ndarray) -> np.ndarray:
    """The dataset's storage encoding (``data.py``: PIL PNG), decoded again."""
    import io

    from PIL import Image

    stream = io.BytesIO()
    Image.fromarray(frame).save(stream, format="PNG")
    with Image.open(io.BytesIO(stream.getvalue())) as image:
        return np.asarray(image).copy()


def state_vector(robot) -> np.ndarray:
    sim = robot.sim
    return np.concatenate((sim.data.qpos[sim.qadr], sim.data.qvel[sim.vadr])).astype(np.float64)


class Views:
    """Segmentation counts, the 448 px native render and apple-hidden ablation renders."""

    def __init__(self, robots):
        self.robots = robots
        r112 = robots["112"]
        self.mj = r112.mj
        mj = self.mj
        self.apple_geoms = {}
        for key in ("112", "224"):
            model = robots[key].model
            body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "apple")
            self.apple_geoms[key] = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
        self.seg = {}
        for n in (SMALL, LARGE, NATIVE):
            renderer = mj.Renderer(r112.model, height=n, width=n)
            renderer.enable_segmentation_rendering()
            warm_up(renderer, r112.sim.data)
            self.seg[n] = renderer
        self.native = {
            k: mj.Renderer(robots[k].model, height=NATIVE, width=NATIVE)
            for k in ("112", "112_replica")
        }
        self.ablation = {
            "112": mj.Renderer(robots["112"].model, height=SMALL, width=SMALL),
            "224": mj.Renderer(robots["224"].model, height=LARGE, width=LARGE),
        }
        for key, renderer in self.native.items():
            warm_up(renderer, robots[key].sim.data)
        for key, renderer in self.ablation.items():
            warm_up(renderer, robots[key].sim.data)

    def apple_pixels(self, size, camera="onboard_rgb"):
        renderer = self.seg[size]
        renderer.update_scene(self.robots["112"].sim.data, camera=camera)
        seg = renderer.render()
        hit = np.isin(seg[..., 0], self.apple_geoms["112"]) & (
            seg[..., 1] == int(self.mj.mjtObj.mjOBJ_GEOM)
        )
        return int(hit.sum())

    def native_rgb(self, key):
        self.native[key].update_scene(self.robots[key].sim.data, camera="onboard_rgb")
        return self.native[key].render().copy()

    def ablation_pair(self, key, camera):
        """(un-ablated, apple-hidden) render of ``camera`` on instance ``key``."""
        robot = self.robots[key]
        model, renderer = robot.model, self.ablation[key]
        geoms = self.apple_geoms[key]
        renderer.update_scene(robot.sim.data, camera=camera)
        plain = renderer.render().copy()
        option = self.mj.MjvOption()
        saved = [int(model.geom_group[g]) for g in geoms]
        try:
            for g in geoms:
                model.geom_group[g] = BASE.HIDDEN_GROUP
            option.geomgroup[BASE.HIDDEN_GROUP] = 0
            renderer.update_scene(robot.sim.data, camera=camera, scene_option=option)
            hidden = renderer.render().copy()
        finally:
            for g, group in zip(geoms, saved, strict=True):
                model.geom_group[g] = group
        return plain, hidden

    def close(self):
        for renderer in (*self.seg.values(), *self.native.values(), *self.ablation.values()):
            renderer.close()


def measure(roots, reader, *, clock):
    """Every frame, count, state and target the run needs, in one pass over the roots."""
    from embodied_jepa import readout_labels
    from embodied_jepa.scripted import apple_collector_policy

    robots = {k: make_robot(LARGE if k.startswith("224") else SMALL) for k in INSTANCES}
    settings = {k: model_settings(robots[k]) for k in ("112", "224")}
    p5 = orp.models_agree(settings["112"], settings["224"])
    views = Views(robots)
    keys = (
        "stored",
        "reset_224",
        "overview",
        "overview_hidden",
        "r448_float",
        "r448_u8",
        "post_112",
        "post_112_hidden",
        "post_224",
        "post_224_hidden",
        "reset_proprio",
        "post_state",
        "applied",
    )
    out = {k: [] for k in keys}
    rows = []
    same = orp.frames_identical
    try:
        for root in roots:
            clock.check("measurement")
            reset = {"object_xy": root["object_xy"], "plate_xy": root["plate_xy"]}
            truths = {k: r.reset(seed=root["seed"], **reset) for k, r in robots.items()}
            truth = truths["112"]
            observation = robots["112"].observe()
            obs = {k: r.observe().images["onboard_rgb"][0].copy() for k, r in robots.items()}
            obs["112"] = observation.images["onboard_rgb"][0].copy()
            stored = reader.first_frame(root["episode_id"])
            labels = readout_labels.load_privileged(
                STORE, reader.row(root["episode_id"]), acknowledge_privileged_training_labels=True
            )
            overview = {
                k: robots[k].sim.render(camera="overview").copy() for k in ("112", "112_replica")
            }
            overview_again = robots["112"].sim.render(camera="overview").copy()
            overview_plain, overview_hidden = views.ablation_pair("112", "overview")
            native = {k: views.native_rgb(k) for k in ("112", "112_replica")}
            r448_float = ic.box_down(native["112"][None], NATIVE // SMALL)[0]
            r448_u8 = np.floor(r448_float + 0.5).astype(np.uint8)
            r448_u8_rep = np.floor(
                ic.box_down(native["112_replica"][None], NATIVE // SMALL)[0] + 0.5
            ).astype(np.uint8)
            reset_pixels = {
                "onboard_112": views.apple_pixels(SMALL),
                "onboard_224": views.apple_pixels(LARGE),
                "onboard_448": views.apple_pixels(NATIVE),
                "overview_112": views.apple_pixels(SMALL, camera="overview"),
            }
            expert0 = apple_collector_policy(truth).action(robots["112"])
            reset_states = {k: state_vector(r) for k, r in robots.items()}
            # --- the look: the same constant on every instance and root ---
            during = []

            def after_step(_step, sim=robots["112"].sim, start=truth, during=during):
                now = sim.task_truth()
                during.append(
                    (
                        float(
                            np.abs(now["object_position"][:2] - start["object_position"][:2]).max()
                        ),
                        float(np.abs(now["plate_position"] - start["plate_position"]).max()),
                        bool(now["hand_contact"]),
                    )
                )

            applied = {
                k: orp.execute_look(r, after_step if k == "112" else None)
                for k, r in robots.items()
            }
            post = {k: r.observe().images["onboard_rgb"][0].copy() for k, r in robots.items()}
            rerender = {k: robots[k].sim.render().copy() for k in ("112", "224")}
            post_truth = robots["112"].sim.task_truth()
            post_states = {k: state_vector(r) for k, r in robots.items()}
            expert_post = apple_collector_policy(truth).action(robots["112"])
            expert_post_224 = apple_collector_policy(truths["224"]).action(robots["224"])
            post_pixels = {
                "onboard_112": views.apple_pixels(SMALL),
                "onboard_224": views.apple_pixels(LARGE),
            }
            l_plain, l_hidden = views.ablation_pair("112", "onboard_rgb")
            lh_plain, lh_hidden = views.ablation_pair("224", "onboard_rgb")
            rows.append(
                {
                    "seed": int(root["seed"]),
                    "split": root["split"],
                    "episode_id": root["episode_id"],
                    "aim_offset": root["aim_offset_xy_m"] is not None,
                    "reset": reset,
                    "apple_xy_truth": [float(v) for v in truth["object_position"][:2]],
                    "apple_xy_label": [
                        float(v) for v in labels["privileged__apple_position_world"][0][:2]
                    ],
                    "expert_step0": [float(v) for v in expert0[FREE]],
                    "recorded_step0": [
                        float(v) for v in np.asarray(labels["collector__base_action"][0])[FREE]
                    ],
                    "expert_post_look": [float(v) for v in expert_post[FREE]],
                    "expert_post_look_224_equal": bool(same(expert_post, expert_post_224)),
                    "apple_pixels_reset": reset_pixels,
                    "apple_pixels_post_look": post_pixels,
                    "apple_xy_move_m": max(d[0] for d in during),
                    "apple_z_move_m": float(
                        post_truth["object_position"][2] - truth["object_position"][2]
                    ),
                    "plate_move_m": max(d[1] for d in during),
                    "hand_contact": any(d[2] for d in during),
                    "look_steps_checked": len(during),
                    "path": {
                        "P0_reset_112_equals_stored": bool(same(obs["112"], stored)),
                        "P1_post_112_replica": bool(same(post["112"], post["112_replica"])),
                        "P2_png_post_112": bool(same(png_roundtrip(post["112"]), post["112"])),
                        "P1_post_224_replica": bool(same(post["224"], post["224_replica"])),
                        "P2_png_post_224": bool(same(png_roundtrip(post["224"]), post["224"])),
                        "P3_physics_224_equals_112": bool(
                            same(reset_states["112"], reset_states["224"])
                            and same(post_states["112"], post_states["224"])
                            and same(reset_states["224"], reset_states["224_replica"])
                            and same(post_states["224"], post_states["224_replica"])
                        ),
                        "P1_overview_replica": bool(same(overview["112"], overview["112_replica"])),
                        "P2_png_overview": bool(
                            same(png_roundtrip(overview["112"]), overview["112"])
                        ),
                        "P4_post_112_rerender": orp.frames_identical(post["112"], rerender["112"]),
                        "P4_post_224_rerender": orp.frames_identical(post["224"], rerender["224"]),
                        "P4_overview_rerender": orp.frames_identical(
                            overview["112"], overview_again
                        ),
                        "P5_model_224_equals_112": bool(p5),
                        "H_P1_reset_224_replica": bool(same(obs["224"], obs["224_replica"])),
                        "R_P1_u8_replica": bool(same(r448_u8, r448_u8_rep)),
                        "R_u8_equals_stored": bool(same(r448_u8, stored)),
                    },
                    "ablation_reproduces": {
                        "L": bool(same(l_plain, post["112"])),
                        "LH": bool(same(lh_plain, post["224"])),
                        "O": bool(same(overview_plain, overview["112"])),
                    },
                }
            )
            out["stored"].append(stored)
            out["reset_224"].append(obs["224"])
            out["overview"].append(overview["112"])
            out["overview_hidden"].append(overview_hidden)
            out["r448_float"].append(r448_float)
            out["r448_u8"].append(r448_u8)
            out["post_112"].append(post["112"])
            out["post_112_hidden"].append(l_hidden)
            out["post_224"].append(post["224"])
            out["post_224_hidden"].append(lh_hidden)
            out["reset_proprio"].append(observation.robot_state[0].astype(np.float64))
            out["post_state"].append(post_states["112"])
            out["applied"].append(np.stack([applied[k] for k in INSTANCES]))
    finally:
        views.close()
        for robot in robots.values():
            robot.sim.close()
    frames = {k: np.stack(v) for k, v in out.items()}
    frames["applied"] = frames["applied"].transpose(1, 0, 2, 3)  # (instances, roots, steps, 14)
    return rows, frames


def derived_dx_post_look(rows, predictions: dict) -> dict:
    """Reported only: the expert's post-look dx if the apple were at each PREDICTED xy."""
    from embodied_jepa.scripted import apple_collector_policy

    robot = make_robot(SMALL)
    values = {name: [] for name in predictions}
    try:
        for i, row in enumerate(rows):
            robot.reset(seed=row["seed"], **row["reset"])
            truth = robot.sim.task_truth()
            orp.execute_look(robot)
            for name, xy_pred in predictions.items():
                moved = dict(truth)
                moved["object_position"] = np.array(
                    [xy_pred[i][0], xy_pred[i][1], truth["object_position"][2]]
                )
                values[name].append(float(apple_collector_policy(moved).action(robot)[6]))
    finally:
        robot.sim.close()
    return {name: np.asarray(v) for name, v in values.items()}


def pixel_summary(values) -> dict:
    values = np.asarray(values)
    visible = values[values > 0]
    return {
        "zero": int((values == 0).sum()),
        "visible": int(visible.size),
        "quantiles_min_p25_p50_p75_p90_max": np.percentile(
            values, [0, 25, 50, 75, 90, 100]
        ).tolist(),
        "visible_median": float(np.median(visible)) if visible.size else None,
    }


# ----- the run -------------------------------------------------------------------------------
SOURCES = {
    # name: (arm, target set, decisional hypothesis or None)
    "anchor_raw112": ("anchor", "reset", None),
    "L_raw": ("L", "post", "L-raw"),
    "L_E0": ("L", "post", "L-E0"),
    "L_A3": ("L", "post", None),
    "L_random": ("L", "post", None),
    "LH_raw": ("LH", "post", "LH-raw"),
    "O_raw": ("O", "reset", "O-raw"),
    "O_E0": ("O", "reset", None),
    "H_raw": ("H", "reset", None),
    "R_float": ("R", "reset", None),
    "R_u8": ("R", "reset", None),
}
HYPOTHESIS_SOURCE = {h: s for s, (_a, _t, h) in SOURCES.items() if h}
HIDDEN_FRAMES = {
    "L-raw": "post_112_hidden",
    "LH-raw": "post_224_hidden",
    "O-raw": "overview_hidden",
}


def run(output: Path, *, smoke: bool = False) -> dict:
    from embodied_jepa.training import peak_rss_bytes, source_identity

    clock = BASE.Clock(GLOBAL_WALL_SECONDS)
    report = {
        "protocol": PROTOCOL,
        "task": TASK,
        "smoke": smoke,
        "source": source_identity(),
        "environment": BASE.environment(),
        "status": "running",
        "runner_sha256": {
            "scripts/probe_observation_reprobe.py": sha256(Path(__file__)),
            "src/embodied_jepa/observation_reprobe.py": sha256(Path(orp.__file__)),
        },
    }
    manifest = json.loads(MANIFEST.read_text())
    try:
        # --- G-hash (inputs), clean tree ---
        actual = {
            name: sha256(ROOT / name) if (ROOT / name).is_file() else None
            for name in manifest["hashes"]
        }
        ic.check_hashes(manifest["hashes"], actual)
        report["hashes"] = actual
        if not smoke:
            ic.check_clean_tree(report["source"]["dirty"])
        task059 = json.loads(TASK059_MANIFEST.read_text())
        want_look = manifest["look_motion"]["sequence_sha256_float32_bytes"]
        if want_look != orp.LOOK_SEQUENCE_SHA256:
            raise orp.GuardError("G-look: module look hash differs from the manifest")
        # --- G-split ---
        roots, _collector = plan_roots()
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
        reader = BASE.FrameReader(dataset, [r["episode_id"] for r in roots])
        # --- frames ---
        rows, frames = measure(roots, reader, clock=clock)
        n = len(rows)
        # --- G-render, G-expert, G-look ---
        ic.check_render(
            [r["path"]["P0_reset_112_equals_stored"] for r in rows], frames["reset_proprio"]
        )
        ic.check_expert(
            [r["expert_step0"] for r in rows],
            [r["recorded_step0"] for r in rows],
            [r["aim_offset"] for r in rows],
            [r["apple_xy_label"] for r in rows],
            [r["apple_xy_truth"] for r in rows],
        )
        report["look"] = orp.check_look(
            sequence_sha=orp.sequence_sha256(orp.look_sequence()),
            want_sha=want_look,
            applied=frames["applied"],
            post_states=frames["post_state"],
            apple_xy_moves=[r["apple_xy_move_m"] for r in rows],
            plate_moves=[r["plate_move_m"] for r in rows],
            contacts=[r["hand_contact"] for r in rows],
            steps_checked=[r["look_steps_checked"] for r in rows],
        ) | {
            "sequence_sha256": orp.sequence_sha256(orp.look_sequence()),
            "apple_z_move_min_max_m": [
                min(r["apple_z_move_m"] for r in rows),
                max(r["apple_z_move_m"] for r in rows),
            ],
            "expert_post_look_224_equal": int(sum(r["expert_post_look_224_equal"] for r in rows)),
        }
        xy = np.array([r["apple_xy_truth"] for r in rows], np.float64)
        targets = {
            "reset": (
                np.array([r["expert_step0"][0] for r in rows], np.float64),
                np.array([r["expert_step0"][1] for r in rows], np.float64),
            ),
            "post": (
                np.array([r["expert_post_look"][0] for r in rows], np.float64),
                np.array([r["expert_post_look"][1] for r in rows], np.float64),
            ),
        }
        train_mask = np.array([r["split"] == "train" for r in rows])
        occ059 = np.array([r["apple_pixels_reset"]["onboard_112"] == 0 for r in rows])
        # --- G-prior ---
        priors_summary = {
            t: ic.prior_summary(xy, dx, occ059, fold, train_mask)
            for t, (dx, _dy) in targets.items()
        }
        if not smoke:
            orp.check_priors(
                priors_summary["reset"],
                task059["calibration"]["values"]["prior_baselines"],
                "reset targets vs TASK-059",
            )
            orp.check_priors(
                priors_summary["post"],
                manifest["calibration"]["values"]["prior_baselines"]["post_look_targets"],
                "post-look targets",
            )
        report["prior_baselines"] = priors_summary
        report["guards"] = "passed"
        # --- render-path validation (demotes, never voids) ---
        path_counts = {k: int(sum(r["path"][k] for r in rows)) for k in rows[0]["path"]}
        validated = orp.validated_arms(path_counts, n)
        ablation_counts = {
            k: int(sum(r["ablation_reproduces"][k] for r in rows)) for k in ("L", "LH", "O")
        }
        report["render_path_validation"] = {
            "counts": path_counts,
            "roots": n,
            "validated_arms": validated,
            "ablation_renderer_reproduces": ablation_counts,
        }
        if smoke:  # hide every real target from the readouts
            noise = np.random.default_rng(0)
            xy = xy.mean(0) + noise.normal(0, 0.017, xy.shape)
            for t in targets:
                dx = np.clip(noise.normal(0.1, 0.4, n), -0.4, 0.4)
                dx[dx == 0] = 0.1
                targets[t] = (dx, noise.normal(-0.39, 0.02, n))
            report["smoke_targets"] = "seeded noise; no real target reached a readout"
        clock.check("features")
        # --- sources ---
        features, digests = BASE.encoder_features(
            {
                "post_112": frames["post_112"],
                "post_112_hidden": frames["post_112_hidden"],
                "overview": frames["overview"],
            },
            task059,
        )
        report["encoder_weights_sha256"] = digests
        flat = lambda a: a.reshape(n, -1) / 255.0  # noqa: E731
        sources = {
            "anchor_raw112": flat(frames["stored"]),
            "L_raw": flat(frames["post_112"]),
            "L_E0": features[("E0", "post_112")],
            "L_A3": features[("A3", "post_112")],
            "L_random": features[("random", "post_112")],
            "LH_raw": flat(frames["post_224"]),
            "O_raw": flat(frames["overview"]),
            "O_E0": features[("E0", "overview")],
            "H_raw": flat(frames["reset_224"]),
            "R_float": flat(frames["r448_float"]),
            "R_u8": flat(frames["r448_u8"]),
        }
        hidden = {
            "L-raw": flat(frames["post_112_hidden"]),
            "L-E0": features[("E0", "post_112_hidden")],
            "LH-raw": flat(frames["post_224_hidden"]),
            "O-raw": flat(frames["overview_hidden"]),
        }
        split_fold = np.where(train_mask, 1, 0)
        priors = {
            t: ic.prior_predictions(xy, dx, occ059, fold, dy=dy) for t, (dx, dy) in targets.items()
        }
        split_priors = {
            t: ic.prior_predictions(xy, dx, occ059, split_fold, dy=dy)
            for t, (dx, dy) in targets.items()
        }
        arm_flags = {
            "L": np.array([r["apple_pixels_post_look"]["onboard_112"] == 0 for r in rows]),
            "LH": np.array([r["apple_pixels_post_look"]["onboard_224"] == 0 for r in rows]),
            "O": np.array([r["apple_pixels_reset"]["overview_112"] == 0 for r in rows]),
            "H": np.array([r["apple_pixels_reset"]["onboard_224"] == 0 for r in rows]),
        }
        base_masks = {"all": np.ones(n, bool), "reset_occluded": occ059, "reset_visible": ~occ059}
        results, kept, preds = {}, {}, {}
        for name, x in sources.items():
            clock.check(f"readout {name}")
            arm, target_set, hypothesis = SOURCES[name]
            dx, dy = targets[target_set]
            masks = {k: m for k, m in base_masks.items() if m.any()}
            flag = arm_flags.get(arm)
            if flag is not None and flag.any() and (~flag).any():
                masks |= {"arm_occluded": flag, "arm_visible": ~flag}
            g, diag = ic.gram(x)
            results[name], kept[name], preds[name] = BASE.evaluate_source(
                name,
                g,
                diag,
                xy,
                dx,
                dy,
                fold,
                priors[target_set],
                masks,
                train_mask,
                split_priors[target_set],
            )
            results[name] |= {
                "arm": arm,
                "target_set": target_set,
                "decisional_hypothesis": hypothesis,
                "feature_dim": int(x.shape[1]),
            }
            if flag is not None:
                results[name]["B_occ_arm_reported"] = {
                    "arm_occluded_roots": int(flag.sum()),
                    "median_cm": float(
                        np.median(ic.xy_error_cm(orp.arm_occ_prior(xy, flag, fold), xy))
                    ),
                }
        # --- Holm over the four decisional hypotheses (all roots) ---
        everyone = np.ones(n, bool)
        pvalues = {}
        for hypothesis, name in HYPOTHESIS_SOURCE.items():
            dx, _dy = targets[SOURCES[name][1]]
            pvalues[hypothesis] = orp.hypothesis_pvalues(
                preds[name]["xy"], preds[name]["dx"], xy, dx, priors[SOURCES[name][1]], everyone
            )
        holm = orp.holm({h: v["p"] for h, v in pvalues.items()})
        # --- spurious check, always run on all four ---
        spurious = {}
        for hypothesis, name in HYPOTHESIS_SOURCE.items():
            clock.check(f"spurious {hypothesis}")
            target_set = SOURCES[name][1]
            dx, _dy = targets[target_set]
            arm = orp.ARM_OF[hypothesis]
            reproduces = ablation_counts[arm] == n
            hidden_eval = None
            if reproduces:
                g_new, norms = ic.cross_gram(hidden[hypothesis], sources[name])
                xy_hat = orp.predict_with_fold_readouts(preds[name]["xy_fits"], fold, g_new, norms)
                dx_hat = orp.predict_with_fold_readouts(preds[name]["dx_fits"], fold, g_new, norms)
                hidden_eval = ic.evaluate(
                    xy_hat, dx_hat[:, 0], xy, dx, priors[target_set], everyone
                )
            spurious[hypothesis] = orp.spurious_verdict(
                hidden_eval, renderer_reproduces=reproduces
            ) | {"hidden_all_roots": ic.public(hidden_eval) if hidden_eval else None}
        # --- pass rule and decision ---
        hypotheses = {}
        for hypothesis, name in HYPOTHESIS_SOURCE.items():
            arm = orp.ARM_OF[hypothesis]
            succeeds = bool(results[name]["all"]["succeeds"])
            rejected = bool(holm["rejected"][hypothesis])
            hypotheses[hypothesis] = {
                "source": name,
                "render_path_validated": bool(validated[arm]),
                "succeeds_unadjusted_all_roots": succeeds,
                "pvalues": pvalues[hypothesis],
                "holm_rejected": rejected,
                "spurious_check": spurious[hypothesis],
                "beats_prior_all_roots": bool(results[name]["all"]["beats_prior"]),
                "passes": orp.passes(
                    validated=validated[arm],
                    succeeds=succeeds,
                    rejected=rejected,
                    spurious=spurious[hypothesis]["spurious"],
                ),
            }
            hypotheses[hypothesis]["qualifier"] = (
                "pass"
                if hypotheses[hypothesis]["passes"]
                else "unadjusted only; not a pass"
                if succeeds
                else "partial information"
                if hypotheses[hypothesis]["beats_prior_all_roots"]
                else None
            )
        report["hypotheses"] = hypotheses
        report["holm"] = holm
        report["decision"] = orp.decide(
            void=False, passed={h: v["passes"] for h, v in hypotheses.items()}
        )
        # --- reported: random floor, pairwise comparisons, T4 ---
        results["L_E0"]["random_floor"] = {
            s: ic.beats_random_floor(kept["L_E0"][s], kept["L_random"][s])
            for s in kept["L_E0"]
            if s in kept["L_random"]
        }
        pairs = (
            ("L_raw", "anchor_raw112"),
            ("LH_raw", "L_raw"),
            ("L_E0", "L_raw"),
            ("O_raw", "anchor_raw112"),
            ("H_raw", "anchor_raw112"),
            ("R_u8", "R_float"),
            ("R_u8", "anchor_raw112"),
        )
        report["pairwise_reported"] = {
            f"{a}_minus_{b}": {
                "T2_targets_differ": SOURCES[a][1] != SOURCES[b][1],
                **{
                    s: ic.compare_sources(kept[a][s], kept[b][s])
                    for s in ("all", "reset_occluded", "reset_visible")
                    if s in kept[a] and s in kept[b]
                },
            }
            for a, b in pairs
        }
        clock.check("T4")
        post_names = [s for s in SOURCES if SOURCES[s][1] == "post"]
        derived_post = derived_dx_post_look(rows, {s: preds[s]["xy"] for s in post_names})
        for name in SOURCES:
            target_set = SOURCES[name][1]
            dx, dy = targets[target_set]
            derived = (
                derived_post[name]
                if target_set == "post"
                else BASE.derived_dx(rows, preds[name]["xy"])
            )
            results[name]["T4_reported"] = {
                s: ic.evaluate_reported(
                    preds[name]["xy"], preds[name]["dy"], derived, xy, dy, dx, priors[target_set], m
                )
                for s, m in base_masks.items()
                if m.any()
            }
        report["results"] = results
        report["descriptive"] = {
            "apple_pixels": {
                "reset": {
                    k: pixel_summary([r["apple_pixels_reset"][k] for r in rows])
                    for k in rows[0]["apple_pixels_reset"]
                },
                "post_look": {
                    k: pixel_summary([r["apple_pixels_post_look"][k] for r in rows])
                    for k in rows[0]["apple_pixels_post_look"]
                },
            },
        }
        report["per_root"] = [
            {
                "seed": r["seed"],
                "split": r["split"],
                "fold": int(fold[i]),
                "reset_occluded": bool(occ059[i]),
                "apple_pixels_reset": r["apple_pixels_reset"],
                "apple_pixels_post_look": r["apple_pixels_post_look"],
                "apple_xy": r["apple_xy_truth"],
                "expert_dx_reset": r["expert_step0"][0],
                "expert_dx_post_look": r["expert_post_look"][0],
                "expert_dy_post_look": r["expert_post_look"][1],
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
        report["outcome"] = report["decision"]["outcome"]
    except (ic.GuardError, VoidRun) as error:
        report["status"] = "void"
        report["decision"] = {"outcome": "V", "reason": str(error)}
        report["outcome"] = "V"
        report["void_reason"] = f"guard: {error}"
    except Exception as error:  # noqa: BLE001 - a crash is recorded as V, then re-raised
        report["status"] = "void: crashed before the report was complete"
        report["decision"] = {"outcome": "V", "reason": f"{type(error).__name__}: {error}"}
        report["outcome"] = "V"
        report["void_reason"] = f"exception: {type(error).__name__}: {error}"
        report["traceback"] = "".join(traceback.format_exception(error))
        report["elapsed_seconds"] = clock.elapsed()
        report["peak_rss_bytes"] = peak_rss_bytes()
        write_report(output / "report.json", report)
        raise
    report["elapsed_seconds"] = clock.elapsed()
    report["peak_rss_bytes"] = peak_rss_bytes()
    write_report(output / "report.json", report)
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
