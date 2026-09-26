"""TASK-062 pre-freeze calibration: where does frozen E0 lose the apple? Nothing is fitted.

A **label-free** diagnostic, run before any encoder is trained and before any readout is fitted
on any new feature. It reads no target (no apple xy, no expert command) and fits no readout. Its
only inputs are rendered frames and encoder activations.

For each of the 190 train + val roots it renders the TASK-061 post-look 112 px onboard frame
(the same reset, warm-up and 8-command look as ``scripts/probe_observation_reprobe.py``) and two
ablations of it: the apple hidden, and the plate hidden (geoms moved to a hidden group, so they
are neither drawn nor cast a shadow). The robot pose is identical on every root after the look,
so every difference between two roots' frames comes from the apple and the plate.

For every representation r -- raw pixels, and, for frozen E0 and the random-init encoder of the
same architecture (TASK-061's L-random), each ViT stage: the patch embedding, the tokens after
each block, the final normed tokens, the CLS token, the projector output and the fused probe
feature -- it reports:

* **apple share** ``S_apple = median_i ||r(x_i) - r(h_i)|| / median_{i<j} ||r(x_i) - r(x_j)||``,
  where ``h_i`` is root i's apple-hidden frame: how large the apple's own effect is compared with
  the typical difference between two roots. The same with the plate hidden gives ``S_plate``.
* **apple energy ratio** ``R_apple = sum_i ||r(x_i) - r(h_i)||^2 / sum_i ||r(x_i) - mean r(x)||^2``:
  the energy of the apple's own effect relative to the across-root variance (``R_plate`` with
  the plate hidden). It is a ratio, not a fraction: hiding the plate can exceed 1.
* **effective rank** (exp of the spectral entropy) of r over the 190 post-look frames.

    uv run --no-sync python scripts/calibrate_encoder_study.py \
        --output outputs/task062-encoder-study/calibration.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import info_ceiling as ic  # noqa: E402
from embodied_jepa import observation_reprobe as orp  # noqa: E402

STORE = ROOT / "data" / "apple-wide-v1"
TASK061_MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-observation-reprobe-v1.json"
SMALL = 112


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load("_probe_info_ceiling", "scripts/probe_info_ceiling.py")
REPROBE = _load("_probe_observation_reprobe", "scripts/probe_observation_reprobe.py")


def hide_render(robot, renderer, bodies, camera="onboard_rgb"):
    """Render with every geom of ``bodies`` moved to the hidden group (not drawn, no shadow)."""
    mj, model = robot.mj, robot.model
    ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, b) for b in bodies]
    geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in ids]
    if not geoms:
        raise RuntimeError(f"no geoms for bodies {bodies}")
    option = mj.MjvOption()
    saved = [int(model.geom_group[g]) for g in geoms]
    try:
        for g in geoms:
            model.geom_group[g] = BASE.HIDDEN_GROUP
        option.geomgroup[BASE.HIDDEN_GROUP] = 0
        renderer.update_scene(robot.sim.data, camera=camera, scene_option=option)
        return renderer.render().copy()
    finally:
        for g, group in zip(geoms, saved, strict=True):
            model.geom_group[g] = group


def plate_bodies(robot):
    mj, model = robot.mj, robot.model
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)]
    found = [n for n in names if n and ("plate" in n or "container" in n)]
    if not found:
        raise RuntimeError(f"no plate body among {names}")
    return found


def render_post_look(roots):
    """Post-look frame, apple-hidden and plate-hidden re-renders for every root."""
    robot = REPROBE.make_robot(SMALL)
    mj = robot.mj
    renderer = mj.Renderer(robot.model, height=SMALL, width=SMALL)
    REPROBE.warm_up(renderer, robot.sim.data)
    plates = plate_bodies(robot)
    out = {"post": [], "apple_hidden": [], "plate_hidden": [], "reproduces": []}
    try:
        for root in roots:
            robot.reset(seed=root["seed"], object_xy=root["object_xy"], plate_xy=root["plate_xy"])
            robot.observe()
            orp.execute_look(robot)
            post = robot.observe().images["onboard_rgb"][0].copy()
            renderer.update_scene(robot.sim.data, camera="onboard_rgb")
            plain = renderer.render().copy()
            out["reproduces"].append(bool(orp.frames_identical(post, plain)))
            out["post"].append(post)
            out["apple_hidden"].append(hide_render(robot, renderer, ["apple"]))
            out["plate_hidden"].append(hide_render(robot, renderer, plates))
    finally:
        renderer.close()
        robot.sim.close()
    return {k: (np.stack(v) if k != "reproduces" else v) for k, v in out.items()}, plates


def encoder_stages(model, frames):
    """Every ViT stage of a LeWM ``VisualModel`` for uint8 frames [N,112,112,3] (float64)."""
    import torch

    stages = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(frames), 32):
            batch = {"onboard_rgb": frames[start : start + 32]}
            pixels, _ = model.pixels(batch, camera="onboard_rgb")
            mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            output = model.model.encoder(
                (pixels - mean) / std, interpolate_pos_encoding=True, output_hidden_states=True
            )
            hidden = output.hidden_states
            parts = {"patch_embedding_tokens": hidden[0][:, 1:]}
            for k in range(1, len(hidden)):
                parts[f"block{k}_tokens"] = hidden[k][:, 1:]
            parts["final_tokens"] = output.last_hidden_state[:, 1:]
            parts["final_tokens_mean"] = output.last_hidden_state[:, 1:].mean(1)
            parts["cls"] = output.last_hidden_state[:, 0]
            parts["projector"] = model.model.projector(output.last_hidden_state[:, 0])
            fused, _ = model.image_features(batch)
            parts["probe_feature"] = fused
            for name, value in parts.items():
                stages.setdefault(name, []).append(
                    value.reshape(len(value), -1).cpu().numpy().astype(np.float64)
                )
    return {k: np.concatenate(v) for k, v in stages.items()}


def effective_rank(x) -> float:
    x = x - x.mean(0)
    s = np.linalg.svd(x, compute_uv=False)
    p = s**2 / max(float((s**2).sum()), 1e-300)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def shares(post, hidden_apple, hidden_plate) -> dict:
    n = len(post)
    iu = np.triu_indices(n, 1)
    sq = (post**2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None] - 2 * post @ post.T, 0.0)
    between = float(np.median(np.sqrt(d2[iu])))
    apple = np.linalg.norm(post - hidden_apple, axis=1)
    plate = np.linalg.norm(post - hidden_plate, axis=1)
    total_var = float(((post - post.mean(0)) ** 2).sum() / n)
    return {
        "between_roots_median_distance": between,
        "S_apple": float(np.median(apple) / between) if between > 0 else None,
        "S_plate": float(np.median(plate) / between) if between > 0 else None,
        "R_apple": float((apple**2).sum() / (n * total_var)) if total_var > 0 else None,
        "R_plate": float((plate**2).sum() / (n * total_var)) if total_var > 0 else None,
        "effective_rank": effective_rank(post),
        "dimension": int(post.shape[1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    begin = time.time()
    roots, _collector = BASE.plan_roots()
    dataset = json.loads((STORE / "meta" / "jepa_manifest.json").read_text())
    ic.check_split(roots, dataset["splits"])
    frames, plates = render_post_look(roots)
    task061 = json.loads(TASK061_MANIFEST.read_text())
    measure = _load("_measure", "scripts/measure_policy_offline_conditionals.py")
    from embodied_jepa.world_model_v2 import _open

    config, store, _settings, _cameras = _open(BASE.CONFIG)
    result = {
        "task": "TASK-062",
        "what": "label-free encoder diagnostic; no target read, no readout fitted",
        "roots": len(roots),
        "plate_bodies_hidden": plates,
        "ablation_renderer_reproduces_post_look": int(sum(frames["reproduces"])),
        "post_look_frames_sha256": hashlib.sha256(frames["post"].tobytes()).hexdigest(),
        "representations": {},
    }
    flat = lambda a: a.reshape(len(a), -1).astype(np.float64) / 255.0  # noqa: E731
    result["representations"]["raw_pixels"] = shares(
        flat(frames["post"]), flat(frames["apple_hidden"]), flat(frames["plate_hidden"])
    )
    for name, arm in (("E0", "a2"), ("random", "a1")):
        _policy, source = measure.build_policy(BASE.CHECKPOINTS / f"{arm}.pt", config, store, "cpu")
        digest = source.weights_sha256()
        want = task061["encoders"][name]["encoder_weights_sha256"]
        ic.check_encoder_digest(name, digest, want)
        stages = {
            k: encoder_stages(source.model, frames[k])
            for k in ("post", "apple_hidden", "plate_hidden")
        }
        result["representations"][name] = {
            stage: shares(
                stages["post"][stage], stages["apple_hidden"][stage], stages["plate_hidden"][stage]
            )
            for stage in stages["post"]
        }
        result["representations"][name]["_weights_sha256"] = digest
    result["elapsed_seconds"] = time.time() - begin
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["representations"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
