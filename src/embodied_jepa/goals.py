"""Freeze image goals and resets before model evaluation, verifying every image hash."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

import numpy as np

from embodied_jepa.collection import reach_goal
from embodied_jepa.embodiment import G1Embodiment
from embodied_jepa.simulation import MuJoCoSimulation
from embodied_jepa.task import TaskThresholds

RUNTIME_SOURCES = (
    "simulation.py",
    "embodiment.py",
    "task.py",
    "goals.py",
    "collection.py",
    "contracts.py",
)
PAIRS = {"reach": ("cube", "target"), "apple_to_plate": ("apple", "plate")}


def _seeds(seeds):
    try:
        seeds = list(seeds)
    except TypeError as error:
        raise ValueError("goal seeds must be distinct nonnegative integers") from error
    if (
        not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("goal seeds must be distinct nonnegative integers")
    return seeds


def _vector(value, size, name):
    if (
        not isinstance(value, list)
        or len(value) != size
        or any(type(x) not in (int, float) for x in value)
    ):
        raise ValueError(f"{name} must be a finite {size}D numeric vector")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _reset(reset, seed, task):
    names = {"seed", "object_xy", "plate_xy", "object_kind", "container_kind"}
    if not isinstance(reset, dict) or set(reset) != names:
        raise ValueError("frozen reset fields are incomplete or unexpected")
    if type(reset["seed"]) is not int or reset["seed"] != seed:
        raise ValueError("frozen reset seed mismatch")
    if (reset["object_kind"], reset["container_kind"]) != PAIRS[task]:
        raise ValueError("frozen reset pair does not match task")
    object_xy = _vector(reset["object_xy"], 2, "object_xy")
    plate_xy = _vector(reset["plate_xy"], 2, "plate_xy")
    for xy in (object_xy, plate_xy):
        if not (0.18 <= xy[0] <= 0.65 and -0.32 <= xy[1] <= 0.32):
            raise ValueError("frozen reset coordinates must lie on the tabletop")
    # Same conservative collision envelopes as the pinned simulator's default
    # unsupported tabletop reset. A final goal cannot start already placed.
    offset = np.abs(object_xy - plate_xy)
    separation = (
        np.linalg.norm(np.maximum(offset - 0.07, 0)) - np.sqrt(2) * 0.023
        if task == "reach"
        else np.linalg.norm(offset) - 0.071 - 0.027
    )
    if separation < 0.005:
        raise ValueError("frozen reset object/container collision envelopes overlap")


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _pixels(path, entry):
    from PIL import Image, UnidentifiedImageError

    if not isinstance(entry, dict):
        raise ValueError("goal entry must be an object")
    name, expected = entry.get("image"), entry.get("sha256")
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or Path(name).suffix != ".png"
        or name in (".", "..")
    ):
        raise ValueError("goal image must be a local PNG filename")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("goal image requires a SHA-256 digest")
    directory = Path(path).resolve().parent
    image_path = directory / name
    if image_path.resolve().parent != directory:
        raise ValueError("goal image path escapes manifest directory")
    try:
        payload = image_path.read_bytes()
    except OSError as error:
        raise ValueError("goal image is missing or unreadable") from error
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("goal image hash mismatch")
    # Decode the exact bytes hashed above, closing the validation/read race.
    try:
        with Image.open(BytesIO(payload)) as image:
            if (
                image.format != "PNG"
                or image.mode != "RGB"
                or not all(1 <= side <= 4096 for side in image.size)
            ):
                raise ValueError("goal image must be bounded RGB PNG")
            pixels = np.array(image)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError("invalid goal PNG") from error
    return {"onboard_rgb": pixels[None]}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_goals(directory, seeds, *, task="apple_to_plate"):
    from PIL import Image

    if not isinstance(task, str) or task not in PAIRS:
        raise ValueError("unsupported goal task")
    seeds = _seeds(seeds)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    manifest = {
        "schema_version": 1,
        "task": task,
        "seeds": seeds,
        "thresholds": asdict(TaskThresholds()),
        "complete": False,
        "reset_distribution": {
            "version": "manipulation_training_xy_v0",
            "object_center_xy_m": [0.34, -0.18],
            "container_center_xy_m": [0.49, -0.09],
            "independent_uniform_jitter_m": 0.006,
            "rng": "numpy.default_rng(seed), object XY then container XY",
            "appearance": "same procedural apple/plate appearances seen separately in training",
        }
        if task == "apple_to_plate"
        else {"version": "reach_default_xy_v0"},
        "asset_manifest_sha256": digest(root / "assets/manifest.json"),
        "action_manifest_sha256": digest(root / "configs/g1_sim_action.json"),
        "runtime_source_hashes": {
            name: digest(Path(__file__).with_name(name)) for name in RUNTIME_SOURCES
        },
        "episodes": [],
    }
    manifest_path = directory / "manifest.json"
    # Preserve the planned cohort and incomplete status if generation fails.
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    for seed in seeds:
        kinds = PAIRS[task]
        sim = MuJoCoSimulation(object_kind=kinds[0], container_kind=kinds[1])
        robot = G1Embodiment(sim)
        try:
            if task == "apple_to_plate":
                rng = np.random.default_rng(seed)
                truth = robot.reset(
                    seed,
                    object_xy=np.array([0.34, -0.18]) + rng.uniform(-0.006, 0.006, 2),
                    plate_xy=np.array([0.49, -0.09]) + rng.uniform(-0.006, 0.006, 2),
                )
            else:
                truth = robot.reset(seed)
            reset = {
                "seed": seed,
                "object_xy": truth["object_position"][:2].tolist(),
                "plate_xy": truth["plate_position"][:2].tolist(),
                "object_kind": kinds[0],
                "container_kind": kinds[1],
            }
            target = None
            if task == "reach":
                target = robot.ee_pose("right")[0] + np.random.default_rng(seed).uniform(
                    [-0.015, -0.045, -0.03], [0.025, -0.025, 0.025]
                )
                oracle = reach_goal(robot, target)
                if not oracle["success"]:
                    raise RuntimeError(f"goal {seed} is not reachable by the oracle")
            else:
                robot.reset(
                    seed,
                    object_xy=truth["plate_position"][:2],
                    plate_xy=truth["plate_position"][:2],
                    object_on_container=True,
                )
                for _ in range(30):
                    robot.observe()
                    action = np.zeros(14, np.float32)
                    action[12:] = -1
                    result = robot.execute(action)
                    if result.applied_action is None:
                        raise RuntimeError(f"goal settling failed: {result.reason}")
                if not sim.task_truth()["placed"]:
                    raise RuntimeError(f"goal {seed} did not settle on plate")
            path = directory / f"goal-{seed}.png"
            Image.fromarray(robot.observe().images["onboard_rgb"][0]).save(path)
            manifest["episodes"].append(
                {
                    "seed": seed,
                    "reset": reset,
                    "target_base": None if target is None else target.tolist(),
                    "image": path.name,
                    "sha256": digest(path),
                }
            )
        finally:
            robot.close()
    manifest["complete"] = True
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(manifest_path)
    return manifest_path


def load_goals(path, *, task, seeds, action_manifest):
    path = Path(path)
    seeds = _seeds(seeds)
    if not isinstance(task, str) or task not in PAIRS:
        raise ValueError("unsupported goal task")
    manifest = json.loads(
        path.read_text(), parse_constant=_reject_constant, object_pairs_hook=_unique_object
    )
    root = Path(__file__).resolve().parents[2]
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or manifest.get("complete") is not True
        or manifest.get("task") != task
        or manifest.get("seeds") != list(seeds)
    ):
        raise ValueError("goal manifest task/seeds/completion mismatch")
    _seeds(manifest["seeds"])
    if manifest.get("thresholds") != asdict(TaskThresholds()):
        raise ValueError("frozen goal thresholds differ from runtime")
    if manifest.get("action_manifest_sha256") != digest(action_manifest):
        raise ValueError("frozen goal action manifest changed")
    if manifest.get("asset_manifest_sha256") != digest(root / "assets/manifest.json"):
        raise ValueError("frozen goal assets changed")
    hashes = manifest.get("runtime_source_hashes")
    if not isinstance(hashes, dict) or set(hashes) != set(RUNTIME_SOURCES):
        raise ValueError("goal manifest must pin the complete runtime source set")
    for name, expected in hashes.items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"frozen goal runtime changed: {name}")
    episodes = manifest.get("episodes")
    if (
        not isinstance(episodes, list)
        or any(not isinstance(e, dict) or type(e.get("seed")) is not int for e in episodes)
        or [e["seed"] for e in episodes] != seeds
    ):
        raise ValueError("frozen goal episodes incomplete")
    images = set()
    for entry in episodes:
        if set(entry) != {"seed", "reset", "target_base", "image", "sha256"}:
            raise ValueError("frozen goal episode fields are incomplete or unexpected")
        _reset(entry["reset"], entry["seed"], task)
        if task == "reach":
            target = _vector(entry["target_base"], 3, "target_base")
            workspace = json.loads(Path(action_manifest).read_text())["workspace_base_m"]["right"]
            if np.any(target < workspace[0]) or np.any(target > workspace[1]):
                raise ValueError("frozen reach target is outside the calibrated workspace")
        elif entry["target_base"] is not None:
            raise ValueError("full task must not supply a privileged reach target")
        _pixels(path, entry)
        if entry["image"] in images:
            raise ValueError("goal entries must use distinct image files")
        images.add(entry["image"])
    return manifest


def goal_pixels(path, entry):
    return _pixels(path, entry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", choices=("reach", "apple_to_plate"), default="apple_to_plate")
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    print(
        freeze_goals(
            args.output, range(args.seed_start, args.seed_start + args.episodes), task=args.task
        )
    )
