"""Frozen reset/image integrity and optional real renderer round trips."""

import copy
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from embodied_jepa import goals  # noqa: E402
from embodied_jepa.task import TaskThresholds  # noqa: E402


@pytest.fixture
def frozen(tmp_path):
    root = Path(goals.__file__).resolve().parents[2]
    image = tmp_path / "goal-1.png"
    Image.fromarray(np.full((8, 9, 3), 127, np.uint8)).save(image)
    manifest = {
        "schema_version": 1,
        "task": "reach",
        "seeds": [1],
        "complete": True,
        "thresholds": asdict(TaskThresholds()),
        "asset_manifest_sha256": goals.digest(root / "assets/manifest.json"),
        "action_manifest_sha256": goals.digest(root / "configs/g1_sim_action.json"),
        "runtime_source_hashes": {
            name: goals.digest(Path(goals.__file__).with_name(name))
            for name in goals.RUNTIME_SOURCES
        },
        "episodes": [
            {
                "seed": 1,
                "reset": {
                    "seed": 1,
                    "object_xy": [0.4, -0.26],
                    "plate_xy": [0.48, -0.10],
                    "object_kind": "cube",
                    "container_kind": "target",
                },
                "target_base": [0.3, -0.15, 0.1],
                "image": image.name,
                "sha256": goals.digest(image),
            }
        ],
    }
    path = tmp_path / "manifest.json"

    def save_load(value=None):
        path.write_text(json.dumps(manifest if value is None else value))
        return goals.load_goals(
            path, task="reach", seeds=[1], action_manifest=root / "configs/g1_sim_action.json"
        )

    return manifest, path, save_load


def test_goal_input_contains_only_verified_rgb_not_reset_or_scoring_truth(frozen):
    _, path, load = frozen
    manifest = load()
    pixels = goals.goal_pixels(path, manifest["episodes"][0])
    assert set(pixels) == {"onboard_rgb"}
    assert pixels["onboard_rgb"].shape == (1, 8, 9, 3)
    assert pixels["onboard_rgb"].dtype == np.uint8
    assert np.all(pixels["onboard_rgb"] == 127)


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), True),
        (("complete",), False),
        (("seeds",), [True]),
        (("runtime_source_hashes",), {}),
        (("runtime_source_hashes", "task.py"), "0" * 64),
        (("asset_manifest_sha256",), "0" * 64),
        (("action_manifest_sha256",), "0" * 64),
        (("episodes", 0, "seed"), True),
        (("episodes", 0, "reset", "seed"), 2),
        (("episodes", 0, "reset", "object_kind"), "apple"),
        (("episodes", 0, "reset", "container_kind"), "plate"),
        (("episodes", 0, "reset", "object_xy"), [0.48, -0.10]),
        (("episodes", 0, "reset", "object_xy"), [5.0, 0.0]),
        (("episodes", 0, "reset", "plate_xy"), [True, 0.0]),
        (("episodes", 0, "target_base"), None),
        (("episodes", 0, "target_base"), [0.0, 0.0]),
        (("episodes", 0, "target_base"), [0.3, -0.15, float("nan")]),
        (("episodes", 0, "target_base"), [0.3, -0.15, 10.0]),
        (("episodes", 0, "sha256"), "0" * 64),
        (("episodes", 0, "image"), "../outside.png"),
        (("episodes", 0, "image"), "/tmp/outside.png"),
    ],
)
def test_corrupt_or_ambiguous_manifests_rejected(frozen, path, value):
    manifest, _, load = frozen
    target = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        load()


def test_unexpected_reset_control_cannot_be_silently_ignored(frozen):
    manifest, _, load = frozen
    manifest["episodes"][0]["reset"]["object_on_container"] = True
    with pytest.raises(ValueError, match="reset fields"):
        load()


def test_manifest_episode_cohort_must_match_exact_order(frozen):
    manifest, _, load = frozen
    manifest["episodes"].append(copy.deepcopy(manifest["episodes"][0]))
    with pytest.raises(ValueError, match="episodes incomplete"):
        load()


def test_hash_is_rechecked_at_use_and_symlink_escape_rejected(frozen, tmp_path):
    _, path, load = frozen
    entry = load()["episodes"][0]
    image = path.parent / entry["image"]
    image.write_bytes(b"changed after initial validation")
    with pytest.raises(ValueError, match="hash mismatch"):
        goals.goal_pixels(path, entry)
    outside = tmp_path.parent / (tmp_path.name + "-outside.png")
    outside.write_bytes(b"external bytes")
    image.unlink()
    image.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        goals.goal_pixels(path, entry)


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_hashed_non_rgb_image_is_rejected(frozen, mode):
    manifest, path, load = frozen
    image = path.parent / manifest["episodes"][0]["image"]
    Image.new(mode, (8, 8)).save(image)
    manifest["episodes"][0]["sha256"] = goals.digest(image)
    with pytest.raises(ValueError, match="RGB PNG"):
        load()


@pytest.mark.parametrize("seeds", [[], [1, 1], [True], [-1], [[1]], None])
def test_invalid_generation_request_has_no_filesystem_side_effect(tmp_path, seeds):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="seeds"):
        goals.freeze_goals(output, seeds)
    assert not output.exists()


def test_failed_generation_preserves_incomplete_plan(tmp_path, monkeypatch):
    def failed_sim(**kwargs):
        raise RuntimeError("injected unavailable rendering runtime")

    monkeypatch.setattr(goals, "MuJoCoSimulation", failed_sim)
    directory = tmp_path / "goals"
    with pytest.raises(RuntimeError, match="unavailable"):
        goals.freeze_goals(directory, [20000])
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["seeds"] == [20000] and manifest["complete"] is False


def test_full_task_rejects_privileged_target_field(frozen):
    manifest, path, _ = frozen
    manifest["task"] = "apple_to_plate"
    manifest["episodes"][0]["reset"].update(object_kind="apple", container_kind="plate")
    path.write_text(json.dumps(manifest))
    root = Path(goals.__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="privileged reach target"):
        goals.load_goals(
            path,
            task="apple_to_plate",
            seeds=[1],
            action_manifest=root / "configs/g1_sim_action.json",
        )


@pytest.mark.parametrize("task,seed", [("reach", 10000), ("apple_to_plate", 20000)])
def test_actual_frozen_goal_and_reset_replay(tmp_path, task, seed):
    if os.getenv("JEPA_TEST_RENDER") != "1":
        pytest.skip("graphics opt-in")
    pytest.importorskip("mujoco")
    path = goals.freeze_goals(tmp_path / task, [seed], task=task)
    root = Path(goals.__file__).resolve().parents[2]
    manifest = goals.load_goals(
        path, task=task, seeds=[seed], action_manifest=root / "configs/g1_sim_action.json"
    )
    entry = manifest["episodes"][0]
    pixels = goals.goal_pixels(path, entry)
    assert pixels["onboard_rgb"].shape[-1] == 3
    reset = entry["reset"]
    if task == "apple_to_plate":
        rng = np.random.default_rng(seed)
        np.testing.assert_array_equal(
            reset["object_xy"], np.array([0.34, -0.18]) + rng.uniform(-0.006, 0.006, 2)
        )
        np.testing.assert_array_equal(
            reset["plate_xy"], np.array([0.49, -0.09]) + rng.uniform(-0.006, 0.006, 2)
        )
    sim = goals.MuJoCoSimulation(
        object_kind=reset["object_kind"], container_kind=reset["container_kind"]
    )
    try:
        first = sim.reset(seed, object_xy=reset["object_xy"], plate_xy=reset["plate_xy"])
        initial = sim.render()
        second = sim.reset(seed, object_xy=reset["object_xy"], plate_xy=reset["plate_xy"])
        replay = sim.render()
        # OpenGL may dither a handful of channels by one level between renders;
        # exact stored goal bytes are verified independently, never regenerated.
        difference = np.abs(initial.astype(np.int16) - replay.astype(np.int16))
        assert difference.max() <= 1
        assert np.count_nonzero(difference) / difference.size < 0.001
        np.testing.assert_array_equal(first["object_position"], second["object_position"])
        assert not first["placed"]  # Evaluation never begins already on the receptacle.
    finally:
        sim.close()
