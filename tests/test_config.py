import json

import pytest
import yaml

from embodied_jepa.config import ExperimentConfig


@pytest.fixture
def config_path(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "native.pt").touch()
    (tmp_path / "lewm.pt").touch()
    (tmp_path / "action.json").write_text(
        json.dumps(
            {
                "action_schema": "ee_delta_grasp_v0",
                "translation_per_step_m": 0.01,
                "rotation_per_step_rad": 0.05,
                "control_dt_s": 0.05,
                "joint_speed_limit_rad_s": 1,
            }
        )
    )
    raw = {
        "schema_version": 0,
        "world_model": {
            "backend": "native_jepa",
            "checkpoints": {"native_jepa": "native.pt", "leworldmodel": "lewm.pt"},
        },
        "embodiment": {
            "backend": "unitree_g1_dex3",
            "transport": "mujoco",
            "action_manifest": "action.json",
        },
        "dataset": {"root": "data"},
        "task": {"name": "reach"},
    }
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_backend_only_override_keeps_shared_experiment(config_path):
    original = ExperimentConfig.load(config_path)
    override = config_path.with_name("lewm.yaml")
    override.write_text("extends: base.yaml\nworld_model:\n  backend: leworldmodel\n")
    other = ExperimentConfig.load(override)
    assert other.backend == "leworldmodel"
    assert other.checkpoint.name == "lewm.pt"
    assert original.planner == other.planner
    assert original.evaluation_seeds == other.evaluation_seeds
    assert original.dataset_root == other.dataset_root
    assert original.action_manifest == other.action_manifest


@pytest.mark.parametrize(
    "update,match",
    [
        ({"surprise": 1}, "unknown"),
        ({"world_model": {"backend": "fake"}}, "unknown"),
        ({"seed": True}, "seed"),
        ({"world_model": {"history": 3}}, "history"),
        ({"dataset": {"root": None}}, "path"),
        ({"evaluation": {"seeds": [1, 1]}}, "distinct"),
        ({"embodiment": {"transport": "sdk2"}}, "MuJoCo"),
    ],
)
def test_invalid_config_fails_before_execution(config_path, update, match):
    raw = yaml.safe_load(config_path.read_text())
    for key, value in update.items():
        if isinstance(value, dict):
            raw.setdefault(key, {}).update(value)
        else:
            raw[key] = value
    config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=match):
        ExperimentConfig.load(config_path)


def test_missing_physical_scale_rejected(config_path):
    action = config_path.with_name("action.json")
    data = json.loads(action.read_text())
    data["translation_per_step_m"] = None
    action.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="translation"):
        ExperimentConfig.load(config_path)


def test_inheritance_cycle_rejected(config_path):
    config_path.write_text("extends: base.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        ExperimentConfig.load(config_path)


def test_cross_directory_inheritance_preserves_path_origins(config_path):
    original = ExperimentConfig.load(config_path)
    child_dir = config_path.parent / "nested"
    child_dir.mkdir()
    child = child_dir / "run.yaml"
    child.write_text("extends: ../base.yaml\n")
    inherited = ExperimentConfig.load(child)
    assert inherited.dataset_root == original.dataset_root
    assert inherited.action_manifest == original.action_manifest
    assert inherited.checkpoint == original.checkpoint


def test_saved_resolved_config_can_be_reloaded(config_path):
    original = ExperimentConfig.load(config_path)
    saved = config_path.with_name("resolved.json")
    original.save(saved)
    restored = ExperimentConfig.load(saved)
    assert restored == original


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_paths_not_resolved_to_existing_directory(config_path, value):
    raw = yaml.safe_load(config_path.read_text())
    raw["dataset"]["root"] = value
    config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="path"):
        ExperimentConfig.load(config_path)
