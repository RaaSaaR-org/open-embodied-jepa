"""SO-101 S0 (TASK-096): the 6-D joint_delta_so101_v0 schema through contracts, data, config and
CEM. Software contract evidence on synthetic data only; no manipulation or learning claim."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import yaml

from embodied_jepa import config as config_module
from embodied_jepa.action_schemas import (
    ACTION_SCHEMAS,
    JOINT_DELTA_SO101_V0,
    SO101_ACTION_NAMES,
    action_schema,
    action_schema_from_record,
    schema_record,
)
from embodied_jepa.config import ExperimentConfig
from embodied_jepa.contracts import (
    ACTION_DIM,
    EE_DELTA_GRASP_V0,
    ActionSchema,
    ContractError,
    ExecutionResult,
    SequenceBatch,
    StateSchema,
    validate_actions,
)
from embodied_jepa.data import DatasetStore, deterministic_fixture
from embodied_jepa.planning import CEMConfig, CEMPlanner
from embodied_jepa.registry import EMBODIMENTS
from embodied_jepa.schema_planning import SchemaCEMConfig, SchemaCEMPlanner, create_planner

SO101 = JOINT_DELTA_SO101_V0


def test_schema_is_registered_with_fixed_names_order_and_dimension():
    assert SO101.version == "joint_delta_so101_v0"
    assert SO101.names == (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
    assert SO101.names == SO101_ACTION_NAMES and SO101.dimension == 6
    assert action_schema("joint_delta_so101_v0") is SO101
    assert action_schema("ee_delta_grasp_v0") is EE_DELTA_GRASP_V0
    assert EE_DELTA_GRASP_V0.dimension == ACTION_DIM == 14
    assert set(ACTION_SCHEMAS) == {"ee_delta_grasp_v0", "joint_delta_so101_v0"}
    with pytest.raises(ContractError, match="unknown action schema"):
        action_schema("joint_delta_so100_v0")


def test_schema_records_round_trip_exactly_and_reject_reordering():
    for schema in ACTION_SCHEMAS.values():
        assert action_schema_from_record(json.loads(json.dumps(schema_record(schema)))) is schema
    reordered = schema_record(SO101) | {"names": list(reversed(SO101.names))}
    with pytest.raises(ContractError, match="incompatible"):
        action_schema_from_record(reordered)
    unregistered = asdict(ActionSchema("joint_delta_other_v0", ("a", "b")))
    with pytest.raises(ContractError, match="unknown"):
        action_schema_from_record(unregistered)


def _batch(schema, width, horizon=2, batch=3):
    state = StateSchema(("arm.q",), ("rad",), "fixture_v0")
    rng = np.random.default_rng(0)
    return SequenceBatch(
        observations={"wrist": rng.integers(0, 256, (batch, horizon + 1, 4, 4, 3), np.uint8)},
        robot_states=np.zeros((batch, horizon + 1, 1), np.float32),
        state_mask=np.ones((batch, horizon + 1, 1), np.bool_),
        actions=rng.uniform(-1, 1, (batch, horizon, width)).astype(np.float32),
        timestamps=np.tile(np.arange(horizon + 1, dtype=np.float64), (batch, 1)),
        terminated=np.zeros((batch, horizon), np.bool_),
        episode_ids=tuple(f"e{i}" for i in range(batch)),
        state_schema=state,
        action_schema=schema,
    )


def test_contracts_validate_six_d_actions_without_coercion():
    batch = _batch(SO101, 6)
    assert batch.actions.shape == (3, 2, 6) and batch.action_schema is SO101
    with pytest.raises(ContractError, match="6 components"):
        _batch(SO101, 14)
    with pytest.raises(ContractError, match="14 components"):
        _batch(EE_DELTA_GRASP_V0, 6)
    action = np.full(6, 0.5, np.float32)
    validate_actions(action, ndim=1, schema=SO101)
    with pytest.raises(ContractError, match="14 components"):
        validate_actions(action, ndim=1)  # the default stays ee_delta_grasp_v0
    with pytest.raises(ContractError, match="numpy float32"):
        validate_actions(action.astype(np.float64), ndim=1, schema=SO101)
    with pytest.raises(ContractError, match="within"):
        validate_actions(np.full(6, 1.5, np.float32), ndim=1, schema=SO101)
    result = ExecutionResult(action, action, "applied", 0.0, action_schema=SO101)
    assert result.applied_action.shape == (6,)


# ----- data -------------------------------------------------------------------------------------
@pytest.fixture
def storage():
    for name in ("pyarrow", "pandas", "PIL"):
        pytest.importorskip(name)


def _so101_episode():
    episode = deterministic_fixture()
    actions = np.linspace(-0.5, 0.5, 4 * 6, dtype=np.float32).reshape(4, 6)
    return replace(
        episode, actions=actions, raw_actions=actions * np.float32(0.02), action_schema=SO101
    )


def _store(path, episode, **kwargs):
    return DatasetStore.create(
        path,
        fps=20,
        state_schema=episode.state_schema,
        action_manifest={"purpose": "software fixture only"},
        provenance={"source": "procedural fixture", "license": "Apache-2.0"},
        robot_type="software_fixture",
        **kwargs,
    )


def test_six_d_episode_round_trips_through_the_store_with_a_placeholder_final_row(
    tmp_path, storage
):
    import pyarrow.parquet as pq

    episode = _so101_episode()
    store = _store(tmp_path / "so101", episode, action_schema=SO101)
    store.write_episode(episode)
    reloaded = DatasetStore(store.root)
    assert reloaded.action_schema is SO101
    assert reloaded.manifest["action_schema"] == schema_record(SO101)
    restored = reloaded.read_episode(episode.episode_id)
    assert restored.action_schema is SO101
    np.testing.assert_array_equal(restored.actions, episode.actions)
    np.testing.assert_array_equal(restored.raw_actions, episode.raw_actions)
    info = json.loads((store.root / "meta/info.json").read_text())
    assert info["features"]["action"]["shape"] == [6]
    assert info["features"]["action"]["names"] == list(SO101.names)
    row = reloaded.manifest["episodes"][0]
    table = pq.read_table(store.root / row["path"]).to_pydict()
    assert table["action_valid"] == [True] * 4 + [False]
    assert table["action"][-1] == [0.0] * 6
    windows = list(reloaded.windows(3))
    assert [w.action_schema for w in windows] == [SO101, SO101]
    np.testing.assert_array_equal(windows[1].actions[0], episode.actions[1:])


def test_store_rejects_mixed_schemas_and_unregistered_ones(tmp_path, storage):
    so101 = _so101_episode()
    store = _store(tmp_path / "so101", so101, action_schema=SO101)
    with pytest.raises(ContractError, match="action schema differs"):
        store.write_episode(deterministic_fixture())
    g1 = _store(tmp_path / "g1", so101)  # default stays ee_delta_grasp_v0
    assert g1.action_schema is EE_DELTA_GRASP_V0
    with pytest.raises(ContractError, match="action schema differs"):
        g1.write_episode(so101)
    with pytest.raises(ContractError):
        _store(tmp_path / "other", so101, action_schema=ActionSchema("joint_delta_x_v0", ("a",)))
    with pytest.raises(ContractError, match=r"\[T,6\]"):
        replace(so101, actions=np.zeros(6, np.float32))


def test_ee_delta_manifest_record_is_unchanged(tmp_path, storage):
    episode = deterministic_fixture()
    store = _store(tmp_path / "g1", episode)
    assert store.manifest["action_schema"] == json.loads(json.dumps(asdict(EE_DELTA_GRASP_V0)))
    store.write_episode(episode)
    info = json.loads((store.root / "meta/info.json").read_text())
    assert info["features"]["action"] == {
        "dtype": "float32",
        "shape": [14],
        "names": list(EE_DELTA_GRASP_V0.names),
    }


# ----- config -----------------------------------------------------------------------------------
@pytest.fixture
def so101_config(tmp_path, monkeypatch):
    # S1 registers the real SO-101 embodiment; this stands in for it in config validation only.
    monkeypatch.setitem(EMBODIMENTS._factories, "so101_test_stub", "builtins:object")
    monkeypatch.setitem(
        config_module.EMBODIMENT_ACTION_SCHEMAS, "so101_test_stub", "joint_delta_so101_v0"
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "native.pt").touch()
    manifest = {
        "action_schema": "joint_delta_so101_v0",
        "control_dt_s": 0.05,
        "joint_delta_per_step_rad": {name: 0.05 for name in SO101.names[:-1]},
    }
    (tmp_path / "action.json").write_text(json.dumps(manifest))
    raw = {
        "schema_version": 0,
        "world_model": {"backend": "native_jepa", "checkpoints": {"native_jepa": "native.pt"}},
        "embodiment": {
            "backend": "so101_test_stub",
            "transport": "mujoco",
            "action_schema": "joint_delta_so101_v0",
            "action_manifest": "action.json",
        },
        "planner": {"horizon": 3, "samples": 16, "elites": 4},
        "dataset": {"root": "data"},
        "task": {"name": "reach"},
    }
    path = tmp_path / "so101.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path, raw, manifest


def test_six_d_config_resolves_to_a_schema_sized_planner(so101_config):
    path, _, _ = so101_config
    loaded = ExperimentConfig.load(path)
    assert loaded.action_schema == "joint_delta_so101_v0"
    assert isinstance(loaded.planner, SchemaCEMConfig)
    assert loaded.planner.action_schema is SO101
    assert loaded.planner.lower_bounds == (-1.0,) * 6
    assert loaded.resolved["embodiment"]["action_schema"] == "joint_delta_so101_v0"
    assert "action_schema" not in loaded.resolved["planner"]
    assert isinstance(create_planner(loaded.planner), SchemaCEMPlanner)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda raw, m: raw["embodiment"].update(backend="unitree_g1_dex3"), "does not execute"),
        (lambda raw, m: raw["embodiment"].update(action_schema="joint_delta_x"), "unsupported"),
        (lambda raw, m: m.update(action_schema="ee_delta_grasp_v0"), "incompatible schema"),
        (lambda raw, m: m.pop("control_dt_s"), "control_dt_s"),
        (lambda raw, m: m["joint_delta_per_step_rad"].pop("wrist_roll"), "joint_delta_per_step"),
        (lambda raw, m: m["joint_delta_per_step_rad"].update(gripper=0.1), "joint_delta_per_step"),
        (lambda raw, m: m["joint_delta_per_step_rad"].update(elbow_flex=0), "joint_delta_per_step"),
        (lambda raw, m: raw["planner"].update(lower_bounds=[-1.0] * 14), "6-vectors"),
    ],
)
def test_six_d_config_rejects_inconsistent_schema_manifest_and_bounds(so101_config, change, match):
    path, raw, manifest = so101_config
    change(raw, manifest)
    path.write_text(yaml.safe_dump(raw))
    (path.parent / "action.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        ExperimentConfig.load(path)


def test_g1_config_keeps_the_fourteen_d_path(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "native.pt").touch()
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
        "world_model": {"backend": "native_jepa", "checkpoints": {"native_jepa": "native.pt"}},
        "embodiment": {
            "backend": "unitree_g1_dex3",
            "transport": "mujoco",
            "action_manifest": "action.json",
        },
        "dataset": {"root": "data"},
        "task": {"name": "reach"},
    }
    path = tmp_path / "g1.yaml"
    path.write_text(yaml.safe_dump(raw))
    loaded = ExperimentConfig.load(path)
    assert loaded.action_schema == "ee_delta_grasp_v0"
    assert type(loaded.planner) is CEMConfig
    assert loaded.planner == CEMConfig(seed=0)
    assert type(create_planner(loaded.planner)) is CEMPlanner
    raw["embodiment"]["action_schema"] = "joint_delta_so101_v0"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="does not execute"):
        ExperimentConfig.load(path)


# ----- CEM --------------------------------------------------------------------------------------
class Integrator:
    def predict(self, latent, actions):
        return latent[:, None, None] + np.cumsum(actions[..., :2], axis=2)

    def distance(self, predicted, goal):
        return np.square(predicted - goal[:, None, None]).sum(-1).astype(np.float32)


def test_schema_cem_plans_six_d_actions_on_a_toy_model():
    config = SchemaCEMConfig(SO101, horizon=4, samples=256, iterations=5, elites=20, seed=14)
    model = Integrator()
    initial = np.zeros((2, 2), np.float32)
    goal = np.array([[1.7, -1.3], [-0.8, 1.2]], np.float32)
    plan = SchemaCEMPlanner(config).plan(model, initial, goal, batch_size=2)
    assert plan.actions.shape == (2, 4, 6) and plan.actions.dtype == np.float32
    assert np.all(np.abs(plan.actions) <= 1)
    assert np.all(plan.costs < np.square(goal).sum(1) * 0.01)
    assert plan.config["action_schema"]["version"] == "joint_delta_so101_v0"
    validate_actions(plan.actions, ndim=3, schema=SO101)
    repeat = SchemaCEMPlanner(config).plan(model, initial, goal, batch_size=2)
    np.testing.assert_array_equal(plan.actions, repeat.actions)
    bounded = SchemaCEMConfig(SO101, lower_bounds=(-0.2,) * 6, upper_bounds=(0.3,) * 6, seed=1)
    plan = SchemaCEMPlanner(bounded).plan(model, initial, goal, batch_size=2)
    assert plan.actions.min() >= -0.2 and plan.actions.max() <= 0.3


def test_schema_cem_matches_the_pinned_planner_on_fourteen_d():
    budget = {"horizon": 3, "samples": 32, "iterations": 4, "elites": 5, "seed": 7}
    model = Integrator()
    initial = np.zeros((2, 2), np.float32)
    goal = np.array([[0.7, -0.3], [-0.4, 0.9]], np.float32)
    pinned = CEMPlanner(CEMConfig(**budget)).plan(model, initial, goal, batch_size=2)
    mirror = SchemaCEMPlanner(SchemaCEMConfig(EE_DELTA_GRASP_V0, **budget)).plan(
        model, initial, goal, batch_size=2
    )
    np.testing.assert_array_equal(pinned.actions, mirror.actions)
    np.testing.assert_array_equal(pinned.requested_actions, mirror.requested_actions)
    np.testing.assert_array_equal(pinned.costs, mirror.costs)
    assert pinned.evaluations == mirror.evaluations
    assert type(create_planner(SchemaCEMConfig(EE_DELTA_GRASP_V0, **budget))) is CEMPlanner


@pytest.mark.parametrize(
    "options",
    [
        {"lower_bounds": (-1.0,) * 14},
        {"upper_bounds": (1.0,) * 5},
        {"lower_bounds": (0.5,) * 6, "upper_bounds": (0.0,) * 6},
        {"elites": 65},
        {"action_schema": ActionSchema("joint_delta_x_v0", ("a",))},
    ],
)
def test_schema_cem_budget_fails_early(options):
    with pytest.raises(ValueError):
        SchemaCEMConfig(**({"action_schema": SO101} | options))


def test_core_import_stays_light():
    import subprocess
    import sys

    code = (
        "import sys, embodied_jepa, embodied_jepa.action_schemas, embodied_jepa.schema_planning,"
        " embodied_jepa.config, embodied_jepa.data;"
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules, sorted(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
