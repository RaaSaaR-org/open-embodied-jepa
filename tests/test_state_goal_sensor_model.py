"""Synthetic state-goal composition fixtures, never manipulation success evidence."""

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from embodied_jepa.config import MODELS  # noqa: E402
from embodied_jepa.contracts import (  # noqa: E402
    ContractError,
    RobotState,
    SequenceBatch,
    StateSchema,
)
from embodied_jepa.models.sensor import SensorWorldModel  # noqa: E402
from embodied_jepa.models.state_goal_sensor import (  # noqa: E402
    FIELDS,
    VISUAL_WEIGHT,
    StateGoalLatent,
    StateGoalSensorWorldModel,
)

NAMES = ("left_elbow_joint.position", *FIELDS, "right_elbow_joint.velocity")
SCHEMA = StateSchema(
    NAMES, tuple("rad/s" if n.endswith("velocity") else "rad" for n in NAMES), "state_goal_v0"
)
METADATA = {"dataset_hash": "d", "split_hash": "s", "action_hash": "a"}
CONFIG = dict(hidden_dim=16, candidate_chunk_size=1, max_horizon=16)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def images(value=128, batch=1):
    return {"onboard_rgb": np.full((batch, 8, 8, 3), value, np.uint8)}


def robot_state(values):
    values = np.asarray(values, np.float32).reshape(1, -1)
    return RobotState(values, np.ones_like(values, bool), np.zeros(1), SCHEMA)


@pytest.fixture
def checkpoint(tmp_path):
    rng = np.random.default_rng(0)
    states = rng.normal(0, [1.0 + i for i in range(len(NAMES))], (2, 3, len(NAMES)))
    states = states.astype(np.float32)
    batch = SequenceBatch(
        {"onboard_rgb": rng.integers(0, 255, (2, 3, 8, 8, 3), dtype=np.uint8)},
        states,
        np.ones_like(states, bool),
        rng.uniform(-0.5, 0.5, (2, 2, 14)).astype(np.float32),
        np.tile(np.array([0.0, 0.05, 0.1]), (2, 1)),
        np.zeros((2, 2), bool),
        ("train-a", "train-b"),
        SCHEMA,
    )
    sensor = SensorWorldModel(SCHEMA, config=CONFIG, metadata=METADATA)
    sensor.fit_normalization([batch], training_episode_ids=batch.episode_ids)
    for _ in range(2):
        sensor.train_step(batch)  # Nonzero action derivatives for prediction checks.
    path = tmp_path / "sensor.pt"
    sensor.save(path)
    return path


def loaded(path, **overrides):
    config = {"sensor_config": dict(torch.load(path, weights_only=True)["config"])}
    config |= overrides
    model = StateGoalSensorWorldModel(SCHEMA, config=config, metadata=METADATA)
    return model.load(path)


def test_registered_backend_creates_the_frozen_composition(checkpoint):
    config = {"sensor_config": torch.load(checkpoint, weights_only=True)["config"]}
    model = MODELS.create(
        "state_goal_sensor_wm_v1", state_schema=SCHEMA, config=config, metadata=METADATA
    )
    assert isinstance(model, StateGoalSensorWorldModel)
    model.load(checkpoint)
    assert model.capabilities.max_horizon == 16


def test_cost_is_normalized_predicted_arm_hand_mse_and_visual_weight_is_zero(checkpoint):
    model = loaded(checkpoint)
    child = SensorWorldModel(SCHEMA, config=CONFIG, metadata=METADATA)
    child.load(checkpoint)
    current = np.arange(len(NAMES), dtype=np.float32) / 10
    z = model.encode(images(), robot_state(current))
    actions = np.random.default_rng(1).uniform(-1, 1, (1, 3, 4, 14)).astype(np.float32)
    predicted = model.predict(z, actions)
    goal_values = np.linspace(-0.3, 0.3, 14, dtype=np.float32)[None]
    goal = model.encode_goal({"state": goal_values, "images": images(10)})
    cost = model.distance(predicted, goal)
    assert cost.shape == (1, 3, 4) and cost.dtype == np.float32 and (cost >= 0).all()

    # Child predictions are unchanged; cost uses the child's TRAIN normalization.
    reference = child.predict(child.encode(images(), robot_state(current)), actions).values
    assert torch.equal(predicted.child.values, reference)
    n = child.visual_dimension
    index = np.array([NAMES.index(f) for f in FIELDS])
    mean = child.sensor_mean[n + index].numpy()
    scale = child.sensor_scale[n + index].numpy()
    expected = np.square(reference[..., n + index].numpy() - (goal_values - mean) / scale).mean(-1)
    np.testing.assert_allclose(cost, expected, rtol=1e-5, atol=1e-7)
    assert len(set(np.round(cost[0, :, -1], 7))) > 1  # Actions change the ranking signal.

    diagnostic = model.pop_diagnostics()
    assert diagnostic[0]["visual_weight"] == VISUAL_WEIGHT == 0.0
    assert len(diagnostic[0]["visual_endpoint_costs"]) == 3
    assert model.pop_diagnostics() == []
    # Changing only the goal image never changes the planning cost.
    other = model.encode_goal({"state": goal_values, "images": images(250)})
    np.testing.assert_array_equal(model.distance(predicted, other), cost)
    no_image = model.encode_goal({"state": goal_values})
    np.testing.assert_array_equal(model.distance(predicted, no_image), cost)
    assert (
        model.pop_diagnostics()[0]["visual_endpoint_costs"]
        != diagnostic[0]["visual_endpoint_costs"]
    )


def test_observed_progress_uses_measured_fields_only(checkpoint):
    model = loaded(checkpoint)
    state = np.arange(len(NAMES), dtype=np.float32) / 7
    goal = model.select(state[None], SCHEMA)
    assert goal.shape == (1, 14)
    assert model.observed_distance(robot_state(state), goal)[0] == 0
    changed = state.copy()
    changed[0] += 5  # left elbow is not a goal field
    changed[-1] += 5  # velocity is not a goal field
    assert model.observed_distance(robot_state(changed), goal)[0] == 0
    changed[NAMES.index(FIELDS[-1])] += 1
    assert model.observed_distance(robot_state(changed), goal)[0] > 0
    distance = model.image_distance(images(1), images(1))
    assert distance.shape == (1,) and distance[0] == 0


def test_invalid_goals_latents_and_training_are_rejected(checkpoint):
    model = loaded(checkpoint)
    z = model.encode(images(), robot_state(np.zeros(len(NAMES))))
    goal = model.encode_goal({"state": np.zeros((1, 14), np.float32)})
    prediction = model.predict(z, np.zeros((1, 2, 3, 14), np.float32))
    for bad in (
        {"state": np.zeros((1, 13), np.float32)},
        {"state": np.full((1, 14), np.nan, np.float32)},
        {"state": np.zeros((1, 14), np.int64)},
        {"state": np.zeros((1, 14), np.float32), "phase": 3},
        {"images": images()},
        np.zeros((1, 14), np.float32),
    ):
        with pytest.raises(ContractError):
            model.encode_goal(bad)
    with pytest.raises(ContractError):
        model.encode_goal({"state": np.zeros((2, 14), np.float32), "images": images()})
    with pytest.raises(ContractError):
        model.distance(StateGoalLatent(prediction.child, object()), goal)
    with pytest.raises(ContractError):
        model.distance(prediction, model.encode_goal({"state": np.zeros((2, 14), np.float32)}))
    with pytest.raises(ContractError, match="training is forbidden"):
        model.train_step(None)
    other = loaded(checkpoint)
    with pytest.raises(ContractError):
        other.distance(prediction, goal)  # latents belong to the first instance
    model.load(checkpoint)
    with pytest.raises(ContractError):
        model.distance(prediction, goal)  # reload invalidates old latents


def test_construction_and_checkpoint_compatibility_fail_closed(checkpoint, tmp_path):
    with pytest.raises(ContractError):
        StateGoalSensorWorldModel(SCHEMA, device="mps")
    with pytest.raises(ContractError):
        StateGoalSensorWorldModel(SCHEMA, config={"visual_weight": 0.5})
    with pytest.raises(ContractError):
        StateGoalSensorWorldModel(SCHEMA, config={"hybrid": True})
    missing = replace(SCHEMA, names=(*NAMES[:-2], "renamed", NAMES[-1]), version="missing")
    with pytest.raises(ContractError, match="fourteen"):
        StateGoalSensorWorldModel(missing)
    wrong_units = replace(SCHEMA, units=("rad",) * (len(NAMES) - 2) + ("deg", "rad/s"))
    with pytest.raises(ContractError, match="radian"):
        StateGoalSensorWorldModel(wrong_units)
    unloaded = StateGoalSensorWorldModel(SCHEMA, config={"sensor_config": CONFIG})
    with pytest.raises(ContractError, match="load"):
        unloaded.encode_goal({"state": np.zeros((1, 14), np.float32)})
    with pytest.raises(ContractError, match="sensor_config"):
        loaded(checkpoint, sensor_config=CONFIG | {"hidden_dim": 32})
    wrong = StateGoalSensorWorldModel(
        SCHEMA,
        config={"sensor_config": torch.load(checkpoint, weights_only=True)["config"]},
        metadata=METADATA | {"dataset_hash": "other"},
    )
    with pytest.raises(ContractError, match="provenance"):
        wrong.load(checkpoint)
    envelope = torch.load(checkpoint, weights_only=True)
    envelope["backend"] = "aligned_sensor_wm_v1"
    torch.save(envelope, tmp_path / "wrong.pt")
    with pytest.raises(ContractError, match="sensor_wm"):
        loaded(checkpoint).load(tmp_path / "wrong.pt")
    envelope = torch.load(checkpoint, weights_only=True)
    envelope["metadata"]["normalization"]["method"] = "val_moments"
    torch.save(envelope, tmp_path / "val.pt")
    with pytest.raises(ContractError, match="TRAIN"):
        loaded(checkpoint).load(tmp_path / "val.pt")


def test_save_copies_unchanged_child_and_refuses_overwrite(checkpoint, tmp_path):
    model = loaded(checkpoint)
    target = model.save(tmp_path / "copy" / "sensor.pt")
    assert target.read_bytes() == checkpoint.read_bytes()
    with pytest.raises(FileExistsError):
        model.save(target)
