"""Synthetic dynamics tests, not evidence of learned robot manipulation."""

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from embodied_jepa.contracts import ContractError, SequenceBatch, StateSchema  # noqa: E402
from embodied_jepa.models.base import VisualLatent  # noqa: E402
from embodied_jepa.models.sensor import SensorWorldModel  # noqa: E402


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def dynamics(seed=42, prefix="train", n=16):
    rng = np.random.default_rng(seed)
    actions = np.zeros((n, 3, 14), np.float32)
    actions[..., 0] = rng.choice([-1.0, 0, 1.0], (n, 3))
    states = np.zeros((n, 4, 1), np.float32)
    states[:, 0, 0] = rng.integers(-2, 3, n)
    states[:, 1:, 0] = states[:, :1, 0] + np.cumsum(actions[..., 0], 1)
    pixels = (
        np.broadcast_to((128 + 12 * states[..., 0])[..., None, None, None], (n, 4, 8, 8, 3))
        .astype(np.uint8)
        .copy()
    )
    return SequenceBatch(
        {"onboard_rgb": pixels},
        states,
        np.ones_like(states, bool),
        actions,
        np.tile(np.arange(4, dtype=np.float64) * 0.05, (n, 1)),
        np.zeros((n, 3), bool),
        tuple(f"{prefix}-{i}" for i in range(n)),
        StateSchema(("joint.q",), ("rad",), "sensor_fixture_v0"),
    )


@pytest.fixture
def batch():
    return dynamics()


def make(batch, **kwargs):
    model = SensorWorldModel(
        batch.state_schema,
        config={"image_size": 4, "hidden_dim": 32, "learning_rate": 0.005},
        metadata={"dataset_hash": "fixture"},
        **kwargs,
    )
    model.fit_normalization([batch], training_episode_ids=batch.episode_ids)
    return model


def test_fit_requires_explicit_training_coverage_and_freezes_before_validation(batch):
    model = SensorWorldModel(batch.state_schema, config={"image_size": 4})
    with pytest.raises(ContractError, match="training-only"):
        model.encode_goal(batch.observation(0).images)
    with pytest.raises(ContractError, match="outside the training"):
        model.fit_normalization([batch], training_episode_ids=("validation",))
    with pytest.raises(ContractError, match="every declared"):
        model.fit_normalization([batch], training_episode_ids=(*batch.episode_ids, "missing"))
    assert not bool(model.normalization_fitted)
    model.fit_normalization([batch], training_episode_ids=batch.episode_ids)
    mean, scale = model.sensor_mean.clone(), model.sensor_scale.clone()
    assert model.metadata["normalization"]["frame_count"] == 64
    np.testing.assert_allclose(mean[-1].item(), batch.robot_states.mean(), atol=1e-6)
    model.diagnostics(dynamics(seed=99, prefix="validation"))
    torch.testing.assert_close(model.sensor_mean, mean, rtol=0, atol=0)
    torch.testing.assert_close(model.sensor_scale, scale, rtol=0, atol=0)
    with pytest.raises(ContractError, match="frozen"):
        model.fit_normalization([batch], training_episode_ids=batch.episode_ids)


def test_proprioception_changes_state_but_goals_and_cost_ignore_goal_proprioception(batch):
    model = make(batch)
    observation = batch.observation(0)
    state = model.encode(observation.images, observation.state)
    changed = model.encode(
        observation.images, replace(observation.state, values=observation.state.values + 1)
    )
    torch.testing.assert_close(
        state.values[:, : model.visual_dimension], changed.values[:, : model.visual_dimension]
    )
    assert not torch.equal(state.values[:, -1], changed.values[:, -1])
    goal = model.encode_goal(batch.observation(3).images)
    assert goal.values.shape == (16, model.visual_dimension)
    actions = batch.actions[:, None]
    prediction = model.predict(state, actions)
    alternative = prediction.values.clone()
    alternative[..., model.visual_dimension :] += 1000
    np.testing.assert_array_equal(
        model.distance(prediction, goal),
        model.distance(VisualLatent(alternative, model._owner), goal),
    )
    assert np.all(model.observed_distance(observation.images, observation.images) == 0)
    with pytest.raises(ContractError, match="not an image goal"):
        model.predict(goal, actions)


def test_learns_action_conditioned_heldout_dynamics_better_than_persistence_and_shuffle(batch):
    model = make(batch)
    validation = dynamics(seed=101, prefix="validation")
    before = model.diagnostics(validation)
    for _ in range(160):
        metrics = model.train_step(batch)
    after = model.diagnostics(validation)
    assert np.isfinite(list(metrics.values())).all()
    assert after["prediction_mse"] < 0.1 * before["prediction_mse"]
    assert after["prediction_mse"] < 0.1 * after["persistence_mse"]
    assert after["prediction_mse"] < 0.1 * after["shuffled_action_mse"]
    assert after["proprio_prediction_mse"] < 0.03
    assert after["action_effect_rms"] > 0.01


def test_recursive_prediction_is_causal_and_candidate_chunks_agree(batch):
    model = make(batch)
    current = batch.observation(0)
    latent = model.encode(current.images, current.state)
    actions = np.repeat(batch.actions[:, None], 3, 1)
    model.config["candidate_chunk_size"] = 2
    full = model.predict(latent, actions).values
    single = model.predict(latent, actions[:, :1]).values
    torch.testing.assert_close(full[:, :1], single)
    actions[:, :, 1:] *= -1
    changed = model.predict(latent, actions).values
    torch.testing.assert_close(full[:, :, 0], changed[:, :, 0], rtol=0, atol=0)
    assert not full.requires_grad


def test_checkpoint_resumes_optimizer_preserves_provenance_and_invalidates_latents(batch, tmp_path):
    model = make(batch)
    model.train_step(batch)
    current = batch.observation(0)
    old = model.encode(current.images, current.state)
    path = tmp_path / "sensor.pt"
    model.save(path)
    raw = torch.load(path, weights_only=True)
    assert raw["preprocessing"]["robot_state"] != "ignored_visual_baseline"
    restored = SensorWorldModel(
        batch.state_schema, config=model.config, metadata={"dataset_hash": "fixture"}
    )
    restored.load(path)
    assert model.train_step(batch) == restored.train_step(batch)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ContractError, match="another model"):
        model.predict(old, batch.actions[:, None])
    wrong = SensorWorldModel(
        batch.state_schema, config=model.config, metadata={"dataset_hash": "other"}
    )
    with pytest.raises(ContractError, match="provenance"):
        wrong.load(path)
    raw["weights"]["sensor_scale"][0] = 0
    torch.save(raw, path)
    with pytest.raises(ContractError, match="normalization scales"):
        restored.load(path)


def test_missing_state_and_invalid_actions_are_rejected(batch):
    model = make(batch)
    # A schema with optional fields still requires complete observations in this backend.
    optional = replace(batch.state_schema, required=(False,))
    masked = replace(batch, state_schema=optional, state_mask=np.zeros_like(batch.state_mask))
    other = SensorWorldModel(optional, config={"image_size": 4})
    with pytest.raises(ContractError, match="every proprioceptive"):
        other.fit_normalization([masked], training_episode_ids=masked.episode_ids)
    current = batch.observation(0)
    actions = batch.actions[:, None].copy()
    actions[0, 0, 0, 0] = np.nan
    with pytest.raises(ContractError):
        model.predict(model.encode(current.images, current.state), actions)


def test_mps_sensor_step_and_roundtrip(batch, tmp_path):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    model = make(batch, device="mps")
    assert np.isfinite(list(model.train_step(batch).values())).all()
    current = batch.observation(0)
    goal = model.encode_goal(batch.observation(3).images)
    latent = model.encode(current.images, current.state)
    before = model.distance(model.predict(latent, batch.actions[:, None]), goal)
    model.save(tmp_path / "mps.pt")
    restored = SensorWorldModel(batch.state_schema, device="mps", config=model.config)
    restored.load(tmp_path / "mps.pt")
    actual = restored.distance(
        restored.predict(restored.encode(current.images, current.state), batch.actions[:, None]),
        restored.encode_goal(batch.observation(3).images),
    )
    np.testing.assert_allclose(before, actual, rtol=1e-5, atol=1e-7)
