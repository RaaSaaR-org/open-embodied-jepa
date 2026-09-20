"""Synthetic contract evidence; these tests make no learned-control claims."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from embodied_jepa.contracts import ContractError, SequenceBatch, StateSchema  # noqa: E402
from embodied_jepa.models import NativeJEPA  # noqa: E402


@pytest.fixture(autouse=True)
def bounded_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(original)


@pytest.fixture
def batch():
    rng = np.random.default_rng(42)
    schema = StateSchema(("arm.q",), ("rad",), "fixture_v0")
    return SequenceBatch(
        observations={"onboard_rgb": rng.integers(0, 256, (3, 3, 32, 40, 3), dtype=np.uint8)},
        robot_states=np.zeros((3, 3, 1), dtype=np.float32),
        state_mask=np.ones((3, 3, 1), dtype=np.bool_),
        actions=rng.uniform(-1, 1, (3, 2, 14)).astype(np.float32),
        timestamps=np.tile(np.arange(3, dtype=np.float64), (3, 1)),
        terminated=np.zeros((3, 2), dtype=np.bool_),
        episode_ids=("a", "b", "c"),
        state_schema=schema,
    )


def model_class(backend):
    if backend == "native":
        return NativeJEPA
    pytest.importorskip("transformers")
    if not Path("third_party/le-wm/jepa.py").exists():
        pytest.skip("optional pinned LeWM source absent; run scripts/fetch_lewm.py")
    from embodied_jepa.models import LeWM

    return LeWM


@pytest.fixture(params=["native", "lewm"])
def model(request, batch):
    return model_class(request.param)(
        batch.state_schema,
        seed=3,
        config={"candidate_chunk_size": 2, "hidden_dim": 48},
        metadata={"dataset_hash": "synthetic-fixture", "action_hash": "test-action-v0"},
    )


def inputs(model, batch):
    current = batch.observation(0)
    latent = model.encode(current.images, current.state)
    goal = model.encode_goal(batch.observation(2).images)
    actions = np.repeat(batch.actions[:, None], 5, axis=1)
    return latent, goal, actions


def test_shared_api_recursive_chunks_and_image_goal(model, batch):
    latent, goal, actions = inputs(model, batch)
    prediction = model.predict(latent, actions)
    cost = model.distance(prediction, goal)
    assert cost.shape == (3, 5, 2)
    assert cost.dtype == np.float32 and np.isfinite(cost).all()
    np.testing.assert_allclose(
        cost[:, :1],
        model.distance(model.predict(latent, actions[:, :1]), goal),
        rtol=1e-5,
        atol=1e-7,
    )
    assert not prediction.values.requires_grad
    assert not goal.values.requires_grad
    # Prefix predictions do not depend on future actions or unavailable robot state.
    changed = actions.copy()
    changed[:, :, 1] *= -1
    changed_cost = model.distance(model.predict(latent, changed), goal)
    np.testing.assert_allclose(cost[:, :, 0], changed_cost[:, :, 0])
    state = replace(batch.observation(0).state, values=np.ones((3, 1), dtype=np.float32))
    alternative = model.encode(batch.observation(0).images, state)
    np.testing.assert_array_equal(latent.values.cpu().numpy(), alternative.values.cpu().numpy())


def test_train_diagnostics_action_sensitivity_and_resume(model, batch, tmp_path):
    for _ in range(2):
        metrics = model.train_step(batch)
        assert all(np.isfinite(value) for value in metrics.values())
    diagnostics = model.diagnostics(batch)
    assert diagnostics["action_effect_rms"] > 0
    assert diagnostics["shuffled_actions_changed"] == 1
    assert model.validation_error(batch) == diagnostics["prediction_mse"]
    assert 0 <= diagnostics["collapsed_fraction"] <= 1
    latent, goal, actions = inputs(model, batch)
    before = model.distance(model.predict(latent, actions), goal)
    path = tmp_path / "model.pt"
    model.save(path)
    restored = type(model)(
        batch.state_schema, config=model.config, seed=999, metadata=model.metadata
    )
    restored.load(path)
    a, g, u = inputs(restored, batch)
    np.testing.assert_array_equal(before, restored.distance(restored.predict(a, u), g))
    assert restored.updates == 2
    # Exact next-step CPU optimizer/RNG resumption, despite unrelated global RNG.
    expected = model.train_step(batch)
    torch.rand(100)
    actual = restored.train_step(batch)
    assert actual == expected
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ContractError, match="another model"):
        restored.predict(latent, actions)


def test_rejects_checkpoint_schema_provenance_and_config(model, batch, tmp_path):
    path = tmp_path / "model.pt"
    model.save(path)
    for kwargs in (
        {"state_schema": replace(batch.state_schema, units=("degree",))},
        {"config": model.config | {"image_size": model.config["image_size"] * 2}},
        {"metadata": {"dataset_hash": "different-data"}},
        {"metadata": {"action_hash": "different-calibration"}},
    ):
        options = {
            "state_schema": batch.state_schema,
            "config": model.config,
            "metadata": model.metadata,
        } | kwargs
        incompatible = type(model)(**options)
        with pytest.raises(ContractError, match="incompatible"):
            incompatible.load(path)


def test_invalid_actions_camera_and_latent_are_rejected(model, batch):
    latent, goal, actions = inputs(model, batch)
    invalid = actions.copy()
    invalid[0, 0, 0, 0] = np.nan
    with pytest.raises(ContractError):
        model.predict(latent, invalid)
    with pytest.raises(ContractError):
        model.predict(latent, actions[:1])
    with pytest.raises(ContractError):
        model.encode_goal({"wrong-camera": batch.observation(0).images["onboard_rgb"]})
    with pytest.raises(ContractError):
        model.distance(goal, latent)
    with pytest.raises(ContractError):
        model.train_step(replace(batch, state_schema=replace(batch.state_schema, version="other")))


def test_ema_target_is_frozen_and_updates(batch):
    model = NativeJEPA(batch.state_schema)
    before = [p.clone() for p in model.target_encoder.parameters()]
    model.train_step(batch)
    assert all(not p.requires_grad and p.grad is None for p in model.target_encoder.parameters())
    assert any(
        not torch.equal(a, b)
        for a, b in zip(before, model.target_encoder.parameters(), strict=True)
    )


def test_constructor_and_training_do_not_modify_global_rng(batch):
    before = torch.get_rng_state().clone()
    model = NativeJEPA(batch.state_schema)
    model.train_step(batch)
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["native", "lewm"])
def test_mps_forward_backward_and_costs(backend, batch, tmp_path):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    model = model_class(backend)(batch.state_schema, device="mps", seed=4)
    metrics = model.train_step(batch)
    assert all(np.isfinite(value) for value in metrics.values())
    latent, goal, actions = inputs(model, batch)
    costs = model.distance(model.predict(latent, actions), goal)
    assert np.isfinite(costs).all()
    path = tmp_path / "mps.pt"
    model.save(path)
    restored = model_class(backend)(batch.state_schema, device="mps", seed=5)
    restored.load(path)
    a, g, u = inputs(restored, batch)
    np.testing.assert_allclose(costs, restored.distance(restored.predict(a, u), g), rtol=1e-5)


def test_lewm_imports_actual_pinned_classes_and_rejects_batch_one(batch):
    model = model_class("lewm")(batch.state_schema)
    assert model.model.__class__.__module__ == "embodied_jepa_upstream_jepa"
    assert model.model.predictor.__class__.__module__ == "embodied_jepa_upstream_module"
    single = replace(
        batch,
        observations={k: v[:1] for k, v in batch.observations.items()},
        robot_states=batch.robot_states[:1],
        state_mask=batch.state_mask[:1],
        actions=batch.actions[:1],
        timestamps=batch.timestamps[:1],
        terminated=batch.terminated[:1],
        episode_ids=("a",),
    )
    with pytest.raises(ContractError, match="at least two"):
        model.train_step(single)


def test_lewm_rejects_modified_source_before_import(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    from embodied_jepa.models import lewm

    (tmp_path / "jepa.py").write_text("# modified upstream source\n")
    monkeypatch.setattr(lewm.subprocess, "check_output", lambda *a, **k: lewm.LEWM_REVISION)
    with pytest.raises(ContractError, match="integrity"):
        lewm.load_upstream(tmp_path)


def test_training_invalidates_latents(model, batch):
    latent, _, actions = inputs(model, batch)
    model.train_step(batch)
    with pytest.raises(ContractError, match="another model"):
        model.predict(latent, actions)
