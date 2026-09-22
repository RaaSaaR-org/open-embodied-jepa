"""Shared state fusion and readout heads on both backends (synthetic; no learned claim)."""

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from embodied_jepa.contracts import (  # noqa: E402
    ContractError,
    ReadoutWorldModel,
    RobotState,
    SequenceBatch,
    StateSchema,
)
from embodied_jepa.models import NativeJEPA  # noqa: E402
from embodied_jepa.models.readout import READOUT_NAMES, READOUTS  # noqa: E402

SCHEMA = StateSchema(("q0", "q1"), ("rad", "rad"), "fixture_v0")


@pytest.fixture(autouse=True)
def bounded_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(int(os.environ.get("JEPA_TEST_THREADS", "4")))
    yield
    torch.set_num_threads(original)


def model_class(name):
    if name == "native":
        return NativeJEPA
    pytest.importorskip("transformers")
    if not Path("third_party/le-wm/jepa.py").exists():
        pytest.skip("optional pinned LeWM source absent; run scripts/fetch_lewm.py")
    from embodied_jepa.models import LeWM

    return LeWM


SHARED = {
    "image_size": 28,
    "latent_dim": 24,
    "hidden_dim": 32,
    "readout_hidden_dim": 16,
    "state_fusion": True,
    "readout_heads": True,
    "candidate_chunk_size": 2,
}


@pytest.fixture(params=["native", "lewm"])
def model(request):
    extra = {"encoder_heads": 3} if request.param == "lewm" else {}
    model = model_class(request.param)(SCHEMA, seed=5, config=SHARED | extra)
    model.fit_state_normalization(
        [0.0, 1.0], [1.0, 0.0], training_episode_ids=["train-a", "train-b"]
    )
    return model


def fixture(batch=3, horizon=4):
    rng = np.random.default_rng(1)
    sequence = SequenceBatch(
        observations={
            "onboard_rgb": rng.integers(0, 256, (batch, horizon + 1, 28, 28, 3), dtype=np.uint8)
        },
        robot_states=rng.normal(size=(batch, horizon + 1, 2)).astype(np.float32),
        state_mask=np.ones((batch, horizon + 1, 2), bool),
        actions=rng.uniform(-1, 1, (batch, horizon, 14)).astype(np.float32),
        timestamps=np.tile(np.arange(horizon + 1, dtype=np.float64), (batch, 1)),
        terminated=np.zeros((batch, horizon), bool),
        episode_ids=tuple(f"e{i}" for i in range(batch)),
        state_schema=SCHEMA,
    )
    targets = {
        spec.name: rng.uniform(0, 1, (batch, horizon + 1, spec.width)).astype(np.float32)
        for spec in READOUTS
    }
    return sequence, targets


def test_declared_readouts_on_encoded_and_predicted_latents(model):
    sequence, targets = fixture()
    assert isinstance(model, ReadoutWorldModel)
    assert model.capabilities.readouts == READOUT_NAMES
    metrics = model.train_step(sequence, readout_targets=targets)
    assert np.isfinite(metrics["readout_predicted_loss"])
    current = sequence.observation(0)
    z = model.encode(current.images, current.state)
    encoded = model.readout(z)
    predicted = model.readout(model.predict(z, np.repeat(sequence.actions[:, None], 2, axis=1)))
    for spec in READOUTS:
        assert encoded[spec.name].shape == (3, spec.width)
        assert predicted[spec.name].shape == (3, 2, 4, spec.width)
        assert not predicted[spec.name].flags.writeable
    for name in ("hand_contact", "apple_held"):
        assert ((predicted[name] >= 0) & (predicted[name] <= 1)).all()
    stats = model.latent_statistics([z, z])
    assert stats["samples"] == 6 and stats["dimension"] == 24
    image = model.image_embedding_statistics([current.images])
    assert image["samples"] == 3 and image["dimension"] == 24


def test_state_fusion_uses_current_proprioception_and_has_no_image_goal(model):
    sequence, _ = fixture()
    current = sequence.observation(0)
    z = model.encode(current.images, current.state)
    moved = RobotState(
        current.state.values + 1.0, current.state.mask, current.state.timestamps, SCHEMA
    )
    other = model.encode(current.images, moved)
    assert not np.allclose(z.values.cpu().numpy(), other.values.cpu().numpy())
    with pytest.raises(ContractError, match="image-only goal"):
        model.encode_goal(current.images)


def test_readout_training_requires_targets_and_frozen_normalization(model):
    sequence, targets = fixture()
    with pytest.raises(ContractError, match="readout targets"):
        model.train_step(sequence)
    with pytest.raises(ContractError, match="must be an array"):
        model.train_step(sequence, readout_targets={k: v[:, :2] for k, v in targets.items()})
    with pytest.raises(ContractError, match="frozen"):
        model.fit_state_normalization([0, 0], [1, 1], training_episode_ids=["x"])


def test_unfitted_state_fusion_is_refused():
    model = NativeJEPA(SCHEMA, seed=1, config=SHARED)
    sequence, targets = fixture()
    with pytest.raises(ContractError, match="normalization"):
        model.train_step(sequence, readout_targets=targets)


def test_readout_checkpoint_roundtrip_and_foreign_latents(model, tmp_path):
    sequence, targets = fixture()
    model.train_step(sequence, readout_targets=targets)
    path = tmp_path / "model.pt"
    model.save(path)
    restored = type(model)(SCHEMA, seed=5, config=model.config, metadata=model.metadata)
    restored.load(path)
    current = sequence.observation(0)
    expected = model.readout(model.encode(current.images, current.state))
    actual = restored.readout(restored.encode(current.images, current.state))
    for name in READOUT_NAMES:
        np.testing.assert_allclose(expected[name], actual[name], rtol=1e-5, atol=1e-6)
    with pytest.raises(ContractError, match="another model"):
        restored.readout(model.encode(current.images, current.state))


def test_disabled_extensions_keep_the_image_only_contract():
    model = NativeJEPA(SCHEMA, seed=1, config={"image_size": 28, "hidden_dim": 32})
    sequence, targets = fixture()
    assert model.capabilities.readouts == ()
    with pytest.raises(ContractError, match="declares no readouts"):
        current = sequence.observation(0)
        model.readout(model.encode(current.images, current.state))
    with pytest.raises(ContractError, match="require readout_heads"):
        model.train_step(sequence, readout_targets=targets)
    assert "readout_head" not in dict(model.named_modules())


def test_regression_loss_ignores_frames_with_a_dropped_apple():
    from embodied_jepa.models.readout import ReadoutHeads

    heads = ReadoutHeads(4, 8)
    latent = torch.randn(2, 5, 4)
    targets = {s.name: torch.zeros(2, 5, s.width) for s in READOUTS}
    base, _ = heads.loss(latent, targets)
    far = dict(targets)
    far["palm_minus_apple"] = targets["palm_minus_apple"].clone()
    far["palm_minus_apple"][:, 2] = 3.0  # an apple lying on the floor, metres away
    far["apple_dropped"] = targets["apple_dropped"].clone()
    far["apple_dropped"][:, 2] = 1.0
    masked, terms = heads.loss(latent, far)
    unmasked, _ = heads.loss(latent, far | {"apple_dropped": targets["apple_dropped"]})
    assert unmasked > masked
    assert torch.isfinite(masked) and terms["apple_dropped"] > 0
