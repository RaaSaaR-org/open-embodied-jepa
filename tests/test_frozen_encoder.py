"""TASK-065 ``models/frozen_encoder``: synthetic contract evidence only.

A tiny random-init DINOv2 of the pinned width stands in for the pinned weights (no pretrained weight
is read), so these tests show wiring, exactness and refusals, not any learned property.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from embodied_jepa.contracts import ContractError, SequenceBatch, StateSchema  # noqa: E402
from embodied_jepa.models import NativeJEPA  # noqa: E402
from embodied_jepa.models import frozen_encoder as fe  # noqa: E402

SCHEMA = StateSchema(("arm.q",), ("rad",), "fixture_v0")
FROZEN = {"frozen_encoder": "dinov2_small_cls", "latent_dim": 384, "hidden_dim": 32}


@pytest.fixture(autouse=True)
def bounded_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(int(os.environ.get("JEPA_TEST_THREADS", "4")))
    yield
    torch.set_num_threads(original)


def _tiny_dinov2(seed=7):
    from transformers import Dinov2Config, Dinov2Model

    config = Dinov2Config(
        hidden_size=384,
        num_hidden_layers=1,
        num_attention_heads=6,
        mlp_ratio=1,
        patch_size=14,
        image_size=224,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Dinov2Model(config)
    return model.float().eval()


@pytest.fixture(autouse=True)
def tiny_frozen_encoder(monkeypatch):
    monkeypatch.setattr(fe.FrozenEncoderMixin, "_load_frozen_module", lambda s, n: _tiny_dinov2())


def _require_lewm():
    if not Path("third_party/le-wm/jepa.py").exists():
        pytest.skip("optional pinned LeWM source absent; run scripts/fetch_lewm.py")


@pytest.fixture(params=["native_jepa", "leworldmodel"])
def backend(request):
    if request.param == "leworldmodel":
        _require_lewm()
    return fe.frozen_encoder_model(request.param)


def _batch(b=3, t=3, seed=0):
    rng = np.random.default_rng(seed)
    return SequenceBatch(
        observations={"onboard_rgb": rng.integers(0, 256, (b, t + 1, 112, 112, 3), np.uint8)},
        robot_states=np.zeros((b, t + 1, 1), np.float32),
        state_mask=np.ones((b, t + 1, 1), np.bool_),
        actions=rng.uniform(-1, 1, (b, t, 14)).astype(np.float32),
        timestamps=np.tile(np.arange(t + 1, dtype=np.float64), (b, 1)),
        terminated=np.zeros((b, t), np.bool_),
        episode_ids=tuple(f"e{i}" for i in range(b)),
        state_schema=SCHEMA,
    )


def _fit(model, batch):
    frames = batch.observations["onboard_rgb"].reshape(-1, 112, 112, 3)
    raw = model.frozen_features(frames)
    model.fit_frozen_feature_normalization(
        raw.mean(0), raw.std(0), training_episode_ids=list(batch.episode_ids)
    )
    return raw.reshape(batch.batch_size, -1, 384)


def _digest(model):
    d = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        d.update(key.encode())
        d.update(value.detach().cpu().contiguous().numpy().tobytes())
    return d.hexdigest()


def test_plain_backends_do_not_know_the_option():
    with pytest.raises(ContractError, match="unknown"):
        NativeJEPA(SCHEMA, config={"frozen_encoder": "dinov2_small_cls"})
    assert "frozen_encoder" not in NativeJEPA.defaults


def test_the_frozen_class_requires_the_option(backend):
    with pytest.raises(ContractError, match="must be one of"):
        backend(SCHEMA, config={"latent_dim": 384})
    with pytest.raises(ContractError, match="must be one of"):
        backend(SCHEMA, config=FROZEN | {"frozen_encoder": "other"})
    with pytest.raises(ContractError, match="supports the backends"):
        fe.frozen_encoder_model("jepa_wms")
    assert fe.frozen_encoder_model("native_jepa") is fe.frozen_encoder_model("native_jepa")


def test_frozen_encoder_is_not_a_parameter_and_never_moves(backend):
    model = backend(SCHEMA, seed=1, config=FROZEN)
    frozen_ids = {id(p) for p in model._frozen_module.parameters()}
    assert frozen_ids.isdisjoint({id(p) for p in model.parameters()})
    assert not any("_frozen_module" in k for k in model.state_dict())
    before = {k: v.clone() for k, v in model._frozen_module.state_dict().items()}
    batch = _batch()
    raw = _fit(model, batch)
    model.train_step_features(raw, batch.actions)
    model.train_step(batch)
    after = model._frozen_module.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_feature_cache_update_equals_the_frame_update(backend):
    """A frozen encoder makes the features a pure function of the frames: the cached-feature
    update is the frame update, bit for bit (CPU)."""
    batch = _batch()
    first = backend(SCHEMA, seed=2, config=FROZEN)
    second = backend(SCHEMA, seed=2, config=FROZEN)
    raw = _fit(first, batch)
    _fit(second, batch)
    for _ in range(2):
        a = first.train_step(batch)
        b = second.train_step_features(raw, batch.actions)
        assert a["loss"] == b["loss"]
    assert _digest(first) == _digest(second)
    assert first._injected_features is None and second._injected_features is None


def test_predict_features_is_the_rollout_in_raw_feature_space(backend):
    batch = _batch()
    model = backend(SCHEMA, seed=3, config=FROZEN)
    raw = _fit(model, batch)
    model.train_step_features(raw, batch.actions)
    predicted = model.predict_features(raw[:, 0], batch.actions)
    assert predicted.shape == (3, 3, 384) and predicted.dtype == np.float32
    # the same rollout through the planner-facing path, mapped back to raw space
    current = batch.observation(0)
    latent = model.encode(current.images, current.state)
    through = model.predict(latent, batch.actions[:, None])
    values = through.values[:, 0] * model.frozen_feature_scale + model.frozen_feature_mean
    np.testing.assert_allclose(values.cpu().numpy(), predicted, rtol=1e-5, atol=1e-5)
    zero = model.predict_features(raw[:, 0], np.zeros_like(batch.actions))
    assert not np.array_equal(zero, predicted)  # the actions reach the prediction
    with pytest.raises(ContractError):
        model.predict_features(raw[:, 0], batch.actions.astype(np.float64))


def test_normalization_is_train_only_and_frozen(backend):
    model = backend(SCHEMA, seed=1, config=FROZEN)
    batch = _batch()
    with pytest.raises(ContractError, match="normalization first"):
        model.train_step_features(np.zeros((3, 4, 384), np.float32), batch.actions)
    _fit(model, batch)
    assert "frozen_feature_normalization_episodes_sha256" in model.metadata
    with pytest.raises(ContractError, match="frozen"):
        model.fit_frozen_feature_normalization(
            np.zeros(384), np.ones(384), training_episode_ids=["x"]
        )
    fresh = backend(SCHEMA, seed=1, config=FROZEN)
    with pytest.raises(ContractError, match="unique"):
        fresh.fit_frozen_feature_normalization(
            np.zeros(384), np.ones(384), training_episode_ids=["x", "x"]
        )
    with pytest.raises(ContractError, match="width"):
        fresh.fit_frozen_feature_normalization(np.zeros(3), np.ones(3), training_episode_ids=["x"])
    fresh.fit_frozen_feature_normalization(np.zeros(384), np.zeros(384), training_episode_ids=["x"])
    assert float(fresh.frozen_feature_scale.min()) == pytest.approx(fe.FEATURE_FLOOR_STD)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"latent_dim": 48}, "latent_dim 384"),
        ({"state_fusion": True}, "state_fusion"),
        ({"readout_heads": True}, "state_fusion/readout_heads"),
        ({"cameras": ["onboard_rgb", "hand"]}, "single configured camera"),
    ],
)
def test_frozen_encoder_refuses_what_it_does_not_support(backend, override, message):
    with pytest.raises(ContractError, match=message):
        backend(SCHEMA, config=FROZEN | override)


def test_train_step_features_validates_its_inputs(backend):
    model = backend(SCHEMA, seed=1, config=FROZEN)
    batch = _batch()
    raw = _fit(model, batch)
    with pytest.raises(ContractError, match="must agree"):
        model.train_step_features(raw[:, :-1], batch.actions)
    with pytest.raises(ContractError):
        model.train_step_features(raw, batch.actions.astype(np.float64))
    with pytest.raises(ContractError, match="within"):
        model.train_step_features(raw, batch.actions * 3)
    bad = raw.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ContractError, match="finite"):
        model.train_step_features(bad, batch.actions)
    assert model._injected_features is None


def test_checkpoint_carries_the_frozen_digest(backend, tmp_path):
    batch = _batch()
    model = backend(SCHEMA, seed=4, config=FROZEN)
    raw = _fit(model, batch)
    model.train_step_features(raw, batch.actions)
    path = tmp_path / "m.pt"
    model.save(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["metadata"][fe.DIGEST_KEY] == model.frozen_encoder_digest
    assert not any("_frozen_module" in k for k in state["weights"])
    again = backend(SCHEMA, seed=4, config=FROZEN)
    again.load(path)
    np.testing.assert_array_equal(
        again.predict_features(raw[:, 0], batch.actions),
        model.predict_features(raw[:, 0], batch.actions),
    )
    state["metadata"][fe.DIGEST_KEY] = "0" * 64
    torch.save(state, path)
    with pytest.raises(ContractError, match=fe.DIGEST_KEY):
        backend(SCHEMA, seed=4, config=FROZEN).load(path)
    with pytest.raises(ContractError, match="differs from the loaded"):
        backend(SCHEMA, config=FROZEN, metadata={fe.DIGEST_KEY: "0" * 64})


def test_a_different_frozen_encoder_is_a_different_model(backend, monkeypatch):
    first = backend(SCHEMA, config=FROZEN)
    monkeypatch.setattr(
        fe.FrozenEncoderMixin, "_load_frozen_module", lambda s, n: _tiny_dinov2(seed=8)
    )
    second = backend(SCHEMA, config=FROZEN)
    assert first.frozen_encoder_digest != second.frozen_encoder_digest


def test_implementation_hash_covers_the_backend_and_the_feature_code():
    native = fe.frozen_encoder_model("native_jepa")(SCHEMA, config=FROZEN)
    plain = NativeJEPA(SCHEMA, config={"latent_dim": 384, "hidden_dim": 32})
    assert native.implementation_sha256 != plain.implementation_sha256
    _require_lewm()
    lewm = fe.frozen_encoder_model("leworldmodel")(SCHEMA, config=FROZEN)
    assert lewm.implementation_sha256 != native.implementation_sha256


def test_backend_swap_is_one_key():
    """The same frozen configuration trains either backend; only the backend name differs."""
    _require_lewm()
    batch = _batch()
    for name in ("native_jepa", "leworldmodel"):
        model = fe.frozen_encoder_model(name)(SCHEMA, seed=5, config=FROZEN)
        raw = _fit(model, batch)
        assert np.isfinite(model.train_step_features(raw, batch.actions)["loss"])
        assert model.predict_features(raw[:, 0], batch.actions).shape == (3, 3, 384)


def test_mps_trains_on_cpu_features(backend):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    batch = _batch()
    model = backend(SCHEMA, device="mps", seed=6, config=FROZEN)
    raw = _fit(model, batch)
    assert model.frozen_feature_mean.device.type == "mps"
    assert next(model._frozen_module.parameters()).device.type == "cpu"
    assert np.isfinite(model.train_step_features(raw, batch.actions)["loss"])
    predicted = model.predict_features(raw[:, 0], batch.actions)
    assert predicted.shape == (3, 3, 384) and np.isfinite(predicted).all()
