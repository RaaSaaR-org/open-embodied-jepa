"""TASK-066 ``models/frozen_tokens``: synthetic contract evidence only.

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
from embodied_jepa.models import frozen_tokens as ft  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = StateSchema(("arm.q",), ("rad",), "fixture_v0")
GRID = 2
DIM = GRID * GRID * 384
FROZEN = {
    "frozen_encoder": "dinov2_small_tokens",
    "token_grid": GRID,
    "latent_dim": DIM,
    "hidden_dim": 32,
}


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
    # Both mixins: some tests build TASK-065's CLS class too, and CI has no pinned weights.
    monkeypatch.setattr(ft.FrozenTokenMixin, "_load_frozen_module", lambda s, n: _tiny_dinov2())
    monkeypatch.setattr(fe.FrozenEncoderMixin, "_load_frozen_module", lambda s, n: _tiny_dinov2())


def _require_lewm():
    if not (ROOT / "third_party/le-wm/jepa.py").exists():
        pytest.skip("optional pinned LeWM source absent; run scripts/fetch_lewm.py")


@pytest.fixture(params=["native_jepa", "leworldmodel"])
def backend(request):
    if request.param == "leworldmodel":
        _require_lewm()
    return ft.frozen_token_model(request.param)


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
    return raw.reshape(batch.batch_size, -1, DIM)


def _digest(model):
    d = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        d.update(key.encode())
        d.update(value.detach().cpu().contiguous().numpy().tobytes())
    return d.hexdigest()


# ----- pooling -----------------------------------------------------------------------------------
def test_pool_tokens_is_the_block_mean_in_row_major_order():
    rng = np.random.default_rng(0)
    grid = rng.normal(size=(2, 16, 16, 384)).astype(np.float32)
    pooled = ft.pool_tokens(grid.reshape(2, -1), 4)
    assert pooled.shape == (2, 16 * 384) and pooled.dtype == np.float32
    cells = pooled.reshape(2, 4, 4, 384)
    want = grid[:, 4:8, 8:12].astype(np.float64).mean(axis=(1, 2))  # grid row 1, column 2
    np.testing.assert_array_equal(cells[:, 1, 2], want.astype(np.float32))
    np.testing.assert_array_equal(ft.pool_tokens(grid.reshape(2, -1), 16), grid.reshape(2, -1))
    one = ft.pool_tokens(grid.reshape(2, -1), 1)
    want_one = grid.astype(np.float64).mean(axis=(1, 2)).astype(np.float32)
    np.testing.assert_array_equal(one, want_one)
    for bad in (3, 0, 5):
        with pytest.raises(ContractError, match="divide"):
            ft.pool_tokens(grid.reshape(2, -1), bad)
    with pytest.raises(ContractError, match="tokens must be"):
        ft.pool_tokens(grid.reshape(2, -1)[:, :-1], 4)


def test_pooling_does_not_depend_on_the_batch():
    rng = np.random.default_rng(1)
    tokens = rng.normal(size=(5, 16 * 16 * 384))
    whole = ft.pool_tokens(tokens, 4)
    split = np.concatenate([ft.pool_tokens(tokens[:2], 4), ft.pool_tokens(tokens[2:], 4)])
    np.testing.assert_array_equal(whole, split)


def test_frozen_features_are_the_pooled_pretrained_tokens(backend):
    from embodied_jepa import pretrained_encoder as pe

    model = backend(SCHEMA, config=FROZEN)
    frames = np.random.default_rng(2).integers(0, 256, (3, 112, 112, 3), np.uint8)
    tokens = pe.features(model._frozen_module, frames)["tokens"]
    np.testing.assert_array_equal(model.frozen_features(frames), ft.pool_tokens(tokens, GRID))


# ----- the option is off unless selected ----------------------------------------------------------
def test_plain_and_cls_classes_do_not_know_the_option():
    with pytest.raises(ContractError, match="unknown"):
        NativeJEPA(SCHEMA, config={"token_grid": 4})
    assert "token_grid" not in NativeJEPA.defaults
    with pytest.raises(ContractError, match="unknown"):
        fe.frozen_encoder_model("native_jepa")(
            SCHEMA,
            config={"frozen_encoder": "dinov2_small_cls", "latent_dim": 384, "token_grid": 4},
        )


def test_the_token_class_requires_the_option(backend):
    with pytest.raises(ContractError, match="must be one of"):
        backend(SCHEMA, config={"latent_dim": DIM, "token_grid": GRID})
    with pytest.raises(ContractError, match="must be one of"):
        backend(SCHEMA, config=FROZEN | {"frozen_encoder": "dinov2_small_cls"})
    with pytest.raises(ContractError, match="positive divisor"):
        backend(SCHEMA, config=FROZEN | {"token_grid": None})
    with pytest.raises(ContractError, match="positive divisor"):
        backend(SCHEMA, config=FROZEN | {"token_grid": 3})
    with pytest.raises(ContractError, match="supports the backends|support the backends"):
        ft.frozen_token_model("jepa_wms")
    assert ft.frozen_token_model("native_jepa") is ft.frozen_token_model("native_jepa")


@pytest.mark.parametrize(
    "override, message",
    [
        ({"latent_dim": 384}, f"latent_dim {DIM}"),
        ({"state_fusion": True}, "state_fusion"),
        ({"readout_heads": True}, "readout_heads"),
        ({"cameras": ["onboard_rgb", "hand"]}, "cameras"),
        ({"action_chunk": 2}, "action_chunk"),
        ({"predictor_step_embedding": True}, "predictor_step_embedding"),
        ({"multistep_tail_weight": 1.0}, "multistep_tail_weight"),
    ],
)
def test_frozen_tokens_refuse_what_they_do_not_support(backend, override, message):
    with pytest.raises(ContractError, match=message):
        backend(SCHEMA, config=FROZEN | override)


def test_no_existing_model_file_changes():
    """TASK-054's E0 and TASK-065's twelve checkpoints enforce implementation hashes over these.

    TASK-072 changed base.py (CUDA support), so those checkpoints load only at their recorded
    revision (benchmarks/manifests/task072-checkpoint-compatibility.json). For base.py the pin
    is checked against the bytes they were written with; every other file still at HEAD.
    """
    import json

    manifest = json.loads((ROOT / "benchmarks/manifests/apple-latent-dynamics-v1.json").read_text())
    retired = json.loads(
        (ROOT / "benchmarks/manifests/task072-checkpoint-compatibility.json").read_text()
    )["changed_files"]
    for path in (
        "src/embodied_jepa/models/base.py",
        "src/embodied_jepa/models/lewm.py",
        "src/embodied_jepa/models/native.py",
        "src/embodied_jepa/models/readout.py",
        "src/embodied_jepa/readout_labels.py",
        "src/embodied_jepa/models/frozen_encoder.py",
        "src/embodied_jepa/pretrained_encoder.py",
    ):
        if path in retired:
            head = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            assert head == retired[path]["sha256_after"], f"{path} changed again; record it"
            got = retired[path]["sha256_before"]
        else:
            got = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        assert got == manifest["hashes"][path], path


# ----- the frozen encoder, reused from TASK-065 ---------------------------------------------------
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
    assert isinstance(model, fe.FrozenEncoderMixin)  # the TASK-065 mixin, reused


def test_feature_cache_update_equals_the_frame_update(backend):
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
    assert predicted.shape == (3, 3, DIM) and predicted.dtype == np.float32
    current = batch.observation(0)
    latent = model.encode(current.images, current.state)
    through = model.predict(latent, batch.actions[:, None])
    values = through.values[:, 0] * model.frozen_feature_scale + model.frozen_feature_mean
    np.testing.assert_allclose(values.cpu().numpy(), predicted, rtol=1e-5, atol=1e-5)
    zero = model.predict_features(raw[:, 0], np.zeros_like(batch.actions))
    assert not np.array_equal(zero, predicted)  # the actions reach the prediction


# ----- the token predictor -----------------------------------------------------------------------
def test_each_predicted_token_sees_every_token_of_its_frame(backend):
    """Bidirectional within the frame: perturbing the last token moves the first token's
    prediction (a causal raster mask would not), and vice versa. A few updates first: upstream's
    AdaLN-zero blocks are the identity at initialisation."""
    model = backend(SCHEMA, seed=4, config=FROZEN)
    batch = _batch()
    raw = _fit(model, batch)
    for _ in range(5):
        model.train_step_features(raw, batch.actions)
    model.eval()
    rng = np.random.default_rng(5)
    x = torch.as_tensor(rng.normal(size=(4, DIM)).astype(np.float32))
    a = torch.as_tensor(rng.uniform(-1, 1, (4, 14)).astype(np.float32))
    with torch.no_grad():
        base = model.next_embedding(x, a).reshape(4, GRID * GRID, 384)
        last = x.clone().reshape(4, GRID * GRID, 384)
        last[:, -1] += 1.0
        moved = model.next_embedding(last.reshape(4, DIM), a).reshape(4, GRID * GRID, 384)
        first = x.clone().reshape(4, GRID * GRID, 384)
        first[:, 0] += 1.0
        moved_first = model.next_embedding(first.reshape(4, DIM), a).reshape(4, GRID * GRID, 384)
    assert not torch.allclose(base[:, 0], moved[:, 0])
    assert not torch.allclose(base[:, -1], moved_first[:, -1])


def test_the_lewm_token_predictor_is_the_pinned_upstream_one():
    _require_lewm()
    model = ft.frozen_token_model("leworldmodel")(SCHEMA, seed=6, config=FROZEN)
    upstream = type(model.model).__module__
    assert upstream == "embodied_jepa_upstream_jepa"
    predictor = model.model.predictor
    assert type(predictor).__name__ == "ARPredictor"
    assert predictor.pos_embedding.shape == (1, GRID * GRID, 384)  # one per token
    for block in predictor.transformer.layers:
        assert type(block).__name__ == "ConditionalBlock"
        assert isinstance(block.attn, ft._WithinFrameAttention)
        assert type(block.attn).__mro__[1].__name__ == "Attention"  # the upstream class
    assert type(model.model.action_encoder).__name__ == "Embedder"
    assert type(model.model.pred_proj).__name__ == "MLP"
    assert not any(k.startswith("model.encoder") for k in model.state_dict())


def test_action_conditioning_reaches_every_token(backend):
    model = backend(SCHEMA, seed=7, config=FROZEN)
    batch = _batch()
    raw = _fit(model, batch)
    for _ in range(3):
        model.train_step_features(raw, batch.actions)
    rng = np.random.default_rng(8)
    x = raw[:, 0]
    one = model.predict_features(x, rng.uniform(-1, 1, (3, 2, 14)).astype(np.float32))
    two = model.predict_features(x, rng.uniform(-1, 1, (3, 2, 14)).astype(np.float32))
    diff = np.abs(one - two).reshape(3, 2, GRID * GRID, 384).max(axis=(0, 1, 3))
    assert (diff > 0).all()


def test_checkpoint_round_trip_and_digest(backend, tmp_path):
    batch = _batch()
    model = backend(SCHEMA, seed=9, config=FROZEN)
    raw = _fit(model, batch)
    model.train_step_features(raw, batch.actions)
    path = tmp_path / "m.pt"
    model.save(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["metadata"][fe.DIGEST_KEY] == model.frozen_encoder_digest
    again = backend(SCHEMA, seed=9, config=FROZEN)
    again.load(path)
    np.testing.assert_array_equal(
        again.predict_features(raw[:, 0], batch.actions),
        model.predict_features(raw[:, 0], batch.actions),
    )
    state["metadata"][fe.DIGEST_KEY] = "0" * 64
    torch.save(state, path)
    with pytest.raises(ContractError, match=fe.DIGEST_KEY):
        backend(SCHEMA, seed=9, config=FROZEN).load(path)


def test_implementation_hash_covers_the_reused_and_new_code():
    native = ft.frozen_token_model("native_jepa")(SCHEMA, config=FROZEN)
    cls = fe.frozen_encoder_model("native_jepa")(
        SCHEMA, config={"frozen_encoder": "dinov2_small_cls", "latent_dim": 384}
    )
    assert native.implementation_sha256 != cls.implementation_sha256
    _require_lewm()
    lewm = ft.frozen_token_model("leworldmodel")(SCHEMA, config=FROZEN)
    assert lewm.implementation_sha256 != native.implementation_sha256


def test_backend_swap_is_one_key():
    """The same configuration trains either backend; only the backend name differs."""
    _require_lewm()
    batch = _batch()
    for name in ("native_jepa", "leworldmodel"):
        model = ft.frozen_token_model(name)(SCHEMA, seed=5, config=FROZEN)
        raw = _fit(model, batch)
        assert np.isfinite(model.train_step_features(raw, batch.actions)["loss"])
        assert model.predict_features(raw[:, 0], batch.actions).shape == (3, 3, DIM)


def test_mps_trains_on_cpu_features(backend):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    batch = _batch()
    model = backend(SCHEMA, device="mps", seed=6, config=FROZEN)
    raw = _fit(model, batch)
    assert next(model._frozen_module.parameters()).device.type == "cpu"
    assert np.isfinite(model.train_step_features(raw, batch.actions)["loss"])
    predicted = model.predict_features(raw[:, 0], batch.actions)
    assert predicted.shape == (3, 3, DIM) and np.isfinite(predicted).all()


def test_the_upstream_causal_mask_would_hide_later_tokens():
    """The other direction of the within-frame test: with the upstream mask restored, the first
    token's prediction ignores the last token. So the adaptation is what makes the grid visible."""
    _require_lewm()
    model = ft.frozen_token_model("leworldmodel")(SCHEMA, seed=4, config=FROZEN)
    batch = _batch()
    raw = _fit(model, batch)
    for _ in range(5):
        model.train_step_features(raw, batch.actions)
    for block in model.model.predictor.transformer.layers:
        block.attn.__class__ = type(block.attn).__mro__[1]  # back to upstream Attention
    model.eval()
    rng = np.random.default_rng(5)
    x = torch.as_tensor(rng.normal(size=(4, DIM)).astype(np.float32))
    a = torch.as_tensor(rng.uniform(-1, 1, (4, 14)).astype(np.float32))
    with torch.no_grad():
        base = model.next_embedding(x, a).reshape(4, GRID * GRID, 384)
        last = x.clone().reshape(4, GRID * GRID, 384)
        last[:, -1] += 1.0
        moved = model.next_embedding(last.reshape(4, DIM), a).reshape(4, GRID * GRID, 384)
    torch.testing.assert_close(base[:, 0], moved[:, 0])
    assert not torch.allclose(base[:, -1], moved[:, -1])
