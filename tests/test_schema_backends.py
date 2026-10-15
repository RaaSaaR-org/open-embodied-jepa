"""SO-101 S0 (TASK-096): native and LeWM backends sized by the action schema, on CPU with tiny
sizes and synthetic data. Contract evidence only; no learning or manipulation claim."""

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from embodied_jepa.action_schemas import JOINT_DELTA_SO101_V0  # noqa: E402
from embodied_jepa.contracts import (  # noqa: E402
    EE_DELTA_GRASP_V0,
    ContractError,
    SequenceBatch,
    StateSchema,
)
from embodied_jepa.models import NativeJEPA, schema_backends  # noqa: E402
from embodied_jepa.models.schema_backends import SchemaNativeJEPA, create_model  # noqa: E402
from embodied_jepa.schema_planning import SchemaCEMConfig, SchemaCEMPlanner  # noqa: E402

SO101 = JOINT_DELTA_SO101_V0
CONFIG = {"candidate_chunk_size": 2, "hidden_dim": 48}
METADATA = {"dataset_hash": "synthetic-fixture", "action_hash": "test-action-v0"}


@pytest.fixture(autouse=True)
def bounded_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(int(os.environ.get("JEPA_TEST_THREADS", "4")))
    yield
    torch.set_num_threads(original)


def make_batch(schema):
    rng = np.random.default_rng(42)
    return SequenceBatch(
        observations={"onboard_rgb": rng.integers(0, 256, (3, 3, 32, 40, 3), dtype=np.uint8)},
        robot_states=np.zeros((3, 3, 1), dtype=np.float32),
        state_mask=np.ones((3, 3, 1), dtype=np.bool_),
        actions=rng.uniform(-1, 1, (3, 2, schema.dimension)).astype(np.float32),
        timestamps=np.tile(np.arange(3, dtype=np.float64), (3, 1)),
        terminated=np.zeros((3, 2), dtype=np.bool_),
        episode_ids=("a", "b", "c"),
        state_schema=StateSchema(("arm.q",), ("rad",), "fixture_v0"),
        action_schema=schema,
    )


def lewm_available():
    pytest.importorskip("transformers")
    if not Path("third_party/le-wm/jepa.py").exists():
        pytest.skip("optional pinned LeWM source absent; run scripts/fetch_lewm.py")


def pinned_class(backend):
    if backend == "native_jepa":
        return NativeJEPA
    lewm_available()
    from embodied_jepa.models import LeWM

    return LeWM


def schema_class(backend):
    if backend == "leworldmodel":
        lewm_available()
        return schema_backends.SchemaLeWM
    return SchemaNativeJEPA


BACKENDS = ["native_jepa", "leworldmodel"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_six_d_backend_trains_predicts_plans_and_reloads(backend, tmp_path):
    cls = schema_class(backend)
    batch = make_batch(SO101)
    model = cls(batch.state_schema, seed=3, config=CONFIG, metadata=METADATA, action_schema=SO101)
    assert model.capabilities.action_schema is SO101
    assert model.action_dim == 6
    for _ in range(2):
        metrics = model.train_step(batch)
        assert all(np.isfinite(value) for value in metrics.values())
    diagnostics = model.diagnostics(batch)
    assert diagnostics["action_effect_rms"] > 0
    current = batch.observation(0)
    latent = model.encode(current.images, current.state)
    goal = model.encode_goal(batch.observation(2).images)
    actions = np.repeat(batch.actions[:, None], 5, axis=1)
    cost = model.distance(model.predict(latent, actions), goal)
    assert cost.shape == (3, 5, 2) and cost.dtype == np.float32 and np.isfinite(cost).all()
    with pytest.raises(ContractError, match="6 components"):
        model.predict(latent, np.zeros((3, 5, 2, 14), np.float32))
    with pytest.raises(ContractError, match="incompatible"):
        model.train_step(make_batch(EE_DELTA_GRASP_V0))
    plan = SchemaCEMPlanner(SchemaCEMConfig(SO101, horizon=2, samples=8, elites=2)).plan(
        model, latent, goal, batch_size=3
    )
    assert plan.actions.shape == (3, 2, 6)
    path = tmp_path / "so101.pt"
    model.save(path)
    restored = cls(
        batch.state_schema, config=model.config, seed=999, metadata=model.metadata,
        action_schema=SO101,
    )  # fmt: skip
    restored.load(path)
    a = restored.encode(current.images, current.state)
    g = restored.encode_goal(batch.observation(2).images)
    np.testing.assert_array_equal(cost, restored.distance(restored.predict(a, actions), g))
    assert restored.train_step(batch) == model.train_step(batch)
    # A schema-sized checkpoint never loads into the pinned 14-D backend, or the reverse.
    pinned = pinned_class(backend)(batch.state_schema, config=CONFIG, metadata=METADATA)
    with pytest.raises(ContractError, match="incompatible"):
        pinned.load(path)
    pinned_path = tmp_path / "pinned.pt"
    pinned.save(pinned_path)
    with pytest.raises(ContractError, match="incompatible"):
        restored.load(pinned_path)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fourteen_d_mirror_matches_the_pinned_backend_exactly(backend):
    """The mirrored __init__ builds the pinned backend's modules in its order and RNG stream."""
    batch = make_batch(EE_DELTA_GRASP_V0)
    pinned = pinned_class(backend)(batch.state_schema, seed=5, config=CONFIG, metadata=METADATA)
    mirror = schema_class(backend)(
        batch.state_schema, seed=5, config=CONFIG, metadata=METADATA,
        action_schema=EE_DELTA_GRASP_V0,
    )  # fmt: skip
    assert pinned.state_dict().keys() == mirror.state_dict().keys()
    for key, value in pinned.state_dict().items():
        torch.testing.assert_close(value, mirror.state_dict()[key], rtol=0, atol=0)
    assert pinned.train_step(batch) == mirror.train_step(batch)
    assert pinned.implementation_sha256 != mirror.implementation_sha256
    assert pinned.capabilities == mirror.capabilities


@pytest.mark.parametrize("backend", BACKENDS)
def test_create_model_keeps_fourteen_d_on_the_pinned_classes(backend):
    pinned = pinned_class(backend)
    state = StateSchema(("arm.q",), ("rad",), "fixture_v0")
    g1 = create_model(backend, state_schema=state, action_schema=EE_DELTA_GRASP_V0, config=CONFIG)
    assert type(g1) is pinned
    so101 = create_model(backend, state_schema=state, action_schema=SO101, config=CONFIG)
    assert type(so101) is schema_class(backend)
    assert so101.capabilities.action_schema is SO101
    with pytest.raises(ContractError, match="action_chunk"):
        create_model(
            backend, state_schema=state, action_schema=SO101, config=CONFIG | {"action_chunk": 2}
        )


def test_create_model_refuses_backends_without_a_schema_sized_variant():
    state = StateSchema(("arm.q",), ("rad",), "fixture_v0")
    with pytest.raises(ContractError, match="schema-sized backends"):
        create_model("jepa_wms", state_schema=state, action_schema=SO101)
