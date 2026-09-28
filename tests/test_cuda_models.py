"""CUDA model and training checks; each skips with "CUDA unavailable" on hosts without a GPU.

Software checks only (finite losses, isolated RNG, bit-identical same-seed runs); they make no
learned-control claim.
"""

import json
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import devices

torch = pytest.importorskip("torch")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.fixture
def batch():
    from embodied_jepa.contracts import SequenceBatch, StateSchema

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
        from embodied_jepa.models import NativeJEPA

        return NativeJEPA
    pytest.importorskip("transformers")
    if not Path("third_party/le-wm/jepa.py").exists():
        pytest.skip("optional pinned LeWM source absent; run scripts/fetch_lewm.py")
    from embodied_jepa.models import LeWM

    return LeWM


@needs_cuda
@pytest.mark.parametrize("backend", ["native", "lewm"])
def test_cuda_forward_backward_costs_and_determinism_flags(backend, batch, restore_determinism):
    model = model_class(backend)(batch.state_schema, device="cuda", seed=4)
    state = devices.determinism_state()
    assert state["deterministic_algorithms"] and state["cudnn_deterministic"]
    assert state["float32_matmul_precision"] == "highest"
    metrics = model.train_step(batch)
    assert all(np.isfinite(value) for value in metrics.values())
    current = batch.observation(0)
    latent = model.encode(current.images, current.state)
    goal = model.encode_goal(batch.observation(2).images)
    actions = np.repeat(batch.actions[:, None], 5, axis=1)
    costs = model.distance(model.predict(latent, actions), goal)
    assert np.isfinite(costs).all()
    assert all(np.isfinite(value) for value in model.diagnostics(batch).values())


@needs_cuda
@pytest.mark.parametrize("backend", ["native", "lewm"])
def test_cuda_same_seed_models_train_bit_identically(backend, batch, restore_determinism):
    losses = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            model = model_class(backend)(batch.state_schema, device="cuda", seed=11)
            losses.append([model.train_step(batch)["loss"] for _ in range(4)])
    assert losses[0] == losses[1]
    # The two known flagged kernels (see embodied_jepa.devices): native_jepa's pooling
    # backward and LeWM's memory-efficient attention backward. A new one should be looked at.
    known = ("adaptive_avg_pool2d_backward_cuda does not", "Memory Efficient attention defaults")
    flagged = [str(item.message) for item in caught if "deterministic" in str(item.message)]
    assert all(message.startswith(known) for message in flagged), flagged


@needs_cuda
def test_cuda_rng_is_isolated_and_checkpointed(batch, tmp_path, restore_determinism):
    import torch

    from embodied_jepa.models import NativeJEPA

    model = NativeJEPA(batch.state_schema, device="cuda", seed=5)
    before = torch.cuda.get_rng_state().clone()
    model.train_step(batch)
    assert torch.equal(before, torch.cuda.get_rng_state())
    assert model._cuda_rng is not None
    path = tmp_path / "model.pt"
    model.save(path)
    assert torch.equal(torch.load(path, weights_only=True)["rng_cuda"], model._cuda_rng)
    reloaded = NativeJEPA(batch.state_schema, device="cuda", seed=5)
    reloaded.load(path)
    assert torch.equal(reloaded._cuda_rng, model._cuda_rng)
    assert model.train_step(batch)["loss"] == reloaded.train_step(batch)["loss"]


@needs_cuda
def test_cuda_training_run_reports_device_determinism_and_memory(tmp_path, restore_determinism):
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    pytest.importorskip("PIL")
    from embodied_jepa.data import DatasetStore, deterministic_fixture
    from embodied_jepa.training import train

    fixture = deterministic_fixture()
    store = DatasetStore.create(
        tmp_path / "corpus",
        fps=20,
        state_schema=fixture.state_schema,
        action_manifest={"physical_scale": [0.01] * 14, "software_fixture_only": True},
        provenance={"source": "procedural unit-test fixture", "license": "Apache-2.0"},
        robot_type="software_fixture",
    )
    for index in range(6):
        store.write_episode(
            replace(
                fixture,
                episode_id=f"episode-{index}",
                session_id=f"session-{index}",
                observations={"onboard_rgb": np.roll(fixture.observations["head"], index, axis=2)},
                state_mask=np.ones_like(fixture.state_mask),
            )
        )
    store.freeze_splits(seed=3)
    runs = []
    for name in ("a", "b"):
        report = train(
            store.root,
            "native_jepa",
            tmp_path / f"{name}.pt",
            steps=4,
            batch_size=2,
            horizon=2,
            validation_every=2,
            validation_batches=1,
            max_seconds=120,
            device="cuda",
            model_config={"hidden_dim": 32},
        )
        assert report["status"] == "completed" and report["device"] == "cuda"
        assert report["environment"]["determinism"]["deterministic_algorithms"] is True
        assert report["environment"]["accelerator"]["cuda_device_name"]
        assert report["peak_cuda_allocated_bytes"] > 0
        curves = (tmp_path / f"{name}.metrics.jsonl").read_text().splitlines()
        events = [json.loads(line) for line in curves]
        runs.append([event["metrics"] for event in events if event["kind"] == "train"])
        assert len(runs[-1]) == 4
    assert runs[0] == runs[1]
