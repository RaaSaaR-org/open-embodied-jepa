"""Synthetic composition/provenance fixtures, never manipulation success evidence."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from embodied_jepa.contracts import ContractError, SequenceBatch, StateSchema  # noqa:E402
from embodied_jepa.models.aligned_sensor import (  # noqa:E402
    FIELDS,
    RECIPE,
    WEIGHTS,
    AlignedLatent,
    AlignedSensorWorldModel,
    hybrid_cost,
    json_hash,
    make_head,
    package_bundle,
    sha256,
)
from embodied_jepa.models.base import VisualLatent  # noqa:E402
from embodied_jepa.models.sensor import SensorWorldModel  # noqa:E402


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


@pytest.fixture
def source(tmp_path, request):
    folder = tmp_path / "inputs"
    folder.mkdir()
    parameters = getattr(request, "param", {})
    names = parameters.get("names", FIELDS)
    schema = StateSchema(
        names, tuple("rad" if name in FIELDS else "rad/s" for name in names), "aligned_synthetic_v0"
    )
    manifest = dict(
        state_schema=asdict(schema),
        action_manifest={"fixture": True},
        splits=dict(train=["train-a", "train-b"], val=["val"], test=["test"], heldout=[]),
        split_policy={"fixture": True},
        episodes=[
            dict(episode_id=e, length=57, metadata={})
            for e in ("train-a", "train-b", "val", "test")
        ],
    )
    dataset = write(folder / "dataset.json", manifest)
    metadata = dict(
        dataset_hash=sha256(dataset),
        action_hash=json_hash(manifest["action_manifest"]),
        split_hash=json_hash({"splits": manifest["splits"], "policy": manifest["split_policy"]}),
    )
    states = np.array(
        [[[0.0] * len(names), [0.05] * len(names)], [[1.0] * len(names), [1.05] * len(names)]],
        np.float32,
    )
    batch = SequenceBatch(
        {"onboard_rgb": np.full((2, 2, 8, 8, 3), 128, np.uint8)},
        states,
        np.ones_like(states, bool),
        np.zeros((2, 1, 14), np.float32),
        np.tile(np.array([0.0, 0.05]), (2, 1)),
        np.zeros((2, 1), bool),
        ("train-a", "train-b"),
        schema,
    )
    sensor = SensorWorldModel(
        schema,
        config=dict(hidden_dim=16, candidate_chunk_size=1, max_horizon=16),
        metadata=metadata,
    )
    sensor.fit_normalization([batch], training_episode_ids=batch.episode_ids)
    sensor_path = folder / "sensor.pt"
    sensor.save(sensor_path)
    samples = [dict(split="train", parent=e, episode_id=e, frame=0) for e in batch.episode_ids]
    samples_path = write(folder / "train_samples.json", samples)
    module = make_head()
    with torch.no_grad():
        for p in module.parameters():
            p.zero_()
        module[-1].bias[:7] = torch.tensor(parameters.get("bias", [0.0] * 7))
    head = dict(
        head=module.state_dict(),
        updates=2000,
        architecture=[1728, 128, 64, 14],
        fields=FIELDS,
        sample_sha256=json_hash(samples),
        config=dict(
            seed=0,
            batch_size=128,
            updates=2000,
            lr=1e-3,
            weight_decay=1e-4,
            gradient_clip=10,
            logvar_bounds=[-6, 3],
        ),
    )
    head_path = folder / "head.pt"
    torch.save(head, head_path)
    goal_source = folder / "goal_source.py"
    goal_source.write_text("# Synthetic provenance fixture, not fitted data.\n")
    registration = write(
        folder / "registration.json",
        dict(
            identities=dict(
                checkpoint=sha256(sensor_path),
                dataset=sha256(dataset),
                **{
                    "prepare/train_samples.json": sha256(samples_path),
                    "script": sha256(goal_source),
                },
            )
        ),
    )
    report = write(
        folder / "report.json",
        dict(status="completed", integrity_verified_after=True, updates=2000),
    )
    seal = write(
        folder / "seal.json",
        dict(artifacts={p.name: sha256(p) for p in (head_path, registration, report)}),
    )
    pairs = [
        dict(parent_id=e, frame0=t, frame1=t + 28, visual_mse=v, pose_mse=q)
        for e, v, q in [("train-a", 2.0, 4.0), ("train-b", 4.0, 8.0)]
        for t in (0, 28)
    ]
    ledger = write(folder / "ledger.json", dict(pairs=pairs))
    calibration = write(
        folder / "calibration.json",
        dict(
            format_version=1,
            recipe=RECIPE,
            stride=28,
            weights=WEIGHTS,
            scales=dict(visual=3.0, pose=6.0),
            fields=list(FIELDS),
            parent_ids=["train-a", "train-b"],
            dataset_hash=sha256(dataset),
            sensor_checkpoint_sha256=sha256(sensor_path),
            goal_head_sha256=sha256(head_path),
            ledger_sha256=sha256(ledger),
            packaging=dict(
                source_sha256="0" * 64,
                protocol_sha256="1" * 64,
                source_revision="synthetic_fixture",
            ),
        ),
    )
    provenance = dict(
        dataset_manifest=dataset,
        goal_registration=registration,
        goal_fit_report=report,
        goal_fit_seal=seal,
        goal_train_samples=samples_path,
        goal_source=goal_source,
        calibration_ledger=ledger,
    )
    return (
        dict(
            sensor_checkpoint=sensor_path,
            head_checkpoint=head_path,
            calibration_json=calibration,
            provenance_files=provenance,
        ),
        sensor,
        batch,
    )


def package(tmp_path, source):
    files, sensor, batch = source
    path = package_bundle(tmp_path / "bundle/model.pt", **files)
    envelope = torch.load(path, weights_only=True)
    model = AlignedSensorWorldModel(batch.state_schema, config=envelope["config"])
    model.load(path)
    return model, path, sensor, batch


def test_immutable_dynamics_and_head_equivalence_with_chunking(tmp_path, source):
    model, path, sensor, batch = package(tmp_path, source)
    observation = batch.observation(0)
    z = model.encode(observation.images, observation.state)
    actions = np.zeros((2, 3, 2, 14), np.float32)
    predicted = model.predict(z, actions)
    expected = sensor.predict(sensor.encode(observation.images, observation.state), actions)
    torch.testing.assert_close(predicted.child.values, expected.values, rtol=0, atol=0)
    goal = model.encode_goal(observation.images)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "task041_goal_head", Path(__file__).parents[1] / "scripts/apple_goal_alignment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    head = module.head_factory()
    head.load_state_dict(torch.load(source[0]["head_checkpoint"], weights_only=True)["head"])
    torch.testing.assert_close(
        goal.pose, head(sensor.encode_goal(observation.images).values)[:, :7], rtol=0, atol=0
    )
    costs = model.distance(predicted, goal)
    assert costs.shape == (2, 3, 2) and costs.dtype == np.float32 and np.isfinite(costs).all()
    assert not predicted.child.values.requires_grad and not goal.pose.requires_grad
    assert all(not p.requires_grad for p in model._sensor.parameters())


def test_visual_component_survives_identical_pose(tmp_path, source):
    model, _, _, batch = package(tmp_path, source)
    observation = batch.observation(0)
    goal = model.encode_goal(observation.images)
    prediction = model.predict(
        model.encode(observation.images, observation.state), np.zeros((2, 1, 1, 14), np.float32)
    )
    changed = prediction.child.values.clone()
    changed[..., :1728] += 1
    variant = AlignedLatent(VisualLatent(changed, model._sensor._owner), model._owner)
    baseline = model.distance(prediction, goal)
    different = model.distance(variant, goal)
    assert not np.array_equal(baseline, different)
    current = observation.images
    altered = {"onboard_rgb": np.full_like(current["onboard_rgb"], 160)}
    assert np.all(model.observed_distance(current, altered) > 0)
    np.testing.assert_array_equal(
        model.observed_distance(current, current), np.zeros(2, np.float32)
    )


def test_distance_matches_fixed_formula_and_progress_is_image_only(tmp_path, source):
    model, _, _, batch = package(tmp_path, source)
    obs = batch.observation(0)
    goal = model.encode_goal(obs.images)
    prediction = model.predict(
        model.encode(obs.images, obs.state), np.zeros((2, 2, 3, 14), np.float32)
    )
    visual = model._sensor.distance(prediction.child, goal.visual)
    q = (prediction.child.values[..., 1728:] - goal.pose[:, None, None]).square().mean(-1).numpy()
    expected = 0.5 * visual / 3.0 + 0.5 * q / 6.0
    np.testing.assert_allclose(model.distance(prediction, goal), expected, rtol=1e-5, atol=1e-7)
    altered = replace(obs.state, values=obs.state.values + 10)
    model.encode(obs.images, altered)
    np.testing.assert_array_equal(
        model.observed_distance(obs.images, obs.images), np.zeros(2, np.float32)
    )
    with pytest.raises(TypeError):
        model.encode_goal(obs.images, altered)
    with pytest.raises(TypeError):
        model.observed_distance(obs.images, obs.images, altered)


def test_reload_invalidates_owner_and_save_roundtrip(tmp_path, source):
    model, path, _, batch = package(tmp_path, source)
    obs = batch.observation(0)
    old = model.encode(obs.images, obs.state)
    oldgoal = model.encode_goal(obs.images)
    model.load(path)
    with pytest.raises(ContractError, match="another aligned"):
        model.predict(old, np.zeros((2, 1, 1, 14), np.float32))
    fresh = model.predict(model.encode(obs.images, obs.state), np.zeros((2, 1, 1, 14), np.float32))
    with pytest.raises(ContractError, match="another aligned"):
        model.distance(fresh, oldgoal)
    destination = tmp_path / "copy/model.pt"
    model.save(destination)
    other = AlignedSensorWorldModel(batch.state_schema, config=model.config).load(destination)
    actual = other.predict(other.encode(obs.images, obs.state), np.zeros((2, 1, 1, 14), np.float32))
    np.testing.assert_array_equal(
        other.distance(actual, other.encode_goal(obs.images)),
        model.distance(fresh, model.encode_goal(obs.images)),
    )
    with pytest.raises(FileExistsError):
        model.save(destination)
    with pytest.raises(ContractError, match="frozen"):
        model.train_step(batch)


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "escape",
        "absolute",
        "source",
        "head",
        "scales",
        "coverage",
        "train_leak",
        "fit_incomplete",
    ],
)
def test_strict_bundle_provenance_and_failed_load_preserves_owner(tmp_path, source, fault):
    model, path, _, batch = package(tmp_path, source)
    obs = batch.observation(0)
    latent = model.encode(obs.images, obs.state)
    owner = model._owner
    envelope = torch.load(path, weights_only=True)
    if fault in ("escape", "absolute"):
        envelope["members"]["sensor"]["path"] = (
            "../inputs/sensor.pt" if fault == "escape" else str(source[0]["sensor_checkpoint"])
        )
    elif fault == "source":
        envelope["implementation_sha256"] = "wrong"
    else:
        role = {
            "hash": "sensor",
            "head": "goal_head",
            "scales": "calibration",
            "coverage": "calibration_ledger",
            "train_leak": "goal_train_samples",
            "fit_incomplete": "goal_fit_report",
        }[fault]
        member = path.parent / envelope["members"][role]["path"]
        if fault == "hash":
            member.write_bytes(b"corrupt")
        elif fault == "head":
            value = torch.load(member, weights_only=True)
            value["updates"] = 2
            torch.save(value, member)
        else:
            value = json.loads(member.read_text())
            if fault == "scales":
                value["scales"]["pose"] = 7.0
            elif fault == "coverage":
                value["pairs"].pop()
            elif fault == "train_leak":
                value[0]["split"] = "val"
            else:
                value["status"] = "incomplete"
            write(member, value)
        # Update outer digest to exercise semantic checks, except raw corruption.
        if fault != "hash":
            envelope["members"][role]["sha256"] = sha256(member)
    torch.save(envelope, path)
    with pytest.raises((ContractError, ValueError, KeyError)):
        model.load(path)
    assert model._owner is owner
    assert model.predict(latent, np.zeros((2, 1, 1, 14), np.float32)).owner is owner


def test_hybrid_cost_rejects_invalid_shapes_scales_and_weights(source):
    with pytest.raises(ContractError):
        hybrid_cost(np.zeros(2), np.zeros(3), dict(visual=1, pose=1))
    with pytest.raises(ContractError):
        hybrid_cost(np.zeros(2), np.zeros(2), dict(visual=0, pose=1))
    with pytest.raises(ContractError):
        AlignedSensorWorldModel(
            source[2].state_schema, config={"weights": {"visual": 0.0, "pose": 1.0}}
        )
    with pytest.raises(ContractError):
        AlignedSensorWorldModel(source[2].state_schema, device="mps")


def test_generic_evaluator_load_and_train_image_calibration(tmp_path, source, monkeypatch):
    import importlib.util
    from types import SimpleNamespace

    from embodied_jepa import data

    model, path, _, batch = package(tmp_path, source)
    manifest = json.loads(source[0]["provenance_files"]["dataset_manifest"].read_text())
    store = SimpleNamespace(
        manifest=manifest,
        manifest_hash=sha256(source[0]["provenance_files"]["dataset_manifest"]),
        state_schema=batch.state_schema,
    )
    monkeypatch.setattr(data, "DatasetStore", lambda root: store)
    spec = importlib.util.spec_from_file_location(
        "aligned_generic_evaluator", Path(__file__).parents[1] / "scripts/evaluate_apple.py"
    )
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    loaded_store, loaded = evaluator.checkpoint_model("synthetic-only", path)
    assert loaded_store is store and loaded.backend == "aligned_sensor_wm_v1"
    frames = np.broadcast_to(
        np.arange(57, dtype=np.uint8)[:, None, None, None], (57, 8, 8, 3)
    ).copy()
    episode = SimpleNamespace(
        episode_id="train-a",
        observations={"onboard_rgb": frames},
        actions=np.zeros((56, 14), np.float32),
    )
    waypoints, calibration = evaluator.calibrate_waypoints(
        loaded, episode, stride=28, horizon=16, dwell=3, training_episode_ids=["train-a", "train-b"]
    )
    assert len(waypoints) == 3
    assert all(w.dwell_observations == 3 for w in waypoints)
    assert calibration["episode_id"] == "train-a"
    assert all(np.isfinite(w.threshold) and w.threshold > 0 for w in waypoints)


def test_nonzero_loaded_head_matches_task041_on_varied_images(tmp_path, source):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "task041_nonzero_head", Path(__file__).parents[1] / "scripts/apple_goal_alignment.py"
    )
    task041 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task041)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        reference = task041.head_factory().eval()
    files, sensor, _ = source
    artifact = torch.load(files["head_checkpoint"], weights_only=True)
    artifact["head"] = reference.state_dict()
    torch.save(artifact, files["head_checkpoint"])
    seal_path = files["provenance_files"]["goal_fit_seal"]
    seal = json.loads(seal_path.read_text())
    seal["artifacts"]["head.pt"] = sha256(files["head_checkpoint"])
    write(seal_path, seal)
    calibration = json.loads(files["calibration_json"].read_text())
    calibration["goal_head_sha256"] = sha256(files["head_checkpoint"])
    write(files["calibration_json"], calibration)
    model, _, _, _ = package(tmp_path, source)
    images = {
        "onboard_rgb": np.random.default_rng(91).integers(0, 256, (5, 19, 23, 3), dtype=np.uint8)
    }
    actual = model.encode_goal(images).pose
    with torch.no_grad():
        expected = reference(sensor.encode_goal(images).values)[:, :7]
    assert actual.shape == (5, 7)
    assert torch.count_nonzero(actual) == 35
    assert torch.all(expected.std(dim=0) > 0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "source",
    [
        {
            "names": ("extra.velocity", *reversed(FIELDS), "other.velocity"),
            "bias": tuple(range(1, 8)),
        }
    ],
    indirect=True,
)
def test_named_q_cost_ignores_extra_fields_and_respects_permutation(tmp_path, source):
    model, _, _, batch = package(tmp_path, source)
    obs = batch.observation(0)
    goal = model.encode_goal(obs.images)
    values = torch.full((2, 1, 1, 1728 + batch.state_schema.dimension), 100.0)
    values[..., :1728] = goal.visual.values[:, None, None]
    for ordinal, name in enumerate(FIELDS):
        column = 1728 + batch.state_schema.names.index(name)
        values[..., column] = goal.pose[:, ordinal, None, None] + (ordinal + 1) * 0.1
    prediction = AlignedLatent(VisualLatent(values, model._sensor._owner), model._owner)
    expected = np.full((2, 1, 1), 0.5 * np.square(np.arange(1, 8) * 0.1).mean() / 6, np.float32)
    np.testing.assert_allclose(model.distance(prediction, goal), expected, rtol=1e-5, atol=1e-7)
