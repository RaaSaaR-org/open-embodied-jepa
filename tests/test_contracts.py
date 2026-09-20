"""Failure cases that would silently corrupt learning or robot commands."""

from dataclasses import replace

import numpy as np
import pytest

from embodied_jepa.contracts import (
    ACTION_DIM,
    ACTION_NAMES,
    ACTION_SCHEMA,
    EE_DELTA_GRASP_V0,
    ActionSchema,
    Capabilities,
    ContractError,
    ExecutionResult,
    Observation,
    RobotState,
    SequenceBatch,
    StateSchema,
    validate_actions,
    validate_costs,
)


@pytest.fixture
def schema():
    return StateSchema(("arm.q", "imu.yaw"), ("rad", "rad"), "fixture_v0", (True, False))


@pytest.fixture
def batch(schema):
    mask = np.ones((2, 4, 2), dtype=np.bool_)
    mask[..., 1] = False
    return SequenceBatch(
        observations={"head": np.zeros((2, 4, 8, 10, 3), dtype=np.uint8)},
        robot_states=np.zeros((2, 4, 2), dtype=np.float32),
        state_mask=mask,
        actions=np.zeros((2, 3, ACTION_DIM), dtype=np.float32),
        timestamps=np.array([[0, 0.1, 0.2, 0.3], [1, 1.1, 1.2, 1.3]], dtype=np.float64),
        terminated=np.array([[False, False, True], [False, False, False]], dtype=np.bool_),
        episode_ids=("episode-a", "episode-b"),
        state_schema=schema,
    )


def test_canonical_roundtrip_and_mask_preservation(batch):
    reconstructed = SequenceBatch.from_mapping(batch.as_mapping())
    assert reconstructed.horizon == 3
    assert reconstructed.batch_size == 2
    snapshot = reconstructed.observation(1)
    assert isinstance(snapshot, Observation)
    assert isinstance(snapshot.state, RobotState)
    assert snapshot.images["head"].shape == (2, 8, 10, 3)
    np.testing.assert_array_equal(snapshot.state.mask, [[True, False], [True, False]])
    np.testing.assert_array_equal(snapshot.state.timestamps, [0.1, 1.1])
    assert snapshot.state.schema == batch.state_schema


def test_sensor_buffers_cannot_mutate_snapshot(batch):
    inputs = batch.as_mapping()
    states = batch.robot_states.copy()
    images = batch.observations["head"].copy()
    inputs.update(robot_states=states, observations={"head": images})
    frozen = SequenceBatch.from_mapping(inputs)
    states[:] = 9
    images[:] = 255
    assert not frozen.robot_states.any()
    assert not frozen.observations["head"].any()
    with pytest.raises(ValueError):
        frozen.actions[0, 0, 0] = 1
    with pytest.raises(TypeError):
        frozen.observations["unannounced-camera"] = images


@pytest.mark.parametrize(
    "field,dtype",
    [
        ("robot_states", np.float64),
        ("actions", np.float64),
        ("state_mask", np.float32),
        ("terminated", np.int64),
        ("timestamps", np.float32),
    ],
)
def test_rejects_implicit_dtype_conversion(batch, field, dtype):
    with pytest.raises(ContractError):
        replace(batch, **{field: getattr(batch, field).astype(dtype)})


@pytest.mark.parametrize("field", ["robot_states", "actions", "timestamps"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_values_even_in_masked_fields(batch, field, bad):
    corrupt = getattr(batch, field).copy()
    corrupt.flat[-1] = bad
    with pytest.raises(ContractError, match="finite"):
        replace(batch, **{field: corrupt})


def test_missing_required_state_fails_but_optional_is_preserved(batch):
    mask = batch.state_mask.copy()
    mask[0, 1, 0] = False
    with pytest.raises(ContractError, match="required"):
        replace(batch, state_mask=mask)


@pytest.mark.parametrize(
    "times",
    [
        [0, 0.1, 0.1, 0.3],
        [0, 0.2, 0.1, 0.3],
        [-0.1, 0, 0.1, 0.2],
    ],
)
def test_time_must_be_monotonic_and_nonnegative(batch, times):
    timestamps = batch.timestamps.copy()
    timestamps[0] = times
    with pytest.raises(ContractError):
        replace(batch, timestamps=timestamps)


def test_terminal_window_cannot_cross_episode(batch):
    terminals = batch.terminated.copy()
    terminals[0, 1] = True
    with pytest.raises(ContractError, match="terminal boundary"):
        replace(batch, terminated=terminals)


@pytest.mark.parametrize(
    "change",
    [
        {"robot_states": np.zeros((2, 3, 2), dtype=np.float32)},
        {"state_mask": np.ones((2, 4, 3), dtype=np.bool_)},
        {"actions": np.zeros((2, 3, 13), dtype=np.float32)},
        {"timestamps": np.zeros((2, 3), dtype=np.float64)},
        {"terminated": np.zeros((2, 4), dtype=np.bool_)},
        {"episode_ids": ("episode-a",)},
        {"episode_ids": ("episode-a", "")},
        {"observations": {}},
        {"observations": {"head": np.zeros((2, 3, 8, 10, 3), dtype=np.uint8)}},
        {"observations": {"head": np.zeros((2, 4, 3, 8, 10), dtype=np.uint8)}},
        {"observations": {"head": np.zeros((2, 4, 8, 10, 3), dtype=np.float32)}},
    ],
)
def test_rejects_shape_and_camera_corruption(batch, change):
    with pytest.raises(ContractError):
        replace(batch, **change)


def test_mapping_rejects_missing_masks_and_unknown_fields(batch):
    mapping = batch.as_mapping()
    del mapping["state_mask"]
    with pytest.raises(ContractError, match="incomplete"):
        SequenceBatch.from_mapping(mapping)
    mapping = batch.as_mapping() | {"future_ground_truth": "not a model input"}
    with pytest.raises(ContractError, match="unknown"):
        SequenceBatch.from_mapping(mapping)


@pytest.mark.parametrize("step", [-1, 4, True, 0.5])
def test_observation_index_rejects_wraparound_or_ambiguous_steps(batch, step):
    with pytest.raises(ContractError):
        batch.observation(step)


def test_freshness_uses_the_supplied_sensor_clock(batch):
    snapshot = batch.observation(0)
    snapshot.require_fresh(now=1.1, max_age=1.2)
    with pytest.raises(ContractError, match="stale"):
        snapshot.require_fresh(now=1.1, max_age=0.5)
    with pytest.raises(ContractError, match="future"):
        snapshot.require_fresh(now=0.5, max_age=10)
    with pytest.raises(ContractError):
        snapshot.require_fresh(now=1.1, max_age=np.inf)


def test_joint_actions_have_distinct_ordered_schema(schema):
    joint = ActionSchema("joint_delta_fixture_v0", ("left.elbow", "right.elbow"))
    actions = np.zeros((1, 4, 3, 2), dtype=np.float32)
    validate_actions(actions, schema=joint)
    with pytest.raises(ContractError):
        validate_actions(actions)
    with pytest.raises(ContractError, match="order is fixed"):
        ActionSchema(ACTION_SCHEMA, ACTION_NAMES[::-1])
    capabilities = Capabilities(EE_DELTA_GRASP_V0, schema, max_horizon=8)
    with pytest.raises(ContractError, match="incompatible"):
        capabilities.require(action_schema=joint, state_schema=schema, horizon=3)


def test_action_bounds_are_not_silently_clipped():
    actions = np.zeros((1, 3, 4, ACTION_DIM), dtype=np.float32)
    actions[..., 0] = -1
    actions[..., 1] = 1
    validate_actions(actions)
    actions[0, 0, 0, 2] = 1.01
    with pytest.raises(ContractError, match=r"\[-1, 1\]"):
        validate_actions(actions)


def test_costs_have_batch_candidate_time_shape_and_finite_values():
    costs = np.zeros((2, 5, 3), dtype=np.float32)
    validate_costs(costs, (2, 5, 3))
    with pytest.raises(ContractError):
        validate_costs(costs, (5, 2, 3))
    costs[0, 0, 0] = np.nan
    with pytest.raises(ContractError):
        validate_costs(costs, (2, 5, 3))


def test_capability_negotiation_checks_order_units_history_horizon_device(schema):
    capabilities = Capabilities(EE_DELTA_GRASP_V0, schema, max_horizon=8, min_history=2)
    requested = dict(action_schema=EE_DELTA_GRASP_V0, state_schema=schema, horizon=3, history=2)
    capabilities.require(**requested)
    for change in (
        {"state_schema": replace(schema, names=schema.names[::-1])},
        {"state_schema": replace(schema, units=("degree", "rad"))},
        {"horizon": 9},
        {"history": 1},
        {"device": "mps"},
        {"horizon": 0},
    ):
        with pytest.raises(ContractError):
            capabilities.require(**(requested | change))


@pytest.mark.parametrize(
    "change",
    [
        {"names": ("q", "q")},
        {"units": ("rad",)},
        {"required": (True,)},
        {"version": ""},
        {"units": ("rad", "")},
    ],
)
def test_state_schema_cannot_hide_ambiguous_units_or_order(schema, change):
    with pytest.raises(ContractError):
        replace(schema, **change)


@pytest.mark.parametrize(
    "change",
    [
        {"names": ("q", 1)},
        {"names": "q"},
        {"names": None},
        {"units": ("rad", 1)},
        {"units": "radians"},
        {"version": 1},
        {"version": " "},
    ],
)
def test_state_schema_rejects_invalid_metadata_types(schema, change):
    with pytest.raises(ContractError):
        replace(schema, **change)


@pytest.mark.parametrize(
    "version,names",
    [
        (1, ("q",)),
        ("", ("q",)),
        (" ", ("q",)),
        ("joint_delta_v0", (1,)),
        ("joint_delta_v0", "q"),
        ("joint_delta_v0", None),
    ],
)
def test_action_schema_rejects_invalid_metadata_types(version, names):
    with pytest.raises(ContractError):
        ActionSchema(version, names)


def test_execution_reports_actual_action_and_rejection_distinctly():
    requested = np.ones(ACTION_DIM, dtype=np.float32)
    applied = requested * 0.5
    result = ExecutionResult(requested, applied, "clipped", 0.1, "workspace boundary")
    np.testing.assert_array_equal(result.applied_action, applied)
    assert result.reason == "workspace boundary"
    assert ExecutionResult(requested, requested, "applied", 0.1).status == "applied"
    assert ExecutionResult(requested, None, "rejected", 0.1, "IK failed").applied_action is None
    assert (
        ExecutionResult(requested, None, "stopped", 0.1, "deadline missed").applied_action is None
    )


@pytest.mark.parametrize(
    "status,applied,reason",
    [
        ("applied", None, ""),
        ("applied", 0.5, ""),
        ("clipped", 1, "workspace"),
        ("clipped", 0.5, ""),
        ("rejected", 0, "IK failed"),
        ("rejected", None, ""),
        ("stopped", 0, "deadline missed"),
        ("unknown", None, "invalid"),
    ],
)
def test_execution_cannot_mislabel_applied_or_unexecuted_action(status, applied, reason):
    requested = np.ones(ACTION_DIM, dtype=np.float32)
    actual = None if applied is None else np.full(ACTION_DIM, applied, dtype=np.float32)
    with pytest.raises(ContractError):
        ExecutionResult(requested, actual, status, 0.1, reason)
