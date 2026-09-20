"""SDK mapping exercises use invented fixture calibration and in-memory endpoints only."""

from dataclasses import asdict, replace

import numpy as np
import pytest

from embodied_jepa.contracts import ContractError
from embodied_jepa.hardware import (
    BODY_JOINTS,
    LEFT_HAND_JOINTS,
    RIGHT_HAND_JOINTS,
    SDK_JOINTS,
    HardwareCalibration,
    JointCalibration,
    MockSDK2Endpoint,
    MockSDK2Transport,
    MockSensorPacket,
    calibration_template,
)


@pytest.fixture
def calibration():
    # These numbers are intentionally test fixtures, never suggested physical settings.
    joints = {
        name: JointCalibration(-1 if i % 2 else 1, 0.1, -1, 1, 2, 3, 0.2)
        for i, name in enumerate(SDK_JOINTS)
    }
    return HardwareCalibration(
        robot_serial="software-fixture",
        robot_variant="fixture-29dof-dual-dex3",
        calibration_id="not-a-physical-calibration",
        source="mock_fixture",
        clock_domain="fake-monotonic",
        camera_calibration_sha256="a" * 64,
        action_manifest_sha256="b" * 64,
        expected_mode_machine=7,
        command_period_s=0.05,
        max_state_age_s=0.1,
        max_sensor_skew_s=0.01,
        joints=joints,
    )


def packet(calibration, *, sequence=0, timestamp=1.0, canonical_q=None, canonical_dq=None):
    q = np.zeros(43) if canonical_q is None else np.asarray(canonical_q)
    dq = np.zeros(43) if canonical_dq is None else np.asarray(canonical_dq)
    sdk_q = np.array(
        [
            calibration.joints[n].sign * v + calibration.joints[n].zero_offset_rad
            for n, v in zip(SDK_JOINTS, q, strict=True)
        ],
        dtype=np.float32,
    )
    sdk_dq = np.array(
        [calibration.joints[n].sign * v for n, v in zip(SDK_JOINTS, dq, strict=True)],
        dtype=np.float32,
    )
    return MockSensorPacket(
        sequence,
        np.pad(sdk_q[:29], (0, 6)),
        np.pad(sdk_dq[:29], (0, 6)),
        sdk_q[29:36],
        sdk_dq[29:36],
        sdk_q[36:],
        sdk_dq[36:],
        np.zeros((8, 8, 3), dtype=np.uint8),
        dict.fromkeys(("body", "left", "right", "camera"), timestamp),
        calibration.expected_mode_machine,
        calibration.clock_domain,
    )


@pytest.fixture
def setup(calibration):
    clock = [1.0]
    endpoint = MockSDK2Endpoint()
    endpoint.push(packet(calibration))
    transport = MockSDK2Transport(calibration, endpoint, clock=lambda: clock[0])
    return transport, endpoint, clock


def test_unfilled_calibration_template_never_enables_transport():
    with pytest.raises(ContractError, match="calibrated"):
        HardwareCalibration.from_mapping(calibration_template())
    with pytest.raises(ContractError, match="version"):
        HardwareCalibration.from_mapping({})


@pytest.mark.parametrize(
    "change",
    [
        {"robot_serial": None},
        {"source": "simulation_only_not_for_hardware"},
        {"camera_calibration_sha256": "unknown"},
        {"max_state_age_s": None},
        {"command_period_s": 0},
        {"expected_mode_machine": True},
    ],
)
def test_incomplete_or_guessed_manifest_rejected(calibration, change):
    with pytest.raises(ContractError):
        replace(calibration, **change)


def test_joint_manifest_rejects_missing_limits_names_gains_and_invalid_sign(calibration):
    incomplete = dict(calibration.joints)
    incomplete.pop(SDK_JOINTS[-1])
    with pytest.raises(ContractError, match="all 29"):
        replace(calibration, joints=incomplete)
    for change in (
        {"sign": 0},
        {"sign": True},
        {"kp": None},
        {"kd": -1},
        {"zero_offset_rad": np.nan},
        {"upper_rad": -2},
    ):
        with pytest.raises(ContractError):
            replace(calibration.joints[SDK_JOINTS[0]], **change)
    mapping = {name: value for name, value in calibration.__dict__.items() if name != "joints"}
    mapping.update(
        version="g1_sdk2_calibration_v0",
        joints={name: asdict(cal) for name, cal in calibration.joints.items()},
    )
    assert HardwareCalibration.from_mapping(mapping) == calibration


def test_no_physical_execution_or_custom_publisher_path(calibration):
    with pytest.raises(NotImplementedError, match="physical"):
        MockSDK2Transport(calibration, MockSDK2Endpoint(), physical_execution=True)
    with pytest.raises(ContractError, match="in-memory"):
        MockSDK2Transport(calibration, object())


def test_initially_disabled_and_explicit_enable_required(setup):
    transport, endpoint, _ = setup
    assert not transport.enabled
    with pytest.raises(ContractError, match="disabled"):
        transport.send_joint_targets(
            np.zeros(43, dtype=np.float32), joint_names=SDK_JOINTS, deadline=1.1
        )
    assert not endpoint.commands


def test_named_mapping_resolves_asymmetric_hands_and_disabled_body_slots(calibration):
    endpoint = MockSDK2Endpoint()
    endpoint.push(packet(calibration))
    # Deliberately choose a different order from SDK/MJCF order, including index-before-middle.
    names = tuple(sorted(SDK_JOINTS))
    transport = MockSDK2Transport(calibration, endpoint, joint_names=names, clock=lambda: 1.0)
    transport.enable_mock()
    targets = np.linspace(-0.04, 0.04, 43, dtype=np.float32)
    result = transport.send_joint_targets(targets, joint_names=names, deadline=1.01)
    assert result["mock_only"] and not result["physical_execution"]
    envelope = endpoint.commands[-1]
    assert envelope["wire_serializable"] is False
    assert len(envelope["body"]["motor_cmd"]) == 35
    assert envelope["body"]["crc"] is None
    assert envelope["body"]["mode_pr"] == 0 and envelope["body"]["mode_machine"] == 7
    assert [x["name"] for x in envelope["body"]["motor_cmd"][:29]] == list(BODY_JOINTS)
    assert all(x["mode"] == 0 and x["kp"] == 0 for x in envelope["body"]["motor_cmd"][29:])
    assert envelope["left"]["motor_cmd"][3]["name"] == "left_hand_middle_0_joint"
    assert envelope["right"]["motor_cmd"][3]["name"] == "right_hand_index_0_joint"
    for channel, ordered in (
        ("body", BODY_JOINTS),
        ("left", LEFT_HAND_JOINTS),
        ("right", RIGHT_HAND_JOINTS),
    ):
        for index, name in enumerate(ordered):
            cmd = envelope[channel]["motor_cmd"][index]
            cal = calibration.joints[name]
            assert cmd["q"] == pytest.approx(
                cal.sign * targets[names.index(name)] + cal.zero_offset_rad
            )
            assert cmd["mode"] == (1 if channel == "body" else index | 0x10)
            assert cmd["dq"] == 0 and cmd["tau"] == 0


def test_sensor_mapping_inverts_offsets_signs_and_preserves_clock(calibration):
    q = np.linspace(-0.5, 0.5, 43)
    dq = np.linspace(-1, 1, 43)
    endpoint = MockSDK2Endpoint()
    endpoint.push(packet(calibration, canonical_q=q, canonical_dq=dq))
    transport = MockSDK2Transport(
        calibration, endpoint, joint_names=SDK_JOINTS[::-1], clock=lambda: 1.0
    )
    raw = transport.read()
    np.testing.assert_allclose(raw["qpos"], q[::-1], atol=1e-7)
    np.testing.assert_allclose(raw["qvel"], dq[::-1], atol=1e-7)
    assert raw["timestamp"] == 1 and len(raw["sensor_timestamps"]) == 4
    assert raw["joint_names"] == SDK_JOINTS[::-1]
    assert raw["clock_domain"] == calibration.clock_domain


@pytest.mark.parametrize(
    "failure", ["stale", "future", "skew", "disconnect", "mode", "fault", "clock"]
)
def test_sensor_faults_latch_mock_disabled(setup, calibration, failure):
    transport, endpoint, clock = setup
    transport.enable_mock()
    original = endpoint.packet
    if failure == "stale":
        clock[0] += 1
    elif failure == "future":
        endpoint.push(packet(calibration, timestamp=2))
    elif failure == "skew":
        endpoint.push(replace(original, timestamps=dict(original.timestamps) | {"camera": 0.95}))
    elif failure == "disconnect":
        endpoint.connected = False
    elif failure == "mode":
        endpoint.push(replace(original, mode_machine=8))
    elif failure == "fault":
        endpoint.push(replace(original, fault=True))
    else:
        endpoint.push(replace(original, clock_domain="different-domain"))
    with pytest.raises(ContractError):
        transport.poll_watchdog()
    assert not transport.enabled and not endpoint.commands
    assert endpoint.stops


def test_stop_requires_fresh_packet_and_explicit_reenable(setup, calibration):
    transport, endpoint, clock = setup
    transport.enable_mock()
    transport.stop("operator test stop")
    with pytest.raises(ContractError, match="new sensor packet"):
        transport.enable_mock()
    clock[0] = 1.05
    endpoint.push(packet(calibration, sequence=1, timestamp=1.05))
    transport.enable_mock()
    assert transport.enabled
    transport.close()
    assert not transport.enabled
    with pytest.raises(RuntimeError, match="closed"):
        transport.read()


@pytest.mark.parametrize(
    "bad", [np.zeros(43), np.zeros(42, dtype=np.float32), np.full(43, np.nan, dtype=np.float32)]
)
def test_malformed_targets_never_reach_endpoint(setup, bad):
    transport, endpoint, _ = setup
    transport.enable_mock()
    with pytest.raises(ContractError):
        transport.send_joint_targets(bad, joint_names=SDK_JOINTS, deadline=1.1)
    assert not endpoint.commands


@pytest.mark.parametrize("kind", ["range", "rate", "deadline", "order"])
def test_command_limits_deadline_and_order(setup, kind):
    transport, endpoint, _ = setup
    transport.enable_mock()
    targets = np.zeros(43, dtype=np.float32)
    targets[0] = {"range": 10, "rate": 0.2}.get(kind, 0)
    names = SDK_JOINTS[::-1] if kind == "order" else SDK_JOINTS
    with pytest.raises(ContractError):
        transport.send_joint_targets(
            targets, joint_names=names, deadline=0.9 if kind == "deadline" else 1.1
        )
    assert not endpoint.commands


def test_consumed_out_of_order_and_too_fast_packets(setup, calibration):
    transport, endpoint, clock = setup
    transport.enable_mock()
    targets = np.zeros(43, dtype=np.float32)
    transport.send_joint_targets(targets, joint_names=SDK_JOINTS, deadline=1.1)
    with pytest.raises(ContractError, match="consumed"):
        transport.send_joint_targets(targets, joint_names=SDK_JOINTS, deadline=1.1)
    clock[0] = 1.05
    endpoint.push(packet(calibration, sequence=1, timestamp=1.05))
    transport.enable_mock()
    transport.send_joint_targets(targets, joint_names=SDK_JOINTS, deadline=1.1)
    endpoint.push(packet(calibration, sequence=2, timestamp=1.051))
    clock[0] = 1.051
    with pytest.raises(ContractError, match="cadence"):
        transport.send_joint_targets(targets, joint_names=SDK_JOINTS, deadline=1.1)
    endpoint.push(packet(calibration, sequence=0, timestamp=1.0))
    with pytest.raises(ContractError, match="backwards"):
        transport.read()


def test_measured_position_velocity_and_malformed_packet_fail(setup, calibration):
    transport, endpoint, _ = setup
    endpoint.push(packet(calibration, canonical_q=np.full(43, 2)))
    with pytest.raises(ContractError, match="range"):
        transport.read()
    endpoint.push(packet(calibration, canonical_dq=np.full(43, 3)))
    with pytest.raises(ContractError, match="velocity"):
        transport.read()
    original = packet(calibration)
    with pytest.raises(ContractError, match="35"):
        replace(original, body_q=np.zeros(29, dtype=np.float32))
    with pytest.raises(ContractError, match="timestamp"):
        replace(original, timestamps={"body": 1})
