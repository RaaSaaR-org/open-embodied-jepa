# G1/Dex3 SDK2 preparation and commissioning

TASK-021 supplies **in-memory mapping and failure-path tests only** in [`hardware.py`](../src/embodied_jepa/hardware.py). It does not import SDK2, start DDS, read a robot camera, serialize a valid wire command, take controller ownership, or execute hardware. `MockSDK2Transport(..., physical_execution=True)` always raises; it accepts only the built-in fake endpoint. Passing a completed calibration does not unlock physical execution. Actual integration/commissioning remains TASK-026.

## Audited sources

Reviewed on 2026-09-20; exact revisions are references, not an assertion of compatibility with an uninspected robot firmware.

| Source | Pinned revision and evidence | License |
| --- | --- | --- |
| Unitree SDK2 Python | [`9c519023d188bfe4643d326868474878ab515ed8`](https://github.com/unitreerobotics/unitree_sdk2_python/tree/9c519023d188bfe4643d326868474878ab515ed8): [G1 joint indices and low-level example](https://github.com/unitreerobotics/unitree_sdk2_python/blob/9c519023d188bfe4643d326868474878ab515ed8/example/g1/low_level/g1_low_level_example.py), [LowCmd IDL](https://github.com/unitreerobotics/unitree_sdk2_python/blob/9c519023d188bfe4643d326868474878ab515ed8/unitree_sdk2py/idl/unitree_hg/msg/dds_/_LowCmd_.py), [HandCmd IDL](https://github.com/unitreerobotics/unitree_sdk2_python/blob/9c519023d188bfe4643d326868474878ab515ed8/unitree_sdk2py/idl/unitree_hg/msg/dds_/_HandCmd_.py), [CRC implementation](https://github.com/unitreerobotics/unitree_sdk2_python/blob/9c519023d188bfe4643d326868474878ab515ed8/unitree_sdk2py/utils/crc.py) | [BSD-3-Clause](https://github.com/unitreerobotics/unitree_sdk2_python/blob/9c519023d188bfe4643d326868474878ab515ed8/LICENSE) |
| Official XR teleoperation | [`817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b` hand controller](https://github.com/unitreerobotics/xr_teleoperate/blob/817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b/teleop/robot_control/robot_hand_unitree.py): side-specific Dex3 enums, topics, mode bits | [Apache-2.0](https://github.com/unitreerobotics/xr_teleoperate/blob/817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b/LICENSE) |
| Unitree SDK2 C++ | [`c753829882fba461ed07ba25aaabee0a25d83663` Dex3 example](https://github.com/unitreerobotics/unitree_sdk2/blob/c753829882fba461ed07ba25aaabee0a25d83663/example/g1/dex3/g1_dex3_example.cpp): hand mode/timeout fields and illustrative stop messages | [BSD-3-Clause](https://github.com/unitreerobotics/unitree_sdk2/blob/c753829882fba461ed07ba25aaabee0a25d83663/LICENSE) |

These are independent schema mappings, not copied SDK implementations. Dependency licenses do not establish the permissions of a particular robot's recordings, firmware, or camera output. Keep source/IDL/firmware versions and notices with future deployment artifacts.

## Named channels and ordering

The mock models the **full-body `rt/lowcmd` profile**, receiving body state from `rt/lowstate`. It uses serial pitch/roll joint mode `mode_pr=0`; parallel A/B mode is unsupported. Body `mode_machine` must match observed/calibrated telemetry. The IDL has 35 motor slots: map 29 named joints and keep slots 29–34 disabled in this profile. [Pinned body example and IDL](https://github.com/unitreerobotics/unitree_sdk2_python/tree/9c519023d188bfe4643d326868474878ab515ed8)

| Body slots | Named order |
| --- | --- |
| 0–5 | Left hip pitch/roll/yaw, knee, ankle pitch/roll |
| 6–11 | Right hip pitch/roll/yaw, knee, ankle pitch/roll |
| 12–14 | Waist yaw/roll/pitch |
| 15–21 | Left shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw |
| 22–28 | Right shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw |

Names match `BODY_JOINTS`; no caller may infer order from a 43-element vector alone. Inputs may choose any complete joint ordering when constructing the transport, but every command must then identify that declared order. Sign and zero offsets are calibrated by name, with `sdk_q = sign * canonical_q + zero_offset_rad`; velocities use the sign without the offset.

Dex3 command/state topics are `rt/dex3/{left,right}/{cmd,state}`. Motor positions are **side-specific**: left uses thumb0/1/2, middle0/1, index0/1; right uses thumb0/1/2, index0/1, middle0/1. The semantic synergy list in the simulation manifest uses index before middle and must therefore be mapped by name. Hand mode encodes motor ID in bits 0–3, status in 4–6, timeout in bit 7; the mock forms position-command mode with status=1 and timeout=0. [Pinned official hand controller](https://github.com/unitreerobotics/xr_teleoperate/blob/817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b/teleop/robot_control/robot_hand_unitree.py)

Do not reuse this envelope for `rt/arm_sdk`. The [separate 7-DOF arm example](https://github.com/unitreerobotics/unitree_sdk2_python/blob/9c519023d188bfe4643d326868474878ab515ed8/example/g1/high_level/g1_arm7_sdk_dds_example.py) uses slot 29's position field as a control-weight signal. That is not a physical joint and has different ownership/enable semantics. A future arm-only deployment needs its own reviewed profile, versioned manifest, and tests; changing only a topic string is insufficient.

## What the local mock validates

`HardwareCalibration` requires a robot identity, source (`mock_fixture` or `measured_hardware`), calibration ID, clock domain, camera/action manifest hashes, expected robot mode, command period, state-age and sensor-skew budgets, and every named joint's sign, offset, canonical position limits, velocity limit, and gains. All fields must be explicit and finite. `simulation_only_not_for_hardware` data is rejected. The test suite uses clearly labeled invented fixture values; none is a suggested hardware setting.

Generate an intentionally unfilled JSON template for commissioning records:

```python
import json
from embodied_jepa.hardware import calibration_template

print(json.dumps(calibration_template(), indent=2))
```

The resulting `g1_sdk2_calibration_v0` template contains nulls and fails `HardwareCalibration.from_mapping` until completed. Camera/action hashes point to separately reviewed manifests containing intrinsic/extrinsic calibration, units, fixed base frame, EE transforms, 14D scales and hand synergies. Store serials/recordings locally when appropriate; publish redacted manifest references with checksums.

`MockSensorPacket` requires all 35 body slots plus 7 positions/velocities per hand, RGB, separate body/left/right/camera acquisition timestamps, a common clock identity, sequence number, robot mode, and a fault flag. It is a fake input seam, not a decoder for actual SDK messages. `read()` converts to named canonical 43D qpos/qvel, rejects missing/disconnected/faulted, stale/future, skewed, backward-clock, out-of-range, or over-speed data, and latches the mock disabled. There is no fabricated camera frame or inferred timestamp. A future adapter must map device clocks to a measured common timebase and decode real motor fault/state fields before creating canonical observations.

`enable_mock()` requires valid telemetry. `send_joint_targets()` checks dtype, shape, declared names, deadline, position and target-rate limits, unique sensor consumption, and command cadence before recording an in-memory envelope. The state limits are canonical; signs/offsets are applied only at the wire-order boundary. After stop, fresh telemetry and explicit re-enable are required, starting from measured positions. A successful response is marked `mock_only=True, physical_execution=False`; it proves message preparation only. `stop(reason)` records a local stop intent and disables recording, without prescribing a physical damping/hold message. `poll_watchdog()` exercises stale-data handling when called; it is not a background real-time watchdog.

The mock envelope retains `crc=None`, `wire_serializable=False`, and a CRC-required marker. Real body messages must use the pinned SDK's exact struct packing/CRC after all fields are populated; test known-good serialized vectors against firmware/SDK. The audited `HandCmd` IDL has no body-style CRC field; do not fabricate one. Transport publication/acknowledgement is not evidence that a motor reached its target. Body and two hand messages are separate deliveries: future code must handle partial publication, sequence/clock correlation, controller ownership, telemetry loss, and watchdog expiry. None of these wire/firmware behaviors is validated by the local mock. [Pinned CRC code](https://github.com/unitreerobotics/unitree_sdk2_python/blob/9c519023d188bfe4643d326868474878ab515ed8/unitree_sdk2py/utils/crc.py), [HandCmd IDL](https://github.com/unitreerobotics/unitree_sdk2_python/blob/9c519023d188bfe4643d326868474878ab515ed8/unitree_sdk2py/idl/unitree_hg/msg/dds_/_HandCmd_.py)

Run `.venv/bin/pytest -q tests/test_hardware.py` on the Mac without SDK2 installed. Tests verify ordering, inverse conversion, incomplete calibration, message slots/modes, malformed commands, limits, disconnects, sensor alignment, stale/consumed packets, enable, stop and close. They do not establish functional safety or physical readiness.

## Physical commissioning sequence — TASK-026, not executed

1. **Inventory and recovery plan.** Record exact EDU4 serial/DOF/locked-waist configuration, both hand revisions and firmware, camera interfaces, controller/SDK/IDL versions, support fixture, power and communications topology. A qualified operator must establish the manufacturer-approved emergency-stop/recovery procedure and verify access before any publisher is enabled. Secure or suspend the robot using approved support so loss of motor torque cannot cause a fall; keep people clear of arm/hand pinch zones. Do not run example scripts that automatically release another controller.
2. **Read-only telemetry.** On the intended supported deployment host, subscribe without any command publisher. Identify every body and hand joint against the physical robot; confirm offsets/signs and locked/unavailable joints. Record faults, packet rates, losses, clock offsets/skew, RGB timestamps and camera extrinsics. Complete acquisition and deadline budgets from measurements. Missing sensors, duplicate names, unknown mode or firmware, mismatched timebase, or invalid limits keep the deployment disabled.
3. **Calibrate offline and replay.** Populate and review the physical calibration, action/scaling and camera manifests. Validate forward/inverse mappings and neutral/hold semantics against recorded telemetry and the mock before enabling a physical publisher. Fit no production limit or gain by copying `configs/g1_sim_action.json` or test fixtures. Confirm whether arm-only SDK control or full-body low-level ownership is actually appropriate; give each a separate reviewed profile.
4. **Implement and test the publisher off-robot.** Build real IDL serialization/CRC, bounded publish cadence, state-time correlation, explicit operator enable, independent heartbeat/watchdog and manufacturer-appropriate stop/damping behavior. Test expired commands, network disconnect, malformed state, dropped one-hand publication, robot mode change and process death using an isolated replay/test environment. Logging must distinguish prepared, published, acknowledged and measured-applied commands. Operator stop must work independently of model inference and Python progress.
5. **Bounded supported-robot checks.** With the operator controlling the physical enable, start from measured positions. Freeze a reviewed tiny displacement/velocity/time budget from the calibrated manifest; no numerical physical value is supplied here. Test one known joint, then one finger/hand, then each arm separately. Recheck sign/order, tracking error, current/temperature, faults, collision clearance, heartbeat and stop response before increasing scope. Any failed gate returns to disabled investigation; do not automatically resume after a stop.
6. **Supervised task admission.** Establish a clear tabletop workspace, object handling constraints and reset protocol. Replay validated bounded trajectories before learned planning. Run short image-goal episodes with frozen thresholds and an operator watching; record requested/published/measured commands, state/action/camera schemas, data/checkpoint/config hashes, outcomes and all stop reasons. Real robot resets are supervised procedures, not simulation `reset()`. Expand cohorts only after the preceding evidence is reviewed.

Required exit artifacts are the completed calibration/ownership manifests, telemetry alignment report, mapping/CRC evidence, measured stop response, bounded joint/hand/arm traces, and supervised trial logs. The current repository has none of that physical evidence and provides no executable hardware-enable route.
