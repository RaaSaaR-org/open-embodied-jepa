# Native G1/Dex3 tabletop runtime

`MuJoCoSimulation` and `G1Embodiment` implement a fixed-pelvis, torque-controlled simulation using the pinned official 29-DOF G1 plus both articulated Dex3 hands. Constructor `object_kind` selects apple (red 27 mm sphere), cube (green 46 mm box), or banana (yellow capsule with 18 mm radius and 35 mm half-length). All are 80 g procedural proxies. `container_kind` selects plate (blue cylinder/low rim), bowl (purple cylinder/high rim), or target (teal square pad). They are explicit geometric proxies, not scanned assets or validated contact models. This is an executable reaching/control foundation. Successful contact-based Apple → Plate manipulation and learned image-goal planning require separate evidence.

## Run from an editable source checkout

Install the `sim` extra and fetch the verified assets as described in [SETUP.md](SETUP.md) and [MUJOCO_SPIKE.md](MUJOCO_SPIKE.md). The default asset and action manifests live in this repository; meshes remain external under ignored `third_party/`.

```python
import numpy as np
from embodied_jepa.simulation import MuJoCoSimulation
from embodied_jepa.embodiment import G1Embodiment

sim = MuJoCoSimulation(width=96, height=96)
robot = G1Embodiment(sim)
try:
    robot.reset(seed=0)
    action = np.zeros(14, dtype=np.float32)
    action[12:] = -1  # Open both hands; zero means halfway closed.
    action[8] = -0.5  # Right EE downward in the fixed pelvis frame.
    for _ in range(6):
        observation = robot.observe()
        result = robot.execute(action)
        if result.status not in ("applied", "clipped"):
            break
finally:
    robot.close()
```

`observe()` returns a synchronized batch of one `onboard_rgb` image and 86 float32 robot fields: 43 named joint positions followed by their 43 named velocities, in actuator order resolved by joint name. Units are radians and radians/second. `state()` returns the cached observation state, never another asynchronous poll. No object pose, goal joint configuration, contact, or success value enters this observation. Rendering can be disabled for physics-only work with `render=False`; `read()`/`observe()` then fail explicitly instead of inventing RGB. Importing either module does not import MuJoCo; constructing the simulator loads the optional runtime.

## Action and controller semantics

[configs/g1_sim_action.json](../configs/g1_sim_action.json) versions the simulation choices. They are **not physical hardware calibration**. `ee_delta_grasp_v0` has left XYZ/RPY, right XYZ/RPY, then left/right grasp. Normalized translations scale by 15 mm per command, rotations by 0.06 rad per command. Grasp -1/+1 maps to the named open/closed Dex3 joint vectors. The left and right fingers are mirrored explicitly and validated against their own MJCF limits; actuator index order is never assumed to match between hands.

Translations and rotation axes use the pelvis frame: X forward, Y left, Z up. Rotations compose as `Rz(dyaw) @ Ry(dpitch) @ Rx(droll) @ R_current`. `denormalize_action()` returns meters, radians, and two grasp fractions in [0,1]; `normalize_action()` inverts that mapping and rejects invalid values. The current fixed pelvis is at world `(0,0,0.793)` m with identity orientation.

Damped six-dimensional IK runs in a separate `MjData`, with named arm Jacobian columns, 80 iterations, 1.5 mm translation tolerance, and 0.02 rad orientation tolerance. It never edits the executing state's qpos. Targets remain inside MJCF limits and side-specific workspaces. IK convergence uses the SO(3) logarithm/geodesic angle, so opposite orientations cannot appear converged. A command whose IK solution changes a joint target by more than 0.1 rad in one 50 ms control step is stopped; grasp commands intersect every finger's rate-feasible synergy interval against the transport's current targets, including after stop/resume, and report the clipped grasp value. An empty intersection stops execution. An observed robot joint speed above 5 rad/s stops a new request. This is a sampled stop threshold, not a mathematical bound on velocities during contact.

The transport interpolates joint targets over 25 physics steps of 2 ms, applies PD torques plus the simulated robot's gravity/bias compensation, and clips torque to each upstream motor's control range. It advances real MuJoCo dynamics; it does not teleport qpos during execution. Reset is the only runtime operation that assigns the live robot/object qpos. Static plate placement also changes only during reset. No attachment, weld, magnetic force, or privileged object-motion rule assists manipulation.

`ExecutionResult.applied_action` records the accepted **target command** after workspace/grasp clipping. It is not a measurement of realized EE displacement: torque limits, controller lag, IK tolerance, inertia, and contacts affect actual motion, which is represented by subsequent observations. Rejected/stopped requests return `applied_action=None` and do not become dataset transitions. Invalid array type, shape, nonfinite values, or out-of-range normalized actions raise `ContractError` before actuation.

Fresh observations are required before each action. The adapter rejects already-consumed simulation timestamps, observations older than 51 ms in simulation time, and wall-clock planning delays exceeding 30 seconds. `stop(reason)` clears the observation cache and zeros active controls without advancing time or changing qpos; this synchronous simulator then remains paused until a new accepted command. It does not claim to implement a physical emergency stop. Nonfinite physics state or MuJoCo warnings abort execution with an exception.

## Reset, cameras, and evaluator truth

`robot.reset(seed, object_xy=None, plate_xy=None, object_kind=None, container_kind=None, object_on_container=False)` resets deterministically. Optional kind arguments validate the constructor-selected geometry; changing kinds requires constructing a new simulator/embodiment, preserving valid model and joint handles. Position overrides are world XY meters constrained to the tabletop. The default apple XY varies over `[0.38,0.42] × [-0.28,-0.24]` m; plate XY is `(0.48,-0.10)` m. Initial elbow flexion is 0.08 rad; both hands use their open synergy endpoints. Ordinary resets conservatively reject object/container XY collision-envelope overlap with 5 mm clearance before changing the live state. Defaults satisfy separation for all pairs. To render a supported goal, pass `object_on_container=True`: the object must fit inside the container rim and starts 1 mm above the container surface plus its support height. This is an explicit reset-only supported placement, never runtime object teleportation. The table top is 0.74 m. The apple begins slightly above it and settles through dynamics. Use separate instances for cube→target, banana→bowl, apple→bowl and cube→plate collection. Reserve apple→plate for the final held-out pair; do not collect its training/validation trajectories. Geometry identity is present only in evaluator metadata and visible RGB, never canonical state fields. Earlier generic asset/control smoke checks are not training data or final evaluation.

The torso-attached camera uses an explicit simulated offset `(0.08,0,0.35)` m, 75-degree vertical field of view, and a downward approximately 60-degree optical direction. This shows both hands, the apple, and plate at the default reset; it is an onboard-view approximation, not a measured G1 camera calibration. The optional `overview` camera is for inspection. For goal images, use a separate simulator with the same scene/camera configuration and a desired reset/trajectory; do not alter the active episode or supply its goal state to the model.

`sim.task_truth()` is a separate evaluator/collection-only API. It returns world object/plate positions, object velocity, hand contact, lift/drop/placement flags, and simulation time. `position_frame='world'`, `base_position_world`, and `base_rotation_world` explicitly support conversion: `p_base = R_base_world.T @ (p_world - base_position_world)`. Scripted data collection may use this truth but must label its controller provenance. Learned planning receives only canonical observations and goal RGB.

Placement is an instantaneous engineering score: object center within 4 cm horizontally of the container, height within 12 mm of container surface plus the proxy's reset support height, speed below 0.1 m/s, and no hand contact. The support-height approximation is orientation-dependent for the cube/capsule; final scoring must account for that limitation. Lift means height above 0.82 m; drop means below 0.70 m. A final benchmark must freeze any dwell requirement, termination, timeout and object-cohort rules separately. These predicates alone do not demonstrate completed manipulation.

## Validation and limitations

```sh
.venv/bin/pytest tests/test_simulation.py tests/test_embodiment.py -q
JEPA_TEST_RENDER=1 .venv/bin/pytest tests/test_simulation.py tests/test_embodiment.py -q
```

Physics/control checks cover deterministic reset and torque rollouts, both-arm translation and actual yaw rotation, named mirrored hand commands, applied-action clipping, solver/rate/velocity failures, stale and expired commands, scratch-IK isolation, state caching, and transport limits. The RGB integration check is explicitly opt-in on a graphics-capable host and allows one 8-bit intensity level of OpenGL rounding variation after an identical reset (observed three differing channel values out of 12,288); physics traces remain bit-exact. Headless control unit tests substitute a clearly labeled fixture image while preserving actual MuJoCo qpos/qvel/time; they do not claim rendering coverage. Tests skip when the optional MuJoCo dependency or pinned assets are absent.

On the local Mac, six downward commands moved both end effectors by more than 12 mm, and five yaw commands changed both measured orientations. A 15-command hand closure remained finite; peak sampled joint speed was approximately 2.28 rad/s (despite the 2 rad/s target ramp), illustrating the command-versus-motion distinction. Real onboard RGB was rendered and visually inspected with both hands, red apple proxy and blue plate visible. This evidence validates the control foundation, not a reliable grasp, controller tuning optimum, dataset coverage, learned policy quality, or physical EDU4 parity. Collision avoidance is not an IK constraint; only the declared workspace/joint/rate checks and MuJoCo contact physics are implemented.

## Bounded contact-grasp experiment

`python scripts/spikes/grasp_probe.py --output outputs/grasp_probe_higher` runs an explicitly privileged scripted cube→target collector probe, capped at 45 wall seconds. It rotates the right hand for a top-down approach, descends, closes the actual Dex3 joints, and lifts through torque control and MuJoCo contacts. The image-goal model path is not involved. Its object position is a deliberately reachable training-scene location, not the default reset cohort; it is not a generalization result.

Two exploratory runs were retained. The first lower lift target ended with object center at 0.81545 m and contact present, missing the fixed 0.82 m lift threshold at the final step despite a transient crossing (`outputs/grasp_probe/report.json`). The second increased only the scripted lift target/budget before re-running, and ended at 0.86050 m with contact and the lift predicate true (`outputs/grasp_probe_higher/report.json`), versus settled start height 0.76289 m. The lift predicate remained true over the final 118 commands (5.9 simulated seconds). It used 405 control commands, 20.25 simulated seconds, and 4.96 wall seconds on this Mac. All accepted commands went through the same embodiment API; no weld, attachment or live-state teleport was introduced. Phase images and per-step truth/command outcomes remain under ignored outputs. The report records the script and action-manifest hashes, phase target/budgets, code checkout revision, seed and both positive/negative outcomes; runtime source files were uncommitted at execution, so the checkout revision alone does not identify the complete runtime implementation.

This establishes a feasible contact-based scripted lift on one training-pair reset. The following suite provides separate exploratory transfer/release evidence. Success rates across valid resets, other pairs and learned planning still need separate evaluation. Keep apple→plate out of development collection and model-selection data.

## Scripted transfer/release baseline

`OracleManipulationPolicy(initial_truth)` in `embodied_jepa.scripted` is a privileged collection/control baseline. Construct it from `sim.task_truth()` after reset, then call `policy.action(robot)`, `robot.execute(action)`, and `policy.advance(result)`. `phase`, `step_count`, `max_steps`, `failure`, and `done` describe progress. A rejected/stopped command ends the trial. The policy never edits simulator state, attaches objects, calls a world model, or passes object truth to a planner. It freezes the initial object/container position and follows eight phases: orient, descend, close, lift, transfer, lower, release, retreat. The full budget is 805 accepted commands (40.25 simulated seconds). It assumes the fixed upright pelvis, right-hand top-down approach, and the current action manifest; it is not a general robot policy.

A five-trial exploratory suite was declared in `outputs/manipulation_oracle_suite/suite_plan.json` before execution and run once with:

```sh
.venv/bin/python scripts/spikes/grasp_probe.py --suite --output outputs/manipulation_oracle_suite
```

Success required the existing `task_truth()['placed']` predicate throughout the final ten accepted steps (0.5 simulated seconds). All five outcomes were retained; there were no retries or apple→plate trials. The aggregate report records plan, runtime-source and action-manifest hashes, and each trial stores requested/applied commands, phase images and truth traces.

| Seed / pair | Outcome | Final object XYZ (m) | Evidence |
| --- | --- | --- | --- |
| 0 / cube→target | Stopped on measured joint-speed threshold during lowering after grasp/transfer | (0.409, -0.160, 0.773) | 581 accepted commands |
| 1 / cube→target | Completed contact grasp, lift, transfer, release and retreat | (0.450, -0.101, 0.771) | 805 commands; placed during final 134 steps (6.7 s) |
| 2 / apple→bowl | Invalid overlapping reset, followed by object ejection and velocity stop during close | (-2.658, -0.735, 0.027) | 221 accepted commands; no lift |
| 3 / banana→bowl | Overlapping reset geometry; velocity stop during close | (0.315, -0.211, 0.758) | 222 accepted commands; no lift |
| 4 / cube→plate | Completed contact grasp, lift, transfer, release and retreat | (0.448, -0.134, 0.775) | 805 commands; placed during final 134 steps (6.7 s) |

The raw outcome is two completed trials out of five attempted; it is **not a reliability estimate**. The selected bowl center `(0.43,-0.16)` was too close to object center `(0.34,-0.18)` for the compiled sphere/capsule and bowl-rim collision envelopes. Those resets were invalid and cannot fairly measure grasp capability. Even the cube scenes had partial proximity to the target/plate edge and should not define the final cohort. Future collection must reject initial object/container penetration and settle objects before freezing the initial truth. Do not tune on the reserved apple→plate pair to repair these development scenes.

These successful trajectories establish that transfer/release can occur through actual dual-Dex3/MuJoCo contacts with this scaffold. They do not establish a robust manipulation controller, coverage across pairs, learned manipulation, or held-out generalization. Existing velocity stops were preserved; no safety threshold was relaxed to obtain success. Earlier lower-lift failure and successful single-lift evidence remain in their original artifact directories.


The subsequent reset review added mandatory separation checks and explicit supported-goal placement. These intentionally reject the old suite's overlapping/proximate starts, so its original results are historical artifacts tied to their recorded source hashes, not a fresh pass of the current validator. The frozen trial definitions remain unchanged; a new invocation reports invalid resets rather than silently accepting them. Probe scripts refuse to overwrite a nonempty output directory. Future probes record raw object velocity as well as position to permit exact scorer replay.

An independent replay of the two successful archived trajectories against the corrected ordered task scorer confirms reach→contact lift→contact transport→release/support dwell, with final success true and 6.65/6.60 s support dwell. `outputs/manipulation_oracle_suite/scorer_replay.json` records the evidence. The older trace omitted raw velocity, so this replay uses backward differences of logged world positions/timestamps; it is a consistency check, not an exact replay of the original simulator velocity. The scorer now accepts actual hand contact as reach evidence because the palm EE site stays about 10 cm from the grasped object.
