import numpy as np
import pytest

from embodied_jepa import arena_e9 as ae
from embodied_jepa import arena_transport as at
from embodied_jepa.contracts import ContractError


def test_import_stays_light():
    import subprocess
    import sys

    code = (
        "import sys, embodied_jepa.arena_e9\n"
        "bad = [m for m in ('torch', 'isaaclab', 'isaaclab_arena', 'omni', 'warp', 'mujoco')"
        " if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_rate_conversion_is_exact_every_two_commands():
    steps = [ae.arena_steps(k) for k in range(10)]
    assert steps == [2, 3] * 5
    for k in range(0, 10, 2):
        assert (steps[k] + steps[k + 1]) * ae.ARENA_DT == pytest.approx(2 * ae.COMMAND_DT)
    with pytest.raises(ContractError):
        ae.arena_steps(-1)


def _pose(xyz, yaw):
    q = ae.matrix_to_quat_xyzw(ae.rot_z(yaw))
    return np.r_[xyz, q]


def test_quaternion_round_trip_and_frame_map():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        r = at.quat_xyzw_to_matrix(q)
        assert np.allclose(at.quat_xyzw_to_matrix(ae.matrix_to_quat_xyzw(r)), r)
    pelvis = _pose([0.26, 0.07, -0.045], 0.3)
    f = ae.FrameMap(pelvis)
    # the Arena pelvis maps onto our fixed pelvis; a point ahead of it maps ahead of ours
    assert np.allclose(f.point(pelvis[:3]), ae.OUR_PELVIS_POS)
    ahead = pelvis[:3] + ae.rot_z(0.3) @ [0.4, -0.1, 0.0]
    assert np.allclose(f.point(ahead), np.add(ae.OUR_PELVIS_POS, [0.4, -0.1, 0.0]))
    assert np.allclose(f.inverse_point(f.point(ahead)), ahead)
    assert np.allclose(f.vector(ae.rot_z(0.3) @ [1, 0, 0]), [1, 0, 0])


def test_layout_keeps_pelvis_offsets_and_shelf_height():
    pelvis = _pose([0.10, 0.07, 0.02], 0.1)
    lay = ae.layout_in_arena([0.34, -0.18], [0.49, -0.09], arena_pelvis_pose=pelvis)
    assert np.allclose(lay["apple"][:2], pelvis[:2] + ae.rot_z(0.1)[:2, :2] @ [0.34, -0.18])
    assert np.allclose(lay["plate"][:2], pelvis[:2] + ae.rot_z(0.1)[:2, :2] @ [0.49, -0.09])
    top = ae.ARENA_SHELF_TOP_Z + ae.ARENA_AIRGAP
    assert lay["apple"][2] == pytest.approx(top + ae.ARENA_APPLE_ORIGIN_ABOVE_BOTTOM)
    assert lay["plate"][2] == pytest.approx(top)
    shifted = ae.layout_in_arena(
        [0.34, -0.18], [0.49, -0.09], arena_pelvis_pose=pelvis, apple_com_offset_xy=[0.01, 0.0]
    )
    assert shifted["apple"][0] == pytest.approx(lay["apple"][0] - 0.01)
    # the tutorial's settled pelvis is 1.5 cm below the shelf top; ours is 5.3 cm above our table
    assert ae.shelf_height_match(arena_pelvis_z=-0.045) == pytest.approx(0.068)


def test_command_action_maps_by_name():
    mirror = [f"j{i}" for i in range(at.G1_NUM_JOINTS)]
    arena = list(reversed(mirror))
    targets = np.arange(at.G1_NUM_JOINTS, dtype=float) / 100
    a = ae.command_action(targets, mirror, arena)
    assert a.shape == (50,)
    assert np.allclose(a[: at.G1_NUM_JOINTS], targets[::-1])
    assert a[at.BASE_HEIGHT] == pytest.approx(0.75)
    assert np.all(a[at.NAVIGATE] == 0)
    assert np.allclose(ae.reorder(a[:43], arena, mirror), targets, atol=1e-6)
    with pytest.raises(ContractError):
        ae.command_action(targets[:-1], mirror, arena)
    with pytest.raises(ContractError):
        ae.reorder(targets, mirror, [*arena[:-1], "x"])


def _record(command, *, apple, plate, vel=(0, 0, 0), hand=0.0, success=False):
    return {
        "command": command,
        "success": success,
        "object_dropped": False,
        "success_inputs": {"apple_plate_force_n": 1.0} if success else None,
        "apple_com_pose": [*apple, 0, 0, 0, 1],
        "apple_com_vel": [*vel, 0, 0, 0],
        "plate_pose": [*plate, 0, 0, 0, 1],
        "pelvis_pose": [0, 0, 0, 0, 0, 0, 1],
        "apple_plate_force_n": 1.0,
        "apple_hand_force_n": hand,
    }


def test_arena_verdicts():
    plate = (0.6, 0.0, -0.03)
    rest = 0.035
    on = (0.61, 0.0, -0.03 + rest)
    records = [_record(c, apple=on, plate=plate) for c in range(30) for _ in range(2)]
    v = ae.arena_verdicts(records, rest_height_m=rest)
    assert v["at_rest_arena"] and not v["arena_success"]
    assert v["final_distance_arena_cm"] == pytest.approx(1.0)
    # contact success at any step counts for Arena, even with the hand on the apple
    records[5] = _record(2, apple=on, plate=plate, hand=2.0, success=True)
    v = ae.arena_verdicts(records, rest_height_m=rest)
    assert v["arena_success"] and v["arena_first_success_step"] == 5
    assert v["arena_success_with_hand_contact"] and v["at_rest_arena"]
    # moving, off-centre, too high or touched in the window: not at rest
    # a reported (PhysX) velocity on a still apple fails only the labelled second reading
    noisy = records[:-1] + [_record(29, apple=on, plate=plate, vel=(0.005, 0, 0))]
    v = ae.arena_verdicts(noisy, rest_height_m=rest)
    assert v["at_rest_arena"] and not v["at_rest_arena_reported_vel"]
    for bad in (
        {"apple": (0.6105, 0.0, -0.03 + rest)},  # moved 0.5 mm in one 0.06 s command
        {"apple": (0.65, 0.0, -0.03 + rest)},
        {"apple": (0.6, 0.0, -0.03 + rest + 0.02)},
        {"hand": 1.0},
    ):
        kw = {"apple": on, "plate": plate, **bad}
        recs = records[:-1] + [_record(29, **kw)]
        assert not ae.arena_verdicts(recs, rest_height_m=rest)["at_rest_arena"], bad
    with pytest.raises(ContractError):
        ae.arena_verdicts([], rest_height_m=rest)
    # the window is the thresholds' (honoured, not just echoed)
    from embodied_jepa.at_rest import AtRestThresholds

    late = records[:-2] + [_record(29, apple=(0.65, 0.0, -0.03 + rest), plate=plate)] * 2
    assert not ae.arena_verdicts(late, rest_height_m=rest)["at_rest_arena"]
    v = ae.arena_verdicts(records, rest_height_m=rest, thresholds=AtRestThresholds(window_steps=5))
    assert v["window_steps"] == 5


def test_kinematic_trajectory():
    names = ["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"]
    default = dict.fromkeys(names, 0.0)
    segs = [{"name": "a", "q": {names[0]: 1.0, names[1]: -1.0}}]
    out_names, rows, labels = ae.kinematic_trajectory(default, segs, settle=2, ramp=4, hold=3)
    assert out_names == names and len(rows) == 2 + 14 == len(labels)
    assert np.allclose(rows[2 + 4], [1.0, -1.0]) and labels[2 + 4] == "a:hold"
    assert np.allclose(rows[-1], [0.0, 0.0])


# ----- plumbing: Arena endpoint + mirror over a displaced MuJoCo scene ----------------------------
def test_arena_mirror_over_mujoco_sham_reproduces_a_plain_e9_attempt():
    pytest.importorskip("mujoco")
    from embodied_jepa import apple_to_plate_v2 as v2
    from embodied_jepa import first_policy as fp
    from embodied_jepa import first_policy_runtime as rt
    from embodied_jepa import isaac_e9 as ie
    from embodied_jepa import resting_expert as rx
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.simulation import MuJoCoSimulation

    class Blind(MuJoCoSimulation):
        def render(self, camera="onboard_rgb"):
            return np.zeros((self.height, self.width, 3), np.uint8)

    plain = G1Embodiment(
        Blind(
            render=False,
            object_kind="apple",
            container_kind="plate",
            width=fp.IMAGE_SIZE,
            height=fp.IMAGE_SIZE,
        )
    )
    v2.apply_v2_scene(plain.model)
    sham = ae.MuJoCoArenaSham()
    endpoint = ae.ArenaEndpoint(sham.call, list(plain.sim.joint_names))
    mirror, _ = ae.make_arena_mirror_robot(endpoint)
    reset = {
        "object_xy": [0.3238940037143524, -0.18840221016330622],
        "plate_xy": [0.47520843833476756, -0.08489034958650456],
    }
    runs = []
    for robot in (plain, mirror):
        recorder = ie.StepRecorder(robot)
        summary, arrays = rx.run_attempt(
            robot,
            rt.configured_bounds(),
            seed=50200,
            reset=reset,
            plate_offset=[0.0, 0.0],
            make_expert=lambda truth: rx.RestingPlaceExpert(truth, **ie.E9),
        )
        runs.append((summary, arrays, recorder.arrays()))
    (s0, a0, r0), (s1, a1, r1) = runs
    assert s0["complete"] and s1["complete"], (s0["stop_reason"], s1["stop_reason"])
    assert s0["at_rest"] == s1["at_rest"]
    # Exact through the reset and the look: frames, joint order, layout and initial pose.
    look = len(r0["rec_time"]) - s0["steps"]
    diff = np.abs(r0["rec_q"] - r1["rec_q"]).max(axis=1)
    assert diff[: look + 1].max() < 1e-9
    # After that the mirror's plain forward pass (Arena has no mj_step substep layout, so the
    # adapter sends no "pre" state) differs from mj_step's lagged kinematics by up to a few
    # mrad, as the Isaac replay's first version did (docs/ISAAC_E9_REPLAY.md §1).
    assert diff.max() < 5e-3
    assert np.abs(a0["apple_pos"][-1] - a1["apple_pos"][-1]).max() < 1e-3
    assert np.abs(np.asarray(mirror.sim.layout_error_m["apple"])).max() < 1e-9
    # Arena-side records: one per Arena step, 2 and 3 per command alternately
    commands = len(r1["rec_time"])
    assert len(endpoint.records) == sum(ae.arena_steps(k) for k in range(commands))
    v = ae.arena_verdicts(endpoint.records, rest_height_m=v2_rest_height(plain))
    assert v["at_rest_arena"] == s0["at_rest"]


def v2_rest_height(robot):
    """Our resting height of the apple centre above the plate's bottom (sham plate frame)."""
    sim = robot.sim
    return float(
        sim.container_surface_z
        + sim.object_support_height
        - (sim.model.body("plate").pos[2] - ae.OUR_PLATE_BASE_HALF)
    )


def test_gravity_offset_holds_the_arm_with_arena_gains():
    pytest.importorskip("mujoco")
    from embodied_jepa.simulation import MuJoCoSimulation

    sim = MuJoCoSimulation(object_kind="apple", container_kind="plate", render=False)
    names = list(sim.joint_names)
    kp = ae.upper_body_kp(names)
    assert kp[names.index("right_shoulder_pitch_joint")] == 100.0
    assert kp[names.index("right_wrist_yaw_joint")] == 20.0
    assert kp[names.index("left_hand_index_0_joint")] == 4.0
    assert kp[names.index("left_knee_joint")] == 0.0 and kp[names.index("waist_yaw_joint")] == 0.0
    sim.data.qpos[sim.qadr[names.index("right_shoulder_pitch_joint")]] = -0.8  # arm forward
    sim.mj.mj_forward(sim.model, sim.data)
    off = ae.gravity_offset(sim)
    bias = sim.data.qfrc_bias[sim.vadr]
    i = names.index("right_shoulder_pitch_joint")
    assert abs(off[i]) > 1e-3 and off[i] * kp[i] == pytest.approx(bias[i])
    assert off[names.index("left_knee_joint")] == 0.0


# ----- e9-arena and the shelf-press probe (docs/ARENA.md §9) -----------------------------------
def _tilted_pose(xyz, yaw, pitch):
    c, s = np.cos(pitch), np.sin(pitch)
    ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.r_[xyz, ae.matrix_to_quat_xyzw(ae.rot_z(yaw) @ ry)]


def test_anchored_target_is_e9_offset_upright_and_world_vertical_when_tilted():
    pelvis = _pose([0.02, 0.08, 0.03], 0.2)
    obj_base = np.array([0.34, -0.18, -0.027])
    anchor = ae.base_to_world(obj_base, pelvis)
    assert np.allclose(ae.world_to_base(anchor, pelvis), obj_base)
    got = ae.anchored_target(anchor, pelvis, dx=-0.015, dz=0.13)
    assert np.allclose(got, obj_base + [-0.015, 0.0, 0.13])
    tilted = _tilted_pose([0.02, 0.08, 0.03], 0.2, np.radians(-4.0))  # leaning back 4 degrees
    got = ae.anchored_target(anchor, tilted, dx=-0.015, dz=0.13)
    offset = ae.base_to_world(got, tilted) - anchor
    assert np.allclose(offset, [*(-0.015 * ae.heading(tilted)[:2]), 0.13])
    assert np.allclose(ae.heading(tilted), [np.cos(0.2), np.sin(0.2), 0.0])


def test_stop_height():
    h = ae.stop_height(finger_drop_m=0.11, apple_com_above_shelf_m=0.026, clearance_m=0.01)
    assert h == pytest.approx(0.094)
    t1 = ae.declared_variant("T1", 0.01, "hold")
    assert t1["stop_height_m"] == pytest.approx(0.1019) and t1["close_mode"] == "hold"
    assert ae.declared_variant("T3", 0.005, "hold")["stop_height_m"] == pytest.approx(0.0969)
    with pytest.raises(ContractError):
        ae.stop_height(finger_drop_m=-0.1, apple_com_above_shelf_m=0.026, clearance_m=0.01)


def test_mirror_state_can_zero_leg_velocity_only():
    names = ["left_hip_pitch_joint", "left_knee_joint", "right_ankle_roll_joint", "waist_yaw_joint"]
    names += ["right_elbow_joint"]
    raw = {
        "pelvis_pose": _pose([0, 0, 0], 0.0),
        "apple_com_pose": [0.3, 0, 0, 0, 0, 0, 1],
        "apple_pose": [0.3, 0, 0, 0, 0, 0, 1],
        "apple_com_vel": [0.0] * 6,
        "plate_pose": [0.5, 0, 0, 0, 0, 0, 1],
        "q": [0.0] * 5,
        "qd": [6.0, -7.0, 8.0, 1.0, 2.0],
    }
    kw = {"arena_names": names, "mirror_names": names, "time": 0.0}
    assert ae.mirror_state(raw, **kw)["qd"] == [6.0, -7.0, 8.0, 1.0, 2.0]
    assert ae.mirror_state(raw, **kw, zero_leg_velocity=True)["qd"] == [0, 0, 0, 1.0, 2.0]


def _initial_truth(obj_base, plate_base):
    base = np.asarray(ae.OUR_PELVIS_POS)
    return {
        "position_frame": "world",
        "base_position_world": base,
        "base_rotation_world": np.eye(3),
        "object_position": base + obj_base,
        "plate_position": base + plate_base,
        "container_surface_z": 0.752,
        "object_support_height": 0.027,
    }


def test_e9_arena_targets_follow_the_live_pelvis_and_objects():
    obj_base = np.array([0.34, -0.18, -0.026])
    plate_base = np.array([0.49, -0.09, -0.047])
    truth = _initial_truth(obj_base, plate_base)
    pelvis = _pose([0.0, 0.0, 0.0], 0.0)
    raw = {
        "pelvis_pose": pelvis,
        "apple_com_pose": [*ae.base_to_world(obj_base, pelvis), 0, 0, 0, 1],
        "plate_pose": [*ae.base_to_world(plate_base, pelvis), 0, 0, 0, 1],
        "hand_min_z": ae.ARENA_SHELF_TOP_Z + 0.03,
    }
    x = ae.ArenaAdaptedE9(truth, live=lambda: raw, stop_height_m=0.09)
    e9 = x.e9  # e9 itself on the same reset: the reference for every unchanged target
    ref = {p.name: np.asarray(p.target_base) for p in e9.phases}
    ee = np.zeros(3)
    assert np.allclose(x.target("orient", raw, ee), ref["orient"])
    assert np.allclose(x.target("descend", raw, ee), obj_base + [-0.015, 0, 0.09])
    assert np.allclose(x.target("close", raw, ee), obj_base + [-0.015, 0, 0.09])
    for name in ("lift", "transfer", "lower", "steady", "open", "clear", "retreat"):
        assert np.allclose(x.target(name, raw, ee), ref[name]), name
    # the base steps back 8 cm: the grasp targets move 8 cm forward in the base frame, and the
    # release is e9's reach-limited rule on the plate 8 cm further away
    from embodied_jepa import resting_expert as rx

    stepped = _pose([-0.08, 0.0, 0.0], 0.0)
    raw2 = {**raw, "pelvis_pose": stepped}
    for name in ("close", "lift"):
        assert np.allclose(x.target(name, raw2, ee), x.target(name, raw, ee) + [0.08, 0, 0])
    release = rx.RestingPlaceExpert.release_pose(
        plate_base[:2] + [0.08 + 0.015, 0.0], reach_radius_m=0.485, floor=0.10, ceiling=0.26
    )
    assert np.allclose(x.target("lower", raw2, ee), release)
    assert np.allclose(x.target("retreat", raw2, ee), release + [0, 0, 0.08])
    # the grasp anchor froze at the first close command: an apple carried along does not drag it
    raw3 = {
        **raw,
        "apple_com_pose": [*ae.base_to_world(obj_base + [0, 0, 0.1], pelvis), 0, 0, 0, 1],
    }
    assert np.allclose(x.target("lift", raw3, ee), ref["lift"])
    # the shelf-clearance close: the palm moves so the hand's lowest point sits 1 cm above the shelf
    servo = ae.ArenaAdaptedE9(truth, live=lambda: raw, stop_height_m=0.09, close_mode="shelf_servo")
    servo.target("descend", raw, ee)
    ee_base = obj_base + [-0.015, 0, 0.09]
    got = servo.target("close", raw, ee_base)
    assert np.allclose(got, ee_base + [0, 0, -0.02])
    with pytest.raises(ContractError):
        ae.ArenaAdaptedE9(truth, live=lambda: raw, stop_height_m=0.09, close_mode="squeeze")


def test_shelf_press_probe_targets():
    truth = _initial_truth(np.array([0.34, -0.18, -0.026]), np.array([0.49, -0.09, -0.047]))
    pelvis = _pose([0.0, 0.0, 0.0], 0.0)
    site = np.array([0.34, -0.18, ae.ARENA_SHELF_TOP_Z + 0.026])
    raw = {"pelvis_pose": pelvis, "hand_min_z": ae.ARENA_SHELF_TOP_Z + 0.05}
    base = ae.world_to_base(site, pelvis)
    for mode, dz in (("press", 0.052), ("high", 0.13)):
        p = ae.ShelfPressProbe(truth, live=lambda: raw, site_w=site, mode=mode)
        assert np.allclose(p.target("close", raw, np.zeros(3)), base + [-0.015, 0, dz])
        assert np.allclose(p.target("orient", raw, np.zeros(3)), base + [-0.015, 0, 0.13])
    p = ae.ShelfPressProbe(truth, live=lambda: raw, site_w=site, mode="hover")
    ee = base + [0.0, 0.0, 0.1]
    assert np.allclose(p.target("hold", raw, ee), [base[0] - 0.015, base[1], ee[2] - 0.04])
    with pytest.raises(ContractError):
        p.target("hold", {"pelvis_pose": pelvis}, ee)
    assert p.max_steps == sum(c for _, c, _ in ae.PROBE_PHASES) <= 740


def test_e9_arena_over_mujoco_sham_completes_like_e9():
    """Plumbing: on the sham (a fixed, upright pelvis) e9-arena with e9's own descent target
    (0.052 m) and the plain close is e9 with live targets; it must rest the apple as e9 does."""
    pytest.importorskip("mujoco")
    from embodied_jepa import first_policy as fp
    from embodied_jepa import first_policy_runtime as rt
    from embodied_jepa import resting_expert as rx
    from embodied_jepa.simulation import MuJoCoSimulation

    names = MuJoCoSimulation(object_kind="apple", container_kind="plate", render=False)
    mirror_names = list(names.joint_names)
    names.close()
    sham = ae.MuJoCoArenaSham()
    endpoint = ae.ArenaEndpoint(sham.call, mirror_names, zero_leg_velocity=True)
    robot, _ = ae.make_arena_mirror_robot(endpoint)
    assert fp.IMAGE_SIZE > 0
    reset = {
        "object_xy": [0.3238940037143524, -0.18840221016330622],
        "plate_xy": [0.47520843833476756, -0.08489034958650456],
    }
    holder = {}

    def make(truth):
        holder["x"] = ae.ArenaAdaptedE9(truth, live=endpoint.live, stop_height_m=0.052)
        return holder["x"]

    summary, _ = rx.run_attempt(
        robot,
        rt.configured_bounds(),
        seed=50200,
        reset=reset,
        plate_offset=[0.0, 0.0],
        make_expert=make,
    )
    assert summary["complete"] and summary["at_rest"], summary["stop_reason"]
    log = holder["x"].log
    assert [e["phase"] for e in log][0] == "orient" and log[0]["command"] > 0  # after the look
    ref = {p.name: p for p in rx.RestingPlaceExpert(holder["x"].e9_truth, **ae.ie.E9).phases}
    first = {e["phase"]: np.asarray(e["target_base"]) for e in reversed(log)}
    # the live apple has settled a few mm below the reset truth; the plate has not moved
    assert np.allclose(first["orient"][:2], ref["orient"].target_base[:2], atol=1e-6)
    assert abs(first["orient"][2] - ref["orient"].target_base[2]) < 5e-3
    assert np.allclose(first["lower"], ref["lower"].target_base, atol=1e-6)
