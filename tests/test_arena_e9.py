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
