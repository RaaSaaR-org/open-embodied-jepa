"""TASK-067 run-time pieces: inputs, controllers, G-privileged, the readout, the runner guards.

NumPy tests run in the core job. Torch tests import-skip without torch (core only). Simulation
tests are graphics opt-in (``JEPA_TEST_RENDER=1``), as the other simulator tests.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_jepa import first_policy as fp
from embodied_jepa import first_policy_perception as fpp
from embodied_jepa import first_policy_runtime as rt
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
BOUNDS = (
    np.array((0.0,) * 6 + (-0.5,) * 6 + (-1.0, -1.0), np.float32),
    np.array((0.0,) * 6 + (0.5,) * 6 + (-1.0, 1.0), np.float32),
)
render = pytest.mark.skipif(os.environ.get("JEPA_TEST_RENDER") != "1", reason="graphics opt-in")


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "_run_first_policy", ROOT / "scripts" / "run_first_policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeFK:
    base_position = np.array([0.0, 0.0, 0.79])
    object_support_height = 0.027
    container_surface_z = 0.748

    def pose9(self, state):
        return np.asarray(state, float)[:9] * 0.0 + 1.0


def observation(value=0.1):
    return SimpleNamespace(state=SimpleNamespace(values=np.full((1, 86), value)), images={})


# ----- bounds, assembly, inputs -----
def test_configured_bounds_are_the_protocols():
    lower, upper = rt.configured_bounds()
    assert np.array_equal(lower, BOUNDS[0]) and np.array_equal(upper, BOUNDS[1])


def test_assemble_pins_and_clips():
    action = rt.assemble([0.9, -0.9, 0.2, 0.0, 0.1, -0.1, 0.3], *BOUNDS)
    assert action[12] == -1.0 and np.all(action[:6] == 0.0)
    assert action[6] == 0.5 and action[7] == -0.5 and action[13] == pytest.approx(0.3)
    assert np.allclose(rt.free_of(action), [0.5, -0.5, 0.2, 0.0, 0.1, -0.1, 0.3])
    with pytest.raises(ContractError):
        rt.assemble([np.nan] * 7, *BOUNDS)
    with pytest.raises(ContractError):
        rt.assemble([0.0] * 6, *BOUNDS)


def test_input_vector_order_and_width():
    x = rt.input_vector([1, 2, 3, 4], 5, np.zeros(86), np.ones(9))
    assert x.shape == (132,) and x.dtype == np.float32
    assert np.array_equal(x[:4], [1, 2, 3, 4])
    assert np.allclose(x[4:37], fp.clock_features(5))
    assert np.all(x[37:123] == 0) and np.all(x[123:] == 1)
    with pytest.raises(ContractError):
        rt.input_vector([1, 2, 3], 5, np.zeros(86), np.ones(9))
    with pytest.raises(ContractError):
        rt.input_vector([1, 2, 3, np.inf], 5, np.zeros(86), np.ones(9))


def test_standardiser_floors_a_constant_column():
    rows = np.c_[np.full(10, 3.0), np.arange(10.0)]
    s = rt.Standardiser.fit(rows)
    out = s(rows)
    assert np.isfinite(out).all() and np.all(out[:, 0] == 0.0)
    assert s.std[0] == fp.INPUT_STD_FLOOR


def test_state_of_takes_one_observation():
    assert rt.state_of(observation(0.2)).shape == (86,)
    with pytest.raises(ContractError):
        rt.state_of(SimpleNamespace(state=SimpleNamespace(values=np.zeros((2, 86)))))


# ----- G-privileged -------------------------------------------------------------------------------
class FakeSim:
    def __init__(self):
        self.calls = 0

    def task_truth(self):
        self.calls += 1
        return {"object_position": np.zeros(3)}


class Cheater:
    def __init__(self, sim):
        self.sim = sim

    def act(self, observation, step):  # noqa: ARG002
        self.sim.task_truth()
        return np.zeros(14, np.float32)


def test_counter_catches_a_read_inside_act_and_restores():
    sim = FakeSim()
    counter = rt.PrivilegedReadCounter(sim)
    sim.task_truth()  # a scorer-like read, outside act
    counter.act(Cheater(sim), None, 0)
    assert counter.total == 2 and counter.in_controller == 1
    record = {"task_truth_total": 2, "task_truth_in_controller": 1, "scorer_evaluations": 1}
    assert not rt.privileged_reads_ok(record)
    counter.remove()
    sim.task_truth()
    assert counter.total == 2  # removed: no longer counted
    assert rt.privileged_reads_ok(
        {"task_truth_total": 3, "task_truth_in_controller": 0, "scorer_evaluations": 3}
    )
    assert not rt.privileged_reads_ok(  # an extra read outside act, e.g. a runner-side leak
        {"task_truth_total": 4, "task_truth_in_controller": 0, "scorer_evaluations": 3}
    )


def test_learned_controller_holds_no_live_handle():
    controller = rt.LearnedController(
        lambda x, step: np.full(7, 0.1),
        rt.Standardiser(np.zeros(132), np.ones(132)),
        [0.3, -0.2, 0.5, -0.1],
        FakeFK(),
        BOUNDS,
    )
    for value in vars(controller).values():
        assert not hasattr(value, "task_truth") and not hasattr(value, "sim")
    action = controller.act(observation(), 3)
    assert action.shape == (14,) and action[12] == -1.0
    assert controller.last_input.shape == (132,)


# ----- controllers -----
def test_replay_exhausts_with_the_harness_message():
    replay = rt.ReplayController(np.zeros((2, 14), np.float32))
    replay.act(None, 1)
    with pytest.raises(ContractError, match=rt.EXHAUSTED_MESSAGE):
        replay.act(None, 2)


def test_c0_offsets_have_the_declared_size_and_are_fixed():
    for i in range(fp.C0_RESETS):
        a, p = rt.c0_offsets(i, 1.2, 2.5)
        assert np.linalg.norm(a) == pytest.approx(0.012) and np.linalg.norm(p) == pytest.approx(
            0.025
        )
        again = rt.c0_offsets(i, 1.2, 2.5)
        assert np.array_equal(a, again[0]) and np.array_equal(p, again[1])
    zero = rt.c0_offsets(0, 0.0, 0.0)
    assert not zero[0].any() and not zero[1].any()


def test_perturbed_truth_moves_xy_only():
    truth = {
        "object_position": np.array([0.3, -0.2, 0.77]),
        "plate_position": np.array([0.5, 0.0, 0.746]),
        "other": 1,
    }
    out = rt.perturbed_truth(truth, [0.01, 0.0], [0.0, -0.02])
    assert np.allclose(out["object_position"], [0.31, -0.2, 0.77])
    assert np.allclose(out["plate_position"], [0.5, -0.02, 0.746])
    assert np.allclose(truth["object_position"], [0.3, -0.2, 0.77])  # the input is untouched


def test_a4_truth_uses_estimates_and_constants_only():
    out = rt.a4_truth([0.33, -0.17, 0.49, -0.1], FakeFK())
    assert np.allclose(out["object_position"], [0.33, -0.17, rt.TABLE_Z + 0.027])
    assert np.allclose(out["plate_position"], [0.49, -0.1, rt.PLATE_BODY_Z])
    assert out["container_surface_z"] == 0.748 and np.array_equal(
        out["base_rotation_world"], np.eye(3)
    )


# ----- perception -----
def test_readout_recovers_a_linear_target_and_folds_are_fixed():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(60, 12))
    targets = features[:, :4] * 0.01 + np.array([0.34, -0.18, 0.49, -0.09])
    readout = fpp.XYReadout(features, targets)
    predicted = readout.predict(features[:5])
    assert np.abs(predicted - targets[:5]).max() < 5e-3
    assert np.array_equal(fpp.fold_of(60), fpp.fold_of(60))
    assert sorted(set(fpp.fold_of(60))) == list(range(fp.READOUT_FOLDS))
    apple, plate = fpp.errors_cm(targets[:2] + [0.01, 0, 0, 0.02], targets[:2])
    assert np.allclose(apple, 1.0) and np.allclose(plate, 2.0)


# ----- runner guards and BC rows -----
def test_seed_whitelists_refuse_cohort_c_and_strangers():
    runner = load_runner()
    runner.check_simulated_seeds("D", fp.COHORT_D, smoke=False)
    runner.check_simulated_seeds("dagger_1", fp.seeds_of("dagger_1"), smoke=False)
    for role, seeds in (
        ("D", (45300,)),
        ("dagger_1", (45000,)),
        ("D", (45000, 45000)),
        ("perception_train", (46256,)),
        ("dagger_2", (46900,)),
    ):
        with pytest.raises(fp.GuardError):
            runner.check_simulated_seeds(role, seeds, smoke=False)
    with pytest.raises(fp.GuardError):
        runner.check_simulated_seeds("D", (45300,), smoke=True)
    runner.check_simulated_seeds("D", (46900, 46901), smoke=True)
    with pytest.raises(fp.GuardError):
        runner.check_simulated_seeds("D", fp.COHORT_D[:2], smoke=True)


def test_bc_rows_mask_the_look_displacement_and_drops():
    runner = load_runner()
    length = 20
    phase = np.r_[np.full(8, -1), np.zeros(8), np.full(length - 16, 2)].astype(np.int8)
    apple = np.tile([0.3, -0.2, 0.77], (length + 1, 1))
    apple[12:, 2] += 0.02  # 2 cm of drift in z from frame 12: 3-D, before close (frames 12-15)
    dropped = np.zeros(length + 1, bool)
    dropped[18] = True
    base = np.zeros((length, 14), np.float32)
    base[:, 6] = 0.9  # above the configured +0.5
    root = {
        "valid": np.r_[np.ones(length, bool), False],
        "states": np.zeros((length + 1, 86)),
        "labels": {
            "collector__phase_index": phase,
            "collector__base_action": base,
            "privileged__apple_position_world": apple,
            "privileged__apple_dropped": dropped,
            "privileged__stages": np.zeros((length + 1, 5), bool),
        },
    }
    rows = runner.bc_rows(root, BOUNDS)
    # frames 8-11 kept; 12-15 displaced before close; 16-17 kept (close: the apple may move);
    # 18 dropped; 19 kept
    assert list(rows["steps"]) == [0, 1, 2, 3, 8, 9, 11]
    assert rows["accounting"] == {
        "policy_steps": 12,
        "dropped": 1,
        "post_displacement_pre_grasp": 4,
        "kept": 7,
    }
    assert np.all(rows["labels"][:, 0] == 0.5)


def test_look_state_guard():
    runner = load_runner()
    runner.check_look_states([{"post_look_state": [0.0, 1.0]}, {"post_look_state": [0.0, 1.0]}])
    with pytest.raises(fp.GuardError):
        runner.check_look_states([{"post_look_state": [0.0, 1.0]}, {"post_look_state": [0.0, 1.1]}])


# ----- torch -----
def test_policy_net_training_selection_and_phase_heads():
    torch = pytest.importorskip("torch")
    from embodied_jepa import first_policy_model as fm

    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 132)).astype(np.float32)
    y = np.tanh(x[:, :7]).astype(np.float32) * 0.4
    steps = rng.integers(0, 745, 300)
    trained = fm.train(
        x, y, steps, x[:50], y[:50], steps[:50], device="cpu", updates=40, select_every=20
    )
    assert trained["state"] is not None and trained["record"]["selected"]["update"] in (20, 40)
    model = fm.load(trained["state"])
    single = fm.predictor(model)(x[0], int(steps[0]))
    assert np.allclose(single, fm.batch_predict(model, x[:1], steps[:1])[0], atol=1e-6)
    none = fm.train(
        x,
        y,
        steps,
        x[:50],
        y[:50],
        steps[:50],
        device="cpu",
        updates=20,
        select_every=20,
        min_output_std=10.0,
    )
    assert none["state"] is None and none["record"]["selected"] is None
    f = fm.PolicyNet(len(fp.PHASES))
    with pytest.raises(ContractError):
        f(torch.zeros(2, 132))
    out = f(torch.zeros(2, 132), torch.as_tensor([0, 7]))
    assert out.shape == (2, 7)
    assert list(fm.phases_of([0, 130, 744])) == [0, 1, 7]


# ----- simulation (graphics opt-in) -----
@render
def test_fk_matches_the_live_palm_and_a4_constants_match_the_scene():
    pytest.importorskip("mujoco")
    robot = rt.make_robot()
    fk = rt.PalmFK()
    try:
        truth = robot.reset(46900, object_xy=[0.33, -0.17], plate_xy=[0.5, -0.1])
        state = rt.state_of(robot.observe())
        position, rotation = robot.ee_pose("right")
        pose = fk.pose9(state)
        assert np.allclose(pose[:3], position, atol=1e-9)
        assert np.allclose(pose[3:6], rotation[:, 0], atol=1e-9)
        assert np.allclose(pose[6:9], rotation[:, 1], atol=1e-9)
        a4 = rt.a4_truth([0.33, -0.17, 0.5, -0.1], fk)
        assert np.allclose(a4["object_position"], truth["object_position"], atol=1e-9)
        assert np.allclose(a4["plate_position"], truth["plate_position"], atol=1e-9)
        assert a4["container_surface_z"] == truth["container_surface_z"]
        assert a4["object_support_height"] == truth["object_support_height"]
        assert np.allclose(a4["base_position_world"], truth["base_position_world"], atol=1e-9)
        assert fk._robot.model is not robot.model  # its own scene, never the live one
    finally:
        robot.close()
        fk.close()


@render
def test_counter_total_equals_scorer_calls_on_a_real_attempt():
    pytest.importorskip("mujoco")
    from embodied_jepa.policy_diagnostics import hold_controller

    robot = rt.make_robot()
    try:
        truth, scorer, counter, _obs, facts = rt.reset_and_look(
            robot, 46901, {"object_xy": [0.33, -0.17], "plate_xy": [0.5, -0.1]}
        )
        assert truth["position_frame"] == "world" and facts["post_look_frame"].shape == (
            112,
            112,
            3,
        )
        record = rt.run_policy_steps(
            robot,
            scorer,
            counter,
            rt.HarnessController(hold_controller()),
            bounds=BOUNDS,
            max_steps=5,
        )
        assert record["executed_steps"] == 5 and rt.privileged_reads_ok(record)
    finally:
        robot.close()


def test_a4_rate_is_capped_at_one():
    apple = {0.5: 32, 0.8: 32, 1.0: 32, 1.2: 32}
    plate = {1.0: 32, 1.5: 32, 2.0: 32, 2.5: 32}
    # reference 28/32 while every level is 32/32: the uncapped rate would be 32/28 > 1
    capped = fp.a4_threshold(28, apple, plate, np.full(128, 0.3), np.full(128, 0.5))
    assert capped == 8  # ceil(0.5 * 16 * 1.0), not ceil(0.5 * 16 * 32/28) = 10
    assert fp.A4_RATE_CAP == 1.0 and fp.frozen_block()["a4_threshold"]["rate_cap"] == 1.0


class _SlowRobot:
    def observe(self):
        raise AssertionError("never reached: the cap fires first")

    def stop(self, reason):
        self.stopped = reason


def test_attempt_wall_cap_is_a_guard_not_a_non_success():
    robot = _SlowRobot()
    counter = rt.PrivilegedReadCounter(FakeSim())
    with pytest.raises(fp.GuardError, match="G-cap"):
        rt.run_policy_steps(robot, None, counter, None, bounds=BOUNDS, wall_seconds=-1.0)
    assert robot.stopped == "attempt_wall_cap"


def test_look_state_guard_uses_the_runs_first_reference():
    runner = load_runner()
    reference = runner.check_look_states([{"post_look_state": [0.0, 1.0]}])
    assert runner.check_look_states([{"post_look_state": [0.0, 1.0]}], reference) is reference
    with pytest.raises(fp.GuardError):  # a later group that agrees with itself but not the run
        runner.check_look_states(
            [{"post_look_state": [0.0, 1.2]}, {"post_look_state": [0.0, 1.2]}], reference
        )


@render
def test_a_frame_mismatch_voids_the_attempt():
    pytest.importorskip("mujoco")
    runner = load_runner()
    runner.worker_init()
    try:
        task = {
            "seed": 46902,
            "reset": {"object_xy": [0.33, -0.17], "plate_xy": [0.5, -0.1]},
            "kind": "hold",
            "expected_frame_sha256": "0" * 64,
            "max_steps": 2,
        }
        with pytest.raises(fp.GuardError, match="G-frame"):
            runner.run_task(task)
        good = runner.run_task(task | {"kind": "frame", "expected_frame_sha256": None})
        again = runner.run_task(task | {"expected_frame_sha256": good["post_look_frame_sha256"]})
        assert again["executed_steps"] == 2
    finally:
        runner._W["robot"].close()
        runner._W["fk"].close()
