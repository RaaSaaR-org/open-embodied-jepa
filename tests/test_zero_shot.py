"""TASK-088 (``zero_shot*``): layouts, scorers, the switch, controllers' inputs, bars, rows and
the planner contract. Synthetic contract evidence only; the simulator tests are graphics opt-in
(``JEPA_TEST_RENDER=1``)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import grounded_wm as gw
from embodied_jepa import play_corpus as pc
from embodied_jepa import zero_shot as zs
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
LEWM = ROOT / "third_party" / "le-wm"
render = pytest.mark.skipif(os.environ.get("JEPA_TEST_RENDER") != "1", reason="graphics opt-in")


# ----- cohorts, seeds, layouts -------------------------------------------------------------------
def test_cohorts_are_disjoint_and_sized():
    seen = set()
    for name, seeds in zs.COHORTS.items():
        assert not (seen & set(seeds)), name
        seen |= set(seeds)
    assert len(zs.COHORTS["gated-reach"]) == 64 and len(zs.COHORTS["gated-grasp"]) == 64
    assert len(zs.COHORTS["k0-reach"]) == 32 and len(zs.COHORTS["dev-grasp"]) == 16
    assert zs.cohort_task("gated-grasp") == "grasp"
    with pytest.raises(ContractError):
        zs.check_seed("gated-reach", 88000)
    # never a seed of TASK-085 / TASK-087
    assert not seen & set(range(850000, 853200)) and not seen & set(range(85000, 88000))


@pytest.mark.parametrize("task", zs.TASKS)
def test_layouts_inside_the_test_object_region(task):
    for seed in range(200):
        layout = zs.sample_layout(task, seed)
        (x0, x1), (y0, y1) = zs.OBJECT_REGION
        assert 1 <= len(layout["objects"]) <= 3
        for name, (x, y) in layout["objects"].items():
            assert x0 <= x <= x1 and y0 <= y <= y1, name
        names = list(layout["objects"])
        pts = [(np.asarray(layout["plate"]), pc.PLATE_RADIUS)] + [
            (np.asarray(layout["objects"][n]), pc.OBJECTS[n]["radius"]) for n in names
        ]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = np.linalg.norm(pts[i][0] - pts[j][0])
                assert d >= pts[i][1] + pts[j][1] + zs.CLEARANCE - 1e-9
        if task == "grasp":
            assert layout["target"] in layout["objects"]
        else:
            assert layout["target"] is None
        pc.check_layout(layout)
    assert zs.sample_layout(task, 5) == zs.sample_layout(task, 5)
    assert zs.sample_layout(task, 5, 1) != zs.sample_layout(task, 5, 0)


def test_grasp_targets_cover_every_object():
    targets = {zs.sample_layout("grasp", s)["target"] for s in range(100)}
    assert targets == set(zs.GRASP_OBJECTS)


# ----- scorers -----------------------------------------------------------------------------------
def test_dwell_never_latches_a_transient_crossing():
    near, far = 0.01, 0.2
    assert not zs.reach_success([far] * 5 + [near] * 9 + [far] * 50)
    assert zs.reach_success([far] * 5 + [near] * 10)
    assert zs.dwell_reached([True] * 9 + [False] + [True] * 10, 10) == 19
    assert not zs.reach_success([zs.REACH_TOLERANCE + 1e-6] * 30)
    assert zs.reach_success([zs.REACH_TOLERANCE] * 10)


def test_lift_needs_height_and_contact_together():
    rise = [0.06] * 25
    assert zs.lift_success(rise, [True] * 25)
    assert not zs.lift_success(rise, [True, False] * 12 + [True])
    assert not zs.lift_success([0.049] * 30, [True] * 30)
    assert not zs.lift_success([0.06] * 19 + [0.0] + [0.06] * 19, [True] * 39)


# ----- the switch --------------------------------------------------------------------------------
def _subgoals(n=4):
    out = []
    for k in range(n):
        out.append(
            {
                "palm": np.array([0.3, -0.1, 0.1 + 0.02 * k]),
                "rotation": np.eye(3),
                "state28": np.zeros(28, np.float32),
            }
        )
    return out


def test_switch_needs_pose_dwell_or_timeout():
    sg = _subgoals()
    sw = zs.SubgoalSwitch(sg)
    hand = np.zeros(7)
    for _ in range(zs.SWITCH_DWELL - 1):
        assert sw.update(sg[0]["palm"], hand, np.eye(3)) == 0
    assert sw.update(sg[0]["palm"], hand, np.eye(3)) == 1
    # palm right, orientation off: no switch until the timeout
    turned = pc.rotation_delta([0, 0, 0.3])
    for _ in range(zs.SUBGOAL_TIMEOUT - 1):
        assert sw.update(sg[1]["palm"], hand, turned) == 1
    assert sw.update(sg[1]["palm"], hand, turned) == 2
    assert sw.log[-1]["reason"] == "timeout"
    # hand off by more than the RMS bound: no switch
    for _ in range(5):
        assert sw.update(sg[2]["palm"], hand + 0.2, np.eye(3)) == 2
    sw.index = 3
    assert sw.update(np.zeros(3), hand, np.eye(3)) == 3  # the last subgoal is kept


def test_rotation_angle():
    assert zs.rotation_angle(np.eye(3), np.eye(3)) == pytest.approx(0.0, abs=1e-6)
    assert zs.rotation_angle(np.eye(3), pc.rotation_delta([0, 0, 0.4])) == pytest.approx(0.4)


# ----- controllers read no object truth ----------------------------------------------------------
class _NoTruthRobot:
    """Exposes proprioception and the embodiment's kinematics only; any simulator access fails."""

    manifest = {
        "translation_per_step_m": 0.015,
        "rotation_per_step_rad": 0.06,
        "right_open_rad": [0.0, 0.2, -0.1, 0.0, 0.1, 0.0, 0.1],
        "right_closed_rad": [0.0, -0.6, -1.1, 0.9, 1.2, 0.9, 1.2],
    }

    def __init__(self):
        from embodied_jepa.contracts import StateSchema

        names = [f"{j}.position" for j in gw.STATE_JOINTS] + [
            f"{j}.velocity" for j in gw.STATE_JOINTS
        ]
        self.state_schema = StateSchema(tuple(names), ("rad",) * 28, "test")

    def _snapshot_kinematics(self):
        return "snapshot"

    def ee_pose(self, side, *, data=None):
        assert data == "snapshot", "the palm pose must come from the measured joints"
        return np.array([0.3, -0.15, 0.07]), np.eye(3)

    @property
    def sim(self):
        raise AssertionError("a controller read the simulator")


class _Obs:
    robot_state = np.zeros((1, 28), np.float32)
    images = {pc.CAMERA: np.zeros((1, 112, 112, 3), np.uint8)}


def test_controllers_read_no_object_truth():
    robot = _NoTruthRobot()
    goal = {"task": "grasp", "subgoals": _subgoals()}
    goal["subgoals"][2]["state28"][7:14] = robot.manifest["right_closed_rad"]
    for controller in (zs.HoldController(), zs.RandomController(1), zs.IKController()):
        for index in range(4):
            action = controller.act(robot, _Obs(), goal, index)
            assert action.shape == (14,) and np.all(np.abs(action) <= 1)
            assert np.all(action[:6] == 0) and action[12] == -1


def test_ik_presses_on_the_grasp_subgoal_only():
    robot = _NoTruthRobot()
    goal = {"task": "grasp", "subgoals": _subgoals()}
    for sg in goal["subgoals"]:
        sg["palm"] = np.array([0.3, -0.15, 0.07])
    c = zs.IKController()
    assert c.act(robot, _Obs(), goal, 1)[8] == 0  # at the goal palm: no z command
    assert c.act(robot, _Obs(), goal, 2)[8] < 0  # grasp: commanded below


def test_random_controller_is_seeded():
    a = [zs.RandomController(3).act(None, None, None, 0) for _ in range(2)]
    assert np.array_equal(a[0], a[1])


def test_synergy_grasp_ends():
    robot = _NoTruthRobot()
    assert zs.synergy_grasp(robot, robot.manifest["right_open_rad"]) == pytest.approx(-1)
    assert zs.synergy_grasp(robot, robot.manifest["right_closed_rad"]) == pytest.approx(1)


# ----- bars, intervals, rows ---------------------------------------------------------------------
def test_bar_rule():
    assert zs.bar_from_ceiling("reach", 32, 32) == 0.80
    assert zs.bar_from_ceiling("reach", 24, 32) == pytest.approx(0.675)
    assert zs.bar_from_ceiling("grasp", 20, 32) == 0.40
    assert zs.bar_from_ceiling("grasp", 12, 32) == pytest.approx(0.3375)
    assert zs.bar_count(0.80, 64) == 52
    assert zs.bar_count(0.40, 64) == 26


def test_exact_interval_known_values():
    lo, hi = zs.exact_interval(118, 128)
    assert lo == pytest.approx(0.861, abs=1e-3) and hi == pytest.approx(0.962, abs=1e-3)
    assert zs.exact_interval(0, 10)[0] == 0.0 and zs.exact_interval(10, 10)[1] == 1.0


def test_mcnemar():
    assert zs.mcnemar_one_sided(0, 0) == 1.0
    assert zs.mcnemar_one_sided(13, 0) == pytest.approx(2**-13)


def _results(p_reach, p_grasp, g_reach=0, g_grasp=0):
    def arm(k):
        return [1] * k + [0] * (64 - k)

    return {
        "reach": {"P": arm(p_reach), "G": arm(g_reach)},
        "grasp": {"P": arm(p_grasp), "G": arm(g_grasp)},
    }


def test_rows():
    bars = {"reach": 0.8, "grasp": 0.4}
    assert zs.row(_results(52, 26), bars)["row"] == "Z3-PASS"
    assert zs.row(_results(52, 25), bars)["row"] == "Z3-REACH"
    assert zs.row(_results(51, 64), bars)["row"] == "Z3-LOW"
    assert zs.row(_results(10, 0, g_reach=32), bars)["row"] == "Z3-LOW"
    assert zs.row(_results(31, 0, g_reach=31), bars)["row"] == "Z3-STOP-CANDIDATE"
    assert zs.row(_results(64, 64), bars, void="x")["row"] == "Z3-VOID"
    r = zs.row(_results(10, 0, g_reach=60, g_grasp=30), bars)
    assert r["G_meets"] == {"reach": True, "grasp": True}


# ----- lambda ------------------------------------------------------------------------------------
def test_pose_lambda_equalises_the_terms():
    rng = np.random.default_rng(0)
    episodes = [{"frames": 50}] * 4
    lat = rng.normal(0, 2.0, (200, 16, 4)).astype(np.float32)
    st = rng.normal(0, 1.0, (200, 28))
    out = zs.pose_lambda(lat, st, episodes, np.zeros(28), np.ones(28), pairs=4000)
    assert out["lambda"] == pytest.approx(4.0, rel=0.1)  # 2*var 4 / 2*var 1
    assert out["lambda"] * out["pose_mse"] == pytest.approx(out["latent_mse"])


# ----- the planner contract ----------------------------------------------------------------------
@pytest.fixture
def torch_mod():
    torch = pytest.importorskip("torch")
    if not (LEWM / "jepa.py").exists():
        pytest.skip("pinned LeWM source not fetched")
    os.chdir(ROOT)
    return torch


SMALL = {
    "depth": 1,
    "heads": 2,
    "head_dim": 16,
    "mlp_dim": 32,
    "proj_hidden": 32,
    "state_hidden": 16,
    "inverse_hidden": 16,
    "readout_hidden": 16,
}


@pytest.mark.parametrize(("arm", "w_lat", "w_pose"), [("P", 1, 0), ("G", 1, 1), ("G", 0, 1)])
def test_planner_contract(torch_mod, arm, w_lat, w_pose):
    torch = torch_mod
    from embodied_jepa import grounded_wm_model as gm
    from embodied_jepa import zero_shot_runtime as zr
    from embodied_jepa.planning import CEMConfig, CEMPlanner

    torch.manual_seed(0)
    m = gm.GroundedWM(arm, config=SMALL)
    m.set_moments(np.zeros(28), np.ones(28), np.zeros(3), np.ones(3))
    m.eval()
    pm = zr.PlannerModel(m, w_lat=w_lat, w_pose=w_pose, lam=2.0, device="cpu")
    rng = np.random.default_rng(0)
    latent = pm.encode(rng.normal(size=(16, gw.K)), rng.normal(size=28))
    goal = pm.encode_goal(rng.normal(size=(16, gw.K)), rng.normal(size=28))
    cand = rng.uniform(-1, 1, (1, 6, 3, 14)).astype(np.float32)
    pred = pm.predict(latent, cand)
    cost = pm.distance(pred, goal)
    assert cost.shape == (1, 6, 3) and cost.dtype == np.float32 and np.isfinite(cost).all()
    # the cost is the declared sum
    z = pred["z"]
    expect = w_lat * (z - goal["z"][:, None, None]).square().mean((-1, -2))
    if w_pose:
        expect = expect + w_pose * 2.0 * (pred["s"][..., :14] - goal["s_std"][:14]).square().mean(
            -1
        )
    assert np.allclose(cost, expect.numpy(), atol=1e-5)
    cfg = CEMConfig(
        horizon=3,
        samples=8,
        iterations=2,
        elites=2,
        seed=1,
        lower_bounds=tuple(zs.FIXED_LOWER.tolist()),
        upper_bounds=tuple(zs.FIXED_UPPER.tolist()),
    )
    plan = CEMPlanner(cfg).plan(pm, latent, goal)
    a = plan.actions[0, 0]
    assert np.all(a[:6] == 0) and a[12] == -1 and np.all(np.abs(a) <= 1)


def test_pose_cost_needs_a_state_model(torch_mod):
    from embodied_jepa import grounded_wm_model as gm
    from embodied_jepa import zero_shot_runtime as zr

    m = gm.GroundedWM("P", config=SMALL)
    with pytest.raises(ContractError):
        zr.PlannerModel(m, w_lat=1, w_pose=1, lam=1.0, device="cpu")


def test_core_import_stays_light():
    code = (
        "import sys, embodied_jepa, embodied_jepa.zero_shot;"
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


# ----- simulator (graphics opt-in) ---------------------------------------------------------------
@render
def test_goals_and_ik_episode_end_to_end():
    from embodied_jepa import zero_shot_runtime as zr

    robot = pc.make_play_robot()
    try:
        goal = zs.make_goal(robot, "reach", 88900)
        assert goal["subgoals"][0]["frame"].shape == (112, 112, 3)
        assert zs.make_goal(robot, "reach", 88900)["target"] == goal["target"]
        out = zr.run_episode(robot, goal, zs.IKController())
        assert out["success"] and out["steps"] <= zs.BUDGET["reach"]
        out = zr.run_episode(robot, goal, zs.HoldController())
        assert not out["success"] and out["steps"] == zs.BUDGET["reach"]
        g = zs.make_goal(robot, "grasp", 88950)
        assert len(g["subgoals"]) == len(zs.SUBGOALS)
    finally:
        robot.close()
