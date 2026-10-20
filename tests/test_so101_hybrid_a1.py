"""TASK-102 A1 tests (``docs/experiments/so101_hybrid_lift_v1.md`` §18): cohorts, seeds and the
salt, the scripted jaw rule, the arm-only hand-over, the grid and its declared order, the
arm-channel planner against the pinned ``SchemaCEMPlanner``, the warm start, the per-step pose
cost, the selection rule and rows in every branch, a short O-pose episode through the real
transport (execute-k, warm start, G-projection, G-rollout), the byte-identity of every pinned
file and the core-import check. CPU only; physics tests skip only when MuJoCo or the pinned
SO-101 assets are absent."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import so101_approach_diag as ad
from embodied_jepa import so101_closed_loop as cl
from embodied_jepa import so101_hybrid as hy
from embodied_jepa import so101_hybrid_a1 as a1
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
OPEN = np.array([0.1, -0.2, 0.3, 0.4, 0.0, 1.74], np.float64)


def _fetched() -> bool:
    manifest = json.loads((ROOT / "assets/so101_manifest.json").read_text())
    root = ROOT / manifest["local_path"]
    return all((root / relative).is_file() for relative in manifest["sha256"])


# ----- cohorts, seeds, salt (§18.3) --
def test_cohorts_are_declared_and_disjoint_from_every_earlier_range():
    q = a1.COHORTS["Q"]
    assert q.seeds == tuple(range(105000, 105008))
    assert q.reserve_seeds == tuple(range(105008, 105024))
    assert a1.COHORTS["debug"].seeds == tuple(range(105900, 105910))
    mine = set(q.seeds) | set(q.reserve_seeds) | set(a1.COHORTS["debug"].seeds)
    earlier = set(range(0, 64)) | set(range(100, 164)) | set(range(1000, 1016))
    earlier |= set(range(990000, 994000)) | set(range(100100, 100103)) | set(range(100900, 100921))
    earlier |= set(range(101000, 105000))
    for c in list(cl.COHORTS.values()) + list(hy.COHORTS.values()) + list(ad.COHORTS.values()):
        earlier |= set(c.seeds) | set(c.reserve_seeds)
    assert not mine & earlier
    used = set(cl.SALTS.values()) | set(hy.SALTS.values()) | set(ad.SALTS.values())
    assert a1.SALTS["cem"] == 10118 and not set(a1.SALTS.values()) & used


def test_seed_guard_keep_rule_and_planner_seed():
    assert a1.check_seed("Q", 105000) == 105000
    for bad in (104999, 105024, 105900, "105000", 105000.0):
        with pytest.raises(a1.GuardError):
            a1.check_seed("Q", bad)
    kept = a1.keep_cohort("Q", lambda s: (s != 105002, "failed"))
    assert kept["kept"] == [105000, 105001, 105008, 105003, 105004, 105005, 105006, 105007]
    assert kept["rejected"] == [{"seed": 105002, "slot": 2, "reason": "failed"}]
    with pytest.raises(a1.GuardError):
        a1.keep_cohort("Q", lambda s: (False, "no"))
    assert a1.planner_seed(105003) == 10118 * 10**6 + 105003


# ----- the scripted jaw and the arm-only hand-over (§18.2) --
def test_scripted_gripper_is_h_pose_jaw_law():
    scales = np.full(5, 0.0625, np.float32)
    for jaw, want in ((1.745, 1.0), (0.79, 1.0), (0.785, -1.0), (0.2, -1.0)):
        g = OPEN.copy()
        g[cl.JAW] = jaw
        assert a1.scripted_gripper(g) == want
        assert cl.pose_action(np.zeros(6), g, scales)[cl.JAW] == want
    with pytest.raises(ContractError):
        a1.scripted_gripper(np.full(6, np.nan))


def test_arm_handover_rule_reads_the_arm_only():
    goal = OPEN
    far = goal + np.array([0.3, 0, 0, 0, 0, -1.0])
    near_bad_jaw = goal + np.array([0.05, 0, 0, 0, 0, -1.0])  # arm RMS 0.022, jaw 1 rad off
    rule = a1.ArmHandoverRule(goal, 0.08)
    assert [rule.update(q) for q in (far, near_bad_jaw, near_bad_jaw, far)] == [1, 1, 1, 1]
    assert [rule.update(q) for q in (near_bad_jaw,) * 3] == [1, 1, 2]
    assert rule.event["how"] == "tolerance" and rule.event["phase1_commands"] == 6
    assert math.isclose(rule.event["jaw_error_rad"], -1.0)
    assert rule.update(far) == 2 and rule.longest_near == 3
    # hy.HandoverRule needs the jaw as well; with the jaw inside 0.15 rad the two agree
    near = goal + np.array([0.05, 0, 0, 0, 0, 0.1])
    seq = (far, near, near, far, near, near, near, far)
    mine, theirs = a1.ArmHandoverRule(goal, 0.08), hy.HandoverRule(goal, 0.08)
    assert [mine.update(q) for q in seq] == [theirs.update(q) for q in seq]
    assert mine.event == theirs.event
    t = a1.ArmHandoverRule(goal, 0.08, timeout=2)
    assert [t.update(far) for _ in range(3)] == [1, 1, 2] and t.event["how"] == "timeout"
    with pytest.raises(ContractError):
        a1.ArmHandoverRule(goal, 0.2)


# ----- the grid (§18.3, §18.4) --
def test_grid_and_declared_order():
    assert len(a1.CONFIGS) == 16
    for c in a1.CONFIGS.values():
        assert (c["samples"], c["iterations"], c["elites"]) == (256, 6, 32)
    order = sorted(a1.CONFIGS, key=a1.declared_index)
    assert order[0] == "last-H16-k1-w0" and order[1] == "mean-H16-k1-w0"
    assert order[2] == "last-H8-k1-w0" and order[-1] == "mean-H8-k4-w1"
    big = a1.check_config("last-H8-k4-w0" + a1.BIG_SUFFIX)
    assert (big["samples"], big["iterations"], big["elites"]) == (1024, 12, 128)
    assert big["horizon"] == 8 and big["execute"] == 4
    with pytest.raises(ContractError):
        a1.check_config("last-H32-k1-w0")


class _Toy:
    """A deterministic stand-in model: the cost is the squared distance of the cumulative arm
    deltas from a target, per step."""

    def __init__(self, target):
        self.target = np.asarray(target, np.float64)

    def predict(self, latent, candidates):
        return np.cumsum(np.asarray(candidates, np.float64), axis=2)

    def distance(self, predictions, goal):
        d = predictions[..., : cl.ARM_JOINTS] - self.target
        return np.ascontiguousarray((d * d).sum(-1).astype(np.float32))


class _Proj:
    def __init__(self, params, targets):
        from embodied_jepa import so101_closed_loop_runtime as rt

        self.inner = rt.Projector(params, targets)

    def __call__(self, requested):
        return self.inner(requested)


def _params():
    limits = np.array([[-1.9, 1.9]] * 5 + [[-0.17453, 1.7453292]])
    return cl.PreviewParams(limits, np.full(5, 0.0625, np.float32), -0.17453, 1.7453292, 0.25)


def test_arm_planner_cold_equals_the_pinned_planner_and_fixes_the_gripper():
    from embodied_jepa.schema_planning import SchemaCEMPlanner

    params = _params()
    targets = np.array([0, -1.0, 1.0, 0.8, 0, 0.6])
    cfg = a1.planner_config("mean-H8-k1-w0", OPEN, 105001)
    assert cfg.lower_bounds[cl.JAW] == cfg.upper_bounds[cl.JAW] == 1.0
    assert cfg.seed == a1.planner_seed(105001)
    toy = _Toy([2, -1, 0.5, 0, 1])
    mine = a1.ArmCEMPlanner(cfg).plan(toy, None, None, projector=_Proj(params, targets))
    pinned = SchemaCEMPlanner(cfg).plan(toy, None, None, projector=_Proj(params, targets))
    assert np.array_equal(mine.actions, pinned.actions)
    assert np.array_equal(mine.requested_actions, pinned.requested_actions)
    assert np.array_equal(mine.costs, pinned.costs)
    assert np.all(mine.requested_actions[..., cl.JAW] == 1.0)
    # the projected gripper channel carries the embodiment's rate limit (0.6 rad -> +0.25 a step)
    applied, chain = ad.project_chain(mine.requested_actions[0][None], targets, params)
    assert np.allclose(chain[0, :5, cl.JAW], [0.85, 1.1, 1.35, 1.6, 1.7453292])
    # a warm start changes the first mean only
    init = a1.shifted_mean(mine.requested_actions[0], 1, cfg)
    warm = a1.ArmCEMPlanner(cfg).plan(
        toy, None, None, projector=_Proj(params, targets), init_mean=init
    )
    assert warm.costs[0] <= mine.costs[0] + 1e-6
    with pytest.raises(ContractError):
        a1.ArmCEMPlanner(cfg).plan(
            toy, None, None, projector=_Proj(params, targets), init_mean=init[:3]
        )


def test_shifted_mean():
    cfg = a1.planner_config("last-H8-k4-w1", OPEN, 105000)
    prev = np.arange(48, dtype=np.float32).reshape(8, 6) / 100
    out = a1.shifted_mean(prev, 4, cfg)
    assert np.array_equal(out[:4], prev[4:])
    assert np.array_equal(out[4:], a1.cold_mean(cfg)[4:])
    assert np.all(a1.cold_mean(cfg)[:, :5] == 0) and np.all(a1.cold_mean(cfg)[:, 5] == 1)
    for bad in (0, 9):
        with pytest.raises(ContractError):
            a1.shifted_mean(prev, bad, cfg)


def test_pose_steps_and_forms():
    goal = OPEN
    q = np.repeat(goal[None], 4, 0)
    q[:, 0] += np.array([0.4, 0.2, 0.1, 0.0])
    q[:, cl.JAW] -= 1.0  # the jaw is not in the cost
    c = a1.arm_pose_steps(q, goal)
    want = (np.array([0.4, 0.2, 0.1, 0.0]) / math.sqrt(5)) ** 2 / 0.08**2
    assert np.allclose(c, want)
    assert np.array_equal(a1.form_steps(c, "last"), c)
    assert np.allclose(a1.form_steps(c, "mean")[-1], want.mean())
    with pytest.raises(ContractError):
        a1.form_steps(c, "min")


def test_rows_first_match_in_declared_order():
    names = list(a1.CONFIGS)
    zero = dict.fromkeys(names, 0)
    assert a1.row(void=True, pose_counts=None, j=None)["row"] == "A1-VOID"
    assert a1.row(void=False, pose_counts=zero, j={})["row"] == "A1-PLANNER-STILL-BOUND"
    pose = zero | {"last-H8-k4-w0": 8, "mean-H16-k1-w0": 7, "last-H16-k1-w1": 6}
    assert a1.admitted(pose) == ["last-H8-k4-w0", "mean-H16-k1-w0"]

    def jr(h, cmd=30.0, close=0.05):
        return {"h": h, "commands_median": cmd, "closest_median": close}

    j = {"last-H8-k4-w0": jr(5), "mean-H16-k1-w0": jr(6)}
    r = a1.row(void=False, pose_counts=pose, j=j)
    assert r["row"] == "A1-CONFIG" and r["chosen"] == "mean-H16-k1-w0"
    # tie on J: more O-pose hand-overs first
    j = {"last-H8-k4-w0": jr(5), "mean-H16-k1-w0": jr(5)}
    assert a1.row(void=False, pose_counts=pose, j=j)["chosen"] == "last-H8-k4-w0"
    # tie on J and O-pose: fewer commands, then closer, then the declared order
    pose2 = zero | {"last-H8-k4-w0": 8, "mean-H16-k1-w0": 8}
    j = {"last-H8-k4-w0": jr(5, 30), "mean-H16-k1-w0": jr(5, 25)}
    assert a1.row(void=False, pose_counts=pose2, j=j)["chosen"] == "mean-H16-k1-w0"
    j = {"last-H8-k4-w0": jr(5, 25, 0.04), "mean-H16-k1-w0": jr(5, 25, 0.05)}
    assert a1.row(void=False, pose_counts=pose2, j=j)["chosen"] == "last-H8-k4-w0"
    j = {"last-H8-k4-w0": jr(5), "mean-H16-k1-w0": jr(5)}
    assert a1.row(void=False, pose_counts=pose2, j=j)["chosen"] == "mean-H16-k1-w0"
    # J below 4/8 everywhere: the larger budget first, then the row
    j = {"last-H8-k4-w0": jr(2), "mean-H16-k1-w0": jr(3)}
    assert a1.needs_larger_budget(pose, j) == "mean-H16-k1-w0"
    with pytest.raises(ContractError):
        a1.row(void=False, pose_counts=pose, j=j)
    with pytest.raises(ContractError):
        a1.row(void=False, pose_counts=pose, j=j,
               big={"config": "last-H8-k4-w0" + a1.BIG_SUFFIX, "h": 6})  # fmt: skip
    big = {"config": "mean-H16-k1-w0" + a1.BIG_SUFFIX, "h": 4}
    r = a1.row(void=False, pose_counts=pose, j=j, big=big)
    assert r["row"] == "A1-CONFIG" and r["chosen"] == big["config"]
    r = a1.row(void=False, pose_counts=pose, j=j, big=big | {"h": 3})
    assert r["row"] == "A1-MODEL-BOUND" and r["chosen"] is None
    assert a1.needs_larger_budget(pose, {"last-H8-k4-w0": jr(4), "mean-H16-k1-w0": jr(3)}) is None
    with pytest.raises(ContractError):
        a1.row(void=False, pose_counts=pose, j={"last-H8-k4-w0": jr(5)})
    with pytest.raises(ContractError):
        a1.admitted({"last-H8-k4-w0": 8})


# ----- one short O-pose episode through the real transport (physics) --
@pytest.fixture
def sim_ok():
    pytest.importorskip("mujoco")
    if not _fetched():
        pytest.skip("pinned SO-101 assets have not been fetched")


def test_pose_oracle_and_short_episode(sim_ok, tmp_path):
    from embodied_jepa import so101_approach_diag_runtime as ar
    from embodied_jepa import so101_closed_loop_runtime as rt
    from embodied_jepa import so101_hybrid_a1_runtime as r1

    robot = rt.make_robot(False)
    try:
        task = rt.lift_task(robot)
        task.reset(105901)
        sim = robot.sim
        params = cl.PreviewParams.of(robot)
        req = np.random.default_rng(5).uniform(-1, 1, (1, 6, 4, 6)).astype(np.float32)
        rec = ar.RecordingProjector(params, sim.targets)
        b = rec(req)
        q1 = sim.data.qpos[sim.qadr].copy() + np.array([0.2, 0, 0, 0, 0, 0])
        scratch = ar.Scratch(sim, render=False)
        oracle = r1.PoseOracle(scratch, goal_q=q1, form="mean")
        oracle.bind(scratch.snapshot(sim), rec)
        cost = oracle.distance(oracle.predict(None, b.actions), None)
        assert cost.shape == (1, 6, 4) and cost.dtype == np.float32
        steps = [
            a1.arm_pose_steps(scratch.run(oracle.snap, rec.last_targets[0, k])["q"], q1)
            for k in range(6)
        ]
        assert np.allclose(cost[0], np.cumsum(steps, 1) / np.arange(1, 5))
        with pytest.raises(a1.GuardError):
            oracle.predict(None, b.actions + 0.5)
        scratch.close()
    finally:
        robot.close()
    seed = 105900
    goal = np.array([0.15, -0.7, 0.8, 0.9, 0.05, 1.745])
    joints = np.repeat(goal[None, None], 4, 1)
    ctx = {"goals": {"arrays": {"seeds": np.array([seed]), "joints": joints}}, "featuriser": None}
    out = r1.episode(
        {
            "config": "last-H8-k4-w1",
            "cost": "pose",
            "seed": seed,
            "slot": 0,
            "output": str(tmp_path),
            "timeout": 6,
        },  # fmt: skip
        ctx,
    )
    assert out["commands"] == 6 and out["plans"] == 2 and out["rollout_checks"] == 6
    assert out["handover"] == "timeout" and out["scripted_gripper"] == 1.0
    with np.load(tmp_path / f"{seed}.npz") as d:
        assert d["replanned"].tolist() == [True, False, False, False, True, False]
        assert np.all(d["requested"][:, cl.JAW] == 1.0)
        assert np.all(np.diff(d["targets"][:, cl.JAW]) >= 0)  # the jaw opens, never closes
    with pytest.raises(FileExistsError):
        r1.episode({"config": "last-H8-k4-w1", "cost": "pose", "seed": seed, "slot": 0,
                    "output": str(tmp_path), "timeout": 6}, ctx)  # fmt: skip
    with pytest.raises(a1.GuardError):
        r1.episode({"config": "last-H8-k4-w1", "cost": "pose", "seed": 105000, "slot": 0,
                    "output": str(tmp_path), "timeout": 6},
                   {"goals": {"arrays": {"seeds": np.array([105000]),
                                         "joints": ctx["goals"]["arrays"]["joints"]}},
                    "featuriser": None})  # fmt: skip


# ----- byte identity (§18.6) --
def test_pinned_files_are_byte_identical():
    import hashlib

    assert a1.differing_pins(ROOT) == []
    for manifest in (cl.MANIFEST, hy.MANIFEST):
        pins = json.loads((ROOT / manifest).read_text())
        files = pins["files_sha256"] | pins.get("composed_files_sha256", {})
        bad = [
            rel
            for rel, pin in files.items()
            if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != pin
        ]
        assert not bad, f"{manifest}: {bad}"
    assert json.loads((ROOT / hy.MANIFEST).read_text())["status"] == "CANDIDATE"


def test_no_core_import():
    import subprocess
    import sys

    code = (
        "import sys, embodied_jepa; "
        "assert 'embodied_jepa.so101_hybrid_a1' not in sys.modules; "
        "import embodied_jepa.so101_hybrid_a1; "
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
