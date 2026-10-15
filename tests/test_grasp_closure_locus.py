"""TASK-094 (docs/experiments/grasp_closure_locus_v1.md): the K follower (bitwise TASK-088's
``IKController`` at c = +1.0; the held level reaches the request after the ramp and nothing else
changes), the displaced goal, K's plumbing check, rows, band and bootstrap; the root types' step
counts and the executed-difference arithmetic; the squeeze branch's reproduction of the
demonstrator (graphics opt-in); M's wins, bars and positive control; S's measures; the L rows in
first-match order with the n_d < 16 case; Stage 0's row; the power recompute's cells and rule; the
runtime's cost decomposition and CEM call (a small model on the CPU); the seed ranges, the pins,
the working-directory guard, the ledger claim's place and the core-import check."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import grasp_closure_locus as gl
from embodied_jepa import grasp_rate_projected as rp
from embodied_jepa import zero_shot as zs
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
LEWM = ROOT / "third_party" / "le-wm"
render = pytest.mark.skipif(os.environ.get("JEPA_TEST_RENDER") != "1", reason="graphics opt-in")


def _manifest():
    return json.loads((ROOT / "configs" / "g1_sim_action.json").read_text())


def _robot():
    m = _manifest()
    m.setdefault("translation_per_step_m", m.get("translation_per_step_m"))
    return types.SimpleNamespace(manifest=m)


def _hand(grasp):
    m = _manifest()
    opened, closed = np.asarray(m["right_open_rad"]), np.asarray(m["right_closed_rad"])
    return (opened + (grasp + 1) * 0.5 * (closed - opened)).astype(np.float32)


def _goal(s_star=0.68):
    rng = np.random.default_rng(0)
    subgoals = []
    for k, g in enumerate((-1.0, -1.0, s_star, s_star)):
        s = rng.normal(0, 0.1, 28).astype(np.float32)
        s[7:14] = _hand(g)
        subgoals.append({
            "frame": rng.integers(0, 255, (112, 112, 3), dtype=np.uint8),
            "state28": s,
            "palm": np.array([0.3, -0.15, 0.1 - 0.02 * k], np.float32),
            "rotation": np.eye(3, dtype=np.float32),
        })  # fmt: skip
    return {"task": "grasp", "seed": 94500, "layout": {"target": "cube"}, "target_z0": 0.8,
            "subgoals": subgoals}  # fmt: skip


@pytest.fixture
def fake_proprio(monkeypatch):
    """A synthetic proprioception (no MuJoCo): a palm and rotation that move with the step."""
    state = {"t": 0}

    def proprio(robot, observation):
        t = state["t"]
        state["t"] += 1
        s = np.zeros(28, np.float32)
        s[7:14] = _hand(-1 + 0.05 * (t % 40))
        return {"state28": s, "palm": np.array([0.3 + 0.001 * t, -0.15, 0.2 - 0.001 * t]),
                "rotation": np.eye(3)}  # fmt: skip

    monkeypatch.setattr(zs, "proprio", proprio)
    return state


def _drive(controller, robot, goal, indices, state):
    out = []
    for k, i in enumerate(indices):
        state["t"] = k
        out.append(controller.act(robot, None, goal, i))
    return np.stack(out)


INDICES = [0] * 5 + [1] * 5 + [2] * 20 + [3] * 20


# ----- K: the follower (§4.1) ------------------------------------------------------------------
def test_held_follower_is_bitwise_ik_at_plus_one(fake_proprio):
    robot, goal = _robot(), _goal()
    a = _drive(zs.IKController(), robot, goal, INDICES, fake_proprio)
    b = _drive(gl.HeldIKController("K-1.0"), robot, goal, INDICES, fake_proprio)
    assert a.dtype == b.dtype == np.float32
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


@pytest.mark.parametrize("arm", ["K-0.9", "K-0.8", "K-0.7", "K-s*", "K-0.6", "K-open"])
def test_held_level_reaches_the_request_and_nothing_else_changes(fake_proprio, arm):
    robot, goal = _robot(), _goal(0.68)
    ik = _drive(zs.IKController(), robot, goal, INDICES, fake_proprio)
    ctrl = gl.HeldIKController(arm)
    held = _drive(ctrl, robot, goal, INDICES, fake_proprio)
    level = gl.arm_level(arm, gl.s_star_of(robot, goal))
    assert ctrl.level == level
    assert np.array_equal(
        np.delete(ik, 13, 1).view(np.uint32), np.delete(held, 13, 1).view(np.uint32)
    )
    req = held[:, 13]
    phase = np.isin(INDICES, gl.CLOSING_SUBGOALS)
    assert np.all(req[~phase] == np.float32(-1.0))
    assert req[phase][-1] == np.float32(level)  # after the ramp
    assert np.array_equal(req, gl.expected_requests(INDICES, level))
    assert abs(gl.s_star_of(robot, goal) - 0.68) < 1e-5


def test_k_arms_levels_and_offsets():
    assert list(gl.K_ARMS) == [
        "K-1.0", "K-0.9", "K-0.8", "K-0.7", "K-s*", "K-0.6", "K-open",
        "K-off-x-1.0", "K-off-x-s*", "K-off-y-1.0", "K-off-y-s*",
    ]  # fmt: skip
    assert gl.OFFSETS == {"x": (0.012, 0.0, 0.0), "y": (0.0, 0.012, 0.0)}
    assert gl.arm_slug("K-off-x-s*") == "K-off-x-sstar" and gl.arm_from_slug("K-sstar") == "K-s*"
    assert gl.arm_from_slug("ik") == "ik"
    with pytest.raises(ContractError):
        gl.HeldIKController("K-0.5")


def test_displaced_goal():
    goal = _goal()
    for axis in ("x", "y"):
        off = gl.displaced_goal(goal, axis)
        assert gl.displacement_ok(goal, off, axis)
        for a, b in zip(goal["subgoals"], off["subgoals"], strict=True):
            assert np.allclose(b["palm"] - a["palm"].astype(np.float64), gl.OFFSETS[axis])
            assert b["frame"] is a["frame"] and b["state28"] is a["state28"]
        assert not gl.displacement_ok(goal, off, "y" if axis == "x" else "x")
    bad = gl.displaced_goal(goal, "x")
    bad["subgoals"][1] = dict(bad["subgoals"][1], state28=bad["subgoals"][1]["state28"] + 1)
    assert not gl.displacement_ok(goal, bad, "x")


def _episode(indices, level, *, lag_extra=0, wrong=False):
    req = gl.expected_requests(indices, level)
    applied = np.zeros((len(indices), 14), np.float32)
    g = -1.0
    for k in range(len(indices)):
        g = float(np.clip(req[k], g - gl.RATE_LIMIT, g + gl.RATE_LIMIT))
        applied[k, 13] = np.float32(g)
    if lag_extra:
        applied[-1, 13] = np.float32(level - 0.01)
    log = [{"index": i, "closing": i in gl.CLOSING_SUBGOALS, "requested": float(r)}
           for i, r in zip(indices, req, strict=True)]  # fmt: skip
    if wrong:
        log[-1]["requested"] = float(np.float32(level) - np.float32(0.001))
    return log, applied


@pytest.mark.parametrize("level", [1.0, 0.9, 0.68, 0.6, -1.0])
def test_k_plumbing(level):
    log, applied = _episode(INDICES, level)
    p = gl.k_plumbing(log, applied, level)
    assert p["ok"], p
    assert p["after_ramp_requests_at_c"] == p["after_ramp_commands"] > 0
    assert 0 <= p["applied_reaches_c_after"] <= 12
    assert p["allowed_applied_at_c"] == p["allowed_commands"] > 0
    # an episode stopped by the embodiment before the rate limit lets the hand reach c
    short = INDICES[:13]
    log_s, applied_s = _episode(short, level)
    assert gl.k_plumbing(log_s, applied_s, level)["ok"]
    if level > 0.5:
        assert gl.k_plumbing(log_s, applied_s, level)["allowed_commands"] == 0
    rec = {
        "subgoal_index": [e["index"] for e in log],
        "requested": [e["requested"] for e in log],
        "applied": applied.tolist(),
        "level": level,
        "plumbing": {"decision_ok": True},
    }
    assert gl.replumb(rec)["ok"]
    assert not gl.replumb(rec | {"plumbing": {"decision_ok": False}})["ok"]
    assert not gl.k_plumbing(*_episode(INDICES, level, wrong=True), level)["ok"]
    if level != -1.0:
        assert not gl.k_plumbing(*_episode(INDICES, level, lag_extra=1), level)["ok"]
    # a stopped episode logs one request more than it executed
    assert gl.k_plumbing(log + [log[-1]], applied, level)["ok"]
    assert not gl.k_plumbing(log + [log[-1]] * 2, applied, level)["ok"]


def test_slip_and_measures():
    assert gl.slipped([0, 0.02, 0.03, 0.03], [True, True, True, False])
    assert not gl.slipped([0, 0.005, 0.0], [True, True, False])
    assert not gl.slipped([0, 0.02, 0.06], [True, True, True])


# ----- K: statistics and rows (§4.2, §4.3) -----------------------------------------------------
def _arms(n, k1, others, k_open=0, seed=0):
    rng = np.random.default_rng(seed)
    base = np.zeros(n, bool)
    base[rng.permutation(n)[:k1]] = True
    out = {"K-1.0": base}
    for arm in gl.K_ARMS:
        if arm == "K-1.0":
            continue
        if arm == "K-open":
            v = np.zeros(n, bool)
            v[:k_open] = True
            out[arm] = v
            continue
        lose = others.get(arm, 0)
        v = base.copy()
        v[np.flatnonzero(base)[:lose]] = False
        out[arm] = v
    return out


def test_bootstrap_and_contrast():
    idx = gl.bootstrap_indices(128)
    assert idx.shape == (10_000, 128)
    assert np.array_equal(idx, gl.bootstrap_indices(128))
    a = np.r_[np.ones(90, bool), np.zeros(38, bool)]
    b = a.copy()
    b[:40] = False
    c = gl.paired_contrast(a, b, idx)
    assert c["delta"] == pytest.approx(40 / 128)
    assert c["discordant_first"] == 40 and c["discordant_second"] == 0
    assert c["p_one_sided"] == pytest.approx(0.5**40)
    lo, hi = c["ci"]
    assert lo < c["delta"] < hi
    means = (a.astype(int) - b.astype(int))[idx].mean(1)
    assert hi == pytest.approx(np.percentile(means, 97.5))


def test_void_band():
    assert gl.void_band(128) == {"low": 63, "high": 114, "open_max": 3}
    assert gl.void_band(32) == {"low": 16, "high": 28, "open_max": 0}


def test_k_rows_first_match():
    n = 128
    assert gl.k_row(_arms(n, 88, {}))["row"] == "K-NULL"
    big = {"K-s*": 40, "K-0.6": 45}
    assert gl.k_row(_arms(n, 88, big))["row"] == "K-CAUSAL"
    # Δ's bound below δ_K but Δ_0.6's not: K-UNCLEAR
    assert gl.k_row(_arms(n, 88, {"K-0.6": 40}))["row"] == "K-UNCLEAR"
    # a significant loss below 0.15 whose interval reaches 0.15: K-UNCLEAR
    r = gl.k_row(_arms(n, 88, {"K-s*": 14, "K-0.6": 14}))
    assert r["row"] == "K-UNCLEAR" and r["delta_p"] < 0.01 and r["delta_ci"][1] >= 0.15
    # K-VOID before anything else
    assert gl.k_row(_arms(n, 120, big))["row"] == "K-VOID"
    assert gl.k_row(_arms(n, 62, big))["row"] == "K-VOID"
    assert gl.k_row(_arms(n, 63, {}))["row"] != "K-VOID"
    assert gl.k_row(_arms(n, 88, big, k_open=4))["row"] == "K-VOID"
    assert gl.k_row(_arms(n, 88, big, k_open=3))["row"] == "K-CAUSAL"
    assert gl.k_row(_arms(n, 88, big), plumbing_ok=False)["row"] == "K-VOID"
    assert gl.k_row(_arms(n, 88, {}), void="a cohort run twice")["row"] == "K-VOID"
    r = gl.k_row(_arms(n, 88, big))
    assert set(r["contrasts"]) == {f"{a} vs {b}" for a, b in gl.K_CONTRASTS}


# ----- M: roots, branches and the executed difference (§5.1-§5.2) ------------------------------
def test_root_types_and_step_counts():
    assert gl.ROOT_TYPES == ("Rs", "R2") and gl.RS_CLOSE == 8 and gl.CLOSE == 35
    assert gl.BRANCH_LENGTH == {"Rs": 97, "R2": 70}
    assert gl.RS_TARGET_GRASP == pytest.approx(0.4545, abs=1e-4)
    assert gl.ROOT_REQUEST["Rs"] == pytest.approx(0.6)
    assert gl.ROOT_GOAL == {"Rs": 2, "R2": 3}
    assert gl.BRANCHES_BY_COHORT["dev-roots"] == ("1.0", "s*", "control")
    assert gl.BRANCHES_BY_COHORT["gated-roots"] == ("1.0", "0.9", "0.8", "0.7", "s*", "control")
    assert gl.branch_level("control", 0.68) == -1.0 and gl.branch_level("s*", 0.68) == 0.68


def test_executed_difference_arithmetic():
    for s in np.linspace(0.40, 0.8179, 60):
        assert gl.window_differences(s, "Rs") >= 7, s
        assert gl.window_differences(s, "R2") == 8, s
    assert gl.window_differences(0.82, "Rs") == 6  # above 0.818: steps 3-8 only
    assert gl.window_differences(0.63, "Rs") == 8  # below 0.636 it differs from step 1
    assert gl.window_differences(1.0, "Rs") == 0
    a = np.zeros((97, 14), np.float32)
    b = a.copy()
    b[1:8, 13] = 0.5
    b[0, 6] = 0.1
    assert gl.executed_differences(a, b) == {"any_channel": 8, "grasp": 7}


@render
def test_squeeze_branch_is_the_demonstrators_continuation():
    from embodied_jepa import play_corpus as pc

    robot = pc.make_play_robot()
    try:
        r = gl.build_locus_root(robot, 94599, gl.BRANCHES_STAGE0)
        g = zs.make_grasp_goal(robot, 94599)
    finally:
        robot.close()
    assert not r["problems"], r["problems"]
    assert r["attempt"] == g["attempt"] and r["layout"] == g["layout"]
    for k in range(4):  # the same subgoals as TASK-088's demonstrator
        assert np.array_equal(r["subgoals"][k]["state28"], g["subgoals"][k]["state28"])
        assert np.array_equal(r["subgoals"][k]["frame"], g["subgoals"][k]["frame"])
    for root in gl.ROOT_TYPES:
        sq = r["branches"][f"{root}/1.0"]
        assert sq["squeeze_checks"] and all(sq["squeeze_checks"].values())
        assert sq["lift"]  # the demonstrator's own continuation lifts
        assert sq["applied"].shape == (gl.BRANCH_LENGTH[root], 14)
        assert sq["frames"].shape == (8, 112, 112, 3) and sq["states"].shape == (8, 28)
        assert not r["branches"][f"{root}/control"]["lift"]
        diff = gl.executed_differences(sq["applied"], r["branches"][f"{root}/s*"]["applied"])
        assert diff["any_channel"] >= 7
    assert abs(zs.synergy_grasp(robot, r["roots"]["Rs"]["t0"]) - gl.RS_TARGET_GRASP) < 1e-3
    assert abs(zs.synergy_grasp(robot, r["roots"]["R2"]["t0"]) - 1.0) < 1e-3


# ----- M: wins, bars, positive control (§5.4-§5.5) ---------------------------------------------
def test_wins_bars_and_positive_control():
    assert list(gl.wins([1, 2, 3], [2, 2, 1])) == [True, False, False]
    assert list(gl.discordant([1, 1, 0, 1], [0, 1, 0, 0], [1, 1, 1, 0])) == [True, False, False,
                                                                             False]  # fmt: skip
    assert gl.ceiling_ranks(24, 32) and not gl.ceiling_ranks(23, 32)
    assert gl.ceiling_ranks(12, 16) and not gl.ceiling_ranks(0, 0)
    assert gl.g_bar(1.0, 32) == 28 and gl.g_bar(0.75, 32) == 22 and gl.g_bar(0.9, 32) == 26
    assert gl.g_ranks([28, 28, 0], 32, 32) and not gl.g_ranks([28, 27, 27], 32, 32)
    assert gl.positive_control_need(64) == 56 and gl.positive_control_need(32) == 28
    assert gl.positive_control_need(128) == 112  # R27.183 (1)
    assert gl.exec_diff_limit(128) == 16 == gl.EXEC_DIFF_VOID and gl.exec_diff_limit(32) == 4
    assert gl.miss_need(32) == 8 and gl.miss_need(17) == 5
    assert gl.search_misses([8, 8, 0], 32) and not gl.search_misses([8, 7, 7], 32)


def test_search_measures_and_executed_level():
    # R27.183 (4): a miss is partial and the substituted plan strictly cheaper than the plan
    m = gl.search_measures(0.5, 1.2, 1.0)
    assert not m["squeezes"] and m["miss"] and not m["partial_preferred"]
    m = gl.search_measures(0.5, 0.9, 1.0)
    assert not m["miss"] and m["partial_preferred"]
    m = gl.search_measures(0.5, 1.0, 1.0)  # a tie is partial preferred, not a miss
    assert not m["miss"] and m["partial_preferred"]
    m = gl.search_measures(0.95, 2.0, 1.0)
    assert m["squeezes"] and not m["miss"] and not m["partial_preferred"]
    assert gl.search_measures(0.9, 2.0, 1.0)["squeezes"]
    # the previous comparison, reported only: CEM's cost against the squeeze branch's
    m = gl.search_measures(0.5, 1.2, 1.0, squeeze_branch_cost=1.5, cem_cost=1.1)
    assert m["miss"] and not m["miss_v0"] and m["partial_preferred_v0"]
    m = gl.search_measures(0.5, 0.8, 1.0, squeeze_branch_cost=0.5, cem_cost=0.7)
    assert not m["miss"] and m["miss_v0"]
    # R27.183 (3): ℓ is the median executed grasp over window steps 3-7
    path = np.array([0.636, 0.818, 0.9, 0.95, 0.7, 1.0, 0.92, 0.1])
    assert gl.window_level(path) == pytest.approx(0.92)
    assert gl.window_level(np.r_[np.ones(7), -1.0]) == 1.0  # step 8 does not count
    assert gl.LEVEL_STEPS == (3, 7)
    mf = _manifest()
    opened, closed = np.asarray(mf["right_open_rad"]), np.asarray(mf["right_closed_rad"])
    lo, hi = np.minimum(opened, closed) - 0.5, np.maximum(opened, closed) + 0.5
    phys = np.stack([lo, hi], 1)
    geo = rp.RateGeometry(opened, closed, mf["joint_speed_limit_rad_s"] * 0.05, phys,
                          phys.astype(np.float32))  # fmt: skip
    t0 = geo.targets(np.float64(gl.RS_TARGET_GRASP)).astype(np.float64)
    plan = np.zeros((8, 14), np.float32)
    plan[:, 12] = -1
    plan[:, 13] = 1.0
    path = gl.executed_level(geo, plan, t0)
    assert path[0] == pytest.approx(gl.RS_TARGET_GRASP + gl.RATE_LIMIT, abs=1e-6)
    assert path[7] == pytest.approx(1.0)
    plan[:, 13] = 0.68
    assert gl.executed_level(geo, plan, t0)[7] == pytest.approx(0.68, abs=1e-6)
    # the squeeze branch's requests, executed from Rs's T₀, close at the rate limit to +1
    plan[:, 13] = gl.squeeze_requests("Rs")
    path = gl.executed_level(geo, plan, t0)
    assert path[0] == pytest.approx(gl.RS_TARGET_GRASP + gl.RATE_LIMIT, abs=1e-6)
    assert path[2:] == pytest.approx(np.ones(6), abs=1e-6)


def test_squeeze_requests_and_substituted_plan():
    """The squeeze branch's requests are the demonstrator's ramp continued from the root: at Rs
    0.6 + 0.2 then +1 (as ``_Branch`` computes them), at R2 +1 throughout."""
    rs, r2 = gl.squeeze_requests("Rs"), gl.squeeze_requests("R2")
    assert rs.dtype == np.float32 and rs.shape == (8,)
    g = -1.0
    for _ in range(gl.RS_CLOSE):
        g = float(np.clip(1.0, g - zs.GRASP_RAMP, g + zs.GRASP_RAMP))
    assert g == pytest.approx(gl.ROOT_REQUEST["Rs"])
    assert rs[0] == np.float32(float(np.clip(1.0, g - zs.GRASP_RAMP, g + zs.GRASP_RAMP)))
    assert rs[0] == pytest.approx(0.8) and np.all(rs[1:] == 1.0) and np.all(r2 == 1.0)
    rng = np.random.default_rng(3)
    plan = rng.uniform(-1, 1, (8, 14)).astype(np.float32)
    for root in gl.ROOT_TYPES:
        sub = gl.substituted_plan(plan, root)
        assert np.array_equal(sub[:, gl.GRASP_DIM], gl.squeeze_requests(root))
        rest = [i for i in range(14) if i != gl.GRASP_DIM]
        assert rp.bits_equal(sub[:, rest], plan[:, rest])
    assert rp.bits_equal(plan, plan.copy())  # the plan itself is not changed
    with pytest.raises(ContractError):
        gl.substituted_plan(plan[:7], "Rs")


# ----- the L rows and the task row (§7) --------------------------------------------------------
def _t(n_d, c, g=(32, 32, 32), s=(0, 0, 0)):
    return {"n_d": n_d, "ceiling_wins": c, "g_wins": list(g), "s_misses": list(s)}


def test_l_rows_first_match():
    ok = _t(32, 32)
    assert gl.l_row({"Rs": ok, "R2": ok})["row"] == "L-MIXED"
    assert gl.l_row({"Rs": ok, "R2": ok}, void="positive control")["row"] == "L-VOID"
    assert gl.l_row({"Rs": _t(15, 0), "R2": _t(10, 0)})["row"] == "L-UNREAD"
    # an unread root type does not decide the row
    assert gl.l_row({"Rs": _t(15, 0), "R2": ok})["row"] == "L-MIXED"
    cost = gl.l_row({"Rs": _t(32, 20), "R2": _t(32, 32, g=(0, 0, 0))})
    assert cost["row"] == "L-COST" and cost["l_cost_root_types"] == ["Rs"]
    assert cost["readings"]["R2"]["own_row"] == "L-MODEL"
    assert gl.l_row({"Rs": ok, "R2": _t(32, 32, g=(27, 27, 32))})["row"] == "L-MODEL"
    assert gl.l_row({"Rs": ok, "R2": _t(32, 32, s=(8, 8, 0))})["row"] == "L-SEARCH"
    assert gl.l_row({"Rs": _t(32, 32, g=(0, 0, 0)), "R2": _t(32, 32, s=(9, 9, 9))})["row"] == (
        "L-MODEL")  # fmt: skip
    assert gl.l_row({"Rs": _t(16, 12), "R2": _t(0, 0)})["read"] == ["Rs"]


def test_task_row_and_s0_row():
    assert gl.task_row({"row": "K-NULL"}, None) == "K-NULL"
    assert gl.task_row({"row": "K-CAUSAL"}, {"row": "L-COST"}) == "K-CAUSAL + L-COST"
    kw = {"n_d": {"Rs": 10, "R2": 9}, "power_ok": True, "stage0_gpu_hours": 0.2,
          "projected_gpu_hours": 0.5}  # fmt: skip
    assert gl.s0_row({"a": True}, **kw)["row"] == "S0-PASS"
    assert gl.s0_row({"a": False}, **kw)["row"] == "S0-ESCALATE"
    assert gl.s0_row({"a": True}, **(kw | {"n_d": {"Rs": 7, "R2": 7}}))["row"] == "S0-ESCALATE"
    assert gl.s0_row({"a": True}, **(kw | {"n_d": {"Rs": 7, "R2": 8}}))["row"] == "S0-PASS"
    assert gl.s0_row({"a": True}, **(kw | {"power_ok": False}))["row"] == "S0-ESCALATE"
    assert gl.s0_row({"a": True}, **(kw | {"stage0_gpu_hours": 0.41}))["row"] == "S0-ESCALATE"
    assert gl.s0_row({"a": True}, **(kw | {"projected_gpu_hours": 0.61}))["row"] == "S0-ESCALATE"


def test_s0p_row_reads_n_d_at_128_roots():
    """Stage 0′ (R27.183 (5)): escalate only if the development rates project n_d below 16 at both
    root types at 128 roots (Rs: 6 / 32 x 128 = 24)."""
    assert gl.projected_nd(6) == 24 and gl.projected_nd(1) == 4 and gl.projected_nd(4) == 16
    kw = {"n_d_dev": {"Rs": 6, "R2": 1}, "stage0_gpu_hours": 0.12, "projected_gpu_hours": 0.4}
    r = gl.s0p_row({"a": True}, **kw)
    assert r["row"] == "S0-PASS" and r["n_d_projected_at_128"] == {"Rs": 24.0, "R2": 4.0}
    assert gl.s0p_row({"a": True}, **(kw | {"n_d_dev": {"Rs": 3, "R2": 3}}))["row"] == (
        "S0-ESCALATE")  # fmt: skip
    assert gl.s0p_row({"a": True}, **(kw | {"n_d_dev": {"Rs": 4, "R2": 0}}))["row"] == "S0-PASS"
    assert gl.s0p_row({"a": False}, **kw)["row"] == "S0-ESCALATE"
    assert gl.s0p_row({"a": True}, **(kw | {"stage0_gpu_hours": 0.41}))["row"] == "S0-ESCALATE"
    assert gl.s0p_row({"a": True}, **(kw | {"projected_gpu_hours": 0.61}))["row"] == (
        "S0-ESCALATE")  # fmt: skip
    # reported: P(n_d >= 16) at 128 roots (0.977; over the rate's uncertainty, 0.84)
    assert gl.read_probability(6 / 32) == pytest.approx(0.977, abs=1e-3)
    assert gl.read_probability_jeffreys(6) == pytest.approx(0.84, abs=0.01)
    assert gl.read_probability(1 / 32) < 1e-4


# ----- power (§8.4) ----------------------------------------------------------------------------
@pytest.mark.parametrize("model", gl.POWER_MODELS)
def test_joint_cells_have_the_declared_marginals(model):
    for delta, d0, d6 in ((0.0, 0.08, 0.0), (0.15, 0.03, 0.25), (0.30, 0.08, 0.30)):
        p = gl.joint_cells(delta, d6, d0, model)
        a = np.array([(i >> 2) & 1 for i in range(8)])
        b = np.array([(i >> 1) & 1 for i in range(8)])
        c = np.array([i & 1 for i in range(8)])
        assert p.sum() == pytest.approx(1)
        assert p[a == 1].sum() == pytest.approx(gl.P1)
        assert p[(a == 1) & (b == 0)].sum() == pytest.approx(delta + d0)
        assert p[(a == 0) & (b == 1)].sum() == pytest.approx(d0)
        assert p[(a == 1) & (c == 0)].sum() == pytest.approx(d6 + d0)
        assert p[(a == 0) & (c == 1)].sum() == pytest.approx(d0)
    with pytest.raises(ContractError):
        gl.joint_cells(0.3, 0.1, 0.03, "nested")


def test_power_cells_and_rule():
    r = gl.k_power_cell(0.0, 0.03, 0.0, sims=40, resamples=500)
    assert sum(r["rates"].values()) == pytest.approx(1) and r["rates"]["K-NULL"] > 0.8
    r = gl.k_power_cell(0.5, 0.03, 0.5, sims=40, resamples=500)
    assert r["rates"]["K-CAUSAL"] > 0.9
    cells = gl.power_grid()
    assert (0.30, 0.08, 0.30, "independent") in cells and (0.0, 0.03, 0.0, "independent") in cells
    rows = [
        {
            "delta": c[0],
            "d0": c[1],
            "delta06": c[2],
            "model": c[3],
            "rates": {"K-NULL": 0.97, "K-CAUSAL": 0.99, "K-UNCLEAR": 0.0},
        }
        for c in cells
    ]
    assert not gl.power_rule(rows)["escalate"]
    rows[0]["rates"] = {"K-NULL": 0.94, "K-CAUSAL": 0.0, "K-UNCLEAR": 0.06}  # Δ = 0, d0 = 0.03
    assert rows[0]["delta"] == 0.0 and gl.power_rule(rows)["escalate"]


# ----- runtime (torch; a small model on the CPU) -----------------------------------------------
@pytest.fixture
def small_pm():
    torch = pytest.importorskip("torch")
    if not (LEWM / "jepa.py").exists():
        pytest.skip("pinned LeWM source not fetched")
    os.chdir(ROOT)
    from embodied_jepa import grounded_wm_model as gm
    from embodied_jepa import zero_shot_runtime as zr

    small = {"depth": 1, "heads": 2, "head_dim": 16, "mlp_dim": 32, "proj_hidden": 32,
             "state_hidden": 16, "inverse_hidden": 16, "readout_hidden": 16}  # fmt: skip
    torch.manual_seed(0)
    m = gm.GroundedWM("G", config=small)
    m.set_moments(np.zeros(28), np.ones(28), np.zeros(3), np.ones(3))
    m.eval()
    return zr.PlannerModel(m, w_lat=1.0, w_pose=1.0, lam=2.0552, device="cpu")


def test_cost_decomposition_and_ceiling(small_pm):
    from embodied_jepa import grasp_closure_locus_runtime as lr
    from embodied_jepa import grounded_wm as gw
    from embodied_jepa import zero_shot_grasp_runtime as zgr

    rng = np.random.default_rng(1)
    root, rs = rng.normal(size=(16, gw.K)), rng.normal(size=28)
    goal, gs = rng.normal(size=(16, gw.K)), rng.normal(size=28)
    seqs = rng.uniform(-1, 1, (3, 8, 14)).astype(np.float32)
    t = lr.rollout_terms(small_pm, root, rs, goal, gs, seqs)
    assert t["total"].shape == (3, 8) and t["hand_joints"].shape == (3, 8, 7)
    assert np.allclose(t["latent"] + t["pose"], t["total"], atol=1e-5)
    assert np.allclose(t["arm"] + t["hand"], t["pose"], atol=1e-6)
    assert np.allclose(t["total"], zgr.rollout_costs(small_pm, root, rs, goal, gs, seqs), atol=0)
    sm = lr.step_mean(t)
    model, _ = zgr.wrap(small_pm, "step-mean")
    lat, g = model.encode(root, rs), model.encode_goal(goal, gs)
    d = model.distance(model.predict(lat, seqs[None]), g)[0]
    assert np.allclose(sm["total"], d[:, -1], atol=1e-6)  # the planner's step-averaged cost
    z = rng.normal(size=(3, 8, 16, gw.K)).astype(np.float32)
    s = rng.normal(size=(3, 8, 28)).astype(np.float32)
    c = lr.ceiling_terms(small_pm, z, s, goal, gs)
    for i in range(3):
        ref = zgr.ceiling_costs(small_pm, z[i], s[i], goal, gs)
        assert np.allclose(c["total"][i], ref, atol=1e-6)


def test_search_plan_is_the_a2_call(small_pm):
    from embodied_jepa import grasp_closure_locus_runtime as lr
    from embodied_jepa import grasp_corpus_wm_planned_close_runtime as pr
    from embodied_jepa import grounded_wm as gw

    rng = np.random.default_rng(2)
    xin = (rng.normal(size=(16, gw.K)), rng.normal(size=28), rng.normal(size=(16, gw.K)),
           rng.normal(size=28))  # fmt: skip
    plan, cost, req = lr.search_plan(small_pm, 940000, *xin)
    ref = pr.a2_plan(small_pm, gl.SETTING, 940000, *xin)
    assert rp.bits_equal(plan, ref) and rp.bits_equal(req, plan)
    assert np.isfinite(cost)
    again, cost2, _ = lr.search_plan(small_pm, 940000, *xin)
    assert rp.bits_equal(again, plan) and cost2 == cost
    assert np.all(plan[:, :6] == 0) and np.all(plan[:, 12] == -1)


def test_substitution_costs_score_both_plans_as_cem_does(small_pm):
    """S's comparator (R27.183 (4)): the plan's and the substituted plan's step-averaged costs, on
    requests, equal the planner's step-mean cost of each alone."""
    from embodied_jepa import grasp_closure_locus_runtime as lr
    from embodied_jepa import grounded_wm as gw
    from embodied_jepa import zero_shot_grasp_runtime as zgr

    rng = np.random.default_rng(4)
    xin = (rng.normal(size=(16, gw.K)), rng.normal(size=28), rng.normal(size=(16, gw.K)),
           rng.normal(size=28))  # fmt: skip
    plan, cost, _ = lr.search_plan(small_pm, 940001, *xin)
    model, _ = zgr.wrap(small_pm, gl.SETTING)
    lat, goal = model.encode(*xin[:2]), model.encode_goal(*xin[2:])
    for root in gl.ROOT_TYPES:
        out = lr.substitution_costs(small_pm, *xin, plan, root)
        for name, seq in (("plan", plan), ("substituted", gl.substituted_plan(plan, root))):
            ref = model.distance(model.predict(lat, seq[None, None]), goal)[0, 0, -1]
            assert out[name]["total"] == pytest.approx(float(ref), rel=1e-5, abs=1e-6)
            assert out[name]["latent"] + out[name]["pose"] == pytest.approx(
                out[name]["total"], rel=1e-5)  # fmt: skip
        assert out["plan"]["total"] == pytest.approx(cost, rel=1e-4, abs=1e-5)


# ----- seeds, pins, guards, import -------------------------------------------------------------
def test_seed_ranges_sizes_and_disjointness():
    c = gl.COHORTS
    assert list(c["dev-k"]) == list(range(94100, 94132))
    assert list(c["dev-roots"]) == list(range(940000, 940032))
    assert list(c["gated-k"]) == list(range(941000, 941128)) and gl.GATED_K_N == 128
    assert list(c["gated-roots"]) == list(range(946000, 946128)) and gl.GATED_ROOTS_N == 128
    assert gl.RETIRED_ROOTS == range(942000, 942064)  # R27.183 (1): never generated or run
    assert set(c["debug-k"]) | set(c["debug-roots"]) == set(gl.DEBUG_SEEDS)
    assert gl.DEBUG_SEEDS == range(94500, 94600)
    names = list(c)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not set(c[a]) & set(c[b]), (a, b)
        for k, r in gl.EARLIER_RANGES.items():
            assert not set(c[a]) & set(r), (a, k)
    assert gl.SALTS == {"bootstrap": 9014, "reserved": 9015}
    for salts in (zs.SALTS, rp.SALTS):
        assert not {9014, 9015} & set(salts.values())
    for k, r in c.items():
        assert not set(r) & set(gl.RETIRED_ROOTS), k
    gl.check_seed("gated-roots", 946000)
    gl.check_seed("gated-roots", 946127)
    for bad in (946128, 945999):
        with pytest.raises(ContractError):
            gl.check_seed("gated-roots", bad)
    for retired in (942000, 942063):  # the retired range is refused under every cohort name
        for cohort in list(c) + ["retired"]:
            with pytest.raises(ContractError, match="retired"):
                gl.check_seed(cohort, retired)
    runner = _runner()
    assert runner.CHECK_RANGES["gated-roots"] == range(946000, 946128)
    assert runner.CHECK_RANGES["retired-roots"] == gl.RETIRED_ROOTS
    for token in ("946000", "946127", "942000", "940031", "941127"):
        assert runner.SEED_PATTERN.search(f" {token} ") and runner._in_ranges(int(token))


def test_frozen_files_match_their_pins():
    check = gl.frozen_files_check(ROOT)
    assert all(check.values()), [k for k, v in check.items() if not v]
    src = "src/embodied_jepa/"
    for rel in (src + "embodiment.py", src + "planning.py", src + "zero_shot.py",
                src + "zero_shot_runtime.py", src + "zero_shot_grasp.py",
                src + "zero_shot_grasp_runtime.py", src + "grasp_rate_projected.py",
                "scripts/run_task093.py", "configs/g1_sim_action.json"):  # fmt: skip
        assert rel in gl.FROZEN_FILES_SHA256
    assert set(gl.CHECKPOINT_SHA256) == {"G-v1-87100", "G-v1-87101", "G-v1-87102"}
    assert gl.LAMBDA == 2.0552 and gl.JOB_CAP_SECONDS == 1800


def test_core_import_stays_light():
    code = (
        "import sys, embodied_jepa, embodied_jepa.grasp_closure_locus;"
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def _runner():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import run_task094 as runner
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    return runner


def test_runner_refuses_another_working_directory(tmp_path, monkeypatch):
    runner = _runner()
    from embodied_jepa import run_tools as rt

    monkeypatch.chdir(tmp_path)
    with pytest.raises(rt.GuardError, match="G-cwd"):
        runner.tree_guard()
    monkeypatch.chdir(ROOT)
    if LEWM.is_dir():
        assert runner.tree_guard()["cwd"] == str(ROOT.resolve())


def test_summarizer_needs_stage_0_for_the_gated_stage(tmp_path):
    script = ROOT / "scripts" / "summarize_task094.py"
    for argv in (["--stage", "gated", "--runs", str(tmp_path)],
                 ["--stage", "0", "--runs", str(tmp_path)]):  # fmt: skip
        res = subprocess.run([sys.executable, str(script), *argv], capture_output=True, text=True)
        assert res.returncode != 0


class _Claimed(Exception):
    pass


def _record_order(runner, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "base_report", lambda *a, **k: {"revision": "r"})
    monkeypatch.setattr(runner, "_lambda_check", lambda: None)
    monkeypatch.setattr(
        runner, "_load_model", lambda s: calls.append(f"model-{s}") or {"checkpoint_sha256": "x"}
    )
    monkeypatch.setattr(runner, "_load_encoder", lambda: calls.append("encoder") or {})

    def claim(key):
        calls.append(f"ledger:{key}")
        raise _Claimed

    monkeypatch.setattr(runner, "ledger_claim", claim)
    return calls


@pytest.mark.parametrize("command", ["rank", "search", "substitute"])
def test_rank_and_search_claim_the_ledger_after_roots_and_model_loads(monkeypatch, tmp_path,
                                                                      command):  # fmt: skip
    """R27.107's class of defect: a guard or a load failing after the claim would spend the
    cohort. The claim must come after the roots, the G checkpoints and the encoder."""
    runner = _runner()
    calls = _record_order(runner, monkeypatch)
    monkeypatch.setattr(runner, "load_roots", lambda *a, **k: calls.append("roots") or (
        {"roots": [], "branches": []}, {}, "s"))  # fmt: skip
    cohort = "dev-roots" if command == "substitute" else "gated-roots"
    search = tmp_path / "search.json"
    search.write_text(json.dumps({"roots_sha256": "s", "completed": True, "task": gl.TASK,
                                  "cohort": cohort}))  # fmt: skip
    args = runner.main.__globals__["argparse"].Namespace(
        cohort=cohort, roots=str(tmp_path), output=str(tmp_path), limit=None,
        cap_seconds=None, check_a2=False, search=str(search))  # fmt: skip
    with pytest.raises(_Claimed):
        getattr(runner, f"cmd_{command}")(args)
    claim = calls.index(f"ledger:{command}-{cohort}")
    assert calls[-1] == f"ledger:{command}-{cohort}"
    for need in ("roots", "encoder", "model-87100", "model-87101", "model-87102"):
        assert calls.index(need) < claim, calls


def test_k_run_claims_the_ledger_after_goals(monkeypatch, tmp_path):
    runner = _runner()
    calls = _record_order(runner, monkeypatch)
    monkeypatch.setattr(runner, "load_goals", lambda *a, **k: calls.append("goals") or (
        [{"seed": s} for s in gl.COHORTS["gated-k"]], "s"))  # fmt: skip
    args = runner.main.__globals__["argparse"].Namespace(
        cohort="gated-k", arm="K-sstar", goals=str(tmp_path), output=str(tmp_path), workers=8,
        limit=None, determinism=False, suffix=None, cap_seconds=None)  # fmt: skip
    with pytest.raises(_Claimed):
        runner.cmd_k_run(args)
    assert calls == ["goals", "ledger:k-gated-k-K-sstar"]
    from embodied_jepa import run_tools as rt

    args.arm = "ik"  # TASK-088's frozen follower is a Stage 0 check only, never a gated arm
    with pytest.raises(rt.GuardError):
        runner.cmd_k_run(args)
