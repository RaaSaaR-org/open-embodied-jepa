"""TASK-089 (``zero_shot_grasp*``): cohorts, branches, A1's wins, bars and pass rule, the shuffled
deal, A2 and the selection rule, rows, Part B's bar and arms, the candidate changes' contract.
Synthetic contract evidence only; the simulator test is graphics opt-in (``JEPA_TEST_RENDER=1``)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import zero_shot as zs
from embodied_jepa import zero_shot_grasp as zg
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
LEWM = ROOT / "third_party" / "le-wm"
render = pytest.mark.skipif(os.environ.get("JEPA_TEST_RENDER") != "1", reason="graphics opt-in")
NAMES = [zg.comparison_name(c) for c in zg.COMPARISONS]


def test_cohorts_disjoint_fresh_and_sized():
    seen = set()
    for name, seeds in zg.COHORTS.items():
        assert not (seen & set(seeds)), name
        seen |= set(seeds)
    for seeds in zs.COHORTS.values():  # never a TASK-088 seed
        assert not (seen & set(seeds))
    assert not seen & set(range(850000, 853200)) and not seen & set(range(85000, 88000))
    assert len(zg.COHORTS["a-roots"]) == 64 and len(zg.COHORTS["bs-grasp"]) == 64
    assert len(zg.COHORTS["k0-roots"]) == 32 and len(zg.COHORTS["bd-grasp"]) == 32
    with pytest.raises(ContractError):
        zg.check_seed("a-roots", 89000)
    assert zg.SALTS["shuffle"] == 8901 and 8901 not in zs.SALTS.values()


def test_branches_and_goals():
    assert set(zg.BRANCHES) == set(zg.ROOTS)
    for root, true in zg.TRUE_BRANCH.items():
        assert list(zg.BRANCHES[root])[0] == true
    assert zg.BRANCHES["R1"]["close"][1] == 1.0 and zg.BRANCHES["R1"]["open"][1] == -1.0
    assert zg.BRANCHES["R2"]["release"] == ("lift", -1.0)
    assert zg.BRANCHES["R2"]["stay"] == ("aim", 1.0)
    assert zs.SUBGOALS[zg.ROOT_GOAL["R1"]] == "grasp" and zs.SUBGOALS[zg.ROOT_GOAL["R2"]] == "lift"
    assert zs.SUBGOALS[zg.ROOT_STATE_SUBGOAL["R1"]] == "pregrasp"
    for root, t, a in zg.COMPARISONS:
        assert t == zg.TRUE_BRANCH[root] and a in zg.BRANCHES[root] and a != t


def test_wins_are_strict():
    assert list(zg.wins([1, 2, 3], [2, 2, 1])) == [True, False, False]


def test_shuffled_deal_is_a_seeded_partition():
    t = np.arange(16 * 14, dtype=np.float32).reshape(16, 14)
    a = -t - 1
    d1, d2 = zg.shuffled_deal(890000, 1, t, a)
    pool = np.concatenate([t, a])
    got = np.concatenate([d1, d2])
    assert d1.shape == d2.shape == (16, 14)
    assert sorted(map(tuple, got)) == sorted(map(tuple, pool))
    e1, _ = zg.shuffled_deal(890000, 1, t, a)
    assert np.array_equal(d1, e1)
    f1, _ = zg.shuffled_deal(890000, 2, t, a)
    assert not np.array_equal(d1, f1)  # keyed by the comparison
    with pytest.raises(ContractError):
        zg.shuffled_deal(1, 3, t, a)


def _bars(rates):
    return zg.k0_bars({"latent+pose": rates, "latent": rates})


def test_bar_rule_and_gating():
    assert zg.a1_bar(1.0) == pytest.approx(0.95)
    assert zg.a1_bar(0.75) == pytest.approx(0.725)
    assert zg.gated(0.75) and not zg.gated(0.74)
    b = _bars({"8": dict.fromkeys(NAMES, 1.0), "16": dict.fromkeys(NAMES, 0.5)})
    assert b["latent+pose"]["8"][NAMES[0]]["bar"] == pytest.approx(0.95)
    assert b["latent+pose"]["16"][NAMES[0]]["bar"] is None
    assert zg.bar_count(0.95, 64) == 61


def test_qualifying_and_ceiling_row():
    good = dict.fromkeys(NAMES, 1.0)
    only_close = {NAMES[0]: 1.0, NAMES[1]: 0.5, NAMES[2]: 0.6}
    only_r2 = {NAMES[0]: 0.5, NAMES[1]: 1.0, NAMES[2]: 1.0}
    assert not zg.ceiling_row(_bars({"8": good, "16": only_r2}))
    # B1 of #197: close gated only at 8 and R2 only at 16 -> no horizon qualifies
    b = _bars({"8": only_close, "16": only_r2})
    assert not zg.qualifying(b, 8) and not zg.qualifying(b, 16) and zg.ceiling_row(b)
    one_r2 = {NAMES[0]: 1.0, NAMES[1]: 0.5, NAMES[2]: 0.9}
    assert zg.qualifying(_bars({"8": one_r2, "16": one_r2}), 8)


def test_a1_pass_rule():
    n = 64
    fam = _bars({"8": {NAMES[0]: 1.0, NAMES[1]: 0.5, NAMES[2]: 1.0}, "16": {}})["latent+pose"]["8"]
    full = np.ones(n, bool)
    half = np.arange(n) % 2 == 0
    model = {NAMES[0]: full, NAMES[1]: half, NAMES[2]: full}
    shuffled = dict.fromkeys(NAMES, half)
    out = zg.a1_pass(fam, model, shuffled, n)
    assert out["pass"] and out["usable"]
    assert not out["comparisons"][NAMES[1]]["gated"]  # ungated: reported only
    # one gated comparison below its bar fails the pass
    m2 = dict(model)
    m2[NAMES[2]] = np.arange(n) < 60
    assert not zg.a1_pass(fam, m2, shuffled, n)["pass"]
    # meeting the count but not beating the shuffled control fails too
    assert not zg.a1_pass(fam, model, dict.fromkeys(NAMES, full), n)["pass"]


def test_a2_measures_and_bars():
    r1 = np.zeros((4, 14))
    r1[:2, 13] = 0.9
    r2 = np.zeros((4, 14))
    r2[:, 13] = 0.9
    r2[0, 8] = -0.1
    r2[1:, 8] = 0.2
    m = zg.a2_measures(r1, r2)
    assert list(m["close_R1"]) == [True, True, False, False]
    assert list(m["lift_R2"]) == [False, True, True, True]
    n = 64
    base = {"close_R1": np.zeros(n, bool), "lift_R2": np.zeros(n, bool)}
    good = {"close_R1": np.arange(n) < 50, "lift_R2": np.arange(n) < 48}
    assert zg.a2_meets(good, base)["meets"]
    weak = {"close_R1": np.arange(n) < 50, "lift_R2": np.arange(n) < 47}
    assert not zg.a2_meets(weak, base)["meets"]
    # 48 but not above the base on the same roots
    assert not zg.a2_meets(good, {k: v.copy() for k, v in good.items()})["meets"]


def test_selection_order_and_rows():
    yes, no = {"meets": True}, {"meets": False}
    a2 = {"step-mean": yes, "h16": yes, "held": yes}
    assert zg.select_change({8: True, 16: True}, a2) == "step-mean"
    assert zg.select_change({8: False, 16: True}, a2) == "h16"
    assert zg.select_change({8: True, 16: False}, {"step-mean": no, "h16": yes, "held": yes}) == (
        "held"
    )
    assert zg.select_change({8: True, 16: True}, dict.fromkeys(a2, no)) is None
    assert zg.a_row(void="x", a1_by_h={8: True}, change="held") == "A-VOID"
    assert zg.a_row(ceiling=True, a1_by_h={8: False}, change=None) == "A-CEILING"
    assert zg.a_row(a1_by_h={8: False, 16: False}, change=None) == "A-FAIL"
    assert zg.a_row(a1_by_h={8: True, 16: False}, change=None) == "A-PLAN"
    assert zg.a_row(a1_by_h={8: True, 16: False}, change="step-mean") == "A-PASS(step-mean)"
    assert zg.clause_fires("A-FAIL") and zg.clause_fires("B-FAIL")
    assert not zg.clause_fires("A-PLAN") and not zg.clause_fires("A-CEILING")


def test_part_b_bar_and_rows():
    assert zg.b_bar(24, 32) == pytest.approx(0.40)
    assert zg.b_bar(14, 32) == pytest.approx(0.9 * 14 / 32)  # binds below 0.445
    assert zg.b_bar(15, 32) == pytest.approx(0.40)
    assert zg.b_bar(9, 32) is None  # escalate below 0.30
    assert zg.b_bar(10, 32) is not None
    n = 64
    rnd = np.zeros(n, bool)
    assert zg.b_row(np.arange(n) < 26, rnd, 0.40)["row"] == "B-PASS"
    assert zg.b_row(np.arange(n) < 25, rnd, 0.40)["row"] == "B-LOW"
    assert zg.b_row(np.arange(n) < 7, rnd, 0.40)["row"] == "B-LOW"  # 7-0: p = 0.0078
    assert zg.b_row(np.arange(n) < 6, rnd, 0.40)["row"] == "B-FAIL"  # 6-0: p = 0.016
    assert zg.b_row(np.arange(n) < 40, rnd, 0.40, void="x")["row"] == "B-VOID"


def test_part_b_runs_admitted():
    zg.check_run("bs-grasp", "G-chg", 87100)
    zg.check_run("bs-grasp", "G-chg", 87101)
    zg.check_run("bs-grasp", "ik", None)
    for bad in (
        ("bd-grasp", "G-chg", 87101),
        ("bs-grasp", "P-chg", 87101),
        ("a-roots", "G-chg", 87100),
        ("bs-grasp", "ik", 87100),
        ("bs-grasp", "G", 87100),
    ):
        with pytest.raises(ContractError):
            zg.check_run(*bad)


def test_core_import_stays_light():
    code = (
        "import sys, embodied_jepa, embodied_jepa.zero_shot_grasp;"
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


# ----- the changes' contract (torch) -------------------------------------------------------------
@pytest.fixture
def small_pm():
    torch = pytest.importorskip("torch")
    if not (LEWM / "jepa.py").exists():
        pytest.skip("pinned LeWM source not fetched")
    os.chdir(ROOT)
    from embodied_jepa import grounded_wm_model as gm
    from embodied_jepa import zero_shot_runtime as zr

    small = {
        "depth": 1,
        "heads": 2,
        "head_dim": 16,
        "mlp_dim": 32,
        "proj_hidden": 32,
        "state_hidden": 16,
        "inverse_hidden": 16,
        "readout_hidden": 16,
    }
    torch.manual_seed(0)
    m = gm.GroundedWM("G", config=small)
    m.set_moments(np.zeros(28), np.ones(28), np.zeros(3), np.ones(3))
    m.eval()
    return zr.PlannerModel(m, w_lat=1.0, w_pose=1.0, lam=2.0, device="cpu")


def test_step_mean_and_held_changes(small_pm):
    from embodied_jepa import grounded_wm as gw
    from embodied_jepa import zero_shot_grasp_runtime as zgr

    rng = np.random.default_rng(0)
    latent = small_pm.encode(rng.normal(size=(16, gw.K)), rng.normal(size=28))
    goal = small_pm.encode_goal(rng.normal(size=(16, gw.K)), rng.normal(size=28))
    cand = rng.uniform(-1, 1, (1, 5, 8, 14)).astype(np.float32)
    base = small_pm.distance(small_pm.predict(latent, cand), goal)
    sm, h = zgr.wrap(small_pm, "step-mean")
    assert h == 8
    d = sm.distance(sm.predict(latent, cand), goal)
    assert np.allclose(d[..., -1], base.mean(-1), atol=1e-6)
    held, h = zgr.wrap(small_pm, "held")
    d_held = held.distance(held.predict(latent, cand), goal)
    same = cand.copy()
    same[..., 13] = same[..., :1, 13]
    assert np.allclose(d_held, small_pm.distance(small_pm.predict(latent, same), goal), atol=1e-6)
    assert zgr.wrap(small_pm, "h16")[1] == 16 and zgr.wrap(small_pm, "base") == (small_pm, 8)
    with pytest.raises(ContractError):
        zgr.wrap(small_pm, "longer")
    # the planner runs with every setting; the left arm stays fixed
    for setting in zg.SETTINGS:
        model, horizon = zgr.wrap(small_pm, setting)
        planner = zgr.make_planner(1, horizon)
        plan = planner.plan(model, latent, goal)
        a = plan.actions[0, 0]
        assert np.all(a[:6] == 0) and a[12] == -1
    # the simulator ceiling uses the same cost on real latents
    z = rng.normal(size=(16, 16, gw.K)).astype(np.float32)
    s = rng.normal(size=(16, 28)).astype(np.float32)
    c = zgr.ceiling_costs(small_pm, z, s, rng.normal(size=(16, gw.K)), rng.normal(size=28))
    assert c.shape == (16,) and np.isfinite(c).all()


# ----- simulator (graphics opt-in) ---------------------------------------------------------------
@render
def test_root_reproduces_the_demonstrator():
    from embodied_jepa import play_corpus as pc

    robot = pc.make_play_robot()
    try:
        r = zg.build_root(robot, 89949)
        g = zs.make_grasp_goal(robot, 89949)
    finally:
        robot.close()
    assert r["attempt"] == g["attempt"] and r["layout"] == g["layout"]
    for k in range(4):  # same subgoal frames and states as TASK-088's demonstrator
        assert np.array_equal(r["subgoals"][k]["state28"], g["subgoals"][k]["state28"])
    for key, b in r["branches"].items():
        assert b["actions"].shape == (16, 14) and b["frames"].shape == (16, 112, 112, 3), key
    assert max(r["branches"]["R2/lift"]["rise"]) > max(r["branches"]["R2/stay"]["rise"])


def test_frozen_k0_bars():
    """The Stage 0 record (§16): every comparison is gated for G's cost at both horizons; P's latent
    cost leaves lift-vs-release ungated at h = 8; both horizons qualify."""
    b = zg.A1_BARS
    assert all(v["gated"] for h in ("8", "16") for v in b["latent+pose"][h].values())
    assert not b["latent"]["8"][NAMES[1]]["gated"] and b["latent"]["16"][NAMES[1]]["gated"]
    assert zg.qualifying(b, 8) and zg.qualifying(b, 16) and not zg.ceiling_row(b)
    assert zg.bar_count(b["latent+pose"]["8"][NAMES[0]]["bar"], 64) == 61
    assert zg.bar_count(b["latent+pose"]["8"][NAMES[2]]["bar"], 64) == 59
