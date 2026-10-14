"""TASK-093 (docs/experiments/grasp_rate_projected_planner_v1.md): the grasp-rate projector
against the embodiment's own clip (bitwise, zero and non-zero arm deltas; ``project_candidates``
itself with ``JEPA_TEST_RENDER=1``), the cumulative clip on synthetic sequences, the planner's
existing hook (an identity projector reproduces the unprojected plan bitwise; both arms draw the
same first-iteration samples), Stage P's measures, references, bar rule, degeneracy checks and
rows, the B rows against TASK-090's, the closed-loop equivalence measure, the seed ranges, the pins
and the core-import check."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import grasp_corpus_wm as gc
from embodied_jepa import grasp_rate_projected as rp
from embodied_jepa import zero_shot as zs
from embodied_jepa.constraints import CandidateProjection
from embodied_jepa.contracts import ContractError
from embodied_jepa.planning import CEMConfig, CEMPlanner

ROOT = Path(__file__).resolve().parents[1]
L = 2.0 / 11.0
render = pytest.mark.skipif(os.environ.get("JEPA_TEST_RENDER") != "1", reason="graphics opt-in")


def _manifest():
    return json.loads((ROOT / "configs" / "g1_sim_action.json").read_text())


def _synthetic_geometry():
    """The manifest's right-hand synergy with wide synthetic MJCF ranges (no MuJoCo)."""
    m = _manifest()
    opened = np.asarray(m["right_open_rad"])
    closed = np.asarray(m["right_closed_rad"])
    lo = np.minimum(opened, closed) - 0.5
    hi = np.maximum(opened, closed) + 0.5
    physical = np.stack([lo, hi], 1)
    return rp.RateGeometry(opened, closed, m["joint_speed_limit_rad_s"] * 0.05, physical,
                           physical.astype(np.float32))  # fmt: skip


@pytest.fixture(scope="module")
def robot():
    """The play robot without its renderer (``_prepare_side`` needs no image)."""
    pytest.importorskip("mujoco")
    from embodied_jepa import play_corpus as pc
    from embodied_jepa.embodiment import G1Embodiment

    sim = pc._play_simulation_class()(width=112, height=112)
    r = G1Embodiment(sim)
    r.reset(0, layout=zs.sample_layout("grasp", 91940, 0))
    yield r
    r.close()


# ----- the projector against the embodiment (§14) ----------------------------------------------
def _embodiment_path(robot, req, t0_targets):
    """``project_candidates``' inner loop at factor 1 (the right side's ``_prepare_side`` step by
    step from the given targets, the scratch pose following the accepted targets); returns the
    applied right grasps until an IK refusal."""
    data = robot.mj.MjData(robot.model)
    data.qpos[:] = robot.sim.data.qpos
    robot._forward_kinematics(data)
    targets = t0_targets.copy()
    out = []
    for k in range(len(req)):
        a = req[k].copy()
        try:
            robot._prepare_side(a, targets, 1, "right", data=data)
        except ContractError:
            break
        out.append(a[13])
        data.qpos[robot.sim.qadr] = targets
        robot._forward_kinematics(data)
    return np.asarray(out, np.float32)


@pytest.mark.parametrize("nonzero", [False, True])
def test_projector_equals_the_embodiments_grasp_clip_bitwise(robot, nonzero):
    geo = rp.RateGeometry.from_robot(robot)
    assert geo.describe()["rate_limit_steepest"] == pytest.approx(L, abs=1e-12)
    idx = rp.hand_indices(robot)
    rng = np.random.default_rng([91940, int(nonzero)])
    compared = clipped = 0
    for _ in range(80):
        t0 = robot.sim.targets.copy()
        t0[idx] = geo.targets(rng.uniform(-1, 1))
        req = np.zeros((8, 14), np.float32)
        req[:, 12] = -1
        req[:, 13] = np.clip(rng.normal(0, 1, 8), -1, 1)
        if nonzero:
            req[:, 6:12] = np.clip(rng.normal(0, 0.15, (8, 6)), -1, 1)
        emb = _embodiment_path(robot, req, t0)
        mine, g = rp.project_grasp(geo, req, t0[idx].astype(np.float64))
        n = len(emb)
        if n == 0:  # the first arm step refused (IK or the joint rate): nothing to compare
            continue
        assert rp.bits_equal(mine[:n, 13], emb)
        assert rp.bits_equal(np.delete(mine, 13, 1), np.delete(req, 13, 1))
        assert np.array_equal(g.astype(np.float32), mine[:, 13])
        compared += n
        clipped += int((mine[:n, 13] != req[:n, 13]).sum())
    assert compared > 300 and clipped > 100  # the test is not trivial


@render
def test_projector_equals_project_candidates_bitwise():
    from embodied_jepa import play_corpus as pc

    robot = pc.make_play_robot()
    try:
        zs.reset(robot, zs.sample_layout("grasp", 91941, 0))
        geo = rp.RateGeometry.from_robot(robot)
        rng = np.random.default_rng(91941)
        compared = 0
        for step in range(12):
            robot.observe()
            for nonzero in (False, True):
                req = np.zeros((1, 8, 8, 14), np.float32)
                req[..., 12] = -1
                req[..., 13] = np.clip(rng.normal(0, 1, (8, 8)), -1, 1)
                if nonzero:
                    req[..., 6:12] = np.clip(rng.normal(0, 0.4, (8, 8, 6)), -1, 1)
                emb = robot.project_candidates(req)
                mine = rp.GraspRateProjector(geo, rp.read_t0(robot))(req)
                f = emb.feasible[0]
                assert rp.bits_equal(mine.actions[0, f, :, 13], emb.actions[0, f, :, 13])
                compared += int(f.sum())
            req1 = zs.HOLD.copy()
            req1[13] = 1.0 if step < 6 else -1.0
            zs.execute(robot, req1)
        assert compared > 0
    finally:
        robot.close()


def test_read_t0_requires_an_unchanged_observation(robot):
    with pytest.raises(ContractError):
        rp.read_t0(robot)  # no observation yet (robot built without rendering)


# ----- the cumulative clip on synthetic sequences (§4.1) ---------------------------------------
def test_cumulative_clip_on_synthetic_sequences():
    geo = _synthetic_geometry()
    t0 = rp.root_t0(geo, -1.0)
    req = np.zeros((3, 8, 14), np.float32)
    req[0, :, 13] = 1.0  # saturated close
    req[1, :, 13] = [1, -1, 1, -1, 1, -1, 1, -1]
    req[2, :, 13] = [-0.9, -0.85, 0.2, 0.25, 0.3, 0.3, 0.3, 0.3]
    out, g = rp.project_grasp(geo, req, t0)
    expect = -1 + L * np.arange(1, 9)
    assert np.allclose(g[0], expect, atol=1e-6)  # the maximum rate from the open hand
    assert g[0, 7] >= rp.C8_THRESHOLD
    prev = np.concatenate([np.full((3, 1), -1.0), g[:, :-1]], 1)
    assert np.all(np.abs(g - prev) <= L + 1e-6)  # around the previous *projected* step
    assert np.allclose(g[1], [-1 + L, -1, -1 + L, -1, -1 + L, -1, -1 + L, -1], atol=1e-6)
    # inside the interval the request passes unchanged
    assert out[2, 0, 13] == np.float32(-0.9) and out[2, 1, 13] == np.float32(-0.85)
    # every other channel bitwise; a projected plan is its own projection
    assert rp.bits_equal(np.delete(out, 13, -1), np.delete(req, 13, -1))
    again, _ = rp.project_grasp(geo, out, t0)
    assert rp.bits_equal(again, out)
    with pytest.raises(ContractError):
        rp.project_grasp(geo, req.astype(np.float64), t0)


def test_projector_hook_contract():
    geo = _synthetic_geometry()
    proj = rp.GraspRateProjector(geo, rp.root_t0(geo, 1.0))
    req = np.zeros((1, 5, 8, 14), np.float32)
    req[..., 13] = -1.0
    res = proj(req)
    assert isinstance(res, CandidateProjection) and res.feasible.all()
    assert not res.actions.flags.writeable
    assert np.allclose(res.actions[0, 0, :, 13], 1 - L * np.arange(1, 9), atol=1e-6)
    assert proj.calls == 1 and proj.first_requested_sha256 is not None


def test_empty_interval_raises_as_the_embodiment():
    geo = _synthetic_geometry()
    with pytest.raises(ContractError, match="rate-feasible"):
        geo.bounds(np.full(7, 50.0))  # targets far outside the synergy's reach


# ----- the planner's existing hook: pairing (§4.2, §14) ----------------------------------------
class _Toy:
    """A deterministic toy planner model: cost = distance of the cumulative commands to a goal."""

    def __init__(self):
        self.first = None

    def predict(self, latent, candidates):
        if self.first is None:
            self.first = np.array(candidates, copy=True)
        return np.cumsum(np.asarray(candidates, np.float64), axis=2)

    def distance(self, predictions, goal):
        return np.square(predictions - goal).mean(-1).astype(np.float32)


def _cfg(**kw):
    base = CEMConfig(horizon=8, samples=40, iterations=4, elites=6, seed=zs.cem_seed(91940),
                     lower_bounds=tuple(zs.FIXED_LOWER.tolist()),
                     upper_bounds=tuple(zs.FIXED_UPPER.tolist()))  # fmt: skip
    return dataclasses.replace(base, **kw)


def test_identity_projector_reproduces_the_unprojected_plan_bitwise():
    goal = np.linspace(-0.5, 0.5, 14)

    def ident(req):
        return CandidateProjection(np.asarray(req), np.ones(req.shape[:2], bool))

    a, b = _Toy(), _Toy()
    plan_u = CEMPlanner(_cfg()).plan(a, None, goal)
    plan_i = CEMPlanner(_cfg(project_candidates=True)).plan(b, None, goal, projector=ident)
    assert rp.bits_equal(plan_u.actions, plan_i.actions)
    assert np.array_equal(plan_u.costs, plan_i.costs)
    assert rp.bits_equal(a.first, b.first)


def test_projected_planner_draws_the_unprojected_first_samples():
    geo = _synthetic_geometry()
    goal = np.linspace(-0.5, 0.5, 14)
    proj = rp.GraspRateProjector(geo, rp.root_t0(geo, -1.0))
    a, b = _Toy(), _Toy()
    CEMPlanner(_cfg()).plan(a, None, goal)
    plan_p = CEMPlanner(_cfg(project_candidates=True)).plan(b, None, goal, projector=proj)
    import hashlib

    assert proj.first_requested_sha256 == hashlib.sha256(a.first.tobytes()).hexdigest()
    # the model scored the projected candidates, and the returned plan is in the rate set
    assert not rp.bits_equal(b.first, a.first)
    assert rp.in_rate_set(geo, plan_p.actions[0], "R1")
    # the elites are refitted on the requests: the requested plan can leave the rate set
    assert plan_p.requested_actions.shape == plan_p.actions.shape


def test_runtime_planner_is_task089s_with_projection_only():
    pytest.importorskip("torch")
    from embodied_jepa import grasp_rate_projected_runtime as rr
    from embodied_jepa import zero_shot_grasp_runtime as zgr

    a = zgr.make_planner(91940, 8).config
    b = rr.projected_planner(91940, 8).config
    assert b.project_candidates and not a.project_candidates
    assert dataclasses.replace(b, project_candidates=False) == a


# ----- Stage P's measures, references, bars and rows (§6.1-§6.2) -------------------------------
def test_measures_and_references():
    geo = _synthetic_geometry()
    seeds = list(range(916000, 916016))
    refs = rp.reference_counts(geo, seeds)
    assert refs["open"] == {"C8": 0, "H8": 0}
    assert refs["closed_down"]["H8"] == 0
    assert refs["max_rate_reported"]["C8"] == len(seeds)
    assert refs["max_rate_reported"]["H8"] == 0  # its vertical command is 0, not above 0
    hold = rp.max_rate_reference()
    hold[:, 8] = 0.3
    assert rp.h8(rp.executed_path(geo, hold, "R2"), hold)
    # H8's grasp half: after three maximum-rate opening steps the hand is below 0.5
    drop = hold.copy()
    drop[2:, 13] = -1.0
    assert not rp.h8(rp.executed_path(geo, drop, "R2"), drop)
    # C8: one step below the maximum rate is enough to fail
    late = rp.max_rate_reference()
    late[0, 13] = -1.0
    assert not rp.c8(rp.executed_path(geo, late, "R1"))
    r1 = rp.random_reference(916000)
    r2 = rp.random_reference(916000)
    assert rp.bits_equal(r1["R1"], r2["R1"]) and not rp.bits_equal(r1["R1"], r1["R2"])
    assert np.all(r1["R1"][:, :6] == 0) and np.all(r1["R1"][:, 12] == -1)


def test_bar_rule_and_degeneracy_checks():
    assert rp.bar_from_control([44, 56, 53]) == 40  # ceil(0.75 x 53) = 40
    assert rp.bar_from_control([64, 64, 64]) == 48  # capped
    assert rp.bar_from_control([20, 21, 22]) == 16  # ceil(15.75)
    refs = {"open": {"C8": 0, "H8": 0}, "closed_down": {"C8": 0, "H8": 0},
            "random": {"C8": 1, "H8": 20}}  # fmt: skip
    ok = rp.degeneracy("C8", [44, 56, 53], refs)
    assert ok["pass"] and ok["bar"] == 40 and ok["random_max"] == 20
    bad_b = rp.degeneracy("H8", [44, 56, 53], refs | {"random": {"C8": 0, "H8": 21}})
    assert not bad_b["pass"] and not bad_b["b_not_by_chance"]
    bad_a = rp.degeneracy("H8", [44, 56, 53], refs | {"closed_down": {"C8": 0, "H8": 1}})
    assert not bad_a["a_can_fail"]
    bad_c = rp.degeneracy("C8", [10, 15, 30], refs)
    assert not bad_c["c_control_min"] and bad_c["control_median"] == 15
    row = rp.s0_row({"C8": ok, "H8": bad_b}, equivalence_ok=True, power_ok=True)
    assert row["row"] == "S0-ESCALATE" and row["failing_measures"] == ["H8"]
    assert rp.s0_row({"C8": ok}, equivalence_ok=False, power_ok=True)["row"] == "S0-ESCALATE"
    assert rp.s0_row({"C8": ok}, equivalence_ok=True, power_ok=False)["row"] == "S0-ESCALATE"
    assert rp.s0_row({"C8": ok}, equivalence_ok=True, power_ok=True)["row"] == "S0-PASS"


def test_p_gate_and_rows():
    bars = {"C8": 40, "H8": 20}
    assert rp.p_gate({"C8": 40, "H8": 20}, bars) and not rp.p_gate({"C8": 39, "H8": 64}, bars)
    assert rp.p_row({87100: True, 87101: False, 87102: True})["row"] == "P-PASS"
    assert rp.p_row({87100: False, 87101: False, 87102: True})["row"] == "P-FAIL"
    assert rp.p_row({87100: True, 87101: True, 87102: True}, void="x")["row"] == "P-VOID"


def test_in_rate_set_and_requests_beyond():
    geo = _synthetic_geometry()
    plan = rp.max_rate_reference()
    out, _ = rp.project_grasp(geo, plan, rp.root_t0(geo, -1.0))
    assert rp.in_rate_set(geo, out, "R1") and not rp.in_rate_set(geo, plan, "R1")
    b = rp.requests_beyond(plan, out[:, 13])
    assert b == {"steps": 8, "beyond": 8}


# ----- the closed loop (§6.3) ------------------------------------------------------------------
def test_b_rows_equal_task090s_with_renamed_rows():
    rng = np.random.default_rng(5)
    seen = set()
    for k in range(40):
        rate = [0.05, 0.15, 0.45, 0.6][k % 4]
        p = rng.random((3, 128)) < rate
        c = rng.random((3, 128)) < 0.05
        r = rng.random(128) < [0.0, 0.3][k % 2]
        mine = rp.b_row(p, c, r, 0.40)
        theirs = gc.b_row(p, c, r, 0.40)
        assert mine["row"] == rp.GC_ROW_NAMES.get(theirs["row"], theirs["row"])
        assert mine["p_beats_v1"] == theirs["p_beats_v1p"]
        assert mine["bootstrap_rate_difference"] == theirs["bootstrap_rate_difference"]
        seen.add(mine["row"])
    assert {"B-GAIN", "B-FAIL"} <= seen
    assert rp.b_row(p, c, r, 0.4, void="x")["row"] == "B-VOID"
    assert rp.pooled_bar(0.40, 128) == 156 and rp.seed_floor(0.40, 128) == 26
    assert rp.clause_fires("B-FAIL") and not rp.clause_fires("B-UNDER")


def _episode(requested, applied, index=2):
    log = [{"index": index, "grasp": [float(g)] + [0.0] * 7} for g in requested]
    a = np.zeros((len(applied), 14), np.float32)
    a[:, 13] = applied
    return {"applied": a.tolist(), "plan_log": log}


def test_closed_loop_equivalence_is_bitwise():
    g = np.float32(-1 + L)
    same = rp.closed_loop_equivalence(_episode([g, g], [g, g]))
    assert same["channel13_differs"] == 0 and same["commands"] == 2
    tiny = np.nextafter(g, np.float32(1))
    diff = rp.closed_loop_equivalence(_episode([g, tiny], [g, g]))
    assert diff["channel13_differs"] == 1 and diff["grasp_phase_channel13_differs"] == 1
    # one more plan than commands (the last request refused) is allowed
    extra = rp.closed_loop_equivalence(_episode([g, g, 1.0], [g, g]))
    assert extra["commands"] == 2 and extra["channel13_differs"] == 0
    with pytest.raises(ContractError):
        rp.closed_loop_equivalence(_episode([g], [g, g, g]))


def test_request_measures():
    ep = _episode([-1 + L], [-1 + L])
    ep["plan_log"][0]["grasp"] = [-1 + L * k for k in range(1, 9)]
    ep["plan_log"][0]["requested_grasp"] = [1.0] * 8
    m = rp.request_measures(ep)
    assert m == {"plan_steps": 8, "requested_steps_beyond_limit": 8}
    ep["plan_log"][0]["requested_grasp"] = ep["plan_log"][0]["grasp"]
    assert rp.request_measures(ep)["requested_steps_beyond_limit"] == 0


def test_power_rule_and_row_rates_plumbing():
    rows = [{"alt": [0.15, 0.05], "n": 128, "sd": 1.0, "power": 0.97}]
    assert not rp.power_rule(rows)["escalate"]
    assert rp.power_rule([dict(rows[0], power=0.79)])["escalate"]
    out = rp.row_rates((0.05, 0.05), 128, 1.0, sims=20)
    assert abs(sum(out["rates"].values()) - 1) < 1e-12
    assert set(out["rates"]) <= {"B-PASS", "B-GAIN", "B-BAR", "B-FAIL", "B-UNDER"}


# ----- seeds, runs, pins, import ---------------------------------------------------------------
def test_seed_ranges_sizes_and_disjointness():
    c = rp.COHORTS
    assert list(c["cal-roots"]) == list(range(916000, 916064))
    assert list(c["p-roots"]) == list(range(917000, 917064))
    assert list(c["bd-grasp"]) == list(range(91900, 91932))
    assert list(c["bs-grasp"]) == list(range(918000, 918128)) and rp.BS_N == 128
    assert set(c["debug-roots"]) | set(c["debug-grasp"]) == set(rp.DEBUG_SEEDS)
    names = list(c)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not set(c[a]) & set(c[b]), (a, b)
        for k, r in rp.EARLIER_RANGES.items():
            assert not set(c[a]) & set(r), (a, k)
    assert rp.SALTS["random_reference"] == 9011
    assert 9011 not in gc.SALTS.values() and 9011 not in zs.SALTS.values()
    rp.check_seed("p-roots", 917063)
    with pytest.raises(ContractError):
        rp.check_seed("p-roots", 917064)


def test_runs_admitted():
    rp.check_run("bd-grasp", "M-proj", 87100)
    rp.check_run("bs-grasp", "M-v1", 87102)
    rp.check_run("debug-grasp", "ik", None)
    for bad in (("bd-grasp", "M-proj", 87101), ("bs-grasp", "M-req", 87100),
                ("bs-grasp", "ik", 87100), ("cal-roots", "M-v1", 87100)):  # fmt: skip
        with pytest.raises(ContractError):
            rp.check_run(*bad)
    assert rp.needs_parts("bs-grasp", "M-proj") == 2 and rp.needs_parts("bd-grasp", "M-v1") is None
    assert [list(r)[0] for r in rp.part_ranges("bs-grasp", 2)] == [918000, 918064]


def test_frozen_files_match_their_pins():
    check = rp.frozen_files_check(ROOT)
    assert all(check.values()), [k for k, v in check.items() if not v]
    for rel in ("src/embodied_jepa/embodiment.py", "configs/g1_sim_action.json",
                "src/embodied_jepa/planning.py", "src/embodied_jepa/zero_shot_grasp_runtime.py",
                "src/embodied_jepa/grasp_corpus_wm_planned_close_runtime.py"):  # fmt: skip
        assert rel in rp.FROZEN_FILES_SHA256
    assert set(rp.CHECKPOINT_SHA256) == {"G-v1-87100", "G-v1-87101", "G-v1-87102"}


def test_core_import_stays_light():
    code = (
        "import sys, embodied_jepa, embodied_jepa.grasp_rate_projected;"
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_runner_refuses_another_working_directory(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import run_task093 as runner
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    from embodied_jepa import run_tools as rt

    monkeypatch.chdir(tmp_path)
    with pytest.raises(rt.GuardError, match="G-cwd"):
        runner.tree_guard()
    monkeypatch.chdir(ROOT)
    if (ROOT / "third_party" / "le-wm").is_dir():
        assert runner.tree_guard()["cwd"] == str(ROOT.resolve())


def test_summarizer_needs_prior_stages(tmp_path):
    script = ROOT / "scripts" / "summarize_task093.py"
    for argv in (["--stage", "bd", "--runs", str(tmp_path)],
                 ["--stage", "bs", "--runs", str(tmp_path), "--bd-summary", "x"],
                 ["--stage", "p", "--runs", str(tmp_path)]):  # fmt: skip
        res = subprocess.run([sys.executable, str(script), *argv], capture_output=True, text=True)
        assert res.returncode != 0
