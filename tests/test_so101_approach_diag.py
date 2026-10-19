"""TASK-102 Δa tests (``docs/experiments/so101_approach_diagnostic_v1.md``): cohorts and salts,
the line states, the cost forms, the projection chain, the candidate set, the statistics, the
rows in their declared order, the simulation copy against the real transport (G-rollout), the
recording projector, the oracle adapters, and the byte-identity of TASK-101's and TASK-102's
pinned files. CPU only; physics tests skip only when MuJoCo or the pinned SO-101 assets are
absent."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import so101_approach_diag as ad
from embodied_jepa import so101_closed_loop as cl
from embodied_jepa import so101_hybrid as hy
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]


def _fetched() -> bool:
    manifest = json.loads((ROOT / "assets/so101_manifest.json").read_text())
    root = ROOT / manifest["local_path"]
    return all((root / relative).is_file() for relative in manifest["sha256"])


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----- cohorts, seeds, salts (§1) --
def test_cohorts_are_declared_and_disjoint_from_every_earlier_range():
    L = ad.COHORTS["L"]
    assert L.seeds == tuple(range(104000, 104016))
    assert L.reserve_seeds == tuple(range(104016, 104032))
    assert ad.COHORTS["debug"].seeds == tuple(range(104900, 104910))
    mine = set(L.seeds) | set(L.reserve_seeds) | set(ad.COHORTS["debug"].seeds)
    earlier = set(range(0, 64)) | set(range(100, 164)) | set(range(1000, 1016))
    earlier |= set(range(990000, 994000)) | set(range(100100, 100103)) | set(range(100900, 100921))
    earlier |= set(range(101000, 104000))
    for c in list(cl.COHORTS.values()) + list(hy.COHORTS.values()):
        earlier |= set(c.seeds) | set(c.reserve_seeds)
    assert not mine & earlier
    assert not set(ad.SALTS.values()) & (set(cl.SALTS.values()) | set(hy.SALTS.values()))


def test_seed_guard_and_keep_rule():
    assert ad.check_seed("L", 104000) == 104000
    assert ad.check_seed("L", np.int64(104031)) == 104031
    for bad in (103999, 104032, 104900, "104000", 104000.0):
        with pytest.raises(ad.GuardError):
            ad.check_seed("L", bad)
    kept = ad.keep_cohort("L", lambda s: (s != 104003, "failed"))
    assert len(kept["kept"]) == 16 and 104003 not in kept["kept"]
    assert kept["kept"][3] == 104016 and kept["rejected"] == [
        {"seed": 104003, "slot": 3, "reason": "failed"}
    ]
    with pytest.raises(ad.GuardError):
        ad.keep_cohort("L", lambda s: (s < 104008, "x"))


def test_planner_seed_and_candidate_stream():
    assert ad.planner_seed(104005) == 10116 * 10**6 + 104005
    a = ad.candidate_rng(104000, 3).uniform(size=4)
    b = np.random.default_rng([10117, 104000, 3]).uniform(size=4)
    assert np.array_equal(a, b)


# ----- the line states and cost forms (§3) --
def test_line_points_lie_at_their_nominal_distance():
    rng = np.random.default_rng(0)
    q0, q1 = rng.normal(size=6), rng.normal(size=6)
    d0 = float(ad.arm_rms(q0, q1))
    ds = ad.line_distances(d0)
    assert ds[-1] == ad.RESET and all(d < d0 for d in ds[:-1])
    assert np.array_equal(ad.line_point(q0, q1, ad.RESET, d0), q0)
    assert np.array_equal(ad.line_point(q0, q1, 0.0, d0), q1)
    for d in ds[:-1]:
        assert math.isclose(float(ad.arm_rms(ad.line_point(q0, q1, d, d0), q1)), d, abs_tol=1e-12)
    with pytest.raises(ContractError):
        ad.line_point(q0, q1, d0 + 0.1, d0)
    short = ad.line_distances(0.3)
    assert short == (0.0, 0.02, 0.04, 0.08, 0.12, 0.16, 0.24, ad.RESET)


def test_part_errors_and_forms():
    rng = np.random.default_rng(1)
    g = rng.normal(size=cl.LATENT_DIM).astype(np.float32)
    p = rng.normal(size=(3, 5, cl.LATENT_DIM)).astype(np.float32)
    e = ad.part_errors(p, g)
    assert e.shape == (3, 5, 3)
    d = (p.astype(np.float64) - g) ** 2
    assert np.allclose(e[..., 0], d.mean(-1))
    assert np.allclose(e[..., 0], (e[..., 1] + e[..., 2]) / 2)
    s = np.arange(1.0, 17.0)
    assert ad.form_cost(s, "last", 8) == 8.0 and ad.form_cost(s, "mean", 8) == 4.5
    assert ad.form_cost(s, "last", 16) == 16.0
    with pytest.raises(ContractError):
        ad.form_cost(s, "last", 17)
    with pytest.raises(ContractError):
        ad.form_cost(s, "max", 8)


def _params():
    limits = np.array(
        [[-1.9, 1.9], [-1.7, 1.7], [-1.6, 1.6], [-1.6, 1.6], [-2.7, 2.7], [-0.17, 1.7]]
    )
    return cl.PreviewParams(limits, np.full(5, 0.05, np.float32), -0.17, 1.7, 0.1)


def test_project_chain_equals_the_chained_projector():
    params = _params()
    rng = np.random.default_rng(2)
    req = rng.uniform(-1, 1, (50, 16, 6)).astype(np.float32)
    targets = np.array([1.85, -1.68, 0.0, 1.58, 0.0, 0.5])  # near limits: clipping happens
    applied, chain = ad.project_chain(req, targets, params)
    ref, final = cl.project_sequences(req, targets, params)
    assert np.array_equal(applied, ref) and np.array_equal(chain[:, -1], final)
    assert not np.array_equal(applied, req)
    for t in (0, 7):
        _, prefix = cl.project_sequences(req[:, : t + 1], targets, params)
        assert np.array_equal(chain[:, t], prefix)


def test_candidate_set_approach_hold_and_random():
    params = _params()
    targets = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0])
    goal = np.array([0.5, -0.6, 0.9, 0.2, 0.1, 1.6])
    c = ad.candidate_set(104000, 2, targets, goal, params)
    assert c.shape == (64, 16, 6) and c.dtype == np.float32
    assert np.array_equal(c[0, 0], cl.pose_action(targets, goal, params.scales))
    assert np.all(c[0, :, 5] == 1.0)  # the goal jaw is open
    assert np.all(c[1, :, :5] == 0)
    applied, chain = ad.project_chain(c[1:2], targets, params)
    assert np.allclose(chain[0, :, 5], targets[5])  # hold keeps the gripper target
    assert np.array_equal(c, ad.candidate_set(104000, 2, targets, goal, params))
    assert not np.array_equal(c[2:], ad.candidate_set(104000, 3, targets, goal, params)[2:])
    # the pose law reaches the goal's arm joints along its own chain
    _, ch = ad.project_chain(ad.approach_sequence(targets, goal, params, 32)[None], targets, params)
    assert np.allclose(ch[0, -1, :5], goal[:5])


def test_pose_cost():
    g = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
    assert ad.pose_cost(g, g) == 0.0
    q = g.copy()
    q[:5] += 0.08
    assert math.isclose(float(ad.pose_cost(q, g)), 1.0)
    q = g.copy()
    q[5] += 0.15
    assert math.isclose(float(ad.pose_cost(q, g)), 1.0)
    assert ad.pose_cost(np.stack([g, q]), g).shape == (2,)


# ----- statistics (§3.4) --
def test_spearman():
    x = np.arange(10.0)
    assert ad.spearman(x, x) == pytest.approx(1.0)
    assert ad.spearman(x, -x) == pytest.approx(-1.0)
    assert ad.spearman(x, x**3) == pytest.approx(1.0)
    assert math.isnan(ad.spearman(x, np.ones(10)))
    assert math.isnan(ad.spearman(x[:2], x[:2]))
    y = np.array([1.0, 2.0, 2.0, 3.0])
    assert np.array_equal(ad.ranks(y), [1.0, 2.5, 2.5, 4.0])
    assert ad.spearman(np.array([1.0, 2, 3, np.nan]), np.array([1.0, 2, 3, 4])) == 1.0


def test_band_ordering_and_d_flat():
    ds = (0.0, 0.02, 0.04, 0.08, 0.12, 0.16, 0.24, 0.32, 0.40, 0.48, 0.64)
    # increasing up to 0.24, flat (noise) beyond
    rng = np.random.default_rng(3)
    per_seed = {}
    for s in range(16):
        per_seed[s] = {d: (d if d <= 0.24 else 0.24 + 0.01 * rng.normal()) for d in ds}
    bands = ad.band_ordering(per_seed)
    assert bands["0.08-0.12"]["resolved"] and bands["0.16-0.24"]["resolved"]
    assert bands["0.08-0.12"]["accuracy"] == 1.0 and bands["0.08-0.12"]["seeds"] == 16
    assert ad.d_flat(bands) == 0.24
    monotone = {s: {d: d for d in ds} for s in range(16)}
    assert ad.d_flat(ad.band_ordering(monotone)) == "> 0.64"
    # a missing state removes that seed from the band
    del monotone[0][0.12]
    assert ad.band_ordering(monotone)["0.08-0.12"]["seeds"] == 15
    low = ad.band_ordering(monotone, ad.LOW_BANDS)
    assert set(low) == {"0.00-0.02", "0.02-0.04", "0.04-0.08"}
    # exactly 12/16 is resolved, 11/16 is not
    for up, want in ((12, True), (11, False)):
        ps = {s: {0.08: 0.0, 0.12: (1.0 if s < up else -1.0)} for s in range(16)}
        assert ad.band_ordering(ps)["0.08-0.12"]["resolved"] is want


# ----- rows (§6) --
def _h(**kw):
    base = {"J-ref": 0, "J-big16": 0, "J-big32": 0, "O-img": 8, "O-pose": 8}
    return base | {k.replace("_", "-"): v for k, v in kw.items()}


def test_rows_first_match_in_declared_order():
    assert ad.row(void=True, h=_h(), delta_rho=1.0)["row"] == "Δa-VOID"
    assert ad.row(void=False, h=_h(J_ref=4), delta_rho=1.0)["row"] == "Δa-NOT-REPRODUCED"
    assert ad.row(void=False, h=_h(J_ref=3, O_pose=5), delta_rho=1.0)["row"] == "Δa-PLANNER-BOUND"
    r = ad.row(void=False, h=_h(O_pose=5, O_img=0), delta_rho=1.0)
    assert r["row"] == "Δa-PLANNER-BOUND" and "Δa-GEOMETRY" in r["fired"]
    assert ad.row(void=False, h=_h(O_img=5), delta_rho=1.0)["row"] == "Δa-GEOMETRY"
    assert ad.row(void=False, h=_h(O_img=6, J_big32=6), delta_rho=1.0)["row"] == "Δa-PLANNER"
    assert ad.row(void=False, h=_h(J_big16=6), delta_rho=0.0)["row"] == "Δa-PLANNER"
    assert ad.row(void=False, h=_h(J_big16=2), delta_rho=0.20)["row"] == "Δa-MODEL"
    assert ad.row(void=False, h=_h(J_big16=2), delta_rho=0.19)["row"] == "Δa-MIXED"
    assert ad.row(void=False, h=_h(J_big16=3), delta_rho=0.9)["row"] == "Δa-MIXED"
    assert ad.row(void=False, h=_h(), delta_rho=math.nan)["row"] == "Δa-MIXED"
    with pytest.raises(ContractError):
        ad.row(void=False, h={"J-ref": 0}, delta_rho=0.0)
    with pytest.raises(ContractError):
        ad.row(void=False, h=_h(O_img=9), delta_rho=0.0)


def test_arms_are_declared():
    assert set(ad.ARMS) == {"J-ref", "J-big16", "J-big32", "O-img", "O-pose"}
    ref = ad.ARMS["J-ref"]
    assert (ref["samples"], ref["iterations"], ref["elites"]) == (
        cl.CEM["samples"],
        cl.CEM["iterations"],
        cl.CEM["elites"],
    )
    assert ad.ARMS["J-big32"]["horizon"] == 32 and ad.ARMS["O-img"]["horizon"] == 16
    for arm in ad.ARMS.values():
        assert arm["elites"] * 8 == arm["samples"]
    with pytest.raises(ContractError):
        ad.check_arm("O-other")


# ----- the simulation copy and the oracles (physics) --
@pytest.fixture
def sim_ok():
    pytest.importorskip("mujoco")
    if not _fetched():
        pytest.skip("pinned SO-101 assets have not been fetched")


def test_scratch_rollout_equals_the_transport(sim_ok):
    from embodied_jepa import so101_approach_diag_runtime as ar
    from embodied_jepa import so101_closed_loop_runtime as rt

    robot = rt.make_robot(False)
    try:
        task = rt.lift_task(robot)
        task.reset(104900)
        sim = robot.sim
        params = cl.PreviewParams.of(robot)
        scratch = ar.Scratch(sim, render=False)
        snap = scratch.snapshot(sim)
        req = np.random.default_rng(4).uniform(-1, 1, (12, 6)).astype(np.float32)
        _, chain = ad.project_chain(req[None], sim.targets, params)
        copy = scratch.run(snap, chain[0])
        real = []
        for step in req:
            robot.observe()
            r = robot.execute(step)
            assert r.status != "stopped"
            real.append(sim.data.qpos[sim.qadr].copy())
        assert np.array_equal(np.asarray(real), copy["q"])
        assert np.array_equal(scratch.data.qpos, sim.data.qpos)
        assert np.array_equal(scratch.data.qvel, sim.data.qvel)
        # the copy restored again reproduces itself, and leaves the real simulation untouched
        before = sim.data.qpos.copy()
        again = scratch.run(snap, chain[0])
        assert np.array_equal(again["q"], copy["q"]) and np.array_equal(sim.data.qpos, before)
        # the single-command G-rollout form of the episode loop
        pre = scratch.snapshot(sim)
        robot.observe()
        robot.execute(np.zeros(6, np.float32))
        scratch.restore(pre)
        scratch.command(sim.targets)
        assert np.array_equal(scratch.data.qpos, sim.data.qpos)
        hold = ad.hold_sequence(sim.targets, params, 3)
        assert np.array_equal(hold[0], robot.hold_action())
    finally:
        robot.close()


def test_recording_projector_and_pose_oracle(sim_ok):
    from embodied_jepa import so101_approach_diag_runtime as ar
    from embodied_jepa import so101_closed_loop_runtime as rt

    robot = rt.make_robot(False)
    try:
        task = rt.lift_task(robot)
        task.reset(104901)
        sim = robot.sim
        params = cl.PreviewParams.of(robot)
        req = np.random.default_rng(5).uniform(-1, 1, (1, 6, 4, 6)).astype(np.float32)
        a = rt.Projector(params, sim.targets)(req)
        rec = ar.RecordingProjector(params, sim.targets)
        b = rec(req)
        assert np.array_equal(a.actions, b.actions) and np.array_equal(a.feasible, b.feasible)
        q1 = sim.data.qpos[sim.qadr].copy()
        scratch = ar.Scratch(sim, render=False)
        oracle = ar.OracleCost("pose", scratch, goal_q=q1)
        oracle.bind(scratch.snapshot(sim), rec)
        cost = oracle.distance(oracle.predict(None, b.actions), None)
        assert cost.shape == (1, 6, 4) and cost.dtype == np.float32
        assert np.all(cost[..., 0] == cost[..., -1])
        want = [
            ad.pose_cost(scratch.run(oracle.snap, rec.last_targets[0, k])["q"][-1], q1)
            for k in range(6)
        ]
        assert np.allclose(cost[0, :, -1], want)
        with pytest.raises(ad.GuardError):
            oracle.predict(None, b.actions + 0.5)
        with pytest.raises(ContractError):
            ar.OracleCost("img", scratch, goal_q=q1)
    finally:
        robot.close()


# ----- byte identity (§5, R28.124) --
def test_pinned_files_are_byte_identical():
    for manifest in (cl.MANIFEST, hy.MANIFEST):
        pins = json.loads((ROOT / manifest).read_text())
        files = pins["files_sha256"] | pins.get("composed_files_sha256", {})
        bad = [rel for rel, pin in files.items() if _sha(ROOT / rel) != pin]
        assert not bad, f"{manifest}: {bad}"


def test_no_core_import():
    import subprocess
    import sys

    code = (
        "import sys, embodied_jepa; "
        "assert 'embodied_jepa.so101_approach_diag' not in sys.modules; "
        "import embodied_jepa.so101_approach_diag; "
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
