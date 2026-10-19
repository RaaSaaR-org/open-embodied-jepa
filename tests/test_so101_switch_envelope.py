"""TASK-101 A2 (§19): the switching-tolerance envelope's sampler, cohorts, rule and hand-over."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import so101_closed_loop as cl
from embodied_jepa import so101_switch_envelope as env

ROOT = Path(__file__).resolve().parents[1]
ASSET_MANIFEST = json.loads((ROOT / "assets/so101_manifest.json").read_text())
FETCHED = all(
    (ROOT / ASSET_MANIFEST["local_path"] / relative).is_file()
    for relative in ASSET_MANIFEST["sha256"]
)


def test_direction_has_unit_rms_is_seeded_and_differs_per_subgoal():
    a = env.direction(102000, 1)
    assert a.shape == (5,) and np.isclose(env.rms(a), 1.0)
    np.testing.assert_array_equal(a, env.direction(102000, 1))
    assert not np.allclose(a, env.direction(102000, 2))
    assert not np.allclose(a, env.direction(102001, 1))
    for level in env.LEVELS_RAD:
        d = env.offset(102000, 2, level)
        assert np.isclose(env.rms(d), level)
        np.testing.assert_allclose(d / level, env.direction(102000, 2))
    with pytest.raises(ValueError):
        env.direction(102000, 3)
    with pytest.raises(ValueError):
        env.offset(102000, 1, float("nan"))


def test_cohorts_are_disjoint_from_every_task101_cohort_and_each_other():
    used = set()
    for c in cl.COHORTS.values():
        used |= set(c.seeds) | set(c.reserve_seeds)
    mine = [env.ENVELOPE, env.K_SECOND, env.D_SECOND]
    seen = set()
    for c in mine:
        seeds = set(c.main()) | set(c.spare())
        assert not seeds & used and not seeds & seen
        seen |= seeds
    assert env.ENVELOPE.count == env.PER_LEVEL == 32
    env.check_seed(102000)
    env.check_seed(102047)
    for bad in (101999, 102048, 101900, 101700):
        with pytest.raises(ValueError):
            env.check_seed(bad)


def _counts(pre, carry):
    return {
        **{("pre", lv): c for lv, c in zip(env.LEVELS_RAD, pre, strict=True)},
        **{("carry", lv): c for lv, c in zip(env.LEVELS_RAD, carry, strict=True)},
    }


def test_rule_picks_the_largest_level_with_every_smaller_level_passing():
    full = [32] * 6
    assert env.envelope_row(_counts(full, full)) == {
        "row": "A2-TAU",
        "tau_s_rad": 0.16,
        "largest_passing_rad": 0.16,
        "current_tolerance_passes": True,
        "reason": "K'' runs with tau_s (§19.5)",
    }
    r = env.envelope_row(_counts(full, [32, 31, 30, 29, 32, 32]))
    assert r["row"] == "A2-TAU" and r["tau_s_rad"] == 0.06  # non-monotone: stops at 0.08
    r = env.envelope_row(_counts([32, 32, 30, 30, 29, 10], full))
    assert r["row"] == "A2-TAU" and r["tau_s_rad"] == 0.08


def test_rule_keeps_004_and_escalates_when_no_level_from_006_passes():
    full = [32] * 6
    r = env.envelope_row(_counts(full, [32, 32, 29, 32, 32, 32]))
    assert r["row"] == "A2-NOT-BINDING" and r["tau_s_rad"] == 0.04
    assert r["largest_passing_rad"] == 0.04 and r["current_tolerance_passes"]
    r = env.envelope_row(_counts([29] * 6, full))
    assert r["row"] == "A2-NOT-BINDING" and r["tau_s_rad"] == 0.04
    assert r["largest_passing_rad"] is None and not r["current_tolerance_passes"]


def test_rule_void_and_validation():
    full = [32] * 6
    assert env.envelope_row(_counts(full, full), kept=31)["row"] == "A2-VOID"
    assert env.envelope_row(_counts(full, full), void="G-zero")["row"] == "A2-VOID"
    with pytest.raises(ValueError):
        env.envelope_row(_counts(full, [33] * 6))


def test_clopper_pearson_matches_known_values():
    lo, hi = env.clopper_pearson(32, 32)
    assert hi == 1.0 and abs(lo - 0.8911) < 1e-3
    lo, hi = env.clopper_pearson(30, 32)
    assert abs(lo - 0.7919) < 1e-3 and abs(hi - 0.9923) < 1e-3
    lo, hi = env.clopper_pearson(0, 32)
    assert lo == 0.0 and abs(hi - 0.1089) < 1e-3


def test_core_import_stays_clean():
    code = (
        "import sys, embodied_jepa.so101_switch_envelope; "
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# ----- physics (headless; plumbing on a debug seed, no count is asserted) --
@pytest.fixture
def make():
    pytest.importorskip("mujoco")
    if not FETCHED:
        pytest.skip("pinned SO-101 assets have not been fetched")
    from embodied_jepa import so101_closed_loop_runtime as rt

    return rt


@pytest.mark.parametrize("variant", ["pre", "carry", "reaim"])
def test_zero_offset_executes_exactly_the_goal_demonstration(make, variant):
    rt = make
    demo = rt.demonstrate(101900, render=False)
    robot = rt.make_robot(False)
    rec = rt.Recorder(robot)
    try:
        out = env.perturbed_lift(
            rec, rt.lift_task(robot), 101900, variant=variant, delta=np.zeros(5)
        )
    finally:
        robot.close()
    applied = np.asarray(rec.applied, np.float32).reshape(-1, 6)
    np.testing.assert_array_equal(applied, demo["applied"])
    command = env.HANDOVER_COMMAND[dict(env.VARIANTS)[variant]]
    ref = next(c["joints"] for c in demo["captured"] if c["command"] == command)
    np.testing.assert_array_equal(np.asarray(out["handover_joints"]), ref)


def test_pre_handover_arrives_at_the_offset_pose(make):
    """Mechanics only: in free space the arm reaches q_pre + δ at command 40."""
    rt = make
    demo = rt.demonstrate(101901, render=False)
    ref = next(c["joints"] for c in demo["captured"] if c["command"] == 40)
    delta = env.offset(102000, 1, 0.04)  # any unit direction; the seed only picks it
    robot = rt.make_robot(False)
    try:
        out = env.perturbed_lift(robot, rt.lift_task(robot), 101901, variant="pre", delta=delta)
    finally:
        robot.close()
    achieved = np.asarray(out["handover_joints"])[:5] - np.asarray(ref)[:5]
    np.testing.assert_allclose(achieved, delta, atol=5e-3)
