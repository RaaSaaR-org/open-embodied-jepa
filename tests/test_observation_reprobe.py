"""TASK-061 observation re-probe: look motion, guards, render-path checks, Holm, decision."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_jepa import info_ceiling as ic
from embodied_jepa import observation_reprobe as orp

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "benchmarks" / "manifests" / "apple-observation-reprobe-v1.json").read_text()
)


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----- the look motion is reset-independent --------------------------------------------------
def test_look_sequence_is_a_frozen_constant_matching_the_manifest():
    assert inspect.signature(orp.look_sequence).parameters == {}
    first, second = orp.look_sequence(), orp.look_sequence()
    assert first.shape == (orp.LOOK_STEPS, 14) and first.dtype == np.float32
    assert np.array_equal(first, second)
    assert not first.flags.writeable
    with pytest.raises(ValueError):
        first[0, 6] = 0.4
    look = MANIFEST["look_motion"]
    assert orp.sequence_sha256(first) == look["sequence_sha256_float32_bytes"]
    assert orp.LOOK_SEQUENCE_SHA256 == look["sequence_sha256_float32_bytes"]
    assert look["steps"] == orp.LOOK_STEPS
    assert np.array_equal(first[0], np.asarray(look["command_14d"], np.float32))
    assert first[0, 6] == 0.0  # right_dx: the reset-dependent component is fixed at 0


def test_look_command_lies_inside_the_collection_bounds():
    collector = _load("_collect_apple_wide_t", "scripts/collect_apple_wide.py")
    command = orp.look_sequence()[0]
    assert np.array_equal(np.clip(command, collector.LOWER, collector.UPPER), command)


class FakeRobot:
    """Records what the look executor asks of it. ``hidden_reset`` stands for everything a
    reset could make different (apple, plate, seed); the executor never sees it."""

    def __init__(self, hidden_reset, *, infeasible_at=None, reject_at=None):
        self.hidden_reset = hidden_reset
        self.requested, self.executed, self.calls = [], [], []
        self.infeasible_at, self.reject_at = infeasible_at, reject_at

    def observe(self):
        self.calls.append("observe")

    def project_candidates(self, requested):
        self.calls.append("project")
        self.requested.append(np.array(requested[0, 0, 0]))
        feasible = len(self.requested) - 1 != self.infeasible_at
        return SimpleNamespace(actions=requested.copy(), feasible=np.array([[feasible]]))

    def execute(self, command):
        self.calls.append("execute")
        self.executed.append(np.array(command))
        rejected = len(self.executed) - 1 == self.reject_at
        return SimpleNamespace(
            applied_action=None if rejected else command.copy(), reason="stopped"
        )


def test_executing_the_look_against_different_resets_issues_identical_command_streams():
    assert list(inspect.signature(orp.execute_look).parameters) == ["robot", "after_step"]
    streams = []
    for reset in ({"apple_xy": (0.30, -0.20)}, {"apple_xy": (0.37, -0.12), "seed": 48123}):
        robot = FakeRobot(reset)
        applied = orp.execute_look(robot)
        assert robot.calls == ["observe", "project", "execute"] * orp.LOOK_STEPS
        assert np.array_equal(applied, orp.look_sequence())
        streams.append((np.stack(robot.requested), np.stack(robot.executed)))
    assert np.array_equal(streams[0][0], streams[1][0])
    assert np.array_equal(streams[0][1], streams[1][1])


def test_the_after_step_observer_cannot_change_the_commands():
    seen = []
    plain = FakeRobot({})
    orp.execute_look(plain)
    observed = FakeRobot({})
    orp.execute_look(observed, lambda step: seen.append(step) or "ignored")
    assert seen == list(range(orp.LOOK_STEPS))
    assert np.array_equal(np.stack(plain.requested), np.stack(observed.requested))


def test_an_infeasible_or_rejected_look_command_is_a_g_look_failure():
    with pytest.raises(orp.GuardError, match="infeasible"):
        orp.execute_look(FakeRobot({}, infeasible_at=3))
    with pytest.raises(orp.GuardError, match="rejected"):
        orp.execute_look(FakeRobot({}, reject_at=5))
    assert issubclass(orp.GuardError, ic.GuardError)  # the runner voids on it


@pytest.mark.skipif(os.environ.get("JEPA_TEST_RENDER") != "1", reason="graphics opt-in")
def test_the_real_look_is_identical_from_two_different_resets():
    pytest.importorskip("mujoco")
    runner = _load("_probe_orp_t1", "scripts/probe_observation_reprobe.py")
    robot = runner.make_robot(112)
    try:
        runs = []
        for object_xy, plate_xy in (((0.31, -0.21), (0.47, -0.10)), ((0.37, -0.14), (0.50, -0.2))):
            robot.reset(seed=48000, object_xy=object_xy, plate_xy=plate_xy)
            applied = orp.execute_look(robot)
            runs.append((applied, runner.state_vector(robot)))
        assert np.array_equal(runs[0][0], runs[1][0])
        assert np.array_equal(runs[0][0], orp.look_sequence())
        assert np.array_equal(runs[0][1], runs[1][1])  # the post-look state carries no reset
    finally:
        robot.sim.close()


# ----- G-look --------------------------------------------------------------------------------
def _look_inputs(roots=3, instances=4):
    sequence = orp.look_sequence().astype(np.float64)
    return {
        "sequence_sha": orp.sequence_sha256(orp.look_sequence()),
        "want_sha": MANIFEST["look_motion"]["sequence_sha256_float32_bytes"],
        "applied": np.broadcast_to(sequence, (instances, roots, *sequence.shape)).copy(),
        "post_states": np.ones((roots, 5)),
        "apple_xy_moves": np.full(roots, 1e-15),
        "plate_moves": np.zeros(roots),
        "contacts": np.zeros(roots, bool),
        "steps_checked": np.full(roots, orp.LOOK_STEPS),
    }


def test_look_guard_passes_a_clean_look():
    report = orp.check_look(**_look_inputs())
    assert report["applied_max_abs_minus_requested"] == 0.0
    assert report["post_look_state_max_abs_spread"] == 0.0


def _tamper(key, fn):
    inputs = _look_inputs()
    inputs[key] = fn(inputs[key])
    return inputs


@pytest.mark.parametrize(
    ("key", "fn", "message"),
    [
        ("sequence_sha", lambda v: "0" * 64, "sequence"),
        ("applied", lambda a: a.__setitem__((2, 1, 4, 7), -0.39) or a, "differ from the requested"),
        ("applied", lambda a: a[:, :1], "shape"),
        ("post_states", lambda s: s + np.arange(3)[:, None] * 1e-3, "post-look joint state"),
        ("apple_xy_moves", lambda m: m + [0, 2e-6, 0], "apple moved"),
        ("plate_moves", lambda m: m + [0, 0, 1e-3], "plate moved"),
        ("contacts", lambda c: np.array([False, True, False]), "touches the apple"),
        ("steps_checked", lambda c: np.array([8, 7, 8]), "not every look command"),
    ],
)
def test_look_guard_refuses_each_violation(key, fn, message):
    with pytest.raises(orp.GuardError, match=message):
        orp.check_look(**_tamper(key, fn))


# ----- G-prior -------------------------------------------------------------------------------
def test_prior_guard_both_directions():
    recorded = MANIFEST["calibration"]["values"]["prior_baselines"]["post_look_targets"]
    orp.check_priors(dict(recorded), recorded, "post-look")
    shifted = dict(recorded)
    shifted["cv_majority_dx_sign_accuracy"] += 2e-9
    with pytest.raises(orp.GuardError, match="G-prior"):
        orp.check_priors(shifted, recorded, "post-look")
    missing = {k: v for k, v in recorded.items() if k != "cv_constant_median_dx_mae"}
    with pytest.raises(orp.GuardError, match="not recomputed"):
        orp.check_priors(missing, recorded, "post-look")


def test_reset_target_priors_in_this_manifest_equal_task059s():
    task059 = json.loads(
        (ROOT / "benchmarks" / "manifests" / "apple-info-ceiling-v1.json").read_text()
    )["calibration"]["values"]["prior_baselines"]
    here = MANIFEST["calibration"]["values"]["prior_baselines"]["reset_targets"]
    assert all(here[k] == task059[k] for k in task059)


# ----- render-path checks (P1-P5): demote, never void ---------------------------------------
def _frame(size=112, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


@pytest.mark.parametrize("size", [112, 224])
def test_frame_check_passes_a_clean_render_and_fails_a_planted_one_pixel_difference(size):
    kept = _frame(size)
    assert orp.frames_identical(kept, kept.copy())
    planted = kept.copy()
    planted[size // 2, size // 3, 1] ^= 1  # one pixel, one level
    assert not orp.frames_identical(kept, planted)
    assert not orp.frames_identical(kept, kept[:-1])
    assert not orp.frames_identical(kept, kept.astype(np.int16))


def test_model_check_both_directions():
    settings = {"scene_sha256": "a" * 64, "offwidth": 640, "offheight": 480, "offsamples": 4}
    assert orp.models_agree(settings, dict(settings))
    assert not orp.models_agree(settings, settings | {"offsamples": 8})
    assert not orp.models_agree(settings, {k: v for k, v in settings.items() if k != "offwidth"})


def _clean_counts(n=190):
    return {k: n for keys in orp.REQUIRED_PATH_CHECKS.values() for k in keys}


def test_validated_arms_all_pass_on_clean_counts():
    assert orp.validated_arms(_clean_counts(), 190) == {"L": True, "LH": True, "O": True}


@pytest.mark.parametrize(
    ("check", "demoted"),
    [
        ("P4_post_112_rerender", "L"),
        ("P4_post_224_rerender", "LH"),
        ("P4_overview_rerender", "O"),
        ("P5_model_224_equals_112", "LH"),
        ("P3_physics_224_equals_112", "LH"),
        ("P1_post_112_replica", "L"),
        ("P2_png_overview", "O"),
    ],
)
def test_one_failing_root_demotes_exactly_its_arm_and_blocks_its_pass(check, demoted):
    counts = _clean_counts()
    counts[check] = 189  # a planted 1-pixel difference on one root
    validated = orp.validated_arms(counts, 190)
    assert validated[demoted] is False
    assert all(v for arm, v in validated.items() if arm != demoted)
    assert not orp.passes(validated=False, succeeds=True, rejected=True, spurious=False)


def test_required_checks_follow_the_protocol_and_missing_counts_are_refused():
    required = MANIFEST["render_path_validation"]["required"]
    for arm, keys in orp.REQUIRED_PATH_CHECKS.items():
        assert sorted(k.split("_")[0] for k in keys) == sorted(required[arm])
    counts = _clean_counts()
    del counts["P5_model_224_equals_112"]
    with pytest.raises(Exception, match="missing"):
        orp.validated_arms(counts, 190)


@pytest.mark.skipif(os.environ.get("JEPA_TEST_RENDER") != "1", reason="graphics opt-in")
def test_warmed_up_renders_are_stable_and_a_planted_pixel_is_caught():
    pytest.importorskip("mujoco")
    runner = _load("_probe_orp_t2", "scripts/probe_observation_reprobe.py")
    robot = runner.make_robot(224)
    try:
        first = robot.sim.render().copy()
        again = robot.sim.render().copy()
        assert orp.frames_identical(first, again)
        planted = again.copy()
        planted[0, 0, 0] ^= 1
        assert not orp.frames_identical(first, planted)
        small = runner.make_robot(112)
        try:
            assert orp.models_agree(runner.model_settings(robot), runner.model_settings(small))
        finally:
            small.sim.close()
    finally:
        robot.sim.close()


# ----- p-values and Holm ---------------------------------------------------------------------
@pytest.mark.parametrize("n", [79, 190])
def test_score_test_p_agrees_with_the_wilson_bar_at_every_count(n):
    p0 = 0.621
    for correct in range(n + 1):
        lower = ic.wilson(correct, n)[0]
        p = orp.score_test_p(correct, n, p0)
        if abs(lower - p0) > 1e-9:
            assert (lower > p0) == (p < 0.025), (correct, lower, p)


def test_score_test_p_is_one_at_degenerate_priors():
    assert orp.score_test_p(10, 10, 1.0) == 1.0
    assert orp.score_test_p(10, 10, 0.0) == 1.0


def test_bootstrap_share_agrees_with_the_percentile_bar():
    rng = np.random.default_rng(1)
    n = 190
    idx = ic.bootstrap_indices(n)
    base = rng.uniform(1.0, 3.0, n)
    for scale in (0.3, 0.45, 0.55, 0.6, 0.7, 1.0):
        source = base * scale * rng.uniform(0.7, 1.3, n)
        share = orp.bootstrap_share_above(source, base, idx, ic._median, 0.6)
        upper = ic.paired_ratio(source, base, idx, ic._median)["ci95"][1]
        if abs(upper - 0.6) > 5e-3:
            assert (upper <= 0.6) == (share <= 0.025), (scale, upper, share)


def test_bootstrap_share_counts_a_zero_baseline_as_above():
    idx = np.array([[0, 0], [0, 1]])
    assert orp.bootstrap_share_above([1.0, 1.0], [0.0, 1.0], idx, ic._median, 0.6) == 1.0


def _synthetic(perfect: bool):
    rng = np.random.default_rng(7)
    n = 190
    xy = np.column_stack([0.34 + rng.uniform(-0.03, 0.03, n), -0.18 + rng.uniform(-0.03, 0.03, n)])
    dx = np.clip((xy[:, 0] - 0.333) / 0.015, -0.4, 0.4)
    dx[dx == 0] = 0.1
    occluded = xy[:, 1] < -0.19
    fold = ic.fold_of(n)
    priors = ic.prior_predictions(xy, dx, occluded, fold)
    if perfect:
        return xy + rng.normal(0, 0.001, xy.shape), dx.copy(), xy, dx, priors
    return priors["B_mean"], priors["B_const"].copy(), xy, dx, priors


def test_a_perfect_readout_gets_tiny_p_and_passes_holm_and_the_bars():
    xy_pred, dx_pred, xy, dx, priors = _synthetic(True)
    mask = np.ones(len(dx), bool)
    p = orp.hypothesis_pvalues(xy_pred, dx_pred, xy, dx, priors, mask)
    assert p["point_conditions_hold"] and p["p"] < 0.00625
    assert ic.evaluate(xy_pred, dx_pred, xy, dx, priors, mask)["succeeds"]


def test_a_prior_only_readout_gets_p_one():
    xy_pred, dx_pred, xy, dx, priors = _synthetic(False)
    p = orp.hypothesis_pvalues(xy_pred, dx_pred, xy, dx, priors, np.ones(len(dx), bool))
    assert not p["point_conditions_hold"] and p["p"] == 1.0


def test_holm_step_down_both_directions():
    thresholds = [0.025 / 4, 0.025 / 3, 0.025 / 2, 0.025]
    assert MANIFEST["multiplicity"]["step_thresholds"] == pytest.approx(thresholds)
    all_pass = orp.holm({"L-raw": 0.001, "L-E0": 0.006, "LH-raw": 0.012, "O-raw": 0.025})
    assert all(all_pass["rejected"].values())
    # Step 2 fails (0.009 > 0.025/3), so every later hypothesis stays accepted even where its
    # own p would pass its own step alone (0.0095 <= 0.025/2, 0.02 <= 0.025).
    blocked = orp.holm({"L-raw": 0.009, "L-E0": 0.0095, "LH-raw": 0.02, "O-raw": 0.0001})
    assert blocked["rejected"] == {"O-raw": True, "L-raw": False, "L-E0": False, "LH-raw": False}
    stops = orp.holm({"L-raw": 0.001, "L-E0": 0.03, "LH-raw": 0.001, "O-raw": 0.001})
    assert stops["rejected"] == {"L-raw": True, "LH-raw": True, "O-raw": True, "L-E0": False}
    with pytest.raises(Exception, match="unknown"):
        orp.holm({"X": 0.0})


def test_holm_ties_follow_the_listed_order():
    steps = orp.holm({h: 0.004 for h in orp.DECISIONAL})["steps"]
    assert [s["hypothesis"] for s in steps] == list(orp.DECISIONAL)


# ----- spurious check ------------------------------------------------------------------------
def test_spurious_verdict_both_directions():
    assert orp.spurious_verdict({"beats_prior": True}, renderer_reproduces=True)["spurious"]
    clean = orp.spurious_verdict({"beats_prior": False}, renderer_reproduces=True)
    assert clean["available"] and not clean["spurious"]
    unavailable = orp.spurious_verdict({"beats_prior": False}, renderer_reproduces=False)
    assert not unavailable["available"] and unavailable["spurious"]


def test_fold_readouts_on_unchanged_features_reproduce_the_out_of_fold_predictions():
    rng = np.random.default_rng(2)
    n = 60
    x = rng.normal(size=(n, 5))
    y = x[:, :2] @ np.array([[1.0, 0.5], [0.2, -1.0]]) + rng.normal(0, 0.01, (n, 2))
    fold = ic.fold_of(n)
    g, diag = ic.gram(x)
    predictions, fits, _ = ic.nested_cv(g, diag, y, fold)
    g_new, norms = ic.cross_gram(x, x)
    again = orp.predict_with_fold_readouts(fits, fold, g_new, norms)
    assert np.allclose(again, predictions)
    # Features without the signal (the "apple hidden") fall back towards the mean.
    hidden = np.zeros_like(x)
    g_h, n_h = ic.cross_gram(hidden, x)
    flat = orp.predict_with_fold_readouts(fits, fold, g_h, n_h)
    assert np.abs(flat - y.mean(0)).max() < np.abs(predictions - y.mean(0)).max()


# ----- decision ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("passed", "row"),
    [
        ({"L-E0": True, "L-raw": True, "O-raw": True}, "O-LOOK-E0"),
        ({"L-raw": True, "LH-raw": True}, "O-LOOK-RAW"),
        ({"LH-raw": True, "O-raw": True}, "O-LOOK-224"),
        ({"O-raw": True}, "O-OVERVIEW"),
        ({}, "O-NONE"),
    ],
)
def test_decision_rows_in_order(passed, row):
    full = {h: passed.get(h, False) for h in orp.DECISIONAL}
    decision = orp.decide(void=False, passed=full)
    assert decision["outcome"] == row
    assert decision["passing_hypotheses"] == [h for h in orp.DECISIONAL if full[h]]


def test_void_precedes_every_row_and_missing_hypotheses_are_refused():
    everything = {h: True for h in orp.DECISIONAL}
    assert orp.decide(void=True, passed=everything)["outcome"] == "V"
    with pytest.raises(Exception, match="missing"):
        orp.decide(void=False, passed={"L-raw": True})
    manifest_rows = [r["row"] for r in MANIFEST["pre_declared_outcomes_in_order"]]
    assert tuple(manifest_rows) == orp.ROWS


@pytest.mark.parametrize(
    "fail",
    ["validated", "succeeds", "rejected", "spurious"],
)
def test_pass_rule_needs_every_condition(fail):
    kwargs = {"validated": True, "succeeds": True, "rejected": True, "spurious": False}
    assert orp.passes(**kwargs)
    kwargs[fail] = not kwargs[fail]
    assert not orp.passes(**kwargs)


def test_arm_occ_prior_falls_back_to_the_training_mean():
    xy = np.arange(40, dtype=float).reshape(20, 2)
    fold = np.arange(20) % 10
    occluded = np.zeros(20, bool)
    occluded[3] = True  # the only occluded root: its training fold has none
    out = orp.arm_occ_prior(xy, occluded, fold)
    fit = fold != fold[3]
    assert np.allclose(out[3], xy[fit].mean(0))
    assert np.isfinite(out).all()


# ----- manifest and hashes -------------------------------------------------------------------
def test_manifest_hashes_include_the_unchanged_task059_probe():
    for name in (
        "src/embodied_jepa/info_ceiling.py",
        "scripts/probe_info_ceiling.py",
        "scripts/calibrate_observation_reprobe.py",
        "benchmarks/manifests/apple-info-ceiling-v1.json",
    ):
        assert name in MANIFEST["hashes"]
    for name, want in MANIFEST["hashes"].items():
        path = ROOT / name
        if path.is_file() and not name.startswith(("data/", "checkpoints/")):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == want, name


def test_manifest_hypotheses_match_the_module():
    assert tuple(MANIFEST["decisional_hypotheses_in_order"]) == orp.DECISIONAL


# ----- void-safe runner ----------------------------------------------------------------------
def test_wall_cap_voids():
    runner = _load("_probe_orp_t3", "scripts/probe_observation_reprobe.py")
    runner.BASE.Clock(3600).check("stage")
    with pytest.raises(runner.VoidRun, match="wall cap"):
        runner.BASE.Clock(-1).check("stage")


def _committed_only_manifest(tmp_path, **hashes):
    committed = json.loads(json.dumps(MANIFEST))
    committed["hashes"] = hashes or {
        "configs/g1_sim_action.json": MANIFEST["hashes"]["configs/g1_sim_action.json"]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(committed))
    return path


def test_a_failed_guard_writes_a_v_report_and_never_overwrites(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    runner = _load("_probe_orp_t4", "scripts/probe_observation_reprobe.py")
    path = _committed_only_manifest(tmp_path, **{"configs/g1_sim_action.json": "0" * 64})
    monkeypatch.setattr(runner, "MANIFEST", path)
    report = runner.run(tmp_path / "out", smoke=True)
    assert report["status"] == "void" and report["outcome"] == "V"
    written = json.loads((tmp_path / "out" / "report.json").read_text())
    assert written["outcome"] == "V" and "g1_sim_action.json" in written["void_reason"]
    with pytest.raises(FileExistsError):
        runner.run(tmp_path / "out", smoke=True)


def test_a_crash_mid_run_writes_a_v_report_with_a_void_reason_and_reraises(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    runner = _load("_probe_orp_t5", "scripts/probe_observation_reprobe.py")
    monkeypatch.setattr(runner, "MANIFEST", _committed_only_manifest(tmp_path))

    def boom():
        raise RuntimeError("injected crash")

    monkeypatch.setattr(runner, "plan_roots", boom)
    with pytest.raises(RuntimeError, match="injected crash"):
        runner.run(tmp_path / "out", smoke=True)
    written = json.loads((tmp_path / "out" / "report.json").read_text())
    assert written["outcome"] == "V" and written["decision"]["outcome"] == "V"
    assert "injected crash" in written["void_reason"] and "Traceback" in written["traceback"]


def test_a_mismatched_module_look_hash_voids(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    runner = _load("_probe_orp_t6", "scripts/probe_observation_reprobe.py")
    monkeypatch.setattr(runner, "MANIFEST", _committed_only_manifest(tmp_path))
    monkeypatch.setattr(runner.orp, "LOOK_SEQUENCE_SHA256", "0" * 64)
    report = runner.run(tmp_path / "out", smoke=True)
    assert report["outcome"] == "V" and "G-look" in report["void_reason"]


def test_non_finite_values_become_null_and_are_listed(tmp_path):
    runner = _load("_probe_orp_t7", "scripts/probe_observation_reprobe.py")
    runner.write_report(
        tmp_path / "report.json", {"a": 1.0, "b": [float("inf"), 2.0], "c": {"d": np.nan}}
    )
    written = json.loads((tmp_path / "report.json").read_text())
    assert written["b"] == [None, 2.0] and written["c"]["d"] is None
    assert sorted(written["non_finite_fields"]) == ["/b/0=inf", "/c/d=nan"]


def test_an_unserialisable_report_still_leaves_a_v_report(tmp_path):
    runner = _load("_probe_orp_t8", "scripts/probe_observation_reprobe.py")
    with pytest.raises(TypeError):
        runner.write_report(tmp_path / "report.json", {"task": "TASK-061", "x": object()})
    written = json.loads((tmp_path / "report.json").read_text())
    assert written["outcome"] == "V" and "serialisation" in written["void_reason"]
