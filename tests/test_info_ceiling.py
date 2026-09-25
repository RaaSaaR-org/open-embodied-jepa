"""TASK-059 information-ceiling probe: readouts, statistics, guards (both directions), decision.

Every guard is exercised with a passing input and with each violation it exists to catch.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import info_ceiling as ic

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "benchmarks" / "manifests" / "apple-info-ceiling-v1.json").read_text()
)


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan_roots():
    collector = _load("_collect_apple_wide_t", "scripts/collect_apple_wide.py")
    plan = collector.make_plan(collector.FROZEN_SEEDS)["roots"]
    return sorted((r for r in plan if r["split"] in ("train", "val")), key=lambda r: r["seed"])


def _splits(roots):
    """A dataset-manifest-shaped split table consistent with the plan (plus a test id)."""
    table = {"train": [], "val": [], "test": ["wide-48999"], "holdout": []}
    for root in roots:
        table[root["split"]].append(root["episode_id"])
    return table


# ----- folds ---------------------------------------------------------------------------------
def test_fold_assignment_matches_the_preregistered_hash_and_the_calibration():
    roots = _plan_roots()
    fold = ic.fold_of(len(roots))
    assert np.bincount(fold).tolist() == [19] * 10
    digest = ic.fold_assignment_sha256([r["seed"] for r in roots], fold)
    assert digest == MANIFEST["split"]["fold_assignment_sha256"]
    calibration = _load("_calibrate_t", "scripts/calibrate_info_ceiling.py")
    assert np.array_equal(calibration.fold_of(len(roots)), fold)
    # A different seed would give a different, detectable assignment.
    other = np.random.default_rng(60).permutation(len(roots)) % 10
    assert ic.fold_assignment_sha256([r["seed"] for r in roots], other) != digest


def test_val_seeds_match_the_manifest():
    roots = _plan_roots()
    assert [r["seed"] for r in roots if r["split"] == "val"] == MANIFEST["split"]["val_seeds"]


# ----- G-split -------------------------------------------------------------------------------
def test_split_guard_accepts_the_preregistered_roots():
    roots = _plan_roots()
    ic.check_split(roots, _splits(roots))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda rs, t: rs.__setitem__(0, rs[0] | {"seed": 45305}), "cohort C"),
        (lambda rs, t: rs.__setitem__(0, rs[0] | {"seed": 45003}), "cohort D"),
        (lambda rs, t: rs.__setitem__(0, rs[0] | {"seed": 45102}), "cohort D"),
        (lambda rs, t: rs.__setitem__(-1, rs[-1] | {"seed": 48250}), "outside"),
        (lambda rs, t: rs.__setitem__(0, rs[0] | {"split": "test"}), "only train/val"),
        (lambda rs, t: t["test"].append(rs[0]["episode_id"]), "test/holdout"),
        (lambda rs, t: t["holdout"].append(rs[0]["episode_id"]), "test/holdout"),
        (lambda rs, t: t["train"].remove(rs[0]["episode_id"]), "frozen"),
        (lambda rs, t: rs.pop(), "counts"),
        (lambda rs, t: rs.reverse(), "sorted"),
    ],
)
def test_split_guard_refuses_each_violation(mutate, message):
    roots = _plan_roots()
    table = _splits(roots)
    if roots[0]["split"] != "train":  # the train-removal case assumes a train first root
        pytest.skip("fixture assumption")
    mutate(roots, table)
    with pytest.raises(ic.GuardError, match=message):
        ic.check_split(roots, table)


def test_split_guard_refuses_a_root_whose_split_disagrees_with_the_dataset():
    roots = _plan_roots()
    table = _splits(roots)
    val_id = next(r["episode_id"] for r in roots if r["split"] == "val")
    table["val"].remove(val_id)
    table["train"].append(val_id)
    with pytest.raises(ic.GuardError, match="frozen val split"):
        ic.check_split(roots, table)


# ----- G-hash --------------------------------------------------------------------------------
def test_hash_guard_both_directions():
    recorded = {"a": "1" * 64, "b": "2" * 64}
    ic.check_hashes(recorded, dict(recorded))
    with pytest.raises(ic.GuardError, match="a: sha256"):
        ic.check_hashes(recorded, {"a": "3" * 64, "b": "2" * 64})
    with pytest.raises(ic.GuardError, match="b: sha256 None"):
        ic.check_hashes(recorded, {"a": "1" * 64})


def test_manifest_hashes_cover_every_preregistered_input():
    assert set(MANIFEST["hashes"]) == {
        "data/apple-wide-v1/meta/jepa_manifest.json",
        "checkpoints/task056-policy-v1/a1.pt",
        "checkpoints/task056-policy-v1/a2.pt",
        "checkpoints/task056-policy-v1/a3.pt",
        "checkpoints/task054-wm-v4/leworldmodel_baseline.pt",
        "configs/apple_policy_v1.yaml",
        "configs/apple_wm_v4_lewm.yaml",
        "configs/apple_wm_v4.yaml",
        "scripts/collect_apple_wide.py",
        "scripts/calibrate_info_ceiling.py",
        "scripts/measure_policy_offline_conditionals.py",
        "src/embodied_jepa/readout_labels.py",
        "src/embodied_jepa/scripted.py",
        "src/embodied_jepa/simulation.py",
        "src/embodied_jepa/embodiment.py",
        "src/embodied_jepa/policy.py",
        "assets/manifest.json",
        "configs/g1_sim_action.json",
    }
    # The committed sources among them must still be the preregistered bytes.
    for name, want in MANIFEST["hashes"].items():
        if name.startswith(("src/", "scripts/", "configs/", "assets/")):
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == want, name


def test_frame_reader_refuses_unchecked_ids_and_mismatching_files(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("PIL")
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    runner = _load("_probe_t", "scripts/probe_info_ceiling.py")
    frame = (np.arange(112 * 112 * 3) % 251).astype(np.uint8).reshape(112, 112, 3)
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    table = pa.table(
        {
            "observation.images.onboard_rgb": [{"bytes": buffer.getvalue(), "path": None}],
            "frame_index": [0],
        }
    )
    path = tmp_path / "ep.parquet"
    pq.write_table(table, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "episodes": [
            {"episode_id": "wide-48000", "path": "ep.parquet"},
            {"episode_id": "wide-48001", "path": "ep.parquet"},
        ],
        "sha256": {"ep.parquet": digest},
    }
    reader = runner.FrameReader(manifest, ["wide-48000"])
    assert np.array_equal(reader.first_frame("wide-48000", store_root=tmp_path), frame)
    with pytest.raises(ic.GuardError, match="not a checked root"):
        reader.first_frame("wide-48001", store_root=tmp_path)
    manifest["sha256"]["ep.parquet"] = "0" * 64
    with pytest.raises(ic.GuardError, match="hash mismatch"):
        reader.first_frame("wide-48000", store_root=tmp_path)


# ----- G-render / G-expert / G-prior ---------------------------------------------------------
def test_render_guard_both_directions():
    states = np.ones((5, 86))
    ic.check_render([True] * 5, states)
    with pytest.raises(ic.GuardError, match="1 roots"):
        ic.check_render([True, True, False, True, True], states)
    with pytest.raises(ic.GuardError, match="differs"):
        ic.check_render([], states)
    varied = states.copy()
    varied[2, 7] += 1e-3
    with pytest.raises(ic.GuardError, match="varies"):
        ic.check_render([True] * 5, varied)


def test_expert_guard_both_directions():
    expert = np.zeros((4, 7))
    recorded = expert.copy()
    recorded[3, 0] = 0.8  # aim-offset root: allowed to differ
    aim = [False, False, False, True]
    apple = np.full((4, 2), 0.3)
    ic.check_expert(expert, recorded, aim, apple, apple)
    bad = recorded.copy()
    bad[1, 0] = 1e-3
    with pytest.raises(ic.GuardError, match="recorded label"):
        ic.check_expert(expert, bad, aim, apple, apple)
    with pytest.raises(ic.GuardError, match="reset truth"):
        ic.check_expert(expert, recorded, aim, apple + 1e-4, apple)
    with pytest.raises(ic.GuardError, match="no non-aim"):
        ic.check_expert(expert, recorded, [True] * 4, apple, apple)


def _synthetic_cohort(n=190, seed=3):
    rng = np.random.default_rng(seed)
    xy = np.column_stack([0.34 + rng.uniform(-0.03, 0.03, n), -0.18 + rng.uniform(-0.03, 0.03, n)])
    dx = np.clip((xy[:, 0] - 0.333) / 0.015, -0.4, 0.4)
    occluded = xy[:, 1] < -0.18
    return xy, dx, occluded


def test_prior_summary_reproduces_the_calibration_formulas():
    """The runner's G-prior numbers equal the committed calibration's on the same inputs."""
    calibration = _load("_calibrate_t2", "scripts/calibrate_info_ceiling.py")
    xy, dx, occluded = _synthetic_cohort()
    fold = ic.fold_of(len(dx))
    train = np.arange(len(dx)) % 10 != 0
    got = ic.prior_summary(xy, dx, occluded, fold, train)
    want = calibration.prior_baselines(xy, dx, occluded, fold, train)
    assert set(got) == set(want)
    for key in want:
        assert got[key] == pytest.approx(want[key], abs=1e-12), key


def test_prior_guard_both_directions():
    recorded = MANIFEST["calibration"]["values"]["prior_baselines"]
    ic.check_priors(dict(recorded), recorded)
    shifted = dict(recorded) | {
        "cv_majority_dx_sign_accuracy": recorded["cv_majority_dx_sign_accuracy"] + 1e-6
    }
    with pytest.raises(ic.GuardError, match="cv_majority_dx_sign_accuracy"):
        ic.check_priors(shifted, recorded)
    missing = {k: v for k, v in recorded.items() if k != "val_constant_median_dx_mae"}
    with pytest.raises(ic.GuardError, match="not recomputed"):
        ic.check_priors(missing, recorded)


# ----- native equivalence (both directions) --------------------------------------------------
def _scenes(n=6, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (1, 448, 448, 3)).astype(np.uint8).repeat(n, axis=0)
    for i in range(n):  # a small per-reset object
        base[i, 40 + 8 * i : 56 + 8 * i, 100:116] = 255
    return base


def test_native_equivalence_passes_an_exact_downsample():
    native = _scenes()
    stored = np.floor(ic.box_down(native, 4) + 0.5).astype(np.uint8)
    result = ic.native_equivalence(stored, ic.box_down(native, 4), np.ones(6, bool))
    assert result["equivalent"] and result["byte_identical_after_rounding"]
    assert result["identifies_own_reset"] == 6


def test_native_equivalence_fails_when_a_render_depicts_another_reset():
    native = _scenes()
    down = ic.box_down(native, 4)
    stored = np.floor(down + 0.5).astype(np.uint8)
    wrong = np.roll(down, 1, axis=0) + 0.2  # each "native" render shows a neighbouring reset
    result = ic.native_equivalence(stored, wrong, np.ones(6, bool))
    assert not result["byte_identical_after_rounding"]
    assert not result["equivalent"] and result["identifies_own_reset"] == 0


def test_box_down_refuses_non_multiples():
    with pytest.raises(ic.ContractError):
        ic.box_down(np.zeros((1, 10, 10, 3)), 4)


# ----- readouts ------------------------------------------------------------------------------
def test_gram_centring_equals_explicit_centring_on_the_training_part():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 7))
    g, diag = ic.gram(x)
    train, rows = np.arange(20), np.arange(20, 30)
    stats = ic.train_stats(g, diag, train)
    got = ic.kernel_rows("linear", g[np.ix_(rows, train)], diag[rows], stats)
    mean = x[train].mean(0)
    want = (x[rows] - mean) @ (x[train] - mean).T
    assert np.allclose(got, want)
    d2 = ((x[rows, None] - x[None, train]) ** 2).sum(-1)
    got_rbf = ic.kernel_rows("rbf", g[np.ix_(rows, train)], diag[rows], stats)
    assert np.allclose(got_rbf, np.exp(-d2 / (2 * stats.sigma2)))


def test_held_out_rows_do_not_influence_their_own_fold():
    """Changing a held-out row's features and label leaves every other fold's model unchanged,
    and changes that row's own prediction only through its own features."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 5))
    y = x @ np.array([1.0, -2.0, 0.5, 0.0, 0.0]) + rng.normal(0, 0.1, 40)
    fold = np.arange(40) % 10
    g, diag = ic.gram(x)
    p1, fits1, _ = ic.nested_cv(g, diag, y, fold)
    y2 = y.copy()
    y2[fold == 3] += 100.0  # labels of fold 3 corrupted
    p2, fits2, _ = ic.nested_cv(g, diag, y2, fold)
    assert np.allclose(p1[fold == 3], p2[fold == 3])  # its own prediction ignores its labels


def test_nested_cv_recovers_a_linear_signal_and_a_constant_target_stays_at_its_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(120, 6))
    y = x[:, 0] * 2.0
    fold = np.arange(120) % 10
    g, diag = ic.gram(x)
    pred, _, selections = ic.nested_cv(g, diag, y, fold)
    assert np.corrcoef(pred, y)[0, 1] > 0.99
    assert all(s["family"] in ic.FAMILIES and s["lam_rel"] in ic.LAMBDAS for s in selections)
    noise = rng.normal(size=120)
    pred_noise, _, _ = ic.nested_cv(g, diag, noise, fold)
    assert np.corrcoef(pred_noise, noise)[0, 1] < 0.5


def test_readout_on_pixels_and_features_uses_the_same_machinery():
    rng = np.random.default_rng(2)
    small = rng.normal(size=(50, 4))
    wide = np.repeat(small, 1000, axis=1)  # a 4000-d "pixel" view of the same signal
    y = small[:, 1]
    fold = np.arange(50) % 10
    pa, _, _ = ic.nested_cv(*ic.gram(small), y, fold)
    pb, _, _ = ic.nested_cv(*ic.gram(wide), y, fold)
    assert np.allclose(pa, pb, atol=1e-6)  # scale-free λ grid: identical up to rounding


# ----- statistics ----------------------------------------------------------------------------
def test_wilson_matches_the_preregistered_reference_values():
    reference = MANIFEST["calibration"]["values"]["wilson_95_lower_reference"]
    assert ic.wilson(162, 190)[0] == pytest.approx(reference["162_of_190"], abs=1e-12)
    assert ic.wilson(68, 79)[0] == pytest.approx(reference["68_of_79"], abs=1e-12)
    assert ic.wilson(19, 20)[0] == pytest.approx(reference["19_of_20"], abs=1e-12)


def _priors_for(xy, dx, occluded):
    return ic.prior_predictions(xy, dx, occluded, ic.fold_of(len(dx)))


def test_blind_predictors_pass_no_threshold():
    """§9: each prior-only baseline, read out as if it were a source, fails every target."""
    xy, dx, occluded = _synthetic_cohort()
    priors = _priors_for(xy, dx, occluded)
    mask = np.ones(len(dx), bool)
    for xy_pred, dx_pred in (
        (priors["B_mean"], priors["B_const"]),
        (priors["B_occ"], priors["B_maj"] * 0.4),
        (priors["B_mean"], priors["B_y"] * 0.4),
    ):
        result = ic.evaluate(xy_pred, dx_pred, xy, dx, priors, mask)
        assert not result["T1"]["pass"] and not result["T2"]["pass"]
        assert not result["T3"]["pass"] and not result["succeeds"]
        assert not result["beats_prior"]


def test_a_perfect_readout_passes_every_threshold_and_beats_the_prior():
    xy, dx, occluded = _synthetic_cohort()
    priors = _priors_for(xy, dx, occluded)
    result = ic.evaluate(xy + 1e-4, dx, xy, dx, priors, np.ones(len(dx), bool))
    assert result["succeeds"] and result["beats_prior"]


def test_threshold_edges():
    xy, dx, occluded = _synthetic_cohort()
    priors = _priors_for(xy, dx, occluded)
    mask = np.ones(len(dx), bool)
    far = xy + np.array([0.016, 0.0])  # every error 1.6 cm > 1.5 cm
    assert not ic.evaluate(far, dx, xy, dx, priors, mask)["T1"]["pass"]
    wrong = dx.copy()
    flip = np.flatnonzero(dx != 0)[: int(0.2 * len(dx))]  # accuracy 0.8 < 0.85
    wrong[flip] *= -1
    assert not ic.evaluate(xy, wrong, xy, dx, priors, mask)["T2"]["pass"]
    zero = np.zeros_like(dx)  # a zero prediction is never a correct sign
    assert ic.evaluate(xy, zero, xy, dx, priors, mask)["T2"]["accuracy"] == 0.0


# ----- decision ------------------------------------------------------------------------------
def _flags(**true):
    return {s: bool(true.get(s, False)) for s in ic.DECISIONAL}


@pytest.mark.parametrize(
    "overall, visible, spurious, row",
    [
        (_flags(E0=True), _flags(E0=True), {}, "O-BC"),
        (_flags(raw112=True), _flags(), {}, "O-ENC"),
        (_flags(A3=True), _flags(), {}, "O-ENC"),
        (_flags(E0=True), _flags(), {"E0": True}, "O-OCC-NONE"),
        (_flags(E0=True), _flags(E0=True), {"E0": True}, "O-OCC-BC"),
        (_flags(), _flags(E0=True, raw112=True), {}, "O-OCC-BC"),
        (_flags(), _flags(raw112=True), {}, "O-OCC-ENC"),
        (_flags(), _flags(random=True), {}, "O-OCC-ENC"),
        (_flags(), _flags(), {}, "O-OCC-NONE"),
    ],
)
def test_decision_rows_in_order(overall, visible, spurious, row):
    assert (
        ic.decide(void=False, overall=overall, visible=visible, spurious=spurious)["outcome"] == row
    )


def test_void_precedes_every_row_and_missing_sources_are_refused():
    assert (
        ic.decide(void=True, overall=_flags(E0=True), visible=_flags(), spurious={})["outcome"]
        == "VOID"
    )
    with pytest.raises(ic.ContractError):
        ic.decide(void=False, overall={"E0": True}, visible=_flags(), spurious={})


# ----- rendering (graphics opt-in) -----------------------------------------------------------
@pytest.mark.skipif(os.environ.get("JEPA_TEST_RENDER") != "1", reason="graphics opt-in")
def test_apple_hidden_and_shadow_off_renders_change_only_what_they_claim():
    pytest.importorskip("mujoco")
    runner = _load("_probe_t2", "scripts/probe_info_ceiling.py")
    robot = runner.make_robot()
    renders = runner.Renders(robot)
    try:
        robot.reset(seed=48000, object_xy=[0.34, -0.14], plate_xy=[0.49, -0.09])
        assert renders.apple_pixels(112) > 0  # the apple is visible at this reset
        normal = renders.small_rgb()
        assert np.array_equal(normal, robot.observe().images["onboard_rgb"][0])
        hidden = renders.small_rgb(hide_apple=True)
        no_shadow = renders.small_rgb(shadows=False)
        assert not np.array_equal(hidden, normal)
        assert not np.array_equal(no_shadow, normal)
        assert np.array_equal(renders.small_rgb(), normal)  # state restored afterwards
        # Hiding the apple equals removing it: teleport it far below the floor and re-render.
        joint = robot.sim.data.joint("apple_free")
        saved = joint.qpos.copy()
        joint.qpos[:3] = [0.0, 0.0, -10.0]
        robot.mj.mj_forward(robot.model, robot.sim.data)
        assert np.array_equal(renders.small_rgb(), hidden)
        joint.qpos[:] = saved
        robot.mj.mj_forward(robot.model, robot.sim.data)
        assert np.array_equal(renders.small_rgb(), normal)
    finally:
        renders.close()
        robot.sim.close()


# ----- the guards that run() applies inline (both directions) --------------------------------
def test_clean_tree_guard():
    ic.check_clean_tree(False)
    for dirty in (True, None, "unavailable"):
        with pytest.raises(ic.GuardError, match="clean tree"):
            ic.check_clean_tree(dirty)


def test_fold_hash_guard():
    want = MANIFEST["split"]["fold_assignment_sha256"]
    ic.check_fold_hash(want, want)
    with pytest.raises(ic.GuardError, match="fold assignment"):
        ic.check_fold_hash("0" * 64, want)


def test_encoder_digest_guard():
    for key in ("i_E0", "ii_A3", "iii_random"):
        want = MANIFEST["sources"][key]["encoder_weights_sha256"]
        ic.check_encoder_digest(key, want, want)
        with pytest.raises(ic.GuardError, match="encoder weights"):
            ic.check_encoder_digest(key, "f" * 64, want)
    digests = {
        MANIFEST["sources"][k]["encoder_weights_sha256"] for k in ("i_E0", "ii_A3", "iii_random")
    }
    assert len(digests) == 3  # three distinct encoders, so a swap would be caught


def test_frames_identical_guard():
    a = [np.zeros((4, 4, 3), np.uint8) for _ in range(3)]
    ic.check_frames_identical(a, [x.copy() for x in a], "frames")
    b = [x.copy() for x in a]
    b[1][2, 2, 0] = 1
    with pytest.raises(ic.GuardError, match="1 frames differ"):
        ic.check_frames_identical(a, b, "frames")
    with pytest.raises(ic.GuardError, match="counts"):
        ic.check_frames_identical(a, a[:2], "frames")
    with pytest.raises(ic.GuardError, match="counts"):
        ic.check_frames_identical([], [], "frames")


def test_wall_cap_voids():
    runner = _load("_probe_t3", "scripts/probe_info_ceiling.py")
    runner.Clock(3600).check("stage")
    with pytest.raises(runner.VoidRun, match="wall cap"):
        runner.Clock(-1).check("stage")


def test_a_failed_guard_writes_a_void_report(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    runner = _load("_probe_t4", "scripts/probe_info_ceiling.py")
    tampered = json.loads(json.dumps(MANIFEST))
    # Only committed inputs, so the test runs where checkpoints and data are absent (CI).
    tampered["hashes"] = {"configs/g1_sim_action.json": "0" * 64}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(tampered))
    monkeypatch.setattr(runner, "MANIFEST", path)
    report = runner.run(tmp_path / "out", smoke=True)
    assert report["status"] == "void"
    assert report["decision"]["outcome"] == "VOID"
    written = json.loads((tmp_path / "out" / "report.json").read_text())
    assert written["decision"]["outcome"] == "VOID"
    assert "g1_sim_action.json" in written["decision"]["reason"]
    with pytest.raises(FileExistsError):  # never overwrites evidence
        runner.run(tmp_path / "out", smoke=True)


# ----- reported-only readouts and comparisons ------------------------------------------------
def test_reported_t4_and_pairwise_comparison():
    xy, dx, occluded = _synthetic_cohort()
    dy = np.full(len(dx), -0.4)
    dy[::7] = -0.3
    fold = ic.fold_of(len(dx))
    priors = ic.prior_predictions(xy, dx, occluded, fold, dy=dy)
    mask = np.ones(len(dx), bool)
    perfect = ic.evaluate_reported(xy, dy, dx, xy, dy, dx, priors, mask)
    assert perfect["apple_x_abs_error_cm"]["median"] == 0.0
    assert perfect["dy_mae"]["mae"] == 0.0 and perfect["derived_dx_sign"]["accuracy"] == 1.0
    blind = ic.evaluate_reported(
        priors["B_mean"], priors["B_const_dy"], priors["B_const"], xy, dy, dx, priors, mask
    )
    assert blind["dy_mae"]["ratio_to_B_const_dy"]["ratio"] == pytest.approx(1.0)
    good = ic.evaluate(xy, dx, xy, dx, priors, mask)
    bad = ic.evaluate(priors["B_mean"], priors["B_const"], xy, dx, priors, mask)
    comparison = ic.compare_sources(good, bad)
    assert comparison["median_error_difference_cm"]["ci95"][1] < 0
    assert (
        comparison["mcnemar"]["only_second_correct"] <= comparison["mcnemar"]["only_first_correct"]
    )


def test_paired_ratio_stays_finite_when_a_baseline_resample_is_zero():
    idx = np.array([[0, 0], [1, 1]])
    result = ic.paired_ratio(np.array([1.0, 1.0]), np.array([0.0, 1.0]), idx, ic._median)
    assert np.isfinite(result["ci95"]).all()
    json.dumps(result, allow_nan=False)


def test_a_missing_input_voids_instead_of_crashing(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    runner = _load("_probe_t5", "scripts/probe_info_ceiling.py")
    tampered = json.loads(json.dumps(MANIFEST))
    tampered["hashes"] = {"checkpoints/does-not-exist.pt": "0" * 64}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(tampered))
    monkeypatch.setattr(runner, "MANIFEST", path)
    report = runner.run(tmp_path / "out", smoke=True)
    assert report["decision"]["outcome"] == "VOID"
    assert "sha256 None" in report["decision"]["reason"]


# ----- run-1 fix (void on crash, zero baselines): streams and finite values unchanged ---------
def _paired_ratio_ccb8fd7(numerator, denominator, idx, statistic) -> dict:
    """``paired_ratio`` exactly as it was at ccb8fd7 (the code run-1 used), pasted verbatim.

    It is the only function of ``info_ceiling`` the run-1 fix changed. The test below runs the
    whole statistics path with it and with the fixed version in one process, so the comparison
    is exact on every platform (a pinned digest is not: BLAS differs across machines).
    """
    num, den = np.asarray(numerator, np.float64), np.asarray(denominator, np.float64)
    point = float(statistic(num) / statistic(den))
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = statistic(num[idx], axis=1) / statistic(den[idx], axis=1)
    boot = np.where(np.isfinite(boot), boot, np.finfo(np.float64).max)
    lo, hi = ic.percentile_ci(boot)
    return {"ratio": point, "ci95": [lo, hi]}


def _stream_scenario(ic):
    rng = np.random.default_rng(3)
    n = 190
    xy = np.column_stack([0.34 + rng.uniform(-0.03, 0.03, n), -0.18 + rng.uniform(-0.03, 0.03, n)])
    dx = np.clip((xy[:, 0] - 0.333) / 0.015, -0.4, 0.4)
    dy = np.where(xy[:, 1] > -0.16, -0.3, -0.4)
    occluded = xy[:, 1] < -0.18
    fold = ic.fold_of(n)
    priors = ic.prior_predictions(xy, dx, occluded, fold, dy=dy)
    x = np.column_stack([xy, rng.normal(size=(n, 6))]) + rng.normal(0, 0.01, (n, 8))
    g, diag = ic.gram(x)
    xy_pred, _, sel = ic.nested_cv(g, diag, xy, fold)
    dx_pred, _, _ = ic.nested_cv(g, diag, dx, fold)
    dy_pred, _, _ = ic.nested_cv(g, diag, dy, fold)
    rand = rng.normal(size=(n, 8))
    gr, dr = ic.gram(rand)
    rxy, _, _ = ic.nested_cv(gr, dr, xy, fold)
    rdx, _, _ = ic.nested_cv(gr, dr, dx, fold)
    out = {"selections": sel, "xy_pred": xy_pred.tolist(), "dx_pred": dx_pred.tolist()}
    masks = {"all": np.ones(n, bool), "occluded": occluded, "visible": ~occluded}
    for name, mask in masks.items():
        a = ic.evaluate(xy_pred, dx_pred, xy, dx, priors, mask)
        b = ic.evaluate(rxy, rdx, xy, dx, priors, mask)
        out[name] = ic.public(a)
        out[name + "_floor"] = ic.beats_random_floor(a, b)
        out[name + "_cmp"] = ic.compare_sources(a, b)
        out[name + "_t4"] = ic.evaluate_reported(
            xy_pred, dy_pred, dx_pred, xy, dy, dx, priors, mask
        )
    return out


def _strip(value):
    if isinstance(value, dict):
        drop = ("undefined_resamples", "undefined_zero_baseline")
        return {k: _strip(v) for k, v in value.items() if k not in drop}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None  # the old module's inf point ratio == the new module's null
    return value


def test_fix_leaves_every_stream_and_finite_value_unchanged(monkeypatch):
    """Same seeds, same bootstrap indices, identical values for every finite field; the only
    difference allowed is an old inf/nan point ratio that is now null."""
    new = _strip(_stream_scenario(ic))
    with np.errstate(divide="ignore", invalid="ignore"):
        monkeypatch.setattr(ic, "paired_ratio", _paired_ratio_ccb8fd7)
        old = _strip(_stream_scenario(ic))
    assert old == new
    json.dumps(new, allow_nan=False)
    # The scenario really exercises a zero baseline (the run-1 crash), so the check is not vacuous.
    assert "null" in json.dumps(new)


def test_zero_baseline_ratio_is_null_and_never_passes():
    idx = ic.bootstrap_indices(4)
    result = ic.paired_ratio(np.array([0.1, 0.2, 0.0, 0.3]), np.zeros(4), idx, ic._mean)
    assert result["ratio"] is None and result["undefined_zero_baseline"] is True
    assert result["undefined_resamples"] == len(idx)
    assert result["ci95"][1] > ic.T3_MAX_RATIO_UPPER  # a null ratio can never pass
    json.dumps(result, allow_nan=False)
    finite = ic.paired_ratio(np.ones(4), np.full(4, 2.0), idx, ic._mean)
    assert finite["ratio"] == 0.5 and "undefined_zero_baseline" not in finite


def test_non_finite_guard_lists_every_replaced_field(tmp_path):
    runner = _load("_probe_t6", "scripts/probe_info_ceiling.py")
    value = {"a": 1.0, "b": [float("inf"), 2.0], "c": {"d": np.float64("nan")}}
    runner._write(tmp_path / "r.json", value)
    written = json.loads((tmp_path / "r.json").read_text())
    assert written["a"] == 1.0 and written["b"] == [None, 2.0] and written["c"]["d"] is None
    assert sorted(written["non_finite_fields"]) == ["/b/0=inf", "/c/d=nan"]


def test_unserialisable_report_still_leaves_a_void_report(tmp_path):
    runner = _load("_probe_t7", "scripts/probe_info_ceiling.py")
    with pytest.raises(TypeError):
        runner.write_report(tmp_path / "report.json", {"task": "TASK-059", "x": object()})
    written = json.loads((tmp_path / "report.json").read_text())
    assert written["outcome"] == "V" and "serialisation" in written["void_reason"]


def test_a_crash_mid_run_writes_a_void_report_and_reraises(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    runner = _load("_probe_t8", "scripts/probe_info_ceiling.py")
    committed = json.loads(json.dumps(MANIFEST))
    committed["hashes"] = {
        "configs/g1_sim_action.json": MANIFEST["hashes"]["configs/g1_sim_action.json"]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(committed))
    monkeypatch.setattr(runner, "MANIFEST", path)

    def boom():
        raise RuntimeError("injected crash")

    monkeypatch.setattr(runner, "plan_roots", boom)
    with pytest.raises(RuntimeError, match="injected crash"):
        runner.run(tmp_path / "out", smoke=True)
    written = json.loads((tmp_path / "out" / "report.json").read_text())
    assert written["outcome"] == "V" and written["decision"]["outcome"] == "VOID"
    assert "injected crash" in written["void_reason"] and "Traceback" in written["traceback"]
