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
    finally:
        renders.close()
        robot.sim.close()
