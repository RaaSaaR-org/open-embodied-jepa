"""TASK-075 (``apple_obs_ceiling_v2``): the preregistered design, its guards and its mechanics.

NumPy tests run in the core job; tests that load the runner import-skip without torch and
MuJoCo; nothing here simulates a tau, corpus or smoke seed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import lewm_planner_v2 as lp
from embodied_jepa import obs_ceiling_v2 as oc
from embodied_jepa import obs_ceiling_v2_offline as off
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "benchmarks" / "manifests" / "apple-obs-ceiling-v2.json").read_text())
T074 = json.loads((ROOT / "benchmarks" / "manifests" / "apple-lewm-planner-v2.json").read_text())
FROZEN_SHA256 = "f6ed707cdca808e38e652f9dd2b742120bd17e74b1df5ef7c83223331c63d8a5"


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner(name: str):
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    return _load_script(name, "scripts/run_obs_ceiling_v2.py")


# ----- the frozen block, the pins and the carried facts -------------------------------------------
def test_manifest_frozen_block_equals_the_module():
    assert MANIFEST["frozen"] == oc.frozen_block()
    assert MANIFEST["frozen_sha256"] == oc.frozen_sha256() == FROZEN_SHA256


def test_pinned_files_match_and_task074_pins_are_unchanged():
    for relative, want in MANIFEST["hashes"].items():
        got = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert got == want, relative
    for relative, want in T074["hashes"].items():
        assert MANIFEST["hashes"][relative] == want, relative
    for own in (
        "src/embodied_jepa/obs_ceiling_v2.py",
        "src/embodied_jepa/obs_ceiling_v2_runtime.py",
        "src/embodied_jepa/obs_ceiling_v2_offline.py",
        "scripts/run_obs_ceiling_v2.py",
        "tests/test_obs_ceiling_v2.py",
        "src/embodied_jepa/hand_crop.py",
        "scripts/run_lewm_planner_v2.py",
    ):
        assert own in MANIFEST["hashes"], own


def test_the_protocol_document_sha_is_recorded():
    doc = ROOT / "docs" / "experiments" / "apple_obs_ceiling_v2.md"
    assert hashlib.sha256(doc.read_bytes()).hexdigest() == MANIFEST["protocol_document_sha256"]


def test_the_source_corpus_and_reference_facts_are_task074s():
    assert oc.SOURCE_CORPUS["frozen_sha256"] == lp.frozen_sha256() == T074["frozen_sha256"]
    assert oc.SOURCE_CORPUS["plan_sha256"] == T074["corpus_plan_sha256"]
    assert oc.SOURCE_CORPUS["split_sizes"] == lp.CORPUS_SPLITS
    assert oc.SOURCE_CORPUS["read_splits"] == lp.READ_SPLITS
    assert oc.SOURCE_CORPUS["name"] == lp.CORPUS == "apple-far-shift-v2"
    results = (ROOT / "docs" / "experiments" / "apple_lewm_planner_v2_results.md").read_text()
    assert "fe7ab9150216887f5759521dcc7fe7c7f54d2e5e8c36fecf6ed713a000b134bd" in results
    assert "2.872" in results and "4.351" in results and "3.123" in results
    ref = oc.TASK074_REFERENCE
    assert round(ref["encoded_median_cm"], 3) == 2.872
    assert round(ref["persistence_median_cm"], 3) == 4.351
    assert round(ref["clock_prior_median_cm"], 3) == 3.123
    assert (ref["windows"], ref["roots"], ref["r_off_rows"], ref["r_off_groups"]) == (
        90,
        22,
        5175,
        225,
    )
    assert oc.CONDITION["shift_cm"] == 9 and oc.CONDITION["shift_step"] == lp.SHIFT_STEP == 300


# ----- seeds --------------------------------------------------------------------------------------
def test_seed_ranges_are_fresh_and_disjoint():
    oc.check_seed_ranges()
    assert oc.SEED_RANGES == {"tau": (55000, 55031)}
    assert oc.SMOKE_SEEDS == (55050, 55099)
    assert len(oc.seeds_of("tau")) == oc.TAU["resets"] == 32
    for name, (low, high) in oc.FORBIDDEN_RANGES.items():
        assert high < 55000 or low > 55999, name


def test_role_seeds_are_guarded():
    assert oc.check_role_seeds("tau", oc.seeds_of("tau")) == oc.seeds_of("tau")
    with pytest.raises(oc.GuardError, match="G-seeds"):
        oc.check_role_seeds("tau", oc.seeds_of("tau")[:-1])
    with pytest.raises(oc.GuardError, match="forbidden"):
        oc.check_role_seeds("tau", (54200,), smoke=True)
    with pytest.raises(oc.GuardError, match="smoke"):
        oc.check_role_seeds("tau", (55000,), smoke=True)
    assert oc.check_role_seeds("tau", (55090, 55091), smoke=True) == (55090, 55091)


# ----- tau ----------------------------------------------------------------------------------------
def _counts(values):
    return dict(zip(oc.TAU["levels_cm"], values, strict=True))


def test_tau_is_the_largest_level_with_every_smaller_level_passing():
    assert oc.tau_from_counts(_counts([32, 32, 31, 30, 27, 29, 20, 5, 1])) == {
        "tau_cm": 1.5,
        "row": "TAU-MEASURED",
    }
    # a later recovery above the bar does not count once a level has failed
    assert oc.tau_from_counts(_counts([32, 32, 32, 32, 32, 32, 32, 32, 28]))["tau_cm"] == 5.0
    assert oc.tau_from_counts(_counts([28, 27, 32, 32, 32, 32, 32, 32, 32]))["tau_cm"] == 0.0
    assert oc.tau_from_counts(_counts([27, 32, 32, 32, 32, 32, 32, 32, 32])) == {
        "tau_cm": None,
        "row": "TAU-NONE",
    }
    string_keys = {str(k): v for k, v in _counts([32] * 9).items()}
    assert oc.tau_from_counts(string_keys)["tau_cm"] == 5.0
    with pytest.raises(ContractError):
        oc.tau_from_counts({0.0: 32})


def test_planted_error_has_the_level_and_one_direction_per_reset():
    for seed in (55000, 55017, 55031):
        unit = None
        for level in oc.TAU["levels_cm"][1:]:
            e = np.asarray(oc.planted_error_m(seed, level))
            assert np.isclose(np.linalg.norm(e), level / 100.0)
            u = e / np.linalg.norm(e)
            unit = u if unit is None else unit
            assert np.allclose(u, unit)
        assert oc.planted_error_m(seed, 0.0) == [0.0, 0.0]
    assert oc.tau_direction(55000) != oc.tau_direction(55001)
    assert oc.tau_direction(55000) == oc.tau_direction(55000)


def test_tau_measured_is_frozen_and_consistent():
    m = oc.TAU_MEASURED
    assert m is not None and MANIFEST["tau_calibration"] is not None
    assert oc.tau_from_counts(m["counts"])["tau_cm"] == m["tau_cm"]
    assert m["seeds"] == list(oc.SEED_RANGES["tau"])
    assert MANIFEST["tau_calibration"]["report_sha256"] == m["report_sha256"]


# ----- views and steps ----------------------------------------------------------------------------
def test_views_and_steps():
    assert oc.VIEWS == ("onboard112", "onboard224", "handcrop", "overview224")
    assert oc.ONBOARD_VIEWS == ("onboard112", "onboard224", "handcrop")
    assert oc.EXTRA_VIEWS == ("overview224",)
    assert oc.RENDERED_VIEWS == ("onboard224", "handcrop", "overview224")
    assert len(oc.RENDER_STEPS) == 30
    assert set(oc.TRAIN_STEPS) | set(oc.DECISION_STEPS) | set(oc.EVAL_STEPS) == set(oc.RENDER_STEPS)
    assert oc.TRAIN_STEPS[0] == 384 and oc.TRAIN_STEPS[-1] == 560 and len(oc.TRAIN_STEPS) == 23
    assert oc.EVAL_STEPS == (421, 437, 453, 469, 485, 501)
    assert set(oc.HIDDEN_STEPS) <= set(oc.RENDER_STEPS)
    assert all(t + oc.CHUNK in lp.O_STAR_CM for t in oc.DECISION_STEPS)


# ----- admission, rows, downstream ----------------------------------------------------------------
GOOD = {
    "c_upper": 1.0,
    "floor_ratio_upper": 0.9,
    "clock_ratio_upper": 0.5,
    "plate_hidden_lower": 3.0,
}
BAD_REP = {"c_upper": 9.0, "clock_ratio_upper": 1.2, "plate_hidden_lower": 0.1}


def test_admission_boundaries():
    tau = 2.0
    assert oc.admitted(GOOD, tau)["admitted"]
    assert oc.admitted(GOOD | {"c_upper": 2.0}, tau)["admitted"]  # c_V <= tau
    assert not oc.admitted(GOOD | {"c_upper": 2.0001}, tau)["admitted"]
    assert not oc.admitted(GOOD | {"floor_ratio_upper": 1.0}, tau)["admitted"]  # strict < 1
    assert not oc.admitted(GOOD | {"clock_ratio_upper": 1.0}, tau)["admitted"]
    assert not oc.admitted(GOOD | {"plate_hidden_lower": 2.0}, tau)["admitted"]  # strict > tau
    rep = {k: GOOD[k] for k in ("c_upper", "clock_ratio_upper", "plate_hidden_lower")}
    assert oc.representation_passes(rep, tau)["passes"]
    assert not oc.representation_passes(BAD_REP, tau)["passes"]


def _views(admitted=(), representation=()):
    bad_off = GOOD | {"c_upper": 9.0}
    return {
        v: {
            "r_off": GOOD if v in admitted else bad_off,
            "r_full": GOOD if v in representation else BAD_REP,
            "r_pix": BAD_REP,
        }
        for v in oc.VIEWS
    }


def test_rows_first_match_in_the_order_of_change():
    d = oc.decide(_views(admitted=("handcrop", "onboard224", "overview224")), 2.0)
    assert d["row"] == "OBS-ONBOARD" and d["chosen_view"] == "onboard224"
    assert (
        oc.decide(_views(admitted=("onboard112", "handcrop")), 2.0)["chosen_view"] == "onboard112"
    )
    d = oc.decide(_views(admitted=("overview224",)), 2.0)
    assert d["row"] == "OBS-EXTRA" and d["chosen_view"] == "overview224"
    d = oc.decide(_views(representation=("onboard112",)), 2.0)
    assert d["row"] == "OBS-REPRESENTATION" and not d["abandonment_clause_fires"]
    d = oc.decide(_views(), 2.0)
    assert d["row"] == "OBS-NONE" and d["abandonment_clause_fires"]
    assert oc.decide(_views(), None)["row"] == "TAU-NONE"
    assert oc.decide(_views(), 2.0, void=True)["row"] == "V"
    with pytest.raises(ContractError):
        oc.decide({"onboard112": _views()["onboard112"]}, 2.0)
    assert set(oc.ROW_CONSEQUENCES) == set(oc.ROWS)
    assert oc.CLAUSE_ROWS == ("OBS-NONE",)


def test_the_downstream_bar_lies_between_c_v_and_tau():
    assert oc.downstream_bar_interval(1.2, 2.0)["feasible"]
    assert not oc.downstream_bar_interval(2.5, 2.0)["feasible"]
    assert oc.bar_is_valid(1.5, 1.2, 2.0) and oc.bar_is_valid(1.2, 1.2, 2.0)
    assert not oc.bar_is_valid(1.0, 1.2, 2.0)  # TASK-074's error: a bar below the ceiling
    assert not oc.bar_is_valid(2.1, 1.2, 2.0)


# ----- folds and statistics -----------------------------------------------------------------------
def test_outer_folds_are_seeded_and_balanced():
    ids = [f"r{i}" for i in range(270)]
    f = oc.outer_folds(ids)
    assert f == oc.outer_folds(ids)
    sizes = np.bincount(list(f.values()))
    assert len(sizes) == oc.OUTER_FOLDS and sizes.max() - sizes.min() <= 1
    for fraction, want in ((0.25, 54), (0.5, 108), (1.0, 216)):
        assert len(oc.learning_subsample(ids[:216], fraction, 0)) == want
    assert oc.learning_subsample(ids[:216], 0.5, 0) != oc.learning_subsample(ids[:216], 0.5, 1)


def test_cluster_statistics():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(40), 4)
    a = rng.uniform(1, 2, 160)
    ci = oc.cluster_median_ci(a, clusters, resamples=500)
    assert ci["ci95"][0] <= ci["median"] <= ci["ci95"][1] and ci["clusters"] == 40
    r = oc.cluster_median_ratio(a, 2 * a, clusters, resamples=500)
    assert np.isclose(r["ratio"], 0.5) and np.allclose(r["ci95"], 0.5)
    d = oc.cluster_median_difference(a + 1, a, clusters, resamples=500)
    assert np.isclose(d["difference"], 1.0) and d["ci95"][0] > 0


# ----- offline pieces -----------------------------------------------------------------------------
def test_gram_is_x_xt_in_any_dtype():
    x = np.random.default_rng(1).integers(0, 255, (7, 5000), dtype=np.uint8)
    g, d = off.gram(x, block=1000)
    want = x.astype(np.float64) @ x.astype(np.float64).T
    assert np.allclose(g, want) and np.allclose(d, np.diag(want))


def test_kernel_fit_recovers_a_linear_target_and_reads_only_fit_rows():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(120, 6))
    y = x[:, :2] @ np.asarray([[1.0, 0.5], [-0.5, 2.0]]) + 0.01 * rng.normal(size=(120, 2))
    g, d = off.gram(x)
    fit = np.arange(100)
    groups = [f"root{i // 4}" for i in fit]
    poisoned = y.copy()
    poisoned[100:] = 1e6  # held rows' labels must never enter the fit
    readout, chosen = off.kernel_fit(g, d, fit, groups, poisoned)
    pred = off.kernel_predict(readout, g, d, np.arange(100, 120))
    assert np.abs(pred - y[100:]).max() < 0.2
    assert chosen["family"] in oc.KERNEL_FAMILIES and chosen["lam_rel"] in oc.KERNEL_LAMBDAS
    assert off.kernel_predict(readout, g, d, np.zeros(0, np.int64)).shape == (0, 2)


def test_the_dual_ridge_equals_the_primal_ridge():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(60, 20)).astype(np.float32)
    y = x[:, :2] * 3.0 + 0.1 * rng.normal(size=(60, 2))
    groups = [i // 3 for i in range(60)]
    dual = off.ridge(x, y, groups, dual=True)
    primal = off.ridge(x, y, groups, dual=False)
    assert dual.lam_rel == primal.lam_rel
    assert np.allclose(dual.predict(x), primal.predict(x), atol=1e-8)


def _synthetic_view(n_roots=15, seed=4):
    rng = np.random.default_rng(seed)
    roots, vis, hid, gram_index = [], {}, {k: {} for k in oc.HIDDEN_KINDS}, {}
    rows, hrows = [], {k: [] for k in oc.HIDDEN_KINDS}
    for i in range(n_roots):
        length = 600
        steps = np.arange(length)
        plate = np.stack([0.47 + 0.02 * rng.normal() + 0 * steps, -0.1 + 0 * steps], 1)
        apple = plate + np.stack([0.002 * (steps - 450), 0.001 * (steps - 450)], 1)
        roots.append(
            {
                "id": f"root{i}",
                "hi": 560,
                "offset": apple - plate,
                "plate": plate,
                "apple": apple,
                "states": {s: rng.normal(size=5) for s in oc.RENDER_STEPS},
                "fold": oc.outer_folds([f"root{j}" for j in range(n_roots)])[f"root{i}"],
            }
        )
        for s in oc.RENDER_STEPS:
            vis[(i, s)] = len(rows)
            rows.append(np.concatenate([roots[i]["offset"][s] * 50, rng.normal(size=6)]))
        for k in oc.HIDDEN_KINDS:
            for s in oc.HIDDEN_STEPS:
                hid[k][(i, s)] = len(hrows[k])
                hrows[k].append(rng.normal(size=8))
    pooled = np.asarray(rows, np.float32)
    hidden = {k: np.asarray(v, np.float32) for k, v in hrows.items()}
    all_rows = np.concatenate([pooled, *hidden.values()])
    for (i, s), r in vis.items():
        gram_index[("vis", i, s)] = r
    at = len(pooled)
    for k in oc.HIDDEN_KINDS:
        for (i, s), r in hid[k].items():
            gram_index[(k, i, s)] = at + r
        at += len(hidden[k])
    by_row = {r: key for key, r in gram_index.items()}
    y_all = np.asarray([roots[by_row[r][1]]["offset"][by_row[r][2]] for r in range(len(by_row))])
    g, d = off.gram(all_rows)
    return off.ViewData(
        roots,
        vis,
        hid,
        pooled,
        pooled + 0.5 * rng.normal(size=pooled.shape).astype(np.float32),
        hidden,
        gram_index,
        {"full": (g, d, y_all), "pix": (g, d, y_all)},
    )


def test_cross_fit_reads_every_moving_window_out_of_fold(monkeypatch):
    data = _synthetic_view()
    seen = []
    original = off.ridge

    def spy(x, y, groups, *, dual):
        seen.append(set(groups))
        return original(x, y, groups, dual=dual)

    monkeypatch.setattr(off, "ridge", spy)
    fit = off.cross_fit(data)
    wins = off.windows(data.roots)
    assert len(wins) == len(fit["clusters"]) > 0
    for key, values in fit["errors"].items():
        assert np.isfinite(values).all() and len(values) == len(wins), key
    # each outer fold's fits never see a held root: 6 ridge fits per fold, in fold order
    per_fold = len(seen) // oc.OUTER_FOLDS
    for k in range(oc.OUTER_FOLDS):
        held = {r["id"] for r in data.roots if r["fold"] == k}
        for groups in seen[k * per_fold : (k + 1) * per_fold]:
            assert not (set(map(str, groups)) & held)
    stats = off.view_statistics(fit, resamples=200)
    assert set(stats["subsets"]) == {"unshifted"}  # the synthetic roots are unshifted
    assert set(stats["r_off"]) == {
        "c_upper",
        "floor_ratio_upper",
        "clock_ratio_upper",
        "plate_hidden_lower",
    }
    assert stats["median_ci"]["r_off"]["median"] < stats["median_ci"]["r_floor"]["median"] + 5


def test_windows_follow_o2s_moving_rule_and_the_band():
    data = _synthetic_view(n_roots=3)
    root = data.roots[0]
    root["offset"] = np.zeros_like(root["offset"])
    root["offset"][437:, 0] = 0.02  # moves by 2 cm between 421 and 437 only
    wins = [t for i, t in off.windows(data.roots) if i == 0]
    assert wins == [421]
    root["hi"] = 440
    assert [t for i, t in off.windows(data.roots) if i == 0] == [421]
    root["hi"] = 436
    assert [t for i, t in off.windows(data.roots) if i == 0] == []


# ----- runtime pieces (NumPy level) ---------------------------------------------------------------
def test_planted_schedule_moves_only_the_target():
    from embodied_jepa import obs_ceiling_v2_runtime as ort

    def base(reset, shift):
        return lambda step: (
            np.asarray(reset["plate_xy"]) + (step >= 300) * np.asarray(shift["vector"])
        )

    make = ort.planted_schedule(base, [0.01, -0.02])
    at = make({"plate_xy": [0.5, -0.1]}, {"vector": [0.0, -0.09]})
    assert np.allclose(at(0), [0.51, -0.12]) and np.allclose(at(405), [0.51, -0.21])


def test_planted_check_refuses_a_target_that_is_not_plate_plus_error(monkeypatch):
    from embodied_jepa import obs_ceiling_v2_runtime as ort

    reset, shift = {"plate_xy": [0.5, -0.1]}, {"step": 300, "vector": [0.0, -0.09]}
    record = {
        "planted_m": [0.01, 0.0],
        "decisions": [
            {"step": 405, "target": [0.51, -0.19], "fallback": False},
            {"step": 421, "target": [0.0, 0.0], "fallback": True},
        ],
    }
    out = ort.planted_check(record, reset, shift)
    assert out["fallbacks"] == 1 and out["max_abs_target_error_m"] < 1e-12
    record["decisions"][0]["target"] = [0.5, -0.19]
    with pytest.raises(lp.GuardError, match="G-planted"):
        ort.planted_check(record, reset, shift)


def test_frame_check_applies_the_render_rule():
    from embodied_jepa import obs_ceiling_v2_runtime as ort

    a = np.zeros((3, 4, 4, 3), np.uint8)
    b = a.copy()
    b[1, 0, 0, 0] = 1  # one level, one pixel: within the rule
    b[2] = 9  # outside
    got = ort.frame_check(a, b)
    assert (got["identical"], got["within_rule"], got["outside_rule"]) == (1, 1, 1)


# ----- the runner ---------------------------------------------------------------------------------
def test_the_runner_reuses_task074s_guards_and_pins_threads():
    source = (ROOT / "scripts" / "run_obs_ceiling_v2.py").read_text()
    assert "class Pool(rg.BoundedPool)" in source
    assert 'L74 = _load("_task074_runner", "scripts/run_lewm_planner_v2.py")' in source
    for key, value in oc.THREAD_ENV.items():
        assert f'"{key}": "{value}"' in source
    assert "G-repro-off" in source and "G-repro-render" in source and "G-gpu" in source


def test_smoke_source_plan_uses_smoke_seeds_only():
    runner = _runner("_run_obs_ceiling_v2_a")
    plan = runner.smoke_source_plan()
    seeds = tuple(r["seed"] for r in plan)
    assert oc.check_role_seeds("smoke", seeds, smoke=True) == seeds
    splits = [r["split"] for r in plan]
    assert (splits.count("train"), splits.count("val"), splits.count("test")) == (10, 4, 2)
    assert runner.SMOKE_TAU_SEEDS[0] >= oc.SMOKE_SEEDS[0]


def test_the_disk_guard_needs_the_floor_plus_the_projected_store(tmp_path, monkeypatch):
    runner = _runner("_run_obs_ceiling_v2_b")

    class Usage:
        free = 12 * 2**30

    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _p: Usage)
    report = {}
    runner.check_disk(report, tmp_path, 270, 1.0e6)  # 10 GiB + 0.4 GB fits in 12 GiB
    with pytest.raises(lp.GuardError, match="G-disk"):
        runner.check_disk({}, tmp_path, 270, 1.0e7)  # 10 GiB + 4 GB does not


def test_the_tau_stage_is_refused_once_tau_is_frozen(tmp_path):
    runner = _runner("_run_obs_ceiling_v2_c")

    class Args:
        smoke = False

    with pytest.raises(lp.GuardError, match="G-spent"):
        runner.stage_tau({}, {}, tmp_path, None, Args)


def test_cli_refuses_misused_flags(tmp_path):
    runner = _runner("_run_obs_ceiling_v2_d")
    out = str(tmp_path / "x")
    for argv in (
        ["preflight", "--output", out, "--evidence", out, "--smoke"],
        ["tau", "--output", out, "--evidence", out, "--scale"],
        ["render", "--output", out, "--evidence", out, "--smoke", "--source", out, "--views", out],
        ["readouts", "--output", out, "--evidence", out, "--source", out],
    ):
        with pytest.raises(SystemExit):
            runner.main(argv)
