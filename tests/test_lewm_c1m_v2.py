"""TASK-077's Stage-0 code (protocol ``docs/experiments/apple_lewm_c1m_v2.md`` §7 step 1)."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import lewm_c1m_v2 as lm
from embodied_jepa import lewm_c1m_v2_runtime as wrt
from embodied_jepa import lewm_c1m_v2_train as tr
from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa import lewm_next_c1m as c1m
from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa import run_tools as rt
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_lewm_c1m_v2.py"
MANIFEST = ROOT / lm.MANIFEST
DOC = ROOT / lm.DOCUMENT


def _load(path: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner():
    return _load(RUNNER, "_run_lewm_c1m_v2")


# ----- the frozen block, the manifest and the protocol ------------------------------------------
def test_manifest_frozen_block_equals_the_module():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["frozen"] == lm.frozen_block()
    assert manifest["frozen_sha256"] == lm.frozen_sha256()
    assert manifest["status"] == lm.STATUS


def test_the_frozen_sha_pin_is_set_only_at_the_freeze():
    manifest = json.loads(MANIFEST.read_text())
    if lm.STATUS == "DRAFT":
        assert manifest["frozen_sha256_pin"] is None
        assert lm.K0_MEASURED is None
    else:  # pragma: no cover - after the freeze
        assert manifest["frozen_sha256_pin"] == lm.frozen_sha256()


def test_carried_code_and_task076_pins_are_unchanged():
    out = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", lm.CARRIED_CODE_REFERENCE],
                         capture_output=True)  # fmt: skip
    if out.returncode != 0:
        pytest.skip("the carried-code reference is not in this clone")
    for path in lm.CARRIED_CODE_FILES:
        want = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", f"{lm.CARRIED_CODE_REFERENCE}:{path}"], text=True
        ).strip()
        got = subprocess.check_output(
            ["git", "-C", str(ROOT), "hash-object", path], text=True
        ).strip()
        assert want == got, path
    from embodied_jepa import plate_twin_v2_harness as hz

    manifest = json.loads((ROOT / lm.TASK076_MANIFEST).read_text())
    assert len(hz.check_pins(manifest["hashes"])) == 84


def test_the_protocol_carries_the_142_approval_fixes():
    text = DOC.read_text()
    assert "at U = 60 000, 22–38 h" in text and "22–41 h" not in text
    assert "allows one repeat per stage, and per job in Stage T (R17.16)" in text
    assert "every other pooled grid" in text and "including\ntrained readout heads" in text
    assert "trained readout heads" in lm.CLAUSE_RESULTS_WORDING


# ----- seeds and salts ---------------------------------------------------------------------------
def test_seed_block_ranges_and_salts_are_the_declared_ones():
    lm.check_seed_ranges()
    assert lm.SEED_BLOCK == (66000, 68999)
    assert lm.SEED_RANGES == {
        "K": (66000, 66031),
        "D": (66100, 66115),
        "S": (66200, 66263),
        "corpus": (67000, 68999),
    }
    assert lm.MODEL_SEEDS == (66800, 66801, 66802) and lm.CALIBRATION_SEED == 66810
    assert lm.DEBUG_SEEDS == (66900, 66999)
    assert sorted([*lm.SALTS.values(), *lm.RESERVED_SALTS.values()]) == list(range(8101, 8113))
    assert len(lm.seeds_of("K")) == lm.K_RESETS == 32
    assert len(lm.seeds_of("D")) == lm.D_RESETS == 16
    assert len(lm.seeds_of("S")) == lm.S_RESETS == 64
    assert len(lm.seeds_of("corpus")) == lm.CORPUS_ROOTS == 2000
    for name in ("c1m_block", "c1_block", "r15_block", "task076_block"):
        assert name in lm.FORBIDDEN_RANGES
    assert lm.FORBIDDEN_RANGES["r15_block"] == (58000, 58999)
    assert lm.FORBIDDEN_RANGES["c1m_block"] == (63000, 64999)
    for low, high in (*lm.SEED_RANGES.values(), lm.DEBUG_SEEDS):
        for f_low, f_high in lm.FORBIDDEN_RANGES.values():
            assert high < f_low or low > f_high
    assert not set(lm.SALTS.values()) & set(c1m.SALTS.values())


def test_seed_guards():
    assert lm.check_seeds("S", lm.seeds_of("S")) == lm.seeds_of("S")
    with pytest.raises(pt.GuardError):
        lm.check_seeds("S", lm.seeds_of("S")[:-1])
    with pytest.raises(pt.GuardError):
        lm.check_seeds("S", (63000,), debug=True)  # C1-M's block
    with pytest.raises(pt.GuardError):
        lm.check_seeds("S", (66200,), debug=True)  # not a debug seed
    with pytest.raises(pt.GuardError):
        lm.check_seeds("S", (np.int64(66900),), debug=True)
    assert lm.check_seeds("K", lm.seeds_of("K", debug=True), debug=True)
    assert lm.check_model_seed(66800) == 66800
    with pytest.raises(pt.GuardError):
        lm.check_model_seed(66992)  # a debug seed in a real run
    assert lm.check_model_seed(66992, debug=True) == 66992


def test_move_draw_corpus_aim_split_and_planted_error():
    for s in lm.seeds_of("K"):
        offset = lm.move_offset(s)
        assert np.linalg.norm(offset) <= 0.04 + 1e-12
        assert c1m._on_table(lm.moved_plate(s))
        # this task's own salt, not C1-M's
        assert lm.move_uniforms(s) != c1m.move_uniforms(s)
    a, b = lm.corpus_aim(67000)
    assert lm.A_LO <= a <= lm.A_HI and -lm.B_HALF_M <= b <= lm.B_HALF_M
    split = lm.corpus_split(lm.seeds_of("corpus"))
    assert {k: len(v) for k, v in split.items()} == lm.SPLIT_SIZES
    assert len(set().union(*map(set, split.values()))) == 2000
    assert split == lm.corpus_split(lm.seeds_of("corpus"))
    debug = lm.corpus_split(lm.seeds_of("corpus", debug=True), debug=True)
    assert {k: len(v) for k, v in debug.items()} == lm.DEBUG_SPLIT_SIZES
    e = np.asarray(lm.planted_error_m(66000, 2.0))
    assert np.isclose(np.linalg.norm(e), 0.02)
    assert np.allclose(lm.PLATE_MEAN, (0.49, -0.09))


def test_condition_is_c1m_as_recorded():
    assert lm.RULE == c1.RULE and lm.COMMIT_STEP == 405 and lm.MOVE_STEP == 300
    assert (lm.RHO_CM, lm.FAMILY, lm.A_LO, lm.READ_STEP, lm.HORIZON) == (4, "disc", -0.5, 465, 60)
    assert len(lm.GRID) == 147 and lm.GRID[0] == (-0.5, -0.03) and lm.GRID[-1] == (0.5, 0.03)
    assert lm.MODEL_CONFIG["token_grid"] == 8 and lm.MODEL_CONFIG["latent_dim"] == 24576
    assert lm.MODEL_CONFIG["max_horizon"] >= 60 and lm.BACKEND == "leworldmodel"
    assert lm.TRAIN_HORIZON == 60 and lm.BATCH_SIZE == 16
    assert (lm.N_FRAMES, lm.N_COMMANDS) == (65, 64)


# ----- the budget and selection ------------------------------------------------------------------
def test_the_budget_block_passes_check_budget_and_never_escalates():
    check = rt.check_budget(lm.BUDGET)
    assert check["consistent"] is True and check["problems"] == []
    assert check["max_request"] == 100_000 == check["min_cap"] == lm.BUDGET["cap"]
    assert lm.budget_rule(1_000, 3_000)["updates"] == 10_000
    assert lm.budget_rule(50_000, 2)["updates"] == 100_000
    assert lm.budget_rule(12_345, 30_001)["updates"] == 65_000
    for u in range(1, 50_001, 997):
        out = lm.budget_rule(u, 1)
        assert out["escalates"] is False and out["updates"] % 20 == 0
        assert 10_000 <= out["updates"] <= 100_000
    with pytest.raises(ContractError):
        lm.budget_rule(50_001, 1)
    with pytest.raises(ContractError):
        lm.budget_rule(lm.NOT_EVALUATED, 1)


def _calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return out


def test_selection_uses_select_checkpoint_and_last_two_never_a_raw_argmin():
    train = ROOT / "src/embodied_jepa/lewm_c1m_v2_train.py"
    calls = _calls(train)
    assert {"select_checkpoint", "last_two_triggered"} <= calls
    assert "argmin" not in calls  # the raw argmin is only reported (min over the curve)
    runner = _calls(RUNNER)
    assert {"select_checkpoint", "last_two_triggered"} <= runner
    text = train.read_text()
    assert "kept = rt.select_checkpoint(curve, lm.SELECTION_TOLERANCE)" in text
    assert "states[kept]" in text


# ----- the sentinel and the ladders -------------------------------------------------------------
def test_the_not_evaluated_sentinel_and_missing_inputs():
    assert lm.NOT_EVALUATED == "not evaluated"
    fields = lm.sentinel_fields(("a", "b"))
    assert fields == {"a": "not evaluated", "b": "not evaluated"}
    lm.check_sentinel(fields, set(), ("a", "b"))
    with pytest.raises(pt.GuardError):  # the early_verdict bug: a row in an unreached field
        lm.check_sentinel({"a": {"row": "L-PASS"}, "b": lm.NOT_EVALUATED}, set(), ("a", "b"))
    with pytest.raises(ContractError):
        lm.decide_k0(tau=lm.NOT_EVALUATED, ceiling=30, r_k=465, palm_speed_median_cm=0.0)
    with pytest.raises(ContractError):
        lm.decide_s(None)
    with pytest.raises(ContractError):
        lm.decide_o(c_plate={"ci95": [0, 0.5]}, prior_ratio={"ci95": [0, 0.5]},
                    hidden={"ci95": [2, 3]}, tau_commit_cm=None, falling=False)  # fmt: skip
    with pytest.raises(ContractError):
        lm.decide_d({"W": 14})
    with pytest.raises(ContractError):
        lm.decide_g({66800: {"G1": True}})


def test_the_runner_fields_start_as_the_sentinel_and_only_set_changes_them():
    run = _runner()
    report: dict = {}
    fields = run.Fields(report, "k0")
    assert all(v == lm.NOT_EVALUATED for v in report["fields"].values())
    fields.set("tau", {"tau_commit_cm": 1.0})
    fields.check()
    with pytest.raises(pt.GuardError):
        fields.set("ceiling", lm.NOT_EVALUATED)
    report["fields"]["decision"] = {"row": "K0-PASS"}  # written without set: refused
    with pytest.raises(pt.GuardError):
        fields.check()


def test_k0_rows():
    tau = lm.decide_tau({0.0: 32, 0.5: 31, 1.0: 29, 1.5: 27, 2.0: 20, 3.0: 10})
    assert tau["tau_commit_cm"] == 1.0
    ok = lm.decide_k0(tau=tau, ceiling=31, r_k=465, palm_speed_median_cm=0.003)
    assert ok["row"] == "K0-PASS" and not ok["clause_fires"]
    for kw, stop in (
        ({"ceiling": 29}, "ceiling_below_30"),
        ({"r_k": 466}, "r_late"),
        ({"r_k": None}, "r_late"),
        ({"palm_speed_median_cm": 0.51}, "palm_fast"),
    ):
        args = {"tau": tau, "ceiling": 31, "r_k": 465, "palm_speed_median_cm": 0.003} | kw
        out = lm.decide_k0(**args)
        assert out["row"] == "CAL-ESCALATE" and out["stops"][stop]
    bad = lm.decide_tau({0.0: 27, 0.5: 31, 1.0: 29, 1.5: 27, 2.0: 20, 3.0: 10})
    out = lm.decide_k0(tau=bad, ceiling=31, r_k=465, palm_speed_median_cm=0.0)
    assert out["stops"]["level0_below_bar"] and out["row"] == "CAL-ESCALATE"
    paths = [{t: (0.0 if t >= 460 else 1.0) for t in range(405, 526)} for _ in range(28)]
    assert lm.read_step_k(paths + [None] * 4) == 460
    assert lm.read_step_k(paths[:27] + [None] * 5) is None
    assert lm.read_step_k(paths[:4]) == 460 and lm.read_step_k(paths[:3] + [None]) is None


def test_corpus_o_t_g_d_rows():
    assert lm.decide_corpus(2000, 40)["row"] == "CORPUS-SEALED"
    assert lm.decide_corpus(2000, 41)["row"] == "CORPUS-ESCALATE"
    base = {"c_plate": {"ci95": [0.4, 0.6]}, "prior_ratio": {"ci95": [0.1, 0.3]},
            "hidden": {"ci95": [2.0, 3.0]}, "tau_commit_cm": 1.0, "falling": False}  # fmt: skip
    assert lm.decide_o(**base)["row"] == "O-PASS"
    assert lm.decide_o(**base | {"hidden": {"ci95": [0.9, 3.0]}})["row"] == "O-ARM-KEYED"
    assert lm.decide_o(**base | {"c_plate": {"ci95": [0.4, 1.1]}})["row"] == "O-NO-BAR"
    assert lm.decide_o(**base | {"prior_ratio": {"ci95": [0.5, 1.0]}})["row"] == "O-NO-BAR"
    assert lm.decide_t(rank_reference=0.09, jobs_complete=True)["row"] == "CAL-T-ESCALATE"
    assert lm.decide_t(rank_reference=0.10, jobs_complete=True)["row"] == "T-DONE"
    with pytest.raises(ContractError):
        lm.decide_t(rank_reference=0.3, jobs_complete=False)
    good = {"G1": True, "G2": True, "G3": True, "G4": True, "G5": True}
    seeds = {s: dict(good) for s in lm.MODEL_SEEDS}
    assert lm.decide_g(seeds)["row"] == "G-PASS"
    for gate in ("G1", "G2", "G3", "G4"):
        bad = {s: dict(good) for s in lm.MODEL_SEEDS}
        bad[66801][gate] = False
        bad[66802]["G5"] = False
        assert lm.decide_g(bad)["row"] == "H-GATE-FAIL"
    bad = {s: dict(good) for s in lm.MODEL_SEEDS}
    bad[66802]["G5"] = False
    assert lm.decide_g(bad)["row"] == "G-NO-BAR"
    d = {"W": 12, "N": 9, "L-shuf": 8, "L-mean": 9, "H-final": 14}
    assert lm.decide_d(d)["row"] == "D-PASS"
    assert lm.decide_d(d | {"W": 11})["row"] == "L-DEV-STOP"
    assert lm.decide_d(d | {"H-final": 13})["row"] == "L-DEV-STOP"
    assert lm.decide_d(d | {"L-mean": 10})["row"] == "L-DEV-STOP"
    g1 = lm.g1_bars(0.33, 0.79)
    assert g1["rank"]["bar"] == 0.16 and g1["std"]["bar"] == 0.39
    assert lm.g1_bars(0.1, 0.3)["rank"]["bar"] == 0.10
    assert lm.g1_bars(0.1, 0.3)["std"]["bar"] == 0.25


def _outcomes(w=60, n=10, shuf=10, mean=12, rand=8, rule=60, sysid=59, ceiling=63):
    def arm(k):
        return [True] * k + [False] * (64 - k)

    return {
        "W": arm(w),
        "N": arm(n),
        "L-shuf": arm(shuf),
        "L-mean": arm(mean),
        "L-rand": arm(rand),
        "H-rule": arm(rule),
        "H-sysid": arm(sysid),
        "H-final": arm(ceiling),
    }


def test_gated_rows_first_match_and_in_both_directions():
    assert lm.decide_s(None, void=True)["row"] == "V"
    out = lm.decide_s(_outcomes())
    assert out["row"] == "L-PASS" and not out["clause_fires"] and out["comparator"] == "H-rule"
    assert lm.decide_s(_outcomes(ceiling=55))["row"] == "S-VOID-CEILING"
    # a twin equal to W: the test fails and W is detectably no better -> the clause
    out = lm.decide_s(_outcomes(w=58, mean=58))
    assert out["row"] == "L-NO-GAIN" and out["clause_fires"]
    assert out["twins"]["L-mean"]["detectably_no_better"]
    # W far below the comparator: detectably inferior -> the clause
    out = lm.decide_s(_outcomes(w=40, rule=60, sysid=60, n=0, shuf=0, mean=0, rand=0))
    assert out["row"] == "L-INFERIOR" and out["clause_fires"]
    # a near miss of a twin test (upper bound >= +7) escalates
    out = lm.decide_s(_outcomes(w=60, mean=55, shuf=10))
    assert out["row"] == "L-TWIN-NEAR" and not out["clause_fires"]
    # G-NI misses without being detectably inferior
    out = lm.decide_s(_outcomes(w=56, rule=63, sysid=60))
    assert out["row"] == "L-NEAR" and not out["clause_fires"]
    # below the bar with G-NI and the tests passing
    out = lm.decide_s(_outcomes(w=55, rule=55, sysid=50, n=10, shuf=10, mean=10, rand=10))
    assert out["row"] == "L-BAR" and not out["clause_fires"]
    # the better comparator is chosen after S; a tie goes to H-rule
    assert lm.decide_s(_outcomes(rule=58, sysid=60))["comparator"] == "H-sysid"
    assert lm.decide_s(_outcomes(rule=60, sysid=60))["comparator"] == "H-rule"


def test_no_clause_row_fires_on_a_run_whose_tests_and_ni_pass():
    rng = np.random.default_rng(5)
    for _ in range(300):
        p = rng.uniform(0.3, 1.0, 8)
        o = {a: (rng.uniform(size=64) < q).tolist() for a, q in zip(_outcomes(), p, strict=True)}
        out = lm.decide_s(o)
        if out["row"] == "S-VOID-CEILING":
            continue
        all_pass = out["G-NI"]["passes"] and all(t["passes"] for t in out["twins"].values())
        if all_pass:
            assert not out["clause_fires"]
        if out["row"] == "L-TWIN-NEAR":
            assert all(t["miss_within_noise"] for t in out["twins"].values() if not t["passes"])


def test_mcnemar_minimum_separation_is_seven():
    assert pt.mcnemar_one_sided(7, 0) < 0.01 <= pt.mcnemar_one_sided(6, 0)
    smallest = min(
        b - c for b in range(65) for c in range(65 - b) if pt.mcnemar_one_sided(b, c) < 0.01
    )
    assert smallest == lm.MIN_SEPARATION == 7


def test_the_clause_rows_scope_and_never_closed():
    assert lm.CLAUSE_ROWS == frozenset({"L-NO-GAIN", "L-INFERIOR"})
    assert not set(lm.NO_CLAUSE_ROWS) & lm.CLAUSE_ROWS
    assert "rho* = 4 cm" in lm.CLAUSE_SCOPE and "pooled tokens" in lm.CLAUSE_SCOPE
    assert "the product goal" in lm.NEVER_CLOSED and "the LeWM backend" in lm.NEVER_CLOSED


# ----- statistics and the Stage-0 simulations ----------------------------------------------------
def test_the_vectorised_bootstrap_is_the_loop_estimator():
    rng = np.random.default_rng(0)
    a, b = rng.uniform(size=64) < 0.9, rng.uniform(size=64) < 0.8
    d = a.astype(int) - b.astype(int)
    loop = np.random.default_rng(8106)
    boot = np.array([d[loop.integers(0, 64, 64)].sum() for _ in range(2000)])
    assert lm.paired_interval(a, b, resamples=2000)["ci95"] == list(
        np.percentile(boot, [2.5, 97.5])
    )
    m = lm._counts_matrix(64, 2000)
    bounds = lm._intervals(d[None], m)
    assert np.allclose(bounds[0], np.percentile(boot, [2.5, 97.5]))


def test_the_simulations_behave():
    ni = lm.simulate_non_inferiority(p_c=31 / 32, gap=0, dependence="nested", share=1.0, trials=300)
    assert ni["single"]["g_ni_pass_rate"] > 0.9 and ni["single"]["l_inferior_rate"] == 0.0
    assert ni["max_of_two"] == ni["single"]  # share 1: one comparator
    far = lm.simulate_non_inferiority(
        p_c=30 / 32, gap=20, dependence="independent", share=0.5, trials=300
    )
    assert far["max_of_two"]["g_ni_pass_rate"] < 0.05
    assert far["max_of_two"]["l_inferior_rate"] > 0.5
    twin = lm.simulate_twin(p_w=60 / 64, advantage=30, reversed_pairs=0.0, trials=300, twins=4)
    assert twin["mcnemar_pass_rate"] > 0.95 and twin["any_twin_fires_rate"] < 0.05
    equal = lm.simulate_twin(p_w=60 / 64, advantage=0, reversed_pairs=1.0, trials=300)
    assert equal["l_no_gain_rate"] > 0.5
    with pytest.raises(ContractError):
        lm.simulate_non_inferiority(p_c=1.0, gap=0, dependence="other", share=0, trials=10)


def test_the_tau_curve_and_wrong_permutation():
    counts = {"0.0": 32, "1.0": 28, "3.0": 8}
    p = lm.tau_curve_probability([0.0, 0.5, 3.0, 9.0], counts)
    assert np.allclose(p, [1.0, 0.9375, 0.25, 0.25])
    perm = lm.wrong_permutation(250)
    assert sorted(perm) == list(range(250)) and not np.any(perm == np.arange(250))


# ----- W's controller form -----------------------------------------------------------------------
def _grid_setup():
    p_hat, h = np.array([0.49, -0.09]), np.array([0.40, -0.20])
    targets = np.asarray([c1.from_box(a, b, p_hat, h) for a, b in lm.GRID])
    chunks = np.zeros((len(targets), 60, 14), np.float32)
    chunks[:, 0, 0] = np.arange(len(targets)) / 1000.0  # identifies the candidate
    return p_hat, h, targets, chunks


def test_choose_from_grid_contracts_to_the_rule_and_logs():
    p_hat, h, targets, chunks = _grid_setup()
    plate = np.array([0.47, -0.08])
    kappa = lm.RULE["kappa"]
    by_index = dict(enumerate(targets))

    def chunk_of(g):
        c = np.zeros((60, 14), np.float32)
        c[0, 1] = 1.0  # a refinement chunk carries its aim in the next entries
        c[0, 2:4] = g
        return c, True

    def predict(commands):
        out = []
        for c in commands:
            g = c[0, 2:4] if c[0, 1] == 1.0 else by_index[int(round(c[0, 0] * 1000))]
            out.append(plate + kappa * (np.asarray(g) - h))  # the reactive plate
        return np.asarray(out)

    feasible = np.ones(len(targets), bool)
    g, log = wrt.choose_from_grid(
        "W", p_hat, h, targets, chunks, feasible, predict, chunk_of, tolerance_m=0.0025, seed=1
    )
    star = c1.clip_to_box(c1.fixed_point(plate, h), p_hat, h, lm.A_LO)
    assert log["converged"] and np.linalg.norm(g - star) < 0.0025
    assert log["rollouts"] <= 147 + lm.REFINE_MAX
    assert not log["fallback_all_infeasible"]


def test_choose_from_grid_ties_infeasible_lrand_and_fallback():
    p_hat, h, targets, chunks = _grid_setup()

    def constant(commands):
        return np.repeat(p_hat[None], len(commands), 0)

    def chunk_of(g):
        return np.zeros((60, 14), np.float32), False

    feasible = np.ones(len(targets), bool)
    feasible[:5] = False
    g, log = wrt.choose_from_grid(
        "N", p_hat, h, targets, chunks, feasible, constant, chunk_of, tolerance_m=0.0025, seed=1
    )
    d = np.linalg.norm(targets - p_hat, axis=1)
    d[:5] = np.inf
    assert log["grid_index"] == int(np.argmin(d))  # ties go to the lowest row-major index
    assert log["iterations"][0]["stopped"] == "infeasible_refinement"
    assert np.allclose(g, targets[log["grid_index"]])
    g2, log2 = wrt.choose_from_grid(
        "L-rand", p_hat, h, targets, chunks, feasible, None, chunk_of, tolerance_m=0.0, seed=7
    )
    assert log2["grid_index"] == lm.l_rand_index(7, feasible) and feasible[log2["grid_index"]]
    g3, log3 = wrt.choose_from_grid(
        "W", p_hat, h, targets, chunks, np.zeros(147, bool), constant, chunk_of,
        tolerance_m=0.0, seed=1,
    )  # fmt: skip
    assert log3["fallback_all_infeasible"] and np.allclose(g3, p_hat)


def test_world_model_arms_read_no_task_truth():
    """G-privileged, statically: W's aim reads the observation, the controller's own palm,
    kinematics, apple estimate and grasp, the stand-in and the readouts, never the hook or the
    simulator."""
    source = ast.get_source_segment(
        (ROOT / "src/embodied_jepa/lewm_c1m_v2_runtime.py").read_text(),
        next(
            n
            for n in ast.parse((ROOT / "src/embodied_jepa/lewm_c1m_v2_runtime.py").read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "WorldModelAim"
        ),
    )
    for forbidden in ("hook", ".sim", "truth", "robot", "current()"):
        assert forbidden not in source, forbidden
    assert lm.NON_PRIVILEGED_ARMS == frozenset(
        {"W", "N", "L-shuf", "L-mean", "L-rand", "H-rule", "H-sysid"}
    )
    assert lm.PRIVILEGED_ARMS >= {"H-final", "H-read", "H-now", "collect", "H-final-planted"}


def test_run_arm_refuses_task_truth_in_a_non_privileged_arm():
    run = _runner()

    class FakePool:
        def map(self, tasks, cap, what):
            return [{"seed": 1, "success": True, "blocked": None, "privileged_ok": True,
                     "task_truth_in_controller": 1}]  # fmt: skip

    with pytest.raises(pt.GuardError):
        run.run_arm(FakePool(), [{"arm": "W"}], "W")
    run.run_arm(FakePool(), [{"arm": "H-final"}], "H-final")  # privileged: allowed


# ----- storage and the prefetcher ----------------------------------------------------------------
def _store(n=4, d=8, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, lm.N_FRAMES, d)).astype(np.float32)
    a = rng.uniform(-0.1, 0.1, (n, lm.N_COMMANDS, 14)).astype(np.float32)
    return tr.RootStore(f, a, range(100, 100 + n))


def test_root_store_windows_are_contiguous_slices():
    s = _store()
    f, a = s.window(2, 3)
    assert f.shape == (61, 8) and a.shape == (60, 14)
    assert np.shares_memory(f, s.features) and f.flags.c_contiguous
    assert np.array_equal(f[0], s.features[2, 3]) and np.array_equal(a[-1], s.commands[2, 62])
    start, commands, targets = s.from_405()
    assert np.array_equal(start, s.features[:, 2]) and targets.shape == (4, 60, 8)
    assert np.array_equal(commands[:, 0], s.commands[:, 2])
    with pytest.raises(ContractError):
        s.window(0, 5)
    with pytest.raises(ContractError):
        tr.RootStore(s.features[:, :64], s.commands, s.roots)


def test_prefetcher_matches_the_sampler_and_zeroes_n(tmp_path):
    s = _store()
    direct = tr.WindowSampler(len(s), 66992)
    feeder = tr.Prefetcher(s, tr.WindowSampler(len(s), 66992))
    zero = tr.Prefetcher(s, tr.WindowSampler(len(s), 66992), zero=True)
    try:
        for _ in range(7):
            roots, offsets = direct.draw()
            f, a, r, o = feeder.next()
            fz, az, rz, oz = zero.next()
            assert np.array_equal(r, roots) and np.array_equal(o, offsets)
            assert np.array_equal(rz, roots) and not az.any()
            for i, (k, j) in enumerate(zip(roots, offsets, strict=True)):
                wf, wa = s.window(int(k), int(j))
                assert np.array_equal(f[i], wf) and np.array_equal(a[i], wa)
                assert np.array_equal(fz[i], wf)
    finally:
        feeder.close()
        zero.close()
    sha = tr.write_store(tmp_path, "train", s.features, s.commands, s.roots)
    again = tr.RootStore.open(tmp_path, "train", mmap=True)
    assert again.roots == s.roots and np.array_equal(again.features, s.features)
    assert set(sha) == {"features", "commands", "roots"}
    with pytest.raises(FileExistsError):
        tr.write_store(tmp_path, "train", s.features, s.commands, s.roots)
    mean, std = tr.moments(s, chunk=3)
    flat = s.features.reshape(-1, 8).astype(np.float64)
    assert np.allclose(mean, flat.mean(0)) and np.allclose(std, flat.std(0))
    fm, fs = tr.moments_file(tmp_path / "features8_train.npy", chunk=3)
    assert np.allclose(fm, mean) and np.allclose(fs, std)
    gather = tr.measure_gather(s, seed=66992, batches=3)
    assert gather["seconds_per_batch"] > 0


# ----- the runner --------------------------------------------------------------------------------
def test_the_runner_uses_run_tools_guards_and_loads_no_other_script():
    text = RUNNER.read_text()
    calls = _calls(RUNNER)
    for name in ("assert_local_import", "install_guards", "gpu_guard", "scale_probe",
                 "MemoryWatch", "check_pins", "check_evidence"):  # fmt: skip
        assert name in calls, name
    assert "require_lock=True" in text
    nri = _load(ROOT / "tests" / "test_no_runner_imports.py", "_nri")
    loaded, loaders = nri.script_loads(RUNNER)
    assert not loaded and not loaders


def test_the_scale_probe_calls_the_train_stages_own_function():
    tree = ast.parse(RUNNER.read_text())
    probes = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "scale_probe"
    ]
    assert len(probes) == 1 and probes[0].args[0].id == "train_job_core"
    stage_train = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stage_train"
    )
    assert any(
        isinstance(n, ast.Call) and getattr(n.func, "id", "") == "train_job_core"
        for n in ast.walk(stage_train)
    )


def test_the_runner_refuses_gated_stages_while_draft_and_misused_flags(tmp_path):
    run = _runner()
    assert set(run.DRAFT_ALLOWED) == {"tests", "simulate", "scale", "k0"}
    for argv in (
        ["k0", "--output", str(tmp_path / "a")],  # no evidence
        ["train", "--output", str(tmp_path / "b"), "--job", "cal-W"],  # no tests record
        ["simulate", "--output", str(tmp_path / "c"), "--workers", "2"],  # workers: debug only
        ["simulate", "--output", str(tmp_path / "d"), "--debug-skip-tests"],
        ["readouts", "--output", str(tmp_path / "e"), "--tau-commit-cm", "1.0"],
    ):
        with pytest.raises(SystemExit):
            run.main(argv)
    assert run.parse_job("cal-W", False) == {"arm": "W", "seed": 66810, "calibration": True}
    assert run.parse_job("N-66801", False)["seed"] == 66801
    with pytest.raises(pt.GuardError):
        run.parse_job("W-66992", False)
    with pytest.raises(pt.GuardError):
        run.parse_job("X-66800", False)


def test_g_tests_record_verification(tmp_path, monkeypatch):
    run = _runner()
    head = run.hz.revision()
    record = {
        "revision": head,
        "exit_status": 0,
        "summary": "1950 passed, 37 skipped in 150.00s",
        "finished_utc": "2999-01-01T00:00:00Z",
        "tracked_tree_dirty": False,
    }
    path = tmp_path / "tests.json"
    path.write_text(json.dumps({"fields": {"tests": record}}))
    assert run.verify_tests_record(path)["checks"]["revision"]
    for bad in (
        {"revision": "0" * 40},
        {"exit_status": 1},
        {"summary": "3 failed, 1947 passed"},
        {"finished_utc": "2000-01-01T00:00:00Z"},
    ):
        path.write_text(json.dumps({"fields": {"tests": record | bad}}))
        with pytest.raises(pt.GuardError):
            run.verify_tests_record(path)
    assert run.summary_ok("1950 passed, 37 skipped") and not run.summary_ok("1 error")


def test_scene_blind_reference_at_the_bar():
    run = _runner()
    ceiling = [True] * 31 + [False]
    proxy = [False] * 8 + [True] * 24
    ref = run.scene_blind_at_bar(ceiling, proxy, 28)
    assert sum(ref) == 28 and not any(ref[:3]) and all(ref[3:8])


def test_stage_modules_import_without_torch_or_mujoco():
    code = (
        "import sys\n"
        "import embodied_jepa.lewm_c1m_v2, embodied_jepa.lewm_c1m_v2_runtime\n"
        "import embodied_jepa.lewm_c1m_v2_offline, embodied_jepa.lewm_c1m_v2_train\n"
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


# ----- the training loop on the CPU (tiny encoder; needs the pinned LeWM source) ------------------
def test_train_model_selects_with_select_checkpoint(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not (ROOT / "third_party/le-wm/jepa.py").exists():
        pytest.skip("optional pinned LeWM source absent; run scripts/fetch_lewm.py")
    from embodied_jepa.models import frozen_tokens as ft

    _tiny_dinov2 = _load(ROOT / "tests" / "test_frozen_tokens.py", "_tft")._tiny_dinov2

    monkeypatch.setattr(ft.FrozenTokenMixin, "_load_frozen_module", lambda s, n: _tiny_dinov2())
    torch.set_num_threads(2)
    rng = np.random.default_rng(1)
    d = lm.LATENT_DIM
    f = (0.1 * rng.standard_normal((3, lm.N_FRAMES, d))).astype(np.float32)
    a = rng.uniform(-0.1, 0.1, (3, lm.N_COMMANDS, 14)).astype(np.float32)
    train = tr.RootStore(f, a, (1, 2, 3))
    val = tr.RootStore(f[:2].copy(), a[:2].copy(), (4, 5))
    calls = []
    real = rt.select_checkpoint

    def spy(curve, tolerance):
        calls.append((list(curve), tolerance))
        return real(curve, tolerance)

    monkeypatch.setattr(rt, "select_checkpoint", spy)
    model, record = tr.train_model(
        arm="N", seed=66992, train=train, val=val, mean=np.zeros(d), std=np.ones(d),
        updates=4, select_every=2, device="cpu", metadata={"protocol": lm.PROTOCOL},
        cap_seconds=600.0,
    )  # fmt: skip
    assert calls and calls[0][1] == 0.01
    assert record["kept_update"] == real(record["val_curve"], 0.01)
    assert record["prefetch"]["batches"] == 4 and len(record["val_curve"]) == 2
