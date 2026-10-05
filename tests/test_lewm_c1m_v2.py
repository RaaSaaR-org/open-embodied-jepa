"""TASK-077's Stage-0 code (protocol ``docs/experiments/apple_lewm_c1m_v2.md`` §7 step 1)."""

from __future__ import annotations

import ast
import json
import math
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


# The frozen block's sha256, set at the freeze (protocol §7 step 3, R17.25) after K0-PASS; the K0
# report is outputs/task077-k0-1/report.json, sha256 9be44fd9...f235 (``lm.K0_MEASURED``).
FROZEN_SHA256_PIN = "f28e5e2cd23d110f40ff043c7308e0bb9b3b71f46a2a4b6940bc536cc5e3548d"


def test_the_frozen_sha_pin_is_set_at_the_freeze():
    """The freeze pins the frozen block; from then on the module's block may not change."""
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["frozen_sha256_pin"] == FROZEN_SHA256_PIN == lm.frozen_sha256()
    assert manifest["status"] == lm.STATUS == "FROZEN"
    assert "**STATUS: FROZEN**" in DOC.read_text()
    assert lm.K0_MEASURED is not None and lm.K0_MEASURED["row"] == "K0-PASS"
    run = _runner()
    assert run.check_frozen_pin(manifest)["pin"] == FROZEN_SHA256_PIN


def test_the_freeze_pins_files_and_the_protocol_document():
    import hashlib

    manifest = json.loads(MANIFEST.read_text())
    want = {*lm.OWN_FILES, *lm.CARRIED_CODE_FILES, "tests/test_lewm_c1m_v2.py", lm.TASK076_MANIFEST}
    assert set(manifest["hashes"]) == want
    for relative, sha in manifest["hashes"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == sha, relative
    assert hashlib.sha256(DOC.read_bytes()).hexdigest() == manifest["protocol_document_sha256"]


def test_k0_measured_values_reproduce_k0s_decision_and_feasibility():
    """The frozen K0 values recompute: tau_commit, the stops and row, and each scene-blind
    proxy's McNemar feasibility (R17.25)."""
    m = lm.K0_MEASURED
    assert m["report_sha256"] == (
        "9be44fd93996c03cacc6cde089225f0f986676020f2ceb2138f062131f1af235"
    )
    assert m["revision"].startswith("306fbdc") and m["protocol_status_at_run"] == "DRAFT"
    seeds = lm.seeds_of("K")
    assert list(m["seeds"]) == [seeds[0], seeds[-1]]
    counts = m["tau"]["counts"]
    assert counts == {"0.0": 32, "0.5": 29, "1.0": 29, "1.5": 18, "2.0": 11, "3.0": 2}
    for level, failed in m["tau"]["failed_seeds"].items():
        assert set(failed) <= set(seeds) and len(seeds) - len(failed) == counts[level]
    tau = lm.decide_tau(counts)
    assert tau["tau_commit_cm"] == m["tau_commit_cm"] == m["tau"]["tau_commit_cm"] == 1.0
    ceiling = [s not in m["ceiling_failed_seeds"] for s in seeds]
    assert sum(ceiling) == m["n_k0"] == 32
    decision = lm.decide_k0(
        tau=tau,
        ceiling=m["n_k0"],
        r_k=m["r_k"],
        palm_speed_median_cm=m["history"]["palm_speed_405_cm_per_step"]["median"],
    )
    assert decision["row"] == m["row"] == "K0-PASS" and decision["stops"] == m["stops"]
    assert m["r_k"] <= lm.READ_STEP and lm.READ_STEP - m["r_k"] == 5  # the thin r margin
    assert [counts["0.5"] - lm.TAU_BAR, counts["1.0"] - lm.TAU_BAR] == [1, 1]  # thin tau margins
    run = _runner()
    for arm in lm.K0_PROXIES:
        item = m["proxies"][arm]
        outcome = [s in item["succeeded_seeds"] for s in seeds]
        assert sum(outcome) == item["count"]
        diff = lm.paired_interval(ceiling, outcome)
        assert [diff["difference"], diff["ci95"]] == item["headroom"]
        if arm in lm.K0_SCENE_BLIND:
            for key, ref in (
                ("feasibility_at_ceiling", ceiling),
                ("feasibility_at_w_bar", run.scene_blind_at_bar(ceiling, outcome, lm.TAU_BAR)),
            ):
                got = lm.mcnemar_feasibility(ref, outcome)
                want = item[key]
                assert (got["b"], got["c"]) == (want["b"], want["c"])
                assert got["predicted_pass_probability"] == want["predicted_pass_probability"]
            assert item["known_risk"] is False


def test_carried_code_and_task076_pins_are_unchanged():
    for path, blob in lm.CARRIED_CODE_BLOBS.items():
        got = subprocess.check_output(
            ["git", "-C", str(ROOT), "hash-object", path], text=True
        ).strip()
        assert got == blob, path
    known = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", lm.CARRIED_CODE_REFERENCE], capture_output=True
    )
    if known.returncode == 0:  # a full clone: the recorded blobs are the reference's
        for path, blob in lm.CARRIED_CODE_BLOBS.items():
            ref = f"{lm.CARRIED_CODE_REFERENCE}:{path}"
            want = subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", ref], text=True
            ).strip()
            assert want == blob, path
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
    for _ in range(120):
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
    """G-privileged, statically: W's aim and every helper it calls read the observation, the
    controller's own palm, kinematics, apple estimate and grasp, the stand-in and the readouts,
    never the hook, the simulator or the truth."""
    path = ROOT / "src/embodied_jepa/lewm_c1m_v2_runtime.py"
    text = path.read_text()
    names = {
        "WorldModelAim",
        "choose_from_grid",
        "encode",
        "rollout_plates",
        "world_model",
        "mean_latent",
        "run_offline_aim_task",
    }
    nodes = [n for n in ast.parse(text).body if getattr(n, "name", None) in names]
    assert {n.name for n in nodes} == names
    for node in nodes:
        source = ast.get_source_segment(text, node)
        for forbidden in ("hook", ".sim", "truth", "robot", "current()", "plate_xy_now"):
            assert forbidden not in source, (node.name, forbidden)
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
    assert sorted(ast.unparse(p.args[0]) for p in probes) == [
        "off.readouts_core",  # Stage O's readouts (R17.28): the readouts stage's own function
        "train_job_core",
    ]
    stage_readouts = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stage_readouts"
    )
    assert any(
        isinstance(n, ast.Call) and ast.unparse(n.func) == "off.readouts_core"
        for n in ast.walk(stage_readouts)
    )
    stage_train = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stage_train"
    )
    assert any(
        isinstance(n, ast.Call) and getattr(n.func, "id", "") == "train_job_core"
        for n in ast.walk(stage_train)
    )


def test_the_runner_refuses_gated_stages_while_draft_and_misused_flags(tmp_path):
    run = _runner()
    assert set(run.DRAFT_ALLOWED) == {"tests", "simulate", "scale", "k0", "oscale"}
    for argv in (
        ["k0", "--output", str(tmp_path / "a")],  # no evidence
        ["train", "--output", str(tmp_path / "b"), "--job", "cal-W"],  # no tests record
        ["simulate", "--output", str(tmp_path / "c"), "--workers", "2"],  # workers: debug only
        ["simulate", "--output", str(tmp_path / "d"), "--debug-skip-tests"],
        ["readouts", "--output", str(tmp_path / "e"), "--tau-commit-cm", "1.0"],
        ["oscale", "--output", str(tmp_path / "f")],  # no scratch
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
    # a reduced latent (2 x 2 grid, 1 536-d) keeps the test fast; the loop is the same code
    small = dict(lm.MODEL_CONFIG, token_grid=2, latent_dim=2 * 2 * 384)
    monkeypatch.setattr(lm, "MODEL_CONFIG", small)
    rng = np.random.default_rng(1)
    d = small["latent_dim"]
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


# ----- R17.20-R17.24: refused resets, determinism, the artifact chain, caps, rows --------------
def test_foreign_reached_and_reached_405():
    assert lm.foreign_reached(0, [True, True, True]) == 1
    assert lm.foreign_reached(2, [True, True, True]) == 0
    assert lm.foreign_reached(0, [True, False, True]) == 2
    assert lm.foreign_reached(2, [False, True, True]) == 1
    assert lm.foreign_reached(1, [False, True, False]) is None
    assert lm.reached_405({"frame405": np.zeros(1)}) and not lm.reached_405({"frame405": None})


def _commit(seed, target, reading, success=True):
    return {
        "seed": seed,
        "commit": {"target": list(target)},
        "decisions": [{"reading": list(reading)}],
        "termination_reason": "step_limit",
        "executed_steps": 800,
        "success": success,
    }


def test_determinism_check_tolerance_and_refused_resets():
    """R17.21 as amended by R17.27 and R17.38: the reading (0.1 cm), the success outcome and the
    commit target (0.6 cm) are gated; refusals must repeat identically."""
    a = [_commit(1, (0.5, -0.1), (0.49, -0.09)), {"seed": 2, "termination_reason": "guard_refusal",
                                                  "executed_steps": 230}]  # fmt: skip
    near = [_commit(1, (0.5004, -0.1), (0.49006, -0.09)), dict(a[1])]
    assert lm.DETERMINISM_TOLERANCE_M == 0.001
    out = lm.determinism_check(a, near)
    assert out["ok"] and out["resets"][1]["refused_identically"]
    assert out["rule"] == lm.DETERMINISM_RULE
    # a target 1.5 mm away (a flipped near-tie, or one refinement iterate less) is within the
    # derived 0.6 cm bound (R17.38)
    far_target = [_commit(1, (0.5015, -0.1), (0.49, -0.09)), dict(a[1])]
    out = lm.determinism_check(a, far_target)
    assert out["ok"] and out["resets"][0]["target_diff_m"] == pytest.approx(0.0015)
    assert out["max_target_diff_m"] == pytest.approx(0.0015)
    assert out["target_bound_m"] == lm.DETERMINISM_TARGET_BOUND_M == 0.006
    assert out["target_rule"] == lm.DETERMINISM_TARGET_RULE
    # the bound is the refinement argument's: 2 x (tau/4) at contraction 0.5, plus the reading
    # gate through 1 / (1 - kappa), rounded up to the next millimetre (tau_commit = 1.0 cm)
    q, tau = abs(lm.RULE["kappa"]), lm.K0_MEASURED["tau_commit_cm"] / 100.0
    derived = 2 * q / (1 - q) * lm.REFINE_FRACTION_OF_TAU * tau
    derived += lm.DETERMINISM_TOLERANCE_M / (1 - lm.RULE["kappa"])
    assert derived == pytest.approx(0.00567, abs=1e-5)
    assert lm.DETERMINISM_TARGET_BOUND_M == math.ceil(derived * 1000) / 1000
    # a target beyond 0.6 cm is V even with the same reading and outcome
    beyond = [_commit(1, (0.5061, -0.1), (0.49, -0.09)), dict(a[1])]
    out = lm.determinism_check(a, beyond)
    assert not out["ok"] and out["resets"][0]["target_diff_m"] == pytest.approx(0.0061)
    edge = [_commit(1, (0.5059, -0.1), (0.49, -0.09)), dict(a[1])]
    assert lm.determinism_check(a, edge)["ok"]
    # ... but not when the success outcome differs
    flipped = [_commit(1, (0.5015, -0.1), (0.49, -0.09), success=False), dict(a[1])]
    assert not lm.determinism_check(a, flipped)["ok"]
    # a reading beyond 0.1 cm is V even with the same target and outcome
    far_reading = [_commit(1, (0.5, -0.1), (0.4912, -0.09)), dict(a[1])]
    assert not lm.determinism_check(a, far_reading)["ok"]
    other = [near[0], dict(a[1], executed_steps=231)]
    assert not lm.determinism_check(a, other)["ok"]
    mixed = [near[0], _commit(2, (0.5, -0.1), (0.49, -0.09))]
    assert not lm.determinism_check(a, mixed)["ok"]


class _FakeCohort:
    def __init__(self, seeds):
        self.resets = {s: lm.reset_of(s) for s in seeds}
        self.est = {
            s: {"estimates": [0.3, -0.2, 0.5, -0.1], "frame_sha256": "f", "state_sha256": "s"}
            for s in seeds
        }


class _FakePool:
    """Every arm fails on ``refused`` (P-3's shared pick); elsewhere W, H-final and H-read
    succeed and the twins fail (an L-PASS-like cohort)."""

    def __init__(self, refused):
        self.refused, self.calls = set(refused), []

    def map(self, tasks, cap, what):
        self.calls.append(tasks)
        out = []
        for task in tasks:
            s, arm = task["seed"], task["arm"]
            base = {"seed": s, "arm": arm, "blocked": None, "privileged_ok": True,
                    "task_truth_in_controller": 0, "seconds": 1.0}  # fmt: skip
            if s in self.refused:
                out.append(
                    base
                    | {
                        "success": False,
                        "termination_reason": "guard_refusal",
                        "executed_steps": 230,
                        "frame405": None,
                        "decisions": [],
                    }
                )
                continue  # fmt: skip
            good = arm in ("W", "H-final", "H-read", "H-rule", "H-sysid")
            rec = _commit(s, (0.5, -0.1), (0.49, -0.09)) | base
            rec |= {"success": good, "frame405": np.full((2, 2, 3), s % 251, np.uint8)}
            rec["commit"] |= {"fallback": False, "clipped": False}
            out.append(rec)
        return out


@pytest.mark.parametrize("role", ["D", "S"])
def test_a_reset_refused_before_405_never_voids_d_or_s(role):
    run = _runner()
    n = 16 if role == "D" else 64
    seeds = tuple(range(66100, 66100 + n)) if role == "D" else tuple(range(66200, 66200 + n))
    refused = {seeds[1], seeds[2]}  # inside the determinism re-run's first four for S
    pool = _FakePool(refused)
    report: dict = {}
    fields = run.Fields(report, "closed")
    row = run.closed_core(
        pool, _FakeCohort(seeds), seeds, role, {"tau_commit_cm": 1.0, "a_lo": lm.A_LO}, [[0.0]],
        fields,
    )  # fmt: skip
    arms = report["fields"]["arms"]
    assert arms["refused_before_405"]["W"] == 2
    assert arms["W"]["count"] == n - 2 and arms["W"]["refused_before_405"] == 2
    # L-shuf: reset 0's foreign frame skips the two refused resets
    shuf = next(c for c in pool.calls if c[0]["arm"] == "L-shuf")
    assert int(shuf[0]["foreign_frame"][0, 0, 0]) == seeds[3] % 251
    assert arms["l_shuf_foreign"][str(seeds[0])] == seeds[3]
    if role == "S":
        det = report["fields"]["determinism"]
        assert det["ok"] and det["resets"][1]["refused_identically"]
        assert row == "L-PASS"  # 62/64 >= 56/64: the refused resets are counted misses
        assert report["fields"]["decision"]["counts"]["W"] == 62
    else:
        assert row == "D-PASS" and report["fields"]["determinism"] == lm.NOT_EVALUATED


def _chain_files(tmp_path, *, debug=False, row="O-PASS"):
    from embodied_jepa import lewm_c1m_v2_offline as off
    from embodied_jepa.models.latent_critic import RidgeReadout

    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "manifest.json").write_text("{}")
    sha = off.sha256_file(corpus / "manifest.json")
    feat = tmp_path / "featurise"
    (feat / "features").mkdir(parents=True)
    suffix = "-DEBUG" if debug else ""
    (feat / "report.json").write_text(json.dumps({
        "outcome": "FEATURISED" + suffix, "debug": debug, "corpus_sha256": sha,
        "fields": {"featurisation": {"files_sha256": {}}}}))  # fmt: skip
    fits = tmp_path / "readouts" / "fits"
    fits.mkdir(parents=True)
    mean, std = np.zeros(4), np.ones(4)
    np.savez(fits / "moments.npz", mean=mean, std=std)
    r = RidgeReadout(np.zeros(4), np.ones(4), np.ones((4, 2)), np.zeros(2), 0.1)
    np.savez(fits / "r8.npz", **r.state())
    np.savez(fits / "r_plate.npz", **r.state())
    latent = np.ones(4, np.float32)
    np.save(fits / "mean_latent.npy", latent)
    import hashlib

    record = {
        "moments": {"path": str(fits / "moments.npz"), "sha256": off.moments_sha256(mean, std)},
        "r8": {"path": str(fits / "r8.npz"), "sha256": r.sha256()},
        "r_plate": {"path": str(fits / "r_plate.npz"), "sha256": r.sha256()},
        "mean_latent": {"path": str(fits / "mean_latent.npy"),
                        "sha256": hashlib.sha256(latent.tobytes()).hexdigest()},
        "sysid": {"coef": [[0.0]]},
    }  # fmt: skip
    (fits.parent / "report.json").write_text(json.dumps({
        "outcome": row + suffix, "debug": debug, "corpus_sha256": sha,
        "fields": {"fits": record}}))  # fmt: skip
    from types import SimpleNamespace

    args = SimpleNamespace(corpus=str(corpus), corpus_sha256=sha, debug=debug,
                           features=str(feat / "features"), fits=str(fits))  # fmt: skip
    return args, record, sha


def test_the_artifact_chain_ties_corpus_features_fits_and_models(tmp_path):
    run = _runner()
    args, record, sha = _chain_files(tmp_path)
    report: dict = {}
    chain = run.artifact_chain(args, report)
    assert chain["corpus_sha256"] == sha == report["corpus_sha256"]
    mean, std, msha = run.load_moments(chain)
    assert msha == record["moments"]["sha256"]
    run.load_readout(chain, "r8")
    run.check_mean_latent(chain)
    job = {"arm": "W", "seed": 66800,
           "metadata": {"corpus_manifest_sha256": sha, "normalisation_sha256": msha}}  # fmt: skip
    assert run.check_job(job, chain) is job
    with pytest.raises(pt.GuardError):
        run.check_job(job | {"metadata": job["metadata"] | {"normalisation_sha256": "x"}}, chain)
    with pytest.raises(pt.GuardError):
        run.check_job(job | {"metadata": job["metadata"] | {"corpus_manifest_sha256": "x"}}, chain)
    # a tampered artifact is refused
    np.savez(record["moments"]["path"], mean=np.ones(4), std=np.ones(4))
    with pytest.raises(pt.GuardError):
        run.load_moments(chain)
    np.save(record["mean_latent"]["path"], np.zeros(4, np.float32))
    with pytest.raises(pt.GuardError):
        run.check_mean_latent(chain)
    from embodied_jepa.models.latent_critic import RidgeReadout

    other = RidgeReadout(np.zeros(4), np.ones(4), np.zeros((4, 2)), np.zeros(2), 0.1)
    np.savez(record["r8"]["path"], **other.state())
    with pytest.raises(pt.GuardError):
        run.load_readout(chain, "r8")
    # the wrong corpus, a missing sha in a real run
    with pytest.raises(pt.GuardError):
        run.artifact_chain(type(args)(**vars(args) | {"corpus_sha256": "0" * 64}), {})
    with pytest.raises(pt.GuardError):
        run.artifact_chain(type(args)(**vars(args) | {"corpus_sha256": None}), {})


def test_the_chain_refuses_debug_or_non_o_pass_reports_in_a_real_run(tmp_path):
    run = _runner()
    args, _r, _s = _chain_files(tmp_path / "a", row="O-NO-BAR")
    with pytest.raises(pt.GuardError):
        run.artifact_chain(args, {})
    args, _r, _s = _chain_files(tmp_path / "b", debug=True)
    with pytest.raises(pt.GuardError):  # a debug report in a real run
        run.artifact_chain(type(args)(**vars(args) | {"debug": False}), {})
    args, _r, _s = _chain_files(tmp_path / "c", debug=True, row="O-NO-BAR")
    assert run.artifact_chain(args, {})["fits"]  # any O row in a debug run


def _feature_split(folder, split, n=3):
    from embodied_jepa import lewm_c1m_v2_offline as off

    paths = off.feature_paths(folder, split)
    for key, path in paths.items():
        if path.suffix == ".json":
            path.write_text(json.dumps(list(range(n))))
        elif path.suffix == ".npz":
            np.savez(path, seeds=np.arange(n))
        else:
            np.save(path, np.full((n, 2), len(key), np.float32))
    return {k: off.sha256_file(p) for k, p in paths.items()}


def test_feature_files_are_checked_against_the_featurise_report(tmp_path):
    """The #143 approval's note 1 (R17.26): every file of a split a stage reads is hashed against
    the featurise report's files_sha256; a tampered, missing or unrecorded file is refused."""
    from embodied_jepa import lewm_c1m_v2_offline as off

    assert set(off.FEATURE_FILES) == {
        "features", "commands", "roots", "hidden8", "pool4", "full405", "table"
    }  # fmt: skip
    files = {s: _feature_split(tmp_path, s) for s in ("train", "val", "gate")}
    out = off.verify_feature_files(tmp_path, files, ("train", "val"))
    assert out["ok"] and out["files"] == 14 and out["splits"] == ["train", "val"]
    with pytest.raises(pt.GuardError):  # a split the report lacks
        off.verify_feature_files(tmp_path, {"train": files["train"]}, ("val",))
    with pytest.raises(pt.GuardError):  # an incomplete record
        off.verify_feature_files(tmp_path, {"val": {"features": "x"}}, ("val",))
    np.save(tmp_path / "features8_val.npy", np.zeros((3, 2), np.float32))  # tampered
    with pytest.raises(pt.GuardError):
        off.verify_feature_files(tmp_path, files, ("val",))
    assert off.verify_feature_files(tmp_path, files, ("train", "gate"))["ok"]
    (tmp_path / "table_gate.npz").unlink()  # missing
    with pytest.raises(pt.GuardError):
        off.verify_feature_files(tmp_path, files, ("gate",))


def test_the_chain_checks_the_feature_files_each_stage_reads(tmp_path):
    run = _runner()
    assert run.FEATURE_SPLITS == {
        "readouts": ("train", "val"),
        "train": ("train", "val"),
        "plan": ("val",),
        "gates": ("gate",),
        "oscale": ("train", "val"),
    }
    args, _record, _sha = _chain_files(tmp_path)
    feat = Path(args.features)
    files = {s: _feature_split(feat, s) for s in ("train", "val", "gate")}
    report_path = feat.parent / "report.json"
    rep = json.loads(report_path.read_text())
    rep["fields"]["featurisation"]["files_sha256"] = files
    report_path.write_text(json.dumps(rep))
    for stage, splits in run.FEATURE_SPLITS.items():
        report: dict = {}
        run.artifact_chain(type(args)(**vars(args) | {"stage": stage}), report)
        assert report["feature_files_verified"]["splits"] == list(splits)
    np.save(feat / "commands_gate.npy", np.zeros((3, 2), np.float32))  # tampered gate split
    run.artifact_chain(type(args)(**vars(args) | {"stage": "plan"}), {})  # plan reads val only
    with pytest.raises(pt.GuardError):
        run.artifact_chain(type(args)(**vars(args) | {"stage": "gates"}), {})


def test_stage_caps_frozen_pin_decide_g_and_stage_t_rows():
    run = _runner()
    from types import SimpleNamespace

    assert run.stage_cap(SimpleNamespace(stage="closed", cohort="D")) == 7_200.0
    assert run.stage_cap(SimpleNamespace(stage="closed", cohort="S")) == 21_600.0
    assert run.stage_cap(SimpleNamespace(stage="train", cohort=None)) == 46_800.0
    manifest = json.loads(MANIFEST.read_text())
    out = run.check_frozen_pin(manifest)
    assert out["pin"] == lm.frozen_sha256() and out["status"] == "FROZEN"
    with pytest.raises(pt.GuardError):
        run.check_frozen_pin(manifest | {"frozen_sha256_pin": "0" * 64})
    good = {"G1": True, "G2": True, "G3": True, "G4": True, "G5": True}
    with pytest.raises(ContractError):
        lm.decide_g({66992: good})  # a debug seed in the frozen ladder
    assert lm.decide_g({66992: good}, debug=True)["row"] == "G-PASS"
    assert "decide_t" in _calls(RUNNER)
    assert "stage_t" in run.STAGE_FIELDS["gates"]


def test_g_threads_pins_the_declared_thread_environment():
    """G-threads (R17.29): the runner sets exactly the declared environment before
    NumPy loads. OPENBLAS_NUM_THREADS = 16 is not a typo for 6: it is the owner ruling of
    2026-09-29 (TASK-073), OpenBLAS's default of one thread per logical CPU on the 16-thread PC,
    pinned so that it cannot drift, and carried unchanged by TASK-074, 075, 076 and C1-M."""
    from embodied_jepa import wm_critic_v2 as wc

    assert (
        lm.THREAD_ENV
        == pt.THREAD_ENV
        == wc.THREAD_ENV
        == {
            "MKL_DYNAMIC": "FALSE",
            "OMP_NUM_THREADS": "6",
            "MKL_NUM_THREADS": "6",
            "OPENBLAS_NUM_THREADS": "16",
        }
    )
    tree = ast.parse(RUNNER.read_text())
    first_numpy = min(
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        and any(a.name.split(".")[0] == "numpy" for a in n.names)
    )
    updates = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "update"
        and ast.unparse(n.func.value) == "os.environ"
    ]
    assert len(updates) == 1 and updates[0].lineno < first_numpy
    assert ast.literal_eval(updates[0].args[0]) == lm.THREAD_ENV


def test_the_stage_o_scale_probe_runs_the_readouts_on_its_synthetic_store(tmp_path):
    """R17.28: ``oscale``'s synthetic featurisation has the real file layout, passes the
    feature-file check and runs the readouts stage's own function (small sizes here)."""
    from embodied_jepa import lewm_c1m_v2_offline as off

    run = _runner()
    files = run.synthetic_featurisation(tmp_path / "feat", {"train": 25, "val": 10})
    assert off.verify_feature_files(tmp_path / "feat", files, ("train", "val"))["files"] == 14
    roots = [json.loads((tmp_path / "feat" / f"roots_{s}.json").read_text()) for s in files]
    assert roots == [list(range(25)), list(range(25, 35))]  # disjoint groups across splits
    out = off.readouts_core(tmp_path / "feat", tmp_path / "fits", tau_commit_cm=1.0, probe=True)
    assert out["rows"] == 35 and out["probe"] and out["decision"]["row"].startswith("O-")


# ----- Erratum 2026-10-05: the Stage C memory fix and the #144 approval's items (R17.30-R17.33) ---
def _small_readout(rows=60, dims=700, seed=3):
    from embodied_jepa import first_policy_perception as fpp

    rng = np.random.default_rng(seed)
    features = rng.standard_normal((rows, dims))
    targets = features[:, :4] * 0.01 + rng.standard_normal((rows, 4)) * 1e-3
    return fpp.XYReadout(features, targets), rng


def test_streamed_estimates_are_bit_identical_to_the_pinned_predict(monkeypatch):
    """R17.30: the streamed estimates equal ``XYReadout.predict`` bit for bit, and no
    ``cross_gram`` call sees more than one chunk of rows."""
    import inspect

    from embodied_jepa import info_ceiling as ic

    run = _runner()
    readout, rng = _small_readout()
    tokens = readout.features[rng.integers(0, 60, 300)] + 0.1 * rng.standard_normal((300, 700))
    reference = readout.predict(tokens)
    assert inspect.signature(ic.cross_gram).parameters["chunk"].default == run.CROSS_GRAM_BLOCK
    assert run.ESTIMATE_CHUNK % run.CROSS_GRAM_BLOCK == 0
    seen, original = [], ic.cross_gram

    def spy(new, reference_rows, **kw):
        seen.append(len(new))
        return original(new, reference_rows, **kw)

    monkeypatch.setattr(ic, "cross_gram", spy)
    for chunk in (32, 64, 128, 256, 320):
        seen.clear()
        got = run.streamed_estimates(
            readout, (tokens[lo : lo + chunk] for lo in range(0, len(tokens), chunk))
        )
        assert np.array_equal(got, reference), chunk
        assert max(seen) == min(chunk, 300) and sum(seen) == 300


def test_streamed_estimates_refuse_a_one_row_final_chunk():
    """R17.36 (the #145 approval's note 1): a final chunk of exactly one row after another chunk
    takes ``cross_gram``'s one-row norm path and is not bit-identical, so it is refused; a single
    chunk (K, D and S) and the corpus's tail of 80 at 128 are not affected."""
    from embodied_jepa import first_policy as fp

    run = _runner()
    readout, rng = _small_readout()
    tokens = readout.features[rng.integers(0, 60, 33)] + 0.1 * rng.standard_normal((33, 700))
    with pytest.raises(fp.GuardError, match="one-row final chunk"):
        run.streamed_estimates(readout, (tokens[lo : lo + 32] for lo in range(0, 33, 32)))
    one = run.streamed_estimates(readout, iter([tokens[:1]]))  # one chunk of one row: pinned path
    assert np.array_equal(one, readout.predict(tokens[:1]))
    assert 2000 % run.ESTIMATE_CHUNK == 80


def test_cohort_estimates_equal_the_pinned_function(monkeypatch):
    """R17.30: the runner's ``cohort_estimates`` returns the pinned function's dict exactly
    (frames, checks and disagreement record included), and refuses a chunk that is not a
    multiple of ``cross_gram``'s block."""
    from embodied_jepa import first_policy_perception as fpp
    from embodied_jepa import plate_twin_v2_harness as hz

    run = _runner()
    readout, _ = _small_readout()
    monkeypatch.setattr(
        fpp,
        "featurise",
        lambda encoder, frames: (
            np.asarray(frames, np.float64).reshape(len(frames), -1)[:, :700] / 255.0
        ),
    )

    class FramePool:
        def map(self, tasks, cap, what):
            out = []
            for t in tasks:
                frame = np.random.default_rng(t["seed"]).integers(0, 256, (112, 112, 3), np.uint8)
                out.append(
                    {
                        "seed": t["seed"],
                        "frame": frame,
                        "post_look_frame_sha256": f"f{t['seed']}",
                        "state_sha256": f"s{t['seed']}",
                        "truth_xy": [0.1, 0.2, 0.3, 0.4],
                    }
                )
            return out

    seeds = tuple(range(66900, 66900 + 70))
    resets = {s: {"object_xy": [0.0, 0.0], "plate_xy": [0.0, 0.0]} for s in seeds}
    pinned = hz.cohort_estimates(FramePool(), readout, None, seeds, resets, 10.0)
    for chunk in (32, 64, None):
        mine = run.cohort_estimates(FramePool(), readout, None, seeds, resets, 10.0, chunk=chunk)
        assert mine == pinned
    with pytest.raises(ValueError):
        run.cohort_estimates(FramePool(), readout, None, seeds, resets, 10.0, chunk=48)
    tree = ast.parse(RUNNER.read_text())
    calls = [
        ast.unparse(n.func)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("cohort_estimates")
    ]
    assert calls and set(calls) == {"cohort_estimates"}  # never the pinned, unstreamed one


def _preflight_args(run, stage: str, tmp_path):
    argv = [stage, "--output", str(tmp_path / "out")]
    if stage in run.SIM_STAGES:
        argv += ["--evidence", str(tmp_path / "evidence")]
    return run.build_parser().parse_args(argv)


def _quiet_preflight(monkeypatch, run):
    """Preflight with the git- and tree-dependent checks stubbed (the pins are read as is)."""
    monkeypatch.setattr(run, "check_code", lambda report: None)
    monkeypatch.setattr(run.hz, "tracked_tree_dirty", lambda: False)
    monkeypatch.setattr(run.hz, "revision", lambda: "0" * 40)
    monkeypatch.setattr(run.rt, "assert_local_import", lambda root, report: None)
    monkeypatch.setattr(run, "check_disk", lambda report, path, minimum_gib: None)


def test_frozen_preflight_checks_pins_and_the_protocol_document_and_refuses_k0(
    monkeypatch, tmp_path
):
    """The #144 approval's items 3 and 4: once FROZEN, preflight checks TASK-077's own pins and
    the protocol document's sha256 directly, refuses a tampered frozen pin, and refuses a
    non-debug K0 (R17.25); the top-level load average is named for when it is taken (R17.32)."""
    run = _runner()
    _quiet_preflight(monkeypatch, run)
    manifest = json.loads(MANIFEST.read_text())
    assert lm.STATUS == "FROZEN"

    class Stop(Exception):
        pass

    def stop():
        raise Stop

    with monkeypatch.context() as m:  # a FROZEN corpus run passes the frozen checks
        m.setattr(lm, "check_seed_ranges", stop)
        report: dict = {}
        with pytest.raises(Stop):
            run.preflight(report, _preflight_args(run, "corpus", tmp_path))
        assert report["own_pins_at_preflight"] == len(manifest["hashes"]) == 13
        assert report["protocol_document_check"]["sha256"] == manifest["protocol_document_sha256"]
        assert report["frozen_pin_check"]["pin"] == FROZEN_SHA256_PIN
    with pytest.raises(pt.GuardError, match="K0 ran once"):
        run.preflight({}, _preflight_args(run, "k0", tmp_path))
    with monkeypatch.context() as m:
        m.setattr(lm, "frozen_sha256", lambda: "0" * 64)
        with pytest.raises(pt.GuardError, match="G-frozen"):
            run.preflight({}, _preflight_args(run, "corpus", tmp_path))
    other = tmp_path / "protocol.md"
    other.write_text(DOC.read_text() + "\nedited\n")
    with monkeypatch.context() as m:
        m.setattr(run, "PROTOCOL_DOCUMENT", other)
        with pytest.raises(pt.GuardError, match="protocol document"):
            run.preflight({}, _preflight_args(run, "corpus", tmp_path))
    with pytest.raises(pt.GuardError, match="pin"):
        run.check_protocol_document({})
    report = {}
    run.preflight(report, run.build_parser().parse_args(["tests", "--output", "x"]))
    assert "load_average_after_preflight" in report and "load_average_at_start" not in report


def test_the_runner_log_option_creates_its_folder_and_refuses_an_existing_log(tmp_path):
    """--log writes the runner's stdout and stderr to a new file, creating its folder."""
    log_path = tmp_path / "outputs" / "run.log"
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('r', {str(RUNNER)!r})\n"
        "r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)\n"
        f"r.open_log(r.Path({str(log_path)!r}))\n"
        "print('to the log'); print('also', file=sys.stderr)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "" and log_path.read_text().splitlines() == ["to the log", "also"]
    again = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert again.returncode != 0 and "FileExistsError" in again.stderr


def test_featurise_corpus_remaps_its_stores_without_changing_a_byte(tmp_path):
    """R17.31: re-mapping the memory-mapped stores every few roots writes the same files as one
    mapping per split (a stand-in encoder; no weights)."""
    torch = pytest.importorskip("torch")
    import types

    from embodied_jepa import lewm_c1m_v2_offline as off

    class Encoder:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, pixel_values):
            p = torch.nn.functional.avg_pool2d(pixel_values, 14)
            t = p.flatten(2).transpose(1, 2).repeat(1, 1, 128)
            return types.SimpleNamespace(
                last_hidden_state=torch.cat([t.new_zeros((len(t), 1, 384)), t], dim=1)
            )

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rng = np.random.default_rng(0)
    entries = {}
    for seed in range(7):
        frames = rng.integers(0, 256, (lm.N_FRAMES, 112, 112, 3), np.uint8)
        entries[seed] = off.write_root(
            corpus,
            seed,
            {
                "frames": frames,
                "commands": rng.standard_normal((lm.N_COMMANDS, 14)).astype(np.float32),
                "plate": rng.standard_normal((lm.N_FRAMES, 2)),
                "palm": rng.standard_normal((lm.N_FRAMES, 2)),
                "hidden_r": frames[-1],
                "target": rng.standard_normal(2),
                "state405": rng.standard_normal(5),
                "apple_estimate": rng.standard_normal(2),
                "last_grasp": rng.standard_normal(14),
            },
        )
    off.seal_corpus(corpus, entries, {"train": [0, 1, 2, 3, 4], "val": [5], "gate": [6]}, {})
    manifest = off.open_corpus(corpus, None)
    assert off.FEATURE_REMAP_ROOTS == 32
    results = [
        off.featurise_corpus(
            corpus, manifest, tmp_path / name, device="cpu", encoder=Encoder(), remap_every=every
        )
        for name, every in (("every2", 2), ("one", None), ("default", off.FEATURE_REMAP_ROOTS))
    ]
    files = [r["files_sha256"] for r in results]
    assert files[0] == files[1] == files[2]
    # G-anchor checks the frozen block's 256 frames (R17.39), across the first roots
    assert lm.FEATURE_ANCHOR["frames"] == 256
    assert all(r["anchor"]["frames"] == 256 for r in results)


# ----- before Stage T (R17.41-R17.42) -------------------------------------------------------------
def test_the_chain_finds_stage_o_fits_by_name_in_the_folder_it_is_given(tmp_path, monkeypatch):
    """Stage O recorded its fits' paths relative to its own worktree (R17.42): a later stage in
    another worktree finds them by name in ``--fits``, and still checks every sha256."""
    run = _runner()
    args, record, _sha = _chain_files(tmp_path / "o")
    relative = {
        k: v | {"path": f"outputs/task077-readouts-1/fits/{Path(v['path']).name}"}
        if "path" in v
        else v
        for k, v in record.items()
    }
    report_path = Path(args.fits).parent / "report.json"
    stored = json.loads(report_path.read_text())
    stored["fields"]["fits"] = relative
    report_path.write_text(json.dumps(stored))
    elsewhere = tmp_path / "another-worktree"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # the recorded relative paths do not exist here
    report: dict = {}
    chain = run.artifact_chain(args, report)
    for name in ("moments", "r8", "r_plate", "mean_latent"):
        want = Path(args.fits).resolve() / Path(record[name]["path"]).name
        assert chain["fits"][name]["path"] == str(want)
    assert report["artifact_chain"]["fits_folder"] == str(Path(args.fits).resolve())
    run.load_moments(chain)
    run.load_readout(chain, "r8")
    run.check_mean_latent(chain)
    # the sha256 checks still bind on the file found by name
    np.savez(Path(args.fits) / "moments.npz", mean=np.ones(4), std=np.ones(4))
    with pytest.raises(pt.GuardError):
        run.load_moments(run.artifact_chain(args, {}))
    # a fit missing from the named folder is refused
    (Path(args.fits) / "r8.npz").unlink()
    with pytest.raises(pt.GuardError, match="r8.npz"):
        run.artifact_chain(args, {})


def test_a_job_checkpoint_is_found_beside_its_report(tmp_path, monkeypatch):
    run = _runner()
    job = tmp_path / "train-W-66800"
    job.mkdir()
    (job / "W-66800.pt").write_bytes(b"weights")
    record = {"arm": "W", "seed": 66800, "checkpoint": "outputs/train-W-66800/W-66800.pt"}
    (job / "report.json").write_text(
        json.dumps({"outcome": "T-JOB-DONE", "debug": False, "fields": {"record": record}})
    )
    monkeypatch.chdir(tmp_path)  # "outputs/..." does not exist here
    loaded = run.load_job(job / "report.json", False)
    assert loaded["checkpoint"] == str((job / "W-66800.pt").resolve())
    (job / "W-66800.pt").unlink()
    with pytest.raises(pt.GuardError):
        run.load_job(job / "report.json", False)


def test_gpu_memory_peak_reads_torchs_allocator_peaks(monkeypatch):
    """R17.41: a GPU stage's report records torch's peak allocated and reserved memory."""
    import types

    run = _runner()
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert run.gpu_memory_peak()["recorded"] is False
    gib = 2**30

    class Props:
        total_memory = 16 * gib

    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        current_device=lambda: 0,
        get_device_name=lambda d: "fake",
        max_memory_allocated=lambda d: 5 * gib,
        max_memory_reserved=lambda d: 6 * gib,
        get_device_properties=lambda d: Props(),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    peak = run.gpu_memory_peak()
    assert peak["recorded"] is True
    assert peak["max_memory_allocated_gib"] == 5.0 and peak["max_memory_reserved_gib"] == 6.0
    assert peak["total_gib"] == 16.0
    cuda.is_initialized = lambda: False
    assert run.gpu_memory_peak()["recorded"] is False


def test_a_gpu_stage_report_carries_gpu_memory_even_when_void(tmp_path):
    """The reading is taken in run()'s finally, for GPU stages only, so a V records it too."""
    out = tmp_path / "train-v"
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('r', {str(RUNNER)!r})\n"
        "r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)\n"
        "def boom(report, args): raise r.lp.GuardError('stop here')\n"
        "r.preflight = boom\n"
        f"args = r.build_parser().parse_args(['train', '--output', {str(out)!r}, '--job', 'cal-W',"
        " '--tests-record', 'x', '--corpus', 'x', '--corpus-sha256', 'x'])\n"
        "rep = r.run(args)\n"
        "print(rep['outcome'], 'gpu_memory' in rep)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split()[-2:] == ["V", "True"]
    stored = json.loads((out / "report.json").read_text())
    assert stored["outcome"] == "V" and "recorded" in stored["gpu_memory"]
    run = _runner()
    assert set(run.GPU_STAGES) == {"featurise", "train", "scale"}
