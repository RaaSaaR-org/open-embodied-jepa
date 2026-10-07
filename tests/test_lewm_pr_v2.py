"""TASK-080's Stage-0 code (protocol ``docs/experiments/apple_lewm_c1m_v2_pred_readout.md`` §8
step 1): seeds and salts, the reused artifacts, the readouts each arm reads, gate-P's aim
construction, the sentinel, the guards and every row ladder in both directions."""

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
from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa import lewm_pr_v2 as pr
from embodied_jepa import lewm_pr_v2_offline as pof
from embodied_jepa import lewm_pr_v2_runtime as prt
from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa import plate_twin_v2_harness as hz
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_lewm_pr_v2.py"
MANIFEST = ROOT / pr.MANIFEST
DOC = ROOT / pr.DOCUMENT


def _load(path: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner():
    return _load(RUNNER, "_run_lewm_pr_v2")


def _calls(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call):
            f = node.func
            out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return out


# ----- the frozen block, the manifest, TASK-077 carried ------------------------------------------
def test_manifest_frozen_block_equals_the_module():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["frozen"] == json.loads(json.dumps(pr.frozen_block(), ensure_ascii=False))
    assert manifest["frozen_sha256"] == pr.frozen_sha256()


# The frozen block's sha256, set at the freeze (protocol §8 step 3, R18.22) after K0′-PASS; the K0′
# report is outputs/task080-k0-1/report.json, sha256 a0939e3e...4c16 (``pr.K0_PRIME_MEASURED``).
FROZEN_SHA256_PIN = "0fc095dc947f0ac74ebf6d1c541098a1e6fec1897592256f97ac8962b97be064"


def test_the_frozen_sha_pin_is_set_at_the_freeze():
    """The freeze pins the frozen block; from then on the module's block may not change."""
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["frozen_sha256_pin"] == FROZEN_SHA256_PIN == pr.frozen_sha256()
    assert manifest["status"] == pr.STATUS == "FROZEN"
    assert "**STATUS: FROZEN**" in DOC.read_text()
    assert pr.K0_PRIME_MEASURED is not None and pr.K0_PRIME_MEASURED["row"] == "K0′-PASS"
    run = _runner()
    assert run.check_frozen_pin(manifest)["pin"] == FROZEN_SHA256_PIN
    assert run.check_protocol_document(manifest)["sha256"] == manifest["protocol_document_sha256"]


def test_the_freeze_pins_files_and_the_protocol_document():
    import hashlib

    manifest = json.loads(MANIFEST.read_text())
    want = {*pr.OWN_FILES, "tests/test_lewm_pr_v2.py", pr.TASK077_MANIFEST, lm.TASK076_MANIFEST}
    assert set(manifest["hashes"]) == want
    for relative, sha in manifest["hashes"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == sha, relative
    assert hashlib.sha256(DOC.read_bytes()).hexdigest() == manifest["protocol_document_sha256"]


def test_k0_prime_measured_values_reproduce_k0_primes_decision():
    """The frozen K0′ values recompute: the per-level counts from the failed seeds, the pooled
    tau_commit over K and K′, the stops and the row (R18.22)."""
    m = pr.K0_PRIME_MEASURED
    assert m["report_sha256"] == (
        "a0939e3eea7b695d1c3029e733970bcd123ea62db76192104c69479ad9314c16"
    )
    assert m["revision"].startswith("931281a") and m["protocol_status_at_run"] == "DRAFT"
    seeds = pr.seeds_of("K")
    assert list(m["seeds"]) == [seeds[0], seeds[-1]] == [65000, 65031]
    counts = m["counts_k_prime"]
    assert counts == {"0.0": 32, "0.5": 29, "1.0": 31, "1.5": 14, "2.0": 18, "3.0": 5}
    for level, failed in m["failed_seeds"].items():
        assert set(failed) <= set(seeds) and len(seeds) - len(failed) == counts[level]
    assert sum(s not in m["ceiling_failed_seeds"] for s in seeds) == m["n_k_prime_0"] == 32
    decision = pr.decide_k0_prime(
        k_prime_counts=counts,
        ceiling=m["n_k_prime_0"],
        r_k=m["r_k_prime"],
        palm_speed_median_cm=m["history"]["palm_speed_405_cm_per_step"]["median"],
    )
    assert decision["row"] == m["row"] == "K0′-PASS" and decision["stops"] == m["stops"]
    assert decision["pooled"] == m["pooled"]
    assert m["pooled"]["counts_k"] == pr.K_COUNTS
    assert m["pooled"]["counts_pooled"] == {
        "0.0": 64, "0.5": 58, "1.0": 60, "1.5": 32, "2.0": 29, "3.0": 7,
    }  # fmt: skip
    assert decision["tau_commit_cm"] == m["tau_commit_cm"] == 1.0
    assert decision["tightened_below_task077"] is m["tightened_below_task077"] is False
    assert m["r_k_prime"] <= pr.R_LATEST and pr.R_LATEST - m["r_k_prime"] == 6  # r's margin
    pooled = m["pooled"]["counts_pooled"]
    assert [pooled["0.5"] - pr.TAU_BAR_POOLED, pooled["1.0"] - pr.TAU_BAR_POOLED] == [2, 4]
    run = _runner()
    assert run.tau_curve() == (m["pooled"]["counts_pooled"], pr.POOLED_RESETS)


def test_task077_is_carried_unchanged():
    """G-frozen: TASK-077's frozen block is its pin and its 13 pinned files are unchanged."""
    manifest = json.loads((ROOT / pr.TASK077_MANIFEST).read_text())
    assert lm.frozen_sha256() == pr.TASK077_FROZEN_SHA256 == manifest["frozen_sha256_pin"]
    assert len(hz.check_pins(manifest["hashes"], ROOT)) == 13
    assert set(pr.OWN_FILES).isdisjoint(manifest["hashes"])
    assert pr.MODEL_SEEDS == lm.MODEL_SEEDS and pr.PRIMARY_SEED == 66800
    assert pr.REUSED["k0_report_sha256"] == lm.K0_MEASURED["report_sha256"]
    assert sorted(pr.REUSED["models"]) == sorted(
        f"{a}-{s}" for a in ("W", "N") for s in pr.MODEL_SEEDS
    )
    for key in ("moments", "r8", "r_plate", "mean_latent", "sysid_file"):
        assert len(pr.REUSED[key]) == 64
    assert pr.OLD_CORPUS_SHA256.startswith("ad8974b2") and pr.OLD_CORPUS_SHA256.endswith("43fb")


def test_seed_block_ranges_and_salts():
    pr.check_seed_ranges()
    assert pr.SEED_RANGES == {
        "K": (65000, 65031),
        "D": (65100, 65115),
        "S": (65200, 65263),
        "corpus": (65300, 65799),
    }
    assert pr.DEBUG_SEEDS == (65900, 65999)
    assert pr.FORBIDDEN_RANGES["task077_block"] == (66000, 68999)
    for name, span in lm.FORBIDDEN_RANGES.items():
        assert pr.FORBIDDEN_RANGES[name] == span
    assert sorted([*pr.SALTS.values(), *pr.RESERVED_SALTS.values()]) == list(range(8201, 8213))
    assert pr.SALTS["l_rand"] == 8209 and pr.SALTS["bootstrap"] == 8206
    assert len(pr.seeds_of("S")) == 64 and len(pr.seeds_of("D")) == 16
    assert len(pr.seeds_of("K")) == 32 and len(pr.seeds_of("corpus")) == 500


def test_seed_guards_and_g_fresh():
    assert pr.check_seeds("K", pr.seeds_of("K")) == pr.seeds_of("K")
    with pytest.raises(pt.GuardError):
        pr.check_seeds("K", pr.seeds_of("K")[:-1])
    with pytest.raises(pt.GuardError):
        pr.check_seeds("S", (66200,), debug=True)  # TASK-077's never-simulated S: forbidden
    with pytest.raises(pt.GuardError):
        pr.check_seeds("D", (65100,), debug=True)  # a debug run simulates 65900-65999 only
    with pytest.raises(pt.GuardError):
        pr.check_seeds("D", (np.int64(65100),))
    assert pr.check_seeds("D", (65910, 65911), debug=True) == (65910, 65911)
    pr.check_fit_roots([67000, 67001, 68999])  # the old corpus's roots
    for bad in ([67000, 65300], [65941], [65000]):
        with pytest.raises(pt.GuardError):
            pr.check_fit_roots(bad)
    with pytest.raises(pt.GuardError):
        pof.fit_readouts({"S": np.zeros((2, 3))}, np.zeros((2, 2)), [65310, 67000])


def test_move_corpus_aim_split_and_planted_error():
    seed = 65200
    assert not np.allclose(pr.move_offset(seed), lm.move_offset(seed))  # salt 8201, not 8101
    assert np.allclose(pr.move_offset(seed), pr.move_offset(seed))
    assert np.linalg.norm(pr.move_offset(seed)) <= pr.RHO_CM / 100.0 + 1e-12
    for s in (65300, 65301, 65799):
        a, b = pr.corpus_aim(s)
        assert pr.A_LO <= a <= pr.A_HI and -pr.B_HALF_M <= b <= pr.B_HALF_M
    split = pr.corpus_split(pr.seeds_of("corpus"))
    assert [len(split[h]) for h in pr.HALVES] == [250, 250]
    assert set(split["gate_p"]).isdisjoint(split["contrast_t"])
    assert split == pr.corpus_split(pr.seeds_of("corpus"))
    assert pr.corpus_split(pr.seeds_of("corpus", debug=True), debug=True)["gate_p"]
    e = pr.planted_error_m(65000, 1.5)
    assert abs(np.linalg.norm(e) - 0.015) < 1e-12
    assert not np.allclose(e, lm.planted_error_m(65000, 1.5))


# ----- statistics -------------------------------------------------------------------------------
def test_the_estimators_are_task077s_with_salt_8206():
    rng = np.random.default_rng(0)
    a, b = rng.uniform(0, 2, 40), rng.uniform(0.5, 3, 40)
    s, t = rng.uniform(size=30) < 0.8, rng.uniform(size=30) < 0.6
    assert pr.median_ci(a, salt=8106) == lm.median_ci(a)
    assert pr.median_ratio_ci(a, b, salt=8106) == lm.median_ratio_ci(a, b)
    assert pr.median_difference_ci(a, b, salt=8106) == lm.median_difference_ci(a, b)
    assert pr.paired_interval(s, t, salt=8106) == lm.paired_interval(s, t)
    assert not np.array_equal(pr.bootstrap_index(40, 50), lm.bootstrap_index(40, 50))  # 8206
    assert np.array_equal(pr.bootstrap_index(5, 3), pr.bootstrap_index(5, 3, salt=8206))
    d = pr.unpaired_median_difference_ci(a, b)
    assert d["ci95"][0] <= d["difference"] <= d["ci95"][1]
    perm = pr.wrong_permutation(50)
    assert not np.any(perm == np.arange(50)) and sorted(perm) == list(range(50))
    sub = pr.nested_subsets(40)
    assert set(sub[0.25]) <= set(sub[0.5]) <= set(sub[0.75]) <= set(sub[1.0])
    assert len(sub[1.0]) == 40


def test_power_reproduces_the_drafts_table_and_g_bar():
    rows = pr.power_table(trials=40_000)
    for row in rows:
        want = pr.POWER_TABLE_DRAFT[(row["p_w"], row["p_t"])]
        got = (row["nested"], row["half"], row["independent"])
        assert np.allclose(got, want, atol=0.02), (row, want)
    assert round(pr.g_bar_power(0.875), 2) == 0.59 and round(pr.g_bar_power(0.9375), 2) == 0.98
    assert pr.discordant_probabilities(0.9, 0.7, "nested") == pytest.approx((0.2, 0.0))
    with pytest.raises(ContractError):
        pr.discordant_probabilities(0.9, 0.7, "other")


def test_the_tau_curve_maps_aim_errors():
    p = pr.tau_curve_probability([0.0, 0.25, 1.0, 5.0], pr.K_COUNTS, pr.K_RESETS)
    assert p[0] == 1.0 and p[1] == pytest.approx((32 + 29) / 2 / 32) and p[3] == 2 / 32
    assert pr.predicted_count([0.0, 0.0], pr.K_COUNTS, pr.K_RESETS) == 64.0


# ----- the row ladders, every row in both directions ---------------------------------------------
def _levels(values):
    return {str(k): v for k, v in zip(pr.TAU_LEVELS_CM, values, strict=True)}


def test_k0_prime_pooled_tau_and_stops():
    ok = dict(k_prime_counts=_levels([32, 30, 29, 20, 10, 3]), ceiling=32, r_k=458,
              palm_speed_median_cm=0.01)  # fmt: skip
    d = pr.decide_k0_prime(**ok)
    assert d["row"] == "K0′-PASS" and d["tau_commit_cm"] == 1.0 and not d["clause_fires"]
    assert d["pooled"]["counts_pooled"]["1.0"] == 58
    tight = pr.decide_k0_prime(**(ok | {"k_prime_counts": _levels([32, 30, 26, 20, 10, 3])}))
    assert tight["row"] == "K0′-PASS" and tight["tau_commit_cm"] == 0.5
    assert tight["tightened_below_task077"]
    for change, stop in (
        ({"k_prime_counts": _levels([27, 27, 27, 20, 10, 3]), "ceiling": 27}, "level0_below_bar"),
        ({"ceiling": 29}, "ceiling_below_30"),
        ({"r_k": 466}, "r_late"),
        ({"r_k": None}, "r_late"),
        ({"palm_speed_median_cm": 0.6}, "palm_fast"),
    ):
        d = pr.decide_k0_prime(**(ok | change))
        assert d["row"] == "CAL-ESCALATE" and d["stops"][stop], change
    undefined = pr.pooled_tau(_levels([23, 0, 0, 0, 0, 0]), _levels([32, 0, 0, 0, 0, 0]))
    assert undefined["tau_commit_cm"] is None
    with pytest.raises(ContractError):
        pr.decide_k0_prime(**(ok | {"ceiling": pr.NOT_EVALUATED}))


def test_corpus_rows_overall_and_per_half():
    split = pr.corpus_split(pr.seeds_of("corpus"))
    assert pr.decide_corpus(split, [])["row"] == "CORPUS-SEALED"
    five = split["gate_p"][:5]  # 2 % of a half: not above it
    assert pr.decide_corpus(split, five)["row"] == "CORPUS-SEALED"
    six = split["gate_p"][:6]  # 2.4 % of gate-P, 1.2 % overall
    d = pr.decide_corpus(split, six)
    assert d["row"] == "CORPUS-ESCALATE" and d["fractions"]["all"] < 0.02
    with pytest.raises(ContractError):
        pr.decide_corpus(split, pr.NOT_EVALUATED)


def _r_inputs(tau=1.0, e_s_hi=0.7, ratio_hi=0.5, e_l_lo=2.0, r0_hi=0.5, w=60.0, n=40.0):
    per = {
        s: {
            "e_S": {"ci95": [0.5, e_s_hi]},
            "S_over_N": {"ci95": [0.3, ratio_hi]},
            "e_L": {"ci95": [e_l_lo, 3.0]},
        }
        for s in pr.MODEL_SEEDS
    }
    aims = {"W": w, "N": n, "L-shuf": 20.0, "L-mean": 20.0, "L-rand": 10.0}
    return {"r0": {"ci95": [0.3, r0_hi]}, "per_seed": per, "aims": aims, "tau_commit_cm": tau}


def test_stage_r_rows_first_match_in_both_directions():
    assert pr.decide_r(pr.r_gates(**_r_inputs()))["row"] == "R-PASS"
    for change, row in (
        ({"r0_hi": 1.01}, "R-VOID-CEILING"),
        ({"e_l_lo": 0.99}, "R-COMMAND-KEYED"),
        ({"e_s_hi": 1.01}, "R-NO-BAR"),
        ({"ratio_hi": 1.0}, "R-NO-BAR"),
        ({"w": 55.9}, "A-NO-BAR"),
        ({"n": 53.1}, "A-TWIN"),
    ):
        d = pr.decide_r(pr.r_gates(**_r_inputs(**change)))
        assert d["row"] == row and not d["clause_fires"], change
    # the first match: a failed ceiling outranks everything below it
    worst = _r_inputs(r0_hi=2.0, e_l_lo=0.1, e_s_hi=2.0, w=0.0)
    assert pr.decide_r(pr.r_gates(**worst))["row"] == "R-VOID-CEILING"
    # A2 is exactly +7/64 at the boundary
    assert pr.r_gates(**_r_inputs(w=60.0, n=53.0))["gates"]["A2"]
    # R1 is <= tau, measured: a tightened tau tightens R1 with it
    assert not pr.r_gates(**_r_inputs(tau=0.5, e_s_hi=0.6))["gates"]["R1"]
    assert pr.decide_r(None, void=True)["row"] == "V"
    bad = _r_inputs()
    bad["aims"].pop("L-rand")
    with pytest.raises(ContractError):
        pr.r_gates(**bad)
    with pytest.raises(ContractError):
        pr.decide_r(pr.NOT_EVALUATED)


def _outcomes(w=60, twins=40, comp=60, ceiling=62, n=64):
    def first(k):
        return [i < k for i in range(n)]

    out = {"W": first(w), "H-final": first(ceiling), "H-rule": first(comp), "H-sysid": first(0)}
    for t in pr.TWINS:
        out[t] = first(twins)
    return out


def test_stage_s_rows_in_both_directions_with_salt_8206():
    assert pr.decide_s(_outcomes())["row"] == "L-PASS"
    assert pr.decide_s(_outcomes(ceiling=55))["row"] == "S-VOID-CEILING"
    assert pr.decide_s(_outcomes(twins=59))["row"] == "L-NO-GAIN"
    assert pr.decide_s(_outcomes(w=30, comp=64, twins=0))["row"] == "L-INFERIOR"
    assert pr.decide_s(_outcomes(w=55, twins=20, comp=55))["row"] == "L-BAR"
    assert pr.decide_s(None, void=True)["row"] == "V"
    for row in ("L-NO-GAIN", "L-INFERIOR"):
        assert row in pr.CLAUSE_ROWS
    for row in pr.NO_CLAUSE_ROWS:
        assert row not in pr.CLAUSE_ROWS
    d = pr.decide_s(_outcomes())
    assert d["G-NI"] == pr.paired_interval(_outcomes()["W"], _outcomes()["H-rule"]) | {
        "passes": True,
        "detectably_inferior": False,
    }
    assert pr.decide_d({"W": 14, "N": 8, "L-shuf": 6, "L-mean": 6, "H-final": 16})["row"] == (
        "D-PASS"
    )
    assert pr.decide_d({"W": 11, "N": 4, "L-shuf": 4, "L-mean": 4, "H-final": 16})["row"] == (
        "L-DEV-STOP"
    )


def test_the_sentinel_and_runner_fields():
    run = _runner()
    report: dict = {}
    fields = run.Fields(report, "rgate")
    assert all(v == pr.NOT_EVALUATED for v in report["fields"].values())
    fields.check()
    with pytest.raises(pt.GuardError):
        fields.set("decision", pr.NOT_EVALUATED)
    report["fields"]["decision"] = {"row": "R-PASS"}
    with pytest.raises(pt.GuardError):
        fields.check()
    with pytest.raises(KeyError):
        fields.set("nothing", 1)


# ----- the readouts each arm reads (§6) and gate-P's aim (§5.2) ---------------------------------
def test_w_and_the_scene_blind_twins_read_r_s_and_n_reads_r_n_never_r8():
    assert prt.READOUT_OF == {"W": "r_s", "L-shuf": "r_s", "L-mean": "r_s", "N": "r_n"}
    for arm in ("H-read", "H-rule", "L-rand", "W-r8"):
        with pytest.raises(pt.GuardError):
            prt.readout_of(arm)
    text = (ROOT / "src/embodied_jepa/lewm_pr_v2_runtime.py").read_text()
    code = ast.unparse(ast.parse(text))  # docstrings kept; check the calls only
    assert "_readout('r8')" not in code and '_readout("r8")' not in code


class _Model:
    def predict_features(self, start, commands):
        # a toy dynamics: the predicted latent at every step is the mean command
        k = len(commands)
        out = np.zeros((k, commands.shape[1], 4), np.float32)
        out[:, :, :2] = np.asarray(commands)[:, :, :2].mean(1, keepdims=True)
        return out


class _Readout:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def predict(self, x):
        self.calls.append(self.name)
        return np.asarray(x)[..., :2].reshape(-1, 2) * 0.0 + 0.45


def _offline_env(monkeypatch, calls):
    monkeypatch.setattr(prt.rtm, "_W", {"fk": None}, raising=False)
    monkeypatch.setattr(prt.rtm, "standin", lambda: None)

    def chunks(standin, fk, apple, state, step, targets, *, last_grasp, horizon):
        t = np.asarray(targets).reshape(-1, 2)
        applied = np.zeros((len(t), horizon, 14), np.float32)
        applied[:, :, :2] = t[:, None, :]
        return None, applied, np.ones(len(t), bool)

    monkeypatch.setattr(prt.pp, "primitive_chunks", chunks)
    monkeypatch.setattr(prt.wrt, "world_model", lambda arm: _Model())
    monkeypatch.setattr(prt.ptr, "_readout", lambda name: _Readout(name, calls))
    monkeypatch.setattr(prt.wrt, "mean_latent", lambda: np.zeros(4, np.float32))


def _aim_task(arm, **extra):
    return {
        "kind": "offline_aim",
        "key": 0,
        "arm": arm,
        "seed": 65901,
        "state": [0.0],
        "apple": [0.4, -0.1],
        "last_grasp": [0.4, -0.1],
        "p_hat": [0.49, -0.09],
        "h": [0.40, -0.20],
        "tau_commit_cm": 1.0,
        "start": np.zeros(4, np.float32),
    } | extra


def test_offline_aims_read_each_arms_own_readout(monkeypatch):
    calls: list[str] = []
    _offline_env(monkeypatch, calls)
    for arm, name in (("W", "r_s"), ("L-shuf", "r_s"), ("N", "r_n")):
        calls.clear()
        out = prt.run_task(_aim_task(arm, keep_plates=True))
        assert set(calls) == {name}, arm
        assert len(out["g"]) == 2
    calls.clear()
    prt.run_task(_aim_task("L-mean", start="mean"))
    assert set(calls) == {"r_s"}
    calls.clear()
    out = prt.run_task(_aim_task("L-rand"))
    assert calls == [] and out["log"]["rollouts"] == 0
    feasible = np.ones(len(pr.GRID), bool)
    assert out["log"]["grid_index"] == pr.l_rand_index(65901, feasible)
    w = prt.run_task(_aim_task("W", keep_plates=True))
    assert len(w["grid_plates"]) == len(pr.GRID) == len(w["grid_targets"])


def test_choose_is_task077s_controller_form_except_l_rands_salt():
    p_hat, h = np.array([0.49, -0.09]), np.array([0.40, -0.20])
    targets = np.asarray([c1.from_box(a, b, p_hat, h) for a, b in pr.GRID])
    chunks = np.zeros((len(targets), 60, 14))
    feasible = np.ones(len(targets), bool)
    fixed = c1.fixed_point(p_hat, h)

    def predict(cmds):
        return np.repeat(fixed[None], len(cmds), 0)

    def chunk_of(g):
        return np.zeros((60, 14)), True

    args = (p_hat, h, targets, chunks, feasible, predict, chunk_of)
    mine = prt.choose("W", *args, tolerance_m=0.0025, seed=1)
    theirs = wrt.choose_from_grid("W", *args, tolerance_m=0.0025, seed=1)
    assert np.allclose(mine[0], theirs[0]) and mine[1] == theirs[1]
    g, log = prt.choose("L-rand", *args, tolerance_m=0.0025, seed=65905)
    assert log["grid_index"] == pr.l_rand_index(65905, feasible)
    draws = {pr.l_rand_index(s, feasible) for s in range(65900, 65960)}
    assert draws != {lm.l_rand_index(s, feasible) for s in range(65900, 65960)}


class _Hook:
    def current(self):
        return np.array([0.52, -0.06])  # the true plate


class _Ctl:
    palm = {405: np.array([0.40, -0.20])}
    apple = np.array([0.4, -0.1])
    last_grasp = (0.4, -0.1)


class _Obs:
    images = {"onboard_rgb": np.zeros((1, 112, 112, 3), np.uint8)}


@pytest.mark.parametrize("half", ["gate_p", "contrast_t"])
def test_gate_p_aims_are_built_from_p_hat_and_never_the_true_plate(monkeypatch, half):
    monkeypatch.setattr(prt.wrt, "encode", lambda frame: (np.zeros(8), np.zeros(4)))
    monkeypatch.setattr(prt.rt2, "state_of", lambda obs: np.zeros(3))
    monkeypatch.setattr(prt.fp2, "CAMERA", "onboard_rgb")

    class Read:
        def predict(self, tokens):
            return np.array([0.49, -0.09])  # p-hat

    task = {"a": 0.2, "b_m": 0.01, "aim_from": pr.AIM_FROM[half]}
    aim = prt.CollectF(_Hook(), task, Read())
    record: dict = {}
    g = aim(_Ctl(), _Obs(), 405, record)
    h = _Ctl.palm[405]
    from_p_hat = c1.from_box(0.2, 0.01, np.array([0.49, -0.09]), h)
    from_true = c1.from_box(0.2, 0.01, _Hook().current(), h)
    assert record["p_hat405"] == [0.49, -0.09] and record["aim_from"] == pr.AIM_FROM[half]
    if half == "gate_p":
        assert np.allclose(g, from_p_hat) and not np.allclose(g, from_true)
    else:
        assert np.allclose(g, from_true)
    with pytest.raises(pt.GuardError):
        prt.CollectF(_Hook(), {"a": 0.0, "b_m": 0.0}, Read())


def test_candidate_arms_read_no_task_truth():
    """G-privileged, statically: the candidate arms' aim and helpers read the observation, the
    controller's own palm, kinematics, apple estimate and grasp, the stand-in and the readouts,
    never the hook, the simulator or the truth."""
    path = ROOT / "src/embodied_jepa/lewm_pr_v2_runtime.py"
    text = path.read_text()
    names = {"PredReadoutAim", "choose", "readout_of", "rollout_plates", "run_offline_aim_task",
             "offline_rule_aim", "_standin_setup"}  # fmt: skip
    nodes = [n for n in ast.parse(text).body if getattr(n, "name", None) in names]
    assert {n.name for n in nodes} == names
    for node in nodes:
        code = ast.unparse(node)
        code = code.replace(ast.get_docstring(node) or "\0", "")
        for forbidden in ("hook", ".sim", "truth", "robot", "current()"):
            assert forbidden not in code, (node.name, forbidden)
    assert {"W", "N", "L-shuf", "L-mean", "L-rand"}.isdisjoint(pr.PRIVILEGED_ARMS)
    assert "collect-f" in pr.PRIVILEGED_ARMS


def test_run_arm_refuses_task_truth_in_a_non_privileged_arm():
    run = _runner()

    class FakePool:
        def map(self, tasks, cap, what):
            return [{"seed": 1, "success": True, "blocked": None, "privileged_ok": True,
                     "task_truth_in_controller": 1}]  # fmt: skip

    with pytest.raises(pt.GuardError):
        run.run_arm(FakePool(), [{"arm": "W"}], "W")
    run.run_arm(FakePool(), [{"arm": "H-final"}], "H-final")


# ----- the offline pieces -------------------------------------------------------------------------
def test_echo_slope_recovers_the_slope_and_aim_summary_maps_the_curve():
    rng = np.random.default_rng(1)
    g = rng.uniform(-1, 1, (147, 2))
    for a in (-0.5 * np.eye(2), np.eye(2)):
        res = [{"grid_targets": g.tolist(), "grid_plates": (g @ a.T + 0.3).tolist()}]
        out = pof.echo_slopes(res)
        assert np.allclose(out["median_matrix"], a)
    data = {"n": 2, "plate405": np.array([[0.5, -0.1]] * 2), "palm405": np.array([[0.4, -0.2]] * 2)}
    star = pof.g_star(data)
    results = [
        {"key": 0, "g": star[0].tolist(), "log": {}},
        {"key": 1, "g": (star[1] + [0.03, 0]).tolist(), "log": {"clipped": True}},
    ]
    summary = pof.aim_summary(results, data, pr.K_COUNTS, pr.K_RESETS)
    assert summary["predicted_count_of_64"] == pytest.approx(32 * (1 + 2 / 32))
    assert summary["clip_binding_fraction"] == 0.5
    with pytest.raises(pt.GuardError):
        pof.aim_summary(results[:1], data, pr.K_COUNTS, pr.K_RESETS)


def test_readouts_fit_and_read_on_predicted_latents():
    rng = np.random.default_rng(2)
    n = 40
    y = rng.uniform(0.4, 0.6, (n, 2))
    pred = {k: np.hstack([y + rng.normal(0, s, y.shape), rng.normal(size=(n, 6))])
            for k, s in (("S", 0.001), ("N", 0.02), ("L", 0.2))}  # fmt: skip
    seeds = np.arange(67000, 67000 + n)
    fitted = pof.fit_readouts(pred, y, seeds)
    assert sorted(fitted) == ["r_l", "r_n", "r_s"]
    e = pof.readout_errors(fitted, pred, y)
    assert np.median(e["e_S"]) < np.median(e["e_N"]) < np.median(e["e_L"])
    stats = pof.readout_statistics(e)
    assert stats["S_over_N"]["ratio"] < 1.0
    cf = pof.crossfit(pred, y, seeds, np.array(["train"] * 30 + ["gate"] * 10))
    assert set(cf["readouts"]["r_s"]) == {"train", "gate", "all"}
    curve = pof.learning_curve(pred, y, seeds, pred, y)
    assert set(curve["r_s"]) == {"0.25", "0.5", "0.75", "1.0"}


def test_fresh_root_store_round_trip(tmp_path):
    arrays = {
        "frames": np.zeros((65, 112, 112, 3), np.uint8),
        "commands": np.zeros((64, 14), np.float32),
        "plate": np.zeros((65, 2)),
        "palm": np.zeros((65, 2)),
        "hidden_r": np.zeros((112, 112, 3), np.uint8),
        "target": np.zeros(2),
        "state405": np.zeros(3),
        "apple_estimate": np.zeros(2),
        "last_grasp": np.zeros(2),
        "p_hat405": np.array([0.49, -0.09]),
    }
    sha = pof.write_root(tmp_path, 65940, arrays)
    with pytest.raises(FileExistsError):
        pof.write_root(tmp_path, 65940, arrays)
    split = {"gate_p": [65940], "contrast_t": [65941]}
    sealed = pof.seal_corpus(tmp_path, {65940: sha}, split, {"debug": True})
    m = pof.open_corpus(tmp_path, sealed["sha256"])
    assert m["split"]["contrast_t"] == [] and m["aim_from"]["gate_p"] == "p_hat"
    with pytest.raises(pt.GuardError):
        pof.open_corpus(tmp_path, "0" * 64)
    with pytest.raises(ContractError):
        pof.write_root(tmp_path, 65941, {k: v for k, v in arrays.items() if k != "p_hat405"})


# ----- the runner ---------------------------------------------------------------------------------
def test_the_runner_uses_run_tools_guards_and_loads_no_other_script():
    text = RUNNER.read_text()
    calls = _calls(RUNNER)
    for name in ("assert_local_import", "install_guards", "gpu_guard", "MemoryWatch",
                 "check_pins", "check_evidence", "verify_feature_files"):  # fmt: skip
        assert name in calls, name
    assert "require_lock=True" in text
    nri = _load(ROOT / "tests" / "test_no_runner_imports.py", "_nri")
    loaded, loaders = nri.script_loads(RUNNER)
    assert not loaded and not loaders


def test_the_runner_refuses_gated_stages_while_draft_and_misused_flags(tmp_path):
    run = _runner()
    assert set(run.DRAFT_ALLOWED) == {"tests", "simulate", "k0", "dryrun"}
    for argv in (
        ["k0", "--output", str(tmp_path / "a")],  # no evidence
        ["featurise", "--output", str(tmp_path / "b"), "--corpus", "x"],  # no tests record
        ["simulate", "--output", str(tmp_path / "c"), "--workers", "2"],  # workers: debug only
        ["simulate", "--output", str(tmp_path / "d"), "--debug-skip-tests"],
        ["dryrun", "--output", str(tmp_path / "e"), "--evidence", "x"],  # no old artifacts
        ["closed", "--output", str(tmp_path / "f"), "--evidence", "x", "--corpus", "c",
         "--corpus-sha256", "s", "--old-features", "f", "--old-fits", "g", "--models", "m"],
    ):  # fmt: skip
        with pytest.raises(SystemExit):
            run.main(argv)


def test_the_preflight_refuses_gated_stages_while_draft(monkeypatch):
    run = _runner()
    from types import SimpleNamespace

    monkeypatch.setattr(run.rt, "assert_local_import", lambda *a, **k: None)
    monkeypatch.setattr(run.hz, "check_pins", lambda pins, *a: dict(pins))
    monkeypatch.setattr(run, "check_task077", lambda report: None)
    monkeypatch.setattr(run.hz, "tracked_tree_dirty", lambda *a: [])
    monkeypatch.setattr(run.pr, "STATUS", "DRAFT")
    monkeypatch.setattr(run, "check_frozen_pin", lambda manifest: {"status": "DRAFT"})
    for stage in ("corpus", "featurise", "rgate", "closed"):
        args = SimpleNamespace(stage=stage, debug=False, debug_skip_tests=False)
        with pytest.raises(pt.GuardError, match="runs only after the freeze"):
            run.preflight({}, args)


def test_the_preflight_refuses_k0_and_dryrun_once_frozen(monkeypatch):
    """K0′ and the dry run ran once, before the freeze (R18.22); once FROZEN they are refused,
    and the own pins and the protocol document are checked."""
    run = _runner()
    from types import SimpleNamespace

    assert pr.STATUS == "FROZEN"
    monkeypatch.setattr(run.rt, "assert_local_import", lambda *a, **k: None)
    monkeypatch.setattr(run, "check_task077", lambda report: None)
    monkeypatch.setattr(run.hz, "tracked_tree_dirty", lambda *a: [])
    for stage in ("k0", "dryrun"):
        report = {}
        args = SimpleNamespace(stage=stage, debug=False, debug_skip_tests=False)
        with pytest.raises(pt.GuardError, match="ran once before the freeze"):
            run.preflight(report, args)
        assert (
            report["own_pins_at_preflight"] == len(json.loads(MANIFEST.read_text())["hashes"]) == 7
        )
        assert report["protocol_document_check"]["path"] == pr.DOCUMENT


def test_stage_modules_import_without_torch_or_mujoco():
    code = (
        "import sys\n"
        "import embodied_jepa.lewm_pr_v2, embodied_jepa.lewm_pr_v2_runtime\n"
        "import embodied_jepa.lewm_pr_v2_offline\n"
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


class _FakeCohort:
    def __init__(self, seeds):
        self.resets = {s: lm.reset_of(s) for s in seeds}
        self.est = {
            s: {"estimates": [0.3, -0.2, 0.5, -0.1], "frame_sha256": "f", "state_sha256": "s"}
            for s in seeds
        }


class _FakePool:
    def __init__(self, refused):
        self.refused, self.calls = set(refused), []

    def map(self, tasks, cap, what):
        self.calls.append(tasks)
        out = []
        for task in tasks:
            s, arm = task["seed"], task["arm"]
            base = {"seed": s, "arm": arm, "blocked": None, "privileged_ok": True,
                    "task_truth_in_controller": 0, "seconds": 1.0,
                    "termination_reason": "step_limit", "executed_steps": 800}  # fmt: skip
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
            out.append(base | {
                "success": good, "frame405": np.full((2, 2, 3), s % 251, np.uint8),
                "decisions": [{"reading": [0.49, -0.09]}],
                "commit": {"target": [0.5, -0.1], "fallback": False, "clipped": False},
            })  # fmt: skip
        return out


@pytest.mark.parametrize("role", ["D", "S"])
def test_closed_core_counts_refused_resets_and_moves_with_salt_8201(role):
    run = _runner()
    n = 16 if role == "D" else 64
    seeds = pr.seeds_of(role)
    refused = {seeds[1], seeds[2]}
    pool = _FakePool(refused)
    report: dict = {}
    fields = run.Fields(report, "closed")
    row = run.closed_core(pool, _FakeCohort(seeds), seeds, role,
                          {"tau_commit_cm": 1.0, "a_lo": pr.A_LO}, [[0.0]], fields)  # fmt: skip
    arms = report["fields"]["arms"]
    assert arms["W"]["count"] == n - 2 and arms["refused_before_405"]["W"] == 2
    shuf = next(c for c in pool.calls if c[0]["arm"] == "L-shuf")
    assert int(shuf[0]["foreign_frame"][0, 0, 0]) == seeds[3] % 251
    first = pool.calls[0][0]
    assert first["move_offset"] == pr.move_offset(seeds[0]).tolist()
    if role == "S":
        assert row == "L-PASS" and report["fields"]["determinism"]["ok"]
    else:
        assert row == "D-PASS"


def test_stage_r_opens_gate_p_only_after_first_outcome_utc():
    tree = ast.parse(RUNNER.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stage_rgate")
    lines = [ast.unparse(n) for n in fn.body]
    stamp = next(i for i, x in enumerate(lines) if x.startswith("report['first_outcome_utc']"))
    opened = [i for i, x in enumerate(lines) if "args.features" in x and "load_roots" in x]
    fits = next(i for i, x in enumerate(lines) if "fit_phase(" in x)
    assert opened and fits < stamp < min(opened)


def test_the_closed_loop_workers_get_r_s_and_r_n_and_s_needs_d_pass(tmp_path):
    run = _runner()
    chain = {
        "paths": {"r_plate": "rp.npz", "r8": "r8.npz", "mean_latent": "m.npy"},
        "jobs": {f"{a}-66800": {"checkpoint": f"{a}.pt", "checkpoint_sha256": a,
                                "metadata": {}} for a in ("W", "N")},
    }  # fmt: skip
    readouts = {n: {"path": f"{n}.npz", "sha256": n * 4} for n in ("r_s", "r_n")}
    config = run.worker_config(chain, readouts=readouts, with_r8=True)
    assert config["r_s"] == "r_s.npz" and config["r_n_sha256"] == "r_nr_nr_nr_n"
    assert "r_s" not in run.worker_config(chain) and "r8" not in run.worker_config(chain)
    argv = ["closed", "--cohort", "S", "--output", str(tmp_path / "s"), "--evidence", "e",
            "--corpus", "c", "--corpus-sha256", "x", "--old-features", "f", "--old-fits", "g",
            "--models", "m", "--stage-r", "r"]  # fmt: skip
    with pytest.raises(SystemExit):
        run.main(argv)
