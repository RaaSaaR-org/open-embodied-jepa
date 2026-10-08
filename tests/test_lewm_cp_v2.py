"""TASK-081's Stage-0 code (protocol ``docs/experiments/apple_lewm_commit_precision_v2.md`` §6.1
step 1): seeds and salts, the carried pins, the solver (equal to the design note's tested
affine_local, every fallback included), W-frozen (equal to TASK-080's controller), the readouts
each arm reads, L-rand's salt, the runner's carried code, the sentinel and every row ladder in both
directions at n = 128."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import commit_precision_dev as cpd
from embodied_jepa import lewm_c1m_v2 as lm
from embodied_jepa import lewm_c1m_v2_runtime as wrt
from embodied_jepa import lewm_cp_v2 as cpv
from embodied_jepa import lewm_cp_v2_runtime as cpr
from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa import lewm_pr_v2 as pr
from embodied_jepa import lewm_pr_v2_runtime as prt
from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa import plate_twin_v2_harness as hz
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_lewm_cp_v2.py"
TASK080_RUNNER = ROOT / "scripts" / "run_lewm_pr_v2.py"
MANIFEST = ROOT / cpv.MANIFEST
P_HAT = np.array([0.47, -0.08])
H = np.array([0.30, -0.16])


def _load(path: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner():
    return _load(RUNNER, "_run_lewm_cp_v2")


# ----- the frozen block, the manifest, the carried pins ------------------------------------------
def test_manifest_frozen_block_equals_the_module():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["frozen"] == json.loads(json.dumps(cpv.frozen_block(), ensure_ascii=False))
    assert manifest["frozen_sha256"] == cpv.frozen_sha256()
    assert manifest["status"] == cpv.STATUS


def test_a_draft_carries_no_pin():
    manifest = json.loads(MANIFEST.read_text())
    if cpv.STATUS == "DRAFT":
        assert manifest["frozen_sha256_pin"] is None and manifest["hashes"] == {}
        assert _runner().check_frozen_pin(manifest)["pin"] is None


def test_task080_task077_and_the_solver_file_are_carried_unchanged():
    m080 = json.loads((ROOT / cpv.TASK080_MANIFEST).read_text())
    assert pr.frozen_sha256() == cpv.TASK080_FROZEN_SHA256 == m080["frozen_sha256_pin"]
    assert len(hz.check_pins(m080["hashes"])) == 7
    assert hz.sha256_file(ROOT / pr.DOCUMENT) == m080["protocol_document_sha256"]
    assert lm.frozen_sha256() == cpv.TASK077_FROZEN_SHA256
    assert hz.sha256_file(ROOT / cpv.SOLVER_FILE) == cpv.SOLVER_FILE_SHA256
    assert cpv.TAU_COMMIT_CM == 1.0
    assert cpv.TAU_CURVE == {"0.0": 64, "0.5": 58, "1.0": 60, "1.5": 32, "2.0": 29, "3.0": 7}
    assert cpv.PRIMARY_SEED == 66800


def test_seed_ranges_and_salts():
    cpv.check_seed_ranges()
    assert cpv.seeds_of("D") == tuple(range(70100, 70116))
    assert cpv.seeds_of("S") == tuple(range(70200, 70328)) and len(cpv.seeds_of("S")) == 128
    for role in ("D", "S"):
        for s in (*cpv.seeds_of(role), *cpv.seeds_of(role, debug=True)):
            assert 70100 <= s <= 71999
            for low, high in cpv.FORBIDDEN_RANGES.values():
                assert not low <= s <= high
    assert cpv.FORBIDDEN_RANGES["task081_development"] == (70000, 70099)
    assert cpv.FORBIDDEN_RANGES["task080_block"] == (65000, 65999)
    assert set(cpv.SALTS.values()) == {8302, 8303, 8304}
    assert cpv.USED_SALTS == {"design_note_development": 8301}
    assert set(cpv.RESERVED_SALTS.values()) == set(range(8305, 8313))
    assert cpv.CARRIED_SALTS == {"move": 8201}
    with pytest.raises(pt.GuardError):
        cpv.check_seeds("S", cpv.seeds_of("S")[:-1])
    with pytest.raises(pt.GuardError):
        cpv.check_seeds("D", cpv.seeds_of("D", debug=True))
    assert cpv.check_seeds("S", cpv.seeds_of("S", debug=True), debug=True)


def test_the_move_draw_is_task080s_salt_8201_on_the_fresh_seeds():
    for s in (70100, 70200, 71900):
        assert cpv.move_offset(s).tolist() == pr.move_offset(s).tolist()


def test_bars_at_128():
    assert (cpv.G_BAR, cpv.DELTA, cpv.MIN_SEPARATION, cpv.MCNEMAR_P) == (112, 16, 7, 0.01)
    assert cpv.G_BAR / cpv.S_RESETS == pr.G_BAR / pr.S_RESETS == 0.875
    assert cpv.DELTA / cpv.S_RESETS == pr.DELTA / pr.S_RESETS == 0.125
    assert pt.mcnemar_one_sided(7, 0) < 0.01 <= pt.mcnemar_one_sided(6, 0)


# ----- the solver ---------------------------------------------------------------------------------
def _setup(slope, *, fixed=None, infeasible=(), noise=0.0, zero=False):
    """A synthetic predictor p~(g) = g* + S (g - g*) (+ noise); a chunk encodes its aim."""
    fixed = c1.from_box(0.31, 0.004, P_HAT, H) if fixed is None else np.asarray(fixed)
    slope = np.asarray(slope, np.float64)
    targets = np.asarray([c1.from_box(a, b, P_HAT, H) for a, b in lm.GRID])
    rng = np.random.default_rng(3)
    bumps = rng.normal(0, noise, (4096, 2))

    def encode(g):
        c = np.zeros((lm.HORIZON, 14), np.float32)
        c[0, :2] = np.asarray(g, np.float32)
        return c

    def predict(commands):
        g = np.asarray(commands, np.float64)[:, 0, :2]
        if zero:
            return np.repeat(fixed[None], len(g), 0)
        key = (np.abs(g * 1e4).astype(np.int64).sum(1)) % 4096
        return fixed + (g - fixed) @ slope.T + bumps[key]

    bad = {i for i in infeasible}

    def chunk_of(g):
        near = np.linalg.norm(targets - np.asarray(g), axis=1)
        i = int(np.argmin(near))
        return encode(g), not (near[i] < 1e-9 and i in bad)

    chunks = np.stack([encode(g) for g in targets])
    feasible = np.array([i not in bad for i in range(len(targets))])
    return fixed, targets, chunks, feasible, predict, chunk_of


def _dev_affine_local(targets, feasible, predict, chunk_of):
    chunks = np.stack([chunk_of(g)[0] for g in targets])
    index = np.flatnonzero(feasible)
    grid = predict(chunks[index])

    def single(g):
        return predict(chunk_of(g)[0][None])[0]

    def feasible_aim(g):
        return bool(chunk_of(g)[1])

    out = cpd.all_variants(P_HAT, H, targets, feasible, grid, single, feasible_aim, known=(),
                           tolerance_m=0.0025, a_lo=lm.A_LO)  # fmt: skip
    return out["affine_local"]


SLOPES = (
    np.diag([-0.13, -0.42]),
    -1.2 * np.eye(2),
    [[-0.9, 0.3], [0.2, -1.05]],
    [[-0.3, 0.1], [0.05, -0.6]],
)


@pytest.mark.parametrize("slope", SLOPES)
@pytest.mark.parametrize("noise", [0.0, 0.004])
def test_affine_local_is_the_design_notes_tested_variant(slope, noise):
    _f, targets, chunks, feasible, predict, chunk_of = _setup(slope, noise=noise)
    g, log = cpr.choose_affine_local("W", P_HAT, H, targets, chunks, feasible, predict, chunk_of)
    dev = _dev_affine_local(targets, feasible, predict, chunk_of)
    assert np.array_equal(g, dev["g"])
    assert log["fallback_grid_argmin"] == dev["fallback_grid_argmin"] is False
    assert log["points"] == dev["points"]
    assert log["residual_cm"] == pytest.approx(dev["residual_cm"])
    assert log["solver"] == "affine_local" and log["rollouts"] == int(feasible.sum()) + 1
    assert log["converged"] is None and "iterations" not in log


def test_affine_local_recovers_a_linear_maps_fixed_point():
    fixed, targets, chunks, feasible, predict, chunk_of = _setup([[-0.3, 0.1], [0.05, -0.6]])
    g, log = cpr.choose_affine_local("W", P_HAT, H, targets, chunks, feasible, predict, chunk_of)
    assert np.linalg.norm(g - fixed) < 1e-9 and log["points"] == 25 and not log["clipped"]
    assert log["design_rank"] == 3
    assert max(log["eigenvalues_real"]) < 0


def test_affine_local_fallbacks_match_the_design_note():
    # near-singular: p~(g) = g on the neighbourhood (slope I), so I - J is singular
    fixed, targets, chunks, feasible, predict, chunk_of = _setup(np.eye(2))
    g, log = cpr.choose_affine_local("W", P_HAT, H, targets, chunks, feasible, predict, chunk_of)
    dev = _dev_affine_local(targets, feasible, predict, chunk_of)
    assert log["fallback_reason"] == "near_singular" and dev["fallback_grid_argmin"]
    assert np.array_equal(g, dev["g"]) and np.array_equal(g, targets[log["grid_index"]])
    # fewer than 4 points: every candidate but three near the argmin infeasible
    _f, targets, chunks, feasible, predict, chunk_of = _setup(np.diag([-0.13, -0.42]))
    _g, log0 = cpr.choose_affine_local("W", P_HAT, H, targets, chunks, feasible, predict, chunk_of)
    best = log0["grid_index"]
    keep = {best, best + 1, best - 1}
    bad = [i for i in range(len(targets)) if i not in keep]
    _f, targets, chunks, feasible, predict, chunk_of = _setup(np.diag([-0.13, -0.42]),
                                                              infeasible=bad)  # fmt: skip
    g, log = cpr.choose_affine_local("W", P_HAT, H, targets, chunks, feasible, predict, chunk_of)
    dev = _dev_affine_local(targets, feasible, predict, chunk_of)
    assert log["fallback_reason"] == "fewer_than_4_points" and dev["fallback_grid_argmin"]
    assert np.array_equal(g, dev["g"])
    # an infeasible chunk at the solution: make the solution coincide with an infeasible target
    fixed = c1.from_box(0.30, 0.0, P_HAT, H)
    i_fixed = int(np.argmin(np.linalg.norm(targets - fixed, axis=1)))
    _f, targets, chunks, feasible, predict, chunk_of = _setup(
        [[-0.3, 0.1], [0.05, -0.6]], fixed=targets[i_fixed], infeasible=[i_fixed]
    )
    g, log = cpr.choose_affine_local("W", P_HAT, H, targets, chunks, feasible, predict, chunk_of)
    dev = _dev_affine_local(targets, feasible, predict, chunk_of)
    assert log["fallback_reason"] == "infeasible_chunk" and dev["fallback_grid_argmin"]
    assert np.array_equal(g, dev["g"])


def test_affine_local_with_all_candidates_infeasible_aims_at_p_hat():
    _f, targets, chunks, _feas, predict, chunk_of = _setup(-0.5 * np.eye(2))
    g, log = cpr.choose_affine_local("W", P_HAT, H, targets, chunks,
                                     np.zeros(len(targets), bool), predict, chunk_of)  # fmt: skip
    assert np.array_equal(g, P_HAT) and log["fallback_all_infeasible"]


def test_for_n_affine_local_commits_the_clipped_prediction():
    """Zero commands: every candidate's prediction is the same, so J = 0 and the aim is clip(p~),
    which is where TASK-080's refinement goes after one step."""
    fixed = c1.from_box(0.2, 0.01, P_HAT, H)
    _f, targets, chunks, feasible, predict, chunk_of = _setup(0 * np.eye(2), fixed=fixed, zero=True)
    g, log = cpr.choose_affine_local("N", P_HAT, H, targets, chunks, feasible, predict, chunk_of)
    frozen, flog = wrt.choose_from_grid("N", P_HAT, H, targets, chunks, feasible, predict,
                                        chunk_of, tolerance_m=0.0025, seed=0)  # fmt: skip
    assert np.allclose(g, c1.clip_to_box(fixed, P_HAT, H, lm.A_LO), atol=1e-12)
    assert np.allclose(g, frozen, atol=1e-12) and flog["converged"]


@pytest.mark.parametrize("slope", SLOPES)
def test_w_frozen_is_task080s_controller(slope):
    _f, targets, chunks, feasible, predict, chunk_of = _setup(slope, noise=0.003)
    args = (P_HAT, H, targets, chunks, feasible, predict, chunk_of)
    g, log = cpr.make_choose("frozen")("W", *args, tolerance_m=0.0025, seed=1)
    theirs = prt.choose("W", *args, tolerance_m=0.0025, seed=1)
    assert np.array_equal(g, theirs[0]) and log.pop("solver") == "frozen" and log == theirs[1]


def test_l_rand_uses_salt_8303_and_no_solver():
    _f, targets, chunks, feasible, predict, chunk_of = _setup(-0.5 * np.eye(2))
    args = (P_HAT, H, targets, chunks, feasible, predict, chunk_of)
    g, log = cpr.make_choose("none")("L-rand", *args, tolerance_m=0.0025, seed=70205)
    assert log["grid_index"] == cpv.l_rand_index(70205, feasible) and log["solver"] == "none"
    assert np.array_equal(g, targets[log["grid_index"]]) and log["rollouts"] == 0
    draws = [cpv.l_rand_index(s, feasible) for s in range(70200, 70260)]
    assert draws != [pr.l_rand_index(s, feasible) for s in range(70200, 70260)]
    with pytest.raises(pt.GuardError):
        cpr.make_choose("affine_local")("L-rand", *args, tolerance_m=0.0025, seed=1)
    with pytest.raises(pt.GuardError):
        cpr.make_choose("cap30")


def test_arms_solvers_and_readouts():
    assert cpv.SOLVER_OF == {"W": "affine_local", "W-frozen": "frozen", "N": "affine_local",
                             "L-shuf": "affine_local", "L-mean": "affine_local",
                             "L-rand": "none"}  # fmt: skip
    assert cpv.READOUTS == {"W": "r_s", "W-frozen": "r_s", "L-shuf": "r_s", "L-mean": "r_s",
                            "N": "r_n"}  # fmt: skip
    assert prt.READOUT_OF == {"W": "r_s", "L-shuf": "r_s", "L-mean": "r_s", "N": "r_n"}
    assert cpv.MODEL_ARM == {"W-frozen": "W"}
    assert cpv.SOLVER["extra_refinement"] is False
    assert (cpv.SOLVER["local_a_steps"], cpv.SOLVER["local_b_m"]) == (2, 0.02)
    assert set(cpv.S_ARMS) == set(cpv.ARMS) and set(cpv.D_ARMS) <= set(cpv.S_ARMS)
    assert cpv.CANDIDATE_ARMS.isdisjoint(cpv.PRIVILEGED_ARMS)
    assert {"H-final", "H-read", "H-now"} <= cpv.PRIVILEGED_ARMS
    assert "W-frozen" not in cpv.TWINS and "W-frozen" not in cpv.COMPARATORS


def test_run_attempt_task_swaps_choose_for_one_task_and_relabels(monkeypatch):
    seen = {}

    def fake(task):
        seen["arm"], seen["solver"] = task["arm"], prt.choose.solver
        wm = {"solver": prt.choose.solver}
        return {"seed": task["seed"], "arm": task["arm"], "decisions": [{"world_model": wm}]}

    original = prt.choose
    monkeypatch.setattr(prt, "run_attempt_task", fake)
    out = cpr.run_task({"kind": "attempt", "arm": "W-frozen", "seed": 71910})
    assert seen == {"arm": "W", "solver": "frozen"} and out["arm"] == "W-frozen"
    assert out["solver"] == "frozen" and prt.choose is original
    out = cpr.run_task({"kind": "attempt", "arm": "L-shuf", "seed": 71910})
    assert seen == {"arm": "L-shuf", "solver": "affine_local"} and prt.choose is original

    def wrong(task):
        return {"seed": 1, "decisions": [{"world_model": {"solver": "frozen"}}]}

    monkeypatch.setattr(prt, "run_attempt_task", wrong)
    with pytest.raises(pt.GuardError, match="G-solver"):
        cpr.run_task({"kind": "attempt", "arm": "W", "seed": 71910})
    assert prt.choose is original
    with pytest.raises(pt.GuardError):
        cpr.run_task({"kind": "attempt", "arm": "W-r8", "seed": 71910})


def test_candidate_choosers_read_no_task_truth():
    path = ROOT / "src/embodied_jepa/lewm_cp_v2_runtime.py"
    tree = ast.parse(path.read_text())
    names = {"choose_affine_local", "choose_l_rand", "make_choose", "_grid"}
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    assert {n.name for n in nodes} == names
    for node in nodes:
        code = ast.unparse(node).replace(ast.get_docstring(node) or "\0", "")
        for forbidden in ("hook", ".sim", "truth", "robot", "current()", "_readout"):
            assert forbidden not in code, (node.name, forbidden)


# ----- the ladders --------------------------------------------------------------------------------
def _outcomes(n=128, **counts):
    base = {"W": 120, "N": 50, "L-shuf": 40, "L-mean": 55, "L-rand": 20, "H-rule": 124,
            "H-sysid": 120, "H-final": 124, "W-frozen": 112}  # fmt: skip
    base |= counts
    out = {}
    for arm, k in base.items():  # nested: every arm's successes are the first k resets
        v = np.zeros(n, bool)
        v[:k] = True
        out[arm] = v
    return out


def test_stage_s_rows_in_both_directions():
    assert cpv.decide_s(_outcomes())["row"] == "L-PASS"
    assert cpv.decide_s(_outcomes(**{"H-final": 111}))["row"] == "S-VOID-CEILING"
    assert cpv.decide_s(_outcomes(**{"H-final": 112}))["row"] == "L-PASS"
    assert cpv.decide_s(_outcomes(**{"L-mean": 118}))["row"] == "L-NO-GAIN"  # b - c = 2
    assert cpv.decide_s(_outcomes(W=111))["row"] == "L-NEAR"  # 13 behind H-rule
    out = cpv.decide_s(_outcomes(W=111, **{"H-rule": 112, "H-sysid": 110}))
    assert out["row"] == "L-BAR" and not out["G-bar"]
    assert cpv.decide_s(_outcomes(W=112, **{"H-rule": 114}))["row"] == "L-PASS"
    near = cpv.decide_s(_outcomes(W=112, **{"H-rule": 128, "H-final": 128}))
    assert near["row"] == "L-NEAR" and not near["G-NI"]["passes"]
    assert not near["G-NI"]["detectably_inferior"]
    inf = cpv.decide_s(_outcomes(W=100, **{"H-rule": 128, "H-final": 128}))
    assert inf["row"] == "L-INFERIOR" and inf["clause_fires"]
    assert cpv.decide_s(_outcomes(**{"N": 113}))["row"] == "L-PASS"  # b = 7, c = 0
    twin_near = cpv.decide_s(_outcomes(**{"N": 115}))  # b = 5: the test fails, upper bound >= 7
    assert twin_near["row"] == "L-TWIN-NEAR" and twin_near["twins"]["N"]["miss_within_noise"]
    assert cpv.decide_s(None, void=True)["row"] == "V"
    with pytest.raises(ContractError):
        cpv.decide_s(_outcomes(n=64))
    assert cpv.decide_s(_outcomes(n=4, W=4, N=0, **{"L-shuf": 0, "L-mean": 0, "L-rand": 0,
                                                    "H-rule": 4, "H-sysid": 4, "H-final": 4,
                                                    "W-frozen": 4}),
                        debug=True)["row"]  # fmt: skip
    for row in cpv.CLAUSE_ROWS:
        assert row in cpv.S_ROWS and row not in cpv.NO_CLAUSE_ROWS


def test_l_twin_near_is_a_miss_within_noise():
    out = _outcomes()
    rng = np.random.default_rng(0)
    w = out["W"]
    n_ = w.copy()
    flip = rng.choice(np.flatnonzero(w), 4, replace=False)
    n_[flip] = False
    n_[np.flatnonzero(~w)[:2]] = True  # b = 4, c = 2: the test fails; the upper bound >= +7?
    out["N"] = n_
    row = cpv.decide_s(out)
    t = row["twins"]["N"]
    assert not t["passes"]
    assert row["row"] == ("L-NO-GAIN" if t["detectably_no_better"] else "L-TWIN-NEAR")


def test_the_s_ladder_matches_task080s_order_at_64_with_its_constants(monkeypatch):
    """Patched to TASK-080's n = 64 constants and salt, the ladder gives TASK-080's rows."""
    monkeypatch.setattr(cpv, "G_BAR", pr.G_BAR)
    monkeypatch.setattr(cpv, "DELTA", pr.DELTA)
    monkeypatch.setattr(cpv, "S_RESETS", 64)
    monkeypatch.setattr(cpv, "paired_interval", lambda a, b: pr.paired_interval(a, b))
    rng = np.random.default_rng(5)
    for _ in range(40):
        rates = rng.uniform(0.3, 1.0, 9)
        out = {a: rng.uniform(size=64) < r for a, r in zip(
            ("W", "N", "L-shuf", "L-mean", "L-rand", "H-rule", "H-sysid", "H-final", "W-frozen"),
            rates, strict=True)}  # fmt: skip
        mine, theirs = cpv.decide_s(out), pr.decide_s(out)
        assert mine["row"] == theirs["row"]
        assert mine["G-NI"]["ci95"] == theirs["G-NI"]["ci95"]


def test_stage_d_rows():
    ok = {"W": 15, "W-frozen": 13, "N": 8, "L-shuf": 5, "L-mean": 9, "H-final": 16, "H-rule": 16}
    out = cpv.decide_d(ok)
    assert out["row"] == "D-PASS" and out["reported_only"] == {"W-frozen": 13, "H-rule": 16}
    assert cpv.decide_d(ok | {"W": 11})["row"] == "L-DEV-STOP"
    assert cpv.decide_d(ok | {"H-final": 13})["row"] == "L-DEV-STOP"
    assert cpv.decide_d(ok | {"L-mean": 13})["row"] == "L-DEV-STOP"
    assert cpv.decide_d(ok | {"W-frozen": 0, "H-rule": 0})["row"] == "D-PASS"
    with pytest.raises(ContractError):
        cpv.decide_d({k: v for k, v in ok.items() if k != "H-rule"})


def test_solver_effect_is_reported_and_the_bootstrap_salt_is_8302():
    out = _outcomes(W=122, **{"W-frozen": 115})
    eff = cpv.decide_s(out)["solver_effect_reported_only"]
    assert eff["difference"] == 7 and (eff["b"], eff["c"]) == (7, 0)
    assert eff["p"] == pytest.approx(0.5**7)
    a, b = out["W"], out["H-rule"]
    assert cpv.paired_interval(a, b) == pr.paired_interval(a, b, salt=8302)


# ----- power --------------------------------------------------------------------------------------
def test_power_pieces():
    assert cpv.g_bar_power(0.906) == pytest.approx(0.908, abs=0.001)
    assert cpv.g_bar_power(0.875) == pytest.approx(0.566, abs=0.001)
    # TASK-080's S: (k+, k-) = (1, 6) fails at n = 64, margin 8
    assert not cpv.ni_passes(1, 6, n=64, margin=8) and cpv.ni_passes(0, 0, n=64, margin=8)
    p = cpv.ni_power(0.953, 0.992, "overlap", p_sysid=62 / 64, trials=2000)
    assert 0.9 < p["single"]["g_ni_pass_rate"] <= 1.0
    assert p["better_of_two"]["g_ni_pass_rate"] <= p["single"]["g_ni_pass_rate"] + 0.02
    size = cpv.ni_power(0.992 - 16 / 128, 0.992, "overlap", trials=2000)
    assert size["single"]["g_ni_pass_rate"] < 0.08
    exact = cpv.twin_power(0.906, 0.75, "nested", n=64)
    assert exact == pytest.approx(pr.POWER_TABLE_DRAFT[(0.906, 0.75)][0], abs=0.01)
    assert cpv.twin_power(0.906, 26 / 64, "independent") > 0.999


# ----- the runner ---------------------------------------------------------------------------------
CARRIED = ("check_task077", "summary_line", "summary_ok", "run_full_tests", "wait_quiet",
           "check_disk", "sim_preflight", "completed", "old_chain", "worker_config",
           "streamed_estimates", "cohort_estimates", "attempt_tasks", "successes", "lean",
           "arm_summary", "open_log")  # fmt: skip


def _functions(path: Path) -> dict:
    text = path.read_text()
    tree = ast.parse(text)
    return {n.name: ast.get_source_segment(text, n) for n in tree.body
            if isinstance(n, ast.FunctionDef)}  # fmt: skip


def test_the_runners_carried_code_is_task080s_verbatim():
    mine, theirs = _functions(RUNNER), _functions(TASK080_RUNNER)
    for name in CARRIED:
        assert mine[name] == theirs[name], name
    assert mine["run_arm_carried"] == theirs["run_arm"].replace(
        "def run_arm(", "def run_arm_carried(", 1
    )


def test_the_runner_uses_run_tools_guards_and_loads_no_other_script():
    calls = set()
    for node in ast.walk(ast.parse(RUNNER.read_text())):
        if isinstance(node, ast.Call):
            f = node.func
            calls.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    for name in ("assert_local_import", "install_guards", "MemoryWatch", "check_pins",
                 "check_evidence", "verify_feature_files"):  # fmt: skip
        assert name in calls, name
    nri = _load(ROOT / "tests" / "test_no_runner_imports.py", "_nri")
    loaded, loaders = nri.script_loads(RUNNER)
    assert not loaded and not loaders


def test_the_runner_refuses_misused_flags(tmp_path):
    run = _runner()
    assert run.DRAFT_ALLOWED == ("simulate",)
    for argv in (
        ["closed", "--output", str(tmp_path / "a")],  # no cohort
        ["closed", "--cohort", "D", "--output", str(tmp_path / "b"), "--evidence", "e"],
        ["closed", "--cohort", "S", "--output", str(tmp_path / "c"), "--evidence", "e",
         "--old-features", "f", "--old-fits", "g", "--models", "m", "--stage-r", "r"],
        ["simulate", "--output", str(tmp_path / "d"), "--workers", "2"],
        ["simulate", "--output", str(tmp_path / "e"), "--cohort", "D"],
        ["simulate", "--output", str(tmp_path / "f"), "--debug-skip-tests"],
    ):  # fmt: skip
        with pytest.raises(SystemExit):
            run.main(argv)


def test_the_preflight_refuses_closed_while_draft(monkeypatch):
    run = _runner()
    from types import SimpleNamespace

    monkeypatch.setattr(run.rt, "assert_local_import", lambda *a, **k: None)
    monkeypatch.setattr(run.hz, "check_pins", lambda pins, *a: dict(pins))
    monkeypatch.setattr(run, "check_task077", lambda report: None)
    monkeypatch.setattr(run, "check_task080", lambda report: None)
    monkeypatch.setattr(run.hz, "tracked_tree_dirty", lambda *a: [])
    monkeypatch.setattr(run.cpv, "STATUS", "DRAFT")
    monkeypatch.setattr(run, "check_frozen_pin", lambda manifest: {"status": "DRAFT"})
    args = SimpleNamespace(stage="closed", debug=False, debug_skip_tests=False)
    with pytest.raises(pt.GuardError, match="runs only after the freeze"):
        run.preflight({}, args)


def test_stage_d_report_must_be_this_tasks_d_pass(tmp_path):
    run = _runner()
    from types import SimpleNamespace

    good = {"task": "TASK-081", "protocol": cpv.PROTOCOL, "outcome": "D-PASS", "debug": False,
            "cohort": "D"}  # fmt: skip
    for bad in ({"task": "TASK-080"}, {"outcome": "L-DEV-STOP"}, {"debug": True},
                {"cohort": "S"}):  # fmt: skip
        path = tmp_path / f"{len(list(tmp_path.iterdir()))}.json"
        path.write_text(json.dumps(good | bad))
        with pytest.raises(pt.GuardError):
            run.stage_d_report(SimpleNamespace(stage_d=str(path)), False)
    path = tmp_path / "ok.json"
    path.write_text(json.dumps(good))
    assert run.stage_d_report(SimpleNamespace(stage_d=str(path)), False)["sha256"]


def test_the_stage_r_report_is_pinned(tmp_path):
    run = _runner()
    from types import SimpleNamespace

    path = tmp_path / "report.json"
    path.write_text(json.dumps({"outcome": "R-PASS", "debug": False}))
    with pytest.raises(pt.GuardError, match="not TASK-080's"):
        run.stage_r_readouts(SimpleNamespace(stage_r=str(path)), {})


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
            solver = cpv.SOLVER_OF.get(arm)
            base = {"seed": s, "arm": arm, "blocked": None, "privileged_ok": True,
                    "task_truth_in_controller": 0, "seconds": 1.0, "solver": solver,
                    "termination_reason": "step_limit", "executed_steps": 800,
                    "final_distance_cm": 3.6}  # fmt: skip
            if s in self.refused:
                out.append(base | {"success": False, "termination_reason": "guard_refusal",
                                   "executed_steps": 230, "frame405": None,
                                   "decisions": []})  # fmt: skip
                continue
            good = arm in ("W", "W-frozen", "H-final", "H-read", "H-rule", "H-sysid")
            wm = {"solver": solver, "converged": arm != "W-frozen" or s % 3 != 0}
            if solver == "affine_local":
                wm |= {"points": 25, "residual_cm": 0.3, "eigenvalues_real": [-0.5, -0.4],
                       "fallback_grid_argmin": False, "clipped": False}  # fmt: skip
            out.append(base | {
                "success": good, "frame405": np.full((2, 2, 3), s % 251, np.uint8),
                "decisions": [{"reading": [0.49, -0.09], "world_model": wm}],
                "commit": {"target": [0.5, -0.1], "fallback": False, "clipped": False,
                           "fixed_point_error_cm": 0.4, "landing_miss_cm": 0.3},
            })  # fmt: skip
        return out


@pytest.mark.parametrize("role", ["D", "S"])
def test_closed_core_runs_every_arm_with_its_solver(role):
    run = _runner()
    n = 16 if role == "D" else 128
    seeds = cpv.seeds_of(role)
    refused = {seeds[1], seeds[2]}
    pool = _FakePool(refused)
    report: dict = {}
    fields = run.Fields(report, "closed")
    row = run.closed_core(pool, _FakeCohort(seeds), seeds, role,
                          {"tau_commit_cm": 1.0, "a_lo": cpv.A_LO}, [[0.0]], fields,
                          debug=False)  # fmt: skip
    arms = report["fields"]["arms"]
    want = cpv.D_ARMS if role == "D" else cpv.S_ARMS
    assert set(arms) - {"refused_before_405"} == set(want)
    assert arms["W"]["count"] == n - 2 and arms["refused_before_405"]["W"] == 2
    shuf = next(c for c in pool.calls if c[0]["arm"] == "L-shuf")
    assert int(shuf[0]["foreign_frame"][0, 0, 0]) == seeds[3] % 251
    assert pool.calls[0][0]["move_offset"] == cpv.move_offset(seeds[0]).tolist()
    rep = report["fields"]["reported"]
    assert rep["solver_effect"]["difference"] == 0
    assert rep["per_arm"]["W"]["affine_local"]["decisions"] == n - 2
    assert rep["per_arm"]["W"]["final_distance_cm"]["within_0_5_of_4cm"] == n
    if role == "S":
        assert row == "L-PASS" and report["fields"]["determinism"]["ok"]
    else:
        assert row == "D-PASS"
        assert report["fields"]["determinism"] == cpv.NOT_EVALUATED


def test_run_arm_refuses_a_wrong_solver_and_task_truth():
    run = _runner()

    class Pool:
        def __init__(self, rec):
            self.rec = rec

        def map(self, tasks, cap, what):
            return [self.rec]

    ok = {"seed": 1, "arm": "W", "success": True, "blocked": None, "privileged_ok": True,
          "task_truth_in_controller": 0, "solver": "affine_local",
          "decisions": [{"world_model": {"solver": "affine_local"}}]}  # fmt: skip
    run.run_arm(Pool(ok), [{"arm": "W"}], "W")
    with pytest.raises(pt.GuardError, match="G-solver"):
        run.run_arm(Pool(ok | {"solver": "frozen"}), [{"arm": "W"}], "W")
    with pytest.raises(pt.GuardError):
        run.run_arm(Pool(ok | {"task_truth_in_controller": 1}), [{"arm": "W"}], "W")


def test_the_sentinel():
    run = _runner()
    report: dict = {}
    fields = run.Fields(report, "closed")
    fields.check()
    report["fields"]["decision"] = {"row": "L-PASS"}
    with pytest.raises(pt.GuardError):
        fields.check()


def test_stage_modules_import_without_torch_or_mujoco():
    code = (
        "import sys\n"
        "import embodied_jepa.lewm_cp_v2, embodied_jepa.lewm_cp_v2_runtime\n"
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
