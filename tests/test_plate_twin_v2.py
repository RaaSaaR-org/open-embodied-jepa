"""TASK-076 (``apple_plate_twin_v2``): the preregistered design, its guards and its mechanics.

NumPy tests run in the core job; the runner's CLI test import-skips without torch and MuJoCo;
nothing here simulates a cohort or smoke seed.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_jepa import lewm_planner_v2 as lp
from embodied_jepa import obs_ceiling_v2 as oc
from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa import plate_twin_v2_offline as off
from embodied_jepa import run_tools as rt
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "benchmarks" / "manifests" / "apple-plate-twin-v2.json").read_text())
T075 = json.loads((ROOT / "benchmarks" / "manifests" / "apple-obs-ceiling-v2.json").read_text())
RUNNER = ROOT / "scripts" / "run_plate_twin_v2.py"
STAGE_FILES = (
    "src/embodied_jepa/plate_twin_v2.py",
    "src/embodied_jepa/plate_twin_v2_runtime.py",
    "src/embodied_jepa/plate_twin_v2_offline.py",
    "src/embodied_jepa/plate_twin_v2_harness.py",
    "scripts/run_plate_twin_v2.py",
)
# The frozen block's sha256, set at the freeze (protocol §13.2 step 4). None while DRAFT.
FROZEN_SHA256_PIN = None


def _runner():
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    spec = importlib.util.spec_from_file_location("_plate_twin_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----- the frozen block, the pin, the manifest ----------------------------------------------------
def test_manifest_frozen_block_equals_the_module():
    assert MANIFEST["frozen"] == pt.frozen_block()
    assert MANIFEST["frozen_sha256"] == pt.frozen_sha256()
    assert MANIFEST["protocol"] == pt.PROTOCOL and MANIFEST["task"] == pt.TASK


def test_the_frozen_sha_pin_is_set_at_the_freeze():
    """While the protocol is a DRAFT the pin is None; the freeze sets it, and from then on the
    module's frozen block may not change."""
    assert MANIFEST["frozen_sha256_pin"] == FROZEN_SHA256_PIN
    if FROZEN_SHA256_PIN is None:
        assert MANIFEST["status"] == pt.STATUS == "DRAFT"
        doc = (ROOT / "docs" / "experiments" / "apple_plate_twin_v2.md").read_text()
        assert "**Status: DRAFT" in doc
    else:
        assert pt.frozen_sha256() == FROZEN_SHA256_PIN
        assert MANIFEST["status"] != "DRAFT"


def test_pinned_files_match_and_task075_pins_are_unchanged():
    for relative, want in MANIFEST["hashes"].items():
        got = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert got == want, relative
    for relative, want in T075["hashes"].items():
        assert MANIFEST["hashes"][relative] == want, relative
    for own in (*STAGE_FILES, "tests/test_plate_twin_v2.py", "src/embodied_jepa/run_tools.py"):
        assert own in MANIFEST["hashes"], own


def test_the_protocol_document_sha_is_recorded():
    doc = ROOT / "docs" / "experiments" / "apple_plate_twin_v2.md"
    assert hashlib.sha256(doc.read_bytes()).hexdigest() == MANIFEST["protocol_document_sha256"]


def test_carried_facts_are_task074s_and_task075s():
    assert pt.SOURCE_CORPUS["manifest_sha256"] == oc.SOURCE_CORPUS["manifest_sha256"]
    assert pt.SOURCE_CORPUS["manifest_sha256"].startswith("fe7ab915")
    assert pt.VIEWS_STORE["manifest_sha256"].startswith("ea627a8f")
    assert pt.VIEWS_STORE["manifest_sha256"].endswith("4d77")
    results = T075["results"]["stages"][0]
    assert results["views_manifest_sha256"] == pt.VIEWS_STORE["manifest_sha256"]
    assert pt.P3_CHECKPOINT_SHA256.startswith("7988162d") and pt.CARRIED == "P-3"
    assert pt.EXPERT == {"release_pitch_rad": 0.45, "release_dx": 0.015}
    assert pt.EXPERT_BUDGETS == (130, 80, 45, 150, 100, 50, 30, 50, 30, 60)
    assert (pt.TRANSFER_START, pt.LOWER_START, pt.PLACE_END) == (405, 505, 725)
    assert pt.DECISION_STEPS == (405, 421, 437, 453, 469, 485)
    assert pt.EVAL_STEPS == (421, 437, 453, 469, 485, 501)
    assert pt.CONDITION["shift_step"] == 300 and pt.CONDITION["shift_cm"] == 9
    assert pt.READ_ROOTS == {"train": 225, "val": 28, "total": 253}


# ----- seeds and salts (§4) -----------------------------------------------------------------------
def test_seed_ranges_are_fresh_disjoint_and_inside_the_task_block():
    pt.check_seed_ranges()
    assert pt.SEED_RANGES == {
        "K": (56000, 56031),
        "M-a": (56040, 56071),
        "M-b": (56080, 56111),
        "D": (56120, 56135),
        "A": (56160, 56191),
        "S": (56200, 56263),
        "U": (56300, 56331),
    }
    assert pt.SMOKE_SEEDS == (56900, 56999)
    sizes = {r: len(pt.seeds_of(r)) for r in pt.COHORT_ROLES}
    assert sizes == {"K": 32, "M-a": 32, "M-b": 32, "D": 16, "A": 32, "S": 64, "U": 32}
    assert pt.FORBIDDEN_RANGES["task075_block"] == (55000, 55999)
    assert pt.FORBIDDEN_RANGES["task074_block"] == (54000, 54999)
    assert pt.FORBIDDEN_RANGES["cohort_C"] == (45300, 45339)
    assert pt.FORBIDDEN_RANGES["task071_072_block"] == (51000, 52199)
    assert pt.FORBIDDEN_RANGES["arena_123"] == (50200, 50231)
    for name, (low, high) in oc.FORBIDDEN_RANGES.items():
        assert pt.FORBIDDEN_RANGES[name] == (low, high)
    for name, (low, high) in pt.FORBIDDEN_RANGES.items():
        assert high < 56000 or low > 56999, name


def test_salts_are_the_declared_ones_and_fresh():
    assert pt.SALTS == {
        "tau_direction": 7601,
        "outer_folds": 7602,
        "inner_folds": 7603,
        "bootstrap": 7604,
        "moving_direction": 7606,
    }
    assert pt.RESERVED_SALTS == {"unused_learning_curve": 7605}
    assert pt.CARRIED_SALTS == {"shift_direction": 7413, "reset_redraw": 7425}
    for folder in ("src", "scripts", "configs"):
        for path in (ROOT / folder).rglob("*.py"):
            if path.name.startswith(("plate_twin_v2", "run_plate_twin_v2")):
                continue
            text = path.read_text()
            for salt in (*pt.SALTS.values(), *pt.RESERVED_SALTS.values()):
                assert not re.search(rf"(?<![0-9a-f]){salt}(?![0-9a-f])", text), (path, salt)


def test_role_seeds_are_guarded():
    assert pt.check_role_seeds("K", pt.seeds_of("K")) == pt.seeds_of("K")
    assert pt.check_role_seeds("smoke", (56900, 56950), smoke=True) == (56900, 56950)
    for bad in ((56899,), (55000,), (56000,), (56900, 56900)):
        with pytest.raises(lp.GuardError):
            pt.check_role_seeds("smoke", bad, smoke=True)
    with pytest.raises(lp.GuardError):
        pt.check_role_seeds("K", pt.seeds_of("K")[:-1])
    with pytest.raises(lp.GuardError):
        pt.check_role_seeds("S", (54500,))
    with pytest.raises(lp.GuardError):
        pt.check_role_seeds("K", (np.int64(56000),))


def test_cohort_digests_are_recorded_and_draws_use_condition_reset():
    assert set(MANIFEST["cohorts"]) == set(pt.COHORT_ROLES)
    for role in pt.COHORT_ROLES:
        assert pt.cohort_digest(role) == MANIFEST["cohorts"][role], role
    values = pt.cohort_values("A", (56160,))["56160"]
    reset = lp.condition_reset(56160)
    assert values["object_xy"] == reset["object_xy"] and "shift_m" not in values
    assert "moving_m" not in values
    s = pt.cohort_values("S", (56200,))["56200"]
    assert s["shift_m"] == lp.shift_vector(56200, lp.condition_reset(56200), 9)
    m = pt.cohort_values("M-a", (56040,))["56040"]
    assert np.isclose(np.linalg.norm(m["moving_m"]), 0.12)
    assert "moving_m" not in pt.cohort_values("U", (56300,))["56300"]


def test_moving_vector_is_on_the_minus_y_arc_with_its_own_salt():
    reset = lp.condition_reset(56040)
    v = np.asarray(pt.moving_vector(56040, reset, 12))
    assert np.isclose(np.linalg.norm(v), 0.12)
    angle = np.degrees(np.arctan2(v[1], v[0])) % 360
    assert 225 - 1e-9 <= angle <= 315 + 1e-9
    draws = [pt.moving_vector(s, lp.condition_reset(s), 9) for s in range(56040, 56060)]
    shifts = [lp.shift_vector(s, lp.condition_reset(s), 9) for s in range(56040, 56060)]
    assert draws != shifts  # salt 7606, not TASK-074's 7413


# ----- budgets, training, the harness rules (§5, §7) ----------------------------------------------
def test_training_budget_is_none_and_any_later_budget_must_pass_check_budget():
    assert pt.TRAINING_BUDGET is None
    for module in ("plate_twin_v2", "plate_twin_v2_runtime", "plate_twin_v2_offline"):
        mod = importlib.import_module(f"embodied_jepa.{module}")
        budget = getattr(mod, "BUDGET", None)
        if budget is not None:  # a reviewed amendment that adds training must pass the rule
            assert rt.check_budget(budget)["consistent"], module
    # TASK-074's block escalates by design, which is why any later BUDGET is checked
    assert not rt.check_budget(lp.BUDGET)["consistent"]


def test_no_stage_code_trains_iteratively():
    forbidden = (
        "torch.optim",
        ".backward(",
        "optimizer",
        "train_model",
        "zero_grad",
        "lr_scheduler",
        "select_every",
    )
    for relative in STAGE_FILES:
        text = (ROOT / relative).read_text()
        for word in forbidden:
            assert word not in text, (relative, word)


def _calls(path: Path) -> list[ast.Call]:
    return [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Call)]


def _name(call: ast.Call) -> str:
    f = call.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        return f"{f.value.id}.{f.attr}"
    return getattr(f, "id", getattr(f, "attr", ""))


def test_the_runner_uses_run_tools_guards():
    calls = {_name(c): c for c in _calls(RUNNER)}
    for needed in (
        "rt.assert_local_import",
        "rt.install_guards",
        "rt.gpu_guard",
        "rt.MemoryWatch",
        "rt.scale_probe",
        "rt.CycledReader",
        "rt.SlotProxy",
    ):
        assert needed in calls, needed
    guard = calls["rt.gpu_guard"]
    keywords = {k.arg: ast.unparse(k.value) for k in guard.keywords}
    assert keywords["require_lock"] == "pt.GPU['require_lock']" and pt.GPU["require_lock"] is True
    assert pt.GPU["min_free_gib"] == 1.0 and pt.GPU["process_cap_gib"] == 3.0
    assert "--board --who oej:task076-O" in pt.GPU["launcher"]


def test_the_scale_probe_calls_the_stages_own_function():
    calls = [c for c in _calls(RUNNER) if _name(c) == "rt.scale_probe"]
    assert len(calls) == 1
    assert ast.unparse(calls[0].args[0]) == "off.offline_core"
    assert "probe" in inspect.signature(off.offline_core).parameters
    runner = RUNNER.read_text()
    assert "off.offline_core(report, reader, views, clock, **kwargs)" in runner


def test_the_runner_loads_no_other_script():
    sys.path.insert(0, str(ROOT / "tests"))
    try:
        from test_no_runner_imports import script_loads
    finally:
        sys.path.pop(0)
    loaded, loaders = script_loads(RUNNER)
    assert not loaded and not loaders


def test_stage_modules_import_without_torch_or_mujoco():
    code = (
        "import sys; import embodied_jepa.plate_twin_v2, embodied_jepa.plate_twin_v2_runtime, "
        "embodied_jepa.plate_twin_v2_offline, embodied_jepa.plate_twin_v2_harness; "
        "print('torch' in sys.modules, 'mujoco' in sys.modules)"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "False False"


def test_caps_are_at_least_five_times_the_estimates():
    caps, est = pt.CAPS_SECONDS, pt.ESTIMATES_SECONDS
    assert caps["invocation"] >= pt.CAP_FACTOR_MIN * est["invocation_max"]
    assert caps["per_attempt"] >= pt.CAP_FACTOR_MIN * est["attempt_lookahead"]
    assert caps["featurisation"] >= pt.CAP_FACTOR_MIN * est["featurisation"]
    assert pt.STAGE0_ATTEMPT_REVIEW_SECONDS == 60.0


# ----- K-pred's plate motion (§3.2) ---------------------------------------------------------------
def test_constant_velocity_moves_from_s0_to_s1_only():
    base, vec = np.array([0.4, -0.1]), np.array([0.0, -0.12])
    assert np.allclose(pt.constant_velocity_xy(base, vec, 405, 405, 525), base)
    assert np.allclose(pt.constant_velocity_xy(base, vec, 300, 405, 525), base)
    assert np.allclose(pt.constant_velocity_xy(base, vec, 465, 405, 525), base + vec / 2)
    assert np.allclose(pt.constant_velocity_xy(base, vec, 600, 405, 525), base + vec)
    to_go = pt.constant_velocity_xy(base, vec, 525, 405, 525) - pt.constant_velocity_xy(
        base, vec, 485, 405, 525
    )
    assert np.isclose(100 * np.linalg.norm(to_go), pt.M_CELLS["M-a"]["distance_to_go_cm"])


def test_palm_driven_rule_uses_the_lagged_palm_and_freezes_after_s1():
    base = np.array([0.4, -0.1])
    palm = {t: np.array([0.001 * (t - 400), 0.0]) for t in range(400, 600)}
    k, lag, s0, s1 = -0.5, 2, 405, 525
    assert np.allclose(pt.palm_driven_xy(base, palm, 405, k, lag, s0, s1), base)
    assert np.allclose(pt.palm_driven_xy(base, palm, 406, k, lag, s0, s1), base)  # palm(405)
    want = base + k * (palm[408] - palm[405])
    assert np.allclose(pt.palm_driven_xy(base, palm, 410, k, lag, s0, s1), want)
    final = pt.palm_driven_xy(base, palm, s1, k, lag, s0, s1)
    assert np.allclose(pt.palm_driven_xy(base, palm, 700, k, lag, s0, s1), final)
    remaining = final - pt.palm_driven_xy(base, palm, 485, k, lag, s0, s1)
    assert np.allclose(remaining, k * (palm[523] - palm[483]))


def test_kappa_and_lag_limits():
    pt.check_kappa(-0.5)
    for bad in (-1.0, 1.0, 1.5, -2.0):
        with pytest.raises(lp.GuardError):
            pt.check_kappa(bad)
    pt.check_lag(2)
    pt.check_lag(1)
    for bad in (0, 3, 40, 2.0):
        with pytest.raises(lp.GuardError):
            pt.check_lag(bad)
    assert pt.CELL_A == pt.CELL_A | {"kappa": -0.5, "L": 2, "s1": 525}
    assert pt.LOOKAHEAD["tolerance_fraction_of_tau"] == 0.25
    assert pt.LOOKAHEAD["max_iterations"] == 10


def test_m2_and_the_stage0_verdict_and_remedies():
    assert np.isclose(pt.m2_cm([0.0, 0.0], [0.006, 0.008]), 0.5)
    ok = pt.stage0_cell_a_verdict([2.5, 2.1, 3.0, 1.0], refused=1, attempts=5)
    assert ok["feasible"] and np.isclose(ok["median_remaining_cm"], 2.3)
    short = pt.stage0_cell_a_verdict([1.0, 1.5, 2.5], refused=0, attempts=3)
    assert not short["feasible"] and not short["motion_ok"]
    refused = pt.stage0_cell_a_verdict([3.0, 3.0], refused=2, attempts=4)
    assert not refused["feasible"] and not refused["refused_ok"]
    assert pt.stage0_cell_a_verdict([], refused=4, attempts=4)["feasible"] is False
    assert pt.remedy_settings(555) == [
        {"L": 1, "s1": 525},
        {"L": 1, "s1": 535},
        {"L": 1, "s1": 545},
    ]  # strictly before the earliest contact
    assert pt.remedy_settings(556)[-1] == {"L": 1, "s1": 555}
    assert pt.remedy_settings(None)[-1] == {"L": 1, "s1": 715}  # before the place's end
    assert all(r["L"] == 1 for r in pt.remedy_settings(None))


# ----- the runtime pieces (no simulator) ----------------------------------------------------------
def test_cv_extrapolation_is_a_least_squares_line():
    from embodied_jepa.plate_twin_v2_runtime import cv_extrapolate

    one = cv_extrapolate([(405, np.array([0.4, -0.1]))], 525)
    assert np.allclose(one, [0.4, -0.1])
    line = [(t, np.array([0.4, -0.1 - 0.001 * (t - 405)])) for t in (405, 421, 437)]
    assert np.allclose(cv_extrapolate(line, 525), [0.4, -0.1 - 0.12])


def test_cell_motion_hook_applies_the_rule_records_and_restores(monkeypatch):
    from embodied_jepa import plate_twin_v2_runtime as ptr

    plate = {"xy": np.array([0.40, -0.10])}
    moves = []

    def fake_move(sim, delta):
        moves.append(np.asarray(delta, float).copy())
        plate["xy"] = plate["xy"] + np.asarray(delta)
        return {}

    monkeypatch.setattr(ptr.ps, "move_plate", fake_move)
    monkeypatch.setattr(ptr, "plate_xy_now", lambda sim: plate["xy"].copy())
    monkeypatch.setattr(ptr, "apple_plate_contact", lambda sim: False)
    state = {"t": 0}

    class Robot:
        sim = object()

        def observe(self):
            t = state["t"]
            state["t"] += 1
            return SimpleNamespace(t=t)

    class FK:
        def pose9(self, values):
            return np.array([0.002 * max(values - 400, 0), 0.001, 0.0])

    monkeypatch.setattr(ptr.rt2, "state_of", lambda obs: obs.t)
    robot = Robot()
    hook = ptr.CellMotion(robot, FK(), {"name": "A", "s0": 405, "s1": 525, "kappa": -0.5, "L": 2})
    for _ in range(530):
        robot.observe()
    s = hook.summary()
    palm = hook.state["palm"]
    assert np.allclose(
        hook.state["plate"][525], np.array([0.40, -0.10]) - 0.5 * (palm[523] - palm[405])
    )
    assert np.isclose(s["remaining_cm"], 100 * 0.5 * np.linalg.norm(palm[523] - palm[483]))
    assert np.isclose(s["m2_cm"], 100 * 0.5 * np.linalg.norm(palm[485] - palm[483]))
    # the first non-zero move is at 408: plate(406) and plate(407) read palm(405) - palm(405)
    assert len(moves) == 525 - 408 + 1
    saved = hook.snapshot()
    hook.state["palm"][999] = np.zeros(2)
    hook.restore(saved)
    assert 999 not in hook.state["palm"]
    hook.remove()
    assert "observe" not in vars(robot)


# ----- K0 (§5) ------------------------------------------------------------------------------------
def test_tau_re_rule_and_planted_direction():
    counts = {
        str(c): n
        for c, n in zip(pt.TAU["levels_cm"], (31, 31, 28, 22, 23, 21, 15, 11, 8), strict=True)
    }
    assert pt.tau_from_counts(counts)["tau_re_cm"] == 1.0
    assert pt.tau_from_counts(counts | {"0.0": 27})["tau_re_cm"] is None
    with pytest.raises(ContractError):
        pt.tau_from_counts({"0.0": 30})
    e1, e2 = pt.planted_error_m(56000, 1.0), pt.planted_error_m(56000, 2.0)
    assert np.isclose(np.linalg.norm(e1), 0.01) and np.allclose(2 * np.asarray(e1), e2)
    assert pt.tau_direction(56000) != oc.tau_direction(56000)  # salt 7601, not 7501


def test_clock_targets_are_the_per_step_median():
    plate_at = {
        s: {t: [0.4 + 0.01 * i, -0.2] for t in pt.DECISION_STEPS} for i, s in enumerate((1, 2, 3))
    }
    targets = pt.clock_targets(plate_at)
    assert set(targets) == set(pt.DECISION_STEPS)
    assert np.allclose(targets[405], [0.41, -0.2])


def test_k0_stops():
    counts = {str(c): 31 for c in pt.TAU["levels_cm"]}
    assert pt.decide_k0(counts, n_clock=20, n_stale=0)["row"] == "K0-PASS"
    assert pt.decide_k0(counts | {"0.0": 29}, 20, 0)["stops"]["ceiling_below_30"]
    assert pt.decide_k0(counts | {"0.0": 27}, 20, 0)["stops"]["level0_below_bar"]
    assert pt.decide_k0(counts, 20, 5)["row"] == "CAL-ESCALATE"
    assert pt.decide_k0(counts, 27, 0)["stops"]["clock_near_ceiling"]
    assert not pt.decide_k0(counts, 26, 0)["stops"]["clock_near_ceiling"]


def test_mcnemar_minimum_separation_and_g2_feasibility():
    assert pt.mcnemar_one_sided(7, 0) == 0.5**7 < 0.01
    assert pt.mcnemar_one_sided(6, 0) > 0.01
    assert pt.mcnemar_one_sided(10, 1) < 0.01 < pt.mcnemar_one_sided(9, 1)
    handover = [True] * 30 + [False] * 2
    clock = [True] * 20 + [False] * 12
    g = pt.g2_feasibility(handover, clock, resamples=500)
    assert (g["b"], g["c"]) == (10, 0)
    assert g["point_p_doubled"] == 0.5**20
    assert 0.9 < g["predicted_pass_probability"] <= 1.0


# ----- the rows -----------------------------------------------------------------------------------
def test_offline_rows_first_match():
    good = {"o1_upper": 0.6, "o3_ratio_upper": 0.5, "o4_lower": 3.0}
    assert pt.decide_offline(good, 1.0)["row"] == "O-PASS"
    assert pt.decide_offline(good | {"o4_lower": 0.9}, 1.0)["row"] == "TWIN-OFF-ARM"
    assert pt.decide_offline(good | {"o4_lower": 0.9, "o1_upper": 2.0}, 1.0)["row"] == (
        "TWIN-OFF-ARM"
    )
    fail = pt.decide_offline(good | {"o3_ratio_upper": 1.0}, 1.0)
    assert fail["row"] == "TWIN-OFF-FAIL" and fail["clause_fires"]
    assert pt.decide_offline(good, 1.0, void=True)["row"] == "V"


def test_dev_rows():
    ok = {"H-twin": [True] * 12 + [False] * 4, "H-handover": [True] * 14 + [False] * 2}
    assert pt.decide_dev(ok)["row"] == "D-PASS"
    assert pt.decide_dev(ok | {"H-twin": [True] * 11 + [False] * 5})["row"] == "TWIN-DEV-STOP"


def _arms(counts: dict, n: int) -> dict:
    return {arm: [True] * c + [False] * (n - c) for arm, c in counts.items()}


def test_gated_rows_first_match():
    s = {"H-twin": 60, "H-handover": 62, "H-clock": 40, "H-stale": 2, "H-floor": 30, "P-stale": 0}
    u = {"H-twin": 30, "P-stale": 31, "H-handover": 31}
    row = pt.decide_gated(_arms(s, 64), _arms(u, 32), resamples=200)
    assert row["row"] == "TWIN-PASS" and row["claim"]
    assert (
        pt.decide_gated(_arms(s | {"H-stale": 9}, 64), _arms(u, 32), resamples=200)["row"]
        == "S-VOID-CONDITION"
    )
    assert (
        pt.decide_gated(_arms(s | {"H-handover": 55}, 64), _arms(u, 32), resamples=200)["row"]
        == "S-VOID-CONDITION"
    )
    assert (
        pt.decide_gated(_arms(s, 64), _arms(u | {"H-handover": 28}, 32), resamples=200)["row"]
        == "U-VOID-CEILING"
    )
    harm = pt.decide_gated(_arms(s, 64), _arms(u | {"H-twin": 28}, 32), resamples=200)
    assert harm["row"] == "TWIN-HARM" and harm["clause_fires"]
    assert (
        pt.decide_gated(_arms(s | {"H-clock": 58}, 64), _arms(u, 32), resamples=200)["row"]
        == "TWIN-PRIOR"
    )
    near = pt.decide_gated(
        _arms(s | {"H-twin": 55, "H-handover": 57}, 64), _arms(u, 32), resamples=500
    )
    assert near["row"] == "TWIN-NEAR" and not near["clause_fires"]
    fail = pt.decide_gated(
        _arms(s | {"H-twin": 40, "H-handover": 63}, 64), _arms(u, 32), resamples=500
    )
    assert fail["row"] == "TWIN-FAIL" and fail["clause_fires"]
    assert pt.decide_gated({}, {}, void=True)["row"] == "V"


def test_kpred_rows_and_the_noise_guard():
    kw = {"offline_row": "O-PASS", "c_plate_cm": 0.6, "tau_re_cm": 1.0, "resamples": 2000}
    admit = _arms({"H-final": 31, "H-now": 10, "H-twin": 12, "H-cv": 14, "H-rule": 30}, 32)
    assert pt.decide_kpred(admit, **kw)["row"] == "PRED-ADMIT(A)"
    assert pt.decide_kpred(admit, **kw | {"c_plate_cm": 1.2})["row"] == "PRED-NO-BAR"
    assert pt.decide_kpred(admit, **kw | {"offline_row": "TWIN-OFF-FAIL"})["row"] == (
        "PRED-NOT-RUN"
    )
    assert pt.decide_kpred(None, **kw)["row"] == "PRED-INFEASIBLE"
    low = admit | _arms({"H-final": 29}, 32)
    assert pt.decide_kpred(low, **kw)["row"] == "PRED-INFEASIBLE"
    none = admit | _arms({"H-cv": 31}, 32)
    out = pt.decide_kpred(none, **kw)
    assert out["row"] == "PRED-NONE" and out["clause_fires"]
    near = admit | _arms({"H-final": 32, "H-cv": 25}, 32)  # a headroom of 7/32
    out = pt.decide_kpred(near, **kw)
    assert out["row"] == "PRED-NEAR" and not out["clause_fires"]
    assert out["headrooms"]["K-P4"]["ci95"][1] >= 8


# ----- Stage O on synthetic features --------------------------------------------------------------
def _synthetic(n_roots=20, seed=0):
    rng = np.random.default_rng(seed)
    roots = [f"r{i:02d}" for i in range(n_roots)]
    vis_root, vis_step, hid_root, hid_step, plates, hplates = [], [], [], [], [], []
    for r in roots:
        p = rng.uniform(-0.1, 0.1, 2)
        for t in pt.DECISION_STEPS:
            vis_root.append(r)
            vis_step.append(t)
            plates.append(p + rng.normal(0, 0.001, 2))
        for t in pt.EVAL_STEPS:
            hid_root.append(r)
            hid_step.append(t)
            hplates.append(p)
    y, hy = np.asarray(plates), np.asarray(hplates)
    w = rng.normal(size=(2, 40))
    feats = {
        "full": (y @ w + rng.normal(0, 0.01, (len(y), 40))).astype(np.float32),
        "floor_full": rng.normal(size=(len(y), 40)).astype(np.float32),
        "pooled": (y @ w[:, :20]).astype(np.float32) + 0.01,
        "hidden_full": rng.normal(size=(len(hy), 40)).astype(np.float32),
    }
    rows = {
        "roots": roots,
        "vis": {"plate": y, "root": np.asarray(vis_root), "step": np.asarray(vis_step)},
        "hid": {"plate": hy, "root": np.asarray(hid_root), "step": np.asarray(hid_step)},
    }
    return rows, feats


def test_outer_folds_are_seeded_and_balanced():
    roots = [f"r{i}" for i in range(253)]
    folds = off.outer_folds(roots)
    assert folds == off.outer_folds(roots)
    sizes = sorted(np.bincount(list(folds.values())))
    assert sizes[-1] - sizes[0] <= 1
    assert folds != oc.outer_folds(roots)  # salt 7602, not TASK-075's 7502


def test_cross_fit_reads_every_row_out_of_fold_and_the_hidden_check_fails_cleanly():
    rows, feats = _synthetic()
    folds = off.outer_folds(rows["roots"])
    fit = off.cross_fit(rows, feats, folds)
    e = fit["errors"]
    assert len(e["r_plate"]) == len(rows["vis"]["root"])
    assert np.median(e["r_plate"]) < np.median(e["clock"])  # the image adds over the clock
    assert np.median(e["hidden"]) > np.median(e["r_plate"])
    stats = off.statistics(fit, 1.0, dict(oc.TAU_MEASURED["counts"]), resamples=200)
    assert stats["decision"]["row"] == "O-PASS"
    assert set(stats["gates"]) == {"o1_upper", "o3_ratio_upper", "o4_lower"}
    assert stats["reported"]["c_plate_cm"] == stats["median_cm"]["r_plate_pool"]["ci95"][1]


def test_clock_prior_reads_only_the_fit_rows():
    y = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [9.0, 9.0]])
    steps = np.array([405, 405, 405, 421])
    out = off.clock_prior(y[:3], steps[:3], np.array([405, 405]))
    assert np.allclose(out, [[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ContractError):
        off.clock_prior(y[:3], steps[:3], np.array([421]))


# ----- the runner's CLI ---------------------------------------------------------------------------
def test_cli_refuses_misused_flags(tmp_path):
    runner = _runner()
    base = ["--output", str(tmp_path / "x"), "--evidence", str(tmp_path)]
    for argv in (
        ["k0", *base, "--truth-estimates"],
        ["kpred", *base, "--cell", "A", "--kpred-L", "1"],
        ["source", *base, "--source", str(tmp_path / "s")],
        ["kpred", *base, "--smoke"],
        ["offline", *base, "--smoke", "--source", "a", "--views", "b"],
        ["dev", *base, "--offline-report", "r.json"],
    ):
        with pytest.raises(SystemExit):
            runner.main(argv)
    assert not (tmp_path / "x").exists()
