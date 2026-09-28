"""TASK-072 M2 (``apple_first_policy_v2_m2``): P-3 on cohort C.

Software checks only: the frozen block and pins, the stored-values cohort path (the behavioural
test ``task056_handover.md`` §7 left owed for any runner that opens cohort C), the seed
whitelists, the decision rows, G7's timing and G-repro. Nothing here simulates a cohort-C seed.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import platform
from pathlib import Path

import numpy as np
import pytest

from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa import first_policy_v2_m2 as fm
from embodied_jepa import first_policy_v2_runtime as rt2

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "benchmarks" / "manifests"
MANIFEST = json.loads((MANIFESTS / "apple-first-policy-v2-m2.json").read_text())
LINUX_MANIFEST = json.loads((MANIFESTS / "apple-first-policy-v2-linux.json").read_text())
LINUX_RESULTS = json.loads((MANIFESTS / "apple-first-policy-v2-linux-results.json").read_text())
POLICY_V1 = json.loads((MANIFESTS / "apple-policy-v1.json").read_text())


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "_run_first_policy_v2_m2", ROOT / "scripts" / "run_first_policy_v2_m2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----- the manifest, pins and evidence -----
def test_manifest_frozen_block_matches_the_module():
    assert MANIFEST["frozen"] == fm.frozen_block()


def test_pins_carry_task072_and_add_the_new_files():
    pins = MANIFEST["hashes"]
    for path, want in LINUX_MANIFEST["hashes"].items():
        assert pins[path] == want, path
    for path in (
        "src/embodied_jepa/first_policy_v2_m2.py",
        "scripts/run_first_policy_v2_m2.py",
        "docs/experiments/apple_first_policy_v2_m2.md",
        "benchmarks/manifests/apple-policy-v1.json",
        "benchmarks/manifests/apple-first-policy-v2-linux.json",
        "benchmarks/manifests/apple-first-policy-v2-linux-results.json",
    ):
        assert path in pins, path
    for path, want in pins.items():
        got = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        assert got == want, path


def test_encoder_pins_are_task072s():
    assert MANIFEST["encoder_digests"] == LINUX_MANIFEST["encoder_digests"]
    assert MANIFEST["mujoco_version"] == LINUX_MANIFEST["mujoco_version"]


def test_evidence_is_task072_run1_as_its_results_manifest_records():
    e = fm.EVIDENCE
    for arm, sha in e["checkpoints"].items():
        assert LINUX_RESULTS["checkpoints"][arm]["sha256"] == sha
        assert LINUX_RESULTS["checkpoints"][arm]["path"] == f"{e['checkpoint_dir']}/{arm}.pt"
    text = json.dumps(LINUX_RESULTS)
    for value in (e["report_sha256"], e["corpus_manifest_sha256"], e["revision"]):
        assert value in text
    assert LINUX_RESULTS["outcome"] == e["outcome"] == "M1-PASS"
    assert LINUX_RESULTS["decision"]["carried"] == e["carried"] == fm.CARRIED
    assert LINUX_RESULTS["m1_b_replay_library_roots"] == e["b_replay_library_roots"] == 112


def test_m2_is_task071s_section_12_unchanged():
    assert fm.M2 is fp2.M2
    assert fm.M2["G1_min_successes"] == 17
    assert fm.M2["G2_min_difference_vs_C"] == fm.M2["G3_min_difference_vs_R"] == 8
    assert fm.M2["G4_min_grasps"] == 20
    assert fm.M2["G5_harness"] == {
        "B-hold_grasp": 0,
        "B-random_grasp": 0,
        "B-oracle_min_successes": 38,
    }
    assert fm.M2["G7_max_median_control_seconds"] == 0.100
    assert set(fm.ARMS) == {"P-3", "C-3", "R-3", "B-replay", "B-oracle", "B-hold", "B-random"}
    assert fm.C_RESETS == 40 and fm.COHORT_C == tuple(range(45300, 45340))


# ----- the cohort: stored values, never recomputed -----
def test_stored_cohort_is_the_manifest_values_and_its_digest():
    resets = fm.stored_cohort(POLICY_V1)
    stored = POLICY_V1["cohorts"]["C_frozen_gating"]["resets"]
    assert tuple(resets) == fm.COHORT_C
    for seed, reset in resets.items():
        assert reset["object_xy"] == stored[str(seed)]["object_xy"]
        assert reset["plate_xy"] == stored[str(seed)]["plate_xy"]
    assert fm.cohort_digest(stored) == POLICY_V1["cohorts"]["C_frozen_gating"]["cohort_sha256"]


def test_stored_cohort_refuses_a_tampered_or_short_cohort():
    tampered = copy.deepcopy(POLICY_V1)
    tampered["cohorts"]["C_frozen_gating"]["resets"]["45300"]["object_xy"][0] += 1e-12
    with pytest.raises(fp2.GuardError, match="digest"):
        fm.stored_cohort(tampered)
    short = copy.deepcopy(POLICY_V1)
    del short["cohorts"]["C_frozen_gating"]["resets"]["45339"]
    with pytest.raises(fp2.GuardError):
        fm.stored_cohort(short)


def test_runner_cohort_stage_uses_the_stored_values_and_never_wide_reset():
    runner = load_runner()

    def forbidden(_seed):
        raise AssertionError("wide_reset must not be called for cohort C")

    seeds, resets = runner.cohort_tasks("run", forbidden)
    assert seeds == fm.COHORT_C
    assert resets == fm.stored_cohort(POLICY_V1)
    with pytest.raises(fp2.GuardError):
        runner.cohort_tasks("preflight", forbidden)


def _reset_seen_by_the_robot(runner, monkeypatch, seed: int, reset: dict) -> dict:
    """Run the runner's ``run_task`` up to the robot's reset and return what reached it."""
    seen = {}

    class Stop(Exception):
        pass

    class Robot:
        def reset(self, seed, *, object_xy, plate_xy):
            seen.update(seed=seed, object_xy=list(object_xy), plate_xy=list(plate_xy))
            raise Stop

    monkeypatch.setitem(runner.LIN._W, "robot", Robot())
    with pytest.raises(Stop):
        runner.run_task({"seed": seed, "reset": reset, "kind": "frame"})
    return seen


def test_runner_executes_the_stored_reset(monkeypatch):
    """Behavioural (task056_handover.md §7): what reaches the robot's reset, through run_task,
    is the stored value for every cohort-C seed, and never a recomputation."""
    runner = load_runner()
    wide_reset = runner.LIN.load_wide_reset()
    resets = fm.stored_cohort(POLICY_V1)
    stored = POLICY_V1["cohorts"]["C_frozen_gating"]["resets"]
    recompute_differs = []
    for seed in fm.COHORT_C:
        seen = _reset_seen_by_the_robot(runner, monkeypatch, seed, resets[seed])
        want = {k: stored[str(seed)][k] for k in ("object_xy", "plate_xy")}
        assert seen == {"seed": seed, **want}, seed
        again = wide_reset(seed)
        if [float(v) for v in again["object_xy"]] != want["object_xy"] or [
            float(v) for v in again["plate_xy"]
        ] != want["plate_xy"]:
            recompute_differs.append(seed)
    if platform.system() == "Linux" and platform.machine() == "x86_64":
        # On the Linux PC recomputation differs from the stored decimals by 1 ULP on 6 seeds
        # (45314 among them), so this test tells stored from recomputed values here.
        assert 45314 in recompute_differs
    # On every platform: a reset value that no generator produces must reach the robot as given.
    marker = {"object_xy": [0.3125, -0.1875], "plate_xy": [0.5, -0.0625]}
    assert _reset_seen_by_the_robot(runner, monkeypatch, 45314, marker) == {
        "seed": 45314,
        **marker,
    }


def test_cohort_seed_whitelist_fails_closed():
    assert fm.check_cohort_seeds(fm.COHORT_C) == fm.COHORT_C
    for bad in (
        tuple(str(s) for s in fm.COHORT_C),  # the string bypass of TASK-056
        fm.COHORT_C[:39],
        tuple(reversed(fm.COHORT_C)),
        fp2.COHORT_D2,
        (*fm.COHORT_C[:39], 45300),
    ):
        with pytest.raises(fp2.GuardError):
            fm.check_cohort_seeds(bad)


def test_rerender_whitelist_is_run1s_seeds_and_refuses_cohort_c():
    runner = load_runner()
    for role in ("perception_train", "perception_heldout"):
        assert runner.check_rerender_seeds(role, fp2.seeds_of(role)) == fp2.seeds_of(role)
    assert runner.check_rerender_seeds("D", fp2.COHORT_D2) == fp2.COHORT_D2
    with pytest.raises(fp2.GuardError, match="cohort C"):
        runner.check_rerender_seeds("D", fm.COHORT_C[:2])
    with pytest.raises(fp2.GuardError):
        runner.check_rerender_seeds("perception_train", fp2.seeds_of("perception_heldout"))
    with pytest.raises(fp2.GuardError):
        runner.check_rerender_seeds("D", tuple(str(s) for s in fp2.COHORT_D2))


# ----- the decision -----
def _vec(k: int) -> list[bool]:
    return [True] * k + [False] * (fm.C_RESETS - k)


def _inputs(p=30, c=10, r=12, replay=20, oracle=40, hold_grasp=0, random_grasp=0, p_grasp=None):
    success = {"P-3": _vec(p), "C-3": _vec(c), "R-3": _vec(r), "B-replay": _vec(replay)}
    success |= {"B-oracle": _vec(oracle), "B-hold": _vec(0), "B-random": _vec(0)}
    grasp = {arm: list(v) for arm, v in success.items()}
    grasp["P-3"] = _vec(max(p, p_grasp or 0))
    grasp["B-hold"], grasp["B-random"] = _vec(hold_grasp), _vec(random_grasp)
    ok = dict.fromkeys(fm.LEARNED, True)
    return success, grasp, ok


def test_decide_m2_pass_and_each_row():
    s, g, ok = _inputs()
    assert fm.decide_m2(s, g, ok, 0.001)["row"] == "M2-PASS"
    assert fm.decide_m2(*_inputs(oracle=37), 0.001)["row"] == "M2-VOID"  # G5
    assert fm.decide_m2(*_inputs(hold_grasp=1), 0.001)["row"] == "M2-VOID"  # G5
    s, g, ok = _inputs()
    ok["R-3"] = False
    assert fm.decide_m2(s, g, ok, 0.001)["row"] == "M2-VOID"  # G6
    vision = fm.decide_m2(*_inputs(p=20, c=15, r=5, replay=10), 0.001)
    assert vision["row"] == "M2-FAIL-VISION" and not vision["gates"]["G2"]
    no_r = fm.decide_m2(*_inputs(p=30, r=25), 0.001)
    assert no_r["row"] == "M2-FAIL" and not no_r["gates"]["G3"]  # R-3 within 8: G3's reading
    assert fm.decide_m2(*_inputs(p=16, c=0, r=0, replay=0), 0.001)["row"] == "M2-FAIL"  # G1
    assert fm.decide_m2(*_inputs(), 0.2)["row"] == "M2-FAIL"  # G7


def test_g1_needs_strictly_more_than_b_replay_and_g4_grasps():
    tie = fm.decide_m2(*_inputs(p=25, replay=25), 0.001)
    assert not tie["gates"]["G1"] and tie["row"] == "M2-FAIL"
    assert fm.decide_m2(*_inputs(p=25, replay=24), 0.001)["gates"]["G1"]
    s, g, ok = _inputs(p=19, c=0, r=0, replay=0, p_grasp=19)
    assert not fm.decide_m2(s, g, ok, 0.001)["gates"]["G4"]


def test_a_gate_that_cannot_be_evaluated_fails():
    s, g, ok = _inputs()
    assert fm.decide_m2(s, g, ok, None)["gates"]["G7"] is False
    assert fm.decide_m2(s, g, ok, float("nan"))["gates"]["G7"] is False
    s["P-3"] = g["P-3"] = None  # the stop rule: the carried arm did not run
    decision = fm.decide_m2(s, g, ok, None)
    assert decision["row"] == "M2-FAIL" and not decision["carried_ran"]
    assert not any(decision["gates"][k] for k in ("G1", "G2", "G3", "G4", "G7"))
    s, g, ok = _inputs()
    s["C-3"] = None  # a control must always run
    with pytest.raises(fp2.GuardError):
        fm.decide_m2(s, g, ok, 0.001)


def test_decide_m2_refuses_inconsistent_counts():
    s, g, ok = _inputs()
    g["P-3"] = _vec(0)  # a counted success needs a grasp
    with pytest.raises(fp2.GuardError):
        fm.decide_m2(s, g, ok, 0.001)
    s, g, ok = _inputs()
    s["B-hold"] = [False] * 16
    with pytest.raises(fp2.GuardError):
        fm.decide_m2(s, g, ok, 0.001)


def test_exact_mcnemar():
    assert fm.mcnemar_exact(0, 0) == 1.0
    assert fm.mcnemar_exact(9, 0) == pytest.approx(0.00390625)  # TASK-072 results, P-3 vs C-3
    assert fm.mcnemar_exact(6, 0) == pytest.approx(0.03125)
    assert fm.mcnemar_exact(4, 0) == pytest.approx(0.125)
    assert fm.mcnemar_exact(3, 3) == 1.0
    assert fm.mcnemar_exact(2, 7) == fm.mcnemar_exact(7, 2)
    pair = fm.paired([True, True, False, False], [True, False, True, False])
    assert pair == {"only_first": 1, "only_second": 1, "n_d": 2, "p_exact_mcnemar": 1.0}


def test_stop_rule():
    assert fm.stop_rule_runs_carried({"P-3": {"grasp": 16}})
    assert not fm.stop_rule_runs_carried({"P-3": {"grasp": 0}})


# ----- G7 and G-privileged through the timing wrapper -----
def test_control_times_add_the_forward_pass_to_the_first_command_only():
    runner = load_runner()
    records = [{"act_seconds": [0.001, 0.002, 0.003]}, {"act_seconds": [0.004, 0.005]}]
    t = runner.control_times(records, [0.5, 0.6])
    assert t["commands"] == 5
    assert t["median_seconds"] == pytest.approx(np.median([0.501, 0.002, 0.003, 0.604, 0.005]))
    assert t["max_seconds"] == pytest.approx(0.604)
    assert t["conservative_forward_every_command_median_seconds"] == pytest.approx(0.503)
    assert runner.control_times([], [])["median_seconds"] is None


def test_timed_controller_keeps_privileged_reads_visible():
    runner = load_runner()

    class Sim:
        def task_truth(self):
            return {}

    sim = Sim()

    class Peeking:
        def act(self, observation, step):
            sim.task_truth()
            return np.zeros(14, np.float32)

        def advance(self, result):
            self.advanced = result

    counter = rt2.PrivilegedReadCounter(sim)
    inner = Peeking()
    timed = runner.TimedController(inner)
    counter.act(timed, None, 0)
    counter.act(timed, None, 1)
    timed.advance("r")
    assert counter.in_controller == 2 and len(timed.seconds) == 2
    assert inner.advanced == "r"
    assert not hasattr(timed, "robot") and not hasattr(timed, "sim")


# ----- G-repro -----
def _run1_facts():
    return {
        "stages": {
            "readouts": {
                "P": {"selection": {"family": "linear", "lam_rel": 0.001, "inner_mse": 1.5}},
                "R": {"selection": {"family": "linear", "lam_rel": 0.01, "inner_mse": 2.5}},
                "fit_rows": 426,
            },
            "S0_P": {"apple_errors_cm": [0.1, 0.2], "plate_errors_cm": [0.3]},
            "BC0": {"C_mean_estimates": [0.3, -0.1, 0.5, -0.1]},
            "M1": {"b_replay_library_roots": ["a", "b"], "b_replay_nearest": ["b", "a"]},
        }
    }


def _recomputed():
    return {
        "selection": {
            "P": {"family": "linear", "lam_rel": 0.001, "inner_mse": 1.5},
            "R": {"family": "linear", "lam_rel": 0.01, "inner_mse": 2.5},
        },
        "fit_rows": 426,
        "apple_errors_cm": [0.1, 0.2],
        "plate_errors_cm": [0.3],
        "c_mean": [0.3, -0.1, 0.5, -0.1],
        "library": ["a", "b"],
        "d2_nearest": ["b", "a"],
    }


def test_reproduction_is_exact_or_void():
    runner = load_runner()
    assert all(runner.check_reproduction(_recomputed(), _run1_facts()).values())
    for key, value in (
        ("apple_errors_cm", [0.1, 0.2000000000000001]),
        ("c_mean", [0.3, -0.1, 0.5, -0.10000000000000002]),
        ("d2_nearest", ["a", "b"]),
        ("fit_rows", 425),
    ):
        changed = _recomputed() | {key: value}
        with pytest.raises(fp2.GuardError, match="G-repro"):
            runner.check_reproduction(changed, _run1_facts())


# ----- a V report carries no verdict -----
def _m2_stage():
    success, grasp, ok = _inputs()
    per_reset = {
        a: {"success": success[a], "grasp": grasp[a], "at_rest": success[a]} for a in success
    }
    return {
        "counts": {a: {"success": sum(v)} for a, v in success.items()},
        "per_reset": per_reset,
        "privileged_ok": ok,
        "control_time": {"median_seconds": 0.001},
    }


@pytest.mark.parametrize("fails", ["end_checks", "decide_m2"])
def test_finish_writes_nothing_when_a_check_or_the_decision_raises(monkeypatch, fails):
    runner = load_runner()

    def boom(*_args, **_kwargs):
        raise fp2.GuardError("G-hash: forced")

    if fails == "end_checks":
        monkeypatch.setattr(runner, "end_checks", boom)
    else:
        monkeypatch.setattr(runner, "end_checks", lambda *a: None)
        monkeypatch.setattr(fm, "decide_m2", boom)
    report = {"outcome": None, "stages": {}}
    with pytest.raises(fp2.GuardError):
        runner.finish(report, {}, "run", _m2_stage())
    assert "M2" not in report["stages"] and "decision" not in report
    assert report["outcome"] is None


def test_finish_writes_the_verdict_after_the_checks(monkeypatch):
    runner = load_runner()
    order = []
    monkeypatch.setattr(
        runner, "end_checks", lambda report, *a: order.append(copy.deepcopy(report))
    )
    report = {"outcome": None, "stages": {}}
    runner.finish(report, {}, "run", _m2_stage())
    assert "decision" not in order[0] and "M2" not in order[0]["stages"]
    assert report["outcome"] == "M2-PASS" and report["stages"]["M2"]["counts"]


def test_a_void_run_writes_no_verdict(monkeypatch, tmp_path):
    runner = load_runner()

    def leaky(report, *_args):
        report["stages"]["M2"] = {"counts": {"P-3": {"success": 40}}}
        report["decision"] = {"row": "M2-PASS"}
        report["outcome"] = "M2-PASS"
        raise fp2.GuardError("G-hash: the tracked tree changed during the run")

    monkeypatch.setattr(runner, "_run", leaky)
    runner.run(tmp_path / "out", tmp_path, "run")
    written = json.loads((tmp_path / "out" / "report.json").read_text())
    assert written["outcome"] == "V"
    assert "decision" not in written and "M2" not in written["stages"]


# ----- G-evidence -----
def _fake_evidence(root: Path, monkeypatch, report: dict):
    e = copy.deepcopy(fm.EVIDENCE)
    (root / e["checkpoint_dir"]).mkdir(parents=True)
    (root / e["corpus"]).mkdir(parents=True)
    (root / e["report"]).parent.mkdir(parents=True)
    (root / e["report"]).write_text(json.dumps(report))
    e["report_sha256"] = hashlib.sha256((root / e["report"]).read_bytes()).hexdigest()
    for arm in e["checkpoints"]:
        path = root / e["checkpoint_dir"] / f"{arm}.pt"
        path.write_bytes(arm.encode())
        e["checkpoints"][arm] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = root / e["corpus"] / "manifest.json"
    manifest.write_text("{}")
    e["corpus_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(fm, "EVIDENCE", e)
    return e


def _run1_report(**changes):
    report = {
        "revision": fm.EVIDENCE["revision"],
        "outcome": "M1-PASS",
        "decision": {"carried": "P-3"},
        "test_split_decoded": False,
        "tracked_tree_dirty": False,
        "smoke": False,
    }
    return report | changes


def test_check_evidence_accepts_the_pinned_files(monkeypatch, tmp_path):
    runner = load_runner()
    _fake_evidence(tmp_path, monkeypatch, _run1_report())
    found = runner.check_evidence(tmp_path)
    assert set(found["sha256"]) == {"report", "P-3", "C-3", "R-3", "corpus_manifest"}


@pytest.mark.parametrize("tamper", ["P-3", "corpus", "report"])
def test_check_evidence_refuses_a_changed_file(monkeypatch, tmp_path, tamper):
    runner = load_runner()
    e = _fake_evidence(tmp_path, monkeypatch, _run1_report())
    path = {
        "P-3": tmp_path / e["checkpoint_dir"] / "P-3.pt",
        "corpus": tmp_path / e["corpus"] / "manifest.json",
        "report": tmp_path / e["report"],
    }[tamper]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(fp2.GuardError, match="G-evidence"):
        runner.check_evidence(tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        {"outcome": "M1-PERCEPTION"},
        {"smoke": True},
        {"test_split_decoded": True},
        {"tracked_tree_dirty": True},
        {"revision": "0" * 40},
        {"decision": {"carried": "P-2"}},
    ],
)
def test_check_evidence_refuses_a_report_that_is_not_run1(monkeypatch, tmp_path, change):
    runner = load_runner()
    _fake_evidence(tmp_path, monkeypatch, _run1_report(**change))
    with pytest.raises(fp2.GuardError, match="G-evidence"):
        runner.check_evidence(tmp_path)
