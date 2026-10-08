"""TASK-083's Stage-0 code (protocol ``docs/experiments/apple_lewm_seed_replication_v2.md`` §6.1
step 1): seeds and salts, the carried pins, the per-seed artifacts (G-seed), the per-seed ladder
(TASK-081's, equal to its ``decide_s`` at the same salt) and the combined row in both directions,
the runner's carried code, the pool sequence, the sentinel and the power pieces."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_jepa import lewm_c1m_v2 as lm
from embodied_jepa import lewm_cp_v2 as cpv
from embodied_jepa import lewm_pr_v2 as pr
from embodied_jepa import lewm_rep_v2 as rep
from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa import plate_twin_v2_harness as hz
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_lewm_rep_v2.py"
TASK081_RUNNER = ROOT / "scripts" / "run_lewm_cp_v2.py"
MANIFEST = ROOT / rep.MANIFEST


def _load(path: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner():
    return _load(RUNNER, "_run_lewm_rep_v2")


# ----- the frozen block, the manifest, the carried pins ------------------------------------------
def test_manifest_frozen_block_equals_the_module():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["frozen"] == json.loads(json.dumps(rep.frozen_block(), ensure_ascii=False))
    assert manifest["frozen_sha256"] == rep.frozen_sha256()
    assert manifest["status"] == rep.STATUS


def test_a_draft_carries_no_pin():
    manifest = json.loads(MANIFEST.read_text())
    if rep.STATUS == "DRAFT":
        assert manifest["frozen_sha256_pin"] is None and manifest["hashes"] == {}
        assert _runner().check_frozen_pin(manifest)["pin"] is None


def test_task081_task080_and_task077_are_carried_unchanged():
    m081 = json.loads((ROOT / rep.TASK081_MANIFEST).read_text())
    assert cpv.frozen_sha256() == rep.TASK081_FROZEN_SHA256 == m081["frozen_sha256_pin"]
    assert len(hz.check_pins(m081["hashes"])) == 6
    assert hz.sha256_file(ROOT / cpv.DOCUMENT) == m081["protocol_document_sha256"]
    assert pr.frozen_sha256() == rep.TASK080_FROZEN_SHA256
    assert lm.frozen_sha256() == rep.TASK077_FROZEN_SHA256
    assert rep.TAU_COMMIT_CM == 1.0 and rep.TAU_CURVE == cpv.TAU_CURVE
    assert rep.SOLVER_OF == {"W": "affine_local", "N": "affine_local", "L-shuf": "affine_local",
                             "L-mean": "affine_local", "L-rand": "none"}  # fmt: skip
    assert rep.READOUTS == {"W": "r_s", "L-shuf": "r_s", "L-mean": "r_s", "N": "r_n"}


def test_the_reused_models_and_readouts():
    assert rep.MODEL_SEEDS == (66801, 66802) and rep.REFERENCE_SEED == 66800
    assert rep.POOL_ORDER == (66801, 66802, 66800)
    for s in rep.POOL_ORDER:
        for arm in ("W", "N"):
            assert rep.REUSED["models"][f"{arm}-{s}"] == pr.REUSED["models"][f"{arm}-{s}"]
    assert rep.REUSED["readouts"][66800]["r_s"] == cpv.REUSED["r_s"]
    assert rep.REUSED["readouts"][66800]["r_n"] == cpv.REUSED["r_n"]
    assert rep.REUSED["stage_r_report_sha256"] == cpv.REUSED["stage_r_report_sha256"]
    shas = [v for d in rep.REUSED["readouts"].values() for v in d.values()]
    assert len(set(shas)) == len(shas) == 12  # every seed's readout is its own
    assert rep.SEED_FLAGS[66800]["last_two_triggered"]
    assert not rep.SEED_FLAGS[66801]["last_two_triggered"]
    assert not rep.SEED_FLAGS[66802]["last_two_triggered"]


def test_seed_ranges_and_salts():
    rep.check_seed_ranges()
    assert rep.seeds_of("S") == tuple(range(75200, 75328)) and len(rep.seeds_of("S")) == 128
    assert rep.seeds_of("S", debug=True) == (75910, 75911, 75912, 75913)
    for s in (*rep.seeds_of("S"), *rep.seeds_of("S", debug=True)):
        assert 75000 <= s <= 75999
        for low, high in rep.FORBIDDEN_RANGES.values():
            assert not low <= s <= high
    assert rep.FORBIDDEN_RANGES["task081_block"] == (70000, 71999)
    assert rep.FORBIDDEN_RANGES["task082_block"] == (72000, 74999)
    assert rep.FORBIDDEN_RANGES["task080_block"] == (65000, 65999)
    assert rep.SALTS == {"bootstrap": 8501, "power": 8502}
    assert set(rep.RESERVED_SALTS.values()) == set(range(8503, 8513))
    assert rep.CARRIED_SALTS == {"move": 8201, "l_rand": 8303}
    assert set(range(8401, 8413)) <= set(rep.EARLIER_SALTS.values())
    assert set(range(8301, 8313)) <= set(rep.EARLIER_SALTS.values())
    with pytest.raises(pt.GuardError):
        rep.check_seeds("S", rep.seeds_of("S")[:-1])
    with pytest.raises(pt.GuardError):
        rep.check_seeds("S", rep.seeds_of("S", debug=True))
    assert rep.check_seeds("S", rep.seeds_of("S", debug=True), debug=True)
    for bad in (66803, "66801", 66801.0, None):
        with pytest.raises(pt.GuardError):
            rep.check_model_seed(bad)


def test_the_move_draw_is_task080s_salt_8201_on_the_fresh_seeds():
    for s in (75200, 75327, 75910):
        assert rep.move_offset(s).tolist() == pr.move_offset(s).tolist()
        assert rep.reset_of(s) == pr.reset_of(s)


def test_bars_carried_at_128():
    assert (rep.G_BAR, rep.DELTA, rep.MIN_SEPARATION, rep.MCNEMAR_P) == (112, 16, 7, 0.01)
    assert rep.S_RESETS == cpv.S_RESETS == 128
    assert rep.TWINS == cpv.TWINS and rep.COMPARATORS == cpv.COMPARATORS
    assert rep.DETERMINISM_RESETS == 4
    assert not rep.CLAUSE_ROWS


# ----- the ladders --------------------------------------------------------------------------------
def _outcomes(n=128, **counts):
    base = {"W": 120, "N": 50, "L-shuf": 40, "L-mean": 55, "L-rand": 20, "H-rule": 124,
            "H-sysid": 120, "H-final": 124}  # fmt: skip
    base |= counts
    out = {}
    for arm, k in base.items():  # nested: every arm's successes are the first k resets
        v = np.zeros(n, bool)
        v[:k] = True
        out[arm] = v
    return out


def test_the_seed_ladder_in_both_directions():
    d = rep.decide_seed
    assert d(_outcomes())["row"] == "L-PASS"
    assert d(_outcomes(**{"H-final": 111}))["row"] == "S-VOID-CEILING"
    assert d(_outcomes(**{"H-final": 112}))["row"] == "L-PASS"
    out = d(_outcomes(**{"L-mean": 118}))
    assert out["row"] == "L-NO-GAIN" and out["detectable_failure"] and not out["clause_fires"]
    assert d(_outcomes(W=111))["row"] == "L-NEAR"
    out = d(_outcomes(W=111, **{"H-rule": 112, "H-sysid": 110}))
    assert out["row"] == "L-BAR" and not out["G-bar"]
    near = d(_outcomes(W=112, **{"H-rule": 128, "H-final": 128}))
    assert near["row"] == "L-NEAR" and not near["G-NI"]["detectably_inferior"]
    inf = d(_outcomes(W=100, **{"H-rule": 128, "H-final": 128}))
    assert inf["row"] == "L-INFERIOR" and inf["detectable_failure"] and not inf["clause_fires"]
    assert d(_outcomes(**{"N": 113}))["row"] == "L-PASS"  # b = 7, c = 0
    tn = d(_outcomes(**{"N": 115}))
    assert tn["row"] == "L-TWIN-NEAR" and tn["twins"]["N"]["miss_within_noise"]
    up = d(_outcomes(W=118, **{"H-rule": 125}))
    assert up["row"] == "L-PASS" and up["G-NI"]["upper_below_zero"]
    assert d(None, void=True)["row"] == "V"
    with pytest.raises(ContractError):
        d(_outcomes(n=64))
    with pytest.raises(ContractError):
        d({k: v for k, v in _outcomes().items() if k != "L-rand"})


def test_the_seed_ladder_is_task081s_decide_s_at_the_same_salt(monkeypatch):
    monkeypatch.setattr(rep, "paired_interval", cpv.paired_interval)
    rng = np.random.default_rng(83)
    arms = ("W", "N", "L-shuf", "L-mean", "L-rand", "H-rule", "H-sysid", "H-final")
    for _ in range(40):
        rates = rng.uniform(0.3, 1.0, len(arms))
        rates[0] = rng.uniform(0.8, 1.0)
        out = {a: rng.uniform(size=128) < r for a, r in zip(arms, rates, strict=True)}
        mine, theirs = rep.decide_seed(out), cpv.decide_s(out)
        assert mine["row"] == theirs["row"]
        assert mine["G-NI"]["ci95"] == theirs["G-NI"]["ci95"]
        assert mine["twins"] == theirs["twins"]


def test_the_bootstrap_salt_is_8501():
    a, b = _outcomes()["W"], _outcomes()["H-rule"]
    assert rep.paired_interval(a, b) == pr.paired_interval(a, b, salt=8501)
    assert rep.median_ci([1.0, 2.0, 3.0]) == pr.median_ci([1.0, 2.0, 3.0], salt=8501)


def test_the_combined_row_in_both_directions():
    c = rep.decide_combined
    assert c({66801: "L-PASS", 66802: "L-PASS"})["row"] == "REP-PASS"
    one = c({66801: "L-PASS", 66802: "L-NEAR"})
    assert one["row"] == "REP-ONE" and one["passed"] == [66801]
    assert not one["detectably_not_replicated"]
    one = c({66801: "L-INFERIOR", 66802: "L-PASS"})
    assert one["row"] == "REP-ONE" and one["detectably_not_replicated"] == [66801]
    none = c({66801: "L-BAR", 66802: "L-NO-GAIN"})
    assert none["row"] == "REP-NONE" and none["detectably_not_replicated"] == [66802]
    assert c({66801: "S-VOID-CEILING", 66802: "S-VOID-CEILING"})["row"] == "REP-VOID-CEILING"
    assert c(None, void=True)["row"] == "V"
    for bad in ({66801: "S-VOID-CEILING", 66802: "L-PASS"}, {66801: "V", 66802: "L-PASS"},
                {66801: "L-PASS"}, {66801: "L-PASS", 66800: "L-PASS"}):  # fmt: skip
        with pytest.raises(ContractError):
            c(bad)
    for row in ("REP-PASS", "REP-ONE", "REP-NONE", "REP-VOID-CEILING"):
        assert row in rep.COMBINED_ROWS and row in rep.R7_BY_ROW
    assert "power" in rep.COMBINED_ROWS["REP-ONE"] and "chance" in rep.COMBINED_ROWS["REP-ONE"]


def test_no_pooling_one_seeds_margin_does_not_rescue_the_other():
    """A seed one reset short of G-bar stays short whatever the other seed scores."""
    strong = rep.decide_seed(_outcomes(W=128, **{"H-rule": 124}))
    weak = rep.decide_seed(_outcomes(W=111, **{"H-rule": 112, "H-sysid": 110}))
    assert strong["row"] == "L-PASS" and weak["row"] == "L-BAR"
    assert rep.decide_combined({66801: strong["row"], 66802: weak["row"]})["row"] == "REP-ONE"


def test_seed_outcomes_and_labels():
    out = {rep.label(a, s): np.ones(4, bool) * (s == 66801)
           for s in rep.MODEL_SEEDS for a in rep.PER_SEED_ARMS}  # fmt: skip
    out |= {a: np.zeros(4, bool) for a in rep.SHARED_ARMS}
    mine = rep.seed_outcomes(out, 66801)
    assert set(mine) == {*rep.PER_SEED_ARMS, *rep.SHARED_ARMS} and mine["W"].all()
    assert not rep.seed_outcomes(out, 66802)["W"].any()
    assert rep.label("W", 66801) == "W[66801]" and rep.label("H-rule", None) == "H-rule"


def test_between_seeds_and_two_sided_mcnemar():
    assert rep.mcnemar_two_sided(0, 0) == 1.0
    assert rep.mcnemar_two_sided(7, 0) == pytest.approx(2 * 0.5**7)
    assert rep.mcnemar_two_sided(3, 3) == 1.0
    a, b = _outcomes(W=120)["W"], _outcomes(W=110)["W"]
    out = rep.between_seeds(a, b)
    assert (out["b"], out["c"], out["difference"]) == (10, 0, 10)


def test_exact_intervals_and_pairs():
    out = rep.counts_and_pairs(rep.seed_outcomes(
        {rep.label(a, 66801): v for a, v in _outcomes().items() if a in rep.PER_SEED_ARMS}
        | {a: v for a, v in _outcomes().items() if a in rep.SHARED_ARMS}, 66801), "W")  # fmt: skip
    assert out["counts"]["W"]["count"] == 120 and "W" not in out["minus_arm"]
    assert out["minus_arm"]["H-rule"]["difference"] == -4
    assert out["minus_arm"]["H-rule"]["c"] == 4
    lo, hi = rep.exact_interval(118, 128)
    assert (round(lo, 3), round(hi, 3)) == (0.861, 0.962)  # TASK-081's W on S


# ----- power --------------------------------------------------------------------------------------
def test_power_pieces():
    p = rep.rep_power(0.953, 0.984, "overlap", trials=400)
    assert 0.85 < min(p["per_seed_pass"]) <= 1.0 and p["both_pass"] <= min(p["per_seed_pass"])
    assert p["both_pass"] + p["exactly_one"] + p["neither"] == pytest.approx(1.0)
    low = rep.rep_power(0.906, 0.992, "independent", trials=400)
    assert low["both_pass"] < 0.3
    size = rep.rep_power(0.984 - 16 / 128, 0.984, "overlap", trials=400)
    assert max(size["per_seed_pass"]) < 0.1
    assert rep._ni_bounds(0, 0) == (0.0, 0.0)
    assert rep.PLANNING["C"] == (0.979, 0.984, 0.992)


# ----- the runner ---------------------------------------------------------------------------------
CARRIED = ("Pool", "check_task077", "summary_line", "summary_ok", "run_full_tests", "wait_quiet",
           "check_disk", "sim_preflight", "completed", "old_chain", "worker_config",
           "streamed_estimates", "cohort_estimates", "attempt_tasks", "run_arm_carried",
           "successes", "lean", "arm_summary", "open_log", "git", "check_task080",
           "check_task080_document", "run_arm")  # fmt: skip


def _definitions(path: Path) -> dict:
    text = path.read_text()
    lines = text.splitlines()
    tree = ast.parse(text)
    return {n.name: "\n".join(lines[n.lineno - 1 : n.end_lineno]) for n in tree.body
            if isinstance(n, ast.FunctionDef | ast.ClassDef)}  # fmt: skip


def test_the_runners_carried_code_is_task081s_verbatim():
    mine, theirs = _definitions(RUNNER), _definitions(TASK081_RUNNER)
    for name in CARRIED:
        assert mine[name] == theirs[name], name


def test_the_runner_uses_run_tools_guards_and_loads_no_other_script():
    calls = set()
    for node in ast.walk(ast.parse(RUNNER.read_text())):
        if isinstance(node, ast.Call):
            f = node.func
            calls.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    for name in ("assert_local_import", "install_guards", "MemoryWatch", "check_pins",
                 "check_evidence", "verify_feature_files", "check_seed_config"):  # fmt: skip
        assert name in calls, name
    nri = _load(ROOT / "tests" / "test_no_runner_imports.py", "_nri")
    loaded, loaders = nri.script_loads(RUNNER)
    assert not loaded and not loaders


def test_the_runner_refuses_misused_flags(tmp_path):
    run = _runner()
    assert run.DRAFT_ALLOWED == ("simulate",)
    for argv in (
        ["closed", "--output", str(tmp_path / "a")],  # no cohort
        ["closed", "--cohort", "S", "--output", str(tmp_path / "b"), "--evidence", "e"],
        ["closed", "--cohort", "D", "--output", str(tmp_path / "c")],
        ["closed", "--cohort", "S", "--output", str(tmp_path / "d"), "--evidence", "e",
         "--old-features", "f", "--old-fits", "g", "--models", "m"],  # no --stage-r
        ["simulate", "--output", str(tmp_path / "e"), "--workers", "2"],
        ["simulate", "--output", str(tmp_path / "f"), "--cohort", "S"],
        ["simulate", "--output", str(tmp_path / "g"), "--debug-skip-tests"],
    ):  # fmt: skip
        with pytest.raises(SystemExit):
            run.main(argv)


def test_the_preflight_refuses_closed_while_draft(monkeypatch):
    run = _runner()
    monkeypatch.setattr(run.rt, "assert_local_import", lambda *a, **k: None)
    monkeypatch.setattr(run.hz, "check_pins", lambda pins, *a: dict(pins))
    for name in ("check_task077", "check_task080", "check_task081"):
        monkeypatch.setattr(run, name, lambda report: None)
    monkeypatch.setattr(run.hz, "tracked_tree_dirty", lambda *a: [])
    monkeypatch.setattr(run.rep, "STATUS", "DRAFT")
    monkeypatch.setattr(run, "check_frozen_pin", lambda manifest: {"status": "DRAFT"})
    args = SimpleNamespace(stage="closed", debug=False, debug_skip_tests=False)
    with pytest.raises(pt.GuardError, match="runs only after the freeze"):
        run.preflight({}, args)


def test_the_stage_r_report_is_pinned(tmp_path):
    run = _runner()
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"outcome": "R-PASS", "debug": False}))
    with pytest.raises(pt.GuardError, match="not TASK-080's"):
        run.stage_r_seed_readouts(SimpleNamespace(stage_r=str(path)), {})


def test_the_stage_r_readouts_are_each_seeds_own(tmp_path, monkeypatch):
    run = _runner()
    seeds = {str(s): {"readouts": {n: {"sha256": rep.REUSED["readouts"][s][n],
                                       "path": f"outputs/x/fits/{n}_{s}.npz"}
                                   for n in ("r_s", "r_n")}}
             for s in rep.POOL_ORDER}  # fmt: skip
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"outcome": "R-PASS", "debug": False,
                                       "fields": {"fits": {"seeds": seeds}}}))  # fmt: skip
    files = {f"{n}_{s}.npz": rep.REUSED["readouts"][s][f"{n}_file"]
             for s in rep.POOL_ORDER for n in ("r_s", "r_n")}  # fmt: skip

    def sha(path):
        path = Path(path)
        if path == report_path:
            return rep.REUSED["stage_r_report_sha256"]
        return files[path.name]

    loaded = []
    monkeypatch.setattr(run.hz, "sha256_file", sha)
    monkeypatch.setattr(run.pof, "load_readout", lambda p, s: loaded.append((Path(p).name, s)))
    report: dict = {}
    out = run.stage_r_seed_readouts(SimpleNamespace(stage_r=str(report_path)), report)
    assert sorted(out) == sorted(rep.POOL_ORDER) and len(loaded) == 6
    assert out[66801]["r_s"]["sha256"] == rep.REUSED["readouts"][66801]["r_s"]
    assert out[66802]["r_n"]["path"].endswith("r_n_66802.npz")
    # a swapped file is refused (G-seed)
    files["r_s_66802.npz"] = rep.REUSED["readouts"][66801]["r_s_file"]
    with pytest.raises(pt.GuardError, match="G-seed"):
        run.stage_r_seed_readouts(SimpleNamespace(stage_r=str(report_path)), {})
    files["r_s_66802.npz"] = rep.REUSED["readouts"][66802]["r_s_file"]
    seeds["66801"]["readouts"]["r_n"]["sha256"] = rep.REUSED["readouts"][66802]["r_n"]
    report_path.write_text(json.dumps({"outcome": "R-PASS", "debug": False,
                                       "fields": {"fits": {"seeds": seeds}}}))  # fmt: skip
    with pytest.raises(pt.GuardError, match="G-seed"):
        run.stage_r_seed_readouts(SimpleNamespace(stage_r=str(report_path)), {})


def _chain():
    jobs = {
        f"{a}-{s}": {"checkpoint": f"/ck/{a}-{s}.pt", "checkpoint_sha256": pr.REUSED["models"][
            f"{a}-{s}"], "metadata": {"seed": s}}
        for a in ("W", "N") for s in rep.POOL_ORDER
    }  # fmt: skip
    paths = {"r_plate": "/f/r_plate.npz", "r8": "/f/r8.npz", "mean_latent": "/f/m.npy"}
    return {"jobs": jobs, "paths": paths}


def _readouts(s):
    return {n: {"path": f"/fits/{n}_{s}.npz", "sha256": rep.REUSED["readouts"][s][n]}
            for n in ("r_s", "r_n")}  # fmt: skip


def test_seed_worker_config_is_task081s_for_66800_and_each_seeds_own():
    run = _runner()
    chain = _chain()
    assert run.seed_worker_config(chain, 66800, _readouts(66800)) == run.worker_config(
        chain, readouts=_readouts(66800), with_r8=True
    )
    for s in rep.MODEL_SEEDS:
        config = run.seed_worker_config(chain, s, _readouts(s))
        assert {a: config["models"][a]["seed"] for a in ("W", "N")} == {"W": s, "N": s}
        assert config["models"]["W"]["sha256"] == pr.REUSED["models"][f"W-{s}"]
        assert config["r_s_sha256"] == rep.REUSED["readouts"][s]["r_s"]
    with pytest.raises(pt.GuardError, match="G-seed"):
        run.seed_worker_config(chain, 66801, _readouts(66802))
    with pytest.raises(pt.GuardError, match="G-seed"):
        run.seed_worker_config(chain, 66803, _readouts(66801))
    config = run.seed_worker_config(chain, 66801, _readouts(66801))
    config["models"]["N"] = dict(config["models"]["N"], sha256=pr.REUSED["models"]["N-66802"])
    with pytest.raises(pt.GuardError, match="G-seed"):
        run.check_seed_config(config, 66801)


def test_check_seed_records_refuses_another_readout():
    run = _runner()
    ok = [{"seed": 1, "decisions": [{"readout": "r_s"}]}]
    run.check_seed_records(ok, "W")
    run.check_seed_records(ok, "H-rule")
    with pytest.raises(pt.GuardError, match="G-seed"):
        run.check_seed_records(ok, "N")
    with pytest.raises(pt.GuardError, match="G-seed"):
        run.check_seed_records([{"seed": 1, "decisions": [{"readout": "r8"}]}], "L-mean")


class _FakeCohort:
    def __init__(self, seeds):
        self.resets = {s: lm.reset_of(s) for s in seeds}
        self.est = {
            s: {"estimates": [0.3, -0.2, 0.5, -0.1], "frame_sha256": "f", "state_sha256": "s"}
            for s in seeds
        }


class _FakePool:
    """Every attempt succeeds for W, the comparators and the ceiling; the twins fail. The frame at
    405 encodes the reset seed and the pool's model seed."""

    def __init__(self, model_seed, refused, log):
        self.model_seed, self.refused, self.log = model_seed, set(refused), log
        self.look_reference = None

    def map(self, tasks, cap, what):
        self.log.append((self.model_seed, tasks[0]["arm"], len(tasks), tasks))
        out = []
        for task in tasks:
            s, arm = task["seed"], task["arm"]
            solver = rep.SOLVER_OF.get(arm)
            base = {"seed": s, "arm": arm, "blocked": None, "privileged_ok": True,
                    "task_truth_in_controller": 0, "seconds": 1.0, "solver": solver,
                    "termination_reason": "step_limit", "executed_steps": 800,
                    "final_distance_cm": 3.6}  # fmt: skip
            if s in self.refused:
                out.append(base | {"success": False, "termination_reason": "guard_refusal",
                                   "executed_steps": 230, "frame405": None,
                                   "decisions": []})  # fmt: skip
                continue
            good = arm in ("W", "H-final", "H-rule", "H-sysid")
            wm = {"solver": solver}
            if solver == "affine_local":
                wm |= {"points": 25, "residual_cm": 0.3, "eigenvalues_real": [-0.5, -0.4],
                       "fallback_grid_argmin": False, "clipped": False}  # fmt: skip
            frame = np.full((2, 2, 3), s % 251, np.uint8)
            frame[0, 0, 1] = self.model_seed % 251
            out.append(base | {
                "success": good, "frame405": frame,
                "decisions": [{"reading": [0.49, -0.09], "world_model": wm,
                               "readout": rep.READOUTS.get(arm)}],
                "commit": {"target": [0.5, -0.1], "fallback": False, "clipped": False,
                           "fixed_point_error_cm": 0.4, "landing_miss_cm": 0.3},
                "post_look_state": [0.0],
            })  # fmt: skip
        return out


class _FakePools:
    def __init__(self, seeds, refused=(), fail_on=None):
        self.co = _FakeCohort(seeds)
        self.log, self.events = [], []
        self.refused, self.fail_on = refused, fail_on

    def open(self, model_seed):
        self.events.append(("open", model_seed))
        pool = _FakePool(model_seed, self.refused, self.log)
        if self.fail_on == model_seed:
            pool.map = lambda *a, **k: (_ for _ in ()).throw(pt.GuardError("boom"))
        return pool

    def close(self, model_seed, pool):
        self.events.append(("close", model_seed))


def test_closed_core_runs_every_arm_in_its_pool_and_decides():
    run = _runner()
    seeds = rep.seeds_of("S")
    pools = _FakePools(seeds, refused={seeds[1], seeds[2]})
    report: dict = {}
    fields = run.Fields(report, "closed")
    row = run.closed_core(pools, seeds, {"tau_commit_cm": 1.0, "a_lo": cpv.A_LO}, [[0.0]],
                          fields, debug=False)  # fmt: skip
    assert pools.events == [("open", 66801), ("close", 66801), ("open", 66802),
                            ("close", 66802), ("open", 66800), ("close", 66800)]  # fmt: skip
    ran = [(m, arm, n) for m, arm, n, _t in pools.log]
    shared = [(m, a) for m, a, _n in ran if a in rep.SHARED_ARMS]
    assert shared == [(66801, a) for a in rep.SHARED_ARMS]  # once, in the first pool
    for s in rep.MODEL_SEEDS:
        arms = [a for m, a, _n in ran if m == s and a not in rep.SHARED_ARMS]
        assert arms == ["W", "N", "L-shuf", "L-mean", "W"]  # the last: the determinism re-run
        shuf = next(t for m, a, _n, t in pools.log if m == s and a == "L-shuf")
        assert int(shuf[0]["foreign_frame"][0, 0, 0]) == seeds[3] % 251  # the next reached
        assert int(shuf[0]["foreign_frame"][0, 0, 1]) == s % 251  # this seed's own W
    assert [a for m, a, _n in ran if m == 66800] == ["W"]
    arms = report["fields"]["arms"]
    want = {rep.label(a, s) for s in rep.MODEL_SEEDS for a in rep.PER_SEED_ARMS}
    want |= set(rep.SHARED_ARMS) | {"W[66800]", "refused_before_405"}
    assert set(arms) == want and arms["W[66801]"]["count"] == 126
    assert set(report["fields"]["determinism"]) == {"66801", "66802"}
    decision = report["fields"]["decision"]
    assert row == decision["row"] == "REP-PASS"
    rows = {k: v["row"] for k, v in decision["per_seed"].items()}
    assert rows == {"66801": "L-PASS", "66802": "L-PASS"}
    rep_ = report["fields"]["reported"]
    assert rep_["reference_w66800"]["reported_only"] and "between_seeds" in rep_
    assert rep_["per_arm"]["W[66801]"]["affine_local"]["decisions"] == 126
    assert rep_["per_seed"]["66802"]["flags"] == rep.SEED_FLAGS[66802]


def test_closed_core_closes_a_pool_that_fails():
    run = _runner()
    seeds = rep.seeds_of("S")
    pools = _FakePools(seeds, fail_on=66802)
    fields = run.Fields({}, "closed")
    with pytest.raises(pt.GuardError):
        run.closed_core(pools, seeds, {"tau_commit_cm": 1.0, "a_lo": cpv.A_LO}, [[0.0]], fields,
                        debug=False)  # fmt: skip
    assert pools.events[-2:] == [("open", 66802), ("close", 66802)]


def test_the_sentinel():
    run = _runner()
    report: dict = {}
    fields = run.Fields(report, "closed")
    fields.check()
    report["fields"]["decision"] = {"row": "REP-PASS"}
    with pytest.raises(pt.GuardError):
        fields.check()


def test_stage_modules_import_without_torch_or_mujoco():
    code = (
        "import sys\n"
        "import embodied_jepa.lewm_rep_v2, embodied_jepa.lewm_cp_v2_runtime\n"
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
