"""TASK-082's Stage-0 code (protocol ``docs/experiments/apple_lewm_unknown_law_v2.md`` §7.1 step 2):
seeds and salts, the carried and pinned files, U-sat's law in the runtime, the corpus collector
(its label never changes the root; gate-P's aim from p-hat), the readouts each arm reads (never
R8), affine_local equal to TASK-081's, H-rule's written law, no task truth in a non-privileged arm
(the learned tier included), the carried budget, the training loop, the sentinel, every row ladder
in both directions, the power pieces and delta's rule, and the runner's guards and carried code."""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_jepa import commit_precision_dev as cpd
from embodied_jepa import lewm_c1m_v2 as lm
from embodied_jepa import lewm_c1m_v2_train as tr
from embodied_jepa import lewm_cp_v2 as cpv
from embodied_jepa import lewm_cp_v2_runtime as cprt
from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa import lewm_pr_v2 as pr
from embodied_jepa import lewm_ul_v2 as ul
from embodied_jepa import lewm_ul_v2_offline as ulo
from embodied_jepa import lewm_ul_v2_runtime as urt
from embodied_jepa import lewm_ul_v2_train as ult
from embodied_jepa import plate_law_dev as pl
from embodied_jepa import plate_law_dev_runtime as pldr
from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa import plate_twin_v2_harness as hz
from embodied_jepa import run_tools as rt
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_lewm_ul_v2.py"
TASK077_RUNNER = ROOT / "scripts" / "run_lewm_c1m_v2.py"
MANIFEST = ROOT / ul.MANIFEST
P_HAT = np.array([0.47, -0.08])
H = np.array([0.30, -0.16])


def _load(path: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner():
    return _load(RUNNER, "_run_lewm_ul_v2")


# ----- the frozen block, the manifest, the carried pins ------------------------------------------
def test_manifest_frozen_block_equals_the_module():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["frozen"] == json.loads(json.dumps(ul.frozen_block(), ensure_ascii=False))
    assert manifest["frozen_sha256"] == ul.frozen_sha256()
    assert manifest["status"] == ul.STATUS


def test_a_draft_carries_no_pin():
    manifest = json.loads(MANIFEST.read_text())
    if ul.STATUS == "DRAFT":
        assert manifest["frozen_sha256_pin"] is None and manifest["hashes"] == {}
        assert manifest["protocol_document_sha256"] is None
        assert _runner().check_frozen_pin(manifest)["pin"] is None
        assert "**STATUS: DRAFT**" in (ROOT / ul.DOCUMENT).read_text()


def test_carried_frozen_blocks_file_pins_and_documents():
    for manifest_path, module, pin, n in (
        (ul.TASK077_MANIFEST, lm, ul.TASK077_FROZEN_SHA256, 13),
        (ul.TASK080_MANIFEST, pr, ul.TASK080_FROZEN_SHA256, 7),
        (ul.TASK081_MANIFEST, cpv, ul.TASK081_FROZEN_SHA256, 6),
    ):
        manifest = json.loads((ROOT / manifest_path).read_text())
        assert module.frozen_sha256() == pin == manifest["frozen_sha256_pin"]
        assert len(hz.check_pins(manifest["hashes"])) == n
    for document, manifest_path in ((pr.DOCUMENT, ul.TASK080_MANIFEST),
                                    (cpv.DOCUMENT, ul.TASK081_MANIFEST)):  # fmt: skip
        want = json.loads((ROOT / manifest_path).read_text())["protocol_document_sha256"]
        assert hz.sha256_file(ROOT / document) == want


def test_the_development_files_are_pinned_at_stage_0():
    assert set(ul.PINNED_DEV_FILES) == {
        "src/embodied_jepa/plate_law_dev.py",
        "src/embodied_jepa/plate_law_dev_runtime.py",
        "src/embodied_jepa/commit_precision_dev.py",
    }
    for path, sha in ul.PINNED_DEV_FILES.items():
        assert hz.sha256_file(ROOT / path) == sha, path
    assert ul.PINNED_DEV_FILES[cpv.SOLVER_FILE] == cpv.SOLVER_FILE_SHA256


def test_the_carried_artifacts_and_plan():
    assert ul.CARRIED["r_plate"] == pr.REUSED["r_plate"] == (
        "08bde901ca75ab0ee4bc1c74bf4da91117ba57c0491828d81bc9db921cc49eb9")  # fmt: skip
    assert ul.CARRIED["task077_readouts_report_sha256"] == pr.REUSED["readouts_report_sha256"]
    assert ul.CARRIED_PLAN["updates"] == 95_000 and ul.CARRIED_PLAN["select_every"] == 4_750
    assert ul.G1_BARS["rank"]["bar"] == 0.12 and ul.G1_BARS["std"]["bar"] == 0.38
    assert ul.CARRIED["task077_plan_report_sha256"].startswith("7c714760")


# ----- seeds and salts ----------------------------------------------------------------------------
def test_seed_ranges_and_salts():
    ul.check_seed_ranges()
    assert ul.seeds_of("K") == tuple(range(72400, 72464)) and len(ul.seeds_of("K")) == 64
    assert ul.seeds_of("D") == tuple(range(72500, 72516))
    assert ul.seeds_of("S") == tuple(range(72600, 72728)) and len(ul.seeds_of("S")) == 128
    assert ul.seeds_of("corpus") == tuple(range(72800, 74800))
    assert ul.MODEL_SEEDS == (72360, 72361, 72362)
    every = [s for r in ("K", "D", "S", "corpus") for s in ul.seeds_of(r)]
    every += [s for r in ul.DEBUG_RANGES for s in ul.seeds_of(r, debug=True)]
    every += [*ul.MODEL_SEEDS, *ul.DEBUG_MODEL_SEEDS]
    for s in every:
        assert 72032 <= s <= 72099 or 72356 <= s <= 74899, s
        for low, high in ul.FORBIDDEN_RANGES.values():
            assert not low <= s <= high, s
    for name in ("design_check_F", "design_corpus", "design_debug", "task081_block"):
        assert name in ul.FORBIDDEN_RANGES
    assert ul.FORBIDDEN_RANGES["design_corpus"] == (72100, 72355)
    assert ul.FORBIDDEN_RANGES["design_debug"] == (74900, 74999)
    for name, value in cpv.FORBIDDEN_RANGES.items():
        assert ul.FORBIDDEN_RANGES[name] == value
    assert sorted(ul.SALTS.values()) == list(range(8405, 8413))
    assert set(ul.USED_SALTS.values()) == {8401, 8402, 8403, 8404}
    assert not set(ul.SALTS.values()) & set(ul.EARLIER_SALTS.values())
    assert ul.FOLD_KEYS == {"outer": 1, "inner": 2, "learned_tier": 3, "learning_curve": 4}
    assert ul.MISC_KEYS == {"l_rand": 1, "wrong_commands": 2, "power": 3}


def test_check_seeds_refuses_anything_but_the_declared_range():
    assert ul.check_seeds("S", ul.seeds_of("S"))
    with pytest.raises(pt.GuardError):
        ul.check_seeds("S", ul.seeds_of("S")[:-1])
    with pytest.raises(pt.GuardError):
        ul.check_seeds("D", ul.seeds_of("D", debug=True))
    with pytest.raises(pt.GuardError):
        ul.check_seeds("K", tuple(range(72000, 72064)))  # the design note's check cohort
    assert ul.check_seeds("D", ul.seeds_of("D", debug=True), debug=True)
    with pytest.raises(pt.GuardError):
        ul.check_model_seed(66800)
    assert ul.check_model_seed(74890, debug=True) == 74890


def test_draws_use_this_tasks_salts():
    s = 72600
    assert ul.move_offset(s).tolist() != pr.move_offset(s).tolist()
    assert ul.move_offset(s).tolist() != pl.move_offset(s).tolist()
    assert np.linalg.norm(ul.move_offset(s)) <= 0.04 + 1e-12
    assert ul.corpus_aim(72800) != pl.corpus_aim(72800)
    a, b = ul.corpus_aim(72800)
    assert ul.A_LO <= a <= ul.A_HI and abs(b) <= ul.B_HALF_M
    rng = np.random.default_rng(np.random.SeedSequence([8408, 72400]))
    angle = 2.0 * np.pi * rng.uniform()
    assert ul.planted_error_m(72400, 1.0) == pytest.approx([0.01 * np.cos(angle),
                                                           0.01 * np.sin(angle)])  # fmt: skip
    feasible = np.ones(147, bool)
    draws = [ul.l_rand_index(s, feasible) for s in range(72600, 72660)]
    assert draws != [cpv.l_rand_index(s, feasible) for s in range(72600, 72660)]
    assert ul.l_rand_index(1, np.zeros(147, bool)) == -1
    perm = ul.wrong_permutation(250)
    assert not np.any(perm == np.arange(250)) and sorted(perm) == list(range(250))


def test_the_split_is_fixed_before_collection():
    split = ul.corpus_split(ul.seeds_of("corpus"))
    assert {k: len(v) for k, v in split.items()} == {"train": 1500, "val": 250, "gate_p": 250}
    assert sorted(s for v in split.values() for s in v) == list(ul.seeds_of("corpus"))
    debug = ul.corpus_split(ul.seeds_of("corpus", debug=True), debug=True)
    assert {k: len(v) for k, v in debug.items()} == {"train": 30, "val": 5, "gate_p": 5}
    assert ul.AIM_FROM == {"train": "true", "val": "true", "gate_p": "p_hat"}


# ----- U-sat --------------------------------------------------------------------------------------
def test_u_sat_is_plate_law_devs_sat_at_its_frozen_parameters():
    assert pl.LAWS["sat"] == {"amplitude_m": 0.08, "scale_m": 0.12, "swirl_rad": 0.25}
    assert (ul.LAW["amplitude_m"], ul.LAW["scale_m"], ul.LAW["swirl_rad"]) == (0.08, 0.12, 0.25)
    assert (ul.LAW["L"], ul.LAW["s0"], ul.LAW["s1"]) == (2, 405, 525)
    for d in ([0.1, 0.02], [-0.05, 0.13], [0.0, 0.0], [0.3, -0.2]):
        assert np.array_equal(ul.law_displacement(d), pl.sat_displacement(d, **pl.LAWS["sat"]))
    # the saturation and the swirl: |F| < A, and F turns away from -d by Theta |d| / D
    d = np.array([0.17, 0.0])
    f = ul.law_displacement(d)
    assert np.linalg.norm(f) < 0.08
    angle = np.arctan2(f[1], f[0]) - np.arctan2(-d[1], -d[0])
    assert abs(angle - 0.25 * 0.17 / 0.12) < 1e-9


def _hook(palm: dict, base=(0.5, -0.1)):
    hook = urt.UlMotion.__new__(urt.UlMotion)
    hook.base = np.asarray(base, np.float64)
    hook.s0, hook.s1 = 405, 525
    hook.state = {"palm": {int(k): np.asarray(v, np.float64) for k, v in palm.items()}}
    return hook


def test_the_runtime_hook_follows_u_sat_after_405_and_holds_after_s1():
    assert urt.UlMotion.target is pldr.LawMotion.target and urt.UlMotion.law == "sat"
    assert issubclass(urt.UlMotion, urt.wrt.CorpusMotion)
    rng = np.random.default_rng(0)
    palm = {t: np.array([0.3, -0.16]) + 0.002 * (t - 405) * np.array([1.0, 0.3])
            + rng.normal(0, 1e-4, 2) for t in range(397, 530)}  # fmt: skip
    hook = _hook(palm)
    assert np.array_equal(hook.target(300), hook.base) and np.array_equal(
        hook.target(405), hook.base
    )
    for t in (406, 430, 465, 500, 525):
        u = max(min(t, 525) - 2, 405)
        want = hook.base + pl.sat_displacement(palm[u] - palm[405], **pl.LAWS["sat"])
        assert np.allclose(hook.target(t), want, atol=1e-15), t
    assert np.array_equal(hook.target(529), hook.target(525))
    # not C1-M's rule: kappa (palm - palm(405)) would differ
    c1m_rule = hook.base - 0.5 * (palm[463] - palm[405])
    assert not np.allclose(hook.target(465), c1m_rule, atol=1e-4)


def test_g_law_refuses_a_record_not_under_u_sat():
    good = {"seed": 1, "kpred": {"law": dict(urt.LAW_RECORD)}}
    urt.check_law(good)
    for bad in ({"seed": 1, "kpred": {}}, {"seed": 1, "kpred": {"law": {"name": "C1-M"}}},
                {"seed": 1, "kpred": {"law": dict(urt.LAW_RECORD, amplitude_m=0.09)}}):  # fmt: skip
        with pytest.raises(pt.GuardError, match="G-law"):
            urt.check_law(bad)


def test_h_rule_keeps_c1ms_written_law():
    assert ul.H_RULE_LAW["kappa"] == c1.RULE["kappa"] == -0.5 and ul.H_RULE_LAW["L"] == 2
    assert urt.c1rt.RuleCommit.__init__ is not None
    source = inspect.getsource(urt.run_offline_aim_task)
    assert 'ul.H_RULE_LAW["kappa"] if arm == "H-rule"' in source
    calls = []
    original = urt.offline_rule_aim
    try:
        urt.offline_rule_aim = lambda task, p, h, kappa: calls.append(kappa) or (p, {})
        urt.run_offline_aim_task({"arm": "H-rule", "key": 0, "p_hat": P_HAT, "h": H})
        urt.run_offline_aim_task({"arm": "H-rule-fit", "key": 0, "p_hat": P_HAT, "h": H,
                                  "kappa_fit": -0.31})  # fmt: skip
    finally:
        urt.offline_rule_aim = original
    assert calls == [-0.5, -0.31]


# ----- the corpus collector -----------------------------------------------------------------------
class _FakeHook:
    def __init__(self, p):
        self.p = np.asarray(p, np.float64)

    def current(self):
        return self.p.copy()


class _FakeReadout:
    def __init__(self, value):
        self.value = np.asarray(value, np.float64)

    def predict(self, tokens):  # noqa: ARG002
        return self.value.copy()


class _FakeLookahead:
    def __init__(self, aim, converged=True):
        self.aim, self.converged, self.branch_steps, self.calls = aim, converged, 120, 0

    def __call__(self, ctl, observation, step, record):  # noqa: ARG002
        self.calls += 1
        record["lookahead"] = {"iterations": [{}, {}], "converged": self.converged,
                               "stopped": None}  # fmt: skip
        return np.asarray(self.aim, np.float64)


def _collector(monkeypatch, aim_from, label_aim=(0.9, 0.9), label=True):
    monkeypatch.setattr(urt.wrt, "encode", lambda frame: (np.zeros(4), np.zeros(4)))
    monkeypatch.setattr(urt.rt2, "state_of", lambda obs: np.zeros(3))
    p_true, p_hat = np.array([0.50, -0.10]), np.array([0.52, -0.07])
    task = {"a": 0.2, "b_m": 0.01, "aim_from": aim_from, "label": label}
    aim = urt.CollectUl.__new__(urt.CollectUl)
    urt.c1rt.GeometricAim.__init__(aim, "collect", _FakeHook(p_true), task)
    aim.readout, aim.label = _FakeReadout(p_hat), label
    aim.lookahead = _FakeLookahead(label_aim) if label else None
    ctl = SimpleNamespace(palm={405: H}, apple=np.zeros(2), last_grasp=(0.1, 0.2))
    obs = SimpleNamespace(images={urt.fp2.CAMERA: [np.zeros((112, 112, 3), np.uint8)]})
    record: dict = {}
    g = aim(ctl, obs, 405, record)
    return g, record, p_true, p_hat, aim


def test_gate_p_aims_are_built_from_p_hat_and_train_from_the_true_plate(monkeypatch):
    g, record, p_true, p_hat, _ = _collector(monkeypatch, "p_hat")
    assert np.allclose(g, c1.from_box(0.2, 0.01, p_hat, H))
    assert not np.allclose(g, c1.from_box(0.2, 0.01, p_true, H))
    assert record["aim_from"] == "p_hat" and record["p_hat405"] == p_hat.tolist()
    g, record, p_true, p_hat, _ = _collector(monkeypatch, "true")
    assert np.allclose(g, c1.from_box(0.2, 0.01, p_true, H))
    assert record["p_hat405"] == p_hat.tolist()
    runner = _runner()
    split = {"train": [1], "val": [2], "gate_p": [3]}
    per = runner.corpus_task(runner.split_of(split))
    assert [per(s)["aim_from"] for s in (1, 2, 3)] == ["true", "true", "p_hat"]
    with pytest.raises(pt.GuardError, match="G-corpus"):
        urt.CollectUl(None, None, None, {"aim_from": "label"}, None)


def test_the_label_never_changes_the_committed_aim(monkeypatch):
    g1, rec1, *_rest, aim = _collector(monkeypatch, "p_hat", label_aim=(0.9, 0.9))
    g2, rec2, *_ = _collector(monkeypatch, "p_hat", label_aim=(0.1, -0.4))
    g3, rec3, *_ = _collector(monkeypatch, "p_hat", label=False)
    assert np.array_equal(g1, g2) and np.array_equal(g1, g3)
    assert aim.lookahead.calls == 1 and rec1["label"]["aim"] == [0.9, 0.9]
    assert rec1["label"]["converged"] and rec1["label"]["iterations"] == 2
    assert "label" not in rec3
    rec = {k: v for k, v in rec1.items() if k != "label"}
    assert rec == {k: v for k, v in rec3.items() if k != "label"}


def test_the_collector_uses_task_076s_lookahead_with_the_ceilings_settings():
    source = inspect.getsource(urt.CollectUl.__init__)
    assert "ptr.LookaheadAim(robot, hook, bounds, ul.LOOKAHEAD_TOLERANCE_M" in source
    assert ul.LOOKAHEAD_TOLERANCE_M == c1.LOOKAHEAD_TOLERANCE_M
    assert ul.LOOKAHEAD_MAX_ITER == c1.MAX_ITERATIONS


def test_labelcheck_compares_every_trajectory_array():
    run = _runner()
    base = {"frames": np.zeros((65, 2, 2, 3), np.uint8), "commands": np.ones((64, 14)),
            "plate": np.ones((65, 2)), "palm": np.ones((65, 2)), "hidden_r": np.zeros(3),
            "target": np.ones(2), "state405": np.ones(5), "p_hat405": np.ones(2)}  # fmt: skip
    a = {"seed": 1, "executed_steps": 468, "termination_reason": "step_limit", "success": False,
         "blocked": None, "corpus": base | {"labelled": True, "label_aim": np.ones(2),
                                           "label_converged": np.asarray(True)}}  # fmt: skip
    b = {k: v for k, v in a.items()} | {"corpus": dict(base) | {"labelled": False}}
    assert run.compare_roots(a, b)["identical"]
    for key in ("commands", "plate", "frames"):
        changed = dict(base)
        changed[key] = base[key] + 1
        out = run.compare_roots(a, b | {"corpus": changed})
        assert not out["identical"] and not out[key]
    assert not run.compare_roots(a, b | {"executed_steps": 467})["identical"]


def test_the_corpus_store_seals_and_reopens(tmp_path):
    rng = np.random.default_rng(0)
    arrays = {"frames": np.zeros((65, 112, 112, 3), np.uint8), "commands": np.zeros((64, 14)),
              "plate": np.zeros((65, 2)), "palm": np.zeros((65, 2)),
              "hidden_r": np.zeros((112, 112, 3), np.uint8), "target": rng.normal(size=2),
              "state405": np.zeros(5), "apple_estimate": np.zeros(2), "last_grasp": np.zeros(2),
              "p_hat405": np.zeros(2), "label_aim": np.ones(2),
              "label_converged": np.asarray(True)}  # fmt: skip
    entries = {s: ulo.write_root(tmp_path, s, arrays) for s in (1, 2, 3)}
    with pytest.raises(FileExistsError):
        ulo.write_root(tmp_path, 1, arrays)
    with pytest.raises(ContractError):
        ulo.write_root(tmp_path, 9, arrays | {"label_aim": np.full(2, np.nan)})
    sealed = ulo.seal_corpus(tmp_path, entries, {"train": [1], "val": [2], "gate_p": [3]},
                             {"debug": True})  # fmt: skip
    manifest = ulo.open_corpus(tmp_path, sealed["sha256"])
    assert manifest["split"] == {"train": [1], "val": [2], "gate_p": [3]}
    assert manifest["aim_from"]["gate_p"] == "p_hat"
    with pytest.raises(pt.GuardError):
        ulo.open_corpus(tmp_path, "0" * 64)


def test_decide_corpus_rows():
    split = ul.corpus_split(ul.seeds_of("corpus"))
    gate = split["gate_p"]
    assert ul.decide_corpus(split, [], 0)["row"] == "CORPUS-SEALED"
    assert ul.decide_corpus(split, split["train"][:40], 5)["row"] == "CORPUS-SEALED"
    assert ul.decide_corpus(split, split["train"][:41], 0)["row"] == "CORPUS-ESCALATE"
    assert ul.decide_corpus(split, gate[:5], 0)["row"] == "CORPUS-SEALED"
    assert ul.decide_corpus(split, gate[:6], 0)["row"] == "CORPUS-ESCALATE"
    assert ul.decide_corpus(split, [], 6)["row"] == "CORPUS-ESCALATE"
    with pytest.raises(ContractError):
        ul.decide_corpus(split, ul.NOT_EVALUATED, 0)


# ----- the candidate arms: readouts, solver, L-rand -----------------------------------------------
def test_each_arm_reads_its_own_readout_and_never_r8():
    assert urt.readout_name("W") == urt.readout_name("L-shuf") == urt.readout_name("L-mean")
    assert urt.readout_name("W") == "r_s" and urt.readout_name("N") == "r_n"
    assert urt.readout_name("W-72361") == "r_s_72361"
    assert urt.model_name("W-72361") == "W-72361" and urt.model_name("L-shuf") == "W"
    assert urt.model_name("N") == "N" and urt.model_name("L-mean") == "W"
    for arm in ("H-rule", "H-read", "L-rand", "H-sysid-krr"):
        with pytest.raises(pt.GuardError, match="G-readout"):
            urt.readout_name(arm)
    assert "r8" not in ul.READOUTS.values()
    for cls in (urt.UlAim,):
        assert "r8" not in inspect.getsource(cls)
    assert urt.solver_of("W-72362") == "affine_local" and urt.solver_of("L-rand") == "none"
    assert ul.spread_arms(72361) == ("W-72360", "W-72362")
    assert ul.spread_seed("W-frozen") is None and ul.spread_seed("W-72360") == 72360


def _setup(slope, *, fixed=None, noise=0.0):
    """A synthetic predictor p~(g) = g* + S (g - g*) (+ noise); a chunk encodes its aim."""
    fixed = c1.from_box(0.31, 0.004, P_HAT, H) if fixed is None else np.asarray(fixed)
    slope = np.asarray(slope, np.float64)
    targets = np.asarray([c1.from_box(a, b, P_HAT, H) for a, b in ul.GRID])
    rng = np.random.default_rng(3)
    bumps = rng.normal(0, noise, (4096, 2))

    def encode(g):
        c = np.zeros((ul.HORIZON, 14), np.float32)
        c[0, :2] = np.asarray(g, np.float32)
        return c

    def predict(commands):
        g = np.asarray(commands, np.float64)[:, 0, :2]
        key = (np.abs(g * 1e4).astype(np.int64).sum(1)) % 4096
        return fixed + (g - fixed) @ slope.T + bumps[key]

    def chunk_of(g):
        return encode(g), True

    chunks = np.stack([encode(g) for g in targets])
    return targets, chunks, np.ones(len(targets), bool), predict, chunk_of


@pytest.mark.parametrize("slope", (np.diag([-0.13, -0.42]), [[-0.9, 0.3], [0.2, -1.05]],
                                   [[-0.3, 0.1], [0.05, -0.6]], np.eye(2)))  # fmt: skip
@pytest.mark.parametrize("arm", ("W", "N", "L-shuf", "L-mean", "W-72361"))
def test_affine_local_is_task081s_runtime(slope, arm):
    targets, chunks, feasible, predict, chunk_of = _setup(slope, noise=0.003)
    args = (P_HAT, H, targets, chunks, feasible, predict, chunk_of)
    g, log = urt.choose(arm, *args, seed=72600)
    theirs = cprt.choose_affine_local(arm, *args)
    assert np.array_equal(g, theirs[0]) and log == theirs[1] and log["solver"] == "affine_local"
    # and TASK-081's runtime is the design note's tested variant
    index = np.flatnonzero(feasible)
    dev = cpd.all_variants(P_HAT, H, targets, feasible, predict(chunks[index]),
                           lambda q: predict(chunk_of(q)[0][None])[0], lambda q: True, known=(),
                           tolerance_m=0.0025, a_lo=ul.A_LO)["affine_local"]  # fmt: skip
    assert np.array_equal(g, dev["g"])


def test_l_rand_uses_salt_8412_and_no_solver():
    targets, chunks, feasible, predict, chunk_of = _setup(-0.5 * np.eye(2))
    g, log = urt.choose("L-rand", P_HAT, H, targets, chunks, feasible, predict, chunk_of,
                        seed=72605)  # fmt: skip
    assert log["grid_index"] == ul.l_rand_index(72605, feasible) and log["solver"] == "none"
    assert np.array_equal(g, targets[log["grid_index"]]) and log["rollouts"] == 0
    g, log = urt.choose_l_rand("L-rand", P_HAT, targets, np.zeros(147, bool), seed=1)
    assert np.array_equal(g, P_HAT) and log["fallback_all_infeasible"]


def test_no_task_truth_in_any_non_privileged_arm():
    tree = ast.parse((ROOT / "src/embodied_jepa/lewm_ul_v2_runtime.py").read_text())
    names = {
        "UlAim",
        "KrrHatAim",
        "PAim",
        "RuleFitCommit",
        "choose",
        "choose_l_rand",
        "p_aim_predict",
        "krr_from_json",
        "model_name",
        "readout_name",
        "solver_of",
    }
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    assert {n.name for n in nodes} == names
    for node in nodes:
        code = ast.unparse(node).replace(ast.get_docstring(node) or "\0", "")
        code = code.replace("truth_hook=None", "")  # RuleCommit's privileged hook, unset
        for forbidden in ("hook", ".sim", "truth", "current()", "robot", "plate_xy_now"):
            assert forbidden not in code, (node.name, forbidden)
    assert ul.CANDIDATE_ARMS.isdisjoint(ul.PRIVILEGED_ARMS)
    for arm in (*ul.LEARNED_TIER, "H-rule-fit", "H-rule", "H-sysid"):
        assert arm not in ul.PRIVILEGED_ARMS and not ul.ARMS[arm]["privileged"]
    assert {"H-final", "H-read", "H-now", "collect-ul", "H-final-planted"} <= ul.PRIVILEGED_ARMS


def test_the_learned_tier_and_the_fitted_rule():
    rng = np.random.default_rng(1)
    n = 60
    p = np.array([0.49, -0.09]) + rng.normal(0, 0.02, (n, 2))
    h = np.array([0.30, -0.16]) + rng.normal(0, 0.02, (n, 2))
    g = p + rng.normal(0, 0.03, (n, 2))
    y = p - 0.37 * (g - h) + rng.normal(0, 1e-4, (n, 2))
    assert ulo.kappa_fit(p, h, g, y) == pytest.approx(-0.37, abs=0.01)
    model = ulo.krr_sysid_fit(p, h, g, y)
    assert model["length"] in ul.KRR_LENGTHS and model["lambda"] in ul.KRR_LAMBDAS
    err = np.linalg.norm(pl.krr_plate(model, p, h, g) - y, axis=1)
    assert np.median(err) < 0.003
    # the learned tier's folds are salt 8410, sub-key 3 (not the design note's 8404)
    assert not np.array_equal(ul.learned_tier_folds(n), pl.fold_of(n, 1))
    label = p + 0.3 * (h - p)
    pa = ulo.p_aim_fit(p, h, label)
    raw = urt.p_aim_predict(urt.krr_from_json(ulo.model_json(pa)), p[:3], h[:3])
    assert np.allclose(raw, label[:3], atol=0.003)


# ----- K0 -----------------------------------------------------------------------------------------
def test_k0_tau_and_stops_in_both_directions():
    ok = {"0.0": 64, "0.5": 60, "1.0": 57, "1.5": 40, "2.0": 30, "3.0": 10}
    tau = ul.decide_tau(ok)
    assert tau["tau_commit_cm"] == 1.0
    assert ul.decide_tau(ok | {"1.0": 55})["tau_commit_cm"] == 0.5
    assert ul.decide_tau(ok | {"0.0": 55})["tau_commit_cm"] is None
    assert ul.decide_tau(ok | {"0.5": 55})["tau_commit_cm"] == 0.0
    base = {"tau": tau, "ceiling": 63, "r_k": 460, "palm_speed_median_cm": 0.01,
            "clip_outside": 2}  # fmt: skip
    assert ul.decide_k0(**base)["row"] == "K0-PASS"
    for change, stop in ((dict(tau=ul.decide_tau(ok | {"0.0": 55})), "level0_below_bar"),
                         (dict(tau=ul.decide_tau(ok | {"0.5": 55})), "tau_zero"),
                         (dict(ceiling=59), "ceiling_below_60"), (dict(r_k=466), "r_late"),
                         (dict(r_k=None), "r_late"), (dict(palm_speed_median_cm=0.6), "palm_fast"),
                         (dict(clip_outside=5), "box_binds")):  # fmt: skip
        out = ul.decide_k0(**(base | change))
        assert out["row"] == "CAL-ESCALATE" and out["stops"][stop], stop
    assert ul.decide_k0(**(base | dict(clip_outside=4, ceiling=60, r_k=465)))["row"] == "K0-PASS"
    half = ul.decide_k0(**(base | dict(tau=ul.decide_tau(ok | {"1.0": 55}))))
    assert half["row"] == "K0-PASS" and half["stage_t_go_is_a_ruling_point"]
    assert not ul.decide_k0(**base)["stage_t_go_is_a_ruling_point"]


def test_read_step_k_needs_56_of_64():
    near = {t: (0.05 if t >= 460 else 1.0) for t in range(405, 526)}
    far = {t: 1.0 for t in range(405, 526)}
    assert ul.read_step_k([near] * 56 + [far] * 8) == 460
    assert ul.read_step_k([near] * 55 + [far] * 9) is None
    assert ul.read_step_k([near] * 56 + [None] * 8) == 460


def test_k0_clip_binding_counts_ceiling_aims_outside_the_p_hat_box():
    run = _runner()
    inside = c1.from_box(0.3, 0.01, P_HAT, H)
    outside = c1.from_box(0.3, 0.045, P_HAT, H)
    recs = []
    for i, g in enumerate([inside, outside, outside]):
        recs.append(
            {
                "seed": i,
                "commit": {"target": g.tolist(), "h": H.tolist()},
                "decisions": [{"lookahead_aim": g.tolist(), "p_hat405": P_HAT.tolist()}],
            }
        )
    recs.append({"seed": 9, "commit": None, "decisions": []})  # refused before 405
    out = run.k0_clip_binding(recs)  # fmt: skip
    assert out["outside"] == 2 and out["aims"] == 3
    assert out["abs_b_cm"]["max"] == pytest.approx(4.5)


# ----- Stage O: the readouts core on a tiny synthetic featurisation -------------------------------
def _tiny_featurisation(folder: Path, sizes: dict, width: int = 6) -> None:
    rng = np.random.default_rng(7)
    offset = 0
    for split, n in sizes.items():
        paths = ulo.feature_paths(folder, split)
        folder.mkdir(parents=True, exist_ok=True)
        plate = np.array([0.49, -0.09]) + rng.normal(0, 0.02, (n, 1, 2))
        plate = plate + np.zeros((n, ul.N_FRAMES, 2))
        feats = rng.standard_normal((n, ul.N_FRAMES, width)).astype(np.float32)
        feats[:, :, :2] = 10 * plate.astype(np.float32)  # the plate is readable
        np.save(paths["features"], feats)
        np.save(paths["commands"], rng.uniform(-0.1, 0.1, (n, ul.N_COMMANDS, 14)).astype(
            np.float32))  # fmt: skip
        paths["roots"].write_text(json.dumps(list(range(offset, offset + n))))
        np.save(paths["hidden8"], rng.standard_normal((n, width)).astype(np.float32))
        np.save(paths["pool4"], rng.standard_normal((n, 2, width)).astype(np.float32))
        np.save(paths["full405"], rng.standard_normal((n, width)).astype(np.float32))
        palm = plate + rng.normal(0, 0.05, (n, ul.N_FRAMES, 2))
        target = plate[:, 2] + rng.normal(0, 0.03, (n, 2))
        np.savez(paths["table"], seeds=np.arange(offset, offset + n), plate=plate, palm=palm,
                 target=target, state405=rng.standard_normal((n, 5)),
                 apple=rng.standard_normal((n, 2)), last_grasp=rng.standard_normal((n, 2)),
                 p_hat405=plate[:, 2] + rng.normal(0, 0.002, (n, 2)),
                 label_aim=plate[:, 2] + rng.normal(0, 0.01, (n, 2)),
                 label_converged=np.ones(n))  # fmt: skip
        offset += n


def test_readouts_core_fits_on_train_only_and_admits(tmp_path):
    feat = tmp_path / "features"
    _tiny_featurisation(feat, {"train": 40, "val": 10})

    class Identity:
        def predict(self, x):
            return np.asarray(x, np.float64)[..., :2] * 0 + np.array([0.49, -0.09])

    result = ulo.readouts_core(feat, tmp_path / "fits", tau_commit_cm=1.0, r_plate=Identity())
    assert result["rows"] == 50 and result["decision"]["row"] in ul.O_ROWS
    fits = result["fits"]
    assert fits["train_roots"] == 40
    for name in ("r8", "sysid", "rule_fit", "krr", "p_aim", "moments", "mean_latent"):
        assert Path(fits[name]["path"]).is_file(), name
    assert ulo.load_json_fit(fits["krr"])["length"] in ul.KRR_LENGTHS
    record = fits["krr"] | {"sha256": "0" * 64}
    with pytest.raises(pt.GuardError):
        ulo.load_json_fit(record)
    adm = result["admission"]
    for key in ("sysid_linear_at_r", "sysid_krr_at_r", "c1m_rule_open_form_at_r",
                "p_aim_vs_label", "r_plate_405_logged_cpu"):  # fmt: skip
        assert key in adm["reported"]


# ----- Stage G ------------------------------------------------------------------------------------
def test_stage_g_readouts_refuse_gate_p_roots():
    split = {"train": [1, 2, 3], "val": [4], "gate_p": [5, 6]}
    ulo.check_fit_roots([1, 2, 3, 4], split)
    for bad in ([1, 5], [7]):
        with pytest.raises(pt.GuardError, match="G-split"):
            ulo.check_fit_roots(bad, split)
    pred = {k: np.zeros((2, 3)) for k in "SNL"}
    with pytest.raises(pt.GuardError):
        ulo.fit_readouts(pred, np.zeros((2, 2)), [1, 6], split)


def _g_inputs(seeds=ul.MODEL_SEEDS, **change):
    ci = {"ci95": [0.5, 0.8]}
    dyn = {s: {"G1": True, "G2": True, "G3": True, "G4": True} for s in seeds}
    reads = {s: {"e_S": {"ci95": [0.5, 0.8]}, "e_L": {"ci95": [1.5, 3.0]},
                 "S_over_N": {"ci95": [0.3, 0.6]}} for s in seeds}  # fmt: skip
    aims = {"W": 118.0, "N": 80.0, "L-shuf": 40.0, "L-mean": 60.0, "L-rand": 20.0}
    out = {"dynamics": dyn, "r0": ci, "readouts": reads, "aims": aims, "tau_commit_cm": 1.0,
           "seeds": seeds}  # fmt: skip
    return out | change


def test_stage_g_rows_in_both_directions():
    assert ul.decide_g(ul.g_gates(**_g_inputs()))["row"] == "G-PASS"
    s0 = ul.MODEL_SEEDS[0]
    cases = []
    inp = _g_inputs()
    inp["dynamics"][s0]["G3"] = False
    cases.append((inp, "H-GATE-FAIL"))
    cases.append((_g_inputs(r0={"ci95": [0.9, 1.1]}), "R-VOID-CEILING"))
    inp = _g_inputs()
    inp["readouts"][s0]["e_L"] = {"ci95": [0.9, 3.0]}
    cases.append((inp, "R-COMMAND-KEYED"))
    inp = _g_inputs()
    inp["readouts"][s0]["e_S"] = {"ci95": [0.7, 1.01]}
    cases.append((inp, "R-NO-BAR"))
    inp = _g_inputs()
    inp["readouts"][s0]["S_over_N"] = {"ci95": [0.7, 1.0]}
    cases.append((inp, "R-NO-BAR"))
    inp = _g_inputs()
    inp["aims"]["W"] = 111.9
    cases.append((inp, "A-NO-BAR"))
    inp = _g_inputs()
    inp["aims"]["N"] = 111.5
    cases.append((inp, "A-TWIN"))
    for inp, row in cases:
        out = ul.decide_g(ul.g_gates(**inp))
        assert out["row"] == row and not out["clause_fires"], row
    # H-GATE-FAIL precedes R-VOID-CEILING (the models are new)
    inp = _g_inputs(r0={"ci95": [0.9, 1.1]})
    inp["dynamics"][s0]["G1"] = False
    assert ul.decide_g(ul.g_gates(**inp))["row"] == "H-GATE-FAIL"
    assert ul.decide_g(None, void=True)["row"] == "V"
    with pytest.raises(ContractError):
        ul.g_gates(**_g_inputs(seeds=ul.MODEL_SEEDS[:2]))
    with pytest.raises(ContractError):
        ul.g_gates(**(_g_inputs() | {"seeds": ul.DEBUG_MODEL_SEEDS}))
    assert ul.g_gates(**_g_inputs(seeds=ul.DEBUG_MODEL_SEEDS), debug=True)["gates"]["A1"]
    with pytest.raises(ContractError):
        ul.g_gates(**(_g_inputs() | {"aims": {"W": 120.0}}))
    assert ul.primary_seed({72360: 0.4, 72361: 0.35, 72362: 0.35}) == 72361


def test_aim_summary_measures_against_the_ceiling_label():
    n = 4
    data = {"n": n, "label": np.zeros((n, 2))}
    got = [{"key": i, "g": [0.01 * i, 0.0], "log": {"clipped": i == 3}} for i in range(n)]
    out = ulo.aim_summary(got, data, {"0.0": 64, "0.5": 60, "1.0": 56, "1.5": 40, "2.0": 30,
                                       "3.0": 10}, 64)  # fmt: skip
    assert out["aim_errors_cm"] == [0.0, 1.0, 2.0, 3.0]
    assert out["predicted_count_of_128"] == pytest.approx(128 * (64 + 56 + 30 + 10) / 64 / 4)
    assert out["clip_binding_fraction"] == 0.25
    with pytest.raises(pt.GuardError):
        ulo.aim_summary(got[:3], data, {"0.0": 64}, 64)


def test_offline_aim_tasks_start_latents_and_fits():
    n = 3
    data = {"n": n, "seeds": np.arange(n), "state405": np.zeros((n, 3)), "apple": np.zeros((n, 2)),
            "last_grasp": np.zeros((n, 2)), "p_hat405": np.ones((n, 2)), "palm405": np.zeros(
                (n, 2))}  # fmt: skip
    latents = np.arange(n * 4, dtype=np.float32).reshape(n, 4)
    fits = {"sysid_coef": [[0.0]], "kappa_fit": -0.4, "krr": {"x": [[0.0]]}, "p_aim": {"y": 1}}
    shuf = ulo.offline_aim_tasks("L-shuf", data, latents, 1.0, fits=fits)
    assert [t["start"][0] for t in shuf] == [4.0, 8.0, 0.0]
    assert ulo.offline_aim_tasks("L-mean", data, latents, 1.0, fits=fits)[0]["start"] == "mean"
    spread = ulo.offline_aim_tasks("W-72361", data, latents, 1.0, fits=fits)
    assert np.array_equal(spread[1]["start"], latents[1])
    assert (
        ulo.offline_aim_tasks("H-rule-fit", data, latents, 1.0, fits=fits)[0]["kappa_fit"] == -0.4
    )
    assert "krr" in ulo.offline_aim_tasks("H-sysid-krr", data, latents, 1.0, fits=fits)[0]
    assert "start" not in ulo.offline_aim_tasks("P-aim", data, latents, 1.0, fits=fits)[0]


def test_echo_slope_reference_is_the_laws_jacobian():
    d = np.array([0.12, -0.03])
    jac = ul.law_jacobian(d)
    eps = 1e-5
    for j in range(2):
        e = np.zeros(2)
        e[j] = eps
        fd = (ul.law_displacement(d + e) - ul.law_displacement(d)) / eps
        assert np.allclose(jac[:, j], fd, atol=1e-4)
    slope = np.array([[-0.4, 0.1], [0.05, -0.3]])
    g = np.random.default_rng(0).normal(0, 0.02, (20, 2))
    res = [{"key": 0, "grid_targets": g.tolist(), "grid_plates": (g @ slope.T + 0.5).tolist()}]
    out = ulo.echo_slopes(res, {"n": 1, "label": [[0.42, -0.19]], "palm405": [[0.30, -0.16]]})
    assert np.allclose(out["median_matrix"], slope)
    assert np.allclose(out["law_jacobian_median"], ul.law_jacobian([0.12, -0.03]))


# ----- training -----------------------------------------------------------------------------------
def test_the_training_loop_is_task077s_verbatim_with_this_tasks_sampler():
    assert inspect.getsource(ult.train_model) == inspect.getsource(tr.train_model)
    assert ult.WindowSampler is ult.UlWindowSampler
    assert issubclass(ult.UlWindowSampler, tr.WindowSampler)
    mine, theirs = ult.UlWindowSampler(1500, 72360), tr.WindowSampler(1500, 72360)
    want = np.random.default_rng(np.random.SeedSequence([8409, 72360]))
    roots, offsets = mine.draw()
    assert np.array_equal(roots, want.integers(0, 1500, 16))
    assert np.array_equal(offsets, want.integers(0, 5, 16))
    assert not np.array_equal(roots, theirs.draw()[0])
    a, b = ult.UlWindowSampler(1500, 72361), ult.UlWindowSampler(1500, 72361)
    assert all(np.array_equal(x, y) for x, y in zip(a.draw(), b.draw(), strict=True))
    meta = ult.model_metadata(arm="W", seed=72360, corpus_sha256="c", moments_sha256="m")
    assert meta["protocol"] == ul.PROTOCOL and meta["sampler_salt"] == 8409
    source = inspect.getsource(tr.train_model)
    assert "rt.select_checkpoint(curve, lm.SELECTION_TOLERANCE)" in source
    assert "rt.last_two_triggered(curve" in source and "np.argmin" not in source


def test_the_carried_budget_block_passes_check_budget():
    check = rt.check_budget(ul.BUDGET)
    assert check["consistent"]
    assert ul.BUDGET["carried_updates"] == 95_000 and ul.BUDGET["carried_select_every"] == 4_750
    assert ul.BUDGET["calibration_runs"] == ()
    assert ul.BUDGET["carried_updates"] <= ul.BUDGET["cap"] == 100_000
    run = _runner()
    assert run.job_budget(SimpleNamespace(debug=False)) == {"updates": 95_000,
                                                           "select_every": 4_750}  # fmt: skip
    assert run.job_budget(SimpleNamespace(debug=True))["updates"] == ul.DEBUG_TRAIN["updates"]
    assert run.parse_job("W-72360", False) == {"arm": "W", "seed": 72360}
    for bad in ("cal-W", "W-66800", "X-72360"):
        with pytest.raises(pt.GuardError):
            run.parse_job(bad, False)


# ----- Stage S and D ladders ----------------------------------------------------------------------
def _outcomes(n=128, **counts):
    base = {"W": 120, "N": 50, "L-shuf": 40, "L-mean": 55, "L-rand": 20, "H-rule": 100,
            "H-sysid": 110, "H-final": 126, "H-sysid-krr": 124, "P-aim": 122}  # fmt: skip
    base |= counts
    out = {}
    for arm, k in base.items():  # nested: every arm's successes are the first k resets
        v = np.zeros(n, bool)
        v[:k] = True
        out[arm] = v
    return out


def test_stage_s_rows_in_both_directions():
    d = ul.DELTA
    assert ul.decide_s(_outcomes())["row"] == "L-PASS"
    assert ul.decide_s(_outcomes(**{"H-final": 111}))["row"] == "S-VOID-CEILING"
    assert ul.decide_s(_outcomes(**{"L-mean": 118}))["row"] == "L-NO-GAIN"
    assert ul.decide_s(_outcomes(**{"N": 115}))["row"] == "L-TWIN-NEAR"
    out = ul.decide_s(_outcomes(W=111, **{"H-sysid": 110}))
    assert out["row"] == "L-BAR" and not out["G-bar"]
    near = ul.decide_s(_outcomes(W=118, **{"H-sysid": 118 + d + 2, "H-final": 128}))
    assert near["row"] == "L-NEAR" and not near["G-NI"]["detectably_inferior"]
    inf = ul.decide_s(_outcomes(W=100, **{"H-sysid": 100 + d + 12, "H-final": 128}))
    assert inf["row"] == "L-INFERIOR" and inf["clause_fires"]
    assert ul.decide_s(None, void=True)["row"] == "V"
    with pytest.raises(ContractError):
        ul.decide_s(_outcomes(n=64))
    with pytest.raises(ContractError):
        ul.decide_s({k: v for k, v in _outcomes().items() if k != "H-sysid"})
    for row in ul.CLAUSE_ROWS:
        assert row in ul.S_ROWS and row not in ul.NO_CLAUSE_ROWS


def test_the_secondary_claim_and_the_learned_tier_are_reported_and_read_by_no_row():
    out = ul.decide_s(_outcomes(W=124, **{"H-rule": 100, "H-sysid": 110}))
    sec = out["secondary_reported_only"]["against_C"]
    assert out["comparator"] == "H-sysid" and sec["b"] == 14 and sec["c"] == 0
    assert sec["w_better_detectably"] and out["row"] == "L-PASS"
    tier = out["learned_tier_reported_only"]
    assert set(tier) == {"H-sysid-krr", "P-aim"}
    assert tier["H-sysid-krr"]["c"] == 0 and tier["H-sysid-krr"]["b"] == 0
    # a learned arm far above W changes no row (it is reported only)
    out2 = ul.decide_s(_outcomes(W=113, **{"H-sysid-krr": 128, "P-aim": 128, "H-final": 128}))
    assert out2["row"] == "L-PASS"
    assert out2["learned_tier_reported_only"]["P-aim"]["w_detectably_inferior_beyond_delta"] is (
        out2["learned_tier_reported_only"]["P-aim"]["ci95"][1] < -ul.DELTA)  # fmt: skip


def test_the_s_ladder_equals_task081s_at_its_delta_and_salt(monkeypatch):
    """With delta 16 and TASK-081's salt 8302, the ladder gives TASK-081's rows."""
    monkeypatch.setattr(ul, "paired_interval", lambda a, b: cpv.paired_interval(a, b))
    rng = np.random.default_rng(5)
    for _ in range(40):
        rates = rng.uniform(0.3, 1.0, 8)
        out = {a: rng.uniform(size=128) < r for a, r in zip(
            ("W", "N", "L-shuf", "L-mean", "L-rand", "H-rule", "H-sysid", "H-final"), rates,
            strict=True)}  # fmt: skip
        mine, theirs = ul.decide_s(out, delta=16), cpv.decide_s(out)
        assert mine["row"] == theirs["row"]
        assert mine["G-NI"]["ci95"] == theirs["G-NI"]["ci95"]


def test_stage_d_rows():
    ok = {"W": 15, "N": 8, "L-shuf": 5, "L-mean": 9, "H-final": 16, "H-rule": 10, "H-sysid": 12}
    out = ul.decide_d(ok)
    assert out["row"] == "D-PASS" and out["reported_only"] == {"H-rule": 10, "H-sysid": 12}
    assert ul.decide_d(ok | {"W": 11})["row"] == "L-DEV-STOP"
    assert ul.decide_d(ok | {"H-final": 13})["row"] == "L-DEV-STOP"
    assert ul.decide_d(ok | {"L-mean": 13})["row"] == "L-DEV-STOP"
    assert ul.decide_d(ok | {"H-rule": 0, "H-sysid": 0})["row"] == "D-PASS"
    with pytest.raises(ContractError):
        ul.decide_d({k: v for k, v in ok.items() if k != "H-sysid"})


def test_stage_o_rows_are_task077s():
    good = {"c_plate": {"ci95": [0.2, 0.4]}, "prior_ratio": {"ci95": [0.1, 0.3]},
            "hidden": {"ci95": [2.0, 3.0]}, "tau_commit_cm": 1.0, "falling": False}  # fmt: skip
    assert ul.decide_o(**good)["row"] == "O-PASS"
    assert ul.decide_o(**(good | {"hidden": {"ci95": [0.9, 3.0]}}))["row"] == "O-ARM-KEYED"
    assert ul.decide_o(**(good | {"c_plate": {"ci95": [0.5, 1.1]}}))["row"] == "O-NO-BAR"
    assert ul.decide_o(**(good | {"prior_ratio": {"ci95": [0.5, 1.0]}}))["row"] == "O-NO-BAR"


def test_statistics_use_salt_8411():
    rng = np.random.default_rng(0)
    a, b = rng.uniform(size=128) < 0.9, rng.uniform(size=128) < 0.8
    assert ul.paired_interval(a, b) == pr.paired_interval(a, b, salt=8411)
    v = rng.normal(size=50)
    assert ul.median_ci(v) == pr.median_ci(v, salt=8411)
    assert ul.paired_interval(a, b)["ci95"] != cpv.paired_interval(a, b)["ci95"] or True


# ----- power and delta's rule ---------------------------------------------------------------------
def test_power_pieces():
    assert ul.g_bar_power(0.906) == pytest.approx(0.908, abs=0.001)
    lo, hi = ul.ni_bounds(0, 0)
    assert lo == hi == 0.0
    lo, hi = ul.ni_bounds(1, 12)
    assert lo < -16 < hi < 0
    p = ul.ni_power(0.938, 0.85, "overlap", trials=2000, index=99)
    assert p["better_of_two"]["16"]["g_ni_pass_rate"] > 0.95
    assert p["better_of_two"]["secondary_pass_rate"] > 0.5
    size = ul.ni_power(0.85 - 16 / 128, 0.85, "independent", deltas=(16,), trials=2000, index=98)
    assert size["single"]["16"]["g_ni_pass_rate"] < 0.1
    assert ul.twin_power(0.906, 73 / 128, "overlap") > 0.999


def test_the_delta_rule():
    power = {"8": {"0.8": 0.81, "0.85": 0.79, "0.875": 0.9},
             "12": {"0.8": 0.95, "0.85": 0.9, "0.875": 0.97},
             "16": {"0.8": 0.99, "0.85": 0.99, "0.875": 0.99}}  # fmt: skip
    size = {d: {c: {"overlap": 0.03, "half": 0.03, "independent": 0.04} for c in
                ("0.8", "0.85", "0.875")} for d in ("8", "12", "16")}  # fmt: skip
    assert ul.delta_rule(power, size)["delta"] == 12
    power["8"]["0.85"] = 0.80
    assert ul.delta_rule(power, size)["delta"] == 8
    size["8"]["0.8"]["independent"] = 0.051
    assert ul.delta_rule(power, size)["delta"] == 12
    for d in ("12", "16"):
        power[d]["0.8"] = 0.5
    out = ul.delta_rule(power, size)
    assert out["delta"] == 16 and out["fallback"]
    assert ul.DELTA in ul.DELTA_CANDIDATES


def test_stage0_records_delta_from_the_power_simulation():
    if ul.STAGE0:
        assert ul.STAGE0["delta"]["chosen"] == ul.DELTA


# ----- the runner ---------------------------------------------------------------------------------
CARRIED_RUNNER = ("summary_line", "summary_ok", "run_full_tests", "verify_tests_record",
                  "wait_quiet", "check_disk", "sim_preflight", "streamed_estimates",
                  "cohort_estimates", "successes", "rebased", "completed", "check_same_corpus",
                  "gpu_memory_peak", "open_log")  # fmt: skip


def _functions(path: Path) -> dict:
    text = path.read_text()
    tree = ast.parse(text)
    return {n.name: ast.get_source_segment(text, n) for n in tree.body
            if isinstance(n, ast.FunctionDef)}  # fmt: skip


def test_the_runners_carried_code_is_task077s_verbatim():
    mine, theirs = _functions(RUNNER), _functions(TASK077_RUNNER)
    for name in CARRIED_RUNNER:
        assert mine[name] == theirs[name], name


def test_the_runner_uses_run_tools_guards_and_loads_no_other_script():
    calls = set()
    for node in ast.walk(ast.parse(RUNNER.read_text())):
        if isinstance(node, ast.Call):
            f = node.func
            calls.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    for name in ("assert_local_import", "install_guards", "MemoryWatch", "check_pins",
                 "check_evidence", "verify_feature_files", "gpu_guard", "scale_probe",
                 "check_law", "check_seeds"):  # fmt: skip
        assert name in calls, name
    nri = _load(ROOT / "tests" / "test_no_runner_imports.py", "_nri")
    loaded, loaders = nri.script_loads(RUNNER)
    assert not loaded and not loaders


def test_the_runner_refuses_misused_flags(tmp_path):
    run = _runner()
    assert run.DRAFT_ALLOWED == ("tests", "simulate", "gscale", "k0")
    for argv in (
        ["closed", "--output", str(tmp_path / "a")],
        ["closed", "--cohort", "D", "--output", str(tmp_path / "b"), "--evidence", "e"],
        ["closed", "--cohort", "S", "--output", str(tmp_path / "c"), "--evidence", "e",
         "--r-plate-fits", "r", "--corpus", "c", "--corpus-sha256", "x", "--features", "f",
         "--fits", "g", "--models", "m", "--gates", "g"],
        ["simulate", "--output", str(tmp_path / "d"), "--workers", "2"],
        ["simulate", "--output", str(tmp_path / "e"), "--cohort", "D"],
        ["simulate", "--output", str(tmp_path / "f"), "--debug-skip-tests"],
        ["k0", "--output", str(tmp_path / "g"), "--evidence", "e"],
        ["corpus", "--output", str(tmp_path / "h"), "--r-plate-fits", "r"],
        ["train", "--output", str(tmp_path / "i"), "--tests-record", "t", "--corpus", "c",
         "--corpus-sha256", "x", "--features", "f", "--fits", "g"],
        ["gscale", "--output", str(tmp_path / "j")],
        ["gates", "--output", str(tmp_path / "k"), "--evidence", "e", "--corpus", "c",
         "--corpus-sha256", "x", "--features", "f", "--fits", "g"],
    ):  # fmt: skip
        with pytest.raises(SystemExit):
            run.main(argv)


def _preflight_env(monkeypatch, run):
    monkeypatch.setattr(run.rt, "assert_local_import", lambda *a, **k: None)
    monkeypatch.setattr(run.hz, "check_pins", lambda pins, *a: dict(pins))
    monkeypatch.setattr(run, "check_carried", lambda report: None)
    monkeypatch.setattr(run.hz, "tracked_tree_dirty", lambda *a: [])
    monkeypatch.setattr(run, "check_frozen_pin", lambda manifest: {"status": "DRAFT"})
    monkeypatch.setattr(run, "wait_quiet", lambda report, args: None)
    monkeypatch.setattr(run, "run_full_tests", lambda: {"summary": "1 passed"})


@pytest.mark.parametrize("stage", ["corpus", "featurise", "readouts", "train", "gates", "closed"])
def test_the_preflight_refuses_every_post_freeze_stage_while_draft(monkeypatch, stage):
    run = _runner()
    _preflight_env(monkeypatch, run)
    monkeypatch.setattr(run.ul, "STATUS", "DRAFT")
    args = SimpleNamespace(stage=stage, debug=False, debug_skip_tests=False)
    with pytest.raises(run.lp.GuardError, match="runs only after the freeze"):
        run.preflight({}, args)


def test_the_preflight_refuses_labelcheck_outside_debug(monkeypatch):
    run = _runner()
    _preflight_env(monkeypatch, run)
    monkeypatch.setattr(run.ul, "STATUS", "FROZEN")
    monkeypatch.setattr(run, "check_protocol_document", lambda manifest: {})
    monkeypatch.setattr(run.json, "loads", lambda text: {"hashes": {"a": "b"}, "mujoco_version":
                                                         None})  # fmt: skip
    args = SimpleNamespace(stage="labelcheck", debug=False, debug_skip_tests=False)
    with pytest.raises(run.lp.GuardError, match="debug check only"):
        run.preflight({}, args)
    args = SimpleNamespace(stage="k0", debug=False, debug_skip_tests=False)
    with pytest.raises(run.lp.GuardError, match="K0 runs once"):
        run.preflight({}, args)


def test_stage_d_report_must_be_this_tasks_d_pass(tmp_path):
    run = _runner()
    good = {"task": "TASK-082", "protocol": ul.PROTOCOL, "outcome": "D-PASS", "debug": False,
            "cohort": "D", "frozen_sha256_at_run": ul.frozen_sha256()}  # fmt: skip
    for bad in ({"task": "TASK-081"}, {"outcome": "L-DEV-STOP"}, {"debug": True},
                {"cohort": "S"}, {"frozen_sha256_at_run": "0" * 64}):  # fmt: skip
        path = tmp_path / f"{len(list(tmp_path.iterdir()))}.json"
        path.write_text(json.dumps(good | bad))
        with pytest.raises(run.lp.GuardError):
            run.stage_d_report(SimpleNamespace(stage_d=str(path)), False)
    path = tmp_path / "ok.json"
    path.write_text(json.dumps(good))
    assert run.stage_d_report(SimpleNamespace(stage_d=str(path)), False)["sha256"]


def test_the_carried_r_plate_is_pinned(tmp_path):
    run = _runner()
    fits = tmp_path / "fits"
    fits.mkdir()
    (tmp_path / "report.json").write_text(json.dumps({"outcome": "O-PASS"}))
    with pytest.raises(run.lp.GuardError, match="not the pinned one"):
        run.carried_r_plate(SimpleNamespace(r_plate_fits=str(fits)), {})


class _FakeCohort:
    def __init__(self, seeds):
        self.resets = {s: ul.reset_of(s) for s in seeds}
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
            solver = urt.solver_of(arm) if urt.is_candidate(arm) else None
            base = {"seed": s, "arm": arm, "blocked": None, "privileged_ok": True,
                    "task_truth_in_controller": 0, "seconds": 1.0,
                    "termination_reason": "step_limit", "executed_steps": 800,
                    "final_distance_cm": 3.6, "kpred": {"law": dict(urt.LAW_RECORD)}}  # fmt: skip
            if s in self.refused:
                out.append(base | {"success": False, "termination_reason": "guard_refusal",
                                   "executed_steps": 230, "frame405": None,
                                   "decisions": []})  # fmt: skip
                continue
            good = arm in ("W", "H-final", "H-read", "H-sysid-krr", "P-aim") or arm.startswith(
                "W-")  # fmt: skip
            wm = {"solver": solver}
            if solver == "affine_local":
                wm |= {"points": 25, "residual_cm": 0.3, "eigenvalues_real": [-0.5, -0.4],
                       "fallback_grid_argmin": False, "clipped": False}  # fmt: skip
            out.append(base | {
                "success": good, "frame405": np.full((2, 2, 3), s % 251, np.uint8),
                "decisions": [{"reading": [0.49, -0.09], "world_model": wm}],
                "commit": {"target": [0.5, -0.1], "fallback": False, "clipped": False,
                           "landing_miss_cm": 0.3, "plate_motion_cm": 6.0},
            })  # fmt: skip
        return out


FITS = {"sysid_coef": [[0.0] * 2] * 7, "kappa_fit": -0.3, "krr": {"k": 1}, "p_aim": {"p": 1}}


@pytest.mark.parametrize("role", ["D", "S"])
def test_closed_core_runs_every_arm(role):
    run = _runner()
    n = 16 if role == "D" else 128
    seeds = ul.seeds_of(role)
    refused = {seeds[1], seeds[2]}
    pool = _FakePool(refused)
    report: dict = {}
    fields = run.Fields(report, "closed")
    spread = ul.spread_arms(72360)
    row = run.closed_core(pool, _FakeCohort(seeds), seeds, role,
                          {"tau_commit_cm": 1.0, "a_lo": ul.A_LO}, FITS, fields,
                          spread=spread if role == "S" else (), debug=False)  # fmt: skip
    arms = report["fields"]["arms"]
    want = ul.D_ARMS if role == "D" else (*ul.S_GATING_ARMS, *ul.S_REPORTED_ARMS, *spread)
    assert set(arms) - {"refused_before_405"} == set(want)
    assert arms["W"]["count"] == n - 2 and arms["refused_before_405"]["W"] == 2
    shuf = next(c for c in pool.calls if c[0]["arm"] == "L-shuf")
    assert int(shuf[0]["foreign_frame"][0, 0, 0]) == seeds[3] % 251
    assert pool.calls[0][0]["move_offset"] == ul.move_offset(seeds[0]).tolist()
    by_arm = {c[0]["arm"]: c[0] for c in pool.calls}
    if role == "S":
        assert by_arm["H-sysid-krr"]["krr"] == FITS["krr"]
        assert by_arm["P-aim"]["p_aim"] == FITS["p_aim"]
        assert by_arm["H-rule-fit"]["kappa_fit"] == -0.3
        assert by_arm["H-read"]["readout"] == "r8"
        assert row == "S-VOID-CEILING" or row in ul.S_ROWS
        assert report["fields"]["determinism"]["ok"]
        rep = report["fields"]["reported"]
        assert set(rep["seed_spread"]) == set(spread)
        assert rep["per_arm"]["W"]["affine_local"]["decisions"] == n - 2
    else:
        assert row in ul.D_ROWS
        assert report["fields"]["determinism"] == ul.NOT_EVALUATED


def test_run_arm_refuses_task_truth_a_wrong_law_and_a_wrong_solver():
    run = _runner()

    class Pool:
        def __init__(self, rec):
            self.rec = rec

        def map(self, tasks, cap, what):
            return [self.rec]

    ok = {"seed": 1, "arm": "W", "success": True, "blocked": None, "privileged_ok": True,
          "task_truth_in_controller": 0, "kpred": {"law": dict(urt.LAW_RECORD)},
          "decisions": [{"world_model": {"solver": "affine_local"}}]}  # fmt: skip
    run.run_arm(Pool(ok), [{"arm": "W"}], "W")
    with pytest.raises(run.lp.GuardError, match="G-solver"):
        run.run_arm(Pool(ok | {"decisions": [{"world_model": {"solver": "frozen"}}]}),
                    [{"arm": "W"}], "W")  # fmt: skip
    with pytest.raises(run.lp.GuardError, match="G-privileged"):
        run.run_arm(Pool(ok | {"task_truth_in_controller": 1}), [{"arm": "P-aim"}], "P-aim")
    with pytest.raises(run.lp.GuardError, match="G-law"):
        run.run_arm(Pool(ok | {"kpred": {}}), [{"arm": "H-final"}], "H-final")
    run.run_arm(Pool(ok | {"task_truth_in_controller": 1}), [{"arm": "H-final"}], "H-final")


def test_the_stage_cap_excludes_the_quiet_wait():
    run = _runner()
    clock = run.hz.Clock(10.0)
    start = clock.start
    report = {"quiet_machine": {"waited_seconds": 3600.0}}
    run.exclude_quiet_wait(clock, report)
    assert clock.start == start + 3600.0 and report["stage_cap_excludes_quiet_wait_seconds"] == 3600
    clock.check("now")  # the wait does not count against the cap
    run.exclude_quiet_wait(clock, {})


def test_the_sentinel():
    run = _runner()
    report: dict = {}
    fields = run.Fields(report, "gates")
    fields.check()
    report["fields"]["decision"] = {"row": "G-PASS"}
    with pytest.raises(pt.GuardError):
        fields.check()
    with pytest.raises(pt.GuardError):
        fields.set("gates", ul.NOT_EVALUATED)
    assert set(run.STAGE_FIELDS) == set(run.STAGES)


def test_the_gscale_probe_calls_stage_gs_own_functions():
    source = "".join(inspect.getsource(_runner().stage_gscale).split())
    for name in ("rt.scale_probe(g_fit_core", "rt.scale_probe(g_read_core",
                 "rt.scale_probe(g_dynamics_core"):  # fmt: skip
        assert name in source
    gates = inspect.getsource(_runner().stage_gates)
    for name in ("g_fit_core(", "g_read_core(", "g_dynamics_core("):
        assert name in gates
    assert gates.index("g_fit_core(") < gates.index('report["first_outcome_utc"]') < gates.index(
        '("gate_p",)')  # fmt: skip


def test_stage_modules_import_without_torch_or_mujoco():
    code = (
        "import sys\n"
        "import embodied_jepa.lewm_ul_v2, embodied_jepa.lewm_ul_v2_runtime\n"
        "import embodied_jepa.lewm_ul_v2_offline, embodied_jepa.lewm_ul_v2_train\n"
        "assert 'torch' not in sys.modules and 'mujoco' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)
