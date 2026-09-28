"""TASK-071 preregistration: the frozen v2 design, its manifest, run-time pieces and runner guards.

NumPy tests run in the core job; torch tests import-skip without torch; simulation tests are
graphics opt-in (``JEPA_TEST_RENDER=1``), as the other simulator tests.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_jepa import first_policy as fp
from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa import first_policy_v2_runtime as rt2
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-first-policy-v2.json"
BOUNDS = (
    np.array((0.0,) * 6 + (-0.5,) * 6 + (-1.0, -1.0), np.float32),
    np.array((0.0,) * 6 + (0.5,) * 6 + (-1.0, 1.0), np.float32),
)
render = pytest.mark.skipif(os.environ.get("JEPA_TEST_RENDER") != "1", reason="graphics opt-in")
TRUTH = {
    "position_frame": "world",
    "base_position_world": np.array([0.0, 0.0, 0.79]),
    "base_rotation_world": np.eye(3),
    "object_position": np.array([0.34, -0.18, 0.77]),
    "plate_position": np.array([0.49, -0.09, 0.746]),
    "container_surface_z": 0.748,
    "object_support_height": 0.027,
}


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "_run_first_policy_v2", ROOT / "scripts" / "run_first_policy_v2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----- the manifest and the carried pieces -----
def test_manifest_frozen_block_equals_the_module():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["frozen"] == fp2.frozen_block()
    assert manifest["results"] is None and manifest["calibration"] is None
    assert manifest["exemption_spent_cited"]["value_at_preregistration"] is False
    assert (
        manifest["encoder_digests"]
        == json.loads(
            (ROOT / "benchmarks" / "manifests" / "apple-first-policy-v1.json").read_text()
        )["encoder_digests"]
    )
    for relative in manifest["hashes"]:
        assert (ROOT / relative).is_file(), relative


def test_task_and_expert_are_task070s():
    from embodied_jepa import apple_to_plate_v2 as v2
    from embodied_jepa import resting_expert as rx
    from embodied_jepa.at_rest import AtRestThresholds

    assert fp2.EXPERT == v2.GATE_EXPERT
    assert fp2.SCENE_VERSION == v2.SCENE_VERSION
    assert fp2.APPLE_CONTACT["condim"] == v2.V2_APPLE_CONDIM
    assert tuple(fp2.APPLE_CONTACT["friction"]) == v2.V2_APPLE_FRICTION
    assert fp2.SUCCESS_METRIC == AtRestThresholds().version
    assert fp2.SETTLE_STEPS == AtRestThresholds().settle_steps
    assert fp2.MAX_POLICY_STEPS == rx.EXPERT_BUDGET
    assert fp2.ATTEMPT_STEPS == fp.MAX_POLICY_STEPS == 800


def test_schedule_is_e9s():
    expert = rt2.make_expert(TRUTH)
    assert tuple(p.name for p in expert.phases) == fp2.EXPERT_PHASES
    assert tuple(p.commands for p in expert.phases) == fp2.EXPERT_BUDGETS
    assert expert.max_steps == fp2.EXPERT_POLICY_STEPS == 725
    assert [fp2.scheduled_phase(s) for s in (0, 129, 130, 254, 255, 404, 405, 724, 799)] == [
        0,
        0,
        1,
        2,
        3,
        3,
        4,
        9,
        9,
    ]
    with pytest.raises(ContractError):
        fp2.scheduled_phase(-1)
    assert fp2.PHASE_CLOSE == fp.PHASES.index("close") == 2  # the pick phases are v1's


def test_clock_is_rescaled_to_e9_and_the_input_is_132():
    assert fp2.clock_features(725)[0] == 1.0 and fp.clock_features(745)[0] == 1.0
    assert np.array_equal(fp2.clock_features(10)[1:], fp.clock_features(10)[1:])
    x = rt2.input_vector(np.zeros(4), 725, np.zeros(86), np.zeros(9))
    assert x.shape == (132,) and x[4] == 1.0


# ----- seeds -----
def test_seed_ranges_are_fresh_disjoint_and_off_every_spent_range():
    fp2.check_seed_ranges()
    seeds = [s for name in fp2.SEED_RANGES for s in fp2.seeds_of(name)]
    assert len(seeds) == len(set(seeds)) == 1000
    assert min(seeds) == 51000 and max(seeds) == 51999
    everything = set(seeds) | set(fp2.COHORT_D2) | set(range(52100, 52200))
    for low, high in ((46000, 46999), (47000, 47199), (50000, 50299), (50600, 50631)):
        assert not everything & set(range(low, high + 1))
    assert not everything & set(fp.COHORT_C) and not everything & set(fp.COHORT_D)
    assert len(fp2.COHORT_D2) == 16 and fp2.D_RESETS == 16
    assert len(fp2.seeds_of("corpus")) == 200 and len(fp2.seeds_of("perception_heldout")) == 128


def test_seed_check_refuses_overlaps_and_leaving_the_block(monkeypatch):
    monkeypatch.setitem(fp2.SEED_RANGES, "dagger_2", (51700, 51871))
    with pytest.raises(fp2.GuardError, match="overlap"):
        fp2.check_seed_ranges()
    monkeypatch.setitem(fp2.SEED_RANGES, "dagger_2", (51744, 51871))
    monkeypatch.setitem(fp2.SEED_RANGES, "corpus", (50900, 50999))
    with pytest.raises(fp2.GuardError):
        fp2.check_seed_ranges()


def test_runner_seed_guard_refuses_cohort_c_and_foreign_seeds():
    runner = load_runner()
    runner.check_simulated_seeds("D", fp2.COHORT_D2, smoke=False)
    runner.check_simulated_seeds("corpus", fp2.seeds_of("corpus"), smoke=False)
    with pytest.raises(fp2.GuardError, match="cohort C"):
        runner.check_simulated_seeds("D", fp.COHORT_C[:2], smoke=False)
    with pytest.raises(fp2.GuardError):
        runner.check_simulated_seeds("D", fp.COHORT_D, smoke=False)  # v1's D is not D2
    with pytest.raises(fp2.GuardError):
        runner.check_simulated_seeds("dagger_1", fp2.seeds_of("dagger_2")[:3], smoke=False)
    with pytest.raises(fp2.GuardError):
        runner.check_simulated_seeds("D", fp2.COHORT_D2, smoke=True)  # smoke uses smoke seeds


# ----- the corpus plan -----
def test_corpus_plan_splits_whole_resets_and_cycles_noise():
    plan = fp2.corpus_plan()
    assert [r["seed"] for r in plan] == list(fp2.seeds_of("corpus"))
    counts = {k: sum(r["split"] == k for r in plan) for k in ("train", "val", "test")}
    assert counts == fp2.EXPECTED_CORPUS_SPLITS
    assert [r["noise_level"] for r in plan[:6]] == [0, 1, 2, 3, 0, 1]
    assert len({r["session_id"] for r in plan}) == 200  # one session per reset
    assert fp2.corpus_plan() == plan  # fixed before any episode


# ----- C0 bars with the TASK-070 ceiling -----
def _levels(levels, value):
    return {level: value for level in levels}


def test_task070_ceiling_is_the_v1_rule_on_task070_counts():
    assert fp2.task070_plate_ceiling() == fp2.TASK070_PLATE_CEILING_CM == 1.5


def test_c0_bars_can_only_tighten():
    full = fp2.c0_bars(32, _levels(fp2.C0_APPLE_LEVELS_CM, 32), _levels(fp2.C0_PLATE_LEVELS_CM, 32))
    assert full["plate_p90_cm"] == 1.5 and full["plate_median_cm"] == 1.5  # v1 would give 2.5
    assert full["apple_p90_cm"] == 1.2 and full["apple_median_cm"] == 0.75
    tight = fp2.c0_bars(
        32, _levels(fp2.C0_APPLE_LEVELS_CM, 32), {1.0: 30, 1.5: 27, 2.0: 32, 2.5: 32}
    )
    assert tight["plate_p90_cm"] == 1.0 and tight["plate_median_cm"] == 1.0
    v1 = fp.c0_bars(32, _levels(fp2.C0_APPLE_LEVELS_CM, 30), {1.0: 30, 1.5: 30, 2.0: 20, 2.5: 0})
    v2 = fp2.c0_bars(32, _levels(fp2.C0_APPLE_LEVELS_CM, 30), {1.0: 30, 1.5: 30, 2.0: 20, 2.5: 0})
    for key in ("apple_median_cm", "apple_p90_cm", "plate_median_cm", "plate_p90_cm"):
        assert v2[key] <= v1[key]
    assert "escalate" in fp2.c0_bars(
        32, _levels(fp2.C0_APPLE_LEVELS_CM, 32), {1.0: 27, 1.5: 32, 2.0: 32, 2.5: 32}
    )
    assert "escalate" in fp2.c0_bars(
        27, _levels(fp2.C0_APPLE_LEVELS_CM, 32), _levels(fp2.C0_PLATE_LEVELS_CM, 32)
    )


# ----- M1 rows -----
def _counts(**overrides):
    counts = {arm: {"grasp": 0, "success": 0} for arm in fp2.M1_ARMS}
    counts["B-oracle"] = {"grasp": 16, "success": 16}
    for arm, (grasp, success) in overrides.items():
        counts[arm.replace("_", "-")] = {"grasp": grasp, "success": success}
    return counts


def test_decide_m1_rows_in_both_directions():
    assert fp2.decide_m1(_counts(B_oracle=(16, 13)), 3)["row"] == "V"
    assert fp2.decide_m1(_counts(B_hold=(1, 0)), 3)["row"] == "V"
    assert fp2.decide_m1(_counts(B_random=(1, 0)), 3)["row"] == "V"
    passed = fp2.decide_m1(_counts(P_1=(4, 1), P_3=(4, 1)), 3)
    assert passed["row"] == "M1-PASS" and passed["carried"] == "P-3"
    assert fp2.decide_m1(_counts(C_3=(9, 9)), 3)["row"] == "M1-PERCEPTION"  # C-3 cannot pass
    motor = fp2.decide_m1(_counts(A4_look=(9, 3)), 3)
    assert motor["row"] == "M1-MOTOR" and motor["run_F"]
    f_ok = fp2.decide_m1(_counts(A4_look=(9, 3)), 3, f_counts={"grasp": 2, "success": 1})
    assert f_ok["row"] == "M1-MOTOR-F-PARTIAL" and not f_ok["clause_fires"]
    f_none = fp2.decide_m1(_counts(A4_look=(9, 3)), 3, f_counts={"grasp": 2, "success": 0})
    assert f_none["row"] == "M1-MOTOR-F-NONE" and f_none["clause_fires"]
    perception = fp2.decide_m1(_counts(A4_look=(9, 2)), 3)
    assert perception["row"] == "M1-PERCEPTION" and perception["clause_fires"]
    with pytest.raises(ContractError):
        fp2.decide_m1(_counts(A4_look=(9, 2)), 0)
    with pytest.raises(ContractError):
        fp2.decide_m1(_counts(P_0=(17, 0)), 3)
    # T71-R1: a counted success needs the latched grasp, so success <= grasp is checked (v1's)
    with pytest.raises(ContractError):
        fp2.decide_m1(_counts(P_0=(0, 1)), 3)
    with pytest.raises(ContractError):
        fp2.decide_m1(_counts(A4_look=(9, 3)), 3, f_counts={"grasp": 0, "success": 1})


def test_counted_success_needs_at_rest_and_a_grasp_t71_r1():
    assert fp2.counted_success(True, True)
    assert not fp2.counted_success(True, False)  # a push onto the plate is not counted
    assert not fp2.counted_success(False, True)
    assert "T71-R1" in fp2.frozen_block()["counted_success"]


class _Truth:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.05
        return {
            "timestamp": self.t,
            "container_surface_z": 0.748,
            "object_support_height": 0.027,
            "object_position": np.array([0.5, -0.1, 0.775]),
            "plate_position": np.array([0.5, -0.1, 0.746]),
            "object_velocity": np.zeros(3),
            "hand_contact": False,
        }


class _Robot:
    """A stand-in robot whose apple always rests on the plate centre."""

    def __init__(self):
        self.sim = SimpleNamespace(task_truth=_Truth())

    def observe(self):
        return SimpleNamespace(state=SimpleNamespace(values=np.zeros((1, 86))))

    def project_candidates(self, command):
        return SimpleNamespace(feasible=np.ones((1, 1), bool), actions=command)

    def execute(self, action):
        return SimpleNamespace(applied_action=np.asarray(action), reason="", status="applied")

    def stop(self, reason):
        self.stopped = reason


class _Scorer:
    def __init__(self, robot, grasp_from):
        self.robot, self.calls, self.grasp_from = robot, 0, grasp_from

    def evaluate(self):
        self.robot.sim.task_truth()
        self.calls += 1
        return {"grasp": self.grasp_from is not None and self.calls > self.grasp_from}


class _Hold:
    def act(self, observation, step):
        return np.zeros(14, np.float32)

    def advance(self, result):
        return None


@pytest.mark.parametrize(("grasp_from", "counted"), [(2, True), (None, False), (12, False)])
def test_attempt_counts_success_only_with_a_grasp_before_the_settle(grasp_from, counted):
    robot = _Robot()
    counter = rt2.PrivilegedReadCounter(robot.sim)
    record = rt2.run_attempt(
        robot, _Scorer(robot, grasp_from), counter, _Hold(), bounds=BOUNDS, max_steps=10
    )
    assert record["at_rest"]  # the stand-in apple rests on the plate in every case
    assert record["success"] is counted
    assert record["grasp_before_settle"] is counted  # 12 latches only inside the settle
    assert rt2.privileged_reads_ok(record) and record["settle_steps"] == 60


def test_ladder_labels_the_settle_and_keeps_v1s_allowances():
    assert set(fp.LADDER_ALLOWANCES_L1) < set(fp2.LADDER_ALLOWANCES_L1)
    assert "settle" in fp2.LADDER_ALLOWANCES_L1["f"]
    assert fp2.ARMS["F"][0] == "L2" and fp2.ARMS["A4-look"][0] == "L3"
    assert fp2.ARMS["B-replay"][0] == "L4" and fp2.ARMS["B-oracle"][0] == "L4"
    assert fp2.LEARNED_ARMS == ("P-0", "P-1", "P-2", "P-3", "C-3", "R-3")


# ----- run-time pieces (NumPy) -----
def _episode(t=800, *, dropped_at=None, drift_at=None):
    phase = np.array(
        [fp2.scheduled_phase(s) for s in range(fp2.EXPERT_POLICY_STEPS)]
        + [fp2.SETTLE_PHASE] * (t - fp2.EXPERT_POLICY_STEPS),
        np.int8,
    )[:t]
    apple = np.zeros((t + 1, 3), np.float32)
    dropped = np.zeros(t + 1, bool)
    if drift_at is not None:
        apple[drift_at:] = [0.02, 0.0, 0.0]
    if dropped_at is not None:
        dropped[dropped_at:] = True
    base = np.zeros((t, 14), np.float32)
    base[:, 6] = 0.9  # clipped to +0.5 by the bounds
    return {
        "frames": np.zeros((t + 1, 2, 2, 3), np.uint8),
        "states": np.zeros((t + 1, 86)),
        "base": base,
        "requested": base.copy(),
        "applied": np.tile(np.arange(t, dtype=np.float32)[:, None], (1, 14)),
        "phase": phase,
        "apple": apple,
        "dropped": dropped,
        "hand_contact": np.zeros(t + 1, bool),
        "latched": np.zeros(t, bool),
    }


def test_bc_rows_keep_policy_steps_and_apply_v1s_mask():
    rows = rt2.bc_rows(_episode(), BOUNDS)
    assert rows["accounting"]["policy_steps"] == 725 == rows["accounting"]["kept"]
    assert rows["steps"][-1] == 724 and np.all(rows["labels"][:, 0] == 0.5)
    drift = rt2.bc_rows(_episode(drift_at=100), BOUNDS)  # drift before close (step 210) only
    assert drift["accounting"]["post_displacement_pre_grasp"] == 110
    dropped = rt2.bc_rows(_episode(dropped_at=500), BOUNDS)
    assert dropped["accounting"]["dropped"] == 225
    short = rt2.bc_rows(_episode(t=300), BOUNDS)  # an attempt stopped early: its rows only
    assert short["accounting"]["policy_steps"] == 300


def test_replay_excludes_the_settle():
    actions = rt2.replay_actions(_episode())
    assert actions.shape == (725, 14) and actions[-1, 0] == 724


def test_corpus_store_seals_refuses_test_and_tampering(tmp_path):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    plan = [
        {"episode_id": "a", "split": "train"},
        {"episode_id": "b", "split": "val"},
        {"episode_id": "c", "split": "test"},
    ]
    hashes = {}
    for root in plan:
        hashes[root["episode_id"]] = rt2.write_episode(
            episodes, root["episode_id"], _episode(t=730), {"split": root["split"]}
        )
    with pytest.raises(FileExistsError):
        rt2.write_episode(episodes, "a", _episode(t=730), {})
    sha = rt2.seal(tmp_path, plan, hashes, {"smoke": True})
    reader = rt2.CorpusReader(tmp_path, sha)
    arrays, meta = reader.episode("a", keys=("phase",))
    assert set(arrays) == {"phase", "frame0"} and meta["split"] == "train"
    with pytest.raises(fp2.GuardError, match="Q-split"):
        reader.episode("c")
    assert not reader.test_split_decoded and reader.decoded == {"a"}
    with pytest.raises(fp2.GuardError):
        rt2.CorpusReader(tmp_path, "0" * 64)
    (episodes / "b.json").write_text("{}\n")
    with pytest.raises(fp2.GuardError, match="sealed"):
        reader.episode("b")


def test_privileged_reads_count_both_harness_reads():
    ok = {
        "task_truth_in_controller": 0,
        "task_truth_total": 20,
        "scorer_evaluations": 10,
        "at_rest_records": 10,
    }
    assert rt2.privileged_reads_ok(ok)
    assert not rt2.privileged_reads_ok(ok | {"task_truth_in_controller": 1})
    assert not rt2.privileged_reads_ok(ok | {"task_truth_total": 21})


class _Done:
    done = True
    max_steps = fp2.EXPERT_POLICY_STEPS

    def action(self, robot):  # pragma: no cover - must never be reached once done
        raise AssertionError("an exhausted expert was asked for a command")


def test_exhausted_expert_ends_policy_steps_and_labels(monkeypatch):
    controller = rt2.ExpertController.__new__(rt2.ExpertController)
    controller.expert, controller.robot = _Done(), None
    with pytest.raises(ContractError, match=rt2.EXHAUSTED_MESSAGE):
        controller.act(None, 0)
    monkeypatch.setattr(rt2, "make_expert", lambda truth: _Done())
    shadow = rt2.ShadowExpert(TRUTH)
    assert shadow.command(None, 7) is None and shadow.exhausted_at_step == 7


def test_settle_is_arm_still_hands_open():
    command = rt2.settle_command(BOUNDS)
    assert np.all(command[:12] == 0.0) and command[12] == -1.0 and command[13] == -1.0


# ----- torch -----
def test_policy_net_heads_follow_e9_phases():
    torch = pytest.importorskip("torch")
    from embodied_jepa import first_policy_v2_model as fm2

    f = fm2.PolicyNet(len(fp2.EXPERT_PHASES))
    with pytest.raises(ContractError):
        f(torch.zeros(2, 132))
    assert f(torch.zeros(2, 132), torch.as_tensor([0, 9])).shape == (2, 7)
    with pytest.raises(ContractError):
        fm2.PolicyNet(8)
    assert list(fm2.phases_of([0, 130, 724, 739])) == [0, 1, 9, 9]
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(64, 132)), rng.uniform(-0.4, 0.4, (64, 7))
    trained = fm2.train(
        x, y, np.arange(64), x[:8], y[:8], np.arange(8), device="cpu", updates=20, select_every=10
    )
    assert trained["record"]["updates"] == 20 and len(trained["record"]["curve"]) == 2
    steps = np.linspace(0, 739, 64).astype(int)  # every e9 phase is present
    f = fm2.train(
        x,
        y,
        steps,
        x[:8],
        y[:8],
        steps[:8],
        heads=10,
        device="cpu",
        updates=20,
        select_every=10,
        min_output_std=0.0,
    )
    model = fm2.load(f["state"], heads=10)
    assert fm2.batch_predict(model, x[:3], [0, 400, 739]).shape == (3, 7)


# ----- simulation (graphics opt-in) -----
@render
def test_v2_attempt_reproduces_task070s_harness_for_e9():
    pytest.importorskip("mujoco")
    from embodied_jepa import resting_expert as rx

    runner = load_runner()
    seed = 52198
    r = runner.load_wide_reset()(seed)
    reset = {
        "object_xy": list(map(float, r["object_xy"])),
        "plate_xy": list(map(float, r["plate_xy"])),
    }
    robot = rt2.make_robot()
    try:
        summary, arrays = rx.run_attempt(
            robot,
            BOUNDS,
            seed=seed,
            reset=reset,
            plate_offset=[0.0, 0.0],
            make_expert=lambda truth: rx.RestingPlaceExpert(truth, **fp2.EXPERT),
        )
        truth, scorer, counter, _obs, _facts = rt2.reset_and_look(robot, seed, reset)
        record = rt2.run_attempt(
            robot, scorer, counter, rt2.ExpertController(truth, robot), bounds=BOUNDS
        )
    finally:
        robot.close()
    policy = arrays["phase"] < len(fp2.EXPERT_PHASES)
    assert np.array_equal(arrays["command"][policy][:, list(fp2.FREE_INDICES)], record["commands"])
    assert record["at_rest_detail"] == summary["at_rest_detail"]
    assert record["latched_success"] == summary["latched_success"]
    assert record["executed_steps"] == 725 and record["settle_steps"] == 60
    assert record["success"] == (summary["at_rest"] and record["grasp_before_settle"])
    assert rt2.privileged_reads_ok(record) and record["task_truth_total"] == 2 * 785


@render
def test_collector_at_noise_zero_is_e9_and_frame0_is_post_look(tmp_path):
    pytest.importorskip("mujoco")
    runner = load_runner()
    runner.worker_init()
    try:
        seed = 52199
        r = runner.load_wide_reset()(seed)
        reset = {
            "object_xy": list(map(float, r["object_xy"])),
            "plate_xy": list(map(float, r["plate_xy"])),
        }
        root = {
            "episode_id": f"look2-{seed}",
            "session_id": f"look2-reset-{seed}",
            "split": "train",
            "noise_level": 0,
            "noise_seed": 1,
        }
        out = runner.run_task(
            {"seed": seed, "reset": reset, "kind": "collect", "folder": str(tmp_path)} | root
        )
        oracle = runner.run_task({"seed": seed, "reset": reset, "kind": "oracle"})
    finally:
        runner._W["robot"].close()
        runner._W["fk"].close()
    with np.load(tmp_path / f"look2-{seed}.npz") as data:
        requested, phase, frame0 = data["requested"], data["phase"], data["frames"][0]
    policy = phase < fp2.SETTLE_PHASE
    assert np.array_equal(
        requested[policy][:, list(fp2.FREE_INDICES)], np.asarray(oracle["commands"], np.float32)
    )
    assert runner.frame_sha(frame0) == out["post_look_frame_sha256"]
    assert out["at_rest"] == oracle["at_rest"] and out["termination"] == "policy_complete"
