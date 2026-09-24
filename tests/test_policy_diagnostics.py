"""TASK-057 diagnostic harness and amendment 1 of ``apple_policy_diagnostics_v1``.

Every guard here was written by injecting its violation first and watching the test fail
(``apple_policy_v1_results.md`` §9's standard); where the injection is cheap to keep, the test
performs it itself so the guard is exercised in both directions on every run.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_jepa import policy_diagnostics as pd
from embodied_jepa.contracts import ContractError
from embodied_jepa.scripted import (
    EarlyReleaseOracleManipulationPolicy,
    OracleManipulationPolicy,
    apple_collector_policy,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-policy-diagnostics-v1.json"
PROTOCOL = ROOT / "docs" / "experiments" / "apple_policy_diagnostics_v1.md"
TRUTH = {
    "position_frame": "world",
    "base_position_world": [0.0, 0.0, 0.0],
    "base_rotation_world": np.eye(3),
    "object_position": [0.34, -0.18, 0.77],
    "plate_position": [0.49, -0.09, 0.75],
    "container_surface_z": 0.76,
    "object_support_height": 0.04,
}


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest():
    return json.loads(MANIFEST.read_text())


# ----- the shadow expert is the collector ----------------------------------------------------
def test_the_shadow_expert_is_the_policy_the_corpus_was_collected_with():
    """Amendment 1's defect: the frozen text named plain OracleManipulationPolicy (805 commands).

    The collector (scripts/collect_apple_wide.py run_root) builds apple_collector_policy, whose
    budget is ROOT_MAX_COMMANDS = 745. Pinned against the collector's own constant and the
    collector's own factory, not against a transcribed number.
    """
    shadow = pd.ShadowExpert(TRUTH)
    assert isinstance(shadow.policy, EarlyReleaseOracleManipulationPolicy)
    reference = apple_collector_policy(TRUTH)
    assert [p.name for p in shadow.policy.phases] == [p.name for p in reference.phases]
    for mine, theirs in zip(shadow.policy.phases, reference.phases, strict=True):
        assert np.array_equal(mine.target_base, theirs.target_base)
        assert mine.commands == theirs.commands and mine.grasp == theirs.grasp
    collector = _load("_collect_apple_wide", "scripts/collect_apple_wide.py")
    assert shadow.budget == pd.SHADOW_EXPERT_BUDGET == collector.ROOT_MAX_COMMANDS == 745
    assert OracleManipulationPolicy(TRUTH).max_steps == 805, "the frozen text's expert"
    # The collector script really does construct this policy (the provenance, not a copy).
    source = (ROOT / "scripts" / "collect_apple_wide.py").read_text()
    assert "policy = apple_collector_policy(sim.task_truth())" in source


def test_the_shadow_expert_guard_refuses_the_frozen_texts_expert(monkeypatch):
    """Injected violation: swap the default factory for plain OracleManipulationPolicy."""
    monkeypatch.setattr(pd, "collector_shadow_policy", OracleManipulationPolicy)
    with pytest.raises(ContractError, match="not the preregistered"):
        pd.ShadowExpert(TRUTH)


def test_the_manifest_and_protocol_name_the_collector_and_keep_the_frozen_text():
    m = manifest()
    definition = m["definitions_every_scope_term_used_in_a_gate_or_stop_rule"][
        "shadow_expert_command"
    ]["definition"]
    assert definition.startswith("scripted.apple_collector_policy(sim.task_truth())")
    superseded = {s["path"]: s for s in m["amendment_1"]["superseded"]}
    frozen = superseded[
        "/definitions_every_scope_term_used_in_a_gate_or_stop_rule/shadow_expert_command/definition"
    ]["frozen_value"]
    assert frozen.startswith("OracleManipulationPolicy(sim.task_truth())"), (
        "the frozen value must be kept verbatim, not silently rewritten"
    )
    assert all(s["reason"] for s in m["amendment_1"]["superseded"])
    assert m["precedence_rule_D1_over_G_SUB"]["exemption_spent"] is False
    text = PROTOCOL.read_text()
    assert "## 13. Amendment 1" in text
    # The frozen sentence is still there, and it is marked.
    assert "It is exhausted after 805 commands." in text
    assert text.count("Superseded by Amendment 1") >= 10
    # No live sentence may still call 805 the shadow expert's budget in the manifest's
    # effective definitions.
    se = m["definitions_every_scope_term_used_in_a_gate_or_stop_rule"]["shadow_expert_command"]
    assert "it_is_exhausted_after_805_commands" not in se
    assert "745" in se["it_is_exhausted_after_745_commands"]


def test_the_phase_table_labels_are_the_collectors():
    """Table E's generating script named phases 5/6 'lower'/'release' (plain oracle's names)."""
    measure = _load("_measure", "scripts/measure_policy_offline_conditionals.py")
    assert measure.PHASES == tuple(p.name for p in apple_collector_policy(TRUTH).phases)
    assert measure.PHASES == pd.SHADOW_EXPERT_PHASES
    correction = manifest()["frozen_offline_conditionals_and_phase_table"][
        "phase_label_correction_amendment_1"
    ]
    assert correction["lower"] == measure.PHASES[5]
    assert correction["release"] == measure.PHASES[6]


# ----- thresholds re-derived from the calibration --------------------------------------------
def _at_least(k, n, p):
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def test_g_sub_thresholds_follow_the_frozen_rule_against_the_blind_reference():
    """tau_c = max(8, floor((r_c + 16) / 2)), r_c = max(1, blind time-indexed score of c)."""
    gate = manifest()["gates"]["G_SUB"]
    blind = gate["blind_reference_time_indexed_complement"]
    assert set(blind) == set(gate["candidates"]) == set(pd.G_SUB_CANDIDATES)
    family = 1.0
    for candidate, b in blind.items():
        reference = max(1, b)
        expected = max(8, (reference + 16) // 2)
        assert gate["per_candidate_threshold"][candidate] == expected, candidate
        # The threshold must discriminate: strictly above what a clock-only controller scores
        # with the same substitution, and never below the frozen 8/16.
        assert expected > b and expected >= 8
        p = _at_least(expected, 16, reference / 16)
        assert gate["one_sided_binomial_at_each_candidates_reference"][candidate] == (
            pytest.approx(p, rel=1e-9)
        )
        family *= 1 - p
    assert gate["family_wise_over_8_candidates_at_their_references"] == pytest.approx(
        1 - family, rel=1e-9
    )
    # The frozen 8/16 would NOT have discriminated for dy. Recorded, and checked.
    assert blind["dy"] >= 8, "the calibration finding that motivated the amendment"


def test_the_d1_grasp_cut_fires_on_the_prior_only_predictor_and_not_on_sound_arms():
    m = manifest()
    checks = m["amendment_1"]["blind_predictor_and_sound_pipeline_checks"]["D1_grasp"]
    prior = checks["prior_only_mean_grasp"]

    def amended(g):
        return g > -0.5

    def frozen(g):
        return abs(g - (-1.0)) >= 1.0

    assert amended(prior) and not frozen(prior), "the defect the amendment fixes"
    assert amended(0.0) and amended(1.0) and not amended(-1.0) and not amended(-0.5)
    for ranges in checks["sound_arms_step_zero_grasp_range"].values():
        for low, high in ranges.values():
            assert not amended(low) and not amended(high)
    assert "> -0.5" in m["gates"]["D1_grasp"]["derived_two_sided_count"]


def test_d1_pipeline_tolerance_sits_between_noise_and_the_smallest_real_change():
    m = manifest()
    gate = m["gates"]["D1"]
    calibration = m["amendment_1"]["blind_predictor_and_sound_pipeline_checks"]["D1_pipeline"]
    tolerance = gate["output_tolerance"]
    noise = max(a["float_noise_max_abs_single_vs_batched"] for a in calibration.values())
    image_arms = [a for name, a in calibration.items() if not name.startswith("A0")]
    smallest = min(a["swapped_reset_image_max_abs_change"] for a in image_arms)
    assert tolerance >= 10 * noise, "a sound pipeline must pass"
    assert tolerance <= smallest / 10, "a swapped reset image must fail"
    assert gate["state_tolerance"] <= 1e-6 and gate["device"] == "cpu"
    assert len(gate["val_root_seeds"]) == 15
    assert not set(gate["val_root_seeds"]) & set(range(45300, 45340))
    # The frozen D1 failed on recorded frames for every arm: it is reported, never decisional.
    frozen = m["amendment_1"]["blind_predictor_and_sound_pipeline_checks"]["D1_frozen"]
    assert not any(v for split in frozen["frozen_d1_would_pass"].values() for v in split.values())
    assert "D1_contrast_reported_not_decisional" in m["gates"]


def test_the_d2_departure_definition_is_calibrated_to_discriminate():
    calibration = manifest()["gates"]["D2"]["persistence_calibration"]
    sound = calibration["open_loop_val_median_first_departure_persistence_5"]
    blind = calibration["prior_only_mean_median_first_departure_persistence_5"]
    assert min(sound.values()) >= 50, "a sound arm must be able to read 'gradual'"
    assert max(blind.values()) <= 5, "the prior-only predictor must read 'immediate'"
    assert pd.DEPARTURE_PERSISTENCE == 5


# ----- harness pieces ------------------------------------------------------------------------
def test_first_departure_step_needs_persistence_and_stops_at_exhaustion():
    tau = 0.1
    assert pd.first_departure_step([0.2, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2], tau, 5) == 2
    assert pd.first_departure_step([0.2] * 4 + [0.0] * 10, tau, 5) is None
    # Exhaustion ends the search: undefined is never read as zero or as a departure.
    assert pd.first_departure_step([0.2, 0.2, None, 0.2, 0.2, 0.2, 0.2, 0.2], tau, 5) is None
    assert pd.first_departure_step([0.2] * 5, tau, 1) == 0
    with pytest.raises(ContractError):
        pd.first_departure_step([0.2], 0.0, 5)


def test_substitution_replaces_exactly_the_named_dimensions():
    controller = np.zeros(14, np.float32)
    controller[12] = -1.0
    controller[list(pd.FREE_INDICES)] = np.arange(1, 8, dtype=np.float32) / 10
    expert = -np.arange(1, 8, dtype=np.float32) / 10
    for name, dims in pd.CONFIGURATIONS.items():
        out = pd.substitute(controller, None if not dims else expert, name)
        for i, dim in enumerate(pd.FREE_NAMES):
            want = expert[i] if dim in dims else controller[pd.FREE_INDICES[i]]
            assert out[pd.FREE_INDICES[i]] == pytest.approx(want), (name, dim)
        assert out[12] == -1.0 and not out[:6].any()
    assert set(pd.G_SUB_CANDIDATES) & set(pd.CONTROLS) == set()
    with pytest.raises(ContractError):
        pd.substitute(controller, None, "dx")
    with pytest.raises(ContractError):
        pd.substitute(controller, expert, "everything")


class _ScriptedStub:
    def __init__(self, budget, message="scripted policy has finished"):
        self.max_steps, self.message, self.step_count = budget, message, 0
        self.phases = ()

    @property
    def done(self):
        return self.step_count >= self.max_steps

    @property
    def phase(self):
        return "stub"

    @property
    def phase_index(self):
        return 0

    def action(self, robot):  # noqa: ARG002
        if self.done:
            raise ContractError(self.message)
        action = np.zeros(14, np.float32)
        action[6:9] = 0.3
        action[12:] = [-1.0, 1.0]
        return action

    def advance(self, result):  # noqa: ARG002
        self.step_count += 1


def test_exhaustion_is_caught_by_its_exact_message_only():
    shadow = pd.ShadowExpert(TRUTH, factory=lambda truth: _ScriptedStub(1))
    assert shadow.command(None, 0) is not None
    shadow.advance(SimpleNamespace(applied_action=np.zeros(14)))
    assert shadow.command(None, 1) is None and shadow.exhausted_at_step == 1
    other = pd.ShadowExpert(TRUTH, factory=lambda truth: _ScriptedStub(0, "something else"))
    with pytest.raises(ContractError, match="something else"):
        other.command(None, 0)


class _FakeRobot:
    """Just enough embodiment for the loop: every command is feasible and executed."""

    def __init__(self, log):
        self.log, self.stopped = log, None

    def observe(self):
        self.log.append("observe")
        return SimpleNamespace(images={}, state=None)

    def project_candidates(self, command):
        return SimpleNamespace(feasible=np.ones((1, 1), bool), actions=command)

    def execute(self, action):
        self.log.append("execute")
        return SimpleNamespace(applied_action=np.asarray(action).copy(), reason=None)

    def stop(self, reason):
        self.stopped = reason


class _LoggingController:
    name = "logging"

    def __init__(self, log):
        self.log = log

    def act(self, observation):  # noqa: ARG002
        self.log.append("act")
        action = np.zeros(14, np.float32)
        action[12:] = -1.0
        return action

    def advance(self, result):  # noqa: ARG002
        return None


class _LoggingShadow(pd.ShadowExpert):
    def __init__(self, log, budget):
        super().__init__(TRUTH, factory=lambda truth: _ScriptedStub(budget))
        self.log = log

    def command(self, robot, step):
        self.log.append("shadow")
        return super().command(robot, step)


def _clip(action, lower, upper):
    """Stand-in for evaluate_policy.clip_to_configured_bounds, which imports torch (policy.py);
    the loop takes its clip as an argument so the core CI job can exercise it."""
    clipped = np.clip(action, lower, upper).astype(np.float32)
    return clipped, bool((clipped != action).any()), (clipped != action)[list(pd.FREE_INDICES)]


def _run(configuration, budget, max_steps=10):
    log = []
    lower = np.array((0.0,) * 6 + (-0.5,) * 6 + (-1.0, -1.0), np.float32)
    upper = np.array((0.0,) * 6 + (0.5,) * 6 + (-1.0, 1.0), np.float32)
    robot = _FakeRobot(log)
    scorer = SimpleNamespace(evaluate=lambda: {"success": False, "grasp": False})
    result = pd.run_diagnostic_attempt(
        _LoggingController(log),
        robot,
        scorer,
        configuration=configuration,
        clip=_clip,
        lower=lower,
        upper=upper,
        max_steps=max_steps,
        deadline_seconds=5.0,
        shadow=_LoggingShadow(log, budget),
        trace=False,
    )
    return result, log, robot


def test_the_shadow_expert_is_computed_strictly_after_act_every_step():
    result, log, _robot = _run("full", budget=100, max_steps=3)
    steps = [log[i : i + 4] for i in range(0, len(log), 4)]
    assert steps == [["observe", "act", "shadow", "execute"]] * 3, log
    assert result["executed_steps"] == 3


def test_exhaustion_terminates_a_substituted_attempt_but_not_an_unsubstituted_one():
    """The frozen protocol's B2 defect, keyed on the substitution set, exercised both ways."""
    substituted, _log, robot = _run("dx", budget=4, max_steps=10)
    assert substituted["termination_reason"] == "shadow_expert_exhausted"
    assert substituted["executed_steps"] == 4 and robot.stopped == "shadow_expert_exhausted"
    assert substituted["shadow_expert_exhausted_at_step"] == 4
    recording, _log, _robot = _run("none", budget=4, max_steps=10)
    assert recording["termination_reason"] == "step_limit"
    assert recording["executed_steps"] == 10
    assert recording["shadow_expert_exhausted_at_step"] == 4
    assert recording["departures"][4:] == [None] * 6, "undefined past exhaustion, never zero"


def test_substituted_components_reach_the_robot_and_unsubstituted_ones_do_not():
    """B3's mechanism, at unit scale: 'full' commands the expert, 'none' commands the controller."""
    commanded = {}
    for configuration in ("full", "none"):
        executed = []
        robot = _FakeRobot([])
        robot.execute = lambda a, executed=executed: (
            executed.append(np.asarray(a).copy())
            or SimpleNamespace(applied_action=np.asarray(a).copy(), reason=None)
        )
        pd.run_diagnostic_attempt(
            _LoggingController([]),
            robot,
            SimpleNamespace(evaluate=lambda: {}),
            configuration=configuration,
            clip=_clip,
            lower=np.array((0.0,) * 6 + (-0.5,) * 6 + (-1.0, -1.0), np.float32),
            upper=np.array((0.0,) * 6 + (0.5,) * 6 + (-1.0, 1.0), np.float32),
            max_steps=1,
            deadline_seconds=5.0,
            shadow=_LoggingShadow([], 10),
            trace=False,
        )
        commanded[configuration] = executed[0]
    assert commanded["full"][6] == pytest.approx(0.3) and commanded["full"][13] == 1.0
    assert commanded["none"][6] == 0.0 and commanded["none"][13] == -1.0


def test_the_guard_refusals_match_the_runner():
    evaluate_policy = _load("_evaluate_policy", "scripts/evaluate_policy.py")
    assert pd.GUARD_REFUSALS == evaluate_policy.GUARD_REFUSALS
