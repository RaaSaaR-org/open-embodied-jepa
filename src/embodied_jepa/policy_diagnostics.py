"""Closed-loop diagnostic harness for TASK-057 (``apple_policy_diagnostics_v1``, amendment 1).

NumPy only; importable without torch or MuJoCo. It never edits, and never imports,
``embodied_jepa.policy`` -- every trained TASK-056 checkpoint enforces ``sha256(policy.py)`` -- so a
learned controller reaches this module only as an object with ``act(observation)``.

**The shadow expert is ``scripted.apple_collector_policy``**, the policy every surviving BC root
was collected with (``scripts/collect_apple_wide.py`` ``run_root``) and whose ``base_action`` is
the BC target. Amendment 1 of the protocol exists because the frozen text named plain
``OracleManipulationPolicy`` instead: a different phase table (805 commands, not 745), different
targets (no palm/transfer x shift) and no early release. ``SHADOW_EXPERT_BUDGET`` is asserted
against the constructed policy rather than trusted.

The shadow expert is computed **strictly after** the controller's ``act`` returns and is never
an argument to it (protocol B4). In a configuration with an empty substitution set it is a
recording only; in a non-empty one its components replace the controller's for the named
dimensions, which puts a privileged, simulator-truth-initialized controller partly in command.
Nothing produced under a non-empty substitution set is a learned result.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from embodied_jepa.contracts import ContractError, validate_actions

FREE_NAMES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "grasp")
FREE_INDICES = (6, 7, 8, 9, 10, 11, 13)
#: The six arm dimensions. ``grasp`` is never in an arm-dimension aggregate (protocol §2).
ARM_DIMENSIONS = FREE_NAMES[:6]

#: The eight G-SUB candidates -- the ONLY configurations the gate ranges over.
G_SUB_CANDIDATES = {
    "all_translation": ("dx", "dy", "dz"),
    "dx": ("dx",),
    "dy": ("dy",),
    "dz": ("dz",),
    "droll": ("droll",),
    "dpitch": ("dpitch",),
    "dyaw": ("dyaw",),
    "grasp": ("grasp",),
}
#: Controls: configurations, but never G-SUB candidates.
CONTROLS = {"none": (), "full": FREE_NAMES}
CONFIGURATIONS = {**G_SUB_CANDIDATES, **CONTROLS}

#: ``apple_collector_policy``'s phase budget: 130+80+45+150+60+100+100+80.
SHADOW_EXPERT_BUDGET = 745
#: ``scripted.OracleManipulationPolicy.action``'s exhaustion message. Caught by exact text only;
#: any other ContractError propagates.
EXHAUSTED_MESSAGE = "scripted policy has finished"
#: Embodiment guard refusals that end one attempt as a physical stop (mirrors
#: ``scripts/evaluate_policy.py`` ``GUARD_REFUSALS``; a test pins the two equal).
GUARD_REFUSALS = ("measured joint velocity limit exceeded",)
#: D2's persistence requirement: a departure counts once it holds this many consecutive steps.
DEPARTURE_PERSISTENCE = 5


def collector_shadow_policy(initial_truth):
    """The shadow expert: exactly the policy the BC corpus's surviving roots were collected with."""
    from embodied_jepa.scripted import apple_collector_policy

    return apple_collector_policy(initial_truth)


class ShadowExpert:
    """The collector policy advanced on the EXECUTED result, as the collector advanced it.

    ``command`` returns the signed free 7-vector, or ``None`` once the policy's phase budget is
    exhausted (recorded in ``exhausted_at_step``). The phase counter advances on whatever the
    robot executed, so after ``orient`` this is "expert targets on the controller's clock".
    """

    def __init__(self, initial_truth, *, factory: Callable = collector_shadow_policy):
        self.policy = factory(initial_truth)
        self.budget = int(self.policy.max_steps)
        if factory is collector_shadow_policy and self.budget != SHADOW_EXPERT_BUDGET:
            raise ContractError(
                f"shadow expert budget {self.budget} != preregistered {SHADOW_EXPERT_BUDGET}"
            )
        self.exhausted_at_step: int | None = None
        self.calls = 0

    @property
    def phase(self) -> str:
        return self.policy.phase

    @property
    def phase_index(self) -> int:
        return int(self.policy.phase_index)

    def command(self, robot, step: int) -> np.ndarray | None:
        if self.exhausted_at_step is not None:
            return None
        try:
            action = self.policy.action(robot)
        except ContractError as error:
            if str(error) != EXHAUSTED_MESSAGE:
                raise
            self.exhausted_at_step = int(step)
            return None
        self.calls += 1
        return np.asarray(action, np.float32)[list(FREE_INDICES)].copy()

    def advance(self, result) -> None:
        if self.exhausted_at_step is None and not self.policy.done:
            self.policy.advance(result)


def substitute(controller_action, expert_free, configuration: str) -> np.ndarray:
    """The controller's 14-vector with the configuration's free dimensions taken from the expert."""
    if configuration not in CONFIGURATIONS:
        raise ContractError(f"unknown configuration {configuration!r}")
    action = np.asarray(controller_action, np.float32).copy()
    if action.shape != (14,):
        raise ContractError("a controller command must be a 14-vector")
    dims = CONFIGURATIONS[configuration]
    if not dims:
        return action
    if expert_free is None:
        raise ContractError("a substituted configuration needs a live shadow expert")
    for name in dims:
        i = FREE_NAMES.index(name)
        action[FREE_INDICES[i]] = expert_free[i]
    return action


def departure(controller_free, expert_free) -> float | None:
    """Median over the six arm dimensions of |controller - shadow expert|; None if undefined."""
    if expert_free is None:
        return None
    a = np.asarray(controller_free, np.float64)[:6]
    b = np.asarray(expert_free, np.float64)[:6]
    return float(np.median(np.abs(a - b)))


def first_departure_step(departures, tau: float, persistence: int = DEPARTURE_PERSISTENCE):
    """First step t with departure > tau on ``persistence`` consecutive steps t..t+k-1.

    ``None`` entries (shadow expert exhausted) end the search: departure is undefined there and
    is never read as zero. Returns ``None`` when no departure is found (censored).
    """
    if not np.isfinite(tau) or tau <= 0 or persistence < 1:
        raise ContractError("departure threshold must be positive and persistence >= 1")
    run = 0
    for t, value in enumerate(departures):
        if value is None:
            return None
        run = run + 1 if value > tau else 0
        if run >= persistence:
            return t - persistence + 1
    return None


# ----- controllers ---------------------------------------------------------------------------
class ConstantController:
    """A blind controller: the same free 7-vector every step, pinned components at their pins."""

    def __init__(self, free, *, name="constant"):
        free = np.asarray(free, np.float32)
        if free.shape != (7,) or not np.isfinite(free).all():
            raise ContractError("a constant controller needs a finite free 7-vector")
        self.action = np.zeros(14, np.float32)
        self.action[12] = -1.0
        self.action[list(FREE_INDICES)] = free
        self.name = name

    def act(self, observation):  # noqa: ARG002 - blind by construction
        return self.action.copy()

    def advance(self, result):  # noqa: ARG002
        return None


class TimeIndexedController:
    """A blind controller that knows only the step index: row ``min(t, len-1)`` of a table."""

    def __init__(self, table, *, name="time_indexed"):
        table = np.asarray(table, np.float32)
        if table.ndim != 2 or table.shape[1] != 7 or not np.isfinite(table).all():
            raise ContractError("a time-indexed controller needs a finite [T, 7] table")
        self.table, self.step, self.name = table, 0, name

    def act(self, observation):  # noqa: ARG002
        action = np.zeros(14, np.float32)
        action[12] = -1.0
        action[list(FREE_INDICES)] = self.table[min(self.step, len(self.table) - 1)]
        return action

    def advance(self, result):
        if result.applied_action is not None:
            self.step += 1


def hold_controller():
    """``benchmark.py``'s hold: zero motion, both hands commanded open (-1)."""
    return ConstantController((0, 0, 0, 0, 0, 0, -1), name="hold")


class RandomController:
    """``benchmark.py``'s random: uniform in the configured bounds, ``default_rng(seed)``."""

    name = "random"

    def __init__(self, seed, lower, upper):
        self.rng = np.random.default_rng(int(seed))
        self.lower = np.asarray(lower, np.float32)
        self.upper = np.asarray(upper, np.float32)

    def act(self, observation):  # noqa: ARG002
        return self.rng.uniform(self.lower, self.upper).astype(np.float32)

    def advance(self, result):  # noqa: ARG002
        return None


class ScriptedOracleController:
    """B1's ``scripted_oracle``: its own collector instance in command (non-learned reference).

    Exhaustion ends the attempt as ``policy_complete``, as ``evaluate_apple.py`` records it.
    """

    name = "scripted_oracle"

    def __init__(self, initial_truth, robot):
        self.policy = collector_shadow_policy(initial_truth)
        self.robot = robot

    def act(self, observation):  # noqa: ARG002 - reads the robot pose, as the collector does
        return np.asarray(self.policy.action(self.robot), np.float32)

    def advance(self, result):
        if not self.policy.done:
            self.policy.advance(result)


class PolicyController:
    """Adapter for a learned policy exposing ``act(images, robot_state)``."""

    name = "policy"

    def __init__(self, policy):
        self.policy = policy

    def act(self, observation):
        return np.asarray(self.policy.act(observation.images, observation.state), np.float32)

    def advance(self, result):  # noqa: ARG002
        return None


# ----- the attempt loop ----------------------------------------------------------------------
def palm_apple_distance(robot) -> float:
    """Scoring-only simulator truth, recorded in the trace; never reaches a controller."""
    truth = robot.sim.task_truth()
    palm = np.asarray(robot.sim.data.site("right_ee").xpos, float)
    return float(np.linalg.norm(palm - np.asarray(truth["object_position"], float)))


def _stage(score) -> str:
    return next(
        (s for s in ("release", "place", "transport", "grasp", "reach") if score.get(s)), "none"
    )


def run_diagnostic_attempt(
    controller,
    robot,
    scorer,
    *,
    configuration: str,
    clip: Callable,
    lower,
    upper,
    max_steps: int,
    deadline_seconds: float,
    shadow: ShadowExpert,
    trace: bool = True,
):
    """One closed-loop attempt under one configuration. Mirrors ``evaluate_policy.run_attempt``.

    Order per step: observe -> ``controller.act`` -> shadow expert (strictly after) ->
    substitution -> configured-bounds clip -> the embodiment's unchanged projection -> execute ->
    advance both -> score.
    """
    if configuration not in CONFIGURATIONS:
        raise ContractError(f"unknown configuration {configuration!r}")
    substituted = bool(CONFIGURATIONS[configuration])
    reason, score, executed, clipped_commands = "step_limit", {}, 0, 0
    first_command, first_expert = None, None
    rows, control_times, departures = [], [], []
    try:
        for step in range(max_steps):
            began = time.perf_counter()
            observation = robot.observe()
            try:
                raw = np.asarray(controller.act(observation), np.float32)
            except ContractError as error:
                if isinstance(controller, ScriptedOracleController) and (
                    str(error) == EXHAUSTED_MESSAGE
                ):
                    reason = "policy_complete"
                    break
                raise
            # Strictly AFTER act() has returned (B4).
            expert = shadow.command(robot, step)
            if substituted and expert is None:
                reason = "shadow_expert_exhausted"
                break
            if first_command is None:
                first_command = raw[list(FREE_INDICES)].copy()
                first_expert = None if expert is None else expert.copy()
            requested = substitute(raw, expert, configuration)
            command, was_clipped, _dims = clip(requested, lower, upper)
            clipped_commands += int(was_clipped)
            controller_free = raw[list(FREE_INDICES)]
            departures.append(departure(controller_free, expert))
            if trace:
                rows.append(
                    {
                        "step": step,
                        "controller": [float(v) for v in controller_free],
                        "shadow_expert": None if expert is None else [float(v) for v in expert],
                        "commanded": [float(v) for v in command[list(FREE_INDICES)]],
                        "shadow_expert_phase": shadow.phase,
                        "palm_apple_m": palm_apple_distance(robot),
                        "stage": _stage(score),
                        "departure": departures[-1],
                    }
                )
            validate_actions(command[None, None, None], ndim=4)
            try:
                projection = robot.project_candidates(command[None, None, None])
            except ContractError as exc:
                if str(exc) not in GUARD_REFUSALS:
                    raise
                reason = "guard_refusal"
                break
            if not bool(projection.feasible[0, 0]):
                reason = "infeasible_command"
                break
            result = robot.execute(projection.actions[0, 0, 0])
            control_times.append(time.perf_counter() - began)
            if result.applied_action is None:
                reason = result.reason or result.status
                break
            executed += 1
            shadow.advance(result)
            controller.advance(result)
            score = scorer.evaluate()
            if score.get("success", False):
                reason = "success"
                break
            if control_times[-1] > deadline_seconds:
                reason = "deadline_miss"
                break
    except BaseException:
        reason = "error"
        raise
    finally:
        robot.stop(reason)
    return {
        "configuration": configuration,
        "substitution_set": list(CONFIGURATIONS[configuration]),
        "controller": getattr(controller, "name", type(controller).__name__),
        "termination_reason": reason,
        "executed_steps": executed,
        "clipped_commands": clipped_commands,
        "shadow_expert_budget": shadow.budget,
        "shadow_expert_exhausted_at_step": shadow.exhausted_at_step,
        "first_command": None if first_command is None else [float(v) for v in first_command],
        "first_shadow_expert": None if first_expert is None else [float(v) for v in first_expert],
        "departures": departures,
        "trace": rows if trace else None,
        "score": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in score.items()},
        "grasp": bool(score.get("grasp", False)),
        "success": bool(score.get("success", False)),
        "median_control_seconds": float(np.median(control_times)) if control_times else None,
    }
