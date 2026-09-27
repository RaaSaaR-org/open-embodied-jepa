"""TASK-067 run-time pieces: inputs, controllers, the privileged-read counter and one episode.

Protocol ``docs/experiments/apple_first_policy_v1.md``. NumPy at import; MuJoCo, torch and the
DINOv2 encoder are imported lazily, only by the functions that need them.

Where privileged information may go, stated once:

* the **reset truth** (``robot.reset``'s return value) is read by the *harness* before any
  controller exists. It builds the DAgger labeller (training only), the C0 and B-oracle
  controllers and the D-oracle-perc arm (all L4), and nothing else;
* an **L1 controller** (``LearnedController``) is constructed from a prediction function, the
  input standardiser, the four readout estimates and a private forward-kinematics model. It holds
  no robot or simulator handle, and it sees the ``Observation`` and the step only;
* ``PrivilegedReadCounter`` wraps the live simulator's ``task_truth`` *after* the reset and the
  scorer's construction. It counts every call and every call made while a controller's ``act``
  is on the stack (G-privileged).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from embodied_jepa import first_policy as fp
from embodied_jepa.contracts import ContractError

ROOT = Path(__file__).resolve().parents[2]
GUARD_REFUSALS = ("measured joint velocity limit exceeded",)
EXHAUSTED_MESSAGE = "scripted policy has finished"
TABLE_Z = 0.743  # simulation.reset: a free object rests at 0.743 + object_support_height
PLATE_BODY_Z = 0.746  # simulation.reset: the plate body height


# ----- action bounds and assembly ---------------------------------------------------------------
def configured_bounds(path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``planner.lower_bounds`` / ``upper_bounds`` of ``configs/apple_wm_v4.yaml``."""
    import yaml

    data = yaml.safe_load((path or ROOT / fp.CONFIG_BOUNDS).read_text())
    lower = np.asarray(data["planner"]["lower_bounds"], np.float32)
    upper = np.asarray(data["planner"]["upper_bounds"], np.float32)
    if lower.shape != (14,) or upper.shape != (14,) or np.any(lower > upper):
        raise ContractError("configured bounds must be two ordered 14-vectors")
    return lower, upper


def assemble(free, lower, upper) -> np.ndarray:
    """A free 7-vector -> a contract 14-vector with the pinned values, clipped to the bounds."""
    free = np.asarray(free, np.float32)
    if free.shape != (7,) or not np.isfinite(free).all():
        raise ContractError("a policy must emit a finite free 7-vector")
    action = np.zeros(14, np.float32)
    action[12] = -1.0
    action[list(fp.FREE_INDICES)] = np.clip(free, -1.0, 1.0)
    return np.clip(action, lower, upper).astype(np.float32)


def free_of(action) -> np.ndarray:
    return np.asarray(action, np.float32)[list(fp.FREE_INDICES)].copy()


def state_of(observation) -> np.ndarray:
    """The 86-d proprioception of a single ``Observation`` (``RobotState.values[0]``)."""
    values = np.asarray(observation.state.values, np.float64)
    if values.shape[0] != 1:
        raise ContractError("one observation at a time")
    return values[0].copy()


# ----- inputs -----------------------------------------------------------------------------------
def input_vector(estimates, step: int, state, palm9) -> np.ndarray:
    """The 132-d raw input, in ``POLICY_INPUTS`` order: estimates, clock, proprioception, palm."""
    parts = [
        np.asarray(estimates, np.float64).reshape(-1),
        fp.clock_features(step).astype(np.float64),
        np.asarray(state, np.float64).reshape(-1),
        np.asarray(palm9, np.float64).reshape(-1),
    ]
    sizes = tuple(len(p) for p in parts)
    if sizes != tuple(fp.POLICY_INPUTS.values()):
        raise ContractError(f"input parts {sizes} are not {tuple(fp.POLICY_INPUTS.values())}")
    x = np.concatenate(parts)
    if not np.isfinite(x).all():
        raise ContractError("G-finite: a policy input is not finite")
    return x.astype(np.float32)


class Standardiser:
    """Per-dimension train moments with the std floored (``INPUT_STD_FLOOR``)."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, np.float64)
        self.std = np.maximum(np.asarray(std, np.float64), fp.INPUT_STD_FLOOR)
        if self.mean.shape != self.std.shape or not np.isfinite(self.mean).all():
            raise ContractError("standardiser moments must be finite and matching")

    @classmethod
    def fit(cls, rows):
        rows = np.asarray(rows, np.float64)
        return cls(rows.mean(axis=0), rows.std(axis=0))

    def __call__(self, rows):
        return ((np.asarray(rows, np.float64) - self.mean) / self.std).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


# ----- forward kinematics on a private model ------------------------------------------------------
class PalmFK:
    """The right palm pose from joint positions, on the controller's OWN MuJoCo model and data.

    It never touches the live simulator: it builds its own scene once and runs
    ``mj_kinematics`` on a scratch ``MjData``. The pelvis is fixed in this scene, so the palm
    pose in the base frame is a function of the arm joint positions alone."""

    def __init__(self):
        from embodied_jepa.embodiment import G1Embodiment
        from embodied_jepa.simulation import MuJoCoSimulation

        self._robot = G1Embodiment(
            MuJoCoSimulation(object_kind="apple", container_kind="plate", width=112, height=112)
        )
        self._mj = self._robot.mj
        self._data = self._mj.MjData(self._robot.model)
        self._qadr = np.asarray(self._robot.sim.qadr)
        # Robot-owned constants for A4-look: the fixed pelvis pose and the scene heights.
        self.base_position = np.asarray(self._robot.model.body("pelvis").pos, float).copy()
        self.object_support_height = float(self._robot.sim.object_support_height)
        self.container_surface_z = float(self._robot.sim.container_surface_z)

    def pose9(self, state) -> np.ndarray:
        state = np.asarray(state, np.float64).reshape(-1)
        joints = len(self._qadr)
        if state.shape != (2 * joints,):
            raise ContractError(f"proprioception must be {2 * joints}-d")
        self._data.qpos[self._qadr] = state[:joints]
        self._mj.mj_kinematics(self._robot.model, self._data)
        position, rotation = self._robot.ee_pose("right", data=self._data)
        return np.concatenate((position, rotation[:, 0], rotation[:, 1])).astype(np.float64)

    def close(self):
        self._robot.close() if hasattr(self._robot, "close") else None


def a4_truth(estimates, fk: PalmFK) -> dict:
    """The dict A4-look builds ``apple_collector_policy`` from: estimated xy, declared constants.

    Nothing here comes from ``task_truth()``: heights and the base pose are constants of the
    committed scene, read from the controller's private model."""
    apple_x, apple_y, plate_x, plate_y = (float(v) for v in np.asarray(estimates).reshape(4))
    return {
        "position_frame": "world",
        "base_position_world": fk.base_position.copy(),
        "base_rotation_world": np.eye(3),
        "object_position": np.array([apple_x, apple_y, TABLE_Z + fk.object_support_height]),
        "plate_position": np.array([plate_x, plate_y, PLATE_BODY_Z]),
        "container_surface_z": fk.container_surface_z,
        "object_support_height": fk.object_support_height,
    }


# ----- G-privileged ------------------------------------------------------------------------------
class PrivilegedReadCounter:
    """Wraps a live simulator's ``task_truth``: every call, and calls inside a controller's act."""

    def __init__(self, sim):
        self.sim = sim
        self.total = 0
        self.in_controller = 0
        self._inside = False
        self._original = sim.task_truth

        def counted(*args, **kwargs):
            self.total += 1
            if self._inside:
                self.in_controller += 1
            return self._original(*args, **kwargs)

        sim.task_truth = counted

    def act(self, controller, observation, step):
        self._inside = True
        try:
            return controller.act(observation, step)
        finally:
            self._inside = False

    def remove(self):
        self.sim.task_truth = self._original


# ----- controllers (all: act(observation, step) -> 14-vector; advance(result)) -------------------
class LearnedController:
    """An L1 (or F: L2) policy. It holds no robot or simulator handle (G-privileged (3))."""

    def __init__(self, predict, standardiser: Standardiser, estimates, fk: PalmFK, bounds):
        self.predict = predict  # (standardised [132] float32, step) -> free 7
        self.standardiser = standardiser
        self.estimates = np.asarray(estimates, np.float64).reshape(4).copy()
        self.fk = fk
        self.lower, self.upper = bounds
        self.last_input = None

    def act(self, observation, step):
        state = state_of(observation)
        raw = input_vector(self.estimates, step, state, self.fk.pose9(state))
        self.last_input = raw
        free = np.asarray(self.predict(self.standardiser(raw[None])[0], step), np.float32)
        return assemble(free, self.lower, self.upper)

    def advance(self, result):  # noqa: ARG002
        return None


class HarnessController:
    """Adapts a ``policy_diagnostics`` controller (act(observation) -> 14-vector)."""

    def __init__(self, inner):
        self.inner = inner

    def act(self, observation, step):  # noqa: ARG002
        return np.asarray(self.inner.act(observation), np.float32)

    def advance(self, result):
        return self.inner.advance(result)


class ReplayController:
    """B-replay: a recorded action sequence, open loop, from post-look step 0."""

    def __init__(self, actions):
        self.actions = np.asarray(actions, np.float32)
        if self.actions.ndim != 2 or self.actions.shape[1] != 14 or len(self.actions) == 0:
            raise ContractError("replay needs a non-empty [T, 14] action array")

    def act(self, observation, step):  # noqa: ARG002
        if step >= len(self.actions):
            raise ContractError(EXHAUSTED_MESSAGE)
        return self.actions[step].copy()

    def advance(self, result):  # noqa: ARG002
        return None


def scripted_controller(truth, robot):
    """``scripted_oracle`` (B-oracle), C0's perturbed expert and A4-look: L3/L4 only."""
    from embodied_jepa.policy_diagnostics import ScriptedOracleController

    return HarnessController(ScriptedOracleController(truth, robot))


def perturbed_truth(truth, apple_offset_m, plate_offset_m) -> dict:
    """C0: the reset truth with the apple and plate xy shifted by fixed offsets (metres)."""
    out = dict(truth)
    apple = np.asarray(truth["object_position"], float).copy()
    plate = np.asarray(truth["plate_position"], float).copy()
    apple[:2] += np.asarray(apple_offset_m, float)
    plate[:2] += np.asarray(plate_offset_m, float)
    out["object_position"], out["plate_position"] = apple, plate
    return out


def c0_offsets(seed_index: int, apple_cm: float, plate_cm: float) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-size offsets in a direction drawn per C0 seed from ``default_rng(6700)``."""
    angles = np.random.default_rng(fp.C0_DIRECTION_SEED).uniform(0, 2 * np.pi, (fp.C0_RESETS, 2))
    a, p = angles[seed_index]
    return (
        apple_cm / 100.0 * np.array([np.cos(a), np.sin(a)]),
        plate_cm / 100.0 * np.array([np.cos(p), np.sin(p)]),
    )


# ----- one episode -------------------------------------------------------------------------------
def make_robot():
    """As the look collector: 112 px onboard, renderer warmed up (its first render discarded)."""
    from embodied_jepa.embodiment import G1Embodiment
    from embodied_jepa.simulation import MuJoCoSimulation

    robot = G1Embodiment(
        MuJoCoSimulation(
            object_kind="apple",
            container_kind="plate",
            width=fp.IMAGE_SIZE,
            height=fp.IMAGE_SIZE,
        )
    )
    robot.sim.render()
    return robot


def reset_and_look(robot, seed: int, reset: dict):
    """Reset, capture the reset truth, build the scorer, install the counter, run the look.

    Returns ``(truth, scorer, counter, post_look_observation, look_facts)``. The counter is
    installed after the reset and after the scorer's construction, so its total counts the
    scorer's ``evaluate`` calls only (and any controller read)."""
    from embodied_jepa import observation_reprobe as orp
    from embodied_jepa.task import AppleToPlateTask

    truth = robot.reset(int(seed), object_xy=reset["object_xy"], plate_xy=reset["plate_xy"])
    if not np.allclose(truth["base_rotation_world"], np.eye(3), atol=1e-7):
        raise fp.GuardError("the scene's pelvis is not upright")
    scorer = AppleToPlateTask(robot)
    counter = PrivilegedReadCounter(robot.sim)
    applied = orp.execute_look(robot)
    requested = np.asarray(orp.look_sequence(), np.float32)
    lower, upper = orp.COLLECTION_LOWER, orp.COLLECTION_UPPER
    expected = np.clip(requested, lower, upper).astype(np.float32)
    if not np.array_equal(np.asarray(applied, np.float32), expected):
        counter.remove()
        raise fp.GuardError("G-look: applied look commands differ from the requested ones")
    observation = robot.observe()
    facts = {
        "post_look_state": state_of(observation).copy(),
        "post_look_frame": np.asarray(observation.images[fp.CAMERA][0]).copy(),
    }
    return truth, scorer, counter, observation, facts


def run_policy_steps(
    robot,
    scorer,
    counter: PrivilegedReadCounter,
    controller,
    *,
    bounds,
    max_steps: int = fp.MAX_POLICY_STEPS,
    wall_seconds: float = fp.ATTEMPT_WALL_SECONDS,
    labeller=None,
    record_inputs: bool = False,
) -> dict:
    """The attempt loop after the look. ``labeller`` (DAgger only) is a ``ShadowExpert``."""
    lower, upper = bounds
    reason, score = "step_limit", {}
    executed = evaluations = 0
    states, steps, labels, commands, stages = [], [], [], [], []
    started = time.monotonic()
    try:
        for step in range(max_steps):
            if time.monotonic() - started > wall_seconds:
                # G-cap (protocol §12-§14): a cap is V, not a non-success.
                reason = "attempt_wall_cap"
                raise fp.GuardError(f"G-cap: an attempt exceeded its {wall_seconds} s cap")
            observation = robot.observe()
            try:
                command = counter.act(controller, observation, step)
            except ContractError as error:
                if str(error) != EXHAUSTED_MESSAGE:
                    raise
                reason = "policy_complete"
                break
            command = np.clip(np.asarray(command, np.float32), lower, upper).astype(np.float32)
            if labeller is not None and step < fp.EXPERT_POLICY_STEPS:
                label = labeller.command(robot, step)
                if label is not None:
                    states.append(state_of(observation).copy())
                    steps.append(step)
                    labels.append(
                        np.clip(label, lower[list(fp.FREE_INDICES)], upper[list(fp.FREE_INDICES)])
                    )
            elif record_inputs:
                states.append(state_of(observation).copy())
                steps.append(step)
            commands.append(free_of(command))
            try:
                projection = robot.project_candidates(command[None, None, None])
            except ContractError as error:
                if str(error) not in GUARD_REFUSALS:
                    raise
                reason = "guard_refusal"
                break
            if not bool(projection.feasible[0, 0]):
                reason = "infeasible_command"
                break
            result = robot.execute(projection.actions[0, 0, 0])
            if result.applied_action is None:
                reason = result.reason or result.status
                break
            controller.advance(result)
            if labeller is not None:
                labeller.advance(result)
            executed += 1
            score = scorer.evaluate()
            evaluations += 1
            stages.append(
                next(
                    (
                        s
                        for s in ("release", "place", "transport", "grasp", "reach")
                        if score.get(s)
                    ),
                    "none",
                )
            )
            if score.get("success", False):
                reason = "success"
                break
    finally:
        robot.stop(reason)
        counter.remove()
    return {
        "termination_reason": reason,
        "executed_steps": executed,
        "grasp": bool(score.get("grasp", False)),
        "success": bool(score.get("success", False)),
        "final_score": {
            k: (bool(v) if isinstance(v, bool | np.bool_) else v) for k, v in score.items()
        },
        "stages": stages,
        "commands": np.asarray(commands, np.float32),
        "task_truth_total": counter.total,
        "task_truth_in_controller": counter.in_controller,
        "scorer_evaluations": evaluations,
        "states": np.asarray(states, np.float64),
        "steps": np.asarray(steps, np.int64),
        "labels": np.asarray(labels, np.float32),
    }


def privileged_reads_ok(record: dict) -> bool:
    """G-privileged (1) and (2): no read inside act, and the total equals the scorer's calls."""
    return (
        record["task_truth_in_controller"] == 0
        and record["task_truth_total"] == record["scorer_evaluations"]
    )
