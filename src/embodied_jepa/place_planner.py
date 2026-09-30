"""TASK-074: the place-phase planner. P-3 picks, then a planner chooses where e9's place
primitive puts the apple.

Protocol ``docs/experiments/apple_lewm_planner_v2.md`` §3.

* **The place primitive** (``PlacePrimitive``) is e9's own plate-dependent phases (transfer,
  lower, steady, open, clear, retreat: ``resting_expert.RestingPlaceExpert`` with e9's kwargs),
  built for a target xy instead of the plate. Its phase clock is e9's: it starts at
  ``TRANSFER_START`` (405) and ends at 725.
  - It reads only the robot's own joint state. The palm pose comes from the controller's private
    kinematic model (``first_policy_runtime.PalmFK``), and the heights from the scene's declared
    constants (``a4_truth``).
  - Aimed at the true plate it is the family ceiling (H-handover, L4). Aimed at a planner's
    target it is not privileged.
* **The planner** (``PlannerController``) runs P-3 unchanged until 405. At every decision step
  (405, 421, ..., 485) it chooses a target and re-targets the primitive, which keeps its phase
  clock. The modes are:
  - ``critic`` (L-plan, L-N): the 49 coarse targets around the post-look plate estimate, then the
    25 fine targets around the coarse choice. Each target's 16-command chunk comes from the
    kinematic stand-in (TASK-073's ``KinematicStandIn``: object-free, ideal tracking). A critic
    scores the chunks, and the tie rule keeps the incumbent;
  - ``shuf``: the same, from a foreign start latent (L-shuf);
  - ``rand``: a seeded uniform coarse, then fine choice among reachable targets (L-rand);
  - ``twin``: R-plate's reading of the frame (H-twin, the perception twin);
  - ``truth``: the harness's true plate (H-handover, L4).
* The controller holds no simulator handle. What it reads comes from the ``Observation``, the
  step and, in ``truth`` mode only, the harness's value.

NumPy at import; MuJoCo and torch are imported lazily by the pieces that need them.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa import first_policy_v2_runtime as rt2
from embodied_jepa import hybrid_selection as hs
from embodied_jepa import lewm_planner_v2 as lp
from embodied_jepa.contracts import ContractError

MODES = ("critic", "shuf", "rand", "twin", "truth")
FREE = list(fp2.FREE_INDICES)
DECISION_SET = frozenset(lp.DECISION_STEPS)


class KinematicPose:
    """``ee_pose`` and ``manifest`` of the robot at a given joint state, on PalmFK's private
    model (the fixed pelvis makes the palm pose a function of the arm joints alone)."""

    def __init__(self, fk):
        self.fk = fk
        self.manifest = fk._robot.manifest

    def update(self, state) -> None:
        self.fk.pose9(state)  # sets the private data's qpos and runs mj_kinematics

    def ee_pose(self, side):
        return self.fk._robot.ee_pose(side, data=self.fk._data)


class PlacePrimitive:
    """e9's place phases for a target xy, on e9's clock from ``TRANSFER_START``."""

    def __init__(self, fk, apple_xy, accepted_grasp: float = -1.0):
        self.fk = fk
        self.pose = KinematicPose(fk)
        self.apple = np.asarray(apple_xy, np.float64).reshape(2)
        self.accepted_grasp = float(accepted_grasp)
        self.expert = None
        self.target = None

    def build(self, target_xy):
        """A fresh e9 for ``target_xy`` (raises ``ContractError`` if its release pose is out of
        reach)."""
        from embodied_jepa.resting_expert import RestingPlaceExpert

        truth = rt2.a4_truth([*self.apple, *np.asarray(target_xy, float).reshape(2)], self.fk)
        expert = RestingPlaceExpert(truth, **fp2.EXPERT)
        budgets = tuple(int(p.commands) for p in expert.phases)
        if budgets != fp2.EXPERT_BUDGETS:
            raise ContractError(f"e9's schedule {budgets} is not the preregistered one")
        return expert

    def retarget(self, target_xy, step: int) -> None:
        """Aim at ``target_xy`` from ``step`` (>= 405), keeping the phase clock."""
        expert = self.build(target_xy)
        if self.expert is not None:
            expert.phase_index = self.expert.phase_index
            expert.phase_step = self.expert.phase_step
            expert.step_count = self.expert.step_count
            expert.failure = self.expert.failure
        else:
            k = int(step) - lp.TRANSFER_START
            if k < 0:
                raise ContractError("the place primitive starts at the transfer phase")
            index = lp.PICK_PHASES
            while index < len(expert.phases) and k >= expert.phases[index].commands:
                k -= expert.phases[index].commands
                index += 1
            expert.phase_index, expert.phase_step, expert.step_count = index, k, int(step)
        expert.accepted_grasp = self.accepted_grasp
        self.expert = expert
        self.target = np.asarray(target_xy, np.float64).reshape(2).copy()

    @property
    def done(self) -> bool:
        return self.expert is None or self.expert.done

    def command(self, state) -> np.ndarray:
        if self.expert is None:
            raise ContractError("the place primitive has no target")
        if self.expert.done:
            raise ContractError(rt2.EXHAUSTED_MESSAGE)
        self.pose.update(state)
        return np.asarray(self.expert.action(self.pose), np.float32)

    def advance(self, result) -> None:
        if result.applied_action is not None:
            self.accepted_grasp = float(result.applied_action[13])
        if self.expert is not None and not self.expert.done:
            self.expert.advance(result)


def reachable(fk, apple_xy, target_xy) -> bool:
    try:
        PlacePrimitive(fk, apple_xy).build(target_xy)
    except ContractError:
        return False
    return True


def primitive_chunks(
    standin, fk, apple_xy, state, step: int, targets, *, last_grasp, horizon: int = lp.CHUNK
):
    """The place primitive's ``horizon``-command chunk for each target, rolled against the
    kinematic stand-in from ``state`` at ``step``. Returns ``requested`` [K, h, 14],
    ``applied`` [K, h, 14] (the world model's input) and ``feasible`` [K] (False for an
    unreachable release pose or a stand-in refusal)."""
    lower, upper = standin.lower, standin.upper
    targets = np.asarray(targets, np.float64).reshape(-1, 2)
    k = len(targets)
    state = np.asarray(state, np.float64).reshape(-1)
    joints = standin.joints
    start_targets = standin.initial_targets(state, last_grasp)
    requested = np.zeros((k, horizon, 14), np.float32)
    applied = np.zeros((k, horizon, 14), np.float32)
    feasible = np.ones(k, bool)
    for i, target in enumerate(targets):
        primitive = PlacePrimitive(fk, apple_xy, accepted_grasp=float(last_grasp[1]))
        try:
            primitive.retarget(target, step)
        except ContractError:
            feasible[i] = False
            continue
        s, tgt = state.copy(), start_targets.copy()
        for j in range(horizon):
            command = np.clip(primitive.command(s), lower, upper).astype(np.float32)
            requested[i, j] = command
            out = standin.project(command, s[:joints], tgt)
            if out is None:
                feasible[i] = False
                break
            applied[i, j], tgt = out
            q = tgt.astype(np.float64)
            v = (q - s[:joints]) / standin.dt
            s = np.concatenate((q, v)).astype(np.float32).astype(np.float64)
            primitive.advance(SimpleNamespace(applied_action=applied[i, j], reason=None))
    return requested, applied, feasible


class PlannerController(rt2.LearnedController):
    """P-3 until 405, then the place primitive aimed by the planner (see the module docstring).

    ``decisions`` records every decision; ``latents`` the pooled latent at each decision (image
    modes); ``chunks`` the stand-in chunk of the executed target (``log_chunks``, O5)."""

    def __init__(
        self,
        predict,
        standardiser,
        estimates,
        fk,
        bounds,
        *,
        mode: str,
        encoder=None,
        critic=None,
        standin=None,
        r_plate=None,
        foreign_latents: dict | None = None,
        rng_seed: int | None = None,
        truth_target=None,
        log_chunks: bool = False,
        keep_costs: bool = False,
    ):
        super().__init__(predict, standardiser, estimates, fk, bounds)
        if mode not in MODES:
            raise ContractError(f"unknown planner mode {mode!r}")
        needs = {
            "critic": (encoder, critic, standin),
            "shuf": (critic, standin, foreign_latents),
            "rand": (rng_seed,),
            "twin": (encoder, r_plate),
            "truth": (truth_target,),
        }[mode]
        if any(x is None for x in needs):
            raise ContractError(f"planner mode {mode!r} is missing a component")
        if log_chunks and standin is None:
            raise ContractError("logging chunks needs the stand-in")
        self.mode = mode
        self.encoder, self.critic, self.standin, self.r_plate = encoder, critic, standin, r_plate
        self.foreign = foreign_latents
        self.rng = np.random.default_rng(rng_seed) if rng_seed is not None else None
        self.truth_target = truth_target
        self.log_chunks, self.keep_costs = log_chunks, keep_costs
        self.apple = np.asarray(self.estimates[:2], np.float64).copy()
        self.anchor = np.asarray(self.estimates[2:], np.float64).copy()  # post-look plate
        self.primitive: PlacePrimitive | None = None
        self.decisions: list[dict] = []
        self.latents: dict[int, np.ndarray] = {}
        self.chunks: dict[int, dict] = {}
        self.last_grasp = (-1.0, -1.0)

    # ----- the choices -----
    def _critic_choice(self, start, state, step: int, record: dict):
        coarse = lp.coarse_targets(self.anchor)
        _r, applied, feasible = primitive_chunks(
            self.standin, self.fk, self.apple, state, step, coarse, last_grasp=self.last_grasp
        )
        record["coarse_infeasible"] = int((~feasible).sum())
        if not feasible.any():
            return None
        costs = np.where(feasible, self.critic.costs(start, applied, step + lp.CHUNK), np.inf)
        ci = lp.choose(costs, lp.COARSE_INCUMBENT)
        fine = lp.fine_targets(coarse[ci])
        _r, f_applied, f_feasible = primitive_chunks(
            self.standin, self.fk, self.apple, state, step, fine, last_grasp=self.last_grasp
        )
        f_costs = np.where(f_feasible, self.critic.costs(start, f_applied, step + lp.CHUNK), np.inf)
        fi = lp.choose(f_costs, lp.FINE_CENTRE)
        record["chosen"] = [int(ci), int(fi)]
        if self.keep_costs:
            record["coarse_costs_cm"] = [float(c) for c in costs]
            record["fine_costs_cm"] = [float(c) for c in f_costs]
        return fine[fi]

    def _rand_choice(self, record: dict):
        coarse = lp.coarse_targets(self.anchor)
        ok = [i for i, g in enumerate(coarse) if reachable(self.fk, self.apple, g)]
        if not ok:
            return None
        ci = int(ok[int(self.rng.integers(len(ok)))])
        fine = lp.fine_targets(coarse[ci])
        ok_f = [i for i, g in enumerate(fine) if reachable(self.fk, self.apple, g)]
        fi = int(ok_f[int(self.rng.integers(len(ok_f)))]) if ok_f else lp.FINE_CENTRE
        record["chosen"] = [ci, fi]
        return fine[fi]

    def decide(self, observation, step: int) -> None:
        started = time.perf_counter()
        state = rt2.state_of(observation)
        record: dict = {"step": int(step), "chosen": None, "fallback": False}
        target = None
        if self.mode in ("critic", "twin", "shuf"):
            if self.mode != "shuf" or self.encoder is not None:
                frame = np.asarray(observation.images[fp2.CAMERA][0])
                tokens, pooled = hs.encode_frame(self.encoder, frame)
                self.latents[int(step)] = pooled
            if self.mode == "twin":
                target = np.asarray(self.r_plate.predict(tokens), np.float64).reshape(2)
                record["reading"] = target.tolist()
            else:
                start = self.foreign[int(step)] if self.mode == "shuf" else pooled
                target = self._critic_choice(start, state, step, record)
        elif self.mode == "rand":
            target = self._rand_choice(record)
        else:
            target = np.asarray(self.truth_target(step), np.float64).reshape(2)
        if self.primitive is None:
            self.primitive = PlacePrimitive(self.fk, self.apple, float(self.last_grasp[1]))
        applied_target = None
        for candidate in (target, self.primitive.target, self.anchor):
            if candidate is None:
                continue
            try:
                self.primitive.retarget(candidate, step)
            except ContractError:
                continue
            applied_target = candidate
            break
        if applied_target is None:
            raise ContractError("the planner has no reachable target, not even the anchor")
        record["fallback"] = target is None or applied_target is not target
        record["target"] = np.asarray(applied_target, np.float64).tolist()
        if self.log_chunks:
            requested, _a, feasible = primitive_chunks(
                self.standin,
                self.fk,
                self.apple,
                state,
                step,
                [applied_target],
                last_grasp=self.last_grasp,
            )
            self.chunks[int(step)] = {
                "requested_free": requested[0][:, FREE].tolist(),
                "feasible": bool(feasible[0]),
            }
        record["seconds"] = time.perf_counter() - started
        self.decisions.append(record)

    # ----- the controller interface -----
    def act(self, observation, step):
        step = int(step)
        if step < lp.TRANSFER_START:
            return super().act(observation, step)
        if step in DECISION_SET:
            self.decide(observation, step)
        if self.primitive is None:
            raise ContractError("the planner reached the place phase without a decision")
        return self.primitive.command(rt2.state_of(observation))

    def advance(self, result):
        if result.applied_action is not None:
            self.last_grasp = tuple(float(v) for v in result.applied_action[12:14])
        if self.primitive is not None:
            self.primitive.advance(result)
        return None


class BranchPlace:
    """One ranking branch: the place primitive aimed at a fixed target from ``step``."""

    def __init__(self, fk, apple_xy, target_xy, step: int, accepted_grasp: float):
        self.primitive = PlacePrimitive(fk, apple_xy, accepted_grasp)
        self.primitive.retarget(target_xy, step)

    def act(self, observation, step):  # noqa: ARG002
        return self.primitive.command(rt2.state_of(observation))

    def advance(self, result):
        self.primitive.advance(result)


class RecordingReread(hs.HybridController):
    """P-far at run time and in its DAgger rollouts: P-3's controller class with the plate
    re-read by R-plate at the decision steps; it keeps each step's raw input (DAgger rows)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, variant="reread", decision_steps=lp.DECISION_STEPS, **kwargs)
        self.inputs: dict[int, np.ndarray] = {}

    def act(self, observation, step):
        command = super().act(observation, step)
        self.inputs[int(step)] = np.asarray(self.last_input, np.float64).copy()
        return command


def target_difference_cm(first: dict, second: dict) -> float:
    """The largest per-decision target difference (cm) between two runs; inf if the decision
    counts differ (the re-run field ``target_cm_max``)."""
    a = [d.get("target") for d in first.get("decisions", [])]
    b = [d.get("target") for d in second.get("decisions", [])]
    if len(a) != len(b):
        return float("inf")
    if not a:
        return 0.0
    diff = [100.0 * float(np.linalg.norm(np.subtract(x, y))) for x, y in zip(a, b, strict=True)]
    return float(max(diff))


def rerun_compare(first: dict, second: dict) -> dict:
    """The determinism re-run of one (arm, seed) under ``lewm_planner_v2.rerun_rule()``:
    ``target_cm_max`` is 0 in the first record and the largest per-decision target difference in
    the second, so its tolerance applies to that difference."""
    from embodied_jepa.run_guards import rerun_matches

    d = target_difference_cm(first, second)
    return rerun_matches(
        dict(first) | {"target_cm_max": 0.0},
        dict(second) | {"target_cm_max": d},
        lp.rerun_rule(),
    )
