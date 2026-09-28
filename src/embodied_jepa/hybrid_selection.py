"""TASK-073: the hybrid controller -- P-3 proposes aims, a critic picks one, P-3 executes it.

Protocol ``docs/experiments/apple_wm_critic_v2.md`` §3. The carried policy P-3 (TASK-072 run-1)
is unchanged: ``LearnedController`` with its mutable ``estimates`` (apple xy, plate xy). At every
decision step the controller re-reads the plate from the current frame with ``R-mid`` (the
incumbent), forms the 25 aim hypotheses, generates each hypothesis' 16-command chunk by
re-querying P-3 step by step against a **kinematic stand-in**, asks a critic for one cost per
chunk, and sets the chosen aim as P-3's plate estimate for the next 16 closed-loop commands.

* **The kinematic stand-in** (``KinematicStandIn``) advances a private copy of the robot's
  kinematics by ideal tracking: each P-3 command is projected through the embodiment's own
  per-side acceptance calculation (``G1Embodiment._prepare_side``: workspace clip, IK, joint-rate
  and grasp-rate limits, with ``project_candidates``' backtracking), and the joints are assumed to
  reach the accepted targets within the step (velocity = target change / control dt). It is
  object-free, reads no simulator state and holds no simulator handle. The previous joint
  targets, which only the live transport knows, are reconstructed from what the controller
  itself knows: the measured arm joint positions, and the hand synergy of its own last executed
  grasp command (returned to it by ``advance``; a closed hand's measured joints are blocked by the
  apple and lie off the synergy).
* **Variants** (all rung L1; none reads simulator truth): ``stale`` (P-stale: no re-read),
  ``reread`` (P-reread), ``critic`` (H-LeWM, H-N, H-copy: the critic decides; with an
  action-blind critic every cost ties and the incumbent is kept), ``shuf`` (H-shuf: the critic
  starts from another reset's latent at the same step, supplied by the harness), ``rand`` (H-rand:
  a seeded uniform pick among the same 25 candidates).

NumPy at import; MuJoCo and torch are imported lazily by the pieces that need them.
"""

from __future__ import annotations

import time

import numpy as np

from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa import first_policy_v2_runtime as rt2
from embodied_jepa import wm_critic_v2 as wc
from embodied_jepa.contracts import ContractError

VARIANTS = ("stale", "reread", "critic", "shuf", "rand")
FREE = list(fp2.FREE_INDICES)
BACKTRACK = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0)  # project_candidates'


class KinematicStandIn:
    """Object-free, ideal-tracking rollout of P-3 against the embodiment's own kinematics."""

    def __init__(self, bounds):
        from embodied_jepa.embodiment import G1Embodiment
        from embodied_jepa.simulation import MuJoCoSimulation

        self.robot = G1Embodiment(
            MuJoCoSimulation(
                object_kind="apple", container_kind="plate", width=112, height=112, render=False
            )
        )
        self.mj, self.model = self.robot.mj, self.robot.model
        self.data = self.mj.MjData(self.model)
        self.qadr = np.asarray(self.robot.sim.qadr)
        self.joints = len(self.qadr)
        self.dt = float(self.robot.sim.control_dt)
        self.lower, self.upper = bounds

    def _set(self, qpos):
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = qpos
        self.robot._forward_kinematics(self.data)

    def project(self, requested, qpos, targets):
        """One command through the embodiment's acceptance calculation with backtracking.

        Returns ``(applied, new_targets)`` or ``None`` if no factor is feasible."""
        self._set(qpos)
        applied = np.asarray(requested, np.float32).copy()
        for side_index, side in enumerate(("left", "right")):
            offset = side_index * 6
            original = applied[offset : offset + 6].copy()
            for factor in BACKTRACK:
                trial = applied.copy()
                trial[offset : offset + 6] = original * factor
                trial_targets = targets.copy()
                try:
                    self.robot._prepare_side(trial, trial_targets, side_index, side, data=self.data)
                except ContractError:
                    continue
                applied, targets = trial, trial_targets
                break
            else:
                return None
        return applied, targets

    def initial_targets(self, state, last_grasp) -> np.ndarray:
        """The transport's previous joint targets as the controller can know them: measured
        joint positions, with each hand on the synergy of its last executed grasp value."""
        targets = np.asarray(state, np.float64)[: self.joints].astype(np.float32).copy()
        robot = self.robot
        for side_index, side in enumerate(("left", "right")):
            opened = np.asarray(robot.manifest[f"{side}_open_rad"])
            closed = np.asarray(robot.manifest[f"{side}_closed_rad"])
            grasp = float(last_grasp[side_index])
            hand = [robot.actuator_for_joint[int(i)] for i in robot.hand_ids[side]]
            values = opened + (grasp + 1) * 0.5 * (closed - opened)
            targets[hand] = robot._joint_commands(values, robot.hand_ids[side])
        return targets

    def chunks(
        self, policy, state, step: int, estimates, horizon: int = wc.CHUNK, last_grasp=(-1.0, -1.0)
    ):
        """Roll P-3 against the stand-in for every row of ``estimates`` [K, 4].

        ``policy(x_standardised [K, 132], steps [K]) -> free [K, 7]``. Returns ``requested``
        [K, h, 14] (P-3's commands, clipped to the bounds), ``applied`` [K, h, 14] (after the
        embodiment's projection; the world model's input) and ``feasible`` [K]."""
        estimates = np.asarray(estimates, np.float64)
        k = len(estimates)
        state = np.asarray(state, np.float64).reshape(-1)
        if state.shape != (2 * self.joints,):
            raise ContractError(f"the stand-in needs a {2 * self.joints}-d state")
        states = np.repeat(state[None], k, axis=0)
        start_targets = self.initial_targets(state, last_grasp)
        targets = [start_targets.copy() for _ in range(k)]
        requested = np.zeros((k, horizon, 14), np.float32)
        applied = np.zeros((k, horizon, 14), np.float32)
        feasible = np.ones(k, bool)
        for j in range(horizon):
            rows = []
            for i in range(k):
                pose = policy.palm9(states[i])
                rows.append(rt2.input_vector(estimates[i], step + j, states[i], pose))
            free = policy.batch(np.stack(rows), np.full(k, step + j))
            for i in range(k):
                command = rt2.assemble(free[i], self.lower, self.upper)
                requested[i, j] = command
                if not feasible[i]:
                    continue
                out = self.project(command, states[i][: self.joints], targets[i])
                if out is None:
                    feasible[i] = False
                    continue
                applied[i, j], new_targets = out
                q = new_targets.astype(np.float64)
                v = (q - states[i][: self.joints]) / self.dt
                states[i] = np.concatenate((q, v)).astype(np.float32).astype(np.float64)
                targets[i] = new_targets
        return requested, applied, feasible

    def close(self):
        self.robot.close()


class P3Policy:
    """P-3's pieces the stand-in needs: palm FK and a batched, standardised forward pass."""

    def __init__(self, model, standardiser, fk):
        self.model = model
        self.standardiser = standardiser
        self.fk = fk

    def palm9(self, state):
        return self.fk.pose9(state)

    def batch(self, raw_rows, steps) -> np.ndarray:
        from embodied_jepa import first_policy_v2_model as fm2

        return fm2.batch_predict(self.model, self.standardiser(raw_rows), steps)


def encode_frame(encoder, frame) -> tuple[np.ndarray, np.ndarray]:
    """The current frame through frozen DINOv2 (CPU, batch size 1): the full tokens (R-mid's
    input, float64 [98304]) and the 4 x 4 pooled latent (the critic's start, float32 [6144])."""
    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa.models.frozen_tokens import pool_tokens

    tokens = pe.features(encoder, np.array(frame, np.uint8, copy=True)[None], batch=1)["tokens"]
    pooled = pool_tokens(tokens, 4)
    return tokens[0], pooled[0]


class HybridController(rt2.LearnedController):
    """P-3 with scheduled re-reads and, for the critic variants, a critic's choice of aim.

    It holds no robot or simulator handle; everything it reads comes from the ``Observation``
    and the step (G-privileged). ``decisions`` records each decision for the report."""

    def __init__(
        self,
        predict,
        standardiser,
        estimates,
        fk,
        bounds,
        *,
        variant: str,
        policy: P3Policy | None = None,
        encoder=None,
        r_mid=None,
        critic=None,
        standin: KinematicStandIn | None = None,
        rng_seed: int | None = None,
        foreign_latents: dict | None = None,
        decision_steps=wc.DECISION_STEPS,
        keep_costs: bool = True,
    ):
        super().__init__(predict, standardiser, estimates, fk, bounds)
        if variant not in VARIANTS:
            raise ContractError(f"unknown variant {variant!r}")
        needs = {
            "reread": (encoder, r_mid),
            "critic": (encoder, r_mid, critic, standin, policy),
            "shuf": (encoder, r_mid, critic, standin, policy, foreign_latents),
            "rand": (encoder, r_mid),
        }.get(variant, ())
        if any(x is None for x in needs):
            raise ContractError(f"variant {variant!r} is missing a component")
        if variant == "rand" and rng_seed is None:
            raise ContractError("H-rand needs its seed")
        self.variant = variant
        self.policy, self.encoder, self.r_mid = policy, encoder, r_mid
        self.critic, self.standin = critic, standin
        self.rng = np.random.default_rng(rng_seed) if rng_seed is not None else None
        self.foreign = foreign_latents
        self.decision_steps = frozenset(int(s) for s in decision_steps)
        self.keep_costs = keep_costs
        self.decisions: list[dict] = []
        self.latents: dict[int, np.ndarray] = {}
        self.last_grasp = (-1.0, -1.0)

    def advance(self, result):
        if result.applied_action is not None:
            self.last_grasp = tuple(float(v) for v in result.applied_action[12:14])
        return None

    def decide(self, observation, step: int) -> None:
        started = time.perf_counter()
        frame = np.asarray(observation.images[fp2.CAMERA][0])
        tokens, pooled = encode_frame(self.encoder, frame)
        self.latents[int(step)] = pooled
        incumbent = np.asarray(self.r_mid.predict(tokens), np.float64).reshape(2)
        record = {"step": int(step), "incumbent": incumbent.tolist()}
        if self.variant == "reread":
            aim = incumbent
            record["chosen"] = wc.INCUMBENT_INDEX
        else:
            aims = wc.candidate_aims(incumbent)
            if self.variant == "rand":
                index = int(self.rng.integers(wc.K))
            else:
                estimates = np.concatenate(
                    (np.repeat(self.estimates[None, :2], wc.K, axis=0), aims), axis=1
                )
                state = rt2.state_of(observation)
                _requested, applied, feasible = self.standin.chunks(
                    self.policy, state, step, estimates, last_grasp=self.last_grasp
                )
                start = self.foreign[int(step)] if self.variant == "shuf" else pooled
                costs = self.critic.costs(start, applied, step + wc.CHUNK)
                costs = np.where(feasible, costs, np.inf)
                index = wc.choose(costs)
                if self.keep_costs:
                    record["costs_cm"] = [float(c) for c in costs]
                record["infeasible"] = int((~feasible).sum())
            aim = aims[index]
            record["chosen"] = int(index)
        self.estimates[2:] = aim
        record["aim"] = np.asarray(aim).tolist()
        record["seconds"] = time.perf_counter() - started
        self.decisions.append(record)

    def act(self, observation, step):
        if self.variant != "stale" and int(step) in self.decision_steps:
            self.decide(observation, int(step))
        return super().act(observation, step)


class ScheduledTruth(rt2.LearnedController):
    """P-truth (L4): P-3 whose plate estimate becomes the true post-shift plate at the shift
    step. The harness supplies the value; it is privileged by construction.

    With ``standin`` and ``policy`` it also records, at every decision step from the shift on,
    the stand-in's chunk for the aim it executes (O5 compares it with the executed commands)."""

    def __init__(
        self,
        predict,
        standardiser,
        estimates,
        fk,
        bounds,
        *,
        switch_step,
        plate_xy,
        standin=None,
        policy=None,
        decision_steps=wc.DECISION_STEPS,
    ):
        super().__init__(predict, standardiser, estimates, fk, bounds)
        self.switch_step = int(switch_step)
        self.true_plate = np.asarray(plate_xy, np.float64).reshape(2)
        self.standin, self.policy = standin, policy
        self.decision_steps = frozenset(int(s) for s in decision_steps)
        self.chunks: dict[int, list] = {}
        self.last_grasp = (-1.0, -1.0)

    def advance(self, result):
        if result.applied_action is not None:
            self.last_grasp = tuple(float(v) for v in result.applied_action[12:14])
        return None

    def act(self, observation, step):
        if int(step) == self.switch_step:
            self.estimates[2:] = self.true_plate
        if (
            self.standin is not None
            and int(step) in self.decision_steps
            and int(step) >= self.switch_step
        ):
            requested, _applied, feasible = self.standin.chunks(
                self.policy,
                rt2.state_of(observation),
                int(step),
                self.estimates[None],
                last_grasp=self.last_grasp,
            )
            self.chunks[int(step)] = {
                "requested_free": requested[0][:, FREE].tolist(),
                "feasible": bool(feasible[0]),
            }
        return super().act(observation, step)


def relative_chunk_errors(chunks: dict, executed_free) -> list[float]:
    """O5: per decision, the mean over the chunk's commands of || stand-in - executed || (free
    7-vectors), divided by the median norm of all executed commands of the attempt."""
    executed = np.asarray(executed_free, np.float64)
    if executed.ndim != 2 or executed.shape[1] != len(FREE) or not len(executed):
        return []
    scale = float(np.median(np.linalg.norm(executed, axis=1)))
    out = []
    for step, record in sorted(chunks.items()):
        chunk = np.asarray(record["requested_free"], np.float64)
        if not record["feasible"] or step + len(chunk) > len(executed) or scale <= 0:
            continue
        diff = np.linalg.norm(chunk - executed[step : step + len(chunk)], axis=1).mean()
        out.append(float(diff / scale))
    return out
