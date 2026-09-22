"""NON-LEARNED object-aware privileged MuJoCo-rollout ceiling, version 2 (TASK-049).

Version 1 (``object_ceiling``, TASK-047) grasped on 5/8 wide-jitter development resets.
Its post-hoc failures were geometric, not dynamic (rollout parity was exact):

* the descent cost was the 3-D distance to an unreachable grasp point, dominated by
  the blocked height, so it barely rewarded xy re-centring, and the fixed 1 cm
  blocked-descent window was then missed by 0.5-2.5 mm;
* the close phase kept pressing towards that unreachable point and the palm drifted
  3.6 cm behind the apple while the fingers closed;
* a transport or lower phase could hand over to the release anywhere inside the
  scorer's 4 cm plate radius; the apple was released 4.5-4.6 cm off-centre and, on
  the other resets, landed several cm away from where it was released.

Version 2 keeps the v1 twin, planner, bounds, grasp schedule and phase order and
changes only these costs and transitions:

``descend``   separable cost ``w_xy * |xy error| + |z error|``: the xy term keeps full
              weight however large the (blocked) height error is. A height-blocked
              descent ends once the xy error is within ``descend_xy_tolerance_m``, or
              once the xy error itself has stopped improving while within
              ``block_xy_m``.
``close``     keeps v1's pressure towards the grasp point (the hand sinks around the
              apple as the fingers curl) and adds a dead-band guard: palm xy drift of
              more than ``close_xy_margin_m`` from the pose achieved at the start of
              the close is penalised with weight ``close_xy_weight``.
``transport`` a *release predictor* runs once per command: the exact twin executes
``lower``     the release itself -- ``release_commands`` commands with the arms held and
              the right hand opening gradually by ``release_ramp`` per command (the
              collector's opening ramp) -- from the live state and records the apple's
              path. It *predicts a placement* when, for ``landing_dwell_commands``
              consecutive commands, the apple is out of hand contact, at the plate
              support height, within ``place_tolerance_m`` of the plate centre and
              slower than ``landing_rest_speed_m_s``. The phase hands over to
              ``release`` only when the carried apple is in hand contact and within
              ``place_tolerance_m`` of the plate centre *and* a placement is predicted.
              The carry cost is xy-weighted towards the plate centre. A blocked
              transport hands over to ``lower`` (a lower drop changes the landing); a
              blocked lower keeps planning and may stall.
``release``   executes exactly the probed sequence (zero-width CEM bounds), so the
              accepted prediction is what the live release does under exact dynamics.

Every cost and transition is expressed in quantities a learned head could predict:
the palm-apple offset, the palm's own pose (proprioception), the apple height rise,
the apple-plate xy offset, hand contact, and whether the apple comes to rest on the
plate after a gradual open-hand release (the release outcome). Simulator truth enters
only this explicitly labelled diagnostic; it is never a model input and is not
registered in ``MODELS``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embodied_jepa.contracts import ContractError, Observation, RobotState
from embodied_jepa.object_ceiling import (
    BLOCKABLE_PHASES,
    PHASES,
    ObjectCeilingConfig,
    ObjectCeilingController,
    ObjectRollout,
    ObjectRolloutModel,
    features,
)
from embodied_jepa.privileged_rollout import RolloutSnapshot

OBJECT_CEILING_V2_LABEL = (
    "NON-LEARNED object-aware privileged MuJoCo-rollout ceiling v2 (simulator object state "
    "in cost, phase transitions and release predictor); not a learned result"
)
CEILING_VERSION = 2
# Phases in which the release predictor runs and may hand over to the release.
RELEASE_CHECK_PHASES = ("transport", "lower")


@dataclass(frozen=True)
class ObjectCeilingV2Config(ObjectCeilingConfig):
    # Descend: xy weight of the separable descent cost; xy tolerance at which a
    # height-blocked descent ends at once; ``block_xy_m`` bounds the xy error at which a
    # height-blocked descent whose xy error has also stalled (< xy_block_progress_m
    # improvement over block_commands) may end.
    descend_xy_weight: float = 3.0
    descend_xy_tolerance_m: float = 0.005
    block_xy_m: float = 0.012
    xy_block_progress_m: float = 0.001
    # Close: weight and dead band of the guard against palm xy drift from the pose
    # achieved when the close began.
    close_xy_weight: float = 1.0
    close_xy_margin_m: float = 0.01
    # Transport/lower: xy weight of the carry cost towards the plate centre.
    place_xy_weight: float = 4.0
    # Release: the right hand opens gradually by ``release_ramp`` per command (the
    # collector's own opening ramp) with the arms held still, for ``release_commands``
    # commands (then retreat); the release probe executes exactly this sequence.
    release_ramp: float = 0.08
    release_commands: int = 60
    # Predicted placement (gates the hand-over to release): during the probed release
    # the apple is, for ``landing_dwell_commands`` consecutive commands, out of hand
    # contact, at the plate support height (+-landing_support_tolerance_m; the table
    # lies only 1.2 cm lower), within ``place_tolerance_m`` of the plate centre and
    # slower than ``landing_rest_speed_m_s``.
    place_tolerance_m: float = 0.03
    landing_support_tolerance_m: float = 0.006
    landing_rest_speed_m_s: float = 0.1
    landing_dwell_commands: int = 4

    def __post_init__(self):
        super().__post_init__()
        if type(self.landing_dwell_commands) is not int or not (
            1 <= self.landing_dwell_commands <= self.release_commands
        ):
            raise ContractError("landing_dwell_commands must lie in [1, release_commands]")
        if not 0 < self.release_ramp <= 2:
            raise ContractError("release_ramp must lie in (0,2]")
        if self.descend_xy_tolerance_m > self.block_xy_m:
            raise ContractError("the immediate descent xy tolerance must not exceed block_xy_m")
        if self.place_tolerance_m > self.plate_radius_m:
            raise ContractError("the place tolerance must lie inside the scorer's plate radius")
        for name in ("descend_xy_weight", "close_xy_weight", "place_xy_weight"):
            if getattr(self, name) <= 0:
                raise ContractError(f"{name} must be positive")


class ObjectRolloutV2Model(ObjectRolloutModel):
    """The v1 exact twin plus a side-effect-free *release probe*.

    ``release_probe`` snapshots the live state and executes one arms-held release
    sequence in the twin, recording the apple's path, speed and hand contact. It records no parity
    candidate and no planning diagnostic, so the v1 runtime parity checks (which
    compare the measured state with the previous CEM search's first steps) are
    unchanged. The probe is simulator truth used only inside this labelled ceiling.
    """

    def release_probe(self, robot_state, right_grasps):
        """Execute arms-held commands with the given right-grasp schedule (left hand open)
        in the twin from the live state; return the apple's path, speed and contact."""
        if not isinstance(robot_state, RobotState) or robot_state.values.shape[0] != 1:
            raise ContractError("release probe requires one RobotState row")
        right_grasps = np.asarray(right_grasps, np.float32)
        if (
            right_grasps.ndim != 1
            or not len(right_grasps)
            or not np.isfinite(right_grasps).all()
            or np.abs(right_grasps).max() > 1
        ):
            raise ContractError("release probe needs a nonempty normalized grasp schedule")
        sim = self.live.sim
        if float(robot_state.timestamps[0]) != float(sim.data.time) or not np.array_equal(
            robot_state.values[0], self._live_state_row()
        ):
            raise ContractError("release probe does not match the planning observation")
        data = self.mj.MjData(sim.model)
        self.mj.mj_copyData(data, sim.model, sim.data)
        self._restore(
            RolloutSnapshot(
                data,
                sim.targets.copy(),
                sim.model.body_pos.copy(),
                float(sim.data.time),
                self._owner,
            )
        )
        executed, apple, speed, contact = 0, [], [], []
        for grasp in right_grasps:
            action = np.zeros(14, np.float32)  # arms held
            action[12:] = (-1.0, grasp)
            self.twin.observe()
            try:
                projection = self.twin.project_candidates(action[None, None, None])
                if not projection.feasible[0, 0]:
                    break
                if self.twin.execute(projection.actions[0, 0, 0]).applied_action is None:
                    break
            except (ContractError, RuntimeError):
                break
            executed += 1
            truth = self.twin.sim.task_truth()
            apple.append(np.asarray(truth["object_position"], np.float64))
            speed.append(float(np.linalg.norm(truth["object_velocity"])))
            contact.append(bool(truth["hand_contact"]))
        row = features(self.twin)
        return {
            "apple_path": np.array(apple).reshape(-1, 3),
            "speed": np.array(speed),
            "contact_path": np.array(contact, bool),
            "apple": row["apple"].copy(),
            "plate": row["plate"].copy(),
            "rest_z": row["rest_z"],
            "contact": row["contact"],
            "dropped": row["dropped"],
            "executed": executed,
            "complete": executed == len(right_grasps),
        }


class ObjectCeilingV2Controller(ObjectCeilingController):
    """TASK-049 object-aware ceiling v2 (see the module docstring). NON-LEARNED."""

    def __init__(self, rollout_model, live_robot, config, *, acknowledge_privileged_ceiling=False):
        super().__init__(
            rollout_model,
            live_robot,
            config,
            acknowledge_privileged_ceiling=acknowledge_privileged_ceiling,
        )
        if not isinstance(config, ObjectCeilingV2Config):
            raise ContractError("object-aware ceiling v2 requires ObjectCeilingV2Config")
        if not isinstance(rollout_model, ObjectRolloutV2Model):
            raise ContractError("object-aware ceiling v2 requires the release-probe twin")
        self.landing = None  # latest release-probe result (live state of this command)
        self.release_probes = 0

    # ----- release predictor ---------------------------------------------------------
    def _placed_at(self):
        """First probed command index that starts a predicted placement, else None."""
        cfg, landing = self.config, self.landing
        if landing is None or not landing["complete"] or landing["dropped"]:
            return None
        path = landing["apple_path"]
        placed = (
            np.logical_not(landing["contact_path"])
            & (np.abs(path[:, 2] - landing["rest_z"]) <= cfg.landing_support_tolerance_m)
            & (np.linalg.norm(path[:, :2] - landing["plate"][:2], axis=1) < cfg.place_tolerance_m)
            & (landing["speed"] < cfg.landing_rest_speed_m_s)
        )
        run = 0
        for index, ok in enumerate(placed):
            run = run + 1 if ok else 0
            if run >= cfg.landing_dwell_commands:
                return index - run + 1
        return None

    def _landing_ok(self):
        return self._placed_at() is not None

    def release_schedule(self):
        """Right-grasp commands of the release phase: a gradual opening from closed."""
        cfg = self.config
        return np.maximum(
            -1.0, 1.0 - cfg.release_ramp * np.arange(1, cfg.release_commands + 1)
        ).astype(np.float32)

    def _bounds(self):
        if self.phase.name != "release":
            return super()._bounds()
        # The release executes exactly the probed sequence: arms held, gradual opening.
        lower = np.zeros(14, np.float32)
        schedule = self.release_schedule()
        lower[12:] = (-1.0, schedule[min(self.phase.commands, len(schedule) - 1)])
        return lower, lower.copy()

    def _probe(self, observation):
        self.landing = self.rollouts.release_probe(observation.state, self.release_schedule())
        self.release_probes += 1

    def step(self, observation, projector):
        self.landing = None
        if (
            self.pending is None
            and self.termination_reason is None
            and self.phase.name in RELEASE_CHECK_PHASES
            and isinstance(observation, Observation)
            and self.steps < self.config.max_steps
        ):
            self._probe(observation)
        decision = super().step(observation, projector)
        if decision.action is not None and self.landing is not None:
            decision.trace["predicted_landing"] = {
                "apple": self.landing["apple"].tolist(),
                "plate_offset_m": float(
                    np.linalg.norm(self.landing["apple"][:2] - self.landing["plate"][:2])
                ),
                "contact": self.landing["contact"],
                "complete": self.landing["complete"],
                "placed_at": self._placed_at(),
                "accepted": self._landing_ok(),
            }
        return decision

    # ----- phase machine -------------------------------------------------------------
    def _enter(self, name, live):
        super()._enter(name, live)
        if name == "descend":
            self.phase.anchor["xy_progress"] = []
        if name == "close":
            # The v1 grasp-point target stays; the achieved palm xy anchors the guard.
            self.phase.anchor["hold_xy"] = live["palm"][:2].copy()

    def _descend_xy_error(self, live):
        target = self._grasp_target(live["apple"], self.config.grasp_height_m)
        return float(np.linalg.norm(live["palm"][:2] - target[:2]))

    def _record_progress(self, live):
        """Append this command's progress measures (once per planning step)."""
        super()._record_progress(live)
        if self.phase.name == "descend":
            self.phase.anchor["xy_progress"].append(self._descend_xy_error(live))

    def _stalled(self, history, threshold):
        cfg = self.config
        return (
            len(history) > cfg.block_commands
            and history[-1 - cfg.block_commands] - history[-1] < threshold
        )

    def _advance(self, live):
        """Apply every satisfied transition on the live state (possibly several)."""
        cfg = self.config
        while True:
            name, commands = self.phase.name, self.phase.commands
            rise = live["apple"][2] - self.start["apple"][2]
            ended, following = None, None
            if name == "approach":
                target = self._grasp_target(live["apple"], cfg.approach_height_m)
                if (
                    np.linalg.norm(live["palm"] - target) < cfg.approach_tolerance_m
                    and live["rotation_error"] < cfg.rotation_tolerance_rad
                ):
                    ended = "reached"
            elif name == "descend":
                target = self._grasp_target(live["apple"], cfg.grasp_height_m)
                xy = self._descend_xy_error(live)
                if np.linalg.norm(live["palm"] - target) < cfg.grasp_tolerance_m:
                    ended = "reached"
                elif self._stalled(self.phase.anchor["progress"], cfg.block_progress_m) and (
                    xy < cfg.descend_xy_tolerance_m
                    or (
                        xy < cfg.block_xy_m
                        and self._stalled(self.phase.anchor["xy_progress"], cfg.xy_block_progress_m)
                    )
                ):
                    ended = "blocked"
            elif name == "close":
                if commands >= cfg.close_commands:
                    ended = "reached"
            elif name == "lift":
                if rise >= cfg.lift_done_m and live["contact"]:
                    ended = "reached"
            elif name in RELEASE_CHECK_PHASES:
                # Release only a carried apple held over the plate centre (the scorer's
                # transport stage needs it) whose probed release places it.
                over = np.linalg.norm(live["apple"][:2] - live["plate"][:2]) < cfg.place_tolerance_m
                if over and live["contact"] and self._landing_ok():
                    ended, following = "release_predicted", "release"
                elif name == "transport" and self._stalled(
                    self.phase.anchor["progress"], cfg.block_progress_m
                ):
                    ended = "blocked"
            elif name == "release":
                if commands >= cfg.release_commands:
                    ended = "reached"
            if ended is None:
                return
            if name in BLOCKABLE_PHASES:
                self.phase.anchor["ended"] = ended
            self._enter(following or PHASES[PHASES.index(name) + 1], live)

    def phase_costs(self, rollout: ObjectRollout, live):
        """Per-step phase cost ``[K,H]`` (meters-equivalent); see the module docstring."""
        cfg, anchor, name = self.config, self.phase.anchor, self.phase.name
        palm, apple = rollout.palm, rollout.apple
        if name not in ("descend", "close", "transport", "lower"):
            return super().phase_costs(rollout, live)
        cost = cfg.rotation_weight * rollout.rotation_error
        if name == "descend":
            error = palm - self._grasp_target(apple, cfg.grasp_height_m)
            cost = cost + cfg.descend_xy_weight * np.linalg.norm(error[..., :2], axis=-1)
            cost = cost + np.abs(error[..., 2])
            cost = cost + cfg.disturbance_weight * np.linalg.norm(
                apple[..., :2] - anchor["apple"][:2], axis=-1
            )
        elif name == "close":
            # v1's pressure towards the grasp point fixed at the close start (the hand
            # sinks around the apple as the fingers curl off the table), plus a dead-band
            # guard against drifting away from the achieved palm xy.
            cost = cost + np.linalg.norm(palm - anchor["target"], axis=-1)
            drift = np.linalg.norm(palm[..., :2] - anchor["hold_xy"], axis=-1)
            cost = cost + cfg.close_xy_weight * np.maximum(0.0, drift - cfg.close_xy_margin_m)
            cost = cost + cfg.disturbance_weight * np.linalg.norm(
                apple[..., :2] - anchor["apple"][:2], axis=-1
            )
        else:
            slip = np.linalg.norm((apple - palm) - anchor["relative"], axis=-1)
            cost = cost + cfg.slip_weight * slip + cfg.contact_penalty * (~rollout.contact)
            cost = cost + cfg.place_xy_weight * np.linalg.norm(
                apple[..., :2] - live["plate"][:2], axis=-1
            )
            if name == "transport":
                start_z = self.start["apple"][2]
                cost = cost + np.maximum(0.0, start_z + cfg.carry_height_m - apple[..., 2])
            else:
                cost = cost + np.abs(apple[..., 2] - anchor["target_z"])
        return cost

    def summary(self):
        return super().summary() | {
            "control_label": OBJECT_CEILING_V2_LABEL,
            "ceiling_version": CEILING_VERSION,
            "release_probes": self.release_probes,
        }
