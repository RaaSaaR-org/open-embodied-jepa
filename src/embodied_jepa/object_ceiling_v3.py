"""NON-LEARNED object-aware privileged MuJoCo-rollout ceiling, version 3 (TASK-051).

Version 2 (``object_ceiling_v2``, TASK-049) succeeded on 5/8 fresh wide-jitter
development resets (gate: 6, with 7 grasps) while the scripted collector succeeded 8/8.
Every failure was the same mode: during the ``close`` phase the closing fingers ejected
the apple.

TASK-051's TRAIN-side diagnosis (``docs/experiments/apple_grasp_closure_diagnosis.md``)
isolated the mechanism. The Dex3 synergy needs eleven commands to travel from open to
closed under the embodiment's joint-rate limit (2.0 rad/s x 0.05 s = 0.1 rad per command
against a 1.1 rad largest per-joint travel), and v2 leaves all six right-arm degrees of
freedom under CEM control throughout those eleven commands. The v2 close cost is nearly
flat there: the palm-to-grasp-point term is dominated by a ~6.4 cm *blocked* height
error, so a 1 cm lateral change alters it by under 0.8 mm; the drift guard exempts the
first centimetre outright; a blocked palm reaches almost the same position over the
horizon whether the vertical command is 0 or -0.5; and the apple has not moved yet, so
the disturbance term is zero for every candidate. Inside that flat region the CEM's
proposal noise decides, and the palm moves 0.05-1.22 cm laterally and 0.40-1.85 cm
vertically while the fingers are still opening (the scripted collector, which never
ejects the apple, moves 0.20-0.22 cm and 0.97-0.99 cm). On the runs that eject, first
hand-apple contact therefore lands at close command 4-8 instead of 10-11: a still-open,
still-moving thumb strikes the apple first and sweeps it 2.5-9.1 cm out of the hand.

Version 3 changes the ``close`` phase, and only the ``close`` phase:

``close``   the right arm holds the lateral position and the orientation achieved by the
            (xy-centred) v2 descent -- the lateral and rotational deltas are pinned to
            zero, so no proposal noise can shear the object -- and the CEM plans the one
            remaining degree of freedom, the vertical delta, restricted to a descent:
            ``[-close_descent_bound, 0]``. The palm may therefore keep sinking around the
            apple as the fingers curl off the table (which the diagnosis shows is
            necessary: a frozen palm ejects the apple by 17 cm) but may never rise while
            the hand is shutting, and the planner can still choose a smaller command when
            contact makes a larger one infeasible (a fixed press trips the embodiment's
            measured joint-velocity guard). The grasp command and the phase length are
            v2's: the eleven-command rate-limited closure is kept deliberately, since
            slowing the synergy ramp makes the ejection worse, not better.

Everything else -- the v2 xy-weighted descent and its blocked-descent rule, the close
cost, the lift/transport carry costs, the release predictor, the exactly-probed release,
every other phase transition, the twin and both runtime rollout-parity checks -- is
inherited unchanged. The CEM still runs on every close command, so the parity checks
stay non-vacuous. Simulator truth enters only this explicitly labelled diagnostic; it is
never a model input and is not registered in ``MODELS``.

The closure needs no prediction of any kind: it is expressed in the commanded palm delta
and the grasp command alone. A learned controller has to decide when to start the close
and whether it succeeded, not how to steer during it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embodied_jepa.contracts import ContractError
from embodied_jepa.object_ceiling_v2 import ObjectCeilingV2Config, ObjectCeilingV2Controller

OBJECT_CEILING_V3_LABEL = (
    "NON-LEARNED object-aware privileged MuJoCo-rollout ceiling v3 (simulator object state "
    "in cost, phase transitions and release predictor; caged grasp closure); not a learned "
    "result"
)
CEILING_VERSION = 3


@dataclass(frozen=True)
class ObjectCeilingV3Config(ObjectCeilingV2Config):
    # Close: the caged closure. The lateral and rotational right-arm deltas are pinned to
    # zero and the vertical delta is planned in [-close_descent_bound, 0], so the palm may
    # only sink while the hand shuts. The grasp schedule is v2's (a step to closed, which
    # the embodiment's joint-rate limit spreads over eleven commands).
    close_descent_bound: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if not 0 < self.close_descent_bound <= self.arm_bound:
            raise ContractError("close_descent_bound must lie in (0, arm_bound]")


class ObjectCeilingV3Controller(ObjectCeilingV2Controller):
    """TASK-051 object-aware ceiling v3 (see the module docstring). NON-LEARNED."""

    def __init__(self, rollout_model, live_robot, config, *, acknowledge_privileged_ceiling=False):
        super().__init__(
            rollout_model,
            live_robot,
            config,
            acknowledge_privileged_ceiling=acknowledge_privileged_ceiling,
        )
        if not isinstance(config, ObjectCeilingV3Config):
            raise ContractError("object-aware ceiling v3 requires ObjectCeilingV3Config")

    def _bounds(self):
        lower, upper = super()._bounds()
        if self.phase.name != "close":
            return lower, upper
        # Cage the apple: hold the achieved lateral pose and orientation, sink only.
        lower[6:8] = upper[6:8] = np.float32(0.0)
        lower[9:12] = upper[9:12] = np.float32(0.0)
        lower[8], upper[8] = np.float32(-self.config.close_descent_bound), np.float32(0.0)
        return lower, upper

    def summary(self):
        return super().summary() | {
            "control_label": OBJECT_CEILING_V3_LABEL,
            "ceiling_version": CEILING_VERSION,
        }
