"""TASK-073: the mid-episode plate shift, a harness hook (simulation-only diagnostic condition).

The plate is a static body (no joint): ``simulation.reset`` places it by writing the model's body
position. The hook does the same thing once, at a declared post-look step: it writes the plate's
``body_pos`` and calls ``mj_forward``, then records what it did. It runs in the harness, outside
every controller's ``act()``: it wraps the robot's ``observe`` for one attempt, and the
observation of the shift step is the first one taken after the move, so its image shows the
plate where it now is. The latched scorer and the at-rest record read the plate's live position,
so scoring after the shift uses the post-shift plate.

Owner D1 (2026-09-29): this is a simulation-only, diagnostic condition on apple-to-plate-v2, not
a new benchmark. NumPy at import; MuJoCo is reached only through the robot passed in.
"""

from __future__ import annotations

import numpy as np

from embodied_jepa.contracts import ContractError

PLATE_BODY = "plate"
PLATE_Z = 0.746  # simulation.reset: the plate body height


class GuardError(ContractError):
    """A plate-shift guard failed (a void, never a counted failure)."""


def _plate_contacts(sim) -> list[str]:
    """Names of the geoms touching a plate geom right after the move (should be none)."""
    model, data = sim.model, sim.data
    plate_id = model.body(PLATE_BODY).id
    touching = []
    for contact in data.contact[: data.ncon]:
        bodies = (int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2]))
        if plate_id in bodies:
            other = contact.geom2 if bodies[0] == plate_id else contact.geom1
            name = model.geom(int(other)).name
            if name != "table":
                touching.append(name)
    return touching


def move_plate(sim, shift_xy) -> dict:
    """Move the static plate by ``shift_xy`` (world, m) now: ``body_pos`` write + ``mj_forward``.

    Refuses a non-finite or 3-d shift and a move that leaves the plate touching anything other
    than the table (a teleport into the hand or the apple would be a physics artefact)."""
    shift = np.asarray(shift_xy, np.float64)
    if shift.shape != (2,) or not np.isfinite(shift).all():
        raise GuardError("a plate shift is a finite world-frame xy vector")
    body = sim.model.body(PLATE_BODY)
    before = np.asarray(body.pos, np.float64).copy()
    if not np.isclose(before[2], PLATE_Z):
        raise GuardError("the plate is not at its reset height")
    body.pos[:2] = before[:2] + shift
    sim.mj.mj_forward(sim.model, sim.data)
    after = np.asarray(sim.data.body(PLATE_BODY).xpos, np.float64).copy()
    if not np.allclose(after[:2], before[:2] + shift, atol=1e-12):
        raise GuardError("the plate did not move to the requested position")
    touching = _plate_contacts(sim)
    if touching:
        raise GuardError(f"the shifted plate touches {sorted(set(touching))}")
    return {
        "before_xy": before[:2].tolist(),
        "after_xy": after[:2].tolist(),
        "shift_xy": shift.tolist(),
        "sim_time": float(sim.data.time),
    }


class PlateShift:
    """Install on a robot after ``reset_and_look``; the plate moves once, just before the
    observation of post-look step ``step`` is taken. ``remove()`` restores ``observe``.

    The hook counts the robot's ``observe`` calls after installation: the attempt loop observes
    once per command step (then once per settle step), so call ``k`` is step ``k``."""

    def __init__(self, robot, step: int, shift_xy):
        if type(step) is not int or step < 1:
            raise GuardError("the shift step is a positive post-look step")
        self.robot = robot
        self.step = step
        self.shift_xy = np.asarray(shift_xy, np.float64).copy()
        self.calls = 0
        self.log: dict | None = None
        self._original = robot.observe

        def observe():
            if self.calls == self.step:
                self.log = move_plate(robot.sim, self.shift_xy) | {"step": self.step}
            self.calls += 1
            return self._original()

        robot.observe = observe

    @property
    def applied(self) -> bool:
        return self.log is not None

    def remove(self) -> None:
        if self.robot.observe is not self._original:
            del self.robot.observe  # the instance attribute; the class method is back
