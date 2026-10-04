"""TASK-076 simulation workers: one attempt of any arm, under TASK-074's step-300 condition, with no
move (cohort U), or under one of K-pred's plate motions (cells M-a, M-b and A).

Protocol ``docs/experiments/apple_plate_twin_v2.md`` §3. The worker is TASK-075's
(``obs_ceiling_v2_runtime.worker_init``, itself TASK-074's and TASK-073's): the v2 robot, P-3's
private kinematics, the bounds, P-3 and the perturber. Nothing pinned by an earlier manifest is
edited: the new arms live here.

* **The aim controller** (``AimController``) runs P-3 unchanged until 405. At every decision step
  (405, 421, ..., 485) its aim strategy names a target and e9's place primitive
  (``place_planner.PlacePrimitive``) is re-targeted, keeping its phase clock; an unreachable
  target falls back to the previous target, then to P-3's post-look plate estimate (TASK-074's
  rule). Strategies: a fixed function of the step (H-handover, H-clock, H-stale, the planted
  error, H-now, H-final on M), R-plate's reading (H-twin, H-floor), H-cv's line, H-rule's fixed
  point and H-final(A)'s privileged look-ahead.
* **K-pred's plate hook** (``CellMotion``) moves the plate at every step's observation, outside
  every controller's ``act`` (a ``body_pos`` write plus ``mj_forward`` through
  ``plate_shift.move_plate``, which refuses a move that leaves the plate touching anything but
  the table). A refused move ends the attempt as a counted failure. The hook also records the
  executed right-palm xy (``PalmFK`` on the observation's joint state) and every apple-plate
  contact from s0 on.
* **H-final(A)'s look-ahead** (``LookaheadAim``) clones the live state (``sim_selector.Brancher``
  plus the plate's ``body_pos`` and the hook's history), rolls the place primitive aimed at g_k to
  s1 with the rule active, and iterates g_{k+1} = the plate at s1 (privileged, the ceiling).

Who reads what follows TASK-071's rule: the reset truth goes to the harness only; the plate
schedule (H-handover, H-now, H-final) is a harness value. NumPy at import; MuJoCo and torch are
imported by the worker initializer.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa import first_policy_v2_runtime as rt2
from embodied_jepa import hybrid_selection as hs
from embodied_jepa import lewm_planner_v2 as lp
from embodied_jepa import lewm_planner_v2_runtime as lrt
from embodied_jepa import obs_ceiling_v2_runtime as ort
from embodied_jepa import place_planner as pp
from embodied_jepa import plate_shift as ps
from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa import wm_critic_v2_runtime as rtm
from embodied_jepa.contracts import ContractError

_T: dict = {}
DECISION_SET = frozenset(pt.DECISION_STEPS)
PALM_FROM = pt.TRANSFER_START - 8  # the hook records the palm from here (H-rule needs d - L)
KEY_STEPS = (pt.TRANSFER_START, 483, 485, 523, 524, 525)


def worker_init(config: dict) -> None:
    """``config``: ``p3_checkpoint``, ``torch_threads`` and optional ``r_plate``,
    ``r_plate_floor`` (npz paths) with ``r_plate_sha256``, ``r_plate_floor_sha256``."""
    ort.worker_init(config)
    _T.clear()
    _T["config"] = dict(config)


def _readout(name: str):
    """A RidgeReadout from the configured npz, checked against its recorded sha256 (G-readout)."""
    from embodied_jepa.models.latent_critic import RidgeReadout

    if name not in _T:
        with np.load(_T["config"][name]) as data:
            readout = RidgeReadout.from_state({k: data[k] for k in data.files})
        want = _T["config"].get(f"{name}_sha256")
        if want is not None and readout.sha256() != want:
            raise lp.GuardError(f"G-readout: {name} differs from its recorded sha256")
        _T[name] = readout
    return _T[name]


def floor_encoder():
    from embodied_jepa import pretrained_encoder as pe

    if "floor" not in _T:
        _T["floor"] = pe.random_init()
    return _T["floor"]


# ----- contacts and K-pred's plate hook ---------------------------------------------------------
def apple_plate_contact(sim) -> bool:
    """Whether any apple geom touches any plate geom now."""
    model, data = sim.model, sim.data
    apple, plate = model.body("apple").id, model.body(ps.PLATE_BODY).id
    for contact in data.contact[: data.ncon]:
        bodies = {int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])}
        if bodies == {apple, plate}:
            return True
    return False


def plate_xy_now(sim) -> np.ndarray:
    return np.asarray(sim.model.body(ps.PLATE_BODY).pos[:2], np.float64).copy()


class CellMotion:
    """K-pred's plate motion for one attempt. Install after ``reset_and_look``: observe call k is
    post-look step k. At call t it first moves the plate to the cell's position for t (a refused
    move raises ``plate_shift.GuardError``, whose message starts with ``SHIFT_BLOCKED_PREFIX``),
    then observes, then records the executed palm xy and any apple-plate contact.

    ``cell``: ``{"name", "s0", "s1"}`` plus ``vector`` (M) or ``kappa`` and ``L`` (A)."""

    def __init__(self, robot, fk, cell: dict):
        self.robot, self.fk = robot, fk
        self.name = str(cell["name"])
        self.s0, self.s1 = int(cell["s0"]), int(cell["s1"])
        self.kind = "A" if self.name == "A" else "M"
        if self.kind == "A":
            pt.check_kappa(cell["kappa"])
            pt.check_lag(cell["L"])
            self.kappa, self.lag, self.vector = float(cell["kappa"]), int(cell["L"]), None
        else:
            self.kappa, self.lag = None, None
            self.vector = np.asarray(cell["vector"], np.float64).reshape(2)
        self.base = plate_xy_now(robot.sim)
        self.state = {
            "calls": 0,
            "palm": {},
            "plate": {},
            "first_contact": None,
            "contact_steps": 0,
            "refused": None,
            "moves": 0,
        }
        self._original = robot.observe
        robot.observe = self._observe

    # the rule
    def target(self, step: int) -> np.ndarray:
        if step <= self.s0:
            return self.base.copy()
        if self.kind == "M":
            return pt.constant_velocity_xy(self.base, self.vector, step, self.s0, self.s1)
        return pt.palm_driven_xy(
            self.base, self.state["palm"], step, self.kappa, self.lag, self.s0, self.s1
        )

    def final_scheduled(self) -> np.ndarray:
        """M: the scheduled final position (H-final's aim on M; a harness value)."""
        if self.kind != "M":
            raise ContractError("only an M cell has a scheduled final position")
        return self.base + self.vector

    def current(self) -> np.ndarray:
        return plate_xy_now(self.robot.sim)

    def apply(self, step: int) -> None:
        if step <= self.s0 or step > self.s1:
            return
        delta = self.target(step) - self.current()
        if not np.any(delta):
            return
        try:
            ps.move_plate(self.robot.sim, delta)
        except ps.GuardError as error:
            self.state["refused"] = {"step": int(step), "reason": str(error)}
            raise
        self.state["moves"] += 1

    def record(self, step: int, observation) -> None:
        s = self.state
        if step >= PALM_FROM:
            s["palm"][int(step)] = self.fk.pose9(rt2.state_of(observation))[:2].copy()
        if step >= self.s0:
            s["plate"][int(step)] = self.current()
            if apple_plate_contact(self.robot.sim):
                s["contact_steps"] += 1
                if s["first_contact"] is None:
                    s["first_contact"] = int(step)

    def _observe(self):
        t = self.state["calls"]
        self.apply(t)
        observation = self._original()
        self.record(t, observation)
        self.state["calls"] += 1
        return observation

    def snapshot(self) -> dict:
        return copy.deepcopy(self.state)

    def restore(self, saved: dict) -> None:
        self.state = copy.deepcopy(saved)

    def remove(self) -> None:
        if self.robot.observe is not self._original:
            del self.robot.observe

    def summary(self) -> dict:
        s = self.state
        plate, palm = s["plate"], s["palm"]

        def at(table, step):
            v = table.get(int(step))
            return None if v is None else [float(x) for x in v]

        out = {
            "cell": self.name,
            "s0": self.s0,
            "s1": self.s1,
            "kappa": self.kappa,
            "L": self.lag,
            "base_xy": self.base.tolist(),
            "plate_at": {str(k): at(plate, k) for k in (*KEY_STEPS, self.s1)},
            "palm_at": {str(k): at(palm, k) for k in (403, 405, 483, 485, 523, self.s1)},
            "first_contact_step": s["first_contact"],
            "contact_steps": s["contact_steps"],
            "contact_before_s1": s["first_contact"] is not None and s["first_contact"] < self.s1,
            "refused": s["refused"],
            "moves": s["moves"],
            "remaining_cm": None,
            "m2_cm": None,
        }
        if 485 in plate and self.s1 in plate:
            out["remaining_cm"] = 100.0 * float(np.linalg.norm(plate[self.s1] - plate[485]))
        if 483 in palm and 485 in palm:
            out["m2_cm"] = pt.m2_cm(palm[483], palm[485], pt.CELL_A["kappa"])
        if self.lag is not None:
            u0 = max(485 - self.lag, self.s0)
            u1 = self.s1 - self.lag
            if u0 in palm and u1 in palm:
                out["palm_remaining_cm"] = 100.0 * float(np.linalg.norm(palm[u1] - palm[u0]))
        return out


# ----- the aim controller and its strategies ------------------------------------------------------
class AimController(rt2.LearnedController):
    """P-3 until 405, then e9's place primitive aimed by ``aim`` at every decision step.

    ``aim(controller, observation, step, record) -> xy or None``. The controller records its own
    executed palm xy from 397 on (proprioception through its private PalmFK)."""

    def __init__(self, predict, standardiser, estimates, fk, bounds, *, arm: str, aim):
        super().__init__(predict, standardiser, estimates, fk, bounds)
        self.arm, self.aim = arm, aim
        self.apple = np.asarray(self.estimates[:2], np.float64).copy()
        self.anchor = np.asarray(self.estimates[2:], np.float64).copy()  # post-look plate
        self.primitive: pp.PlacePrimitive | None = None
        self.decisions: list[dict] = []
        self.palm: dict[int, np.ndarray] = {}
        self.last_grasp = (-1.0, -1.0)

    def decide(self, observation, step: int) -> None:
        started = time.perf_counter()
        record: dict = {"step": int(step), "fallback": False}
        target = self.aim(self, observation, step, record)
        if self.primitive is None:
            self.primitive = pp.PlacePrimitive(self.fk, self.apple, float(self.last_grasp[1]))
        applied = None
        for candidate in (target, self.primitive.target, self.anchor):
            if candidate is None:
                continue
            try:
                self.primitive.retarget(candidate, step)
            except ContractError:
                continue
            applied = candidate
            break
        if applied is None:
            raise ContractError("the aim controller has no reachable target, not even the anchor")
        record["fallback"] = target is None or applied is not target
        record["target"] = np.asarray(applied, np.float64).tolist()
        record["seconds"] = time.perf_counter() - started
        self.decisions.append(record)

    def act(self, observation, step):
        step = int(step)
        state = rt2.state_of(observation)
        if step >= PALM_FROM:
            self.palm[step] = self.fk.pose9(state)[:2].copy()
        if step < pt.TRANSFER_START:
            return super().act(observation, step)
        if step in DECISION_SET:
            self.decide(observation, step)
        if self.primitive is None:
            raise ContractError("the aim controller reached the place phase without a decision")
        return self.primitive.command(state)

    def advance(self, result):
        if result.applied_action is not None:
            self.last_grasp = tuple(float(v) for v in result.applied_action[12:14])
        if self.primitive is not None:
            self.primitive.advance(result)
        return None


class FixedAim:
    """A function of the step: H-handover (+ a planted error), H-clock, H-stale, H-now, H-final
    on an M cell."""

    def __init__(self, function):
        self.function = function

    def __call__(self, ctl, observation, step, record):  # noqa: ARG002
        return np.asarray(self.function(int(step)), np.float64).reshape(2)


class TwinAim:
    """R-plate's reading of the current onboard frame (H-twin; H-floor with the floor encoder
    and readout). The full tokens are encoded on the CPU, batch size 1."""

    def __init__(self, encoder, readout):
        self.encoder, self.readout = encoder, readout
        self.readings: list[tuple[int, np.ndarray]] = []

    def read(self, observation, step: int) -> np.ndarray:
        frame = np.asarray(observation.images[fp2.CAMERA][0])
        tokens, _pooled = hs.encode_frame(self.encoder, frame)
        reading = np.asarray(self.readout.predict(tokens), np.float64).reshape(2)
        self.readings.append((int(step), reading))
        return reading

    def __call__(self, ctl, observation, step, record):  # noqa: ARG002
        reading = self.read(observation, step)
        record["reading"] = reading.tolist()
        return reading


def cv_extrapolate(readings, s1: int) -> np.ndarray:
    """H-cv: a least-squares line per axis through ``(step, xy)`` readings, evaluated at s1; with
    one reading, the reading."""
    steps = np.asarray([s for s, _ in readings], np.float64)
    xy = np.asarray([r for _, r in readings], np.float64).reshape(-1, 2)
    if len(steps) == 1:
        return xy[0].copy()
    design = np.stack([np.ones_like(steps), steps], axis=1)
    coef, *_ = np.linalg.lstsq(design, xy, rcond=None)
    return np.array([1.0, float(s1)]) @ coef


class CvAim(TwinAim):
    def __init__(self, encoder, readout, s1: int):
        super().__init__(encoder, readout)
        self.s1 = int(s1)

    def __call__(self, ctl, observation, step, record):  # noqa: ARG002
        reading = self.read(observation, step)
        record["reading"] = reading.tolist()
        target = cv_extrapolate(self.readings, self.s1)
        record["extrapolated"] = target.tolist()
        return target


def standin_palm_xy(standin, fk, apple_xy, state, step: int, target_xy, *, last_grasp, horizon):
    """The right palm xy after ``horizon`` commands of the place primitive aimed at ``target_xy``
    from ``state`` at ``step``, on the kinematic stand-in (object-free, ideal tracking; TASK-073's
    ``KinematicStandIn``). None if the release pose is unreachable or the stand-in refuses."""
    state = np.asarray(state, np.float64).reshape(-1)
    if horizon <= 0:
        return fk.pose9(state)[:2].copy()
    primitive = pp.PlacePrimitive(fk, apple_xy, accepted_grasp=float(last_grasp[1]))
    try:
        primitive.retarget(target_xy, step)
    except ContractError:
        return None
    s = state.copy()
    tgt = standin.initial_targets(s, last_grasp)
    for _ in range(int(horizon)):
        command = np.clip(primitive.command(s), standin.lower, standin.upper).astype(np.float32)
        out = standin.project(command, s[: standin.joints], tgt)
        if out is None:
            return None
        applied, tgt = out
        q = tgt.astype(np.float64)
        v = (q - s[: standin.joints]) / standin.dt
        s = np.concatenate((q, v)).astype(np.float32).astype(np.float64)
        primitive.advance(_Applied(applied))
    return fk.pose9(s)[:2].copy()


class _Applied:
    def __init__(self, applied):
        self.applied_action, self.reason = applied, None


class RuleAim(TwinAim):
    """H-rule (cell A only; not a world model, knows the rule): the fixed point of
    g = p + kappa (palm_g(s1 - L) - palm(max(d - L, s0))) from H-twin's reading p, the robot's own
    executed palm history and the stand-in's palm path under aim g; iterated to the look-ahead's
    tolerance, at most its iteration cap."""

    def __init__(self, encoder, readout, standin, cell: dict, tolerance_m: float, max_iter: int):
        super().__init__(encoder, readout)
        self.standin = standin
        self.kappa, self.lag = float(cell["kappa"]), int(cell["L"])
        self.s0, self.s1 = int(cell["s0"]), int(cell["s1"])
        self.tolerance, self.max_iter = float(tolerance_m), int(max_iter)

    def __call__(self, ctl, observation, step, record):
        p = self.read(observation, step)
        record["reading"] = p.tolist()
        ref = ctl.palm[max(int(step) - self.lag, self.s0)]
        state = rt2.state_of(observation)
        horizon = self.s1 - self.lag - int(step)
        g, steps, converged = p.copy(), [], False
        for _ in range(self.max_iter):
            palm_end = standin_palm_xy(
                self.standin,
                ctl.fk,
                ctl.apple,
                state,
                step,
                g,
                last_grasp=ctl.last_grasp,
                horizon=horizon,
            )
            if palm_end is None:
                steps.append({"stopped": "unreachable_or_refused"})
                break
            nxt = p + self.kappa * (palm_end - ref)
            gap = float(np.linalg.norm(nxt - g))
            steps.append({"g": nxt.tolist(), "gap_cm": 100.0 * gap})
            g = nxt
            if gap <= self.tolerance:
                converged = True
                break
        record["rule"] = {"iterations": steps, "converged": converged}
        return g


class PlateBrancher:
    """``sim_selector.Brancher``'s save/branch/restore, plus the plate's ``body_pos`` and the
    cell hook's history (the plate is a static body: its position lives in the model)."""

    def __init__(self, robot, hook: CellMotion):
        from embodied_jepa.sim_selector import Brancher

        self.robot, self.hook = robot, hook
        self.inner = Brancher(robot)

    def save(self) -> dict:
        return {
            "inner": self.inner._save(),
            "plate_pos": np.asarray(self.robot.sim.model.body(ps.PLATE_BODY).pos).copy(),
            "hook": self.hook.snapshot(),
        }

    def restore(self, saved: dict) -> None:
        # the model's plate position, then the whole MjData copy (exact, as TASK-074's branches)
        self.robot.sim.model.body(ps.PLATE_BODY).pos[:] = saved["plate_pos"]
        self.inner._restore(saved["inner"])
        self.hook.restore(saved["hook"])

    def rollout(self, ctl, observation, step: int, aim_xy, bounds, s1: int) -> dict:
        """The place primitive aimed at ``aim_xy`` from ``step`` (the live observation) to s1,
        with the cell's rule; returns the plate at s1 (or where the branch stopped)."""
        lower, upper = bounds
        try:
            branch = pp.BranchPlace(ctl.fk, ctl.apple, aim_xy, step, ctl.last_grasp[1])
        except ContractError:
            return {"plate": None, "stopped": "unreachable", "steps": 0}
        obs, stopped, j = observation, None, int(step)
        for j in range(int(step), int(s1)):
            command = np.clip(np.asarray(branch.act(obs, j), np.float32), lower, upper)
            try:
                projection = self.robot.project_candidates(command[None, None, None])
            except ContractError as error:
                if str(error) not in rt2.GUARD_REFUSALS:
                    raise
                stopped = "guard_refusal"
                break
            if not bool(projection.feasible[0, 0]):
                stopped = "infeasible_command"
                break
            result = self.robot.execute(projection.actions[0, 0, 0])
            if result.applied_action is None:
                stopped = result.reason or result.status or "rejected"
                break
            branch.advance(result)
            try:
                self.hook.apply(j + 1)
            except ps.GuardError:
                stopped = "plate_move_refused"
                break
            obs = type(self.robot).observe(self.robot)  # bypasses the live hook's wrapper
            self.hook.record(j + 1, obs)
        return {"plate": self.hook.current(), "stopped": stopped, "steps": j + 1 - int(step)}


class LookaheadAim:
    """H-final(A): g0 = the true plate now; g_{k+1} = the plate at s1 after the place primitive
    aimed at g_k runs in cloned state; stop at |g_{k+1} - g_k| <= tolerance (aim at g_{k+1}), or
    after ``max_iter`` iterations (aim at the last iterate, logged as not converged).
    Privileged: it holds the robot (the ceiling arm)."""

    def __init__(self, robot, hook: CellMotion, bounds, tolerance_m: float, max_iter: int):
        self.robot, self.hook, self.bounds = robot, hook, bounds
        self.tolerance, self.max_iter = float(tolerance_m), int(max_iter)
        self.brancher = PlateBrancher(robot, hook)
        self.branch_steps = 0

    def __call__(self, ctl, observation, step, record):
        sim = self.robot.sim
        blank = np.zeros((sim.height, sim.width, 3), np.uint8)
        saved = self.brancher.save()
        g = self.hook.current()
        steps, converged, stopped = [], False, None
        sim.render = lambda camera="onboard_rgb": blank.copy()  # noqa: ARG005
        try:
            for _ in range(self.max_iter):
                self.brancher.restore(saved)
                out = self.brancher.rollout(ctl, observation, step, g, self.bounds, self.hook.s1)
                self.branch_steps += out["steps"]
                if out["plate"] is None:
                    stopped = out["stopped"]
                    steps.append({"stopped": stopped})
                    break
                nxt = np.asarray(out["plate"], np.float64)
                gap = float(np.linalg.norm(nxt - g))
                steps.append({"g": nxt.tolist(), "gap_cm": 100.0 * gap, "stopped": out["stopped"]})
                g = nxt
                if gap <= self.tolerance:
                    converged = True
                    break
        finally:
            del sim.render
            self.brancher.restore(saved)
        record["lookahead"] = {"iterations": steps, "converged": converged, "stopped": stopped}
        return g


# ----- one attempt ------------------------------------------------------------------------------
def _controller(task: dict, truth: dict, hook):  # noqa: ARG001
    W = rtm._W
    arm = task["arm"]
    robot, fk, bounds = W["robot"], W["fk"], W["bounds"]
    _model, standardiser = W["p3"]
    estimates = np.asarray(task.get("estimates", np.zeros(4)), np.float64)
    common = (W["predict"], standardiser, estimates, fk, bounds)
    plate_at = rtm.plate_schedule(task["reset"], task.get("shift"))
    cell = task.get("cell")
    tolerance = pt.LOOKAHEAD["tolerance_fraction_of_tau"] * float(task.get("tau_cm", 1.0)) / 100.0

    def aim_of(strategy):
        return AimController(*common, arm=arm, aim=strategy)

    if arm == "P-stale":
        return rt2.LearnedController(*common)
    if cell is None:
        if arm == "H-handover":
            planted = np.asarray(task.get("planted_m", [0.0, 0.0]), np.float64)
            return aim_of(FixedAim(lambda s: plate_at(s) + planted))
        if arm == "H-clock":
            targets = {int(k): np.asarray(v, np.float64) for k, v in task["clock"].items()}
            return aim_of(FixedAim(lambda s: targets[s]))
        if arm == "H-stale":
            anchor = estimates[2:].copy()
            return aim_of(FixedAim(lambda s: anchor))  # noqa: ARG005
        if arm == "H-twin":
            return aim_of(TwinAim(rtm.encoder(), _readout("r_plate")))
        if arm == "H-floor":
            return aim_of(TwinAim(floor_encoder(), _readout("r_plate_floor")))
        raise lp.GuardError(f"unknown arm {arm!r}")
    if arm == "H-now":
        return aim_of(FixedAim(lambda s: hook.current()))  # noqa: ARG005
    if arm == "H-twin":
        return aim_of(TwinAim(rtm.encoder(), _readout("r_plate")))
    if arm == "H-cv":
        return aim_of(CvAim(rtm.encoder(), _readout("r_plate"), cell["s1"]))
    if arm == "H-final":
        if hook.kind == "M":
            final = hook.final_scheduled()
            return aim_of(FixedAim(lambda s: final))  # noqa: ARG005
        return aim_of(LookaheadAim(robot, hook, bounds, tolerance, pt.LOOKAHEAD["max_iterations"]))
    if arm == "H-rule":
        if hook.kind != "A":
            raise lp.GuardError("H-rule runs on cell A only")
        return aim_of(
            RuleAim(
                rtm.encoder(),
                _readout("r_plate"),
                rtm.standin(),
                cell,
                tolerance,
                pt.LOOKAHEAD["max_iterations"],
            )
        )
    raise lp.GuardError(f"unknown K-pred arm {arm!r}")


def run_attempt_task(task: dict) -> dict:
    """One attempt. ``task``: ``seed``, ``reset``, ``arm``, ``estimates``, the expected post-look
    frame and state sha256, and one of ``shift`` (TASK-074's step-300 move) or ``cell``
    (K-pred), or neither (cohort U); ``planted_m``, ``clock``, ``tau_cm`` where the arm needs
    them."""
    started = time.monotonic()
    W = rtm._W
    robot = W["robot"]
    truth, scorer, counter, _obs, facts = rt2.reset_and_look(robot, task["seed"], task["reset"])
    out = {
        "seed": task["seed"],
        "arm": task["arm"],
        "post_look_frame_sha256": rtm.frame_sha(facts["post_look_frame"]),
        "post_look_state": facts["post_look_state"].tolist(),
        "truth_xy": rtm.truth_xy(truth),
    }
    try:
        out["render_retries"] = rtm.check_post_look_frame(robot, task, facts["post_look_frame"])
    except lp.GuardError:
        counter.remove()
        robot.stop("frame_mismatch")
        raise
    shift, cell = task.get("shift"), task.get("cell")
    if shift and cell:
        raise lp.GuardError("an attempt has the step-300 shift or a K-pred cell, not both")
    hook = None
    if shift:
        hook = ps.PlateShift(robot, int(shift["step"]), shift["vector"])
    elif cell:
        hook = CellMotion(robot, W["fk"], cell)
    inner = None
    blocked = None
    try:
        inner = _controller(task, truth, hook)
        controller = rtm.Timed(inner)
        record = rt2.run_attempt(
            robot,
            scorer,
            counter,
            controller,
            bounds=W["bounds"],
            max_steps=int(task.get("max_steps", fp2.MAX_POLICY_STEPS)),
            settle_steps=int(task.get("settle_steps", fp2.SETTLE_STEPS)),
            wall_seconds=float(task.get("wall_seconds", pt.CAPS_SECONDS["per_attempt"])),
        )
    except ps.GuardError as error:
        if not str(error).startswith(lp.SHIFT_BLOCKED_PREFIX):
            raise
        blocked = str(error)
    finally:
        if hook is not None:
            hook.remove()
    if blocked is not None:
        out |= {
            "termination_reason": "plate_move_refused" if cell else "shift_blocked",
            "blocked": blocked,
            "complete": False,
            "at_rest": False,
            "latched_success": False,
            "success": False,
            "grasp": None,
            "first_grasp_step": None,
            "first_place_step": None,
            "executed_steps": None,
            "final_distance_cm": None,
            "privileged_ok": True,
            "commands": np.zeros((0, 7), np.float32),
        }
        refused = getattr(hook, "state", {}).get("refused") if cell else None
        out["executed_steps"] = refused["step"] if refused else int(shift["step"])
    else:
        if shift and record["executed_steps"] >= int(shift["step"]) and not hook.applied:
            raise lp.GuardError("G-shift: the plate shift was not applied")
        detail = record.get("at_rest_detail") or {}
        out.update({k: record[k] for k in lrt.KEEP})
        out["final_distance_cm"] = (
            None if detail.get("final_distance_m") is None else 100.0 * detail["final_distance_m"]
        )
        out["privileged_ok"] = rt2.privileged_reads_ok(record)
        out["commands"] = record["commands"]
    out["blocked"] = blocked
    out["decisions"] = getattr(inner, "decisions", []) if inner is not None else []
    if isinstance(hook, CellMotion):
        out["kpred"] = hook.summary()
        aim = getattr(inner, "aim", None)
        if isinstance(aim, LookaheadAim):
            out["kpred"]["lookahead_branch_steps"] = aim.branch_steps
    if isinstance(hook, ps.PlateShift):
        out["shift_log"] = hook.log
    out["seconds"] = time.monotonic() - started
    return out


def run_task(task: dict) -> dict:
    kind = task["kind"]
    if kind == "attempt":
        return run_attempt_task(task)
    if kind == "frame":
        return rtm.run_frame_task(task)
    if kind == "collect":
        return lrt.run_collect_task(task)
    if kind == "views":
        return ort.run_views_task(task)
    raise lp.GuardError(f"unknown task kind {kind!r}")
