"""C1 feasibility workers: one attempt under cell A's reactive-plate rule with a single aim
committed at 405 (development only; no world model).

Record ``docs/experiments/apple_lewm_next_v2_c1_feasibility.md``; constants
``lewm_next_c1.py``. The worker is TASK-076's (``plate_twin_v2_runtime.worker_init``: the v2 robot,
P-3, its private kinematics, the bounds), and TASK-076's code is reused unchanged:

* the plate hook is ``plate_twin_v2_runtime.CellMotion`` (cell A: kappa = -0.5, L = 2, s0 = 405,
  s1 = 525), subclassed only to keep the whole plate path and to capture the onboard 112 px frame
  and its plate-hidden render (TASK-075's renderer) at named steps;
* the controller is ``AimController`` with one decision, at 405 (``CommitController``); H-now-reaim
  is TASK-076's ``AimController`` unchanged (decisions 405-485);
* H-final(commit) is TASK-076's ``LookaheadAim`` (privileged cloned-state look-ahead), called once;
* H-rule is TASK-076's ``RuleAim`` fixed point, called once, its aim clipped to the box.

The privileged proxies, the reach arms and the corpus collector read the true plate through the
hook (as H-now did in TASK-076); H-rule and H-sysid read the plate from the image (R-plate on the
CPU, batch 1), and their ``-true`` versions from the hook. NumPy at import; MuJoCo and torch are
imported by the worker initializer.
"""

from __future__ import annotations

import time

import numpy as np

from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa import first_policy_v2_runtime as rt2
from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa import lewm_planner_v2 as lp
from embodied_jepa import lewm_planner_v2_runtime as lrt
from embodied_jepa import plate_shift as ps
from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa import plate_twin_v2_runtime as ptr
from embodied_jepa import wm_critic_v2_runtime as rtm
from embodied_jepa.contracts import ContractError


def worker_init(config: dict) -> None:
    ptr.worker_init(config)


# ----- the hook: cell A, with the whole plate path and frame captures ----------------------------
class C1Motion(ptr.CellMotion):
    """``CellMotion`` on cell A; at each step in ``capture`` it keeps the observation's onboard
    112 px frame and a plate-hidden render of the same state (``obs_ceiling_v2_runtime.capture``).
    Captures live outside ``state``, so the look-ahead's snapshots never copy them."""

    def __init__(self, robot, fk, capture=()):
        super().__init__(robot, fk, dict(c1.RULE))
        self.capture_steps = frozenset(int(s) for s in capture)
        self.captured: dict[int, dict] = {}

    def _observe(self):
        t = self.state["calls"]
        observation = super()._observe()
        if t in self.capture_steps:
            from embodied_jepa import obs_ceiling_v2_runtime as ort

            hidden, _ = ort.capture("onboard112", "plate_hidden")
            self.captured[int(t)] = {
                "visible": np.asarray(observation.images[fp2.CAMERA][0], np.uint8).copy(),
                "hidden": np.asarray(hidden, np.uint8).copy(),
                "plate": self.current().tolist(),
            }
        return observation

    def summary(self) -> dict:
        out = super().summary()
        plate, palm = self.state["plate"], self.state["palm"]
        out["plate_path"] = {str(t): plate[t].tolist() for t in sorted(plate) if t <= self.s1}
        out["palm_at"] |= {str(t): palm[t].tolist() for t in (403, 404, 405) if t in palm}
        if c1.COMMIT_STEP in plate and self.s1 in plate:
            out["remaining_405_cm"] = 100.0 * float(
                np.linalg.norm(plate[self.s1] - plate[c1.COMMIT_STEP])
            )
        if 404 in palm and 405 in palm:
            out["palm_speed_405_cm"] = 100.0 * float(np.linalg.norm(palm[405] - palm[404]))
        if 403 in palm and 405 in palm:
            out["m2_405_cm"] = pt.m2_cm(palm[403], palm[405], c1.RULE["kappa"])
        return out


# ----- the controller: one decision, at 405 -------------------------------------------------------
class CommitController(ptr.AimController):
    """``AimController`` with its only decision at 405; e9's primitive then runs the transfer,
    lower and open with no re-aim."""

    def act(self, observation, step):
        step = int(step)
        state = rt2.state_of(observation)
        if step >= ptr.PALM_FROM:
            self.palm[step] = self.fk.pose9(state)[:2].copy()
        if step < pt.TRANSFER_START:
            return rt2.LearnedController.act(self, observation, step)
        if step == c1.COMMIT_STEP:
            self.decide(observation, step)
        if self.primitive is None:
            raise ContractError("the commit controller reached the place phase without a decision")
        return self.primitive.command(state)


def _box_record(record: dict, g, p, h) -> None:
    a, b = c1.to_box(g, p, h)
    record["box"] = {"p": np.asarray(p).tolist(), "h": np.asarray(h).tolist(), "a": a, "b_m": b}


class GeometricAim:
    """A privileged aim from the true plate p and the proprioceptive palm h at 405: the proxies
    (H-now, N-, shuf-, mean-proxy), a reach candidate (a, b) or the corpus's drawn (a, b)."""

    def __init__(self, arm: str, hook: C1Motion, task: dict):
        self.arm, self.hook, self.task = arm, hook, task

    def __call__(self, ctl, observation, step, record):  # noqa: ARG002
        p, h = self.hook.current(), np.asarray(ctl.palm[int(step)], np.float64)
        a_lo = self.task.get("a_lo")
        if self.arm in c1.PROXY_ARMS:
            foreign = self.task.get("foreign_plate")
            g = c1.proxy_aim(self.arm, p, h, a_lo, foreign_p=foreign)
        elif self.arm in ("reach", "collect"):
            g = c1.from_box(self.task["a"], self.task["b_m"], p, h)
        else:
            raise lp.GuardError(f"unknown geometric arm {self.arm!r}")
        _box_record(record, g, p, h)
        return g


class RuleCommit(ptr.RuleAim):
    """H-rule at 405 (TASK-076's fixed point g = p + kappa (palm_g(s1 - L) - palm(405)) on the
    kinematic stand-in), from R-plate's reading (H-rule) or the true plate (H-rule-true); the final
    aim is clipped to the box around its own plate input and h."""

    def __init__(self, encoder, readout, standin, *, truth_hook, a_lo: float):
        super().__init__(
            encoder, readout, standin, c1.RULE, c1.LOOKAHEAD_TOLERANCE_M, c1.MAX_ITERATIONS
        )
        self.truth_hook, self.a_lo = truth_hook, float(a_lo)

    def __call__(self, ctl, observation, step, record):
        if self.truth_hook is not None:
            p = self.truth_hook.current()
        else:
            p = self.read(observation, step)
            record["reading"] = p.tolist()
        h = np.asarray(ctl.palm[int(step)], np.float64)
        ref = ctl.palm[max(int(step) - self.lag, self.s0)]
        state = rt2.state_of(observation)
        horizon = self.s1 - self.lag - int(step)
        g, steps, converged = p.copy(), [], False
        for _ in range(self.max_iter):
            palm_end = ptr.standin_palm_xy(
                self.standin, ctl.fk, ctl.apple, state, step, g,
                last_grasp=ctl.last_grasp, horizon=horizon,
            )  # fmt: skip
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
        clipped = c1.clip_to_box(g, p, h, self.a_lo)
        record["rule"] = {
            "iterations": steps,
            "converged": converged,
            "unclipped": g.tolist(),
            "clipped": bool(np.any(clipped != g)),
        }
        _box_record(record, clipped, p, h)
        return clipped


class SysidAim(ptr.TwinAim):
    """H-sysid at 405: a linear regression of plate(r) on (plate, palm, aim), fitted on the smoke
    corpus, inverted with W's controller form (grid, then refinement, clipped); its plate input is
    R-plate's reading (H-sysid) or the true plate (H-sysid-true, with its true-plate fit)."""

    def __init__(self, encoder, readout, coef, *, truth_hook, a_lo: float):
        super().__init__(encoder, readout)
        self.coef = np.asarray(coef, np.float64)
        self.truth_hook, self.a_lo = truth_hook, float(a_lo)

    def __call__(self, ctl, observation, step, record):
        if self.truth_hook is not None:
            p = self.truth_hook.current()
        else:
            p = self.read(observation, step)
            record["reading"] = p.tolist()
        h = np.asarray(ctl.palm[int(step)], np.float64)

        def predict(g):
            return c1.sysid_predict(self.coef, p, h, g)[0]

        g, log = c1.choose_aim(predict, p, h, self.a_lo)
        record["sysid"] = log
        _box_record(record, g, p, h)
        return g


# ----- one attempt --------------------------------------------------------------------------------
def _controller(task: dict, hook: C1Motion):
    W = rtm._W
    arm = task["arm"]
    robot, fk, bounds = W["robot"], W["fk"], W["bounds"]
    _model, standardiser = W["p3"]
    estimates = np.asarray(task["estimates"], np.float64)
    common = (W["predict"], standardiser, estimates, fk, bounds)

    def commit(aim):
        return CommitController(*common, arm=arm, aim=aim)

    if arm == "H-now-reaim":  # TASK-076's H-now: re-aims at the current plate at 405, ..., 485
        return ptr.AimController(*common, arm=arm, aim=ptr.FixedAim(lambda s: hook.current()))  # noqa: ARG005
    if arm == "H-final":
        return commit(
            ptr.LookaheadAim(robot, hook, bounds, c1.LOOKAHEAD_TOLERANCE_M, c1.MAX_ITERATIONS)
        )
    if arm in c1.PROXY_ARMS or arm in ("reach", "collect"):
        return commit(GeometricAim(arm, hook, task))
    truth = hook if arm.endswith("-true") else None
    readout = None if truth is not None else ptr._readout("r_plate")
    encoder = None if truth is not None else rtm.encoder()
    if arm in ("H-rule", "H-rule-true"):
        return commit(
            RuleCommit(encoder, readout, rtm.standin(), truth_hook=truth, a_lo=task["a_lo"])
        )
    if arm in ("H-sysid", "H-sysid-true"):
        return commit(
            SysidAim(encoder, readout, task["sysid_coef"], truth_hook=truth, a_lo=task["a_lo"])
        )
    raise lp.GuardError(f"unknown C1 arm {arm!r}")


def run_attempt_task(task: dict) -> dict:
    """One attempt. ``task``: ``seed``, ``reset``, ``arm``, ``estimates``, the expected post-look
    frame and state sha256, and per arm ``a_lo``, ``foreign_plate``, ``a``/``b_m``,
    ``sysid_coef``, ``capture`` (steps)."""
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
    hook = C1Motion(robot, W["fk"], task.get("capture", ()))
    inner, blocked = None, None
    try:
        inner = _controller(task, hook)
        record = rt2.run_attempt(
            robot,
            scorer,
            counter,
            rtm.Timed(inner),
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
        hook.remove()
    if blocked is not None:
        refused = hook.state.get("refused")
        out |= {
            "termination_reason": "plate_move_refused",
            "complete": False,
            "at_rest": False,
            "latched_success": False,
            "success": False,
            "grasp": None,
            "first_grasp_step": None,
            "first_place_step": None,
            "executed_steps": refused["step"] if refused else None,
            "final_distance_cm": None,
            "privileged_ok": True,
            "commands": np.zeros((0, 7), np.float32),
        }
    else:
        detail = record.get("at_rest_detail") or {}
        out.update({k: record[k] for k in lrt.KEEP})
        out["final_distance_cm"] = (
            None if detail.get("final_distance_m") is None else 100.0 * detail["final_distance_m"]
        )
        out["privileged_ok"] = rt2.privileged_reads_ok(record)
        out["commands"] = record["commands"]
    out["blocked"] = blocked
    decisions = getattr(inner, "decisions", []) if inner is not None else []
    out["decisions"] = decisions
    out["kpred"] = hook.summary()
    aim = getattr(inner, "aim", None)
    if isinstance(aim, ptr.LookaheadAim):
        out["kpred"]["lookahead_branch_steps"] = aim.branch_steps
    plate, palm = hook.state["plate"], hook.state["palm"]
    if decisions and c1.COMMIT_STEP in plate and c1.COMMIT_STEP in palm:
        first = decisions[0]
        p, h = plate[c1.COMMIT_STEP], palm[c1.COMMIT_STEP]
        target = np.asarray(first["target"], np.float64)
        a, b = c1.to_box(target, p, h)
        out["commit"] = {
            "p": p.tolist(),
            "h": h.tolist(),
            "target": target.tolist(),
            "a_true": a,
            "b_true_m": b,
            "fallback": bool(first["fallback"]),
        }
        if c1.RULE["s1"] in plate:
            out["commit"]["landing_miss_cm"] = 100.0 * float(
                np.linalg.norm(target - plate[c1.RULE["s1"]])
            )
    out["captured"] = {str(k): v for k, v in hook.captured.items()}
    out["seconds"] = time.monotonic() - started
    return out


def run_task(task: dict) -> dict:
    if task["kind"] == "attempt":
        return run_attempt_task(task)
    return ptr.run_task(task)
