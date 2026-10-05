"""C1-M feasibility workers: one attempt under cell A's reactive-plate rule with a single aim
committed at 405, after a declared post-pick plate move at step 300 (development only; no world
model).

Record ``docs/experiments/apple_lewm_next_v2_c1m_feasibility.md``; constants
``lewm_next_c1m.py``. Reused unchanged: TASK-076's worker (``plate_twin_v2_runtime.worker_init``),
``CellMotion``, ``AimController``, ``LookaheadAim`` (H-final) and ``PlateBrancher``; C1's
``C1Motion``, ``CommitController``, ``GeometricAim``, ``RuleCommit``, ``SysidAim`` and its
controller factory. New here:

* ``MoveMotion``: ``C1Motion`` plus the move: just before the observation of step 300 it calls
  ``plate_shift.move_plate`` (TASK-073's hook function) by the reset's stored offset, and the rule's
  base becomes the moved plate;
* ``MeanProxyAim``: the mean-proxy with the move distribution's mean plate;
* ``PlantedLookahead``: H-final(commit)'s aim plus a planted error (M-F4);
* ``ReadLookahead``: H-read, H-final(commit)'s cloned look-ahead with plate(r) read by the pooled
  readout from the frame rendered at r in the clone (M-F5b);
* ``StaleRule``: H-rule fed P-3's post-look (pre-move) plate estimate (M-F6).

NumPy at import; MuJoCo and torch are imported by the worker initializer.
"""

from __future__ import annotations

import time

import numpy as np

from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa import first_policy_v2_runtime as rt2
from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa import lewm_next_c1_runtime as c1rt
from embodied_jepa import lewm_next_c1m as c1m
from embodied_jepa import lewm_planner_v2 as lp
from embodied_jepa import lewm_planner_v2_runtime as lrt
from embodied_jepa import place_planner as pp
from embodied_jepa import plate_shift as ps
from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa import plate_twin_v2_runtime as ptr
from embodied_jepa import wm_critic_v2_runtime as rtm
from embodied_jepa.contracts import ContractError


def worker_init(config: dict) -> None:
    ptr.worker_init(config)


# ----- the hook: cell A after the step-300 move ---------------------------------------------------
class MoveMotion(c1rt.C1Motion):
    """``C1Motion`` plus the declared move: just before the observation of step 300 the plate is
    moved by ``offset`` (``plate_shift.move_plate``) and the rule's base becomes the moved plate.
    A blocked move (``SHIFT_BLOCKED_PREFIX``) is recorded as the attempt's refusal and re-raised
    (a counted failure); any other move refusal propagates (a guard)."""

    def __init__(self, robot, fk, offset, capture=()):
        super().__init__(robot, fk, capture)
        self.offset = None if offset is None else np.asarray(offset, np.float64).reshape(2)
        self.reset_plate = self.base.copy()
        self.move_log: dict | None = None

    def _observe(self):
        t = self.state["calls"]
        if t == c1m.MOVE_STEP and self.offset is not None and self.move_log is None:
            try:
                log = ps.move_plate(self.robot.sim, self.offset)
            except ps.GuardError as error:
                if str(error).startswith(lp.SHIFT_BLOCKED_PREFIX):
                    self.state["refused"] = {"step": int(t), "reason": str(error), "move": True}
                raise
            self.move_log = log | {"step": int(t)}
            self.base = self.current()
        return super()._observe()

    def summary(self) -> dict:
        out = super().summary()
        out["move"] = {
            "offset": None if self.offset is None else self.offset.tolist(),
            "reset_plate": self.reset_plate.tolist(),
            "applied": self.move_log is not None,
            "log": self.move_log,
        }
        return out


# ----- aims ---------------------------------------------------------------------------------------
class MeanProxyAim:
    """The mean-proxy: (p-bar - kappa h) / (1 - kappa), clipped to the box around the true plate,
    with p-bar the move distribution's mean plate (R16.3)."""

    def __init__(self, hook: MoveMotion, plate_mean, a_lo: float):
        self.hook, self.a_lo = hook, float(a_lo)
        self.plate_mean = np.asarray(plate_mean, np.float64).reshape(2)

    def __call__(self, ctl, observation, step, record):  # noqa: ARG002
        p, h = self.hook.current(), np.asarray(ctl.palm[int(step)], np.float64)
        g = c1.clip_to_box(c1.fixed_point(self.plate_mean, h), p, h, self.a_lo)
        c1rt._box_record(record, g, p, h)
        record["plate_mean"] = self.plate_mean.tolist()
        return g


class PlantedLookahead(ptr.LookaheadAim):
    """M-F4: H-final(commit)'s converged aim plus a planted error (world xy, m), unclipped."""

    def __init__(self, robot, hook, bounds, tolerance_m, max_iter, planted_m):
        super().__init__(robot, hook, bounds, tolerance_m, max_iter)
        self.planted = np.asarray(planted_m, np.float64).reshape(2)

    def __call__(self, ctl, observation, step, record):
        g = super().__call__(ctl, observation, step, record)
        record["lookahead_aim"] = np.asarray(g).tolist()
        record["planted_m"] = self.planted.tolist()
        return np.asarray(g, np.float64) + self.planted


def pooled_reading(encoder, readout, frame, grid: int) -> np.ndarray:
    """The readout's plate from one onboard frame: frozen DINOv2 tokens (CPU, batch 1) pooled to
    ``grid`` x ``grid`` (``frozen_tokens.pool_tokens``), the corpus's featurisation."""
    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa.models.frozen_tokens import pool_tokens

    tokens = pe.features(encoder, np.array(frame, np.uint8, copy=True)[None], batch=1)["tokens"]
    pooled = pool_tokens(tokens, int(grid))
    return np.asarray(readout.predict(pooled[0]), np.float64).reshape(2)


class ReadLookahead(ptr.LookaheadAim):
    """H-read (M-F5b): H-final(commit)'s cloned look-ahead in which every roll-out of the place
    primitive aimed at g_k runs to r with the rule active, the onboard frame at r is rendered in the
    clone (earlier steps blank, as H-final), and g_{k+1} = the pooled readout's reading of it.
    g0 = the true plate at 405 (as H-final; a disclosed privileged start of the fixed point). Stop
    at |g_{k+1} - g_k| <= tolerance, at most ``max_iter`` iterations, or when a branch stops before
    r (aim at the last iterate). Privileged: it holds the robot."""

    def __init__(self, robot, hook, bounds, tolerance_m, max_iter, *, read_step, encoder,
                 readout, grid):  # fmt: skip
        super().__init__(robot, hook, bounds, tolerance_m, max_iter)
        self.read_step = int(read_step)
        self.encoder, self.readout, self.grid = encoder, readout, int(grid)

    def _rollout_read(self, ctl, observation, step: int, aim_xy, blank_render) -> dict:
        """``PlateBrancher.rollout``'s loop to r, rendering the frame at r."""
        robot, hook = self.robot, self.hook
        lower, upper = self.bounds
        try:
            branch = pp.BranchPlace(ctl.fk, ctl.apple, aim_xy, step, ctl.last_grasp[1])
        except ContractError:
            return {"frame": None, "stopped": "unreachable", "steps": 0}
        obs, stopped, frame, j = observation, None, None, int(step)
        for j in range(int(step), self.read_step):
            command = np.clip(np.asarray(branch.act(obs, j), np.float32), lower, upper)
            try:
                projection = robot.project_candidates(command[None, None, None])
            except ContractError as error:
                if str(error) not in rt2.GUARD_REFUSALS:
                    raise
                stopped = "guard_refusal"
                break
            if not bool(projection.feasible[0, 0]):
                stopped = "infeasible_command"
                break
            result = robot.execute(projection.actions[0, 0, 0])
            if result.applied_action is None:
                stopped = result.reason or result.status or "rejected"
                break
            branch.advance(result)
            try:
                hook.apply(j + 1)
            except ps.GuardError:
                stopped = "plate_move_refused"
                break
            if j + 1 == self.read_step:
                del robot.sim.render  # the real renderer, for the frame at r only
                try:
                    obs = type(robot).observe(robot)
                finally:
                    robot.sim.render = blank_render
                frame = np.asarray(obs.images[fp2.CAMERA][0], np.uint8).copy()
            else:
                obs = type(robot).observe(robot)  # bypasses the live hook's wrapper
            hook.record(j + 1, obs)
        return {
            "frame": frame,
            "stopped": stopped,
            "steps": j + 1 - int(step),
            "plate_true": hook.current(),
        }

    def __call__(self, ctl, observation, step, record):
        if self.read_step <= int(step):
            raise lp.GuardError("G-read: r must lie after the commit step")
        sim = self.robot.sim
        blank = np.zeros((sim.height, sim.width, 3), np.uint8)

        def blank_render(camera="onboard_rgb"):  # noqa: ARG001
            return blank.copy()

        saved = self.brancher.save()
        g = self.hook.current()
        steps, converged, stopped = [], False, None
        sim.render = blank_render
        try:
            for _ in range(self.max_iter):
                self.brancher.restore(saved)
                out = self._rollout_read(ctl, observation, step, g, blank_render)
                self.branch_steps += out["steps"]
                if out["frame"] is None:
                    stopped = out["stopped"] or "no_frame"
                    steps.append({"stopped": stopped})
                    break
                nxt = pooled_reading(self.encoder, self.readout, out["frame"], self.grid)
                gap = float(np.linalg.norm(nxt - g))
                true_r = np.asarray(out["plate_true"], np.float64)
                steps.append(
                    {
                        "g": nxt.tolist(),
                        "gap_cm": 100.0 * gap,
                        "plate_r_true": true_r.tolist(),
                        "reading_error_cm": 100.0 * float(np.linalg.norm(nxt - true_r)),
                    }
                )
                g = nxt
                if gap <= self.tolerance:
                    converged = True
                    break
        finally:
            if "render" in vars(sim):
                del sim.render
            self.brancher.restore(saved)
        record["lookahead"] = {
            "iterations": steps,
            "converged": converged,
            "stopped": stopped,
            "grid": self.grid,
            "read_step": self.read_step,
        }
        return g


class StaleRule(c1rt.RuleCommit):
    """H-rule-stale (M-F6, reported): H-rule's fixed point fed P-3's post-look plate estimate (the
    pre-move reading P-3 itself acts on), clipped to the box around that estimate."""

    def __init__(self, standin, stale_plate, *, a_lo: float):
        super().__init__(None, None, standin, truth_hook=None, a_lo=a_lo)
        self.stale = np.asarray(stale_plate, np.float64).reshape(2).copy()

    def read(self, observation, step: int) -> np.ndarray:  # noqa: ARG002
        self.readings.append((int(step), self.stale.copy()))
        return self.stale.copy()


# ----- one attempt --------------------------------------------------------------------------------
def _controller(task: dict, hook: MoveMotion):
    W = rtm._W
    arm = task["arm"]
    robot, fk, bounds = W["robot"], W["fk"], W["bounds"]
    _model, standardiser = W["p3"]
    estimates = np.asarray(task["estimates"], np.float64)
    common = (W["predict"], standardiser, estimates, fk, bounds)

    def commit(aim):
        return c1rt.CommitController(*common, arm=arm, aim=aim)

    tolerance, max_iter = c1.LOOKAHEAD_TOLERANCE_M, c1.MAX_ITERATIONS
    if arm == "mean-proxy":
        return commit(MeanProxyAim(hook, task["plate_mean"], task["a_lo"]))
    if arm == "H-final-planted":
        return commit(PlantedLookahead(robot, hook, bounds, tolerance, max_iter, task["planted_m"]))
    if arm == "H-read":
        readout = ptr._readout(task["readout"])
        return commit(
            ReadLookahead(
                robot, hook, bounds, tolerance, max_iter,
                read_step=task["read_step"], encoder=rtm.encoder(), readout=readout,
                grid=task["grid"],
            )
        )  # fmt: skip
    if arm == "H-rule-stale":
        return commit(StaleRule(rtm.standin(), estimates[2:], a_lo=task["a_lo"]))
    return c1rt._controller(task, hook)  # H-final, H-now, N-, shuf-proxy, reach, collect, ...


def run_attempt_task(task: dict) -> dict:
    """One attempt. ``task``: ``seed``, ``reset``, ``arm``, ``estimates``, the expected post-look
    frame and state sha256, ``move_offset`` (world xy, m; None for no move, debug only), and per
    arm ``a_lo``, ``foreign_plate``, ``plate_mean``, ``a``/``b_m``, ``sysid_coef``,
    ``planted_m``, ``readout``/``grid``/``read_step``, ``capture`` (steps)."""
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
    hook = MoveMotion(robot, W["fk"], task["move_offset"], task.get("capture", ()))
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
            "termination_reason": "move_blocked"
            if refused and refused.get("move")
            else "plate_move_refused",
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
        if (
            task["move_offset"] is not None
            and int(record["executed_steps"]) >= c1m.MOVE_STEP
            and hook.move_log is None
        ):
            raise lp.GuardError("G-move: the attempt passed step 300 without the declared move")
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
