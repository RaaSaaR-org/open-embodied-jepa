"""TASK-082 design note's development workers: one attempt under a candidate plate law (not gated).

C1-M's attempt (``lewm_next_c1m_runtime``) with one change: the hook's plate position after the
commit step follows ``plate_law_dev.displacement(law, ...)`` instead of cell A's linear rule. The
step-300 move, P-3's pick, the single aim committed at 405, e9's place primitive, the scorer and
every arm's code are reused unchanged; ``H-rule-true`` is therefore still handed C1-M's law
(kappa = -0.5), which under ``sat`` or ``play`` is the wrong law. New here:

* ``LawMotion``: ``MoveMotion`` whose ``target`` follows the task's law (also inside the
  look-ahead's branches, which call the same hook);
* ``KrrAim``: H-sysid-krr, the RBF kernel ridge of ``plate_law_dev`` inverted with the same
  controller form as H-sysid (``lewm_next_c1.choose_aim``: grid, refinement, clip), true plate.

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
from embodied_jepa import lewm_next_c1m_runtime as c1mrt
from embodied_jepa import lewm_planner_v2 as lp
from embodied_jepa import lewm_planner_v2_runtime as lrt
from embodied_jepa import plate_law_dev as pl
from embodied_jepa import plate_shift as ps
from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa import plate_twin_v2_runtime as ptr
from embodied_jepa import wm_critic_v2_runtime as rtm

PRIVILEGED_ARMS = frozenset({"H-final", "H-now", "collect"})


def worker_init(config: dict) -> None:
    ptr.worker_init(config)


class LawMotion(c1mrt.MoveMotion):
    def __init__(self, robot, fk, offset, law: str):
        if law not in pl.LAWS:
            raise lp.GuardError(f"unknown plate law {law!r}")
        super().__init__(robot, fk, offset)
        self.law = law

    def target(self, step: int) -> np.ndarray:
        if step <= self.s0:
            return self.base.copy()
        return self.base + pl.displacement(self.law, self.state["palm"], step)


class KrrAim:
    def __init__(self, hook, model: dict, a_lo: float):
        self.hook, self.a_lo = hook, float(a_lo)
        self.model = {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in model.items()}

    def __call__(self, ctl, observation, step, record):  # noqa: ARG002
        p, h = self.hook.current(), np.asarray(ctl.palm[int(step)], np.float64)

        def predict(g):
            return pl.krr_plate(self.model, p, h, g)[0]

        g, log = c1.choose_aim(predict, p, h, self.a_lo)
        record["sysid"] = log
        c1rt._box_record(record, g, p, h)
        return g


def _controller(task: dict, hook: LawMotion):
    if task["arm"] == "H-sysid-krr-true":
        W = rtm._W
        _model, standardiser = W["p3"]
        common = (W["predict"], standardiser, np.asarray(task["estimates"], np.float64),
                  W["fk"], W["bounds"])  # fmt: skip
        aim = KrrAim(hook, task["krr"], task["a_lo"])
        return c1rt.CommitController(*common, arm=task["arm"], aim=aim)
    return c1mrt._controller(task, hook)  # H-final, H-now, H-rule-true, H-sysid-true, collect


def run_attempt_task(task: dict) -> dict:
    """One attempt; ``task`` as ``lewm_next_c1m_runtime.run_attempt_task`` plus ``law``."""
    started = time.monotonic()
    W = rtm._W
    robot = W["robot"]
    truth, scorer, counter, _obs, facts = rt2.reset_and_look(robot, task["seed"], task["reset"])
    out = {
        "seed": task["seed"],
        "arm": task["arm"],
        "law": task["law"],
        "post_look_frame_sha256": rtm.frame_sha(facts["post_look_frame"]),
        "post_look_state": facts["post_look_state"].tolist(),
    }
    try:
        out["render_retries"] = rtm.check_post_look_frame(robot, task, facts["post_look_frame"])
    except lp.GuardError:
        counter.remove()
        robot.stop("frame_mismatch")
        raise
    hook = LawMotion(robot, W["fk"], task["move_offset"], task["law"])
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
        refused = hook.state.get("refused") or {}
        out |= {"termination_reason": "plate_move_refused", "success": False,
                "executed_steps": refused.get("step"), "final_distance_cm": None,
                "privileged_ok": True}  # fmt: skip
    else:
        if int(record["executed_steps"]) >= c1m.MOVE_STEP and hook.move_log is None:
            raise lp.GuardError("G-move: the attempt passed step 300 without the declared move")
        detail = record.get("at_rest_detail") or {}
        out.update({k: record[k] for k in lrt.KEEP if k != "commands"})
        out["final_distance_cm"] = (
            None if detail.get("final_distance_m") is None else 100.0 * detail["final_distance_m"]
        )
        out["privileged_ok"] = rt2.privileged_reads_ok(record)
    out["blocked"] = blocked
    decisions = getattr(inner, "decisions", []) if inner is not None else []
    plate, palm = hook.state["plate"], hook.state["palm"]
    if decisions and c1.COMMIT_STEP in plate and c1.COMMIT_STEP in palm:
        p, h = plate[c1.COMMIT_STEP], palm[c1.COMMIT_STEP]
        target = np.asarray(decisions[0]["target"], np.float64)
        a, b = c1.to_box(target, p, h)
        final = plate.get(pl.S1)
        out["commit"] = {
            "p": p.tolist(),
            "h": h.tolist(),
            "target": target.tolist(),
            "a_true": a,
            "b_true_m": b,
            "fallback": bool(decisions[0]["fallback"]),
            "plate_s1": None if final is None else final.tolist(),
            "landing_miss_cm": None
            if final is None
            else 100.0 * float(np.linalg.norm(target - final)),
            "plate_motion_cm": None if final is None else 100.0 * float(np.linalg.norm(final - p)),
        }
        lookahead = decisions[0].get("lookahead") if isinstance(decisions[0], dict) else None
        if lookahead is not None:
            out["commit"]["lookahead_converged"] = lookahead.get("converged")
    out["seconds"] = time.monotonic() - started
    return out


def run_task(task: dict) -> dict:
    if task["kind"] == "attempt":
        return run_attempt_task(task)
    return ptr.run_task(task)
