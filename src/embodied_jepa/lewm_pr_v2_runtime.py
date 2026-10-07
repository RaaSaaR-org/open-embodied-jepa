"""TASK-080 simulation workers (protocol ``docs/experiments/apple_lewm_c1m_v2_pred_readout.md``
§5, §6, §9.4): one closed-loop attempt of any arm, one fresh-corpus root, the kinematic stand-in's
chunks and the offline aims.

TASK-077's worker (``lewm_c1m_v2_runtime``) is imported and reused unchanged: its initializer, the
CPU encoder, the world models, ``choose_from_grid`` (W's controller form, a pure function of the
predictor), ``CorpusMotion``, ``CommandLog``, the stand-in chunks and every non-candidate arm's
controller (H-final, H-rule, H-sysid, H-read, H-now, K0's planted look-ahead). New here:

* **the readouts** (§4, §6): W, L-shuf and L-mean read their predicted plate with **R-S**, N with
  **R-N** (:data:`READOUT_OF`); the configuration names them ``r_s`` and ``r_n`` with their sha256
  and the worker loads them through TASK-076's checked ``_readout``. **No candidate arm reads R8**
  (only the privileged H-read does);
* **L-rand** draws with this task's salt 8209;
* **the fresh corpus's collector** (``CollectF``, §5.2): gate-P's aim is built from **p̂**, R-plate's
  reading of the onboard 405 frame on the CPU (as the closed loop builds its grid), never from the
  true plate; contrast-T's from the true plate (as TASK-077's corpus); both log p̂;
* **the offline aims** of W, N, L-shuf, L-mean, L-rand, H-rule and H-sysid from a logged 405 state
  (§9.4; Stage 0's dry run and Stage R), with W's predicted plate over the whole grid kept for the
  reported echo slope (§5.3 point 4).

Who reads what follows TASK-071's rule (G-privileged): W, N, L-shuf, L-mean, L-rand, H-rule and
H-sysid read the observation and the robot's own joint state only. NumPy at import.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa import first_policy_v2_runtime as rt2
from embodied_jepa import lewm_c1m_v2 as lm
from embodied_jepa import lewm_c1m_v2_runtime as wrt
from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa import lewm_next_c1_runtime as c1rt
from embodied_jepa import lewm_planner_v2 as lp
from embodied_jepa import lewm_planner_v2_runtime as lrt
from embodied_jepa import lewm_pr_v2 as pr
from embodied_jepa import place_planner as pp
from embodied_jepa import plate_shift as ps
from embodied_jepa import plate_twin_v2_runtime as ptr
from embodied_jepa import wm_critic_v2_runtime as rtm

READOUT_OF = dict(pr.READOUTS)  # arm -> "r_s" / "r_n"; never "r8"
ROLLOUT_CHUNK = wrt.ROLLOUT_CHUNK


def worker_init(config: dict) -> None:
    """``config``: TASK-077's worker keys (``p3_checkpoint``, ``torch_threads``, ``r_plate``,
    ``models``, ``mean_latent`` and their sha256) plus ``r_s``/``r_n`` (npz) with their sha256,
    and ``r8`` only for the privileged H-read."""
    wrt.worker_init(config)


def readout_of(arm: str):
    """The predicted-plate readout an arm reads (G-readout: its sha256 is checked on load)."""
    name = READOUT_OF.get(arm)
    if name is None:
        raise lp.GuardError(f"G-readout: {arm} reads no predicted-latent readout")
    return ptr._readout(name)


def rollout_plates(model, start, commands, readout, *, zero: bool = False) -> np.ndarray:
    """``wrt.rollout_plates`` with the arm's own readout in R8's place: [K, 60, 14] -> [K, 2]."""
    return wrt.rollout_plates(model, start, commands, readout, zero=zero)


def choose(arm, p_hat, h, targets, chunks, feasible, predict, chunk_of, *, tolerance_m, seed):
    """W's controller form (``wrt.choose_from_grid``, unchanged) for W, N, L-shuf and L-mean;
    L-rand draws a feasible candidate with salt 8209 and no refinement (§6)."""
    if arm != "L-rand":
        return wrt.choose_from_grid(
            arm, p_hat, h, targets, chunks, feasible, predict, chunk_of,
            tolerance_m=tolerance_m, seed=seed,
        )  # fmt: skip
    p_hat = np.asarray(p_hat, np.float64).reshape(2)
    targets = np.asarray(targets, np.float64).reshape(-1, 2)
    feasible = np.asarray(feasible, bool)
    log = {"arm": arm, "feasible": int(feasible.sum()), "candidates": int(len(targets))}
    if not feasible.any():
        log |= {"fallback_all_infeasible": True, "clipped": False, "rollouts": 0}
        return p_hat.copy(), log
    i = pr.l_rand_index(seed, feasible)
    log |= {
        "fallback_all_infeasible": False,
        "grid_index": i,
        "grid_best": list(pr.GRID[i]),
        "clipped": False,
        "rollouts": 0,
    }
    return targets[i].copy(), log


class PredReadoutAim(ptr.TwinAim):
    """§6 at 405 for W, N, L-shuf, L-mean and L-rand: TASK-077's ``WorldModelAim`` with each arm's
    own readout (R-S or R-N) in R8's place and L-rand's salt 8209. ``start``: "own" (W, N), a
    foreign frame (L-shuf) or "mean" (L-mean); L-rand reads no model."""

    def __init__(self, encoder, readout, *, arm: str, start, tau_commit_cm: float, seed: int):
        super().__init__(encoder, readout)
        self.arm, self.start, self.seed = arm, start, int(seed)
        self.tolerance = pr.REFINE_FRACTION_OF_TAU * float(tau_commit_cm) / 100.0

    def __call__(self, ctl, observation, step, record):
        started = time.perf_counter()
        frame = np.asarray(observation.images[fp2.CAMERA][0])
        tokens, own = wrt.encode(frame)
        p_hat = np.asarray(self.readout.predict(tokens), np.float64).reshape(2)
        self.readings.append((int(step), p_hat))
        record["reading"] = p_hat.tolist()
        h = np.asarray(ctl.palm[int(step)], np.float64)
        state = rt2.state_of(observation)
        targets = np.asarray([c1.from_box(a, b, p_hat, h) for a, b in pr.GRID])
        _req, chunks, feasible = pp.primitive_chunks(
            rtm.standin(), ctl.fk, ctl.apple, state, int(step), targets,
            last_grasp=ctl.last_grasp, horizon=pr.HORIZON,
        )  # fmt: skip

        def chunk_of(g):
            _r, applied, ok = pp.primitive_chunks(
                rtm.standin(), ctl.fk, ctl.apple, state, int(step), np.asarray(g)[None],
                last_grasp=ctl.last_grasp, horizon=pr.HORIZON,
            )  # fmt: skip
            return applied[0], bool(ok[0])

        predict = None
        if self.arm != "L-rand":
            if self.start is None:
                raise lp.GuardError("G-frames: L-shuf reached 405 without a foreign frame")
            if isinstance(self.start, str) and self.start == "own":
                start = own
            elif isinstance(self.start, str) and self.start == "mean":
                start = wrt.mean_latent()
            else:
                _t, start = wrt.encode(self.start)
            model, readout = wrt.world_model(self.arm), readout_of(self.arm)
            zero = self.arm == "N"

            def predict(commands):
                return rollout_plates(model, start, commands, readout, zero=zero)

            record["start_latent_sha256"] = hashlib.sha256(
                np.ascontiguousarray(start, np.float32).tobytes()
            ).hexdigest()
            record["readout"] = READOUT_OF[self.arm]
        g, log = choose(
            self.arm, p_hat, h, targets, chunks, feasible, predict, chunk_of,
            tolerance_m=self.tolerance, seed=self.seed,
        )  # fmt: skip
        log["seconds"] = time.perf_counter() - started
        record["world_model"] = log
        c1rt._box_record(record, g, p_hat, h)
        return g


class CollectF(c1rt.GeometricAim):
    """The fresh corpus's privileged scripted collector (§5.2): (a, b) from salt 8202 over the box
    built from **p̂** (gate-P) or the true plate (contrast-T), with the palm h at 405. Both halves
    log p̂ (R-plate's reading of the onboard 405 frame on the CPU) and the 405 state. The collector
    records the true plate as the target's reference and scores the attempt: privileged data, not
    a learned result; no privileged read reaches any arm."""

    def __init__(self, hook, task: dict, readout):
        super().__init__("collect", hook, task)
        self.readout = readout
        if task.get("aim_from") not in ("p_hat", "true"):
            raise lp.GuardError(
                "G-corpus: a fresh root's aim is built from p_hat or the true plate"
            )

    def __call__(self, ctl, observation, step, record):
        frame = np.asarray(observation.images[fp2.CAMERA][0])
        tokens, _pooled = wrt.encode(frame)
        p_hat = np.asarray(self.readout.predict(tokens), np.float64).reshape(2)
        h = np.asarray(ctl.palm[int(step)], np.float64)
        p_true = np.asarray(self.hook.current(), np.float64)
        base = p_hat if self.task["aim_from"] == "p_hat" else p_true
        g = c1.from_box(self.task["a"], self.task["b_m"], base, h)
        c1rt._box_record(record, g, base, h)
        record["p_hat405"] = p_hat.tolist()
        record["aim_from"] = self.task["aim_from"]
        record["state405"] = rt2.state_of(observation).tolist()
        record["apple_estimate"] = np.asarray(ctl.apple, np.float64).tolist()
        record["last_grasp"] = [float(v) for v in ctl.last_grasp]
        return g


def _common(task: dict):
    W = rtm._W
    _model, standardiser = W["p3"]
    return (
        W["predict"],
        standardiser,
        np.asarray(task["estimates"], np.float64),
        W["fk"],
        W["bounds"],
    )


def _controller(task: dict, hook):
    arm = task["arm"]
    if arm in pr.CANDIDATE_ARMS:
        start = {"W": "own", "N": "own", "L-mean": "mean", "L-rand": None}.get(arm)
        if arm == "L-shuf":  # None only when no other reset reached 405 (R17.20)
            frame = task["foreign_frame"]
            start = None if frame is None else np.asarray(frame, np.uint8)
        aim = PredReadoutAim(
            rtm.encoder(),
            ptr._readout("r_plate"),
            arm=arm,
            start=start,
            tau_commit_cm=float(task["tau_commit_cm"]),
            seed=int(task["seed"]),
        )
        return c1rt.CommitController(*_common(task), arm=arm, aim=aim)
    if arm == "collect-f":
        aim = CollectF(hook, task, ptr._readout("r_plate"))
        return c1rt.CommitController(*_common(task), arm=arm, aim=aim)
    return wrt._controller(task, hook)  # H-final, H-rule, H-sysid, H-read, H-now, planted


def run_attempt_task(task: dict) -> dict:
    """One attempt (or one fresh-corpus root, ``arm == "collect-f"``): TASK-077's
    ``run_attempt_task`` with this module's controllers (the rest of its body is unchanged)."""
    started = time.monotonic()
    W = rtm._W
    robot = W["robot"]
    truth, scorer, counter, _obs, facts = rt2.reset_and_look(robot, task["seed"], task["reset"])
    corpus = task["arm"] == "collect-f"
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
    hook = wrt.CorpusMotion(robot, W["fk"], task["move_offset"], corpus=corpus)
    inner, logged, blocked = None, None, None
    try:
        inner = _controller(task, hook)
        logged = wrt.CommandLog(inner)
        record = rt2.run_attempt(
            robot,
            scorer,
            counter,
            logged,
            bounds=W["bounds"],
            max_steps=lm.STOP_STEP + 1
            if corpus
            else int(task.get("max_steps", fp2.MAX_POLICY_STEPS)),
            settle_steps=0 if corpus else int(task.get("settle_steps", fp2.SETTLE_STEPS)),
            wall_seconds=float(task.get("wall_seconds", pr.CAPS_SECONDS["per_attempt"])),
        )
    except ps.GuardError as error:
        if not str(error).startswith(lp.SHIFT_BLOCKED_PREFIX):
            raise
        blocked = str(error)
    finally:
        hook.remove()
    if blocked is not None:
        out |= wrt._blocked_out(hook)
    else:
        if (
            task["move_offset"] is not None
            and int(record["executed_steps"]) >= lm.MOVE_STEP
            and hook.move_log is None
        ):
            raise lp.GuardError("G-move: the attempt passed step 300 without the declared move")
        detail = record.get("at_rest_detail") or {}
        out.update({k: record[k] for k in lrt.KEEP})
        out["final_distance_cm"] = (
            None if detail.get("final_distance_m") is None else 100.0 * detail["final_distance_m"]
        )
        out["privileged_ok"] = rt2.privileged_reads_ok(record)
        out["task_truth_in_controller"] = int(record["task_truth_in_controller"])
    out["blocked"] = blocked
    decisions = getattr(inner, "decisions", []) if inner is not None else []
    out["decisions"] = decisions
    out["kpred"] = hook.summary()
    aim = getattr(inner, "aim", None)
    if isinstance(aim, ptr.LookaheadAim):
        out["kpred"]["lookahead_branch_steps"] = aim.branch_steps
    plate, palm = hook.state["plate"], hook.state["palm"]
    if decisions and lm.COMMIT_STEP in plate and lm.COMMIT_STEP in palm:
        first = decisions[0]
        p, h = plate[lm.COMMIT_STEP], palm[lm.COMMIT_STEP]
        target = np.asarray(first["target"], np.float64)
        a, b = c1.to_box(target, p, h)
        out["commit"] = {
            "p": p.tolist(),
            "h": h.tolist(),
            "target": target.tolist(),
            "a_true": a,
            "b_true_m": b,
            "fallback": bool(first["fallback"]),
            "clipped": bool(first.get("world_model", {}).get("clipped", False))
            or bool(first.get("rule", {}).get("clipped", False)),
        }
        if lm.RULE["s1"] in plate:
            out["commit"]["landing_miss_cm"] = 100.0 * float(
                np.linalg.norm(target - plate[lm.RULE["s1"]])
            )
            g_star = c1.fixed_point(p, h)
            out["commit"]["fixed_point_error_cm"] = 100.0 * float(np.linalg.norm(target - g_star))
    out["frame405"] = hook.frames.get(lm.COMMIT_STEP)
    if corpus:
        out["corpus"] = corpus_arrays(hook, logged, decisions)
    out["seconds"] = time.monotonic() - started
    return out


def corpus_arrays(hook, logged, decisions) -> dict:
    """TASK-077's root arrays plus p̂ at 405 and the half's aim construction."""
    arrays = wrt.corpus_arrays(hook, logged, decisions)
    if arrays["complete"]:
        first = decisions[0]
        arrays["p_hat405"] = np.asarray(first["p_hat405"], np.float64)
        arrays["aim_from"] = str(first["aim_from"])
    return arrays


# ----- offline aims (Stage 0's dry run and Stage R, §9.4) ---------------------------------------
def _standin_setup(task: dict):
    W = rtm._W
    state = np.asarray(task["state"], np.float64)
    apple = np.asarray(task["apple"], np.float64)
    last_grasp = tuple(float(v) for v in task["last_grasp"])

    def chunks_of(targets):
        _req, applied, ok = pp.primitive_chunks(
            rtm.standin(), W["fk"], apple, state, pr.COMMIT_STEP,
            np.asarray(targets, np.float64).reshape(-1, 2),
            last_grasp=last_grasp, horizon=pr.HORIZON,
        )  # fmt: skip
        return applied, ok

    return state, apple, last_grasp, chunks_of


def offline_rule_aim(task: dict, p_hat, h):
    """H-rule's offline aim: ``RuleCommit``'s iteration from a logged 405 state (the palm at
    max(405 - L, s0) is the palm at 405), clipped to the box around p̂ and h."""
    W = rtm._W
    state, apple, last_grasp, _ = _standin_setup(task)
    lag, s0, s1 = int(pr.RULE["L"]), int(pr.RULE["s0"]), int(pr.RULE["s1"])
    ref = h if max(pr.COMMIT_STEP - lag, s0) == pr.COMMIT_STEP else None
    if ref is None:
        raise lp.GuardError("the offline H-rule needs palm(405) as its reference")
    g, steps, converged = p_hat.copy(), [], False
    for _ in range(c1.MAX_ITERATIONS):
        palm_end = ptr.standin_palm_xy(
            rtm.standin(), W["fk"], apple, state, pr.COMMIT_STEP, g,
            last_grasp=last_grasp, horizon=s1 - lag - pr.COMMIT_STEP,
        )  # fmt: skip
        if palm_end is None:
            steps.append({"stopped": "unreachable_or_refused"})
            break
        nxt = p_hat + float(pr.RULE["kappa"]) * (palm_end - ref)
        gap = float(np.linalg.norm(nxt - g))
        steps.append({"gap_cm": 100.0 * gap})
        g = nxt
        if gap <= c1.LOOKAHEAD_TOLERANCE_M:
            converged = True
            break
    clipped = c1.clip_to_box(g, p_hat, h, pr.A_LO)
    return clipped, {
        "iterations": len(steps),
        "converged": converged,
        "clipped": bool(np.any(clipped != g)),
    }


def run_offline_aim_task(task: dict) -> dict:
    """One arm's offline aim from a logged 405 state (§9.4): the grid from the p̂ given, the
    stand-in chunks, the arm's own controller code and readout, no fallbacks. ``start``: a latent
    (W, N, L-shuf) or ``"mean"`` (L-mean); H-rule and H-sysid read no model. With ``keep_plates``
    W's predicted plate for every feasible candidate is returned (the echo slope, §5.3)."""
    arm = task["arm"]
    p_hat, h = np.asarray(task["p_hat"], np.float64), np.asarray(task["h"], np.float64)
    if arm == "H-rule":
        g, log = offline_rule_aim(task, p_hat, h)
        return {"key": task["key"], "arm": arm, "g": g.tolist(), "log": log}
    if arm == "H-sysid":
        coef = np.asarray(task["sysid_coef"], np.float64)

        def predict_sysid(g):
            return c1.sysid_predict(coef, p_hat, h, g)[0]

        g, log = c1.choose_aim(predict_sysid, p_hat, h, pr.A_LO)
        log = {"converged": bool(log["converged"]), "clipped": False}  # choose_aim always clips
        return {"key": task["key"], "arm": arm, "g": np.asarray(g).tolist(), "log": log}
    _state, _apple, _lg, chunks_of = _standin_setup(task)
    targets = np.asarray([c1.from_box(a, b, p_hat, h) for a, b in pr.GRID])
    chunks, feasible = chunks_of(targets)

    def chunk_of(g):
        applied, ok = chunks_of(np.asarray(g)[None])
        return applied[0], bool(ok[0])

    predict, seen = None, {}
    if arm != "L-rand":
        start = task["start"]
        start = wrt.mean_latent() if isinstance(start, str) and start == "mean" else start
        start = np.asarray(start, np.float32)
        model, readout, zero = wrt.world_model(arm), readout_of(arm), arm == "N"

        def predict(commands):
            out = rollout_plates(model, start, commands, readout, zero=zero)
            if "grid" not in seen:
                seen["grid"] = out.copy()
            return out

    tolerance = pr.REFINE_FRACTION_OF_TAU * float(task["tau_commit_cm"]) / 100.0
    g, log = choose(
        arm, p_hat, h, targets, chunks, feasible, predict, chunk_of,
        tolerance_m=tolerance, seed=int(task["seed"]),
    )  # fmt: skip
    log.pop("scores_cm", None)
    log.pop("iterations", None)
    out = {"key": task["key"], "arm": arm, "g": np.asarray(g).tolist(), "log": log}
    if task.get("keep_plates") and "grid" in seen:
        out["grid_targets"] = targets[np.asarray(feasible, bool)].tolist()
        out["grid_plates"] = seen["grid"].tolist()
    return out


def run_task(task: dict) -> dict:
    if task["kind"] == "offline_aim":
        return run_offline_aim_task(task)
    if task["kind"] == "attempt":
        return run_attempt_task(task)
    if task["kind"] == "chunks":
        return wrt.run_chunks_task(task)
    return ptr.run_task(task)
