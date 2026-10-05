"""TASK-077 simulation workers: one attempt of any arm under C1-M, one corpus root, and the
kinematic stand-in's command chunks (protocol ``docs/experiments/apple_lewm_c1m_v2.md`` §4.3, §5).

Reused unchanged: TASK-076's worker (``plate_twin_v2_runtime.worker_init``: the v2 robot, P-3, its
private kinematics, the bounds), ``TwinAim`` and ``_readout`` (G-readout); C1's
``CommitController``, ``GeometricAim``, ``RuleCommit``, ``SysidAim`` and ``clip_to_box``; C1-M's
``MoveMotion`` (the step-300 move plus cell A's rule), ``PlantedLookahead``, ``ReadLookahead``,
``MeanProxyAim`` and its controller factory (H-final(commit), H-now, H-rule, H-sysid, H-read and
K0's proxies). New here:

* ``CorpusMotion``: ``MoveMotion`` plus the corpus captures (the onboard 112 px frames of steps
  403-467, the plate-hidden render at r) and the 405 frame of every attempt (L-shuf's foreign
  frame is the logged 405 frame of W's attempt on the next reset);
* ``CommandLog``: the executed normalised 14-D commands (``result.applied_action``) per step;
* ``WorldModelAim``: W's controller form (§5.1) for W, N, L-shuf, L-mean and L-rand: the grid from
  R-plate's reading and the proprioceptive palm, the stand-in's 60 commands per candidate, the
  CPU roll-out to r, R8's predicted plate, the lowest |p~(g) - g| and at most 10 clipped
  refinements. Its decision (:func:`choose_from_grid`) is a pure function of the predictor;
* ``CollectAim``: the corpus's privileged scripted aim, e9's aim committed to (a, b) drawn over the
  box from the true plate and palm (as C1-M's ``collect``), which also logs the 405 state.

Who reads what follows TASK-071's rule: W, N, L-shuf, L-mean, L-rand, H-rule and H-sysid read the
observation and the robot's own joint state only (``task_truth_in_controller`` is 0, G-privileged).
NumPy at import; MuJoCo and torch are imported by the worker initializer.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa import first_policy_v2_runtime as rt2
from embodied_jepa import lewm_c1m_v2 as lm
from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa import lewm_next_c1_runtime as c1rt
from embodied_jepa import lewm_next_c1m_runtime as c1mrt
from embodied_jepa import lewm_planner_v2 as lp
from embodied_jepa import lewm_planner_v2_runtime as lrt
from embodied_jepa import place_planner as pp
from embodied_jepa import plate_shift as ps
from embodied_jepa import plate_twin_v2_runtime as ptr
from embodied_jepa import wm_critic_v2_runtime as rtm

_M: dict = {}
ROLLOUT_CHUNK = 16  # candidates per CPU predict call (bounds the [k, 60, 24 576] output)


def worker_init(config: dict) -> None:
    """``config``: ``p3_checkpoint``, ``torch_threads`` (1) and optional ``r_plate``, ``r8``
    (npz) with their sha256, ``models`` (arm -> {path, sha256, seed, metadata}) and
    ``mean_latent`` / ``mean_latent_sha256`` (npy)."""
    ptr.worker_init(config)
    _M.clear()
    _M["config"] = dict(config)


# ----- the world models on the CPU ---------------------------------------------------------------
def world_model(arm: str):
    """The trained W or N of the primary seed, on the CPU (G-checkpoint against its sha256)."""
    from embodied_jepa import lewm_c1m_v2_train as tr

    name = "N" if arm == "N" else "W"
    key = ("model", name)
    if key not in _M:
        spec = _M["config"]["models"][name]
        _M[key] = tr.load_model(
            spec["path"],
            seed=int(spec["seed"]),
            metadata=dict(spec["metadata"]),
            expected_sha256=spec["sha256"],
        )
    return _M[key]


def mean_latent() -> np.ndarray:
    if "mean_latent" not in _M:
        path = _M["config"]["mean_latent"]
        data = np.load(path)
        digest = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
        if digest != _M["config"]["mean_latent_sha256"]:
            raise lp.GuardError("G-readout: the mean latent differs from its recorded sha256")
        _M["mean_latent"] = np.asarray(data, np.float32)
    return _M["mean_latent"]


def encode(frame) -> tuple[np.ndarray, np.ndarray]:
    """One onboard frame through the frozen DINOv2 on the CPU (batch 1): the full tokens (R-plate's
    input, float64 [98 304]) and the 8 x 8 pooled latent (W's start, float32 [24 576])."""
    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa.models.frozen_tokens import pool_tokens

    frame = np.array(frame, np.uint8, copy=True)
    tokens = pe.features(rtm.encoder(), frame[None], batch=1)["tokens"]
    return tokens[0], pool_tokens(tokens, lm.TOKEN_GRID)[0]


def rollout_plates(model, start, commands, r8, *, zero: bool = False) -> np.ndarray:
    """R8's plate at r from the predicted latent: ``commands`` [K, 60, 14] -> [K, 2]. With
    ``zero`` (N) one zero-command roll-out stands for every candidate (they are identical)."""
    commands = np.asarray(commands, np.float32)
    k = len(commands)
    if zero:
        one = np.zeros((1, lm.HORIZON, 14), np.float32)
        latent = model.predict_features(np.asarray(start, np.float32)[None], one)[:, -1]
        return np.repeat(np.asarray(r8.predict(latent), np.float64).reshape(1, 2), k, 0)
    out = np.empty((k, 2))
    for lo in range(0, k, ROLLOUT_CHUNK):
        part = np.ascontiguousarray(commands[lo : lo + ROLLOUT_CHUNK])
        starts = np.repeat(np.asarray(start, np.float32)[None], len(part), 0)
        latent = model.predict_features(starts, part)[:, -1]
        out[lo : lo + len(part)] = np.asarray(r8.predict(latent), np.float64).reshape(-1, 2)
    return out


# ----- W's controller form, as a pure function --------------------------------------------------
def choose_from_grid(
    arm: str,
    p_hat,
    h,
    targets,
    chunks,
    feasible,
    predict,
    chunk_of,
    *,
    tolerance_m: float,
    seed: int,
    a_lo: float = lm.A_LO,
    max_refine: int = lm.REFINE_MAX,
):
    """§5.1 steps 1-4. ``predict(commands [K, 60, 14]) -> plates [K, 2]``; ``chunk_of(g) ->
    (commands [60, 14], feasible)``. Infeasible candidates are dropped; if all are, the aim is
    p-hat (a counted attempt). L-rand draws a feasible candidate (salt 8111), no refinement.
    Otherwise the lowest |p~(g) - g| over the feasible grid wins (ties to the lowest row-major
    index), then at most ``max_refine`` single-candidate roll-outs g <- clip(p~(g)) (C1's
    ``choose_aim`` loop), stopping once the move is <= ``tolerance_m``; a refined aim whose
    stand-in chunk is infeasible is not taken (the loop stops, logged)."""
    p_hat = np.asarray(p_hat, np.float64).reshape(2)
    h = np.asarray(h, np.float64).reshape(2)
    targets = np.asarray(targets, np.float64).reshape(-1, 2)
    feasible = np.asarray(feasible, bool)
    index = np.flatnonzero(feasible)
    log: dict = {"arm": arm, "feasible": int(len(index)), "candidates": int(len(targets))}
    if len(index) == 0:
        log |= {"fallback_all_infeasible": True, "clipped": False, "rollouts": 0}
        return p_hat.copy(), log
    log["fallback_all_infeasible"] = False
    if arm == "L-rand":
        i = lm.l_rand_index(seed, feasible)
        log |= {"grid_index": i, "grid_best": list(lm.GRID[i]), "clipped": False, "rollouts": 0}
        return targets[i].copy(), log
    chunks = np.asarray(chunks)
    plates = np.asarray(predict(chunks[index]), np.float64).reshape(-1, 2)
    scores = np.linalg.norm(plates - targets[index], axis=1)
    j = int(np.argmin(scores))
    best = int(index[j])
    g, commands_g = targets[best].copy(), chunks[best]
    log |= {
        "grid_index": best,
        "grid_best": list(lm.GRID[best]),
        "grid_best_score_cm": 100.0 * float(scores[j]),
        "scores_cm": (100.0 * scores).round(5).tolist(),
    }
    steps, converged, clipped, rollouts = [], False, False, len(index)
    for _ in range(int(max_refine)):
        pg = np.asarray(predict(np.asarray(commands_g)[None]), np.float64).reshape(2)
        rollouts += 1
        nxt = c1.clip_to_box(pg, p_hat, h, a_lo)
        was_clipped = bool(np.linalg.norm(nxt - pg) > 1e-12)
        gap = float(np.linalg.norm(nxt - g))
        commands_nxt, ok = chunk_of(nxt)
        if not ok:
            steps.append({"stopped": "infeasible_refinement", "g": nxt.tolist()})
            break
        g, commands_g, clipped = nxt, commands_nxt, clipped or was_clipped
        steps.append({"g": nxt.tolist(), "gap_cm": 100.0 * gap, "clipped": was_clipped})
        if gap <= tolerance_m:
            converged = True
            break
    log |= {"iterations": steps, "converged": converged, "clipped": clipped, "rollouts": rollouts}
    return g, log


class WorldModelAim(ptr.TwinAim):
    """§5.1 at 405 for W, N, L-shuf, L-mean and L-rand. ``start``: "own" (W, N), a foreign frame
    (L-shuf) or "mean" (L-mean); L-rand reads no model."""

    def __init__(self, encoder, readout, *, arm: str, start, tau_commit_cm: float, seed: int):
        super().__init__(encoder, readout)
        self.arm, self.start, self.seed = arm, start, int(seed)
        self.tolerance = lm.REFINE_FRACTION_OF_TAU * float(tau_commit_cm) / 100.0

    def __call__(self, ctl, observation, step, record):
        started = time.perf_counter()
        frame = np.asarray(observation.images[fp2.CAMERA][0])
        tokens, own = encode(frame)
        p_hat = np.asarray(self.readout.predict(tokens), np.float64).reshape(2)
        self.readings.append((int(step), p_hat))
        record["reading"] = p_hat.tolist()
        h = np.asarray(ctl.palm[int(step)], np.float64)
        state = rt2.state_of(observation)
        targets = np.asarray([c1.from_box(a, b, p_hat, h) for a, b in lm.GRID])
        _req, chunks, feasible = pp.primitive_chunks(
            rtm.standin(),
            ctl.fk,
            ctl.apple,
            state,
            int(step),
            targets,
            last_grasp=ctl.last_grasp,
            horizon=lm.HORIZON,
        )

        def chunk_of(g):
            _r, applied, ok = pp.primitive_chunks(
                rtm.standin(),
                ctl.fk,
                ctl.apple,
                state,
                int(step),
                np.asarray(g)[None],
                last_grasp=ctl.last_grasp,
                horizon=lm.HORIZON,
            )
            return applied[0], bool(ok[0])

        if self.arm == "L-rand":
            predict = None
        else:
            if isinstance(self.start, str) and self.start == "own":
                start = own
            elif isinstance(self.start, str) and self.start == "mean":
                start = mean_latent()
            else:
                _t, start = encode(self.start)
            model, r8 = world_model(self.arm), ptr._readout("r8")
            zero = self.arm == "N"

            def predict(commands):
                return rollout_plates(model, start, commands, r8, zero=zero)

            record["start_latent_sha256"] = hashlib.sha256(
                np.ascontiguousarray(start, np.float32).tobytes()
            ).hexdigest()
        g, log = choose_from_grid(
            self.arm,
            p_hat,
            h,
            targets,
            chunks,
            feasible,
            predict,
            chunk_of,
            tolerance_m=self.tolerance,
            seed=self.seed,
        )
        log["seconds"] = time.perf_counter() - started
        record["world_model"] = log
        c1rt._box_record(record, g, p_hat, h)
        return g


class CollectAim(c1rt.GeometricAim):
    """The corpus collector (privileged, scripted): (a, b) over the box from the true plate and
    palm at 405, as C1-M's ``collect``; it also logs what the stand-in needs offline."""

    def __call__(self, ctl, observation, step, record):
        g = super().__call__(ctl, observation, step, record)
        record["state405"] = rt2.state_of(observation).tolist()
        record["apple_estimate"] = np.asarray(ctl.apple, np.float64).tolist()
        record["last_grasp"] = [float(v) for v in ctl.last_grasp]
        return g


# ----- the hook and the command log -------------------------------------------------------------
class CorpusMotion(c1mrt.MoveMotion):
    """``MoveMotion`` plus the frames: the 405 frame always; with ``corpus`` the onboard frames of
    steps 403-467 and the plate-hidden render at r (TASK-075's renderer)."""

    def __init__(self, robot, fk, offset, *, corpus: bool = False):
        super().__init__(robot, fk, offset, capture=())
        self.corpus = bool(corpus)
        self.frames: dict[int, np.ndarray] = {}
        self.hidden_r: np.ndarray | None = None
        self.plate_kept: dict[int, np.ndarray] = {}

    def _observe(self):
        t = self.state["calls"]
        observation = super()._observe()
        keep = t == lm.COMMIT_STEP or (self.corpus and lm.FRAME_STEPS[0] <= t <= lm.FRAME_STEPS[1])
        if keep:
            self.frames[int(t)] = np.asarray(observation.images[fp2.CAMERA][0], np.uint8).copy()
            self.plate_kept[int(t)] = self.current()  # the hook's own path starts at s0 = 405
        if self.corpus and t == lm.READ_STEP:
            from embodied_jepa import obs_ceiling_v2_runtime as ort

            hidden, _ = ort.capture("onboard112", "plate_hidden")
            self.hidden_r = np.asarray(hidden, np.uint8).copy()
        return observation


class CommandLog(rtm.Timed):
    """``Timed`` plus the executed normalised 14-D command of every step."""

    def __init__(self, inner):
        super().__init__(inner)
        self.step = None
        self.executed: dict[int, np.ndarray] = {}

    def act(self, observation, step):
        self.step = int(step)
        return super().act(observation, step)

    def advance(self, result):
        if result.applied_action is not None and self.step is not None:
            self.executed[self.step] = np.asarray(result.applied_action, np.float32).copy()
        return super().advance(result)


# ----- one attempt --------------------------------------------------------------------------------
def _controller(task: dict, hook: CorpusMotion):
    W = rtm._W
    arm = task["arm"]
    if arm in lm.CANDIDATE_ARMS:
        _model, standardiser = W["p3"]
        common = (
            W["predict"],
            standardiser,
            np.asarray(task["estimates"], np.float64),
            W["fk"],
            W["bounds"],
        )
        start = {"W": "own", "N": "own", "L-mean": "mean", "L-rand": None}.get(arm)
        if arm == "L-shuf":
            start = np.asarray(task["foreign_frame"], np.uint8)
        aim = WorldModelAim(
            rtm.encoder(),
            ptr._readout("r_plate"),
            arm=arm,
            start=start,
            tau_commit_cm=float(task["tau_commit_cm"]),
            seed=int(task["seed"]),
        )
        return c1rt.CommitController(*common, arm=arm, aim=aim)
    if arm == "collect":
        _model, standardiser = W["p3"]
        common = (
            W["predict"],
            standardiser,
            np.asarray(task["estimates"], np.float64),
            W["fk"],
            W["bounds"],
        )
        return c1rt.CommitController(*common, arm=arm, aim=CollectAim(arm, hook, task))
    return c1mrt._controller(task, hook)


def _blocked_out(hook) -> dict:
    refused = hook.state.get("refused")
    return {
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
        "task_truth_in_controller": 0,
    }


def run_attempt_task(task: dict) -> dict:
    """One attempt (or one corpus root, ``arm == "collect"``). ``task``: ``seed``, ``reset``,
    ``arm``, ``estimates``, the expected post-look frame and state sha256, ``move_offset``, and
    per arm ``tau_commit_cm``, ``foreign_frame``, ``a``/``b_m``, ``a_lo``, ``sysid_coef``,
    ``planted_m``, ``plate_mean``, ``foreign_plate``, ``readout``/``grid``/``read_step``."""
    started = time.monotonic()
    W = rtm._W
    robot = W["robot"]
    truth, scorer, counter, _obs, facts = rt2.reset_and_look(robot, task["seed"], task["reset"])
    corpus = task["arm"] == "collect"
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
    hook = CorpusMotion(robot, W["fk"], task["move_offset"], corpus=corpus)
    inner, logged, blocked = None, None, None
    try:
        inner = _controller(task, hook)
        logged = CommandLog(inner)
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
            wall_seconds=float(task.get("wall_seconds", lm.CAPS_SECONDS["per_attempt"])),
        )
    except ps.GuardError as error:
        if not str(error).startswith(lp.SHIFT_BLOCKED_PREFIX):
            raise
        blocked = str(error)
    finally:
        hook.remove()
    if blocked is not None:
        out |= _blocked_out(hook)
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
            g_star = c1.fixed_point(p, h)  # the rule's fixed point (reported offline aim error)
            out["commit"]["fixed_point_error_cm"] = 100.0 * float(np.linalg.norm(target - g_star))
    out["frame405"] = hook.frames.get(lm.COMMIT_STEP)
    if corpus:
        out["corpus"] = corpus_arrays(hook, logged, decisions)
    out["seconds"] = time.monotonic() - started
    return out


def corpus_arrays(hook: CorpusMotion, logged: CommandLog | None, decisions) -> dict:
    """A root's kept arrays, or ``{"complete": False, "why": ...}`` when it does not reach 467."""
    lo, hi = lm.FRAME_STEPS
    c_lo, c_hi = lm.COMMAND_STEPS
    why = None
    if not decisions or decisions[0].get("fallback"):
        why = "fallback" if decisions else "no_decision"
    elif any(t not in hook.frames for t in range(lo, hi + 1)):
        why = "frames_missing"
    elif logged is None or any(t not in logged.executed for t in range(c_lo, c_hi + 1)):
        why = "commands_missing"
    elif hook.hidden_r is None:
        why = "hidden_missing"
    if why is not None:
        return {"complete": False, "why": why}
    plate, palm = hook.plate_kept, hook.state["palm"]
    first = decisions[0]
    return {
        "complete": True,
        "frames": np.stack([hook.frames[t] for t in range(lo, hi + 1)]),
        "commands": np.stack([logged.executed[t] for t in range(c_lo, c_hi + 1)]),
        "plate": np.stack([plate[t] for t in range(lo, hi + 1)]),
        "palm": np.stack([palm[t] for t in range(lo, hi + 1)]),
        "hidden_r": hook.hidden_r,
        "target": np.asarray(first["target"], np.float64),
        "state405": np.asarray(first["state405"], np.float64),
        "apple_estimate": np.asarray(first["apple_estimate"], np.float64),
        "last_grasp": np.asarray(first["last_grasp"], np.float64),
    }


# ----- the stand-in's chunks (Stage G, reported; the offline aims) ------------------------------
def run_chunks_task(task: dict) -> dict:
    """``primitive_chunks`` (horizon 60) from a logged 405 state for the given targets."""
    W = rtm._W
    _req, applied, feasible = pp.primitive_chunks(
        rtm.standin(),
        W["fk"],
        np.asarray(task["apple"], np.float64),
        np.asarray(task["state"], np.float64),
        int(task["step"]),
        np.asarray(task["targets"], np.float64).reshape(-1, 2),
        last_grasp=tuple(float(v) for v in task["last_grasp"]),
        horizon=lm.HORIZON,
    )
    return {"key": task["key"], "chunks": applied, "feasible": feasible}


def run_offline_aim_task(task: dict) -> dict:
    """Stage G's offline aim (reported only): W's controller form from a gate root's logged 405
    state, R-plate's reading and a start latent (``"mean"`` for L-mean), on the CPU."""
    W = rtm._W
    arm = task["arm"]
    state = np.asarray(task["state"], np.float64)
    apple = np.asarray(task["apple"], np.float64)
    last_grasp = tuple(float(v) for v in task["last_grasp"])
    p_hat, h = np.asarray(task["p_hat"], np.float64), np.asarray(task["h"], np.float64)
    targets = np.asarray([c1.from_box(a, b, p_hat, h) for a, b in lm.GRID])
    _req, chunks, feasible = pp.primitive_chunks(
        rtm.standin(),
        W["fk"],
        apple,
        state,
        lm.COMMIT_STEP,
        targets,
        last_grasp=last_grasp,
        horizon=lm.HORIZON,
    )

    def chunk_of(g):
        _r, applied, ok = pp.primitive_chunks(
            rtm.standin(),
            W["fk"],
            apple,
            state,
            lm.COMMIT_STEP,
            np.asarray(g)[None],
            last_grasp=last_grasp,
            horizon=lm.HORIZON,
        )
        return applied[0], bool(ok[0])

    predict = None
    if arm != "L-rand":
        start = task["start"]
        start = mean_latent() if isinstance(start, str) and start == "mean" else start
        model, r8, zero = world_model(arm), ptr._readout("r8"), arm == "N"

        def predict(commands):
            return rollout_plates(model, np.asarray(start, np.float32), commands, r8, zero=zero)

    tolerance = lm.REFINE_FRACTION_OF_TAU * float(task["tau_commit_cm"]) / 100.0
    g, log = choose_from_grid(
        arm,
        p_hat,
        h,
        targets,
        chunks,
        feasible,
        predict,
        chunk_of,
        tolerance_m=tolerance,
        seed=int(task["seed"]),
    )
    log.pop("scores_cm", None)
    return {"key": task["key"], "arm": arm, "g": np.asarray(g).tolist(), "log": log}


def run_task(task: dict) -> dict:
    if task["kind"] == "offline_aim":
        return run_offline_aim_task(task)
    if task["kind"] == "attempt":
        return run_attempt_task(task)
    if task["kind"] == "chunks":
        return run_chunks_task(task)
    return ptr.run_task(task)
