"""TASK-082 simulation workers (protocol ``docs/experiments/apple_lewm_unknown_law_v2.md`` §2, §4.3,
§6, §7): one attempt of any arm under U-sat, one corpus root with its ceiling label, the
kinematic stand-in's chunks and the offline aims of Stage G.

Reused unchanged: TASK-076's worker (``plate_twin_v2_runtime``: the v2 robot, P-3, its private
kinematics, the bounds, ``TwinAim``, ``LookaheadAim``, the checked ``_readout``); C1's
``CommitController``, ``GeometricAim``, ``RuleCommit`` (C1-M's written law, kappa = -0.5),
``SysidAim``; C1-M's ``PlantedLookahead`` and ``ReadLookahead``; TASK-077's ``CorpusMotion``,
``CommandLog``, ``encode``, the corpus arrays and the stand-in chunks; TASK-081's
``choose_affine_local`` (the solver, unchanged); the design note's U-sat law
(``plate_law_dev_runtime.LawMotion.target``, pinned) and kernel ridge (``plate_law_dev``).
New here:

* ``UlMotion``: TASK-077's ``CorpusMotion`` whose plate follows **U-sat** after 405 (the design
  note's ``LawMotion.target``, also inside every look-ahead branch, which calls the same hook);
  every record logs the law (G-law);
* ``UlAim``: W, N, L-shuf, L-mean, L-rand and the spread arms ``W-<seed>``: TASK-080's
  ``PredReadoutAim`` form (the encoded 405 frame, the p-hat-and-h grid, the stand-in chunks, each
  arm's own model and readout, R-S or R-N, never R8) committed by TASK-081's affine_local; L-rand
  with salt 8412, sub-key 1;
* ``CollectUl``: the corpus collector: H-final(commit)'s look-ahead once at 405 **in cloned
  state** (the label, privileged corpus data), then the root's own aim (a, b) over the box built
  from the true plate (train, val) or from p-hat (gate-P); p-hat logged on every root;
* ``KrrHatAim`` (H-sysid-krr), ``PAim`` (P-aim) and ``RuleFitCommit`` (H-rule-fit), the reported
  learned tier and the fitted-kappa rule, each reading R-plate's p-hat and the palm only;
* ``ReadingAim``: K0's planted look-ahead plus R-plate's CPU reading of the 405 frame, logged for
  the clip-binding check (it never changes the aim).

Who reads what follows TASK-071's rule (G-privileged): every non-privileged arm reads the
observation and the robot's own joint state only. NumPy at import; MuJoCo and torch are imported
by the worker initializer.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa import first_policy_v2_runtime as rt2
from embodied_jepa import lewm_c1m_v2_runtime as wrt
from embodied_jepa import lewm_cp_v2_runtime as cprt
from embodied_jepa import lewm_next_c1 as c1
from embodied_jepa import lewm_next_c1_runtime as c1rt
from embodied_jepa import lewm_next_c1m_runtime as c1mrt
from embodied_jepa import lewm_planner_v2 as lp
from embodied_jepa import lewm_planner_v2_runtime as lrt
from embodied_jepa import lewm_ul_v2 as ul
from embodied_jepa import place_planner as pp
from embodied_jepa import plate_law_dev as pl
from embodied_jepa import plate_law_dev_runtime as pldr
from embodied_jepa import plate_shift as ps
from embodied_jepa import plate_twin_v2 as pt
from embodied_jepa import plate_twin_v2_runtime as ptr
from embodied_jepa import wm_critic_v2_runtime as rtm

_U: dict = {}
ROLLOUT_CHUNK = wrt.ROLLOUT_CHUNK
LAW_RECORD = {k: ul.LAW[k] for k in ("name", "dev_law", "amplitude_m", "scale_m", "swirl_rad",
                                      "L", "s0", "s1")}  # fmt: skip


def worker_init(config: dict) -> None:
    """``config``: TASK-077's worker keys (``p3_checkpoint``, ``torch_threads``, ``r_plate``,
    ``mean_latent`` and their sha256) plus ``models`` (name -> {path, sha256, seed, metadata}
    for "W", "N" and every "W-<seed>"), the readouts ``r_s``, ``r_n`` and ``r_s_<seed>`` with
    their sha256, and ``r8`` only for the privileged H-read."""
    wrt.worker_init(config)
    _U.clear()
    _U["config"] = dict(config)


# ----- the hook: U-sat after the step-300 move ---------------------------------------------------
class UlMotion(wrt.CorpusMotion):
    """TASK-077's ``CorpusMotion`` (the step-300 move, the frames, the plate-hidden render at r)
    whose plate follows U-sat after 405: ``target`` is the design note's ``LawMotion.target`` with
    ``law = "sat"`` (pinned), so every look-ahead branch, which applies the same hook, runs under
    U-sat too."""

    law = ul.LAW_NAME
    target = pldr.LawMotion.target

    def summary(self) -> dict:
        out = super().summary()
        out["law"] = dict(LAW_RECORD)
        out["kappa"] = None  # C1Motion's cell-A kappa is not the plate's law here
        return out


def check_law(record: dict) -> None:
    """G-law: a record ran under U-sat at the frozen parameters."""
    law = (record.get("kpred") or {}).get("law")
    if law != LAW_RECORD:
        raise lp.GuardError(f"G-law: seed {record.get('seed')} ran under {law!r}, not U-sat")


# ----- models and readouts on the CPU ------------------------------------------------------------
def model_name(arm: str) -> str:
    """The model an arm rolls out: N's for N, a spread arm's own seed, W's otherwise."""
    if arm == "N":
        return "N"
    if ul.spread_seed(arm) is not None:
        return arm
    return "W"


def readout_name(arm: str) -> str:
    """R-S or R-N of the primary seed, or R-S of a spread arm's seed; never R8 (G-readout)."""
    seed = ul.spread_seed(arm)
    if seed is not None:
        return f"r_s_{seed}"
    name = ul.READOUTS.get(arm)
    if name is None:
        raise lp.GuardError(f"G-readout: {arm} reads no predicted-latent readout")
    return name


def world_model(name: str):
    """A trained model on the CPU (G-checkpoint against its sha256)."""
    from embodied_jepa import lewm_c1m_v2_train as tr

    key = ("model", name)
    if key not in _U:
        spec = _U["config"]["models"][name]
        _U[key] = tr.load_model(
            spec["path"],
            seed=int(spec["seed"]),
            metadata=dict(spec["metadata"]),
            expected_sha256=spec["sha256"],
        )
    return _U[key]


# ----- the candidate arms' choice after the grid -------------------------------------------------
def choose_l_rand(arm, p_hat, targets, feasible, *, seed: int):
    """L-rand with salt 8412, sub-key 1: TASK-081's L-rand branch with this task's draw."""
    p_hat = np.asarray(p_hat, np.float64).reshape(2)
    targets = np.asarray(targets, np.float64).reshape(-1, 2)
    feasible = np.asarray(feasible, bool)
    log = {"arm": arm, "solver": "none", "feasible": int(feasible.sum()),
           "candidates": int(len(targets))}  # fmt: skip
    if not feasible.any():
        log |= {"fallback_all_infeasible": True, "clipped": False, "rollouts": 0}
        return p_hat.copy(), log
    i = ul.l_rand_index(seed, feasible)
    log |= {"fallback_all_infeasible": False, "grid_index": i, "grid_best": list(ul.GRID[i]),
            "clipped": False, "rollouts": 0}  # fmt: skip
    return targets[i].copy(), log


def choose(arm, p_hat, h, targets, chunks, feasible, predict, chunk_of, *, seed: int):
    """L-rand draws; every other candidate arm runs TASK-081's ``choose_affine_local`` unchanged."""
    if arm == "L-rand":
        return choose_l_rand(arm, p_hat, targets, feasible, seed=seed)
    return cprt.choose_affine_local(arm, p_hat, h, targets, chunks, feasible, predict, chunk_of)


def solver_of(arm: str) -> str:
    if ul.spread_seed(arm) is not None:
        return "affine_local"
    return ul.SOLVER_OF[arm]


class UlAim(ptr.TwinAim):
    """§6.1 at 405 for W, N, L-shuf, L-mean, L-rand and the spread arms: TASK-080's
    ``PredReadoutAim`` with this task's models, readouts and solver. ``start``: "own" (W, N, the
    spread arms), a foreign frame (L-shuf) or "mean" (L-mean); L-rand reads no model."""

    def __init__(self, encoder, readout, *, arm: str, start, seed: int):
        super().__init__(encoder, readout)
        self.arm, self.start, self.seed = arm, start, int(seed)

    def __call__(self, ctl, observation, step, record):
        started = time.perf_counter()
        frame = np.asarray(observation.images[fp2.CAMERA][0])
        tokens, own = wrt.encode(frame)
        p_hat = np.asarray(self.readout.predict(tokens), np.float64).reshape(2)
        self.readings.append((int(step), p_hat))
        record["reading"] = p_hat.tolist()
        h = np.asarray(ctl.palm[int(step)], np.float64)
        state = rt2.state_of(observation)
        targets = np.asarray([c1.from_box(a, b, p_hat, h) for a, b in ul.GRID])
        _req, chunks, feasible = pp.primitive_chunks(
            rtm.standin(), ctl.fk, ctl.apple, state, int(step), targets,
            last_grasp=ctl.last_grasp, horizon=ul.HORIZON,
        )  # fmt: skip

        def chunk_of(g):
            _r, applied, ok = pp.primitive_chunks(
                rtm.standin(), ctl.fk, ctl.apple, state, int(step), np.asarray(g)[None],
                last_grasp=ctl.last_grasp, horizon=ul.HORIZON,
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
            model = world_model(model_name(self.arm))
            readout = ptr._readout(readout_name(self.arm))
            zero = self.arm == "N"

            def predict(commands):
                return wrt.rollout_plates(model, start, commands, readout, zero=zero)

            record["start_latent_sha256"] = hashlib.sha256(
                np.ascontiguousarray(start, np.float32).tobytes()
            ).hexdigest()
            record["readout"] = readout_name(self.arm)
            record["model"] = model_name(self.arm)
        g, log = choose(self.arm, p_hat, h, targets, chunks, feasible, predict, chunk_of,
                        seed=self.seed)  # fmt: skip
        log["seconds"] = time.perf_counter() - started
        record["world_model"] = log
        c1rt._box_record(record, g, p_hat, h)
        return g


# ----- the learned tier and the fitted-kappa rule (reported only) --------------------------------
def krr_from_json(model: dict) -> dict:
    return {k: (np.asarray(v, np.float64) if isinstance(v, list) else v) for k, v in model.items()}


def p_aim_predict(model: dict, p_hat, h) -> np.ndarray:
    """P-aim's aim before the clip: p-hat + the kernel ridge's g - p-hat on standardised
    (p-hat, h)."""
    p_hat = np.asarray(p_hat, np.float64).reshape(-1, 2)
    x = np.concatenate([p_hat, np.asarray(h, np.float64).reshape(-1, 2)], axis=1)
    return p_hat + pl.krr_predict(model, x)


class KrrHatAim(ptr.TwinAim):
    """H-sysid-krr: the design note's kernel ridge of plate(r) - p-hat on (p-hat, h, g), fitted on
    this corpus's train split, inverted with H-sysid's controller form (``choose_aim``: grid,
    refinement, clip); its plate input is R-plate's reading."""

    def __init__(self, encoder, readout, model: dict, *, a_lo: float):
        super().__init__(encoder, readout)
        self.model, self.a_lo = krr_from_json(model), float(a_lo)

    def __call__(self, ctl, observation, step, record):
        p = self.read(observation, step)
        record["reading"] = p.tolist()
        h = np.asarray(ctl.palm[int(step)], np.float64)

        def predict(g):
            return pl.krr_plate(self.model, p, h, g)[0]

        g, log = c1.choose_aim(predict, p, h, self.a_lo)
        record["sysid"] = log
        c1rt._box_record(record, g, p, h)
        return g


class PAim(ptr.TwinAim):
    """P-aim: the kernel ridge of the ceiling's logged aim (as g - p-hat) on (p-hat, h), predicted
    once at 405 and clipped to the box; no forward model and no solver."""

    def __init__(self, encoder, readout, model: dict, *, a_lo: float):
        super().__init__(encoder, readout)
        self.model, self.a_lo = krr_from_json(model), float(a_lo)

    def __call__(self, ctl, observation, step, record):
        p = self.read(observation, step)
        record["reading"] = p.tolist()
        h = np.asarray(ctl.palm[int(step)], np.float64)
        raw = p_aim_predict(self.model, p, h)[0]
        g = c1.clip_to_box(raw, p, h, self.a_lo)
        record["p_aim"] = {"unclipped": raw.tolist(), "clipped": bool(np.any(g != raw))}
        c1rt._box_record(record, g, p, h)
        return g


class RuleFitCommit(c1rt.RuleCommit):
    """H-rule-fit: H-rule's form (``RuleCommit``) with kappa fitted on the train split."""

    def __init__(self, encoder, readout, standin, *, kappa: float, a_lo: float):
        super().__init__(encoder, readout, standin, truth_hook=None, a_lo=a_lo)
        pt.check_kappa(kappa)
        self.kappa = float(kappa)

    def __call__(self, ctl, observation, step, record):
        g = super().__call__(ctl, observation, step, record)
        record["rule"]["kappa"] = self.kappa
        return g


class ReadingAim:
    """K0's planted look-ahead plus R-plate's CPU reading of the 405 frame (logged only, for the
    clip-binding check; it never changes the aim)."""

    def __init__(self, inner, readout):
        self.inner, self.readout = inner, readout
        self.branch_steps = 0

    def __call__(self, ctl, observation, step, record):
        g = self.inner(ctl, observation, step, record)
        self.branch_steps = getattr(self.inner, "branch_steps", 0)
        frame = np.asarray(observation.images[fp2.CAMERA][0])
        tokens, _pooled = wrt.encode(frame)
        reading = np.asarray(self.readout.predict(tokens), np.float64).reshape(2)
        record["p_hat405"] = reading.tolist()
        return g


# ----- the corpus collector with its ceiling label ----------------------------------------------
class CollectUl(c1rt.GeometricAim):
    """The corpus's privileged scripted collector (§4.3): at 405, with ``label``, H-final(commit)'s
    look-ahead runs once in cloned state (``LookaheadAim``: branches under U-sat; the brancher
    restores the live state) and its aim is logged as the root's label with its convergence flag;
    the root then commits its own (a, b) over the box built from the true plate (train, val) or
    from p-hat (gate-P), with the palm h. p-hat (R-plate's reading of the onboard 405 frame on the
    CPU) is logged on every root. Privileged data, not a learned result."""

    def __init__(self, robot, hook, bounds, task: dict, readout):
        super().__init__("collect", hook, task)
        if task.get("aim_from") not in ("p_hat", "true"):
            raise lp.GuardError("G-corpus: a root's aim is built from p_hat or the true plate")
        self.readout = readout
        self.label = bool(task.get("label", True))
        self.lookahead = (
            ptr.LookaheadAim(robot, hook, bounds, ul.LOOKAHEAD_TOLERANCE_M, ul.LOOKAHEAD_MAX_ITER)
            if self.label
            else None
        )

    def __call__(self, ctl, observation, step, record):
        if self.lookahead is not None:
            sub: dict = {}
            g_label = np.asarray(self.lookahead(ctl, observation, step, sub), np.float64)
            info = sub["lookahead"]
            record["label"] = {
                "aim": g_label.tolist(),
                "converged": bool(info["converged"]),
                "stopped": info["stopped"],
                "iterations": len(info["iterations"]),
                "branch_steps": int(self.lookahead.branch_steps),
            }
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


# ----- the controllers ---------------------------------------------------------------------------
def _common(task: dict):
    W = rtm._W
    _model, standardiser = W["p3"]
    return (W["predict"], standardiser, np.asarray(task["estimates"], np.float64), W["fk"],
            W["bounds"])  # fmt: skip


def is_candidate(arm: str) -> bool:
    return arm in ul.CANDIDATE_ARMS or ul.spread_seed(arm) is not None


def _controller(task: dict, hook: UlMotion):
    arm = task["arm"]
    W = rtm._W

    def commit(aim):
        return c1rt.CommitController(*_common(task), arm=arm, aim=aim)

    if is_candidate(arm):
        start = {"W": "own", "N": "own", "L-mean": "mean", "L-rand": None}.get(arm, "own")
        if arm == "L-shuf":  # None only when no other reset reached 405 (R17.20)
            frame = task["foreign_frame"]
            start = None if frame is None else np.asarray(frame, np.uint8)
        return commit(UlAim(rtm.encoder(), ptr._readout("r_plate"), arm=arm, start=start,
                            seed=int(task["seed"])))  # fmt: skip
    if arm == "collect-ul":
        return commit(CollectUl(W["robot"], hook, W["bounds"], task, ptr._readout("r_plate")))
    if arm == "H-sysid-krr":
        return commit(KrrHatAim(rtm.encoder(), ptr._readout("r_plate"), task["krr"],
                                a_lo=task["a_lo"]))  # fmt: skip
    if arm == "P-aim":
        return commit(PAim(rtm.encoder(), ptr._readout("r_plate"), task["p_aim"],
                           a_lo=task["a_lo"]))  # fmt: skip
    if arm == "H-rule-fit":
        return commit(RuleFitCommit(rtm.encoder(), ptr._readout("r_plate"), rtm.standin(),
                                    kappa=float(task["kappa_fit"]), a_lo=task["a_lo"]))  # fmt: skip
    if arm == "H-rule" and c1.RULE["kappa"] != ul.H_RULE_LAW["kappa"]:
        raise lp.GuardError("G-law: H-rule runs only with C1-M's written kappa = -0.5")
    if arm == "H-final-planted" and task.get("log_p_hat"):
        inner = c1mrt.PlantedLookahead(W["robot"], hook, W["bounds"], c1.LOOKAHEAD_TOLERANCE_M,
                                       c1.MAX_ITERATIONS, task["planted_m"])  # fmt: skip
        return commit(ReadingAim(inner, ptr._readout("r_plate")))
    if arm not in ("H-final", "H-rule", "H-sysid", "H-read", "H-now", "H-final-planted"):
        raise lp.GuardError(f"not a TASK-082 arm: {arm!r}")
    return wrt._controller(task, hook)  # H-final, H-rule, H-sysid, H-read, H-now, planted


def run_attempt_task(task: dict) -> dict:
    """One attempt (or one corpus root, ``arm == "collect-ul"``): TASK-080's ``run_attempt_task``
    body with U-sat's hook and this module's controllers; the commit record keeps the plate at s1
    (U-sat has no closed-form fixed point)."""
    started = time.monotonic()
    W = rtm._W
    robot = W["robot"]
    truth, scorer, counter, _obs, facts = rt2.reset_and_look(robot, task["seed"], task["reset"])
    corpus = task["arm"] == "collect-ul"
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
    hook = UlMotion(robot, W["fk"], task["move_offset"], corpus=corpus)
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
            max_steps=ul.STOP_STEP + 1
            if corpus
            else int(task.get("max_steps", fp2.MAX_POLICY_STEPS)),
            settle_steps=0 if corpus else int(task.get("settle_steps", fp2.SETTLE_STEPS)),
            wall_seconds=float(task.get("wall_seconds", ul.CAPS_SECONDS["per_attempt"])),
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
            and int(record["executed_steps"]) >= ul.MOVE_STEP
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
    out["law"] = dict(LAW_RECORD)
    aim = getattr(inner, "aim", None)
    if isinstance(aim, ptr.LookaheadAim | ReadingAim):
        out["kpred"]["lookahead_branch_steps"] = aim.branch_steps
    plate, palm = hook.state["plate"], hook.state["palm"]
    if decisions and ul.COMMIT_STEP in plate and ul.COMMIT_STEP in palm:
        first = decisions[0]
        p, h = plate[ul.COMMIT_STEP], palm[ul.COMMIT_STEP]
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
            or bool(first.get("rule", {}).get("clipped", False))
            or bool(first.get("p_aim", {}).get("clipped", False)),
        }
        if "lookahead" in first:
            out["commit"]["lookahead_converged"] = bool(first["lookahead"].get("converged"))
        s1 = ul.LAW["s1"]
        if s1 in plate:
            out["commit"]["plate_s1"] = plate[s1].tolist()
            out["commit"]["landing_miss_cm"] = 100.0 * float(np.linalg.norm(target - plate[s1]))
            out["commit"]["plate_motion_cm"] = 100.0 * float(np.linalg.norm(plate[s1] - p))
    out["frame405"] = hook.frames.get(ul.COMMIT_STEP)
    if corpus:
        out["corpus"] = corpus_arrays(hook, logged, decisions)
    out["seconds"] = time.monotonic() - started
    return out


def corpus_arrays(hook, logged, decisions) -> dict:
    """TASK-077's root arrays plus p-hat at 405, the split's aim construction and the label."""
    arrays = wrt.corpus_arrays(hook, logged, decisions)
    if arrays["complete"]:
        first = decisions[0]
        arrays["p_hat405"] = np.asarray(first["p_hat405"], np.float64)
        arrays["aim_from"] = str(first["aim_from"])
        label = first.get("label")
        arrays["label_aim"] = (
            np.full(2, np.nan) if label is None else np.asarray(label["aim"], np.float64)
        )
        arrays["label_converged"] = np.asarray(bool(label and label["converged"]))
        arrays["labelled"] = label is not None
    return arrays


# ----- offline aims (Stage G, §9.4) ---------------------------------------------------------------
def _standin_setup(task: dict):
    W = rtm._W
    state = np.asarray(task["state"], np.float64)
    apple = np.asarray(task["apple"], np.float64)
    last_grasp = tuple(float(v) for v in task["last_grasp"])

    def chunks_of(targets):
        _req, applied, ok = pp.primitive_chunks(
            rtm.standin(), W["fk"], apple, state, ul.COMMIT_STEP,
            np.asarray(targets, np.float64).reshape(-1, 2),
            last_grasp=last_grasp, horizon=ul.HORIZON,
        )  # fmt: skip
        return applied, ok

    return state, apple, last_grasp, chunks_of


def offline_rule_aim(task: dict, p_hat, h, kappa: float):
    """``RuleCommit``'s iteration from a logged 405 state (TASK-080's ``offline_rule_aim`` with the
    gain given: C1-M's -0.5 for H-rule, the fitted kappa for H-rule-fit), clipped to the box."""
    W = rtm._W
    state, apple, last_grasp, _ = _standin_setup(task)
    lag, s0, s1 = int(c1.RULE["L"]), int(c1.RULE["s0"]), int(c1.RULE["s1"])
    if max(ul.COMMIT_STEP - lag, s0) != ul.COMMIT_STEP:
        raise lp.GuardError("the offline H-rule needs palm(405) as its reference")
    ref = np.asarray(h, np.float64)
    g, steps, converged = np.asarray(p_hat, np.float64).copy(), [], False
    for _ in range(c1.MAX_ITERATIONS):
        palm_end = ptr.standin_palm_xy(
            rtm.standin(), W["fk"], apple, state, ul.COMMIT_STEP, g,
            last_grasp=last_grasp, horizon=s1 - lag - ul.COMMIT_STEP,
        )  # fmt: skip
        if palm_end is None:
            steps.append({"stopped": "unreachable_or_refused"})
            break
        nxt = p_hat + float(kappa) * (palm_end - ref)
        gap = float(np.linalg.norm(nxt - g))
        steps.append({"gap_cm": 100.0 * gap})
        g = nxt
        if gap <= c1.LOOKAHEAD_TOLERANCE_M:
            converged = True
            break
    clipped = c1.clip_to_box(g, p_hat, h, ul.A_LO)
    return clipped, {"iterations": len(steps), "converged": converged,
                     "clipped": bool(np.any(clipped != g)), "kappa": float(kappa)}  # fmt: skip


def run_offline_aim_task(task: dict) -> dict:
    """One arm's offline aim from a logged 405 state (§9.4): the grid from the p-hat given, the
    stand-in chunks, the arm's own controller code and readout. ``start``: a latent (W, N,
    L-shuf, the spread arms) or ``"mean"`` (L-mean); the hand-written and learned-tier arms read no
    model. With ``keep_plates`` W's predicted plate over the feasible grid is returned (the echo
    slope)."""
    arm = task["arm"]
    p_hat, h = np.asarray(task["p_hat"], np.float64), np.asarray(task["h"], np.float64)
    if arm in ("H-rule", "H-rule-fit"):
        kappa = ul.H_RULE_LAW["kappa"] if arm == "H-rule" else float(task["kappa_fit"])
        g, log = offline_rule_aim(task, p_hat, h, kappa)
        return {"key": task["key"], "arm": arm, "g": g.tolist(), "log": log}
    if arm in ("H-sysid", "H-sysid-krr"):
        if arm == "H-sysid":
            coef = np.asarray(task["sysid_coef"], np.float64)

            def predict_plate(g):
                return c1.sysid_predict(coef, p_hat, h, g)[0]
        else:
            model = krr_from_json(task["krr"])

            def predict_plate(g):
                return pl.krr_plate(model, p_hat, h, g)[0]

        g, log = c1.choose_aim(predict_plate, p_hat, h, ul.A_LO)
        log = {"converged": bool(log["converged"]), "clipped": False}  # choose_aim clips
        return {"key": task["key"], "arm": arm, "g": np.asarray(g).tolist(), "log": log}
    if arm == "P-aim":
        raw = p_aim_predict(krr_from_json(task["p_aim"]), p_hat, h)[0]
        g = c1.clip_to_box(raw, p_hat, h, ul.A_LO)
        return {"key": task["key"], "arm": arm, "g": g.tolist(),
                "log": {"clipped": bool(np.any(g != raw))}}  # fmt: skip
    if not is_candidate(arm):
        raise lp.GuardError(f"no offline aim for {arm!r}")
    _state, _apple, _lg, chunks_of = _standin_setup(task)
    targets = np.asarray([c1.from_box(a, b, p_hat, h) for a, b in ul.GRID])
    chunks, feasible = chunks_of(targets)

    def chunk_of(g):
        applied, ok = chunks_of(np.asarray(g)[None])
        return applied[0], bool(ok[0])

    predict, seen = None, {}
    if arm != "L-rand":
        start = task["start"]
        start = wrt.mean_latent() if isinstance(start, str) and start == "mean" else start
        start = np.asarray(start, np.float32)
        model, readout = world_model(model_name(arm)), ptr._readout(readout_name(arm))
        zero = arm == "N"

        def predict(commands):
            out = wrt.rollout_plates(model, start, commands, readout, zero=zero)
            if "grid" not in seen:
                seen["grid"] = out.copy()
            return out

    g, log = choose(arm, p_hat, h, targets, chunks, feasible, predict, chunk_of,
                    seed=int(task["seed"]))  # fmt: skip
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
    return ptr.run_task(task)  # frames and G-repro
