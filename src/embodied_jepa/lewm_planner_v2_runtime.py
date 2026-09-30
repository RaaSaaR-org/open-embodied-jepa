"""TASK-074 simulation workers: one attempt of any arm, one corpus root (TASK-073's collector),
one ranking reset and one P-far DAgger rollout.

Protocol ``docs/experiments/apple_lewm_planner_v2.md``. The worker is TASK-073's
(``wm_critic_v2_runtime.worker_init``). It builds the v2 robot, P-3's private kinematics, the
bounds, P-3, the stand-in and the perturber, and loads the frozen encoder lazily. On top of it,
this module loads TASK-074's world models, readouts and P-far from files on first use.

Where privileged information goes is TASK-071's rule, unchanged. The reset truth is read by the
harness only. An L1 controller holds no robot handle. ``PrivilegedReadCounter`` counts every truth
read and every read inside ``act``. The plate shift is a harness hook outside ``act``.

NumPy at import; MuJoCo and torch are imported by the worker initializer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa import first_policy_v2_runtime as rt2
from embodied_jepa import hybrid_selection as hs
from embodied_jepa import lewm_planner_v2 as lp
from embodied_jepa import place_planner as pp
from embodied_jepa import plate_shift as ps
from embodied_jepa import sim_selector as ss
from embodied_jepa import wm_critic_v2_runtime as rtm

_X: dict = {}
frame_sha = rtm.frame_sha
reset_worker_signals = rtm.reset_worker_signals


def worker_init(config: dict) -> None:
    """``config``: ``p3_checkpoint``, ``torch_threads`` and optional ``models`` (name -> W/N
    checkpoint), ``r_off``, ``r_plate`` (npz), ``p_far`` (P-far checkpoint)."""
    rtm.worker_init(config)
    _X.clear()
    _X["config"] = dict(config)
    _X["cache"] = {}


def _readout(name: str):
    from embodied_jepa.models.latent_critic import RidgeReadout

    key = ("readout", name)
    if key not in _X["cache"]:
        with np.load(_X["config"][name]) as data:
            _X["cache"][key] = RidgeReadout.from_state({k: data[k] for k in data.files})
    return _X["cache"][key]


def critic(name: str):
    """``W0``/``W1``/``W2`` (the W seeds, W0 the val-selected primary), ``N`` or ``copy``, with
    this protocol's cost target (o_f at every target step)."""
    from embodied_jepa.models.latent_critic import CopyLast, LatentCritic, ZeroActions

    key = ("critic", name)
    if key not in _X["cache"]:
        if name == "copy":
            model = CopyLast()
        elif name == "N":
            model = ZeroActions(rtm._world_model(_X["config"]["models"]["N"]))
        else:
            model = rtm._world_model(_X["config"]["models"][name])
        _X["cache"][key] = LatentCritic(model, _readout("r_off"), lp.cost_targets(), lp.CHUNK)
    return _X["cache"][key]


def p_far(path: str | None = None):
    """P-far's (predict, standardiser) from a checkpoint (TASK-072's format); by default the
    configured one, in a DAgger rollout the iteration's."""
    import torch

    from embodied_jepa import first_policy_v2_model as fm2

    path = str(path or _X["config"]["p_far"])
    key = ("p_far", path)
    if key not in _X["cache"]:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = fm2.load(saved["state"], heads=int(saved["heads"]))
        standardiser = rt2.Standardiser(saved["mean"].numpy(), saved["std"].numpy())
        _X["cache"][key] = (fm2.predictor(model), standardiser)
    return _X["cache"][key]


def plate_schedule(reset: dict, shift: dict | None):
    return rtm.plate_schedule(reset, shift)


def _controller(task: dict, truth: dict):
    """The arm's controller. L1 arms receive only the harness's post-look estimates."""
    from embodied_jepa.policy_diagnostics import RandomController, hold_controller

    W = rtm._W
    arm = task["arm"]
    robot, fk, bounds = W["robot"], W["fk"], W["bounds"]
    _model, standardiser = W["p3"]
    estimates = np.asarray(task.get("estimates", np.zeros(4)), np.float64)
    shift = task.get("shift")
    plate_at = plate_schedule(task["reset"], shift)
    common = (W["predict"], standardiser, estimates, fk, bounds)
    log = bool(task.get("log_chunks"))
    if arm == "P-stale":
        return rt2.LearnedController(*common)
    if arm == "P-truth":
        return hs.ScheduledTruth(
            *common,
            switch_step=int(shift["step"]) if shift else 0,
            plate_xy=plate_at(10**9),
        )
    if arm == "H-handover":
        return pp.PlannerController(
            *common,
            mode="truth",
            truth_target=plate_at,
            standin=rtm.standin() if log else None,
            log_chunks=log,
        )
    if arm == "H-twin":
        return pp.PlannerController(
            *common, mode="twin", encoder=rtm.encoder(), r_plate=_readout("r_plate")
        )
    if arm in lp.CRITIC_ARMS:
        foreign = None
        if arm == "L-shuf":
            foreign = {int(k): np.asarray(v, np.float32) for k, v in task["foreign"].items()}
        return pp.PlannerController(
            *common,
            mode="shuf" if arm == "L-shuf" else "critic",
            encoder=rtm.encoder(),
            critic=critic(lp.CRITIC_ARMS[arm]),
            standin=rtm.standin(),
            foreign_latents=foreign,
            keep_costs=bool(task.get("keep_costs")),
        )
    if arm == "L-rand":
        seed = np.random.SeedSequence([lp.SEEDS["l_rand"], int(task["seed"])])
        return pp.PlannerController(*common, mode="rand", rng_seed=int(seed.generate_state(1)[0]))
    if arm == "P-far":
        predict, far_std = p_far()
        return pp.RecordingReread(
            predict,
            far_std,
            estimates,
            fk,
            bounds,
            encoder=rtm.encoder(),
            r_mid=_readout("r_plate"),
        )
    if arm == "B-oracle-shift":
        offset = np.asarray(shift["vector"], np.float64) if shift else np.zeros(2)
        return rt2.ExpertController(rt2.perturbed_truth(truth, [0.0, 0.0], offset), robot)
    if arm == "B-hold":
        return rt2.HarnessController(hold_controller())
    if arm == "B-random":
        lower, upper = bounds
        seed = np.random.SeedSequence([lp.SEEDS["random_controller"], int(task["seed"])])
        return rt2.HarnessController(RandomController(int(seed.generate_state(1)[0]), lower, upper))
    raise lp.GuardError(f"unknown arm {arm!r}")


KEEP = (
    "termination_reason",
    "complete",
    "executed_steps",
    "at_rest",
    "latched_success",
    "grasp",
    "first_grasp_step",
    "first_place_step",
    "success",
    "task_truth_total",
    "task_truth_in_controller",
    "scorer_evaluations",
    "at_rest_records",
)


def run_attempt_task(task: dict) -> dict:
    robot = rtm._W["robot"]
    truth, scorer, counter, _obs, facts = rt2.reset_and_look(robot, task["seed"], task["reset"])
    out = {
        "seed": task["seed"],
        "arm": task["arm"],
        "post_look_frame_sha256": frame_sha(facts["post_look_frame"]),
        "post_look_state": facts["post_look_state"].tolist(),
        "truth_xy": rtm.truth_xy(truth),
    }
    try:
        out["render_retries"] = rtm.check_post_look_frame(robot, task, facts["post_look_frame"])
    except lp.GuardError:
        counter.remove()
        robot.stop("frame_mismatch")
        raise
    shift = task.get("shift")
    hook = ps.PlateShift(robot, int(shift["step"]), shift["vector"]) if shift else None
    try:
        inner = _controller(task, truth)
        controller = rtm.Timed(inner)
        record = rt2.run_attempt(
            robot,
            scorer,
            counter,
            controller,
            bounds=rtm._W["bounds"],
            max_steps=int(task.get("max_steps", fp2.MAX_POLICY_STEPS)),
            settle_steps=int(task.get("settle_steps", fp2.SETTLE_STEPS)),
            wall_seconds=float(task.get("wall_seconds", lp.CAPS_SECONDS["per_attempt"])),
        )
    except ps.GuardError as error:
        if not str(error).startswith(lp.SHIFT_BLOCKED_PREFIX):
            raise
        return out | blocked_record(task, str(error), inner)
    finally:
        if hook is not None:
            hook.remove()
    if shift and record["executed_steps"] >= int(shift["step"]) and not hook.applied:
        raise lp.GuardError("G-shift: the plate shift was not applied")
    detail = record.get("at_rest_detail") or {}
    out.update({k: record[k] for k in KEEP})
    out["final_distance_cm"] = (
        None if detail.get("final_distance_m") is None else 100.0 * detail["final_distance_m"]
    )
    out["privileged_ok"] = rt2.privileged_reads_ok(record)
    out["shift_log"] = hook.log if hook is not None else None
    out["commands"] = record["commands"]
    out["decisions"] = getattr(inner, "decisions", [])
    if isinstance(inner, pp.PlannerController | hs.HybridController):
        out["latents"] = {int(k): v for k, v in inner.latents.items()}
    if isinstance(inner, pp.PlannerController) and inner.chunks:
        out["o5_relative_errors"] = hs.relative_chunk_errors(inner.chunks, record["commands"])
    return out


def blocked_record(task: dict, reason: str, inner) -> dict:
    """``SHIFT_BLOCKED``: the move would have put the plate into something other than the table
    (the apple not yet lifted, or the hand). The attempt ends as a counted failure, not a V."""
    return {
        "termination_reason": "shift_blocked",
        "shift_blocked": reason,
        "complete": False,
        "executed_steps": int(task["shift"]["step"]),
        "at_rest": False,
        "latched_success": False,
        "grasp": None,
        "first_grasp_step": None,
        "first_place_step": None,
        "success": False,
        "final_distance_cm": None,
        "privileged_ok": True,
        "shift_log": None,
        "commands": np.zeros((0, 7), np.float32),
        "decisions": getattr(inner, "decisions", []),
        "latents": {int(k): v for k, v in getattr(inner, "latents", {}).items()},
    }


def run_collect_task(task: dict) -> dict:
    """One ``apple-far-shift-v2`` root with TASK-073's collector (e9 from the post-shift truth
    plus the root's mis-aim, TASK-048's noise, the shift, then the settle). Privileged, scripted.
    If the move is blocked (``SHIFT_BLOCKED``), the root is collected again unshifted, flagged."""
    try:
        return rtm.run_collect_task(task) | {"shift_blocked": None}
    except ps.GuardError as error:
        if not str(error).startswith(lp.SHIFT_BLOCKED_PREFIX) or task.get("shift") is None:
            raise
        rtm._W["robot"].stop("shift_blocked")
        return rtm.run_collect_task(task | {"shift": None}) | {"shift_blocked": str(error)}


def run_rank_task(task: dict) -> dict:
    """H-twin on one R reset with the shift. At each decision step:
    - the 49 coarse targets around the post-look estimate are each run for 16 commands in cloned
      state, giving the true outcome (privileged, scoring only);
    - their stand-in chunks are generated, and these are what every ranker scores;
    - the same is done for the 25 fine targets around the coarse target with the lowest true
      cost."""
    W = rtm._W
    robot = W["robot"]
    truth, scorer, counter, _obs, facts = rt2.reset_and_look(robot, task["seed"], task["reset"])
    counter.remove()  # the branches read truth for scoring by design; no controller is scored
    render_retries = rtm.check_post_look_frame(robot, task, facts["post_look_frame"])
    shift = task["shift"]
    plate_at = plate_schedule(task["reset"], shift)
    hook = ps.PlateShift(robot, int(shift["step"]), shift["vector"])
    _model, standardiser = W["p3"]
    main = pp.PlannerController(
        W["predict"],
        standardiser,
        np.asarray(task["estimates"], np.float64),
        W["fk"],
        W["bounds"],
        mode="twin",
        encoder=rtm.encoder(),
        r_plate=_readout("r_plate"),
    )
    brancher = ss.Brancher(robot)
    groups = []
    lower, upper = W["bounds"]
    stopped = None
    try:
        for step in range(lp.DECISION_STEPS[-1] + 1):
            observation = robot.observe()
            if step in pp.DECISION_SET:
                tokens, pooled = hs.encode_frame(rtm.encoder(), observation.images[fp2.CAMERA][0])
                reading = np.asarray(_readout("r_plate").predict(tokens), np.float64)
                state = rt2.state_of(observation)
                coarse = lp.coarse_targets(main.anchor)
                best = None
                for kind in ("coarse", "fine"):
                    targets = coarse if kind == "coarse" else lp.fine_targets(coarse[best])
                    _r, applied, feasible = pp.primitive_chunks(
                        rtm.standin(),
                        W["fk"],
                        main.apple,
                        state,
                        step,
                        targets,
                        last_grasp=main.last_grasp,
                    )
                    branches, index = [], []
                    for i, g in enumerate(targets):
                        try:
                            branches.append(
                                pp.BranchPlace(W["fk"], main.apple, g, step, main.last_grasp[1])
                            )
                            index.append(i)
                        except lp.ContractError:
                            continue
                    offsets = np.full((len(targets), 2), np.nan)
                    reasons = [None] * len(targets)
                    if branches:
                        got, why = brancher.outcomes(
                            branches, step, W["bounds"], plate_at(step + lp.CHUNK)
                        )
                        offsets[index] = got
                        for i, r in zip(index, why, strict=True):
                            reasons[i] = r
                    for i in set(range(len(targets))) - set(index):
                        reasons[i] = "unreachable"
                    true_costs = lp.terminal_cost_cm(offsets)
                    true_costs = np.where(np.isfinite(true_costs), true_costs, np.nan)
                    if kind == "coarse":
                        finite = np.where(np.isfinite(true_costs), true_costs, np.inf)
                        best = int(np.argmin(finite))
                    groups.append(
                        {
                            "step": step,
                            "kind": kind,
                            "targets": targets.tolist(),
                            "anchor": main.anchor.tolist(),
                            "reading": reading.tolist(),
                            "latent": pooled,
                            "chunks": applied,
                            "feasible": feasible,
                            "true_offsets_cm": offsets,
                            "true_costs_cm": true_costs,
                            "stopped": reasons,
                        }
                    )
            try:
                command = main.act(observation, step)
            except lp.ContractError:
                stopped = "controller"
                break
            command = np.clip(np.asarray(command, np.float32), lower, upper)
            try:
                projection = robot.project_candidates(command[None, None, None])
            except rt2.ContractError as error:
                if str(error) not in rt2.GUARD_REFUSALS:
                    raise
                stopped = "guard_refusal"
                break
            if not bool(projection.feasible[0, 0]):
                stopped = "infeasible_command"
                break
            result = robot.execute(projection.actions[0, 0, 0])
            if result.applied_action is None:
                stopped = result.reason or "rejected"
                break
            main.advance(result)
    finally:
        hook.remove()
        robot.stop("rank_done")
    return {
        "seed": task["seed"],
        "post_look_frame_sha256": frame_sha(facts["post_look_frame"]),
        "render_retries": render_retries,
        "groups": groups,
        "shift_applied": hook.applied,
        "stopped": stopped,
    }


def run_dagger_task(task: dict) -> dict:
    """One P-far DAgger rollout: P-far-k with the R-plate re-read, labelled by e9 built from the
    post-shift truth and advanced on the executed results (TASK-072's shadow expert). Returns the
    rows: each labelled step's raw input and label."""
    W = rtm._W
    robot = W["robot"]
    truth, scorer, counter, _obs, facts = rt2.reset_and_look(robot, task["seed"], task["reset"])
    rtm.check_post_look_frame(robot, task, facts["post_look_frame"])
    shift = task.get("shift")
    hook = ps.PlateShift(robot, int(shift["step"]), shift["vector"]) if shift else None
    offset = np.asarray(shift["vector"], np.float64) if shift else np.zeros(2)
    labeller = rt2.shadow_expert(rt2.perturbed_truth(truth, [0.0, 0.0], offset))
    predict, far_std = p_far(task["p_far"])
    controller = pp.RecordingReread(
        predict,
        far_std,
        np.asarray(task["estimates"], np.float64),
        W["fk"],
        W["bounds"],
        encoder=rtm.encoder(),
        r_mid=_readout("r_plate"),
    )
    try:
        record = rt2.run_attempt(
            robot,
            scorer,
            counter,
            controller,
            bounds=W["bounds"],
            max_steps=int(task.get("max_steps", fp2.MAX_POLICY_STEPS)),
            settle_steps=int(task.get("settle_steps", fp2.SETTLE_STEPS)),
            labeller=labeller,
        )
    finally:
        if hook is not None:
            hook.remove()
    steps = [int(s) for s in record["steps"]]
    inputs = [controller.inputs[s] for s in steps]
    return {
        "seed": task["seed"],
        "success": bool(record["success"]),
        "termination_reason": record["termination_reason"],
        "steps": steps,
        "inputs": np.asarray(inputs, np.float64).reshape(-1, 132),
        "labels": np.asarray(record["labels"], np.float32).reshape(-1, 7),
    }


def run_task(task: dict) -> dict:
    kind = task["kind"]
    if kind == "attempt":
        return run_attempt_task(task)
    if kind == "frame":
        return rtm.run_frame_task(task)
    if kind == "collect":
        return run_collect_task(task)
    if kind == "rank":
        return run_rank_task(task)
    if kind == "dagger":
        return run_dagger_task(task)
    raise lp.GuardError(f"unknown task kind {kind!r}")


def write_npz(folder: Path, name: str, arrays: dict) -> Path:
    path = Path(folder) / name
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    np.savez(path, **arrays)
    return path
