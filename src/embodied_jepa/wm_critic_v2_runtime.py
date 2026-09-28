"""TASK-073 simulation workers: one attempt of any arm, one corpus root, one ranking reset.

Protocol ``docs/experiments/apple_wm_critic_v2.md``. Every function here runs in a spawned
simulation worker (``worker_init`` then ``run_task``), so it lives in an importable module. The
worker builds the v2 robot (G-scene), P-3's private forward kinematics, the configured bounds,
P-3 itself (TASK-072 run-1's checkpoint, loaded from the path the harness passes after checking
its sha256), the frozen DINOv2 encoder on the CPU, and the kinematic stand-in; world models and
readouts are loaded from files on first use and cached.

Where privileged information goes is TASK-071's rule, unchanged: the reset truth is read by the
harness only; an L1 controller holds no robot handle; ``PrivilegedReadCounter`` counts every
truth read and every read inside ``act``. The plate shift is a harness hook outside ``act``.

NumPy at import; MuJoCo and torch are imported by ``worker_init``.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa import first_policy_v2_runtime as rt2
from embodied_jepa import hybrid_selection as hs
from embodied_jepa import plate_shift as ps
from embodied_jepa import sim_selector as ss
from embodied_jepa import wm_critic_v2 as wc

_W: dict = {}
L1_KINDS = {
    "P-stale",
    "P-reread",
    "H-LeWM",
    "H-LeWM-s1",
    "H-LeWM-s2",
    "H-N",
    "H-shuf",
    "H-rand",
    "H-copy",
}
CRITIC_ARMS = {
    "H-LeWM": "W0",
    "H-LeWM-s1": "W1",
    "H-LeWM-s2": "W2",
    "H-N": "N",
    "H-copy": "copy",
    "H-shuf": "W0",
}


def frame_sha(frame) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame, np.uint8).tobytes()).hexdigest()


def truth_xy(truth) -> list[float]:
    return [float(v) for v in (*truth["object_position"][:2], *truth["plate_position"][:2])]


def worker_init(config: dict) -> None:
    """``config``: ``p3_checkpoint``, ``torch_threads`` and optional ``models`` (name -> W/N
    checkpoint path), ``r_off``, ``r_mid`` (npz paths)."""
    import torch

    from embodied_jepa import first_policy_v2_model as fm2
    from embodied_jepa import pretrained_encoder as pe

    torch.set_num_threads(int(config.get("torch_threads", fp2.WORKER_TORCH_THREADS)))
    _W["config"] = dict(config)
    _W["robot"] = rt2.make_robot()
    _W["fk"] = rt2.PalmFK()
    _W["bounds"] = rt2.configured_bounds()
    saved = torch.load(config["p3_checkpoint"], map_location="cpu", weights_only=True)
    model = fm2.load(saved["state"], heads=int(saved["heads"]))
    _W["p3"] = (model, rt2.Standardiser(saved["mean"].numpy(), saved["std"].numpy()))
    _W["predict"] = fm2.predictor(model)
    _W["policy"] = hs.P3Policy(model, _W["p3"][1], _W["fk"])
    _W["encoder"] = pe.load_pretrained()
    _W["standin"] = hs.KinematicStandIn(_W["bounds"])
    _W["perturber"] = load_perturber()
    _W["cache"] = {}


def load_perturber():
    """TASK-048's seeded OU + burst perturbation, loaded unchanged from its pinned script."""
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts" / "collect_apple_wide.py"
    spec = importlib.util.spec_from_file_location("_collect_apple_wide", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Perturber


def _readout(name: str):
    from embodied_jepa.models.latent_critic import RidgeReadout

    key = ("readout", name)
    if key not in _W["cache"]:
        path = _W["config"][name]
        with np.load(path) as data:
            _W["cache"][key] = RidgeReadout.from_state({k: data[k] for k in data.files})
    return _W["cache"][key]


def _world_model(path: str):
    from embodied_jepa import token_dynamics as td
    from embodied_jepa.models.frozen_tokens import frozen_token_model

    key = ("model", path)
    if key not in _W["cache"]:
        model = frozen_token_model(td.BACKEND)(
            _W["robot"].state_schema, device="cpu", seed=0, config=td.MODEL_CONFIG
        )
        model.load(path)
        model.eval()
        _W["cache"][key] = model
    return _W["cache"][key]


def critic(name: str):
    """``W0``/``W1``/``W2`` (the W seeds, W0 the val-selected primary), ``N`` or ``copy``."""
    from embodied_jepa.models.latent_critic import CopyLast, LatentCritic, ZeroActions

    key = ("critic", name)
    if key not in _W["cache"]:
        if name == "copy":
            model = CopyLast()
        elif name == "N":
            model = ZeroActions(_world_model(_W["config"]["models"]["N"]))
        else:
            model = _world_model(_W["config"]["models"][name])
        _W["cache"][key] = LatentCritic(model, _readout("r_off"), wc.O_STAR_CM, wc.CHUNK)
    return _W["cache"][key]


def plate_schedule(reset: dict, shift: dict | None):
    """``step -> plate xy`` as the harness will have it (the reset plate, moved at the shift)."""
    base = np.asarray(reset["plate_xy"], np.float64)
    if shift is None:
        return lambda step: base.copy()
    moved = base + np.asarray(shift["vector"], np.float64)
    return lambda step: moved.copy() if step >= int(shift["step"]) else base.copy()


def _controller(task: dict, truth: dict):
    """The arm's controller. L1 arms receive only the harness's post-look estimates."""
    from embodied_jepa.policy_diagnostics import RandomController, hold_controller

    arm = task["arm"]
    robot, fk, bounds = _W["robot"], _W["fk"], _W["bounds"]
    model, standardiser = _W["p3"]
    estimates = np.asarray(task.get("estimates", np.zeros(4)), np.float64)
    shift = task.get("shift")
    plate_at = plate_schedule(task["reset"], shift)
    common = (_W["predict"], standardiser, estimates, fk, bounds)
    if arm == "P-stale":
        return rt2.LearnedController(*common)
    if arm == "P-reread":
        return hs.HybridController(
            *common, variant="reread", encoder=_W["encoder"], r_mid=_readout("r_mid")
        )
    if arm in CRITIC_ARMS:
        variant = "shuf" if arm == "H-shuf" else "critic"
        foreign = None
        if arm == "H-shuf":
            foreign = {int(k): np.asarray(v, np.float32) for k, v in task["foreign"].items()}
        return hs.HybridController(
            *common,
            variant=variant,
            policy=_W["policy"],
            encoder=_W["encoder"],
            r_mid=_readout("r_mid"),
            critic=critic(CRITIC_ARMS[arm]),
            standin=_W["standin"],
            foreign_latents=foreign,
        )
    if arm == "H-rand":
        seed = np.random.SeedSequence([wc.SEEDS["h_rand"], int(task["seed"])])
        return hs.HybridController(
            *common,
            variant="rand",
            encoder=_W["encoder"],
            r_mid=_readout("r_mid"),
            rng_seed=int(seed.generate_state(1)[0]),
        )
    if arm == "P-truth":
        log = bool(task.get("log_chunks"))
        return hs.ScheduledTruth(
            *common,
            switch_step=int(shift["step"]) if shift else 0,
            plate_xy=plate_at(10**9),
            standin=_W["standin"] if log else None,
            policy=_W["policy"] if log else None,
        )
    if arm == "H-sim":
        if task.get("h_sim_centre") == "truth":

            def centre(observation, step):  # noqa: ARG001 - privileged by design (K0)
                return plate_at(step)

        else:

            def centre(observation, step):  # noqa: ARG001
                tokens, _ = hs.encode_frame(_W["encoder"], observation.images[fp2.CAMERA][0])
                return _readout("r_mid").predict(tokens)

        return ss.SimSelector(*common, robot=robot, centre=centre, plate_at=plate_at)
    if arm == "B-oracle-shift":
        offset = np.asarray(shift["vector"], np.float64) if shift else np.zeros(2)
        return rt2.ExpertController(rt2.perturbed_truth(truth, [0.0, 0.0], offset), robot)
    if arm == "B-hold":
        return rt2.HarnessController(hold_controller())
    if arm == "B-random":
        lower, upper = bounds
        seed = np.random.SeedSequence([wc.SEEDS["random_controller"], int(task["seed"])])
        return rt2.HarnessController(RandomController(int(seed.generate_state(1)[0]), lower, upper))
    raise wc.GuardError(f"unknown arm {arm!r}")


class Timed:
    """Times ``act`` per command (G7 reads the decision records; this is reported beside)."""

    def __init__(self, inner):
        self.inner = inner
        self.seconds: list[float] = []

    def act(self, observation, step):
        started = time.perf_counter()
        try:
            return self.inner.act(observation, step)
        finally:
            self.seconds.append(time.perf_counter() - started)

    def advance(self, result):
        return self.inner.advance(result)


def run_attempt_task(task: dict) -> dict:
    robot = _W["robot"]
    truth, scorer, counter, _obs, facts = rt2.reset_and_look(robot, task["seed"], task["reset"])
    out = {
        "seed": task["seed"],
        "arm": task["arm"],
        "post_look_frame_sha256": frame_sha(facts["post_look_frame"]),
        "post_look_state": facts["post_look_state"].tolist(),
        "truth_xy": [
            float(v) for v in (*truth["object_position"][:2], *truth["plate_position"][:2])
        ],
    }
    if task.get("expected_frame_sha256") and (
        out["post_look_frame_sha256"] != task["expected_frame_sha256"]
    ):
        counter.remove()
        robot.stop("frame_mismatch")
        raise wc.GuardError(f"G-frame: seed {task['seed']} post-look frame differs")
    shift = task.get("shift")
    hook = ps.PlateShift(robot, int(shift["step"]), shift["vector"]) if shift else None
    try:
        inner = _controller(task, truth)
        controller = Timed(inner)
        record = rt2.run_attempt(
            robot,
            scorer,
            counter,
            controller,
            bounds=_W["bounds"],
            max_steps=int(task.get("max_steps", fp2.MAX_POLICY_STEPS)),
            settle_steps=int(task.get("settle_steps", fp2.SETTLE_STEPS)),
            wall_seconds=float(task.get("wall_seconds", wc.CAPS_SECONDS["per_attempt"])),
        )
    finally:
        if hook is not None:
            hook.remove()
    if shift and record["executed_steps"] >= int(shift["step"]) and not hook.applied:
        raise wc.GuardError("G-shift: the plate shift was not applied")
    detail = record.get("at_rest_detail") or {}
    out.update(
        {
            k: record[k]
            for k in (
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
        }
    )
    out["final_distance_cm"] = (
        None if detail.get("final_distance_m") is None else 100.0 * detail["final_distance_m"]
    )
    out["privileged_ok"] = rt2.privileged_reads_ok(record)
    out["shift_log"] = hook.log if hook is not None else None
    out["act_seconds"] = controller.seconds
    out["commands"] = record["commands"]
    out["decisions"] = getattr(inner, "decisions", [])
    if isinstance(inner, hs.HybridController):
        out["latents"] = {int(k): v for k, v in inner.latents.items()}
    if isinstance(inner, hs.ScheduledTruth) and inner.chunks:
        out["o5_relative_errors"] = hs.relative_chunk_errors(inner.chunks, record["commands"])
    return out


def run_frame_task(task: dict) -> dict:
    robot = _W["robot"]
    truth, _scorer, counter, _obs, facts = rt2.reset_and_look(robot, task["seed"], task["reset"])
    counter.remove()
    robot.stop("frame_only")
    return {
        "seed": task["seed"],
        "frame": facts["post_look_frame"],
        "post_look_frame_sha256": frame_sha(facts["post_look_frame"]),
        "post_look_state": facts["post_look_state"].tolist(),
        "truth_xy": [
            float(v) for v in (*truth["object_position"][:2], *truth["plate_position"][:2])
        ],
    }


# ----- the corpus: one apple-shift-v2 root ---------------------------------------------------
EPISODE_ARRAYS = (*rt2.EPISODE_ARRAYS, *wc.EXTRA_EPISODE_ARRAYS)


def write_episode(folder: Path, episode_id: str, arrays: dict, meta: dict) -> dict:
    import json

    folder = Path(folder)
    npz, meta_path = folder / f"{episode_id}.npz", folder / f"{episode_id}.json"
    for path in (npz, meta_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    missing = set(EPISODE_ARRAYS) - set(arrays)
    if missing:
        raise wc.GuardError(f"an episode lacks {sorted(missing)}")
    np.savez_compressed(npz, **{k: arrays[k] for k in EPISODE_ARRAYS})
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return {"npz_sha256": rt2.sha256_file(npz), "meta_sha256": rt2.sha256_file(meta_path)}


def run_collect_task(task: dict) -> dict:
    """One root: e9 from the post-shift truth plus the root's mis-aim, TASK-048's noise on its
    commands (the label is e9's clean command), the plate shift at K0's step, then the settle.
    Privileged scripted collector by design; not learned."""
    from embodied_jepa.at_rest import AppleAtRestCheck

    robot = _W["robot"]
    lower, upper = _W["bounds"]
    truth, scorer, counter, _obs, facts = rt2.reset_and_look(robot, task["seed"], task["reset"])
    counter.remove()  # privileged collector: truth is read by design
    shift = task.get("shift")
    offset = np.asarray(shift["vector"], np.float64) if shift else np.zeros(2)
    aim = offset + np.asarray(task["misaim_m"], np.float64)
    expert = rt2.make_expert(rt2.perturbed_truth(truth, [0.0, 0.0], aim))
    perturber = _W["perturber"](task["noise_level"], task["noise_seed"])
    hook = ps.PlateShift(robot, int(shift["step"]), shift["vector"]) if shift else None
    check = AppleAtRestCheck(robot)
    max_steps = int(task.get("max_steps", fp2.MAX_POLICY_STEPS))
    rows = {k: [] for k in EPISODE_ARRAYS}
    stages = {"grasp": False, "place": False}

    def observe_row(obs):
        now = robot.sim.task_truth()
        rows["frames"].append(np.asarray(obs.images[fp2.CAMERA][0], np.uint8).copy())
        rows["states"].append(rt2.state_of(obs))
        rows["apple"].append(np.asarray(now["object_position"], np.float32))
        rows["plate"].append(np.asarray(now["plate_position"], np.float32))
        rows["dropped"].append(bool(now["dropped"]))
        rows["hand_contact"].append(bool(now["hand_contact"]))

    def step(base, requested, phase):
        try:
            projection = robot.project_candidates(requested[None, None, None])
        except rt2.ContractError as error:
            if str(error) not in rt2.GUARD_REFUSALS:
                raise
            return "guard_refusal"
        if not bool(projection.feasible[0, 0]):
            return "infeasible_command"
        result = robot.execute(projection.actions[0, 0, 0])
        if result.applied_action is None:
            return result.reason or result.status or "rejected"
        if phase < fp2.SETTLE_PHASE:
            expert.advance(result)
        score = scorer.evaluate()
        check.record()
        rows["base"].append(np.asarray(base, np.float32))
        rows["requested"].append(requested)
        rows["applied"].append(np.asarray(result.applied_action, np.float32))
        rows["phase"].append(phase)
        rows["latched"].append(bool(score["success"]))
        if phase < fp2.SETTLE_PHASE:
            stages["grasp"] |= bool(score.get("grasp", False))
            stages["place"] |= bool(score.get("place", False))
        observe_row(robot.observe())
        return None

    try:
        observe_row(robot.observe())
        termination, steps = "step_limit", 0
        while not expert.done and steps < max_steps:
            phase = expert.phase_index
            base = np.asarray(expert.action(robot), np.float32)
            requested, _kind = perturber(base)
            requested = np.clip(requested, lower, upper).astype(np.float32)
            stop = step(base, requested, phase)
            if stop:
                termination = stop
                break
            steps += 1
        else:
            termination = "policy_complete" if expert.done else "step_limit"
        complete = termination in ("policy_complete", "step_limit")
        if complete:
            settle = rt2.settle_command(_W["bounds"])
            for _ in range(int(task.get("settle_steps", fp2.SETTLE_STEPS))):
                stop = step(settle, settle, fp2.SETTLE_PHASE)
                if stop:
                    termination, complete = f"settle_{stop}", False
                    break
    finally:
        if hook is not None:
            hook.remove()
    robot.stop(termination)
    verdict = check.verdict() if complete and len(rows["phase"]) >= 20 else None
    arrays = {
        "frames": np.asarray(rows["frames"], np.uint8),
        "states": np.asarray(rows["states"], np.float64),
        "base": np.asarray(rows["base"], np.float32).reshape(-1, 14),
        "requested": np.asarray(rows["requested"], np.float32).reshape(-1, 14),
        "applied": np.asarray(rows["applied"], np.float32).reshape(-1, 14),
        "phase": np.asarray(rows["phase"], np.int8),
        "apple": np.asarray(rows["apple"], np.float32),
        "plate": np.asarray(rows["plate"], np.float32),
        "dropped": np.asarray(rows["dropped"], bool),
        "hand_contact": np.asarray(rows["hand_contact"], bool),
        "latched": np.asarray(rows["latched"], bool),
    }
    at_rest = bool(verdict["at_rest"]) if verdict else False
    summary = {
        "episode_id": task["episode_id"],
        "split": task["split"],
        "shifted": bool(shift),
        "misaimed": bool(task["misaimed"]),
        "noise_level": task["noise_level"],
        "termination": termination,
        "complete": bool(complete),
        "policy_steps": int(steps),
        "transitions": int(len(arrays["phase"])),
        "at_rest": at_rest,
        "grasp_before_settle": stages["grasp"],
        "place_before_settle": stages["place"],
        "success": wc.counted_success(at_rest, stages["grasp"], stages["place"]),
        "off_plate": bool(verdict is not None and not verdict["inside_all"]),
        "hand_contact_any": bool(arrays["hand_contact"].any()),
        "shift_applied": bool(hook is not None and hook.applied),
    }
    meta = {
        "seed": task["seed"],
        "session_id": task["session_id"],
        "reset": task["reset"],
        "shift": shift,
        "shift_log": hook.log if hook is not None else None,
        "misaim_m": list(task["misaim_m"]),
        "noise_seed": task["noise_seed"],
        "truth_xy": [
            float(v) for v in (*truth["object_position"][:2], *truth["plate_position"][:2])
        ],
        "post_look_frame_sha256": frame_sha(facts["post_look_frame"]),
        "decision_frame": fp2.DECISION_FRAME,
        "privileged_scripted_collector": True,
        "at_rest_detail": verdict,
        **summary,
    }
    hashes = write_episode(Path(task["folder"]), task["episode_id"], arrays, meta)
    return summary | hashes


# ----- the ranking stage: one cohort-R reset ---------------------------------------------------
def run_rank_task(task: dict) -> dict:
    """P-reread on one R reset with the shift; at each R point, the 25 aims around R-mid's
    incumbent are each run for 16 commands in cloned state (true outcomes, privileged, scoring
    only) and their stand-in chunks are generated (what every ranker scores)."""
    robot = _W["robot"]
    truth, scorer, counter, _obs, facts = rt2.reset_and_look(robot, task["seed"], task["reset"])
    counter.remove()  # the branches read truth for scoring by design; no controller is scored
    shift = task["shift"]
    plate_at = plate_schedule(task["reset"], shift)
    hook = ps.PlateShift(robot, int(shift["step"]), shift["vector"])
    model, standardiser = _W["p3"]
    main = hs.HybridController(
        _W["predict"],
        standardiser,
        np.asarray(task["estimates"], np.float64),
        _W["fk"],
        _W["bounds"],
        variant="reread",
        encoder=_W["encoder"],
        r_mid=_readout("r_mid"),
    )
    brancher = ss.Brancher(robot)
    points = {int(p) for p in task["points"]}
    groups = []
    lower, upper = _W["bounds"]
    try:
        for step in range(max(points) + 1):
            observation = robot.observe()
            if step in points:
                tokens, pooled = hs.encode_frame(_W["encoder"], observation.images[fp2.CAMERA][0])
                incumbent = np.asarray(_readout("r_mid").predict(tokens), np.float64)
                aims = wc.candidate_aims(incumbent)
                estimates = np.concatenate(
                    (np.repeat(main.estimates[None, :2], wc.K, axis=0), aims), axis=1
                )
                _req, applied, feasible = _W["standin"].chunks(
                    _W["policy"],
                    rt2.state_of(observation),
                    step,
                    estimates,
                    last_grasp=main.last_grasp,
                )
                branches = [
                    rt2.LearnedController(_W["predict"], standardiser, e, _W["fk"], _W["bounds"])
                    for e in estimates
                ]
                offsets, reasons = brancher.outcomes(
                    branches, step, _W["bounds"], plate_at(step + wc.CHUNK)
                )
                groups.append(
                    {
                        "step": step,
                        "incumbent": incumbent.tolist(),
                        "latent": pooled,
                        "chunks": applied,
                        "feasible": feasible,
                        "true_offsets_cm": offsets,
                        "true_costs_cm": wc.critic_cost_cm(offsets, step + wc.CHUNK),
                        "stopped": list(reasons),
                    }
                )
            command = np.clip(np.asarray(main.act(observation, step), np.float32), lower, upper)
            try:
                projection = robot.project_candidates(command[None, None, None])
            except rt2.ContractError as error:
                if str(error) not in rt2.GUARD_REFUSALS:
                    raise
                break
            if not bool(projection.feasible[0, 0]):
                break
            result = robot.execute(projection.actions[0, 0, 0])
            if result.applied_action is None:
                break
            main.advance(result)
    finally:
        hook.remove()
        robot.stop("rank_done")
    return {
        "seed": task["seed"],
        "post_look_frame_sha256": frame_sha(facts["post_look_frame"]),
        "groups": groups,
        "shift_applied": hook.applied,
    }


def run_task(task: dict) -> dict:
    kind = task["kind"]
    if kind == "attempt":
        return run_attempt_task(task)
    if kind == "frame":
        return run_frame_task(task)
    if kind == "collect":
        return run_collect_task(task)
    if kind == "rank":
        return run_rank_task(task)
    raise wc.GuardError(f"unknown task kind {kind!r}")
