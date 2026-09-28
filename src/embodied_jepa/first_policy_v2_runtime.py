"""TASK-071 run-time pieces: the v2 robot, e9 as controller and labeller, the at-rest attempt,
and the fresh ``apple-look-v2`` corpus store.

Protocol ``docs/experiments/apple_first_policy_v2.md``. NumPy at import; MuJoCo, torch and the
DINOv2 encoder are imported lazily. v1's pieces (``first_policy_runtime``) are reused unchanged
where the protocol carries them: the bounds and assembly, the standardiser, the private
forward-kinematics model, the A4-look truth dict, the privileged-read counter, the reset and look.

Where privileged information may go is v1's rule, unchanged:

* the **reset truth** is read by the *harness* before any controller exists. It builds the
  DAgger labeller (training only), the C0, B-oracle and A4-look experts' ground truth where
  applicable (L3/L4), the corpus collector (privileged by design) and nothing else;
* an **L1 controller** holds no robot or simulator handle and sees the ``Observation`` and the
  step only;
* ``PrivilegedReadCounter`` counts every ``task_truth`` call and every call made while a
  controller's ``act`` is on the stack. In v2 the harness reads truth twice per executed step --
  once for the latched scorer and once for the at-rest record -- so the expected total is the
  scorer's evaluations plus the at-rest records, and the in-controller count must be 0.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from embodied_jepa import first_policy_runtime as rt
from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa.contracts import ContractError

GUARD_REFUSALS = rt.GUARD_REFUSALS
EXHAUSTED_MESSAGE = rt.EXHAUSTED_MESSAGE
configured_bounds = rt.configured_bounds
assemble = rt.assemble
free_of = rt.free_of
state_of = rt.state_of
Standardiser = rt.Standardiser
PalmFK = rt.PalmFK
a4_truth = rt.a4_truth
PrivilegedReadCounter = rt.PrivilegedReadCounter
HarnessController = rt.HarnessController
ReplayController = rt.ReplayController
perturbed_truth = rt.perturbed_truth
reset_and_look = rt.reset_and_look


# ----- the v2 robot -------------------------------------------------------------------------------
def make_robot():
    """v1's look-ready 112 px robot, switched to ``apple_to_plate_v2`` (apple condim 6)."""
    from embodied_jepa import apple_to_plate_v2 as v2

    robot, record = v2.make_v2_robot()
    if (
        record["scene_version"] != fp2.SCENE_VERSION
        or record["apple_condim"] != fp2.APPLE_CONTACT["condim"]
        or not np.allclose(record["apple_friction"], fp2.APPLE_CONTACT["friction"])
    ):
        raise fp2.GuardError(f"G-scene: the robot is not apple-to-plate-v2 ({record})")
    return robot


# ----- inputs -----------------------------------------------------------------------------------
def input_vector(estimates, step: int, state, palm9) -> np.ndarray:
    """The 132-d raw input, in ``POLICY_INPUTS`` order, with e9's clock (scale 725)."""
    parts = [
        np.asarray(estimates, np.float64).reshape(-1),
        fp2.clock_features(step).astype(np.float64),
        np.asarray(state, np.float64).reshape(-1),
        np.asarray(palm9, np.float64).reshape(-1),
    ]
    sizes = tuple(len(p) for p in parts)
    if sizes != tuple(fp2.POLICY_INPUTS.values()):
        raise ContractError(f"input parts {sizes} are not {tuple(fp2.POLICY_INPUTS.values())}")
    x = np.concatenate(parts)
    if not np.isfinite(x).all():
        raise ContractError("G-finite: a policy input is not finite")
    return x.astype(np.float32)


class LearnedController(rt.LearnedController):
    """An L1 (or F: L2) policy. It holds no robot or simulator handle (G-privileged (3))."""

    def act(self, observation, step):
        state = state_of(observation)
        raw = input_vector(self.estimates, step, state, self.fk.pose9(state))
        self.last_input = raw
        free = np.asarray(self.predict(self.standardiser(raw[None])[0], step), np.float32)
        return assemble(free, self.lower, self.upper)


# ----- e9 ---------------------------------------------------------------------------------------
def make_expert(truth):
    """The frozen e9 (``RestingPlaceExpert(release_pitch_rad=0.45, release_dx=0.015)``), with
    its schedule checked against the preregistered one. Privileged and scripted: never learned."""
    from embodied_jepa.resting_expert import RestingPlaceExpert

    expert = RestingPlaceExpert(truth, **fp2.EXPERT)
    names = tuple(p.name for p in expert.phases)
    budgets = tuple(int(p.commands) for p in expert.phases)
    if names != fp2.EXPERT_PHASES or budgets != fp2.EXPERT_BUDGETS:
        raise ContractError(f"e9's schedule {names} / {budgets} is not the preregistered one")
    return expert


class ExpertController:
    """e9 in command (C0, B-oracle, A4-look): L3/L4 only. Exhaustion ends the policy steps.

    ``RestingPlaceExpert.action`` indexes its phase before its parent's exhaustion check, so
    exhaustion is raised here, with v1's message, before the expert is asked."""

    def __init__(self, truth, robot):
        self.expert = make_expert(truth)
        self.robot = robot

    def act(self, observation, step):  # noqa: ARG002 - reads the robot pose, as the expert does
        if self.expert.done:
            raise ContractError(EXHAUSTED_MESSAGE)
        return np.asarray(self.expert.action(self.robot), np.float32)

    def advance(self, result):
        if not self.expert.done:
            self.expert.advance(result)


class ShadowExpert:
    """The DAgger labeller: e9 advanced on the EXECUTED result, on the learner's clock
    (``policy_diagnostics.ShadowExpert``'s semantics). ``command`` returns the signed free
    7-vector, or ``None`` once e9's budget is exhausted. Privileged; training time only."""

    def __init__(self, truth):
        self.policy = make_expert(truth)
        self.budget = int(self.policy.max_steps)
        if self.budget != fp2.EXPERT_POLICY_STEPS:
            raise ContractError("the shadow expert's budget is not e9's")
        self.exhausted_at_step: int | None = None
        self.calls = 0

    def command(self, robot, step: int) -> np.ndarray | None:
        if self.exhausted_at_step is not None:
            return None
        if self.policy.done:
            self.exhausted_at_step = int(step)
            return None
        self.calls += 1
        action = np.asarray(self.policy.action(robot), np.float32)
        return action[list(fp2.FREE_INDICES)].copy()

    def advance(self, result) -> None:
        if self.exhausted_at_step is None and not self.policy.done:
            self.policy.advance(result)


def shadow_expert(truth) -> ShadowExpert:
    return ShadowExpert(truth)


def settle_command(bounds) -> np.ndarray:
    """The task's settle: arm still, both hands open (as ``resting_expert.run_attempt``)."""
    lower, upper = bounds
    command = np.zeros(14, np.float32)
    command[12:] = -1.0
    return np.clip(command, lower, upper).astype(np.float32)


# ----- one attempt --------------------------------------------------------------------------------
def _stage(score: dict) -> str:
    return next(
        (s for s in ("release", "place", "transport", "grasp", "reach") if score.get(s)), "none"
    )


def run_attempt(
    robot,
    scorer,
    counter,
    controller,
    *,
    bounds,
    max_steps: int = fp2.MAX_POLICY_STEPS,
    settle_steps: int = fp2.SETTLE_STEPS,
    wall_seconds: float = fp2.ATTEMPT_WALL_SECONDS,
    labeller=None,
    record_inputs: bool = False,
) -> dict:
    """The v2 attempt after the look: up to ``max_steps`` controller commands, then the task's
    settle, then the ``apple_at_rest_v0`` verdict over the final window.

    Unlike v1's loop, a latched success does not end the attempt: it is recorded and the attempt
    runs on. A guard refusal, an infeasible command or a rejected command ends the attempt as not
    at rest (a failure, not a void). Exceeding ``wall_seconds`` is G-cap (a void)."""
    from embodied_jepa.at_rest import AppleAtRestCheck

    lower, upper = bounds
    check = AppleAtRestCheck(robot)
    reason = "step_limit"
    score: dict = {}
    executed = evaluations = records = 0
    latched_step = None
    grasp_seen = False
    states, steps, labels, commands, stages = [], [], [], [], []
    started = time.monotonic()

    def execute(command):
        """The executed result, or the reason the command did not execute."""
        try:
            projection = robot.project_candidates(command[None, None, None])
        except ContractError as error:
            if str(error) not in GUARD_REFUSALS:
                raise
            return "guard_refusal"
        if not bool(projection.feasible[0, 0]):
            return "infeasible_command"
        result = robot.execute(projection.actions[0, 0, 0])
        if result.applied_action is None:
            return result.reason or result.status or "rejected"
        return result

    try:
        for step in range(max_steps):
            if time.monotonic() - started > wall_seconds:
                reason = "attempt_wall_cap"
                raise fp2.GuardError(f"G-cap: an attempt exceeded its {wall_seconds} s cap")
            observation = robot.observe()
            try:
                command = counter.act(controller, observation, step)
            except ContractError as error:
                if str(error) != EXHAUSTED_MESSAGE:
                    raise
                reason = "policy_complete"
                break
            command = np.clip(np.asarray(command, np.float32), lower, upper).astype(np.float32)
            if labeller is not None and step < fp2.EXPERT_POLICY_STEPS:
                label = labeller.command(robot, step)
                if label is not None:
                    states.append(state_of(observation).copy())
                    steps.append(step)
                    labels.append(
                        np.clip(label, lower[list(fp2.FREE_INDICES)], upper[list(fp2.FREE_INDICES)])
                    )
            elif record_inputs:
                states.append(state_of(observation).copy())
                steps.append(step)
            commands.append(free_of(command))
            result = execute(command)
            if isinstance(result, str):
                reason = result
                break
            controller.advance(result)
            if labeller is not None:
                labeller.advance(result)
            executed += 1
            score = scorer.evaluate()
            evaluations += 1
            check.record()
            records += 1
            grasp_seen = grasp_seen or bool(score.get("grasp", False))
            if score.get("success", False) and latched_step is None:
                latched_step = step
            stages.append(_stage(score))
        complete = reason in ("step_limit", "policy_complete")
        settled = 0
        if complete:
            settle = settle_command(bounds)
            for _ in range(settle_steps):
                if time.monotonic() - started > wall_seconds:
                    reason = "attempt_wall_cap"
                    raise fp2.GuardError(f"G-cap: an attempt exceeded its {wall_seconds} s cap")
                robot.observe()
                result = execute(settle)
                if isinstance(result, str):
                    reason = f"settle_{result}"
                    complete = False
                    break
                score = scorer.evaluate()
                evaluations += 1
                check.record()
                records += 1
                grasp_seen = grasp_seen or bool(score.get("grasp", False))
                if score.get("success", False) and latched_step is None:
                    latched_step = executed + settled
                settled += 1
    finally:
        robot.stop(reason)
        counter.remove()
    window = check.thresholds.window_steps
    verdict = check.verdict() if complete and records >= window else None
    return {
        "termination_reason": reason,
        "complete": bool(complete),
        "executed_steps": executed,
        "settle_steps": settled,
        "at_rest": bool(verdict["at_rest"]) if verdict else False,
        "at_rest_detail": verdict,
        "latched_success": latched_step is not None,
        "first_latched_step": latched_step,
        "grasp": bool(grasp_seen),
        "success": bool(verdict["at_rest"]) if verdict else False,  # the gated metric
        "final_score": {
            k: (bool(v) if isinstance(v, bool | np.bool_) else v) for k, v in score.items()
        },
        "stages": stages,
        "commands": np.asarray(commands, np.float32),
        "task_truth_total": counter.total,
        "task_truth_in_controller": counter.in_controller,
        "scorer_evaluations": evaluations,
        "at_rest_records": records,
        "states": np.asarray(states, np.float64),
        "steps": np.asarray(steps, np.int64),
        "labels": np.asarray(labels, np.float32),
    }


def privileged_reads_ok(record: dict) -> bool:
    """G-privileged (1)+(2): no read inside act; the total equals the harness's two reads per
    executed step (the latched scorer and the at-rest record)."""
    return record["task_truth_in_controller"] == 0 and record["task_truth_total"] == (
        record["scorer_evaluations"] + record["at_rest_records"]
    )


# ----- the apple-look-v2 corpus store ----------------------------------------------------------
EPISODE_ARRAYS = (
    "frames",  # uint8 [T+1, 112, 112, 3]; frame 0 is the post-look frame
    "states",  # float64 [T+1, 86]
    "base",  # float32 [T, 14]: e9's command before noise (the BC label source)
    "requested",  # float32 [T, 14]: after noise, clipped to the collection bounds
    "applied",  # float32 [T, 14]: the executed (normalised) command
    "phase",  # int8 [T]: e9's phase index; SETTLE_PHASE for the task's settle
    "apple",  # float32 [T+1, 3]: privileged, label only
    "dropped",  # bool [T+1]: privileged, label only
    "hand_contact",  # bool [T+1]: privileged, label only
    "latched",  # bool [T]: the latched v1 scorer's per-step success
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_episode(folder: Path, episode_id: str, arrays: dict, meta: dict) -> dict:
    """Write one episode's arrays and metadata; refuse to overwrite. Returns their sha256."""
    folder = Path(folder)
    npz, meta_path = folder / f"{episode_id}.npz", folder / f"{episode_id}.json"
    for path in (npz, meta_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    missing = set(EPISODE_ARRAYS) - set(arrays)
    if missing:
        raise ContractError(f"an episode lacks {sorted(missing)}")
    np.savez_compressed(npz, **{k: arrays[k] for k in EPISODE_ARRAYS})
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return {"npz_sha256": sha256_file(npz), "meta_sha256": sha256_file(meta_path)}


def seal(corpus: Path, plan: list[dict], episodes: dict, provenance: dict) -> str:
    """Write ``manifest.json`` (plan, splits, per-episode hashes, provenance); return its sha256."""
    corpus = Path(corpus)
    path = corpus / "manifest.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    splits = {k: [] for k in ("train", "val", "test")}
    for root in plan:
        splits[root["split"]].append(root["episode_id"])
    manifest = {
        "corpus": fp2.CORPUS,
        "protocol": fp2.PROTOCOL,
        "task": fp2.TASK,
        "privileged_scripted_collector": True,
        "learned_control": False,
        "scene_version": fp2.SCENE_VERSION,
        "expert": {"class": fp2.EXPERT_CLASS, "kwargs": fp2.EXPERT},
        "plan": plan,
        "splits": splits,
        "episodes": episodes,
        "provenance": provenance,
    }
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return sha256_file(path)


class CorpusReader:
    """Reads only the allowed splits of a sealed ``apple-look-v2`` corpus, verifying hashes.

    Any other episode id is refused before a file is opened (Q-split); ``decoded`` records
    every episode read, and ``test_split_decoded`` is checked against it."""

    def __init__(self, corpus: Path, manifest_sha256: str, splits=fp2.READ_SPLITS):
        self.corpus = Path(corpus)
        path = self.corpus / "manifest.json"
        if sha256_file(path) != manifest_sha256:
            raise fp2.GuardError("G-data: the corpus manifest differs from the sealed hash")
        self.manifest = json.loads(path.read_text())
        self.allowed = {e for split in splits for e in self.manifest["splits"][split]}
        self.decoded: set[str] = set()

    def episode(self, episode_id: str, keys=EPISODE_ARRAYS) -> tuple[dict, dict]:
        """``keys`` of one readable episode, plus ``frame0`` (the post-look frame), and its
        metadata. Hashes are verified before anything is decoded."""
        if episode_id not in self.allowed:
            raise fp2.GuardError(f"Q-split: {episode_id} is not in a readable split")
        record = self.manifest["episodes"][episode_id]
        npz = self.corpus / "episodes" / f"{episode_id}.npz"
        meta = self.corpus / "episodes" / f"{episode_id}.json"
        if sha256_file(npz) != record["npz_sha256"] or sha256_file(meta) != record["meta_sha256"]:
            raise fp2.GuardError(f"G-data: {episode_id} differs from its sealed hash")
        self.decoded.add(episode_id)
        with np.load(npz) as data:
            arrays = {k: data[k] for k in keys}
            arrays["frame0"] = (arrays["frames"] if "frames" in arrays else data["frames"])[
                fp2.DECISION_FRAME
            ].copy()
        return arrays, json.loads(meta.read_text())

    @property
    def test_split_decoded(self) -> bool:
        return bool(self.decoded & set(self.manifest["splits"]["test"]))


def bc_rows(arrays: dict, bounds) -> dict:
    """BC-0 rows of one corpus root (protocol §3): e9's policy steps, v1's mask, clipped labels.

    A step is dropped when the apple's 3-D drift from frame 0 exceeds 1 cm while e9's phase is
    before ``close``, or when the apple is dropped (``cloning.sample_mask``'s rule). Settle
    steps are the task's, not the policy's, and are never rows."""
    lower, upper = bounds
    free = list(fp2.FREE_INDICES)
    phase = np.asarray(arrays["phase"], int)
    base = np.asarray(arrays["base"], np.float64)
    apple = np.asarray(arrays["apple"], np.float64)
    dropped = np.asarray(arrays["dropped"], bool)
    keep = []
    accounting = {"policy_steps": 0, "dropped": 0, "post_displacement_pre_grasp": 0, "kept": 0}
    for t in range(len(phase)):
        if phase[t] >= fp2.SETTLE_PHASE or t >= fp2.EXPERT_POLICY_STEPS:
            continue
        accounting["policy_steps"] += 1
        drift = float(np.linalg.norm(apple[t] - apple[0]))
        if dropped[t]:
            accounting["dropped"] += 1
            continue
        if drift > fp2.DISPLACEMENT_LIMIT_M and phase[t] < fp2.PHASE_CLOSE:
            accounting["post_displacement_pre_grasp"] += 1
            continue
        keep.append(t)
    keep = np.asarray(keep, int)
    accounting["kept"] = len(keep)
    return {
        "states": np.asarray(arrays["states"], np.float64)[keep],
        "steps": keep.astype(np.int64),
        "labels": np.clip(base[keep][:, free], lower[free], upper[free]).astype(np.float32),
        "accounting": accounting,
    }


def replay_actions(arrays: dict) -> np.ndarray:
    """B-replay's recording: the executed commands of the root's policy steps (no settle)."""
    phase = np.asarray(arrays["phase"], int)
    policy = np.flatnonzero(phase < fp2.SETTLE_PHASE)
    applied = np.asarray(arrays["applied"], np.float32)[policy]
    return applied[: fp2.MAX_POLICY_STEPS]
