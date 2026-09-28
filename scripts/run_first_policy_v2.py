"""TASK-071 runner (``apple_first_policy_v2``): v1's first learned policy on apple-to-plate-v2.

Protocol ``docs/experiments/apple_first_policy_v2.md``; design ``first_policy_v2.py``; run-time
pieces ``first_policy_v2_runtime.py`` and ``first_policy_v2_model.py``, plus v1's
``first_policy_perception.py`` unchanged. **Learned Apple->Plate is still 0 successes.**

Stages (a guard failure or an exception is V, and ``report.json`` is still written):

1. preflight: G-frozen, G-hash, the clean tree, G-seeds, G-device, G-weights, G-scene;
2. corpus: the fresh ``apple-look-v2`` corpus, e9 under v2 on 200 roots (noise levels 0-3 as
   TASK-048/064), sealed with per-episode sha256; read back through ``CorpusReader`` (train and
   val only; the test split is never decoded);
3. perception frames: reset + look on the perception seeds (worker path); the corpus's train and
   val post-look frames; DINOv2 tokens at batch size 1 (pretrained and floor);
4. readouts: the frozen P and R readouts, cross-fitted BC-0 estimates, S0-P's held-out errors;
5. C0: nine conditions x 32 resets of e9, at rest; the bars (``c0_bars``, with the TASK-070
   plate ceiling) or CAL-ESCALATE;
6. S0-P (S0-APPLE-FAIL / S0-PLATE-FAIL) and the A4-look threshold T;
7. BC-0; P-0, C-0, R-0 on MPS; S0-D1; three DAgger iterations for P, C and R;
8. M1 on D2: every M1 arm once per reset; ``decide_m1``; D-oracle-perc; F only on M1-MOTOR.

Every attempt after the look runs its commands (at most 740), then the task's 60-step settle,
and is scored by ``apple_at_rest_v0``; the latched v1 scorer is recorded beside it.

    uv run --no-sync python scripts/run_first_policy_v2.py run \\
        --output outputs/task071-first-policy-v2/run-1 \\
        --checkpoints checkpoints/task071-first-policy-v2/run-1 \\
        --corpus data/apple-look-v2/run-1

``smoke`` runs every stage on CPU on a tiny subset, with the smoke seeds (52100-52199) in every
simulated role, 2 workers, 40 policy steps (``--smoke-max-steps 740`` for full-length attempts),
20 updates and noise targets. Nothing in a smoke is read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import first_policy_v2 as fp2  # noqa: E402
from embodied_jepa import first_policy_v2_runtime as rt2  # noqa: E402

MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-first-policy-v2.json"
SMOKE_SEEDS = tuple(range(fp2.SMOKE_SEEDS[0], fp2.SMOKE_SEEDS[1] + 1))
SMOKE = {
    "workers": 2,
    "updates": 20,
    "select_every": 10,
    "per_role": 2,
    "corpus_roots": 6,
    "max_steps": 40,
}
CORPUS_KEYS = ("states", "base", "applied", "phase", "apple", "dropped")  # plus frame 0
SMOKE_OFFSETS = {
    "corpus": 0,
    "perception_train": 10,
    "perception_heldout": 20,
    "calibration_C0": 30,
    "dagger_1": 40,
    "dagger_2": 50,
    "dagger_3": 60,
    "D": 70,
}


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R65 = _load("_task065_runner", "scripts/train_apple_latent_dynamics.py")
log = R65.log


def load_wide_reset():
    return _load("_evaluate_apple_reset", "scripts/evaluate_apple.py").wide_reset


def load_perturber():
    """TASK-048's seeded OU + burst perturbation, loaded unchanged (its file is pinned)."""
    return _load("_collect_apple_wide", "scripts/collect_apple_wide.py").Perturber


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def frame_sha(frame) -> str:
    return sha256_bytes(np.ascontiguousarray(frame, np.uint8).tobytes())


# ----- guards -----
def check_simulated_seeds(role: str, seeds, smoke: bool) -> None:
    """G-seeds: a whitelist per role; cohort C is refused before anything is simulated."""
    seeds = tuple(int(s) for s in seeds)
    if set(seeds) & set(fp2.COHORT_C):
        raise fp2.GuardError("G-seeds: cohort C is not opened by this run")
    if len(set(seeds)) != len(seeds):
        raise fp2.GuardError(f"G-seeds: duplicate seeds in {role}")
    if smoke:
        allowed = set(SMOKE_SEEDS)
    elif role == "D":
        allowed = set(fp2.COHORT_D2)
    else:
        allowed = set(fp2.seeds_of(role))
    outside = sorted(set(seeds) - allowed)
    if outside:
        raise fp2.GuardError(f"G-seeds: {outside[:4]} are not {role} seeds")


def check_look_states(records: list[dict], reference=None, tolerance: float = 1e-6):
    """G-look (b): every attempt's post-look joint state equals the run's first attempt's."""
    states = [np.asarray(r["post_look_state"]) for r in records if "post_look_state" in r]
    if not states:
        return reference
    reference = states[0] if reference is None else np.asarray(reference)
    if max(float(np.abs(s - reference).max()) for s in states) > tolerance:
        raise fp2.GuardError("G-look: the post-look joint state differs from the first attempt's")
    return reference


def check_privileged(records: list[dict], arm: str) -> None:
    """G-privileged (1)+(2) for an L1 or L2 arm's evaluation attempts."""
    for record in records:
        if not rt2.privileged_reads_ok(record):
            raise fp2.GuardError(f"G-privileged: {arm} seed {record['seed']} read task truth")


# ----- the worker -----
_W: dict = {}


def worker_init() -> None:
    import torch

    torch.set_num_threads(fp2.WORKER_TORCH_THREADS)
    _W["robot"] = rt2.make_robot()
    _W["fk"] = rt2.PalmFK()
    _W["bounds"] = rt2.configured_bounds()
    _W["models"] = {}
    _W["perturber"] = load_perturber()


def _learned(spec: dict):
    from embodied_jepa import first_policy_v2_model as fm2

    key = spec["checkpoint"]
    if key not in _W["models"]:
        import torch

        saved = torch.load(key, map_location="cpu", weights_only=True)
        model = fm2.load(saved["state"], heads=int(saved["heads"]))
        _W["models"][key] = (model, rt2.Standardiser(saved["mean"].numpy(), saved["std"].numpy()))
    model, standardiser = _W["models"][key]
    return rt2.LearnedController(
        fm2.predictor(model), standardiser, spec["estimates"], _W["fk"], _W["bounds"]
    )


def collect_root(task: dict, truth, scorer, out: dict) -> dict:
    """One ``apple-look-v2`` root: e9 from the reset truth through TASK-048's perturbation (the
    label is e9's clean command), then the task's settle. Privileged by design; not learned."""
    from embodied_jepa.at_rest import AppleAtRestCheck

    robot = _W["robot"]
    lower, upper = _W["bounds"]
    expert = rt2.make_expert(truth)
    perturber = _W["perturber"](task["noise_level"], task["noise_seed"])
    check = AppleAtRestCheck(robot)
    max_steps = int(task.get("max_steps", fp2.MAX_POLICY_STEPS))
    rows = {k: [] for k in rt2.EPISODE_ARRAYS}

    def observe_row(obs):
        now = robot.sim.task_truth()
        rows["frames"].append(np.asarray(obs.images[fp2.CAMERA][0], np.uint8).copy())
        rows["states"].append(rt2.state_of(obs))
        rows["apple"].append(np.asarray(now["object_position"], np.float32))
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
        observe_row(robot.observe())
        return None

    observe_row(robot.observe())  # a fresh observation before the first command, as run_attempt
    termination = "step_limit"
    steps = 0
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
        for _ in range(fp2.SETTLE_STEPS):
            stop = step(settle, settle, fp2.SETTLE_PHASE)
            if stop:
                termination, complete = f"settle_{stop}", False
                break
    robot.stop(termination)
    verdict = check.verdict() if complete else None
    arrays = {
        "frames": np.asarray(rows["frames"], np.uint8),
        "states": np.asarray(rows["states"], np.float64),
        "base": np.asarray(rows["base"], np.float32).reshape(-1, 14),
        "requested": np.asarray(rows["requested"], np.float32).reshape(-1, 14),
        "applied": np.asarray(rows["applied"], np.float32).reshape(-1, 14),
        "phase": np.asarray(rows["phase"], np.int8),
        "apple": np.asarray(rows["apple"], np.float32),
        "dropped": np.asarray(rows["dropped"], bool),
        "hand_contact": np.asarray(rows["hand_contact"], bool),
        "latched": np.asarray(rows["latched"], bool),
    }
    summary = {
        "episode_id": task["episode_id"],
        "split": task["split"],
        "noise_level": task["noise_level"],
        "termination": termination,
        "complete": bool(complete),
        "policy_steps": int(steps),
        "transitions": int(len(arrays["phase"])),
        "at_rest": bool(verdict["at_rest"]) if verdict else False,
        "at_rest_detail": verdict,
        "latched_success": bool(arrays["latched"].any()),
    }
    meta = {
        "seed": task["seed"],
        "episode_id": task["episode_id"],
        "session_id": task["session_id"],
        "split": task["split"],
        "noise_level": task["noise_level"],
        "noise_seed": task["noise_seed"],
        "reset": task["reset"],
        "truth_xy": out["truth_xy"],
        "post_look_frame_sha256": out["post_look_frame_sha256"],
        "decision_frame": fp2.DECISION_FRAME,
        "privileged_scripted_collector": True,
        **{k: v for k, v in summary.items() if k not in ("episode_id", "split", "noise_level")},
    }
    hashes = rt2.write_episode(Path(task["folder"]), task["episode_id"], arrays, meta)
    return summary | hashes


def run_task(task: dict) -> dict:
    """One attempt in a worker. ``task["kind"]`` names the controller."""
    from embodied_jepa.policy_diagnostics import RandomController, hold_controller

    robot = _W["robot"]
    truth, scorer, counter, observation, facts = rt2.reset_and_look(
        robot, task["seed"], task["reset"]
    )
    out = {
        "seed": task["seed"],
        "kind": task["kind"],
        "post_look_frame_sha256": frame_sha(facts["post_look_frame"]),
        "post_look_state": facts["post_look_state"].tolist(),
        "truth_xy": [
            float(v) for v in (*truth["object_position"][:2], *truth["plate_position"][:2])
        ],
    }
    if task["kind"] == "frame":
        counter.remove()
        robot.stop("frame_only")
        out["frame"] = facts["post_look_frame"]
        return out
    if task["kind"] == "collect":
        counter.remove()  # the collector reads truth by design (privileged; labels only)
        return out | collect_root(task, truth, scorer, out)
    expected = task.get("expected_frame_sha256")
    if expected is not None and out["post_look_frame_sha256"] != expected:
        counter.remove()
        robot.stop("frame_mismatch")
        raise fp2.GuardError(f"S0-D1/G-frame: seed {task['seed']} post-look frame differs")
    kind = task["kind"]
    if kind == "learned":
        controller = _learned(task)
    elif kind == "a4":
        controller = rt2.ExpertController(rt2.a4_truth(task["estimates"], _W["fk"]), robot)
    elif kind == "oracle":
        controller = rt2.ExpertController(truth, robot)
    elif kind == "c0":
        controller = rt2.ExpertController(
            rt2.perturbed_truth(truth, task["apple_offset"], task["plate_offset"]), robot
        )
    elif kind == "hold":
        controller = rt2.HarnessController(hold_controller())
    elif kind == "random":
        lower, upper = _W["bounds"]
        controller = rt2.HarnessController(
            RandomController(fp2.RANDOM_CONTROLLER_SEED, lower, upper)
        )
    elif kind == "replay":
        controller = rt2.ReplayController(task["actions"])
    else:
        counter.remove()
        raise fp2.GuardError(f"unknown controller kind {kind!r}")
    labeller = rt2.shadow_expert(truth) if task.get("label") else None
    record = rt2.run_attempt(
        robot,
        scorer,
        counter,
        controller,
        bounds=_W["bounds"],
        max_steps=int(task.get("max_steps", fp2.MAX_POLICY_STEPS)),
        settle_steps=int(task.get("settle_steps", fp2.SETTLE_STEPS)),
        labeller=labeller,
    )
    if kind == "learned" and task.get("first_input_check"):
        out["first_input"] = (
            None if controller.last_input is None else controller.last_input.tolist()
        )
    if record["states"].size:
        out["palm9"] = np.stack([_W["fk"].pose9(s) for s in record["states"]]).tolist()
    out.update({k: v for k, v in record.items() if k != "commands"})
    out["commands"] = record["commands"].tolist()
    for key in ("states", "steps", "labels"):
        out[key] = np.asarray(record[key]).tolist()
    return out


class Pool:
    look_reference = None

    def __init__(self, workers: int):
        self.pool = mp.get_context("spawn").Pool(workers, initializer=worker_init)

    def map(self, tasks: list[dict], cap: float, what: str) -> list[dict]:
        started = time.monotonic()
        result = self.pool.map_async(run_task, tasks, chunksize=1)
        remaining = cap - (time.monotonic() - started)
        try:
            out = result.get(timeout=max(remaining, 1.0))
        except mp.TimeoutError as error:
            raise fp2.GuardError(f"G-cap: {what} exceeded its {cap} s cap") from error
        self.look_reference = check_look_states(out, self.look_reference)  # every attempt
        return out

    def close(self):
        self.pool.terminate()
        self.pool.join()


# ----- data -----
def inputs_of(estimates_per_row, steps, states, palm9) -> np.ndarray:
    return np.stack(
        [
            rt2.input_vector(e, int(s), x, p)
            for e, s, x, p in zip(estimates_per_row, steps, states, palm9, strict=True)
        ]
    )


def save_checkpoint(path: Path, trained: dict, standardiser, heads: int, meta: dict) -> str:
    import torch

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    torch.save(
        {
            "state": trained["state"],
            "heads": heads,
            "mean": torch.as_tensor(standardiser.mean),
            "std": torch.as_tensor(standardiser.std),
            "metadata": meta,
        },
        path,
    )
    return sha256_bytes(path.read_bytes())


def count(records) -> dict:
    """``success`` is at rest (the gated metric); ``latched`` and ``grasp`` are the v1 scorer's."""
    return {
        "grasp": sum(bool(r["grasp"]) for r in records),
        "success": sum(bool(r["at_rest"]) for r in records),
        "latched": sum(bool(r["latched_success"]) for r in records),
    }


def attempt_summary(records) -> list[dict]:
    keep = (
        "seed",
        "termination_reason",
        "complete",
        "executed_steps",
        "settle_steps",
        "grasp",
        "at_rest",
        "at_rest_detail",
        "latched_success",
        "first_latched_step",
        "final_score",
        "task_truth_total",
        "task_truth_in_controller",
        "scorer_evaluations",
        "at_rest_records",
    )
    out = []
    for r in records:
        item = {k: r.get(k) for k in keep}
        stages = r.get("stages", [])
        item["stage_occupancy"] = {s: stages.count(s) for s in sorted(set(stages))}
        commands = np.asarray(r.get("commands", []), np.float64).reshape(-1, 7)
        if len(commands):
            item["command_signed_mean"] = commands.mean(axis=0).tolist()
            item["command_signed_q10_q50_q90"] = np.quantile(
                commands, [0.1, 0.5, 0.9], axis=0
            ).tolist()
        out.append(item)
    return out


def corpus_summary(records: list[dict]) -> dict:
    by_level = {}
    for level in fp2.NOISE_LEVELS:
        rows = [r for r in records if r["noise_level"] == level]
        by_level[str(level)] = {
            "roots": len(rows),
            "at_rest": sum(r["at_rest"] for r in rows),
            "latched": sum(r["latched_success"] for r in rows),
            "complete": sum(r["complete"] for r in rows),
        }
    return {
        "roots": len(records),
        "at_rest": sum(r["at_rest"] for r in records),
        "latched": sum(r["latched_success"] for r in records),
        "terminations": {
            t: sum(r["termination"] == t for r in records)
            for t in sorted({r["termination"] for r in records})
        },
        "by_noise_level": by_level,
    }


def end_checks(report, manifest, smoke) -> None:
    """G-hash after the last stage, on every row (early stops included)."""
    report["pinned_hashes_at_end"] = R65.check_pins(manifest["hashes"])
    report["revision_at_end"] = R65.revision()
    if not smoke and R65.tracked_tree_dirty():
        raise fp2.GuardError("G-hash: the tracked tree changed during the run")


# ----- the run -----
def run(
    output: Path, checkpoints: Path, corpus: Path, smoke: bool, smoke_max_steps: int | None = None
) -> dict:
    output, checkpoints, corpus = Path(output), Path(checkpoints), Path(corpus)
    for path in (output, checkpoints, corpus):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    output.mkdir(parents=True)
    checkpoints.mkdir(parents=True)
    (corpus / "episodes").mkdir(parents=True)
    report = {
        "protocol": fp2.PROTOCOL,
        "task": fp2.TASK,
        "smoke": smoke,
        "smoke_max_steps": smoke_max_steps,
        "outcome": None,
        "stages": {},
        "paths": {"output": str(output), "checkpoints": str(checkpoints), "corpus": str(corpus)},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    clock = R65.Clock(fp2.CAPS_SECONDS["global"])
    pool, reader = None, None
    try:
        reader, pool = _run(report, output, checkpoints, corpus, clock, smoke, smoke_max_steps)
    except Exception as error:  # noqa: BLE001 - every failure is V, and the report is written
        report["outcome"] = "V"
        report["void_reason"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        log(f"VOID: {report['void_reason']}")
    finally:
        pool = report.pop("_pool", pool)
        if pool is not None:
            pool.close()
        reader = report.pop("_reader", reader)
        if reader is not None:
            report["decoded_episodes"] = len(reader.decoded)
            report["test_split_decoded"] = reader.test_split_decoded
        report["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["total_seconds"] = clock.elapsed()
        R65.write_report(output / "report.json", report)
        log(f"report written: outcome {report.get('outcome')}")
    return report


def _run(report, output, checkpoints, corpus, clock, smoke, smoke_max_steps=None):  # noqa: C901
    import torch

    from embodied_jepa import first_policy_perception as fpp
    from embodied_jepa import first_policy_v2_model as fm2
    from embodied_jepa import pretrained_encoder as pe

    # ----- 1. preflight -----
    manifest = json.loads(MANIFEST.read_text())
    if manifest["frozen"] != fp2.frozen_block():
        raise fp2.GuardError("G-frozen: the manifest's frozen block differs from the module")
    report["pinned_hashes_at_preflight"] = R65.check_pins(manifest["hashes"])
    dirty = R65.tracked_tree_dirty()
    report["revision"], report["tracked_tree_dirty"] = R65.revision(), bool(dirty)
    if not smoke:
        R65.check_clean(dirty)
    fp2.check_seed_ranges()
    mps = bool(torch.backends.mps.is_available())
    device = "cpu" if smoke else fp2.TRAIN_DEVICE
    if not smoke and not mps:
        raise fp2.GuardError("G-device: MPS is not available")
    torch.set_num_threads(6)
    import mujoco

    if mujoco.__version__ != manifest["mujoco_version"]:
        raise fp2.GuardError(f"G-hash: MuJoCo {mujoco.__version__} is not the pinned version")
    report["environment"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "mps_available": mps,
        "train_device": device,
    }
    pretrained = pe.load_pretrained()
    floor = pe.random_init()
    digests = {"pretrained": pe.weights_digest(pretrained), "floor": pe.weights_digest(floor)}
    report["encoder_digests"] = digests
    if digests != manifest["encoder_digests"]:
        raise fp2.GuardError("G-weights: an encoder digest differs from its pin")
    bounds = rt2.configured_bounds()
    wide_reset = load_wide_reset()
    workers = SMOKE["workers"] if smoke else fp2.SIM_WORKERS
    pool = Pool(workers)
    report["_pool"] = pool
    report["stages"]["preflight"] = {"seconds": clock.elapsed(), "workers": workers}

    def seeds_for(role):
        if smoke:
            base = SMOKE_OFFSETS[role]
            n = SMOKE["corpus_roots"] if role == "corpus" else SMOKE["per_role"]
            return tuple(SMOKE_SEEDS[base : base + n])
        return fp2.COHORT_D2 if role == "D" else fp2.seeds_of(role)

    def reset_of(seed):
        r = wide_reset(int(seed))
        return {
            "object_xy": list(map(float, r["object_xy"])),
            "plate_xy": list(map(float, r["plate_xy"])),
        }

    def tasks(role, kind, **extra):
        seeds = seeds_for(role)
        check_simulated_seeds(role, seeds, smoke)
        return [{"seed": int(s), "reset": reset_of(s), "kind": kind, **extra} for s in seeds]

    max_steps = (smoke_max_steps or SMOKE["max_steps"]) if smoke else fp2.MAX_POLICY_STEPS
    report["data"] = {
        "seeds": {
            role: list(map(int, seeds_for(role)))
            for role in (
                "corpus",
                "perception_train",
                "perception_heldout",
                "calibration_C0",
                "dagger_1",
                "dagger_2",
                "dagger_3",
                "D",
            )
        }
    }

    # ----- 2. the corpus -----
    clock.check("corpus")
    started = time.monotonic()
    plan = fp2.corpus_plan(seeds_for("corpus"))
    collect_tasks = [
        t | root | {"folder": str(corpus / "episodes"), "max_steps": max_steps}
        for t, root in zip(tasks("corpus", "collect"), plan, strict=True)
    ]
    for task, root in zip(collect_tasks, plan, strict=True):
        if task["seed"] != root["seed"] or task["kind"] != "collect":
            raise fp2.GuardError("G-data: the corpus plan and its tasks disagree")
    collected = pool.map(collect_tasks, fp2.CAPS_SECONDS["corpus_collection"], "corpus")
    episodes = {
        r["episode_id"]: {"npz_sha256": r["npz_sha256"], "meta_sha256": r["meta_sha256"]}
        for r in collected
    }
    manifest_sha = rt2.seal(
        corpus,
        plan,
        episodes,
        {
            "revision": report["revision"],
            "tracked_tree_dirty": report["tracked_tree_dirty"],
            "environment": report["environment"],
            "smoke": smoke,
        },
    )
    splits = {k: sum(root["split"] == k for root in plan) for k in ("train", "val", "test")}
    if not smoke and splits != fp2.EXPECTED_CORPUS_SPLITS:
        raise fp2.GuardError(f"G-data: corpus splits {splits} != {fp2.EXPECTED_CORPUS_SPLITS}")
    reader = rt2.CorpusReader(corpus, manifest_sha)
    report["_reader"] = reader
    report["stages"]["corpus"] = {
        "manifest_sha256": manifest_sha,
        "path": str(corpus),
        "splits": splits,
        "seconds": time.monotonic() - started,
        "all": corpus_summary(collected),
        "by_split": {
            k: corpus_summary([r for r in collected if r["split"] == k])
            for k in ("train", "val", "test")
        },
        "attempts": [
            {k: r[k] for k in ("seed", "split", "noise_level", "termination", "at_rest")}
            | {"latched": r["latched_success"], "steps": r["transitions"]}
            for r in collected
        ],
    }
    roots = {}
    for split in fp2.READ_SPLITS:
        roots[split] = []
        for root in [r for r in plan if r["split"] == split]:
            arrays, meta = reader.episode(root["episode_id"], keys=CORPUS_KEYS)
            roots[split].append(
                {
                    "episode_id": root["episode_id"],
                    "seed": root["seed"],
                    "at_rest": bool(meta["at_rest"]),
                    "frame0": arrays["frame0"],
                    "frame_sha": meta["post_look_frame_sha256"],
                    "xy": meta["truth_xy"],
                    "arrays": arrays,
                }
            )
            if frame_sha(arrays["frame0"]) != meta["post_look_frame_sha256"]:
                raise fp2.GuardError(f"G-data: {root['episode_id']} frame 0 is not post-look")
    report["data"] |= {
        "corpus_manifest_sha256": manifest_sha,
        "train_roots": [r["episode_id"] for r in roots["train"]],
        "val_roots": [r["episode_id"] for r in roots["val"]],
    }

    # ----- 3. perception frames -----
    clock.check("perception")
    started = time.monotonic()
    frame_tasks = {
        role: tasks(role, "frame") for role in ("perception_train", "perception_heldout")
    }
    both = pool.map(
        frame_tasks["perception_train"] + frame_tasks["perception_heldout"],
        fp2.CAPS_SECONDS["perception_collection"],
        "perception collection",
    )
    n_train = len(frame_tasks["perception_train"])
    frames = {"perception_train": both[:n_train], "perception_heldout": both[n_train:]}
    report["stages"]["perception_frames"] = {
        "seconds": time.monotonic() - started,
        "train_roots": len(roots["train"]),
        "val_roots": len(roots["val"]),
    }

    def tokens(encoder, frame_list):
        return fpp.featurise(encoder, np.stack(frame_list))

    fit_frames = [r["frame0"] for r in roots["train"]] + [
        f["frame"] for f in frames["perception_train"]
    ]
    fit_xy = np.asarray(
        [r["xy"] for r in roots["train"]] + [f["truth_xy"] for f in frames["perception_train"]]
    )
    if smoke:
        fit_xy = fit_xy + np.random.default_rng(0).normal(0, 0.01, fit_xy.shape)  # noise targets
    held_frames = [f["frame"] for f in frames["perception_heldout"]]
    held_xy = np.asarray([f["truth_xy"] for f in frames["perception_heldout"]])
    val_frames = [r["frame0"] for r in roots["val"]]
    feats = {}
    for name, encoder in (("P", pretrained), ("R", floor)):
        feats[name] = {
            "fit": tokens(encoder, fit_frames),
            "held": tokens(encoder, held_frames),
            "val": tokens(encoder, val_frames),
        }

    # ----- 4. readouts -----
    clock.check("readouts")
    readouts, cross = {}, {}
    for name in ("P", "R"):
        readouts[name] = fpp.XYReadout(feats[name]["fit"], fit_xy)
        cross[name], _sel = fpp.cross_fitted(feats[name]["fit"], fit_xy)
    held_est = readouts["P"].predict(feats["P"]["held"])
    apple_err, plate_err = fpp.errors_cm(held_est, held_xy)
    report["stages"]["readouts"] = {
        name: {"selection": readouts[name].selection} for name in readouts
    } | {"fit_rows": len(fit_xy)}

    # ----- 5. C0 (e9, at rest) -----
    clock.check("C0")
    started = time.monotonic()
    c0_seeds = seeds_for("calibration_C0")
    angles = np.random.default_rng(fp2.C0_DIRECTION_SEED).uniform(0, 2 * np.pi, (fp2.C0_RESETS, 2))
    conditions = [("reference", 0.0, 0.0)]
    conditions += [(f"apple_{a}", a, 0.0) for a in fp2.C0_APPLE_LEVELS_CM]
    conditions += [(f"plate_{p}", 0.0, p) for p in fp2.C0_PLATE_LEVELS_CM]
    c0_tasks = []
    for name, a_cm, p_cm in conditions:
        for i, task in enumerate(tasks("calibration_C0", "c0", max_steps=max_steps)):
            a, p = angles[i]
            c0_tasks.append(
                task
                | {
                    "condition": name,
                    "apple_offset": (a_cm / 100.0 * np.array([np.cos(a), np.sin(a)])).tolist(),
                    "plate_offset": (p_cm / 100.0 * np.array([np.cos(p), np.sin(p)])).tolist(),
                }
            )
    c0_records = pool.map(c0_tasks, fp2.CAPS_SECONDS["c0"], "C0")
    c0, c0_latched = {}, {}
    for task, record in zip(c0_tasks, c0_records, strict=True):
        c0[task["condition"]] = c0.get(task["condition"], 0) + int(record["at_rest"])
        c0_latched[task["condition"]] = c0_latched.get(task["condition"], 0) + int(
            record["latched_success"]
        )
    if smoke:
        c0 = {k: round(v * fp2.C0_RESETS / max(len(c0_seeds), 1)) for k, v in c0.items()}
    apple_c0 = {a: c0[f"apple_{a}"] for a in fp2.C0_APPLE_LEVELS_CM}
    plate_c0 = {p: c0[f"plate_{p}"] for p in fp2.C0_PLATE_LEVELS_CM}
    bars = fp2.c0_bars(c0["reference"], apple_c0, plate_c0)
    report["stages"]["C0"] = {
        "at_rest_of_32": c0,
        "latched_of_32": c0_latched,
        "bars": bars,
        "seconds": time.monotonic() - started,
        "attempts": [
            {"condition": t["condition"], "seed": t["seed"]}
            | {k: r.get(k) for k in ("termination_reason", "at_rest", "latched_success")}
            | {
                "final_distance_cm": None
                if not r.get("at_rest_detail")
                else r["at_rest_detail"]["final_distance_m"] * 100
            }
            for t, r in zip(c0_tasks, c0_records, strict=True)
        ],
    }
    if "escalate" in bars and not smoke:
        report["outcome"] = "CAL-ESCALATE"
        end_checks(report, manifest, smoke)
        return reader, pool
    if "escalate" in bars:
        bars = {
            "apple_median_cm": 0.75,
            "apple_p90_cm": 1.2,
            "plate_median_cm": 1.5,
            "plate_p90_cm": 1.5,
        }

    # ----- 6. S0-P and T -----
    s0 = (
        fp2.s0_perception(apple_err, plate_err, bars)
        if not smoke
        else {"apple_passes": True, "plate_passes": True}
    )
    report["stages"]["S0_P"] = s0 | {
        "apple_errors_cm": apple_err.tolist(),
        "plate_errors_cm": plate_err.tolist(),
    }
    if not smoke and not s0["apple_passes"]:
        report["outcome"] = "S0-APPLE-FAIL"
        report["clause_fires"] = True
        end_checks(report, manifest, smoke)
        return reader, pool
    if not smoke and not s0["plate_passes"]:
        report["outcome"] = "S0-PLATE-FAIL"
        end_checks(report, manifest, smoke)
        return reader, pool
    reference = max(c0["reference"], 1)
    threshold = fp2.a4_threshold(reference, apple_c0, plate_c0, apple_err, plate_err)
    report["stages"]["a4_threshold"] = threshold

    # ----- 7. BC-0 -----
    clock.check("BC-0")
    fk = rt2.PalmFK()
    n_corpus = len(roots["train"])
    val_est = {name: readouts[name].predict(feats[name]["val"]) for name in ("P", "R")}
    c_mean = cross["P"][:n_corpus].mean(axis=0)
    data = {
        arm: {"states": [], "steps": [], "labels": [], "est": [], "palm9": []}
        for arm in ("P", "C", "R")
    }
    val = {
        arm: {"states": [], "steps": [], "labels": [], "est": [], "palm9": []}
        for arm in ("P", "C", "R")
    }
    mask_accounting = {"train": {}, "val": {}}
    for split, target in (("train", data), ("val", val)):
        for i, root in enumerate(roots[split]):
            rows_i = rt2.bc_rows(root["arrays"], bounds)
            for key, value in rows_i["accounting"].items():
                mask_accounting[split][key] = mask_accounting[split].get(key, 0) + value
            palm = [fk.pose9(s) for s in rows_i["states"]]
            for arm in ("P", "C", "R"):
                if arm == "C":
                    est = c_mean
                elif split == "train":
                    est = cross[arm][i]
                else:
                    est = val_est[arm][i]
                n = len(rows_i["steps"])
                target[arm]["states"].extend(rows_i["states"])
                target[arm]["steps"].extend(rows_i["steps"])
                target[arm]["labels"].extend(rows_i["labels"])
                target[arm]["est"].extend([est] * n)
                target[arm]["palm9"].extend(palm)
    if smoke:
        for d in (data, val):
            for arm in d:
                d[arm]["labels"] = list(
                    np.random.default_rng(1)
                    .uniform(-0.4, 0.4, (len(d[arm]["labels"]), 7))
                    .astype(np.float32)
                )
    standardisers = {}
    for arm in ("P", "C", "R"):
        x = inputs_of(data[arm]["est"], data[arm]["steps"], data[arm]["states"], data[arm]["palm9"])
        standardisers[arm] = rt2.Standardiser.fit(x)
    report["stages"]["BC0"] = {
        "rows": len(data["P"]["steps"]),
        "val_rows": len(val["P"]["steps"]),
        "roots": len(roots["train"]),
        "val_roots": len(roots["val"]),
        "C_mean_estimates": c_mean.tolist(),
        "C_mean_definition": "mean over the train roots of the cross-fitted P estimates (per root)",
        "mask_accounting": mask_accounting,
    }

    trainings = {}
    report["stages"]["trainings"] = trainings  # filled as trainings finish, so a V keeps them

    def train_arm(arm, name, heads=1):
        clock.check(f"training {name}")
        s = standardisers[arm]
        x = s(
            inputs_of(data[arm]["est"], data[arm]["steps"], data[arm]["states"], data[arm]["palm9"])
        )
        vx = s(inputs_of(val[arm]["est"], val[arm]["steps"], val[arm]["states"], val[arm]["palm9"]))
        trained = fm2.train(
            x,
            np.asarray(data[arm]["labels"]),
            data[arm]["steps"],
            vx,
            np.asarray(val[arm]["labels"]),
            val[arm]["steps"],
            heads=heads,
            device=device,
            updates=SMOKE["updates"] if smoke else None,
            select_every=SMOKE["select_every"] if smoke else None,
            min_output_std=0.0 if smoke else None,
        )
        path = checkpoints / f"{name}.pt"
        record = trained["record"] | {"rows": len(x)}
        if trained["state"] is None:
            record["no_eligible_checkpoint"] = True
            trainings[name] = record
            return None
        record["sha256"] = save_checkpoint(path, trained, s, heads, {"arm": name})
        trainings[name] = record
        return str(path)

    ckpt = {
        "P-0": train_arm("P", "P-0"),
        "C-0": train_arm("C", "C-0"),
        "R-0": train_arm("R", "R-0"),
    }

    # ----- S0-D1 on 8 held-out perception seeds (before any DAgger rollout) -----
    clock.check("S0-D1")
    d1_arm = next((name for name in ("P-0", "C-0", "R-0") if ckpt[name]), None)
    if d1_arm is None:
        raise fp2.GuardError("S0-D1: no family has an eligible BC-0 checkpoint to check")
    d1_est = {"P-0": held_est, "R-0": readouts["R"].predict(feats["R"]["held"])}.get(
        d1_arm, np.repeat(c_mean[None], len(held_est), axis=0)
    )
    held_tasks = tasks("perception_heldout", "learned")[:8]
    d1 = pool.map(
        [
            t
            | {
                "checkpoint": ckpt[d1_arm],
                "estimates": d1_est[i].tolist(),
                "expected_frame_sha256": frames["perception_heldout"][i]["post_look_frame_sha256"],
                "first_input_check": True,
                "max_steps": 1,
                "settle_steps": 0,
            }
            for i, t in enumerate(held_tasks)
        ],
        fp2.CAPS_SECONDS["per_rollout_batch"],
        "S0-D1",
    )
    saved = torch.load(ckpt[d1_arm], map_location="cpu", weights_only=True)
    model = fm2.load(saved["state"])
    s = rt2.Standardiser(saved["mean"].numpy(), saved["std"].numpy())
    offline_inputs = np.stack(
        [
            rt2.input_vector(
                d1_est[i],
                0,
                frames["perception_heldout"][i]["post_look_state"],
                fk.pose9(frames["perception_heldout"][i]["post_look_state"]),
            )
            for i in range(len(d1))
        ]
    )
    offline = fm2.batch_predict(model, s(offline_inputs), np.zeros(len(d1), int))  # one batch
    input_worst = act_worst = 0.0
    for i, r in enumerate(d1):
        live_input = np.asarray(r["first_input"], np.float32)
        input_worst = max(input_worst, float(np.abs(live_input - offline_inputs[i]).max()))
        expected = rt2.free_of(rt2.assemble(offline[i], *bounds))
        act_worst = max(act_worst, float(np.abs(expected - np.asarray(r["commands"][0])).max()))
    report["stages"]["S0_D1"] = {
        "arm": d1_arm,
        "seeds": len(d1),
        "frames_identical": True,  # a mismatch raises G-frame inside the worker
        "max_abs_input_difference": input_worst,
        "max_abs_act_difference": act_worst,
        "tolerance": 1e-4,
    }
    if input_worst > 1e-4 or act_worst > 1e-4:
        raise fp2.GuardError(
            f"S0-D1: live differs from offline (input {input_worst}, act {act_worst})"
        )

    # ----- DAgger -----
    dagger_report = {}
    report["stages"]["dagger"] = dagger_report
    for k in range(1, fp2.DAGGER_ITERATIONS + 1):
        role = f"dagger_{k}"
        clock.check(role)
        frame_records = pool.map(
            tasks(role, "frame"), fp2.CAPS_SECONDS["per_rollout_batch"], f"{role} frames"
        )
        frames_k = [f["frame"] for f in frame_records]
        est = {
            "P": readouts["P"].predict(tokens(pretrained, frames_k)),
            "R": readouts["R"].predict(tokens(floor, frames_k)),
        }
        est["C"] = np.repeat(c_mean[None], len(frames_k), axis=0)
        dagger_report[role] = {}
        for arm in ("P", "C", "R"):
            previous = ckpt[f"{arm}-{k - 1}"]
            if previous is None:
                ckpt[f"{arm}-{k}"] = None
                dagger_report[role][arm] = {"skipped": "no_eligible_checkpoint"}
                continue
            arm_tasks = [
                t
                | {
                    "checkpoint": previous,
                    "estimates": est[arm][i].tolist(),
                    "label": True,
                    "expected_frame_sha256": frame_records[i]["post_look_frame_sha256"],
                    "max_steps": max_steps,
                }
                for i, t in enumerate(tasks(role, "learned"))
            ]
            records = pool.map(arm_tasks, fp2.CAPS_SECONDS["per_rollout_batch"], f"{role} {arm}")
            added = 0
            for i, r in enumerate(records):
                n = len(r["steps"])
                data[arm]["states"].extend(np.asarray(r["states"]))
                data[arm]["steps"].extend(r["steps"])
                data[arm]["labels"].extend(np.asarray(r["labels"], np.float32))
                data[arm]["est"].extend([est[arm][i]] * n)
                data[arm]["palm9"].extend(np.asarray(r["palm9"]) if n else [])
                added += n
            if smoke:
                data[arm]["labels"] = list(
                    np.random.default_rng(k)
                    .uniform(-0.4, 0.4, (len(data[arm]["labels"]), 7))
                    .astype(np.float32)
                )
            dagger_report[role][arm] = {
                "rows_added": added,
                "rows_total": len(data[arm]["steps"]),
                "rollouts": count(records),
            }
            ckpt[f"{arm}-{k}"] = train_arm(arm, f"{arm}-{k}")

    # ----- 8. M1 on D2 -----
    clock.check("M1")
    d_frames = pool.map(tasks("D", "frame"), fp2.CAPS_SECONDS["per_rollout_batch"], "D2 frames")
    d_frame_list = [f["frame"] for f in d_frames]
    d_est = {
        "P": readouts["P"].predict(tokens(pretrained, d_frame_list)),
        "R": readouts["R"].predict(tokens(floor, d_frame_list)),
    }
    d_truth = np.asarray([f["truth_xy"] for f in d_frames])
    library = [r for r in roots["train"] if r["at_rest"]]
    if not library and smoke:
        library = roots["train"]  # smoke only: a tiny, truncated corpus holds no at-rest root
    if not library:
        raise fp2.GuardError("B-replay: the retrieval library is empty")
    lib_cls = fpp.featurise_cls(pretrained, [r["frame0"] for r in library])
    d_cls = fpp.featurise_cls(pretrained, d_frame_list)
    mu, sd = lib_cls.mean(axis=0), np.maximum(lib_cls.std(axis=0), fp2.INPUT_STD_FLOOR)
    nearest = [
        int(np.argmin(np.linalg.norm((lib_cls - mu) / sd - (c - mu) / sd, axis=1))) for c in d_cls
    ]

    def d_tasks(kind, per_seed=lambda i: {}):
        base = tasks("D", kind, max_steps=max_steps)
        return [
            t | {"expected_frame_sha256": d_frames[i]["post_look_frame_sha256"]} | per_seed(i)
            for i, t in enumerate(base)
        ]

    arms = {}
    for name in ("P-0", "P-1", "P-2", "P-3", "C-3", "R-3"):
        path = ckpt[name]
        family = name[0]
        if path is None:
            arms[name] = None  # no eligible checkpoint: 0/16
            continue
        est = d_est.get(family, np.repeat(c_mean[None], len(d_frames), axis=0))
        arms[name] = pool.map(
            d_tasks(
                "learned", lambda i, p=path, e=est: {"checkpoint": p, "estimates": e[i].tolist()}
            ),
            fp2.CAPS_SECONDS["per_rollout_batch"],
            f"M1 {name}",
        )
        check_privileged(arms[name], name)
    arms["A4-look"] = pool.map(
        d_tasks("a4", lambda i: {"estimates": d_est["P"][i].tolist()}),
        fp2.CAPS_SECONDS["per_rollout_batch"],
        "M1 A4-look",
    )
    for name, kind in (("B-oracle", "oracle"), ("B-hold", "hold"), ("B-random", "random")):
        arms[name] = pool.map(d_tasks(kind), fp2.CAPS_SECONDS["per_rollout_batch"], f"M1 {name}")
    arms["B-replay"] = pool.map(
        d_tasks(
            "replay",
            lambda i: {"actions": rt2.replay_actions(library[nearest[i]]["arrays"]).tolist()},
        ),
        fp2.CAPS_SECONDS["per_rollout_batch"],
        "M1 B-replay",
    )
    zero = {"grasp": 0, "success": 0, "latched": 0}
    counts = {name: (count(r) if r is not None else dict(zero)) for name, r in arms.items()}
    decision = fp2.decide_m1(counts | {"D-oracle-perc": dict(zero)}, threshold)
    carried = decision.get("carried", "P-3")
    if ckpt.get(carried):
        arms["D-oracle-perc"] = pool.map(
            d_tasks(
                "learned", lambda i: {"checkpoint": ckpt[carried], "estimates": d_truth[i].tolist()}
            ),
            fp2.CAPS_SECONDS["per_rollout_batch"],
            "M1 D-oracle-perc",
        )
    counts["D-oracle-perc"] = count(arms.get("D-oracle-perc") or [])
    decision = fp2.decide_m1(counts, threshold)
    f_counts = None
    if decision["row"] == "M1-MOTOR":
        # F: per-phase heads on P-3's final aggregate, evaluated once on D2.
        ckpt["F"] = train_arm("P", "F", heads=len(fp2.EXPERT_PHASES))
        if ckpt["F"] is None:
            f_counts = dict(zero)
        else:
            arms["F"] = pool.map(
                d_tasks(
                    "learned",
                    lambda i: {"checkpoint": ckpt["F"], "estimates": d_est["P"][i].tolist()},
                ),
                fp2.CAPS_SECONDS["per_rollout_batch"],
                "M1 F",
            )
            check_privileged(arms["F"], "F")
            f_counts = count(arms["F"])
        decision = fp2.decide_m1(counts, threshold, f_counts=f_counts)
    report["stages"]["M1"] = {
        "counts": counts,
        "f_counts": f_counts,
        "a4_threshold": threshold,
        "b_replay_library_roots": [r["episode_id"] for r in library],
        "b_replay_nearest": [library[j]["episode_id"] for j in nearest],
        "attempts": {name: attempt_summary(r or []) for name, r in arms.items()},
        "no_eligible_checkpoint": [n for n, p in ckpt.items() if p is None],
    }
    report["decision"] = decision
    report["outcome"] = decision["row"]
    if decision["row"] == "V":
        report["void_reason"] = decision.get("void_reason")
    report["clause_fires"] = bool(decision.get("clause_fires"))
    report["checkpoints"] = {n: p for n, p in ckpt.items()}
    end_checks(report, manifest, smoke)
    return reader, pool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("run", "smoke"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--smoke-max-steps",
        type=int,
        default=None,
        help="smoke only: policy commands per attempt (default 40; 740 is full length)",
    )
    args = parser.parse_args()
    if args.smoke_max_steps is not None and (
        args.mode != "smoke" or not 1 <= args.smoke_max_steps <= fp2.MAX_POLICY_STEPS
    ):
        parser.error("--smoke-max-steps is for smoke runs only, from 1 to 740")
    report = run(
        args.output,
        args.checkpoints,
        args.corpus,
        smoke=args.mode == "smoke",
        smoke_max_steps=args.smoke_max_steps,
    )
    return 0 if report.get("outcome") not in (None, "V") else 1


if __name__ == "__main__":
    sys.exit(main())
