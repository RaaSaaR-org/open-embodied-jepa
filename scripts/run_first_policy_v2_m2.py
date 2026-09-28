"""TASK-072 M2 runner (``apple_first_policy_v2_m2``): P-3 on cohort C, with its controls.

Protocol ``docs/experiments/apple_first_policy_v2_m2.md``; design ``first_policy_v2_m2.py``
(M2 as TASK-071 §12 preregistered it, plus what the go-ahead fixes). It reuses TASK-072's runner
(``scripts/run_first_policy_v2_linux.py``, pinned, unmodified) for the worker's robot, the
checkpoint loader and the report helpers, and adds only:

- G-evidence: TASK-072 run-1's report, P-3/C-3/R-3 checkpoints and corpus manifest by sha256;
- G-repro: the readouts refitted from run-1's frames must reproduce run-1's recorded facts;
- G-cohort / G-seeds: the 40 stored cohort-C resets, never recomputed, and only those seeds;
- G7's timing: each learned command's ``act()`` time, plus the one DINOv2 forward pass.

Modes: ``preflight`` stops after G-repro and simulates no cohort-C seed (the readiness check);
``smoke`` runs the cohort stage on four smoke seeds (52100-52103, wide_reset stand-ins) and
nothing in it is read; ``run`` is the gated run. **Learned Apple->Plate on the frozen benchmark
is still 0 successes.**

    uv run --no-sync python scripts/run_first_policy_v2_m2.py run \\
        --output outputs/task072-m2-cohort-c/run-1 \\
        --evidence /home/huhn/develop/emai/worktrees/task072-run
"""

from __future__ import annotations

import argparse
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

from embodied_jepa import devices  # noqa: E402
from embodied_jepa import first_policy_v2 as fp2  # noqa: E402
from embodied_jepa import first_policy_v2_linux as fpl  # noqa: E402
from embodied_jepa import first_policy_v2_m2 as fm  # noqa: E402
from embodied_jepa import first_policy_v2_runtime as rt2  # noqa: E402

MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-first-policy-v2-m2.json"
POLICY_V1_MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-policy-v1.json"
fpl.configure_headless()  # MUJOCO_GL=egl before any MuJoCo import; workers inherit it


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LIN = _load("_task072_runner", "scripts/run_first_policy_v2_linux.py")
R65 = LIN.R65
log = LIN.log
SMOKE_STAND_IN = tuple(range(fp2.SMOKE_SEEDS[0], fp2.SMOKE_SEEDS[0] + 4))  # 52100-52103
SMOKE = {"workers": 4, "max_steps": 40}
MODES = ("preflight", "smoke", "run")


# ----- seeds and resets ---------------------------------------------------------------------------
def check_rerender_seeds(role: str, seeds) -> tuple[int, ...]:
    """G-seeds for the re-rendered (frame-only) roles: exactly run-1's seeds; never cohort C."""
    seeds = tuple(seeds)
    if any(type(s) is not int for s in seeds):
        raise fp2.GuardError("G-seeds: seeds must be plain ints")
    if set(seeds) & set(fm.COHORT_C):
        raise fp2.GuardError("G-seeds: cohort C is not re-rendered before the cohort stage")
    allowed = fp2.COHORT_D2 if role == "D" else fp2.seeds_of(role)
    if seeds != tuple(allowed):
        raise fp2.GuardError(f"G-seeds: {role} must be run-1's {role} seeds, in order")
    return seeds


def cohort_tasks(mode: str, wide_reset=None) -> tuple[tuple[int, ...], dict]:
    """The cohort stage's seeds and resets. ``run``: the stored cohort-C values, never
    recomputed. ``smoke``: four smoke seeds from ``wide_reset`` (stand-ins, nothing read)."""
    if mode == "run":
        resets = fm.stored_cohort(json.loads(POLICY_V1_MANIFEST.read_text()))
        return fm.check_cohort_seeds(tuple(resets)), resets
    if mode == "smoke":
        resets = {}
        for seed in SMOKE_STAND_IN:
            r = wide_reset(int(seed))
            resets[seed] = {
                "object_xy": list(map(float, r["object_xy"])),
                "plate_xy": list(map(float, r["plate_xy"])),
            }
        return SMOKE_STAND_IN, resets
    raise fp2.GuardError(f"mode {mode!r} has no cohort stage")


# ----- the worker ---------------------------------------------------------------------------------
class TimedController:
    """Times a learned controller's ``act()`` (G7). It holds no robot or simulator handle, and
    it is called inside the privileged-read counter's ``act``, so G-privileged still covers it."""

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
        self.inner.advance(result)


def worker_init() -> None:
    LIN.worker_init()  # TASK-072's worker: the v2 robot (G-scene), FK, bounds, model cache


def run_task(task: dict) -> dict:
    """One M2 attempt (or a frame-only render) in a worker. ``task["kind"]`` names it."""
    from embodied_jepa.policy_diagnostics import RandomController, hold_controller

    w = LIN._W
    robot = w["robot"]
    truth, scorer, counter, _observation, facts = rt2.reset_and_look(
        robot, task["seed"], task["reset"]
    )
    out = {
        "seed": task["seed"],
        "kind": task["kind"],
        "reset": task["reset"],
        "post_look_frame_sha256": LIN.frame_sha(facts["post_look_frame"]),
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
    if out["post_look_frame_sha256"] != task["expected_frame_sha256"]:
        counter.remove()
        robot.stop("frame_mismatch")
        raise fp2.GuardError(f"G-frame: seed {task['seed']} post-look frame differs")
    kind = task["kind"]
    if kind == "learned":
        controller = TimedController(LIN._learned(task))
    elif kind == "oracle":
        controller = rt2.ExpertController(truth, robot)
    elif kind == "hold":
        controller = rt2.HarnessController(hold_controller())
    elif kind == "random":
        lower, upper = w["bounds"]
        controller = rt2.HarnessController(
            RandomController(fp2.RANDOM_CONTROLLER_SEED, lower, upper)
        )
    elif kind == "replay":
        controller = rt2.ReplayController(task["actions"])
    else:
        counter.remove()
        raise fp2.GuardError(f"unknown controller kind {kind!r}")
    record = rt2.run_attempt(
        robot,
        scorer,
        counter,
        controller,
        bounds=w["bounds"],
        max_steps=int(task["max_steps"]),
        settle_steps=fp2.SETTLE_STEPS,
    )
    out.update({k: v for k, v in record.items() if k not in ("states", "steps", "labels")})
    out["commands"] = record["commands"].tolist()
    if kind == "learned":
        out["act_seconds"] = controller.seconds
    return out


class Pool:
    look_reference = None

    def __init__(self, workers: int):
        self.pool = mp.get_context("spawn").Pool(workers, initializer=worker_init)

    def map(self, tasks: list[dict], cap: float, what: str) -> list[dict]:
        started = time.monotonic()
        result = self.pool.map_async(run_task, tasks, chunksize=1)
        try:
            out = result.get(timeout=max(cap - (time.monotonic() - started), 1.0))
        except mp.TimeoutError as error:
            raise fp2.GuardError(f"G-cap: {what} exceeded its {cap} s cap") from error
        self.look_reference = LIN.check_look_states(out, self.look_reference)  # every attempt
        return out

    def close(self):
        self.pool.terminate()
        self.pool.join()


# ----- G-evidence and G-repro ---------------------------------------------------------------------
def check_evidence(evidence: Path) -> dict:
    """TASK-072 run-1's report, carried checkpoints and corpus manifest, by sha256."""
    e = fm.EVIDENCE
    found = {"report": R65.sha256_file(evidence / e["report"])}
    if found["report"] != e["report_sha256"]:
        raise fp2.GuardError("G-evidence: run-1's report.json differs from its pin")
    for arm, want in e["checkpoints"].items():
        found[arm] = R65.sha256_file(evidence / e["checkpoint_dir"] / f"{arm}.pt")
        if found[arm] != want:
            raise fp2.GuardError(f"G-evidence: {arm}.pt differs from its pin")
    found["corpus_manifest"] = R65.sha256_file(evidence / e["corpus"] / "manifest.json")
    if found["corpus_manifest"] != e["corpus_manifest_sha256"]:
        raise fp2.GuardError("G-evidence: run-1's corpus manifest differs from its pin")
    run1 = json.loads((evidence / e["report"]).read_text())
    if (
        run1["revision"] != e["revision"]
        or run1["outcome"] != e["outcome"]
        or run1["decision"]["carried"] != e["carried"]
        or run1["test_split_decoded"]
        or run1["tracked_tree_dirty"]
        or run1["smoke"]
    ):
        raise fp2.GuardError("G-evidence: run-1's report is not the recorded M1-PASS run")
    return {"sha256": found, "run1": run1}


def check_reproduction(recomputed: dict, run1: dict) -> dict:
    """G-repro: every recorded readout fact of run-1 is reproduced exactly (see the design)."""
    stages = run1["stages"]
    checks = {
        "readout_P_selection": recomputed["selection"]["P"] == stages["readouts"]["P"]["selection"],
        "readout_R_selection": recomputed["selection"]["R"] == stages["readouts"]["R"]["selection"],
        "fit_rows": recomputed["fit_rows"] == stages["readouts"]["fit_rows"],
        "s0p_apple_errors": recomputed["apple_errors_cm"] == stages["S0_P"]["apple_errors_cm"],
        "s0p_plate_errors": recomputed["plate_errors_cm"] == stages["S0_P"]["plate_errors_cm"],
        "c_mean": recomputed["c_mean"] == stages["BC0"]["C_mean_estimates"],
        "b_replay_library": recomputed["library"] == stages["M1"]["b_replay_library_roots"],
        "d2_b_replay_nearest": recomputed["d2_nearest"] == stages["M1"]["b_replay_nearest"],
    }
    if not all(checks.values()):
        failed = [k for k, ok in checks.items() if not ok]
        raise fp2.GuardError(f"G-repro: run-1's readout facts are not reproduced: {failed}")
    return checks


def _json_floats(values) -> list:
    """Floats as run-1's report stored them (the JSON round trip), for exact comparison."""
    return json.loads(json.dumps(np.asarray(values, np.float64).tolist()))


# ----- G7 -----------------------------------------------------------------------------------------
def control_times(records: list[dict], forward_seconds: list[float]) -> dict:
    """G7's per-command control times for the carried arm (``fm.CONTROL_TIME``)."""
    per_command, conservative = [], []
    for record, forward in zip(records, forward_seconds, strict=True):
        acts = list(record["act_seconds"])
        if not acts:
            continue
        per_command.extend([acts[0] + forward, *acts[1:]])
        conservative.extend(a + forward for a in acts)
    if not per_command:
        return {"median_seconds": None, "commands": 0}
    t = np.asarray(per_command)
    return {
        "median_seconds": float(np.median(t)),
        "p90_seconds": float(np.quantile(t, 0.9)),
        "max_seconds": float(t.max()),
        "commands": len(t),
        "forward_and_readout_seconds": {
            "median": float(np.median(forward_seconds)),
            "max": float(np.max(forward_seconds)),
        },
        "conservative_forward_every_command_median_seconds": float(np.median(conservative)),
    }


# ----- the run ------------------------------------------------------------------------------------
def end_checks(report, manifest, mode) -> None:
    report["pinned_hashes_at_end"] = R65.check_pins(manifest["hashes"])
    fpl.check_determinism(report)  # G-device at the end: still strict
    report["revision_at_end"] = R65.revision()
    if mode != "smoke" and R65.tracked_tree_dirty():
        raise fp2.GuardError("G-hash: the tracked tree changed during the run")


def _run(report, evidence: Path, clock, mode: str, smoke_max_steps: int | None):  # noqa: C901
    import torch

    from embodied_jepa import first_policy_perception as fpp
    from embodied_jepa import pretrained_encoder as pe

    # ----- 1. preflight -----
    manifest = json.loads(MANIFEST.read_text())
    if manifest["frozen"] != fm.frozen_block():
        raise fp2.GuardError("G-frozen: the manifest's frozen block differs from the module")
    report["pinned_hashes_at_preflight"] = R65.check_pins(manifest["hashes"])
    dirty = R65.tracked_tree_dirty()
    report["revision"], report["tracked_tree_dirty"] = R65.revision(), bool(dirty)
    if mode != "smoke":
        R65.check_clean(dirty)
    report["platform"] = fpl.check_platform()  # G-platform
    if not devices.available("cuda"):
        raise fp2.GuardError("G-device: CUDA is not available")
    device = devices.require("cuda", strict=fm.DETERMINISM["strict"])  # run-1's process state
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
        "accelerator": devices.accelerator_info(device),
        "determinism": devices.determinism_state(),
        "torch_threads_main": torch.get_num_threads(),
    }
    pretrained, floor = pe.load_pretrained(), pe.random_init()
    digests = {"pretrained": pe.weights_digest(pretrained), "floor": pe.weights_digest(floor)}
    report["encoder_digests"] = digests
    if digests != manifest["encoder_digests"]:
        raise fp2.GuardError("G-weights: an encoder digest differs from its pin")

    # ----- 2. G-evidence and the stop rule -----
    found = check_evidence(evidence)
    run1 = found["run1"]
    report["evidence"] = {"root": str(evidence), "sha256": found["sha256"]}
    runs_carried = fm.stop_rule_runs_carried(run1["stages"]["M1"]["counts"])
    report["stop_rule"] = {"carried_runs": runs_carried, "rule": fm.STOP_RULE}
    ckpt_dir = evidence / fm.EVIDENCE["checkpoint_dir"]
    checkpoints = {arm: str(ckpt_dir / f"{arm}.pt") for arm in fm.LEARNED}
    for arm, path in checkpoints.items():  # the checkpoints load, on the CPU, before any worker
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if int(saved["heads"]) != 1:
            raise fp2.GuardError(f"G-evidence: {arm} is not a one-head policy")
    workers = SMOKE["workers"] if mode == "smoke" else fm.SIM_WORKERS
    pool = Pool(workers)
    report["_pool"] = pool
    report["stages"]["preflight"] = {"seconds": clock.elapsed(), "workers": workers}

    # ----- 3. run-1's corpus: the train split only -----
    clock.check("corpus")
    reader = rt2.CorpusReader(
        evidence / fm.EVIDENCE["corpus"], fm.EVIDENCE["corpus_manifest_sha256"], splits=("train",)
    )
    report["_reader"] = reader
    train_ids = list(reader.manifest["splits"]["train"])
    if train_ids != run1["data"]["train_roots"]:
        raise fp2.GuardError("G-data: the corpus's train roots are not run-1's")
    roots = []
    for episode_id in train_ids:
        arrays, meta = reader.episode(episode_id, keys=("phase", "applied"))
        if LIN.frame_sha(arrays["frame0"]) != meta["post_look_frame_sha256"]:
            raise fp2.GuardError(f"G-data: {episode_id} frame 0 is not post-look")
        roots.append(
            {
                "episode_id": episode_id,
                "success": bool(meta["success"]),
                "frame0": arrays["frame0"],
                "xy": meta["truth_xy"],
                "arrays": arrays,
            }
        )

    # ----- 4. re-render run-1's perception and D2 post-look frames (frame only) -----
    clock.check("perception")
    wide_reset = LIN.load_wide_reset()
    started = time.monotonic()
    frame_tasks = []
    for role in ("perception_train", "perception_heldout", "D"):
        seeds = check_rerender_seeds(role, run1["data"]["seeds"][role])
        for seed in seeds:
            r = wide_reset(seed)
            reset = {
                "object_xy": list(map(float, r["object_xy"])),
                "plate_xy": list(map(float, r["plate_xy"])),
            }
            frame_tasks.append({"seed": seed, "reset": reset, "kind": "frame", "role": role})
    rendered = pool.map(frame_tasks, fm.CAPS_SECONDS["perception_collection"], "re-render")
    frames = {role: [] for role in ("perception_train", "perception_heldout", "D")}
    for task, out in zip(frame_tasks, rendered, strict=True):
        frames[task["role"]].append(out)
    report["stages"]["rerender"] = {
        "seconds": time.monotonic() - started,
        "frames": {role: len(v) for role, v in frames.items()},
    }

    # ----- 5. the readouts, refitted; G-repro -----
    clock.check("readouts")
    started = time.monotonic()
    fit_frames = [r["frame0"] for r in roots] + [f["frame"] for f in frames["perception_train"]]
    fit_xy = np.asarray(
        [r["xy"] for r in roots] + [f["truth_xy"] for f in frames["perception_train"]]
    )
    held_frames = [f["frame"] for f in frames["perception_heldout"]]
    held_xy = np.asarray([f["truth_xy"] for f in frames["perception_heldout"]])
    readouts = {}
    fit_feats = {}
    for name, encoder in (("P", pretrained), ("R", floor)):
        fit_feats[name] = fpp.featurise(encoder, np.stack(fit_frames))
        readouts[name] = fpp.XYReadout(fit_feats[name], fit_xy)
    cross_p, _sel = fpp.cross_fitted(fit_feats["P"], fit_xy)
    c_mean = cross_p[: len(roots)].mean(axis=0)
    held_est = readouts["P"].predict(fpp.featurise(pretrained, np.stack(held_frames)))
    apple_err, plate_err = fpp.errors_cm(held_est, held_xy)
    library = [r for r in roots if r["success"]]  # counted successes (T71-R1/R2), as run-1
    lib_cls = fpp.featurise_cls(pretrained, [r["frame0"] for r in library])
    mu, sd = lib_cls.mean(axis=0), np.maximum(lib_cls.std(axis=0), fp2.INPUT_STD_FLOOR)

    def nearest_of(frame_list):
        cls = fpp.featurise_cls(pretrained, frame_list)
        return [
            int(np.argmin(np.linalg.norm((lib_cls - mu) / sd - (c - mu) / sd, axis=1))) for c in cls
        ]

    d2_nearest = nearest_of([f["frame"] for f in frames["D"]])
    recomputed = {
        "selection": {n: json.loads(json.dumps(readouts[n].selection)) for n in readouts},
        "fit_rows": len(fit_xy),
        "apple_errors_cm": _json_floats(apple_err),
        "plate_errors_cm": _json_floats(plate_err),
        "c_mean": _json_floats(c_mean),
        "library": [r["episode_id"] for r in library],
        "d2_nearest": [library[j]["episode_id"] for j in d2_nearest],
    }
    report["stages"]["reproduction"] = {
        "checks": check_reproduction(recomputed, run1),
        "seconds": time.monotonic() - started,
        "library_roots": len(library),
    }
    del fit_feats
    if mode == "preflight":
        # G-cohort on the stored values (a JSON read: nothing is rendered or simulated)
        stored = fm.stored_cohort(json.loads(POLICY_V1_MANIFEST.read_text()))
        report["stages"]["cohort_check"] = {
            "seeds": list(fm.check_cohort_seeds(tuple(stored))),
            "cohort_sha256": fm.COHORT_SOURCE["cohort_sha256"],
        }
        report["outcome"] = "PREFLIGHT-READY"
        end_checks(report, manifest, mode)
        return

    # ----- 6. the cohort: post-look frames, estimates, G7's forward timing -----
    clock.check("cohort frames")
    cohort_started = time.monotonic()
    seeds, resets = cohort_tasks(mode, wide_reset)
    max_steps = (smoke_max_steps or SMOKE["max_steps"]) if mode == "smoke" else fp2.MAX_POLICY_STEPS
    report["cohort"] = {
        "mode": mode,
        "seeds": list(seeds),
        "source": fm.COHORT_SOURCE if mode == "run" else "smoke stand-ins from wide_reset",
        "resets": {str(s): resets[s] for s in seeds},
        "max_policy_steps": max_steps,
    }
    # The void rule's boundary (§3.7): from here on, the cohort stage's seeds are simulated.
    report["cohort_first_render_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    c_frames = pool.map(
        [{"seed": s, "reset": resets[s], "kind": "frame"} for s in seeds],
        fm.CAPS_SECONDS["per_rollout_batch"],
        "cohort frames",
    )
    for s, f in zip(seeds, c_frames, strict=True):  # results come back in seed order
        if f["seed"] != s:
            raise fp2.GuardError("G-cohort: a rendered frame is not its seed's")
    c_list = [f["frame"] for f in c_frames]
    tokens, forward_seconds = {"P": [], "R": []}, []
    for frame in c_list:  # batch size 1, as featurise; the P forward + readout is timed (G7)
        t0 = time.perf_counter()
        tok = fpp.featurise(pretrained, frame[None])
        readouts["P"].predict(tok)
        forward_seconds.append(time.perf_counter() - t0)
        tokens["P"].append(tok)
        tokens["R"].append(fpp.featurise(floor, frame[None]))
    est = {n: readouts[n].predict(np.concatenate(tokens[n])) for n in ("P", "R")}
    est["C"] = np.repeat(c_mean[None], len(seeds), axis=0)
    truth_xy = np.asarray([f["truth_xy"] for f in c_frames])
    c_apple_err, c_plate_err = fpp.errors_cm(est["P"], truth_xy)
    nearest = nearest_of(c_list)
    report["stages"]["cohort_frames"] = {
        "post_look_frame_sha256": [f["post_look_frame_sha256"] for f in c_frames],
        "estimates": {n: est[n].tolist() for n in est},
        "readout_errors_cm_descriptive": {
            "apple": c_apple_err.tolist(),
            "plate": c_plate_err.tolist(),
        },
        "b_replay_nearest": [library[j]["episode_id"] for j in nearest],
        "forward_and_readout_seconds": forward_seconds,
    }
    report["stages"]["cohort_frames"]["seconds"] = time.monotonic() - cohort_started
    arms_started = time.monotonic()

    # ----- 7. the arms, each on each reset once -----
    def arm_tasks(kind, per_seed=lambda i: {}):
        return [
            {
                "seed": s,
                "reset": resets[s],
                "kind": kind,
                "max_steps": max_steps,
                "expected_frame_sha256": c_frames[i]["post_look_frame_sha256"],
            }
            | per_seed(i)
            for i, s in enumerate(seeds)
        ]

    arms = {}
    for arm in fm.LEARNED:
        if arm == fm.CARRIED and not runs_carried:
            arms[arm] = None  # the stop rule: the carried arm does not run; M2 fails
            continue
        e = est[arm[0]]
        clock.check(f"arm {arm}")
        arms[arm] = pool.map(
            arm_tasks(
                "learned",
                lambda i, p=checkpoints[arm], e=e: {"checkpoint": p, "estimates": e[i].tolist()},
            ),
            fm.CAPS_SECONDS["per_rollout_batch"],
            f"M2 {arm}",
        )
    clock.check("arm B-replay")
    arms["B-replay"] = pool.map(
        arm_tasks(
            "replay",
            lambda i: {"actions": rt2.replay_actions(library[nearest[i]]["arrays"]).tolist()},
        ),
        fm.CAPS_SECONDS["per_rollout_batch"],
        "M2 B-replay",
    )
    for arm, kind in (("B-oracle", "oracle"), ("B-hold", "hold"), ("B-random", "random")):
        clock.check(f"arm {arm}")
        arms[arm] = pool.map(arm_tasks(kind), fm.CAPS_SECONDS["per_rollout_batch"], f"M2 {arm}")

    # ----- 8. the decision -----
    privileged_ok = {
        arm: all(rt2.privileged_reads_ok(r) for r in arms[arm])
        for arm in fm.LEARNED
        if arms[arm] is not None
    }
    timing = (
        control_times(arms[fm.CARRIED], forward_seconds)
        if arms[fm.CARRIED] is not None
        else {"median_seconds": None, "commands": 0}
    )
    counts = {arm: LIN.count(r) if r is not None else None for arm, r in arms.items()}
    per_reset = {
        arm: None
        if r is None
        else {k: [bool(x[k]) for x in r] for k in ("success", "grasp", "at_rest")}
        for arm, r in arms.items()
    }
    m2_stage = {
        "counts": counts,
        "per_reset": per_reset,
        "privileged_ok": privileged_ok,
        "control_time": timing,
        "attempts": {arm: LIN.attempt_summary(r or []) for arm, r in arms.items()},
        "seconds": time.monotonic() - arms_started,
    }
    finish(report, manifest, mode, m2_stage)


def finish(report: dict, manifest: dict, mode: str, m2_stage: dict) -> None:
    """Decide, run the end checks, and only then write the counts and the verdict.

    Nothing that reveals the result reaches ``report`` before ``decide_m2`` and ``end_checks``
    have both succeeded, so a V report carries no arm count, per-reset result or row."""
    decision = None
    if mode == "run":
        per_reset = m2_stage["per_reset"]
        decision = fm.decide_m2(
            {a: None if v is None else v["success"] for a, v in per_reset.items()},
            {a: None if v is None else v["grasp"] for a, v in per_reset.items()},
            m2_stage["privileged_ok"],
            m2_stage["control_time"]["median_seconds"],
        )
    end_checks(report, manifest, mode)
    report["stages"]["M2"] = m2_stage
    if decision is not None:
        report["decision"] = decision
        report["outcome"] = decision["row"]
    else:
        report["outcome"] = "SMOKE-COMPLETE"  # nothing in a smoke is read


def run(output: Path, evidence: Path, mode: str, smoke_max_steps: int | None = None) -> dict:
    output, evidence = Path(output), Path(evidence).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    report = {
        "protocol": fm.PROTOCOL,
        "task": fm.TASK,
        "step": fm.STEP,
        "mode": mode,
        "smoke_max_steps": smoke_max_steps,
        "outcome": None,
        "stages": {},
        "paths": {"output": str(output), "evidence": str(evidence)},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    clock = R65.Clock(fm.CAPS_SECONDS["global"])
    try:
        _run(report, evidence, clock, mode, smoke_max_steps)
    except Exception as error:  # noqa: BLE001 - every failure is V, and the report is written
        report["outcome"] = "V"
        report.pop("decision", None)  # a V carries no verdict and no arm results
        report["stages"].pop("M2", None)
        report["void_reason"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        log(f"VOID: {report['void_reason']}")
    finally:
        pool = report.pop("_pool", None)
        if pool is not None:
            pool.close()
        reader = report.pop("_reader", None)
        if reader is not None:
            report["decoded_episodes"] = len(reader.decoded)
            report["test_split_decoded"] = reader.test_split_decoded
        report["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["total_seconds"] = clock.elapsed()
        R65.write_report(output / "report.json", report)
        log(f"report written: outcome {report.get('outcome')}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evidence",
        type=Path,
        required=True,
        help="the checkout that holds TASK-072 run-1's outputs/, checkpoints/ and data/",
    )
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
    report = run(args.output, args.evidence, args.mode, args.smoke_max_steps)
    return 0 if report.get("outcome") not in (None, "V") else 1


if __name__ == "__main__":
    sys.exit(main())
