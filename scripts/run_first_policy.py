"""TASK-067 runner (``apple_first_policy_v1``): an information-matched learned policy on D.

Protocol ``docs/experiments/apple_first_policy_v1.md``; design ``first_policy.py``;
run-time pieces ``first_policy_runtime.py``, ``first_policy_model.py`` and
``first_policy_perception.py``. **Learned Apple->Plate is still 0 successes.**

Stages (a guard failure or an exception is V, and ``report.json`` is still written):

1. preflight: G-hash, the clean tree, G-seeds, G-device, G-data, Q-split, G-weights;
2. perception frames: reset + look on the perception seeds (worker path), the corpus's train and
   val post-look frames; DINOv2 tokens and CLS at batch size 1 (pretrained and floor);
3. readouts: the frozen P and R readouts, cross-fitted BC-0 estimates, S0-P's held-out errors;
4. C0: nine conditions x 32 resets; the bars (``c0_bars``) or CAL-ESCALATE;
5. S0-P (S0-APPLE-FAIL / S0-PLATE-FAIL) and the A4-look threshold T; S0-D1 on 8 seeds;
6. BC-0 data; P-0, C-0, R-0 on MPS; three DAgger iterations for P, C and R;
7. M1 on D: every M1 arm once per reset; ``decide_m1``; D-oracle-perc; F only on M1-MOTOR.

    uv run --no-sync python scripts/run_first_policy.py run \\
        --output outputs/task067-first-policy/run-1 \\
        --checkpoints checkpoints/task067-first-policy/run-1

``smoke`` runs every stage on CPU on a tiny subset, with the reserved smoke seeds (46900-46999)
in every simulated role, 2 workers, 20 updates and noise targets. Nothing in a smoke is read.
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

from embodied_jepa import first_policy as fp  # noqa: E402
from embodied_jepa import first_policy_runtime as rt  # noqa: E402

MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-first-policy-v1.json"
SMOKE_SEEDS = tuple(range(fp.SMOKE_SEEDS[0], fp.SMOKE_SEEDS[1] + 1))
PHASE_CLOSE = fp.PHASES.index("close")  # 2, as world_model_v2.PHASE_CLOSE
DISPLACEMENT_LIMIT_M = 0.01  # cloning.DISPLACEMENT_LIMIT_M
SMOKE = {
    "workers": 2,
    "updates": 20,
    "select_every": 10,
    "train_roots": 4,
    "val_roots": 2,
    "per_role": 2,
    "max_steps": 40,
    "c0_resets": 2,
}


def load_task065_runner():
    spec = importlib.util.spec_from_file_location(
        "_task065_runner", ROOT / "scripts" / "train_apple_latent_dynamics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_wide_reset():
    spec = importlib.util.spec_from_file_location(
        "_evaluate_apple_reset", ROOT / "scripts" / "evaluate_apple.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.wide_reset


R65 = load_task065_runner()
log = R65.log


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def frame_sha(frame) -> str:
    return sha256_bytes(np.ascontiguousarray(frame, np.uint8).tobytes())


# ----- guards -----
def check_simulated_seeds(role: str, seeds, smoke: bool) -> None:
    """G-seeds: a whitelist per role; cohort C is refused before anything is simulated."""
    seeds = tuple(int(s) for s in seeds)
    if set(seeds) & set(fp.COHORT_C):
        raise fp.GuardError("G-seeds: cohort C is not opened by this run")
    if len(set(seeds)) != len(seeds):
        raise fp.GuardError(f"G-seeds: duplicate seeds in {role}")
    if smoke:
        allowed = set(SMOKE_SEEDS)
    elif role == "D":
        allowed = set(fp.COHORT_D)
    else:
        allowed = set(fp.seeds_of(role))
    outside = sorted(set(seeds) - allowed)
    if outside:
        raise fp.GuardError(f"G-seeds: {outside[:4]} are not {role} seeds")


def check_look_states(records: list[dict], reference=None, tolerance: float = 1e-6):
    """G-look (b): every attempt's post-look joint state equals the run's first attempt's.

    Returns the reference (the first state seen when none is given)."""
    states = [np.asarray(r["post_look_state"]) for r in records if "post_look_state" in r]
    if not states:
        return reference
    reference = states[0] if reference is None else np.asarray(reference)
    if max(float(np.abs(s - reference).max()) for s in states) > tolerance:
        raise fp.GuardError("G-look: the post-look joint state differs from the first attempt's")
    return reference


def check_privileged(records: list[dict], arm: str) -> None:
    """G-privileged (1)+(2) for an L1 or L2 arm's evaluation attempts."""
    for record in records:
        if not rt.privileged_reads_ok(record):
            raise fp.GuardError(f"G-privileged: {arm} seed {record['seed']} read task truth")


# ----- the worker -----
_W: dict = {}


def worker_init() -> None:
    import torch

    torch.set_num_threads(fp.WORKER_TORCH_THREADS)
    _W["robot"] = rt.make_robot()
    _W["fk"] = rt.PalmFK()
    _W["bounds"] = rt.configured_bounds()
    _W["models"] = {}


def _learned(spec: dict):
    from embodied_jepa import first_policy_model as fm

    key = spec["checkpoint"]
    if key not in _W["models"]:
        import torch

        saved = torch.load(key, map_location="cpu", weights_only=True)
        model = fm.load(saved["state"], heads=int(saved["heads"]))
        _W["models"][key] = (model, rt.Standardiser(saved["mean"].numpy(), saved["std"].numpy()))
    model, standardiser = _W["models"][key]
    return rt.LearnedController(
        fm.predictor(model), standardiser, spec["estimates"], _W["fk"], _W["bounds"]
    )


def run_task(task: dict) -> dict:
    """One attempt in a worker. ``task["kind"]`` names the controller; see ``controller_for``."""
    from embodied_jepa.policy_diagnostics import RandomController, ShadowExpert, hold_controller

    robot = _W["robot"]
    truth, scorer, counter, observation, facts = rt.reset_and_look(
        robot, task["seed"], task["reset"]
    )
    out = {
        "seed": task["seed"],
        "kind": task["kind"],
        "post_look_frame_sha256": frame_sha(facts["post_look_frame"]),
        "post_look_state": facts["post_look_state"].tolist(),
    }
    if task["kind"] == "frame":
        counter.remove()
        robot.stop("frame_only")
        out["frame"] = facts["post_look_frame"]
        out["truth_xy"] = [*truth["object_position"][:2], *truth["plate_position"][:2]]
        return out
    expected = task.get("expected_frame_sha256")
    if expected is not None and out["post_look_frame_sha256"] != expected:
        counter.remove()
        robot.stop("frame_mismatch")
        raise fp.GuardError(f"S0-D1/G-frame: seed {task['seed']} post-look frame differs")
    kind = task["kind"]
    if kind == "learned":
        controller = _learned(task)
    elif kind == "a4":
        controller = rt.scripted_controller(rt.a4_truth(task["estimates"], _W["fk"]), robot)
    elif kind == "oracle":
        controller = rt.scripted_controller(truth, robot)
    elif kind == "c0":
        controller = rt.scripted_controller(
            rt.perturbed_truth(truth, task["apple_offset"], task["plate_offset"]), robot
        )
    elif kind == "hold":
        controller = rt.HarnessController(hold_controller())
    elif kind == "random":
        lower, upper = _W["bounds"]
        controller = rt.HarnessController(RandomController(fp.RANDOM_CONTROLLER_SEED, lower, upper))
    elif kind == "replay":
        controller = rt.ReplayController(task["actions"])
    else:
        counter.remove()
        raise fp.GuardError(f"unknown controller kind {kind!r}")
    labeller = ShadowExpert(truth) if task.get("label") else None
    record = rt.run_policy_steps(
        robot,
        scorer,
        counter,
        controller,
        bounds=_W["bounds"],
        max_steps=int(task.get("max_steps", fp.MAX_POLICY_STEPS)),
        labeller=labeller,
    )
    if kind == "learned" and task.get("first_input_check"):
        out["first_input"] = (
            None if controller.last_input is None else controller.last_input.tolist()
        )
    if record["states"].size:
        out["palm9"] = np.stack([_W["fk"].pose9(s) for s in record["states"]]).tolist()
    out.update({k: v for k, v in record.items() if k not in ("commands",)})
    out["commands"] = record["commands"].tolist()
    for key in ("states", "steps", "labels"):
        out[key] = np.asarray(record[key]).tolist()
    return out


class Pool:
    def __init__(self, workers: int):
        self.pool = mp.get_context("spawn").Pool(workers, initializer=worker_init)

    look_reference = None

    def map(self, tasks: list[dict], cap: float, what: str) -> list[dict]:
        out = self._map(tasks, cap, what)
        self.look_reference = check_look_states(out, self.look_reference)  # every attempt
        return out

    def _map(self, tasks: list[dict], cap: float, what: str) -> list[dict]:
        started = time.monotonic()
        result = self.pool.map_async(run_task, tasks, chunksize=1)
        remaining = cap - (time.monotonic() - started)
        try:
            out = result.get(timeout=max(remaining, 1.0))
        except mp.TimeoutError as error:
            raise fp.GuardError(f"G-cap: {what} exceeded its {cap} s cap") from error
        return out

    def close(self):
        self.pool.terminate()
        self.pool.join()


# ----- data -----
def corpus_roots(store, smoke):
    """Train and val roots of apple-look-v1 with their metadata, sorted by seed."""
    rows = {r["episode_id"]: r for r in store.manifest["episodes"]}
    out = {}
    for split in fp.READ_SPLITS:
        ids = [e for e in store.manifest["splits"][split] if rows[e]["metadata"]["kind"] == "root"]
        ids.sort(key=lambda e: rows[e]["metadata"]["reset_seed"])
        if smoke:
            ids = ids[: SMOKE["train_roots"] if split == "train" else SMOKE["val_roots"]]
        out[split] = [rows[e] for e in ids]
    return out


def read_root(reader, row) -> dict:
    """Frames 8 (post-look), states, executed actions and labels of one train/val root."""
    import pyarrow  # noqa: F401  (the reader imports it after its guards)

    episode_id = row["episode_id"]
    frames, actions, valid = reader.episode(episode_id, frames=True)
    table, _ = reader._table(episode_id, ["observation.state"])
    states = np.asarray(table["observation.state"].to_pylist(), np.float64)
    labels = reader.labels(episode_id)
    return {
        "episode_id": episode_id,
        "seed": int(row["metadata"]["reset_seed"]),
        "aim": bool(row["metadata"]["aim_offset_applied"]),
        "success": bool(row["metadata"]["privileged_outcome_labels"]["stages"]["success"]),
        "frame8": frames[fp.DECISION_FRAME],
        "actions": actions,
        "valid": valid,
        "states": states,
        "labels": labels,
        "xy": [
            *labels["privileged__apple_position_world"][0, :2],
            *labels["privileged__plate_position_world"][0, :2],
        ],
    }


def bc_rows(root: dict, bounds) -> dict:
    """BC-0 rows of one non-aim root (protocol §3): policy steps, masked, clipped labels."""
    lower, upper = bounds
    free = list(fp.FREE_INDICES)
    labels = root["labels"]
    phase = np.asarray(labels["collector__phase_index"])
    base = np.asarray(labels["collector__base_action"], np.float64)
    apple = np.asarray(labels["privileged__apple_position_world"], np.float64)
    dropped = np.asarray(labels["privileged__apple_dropped"], bool)
    keep, steps = [], []
    accounting = {"policy_steps": 0, "dropped": 0, "post_displacement_pre_grasp": 0, "kept": 0}
    for t in range(len(phase)):
        if phase[t] < 0 or not root["valid"][t]:
            continue
        accounting["policy_steps"] += 1
        # TASK-056's mask (cloning.sample_mask, claim audit S5-18): the apple's 3-D drift from
        # its first frame exceeds 1 cm while the collector's phase is before close, or dropped.
        drift = float(np.linalg.norm(apple[t] - apple[0]))
        displaced = drift > DISPLACEMENT_LIMIT_M and phase[t] < PHASE_CLOSE
        if dropped[t]:
            accounting["dropped"] += 1
            continue
        if displaced:
            accounting["post_displacement_pre_grasp"] += 1
            continue
        keep.append(t)
        steps.append(t - fp.DECISION_FRAME)
    keep = np.asarray(keep, int)
    accounting["kept"] = len(keep)
    return {
        "states": root["states"][keep],
        "steps": np.asarray(steps, np.int64),
        "labels": np.clip(base[keep][:, free], lower[free], upper[free]).astype(np.float32),
        "accounting": accounting,
    }


def inputs_of(estimates_per_row, steps, states, palm9) -> np.ndarray:
    return np.stack(
        [
            rt.input_vector(e, int(s), x, p)
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
    return {
        "grasp": sum(bool(r["grasp"]) for r in records),
        "success": sum(bool(r["success"]) for r in records),
    }


def attempt_summary(records) -> list[dict]:
    keep = (
        "seed",
        "termination_reason",
        "executed_steps",
        "grasp",
        "success",
        "final_score",
        "task_truth_total",
        "task_truth_in_controller",
        "scorer_evaluations",
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


def end_checks(report, manifest, smoke) -> None:
    """G-hash after the last stage, on every row (early stops included)."""
    report["pinned_hashes_at_end"] = R65.check_pins(manifest["hashes"])
    if not smoke and R65.tracked_tree_dirty():
        raise fp.GuardError("G-hash: the tracked tree changed during the run")


# ----- the run -----
def run(output: Path, checkpoints: Path, smoke: bool) -> dict:
    output, checkpoints = Path(output), Path(checkpoints)
    for path in (output, checkpoints):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    output.mkdir(parents=True)
    checkpoints.mkdir(parents=True)
    report = {
        "protocol": fp.PROTOCOL,
        "task": fp.TASK,
        "smoke": smoke,
        "outcome": None,
        "stages": {},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    clock = R65.Clock(fp.CAPS_SECONDS["global"])
    pool, reader = None, None
    try:
        reader, pool = _run(report, output, checkpoints, clock, smoke)
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
            report["test_split_decoded"] = bool(reader.decoded - reader.allowed)
        report["total_seconds"] = clock.elapsed()
        R65.write_report(output / "report.json", report)
        log(f"report written: outcome {report.get('outcome')}")
    return report


def _run(report, output, checkpoints, clock, smoke):  # noqa: C901 - one linear protocol
    import torch

    from embodied_jepa import first_policy_model as fm
    from embodied_jepa import first_policy_perception as fpp
    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa.data import DatasetStore

    # ----- 1. preflight -----
    manifest = json.loads(MANIFEST.read_text())
    if manifest["frozen"] != fp.frozen_block():
        raise fp.GuardError("G-frozen: the manifest's frozen block differs from the module")
    report["pinned_hashes_at_preflight"] = R65.check_pins(manifest["hashes"])
    dirty = R65.tracked_tree_dirty()
    report["revision"], report["tracked_tree_dirty"] = R65.revision(), bool(dirty)
    if not smoke:
        R65.check_clean(dirty)
    fp.check_seed_ranges()
    mps = bool(torch.backends.mps.is_available())
    device = "cpu" if smoke else fp.TRAIN_DEVICE
    if not smoke and not mps:
        raise fp.GuardError("G-device: MPS is not available")
    torch.set_num_threads(6)
    report["environment"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "mps_available": mps,
        "train_device": device,
    }
    store = DatasetStore(ROOT / fp.DATASET)  # verifies every recorded hash
    splits = store.manifest["splits"]
    rows = {r["episode_id"]: r for r in store.manifest["episodes"]}
    sessions = {
        k: len({rows[e]["session_id"] for e in splits[k]}) for k in ("train", "val", "test")
    }
    episodes = {k: len(splits[k]) for k in ("train", "val", "test")}
    R65.check_data(store.manifest_hash, sessions, episodes)
    reader = R65.TrainValReader(store, set(splits["train"]) | set(splits["val"]))
    report["_reader"] = reader
    pretrained = pe.load_pretrained()
    floor = pe.random_init()
    digests = {"pretrained": pe.weights_digest(pretrained), "floor": pe.weights_digest(floor)}
    report["encoder_digests"] = digests
    if not smoke and (
        digests["pretrained"] != manifest["encoder_digests"]["pretrained"]
        or digests["floor"] != manifest["encoder_digests"]["floor"]
    ):
        raise fp.GuardError("G-weights: an encoder digest differs from its pin")
    bounds = rt.configured_bounds()
    wide_reset = load_wide_reset()
    workers = SMOKE["workers"] if smoke else fp.SIM_WORKERS
    pool = Pool(workers)
    report["_pool"] = pool
    report["stages"]["preflight"] = {"seconds": clock.elapsed(), "workers": workers}

    def seeds_for(role, n=None):
        if smoke:
            base = {
                "perception_train": 0,
                "perception_heldout": 10,
                "calibration_C0": 20,
                "dagger_1": 30,
                "dagger_2": 40,
                "dagger_3": 50,
                "D": 60,
            }[role]
            return tuple(SMOKE_SEEDS[base : base + SMOKE["per_role"]])
        return fp.COHORT_D if role == "D" else fp.seeds_of(role)

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

    max_steps = SMOKE["max_steps"] if smoke else fp.MAX_POLICY_STEPS

    # ----- 2. perception frames -----
    clock.check("perception")
    started = time.monotonic()
    frame_tasks = {
        role: tasks(role, "frame") for role in ("perception_train", "perception_heldout")
    }
    both = pool.map(
        frame_tasks["perception_train"] + frame_tasks["perception_heldout"],
        fp.CAPS_SECONDS["perception_collection"],
        "perception collection",
    )
    n_train = len(frame_tasks["perception_train"])
    frames = {"perception_train": both[:n_train], "perception_heldout": both[n_train:]}
    corpus = corpus_roots(store, smoke)
    roots = {split: [read_root(reader, row) for row in corpus[split]] for split in corpus}
    if not smoke:
        got = {k: len(v) for k, v in roots.items()}
        if got != fp.EXPECTED_ROOTS:
            raise fp.GuardError(f"G-data: roots {got} != {fp.EXPECTED_ROOTS}")
        non_aim = sum(not r["aim"] for r in roots["train"])
        if non_aim != 134:
            raise fp.GuardError(f"G-data: {non_aim} non-aim train roots, not 134")
    report["data"] = {
        "dataset_manifest_sha256": store.manifest_hash,
        "train_roots": [r["episode_id"] for r in roots["train"]],
        "val_roots": [r["episode_id"] for r in roots["val"]],
        "seeds": {
            role: list(map(int, seeds_for(role)))
            for role in (
                "perception_train",
                "perception_heldout",
                "calibration_C0",
                "dagger_1",
                "dagger_2",
                "dagger_3",
                "D",
            )
        },
    }
    report["stages"]["perception_frames"] = {
        "seconds": time.monotonic() - started,
        "train_roots": len(roots["train"]),
        "val_roots": len(roots["val"]),
    }

    def tokens(encoder, frame_list):
        return fpp.featurise(encoder, np.stack(frame_list))

    fit_frames = [r["frame8"] for r in roots["train"]] + [
        f["frame"] for f in frames["perception_train"]
    ]
    fit_xy = np.asarray(
        [r["xy"] for r in roots["train"]] + [f["truth_xy"] for f in frames["perception_train"]]
    )
    if smoke:
        fit_xy = fit_xy + np.random.default_rng(0).normal(0, 0.01, fit_xy.shape)  # noise targets
    held_frames = [f["frame"] for f in frames["perception_heldout"]]
    held_xy = np.asarray([f["truth_xy"] for f in frames["perception_heldout"]])
    val_frames = [r["frame8"] for r in roots["val"]]
    feats = {}
    for name, encoder in (("P", pretrained), ("R", floor)):
        feats[name] = {
            "fit": tokens(encoder, fit_frames),
            "held": tokens(encoder, held_frames),
            "val": tokens(encoder, val_frames),
        }

    # ----- 3. readouts -----
    clock.check("readouts")
    readouts, cross = {}, {}
    for name in ("P", "R"):
        readouts[name] = fpp.XYReadout(feats[name]["fit"], fit_xy)
        cross[name], _sel = fpp.cross_fitted(feats[name]["fit"], fit_xy)
    held_est = readouts["P"].predict(feats["P"]["held"])
    apple_err, plate_err = fpp.errors_cm(held_est, held_xy)
    report["stages"]["readouts"] = {
        name: {"selection": readouts[name].selection} for name in readouts
    }

    # ----- 4. C0 -----
    clock.check("C0")
    started = time.monotonic()
    c0_seeds = seeds_for("calibration_C0")
    conditions = [("reference", 0.0, 0.0)]
    conditions += [(f"apple_{a}", a, 0.0) for a in fp.C0_APPLE_LEVELS_CM]
    conditions += [(f"plate_{p}", 0.0, p) for p in fp.C0_PLATE_LEVELS_CM]
    c0_tasks = []
    for name, a_cm, p_cm in conditions:
        for i, task in enumerate(tasks("calibration_C0", "c0", max_steps=max_steps)):
            a_off, p_off = rt.c0_offsets(i, a_cm, p_cm)
            c0_tasks.append(
                task
                | {
                    "condition": name,
                    "apple_offset": a_off.tolist(),
                    "plate_offset": p_off.tolist(),
                }
            )
    c0_records = pool.map(c0_tasks, fp.CAPS_SECONDS["c0"], "C0")
    c0 = {}
    for task, record in zip(c0_tasks, c0_records, strict=True):
        c0[task["condition"]] = c0.get(task["condition"], 0) + int(record["success"])
    if smoke:
        c0 = {k: round(v * fp.C0_RESETS / max(len(c0_seeds), 1)) for k, v in c0.items()}
    apple_c0 = {a: c0[f"apple_{a}"] for a in fp.C0_APPLE_LEVELS_CM}
    plate_c0 = {p: c0[f"plate_{p}"] for p in fp.C0_PLATE_LEVELS_CM}
    bars = fp.c0_bars(c0["reference"], apple_c0, plate_c0)
    report["stages"]["C0"] = {
        "successes_of_32": c0,
        "bars": bars,
        "seconds": time.monotonic() - started,
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
            "plate_p90_cm": 2.5,
        }

    # ----- 5. S0-P, T, S0-D1 ----------------------------------------------------------------------
    s0 = (
        fp.s0_perception(apple_err, plate_err, bars)
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
    reference = c0["reference"] if "reference" in c0 else fp.C0_RESETS
    threshold = fp.a4_threshold(max(reference, 1), apple_c0, plate_c0, apple_err, plate_err)
    report["stages"]["a4_threshold"] = threshold

    # ----- 6. training data, BC-0 --------------------------------------------------------------
    clock.check("BC-0")
    fk = rt.PalmFK()
    n_corpus = len(roots["train"])
    bc_roots = [i for i, r in enumerate(roots["train"]) if not r["aim"]]
    val_roots = [i for i, r in enumerate(roots["val"]) if not r["aim"]]
    val_est = {name: readouts[name].predict(feats[name]["val"]) for name in ("P", "R")}
    c_mean = cross["P"][:n_corpus][bc_roots].mean(axis=0)
    data = {
        arm: {"states": [], "steps": [], "labels": [], "est": [], "palm9": []}
        for arm in ("P", "C", "R")
    }
    val = {
        arm: {"states": [], "steps": [], "labels": [], "est": [], "palm9": []}
        for arm in ("P", "C", "R")
    }
    mask_accounting = {"train": {}, "val": {}}
    for split, index, target in (("train", bc_roots, data), ("val", val_roots, val)):
        for i in index:
            rows_i = bc_rows(roots[split][i], bounds)
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
        standardisers[arm] = rt.Standardiser.fit(x)
    report["stages"]["BC0"] = {
        "rows": len(data["P"]["steps"]),
        "val_rows": len(val["P"]["steps"]),
        "roots": len(bc_roots),
        "val_roots": len(val_roots),
        "C_mean_estimates": c_mean.tolist(),
        "C_mean_definition": "mean over the BC-0 roots of the cross-fitted P estimates (per root)",
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
        trained = fm.train(
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
        raise fp.GuardError("S0-D1: no family has an eligible BC-0 checkpoint to check")
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
            }
            for i, t in enumerate(held_tasks)
        ],
        fp.CAPS_SECONDS["per_rollout_batch"],
        "S0-D1",
    )
    saved = torch.load(ckpt[d1_arm], map_location="cpu", weights_only=True)
    model = fm.load(saved["state"])
    s = rt.Standardiser(saved["mean"].numpy(), saved["std"].numpy())
    # The offline rows are assembled in the main process from the training path (estimate,
    # step 0, the recorded post-look state, FK), independently of the worker's live input.
    offline_inputs = np.stack(
        [
            rt.input_vector(
                d1_est[i],
                0,
                frames["perception_heldout"][i]["post_look_state"],
                fk.pose9(frames["perception_heldout"][i]["post_look_state"]),
            )
            for i in range(len(d1))
        ]
    )
    offline = fm.batch_predict(model, s(offline_inputs), np.zeros(len(d1), int))  # one batch
    input_worst = act_worst = 0.0
    for i, r in enumerate(d1):
        live_input = np.asarray(r["first_input"], np.float32)
        input_worst = max(input_worst, float(np.abs(live_input - offline_inputs[i]).max()))
        expected = rt.free_of(rt.assemble(offline[i], *bounds))
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
        raise fp.GuardError(
            f"S0-D1: live differs from offline (input {input_worst}, act {act_worst})"
        )

    # ----- DAgger -----
    dagger_report = {}
    for k in range(1, fp.DAGGER_ITERATIONS + 1):
        role = f"dagger_{k}"
        clock.check(role)
        frame_records = pool.map(
            tasks(role, "frame"), fp.CAPS_SECONDS["per_rollout_batch"], f"{role} frames"
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
                # No eligible checkpoint: this family's later iterations score 0/16 as well
                # (protocol section 5.2, amendment 1 item 6).
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
            records = pool.map(arm_tasks, fp.CAPS_SECONDS["per_rollout_batch"], f"{role} {arm}")
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
    report["stages"]["dagger"] = dagger_report
    report["stages"]["trainings"] = trainings

    # ----- 7. M1 on D -----
    clock.check("M1")
    d_frames = pool.map(tasks("D", "frame"), fp.CAPS_SECONDS["per_rollout_batch"], "D frames")
    d_frame_list = [f["frame"] for f in d_frames]
    d_est = {
        "P": readouts["P"].predict(tokens(pretrained, d_frame_list)),
        "R": readouts["R"].predict(tokens(floor, d_frame_list)),
    }
    d_truth = np.asarray([f["truth_xy"] for f in d_frames])
    library = [r for r in roots["train"] if r["success"] and not r["aim"]]
    if not library and smoke:
        library = roots["train"]  # smoke only: a 4-root subset may hold no success
    if not library:
        raise fp.GuardError("B-replay: the retrieval library is empty")
    lib_cls = fpp.featurise_cls(pretrained, [r["frame8"] for r in library])
    d_cls = fpp.featurise_cls(pretrained, d_frame_list)
    mu, sd = lib_cls.mean(axis=0), np.maximum(lib_cls.std(axis=0), fp.INPUT_STD_FLOOR)
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
            fp.CAPS_SECONDS["per_rollout_batch"],
            f"M1 {name}",
        )
        check_privileged(arms[name], name)
    arms["A4-look"] = pool.map(
        d_tasks("a4", lambda i: {"estimates": d_est["P"][i].tolist()}),
        fp.CAPS_SECONDS["per_rollout_batch"],
        "M1 A4-look",
    )
    for name, kind in (("B-oracle", "oracle"), ("B-hold", "hold"), ("B-random", "random")):
        arms[name] = pool.map(d_tasks(kind), fp.CAPS_SECONDS["per_rollout_batch"], f"M1 {name}")
    arms["B-replay"] = pool.map(
        d_tasks(
            "replay",
            lambda i: {"actions": library[nearest[i]]["actions"][fp.DECISION_FRAME : -1].tolist()},
        ),
        fp.CAPS_SECONDS["per_rollout_batch"],
        "M1 B-replay",
    )
    counts = {
        name: (count(r) if r is not None else {"grasp": 0, "success": 0})
        for name, r in arms.items()
    }
    decision = fp.decide_m1(counts | {"D-oracle-perc": {"grasp": 0, "success": 0}}, threshold)
    carried = decision.get("carried", "P-3")
    if ckpt.get(carried):
        arms["D-oracle-perc"] = pool.map(
            d_tasks(
                "learned", lambda i: {"checkpoint": ckpt[carried], "estimates": d_truth[i].tolist()}
            ),
            fp.CAPS_SECONDS["per_rollout_batch"],
            "M1 D-oracle-perc",
        )
    counts["D-oracle-perc"] = count(arms.get("D-oracle-perc") or [])
    decision = fp.decide_m1(counts, threshold)
    f_counts = None
    if decision["row"] == "M1-MOTOR":
        # F: per-phase heads on P-3's final aggregate, evaluated once on D.
        ckpt["F"] = train_arm("P", "F", heads=len(fp.PHASES))
        if ckpt["F"] is None:
            f_counts = {"grasp": 0, "success": 0}
        else:
            arms["F"] = pool.map(
                d_tasks(
                    "learned",
                    lambda i: {"checkpoint": ckpt["F"], "estimates": d_est["P"][i].tolist()},
                ),
                fp.CAPS_SECONDS["per_rollout_batch"],
                "M1 F",
            )
            check_privileged(arms["F"], "F")
            f_counts = count(arms["F"])
        decision = fp.decide_m1(counts, threshold, f_counts=f_counts)
    report["stages"]["M1"] = {
        "counts": counts,
        "f_counts": f_counts,
        "a4_threshold": threshold,
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
    args = parser.parse_args()
    report = run(args.output, args.checkpoints, smoke=args.mode == "smoke")
    return 0 if report.get("outcome") not in (None, "V") else 1


if __name__ == "__main__":
    sys.exit(main())
