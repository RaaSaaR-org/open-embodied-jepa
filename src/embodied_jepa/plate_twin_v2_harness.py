"""TASK-076 harness pieces that the pinned runner chain keeps in script files.

New runners may not load another script (``tests/test_no_runner_imports.py``, audit F16), and the
pinned runners (``run_wm_critic_v2.py``, ``run_lewm_planner_v2.py``, ``run_first_policy_v2_m2.py``,
``train_apple_latent_dynamics.py``) stay as they are. The pieces TASK-076's runner needs are
ported here, behaviour unchanged, and the guards come from ``run_tools`` (not copied):

* report I/O and provenance (``train_apple_latent_dynamics``'s ``write_report``, ``check_pins``,
  ``revision``, ``tracked_tree_dirty``, ``Clock``);
* G-evidence for TASK-072 run-1 and G-repro, the exact refit of P-3's post-look readout
  (``run_first_policy_v2_m2.check_evidence`` / ``check_reproduction`` and
  ``run_wm_critic_v2.refit_p_readout`` / ``render_majority`` / ``cohort_estimates``); the
  wide-jitter reset is ``wm_critic_v2.wide_reset_values``, which a test pins equal to
  ``scripts/evaluate_apple.wide_reset``;
* the bounded pool with TASK-071's G-look check, and the per-attempt report rows.

NumPy at import; MuJoCo and torch are reached only through the pool's workers and the refit.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from embodied_jepa import first_policy_v2 as fp2
from embodied_jepa import first_policy_v2_m2 as fm
from embodied_jepa import lewm_planner_v2 as lp
from embodied_jepa import run_guards as rg
from embodied_jepa import wm_critic_v2 as wc

ROOT = Path(__file__).resolve().parents[2]
GIB = 2**30


# ----- report I/O and provenance ------------------------------------------------------------------
def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_json(value):
    """Every non-finite float becomes None and is listed (a report is always writable)."""
    found = []

    def clean(item, path):
        if isinstance(item, dict):
            return {str(k): clean(v, f"{path}/{k}") for k, v in item.items()}
        if isinstance(item, list | tuple):
            return [clean(v, f"{path}/{i}") for i, v in enumerate(item)]
        if isinstance(item, np.ndarray):
            return clean(item.tolist(), path)
        if isinstance(item, np.bool_):
            return bool(item)
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, float | np.floating):
            if not np.isfinite(item):
                found.append(f"{path}={float(item)!r}")
                return None
            return float(item)
        return item

    return clean(value, ""), found


def write_report(path: Path, report: dict) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    clean, found = finite_json(report)
    clean["non_finite_fields"] = found
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(clean, indent=1, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class Clock:
    def __init__(self, cap: float):
        self.start, self.cap = time.monotonic(), float(cap)

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def check(self, stage: str) -> None:
        if self.elapsed() > self.cap:
            raise lp.GuardError(
                f"G-cap: the invocation cap {self.cap} s was reached before {stage}"
            )


def check_pins(pinned: dict, root: Path = ROOT) -> dict:
    """G-hash: every pinned file's sha256; a missing or different file is V."""
    found = {}
    for relative, want in sorted(pinned.items()):
        path = Path(root) / relative
        if not path.is_file():
            raise lp.GuardError(f"G-hash: pinned file missing: {relative}")
        got = sha256_file(path)
        if got != want:
            raise lp.GuardError(f"G-hash: {relative} sha256 {got[:12]} != pinned {want[:12]}")
        found[relative] = got
    return found


def tracked_tree_dirty(root: Path = ROOT) -> list[str]:
    out = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    return [line for line in out.splitlines() if line.strip()]


def revision(root: Path = ROOT) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def check_clean(dirty: list[str]) -> None:
    if dirty:
        raise lp.GuardError(f"G-hash: the tracked tree is dirty: {dirty[:5]}")


def mem_available_bytes() -> int:
    with open("/proc/meminfo") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise lp.GuardError("G-memory: /proc/meminfo has no MemAvailable")


def gpu_snapshot() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


# ----- G-evidence and G-repro (TASK-072 run-1) ---------------------------------------------------
def check_evidence(evidence: Path) -> dict:
    """TASK-072 run-1's report, carried checkpoints and corpus manifest, by sha256."""
    e = fm.EVIDENCE
    evidence = Path(evidence)
    found = {"report": sha256_file(evidence / e["report"])}
    if found["report"] != e["report_sha256"]:
        raise fp2.GuardError("G-evidence: run-1's report.json differs from its pin")
    for arm, want in e["checkpoints"].items():
        found[arm] = sha256_file(evidence / e["checkpoint_dir"] / f"{arm}.pt")
        if found[arm] != want:
            raise fp2.GuardError(f"G-evidence: {arm}.pt differs from its pin")
    found["corpus_manifest"] = sha256_file(evidence / e["corpus"] / "manifest.json")
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


def p3_checkpoint(evidence: Path) -> str:
    return str(Path(evidence) / lp.EVIDENCE["checkpoint_dir"] / f"{lp.CARRIED}.pt")


def check_rerender_seeds(role: str, seeds) -> tuple[int, ...]:
    """G-seeds for G-repro's frame-only re-renders: exactly run-1's seeds; never cohort C."""
    seeds = tuple(seeds)
    if any(type(s) is not int for s in seeds):
        raise fp2.GuardError("G-seeds: seeds must be plain ints")
    if set(seeds) & set(fm.COHORT_C):
        raise fp2.GuardError("G-seeds: cohort C is not re-rendered")
    allowed = fp2.COHORT_D2 if role == "D" else fp2.seeds_of(role)
    if seeds != tuple(allowed):
        raise fp2.GuardError(f"G-seeds: {role} must be run-1's {role} seeds, in order")
    return seeds


def json_floats(values) -> list:
    """Floats as run-1's report stored them (the JSON round trip), for exact comparison."""
    return json.loads(json.dumps(np.asarray(values, np.float64).tolist()))


def check_reproduction(recomputed: dict, run1: dict) -> dict:
    """G-repro: every recorded readout fact of run-1 is reproduced exactly."""
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


def check_look_states(records: list[dict], reference=None, tolerance: float = 1e-6):
    """G-look (b): every attempt's post-look joint state equals the run's first attempt's."""
    states = [np.asarray(r["post_look_state"]) for r in records if "post_look_state" in r]
    if not states:
        return reference
    reference = states[0] if reference is None else np.asarray(reference)
    if max(float(np.abs(s - reference).max()) for s in states) > tolerance:
        raise fp2.GuardError("G-look: the post-look joint state differs from the first attempt's")
    return reference


class Pool(rg.BoundedPool):
    """``run_guards.BoundedPool`` on TASK-076's worker; a dead worker or a cap is the protocol's
    GuardError; every map runs TASK-071's G-look check."""

    def __init__(self, workers: int, config: dict):
        from embodied_jepa import plate_twin_v2_runtime as ptr

        super().__init__(workers, ptr.run_task, initializer=ptr.worker_init, initargs=(config,))
        self.look_reference = None

    def map(self, tasks: list[dict], cap: float, what: str) -> list[dict]:
        try:
            out = super().map(tasks, cap, what)
        except (rg.WorkerDied, rg.MapCapExceeded) as error:
            raise lp.GuardError(str(error)) from error
        self.look_reference = check_look_states(out, self.look_reference)
        return out


def render_majority(pool, tasks: list[dict], cap: float, what: str):
    """Every frame task rendered twice; where the two differ a third render decides by majority,
    allowed only when the simulation state is bitwise equal and the odd frame is within the
    renderer's characterised effect (owner ruling 2026-09-29); otherwise a V."""
    rendered = pool.map(tasks + tasks, cap, f"{what} (two renders each)")
    first, second = rendered[: len(tasks)], rendered[len(tasks) :]
    chosen, disagreements = [], {}
    for task, a, b in zip(tasks, first, second, strict=True):
        if a["post_look_frame_sha256"] == b["post_look_frame_sha256"]:
            if a["state_sha256"] != b["state_sha256"]:
                raise wc.GuardError(f"G-frame: seed {task['seed']} same frame, different state")
            chosen.append(a)
            continue
        c = pool.map([task], cap, f"{what} (third render)")[0]
        trio = (a, b, c)
        shas = [f["post_look_frame_sha256"] for f in trio]
        states = {f["state_sha256"] for f in trio}
        winner = next((f for f in trio if shas.count(f["post_look_frame_sha256"]) >= 2), None)
        record = {"frames": shas, "states_equal": len(states) == 1}
        if winner is not None:
            record["odd"] = [
                wc.frame_difference(f["frame"], winner["frame"])
                for f in trio
                if f["post_look_frame_sha256"] != winner["post_look_frame_sha256"]
            ]
        disagreements[str(task["seed"])] = record
        if len(states) != 1:
            raise wc.GuardError(f"G-frame: seed {task['seed']} renders disagree, state differs")
        if winner is None:
            raise wc.GuardError(f"G-frame: seed {task['seed']} has no majority: {record}")
        if not all(wc.render_difference_allowed(d) for d in record["odd"]):
            raise wc.GuardError(f"G-frame: seed {task['seed']} odd render beyond the effect")
        chosen.append(winner)
    return chosen, disagreements


def cohort_estimates(pool, readout, encoder, seeds, resets, cap: float) -> dict:
    """Post-look frames of a cohort and P-3's post-look estimates from them (P's readout).
    ``readout`` None is the smoke's stand-in: the estimates are the reset truth (smokes only)."""
    tasks = [
        {"kind": "frame", "seed": s, "reset": {k: resets[s][k] for k in ("object_xy", "plate_xy")}}
        for s in seeds
    ]
    frames, disagreements = render_majority(pool, tasks, cap, "cohort frames")
    for s, f in zip(seeds, frames, strict=True):
        if f["seed"] != s:
            raise wc.GuardError("G-cohort: a rendered frame is not its seed's")
    if readout is None:
        estimates = np.asarray([f["truth_xy"] for f in frames], np.float64)
    else:
        from embodied_jepa import first_policy_perception as fpp

        tokens = np.concatenate([fpp.featurise(encoder, f["frame"][None]) for f in frames])
        estimates = readout.predict(tokens)
    out = {
        s: {
            "estimates": np.asarray(estimates[i], np.float64).tolist(),
            "frame_sha256": frames[i]["post_look_frame_sha256"],
            "state_sha256": frames[i]["state_sha256"],
            "truth_xy": frames[i]["truth_xy"],
        }
        for i, s in enumerate(seeds)
    }
    return out | {"_render_disagreements": disagreements}


def refit_p_readout(report: dict, pool, evidence: Path, run1: dict):
    """G-repro: P-3's post-look readout refitted exactly as TASK-072 run-1 fitted it; every
    recorded readout fact of run-1 must be reproduced. Returns the P readout and the encoder."""
    from embodied_jepa import first_policy_perception as fpp
    from embodied_jepa import first_policy_v2_runtime as rt2
    from embodied_jepa import pretrained_encoder as pe

    started = time.monotonic()
    pretrained, floor = pe.load_pretrained(), pe.random_init()
    reader = rt2.CorpusReader(
        Path(evidence) / fm.EVIDENCE["corpus"],
        fm.EVIDENCE["corpus_manifest_sha256"],
        splits=("train",),
    )
    train_ids = list(reader.manifest["splits"]["train"])
    if train_ids != run1["data"]["train_roots"]:
        raise wc.GuardError("G-data: the corpus's train roots are not run-1's")
    roots = []
    for episode_id in train_ids:
        arrays, meta = reader.episode(episode_id, keys=("phase",))
        roots.append(
            {
                "episode_id": episode_id,
                "success": bool(meta["success"]),
                "frame0": arrays["frame0"],
                "xy": meta["truth_xy"],
            }
        )
    frame_tasks = []
    for role in ("perception_train", "perception_heldout", "D"):
        for seed in check_rerender_seeds(role, run1["data"]["seeds"][role]):
            r = wc.wide_reset_values(seed)
            reset = {
                "object_xy": list(map(float, r["object_xy"])),
                "plate_xy": list(map(float, r["plate_xy"])),
            }
            frame_tasks.append({"kind": "frame", "seed": seed, "reset": reset, "role": role})
    rendered, disagreements = render_majority(pool, frame_tasks, 1800.0, "re-render")
    report["stages"]["rerender_disagreements"] = disagreements
    frames = {role: [] for role in ("perception_train", "perception_heldout", "D")}
    for task, out in zip(frame_tasks, rendered, strict=True):
        frames[task["role"]].append(out)
    fit_frames = [r["frame0"] for r in roots] + [f["frame"] for f in frames["perception_train"]]
    fit_xy = np.asarray(
        [r["xy"] for r in roots] + [f["truth_xy"] for f in frames["perception_train"]]
    )
    readouts, feats = {}, {}
    for name, encoder in (("P", pretrained), ("R", floor)):
        feats[name] = fpp.featurise(encoder, np.stack(fit_frames))
        readouts[name] = fpp.XYReadout(feats[name], fit_xy)
    cross_p, _ = fpp.cross_fitted(feats["P"], fit_xy)
    c_mean = cross_p[: len(roots)].mean(axis=0)
    held = [f["frame"] for f in frames["perception_heldout"]]
    held_xy = np.asarray([f["truth_xy"] for f in frames["perception_heldout"]])
    apple_err, plate_err = fpp.errors_cm(
        readouts["P"].predict(fpp.featurise(pretrained, np.stack(held))), held_xy
    )
    library = [r for r in roots if r["success"]]
    lib_cls = fpp.featurise_cls(pretrained, [r["frame0"] for r in library])
    mu, sd = lib_cls.mean(axis=0), np.maximum(lib_cls.std(axis=0), fp2.INPUT_STD_FLOOR)
    cls = fpp.featurise_cls(pretrained, [f["frame"] for f in frames["D"]])
    d2_nearest = [
        int(np.argmin(np.linalg.norm((lib_cls - mu) / sd - (c - mu) / sd, axis=1))) for c in cls
    ]
    recomputed = {
        "selection": {n: json.loads(json.dumps(readouts[n].selection)) for n in readouts},
        "fit_rows": len(fit_xy),
        "apple_errors_cm": json_floats(apple_err),
        "plate_errors_cm": json_floats(plate_err),
        "c_mean": json_floats(c_mean),
        "library": [r["episode_id"] for r in library],
        "d2_nearest": [library[j]["episode_id"] for j in d2_nearest],
    }
    report["stages"]["reproduction"] = {
        "checks": check_reproduction(recomputed, run1),
        "seconds": time.monotonic() - started,
        "decoded_train_roots": len(reader.decoded),
        "test_split_decoded": reader.test_split_decoded,
    }
    return readouts["P"], pretrained


# ----- the smoke source corpus (TASK-074's format) -----------------------------------------------
def seal_far_corpus(corpus: Path, plan: list[dict], episodes: dict, provenance: dict) -> str:
    """An ``apple-far-shift-v2``-format manifest (TASK-074's ``seal``) for a smoke corpus."""
    from embodied_jepa import wm_critic_v2_runtime as rtm

    path = Path(corpus) / "manifest.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    splits = {k: [] for k in ("train", "val", "test")}
    for root in plan:
        splits[root["split"]].append(root["episode_id"])
    manifest = {
        "corpus": lp.CORPUS,
        "protocol": lp.PROTOCOL,
        "task": lp.TASK,
        "privileged_scripted_collector": True,
        "learned_control": False,
        "scene_version": lp.SCENE_VERSION,
        "expert": {"class": "resting_expert.RestingPlaceExpert", "kwargs": lp.EXPERT},
        "episode_arrays": list(rtm.EPISODE_ARRAYS),
        "plan": plan,
        "splits": splits,
        "episodes": episodes,
        "provenance": provenance,
    }
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return sha256_file(path)


# ----- per-attempt report rows -------------------------------------------------------------------
def strip(records: list[dict]) -> list[dict]:
    """Per-attempt facts for the report (commands hashed)."""
    out = []
    for r in records:
        item = {k: v for k, v in r.items() if k not in ("commands", "post_look_state")}
        if "commands" in r:
            item["commands_sha256"] = sha256_bytes(
                np.ascontiguousarray(np.asarray(r["commands"], np.float32)).tobytes()
            )
        out.append(item)
    return out
