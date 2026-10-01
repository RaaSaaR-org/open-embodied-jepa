"""TASK-075 runner (``apple_obs_ceiling_v2``): the place tolerance tau, the multi-view re-render
and the cross-fitted readout ceilings of each view.

Protocol ``docs/experiments/apple_obs_ceiling_v2.md``; design ``src/embodied_jepa/
obs_ceiling_v2.py``; workers ``obs_ceiling_v2_runtime.py``; offline pieces
``obs_ceiling_v2_offline.py``. The run guards are TASK-074's (``run_guards`` through
``scripts/run_lewm_planner_v2.py``, imported unmodified): the bounded pool, the PSS watch with its
RSS fallback, the signal-safe V report. This runner adds a disk watch and a GPU guard.

Modes (every stage writes ``<output>/report.json``, refuses to overwrite it, and is V on any
guard, crash or cap):

- ``preflight``: the guards, G-evidence and G-repro (TASK-072 run-1's readouts refitted).
- ``tau``: Stage 0, development: the place tolerance on seeds 55000-55031, before the freeze.
  Refused once ``TAU_MEASURED`` is frozen.
- ``source``: smoke only: a TASK-074-format smoke corpus on smoke seeds (TASK-074's collector),
  the stand-in for ``apple-far-shift-v2`` in the render and readouts smokes.
- ``render``: Stage 1: re-simulate the source corpus's train and val roots and render the new
  views (``apple-far-shift-v2-views``), checked against the stored roots.
- ``readouts``: Stage 2: G-repro-off, the per-view featurisation, the cross-fitted readouts, the
  statistics and the row. With ``--smoke --scale`` it also runs the full-scale memory probe.

**Learned Apple->Plate on the frozen benchmark is still 0 successes.** This is a measurement
study; it makes no control claim.
"""

from __future__ import annotations

import os

# Pin BLAS/MKL threading before NumPy or torch is imported (TASK-073's owner ruling, carried).
os.environ.update(
    {
        "MKL_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": "6",
        "MKL_NUM_THREADS": "6",
        "OPENBLAS_NUM_THREADS": "16",
    }
)

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import devices  # noqa: E402
from embodied_jepa import first_policy_v2_linux as fpl  # noqa: E402
from embodied_jepa import lewm_planner_v2 as lp  # noqa: E402
from embodied_jepa import obs_ceiling_v2 as oc  # noqa: E402
from embodied_jepa import obs_ceiling_v2_runtime as ort  # noqa: E402
from embodied_jepa import run_guards as rg  # noqa: E402

MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-obs-ceiling-v2.json"
fpl.configure_headless()  # MUJOCO_GL=egl before any MuJoCo import; workers inherit it
MODES = ("preflight", "tau", "source", "render", "readouts")
SMOKE_MODES = ("tau", "source", "render", "readouts")
GIB = 2**30


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L74 = _load("_task074_runner", "scripts/run_lewm_planner_v2.py")  # pinned; imported, unmodified
RUN, R65, LIN, M2R = L74.RUN, L74.R65, L74.LIN, L74.M2R
log = R65.log
WORKERS_OVERRIDE: dict = {}
SMOKE_TAU_SEEDS = (55090, 55093)
SMOKE_TAU_LEVELS = (0.0, 3.0)
SMOKE_SOURCE = (55050, 55065)  # 16 roots: 10 train, 4 val, 2 test
SMOKE_TAU_CM = 2.0  # a placeholder tau for the smoke's decision code path (nothing is read)


def sim_workers() -> int:
    return int(WORKERS_OVERRIDE.get("sim", oc.SIM_WORKERS))


# ----- pool, memory and disk, signals ---------------------------------------------------------
class Pool(rg.BoundedPool):
    """``run_guards.BoundedPool`` on TASK-075's worker, with TASK-071's G-look check."""

    def __init__(self, workers: int, config: dict):
        super().__init__(workers, ort.run_task, initializer=ort.worker_init, initargs=(config,))
        self.look_reference = None

    def map(self, tasks: list[dict], cap: float, what: str) -> list[dict]:
        try:
            out = super().map(tasks, cap, what)
        except (rg.WorkerDied, rg.MapCapExceeded) as error:
            raise lp.GuardError(str(error)) from error
        self.look_reference = LIN.check_look_states(out, self.look_reference)
        return out


class Watch(L74.MemoryWatch):
    """TASK-074's PSS watch plus the disk watch (``oc.DISK``) on ``disk_path``."""

    def __init__(self, ceiling_bytes: int, interval: float):
        super().__init__(ceiling_bytes, interval)
        self.disk_path = None
        self.min_disk_free = None

    def run(self):
        while not self._stop.wait(self.interval):
            both = self.sample()
            if both[oc.MEMORY["measure_key"]] > self.ceiling and self.reason is None:
                self.reason = (
                    f"G-memory: process-tree PSS {both['pss'] / GIB:.2f} GiB > "
                    f"{self.ceiling / GIB:.2f} GiB"
                )
                os.kill(os.getpid(), signal.SIGUSR1)
            if self.disk_path is not None:
                free = shutil.disk_usage(self.disk_path).free
                self.min_disk_free = (
                    free if self.min_disk_free is None else min(self.min_disk_free, free)
                )
                if free < oc.DISK["min_free_gib"] * GIB and self.reason is None:
                    self.reason = f"G-disk: {free / GIB:.1f} GiB free < {oc.DISK['min_free_gib']}"
                    os.kill(os.getpid(), signal.SIGUSR1)

    def summary(self) -> dict:
        out = super().summary()
        if self.min_disk_free is not None:
            out["min_disk_free_gib"] = self.min_disk_free / GIB
        return out


def install_guards() -> Watch:
    watch = Watch(oc.MEMORY["ceiling_gib"] * GIB, oc.MEMORY["sample_seconds"])
    RUN.GUARD["stopping"] = False
    RUN.GUARD["late_signals"] = []

    def handler(signum, _frame):
        utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        name = signal.Signals(signum).name
        if RUN.GUARD["stopping"]:
            RUN.GUARD["late_signals"].append({"signal": name, "utc": utc})
            return
        RUN.GUARD["stopping"] = True
        reason = watch.reason if signum == signal.SIGUSR1 else f"received {name}"
        raise RUN.StageInterrupted(f"{reason} ({name})", utc)

    for s in (*RUN.STOP_SIGNALS, signal.SIGUSR1):
        signal.signal(s, handler)
    watch.start()
    return watch


def gpu_guard(report: dict) -> None:
    """``oc.GPU``: start only with enough free GPU memory; cap this process."""
    import torch

    free, total = torch.cuda.mem_get_info()
    need = (oc.GPU["min_free_gib_after"] + oc.GPU["process_cap_gib"]) * GIB
    report["gpu_free_gib_at_start"] = free / GIB
    report["gpu_total_gib"] = total / GIB
    if free < need:
        raise lp.GuardError(f"G-gpu: {free / GIB:.2f} GiB free < {need / GIB:.1f} GiB")
    torch.cuda.set_per_process_memory_fraction(oc.GPU["process_cap_gib"] * GIB / total)
    report["gpu_process_cap_gib"] = oc.GPU["process_cap_gib"]


def gpu_peak(report: dict) -> None:
    import torch

    if torch.cuda.is_initialized():
        report["gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / GIB
        report["gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / GIB


# ----- preflight ----------------------------------------------------------------------------------
def preflight(report: dict, mode: str, evidence: Path) -> dict:
    import mujoco
    import torch

    from embodied_jepa import pretrained_encoder as pe

    manifest = json.loads(MANIFEST.read_text())
    if manifest["frozen"] != oc.frozen_block():
        raise lp.GuardError("G-frozen: the manifest's frozen block differs from the module")
    if manifest["frozen_sha256"] != oc.frozen_sha256():
        raise lp.GuardError("G-frozen: the manifest's frozen_sha256 differs from the module")
    report["frozen_sha256"] = oc.frozen_sha256()
    report["pinned_hashes_at_preflight"] = R65.check_pins(manifest["hashes"])
    dirty = R65.tracked_tree_dirty()
    report["revision"], report["tracked_tree_dirty"] = R65.revision(), bool(dirty)
    if mode != "smoke":
        R65.check_clean(dirty)
    report["platform"] = fpl.check_platform()
    report["thread_env"] = {k: os.environ.get(k) for k in oc.THREAD_ENV}
    if report["thread_env"] != oc.THREAD_ENV:
        raise lp.GuardError(f"G-threads: {report['thread_env']} is not {oc.THREAD_ENV}")
    load = os.getloadavg()
    report["load_average_at_start"] = list(load)
    quiet = (
        load[0] <= oc.QUIET_MACHINE["max_load_average_1min"]
        and load[1] <= oc.QUIET_MACHINE["max_load_average_5min"]
    )
    report["quiet_machine"] = bool(quiet)
    L74.check_memory_available(report, mode)
    if not quiet and mode != "smoke":
        raise lp.GuardError(f"G-quiet: load averages {load[:2]} exceed the quiet-machine rule")
    if not devices.available("cuda"):
        raise lp.GuardError("G-device: CUDA is not available")
    device = devices.require("cuda", strict=True)
    torch.set_num_threads(6)
    if mujoco.__version__ != manifest["mujoco_version"]:
        raise lp.GuardError(f"G-hash: MuJoCo {mujoco.__version__} is not the pinned version")
    report["environment"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "accelerator": devices.accelerator_info(device),
        "determinism": devices.determinism_state(),
    }
    digests = {
        "pretrained": pe.weights_digest(pe.load_pretrained()),
        "floor": pe.weights_digest(pe.random_init()),
    }
    report["encoder_digests"] = digests
    if digests != manifest["encoder_digests"]:
        raise lp.GuardError("G-weights: an encoder digest differs from its pin")
    oc.check_seed_ranges()
    found = M2R.check_evidence(evidence)
    report["evidence"] = {"root": str(evidence), "sha256": found["sha256"]}
    report["_run1"] = found["run1"]
    report["gpu_at_start"] = RUN.gpu_snapshot()
    return manifest


def end_checks(report: dict, manifest: dict, mode: str) -> None:
    report["pinned_hashes_at_end"] = R65.check_pins(manifest["hashes"])
    fpl.check_determinism(report)
    report["revision_at_end"] = R65.revision()
    if mode != "smoke" and R65.tracked_tree_dirty():
        raise lp.GuardError("G-hash: the tracked tree changed during the run")


def read_stage_report(path, sha, outcome, smoke: bool, into: dict) -> dict:
    """G-evidence for an upstream TASK-075 report: sha256, protocol, outcome and kind, this
    frozen block and a clean tree (a real run)."""
    if R65.sha256_file(path) != sha:
        raise lp.GuardError(f"G-evidence: {path} differs from its sha256")
    upstream = json.loads(Path(path).read_text())
    if upstream.get("protocol") != oc.PROTOCOL:
        raise lp.GuardError(f"G-evidence: {path} is not a TASK-075 report")
    if upstream.get("outcome") != outcome or bool(upstream.get("smoke")) != bool(smoke):
        raise lp.GuardError(f"G-evidence: {path} is not a recorded {outcome} run of this kind")
    if upstream.get("frozen_sha256") != oc.frozen_sha256():
        raise lp.GuardError(f"G-evidence: {path} ran under another frozen block")
    if not smoke and upstream.get("tracked_tree_dirty") is not False:
        raise lp.GuardError(f"G-evidence: {path} did not run on a clean tree")
    into.setdefault("upstream", {})[outcome] = {
        "path": str(path),
        "sha256": sha,
        "revision": upstream.get("revision"),
    }
    return upstream


# ----- tau (Stage 0, development) -----------------------------------------------------------------
def stage_tau(report, manifest, evidence, clock, args):
    if args.smoke:
        seeds = oc.check_role_seeds(
            "tau", tuple(range(SMOKE_TAU_SEEDS[0], SMOKE_TAU_SEEDS[1] + 1)), smoke=True
        )
        levels = SMOKE_TAU_LEVELS
    else:
        if oc.TAU_MEASURED is not None:
            raise lp.GuardError("G-spent: the tau calibration has run and is frozen")
        seeds = oc.check_role_seeds("tau", oc.seeds_of("tau"))
        levels = oc.TAU["levels_cm"]
    resets = {s: lp.condition_reset(s) for s in seeds}
    shifts = {
        s: {
            "step": oc.CONDITION["shift_step"],
            "vector": lp.shift_vector(s, resets[s], oc.CONDITION["shift_cm"]),
        }
        for s in seeds
    }
    pool = Pool(sim_workers(), {"p3_checkpoint": L74.p3_checkpoint(evidence), "torch_threads": 1})
    report["_pool"] = pool
    readout, encoder = RUN.refit_p_readout(report, pool, evidence, report["_run1"])
    L74.mark_first_render(report)
    est = L74.estimates_for(report, pool, readout, encoder, seeds, resets)
    counts, at_rest, per_reset, planted, attempts = {}, {}, {}, {}, {}
    for cm in levels:
        clock.check(f"tau {cm}")
        tasks = RUN.arm_tasks(
            "H-handover",
            seeds,
            resets,
            est,
            lambda s: shifts[s],
            per_seed=lambda s, cm=cm: {
                "kind": "planted",
                "planted_m": oc.planted_error_m(s, cm),
                "level_cm": cm,
            },
        )
        records = pool.map(tasks, 7200.0, f"tau {cm} cm")
        checks = [
            ort.planted_check(r, resets[r["seed"]], shifts[r["seed"]])
            for r in records
            if r.get("termination_reason") != "shift_blocked"
        ]
        key = str(float(cm))
        counts[key] = sum(bool(r["success"]) for r in records)
        at_rest[key] = sum(bool(r["at_rest"]) for r in records)
        per_reset[key] = [bool(r["success"]) for r in records]
        planted[key] = {
            "max_abs_target_error_m": max(c["max_abs_target_error_m"] for c in checks),
            "fallback_decisions": sum(c["fallbacks"] for c in checks),
            "shift_blocked": sum(r.get("termination_reason") == "shift_blocked" for r in records),
        }
        attempts[key] = L74.strip(records)
        log(f"tau {cm} cm: {counts[key]}/{len(seeds)} counted, {at_rest[key]} at rest")
    result = {
        "outcome": "TAU-SMOKE" if args.smoke else None,
        "tau": {
            "seeds": list(seeds),
            "levels_cm": list(levels),
            "counts": counts,
            "at_rest": at_rest,
            "per_reset": per_reset,
            "planted_check": planted,
            "directions_rad": {str(s): oc.tau_direction(s) for s in seeds},
            "estimates": {str(s): v for s, v in est.items()},
            "attempts": attempts,
        },
    }
    if not args.smoke:
        verdict = oc.tau_from_counts(counts)
        result["outcome"] = verdict["row"]
        result["tau"] |= verdict
    return result


# ----- the smoke source corpus -----------------------------------------------------------------
def smoke_source_plan() -> list[dict]:
    """16 smoke-seed roots with the frozen apple-far-shift-v2 plan's rules (10 / 4 / 2)."""
    plan = []
    frozen = lp.corpus_plan()
    for i, seed in enumerate(range(SMOKE_SOURCE[0], SMOKE_SOURCE[1] + 1)):
        root = frozen[i]
        reset = lp.condition_reset(seed)
        shift = None
        if root["shift"] is not None or i % 4 != 3:
            cm = root["shift"]["cm"] if root["shift"] else 9
            shift = {"step": lp.SHIFT_STEP, "vector": lp.shift_vector(seed, reset, cm), "cm": cm}
        plan.append(
            root
            | {
                "seed": seed,
                "episode_id": f"farshift2-smoke75-{seed}",
                "session_id": f"farshift2-smoke75-reset-{seed}",
                "split": "train" if i < 10 else "val" if i < 14 else "test",
                "reset": reset,
                "shift": shift,
            }
        )
    return plan


def stage_source(report, manifest, evidence, clock, args):
    if not args.smoke:
        raise lp.GuardError("G-mode: the source stage is smoke only (the real source is sealed)")
    corpus = Path(args.source)
    if corpus.exists():
        raise FileExistsError(f"refusing to overwrite {corpus}")
    plan = smoke_source_plan()
    oc.check_role_seeds("smoke", tuple(r["seed"] for r in plan), smoke=True)
    (corpus / "episodes").mkdir(parents=True)
    pool = Pool(sim_workers(), {"p3_checkpoint": L74.p3_checkpoint(evidence), "torch_threads": 1})
    report["_pool"] = pool
    tasks = [
        {"kind": "collect", "folder": str(corpus / "episodes")}
        | root
        | {"shift": None if root["shift"] is None else dict(root["shift"])}
        for root in plan
    ]
    L74.mark_first_render(report)
    records = pool.map(tasks, 3600.0, "smoke source")
    episodes = {
        r["episode_id"]: {"npz_sha256": r["npz_sha256"], "meta_sha256": r["meta_sha256"]}
        for r in records
    }
    sha = L74.seal(
        corpus,
        plan,
        episodes,
        {"revision": report["revision"], "k1_report_sha256": None, "smoke": True, "for": oc.TASK},
    )
    return {"outcome": "SOURCE-SEALED", "source": {"manifest_sha256": sha, "path": str(corpus)}}


# ----- render (Stage 1) ---------------------------------------------------------------------------
def open_source(args, smoke: bool):
    """The source corpus (train and val only; Q-split refuses the test roots) and its checks."""
    from embodied_jepa import first_policy_v2_runtime as rt2

    sha = args.source_sha256 if smoke else oc.SOURCE_CORPUS["manifest_sha256"]
    if not smoke and args.source_sha256 not in (None, sha):
        raise lp.GuardError("G-evidence: --source-sha256 is not the sealed apple-far-shift-v2")
    reader = rt2.CorpusReader(Path(args.source), sha, splits=oc.SOURCE_CORPUS["read_splits"])
    m = reader.manifest
    if m.get("protocol") != oc.SOURCE_CORPUS["protocol"] or m.get("corpus") != lp.CORPUS:
        raise lp.GuardError("G-evidence: the source is not a TASK-074 apple-far-shift-v2 corpus")
    if bool(m.get("provenance", {}).get("smoke")) != bool(smoke):
        raise lp.GuardError("G-evidence: the source's smoke flag does not match this stage")
    test_ids = sorted(m["splits"]["test"])
    excluded = hashlib.sha256(json.dumps(test_ids).encode()).hexdigest()
    if not smoke:
        if lp.plan_digest(m["plan"]) != oc.SOURCE_CORPUS["plan_sha256"]:
            raise lp.GuardError("G-evidence: the source plan differs from apple-far-shift-v2's")
        if excluded != oc.SOURCE_CORPUS["excluded_test_ids_sha256"]:
            raise lp.GuardError("G-evidence: the source's test split is not the recorded one")
        sizes = {k: len(v) for k, v in m["splits"].items()}
        if sizes != oc.SOURCE_CORPUS["split_sizes"]:
            raise lp.GuardError(f"G-evidence: the source's splits are {sizes}")
    return reader, {
        "path": str(reader.corpus),
        "manifest_sha256": sha,
        "excluded_test_roots": len(test_ids),
        "excluded_test_ids_sha256": excluded,
    }


def read_ids(reader) -> list[str]:
    return [e for s in oc.SOURCE_CORPUS["read_splits"] for e in reader.manifest["splits"][s]]


SMOKE_BYTES_PER_ROOT = None  # measured by the render smoke; the manifest records it


def check_disk(report: dict, path: Path, roots: int, bytes_per_root: float | None) -> None:
    free = shutil.disk_usage(path).free
    projected = 0.0 if bytes_per_root is None else 1.5 * roots * float(bytes_per_root)
    need = oc.DISK["min_free_gib"] * GIB + projected
    report["disk_at_start"] = {
        "free_gib": free / GIB,
        "projected_store_gib": projected / GIB,
        "needed_gib": need / GIB,
    }
    if free < need:
        raise lp.GuardError(f"G-disk: {free / GIB:.1f} GiB free < {need / GIB:.1f} GiB needed")


def stage_render(report, manifest, evidence, clock, args):
    reader, source = open_source(args, args.smoke)
    report["_reader"] = reader
    out = Path(args.views)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    ids = read_ids(reader)
    per_root = manifest.get("render_smoke", {}).get("bytes_per_root_max")
    if not args.smoke and per_root is None:
        raise lp.GuardError("G-disk: the manifest has no measured bytes per root")
    out.parent.mkdir(parents=True, exist_ok=True)
    check_disk(report, out.parent, len(ids), per_root)
    report["_watch"].disk_path = str(out.parent)
    tasks = []
    for episode_id in ids:
        record = reader.manifest["episodes"][episode_id]
        meta_path = reader.corpus / "episodes" / f"{episode_id}.json"
        if R65.sha256_file(meta_path) != record["meta_sha256"]:
            raise lp.GuardError(f"G-data: {episode_id}'s metadata differs from its sealed hash")
        meta = json.loads(meta_path.read_text())
        tasks.append(
            {
                "kind": "views",
                "episode_id": episode_id,
                "meta": {
                    k: meta[k]
                    for k in ("seed", "reset", "shift", "misaim_m", "noise_level", "noise_seed")
                },
                "source_npz": str(reader.corpus / "episodes" / f"{episode_id}.npz"),
                "source_npz_sha256": record["npz_sha256"],
                "folder": str(out / "episodes"),
                "transitions": meta["transitions"],
            }
        )
        reader.decoded.add(episode_id)  # the worker decodes it (after its own sha256 check)
    (out / "episodes").mkdir(parents=True)
    pool = Pool(sim_workers(), {"p3_checkpoint": L74.p3_checkpoint(evidence), "torch_threads": 1})
    report["_pool"] = pool
    L74.mark_first_render(report)
    clock.check("render")
    records = pool.map(tasks, 4 * 3600.0, "views")
    by_id = {t["episode_id"]: t for t in tasks}
    bad = [
        r["episode_id"]
        for r in records
        if not all(r["arrays_equal"].values())
        or r["frame_check"]["outside_rule"]
        or r["rows"] != by_id[r["episode_id"]]["transitions"]
    ]
    check = {
        "roots": len(records),
        "arrays_equal_roots": sum(all(r["arrays_equal"].values()) for r in records),
        "frames_identical": sum(r["frame_check"]["identical"] for r in records),
        "frames_within_rule": sum(r["frame_check"]["within_rule"] for r in records),
        "frames_outside_rule": sum(r["frame_check"]["outside_rule"] for r in records),
        "max_level": max(r["frame_check"]["max_level"] for r in records),
        "max_pixels": max(r["frame_check"]["pixels"] for r in records),
        "failed_roots": bad,
        "bytes_total": sum(r["bytes"] for r in records),
        "bytes_per_root_max": max(r["bytes"] for r in records),
        "seconds_per_root_median": float(np.median([r["seconds"] for r in records])),
    }
    report["stages"]["render_check"] = check
    if bad:
        raise lp.GuardError(f"G-repro-render: {len(bad)} roots differ from the sealed ones")
    manifest_path = out / "manifest.json"
    views_manifest = {
        "corpus": "apple-far-shift-v2-views",
        "protocol": oc.PROTOCOL,
        "task": oc.TASK,
        "privileged_scripted_collector": True,
        "learned_control": False,
        "source": source,
        "views": list(oc.RENDERED_VIEWS),
        "view_specs": oc.VIEW_SPECS,
        "hidden_kinds": list(oc.HIDDEN_KINDS),
        "steps": {"render": list(oc.RENDER_STEPS), "hidden": list(oc.HIDDEN_STEPS)},
        "splits": {s: list(reader.manifest["splits"][s]) for s in oc.SOURCE_CORPUS["read_splits"]},
        "episodes": {
            r["episode_id"]: {"npz_sha256": r["npz_sha256"], "bytes": r["bytes"]} for r in records
        },
        "provenance": {
            "revision": report["revision"],
            "frozen_sha256": oc.frozen_sha256(),
            "smoke": bool(args.smoke),
        },
    }
    manifest_path.write_text(json.dumps(views_manifest, indent=1, sort_keys=True) + "\n")
    return {
        "outcome": "VIEWS-SEALED",
        "views": {
            "path": str(out),
            "manifest_sha256": R65.sha256_file(manifest_path),
            "source": source,
            "per_root": [{k: v for k, v in r.items() if k != "arrays_equal"} for r in records],
        },
    }


# ----- readouts (Stage 2) -------------------------------------------------------------------------
class ViewsReader:
    """The sealed views store: every read verifies the file's sha256 first."""

    def __init__(self, path: Path, manifest_sha256: str):
        self.path = Path(path)
        if R65.sha256_file(self.path / "manifest.json") != manifest_sha256:
            raise lp.GuardError("G-data: the views manifest differs from its sealed hash")
        self.manifest = json.loads((self.path / "manifest.json").read_text())

    def arrays(self, episode_id: str, keys) -> dict:
        npz = self.path / "episodes" / f"{episode_id}.npz"
        if R65.sha256_file(npz) != self.manifest["episodes"][episode_id]["npz_sha256"]:
            raise lp.GuardError(f"G-data: {episode_id}'s views differ from their sealed hash")
        with np.load(npz) as data:
            return {k: data[k] for k in keys}


class CycledViews:
    """The views store behind ``L74.CycledReader``'s slots (the scale probe)."""

    def __init__(self, views: ViewsReader, slot_source: dict):
        self.views, self.slot_source = views, slot_source

    def arrays(self, episode_id: str, keys) -> dict:
        return self.views.arrays(self.slot_source[episode_id], keys)


def band_hi(length: int) -> int:
    return min(oc.FEATURE_BAND[1], int(length) - 1)


def load_roots(source) -> list[dict]:
    roots = []
    for split in oc.SOURCE_CORPUS["read_splits"]:
        for episode_id in source.manifest["splits"][split]:
            arrays, meta = source.episode(episode_id, keys=("apple", "plate", "states"))
            length = len(arrays["apple"])
            hi = band_hi(length)
            if hi <= oc.FEATURE_BAND[0] + oc.CHUNK:  # lewm_planner_v2 featurisation's rule
                continue
            apple = np.asarray(arrays["apple"], np.float64)[:, :2]
            plate = np.asarray(arrays["plate"], np.float64)[:, :2]
            roots.append(
                {
                    "id": episode_id,
                    "split": split,
                    "shifted": meta.get("shift") is not None,
                    "hi": hi,
                    "offset": apple - plate,
                    "plate": plate,
                    "apple": apple,
                    "states": {
                        s: np.asarray(arrays["states"][s], np.float64)
                        for s in oc.RENDER_STEPS
                        if s <= hi
                    },
                }
            )
            del arrays
    return roots


def expected_steps(root: dict) -> tuple[list[int], list[int]]:
    """The rendered and hidden steps a root has (the band and its length bound both)."""
    hi = root["hi"]
    return [s for s in oc.RENDER_STEPS if s <= hi], [s for s in oc.HIDDEN_STEPS if s <= hi]


def view_frames(view: str, root: dict, source, views):
    """``(visible [n_vis, s, s, 3], hidden {kind: [n_hid, s, s, 3]})`` of one root, in step
    order; the reference view's visible frames are the source corpus's own (copied)."""
    want_vis, want_hid = expected_steps(root)
    keys = ["steps", "hidden_steps", *[f"{view}__{k}" for k in oc.HIDDEN_KINDS]]
    if view != oc.REFERENCE_VIEW:
        keys.append(view)
    got = views.arrays(root["id"], keys)
    steps = [int(s) for s in got["steps"]]
    hidden_steps = [int(s) for s in got["hidden_steps"]]
    if [s for s in steps if s <= root["hi"]] != want_vis or [
        s for s in hidden_steps if s <= root["hi"]
    ] != want_hid:
        raise lp.GuardError(f"G-data: {root['id']}'s rendered steps are not the frozen ones")
    take = [steps.index(s) for s in want_vis]
    take_h = [hidden_steps.index(s) for s in want_hid]
    if view == oc.REFERENCE_VIEW:
        arrays, _meta = source.episode(root["id"], keys=("frames",))
        visible = arrays["frames"][np.asarray(want_vis)].copy()  # memory: a copy, not a view
        del arrays
    else:
        visible = got[view][take]
    hidden = {k: got[f"{view}__{k}"][take_h] for k in oc.HIDDEN_KINDS}
    return visible, hidden


def build_view(view, roots, source, views, encoder, floor, clock) -> tuple:
    """Featurise one view and build its Gram matrices (``obs_ceiling_v2_offline.ViewData``).

    Memory: the view's frames are written into one preallocated array (visible rows of every
    root, then each hidden kind's rows); the full tokens are held one column slice at a time
    (``full_token_gram``); every frame is a Gram row."""
    from embodied_jepa import obs_ceiling_v2_offline as off

    px = oc.VIEW_SPECS[view]["frame_px"]
    counts = [expected_steps(r) for r in roots]
    n_vis = sum(len(v) for v, _h in counts)
    n_hid = sum(len(h) for _v, h in counts)
    offsets = {"vis": 0}
    for j, k in enumerate(oc.HIDDEN_KINDS):
        offsets[k] = n_vis + j * n_hid
    all_frames = np.empty((n_vis + len(oc.HIDDEN_KINDS) * n_hid, px, px, 3), np.uint8)
    vis, hid, gram_index = {}, {k: {} for k in oc.HIDDEN_KINDS}, {}
    at = {"vis": 0, **{k: 0 for k in oc.HIDDEN_KINDS}}
    for i, r in enumerate(roots):
        clock.check(f"frames {view}")
        visible, hidden = view_frames(view, r, source, views)
        want_vis, want_hid = counts[i]
        for j, s in enumerate(want_vis):
            row = at["vis"]
            all_frames[row] = visible[j]
            vis[(i, s)] = row
            gram_index[("vis", i, s)] = row
            at["vis"] += 1
        for k in oc.HIDDEN_KINDS:
            for j, s in enumerate(want_hid):
                hid[k][(i, s)] = at[k]
                gram_index[(k, i, s)] = offsets[k] + at[k]
                all_frames[offsets[k] + at[k]] = hidden[k][j]
                at[k] += 1
        del visible, hidden
    started = time.monotonic()
    pooled_all, g_full, d_full = off.full_token_gram(encoder, all_frames, device="cuda")
    record = {"frames": int(len(all_frames)), "visible": n_vis, "hidden_per_kind": n_hid}
    record["anchor"] = off.anchor_check(encoder, all_frames[:32], pooled_all[:32])
    floor_pooled, _ = off.featurise(floor, all_frames[:n_vis], device="cuda")
    record["featurise_and_full_gram_seconds"] = time.monotonic() - started
    started = time.monotonic()
    g_pix, d_pix = off.gram(all_frames.reshape(len(all_frames), -1))
    record["pixel_gram_seconds"] = time.monotonic() - started
    del all_frames
    by_row = {row: key for key, row in gram_index.items()}
    y_all = np.asarray([roots[by_row[r][1]]["offset"][by_row[r][2]] for r in range(len(by_row))])
    pooled_hidden = {k: pooled_all[offsets[k] : offsets[k] + n_hid] for k in oc.HIDDEN_KINDS}
    data = off.ViewData(
        roots,
        vis,
        hid,
        pooled_all[:n_vis],
        floor_pooled,
        pooled_hidden,
        gram_index,
        {"full": (g_full, d_full, y_all), "pix": (g_pix, d_pix, y_all)},
    )
    return data, record


def repro_off(report, source, encoder, clock, smoke: bool) -> dict:
    """G-repro-off: TASK-074's own R_off path on the reference view reproduces its train
    stage's numbers exactly (protocol §5.2)."""
    from embodied_jepa import lewm_planner_v2_offline as off74

    started = time.monotonic()
    sources = [("far", source, s) for s in oc.SOURCE_CORPUS["read_splits"]]
    table, plate_rows, _af, _ar = L74.featurise_sources(sources, encoder, clock)
    del plate_rows, _af, _ar
    idx = {s: [i for i, r in enumerate(table.roots) if r["split"] == s] for s in ("train", "val")}
    r_off, rows = off74.fit_r_off(table, idx["train"])
    starts, owners = table.windows(idx["val"], start_steps=set(lp.DECISION_STEPS))
    base = off74.o2_statistics({}, {}, r_off, table, starts, owners)["baselines"]
    ref = oc.TASK074_REFERENCE
    got = {
        "encoded_median_cm": base["encoded_median_cm"],
        "persistence_median_cm": base["persistence_median_cm"],
        "clock_prior_median_cm": base["clock_prior_median_cm"],
        "windows": base["windows"],
        "roots": base["roots"],
        "r_off_readout_sha256": r_off.sha256(),
        "r_off_rows": int(len(rows)),
        "r_off_groups": int(r_off.selection["groups"]),
    }
    matches = {k: got[k] == ref[k] for k in got}
    record = {
        "got": got,
        "matches": matches,
        "frames": int(table.size),
        "seconds": time.monotonic() - started,
    }
    del table
    if not smoke and not all(matches.values()):
        raise lp.GuardError(f"G-repro-off: TASK-074's R_off numbers are not reproduced: {got}")
    return record


def readouts_core(report, source, views, clock, *, smoke: bool, probe: bool = False) -> dict:
    from embodied_jepa import obs_ceiling_v2_offline as off
    from embodied_jepa import pretrained_encoder as pe

    encoder, floor = pe.load_pretrained(), pe.random_init()
    out = {"repro_off": repro_off(report, source, encoder, clock, smoke)}
    started = time.monotonic()
    roots = load_roots(source)
    folds = oc.outer_folds([r["id"] for r in roots])
    for r in roots:
        r["fold"] = folds[r["id"]]
    out["roots"] = {
        "read": len(roots),
        "by_split": {
            s: sum(r["split"] == s for r in roots) for s in oc.SOURCE_CORPUS["read_splits"]
        },
        "fold_sizes": [sum(r["fold"] == k for r in roots) for k in range(oc.OUTER_FOLDS)],
        "seconds": time.monotonic() - started,
    }
    if not probe:
        # The readouts stage's boundary (VOID_RULE): before the first out-of-fold fit of any view.
        report["first_outcome_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stats, records, errors = {}, {}, {}
    for view in oc.VIEWS:
        clock.check(f"view {view}")
        data, record = build_view(view, roots, source, views, encoder, floor, clock)
        started = time.monotonic()
        fit = off.cross_fit(data, log=log)
        record["fit_seconds"] = time.monotonic() - started
        record["selections"] = fit["selections"]
        started = time.monotonic()
        stats[view] = off.view_statistics(fit)
        record["statistics_seconds"] = time.monotonic() - started
        records[view] = record
        errors[view] = fit
        log(f"view {view}: {stats[view]['windows']} windows on {stats[view]['roots']} roots")
        del data
        if "_watch" in report:
            record["tree_pss_gib_after"] = report["_watch"].sample()["pss"] / GIB
    tau = SMOKE_TAU_CM if smoke else oc.TAU_MEASURED["tau_cm"]
    decision = oc.decide(
        {v: {k: stats[v][k] for k in ("r_off", "r_full", "r_pix")} for v in oc.VIEWS}, tau
    )
    decision["downstream_bar_interval"] = (
        oc.downstream_bar_interval(stats[decision["chosen_view"]]["r_off"]["c_upper"], tau)
        if decision["row"] == "OBS-ONBOARD"
        else None
    )
    decision["learning_curve_still_falling"] = {
        v: stats[v]["learning_curve_still_falling"] for v in oc.VIEWS
    }
    if smoke:
        decision["smoke_tau_placeholder_cm"] = SMOKE_TAU_CM
    out |= {"views": records, "statistics": stats, "decision": decision}
    return out | {"_errors": errors}


def save_errors(folder: Path, errors: dict) -> dict:
    shas = {}
    for view, fit in errors.items():
        path = folder / f"errors_{view}.npz"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        np.savez(
            path,
            clusters=fit["clusters"],
            windows=np.asarray([f"{r}@{t}" for r, t in fit["windows"]]),
            **{f"e__{k}": v for k, v in fit["errors"].items()},
        )
        shas[view] = R65.sha256_file(path)
    return shas


def scale_probe(report, source, views, clock) -> dict:
    """The readouts stage's memory at full scale (``oc.MEMORY['scale_rule']``): the stage's own
    function on the smoke source and views presented at 240 train + 30 val slots, every slot
    decoded afresh. Memory only; nothing is read."""
    started = time.monotonic()
    watch = report["_watch"]
    before = watch.sample()
    counts = dict(oc.SOURCE_CORPUS["split_sizes"])
    counts.pop("test")
    cycled = L74.CycledReader(source, counts)
    result = readouts_core(
        report, cycled, CycledViews(views, cycled.source), clock, smoke=True, probe=True
    )
    peak = watch.peak_pss / GIB
    limit = oc.MEMORY["ceiling_gib"] - oc.SCALE_MARGIN_GIB
    record = {
        "slots": counts,
        "decoded_per_slot": True,
        "roots": result["roots"],
        "views": {v: {k: r[k] for k in r if k != "selections"} for v, r in result["views"].items()},
        "tree_pss_gib_before": before["pss"] / GIB,
        "stage_peak_tree_pss_gib_so_far": peak,
        "ceiling_gib": oc.MEMORY["ceiling_gib"],
        "margin_gib": oc.SCALE_MARGIN_GIB,
        "within_margin": bool(peak <= limit),
        "seconds": time.monotonic() - started,
        "note": "smoke data at the real stage's sizes, decoded per slot; memory only",
    }
    if not record["within_margin"]:
        raise lp.GuardError(
            f"G-memory-margin: the scale peak {peak:.2f} GiB is above {limit:.2f} GiB"
        )
    return record


def stage_readouts(report, manifest, evidence, clock, args):
    upstream = read_stage_report(
        args.render_report, args.render_sha256, "VIEWS-SEALED", args.smoke, report
    )
    reader, source = open_source(args, args.smoke)
    report["_reader"] = reader
    if upstream["stages"]["views"]["source"]["manifest_sha256"] != source["manifest_sha256"]:
        raise lp.GuardError("G-evidence: the views were not rendered from this source")
    views = ViewsReader(Path(args.views), upstream["stages"]["views"]["manifest_sha256"])
    gpu_guard(report)
    result = readouts_core(report, reader, views, clock, smoke=args.smoke)
    errors = result.pop("_errors")
    result["errors_sha256"] = save_errors(Path(args.output), errors)
    if reader.test_split_decoded:
        raise lp.GuardError("Q-split: a test root was decoded")
    outcome = result["decision"]["row"]
    if args.smoke and args.scale:
        del errors
        import gc

        gc.collect()
        result["scale"] = scale_probe(report, reader, views, clock)
    gpu_peak(report)
    return {"outcome": "READOUTS-SMOKE" if args.smoke else outcome} | {
        k: v for k, v in result.items()
    }


STAGES = {
    "tau": stage_tau,
    "source": stage_source,
    "render": stage_render,
    "readouts": stage_readouts,
}


def run(args) -> dict:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    evidence = Path(args.evidence).resolve()
    report = {
        "protocol": oc.PROTOCOL,
        "task": oc.TASK,
        "mode": args.mode,
        "smoke": bool(args.smoke),
        "outcome": None,
        "workers": {"sim": sim_workers()},
        "stages": {},
        "paths": {"output": str(output), "evidence": str(evidence)},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    clock = R65.Clock(oc.CAPS_SECONDS["global"])
    watch = None
    try:
        watch = install_guards()
        report["_watch"] = watch
        kind = "smoke" if report["smoke"] else args.mode
        manifest = preflight(report, kind, evidence)
        if args.mode == "preflight":
            pool = Pool(
                sim_workers(), {"p3_checkpoint": L74.p3_checkpoint(evidence), "torch_threads": 1}
            )
            report["_pool"] = pool
            RUN.refit_p_readout(report, pool, evidence, report["_run1"])
            result = {"outcome": "PREFLIGHT-READY"}
        else:
            result = STAGES[args.mode](report, manifest, evidence, clock, args)
        end_checks(report, manifest, kind)
        for key, value in result.items():
            if key == "outcome":
                report["outcome"] = value
            else:
                report["stages"][key] = value
    except BaseException as error:  # noqa: BLE001 - every failure, signal included, is V
        RUN.void(report, error)
    finally:
        RUN.stop_handling()
        pool = report.pop("_pool", None)
        if watch is not None:
            L74.finish_guards(report, watch, pool)
        elif pool is not None:
            report["pool_close"] = pool.close()
        report.pop("_watch", None)
        reader = report.pop("_reader", None)
        if reader is not None:
            report["decoded_episodes"] = len(reader.decoded)
            report["test_split_decoded"] = reader.test_split_decoded
        report.pop("_run1", None)
        report["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["total_seconds"] = clock.elapsed()
        R65.write_report(output / "report.json", report)
        log(f"report written: outcome {report.get('outcome')}")
    return report


NEEDS = {
    "source": ("source",),
    "render": ("source", "views"),
    "readouts": ("source", "views", "render_report", "render_sha256"),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--evidence",
        required=True,
        help="the checkout that holds TASK-072 run-1's outputs/, checkpoints/, data/",
    )
    for name in ("source", "source-sha256", "views", "render-report", "render-sha256"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--smoke", action="store_true", help="mechanics only; nothing is read")
    parser.add_argument("--scale", action="store_true", help="readouts smoke: full-scale probe")
    parser.add_argument("--workers", type=int, default=None, help="smoke only (1-6)")
    args = parser.parse_args(argv)
    if args.workers is not None:
        if not args.smoke or not 1 <= args.workers <= 6:
            parser.error("--workers is for smoke runs only, from 1 to 6")
        WORKERS_OVERRIDE["sim"] = args.workers
    if args.smoke and args.mode not in SMOKE_MODES:
        parser.error(f"--smoke is for the modes {', '.join(SMOKE_MODES)}")
    if args.scale and not (args.smoke and args.mode == "readouts"):
        parser.error("--scale is for the readouts smoke only")
    if args.smoke and args.mode in ("render", "readouts") and args.source_sha256 is None:
        parser.error("a smoke render or readouts needs --source-sha256 (the smoke source)")
    missing = [n for n in NEEDS.get(args.mode, ()) if getattr(args, n) is None]
    if missing:
        parser.error(f"{args.mode} needs {', '.join('--' + m.replace('_', '-') for m in missing)}")
    report = run(args)
    return 0 if report.get("outcome") not in (None, "V") else 1


if __name__ == "__main__":
    sys.exit(main())
