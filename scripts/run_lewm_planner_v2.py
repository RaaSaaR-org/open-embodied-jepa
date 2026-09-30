"""TASK-074 runner (``apple_lewm_planner_v2``): the staged run from K1 to the gated cohorts.

Protocol ``docs/experiments/apple_lewm_planner_v2.md``; design ``src/embodied_jepa/
lewm_planner_v2.py``; controllers ``place_planner.py``; workers ``lewm_planner_v2_runtime.py``;
offline pieces ``lewm_planner_v2_offline.py``. The run guards are ``run_guards`` (#111): the
bounded pool, the PSS measure with its RSS fallback, and ``RerunRule``.

Modes (every stage writes ``<output>/report.json``, refuses to overwrite it, and is V on any
guard, crash or cap):

- ``preflight``: the guards, G-evidence and G-repro (TASK-072 run-1's readouts refitted and
  reproduced exactly). It simulates no TASK-074 seed.
- ``smoke``: the GO's render check on smoke seeds (54650-54699). Nothing in it is read.
- ``k1``: the condition gate on cohort K1 (development, no world model).
- ``corpus``: ``apple-far-shift-v2`` (after K1-PASS).
- ``train``: featurisation, R_off and R_plate, the blind baselines, the budget rule,
  W and N x 3, O1 and O2.
- ``pfar``: the data-matched BC control P-far (BC + 3 DAgger), a reported arm.
- ``rank``: O3/O4 groups on cohort R.
- ``decide``: the offline decision (O1-O5).
- ``d3``: the development closed loop on cohort D3 (after OFFLINE-PASS).
- ``gated``: cohorts S and U. It is refused until an owner authorisation record is pinned in
  the manifest (``gated_authorization``).

``--smoke`` runs k1, corpus, train, pfar, rank, decide and d3 on smoke seeds with tiny budgets.
It is mechanics only, and nothing in it is read. The train smoke also runs the train stage's
memory probe at full corpus scale ("train-scale").

**Learned Apple->Plate on the frozen benchmark is still 0 successes.**
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
import importlib.util
import json
import platform
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import devices  # noqa: E402
from embodied_jepa import first_policy_v2_linux as fpl  # noqa: E402
from embodied_jepa import first_policy_v2_m2 as fm  # noqa: E402
from embodied_jepa import lewm_planner_v2 as lp  # noqa: E402
from embodied_jepa import lewm_planner_v2_runtime as lrt  # noqa: E402
from embodied_jepa import run_guards as rg  # noqa: E402
from embodied_jepa import wm_critic_v2 as wc  # noqa: E402

MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-lewm-planner-v2.json"
fpl.configure_headless()  # MUJOCO_GL=egl before any MuJoCo import; workers inherit it
MODES = ("preflight", "smoke", "k1", "corpus", "train", "pfar", "rank", "decide", "d3", "gated")
SMOKE_STAGES = ("k1", "corpus", "train", "pfar", "rank", "decide", "d3")


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN = _load("_task073_runner", "scripts/run_wm_critic_v2.py")  # pinned; imported, unmodified
R65, LIN, M2R = RUN.R65, RUN.LIN, RUN.M2R
log = R65.log
GIB = 2**30
WORKERS_OVERRIDE: dict = {}


def sim_workers() -> int:
    return int(WORKERS_OVERRIDE.get("sim", lp.SIM_WORKERS))


def h_workers() -> int:
    return int(WORKERS_OVERRIDE.get("h", lp.H_WORKERS))


# ----- pool, memory, signals (run_guards; #108 and its review follow-ups) ------------------------
class Pool(rg.BoundedPool):
    """``run_guards.BoundedPool`` on TASK-074's worker, raising the protocol's GuardError on a dead
    worker or a cap, with TASK-071's G-look check on every map."""

    def __init__(self, workers: int, config: dict):
        super().__init__(workers, lrt.run_task, initializer=lrt.worker_init, initargs=(config,))
        self.look_reference = None

    def map(self, tasks: list[dict], cap: float, what: str) -> list[dict]:
        try:
            out = super().map(tasks, cap, what)
        except (rg.WorkerDied, rg.MapCapExceeded) as error:
            raise lp.GuardError(str(error)) from error
        self.look_reference = LIN.check_look_states(out, self.look_reference)
        return out


class MemoryWatch(threading.Thread):
    """Samples the process tree (``run_guards.process_tree_memory``: PSS with the RSS fallback);
    above the ceiling it records why and sends SIGUSR1 to the main process once."""

    def __init__(self, ceiling_bytes: int, interval: float):
        super().__init__(daemon=True)
        self.ceiling, self.interval = int(ceiling_bytes), float(interval)
        self.peak_pss = self.peak_rss = self.peak_processes = 0
        self.fallback_samples = self.max_fallback_processes = self.samples = 0
        self.reason = None
        self._stop = threading.Event()

    def sample(self) -> dict:
        per: dict = {}
        both = rg.process_tree_memory(per=per)
        self.peak_pss = max(self.peak_pss, both["pss"])
        self.peak_rss = max(self.peak_rss, both["rss"])
        self.peak_processes = max(self.peak_processes, len(per))
        if both["pss_fallback_processes"]:
            self.fallback_samples += 1
            self.max_fallback_processes = max(
                self.max_fallback_processes, both["pss_fallback_processes"]
            )
        self.samples += 1
        return both

    def run(self):
        while not self._stop.wait(self.interval):
            both = self.sample()
            if both[lp.MEMORY["measure_key"]] > self.ceiling and self.reason is None:
                self.reason = (
                    f"G-memory: process-tree PSS {both['pss'] / GIB:.2f} GiB > "
                    f"{self.ceiling / GIB:.2f} GiB"
                )
                os.kill(os.getpid(), signal.SIGUSR1)

    def stop(self):
        self._stop.set()

    def summary(self) -> dict:
        return {
            "measure": lp.MEMORY["measure_key"],
            "ceiling_gib": self.ceiling / GIB,
            "peak_tree_pss_gib": self.peak_pss / GIB,
            "peak_tree_rss_gib": self.peak_rss / GIB,
            "peak_processes": self.peak_processes,
            "samples": self.samples,
            "samples_with_rss_fallback": self.fallback_samples,
            "max_rss_fallback_processes": self.max_fallback_processes,
            "interval_seconds": self.interval,
        }


def install_guards() -> MemoryWatch:
    """Stop signals and the memory ceiling end the stage as V with a written report (TASK-073's
    signal-safe handler, carried: a flag, never SIG_IGN)."""
    watch = MemoryWatch(lp.MEMORY["ceiling_gib"] * GIB, lp.MEMORY["sample_seconds"])
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


def finish_guards(report: dict, watch: MemoryWatch, pool) -> None:
    RUN.stop_handling()
    error = sys.exc_info()[1]
    if isinstance(error, RUN.StageInterrupted) and report.get("outcome") != "V":
        RUN.void(report, error)
    if pool is not None:
        report["pool_close"] = pool.close()
    watch.stop()
    report["memory"] = watch.summary()
    if RUN.GUARD["late_signals"]:
        report["signals_during_cleanup"] = list(RUN.GUARD["late_signals"])


def check_memory_available(report: dict, kind: str) -> None:
    available = RUN.mem_available_bytes()
    report["mem_available_at_start_gib"] = available / GIB
    need = (lp.MEMORY["ceiling_gib"] + lp.MEMORY["headroom_gib"]) * GIB
    if available < need and kind != "smoke":
        raise lp.GuardError(
            f"G-memory: MemAvailable {available / GIB:.1f} GiB < {need / GIB:.0f} GiB at start"
        )


# ----- preflight ----------------------------------------------------------------------------------
def preflight(report: dict, mode: str, evidence: Path) -> dict:
    import mujoco
    import torch

    from embodied_jepa import pretrained_encoder as pe

    manifest = json.loads(MANIFEST.read_text())
    if manifest["frozen"] != lp.frozen_block():
        raise lp.GuardError("G-frozen: the manifest's frozen block differs from the module")
    if manifest["frozen_sha256"] != lp.frozen_sha256():
        raise lp.GuardError("G-frozen: the manifest's frozen_sha256 differs from the module")
    report["frozen_sha256"] = lp.frozen_sha256()
    report["pinned_hashes_at_preflight"] = R65.check_pins(manifest["hashes"])
    dirty = R65.tracked_tree_dirty()
    report["revision"], report["tracked_tree_dirty"] = R65.revision(), bool(dirty)
    if mode != "smoke":
        R65.check_clean(dirty)
    report["platform"] = fpl.check_platform()
    report["thread_env"] = {k: os.environ.get(k) for k in lp.THREAD_ENV}
    if report["thread_env"] != lp.THREAD_ENV:
        raise lp.GuardError(f"G-threads: {report['thread_env']} is not {lp.THREAD_ENV}")
    load = os.getloadavg()
    report["load_average_at_start"] = list(load)
    quiet = (
        load[0] <= lp.QUIET_MACHINE["max_load_average_1min"]
        and load[1] <= lp.QUIET_MACHINE["max_load_average_5min"]
    )
    report["quiet_machine"] = bool(quiet)
    check_memory_available(report, mode)
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
    lp.check_seed_ranges()
    for role in lp.COHORT_ROLES:
        lp.stored_cohort(manifest, role)
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


def p3_checkpoint(evidence: Path) -> str:
    return str(evidence / lp.EVIDENCE["checkpoint_dir"] / f"{lp.CARRIED}.pt")


def mark_first_render(report: dict) -> None:
    report["cohort_first_render_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_stage_report(path, sha, outcome, smoke: bool, into: dict | None = None) -> dict:
    """G-evidence for an upstream report: its sha256, this protocol, its outcome and kind, this
    frozen block, and a clean tree. ``into`` (the consuming report) records what was consumed."""
    if R65.sha256_file(path) != sha:
        raise lp.GuardError(f"G-evidence: {path} differs from its sha256")
    upstream = json.loads(Path(path).read_text())
    if upstream.get("protocol") != lp.PROTOCOL:
        raise lp.GuardError(f"G-evidence: {path} is not a TASK-074 report")
    if upstream.get("outcome") != outcome or bool(upstream.get("smoke")) != bool(smoke):
        raise lp.GuardError(f"G-evidence: {path} is not a recorded {outcome} run of this kind")
    if upstream.get("frozen_sha256") != lp.frozen_sha256():
        raise lp.GuardError(f"G-evidence: {path} ran under another frozen block")
    if upstream.get("tracked_tree_dirty") is not False:
        raise lp.GuardError(f"G-evidence: {path} did not run on a clean tree")
    if into is not None:
        into.setdefault("upstream", {})[outcome] = {
            "path": str(path),
            "sha256": sha,
            "revision": upstream.get("revision"),
            "frozen_sha256": upstream.get("frozen_sha256"),
        }
    return upstream


def check_corpus_provenance(report, reader, manifest, smoke: bool, k1_sha: str | None) -> None:
    """G-evidence for a corpus: its plan is the frozen plan (a real run), and it was collected
    after the K1 report this stage consumed. The consumed corpus is recorded."""
    provenance = reader.manifest.get("provenance", {})
    if not smoke and lp.plan_digest(reader.manifest["plan"]) != manifest["corpus_plan_sha256"]:
        raise lp.GuardError("G-evidence: the corpus plan differs from the manifest's digest")
    if k1_sha is not None and provenance.get("k1_report_sha256") != k1_sha:
        raise lp.GuardError("G-evidence: the corpus was not collected after this K1 report")
    report.setdefault("upstream", {})["corpus"] = {
        "path": str(reader.corpus),
        "manifest_sha256": R65.sha256_file(reader.corpus / "manifest.json"),
        "k1_report_sha256": provenance.get("k1_report_sha256"),
    }


# ----- cohorts ------------------------------------------------------------------------------------
SMOKE_ROLES = {"K1": (54650, 54653), "R": (54654, 54656), "D3": (54657, 54660)}
SMOKE_CORPUS = (54661, 54676)  # 16 roots: 10 train, 4 val, 2 test
SMOKE_RERUN = 2  # the d3 smoke re-runs L-plan and H-twin on its first two seeds
SMOKE_BUDGET = {
    "calibration_updates": 200,
    "calibration_select_every": 100,
    "updates": 200,
    "select_every": 100,
    "p_far_updates": 200,
    "p_far_select_every": 100,
    "dagger_rollouts": (2, 2, 2),
}


def cohort(manifest: dict, role: str, smoke: bool):
    if not smoke:
        resets = lp.stored_cohort(manifest, role)
        return resets, lp.check_role_seeds(role, tuple(resets))
    low, high = SMOKE_ROLES[role]
    seeds = lp.check_role_seeds(role, tuple(range(low, high + 1)), smoke=True)
    resets = {}
    for s in seeds:
        r = lp.condition_reset(s)
        resets[s] = {
            "seed": s,
            **r,
            "shift_m": {str(m): lp.shift_vector(s, r, m) for m in lp.SHIFT_GRID_CM},
        }
    return resets, seeds


def blocked(records: dict) -> dict:
    """Blocked moves per arm (``lewm_planner_v2.SHIFT_BLOCKED``)."""
    return {
        arm: sum(r.get("termination_reason") == "shift_blocked" for r in recs)
        for arm, recs in records.items()
    }


def successes(records) -> list[bool]:
    return [bool(r["success"]) for r in records]


def strip(records: list[dict]) -> list[dict]:
    """Per-attempt facts for the report (commands hashed; latents summarised)."""
    out = []
    for r in records:
        item = {k: v for k, v in r.items() if k not in ("commands", "latents", "decisions")}
        if "commands" in r:
            item["commands_sha256"] = R65.sha256_bytes(
                np.ascontiguousarray(np.asarray(r["commands"], np.float32)).tobytes()
            )
        decisions = r.get("decisions") or []
        item["decisions"] = [
            {k: v for k, v in d.items() if not k.endswith("costs_cm")} for d in decisions
        ]
        item["decision_seconds"] = [d["seconds"] for d in decisions]
        out.append(item)
    return out


def estimates_for(report, pool, readout, encoder, seeds, resets) -> dict:
    est = RUN.cohort_estimates(pool, readout, encoder, seeds, resets, 1800.0)
    report["stages"].setdefault("render_disagreements", {}).update(
        est.pop("_render_disagreements", {})
    )
    return est


# ----- K1 -----------------------------------------------------------------------------------------
def stage_k1(report, manifest, evidence, clock, smoke=False):
    resets, seeds = cohort(manifest, "K1", smoke)
    pool = Pool(sim_workers(), {"p3_checkpoint": p3_checkpoint(evidence), "torch_threads": 1})
    report["_pool"] = pool
    readout, encoder = RUN.refit_p_readout(report, pool, evidence, report["_run1"])
    mark_first_render(report)
    est = estimates_for(report, pool, readout, encoder, seeds, resets)
    cells, chosen, o5 = {}, None, None
    for cm in lp.SHIFT_GRID_CM if not smoke else (9,):

        def shift_of(s, cm=cm):
            return {"step": lp.SHIFT_STEP, "vector": resets[s]["shift_m"][str(cm)]}

        counts, per_reset, records = {}, {}, {}
        while (
            arm := lp.k1_next_arm(counts)
            if not smoke
            else next((a for a in lp.K1_ARMS if a not in counts), None)
        ) is not None:
            clock.check(f"K1 {cm} {arm}")
            extra = {"log_chunks": True} if arm == "H-handover" else {}
            recs = pool.map(
                RUN.arm_tasks(arm, seeds, resets, est, shift_of, **extra), 7200.0, f"K1 {arm}"
            )
            counts[arm] = sum(successes(recs))
            per_reset[arm] = successes(recs)
            records[arm] = recs
            log(f"K1 |d| {cm} cm: {arm} {counts[arm]}/{len(seeds)}")
        verdict = lp.k1_cell_passes(counts)
        cells[str(cm)] = {
            "counts": counts,
            "per_reset": per_reset,
            "shift_blocked": blocked(records),
            **verdict,
            "attempts": {a: strip(r) for a, r in records.items()},
        }
        if verdict["passes"] or smoke:
            chosen = cm
            errors = [
                e for r in records.get("H-handover", []) for e in r.get("o5_relative_errors", [])
            ]
            o5 = lp.o5_passes(errors) | {"decisions": len(errors), "values": errors}
            break
    selection = lp.k1_select({int(k): v["counts"] for k, v in cells.items()})
    if smoke:
        selection = {
            "row": "K1-PASS",
            "shift_step": lp.SHIFT_STEP,
            "shift_cm": chosen,
            "smoke_forced": True,
        }
    return {
        "outcome": selection["row"],
        "k1": {
            "cells": cells,
            "selection": selection,
            "o5": o5,
            "estimates": {str(s): v for s, v in est.items()},
        },
    }


# ----- corpus -------------------------------------------------------------------------------------
def smoke_plan() -> list[dict]:
    """16 smoke-seed roots with the frozen plan's rules (10 / 4 / 2)."""
    plan = []
    seeds = range(SMOKE_CORPUS[0], SMOKE_CORPUS[1] + 1)
    frozen = lp.corpus_plan()
    for i, seed in enumerate(seeds):
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
                "episode_id": f"farshift2-smoke-{seed}",
                "session_id": f"farshift2-smoke-reset-{seed}",
                "split": "train" if i < 10 else "val" if i < 14 else "test",
                "reset": reset,
                "shift": shift,
            }
        )
    return plan


def seal(corpus: Path, plan: list[dict], episodes: dict, provenance: dict) -> str:
    path = corpus / "manifest.json"
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
        "episode_arrays": list(lrt.rtm.EPISODE_ARRAYS),
        "plan": plan,
        "splits": splits,
        "episodes": episodes,
        "provenance": provenance,
    }
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return R65.sha256_file(path)


def corpus_summary(records: list[dict]) -> dict:
    def block(rows):
        return {
            "roots": len(rows),
            "complete": sum(r["complete"] for r in rows),
            "counted_success": sum(r["success"] for r in rows),
            "at_rest": sum(r["at_rest"] for r in rows),
            "off_plate": sum(r["off_plate"] for r in rows),
            "grasp": sum(r["grasp_before_settle"] for r in rows),
            "shift_applied": sum(r["shift_applied"] for r in rows),
            "shift_blocked": sum(bool(r.get("shift_blocked")) for r in rows),
        }

    out = {"all": block(records)}
    for key in ("split", "shifted", "misaimed", "noise_level"):
        for value in sorted({r[key] for r in records}, key=str):
            out[f"{key}={value}"] = block([r for r in records if r[key] == value])
    return out


def stage_corpus(report, manifest, evidence, clock, args):
    read_stage_report(args.k1_report, args.k1_sha256, "K1-PASS", args.smoke, report)
    corpus = Path(args.corpus)
    if corpus.exists():
        raise FileExistsError(f"refusing to overwrite {corpus}")
    plan = lp.corpus_plan() if not args.smoke else smoke_plan()
    lp.check_role_seeds("corpus", tuple(r["seed"] for r in plan), smoke=args.smoke)
    if not args.smoke and lp.plan_digest(plan) != manifest["corpus_plan_sha256"]:
        raise lp.GuardError("G-cohort: the corpus plan differs from the manifest's digest")
    (corpus / "episodes").mkdir(parents=True)
    pool = Pool(sim_workers(), {"p3_checkpoint": p3_checkpoint(evidence), "torch_threads": 1})
    report["_pool"] = pool
    tasks = [
        {"kind": "collect", "folder": str(corpus / "episodes")}
        | root
        | {"shift": None if root["shift"] is None else dict(root["shift"])}
        for root in plan
    ]
    mark_first_render(report)
    clock.check("corpus")
    records = pool.map(tasks, 3 * 3600.0, "corpus")
    episodes = {
        r["episode_id"]: {"npz_sha256": r["npz_sha256"], "meta_sha256": r["meta_sha256"]}
        for r in records
    }
    sha = seal(
        corpus,
        plan,
        episodes,
        {"revision": report["revision"], "k1_report_sha256": args.k1_sha256, "smoke": args.smoke},
    )
    return {
        "outcome": "CORPUS-SEALED",
        "corpus": {"manifest_sha256": sha, "path": str(corpus), "summary": corpus_summary(records)},
    }


def check_corpus_kind(reader, smoke: bool) -> None:
    """G-evidence: a smoke stage reads a smoke corpus, a real stage a real one."""
    if bool(reader.manifest.get("provenance", {}).get("smoke")) != bool(smoke):
        raise lp.GuardError("G-evidence: the corpus's smoke flag does not match this stage")
    if reader.manifest.get("protocol") != lp.PROTOCOL:
        raise lp.GuardError("G-evidence: the corpus is not TASK-074's")


def file_sha(path) -> str:
    return R65.sha256_file(path)


def verify_artifacts(files: dict, what: str) -> None:
    """G-evidence: every artefact a worker will load matches the sha256 its stage recorded."""
    for path, want in files.items():
        if file_sha(path) != want:
            raise lp.GuardError(f"G-evidence: {what} artefact {path} differs from its sha256")


def critic_shas(critic: dict) -> dict:
    return {
        critic["r_off"]: critic["sha256"]["r_off"],
        critic["r_plate"]: critic["sha256"]["r_plate"],
    } | {path: critic["sha256"]["models"][name] for name, path in critic["models"].items()}


# ----- train (and the full-scale memory probe) ----------------------------------------------------
def band_root(table, root: dict, arrays: dict, pooled, plate) -> None:
    band = lp.FEATURE_BAND
    hi = min(band[1], len(arrays["frames"]) - 1)
    sl = slice(band[0], hi + 1)
    actions = np.zeros((len(pooled), 14), np.float32)
    n_act = min(len(arrays["applied"]) - band[0], len(pooled))
    actions[:n_act] = arrays["applied"][band[0] : band[0] + n_act]
    table.add(root, pooled, actions, arrays["apple"][sl], plate, band[0])


def featurise_sources(sources, encoder, clock):
    """The train stage's featurisation: the band frames of every root of ``sources`` (``(kind,
    reader, split)``) into one ``Table``, plus the far roots' decision frames (R-plate's rows) and
    the anchor frames.

    Memory (the TASK-074 train run-1 V, protocol addendum A1): a decoded episode's arrays are
    freed when the next episode is read, so nothing kept from an episode may be a view into them.
    ``frames[t]`` is a view that keeps the whole decoded ``frames`` array (about 27 MiB) alive, so
    the decision frames, their plate xy and the anchor frames are copied. The copies hold the same
    bytes, so no computed number changes. ``train_scale_probe`` calls this same function."""
    from embodied_jepa import first_policy_v2_runtime as rt2
    from embodied_jepa import lewm_planner_v2_offline as off

    band = lp.FEATURE_BAND
    band_len = band[1] - band[0] + 1
    n_roots = sum(len(rd.manifest["splits"][sp]) for _k, rd, sp in sources)
    table = off.Table(capacity=n_roots * band_len)
    plate_rows = {"train": [], "val": []}
    anchor_frames, anchor_rows = [], []
    keys = (*rt2.EPISODE_ARRAYS, *lp.EXTRA_EPISODE_ARRAYS)
    for source, rd, split in sources:
        for episode_id in rd.manifest["splits"][split]:
            clock.check("features")
            arrays, meta = rd.episode(
                episode_id, keys=keys if source == "far" else rt2.EPISODE_ARRAYS
            )
            frames = arrays["frames"]
            hi = min(band[1], len(frames) - 1)
            if hi <= band[0] + lp.CHUNK:
                continue
            sl = slice(band[0], hi + 1)
            if source == "far":
                plate = arrays["plate"][sl]
            else:
                plate = np.repeat(np.asarray([[*meta["truth_xy"][2:], 0.746]]), hi + 1 - band[0], 0)
            pooled = off.pooled_features(encoder, frames[sl], device="cuda")
            root = {
                "id": episode_id,
                "split": split if source == "far" else f"look-{split}",
                "source": source,
                "shift_step": (meta.get("shift") or {}).get("step") if source == "far" else None,
            }
            band_root(table, root, arrays, pooled, plate)
            if source == "far" and len(anchor_frames) < lp.FEATURE_ANCHOR["frames"]:
                take = min(16, lp.FEATURE_ANCHOR["frames"] - len(anchor_frames))
                anchor_frames.extend(f.copy() for f in frames[sl][:take])  # memory: copies
                anchor_rows.extend(range(table.size - len(pooled), table.size - len(pooled) + take))
            if source == "far":
                for t in lp.DECISION_STEPS:
                    if t < len(frames):
                        plate_rows[split].append(
                            (episode_id, frames[t].copy(), arrays["plate"][t, :2].copy())
                        )  # memory: copies, not views that keep the episode alive
            del arrays, frames, pooled, plate
    table.seal()
    return table, plate_rows, anchor_frames, anchor_rows


def train_context(table, idx: dict, schema, metadata: dict) -> dict:
    train_roots = idx["train"] + idx["look-train"] + idx["look-val"]
    train_starts, _ = table.windows(train_roots)
    val_starts, _ = table.windows(idx["val"], stride=4)
    rows = np.concatenate(
        [
            np.arange(table.roots[i]["first"], table.roots[i]["first"] + table.roots[i]["length"])
            for i in train_roots
        ]
    )
    mean, std = RUN.R65_moments(table.features, rows)
    return {
        "table": table,
        "train_starts": train_starts,
        "val_starts": val_starts,
        "scale": np.maximum(std, RUN.ld_floor()),
        "mean": mean,
        "std": std,
        "schema": schema,
        "device": "cuda",
        "train_ids": sorted(table.roots[i]["id"] for i in train_roots),
        "metadata": metadata,
        "cap_seconds": lp.CAPS_SECONDS["per_model"],
        "rows": rows,
    }


def offline_gates(ctx, table, idx, w: dict, n: dict, r_off) -> dict:
    from embodied_jepa import lewm_planner_v2_offline as off
    from embodied_jepa import token_dynamics as td

    enc = td.Moments(np.zeros(table.features.shape[1]))
    rows = ctx["rows"]
    for lo in range(0, len(rows), 8192):
        enc.add(table.features[rows[lo : lo + 8192]] / ctx["scale"])
    basis = td.projection_basis(enc)
    del enc
    windows = {
        "overall": table.windows(idx["val"], stride=4),
        "shifted": table.windows(
            [i for i in idx["val"] if table.roots[i]["shift_step"] is not None], stride=4
        ),
    }
    o1 = off.o1_statistics(w, n, table, windows, ctx["scale"], basis)
    starts, owners = table.windows(idx["val"], start_steps=set(lp.DECISION_STEPS))
    o2 = off.o2_statistics(w, n, r_off, table, starts, owners)
    return {"o1": o1, "o2": o2}


def stage_train(report, manifest, evidence, clock, args):
    from embodied_jepa import first_policy_v2_runtime as rt2
    from embodied_jepa import lewm_planner_v2_offline as off
    from embodied_jepa import pretrained_encoder as pe
    from embodied_jepa import token_dynamics as td

    read_stage_report(args.k1_report, args.k1_sha256, "K1-PASS", args.smoke, report)
    reader = rt2.CorpusReader(Path(args.corpus), args.corpus_sha256, splits=lp.READ_SPLITS)
    report["_reader"] = reader
    check_corpus_kind(reader, args.smoke)
    check_corpus_provenance(report, reader, manifest, args.smoke, args.k1_sha256)
    look = rt2.CorpusReader(
        evidence / fm.EVIDENCE["corpus"],
        fm.EVIDENCE["corpus_manifest_sha256"],
        splits=("train", "val"),
    )
    encoder = pe.load_pretrained()
    look_splits = ("train", "val") if not args.smoke else ("val",)
    sources = [("far", reader, s) for s in lp.READ_SPLITS] + [
        ("look", look, s) for s in look_splits
    ]
    started = time.monotonic()
    table, plate_rows, anchor_frames, anchor_rows = featurise_sources(sources, encoder, clock)
    report["stages"]["features"] = {
        "frames": int(table.size),
        "roots": len(table.roots),
        "seconds": time.monotonic() - started,
        "anchor": off.anchor_check(
            encoder, np.stack(anchor_frames), table.features[np.asarray(anchor_rows)]
        ),
        "test_split_decoded": reader.test_split_decoded,
    }
    started = time.monotonic()
    tok_train = off.full_tokens_cpu(encoder, np.stack([r[1] for r in plate_rows["train"]]))
    r_plate = off.fit_r_plate(
        tok_train,
        np.stack([r[2] for r in plate_rows["train"]]),
        [r[0] for r in plate_rows["train"]],
    )
    del tok_train
    idx = {
        s: [i for i, r in enumerate(table.roots) if r["split"] == s]
        for s in ("train", "val", "look-train", "look-val")
    }
    r_off, _rows = off.fit_r_off(table, idx["train"])
    # The train stage's boundary (VOID_RULE): everything above is fitted on train data only;
    # from here on, numbers on val roots are computed.
    report["first_outcome_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tok_val = off.full_tokens_cpu(encoder, np.stack([r[1] for r in plate_rows["val"]]))
    val_err = 100 * np.linalg.norm(
        r_plate.predict(tok_val) - np.stack([r[2] for r in plate_rows["val"]]), axis=1
    )
    del tok_val
    out_dir = Path(args.output)
    np.savez(out_dir / "r_plate.npz", **r_plate.state())
    np.savez(out_dir / "r_off.npz", **r_off.state())
    report["stages"]["readouts"] = {
        "r_plate": r_plate.selection
        | {
            "sha256": r_plate.sha256(),
            "val_error_cm_median": float(np.median(val_err)),
            "val_error_cm_p90": float(np.quantile(val_err, 0.9)),
        },
        "r_off": r_off.selection | {"sha256": r_off.sha256()},
        "seconds": time.monotonic() - started,
    }
    o2_starts, o2_owners = table.windows(idx["val"], start_steps=set(lp.DECISION_STEPS))
    report["stages"]["blind_baselines"] = off.o2_statistics(
        {}, {}, r_off, table, o2_starts, o2_owners
    )["baselines"]
    R65.write_report(out_dir / "baselines.json", report["stages"]["blind_baselines"])
    schema = rt2.make_robot().state_schema
    ctx = train_context(table, idx, schema, {"corpus_manifest_sha256": args.corpus_sha256})
    ckpt = Path(args.checkpoints)
    ckpt.mkdir(parents=True, exist_ok=False)
    b = dict(lp.BUDGET) | (SMOKE_BUDGET if args.smoke else {})
    calibration = {}
    for arm, seed in b["calibration_runs"]:
        clock.check(f"calibration {arm}")
        _m, rec = off.train_model(
            ctx, arm, seed, b["calibration_updates"], b["calibration_select_every"]
        )
        calibration[f"{arm}-{seed}"] = rec | {
            "saturation_update": td.saturation_update(rec["val_curve"], b["saturation_tolerance"])
        }
    budget = lp.budget_updates([v["saturation_update"] for v in calibration.values()])
    report["stages"]["budget"] = {"calibration": calibration, "rule": budget}
    if budget["escalate"] and not args.smoke:
        return {"outcome": "ESCALATE-BUDGET"}
    updates = budget["updates"] if not args.smoke else b["updates"]
    for attempt in range(2):
        models, records = {}, {}
        every = updates // b["selection_points"] if not args.smoke else b["select_every"]
        for arm in ("W", "N"):
            for seed in lp.MODEL_SEEDS:
                clock.check(f"model {arm}-{seed}")
                path = ckpt / f"{arm}-{seed}-u{updates}.pt"
                model, rec = off.train_model(ctx, arm, seed, updates, every, path)
                rec["checkpoint"] = str(path)
                rec["checkpoint_sha256"] = R65.sha256_file(path)
                models[(arm, seed)], records[f"{arm}-{seed}"] = model, rec
        last_two = any(r["selected_in_last_two_points"] for r in records.values())
        report["stages"][f"models_u{updates}"] = records
        if not last_two or args.smoke:
            break
        if attempt == 1 or updates >= b["cap"]:
            return {"outcome": "ESCALATE-BUDGET-LAST-TWO"}
        updates = min(2 * updates, b["cap"])
    w = {s: models[("W", s)] for s in lp.MODEL_SEEDS}
    n = {s: models[("N", s)] for s in lp.MODEL_SEEDS}
    primary = min(lp.MODEL_SEEDS, key=lambda s: (records[f"W-{s}"]["selected_val_criterion"], s))
    if reader.test_split_decoded:
        raise lp.GuardError("Q-split: a test root was decoded")
    gates = offline_gates(ctx, table, idx, w, n, r_off)
    order = [primary, *[x for x in lp.MODEL_SEEDS if x != primary]]
    model_paths = {f"W{i}": records[f"W-{s}"]["checkpoint"] for i, s in enumerate(order)} | {
        "N": records[f"N-{primary}"]["checkpoint"]
    }
    result = {
        "outcome": "TRAIN-COMPLETE",
        "corpus_manifest_sha256": args.corpus_sha256,
        "primary_w_seed": primary,
        "w_order": order,
        "o1": {
            str(s): v["gate"]
            | {"statistics": {f"{k[0]}@h{k[1]}": v2 for k, v2 in v["statistics"].items()}}
            for s, v in gates["o1"].items()
        },
        "o2": {str(s): gates["o2"][s] for s in lp.MODEL_SEEDS},
        "o2_baselines": gates["o2"]["baselines"],
        "critic": {
            "r_off": str(out_dir / "r_off.npz"),
            "r_plate": str(out_dir / "r_plate.npz"),
            "models": model_paths,
            "sha256": {
                "r_off": file_sha(out_dir / "r_off.npz"),
                "r_plate": file_sha(out_dir / "r_plate.npz"),
                "models": {name: file_sha(path) for name, path in model_paths.items()},
            },
        },
    }
    if args.smoke:
        del w, n, models, ctx, gates
        table = None
        import gc

        gc.collect()
        result["train_scale"] = train_scale_probe(report, reader, look, encoder, clock)
    return result


TRAIN_SCALE_ROOTS = {s: lp.CORPUS_SPLITS[s] for s in lp.READ_SPLITS}  # 240 train, 30 val
TRAIN_SCALE_MARGIN_GIB = 2.0  # the probe's peak must stay this far below the ceiling (addendum A1)


class CycledReader:
    """A smoke corpus reader presented at the real corpus's split sizes: slot ``j`` of a split
    is served by a smoke episode, decoded afresh on every read exactly as ``CorpusReader`` does,
    so the featurisation holds and frees the same arrays as on the real corpus."""

    def __init__(self, reader, counts: dict):
        self.reader = reader
        pool = [e for s in lp.READ_SPLITS for e in reader.manifest["splits"][s]]
        self.source, splits, k = {}, {}, 0
        for split, count in counts.items():
            splits[split] = []
            for j in range(count):
                slot = f"scale-{split}-{j}"
                self.source[slot] = pool[k % len(pool)]
                splits[split].append(slot)
                k += 1
        self.manifest = {"splits": splits}

    def episode(self, episode_id: str, keys):
        return self.reader.episode(self.source[episode_id], keys=keys)


def train_scale_probe(report, reader, look, encoder, clock) -> dict:
    """The train stage's memory at full corpus scale (``MEMORY['train_scale_rule']``), on smoke
    data, through the train stage's own code path (addendum A1: the pre-A1 probe cycled 14
    decoded smoke episodes held in memory, so it missed what the real featurisation kept per
    episode, and under-measured the real peak by more than 3 GiB).

    ``featurise_sources`` runs on 270 far roots (240 train + 30 val slots, each a smoke episode
    decoded afresh) and on ``apple-look-v2-linux``'s 170 train + 20 val roots, as the real stage
    reads them. Then, in the real stage's order: R-plate on the train decision frames' full
    tokens, R_off, the val tokens, the blind baselines, the train context, W and N x 3 briefly
    (each with one val-criterion evaluation, the real run-1 V's site), and O1/O2 on the real val
    window count. The peak is the stage's MemoryWatch peak; it must be at least
    ``TRAIN_SCALE_MARGIN_GIB`` below the ceiling, or the smoke fails (G-memory-margin)."""
    from embodied_jepa import first_policy_v2_runtime as rt2
    from embodied_jepa import lewm_planner_v2_offline as off

    started = time.monotonic()
    watch = report["_watch"]
    before = watch.sample()
    sources = [("far", CycledReader(reader, TRAIN_SCALE_ROOTS), s) for s in lp.READ_SPLITS] + [
        ("look", look, s) for s in ("train", "val")
    ]
    table, plate_rows, _anchor_frames, _anchor_rows = featurise_sources(sources, encoder, clock)
    after_features = watch.sample()
    clock.check("train-scale")
    tok_train = off.full_tokens_cpu(encoder, np.stack([r[1] for r in plate_rows["train"]]))
    plate_xy = np.random.default_rng(0).normal(0, 0.05, (len(tok_train), 2))
    r_plate = off.fit_r_plate(tok_train, plate_xy, [r[0] for r in plate_rows["train"]])
    r_plate_rows = int(len(tok_train))
    del tok_train
    idx = {
        s: [i for i, r in enumerate(table.roots) if r["split"] == s]
        for s in ("train", "val", "look-train", "look-val")
    }
    r_off, _rows = off.fit_r_off(table, idx["train"])
    tok_val = off.full_tokens_cpu(encoder, np.stack([r[1] for r in plate_rows["val"]]))
    r_plate.predict(tok_val)
    del tok_val
    o2_starts, o2_owners = table.windows(idx["val"], start_steps=set(lp.DECISION_STEPS))
    off.o2_statistics({}, {}, r_off, table, o2_starts, o2_owners)
    schema = rt2.make_robot().state_schema
    ctx = train_context(table, idx, schema, {"train_scale_probe": True})
    w, n = {}, {}
    for seed in lp.MODEL_SEEDS:
        w[seed], _ = off.train_model(ctx, "W", seed, 100, 100)
        n[seed], _ = off.train_model(ctx, "N", seed, 100, 100)
    offline_gates(ctx, table, idx, w, n, r_off)
    after = watch.sample()
    peak = watch.peak_pss / GIB
    limit = lp.MEMORY["ceiling_gib"] - TRAIN_SCALE_MARGIN_GIB
    record = {
        "frames": int(table.size),
        "roots": len(table.roots),
        "far_slots": TRAIN_SCALE_ROOTS,
        "decoded_per_slot": True,
        "r_plate_rows": r_plate_rows,
        "r_plate_lambda": r_plate.lam_rel,
        "train_windows": int(len(ctx["train_starts"])),
        "val_criterion_windows": int(len(ctx["val_starts"])),
        "o1_windows": {
            "overall": int(len(table.windows(idx["val"], stride=4)[0])),
        },
        "tree_pss_gib_before": before["pss"] / GIB,
        "tree_pss_gib_after_features": after_features["pss"] / GIB,
        "tree_pss_gib_after": after["pss"] / GIB,
        "stage_peak_tree_pss_gib_so_far": peak,
        "ceiling_gib": lp.MEMORY["ceiling_gib"],
        "margin_gib": TRAIN_SCALE_MARGIN_GIB,
        "within_margin": bool(peak <= limit),
        "seconds": time.monotonic() - started,
        "note": "smoke data at the real stage's sizes, decoded per slot; memory only, nothing is "
        "read",
    }
    if not record["within_margin"]:
        raise lp.GuardError(
            f"G-memory-margin: the train-scale peak {peak:.2f} GiB is above "
            f"{limit:.2f} GiB (the ceiling less {TRAIN_SCALE_MARGIN_GIB} GiB)"
        )
    return record


# ----- P-far (the data-matched BC control) --------------------------------------------------------
def pfar_rows(reader, plan_by_id, readout, encoder, fk, bounds, split) -> dict:
    """BC rows of the complete, non-mis-aimed roots of ``split`` (P_FAR['inputs'])."""
    from embodied_jepa import first_policy_perception as fpp
    from embodied_jepa import first_policy_v2_runtime as rt2

    x, y, steps = [], [], []
    used = 0
    for episode_id in reader.manifest["splits"][split]:
        root = plan_by_id[episode_id]
        if root["misaimed"]:
            continue
        arrays, meta = reader.episode(
            episode_id, keys=(*rt2.EPISODE_ARRAYS, *lp.EXTRA_EPISODE_ARRAYS)
        )
        if not meta["complete"]:
            continue
        used += 1
        est = readout.predict(fpp.featurise(encoder, arrays["frame0"][None]))[0]
        rows = rt2.bc_rows(arrays, bounds)
        for state, step, label in zip(rows["states"], rows["steps"], rows["labels"], strict=True):
            e = np.asarray(est, np.float64).copy()
            if step >= lp.TRANSFER_START:
                last = max(t for t in lp.DECISION_STEPS if t <= step)
                e[2:] = arrays["plate"][last, :2]
            x.append(rt2.input_vector(e, int(step), state, fk.pose9(state)))
            y.append(label)
            steps.append(int(step))
    return {
        "x": np.asarray(x, np.float64).reshape(-1, 132),
        "y": np.asarray(y, np.float32).reshape(-1, 7),
        "steps": np.asarray(steps, np.int64),
        "roots": used,
    }


def save_policy(path: Path, trained: dict, standardiser, meta: dict) -> str:
    import torch

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    torch.save(
        {
            "state": trained["state"],
            "heads": 1,
            "mean": torch.as_tensor(standardiser.mean),
            "std": torch.as_tensor(standardiser.std),
            "metadata": meta,
        },
        path,
    )
    return R65.sha256_file(path)


def stage_pfar(report, manifest, evidence, clock, args):
    from embodied_jepa import first_policy_v2_model as fm2
    from embodied_jepa import first_policy_v2_runtime as rt2

    train = read_stage_report(
        args.train_report, args.train_sha256, "TRAIN-COMPLETE", args.smoke, report
    )
    critic = train["stages"]["critic"]
    reader = rt2.CorpusReader(Path(args.corpus), args.corpus_sha256, splits=lp.READ_SPLITS)
    report["_reader"] = reader
    check_corpus_kind(reader, args.smoke)
    check_corpus_provenance(report, reader, manifest, args.smoke, None)
    if args.corpus_sha256 != train["stages"]["corpus_manifest_sha256"]:
        raise lp.GuardError("G-evidence: P-far's corpus is not the one the train stage used")
    verify_artifacts({critic["r_plate"]: critic["sha256"]["r_plate"]}, "train")
    plan = reader.manifest["plan"]
    plan_by_id = {r["episode_id"]: r for r in plan}
    pool = Pool(
        sim_workers(),
        {
            "p3_checkpoint": p3_checkpoint(evidence),
            "torch_threads": 1,
            "r_plate": critic["r_plate"],
        },
    )
    report["_pool"] = pool
    readout, encoder = RUN.refit_p_readout(report, pool, evidence, report["_run1"])
    fk = rt2.PalmFK()
    bounds = rt2.configured_bounds()
    data = pfar_rows(reader, plan_by_id, readout, encoder, fk, bounds, "train")
    val = pfar_rows(reader, plan_by_id, readout, encoder, fk, bounds, "val")
    standardiser = rt2.Standardiser.fit(data["x"])
    ckpt = Path(args.checkpoints)
    ckpt.mkdir(parents=True, exist_ok=False)
    b = SMOKE_BUDGET if args.smoke else {}
    trainings = {}
    report["stages"]["trainings"] = trainings

    def fit(name):
        clock.check(f"P-far {name}")
        trained = fm2.train(
            standardiser(data["x"]),
            data["y"],
            data["steps"],
            standardiser(val["x"]),
            val["y"],
            val["steps"],
            device=lp.P_FAR["device"],
            seed=lp.SEEDS["p_far_sampler"],
            updates=b.get("p_far_updates"),
            select_every=b.get("p_far_select_every"),
            min_output_std=0.0 if args.smoke else None,
        )
        record = trained["record"] | {"rows": int(len(data["x"]))}
        if trained["state"] is None:
            trainings[name] = record | {"no_eligible_checkpoint": True}
            raise lp.GuardError(f"P-far: {name} has no eligible checkpoint")
        path = ckpt / f"{name}.pt"
        record["sha256"] = save_policy(path, trained, standardiser, {"arm": name})
        trainings[name] = record
        return str(path)

    # P-far's void boundary: the first val-selected fit is its first outcome (VOID_RULE)
    report["first_outcome_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    current = fit("P-far-0")
    train_roots = [r for r in plan if r["split"] == "train"]
    sizes = b.get("dagger_rollouts", lp.P_FAR["dagger_rollouts"])
    mark_first_render(report)
    offset = 0
    dagger = {}
    for k, size in enumerate(sizes, start=1):
        roots = train_roots[offset : offset + size]
        offset += size
        seeds = tuple(r["seed"] for r in roots)
        if args.smoke:
            lp.check_role_seeds("corpus", seeds, smoke=True)
        elif not set(seeds) <= set(lp.seeds_of("corpus")):
            raise lp.GuardError("G-seeds: a DAgger rollout seed is not a corpus seed")
        resets = {
            r["seed"]: {k2: r["reset"][k2] for k2 in ("object_xy", "plate_xy")} for r in roots
        }
        est = estimates_for(report, pool, readout, encoder, seeds, resets)
        tasks = [
            {
                "kind": "dagger",
                "seed": r["seed"],
                "reset": resets[r["seed"]],
                "shift": None if r["shift"] is None else dict(r["shift"]),
                "estimates": est[r["seed"]]["estimates"],
                "expected_frame_sha256": est[r["seed"]]["frame_sha256"],
                "expected_state_sha256": est[r["seed"]]["state_sha256"],
                "p_far": current,
            }
            for r in roots
        ]
        clock.check(f"DAgger {k}")
        rollouts = pool.map(tasks, 3 * 3600.0, f"DAgger {k}")
        added = 0
        for r in rollouts:
            if len(r["steps"]):
                data["x"] = np.concatenate((data["x"], r["inputs"]))
                data["y"] = np.concatenate((data["y"], r["labels"]))
                data["steps"] = np.concatenate((data["steps"], np.asarray(r["steps"], np.int64)))
                added += len(r["steps"])
        dagger[f"dagger_{k}"] = {
            "rollouts": len(rollouts),
            "rollout_successes": sum(bool(r["success"]) for r in rollouts),
            "rows_added": added,
            "rows_total": int(len(data["x"])),
        }
        current = fit(f"P-far-{k}")
    report["stages"]["dagger"] = dagger
    if args.smoke:
        report["stages"]["pfar_scale"] = pfar_scale_probe(report, data, val, standardiser, b)
    return {
        "outcome": "PFAR-COMPLETE",
        "p_far": {
            "checkpoint": current,
            "checkpoint_sha256": R65.sha256_file(current),
            "bc_roots": data_roots(plan, reader),
            "val_rows": int(len(val["x"])),
        },
    }


def pfar_scale_probe(report, data, val, standardiser, b) -> dict:
    """P-far's memory at full scale, on smoke rows (MEMORY; DELEGATED_CHOICES['memory']): the BC
    and DAgger rows tiled to the real stage's size (the non-mis-aimed train roots, about 120, plus
    40 + 50 + 60 rollouts, at about 725 rows each), one CUDA training, with the pool's workers
    alive. Memory only; nothing is read."""
    from embodied_jepa import first_policy_v2_model as fm2

    started = time.monotonic()
    watch = report["_watch"]
    before = watch.sample()
    rows = int((120 + sum(lp.P_FAR["dagger_rollouts"])) * 725)
    reps = int(np.ceil(rows / len(data["x"])))
    x = np.concatenate([data["x"]] * reps)[:rows]
    y = np.concatenate([data["y"]] * reps)[:rows]
    steps = np.concatenate([data["steps"]] * reps)[:rows]
    fm2.train(
        standardiser(x),
        y,
        steps,
        standardiser(val["x"]),
        val["y"],
        val["steps"],
        device=lp.P_FAR["device"],
        seed=lp.SEEDS["p_far_sampler"],
        updates=b.get("p_far_updates"),
        select_every=b.get("p_far_select_every"),
        min_output_std=0.0,
    )
    after = watch.sample()
    return {
        "rows": rows,
        "tree_pss_gib_before": before["pss"] / GIB,
        "tree_pss_gib_after": after["pss"] / GIB,
        "stage_peak_tree_pss_gib_so_far": watch.peak_pss / GIB,
        "ceiling_gib": lp.MEMORY["ceiling_gib"],
        "seconds": time.monotonic() - started,
        "note": "smoke rows tiled to the real stage's size; memory only, nothing is read",
    }


def data_roots(plan, reader) -> int:
    ids = set(reader.manifest["splits"]["train"])
    return sum(1 for r in plan if r["episode_id"] in ids and not r["misaimed"])


# ----- rank (O3, O4) ------------------------------------------------------------------------------
def load_critics(critic_files: dict, schema) -> dict:
    from embodied_jepa import token_dynamics as td
    from embodied_jepa.models.frozen_tokens import frozen_token_model
    from embodied_jepa.models.latent_critic import CopyLast, LatentCritic, RidgeReadout, ZeroActions

    with np.load(critic_files["r_off"]) as data:
        r_off = RidgeReadout.from_state({k: data[k] for k in data.files})

    def model(path):
        m = frozen_token_model(td.BACKEND)(schema, device="cpu", seed=0, config=td.MODEL_CONFIG)
        m.load(path)
        m.eval()
        return m

    targets = lp.cost_targets()
    critics = {
        name: LatentCritic(model(path), r_off, targets, lp.CHUNK)
        for name, path in critic_files["models"].items()
        if name.startswith("W")
    }
    critics["N"] = LatentCritic(ZeroActions(model(critic_files["models"]["N"])), r_off, targets)
    critics["copy"] = LatentCritic(CopyLast(), r_off, targets)
    return critics


def stage_rank(report, manifest, evidence, clock, args):
    from embodied_jepa import lewm_planner_v2_offline as off
    from embodied_jepa.first_policy_v2_runtime import make_robot

    train = read_stage_report(
        args.train_report, args.train_sha256, "TRAIN-COMPLETE", args.smoke, report
    )
    k1 = read_stage_report(args.k1_report, args.k1_sha256, "K1-PASS", args.smoke, report)
    cm = int(k1["stages"]["k1"]["selection"]["shift_cm"])
    critic_files = train["stages"]["critic"]
    verify_artifacts(critic_shas(critic_files), "train")
    resets, seeds = cohort(manifest, "R", args.smoke)
    pool = Pool(
        h_workers(),
        {
            "p3_checkpoint": p3_checkpoint(evidence),
            "torch_threads": 1,
            "r_plate": critic_files["r_plate"],
        },
    )
    report["_pool"] = pool
    readout, encoder = RUN.refit_p_readout(report, pool, evidence, report["_run1"])
    mark_first_render(report)
    est = estimates_for(report, pool, readout, encoder, seeds, resets)
    tasks = [
        {
            "kind": "rank",
            "seed": s,
            "reset": {k: resets[s][k] for k in ("object_xy", "plate_xy")},
            "shift": {"step": lp.SHIFT_STEP, "vector": resets[s]["shift_m"][str(cm)]},
            "estimates": est[s]["estimates"],
            "expected_frame_sha256": est[s]["frame_sha256"],
            "expected_state_sha256": est[s]["state_sha256"],
        }
        for s in seeds
    ]
    clock.check("rank")
    results = pool.map(tasks, 4 * 3600.0, "rank")
    groups = []
    for r in results:
        if r["render_retries"]:
            report["stages"].setdefault("render_retries", {})[str(r["seed"])] = r["render_retries"]
        for g in r["groups"]:
            groups.append(g | {"seed": r["seed"]})
    critics = load_critics(critic_files, make_robot().state_schema)
    blind_only = off.rank_statistics(
        groups, {k: v for k, v in critics.items() if k in ("copy", "N")}
    )
    report["stages"]["blind_rankers"] = blind_only
    R65.write_report(Path(args.output) / "blind.json", blind_only)
    stats = off.rank_statistics(groups, critics)
    return {
        "outcome": "RANK-COMPLETE",
        "rank": {
            "shift_cm": cm,
            "groups": len(groups),
            "groups_per_step": {
                str(t): sum(g["step"] == t for g in groups) for t in lp.DECISION_STEPS
            },
            "stopped": {str(r["seed"]): r["stopped"] for r in results},
            "statistics": stats,
        },
    }


# ----- the offline decision -----------------------------------------------------------------------
def stage_decide(report, manifest, evidence, clock, args):
    train = read_stage_report(
        args.train_report, args.train_sha256, "TRAIN-COMPLETE", args.smoke, report
    )
    rank = read_stage_report(
        args.rank_report, args.rank_sha256, "RANK-COMPLETE", args.smoke, report
    )
    k1 = read_stage_report(args.k1_report, args.k1_sha256, "K1-PASS", args.smoke, report)
    t = train["stages"]
    order = [int(s) for s in t["w_order"]]
    stats = rank["stages"]["rank"]["statistics"]
    seeds = {
        s: {
            "O1": t["o1"][str(s)],
            "O2": t["o2"][str(s)]["gate"],
            "O3": stats[f"W{i}"]["O3"],
            "O4": stats[f"W{i}"]["O4"],
        }
        for i, s in enumerate(order)
    }
    o5 = k1["stages"]["k1"]["o5"]
    decision = lp.decide_offline(seeds, o5)
    out = {
        "row": decision["row"],
        "abandonment_clause_fires": decision["abandonment_clause_fires"],
        "smoke": bool(args.smoke),
        "frozen_sha256": lp.frozen_sha256(),
        "revision": report["revision"],
        "shift_step": lp.SHIFT_STEP,
        "shift_cm": int(k1["stages"]["k1"]["selection"]["shift_cm"]),
        "critic": t["critic"],
        "per_seed": {str(s): {k: v.get("passes") for k, v in g.items()} for s, g in seeds.items()},
    }
    R65.write_report(Path(args.output) / "decision.json", out)
    return {"outcome": decision["row"], "decision": out}


# ----- D3 -----------------------------------------------------------------------------------------
def closed_loop(report, manifest, evidence, clock, role, arms, config, cm, workers, smoke):
    """``arms`` once per reset of ``role``; H-twin runs first (L-shuf reads its latents)."""
    resets, seeds = cohort(manifest, role, smoke)
    pool = Pool(workers, {"p3_checkpoint": p3_checkpoint(evidence)} | config)
    report["_pool"] = pool
    readout, encoder = RUN.refit_p_readout(report, pool, evidence, report["_run1"])
    mark_first_render(report)
    est = estimates_for(report, pool, readout, encoder, seeds, resets)

    def shift_of(s):
        return (
            None if cm is None else {"step": lp.SHIFT_STEP, "vector": resets[s]["shift_m"][str(cm)]}
        )

    order = ["H-twin", *[a for a in arms if a != "H-twin"]]
    records = {}
    for arm in order:
        clock.check(f"{role} {arm}")
        extra = {}
        if arm == "L-shuf":
            latents = [r["latents"] for r in records["H-twin"]]
            foreign = {
                s: wc.shuf_latents(latents, i, steps=lp.DECISION_STEPS) for i, s in enumerate(seeds)
            }
            report["stages"].setdefault("l_shuf_substitutions", {}).update(
                {str(s): v[1] for s, v in foreign.items() if v[1]}
            )
            extra["per_seed"] = lambda s, f=foreign: {"foreign": f[s][0]}
        records[arm] = pool.map(
            RUN.arm_tasks(arm, seeds, resets, est, shift_of, **extra), 7200.0, f"{role} {arm}"
        )
        if arm in lp.L1_ARMS and not all(r["privileged_ok"] for r in records[arm]):
            raise lp.GuardError(f"G-privileged: {arm} read simulator truth inside act()")
    return seeds, resets, est, shift_of, records, pool


def stage_d3(report, manifest, evidence, clock, args):
    decision = json.loads(Path(args.offline).read_text())
    if R65.sha256_file(args.offline) != args.offline_sha256 or bool(decision.get("smoke")) != bool(
        args.smoke
    ):
        raise lp.GuardError("G-evidence: the offline decision differs from its sha256 or kind")
    if decision.get("frozen_sha256") != lp.frozen_sha256():
        raise lp.GuardError("G-evidence: the offline decision ran under another frozen block")
    report.setdefault("upstream", {})["decision"] = {
        "path": str(args.offline),
        "sha256": args.offline_sha256,
        "revision": decision.get("revision"),
    }
    if decision["row"] != "OFFLINE-PASS" and not args.smoke:
        raise lp.GuardError("G-evidence: the offline decision is not OFFLINE-PASS")
    pfar = read_stage_report(
        args.pfar_report, args.pfar_sha256, "PFAR-COMPLETE", args.smoke, report
    )
    verify_artifacts(critic_shas(decision["critic"]), "train")
    verify_artifacts(
        {pfar["stages"]["p_far"]["checkpoint"]: pfar["stages"]["p_far"]["checkpoint_sha256"]},
        "P-far",
    )
    config = {k: v for k, v in decision["critic"].items() if k != "sha256"} | {
        "p_far": pfar["stages"]["p_far"]["checkpoint"],
        "torch_threads": lp.H_WORKER_TORCH_THREADS,
    }
    seeds, resets, est, shift_of, records, pool = closed_loop(
        report,
        manifest,
        evidence,
        clock,
        "D3",
        lp.D3_ARMS,
        config,
        int(decision["shift_cm"]),
        h_workers(),
        args.smoke,
    )
    counts = {arm: sum(successes(r)) for arm, r in records.items()}
    result = {
        "counts_of_16": counts,
        "decision": lp.decide_d3(counts),
        "shift_blocked": blocked(records),
        "latency": latency(records["L-plan"]),
        "fallbacks": {
            arm: sum(bool(d.get("fallback")) for r in recs for d in r.get("decisions", []))
            for arm, recs in records.items()
        },
        "attempts": {a: strip(r) for a, r in records.items()},
    }
    if args.smoke:
        result["rerun"] = rerun_measurement(report, pool, seeds, resets, est, shift_of, records)
    return {"outcome": result["decision"]["row"] if not args.smoke else "D3-SMOKE", "d3": result}


def latency(records) -> dict:
    seconds = [d["seconds"] for r in records for d in r.get("decisions", [])]
    return {
        "decisions": len(seconds),
        "median_seconds": float(np.median(seconds)) if seconds else None,
        "p90_seconds": float(np.quantile(seconds, 0.9)) if seconds else None,
        "max_seconds": float(np.max(seconds)) if seconds else None,
        "gpu": RUN.gpu_snapshot(),
    }


def rerun_measurement(report, pool, seeds, resets, est, shift_of, records) -> dict:
    """The re-run tolerances' evidence (RERUN_MARGIN_RULE): L-plan and H-twin again on the first
    seeds, compared by ``place_planner.rerun_compare`` (``run_guards.RerunRule``)."""
    from embodied_jepa import place_planner as pp

    first = seeds[:SMOKE_RERUN]
    out = {}
    for arm in lp.GATED["determinism_rerun_arms"]:
        again = pool.map(RUN.arm_tasks(arm, first, resets, est, shift_of), 3600.0, f"rerun {arm}")
        comparisons = []
        for a, b in zip(records[arm][: len(first)], again, strict=True):
            c = pp.rerun_compare(a, b)
            comparisons.append(
                {
                    "seed": a["seed"],
                    "matches": c["matches"],
                    "missing": c["missing"],
                    "differences": {k: v for k, v in c["differences"].items()},
                    "max_abs_command_difference": c["max_abs_command_difference"],
                    "target_cm_max": pp.target_difference_cm(a, b),
                }
            )
        out[arm] = comparisons
    return out


def stage_gated(report, manifest, evidence, clock, args):
    record = manifest.get("gated_authorization")
    if not record:
        raise lp.GuardError(
            "G-authorisation: no owner or delegated authorisation record is in the manifest"
        )
    report["gated_authorization"] = lp.check_authorisation(record)
    raise lp.GuardError("G-authorisation: the gated stage is written by its authorisation PR")


# ----- smoke: the GO's render check ---------------------------------------------------------------
def stage_smoke(report, manifest, evidence, clock, args):
    seeds = lp.check_role_seeds("smoke", lp.smoke_seeds()[:32], smoke=True)
    resets = {s: lp.condition_reset(s) for s in seeds}
    pool = Pool(sim_workers(), {"p3_checkpoint": p3_checkpoint(evidence), "torch_threads": 1})
    report["_pool"] = pool
    RUN.refit_p_readout(report, pool, evidence, report["_run1"])
    tasks = [
        {"kind": "frame", "seed": s, "reset": {k: resets[s][k] for k in ("object_xy", "plate_xy")}}
        for _ in range(4)
        for s in seeds
    ]
    rendered = pool.map(tasks, 1800.0, "render check")
    check = RUN.render_check_verdict(tasks, rendered)
    return {
        "outcome": "SMOKE-COMPLETE",
        "render_check": check | {"workers": sim_workers(), "renders": len(tasks)},
    }


# ----- run ----------------------------------------------------------------------------------------
STAGES = {
    "smoke": stage_smoke,
    "corpus": stage_corpus,
    "train": stage_train,
    "pfar": stage_pfar,
    "rank": stage_rank,
    "decide": stage_decide,
    "d3": stage_d3,
    "gated": stage_gated,
}


def run(args) -> dict:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    evidence = Path(args.evidence).resolve()
    report = {
        "protocol": lp.PROTOCOL,
        "task": lp.TASK,
        "mode": args.mode,
        "smoke": bool(args.smoke or args.mode == "smoke"),
        "outcome": None,
        "workers": {"sim": sim_workers(), "h": h_workers()},
        "stages": {},
        "paths": {"output": str(output), "evidence": str(evidence)},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    clock = R65.Clock(lp.CAPS_SECONDS["global"])
    watch = None
    try:
        watch = install_guards()  # inside the try: a signal from here on writes the V report
        report["_watch"] = watch
        kind = "smoke" if report["smoke"] else args.mode
        manifest = preflight(report, kind, evidence)
        if args.mode == "preflight":
            pool = Pool(
                sim_workers(), {"p3_checkpoint": p3_checkpoint(evidence), "torch_threads": 1}
            )
            report["_pool"] = pool
            RUN.refit_p_readout(report, pool, evidence, report["_run1"])
            result = {"outcome": "PREFLIGHT-READY"}
        elif args.mode == "k1":
            result = stage_k1(report, manifest, evidence, clock, smoke=args.smoke)
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
        RUN.stop_handling()  # first: from here on a signal is recorded, not raised
        pool = report.pop("_pool", None)
        if watch is not None:
            finish_guards(report, watch, pool)
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
    "corpus": ("k1_report", "k1_sha256", "corpus"),
    "train": ("k1_report", "k1_sha256", "corpus", "corpus_sha256", "checkpoints"),
    "pfar": ("train_report", "train_sha256", "corpus", "corpus_sha256", "checkpoints"),
    "rank": ("train_report", "train_sha256", "k1_report", "k1_sha256"),
    "decide": (
        "train_report",
        "train_sha256",
        "rank_report",
        "rank_sha256",
        "k1_report",
        "k1_sha256",
    ),
    "d3": ("offline", "offline_sha256", "pfar_report", "pfar_sha256"),
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
    for name in (
        "k1-report",
        "k1-sha256",
        "corpus",
        "corpus-sha256",
        "checkpoints",
        "train-report",
        "train-sha256",
        "rank-report",
        "rank-sha256",
        "pfar-report",
        "pfar-sha256",
        "offline",
        "offline-sha256",
    ):
        parser.add_argument(f"--{name}")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="a stage on smoke seeds with tiny budgets (mechanics only; nothing is read)",
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="smoke only: every pool's worker count (1-6)"
    )
    args = parser.parse_args(argv)
    if args.workers is not None:
        if not (args.smoke or args.mode == "smoke") or not 1 <= args.workers <= 6:
            parser.error("--workers is for smoke runs only, from 1 to 6")
        WORKERS_OVERRIDE.update({"sim": args.workers, "h": args.workers})
    if args.smoke and args.mode not in SMOKE_STAGES:
        parser.error(f"--smoke is for the stages {', '.join(SMOKE_STAGES)}")
    missing = [n for n in NEEDS.get(args.mode, ()) if getattr(args, n) is None]
    if missing:
        parser.error(f"{args.mode} needs {', '.join('--' + m.replace('_', '-') for m in missing)}")
    report = run(args)
    return 0 if report.get("outcome") not in (None, "V") else 1


if __name__ == "__main__":
    sys.exit(main())
