"""TASK-076 runner (``apple_plate_twin_v2``): K0, Stage O, Stage D, the gated S/U stage and K-pred,
and their Stage-0 smokes.

Protocol ``docs/experiments/apple_plate_twin_v2.md``; frozen block ``src/embodied_jepa/
plate_twin_v2.py``; workers ``plate_twin_v2_runtime.py``; Stage O ``plate_twin_v2_offline.py``;
ported harness pieces ``plate_twin_v2_harness.py``. Guards from ``embodied_jepa.run_tools``
(``assert_local_import``, ``install_guards`` with ``MemoryWatch``, ``gpu_guard``,
``scale_probe``); this runner loads no other script.

Modes (every stage writes ``<output>/report.json``, refuses to overwrite it, and is V on any
guard, crash, signal or cap):

- ``preflight``: the guards, the pins, G-evidence and G-repro.
- ``k0``: K0, development, on a reviewer's reported GO, before the freeze (refused once
  ``K0_MEASURED`` is frozen).
- ``offline``: Stage O on the sealed stores; the only GPU stage (run it through
  ``scripts/gpu_run.sh``; it refuses to start outside the lock).
- ``dev``, ``gated``, ``kpred``: the closed-loop stages (CPU); ``kpred`` runs one cell.
- ``source``, ``render``: smoke only: a TASK-074-format smoke corpus and its TASK-075-format
  views store, on smoke seeds, the stand-ins for the sealed stores in the offline smoke.

``--smoke`` simulates smoke seeds 56900-56999 only; nothing in a smoke is read. **No arm runs a
world model; learned Apple->Plate on the frozen v1 benchmark is still 0/150.**
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
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import devices  # noqa: E402
from embodied_jepa import first_policy_v2_linux as fpl  # noqa: E402
from embodied_jepa import lewm_planner_v2 as lp  # noqa: E402
from embodied_jepa import obs_ceiling_v2 as oc  # noqa: E402
from embodied_jepa import plate_twin_v2 as pt  # noqa: E402
from embodied_jepa import plate_twin_v2_harness as hz  # noqa: E402
from embodied_jepa import run_tools as rt  # noqa: E402

MANIFEST = ROOT / "benchmarks" / "manifests" / "apple-plate-twin-v2.json"
fpl.configure_headless()  # MUJOCO_GL=egl before any MuJoCo import; workers inherit it
MODES = ("preflight", "k0", "offline", "dev", "gated", "kpred", "source", "render")
SMOKE_ONLY = ("source", "render")
GIB = 2**30
log = hz.log
SMOKE = {
    "k0_seeds": (56900, 56903),
    "k0_levels": (0.0, 3.0),
    "kpred_seeds": {"M-a": (56910, 56917), "M-b": (56920, 56927), "A": (56930, 56945)},
    "source": (56960, 56983),  # 24 roots: 16 train, 6 val, 2 test
    "dev_seeds": (56946, 56949),
    "s_seeds": (56950, 56955),
    "u_seeds": (56956, 56959),
    "tau_cm": 1.0,  # a placeholder tau_re (TASK-075's tau) for the smoke's code paths
}


def parse_range(text: str) -> tuple[int, int]:
    low, _, high = text.partition("-")
    return int(low), int(high or low)


# ----- preflight ----------------------------------------------------------------------------------
def preflight(report: dict, args, *, gpu: bool) -> dict:
    import mujoco
    import torch

    from embodied_jepa import pretrained_encoder as pe

    rt.assert_local_import(ROOT, report)  # F1d: this checkout's own src
    manifest = json.loads(MANIFEST.read_text())
    if manifest["frozen"] != pt.frozen_block():
        raise lp.GuardError("G-frozen: the manifest's frozen block differs from the module")
    if manifest["frozen_sha256"] != pt.frozen_sha256():
        raise lp.GuardError("G-frozen: the manifest's frozen_sha256 differs from the module")
    report["frozen_sha256"] = pt.frozen_sha256()
    report["protocol_status"] = manifest["status"]
    report["pinned_hashes_at_preflight"] = hz.check_pins(manifest["hashes"])
    dirty = hz.tracked_tree_dirty()
    report["revision"], report["tracked_tree_dirty"] = hz.revision(), bool(dirty)
    if not args.smoke:
        hz.check_clean(dirty)
    report["platform"] = fpl.check_platform()
    report["thread_env"] = {k: os.environ.get(k) for k in pt.THREAD_ENV}
    if report["thread_env"] != pt.THREAD_ENV:
        raise lp.GuardError(f"G-threads: {report['thread_env']} is not {pt.THREAD_ENV}")
    load = os.getloadavg()
    report["load_average_at_start"] = list(load)
    quiet = (
        load[0] <= pt.QUIET_MACHINE["max_load_average_1min"]
        and load[1] <= pt.QUIET_MACHINE["max_load_average_5min"]
    )
    report["quiet_machine"] = bool(quiet)
    available = hz.mem_available_bytes()
    report["mem_available_at_start_gib"] = available / GIB
    need = (pt.MEMORY["ceiling_gib"] + pt.MEMORY["headroom_gib"]) * GIB
    if not args.smoke and available < need:
        raise lp.GuardError(f"G-memory: MemAvailable {available / GIB:.1f} GiB < {need / GIB:.0f}")
    if not quiet and not args.smoke:
        raise lp.GuardError(f"G-quiet: load averages {load[:2]} exceed the quiet-machine rule")
    torch.set_num_threads(6)
    if gpu:
        if not devices.available("cuda"):
            raise lp.GuardError("G-device: CUDA is not available")
        device = devices.require("cuda", strict=True)
        report["accelerator"] = devices.accelerator_info(device)
    else:  # the CPU stages take no GPU lock and touch no CUDA context (protocol §8)
        devices.configure_determinism("cuda", strict=True)
    if mujoco.__version__ != manifest["mujoco_version"]:
        raise lp.GuardError(f"G-hash: MuJoCo {mujoco.__version__} is not the pinned version")
    report["environment"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "determinism": devices.determinism_state(),
    }
    digests = {
        "pretrained": pe.weights_digest(pe.load_pretrained()),
        "floor": pe.weights_digest(pe.random_init()),
    }
    report["encoder_digests"] = digests
    if digests != manifest["encoder_digests"]:
        raise lp.GuardError("G-weights: an encoder digest differs from its pin")
    pt.check_seed_ranges()
    for role in pt.COHORT_ROLES:
        if pt.cohort_digest(role) != manifest["cohorts"][role]:
            raise lp.GuardError(f"G-cohort: the {role} draws differ from the manifest's digest")
    found = hz.check_evidence(Path(args.evidence))
    report["evidence"] = {"root": str(args.evidence), "sha256": found["sha256"]}
    report["_run1"] = found["run1"]
    return manifest


def end_checks(report: dict, manifest: dict, smoke: bool) -> None:
    report["pinned_hashes_at_end"] = hz.check_pins(manifest["hashes"])
    fpl.check_determinism(report)
    report["revision_at_end"] = hz.revision()
    if not smoke and hz.tracked_tree_dirty():
        raise lp.GuardError("G-hash: the tracked tree changed during the run")


# ----- cohorts ------------------------------------------------------------------------------------
def cohort(role: str, args, smoke_range=None) -> tuple[tuple[int, ...], dict]:
    if args.smoke:
        low, high = parse_range(args.seeds) if args.seeds else smoke_range
        seeds = pt.check_role_seeds(role, tuple(range(low, high + 1)), smoke=True)
    else:
        seeds = pt.check_role_seeds(role, pt.seeds_of(role))
    return seeds, {int(k): v for k, v in pt.cohort_values(role, seeds).items()}


def mark_first_render(report: dict) -> None:
    report["cohort_first_render_utc"] = hz.utc()


def make_pool(report, args, extra: dict | None = None) -> hz.Pool:
    workers = int(args.workers or pt.SIM_WORKERS)
    config = {
        "p3_checkpoint": hz.p3_checkpoint(Path(args.evidence)),
        "torch_threads": pt.WORKER_TORCH_THREADS,
    } | (extra or {})
    pool = hz.Pool(workers, config)
    report["_pool"] = pool
    report["workers"] = workers
    return pool


def estimates_for(report, pool, args, seeds, resets) -> dict:
    """P-3's post-look estimates: G-repro's refitted readout (every real stage), or the smoke's
    stand-in, the reset truth (``--truth-estimates``, smokes only; disclosed in the report)."""
    if args.truth_estimates:
        report["estimates_source"] = "smoke stand-in: the reset truth (G-repro not run)"
        readout = encoder = None
    else:
        report["estimates_source"] = "G-repro: TASK-072 run-1's P readout, refitted exactly"
        if "_p_readout" not in report:
            report["_p_readout"] = hz.refit_p_readout(
                report, pool, Path(args.evidence), report["_run1"]
            )
        readout, encoder = report["_p_readout"]
    mark_first_render(report)
    est = hz.cohort_estimates(pool, readout, encoder, seeds, resets, 1800.0)
    report["stages"].setdefault("render_disagreements", {}).update(
        est.pop("_render_disagreements", {})
    )
    return est


def attempt_tasks(arm, seeds, resets, est, *, role, per_seed=None, **extra) -> list[dict]:
    tasks = []
    for s in seeds:
        r = resets[s]
        task = {
            "kind": "attempt",
            "arm": arm,
            "seed": s,
            "reset": {"object_xy": r["object_xy"], "plate_xy": r["plate_xy"]},
            "shift": (
                {"step": pt.CONDITION["shift_step"], "vector": r["shift_m"]}
                if role in pt.SHIFTED_ROLES
                else None
            ),
            "estimates": est[s]["estimates"],
            "expected_frame_sha256": est[s]["frame_sha256"],
            "expected_state_sha256": est[s]["state_sha256"],
            "wall_seconds": pt.CAPS_SECONDS["per_attempt"],
        }
        tasks.append(task | (per_seed(s) if per_seed else {}) | extra)
    return tasks


def successes(records) -> list[bool]:
    return [bool(r["success"]) for r in records]


def check_privileged(records, arm: str) -> None:
    """A privileged read in a non-privileged arm is a V (protocol §7)."""
    spec = pt.ARMS.get(arm) or pt.KPRED_ARM_SPECS.get(arm)
    if spec and not spec["privileged"]:
        for r in records:
            if r.get("blocked") is None and not r.get("privileged_ok", False):
                raise lp.GuardError(f"G-privileged: {arm} seed {r['seed']} read task truth")


def run_arm(pool, tasks, what: str) -> list[dict]:
    records = pool.map(tasks, pt.CAPS_SECONDS["invocation"], what)
    check_privileged(records, tasks[0]["arm"])
    return records


def timing(records) -> dict:
    seconds = [float(r["seconds"]) for r in records]
    return {"median": float(np.median(seconds)), "max": float(np.max(seconds)), "n": len(seconds)}


# ----- K0 -----------------------------------------------------------------------------------------
def stage_k0(report, manifest, args, clock):
    if not args.smoke and pt.K0_MEASURED is not None:
        raise lp.GuardError("G-spent: K0 has run and its values are frozen")
    seeds, resets = cohort("K", args, SMOKE["k0_seeds"])
    levels = SMOKE["k0_levels"] if args.smoke else pt.TAU["levels_cm"]
    pool = make_pool(report, args)
    est = estimates_for(report, pool, args, seeds, resets)
    counts, per_reset, attempts = {}, {}, {}
    for cm in levels:
        clock.check(f"tau {cm}")
        tasks = attempt_tasks(
            "H-handover",
            seeds,
            resets,
            est,
            role="K",
            per_seed=lambda s, cm=cm: {"planted_m": pt.planted_error_m(s, cm), "level_cm": cm},
        )
        records = run_arm(pool, tasks, f"tau {cm} cm")
        key = str(float(cm))
        counts[key] = sum(successes(records))
        per_reset[key] = successes(records)
        attempts[key] = hz.strip(records)
        log(f"tau {cm} cm: {counts[key]}/{len(seeds)}")
    plate_at = {
        s: {
            t: (np.asarray(resets[s]["plate_xy"]) + np.asarray(resets[s]["shift_m"])).tolist()
            for t in pt.DECISION_STEPS
        }
        for s in seeds
    }
    clock_targets = pt.clock_targets(plate_at)
    arms = {}
    for arm, extra in (("H-clock", {"clock": clock_targets}), ("H-stale", {})):
        clock.check(arm)
        records = run_arm(pool, attempt_tasks(arm, seeds, resets, est, role="K", **extra), arm)
        arms[arm] = {
            "count": sum(successes(records)),
            "per_reset": successes(records),
            "attempts": hz.strip(records),
        }
        log(f"{arm}: {arms[arm]['count']}/{len(seeds)}")
    result = {
        "seeds": list(seeds),
        "levels_cm": list(levels),
        "counts": counts,
        "per_reset": per_reset,
        "directions_rad": {str(s): pt.tau_direction(s) for s in seeds},
        "clock_targets": {str(k): v for k, v in clock_targets.items()},
        "arms": {a: {k: v for k, v in r.items() if k != "attempts"} for a, r in arms.items()},
        "attempts": attempts | {a: r["attempts"] for a, r in arms.items()},
        "estimates": {str(s): v for s, v in est.items()},
    }
    zero = per_reset[str(0.0)]
    result["g2_feasibility"] = pt.g2_feasibility(zero, arms["H-clock"]["per_reset"])
    if args.smoke:
        return {"outcome": "K0-SMOKE", "k0": result}
    decision = pt.decide_k0(counts, arms["H-clock"]["count"], arms["H-stale"]["count"])
    return {"outcome": decision["row"], "k0": result | {"decision": decision}}


# ----- the smoke source corpus and views store ----------------------------------------------------
def smoke_source_plan() -> list[dict]:
    """Smoke-seed roots with the frozen apple-far-shift-v2 plan's rules (16 / 6 / 2)."""
    plan = []
    frozen = lp.corpus_plan()
    low, high = SMOKE["source"]
    for i, seed in enumerate(range(low, high + 1)):
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
                "episode_id": f"farshift2-smoke76-{seed}",
                "session_id": f"farshift2-smoke76-reset-{seed}",
                "split": "train" if i < 16 else "val" if i < 22 else "test",
                "reset": reset,
                "shift": shift,
            }
        )
    return plan


def stage_source(report, manifest, args, clock):
    corpus = Path(args.source)
    if corpus.exists():
        raise FileExistsError(f"refusing to overwrite {corpus}")
    plan = smoke_source_plan()
    pt.check_role_seeds("smoke", tuple(r["seed"] for r in plan), smoke=True)
    (corpus / "episodes").mkdir(parents=True)
    pool = make_pool(report, args)
    tasks = [
        {"kind": "collect", "folder": str(corpus / "episodes")}
        | root
        | {"shift": None if root["shift"] is None else dict(root["shift"])}
        for root in plan
    ]
    mark_first_render(report)
    records = pool.map(tasks, 3600.0, "smoke source")
    episodes = {
        r["episode_id"]: {"npz_sha256": r["npz_sha256"], "meta_sha256": r["meta_sha256"]}
        for r in records
    }
    sha = hz.seal_far_corpus(
        corpus,
        plan,
        episodes,
        {"revision": report["revision"], "k1_report_sha256": None, "smoke": True, "for": pt.TASK},
    )
    return {"outcome": "SOURCE-SEALED", "source": {"manifest_sha256": sha, "path": str(corpus)}}


def open_source(args):
    """The source corpus (train and val only; the reader refuses the test roots)."""
    from embodied_jepa import first_policy_v2_runtime as rt2

    sha = args.source_sha256 if args.smoke else pt.SOURCE_CORPUS["manifest_sha256"]
    if not args.smoke and args.source_sha256 not in (None, sha):
        raise lp.GuardError("G-evidence: --source-sha256 is not the sealed apple-far-shift-v2")
    reader = rt2.CorpusReader(Path(args.source), sha, splits=pt.SOURCE_CORPUS["read_splits"])
    m = reader.manifest
    if m.get("protocol") != lp.PROTOCOL or m.get("corpus") != lp.CORPUS:
        raise lp.GuardError("G-evidence: the source is not a TASK-074 apple-far-shift-v2 corpus")
    if bool(m.get("provenance", {}).get("smoke")) != bool(args.smoke):
        raise lp.GuardError("G-evidence: the source's smoke flag does not match this stage")
    test_ids = sorted(m["splits"]["test"])
    excluded = hashlib.sha256(json.dumps(test_ids).encode()).hexdigest()
    if not args.smoke:
        if lp.plan_digest(m["plan"]) != pt.SOURCE_CORPUS["plan_sha256"]:
            raise lp.GuardError("G-evidence: the source plan differs from apple-far-shift-v2's")
        if excluded != pt.SOURCE_CORPUS["excluded_test_ids_sha256"]:
            raise lp.GuardError("G-evidence: the source's test split is not the recorded one")
    return reader, {
        "path": str(reader.corpus),
        "manifest_sha256": sha,
        "excluded_test_ids_sha256": excluded,
    }


def stage_render(report, manifest, args, clock):
    """Smoke only: the TASK-075 views store of the smoke source (obs_ceiling_v2_runtime)."""
    reader, source = open_source(args)
    report["_reader"] = reader
    out = Path(args.views)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    ids = [e for s in pt.SOURCE_CORPUS["read_splits"] for e in reader.manifest["splits"][s]]
    tasks = []
    for episode_id in ids:
        record = reader.manifest["episodes"][episode_id]
        meta_path = reader.corpus / "episodes" / f"{episode_id}.json"
        if hz.sha256_file(meta_path) != record["meta_sha256"]:
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
            }
        )
        reader.decoded.add(episode_id)
    (out / "episodes").mkdir(parents=True)
    pool = make_pool(report, args)
    mark_first_render(report)
    records = pool.map(tasks, 3600.0, "smoke views")
    bad = [
        r["episode_id"]
        for r in records
        if not all(r["arrays_equal"].values()) or r["frame_check"]["outside_rule"]
    ]
    if bad:
        raise lp.GuardError(f"G-repro-render: {bad} differ from the smoke source")
    views_manifest = {
        "corpus": "apple-far-shift-v2-views",
        "protocol": oc.PROTOCOL,
        "task": pt.TASK,
        "privileged_scripted_collector": True,
        "learned_control": False,
        "source": source,
        "views": list(oc.RENDERED_VIEWS),
        "hidden_kinds": list(oc.HIDDEN_KINDS),
        "steps": {"render": list(oc.RENDER_STEPS), "hidden": list(oc.HIDDEN_STEPS)},
        "splits": {s: list(reader.manifest["splits"][s]) for s in pt.SOURCE_CORPUS["read_splits"]},
        "episodes": {
            r["episode_id"]: {"npz_sha256": r["npz_sha256"], "bytes": r["bytes"]} for r in records
        },
        "provenance": {"revision": report["revision"], "smoke": True},
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(views_manifest, indent=1, sort_keys=True) + "\n")
    return {
        "outcome": "VIEWS-SEALED",
        "views": {"path": str(out), "manifest_sha256": hz.sha256_file(path), "roots": len(records)},
    }


# ----- Stage O ------------------------------------------------------------------------------------
def tau_for(args) -> tuple[float, dict]:
    if args.smoke:
        return SMOKE["tau_cm"], dict(oc.TAU_MEASURED["counts"])
    if pt.K0_MEASURED is None:
        raise lp.GuardError("G-order: Stage O needs K0's frozen tau_re")
    return pt.K0_MEASURED["tau_re_cm"], pt.K0_MEASURED["counts"]


def stage_offline(report, manifest, args, clock):
    from embodied_jepa import plate_twin_v2_offline as off
    from embodied_jepa import pretrained_encoder as pe

    reader, source = open_source(args)
    report["_reader"] = reader
    views_sha = args.views_sha256 if args.smoke else pt.VIEWS_STORE["manifest_sha256"]
    views = off.ViewsReader(Path(args.views), views_sha)
    rt.gpu_guard(
        report,
        min_free_gib=pt.GPU["min_free_gib"],
        process_cap_gib=pt.GPU["process_cap_gib"],
        require_lock=pt.GPU["require_lock"],
    )
    tau_cm, tau_counts = tau_for(args)
    encoder, floor = pe.load_pretrained(), pe.random_init()
    kwargs = dict(
        tau_re_cm=tau_cm,
        tau_counts=tau_counts,
        encoder=encoder,
        floor=floor,
        device="cuda",
        log=log,
    )
    result = off.offline_core(report, reader, views, clock, **kwargs)
    fit, closed = result.pop("_fit"), result.pop("_closed")
    folder = Path(args.output).resolve()  # workers read the readouts by this path
    errors = folder / "errors.npz"
    np.savez(
        errors,
        clusters=fit["clusters"],
        steps=fit["steps"],
        hidden_clusters=fit["hidden_clusters"],
        signed=fit["signed"],
        **{f"e__{k}": v for k, v in fit["errors"].items()},
    )
    result["errors_sha256"] = hz.sha256_file(errors)
    result["readouts"] = {
        name: {
            "path": str(folder / f"{name}.npz"),
            "sha256": off.save_readout(folder / f"{name}.npz", r),
        }
        for name, r in closed.items()
    }
    if reader.test_split_decoded:
        raise lp.GuardError("Q-split: a test root was decoded")
    if args.smoke and args.scale:
        del fit, closed
        import gc

        gc.collect()
        counts = {
            "train": pt.SOURCE_CORPUS["split_sizes"]["train"],
            "val": pt.SOURCE_CORPUS["split_sizes"]["val"],
        }
        cycled = rt.CycledReader(reader, counts)
        probe = rt.scale_probe(
            off.offline_core,
            report,
            cycled,
            rt.SlotProxy(views, cycled.source),
            clock,
            watch=report["_watch"],
            ceiling_gib=pt.MEMORY["ceiling_gib"],
            margin_gib=pt.SCALE_MARGIN_GIB,
            **kwargs,
        )
        out = probe.pop("result")
        probe["roots"] = out["roots"]
        probe["featurisation"] = out["featurisation"]
        result["scale"] = probe
    import torch

    if torch.cuda.is_initialized():
        report["gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / GIB
        report["gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / GIB
    row = result["statistics"]["decision"]["row"]
    return {"outcome": "OFFLINE-SMOKE" if args.smoke else row, "offline": result}


# ----- the closed loop: D, S/U, K-pred ------------------------------------------------------------
def readout_config(args) -> dict:
    """The closed-loop readouts: from the recorded Stage O report (sha256-checked, O-PASS for a
    real run)."""
    if not args.offline_report:
        return {}
    if hz.sha256_file(args.offline_report) != args.offline_sha256:
        raise lp.GuardError("G-evidence: the offline report differs from its sha256")
    upstream = json.loads(Path(args.offline_report).read_text())
    want = "OFFLINE-SMOKE" if args.smoke else "O-PASS"
    if upstream.get("protocol") != pt.PROTOCOL or upstream.get("outcome") != want:
        raise lp.GuardError(f"G-order: the offline report is not a recorded {want}")
    if not args.smoke and upstream.get("tracked_tree_dirty") is not False:
        raise lp.GuardError("G-evidence: the offline report did not run on a clean tree")
    readouts = upstream["stages"]["offline"]["readouts"]
    config = {}
    for name in ("r_plate", "r_plate_floor"):
        path = readouts[name]["path"]
        config[name] = path
        config[f"{name}_sha256"] = readouts[name]["sha256"]
    return config


def clock_for(args, seeds, resets) -> dict:
    """H-clock's frozen targets (K0); in a smoke, the same rule on the smoke seeds (stand-in)."""
    if not args.smoke:
        return {int(k): v for k, v in pt.K0_MEASURED["clock_targets"].items()}
    plate_at = {
        s: {
            t: (np.asarray(resets[s]["plate_xy"]) + np.asarray(resets[s]["shift_m"])).tolist()
            for t in pt.DECISION_STEPS
        }
        for s in seeds
    }
    return pt.clock_targets(plate_at)


def stage_dev(report, manifest, args, clock):
    config = readout_config(args)
    if not args.smoke and not config:
        raise lp.GuardError("G-order: Stage D needs the recorded O-PASS")
    seeds, resets = cohort("D", args, SMOKE["dev_seeds"])
    pool = make_pool(report, args, config)
    est = estimates_for(report, pool, args, seeds, resets)
    per = {}
    for arm in pt.D_ARMS:
        clock.check(arm)
        records = run_arm(pool, attempt_tasks(arm, seeds, resets, est, role="D"), arm)
        per[arm] = {
            "per_reset": successes(records),
            "count": sum(successes(records)),
            "timing": timing(records),
            "attempts": hz.strip(records),
        }
    decision = pt.decide_dev({a: per[a]["per_reset"] for a in pt.D_ARMS})
    return {
        "outcome": "DEV-SMOKE" if args.smoke else decision["row"],
        "dev": {"seeds": list(seeds), "arms": per, "decision": decision},
    }


def stage_gated(report, manifest, args, clock):
    from embodied_jepa import place_planner as pp

    config = readout_config(args)
    if not args.smoke and not config:
        raise lp.GuardError("G-order: the gated stage needs the recorded O-PASS and D-PASS")
    s_seeds, s_resets = cohort("S", args, SMOKE["s_seeds"])
    u_args = argparse.Namespace(**{**vars(args), "seeds": args.u_seeds})
    u_seeds, u_resets = cohort("U", u_args, SMOKE["u_seeds"])
    pool = make_pool(report, args, config)
    est = estimates_for(report, pool, args, s_seeds + u_seeds, s_resets | u_resets)
    targets = clock_for(args, s_seeds, s_resets)
    s, u, raw = {}, {}, {}
    for arm in pt.S_ARMS:
        clock.check(f"S {arm}")
        extra = {"clock": targets} if arm == "H-clock" else {}
        records = run_arm(
            pool, attempt_tasks(arm, s_seeds, s_resets, est, role="S", **extra), f"S {arm}"
        )
        raw[arm] = records
        s[arm] = {
            "per_reset": successes(records),
            "count": sum(successes(records)),
            "timing": timing(records),
            "attempts": hz.strip(records),
        }
    for arm in pt.U_ARMS:
        clock.check(f"U {arm}")
        records = run_arm(pool, attempt_tasks(arm, u_seeds, u_resets, est, role="U"), f"U {arm}")
        u[arm] = {
            "per_reset": successes(records),
            "count": sum(successes(records)),
            "timing": timing(records),
            "attempts": hz.strip(records),
        }
    first = s_seeds[: pt.RERUN["seeds"]]
    again = run_arm(
        pool, attempt_tasks(pt.RERUN["arm"], first, s_resets, est, role="S"), "determinism re-run"
    )
    reruns = [
        pp.rerun_verdict(a, b)
        for a, b in zip(raw[pt.RERUN["arm"]][: len(first)], again, strict=True)
    ]
    void = any(r["void"] for r in reruns)
    if void and not args.smoke:
        raise lp.GuardError("G-rerun: H-twin's determinism re-run differs")
    decision = pt.decide_gated(
        {a: s[a]["per_reset"] for a in pt.S_ARMS}, {a: u[a]["per_reset"] for a in pt.U_ARMS}
    )
    return {
        "outcome": "GATED-SMOKE" if args.smoke else decision["row"],
        "gated": {
            "s_seeds": list(s_seeds),
            "u_seeds": list(u_seeds),
            "S": s,
            "U": u,
            "rerun": [
                {k: v for k, v in r.items() if k != "differences"}
                | {"differences": {k: str(v) for k, v in r["differences"].items()}}
                for r in reruns
            ],
            "decision": decision,
        },
    }


def cell_spec(name: str, args, resets: dict) -> dict:
    """K-pred's cell for every seed: M's vector, or A's (kappa, L, s1); Stage-0 remedy settings
    (--kpred-L, --kpred-s1) are smoke-only overrides of cell A."""
    if name in pt.M_CELLS:
        return {
            s: {"name": name, "s0": pt.KPRED["s0"], "s1": pt.KPRED["s1"], "vector": r["moving_m"]}
            for s, r in resets.items()
        }
    lag = int(args.kpred_L or pt.CELL_A["L"])
    s1 = int(args.kpred_s1 or pt.CELL_A["s1"])
    pt.check_lag(lag)
    pt.check_kappa(pt.CELL_A["kappa"])
    if not pt.LOWER_START <= s1 < pt.PLACE_END or (s1 - pt.CELL_A["s1"]) % pt.STAGE0_S1_STEP:
        raise lp.GuardError("G-s1: s1 is 525 or later in steps of 10, before the place's end")
    return {
        s: {"name": "A", "s0": pt.KPRED["s0"], "s1": s1, "kappa": pt.CELL_A["kappa"], "L": lag}
        for s in resets
    }


def kpred_summary(name: str, by_arm: dict) -> dict:
    """The Stage-0 mechanics of a cell: contacts, refusals, remaining motion, m2, times."""
    out = {}
    first_contacts = []
    for arm, records in by_arm.items():
        rows = [r["kpred"] for r in records]
        refused = sum(r["blocked"] is not None for r in records)
        remaining = [
            k["remaining_cm"]
            for k, r in zip(rows, records, strict=True)
            if r["blocked"] is None and k["remaining_cm"] is not None
        ]
        m2 = [k["m2_cm"] for k in rows if k["m2_cm"] is not None]
        palm_moves = [
            k["palm_remaining_cm"] for k in rows if k.get("palm_remaining_cm") is not None
        ]
        contacts = [k["first_contact_step"] for k in rows if k["first_contact_step"] is not None]
        first_contacts += contacts
        item = {
            "count": sum(successes(records)),
            "attempts": len(records),
            "refused": refused,
            "refused_steps": [k["refused"]["step"] for k in rows if k["refused"]],
            "contact_before_s1": sum(bool(k["contact_before_s1"]) for k in rows),
            "first_contact_min": min(contacts) if contacts else None,
            "remaining_cm": {
                "median": float(np.median(remaining)) if remaining else None,
                "values": remaining,
            },
            "m2_cm": {
                "median": float(np.median(m2)) if m2 else None,
                "max": float(np.max(m2)) if m2 else None,
            },
            "palm_remaining_cm": {
                "median": float(np.median(palm_moves)) if palm_moves else None,
                "max": float(np.max(palm_moves)) if palm_moves else None,
            },
            "seconds": timing(records),
        }
        looks = [d.get("lookahead") for r in records for d in r["decisions"] if d.get("lookahead")]
        if looks:
            iters = [len(x["iterations"]) for x in looks]
            item["lookahead"] = {
                "decisions": len(looks),
                "converged": sum(bool(x["converged"]) for x in looks),
                "iterations_median": float(np.median(iters)),
                "iterations_max": int(max(iters)),
                "branch_steps_max": max(
                    int(r["kpred"].get("lookahead_branch_steps", 0)) for r in records
                ),
            }
        out[arm] = item
    verdict = None
    if name == "A" and "H-final" in out:
        f = out["H-final"]
        verdict = pt.stage0_cell_a_verdict(f["remaining_cm"]["values"], f["refused"], f["attempts"])
    return {
        "arms": out,
        "first_contact_min": min(first_contacts) if first_contacts else None,
        "contact_before_s1_any": any(v["contact_before_s1"] for v in out.values()),
        "cell_a_verdict": verdict,
    }


def stage_kpred(report, manifest, args, clock):
    name = args.cell
    if name == "A" and pt.CELL_A["status"] != "kept":
        raise lp.GuardError("G-cell: cell A was removed at Stage 0")
    config = readout_config(args)
    if not args.smoke and not config:
        raise lp.GuardError("G-order: K-pred needs the recorded O-PASS")
    seeds, resets = cohort(name, args, SMOKE["kpred_seeds"][name])
    cells = cell_spec(name, args, resets)
    arms = KPRED_ARMS_OF(name, args, config)
    pool = make_pool(report, args, config)
    est = estimates_for(report, pool, args, seeds, resets)
    tau_cm = SMOKE["tau_cm"] if args.smoke else pt.K0_MEASURED["tau_re_cm"]
    by_arm, per = {}, {}
    for arm in arms:
        clock.check(arm)
        tasks = attempt_tasks(
            arm, seeds, resets, est, role=name, per_seed=lambda s: {"cell": cells[s]}, tau_cm=tau_cm
        )
        records = run_arm(pool, tasks, f"{name} {arm}")
        by_arm[arm] = records
        per[arm] = {
            "per_reset": successes(records),
            "count": sum(successes(records)),
            "attempts": hz.strip(records),
        }
        log(f"{name} {arm}: {per[arm]['count']}/{len(seeds)}")
    result = {
        "cell": name,
        "seeds": list(seeds),
        "spec": cells[seeds[0]] | {"vector": None},
        "arms": per,
        "stage0": kpred_summary(name, by_arm),
    }
    if not args.smoke and name == "A":
        result["m2_star"] = pt.median_upper(
            [r["kpred"]["m2_cm"] for r in by_arm["H-final"] if r["kpred"]["m2_cm"] is not None]
        )
        for arm in ("H-now", "H-twin", "H-cv", "H-rule"):
            result.setdefault("paired", {})[f"H-final - {arm}"] = pt.paired_interval(
                per["H-final"]["per_reset"], per[arm]["per_reset"]
            )
    return {"outcome": "KPRED-SMOKE" if args.smoke else "KPRED-DONE", "kpred": result}


def KPRED_ARMS_OF(name: str, args, config: dict) -> tuple:  # noqa: N802
    arms = pt.KPRED_ARMS["A" if name == "A" else "M"]
    if args.arms:
        chosen = tuple(a.strip() for a in args.arms.split(","))
        if not args.smoke or not set(chosen) <= set(arms):
            raise lp.GuardError("G-arms: a real K-pred cell runs every arm of its cell")
        arms = chosen
    needs_readout = set(arms) & set(pt.IMAGE_READING_ARMS)
    if needs_readout and not config:
        raise lp.GuardError(f"G-readout: {sorted(needs_readout)} need the closed-loop R-plate")
    return arms


STAGES = {
    "k0": stage_k0,
    "offline": stage_offline,
    "dev": stage_dev,
    "gated": stage_gated,
    "kpred": stage_kpred,
    "source": stage_source,
    "render": stage_render,
}


def run(args) -> dict:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    report = {
        "protocol": pt.PROTOCOL,
        "task": pt.TASK,
        "mode": args.mode,
        "smoke": bool(args.smoke),
        "outcome": None,
        "stages": {},
        "argv": sys.argv[1:],
        "paths": {"output": str(output), "evidence": str(Path(args.evidence).resolve())},
        "started_utc": hz.utc(),
        "no_world_model": True,
    }
    clock = hz.Clock(pt.CAPS_SECONDS["invocation"])
    watch = rt.MemoryWatch(
        pt.MEMORY["ceiling_gib"] * GIB, pt.MEMORY["sample_seconds"], measure=pt.MEMORY["measure"]
    )
    guards = rt.install_guards(watch=watch, log=log)  # F11: the first signal is recorded
    report["_watch"] = watch
    try:
        manifest = preflight(report, args, gpu=args.mode == "offline")
        if args.mode == "preflight":
            pool = make_pool(report, args)
            if not args.truth_estimates:
                hz.refit_p_readout(report, pool, Path(args.evidence), report["_run1"])
            result = {"outcome": "PREFLIGHT-READY"}
        else:
            result = STAGES[args.mode](report, manifest, args, clock)
        end_checks(report, manifest, args.smoke)
        for key, value in result.items():
            if key == "outcome":
                report["outcome"] = value
            else:
                report["stages"][key] = value
    except BaseException as error:  # noqa: BLE001 - every failure, signal included, is V
        guards.void(report, error)
    finally:
        pool = report.pop("_pool", None)
        guards.finish(report, pool)
        if pool is not None:
            report["pool_close"] = pool.close_record
        report.pop("_watch", None)
        report.pop("_p_readout", None)
        reader = report.pop("_reader", None)
        if reader is not None:
            report["decoded_episodes"] = len(reader.decoded)
            report["test_split_decoded"] = reader.test_split_decoded
        report.pop("_run1", None)
        report["ended_utc"] = hz.utc()
        report["total_seconds"] = clock.elapsed()
        hz.write_report(output / "report.json", report)
        log(f"report written: outcome {report.get('outcome')}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--evidence",
        required=True,
        help="the root that holds TASK-072 run-1's outputs/, checkpoints/, data/",
    )
    for name in (
        "source",
        "source-sha256",
        "views",
        "views-sha256",
        "offline-report",
        "offline-sha256",
    ):
        parser.add_argument(f"--{name}")
    parser.add_argument("--cell", choices=pt.KPRED_ROLES)
    parser.add_argument("--smoke", action="store_true", help="smoke seeds only; nothing is read")
    parser.add_argument("--scale", action="store_true", help="offline smoke: the scale probe")
    parser.add_argument(
        "--truth-estimates",
        action="store_true",
        help="smoke only: P-3's estimates from the reset truth (no G-repro)",
    )
    parser.add_argument("--seeds", help="smoke only: a smoke-seed range a-b")
    parser.add_argument("--u-seeds", help="smoke only: the gated smoke's U range a-b")
    parser.add_argument("--arms", help="smoke only: a subset of the cell's arms")
    parser.add_argument("--kpred-L", type=int, help="smoke only: cell A's remedy L")
    parser.add_argument("--kpred-s1", type=int, help="smoke only: cell A's remedy s1")
    parser.add_argument("--workers", type=int, default=None, help="smoke only (1-6)")
    args = parser.parse_args(argv)
    smoke_only = (
        "scale",
        "truth_estimates",
        "seeds",
        "u_seeds",
        "arms",
        "kpred_L",
        "kpred_s1",
        "workers",
    )
    used = [n for n in smoke_only if getattr(args, n) not in (None, False)]
    if used and not args.smoke:
        parser.error(f"{', '.join(used)} are for smoke runs only")
    if args.workers is not None and not 1 <= args.workers <= 6:
        parser.error("--workers is from 1 to 6")
    if args.mode in SMOKE_ONLY and not args.smoke:
        parser.error(f"{args.mode} is a smoke-only mode")
    if args.mode == "source" and args.source is None:
        parser.error("source needs --source")
    if args.mode == "kpred" and not args.cell:
        parser.error("kpred needs --cell")
    if args.mode in ("render", "offline") and (args.source is None or args.views is None):
        parser.error(f"{args.mode} needs --source and --views")
    if args.smoke and args.mode in ("render", "offline") and args.source_sha256 is None:
        parser.error("a smoke render or offline needs --source-sha256")
    if args.smoke and args.mode == "offline" and args.views_sha256 is None:
        parser.error("an offline smoke needs --views-sha256")
    if bool(args.offline_report) != bool(args.offline_sha256):
        parser.error("--offline-report and --offline-sha256 go together")
    report = run(args)
    return 0 if report.get("outcome") not in (None, "V") else 1


if __name__ == "__main__":
    sys.exit(main())
