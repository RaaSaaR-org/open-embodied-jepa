"""TASK-077's runner: every stage of ``docs/experiments/apple_lewm_c1m_v2.md`` (STATUS DRAFT).

Frozen block ``src/embodied_jepa/lewm_c1m_v2.py``; workers ``lewm_c1m_v2_runtime.py``; offline
stages ``lewm_c1m_v2_offline.py``; training ``lewm_c1m_v2_train.py``. Guards from
``embodied_jepa.run_tools`` and TASK-076's harness module; this runner loads no other script.

One invocation runs one stage and writes ``<output>/report.json`` (it refuses an existing output):

* ``tests``      -- the full pytest suite at HEAD, recorded (``pre_run_tests.json``; G-tests for GPU
  jobs, which verify this record instead of holding the GPU lock through the suite);
* ``k0``         -- K0 on K (66000-66031), CPU, before the freeze (on a reported GO only);
* ``corpus``     -- Stage C, the 2 000-root corpus (CPU);
* ``featurise``  -- Stage O's featurisation (CUDA, through ``scripts/gpu_run.sh``);
* ``readouts``   -- Stage O's admission and the train-only fits (CPU);
* ``train``      -- one Stage T job (CUDA, through ``scripts/gpu_run.sh``): ``--job cal-W``,
  ``cal-N``, ``W-<seed>`` or ``N-<seed>``;
* ``plan``       -- Stage T's budget rule, G1's bars and the truncation controls (CPU), after both
  calibration jobs;
* ``gates``      -- Stage G on the gate split (CPU, one torch thread) and the primary seed;
* ``closed``     -- Stage D (``--cohort D``) or Stage S (``--cohort S``) (CPU);
* ``simulate``   -- the Stage-0 simulations (CPU; no simulator);
* ``scale``      -- Stage 0's scale probe of a Stage T job at the real sizes on synthetic features
  (CUDA, through ``scripts/gpu_run.sh``): the measured per-update time with the contiguous gather
  and prefetch, the gather alone, the peak PSS, and a bit-identity re-run.

Every CPU stage runs the full test suite itself first (G-tests); every stage checks the clean
tree, TASK-076's pins and C1's and C1-M's code (G-hash). While the protocol is DRAFT, only
``tests``, ``simulate``, ``scale``, ``k0`` and ``--debug`` runs are allowed (§7: nothing from C, D,
S or the corpus is simulated before its GO, and those stages run only after the freeze).
``--debug`` simulates debug seeds 66900-66999 only, at small sizes, with declared stand-ins;
nothing in it is read.
"""

from __future__ import annotations

import os

os.environ.update(
    {
        "MKL_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": "6",
        "MKL_NUM_THREADS": "6",
        "OPENBLAS_NUM_THREADS": "16",
    }
)

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import devices  # noqa: E402
from embodied_jepa import first_policy_v2_linux as fpl  # noqa: E402
from embodied_jepa import lewm_c1m_v2 as lm  # noqa: E402
from embodied_jepa import lewm_c1m_v2_offline as off  # noqa: E402
from embodied_jepa import lewm_c1m_v2_train as tr  # noqa: E402
from embodied_jepa import lewm_next_c1 as c1  # noqa: E402
from embodied_jepa import lewm_planner_v2 as lp  # noqa: E402
from embodied_jepa import plate_twin_v2_harness as hz  # noqa: E402
from embodied_jepa import run_guards as rg  # noqa: E402
from embodied_jepa import run_tools as rt  # noqa: E402

fpl.configure_headless()
GIB = 2**30
log = hz.log
MAP_CAP_SECONDS = 3600.0
QUIET_POLL_SECONDS = 30.0
QUIET_WAIT_CAP_SECONDS = 4 * 3600.0
CORPUS_CHUNK = 60  # corpus roots per pool map (bounds the frames held at once)
STAGES = (
    "tests",
    "k0",
    "corpus",
    "featurise",
    "readouts",
    "train",
    "plan",
    "gates",
    "closed",
    "simulate",
    "scale",
)
CPU_STAGES = ("k0", "corpus", "readouts", "plan", "gates", "closed", "simulate")
GPU_STAGES = ("featurise", "train", "scale")
SIM_STAGES = ("k0", "corpus", "gates", "closed")  # stages with a simulator pool
DRAFT_ALLOWED = ("tests", "simulate", "scale", "k0")
DEBUG = {  # declared debug stand-ins (nothing in a debug run is read)
    "tau_commit_cm": lm.TAU_COMMIT_RECORD_CM,
    "calibration_updates": 400,
    "calibration_select_every": 20,
    "model_updates": 200,
    "model_select_every": 10,
    "tau_counts": {"0.0": 32, "0.5": 31, "1.0": 29, "1.5": 25, "2.0": 20, "3.0": 12},
}
STAGE_FIELDS = {
    "k0": ("tau", "ceiling", "r_k", "history", "proxies", "decision"),
    "corpus": ("split", "roots", "excluded", "sealed", "decision"),
    "featurise": ("featurisation", "anchor"),
    "readouts": ("admission", "fits", "decision"),
    "train": ("job", "record", "checkpoint"),
    "plan": ("calibration", "budget", "g1", "decision"),
    "gates": ("primary_seed", "per_seed", "offline_aims", "decision"),
    "closed": ("arms", "determinism", "decision"),
    "simulate": ("simulations",),
    "scale": ("gather", "probe", "bit_identity", "caps"),
    "tests": ("tests",),
}


class Pool(rg.BoundedPool):
    """``run_guards.BoundedPool`` on TASK-077's worker; a dead worker or a cap is a GuardError;
    every map of attempts runs TASK-071's G-look check (as TASK-076's harness pool)."""

    def __init__(self, workers: int, config: dict):
        from embodied_jepa import lewm_c1m_v2_runtime as wrt

        super().__init__(workers, wrt.run_task, initializer=wrt.worker_init, initargs=(config,))
        self.look_reference = None

    def map(self, tasks: list[dict], cap: float, what: str) -> list[dict]:
        try:
            out = super().map(tasks, cap, what)
        except (rg.WorkerDied, rg.MapCapExceeded) as error:
            raise lp.GuardError(str(error)) from error
        self.look_reference = hz.check_look_states(out, self.look_reference)
        return out


class Fields:
    """G-sentinel: every stage field starts as ``"not evaluated"``; only ``set`` changes one."""

    def __init__(self, report: dict, stage: str):
        self.report, self.names = report, STAGE_FIELDS[stage]
        self.evaluated: set[str] = set()
        report["fields"] = lm.sentinel_fields(self.names)

    def set(self, name: str, value) -> None:
        if name not in self.names:
            raise KeyError(name)
        if lm.is_missing(value):
            raise lm.GuardError(f"G-sentinel: {name} set to a missing value")
        self.report["fields"][name] = value
        self.evaluated.add(name)

    def get(self, name: str):
        return self.report["fields"][name]

    def check(self) -> None:
        lm.check_sentinel(self.report["fields"], self.evaluated, self.names)


# ----- preflight ----------------------------------------------------------------------------------
def git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def check_code(report: dict) -> None:
    """G-hash: C1's and C1-M's code files are byte-identical to main at the draft's merge;
    TASK-077's own files are committed."""
    carried = {}
    for path in lm.CARRIED_CODE_FILES:
        want = lm.CARRIED_CODE_BLOBS[path]  # the blob at the reference, recorded
        got = git("hash-object", path)
        if want != got:
            raise lp.GuardError(f"G-hash: {path} differs from {lm.CARRIED_CODE_REFERENCE[:7]}")
        carried[path] = got
    for path in lm.OWN_FILES:
        try:
            git("ls-files", "--error-unmatch", path)
        except subprocess.CalledProcessError as error:
            raise lp.GuardError(f"G-hash: {path} is not committed") from error
    report["carried_code_git_blobs"] = carried
    report["own_code_git_blobs"] = {p: git("hash-object", p) for p in lm.OWN_FILES}


def summary_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def summary_ok(summary: str) -> bool:
    low = summary.lower()
    return " passed" in low and "failed" not in low and "error" not in low


def run_full_tests() -> dict:
    """G-tests (CPU stages): the full suite from the worktree root at HEAD (CUDA hidden)."""
    started = hz.utc()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    record = {
        "revision": hz.revision(),
        "tracked_tree_dirty": bool(hz.tracked_tree_dirty()),
        "exit_status": int(proc.returncode),
        "summary": summary_line(proc.stdout),
        "started_utc": started,
        "finished_utc": hz.utc(),
        "command": "python -m pytest -q -p no:cacheprovider (CUDA_VISIBLE_DEVICES='')",
    }
    if record["exit_status"] != 0 or not summary_ok(record["summary"]):
        raise lp.GuardError(f"G-tests: the suite did not pass: {record['summary']!r}")
    return record


def verify_tests_record(path) -> dict:
    """G-tests (GPU jobs): a recorded run at HEAD, exit 0, no failure, after HEAD's commit."""
    record = json.loads(Path(path).read_text())
    tests = record.get("fields", {}).get("tests", record)
    head = hz.revision()
    committed = git("log", "-1", "--format=%cI", "HEAD")
    from datetime import datetime

    finished = datetime.fromisoformat(tests["finished_utc"].replace("Z", "+00:00"))
    checks = {
        "revision": tests["revision"] == head,
        "exit_status": tests["exit_status"] == 0,
        "summary": summary_ok(tests["summary"]),
        "after_commit": finished > datetime.fromisoformat(committed),
        "clean": not tests.get("tracked_tree_dirty", True),
    }
    if not all(checks.values()):
        raise lp.GuardError(f"G-tests: the recorded run does not qualify: {checks}")
    return {"path": str(Path(path).resolve()), "checks": checks, "record": tests}


def wait_quiet(report: dict, args) -> None:
    rule = lm.QUIET_MACHINE
    started = time.monotonic()
    while True:
        one, five, _ = os.getloadavg()
        if one <= rule["max_load_average_1min"] and five <= rule["max_load_average_5min"]:
            break
        if args.debug:
            break
        if time.monotonic() - started > QUIET_WAIT_CAP_SECONDS:
            raise lp.GuardError(f"G-quiet: the machine stayed busy ({one:.2f}, {five:.2f})")
        log(f"waiting for a quiet machine (load {one:.2f}, {five:.2f})")
        time.sleep(QUIET_POLL_SECONDS)
    report["quiet_machine"] = {
        "rule": rule,
        "waited_seconds": time.monotonic() - started,
        "load_average_at_start": list(os.getloadavg()),
    }


def check_disk(report: dict, path: Path, minimum_gib: float) -> None:
    free = shutil.disk_usage(path).free / GIB
    report.setdefault("disk_free_gib", {})[str(path)] = free
    if free < minimum_gib:
        raise lp.GuardError(f"G-disk: {free:.1f} GiB free < {minimum_gib} GiB")


def preflight(report: dict, args) -> dict:
    """Every stage's common guards."""
    rt.assert_local_import(ROOT, report)
    manifest = json.loads((ROOT / lm.TASK076_MANIFEST).read_text())
    report["task076_pins_at_preflight"] = len(hz.check_pins(manifest["hashes"]))
    check_code(report)
    dirty = hz.tracked_tree_dirty()
    report["revision"], report["tracked_tree_dirty"] = hz.revision(), bool(dirty)
    hz.check_clean(dirty)
    report["protocol_status"] = lm.STATUS
    report["frozen_sha256_at_run"] = lm.frozen_sha256()
    if lm.STATUS != "FROZEN" and args.stage not in DRAFT_ALLOWED and not args.debug:
        raise lp.GuardError(f"G-frozen: {args.stage} runs only after the freeze (STATUS FROZEN)")
    report["thread_env"] = {k: os.environ.get(k) for k in lm.THREAD_ENV}
    if report["thread_env"] != lm.THREAD_ENV:
        raise lp.GuardError(f"G-threads: {report['thread_env']} is not {lm.THREAD_ENV}")
    lm.check_seed_ranges()
    start_min = lm.DISK_MIN_START_GIB.get({"corpus": "C", "featurise": "O"}.get(args.stage, ""))
    check_disk(report, ROOT, start_min or lm.DISK_MIN_GIB)
    if args.stage in CPU_STAGES:
        wait_quiet(report, args)
    if args.stage in CPU_STAGES and args.stage != "tests":
        if args.debug and args.debug_skip_tests:
            report["g_tests"] = {"skipped": "debug run with --debug-skip-tests (nothing is read)"}
        else:
            report["g_tests"] = run_full_tests()
    if args.stage in GPU_STAGES:
        report["g_tests"] = verify_tests_record(args.tests_record)
    return manifest


def sim_preflight(report: dict, args, manifest: dict) -> None:
    import mujoco
    import torch

    from embodied_jepa import pretrained_encoder as pe

    report["platform"] = fpl.check_platform()
    available = hz.mem_available_bytes()
    report["mem_available_at_start_gib"] = available / GIB
    if available < (lm.MEMORY["ceiling_gib"] + lm.MEMORY["headroom_gib"]) * GIB:
        raise lp.GuardError(f"G-memory: MemAvailable {available / GIB:.1f} GiB is too low")
    torch.set_num_threads(6)
    devices.configure_determinism("cuda", strict=True)  # no CUDA context is created
    if mujoco.__version__ != manifest["mujoco_version"]:
        raise lp.GuardError(f"G-hash: MuJoCo {mujoco.__version__} is not the pinned version")
    report["environment"] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
    }
    digests = {
        "pretrained": pe.weights_digest(pe.load_pretrained()),
        "floor": pe.weights_digest(pe.random_init()),
    }
    if digests != manifest["encoder_digests"]:
        raise lp.GuardError("G-weights: an encoder digest differs from TASK-076's pin")
    report["encoder_digests"] = digests
    found = hz.check_evidence(Path(args.evidence))
    report["evidence"] = {"root": str(args.evidence), "sha256": found["sha256"]}
    report["_run1"] = found["run1"]


# ----- cohorts and attempts ----------------------------------------------------------------------
class Cohorts:
    """Seeds, resets and P-3's post-look estimates (G-repro), rendered lazily per role."""

    def __init__(self, args, pool, p_readout, encoder, report):
        self.debug = bool(args.debug)
        self.pool, self.p_readout, self.encoder, self.report = pool, p_readout, encoder, report
        self.resets: dict[int, dict] = {}
        self.est: dict[int, dict] = {}
        self.digests: dict[str, str] = {}

    def seeds(self, role: str) -> tuple[int, ...]:
        seeds = lm.check_seeds(role, lm.seeds_of(role, debug=self.debug), debug=self.debug)
        if role not in self.digests:
            resets = {s: lm.reset_of(s) for s in seeds}
            self.resets |= resets
            self.digests[role] = lp.plan_digest({str(s): v for s, v in resets.items()})
            est = hz.cohort_estimates(
                self.pool, self.p_readout, self.encoder, seeds, resets, 1800.0
            )
            disagreements = est.pop("_render_disagreements", {})
            self.report.setdefault("render_disagreements", {})[role] = disagreements
            self.est |= est
            self.report["resets_digest"] = dict(self.digests)
        return seeds


def attempt_tasks(arm: str, seeds, co: Cohorts, per_seed=None, **extra) -> list[dict]:
    tasks = []
    for s in seeds:
        task = {
            "kind": "attempt",
            "arm": arm,
            "seed": s,
            "reset": co.resets[s],
            "estimates": co.est[s]["estimates"],
            "expected_frame_sha256": co.est[s]["frame_sha256"],
            "expected_state_sha256": co.est[s]["state_sha256"],
            "wall_seconds": lm.CAPS_SECONDS["per_attempt"],
            "move_offset": lm.move_offset(s).tolist(),
        }
        tasks.append(task | (per_seed(s) if per_seed else {}) | extra)
    return tasks


def run_arm(pool, tasks, what: str) -> list[dict]:
    records = pool.map(tasks, MAP_CAP_SECONDS, what)
    arm = tasks[0]["arm"]
    if arm not in lm.PRIVILEGED_ARMS:
        for r in records:
            if r.get("blocked") is None and (
                not r.get("privileged_ok", False) or r.get("task_truth_in_controller", 1) != 0
            ):
                raise lp.GuardError(f"G-privileged: {arm} seed {r['seed']} read task truth")
    log(f"{what}: {sum(bool(r['success']) for r in records)}/{len(records)}")
    return records


def successes(records) -> list[bool]:
    return [bool(r["success"]) for r in records]


def lean(records, *, keep_path: bool = False) -> list[dict]:
    out = []
    for item in hz.strip(records):
        item = dict(item)
        for key in ("frame405", "corpus", "captured"):
            item.pop(key, None)
        if not keep_path and "kpred" in item:
            item["kpred"] = {k: v for k, v in item["kpred"].items() if k != "plate_path"}
        out.append(item)
    return out


def arm_summary(records) -> dict:
    misses = [
        r["commit"]["landing_miss_cm"]
        for r in records
        if r.get("commit", {}).get("landing_miss_cm") is not None
    ]
    fixed = [
        r["commit"]["fixed_point_error_cm"]
        for r in records
        if r.get("commit", {}).get("fixed_point_error_cm") is not None
    ]

    def stats(v):
        if not v:
            return {"median": None, "p87_5": None, "max": None}
        return {
            "median": float(np.median(v)),
            "p87_5": float(np.percentile(v, 87.5)),
            "max": float(np.max(v)),
        }

    commits = [r for r in records if r.get("commit")]
    return {
        "count": sum(successes(records)),
        "n": len(records),
        "per_reset": successes(records),
        "refused": sum(r["blocked"] is not None for r in records),
        "fallbacks": sum(bool(r["commit"]["fallback"]) for r in commits),
        "clip_binding_fraction": (
            sum(bool(r["commit"].get("clipped")) for r in commits) / len(commits)
            if commits
            else None
        ),
        "landing_miss_cm": stats(misses),
        "fixed_point_error_cm": stats(fixed),
        "seconds": {
            "median": float(np.median([r["seconds"] for r in records])),
            "max": float(np.max([r["seconds"] for r in records])),
        },
    }


def open_pool(report, args, config: dict):
    default = lm.WM_WORKERS if args.stage in ("gates", "closed") else lm.SIM_WORKERS
    workers = int(args.workers or default)
    report["workers"] = workers
    return Pool(workers, {"torch_threads": lm.WORKER_TORCH_THREADS} | config)


def sim_setup(report, args, manifest, config: dict):
    """The pool, P-3's refitted readout (G-repro) and the cohorts."""
    sim_preflight(report, args, manifest)
    base = {"p3_checkpoint": hz.p3_checkpoint(Path(args.evidence))}
    pool = open_pool(report, args, base | config)
    p_readout, encoder = hz.refit_p_readout(report, pool, Path(args.evidence), report["_run1"])
    report["first_render_utc"] = hz.utc()
    return pool, Cohorts(args, pool, p_readout, encoder, report)


# ----- K0 (CPU, before the freeze) ---------------------------------------------------------------
def scene_blind_at_bar(ceiling, proxy, bar: int) -> list[bool]:
    """The ceiling's outcomes reduced to ``bar`` successes, removing the successes where the proxy
    failed first (lowest index first): the least favourable reference for the twin test."""
    ref = np.asarray(ceiling, bool).copy()
    proxy = np.asarray(proxy, bool)
    excess = int(ref.sum()) - int(bar)
    for i in np.flatnonzero(ref & ~proxy):
        if excess <= 0:
            break
        ref[i], excess = False, excess - 1
    for i in np.flatnonzero(ref):
        if excess <= 0:
            break
        ref[i], excess = False, excess - 1
    return ref.tolist()


def stage_k0(report, args, fields: Fields, manifest) -> str:
    pool, co = sim_setup(report, args, manifest, {})
    try:
        seeds = co.seeds("K")
        counts, levels = {}, {}
        for level in lm.TAU_LEVELS_CM:
            tasks = attempt_tasks(
                "H-final-planted",
                seeds,
                co,
                lambda s, level=level: {"planted_m": lm.planted_error_m(s, level)},
            )
            records = run_arm(pool, tasks, f"K0 planted {level} cm")
            counts[level] = sum(successes(records))
            levels[level] = records
        tau = lm.decide_tau(counts)
        fields.set("tau", tau | {"arms": {str(k): arm_summary(r) for k, r in levels.items()}})
        ceiling_records = levels[0.0]
        ceiling = successes(ceiling_records)
        fields.set("ceiling", {"N_K0": int(sum(ceiling)), "per_reset": ceiling})
        s1 = lm.RULE["s1"]
        distances, speeds, m2, remaining = [], [], [], []
        for r in ceiling_records:
            k = r["kpred"]
            path = {int(t): np.asarray(v) for t, v in k["plate_path"].items()}
            if r["blocked"] is not None or s1 not in path:
                distances.append(None)
            else:
                distances.append(
                    {t: 100.0 * float(np.linalg.norm(v - path[s1])) for t, v in path.items()}
                )
                remaining.append(k.get("remaining_405_cm"))
            if k.get("palm_speed_405_cm") is not None:
                speeds.append(k["palm_speed_405_cm"])
            if k.get("m2_405_cm") is not None:
                m2.append(k["m2_405_cm"])
        r_k = lm.read_step_k(distances)
        fields.set("r_k", {"r_K": r_k if r_k is not None else "undefined"})
        history = {
            "palm_speed_405_cm": {
                "median": float(np.median(speeds)) if speeds else None,
                "max": float(np.max(speeds)) if speeds else None,
            },
            "m2_405_cm": {
                "median": float(np.median(m2)) if m2 else None,
                "max": float(np.max(m2)) if m2 else None,
            },
            "remaining_405_cm_median": float(np.median([v for v in remaining if v is not None]))
            if remaining
            else None,
        }
        fields.set("history", history)
        n = len(seeds)
        moved = {s: lm.moved_plate(s).tolist() for s in seeds}
        proxies = {}
        for arm in lm.K0_PROXIES:
            per = None
            if arm == "shuf-proxy":

                def per(s):
                    return {"foreign_plate": moved[seeds[lm.foreign_index(seeds.index(s), n)]]}

            extra = {"a_lo": lm.A_LO} | (
                {"plate_mean": list(lm.PLATE_MEAN)} if arm == "mean-proxy" else {}
            )
            records = run_arm(pool, attempt_tasks(arm, seeds, co, per, **extra), f"K0 {arm}")
            outcome = successes(records)
            item = {
                "arm": arm_summary(records),
                "headroom": lm.paired_interval(ceiling, outcome),
            }
            if arm in lm.K0_SCENE_BLIND:
                item["feasibility_at_ceiling"] = lm.mcnemar_feasibility(ceiling, outcome)
                at_bar = scene_blind_at_bar(ceiling, outcome, lm.TAU_BAR)
                item["feasibility_at_w_bar"] = lm.mcnemar_feasibility(at_bar, outcome)
                item["known_risk"] = bool(
                    item["feasibility_at_w_bar"]["predicted_pass_probability"] < lm.FEASIBILITY_BAR
                )
            proxies[arm] = item
        ni = [
            lm.simulate_non_inferiority(
                p_c=min(30, sum(ceiling) * 32 // max(n, 1)) / 32,
                gap=gap,
                dependence="independent",
                share=0.5,
                config_index=900 + i,
            )
            for i, gap in enumerate((0, 2, 4, 8))
        ]
        fields.set(
            "proxies",
            {
                "proxies": proxies,
                "ni_simulation_at_k0_counts": ni,
                "note": "the comparator's true rate is taken as min(30/32, N_K(0)): K0 runs no "
                "comparator arm (H-rule and H-sysid scored 30/32 each in C1-M's M-F6)",
            },
        )
        palm = history["palm_speed_405_cm"]["median"]
        decision = lm.decide_k0(
            tau=tau,
            ceiling=int(sum(ceiling)),
            r_k=r_k,
            palm_speed_median_cm=palm if palm is not None else float("inf"),
        )
        fields.set("decision", decision)
        return decision["row"]
    finally:
        report["pool_close"] = pool.close()


# ----- Stage C ------------------------------------------------------------------------------------
def stage_corpus(report, args, fields: Fields, manifest) -> str:
    pool, co = sim_setup(report, args, manifest, {})
    folder = Path(args.output) / "corpus"
    folder.mkdir()
    try:
        seeds = co.seeds("corpus")
        split = lm.corpus_split(seeds, debug=args.debug)
        fields.set("split", split)
        entries, excluded = {}, {}
        for lo in range(0, len(seeds), CORPUS_CHUNK):
            part = seeds[lo : lo + CORPUS_CHUNK]
            tasks = attempt_tasks(
                "collect",
                part,
                co,
                lambda s: dict(zip(("a", "b_m"), lm.corpus_aim(s), strict=True)),
            )
            for r in run_arm(pool, tasks, f"corpus {part[0]}-{part[-1]}"):
                arrays = r.get("corpus") or {"complete": False, "why": "blocked"}
                if r["blocked"] is not None or not arrays["complete"]:
                    excluded[int(r["seed"])] = (
                        r.get("termination_reason") if r["blocked"] else (arrays["why"])
                    )
                    continue
                entries[int(r["seed"])] = off.write_root(folder, r["seed"], arrays)
            check_disk(report, folder, lm.DISK_MIN_GIB)
        fields.set("roots", len(entries))
        fields.set("excluded", {str(k): v for k, v in sorted(excluded.items())})
        sealed = off.seal_corpus(
            folder,
            entries,
            split,
            {
                "revision": report["revision"],
                "privileged_scripted_collector": True,
                "learned_control": False,
                "debug": bool(args.debug),
                "resets_digest": co.digests.get("corpus"),
            },
        )
        fields.set("sealed", sealed)
        decision = lm.decide_corpus(len(seeds), len(excluded))
        fields.set("decision", decision)
        return decision["row"]
    finally:
        report["pool_close"] = pool.close()


# ----- Stage O ------------------------------------------------------------------------------------
def stage_featurise(report, args, fields: Fields, manifest) -> str:
    import torch

    rt.gpu_guard(report, min_free_gib=lm.GPU["min_free_gib"], require_lock=True)
    devices.configure_determinism("cuda", strict=True)
    torch.set_num_threads(6)
    corpus = open_corpus_checked(args)
    out = Path(args.output) / "features"
    result = off.featurise_corpus(Path(args.corpus), corpus, out, device="cuda")
    fields.set("featurisation", result)
    fields.set("anchor", result["anchor"])
    return "FEATURISED"


def open_corpus_checked(args) -> dict:
    expected = None if args.debug else args.corpus_sha256
    if not args.debug and not expected:
        raise lp.GuardError("G-split: the sealed corpus manifest's sha256 is required")
    return off.open_corpus(Path(args.corpus), expected)


def tau_commit(args):
    if args.tau_commit_cm is not None:
        if not args.debug and lm.K0_MEASURED is None:
            raise lp.GuardError("G-frozen: tau_commit comes from K0's frozen values")
        return float(args.tau_commit_cm)
    if lm.K0_MEASURED is not None:
        return float(lm.K0_MEASURED["tau_commit_cm"])
    if args.debug:
        return DEBUG["tau_commit_cm"]
    raise lp.GuardError("G-frozen: tau_commit is not measured yet (K0)")


def stage_readouts(report, args, fields: Fields, manifest) -> str:
    out = Path(args.output) / "fits"
    report["first_outcome_utc"] = hz.utc()
    result = off.readouts_core(Path(args.features), out, tau_commit_cm=tau_commit(args))
    fields.set("admission", result["admission"])
    fields.set("fits", result["fits"])
    fields.set("decision", result["decision"])
    report["readouts_detail"] = {k: v for k, v in result.items() if k not in ("admission", "fits")}
    return result["decision"]["row"]


# ----- Stage T ------------------------------------------------------------------------------------
def parse_job(job: str, debug: bool) -> dict:
    if job in ("cal-W", "cal-N"):
        seed = lm.DEBUG_CALIBRATION_SEED if debug else lm.CALIBRATION_SEED
        return {"arm": job[-1], "seed": seed, "calibration": True}
    arm, _, seed = job.partition("-")
    if arm not in lm.ARMS_TRAINED or not seed.isdigit():
        raise lp.GuardError(f"unknown Stage T job {job!r}")
    return {"arm": arm, "seed": lm.check_model_seed(int(seed), debug=debug), "calibration": False}


def job_budget(args, spec: dict) -> dict:
    if spec["calibration"]:
        if args.debug:
            return {
                "updates": DEBUG["calibration_updates"],
                "select_every": DEBUG["calibration_select_every"],
            }
        return {
            "updates": lm.BUDGET["calibration_updates"],
            "select_every": lm.BUDGET["calibration_select_every"],
        }
    if args.debug and not args.plan:
        return {"updates": DEBUG["model_updates"], "select_every": DEBUG["model_select_every"]}
    plan = completed(args.plan, ("T-PLANNED",), args.debug)
    budget = plan["fields"]["budget"]
    return {"updates": int(budget["updates"]), "select_every": int(budget["select_every"])}


def train_context(args, *, train=None, val=None, mean=None, std=None, moments_sha=None) -> dict:
    """The stores and the normalisation a Stage T job reads (Stage O's train-only moments)."""
    fits = Path(args.fits) if args.fits else None
    if mean is None:
        with np.load(fits / "moments.npz") as data:
            mean, std = data["mean"], data["std"]
        moments_sha = off.moments_sha256(mean, std)
    if train is None:
        train = tr.RootStore.open(Path(args.features), "train")  # in RAM: about 9.6 GB
        val = tr.RootStore.open(Path(args.features), "val", mmap=True)
    return {"train": train, "val": val, "mean": mean, "std": std, "moments_sha256": moments_sha}


def train_job_core(
    ctx: dict,
    spec: dict,
    budget: dict,
    *,
    output: Path,
    corpus_sha: str,
    cap: float,
    probe: bool = False,
) -> dict:
    """One Stage T job: train, select, save. The scale probe calls this same function."""
    metadata = tr.model_metadata(
        arm=spec["arm"],
        seed=spec["seed"],
        corpus_sha256=corpus_sha,
        moments_sha256=ctx["moments_sha256"],
    )
    model, record = tr.train_model(
        arm=spec["arm"],
        seed=spec["seed"],
        train=ctx["train"],
        val=ctx["val"],
        mean=ctx["mean"],
        std=ctx["std"],
        updates=budget["updates"],
        select_every=budget["select_every"],
        device="cuda",
        metadata=metadata,
        cap_seconds=cap,
        log=log,
        probe=probe,
    )
    path = output / f"{spec['arm']}-{spec['seed']}.pt"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    model.save(path)
    record |= {
        "checkpoint": str(path),
        "checkpoint_sha256": tr.sha256_file(path),
        "state_sha256": tr.state_sha256(model),
        "metadata": metadata,
    }
    del model
    return record


def stage_train(report, args, fields: Fields, manifest) -> str:
    import torch

    rt.gpu_guard(report, min_free_gib=lm.GPU["min_free_gib"], require_lock=True)
    report["determinism"] = devices.configure_determinism("cuda", strict=True)
    torch.set_num_threads(6)
    spec = parse_job(args.job, args.debug)
    budget = job_budget(args, spec)
    fields.set("job", spec | budget)
    corpus = open_corpus_checked(args)
    ctx = train_context(args)
    record = train_job_core(
        ctx,
        spec,
        budget,
        output=Path(args.output),
        corpus_sha=corpus["_sha256"],
        cap=lm.CAPS_SECONDS["T_job"],
    )
    fields.set("record", record)
    fields.set("checkpoint", {"path": record["checkpoint"], "sha256": record["checkpoint_sha256"]})
    return "T-JOB-DONE"


def completed(path, rows, debug: bool) -> dict:
    """An earlier stage's report whose outcome is one of ``rows`` (``-DEBUG`` in a debug run;
    a debug report never feeds a real run and the reverse)."""
    report = json.loads(Path(path).read_text())
    allowed = {f"{r}-DEBUG" for r in rows} if debug else set(rows)
    if report.get("outcome") not in allowed or bool(report.get("debug")) != bool(debug):
        raise lp.GuardError(f"G-plan: {path} ended {report.get('outcome')}, not one of {rows}")
    return report


def load_job(path, debug: bool) -> dict:
    return completed(path, ("T-JOB-DONE",), debug)["fields"]["record"]


def stage_plan(report, args, fields: Fields, manifest) -> str:
    """The budget rule (u_sat = select_checkpoint(curve, 0.01) of each calibration run), G1's
    bars from the calibration W on val at h = 60 (CPU) and the truncation controls."""
    import torch

    torch.set_num_threads(1)
    cal_w, cal_n = load_job(args.cal_w, args.debug), load_job(args.cal_n, args.debug)
    u_w = rt.select_checkpoint(cal_w["val_curve"], lm.SELECTION_TOLERANCE)
    u_n = rt.select_checkpoint(cal_n["val_curve"], lm.SELECTION_TOLERANCE)
    fields.set(
        "calibration",
        {
            "u_sat_W": u_w,
            "u_sat_N": u_n,
            "last_two_W": rt.last_two_triggered(cal_w["val_curve"], lm.SELECTION_TOLERANCE),
            "last_two_N": rt.last_two_triggered(cal_n["val_curve"], lm.SELECTION_TOLERANCE),
        },
    )
    if args.debug:
        budget = {"updates": DEBUG["model_updates"], "select_every": DEBUG["model_select_every"]}
        budget |= {"debug_standin": True, "rule_reading": lm.budget_rule(max(u_w, 1000), 1000)}
    else:
        budget = lm.budget_rule(u_w, u_n)
    fields.set("budget", budget)
    val = tr.RootStore.open(Path(args.features), "val", mmap=True)
    with np.load(Path(args.fits) / "moments.npz") as data:
        scale = np.maximum(data["std"], lm.METRIC_FLOOR_STD)
    model = tr.load_model(
        cal_w["checkpoint"],
        seed=cal_w["seed"],
        metadata=cal_w["metadata"],
        expected_sha256=cal_w["checkpoint_sha256"],
    )
    start, commands, targets = val.from_405()
    pred = off.predict_at(model, np.asarray(start), np.asarray(commands), (lm.HORIZON,))
    encoded = np.asarray(targets[:, lm.HORIZON - 1], np.float64) / scale
    g1 = off.calibrate_g1(np.asarray(pred[lm.HORIZON], np.float64) / scale, encoded)
    fields.set("g1", g1)
    reference = g1["stats"]["effective_rank_ratio"]
    row = "CAL-T-ESCALATE" if reference < lm.CAL_T_ESCALATE_BELOW else "T-PLANNED"
    fields.set("decision", {"row": row, "rank_reference": reference, "clause_fires": False})
    return row


# ----- Stage G ------------------------------------------------------------------------------------
def stage_gates(report, args, fields: Fields, manifest) -> str:
    import torch

    torch.set_num_threads(1)
    report["first_outcome_utc"] = hz.utc()
    plan = completed(args.plan, ("T-PLANNED",), args.debug)["fields"]
    bars = plan["g1"]["bars"]
    jobs = [load_job(p, args.debug) for p in args.models]
    by = {(j["arm"], j["seed"]): j for j in jobs}
    seeds = sorted({j["seed"] for j in jobs})
    expected = (lm.DEBUG_MODEL_SEEDS[0],) if args.debug else lm.MODEL_SEEDS
    if tuple(seeds) != tuple(expected) or any(
        ("N", s) not in by or ("W", s) not in by for s in seeds
    ):
        raise lp.GuardError(f"G-plan: Stage G needs W and N of every seed {expected}")
    primary = min(seeds, key=lambda s: (by[("W", s)]["kept_val_criterion"], s))
    fields.set(
        "primary_seed",
        {
            "seed": primary,
            "rule": "the W seed with the lowest val "
            "criterion at its kept checkpoint (ties to the lower seed)",
        },
    )
    gate = tr.RootStore.open(Path(args.features), "gate", mmap=True)
    table = off.load_table(Path(args.features), "gate")
    with np.load(Path(args.fits) / "moments.npz") as data:
        scale = np.maximum(data["std"], lm.METRIC_FLOOR_STD)
    from embodied_jepa.models.latent_critic import RidgeReadout

    with np.load(Path(args.fits) / "r8.npz") as data:
        r8 = RidgeReadout.from_state({k: data[k] for k in data.files})
    tau = tau_commit(args)
    found = hz.check_evidence(Path(args.evidence))
    report["evidence"] = {"root": str(args.evidence), "sha256": found["sha256"]}
    config = model_config(args, primary)
    config.pop("_sysid_coef")
    config |= {"torch_threads": 1, "p3_checkpoint": hz.p3_checkpoint(Path(args.evidence))}
    pool = Pool(int(args.workers or lm.WM_WORKERS), config)
    report["workers"] = int(args.workers or lm.WM_WORKERS)
    try:
        targets = [np.asarray(t) for t in table["target"]]
        tasks = [
            {
                "kind": "chunks",
                "key": i,
                "state": table["state405"][i].tolist(),
                "apple": table["apple"][i].tolist(),
                "last_grasp": table["last_grasp"][i].tolist(),
                "step": lm.COMMIT_STEP,
                "targets": [targets[i].tolist()],
            }
            for i in range(len(gate))
        ]
        chunks = pool.map(tasks, MAP_CAP_SECONDS, "stand-in chunks of the gate roots' aims")
        fields.set("offline_aims", offline_aims(args, pool, gate, table, tau))
    finally:
        report["pool_close"] = pool.close()
    standin = np.stack([c["chunks"][0] for c in sorted(chunks, key=lambda c: c["key"])])
    per_seed, gates = {}, {}
    for s in seeds:
        w = tr.load_model(
            by[("W", s)]["checkpoint"],
            seed=s,
            metadata=by[("W", s)]["metadata"],
            expected_sha256=by[("W", s)]["checkpoint_sha256"],
        )
        n = tr.load_model(
            by[("N", s)]["checkpoint"],
            seed=s,
            metadata=by[("N", s)]["metadata"],
            expected_sha256=by[("N", s)]["checkpoint_sha256"],
        )
        result = off.seed_gates(
            w_model=w,
            n_model=n,
            gate=gate,
            table=table,
            scale=scale,
            r8=r8,
            bars=bars,
            tau_commit_cm=tau,
            standin_commands=standin,
        )
        result["n_unsaturated"] = by[("N", s)]["last_two_triggered"]
        per_seed[str(s)] = result
        gates[s] = result["gates"]
        del w, n
    fields.set("per_seed", per_seed)
    if args.debug:
        decision = lm.decide_g({lm.DEBUG_MODEL_SEEDS[0]: gates[seeds[0]]})
    else:
        decision = lm.decide_g(gates)
    fields.set("decision", decision)
    return decision["row"]


# ----- Stages D and S -----------------------------------------------------------------------------
def model_config(args, primary: int) -> dict:
    """The closed-loop workers' configuration: the train-only fits and the primary seed's W, N."""
    fits = Path(args.fits)
    record = json.loads((fits.parent / "report.json").read_text())["fields"]["fits"]
    models = {}
    for path in args.models:
        job = load_job(path, args.debug)
        if job["seed"] == primary:
            models[job["arm"]] = {
                "path": job["checkpoint"],
                "sha256": job["checkpoint_sha256"],
                "seed": job["seed"],
                "metadata": job["metadata"],
            }
    if sorted(models) != ["N", "W"]:
        raise lp.GuardError("G-plan: the primary seed's W and N are required")
    return {
        "r_plate": record["r_plate"]["path"],
        "r_plate_sha256": record["r_plate"]["sha256"],
        "r8": record["r8"]["path"],
        "r8_sha256": record["r8"]["sha256"],
        "mean_latent": record["mean_latent"]["path"],
        "mean_latent_sha256": record["mean_latent"]["sha256"],
        "models": models,
        "_sysid_coef": record["sysid"]["coef"],
    }


def offline_aims(args, pool, gate, table, tau: float) -> dict:
    """§7 step 7 (reported only, before any closed loop): each candidate-choosing arm's aim from
    the gate roots' logged 405 states with the same controller code, its error against the rule's
    fixed point g* = (p - kappa h) / (1 - kappa), and the tau-curve-mapped predicted count."""
    from embodied_jepa.models.latent_critic import RidgeReadout

    fits = Path(args.fits)
    with np.load(fits / "r_plate.npz") as data:
        r_plate = RidgeReadout.from_state({k: data[k] for k in data.files})
    full = np.load(Path(args.features) / "full405_gate.npy", mmap_mode="r")
    p_hat = np.stack([r_plate.predict(np.asarray(full[i], np.float64)) for i in range(len(gate))])
    latents = np.asarray(gate.features[:, tr.START_INDEX])
    n = len(gate)
    p_true = table["plate"][:, tr.START_INDEX]
    h = table["palm"][:, tr.START_INDEX]
    g_star = np.stack([c1.fixed_point(p_true[i], h[i]) for i in range(n)])
    tau_counts = DEBUG["tau_counts"] if lm.K0_MEASURED is None else lm.K0_MEASURED["tau"]["counts"]
    out = {}
    for arm in lm.CANDIDATE_ARMS:
        tasks = []
        for i in range(n):
            start = {"L-shuf": latents[lm.foreign_index(i, n)], "L-mean": "mean"}.get(
                arm, latents[i]
            )
            tasks.append(
                {
                    "kind": "offline_aim",
                    "key": i,
                    "arm": arm,
                    "seed": int(gate.roots[i]),
                    "state": table["state405"][i].tolist(),
                    "apple": table["apple"][i].tolist(),
                    "last_grasp": table["last_grasp"][i].tolist(),
                    "p_hat": p_hat[i].tolist(),
                    "h": h[i].tolist(),
                    "start": start,
                    "tau_commit_cm": tau,
                }
            )
        got = sorted(
            pool.map(tasks, MAP_CAP_SECONDS, f"offline aims {arm}"), key=lambda r: r["key"]
        )
        aims = np.asarray([r["g"] for r in got])
        errors = 100.0 * np.linalg.norm(aims - g_star, axis=1)
        prob = lm.tau_curve_probability(errors, tau_counts)
        out[arm] = {
            "aim_error_cm": lm.median_ci(errors),
            "predicted_count_of_64": float(lm.S_RESETS * prob.mean()),
            "clip_binding_fraction": float(np.mean([bool(r["log"].get("clipped")) for r in got])),
            "fallbacks": int(sum(bool(r["log"].get("fallback_all_infeasible")) for r in got)),
        }
    return {
        "arms": out,
        "tau_counts": tau_counts,
        "note": "reported only; p-hat from R-plate on Stage O's (CUDA) full tokens at 405 and the "
        "start latents from Stage O's 8 x 8 store (the closed loop encodes on the CPU)",
    }


def stage_closed(report, args, fields: Fields, manifest) -> str:
    # a debug run reads any gates row (mechanics only); a real run only after G-PASS (and S
    # only after D-PASS, which its GO checks)
    rows = ("G-PASS", "H-GATE-FAIL", "G-NO-BAR") if args.debug else ("G-PASS",)
    gates = completed(args.gates, rows, args.debug)
    primary = int(gates["fields"]["primary_seed"]["seed"])
    config = model_config(args, primary)
    coef = config.pop("_sysid_coef")
    report["primary_seed"] = primary
    role = args.cohort
    pool, co = sim_setup(report, args, manifest, config)
    try:
        seeds = co.seeds(role)
        report["first_outcome_utc"] = hz.utc()
        tau = tau_commit(args)
        arms = lm.D_ARMS if role == "D" else lm.S_ARMS
        records = {}
        common = {"tau_commit_cm": tau, "a_lo": lm.A_LO}
        records["W"] = run_arm(pool, attempt_tasks("W", seeds, co, **common), f"{role} W")
        frames = {r["seed"]: r["frame405"] for r in records["W"]}
        if any(f is None for f in frames.values()):
            raise lp.GuardError("G-frames: W's attempt did not log its 405 frame")
        n = len(seeds)
        for arm in arms:
            if arm == "W":
                continue
            per, extra = None, dict(common)
            if arm == "L-shuf":

                def per(s):
                    return {"foreign_frame": frames[seeds[lm.foreign_index(seeds.index(s), n)]]}

            if arm == "H-sysid":
                extra["sysid_coef"] = coef
            if arm == "H-read":
                extra |= {"readout": "r8", "grid": lm.TOKEN_GRID, "read_step": lm.READ_STEP}
            tasks = attempt_tasks(arm, seeds, co, per, **extra)
            records[arm] = run_arm(pool, tasks, f"{role} {arm}")
        determinism = lm.NOT_EVALUATED
        if role == "S":
            first = seeds[: lm.DETERMINISM_RESETS]
            again = run_arm(pool, attempt_tasks("W", first, co, **common), "S W determinism")
            diffs = [
                float(np.max(np.abs(np.asarray(a["commit"]["target"]) - b["commit"]["target"])))
                for a, b in zip(again, records["W"][: len(first)], strict=True)
            ]
            determinism = {"max_abs_diff_m": max(diffs), "ok": max(diffs) <= 1e-6}
            if not determinism["ok"]:
                raise lp.GuardError(f"G-determinism: W's aims differ by {max(diffs)} m")
            fields.set("determinism", determinism)
        fields.set(
            "arms",
            {a: arm_summary(r) | {"attempts": lean(r)} for a, r in records.items()},
        )
        outcomes = {a: successes(r) for a, r in records.items()}
        if role == "D":
            decision = lm.decide_d({a: sum(v) for a, v in outcomes.items()})
        else:
            decision = lm.decide_s(outcomes)
        fields.set("decision", decision)
        return decision["row"]
    finally:
        report["pool_close"] = pool.close()


# ----- Stage 0: the simulations and the scale probe ----------------------------------------------
def stage_simulate(report, args, fields: Fields, manifest) -> str:
    started = time.monotonic()
    result = lm.stage0_simulations(trials=args.trials or lm.SIM_TRIALS)
    result["seconds"] = time.monotonic() - started
    fields.set("simulations", result)
    return "SIMULATED"


def synthetic_store(folder: Path, split: str, roots: int, seed: int) -> dict:
    """Synthetic features at the real layout (no corpus, no rendered frame)."""
    from numpy.lib.format import open_memmap

    rng = np.random.default_rng(np.random.SeedSequence([lm.SALTS["bootstrap"], 77, seed]))
    folder.mkdir(parents=True, exist_ok=True)
    feats = open_memmap(
        folder / f"features8_{split}.npy", "w+", np.float32, (roots, lm.N_FRAMES, lm.LATENT_DIM)
    )
    for i in range(roots):
        feats[i] = rng.standard_normal((lm.N_FRAMES, lm.LATENT_DIM), dtype=np.float32)
    feats.flush()
    del feats
    commands = rng.uniform(-0.1, 0.1, (roots, lm.N_COMMANDS, 14)).astype(np.float32)
    np.save(folder / f"commands_{split}.npy", commands)
    (folder / f"roots_{split}.json").write_text(json.dumps(list(range(roots))))
    return {"roots": roots, "bytes": roots * lm.N_FRAMES * lm.LATENT_DIM * 4}


def stage_scale(report, args, fields: Fields, manifest) -> str:
    """Stage 0's scale probe (§7 step 1): ``train_job_core`` itself at the real sizes on
    synthetic features (1 500 train and 250 val roots), the gather alone, and a bit-identity
    re-run of a short job on a debug seed."""
    import torch

    rt.gpu_guard(report, min_free_gib=lm.GPU["min_free_gib"], require_lock=True)
    report["determinism"] = devices.configure_determinism("cuda", strict=True)
    torch.set_num_threads(6)
    scratch = Path(args.scratch)
    if scratch.exists():
        raise FileExistsError(f"refusing to overwrite {scratch}")
    sizes = {"train": args.scale_train_roots, "val": args.scale_val_roots}
    try:
        made = {s: synthetic_store(scratch, s, n, i) for i, (s, n) in enumerate(sizes.items())}
        report["synthetic"] = made
        train = tr.RootStore.open(scratch, "train")  # loaded into RAM, as a Stage T job does
        val = tr.RootStore.open(scratch, "val", mmap=True)
        fields.set("gather", tr.measure_gather(train, seed=lm.DEBUG_MODEL_SEEDS[0]))
        mean, std = np.zeros(lm.LATENT_DIM), np.ones(lm.LATENT_DIM)
        ctx = train_context(
            args,
            train=train,
            val=val,
            mean=mean,
            std=std,
            moments_sha=off.moments_sha256(mean, std),
        )
        spec = {"arm": "W", "seed": lm.DEBUG_MODEL_SEEDS[0], "calibration": False}
        budget = {"updates": args.scale_updates, "select_every": args.scale_updates // 2}
        out = Path(args.output) / "checkpoints"
        out.mkdir()
        probe = rt.scale_probe(
            train_job_core,
            ctx,
            spec,
            budget,
            output=out,
            corpus_sha="synthetic",
            cap=lm.CAPS_SECONDS["T_job"],
            watch=report["_watch"],
            ceiling_gib=lm.MEMORY["train_ceiling_gib"],
            margin_gib=2.0,
        )
        fields.set("probe", probe)
        # bit identity: the same short job twice (N, so the zeroed commands path runs too)
        short = {"updates": 200, "select_every": 100}
        runs = []
        for k in range(2):
            folder = out / f"identity-{k}"
            folder.mkdir()
            spec_n = {"arm": "N", "seed": lm.DEBUG_MODEL_SEEDS[1], "calibration": False}
            runs.append(
                train_job_core(
                    ctx,
                    spec_n,
                    short,
                    output=folder,
                    corpus_sha="synthetic",
                    cap=lm.CAPS_SECONDS["T_job"],
                )
            )
        fields.set(
            "bit_identity",
            {
                "state_sha256": [r["state_sha256"] for r in runs],
                "val_curves_equal": runs[0]["val_curve"] == runs[1]["val_curve"],
                "identical": runs[0]["state_sha256"] == runs[1]["state_sha256"],
            },
        )
        per = probe["result"]["per_update_seconds"]
        worst = per["p95"] * lm.BUDGET["cap"] + 20 * probe["result"]["seconds"] / max(
            1, args.scale_updates
        )
        fields.set(
            "caps",
            {
                "per_update_median_s": per["median"],
                "per_update_p95_s": per["p95"],
                "worst_case_job_s_at_cap": per["p95"] * lm.BUDGET["cap"],
                "cap_T_job_s": lm.CAPS_SECONDS["T_job"],
                "cap_over_worst": lm.CAPS_SECONDS["T_job"] / (per["p95"] * lm.BUDGET["cap"]),
                "note": "the worst case is 100 000 updates at the probe's 95th-percentile update "
                "time; the 20 val selections add little at 250 val roots",
                "estimate_with_selection_s": worst,
            },
        )
    finally:
        if scratch.exists() and not args.keep_scratch:
            shutil.rmtree(scratch)
            report["synthetic_removed"] = True
    return "SCALE-PROBED"


def stage_tests(report, args, fields: Fields, manifest) -> str:
    fields.set("tests", run_full_tests())
    return "TESTS-PASS"


# ----- the run ------------------------------------------------------------------------------------
RUNNERS = {
    "tests": stage_tests,
    "k0": stage_k0,
    "corpus": stage_corpus,
    "featurise": stage_featurise,
    "readouts": stage_readouts,
    "train": stage_train,
    "plan": stage_plan,
    "gates": stage_gates,
    "closed": stage_closed,
    "simulate": stage_simulate,
    "scale": stage_scale,
}
STAGE_CAPS = {
    "k0": "K0",
    "corpus": "C",
    "featurise": "O_featurisation",
    "readouts": "O_readouts",
    "train": "T_job",
    "plan": "G",
    "gates": "G",
    "closed": "S",
    "scale": "T_job",
}


def run(args) -> dict:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    report = {
        "protocol": lm.PROTOCOL,
        "task": lm.TASK,
        "stage": args.stage,
        "debug": bool(args.debug),
        "outcome": None,
        "stages": {},  # the harness records G-repro here
        "argv": sys.argv[1:],
        "paths": {"output": str(output.resolve())},
        "started_utc": hz.utc(),
        "salts": lm.SALTS,
        "seed_ranges": lm.DEBUG_RANGES if args.debug else lm.SEED_RANGES,
    }
    fields = Fields(report, args.stage)
    cap = lm.CAPS_SECONDS.get(STAGE_CAPS.get(args.stage, ""), 4 * 3600.0)
    clock = hz.Clock(cap)
    ceiling = (
        lm.MEMORY["train_ceiling_gib"]
        if args.stage in ("train", "scale")
        else (lm.MEMORY["ceiling_gib"])
    )
    watch = rt.MemoryWatch(
        ceiling * GIB,
        lm.MEMORY["sample_seconds"],
        measure=lm.MEMORY["measure"],
        disk_path=ROOT,
        min_disk_free_bytes=int(lm.DISK_MIN_GIB * GIB),
    )
    guards = rt.install_guards(watch=watch, log=log)
    report["_watch"] = watch
    try:
        manifest = preflight(report, args)
        outcome = RUNNERS[args.stage](report, args, fields, manifest)
        clock.check("the end of the stage")
        if hz.tracked_tree_dirty():
            raise lp.GuardError("G-hash: the tracked tree changed during the run")
        report["revision_at_end"] = hz.revision()
        fields.check()
        report["outcome"] = f"{outcome}-DEBUG" if args.debug else outcome
    except BaseException as error:  # noqa: BLE001 - every failure, signal included, is V
        guards.void(report, error)
    finally:
        guards.finish(report)
        report.pop("_watch", None)
        report.pop("_run1", None)
        report["ended_utc"] = hz.utc()
        report["total_seconds"] = clock.elapsed()
        hz.write_report(output / "report.json", report)
        log(f"report written: outcome {report.get('outcome')}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug", action="store_true", help="debug seeds only; nothing is read")
    parser.add_argument(
        "--debug-skip-tests",
        action="store_true",
        help="debug only: skip G-tests' own pytest run (recorded)",
    )
    parser.add_argument("--workers", type=int, default=None, help="debug only (1-6)")
    parser.add_argument("--evidence", help="TASK-072 run-1's evidence root (G-repro)")
    parser.add_argument("--tests-record", help="GPU jobs: a `tests` stage report at HEAD")
    parser.add_argument("--corpus", help="the sealed corpus folder")
    parser.add_argument("--corpus-sha256", help="the sealed corpus manifest's sha256")
    parser.add_argument("--features", help="Stage O's featurisation folder")
    parser.add_argument("--fits", help="Stage O's fits folder (inside the readouts output)")
    parser.add_argument("--tau-commit-cm", type=float, default=None, help="debug only")
    parser.add_argument("--job", help="Stage T: cal-W, cal-N, W-<seed> or N-<seed>")
    parser.add_argument("--plan", help="Stage T's plan report")
    parser.add_argument("--cal-w", help="the calibration W job's report")
    parser.add_argument("--cal-n", help="the calibration N job's report")
    parser.add_argument("--models", nargs="*", default=[], help="Stage T job reports")
    parser.add_argument("--gates", help="Stage G's report")
    parser.add_argument("--cohort", choices=("D", "S"))
    parser.add_argument("--trials", type=int, default=None, help="simulate: trials per config")
    parser.add_argument("--scratch", help="scale: the synthetic store's folder (removed after)")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--scale-train-roots", type=int, default=lm.SPLIT_SIZES["train"])
    parser.add_argument("--scale-val-roots", type=int, default=lm.SPLIT_SIZES["val"])
    parser.add_argument("--scale-updates", type=int, default=600)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers is not None and (not args.debug or not 1 <= args.workers <= 6):
        parser.error("--workers is for debug runs only, from 1 to 6")
    if args.debug_skip_tests and not args.debug:
        parser.error("--debug-skip-tests is for debug runs only")
    if args.tau_commit_cm is not None and not args.debug:
        parser.error("--tau-commit-cm is for debug runs only (K0's value is frozen)")
    if args.stage in SIM_STAGES and not args.evidence:
        parser.error(f"{args.stage} needs --evidence (G-repro)")
    if args.stage in GPU_STAGES and not args.tests_record:
        parser.error(f"{args.stage} needs --tests-record (G-tests for GPU jobs)")
    if args.stage == "scale" and not args.scratch:
        parser.error("scale needs --scratch")
    if args.stage == "closed" and not args.cohort:
        parser.error("closed needs --cohort")
    if args.stage == "train" and not args.job:
        parser.error("train needs --job")
    report = run(args)
    return 0 if report.get("outcome") not in (None, "V") else 1


if __name__ == "__main__":
    sys.exit(main())
