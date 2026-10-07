"""TASK-080's runner: every stage of ``docs/experiments/apple_lewm_c1m_v2_pred_readout.md``.

Frozen block ``src/embodied_jepa/lewm_pr_v2.py`` (STATUS DRAFT until the freeze); workers
``lewm_pr_v2_runtime.py``; offline parts ``lewm_pr_v2_offline.py``. TASK-077's modules are
imported, never edited; guards from ``embodied_jepa.run_tools`` and TASK-076's harness module.
This runner loads no other script.

One invocation runs one stage and writes ``<output>/report.json`` (it refuses an existing output):

* ``tests``     -- the full pytest suite at HEAD, recorded (G-tests for the GPU stage);
* ``k0``        -- K0′ on K′ (65000-65031), CPU, before the freeze (on a reported GO only);
* ``corpus``    -- Stage C′, the fresh 500-root corpus ``apple-c1m-v2-f`` (CPU);
* ``featurise`` -- Stage R's featurisation of the fresh corpus (CUDA, through
  ``scripts/gpu_run.sh --wait --min-free-gib 8 --board --who oej:task080-featurise``);
* ``rgate``     -- Stage R on the CPU: the roll-outs and fits R-S, R-N, R-L on the 1 995 old roots,
  then (after ``first_outcome_utc``) gate-P's R0-R3, A1-A2 and every reported reading;
* ``closed``    -- Stage D (``--cohort D``) or Stage S (``--cohort S``) (CPU);
* ``simulate``  -- §10's power: G-bar, the twin table, and the power at a dry run's predicted
  rates (re-mapped through K0′'s pooled curve when its report is given);
* ``dryrun``    -- R18.13's development dry run (CPU): Stage R's own code with the readouts
  fitted on the 1 745 train and old-gate roots and every arm's offline aims on the 250 val roots,
  mapped through K0's curve; reported only, sets no bar.

Every stage checks the clean tree, TASK-076's pins, TASK-077's 13 pins and frozen block (G-frozen),
and the reused artifacts by sha256 (G-hash). While the protocol is DRAFT only ``tests``,
``simulate``, ``k0``, ``dryrun`` and ``--debug`` runs are allowed. ``--debug`` simulates debug
seeds 65900-65999 only, at small sizes; nothing in it is read.
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

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import devices  # noqa: E402
from embodied_jepa import first_policy_v2_linux as fpl  # noqa: E402
from embodied_jepa import lewm_c1m_v2 as lm  # noqa: E402
from embodied_jepa import lewm_c1m_v2_offline as off  # noqa: E402
from embodied_jepa import lewm_c1m_v2_train as tr  # noqa: E402
from embodied_jepa import lewm_planner_v2 as lp  # noqa: E402
from embodied_jepa import lewm_pr_v2 as pr  # noqa: E402
from embodied_jepa import lewm_pr_v2_offline as pof  # noqa: E402
from embodied_jepa import plate_twin_v2_harness as hz  # noqa: E402
from embodied_jepa import run_guards as rg  # noqa: E402
from embodied_jepa import run_tools as rt  # noqa: E402

fpl.configure_headless()
GIB = 2**30
log = hz.log
PROTOCOL_DOCUMENT = ROOT / pr.DOCUMENT
MAP_CAP_SECONDS = 3600.0
QUIET_POLL_SECONDS = 30.0
QUIET_WAIT_CAP_SECONDS = 4 * 3600.0
CORPUS_CHUNK = 60
ESTIMATE_CHUNK = 128
CROSS_GRAM_BLOCK = 32
STAGES = ("tests", "k0", "corpus", "featurise", "rgate", "closed", "simulate", "dryrun")
CPU_STAGES = ("k0", "corpus", "rgate", "closed", "simulate", "dryrun")
GPU_STAGES = ("featurise",)
SIM_STAGES = ("k0", "corpus", "closed")
DRAFT_ALLOWED = ("tests", "simulate", "k0", "dryrun")
OLD_STAGES = ("rgate", "closed", "dryrun")  # stages that read TASK-077's artifacts
DEBUG = {"tau_commit_cm": 1.0, "fit_roots_per_split": 12, "eval_roots": 6}
STAGE_FIELDS = {
    "k0": ("tau", "ceiling", "r_k", "history", "decision"),
    "corpus": ("split", "roots", "excluded", "sealed", "decision"),
    "featurise": ("featurisation", "anchor"),
    "rgate": ("fits", "readings", "offline_aims", "reported", "gates", "decision"),
    "closed": ("arms", "determinism", "decision"),
    "simulate": ("power",),
    "dryrun": ("fits", "readings", "offline_aims", "reported", "summary"),
    "tests": ("tests",),
}
STAGE_CAPS = {
    "k0": "K0",
    "corpus": "C",
    "featurise": "R_featurisation",
    "rgate": "R",
    "dryrun": "dryrun",
}


class Pool(rg.BoundedPool):
    """``run_guards.BoundedPool`` on TASK-080's worker; a dead worker or a cap is a GuardError;
    every map of attempts runs TASK-071's G-look check (as TASK-076's harness pool)."""

    def __init__(self, workers: int, config: dict):
        from embodied_jepa import lewm_pr_v2_runtime as prt

        super().__init__(workers, prt.run_task, initializer=prt.worker_init, initargs=(config,))
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
        report["fields"] = pr.sentinel_fields(self.names)

    def set(self, name: str, value) -> None:
        if name not in self.names:
            raise KeyError(name)
        if pr.is_missing(value):
            raise pr.GuardError(f"G-sentinel: {name} set to a missing value")
        self.report["fields"][name] = value
        self.evaluated.add(name)

    def get(self, name: str):
        return self.report["fields"][name]

    def check(self) -> None:
        pr.check_sentinel(self.report["fields"], self.evaluated, self.names)


# ----- preflight ----------------------------------------------------------------------------------
def git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def check_task077(report: dict) -> None:
    """G-frozen (TASK-077 carried): its frozen block's sha256 is its pin, its 13 pinned files
    (code, tests, TASK-076's manifest) match their sha256, and TASK-080's own files are
    committed."""
    manifest = json.loads((ROOT / pr.TASK077_MANIFEST).read_text())
    if lm.frozen_sha256() != pr.TASK077_FROZEN_SHA256 or (
        manifest.get("frozen_sha256_pin") != pr.TASK077_FROZEN_SHA256
    ):
        raise lp.GuardError("G-frozen: TASK-077's frozen block is not its pin")
    report["task077_pins_at_preflight"] = len(hz.check_pins(manifest["hashes"]))
    for path in pr.OWN_FILES:
        try:
            git("ls-files", "--error-unmatch", path)
        except subprocess.CalledProcessError as error:
            raise lp.GuardError(f"G-hash: {path} is not committed") from error
    report["own_code_git_blobs"] = {p: git("hash-object", p) for p in pr.OWN_FILES}


def check_frozen_pin(manifest: dict) -> dict:
    sha = pr.frozen_sha256()
    pin = manifest.get("frozen_sha256_pin")
    if pr.STATUS == "FROZEN":
        if pin != sha:
            raise lp.GuardError(f"G-frozen: the frozen block {sha[:12]} is not the pin {pin}")
    elif pin is not None:
        raise lp.GuardError("G-frozen: a DRAFT protocol carries no pin")
    return {"status": pr.STATUS, "frozen_sha256": sha, "pin": pin}


def check_protocol_document(manifest: dict) -> dict:
    want = manifest.get("protocol_document_sha256")
    got = hz.sha256_file(PROTOCOL_DOCUMENT) if PROTOCOL_DOCUMENT.exists() else None
    if not want or got != want:
        raise lp.GuardError(f"G-hash: the protocol document {str(got)[:12]} is not the pin")
    return {"path": pr.DOCUMENT, "sha256": got}


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
    """G-tests (GPU stage): a recorded run at HEAD, exit 0, no failure, after HEAD's commit."""
    from datetime import datetime

    record = json.loads(Path(path).read_text())
    tests = record.get("fields", {}).get("tests", record)
    head = hz.revision()
    committed = git("log", "-1", "--format=%cI", "HEAD")
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
    rule = pr.QUIET_MACHINE
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
    check_task077(report)
    dirty = hz.tracked_tree_dirty()
    report["revision"], report["tracked_tree_dirty"] = hz.revision(), bool(dirty)
    hz.check_clean(dirty)
    report["protocol_status"] = pr.STATUS
    report["frozen_sha256_at_run"] = pr.frozen_sha256()
    own = json.loads((ROOT / pr.MANIFEST).read_text())
    report["frozen_pin_check"] = check_frozen_pin(own)
    if pr.STATUS == "FROZEN":
        if not own.get("hashes"):
            raise lp.GuardError("G-hash: a FROZEN manifest pins its files")
        report["own_pins_at_preflight"] = len(hz.check_pins(own["hashes"]))
        report["protocol_document_check"] = check_protocol_document(own)
    if pr.STATUS != "FROZEN" and args.stage not in DRAFT_ALLOWED and not args.debug:
        raise lp.GuardError(f"G-frozen: {args.stage} runs only after the freeze (STATUS FROZEN)")
    if pr.STATUS == "FROZEN" and args.stage in ("k0", "dryrun") and not args.debug:
        raise lp.GuardError(f"G-frozen: {args.stage} ran once before the freeze; not repeated")
    report["thread_env"] = {k: os.environ.get(k) for k in pr.THREAD_ENV}
    if report["thread_env"] != pr.THREAD_ENV:
        raise lp.GuardError(f"G-threads: {report['thread_env']} is not {pr.THREAD_ENV}")
    pr.check_seed_ranges()
    start_min = pr.DISK_MIN_START_GIB.get(
        {"corpus": "C", "featurise": "R_featurisation"}.get(args.stage, "")
    )
    check_disk(report, ROOT, start_min or pr.DISK_MIN_GIB)
    if args.stage in CPU_STAGES:
        wait_quiet(report, args)
    if args.stage in CPU_STAGES:
        if args.debug and args.debug_skip_tests:
            report["g_tests"] = {"skipped": "debug run with --debug-skip-tests (nothing is read)"}
        else:
            report["g_tests"] = run_full_tests()
    report["load_average_after_preflight"] = list(os.getloadavg())
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
    if available < (pr.MEMORY["ceiling_gib"] + pr.MEMORY["headroom_gib"]) * GIB:
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


# ----- TASK-077's artifacts, checked by sha256 (G-hash, §2.2) -----------------------------------
def completed(path, rows, debug: bool) -> dict:
    """An earlier TASK-080 stage's report whose outcome is one of ``rows`` (``-DEBUG`` in a debug
    run; a debug report never feeds a real run and the reverse)."""
    report = json.loads(Path(path).read_text())
    allowed = {f"{r}-DEBUG" for r in rows} if debug else set(rows)
    if report.get("outcome") not in allowed or bool(report.get("debug")) != bool(debug):
        raise lp.GuardError(f"G-plan: {path} ended {report.get('outcome')}, not one of {rows}")
    return report


def old_chain(args, report: dict, splits) -> dict:
    """TASK-077's featurisation (the splits read, every file against the featurise report), its
    Stage O fits (moments, R8, R-plate, the mean latent by content sha256, sysid.json by file
    sha256) and the six Stage T checkpoints (each report T-JOB-DONE, each checkpoint at its
    recorded sha256), each against TASK-080's frozen ``REUSED`` record."""
    from embodied_jepa.models.latent_critic import RidgeReadout

    want = pr.REUSED
    features = Path(args.old_features)
    feat_path = features.parent / "report.json"
    if hz.sha256_file(feat_path) != want["featurise_report_sha256"]:
        raise lp.GuardError("G-hash: TASK-077's featurise report differs from its sha256")
    feat = json.loads(feat_path.read_text())
    if feat.get("outcome") != "FEATURISED" or feat.get("corpus_sha256") != pr.OLD_CORPUS_SHA256:
        raise lp.GuardError("G-hash: not TASK-077's FEATURISED report of apple-c1m-v2")
    verified = off.verify_feature_files(
        features, feat["fields"]["featurisation"]["files_sha256"], splits
    )
    fits = Path(args.old_fits)
    ro_path = fits.parent / "report.json"
    if hz.sha256_file(ro_path) != want["readouts_report_sha256"]:
        raise lp.GuardError("G-hash: TASK-077's readouts report differs from its sha256")
    ro = json.loads(ro_path.read_text())
    if ro.get("outcome") != "O-PASS" or ro.get("corpus_sha256") != pr.OLD_CORPUS_SHA256:
        raise lp.GuardError("G-hash: not TASK-077's O-PASS readouts report")
    rec = ro["fields"]["fits"]
    with np.load(fits / "moments.npz") as d:
        mean, std = d["mean"], d["std"]
    if (
        off.moments_sha256(mean, std) != want["moments"]
        or rec["moments"]["sha256"] != want["moments"]
    ):
        raise lp.GuardError("G-hash: the moments differ from TASK-077's sha256")
    readouts = {}
    for name in ("r8", "r_plate"):
        with np.load(fits / f"{name}.npz") as d:
            readouts[name] = RidgeReadout.from_state({k: d[k] for k in d.files})
        if readouts[name].sha256() != want[name] or rec[name]["sha256"] != want[name]:
            raise lp.GuardError(f"G-hash: {name} differs from TASK-077's sha256")
    mean_latent = np.ascontiguousarray(np.load(fits / "mean_latent.npy"), np.float32)
    if pof.array_sha256(mean_latent) != want["mean_latent"]:
        raise lp.GuardError("G-hash: the mean latent differs from TASK-077's sha256")
    if hz.sha256_file(fits / "sysid.json") != want["sysid_file"]:
        raise lp.GuardError("G-hash: sysid.json differs from TASK-077's sha256")
    jobs = {}
    for path in args.models:
        path = Path(path)
        job_report = json.loads(path.read_text())
        if job_report.get("outcome") != "T-JOB-DONE" or job_report.get("debug"):
            raise lp.GuardError(f"G-hash: {path} is not a real T-JOB-DONE report")
        record = dict(job_report["fields"]["record"])
        key = f"{record['arm']}-{int(record['seed'])}"
        if record["checkpoint_sha256"] != want["models"].get(key):
            raise lp.GuardError(f"G-hash: {key}'s checkpoint is not TASK-077's")
        meta = record["metadata"]
        if meta.get("corpus_manifest_sha256") != pr.OLD_CORPUS_SHA256 or (
            meta.get("normalisation_sha256") != want["moments"]
        ):
            raise lp.GuardError(f"G-hash: {key} was trained on another corpus or moments")
        record["checkpoint"] = str(path.parent / Path(record["checkpoint"]).name)
        record["report"] = str(path.resolve())
        if hz.sha256_file(record["checkpoint"]) != record["checkpoint_sha256"]:
            raise lp.GuardError(f"G-hash: {key}'s checkpoint file differs from its sha256")
        jobs[key] = record
    if sorted(jobs) != sorted(pr.REUSED["models"]):
        raise lp.GuardError(f"G-hash: the six TASK-077 models are required, got {sorted(jobs)}")
    report["old_chain"] = {
        "features": str(features.resolve()),
        "feature_files_verified": verified,
        "fits": str(fits.resolve()),
        "models": {
            k: {"report": j["report"], "checkpoint_sha256": j["checkpoint_sha256"]}
            for k, j in sorted(jobs.items())
        },
        "last_two_triggered": {k: bool(j.get("last_two_triggered")) for k, j in jobs.items()},
        "primary_seed": pr.PRIMARY_SEED,
        "primary_seed_flag": pr.PRIMARY_SEED_FLAG,
    }
    return {
        "readouts": readouts,
        "scale": np.maximum(std, lm.METRIC_FLOOR_STD),
        "mean_latent": mean_latent,
        "sysid_coef": rec["sysid"]["coef"],
        "jobs": jobs,
        "paths": {
            "r_plate": str(fits / "r_plate.npz"),
            "r8": str(fits / "r8.npz"),
            "mean_latent": str(fits / "mean_latent.npy"),
        },
    }


def load_models(chain: dict, seed: int) -> dict:
    out = {}
    for arm in ("W", "N"):
        job = chain["jobs"][f"{arm}-{int(seed)}"]
        out[arm] = tr.load_model(
            job["checkpoint"],
            seed=int(seed),
            metadata=job["metadata"],
            expected_sha256=job["checkpoint_sha256"],
        )
    return out


def worker_config(chain: dict, *, readouts: dict | None = None, with_r8: bool = False) -> dict:
    """The workers' configuration: R-plate, the mean latent, the primary seed's W and N, and (when
    fitted) R-S and R-N of the primary seed. R8 only for the privileged H-read."""
    w = pr.REUSED
    config = {
        "r_plate": chain["paths"]["r_plate"],
        "r_plate_sha256": w["r_plate"],
        "mean_latent": chain["paths"]["mean_latent"],
        "mean_latent_sha256": w["mean_latent"],
        "models": {
            arm: {
                "path": chain["jobs"][f"{arm}-{pr.PRIMARY_SEED}"]["checkpoint"],
                "sha256": chain["jobs"][f"{arm}-{pr.PRIMARY_SEED}"]["checkpoint_sha256"],
                "seed": pr.PRIMARY_SEED,
                "metadata": chain["jobs"][f"{arm}-{pr.PRIMARY_SEED}"]["metadata"],
            }
            for arm in ("W", "N")
        },
        "torch_threads": pr.WORKER_TORCH_THREADS,
    }
    if readouts is not None:
        for name in ("r_s", "r_n"):
            config[name] = readouts[name]["path"]
            config[f"{name}_sha256"] = readouts[name]["sha256"]
    if with_r8:
        config["r8"], config["r8_sha256"] = chain["paths"]["r8"], w["r8"]
    return config


# ----- cohorts and attempts (TASK-077's runner's helpers) --------------------------------------
class Cohorts:
    def __init__(self, args, pool, p_readout, encoder, report):
        self.debug = bool(args.debug)
        self.pool, self.p_readout, self.encoder, self.report = pool, p_readout, encoder, report
        self.resets: dict[int, dict] = {}
        self.est: dict[int, dict] = {}
        self.digests: dict[str, str] = {}

    def seeds(self, role: str) -> tuple[int, ...]:
        seeds = pr.check_seeds(role, pr.seeds_of(role, debug=self.debug), debug=self.debug)
        if role not in self.digests:
            resets = {s: pr.reset_of(s) for s in seeds}
            self.resets |= resets
            self.digests[role] = lp.plan_digest({str(s): v for s, v in resets.items()})
            est = cohort_estimates(self.pool, self.p_readout, self.encoder, seeds, resets, 1800.0)
            self.report.setdefault("render_disagreements", {})[role] = est.pop(
                "_render_disagreements", {}
            )
            self.est |= est
            self.report["resets_digest"] = dict(self.digests)
        return seeds


def streamed_estimates(readout, token_chunks) -> np.ndarray:
    """TASK-077's ``streamed_estimates`` (R17.30, R17.36), unchanged."""
    from embodied_jepa import first_policy as fp
    from embodied_jepa import info_ceiling as ic

    grams, norms = [], []
    for tokens in token_chunks:
        g_rt, norm = ic.cross_gram(np.asarray(tokens, np.float64), readout.features)
        grams.append(g_rt)
        norms.append(norm)
        del tokens
    if len(norms) > 1 and len(norms[-1]) == 1:
        raise fp.GuardError("G-cohort: a one-row final chunk is not bit-identical (R17.36)")
    out = readout.readout.predict(np.concatenate(grams), np.concatenate(norms))
    if not np.isfinite(out).all():
        raise fp.GuardError("G-finite: a readout estimate is not finite")
    return out


def cohort_estimates(pool, readout, encoder, seeds, resets, cap: float, *, chunk=None) -> dict:
    """TASK-077's ``cohort_estimates`` (streamed tokens, R17.30), unchanged."""
    from embodied_jepa import first_policy_perception as fpp

    chunk = ESTIMATE_CHUNK if chunk is None else int(chunk)
    if chunk < 1 or chunk % CROSS_GRAM_BLOCK:
        raise ValueError(f"the estimate chunk must be a multiple of {CROSS_GRAM_BLOCK}")
    tasks = [{"kind": "frame", "seed": s, "reset": resets[s]} for s in seeds]
    frames, disagreements = hz.render_majority(pool, tasks, cap, "cohort frames")
    for s, f in zip(seeds, frames, strict=True):
        if f["seed"] != s:
            raise lp.GuardError("G-cohort: a rendered frame is not its seed's")
    chunks = (
        np.concatenate([fpp.featurise(encoder, f["frame"][None]) for f in frames[lo : lo + chunk]])
        for lo in range(0, len(frames), chunk)
    )
    estimates = streamed_estimates(readout, chunks)
    out = {
        s: {
            "estimates": estimates[i].tolist(),
            "frame_sha256": frames[i]["post_look_frame_sha256"],
            "state_sha256": frames[i]["state_sha256"],
            "truth_xy": frames[i]["truth_xy"],
        }
        for i, s in enumerate(seeds)
    }
    if disagreements:
        out["_render_disagreements"] = disagreements
    return out


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
            "wall_seconds": pr.CAPS_SECONDS["per_attempt"],
            "move_offset": pr.move_offset(s).tolist(),
        }
        tasks.append(task | (per_seed(s) if per_seed else {}) | extra)
    return tasks


def run_arm(pool, tasks, what: str) -> list[dict]:
    records = pool.map(tasks, MAP_CAP_SECONDS, what)
    arm = tasks[0]["arm"]
    if arm not in pr.PRIVILEGED_ARMS:
        for r in records:
            if r.get("blocked") is None and (
                not r.get("privileged_ok", False) or r.get("task_truth_in_controller", 1) != 0
            ):
                raise lp.GuardError(f"G-privileged: {arm} seed {r['seed']} read task truth")
    log(f"{what}: {sum(bool(r['success']) for r in records)}/{len(records)}")
    return records


def successes(records) -> list[bool]:
    return [bool(r["success"]) for r in records]


def lean(records) -> list[dict]:
    out = []
    for item in hz.strip(records):
        item = dict(item)
        for key in ("frame405", "corpus", "captured"):
            item.pop(key, None)
        if "kpred" in item:
            item["kpred"] = {k: v for k, v in item["kpred"].items() if k != "plate_path"}
        out.append(item)
    return out


def arm_summary(records) -> dict:
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
        "refused_before_405": sum(1 for r in records if not r.get("commit")),
        "fallbacks": sum(bool(r["commit"]["fallback"]) for r in commits),
        "clip_binding_fraction": (
            sum(bool(r["commit"].get("clipped")) for r in commits) / len(commits)
            if commits
            else None
        ),
        "landing_miss_cm": stats([r["commit"]["landing_miss_cm"] for r in commits
                                  if r["commit"].get("landing_miss_cm") is not None]),
        "fixed_point_error_cm": stats([r["commit"]["fixed_point_error_cm"] for r in commits
                                       if r["commit"].get("fixed_point_error_cm") is not None]),
        "seconds": {
            "median": float(np.median([r["seconds"] for r in records])),
            "max": float(np.max([r["seconds"] for r in records])),
        },
    }  # fmt: skip


def sim_setup(report, args, manifest, config: dict, workers: int):
    sim_preflight(report, args, manifest)
    base = {"p3_checkpoint": hz.p3_checkpoint(Path(args.evidence))}
    report["workers"] = workers
    pool = Pool(workers, {"torch_threads": pr.WORKER_TORCH_THREADS} | base | config)
    p_readout, encoder = hz.refit_p_readout(report, pool, Path(args.evidence), report["_run1"])
    report["first_render_utc"] = hz.utc()
    return pool, Cohorts(args, pool, p_readout, encoder, report)


def tau_commit(args) -> float:
    if pr.K0_PRIME_MEASURED is not None:
        return float(pr.K0_PRIME_MEASURED["tau_commit_cm"])
    if args.debug:
        return DEBUG["tau_commit_cm"]
    raise lp.GuardError("G-frozen: tau_commit is not measured yet (K0′)")


def tau_curve() -> tuple[dict, int]:
    """The pooled curve once K0′ is in the frozen block; TASK-077's K curve before (dry run)."""
    if pr.K0_PRIME_MEASURED is not None:
        return dict(pr.K0_PRIME_MEASURED["pooled"]["counts_pooled"]), pr.POOLED_RESETS
    return dict(pr.K_COUNTS), pr.K_RESETS


# ----- K0′ (CPU, before the freeze) --------------------------------------------------------------
def stage_k0(report, args, fields: Fields, manifest) -> str:
    """K0′ on K′'s 32 fresh resets (§8 step 2): H-final(commit)'s converged aim plus a planted
    error at each level (salt 8207), every level on the same resets; the ceiling, r_K′, the palm
    speed at 405; the pooled tau_commit over K and K′."""
    pool, co = sim_setup(report, args, manifest, {}, int(args.workers or pr.SIM_WORKERS))
    try:
        seeds = co.seeds("K")
        counts, levels = {}, {}
        for level in pr.TAU_LEVELS_CM:
            tasks = attempt_tasks(
                "H-final-planted",
                seeds,
                co,
                lambda s, level=level: {"planted_m": pr.planted_error_m(s, level)},
            )
            records = run_arm(pool, tasks, f"K0′ planted {level} cm")
            counts[level] = sum(successes(records))
            levels[level] = records
        k_prime_counts = {str(k): v for k, v in counts.items()}
        fields.set(
            "tau",
            {
                "counts_k_prime": k_prime_counts,
                "failed_seeds": {
                    str(k): [int(r["seed"]) for r in recs if not r["success"]]
                    for k, recs in levels.items()
                },
                "arms": {str(k): arm_summary(r) for k, r in levels.items()},
            },
        )
        ceiling_records = levels[0.0]
        ceiling = successes(ceiling_records)
        fields.set("ceiling", {"N_K_prime_0": int(sum(ceiling)), "per_reset": ceiling})
        s1 = pr.RULE["s1"]
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
        fields.set("r_k", {"r_K_prime": r_k if r_k is not None else "undefined"})
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
        palm = history["palm_speed_405_cm"]["median"]
        decision = pr.decide_k0_prime(
            k_prime_counts=k_prime_counts,
            ceiling=int(sum(ceiling)),
            r_k=r_k,
            palm_speed_median_cm=palm if palm is not None else float("inf"),
        )
        fields.set("decision", decision)
        return decision["row"]
    finally:
        report["pool_close"] = pool.close()


# ----- Stage C′ -----------------------------------------------------------------------------------
def stage_corpus(report, args, fields: Fields, manifest) -> str:
    """The fresh corpus (§5.2): gate-P's aims from p̂, contrast-T's from the true plate; (a, b)
    from salt 8202; the split (salt 8203) fixed before collection."""
    config = {"r_plate": str(Path(args.old_fits) / "r_plate.npz")}
    config["r_plate_sha256"] = pr.REUSED["r_plate"]
    pool, co = sim_setup(report, args, manifest, config, int(args.workers or pr.SIM_WORKERS))
    folder = Path(args.output) / "corpus"
    folder.mkdir()
    try:
        seeds = co.seeds("corpus")
        split = pr.corpus_split(seeds, debug=args.debug)
        fields.set("split", split)
        half_of = {int(s): h for h, v in split.items() for s in v}
        entries, excluded = {}, {}
        for lo in range(0, len(seeds), CORPUS_CHUNK):
            part = seeds[lo : lo + CORPUS_CHUNK]

            def per(s):
                a, b = pr.corpus_aim(s)
                return {"a": a, "b_m": b, "aim_from": pr.AIM_FROM[half_of[int(s)]]}

            tasks = attempt_tasks("collect-f", part, co, per)
            for r in run_arm(pool, tasks, f"fresh corpus {part[0]}-{part[-1]}"):
                arrays = r.get("corpus") or {"complete": False, "why": "blocked"}
                if r["blocked"] is not None or not arrays["complete"]:
                    excluded[int(r["seed"])] = (
                        r.get("termination_reason") if r["blocked"] else arrays["why"]
                    )
                    continue
                if arrays["aim_from"] != pr.AIM_FROM[half_of[int(r["seed"])]]:
                    raise lp.GuardError("G-corpus: a root's aim is not its half's construction")
                entries[int(r["seed"])] = pof.write_root(folder, r["seed"], arrays)
            check_disk(report, folder, pr.DISK_MIN_GIB)
        fields.set("roots", len(entries))
        fields.set("excluded", {str(k): v for k, v in sorted(excluded.items())})
        sealed = pof.seal_corpus(
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
        decision = pr.decide_corpus(split, excluded)
        fields.set("decision", decision)
        return decision["row"]
    finally:
        report["pool_close"] = pool.close()


# ----- Stage R: featurisation (GPU) ---------------------------------------------------------------
def open_fresh(args) -> dict:
    expected = args.corpus_sha256
    if not args.debug and not expected:
        raise lp.GuardError("G-split: the sealed fresh corpus manifest's sha256 is required")
    manifest = pof.open_corpus(Path(args.corpus), expected)
    if bool(manifest["provenance"].get("debug")) != bool(args.debug):
        raise lp.GuardError("G-split: a debug corpus never feeds a real run and the reverse")
    return manifest


def stage_featurise(report, args, fields: Fields, manifest) -> str:
    import torch

    rt.gpu_guard(report, min_free_gib=pr.GPU["min_free_gib"], require_lock=True)
    devices.configure_determinism("cuda", strict=True)
    torch.set_num_threads(6)
    corpus = open_fresh(args)
    report["corpus_sha256"] = corpus["_sha256"]
    result = pof.featurise_fresh(Path(args.corpus), corpus, Path(args.output) / "features",
                                 device="cuda")  # fmt: skip
    fields.set("featurisation", result)
    fields.set("anchor", result["anchor"])
    return "FEATURISED"


def fresh_chain(args, report: dict) -> dict:
    corpus = open_fresh(args)
    feat = completed(Path(args.features).parent / "report.json", ("FEATURISED",), args.debug)
    if feat.get("corpus_sha256") != corpus["_sha256"]:
        raise lp.GuardError("G-split: the featurise report was made from another corpus")
    files = feat["fields"]["featurisation"]["files_sha256"]
    report["fresh_feature_files_verified"] = off.verify_feature_files(
        Path(args.features), files, pr.HALVES
    )
    report["corpus_sha256"] = corpus["_sha256"]
    return {"corpus_sha256": corpus["_sha256"]}


# ----- the shared offline core: fits on old roots, readings and offline aims ----------------------
def chunks_for(pool, data: dict, what: str) -> tuple[np.ndarray, np.ndarray, float]:
    started = time.monotonic()
    got = pool.map(pof.chunk_tasks(data), MAP_CAP_SECONDS, what)
    chunks, feasible = pof.collect_chunks(got, data["n"])
    return chunks, feasible, time.monotonic() - started


def fit_phase(report, args, chain, fit: dict, folder: Path, *, crossfit: bool) -> dict:
    """The roll-outs and the fits on the fit roots (G-fresh: old roots only), per model seed;
    the reported cross-fit when asked."""
    import torch

    torch.set_num_threads(1)
    pr.check_fit_roots(fit["seeds"])
    pool = Pool(int(args.workers or pr.WM_WORKERS), worker_config(chain)
                | {"p3_checkpoint": hz.p3_checkpoint(Path(args.evidence))})  # fmt: skip
    try:
        chunks, feasible, seconds = chunks_for(pool, fit, "stand-in chunks of the fit roots")
    finally:
        report["pool_close_chunks"] = pool.close()
    out = {
        "fit_roots": {"n": fit["n"], "by_split": {s: int((fit["split"] == s).sum())
                                                   for s in np.unique(fit["split"])}},
        "standin": {"infeasible": int((~feasible).sum()), "seconds": seconds,
                    "chunks_sha256": pof.array_sha256(chunks)},
        "seeds": {},
    }  # fmt: skip
    readouts, preds = {}, {}
    for s in pr.MODEL_SEEDS:
        t0 = time.monotonic()
        models = load_models(chain, s)
        pred = pof.rollouts(models["W"], models["N"], fit["start"], chunks, chain["mean_latent"])
        t1 = time.monotonic()
        fitted = pof.fit_readouts(pred, fit["plate_r"], fit["seeds"])
        t2 = time.monotonic()
        saved = pof.save_readouts(folder, s, fitted)
        item = {"readouts": saved, "rollout_seconds": t1 - t0, "fit_seconds": t2 - t1}
        if crossfit:
            item["crossfit"] = pof.crossfit(pred, fit["plate_r"], fit["seeds"], fit["split"])
            item["crossfit_seconds"] = time.monotonic() - t2
        out["seeds"][str(s)] = item
        readouts[s] = {"fitted": fitted, "models": models, "saved": saved}
        preds[s] = pred
        log(f"seed {s}: fits done ({time.monotonic() - t0:.0f} s)")
    return {"record": out, "readouts": readouts, "preds": preds}


def release_models(phase: dict) -> None:
    """Free the main process's models and fit-root predictions before the world-model workers
    start (G-memory: each worker holds about 1.7 GiB)."""
    import gc

    for item in phase["readouts"].values():
        item.pop("models", None)
    phase["preds"].clear()
    gc.collect()


def read_phase(chain, fitted: dict, ev: dict, eval_chunks) -> tuple[dict, dict]:
    """e_S, e_N, e_L per seed on the evaluated roots, the encoded ceiling (R8 on the encoded frame
    at r, R0's form) and R8 on W's stand-in prediction (reported continuity)."""
    r8 = chain["readouts"]["r8"]
    ceiling = off.errors_cm(r8.predict(ev["enc_r"]), ev["plate_r"])
    readings = {"R0_encoded_ceiling": pr.median_ci(ceiling), "seeds": {}}
    preds = {}
    for s in pr.MODEL_SEEDS:
        models = fitted[s]["models"]
        pred = pof.rollouts(
            models["W"], models["N"], ev["start"], eval_chunks, chain["mean_latent"]
        )
        e = pof.readout_errors(fitted[s]["fitted"], pred, ev["plate_r"])
        stats = pof.readout_statistics(e)
        stats["frozen_R8_on_S"] = pr.median_ci(off.errors_cm(r8.predict(pred["S"]), ev["plate_r"]))
        stats["e_S_minus_ceiling"] = pr.median_difference_ci(e["e_S"], ceiling)
        readings["seeds"][str(s)] = stats
        preds[s] = {"pred": pred, "errors": e}
    return readings, preds


def aims_phase(report, args, chain, fitted: dict, ev: dict, p_hat, tau: float) -> dict:
    """Every arm's offline aims (§9.4) on the evaluated roots with the primary seed's readouts,
    mapped through the tau curve; W's predicted plate over the grid for the echo slope."""
    counts, resets = tau_curve()
    saved = fitted[pr.PRIMARY_SEED]["saved"]
    config = worker_config(chain, readouts=saved)
    config["p3_checkpoint"] = hz.p3_checkpoint(Path(args.evidence))
    pool = Pool(int(args.workers or pr.WM_WORKERS), config)
    latents = ev["start"]
    arms, slopes, seconds = {}, None, {}
    try:
        for arm in (*pr.R_ARMS, *pr.OFFLINE_REPORTED_ARMS):
            t0 = time.monotonic()
            tasks = pof.offline_aim_tasks(
                arm, ev, p_hat, latents, tau, sysid_coef=chain["sysid_coef"], keep=True
            )
            got = pool.map(tasks, MAP_CAP_SECONDS, f"offline aims {arm}")
            arms[arm] = pof.aim_summary(got, ev, counts, resets)
            if arm == "W":
                slopes = pof.echo_slopes(got)
            seconds[arm] = time.monotonic() - t0
            log(f"offline aims {arm}: {arms[arm]['predicted_count_of_64']:.1f}/64 "
                f"({seconds[arm]:.0f} s)")  # fmt: skip
    finally:
        report["pool_close_aims"] = pool.close()
    w = arms["W"]["predicted_count_of_64"]
    return {
        "primary_seed": pr.PRIMARY_SEED,
        "primary_seed_flag": pr.PRIMARY_SEED_FLAG,
        "tau_commit_cm": tau,
        "tau_counts": counts,
        "tau_resets": resets,
        "arms": arms,
        "w_minus": {a: w - arms[a]["predicted_count_of_64"] for a in pr.TWINS},
        "echo_slope_W": slopes,
        "seconds": seconds,
        "note": "predicted counts are predictions, never closed-loop counts; p-hat from R-plate "
        "on the stored (CUDA) full tokens at 405",
    }


# ----- R18.13's dry run (development) ---------------------------------------------------------
def stage_dryrun(report, args, fields: Fields, manifest) -> str:
    """R18.13: Stage R's own fit, reading and offline-aim code with the readouts fitted on the
    1 745 train and old-gate roots and every arm's offline aims on TASK-077's 250 val roots,
    mapped through K0's curve. Development only; it sets no bar and gates nothing. Optimistic
    twice over: val selected the checkpoints, and val's aims are built from the true plate."""
    import torch

    torch.set_num_threads(1)
    chain = old_chain(args, report, pr.OLD_SPLITS)
    limit = DEBUG["fit_roots_per_split"] if args.debug else None
    fit = pof.load_roots(Path(args.old_features), ("train", "gate"), limit=limit)
    ev = pof.load_roots(
        Path(args.old_features), ("val",), limit=DEBUG["eval_roots"] if args.debug else None,
        full405=True,
    )  # fmt: skip
    report["roots"] = {"fit": fit["n"], "eval_val": ev["n"]}
    tau = 1.0 if pr.K0_PRIME_MEASURED is None else tau_commit(args)
    t0 = time.monotonic()
    phase = fit_phase(report, args, chain, fit, Path(args.output) / "fits", crossfit=True)
    fields.set("fits", phase["record"])
    t_fit = time.monotonic() - t0
    pool = Pool(int(args.workers or pr.WM_WORKERS), worker_config(chain)
                | {"p3_checkpoint": hz.p3_checkpoint(Path(args.evidence))})  # fmt: skip
    try:
        ev_chunks, ev_feasible, _ = chunks_for(pool, ev, "stand-in chunks of the val roots")
    finally:
        report["pool_close_eval_chunks"] = pool.close()
    readings, preds = read_phase(chain, phase["readouts"], ev, ev_chunks)
    readings["eval_standin_infeasible"] = int((~ev_feasible).sum())
    fields.set("readings", readings)
    t_read = time.monotonic() - t0 - t_fit
    p_hat = pof.r_plate_readings(chain["readouts"]["r_plate"], ev["full405"])
    curve = {}
    for s in pr.MODEL_SEEDS:
        curve[str(s)] = pof.learning_curve(
            phase["preds"][s], fit["plate_r"], fit["seeds"], preds[s]["pred"], ev["plate_r"]
        )
    at_15 = {
        str(s): {k: float(np.mean(v <= 1.5 * tau)) for k, v in preds[s]["errors"].items()}
        for s in pr.MODEL_SEEDS
    }
    release_models(phase)
    del preds
    aims = aims_phase(report, args, chain, phase["readouts"], ev, p_hat, tau)
    fields.set("offline_aims", aims)
    fields.set(
        "reported",
        {
            "r_plate_405_error_val": pr.median_ci(off.errors_cm(p_hat, ev["plate405"])),
            "learning_curve_on_val": curve,
            "at_1_5_tau": at_15,
        },
    )
    a = aims["arms"]
    rates = {k: v["predicted_count_of_64"] / 64.0 for k, v in a.items()}
    table = pr.mcnemar_pass_table(pr.S_RESETS)
    power = {
        t: {c: pr.twin_power(rates["W"], rates[t], c, index=i * 3 + j, table=table,
                             trials=args.trials or pr.POWER_TRIALS)
            for j, c in enumerate(pr.POWER_COUPLINGS)}
        for i, t in enumerate(pr.TWINS)
    }  # fmt: skip
    gates_form = pr.r_gates(
        r0=readings["R0_encoded_ceiling"],
        per_seed={int(s): v for s, v in readings["seeds"].items()},
        aims={k: v["predicted_count_of_64"] for k, v in a.items()},
        tau_commit_cm=tau,
    )
    fields.set(
        "summary",
        {
            "development_only": True,
            "sets_no_bar": True,
            "optimistic_twice": "val selected the checkpoints; val's aims are built from the true "
            "plate (TASK-077's corpus construction)",
            "w_minus_n_of_64": aims["w_minus"]["N"],
            "w_minus_n_below_10": bool(aims["w_minus"]["N"] < 10.0),
            "predicted_counts_of_64": {k: v["predicted_count_of_64"] for k, v in a.items()},
            "g_bar_power_at_w": pr.g_bar_power(rates["W"]),
            "twin_power_at_predicted_rates": power,
            "stage_r_forms_on_val_not_gates": gates_form,
            "seconds": {"fits": t_fit, "readings": t_read, "aims": aims["seconds"]},
        },
    )
    return "DRYRUN-DONE"


# ----- Stage R (CPU) ------------------------------------------------------------------------------
def stage_rgate(report, args, fields: Fields, manifest) -> str:
    """§8 step 5 after the featurisation: the fits on the 1 995 old roots (and their reported
    cross-fit), then ``first_outcome_utc``, then gate-P: R0-R3 on all three seeds, A1-A2 on the
    primary seed, and every reported reading (contrast-T, the confound's size, the learning curve,
    G1-G4, R-plate's error at 405 on F, 1.5 tau)."""
    import torch

    torch.set_num_threads(1)
    chain = old_chain(args, report, pr.OLD_SPLITS)
    fresh_chain(args, report)
    plan = json.loads(Path(args.plan).read_text())
    if plan.get("outcome") != "T-PLANNED" or plan.get("corpus_sha256") != pr.OLD_CORPUS_SHA256:
        raise lp.GuardError("G-plan: not TASK-077's T-PLANNED report (G1's bars)")
    bars = plan["fields"]["g1"]["bars"]
    tau = tau_commit(args)
    limit = DEBUG["fit_roots_per_split"] if args.debug else None
    fit = pof.load_roots(Path(args.old_features), pr.OLD_SPLITS, limit=limit)
    report["roots"] = {"fit": fit["n"]}
    phase = fit_phase(report, args, chain, fit, Path(args.output) / "fits", crossfit=True)
    fields.set("fits", phase["record"])
    report["first_outcome_utc"] = hz.utc()  # gate-P is opened only after this line
    halves = {h: pof.load_roots(Path(args.features), (h,), full405=True) for h in pr.HALVES}
    readings, aims_by = {}, None
    preds_by, chunks_by = {}, {}
    pool = Pool(int(args.workers or pr.WM_WORKERS), worker_config(chain)
                | {"p3_checkpoint": hz.p3_checkpoint(Path(args.evidence))})  # fmt: skip
    try:
        for h, data in halves.items():
            chunks_by[h] = chunks_for(pool, data, f"stand-in chunks of {h}")
    finally:
        report["pool_close_eval_chunks"] = pool.close()
    for h, data in halves.items():
        readings[h], preds_by[h] = read_phase(chain, phase["readouts"], data, chunks_by[h][0])
        readings[h]["standin_infeasible"] = int((~chunks_by[h][1]).sum())
    fields.set("readings", readings)
    gate = halves["gate_p"]
    confound = {
        str(s): pr.unpaired_median_difference_ci(
            preds_by["gate_p"][s]["errors"]["e_S"], preds_by["contrast_t"][s]["errors"]["e_S"]
        )
        for s in pr.MODEL_SEEDS
    }
    curve = {
        str(s): pof.learning_curve(phase["preds"][s], fit["plate_r"], fit["seeds"],
                                   preds_by["gate_p"][s]["pred"], gate["plate_r"])
        for s in pr.MODEL_SEEDS
    }  # fmt: skip
    dynamics = {}
    store = tr.RootStore.open(Path(args.features), "gate_p", mmap=True)
    for s in pr.MODEL_SEEDS:
        m = phase["readouts"][s]["models"]
        dynamics[str(s)] = pof.dynamics_report(m["W"], m["N"], store, chain["scale"], bars)
    at_15 = {
        str(s): {
            k: float(np.mean(v <= 1.5 * tau)) for k, v in preds_by["gate_p"][s]["errors"].items()
        }
        for s in pr.MODEL_SEEDS
    }
    e_s_gate = {s: preds_by["gate_p"][s]["errors"]["e_S"] for s in pr.MODEL_SEEDS}
    release_models(phase)
    del preds_by
    aims_by = aims_phase(report, args, chain, phase["readouts"], gate, gate["p_hat405"], tau)
    fields.set("offline_aims", aims_by)
    r_plate_f = {
        h: pr.median_ci(
            off.errors_cm(
                pof.r_plate_readings(chain["readouts"]["r_plate"], d["full405"]), d["plate405"]
            )
        )
        for h, d in halves.items()
    }
    fields.set(
        "reported",
        {
            "confound_gate_p_minus_contrast_t_e_S": confound,
            "learning_curve_on_gate_p": curve,
            "g1_g4_on_gate_p": dynamics,
            "r_plate_405_error_on_F": r_plate_f,
            "p_hat_logged_vs_stored_tokens_max_cm": float(100.0 * np.abs(
                pof.r_plate_readings(chain["readouts"]["r_plate"], gate["full405"])
                - gate["p_hat405"]).max()),
            "at_1_5_tau": at_15,
            "e_S_gate_p_median_cm": {str(s): float(np.median(v)) for s, v in e_s_gate.items()},
        },
    )  # fmt: skip
    gp = readings["gate_p"]
    gates = pr.r_gates(
        r0=gp["R0_encoded_ceiling"],
        per_seed={int(s): v for s, v in gp["seeds"].items()},
        aims={k: v["predicted_count_of_64"] for k, v in aims_by["arms"].items()},
        tau_commit_cm=tau,
    )
    fields.set("gates", gates)
    decision = pr.decide_r(gates)
    fields.set("decision", decision)
    return decision["row"]


# ----- Stages D and S -----------------------------------------------------------------------------
def stage_closed(report, args, fields: Fields, manifest) -> str:
    rows = tuple(pr.R_ROWS) if args.debug else ("R-PASS",)
    chain = old_chain(args, report, ())
    stage_r = completed(args.stage_r, rows, args.debug)
    fresh = open_fresh(args)
    if stage_r.get("corpus_sha256") != fresh["_sha256"]:
        raise lp.GuardError("G-split: Stage R read another fresh corpus")
    saved = stage_r["fields"]["fits"]["seeds"][str(pr.PRIMARY_SEED)]["readouts"]
    folder = Path(args.stage_r).parent / "fits"
    readouts = {k: v | {"path": str(folder / Path(v["path"]).name)} for k, v in saved.items()}
    for name in ("r_s", "r_n"):
        pof.load_readout(readouts[name]["path"], readouts[name]["sha256"])
    config = worker_config(chain, readouts=readouts, with_r8=True)
    report["primary_seed"] = pr.PRIMARY_SEED
    pool, co = sim_setup(report, args, manifest, config, int(args.workers or pr.WM_WORKERS))
    try:
        seeds = co.seeds(args.cohort)
        report["first_outcome_utc"] = hz.utc()
        common = {"tau_commit_cm": tau_commit(args), "a_lo": pr.A_LO}
        return closed_core(pool, co, seeds, args.cohort, common, chain["sysid_coef"], fields)
    finally:
        report["pool_close"] = pool.close()


def closed_core(pool, co, seeds, role: str, common: dict, coef, fields: Fields) -> str:
    """TASK-077's ``closed_core`` (R17.20: a reset refused before 405 is a fail-fail pair; L-shuf's
    foreign frame from the next reset that reached 405), with this task's ladders."""
    arms = pr.D_ARMS if role == "D" else pr.S_ARMS
    records = {"W": run_arm(pool, attempt_tasks("W", seeds, co, **common), f"{role} W")}
    reached = [lm.reached_405(r) for r in records["W"]]
    foreign = {}
    for i, s in enumerate(seeds):
        j = lm.foreign_reached(i, reached)
        if j is None and reached[i]:
            raise lp.GuardError("G-frames: no other reset reached 405 for L-shuf's frame")
        foreign[s] = None if j is None else records["W"][j]["frame405"]
    for arm in arms:
        if arm == "W":
            continue
        per, extra = None, dict(common)
        if arm == "L-shuf":

            def per(s):
                return {"foreign_frame": foreign[s]}

        if arm == "H-sysid":
            extra["sysid_coef"] = coef
        if arm == "H-read":
            extra |= {"readout": "r8", "grid": pr.TOKEN_GRID, "read_step": pr.READ_STEP}
        records[arm] = run_arm(pool, attempt_tasks(arm, seeds, co, per, **extra), f"{role} {arm}")
    refused = {
        a: [int(r["seed"]) for r in recs if not lm.reached_405(r)] for a, recs in records.items()
    }
    if role == "S":
        first = list(seeds[: pr.DETERMINISM_RESETS])
        again = run_arm(pool, attempt_tasks("W", first, co, **common), "S W determinism")
        determinism = pr.determinism_check(records["W"][: len(first)], again)
        fields.set("determinism", determinism)
        if not determinism["ok"]:
            raise lp.GuardError(f"G-determinism: W's re-run differs: {determinism['resets']}")
    fields.set(
        "arms",
        {a: arm_summary(r) | {"attempts": lean(r)} for a, r in records.items()}
        | {"refused_before_405": {"by_arm": refused, "W": len(refused["W"])}},
    )
    outcomes = {a: successes(r) for a, r in records.items()}
    if role == "D":
        decision = pr.decide_d({a: sum(v) for a, v in outcomes.items()})
    else:
        decision = pr.decide_s(outcomes)
    fields.set("decision", decision)
    return decision["row"]


# ----- §10's power --------------------------------------------------------------------------------
def stage_simulate(report, args, fields: Fields, manifest) -> str:
    """G-bar's exact power, the twin table (§10) and, with ``--dryrun``, the power at the dry
    run's predicted rates; with ``--k0-prime`` too, the dry run's per-root aim errors re-mapped
    through K0′'s pooled curve (the curve slot the freeze fills)."""
    started = time.monotonic()
    trials = int(args.trials or pr.POWER_TRIALS)
    result = {
        "g_bar": {str(r): pr.g_bar_power(r) for r in (0.875, 0.89, 0.906, 0.922, 0.9375, 0.969)},
        "twin_table": pr.power_table(trials=trials),
        "trials": trials,
    }
    if args.dryrun:
        dry = completed(args.dryrun, ("DRYRUN-DONE",), args.debug)
        arms = dry["fields"]["offline_aims"]["arms"]
        curves = {"K (TASK-077, 32 resets)": (dict(pr.K_COUNTS), pr.K_RESETS)}
        if args.k0_prime:
            k0p = completed(args.k0_prime, ("K0′-PASS",), args.debug)
            pooled = k0p["fields"]["decision"]["pooled"]["counts_pooled"]
            curves["pooled K and K′ (64 resets)"] = (pooled, pr.POOLED_RESETS)
        table = pr.mcnemar_pass_table(pr.S_RESETS)
        mapped = {}
        for name, (counts, resets) in curves.items():
            counts_by = {a: pr.predicted_count(v["aim_errors_cm"], counts, resets)
                         for a, v in arms.items()}  # fmt: skip
            rates = {a: c / 64.0 for a, c in counts_by.items()}
            mapped[name] = {
                "predicted_counts_of_64": counts_by,
                "g_bar_power_at_w": pr.g_bar_power(rates["W"]),
                "twin_power": {
                    t: {c: pr.twin_power(rates["W"], rates[t], c, index=100 + i * 3 + j,
                                         table=table, trials=trials)
                        for j, c in enumerate(pr.POWER_COUPLINGS)}
                    for i, t in enumerate(pr.TWINS)
                },
            }  # fmt: skip
        result["dry_run"] = mapped
    result["seconds"] = time.monotonic() - started
    fields.set("power", result)
    return "SIMULATED"


def stage_tests(report, args, fields: Fields, manifest) -> str:
    fields.set("tests", run_full_tests())
    return "TESTS-PASS"


# ----- the run ------------------------------------------------------------------------------------
RUNNERS = {
    "tests": stage_tests,
    "k0": stage_k0,
    "corpus": stage_corpus,
    "featurise": stage_featurise,
    "rgate": stage_rgate,
    "closed": stage_closed,
    "simulate": stage_simulate,
    "dryrun": stage_dryrun,
}


def gpu_memory_peak() -> dict:
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return {"recorded": False, "reason": "CUDA was not initialised in this process"}
    device = torch.cuda.current_device()
    return {
        "recorded": True,
        "max_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
        "max_memory_reserved_gib": torch.cuda.max_memory_reserved(device) / GIB,
    }


def stage_cap(args) -> float:
    if args.stage == "closed":
        return pr.CAPS_SECONDS[args.cohort]
    return pr.CAPS_SECONDS.get(STAGE_CAPS.get(args.stage, ""), 4 * 3600.0)


def run(args) -> dict:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    report = {
        "protocol": pr.PROTOCOL,
        "task": pr.TASK,
        "stage": args.stage,
        "debug": bool(args.debug),
        "outcome": None,
        "stages": {},
        "argv": sys.argv[1:],
        "paths": {"output": str(output.resolve())},
        "started_utc": hz.utc(),
        "salts": pr.SALTS,
        "seed_ranges": pr.DEBUG_RANGES if args.debug else pr.SEED_RANGES,
    }
    fields = Fields(report, args.stage)
    clock = hz.Clock(stage_cap(args))
    watch = rt.MemoryWatch(
        pr.MEMORY["ceiling_gib"] * GIB,
        pr.MEMORY["sample_seconds"],
        measure=pr.MEMORY["measure"],
        disk_path=ROOT,
        min_disk_free_bytes=int(pr.DISK_MIN_GIB * GIB),
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
        report["load_average_at_end"] = list(os.getloadavg())
        fields.check()
        report["outcome"] = f"{outcome}-DEBUG" if args.debug else outcome
    except BaseException as error:  # noqa: BLE001 - every failure, signal included, is V
        guards.void(report, error)
    finally:
        if args.stage in GPU_STAGES:
            try:
                report["gpu_memory"] = gpu_memory_peak()
            except Exception as error:  # noqa: BLE001
                report["gpu_memory"] = {"recorded": False, "reason": repr(error)}
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
    parser.add_argument("--debug-skip-tests", action="store_true", help="debug only")
    parser.add_argument("--workers", type=int, default=None, help="debug only (1-6)")
    parser.add_argument("--evidence", help="TASK-072 run-1's evidence root (G-repro)")
    parser.add_argument("--log", help="also write stdout and stderr to this new file")
    parser.add_argument("--tests-record", help="the GPU stage: a `tests` stage report at HEAD")
    parser.add_argument("--corpus", help="the sealed fresh corpus folder")
    parser.add_argument("--corpus-sha256", help="the sealed fresh corpus manifest's sha256")
    parser.add_argument("--features", help="the fresh corpus's featurisation folder")
    parser.add_argument("--old-features", help="TASK-077's featurisation folder")
    parser.add_argument("--old-fits", help="TASK-077's Stage O fits folder")
    parser.add_argument("--models", nargs="*", default=[], help="TASK-077's six job reports")
    parser.add_argument("--plan", help="TASK-077's Stage T plan report (G1's bars, reported)")
    parser.add_argument("--stage-r", help="Stage R's report (closed)")
    parser.add_argument("--cohort", choices=("D", "S"))
    parser.add_argument("--dryrun", help="simulate: the dry run's report")
    parser.add_argument("--k0-prime", help="simulate: K0′'s report (the pooled curve)")
    parser.add_argument("--trials", type=int, default=None, help="simulate / dryrun power trials")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers is not None and (not args.debug or not 1 <= args.workers <= 6):
        parser.error("--workers is for debug runs only, from 1 to 6")
    if args.debug_skip_tests and not args.debug:
        parser.error("--debug-skip-tests is for debug runs only")
    if args.stage in (*SIM_STAGES, "rgate", "dryrun") and not args.evidence:
        parser.error(f"{args.stage} needs --evidence (G-repro, the stand-in workers)")
    if args.stage in GPU_STAGES and not args.tests_record:
        parser.error(f"{args.stage} needs --tests-record (G-tests for the GPU stage)")
    if args.stage in OLD_STAGES and not (args.old_features and args.old_fits and args.models):
        parser.error(f"{args.stage} needs --old-features, --old-fits and --models (G-hash)")
    if args.stage == "corpus" and not args.old_fits:
        parser.error("corpus needs --old-fits (R-plate builds gate-P's aims)")
    if args.stage in ("featurise", "rgate", "closed"):
        if not args.corpus:
            parser.error(f"{args.stage} needs --corpus (G-split)")
        if not args.debug and not args.corpus_sha256:
            parser.error(f"{args.stage} needs --corpus-sha256 (§8 step 4)")
    if args.stage == "rgate" and not (args.features and args.plan):
        parser.error("rgate needs --features and --plan")
    if args.stage == "closed" and not (args.cohort and args.stage_r):
        parser.error("closed needs --cohort and --stage-r")
    (ROOT / "outputs").mkdir(exist_ok=True)
    if args.log:
        open_log(Path(args.log))
    report = run(args)
    return 0 if report.get("outcome") not in (None, "V") else 1


def open_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "x", buffering=1)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(handle.fileno(), 1)
    os.dup2(handle.fileno(), 2)
    sys.stdout.reconfigure(line_buffering=True)


if __name__ == "__main__":
    sys.exit(main())
