"""TASK-081's runner: Stages D and S of ``docs/experiments/apple_lewm_commit_precision_v2.md``.

Frozen block ``src/embodied_jepa/lewm_cp_v2.py``; workers ``lewm_cp_v2_runtime.py``. This runner
loads no other script (``tests/test_no_runner_imports.py``): the pieces of TASK-080's runner it
needs (TASK-077's chain ``old_chain``, the worker configuration, the simulation preflight, the
cohort estimates, ``completed``, ``attempt_tasks``, ``run_arm``, ``arm_summary``, ``lean``, the
G-tests, G-quiet and G-disk helpers) are **copied verbatim** below from
``scripts/run_lewm_pr_v2.py`` at its pinned sha256 ``e30388ce...d6fa``; a test checks that each
copy's source equals TASK-080's (only ``run_arm`` is renamed ``run_arm_carried``). Nothing of
TASK-076, TASK-077 or TASK-080 is edited.

One invocation runs one stage and writes ``<output>/report.json`` (it refuses an existing output):

* ``closed``   -- Stage D (``--cohort D``, 70100-70115) or Stage S (``--cohort S``, 70200-70327),
  CPU, from TASK-080's Stage R report (R-S and R-N of seed 66800, by content sha256); S needs this
  task's D-PASS report;
* ``simulate`` -- §8's power (salt 8304), CPU.

Every stage checks the clean tree, TASK-076's pins, TASK-077's 13 pins and frozen block, TASK-080's
frozen block, seven file pins and protocol document, the solver file's sha256 and, once FROZEN,
this task's own pins and protocol document. ``--debug`` simulates debug seeds 71900-71999 only;
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

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import devices  # noqa: E402
from embodied_jepa import first_policy_v2_linux as fpl  # noqa: E402
from embodied_jepa import lewm_c1m_v2 as lm  # noqa: E402
from embodied_jepa import lewm_c1m_v2_offline as off  # noqa: E402
from embodied_jepa import lewm_cp_v2 as cpv  # noqa: E402
from embodied_jepa import lewm_planner_v2 as lp  # noqa: E402
from embodied_jepa import lewm_pr_v2 as pr  # noqa: E402
from embodied_jepa import lewm_pr_v2_offline as pof  # noqa: E402
from embodied_jepa import plate_twin_v2_harness as hz  # noqa: E402
from embodied_jepa import run_guards as rg  # noqa: E402
from embodied_jepa import run_tools as rt  # noqa: E402

fpl.configure_headless()
GIB = 2**30
log = hz.log
PROTOCOL_DOCUMENT = ROOT / cpv.DOCUMENT
TASK080_DOCUMENT = ROOT / pr.DOCUMENT
MAP_CAP_SECONDS = 3600.0
QUIET_POLL_SECONDS = 30.0
QUIET_WAIT_CAP_SECONDS = 4 * 3600.0
ESTIMATE_CHUNK = 128
CROSS_GRAM_BLOCK = 32
STAGES = ("closed", "simulate")
STAGE_FIELDS = {
    "closed": ("arms", "reported", "determinism", "decision"),
    "simulate": ("power",),
}
DRAFT_ALLOWED = ("simulate",)


class Pool(rg.BoundedPool):
    """``run_guards.BoundedPool`` on TASK-081's worker; a dead worker or a cap is a GuardError;
    every map of attempts runs TASK-071's G-look check (as TASK-080's pool)."""

    def __init__(self, workers: int, config: dict):
        from embodied_jepa import lewm_cp_v2_runtime as cpr

        super().__init__(workers, cpr.run_task, initializer=cpr.worker_init, initargs=(config,))
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
        report["fields"] = cpv.sentinel_fields(self.names)

    def set(self, name: str, value) -> None:
        if name not in self.names:
            raise KeyError(name)
        if cpv.is_missing(value):
            raise cpv.GuardError(f"G-sentinel: {name} set to a missing value")
        self.report["fields"][name] = value
        self.evaluated.add(name)

    def check(self) -> None:
        cpv.check_sentinel(self.report["fields"], self.evaluated, self.names)


Cohorts = SimpleNamespace  # the carried annotation: an object with .resets and .est


# ----- carried verbatim from scripts/run_lewm_pr_v2.py (TASK-080, pinned) ------------------------
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


def run_arm_carried(pool, tasks, what: str) -> list[dict]:
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


def open_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "x", buffering=1)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(handle.fileno(), 1)
    os.dup2(handle.fileno(), 2)
    sys.stdout.reconfigure(line_buffering=True)


# ----- preflight ----------------------------------------------------------------------------------
def git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def check_task080(report: dict) -> None:
    """G-frozen (TASK-080 carried): its frozen block is its pin, its seven pinned files and its
    protocol document match, and the solver file is the design note's."""
    manifest = json.loads((ROOT / cpv.TASK080_MANIFEST).read_text())
    if pr.frozen_sha256() != cpv.TASK080_FROZEN_SHA256 or (
        manifest.get("frozen_sha256_pin") != cpv.TASK080_FROZEN_SHA256
    ):
        raise lp.GuardError("G-frozen: TASK-080's frozen block is not its pin")
    report["task080_pins_at_preflight"] = len(hz.check_pins(manifest["hashes"]))
    report["task080_protocol_document"] = check_task080_document(manifest)
    got = hz.sha256_file(ROOT / cpv.SOLVER_FILE)
    if got != cpv.SOLVER_FILE_SHA256:
        raise lp.GuardError(f"G-hash: {cpv.SOLVER_FILE} {got[:12]} is not its pin")
    report["solver_file_sha256"] = got
    for path in cpv.OWN_FILES:
        try:
            git("ls-files", "--error-unmatch", path)
        except subprocess.CalledProcessError as error:
            raise lp.GuardError(f"G-hash: {path} is not committed") from error
    report["own_code_git_blobs"] = {p: git("hash-object", p) for p in cpv.OWN_FILES}


def check_task080_document(manifest: dict) -> dict:
    want = manifest.get("protocol_document_sha256")
    got = hz.sha256_file(TASK080_DOCUMENT) if TASK080_DOCUMENT.exists() else None
    if not want or got != want:
        raise lp.GuardError(f"G-hash: TASK-080's protocol document {str(got)[:12]} is not its pin")
    return {"path": pr.DOCUMENT, "sha256": got}


def check_frozen_pin(manifest: dict) -> dict:
    sha = cpv.frozen_sha256()
    pin = manifest.get("frozen_sha256_pin")
    if cpv.STATUS == "FROZEN":
        if pin != sha:
            raise lp.GuardError(f"G-frozen: the frozen block {sha[:12]} is not the pin {pin}")
    elif pin is not None:
        raise lp.GuardError("G-frozen: a DRAFT protocol carries no pin")
    return {"status": cpv.STATUS, "frozen_sha256": sha, "pin": pin}


def check_protocol_document(manifest: dict) -> dict:
    want = manifest.get("protocol_document_sha256")
    got = hz.sha256_file(PROTOCOL_DOCUMENT) if PROTOCOL_DOCUMENT.exists() else None
    if not want or got != want:
        raise lp.GuardError(f"G-hash: the protocol document {str(got)[:12]} is not the pin")
    return {"path": cpv.DOCUMENT, "sha256": got}


def preflight(report: dict, args) -> dict:
    """Every stage's common guards."""
    rt.assert_local_import(ROOT, report)
    manifest = json.loads((ROOT / lm.TASK076_MANIFEST).read_text())
    report["task076_pins_at_preflight"] = len(hz.check_pins(manifest["hashes"]))
    check_task077(report)
    check_task080(report)
    dirty = hz.tracked_tree_dirty()
    report["revision"], report["tracked_tree_dirty"] = hz.revision(), bool(dirty)
    hz.check_clean(dirty)
    report["protocol_status"] = cpv.STATUS
    report["frozen_sha256_at_run"] = cpv.frozen_sha256()
    own = json.loads((ROOT / cpv.MANIFEST).read_text())
    report["frozen_pin_check"] = check_frozen_pin(own)
    if cpv.STATUS == "FROZEN":
        if not own.get("hashes"):
            raise lp.GuardError("G-hash: a FROZEN manifest pins its files")
        report["own_pins_at_preflight"] = len(hz.check_pins(own["hashes"]))
        report["protocol_document_check"] = check_protocol_document(own)
    if cpv.STATUS != "FROZEN" and args.stage not in DRAFT_ALLOWED and not args.debug:
        raise lp.GuardError(f"G-frozen: {args.stage} runs only after the freeze (STATUS FROZEN)")
    report["thread_env"] = {k: os.environ.get(k) for k in cpv.THREAD_ENV}
    if report["thread_env"] != cpv.THREAD_ENV:
        raise lp.GuardError(f"G-threads: {report['thread_env']} is not {cpv.THREAD_ENV}")
    cpv.check_seed_ranges()
    check_disk(report, ROOT, cpv.DISK_MIN_GIB)
    wait_quiet(report, args)
    if args.debug and args.debug_skip_tests:
        report["g_tests"] = {"skipped": "debug run with --debug-skip-tests (nothing is read)"}
    else:
        report["g_tests"] = run_full_tests()
    report["load_average_after_preflight"] = list(os.getloadavg())
    return manifest


# ----- inputs (G-hash, §2.2) ----------------------------------------------------------------------
def stage_r_readouts(args, report: dict) -> dict:
    """TASK-080's Stage R report at its recorded sha256 (R-PASS, not debug) and the primary
    seed's R-S and R-N at their content sha256s."""
    from embodied_jepa import lewm_pr_v2_offline as pof

    got = hz.sha256_file(args.stage_r)
    if got != cpv.REUSED["stage_r_report_sha256"]:
        raise lp.GuardError(f"G-hash: the Stage R report {got[:12]} is not TASK-080's")
    stage_r = completed(args.stage_r, ("R-PASS",), False)
    saved = stage_r["fields"]["fits"]["seeds"][str(cpv.PRIMARY_SEED)]["readouts"]
    folder = Path(args.stage_r).parent / "fits"
    readouts = {}
    for name in ("r_s", "r_n"):
        if saved[name]["sha256"] != cpv.REUSED[name]:
            raise lp.GuardError(f"G-hash: {name} in the Stage R report is not the pinned one")
        path = str(folder / Path(saved[name]["path"]).name)
        pof.load_readout(path, cpv.REUSED[name])
        readouts[name] = {"path": path, "sha256": cpv.REUSED[name]}
    report["stage_r"] = {"report": str(Path(args.stage_r).resolve()), "sha256": got,
                         "readouts": readouts}  # fmt: skip
    return readouts


def stage_d_report(args, debug: bool) -> dict:
    """This task's D-PASS report (Stage S only after D-PASS, §6.1 step 4)."""
    report = json.loads(Path(args.stage_d).read_text())
    want = "D-PASS-DEBUG" if debug else "D-PASS"
    if (
        report.get("task") != cpv.TASK
        or report.get("protocol") != cpv.PROTOCOL
        or report.get("outcome") != want
        or bool(report.get("debug")) != bool(debug)
        or report.get("cohort") != "D"
    ):
        raise lp.GuardError(f"G-plan: {args.stage_d} is not this task's {want} report")
    return {"path": str(Path(args.stage_d).resolve()), "sha256": hz.sha256_file(args.stage_d)}


# ----- the closed loop ----------------------------------------------------------------------------
def run_arm(pool, tasks, what: str) -> list[dict]:
    """TASK-080's ``run_arm`` (G-privileged) plus G-solver on every candidate attempt."""
    records = run_arm_carried(pool, tasks, what)
    arm = tasks[0]["arm"]
    if arm in cpv.SOLVER_OF:
        from embodied_jepa import lewm_cp_v2_runtime as cpr

        want = cpv.SOLVER_OF[arm]
        for r in records:
            got = cpr.logged_solver(r)
            if r.get("arm") != arm or r.get("solver") != want or got not in (None, want):
                raise lp.GuardError(f"G-solver: {arm} seed {r['seed']} ran {got!r}")
    return records


def closed_core(pool, co, seeds, role: str, common: dict, coef, fields: Fields, *,
                debug: bool) -> str:  # fmt: skip
    """TASK-080's ``closed_core`` with this task's arms and ladders (R17.20: a reset refused before
    405 is a fail-fail pair; L-shuf's foreign frame from the next reset W reached 405 on)."""
    arms = cpv.D_ARMS if role == "D" else cpv.S_ARMS
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
        records[arm] = run_arm(pool, attempt_tasks(arm, seeds, co, per, **extra),
                               f"{role} {arm}")  # fmt: skip
    refused = {
        a: [int(r["seed"]) for r in recs if not lm.reached_405(r)] for a, recs in records.items()
    }
    if role == "S":
        first = list(seeds[: cpv.DETERMINISM_RESETS])
        again = run_arm(pool, attempt_tasks("W", first, co, **common), "S W determinism")
        determinism = cpv.determinism_check(records["W"][: len(first)], again)
        fields.set("determinism", determinism)
        if not determinism["ok"]:
            raise lp.GuardError(f"G-determinism: W's re-run differs: {determinism['resets']}")
    fields.set(
        "arms",
        {a: arm_summary(r) | {"attempts": lean(r)} for a, r in records.items()}
        | {"refused_before_405": {"by_arm": refused, "W": len(refused["W"])}},
    )
    fields.set("reported", reported(records, n=len(seeds)))
    outcomes = {a: successes(r) for a, r in records.items()}
    if role == "D":
        decision = cpv.decide_d({a: sum(v) for a, v in outcomes.items()})
    else:
        decision = cpv.decide_s(outcomes, debug=debug)
    fields.set("decision", decision)
    return decision["row"]


def _stats(values) -> dict:
    v = [float(x) for x in values if x is not None]
    if not v:
        return {"n": 0}
    out = {
        "n": len(v),
        "median": float(np.median(v)),
        "p12_5": float(np.percentile(v, 12.5)),
        "p87_5": float(np.percentile(v, 87.5)),
        "max": float(np.max(v)),
    }
    if len(v) >= 2:
        out["median_ci95"] = cpv.median_ci(v)["ci95"]
    return out


def reported(records: dict, *, n: int) -> dict:
    """§7.3, reported only: the solver's effect, aim errors against H-final(commit)'s aim, the
    tau-curve predictions, the final-distance distributions and affine_local's diagnostics."""
    out: dict = {}
    if "W-frozen" in records:
        w, wf = successes(records["W"]), successes(records["W-frozen"])
        capped = [
            r["seed"] for r in records["W-frozen"]
            if (r.get("decisions") or [{}])[0].get("world_model", {}).get("converged") is False
        ]  # fmt: skip
        on = [i for i, r in enumerate(records["W-frozen"]) if r["seed"] in capped]
        out["solver_effect"] = cpv.solver_effect(w, wf) | {
            "w_frozen_not_converged_seeds": capped,
            "on_not_converged": {
                "W": sum(w[i] for i in on),
                "W-frozen": sum(wf[i] for i in on),
                "resets": len(on),
            },  # fmt: skip
        }
    ref = {}
    if "H-final" in records:
        ref = {r["seed"]: np.asarray(r["commit"]["target"], np.float64)
               for r in records["H-final"] if r.get("commit")}  # fmt: skip
    per_arm = {}
    for arm, recs in records.items():
        commits = [r for r in recs if r.get("commit")]
        entry = {}
        if ref and arm in cpv.CANDIDATE_ARMS | {"H-rule", "H-sysid"}:
            err = [100.0 * float(np.linalg.norm(np.asarray(r["commit"]["target"]) - ref[r["seed"]]))
                   for r in commits if r["seed"] in ref]  # fmt: skip
            entry["aim_error_vs_h_final_cm"] = _stats(err)
        fp_err = [r["commit"].get("fixed_point_error_cm") for r in commits]
        fp_err = [e for e in fp_err if e is not None]
        if fp_err:
            entry["tau_curve_predicted_count"] = pr.predicted_count(
                fp_err, cpv.TAU_CURVE, cpv.TAU_CURVE_RESETS, cohort=n
            )
        dist = {"all": [], "successes": [], "failures": []}
        for r in recs:
            d = r.get("final_distance_cm")
            if d is None:
                continue
            dist["all"].append(d)
            dist["successes" if r.get("success") else "failures"].append(d)
        entry["final_distance_cm"] = {k: _stats(v) for k, v in dist.items()}
        entry["final_distance_cm"]["within_0_5_of_4cm"] = int(
            sum(3.5 <= d <= 4.5 for d in dist["all"])
        )
        if cpv.SOLVER_OF.get(arm) == "affine_local":
            logs = [(r.get("decisions") or [{}])[0].get("world_model", {}) for r in commits]
            logs = [g for g in logs if g.get("solver") == "affine_local"]
            reasons: dict = {}
            for g in logs:
                if g.get("fallback_grid_argmin"):
                    reasons[g["fallback_reason"]] = reasons.get(g["fallback_reason"], 0) + 1
            eig = [g["eigenvalues_real"] for g in logs if g.get("eigenvalues_real")]
            entry["affine_local"] = {
                "decisions": len(logs),
                "fallbacks": reasons,
                "clipped": int(sum(bool(g.get("clipped")) for g in logs)),
                "points": _stats([g.get("points") for g in logs]),
                "residual_cm": _stats([g.get("residual_cm") for g in logs]),
                "eigenvalue_real_min": _stats([min(e) for e in eig]),
                "eigenvalue_real_max": _stats([max(e) for e in eig]),
                "complex_pairs": int(sum(bool(g.get("eigenvalues_complex")) for g in logs)),
            }
        per_arm[arm] = entry
    out["per_arm"] = per_arm
    out["primary_seed_flag"] = cpv.PRIMARY_SEED_FLAG
    return out


def stage_closed(report, args, fields: Fields, manifest) -> str:
    chain = old_chain(args, report, ())
    readouts = stage_r_readouts(args, report)
    if args.cohort == "S" and not args.debug:  # Stage S only after this task's D-PASS
        report["stage_d"] = stage_d_report(args, False)
    elif args.cohort == "S" and args.stage_d:
        report["stage_d"] = stage_d_report(args, True)
    config = worker_config(chain, readouts=readouts, with_r8=True)
    report["primary_seed"] = cpv.PRIMARY_SEED
    report["cohort"] = args.cohort
    workers = int(args.workers or cpv.WM_WORKERS)
    sim_preflight(report, args, manifest)
    base = {"p3_checkpoint": hz.p3_checkpoint(Path(args.evidence))}
    report["workers"] = workers
    pool = Pool(workers, {"torch_threads": cpv.WORKER_TORCH_THREADS} | base | config)
    try:
        p_readout, encoder = hz.refit_p_readout(report, pool, Path(args.evidence),
                                                report["_run1"])  # fmt: skip
        report["first_render_utc"] = hz.utc()
        seeds = cpv.check_seeds(args.cohort, cpv.seeds_of(args.cohort, debug=args.debug),
                                debug=args.debug)  # fmt: skip
        resets = {s: cpv.reset_of(s) for s in seeds}
        report["resets_digest"] = lp.plan_digest({str(s): v for s, v in resets.items()})
        est = cohort_estimates(pool, p_readout, encoder, seeds, resets, 1800.0)
        report["render_disagreements"] = est.pop("_render_disagreements", {})
        co = SimpleNamespace(resets=resets, est=est)
        report["first_outcome_utc"] = hz.utc()
        common = {"tau_commit_cm": cpv.TAU_COMMIT_CM, "a_lo": cpv.A_LO}
        return closed_core(pool, co, seeds, args.cohort, common, chain["sysid_coef"], fields,
                           debug=args.debug)  # fmt: skip
    finally:
        report["pool_close"] = pool.close()


def stage_simulate(report, args, fields: Fields, manifest) -> str:
    started = time.monotonic()
    power = cpv.power_tables(trials=int(args.trials or cpv.POWER_TRIALS))
    power["seconds"] = time.monotonic() - started
    fields.set("power", power)
    return "SIMULATED"


RUNNERS = {"closed": stage_closed, "simulate": stage_simulate}


def stage_cap(args) -> float:
    return cpv.CAPS_SECONDS[args.cohort if args.stage == "closed" else "simulate"]


def run(args) -> dict:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    report = {
        "protocol": cpv.PROTOCOL,
        "task": cpv.TASK,
        "stage": args.stage,
        "cohort": args.cohort,
        "debug": bool(args.debug),
        "outcome": None,
        "argv": sys.argv[1:],
        "paths": {"output": str(output.resolve())},
        "started_utc": hz.utc(),
        "salts": cpv.SALTS,
        "seed_ranges": cpv.DEBUG_RANGES if args.debug else cpv.SEED_RANGES,
    }
    fields = Fields(report, args.stage)
    clock = hz.Clock(stage_cap(args))
    watch = rt.MemoryWatch(
        cpv.MEMORY["ceiling_gib"] * GIB,
        cpv.MEMORY["sample_seconds"],
        measure=cpv.MEMORY["measure"],
        disk_path=ROOT,
        min_disk_free_bytes=int(cpv.DISK_MIN_GIB * GIB),
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
    parser.add_argument("--old-features", help="TASK-077's featurisation folder")
    parser.add_argument("--old-fits", help="TASK-077's Stage O fits folder")
    parser.add_argument("--models", nargs="*", default=[], help="TASK-077's six job reports")
    parser.add_argument("--stage-r", help="TASK-080's Stage R report (R-S and R-N)")
    parser.add_argument("--stage-d", help="this task's Stage D D-PASS report (closed --cohort S)")
    parser.add_argument("--cohort", choices=("D", "S"))
    parser.add_argument("--trials", type=int, default=None, help="simulate: power trials")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers is not None and (not args.debug or not 1 <= args.workers <= 6):
        parser.error("--workers is for debug runs only, from 1 to 6")
    if args.debug_skip_tests and not args.debug:
        parser.error("--debug-skip-tests is for debug runs only")
    if args.stage == "closed":
        if not args.cohort:
            parser.error("closed needs --cohort")
        if not (args.evidence and args.old_features and args.old_fits and args.models):
            parser.error("closed needs --evidence, --old-features, --old-fits and --models")
        if not args.stage_r:
            parser.error("closed needs --stage-r (TASK-080's Stage R report)")
        if args.cohort == "S" and not args.debug and not args.stage_d:
            parser.error("closed --cohort S needs --stage-d (this task's D-PASS)")
    elif args.cohort:
        parser.error("--cohort is for closed only")
    (ROOT / "outputs").mkdir(exist_ok=True)
    if args.log:
        open_log(Path(args.log))
    report = run(args)
    return 0 if report.get("outcome") not in (None, "V") else 1


if __name__ == "__main__":
    sys.exit(main())
