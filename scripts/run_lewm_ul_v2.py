"""TASK-082's runner: every stage of ``docs/experiments/apple_lewm_unknown_law_v2.md`` (DRAFT).

Frozen block ``src/embodied_jepa/lewm_ul_v2.py``; workers ``lewm_ul_v2_runtime.py``; offline
stages ``lewm_ul_v2_offline.py``; training ``lewm_ul_v2_train.py``. Guards from
``embodied_jepa.run_tools`` and TASK-076's harness module. This runner loads no other script
(``tests/test_no_runner_imports.py``): the helpers of TASK-077's runner it needs are **copied
verbatim** below from ``scripts/run_lewm_c1m_v2.py`` (a test checks each copy's source).

One invocation runs one stage and writes ``<output>/report.json`` (it refuses an existing output):

* ``tests``      -- the full pytest suite at HEAD, recorded (G-tests for GPU jobs);
* ``k0``         -- K0 on K (72400-72463), CPU, before the freeze (on a reported GO only);
* ``corpus``     -- Stage C, the 2 000-root corpus ``apple-ul-v2`` with its ceiling labels (CPU);
* ``labelcheck`` -- Stage 0 only (debug): the labelling look-ahead leaves a root's executed
  trajectory unchanged (each debug root collected with and without the label);
* ``featurise``  -- Stage O's featurisation (CUDA, through ``scripts/gpu_run.sh``);
* ``readouts``   -- Stage O's admission and the train-only fits, the learned tier included (CPU);
* ``train``      -- one Stage T job (CUDA, through ``scripts/gpu_run.sh``): ``--job W-<seed>`` or
  ``N-<seed>``, the carried budget U = 95 000 (R20.5; no calibration jobs);
* ``gates``      -- Stage G (CPU, one torch thread): the readouts on predicted latents fitted on
  train + val, then gate-P: G1-G4, R0-R3, A1-A2;
* ``gscale``     -- Stage 0's scale probe of Stage G's fits, roll-outs and dynamics gates at the
  real sizes on synthetic latents (CPU);
* ``closed``     -- Stage D (``--cohort D``) or Stage S (``--cohort S``) (CPU);
* ``simulate``   -- §10's power simulation, which fixes delta by R20.10's rule (CPU).

Every GPU stage runs through ``scripts/gpu_run.sh --wait --min-free-gib 8 --board --who
oej:task082-<stage> -- ...``. Every CPU stage runs the full test suite itself first (G-tests); every
stage checks the clean tree, TASK-076's pins, TASK-077's, TASK-080's and TASK-081's frozen blocks,
file pins and documents, the pinned development files (``plate_law_dev.py``,
``plate_law_dev_runtime.py``, ``commit_precision_dev.py``) and, once FROZEN, TASK-082's own pins.
While the protocol is DRAFT only ``tests``, ``simulate``, ``gscale``, ``k0`` and ``--debug`` runs
are allowed. ``--debug`` simulates debug seeds 74800-74899 only; nothing in it is read.
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
from embodied_jepa import lewm_cp_v2 as cpv  # noqa: E402
from embodied_jepa import lewm_planner_v2 as lp  # noqa: E402
from embodied_jepa import lewm_pr_v2 as pr  # noqa: E402
from embodied_jepa import lewm_ul_v2 as ul  # noqa: E402
from embodied_jepa import lewm_ul_v2_offline as ulo  # noqa: E402
from embodied_jepa import lewm_ul_v2_train as ult  # noqa: E402
from embodied_jepa import plate_twin_v2_harness as hz  # noqa: E402
from embodied_jepa import run_guards as rg  # noqa: E402
from embodied_jepa import run_tools as rt  # noqa: E402

fpl.configure_headless()
GIB = 2**30
log = hz.log
PROTOCOL_DOCUMENT = ROOT / ul.DOCUMENT
MAP_CAP_SECONDS = 3600.0
QUIET_POLL_SECONDS = 30.0
QUIET_WAIT_CAP_SECONDS = 4 * 3600.0
CORPUS_CHUNK = 60  # corpus roots per pool map (bounds the frames held at once)
ESTIMATE_CHUNK = 128  # post-look frames featurised at once for P-3's estimates (R17.30)
CROSS_GRAM_BLOCK = 32  # info_ceiling.cross_gram's row block
STAGES = (
    "tests",
    "k0",
    "corpus",
    "labelcheck",
    "featurise",
    "readouts",
    "train",
    "gates",
    "gscale",
    "closed",
    "simulate",
)
CPU_STAGES = ("k0", "corpus", "labelcheck", "readouts", "gates", "gscale", "closed", "simulate")
GPU_STAGES = ("featurise", "train")
SIM_STAGES = ("k0", "corpus", "labelcheck", "gates", "closed")  # stages with a simulator pool
DRAFT_ALLOWED = ("tests", "simulate", "gscale", "k0")
DEBUG_ONLY = ("labelcheck",)
R_PLATE_STAGES = ("k0", "corpus", "labelcheck", "readouts", "closed")  # read the carried R-plate
DEBUG = {  # declared debug stand-ins (nothing in a debug run is read)
    "tau_commit_cm": 1.0,
    "tau_counts": {"0.0": 64, "0.5": 60, "1.0": 57, "1.5": 40, "2.0": 30, "3.0": 10},
}
STAGE_FIELDS = {
    "tests": ("tests",),
    "k0": ("tau", "ceiling", "r_k", "history", "clip_binding", "reported", "decision"),
    "corpus": ("split", "roots", "excluded", "labels", "sealed", "scale", "decision"),
    "labelcheck": ("pairs", "decision"),
    "featurise": ("featurisation", "anchor"),
    "readouts": ("admission", "fits", "decision"),
    "train": ("job", "record", "checkpoint"),
    "gates": (
        "primary_seed",
        "fits",
        "readings",
        "dynamics",
        "offline_aims",
        "reported",
        "gates",
        "decision",
    ),
    "gscale": ("synthetic", "probe", "caps"),
    "closed": ("arms", "reported", "determinism", "decision"),
    "simulate": ("power",),
}
STAGE_CAPS = {
    "k0": "K0",
    "corpus": "C",
    "labelcheck": "labelcheck",
    "featurise": "O_featurisation",
    "readouts": "O_readouts",
    "train": "T_job",
    "gates": "G",
    "gscale": "G",
    "simulate": "simulate",
}
FEATURE_SPLITS = {  # the splits each stage verifies before it reads them
    "readouts": ("train", "val"),
    "train": ("train", "val"),
    "gates": ("train", "val"),  # gate-P is verified after first_outcome_utc
    "closed": (),
}
O_ROWS_READ = {False: ("O-PASS",), True: tuple(ul.O_ROWS)}
G_ROWS_READ = {False: ("G-PASS",), True: tuple(r for r in ul.G_ROWS if r != "V")}


class Pool(rg.BoundedPool):
    """``run_guards.BoundedPool`` on TASK-082's worker; a dead worker or a cap is a GuardError;
    every map of attempts runs TASK-071's G-look check (as TASK-076's harness pool)."""

    def __init__(self, workers: int, config: dict):
        from embodied_jepa import lewm_ul_v2_runtime as urt

        super().__init__(workers, urt.run_task, initializer=urt.worker_init, initargs=(config,))
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
        report["fields"] = ul.sentinel_fields(self.names)

    def set(self, name: str, value) -> None:
        if name not in self.names:
            raise KeyError(name)
        if ul.is_missing(value):
            raise ul.GuardError(f"G-sentinel: {name} set to a missing value")
        self.report["fields"][name] = value
        self.evaluated.add(name)

    def get(self, name: str):
        return self.report["fields"][name]

    def check(self) -> None:
        ul.check_sentinel(self.report["fields"], self.evaluated, self.names)


# ----- carried verbatim from scripts/run_lewm_c1m_v2.py (TASK-077, pinned) -----------------------
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


def streamed_estimates(readout, token_chunks) -> np.ndarray:
    """``XYReadout.predict`` over all rows, from token chunks (R17.30, Erratum 2026-10-05).

    Each chunk's cross-Gram against the readout's training rows is computed by the pinned
    ``info_ceiling.cross_gram`` and the chunk's tokens are then dropped; the small [N, n_train]
    Gram and the norms are concatenated and the pinned kernel-ridge ``Readout.predict`` runs once
    on all N rows, exactly as ``XYReadout.predict`` does.

    Bit-identity is measured, not proved (correction 2026-10-05, R17.36; the #145 approval's
    note 1). It held at every size tested (tests and ``scripts/probe_task077_memory.py``), and
    a chunk of 100, not a multiple of ``CROSS_GRAM_BLOCK``, was bit-identical too, so the
    multiple-of-32 rule is conservative, not the operative condition. It fails when the final
    chunk has exactly one row: ``cross_gram``'s norm line (``einsum("ij,ij->i")``) takes a
    different reduction path on a one-row array, and the estimates then differ by about 1e-15.
    That cannot occur in TASK-077 (2 000 corpus seeds leave a tail of 80 at 128; K, D and S are
    one chunk each), and a one-row final chunk after another chunk is refused here."""
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
    """``plate_twin_v2_harness.cohort_estimates`` with the tokens streamed (R17.30).

    The pinned function featurises every seed's post-look frame at once and ``cross_gram`` then
    makes two float64 copies of them: at Stage C's 2 000 seeds that is 1.47 GiB of tokens held
    with 2.93 GiB of copies, which voided the first Stage C run on G-memory. Here the frames are
    rendered exactly as there (``render_majority``, the same tasks, cap and checks), and the
    tokens are featurised (batch size 1, ``first_policy_perception.featurise``, as there) and
    reduced to their cross-Gram ``chunk`` frames at a time. The output is the same dict, with
    estimates that were bit-identical wherever measured (see ``streamed_estimates`` for the one
    known exception, a one-row final chunk, which is refused). ``chunk`` is kept a multiple of
    ``CROSS_GRAM_BLOCK``: a conservative rule, not the condition for bit-identity (R17.36)."""
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


def successes(records) -> list[bool]:
    return [bool(r["success"]) for r in records]


def rebased(recorded: str, folder) -> str:
    """An artifact recorded by an earlier stage, found by its file name in ``folder``, the folder
    named on this stage's command line (Stage T onwards, R17.42).

    Stage O wrote its fits' paths relative to its own worktree (``outputs/task077-readouts-1/
    fits/...``) and a Stage T job writes its checkpoint's path relative to its own; a later stage
    run from another worktree, at a later revision, cannot open such a path. The file is
    therefore looked up in the folder the command line names (``--fits``, or the job report's
    own folder), and its identity is still checked by sha256 where it is read (the moments, R8,
    R-plate, the mean latent and every checkpoint). ``sysid.json`` is only required to exist: it
    is not read (H-sysid's coefficients come from the readouts report) and has no sha256. Only
    the file name is kept, which suits the flat ``fits/`` and job folders; an artifact recorded
    in a subfolder would need its relative path kept (R17.45)."""
    path = Path(folder).resolve() / Path(recorded).name
    if not path.is_file():
        raise lp.GuardError(f"G-split: {Path(recorded).name} is not in {Path(folder).resolve()}")
    return str(path)


def completed(path, rows, debug: bool) -> dict:
    """An earlier stage's report whose outcome is one of ``rows`` (``-DEBUG`` in a debug run;
    a debug report never feeds a real run and the reverse)."""
    report = json.loads(Path(path).read_text())
    allowed = {f"{r}-DEBUG" for r in rows} if debug else set(rows)
    if report.get("outcome") not in allowed or bool(report.get("debug")) != bool(debug):
        raise lp.GuardError(f"G-plan: {path} ended {report.get('outcome')}, not one of {rows}")
    return report


def check_same_corpus(earlier: dict, chain: dict, what: str) -> None:
    if earlier.get("corpus_sha256") != chain["corpus_sha256"]:
        raise lp.GuardError(f"G-split: the {what} report was made from another corpus")


def gpu_memory_peak() -> dict:
    """The CUDA caching allocator's peaks for this process (R17.41): torch's
    ``max_memory_allocated`` and ``max_memory_reserved`` since the process started, read at the
    end of a GPU stage (also after a V). The process is fresh, so no reset is needed. The driver's
    own overhead (the CUDA context, about 0.4 GiB on this card) is not in either number."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return {"recorded": False, "reason": "CUDA was not initialised in this process"}
    device = torch.cuda.current_device()
    return {
        "recorded": True,
        "device": int(device),
        "name": torch.cuda.get_device_name(device),
        "max_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
        "max_memory_reserved_gib": torch.cuda.max_memory_reserved(device) / GIB,
        "total_gib": torch.cuda.get_device_properties(device).total_memory / GIB,
        "note": "torch.cuda.max_memory_allocated / max_memory_reserved at the stage's end; the "
        "CUDA context's own memory is not included",
    }


def open_log(path: Path) -> None:
    """Point this process's stdout and stderr (and so its workers' and G-tests') at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "x", buffering=1)  # refuses an existing log
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(handle.fileno(), 1)
    os.dup2(handle.fileno(), 2)
    sys.stdout.reconfigure(line_buffering=True)


# ----- preflight ----------------------------------------------------------------------------------
def git(*args) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def _document_pin(manifest: dict, document: str, what: str) -> dict:
    want = manifest.get("protocol_document_sha256")
    path = ROOT / document
    got = hz.sha256_file(path) if path.exists() else None
    if not want or got != want:
        raise lp.GuardError(f"G-hash: {what}'s protocol document {str(got)[:12]} is not its pin")
    return {"path": document, "sha256": got}


def check_carried(report: dict) -> None:
    """G-frozen and G-hash on everything carried (§2.2): TASK-077's, TASK-080's and TASK-081's
    frozen blocks are their pins, their file pins and (TASK-080, TASK-081) protocol documents
    match, the pinned development files have their Stage-0 sha256, and TASK-082's own files are
    committed."""
    carried = {}
    for what, manifest_path, module, pin, document in (
        ("TASK-077", ul.TASK077_MANIFEST, lm, ul.TASK077_FROZEN_SHA256, None),
        ("TASK-080", ul.TASK080_MANIFEST, pr, ul.TASK080_FROZEN_SHA256, pr.DOCUMENT),
        ("TASK-081", ul.TASK081_MANIFEST, cpv, ul.TASK081_FROZEN_SHA256, cpv.DOCUMENT),
    ):
        manifest = json.loads((ROOT / manifest_path).read_text())
        if module.frozen_sha256() != pin or manifest.get("frozen_sha256_pin") != pin:
            raise lp.GuardError(f"G-frozen: {what}'s frozen block is not its pin")
        item = {"frozen_sha256": pin, "file_pins": len(hz.check_pins(manifest["hashes"]))}
        if document is not None:
            item["document"] = _document_pin(manifest, document, what)
        carried[what] = item
    dev = {}
    for path, want in ul.PINNED_DEV_FILES.items():
        got = hz.sha256_file(ROOT / path)
        if got != want:
            raise lp.GuardError(f"G-hash: {path} {got[:12]} is not its Stage-0 pin")
        dev[path] = got
    carried["pinned_dev_files"] = dev
    report["carried_checks"] = carried
    for path in ul.OWN_FILES:
        try:
            git("ls-files", "--error-unmatch", path)
        except subprocess.CalledProcessError as error:
            raise lp.GuardError(f"G-hash: {path} is not committed") from error
    report["own_code_git_blobs"] = {p: git("hash-object", p) for p in ul.OWN_FILES}


def check_frozen_pin(manifest: dict) -> dict:
    """G-frozen (direct): after the freeze, the manifest's pin must equal this revision's frozen
    block; before it, the pin must be unset."""
    sha = ul.frozen_sha256()
    pin = manifest.get("frozen_sha256_pin")
    if ul.STATUS == "FROZEN":
        if pin != sha:
            raise lp.GuardError(f"G-frozen: the frozen block {sha[:12]} is not the pin {pin}")
    elif pin is not None:
        raise lp.GuardError("G-frozen: a DRAFT protocol carries no pin")
    return {"status": ul.STATUS, "frozen_sha256": sha, "pin": pin}


def check_protocol_document(manifest: dict) -> dict:
    return _document_pin(manifest, ul.DOCUMENT, "TASK-082")


def preflight(report: dict, args) -> dict:
    """Every stage's common guards."""
    rt.assert_local_import(ROOT, report)
    manifest = json.loads((ROOT / lm.TASK076_MANIFEST).read_text())
    report["task076_pins_at_preflight"] = len(hz.check_pins(manifest["hashes"]))
    check_carried(report)
    dirty = hz.tracked_tree_dirty()
    report["revision"], report["tracked_tree_dirty"] = hz.revision(), bool(dirty)
    hz.check_clean(dirty)
    report["protocol_status"] = ul.STATUS
    report["frozen_sha256_at_run"] = ul.frozen_sha256()
    own = json.loads((ROOT / ul.MANIFEST).read_text())
    report["frozen_pin_check"] = check_frozen_pin(own)
    if ul.STATUS == "FROZEN":
        if not own.get("hashes"):
            raise lp.GuardError("G-hash: a FROZEN manifest pins its files")
        report["own_pins_at_preflight"] = len(hz.check_pins(own["hashes"]))
        report["protocol_document_check"] = check_protocol_document(own)
    if ul.STATUS != "FROZEN" and args.stage not in DRAFT_ALLOWED and not args.debug:
        raise lp.GuardError(f"G-frozen: {args.stage} runs only after the freeze (STATUS FROZEN)")
    if ul.STATUS == "FROZEN" and args.stage == "k0" and not args.debug:
        raise lp.GuardError("G-frozen: K0 runs once, before the freeze; it is not repeated")
    if args.stage in DEBUG_ONLY and not args.debug:
        raise lp.GuardError(f"G-frozen: {args.stage} is a Stage-0 debug check only")
    report["thread_env"] = {k: os.environ.get(k) for k in ul.THREAD_ENV}
    if report["thread_env"] != ul.THREAD_ENV:
        raise lp.GuardError(f"G-threads: {report['thread_env']} is not {ul.THREAD_ENV}")
    ul.check_seed_ranges()
    start_min = ul.DISK_MIN_START_GIB.get({"corpus": "C", "featurise": "O"}.get(args.stage, ""))
    check_disk(report, ROOT, start_min or ul.DISK_MIN_GIB)
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


# ----- the carried R-plate (§2.3, G-hash) ---------------------------------------------------------
def carried_r_plate(args, report: dict) -> dict:
    """TASK-077 Stage O's R-plate by content sha256, beside its O-PASS readouts report (file
    sha256): the 405 plate reading that builds every grid."""
    from embodied_jepa.models.latent_critic import RidgeReadout

    fits = Path(args.r_plate_fits)
    ro = fits.parent / "report.json"
    got = hz.sha256_file(ro)
    if got != ul.CARRIED["task077_readouts_report_sha256"]:
        raise lp.GuardError(f"G-hash: TASK-077's readouts report {got[:12]} is not the pinned one")
    record = json.loads(ro.read_text())
    if (
        record.get("outcome") != "O-PASS"
        or record.get("corpus_sha256") != ul.CARRIED["task077_corpus_sha256"]
    ):
        raise lp.GuardError("G-hash: not TASK-077's O-PASS readouts report of apple-c1m-v2")
    path = fits / "r_plate.npz"
    with np.load(path) as data:
        readout = RidgeReadout.from_state({k: data[k] for k in data.files})
    if readout.sha256() != ul.CARRIED["r_plate"]:
        raise lp.GuardError("G-hash: R-plate differs from TASK-077's content sha256")
    report["r_plate"] = {"path": str(path.resolve()), "sha256": ul.CARRIED["r_plate"],
                         "readouts_report_sha256": got}  # fmt: skip
    return {"path": str(path.resolve()), "sha256": ul.CARRIED["r_plate"], "readout": readout}


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
        seeds = ul.check_seeds(role, ul.seeds_of(role, debug=self.debug), debug=self.debug)
        if role not in self.digests:
            resets = {s: ul.reset_of(s) for s in seeds}
            self.resets |= resets
            self.digests[role] = lp.plan_digest({str(s): v for s, v in resets.items()})
            est = cohort_estimates(self.pool, self.p_readout, self.encoder, seeds, resets, 1800.0)
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
            "wall_seconds": ul.CAPS_SECONDS["per_attempt"],
            "move_offset": ul.move_offset(s).tolist(),
        }
        tasks.append(task | (per_seed(s) if per_seed else {}) | extra)
    return tasks


def run_arm(pool, tasks, what: str) -> list[dict]:
    """One arm's attempts: G-privileged (no task truth in a non-privileged arm), G-law (every
    record ran under U-sat) and G-solver (a candidate arm logged its declared solver)."""
    from embodied_jepa import lewm_ul_v2_runtime as urt

    records = pool.map(tasks, MAP_CAP_SECONDS, what)
    arm = tasks[0]["arm"]
    for r in records:
        urt.check_law(r)
        if (
            arm not in ul.PRIVILEGED_ARMS
            and r.get("blocked") is None
            and (not r.get("privileged_ok", False) or r.get("task_truth_in_controller", 1) != 0)
        ):
            raise lp.GuardError(f"G-privileged: {arm} seed {r['seed']} read task truth")
        if urt.is_candidate(arm):
            got = ((r.get("decisions") or [{}])[0].get("world_model") or {}).get("solver")
            if got not in (None, urt.solver_of(arm)):
                raise lp.GuardError(f"G-solver: {arm} seed {r['seed']} ran {got!r}")
    log(f"{what}: {sum(bool(r['success']) for r in records)}/{len(records)}")
    return records


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


def _stats(values) -> dict:
    v = [float(x) for x in values if x is not None]
    if not v:
        return {"n": 0}
    out = {
        "n": len(v),
        "median": float(np.median(v)),
        "p87_5": float(np.percentile(v, 87.5)),
        "max": float(np.max(v)),
    }
    if len(v) >= 2:
        out["median_ci95"] = ul.median_ci(v)["ci95"]
    return out


def arm_summary(records) -> dict:
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
        "landing_miss_cm": _stats([r["commit"].get("landing_miss_cm") for r in commits]),
        "plate_motion_cm": _stats([r["commit"].get("plate_motion_cm") for r in commits]),
        "seconds": {
            "median": float(np.median([r["seconds"] for r in records])),
            "max": float(np.max([r["seconds"] for r in records])),
        },
    }


def sim_setup(report, args, manifest, config: dict, workers: int):
    """The pool, P-3's refitted readout (G-repro) and the cohorts."""
    sim_preflight(report, args, manifest)
    base = {"p3_checkpoint": hz.p3_checkpoint(Path(args.evidence))}
    report["workers"] = workers
    pool = Pool(workers, {"torch_threads": ul.WORKER_TORCH_THREADS} | base | config)
    p_readout, encoder = hz.refit_p_readout(report, pool, Path(args.evidence), report["_run1"])
    report["first_render_utc"] = hz.utc()
    return pool, Cohorts(args, pool, p_readout, encoder, report)


def tau_commit(args) -> float:
    if ul.K0_MEASURED is not None:
        return float(ul.K0_MEASURED["tau_commit_cm"])
    if args.debug:
        return DEBUG["tau_commit_cm"]
    raise lp.GuardError("G-frozen: tau_commit is not measured yet (K0)")


def tau_curve() -> dict:
    if ul.K0_MEASURED is not None:
        return dict(ul.K0_MEASURED["tau"]["counts"])
    return dict(DEBUG["tau_counts"])


# ----- K0 (CPU, before the freeze) ---------------------------------------------------------------
def stage_k0(report, args, fields: Fields, manifest) -> str:
    """K0 on K's 64 fresh resets (§7.2): H-final(commit)'s converged aim plus a planted error at
    each level (salt 8408), every level on the same resets; the ceiling, r_K, the palm speed at 405,
    the aim-box clip-binding count (level 0, the box built from p-hat and h); H-rule (C1-M's
    written law, on p-hat) and H-now reported only."""
    r_plate = carried_r_plate(args, report)
    config = {"r_plate": r_plate["path"], "r_plate_sha256": r_plate["sha256"]}
    pool, co = sim_setup(report, args, manifest, config, int(args.workers or ul.SIM_WORKERS))
    try:
        seeds = co.seeds("K")
        counts, levels = {}, {}
        for level in ul.TAU_LEVELS_CM:
            tasks = attempt_tasks(
                "H-final-planted",
                seeds,
                co,
                lambda s, level=level: {"planted_m": ul.planted_error_m(s, level)},
                log_p_hat=bool(level == 0.0),
            )
            records = run_arm(pool, tasks, f"K0 planted {level} cm")
            counts[level] = sum(successes(records))
            levels[level] = records
        tau = ul.decide_tau(counts)
        fields.set(
            "tau",
            tau
            | {
                "failed_seeds": {str(k): [int(r["seed"]) for r in recs if not r["success"]]
                                 for k, recs in levels.items()},
                "arms": {str(k): arm_summary(r) for k, r in levels.items()},
            },
        )  # fmt: skip
        ceiling_records = levels[0.0]
        ceiling = successes(ceiling_records)
        fields.set("ceiling", {"N_K0": int(sum(ceiling)), "per_reset": ceiling})
        s1 = ul.LAW["s1"]
        distances, speeds, remaining = [], [], []
        for r in ceiling_records:
            k = r["kpred"]
            path = {int(t): np.asarray(v) for t, v in k.get("plate_path", {}).items()}
            if r["blocked"] is not None or s1 not in path:
                distances.append(None)
            else:
                distances.append(
                    {t: 100.0 * float(np.linalg.norm(v - path[s1])) for t, v in path.items()}
                )
                remaining.append(k.get("remaining_405_cm"))
            if k.get("palm_speed_405_cm") is not None:
                speeds.append(k["palm_speed_405_cm"])
        r_k = ul.read_step_k(distances)
        fields.set("r_k", {"r_K": r_k if r_k is not None else "undefined", "r": ul.READ_STEP})
        history = {
            "palm_speed_405_cm": _stats(speeds),
            "remaining_405_cm": _stats([v for v in remaining if v is not None]),
        }
        fields.set("history", history)
        clip = k0_clip_binding(ceiling_records)
        fields.set("clip_binding", clip)
        reported = {}
        for arm in ul.K0_REPORTED:
            records = run_arm(pool, attempt_tasks(arm, seeds, co, a_lo=ul.A_LO), f"K0 {arm}")
            outcome = successes(records)
            reported[arm] = {
                "arm": arm_summary(records),
                "ceiling_minus_arm": ul.paired_interval(ceiling, outcome),
                "note": "reported only; gates nothing (δ is fixed in Stage 0)",
            }
        fields.set("reported", reported)
        palm = history["palm_speed_405_cm"].get("median")
        decision = ul.decide_k0(
            tau=tau,
            ceiling=int(sum(ceiling)),
            r_k=r_k,
            palm_speed_median_cm=palm if palm is not None else float("inf"),
            clip_outside=clip["outside"],
        )
        fields.set("decision", decision)
        return decision["row"]
    finally:
        report["pool_close"] = pool.close()


def k0_clip_binding(records) -> dict:
    """§7.2's clip-binding check: each level-0 ceiling aim in the box coordinates of the box built
    from p-hat (R-plate's CPU reading of the 405 frame, logged) and h; a refused attempt has no
    aim and is not counted outside."""
    rows = []
    for r in records:
        first = (r.get("decisions") or [None])[0]
        if not r.get("commit") or first is None or "p_hat405" not in first:
            continue
        g = np.asarray(first.get("lookahead_aim", r["commit"]["target"]), np.float64)
        rows.append(ul.box_coordinates(g, first["p_hat405"], r["commit"]["h"]) | {
            "seed": int(r["seed"])})  # fmt: skip
    a = [abs(x["a"]) for x in rows]
    b = [abs(x["b_m"]) * 100.0 for x in rows]
    return {
        "outside": int(sum(x["outside"] for x in rows)),
        "aims": len(rows),
        "limit": ul.CLIP_OUTSIDE_MAX,
        "abs_a": _stats(a),
        "abs_b_cm": _stats(b),
        "per_reset": rows,
    }


# ----- Stage C ------------------------------------------------------------------------------------
def split_of(split: dict) -> dict[int, str]:
    return {int(s): name for name, seeds in split.items() for s in seeds}


def corpus_task(split_by_seed: dict, *, label: bool = True):
    def per(s):
        a, b = ul.corpus_aim(s)
        return {"a": a, "b_m": b, "aim_from": ul.AIM_FROM[split_by_seed[int(s)]], "label": label}

    return per


def stage_corpus(report, args, fields: Fields, manifest) -> str:
    """Stage C (§4.3): every root under U-sat, its label from one look-ahead in cloned state, its
    own aim from the true plate (train, val) or p-hat (gate-P); (a, b) from salt 8406; the split
    (salt 8407) fixed before collection."""
    r_plate = carried_r_plate(args, report)
    config = {"r_plate": r_plate["path"], "r_plate_sha256": r_plate["sha256"]}
    pool, co = sim_setup(report, args, manifest, config, int(args.workers or ul.SIM_WORKERS))
    folder = Path(args.output) / "corpus"
    folder.mkdir()
    started = time.monotonic()
    try:
        seeds = co.seeds("corpus")
        split = ul.corpus_split(seeds, debug=args.debug)
        fields.set("split", split)
        by = split_of(split)
        entries, excluded, labels, seconds, branch = {}, {}, {}, [], []
        collect_started = time.monotonic()
        for lo in range(0, len(seeds), CORPUS_CHUNK):
            part = seeds[lo : lo + CORPUS_CHUNK]
            tasks = attempt_tasks("collect-ul", part, co, corpus_task(by))
            for r in run_arm(pool, tasks, f"corpus {part[0]}-{part[-1]}"):
                seconds.append(float(r["seconds"]))
                label = ((r.get("decisions") or [{}])[0] or {}).get("label") or {}
                branch.append(label.get("branch_steps"))
                arrays = r.get("corpus") or {"complete": False, "why": "blocked"}
                if r["blocked"] is not None or not arrays["complete"]:
                    excluded[int(r["seed"])] = (
                        r.get("termination_reason") if r["blocked"] else arrays["why"]
                    )
                    continue
                if arrays["aim_from"] != ul.AIM_FROM[by[int(r["seed"])]]:
                    raise lp.GuardError("G-corpus: a root's aim is not its split's construction")
                if not arrays["labelled"]:
                    raise lp.GuardError("G-corpus: a kept root lacks its ceiling label")
                labels[int(r["seed"])] = bool(arrays["label_converged"])
                entries[int(r["seed"])] = ulo.write_root(folder, r["seed"], arrays)
            check_disk(report, folder, ul.DISK_MIN_GIB)
        collect_seconds = time.monotonic() - collect_started
        fields.set("roots", len(entries))
        fields.set("excluded", {str(k): v for k, v in sorted(excluded.items())})
        unconverged = {
            name: int(sum(1 for s in v if int(s) in labels and not labels[int(s)]))
            for name, v in split.items()
        }
        fields.set("labels", {"unconverged_by_split": unconverged, "labelled": len(labels),
                              "converged": int(sum(labels.values()))})  # fmt: skip
        sealed = ulo.seal_corpus(
            folder,
            entries,
            split,
            {
                "revision": report["revision"],
                "privileged_scripted_collector": True,
                "ceiling_labels_privileged": True,
                "learned_control": False,
                "debug": bool(args.debug),
                "resets_digest": co.digests.get("corpus"),
            },
        )
        fields.set("sealed", sealed)
        report["corpus_sha256"] = sealed["sha256"]
        per_root = _stats(seconds)
        workers = int(report["workers"])
        worst = per_root.get("max", 0.0) * ul.CORPUS_ROOTS / workers
        fields.set(
            "scale",
            {
                "per_root_seconds": per_root,
                "collect_wall_seconds": collect_seconds,
                "wall_seconds_per_root": collect_seconds / max(len(seeds), 1),
                "stage_seconds_so_far": time.monotonic() - started,
                "lookahead_branch_steps": _stats(branch),
                "worst_case_2000_roots_attempts_s": worst,
                "workers": workers,
                "root_bytes": _stats([(folder / f"{s}.npz").stat().st_size for s in entries]),
                "note": "the worst case is every root at the slowest measured root on the stage's "
                "workers; setup, estimates and G-tests are not in it",
            },
        )
        decision = ul.decide_corpus(split, excluded, unconverged["gate_p"])
        fields.set("decision", decision)
        return decision["row"]
    finally:
        report["pool_close"] = pool.close()


def stage_labelcheck(report, args, fields: Fields, manifest) -> str:
    """Stage 0 (§4.3, debug only): each labelcheck root collected twice, with and without the
    labelling look-ahead; the executed trajectory (commands, plate and palm at every kept step,
    the kept frames, the plate-hidden render, the committed aim, the executed steps and the
    outcome) must be identical."""
    r_plate = carried_r_plate(args, report)
    config = {"r_plate": r_plate["path"], "r_plate_sha256": r_plate["sha256"]}
    pool, co = sim_setup(report, args, manifest, config, int(args.workers or ul.SIM_WORKERS))
    try:
        seeds = co.seeds("labelcheck")
        by = {int(s): "train" for s in seeds}
        runs = {}
        for label in (True, False):
            tasks = attempt_tasks("collect-ul", seeds, co, corpus_task(by, label=label))
            runs[label] = run_arm(pool, tasks, f"labelcheck label={label}")
        pairs = [compare_roots(a, b) for a, b in zip(runs[True], runs[False], strict=True)]
        fields.set("pairs", pairs)
        same = all(p["identical"] for p in pairs)
        row = "LABELS-UNCHANGED" if same else "LABELS-CHANGED"
        fields.set("decision", {"row": row, "roots": len(pairs), "clause_fires": False})
        return row
    finally:
        report["pool_close"] = pool.close()


def compare_roots(with_label: dict, without: dict) -> dict:
    """Whether two collections of one root executed the same trajectory (bit-exact arrays)."""
    if with_label["seed"] != without["seed"]:
        raise lp.GuardError("labelcheck pairs the same roots")
    a, b = with_label.get("corpus") or {}, without.get("corpus") or {}
    out = {"seed": int(with_label["seed"]), "labelled": bool(a.get("labelled"))}
    keys = ("frames", "commands", "plate", "palm", "hidden_r", "target", "state405", "p_hat405")
    for key in keys:
        if key in a and key in b:
            out[key] = bool(np.array_equal(np.asarray(a[key]), np.asarray(b[key])))
        else:
            out[key] = bool(key not in a and key not in b)
    for key in ("executed_steps", "termination_reason", "success", "blocked"):
        out[key] = with_label.get(key) == without.get(key)
    out["identical"] = bool(all(v for k, v in out.items() if k not in ("seed", "labelled")))
    if a.get("labelled"):
        out["label_aim"] = np.asarray(a["label_aim"]).tolist()
        out["label_converged"] = bool(a["label_converged"])
    return out


# ----- Stage O ------------------------------------------------------------------------------------
def open_corpus_checked(args) -> dict:
    """Every stage after C checks the sealed corpus manifest's sha256 (a debug run checks it when
    given); a debug corpus never feeds a real run and the reverse."""
    expected = args.corpus_sha256
    if not args.debug and not expected:
        raise lp.GuardError("G-split: the sealed corpus manifest's sha256 is required")
    if not args.corpus:
        raise lp.GuardError("G-split: the sealed corpus folder is required")
    manifest = ulo.open_corpus(Path(args.corpus), expected)
    if bool(manifest["provenance"].get("debug")) != bool(args.debug):
        raise lp.GuardError("G-split: a debug corpus never feeds a real run and the reverse")
    return manifest


def stage_featurise(report, args, fields: Fields, manifest) -> str:
    import torch

    rt.gpu_guard(report, min_free_gib=ul.GPU["min_free_gib"], require_lock=True)
    devices.configure_determinism("cuda", strict=True)
    torch.set_num_threads(6)
    corpus = open_corpus_checked(args)
    report["corpus_sha256"] = corpus["_sha256"]
    result = ulo.featurise(Path(args.corpus), corpus, Path(args.output) / "features",
                           device="cuda")  # fmt: skip
    fields.set("featurisation", result)
    fields.set("anchor", result["anchor"])
    return "FEATURISED"


def artifact_chain(args, report: dict) -> dict:
    """G-split: the sealed corpus, its featurisation (FEATURISED; every file of the splits this
    stage reads hashed against the featurise report) and Stage O's fits (O-PASS; any O row in a
    debug run), each on the same corpus manifest and of this task."""
    corpus = open_corpus_checked(args)
    chain = {"corpus_sha256": corpus["_sha256"], "split": corpus["split"]}
    if getattr(args, "features", None):
        feat = completed(Path(args.features).parent / "report.json", ("FEATURISED",), args.debug)
        if feat.get("protocol") != ul.PROTOCOL:
            raise lp.GuardError("G-split: the featurise report is not TASK-082's")
        check_same_corpus(feat, chain, "featurise")
        files = feat["fields"]["featurisation"]["files_sha256"]
        chain["features_files_sha256"] = files
        report["feature_files_verified"] = ulo.verify_feature_files(
            Path(args.features), files, FEATURE_SPLITS.get(args.stage, ())
        )
    if getattr(args, "fits", None):
        ro = completed(Path(args.fits).parent / "report.json", O_ROWS_READ[bool(args.debug)],
                       args.debug)  # fmt: skip
        if ro.get("protocol") != ul.PROTOCOL:
            raise lp.GuardError("G-split: the readouts report is not TASK-082's")
        check_same_corpus(ro, chain, "readouts")
        chain["fits"] = {
            k: v | {"path": rebased(v["path"], args.fits)}
            if isinstance(v, dict) and "path" in v
            else v
            for k, v in ro["fields"]["fits"].items()
        }
        chain["tau_commit_cm"] = ro["fields"]["admission"].get("tau_commit_cm")
    report["artifact_chain"] = {
        "corpus_sha256": chain["corpus_sha256"],
        "fits": None
        if "fits" not in chain
        else {
            k: v.get("sha256")
            for k, v in chain["fits"].items()
            if isinstance(v, dict) and "sha256" in v
        },  # fmt: skip
        "fits_folder": str(Path(args.fits).resolve()) if "fits" in chain else None,
    }
    report["corpus_sha256"] = chain["corpus_sha256"]
    return chain


def load_moments(chain: dict):
    record = chain["fits"]["moments"]
    with np.load(record["path"]) as data:
        mean, std = data["mean"], data["std"]
    sha = ulo.moments_sha256(mean, std)
    if sha != record["sha256"]:
        raise lp.GuardError("G-split: the moments differ from Stage O's recorded sha256")
    return mean, std, sha


def load_r8(chain: dict):
    record = chain["fits"]["r8"]
    return ulo.load_readout(record["path"], record["sha256"])


def load_mean_latent(chain: dict) -> np.ndarray:
    record = chain["fits"]["mean_latent"]
    data = np.ascontiguousarray(np.load(record["path"]), np.float32)
    if ulo.array_sha256(data) != record["sha256"]:
        raise lp.GuardError("G-readout: the mean latent differs from Stage O's recorded sha256")
    return data


def load_fits(chain: dict) -> dict:
    """H-sysid's coefficients, H-rule-fit's kappa, H-sysid-krr and P-aim, each checked."""
    f = chain["fits"]
    return {
        "sysid_coef": ulo.load_json_fit(f["sysid"])["coef"],
        "kappa_fit": float(ulo.load_json_fit(f["rule_fit"])["kappa"]),
        "krr": ulo.load_json_fit(f["krr"]),
        "p_aim": ulo.load_json_fit(f["p_aim"]),
    }


def stage_readouts(report, args, fields: Fields, manifest) -> str:
    """Stage O's CPU part: O1, O3, O4 cross-fitted over train + val; the train-only fits."""
    chain = artifact_chain(args, report)
    r_plate = carried_r_plate(args, report)
    report["first_outcome_utc"] = hz.utc()
    result = ulo.readouts_core(
        Path(args.features),
        Path(args.output) / "fits",
        tau_commit_cm=tau_commit(args),
        r_plate=r_plate["readout"],
    )
    fields.set("admission", result["admission"] | {"tau_commit_cm": result["tau_commit_cm"]})
    fields.set("fits", result["fits"])
    fields.set("decision", result["decision"])
    report["readouts_detail"] = {"rows": result["rows"], "selections": result["selections"],
                                 "seconds": result["seconds"]}  # fmt: skip
    report["corpus_sha256"] = chain["corpus_sha256"]
    return result["decision"]["row"]


# ----- Stage T ------------------------------------------------------------------------------------
def parse_job(job: str, debug: bool) -> dict:
    arm, _, seed = job.partition("-")
    if arm not in ul.ARMS_TRAINED or not seed.isdigit():
        raise lp.GuardError(f"unknown Stage T job {job!r} (W-<seed> or N-<seed>)")
    return {"arm": arm, "seed": ul.check_model_seed(int(seed), debug=debug)}


def job_budget(args) -> dict:
    """The carried budget (R20.5: U = 95 000, 20 selections); debug: a few hundred updates."""
    if args.debug:
        return dict(ul.DEBUG_TRAIN) | {"debug_standin": True}
    return {"updates": ul.CARRIED_PLAN["updates"], "select_every": ul.CARRIED_PLAN["select_every"]}


def train_job_core(ctx: dict, spec: dict, budget: dict, *, output: Path, corpus_sha: str,
                   cap: float, probe: bool = False) -> dict:  # fmt: skip
    """One Stage T job: train, select, save."""
    metadata = ult.model_metadata(arm=spec["arm"], seed=spec["seed"], corpus_sha256=corpus_sha,
                                  moments_sha256=ctx["moments_sha256"])  # fmt: skip
    model, record = ult.train_model(
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
        "checkpoint_sha256": ult.sha256_file(path),
        "state_sha256": ult.state_sha256(model),
        "metadata": metadata,
    }
    del model
    return record


def stage_train(report, args, fields: Fields, manifest) -> str:
    import torch

    rt.gpu_guard(report, min_free_gib=ul.GPU["min_free_gib"], require_lock=True)
    report["determinism"] = devices.configure_determinism("cuda", strict=True)
    torch.set_num_threads(6)
    spec = parse_job(args.job, args.debug)
    budget = job_budget(args)
    fields.set("job", spec | budget)
    chain = artifact_chain(args, report)
    mean, std, moments_sha = load_moments(chain)
    ctx = {
        "train": ult.RootStore.open(Path(args.features), "train"),  # in RAM: about 9.6 GB
        "val": ult.RootStore.open(Path(args.features), "val", mmap=True),
        "mean": mean,
        "std": std,
        "moments_sha256": moments_sha,
    }
    record = train_job_core(
        ctx,
        spec,
        budget,
        output=Path(args.output),
        corpus_sha=chain["corpus_sha256"],
        cap=ul.CAPS_SECONDS["T_job"],
    )
    fields.set("record", record)
    fields.set("checkpoint", {"path": record["checkpoint"], "sha256": record["checkpoint_sha256"]})
    return "T-JOB-DONE"


def load_jobs(args, chain: dict) -> dict:
    """Every Stage T job of this task (T-JOB-DONE; checkpoint found beside its report), each on
    this corpus with Stage O's moments, W and N of every model seed."""
    jobs = {}
    for path in args.models:
        report = completed(path, ("T-JOB-DONE",), args.debug)
        if report.get("protocol") != ul.PROTOCOL:
            raise lp.GuardError(f"G-split: {path} is not a TASK-082 job")
        record = dict(report["fields"]["record"])
        record["checkpoint"] = rebased(record["checkpoint"], Path(path).parent)
        meta = record["metadata"]
        if meta.get("corpus_manifest_sha256") != chain["corpus_sha256"]:
            raise lp.GuardError(f"G-split: {record['arm']}-{record['seed']} used another corpus")
        if meta.get("normalisation_sha256") != chain["fits"]["moments"]["sha256"]:
            raise lp.GuardError(f"G-split: {record['arm']}-{record['seed']} used other moments")
        if meta.get("sampler_salt") != ul.SALTS["sampler"] or meta.get("protocol") != ul.PROTOCOL:
            raise lp.GuardError(f"G-split: {record['arm']}-{record['seed']} is not this recipe's")
        if hz.sha256_file(record["checkpoint"]) != record["checkpoint_sha256"]:
            raise lp.GuardError(f"G-hash: {record['arm']}-{record['seed']}'s checkpoint differs")
        jobs[(record["arm"], int(record["seed"]))] = record
    seeds = ul.DEBUG_MODEL_SEEDS if args.debug else ul.MODEL_SEEDS
    want = {(a, s) for a in ul.ARMS_TRAINED for s in seeds}
    if set(jobs) != want:
        raise lp.GuardError(f"G-plan: Stage G needs W and N of every seed {seeds}")
    return jobs


def model_spec(job: dict) -> dict:
    return {"path": job["checkpoint"], "sha256": job["checkpoint_sha256"],
            "seed": int(job["seed"]), "metadata": job["metadata"]}  # fmt: skip


def load_model(job: dict):
    return ult.load_model(job["checkpoint"], seed=int(job["seed"]), metadata=job["metadata"],
                          expected_sha256=job["checkpoint_sha256"])  # fmt: skip


# ----- Stage G ------------------------------------------------------------------------------------
def chunks_for(pool, data: dict, what: str):
    from embodied_jepa import lewm_pr_v2_offline as pof

    started = time.monotonic()
    got = pool.map(pof.chunk_tasks(data), MAP_CAP_SECONDS, what)
    chunks, feasible = pof.collect_chunks(got, data["n"])
    return chunks, feasible, time.monotonic() - started


def g_fit_core(models: dict, fit: dict, chunks, mean_latent, split: dict, folder: Path, *,
               crossfit: bool = True, probe: bool = False) -> dict:  # fmt: skip
    """Stage G's fit phase (§5): per model seed, W's stand-in, N's zero-command and W's
    mean-latent roll-outs to r from every fit root's 405 latent, and R-S, R-N, R-L fitted on them
    (train + val only); the reported cross-fit. ``models[seed]`` -> {"W", "N"} loaded models.
    Returns the record, the fitted readouts and the fit-root predictions (the learning curve)."""
    record = {"fit_roots": int(fit["n"]), "seeds": {}, "probe": bool(probe)}
    fitted, preds = {}, {}
    for s, m in models.items():
        t0 = time.monotonic()
        pred = ulo.rollouts(m["W"], m["N"], fit["start"], chunks, mean_latent)
        t1 = time.monotonic()
        readouts = ulo.fit_readouts(pred, fit["plate_r"], fit["seeds"], split)
        t2 = time.monotonic()
        saved = ulo.save_readouts(folder, s, readouts)
        item = {"readouts": saved, "rollout_seconds": t1 - t0, "fit_seconds": t2 - t1}
        if crossfit:
            item["crossfit"] = ulo.crossfit(pred, fit["plate_r"], fit["seeds"], fit["split"],
                                            split)  # fmt: skip
            item["crossfit_seconds"] = time.monotonic() - t2
        record["seeds"][str(s)] = item
        fitted[s], preds[s] = readouts, pred
        log(f"Stage G fits, seed {s}: {time.monotonic() - t0:.0f} s")
    return {"record": record, "fitted": fitted, "preds": preds}


def g_read_core(models: dict, fitted: dict, gate: dict, chunks, mean_latent, r8, *,
                probe: bool = False) -> dict:  # fmt: skip
    """Stage G's reading on gate-P: e_S, e_N, e_L per seed (R1-R3), R0 (R8 on the encoded frame at
    r) and the predictions (the learning curve, 1.5 tau)."""
    ceiling = ulo.errors_cm(r8.predict(gate["enc_r"]), gate["plate_r"])
    readings = {"R0_encoded_ceiling": ul.median_ci(ceiling), "seeds": {}, "probe": bool(probe)}
    preds = {}
    for s, m in models.items():
        t0 = time.monotonic()
        pred = ulo.rollouts(m["W"], m["N"], gate["start"], chunks, mean_latent)
        e = ulo.readout_errors(fitted[s], pred, gate["plate_r"])
        stats = ulo.readout_statistics(e)
        stats["R8_on_S_reported"] = ul.median_ci(ulo.errors_cm(r8.predict(pred["S"]),
                                                               gate["plate_r"]))  # fmt: skip
        stats["e_S_minus_ceiling"] = ul.median_difference_ci(e["e_S"], ceiling)
        stats["seconds"] = time.monotonic() - t0
        readings["seeds"][str(s)] = stats
        preds[s] = {"pred": pred, "errors": e}
    return {"readings": readings, "preds": preds}


def g_dynamics_core(models: dict, store, scale, *, probe: bool = False) -> dict:
    """G1-G4 of every seed on gate-P (E60) with the carried G1 bars."""
    out = {}
    for s, m in models.items():
        t0 = time.monotonic()
        out[str(s)] = ulo.dynamics(m["W"], m["N"], store, scale) | {
            "seconds": time.monotonic() - t0, "probe": bool(probe)}  # fmt: skip
    return out


def worker_models(jobs: dict, primary: int, seeds) -> dict:
    models = {"W": model_spec(jobs[("W", primary)]), "N": model_spec(jobs[("N", primary)])}
    for arm in ul.spread_arms(primary, seeds):
        models[arm] = model_spec(jobs[("W", ul.spread_seed(arm))])
    return models


def worker_readouts(saved: dict, primary: int, seeds) -> dict:
    """The predicted-latent readouts the workers load, by name, each with its sha256."""
    out = {}
    for name in ("r_s", "r_n"):
        out[name] = saved[str(primary)]["readouts"][name]["path"]
        out[f"{name}_sha256"] = saved[str(primary)]["readouts"][name]["sha256"]
    for arm in ul.spread_arms(primary, seeds):
        s = ul.spread_seed(arm)
        out[f"r_s_{s}"] = saved[str(s)]["readouts"]["r_s"]["path"]
        out[f"r_s_{s}_sha256"] = saved[str(s)]["readouts"]["r_s"]["sha256"]
    return out


def stage_gates(report, args, fields: Fields, manifest) -> str:
    """§7.1 step 8: the primary seed from val; the roll-outs and fits of R-S, R-N, R-L on the
    train + val roots (and their reported cross-fit); ``first_outcome_utc``; then gate-P: G1-G4,
    R0-R3 on every seed, A1-A2 on the primary seed's offline aims, and every reported reading."""
    import gc

    import torch

    torch.set_num_threads(1)
    chain = artifact_chain(args, report)
    split = chain["split"]
    jobs = load_jobs(args, chain)
    seeds = tuple(ul.DEBUG_MODEL_SEEDS if args.debug else ul.MODEL_SEEDS)
    kept = {s: jobs[("W", s)]["kept_val_criterion"] for s in seeds}
    primary = ul.primary_seed(kept)
    fields.set("primary_seed", {
        "seed": primary, "kept_val_criterion": kept,
        "last_two_triggered": {f"{a}-{s}": bool(jobs[(a, s)]["last_two_triggered"])
                               for a, s in sorted(jobs)},
        "rule": "the W seed with the lowest val criterion at its kept checkpoint (ties to the "
        "lower seed), fixed before gate-P is opened",
    })  # fmt: skip
    tau = tau_commit(args)
    counts = tau_curve()
    mean, std, _sha = load_moments(chain)
    scale = np.maximum(std, ul.METRIC_FLOOR_STD)
    mean_latent = load_mean_latent(chain)
    r8 = load_r8(chain)
    fits = load_fits(chain)
    found = hz.check_evidence(Path(args.evidence))
    report["evidence"] = {"root": str(args.evidence), "sha256": found["sha256"]}
    p3 = hz.p3_checkpoint(Path(args.evidence))
    workers = int(args.workers or ul.WM_WORKERS)
    report["workers"] = workers
    fit = ulo.load_roots(Path(args.features), ul.FIT_SPLITS)
    ulo.check_fit_roots(fit["seeds"], split)
    t0 = time.monotonic()
    pool = Pool(workers, {"torch_threads": ul.WORKER_TORCH_THREADS, "p3_checkpoint": p3})
    try:
        fit_chunks, fit_feasible, chunk_seconds = chunks_for(pool, fit, "stand-in chunks (fit)")
    finally:
        report["pool_close_fit_chunks"] = pool.close()
    models = {s: {"W": load_model(jobs[("W", s)]), "N": load_model(jobs[("N", s)])}
              for s in seeds}  # fmt: skip
    folder = Path(args.output) / "fits"
    phase = g_fit_core(models, fit, fit_chunks, mean_latent, split, folder)
    phase["record"] |= {"standin_infeasible": int((~fit_feasible).sum()),
                        "standin_seconds": chunk_seconds,
                        "chunks_sha256": ulo.array_sha256(fit_chunks),
                        "seconds": time.monotonic() - t0}  # fmt: skip
    fields.set("fits", phase["record"])
    report["first_outcome_utc"] = hz.utc()  # gate-P is opened only after this line
    files = chain["features_files_sha256"]
    report["gate_p_files_verified"] = ulo.verify_feature_files(Path(args.features), files,
                                                               ("gate_p",))  # fmt: skip
    gate = ulo.load_roots(Path(args.features), ("gate_p",), full405=True)
    pool = Pool(workers, {"torch_threads": ul.WORKER_TORCH_THREADS, "p3_checkpoint": p3})
    try:
        gate_chunks, gate_feasible, _s = chunks_for(pool, gate, "stand-in chunks (gate-P)")
    finally:
        report["pool_close_gate_chunks"] = pool.close()
    t1 = time.monotonic()
    read = g_read_core(models, phase["fitted"], gate, gate_chunks, mean_latent, r8)
    read["readings"] |= {"standin_infeasible": int((~gate_feasible).sum()),
                         "seconds": time.monotonic() - t1}  # fmt: skip
    fields.set("readings", read["readings"])
    t2 = time.monotonic()
    store = ult.RootStore.open(Path(args.features), "gate_p", mmap=True)
    dyn = g_dynamics_core(models, store, scale)
    fields.set("dynamics", dyn | {"seconds": time.monotonic() - t2})
    curve = {str(s): ulo.learning_curve(phase["preds"][s], fit["plate_r"], fit["seeds"],
                                        read["preds"][s]["pred"], gate["plate_r"], split)
             for s in seeds}  # fmt: skip
    at_15 = {str(s): {k: float(np.mean(v <= 1.5 * tau)) for k, v in
                      read["preds"][s]["errors"].items()} for s in seeds}  # fmt: skip
    del models, phase["preds"], read["preds"]
    gc.collect()
    t3 = time.monotonic()
    config = {"torch_threads": ul.WORKER_TORCH_THREADS, "p3_checkpoint": p3,
              "mean_latent": chain["fits"]["mean_latent"]["path"],
              "mean_latent_sha256": chain["fits"]["mean_latent"]["sha256"],
              "models": worker_models(jobs, primary, seeds)}  # fmt: skip
    config |= worker_readouts(phase["record"]["seeds"], primary, seeds)
    arms_out, slopes, seconds = {}, None, {}
    pool = Pool(workers, config)
    try:
        for arm in (*ul.R_ARMS, *ul.OFFLINE_REPORTED_ARMS, *ul.spread_arms(primary, seeds)):
            ta = time.monotonic()
            tasks = ulo.offline_aim_tasks(arm, gate, gate["start"], tau, fits=fits, keep=True)
            got = pool.map(tasks, MAP_CAP_SECONDS, f"offline aims {arm}")
            arms_out[arm] = ulo.aim_summary(got, gate, counts, ul.K_RESETS)
            if arm == "W":
                slopes = ulo.echo_slopes(got, gate)
            seconds[arm] = time.monotonic() - ta
            log(f"offline aims {arm}: {arms_out[arm]['predicted_count_of_128']:.1f}/128 "
                f"({seconds[arm]:.0f} s)")  # fmt: skip
    finally:
        report["pool_close_aims"] = pool.close()
    w = arms_out["W"]["predicted_count_of_128"]
    fields.set("offline_aims", {
        "primary_seed": primary, "tau_commit_cm": tau, "tau_counts": counts,
        "tau_resets": ul.K_RESETS, "arms": arms_out,
        "w_minus": {a: w - arms_out[a]["predicted_count_of_128"] for a in
                    (*ul.TWINS, *ul.OFFLINE_REPORTED_ARMS, *ul.spread_arms(primary, seeds))},
        "unconverged_labels": int((~gate["label_converged"]).sum()),
        "seconds": seconds, "total_seconds": time.monotonic() - t3,
        "note": "predicted counts are predictions, never closed-loop counts; the reference is "
        "each gate-P root's logged ceiling label; p-hat is the collector's logged CPU reading",
    })  # fmt: skip
    fields.set("reported", {
        "learning_curve_on_gate_p": curve,
        "at_1_5_tau": at_15,
        "echo_slope_W": slopes,
        "copy_last_and_prior": "in dynamics (copy-last MSE per horizon) and Stage O (the "
        "constant prior)",
    })  # fmt: skip
    gates = ul.g_gates(
        dynamics={s: dyn[str(s)]["passes"] for s in seeds},
        r0=read["readings"]["R0_encoded_ceiling"],
        readouts={s: read["readings"]["seeds"][str(s)] for s in seeds},
        aims={a: arms_out[a]["predicted_count_of_128"] for a in ul.R_ARMS},
        tau_commit_cm=tau,
        seeds=seeds,
        debug=args.debug,
    )
    fields.set("gates", gates)
    decision = ul.decide_g(gates)
    fields.set("decision", decision)
    return decision["row"]


# ----- Stage 0's scale probe of Stage G (synthetic latents, real sizes) --------------------------
def synthetic_roots(n: int, offset: int, *, key: int) -> dict:
    """Synthetic fit or gate-P roots at the real latent width (no corpus, no rendered frame)."""
    rng = np.random.default_rng(np.random.SeedSequence([ul.SALTS["bootstrap"], 77, key]))
    seeds = np.arange(offset, offset + n, dtype=np.int64)
    plate = np.asarray([0.49, -0.09]) + rng.normal(0.0, 0.02, (n, 2))
    return {
        "n": n,
        "seeds": seeds,
        "split": np.full(n, "train"),
        "start": rng.standard_normal((n, ul.LATENT_DIM), dtype=np.float32),
        "enc_r": rng.standard_normal((n, ul.LATENT_DIM), dtype=np.float32),
        "plate_r": plate,
    }


def stage_gscale(report, args, fields: Fields, manifest) -> str:
    """The scale probe of Stage G (§11's cap): ``g_fit_core`` (roll-outs, R-S/R-N/R-L fits and
    the cross-fit on 1 750 roots per seed, three seeds), ``g_read_core`` on 250 gate-P roots and
    ``g_dynamics_core`` on a synthetic 250-root gate-P store, each through
    ``run_tools.scale_probe``, with synthetic latents and stand-in chunks and three models of the
    real architecture (debug checkpoints when given, else untrained). The offline aims, whose cost
    is per root and per worker, are scaled from the debug Stage G report. CPU; nothing is read."""
    import torch

    torch.set_num_threads(1)
    scratch = Path(args.scratch)
    if scratch.exists():
        raise FileExistsError(f"refusing to overwrite {scratch}")
    fit_n, gate_n = args.scale_fit_roots, args.scale_gate_roots
    try:
        seeds = ul.DEBUG_MODEL_SEEDS
        if args.models:
            jobs = {}
            for path in args.models:
                job = completed(path, ("T-JOB-DONE",), True)["fields"]["record"]
                job["checkpoint"] = rebased(job["checkpoint"], Path(path).parent)
                jobs[(job["arm"], int(job["seed"]))] = job
            models = {s: {"W": load_model(jobs[("W", s)]), "N": load_model(jobs[("N", s)])}
                      for s in seeds}  # fmt: skip
            source = "debug checkpoints"
        else:
            models = {}
            for s in seeds:
                meta = ult.model_metadata(arm="W", seed=s, corpus_sha256="synthetic",
                                          moments_sha256="synthetic")  # fmt: skip
                models[s] = {a: ult.build_model(seed=s, device="cpu", metadata=meta)
                             for a in ("W", "N")}  # fmt: skip
                for m in models[s].values():
                    m.fit_frozen_feature_normalization(
                        np.zeros(ul.LATENT_DIM),
                        np.ones(ul.LATENT_DIM),
                        training_episode_ids=("synthetic",),
                    )
                    m.eval()  # fmt: skip
            source = "untrained, real architecture"
        fit = synthetic_roots(fit_n, 0, key=1)
        gate = synthetic_roots(gate_n, fit_n, key=2)
        split = {"train": [int(s) for s in fit["seeds"]], "val": [],
                 "gate_p": [int(s) for s in gate["seeds"]]}  # fmt: skip
        rng = np.random.default_rng(np.random.SeedSequence([ul.SALTS["bootstrap"], 78]))
        fit_chunks = rng.uniform(-0.1, 0.1, (fit_n, ul.HORIZON, 14)).astype(np.float32)
        gate_chunks = rng.uniform(-0.1, 0.1, (gate_n, ul.HORIZON, 14)).astype(np.float32)
        mean_latent = fit["start"].mean(0)
        fields.set("synthetic", {"fit_roots": fit_n, "gate_roots": gate_n, "models": source})
        watch = report["_watch"]
        fit_probe = rt.scale_probe(
            g_fit_core,
            models,
            fit,
            fit_chunks,
            mean_latent,
            split,
            scratch / "fits",
            watch=watch,
            ceiling_gib=ul.MEMORY["ceiling_gib"],
            margin_gib=2.0,
        )
        fitted = fit_probe.pop("result")["fitted"]
        r8 = ulo.ridge(gate["enc_r"][:64], gate["plate_r"][:64], gate["seeds"][:64].astype(str))
        read_probe = rt.scale_probe(g_read_core, models, fitted, gate, gate_chunks, mean_latent,
                                    r8, watch=watch, ceiling_gib=ul.MEMORY["ceiling_gib"],
                                    margin_gib=2.0)  # fmt: skip
        read_probe.pop("result")
        store_dir = scratch / "store"
        ult.tr.write_store(store_dir, "gate_p",
                           rng.standard_normal((gate_n, ul.N_FRAMES, ul.LATENT_DIM),
                                               dtype=np.float32),
                           rng.uniform(-0.1, 0.1, (gate_n, ul.N_COMMANDS, 14)).astype(np.float32),
                           gate["seeds"])  # fmt: skip
        store = ult.RootStore.open(store_dir, "gate_p", mmap=True)
        dyn_probe = rt.scale_probe(g_dynamics_core, models, store, np.ones(ul.LATENT_DIM),
                                   watch=watch, ceiling_gib=ul.MEMORY["ceiling_gib"],
                                   margin_gib=2.0)  # fmt: skip
        dyn_probe.pop("result")
        fields.set("probe", {"fit": fit_probe, "read": read_probe, "dynamics": dyn_probe})
        aims = None
        if args.gates:
            gates = completed(args.gates, G_ROWS_READ[True], True)
            aims_record = gates["fields"]["offline_aims"]
            per_root = {a: s / max(len(aims_record["arms"][a]["aim_errors_cm"]), 1)
                        for a, s in aims_record["seconds"].items()}  # fmt: skip
            aims = {
                "per_root_wall_seconds_debug": per_root,
                "scaled_250_roots_s": float(sum(per_root.values()) * ul.SPLIT_SIZES["gate_p"]),
                "workers": int(gates.get("workers", ul.WM_WORKERS)),
                "note": "wall seconds per gate-P root per arm in the debug Stage G (on its "
                "workers), scaled linearly to 250 roots",
            }
        offline = 0.0 if aims is None else aims["scaled_250_roots_s"]
        worst = fit_probe["seconds"] + read_probe["seconds"] + dyn_probe["seconds"] + offline
        cap = ul.CAPS_SECONDS["G"]
        fields.set("caps", {
            "fit_seconds": fit_probe["seconds"], "read_seconds": read_probe["seconds"],
            "dynamics_seconds": dyn_probe["seconds"], "offline_aims": aims,
            "worst_case_s": worst, "cap_G_s": cap, "cap_over_worst": cap / worst,
            "peak_tree_gib": max(fit_probe["probe_peak_tree_gib"],
                                 read_probe["probe_peak_tree_gib"],
                                 dyn_probe["probe_peak_tree_gib"]),
            "note": "the stand-in chunks of the 2 000 roots and G-tests are not in it",
        })  # fmt: skip
    finally:
        if scratch.exists() and not args.keep_scratch:
            shutil.rmtree(scratch)
            report["synthetic_removed"] = True
    return "G-SCALE-PROBED"


# ----- Stages D and S -----------------------------------------------------------------------------
def stage_d_report(args, debug: bool) -> dict:
    """This task's D-PASS report (Stage S only after D-PASS, §7.1 step 10)."""
    report = json.loads(Path(args.stage_d).read_text())
    want = "D-PASS-DEBUG" if debug else "D-PASS"
    if (
        report.get("task") != ul.TASK
        or report.get("protocol") != ul.PROTOCOL
        or report.get("outcome") != want
        or bool(report.get("debug")) != bool(debug)
        or report.get("cohort") != "D"
    ):
        raise lp.GuardError(f"G-fresh: {args.stage_d} is not this task's {want} report")
    if not debug and report.get("frozen_sha256_at_run") != ul.frozen_sha256():
        raise lp.GuardError("G-fresh: Stage D ran under another frozen block")
    return {"path": str(Path(args.stage_d).resolve()), "sha256": hz.sha256_file(args.stage_d)}


def stage_closed(report, args, fields: Fields, manifest) -> str:
    chain = artifact_chain(args, report)
    gates = completed(args.gates, G_ROWS_READ[bool(args.debug)], args.debug)
    if gates.get("protocol") != ul.PROTOCOL:
        raise lp.GuardError("G-fresh: the gates report is not TASK-082's")
    check_same_corpus(gates, chain, "gates")
    report["gates_report"] = {"path": str(Path(args.gates).resolve()),
                              "sha256": hz.sha256_file(args.gates),
                              "outcome": gates["outcome"]}  # fmt: skip
    if args.cohort == "S" and not args.debug:
        report["stage_d"] = stage_d_report(args, False)
    elif args.cohort == "S" and args.stage_d:
        report["stage_d"] = stage_d_report(args, True)
    jobs = load_jobs(args, chain)
    seeds = tuple(ul.DEBUG_MODEL_SEEDS if args.debug else ul.MODEL_SEEDS)
    primary = int(gates["fields"]["primary_seed"]["seed"])
    report["primary_seed"] = primary
    report["cohort"] = args.cohort
    saved = gates["fields"]["fits"]["seeds"]
    folder = Path(args.gates).parent / "fits"
    readouts = {}
    for s in seeds:
        readouts[str(s)] = {"readouts": {}}
        for name in ("r_s", "r_n"):
            rec = saved[str(s)]["readouts"][name]
            path = rebased(rec["path"], folder)
            ulo.load_readout(path, rec["sha256"])
            readouts[str(s)]["readouts"][name] = {"path": path, "sha256": rec["sha256"]}
    r_plate = carried_r_plate(args, report)
    load_mean_latent(chain)
    load_r8(chain)
    fits = load_fits(chain)
    config = {
        "r_plate": r_plate["path"],
        "r_plate_sha256": r_plate["sha256"],
        "r8": chain["fits"]["r8"]["path"],
        "r8_sha256": chain["fits"]["r8"]["sha256"],
        "mean_latent": chain["fits"]["mean_latent"]["path"],
        "mean_latent_sha256": chain["fits"]["mean_latent"]["sha256"],
        "models": worker_models(jobs, primary, seeds),
    } | worker_readouts(readouts, primary, seeds)
    pool, co = sim_setup(report, args, manifest, config, int(args.workers or ul.WM_WORKERS))
    try:
        cohort = ul.check_seeds(args.cohort, co.seeds(args.cohort), debug=args.debug)
        report["first_outcome_utc"] = hz.utc()
        common = {"tau_commit_cm": tau_commit(args), "a_lo": ul.A_LO}
        spread = ul.spread_arms(primary, seeds)
        return closed_core(pool, co, cohort, args.cohort, common, fits, fields, spread=spread,
                           debug=args.debug)  # fmt: skip
    finally:
        report["pool_close"] = pool.close()


def closed_core(pool, co, seeds, role: str, common: dict, fits: dict, fields: Fields, *,
                spread=(), debug: bool) -> str:  # fmt: skip
    """Stage D or S on ``seeds`` (the stage's own code; tests drive it with a fake pool). A reset
    refused before 405 fails for every arm (R17.20); L-shuf's foreign frame is the next reset in
    cohort order, cyclically, whose W attempt reached 405."""
    arms = ul.D_ARMS if role == "D" else (*ul.S_GATING_ARMS, *ul.S_REPORTED_ARMS, *spread)
    records = {"W": run_arm(pool, attempt_tasks("W", seeds, co, **common), f"{role} W")}
    reached = [lm.reached_405(r) for r in records["W"]]
    foreign = {}
    for i, s in enumerate(seeds):
        j = lm.foreign_reached(i, reached)
        if j is None and reached[i]:
            raise lp.GuardError("G-frames: no other reset reached 405 for L-shuf's frame")
        foreign[s] = None if j is None else records["W"][j]["frame405"]
    extras = {
        "H-sysid": {"sysid_coef": fits["sysid_coef"]},
        "H-sysid-krr": {"krr": fits["krr"]},
        "P-aim": {"p_aim": fits["p_aim"]},
        "H-rule-fit": {"kappa_fit": fits["kappa_fit"]},
        "H-read": {"readout": "r8", "grid": ul.TOKEN_GRID, "read_step": ul.READ_STEP},
    }
    for arm in arms:
        if arm == "W":
            continue
        per = None
        if arm == "L-shuf":

            def per(s):
                return {"foreign_frame": foreign[s]}

        extra = dict(common) | extras.get(arm, {})
        records[arm] = run_arm(pool, attempt_tasks(arm, seeds, co, per, **extra), f"{role} {arm}")
    refused = {
        a: [int(r["seed"]) for r in recs if not lm.reached_405(r)] for a, recs in records.items()
    }
    if role == "S":
        first = list(seeds[: ul.DETERMINISM_RESETS])
        again = run_arm(pool, attempt_tasks("W", first, co, **common), "S W determinism")
        determinism = ul.determinism_check(records["W"][: len(first)], again)
        fields.set("determinism", determinism)
        if not determinism["ok"]:
            raise lp.GuardError(f"G-determinism: W's re-run differs: {determinism['resets']}")
    fields.set(
        "arms",
        {a: arm_summary(r) | {"attempts": lean(r)} for a, r in records.items()}
        | {"refused_before_405": {"by_arm": refused, "W": len(refused["W"])}},
    )
    outcomes = {a: successes(r) for a, r in records.items()}
    fields.set("reported", reported(records, outcomes, n=len(seeds), spread=spread))
    if role == "D":
        decision = ul.decide_d({a: sum(v) for a, v in outcomes.items()})
    else:
        decision = ul.decide_s(outcomes, debug=debug)
    fields.set("decision", decision)
    return decision["row"]


def reported(records: dict, outcomes: dict, *, n: int, spread=()) -> dict:
    """§9.5, reported only: counts with exact intervals, every paired difference from W, the seed
    spread (each spread arm against W and C), aim errors against H-final(commit)'s aim on the
    same reset, the tau-curve prediction of those errors, final distances, clip-binding and
    affine_local's diagnostics."""
    out: dict = {
        "counts": {a: {"count": int(sum(v)), "n": len(v), "ci95": ul.exact_interval(sum(v), len(v))}
                   for a, v in outcomes.items()},
        "w_minus_arm": {a: ul.paired_interval(outcomes["W"], v) for a, v in outcomes.items()
                        if a != "W"},
    }  # fmt: skip
    if all(a in outcomes for a in ul.COMPARATORS) and spread:
        c_arm = ul.comparator(outcomes)
        out["seed_spread"] = {a: {
            "minus_W": ul.paired_interval(outcomes[a], outcomes["W"]),
            "minus_C": ul.paired_interval(outcomes[a], outcomes[c_arm]),
            "comparator": c_arm} for a in spread if a in outcomes}  # fmt: skip
    ref = {}
    if "H-final" in records:
        ref = {r["seed"]: np.asarray(r["commit"]["target"], np.float64)
               for r in records["H-final"] if r.get("commit")}  # fmt: skip
    per_arm = {}
    for arm, recs in records.items():
        commits = [r for r in recs if r.get("commit")]
        entry: dict = {}
        if ref and arm not in ul.PRIVILEGED_ARMS:
            err = [100.0 * float(np.linalg.norm(np.asarray(r["commit"]["target"]) - ref[r["seed"]]))
                   for r in commits if r["seed"] in ref]  # fmt: skip
            entry["aim_error_vs_h_final_cm"] = _stats(err)
            if err:
                entry["tau_curve_predicted_count"] = ul.predicted_count(err, tau_curve(),
                                                                        cohort=n)  # fmt: skip
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
        logs = [(r.get("decisions") or [{}])[0].get("world_model", {}) for r in commits]
        logs = [g for g in logs if g.get("solver") == "affine_local"]
        if logs:
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
                "rank_below_3": int(sum(g.get("design_rank", 3) < 3 for g in logs)),
                "residual_cm": _stats([g.get("residual_cm") for g in logs]),
                "eigenvalue_real_min": _stats([min(e) for e in eig]),
                "eigenvalue_real_max": _stats([max(e) for e in eig]),
                "complex_pairs": int(sum(bool(g.get("eigenvalues_complex")) for g in logs)),
            }
        per_arm[arm] = entry
    out["per_arm"] = per_arm
    return out


# ----- §10's power simulation --------------------------------------------------------------------
def stage_simulate(report, args, fields: Fields, manifest) -> str:
    started = time.monotonic()
    power = ul.power_tables(trials=int(args.trials or ul.POWER_TRIALS))
    power["seconds"] = time.monotonic() - started
    fields.set("power", power)
    return "SIMULATED"


def stage_tests(report, args, fields: Fields, manifest) -> str:
    fields.set("tests", run_full_tests())
    return "TESTS-PASS"


# ----- the run ------------------------------------------------------------------------------------
RUNNERS = {
    "tests": stage_tests,
    "k0": stage_k0,
    "corpus": stage_corpus,
    "labelcheck": stage_labelcheck,
    "featurise": stage_featurise,
    "readouts": stage_readouts,
    "train": stage_train,
    "gates": stage_gates,
    "gscale": stage_gscale,
    "closed": stage_closed,
    "simulate": stage_simulate,
}


def stage_cap(args) -> float:
    """The stage's wall cap (§11); Stage D and Stage S have their own."""
    if args.stage == "closed":
        return ul.CAPS_SECONDS[args.cohort]
    return ul.CAPS_SECONDS.get(STAGE_CAPS.get(args.stage, ""), 4 * 3600.0)


def exclude_quiet_wait(clock, report: dict) -> None:
    """R20.18: a stage's wall cap counts from the end of G-quiet's wait, which has its own cap
    (4 h); the excluded seconds are recorded. (``simulate-1`` was voided by its 3 600 s cap after
    a 73-minute wait for a quiet machine, with its computation complete.)"""
    waited = float((report.get("quiet_machine") or {}).get("waited_seconds", 0.0))
    clock.start += waited
    report["stage_cap_excludes_quiet_wait_seconds"] = waited


def run(args) -> dict:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    report = {
        "protocol": ul.PROTOCOL,
        "task": ul.TASK,
        "stage": args.stage,
        "cohort": args.cohort,
        "debug": bool(args.debug),
        "outcome": None,
        "stages": {},  # the harness records G-repro here
        "argv": sys.argv[1:],
        "paths": {"output": str(output.resolve())},
        "started_utc": hz.utc(),
        "salts": ul.SALTS,
        "seed_ranges": ul.DEBUG_RANGES if args.debug else ul.SEED_RANGES,
        "law": ul.LAW,
    }
    fields = Fields(report, args.stage)
    clock = hz.Clock(stage_cap(args))
    ceiling = ul.MEMORY["train_ceiling_gib"] if args.stage == "train" else ul.MEMORY["ceiling_gib"]
    watch = rt.MemoryWatch(
        ceiling * GIB,
        ul.MEMORY["sample_seconds"],
        measure=ul.MEMORY["measure"],
        disk_path=ROOT,
        min_disk_free_bytes=int(ul.DISK_MIN_GIB * GIB),
    )
    guards = rt.install_guards(watch=watch, log=log)
    report["_watch"] = watch
    try:
        manifest = preflight(report, args)
        exclude_quiet_wait(clock, report)
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
            except Exception as error:  # noqa: BLE001 - a reading never changes the outcome
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
    parser.add_argument("--tests-record", help="GPU jobs: a `tests` stage report at HEAD")
    parser.add_argument("--r-plate-fits", help="TASK-077 Stage O's fits folder (R-plate)")
    parser.add_argument("--corpus", help="the sealed corpus folder")
    parser.add_argument("--corpus-sha256", help="the sealed corpus manifest's sha256")
    parser.add_argument("--features", help="Stage O's featurisation folder")
    parser.add_argument("--fits", help="Stage O's fits folder (inside the readouts output)")
    parser.add_argument("--job", help="Stage T: W-<seed> or N-<seed>")
    parser.add_argument("--models", nargs="*", default=[], help="Stage T job reports")
    parser.add_argument("--gates", help="Stage G's report")
    parser.add_argument("--stage-d", help="this task's Stage D D-PASS report (closed --cohort S)")
    parser.add_argument("--cohort", choices=("D", "S"))
    parser.add_argument("--trials", type=int, default=None, help="simulate: power trials")
    parser.add_argument("--scratch", help="gscale: the synthetic store's folder (removed after)")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--scale-fit-roots", type=int, default=sum(
        ul.SPLIT_SIZES[s] for s in ul.FIT_SPLITS))  # fmt: skip
    parser.add_argument("--scale-gate-roots", type=int, default=ul.SPLIT_SIZES["gate_p"])
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers is not None and (not args.debug or not 1 <= args.workers <= 6):
        parser.error("--workers is for debug runs only, from 1 to 6")
    if args.debug_skip_tests and not args.debug:
        parser.error("--debug-skip-tests is for debug runs only")
    if args.stage in SIM_STAGES and not args.evidence:
        parser.error(f"{args.stage} needs --evidence (G-repro)")
    if args.stage in GPU_STAGES and not args.tests_record:
        parser.error(f"{args.stage} needs --tests-record (G-tests for GPU jobs)")
    if args.stage in R_PLATE_STAGES and not args.r_plate_fits:
        parser.error(f"{args.stage} needs --r-plate-fits (TASK-077's R-plate)")
    if args.stage in ("featurise", "readouts", "train", "gates", "closed"):
        if not args.corpus:
            parser.error(f"{args.stage} needs --corpus (G-split: the sealed manifest is checked)")
        if not args.debug and not args.corpus_sha256:
            parser.error(f"{args.stage} needs --corpus-sha256")
    if args.stage in ("readouts", "train", "gates", "closed") and not args.features:
        parser.error(f"{args.stage} needs --features")
    if args.stage in ("train", "gates", "closed") and not args.fits:
        parser.error(f"{args.stage} needs --fits")
    if args.stage in ("gates", "closed") and not args.models:
        parser.error(f"{args.stage} needs --models (the six Stage T job reports)")
    if args.stage == "closed":
        if not args.cohort:
            parser.error("closed needs --cohort")
        if not args.gates:
            parser.error("closed needs --gates (Stage G's report)")
        if args.cohort == "S" and not args.debug and not args.stage_d:
            parser.error("closed --cohort S needs --stage-d (this task's D-PASS)")
    elif args.cohort:
        parser.error("--cohort is for closed only")
    if args.stage == "gscale" and not args.scratch:
        parser.error("gscale needs --scratch")
    if args.stage == "train" and not args.job:
        parser.error("train needs --job")
    (ROOT / "outputs").mkdir(exist_ok=True)
    if args.log:
        open_log(Path(args.log))
    report = run(args)
    return 0 if report.get("outcome") not in (None, "V") else 1


if __name__ == "__main__":
    sys.exit(main())
