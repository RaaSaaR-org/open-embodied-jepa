"""TASK-072 design (``apple_first_policy_v2_linux``): TASK-071 replicated on the Linux PC.

Protocol ``docs/experiments/apple_first_policy_v2_linux.md``. Everything is TASK-071's
``first_policy_v2`` design, imported unchanged (same seeds, arms, corpus plan, readout, DAgger
schedule, caps, rows and M1 gate on the development cohort D2), except what this module changes,
each a declared deviation:

- training on **CUDA** with the project's deterministic setup in **strict** mode
  (``embodied_jepa.devices``), instead of MPS;
- the Linux PC (x86-64, Ubuntu 24.04, NVIDIA EGL rendering) instead of the Mac;
- **16** CPU simulation workers instead of 8, and the render check at 16;
- the random-init DINOv2 floor's weight digest re-pinned from a Linux measurement (the manifest's
  ``encoder_digests``); the pretrained digest is unchanged.

It adds the declared reading of the run against TASK-071 run-1 (``read_replication``). Nothing
in ``first_policy_v2`` or its runner is modified. **Learned Apple->Plate on the frozen benchmark
is still 0 successes.**
"""

from __future__ import annotations

import copy
import os
import platform

from embodied_jepa import first_policy as fp
from embodied_jepa import first_policy_v2 as fp2

PROTOCOL = "apple_first_policy_v2_linux"
TASK = "TASK-072"
GuardError = fp2.GuardError

TRAIN_DEVICE = "cuda"
STRICT_DETERMINISM = True  # owner (T72 Stage B note): strict, and no silent warn-only fallback
SIM_WORKERS = 16
FEATURE_DEVICE = fp2.FEATURE_DEVICE  # cpu, unchanged
ROLLOUT_DEVICE = fp2.ROLLOUT_DEVICE  # cpu, unchanged
PLATFORM = {"system": "Linux", "machine": "x86_64", "mujoco_gl": "egl"}

RENDER_CHECK = (
    "before the pre-run GO, the pre-run reviewer confirms from scripts/check_first_policy_v2_"
    "linux_render.py at the gated worker count (16), on the Linux PC, that the post-look "
    "re-render is bit-identical (verdict IDENTICAL); otherwise the run does not start and the "
    "issue goes to the owner"
)

# ----- the comparison target: TASK-071 run-1, as recorded -----------------------------------------
# benchmarks/manifests/apple-first-policy-v2-results.json (merged in #97); report sha256 below.
RUN1 = {
    "results_manifest": "benchmarks/manifests/apple-first-policy-v2-results.json",
    "report_sha256": "77aae2077f5e1070f9cdc63b9f228f32515713e869b999b5303a5134c101d90b",
    "revision": "9e23cedb1fc8a5012cd1fdf4661aa4537b9189fa",
    "machine": "Mac (macOS 26.5.1 arm64), training on MPS, 8 CPU simulation workers",
    "outcome": "M1-PASS",
    "carried": "P-3",
    "m1_success_of_16": {
        "P-0": 4,
        "P-1": 9,
        "P-2": 15,
        "P-3": 16,
        "C-3": 3,
        "R-3": 16,
        "A4-look": 16,
        "D-oracle-perc": 16,
        "B-oracle": 16,
        "B-replay": 9,
        "B-hold": 0,
        "B-random": 0,
    },
    "corpus_counted_success_of_200": 129,
}

# ----- the declared reading (first matching row) -------------------------------------------------
REPLICATION_P3_MIN = 14  # of 16 counted successes for P-3
REPLICATION_ROWS = (
    "V",
    "REP-EARLY-STOP",
    "REPLICATED",
    "REPLICATED-M1-ONLY",
    "NOT-REPLICATED",
)
REPLICATION_RULE = {
    "V": "the run is void (protocol §14); nothing is read; one from-scratch repeat, and a second "
    "V closes TASK-072's replication as INCONCLUSIVE",
    "REP-EARLY-STOP": "the run ends at CAL-ESCALATE, S0-APPLE-FAIL or S0-PLATE-FAIL; run-1 passed "
    "those stages, so the pipeline did not replicate before any policy was evaluated; the owner "
    "decides",
    "REPLICATED": "outcome M1-PASS and P-3 reaches at least 14/16 counted successes on D2",
    "REPLICATED-M1-ONLY": "outcome M1-PASS (some P-k >= 1/16) but P-3 below 14/16: the "
    "existence result replicates, the rate does not",
    "NOT-REPLICATED": "any other M1 row (M1-MOTOR-F-PARTIAL, M1-MOTOR-F-NONE, M1-PERCEPTION)",
}
DESCRIPTIVE_COMPARISONS = (
    "every M1 arm's counted successes against run-1's, as a signed difference (one run each; no "
    "test)",
    "P-3's per-reset success set against run-1's on the same 16 D2 seeds (agreement count)",
    "whether R-3 is within 2 of P-3, and whether C-3 is at least 8 below P-3 (run-1's two vision "
    "findings), stated as observed, not as gates",
    "the corpus's counted successes against 129/200, the C0 table, the bars, S0-P's errors, the "
    "DAgger rollout counts and every training's selected update",
)


def read_replication(report: dict) -> dict:
    """The declared reading of a finished run's ``report.json`` against run-1.

    It never raises: the runner calls it while writing the report, whatever the run's end."""
    try:
        return _read_replication(report)
    except Exception as error:  # noqa: BLE001 - the report must still be written
        return {"row": None, "error": f"{type(error).__name__}: {error}"}


def _read_replication(report: dict) -> dict:
    if report.get("smoke"):
        return {"row": None, "note": "smoke run: not read"}
    outcome = report.get("outcome")
    if outcome in (None, "V"):
        return {"row": "V"}
    if outcome in fp2.DECLARED_EARLY_STOPS:
        return {"row": "REP-EARLY-STOP", "outcome": outcome}
    counts = report["stages"]["M1"]["counts"]
    p3 = int(counts["P-3"]["success"])
    reading = {
        "outcome": outcome,
        "p3_success_of_16": p3,
        "p3_min": REPLICATION_P3_MIN,
        "difference_vs_run1": {
            arm: int(counts[arm]["success"]) - value
            for arm, value in RUN1["m1_success_of_16"].items()
            if arm in counts
        },
    }
    if outcome == "M1-PASS":
        row = "REPLICATED" if p3 >= REPLICATION_P3_MIN else "REPLICATED-M1-ONLY"
    else:
        row = "NOT-REPLICATED"
    return {"row": row} | reading


# ----- platform, headless rendering, determinism --------------------------------------------------
def configure_headless() -> None:
    """MuJoCo must render through EGL over ssh; set before any MuJoCo import (workers inherit)."""
    os.environ.setdefault("MUJOCO_GL", PLATFORM["mujoco_gl"])


def check_platform() -> dict:
    """G-platform: the Linux PC's platform and EGL rendering, as preregistered."""
    found = {
        "system": platform.system(),
        "machine": platform.machine(),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
    }
    if found != PLATFORM:
        raise GuardError(f"G-platform: {found} is not the preregistered {PLATFORM}")
    return found


def check_determinism(report: dict) -> dict:
    """G-device at the end: the process is still in strict deterministic mode."""
    from embodied_jepa import devices

    state = devices.determinism_state()
    report["determinism_at_end"] = state
    if not state["deterministic_algorithms"] or (
        STRICT_DETERMINISM and state["deterministic_algorithms_warn_only"]
    ):
        raise GuardError("G-device: the process left strict deterministic mode during the run")
    return state


def frozen_block() -> dict:
    """TASK-071's frozen block with this protocol's declared changes; the manifest must equal it."""
    block = copy.deepcopy(fp2.frozen_block())
    block["protocol"] = PROTOCOL
    block["task"] = TASK
    block["carried_from"] = fp2.PROTOCOL
    block["training"]["device"] = TRAIN_DEVICE
    block["devices"] = {
        "features": FEATURE_DEVICE,
        "training": TRAIN_DEVICE,
        "rollouts": ROLLOUT_DEVICE,
    }
    block["sim_workers"] = SIM_WORKERS
    block["render_check"] = RENDER_CHECK
    block["platform"] = PLATFORM
    block["determinism"] = {
        "module": "embodied_jepa.devices",
        "strict": STRICT_DETERMINISM,
        "fallback_to_warn_only": False,
    }
    block["replication"] = {
        "target": RUN1,
        "p3_min_of_16": REPLICATION_P3_MIN,
        "rows": REPLICATION_ROWS,
        "rule": REPLICATION_RULE,
        "descriptive": DESCRIPTIVE_COMPARISONS,
    }
    return fp._plain(block)


# Keys of the frozen block this protocol changes or adds; everything else equals TASK-071's.
CHANGED_KEYS = (
    "protocol",
    "task",
    "carried_from",
    "training",
    "devices",
    "sim_workers",
    "render_check",
    "platform",
    "determinism",
    "replication",
)
