"""Summarize a completed TASK-057 diagnostic run for the results document (read-only).

Reads ``<run>/report.json`` and the per-attempt files under ``<run>/attempts/``, and writes one
JSON file with every number ``docs/experiments/apple_policy_diagnostics_v1_results.md`` cites
that the runner's report does not already contain:

* SHA-256 of every per-attempt file and of ``report.json`` (amendment 2, rule 2);
* per arm and per attempt, from the per-step trace: palm-apple distance at step zero and its
  minimum, first contact, contact steps, first reach/grasp, maximum lift;
* for every attempt that latched ``grasp``: the per-step contact fraction and apple height
  after the latch, the controller's and the shadow expert's mean translation commands after
  the latch, and the shadow-expert phase at the latch;
* the named D2 traces and each D3 candidate's reach/contact counts.

It never re-runs anything and never touches a cohort.

    uv run --no-sync python scripts/summarize_policy_diagnostics.py \
        --run outputs/task057-diagnostics/run-1 \
        --output outputs/task057-diagnostics/run-1-summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ARMS = ("A0_proprio_only", "A1_random_encoder", "A2_bc_frozen_e0", "A3_bc_finetuned_e0")
CANDIDATES = ("all_translation", "dx", "dy", "dz", "droll", "dpitch", "dyaw", "grasp")
NAMED = {
    "reached_grasp": (("A2_bc_frozen_e0", 45100), ("A3_bc_finetuned_e0", 45006)),
    "failing_contrast": (("A2_bc_frozen_e0", 45000), ("A3_bc_finetuned_e0", 45000)),
}
REACHED = ("reach", "grasp", "transport", "place", "release")
GRASPED = ("grasp", "transport", "place", "release")


def _first(flags):
    return next((i for i, flag in enumerate(flags) if flag), None)


def attempt_summary(attempt) -> dict:
    trace = attempt.get("trace") or []
    distance = np.array([row["palm_apple_m"] for row in trace], float)
    after = [row["after"] for row in trace if "after" in row]
    contact = [bool(a["hand_contact"]) for a in after]
    height = np.array([a["object_height_m"] for a in after], float)
    stages = [a["stage"] for a in after]
    out = {
        "seed": attempt["seed"],
        "termination_reason": attempt["termination_reason"],
        "executed_steps": attempt["executed_steps"],
        "grasp": bool(attempt["grasp"]),
        "success": bool(attempt["success"]),
        "palm_apple_m_step0": float(distance[0]) if len(distance) else None,
        "palm_apple_m_min": float(distance.min()) if len(distance) else None,
        "first_contact_step": _first(contact),
        "contact_steps": int(sum(contact)),
        "first_reach_step": _first([s in REACHED for s in stages]),
        "first_grasp_step": _first([s in GRASPED for s in stages]),
        "max_lift_m": float(height.max() - height[0]) if len(height) else None,
    }
    latch = out["first_grasp_step"]
    if latch is not None:
        rows = trace[latch:]
        held = np.array([row["after"]["hand_contact"] for row in rows], bool)
        lift = np.array([row["after"]["object_height_m"] for row in rows], float)
        controller = np.array([row["controller"] for row in rows], float)
        expert = [row["shadow_expert"] for row in rows if row["shadow_expert"] is not None]
        out["after_grasp"] = {
            "latch_step": latch,
            "steps": len(rows),
            "contact_fraction": float(held.mean()),
            "first_contact_loss": int(np.argmax(~held)) if (~held).any() else None,
            "height_min_m": float(lift.min()),
            "height_max_m": float(lift.max()),
            "controller_mean_dx_dy_dz": controller[:, :3].mean(0).tolist(),
            "controller_median_abs_dx_dy_dz": np.median(np.abs(controller[:, :3]), 0).tolist(),
            "controller_mean_grasp": float(controller[:, 6].mean()),
            "shadow_expert_mean_dx_dy_dz": np.array(expert)[:, :3].mean(0).tolist()
            if expert
            else None,
            "shadow_expert_defined_steps": len(expert),
            "shadow_expert_phase_at_latch": trace[latch]["shadow_expert_phase"],
            "final_object_plate_distance_m": attempt["score"].get("object_plate_distance_m"),
        }
    return out


def summarize(run: Path) -> dict:
    attempts_dir = run / "attempts"
    files = sorted(attempts_dir.rglob("*.json"))
    hashes = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report = json.loads((run / "report.json").read_text())

    def load(stage):
        return {
            int(p.stem): json.loads(p.read_text())
            for p in sorted((attempts_dir / stage).glob("*.json"))
        }

    arms = {}
    for arm in ARMS:
        rows = [attempt_summary(a) for a in load(f"none-{arm}").values()]
        step0 = [r["palm_apple_m_step0"] for r in rows]
        minimum = [r["palm_apple_m_min"] for r in rows]
        arms[arm] = {
            "attempts": rows,
            "reach_resets": sum(r["first_reach_step"] is not None for r in rows),
            "contact_resets": sum(r["first_contact_step"] is not None for r in rows),
            "grasp_resets": sum(r["grasp"] for r in rows),
            "median_palm_apple_m_step0": float(np.median(step0)),
            "median_palm_apple_m_min": float(np.median(minimum)),
            "resets_never_closer_than_step0_by_1cm": sum(
                1 for a, b in zip(step0, minimum, strict=True) if a - b < 0.01
            ),
        }
    d3 = {}
    for candidate in CANDIDATES:
        rows = [attempt_summary(a) for a in load(f"D3-A2-{candidate}").values()]
        d3[candidate] = {
            "reach_resets": sum(r["first_reach_step"] is not None for r in rows),
            "contact_resets": sum(r["first_contact_step"] is not None for r in rows),
            "grasp_resets": sum(r["grasp"] for r in rows),
            "successes": sum(r["success"] for r in rows),
            "terminations": sorted({r["termination_reason"] for r in rows}),
        }
    named = {
        kind: [
            next(r for r in arms[arm]["attempts"] if r["seed"] == seed) | {"arm": arm}
            for arm, seed in pairs
        ]
        for kind, pairs in NAMED.items()
    }
    return {
        "run": str(run),
        "report_sha256": hashlib.sha256((run / "report.json").read_bytes()).hexdigest(),
        "report_source": report["source"],
        "report_outcome": report.get("outcome"),
        "attempt_files": len(files),
        "attempt_file_sha256": hashes,
        "arms_none": arms,
        "d3": d3,
        "named_d2_traces": named,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    summary = summarize(args.run)
    args.output.write_text(json.dumps(summary, indent=1, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("report_sha256", "attempt_files")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
