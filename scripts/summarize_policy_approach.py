"""TASK-057: approach, carry and D3-drop statistics from the run-1 per-step traces.

Read-only; prints JSON. Supplies the trace-derived numbers in
``docs/experiments/apple_policy_diagnostics_v1_results.md`` §4-§5 that
``summarize_policy_diagnostics.py`` does not (per-reset command means, the representative trace,
like-for-like |.| statistics after the grasp latch, dropped counts per D3 candidate).

    uv run --no-sync python scripts/summarize_policy_approach.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RUN = Path("outputs/task057-diagnostics/run-1/attempts")
ARMS = ("A0_proprio_only", "A1_random_encoder", "A2_bc_frozen_e0", "A3_bc_finetuned_e0")
CANDIDATES = ("all_translation", "dx", "dy", "dz", "droll", "dpitch", "dyaw", "grasp")
#: A reset "descends" when the arm's mean commanded dz after step 100 is below this.
DESCEND = -0.15


def load(stage):
    return {int(p.stem): json.loads(p.read_text()) for p in sorted((RUN / stage).glob("*.json"))}


def arm_approach(arm):
    per_reset = {}
    for seed, a in load(f"none-{arm}").items():
        t = a["trace"]
        c = np.array([r["commanded"] for r in t], float)
        d = np.array([r["palm_apple_m"] for r in t], float)
        late = c[100:, :3] if len(c) > 100 else c[:, :3]
        per_reset[seed] = {
            "steps": len(t),
            "early_mean_dx_dy_dz_0_19": c[:20, :3].mean(0).round(4).tolist(),
            "late_mean_dx_dy_dz_100_on": late.mean(0).round(4).tolist(),
            "palm_apple_m_step0": round(float(d[0]), 4),
            "palm_apple_m_step130": round(float(d[min(130, len(d) - 1)]), 4),
            "palm_apple_m_final": round(float(d[-1]), 4),
            "palm_apple_m_min": round(float(d.min()), 4),
            "descends": bool(late[:, 2].mean() < DESCEND),
            "termination": a["termination_reason"],
        }
    late = np.array([v["late_mean_dx_dy_dz_100_on"] for v in per_reset.values()])
    return {
        "across_reset_mean_late": late.mean(0).round(4).tolist(),
        "across_reset_std_late": late.std(0).round(4).tolist(),
        "max_abs_per_reset_late_mean": np.abs(late).max(0).round(4).tolist(),
        "descending_resets": sorted(s for s, v in per_reset.items() if v["descends"]),
        "per_reset": per_reset,
    }


def representative(arm="A2_bc_frozen_e0", seed=45001, start=130):
    t = load(f"none-{arm}")[seed]["trace"][start:]
    c = np.array([r["controller"] for r in t], float)
    e = np.array([r["shadow_expert"] for r in t if r["shadow_expert"] is not None], float)
    d = np.array([r["palm_apple_m"] for r in t], float)
    return {
        "arm": arm,
        "seed": seed,
        "from_step": start,
        "palm_apple_m_min_max": [round(float(d.min()), 4), round(float(d.max()), 4)],
        "controller_mean_dx_dy_dz": c[:, :3].mean(0).round(3).tolist(),
        "controller_std_dx_dy_dz": c[:, :3].std(0).round(4).tolist(),
        "expert_defined_steps": len(e),
        "expert_mean_dx_dy_dz": e[:, :3].mean(0).round(3).tolist(),
        "expert_mean_abs_dx_dy_dz": np.abs(e[:, :3]).mean(0).round(3).tolist(),
    }


def after_grasp(arm, seed):
    t = load(f"none-{arm}")[seed]["trace"]
    latch = next(i for i, r in enumerate(t) if r["after"]["stage"] in ("grasp", "transport"))
    rows = t[latch:]
    c = np.array([r["controller"] for r in rows], float)
    e = np.array([r["shadow_expert"] for r in rows if r["shadow_expert"] is not None], float)
    return {
        "arm": arm,
        "seed": seed,
        "latch_step": latch,
        "expert_phase_at_latch": t[latch]["shadow_expert_phase"],
        "controller_mean_abs_dx_dy": np.abs(c[:, :2]).mean(0).round(3).tolist(),
        "expert_mean_abs_dx_dy": np.abs(e[:, :2]).mean(0).round(3).tolist(),
        "expert_defined_steps": len(e),
    }


def d3_dropped():
    out = {}
    for candidate in CANDIDATES:
        attempts = load(f"D3-A2-{candidate}").values()
        dropped = [a for a in attempts if a["score"].get("dropped")]
        far = [a for a in dropped if (a["score"].get("object_plate_distance_m") or 0) > 1.0]
        out[candidate] = {"dropped": len(dropped), "dropped_more_than_1m_from_plate": len(far)}
    return out


def failing_contrasts():
    out = {}
    for arm in ("A2_bc_frozen_e0", "A3_bc_finetuned_e0"):
        d = np.array([r["palm_apple_m"] for r in load(f"none-{arm}")[45000]["trace"]], float)
        out[arm] = {
            "step0": round(float(d[0]), 4),
            "step130": round(float(d[130]), 4),
            "max": round(float(d.max()), 4),
            "final": round(float(d[-1]), 4),
        }
    return out


def main() -> int:
    summary = {
        "approach": {arm: arm_approach(arm) for arm in ARMS},
        "representative_trace": representative(),
        "after_grasp": [
            after_grasp("A2_bc_frozen_e0", 45100),
            after_grasp("A3_bc_finetuned_e0", 45006),
        ],
        "d3_dropped": d3_dropped(),
        "failing_contrasts_45000": failing_contrasts(),
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
