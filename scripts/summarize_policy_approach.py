"""TASK-057: per-arm approach motion from the run-1 per-step traces (read-only, prints only).

uv run --no-sync python scripts/summarize_policy_approach.py
"""

import json
from pathlib import Path

import numpy as np

for arm in ["A0_proprio_only", "A1_random_encoder", "A2_bc_frozen_e0", "A3_bc_finetuned_e0"]:
    early, late, dist, step0, exp0 = [], [], [], [], []
    for p in sorted(Path(f"outputs/task057-diagnostics/run-1/attempts/none-{arm}").glob("*.json")):
        t = json.loads(p.read_text())["trace"]
        c = np.array([r["commanded"] for r in t])
        early.append(c[:20, :3].mean(0))
        late.append(c[100:, :3].mean(0) if len(c) > 100 else c[:, :3].mean(0))
        step0.append(c[0, :3])
        exp0.append(np.array(t[0]["shadow_expert"][:3]))
        d = [r["palm_apple_m"] for r in t]
        dist.append((d[0], d[min(130, len(d) - 1)], min(d)))
    early, late, dist, step0, exp0 = map(np.array, (early, late, dist, step0, exp0))
    print(arm)
    print("  step0 commanded dx,dy,dz mean", step0.mean(0).round(3), "std", step0.std(0).round(3))
    print("  step0 expert    dx,dy,dz mean", exp0.mean(0).round(3), "std", exp0.std(0).round(3))
    print("  steps 0-19 mean", early.mean(0).round(3), "std across resets", early.std(0).round(3))
    print(
        "  steps 100+ mean",
        late.mean(0).round(3),
        "std across resets",
        late.std(0).round(3),
        "median |.|",
        np.median(np.abs(late), 0).round(3),
    )
    print(
        "  palm-apple median start/step130/min",
        np.median(dist, 0).round(3),
        "change start->130 min/max",
        (dist[:, 1] - dist[:, 0]).min().round(3),
        (dist[:, 1] - dist[:, 0]).max().round(3),
    )
