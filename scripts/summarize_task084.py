"""TASK-084 (Phase 0) part A: the decision from the eight run directories.

Protocol: ``docs/experiments/jepa_wms_pusht_calibration.md`` §4. Reads
``<runs>/{upstream,ours}-s{1,2,3,4}/{episodes.jsonl,summary.json}`` (or a seed's one allowed
re-run ``-r2`` when the first did not complete), checks that both arms ran the same 96 episodes
with one revision, and applies the first matching row of §4.5. Writes ``<output>`` (JSON) and
refuses an existing file. NumPy only; runs in this repository's environment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PUBLISHED_PERCENT = 70.2  # arXiv 2512.24497, Table 1 (v1) / Table 2 (v4), JEPA-WM, Push-T, CEM L2
WINDOW_POINTS = 10.0
META_SEEDS = (1, 2, 3, 4)
EPISODES_PER_SEED = 24
ARMS = ("upstream", "ours")
BOOTSTRAP_SALT = 8602
RESAMPLES = 20_000
CHECKPOINT_SHA256 = "9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb"
ACTION_SCALE = (3.703, 3.301)  # the bridge's b (protocol §4.1)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p (binomial on the b + c discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def run_folder(runs: Path, arm: str, seed: int) -> tuple[Path, list[str]]:
    """The run directory that counts: ``{arm}-s{seed}``, or its one allowed re-run
    ``{arm}-s{seed}-r2`` when the first did not complete (protocol §4.3)."""
    first, rerun = runs / f"{arm}-s{seed}", runs / f"{arm}-s{seed}-r2"
    if rerun.exists():
        if (first / "summary.json").exists():
            return rerun, [f"{rerun.name} exists although {first.name} completed"]
        return rerun, []
    return first, []


def load(runs: Path) -> dict:
    data: dict = {"void": []}
    revisions = set()
    for arm in ARMS:
        rows = []
        for seed in META_SEEDS:
            folder, problems = run_folder(runs, arm, seed)
            data["void"] += problems
            if not (folder / "summary.json").exists():
                data["void"].append(f"{folder.name}: no completed run")
                continue
            summary = json.loads((folder / "summary.json").read_text())
            if summary["arm"] != arm or int(summary["meta_seed"]) != seed:
                data["void"].append(f"{folder.name}: summary names another arm or seed")
            if summary["checkpoint_sha256"] != CHECKPOINT_SHA256:
                data["void"].append(f"{folder.name}: another checkpoint")
            if arm == "ours" and not np.allclose(
                summary["action_scale_raw"], ACTION_SCALE, atol=1e-3
            ):
                data["void"].append(f"{folder.name}: another action scale")
            revisions.add((summary["repository_revision"], summary["upstream_revision"]))
            episodes = [json.loads(line) for line in (folder / "episodes.jsonl").open()]
            if len(episodes) != EPISODES_PER_SEED:
                data["void"].append(f"{folder.name}: {len(episodes)} episodes")
            for e in episodes:
                e["meta_seed"] = seed
            rows += episodes
        data[arm] = rows
    if len(revisions) > 1:
        data["void"].append(f"the runs used different revisions: {sorted(revisions)}")
    data["revisions"] = sorted(revisions)
    return data


def decide(data: dict) -> dict:
    expected = len(META_SEEDS) * EPISODES_PER_SEED
    void = list(data.get("void", []))
    for arm in ARMS:
        if len(data[arm]) != expected:
            void.append(f"{arm} ran {len(data[arm])} of {expected} episodes")
    if not void:
        for u, o in zip(data["upstream"], data["ours"], strict=True):
            key = ("meta_seed", "episode", "ep_seed", "init_state", "goal_state")
            if any(u[k] != o[k] for k in key):
                void.append(f"episode {u['meta_seed']}/{u['episode']} differs between the arms")
                break
    if void:
        return {"row": "P0-VOID", "reasons": void}
    up = np.array([bool(r["success"]) for r in data["upstream"]])
    ours = np.array([bool(r["success"]) for r in data["ours"]])
    n = len(up)
    k_up, k_ours = int(up.sum()), int(ours.sum())
    d = ours.astype(float) - up.astype(float)
    rng = np.random.default_rng(BOOTSTRAP_SALT)
    idx = rng.integers(0, n, size=(RESAMPLES, n))
    boot = d[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    b = int((up & ~ours).sum())  # upstream only
    c = int((~up & ours).sum())  # ours only
    p_up, p_ours, diff = 100 * k_up / n, 100 * k_ours / n, 100 * float(d.mean())
    g_repro = abs(p_up - PUBLISHED_PERCENT) <= WINDOW_POINTS
    g_plan_rel = diff >= -WINDOW_POINTS
    g_plan_abs = p_ours >= PUBLISHED_PERCENT - WINDOW_POINTS
    g_plan = g_plan_rel and g_plan_abs
    if g_repro and g_plan:
        row = "P0-PASS"
    elif g_repro:
        row = "P0-PLANNER-GAP"
    else:
        row = "P0-REPRO-FAIL"
    return {
        "row": row,
        "G-REPRO": bool(g_repro),
        "G-PLAN": bool(g_plan),
        "G-PLAN-REL": bool(g_plan_rel),
        "G-PLAN-ABS": bool(g_plan_abs),
        "G-PLAN-clear (reported)": bool(100 * lo > -WINDOW_POINTS),
        "episodes": n,
        "upstream": {
            "successes": k_up,
            "percent": p_up,
            "wilson95_percent": [100 * x for x in wilson(k_up, n)],
        },
        "ours": {
            "successes": k_ours,
            "percent": p_ours,
            "wilson95_percent": [100 * x for x in wilson(k_ours, n)],
        },
        "ours_minus_upstream_points": diff,
        "paired_bootstrap95_points": [100 * float(lo), 100 * float(hi)],
        "discordant": {"upstream_only": b, "ours_only": c},
        "mcnemar_exact_two_sided_p": mcnemar_exact(b, c),
        "ours_minus_published_points": p_ours - PUBLISHED_PERCENT,
        "upstream_minus_published_points": p_up - PUBLISHED_PERCENT,
        "per_seed": {
            str(s): {
                arm: int(sum(r["success"] for r in data[arm] if r["meta_seed"] == s))
                for arm in ARMS
            }
            for s in META_SEEDS
        },
        "seconds_per_episode_median": {
            arm: float(np.median([r["seconds"] for r in data[arm]])) for arm in ARMS
        },
        "published_percent": PUBLISHED_PERCENT,
        "window_points": WINDOW_POINTS,
        "bootstrap": {"salt": BOOTSTRAP_SALT, "resamples": RESAMPLES},
        "revisions": data.get("revisions"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = decide(load(args.runs))
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("row",) if k in result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
