"""TASK-072: write the results manifest from the Linux run's ``report.json`` (read-only).

    uv run --no-sync python scripts/summarize_first_policy_v2_linux.py \\
        --run-root ../worktrees/task072-run \\
        --run1-report outputs/task071-first-policy-v2/run-1/report.json \\
        --log ../worktrees/task072-run-1.log \\
        --output benchmarks/manifests/apple-first-policy-v2-linux-results.json

Every measured number is copied from the two reports (this run and TASK-071's Mac run-1) or
computed from their per-attempt records: Wilson intervals, per-reset agreement with run-1,
exact McNemar p-values, final-distance quantiles and step ranges. Checked, and the script
refuses to write otherwise: the report and run-1 hashes, each arm's count against its success
seeds, seed alignment with run-1, every checkpoint sha256 (recomputed from the file) against the
report, the corpus manifest sha256, the pins, and that no integer in the cohort-C guard band
appears anywhere in the report. ``run.void`` and ``run.cohort_C_simulated`` are derived from the
report, not asserted.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from math import comb
from pathlib import Path

import numpy as np

RUN1_REPORT_SHA256 = "77aae2077f5e1070f9cdc63b9f228f32515713e869b999b5303a5134c101d90b"
COHORT_C_BAND = (45000, 46000)  # cohort C is 45300-45339; the band is deliberately wider
ARMS = (
    "P-0",
    "P-1",
    "P-2",
    "P-3",
    "C-3",
    "R-3",
    "A4-look",
    "D-oracle-perc",
    "B-oracle",
    "B-replay",
    "B-hold",
    "B-random",
)
LEARNED = ("P-0", "P-1", "P-2", "P-3", "C-3", "R-3")
D2 = list(range(52000, 52016))
RNG_SEEDS = {
    "c0_directions": 7100,
    "readout_folds": 7101,
    "train_sampler": 7102,
    "random_controller": 7103,
    "model": 7104,
    "corpus_split": 7105,
    "corpus_plan_salt": 7106,
}
PRERUN = {
    "verdict": "PRE-RUN: GO (reported by the fresh pre-run reviewer; posted on PR #99 as "
    "comment 5871360006; the orchestrator was told before the run started)",
    "render_check": "IDENTICAL at 16 workers (16 distinct workers; 32/32 seeds on >= 2 workers; "
    "negative control raised G-frame at seed 52132)",
    "render_report": "in the pre-run reviewer's own temporary worktree at db65816, since "
    "removed; the sha256 is as posted in the GO comment",
    "render_report_sha256": "52bf2188a7e5b81e8d647e456c2470a072c144f647d13b1d00a8248c546e4aa8",
    "non_blocking_items": [
        "frozen block void_rule is inherited from TASK-071 and says 'closes TASK-071 as "
        "INCONCLUSIVE'; REPLICATION_RULE['V'] cites 'protocol §14' (TASK-071's section), while "
        "this protocol's void rule is §6. Neither changes behaviour; both are frozen, so any "
        "correction goes through the amendment log. Not amended; the run was not void.",
        "determinism sdp_backends report flash and mem_efficient attention enabled (also at "
        "the end of the run). Unused here: the policy is an MLP and the DINOv2 features run on "
        "the CPU; strict mode would raise rather than fall back (inferred, not tested).",
    ],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 6), round(min(1.0, c + h), 6)]


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def band_integers(node, band=COHORT_C_BAND) -> list[int]:
    """Every integer (not bool) anywhere in ``node`` that lies in ``band``, inclusive."""
    found: list[int] = []
    stack = [node]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
        elif isinstance(x, int) and not isinstance(x, bool) and band[0] <= x <= band[1]:
            found.append(x)
    return found


def is_void(report: dict) -> bool:
    return report["outcome"] == "V" or report["replication"]["row"] == "V"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def m1_summary(stages: dict) -> dict:
    out = {}
    for arm in ARMS:
        attempts = stages["M1"]["attempts"][arm]
        dist = np.array([a["final_score"]["object_plate_distance_m"] * 100 for a in attempts])
        out[arm] = {
            "seeds": [a["seed"] for a in attempts],
            "success_seeds": [a["seed"] for a in attempts if a["success"]],
            "terminations": dict(
                sorted(collections.Counter(a["termination_reason"] for a in attempts).items())
            ),
            "final_distance_cm_q10_q50_q90": [
                round(float(v), 2) for v in np.percentile(dist, [10, 50, 90])
            ],
            "task_truth_in_controller": sum(a.get("task_truth_in_controller", 0) for a in attempts),
        }
    return out


def summarize(
    report: dict, run1: dict, *, report_sha256: str, run1_sha256: str, run_root: Path, log: Path
) -> dict:
    check(run1_sha256 == RUN1_REPORT_SHA256, "run-1 report sha256 differs from TASK-071's")
    check(report["protocol"] == "apple_first_policy_v2_linux", "not a TASK-072 report")
    check(not report["smoke"], "smoke report")
    in_band = band_integers(report)
    check(not in_band, f"cohort-C band integers in the report: {sorted(set(in_band))[:10]}")
    check(
        report["pinned_hashes_at_preflight"] == report["pinned_hashes_at_end"],
        "pins changed during the run",
    )
    s, ms = report["stages"], run1["stages"]
    lin, mac = m1_summary(s), m1_summary(ms)

    m1 = {}
    for arm in ARMS:
        counts = s["M1"]["counts"][arm]
        k = counts["success"]
        check(k == len(lin[arm]["success_seeds"]), f"{arm}: count != success seeds")
        check(lin[arm]["seeds"] == mac[arm]["seeds"] == D2, f"{arm}: seeds not D2 or not aligned")
        a, b = set(lin[arm]["success_seeds"]), set(mac[arm]["success_seeds"])
        row = {k2: v for k2, v in lin[arm].items() if k2 != "seeds"}
        m1[arm] = {
            "counts": counts,
            "wilson95": wilson(k, 16),
            **row,
            "run1": {
                "success": ms["M1"]["counts"][arm]["success"],
                "success_seeds": mac[arm]["success_seeds"],
            },
            "agreement_with_run1": {
                "resets_agreeing": sum((x in a) == (x in b) for x in D2),
                "both": len(a & b),
                "only_linux": sorted(a - b),
                "only_run1": sorted(b - a),
                "exact_mcnemar_two_sided_p": round(mcnemar(len(a - b), len(b - a)), 6),
            },
        }
    pairs = {}
    for a, b in (("P-3", "C-3"), ("P-3", "R-3"), ("P-3", "B-replay")):
        sa, sb = set(lin[a]["success_seeds"]), set(lin[b]["success_seeds"])
        pairs[f"{a} vs {b}"] = {
            "only_first": len(sa - sb),
            "only_second": len(sb - sa),
            "exact_mcnemar_two_sided_p": round(mcnemar(len(sa - sb), len(sb - sa)), 6),
        }
    wins = [a for arm in LEARNED for a in s["M1"]["attempts"][arm] if a["success"]]
    grasp = [a["first_grasp_step"] for a in wins]
    place = [a["first_place_step"] for a in wins]

    corpus_fields = (
        "success",
        "at_rest",
        "latched",
        "split",
        "termination",
        "steps",
        "noise_level",
    )
    la = {a["seed"]: a for a in s["corpus"]["attempts"]}
    ma = {a["seed"]: a for a in ms["corpus"]["attempts"]}
    check(la.keys() == ma.keys(), "corpus seeds differ from run-1")
    corpus_diff = sorted(x for x in la if any(la[x][f] != ma[x][f] for f in corpus_fields))
    c0_fields = ("seed", "condition", "success", "at_rest", "latched_success", "termination_reason")
    c0_diff = sum(
        any(x[f] != y[f] for f in c0_fields)
        for x, y in zip(s["C0"]["attempts"], ms["C0"]["attempts"], strict=True)
    )

    checkpoints = {}
    for arm, rel in report["checkpoints"].items():
        digest = sha256(run_root / rel)
        check(digest == s["trainings"][arm]["sha256"], f"{arm}: checkpoint sha256 != report")
        checkpoints[arm] = {"path": rel, "sha256": digest}
    corpus_sha = sha256(run_root / report["paths"]["corpus"] / "manifest.json")
    check(corpus_sha == s["corpus"]["manifest_sha256"], "corpus manifest sha256 != report")

    dagger = {
        it: {
            arm: {
                **v[arm]["rollouts"],
                "rows_added": v[arm]["rows_added"],
                "rows_total": v[arm]["rows_total"],
                "run1_success": ms["dagger"][it][arm]["rollouts"]["success"],
            }
            for arm in v
        }
        for it, v in s["dagger"].items()
    }
    trainings = {
        arm: {
            "rows": v["rows"],
            "seconds": round(v["seconds"], 1),
            "selected": v["selected"],
            "updates": v["updates"],
            "sha256": v["sha256"],
            "run1_selected_update": ms["trainings"][arm]["selected"]["update"],
        }
        for arm, v in sorted(s["trainings"].items())
    }
    scalars = lambda d: {k: v for k, v in d.items() if not isinstance(v, list)}  # noqa: E731
    s0p = {**scalars(s["S0_P"]), "run1": scalars(ms["S0_P"])}
    p3, c3, r3 = (s["M1"]["counts"][a]["success"] for a in ("P-3", "C-3", "R-3"))

    return {
        "name": "apple-first-policy-v2-linux-results",
        "task": "TASK-072",
        "protocol": "apple_first_policy_v2_linux",
        "document": "docs/experiments/apple_first_policy_v2_linux_results.md",
        "preregistration_manifest": "benchmarks/manifests/apple-first-policy-v2-linux.json",
        "generator": "scripts/summarize_first_policy_v2_linux.py",
        "outcome": report["outcome"],
        "decision": report["decision"],
        "clause_fires": report["clause_fires"],
        "replication": {
            **report["replication"],
            "rule": "REPLICATED iff outcome M1-PASS and P-3 >= 14/16 counted successes "
            "(protocol §5, fixed before the run)",
            "p3_per_reset_agreement_with_run1": m1["P-3"]["agreement_with_run1"],
            "descriptive": {
                "r3_within_2_of_p3": abs(r3 - p3) <= 2,
                "c3_at_least_8_below_p3": p3 - c3 >= 8,
            },
            "comparison_target": {
                "manifest": "benchmarks/manifests/apple-first-policy-v2-results.json",
                "report": "outputs/task071-first-policy-v2/run-1/report.json (Mac run-1, "
                "copied to the Linux PC)",
                "report_sha256": run1_sha256,
            },
        },
        "claim_scope": "replication of TASK-071's development result on the Linux PC; an "
        "existence result on the non-gating development cohort D2 of apple-to-plate-v2; a "
        "learned policy with a DINOv2 encoder (BC/DAgger), not LeWM; the random-init floor R-3 "
        "ties P-3, so no evidence that pretrained vision helps; the official learned "
        "Apple->Plate count on the frozen benchmark is still 0; cohort C untouched; M2 not run",
        "counted_success": "apple_at_rest_v0 AND latched grasp AND latched place before the "
        "settle (owner rulings T71-R1, T71-R2)",
        "m1_d2": m1,
        "m1_paired_linux": pairs,
        "m1_learned_success_step_ranges": {
            "first_grasp": [min(grasp), max(grasp)],
            "first_place": [min(place), max(place)],
        },
        "m1_b_replay_nearest": s["M1"]["b_replay_nearest"],
        "m1_b_replay_library_roots": len(s["M1"]["b_replay_library_roots"]),
        "f_counts": s["M1"]["f_counts"],
        "a4_threshold": s["a4_threshold"],
        "corpus": {
            "path": s["corpus"]["path"],
            "manifest_sha256": s["corpus"]["manifest_sha256"],
            "splits": s["corpus"]["splits"],
            "all": s["corpus"]["all"],
            "train": s["corpus"]["by_split"]["train"],
            "run1_all_success": ms["corpus"]["all"]["success"],
            "roots_differing_from_run1_in_outcome_split_termination_or_steps": corpus_diff,
            "seconds": round(s["corpus"]["seconds"], 1),
        },
        "c0": {
            **{
                k: s["C0"][k]
                for k in ("counted_success_of_32", "at_rest_of_32", "latched_of_32", "bars")
            },
            "attempts_differing_from_run1": c0_diff,
            "seconds": round(s["C0"]["seconds"], 1),
        },
        "s0_p": s0p,
        "s0_d1": s["S0_D1"],
        "readouts": {
            "P": s["readouts"]["P"]["selection"],
            "R": s["readouts"]["R"]["selection"],
            "fit_rows": s["readouts"]["fit_rows"],
        },
        "bc0": s["BC0"],
        "dagger_rollouts": dagger,
        "trainings": trainings,
        "checkpoints": checkpoints,
        "prerun": PRERUN,
        "run": {
            "revision": report["revision"],
            "revision_at_end": report["revision_at_end"],
            "tracked_tree_dirty": report["tracked_tree_dirty"],
            "worktree": "/home/huhn/develop/emai/worktrees/task072-run (clean checkout of "
            "db65816, own .venv)",
            "command": "uv run --no-sync python scripts/run_first_policy_v2_linux.py run "
            "--output outputs/task072-first-policy-v2-linux/run-1 --checkpoints "
            "checkpoints/task072-first-policy-v2-linux/run-1 --corpus "
            "data/apple-look-v2-linux/run-1",
            "started_utc": report["started_utc"],
            "ended_utc": report["ended_utc"],
            "total_seconds": round(report["total_seconds"], 1),
            "global_cap_seconds": 36000,
            "platform": report["platform"],
            "environment": report["environment"],
            "determinism_at_end": report["determinism_at_end"],
            "sim_workers": s["preflight"]["workers"],
            "encoder_digests": report["encoder_digests"],
            "pins_at_preflight": len(report["pinned_hashes_at_preflight"]),
            "pins_at_end": len(report["pinned_hashes_at_end"]),
            "pins_unchanged": True,
            "non_finite_fields": report["non_finite_fields"],
            "test_split_decoded": report["test_split_decoded"],
            "decoded_episodes": report["decoded_episodes"],
            "smoke": report["smoke"],
            "void": is_void(report),
            "report": report["paths"]["output"] + "/report.json",
            "report_sha256": report_sha256,
            "log": f"{log} (one line: report written, outcome {report['outcome']})",
            "log_sha256": sha256(log),
            "paths": report["paths"],
            "cohort_C_simulated": bool(in_band),
            "rng_seeds": RNG_SEEDS,
            "seeds_simulated": {k: [v[0], v[-1]] for k, v in report["data"]["seeds"].items()},
        },
        "note": "Learned Apple->Plate on the frozen benchmark is still 0 successes.",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/task072-first-policy-v2-linux/run-1/report.json"),
        help="relative to --run-root",
    )
    parser.add_argument("--run1-report", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report_path = args.run_root / args.report
    manifest = summarize(
        json.loads(report_path.read_bytes()),
        json.loads(args.run1_report.read_bytes()),
        report_sha256=sha256(report_path),
        run1_sha256=sha256(args.run1_report),
        run_root=args.run_root,
        log=args.log.resolve(),
    )
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    args.output.write_text(text)


if __name__ == "__main__":
    main()
