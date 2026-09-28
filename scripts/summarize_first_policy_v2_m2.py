"""TASK-072 M2: write the results manifest from the gated run's ``report.json`` (read-only).

    uv run --no-sync python scripts/summarize_first_policy_v2_m2.py \\
        --run-root ../worktrees/task072-m2-run \\
        --evidence ../worktrees/task072-run \\
        --output benchmarks/manifests/apple-first-policy-v2-m2-results.json

Every measured number is copied from the report or computed from its per-reset records (Wilson
intervals, exact McNemar, final-distance quantiles, step ranges). The script refuses to write
unless all of these hold:
- the report's sha256 is the one recorded here, and the run is a clean, non-void ``run``-mode
  run at the merged revision;
- the cohort is exactly the stored cohort-C values of ``apple-policy-v1.json``;
- every arm's attempts are in cohort order and its counts equal its per-reset records;
- ``decide_m2``, recomputed from the per-reset records, equals the recorded decision, row and
  gates;
- the carried checkpoints' sha256, recomputed from the files, equal the pinned evidence;
- G-repro held, the pins at the end equal the manifest's, and the test split was not decoded.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import first_policy_v2_m2 as fm  # noqa: E402

REPORT = "outputs/task072-m2-cohort-c/run-1/report.json"
REPORT_SHA256 = "aa274cddbdb31a3641bc7bc55abc7a43e3ceb0162d60b573b7c68d8d2e301baa"
MERGED_REVISION = "23e25937b1f195d7cf5acf6693fe42334021ac69"
LOG = "task072-m2-run-1.log"  # next to the run worktree
PRERUN = {
    "verdict": "GO, reported by the independent pre-run reviewer and posted on PR #102 as "
    "comment 5879303699; the orchestrator was told, and told the owner, before the run started",
    "render_check": "IDENTICAL at 16 workers (16 workers used; 32/32 seeds rendered on >= 2 "
    "workers, 4 renders each, including after 740-step attempts; negative control raised G-frame "
    "at seed 52132); revision 23e2593, clean, 19 s",
    "render_report_sha256_prefix": "4a72bb2e…ef2c",
    "preflight": "PREFLIGHT-READY in 41 s at 23e2593 (G-repro 8/8 exact; G-cohort passed), "
    "report sha256 9953997c…9c8f",
    "earlier_review": "REQUEST CHANGES at 714b9d0 (verdict leak on V, a non-discriminating "
    "stored-reset test, the render check missing from the GO); fixed in 3350f3d before the GO",
}
G3_DECLARED_READING = (
    "a learned visuomotor policy works; encoder pretraining contributes nothing measurable"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 6), round(min(1.0, centre + half), 6)]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def summarize(report: dict, report_sha: str, evidence: Path, policy_v1: dict) -> dict:
    check(report_sha == REPORT_SHA256, "the report is not the gated run's")
    check(report["mode"] == "run" and report["protocol"] == fm.PROTOCOL, "not a gated M2 run")
    check(report["outcome"] not in (None, "V") and "void_reason" not in report, "the run is void")
    check(
        report["revision"] == report["revision_at_end"] == MERGED_REVISION,
        "the run is not at the merged revision",
    )
    check(report["tracked_tree_dirty"] is False, "the tree was dirty")
    check(report["test_split_decoded"] is False, "the test split was decoded")
    check(report["non_finite_fields"] == [], "non-finite fields in the report")
    manifest = json.loads((ROOT / "benchmarks/manifests/apple-first-policy-v2-m2.json").read_text())
    check(report["pinned_hashes_at_end"] == manifest["hashes"], "pins at the end differ")
    check(report["pinned_hashes_at_preflight"] == manifest["hashes"], "pins at preflight differ")
    check(all(report["stages"]["reproduction"]["checks"].values()), "G-repro did not hold")
    check(
        report["started_utc"] <= report["cohort_first_render_utc"] <= report["ended_utc"],
        "the cohort-C boundary marker is missing or out of order",
    )

    # the cohort: exactly the stored values
    stored = fm.stored_cohort(policy_v1)
    seeds = report["cohort"]["seeds"]
    check(tuple(seeds) == fm.COHORT_C, "the cohort seeds are not cohort C")
    check(
        {int(s): v for s, v in report["cohort"]["resets"].items()} == stored,
        "the executed resets are not the stored cohort-C values",
    )

    # the evidence: recomputed from the files
    for arm, want in fm.EVIDENCE["checkpoints"].items():
        got = sha256(evidence / fm.EVIDENCE["checkpoint_dir"] / f"{arm}.pt")
        check(got == want == report["evidence"]["sha256"][arm], f"{arm} checkpoint mismatch")
    check(
        sha256(evidence / fm.EVIDENCE["report"]) == fm.EVIDENCE["report_sha256"],
        "run-1's report changed",
    )

    # per arm: order, counts, per-reset records
    m2 = report["stages"]["M2"]
    arms = {}
    for arm in fm.ARMS:
        attempts = m2["attempts"][arm]
        check([a["seed"] for a in attempts] == seeds, f"{arm} attempts are not in cohort order")
        per = m2["per_reset"][arm]
        for key in ("success", "grasp", "at_rest"):
            check(per[key] == [bool(a[key]) for a in attempts], f"{arm} {key} records disagree")
            check(m2["counts"][arm][key] == sum(per[key]), f"{arm} {key} count disagrees")
        distances = [
            100 * a["final_score"]["object_plate_distance_m"]
            for a in attempts
            if a.get("final_score") and a["final_score"].get("object_plate_distance_m") is not None
        ]
        success = [a for a in attempts if a["success"]]
        k = m2["counts"][arm]["success"]
        arms[arm] = {
            "counted_success_of_40": k,
            "wilson_95": wilson(k, fm.C_RESETS),
            "grasp_of_40": m2["counts"][arm]["grasp"],
            "at_rest_of_40": m2["counts"][arm]["at_rest"],
            "at_rest_not_counted": m2["counts"][arm]["at_rest_not_counted"],
            "latched_v1_of_40": m2["counts"][arm]["latched"],
            "terminations": dict(
                sorted(collections.Counter(a["termination_reason"] for a in attempts).items())
            ),
            "final_distance_cm_q10_q50_q90": [
                round(float(q), 2) for q in np.quantile(distances, [0.1, 0.5, 0.9])
            ],
            "failed_seeds": [s for s, ok in zip(seeds, per["success"], strict=True) if not ok],
            "success_first_grasp_step_range": [
                min(a["first_grasp_step"] for a in success),
                max(a["first_grasp_step"] for a in success),
            ]
            if success
            else None,
            "success_first_place_step_range": [
                min(a["first_place_step"] for a in success),
                max(a["first_place_step"] for a in success),
            ]
            if success
            else None,
        }

    # the decision, recomputed
    recomputed = fm.decide_m2(
        {a: m2["per_reset"][a]["success"] for a in fm.ARMS},
        {a: m2["per_reset"][a]["grasp"] for a in fm.ARMS},
        m2["privileged_ok"],
        m2["control_time"]["median_seconds"],
    )
    decision = report["decision"]
    check(json.loads(json.dumps(recomputed)) == decision, "decide_m2 does not reproduce")
    check(decision["row"] == report["outcome"], "the outcome is not the decision's row")
    only_g3 = [g for g, ok in decision["gates"].items() if not ok] == ["G3"]

    errors = report["stages"]["cohort_frames"]["readout_errors_cm_descriptive"]
    return {
        "name": "apple-first-policy-v2-m2-results",
        "protocol": fm.PROTOCOL,
        "task": fm.TASK,
        "step": fm.STEP,
        "document": "docs/experiments/apple_first_policy_v2_m2_results.md",
        "preregistration_manifest": "benchmarks/manifests/apple-first-policy-v2-m2.json",
        "generator": "scripts/summarize_first_policy_v2_m2.py",
        "outcome": decision["row"],
        "gates": decision["gates"],
        "failed_gates": [g for g, ok in decision["gates"].items() if not ok],
        "g3_declared_reading": G3_DECLARED_READING if not decision["gates"]["G3"] else None,
        "reading_side_by_side": (
            "row M2-FAIL (first match, §12) and G3's declared reading, reported side by side "
            "without choosing between them (doc §3.2); the owner decides"
            if only_g3
            else None
        ),
        "success_of_40": decision["success_of_40"],
        "grasp_of_40": decision["grasp_of_40"],
        "paired": decision["paired"],
        "arms": arms,
        "control_time": m2["control_time"],
        "privileged_ok": m2["privileged_ok"],
        "stop_rule": report["stop_rule"],
        "readout_errors_on_cohort_C_cm_descriptive": {
            k: {
                "median": round(float(np.median(v)), 3),
                "p90": round(float(np.quantile(v, 0.9)), 3),
                "max": round(float(np.max(v)), 3),
            }
            for k, v in errors.items()
        },
        "b_replay_distinct_roots": len(set(report["stages"]["cohort_frames"]["b_replay_nearest"])),
        "reproduction": report["stages"]["reproduction"]["checks"],
        "cohort": {
            "seeds": [seeds[0], seeds[-1]],
            "n": len(seeds),
            "source": fm.COHORT_SOURCE,
            "executed_resets_equal_stored_values": True,
        },
        "run": {
            "command": "MUJOCO_GL=egl nohup uv run --no-sync python "
            "scripts/run_first_policy_v2_m2.py run --output outputs/task072-m2-cohort-c/run-1 "
            "--evidence "
            "/home/huhn/develop/emai/worktrees/task072-run",
            "report": REPORT,
            "report_sha256": report_sha,
            "log": LOG,
            "revision": report["revision"],
            "tracked_tree_dirty": report["tracked_tree_dirty"],
            "started_utc": report["started_utc"],
            "cohort_first_render_utc": report["cohort_first_render_utc"],
            "ended_utc": report["ended_utc"],
            "total_seconds": round(report["total_seconds"], 1),
            "stage_seconds": {
                k: round(v["seconds"], 1)
                for k, v in report["stages"].items()
                if isinstance(v, dict) and "seconds" in v
            },
            "platform": report["platform"],
            "environment": report["environment"],
            "determinism_at_end": report["determinism_at_end"],
            "encoder_digests": report["encoder_digests"],
            "evidence": report["evidence"]["sha256"],
            "decoded_episodes": report["decoded_episodes"],
            "test_split_decoded": report["test_split_decoded"],
            "void": False,
            "cohort_C_simulated": True,
        },
        "prerun": PRERUN,
        "claim_scope": "cohort C (40 resets) of apple-to-plate-v2, one run, one camera (112 px "
        "onboard); a BC/DAgger MLP on a frozen DINOv2 readout, not LeWM; the frozen v1 "
        "benchmark (0/150 per model) is unchanged",
        "note": "Learned Apple->Plate on the frozen benchmark is still 0 successes.",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    path = args.run_root / REPORT
    report = json.loads(path.read_text())
    policy_v1 = json.loads((ROOT / "benchmarks/manifests/apple-policy-v1.json").read_text())
    out = summarize(report, sha256(path), args.evidence, policy_v1)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.output}: {out['outcome']}, failed gates {out['failed_gates']}")


if __name__ == "__main__":
    main()
