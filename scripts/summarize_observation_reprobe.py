"""TASK-061: build the committed results manifest from the gated run's ``report.json``.

Read-only. Every number in ``docs/experiments/apple_observation_reprobe_v1_results.md`` is
copied from the manifest this script writes. Every manifest value is taken from ``report.json``
by a recorded field path; nothing is recomputed. The one exception is the reuse check (protocol
§3.3), which compares the anchor and R-float sources field by field with TASK-059's committed
results manifest and records every difference.

    uv run --no-sync python scripts/summarize_observation_reprobe.py \
        --report outputs/task061-observation-reprobe/run-1/report.json \
        --output benchmarks/manifests/apple-observation-reprobe-v1-results.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK059_RESULTS = ROOT / "benchmarks" / "manifests" / "apple-info-ceiling-v1-results.json"
STRATA = ("all", "reset_occluded", "reset_visible", "arm_occluded", "arm_visible")
# Reuse check: this run's source/stratum -> TASK-059's source/stratum (same roots and targets).
REUSE = {
    "anchor_raw112": "raw112",
    "R_float": "raw448down",
}
REUSE_STRATA = {
    "all": "all",
    "reset_occluded": "occluded",
    "reset_visible": "visible",
    "secondary_train_to_val": "secondary_train_to_val",
}


def _task059_summarizer():
    spec = importlib.util.spec_from_file_location(
        "_summarize_info_ceiling", ROOT / "scripts" / "summarize_info_ceiling.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _task059_summarizer()


def reuse_check(results: dict, task059: dict) -> dict:
    """Field-by-field difference between the anchor/R-float and TASK-059's same sources."""
    out = {}
    for mine, theirs in REUSE.items():
        rows = {}
        for stratum, old_stratum in REUSE_STRATA.items():
            new = BASE._stratum(results[mine][stratum])
            old = task059["sources"][theirs][old_stratum]
            diffs = {}
            for key, value in new.items():
                if key not in old:
                    continue
                a, b = value, old[key]
                if isinstance(a, list):
                    delta = max(abs(x - y) for x, y in zip(a, b, strict=True))
                elif isinstance(a, bool) or a is None or b is None:
                    delta = 0.0 if a == b else float("inf")
                else:
                    delta = abs(a - b)
                diffs[key] = delta
            rows[stratum] = {
                "max_abs_difference": max(diffs.values()),
                "fields_differing": sorted(k for k, v in diffs.items() if v != 0.0),
            }
        out[f"{mine}_vs_task059_{theirs}"] = rows
    return out


def summarize(report: dict, report_path: Path) -> dict:
    results = report["results"]
    task059 = json.loads(TASK059_RESULTS.read_text())
    sources = {}
    for name, res in results.items():
        entry = {
            "arm": res["arm"],
            "target_set": res["target_set"],
            "decisional_hypothesis": res["decisional_hypothesis"],
            "feature_dim": res["feature_dim"],
            **{s: BASE._stratum(res[s]) for s in STRATA if s in res},
            "secondary_train_to_val": BASE._stratum(res["secondary_train_to_val"]),
            "T4": {s: BASE._t4(t) for s, t in res["T4_reported"].items()},
            "selections": {
                target: [(s["family"], s["lam_rel"]) for s in sel]
                for target, sel in res["selections"].items()
            },
        }
        for key in ("B_occ_arm_reported", "random_floor"):
            if key in res:
                entry[key] = res[key]
        sources[name] = entry
    hypotheses = {}
    for name, h in report["hypotheses"].items():
        hidden = h["spurious_check"]["hidden_all_roots"]
        hypotheses[name] = {
            "source": h["source"],
            "passes": h["passes"],
            "qualifier": h["qualifier"],
            "render_path_validated": h["render_path_validated"],
            "succeeds_unadjusted_all_roots": h["succeeds_unadjusted_all_roots"],
            "holm_rejected": h["holm_rejected"],
            "pvalues": h["pvalues"],
            "beats_prior_all_roots": h["beats_prior_all_roots"],
            "spurious_check": {
                "available": h["spurious_check"]["available"],
                "spurious": h["spurious_check"]["spurious"],
                "reading": h["spurious_check"]["reading"],
                "hidden_all_roots": BASE._stratum(hidden) if hidden else None,
            },
        }
    return {
        "protocol": report["protocol"],
        "task": report["task"],
        "status": report["status"],
        "outcome": report["outcome"],
        "decision": report["decision"],
        # Fixed status lines (protocol §8), not report fields: this task trains no controller.
        "learned_apple_to_plate_successes": 0,
        "exemption_spent": False,
        "provenance": {
            "report": str(report_path),
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "source": report["source"],
            "runner_sha256": report["runner_sha256"],
            "environment": report["environment"],
            "elapsed_seconds": report["elapsed_seconds"],
            "peak_rss_bytes": report["peak_rss_bytes"],
            "input_hashes": report["hashes"],
            "encoder_weights_sha256": report["encoder_weights_sha256"],
            "fold_assignment_sha256": report["fold_assignment_sha256"],
            "guards": report["guards"],
            "non_finite_fields": report["non_finite_fields"],
        },
        "look": {k: v for k, v in report["look"].items() if k != "applied_commands"},
        "render_path_validation": report["render_path_validation"],
        "prior_baselines": report["prior_baselines"],
        "holm": report["holm"],
        "hypotheses": hypotheses,
        "sources": sources,
        "pairwise_reported": report["pairwise_reported"],
        "descriptive": report["descriptive"],
        "reuse_check_vs_task059": reuse_check(results, task059),
        "field_paths": "sources.<s>.<stratum> <- report.results.<s>.<stratum>; "
        "sources.<s>.T4.<stratum> <- report.results.<s>.T4_reported.<stratum>; "
        "hypotheses.<h> <- report.hypotheses.<h>; holm <- report.holm",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report.get("smoke"):
        raise SystemExit("refusing to summarize a smoke report")
    if report.get("status") != "complete":
        raise SystemExit(f"refusing to summarize a {report.get('status')!r} report")
    summary = summarize(report, args.report)
    args.output.write_text(json.dumps(summary, indent=1, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {args.output}: outcome {summary['outcome']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
