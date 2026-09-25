"""TASK-059: build the committed results manifest from the gated run's ``report.json``.

Read-only. Every number in ``docs/experiments/apple_info_ceiling_v1_results.md`` is copied from
the manifest this script writes, and every manifest value is taken from ``report.json`` by the
field path recorded next to it -- nothing is recomputed, so a reviewer can check each number
against the artifact (whose sha256 is recorded) and against the code that computed it
(``src/embodied_jepa/info_ceiling.py``, ``scripts/probe_info_ceiling.py``).

    uv run --no-sync python scripts/summarize_info_ceiling.py \
        --report outputs/task059-info-ceiling/run-2/report.json \
        --output benchmarks/manifests/apple-info-ceiling-v1-results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCES = ("E0", "A3", "random", "raw112", "raw448", "raw448down")
STRATA = ("all", "occluded", "visible", "secondary_train_to_val")


def _stratum(e: dict) -> dict:
    t1, t2, t3 = e["T1"], e["T2"], e["T3"]
    return {
        "n": e["n"],
        "T1_median_cm": t1["median"],
        "T1_ci95": t1["ci95"],
        "T1_ratio_to_B_occ": t1["ratio_to_B_occ"]["ratio"],
        "T1_ratio_to_B_occ_ci95": t1["ratio_to_B_occ"]["ci95"],
        "B_occ_median_cm": t1["B_occ_median_cm"],
        "B_mean_median_cm": t1["B_mean_median_cm"],
        "T1_pass": t1["pass"],
        "T2_correct": t2["correct"],
        "T2_accuracy": t2["accuracy"],
        "T2_wilson95": t2["wilson95"],
        "B_maj_accuracy": t2["B_maj_accuracy"],
        "B_y_accuracy": t2["B_y_accuracy"],
        "T2_pass": t2["pass"],
        "T3_mae": t3["mae"],
        "T3_ci95": t3["ci95"],
        "T3_ratio_to_B_const": t3["ratio_to_B_const"]["ratio"],
        "T3_ratio_to_B_const_ci95": t3["ratio_to_B_const"]["ci95"],
        "B_const_mae": t3["B_const_mae"],
        "T3_pass": t3["pass"],
        "succeeds": e["succeeds"],
        "beats_prior": e["beats_prior"],
    }


def _t4(t: dict) -> dict:
    x, dy, d = t["apple_x_abs_error_cm"], t["dy_mae"], t["derived_dx_sign"]
    return {
        "n": t["n"],
        "apple_x_median_cm": x["median"],
        "apple_x_ci95": x["ci95"],
        "apple_x_B_mean_median_cm": x["B_mean_median_cm"],
        "dy_mae": dy["mae"],
        "dy_B_const_mae": dy["B_const_dy_mae"],
        "dy_ratio": dy["ratio_to_B_const_dy"]["ratio"],
        "dy_zero_baseline": bool(dy["ratio_to_B_const_dy"].get("undefined_zero_baseline")),
        "derived_dx_sign_correct": d["correct"],
        "derived_dx_sign_accuracy": d["accuracy"],
        "derived_dx_sign_wilson95": d["wilson95"],
    }


def summarize(report: dict, report_path: Path) -> dict:
    results = report["results"]
    out = {
        "protocol": report["protocol"],
        "task": report["task"],
        "status": report["status"],
        "outcome": report["outcome"],
        "decision": report["decision"],
        "provenance": {
            "report": str(report_path),
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "source": report["source"],
            "environment": report["environment"],
            "elapsed_seconds": report["elapsed_seconds"],
            "peak_rss_bytes": report["peak_rss_bytes"],
            "input_hashes": report["hashes"],
            "encoder_weights_sha256": report["encoder_weights_sha256"],
            "fold_assignment_sha256": report["fold_assignment_sha256"],
            "guards": report["guards"],
            "non_finite_fields": report["non_finite_fields"],
        },
        "prior_baselines": report["prior_baselines"],
        "sources": {},
        "pairwise_decisional": {
            pair: {
                stratum: {
                    "median_error_difference_cm": c["median_error_difference_cm"],
                    "mcnemar": c["mcnemar"],
                }
                for stratum, c in value.items()
            }
            for pair, value in report["pairwise_decisional"].items()
        },
        "explanation_check": report["explanation_check"],
        "descriptive": {
            k: v for k, v in report["descriptive"].items() if k != "visibility_over_time_per_root"
        },
        "field_paths": "sources.<s>.<stratum> <- report.results.<s>.<stratum>; "
        "sources.<s>.T4.<stratum> <- report.results.<s>.T4_reported.<stratum>; "
        "random_floor <- report.results.<E0|A3>.random_floor",
    }
    for name in SOURCES:
        res = results[name]
        entry = {
            "decisional": res["decisional"],
            "feature_dim": res["feature_dim"],
            **{stratum: _stratum(res[stratum]) for stratum in STRATA},
            "T4": {stratum: _t4(t) for stratum, t in res["T4_reported"].items()},
            "selections": {
                target: [(s["family"], s["lam_rel"]) for s in sel]
                for target, sel in res["selections"].items()
            },
        }
        if "random_floor" in res:
            entry["random_floor"] = res["random_floor"]
        out["sources"][name] = entry
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report.get("smoke"):
        raise SystemExit("refusing to summarize a smoke report")
    summary = summarize(report, args.report)
    args.output.write_text(json.dumps(summary, indent=1, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {args.output}: outcome {summary['outcome']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
