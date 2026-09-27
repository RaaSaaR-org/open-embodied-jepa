"""TASK-065 results manifest from the gated run's ``report.json`` (``apple_latent_dynamics_v1``).

Writes ``benchmarks/manifests/apple-latent-dynamics-v1-results.json``. It restates the numbers the
results document quotes, each copied from the report and rounded to 3 decimals, with the report's
sha256, so a reviewer can check every restated number against the report by script. Computes
nothing new. A world-model test only; **learned Apple->Plate is still 0 successes.**

    uv run --no-sync python scripts/summarize_latent_dynamics.py \\
        --report outputs/task065-latent-dynamics/run-1/report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "manifests" / "apple-latent-dynamics-v1-results.json"
GATED = ("8", "16")
EALL_H = ("1", "2", "4", "8", "16")
POST_H = ("1", "2", "4", "8", "16", "32", "64")
RATIOS = ("W_over_copy_last", "W_over_N", "wrong_over_W", "zero_over_W", "N_over_copy_last")
POST_SOURCES = ("W", "W_wrong_actions", "W_zero_actions", "N", "W_frame8_probe")


def r3(value):
    return None if value is None else round(float(value), 3)


def ratio(entry: dict) -> dict:
    return {
        "ratio": r3(entry["ratio"]),
        "ci95": [r3(v) for v in entry["ci95"]],
        "undefined_resamples": entry.get("undefined_resamples"),
    }


def t1(entry: dict) -> dict:
    out = {
        "median_cm": r3(entry["median"]),
        "ci95": [r3(v) for v in entry["ci95"]],
        "ratio_to_B_occ": ratio(entry["ratio_to_B_occ"]),
    }
    if "excess_over_encoded_cm" in entry:
        e = entry["excess_over_encoded_cm"]
        out["excess_over_encoded_cm"] = {
            "difference": r3(e["difference"]),
            "ci95": [r3(v) for v in e["ci95"]],
        }
    return out


def collapse(entry: dict) -> dict:
    keys = (
        "std_ratio",
        "effective_rank_ratio",
        "predicted_collapsed_fraction",
        "predicted_effective_rank",
        "encoded_effective_rank",
        "predicted_std_mean",
        "encoded_std_mean",
    )
    return {k: r3(entry[k]) for k in keys}


def summarize(report: dict, report_sha: str, report_path: str) -> dict:
    out = {
        "task": report["task"],
        "protocol": report["protocol"],
        "document": "docs/experiments/apple_latent_dynamics_v1_results.md",
        "report": report_path,
        "report_sha256": report_sha,
        "outcome": report["outcome"],
        "decision": report.get("decision"),
        "void_reason": report.get("void_reason"),
        "learned_apple_to_plate_successes": report["learned_apple_to_plate_successes"],
        "exemption_spent": report["exemption_spent"],
        "control_formulation": report["control_formulation"],
        "test_split_decoded": report["test_split_decoded"],
        "revision": report.get("revision"),
        "tracked_tree_dirty": report.get("tracked_tree_dirty"),
        "environment": report.get("environment"),
        "total_seconds": r3(report.get("total_seconds")),
        "stages_elapsed_seconds": {k: r3(v) for k, v in report.get("stages", {}).items()},
        "peak_rss_bytes": report.get("peak_rss_bytes"),
        "non_finite_fields": report.get("non_finite_fields"),
    }
    if report["outcome"] == "V":
        return out
    out["data"] = report["data"]
    out["anchor"] = report["anchor"]
    out["labels"] = {k: report["labels"][k] for k in report["labels"]}
    out["features"] = {k: report["features"][k] for k in report["features"]}
    out["weights_digest"] = report["weights"]["digest"]
    out["evaluation_sets"] = report["evaluation_sets"]
    out["apple_moved_more_than_1mm_roots"] = report["apple_moved_more_than_1mm_roots"]
    out["models"] = [
        {
            "arm": m["arm"],
            "seed": m["seed"],
            "half": m["half"],
            "selected_update": m["selected_update"],
            "selected_val_criterion": r3(m["selected_val_criterion"]),
            "train_seconds": r3(m["train_seconds"]),
            "seconds_with_evaluation": r3(m["seconds_with_evaluation"]),
            "checkpoint_sha256": m["checkpoint_sha256"],
            # protocol section 10: every model's val curve and selected update, always reported
            "val_curve": [[int(u), r3(v)] for u, v in m["val_curve"]],
            "losses": [[int(u), r3(v)] for u, v in m["losses"]],
        }
        for m in report["models"]
    ]
    out["gates"] = report["gates"]
    out["eall"] = {
        s: {
            h: {
                **{k: ratio(v[k]) for k in RATIOS},
                **{k: r3(v[k]) for k in v if k.startswith("mse_")},
                **{k: collapse(v[k]) for k in v if k.startswith("collapse_")},
            }
            for h, v in per_h.items()
        }
        for s, per_h in report["eall"].items()
    }
    ep = report["epost"]
    out["epost"] = {
        "encoded": {h: t1(ep["encoded_and_copy_last"]["encoded"][h]) for h in POST_H},
        "copy_last": {h: t1(ep["encoded_and_copy_last"]["copy_last"][h]) for h in POST_H},
        "per_seed": {
            s: {
                h: {
                    **{src: t1(v[src]) for src in POST_SOURCES},
                    "W_minus_copy_last_cm": {
                        "difference": r3(v["W_minus_copy_last_cm"]["difference"]),
                        "ci95": [r3(x) for x in v["W_minus_copy_last_cm"]["ci95"]],
                    },
                    "W_minus_N_cm": {
                        "difference": r3(v["W_minus_N_cm"]["difference"]),
                        "ci95": [r3(x) for x in v["W_minus_N_cm"]["ci95"]],
                    },
                    "latent_mse": {k: ratio(x) for k, x in v["latent_mse"].items()},
                }
                for h, v in per_h.items()
            }
            for s, per_h in ep["per_seed"].items()
        },
    }
    out["val_secondary"] = {
        s: {
            h: {k: r3(x["median_cm"]) for k, x in v.items() if k != "selection"}
            for h, v in per_h.items()
        }
        for s, per_h in report["val_secondary"].items()
    }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument(
        "--void-report",
        type=Path,
        action="append",
        default=[],
        help="an earlier void run's report.json, recorded as a void run",
    )
    args = parser.parse_args(argv)
    payload = args.report.read_bytes()
    report = json.loads(payload)
    path = args.report.resolve()
    shown = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(args.report)
    result = summarize(report, hashlib.sha256(payload).hexdigest(), shown)
    result["void_runs"] = []
    for path in args.void_report:
        raw = path.read_bytes()
        void = json.loads(raw)
        if void.get("outcome") != "V":
            raise SystemExit(f"{path} is not a void run")
        result["void_runs"].append(
            {
                "report": str(path),
                "report_sha256": hashlib.sha256(raw).hexdigest(),
                "revision": void.get("revision"),
                "outcome": "V",
                "void_reason": void.get("void_reason"),
                "stages_elapsed_seconds": {k: r3(v) for k, v in void.get("stages", {}).items()},
                "total_seconds": r3(void.get("total_seconds")),
            }
        )
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.output}: outcome {result['outcome']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
