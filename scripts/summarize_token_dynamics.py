"""TASK-066 results manifest from the gated run's ``report.json`` (``apple_token_dynamics_v1``).

Writes ``benchmarks/manifests/apple-token-dynamics-v1-results.json``. It restates the numbers the
results document quotes, each copied from the report and rounded to 3 decimals, with the report's
sha256, so a reviewer can check every restated number against the report by script. It
computes no new statistic. It derives three fields from the report and TASK-065's results
manifest: the saturation list, the rows that also match (protocol section 11) and the
side-by-side with TASK-065. A world-model test only; **learned Apple->Plate is still 0
successes.**

    uv run --no-sync python scripts/summarize_token_dynamics.py \\
        --report outputs/task066-token-dynamics/run-1/report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "manifests" / "apple-token-dynamics-v1-results.json"
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


def comparative(entry: dict) -> dict:
    keys = ("difference", "projected_rank_W", "projected_rank_N", "projected_rank_encoded")
    return {k: r3(entry[k]) for k in keys} | {
        "ci95": [r3(v) for v in entry["ci95"]],
        "undefined_resamples": entry["undefined_resamples"],
        "resamples": entry["resamples"],
        "sessions": entry["sessions"],
    }


def summarize(report: dict, report_sha: str, report_path: str) -> dict:
    out = {
        "task": report["task"],
        "protocol": report["protocol"],
        "document": "docs/experiments/apple_token_dynamics_v1_results.md",
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
            "selected_in_last_two_points": m["selected_in_last_two_points"],
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
    out["frozen"] = report["frozen"]
    out["ceiling"] = {
        h: {
            "median_cm": r3(v["median_cm"]),
            "ratio_to_B_occ_ci95": [r3(x) for x in v["ratio_to_B_occ_ci95"]],
            "meets_t1": v["meets_t1"],
        }
        for h, v in report["ceiling"].items()
    }
    out["eall"] = {
        s: {
            h: {
                **{k: ratio(v[k]) for k in RATIOS},
                **{k: r3(v[k]) for k in v if k.startswith("mse_")},
                **{k: collapse(v[k]) for k in v if k.startswith("collapse_")},
                **(
                    {"rank_W_over_N": comparative(v["rank_W_over_N"])}
                    if "rank_W_over_N" in v
                    else {}
                ),
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
    out["saturation"] = saturation(report["models"])
    out["rows_also_matching"] = rows_also_matching(report)
    out["task065_side_by_side_cm"] = side_by_side(out)
    return out


def saturation(models) -> dict:
    """Reported only: models whose selected checkpoint is among their last two val points."""
    late = [
        f"{m['arm']}-s{m['seed']}-{m['half']}" for m in models if m["selected_in_last_two_points"]
    ]
    final = [
        f"{m['arm']}-s{m['seed']}-{m['half']}"
        for m in models
        if m["selected_update"] == m["val_curve"][-1][0]
    ]
    return {"selected_in_last_two_points": late, "selected_final_update": final}


def rows_also_matching(report) -> list:
    """Protocol section 11: every row whose condition also holds (evaluated from the gates and
    the ceiling, in order), not only the first."""
    seeds = {s: g["seed"] for s, g in report["gates"].items()}
    passing = [s for s, g in seeds.items() if g["passes"]]
    dynamics = [s for s, g in seeds.items() if g["dynamics"]]
    latent = [s for s, g in seeds.items() if g["latent"]]
    ceiling_ok = all(v["meets_t1"] for v in report["ceiling"].values())
    rows = []
    if len(passing) == len(seeds):
        rows.append("WM-TOK-DYNAMICS")
    if 0 < len(passing) < len(seeds):
        rows.append("WM-TOK-UNSTABLE")
    if not passing and len(dynamics) >= 2 and not ceiling_ok:
        rows.append("WM-TOK-CEILING")
    if not passing and len(dynamics) >= 2:
        rows.append("WM-TOK-APPLE-LOST")
    if not passing and len(latent) >= 2:
        rows.append("WM-TOK-COLLAPSE")
    if not rows:  # section 11: "otherwise", i.e. only when no row above matches
        rows.append("WM-TOK-NO-DYNAMICS")
    return rows


TASK065 = ROOT / "benchmarks" / "manifests" / "apple-latent-dynamics-v1-results.json"


def side_by_side(out) -> dict:
    """Context only (protocol section 8): apple readability in cm and G1's rank ratio, TASK-065
    (frozen CLS) against this task (pooled tokens). Latent MSE is not comparable across latents."""
    prior = json.loads(TASK065.read_text())
    table = {}
    for h in GATED:
        table[h] = {
            "encoded_cm": {
                "task065_cls": prior["epost"]["encoded"][h]["median_cm"],
                "task066_tokens": out["epost"]["encoded"][h]["median_cm"],
            },
            "W_cm": {
                "task065_cls": [prior["epost"]["per_seed"][s][h]["W"]["median_cm"] for s in "012"],
                "task066_tokens": [out["epost"]["per_seed"][s][h]["W"]["median_cm"] for s in "012"],
            },
            "rank_ratio_W": {
                "task065_cls": [
                    prior["eall"][s][h]["collapse_W"]["effective_rank_ratio"] for s in "012"
                ],
                "task066_tokens": [
                    out["eall"][s][h]["collapse_W"]["effective_rank_ratio"] for s in "012"
                ],
            },
        }
    return table


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
