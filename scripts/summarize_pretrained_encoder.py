"""TASK-063: build the committed results manifest from the gated run's ``report.json``.

Read-only. Every number in ``docs/experiments/apple_pretrained_encoder_v1_results.md`` is copied
from the manifest this script writes. Every manifest value is taken from ``report.json`` by a
recorded field path; nothing is recomputed.

    uv run --no-sync python scripts/summarize_pretrained_encoder.py \
        --report outputs/task063-pretrained-encoder/run-1/report.json \
        --output benchmarks/manifests/apple-pretrained-encoder-v1-results.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATA = ("all", "reset_occluded", "reset_visible", "arm_occluded", "arm_visible")
DECISIONAL = {"P_cls": "P-cls", "P_tok": "P-tok"}
FLOORS = {"R_cls": "R-cls", "R_tok": "R-tok"}


def _task059_summarizer():
    spec = importlib.util.spec_from_file_location(
        "_summarize_info_ceiling", ROOT / "scripts" / "summarize_info_ceiling.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _task059_summarizer()


def _stats(s: dict) -> dict:
    keys = ("latent_std_mean", "latent_std_min", "collapsed_fraction", "effective_rank")
    return {k: s[k] for k in keys} | {"samples": s["samples"], "dimension": s["dimension"]}


def summarize(report: dict, report_path: Path) -> dict:
    sources = {}
    for name, res in report["results"].items():
        sources[name] = {
            "decisional_arm": DECISIONAL.get(name),
            "floor": FLOORS.get(name),
            "feature_dim": res["feature_dim"],
            "feature_sha256": report["feature_sha256"][name],
            **{s: BASE._stratum(res[s]) for s in STRATA if s in res},
            "secondary_train_to_val": BASE._stratum(res["secondary_train_to_val"]),
            "selections": {
                target: [(s["family"], s["lam_rel"]) for s in sel]
                for target, sel in res["selections"].items()
            },
        }
    arms = {}
    for arm, e in report["arms"].items():
        hidden = e["spurious_check"]["hidden_all_roots"]
        arms[arm] = {
            "source": e["source"],
            "floor": e["floor"],
            "readout_point": e["readout_point"],
            "state": e["state"],
            "seconds": e["seconds"],
            "per_arm_cap_seconds": e["per_arm_cap_seconds"],
            "non_finite_features": e["non_finite_features"],
            "succeeds": e["succeeds"],
            "beats_prior": e["beats_prior"],
            "beats_floor": e["beats_floor"],
            "floor_comparison": e.get("floor_comparison"),
            "pvalues": e["pvalues"],
            "p": e["p"],
            "holm_rejected": e["holm_rejected"],
            "spurious": e["spurious"],
            "spurious_check": {
                "available": e["spurious_check"]["available"],
                "reading": e["spurious_check"]["reading"],
                "hidden_all_roots": BASE._stratum(hidden) if hidden else None,
            },
            "passes": e["passes"],
            "qualifier": e["qualifier"],
        }
    return {
        "protocol": report["protocol"],
        "task": report["task"],
        "status": report["status"],
        "outcome": report["outcome"],
        "decision": report["decision"],
        # Fixed status lines (protocol §11); the report carries the same values.
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
            "weights": report["weights"],
            "reference_encoder_weights_sha256": report["reference_encoder_weights_sha256"],
            "fold_assignment_sha256": report["fold_assignment_sha256"],
            "post_look_frames_sha256": report["post_look_frames_sha256"],
            "ablation_renderer_reproduces_L": report["ablation_renderer_reproduces_L"],
            "stage_code_equals_task061_features": report["stage_code_equals_task061_features"],
            "G_repro": report["G_repro"],
            "feature_seconds": report["feature_seconds"],
            "feature_failures": report["feature_failures"],
            "readout_seconds": report["readout_seconds"],
            "non_finite_fields": report["non_finite_fields"],
        },
        "anchor": report["anchor"],
        "diagnostics": {k: _stats(v) for k, v in report["diagnostics"].items()},
        "label_free": report["label_free"],
        "holm": report["holm"],
        "arms": arms,
        "sources": sources,
        "pairwise_reported": report["pairwise_reported"],
        "field_paths": "sources.<s>.<stratum> <- report.results.<s>.<stratum>; "
        "arms.<a> <- report.arms.<a>; holm <- report.holm; anchor <- report.anchor; "
        "diagnostics <- report.diagnostics; label_free <- report.label_free; "
        "pairwise_reported <- report.pairwise_reported",
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
