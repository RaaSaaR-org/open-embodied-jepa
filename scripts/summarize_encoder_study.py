"""TASK-062: build the committed results manifest from the gated run's ``report.json``.

Read-only. Every number in ``docs/experiments/apple_encoder_study_v1_results.md`` is copied from
the manifest this script writes. Every manifest value is taken from ``report.json`` by a recorded
field path; nothing is recomputed.

    uv run --no-sync python scripts/summarize_encoder_study.py \
        --report outputs/task062-encoder-study/run-1/report.json \
        --output benchmarks/manifests/apple-encoder-study-v1-results.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATA = ("all", "reset_occluded", "reset_visible", "arm_occluded", "arm_visible")
DECISIONAL = {"A_tok": "A-tok", "A_sig": "A-sig", "A_rec": "A-rec", "A_plain": "A-plain"}


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
    results = report["results"]
    sources = {}
    for name, res in results.items():
        sources[name] = {
            "decisional_arm": DECISIONAL.get(name),
            "feature_dim": res["feature_dim"],
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
            "state": e["state"],
            "trained": e["trained"],
            "not_collapsed": e["not_collapsed"],
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
    encoders = {
        k: {
            key: v[key]
            for key in (
                "status",
                "completed_steps",
                "elapsed_seconds",
                "device",
                "init_weights_digest",
                "encoder_digest",
                "checkpoint",
                "checkpoint_sha256",
                "overrides",
                "image_sigreg_weight",
                "reconstruction_weight",
                "readout_targets",
                "train_windows",
            )
        }
        | ({"decoder_sha256": v["decoder_sha256"]} if "decoder_sha256" in v else {})
        for k, v in report["encoders"].items()
    }
    diagnostics = {
        k: {
            "collapse_training_half": {
                f: _stats(s) for f, s in v["collapse_training_half"].items()
            },
            "collapse_post_look": {f: _stats(s) for f, s in v["collapse_post_look"].items()},
            "action_sensitivity": v["action_sensitivity"],
            "label_free_own_held_out_roots": v["label_free_own_held_out_roots"],
        }
        for k, v in report["diagnostics"].items()
    }
    split = {h: v for h, v in report["training_split"].items() if h != "episode_ids"}
    return {
        "protocol": report["protocol"],
        "task": report["task"],
        "status": report["status"],
        "outcome": report["outcome"],
        "decision": report["decision"],
        # Fixed status lines (protocol §10), not report fields: this task trains no controller.
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
            "F_plain_init_digest": report["F_plain_init_digest"],
            "fold_assignment_sha256": report["fold_assignment_sha256"],
            "post_look_frames_sha256": report["post_look_frames_sha256"],
            "ablation_renderer_reproduces_L": report["ablation_renderer_reproduces_L"],
            "stage_code_equals_task061_features": report["stage_code_equals_task061_features"],
            "non_finite_fields": report["non_finite_fields"],
        },
        "anchor": report["anchor"],
        "training_split": split,
        "training_data": report["training_data"],
        "encoders": encoders,
        "diagnostics": diagnostics,
        "label_free_reference": report["label_free_reference"],
        "holm": report["holm"],
        "arms": arms,
        "sources": sources,
        "pairwise_reported": report["pairwise_reported"],
        "field_paths": "sources.<s>.<stratum> <- report.results.<s>.<stratum>; "
        "arms.<a> <- report.arms.<a>; encoders/diagnostics <- report.encoders/diagnostics; "
        "holm <- report.holm; anchor <- report.anchor",
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
