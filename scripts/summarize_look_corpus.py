"""TASK-064: build the committed results manifest from the gated run's ``collection_report.json``.

Read-only. Every number in ``docs/experiments/apple_look_corpus_v1_results.md`` is copied from the
manifest this script writes. Every manifest value is taken from the report (or from the sealed
dataset's own manifest, for its hash and size) by a recorded field path; nothing is recomputed.
The corpus data is never committed; only this manifest and its hashes are.

    uv run --no-sync python scripts/summarize_look_corpus.py \
        --report data/apple-look-v1-work/collection_report.json \
        --dataset data/apple-look-v1 \
        --output benchmarks/manifests/apple-look-corpus-v1-results.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATA = ("all", "reset_occluded", "reset_visible")
SOURCES = ("P_cls", "R_cls", "P_tok", "R_tok", "P_mean", "R_mean", "L_raw")


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load("_summarize_info_ceiling", "scripts/summarize_info_ceiling.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pairwise(entry: dict) -> dict:
    out = {}
    for stratum, comparison in entry.items():
        diff = comparison["median_error_difference_cm"]
        mc = comparison["mcnemar"]
        out[stratum] = {
            "median_T1_difference_cm": diff["difference"],
            "ci95": diff["ci95"],
            "mcnemar_first_only": mc["only_first_correct"],
            "mcnemar_second_only": mc["only_second_correct"],
            "mcnemar_p_two_sided": mc["p_two_sided"],
        }
    return out


def summarize_gate(gate: dict) -> dict:
    if not gate or gate.get("skipped"):
        return {"evaluated": False}
    sources = {}
    for name in SOURCES:
        if name not in gate["results"]:
            continue
        res = gate["results"][name]
        sources[name] = {
            "feature_dim": res["feature_dim"],
            "feature_sha256": gate["feature_sha256"][name],
            **{s: BASE._stratum(res[s]) for s in STRATA if s in res},
            "secondary_train_to_val": BASE._stratum(res["secondary_train_to_val"]),
        }
    arm = gate["arm"]
    floor = arm.get("floor_comparison") or {}
    hidden = (arm.get("spurious_check") or {}).get("hidden_all_roots")
    return {
        "evaluated": True,
        "roots": gate["roots"],
        "splits": gate["splits"],
        "test_split_decoded": gate["test_split_decoded"],
        "fold_assignment_sha256": gate["fold_assignment_sha256"],
        "stored_post_look_frames_sha256": gate["stored_post_look_frames_sha256"],
        "q_expert": gate["expert"],
        "apple_label_max_abs_m": gate["apple_label_max_abs_m"],
        "ablation_renderer_reproduces": gate["ablation_renderer_reproduces"],
        "visibility": gate["visibility"],
        "prior_summary": gate["prior_summary"],
        "weights": gate["weights"],
        "environment": gate["environment"],
        "G_repro": gate.get("G_repro"),
        "P_cls": {
            "succeeds": arm["succeeds"],
            "beats_prior": arm.get("beats_prior"),
            "beats_floor": arm["beats_floor"],
            "floor_median_T1_difference_cm": floor.get("median_error_difference_cm", {}).get(
                "difference"
            ),
            "floor_median_T1_difference_ci95": floor.get("median_error_difference_cm", {}).get(
                "ci95"
            ),
            "floor_T2_accuracy_higher": floor.get("T2_accuracy_higher"),
            "floor_mcnemar": floor.get("mcnemar"),
            "pvalues": arm.get("pvalues"),
            "p": arm["p"],
            "alpha_one_sided": 0.025,
            "spurious": arm["spurious"],
            "spurious_reading": (arm.get("spurious_check") or {}).get("reading"),
            "hidden_all_roots": BASE._stratum(hidden) if hidden else None,
            "passes": arm["passes"],
        },
        "gate_passes": gate["passes"],
        "sources": sources,
        "pairwise_reported": {k: _pairwise(v) for k, v in gate["pairwise_reported"].items()},
        "seconds": gate["seconds"],
        "peak_rss_bytes": gate["peak_rss_bytes"],
    }


def summarize(report: dict, report_path: Path, dataset: Path | None) -> dict:
    out = {
        "task": report["task"],
        "protocol": report["protocol"],
        "document": "docs/experiments/apple_look_corpus_v1_results.md",
        "source_report": str(report_path),
        "source_report_sha256": sha256(report_path),
        "status": report["status"],
        "outcome": report["outcome"],
        "decision": report["decision"],
        "void_reason": report.get("void_reason"),
        "invoked_via": report.get("invoked_via"),
        "seed_set": report.get("seed_set"),
        "learned_apple_to_plate_successes": report["learned_apple_to_plate_successes"],
        "exemption_spent": report["exemption_spent"],
        "non_finite_fields": report.get("non_finite_fields", []),
        "total_seconds": report.get("total_seconds"),
    }
    if report["status"] != "complete":
        return out
    acc = report["acceptance"]
    out |= {
        "dataset_manifest_sha256": report["dataset_manifest_sha256"],
        "plan_sha256": report["plan_sha256"],
        "plan_sha256_rounded": report["plan_sha256_rounded"],
        "supervisor": report["supervisor"],
        "assembly_seconds": report["assembly_seconds"],
        "source_changed": report["source_changed"],
        "inputs_changed": report["inputs_changed"],
        "pinned_hashes_at_finalize": report.get("pinned_hashes_at_finalize"),
        "verdict": report["verdict"],
        "acceptance": {
            k: acc[k]
            for k in (
                "episodes",
                "root_episodes",
                "branch_episodes",
                "transitions",
                "root_full_success",
                "root_grasp",
                "grasp_phase_attempts",
                "grasp_phase_failures",
                "grasp_phase_failure_fraction",
                "grasp_phase_successes",
                "branch_grasp",
                "branch_dropped_after_grasp",
                "terminations",
                "failure_stage_counts",
                "branch_kind_outcomes",
                "noise_level_outcomes",
                "camera_shapes",
                "by_split",
                "split_counts",
                "split_sessions",
                "sessions_spanning_splits",
                "sessions_off_plan",
                "normalization_is_train_only",
                "label_errors",
                "privileged_labels_gated",
                "action_coverage",
                "branch_pair_rgb_distinct_at_16",
                "audit",
                "integrity",
                "look",
            )
        },
        "root_seconds": {
            "min": min(r["seconds"] for r in report["root_statuses"] if "seconds" in r),
            "max": max(r["seconds"] for r in report["root_statuses"] if "seconds" in r),
            "mean": sum(r["seconds"] for r in report["root_statuses"] if "seconds" in r)
            / max(1, sum("seconds" in r for r in report["root_statuses"])),
        },
        "readability": summarize_gate(report.get("readability")),
    }
    if dataset is not None:
        manifest = Path(dataset) / "meta" / "jepa_manifest.json"
        out["dataset_manifest_file_sha256"] = sha256(manifest)
        out["dataset_manifest_matches_report"] = (
            out["dataset_manifest_file_sha256"] == report["dataset_manifest_sha256"]
        )
        provenance = json.loads(manifest.read_text())["provenance"]
        out["dataset_provenance"] = {
            k: provenance[k]
            for k in (
                "source_revision",
                "tracked_tree_dirty",
                "collector_sha256",
                "readability_gate_sha256",
                "look_corpus_module_sha256",
                "protocol_sha256",
                "manifest_sha256",
                "runtime",
                "workers",
                "max_seconds",
                "worker_seconds",
            )
            if k in provenance
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    out = summarize(report, args.report, args.dataset)
    args.output.write_text(json.dumps(out, indent=1, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: out.get(k) for k in ("status", "outcome")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
