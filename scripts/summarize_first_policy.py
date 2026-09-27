"""TASK-067: write the results manifest from a gated run's ``report.json`` (read-only).

    uv run --no-sync python scripts/summarize_first_policy.py \\
        --report outputs/task067-first-policy/run-1/report.json \\
        --render outputs/task067-scratch/prerun-go-render-1/report.json \\
        --output benchmarks/manifests/apple-first-policy-v1-results.json

Every number in the results manifest is copied from the two reports; nothing is recomputed
except the sha256 of each report and the C0 bar rule, which is re-applied from the recorded
counts with ``first_policy.c0_bars`` as a consistency check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import first_policy as fp  # noqa: E402


def summarize(report_path: Path, render_path: Path) -> dict:
    report_bytes = report_path.read_bytes()
    render_bytes = render_path.read_bytes()
    report = json.loads(report_bytes)
    render = json.loads(render_bytes)
    c0 = report["stages"]["C0"]["successes_of_32"]
    apple = {a: c0[f"apple_{a}"] for a in fp.C0_APPLE_LEVELS_CM}
    plate = {p: c0[f"plate_{p}"] for p in fp.C0_PLATE_LEVELS_CM}
    recomputed = fp.c0_bars(c0["reference"], apple, plate)
    if recomputed != report["stages"]["C0"]["bars"]:
        raise SystemExit("the recorded C0 bars do not follow from the recorded counts")
    return {
        "protocol": fp.PROTOCOL,
        "task": fp.TASK,
        "run": "run-1",
        "outcome": report["outcome"],
        "void_reason": report.get("void_reason"),
        "clause_fires": bool(report.get("clause_fires", False)),
        "report": {
            "path": str(report_path),
            "sha256": hashlib.sha256(report_bytes).hexdigest(),
            "revision": report["revision"],
            "tracked_tree_dirty": report["tracked_tree_dirty"],
            "started_utc": report["started_utc"],
            "total_seconds": report["total_seconds"],
            "environment": report["environment"],
            "encoder_digests": report["encoder_digests"],
            "test_split_decoded": report["test_split_decoded"],
            "decoded_episodes": report["decoded_episodes"],
            "non_finite_fields": report["non_finite_fields"],
            "pinned_files_at_preflight": len(report["pinned_hashes_at_preflight"]),
            "pinned_files_at_end": len(report.get("pinned_hashes_at_end") or {}),
            "stages_recorded": sorted(report["stages"]),
        },
        "data": {
            "dataset_manifest_sha256": report["data"]["dataset_manifest_sha256"],
            "train_roots": len(report["data"]["train_roots"]),
            "val_roots": len(report["data"]["val_roots"]),
            "seeds_simulated": {
                role: [min(v), max(v), len(v)]
                for role, v in report["data"]["seeds"].items()
                if role in ("perception_train", "perception_heldout", "calibration_C0")
            },
        },
        "readout_selection": report["stages"]["readouts"],
        "C0": {
            "successes_of_32": c0,
            "bars": report["stages"]["C0"]["bars"],
            "bars_recomputed_from_counts": recomputed,
            "seconds": report["stages"]["C0"]["seconds"],
        },
        "not_reached": [
            "S0-P (its held-out errors were computed in memory and not written; none was seen)",
            "S0-D1",
            "BC-0 and every training",
            "DAgger",
            "M1 on cohort D (no D reset was simulated)",
        ],
        "prerun_render_check": {
            "path": str(render_path),
            "sha256": hashlib.sha256(render_bytes).hexdigest(),
            "verdict": render["verdict"],
            "revision": render["revision"],
            "workers": render["workers"],
            "seeds_rendered_on_two_or_more_workers": render[
                "seeds_rendered_on_two_or_more_workers"
            ],
            "negative_control": render["negative_control"],
        },
        "learned_apple_to_plate_successes": 0,
        "exemption_spent": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    result = summarize(args.report, args.render)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("outcome", "clause_fires")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
