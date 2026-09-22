"""World model v3 (TASK-052): two cameras, motion-weighted readouts, ranking gates.

TASK-050's three arms all failed their preregistered offline gates. The measured
defect was readout precision under motion, and the most planning-relevant number
was descriptive rather than gated: the predicted approach cost ordered candidate
actions at Spearman rho ~= 0.16.

This module reuses the ``world_model_v2`` runner (which now carries the model's
whole camera list) and adds what v3 changes:

- ``candidate_ranking_metrics``: the metric a CEM actually needs. Among the
  sibling candidate actions that leave one shared pre-grasp state, how well does
  the predicted approach cost order them, and how much true cost is lost by
  taking the predicted best? This replaces v2's absolute calibration gate G6,
  which stays as a descriptive number.
- ``evaluate_gates``: the v3 gate set, from
  ``benchmarks/manifests/apple-world-model-v3.json``.

Everything else - the train/val/test discipline, the privileged-label policy, the
opaque latents, the persistence/shuffled/sibling controls - is v2's and unchanged.
Nothing here is a closed-loop or learned-control result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from embodied_jepa.world_model_v2 import (
    APPROACH_OFFSET_M,
    GATE_HORIZON,
    SIBLING_GATE_HORIZON,
    _median,
    evaluate,
    predicted_readouts,
    sibling_groups,
    spearman,
    train,
)

PROTOCOL = "apple_world_model_v3"
RANKING_HORIZONS = (8, 16, 32)
RANKING_GATE_HORIZON = 16
# A candidate set is only rankable when its true costs actually differ. Measured from
# val LABELS alone before any v3 model existed (disclosed in the protocol): at h=16,
# 18 of 20 val sibling groups clear 5 mm, against 10 of 20 at h=8.
RANKING_MIN_SPREAD_M = 0.005
RANKING_MIN_CANDIDATES = 3


def _approach_cost(offsets):
    return np.linalg.norm(np.asarray(offsets, np.float64) - APPROACH_OFFSET_M, axis=-1)


def _normalized_ranks(values):
    """Within-group ranks mapped to [0,1]; ties share the average rank."""
    values = np.asarray(values, np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), np.float64)
    start = 0
    ordered = values[order]
    while start < len(values):
        end = start
        while end + 1 < len(values) and ordered[end + 1] == ordered[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2
        start = end + 1
    return ranks / (len(values) - 1)


def candidate_ranking_metrics(model, arrays, state_schema):
    """How well the predicted approach cost orders sibling candidates from one state.

    Each val root contributes one pre-grasp state and the executed action sequences of
    its continuation and its branches. Every candidate is rolled out from the *same*
    encoded observation, so the comparison is exactly the choice a CEM makes: one
    state, several action sequences, one ranking. The persistence control cannot be
    reported here because it is constant across the candidates of a state - it has no
    ranking at all, which is why an absolute calibration gate could not test this.

    The shuffled control ranks candidate ``i`` by the prediction made with candidate
    ``i+1``'s actions, which reuses the same rollouts and costs nothing extra.
    """
    groups = sibling_groups(arrays)
    longest = max(RANKING_HORIZONS)
    true = arrays.targets["palm_minus_apple"]
    dropped = arrays.targets["apple_dropped"][:, 0] > 0.5
    records = {str(h): [] for h in RANKING_HORIZONS}
    for root in sorted(groups):
        members = groups[root]
        if len(members) < RANKING_MIN_CANDIDATES:
            continue
        anchor = next((m for m in members if m[0] == root), members[0])
        rows = np.array([anchor[1]])
        predictions = {}
        for name, start, stop, _ in members:
            steps = min(longest, stop - 1 - start)
            if steps < min(RANKING_HORIZONS):
                continue
            actions = arrays.actions[start : start + steps][None]
            readouts = predicted_readouts(model, arrays, rows, actions, state_schema)
            predictions[name] = (start, steps, readouts)
        names = sorted(predictions)
        for h in RANKING_HORIZONS:
            usable = [
                name
                for name in names
                if predictions[name][1] >= h and not dropped[predictions[name][0] + h]
            ]
            if len(usable) < RANKING_MIN_CANDIDATES:
                continue
            true_cost = _approach_cost([true[predictions[n][0] + h] for n in usable])
            predicted_cost = _approach_cost(
                [predictions[n][2]["palm_minus_apple"][0, h - 1] for n in usable]
            )
            # Same rollouts, shifted by one candidate: sibling i judged by sibling i+1.
            shuffled_cost = np.roll(predicted_cost, 1)
            records[str(h)].append(
                {
                    "root": root,
                    "candidates": len(usable),
                    "true": true_cost,
                    "predicted": predicted_cost,
                    "shuffled": shuffled_cost,
                }
            )
    return {
        "minimum_true_cost_spread_m": RANKING_MIN_SPREAD_M,
        "minimum_candidates": RANKING_MIN_CANDIDATES,
        **{h: _summarize_ranking(entries) for h, entries in records.items()},
    }


def _summarize_ranking(entries):
    """Pooled within-state rank correlation, top-1 regret and the controls."""
    rankable = [e for e in entries if e["true"].max() - e["true"].min() >= RANKING_MIN_SPREAD_M]
    result = {
        "groups_evaluated": len(entries),
        "groups_ranked": len(rankable),
        "candidates_median": _median([e["candidates"] for e in entries]),
        "true_cost_spread_median_m": _median(
            [float(e["true"].max() - e["true"].min()) for e in entries]
        ),
    }
    for key in ("predicted", "shuffled"):
        pooled_model, pooled_true, per_group, regret, normalized = [], [], [], [], []
        for entry in rankable:
            pooled_model.extend(_normalized_ranks(entry[key]))
            pooled_true.extend(_normalized_ranks(entry["true"]))
            per_group.append(spearman(entry[key], entry["true"]))
            chosen = float(entry["true"][int(np.argmin(entry[key]))])
            best, worst = float(entry["true"].min()), float(entry["true"].max())
            regret.append(chosen - best)
            normalized.append((chosen - best) / (worst - best))
        prefix = "" if key == "predicted" else "shuffled_"
        finite = [value for value in per_group if value is not None]
        result |= {
            f"{prefix}within_state_spearman_pooled": spearman(pooled_model, pooled_true),
            f"{prefix}within_state_spearman_mean": (float(np.mean(finite)) if finite else None),
            f"{prefix}within_state_spearman_groups": len(finite),
            f"{prefix}top1_regret_median_m": _median(regret),
            f"{prefix}top1_regret_mean_m": float(np.mean(regret)) if regret else None,
            f"{prefix}top1_regret_normalized_median": _median(normalized),
        }
    # Label-only baseline: the expected regret of choosing a candidate uniformly at
    # random. It uses no model output, so it is a property of the val cohort.
    random_regret = [float(e["true"].mean() - e["true"].min()) for e in rankable]
    random_normalized = [
        float((e["true"].mean() - e["true"].min()) / (e["true"].max() - e["true"].min()))
        for e in rankable
    ]
    result |= {
        "random_choice_regret_median_m": _median(random_regret),
        "random_choice_regret_normalized_median": _median(random_normalized),
    }
    return result


def ranking_metrics(model, arrays, state_schema):
    return {"ranking": candidate_ranking_metrics(model, arrays, state_schema)}


def evaluate_gates(metrics, gates):
    """Apply the preregistered v3 thresholds (``gates`` from the frozen manifest).

    Same shape as v2's, with v2's absolute cost-calibration gate G6 replaced by the
    within-state ranking gates G6a/G6b. v2's calibration number is still reported under
    ``metrics.windows``, so the two protocols stay comparable.
    """
    w = metrics["windows"][str(GATE_HORIZON)]
    s = metrics["siblings"][str(SIBLING_GATE_HORIZON)]
    r = metrics["ranking"][str(RANKING_GATE_HORIZON)]
    c = metrics["collapse"]
    enough = r["groups_ranked"] >= gates["G6_minimum_ranked_groups"]

    def ratio(a, b):
        return None if a is None or b is None or b <= 0 else a / b

    checks = {
        "G1_palm_apple_moving_h8": (
            w["palm_apple_moving_median_m"],
            "<=",
            gates["G1_palm_apple_moving_h8_median_m"],
        ),
        "G2a_vs_persistence_h8": (
            ratio(w["palm_apple_moving_median_m"], w["palm_apple_moving_persistence_median_m"]),
            "<=",
            gates["G2_max_ratio_to_control"],
        ),
        "G2b_vs_shuffled_actions_h8": (
            ratio(w["palm_apple_moving_median_m"], w["palm_apple_moving_shuffled_median_m"]),
            "<=",
            gates["G2_max_ratio_to_control"],
        ),
        "G3_apple_plate_h8": (w["apple_plate_median_m"], "<=", gates["G3_apple_plate_h8_median_m"]),
        "G4_apple_height_grasp_h8": (
            w["apple_height_grasp_median_abs_m"],
            "<=",
            gates["G4_apple_height_grasp_h8_median_abs_m"],
        ),
        "G5_held_auroc_lift_h8": (
            w["held_lift_auroc"]
            if min(w["held_lift_positives"], w["held_lift_negatives"])
            >= gates["G5_minimum_per_class"]
            else None,
            ">=",
            gates["G5_held_auroc_min"],
        ),
        "G6a_within_state_ranking_spearman_h16": (
            r["within_state_spearman_pooled"] if enough else None,
            ">=",
            gates["G6a_within_state_spearman_min"],
        ),
        "G6b_top1_regret_h16": (
            r["top1_regret_median_m"] if enough else None,
            "<=",
            gates["G6b_top1_regret_max_m"],
        ),
        "G7a_sibling_own_beats_swapped_h16": (
            s["own_beats_swapped_fraction"]
            if s["qualifying_ordered_pairs"] >= gates["G7_minimum_pairs"]
            else None,
            ">=",
            gates["G7a_own_beats_swapped_min"],
        ),
        "G7b_sibling_divergence_spearman_h16": (
            s["divergence_spearman"],
            ">=",
            gates["G7b_divergence_spearman_min"],
        ),
        "G8a_collapsed_fraction": (
            c["collapsed_fraction"],
            "<=",
            gates["G8_max_collapsed_fraction"],
        ),
        "G8b_effective_rank": (c["effective_rank"], ">=", gates["G8_min_effective_rank"]),
        "G8c_latent_std_mean": (c["latent_std_mean"], ">=", gates["G8_min_latent_std_mean"]),
    }
    result = {}
    for name, (value, op, threshold) in checks.items():
        passed = value is not None and (value <= threshold if op == "<=" else value >= threshold)
        result[name] = {"value": value, "op": op, "threshold": threshold, "passed": bool(passed)}
    result["all_passed"] = all(item["passed"] for item in result.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--protocol-manifest", type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "mps"), default="mps")
        command.add_argument("--workers", type=int, default=8)
        command.add_argument("--limit-episodes", type=int, help="smoke subsets only")
        command.add_argument("--acknowledge-privileged-training-labels", action="store_true")
    train_parser = commands.choices["train"]
    train_parser.add_argument("--output", type=Path)
    train_parser.add_argument(
        "--require-clean", action="store_true", help="refuse a dirty or unversioned checkout"
    )
    train_parser.add_argument("--smoke-steps", type=int, help="smoke only: override steps")
    train_parser.add_argument(
        "--smoke-max-seconds", type=float, help="smoke only: override max_seconds"
    )
    train_parser.add_argument(
        "--smoke-validation-every", type=int, help="smoke only: override validation_every"
    )
    evaluate_parser = commands.choices["evaluate"]
    evaluate_parser.add_argument("--checkpoint", type=Path)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.protocol_manifest.read_text())
    frozen = manifest["frozen"]
    common = dict(
        device=args.device,
        workers=args.workers,
        limit_episodes=args.limit_episodes,
        protocol=PROTOCOL,
        acknowledge_privileged_training_labels=args.acknowledge_privileged_training_labels,
    )
    if args.command == "train":
        budget = dict(frozen["training"])
        for key in ("steps", "max_seconds", "validation_every"):
            override = getattr(args, f"smoke_{key}")
            if override is not None:
                budget[key] = override
        report = train(
            args.config,
            output=args.output,
            steps=budget["steps"],
            batch_size=budget["batch_size"],
            horizon=budget["horizon"],
            seed=budget["seed"],
            max_seconds=budget["max_seconds"],
            validation_every=budget["validation_every"],
            selection_count=budget["selection_windows"],
            final_lr_fraction=budget["final_lr_fraction"],
            eligibility=frozen["selection_eligibility"],
            require_clean=args.require_clean,
            **common,
        )
        summary = {k: report.get(k) for k in ("status", "completed_steps", "best_step")}
    else:
        report = evaluate(
            args.config,
            checkpoint=args.checkpoint,
            output=args.output,
            gates=frozen["gates"],
            gate_function=evaluate_gates,
            extra_metrics=ranking_metrics,
            **common,
        )
        summary = {"all_passed": report["gates"]["all_passed"]} | {
            k: v["passed"] for k, v in report["gates"].items() if k != "all_passed"
        }
    print(json.dumps(summary, indent=2))
    return 0 if report.get("status", "completed") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
