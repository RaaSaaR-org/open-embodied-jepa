"""Paired bootstrap of the world model v4 one-factor contrasts, iid and episode-clustered.

Successor to ``scripts/bootstrap_wm_v3_contrasts.py``. It keeps that script's design
and conventions so the v3 and v4 numbers are comparable, and adds the resampling design
TASK-052 said it was missing.

**Two designs, both reported.**

``iid`` is v3's: each of ``--resamples`` draws takes ``n`` window indices with
replacement from the ``n`` moving windows and applies the *same* indices to both arms of
a contrast. The v3 report recorded this as a lower bound on the uncertainty, because the
moving windows are stride-4 windows from 80 validation episodes and consecutive windows
share most of their transitions.

``cluster`` is the design v3 could not compute: each draw takes ``E`` **episodes** with
replacement from the ``E`` validation episodes that contribute a moving window, and
pools every moving window of the drawn episodes. Windows within an episode stay
together, so the within-episode correlation is respected and the interval is wider. This
needs each window's episode index, which ``world_model_v4 evaluate --dump-errors``
records and the v3 dump did not.

Both are paired: both arms of a contrast are resampled on the identical draw, which is
legitimate only because every arm was evaluated on the identical cohort -- the script
asserts the moving masks and episode indices are equal before it resamples.

**What neither interval is.** With one seed per arm, neither says anything about
run-to-run variation: both quantify cohort sampling for two fixed checkpoints.

    uv run --no-sync python scripts/bootstrap_wm_v4_contrasts.py \
        --errors outputs/task054-wm-v4 \
        --output outputs/task054-wm-v4/paired-bootstrap.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# The one-factor contrasts, as ``(label, baseline_arm, changed_arm)``. Every arm differs
# from the E0 baseline by exactly one declared option, so E0 is the baseline throughout
# and the reported difference is always ``median(changed) - median(baseline)``.
CONTRASTS = (
    ("E1_minus_E0_action_chunk", "E0", "E1"),
    ("E2_minus_E0_multistep_tail_weight", "E0", "E2"),
    ("E3_minus_E0_predictor_step_embedding", "E0", "E3"),
)
ARMS = {
    "E0": "leworldmodel_baseline",
    "E1": "leworldmodel_chunk",
    "E2": "leworldmodel_tail",
    "E3": "leworldmodel_step",
}
FIELDS = ("rollout", "encoded_target", "rollout_excess")


def _field(data, name, mask):
    """One per-window error vector on the moving cohort, in float64."""
    if name == "rollout_excess":
        # A per-window difference, whose *median* is not the difference of medians the
        # decomposition reports. It is bootstrapped as its own quantity and labelled as
        # such: see ``rollout_excess_is_per_window`` in the report.
        values = np.asarray(data["rollout"], np.float64) - np.asarray(
            data["encoded_target"], np.float64
        )
    else:
        values = np.asarray(data[name], np.float64)
    return values[mask]


def _iid_draws(rng, count, resamples):
    return rng.integers(0, count, size=(resamples, count))


def _cluster_draws(rng, episodes, resamples):
    """Per resample, the pooled window positions of ``E`` episodes drawn with replacement."""
    unique = np.unique(episodes)
    members = [np.flatnonzero(episodes == episode) for episode in unique]
    picks = rng.integers(0, len(unique), size=(resamples, len(unique)))
    return [np.concatenate([members[i] for i in row]) for row in picks], len(unique)


def paired_intervals(errors, seed, resamples):
    """Difference of medians and paired percentile intervals under both designs."""
    from embodied_jepa.contracts import ContractError

    reference = errors[CONTRASTS[0][1]]
    mask = np.asarray(reference["moving"], bool)
    episodes = np.asarray(reference["episode"], np.int64)[mask]
    for arm, data in errors.items():
        if not np.array_equal(np.asarray(data["moving"], bool), mask):
            raise ContractError(f"arm {arm} was scored on a different moving cohort")
        if not np.array_equal(np.asarray(data["episode"], np.int64)[mask], episodes):
            raise ContractError(f"arm {arm} was scored on differently ordered windows")
    count = int(mask.sum())

    designs = {}
    rng = np.random.default_rng(seed)
    designs["iid"] = ([row for row in _iid_draws(rng, count, resamples)], count)
    rng = np.random.default_rng(seed)
    cluster_rows, clusters = _cluster_draws(rng, episodes, resamples)
    designs["cluster"] = (cluster_rows, clusters)

    result = {}
    for label, baseline, changed in CONTRASTS:
        for field in FIELDS:
            a = _field(errors[baseline], field, mask)
            b = _field(errors[changed], field, mask)
            entry = {
                "baseline_arm": baseline,
                "changed_arm": changed,
                "field": field,
                "point_m": float(np.median(b) - np.median(a)),
            }
            for design, (rows, _) in designs.items():
                spread = np.array(
                    [np.median(b[row]) - np.median(a[row]) for row in rows], np.float64
                )
                low, high = np.percentile(spread, [2.5, 97.5])
                entry[design] = {
                    "ci95_m": [float(low), float(high)],
                    "crosses_zero": bool(low < 0 < high),
                }
            result[f"{label}_{field}"] = entry
    return count, designs["cluster"][1], result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--errors",
        required=True,
        help="directory holding <arm>-errors.npz from world_model_v4 evaluate --dump-errors",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20540)
    parser.add_argument("--resamples", type=int, default=20000)
    args = parser.parse_args()

    from embodied_jepa.training import source_identity
    from embodied_jepa.world_model_v2 import _sha256

    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite a bootstrap report")
    root = Path(args.errors)
    errors = {arm: np.load(root / f"{stem}-errors.npz") for arm, stem in ARMS.items()}
    count, clusters, contrasts = paired_intervals(errors, args.seed, args.resamples)

    result = {
        "format_version": 1,
        "measurement": "paired_bootstrap_of_one_factor_contrasts",
        "split": "val",
        "test_episodes_decoded": 0,
        "method": (
            "difference of medians between two arms on the identical moving windows, "
            "under two resampling designs. 'iid' draws n window indices with "
            "replacement (the TASK-052 design, kept for comparability). 'cluster' "
            "draws E validation episodes with replacement and pools all their moving "
            "windows, so correlated windows stay together. Both are applied to both "
            "arms of a contrast (paired); each interval is the 2.5/97.5 percentiles."
        ),
        "limitation": (
            "the iid design resamples stride-4 windows from 80 val episodes as if they "
            "were independent and understates the variance; read it as a lower bound "
            "and prefer the clustered interval. With one seed per arm neither design "
            "says anything about run-to-run variation: both quantify cohort sampling "
            "for two fixed checkpoints."
        ),
        "rollout_excess_is_per_window": (
            "the rollout_excess rows bootstrap the median of the per-window difference "
            "(rollout - encoded_target), which is not the difference of the two medians "
            "that gate G9 reads; they bound the sampling of a related quantity, not of "
            "the gate value itself"
        ),
        "inputs": {arm: f"{root.name}/{stem}-errors.npz" for arm, stem in ARMS.items()},
        "moving_windows": count,
        "episodes_with_moving_windows": clusters,
        "bootstrap_resamples": args.resamples,
        "seed": args.seed,
        # source_identity() hashes src/embodied_jepa only, so it does not cover this
        # script; hash it separately or the report pins none of the code that wrote it.
        "analysis_source": source_identity() | {"script_sha256": _sha256(Path(__file__))},
        "contrasts": contrasts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for label, entry in contrasts.items():
        line = f"{label:48s} {entry['point_m'] * 100:+7.3f} cm"
        for design in ("iid", "cluster"):
            low, high = entry[design]["ci95_m"]
            line += f"  {design}[{low * 100:+6.3f},{high * 100:+6.3f}]"
            if entry[design]["crosses_zero"]:
                line += "*"
        print(line)


if __name__ == "__main__":
    main()
