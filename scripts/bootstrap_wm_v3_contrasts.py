"""Paired bootstrap of the world model v3 one-factor contrasts on the gate's own cohort.

TASK-052's design has exactly three pairs of arms that differ by one declared factor
(A vs B the camera set, C vs A the readout shaping, D vs B the patch size). This script
puts a cohort-sampling interval on each difference of medians, using the per-window error
vectors that ``scripts/decompose_wm_v3_readout.py --dump-errors`` writes.

**Design, stated in full because it is the contested part.** Both arms of a contrast are
resampled on the *identical* window indices (a paired bootstrap), which is legitimate only
because every arm was evaluated on the same cohort -- the script asserts the moving masks
are byte-identical before it resamples. Each of ``--resamples`` draws takes ``n`` window
indices with replacement from the ``n`` moving windows via ``np.random.default_rng(seed)``,
recomputes ``median(b) - median(a)``, and the interval is the 2.5/97.5 percentiles of that
distribution.

**What this interval is not.** The moving windows are stride-4 windows drawn from only 80
validation episodes, so consecutive windows share most of their transitions and are far
from independent. Resampling them iid therefore **understates** the variance: read the
result as a lower bound on the uncertainty, not as a confidence interval. An
episode-clustered resample would be wider. And with one seed per arm none of this says
anything about run-to-run variation -- it quantifies cohort sampling for two fixed
checkpoints, nothing more.

    uv run --no-sync python scripts/bootstrap_wm_v3_contrasts.py \
        --errors outputs/task052-decomposition \
        --output outputs/task052-decomposition/paired-bootstrap.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# The one-factor contrasts, as ``(label, baseline_arm, changed_arm)``. The reported
# difference is always ``median(changed) - median(baseline)``.
CONTRASTS = (
    ("A_minus_B_second_camera", "B", "A"),
    ("C_minus_A_drop_readout_shaping", "A", "C"),
    ("D_minus_B_patch_14_to_8", "B", "D"),
)
ARMS = {
    "A": "leworldmodel",
    "B": "leworldmodel_onboard",
    "C": "leworldmodel_v2_readout",
    "D": "leworldmodel_fine",
}
FIELDS = ("rollout", "encoded_target")


def paired_intervals(errors, seed, resamples):
    """Difference-of-medians and a paired percentile interval for every contrast."""
    from embodied_jepa.contracts import ContractError

    mask = errors["A"]["moving"]
    for arm, data in errors.items():
        if not np.array_equal(data["moving"], mask):
            raise ContractError(f"arm {arm} was scored on a different moving cohort")
    count = int(mask.sum())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, count, size=(resamples, count))

    result = {}
    for label, baseline, changed in CONTRASTS:
        for field in FIELDS:
            a = np.asarray(errors[baseline][field][mask], np.float64)
            b = np.asarray(errors[changed][field][mask], np.float64)
            spread = np.median(b[draws], axis=1) - np.median(a[draws], axis=1)
            low, high = np.percentile(spread, [2.5, 97.5])
            result[f"{label}_{field}"] = {
                "baseline_arm": baseline,
                "changed_arm": changed,
                "field": field,
                "point_m": float(np.median(b) - np.median(a)),
                "ci95_m": [float(low), float(high)],
                "crosses_zero": bool(low < 0 < high),
            }
    return count, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--errors",
        required=True,
        help="directory holding <arm>-errors.npz from decompose_wm_v3_readout.py",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20520)
    parser.add_argument("--resamples", type=int, default=20000)
    args = parser.parse_args()

    from embodied_jepa.training import source_identity
    from embodied_jepa.world_model_v2 import _sha256

    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite a bootstrap report")
    root = Path(args.errors)
    errors = {arm: np.load(root / f"{stem}-errors.npz") for arm, stem in ARMS.items()}
    count, contrasts = paired_intervals(errors, args.seed, args.resamples)

    result = {
        "format_version": 1,
        "measurement": "paired_bootstrap_of_one_factor_contrasts",
        "split": "val",
        "test_episodes_decoded": 0,
        "method": (
            "difference of medians between two arms on the identical moving windows; "
            "each resample draws n window indices with replacement and is applied to "
            "both arms (paired); interval is the 2.5/97.5 percentiles"
        ),
        "limitation": (
            "the moving windows are stride-4 windows from 80 val episodes and are "
            "strongly correlated within an episode, so resampling them iid understates "
            "the variance; read these as a lower bound on the uncertainty. With one seed "
            "per arm they say nothing about run-to-run variation."
        ),
        "inputs": {arm: f"{root.name}/{stem}-errors.npz" for arm, stem in ARMS.items()},
        "moving_windows": count,
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
        low, high = entry["ci95_m"]
        print(
            f"{label:45s} {entry['point_m'] * 100:+7.3f} cm "
            f"[{low * 100:+7.3f}, {high * 100:+7.3f}]"
            f"{'  crosses zero' if entry['crosses_zero'] else ''}"
        )


if __name__ == "__main__":
    main()
