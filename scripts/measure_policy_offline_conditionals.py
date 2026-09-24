"""Offline conditionals and the phase-restricted error for the TASK-057 preregistration.

Generates Tables C, D and E of ``docs/experiments/apple_policy_diagnostics_v1.md`` and the
manifest's ``frozen_offline_conditionals_and_phase_table``. Committed because §9's own standing
practice is that **a quantity existing only in a git-ignored run artifact is not citable**, and
Table A is already held to exactly that standard: a third party must be able to re-derive these
numbers from committed code.

**Why this is a script rather than a change to ``cloning.evaluate_policy``.** That function
returns one *unconditional* per-dimension median and the *predictions'* standard deviation, and
neither can produce what these tables need:

* with val saturation at ~44.5% -- below half -- an unconditional median necessarily sits inside
  the NON-saturated group, so it is mathematically incapable of testing "large error on the
  saturated rows";
* the *target* standard deviation is never computed anywhere in ``cloning.py``, so the word
  "compressed" had no referent until Table D existed.

``cloning.py`` participates in no checkpoint ``implementation_sha256`` (``VisualModel``'s digest
covers ``models/base.py``, the backend module, ``models/readout.py`` and ``readout_labels.py``;
``ClonedPolicy``'s covers ``policy.py`` alone), so editing it would have been *safe* -- but a
separate reader keeps the trained arms' provenance surface untouched, which is the standing
preference while any TASK-056 checkpoint must stay loadable.

Read-only: loads the four checkpoints and the val split, writes one JSON report. It decodes the
val split only; ``world_model_v2.load_split`` refuses test and holdout.

    uv run --no-sync python scripts/measure_policy_offline_conditionals.py \
        --config configs/apple_policy_v1.yaml --output outputs/task057-diagnostics/offline.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROTOCOL = "apple_policy_diagnostics_v1"
TASK = "TASK-057"

#: Collector phase order, from ``scripted.OracleManipulationPolicy.phases``.
PHASES = ("orient", "descend", "close", "lift", "transfer", "lower", "release", "retreat")
#: The four TASK-056 arms, by checkpoint stem and by the label the manifest uses.
ARMS = {
    "a0": "A0_proprio_only",
    "a1": "A1_random_encoder",
    "a2": "A2_bc_frozen_e0",
    "a3": "A3_bc_finetuned_e0",
}
#: A target at or beyond this magnitude is one the scripted expert clipped (scripted.py:85).
TRANSLATION_CLIP = 0.4
TOLERANCE = 1e-6


def _runner():
    """``scripts/evaluate_policy.py`` is not an importable module; load it by path.

    Its ``frozen_flag_for`` is reused rather than reimplemented: inverting that flag once made
    three of four arms unloadable, and a second copy of the rule is a second chance to invert it.
    """
    spec = importlib.util.spec_from_file_location(
        "_evaluate_policy", ROOT / "scripts" / "evaluate_policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_policy(checkpoint, config, store, device):
    """Rebuild an arm exactly as the closed-loop runner rebuilds it, then verify the pairing."""
    import torch

    from embodied_jepa.config import MODELS
    from embodied_jepa.policy import ClonedPolicy, FrozenEncoder, NoEncoder

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    provenance = saved.get("feature_source") or {}
    if provenance.get("kind") == "none":
        source = NoEncoder()
    else:
        model = MODELS.create(
            config.backend,
            state_schema=store.state_schema,
            device=device,
            seed=0,
            config=config.model_settings,
        )
        # frozen=False for the digest probe: FrozenEncoder(frozen=True) mutates the SHARED model
        # (eval() + requires_grad_(False)), so a throwaway frozen instance would freeze the arm
        # that is about to be restored as trainable.
        if (
            provenance.get("model_weights_sha256")
            != FrozenEncoder(model, frozen=False).weights_sha256()
        ):
            model.load(config.checkpoint)
        source = FrozenEncoder(model, frozen=_runner().frozen_flag_for(saved))
    policy = ClonedPolicy(store.state_schema, source, device=device, seed=0)
    policy.load(checkpoint)  # refuses a head paired with the wrong encoder
    return policy, source


def predictions(policy, arrays, rows, features, chunk=512):
    """The arm's predicted free-action components for every sampled row."""
    import torch

    from embodied_jepa.policy import FREE_ACTION_INDICES

    policy.eval()
    out = np.empty((len(rows), len(FREE_ACTION_INDICES)), np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), chunk):
            part = rows[start : start + chunk]
            visual = None
            if features is not None:
                visual = torch.from_numpy(features[start : start + len(part)]).to(
                    policy.device_name
                )
            state = policy.normalized_state(arrays.states[part], arrays.mask[part])
            out[start : start + len(part)] = policy(visual, state).detach().cpu().numpy()
    return out


def conditionals(predicted, targets, phase, names, dz_index):
    """Tables C, D and E for one arm."""
    error = np.abs(predicted - targets)
    dz = targets[:, dz_index]
    saturated = np.abs(dz) >= TRANSLATION_CLIP - TOLERANCE
    positive = dz >= TRANSLATION_CLIP - TOLERANCE
    negative = dz <= -TRANSLATION_CLIP + TOLERANCE
    if not (saturated.any() and positive.any() and negative.any()):
        raise ValueError("the val cohort must contain saturated targets of both signs")
    result = {
        "unconditional_median_error": {
            n: float(np.median(error[:, i])) for i, n in enumerate(names)
        },
        "median_error_on_dz_saturated_rows": {
            n: float(np.median(error[saturated, i])) for i, n in enumerate(names)
        },
        "median_error_on_dz_unsaturated_rows": {
            n: float(np.median(error[~saturated, i])) for i, n in enumerate(names)
        },
        "predicted_std_per_dimension": {
            n: float(predicted[:, i].std()) for i, n in enumerate(names)
        },
        "mean_predicted_dz_on_target_plus_max": float(predicted[positive, dz_index].mean()),
        "mean_predicted_dz_on_target_minus_max": float(predicted[negative, dz_index].mean()),
        "median_predicted_dz_on_target_plus_max": float(np.median(predicted[positive, dz_index])),
        "median_predicted_dz_on_target_minus_max": float(np.median(predicted[negative, dz_index])),
        "by_phase_median_error": {},
    }
    for index, name in enumerate(PHASES):
        mask = phase == index
        if not mask.any():
            continue
        result["by_phase_median_error"][name] = {"rows": int(mask.sum())} | {
            n: float(np.median(error[mask, i])) for i, n in enumerate(names)
        }
    return result


def measure(config_path, *, output, device, workers):
    from embodied_jepa.cloning import load_bc_split, precompute_features
    from embodied_jepa.policy import FREE_ACTION_INDICES, FREE_ACTION_NAMES
    from embodied_jepa.training import source_identity
    from embodied_jepa.world_model_v2 import _open

    output = Path(output)
    if output.exists():
        raise FileExistsError("refusing to overwrite an existing conditionals report")
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    config, store, _settings, cameras = _open(config_path)
    arrays, rows, counts, base = load_bc_split(
        store, "val", cameras, workers=workers, limit=None, acknowledge=True
    )
    targets = base[rows][:, list(FREE_ACTION_INDICES)]
    phase = arrays.phase[rows]
    names = list(FREE_ACTION_NAMES)
    dz_index = names.index("right_dz")
    dz = targets[:, dz_index]

    report = {
        "format_version": 1,
        "protocol": PROTOCOL,
        "task": TASK,
        "split": "val",
        "val_rows": int(len(rows)),
        "surviving_val_roots": counts["surviving_root_episodes"],
        "device": device,
        "source": source_identity(),
        # Reported separately from the corpus-wide 44.886%, which is the TRAIN cohort. Pooling
        # them would blend two populations, and the thresholds are defined on val.
        "val_saturation": {
            "dz_abs_at_max_rate": float((np.abs(dz) >= TRANSLATION_CLIP - TOLERANCE).mean()),
            "dz_at_positive_max_rate": float((dz >= TRANSLATION_CLIP - TOLERANCE).mean()),
            "dz_at_negative_max_rate": float((dz <= -TRANSLATION_CLIP + TOLERANCE).mean()),
            "dz_target_mean": float(dz.mean()),
        },
        # The referent "compressed" lacked: cloning.evaluate_policy reports the PREDICTIONS' std
        # and never the target's, so a ratio could not be formed from its output.
        "target_std_per_dimension": {n: float(targets[:, i].std()) for i, n in enumerate(names)},
        "arms": {},
    }

    for stem, label in ARMS.items():
        checkpoint = ROOT / "checkpoints" / "task056-policy-v1" / f"{stem}.pt"
        policy, source = build_policy(checkpoint, config, store, device)
        features = precompute_features(source, arrays, rows)
        predicted = predictions(policy, arrays, rows, features)
        report["arms"][label] = conditionals(predicted, targets, phase, names, dz_index)

    report["elapsed_seconds"] = time.perf_counter() - started
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "apple_policy_v1.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    report = measure(args.config, output=args.output, device=args.device, workers=args.workers)
    print(
        json.dumps(
            {
                "val_rows": report["val_rows"],
                "val_dz_saturation": report["val_saturation"]["dz_abs_at_max_rate"],
                "target_dz_std": report["target_std_per_dimension"]["right_dz"],
                "elapsed_seconds": round(report["elapsed_seconds"], 1),
            },
            indent=2,
        )
    )
    return 0


# world_model_v2.load_split uses a SPAWN pool, which re-imports __main__ in every worker. Without
# this guard the module body re-executes per worker and the run deadlocks with near-zero CPU --
# which looks exactly like "still decoding" rather than like a failure.
if __name__ == "__main__":
    sys.exit(main())
