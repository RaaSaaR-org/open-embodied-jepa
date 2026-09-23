"""Split the world model v3 palm-apple gate error into encoder and rollout terms.

TASK-052's primary gate G1 is the median palm-apple readout error at h = 8 on the
moving validation windows, computed from a *predicted* latent. That number mixes two
independent sources of error: how well the encoder and readout head localize the apple
at all, and how much the action-conditioned prediction step adds on top. This script
separates them on the identical cohort the gate uses, so a failed gate can be attributed.

For each window it measures, in metres:

- ``encoded_start``   readout of the directly encoded START observation against the truth
                      at the start frame -- the encoder's own error, with no prediction;
- ``encoded_target``  readout of the directly encoded TARGET observation against the truth
                      at the target frame -- what a *perfect* predictor could achieve with
                      this encoder and head;
- ``rollout``         the gate value itself, reproduced here as a cross-check;
- ``rollout_excess``  rollout minus encoded_target, the prediction step's contribution.

This is descriptive: it gates nothing, changes no threshold and reads no new split. It
runs on val, which selection and the gates already used. Test is never opened.

    uv run --no-sync python scripts/decompose_wm_v3_readout.py \
        --config configs/apple_wm_v3_lewm_onboard.yaml \
        --checkpoint checkpoints/task052-wm-v3/leworldmodel_onboard.pt \
        --device mps --output outputs/task052-decomposition/leworldmodel_onboard.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from embodied_jepa.contracts import ContractError
from embodied_jepa.world_model_v2 import (
    GATE_HORIZON,
    HORIZONS,
    MOVING_THRESHOLD_M,
    _open,
    _sha256,
    load_split,
    predicted_readouts,
    readout_rows,
    window_starts,
)
from embodied_jepa.world_model_v2 import (
    _window_actions as window_actions,
)


def _median(values):
    return float(np.median(values)) if len(values) else None


def decompose(model, arrays, windows, state_schema, horizon=GATE_HORIZON, per_window=None):
    """Encoder/rollout split of the palm-apple error on the gate's own moving cohort.

    ``per_window`` is an optional dict that collects the per-window error vectors and the
    moving mask, so a caller can bootstrap paired differences between two arms over the
    identical windows.
    """
    starts = arrays.offsets[windows[:, 0]] + windows[:, 1]
    targets = starts + horizon
    true = arrays.targets
    dropped = true["apple_dropped"][:, 0] > 0.5
    valid = ~dropped[starts] & ~dropped[targets]
    pma_start = true["palm_minus_apple"][starts]
    pma_target = true["palm_minus_apple"][targets]
    displacement = np.linalg.norm(pma_target - pma_start, axis=1)
    moving = valid & (displacement >= MOVING_THRESHOLD_M)

    actions = window_actions(arrays, windows, horizon)
    rollout = np.linalg.norm(
        predicted_readouts(model, arrays, starts, actions, state_schema)["palm_minus_apple"][
            :, horizon - 1
        ]
        - pma_target,
        axis=1,
    )
    at_start = readout_rows(model, arrays, starts, state_schema)["palm_minus_apple"]
    at_target = readout_rows(model, arrays, targets, state_schema)["palm_minus_apple"]
    encoded_start = np.linalg.norm(at_start - pma_start, axis=1)
    encoded_target = np.linalg.norm(at_target - pma_target, axis=1)
    persistence = np.linalg.norm(at_start - pma_target, axis=1)

    if per_window is not None:
        per_window.update(
            moving=moving,
            rollout=rollout,
            encoded_start=encoded_start,
            encoded_target=encoded_target,
        )
    rollout_median = _median(rollout[moving])
    encoded_target_median = _median(encoded_target[moving])
    return {
        "horizon": horizon,
        "moving_threshold_m": MOVING_THRESHOLD_M,
        "valid_windows": int(valid.sum()),
        "moving_windows": int(moving.sum()),
        "true_displacement_median_m": _median(displacement[moving]),
        "encoded_start_median_m": _median(encoded_start[moving]),
        "encoded_target_median_m": encoded_target_median,
        "rollout_median_m": rollout_median,
        "persistence_median_m": _median(persistence[moving]),
        "rollout_excess_median_m": rollout_median - encoded_target_median,
        "encoder_share_of_rollout": encoded_target_median / rollout_median,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--dump-errors",
        help="also write the per-window error vectors and moving mask to this .npz, so "
        "two arms can be compared by a paired bootstrap over the identical windows",
    )
    parser.add_argument(
        "--acknowledge-privileged-training-labels",
        action="store_true",
        help="this measurement scores against privileged labels; say so explicitly",
    )
    args = parser.parse_args()
    if not args.acknowledge_privileged_training_labels:
        raise ContractError("this measurement scores against privileged labels; acknowledge it")

    import torch

    from embodied_jepa.config import MODELS
    from embodied_jepa.training import source_identity

    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite a decomposition report")

    config, store, settings, cameras = _open(args.config)
    checkpoint = Path(args.checkpoint)
    envelope = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = MODELS.create(
        config.backend,
        state_schema=store.state_schema,
        device=args.device,
        seed=envelope["seed"],
        config=settings,
        metadata=envelope["metadata"],
    )
    model.load(checkpoint)
    arrays = load_split(
        store,
        "val",
        cameras,
        workers=args.workers,
        acknowledge_privileged_training_labels=True,
    )
    windows = window_starts(arrays, max(HORIZONS), stride=4)
    per_window = {} if args.dump_errors else None
    result = {
        "format_version": 1,
        "measurement": "palm_apple_encoder_vs_rollout_decomposition",
        "split": "val",
        "test_episodes_decoded": 0,
        "backend": config.backend,
        "cameras": list(cameras),
        "config": str(Path(args.config).name),
        "device": args.device,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_step": envelope["metadata"].get("runner_state", {}).get("step"),
        "model_implementation_sha256": model.implementation_sha256,
        "analysis_source": source_identity(),
        "provenance": envelope["metadata"],
        "val_episodes": len(arrays.episode_ids),
        "val_observations": int(len(arrays.states)),
        "decomposition": decompose(
            model, arrays, windows, store.state_schema, per_window=per_window
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if per_window is not None:
        errors = Path(args.dump_errors)
        if errors.exists():
            raise FileExistsError("refusing to overwrite a per-window error dump")
        errors.parent.mkdir(parents=True, exist_ok=True)
        np.savez(errors, **per_window)
    print(json.dumps(result["decomposition"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
