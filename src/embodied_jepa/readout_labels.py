"""Readout TARGETS built from gated privileged label sidecars (NumPy only).

These arrays are training targets for the shared readout heads and scoring
references for offline evaluation. They are simulator truth: never a model input,
planner input or planning cost. Loading them requires the same explicit
acknowledgement as ``embodied_jepa.training_labels``.
"""

from __future__ import annotations

import numpy as np

from embodied_jepa import training_labels
from embodied_jepa.contracts import ContractError

READOUT_LABELS_VERSION = "apple_readout_targets_v1"
# Apple counts as held when the hand touches it and it has risen this far above its
# resting height at the reset (the scorer's grasp stage uses 5 cm; 2 cm marks lift-off).
HELD_RISE_M = 0.02
TARGET_NAMES = (
    "palm_minus_apple",
    "apple_height",
    "apple_minus_plate",
    "hand_contact",
    "apple_held",
)


def load_privileged(root, row, *, acknowledge_privileged_training_labels=False):
    """Hash-verified privileged + collector labels of one manifest episode row."""
    reference = row["metadata"].get("training_labels")
    if reference is None:
        raise ContractError(f"episode {row['episode_id']} has no training-label sidecar")
    return training_labels.load(
        root,
        reference,
        groups=("privileged", "collector"),
        acknowledge_privileged_training_labels=acknowledge_privileged_training_labels,
    )


def targets(labels, rest_apple_z):
    """Per-observation readout targets ``{name: float32 [T+1, width]}``.

    ``rest_apple_z`` is the apple's resting height at the reset (its first root frame).
    """
    if not np.isfinite(rest_apple_z):
        raise ContractError("rest apple height must be finite")
    palm_minus_apple = np.asarray(labels["privileged__palm_minus_apple_world"], np.float32)
    apple = np.asarray(labels["privileged__apple_position_world"], np.float32)
    plate = np.asarray(labels["privileged__plate_position_world"], np.float32)
    contact = np.asarray(labels["privileged__hand_contact"], bool)
    count = len(apple)
    if any(len(x) != count for x in (palm_minus_apple, plate, contact)) or count < 2:
        raise ContractError("label arrays disagree in length")
    height = apple[:, 2:3] - np.float32(rest_apple_z)
    held = contact & (height[:, 0] >= HELD_RISE_M)
    result = {
        "palm_minus_apple": palm_minus_apple,
        "apple_height": height.astype(np.float32),
        "apple_minus_plate": (apple - plate).astype(np.float32),
        "hand_contact": contact[:, None].astype(np.float32),
        "apple_held": held[:, None].astype(np.float32),
    }
    for name, value in result.items():
        if not np.isfinite(value).all():
            raise ContractError(f"non-finite readout target {name}")
    return result
