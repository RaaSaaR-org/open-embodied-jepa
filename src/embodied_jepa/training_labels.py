"""Per-episode training-time label sidecars, physically separated from model inputs.

A corpus that carries sidecars keeps them under ``<dataset>/labels/<episode_id>.npz``.
They are never part of the canonical ``Episode``/``SequenceBatch`` (cameras,
proprioception, actions, timestamps), so no model, planner or evaluator sees
them through the dataset API.

Keys are grouped by prefix, and each group declares who may read it:

- ``robot__``: derived from robot kinematics only (for example the hand-crop
  window). Permitted as model-side metadata.
- ``collector__``: internals of the privileged scripted collector (its phase,
  base action, injected perturbation). The collector read simulator truth at
  reset, so these are training/analysis labels only.
- ``privileged__``: simulator truth (apple, plate and palm positions, their
  offsets, hand contact, task stages). Training-time auxiliary targets or
  analysis only; never a model input, planner input or planning cost.

Reading ``collector__`` or ``privileged__`` requires an explicit
``acknowledge_privileged_training_labels=True``. Each sidecar's SHA-256 is
recorded in the owning episode's metadata, which the dataset manifest hashes.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np

from embodied_jepa.contracts import ContractError

SCHEMA_VERSION = "apple_training_labels_v1"
DIRECTORY = "labels"
GROUPS = ("robot", "collector", "privileged")
PRIVILEGED_GROUPS = frozenset({"collector", "privileged"})
USE_POLICY = {
    "robot": "robot-kinematics-derived; may accompany model inputs",
    "collector": "privileged scripted-collector internals; training/analysis labels only",
    "privileged": (
        "simulator truth; training-time auxiliary targets or analysis only; "
        "never a model input, planner input or planning cost"
    ),
}


def _check_key(key):
    group, _, name = key.partition("__")
    if group not in GROUPS or not name:
        raise ContractError(f"label key {key!r} must be prefixed with one of {GROUPS}")
    return group


def encode(labels):
    """Serialize a label mapping to deterministic npz bytes; returns (bytes, sha256)."""
    if not labels:
        raise ContractError("a label sidecar needs at least one array")
    arrays = {}
    for key in sorted(labels):
        _check_key(key)
        value = np.asarray(labels[key])
        if value.dtype == object or (value.dtype.kind == "f" and not np.isfinite(value).all()):
            raise ContractError(f"label {key} must be a finite numeric/boolean array")
        arrays[key] = value
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    payload = stream.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def write(root, episode_id, labels):
    payload, _ = encode(labels)
    return write_encoded(root, episode_id, payload)


def write_encoded(root, episode_id, payload):
    """Write already-encoded sidecar bytes verbatim (used when assembling shards)."""
    digest = hashlib.sha256(payload).hexdigest()
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        for key in data.files:
            _check_key(key)
    relative = f"{DIRECTORY}/{episode_id}.npz"
    path = Path(root) / relative
    if path.exists():
        raise FileExistsError(f"refusing to overwrite label sidecar {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return {"path": relative, "sha256": digest, "schema": SCHEMA_VERSION}


def load(
    root,
    reference,
    *,
    groups=("robot",),
    acknowledge_privileged_training_labels=False,
):
    """Load the requested groups of one sidecar after verifying its recorded hash."""
    groups = tuple(groups)
    if not groups or any(group not in GROUPS for group in groups):
        raise ContractError(f"groups must be drawn from {GROUPS}")
    if PRIVILEGED_GROUPS & set(groups) and acknowledge_privileged_training_labels is not True:
        raise ContractError(
            "collector/privileged labels are training-time auxiliary data; pass "
            "acknowledge_privileged_training_labels=True and never feed them to a model "
            "input, planner or planning cost"
        )
    if reference.get("schema") != SCHEMA_VERSION:
        raise ContractError("unsupported label sidecar schema")
    relative = Path(reference["path"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != DIRECTORY:
        raise ContractError("invalid label sidecar path")
    payload = (Path(root) / relative).read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ContractError(f"label sidecar hash mismatch: {relative}")
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files if _check_key(key) in groups}
