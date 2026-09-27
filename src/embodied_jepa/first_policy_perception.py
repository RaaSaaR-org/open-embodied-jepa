"""TASK-067 perception: DINOv2 tokens of the post-look frame -> apple and plate xy.

The readout is TASK-063's probe (``info_ceiling``: linear and RBF kernel ridge, family and lambda
by inner CV), unchanged. NumPy here; the encoder (torch, transformers) is imported by
``featurise`` only.
"""

from __future__ import annotations

import numpy as np

from embodied_jepa import first_policy as fp
from embodied_jepa import info_ceiling as ic
from embodied_jepa.contracts import ContractError


def featurise(encoder, frames) -> np.ndarray:
    """uint8 [N, 112, 112, 3] -> float64 tokens [N, 98304], one frame at a time (batch size 1).

    Batch size 1 everywhere, offline and live, so TASK-065's float32 batch-size effect cannot
    separate the two paths (protocol §7.3)."""
    from embodied_jepa import pretrained_encoder as pe

    frames = np.asarray(frames)
    rows = [
        pe.features(encoder, frames[i : i + 1], batch=1)[fp.READOUT_FEATURE]
        for i in range(len(frames))
    ]
    out = np.concatenate(rows)
    if not np.isfinite(out).all():
        raise fp.GuardError("G-finite: a DINOv2 feature is not finite")
    return out


def fold_of(n: int) -> np.ndarray:
    """10 folds over n rows, ``default_rng(6701)`` (protocol §5.1)."""
    fold = np.arange(n) % fp.READOUT_FOLDS
    np.random.default_rng(fp.READOUT_FOLD_SEED).shuffle(fold)
    return fold


class XYReadout:
    """A frozen kernel-ridge readout of [apple x, apple y, plate x, plate y] (world, m)."""

    def __init__(self, features, targets):
        features = np.asarray(features, np.float64)
        targets = np.asarray(targets, np.float64)
        if targets.shape != (len(features), 4) or not np.isfinite(targets).all():
            raise ContractError("the readout needs finite [N, 4] xy targets")
        self.features = features
        g, diag = ic.gram(features)
        index = np.arange(len(features))
        self.selection = ic.select(g, diag, index, targets, outer=fp.READOUT_FOLDS)
        self.readout = ic.fit_readout(
            self.selection["family"], self.selection["lam_rel"], g, diag, index, targets
        )

    def predict(self, features) -> np.ndarray:
        g_rt, norms = ic.cross_gram(np.asarray(features, np.float64), self.features)
        out = self.readout.predict(g_rt, norms)
        if not np.isfinite(out).all():
            raise fp.GuardError("G-finite: a readout estimate is not finite")
        return out


def cross_fitted(features, targets) -> tuple[np.ndarray, list[dict]]:
    """Out-of-fold estimates over the readout's own training rows (10 folds, rng 6701)."""
    targets = np.asarray(targets, np.float64)
    g, diag = ic.gram(np.asarray(features, np.float64))
    predictions, _readouts, selections = ic.nested_cv(g, diag, targets, fold_of(len(targets)))
    return predictions, selections


def errors_cm(estimates, truth) -> tuple[np.ndarray, np.ndarray]:
    """Per-row apple and plate xy errors in cm."""
    estimates = np.asarray(estimates, np.float64).reshape(-1, 4)
    truth = np.asarray(truth, np.float64).reshape(-1, 4)
    apple = 100.0 * np.linalg.norm(estimates[:, :2] - truth[:, :2], axis=1)
    plate = 100.0 * np.linalg.norm(estimates[:, 2:] - truth[:, 2:], axis=1)
    return apple, plate


def featurise_cls(encoder, frames) -> np.ndarray:
    """The CLS read-out at batch size 1 (B-replay's retrieval key only)."""
    from embodied_jepa import pretrained_encoder as pe

    frames = np.asarray(frames)
    return np.concatenate(
        [pe.features(encoder, frames[i : i + 1], batch=1)["cls"] for i in range(len(frames))]
    )
