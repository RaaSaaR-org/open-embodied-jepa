"""TASK-073: the latent critic -- a world model's prediction read out as a cost, and nothing else.

``LatentCritic.costs`` takes the raw pooled token latent of the current frame (6144-d, the
TASK-066 latent), K candidate 16-command chunks and the target step, rolls every chunk through the
world model in one batch, reads the predicted apple-minus-plate offset with the ridge readout
``R_off`` and returns ``J_k = || R_off(z_hat_k) - o*(t + 16) ||`` in cm. It returns costs only:
the selector never sees a latent (architecture rule: planners never inspect latents).

The world model is any object with ``predict_features(start [N, D], actions [N, T, 14]) ->
[N, T, D]`` (``models.frozen_tokens`` via ``FrozenEncoderMixin``), so the LeWM and native backends
swap by configuration. Three action-blind stand-ins share the same interface for the controls:
``CopyLast`` (the prediction is the start latent) and the no-action model N (a trained model
that is always given zero actions). With any of them every candidate gets the same cost and the
tie rule returns the incumbent.

``RidgeReadout`` is a linear ridge readout on train-standardised inputs, with its lambda chosen
by grouped inner cross-validation; it fits in the primal (``D`` small enough) or the dual (``N``
small, as for the full 98 304-d DINOv2 tokens). Its state, and the critic's, are hashed so that a
run can pin them beside the world-model checkpoint. NumPy at import; torch is never needed here.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from embodied_jepa.contracts import ContractError

STD_FLOOR = 1e-6


def grouped_folds(groups, folds: int, seed: int) -> np.ndarray:
    """Fold index per row, whole groups per fold, from ``default_rng(seed)`` over sorted groups."""
    labels = np.unique(np.asarray(groups))
    order = np.random.default_rng(seed).permutation(len(labels))
    fold_of_label = {labels[j]: i % folds for i, j in enumerate(order)}
    return np.asarray([fold_of_label[g] for g in np.asarray(groups)], int)


class RidgeReadout:
    """``y = ((x - mean) / std) @ w + b``. The ridge is ``lambda_rel`` times the mean diagonal of
    the linear kernel ``z z^T`` (TASK-063's kernel-ridge convention) in both forms, so the primal
    and the dual give the same readout."""

    def __init__(self, mean, std, weights, bias, lam_rel, selection=None):
        self.mean = np.asarray(mean, np.float64)
        self.std = np.asarray(std, np.float64)
        self.weights = np.asarray(weights, np.float64)
        self.bias = np.asarray(bias, np.float64)
        self.lam_rel = float(lam_rel)
        self.selection = selection or {}
        if (
            self.mean.shape != self.std.shape
            or self.weights.shape[0] != self.mean.shape[0]
            or self.bias.shape != (self.weights.shape[1],)
        ):
            raise ContractError("ridge readout shapes disagree")

    @staticmethod
    def _solve(x, y, lam_rel, *, dual):
        """Weights for standardised ``x`` [N, D] and centred ``y`` [N, k]."""
        if dual:
            gram = x @ x.T
            scale = float(np.mean(np.diag(gram)))
            alpha = np.linalg.solve(gram + lam_rel * scale * np.eye(len(gram)), y)
            return x.T @ alpha
        cov = x.T @ x
        scale = float(np.trace(cov)) / len(x)  # the mean kernel diagonal, as in the dual form
        return np.linalg.solve(cov + lam_rel * scale * np.eye(len(cov)), x.T @ y)

    BLOCK = 8192

    @classmethod
    def _kernel(cls, x):
        """Column moments (float64) and ``z z^T`` of the standardised rows, in column blocks."""
        n, d = x.shape
        mean, sq = np.zeros(d), np.zeros(d)
        for lo in range(0, d, cls.BLOCK):
            b = np.asarray(x[:, lo : lo + cls.BLOCK], np.float64)
            if not np.isfinite(b).all():
                raise ContractError("a ridge readout needs finite, aligned rows")
            mean[lo : lo + cls.BLOCK] = b.mean(0)
            sq[lo : lo + cls.BLOCK] = b.std(0)
        std = np.maximum(sq, STD_FLOOR)
        gram = np.zeros((n, n))
        for lo in range(0, d, cls.BLOCK):
            b = np.asarray(x[:, lo : lo + cls.BLOCK], np.float64) - mean[lo : lo + cls.BLOCK]
            b /= std[lo : lo + cls.BLOCK]
            gram += b @ b.T
        return mean, std, gram

    @classmethod
    def _weights(cls, x, mean, std, alpha):
        d = x.shape[1]
        w = np.zeros((d, alpha.shape[1]))
        for lo in range(0, d, cls.BLOCK):
            b = np.asarray(x[:, lo : lo + cls.BLOCK], np.float64) - mean[lo : lo + cls.BLOCK]
            b /= std[lo : lo + cls.BLOCK]
            w[lo : lo + cls.BLOCK] = b.T @ alpha
        return w

    @classmethod
    def fit(cls, x, y, groups, *, lambdas, folds: int, seed: int, dual: bool = False):
        """Standardise on all rows, pick lambda by grouped inner CV (MSE; ties to the smaller
        lambda), refit on all rows.

        The dual form works from the linear kernel ``z z^T`` alone, built in column blocks
        without materialising the standardised copy (memory; TASK-073 protocol §15): each fold's
        fit and held-out predictions are the kernel's sub-blocks, which is the same estimator as
        solving on ``z[fit]`` and predicting ``z[held] @ w``."""
        y = np.asarray(y, np.float64)
        if y.ndim == 1:
            y = y[:, None]
        if len(x) != len(y) or len(x) != len(groups):
            raise ContractError("a ridge readout needs finite, aligned rows")
        fold = grouped_folds(groups, folds, seed)
        sse = {float(lam): 0.0 for lam in lambdas}
        if dual:
            x = np.asarray(x)
            mean, std, gram = cls._kernel(x)
            for k in range(folds):
                fit, held = fold != k, fold == k
                if not held.any() or not fit.any():
                    raise ContractError("an inner fold is empty")
                y_mean = y[fit].mean(0)
                g_ff, g_hf = gram[np.ix_(fit, fit)], gram[np.ix_(held, fit)]
                scale = float(np.mean(np.diag(g_ff)))
                for lam in lambdas:
                    alpha = np.linalg.solve(
                        g_ff + float(lam) * scale * np.eye(len(g_ff)), y[fit] - y_mean
                    )
                    sse[float(lam)] += float(((g_hf @ alpha + y_mean - y[held]) ** 2).sum())
            best = min(sse, key=lambda lam: (sse[lam], lam))
            y_mean = y.mean(0)
            scale = float(np.mean(np.diag(gram)))
            alpha = np.linalg.solve(gram + best * scale * np.eye(len(gram)), y - y_mean)
            w = cls._weights(x, mean, std, alpha)
        else:
            x = np.asarray(x, np.float64)
            if not np.isfinite(x).all():
                raise ContractError("a ridge readout needs finite, aligned rows")
            mean = x.mean(0)
            std = np.maximum(x.std(0), STD_FLOOR)
            z = (x - mean) / std
            for k in range(folds):
                fit, held = fold != k, fold == k
                if not held.any() or not fit.any():
                    raise ContractError("an inner fold is empty")
                y_mean = y[fit].mean(0)
                for lam in lambdas:
                    w = cls._solve(z[fit], y[fit] - y_mean, float(lam), dual=False)
                    sse[float(lam)] += float(((z[held] @ w + y_mean - y[held]) ** 2).sum())
            best = min(sse, key=lambda lam: (sse[lam], lam))
            y_mean = y.mean(0)
            w = cls._solve(z, y - y_mean, best, dual=False)
        selection = {
            "lam_rel": best,
            "inner_mse": sse[best] / len(y),
            "folds": folds,
            "fold_seed": seed,
            "rows": int(len(y)),
            "groups": int(len(np.unique(np.asarray(groups)))),
            "form": "dual" if dual else "primal",
        }
        return cls(mean, std, w, y_mean, best, selection)

    def predict(self, x) -> np.ndarray:
        x = np.asarray(x, np.float64)
        squeeze = x.ndim == 1
        out = (
            (x.reshape(-1, self.mean.shape[0]) - self.mean) / self.std
        ) @ self.weights + self.bias
        if not np.isfinite(out).all():
            raise ContractError("G-finite: a readout output is not finite")
        return out[0] if squeeze else out

    def state(self) -> dict:
        return {
            "mean": self.mean.astype(np.float64),
            "std": self.std.astype(np.float64),
            "weights": self.weights.astype(np.float64),
            "bias": self.bias.astype(np.float64),
            "lam_rel": np.asarray(self.lam_rel),
        }

    @classmethod
    def from_state(cls, state: dict, selection=None):
        return cls(
            state["mean"],
            state["std"],
            state["weights"],
            state["bias"],
            float(state["lam_rel"]),
            selection,
        )

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for key, value in sorted(self.state().items()):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(value, np.float64).tobytes())
        return digest.hexdigest()


class CopyLast:
    """The action-blind persistence stand-in: every predicted latent is the start latent."""

    def predict_features(self, features, actions):
        features = np.asarray(features, np.float32)
        actions = np.asarray(actions)
        return np.repeat(features[:, None], actions.shape[1], axis=1)


class ZeroActions:
    """Wraps the no-action model N: it only ever sees zero actions (as in its training)."""

    def __init__(self, model):
        self.model = model

    def predict_features(self, features, actions):
        return self.model.predict_features(features, np.zeros_like(np.asarray(actions, np.float32)))


class LatentCritic:
    """Costs of candidate chunks from a world model's predicted latents (cm); never latents."""

    def __init__(self, model, r_off: RidgeReadout, o_star_cm: dict, horizon: int = 16):
        self.model = model
        self.r_off = r_off
        self.o_star = {int(k): np.asarray(v, np.float64) for k, v in o_star_cm.items()}
        self.horizon = int(horizon)

    def predicted_offsets_cm(self, start, chunks) -> np.ndarray:
        """R_off of the predicted latent after ``horizon`` commands, per candidate [K, 2] (cm)."""
        start = np.asarray(start, np.float32).reshape(1, -1)
        chunks = np.asarray(chunks, np.float32)
        if chunks.ndim != 3 or chunks.shape[1] != self.horizon or chunks.shape[2] != 14:
            raise ContractError(f"chunks must be [K, {self.horizon}, 14]")
        starts = np.repeat(start, len(chunks), axis=0)
        predicted = self.model.predict_features(starts, np.ascontiguousarray(chunks))
        if predicted.shape[:2] != (len(chunks), self.horizon):
            raise ContractError("the world model returned the wrong rollout shape")
        return 100.0 * self.r_off.predict(predicted[:, -1])

    def costs(self, start, chunks, target_step: int) -> np.ndarray:
        target = self.o_star[int(target_step)]
        offsets = self.predicted_offsets_cm(start, chunks)
        return np.linalg.norm(offsets - target, axis=1)

    def sha256(self) -> str:
        """The readout and the target table (the model is pinned by its own checkpoint hash)."""
        digest = hashlib.sha256(self.r_off.sha256().encode())
        digest.update(
            json.dumps({str(k): v.tolist() for k, v in sorted(self.o_star.items())}).encode()
        )
        digest.update(str(self.horizon).encode())
        return digest.hexdigest()
