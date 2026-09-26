"""A frozen external image encoder in front of an unchanged backend predictor (TASK-065).

``frozen_encoder_model("leworldmodel")`` (or ``"native_jepa"``) returns the backend class with
``FrozenEncoderMixin`` in front of it. Its image pathway is the pinned, frozen DINOv2 ViT-S/14 CLS
feature of ``pretrained_encoder`` (TASK-063's input handling and read-out point), standardised
per dimension by train-only moments; the backend's own predictor, loss and optimizer run on it
unchanged.

**No existing model file changes.** ``models/base.py``, ``models/lewm.py``,
``models/native.py``, ``models/readout.py`` and ``readout_labels.py`` are untouched, so every
earlier model, and every checkpoint whose ``implementation_sha256`` covers them (the TASK-054 E0
checkpoint among them), is bit-for-bit what it was. The option is off unless a run selects this
class, and the class refuses to run without it.

* **The latent is the feature, standardised** (``fit_frozen_feature_normalization``, std floored
  at 1e-3, fitted once on training frames before any update). There is no projector, so a
  prediction maps back to the raw feature exactly (``predict_features``), and an external probe
  fitted on the raw feature reads it.
* **The encoder never trains.** It is a plain attribute, not a submodule: never in
  ``parameters()``, the optimizer, ``.to(device)`` or the state dict. Its weights digest is in the
  model's metadata, which the checkpoint envelope saves and ``load`` compares, so a checkpoint is
  refused under a different frozen encoder. ``implementation_sha256`` also covers this file, the
  backend's own file and ``pretrained_encoder.py``.
* **Features are always CPU float32**, whatever the model device, so a feature cache built with
  ``frozen_features`` is the feature the model computes from frames. ``train_step_features``
  therefore makes exactly the update ``train_step`` makes from the frames (tested bit for bit).
* The backend's own image encoder (and LeWM's projector) are still built, so the predictor's
  initialisation stream is the backend's; they never receive a gradient.

Torch and transformers are imported with the backend, never by ``import embodied_jepa``.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import numpy as np
import torch

from embodied_jepa.contracts import (
    EE_DELTA_GRASP_V0,
    ContractError,
    SequenceBatch,
    validate_actions,
)

# name -> (module, loader, feature width). The loader returns a CPU float32 eval module after
# checking its pinned files; ``module.features(model, frames)["cls"]`` reads it.
FROZEN_ENCODERS = {
    "dinov2_small_cls": ("embodied_jepa.pretrained_encoder", "load_pretrained", 384),
}
BACKENDS = {
    "leworldmodel": ("embodied_jepa.models.lewm", "LeWM"),
    "native_jepa": ("embodied_jepa.models.native", "NativeJEPA"),
}
FEATURE_FLOOR_STD = 1e-3
DIGEST_KEY = "frozen_encoder_weights_digest"


class FrozenEncoderMixin:
    """Placed before a ``VisualModel`` backend in the MRO; see the module docstring."""

    def __init__(self, state_schema, device="cpu", seed=0, config=None, metadata=None):
        merged = self.defaults | dict(config or {})
        name = merged["frozen_encoder"]
        if name not in FROZEN_ENCODERS:
            raise ContractError(f"frozen_encoder must be one of {sorted(FROZEN_ENCODERS)}")
        width = FROZEN_ENCODERS[name][2]
        if merged["latent_dim"] != width:
            raise ContractError(f"frozen_encoder {name!r} requires latent_dim {width}")
        if merged["state_fusion"] or merged["readout_heads"]:
            raise ContractError("frozen_encoder is not combined with state_fusion/readout_heads")
        if merged["cameras"] is not None:
            raise ContractError("frozen_encoder reads the single configured camera")
        module = self._load_frozen_module(name)
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        from embodied_jepa import pretrained_encoder

        digest = pretrained_encoder.weights_digest(module)
        metadata = dict(metadata or {})
        if metadata.get(DIGEST_KEY, digest) != digest:
            raise ContractError("declared frozen encoder digest differs from the loaded encoder")
        metadata[DIGEST_KEY] = digest
        super().__init__(state_schema, device, seed, config, metadata)
        # Plain attributes, set after the backend is built: never submodules or parameters.
        object.__setattr__(self, "_frozen_module", module.eval())
        object.__setattr__(self, "_injected_features", None)
        self.register_buffer("frozen_feature_mean", torch.zeros(width, device=self.device_name))
        self.register_buffer("frozen_feature_scale", torch.ones(width, device=self.device_name))
        self.register_buffer(
            "frozen_feature_normalization_fitted", torch.tensor(False, device=self.device_name)
        )

    def _load_frozen_module(self, name):
        module_name, loader, _ = FROZEN_ENCODERS[name]
        return getattr(importlib.import_module(module_name), loader)()

    @property
    def frozen_encoder_digest(self):
        return self.metadata[DIGEST_KEY]

    @property
    def implementation_sha256(self):
        digest = hashlib.sha256(super().implementation_sha256.encode())
        backend = next(
            c for c in type(self).__mro__ if c.__module__ in {m for m, _ in BACKENDS.values()}
        )
        name = self.config["frozen_encoder"]
        for path in (
            Path(__file__),
            Path(inspect.getfile(backend)),
            Path(importlib.import_module(FROZEN_ENCODERS[name][0]).__file__),
        ):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    # ----- features and their train-only normalisation ---------------------------------------
    @torch.no_grad()
    def frozen_features(self, frames):
        """Raw frozen features ``float32 [N, width]`` of uint8 frames ``[N, H, W, 3]`` (CPU)."""
        from embodied_jepa import pretrained_encoder

        # A writable copy: canonical batches hold read-only snapshots torch must not alias.
        frames = np.array(frames, copy=True)
        return pretrained_encoder.features(self._frozen_module, frames)["cls"].astype(np.float32)

    @torch.no_grad()
    def fit_frozen_feature_normalization(self, mean, std, *, training_episode_ids):
        """Freeze train-split feature moments once, before any update (std floored)."""
        if bool(self.frozen_feature_normalization_fitted) or self.updates:
            raise ContractError("frozen-feature normalization is frozen; create a new model")
        ids = tuple(training_episode_ids)
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(x, str) for x in ids):
            raise ContractError("training_episode_ids must be unique episode names")
        mean = np.asarray(mean, np.float64)
        std = np.asarray(std, np.float64)
        shape = (self.config["latent_dim"],)
        if mean.shape != shape or std.shape != shape:
            raise ContractError("feature moments must match the frozen feature width")
        if not (np.isfinite(mean).all() and np.isfinite(std).all()) or (std < 0).any():
            raise ContractError("feature moments must be finite with nonnegative std")
        scale = np.maximum(std, FEATURE_FLOOR_STD)
        self.frozen_feature_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.frozen_feature_scale.copy_(torch.as_tensor(scale, dtype=torch.float32))
        self.frozen_feature_normalization_fitted.fill_(True)
        self.metadata["frozen_feature_normalization_episodes_sha256"] = hashlib.sha256(
            json.dumps(sorted(ids)).encode()
        ).hexdigest()

    def _normalize(self, raw):
        if not bool(self.frozen_feature_normalization_fitted):
            raise ContractError("fit train-only frozen-feature normalization first")
        values = torch.as_tensor(np.array(raw, dtype=np.float32, copy=True))
        if values.shape[-1] != self.config["latent_dim"] or not torch.isfinite(values).all():
            raise ContractError("frozen features must be finite with the frozen feature width")
        values = values.to(self.device_name)
        return (values - self.frozen_feature_mean) / self.frozen_feature_scale

    # ----- the image pathway ------------------------------------------------------------------
    def image_features(self, images, *, sequence=False, target=False):
        """The frozen feature, standardised: the same latent online and as target."""
        name = self.config["camera"]
        if name not in images:
            raise ContractError(f"missing configured camera {name!r}")
        array = images[name]
        rank = 5 if sequence else 4
        if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or array.ndim != rank:
            raise ContractError("model camera requires uint8 channels-last RGB")
        prefix = array.shape[:-3]
        injected = self._injected_features
        if injected is not None:
            if tuple(injected.shape[:-1]) != tuple(prefix):
                raise ContractError("injected features do not match the batch")
            return injected.reshape(-1, injected.shape[-1]), prefix
        raw = self.frozen_features(array.reshape(-1, *array.shape[-3:]))
        return self._normalize(raw), prefix

    @contextmanager
    def _inject(self, normalized):
        object.__setattr__(self, "_injected_features", normalized)
        try:
            yield
        finally:
            object.__setattr__(self, "_injected_features", None)

    # ----- training from cached features ------------------------------------------------------
    def train_step_features(self, features, actions):
        """One update from cached raw features ``[B, T+1, width]`` and normalised float32 actions
        ``[B, T, 14]``: the backend's own ``train_step``, with the features standing in for the
        frames' ``frozen_features`` (a frozen encoder makes those a pure function of the frames).
        """
        features = np.asarray(features)
        actions = np.asarray(actions)
        if (
            features.ndim != 3
            or actions.ndim != 3
            or features.shape[0] != actions.shape[0]
            or features.shape[1] != actions.shape[1] + 1
        ):
            raise ContractError("features [B,T+1,D] and actions [B,T,14] must agree")
        batch_size, horizon = actions.shape[:2]
        normalized = self._normalize(features)
        schema = self.state_schema
        placeholder = SequenceBatch(
            # 1x1 placeholder frames: the image pathway reads the injected features instead.
            observations={
                self.config["camera"]: np.zeros((batch_size, horizon + 1, 1, 1, 3), np.uint8)
            },
            robot_states=np.zeros((batch_size, horizon + 1, schema.dimension), np.float32),
            state_mask=np.ones((batch_size, horizon + 1, schema.dimension), np.bool_),
            actions=actions,
            timestamps=np.tile(np.arange(horizon + 1, dtype=np.float64), (batch_size, 1)),
            terminated=np.zeros((batch_size, horizon), np.bool_),
            episode_ids=tuple(f"features-{i}" for i in range(batch_size)),
            state_schema=schema,
        )
        with self._inject(normalized):
            return self.train_step(placeholder)

    # ----- prediction in the raw feature space ------------------------------------------------
    @torch.no_grad()
    def predict_features(self, features, actions):
        """Recursive rollout from raw start features ``[N, width]`` under normalised float32
        ``actions [N, T, 14]`` -> raw predicted features ``float32 [N, T, width]``.

        An evaluation path for probes fitted on the public frozen feature. Planners still only
        ever receive ``VisualLatent`` from ``encode``/``predict``."""
        self.eval()
        actions = np.asarray(actions)
        if actions.ndim != 3 or actions.shape[-1] != EE_DELTA_GRASP_V0.dimension:
            raise ContractError("actions must be [N, T, 14]")
        if actions.shape[1] > self.config["max_horizon"]:
            raise ContractError("rollout exceeds max_horizon")
        validate_actions(actions[:, None])
        initial = self._normalize(features)
        if initial.ndim != 2 or initial.shape[0] != actions.shape[0]:
            raise ContractError("features [N,D] and actions [N,T,14] must agree")
        chunk = self.config["candidate_chunk_size"]
        outputs = []
        with self.rng_scope():
            for start in range(0, len(actions), chunk):
                part = torch.from_numpy(np.array(actions[start : start + chunk], copy=True))
                predicted = self.rollout(initial[start : start + chunk], part.to(self.device_name))
                raw = predicted * self.frozen_feature_scale + self.frozen_feature_mean
                outputs.append(raw.cpu())
        result = torch.cat(outputs).numpy().astype(np.float32)
        if not np.isfinite(result).all():
            raise ContractError("model produced non-finite predicted features")
        return result


@cache
def frozen_encoder_model(backend: str):
    """The backend class with the frozen encoder in front; ``frozen_encoder`` is required."""
    if backend not in BACKENDS:
        raise ContractError(f"frozen_encoder supports the backends {sorted(BACKENDS)}")
    module_name, class_name = BACKENDS[backend]
    base = getattr(importlib.import_module(module_name), class_name)
    return type(
        f"FrozenEncoder{class_name}",
        (FrozenEncoderMixin, base),
        {"__module__": __name__, "defaults": base.defaults | {"frozen_encoder": None}},
    )
