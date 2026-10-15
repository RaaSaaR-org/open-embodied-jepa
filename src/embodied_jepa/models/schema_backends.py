"""Native and LeWM backends sized by a declared action schema (SO-101 S0, TASK-096).

``NativeJEPA`` and ``LeWM`` take 14-D ``ee_delta_grasp_v0`` actions (``nn.Linear(dim + 14, ...)``
and ``Embedder(input_dim=14, ...)``), and ``models/base.py`` declares, validates, saves and loads
that schema. Those files feed the implementation hash every native/LeWM checkpoint since TASK-072
enforces, ``models/lewm.py`` also feeds TASK-087's G hash, and ``contracts.py``,
``planning.py`` and ``models/lewm.py`` are byte-pinned by the frozen G1 protocols
(TASK-091–TASK-094). **None of them changes**: this module adds subclasses that differ only in
where the action width comes from.

* ``SchemaNativeJEPA`` and ``SchemaLeWM`` take ``action_schema`` (a registered
  ``ActionSchema``) and build the backend's modules with that width, in the backend's own order
  and RNG stream. Their ``__init__`` mirrors the backend's; a test holds each to the backend's
  exact initial weights when given ``ee_delta_grasp_v0``.
* ``capabilities``, ``predict``, ``save`` and ``load`` are the base versions with the declared
  schema in place of ``EE_DELTA_GRASP_V0``. Training (``train_step``) and the rollout are the
  backend's, unchanged; they read the width from the batch.
* ``implementation_sha256`` covers ``models/base.py``, this file, the readout files and the
  backend's own file, so a checkpoint from these classes never loads into the pinned backends
  (or the reverse).
* ``action_chunk`` > 1 is refused for a schema other than ``ee_delta_grasp_v0``, because
  ``models/base.py`` sizes the chunk encoder by the 14-D schema.

``SchemaLeWM`` is built on first access (``models/lewm.py`` imports ``transformers``), so the
native subclass needs only the learning extra. ``create_model`` keeps ``ee_delta_grasp_v0`` on
the pinned ``NativeJEPA``/``LeWM`` through ``MODELS``, so a G1 run never reaches these classes;
another schema gets the subclass for the same backend key, so the backend swap stays a one-line
config change. Torch is imported with this module, never by ``import embodied_jepa``.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
from dataclasses import asdict
from functools import cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

from embodied_jepa import devices
from embodied_jepa.action_schemas import action_schema_from_record, schema_record
from embodied_jepa.contracts import (
    EE_DELTA_GRASP_V0,
    ActionSchema,
    Capabilities,
    ContractError,
    validate_actions,
)
from embodied_jepa.models import readout as readout_module
from embodied_jepa.models.base import VisualLatent, VisualModel
from embodied_jepa.models.native import NativeJEPA


def _checked_schema(action_schema, config) -> ActionSchema:
    if not isinstance(action_schema, ActionSchema):
        raise ContractError("action_schema must be an ActionSchema")
    action_schema_from_record(schema_record(action_schema))  # registered schemas only
    chunk = dict(config or {}).get("action_chunk", 1)
    if action_schema != EE_DELTA_GRASP_V0 and chunk != 1:
        raise ContractError(
            "action_chunk > 1 is sized by ee_delta_grasp_v0 in models/base.py; "
            f"it is not available for {action_schema.version}"
        )
    return action_schema


class SchemaActionMixin:
    """Placed before the backend in the MRO: the schema-dependent parts of ``VisualModel``."""

    action_schema: ActionSchema
    pinned_backend: type  # the backend class this one mirrors

    @property
    def action_dim(self) -> int:
        return self.action_schema.dimension

    @property
    def capabilities(self):
        return Capabilities(
            self.action_schema,
            self.state_schema,
            self.config["max_horizon"],
            supported_devices=devices.SUPPORTED_DEVICES,
            readouts=self.declared_readouts,
        )

    @torch.no_grad()
    def predict(self, z, actions):
        # VisualModel.predict with the declared schema in validate_actions.
        self.eval()
        validate_actions(actions, schema=self.action_schema)
        initial = self.check_latent(z, 2)
        batch, candidates, horizon, dimension = actions.shape
        if batch != initial.shape[0] or horizon > self.config["max_horizon"]:
            raise ContractError("action batch/horizon is incompatible with encoded observation")
        chunk = self.config["candidate_chunk_size"]
        outputs = []
        for start in range(0, candidates, chunk):
            part = torch.from_numpy(np.array(actions[:, start : start + chunk], copy=True))
            part = part.to(self.device_name)
            count = part.shape[1]
            state = initial[:, None].expand(-1, count, -1).reshape(batch * count, -1)
            action = part.reshape(batch * count, horizon, dimension)
            outputs.append(self.rollout(state, action).reshape(batch, count, horizon, -1))
        result = torch.cat(outputs, dim=1)
        if not torch.isfinite(result).all():
            raise ContractError("model produced non-finite predicted latents")
        return VisualLatent(result, self._owner)

    def save(self, path):
        # VisualModel.save with the declared schema in the envelope.
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "format_version": 1,
            "backend": self.backend,
            "config": self.config,
            "state_schema": asdict(self.state_schema),
            "action_schema": asdict(self.action_schema),
            "metadata": self.metadata,
            "seed": self.seed,
            "updates": self.updates,
            "weights": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng_cpu": self._rng,
            "rng_mps": self._mps_rng,
            "rng_cuda": self._cuda_rng,
            "source_revision": self.source_revision,
            "implementation_sha256": self.implementation_sha256,
            "preprocessing": {
                "pixels": "uint8_rgb",
                "actions": "normalized_-1_1",
                "robot_state": "train_normalized_learned_state_fusion"
                if self.config["state_fusion"]
                else "ignored_visual_baseline",
                "readouts": readout_module.READOUT_VERSION
                if self.config["readout_heads"]
                else None,
            },
        }
        temporary = path.with_name(path.name + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def load(self, path):
        # VisualModel.load with the declared schema in the compatibility check.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        expected = {
            "format_version": 1,
            "backend": self.backend,
            "config": self.config,
            "state_schema": asdict(self.state_schema),
            "action_schema": asdict(self.action_schema),
            "source_revision": self.source_revision,
            "implementation_sha256": self.implementation_sha256,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ContractError(f"checkpoint {key} is incompatible")
        for key, value in self.metadata.items():
            if checkpoint.get("metadata", {}).get(key) != value:
                raise ContractError(f"checkpoint provenance {key!r} is incompatible")
        self.load_state_dict(checkpoint["weights"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.metadata = checkpoint["metadata"]
        self.seed = checkpoint["seed"]
        self.updates = checkpoint["updates"]
        self._rng = checkpoint["rng_cpu"]
        self._mps_rng = checkpoint["rng_mps"]
        self._cuda_rng = checkpoint.get("rng_cuda")
        self._owner = object()
        self.eval()

    @property
    def implementation_sha256(self):
        # base.py + this file + readout files (VisualModel's digest), then the backend's file.
        digest = hashlib.sha256(super().implementation_sha256.encode())
        path = Path(inspect.getfile(self.pinned_backend))
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
        return digest.hexdigest()


class SchemaNativeJEPA(SchemaActionMixin, NativeJEPA):
    """``NativeJEPA`` whose predictor takes ``action_schema.dimension`` action inputs."""

    pinned_backend = NativeJEPA

    def __init__(
        self,
        state_schema,
        device="cpu",
        seed=0,
        config=None,
        metadata=None,
        *,
        action_schema: ActionSchema,
    ):
        schema = _checked_schema(action_schema, config)
        # Mirrors NativeJEPA.__init__ (models/native.py); only the action width differs.
        VisualModel.__init__(self, state_schema, device, seed, config, metadata)
        self.action_schema = schema
        if not 0 <= self.config["ema_decay"] < 1:
            raise ContractError("ema_decay must lie in [0,1)")
        for name in ("variance_weight", "covariance_weight", "multistep_weight"):
            if not np.isfinite(self.config[name]) or self.config[name] < 0:
                raise ContractError(f"{name} must be nonnegative and finite")
        dim, hidden = self.config["latent_dim"], self.config["hidden_dim"]
        with self.rng_scope():
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 5, stride=2, padding=2),
                nn.SiLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, hidden),
                nn.SiLU(),
                nn.Linear(hidden, dim),
            )
            self.predictor = nn.Sequential(
                nn.Linear(dim + schema.dimension, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, dim),
            )
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.finish_init()


@cache
def _schema_lewm():
    """``SchemaLeWM``, built on first use so this module needs no LeWM extra until then."""
    from transformers import ViTConfig, ViTModel

    from embodied_jepa.models.lewm import LeWM, load_upstream

    class SchemaLeWM(SchemaActionMixin, LeWM):
        """``LeWM`` whose action embedder takes ``action_schema.dimension`` inputs."""

        pinned_backend = LeWM

        def __init__(
            self,
            state_schema,
            device="cpu",
            seed=0,
            config=None,
            metadata=None,
            *,
            action_schema: ActionSchema,
        ):
            schema = _checked_schema(action_schema, config)
            # Mirrors LeWM.__init__ (models/lewm.py); only the action embedder's width differs.
            VisualModel.__init__(self, state_schema, device, seed, config, metadata)
            self.action_schema = schema
            for name in (
                "patch_size",
                "encoder_depth",
                "encoder_heads",
                "predictor_depth",
                "predictor_heads",
                "predictor_head_dim",
                "sigreg_projections",
            ):
                if type(self.config[name]) is not int or self.config[name] < 1:
                    raise ContractError(f"{name} must be a positive integer")
            if (
                self.config["image_size"] % self.config["patch_size"]
                or self.config["latent_dim"] % self.config["encoder_heads"]
            ):
                raise ContractError(
                    "image/patch size and embedding/encoder heads must divide evenly"
                )
            for name in ("sigreg_weight", "multistep_weight"):
                if not np.isfinite(self.config[name]) or self.config[name] < 0:
                    raise ContractError(f"{name} must be nonnegative and finite")
            upstream, components = load_upstream(self.config["source_path"])
            dim, hidden = self.config["latent_dim"], self.config["hidden_dim"]
            with self.rng_scope():
                encoder = ViTModel(
                    ViTConfig(
                        hidden_size=dim,
                        num_hidden_layers=self.config["encoder_depth"],
                        num_attention_heads=self.config["encoder_heads"],
                        intermediate_size=dim * 4,
                        image_size=self.config["image_size"],
                        patch_size=self.config["patch_size"],
                    ),
                    add_pooling_layer=False,
                    use_mask_token=False,
                )
                self.model = upstream.JEPA(
                    encoder=encoder,
                    predictor=components.ARPredictor(
                        num_frames=1,
                        input_dim=dim,
                        hidden_dim=dim,
                        output_dim=dim,
                        depth=self.config["predictor_depth"],
                        heads=self.config["predictor_heads"],
                        dim_head=self.config["predictor_head_dim"],
                        mlp_dim=hidden,
                        dropout=0.0,
                        emb_dropout=0.0,
                    ),
                    action_encoder=components.Embedder(input_dim=schema.dimension, emb_dim=dim),
                    projector=components.MLP(dim, hidden, dim, norm_fn=nn.BatchNorm1d),
                    pred_proj=components.MLP(dim, hidden, dim, norm_fn=nn.BatchNorm1d),
                )
                self.sigreg = components.SIGReg(
                    knots=17, num_proj=self.config["sigreg_projections"]
                )
            self.finish_init()

    SchemaLeWM.__qualname__ = "SchemaLeWM"
    return SchemaLeWM


def __getattr__(name):
    if name == "SchemaLeWM":
        return _schema_lewm()
    raise AttributeError(name)


SCHEMA_BACKENDS = {"native_jepa": lambda: SchemaNativeJEPA, "leworldmodel": _schema_lewm}


def create_model(
    backend: str,
    *,
    state_schema,
    action_schema: ActionSchema,
    device="cpu",
    seed=0,
    config=None,
    metadata=None,
):
    """The pinned backend for ``ee_delta_grasp_v0``; the schema subclass for another schema."""
    if not isinstance(action_schema, ActionSchema):
        raise ContractError("action_schema must be an ActionSchema")
    kwargs = {
        "state_schema": state_schema,
        "device": device,
        "seed": seed,
        "config": config,
        "metadata": metadata,
    }
    if action_schema == EE_DELTA_GRASP_V0:
        import embodied_jepa.config  # noqa: F401  (registers the MODELS entries)
        from embodied_jepa.registry import MODELS

        return MODELS.create(backend, **kwargs)
    if backend not in SCHEMA_BACKENDS:
        raise ContractError(
            f"backend {backend!r} takes only ee_delta_grasp_v0; schema-sized backends: "
            f"{sorted(SCHEMA_BACKENDS)}"
        )
    return SCHEMA_BACKENDS[backend]()(**kwargs, action_schema=action_schema)
