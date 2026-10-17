"""Schema-sized, two-camera frozen-token world models for TASK-100 (SO-101 S4, protocol §4).

``schema_frozen_token_model("leworldmodel")`` (or ``"native_jepa"``) returns a class that puts
TASK-066's frozen-token composition (``models/frozen_tokens.FrozenTokenMixin``: the pinned
upstream LeWM ``ARPredictor``, ``Embedder`` and ``pred_proj`` with the token grid as the sequence
axis, bidirectional within the frame, the action embedding conditioning every token; or the
native residual token step) in front of S0's ``SchemaLeWM`` / ``SchemaNativeJEPA``
(``models/schema_backends``). Only what the schema and the two-camera latent change is new:

* the action width is ``action_schema.dimension`` (6 for ``joint_delta_so101_v0``), not 14;
* the token count and width are configuration (``token_count``, ``token_width``: 32 x 384 for
  g4, 128 x 96 for g8), not a pooled single-camera grid; the latent is the flat
  ``token_count * token_width`` vector (wrist tokens, then overview tokens), and the per-token
  position embedding carries camera identity;
* the model reads **cached, already standardised features only** (``so101_wm_data`` featurises
  and standardises with the fit set's moments, floor 1e-6, §3): the frozen-feature normalisation
  buffers of ``FrozenEncoderMixin`` are fixed to the identity, and ``frozen_features`` refuses.
  The frozen encoder named by ``frozen_encoder`` (``dinov2_small_224`` or its seed-0 random init
  ``dinov2_small_224_random_init``) is loaded once for its weights digest, which goes into the
  checkpoint metadata and is compared on load;
* ``train_step_features`` and ``predict_features`` are ``FrozenEncoderMixin``'s with the declared
  schema in place of ``ee_delta_grasp_v0``; training is the backend's own ``train_step``.

**No existing file changes**: ``models/frozen_tokens.py``, ``models/frozen_encoder.py``,
``models/base.py``, the backend files, ``models/schema_backends.py`` and
``pretrained_encoder.py`` keep their bytes. ``implementation_sha256`` covers all of them and this
file, so a checkpoint from these classes loads only under this exact code.

Torch and transformers are imported with this module, never by ``import embodied_jepa``.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
from functools import cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

from embodied_jepa.contracts import ActionSchema, ContractError, SequenceBatch, validate_actions
from embodied_jepa.models import frozen_encoder as fe
from embodied_jepa.models import frozen_tokens as ft
from embodied_jepa.models import schema_backends as sb
from embodied_jepa.models.base import VisualModel

FROZEN_ENCODERS = {
    "dinov2_small_224": "load_pretrained",
    "dinov2_small_224_random_init": "random_init",
}
OPTION_KEYS = ("frozen_encoder", "token_count", "token_width")
REFUSED_ACTIVE = ft.REFUSED_ACTIVE
PLACEHOLDER_CAMERA = "cached_features"


class SchemaFrozenTokenMixin(ft.FrozenTokenMixin):
    """Placed before ``SchemaActionMixin`` and the backend in the MRO; see the module docstring."""

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
        schema = sb._checked_schema(action_schema, config)
        merged = self.defaults | dict(config or {})
        name, tokens, width = (merged[k] for k in OPTION_KEYS)
        if name not in FROZEN_ENCODERS:
            raise ContractError(f"frozen_encoder must be one of {sorted(FROZEN_ENCODERS)}")
        for key, value in (("token_count", tokens), ("token_width", width)):
            if type(value) is not int or value < 1:
                raise ContractError(f"{key} must be a positive integer")
        if merged["latent_dim"] != tokens * width:
            raise ContractError(f"{tokens} tokens x {width} require latent_dim {tokens * width}")
        if merged["camera"] != PLACEHOLDER_CAMERA:
            raise ContractError(f"camera must be {PLACEHOLDER_CAMERA!r}: cached features only")
        for key, inactive in REFUSED_ACTIVE.items():
            if merged[key] != inactive:
                raise ContractError(f"frozen tokens are not combined with {key}")
        module = self._load_frozen_module(name)
        from embodied_jepa import pretrained_encoder

        digest = pretrained_encoder.weights_digest(module)
        del module
        metadata = dict(metadata or {})
        if metadata.get(fe.DIGEST_KEY, digest) != digest:
            raise ContractError("declared frozen encoder digest differs from the loaded encoder")
        metadata[fe.DIGEST_KEY] = digest
        VisualModel.__init__(self, state_schema, device, seed, config, metadata)
        self.action_schema = schema
        self.token_count, self.token_width = tokens, width
        with self.rng_scope():
            self._build_token_predictor()
        self.finish_init()
        object.__setattr__(self, "_frozen_module", None)
        object.__setattr__(self, "_injected_features", None)
        dim = self.config["latent_dim"]
        # Identity normalisation: the features arrive standardised by the fit set's moments.
        self.register_buffer("frozen_feature_mean", torch.zeros(dim, device=self.device_name))
        self.register_buffer("frozen_feature_scale", torch.ones(dim, device=self.device_name))
        self.register_buffer(
            "frozen_feature_normalization_fitted", torch.tensor(True, device=self.device_name)
        )

    def _load_frozen_module(self, name):
        from embodied_jepa import pretrained_encoder

        return getattr(pretrained_encoder, FROZEN_ENCODERS[name])()

    def fit_frozen_feature_normalization(self, *_args, **_kwargs):
        raise ContractError("features arrive standardised; the model normalisation is the identity")

    # ----- the token predictor, per backend (FrozenTokenMixin's, with the schema's width) -----
    def _build_token_predictor(self):
        tokens, width, adim = self.token_count, self.token_width, self.action_schema.dimension
        if self.backend == "leworldmodel":
            for key in ("predictor_depth", "predictor_heads", "predictor_head_dim"):
                if type(self.config[key]) is not int or self.config[key] < 1:
                    raise ContractError(f"{key} must be a positive integer")
            if type(self.config["sigreg_projections"]) is not int:
                raise ContractError("sigreg_projections must be a positive integer")
            for key in ("sigreg_weight", "multistep_weight"):
                if not np.isfinite(self.config[key]) or self.config[key] < 0:
                    raise ContractError(f"{key} must be nonnegative and finite")
            from embodied_jepa.models.lewm import load_upstream

            upstream, components = load_upstream(self.config["source_path"])
            predictor = components.ARPredictor(
                num_frames=tokens,  # the token sequence of both cameras is the sequence axis
                input_dim=width,
                hidden_dim=width,
                output_dim=width,
                depth=self.config["predictor_depth"],
                heads=self.config["predictor_heads"],
                dim_head=self.config["predictor_head_dim"],
                mlp_dim=self.config["hidden_dim"],
                dropout=0.0,
                emb_dropout=0.0,
            )
            within_frame = ft._bidirectional(components)
            for block in predictor.transformer.layers:
                block.attn.__class__ = within_frame
            self.model = upstream.JEPA(
                encoder=nn.Identity(),
                predictor=predictor,
                action_encoder=components.Embedder(input_dim=adim, emb_dim=width),
                pred_proj=components.MLP(
                    width, self.config["hidden_dim"], width, norm_fn=nn.BatchNorm1d
                ),
            )
            self.sigreg = components.SIGReg(knots=17, num_proj=self.config["sigreg_projections"])
        elif self.backend == "native_jepa":
            if not 0 <= self.config["ema_decay"] < 1:
                raise ContractError("ema_decay must lie in [0,1)")
            for key in ("variance_weight", "covariance_weight", "multistep_weight"):
                if not np.isfinite(self.config[key]) or self.config[key] < 0:
                    raise ContractError(f"{key} must be nonnegative and finite")
            self.encoder = nn.Identity()
            self.target_encoder = nn.Identity()
            self.token_step = ft.NativeTokenStep(tokens, width, self.config["hidden_dim"], adim)
        else:  # pragma: no cover -- schema_frozen_token_model refuses other backends
            raise ContractError(f"no token predictor for {self.backend}")

    def frozen_features(self, frames):  # pragma: no cover -- cached features only
        raise ContractError("TASK-100 models read cached features (so101_wm_data) only")

    # ----- training and prediction on cached, standardised features ---------------------------
    def train_step_features(self, features, actions):
        """One update from standardised features ``[B, T+1, D]`` and normalised float32 actions
        ``[B, T, A]``: the backend's own ``train_step`` with the features injected."""
        features = np.asarray(features)
        actions = np.asarray(actions)
        if (
            features.ndim != 3
            or actions.ndim != 3
            or features.shape[0] != actions.shape[0]
            or features.shape[1] != actions.shape[1] + 1
        ):
            raise ContractError("features [B,T+1,D] and actions [B,T,A] must agree")
        batch_size, horizon = actions.shape[:2]
        normalized = self._normalize(features)
        schema = self.state_schema
        placeholder = SequenceBatch(
            observations={
                PLACEHOLDER_CAMERA: np.zeros((batch_size, horizon + 1, 1, 1, 3), np.uint8)
            },
            robot_states=np.zeros((batch_size, horizon + 1, schema.dimension), np.float32),
            state_mask=np.ones((batch_size, horizon + 1, schema.dimension), np.bool_),
            actions=actions,
            timestamps=np.tile(np.arange(horizon + 1, dtype=np.float64), (batch_size, 1)),
            terminated=np.zeros((batch_size, horizon), np.bool_),
            episode_ids=tuple(f"features-{i}" for i in range(batch_size)),
            state_schema=schema,
            action_schema=self.action_schema,
        )
        with self._inject(normalized):
            return self.train_step(placeholder)

    @torch.no_grad()
    def predict_features(self, features, actions):
        """Recursive roll-out from standardised start features ``[N, D]`` under ``[N, T, A]``
        actions -> predicted standardised features ``float32 [N, T, D]`` (CPU)."""
        self.eval()
        actions = np.asarray(actions)
        if actions.ndim != 3 or actions.shape[-1] != self.action_schema.dimension:
            raise ContractError(f"actions must be [N, T, {self.action_schema.dimension}]")
        if actions.shape[1] > self.config["max_horizon"]:
            raise ContractError("rollout exceeds max_horizon")
        validate_actions(actions[:, None], schema=self.action_schema)
        initial = self._normalize(features)
        if initial.ndim != 2 or initial.shape[0] != actions.shape[0]:
            raise ContractError("features [N,D] and actions [N,T,A] must agree")
        chunk = self.config["candidate_chunk_size"]
        outputs = []
        with self.rng_scope():
            for start in range(0, len(actions), chunk):
                part = torch.from_numpy(np.array(actions[start : start + chunk], copy=True))
                predicted = self.rollout(initial[start : start + chunk], part.to(self.device_name))
                outputs.append(predicted.float().cpu())
        result = torch.cat(outputs).numpy().astype(np.float32)
        if not np.isfinite(result).all():
            raise ContractError("model produced non-finite predicted features")
        return result

    @torch.no_grad()
    def predict_at(self, features, actions, horizons) -> dict:
        """``{h: float32 [N, D]}``: the roll-out's predictions at the requested horizons."""
        full = self.predict_features(features, np.asarray(actions)[:, : max(horizons)])
        return {h: full[:, h - 1] for h in horizons}

    @property
    def implementation_sha256(self):
        digest = hashlib.sha256(VisualModel.implementation_sha256.fget(self).encode())
        backend = next(
            c for c in type(self).__mro__ if c.__module__ in {m for m, _ in fe.BACKENDS.values()}
        )
        from embodied_jepa import pretrained_encoder

        for path in (
            Path(__file__),
            Path(ft.__file__),
            Path(fe.__file__),
            Path(sb.__file__),
            Path(inspect.getfile(backend)),
            Path(pretrained_encoder.__file__),
        ):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()


@cache
def schema_frozen_token_model(backend: str):
    """The schema-sized backend with the two-camera frozen-token latent in front."""
    if backend not in sb.SCHEMA_BACKENDS:
        raise ContractError(
            f"schema frozen tokens support the backends {sorted(sb.SCHEMA_BACKENDS)}"
        )
    schema_class = sb.SCHEMA_BACKENDS[backend]()
    module_name, class_name = fe.BACKENDS[backend]
    base = getattr(importlib.import_module(module_name), class_name)
    return type(
        f"SchemaFrozenToken{class_name}",
        (SchemaFrozenTokenMixin, sb.SchemaActionMixin, base),
        {
            "__module__": __name__,
            "defaults": base.defaults | dict.fromkeys(OPTION_KEYS),
            "pinned_backend": schema_class.pinned_backend,
        },
    )


def create(
    backend: str, *, state_schema, action_schema, config, device="cpu", seed=0, metadata=None
):
    return schema_frozen_token_model(backend)(
        state_schema, device, seed, config, metadata, action_schema=action_schema
    )


def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
