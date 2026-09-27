"""Frozen DINOv2 patch tokens, pooled to a coarse grid, as a world-model latent (TASK-066).

``frozen_token_model("leworldmodel")`` (or ``"native_jepa"``) returns the backend class with
``FrozenTokenMixin`` in front of it. The latent is the pinned, frozen DINOv2 ViT-S/14 final patch
tokens (``pretrained_encoder``, TASK-063's input handling and read-out point ``tokens``), average
pooled from the 16 x 16 patch grid to a ``token_grid`` x ``token_grid`` grid and standardised per
dimension by train-only moments. A prediction step is a **token predictor**: it maps the whole
grid of one frame and one action to the grid of the next frame.

**Reuse, not edits.** ``FrozenTokenMixin`` subclasses TASK-065's ``FrozenEncoderMixin`` and keeps
its train-only normalisation, its feature injection, ``train_step_features`` and
``predict_features`` unchanged; it overrides only construction, the feature read-out and the
prediction step. No existing file changes: ``models/frozen_encoder.py``, ``models/base.py``, the
backend files, ``readout.py``, ``readout_labels.py`` and ``pretrained_encoder.py`` keep their
bytes, so every earlier checkpoint whose ``implementation_sha256`` covers them (TASK-054's E0 and
TASK-065's twelve) stays loadable. The option is off unless a run selects this class; the plain
backends and the CLS class reject its keys.

* **Latent layout.** ``[..., token_grid**2 * 384]``, row-major over (grid row, grid column,
  channel). The model envelope (``VisualModel``) sees a flat vector, so the backend's own
  ``train_step``, ``rollout``, losses and optimizer run on it unchanged; only
  ``next_embedding`` reshapes it to ``[N, P, 384]`` tokens and back.
* **Pooling** (``pool_tokens``) is a plain mean over each block of patches, in float64, cast to
  float32. It is layout-independent; the tokens it pools are not (batched float32 inference
  depends on the batch size, TASK-065 run-1), and a run that caches features must bound that.
* **The predictor per backend** (the backend swap stays one key):

  - ``leworldmodel`` -- the pinned upstream LeWM ``ARPredictor`` (AdaLN-zero conditioned
    transformer), ``Embedder`` (action encoder) and ``pred_proj`` (BatchNorm MLP), composed by the
    pinned upstream ``JEPA.predict``, with the **token grid as the sequence axis** (history one,
    so there is no time axis) and a learned position embedding per token. Two adaptations, both
    outside the pinned source: the action embedding conditions every token of the frame (it is
    passed once per frame and broadcast), and attention is **bidirectional within the frame**
    (upstream masks causally over its sequence axis, which over a raster of tokens would blind
    each token to the tokens after it). The attention modules are the upstream ones with their
    upstream initialisation; only their mask argument is fixed to ``causal=False``.
  - ``native_jepa`` -- a native residual token step: each token plus its position embedding,
    the frame's mean token and the action go through one shared MLP, and the output is added to
    the token (native's residual form). Not run by TASK-066; it exists so that the swap is one key.

* **Nothing of the backend's image pathway is built** (a frozen external encoder replaces it),
  so the predictor's initialisation stream is this module's, not the backend's.

Torch and transformers are imported with the backend, never by ``import embodied_jepa``.
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

from embodied_jepa.contracts import EE_DELTA_GRASP_V0, ContractError
from embodied_jepa.models.base import VisualModel
from embodied_jepa.models.frozen_encoder import (
    BACKENDS,
    DIGEST_KEY,
    FrozenEncoderMixin,
)

# name -> (module, loader, token width, native patch grid). ``module.features(model, frames)``
# returns ``tokens`` [N, grid * grid * width] (row-major over the patch grid, then channel).
FROZEN_TOKEN_ENCODERS = {
    "dinov2_small_tokens": ("embodied_jepa.pretrained_encoder", "load_pretrained", 384, 16),
}
OPTION_KEYS = ("frozen_encoder", "token_grid")
# TASK-054's prediction-step options and TASK-050/052's fusion options are not combined with
# the token latent (none is preregistered with it).
REFUSED_ACTIVE = {
    "state_fusion": False,
    "readout_heads": False,
    "cameras": None,
    "action_chunk": 1,
    "predictor_step_embedding": False,
    "multistep_tail_weight": 0.0,
}


def pool_tokens(tokens, grid: int, *, native_grid: int = 16, width: int = 384) -> np.ndarray:
    """Average-pool flat patch tokens ``[N, native_grid**2 * width]`` to ``[N, grid**2 * width]``
    (float64 mean over each ``k x k`` block of patches, ``k = native_grid // grid``; float32 out).
    """
    tokens = np.asarray(tokens)
    if type(grid) is not int or grid < 1 or native_grid % grid:
        raise ContractError(f"token_grid must divide {native_grid}")
    if tokens.ndim != 2 or tokens.shape[1] != native_grid * native_grid * width:
        raise ContractError(f"tokens must be [N, {native_grid * native_grid * width}]")
    k = native_grid // grid
    x = tokens.astype(np.float64, copy=False).reshape(len(tokens), grid, k, grid, k, width)
    return x.mean(axis=(2, 4)).reshape(len(tokens), grid * grid * width).astype(np.float32)


class _WithinFrameAttention:
    """Marker base; the concrete class is made per upstream module (see ``_bidirectional``)."""


def _bidirectional(components):
    """A subclass of the pinned upstream ``Attention`` whose mask is fixed to non-causal."""
    base = components.Attention

    class WithinFrameAttention(base, _WithinFrameAttention):
        def forward(self, x, causal=False):  # noqa: ARG002 -- always bidirectional
            return base.forward(self, x, causal=False)

    return WithinFrameAttention


class NativeTokenStep(nn.Module):
    """Native residual token step: ``token + MLP([token + pos, mean token, action])``."""

    def __init__(self, tokens: int, width: int, hidden: int, action_dim: int):
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, tokens, width))
        self.mlp = nn.Sequential(
            nn.Linear(2 * width + action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, width),
        )

    def forward(self, tokens, actions):
        h = tokens + self.position
        context = h.mean(1, keepdim=True).expand_as(h)
        a = actions[:, None].expand(-1, tokens.shape[1], -1)
        return tokens + self.mlp(torch.cat((h, context, a), dim=-1))


class FrozenTokenMixin(FrozenEncoderMixin):
    """Placed before a ``VisualModel`` backend in the MRO; see the module docstring."""

    def __init__(self, state_schema, device="cpu", seed=0, config=None, metadata=None):
        merged = self.defaults | dict(config or {})
        name, grid = merged["frozen_encoder"], merged["token_grid"]
        if name not in FROZEN_TOKEN_ENCODERS:
            raise ContractError(f"frozen_encoder must be one of {sorted(FROZEN_TOKEN_ENCODERS)}")
        _, _, width, native = FROZEN_TOKEN_ENCODERS[name]
        if type(grid) is not int or grid < 1 or native % grid:
            raise ContractError(f"token_grid must be a positive divisor of {native}")
        if merged["latent_dim"] != grid * grid * width:
            raise ContractError(f"token_grid {grid} requires latent_dim {grid * grid * width}")
        for key, inactive in REFUSED_ACTIVE.items():
            if merged[key] != inactive:
                raise ContractError(f"frozen tokens are not combined with {key}")
        module = self._load_frozen_module(name)
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        from embodied_jepa import pretrained_encoder

        digest = pretrained_encoder.weights_digest(module)
        metadata = dict(metadata or {})
        if metadata.get(DIGEST_KEY, digest) != digest:
            raise ContractError("declared frozen encoder digest differs from the loaded encoder")
        metadata[DIGEST_KEY] = digest
        # The envelope only: the backend's image encoder is never built (see the docstring).
        VisualModel.__init__(self, state_schema, device, seed, config, metadata)
        self.token_count, self.token_width = grid * grid, width
        with self.rng_scope():
            self._build_token_predictor()
        self.finish_init()
        object.__setattr__(self, "_frozen_module", module.eval())
        object.__setattr__(self, "_injected_features", None)
        dim = self.config["latent_dim"]
        self.register_buffer("frozen_feature_mean", torch.zeros(dim, device=self.device_name))
        self.register_buffer("frozen_feature_scale", torch.ones(dim, device=self.device_name))
        self.register_buffer(
            "frozen_feature_normalization_fitted", torch.tensor(False, device=self.device_name)
        )

    def _load_frozen_module(self, name):
        module_name, loader, _, _ = FROZEN_TOKEN_ENCODERS[name]
        return getattr(importlib.import_module(module_name), loader)()

    # ----- the token predictor, per backend ---------------------------------------------------
    def _build_token_predictor(self):
        tokens, width = self.token_count, self.token_width
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
                num_frames=tokens,  # the token grid is the sequence axis
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
            within_frame = _bidirectional(components)
            for block in predictor.transformer.layers:
                block.attn.__class__ = within_frame
            self.model = upstream.JEPA(
                encoder=nn.Identity(),
                predictor=predictor,
                action_encoder=components.Embedder(
                    input_dim=EE_DELTA_GRASP_V0.dimension, emb_dim=width
                ),
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
            # Native's train_step reads an online and an EMA target encoder; the frozen feature
            # replaces both, so both are parameter-free identities.
            self.encoder = nn.Identity()
            self.target_encoder = nn.Identity()
            self.token_step = NativeTokenStep(
                tokens, width, self.config["hidden_dim"], EE_DELTA_GRASP_V0.dimension
            )
        else:  # pragma: no cover -- frozen_token_model refuses other backends
            raise ContractError(f"no token predictor for {self.backend}")

    def next_embedding(self, state, actions):
        """One prediction step on flat latents ``[..., P * width]`` and actions ``[..., 14]``."""
        shape = state.shape
        tokens = state.reshape(-1, self.token_count, self.token_width)
        actions = actions.reshape(-1, actions.shape[-1])
        if len(actions) != len(tokens):
            raise ContractError("latents and actions disagree on the batch")
        if self.backend == "leworldmodel":
            condition = self.model.action_encoder(actions[:, None])  # [N, 1, width], per frame
            out = self.model.predict(tokens, condition)  # upstream predictor + pred_proj
        else:
            out = self.token_step(tokens, actions)
        return out.reshape(shape)

    def embed(self, pixels):  # pragma: no cover -- the frozen feature replaces the pathway
        raise ContractError("frozen-token models read frames through frozen_features only")

    goal_embed = embed

    # ----- features ---------------------------------------------------------------------------
    @torch.no_grad()
    def frozen_features(self, frames):
        """Pooled raw frozen tokens ``float32 [N, P * width]`` of uint8 frames (CPU)."""
        from embodied_jepa import pretrained_encoder

        # A writable copy: canonical batches hold read-only snapshots torch must not alias.
        frames = np.array(frames, copy=True)
        tokens = pretrained_encoder.features(self._frozen_module, frames)["tokens"]
        _, _, width, native = FROZEN_TOKEN_ENCODERS[self.config["frozen_encoder"]]
        return pool_tokens(tokens, self.config["token_grid"], native_grid=native, width=width)

    @property
    def implementation_sha256(self):
        digest = hashlib.sha256(VisualModel.implementation_sha256.fget(self).encode())
        backend = next(
            c for c in type(self).__mro__ if c.__module__ in {m for m, _ in BACKENDS.values()}
        )
        name = self.config["frozen_encoder"]
        for path in (
            Path(__file__),
            Path(inspect.getfile(FrozenEncoderMixin)),
            Path(inspect.getfile(backend)),
            Path(importlib.import_module(FROZEN_TOKEN_ENCODERS[name][0]).__file__),
        ):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()


@cache
def frozen_token_model(backend: str):
    """The backend class with the frozen token latent in front; both option keys are required."""
    if backend not in BACKENDS:
        raise ContractError(f"frozen tokens support the backends {sorted(BACKENDS)}")
    module_name, class_name = BACKENDS[backend]
    base = getattr(importlib.import_module(module_name), class_name)
    return type(
        f"FrozenToken{class_name}",
        (FrozenTokenMixin, base),
        {"__module__": __name__, "defaults": base.defaults | dict.fromkeys(OPTION_KEYS)},
    )
