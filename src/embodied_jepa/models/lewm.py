"""Optional adapter around pinned, unmodified upstream LeWorldModel classes."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
from torch import nn
from transformers import ViTConfig, ViTModel

from embodied_jepa.contracts import ContractError
from embodied_jepa.models.base import VisualModel

LEWM_REVISION = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
SOURCE_HASHES = {
    "jepa.py": "41bad7fd21e0f14aea4c9c3d39a9c87037e787746d953ab62cdc0677e938ce96",
    "module.py": "0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620",
    "LICENSE": "4882021920cbd61e050e4dc2e02440e9bd36ea4cbc5803e738a2ee3d63502e07",
}


def load_upstream(source):
    source = Path(source)
    if not source.is_dir():
        raise ContractError("LeWM source missing; run python scripts/fetch_lewm.py")
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != LEWM_REVISION:
        raise ContractError(f"LeWM requires source revision {LEWM_REVISION}")
    for filename, expected in SOURCE_HASHES.items():
        if hashlib.sha256((source / filename).read_bytes()).hexdigest() != expected:
            raise ContractError(f"LeWM source integrity failure: {filename}")
    modules = []
    for name in ("jepa", "module"):
        spec = importlib.util.spec_from_file_location(
            f"embodied_jepa_upstream_{name}", source / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


class LeWM(VisualModel):
    backend = "leworldmodel"
    defaults = VisualModel.defaults | {
        "image_size": 56,
        "latent_dim": 48,
        "patch_size": 14,
        "encoder_depth": 2,
        "encoder_heads": 3,
        "predictor_depth": 2,
        "predictor_heads": 2,
        "predictor_head_dim": 24,
        "sigreg_weight": 0.09,
        "sigreg_projections": 128,
        "multistep_weight": 1.0,
        "source_path": "third_party/le-wm",
    }

    def __init__(self, state_schema, device="cpu", seed=0, config=None, metadata=None):
        super().__init__(state_schema, device, seed, config, metadata)
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
            raise ContractError("image/patch size and embedding/encoder heads must divide evenly")
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
                action_encoder=components.Embedder(input_dim=14, emb_dim=dim),
                projector=components.MLP(dim, hidden, dim, norm_fn=nn.BatchNorm1d),
                pred_proj=components.MLP(dim, hidden, dim, norm_fn=nn.BatchNorm1d),
            )
            self.sigreg = components.SIGReg(knots=17, num_proj=self.config["sigreg_projections"])
        self.finish_init()

    @property
    def source_revision(self):
        return LEWM_REVISION

    def embed(self, pixels):
        mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return self.model.encode({"pixels": ((pixels - mean) / std)[:, None]})["emb"][:, 0]

    def next_embedding(self, state, actions):
        shape = state.shape
        state = state.reshape(-1, 1, shape[-1])
        actions = actions.reshape(-1, 1, actions.shape[-1])
        conditioning = self.model.action_encoder(actions)
        return self.model.predict(state, conditioning)[:, 0].reshape(shape)

    def train_step(self, batch, readout_targets=None):
        self.check_batch(batch)
        if batch.batch_size < 2:
            raise ContractError("LeWM BatchNorm requires at least two sequences during training")
        self.train()
        with self.rng_scope():
            actions = self.sequence_tensors(batch)
            embeddings = self.observe_sequence(batch)
            one_step = self.next_embedding(embeddings[:, :-1], actions)
            prediction_loss = (one_step - embeddings[:, 1:]).square().mean()
            recursive = self.rollout(embeddings[:, 0], actions)
            multistep = self.multistep_loss(recursive, embeddings[:, 1:])
            sigreg = self.sigreg(embeddings.transpose(0, 1))
            loss = (
                prediction_loss
                + self.config["multistep_weight"] * multistep
                + self.config["sigreg_weight"] * sigreg
            )
            readout_loss, readout_metrics = self.readout_loss(
                embeddings, recursive, readout_targets
            )
            loss = loss + readout_loss
            grad_norm = self.optimize(loss, embeddings)
        self.eval()
        return {
            "loss": loss.item(),
            "prediction_loss": prediction_loss.item(),
            "multistep_loss": multistep.item(),
            "sigreg_loss": sigreg.item(),
            "gradient_norm": grad_norm,
            "updates": float(self.updates),
        } | readout_metrics
