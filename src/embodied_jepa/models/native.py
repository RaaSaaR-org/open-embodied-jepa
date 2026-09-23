"""Compact visual JEPA with EMA targets and explicit noncollapse penalties."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from embodied_jepa.contracts import ContractError
from embodied_jepa.models.base import VisualModel


class NativeJEPA(VisualModel):
    backend = "native_jepa"
    defaults = VisualModel.defaults | {
        "ema_decay": 0.99,
        "variance_weight": 1.0,
        "covariance_weight": 0.01,
        "multistep_weight": 1.0,
    }

    def __init__(self, state_schema, device="cpu", seed=0, config=None, metadata=None):
        super().__init__(state_schema, device, seed, config, metadata)
        if not 0 <= self.config["ema_decay"] < 1:
            raise ContractError("ema_decay must lie in [0,1)")
        for name in ("variance_weight", "covariance_weight", "multistep_weight"):
            if not np.isfinite(self.config[name]) or self.config[name] < 0:
                raise ContractError(f"{name} must be nonnegative and finite")
        dim, hidden = self.config["latent_dim"], self.config["hidden_dim"]
        with self.rng_scope():
            # The 4x4 spatial grid preserves object location; global pooling would
            # hide the spatial differences needed for an image-goal task.
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
                nn.Linear(dim + 14, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, dim),
            )
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.finish_init()

    def embed(self, pixels):
        return self.encoder(pixels)

    def goal_embed(self, pixels):
        return self.target_encoder(pixels)

    def next_embedding(self, state, actions):
        return state + self.predictor(torch.cat((state, actions), dim=-1))

    def train_step(self, batch, readout_targets=None):
        self.train()
        self.target_encoder.eval()
        with self.rng_scope():
            actions = self.sequence_tensors(batch)
            embeddings = self.observe_sequence(batch)
            with torch.no_grad():
                targets = self.observe_sequence(batch, target=True)
            one_step = self.next_embedding(embeddings[:, :-1], actions)
            prediction_loss = F.mse_loss(one_step, targets[:, 1:])
            recursive = self.rollout(embeddings[:, 0], actions)
            multistep = self.multistep_loss(recursive, targets[:, 1:])
            samples = embeddings.reshape(-1, embeddings.shape[-1])
            # Population variance is defined for every legal canonical batch.
            variance = F.relu(1 - torch.sqrt(samples.var(0, unbiased=False) + 1e-4)).mean()
            centered = samples - samples.mean(0)
            covariance = centered.T @ centered / max(samples.shape[0] - 1, 1)
            offdiag = covariance - torch.diag_embed(covariance.diagonal())
            covariance_loss = offdiag.square().sum() / samples.shape[-1]
            loss = (
                prediction_loss
                + self.config["multistep_weight"] * multistep
                + self.config["variance_weight"] * variance
                + self.config["covariance_weight"] * covariance_loss
            )
            readout_loss, readout_metrics = self.readout_loss(
                embeddings, recursive, readout_targets
            )
            loss = loss + readout_loss
            grad_norm = self.optimize(loss, targets)
            decay = self.config["ema_decay"]
            with torch.no_grad():
                for target, online in zip(
                    self.target_encoder.parameters(), self.encoder.parameters(), strict=True
                ):
                    target.lerp_(online, 1 - decay)
        self.eval()
        return {
            "loss": loss.item(),
            "prediction_loss": prediction_loss.item(),
            "multistep_loss": multistep.item(),
            "variance_loss": variance.item(),
            "covariance_loss": covariance_loss.item(),
            "gradient_norm": grad_norm,
            "updates": float(self.updates),
        } | readout_metrics
