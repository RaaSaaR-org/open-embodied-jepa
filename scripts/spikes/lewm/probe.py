"""Bounded compatibility probe of pinned, unmodified upstream LeWM classes.

This is synthetic execution evidence, not training or benchmark evidence.
See docs/LEWM_SPIKE.md for setup, contracts, and deliberately excluded claims.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch
from transformers import ViTConfig, ViTModel

REVISION = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"


def source_module(root: Path, filename: str):
    spec = importlib.util.spec_from_file_location(f"lewm_probe_{filename}", root / f"{filename}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def probe(root: Path, device: str, profile: str, steps: int, history_frames: int) -> dict:
    torch.manual_seed(3072)
    upstream = source_module(root, "jepa")
    components = source_module(root, "module")
    # The upstream profile matches its tiny ViT / predictor config. Smoke scales
    # architecture and pixels down, while exercising the same upstream classes.
    full = profile == "upstream"
    size, dim, layers = (224, 192, 12) if full else (56, 48, 2)
    predictor = dict(
        num_frames=history_frames,
        input_dim=dim,
        hidden_dim=dim,
        output_dim=dim,
        depth=6 if full else 2,
        heads=16 if full else 2,
        mlp_dim=2048 if full else 96,
        dim_head=64 if full else 24,
        dropout=0.1,
        emb_dropout=0.0,
    )
    encoder = ViTModel(
        ViTConfig(
            hidden_size=dim,
            num_hidden_layers=layers,
            num_attention_heads=3,
            intermediate_size=dim * 4,
            image_size=size,
            patch_size=14,
        ),
        add_pooling_layer=False,
        use_mask_token=False,
    )
    model = upstream.JEPA(
        encoder=encoder,
        predictor=components.ARPredictor(**predictor),
        action_encoder=components.Embedder(input_dim=14, emb_dim=dim),
        projector=components.MLP(dim, 2048 if full else 96, dim, norm_fn=torch.nn.BatchNorm1d),
        pred_proj=components.MLP(dim, 2048 if full else 96, dim, norm_fn=torch.nn.BatchNorm1d),
    ).to(device)
    regularizer = components.SIGReg(knots=17, num_proj=1024).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)

    # Canonical uint8 channels-last camera sequence and normalized G1 action shape.
    # These random images/actions are not robot data or physical demonstrations.
    camera = torch.randint(0, 256, (2, history_frames + 1, size, size, 3), dtype=torch.uint8)
    actions = torch.rand(2, history_frames, 14) * 2 - 1
    fixture_hash = hashlib.sha256(camera.numpy().tobytes() + actions.numpy().tobytes()).hexdigest()
    pixels = camera.permute(0, 1, 4, 2, 3).float().to(device) / 255
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    pixels = (pixels - mean) / std
    actions = actions.to(device)
    synchronize(device)
    start = time.perf_counter()
    losses = []
    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        encoded = model.encode({"pixels": pixels, "action": actions})
        prediction = model.predict(encoded["emb"][:, :history_frames], encoded["act_emb"])
        prediction_loss = (prediction - encoded["emb"][:, 1:]).square().mean()
        sigreg = regularizer(encoded["emb"].transpose(0, 1))
        loss = prediction_loss + 0.09 * sigreg
        if not torch.isfinite(loss):
            raise AssertionError("non-finite loss")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise AssertionError("missing or non-finite gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(
            {"total": loss.item(), "prediction": prediction_loss.item(), "sigreg": sigreg.item()}
        )
    synchronize(device)
    train_seconds = time.perf_counter() - start

    model.eval()
    # H frames imply H-1 executed actions followed by four future candidates.
    past = history_frames - 1
    candidates = torch.zeros(2, 2, past + 4, 14, device=device)
    candidates[:, :, :past] = actions[:, None, :past]
    candidates[:, 0, past:] = -0.5
    candidates[:, 1, past:] = 0.5
    with torch.no_grad():
        synchronize(device)
        start = time.perf_counter()
        result = model.rollout(
            {"pixels": pixels[:, None, :history_frames].expand(-1, 2, -1, -1, -1, -1)},
            candidates,
            history_size=history_frames,
        )["predicted_emb"]
        synchronize(device)
        rollout_seconds = time.perf_counter() - start
        if (
            tuple(result.shape) != (2, 2, history_frames + 4, dim)
            or not torch.isfinite(result).all()
        ):
            raise AssertionError(f"invalid rollout {result.shape}")
        futures = result[:, :, history_frames:]
        goal = model.encode({"pixels": pixels[:, -1:]})["emb"]
        costs = (futures - goal[:, None]).square().mean(-1)
        if tuple(costs.shape) != (2, 2, 4) or not torch.isfinite(costs).all():
            raise AssertionError("invalid per-step image-goal costs")
        action_delta = (futures[:, 0] - futures[:, 1]).abs().max().item()
        if action_delta <= 1e-8:
            raise AssertionError("actions do not affect predictions after optimizer update")
        # Independent recurrence validates alignment against upstream rollout.
        history = model.encode({"pixels": pixels[:, :history_frames]})["emb"]
        history = history[:, None].expand(-1, 2, -1, -1).reshape(4, history_frames, dim).clone()
        candidate_flat = candidates.reshape(4, past + 4, 14)
        expected = []
        for step in range(4):
            conditioning = model.action_encoder(candidate_flat[:, step : step + history_frames])
            next_embedding = model.predict(history[:, -history_frames:], conditioning)[:, -1:]
            expected.append(next_embedding)
            history = torch.cat([history, next_embedding], dim=1)
        expected = torch.cat(expected, dim=1).reshape(2, 2, 4, dim)
        torch.testing.assert_close(futures, expected, rtol=1e-4, atol=1e-5)

    return {
        "device": device,
        "profile": profile,
        "image_size": size,
        "latent_dim": dim,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "encoder_layers": layers,
        "predictor": predictor,
        "action_dim": 14,
        "history_frames": history_frames,
        "future_horizon": 4,
        "seed": 3072,
        "steps": steps,
        "batch_size": 2,
        "fixture_sha256": fixture_hash,
        "losses": losses,
        "all_gradients_finite": True,
        "rollout_shape": list(result.shape),
        "goal_cost_shape": list(costs.shape),
        "rollout_matches_recurrence": True,
        "action_max_absolute_delta": action_delta,
        "train_seconds": train_seconds,
        "rollout_seconds": rollout_seconds,
        "requires_future_robot_state": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("third_party/le-wm"))
    parser.add_argument("--device", choices=["cpu", "mps", "all"], default="all")
    parser.add_argument("--profile", choices=["smoke", "upstream"], default="smoke")
    parser.add_argument("--history", type=int, choices=[1, 3], default=3)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("outputs/lewm-spike.json"))
    args = parser.parse_args()
    if not 1 <= args.steps <= 10:
        parser.error("--steps must be between 1 and 10 for this bounded probe")
    if git(args.source, "rev-parse", "HEAD") != REVISION:
        parser.error(f"source must be pinned to {REVISION}")
    if git(args.source, "status", "--porcelain", "--untracked-files=no"):
        parser.error("source has modified tracked files")
    if os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        parser.error("disable implicit CPU fallback to measure actual MPS compatibility")
    torch.set_num_threads(4)
    versions = {
        name: importlib.metadata.version(name)
        for name in ["torch", "transformers", "einops", "numpy"]
    }
    report = {
        "kind": "synthetic_compatibility_probe",
        "source_revision": REVISION,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": versions,
        "threads": 4,
        "mps_available": torch.backends.mps.is_available(),
        "source_sha256": {
            name: hashlib.sha256((args.source / name).read_bytes()).hexdigest()
            for name in ["jepa.py", "module.py", "LICENSE"]
        },
        "results": [],
    }
    devices = ["cpu", "mps"] if args.device == "all" else [args.device]
    failed = False
    for device in devices:
        if device == "mps" and not torch.backends.mps.is_available():
            report["results"].append({"device": device, "status": "unavailable"})
            failed |= args.device == "mps"
            continue
        try:
            result = probe(args.source, device, args.profile, args.steps, args.history)
            report["results"].append({"status": "passed", **result})
        except Exception as error:
            report["results"].append(
                {"device": device, "status": "failed", "error": f"{type(error).__name__}: {error}"}
            )
            failed = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
