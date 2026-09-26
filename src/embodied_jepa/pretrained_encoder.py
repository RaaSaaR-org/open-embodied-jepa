"""TASK-063: the pinned, frozen DINOv2 ViT-S/14 encoder and its seed-0 random-init floor.

Protocol ``apple_pretrained_encoder_v1`` (``docs/experiments/apple_pretrained_encoder_v1.md``).
This module only builds the two encoders and reads their features; it trains nothing and fits
nothing. It is a readability instrument, not a control component.

* **Pretrained** (``load_pretrained``): the Hugging Face transformers conversion
  ``facebook/dinov2-small`` at a pinned revision, read from local files fetched by
  ``scripts/fetch_dinov2.py``. Every file's sha256 is checked before it is read, the state dict
  is loaded ``strict``, and nothing is fetched from the network.
* **Floor** (``random_init``): the same ``Dinov2Config`` (the same pinned ``config.json``),
  built under ``torch.manual_seed(0)`` inside a forked RNG with transformers' own initialiser.
* **Input** (``preprocess``): the 112 px uint8 RGB frame, scaled to [0, 1], resized to 224 x 224
  by bicubic interpolation (no crop), then ImageNet-normalised. 224 px is DINOv2's standard
  evaluation resolution; with patch 14 it gives a 16 x 16 token grid.
* **Read-out points** (``features``), both after the final LayerNorm:
  ``cls`` -- the CLS token (``pooler_output``), 384-d;
  ``tokens`` -- the 256 patch tokens (``last_hidden_state[:, 1:]``), flattened, 98 304-d.

Everything runs on CPU in float32 with eager attention, so the features are bit-reproducible on
one machine. Torch and transformers are imported lazily: ``import embodied_jepa`` stays
torch-free.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from embodied_jepa.contracts import ContractError

HF_REPOSITORY = "facebook/dinov2-small"
HF_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
FILE_SHA256 = {
    "config.json": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
    "model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
}
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[2] / "third_party" / "dinov2-small"
INPUT_SIZE = 224
FRAME_SIZE = 112
PATCH = 14
GRID = INPUT_SIZE // PATCH  # 16
WIDTH = 384
FLOOR_SEED = 0
BATCH = 16
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
READOUT_POINTS = ("cls", "tokens")


class WeightsError(ContractError):
    """The local weights are missing or are not the pinned ones (a guard failure)."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_files(directory: Path, pinned: dict = FILE_SHA256) -> dict:
    """sha256 of every pinned file; raises ``WeightsError`` on a missing or different file."""
    found = {}
    for name, want in pinned.items():
        path = Path(directory) / name
        if not path.is_file():
            raise WeightsError(f"G-weights: {path} is missing (run scripts/fetch_dinov2.py)")
        got = sha256(path)
        if got != want:
            raise WeightsError(f"G-weights: {name} sha256 {got[:12]} != pinned {want[:12]}")
        found[name] = got
    return found


def weights_digest(module) -> str:
    """sha256 over a module's state dict in key order (TASK-062's ``weights_digest`` rule)."""
    digest = hashlib.sha256()
    state = module.state_dict()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _config(directory: Path):
    from transformers import Dinov2Config

    config = Dinov2Config.from_json_file(str(Path(directory) / "config.json"))
    config._attn_implementation = "eager"
    if (config.patch_size, config.hidden_size) != (PATCH, WIDTH):
        raise WeightsError("G-weights: config is not ViT-S/14")
    return config


def _build(config, seed: int):
    """``Dinov2Model(config)`` under ``torch.manual_seed(seed)`` in a forked CPU RNG."""
    import torch
    from transformers import Dinov2Model

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Dinov2Model(config)
    return model.float().eval()


def load_pretrained(directory: Path = DEFAULT_DIRECTORY, *, pinned: dict = FILE_SHA256):
    """The pinned pretrained encoder (CPU, float32, eval), after checking every file's sha256."""
    from safetensors.torch import load_file

    check_files(directory, pinned)
    model = _build(_config(directory), FLOOR_SEED)
    state = load_file(str(Path(directory) / "model.safetensors"))
    model.load_state_dict(state, strict=True)
    return model.eval()


def random_init(directory: Path = DEFAULT_DIRECTORY, *, pinned: dict = FILE_SHA256):
    """The floor: the same architecture at its seed-0 initialisation (CPU, float32, eval)."""
    check_files(directory, {"config.json": pinned["config.json"]})
    return _build(_config(directory), FLOOR_SEED)


def preprocess(frames):
    """uint8 [N, 112, 112, 3] RGB -> float32 [N, 3, 224, 224], ImageNet-normalised."""
    import torch
    import torch.nn.functional as F

    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (112, 112, 3):
        raise ContractError(
            f"expected uint8 [N, 112, 112, 3] frames, got {frames.dtype} {frames.shape}"
        )
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(
        x, size=(INPUT_SIZE, INPUT_SIZE), mode="bicubic", align_corners=False, antialias=False
    )
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def features(model, frames, *, batch: int = BATCH, hidden_states: bool = False) -> dict:
    """float64 read-outs of uint8 frames: ``cls`` [N, 384] and ``tokens`` [N, 256 * 384].

    With ``hidden_states`` (the label-free calibration only) it also returns the patch tokens
    after the patch embedding and after every block (before the final LayerNorm)."""
    import torch

    model.eval()
    out: dict[str, list] = {}
    with torch.no_grad():
        for start in range(0, len(frames), batch):
            pixels = preprocess(frames[start : start + batch])
            result = model(pixel_values=pixels, output_hidden_states=hidden_states)
            last = result.last_hidden_state
            if last.shape[1:] != (1 + GRID * GRID, WIDTH):
                raise ContractError(f"unexpected token grid {tuple(last.shape)}")
            parts = {"cls": result.pooler_output, "tokens": last[:, 1:]}
            if hidden_states:
                parts["patch_embedding_tokens"] = result.hidden_states[0][:, 1:]
                for k in range(1, len(result.hidden_states)):
                    parts[f"block{k}_tokens"] = result.hidden_states[k][:, 1:]
                parts["tokens_mean"] = last[:, 1:].mean(1)
            for name, value in parts.items():
                out.setdefault(name, []).append(
                    value.reshape(len(value), -1).cpu().numpy().astype(np.float64)
                )
    joined = {k: np.concatenate(v) for k, v in out.items()}
    for name, value in joined.items():
        if not np.isfinite(value).all():
            raise ContractError(f"non-finite {name} features")
    return joined
