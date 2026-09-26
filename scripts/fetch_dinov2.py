"""Fetch the pinned DINOv2 ViT-S/14 weights (Apache-2.0) for TASK-063, and verify them.

Downloads, into the git-ignored ``third_party/dinov2-small/``:

* the Hugging Face transformers conversion ``facebook/dinov2-small`` at the pinned repository
  revision: ``config.json``, ``model.safetensors`` and the model card ``README.md``;
* with ``--verify-official``, the original FAIR checkpoint ``dinov2_vits14_pretrain.pth`` from
  ``dl.fbaipublicfiles.com``, and checks that every tensor of the conversion equals the original
  tensor it was converted from (the qkv projection is split into query/key/value).

Every file is checked against the sha256 pinned below before it is used; an existing file with a
different hash is left untouched and the script stops. Nothing is loaded from the network at run
time: the TASK-063 runner reads only these local files.

    uv run --no-sync python scripts/fetch_dinov2.py --verify-official
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "third_party" / "dinov2-small"
HF_REPOSITORY = "facebook/dinov2-small"
HF_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
HF_FILES = {
    "config.json": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
    "model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
    "README.md": "4c20dca454a8e5c670e8de5c7e6040f512aeca5438516f7623eedc4e3b00599c",
}
OFFICIAL_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
OFFICIAL_FILE = "official/dinov2_vits14_pretrain.pth"
OFFICIAL_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, path: Path, want: str) -> None:
    if path.exists():
        got = sha256(path)
        if got != want:
            raise SystemExit(f"{path} exists with sha256 {got}, not the pinned {want}; untouched")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as stream:
        while block := response.read(1 << 20):
            stream.write(block)
    got = sha256(partial)
    if got != want:
        partial.unlink()
        raise SystemExit(f"{url} has sha256 {got}, not the pinned {want}")
    partial.rename(path)


def official_key_map(official: dict, blocks: int) -> dict:
    """HF ``Dinov2Model`` state-dict key -> tensor from the original checkpoint.

    Mirrors ``transformers``' own ``convert_dinov2_to_hf`` mapping for a backbone without
    registers or SwiGLU (ViT-S/14)."""
    mapped = {
        "embeddings.cls_token": official["cls_token"],
        "embeddings.mask_token": official["mask_token"],
        "embeddings.position_embeddings": official["pos_embed"],
        "embeddings.patch_embeddings.projection.weight": official["patch_embed.proj.weight"],
        "embeddings.patch_embeddings.projection.bias": official["patch_embed.proj.bias"],
        "layernorm.weight": official["norm.weight"],
        "layernorm.bias": official["norm.bias"],
    }
    for i in range(blocks):
        src, dst = f"blocks.{i}.", f"encoder.layer.{i}."
        for part in ("weight", "bias"):
            qkv = official[f"{src}attn.qkv.{part}"]
            width = qkv.shape[0] // 3
            for j, name in enumerate(("query", "key", "value")):
                mapped[f"{dst}attention.attention.{name}.{part}"] = qkv[j * width : (j + 1) * width]
            mapped[f"{dst}attention.output.dense.{part}"] = official[f"{src}attn.proj.{part}"]
            mapped[f"{dst}norm1.{part}"] = official[f"{src}norm1.{part}"]
            mapped[f"{dst}norm2.{part}"] = official[f"{src}norm2.{part}"]
            mapped[f"{dst}mlp.fc1.{part}"] = official[f"{src}mlp.fc1.{part}"]
            mapped[f"{dst}mlp.fc2.{part}"] = official[f"{src}mlp.fc2.{part}"]
        mapped[f"{dst}layer_scale1.lambda1"] = official[f"{src}ls1.gamma"]
        mapped[f"{dst}layer_scale2.lambda1"] = official[f"{src}ls2.gamma"]
    return mapped


def compare_with_official(converted_path: Path, official_path: Path) -> dict:
    """Every converted tensor against the original one (exact equality)."""
    import torch
    from safetensors.torch import load_file

    converted = load_file(str(converted_path))
    official = torch.load(str(official_path), map_location="cpu", weights_only=True)
    blocks = 1 + max(int(k.split(".")[1]) for k in official if k.startswith("blocks."))
    mapped = official_key_map(official, blocks)
    unused = sorted(set(official) - _sources_used(blocks))
    missing = sorted(set(converted) - set(mapped))
    extra = sorted(set(mapped) - set(converted))
    differing, max_delta = [], 0.0
    for key in sorted(set(converted) & set(mapped)):
        a, b = converted[key], mapped[key].to(converted[key].dtype)
        if a.shape != b.shape or not torch.equal(a, b):
            differing.append(key)
            if a.shape == b.shape:
                max_delta = max(max_delta, float((a - b).abs().max()))
    return {
        "converted_tensors": len(converted),
        "official_tensors": len(official),
        "blocks": blocks,
        "compared": len(set(converted) & set(mapped)),
        "converted_without_official_source": missing,
        "mapped_without_converted_tensor": extra,
        "official_tensors_unused": unused,
        "differing": differing,
        "max_abs_delta_of_differing": max_delta,
        "identical": not (missing or extra or differing),
    }


def _sources_used(blocks: int) -> set:
    used = {
        "cls_token",
        "mask_token",
        "pos_embed",
        "patch_embed.proj.weight",
        "patch_embed.proj.bias",
        "norm.weight",
        "norm.bias",
    }
    for i in range(blocks):
        for part in ("weight", "bias"):
            for name in ("attn.qkv", "attn.proj", "norm1", "norm2", "mlp.fc1", "mlp.fc2"):
                used.add(f"blocks.{i}.{name}.{part}")
        used |= {f"blocks.{i}.ls1.gamma", f"blocks.{i}.ls2.gamma"}
    return used


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--verify-official", action="store_true")
    args = parser.parse_args()
    base = f"https://huggingface.co/{HF_REPOSITORY}/resolve/{HF_REVISION}"
    for name, want in HF_FILES.items():
        _download(f"{base}/{name}", args.destination / name, want)
    result = {
        "repository": HF_REPOSITORY,
        "revision": HF_REVISION,
        "files": {name: sha256(args.destination / name) for name in HF_FILES},
    }
    if args.verify_official:
        official = args.destination / OFFICIAL_FILE
        _download(OFFICIAL_URL, official, OFFICIAL_SHA256)
        result["official"] = {"url": OFFICIAL_URL, "sha256": sha256(official)}
        result["official_equivalence"] = compare_with_official(
            args.destination / "model.safetensors", official
        )
        if not result["official_equivalence"]["identical"]:
            print(json.dumps(result, indent=1))
            return 1
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
