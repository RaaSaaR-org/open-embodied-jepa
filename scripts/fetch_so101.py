"""Fetch the pinned SO-101 MuJoCo model (MuJoCo Menagerie) without modifying an existing checkout.

The revision, sparse path, licence and per-file SHA-256 live in ``assets/so101_manifest.json``;
the checkout goes to the ignored ``third_party/mujoco_menagerie_so101`` (SO-101 S1, TASK-097).
The directory is SO-101-specific because it is a sparse checkout of one subtree at one pin: a
later Menagerie model at another revision gets its own directory instead of colliding here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets/so101_manifest.json"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def verify(target: Path, manifest: dict) -> None:
    """The sparse directory holds exactly the manifest's files, each with its SHA-256."""
    directory = target / manifest["sparse_path"]
    present = {str(path.relative_to(target)) for path in directory.rglob("*") if path.is_file()}
    if present != set(manifest["sha256"]):
        raise RuntimeError(
            f"{directory} file set differs from the manifest: "
            f"missing {sorted(set(manifest['sha256']) - present)}, "
            f"unexpected {sorted(present - set(manifest['sha256']))}"
        )
    for relative, expected in manifest["sha256"].items():
        path = target / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Asset hash mismatch: {path}")


def fetch() -> Path:
    manifest = json.loads(MANIFEST.read_text())
    target = ROOT / manifest["local_path"]
    revision = manifest["revision"]
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "--filter=blob:none", "--no-checkout", manifest["source"], str(target))
        git("-C", str(target), "sparse-checkout", "init", "--cone")
        git("-C", str(target), "sparse-checkout", "set", manifest["sparse_path"])
        git("-C", str(target), "checkout", "--detach", revision)
    if git("-C", str(target), "rev-parse", "HEAD") != revision:
        raise RuntimeError(
            f"Existing {target} has another revision; preserve it and move it aside."
        )
    if git("-C", str(target), "status", "--porcelain"):
        raise RuntimeError(f"Existing {target} has local modifications; refusing to use them.")
    verify(target, manifest)
    return target / manifest["robot"]


if __name__ == "__main__":
    print(fetch())
