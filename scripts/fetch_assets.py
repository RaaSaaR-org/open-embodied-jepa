"""Fetch the pinned official G1/Dex3 assets without modifying an existing checkout."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def fetch() -> Path:
    manifest = json.loads((ROOT / "assets/manifest.json").read_text())
    target = ROOT / manifest["local_path"]
    revision = manifest["revision"]
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "--filter=blob:none", "--no-checkout", manifest["source"], str(target))
        git("-C", str(target), "sparse-checkout", "init", "--cone")
        git("-C", str(target), "sparse-checkout", "set", "unitree_robots/g1")
        git("-C", str(target), "checkout", "--detach", revision)
    if git("-C", str(target), "rev-parse", "HEAD") != revision:
        raise RuntimeError(
            f"Existing {target} has another revision; preserve it and move it aside."
        )
    if git("-C", str(target), "status", "--porcelain"):
        raise RuntimeError(f"Existing {target} has local modifications; refusing to use them.")
    for relative, expected in manifest["sha256"].items():
        path = target / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Asset hash mismatch: {path}")
    return target / manifest["scene"]


if __name__ == "__main__":
    print(fetch())
