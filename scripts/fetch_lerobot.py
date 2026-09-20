"""Fetch the optional Apache-2.0 LeRobot reader at its audited source revision."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from embodied_jepa.data import LEROBOT_REVISION


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def fetch(destination: Path) -> Path:
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        git(
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "https://github.com/huggingface/lerobot.git",
            str(destination),
        )
        git("-C", str(destination), "checkout", "--detach", LEROBOT_REVISION)
    revision = git("-C", str(destination), "rev-parse", "HEAD")
    dirty = git("-C", str(destination), "status", "--porcelain", "--untracked-files=no")
    if revision != LEROBOT_REVISION or dirty:
        raise RuntimeError("Existing LeRobot checkout differs from audited source; left untouched")
    if not (destination / "LICENSE").is_file():
        raise RuntimeError("Pinned LeRobot checkout is missing its license")
    print(f"LeRobot reader ready: {destination} @ {revision}; retain its Apache-2.0 LICENSE")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("third_party/lerobot"))
    fetch(parser.parse_args().destination)
