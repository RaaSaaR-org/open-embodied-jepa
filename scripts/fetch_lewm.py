"""Fetch optional MIT-licensed upstream model source at the audited revision."""

import argparse
import subprocess
from pathlib import Path

REVISION = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"


def fetch(destination):
    if destination.exists():
        revision = subprocess.check_output(
            ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(destination), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip()
        if revision != REVISION or dirty:
            raise SystemExit(
                "Existing checkout differs from audited clean revision; left untouched."
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/lucas-maes/le-wm.git", str(destination)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", REVISION], check=True
        )
    print(f"LeWM source ready: {destination} @ {REVISION}; retain its MIT LICENSE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("third_party/le-wm"))
    fetch(parser.parse_args().destination)
