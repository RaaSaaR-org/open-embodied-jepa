"""Fetch optional noncommercial upstream source without installing or vendoring it."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "src/embodied_jepa/models/jepa_wms_source.json"
)


def fetch(destination):
    manifest = json.loads(MANIFEST_PATH.read_text())
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", manifest["repository"], str(destination)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", manifest["revision"]],
            check=True,
        )
    revision = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != manifest["revision"]:
        raise SystemExit("Existing source differs from the pinned revision; left untouched")
    for relative, expected in manifest["files"].items():
        if hashlib.sha256((destination / relative).read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Source integrity failure: {relative}; checkout left untouched")
    print(f"Optional CC-BY-NC-4.0 source: {destination} @ {revision}; retain its notices")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("third_party/jepa-wms"))
    parser.add_argument("--accept-noncommercial-source", action="store_true", required=True)
    fetch(parser.parse_args().destination)
