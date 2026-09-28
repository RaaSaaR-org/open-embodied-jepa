"""Raw and canonical sha256 hashes of a converted USD package (needs only ``pxr``).

The MJCF importer embeds a random temporary directory name (``/tmp/tmpXXXXXXXX/``) in every
layer's ``doc`` string, so two conversions of the same MJCF are not byte-identical. The
canonical hash exports each layer to USDA text with ``Sdf.Layer.ExportToString`` and
replaces that directory name with ``/tmp/<tmp>/`` before hashing, so it detects every other
difference. Raw hashes pin the exact file set that was used; canonical hashes check that a
reconversion is reproducible.

Usage inside the container: ``/isaac-sim/python.sh hash_usd.py <usd-package-dir> [...]``.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

TMP = re.compile(r"/tmp/tmp[A-Za-z0-9_]+/")


def tree_hash(files: dict[str, str]) -> str:
    return hashlib.sha256(
        "".join(f"{k}\0{v}\n" for k, v in sorted(files.items())).encode()
    ).hexdigest()


def hash_package(root: Path) -> dict:
    from pxr import Sdf

    raw, canonical = {}, {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = str(path.relative_to(root))
        raw[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix in (".usd", ".usda", ".usdc"):
            layer = Sdf.Layer.OpenAsAnonymous(str(path))
            text = TMP.sub("/tmp/<tmp>/", layer.ExportToString())
        else:
            text = path.read_bytes().decode("latin-1")
        canonical[rel] = hashlib.sha256(text.encode()).hexdigest()
    return {
        "files_sha256": raw,
        "tree_sha256": tree_hash(raw),
        "files_canonical_sha256": canonical,
        "canonical_tree_sha256": tree_hash(canonical),
        "definition": "tree = sha256 over sorted 'relpath\\0sha256\\n'; canonical = sha256 of "
        "Sdf.Layer.ExportToString() with /tmp/tmpXXXX/ replaced by /tmp/<tmp>/",
    }


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        h = hash_package(Path(arg))
        keys = ("tree_sha256", "canonical_tree_sha256")
        print(json.dumps({"package": arg, **{k: h[k] for k in keys}}))
