"""Report installed package versions/licenses for reproducibility and license review."""

import importlib.metadata
import json


def inventory():
    rows = []
    for dist in importlib.metadata.distributions():
        metadata = dist.metadata
        license_text = metadata.get("License-Expression") or metadata.get("License")
        classifiers = metadata.get_all("Classifier") or []
        rows.append(
            {
                "name": metadata["Name"],
                "version": dist.version,
                "license": license_text,
                "license_classifiers": [c for c in classifiers if c.startswith("License ::")],
            }
        )
    return sorted(rows, key=lambda r: r["name"].lower())


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2))
