"""Write the Isaac v2 scene manifest and the onboard camera manifest from ``MuJoCoSimulation``.

Development tooling (TASK-025). Run on the host with the project environment. MuJoCo is the
single authority for geometry, masses, friction and the camera; the PhysX mapping and the
render settings are the constants in ``embodied_jepa.isaac_scene``:

    uv run --no-sync python scripts/isaac/write_scene_manifests.py \\
        --scene configs/isaac/apple_to_plate_v2_scene_v1.json \\
        --camera configs/isaac/onboard_camera_v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_jepa import isaac_scene
from embodied_jepa.apple_to_plate_v2 import apply_v2_scene
from embodied_jepa.simulation import MuJoCoSimulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="replace existing files")
    args = parser.parse_args()
    for path in (args.scene, args.camera):
        if path.exists() and not args.force:
            raise SystemExit(f"refusing to overwrite {path}")
    sim = MuJoCoSimulation(width=112, height=112, render=False)
    apply_v2_scene(sim.model)
    manifests = {
        args.scene: isaac_scene.build_scene_manifest(sim),
        args.camera: isaac_scene.build_camera_manifest(sim),
    }
    sim.close()
    for path, manifest in manifests.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{path} sha256(canonical json) {isaac_scene.canonical_sha256(manifest)}")


if __name__ == "__main__":
    main()
