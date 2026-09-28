"""Write the name-addressed G1 + Dex3 joint manifest that ``IsaacTransport`` consumes.

Development tooling (TASK-025). Run on the host with the project environment; the manifest
is derived from ``MuJoCoSimulation`` (pinned MJCF + the transport's PD gains), so MuJoCo stays
the single authority for joint order, limits, torque ranges, gains and passive damping:

    uv run --no-sync python scripts/isaac/write_joint_manifest.py \\
        --output configs/isaac/g1_dex3_joint_manifest_v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_jepa.isaac_transport import joint_manifest_from_mujoco, manifest_sha256
from embodied_jepa.simulation import MuJoCoSimulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    sim = MuJoCoSimulation(render=False)
    manifest = joint_manifest_from_mujoco(sim)
    sim.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.output} sha256(canonical json) {manifest_sha256(manifest)}")


if __name__ == "__main__":
    main()
