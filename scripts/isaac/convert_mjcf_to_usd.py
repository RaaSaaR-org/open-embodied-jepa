"""Convert the project's pinned G1 + Dex3 MJCF to USD with the Isaac Sim MJCF importer.

Development tooling (not a gated experiment). Runs INSIDE the ``isaaclab_arena`` container
with Isaac Sim's Python, started by ``scripts/isaac/run_isaac.sh``; it is not part of the
``embodied_jepa`` package and adds nothing to ``uv.lock``.

The input is the unmodified, sha-pinned ``g1_29dof_with_hand.xml`` from
``assets/manifest.json`` (mounted read-only). The importer is ``isaacsim.asset.importer.mjcf``
(``MJCFImporter``, which wraps ``mujoco-usd-converter``). Options are fixed below and written
to ``conversion.json`` together with the importer/converter versions, the raw sha256 of every
output file and the canonical hashes of ``hash_usd.py``. Reruns are not byte-identical (the
importer embeds a random temp-dir name); compare ``canonical_tree_sha256`` instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hash_usd import hash_package  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mjcf", type=Path, required=True, help="pinned g1_29dof_with_hand.xml")
parser.add_argument("--expected_sha256", required=True, help="sha256 from assets/manifest.json")
parser.add_argument("--output", type=Path, required=True, help="new output directory")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

# Fixed importer options (recorded). import_scene=False: robot only, no PhysicsScene prim,
# so the asset can be spawned into an Isaac Lab scene. Collision geometry is the MJCF's own
# (no collision_from_visuals); meshes are not merged; self-collision off as in the importer
# default (the MuJoCo model's own contype/conaffinity filtering is not reproduced by this).
OPTIONS = {
    "import_scene": False,
    "merge_mesh": False,
    "debug_mode": False,
    "collision_from_visuals": False,
    "collision_type": "Convex Hull",
    "allow_self_collision": False,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


source_sha = sha256(args.mjcf)
if source_sha != args.expected_sha256:
    raise SystemExit(f"MJCF hash {source_sha} != pinned {args.expected_sha256}")
if args.output.exists() and any(args.output.iterdir()):
    raise SystemExit(f"refusing to overwrite non-empty {args.output}")
args.output.mkdir(parents=True, exist_ok=True)

app = AppLauncher(args).app

import importlib  # noqa: E402
import importlib.metadata as md  # noqa: E402

import omni.kit.app  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.asset.importer.mjcf")
app.update()

from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig  # noqa: E402


def ext_version(name: str) -> str | None:
    manager = omni.kit.app.get_app().get_extension_manager()
    ext_id = manager.get_enabled_extension_id(name)
    return ext_id or None


def main() -> None:
    # The importer writes next to its usd_path and needs the meshes beside the MJCF; copy the
    # read-only pinned input into a scratch directory inside the output (removed afterwards).
    work = args.output / "_source"
    shutil.copytree(args.mjcf.parent, work, ignore=shutil.ignore_patterns("*.png", "images"))
    if sha256(work / args.mjcf.name) != source_sha:
        raise SystemExit("copied MJCF hash changed")
    usd_root = args.output / "usd"
    config = MJCFImporterConfig(
        mjcf_path=str(work / args.mjcf.name), usd_path=str(usd_root), **OPTIONS
    )
    t0 = time.perf_counter()
    final = Path(MJCFImporter(config).import_mjcf())
    seconds = time.perf_counter() - t0
    shutil.rmtree(work)

    hashes = hash_package(usd_root)
    files = hashes["files_sha256"]
    versions = {}
    for module in ("mujoco_usd_converter", "mujoco", "usdex.core", "newton"):
        try:
            mod = importlib.import_module(module)
            versions[module] = {"version": getattr(mod, "__version__", None), "path": mod.__file__}
        except ImportError:
            versions[module] = None
    for dist in ("mujoco-usd-converter", "mujoco", "usd-core", "usd-exchange", "newton"):
        try:
            versions[f"dist:{dist}"] = md.version(dist)
        except md.PackageNotFoundError:
            versions[f"dist:{dist}"] = None
    record = {
        "source_mjcf": args.mjcf.name,
        "source_sha256": source_sha,
        "importer_extension": ext_version("isaacsim.asset.importer.mjcf"),
        "transformer_extension": ext_version("isaacsim.asset.transformer.rules"),
        "isaac_sim_version": Path("/isaac-sim/VERSION").read_text().strip(),
        "python_package_versions": versions,
        "options": OPTIONS,
        "main_usd": str(final.relative_to(usd_root)),
        "main_usd_sha256": files.get(str(final.relative_to(usd_root))),
        "main_usd_canonical_sha256": hashes["files_canonical_sha256"].get(
            str(final.relative_to(usd_root))
        ),
        **hashes,
        "conversion_s": seconds,
    }
    (args.output / "conversion.json").write_text(json.dumps(record, indent=2))
    summary = {k: v for k, v in record.items() if not k.startswith("files")}
    print("CONVERT_RESULT " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
