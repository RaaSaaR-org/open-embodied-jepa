"""Assemble a new sealed corpus without changing any source split assignment."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from embodied_jepa.contracts import ContractError
from embodied_jepa.data import DatasetStore
from embodied_jepa.training import json_hash, source_identity


def assemble(sources, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dataset {output}")
    stores = [DatasetStore(path) for path in sources]
    if not stores:
        raise ContractError("at least one source corpus is required")
    hashes = [store.manifest_hash for store in stores]
    if len(set(hashes)) != len(hashes):
        raise ContractError("duplicate source corpus")
    reference = stores[0].manifest
    keys = ("fps", "robot_type", "state_schema", "action_schema", "action_manifest")
    splits = {name: [] for name in ("train", "val", "test", "holdout")}
    sessions, lineage = {}, []
    for store, digest in zip(stores, hashes, strict=True):
        manifest = store.manifest
        if any(manifest[key] != reference[key] for key in keys):
            raise ContractError("source state/action conversion/FPS/robot schemas differ")
        assigned = manifest.get("splits")
        if not assigned or set(assigned) != set(splits):
            raise ContractError("source must have frozen train/val/test/holdout assignments")
        partition = [name for names in assigned.values() for name in names]
        if len(partition) != len(set(partition)) or set(partition) != set(store.episode_ids):
            raise ContractError("source splits must partition every episode exactly once")
        rows = {row["episode_id"]: row for row in manifest["episodes"]}
        for split, names in assigned.items():
            for name in names:
                row = rows[name]
                if (row["object_id"].lower(), row["container_id"].lower()) == ("apple", "plate"):
                    raise ContractError("Apple→Plate episodes are excluded from the final corpus")
                session = row["session_id"]
                if session in sessions and sessions[session] != split:
                    raise ContractError(f"source session crosses split boundary: {session}")
                sessions[session] = split
                splits[split].append(f"{digest}:{name}")
        lineage.append(
            {
                "source_path": str(store.root.resolve()),
                "manifest_sha256": digest,
                "split_sha256": json_hash(
                    {"splits": assigned, "policy": manifest.get("split_policy")}
                ),
                "episode_prefix": digest + ":",
                "provenance": manifest["provenance"],
            }
        )
    if not splits["train"] or not splits["val"]:
        raise ContractError("assembled corpus needs nonempty train and validation partitions")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        assembled = DatasetStore.create(
            staging,
            fps=reference["fps"],
            state_schema=stores[0].state_schema,
            action_manifest=reference["action_manifest"],
            robot_type=reference["robot_type"],
            provenance={
                "producer": "assemble_mvp_data.py",
                "source": source_identity(),
                "license": "source licenses retained individually in sources.provenance",
                "sources": lineage,
                "split_assignment": "preserved; original session IDs; no refitting",
                "excluded_pair": ["apple", "plate"],
            },
        )
        for store, digest in zip(stores, hashes, strict=True):
            for name in store.episode_ids:
                episode = store.read_episode(name)
                assembled.write_episode(
                    replace(
                        episode,
                        episode_id=f"{digest}:{name}",
                        metadata=dict(episode.metadata)
                        | {"assembly_lineage": {"manifest_sha256": digest, "episode_id": name}},
                    )
                )
            refreshed = DatasetStore(store.root)
            if refreshed.manifest_hash != digest:
                raise ContractError("source manifest changed during assembly")
        assembled.freeze_split_assignments(
            splits,
            provenance={"method": "preserved_source_assignments", "sources": lineage},
        )
        assembled.fit_normalization()
        assembled.verify()
        if output.exists():
            raise FileExistsError(f"output appeared during assembly: {output}")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return DatasetStore(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    store = assemble(args.source, args.output)
    print(
        json.dumps(
            {
                "dataset": str(store.root),
                "manifest_sha256": store.manifest_hash,
                "episodes": len(store.episode_ids),
                "splits": {key: len(value) for key, value in store.manifest["splits"].items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
