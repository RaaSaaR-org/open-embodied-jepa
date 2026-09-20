# LeRobot v3 data and canonical sequences

`embodied_jepa.data` implements a local LeRobot v3 storage profile with lossless PNG images embedded in Parquet. It is actual Parquet with LeRobot feature, task, and episode metadata. It is not an NPZ container advertised as LeRobot. The supported upstream reader is **LeRobot v0.4.4, commit `8fff0fde7c79f23a93d845d1a50e985de01f8b8a`**. Our loader requires the additional JEPA manifest to establish state/action identity, executed-action semantics, provenance, and splits; importing an arbitrary third-party LeRobot dataset without those facts is not supported.

The official [v3 documentation](https://huggingface.co/docs/lerobot/v0.4.4/lerobot-dataset-v3) describes indexed Parquet storage. The pinned [feature conversion and loading code](https://github.com/huggingface/lerobot/blob/8fff0fde7c79f23a93d845d1a50e985de01f8b8a/src/lerobot/datasets/utils.py) supports `image` features backed by Hugging Face `datasets.Image`. The pinned [metadata implementation](https://github.com/huggingface/lerobot/blob/8fff0fde7c79f23a93d845d1a50e985de01f8b8a/src/lerobot/datasets/lerobot_dataset.py) reads `tasks.parquet` with a task-name index; this supersedes the overview documentation's older `tasks.jsonl` description. No upstream implementation is copied into the project.

## On disk

```text
dataset/
  data/chunk-000/file-000.parquet
  data/chunk-000/file-001.parquet
  meta/info.json
  meta/tasks.parquet
  meta/episodes/chunk-000/file-000.parquet
  meta/jepa_manifest.json
  meta/jepa_splits.json          # after freezing
  meta/stats.json                # after fitting on training only
  meta/jepa_normalization.json
```

Each data file holds one completed episode. v3 episode metadata gives global frame offsets and file/chunk indices; readers must use it rather than assume filenames encode semantic episode IDs. One file per episode is a deliberate small-corpus tradeoff, compatible with the v3 indexing scheme. RGB is an Arrow struct with PNG `bytes` and null `path`, requiring no video decoder. The writer closes every Parquet file before returning; no separate `finalize()` is required for our `DatasetStore`.

The JEPA manifest includes hashes of all data and metadata files, exact state/action schemas, source/license provenance, physical action conversion metadata, robot identity, collection IDs, episode outcome, and arbitrary JSON collection metadata. `DatasetStore.manifest_hash` is the SHA-256 of that complete manifest, which every experiment records. Identical inputs and pinned writer versions reproduce identical hashes; changing a split or fitting statistics creates a new manifest hash. Record the final hash after splits and normalization, before training. Callers supply real camera/action calibration and collection-policy metadata; the library does not invent them.

The store is a **single-writer** local API. Opening verifies all recorded hashes; reading verifies the episode shard and its contents. Append validates current hashes and refuses duplicate IDs, schema changes, or an existing uncommitted shard. JSON replacement is atomic, but the collection of Parquet/metadata writes is not a multi-file transaction: interrupted writes fail validation and require explicit recovery or a fresh dataset version. Parallel append and editing files behind a live store are unsupported. Split freezing seals further appends.

## T actions require T+1 observations

An `Episode` supplies T+1 RGB/state/mask/timestamp snapshots and T **actually executed** 14D normalized actions. Optional `raw_actions [T,14]` records the corresponding physical EE increments and absolute grasp synergy targets; `robot_states` retain physical units defined by `StateSchema`. Collectors must use `ExecutionResult.applied_action`, exclude rejected/stopped commands, and provide the embodiment's conversion manifest. Physical action masks distinguish unavailable raw commands from observed zeroes.

| Stored field | Semantics |
| --- | --- |
| `observation.images.<camera>` | Lossless uint8 RGB, one complete image per row |
| `observation.state`, `observation.state_mask` | Physical ordered state and explicit validity |
| `action`, `action_valid` | Executed action at row t leads to row t+1; false only on the final observation |
| `action.physical`, `action.physical_valid` | Raw physical command where available; masked otherwise |
| `timestamp` | LeRobot float32 episode-relative seconds |
| `observation.timestamp` | Exact float64 source-clock seconds for canonical loading |
| `next.done` | Action on this row ends collection; true on the final real transition |
| `next.terminated`, `next.truncated` | Separate task/environment termination versus collection/time-budget truncation |
| `index`, `episode_index`, `frame_index`, `task_index` | Global, episode-local and task indexing |

The final observation has a zero placeholder action with `action_valid=False`; it is **never** interpreted as an executed command or sampled as a transition start. Upstream LeRobot can read this row, so third-party policy-training code must honor `action_valid`. Our loader always excludes it. Our `SequenceBatch.terminated` is a sequence boundary marker for either termination or truncation; the stored episode and Parquet fields preserve the distinction for analysis.

An episode must have exactly one terminal or truncation reason. Interior terminal boundaries, missing cameras/frames, invalid states/actions, changed joint ordering, nonmonotonic times, and dropped sampling intervals fail. The frame interval must match `1/fps` within relative tolerance `1e-3` plus absolute `1e-6` seconds; this profile expects synchronized simulation collection rather than accepting irregular physical-camera streams silently. Every canonical sequence remains inside a single episode. Data arrays use the contract dtypes; LeRobot's tensor conversion can round scalar float64 timestamps to its default float32, while the project loader preserves the stored float64 clock.

State dimension comes from the schema, including named 86D qpos/qvel when supplied by the G1 adapter. The software fixture has synthetic state fields and never pretends to be a robot recording. The tests cover 1D, 2D, and 86D storage to ensure ordering/shape handling is independent of the robot.

## Collection and training API

```python
from embodied_jepa.data import DatasetStore, deterministic_fixture

episode = deterministic_fixture()  # software-only fixture, not physics evidence
store = DatasetStore.create(
    "data/example", fps=20, state_schema=episode.state_schema,
    action_manifest={"physical_scale": [0.01] * 14, "purpose": "software fixture only"},
    provenance={"source": "procedural fixture", "license": "Apache-2.0"},
    robot_type="software_fixture",
)
store.write_episode(episode)
reloaded = store.read_episode(episode.episode_id)
transition = reloaded.sequence(start=0, horizon=1)
```

For real collection, construct `Episode` from synchronized simulation observations and execution responses. IDs include episode, session, task text, object, and container. After collecting at least three independent non-held-out sessions:

```python
store.freeze_splits(seed=0, heldout_combinations=(("apple", "plate"),))
normalization = store.fit_normalization()
batch = store.sample_batch(batch_size=8, horizon=3, split="train", seed=42)
for window in store.windows(horizon=3, split="val"):
    pass  # validation only; do not update model or normalization
print(store.manifest_hash)
```

The default ratio is 80/10/10 with at least one session each for validation and test. Any session containing a reserved object/container pair goes wholly into `holdout`, including its other episodes. Remaining session names are sorted, seeded, and assigned as complete groups. With a small number of sessions, actual proportions differ from the target. Episode order does not control membership. Test goals/thresholds and reserved combinations still need to be declared before the final experiment; splitting alone cannot establish scientific independence if a collector reuses sessions or scenes under new IDs.

`meta/jepa_splits.json` is authoritative membership. `info.json` exposes the storage range as `all`, not a misleading all-data `train` range. To use the upstream reader for one split, translate its episode IDs to `episode_index` and pass `episodes=[...]` explicitly. `fit_normalization()` has no arbitrary-split argument: it fits frozen training episodes only and excludes masked fields per dimension. It stores valid counts, neutral mean=0/std=1 for never-observed optional fields, and explicit fit episode IDs. Models must still consume the validity mask. Image preprocessing remains inside model adapters.

## Validation

Install the `data` extra plus development dependencies for the local suite. The optional official-reader check additionally needs the `data-compat` and `learning` extras and a clean checkout of the pinned LeRobot source:

```sh
.venv/bin/pytest -q tests/test_data.py
git clone --filter=blob:none --no-checkout https://github.com/huggingface/lerobot.git third_party/lerobot
git -C third_party/lerobot checkout --detach 8fff0fde7c79f23a93d845d1a50e985de01f8b8a
LEROBOT_SOURCE=third_party/lerobot .venv/bin/pytest -q tests/test_data.py
```

Use an existing matching checkout instead of cloning over it. The compatibility test checks its revision, imports the actual upstream `LeRobotDataset`, reads two episodes and task mappings offline, decodes images, and compares state/action/final-row values. It does not mock the official reader or test only our own round trip. Compatibility is scoped to the pinned reader, PNG profile, and the recorded environment; no video import, Hub upload, other LeRobot version, or physical G1 dataset is claimed by these fixtures.
