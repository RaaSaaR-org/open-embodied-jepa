# Dataset and collection plan

No robot recordings or dataset access have been provided. TASK-001 records availability; TASK-010 produces the first versioned corpus. The same corpus and sample interpretation must serve both backends.

## Acquisition

1. Inventory existing G1 demonstrations and permissions, camera streams, state fields, actual executed actions, timestamps, and robot variant.
2. If suitable data is unavailable, collect G1/Dex3 MuJoCo episodes locally using scripted controllers or teleoperation. Keep held-out object/task combinations out of training.
3. Build a tiny deterministic synthetic fixture for loader/contract tests. It verifies software only and does not count as manipulation evidence.
4. Begin with a proposed pilot of 100–300 varied simulation episodes, collected serially or in a measured small worker pool on the Mac, across reach, grasp, transport, placement, and failures. This is a sizing hypothesis; expand only after coverage and learning-curve checks. Record actual frame counts and storage requirements after the pilot.

Collection must include action diversity, unsuccessful behavior, varied object/goal poses, and reset metadata. Do not train only on successful expert trajectories. Sample broad but valid actions inside calibrated workspace constraints.

## Storage and schema

Use a pinned LeRobotDataset format/library revision; v3 is the initial candidate. Its official documentation describes tabular data, video, and metadata storage; test the selected version before claiming compatibility. [LeRobotDataset v3 documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

The canonical loader emits observations, robot states, normalized executed actions, next observations/states, timestamps, episode/task/object IDs, terminal status, and masks. Preserve raw states/actions and their conversion metadata for auditing. Do not pretend an image-only dataset contains G1 state or a compatible action space.

Validate timestamp monotonicity, camera/state alignment, frame availability, dropped frames, units, joint ordering, action bounds, and terminal windows. Set alignment tolerances from the verified control frequency. Reject corrupt samples with an explicit report. Missing optional sensors are masked; missing required fields fail validation.

## Splits and leakage

Use episode/session groups, never random frame splitting. Start with proposed 80/10/10 train/validation/in-distribution-test allocation, subject to enough independent sessions. Reserve a separate compositional holdout before collection: train on cube → target, banana → bowl, and other object → container combinations; test apple → plate without task-specific updates. Publish precisely which objects, containers, appearances, and pairings were seen.

Select model settings using validation only. Freeze test manifests and goal-image pools before final evaluation. Goal images express the desired outcome and may be supplied at test time; they must not enter training, normalization fitting, or hyperparameter selection. Avoid near-duplicate scenes or sessions leaking across splits.

## Manifest requirements

Record dataset ID/revision, provenance/license, file hashes, format/library versions, episode assignments, camera calibration/resolution, state/action schemas, physical scaling, control frequency, normalization statistics, collection policy, object identities, and exclusions. Manifests and split files are versioned; large recordings live outside Git. Both backends log the exact same manifest hash.
