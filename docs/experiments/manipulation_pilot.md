# Manipulation pilot v0

This is privileged scripted collection on four development pairs, not learned manipulation or held-out evaluation. The protocol was fixed before collection: request 100 episodes with unique seeds 1000–1099, cycle cube→target, banana→bowl, apple→bowl and cube→plate, and exclude apple→plate entirely. Use the unmodified `OracleManipulationPolicy`, up to 805 commands per episode, one synchronized observation per 50 ms action, and a 600-second total wall budget with 30 seconds reserved for sealing/reporting.

Reset object XY is `(0.34,-0.18)` m and container XY is `(0.49,-0.09)` m, each with independent uniform ±6 mm jitter generated from the episode seed. The current simulator rejects overlapping starts. Initial object clearance above the table is explicit; settling occurs during the recorded initial approach, with no unrecorded actions or subsampled transitions. Policy/controller parameters and success rules do not change during the run.

Success and stage coverage use `AppleToPlateTask` for every development pair. The dataset keeps unsuccessful episodes and complete observed prefixes of at least eight transitions. Rejected commands never become transitions; an applied command lacking a following RGB/state snapshot is counted as an incomplete execution and omitted while preserving the preceding valid prefix. Every attempted episode remains in the success denominator; unattempted episodes after the wall budget are listed separately.

```sh
.venv/bin/python -m embodied_jepa.manipulation_collection --output data/manipulation-pilot-v0 --episodes 100 --seed 1000 --max-seconds 600
```

The collector writes the frozen plan before recorded trials, per-attempt JSONL, and a final report. LeRobot v3 PNG/Parquet episodes contain applied normalized actions, corresponding physical command values, named robot-only state/masks, timestamps, all 20 Hz camera frames, phase labels, ordered scoring evidence and controller provenance. Privileged reset/scoring truth stays in episode metadata, outside canonical model input fields. Unique reset sessions are split 80/10/10; normalization fits only the training sessions. Existing output directories are never overwritten.

Source-file, asset, action-manifest, plan and final dataset hashes identify the exact artifacts; software versions and machine identity are recorded. The source checkout may contain uncommitted implementation work, so source hashes supplement its Git revision. Final measured counts and limitations are appended after this bounded run; no performance outcome is assumed in advance.

## Actual outcome and clock incident

The sealed corpus contains **66 stored episodes and 27,808 complete 20 Hz transitions** (23.17 simulated minutes), using 256,733,171 bytes at sealing. Stored coverage is 17 cube→target, 17 banana→bowl, 16 apple→bowl and 16 cube→plate episodes. The split contains 52 training, seven validation and seven test reset sessions, with zero held-out-pair episodes. Training-only normalization, file hashes, source snapshots, split disjointness, frame/action alignment and representative canonical episode reads were verified.

The root ordered scorer recorded reach in all 66 stored episodes and contact-grasp plus transport in 33; none completed place/release. The 66 recorded control stops were 62 measured-joint-velocity stops, two IK failures and two joint-target-rate failures. Phase coverage is 8,580 orient, 5,280 descend, 2,932 close, 5,601 lift, 5,353 transfer and 62 lower transitions. There are **no release or retreat examples**. This is a shared corpus of real simulated manipulation attempts and useful failure prefixes, not a corpus of successful full-task demonstrations. The fixed oracle was not tuned during collection and safety thresholds were not relaxed.

The requested wall budget was **not met by this original run**. A large host wall-clock/monotonic-clock divergence appeared during collection: the Python monotonic clock was `mach_absolute_time()`, which excludes macOS suspension. Host suspension is the likely explanation; wall-clock adjustment cannot be ruled out from these measurements alone. At detection, the collector had used roughly 337 seconds across recorded active trials but over 30 minutes of civil elapsed time. It was explicitly interrupted, then its valid corpus was sealed; the report records 1,855.99 seconds through sealing instead of claiming a ten-minute run.

The interruption arrived while serializing episode 67 (seed1066, apple→bowl), after an observed prefix of at least eight transitions existed in memory. That prefix was lost; its exact count and final physical outcome are unavailable. The interrupted attempt is recorded explicitly and contributes to the attempted denominator: **zero verified full successes among 67 attempts**, comprising 66 scored failures and one interrupted/unscored attempt. The remaining 33 planned episodes were unattempted. No missing episode was fabricated or silently omitted.

The current collector fixes this operational defect: budget checks use the maximum of wall-clock and monotonic elapsed time, counting suspension while guarding against backward clock adjustments. SIGINT requests a graceful boundary stop, preserving the current complete observed prefix and writing the report before exit. Five collector tests cover deterministic held-out exclusion, complete-prefix retention after missing RGB, failure denominators/splits, global-budget stopping, suspension/backward-clock behavior, and interruption preservation. These fixes were made **after** v0 collection; no second collection run or mutation of v0 recordings occurred.

The exact executed pre-fix source files were hash-verified and preserved under `data/manipulation-pilot-v0/meta/collection_source/` before fixing the clocks. This supplements the checkout revision and makes the historical run distinguishable from the corrected collector. Final dataset manifest hash:

```text
8152aa39ce50de0bc37d639051836310e7c2056be932c4d33b00ca58c62d19ec
```

Detailed local evidence is in `collection_plan.json`, `attempts.jsonl`, `collection_report.json`, `meta/jepa_manifest.json`, `meta/jepa_splits.json` and `meta/jepa_normalization.json` beneath the dataset root. These generated artifacts remain ignored and were not uploaded. A future collection must use a new output directory and protocol version; this pilot cannot support a claim of reliable placement or held-out success.
