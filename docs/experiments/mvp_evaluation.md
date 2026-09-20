# Frozen MVP comparison protocol

Declared before any final model evaluation. This is a bounded Mac research MVP validation, not a target success-rate guarantee. Development reaching and controller experiments remain separate and retain their negative outcomes.

## Question and cohort

Can the shared image-goal CEM/MPC runtime execute both trained model adapters on the same held-out Apple → Plate composition, and what physical task success and compute does each achieve?

Run 50 fixed resets, seeds 20000–20049, for each of three training seeds (0, 1, 2) and each model. Each training seed has its own checkpoint selected only on validation data. Report each seed separately; repeated training seeds share the same physical resets and are not 150 independent environments. Hold and bounded-random controls each run the same 50 resets once. No tuning, additional training, checkpoint selection or reset selection may use these results. Any later change requires a new protocol/version.

| Factor | Seen in training | Final test |
| --- | --- | --- |
| Objects | Procedural cube, apple and banana | Same procedural apple geometry/color |
| Containers | Target, bowl and plate | Same procedural plate geometry/color |
| Pairings | Cube→target, banana→bowl, apple→bowl, cube→plate | Apple→plate, excluded from all training/validation corpora |
| Appearance | Fixed procedural materials, camera and lighting | Same; no unseen-appearance claim |
| Robot | Pinned G1 with dual articulated Dex3, fixed pelvis | Same simulation asset/control configuration |
| Position distribution | Recorded development reset distributions | Object (.34, −.18), plate (.49, −.09), independent uniform ±.006 m offsets |

The final reset distribution matches the manipulation pilot; the supplementary release corpus uses a closer plate center (.45, −.10). Both distributions are explicitly recorded. This tests one held-out object/container pairing, not unseen objects, general appearance robustness, or hardware transfer.

`python -m embodied_jepa.goals --output data/goals/mvp-v0 --episodes 50 --seed-start 20000 --task apple_to_plate` seals RGB goal hashes, exact reset coordinates, seeds, thresholds, assets, action manifest and relevant runtime source hashes before model execution. Goals depict an independently reset and settled object supported on the plate. Goal images are inference inputs only, never training/normalization/validation samples. Runtime validates the frozen manifest before a run.

## Common controller and scoring

Both models use CPU inference with four Torch threads, canonical 14D actions, horizon 4, 64 CEM samples, 3 iterations and 8 elites, a 5 s per-replan deadline, and at most 805 commands at 20 Hz. The left arm remains fixed/open; right translation/rotation candidates are bounded to ±0.5 normalized units, right grasp to [−1, 1]. These bounds and the whole configuration are shared; changing `world_model.backend` selects the corresponding adapter/checkpoint without planner or task changes. Architecture/preprocessing/training compute are reported separately.

The ordered scorer requires reach/contact, contact-supported lift ≥.05 m, transport within .04 m of plate center while lifted/contacting, then supported release with speed ≤.1 m/s, height tolerance .012 m and no hand contact for .15 s. Reach-only tolerance is .025 m. All limits are pinned in the manifest. Safety stops, rejected commands, exceptions, unreachable goals, deadline misses and timeouts count as attempted failures. No simulator object state enters model encoding or planner cost; privileged truth is used for reset generation and scoring only.

Record full/stage outcomes, Wilson 95% intervals per training seed, termination/limit reasons, actual executed/requested actions, plan p50/p95, candidate throughput, replans and process peak RSS. Repeated-seed pooled descriptive counts must not be presented as independent-trial confidence intervals. Forbidden-contact classification is currently unavailable: record null with reason, not zero collisions. Contact physics and safety guards do not establish physical robot safety.

## Budgets and evidence

Final model evaluation and controls run serially with a shared 1,800 s true wall budget, counting host suspension. Stop launching new runs when exhausted; retain every partial run and explicitly report planned versus completed attempts. A timing budget failure is incomplete evaluation, never a successful reduced cohort. Keep each run's JSON/schema validation, source/config/data/split/checkpoint/action/goal hashes, dependency snapshot, initial/goal/final images and sampled rollouts. No final results exist when this protocol is declared.

Training follows [mvp_final.md](mvp_final.md). A checkpoint without the preregistered noncollapse eligibility is a failed training attempt and cannot be substituted with an unqualified latest checkpoint. Full research success is distinct from an implemented runtime and a completed negative benchmark; the acceptance report states both.

## Known controller-model mismatch declared before evaluation

CEM scores canonical normalized candidate commands before the embodiment applies joint-rate clipping. This is especially relevant for absolute grasp targets: a request to close from −1 to +1 may be accepted as only about −.82 in the first control interval. Training records the applied target. The planner currently does not project its entire candidate sequence through these state-dependent actuator limits, so predicted and executed grasp transitions can differ. Requested/applied traces expose the mismatch. Both backends face the same limitation; resulting success rates characterize this bounded runtime, not the best achievable control from either representation. Feasible candidate projection is a concrete follow-up, requiring a new protocol rather than post-result tuning.
