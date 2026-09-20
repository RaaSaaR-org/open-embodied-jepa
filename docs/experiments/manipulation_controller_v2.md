# Early release controller: frozen v2 development probe

Declared before execution: the v1 original controller reaches the container at approximately command450, then holds full closure until565. Both original trials stopped at command549 during transfer, before lowering. Contact-lift and transport were already scored by approximately440. The cube slowly slips during this unnecessary hold and the compressed thumb subsequently accelerates beyond the unchanged5rad/s guard.

Hypothesis: shortening transfer to60 commands and gradually opening at the high transfer pose avoids this prolonged compression. The cube must then fall onto the container and satisfy the existing ordered support/rest scorer. This tests release from a short height, not gentle placement or force control.

Four trials are frozen: cube→plate then cube→target, each with opening ramp0.08 then0.04 normalized synergy units per accepted command. Seeds3000–3003, object(.34,−.18), container(.45,−.10). Original orient/descend/close/lift unchanged; transfer60, release_high100, lower_open100, retreat80 commands. Existing physics, gains, torque limits, velocity guards and root `AppleToPlateTask` thresholds unchanged. No Apple→Plate, retries, or tuning after outcomes. Maximum300true-wall seconds,270for execution plus30finalization.

```sh
.venv/bin/python scripts/spikes/grasp_probe.py --controller-v2 --output outputs/manipulation_controller_v2
```

Every outcome and complete20Hz RGB/state/applied-action transition is retained, with exact source snapshots and a pre-execution plan. Outputs never overwrite earlier artifacts. This is privileged simulated controller development, not learned control or final benchmark evaluation.

## Probe result and frozen release collection

All four fixed trials succeeded in35.43true-wall seconds: real contact-lift and transport followed by release and sustained supported rest, with745accepted commands each and no safety stops. Final supported dwell was13.30s,12.80s,13.30s and12.90s. Both candidates passed; freeze the first predeclared candidate, opening_ramp0.08, for subsequent collection. This demonstrates mechanical reachability at these two cube layouts, not reliability or Apple→Plate learned performance.

The first reach/grasp/transport timestamps were11.00/14.25/21.75simulated seconds for all trials. Supported release first passed at24.10,24.60,24.10 and24.50seconds respectively. Valid reset separation, real collisions, existing torque/velocity guards and scoring thresholds were preserved. Verified all2980recorded transitions haveT+1 RGB/state observations and50ms timestamps, and every source snapshot matches its declared hash.

Plan SHA-256: `daa77885932131739b905a6e23e2a3414e289002a041a5460ee93328833e756b`.
Report SHA-256: `b55b0b6a3b558765afc4ea91316ddfdb9b83cf0602053ebde2786693d475c2cf`.
Artifacts: `outputs/manipulation_controller_v2/`, including full traces and original source snapshots. Diagnostic trials are excluded from final training datasets.

Before new collection, freeze24attempts with seeds3000–3023, round-robin cube→target, banana→bowl, apple→bowl, cube→plate, original cube coordinates(.34,−.18) and container(.45,−.10), independent ±.006m xy jitter,745command cap and300true-wall seconds including30seconds finalization reserve. Apply the same `EarlyReleaseOracleManipulationPolicy(opening_ramp=.08)` to every pair. Invalid reset overlaps count as failed attempts, with no resampling. Keep every complete prefix of at least8transitions, including failures. Freeze session-grouped80/10/10splits and fit normalization on training only. This is a new corpus; older data remain immutable.

```sh
.venv/bin/python -m embodied_jepa.manipulation_collection --output data/manipulation-release-v1 --episodes 24 --seed 3000 --max-seconds 300 --max-commands 745 --policy early_release_v1 --container-center .45 -.10
```

Collector validation: six tests pass, covering policy dispatch, invalid-reset accounting, observed-prefix preservation after RGB failure, budget and interruption accounting, clock suspension, and registered source snapshots.
