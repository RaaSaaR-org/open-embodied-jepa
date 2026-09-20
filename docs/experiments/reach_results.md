# Development reach results — negative evidence retained

All measurements below are five-reset **development** runs, not sealed generalization tests or estimates of deployment reliability. Goals are independently oracle-reachable; the same CEM budget, physical action bounds, task thresholds and seeds10000–10004 serve both models. Apple→plate was not used. Artifact summaries/checkpoint hashes are versioned in `benchmarks/manifests/reach-development-results.json`; complete logs/images remain local under `outputs/<run_id>/`.

| Run | Reach successes / attempts | Interpretation |
| --- | --- | --- |
| Scripted oracle |5/5| Confirms these development goals are reachable |
| Open-hand hold |0/5| Stationary baseline |
| Bounded random |0/5| All five stopped by control guards |
| Native v0 raw-loss selector |3/5| Selected representation collapsed; does not pass research-quality gate |
| LeWM v0 |0/5| Noncollapsed, action-dependent prediction did not yield successful control |
| Native v1 guarded relative selector |0/5| Online latent guard passed; EMA-target metric mismatch invalidated the apparent prediction improvement |
| LeWM v1 |0/5| Same selected training step as v0; all episodes retained |
| Native v2 target-space selector |1/5| Noncollapsed online and target representations; prediction improves on persistence and shuffled actions |
| LeWM v2 target-space selector |1/5| Same corrected controls; limited reach progress |

For 1/5 the Wilson 95% interval is [0.036, 0.624]; for 3/5 it is [0.231, 0.882]; for 0/5 it is [0, 0.434]. Small cohorts and overlapping intervals preclude strong success-rate comparisons. Timing is development instrumentation while other project work may have been running; it is not isolated throughput characterization.

The 100-episode reach corpus has4,836 executed transitions. Failed/shortened motion prefixes are retained; only right-arm XYZ varies. It is a reach pilot, not a complete manipulation corpus. The versioned coverage report includes per-axis ranges, state ranges, timing, storage and group splits. Action/state sensors are synchronized at20Hz; raw physical applied actions are preserved. No apple→plate pairing appears in training.

Nativev0's raw MSE selector favored a tiny-variance representation. V1 added a declared online collapse guard and divided prediction error by persistence. Subsequent review showed that native persistence compared the **online current embedding to EMA future targets**, while a meaningful persistence control stays entirely in the target space. On the selectedv1 validation cohort, reported mixed-space persistence was0.460379 but target-space persistence was0.0000453 against prediction0.005207. The claimed relative advantage was therefore misleading. V0/v1 artifacts and their source implementations are retained in Git; they are negative experiments, not quietly replaced successes.

V2 corrects target-space diagnostics and requires both online and target collapse guards before checkpoint selection. Its 3,000-update hypothesis was declared before execution. Native selected step 1,500: prediction MSE 0.01884, target-space persistence 0.28441, shuffled-action MSE 0.08012, online/target standard deviations 0.845/0.856. LeWM selected step 2,100: prediction 0.00946, persistence 0.05448, shuffled actions 0.01611, standard deviation 0.848. Both have zero collapsed dimensions by the declared threshold. These within-model controls pass; raw latent errors are not compared across models.

Both v2 policies reach seed 10003 and are stopped by control guards on the other four resets. This is weak development progress over the 0/5 hold/random controls, with wide overlapping intervals; it does not establish reliable learned control or full manipulation. Native summary generation initially failed on tuple-valued configuration bounds; after correcting schema validation, the summary was recovered from the five original saved episode records without rerunning or excluding episodes. The summary records this recovery. V2 checkpoints load against preserved commit `772b5ab`; the subsequent MPS diagnostic fix deliberately changes the checkpoint code identity.
