# Matched Apple→Plate branches v1: collection and audit results

The frozen experiment completed **528/528 branches across 66 roots**, adding
**8,448 actual transitions** with no rejection, runtime error, interruption or
missing attempt. The independent read-only audit passed with no findings, and
the preregistered action-contrast screen passed for **48/48 TRAIN roots and 18/18
VAL roots**. The corpus is ready for the separately registered model experiment.
This is data feasibility evidence, not learned manipulation success.

Protocol: [apple_branches_v1.md](apple_branches_v1.md). Reproducibility record:
[apple-branches-v1.json](../../benchmarks/manifests/apple-branches-v1.json).
Collection source: `662a1ab1092b4b4653f1145dc673dc001cc8cc22`.
Sealed dataset: `data/apple-branches-v1`, manifest SHA-256
`6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`.

Collection consumed **386.53 supervised true-wall seconds** and **370.71 worker
CPU seconds**, within the 600-second wall cap. No retry or extension occurred.
The environment inventory was captured after collection: macOS ARM64, Python
3.12.13, MuJoCo 3.13.0, NumPy 2.5.3. Its exact package inventory/hash is retained
in the manifest; it was not part of the original plan hash.

## Cohort and data integrity

The corpus retains all 32 original demonstrations and adds 528 branches, for
**560 episodes and 24,580 total transitions**. Frozen episode counts are TRAIN
410, VAL 147, TEST 3, HOLDOUT 0. Branches and their parent demonstrations remain in
the same original session and partition. TRAIN roots come from eight nominal
parents; VAL roots come from all three original perturbed parents. The audit
independently verified exact normalization membership in TRAIN.

All **32 original encoded episode payloads remain byte-identical**, including the
three TEST episodes. Their individual source/copy SHA-256 values are recorded
in the manifest. Neither collection nor audit decoded TEST images. The audit
only decoded the 11 selected TRAIN/VAL parents and their new branches.

All 66 sibling groups start from exactly identical RGB, proprioception, state
mask and timestamp tensors. Each root also matches its recorded parent RGB/mask/
time exactly, with the declared 1e-5 absolute proprio tolerance. Collector replay
checked applied actions to 2e-6 and all intermediate sensor states; repeated
restoration checks verified identical one-step continuation. This validates the
selected roots under the TASK036 joint-bound repair; it does not claim every
possible historical trajectory is unchanged.

The audit matched all 8,448 canonical applied/raw labels and T+1 sensor frames to
the durable command journals, checked requested branch transformations and the
20 Hz action interval, and verified recorded-continuation applied actions against
the parent. There were no uncertain or discarded prefixes. Full MuJoCo data,
controller targets, grasp history and ordered scorer state were restored between
branches; privileged state stayed inside collection and scoring.

## Observable action contrast

Each split/phase contains all eight declared alternatives. Whole-frame RGB RMS
uses [0,1] pixels; the preregistered distinction threshold is 1/255. Counts below
are unordered pairs of branches at a shared root, not independent trial counts.

| Split | Horizon | Distinct pairs / all pairs | Median RGB RMS | Median actual-action RMS |
|---|---:|---:|---:|---:|
| TRAIN | 8 | 1296 / 1344 | 0.07783 | 0.14463 |
| TRAIN | 16 | 1296 / 1344 | 0.10454 | 0.14920 |
| VAL | 8 | 492 / 504 | 0.07703 | 0.15812 |
| VAL | 16 | 493 / 504 | 0.10582 | 0.14518 |

All 66 roots had eight complete H16 branches and at least two visibly distinct
pairs, exceeding the gate of four complete branches and two distinct pairs in
75% of roots in each split. The independent audit reproduced every reported
paired metric and both gate decisions. Detailed phase/split/horizon summaries
are in the manifest; complete per-root pairs are in the local audit JSON.

Whole-frame differences may primarily reflect robot motion and do not establish
apple visibility, contact observability, correct goal selection, or learned
causal ranking. These are 0.8-second local continuations. All 528 final branch
scorer success flags were false; the collection is not an end-to-end placement
evaluation. The prior learned control result remains 0/6, and the fresh final
control cohort remains unexecuted. The next scientific gate requires predictions
using each branch's own actions to outperform wrong-sibling actions on VAL,
with root-grouped uncertainty; passing this data screen does not satisfy it.

## Audit and provenance limitations

The independent audit completed in **6.13 supervised wall seconds**, below its
120-second cap, without physics or model inference. Its script and result hashes
are in the manifest. Two software audit tests passed; Ruff and diff checks passed.
The full result is `outputs/apple-branches-v1-audit.json`. Collection records and
atomic per-command journals remain in the dataset directory; the frozen plan is
`data/apple-branches-v1-plan.json`.

During collection, the unrelated audit script and its test briefly existed as
untracked worktree files. They were moved outside the repository before completion;
no tracked runtime, collector or protocol file changed. Pre/post revision, clean
state, Python source, collector/protocol/action/assets and parent payload checks
passed. This records the incidental event transparently; it does not claim
continuous filesystem monitoring. The audit files were restored only after the
coordinator confirmed collection completion.
