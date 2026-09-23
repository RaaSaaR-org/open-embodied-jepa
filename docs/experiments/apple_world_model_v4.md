# Apple world model v4: redesign the action-conditioned prediction step (TASK-054)

**Preregistration. Written before any frozen run and committed before the first one
starts.** Every threshold, cohort, budget, arm, command and decision rule below is fixed
at the commit that carries this file. Nothing here is a control or manipulation result:
this is offline evaluation of learned models on recorded validation data. Learned
Apple→Plate is at **0 successes** and nothing in this protocol changes that.

Predecessor: `apple_world_model_v3.md` and `apple_world_model_v3_results.md` (TASK-052).
Frozen manifest: `benchmarks/manifests/apple-world-model-v4.json`.

## The question

TASK-052's four arms all failed their gate set. Its merged encoder/rollout decomposition
then split the primary gate's error on the identical moving validation windows:

| median, moving windows, h = 8 | encoded TARGET | rollout (G1) | rollout excess |
|---|---|---|---|
| v2 LeWM/onboard *(carried over, **not** recomputed)* | 3.26 cm | 3.65 cm | 0.39 cm |
| v3 **arm B** (onboard, patch 14) | 2.59 cm | 3.65 cm | **1.06 cm** |
| v3 **arm D** (onboard, patch 8) | 2.47 cm | 3.30 cm | **0.83 cm** |

**encoded TARGET** is the readout of a directly encoded target-frame observation: what a
*perfect* predictor could achieve with that encoder and head. **Rollout excess** is the
gate value minus it — the prediction step's own contribution.

The encoder improved 21–24 % while the rollout excess roughly doubled to tripled. The
error moved out of the encoder and into the action-conditioned prediction step. That,
plus G7a (siblings' own actions beat swapped ones) failing on arms A and B, is what
TASK-052's own earliest-failing-group rule selects: **branch 2, redesign the action
conditioning, on one camera.**

Two constraints on that reading are carried here rather than argued away.

- **Arm D's encoded-target error alone (2.47 cm) already exceeds the 1.5 cm G1
  threshold.** A perfect predictor on v3's best encoder would still fail G1. Both terms
  fail; only one grew. This protocol does not claim the encoder is solved and does not
  attack it.
- **The v2 row is quoted from `apple_world_model_v2_results.md`, not recomputed.** The v2
  checkpoints were written by a different model implementation and this code will not
  load them. Wherever the 0.39 cm appears below it carries that caveat.

Settled findings this protocol builds on rather than re-litigates: the FK hand-crop
second camera **hurt** (A ↔ B, 6.62 → 3.65 cm on G1; the only one-factor contrast in
TASK-052 whose size clearly exceeds sampling noise), so v4 is one camera. The motion
weighting and the auxiliary position readouts were bundled in TASK-052 and cannot be
separated by that design, so v4 keeps them exactly as arm B had them. Patch 14 → 8
(B ↔ D) moved G1 by 0.34 cm at 2.1× the compute, inside its own noise under the
conservative reading, so v4 keeps patch 14. Arm B's step budget binds and arm D's tail is
rising, so "train longer" is not supported by either curve and is not attempted here.

## Hypothesis

**H.** The v3 rollout excess is a property of *how* the prediction step is conditioned —
one shared single-step module, conditioned on one action at a time, trained against a
multistep loss that weights every step equally — and at least one of three changes to
that conditioning reduces the excess enough to bring the rollout error below the model's
own persistence readout (G2a < 0.8).

**H is falsifiable and has never held.** G2a has failed in every protocol: v2 0.835, v3
arms 1.010 / 0.876 / 0.940 / 0.831. It is the cleanest statement of "the prediction step
does something", which is why it is the primary gate.

## Design: four arms, one camera, one factor each

Every arm is TASK-052 **arm B**'s configuration (`apple_wm_v3_lewm_onboard.yaml`) with at
most one shared, backend-agnostic option of `models/base.py` switched on. Every option is
off at its default, so a model written before TASK-054 keeps its exact predictor and its
exact loss; `tests/test_predictor_conditioning.py` asserts that the default rollout is
bit-for-bit the earlier autoregressive loop and the default multistep loss is the earlier
unweighted mean.

| | **E0** baseline | **E1** chunk | **E2** tail | **E3** step |
|---|---|---|---|---|
| Role | control | action-chunk tokens | multistep weighting | non-shared predictor |
| Config | `apple_wm_v4_lewm.yaml` | `apple_wm_v4_lewm_chunk.yaml` | `apple_wm_v4_lewm_tail.yaml` | `apple_wm_v4_lewm_step.yaml` |
| `action_chunk` | 1 | **4** | 1 | 1 |
| `multistep_tail_weight` | 0.0 | 0.0 | **3.0** | 0.0 |
| `predictor_step_embedding` | false | false | false | **true** |
| Parameters (measured in the pilots) | 2,415,398 | 2,462,886 | 2,415,398 | 2,423,590 |
| Δ against E0 | — | +47,488 | 0 | +8,192 |

**E0 is arm B.** Its parameter count, 2,415,398, is exactly the count TASK-052 published
for arm B, and every other key in `apple_wm_v4.yaml` is arm B's. E0 exists because the
three interventions must be compared against a number produced by *this* revision: the
shared changes alter `models/base.py`, so the implementation hash moves and the v3
checkpoints cannot be loaded. E0 is a same-seed re-run of arm B's configuration, not an
independent seed, so it is a **revision control, not a variance estimate** (see
*Limitations*).

**What each option does.**

- **E1, `action_chunk: 4`.** Before step *k* the latent gets a learned joint embedding of
  actions `[k, k+4)` (zero-padded past the end of the sequence) through one shared MLP;
  the step's own action still reaches the backend predictor unchanged. This is "condition
  on a chunk of future actions jointly rather than one action per step". K = 4 is half the
  planner's horizon of 8 and a quarter of the training horizon of 16, so every training
  step has real future actions to condition on and the padding is confined to the last
  three steps of a window.
- **E2, `multistep_tail_weight: 3.0`.** The multistep latent loss is weighted by a linear
  ramp from 1 at the first predicted step to 4 at the last (16th), renormalized to mean 1
  so the term's total weight against the one-step and SIGReg terms is unchanged. This is
  "stronger multistep weighting": the rollout excess grows with horizon, and a uniform
  mean spends most of its effort where the rollout is still easy. The magnitude 3.0 is the
  same magnitude TASK-052 used for its only other loss-shaping constant
  (`readout_moving_weight`), chosen for that consistency rather than from any measurement.
  **Declared limitation:** it reweights the latent multistep term only, not the
  predicted-readout term, which stays uniform. So E2 tests "does emphasising late latent
  prediction reduce the rollout excess" and nothing wider.
- **E3, `predictor_step_embedding: true`.** A learned per-step vector is added to the
  latent before each rollout step, so the rollout is horizon-conditioned rather than one
  shared step function applied autoregressively. The embedding is **zero-initialized**, so
  at step 0 E3's rollout is bit-for-bit E0's and any departure has to be learned. The new
  modules are drawn last in `finish_init`, so E0, E1, E2 and E3 share every other weight
  at initialization.

**Architecture invariants held.** All three options live in backend-agnostic shared code;
the backend swap stays the one-line `world_model.backend` change (`apple_wm_v4.yaml` is
the native backend on the same shared settings, and the LeWM arms are `extends:` plus one
line). The planner never inspects a latent: `rollout` shifts the *input* of each step and
returns the raw predictor outputs, so a readout or a planner sees the same kind of latent
either way. Readouts remain the only declared way to get a quantity out of a latent.

**The native backend is not a frozen arm.** The budget does not fit a fifth run. It is
exercised only by the smoke, so the one-line-swap invariant is demonstrated as a software
property, not as a second trained comparison.

## Data, splits and leakage discipline

Unchanged from TASK-050 and TASK-052, and not re-derived here.

- Corpus `data/apple-wide-v1`; the same dataset, split and action hashes as TASK-052.
- 677 train episodes, 80 val episodes. Splits are grouped by whole reset
  (`split_policy.group_by: session_id`); train, val and test share no root episode.
- **`test` and `holdout` are never decoded.** `load_split` refuses any split other than
  `train`/`val` and raises if the requested episodes intersect test or holdout. Every gate
  report must carry `test_episodes_decoded: 0`.
- Normalization is fitted on the **train** split only; the runner raises if the manifest's
  `normalization.episode_ids` is not exactly `splits.train`.
- Updates use train only. Checkpoint selection and every gate use val only.
- **No simulator reset is opened in this task.** There is no closed loop, no rollout in
  MuJoCo and no evaluation cohort. The frozen final cohort (resets 44000–44019) and TEST
  are untouched, trivially.
- Privileged labels are used as readout-head training targets, as the readout loss's
  per-frame weight and as val scoring references — never as a model input, a planner
  input, a planning cost or a window filter. Unchanged from v3.

## Training (frozen; `frozen.training`)

Identical to TASK-052's, so the comparison with the v3 numbers is like-for-like.

| | Value |
|---|---|
| Steps | 15,000 per arm |
| Batch | 32 windows of 16 transitions (uniform over all train windows, with replacement) |
| Optimizer | AdamW, lr 3e-4 cosine-decayed to 3e-5 over the 15,000 steps, weight decay 1e-4, grad clip 1.0 |
| Seed | 0 (model init and window sampler) |
| Device | MPS (Apple M5 Pro, 48 GB), one run at a time, in the order E0, E1, E2, E3 |
| Wall-clock cap | 10,800 s per run including decoding; hitting it stops with status `time_budget` |
| Validation | every 1,000 steps and at the last step, on the fixed selection cohort |

**Measured budget** (from the disclosed pilots, steady-state seconds per update ×
15,000): E0 0.2303 s → 0.96 h, E1 0.2312 s → 0.96 h, E2 0.2307 s → 0.96 h, E3 0.2328 s →
0.97 h — **about 3.85 h of MPS training in total**, plus about 12 s of decoding and about
1 min of evaluation per arm. The 10,800 s cap is a runaway guard with about 2.8× headroom,
not the expected time. Peak host RSS is 9.1 GB per arm (one camera).

**Longer training is not attempted**, and neither is a second seed; both would take the
task past its budget. Each arm's 16 validation points are reported, so whether the step
budget binds stays answerable descriptively.

**Selection rule (val only).** Unchanged from TASK-050 and TASK-052, so selection scores
stay comparable across the three protocols.

- **Cohort.** 1,024 val windows, drawn with seed 10001 from the stride-4 horizon-8 windows.
- **Score.** Median palm–apple readout error at h = 8 on **predicted** latents, over the
  cohort's *moving* windows. Lower is better.
- **Eligibility.** Encoded latents must have collapsed fraction ≤ 0.05, mean per-dim std
  ≥ 0.1 and effective rank ≥ 2.
- **Choice.** The best eligible checkpoint is the arm's `.pt`; the last state is always
  saved as `.latest.pt`; the untrained step-0 state is validated and logged but is never
  selectable. If no checkpoint is eligible the run reports `selection_failed` and the
  gates are evaluated on `.latest.pt` (declared now).

**Disclosure.** Selection and the gates both use val, so G1 is flattered. Test stays
untouched for a later unbiased check.

## Gates (frozen; `frozen.gates`)

Thirteen gates are carried over from `apple_world_model_v3.md` **verbatim**, with the
same values, the same comparison directions and the same cohort rules. **No carried-over
threshold is loosened.** One gate is new.

| Gate | Threshold | Source |
|---|---|---|
| **G1** palm–apple readout error, moving windows, h = 8 (median) | ≤ **1.5 cm** | v3 verbatim |
| **G2a** G1 ÷ the model's own persistence readout | ≤ **0.8** | v3 verbatim — **PRIMARY** |
| **G2b** G1 ÷ the shuffled-action control | ≤ 0.8 | v3 verbatim |
| **G3** apple–plate error, valid windows, h = 8 (median) | ≤ 2.0 cm | v3 verbatim |
| **G4** apple height, grasp cohort, h = 8 (median abs) | ≤ 1.0 cm | v3 verbatim |
| **G5** `apple_held` AUROC, lift cohort (≥ 20 per class) | ≥ 0.85 | v3 verbatim |
| **G6a** within-state candidate ranking ρ, h = 16 (≥ 12 ranked groups) | ≥ 0.5 | v3 verbatim |
| **G6b** median top-1 regret, h = 16 | ≤ 4 mm | v3 verbatim |
| **G7a** siblings' own actions beat swapped, h = 16 (≥ 20 pairs) | ≥ 0.70 | v3 verbatim |
| **G7b** sibling divergence ρ, h = 16 | ≥ 0.5 | v3 verbatim |
| **G8a** collapsed latent fraction | ≤ 0.05 | v3 verbatim |
| **G8b** effective rank | ≥ 4.0 | v3 verbatim |
| **G8c** mean latent std | ≥ 0.1 | v3 verbatim |
| **G9** rollout excess, moving windows, h = 8 (median) | ≤ **0.53 cm** | **new** |

An arm **passes** only if all fourteen pass. G6a/G6b, G5 and G7a are `None` (and so fail)
when their cohort rule is not met; the cohort rules are frozen in the manifest, so a later
source edit cannot redefine what the gate is computed over.

### G2a is the primary gate

It is the one number that states "the prediction step does something": the rollout's
readout error divided by the error of simply reading out the *encoded start frame* and
assuming nothing moves. It has never passed — v2 0.835, v3 1.010 / 0.876 / 0.940 / 0.831
— and the pre-declared next step of the whole line hangs on it (see *Pre-declared
outcomes*).

### G9 and why its threshold is 0.53 cm

G9 is the **rollout excess**: the gate value G1 minus the encoded-target error on the
*identical* windows. TASK-052 showed that this is the term that regressed, and it is the
term this protocol attacks, so it is gated directly rather than only described.

The threshold is set from the v3 numbers, as follows.

1. **The number to move is 1.06 cm**, arm B's excess, because E0 is arm B and the three
   interventions are one factor from it. Arm D's 0.83 cm is not the reference: D changes
   the patch grid, which v4 does not.
2. **"Beat 0.83 cm" would be too weak.** The B ↔ D contrast that produced it moved G1 by
   0.34 cm, and TASK-052's conservative reading is that B ↔ D is *not* distinguishable
   from its own sampling noise. A threshold inside that band would be passable by noise.
3. **The threshold is half of arm B's excess: 1.06 / 2 = 0.53 cm.** Halving is a
   substantive reduction, three times the size of the largest contrast TASK-052 could not
   separate from noise, and it is the amount needed for the excess to stop being the term
   that grew: 0.53 cm lands between v2's 0.39 cm and v3 D's 0.83 cm.
4. **It deliberately does not require beating 0.39 cm.** That number comes from a v2
   checkpoint with an incompatible implementation hash that was never recomputed, so
   making it a threshold would freeze a gate against a figure this code cannot reproduce.

**What a G9 pass would and would not mean, declared now.**

- **G9 can pass while G1 fails, by construction.** An excess of 0.53 cm on top of arm B's
  encoder term (2.59 cm) is a rollout of about 3.1 cm, still about 2.1× G1's threshold.
  G9 bounds the term this task attacks; it is not a claim about absolute accuracy. An arm
  that passes G9 and fails G1 has **not** passed the gate set and does not start anything.
- **G9 is not passable by a do-nothing predictor given a v3-quality encoder.** An identity
  rollout makes the rollout error equal the persistence error, so its excess would be
  `persistence − encoded_target`: 1.57 cm for arm B and 1.51 cm for arm D, far above 0.53.
- **But G9 *is* passable by a degenerate model, and this was measured rather than
  reasoned.** In pilot `smoke-a` (40 steps, 8 episodes) the excess was **−1.66 cm**,
  because a near-constant readout head makes the encoded-target readout no better than the
  rollout. A G9 pass therefore means nothing on its own.
- **Pre-declared reading condition.** A G9 pass counts as evidence for the branch-2
  hypothesis only if that arm's `encoded_target_median_m` is **≤ 3.89 cm** (1.5 × arm B's
  2.59 cm). That band comfortably contains both v3 one-camera arms (2.59, 2.47 cm) and
  excludes both two-camera arms (5.45, 6.31 cm) and any degenerate readout. Outside the
  band, G9 is reported as **uninterpretable** rather than as a pass. This is a reading
  rule fixed in advance; it never changes G9's threshold or its pass/fail computation.
- **G9 is read jointly with G1, G2a, G2b, G7a and G8**, never on its own.

### Cohort-identity self-check (hard failure)

The decomposition recomputes two quantities the evaluator already computes — the rollout
median and the persistence median — and `world_model_v4.evaluate_gates` **raises** unless
both, and the moving-window count, are equal. If the decomposition were measured on a
different cohort, G9 would not be comparable with G1 and G2a. The pilots exercised this
assertion on all four arm configurations, including `action_chunk: 4`, where the rollout
must be taken over the evaluator's own 16-step action horizon rather than 8 for the two
computations to coincide.

## Contrasts and uncertainty

Three one-factor contrasts carry causal weight, all against E0: **E1 − E0** (action
chunk), **E2 − E0** (multistep tail weight), **E3 − E0** (step embedding). Each is
reported for `rollout`, `encoded_target` and the per-window `rollout − encoded_target`.

`scripts/bootstrap_wm_v4_contrasts.py`, seed 20540, 20,000 resamples, reports **two**
paired designs for every contrast:

- **`iid`** — TASK-052's design, kept verbatim for comparability: resample *n* window
  indices with replacement from the *n* moving windows and apply the same indices to both
  arms. TASK-052 recorded this as a **lower bound** on the uncertainty, because the moving
  windows are stride-4 windows from only 80 val episodes and consecutive windows share
  most of their transitions.
- **`cluster`** — the design TASK-052 said was never computed: resample the *E* validation
  episodes that contribute a moving window, with replacement, and pool all their moving
  windows, so correlated windows stay together. This needs each window's episode index,
  which the v3 dump did not record and `world_model_v4 evaluate --dump-errors` now does.

**Both are reported for every contrast, and the clustered interval is the one read.** If
the two disagree about whether a contrast crosses zero, the conservative (clustered)
reading is adopted and the disagreement is reported, exactly as TASK-052 did.

Neither design is a run-to-run interval. **With one seed per arm there is no estimate of
how much of any gap a different seed would reproduce, and no bootstrap over windows can
supply one.** Every difference reported in the results will be a single-run difference.

Raw latent MSEs are not compared across arms or backends. Comparisons are in physical
readout units and compute.

## Pre-declared outcomes

Fixed before any number is seen. The primary gate is **G2a**, and it alone selects between
the two branches.

**Outcome A — at least one arm reaches G2a < 0.8.** The prediction step demonstrably beats
the model's own persistence readout for the first time in this line. Then:

- CEM over this cost is **kept**, and the control-formulation clause below does not fire.
- The **winning option** is the arm with the lowest G2a; ties are broken by lower G9, then
  by lower G1. It is carried forward.
- The next protocol attacks the **encoder** term, because arm D's 2.47 cm encoded target
  alone still exceeds the 1.5 cm G1 threshold and a perfect predictor would not fix that.
  Concretely: the single-frame information-ceiling measurement at 224 px, which TASK-052
  demoted but did not drop, becomes the primary line, combined with the winning predictor
  option.
- **The closed loop still does not start.** G1 must pass first, and no outcome of this
  protocol starts it.

**Outcome B — no arm reaches G2a < 0.8.** Then the clause TASK-052 recorded fires, as
written:

> **If the action-conditioning redesign does not move G2a below 0.8, behaviour cloning
> with the world model as a critic becomes the primary line and CEM over this cost is
> abandoned.**

That clause was already recorded as **TRIGGERED** by TASK-052 (G2a ≥ 0.8 on all four v3
arms) and deferred by exactly one protocol — this one. Under Outcome B it is not deferred
again: **no further predictor-architecture protocol is preregistered**, and the next task
is behaviour cloning with the world model as a critic or residual.

**Outcome C — E0, the baseline, also reaches G2a < 0.8.** This is called out separately
because it is the most informative failure mode available and TASK-052 had no replication
at all. E0 is arm B's configuration at the same seed. If E0's G2a lands below 0.8 while
v3's arm B was 0.876, the difference is revision or run-to-run variation, **not** any
intervention. Then:

- No intervention is credited, whatever E1–E3 do, and this is the **headline** result.
- The next protocol is a **replication**: the baseline at ≥ 3 seeds, before any further
  architecture work.
- Outcome A's branch does not apply even if some arm passes G2a.

**In every outcome:** arms that fail are reported with their numbers; the gate table,
the encoder/rollout decomposition and both bootstrap designs are published for all four
arms whatever they say; and nothing is retuned and re-reported after the gates are read.

## Frozen commands

Each runs from a clean committed checkout; `--require-clean` refuses a dirty or
unversioned tree. Arms run one at a time, in the order E0, E1, E2, E3. Every artifact
path is absolute, under the main checkout's git-ignored `checkpoints/` and `outputs/`.

```sh
# train (per arm; <cfg> and <name> from the design table)
uv run --no-sync python -m embodied_jepa.world_model_v4 train \
  --config configs/<cfg>.yaml \
  --protocol-manifest benchmarks/manifests/apple-world-model-v4.json \
  --device mps --workers 8 --require-clean \
  --acknowledge-privileged-training-labels

# evaluate (per arm, on the selected checkpoint)
uv run --no-sync python -m embodied_jepa.world_model_v4 evaluate \
  --config configs/<cfg>.yaml \
  --protocol-manifest benchmarks/manifests/apple-world-model-v4.json \
  --device mps --workers 8 --require-clean \
  --checkpoint  <main>/checkpoints/task054-wm-v4/<name>.pt \
  --output      <main>/outputs/task054-wm-v4/<name>-val-gates.json \
  --dump-errors <main>/outputs/task054-wm-v4/<name>-errors.npz \
  --acknowledge-privileged-training-labels

# contrasts, after all four arms
uv run --no-sync python scripts/bootstrap_wm_v4_contrasts.py \
  --errors <main>/outputs/task054-wm-v4 \
  --output <main>/outputs/task054-wm-v4/paired-bootstrap.json \
  --seed 20540 --resamples 20000
```

The runner writes its progress and every arm's gate summary to
`<main>/outputs/task054-runner.log`.

**Each gated run executes exactly once.** `train` refuses to overwrite existing training
artifacts and `evaluate` refuses to overwrite a report, so a second attempt cannot
silently replace the first. If a run fails for an infrastructure reason, the failure and
the new path are disclosed in the results document.

## Artifacts

- `<main>/checkpoints/task054-wm-v4/{leworldmodel_baseline,leworldmodel_chunk,leworldmodel_tail,leworldmodel_step}.{pt,latest.pt,run.json,metrics.jsonl}`
- `<main>/outputs/task054-wm-v4/<name>-val-gates.json`, `<name>-errors.npz`,
  `paired-bootstrap.json`
- `<main>/outputs/task054-runner.log`

Nothing under `data/` is written. TASK-050's and TASK-052's artifacts are not touched.
Checkpoints and error dumps are git-ignored; the manifest and the hashes are committed.

## Not done in this task (declared)

- **The closed loop.** Not started, under any outcome.
- **The test split.** Never decoded.
- **The encoder.** Not changed. The camera set, patch grid, readout heads and readout loss
  shaping are arm B's, unchanged.
- **The native backend as a frozen arm.** Only smoked.
- **Extra seeds, longer training, ensembles, combinations of E1–E3.** Not run. A combined
  arm is explicitly excluded: it would not be one factor from anything.
- **Behaviour cloning.** Not run here; it is Outcome B's next task.

## Pilots before freezing (disclosed)

All pilots are under `checkpoints/task054-scratch/` and `outputs/task054-scratch/`, run
from the committed revision that carries the code but not yet this document.

- **`smoke-a`.** E0 on 8 train and 8 val episodes, 40 steps, plus `evaluate` on 12 val
  episodes with `--dump-errors`. Plumbing only: it confirmed the v4 gate wiring, the
  cohort-identity assertion and the error dump end to end. **Its gate values were visible
  to the author** and are reported above for exactly one purpose: its rollout excess of
  −1.66 cm is the measured demonstration that G9 is passable by a degenerate readout,
  which is why the reading condition above exists. No threshold was changed after seeing
  it; every threshold is either copied verbatim from TASK-052 or derived from TASK-052's
  published arm B figures.
- **`timing-{baseline,chunk,tail,step}`.** All four arms, full corpus, 120 steps, for wall
  clock and parameter counts only: 0.2303 / 0.2312 / 0.2307 / 0.2328 s per update, peak
  host RSS 9.1 GB, decode 12.3–12.5 s. These fixed the budget table. Their `evaluate` runs
  on 20 val episodes confirmed the cohort-identity assertion holds for every option,
  including `action_chunk: 4`.
- **No pilot informed any threshold**, and no pilot checkpoint is a frozen arm.

## Limitations, stated before the results

- **One seed per arm, one run each.** Every difference between arms will be a single-run
  difference with no run-to-run uncertainty estimate. The bootstrap intervals quantify
  cohort sampling for fixed checkpoints and nothing else.
- **E0 is a revision control, not a replication.** It re-runs arm B's configuration at
  arm B's seed. It cannot distinguish "the intervention did nothing" from "this seed is
  lucky"; Outcome C exists because of that.
- **Selection and the gates both use val**, which flatters G1 and every gate derived from
  it. Test was never decoded, so an unbiased check is still available.
- **The arms are not unmodified LeWM.** Each is the pinned upstream encoder, predictor,
  projector and SIGReg objective (revision `8edfeb33`) plus the shared state fusion, the
  shared camera fusion, the readout loss and — in E1 and E3 — the shared prediction-step
  conditioning.
- **The ranking cohort is narrow**: 18 ranked groups from 20 val resets at h = 16, while
  the planner's horizon is 8. At h = 8 no v3 arm reached ρ ≥ 0.5.
- **The corpus is privileged scripted-collector data** with injected perturbations; every
  limitation of `apple_wide_collection_results_v1.md` still applies.
- **Nothing in this protocol is a manipulation result.** A gate pass is an offline property
  of a checkpoint on recorded validation data. Learned Apple→Plate stays at 0 successes.
