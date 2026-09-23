# Apple world model v4: results (TASK-054)

**All four arms FAIL the preregistered gate set, and the primary gate fails on every
one.** G2a — the rollout's readout error divided by the model's own persistence readout —
is 0.8763 / 0.8814 / 0.8635 / 0.9036 against a threshold of 0.8. **None of the three
redesigns of the action-conditioned prediction step moved it below 0.8, and none came
close.** By the protocol's pre-declared reading this fires **Outcome B**: the
control-formulation clause TASK-052 recorded is not deferred again, behaviour cloning with
the world model as a critic becomes the primary line, and **CEM over this cost is
abandoned**.

**The control passed more gates than every intervention.** E0 — TASK-052 arm B's
configuration re-trained at this revision — passes 10 of 14; E1, E2 and E3 each pass 9.

Learned Apple→Plate remains at **0 successes**. Nothing in this document is a control
result, a closed-loop result or working manipulation. Every number is offline evaluation
of learned models on recorded validation data. No simulator reset was opened and no
planner was run. The protocol is `apple_world_model_v4.md`, written at `50b5765` and amended
twice by a fresh-context pre-run review, at `2bf3999` and `46d62eb`, both before any gated
run. **The runs themselves were made from `46d62eb` (E0's training) and `c9cf9a6` (E1, E2
and E3's training, and all four evaluations)**; no run was made from `2bf3999`. The
protocol document is byte-identical across `46d62eb`, `c9cf9a6` and this commit. The
manifest is byte-identical across `46d62eb` and `c9cf9a6`; this commit's copy differs only
because the results were written into its `null` placeholders — **every pre-declared block
(`frozen`, including `frozen.gates`, and `pre_declared_outcomes`) is byte-identical from
`46d62eb` through here.** See *The runner incident*.

## Runs

Four runs, each executed once, sequentially on one machine, each from a clean committed
checkout (`--require-clean`, `dirty: false` in every report; see
`outputs/task054-runner.log`).

| | **E0** baseline | **E1** chunk | **E2** tail | **E3** step |
|---|---|---|---|---|
| Role | control (= v3 arm B) | action-chunk tokens | multistep weighting | non-shared predictor |
| Config | `apple_wm_v4_lewm.yaml` | `..._chunk.yaml` | `..._tail.yaml` | `..._step.yaml` |
| `action_chunk` | 1 | **4** | 1 | 1 |
| `multistep_tail_weight` | 0.0 | 0.0 | **3.0** | 0.0 |
| `predictor_step_embedding` | false | false | false | **true** |
| Steps | 15,000 | 15,000 | 15,000 | 15,000 |
| Status | `completed` | `completed` | `completed` | `completed` |
| Selected step | **15,000** (last) | 14,000 | **15,000** (last) | **8,000** |
| Selection score (val) | 3.54 cm | 3.62 cm | 3.37 cm | 4.22 cm |
| Parameters | 2,415,398 | 2,462,886 | 2,415,398 | 2,423,590 |
| Wall clock | 3,512 s | 3,884 s | 3,553 s | 3,521 s |
| Process CPU | 1,531 s | 1,583 s | 1,575 s | 1,552 s |
| Peak host RSS | 9.11 GB | 9.13 GB | 8.97 GB | 8.78 GB |
| Evaluation wall clock (runner log) | 16 s | 17 s | 16 s | 17 s |
| Evaluation wall clock (in report) | 14.0 s | 15.6 s | 14.7 s | 14.7 s |
| Checkpoint SHA-256 | `01ba8d0500…` | `ccc8a77e59…` | `6f4c02b427…` | `e6bf4aaaba…` |

The two evaluation rows are on different bases and both are given because they are not
interchangeable: the first is the subprocess wall clock from `outputs/task054-runner.log`,
the second each gate report's own `elapsed_seconds`. `apple_world_model_v3_results.md` used
the in-report basis for its identically-labelled row, so the **second** row is the one
comparable with v3.

**One-factor isolation, checked from the resolved `model_config` blocks in the run
reports:** against E0, exactly one key differs in each arm — `action_chunk` 1→4 (E1),
`multistep_tail_weight` 0.0→3.0 (E2), `predictor_step_embedding` false→true (E3). Every
other key, plus seed, device, budget, data counts and every hash, is identical across the
four arms. The parameter deltas are exactly the declared ones: +47,488 for E1's chunk MLP,
0 for E2 (loss-only), +8,192 for E3's 64×128 step embedding.

- **Code revision** `c9cf9a6` for E1, E2, E3 and for all four evaluations. **E0's training
  ran at `46d62eb`** — see *The runner incident* below. The Python source SHA-256 is
  `3cb9c6f1df…` and the model implementation SHA-256 is `4ad0a6856d…` for **every** run
  and **every** evaluation, including E0's, so the revision difference touched no hashed
  source.
- **Seed** 0 (model init and window sampler) for all four arms.
- **Device** MPS, Apple M5 Pro, 48 GB.
- **Budget** the frozen cap is `max_seconds: 10800` per arm. No arm hit it; the longest
  (E1) used 3,884 s. Training totalled 14,470 s = **4.02 h** against the preregistered
  estimate of 3.85 h. Against each arm's own preregistered rate × 15,000 steps, training
  wall clock ran E0 +1.7 %, **E1 +12.0 %**, E2 +2.7 %, E3 +0.8 %. E1's overrun is
  plausibly the action-chunk MLP, but that is an **attribution, not a measurement** — this
  protocol ran no timing control that would separate it from machine load.
  Batch 32 × horizon 16, AdamW lr 3e-4 cosine-decayed to 3e-5, weight decay
  1e-4, grad clip 1.0, validation every 1,000 steps — all as frozen.
- **Data** `data/apple-wide-v1`, dataset manifest SHA-256 `028e130576…`, split hash
  `51f8e09df1…`, action hash `da987bb079…`, selection-window SHA-256 `16eb332c30…` — all
  four identical across arms and **identical to TASK-050's and TASK-052's**. 677 train
  episodes, 80 val episodes.
- **`test_episodes_decoded: 0`** in all four run reports and all four gate reports. As
  TASK-052 recorded, that field is a **literal emitted by the reporter, not a counter**;
  the real guarantee is structural — `load_split` refuses any split other than
  `train`/`val` and raises if the requested episodes intersect test or holdout, and the
  runner raises unless the manifest's `normalization.episode_ids` equals `splits.train`
  exactly. Updates used train only; selection and every gate used val only.
- **Artifacts** (git-ignored, under the main checkout)
  `checkpoints/task054-wm-v4/{leworldmodel_baseline,leworldmodel_chunk,leworldmodel_tail,leworldmodel_step}.{pt,latest.pt,run.json,metrics.jsonl}`
  and `outputs/task054-wm-v4/<name>-val-gates.json`, `<name>-errors.npz`,
  `paired-bootstrap.json`, plus `outputs/task054-runner.log`. Nothing under `data/` was
  touched; TASK-050's and TASK-052's artifacts are intact.

## Gate table (val only; h = 8 unless stated)

| Gate | Threshold | E0 (control) | E1 (chunk) | E2 (tail) | E3 (step) |
|---|---|---|---|---|---|
| **G1** palm–apple, moving windows | ≤ 1.5 cm | **3.65 cm FAIL** | **3.66 cm FAIL** | **3.48 cm FAIL** | **4.41 cm FAIL** |
| **G2a** ÷ own persistence readout | ≤ 0.8 | **0.8763 FAIL** | **0.8814 FAIL** | **0.8635 FAIL** | **0.9036 FAIL** |
| **G2b** ÷ shuffled-action control | ≤ 0.8 | 0.6656 PASS | 0.6908 PASS | 0.6879 PASS | 0.7571 PASS |
| **G3** apple–plate, valid windows | ≤ 2.0 cm | 1.97 cm PASS | 1.84 cm PASS | 1.85 cm PASS | **2.39 cm FAIL** |
| **G4** apple height, grasp cohort | ≤ 1.0 cm | 0.15 cm PASS | 0.13 cm PASS | 0.17 cm PASS | 0.26 cm PASS |
| **G5** `apple_held` AUROC, lift | ≥ 0.85 | 0.9995 PASS | 0.9996 PASS | 0.9997 PASS | 0.9992 PASS |
| **G6a** within-state ranking ρ, h = 16 | ≥ 0.5 | 0.5438 PASS | **0.2865 FAIL** | **0.3296 FAIL** | **0.4141 FAIL** |
| **G6b** median top-1 regret, h = 16 | ≤ 4 mm | 0.0 mm PASS | 2.3 mm PASS | 0.0 mm PASS | 0.0 mm PASS |
| **G7a** siblings own < swapped, h = 16 | ≥ 0.70 | **0.6981 FAIL** | **0.6321 FAIL** | **0.6604 FAIL** | 0.7264 PASS |
| **G7b** sibling divergence ρ, h = 16 | ≥ 0.5 | 0.5766 PASS | 0.6264 PASS | 0.5806 PASS | 0.6358 PASS |
| **G8a** collapsed fraction | ≤ 0.05 | 0.000 PASS | 0.000 PASS | 0.000 PASS | 0.000 PASS |
| **G8b** effective rank | ≥ 4 | 7.74 PASS | 8.23 PASS | 8.01 PASS | 6.74 PASS |
| **G8c** mean latent std | ≥ 0.1 | 0.868 PASS | 0.869 PASS | 0.850 PASS | 0.873 PASS |
| **G9** rollout excess | ≤ 0.53 cm | **1.058 cm FAIL** | **1.118 cm FAIL** | **0.903 cm FAIL** | **0.796 cm FAIL** |
| Gates passed of 14 | | **10** | 9 | 9 | 9 |
| **Overall** | all gates | **FAIL** | **FAIL** | **FAIL** | **FAIL** |

Every threshold and comparison direction in the gate reports matches
`benchmarks/manifests/apple-world-model-v4.json` (`frozen.gates`) and the preregistration
table verbatim, and is identical across the four arms. No threshold was changed.

**One row of that table must not be read on its own: G9.** E3 has the lowest rollout
excess (0.796 cm) *and* the worst G1 (4.41 cm), because its encoder degraded — the excess
is the gate value minus the encoder term, so damaging the encoder shrinks it. E3 did not
improve the prediction step. The only arm whose G9 moved the right way with an intact
encoder is E2, and it still fails by a factor of 1.7. This is the failure mode the
preregistration named in advance when it declared the G9 reading condition necessary but
not sufficient.

## The headline: E0 reproduces v3 arm B bit-for-bit

E0 is TASK-052 arm B's configuration, at arm B's seed, re-trained after `models/base.py`
gained three new config keys. **It reproduced arm B exactly.** Verified three ways:

- **All 13 carried gates are identical as exact floats** — not to four decimals, but as
  IEEE doubles: G1 `0.036480627954006195`, G2a `0.8763…`, and the other eleven.
- **All 11 decomposition quantities are identical**, including `encoded_start`
  2.3536 cm, `encoded_target` 2.5903 cm, `rollout` 3.6481 cm, `persistence` 4.1628 cm,
  `rollout_excess` 1.0578 cm, and both cohort counts (1,462 moving of 3,961 valid).
- **All 170 weight tensors are bit-identical** after 15,000 MPS training steps
  (`torch.equal` on every tensor). `best_selection_score` matches to full precision
  (`0.035446153953671455`), as do `best_step` 15,000 and the parameter count 2,415,398.

What differs is only what should: the revision, the Python source hash, the implementation
hash, and the checkpoint **file** hash — the envelope stores the config (which gained three
keys at their OFF defaults: `action_chunk: 1`, `multistep_tail_weight: 0.0`,
`predictor_step_embedding: false`) and the implementation hash, so the file bytes differ
while the weights inside do not.

Two things follow.

1. **The three new options are provably inert at their defaults**, end-to-end through
   15,000 training steps and a full evaluation — not merely in unit tests. The invariant
   that a model written before TASK-054 keeps its exact predictor and its exact loss holds
   in the strongest available sense.
2. **E0 is a genuine control.** Any difference E1–E3 show is attributable to the one
   option each changed, not to the revision. It is *not* a replication in the useful
   sense: same seed, same data order, same everything, so it says nothing about run-to-run
   variance. **Outcome C does not fire** — E0's G2a is 0.8763, above 0.8.

## The encoder / rollout decomposition

Measured inside the gate evaluation, on the gate's own cohort. The `rollout` and
`persistence` columns are recomputed independently of `window_metrics` and asserted equal
to it bit-for-bit before the gates are written; that assertion passed for all four arms,
including E1 where `action_chunk` makes the rollout sensitive to the action horizon.

| median, moving windows, h = 8 | encoded START | **encoded TARGET** | rollout (G1) | **rollout excess** | encoder share |
|---|---|---|---|---|---|
| v2 LeWM/onboard *(carried over, **not** recomputed)* | 3.13 cm | 3.26 cm | 3.65 cm | **0.39 cm** | 89 % |
| v3 arm B / **E0** (identical) | 2.3536 cm | **2.5903 cm** | 3.6481 cm | **1.0578 cm** | 71.0 % |
| **E1** (chunk) | 2.3548 cm | 2.5403 cm | 3.6579 cm | **1.1176 cm** | 69.5 % |
| **E2** (tail) | 2.3978 cm | 2.5723 cm | 3.4754 cm | **0.9031 cm** | 74.0 % |
| **E3** (step) | 3.3163 cm | 3.6130 cm | 4.4087 cm | **0.7957 cm** | 82.0 % |

All four arms' `encoded_target` is inside the pre-declared 3.89 cm reading band, so G9 is
interpretable for all four — and all four fail it. The best excess is E3's 0.7957 cm,
still **1.50× the 0.53 cm threshold**, and E3 reaches it only by making the encoder much
worse (see below). The best excess among arms with an intact encoder is E2's 0.9031 cm, a
**14.6 % reduction** on E0's 1.0578 cm where **49.9 % was required**.

## The one-factor readings

Three contrasts carry causal weight, all against E0. Each rests on **a single seed per
arm**, so every difference is a one-run difference.

Point estimates are differences of medians on the identical 1,462 moving windows. Intervals
are paired bootstraps, 20,000 resamples, seed 20540, both arms resampled on the identical
draw; `scripts/bootstrap_wm_v4_contrasts.py`, report at
`outputs/task054-wm-v4/paired-bootstrap.json`. **The clustered design is the one read**, as
preregistered: it resamples the 67 validation episodes that contribute a moving window and
pools all their windows, so correlated windows stay together. The iid design is TASK-052's
and is reported for comparability only; TASK-052 recorded it as a lower bound on the
uncertainty and this run agrees — the clustered intervals are **1.46×–2.28× wider**
across the nine contrast/field combinations (1.66×–2.28× over the six shown below).

| contrast | encoded target | rollout (G1) | rollout excess |
|---|---|---|---|
| **E1 − E0** (action chunk) | −0.050 cm | **+0.010 cm** | **+0.060 cm** |
| clustered 95 % | [−0.334, +0.226] | [−0.407, +0.367] | — |
| **E2 − E0** (multistep tail) | −0.018 cm | **−0.173 cm** | **−0.155 cm** |
| clustered 95 % | [−0.368, +0.209] | [−0.804, +0.132] | — |
| **E3 − E0** (step embedding) | **+1.023 cm** | **+0.761 cm** | **−0.262 cm** |
| clustered 95 % | **[+0.485, +1.435]** | **[+0.155, +1.199]** | — |

**There is no interval for the rollout-excess column, and this is a real gap.** The
bootstrap's `rollout_excess` field resamples the median of the *per-window* difference
`rollout − encoded_target`; gate G9 is the difference of the *two medians*. They are
different estimands and they differ substantially here — for E2 − E0 the bootstrap's
quantity is −0.010 cm where the gate-consistent difference is −0.155 cm. So the bootstrap
bounds a related quantity, not the gate value, and **no interval on the G9 contrast is
available from this run**. No gate reads a bootstrap output; no bootstrap interval should
be quoted as bounding G9.

### E1 ↔ E0 — action-chunk conditioning: no effect, and it cost the ranking

Conditioning each step jointly on the next four actions did **nothing** to the term it
targeted. The rollout is 0.010 cm *worse*, the excess 0.060 cm *worse*, and both intervals
cross zero under both designs. G2a moved the wrong way (0.8763 → 0.8814). The one clear
change is a **collapse of the candidate-ranking gate, G6a 0.5438 → 0.2865** — the arm
became markedly worse at the metric a CEM actually uses — and G7a fell 0.6981 → 0.6321.

**This negative result is stronger than it looks, because E1 was measured with an
advantage.** As preregistered, the evaluator rolls 16 action steps and reads step 8, so
E1's step-8 latent depends on actions 0…10 while E0's depends on 0…7. A horizon-8 CEM
cannot supply actions 8…10 and would zero-pad them. E1 therefore saw three future actions
the planner could not give it — **and still did not beat the control.** Since no arm passes
G2a, the mandatory planner-consistent re-measurement of E1 is moot: it could only make E1
worse.

### E2 ↔ E0 — stronger multistep weighting: the right direction, far too small, not established

Ramping the multistep latent loss towards the late rollout steps is the only intervention
that moved the target term the right way, and **almost all of its gain is in the term this
task attacked**: of the 0.173 cm rollout improvement, 0.155 cm is rollout excess and only
0.018 cm is the encoder. The mechanism behaved as designed.

It is not enough, and it is not established.

- **Not enough:** the excess fell 1.0578 → 0.9031 cm, a 14.6 % reduction where 49.9 % was
  needed. G2a moved 0.8763 → 0.8635, still 7.9 % above the threshold. Extrapolating this
  size of effect linearly, the ramp would have to be about **3.4× as effective to reach
  G9** and about **5.9× to reach G2a**. (Linear extrapolation of a single arm is a crude
  guide to magnitude, not a prediction.)
- **Not established:** the rollout difference's clustered interval is
  [−0.804, +0.132] cm and **crosses zero**. Under the iid design it does not
  ([−0.416, −0.006]), and the protocol pre-declared that the clustered reading wins when
  the two disagree. So E2 − E0 is **not distinguishable from cohort sampling**.
- It also cost the ranking gate: **G6a 0.5438 → 0.3296**.

### E3 ↔ E0 — a horizon-conditioned predictor: it damaged the encoder

Adding a learned per-step vector to the latent before each rollout step made the model
**worse on the gate and much worse on the encoder**, and this is the one contrast in this
task whose intervals exclude zero under **both** designs:

| | E0 | E3 |
|---|---|---|
| G1 palm–apple, moving | 3.6481 cm | **4.4087 cm** |
| encoded TARGET | 2.5903 cm | **3.6130 cm** |
| palm–apple, all valid windows | 0.78 cm | **1.23 cm** |
| G8b effective rank | 7.74 | 6.74 |
| image-only effective rank | 7.19 | **6.00** |
| G6a ranking ρ | 0.5438 | 0.4141 |
| **G7a own beats swapped** | 0.6981 | **0.7264 PASS** |
| Selected step | 15,000 | **8,000** |

The step embedding is applied only to the rollout, yet the damage is in the **encoder**:
`encoded_target` rose 1.023 cm [+0.485, +1.435]. The encoder and the readout head are
trained through the same shared loss as the predictor, so extra per-step predictor capacity
changes the gradients reaching them. That is a mechanism **hypothesis**, not a measured
cause; this protocol contains no control that would separate it from alternatives.

E3's rollout excess is the lowest of the four (0.7957 cm), but that is an artefact of a
degraded encoder, not a better predictor: its rollout is the *worst* of the four. This is
exactly the failure mode the preregistration warned about when it declared the reading
condition necessary but not sufficient — E3 sits inside the 3.89 cm band at 3.6130 cm and
still produces a misleading excess. **Read G9 alone and E3 looks like the best arm; read it
with G1 and it is plainly the worst.**

**One genuinely positive result sits inside this failure: E3 is the only arm in v4 that
passes G7a** (0.7264 ≥ 0.70), the sibling action-sensitivity gate that failed on v3 arms A
and B and on v4 E0, E1 and E2. Making the predictor horizon-conditioned *did* improve how
much the prediction follows the executed actions.

**But the margin is three sibling pairs, and that must be read with the number.** G7a is a
fraction over **106 qualifying ordered pairs**, identical in all four arms, so the gate
moves in steps of 1/106 = 0.0094 and the 0.70 threshold sits at 74.2 pairs:

| arm | G7a | pairs won |
|---|---|---|
| E0 | 0.6981 | 74 / 106 |
| E1 | 0.6321 | 67 / 106 |
| E2 | 0.6604 | 70 / 106 |
| **E3** | **0.7264** | **77 / 106** |

So E3 passes on **3 more pairs out of 106** than the control, and **E0 misses the threshold
by a single pair** — one more win would put it at 0.7075 and a pass. No uncertainty
interval was computed for G7a, unlike every rollout and encoder contrast in this document.
It bought that at the cost of the encoder, and it is one seed. It is the only thread in this
task worth pulling on, and it is a thin one.

E3 is also the only arm whose validation curve clearly turned over: its best is step 8,000
and its last step is 0.78 cm worse (4.22 → 5.00 cm). "Train E3 longer" is not supported.

## What else the numbers say

- **Every arm still beats its shuffled-action and zero-action controls.** G2b passes
  everywhere (0.666–0.757); zero-action medians are 5.21 / 5.66 / 5.24 / 5.24 cm against
  predicted 3.65 / 3.66 / 3.48 / 4.41 cm. The actions carry information; the readout under
  motion is what is imprecise. **But every arm still loses to its own persistence readout
  by the required margin** — that is G2a, and it is the whole story of this task.
- **Every intervention hurt candidate ranking, and this is the most consistent pattern in
  the task.** G6a is 0.5438 for the control and 0.2865 / 0.3296 / 0.4141 for E1 / E2 / E3
  — the control is the *only* v4 arm that passes it. The one metric a CEM directly needs
  got worse under **all three** predictor redesigns, by 0.13 to 0.26. See the section
  below for what that does and does not support.
  G6b passes almost everywhere but discriminates nothing (median 0.0 mm for three arms
  against a label-derived random-choice baseline of 8.44 mm); its preregistered null pass
  rate is 5.9 %, so it must not be quoted alone.
- **The all-window / moving-window gap is unchanged.** Palm–apple error over all valid
  windows is 0.78 / 0.83 / 0.82 / 1.23 cm against 3.65 / 3.66 / 3.48 / 4.41 cm on the
  moving windows, with a true displacement of 2.49 cm. A predictor that saw the offset
  exactly and then assumed it froze would still beat every arm on the moving windows.
- **No collapse anywhere** (G8 passes on all four). Effective rank 6.74–8.23.
- **The step budget binds for E0 and E2**, both selected at the last step and still
  descending; E1's best is step 14,000 with its last step 0.08 cm worse; E3's turned over
  at 8,000. So "train longer" remains live for E0 and E2 only — and it was live for arm B
  after v3 too, and is still untested.

## Is there a prediction / discrimination trade-off? The numbers say no

It is tempting to read the results as a trade: improve the rollout term and you pay for it
in candidate ranking. **The data does not support that, and it is worth stating because the
reading is plausible and wrong.**

| arm | rollout excess | G6a | G1 |
|---|---|---|---|
| v3 A (two cameras) | 0.3075 cm | 0.3280 | 6.6176 cm |
| v3 B / v4 E0 *(one arm, one checkpoint)* | 1.0578 cm | 0.5438 | 3.6481 cm |
| v3 C (uniform readout) | 0.6601 cm | 0.5155 | 6.1128 cm |
| v3 D (patch 8) | 0.8348 cm | 0.3776 | 3.3043 cm |
| v4 E1 (chunk) | 1.1176 cm | 0.2865 | 3.6579 cm |
| v4 E2 (tail) | 0.9031 cm | 0.3296 | 3.4754 cm |
| v4 E3 (step) | 0.7957 cm | 0.4141 | 4.4087 cm |

The table has **seven rows, not eight arms**: v3 B and v4 E0 are the same checkpoint, bit
for bit, so counting both would double-count one arm.

Across the seven distinct arms the Spearman correlation between rollout excess and G6a is
**-0.107**; counting E0 as a separate eighth point and averaging the tied ranks gives
**+0.084**. **The estimate changes sign on that bookkeeping choice** — which is what a rank
correlation on seven points is worth. Within the four v4 arms it is **-0.400**; within the
four v3 arms it is **+0.800**: two subsets of the same evidence, opposite signs. (Computed
with the evaluator's own `world_model_v2.spearman`, which averages tied ranks.)

And the within-v4 gradient runs the *wrong way for a trade-off*: the arm that cut the excess
most (E3, −0.262 cm) lost the **least** G6a (−0.130), while the arm that *raised* the excess
(E1, +0.060 cm) lost the **most** (−0.257).

**What is true, and it is the more interesting statement:** all three predictor redesigns
cost candidate ranking, and they did so **regardless of what they did to the rollout term**.
E1 made the rollout term worse and still lost 0.257 of G6a. So whatever damages ranking is
not paid for out of prediction accuracy — it is some other route, and this protocol contains
no measurement that identifies it.

**What this is not.** One seed per arm; the arms are not independent replications; no
mechanism has been demonstrated; and these eight arms come from two protocols that differ
in more than one factor. This is an **observed pattern across arms**, not a capacity
constraint, not a law, and not something to design the next task around without measuring
it directly.

## Reading, as pre-declared

The decision rule is applied as written, not re-interpreted after the fact.

- **No arm passes → the closed loop does not start.** Unchanged from v2 and v3.
- **The primary gate G2a fails on all four arms, so Outcome B fires.** The protocol
  pre-declared, before any number was seen:

  > **Outcome B — no arm passes G2a.** The control-formulation clause recorded by TASK-052
  > fires as written and is not deferred again: behaviour cloning with the world model as
  > a critic or residual becomes the primary line, CEM over this cost is abandoned, and no
  > further predictor-architecture protocol is preregistered.

  That is the outcome. **CEM over this cost is abandoned.** The clause was already recorded
  as TRIGGERED by TASK-052 and deferred by exactly one protocol — this one. It is not
  deferred again.
- **Outcome A does not apply** (no arm other than E0 passes G2a), and **Outcome C does not
  apply** (E0's G2a is 0.8763, above 0.8).
- **The next task is behaviour cloning with the world model as a critic or residual.** No
  further predictor-architecture protocol is preregistered, as declared.
- **Nothing was retuned and re-reported.** No threshold moved, no arm was rerun, and no
  result here was produced after seeing another.

### What this task establishes, and what it does not

It **establishes**, on one seed each, that none of the three named branch-2 candidates
rescues the prediction step at this budget: joint action-chunk conditioning does nothing
(and it was measured with an advantage the planner cannot give it), a tail-weighted
multistep loss moves the right term by about a seventh of what is needed and not beyond
cohort noise, and a horizon-conditioned predictor trades encoder quality for action
sensitivity.

It does **not** establish that the prediction step cannot be fixed. Three specific designs
at one seed and one budget is a narrow test of a broad hypothesis. What it does establish
is that the pre-declared decision rule's condition is met, and the rule is being followed
rather than renegotiated after the fact.

The encoder is also **not** solved, and this task did not attack it: E0's encoded-target
error of 2.5903 cm alone still exceeds the 1.5 cm G1 threshold, so a perfect predictor on
this encoder would still fail G1. Both terms fail. That was true before this task and is
true after it.

## The runner incident (disclosed)

The first launch of the frozen runner **died silently after E0's training**, before any
evaluation. The cause was a script bug, not a crash in the experiment: `status` is a
read-only special variable in zsh, aliased to `?`, so the `status=$?` idiom raised
`scripts/run_apple_wm_v4.sh:52: read-only variable: status` and killed the script. The
error went to the launching shell's stderr, not the runner log, so the log ended mid-run
with no error in it. The bug was present from the script's first commit; the pilots never
hit it because they used a different driver.

- **Nothing was retrained and nothing was overwritten.** E0's training artifacts were
  complete and valid. The fixed runner (`c9cf9a6`) is resumable and skips any arm whose
  artifacts already exist; it logged `train SKIPPED, training artifacts already exist
  (resume)` for E0, read `status=completed best_checkpoint_exists=yes best_step=15000` from
  the run report and evaluated the existing checkpoint. `train` refuses to overwrite
  training artifacts and `evaluate` refuses to overwrite a report, so a gated run cannot be
  silently repeated in any case.
- **Provenance consequence.** E0 was **trained** at `46d62eb`; E1, E2, E3 and all four
  **evaluations** ran at `c9cf9a6`. The Python source hash (`3cb9c6f1df…`) and the model
  implementation hash (`4ad0a6856d…`) are identical across every run and every evaluation,
  because the runner lives in `scripts/`, which `source_identity()` does not hash, and
  `docs/experiments/apple_world_model_v4.md` and
  `benchmarks/manifests/apple-world-model-v4.json` are byte-identical between the two
  commits.
- **The preregistered budget was not changed** in response to E0's binding step budget.

## Two preregistration defects, found after launch and not silently fixed

Both were found by the pre-run reviewer *after* the runs had started, so the
preregistration was left as it was: a document edited after launch is a different artifact
from one edited before. Both are documentation-only; no code reads either text and no gate
value depends on them.

1. **The preregistration contradicts itself about the pilots.** In the G9 rationale it says
   the four 120-step timing pilots were "failing G2a and every other gate". That is false.
   Measured from `outputs/task054-scratch/timing-*-gates.json`, each passed **five** of
   fourteen: E0/E2/E3 passed G4, G5, G8a, G8c, G9; E1 passed G5, G7b, G8a, G8c, G9. The
   same document's Pilots section and the frozen manifest both state this correctly; only
   the G9 rationale is wrong, and it is the more load-bearing of the two. The error runs in
   the direction that flatters the argument being made (that the pilots were worthless and
   so a G9 pass means nothing). **The argument survives**: all four pilots did pass G9 while
   failing G2a at 0.849–0.928, and G5 is a saturated metric TASK-052 already flagged, G8a
   and G8c are floors a non-collapsed latent clears trivially, and G4/G7b say nothing at 120
   steps.
2. **The manifest's Outcome A lacks the provisional clause the document carries.** The
   document says that if E1 were the only arm passing G2a, the decision to keep CEM would be
   provisional on a planner-consistent re-measurement; the manifest's
   `pre_declared_outcomes.A_some_arm_other_than_E0_passes_g2a` states "CEM over this cost is
   kept" flatly. The cause was the author's own guard: the manifest generation script
   asserts `pre_declared_outcomes` is unchanged and aborts otherwise, so the clause could
   not be written through. **Moot in the event** — Outcome A did not fire — but a verifier
   reading only the machine-readable artifact would have got the unqualified version.

## Handoff facts for the next task

The next task is **behaviour cloning with the world model as a critic** rather than as the
forward model of a sampling planner, on the same LeWM backend. It is not designed or
started here. These are the facts it inherits, all from this run's artifacts:

- **The encoder is the component that works, and it is still improving.** A directly
  encoded observation reads the palm–apple offset to **2.3536 cm** (E0 `encoded_start`),
  against v2's 3.13 cm — and v3's one-camera encoders were already 21–24 % better than
  v2's. It is **not** good enough on its own: E0's `encoded_target` of 2.5903 cm alone
  exceeds the 1.5 cm G1 threshold, so the encoder is unfinished, not solved.
- **Nothing collapses.** G8 passes on all four arms; effective rank 6.74–8.23, mean latent
  std 0.850–0.873, collapsed fraction 0.000 everywhere.
- **The held/grasp readouts are near-perfect.** `apple_held` AUROC on the lift cohort is
  0.9992–0.9997. **But this is weak evidence**: the *shuffled-action* AUROC on the same
  cohort is **0.714–0.812 in this run's own four arms** (TASK-052 measured 0.62–0.77),
  so most of it comes from the encoded state and the fused proprioception, not from
  action-conditioned prediction. A critic must not be credited with it.
- **Top-1 regret is 0.0 mm in three of four arms** (E1 is 2.3 mm), against a label-derived
  random-choice baseline of 8.44 mm. **But G6b discriminates nothing** — its preregistered
  null pass rate is 5.9 % and it reads 0.0 mm even for arms whose ranking ρ is 0.29. It must
  not be quoted on its own as evidence that ranking works: the lowest ranking ρ that still
  reads 0.0 mm is **0.33** (v3 A 0.3280, v4 E2 0.3296), both *failing* G6a. The arm with the
  worst ρ in this task, E1 at 0.2865, is the one arm whose regret is **not** 0.0 mm
  (2.3 mm) — so G6b does not even order the arms the way G6a does.
- **What actually fails is the prediction step under motion**, and three redesigns of it did
  not fix it. The persistence baseline is the thing to beat and it has never been beaten:
  G2a is 0.8635 at best here, 0.831 at best in v3, 0.835 in v2.
- **Candidate ranking is fragile**: the control passes G6a and all three interventions break
  it, for reasons this protocol did not identify (see above).
- The corpus, splits and hashes are unchanged and reusable: dataset `028e130576…`, split
  `51f8e09df1…`, action `da987bb079…`. **Test has still never been decoded**, so one
  unbiased check remains available.

## Honest labelling and limits

- **Nothing here is a manipulation result.** No closed loop was run. Learned Apple→Plate
  stays at 0 successes. A gate pass is an offline property of a checkpoint on recorded
  validation data.
- **One seed per arm, one run each.** Every difference between arms is a single-run
  difference. The bootstrap intervals quantify **cohort sampling for fixed checkpoints**
  and say nothing about run-to-run variation. E0 is a revision control at arm B's seed, not
  a replication, and it cannot distinguish "the intervention did nothing" from "this seed
  is unlucky".
- **No interval is available for the G9 contrast** (see above): the bootstrap's
  `rollout_excess` field is a different estimand.
- **Selection and the gates both used val**, which flatters G1 and everything derived from
  it. Test was never decoded, so an unbiased check is still available.
- **E1 was measured with three future actions a horizon-8 planner cannot supply.** Declared
  before the run; it flatters E1, which still failed.
- **The arms are not unmodified LeWM.** Each is the pinned upstream encoder, predictor,
  projector and SIGReg objective (revision `8edfeb33`) plus the shared state fusion, camera
  fusion, readout loss, and — in E1 and E3 — the shared prediction-step conditioning. E1
  and E3 change the recursive rollout only; the one-step teacher-forced term is untouched.
  E2 reweights the latent multistep term only, not the predicted-readout term.
- **Privileged labels** were used for readout-head training targets, for the readout loss's
  per-frame weight and for val scoring cohorts — never as a model input, a planner input, a
  planning cost or a window filter.
- **Latent MSEs are not compared across arms or backends.** The comparison is in physical
  readout units and compute.
- **The native backend is not a frozen arm** of this protocol. It is exercised only by the
  smoke, so the one-line-swap invariant is demonstrated as a software property.
- **The ranking cohort is 18 groups from 20 val resets** at h = 16 while the planner's
  horizon is 8, and 67 of 80 val episodes contribute a moving window.
- The corpus is privileged scripted-collector data with injected perturbations, and every
  limitation of `apple_wide_collection_results_v1.md` still applies.
