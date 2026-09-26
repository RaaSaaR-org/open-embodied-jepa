# Apple→Plate encoder study v1: can a frozen encoder expose the post-look apple? (TASK-062)

**Status: preregistration. No encoder has been trained for this study, and no readout has been
fitted on any feature it introduces.** The only numbers here are TASK-061's published results
and a pre-freeze calibration (`scripts/calibrate_encoder_study.py`, committed with this
document). The calibration fits **no readout**, and **no target or label** (apple position,
expert command, privileged label) enters any of its statistics. It renders frames, using each
root's reset coordinates only to render them, and it measures encoder activations. **The test
split is never decoded. Cohort D (45000–45007, 45100–45107) and cohort C (45300–45339) are never
simulated. `exemption_spent` (`benchmarks/manifests/apple-policy-diagnostics-v1.json`) stays
`false`.**

This is a **perception/representation task**. It trains image encoders, not controllers. It
collects no corpus and preregisters no control formulation: CEM over the world-model cost was
abandoned at TASK-054 and the behaviour-cloning line at TASK-057 (`docs/DECISIONS.md`), and both
clauses still hold. **Learned Apple→Plate is still 0 successes.** Nothing here is a control
result.

Predecessor: [`apple_observation_reprobe_v1.md`](apple_observation_reprobe_v1.md) /
[`_results.md`](apple_observation_reprobe_v1_results.md) (TASK-061, outcome **O-LOOK-RAW**).
Manifest: `benchmarks/manifests/apple-encoder-study-v1.json`.

**The task owner's ordering (2026-09-26).** O-LOOK-RAW's consequence row named two things: a
look-prefix 112 px corpus, and encoder/representation work preregistered separately. The owner
chose the encoder work first. A corpus is only worth collecting once some frozen encoder exposes
the apple in the post-look frame.

---

## 1. The question

TASK-061 showed that after a fixed 8-command look, the 112 px onboard frame carries the apple
position and the expert's first reset-dependent command:

- **L-raw** (raw pixels) passed: T1 0.469 cm [0.419, 0.528], ratio to B-occ 0.242 [0.211, 0.289];
  T2 180/190.
- **L-E0** (the frozen E0 encoder) failed: T1 1.273 cm, ratio 0.656 [0.519, 0.779]; T2 147/190.
- **L-random** (a random-init encoder of E0's architecture, reported only) was about as good as
  E0 on position and better on the sign: T1 1.124 cm, ratio 0.580 [0.513, 0.711]; T2 176/190.
  It failed only the T1 ratio bar.

**Question:** can a change to how the encoder is trained or read out give a **frozen** encoder
whose features pass TASK-061's L-hypothesis probe? The conditions are:
- the same post-look 112 px frames, the same readouts and the unchanged TASK-059 bars;
- only roots the encoder never saw;
- and the encoder must do better than a random-init encoder of the same architecture.

## 2. What is reused unchanged

The probe is TASK-061's L arm, executed by TASK-061's own code. `scripts/probe_observation_reprobe.py`,
`src/embodied_jepa/observation_reprobe.py`, `scripts/probe_info_ceiling.py` and
`src/embodied_jepa/info_ceiling.py` are loaded as they are, and their bytes are pinned (G-hash, §11).

- **Frames.** TASK-061's `measure` renders them: the same 190 train + val roots, the renderer
  warm-up (every renderer renders once and discards the result), the 8-command look, and the
  apple-hidden ablation render. TASK-061's G-render, G-expert, G-look and G-prior run unchanged.
  Only the post-look 112 px frame (`post_112`) and its apple-hidden twin are used.
- **Targets.** T1 is the reset apple xy. T2 and T3 are the sign and value of the expert's
  `right_dx` at the post-look state (TASK-061 §4).
- **Readouts, folds and statistics.** Linear and RBF kernel ridge with nested inner CV. Grouped
  10-fold CV with `default_rng(59)` and TASK-061's fold hash `1dcc9086…c9b172a`. The same 10 000
  paired bootstrap resamples (seed 5901), Wilson intervals, and baselines B-occ, B-maj and
  B-const on the post-look targets.
- **Bars** (TASK-059 §9), on all 190 roots:
  - T1: median ≤ 1.5 cm **and** upper 95 % bound of the ratio to B-occ ≤ 0.6;
  - T2: accuracy ≥ 0.85 **and** Wilson lower bound > B-maj (0.621);
  - T3: upper bound of the MAE ratio to B-const ≤ 0.6.
- **Random-floor definition** (TASK-059 §9, `info_ceiling.beats_random_floor`, unchanged): the
  paired bootstrap 95 % CI of the median T1 error difference (source − floor) lies below 0,
  **and** the source's T2 accuracy is higher than the floor's.
- **Spurious check** (TASK-061 §7.4): the same out-of-fold readouts, applied to the apple-hidden
  frames. If those still beat the prior on all roots, the cue is not the apple.

**What changes:**
- the features (§5);
- the rule for which encoder produces a root's features (§4);
- the multiplicity family (§7);
- the floor as a pass condition (§6).

The bars do not change.

## 3. Why might E0 lose the apple? The explanations and a cheap test

### 3.1 What E0's probe feature is, and how E0 was trained

E0 (`checkpoints/task054-wm-v4/leworldmodel_baseline.pt`, the TASK-054 LeWM baseline) is a ViT:
112 px input, patch 14 (an 8 × 8 token grid), width 128, 4 blocks, 4 heads. The feature TASK-061
read, `FrozenEncoder.features` → `image_features`, is:

**CLS token → projector MLP (BatchNorm) → per-camera linear fusion → a 128-d image feature.**

The world model's latent is that image feature **plus** a learned embedding of the proprioception
(`state_fusion`). E0 was trained with the TASK-054 recipe. Every loss term acts on that **fused**
latent:

- one-step and multistep latent prediction;
- SIGReg at weight 0.09;
- **privileged readout heads**, on the fused latent (encoded and predicted):
  - the core heads include `palm_minus_apple` and `apple_minus_plate`;
  - an **absolute `apple_position`** auxiliary head at weight 0.5 (`readout_auxiliary_weight`,
    `models/readout.py`).

**So E0 was already supervised to localise the apple.** Those labels come from only 170 distinct
resets, and they are fitted on the expert's own trajectories.

### 3.2 Candidate explanations

| id | explanation | what it predicts | tested by |
|---|---|---|---|
| **E-pool** | The apple survives in the spatial tokens. The single CLS token and the projector pool it away. | Token features read the apple, and the pooled feature does not. | **A-tok** |
| **E-collapse** | The image feature sits in a low-rank regime. SIGReg acts only on the fused latent, at 0.09, so the state embedding can supply the spread while the image pathway stays low-rank. | The image feature has a low effective rank on post-look frames, and anti-collapse regularisation **of the image feature itself** helps. | **A-sig** |
| **E-objective** | Latent prediction gives little incentive to keep a small, static object: the apple does not move until grasped, and the arm dominates frame-to-frame change. The readout heads do reward apple localisation, but only through the fused latent and on 170 resets. | A term that forces the image feature to keep **all** image content (pixel reconstruction) helps. | **A-rec** |
| **E-supervision** | The privileged heads let the encoder fit each training reset through its most salient cue (the plate, which `apple_minus_plate` also involves) instead of localising the apple. This would generalise poorly to roots it never saw. | Removing the heads (the plain LeWM objective) helps. | **A-plain** |
| E-capacity | 128 dimensions are too few. | **Argued against, not tested:** E0's image feature uses about 3 effective dimensions of 128 on these frames (§3.3). That is equally consistent with E-collapse; it is not proof that capacity suffices. | — |
| E-input | Patch 14 at 112 px is too coarse for a median 12 px apple. | **Argued against, not tested:** E0's patch embedding and tokens respond to the apple (§3.3). That is a response size, not readability. TASK-061's LH arm (224 against 112 px) is evidence about raw-pixel resolution, not about patch-14 tokenisation. | — |
| E-distribution | The post-look pose is off E0's training distribution. The look is the expert's step-0 command without its reset-dependent `dx`, so the post-look pose lies near, but not on, the expert path. | **Not separable here.** Every arm trains on the same corpus, so this is a **shared confound**. §10's recommended next step under the negative rows (an externally pretrained encoder) does not depend on this corpus. | — |

The evidence already in hand: E0 is worse than a random init of its own architecture (§1). So
**training** moved the image feature away from the apple. That points at what training does:
- the objective (E-objective);
- the regime (E-collapse);
- the supervision (E-supervision);
- or the route from the tokens to the latent (E-pool).

### 3.3 The cheap test, before any training (label-free calibration)

`scripts/calibrate_encoder_study.py` renders, for every root, the post-look frame and two
ablations of it: the apple hidden and the plate hidden. The robot pose after the look is the same
on every root, so every difference between two roots' frames comes from the apple and the plate.
For each representation r it reports:

- **S_apple** = median_i ‖r(x_i) − r(h_i)‖ / median_{i<j} ‖r(x_i) − r(x_j)‖. This is the size of
  the apple's own effect, relative to the typical difference between two roots. S_plate is the
  same with the plate hidden.
- **Effective rank** over the 190 post-look frames.
- R_apple, an energy ratio (see the caveat below).

The artifact is `outputs/task062-encoder-study/calibration-v2.json`, sha256 in the manifest. Its
label-free table is identical to v1's. Selected rows:

| representation | S_apple | S_plate | effective rank |
|---|---|---|---|
| raw pixels | 0.245 | 1.547 | 18.0 |
| E0 patch-embedding tokens | 0.275 | 3.043 | 13.9 |
| E0 final tokens (after the last LayerNorm) | 0.338 | 3.055 | 14.7 |
| E0 CLS token | 0.177 | 3.800 | 7.4 |
| E0 projector output | 0.164 | 4.309 | 4.7 |
| **E0 probe feature** (what TASK-061 read) | **0.151** | 4.140 | **3.2** |
| random-init final tokens | 0.221 | 1.808 | 16.4 |
| random-init CLS token | 0.229 | 3.168 | 11.1 |
| random-init probe feature (TASK-061's L-random) | 0.283 | 3.459 | 10.0 |

**Reading (interpretation, to be tested, not a result):**

- In E0 the apple's relative effect is **largest in the final tokens** (0.338) and **smallest in
  the probe feature** (0.151). Along the head the effective rank falls from 14.7 to 3.2. The
  random init keeps about 10 effective dimensions and does not show that drop.
- In E0 the plate dominates every stage after the patch embedding more than it does in raw pixels
  (S_plate about 3–4, against 1.5). That fits E-supervision as well as E-pool and E-collapse.

**Caveat, disclosed:** R_apple (the apple's effect energy over the across-root variance) does not
agree for the CLS token: 0.527 for E0 against 0.132 for random init. The effect is heavy-tailed:
E0's CLS responds strongly to the apple on some roots and weakly on most. Neither statistic
measures **readability**. The probe measures that, which is why the explanations are tested in
the run rather than assumed. E0-tok (§5.3) is the cheap readout test of E-pool on E0 itself, and
it runs before any training.

**Collapse reference** (also in `calibration-v2.json`). `VisualModel._statistics` over 1024
train-split frames (64 episodes and the frames both drawn with seed 6220; only frames and
proprioception decoded):

| model | image feature: std mean / collapsed fraction / effective rank | fused latent: same |
|---|---|---|
| E0 | 0.876 / 0.000 / 7.0 | 0.935 / 0.000 / 7.5 |
| seed-0 random init | 0.013 / 0.266 / 8.5 | 0.099 / 0.000 / 13.4 |

E0 clears TASK-054's eligibility values (std mean ≥ 0.1, collapsed fraction ≤ 0.05, effective
rank ≥ 2) on the **image feature** by a wide margin. The untrained init does not. §7's collapse
gate is therefore applied to the image feature, which is what the probe reads.

## 4. Held-out design: cross-fitting over the TASK-061 folds

The probe needs all 190 roots, and an encoder may not be tested on a root it was trained on. Both
requirements hold through **two-way cross-fitting**:

- **Halves.** Half **A** is TASK-061 folds 0–4 (95 roots). Half **B** is folds 5–9 (95 roots).
- **Encoders.** Every trained configuration is trained **twice**, from the same initialisation:
  - e(A) is trained only on data of the **train-split** roots in half B;
  - e(B) is trained only on data of the train-split roots in half A.
- **Features.** Each encoder produces features for all 190 roots, but it produces **held-out
  predictions only for its excluded half**:
  - a root in fold k is predicted from the features of the encoder that excluded k's half;
  - the outer readout for fold k is trained and selected on the other 171 roots, using that
    same encoder's features.
- **Seen and unseen roots.** The readout's training roots include roots the encoder saw; the
  held-out fold never does. The readout is therefore fitted on a mix of seen and unseen roots and
  tested on unseen ones. Any mismatch between the two is expected to work against the arm.
- **Training data.** A training root contributes its root episode and its branch episodes
  (grouped by `root_episode_id`):
  - e(A): 84 roots, 336 episodes, 85 627 frames;
  - e(B): 86 roots, 341 episodes, 89 824 frames (3 of its roots have 3 episodes instead of 4).
- **Never used for training:**
  - the **val** split's 20 roots and their episodes (they are always held out, and never decoded
    for training);
  - the test split;
  - any val-based model selection. Each encoder is the **final** state of a fixed budget.
- **State normalisation** of the fused proprioception is fitted on the training half's episodes
  only, with the same Welford moments as `DatasetStore.fit_normalization`.
- **Secondary estimate** (train 170 → val 20, reported only). For val roots in half h, the
  readout is fitted on the 170 train roots with e(h)'s features.

**Why two halves, not ten.** Ten-way cross-fitting would need ten encoders per configuration, and
each E0-budget encoder costs about an hour on MPS. Each encoder here sees half of the train roots
(about 88 000 frames, against E0's 175 451). Every arm and the R0 control (§5.3) pay that cost
equally, and R0 measures it.

**E0 itself is not held out.** E0 was trained on all 170 train roots, and its checkpoint was
selected on the val split. It therefore appears only as the TASK-061 anchor and in a reported-only
diagnostic, never as a decisional arm.

## 5. Arms

### 5.1 Shared training recipe (TASK-054's E0 recipe, cross-fitted)

The recipe is `configs/apple_wm_v4_lewm.yaml` (through `configs/apple_policy_v1.yaml`, hashed),
with the TASK-054 budget (`benchmarks/manifests/apple-world-model-v4.json`, `frozen.training`):

- LeWM backend; model seed 0;
- 15 000 steps, batch 32, horizon 16;
- AdamW at 3e-4 with weight decay 1e-4, cosine decay to 0.1× (the TASK-054 schedule);
- gradient clip 1.0;
- readout heads with privileged training labels, and state fusion, exactly as E0.

The recipe's step is `LeWM.train_step`, unchanged.

The changes from E0's training:

1. training-half data only (§4);
2. no validation pass and no selection: the final weights are the encoder;
3. the window sampler seed is 6200 for e(A) and 6201 for e(B).

**Every model starts from its architecture's seed-0 initialisation, and that initialisation is
its random floor (§6).** A guard checks the digest before training (§11):

- with the readout heads on, the model's weight digest is `cae0ad88…`, which is also TASK-061's
  a1/L-random;
- with them off, it is `f760fb42…`.

The ViT encoder is identical in both. Only the per-camera fusion layer's initialisation differs,
because it is drawn after the heads. Training runs on **MPS**. Feature extraction and all
readouts run on **CPU**, `eval()`, `no_grad`.

### 5.2 The four decisional arms

Each arm makes one change, and each is tied to one explanation of §3.2.

| arm | tests | training change from the recipe | feature read by the probe | floor |
|---|---|---|---|---|
| **A-tok** | E-pool | **none**: this is the R0 model (the recipe as it is) | the **final patch tokens** (`last_hidden_state[:, 1:]`, after the final LayerNorm, 64 × 128 = 8192-d), no pooling | **F-tok** |
| **A-sig** | E-collapse | an extra **SIGReg term on the image feature** (before state fusion), over the window's frames, weight **1.0**. The recipe's SIGReg on the fused latent stays at 0.09. | the probe feature | **F-cls** |
| **A-rec** | E-objective | a **pixel-reconstruction** term: a decoder reconstructs the ImageNet-normalised 112 px frame from the 128-d image feature (before state fusion), weight **1.0** | the probe feature | **F-cls** |
| **A-plain** | E-supervision | **readout heads off** (`readout_heads: false`, one existing config key). This is the plain LeWM objective: prediction, multistep and SIGReg, with state fusion kept. | the probe feature | **F-plain** |

- **A-tok** needs no model of its own. It reads R0's tokens, so it isolates the readout point with
  the objective fixed.
- **A-sig, A-rec and A-plain** keep TASK-061's feature, which is the world model's image
  latent. A pass there would be a drop-in LeWM encoder.
- **Single values, no sweep.**
  - A-sig's 1.0 is about 11× the recipe's SIGReg weight, and it acts directly on the pathway the
    probe reads.
  - A-rec's 1.0 puts the pixel loss (starting near 1) on the same scale as the other terms.
- **A-sig and A-rec share one training step** (`encoder_study.study_train_step`, new code). It is
  `LeWM.train_step` with `observe_sequence` and `VisualModel.optimize` inlined, plus the two
  optional terms. PR 2 must test that with both terms at weight 0 it equals `LeWM.train_step`
  exactly. No hashed model file changes, so E0 still loads under its implementation hash.
- **A-rec's decoder:**
  - architecture: Linear 128 → 256·7·7, then four ConvTranspose2d stages (kernel 4, stride 2,
    padding 1) with channels 256→128→64→32→3 and GELU between them, from 7 to 112 px;
  - initialisation: under `torch.manual_seed(6210)`, inside a forked RNG, so the model's own
    initialisation is untouched;
  - loss: the mean squared error on the **first frame of each window** (32 frames per step);
  - optimiser: its own AdamW with the same learning rate, schedule and weight decay. Gradient
    clipping at 1.0 covers the model and decoder parameters jointly.
- **A-plain** runs `LeWM.train_step` unchanged, with no readout targets.

### 5.3 Reported-only sources (none of them decides anything)

| source | what it is | why it is here |
|---|---|---|
| **L-raw** | raw post-look pixels (TASK-061) | the **ceiling** reference, and a reproduction anchor (§9) |
| **L-E0** | frozen E0 probe feature (TASK-061) | reproduction anchor |
| **F-cls** = L-random | seed-0 random init (heads on), probe feature | **floor for A-sig and A-rec**; also a reproduction anchor |
| **F-tok** | seed-0 random init, final patch tokens | **floor for A-tok** |
| **F-plain** | seed-0 random init with the heads off, probe feature | **floor for A-plain** |
| **R0-cls** | R0 (recipe unchanged, cross-fitted), probe feature | Control: does E0's failure reproduce on held-out roots with half the data? Every arm is also compared with it. |
| **E0-tok** | frozen E0, final patch tokens | **The cheap readout test of E-pool on E0 itself.** It runs before any training. It is **not held out** (E0 saw the 170 train roots), so it can never be decisional. |

**Pre-declared readings.**
- If E0-tok succeeds while L-E0 does not, E-pool is supported on E0 itself, with the held-out
  caveat.
- If R0-cls succeeds, E0's failure did not reproduce under cross-fitting. The arm comparisons are
  then read against R0-cls rather than E0.

## 6. The floors and the ceiling

- **Floor, same architecture, same readout point.** An arm's floor is its own seed-0
  initialisation, read at the arm's own feature:
  - F-tok for A-tok;
  - F-cls for A-sig and A-rec;
  - F-plain for A-plain.

  **An arm must beat its floor**, by `beats_random_floor` (§2), not merely beat the prior.
- **F-cls is TASK-061's L-random** (the same seed-0 model, digest `cae0ad88…`). It reproduces its
  TASK-061 numbers bit for bit (§9): T1 1.124 cm, ratio upper bound 0.711, T2 176/190. To beat
  it, an arm needs a median error clearly below it **and** at least 177/190 on T2.
- **F-tok and F-plain are new.** F-plain differs from F-cls only in the fusion layer's
  initialisation. The tokens are a random linear-plus-attention map of the patches, so F-tok may
  read the apple nearly as well as raw pixels do. **Beating it is expected to be hard, and that is
  the point.** A-tok passes only if training adds something beyond the architecture.
- **Ceiling.** L-raw (0.469 cm, 180/190) is the reference. Being below it is not a failure. It is
  reported for every arm as the paired difference arm − L-raw.

## 7. Pass rule, demotion and multiplicity

**Arm states.** An arm is in exactly one of three states:

- **Not evaluated (mechanical failure).** One of its encoders stopped at the per-encoder cap, or
  hit a non-finite loss or gradient. Its weights are never probed, and it cannot pass. This says
  nothing about its explanation.
- **Evaluated, collapse-demoted.** Both encoders trained. The collapse gate (condition 2 below)
  fails for at least one of them. The arm is probed and fully reported, but it cannot pass.
- **Evaluated.** Both encoders trained, and the gate holds.

An arm **passes** only if all of the following hold:

1. **Trained.** Both of its encoders complete 15 000 steps within the per-encoder cap, with finite
   losses.
2. **Not collapsed.** For each of its two encoders, on 1024 frames sampled (seed 6220) from that
   encoder's own training half, the **image feature** (the probe feature, for every arm) has:
   - `collapsed_fraction` ≤ 0.05;
   - `effective_rank` ≥ 2;
   - `latent_std_mean` ≥ 0.1.

   These are TASK-054's eligibility values, computed with `VisualModel._statistics`. E0 clears
   them on its image feature (§3.3). For A-tok the gate is R0's image feature: the model is
   gated, not the tokens.
3. **Succeeds** on all 190 roots under the unchanged bars (`info_ceiling.evaluate`).
4. **Beats its floor** (`beats_random_floor`, all 190 roots).
5. **Holm rejection** at a family-wise one-sided α = 0.025 over the **four** arms.
   - p_arm = max(p_T1, p_T2, p_T3, p_F):
     - p_T1, p_T2 and p_T3 are TASK-061's (`observation_reprobe.hypothesis_pvalues`, unchanged);
     - **p_F** is the share of the same 10 000 paired resamples (`ic.bootstrap_indices(190)`,
       seed 5901) whose median error difference, arm − floor, is ≥ 0.
   - p_arm = 1 if a point condition fails: T1 median > 1.5 cm, T2 accuracy < 0.85, or T2 accuracy
     not higher than the floor's. A not-evaluated arm has p_arm = 1.
   - The thresholds are 0.00625, 0.00833, 0.0125 and 0.025. Ties go to the listed order: A-tok,
     A-sig, A-rec, A-plain.
   - Conditions 3 and 4 are kept alongside Holm, so no arm passes on a bootstrap share that the
     percentile bar itself would reject.
6. **Not spurious.** The apple-hidden check, with each root's hidden frame featurised by that
   root's own held-out encoder. The ablation renderer must reproduce the un-ablated frame on
   190/190 roots (TASK-061's `ablation_reproduces["L"]`).

**Qualifiers, not rows:**
- "unadjusted only; not a pass": meets conditions 3 and 4 but not Holm;
- "meets the bars, not the floor": meets condition 3 but not 4;
- "collapse-demoted": meets the bars, but fails condition 2;
- "partial information": beats the prior on all roots without succeeding.

## 8. Diagnostics (reported; only the collapse gate of §7.2 decides anything)

- **Collapse**, for every trained encoder:
  - `_statistics` of the image feature and of the flattened final tokens, on its 1024
    training-half frames;
  - the same on the 190 post-look frames.
- **Action sensitivity**, for every trained world model, on 512 training-half windows of horizon 8
  (seed 6230). The statistic is the median over windows of ‖ẑ₈(a) − ẑ₈(a_π)‖ / ‖ẑ₈(a) − z₈‖:
  - ẑ₈ is the 8-step rollout;
  - a_π is the actions of another window (a seeded derangement);
  - z₈ is the encoded target.

  A world model that ignores its actions scores near 0.
- **The label-free table of §3.3**, recomputed for every trained encoder on the roots of its own
  held-out half.

## 9. Reproduction anchors (G-anchor)

Before any encoder is trained, the run recomputes **L-raw, L-E0 and L-random (= F-cls)** through
TASK-061's code on the frames it has just rendered. It compares them with TASK-061 run-1's report
(`outputs/task061-observation-reprobe/run-1/report.json`, sha256 `289470f4…fa88`, pinned in
G-hash). Compared are:

- every field of `results[name]` that `evaluate_source` and the stratum masks produce: the
  primary estimate on every stratum, the secondary estimate and the per-fold selections;
- every root's out-of-fold xy, dx and dy prediction in `per_root`.

**Handling, fixed now:**
- **Bit-exact** (max |Δ| = 0 everywhere): the anchors hold.
- **Any T2 count, pass/fail flag, `succeeds`/`beats_prior` reading or selection differs, or any
  numeric max |Δ| > 1e-6:** G-anchor fails and the run is **V**. The probe would not be TASK-061's
  instrument. The owner is told before any repeat, because a repeat on the same code would
  presumably fail the same way.
- **A numeric difference with 0 < max |Δ| ≤ 1e-6, and every count, flag and selection identical:**
  - the run continues and the difference is recorded as a reproduction caveat;
  - it is escalated to the task owner before the results document is written;
  - no row changes.

  TASK-061's own anchor reproduced TASK-059 bit for bit on this machine, and kernel-ridge values
  are not bit-reproducible across machines (BLAS). The run therefore uses the same machine.
- **Warm-up rule:** the frames come from TASK-061's own `measure`, which warms up every renderer
  (one discarded render) before any frame is kept. The calibration's renderer, reused for the
  label-free table, does the same, and its post-look frames must equal `measure`'s byte for byte.

## 10. Pre-declared outcomes (first matching row)

| row | condition | reading | next task implied (a recommendation; the owner chooses) |
|---|---|---|---|
| **V** | a guard fails (§11, incl. G-anchor), or the run stops before writing a complete report (crash, global wall cap) | nothing is read | one repeat (§12) |
| **O-ENC-LATENT** | A-sig, A-rec or A-plain passes | A changed objective, regime or supervision gives LeWM's own 128-d image latent the apple, on held-out roots, beyond its random init. | **Preregister the look-prefix 112 px corpus with that arm's recipe as the encoder.** If more than one arm passes, the one with the smallest p_arm is used (ties: listed order). No control formulation is preregistered until the encoder, retrained on that corpus, passes this probe on its decision frames. |
| **O-ENC-TOKENS** | A-tok passes, and no other arm does | The recipe's tokens carry the apple beyond random init. Its pooled latent does not. | **Preregister the look-prefix corpus, plus, separately preregistered, a LeWM variant whose latent is the token grid (or a pooling that keeps it).** The pooled 128-d latent is not used for the apple. |
| **O-ENC-INCOMPLETE** | no arm passes, and at least one arm was **not evaluated** (mechanical failure, §7) | The explanations of the failed arms are untested. | **The abandonment clause does not fire.** The owner decides whether a disclosed amendment re-runs the unevaluated arm or arms. |
| **O-ENC-ARCH** | no arm passes, every arm was evaluated, and at least one arm **succeeds** (bars) and is not spurious | Features of an evaluated arm expose the apple, but no arm meets every pass condition (floor, Holm or collapse gate). | **The abandonment clause fires** (below). Recommended next: one preregistered test of an externally pretrained frozen encoder against its own random-init floor, on this probe, before any corpus. |
| **O-ENC-NONE** | no arm passes, every arm was evaluated, and no arm succeeds without being spurious | No tested change to the readout point, regime, objective or supervision gives a frozen encoder that exposes the apple on held-out roots. | **The abandonment clause fires.** Recommended next: as for O-ENC-ARCH. If that test also fails, no look-prefix corpus is collected. The remaining route is then the observation change TASK-061 already validated: the `overview` camera, a **hardware/workspace change** on the robot. |

**Abandonment clause (for O-ENC-ARCH and O-ENC-NONE).**
- **What closes.** The line *"train a LeWM-family encoder on `apple-wide-v1` train-split frames so
  that its frozen features expose the post-look apple"*. No further readout-point, regularisation,
  objective or supervision variant of the TASK-054 recipe is preregistered on this corpus without
  new evidence of a different kind.
- **Recorded as untested, not refuted:** capacity, input handling and the shared E-distribution
  confound. They are argued against (§3.2), and this closure is limited by that. The recommended
  next step (an encoder that was not trained on this corpus) does not depend on them.
- **What does not close:** the LeWM backend, the encoder as a component, and the product goal.

**Always reported, whatever the row:**
- every arm's numbers, p-values, Holm step, floor comparison, spurious check, state and collapse
  diagnostics;
- every reported-only source (§5.3) on every stratum, and the secondary estimate;
- the paired comparisons:
  - each arm with its floor, with L-raw and with R0-cls;
  - A-tok with E0-tok;
  - R0-cls with L-E0;
- the rows that also match further down.

**In every outcome:** nothing is refitted, re-thresholded or retrained after the numbers are seen.
Learned Apple→Plate stays at 0 successes, and `exemption_spent` stays `false`. The executing agent
recommends a next task and does not choose it.

**Stated in advance.**
- **O-ENC-ARCH and O-ENC-NONE are live possibilities.**
- F-tok may well succeed on its own, which would make A-tok's floor hard to beat.
- Each encoder sees half the training data that E0 saw.
- There is one training run per arm and half. Training-seed variance is not measured.

## 11. Guards: any failure voids the run

| guard | condition |
|---|---|
| **G-hash** | Every file in the manifest's `hashes` matches. That covers:<br>• TASK-061's runner and module, TASK-059's runner and `info_ceiling.py`;<br>• TASK-061's manifest and run-1 report;<br>• this study's calibration script and artifact;<br>• the E0 checkpoint and the task056 a1/a2 checkpoints;<br>• the configs and the dataset manifest;<br>• `models/base.py`, `models/lewm.py`, `models/readout.py` and `policy.py`.<br>Every episode file is checked against the dataset manifest's sha256 before it is decoded. The encoder digests (E0, random) match TASK-061's. The tree is clean. |
| **G-init** | Before training, every model's weight digest equals its architecture's seed-0 digest (§5.1). Every floor's features come from a model with that digest. |
| **G-split** | TASK-061's G-split: exactly the 190 train + val roots and the fold hash. In addition, every episode decoded for training must:<br>(a) be in the dataset's train split;<br>(b) have its `root_episode_id` among the train roots of the encoder's training half;<br>(c) have no root in the half for which the encoder produces held-out predictions.<br>The training root sets of e(A) and e(B) are disjoint, and together they are exactly the 170 train roots. No val, test or holdout episode is ever decoded for training. |
| **G-render, G-expert, G-look, G-prior** | TASK-061's, unchanged, run by TASK-061's `measure` and checks. The calibration renderer's post-look frames must equal `measure`'s. |
| **G-anchor** | §9, including that the stage code (§5.3) reproduces TASK-061's E0 and L-random features exactly. |

**Not guards.** An arm-level event never voids the run. Instead:
- a per-encoder budget stop, or a non-finite loss or gradient, makes the arm **not evaluated**;
- a failed collapse gate makes the arm **collapse-demoted**.

Any other exception is a crash, and a crash is V.

**PR 2's tests** must exercise, in both directions:
- every guard, including G-anchor's three branches, G-init and G-split's half-disjointness;
- the cross-fitted nested CV. With identical features for both halves it must equal
  `info_ceiling.nested_cv` exactly, and each held-out fold's features must come from the encoder
  that excluded it;
- the study step, which equals `LeWM.train_step` with both extra terms at weight 0;
- p_F, the floor condition and the Holm step-down over four arms;
- the spurious check with per-half hidden features;
- the three arm states and every outcome row, including O-ENC-INCOMPLETE;
- a crash that still writes `report.json` with outcome V and a `void_reason`;
- non-finite values written as `null` and listed in `non_finite_fields`.

## 12. Void rule

- A run that stops early for any reason (a guard, a crash, the global wall cap) is **V**. Nothing
  in it is read.
- **Exactly one from-scratch repeat** is allowed, into `run-2`: the same seeds, caps and device,
  and a clean tree. Every encoder is retrained. MPS training is not bit-deterministic, so the
  repeat's weights will differ; their digests are recorded.
- **A second void closes TASK-062 as INCONCLUSIVE.**
- A fix between the runs is limited to runner mechanics. It goes through a reviewed PR and a fresh
  pre-run GO, and it is disclosed in the results.
- Any owner ruling is recorded with a UTC timestamp before the run it affects.

## 13. Budget, device, seeds, recording

- **Device:** training on **MPS**; frames, features, readouts and statistics on **CPU**.
- **Caps:**
  - **7200 s per encoder.** E0's full-data run took 3512 s, including 16 validation passes. An
    encoder that stops at the cap makes its arm not evaluated; that is not a void.
  - **18 h global wall cap** (64 800 s) for the whole run. Exceeding it is V.
  - Expected: about 8 encoders × 1 h, plus about 45 min for frames, anchors, diagnostics and
    readouts.
- **Seeds:**
  - model initialisation 0 (every arm, both halves, and the floors);
  - window samplers 6200 (e(A)) and 6201 (e(B));
  - A-rec decoder 6210;
  - collapse sample 6220;
  - action-sensitivity sample 6230;
  - probe seeds unchanged: folds 59, inner 5900 + k, bootstrap 5901.
- **Recorded:**
  - the code revision (with a clean tree required), and every hash;
  - the training episode ids per encoder, with their sha256;
  - per-encoder initial and final weight digests, training curves (every 25 steps), elapsed
    time and completion;
  - collapse and action-sensitivity diagnostics;
  - per-root out-of-fold predictions for every source, and selections per fold;
  - p-values, Holm steps, floor comparisons and spurious checks;
  - the anchor comparison;
  - elapsed time and peak RSS.
- **Output:** `outputs/task062-encoder-study/run-1/`, with the encoders under `encoders/`. The
  runner refuses to overwrite. Nothing under `data/` or `checkpoints/` is written or modified.

## 14. Pre-freeze calibration (fits nothing)

```
uv run --no-sync python scripts/calibrate_encoder_study.py \
    --output outputs/task062-encoder-study/calibration-v2.json
```

- **What it reads.**
  - The 190 roots of the frozen plan, checked against the dataset's splits. It renders their
    post-look frames through TASK-061's `make_robot`, warm-up and look. Each root's reset
    coordinates are used only to render.
  - For the collapse reference, the frames and proprioception of 64 train-split episodes
    (`DatasetStore.read_episode`).
  - It builds E0 and the random init through TASK-061's encoder loader, and checks both digests.
- **What it does not read.** No target, no expert command and no label sidecar enters any
  statistic, and it fits nothing. The ablation renderer reproduces the post-look frame on 190/190
  roots.
- **Superseded artifacts.** Both are hashed in the manifest.
  - `calibration-draft-1.json` mislabelled the energy ratio as a "variance fraction" and divided
    it by 2n.
  - `calibration-v1.json` had no collapse reference.

  Every S value and every effective rank is identical across the three files, and v1's and v2's
  label-free tables are identical in every field.

## 15. Not done (declared)

- No controller is trained or evaluated. No closed-loop attempt is run. No corpus is collected,
  and no dataset is written.
- No cohort is opened. The test split is not decoded, and cohorts C and D are never simulated.
  - `DatasetStore.verify`, reached through the encoder loader, reads episode bytes only to check
    their sha256, as in TASK-059 and TASK-061.
- No change is made to `models/`, `policy.py`, `simulation.py`, `info_ceiling.py`,
  `observation_reprobe.py`, TASK-061's runner or any checkpoint. The study step, the A-rec decoder
  and the cross-fitting live in new code.
- **No encoder is trained on post-look frames.** The training data is the existing train-split
  corpus. That leaves E-distribution as a shared, untested confound (§3.2).
- No external pretrained encoder is tested. That is the recommended next step under O-ENC-ARCH and
  O-ENC-NONE, and it would need its own licence and provenance review (`docs/DEPENDENCIES.md`).
- Capacity, input handling and augmentation have no arm. The first two are argued against, not
  tested (§3.2). Augmentation has no specific support in the TASK-061 evidence.
- Each configuration has one training run per half. Seed variance is not measured.

## 16. Process

1. **PR 1:** this document, the manifest, the calibration script and the task card. It merges on
   an independent reviewer's **reported** APPROVE and green CI.
2. **PR 2:** the runner, the cross-fitting, study-step and decoder code, and the tests of §11. It
   merges on a reported APPROVE and green CI.
3. **The gated run** starts only on the pre-run reviewer's **reported** verdict, delivered as a
   message, never on a review file read from disk.
4. **PR 3:** the results document, the results manifest, a summarize script and the recommended
   next task. The card goes to done.
5. **One task, one agent.** A protocol defect found after the freeze is escalated to the task
   owner and fixed only through a disclosed amendment.
