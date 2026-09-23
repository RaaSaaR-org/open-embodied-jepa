# Apple world model v3: results (TASK-052)

**All four arms FAIL the preregistered gate set.** No arm passes the primary gate G1
(median palm–apple readout error at h = 8 on moving windows, ≤ 1.5 cm): the best arm is
3.30 cm, 2.2× the threshold. No arm passes G2a either, so **no arm beats its own
persistence readout by the required margin**. By the protocol's pre-declared readings the
closed loop **does not start**. Learned Apple→Plate remains at **0 successes**; nothing in
this document is a control result, a closed-loop result or working manipulation. Every
number here is offline evaluation of learned models on recorded validation data. The
protocol is `apple_world_model_v3.md`, written at `dbba278` and last amended by the
pre-run review at `e2f8227`, which is also the revision every arm was trained and
evaluated at.

**Both of v3's headline hypotheses were wrong in the direction they were predicted to
help.** Adding a second camera made readout precision markedly *worse*, and the v3 readout
shaping (the motion-weighted loss together with the auxiliary position targets) made the
CEM-relevant ranking metrics *worse*. Those are negative results and they are kept.

## Runs

Four runs, each executed once, sequentially on one machine, each from the same clean
committed checkout (no two runs overlapped; see `outputs/task052-runner.log`).

| | **A** `leworldmodel` | **B** `leworldmodel_onboard` | **C** `leworldmodel_v2_readout` | **D** `leworldmodel_fine` |
|---|---|---|---|---|
| Role | primary | camera ablation | readout ablation | resolution ablation |
| Config | `apple_wm_v3_lewm.yaml` | `apple_wm_v3_lewm_onboard.yaml` | `apple_wm_v3_lewm_v2_readout.yaml` | `apple_wm_v3_lewm_fine.yaml` |
| Cameras | onboard + hand crop | onboard | onboard + hand crop | onboard |
| Patch size | 14 (8×8 = 64 tokens) | 14 | 14 | **8** (14×14 = 196) |
| `readout_moving_weight` | 3.0 | 3.0 | **0.0** | 3.0 |
| `readout_auxiliary_weight` | 0.5 | 0.5 | **0.0** | 0.5 |
| Steps | 15,000 / 15,000 | 15,000 / 15,000 | 15,000 / 15,000 | 15,000 / 15,000 |
| Status | `completed` | `completed` | `completed` | `completed` |
| Selected step | 10,000 | **15,000** (the last) | 10,000 | 13,000 |
| Selection score (val) | 5.99 cm | 3.54 cm | 6.27 cm | 3.35 cm |
| Parameters | 2,431,782 | 2,415,398 | 2,431,782 | 2,381,606 |
| Wall clock | 5,405 s | 3,513 s | 5,363 s | 7,381 s |
| Process CPU | 1,822 s | 1,562 s | 1,760 s | 1,635 s |
| Peak host RSS | 15.58 GB | 9.17 GB | 15.50 GB | 9.17 GB |
| Evaluation wall clock | 15.7 s | 13.7 s | 15.7 s | 17.5 s |
| Checkpoint SHA-256 | `1e8fe9f37d94…` | `7e2e97352ab6…` | `d3989adc3793…` | `49aecbb114e1…` |
| Implementation SHA-256 | `649cc14345ee…` | the same | the same | the same |

Every arm's `model_config` in its `run.json` is identical except for the cells in bold or
otherwise marked above; the parameter deltas are exactly the declared ones (A − B =
16,384 = one 128×128 camera projection; B − D = −33,792 = the patch and position
embedding change).

- **Code revision** `e2f822765599c920583033b5061f44ff08ba689e` for all four training runs
  and all four evaluations, with `dirty: false` (`--require-clean`). The Python source
  SHA-256 is `476a527ed6ae85beddff171bd59c5785539d7b49aeda12355112669e4358710f` for every
  run and every evaluation, and it still reproduces from the tree in this PR.
- **Seed** 0 (model init and window sampler) for all four arms.
- **Device** MPS, Apple M5 Pro, 48 GB; macOS 26.5.1 arm64; Python 3.12.13, torch 2.14.0,
  NumPy 2.5.3.
- **Budget** the frozen cap is `max_seconds: 10800` per arm. No arm hit it: the longest
  (D) used 7,381 s. Training totalled 21,662 s = 6.02 h against the preregistered
  estimate of 5.88 h. Batch 32 × horizon 16, AdamW lr 3e-4 cosine-decayed to 3e-5, weight
  decay 1e-4, grad clip 1.0, validation every 1,000 steps — all as frozen.
- **Data** `data/apple-wide-v1`, dataset manifest SHA-256 `028e130576c0…`, split hash
  `51f8e09df1c0…`, action hash `da987bb0798c…`, all identical to TASK-050. 677 train
  episodes (175,451 observations, 164,822 horizon-16 windows), 80 val episodes (20,406
  observations). Selection-window SHA-256 `16eb332c3087…`, identical across all four arms.
- **`test_episodes_decoded: 0`** in all four `run.json` files and all four gate reports.
  That field is a literal emitted by the reporter, so the real guarantee is structural:
  `load_split()` refuses any split other than `train`/`val` and raises if the requested
  episodes intersect test or holdout. Splits are grouped by whole reset
  (`split_policy.group_by: session_id`, `split_seed: 48`, plan SHA-256 `15ed1a99…`), and a
  root-prefix check finds 0 shared roots between train, val and test. Normalization is fit
  on train only (`normalization.fit_split: "train"`, 677 episode ids identical to
  `splits.train`; the runner raises otherwise). Updates used train only; selection and
  every gate used val only.
- **Artifacts** (git-ignored, under the main checkout)
  `checkpoints/task052-wm-v3/{leworldmodel,leworldmodel_onboard,leworldmodel_v2_readout,leworldmodel_fine}.{pt,latest.pt,run.json,metrics.jsonl}`
  and `outputs/task052-wm-v3/<name>-val-gates.json`, plus `outputs/task052-runner.log`.
  Nothing outside `*/task052-wm-v3/` was written; TASK-050's `task050-wm-v2` artifacts are
  untouched.

## Gate table (val only; h = 8 unless stated)

| Gate | Threshold | A (primary) | B (onboard) | C (v2 readout) | D (patch 8) |
|---|---|---|---|---|---|
| **G1** palm–apple, moving windows | ≤ 1.5 cm | **6.62 cm FAIL** | **3.65 cm FAIL** | **6.11 cm FAIL** | **3.30 cm FAIL** |
| **G2a** ÷ own persistence readout | ≤ 0.8 | **1.010 FAIL** | **0.876 FAIL** | **0.940 FAIL** | **0.831 FAIL** |
| **G2b** ÷ shuffled-action control | ≤ 0.8 | 0.765 PASS | 0.666 PASS | 0.745 PASS | 0.683 PASS |
| **G3** apple–plate, valid windows | ≤ 2.0 cm | **2.79 cm FAIL** | 1.97 cm PASS | **2.95 cm FAIL** | 1.80 cm PASS |
| **G4** apple height, grasp cohort | ≤ 1.0 cm | 0.346 cm PASS | 0.149 cm PASS | 0.224 cm PASS | 0.175 cm PASS |
| **G5** `apple_held` AUROC, lift cohort | ≥ 0.85 | 0.9996 PASS | 0.9995 PASS | 0.9993 PASS | 0.9997 PASS |
| **G6a** within-state ranking ρ, h = 16 | ≥ 0.5 | **0.328 FAIL** | 0.544 PASS | 0.516 PASS | **0.378 FAIL** |
| **G6b** median top-1 regret, h = 16 | ≤ 4 mm | 0.0 mm PASS | 0.0 mm PASS | 0.0 mm PASS | 0.0 mm PASS |
| **G7a** siblings own < swapped, h = 16 | ≥ 0.70 | **0.613 FAIL** | **0.698 FAIL** | 0.717 PASS | 0.708 PASS |
| **G7b** sibling divergence ρ, h = 16 | ≥ 0.5 | 0.592 PASS | 0.577 PASS | 0.571 PASS | 0.592 PASS |
| **G8a** collapsed fraction | ≤ 0.05 | 0.000 PASS | 0.000 PASS | 0.000 PASS | 0.000 PASS |
| **G8b** effective rank | ≥ 4 | 5.12 PASS | 7.74 PASS | 5.36 PASS | 7.54 PASS |
| **G8c** mean latent std | ≥ 0.1 | 0.696 PASS | 0.868 PASS | 0.732 PASS | 0.877 PASS |
| Gates passed | 13 | 8 / 13 | 10 / 13 | 10 / 13 | 10 / 13 |
| **Overall** | all gates | **FAIL** | **FAIL** | **FAIL** | **FAIL** |

Every threshold and comparison direction in the gate reports matches
`benchmarks/manifests/apple-world-model-v3.json` (`frozen.gates`) and the preregistration
table verbatim, and is identical across the four arms. No threshold was changed.

## The one-factor readings

The design is not a full factorial. Exactly three pairs differ by one declared factor, and
those are the only comparisons that carry causal weight. **Each rests on a single seed per
arm**, so every difference below is a one-run difference; none of them has an error bar.

### A ↔ B — the second camera (the v3 change): it HURT, and by a lot

Removing the FK-placed hand crop, changing exactly one 128×128 fusion tensor and nothing
else:

| | A (onboard + hand crop) | B (onboard) |
|---|---|---|
| G1 palm–apple, moving | **6.62 cm** | **3.65 cm** |
| G2a ÷ persistence | 1.010 | 0.876 |
| G3 apple–plate | 2.79 cm | 1.97 cm |
| palm–apple, all valid windows | 1.26 cm | 0.78 cm |
| G6a ranking ρ | 0.328 | 0.544 |
| G7a own beats swapped | 0.613 | 0.698 |
| G8b effective rank | 5.12 | 7.74 |
| image-only effective rank | 4.53 | 7.19 |
| Peak host RSS | 15.58 GB | 9.17 GB |

The second camera is worse on **nine of the thirteen gates**, at 1.7× the memory and 1.5×
the wall clock. It is better on only two, both by a hair: G5 (0.99962 against 0.99949, a
saturated metric) and G7b (0.592 against 0.577). G6b and G8a are ties. The two-camera arms
(A and C) both sit
at 6.1–6.6 cm with effective rank 5.1–5.4; the one-camera arms (B and D) both sit at
3.3–3.6 cm with effective rank 7.5–7.7. That pattern is consistent across the design, but
only A ↔ B is the controlled contrast.

The drop in effective rank — of the *image pathway alone*, 4.53 versus 7.19 — is the most
informative descriptive number: summing two camera embeddings produced a lower-variety
latent, not a richer one. This is a mechanism hypothesis, not a measured cause; the
protocol did not include a control that would separate "the crop adds noise" from "the
additive fusion destroys variety".

### A ↔ C — the v3 readout loss shaping: it HURT the CEM-relevant metrics

Arm C drops **both** parts of the v3 readout shaping at once — the motion weighting
(`readout_moving_weight` 3.0 → 0.0) and the auxiliary absolute-position readouts
(`readout_auxiliary_weight` 0.5 → 0.0). They were preregistered as one factor and they
**cannot be separated by this data**.

| | A (v3 shaping) | C (uniform, no auxiliary) |
|---|---|---|
| G1 palm–apple, moving | 6.62 cm | **6.11 cm** |
| G4 apple height | 0.346 cm | **0.224 cm** |
| G6a ranking ρ | 0.328 | **0.516** |
| G7a own beats swapped | 0.613 | **0.717** |
| G2a ÷ persistence | 1.010 | **0.940** |
| G3 apple–plate | **2.79 cm** | 2.95 cm |

Removing the shaping improved five of the six rows shown, including the two gates C passes
and A does not (G6a, G7a). The motion weighting was introduced specifically to improve
readout precision on moving windows, and on that exact metric the arm without it is 0.5 cm
*better*. Over the full gate set, A is better than C only on G3 (by 0.16 cm), G5 (0.99962
against 0.99934, saturated) and G7b (0.592 against 0.571).

### B ↔ D — the encoder patch grid: it helped, modestly, at 2.1× the cost

Patch 14 → 8 (64 → 196 tokens per camera), one camera in both arms:

| | B (patch 14) | D (patch 8) |
|---|---|---|
| G1 palm–apple, moving | 3.648 cm | **3.304 cm** |
| G2a ÷ persistence | 0.876 | **0.831** |
| G3 apple–plate | 1.97 cm | **1.80 cm** |
| G6a ranking ρ | **0.544** | 0.378 |
| G7a own beats swapped | 0.698 | **0.708** |
| G8b effective rank | **7.74** | 7.54 |
| Wall clock | 3,513 s | 7,381 s |

A 0.34 cm (9 %) improvement on the primary gate for 2.1× the training time, and a clear
*regression* on the ranking gate. The finer grid moves G1 in the right direction but
nowhere near the 1.5 cm threshold, and it is not a free win.

**This 0.34 cm should probably be read as noise.** Within arm D's own single run the
validation selection score swings 7.50 → 12.34 → 15.41 → 4.99 cm between steps 5,000 and
8,000 — step-to-step jitter an order of magnitude larger than the B ↔ D gap. With one seed
per arm there is no run-to-run variance estimate, so B ↔ D is not distinguishable from
noise. The A ↔ B camera gap (2.97 cm) is large enough to survive that objection; B ↔ D is
not.

**The largest single lever measured on G1 is the camera set (−2.97 cm from removing the
second camera), not the patch grid (−0.34 cm).**

## Comparison with v2 (TASK-050)

The nearest v2 analogue is its primary LeWM/onboard arm. It is **not** a controlled
successor of any v3 arm: v3 changed the readout-head width (256 → 512), the
proprioception normalization policy and the code revision for every arm. The comparison is
for orientation only.

| | v2 LeWM/onboard | v3 B (onboard) | v3 D (patch 8) |
|---|---|---|---|
| G1 palm–apple, moving | 3.6453 cm | 3.6481 cm | **3.3043 cm** |
| G2a ÷ persistence | 0.8351 | 0.8763 | 0.8308 |
| G2b ÷ shuffled | 0.699 | 0.666 | 0.683 |
| G3 apple–plate | 1.93 cm | 1.97 cm | 1.80 cm |
| palm–apple, all valid windows | 0.77 cm | 0.78 cm | 0.92 cm |
| v2's cross-window cost ρ (descriptive) | 0.165 | 0.097 | 0.228 |

**v3 arm B reproduces v2's G1 to within 0.03 mm (3.6481 cm against 3.6453 cm) and is
*worse* on G2a (0.876 against 0.835).** The near-equality is a coincidence of two
independent runs, not the same number reported twice — the two checkpoints, revisions and
configurations differ — but the reading is plain: with everything v3 added except the
finer patch grid, the one-camera arm did not move the primary gate at all. Only arm D
improved it, by 0.34 cm. On v2's own descriptive cross-window cost Spearman, three of the
four v3 arms are *worse* than v2's 0.165.

## The genuinely positive result: within-state candidate ranking

This is the one place where v3 measures something better than v2, and it is the metric a
CEM actually uses.

| h = 16, 18 ranked groups, 68 pooled points | A | B | C | D |
|---|---|---|---|---|
| G6a pooled within-state ρ | 0.328 | **0.544** | **0.516** | 0.378 |
| mean per-group ρ | 0.328 | 0.544 | 0.528 | 0.389 |
| G6b median top-1 regret | **0.0 mm** | **0.0 mm** | **0.0 mm** | **0.0 mm** |
| mean top-1 regret | 3.47 mm | 3.30 mm | 3.24 mm | 3.02 mm |
| shuffled-action control, pooled ρ | −0.093 | +0.058 | −0.093 | −0.013 |
| shuffled-action control, median regret | 8.44 mm | 6.57 mm | 9.46 mm | 8.39 mm |
| label-derived random-choice median regret | 8.44 mm | 8.44 mm | 8.44 mm | 8.44 mm |

- **Median top-1 regret is exactly 0 m in all four arms**: in at least half of the 18
  groups the model picks the truly best candidate. The label-derived random-choice
  baseline is 8.44 mm absolute / 0.531 normalized (a unitless ratio, not a length), and
  the shuffled-model control is 6.6–9.5 mm — so the pass is not an artefact of easy groups.
  **But G6b discriminates nothing here**: it is 0.0 mm even for arm A, whose ranking ρ is
  0.328, and its preregistered null pass rate is 5.9 %. It must not be quoted on its own
  as a success.
- **G6a passes for B and C** (0.544, 0.516) against v2's cross-window ρ of 0.165. The
  preregistration's Monte-Carlo null (20,000 draws of a uniformly random ranker over the
  real val cohort) puts G6a's pass rate at 1.0e-4 and G6b's at 5.9 %, so G6b alone is weak
  and must not be read on its own, but **the G6a+G6b pair is not passable by chance — and
  arms B and C pass the pair.**
- **The cohort is narrow: 18 ranked groups of 3–4 candidates from 20 val resets, 68 pooled
  points.** This is necessary, not sufficient, evidence, exactly as preregistered. It
  cannot establish that a planner will work.
- `start_state_max_label_mismatch_m` is 0.0 for every arm: the siblings' start states
  coincide exactly, so the only thing differing between candidates is the actions.
- Descriptively at other horizons the ranking is weaker: pooled ρ at h = 8 (10 ranked
  groups) is 0.26 / 0.36 / 0.46 / 0.38 and at h = 32 (17 groups) 0.335 / 0.359 / 0.416 /
  0.395. **The planner's own horizon is 8**, and no arm reaches ρ ≥ 0.5 there. The gate
  horizon of 16 was chosen from val labels before any v3 model existed, because only 10 of
  20 groups clear the 5 mm spread rule at h = 8; that is a limitation of the cohort and it
  bites here.

**So the models order candidate actions far better than they localise the apple.** That is
the single most useful thing this protocol learned.

## What else the numbers say

- **Arm A is worse than doing nothing.** G2a = 1.010 means A's 8-step rollout readout is
  1 % *further* from the truth than A's own persistence readout on the same windows
  (6.62 cm against 6.55 cm). The primary arm's dynamics contribute nothing to palm–apple
  accuracy.
- **Every arm still beats its shuffled-action and zero-action controls** (G2b passes
  everywhere; zero-action medians 8.59 / 5.21 / 8.94 / 4.68 cm against predicted 6.62 /
  3.65 / 6.11 / 3.30 cm). The actions carry information; the readout is what is imprecise.
- **The all-window / moving-window gap that v3 set out to close is still there.** At
  h = 8 the palm–apple error over all 3,961 valid windows is 1.26 / 0.78 / 0.87 / 0.92 cm,
  against 6.62 / 3.65 / 6.11 / 3.30 cm on the 1,462 moving windows, with a true
  displacement of 2.49 cm. A predictor that saw the offset exactly and then assumed it
  froze would still beat every arm on the moving windows.

  | h | valid | moving | B predicted | B persistence | B shuffled | B zero-action | true displacement |
  |---|---|---|---|---|---|---|---|
  | 1 | 4,010 | 131 | 3.10 cm | 3.26 | 3.83 | 3.27 | 1.30 cm |
  | 4 | 3,991 | 1,004 | 4.25 cm | 4.06 | 4.82 | 4.97 | 1.67 cm |
  | 8 | 3,961 | 1,462 | 3.65 cm | 4.16 | 5.48 | 5.21 | 2.49 cm |
  | 16 | 3,901 | 1,943 | 3.17 cm | 4.02 | 5.94 | 6.04 | 3.14 cm |

- **G5 is passed but is weak evidence, and the control says so.** The lift-cohort
  `apple_held` AUROC is ≈ 1.0 for every arm, but the *shuffled-action* AUROC on the same
  cohort is still 0.619 / 0.768 / 0.708 / 0.764 — far above chance. Most of that AUROC
  comes from the encoded state and the fused proprioception, not from action-conditioned
  prediction.
- **No collapse anywhere** (G8 passes on all four), but the two-camera arms have a
  materially lower effective rank (5.12, 5.36) than the one-camera arms (7.74, 7.54), and
  the image-only pathway shows the same split (4.53, 4.75 versus 7.19, 7.09).
- **The step budget binds for the one-camera arms.** B's validation curve falls
  monotonically to its last point and B's selected checkpoint *is* step 15,000 (score
  3.54 cm, still improving); D's best is step 13,000 with 15,000 within 0.02 cm. A and C
  plateau near 6.0–6.3 cm from about step 8,000. So "train B or D longer" is a live,
  untested option; "train A or C longer" is not supported by their curves.
- **Descriptive grasp outcome from the pre-grasp state** (h = 64, 63 siblings, 29
  positive): AUROC 0.877 / 0.822 / 0.790 / 0.941. **`apple_dropped`** AUROC over all
  windows: 0.969 / 0.972 / 0.973 / 0.975.
- **v2's absolute approach-cost calibration** (the old G6, threshold ln 1.5 = 0.405),
  reported descriptively: 1.048 / 0.906 / 1.027 / 0.772. Every arm would still fail it.

## Reading, as pre-declared

The protocol's decision rule is applied as written, not re-interpreted after the fact.

- **No arm passes → the closed loop does not start.** That is the decision, and it holds
  regardless of which arm one prefers.
- The protocol then selects the next protocol by the earliest failing group. **The four
  arms do not agree on which branch that is, and this document does not pick the more
  flattering one.**
  - Branch 1 (G8, collapse) is not triggered: G8 passes on all four arms.
  - Branch 2 (G2b or G7 fails → action conditioning) **is** triggered for **arms A and B**,
    where G7a fails (0.613, 0.698 against ≥ 0.70). Its instruction is to redesign the
    action conditioning and *not* attack readouts again. Note the tension with the rest
    of the evidence: G2b passes comfortably on every arm, G7b passes on every arm, and
    the ranking gate G6a passes for B — so "the prediction does not use the actions", the
    rationale attached to branch 2, is not what the numbers show. The rule is reported as
    written rather than re-interpreted, and the tension is reported with it.
  - Branch 4 is triggered for **arms C and D**, where G7 passes in full: G1 fails and
    **G2a does not improve materially over v2's 0.835** — the best v3 G2a is D's 0.831, a
    0.5 % relative change, with B 0.876, C 0.940 and A 1.010 clearly worse. Its
    preregistered text reads: the evidence points at the **data**, not the architecture —
    112 px onboard frames of a ~2 cm apple, recorded by a scripted collector, may simply
    not determine the palm–apple offset to 1.5 cm. Its next step is to **measure the
    information ceiling directly**: train a readout on a *single* frame at 224 px and at
    the native render resolution, with no dynamics at all, before spending more on models.
- The protocol also committed, in advance, that **if outcome 4 holds — if after v3 the
  model still cannot beat its own persistence readout under motion — then CEM over this
  cost is probably the wrong control formulation for this task**, and the next task should
  test a learned policy instead: behaviour cloning on the scripted corpus, with the world
  model used as a critic or a residual rather than as the forward model of a sampling
  planner. Outcome 4 holds. That reading was committed before the runs and is recorded
  here unchanged.
- The three readings are complementary, and the ranking result argues for the order among
  them: the models rank candidate actions far better than they localise the apple, so the
  world model looks more promising as a *critic* than as a metric forward model. The
  information-ceiling measurement is the cheapest of the three and settles whether any
  amount of modelling at 112 px can reach 1.5 cm.
- **No arm's result is re-tuned against val and re-reported.** Nothing was changed after
  the gates were read. Any rerun is a new, disclosed protocol version.

## Verification

A fresh-context post-run verifier re-derived, from the raw JSON artifacts alone, every
gate value, threshold and pass/fail decision in the table above and the per-arm pass
counts (all confirmed to six significant figures). It also confirmed:

- **No leakage.** `test_episodes_decoded == 0` in all eight reports; 0 shared roots
  between train, val and test; normalization fitted on exactly the 677 train episodes;
  selection and gates on val only. It noted that the field is a literal backed by the
  `load_split` guard rather than a measured counter, which is recorded above.
- **No overwritten evidence.** Only `outputs/task052-*`, `checkpoints/task052-*` and the
  runner log have mtimes on or after the run date; nothing under `data/` was touched and
  every TASK-050 artifact under `*/task050-wm-v2/` is intact and still readable.
- **No threshold movement.** Every threshold matches across the preregistration, the
  frozen manifest and all four gate reports, and
  `git diff e2f8227 HEAD -- benchmarks/manifests/apple-world-model-v3.json docs/experiments/apple_world_model_v3.md configs/apple_wm_v3*.yaml`
  is **empty** — the protocol, manifest and all five configs are bit-identical between the
  commit the runs were made from and this PR. `e2f8227` was committed 65 seconds before
  the runner's first log line.
- **The selection rule was applied as preregistered.** Recomputing `argmin(score)` over
  the eligible validation entries with step > 0 reproduces `best_step` for all four arms
  (10,000 / 15,000 / 10,000 / 13,000). The gate evaluation used `<arm>.pt` (the selected
  checkpoint), not `.latest.pt`, for every arm — `checkpoint_sha256` equals
  `best_checkpoint_sha256` and differs from `latest_checkpoint_sha256` in all four cases.
  No arm reported `selection_failed`. Arm D's step 0 was additionally *ineligible*
  (`latent_std_mean` 0.098 < 0.1).
- **Provenance is complete** for all four arms: revision, dirty flag, Python source hash,
  dataset/split/action hashes, best and latest checkpoint hashes, implementation hash,
  seed, device, budget, elapsed, completed steps, parameters, peak RSS and the declared
  privileged-label use.
- **One-factor isolation.** From the resolved `model_config` blocks, exactly four keys
  differ anywhere in the design (`cameras`, `patch_size`, `readout_moving_weight`,
  `readout_auxiliary_weight`); all 23 other keys, plus seed, device, budget, data counts
  and every hash, are identical across the four arms. A ↔ B differs in `cameras` alone
  (Δ 16,384 parameters = exactly one 128×128 projection).

Three claims made in the task hand-off were **corrected** against the artifacts before
this document was written, and are stated here so the correction is on the record:

1. The arms use **patch 14**, not patch 16, for A/B/C (112 / 14 = 8 tokens per side).
2. **Arm D is onboard-only**, like B — not a two-camera arm; its config extends arm B's.
   The controlled resolution contrast is therefore **B ↔ D** (3.648 → 3.304 cm), not
   A ↔ D. The claim that spatial resolution is the largest lever on readout precision is
   **not supported**: the camera set is (A ↔ B, 6.62 → 3.65 cm).
3. **A ↔ C is not "the motion weighting"** on its own: arm C drops the motion weighting
   *and* the auxiliary position readouts together, so the two cannot be separated here.

The verifier's own report contained one unit slip — it wrote the B ↔ D G1 difference as
0.34 mm. It is 0.0364806 − 0.0330430 = 0.00344 m = **0.34 cm**, as stated throughout this
document. The hand-off's "budget 5,400 s for arm A" was likewise a conflation: 5,405 s is
arm A's elapsed time, and 10,800 s is the frozen per-arm budget.

Paths recorded in `run.json.config` point at the throwaway worktree the runs were launched
from. The committed configs are the ones cited in this document; the full resolved config
is embedded in each `run.json` under `config_resolved`, and the source tree hash pins the
code.

## Honest labelling and limits

- **Nothing here is a manipulation result.** No closed loop was run in this task. Learned
  Apple→Plate stays at 0 successes. A gate pass is an offline property of a checkpoint on
  recorded validation data.
- **One seed per arm, one run each.** Every difference between arms is a single-run
  difference with no uncertainty estimate. The ablation readings are the best evidence
  this budget buys, not settled findings.
- **Selection and the gates both used val**, which flatters G1. Test was never decoded, so
  an unbiased check is still available.
- **The arms are not unmodified LeWM.** Each is the pinned upstream encoder, predictor,
  projector and SIGReg objective (revision `8edfeb33`) **plus** the shared state fusion,
  the shared camera fusion and the readout loss.
- **Privileged labels** were used for readout-head training targets, for the readout
  loss's per-frame weight (arms A, B, D) and for val scoring cohorts — never as a model
  input, a planner input, a planning cost or a window filter. The change of policy from
  v2 (labels as a loss weight) was disclosed in the preregistration.
- **Latent MSEs are not compared across backends or arms**; the comparison is in physical
  readout units and compute. All four arms are the same backend.
- **The native backend is not a frozen arm** of this protocol. It is exercised only by the
  smoke, so the one-line-swap invariant is demonstrated as a software property, not as a
  second trained comparison.
- **The ranking cohort is 18 groups from 20 val resets**, at a horizon of 16 while the
  planner's horizon is 8. At h = 8 no arm reaches ρ ≥ 0.5.
- The corpus is privileged scripted-collector data with injected perturbations, and every
  limitation of `apple_wide_collection_results_v1.md` still applies.
