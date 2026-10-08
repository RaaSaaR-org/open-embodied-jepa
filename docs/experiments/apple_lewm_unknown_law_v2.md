# Apple→Plate LeWM committed aim under U-sat, a plate law no hand-written arm is given (TASK-082)

**STATUS: DRAFT** (R20.1–R20.16, decided by Claude under owner delegation; nothing is frozen).
This document is the preregistration the design note recommends; its independent review is
recorded on its PR. **Stage 0 is done** (R20.17–R20.22, record
[apple_lewm_unknown_law_v2_stage0.md](apple_lewm_unknown_law_v2_stage0.md)): the code, the debug
smokes on 74800–74899, the scale probes and the power simulation that fixed δ. **No seed of K, D,
S or the corpus has been simulated and no gated model has been trained.** Approving Stage 0
approves neither K0 nor the freeze (§7.1).

**K0 ended CAL-ESCALATE** (R20.23–R20.25, record
[apple_lewm_unknown_law_v2_k0.md](apple_lewm_unknown_law_v2_k0.md)): τ_commit = 0 (0.5 cm reached
55/64 against 56/64) and r_K = 466 > 465. Nothing is frozen, no later stage runs, and TASK-082
closes without the clause. The text below is kept as written.

- **Admitted by:** the TASK-082 design note
  ([apple_lewm_unknown_law_v2_design.md](apple_lewm_unknown_law_v2_design.md), DRAFT,
  R19.24–R19.25, #166), whose §7 recommends this preregistration.
- **Rulings:** R20.1–R20.16 in [DECISIONS.md](../DECISIONS.md), decision 2026-10-08 (l). Each is
  **decided by Claude under owner delegation (2026-09-30)**. They build on R9.8 and R9.9 (the
  claim split and the random-choice test), R15 and R16 (C1-M), R17 (TASK-077, whose recipe is
  retrained here), R18 (TASK-080, whose readout and fresh-root gate are carried) and R19 (TASK-081,
  whose solver, cohort size and rows are carried).
- **Task card:** `.mc/tasks/done/TASK-082-lewm-under-a-plate-law-no-hand-written-arm-is-given.md`.
- **Templates:** TASK-077 ([apple_lewm_c1m_v2.md](apple_lewm_c1m_v2.md)) for the corpus, the model,
  its training and its dynamics gates; TASK-080
  ([apple_lewm_c1m_v2_pred_readout.md](apple_lewm_c1m_v2_pred_readout.md)) for the readout on
  predicted latents and its offline gate; TASK-081
  ([apple_lewm_commit_precision_v2.md](apple_lewm_commit_precision_v2.md)) for the solver, the
  arms, the 128-reset cohort and the row ladder. Where this document says "carried", the section
  named is in force unchanged; only the differences are written out here.

The canonical status sentence (DECISIONS 2026-10-02, R7, as changed by R19.21) is **not changed by
this draft** (R20.15). It is quoted verbatim and in full in DECISIONS.md (R19.21). Its v2 LeWM part
says that TASK-081's W reached 118/128 counted successes on 128 fresh gated resets under C1-M (one
run, one model seed flagged `last_two_triggered`, simulation only; exact 95 % interval
0.861–0.962), the row L-PASS, the primary claim "LeWM-driven closed-loop success", and that
"LeWM needed" is not shown, because the non-learned H-rule, which is given the simulator's plate
law, scored measurably higher (125/128; W − H-rule −7/128, 95 % interval −13/128 to −2/128).

---

## 0. What changes from TASK-081, stated first

TASK-081 closed **L-PASS** under C1-M ([results](apple_lewm_commit_precision_v2_results.md),
R19.20–R19.23). "LeWM needed" (R9.8's secondary claim) cannot be tested under C1-M: H-rule is handed
C1-M's linear law, and H-sysid's linear form contains it (design note §1). R19.23 (1) and R19.24
therefore pursue a condition in which **no hand-written arm is given the plate law**.

TASK-082 keeps C1-M except the plate's law after the commit step, and therefore retrains:

1. **The condition is U-sat** (§2.1; R20.2): C1-M with the plate law after 405 replaced by the
   design note's saturating, swirling static law. **Declared a task change** (R2's rule, DECISIONS
   2026-10-02: the target must still be predicted), not a variant of C1-M. Every result under it is a U-sat result; none reaches back
   over TASK-077, TASK-080 or TASK-081.
2. **The model is retrained** (§4; R20.5): TASK-077's recipe unchanged (8 × 8 pooled frozen DINOv2
   tokens of the onboard 112 px frame, history one, T = 60, batch 16, the selection rule), W and N,
   three model seeds each, on a **new 2 000-root corpus collected under U-sat** (§4.3). TASK-077's
   checkpoints learned C1-M's law and are not used.
3. **The readouts on predicted latents are refitted** (§5; R20.8): R-S, R-N and R-L in TASK-080's
   form, on the new corpus, gated on a fresh split whose aims are built as the closed loop builds
   them.
4. **K0 is re-run under U-sat on 64 fresh resets** (§7.2; R20.7): τ_commit, the read step r (the
   horizon coverage check), the ceiling, the palm speed and a new aim-box clip-binding check.
5. **The comparators change role.** H-rule keeps **C1-M's written law** (κ = −0.5), now a declared
   prior that is wrong under U-sat; H-sysid's linear form is refitted on the new corpus. A **new
   reported-only tier** of learned non-LeWM baselines (a kernel-ridge sysid, H-sysid-krr, and a
   model-free regression onto the ceiling's aims, P-aim) is added (§6.2; R20.4).
6. **The secondary claim becomes testable** (§1; R20.3): "LeWM needed" in R9.8's sense, W
   detectably better than the better of H-rule and H-sysid, is reported with its power simulated
   in Stage 0. It is still never a gate.
7. **δ is re-argued** by a pre-declared rule from Stage 0's power simulation (§9.2; R20.10).
8. **Fresh seeds** in TASK-082's block 72000–74999 (R19.24) and salts 8405–8412 (§8; R20.11).

Carried from TASK-081 unchanged: the solver affine_local for W and its twins (§6.1), the twins'
definitions, the 128-reset gated cohort with G-bar's fraction 0.875 (112/128), the four exact
one-sided McNemar tests at p < 0.01, R17.15's "detectably" with its count +7, the row ladder and
its order, the determinism re-run, the void rule and the guards.

LeWM still makes **one decision**: the place aim committed at step 405. P-3 (behaviour cloning,
not a world model) picks before 405; e9's scripted place primitive executes the committed aim after
405, with no re-aim. The claim, if any, is **"LeWM-driven aim selection at 405"** under U-sat, not
"a LeWM policy".

## 1. The question and the claims (R9.8, R9.9; R20.3)

**Question.** Under U-sat, a declared simulation-only plate law that no hand-written arm is given,
does a LeWM token predictor retrained with TASK-077's recipe on a U-sat corpus, read by a readout
fitted on its own predictions and committed by affine_local, reach R9.8's primary claim on 128
fresh gated resets, and is it detectably better than the best hand-written arm?

**Primary claim, "LeWM-driven closed-loop success"** (R9.8; all of the following, on cohort S):
1. W's predictor makes the decision from the encoded current frame, with no privileged read at run
   time (`task_truth_in_controller` = 0, carried).
2. **The bar:** W(S) ≥ 112/128 (G-bar, §9.3).
3. **Non-inferiority:** W is non-inferior to the better of H-rule and H-sysid on S within δ
   (G-NI). δ is fixed from {8, 12, 16}/128 by §9.2's declared rule on Stage 0's power simulation
   and written into the frozen block at the freeze; it is **an allocation sized by a power
   simulation, not calibrated** on any measured tolerance.
4. **W beats the action-blind twin N, both scene-blind twins (L-shuf, L-mean) and a random choice
   (L-rand)**, each by an exact one-sided McNemar test at p < 0.01.

Items 2 and 3 must both hold. Item 4 shows that the prediction is used.

**Secondary claim, "LeWM needed"** (R9.8): W detectably better than the better of H-rule and H-sysid
on S (exact one-sided McNemar p < 0.01). **Reported only, never a gate.** Unlike TASK-077 to
TASK-081, it is **not declared "not expected"**: under U-sat, in the design note's development check
(one run, 32 resets, arms reading the true plate, not gated), H-rule with C1-M's written law scored
23/32 and the linear H-sysid 28/32 against the privileged ceiling's 32/32 (ceiling − arm +9
[+4, +14] and +4 [+1, +8]). It is **a live chance, not an expectation**: W must be close to the
ceiling, and the comparators will read the plate from the image, not the true plate. Stage 0
simulates its power (§10). A primary pass never supports a "LeWM needed" statement; only the
secondary test does.

**A reported tier, "LeWM against learned non-LeWM baselines"** (R20.4): W against H-sysid-krr and
P-aim (§6.2), non-inferiority within the same δ and the exact one-sided McNemar test in both
directions, **reported only**. **"LeWM needed" against these baselines is not expected, and is
stated so now**: in the development check a kernel ridge on (plate, palm, aim) fitted on 256 roots
reached the ceiling under U-sat (H-sysid-krr 32/32 on the true plate, aim error 0.072 cm in median),
so there is no headroom left for W over a learned low-dimensional forward model (design note §2,
§4.3). If W is detectably inferior to a learned baseline (upper bound of W − arm < −δ), the results
document says so beside the row; it gates and fires nothing.

**What a primary pass would show:** "LeWM-driven aim selection at 405, followed by e9's place,
reaches the bar, is non-inferior to the best hand-written arm within the declared δ and beats its
action-blind, scene-blind and random twins, under a declared simulation-only plate law (U-sat) that
no hand-written arm is given, after the recipe was retrained on a corpus under that law; LeWM ranks
aims inside a box built from a non-LeWM single-frame readout (R-plate) and the proprioceptive palm,
its predicted plate is read by a ridge fitted on its own predictions under the kinematic stand-in's
commands, and the committed aim is affine_local's clipped fixed point". It would change R7's
sentence only through its own reviewed ruling (§14).

**What a secondary pass would show, and what it would not:** that, under U-sat, more than a
hand-written rule with C1-M's written law and a fixed linear system identification is needed (a
flexible learned model, or another model that represents the law's saturation and swirl), and
that LeWM, from pixels, is one model that suffices. It would **not** show that LeWM is needed rather
than any learned model (the reported tier is expected to tie W or beat it), nor anything beyond the
two named comparator forms. Its wording must name them (design note §2, §7 item 4).

**What no row would show:** a LeWM policy (P-3 picks and e9 places); anything about v1's 0/150,
Arena or the real G1; that pretraining matters (no random-init floor is run for W); anything about
C1-M, U-play or other laws; that TASK-081's result transfers to U-sat or back.

## 2. The condition and what is carried

### 2.1 U-sat (R20.2)

C1-M exactly as TASK-081 §2.1 records it, **except the plate's law after 405**:

| quantity | value | source |
|---|---|---|
| task, reset, success | `apple-to-plate-v2`; `wm_critic_v2.wide_reset_values(seed)`; counted success = at rest after a latched grasp and a latched place (T71-R1/R2) | v2, TASK-071 (carried) |
| pick | P-3 (`7988162d…60be8`) with G-repro's post-look estimates | TASK-072, TASK-076 (carried) |
| the move | at the observation of step 300, ρ\* = 4 cm (disc), off-table re-draw and blocked move as R16.3, **drawn with this task's salt 8405** on fresh seeds | R16.3 (carried form) |
| **the law (new)** | from s0 = 405 to s1 = 525, the plate's displacement from its position at 405 is **F(d) = −A tanh(\|d\|/D) R(Θ \|d\|/D) d/\|d\|**, d = palm(t − L) − palm(405), **A = 8 cm, D = 12 cm, Θ = 0.25 rad, L = 2**; held after s1 | design note §3.1 (`plate_law_dev.LAWS["sat"]`, `sat_displacement`) |
| the commitment | one aim committed at c = 405; e9's place primitive executes it with no re-aim | R9.2 (carried) |
| the box and grid | g = p̂ + a·(h − p̂) + b·n, a ∈ [−0.5, 0.5] in steps of 0.05, b ∈ [−3, +3] cm in steps of 1 cm: 147 candidates | M-F2 (carried) |
| the read step | **r = 465, h = 60**, kept if K0's horizon check passes (§7.2); otherwise CAL-ESCALATE | TASK-077 §2.1 (carried, re-checked) |
| τ_commit | **re-measured in K0 on 64 fresh resets** (§7.2) | R20.7 |

The law's parameters were fixed in `plate_law_dev.py` before any seed of the development block was
simulated (design note §3, commit `ae680ca`). Under U-sat the plate comes toward the hand, as under
C1-M, but its travel saturates at A and it turns sideways by an angle that grows with the palm's
travel (about 20° and 7.1 cm at a 17 cm travel). The law is memoryless and smooth; by arithmetic its
Lipschitz constant stays below 1 at the box's travels, so the fixed point should be unique (design
note §3.1). **It is imposed by the simulator and transfers as such to nothing physical** (§13).

**Why it is a task change (R2's rule).** The decision-relevant consequence (where the plate ends for
a committed aim) follows a law that differs from C1-M's in form, not only in a parameter; a model
trained under C1-M learned the other law; every comparator's knowledge of the law changes. The
declared target still has to be predicted: in the development check H-now (the true plate at 405)
scored 2/32 under U-sat.

**Why U-sat and not U-play.** U-play left H-rule and H-sysid at 31/32 on the true plate (design note
§4.2), inside the scorer's noise, so it gives the secondary claim no room. **This choice was made
after the development check read the hand-written arms' counts under both laws** (§13, disclosure).

### 2.2 Code (the Stage-0 PR adds new modules only)

Imported, not edited, with their pins checked (G-frozen, G-hash): TASK-077's modules
(`lewm_c1m_v2.py`, `_runtime`, `_offline`, `_train`), TASK-080's (`lewm_pr_v2.py`, `_runtime`,
`_offline`), TASK-081's (`lewm_cp_v2.py`, `_runtime`) and the solver functions in
`commit_precision_dev.py`; the design note's law and estimators in `plate_law_dev.py` and its worker
hook `plate_law_dev_runtime.LawMotion` and `KrrAim`, **pinned at their sha256 at Stage 0** (they
were written as development code; this protocol makes them frozen inputs, as TASK-081 did for
`commit_precision_dev.py`). New modules (Stage 0, R20.17): `lewm_ul_v2.py` (the frozen block),
`lewm_ul_v2_runtime.py`, `lewm_ul_v2_offline.py`, `lewm_ul_v2_train.py` (TASK-077's training loop
copied verbatim, with this task's sampler salt 8409), `scripts/run_lewm_ul_v2.py`,
`tests/test_lewm_ul_v2.py`, `benchmarks/manifests/apple-lewm-ul-v2.json`. The new runner may not
load other scripts (`tests/test_no_runner_imports.py`).

### 2.3 Artifacts carried, each checked by sha256 before use (G-hash)

| artifact | source | sha256 (as recorded) | why it carries |
|---|---|---|---|
| R-plate (the 405 plate reading that builds every grid) | TASK-077 Stage O | content `08bde901…9eb9` | the plate is static from 300 to 405 and its appearance does not change; the law acts only after 405, so the 405 frame's distribution under U-sat is C1-M's (design note §5.1). Its error on the new corpus is reported (§9.1) |
| TASK-077's training budget U = 95 000 and G1's calibrated bars (rank 0.12, std 0.38) | TASK-077 Stage T plan (§7.7), report `7c714760…7cdfb` | – | §4.4, R20.5 |
| TASK-081's frozen block and its six file pins; TASK-080's and TASK-077's frozen blocks and pins | the three manifests | `77ccc636…705a`, `0fc095dc…be064`, `f28e5e2c…548d` | imported code (§2.2) |

Nothing else of TASK-077 to TASK-081 is read: no checkpoint, corpus, featurisation, readout fit,
sysid fit, mean latent or normalisation moment. TASK-076's evidence root is read for G-repro and P-3,
as before.

## 3. Caveats stated before any number

1. **Every development number is optimistic for the comparators.** The design note's arms read the
   **true plate**; here every non-privileged arm reads R-plate's p̂ (TASK-077 Stage O's cross-fitted
   median error 0.148 cm on C1-M's roots). With p̂, every arm loses some resets (TASK-081: H-sysid
   121/128 against the ceiling's 126/128 under C1-M).
2. **32 resets do not order arms near the ceiling.** In the same development run, under C1-M's own
   law, the ceiling scored 30/32 while H-rule scored 32/32; the scorer's 4 cm radius adds noise to
   every count (TASK-081 results §3.2). The development counts size the design; they are not bars.
3. **The linear H-sysid's development loss under U-sat is thin** (28/32, aim error 87.5th percentile
   0.72 cm, maximum 1.05 cm). Fitted on 1 500 roots instead of 256 it may come closer to the
   ceiling; a linear form cannot represent the saturation or the swirl, so the misspecification
   remains (offline cross-fitted error 0.325 cm in median on 256 roots, design note §4.2).
4. **The learned tier leaves no headroom** (§1). It is reported so that no secondary pass is read as
   "LeWM needed" against learned models.
5. **W has never been trained under U-sat.** TASK-077's recipe passed G1–G4 under C1-M on all three
   seeds; whether it learns U-sat's swirl over 60 recursive steps is unmeasured.

## 4. The model, the corpus and the training (R20.5, R20.6)

### 4.1 The latent and the predictor (TASK-077 §4.1–§4.2, carried unchanged)

The frozen pretrained DINOv2 ViT-S/14's final patch tokens of the onboard 112 px frame (upsampled to
224 px), average-pooled to an 8 × 8 grid (24 576-d), standardised per dimension by train-only
moments (floored at 1e-3); the pinned upstream LeWM predictor adapted to a token sequence
(`frozen_token_model("leworldmodel")`), history one, depth 2, 2 heads × 24, MLP 128, zero dropout,
SIGReg weight 0, recursive multistep loss weight 1.0, learning rate 3e-4, weight decay 1e-4,
gradient clip 1.0; training windows of T = 60 transitions, batch 16 (TASK-077 §4.2's three declared
departures from TASK-066, unchanged). The backend swap stays one key.

### 4.2 Why retrain, and why the recipe unchanged

- **Retrain.** W-66800 and every TASK-077 checkpoint learned C1-M's law from a corpus under C1-M
  (design note §5.1). A model that has never seen U-sat cannot be expected to predict it; any use of
  those checkpoints would test transfer, a different question.
- **Unchanged.** A changed recipe would confound the law with the model. TASK-077's recipe is the
  only one gated at h = 60 (G1–G4 on all three seeds) and the only one with a gated closed-loop pass
  (TASK-081). The corpus size (2 000 roots) is TASK-077's.
- **Three seeds per arm** (the design note's recommendation; TASK-077 §14 item 2): every dynamics
  gate on all seeds, the seed-to-seed spread that R19.23 (3) asks for, and the two non-primary seeds'
  W run in closed loop as reported arms (§6.3). A one-pair plan (about 10 h) would leave the primary
  seed's choice and the gates without replication; the note does not argue for it, so it is not
  taken.

### 4.3 The corpus `apple-ul-v2` (privileged scripted collector; Stage C)

- **2 000 roots, seeds 72800–74799** (§8), collected under U-sat as TASK-077 §4.3: P-3 picks; the
  move at 300 (salt 8405); at 405 e9's aim is committed to (a, b) drawn uniformly over the box
  (salt 8406); kept per root: the onboard 112 px frames of steps 403–467, the executed normalised
  14-D commands of 403–466, the true plate and palm at every kept step, a, b, the robot state at
  405, a plate-hidden render at r and the 405 frame for the full-token features. A root stops at
  step 467.
- **Splits by root, fixed before collection** (one permutation, salt 8407): **train 1 500, val 250,
  gate-P 250**. Train fits everything that is fitted (§4.4, §5, §6); val selects checkpoints and the
  primary seed; **gate-P is read only by Stage G (§9.4), after its `first_outcome_utc`, and never for
  a fit or a selection.**
- **How the aim is built, by split** (TASK-080 §5's confound, handled without a second corpus):
  train and val roots build the box from the **true** plate p (TASK-077's collector, so the training
  distribution is the recipe's); **gate-P roots build it from p̂, R-plate's reading of the onboard
  405 frame** (CPU DINOv2 full tokens, as the closed loop computes it), as TASK-080's gate-P split
  did. TASK-080 measured the confound's size under C1-M (median e_S on gate-P minus contrast-T, its
  record §8.4): +0.035 cm for the primary seed and +0.011 and +0.047 cm for the other two, every
  interval including 0, which bounds only a small confound (TASK-080 §5.3 point 4); so no contrast
  split is collected here; the closed loop remains the
  final check (TASK-080 §5.3 point 5).
- **The ceiling's aim per root (new).** At 405, before the root's own aim is committed, H-final(commit)'s
  look-ahead (`LookaheadAim`, branches under U-sat) runs once **in cloned state** and its aim is
  logged; the root then continues with its own uniform aim. These labels are privileged corpus
  data: P-aim is fitted on the train roots' labels (§6.2), and gate-P's labels are the reference for
  Stage G's offline aim errors (§9.4). Stage 0 tests that a root's executed trajectory is identical
  with and without the labelling pass. **A label whose look-ahead does not converge** is the aim
  H-final(commit) would commit in closed loop (its last iterate at its cap, as K0's and every closed
  loop's ceiling commits), logged with a convergence flag; it is used like any other label, and the
  number of unconverged labels per split is reported. More than 2 % unconverged labels in gate-P is
  **CORPUS-ESCALATE** (the reference for A1 and A2 would then be unsettled on too many roots).
- **Exclusions:** a root that does not reach 467 is excluded and counted. More than 2 % of all roots,
  or more than 2 % of gate-P, excluded is **CORPUS-ESCALATE** (escalate, no clause). The development
  corpus lost 1 of 256 roots under U-sat (a guard refusal before 405).

### 4.4 Training (Stage T; GPU; one `gpu_run.sh --wait` job per model)

- **Arms:** W (true executed commands) and N (every command zero), **model seeds 72360, 72361,
  72362**, on the train split; W and N of a seed draw the same windows in the same order (sampler
  `SeedSequence([8409, seed])`). Windows of 60 transitions starting uniformly at 403–407; batch 16;
  normalisation over every kept frame of the train split; CUDA with strict determinism for training
  only; **every prediction that a gate or an arm reads is computed on the CPU, float32, one torch
  thread** (carried).
- **Selection** (carried): the val criterion is the normalised MSE of the recursive prediction over
  steps 1…60 from frame 405 of every val root, at 20 evenly spaced points; the kept checkpoint is
  `run_tools.select_checkpoint(curve, 0.01)`. `last_two_triggered` is recorded per model and stated
  beside every comparison of its seed; it changes no budget and no row (TASK-077 §4.5).
- **The primary seed** is the W seed with the lowest val criterion at its kept checkpoint, fixed by
  Stage G's report before gate-P is opened; N uses the same seed (TASK-077 §4.4, carried).
- **No calibration jobs (R20.5).** The budget U = **95 000** updates and G1's calibrated bars (rank
  0.12, std 0.38) are **carried** from TASK-077's plan (§7.7 of that protocol), not re-measured.
  Reasons:
  - TASK-077's budget rule (U = 2 × the larger calibration saturation, rounded, capped at 100 000)
    depends on the measured saturation, so on the data as well as on the recipe and corpus size; a
    re-measured U could differ. **But the cap bounds any increase:** U = 95 000 is within 5 % of the
    100 000 cap, so a re-run calibration could raise U by at most 5 000 updates; otherwise it could
    only lower it, which would not help W. The recipe and corpus size are unchanged, and the law changes the
    plate's path after 405 only (median travel from 405 to s1 7.17 cm under U-sat against 8.71 cm
    under C1-M on the development corpus).
  - The two calibration jobs cost about 5.2 h of GPU queue time (TASK-077's logged start and end
    times: 9 301 s and 9 590 s), about a sixth of Stage T, for numbers that would most likely land
    near the carried ones, at a cap that U already almost reaches.
  - **Disclosed risk:** TASK-077's three W curves had their raw minimum at the last point (95 000);
    W may be slightly under-trained here too. G1's bars are labelled **carried** (§9.4), and the
    truncation check that tested them (rank-1, 2, 4 truncations of cal-W's val predictions all
    failed G1; rank-4 by 0.0013) is not repeated. The rank reference (0.254) was measured on
    C1-M's val roots.
- The budget block still passes `run_tools.check_budget` as TASK-077's did (a Stage-0 test). There
  is no budget row.

## 5. The readouts on predicted latents (TASK-080 §4, carried form; R20.8)

Dual ridges of TASK-077 Stage O's form (relative λ grid, 5 inner folds grouped by root, fold salt
8410), from the raw 8 × 8 predicted latent at r to the true plate (x, y) at r, one per model seed s:

| readout | fitted on | read by |
|---|---|---|
| **R-S^s** | W^s's predictions at r from each fit root's encoded 405 frame under the stand-in chunk of that root's own committed aim | W^s, L-shuf and L-mean |
| **R-N^s** | N^s's predictions at r from the same frames under zero commands | N^s |
| **R-L^s** (screen only) | W^s's predictions at r from L-mean's mean 405 latent under the same stand-in chunks | nothing at run time; R3 only |

**The fit set is train + val, 1 750 roots.** gate-P is the gate and is never fitted on. The val
roots are out of sample for W's training (they only selected the checkpoint), as in TASK-080 §4.2.
**R8** (the encoded-frame readout at r, Stage O, train split) is used only for R0, the encoded
ceiling, and the privileged H-read. Stand-in chunks: `place_planner.primitive_chunks`, horizon 60,
from each root's logged 405 state (TASK-080 §4.1, carried).

## 6. Arms (R20.9)

### 6.1 Gating arms (TASK-081 §4, with this task's models and readouts)

| arm | what it is | readout | solver |
|---|---|---|---|
| **W** | the primary seed's W from the encoded 405 frame; the p̂-and-h grid; stand-in chunks; score \|p̃(g) − g\| | **R-S** | **affine_local** (TASK-081 §3, unchanged) |
| **N** (action-blind) | the primary seed's N, zero commands | **R-N** | affine_local |
| **L-shuf** (scene-blind) | W from the logged 405 frame of W's attempt on the next reset in cohort order that reached 405, with this reset's candidate chunks (R17.20) | R-S | affine_local |
| **L-mean** (scene-blind) | W from the mean encoded 405 latent of **this corpus's** train split | R-S | affine_local |
| **L-rand** | a feasible grid candidate drawn uniformly (salt 8412, sub-key 1) | none | none |
| **H-rule** | `RuleCommit` with **C1-M's written law** (κ = −0.5, L = 2) on p̂ and h, its fixed point on the kinematic stand-in, single commit, clipped. **A declared prior that is not the simulator's law** | none | – |
| **H-sysid** | `SysidAim`: TASK-077's linear form plate(r) ≈ c + A·p̂ + B·h + C·g, **refitted on this corpus's train split with R-plate's readings**, inverted with TASK-077's controller form (`lewm_next_c1.choose_aim`), clipped | none | its own |
| **H-final(commit)** | the ceiling: `LookaheadAim` once at 405 in cloned state (branches under U-sat), unclipped; **privileged, not learned** | — | – |

G-NI's comparator C is the better of H-rule and H-sysid on S (the larger count; a tie goes to
H-rule). Choosing the better one after S is conservative for W.

**Why the comparator set is hand-written only, a declared narrowing of R9.8 (R20.4; design note
§8, first question).** R9.8 says "the best non-world-model arm". P-aim is plainly a non-world-model
arm, so keeping it out of G-NI **narrows R9.8's comparator set, and this is declared, not
assumed**. The distinction drawn is the design note's: **a fixed, hand-chosen form** (H-rule's
written law; H-sysid's linear form with seven coefficients, which is also a forward model on
(p̂, h, g) fitted on the train split) **against a flexible learned regressor** (H-sysid-krr, a kernel
ridge on the same inputs; P-aim, a kernel ridge onto the ceiling's aims). Reasons:
- **Comparability.** TASK-077, TASK-080 and TASK-081 read R9.8's comparator as the better of these
  two fixed forms; keeping them makes the primary row comparable across the line, and the secondary
  claim is R9.8's own "LeWM needed" against "the best hand-written arm".
- **The flexible tier is expected to leave no headroom** (§1): gating on it would demand parity with
  a learned low-dimensional model that the development check puts at the ceiling, which R9.8's
  primary claim does not ask for, and the line's comparability would be lost.
- **P-aim is trained on privileged labels** (the ceiling's aims, as P-3 is trained on e9's privileged
  demonstrations), and it is a new arm with no development reading on p̂.
Both learned arms are reported with the same δ and both directions of the McNemar test (§9.5).
**Any restatement of a TASK-082 result, primary or secondary, must say that G-NI was against the
hand-written arms only and must state the learned tier's reading**, including a detectable
inferiority of W to either learned arm (upper bound of W − arm < −δ) when it holds (§14).

### 6.2 The reported tier of learned non-LeWM baselines (new; reported only)

Each reads only p̂ and h at run time (`task_truth_in_controller` = 0; G-privileged applies); each is
fitted on this corpus's train split with R-plate's readings, the same data W's corpus provides.

| arm | what it is |
|---|---|
| **H-sysid-krr** (primary learned baseline) | the design note's RBF kernel ridge of plate(r) − p̂ on standardised (p̂, h, g) (`plate_law_dev.krr_fit`: length and λ chosen by 5-fold CV over the declared grids, folds salt 8410), inverted with the same controller form as H-sysid (`KrrAim`: `lewm_next_c1.choose_aim`, grid, refinement, clip to the box) |
| **P-aim** (model-free) | a kernel ridge of the ceiling's logged aim, as g − p̂, on standardised (p̂, h) over the train roots (the same kernel family and CV), predicted once at 405 and clipped to the box; no forward model and no solver |

No MLP baseline is run (one learned forward model and one model-free regressor are the tier; the
design note's §8 named an MLP as an option, not a requirement).

### 6.3 Other reported arms (no row reads them)

| arm | what it is |
|---|---|
| **W-s′, W-s″** (the W of the two seeds that Stage G does not choose as primary, each named by its seed) | the same controller as W with that seed's W and R-S; reported with their own exact intervals and paired differences from W and from C, so that the seed-to-seed spread is visible |
| **H-rule-fit** | H-rule's form with κ fitted on the train split (least squares of plate(r) − p̂ on g − h), not in the gate (R20.4) |
| **H-read**, **H-now** | privileged, as TASK-081: the look-ahead read by R8 from the frame rendered at r in the clone; the true plate at 405 |

**Infeasible candidates** are dropped for every arm alike; if all are infeasible the arm aims at p̂
(a counted attempt). Every arm's clip-binding fraction is reported (§9.6). A reset refused before
405 is a failure in every arm and stays in the denominator (R17.20).

## 7. Stages and the staged GO flow (R20.1)

Nothing from K, D, S or the corpus is simulated before the reported GO of its stage. Every stage runs
from a clean worktree of the merged revision (`scripts/new_worktree.sh <dir> --run --from <rev>`),
after G-tests. GPU jobs go through `scripts/gpu_run.sh --wait --min-free-gib 8 --board`; resident
services (the GR00T server, other projects' queues) are never stopped or reconfigured.

### 7.1 The sequence

1. **This DRAFT** and its independent review.
2. **Stage 0 (on the draft's merge; code, no cohort seed).** New modules only (§2.2), the DRAFT
   manifest, the frozen-block candidate and its tests: seed ranges and salts against every forbidden
   range (§8); the pins of every imported file, `plate_law_dev.py` and `plate_law_dev_runtime.py`
   included; that the runtime's law equals `sat_displacement` at A, D, Θ, L; that the labelling pass
   leaves a root's executed trajectory identical; that a gate-P aim is built from p̂ and never from
   the true plate; that W, L-shuf and L-mean read R-S, N reads R-N and none reads R8; the affine_local
   runtime equal to TASK-081's; that H-rule's law is C1-M's κ = −0.5 and not U-sat's; no task truth
   in any non-privileged arm (the learned tier included); `check_budget` on the carried budget block;
   `select_checkpoint` and `last_two_triggered` used and no raw argmin; the `"not evaluated"`
   sentinel; every row of §9 in both directions; `install_guards`, `assert_local_import`, no runner
   imports. **Debug smokes on 74800–74899 only** (nothing read): a few corpus roots per split with
   labels; a tiny training run (a few hundred updates) on the real cache layout through
   `gpu_run.sh`; a Stage-G pass on debug roots; one D-style and one S-style attempt of every arm.
   **Scale probes** that set §11's caps (Stage C's per-root cost with the labelling pass, Stage G,
   S's per-attempt cost). **The power simulation of §10** (salt 8412, sub-key 3), which fixes δ by
   §9.2's rule. The frozen-block candidate's sha256 is recorded.
3. **K0 (on a reported GO; CPU only; before the freeze).** §7.2.
4. **The freeze.** STATUS FROZEN with K0's values and δ in the frozen block, the sha pin, the
   manifest's file pins and this document's sha256, merged on an independent reviewer's reported
   APPROVE.
5. **Stage C, the corpus (on a GO; CPU).** §4.3. Rows CORPUS-ESCALATE or CORPUS-SEALED. The sealed
   manifest's sha256 is checked by every later stage.
6. **Stage O, featurisation and fits (on a GO; GPU for featurisation only, then CPU).**
   Featurisation of every kept frame (8 × 8; 4 × 4 at 405 and r, reported; full tokens at 405) with
   G-anchor; then, on train only: the normalisation moments, R8, H-sysid, H-rule-fit's κ,
   H-sysid-krr, P-aim and L-mean's mean 405 latent; and TASK-077's admission gates O1, O3, O4,
   cross-fitted over train + val (5 outer folds, salt 8410), with the learning curve reported. Rows
   (carried): O-ARM-KEYED, O-NO-BAR, O-PASS (§9.1).
7. **Stage T, training (on a GO; GPU; six jobs, each its own `gpu_run.sh --wait` slot).** §4.4.
   Each job ends T-JOB-DONE or V. No plan step and no calibration jobs (R20.5).
8. **Stage G, the offline gates (on a GO; CPU).** The primary seed is fixed from val (§4.4); the
   roll-outs and fits of R-S, R-N and R-L on the 1 750 train + val roots; `first_outcome_utc` is
   written; only then is gate-P opened: G1–G4 on all three seeds, R0–R3 on all three seeds, A1 and
   A2 on the primary seed (§9.4). Rows: §9.4.
9. **Stage D, the development closed loop (on a GO after G-PASS; CPU).** §7.3.
10. **Stage S, gated (on a reported GO after D-PASS; CPU).** Cohort S, every arm of §6 once per reset,
    paired, with TASK-077's determinism re-run of W on S's first four resets (reading gated at
    0.1 cm, commit target at 0.6 cm, success identical; R17.21, R17.27, R17.38). Rows: §9.3.
11. **Results PR.** Every arm is reported, privileged, hand-written and learned-tier arms labelled as
    such, and an independent reviewer checks every restated number.

A later stage's GO approves that stage only. If an invocation ends in anything other than one of its
declared rows, nothing further is launched and the case is ruled under §11 first.

### 7.2 K0 under U-sat (R20.7; 64 fresh resets 72400–72463)

On K's 64 resets, every attempt under U-sat, at ρ\* = 4 cm (disc):
- **τ_commit:** H-final(commit)'s converged aim plus a planted error of 0, 0.5, 1, 1.5, 2 or 3 cm in a
  per-reset uniform direction (salt 8408), every level on the same 64 resets. τ_commit is the
  largest level such that every level up to it reaches **≥ 56/64** (TASK-080's pooled rule at
  n = 64, R18.19). The τ curve (the six counts) maps offline aim errors to predicted counts (§9.4).
  64 resets, not 32, because a new law cannot be pooled with C1-M's K and K′, and TASK-077's 32-reset
  K left two levels one success above the bar.
- **The ceiling:** N_K(0), the level-0 count.
- **The read step and the horizon coverage check:** r_K, the earliest step at which ≥ 56/64 of the
  level-0 attempts have the plate within 0.1 cm of plate(525) (R14.3's form). The recipe trains and
  rolls out exactly 60 steps from 405, so **r = 465 is kept if r_K ≤ 465**; a later or undefined
  r_K would need a different horizon, a recipe change, and stops K0.
- **The history check:** the median palm speed at 405 (TASK-077 §2.1, carried).
- **The aim-box clip-binding check (new):** each level-0 ceiling aim expressed in the box coordinates
  (a, b) of the box built from **p̂** (R-plate's CPU reading of the 405 frame, logged) and h. The
  count of resets whose ceiling aim lies outside the box (a ∉ [−0.5, 0.5] or |b| > 3 cm) is the
  number of resets on which a perfectly predicting boxed arm would be clipped. Reported with the
  median and maximum of |b| and of a. (Development, true plate: the ceiling's aim lay 0.95 cm
  sideways of the plate–palm line in median, 2.1 cm at most; design note §8.)
- **Reported only:** H-rule (on p̂, C1-M's written law) and H-now on the same 64 resets, with their
  paired differences from the ceiling; they are reported beside §10's table at the freeze and gate
  nothing; δ is already fixed by Stage 0 (§9.2) and K0 changes no rule.
- **Stops, all CAL-ESCALATE** (escalate, no clause; nothing is frozen): level 0 < 56/64 (τ_commit
  undefined); N_K(0) < 60/64 (TASK-077's 30/32 fraction); r_K later than 465 or undefined; median
  palm speed at 405 > 0.5 cm per step; **more than 4/64 ceiling aims outside the box** (an
  allocation: beyond it the box itself, not the predictor, would bind the boxed arms on more than
  about 8/128 resets, which is at or above the smallest δ that §9.2 can choose, so the box alone
  could decide G-NI).
- **K0-PASS:** τ_commit, the τ curve, N_K(0), r_K, the clip count and the reported arms enter the
  frozen block. About 512 attempts (384 planted plus 128 reported), about 15 min on 6 workers.
- **τ_commit = 0** (level 0 reaches 56/64 but 0.5 cm does not) is also **CAL-ESCALATE**: O1, R0
  and R1 would be unattainable.
- **If τ_commit falls to 0.5 cm,** every bar tied to it tightens with it (O1, R0, R1, the τ curve);
  that is the honest consequence of a measured tolerance and not an escalation by itself
  (TASK-080 §8 step 2's wording). G-bar's fraction 0.875 does not change. **But because R1 would
  then be likely to fail** (TASK-080's e_S upper bounds were 0.61–0.66 cm under C1-M) **and Stage T
  costs about 31 GPU-hours, a τ_commit below 1.0 cm makes Stage T's GO a ruling point**: the freeze
  PR states the risk, and Stage T's GO needs its own recorded ruling (proceed, or close TASK-082 as
  CAL-ESCALATE without the clause) before any training job is launched. No bar changes either way.

### 7.3 Stage D (16 fresh resets 72500–72515)

W, N, L-shuf, L-mean, H-final(commit), and, reported only, H-rule and H-sysid. **L-DEV-STOP**
(escalate, no clause) if W < 12/16, or H-final(commit) < 14/16, or W − max(N, L-shuf, L-mean) <
+3/16 (carried from TASK-080 §8 step 6 with TASK-077's false-stop probabilities); otherwise
**D-PASS**. Nothing is fitted or tuned on D. D is a development cohort that gates only Stage S's GO.

## 8. Seeds, cohorts and salts (R20.11)

**Block 72000–74999** (R19.24): outside every forbidden range of TASK-081's list and TASK-081's
block 70000–71999 (`plate_law_dev.check_seed_ranges`, tested). The design note used 72000–72031
(check cohort F), 72100–72355 (the 256-root development corpus) and 74900–74999 (debug); **none of
them is used here**. Every range below lies in 72032–72099 or 72356–74899, the ranges R19.24 left
for a protocol; Stage 0's test re-checks this in code, with the design note's three ranges added to
the forbidden ones.

| seeds | use |
|---|---|
| 72360, 72361, 72362 | model seeds of W and N (torch; not resets) |
| 72400–72463 | **K**: K0 (64 resets) |
| 72500–72515 | **D**: the development closed loop (16 resets) |
| 72600–72727 | **S**: the gated cohort (128 resets) |
| 72800–74799 | **corpus** `apple-ul-v2`: 2 000 roots (train 1 500, val 250, gate-P 250) |
| 74800–74899 | **debug**: runner mechanics only; nothing in it is read |
| 72032–72099, 72356–72359, 72363–72399, 72464–72499, 72516–72599, 72728–72799 | reserved; any use needs its own ruling |

| salt | use |
|---|---|
| 8401–8404 | used by the design note's development check (move draw, corpus aims, bootstrap, sysid folds); **not reused** |
| 8405 | the move draw per reset or root: `default_rng(SeedSequence([8405, seed, k]))`, k the re-draw index |
| 8406 | the corpus's uniform (a, b) per root |
| 8407 | the corpus split (train / val / gate-P) |
| 8408 | τ_commit's planted-error direction per reset (K0) |
| 8409 | the training window sampler: `SeedSequence([8409, model seed])` |
| 8410 | every fold assignment, each with a declared sub-key: Stage O's outer folds (1), inner λ CV in every ridge (2), the learned tier's CV (3), the nested learning-curve subsample (4) |
| 8411 | every bootstrap (10 000 resamples, root- or reset-clustered) |
| 8412 | with sub-keys: L-rand's draw per reset (1), G4's wrong-command permutation across gate-P roots (2), Stage 0's power simulations (3) |

**The search** (2026-10-08, at `8c8cd8a`). `git grep -w` over all 126 local and remote refs (paths
`src`, `scripts`, `tests`, `configs`, `benchmarks`, `docs`, `.mc`) for the range ends and seeds
72360–72362, 72400, 72463, 72500, 72515, 72600, 72727, 72800, 74799, 74800, 74899 and the salts
8405–8412: no seed or salt use. The only hits are block declarations: `74899` and `8405`/`8412` in
R19.24's text, the design note, the TASK-082 card and `plate_law_dev.RESERVED = (72356, 74899)`. The
working files (`src`, `scripts`) of the 60 worktrees under `/home/huhn/develop/emai/worktrees/` hold
none of 72360, 72400, 72600, 72800, 74800. No R20 label exists on any ref.

## 9. Gates, bars and rows

Intervals are root- or reset-clustered bootstrap percentile intervals, 10 000 resamples, 95 %, salt
8411, with TASK-080's and TASK-081's estimators (`lewm_pr_v2.paired_interval`, `median_ci` and the
ratio and difference forms). Every bar is labelled **measured**, **definitional**, **carried** or
**allocation**.

### 9.1 K0, Stage C, Stage O

- K0: CAL-ESCALATE or K0-PASS (§7.2).
- Stage C: CORPUS-ESCALATE or CORPUS-SEALED (§4.3).
- Stage O (TASK-077 §8.1, carried, with this task's τ_commit and r): **O1** the upper bound of the
  cross-fitted 8 × 8 readout's median plate error at r ≤ τ_commit (**measured**); **O3** its ratio to
  the constant prior at r < 1.0 (**definitional**); **O4** the lower bound of the same readout's
  median error on the plate-hidden renders at r > τ_commit (**measured**). Rows, first match: V,
  O-ARM-KEYED (O4 fails), O-NO-BAR (O1 or O3 fails), O-PASS; each escalates without the clause.
  Reported: the learning curve, R-plate's error at 405 on the new corpus, H-sysid's,
  H-sysid-krr's and the open form's cross-fitted errors at r (the design note's §4.2 table at full
  size), P-aim's cross-fitted aim error against the ceiling's labels.

### 9.2 δ, fixed at the freeze by a declared rule (R20.10)

δ is chosen by Stage 0's power simulation (§10), **before K0 and before any cohort seed is
simulated**, by this rule: **δ is the smallest of 8/128, 12/128 and 16/128 at which G-NI's power at
W = C is at least 0.80 for every C in {0.80, 0.85, 0.875} under the "half" coupling, and the test's
size at the margin (W = C − δ) is at most 5 % in every coupling; if none qualifies, δ = 16/128**
(TASK-081's allocation, carried as the fallback). **A smaller δ cuts both ways:** it makes G-NI
harder to pass and also makes L-INFERIOR (upper bound of W − C < −δ), and so the clause, easier to
fire; Stage 0 reports L-INFERIOR's false-fire rate at the margin for every candidate δ beside the
rule. The planning set for C spans the development
check's better hand-written arm on the true plate (H-sysid-true 28/32 = 0.875) and two lower values
for the loss to p̂ (§3, caveat 1). The rule only ever tightens the margin relative to TASK-081.
δ is an **allocation**; its label stays so.

**Stage 0's result (R20.20): δ = 16/128.** G-NI's power at W = C under the half coupling was
0.387 / 0.462 / 0.526 at δ = 8, 0.709 / 0.792 / 0.853 at δ = 12 and 0.917 / 0.960 / 0.978 at δ = 16
(C = 0.80 / 0.85 / 0.875); the size at the margin was at most 3.97 % for every δ. Only 16/128
qualifies, so the rule does not tighten TASK-081's margin (record §4). It enters the frozen block at
the freeze.

### 9.3 Stage S, the gated rows (n = 128; TASK-081 §7.2 with this task's δ)

- **G-bar:** W(S) ≥ **112/128** (τ_commit's bar fraction 0.875, TASK-076 G1's form; **carried**).
  **S-VOID-CEILING** if H-final(commit)(S) < 112/128.
- **G-NI:** the lower bound of the paired 95 % interval of W − C > **−δ**, C the better of H-rule and
  H-sysid on S; TASK-080's estimator (the 2.5th percentile of the reset-bootstrap sum of W − C over
  the paired resets).
- **G-N, G-shuf, G-mean, G-rand:** W > arm, exact one-sided McNemar p < 0.01 (**carried**).
- **"Detectably"** as R17.15 and TASK-081 §7.2: a twin test passes at McNemar p < 0.01; W is
  detectably no better than a twin when that test fails and the upper bound of W − twin < **+7**
  resets (a count, **definitional**, independent of n); W is detectably inferior when the upper bound
  of W − C < −δ.

| row (first match) | condition | consequence |
|---|---|---|
| **V** | the void rule (§11) | one repeat after a recorded fix |
| **S-VOID-CEILING** | H-final(commit)(S) < 112/128 | escalate, no clause, no claim |
| **L-NO-GAIN** | for at least one of N, L-shuf, L-mean, L-rand, the McNemar test fails **and** W is detectably no better | **the clause fires** (§12) |
| **L-INFERIOR** | W is detectably inferior beyond δ (upper bound of W − C < −δ) | **the clause fires** (§12) |
| **L-PASS** | G-bar, G-NI and the four McNemar tests all pass | **the primary claim, "LeWM-driven closed-loop success"**; the secondary claim and the learned tier are then reported |
| **L-TWIN-NEAR** | at least one McNemar test fails, and every failed one is a miss within noise | escalate, no clause, no claim |
| **L-NEAR** | G-NI fails, but W is not detectably inferior beyond δ | escalate, no clause, no claim |
| **L-BAR** | otherwise (G-bar fails, with G-NI and the tests passing) | escalate, no clause, no claim |

The ladder is TASK-081's (`lewm_cp_v2.decide_s`'s order) with this task's δ and salt. No reported
arm (§6.2, §6.3) is read by any row. Any repeat after L-TWIN-NEAR, L-NEAR or L-BAR needs fresh seeds
and its own ruling.

### 9.4 Stage G, the offline gates (gate-P, 250 roots; h = 60)

Stage G merges TASK-077's dynamics gates (re-gated, because the models are new) and TASK-080's
fresh-root readout gate (because the readouts are new) into one opening of gate-P.

**Dynamics** (TASK-077 §8.2's definitions; window set E60 = one window per gate-P root from the
encoded 405 frame under that root's executed commands; all three seeds):

| gate | passes when | source |
|---|---|---|
| **G1** no collapse | (i) collapsed fraction ≤ 0.05; (ii) effective-rank ratio ≥ 0.12 and std ratio ≥ 0.38 of W's predicted against the encoded latents at h = 60; (iii) the lower bound of (rank ratio W − rank ratio N) > 0 | (i) **carried** (TASK-065); (ii) **carried** from TASK-077's calibration (§4.4); (iii) **definitional** |
| **G2** copy-last | upper bound of MSE(W) / MSE(copy-last) ≤ 0.8 | **carried** |
| **G3** no-action | upper bound of MSE(W) / MSE(N) < 1.0 | **definitional** |
| **G4** action sensitivity | lower bounds of MSE(W, wrong) / MSE(W, true) and MSE(W, zero) / MSE(W, true) ≥ 1.10 (wrong: another gate-P root's commands, salt 8412 sub-key 2) | **carried** |

**Readouts and offline aims** (TASK-080 §9.4's form; e_X is readout X's plate error at r on its
model's prediction from the root's encoded 405 frame under the root's own stand-in chunk, zero
commands for N):

| gate | passes when | seeds | source |
|---|---|---|---|
| **R0** ceiling | upper bound of R8's median error on the **encoded** frame at r ≤ τ_commit | (no model) | **measured** |
| **R1** precision | upper bound of median e_S ≤ τ_commit | all three | **measured** |
| **R2** against N | upper bound of median e_S / median e_N (paired by root) < 1.0 | all three | **definitional** |
| **R3** commands-alone screen | **lower** bound of median e_L > τ_commit | all three | **measured** (O4's form) |
| **A1** offline aims, the bar | W's predicted count ≥ 112/128 | primary | **measured** (K0's τ curve) against G-bar |
| **A2** offline aims, the twins | W's predicted count − each of N's, L-shuf's, L-mean's and L-rand's ≥ +7 | primary | a necessary floor (R17.15's count), as TASK-080 |

**The offline aims** (TASK-080 §9.4's form, two declared departures): each arm's own controller
code on every gate-P root's logged 405 state, with the stand-in chunks and the grid from the logged
p̂, **with affine_local** for W and the twins (TASK-081's solver); **the aim error is measured against
the root's logged ceiling aim** (§4.3), not against a closed-form fixed point, because U-sat has
none (TASK-080 used C1-M's g\* = (p − κh)/(1 − κ)). K0's planted errors are around the same ceiling
aim, so the τ curve maps exactly this error. The predicted count of 128 maps each root's aim error
through K0's τ curve (TASK-077's piecewise-linear mapping) and scales the mean to 128. **A predicted
count is a prediction, not a closed-loop count**, and is never reported as one. Reported beside them:
the offline aims and predicted counts of H-rule, H-sysid, H-sysid-krr, P-aim, H-rule-fit and the two
non-primary seeds' W (the offline readings of G-NI, of the secondary claim and of the learned tier).

**Rows, first match:**

| row | condition | consequence |
|---|---|---|
| **V** | the void rule (§11) | one repeat after a recorded fix |
| **H-GATE-FAIL** | any of G1–G4 fails on any seed | escalate, no clause (TASK-077's row; its declared remedies need their own preregistration). It precedes R-VOID-CEILING (TASK-080 had no dynamics gate, because its models were already gated): the models are new here, and a readout gate on a model that fails its dynamics gates is not informative |
| **R-VOID-CEILING** | R0 fails | escalate, no clause |
| **R-COMMAND-KEYED** | R3 fails on any seed | escalate, no clause |
| **R-NO-BAR** | R1 or R2 fails on any seed | escalate, no clause |
| **A-NO-BAR** | A1 fails | escalate, no clause |
| **A-TWIN** | A2 fails for any of the four | escalate, no clause |
| **G-PASS** | otherwise | Stage D may get its GO |

**Reported only in Stage G:** each readout's cross-fit over the 1 750 fit roots (5 outer folds, salt
8410 sub-key 1); the learning curve on gate-P; every gate at h = 16 and 30; copy-last's and the
constant prior's readings; the echo slope (TASK-080 §5.3 point 4: the per-root 2 × 2 slope of W's
predicted plate against the aim over the feasible grid, here beside U-sat's local Jacobian at the
ceiling aim, computed from the law); the clip-binding fraction of every arm's offline aim; every
statistic at 1.5 τ_commit beside τ_commit.

### 9.5 Reported in every Stage S row (not gates)

1. Every arm's count with its exact binomial 95 % interval; every paired difference from W with its
   discordant counts, interval and exact one-sided McNemar p; H-read's and H-now's counts.
2. **The secondary claim:** W − C with its discordant counts, interval and the exact one-sided
   McNemar p of W > C; the same against H-rule and H-sysid separately.
3. **The learned tier:** for H-sysid-krr and P-aim, W − arm with its interval, whether the lower
   bound clears −δ (non-inferiority reading), and the exact one-sided McNemar p in both directions.
   "LeWM needed" against them is not expected (§1).
4. **The seed spread:** the two non-primary seeds' W counts (W-s′, W-s″), their differences
   from W and from C, and each seed's `last_two_triggered` flag.
5. Aim error against H-final(commit)'s aim on the same reset for every non-privileged arm (median
   with interval, 87.5th percentile, maximum), the landing miss, each arm's τ-curve prediction (a
   prediction, not a count), and the Stage G predicted counts against the S counts.
6. Every arm's final-distance distribution (TASK-081 §7.3 item 4), including the attempts that ended
   within 0.5 cm of the scorer's 4 cm radius.
7. affine_local's diagnostics for W and the twins (fallbacks, clip-binding, points in the fit, J's
   eigenvalues, the logged residual).

### 9.6 Clip-binding

Reported in K0 (the ceiling, §7.2), in Stage G (every arm's offline aim) and in D and S (every arm's
committed aim): the fraction of attempts whose chosen or solved aim was clipped to the box. For W and
H-sysid it answers the design note's last open question (§8, "swirl and the box").

## 10. Power (R20.12; recomputed in Stage 0)

**Planning rates.**
- **C (the better hand-written arm):** {0.80, 0.85, 0.875}. 0.875 is the development check's
  H-sysid-true (28/32, the true plate, one run, 32 resets); H-rule-true was 23/32 (0.72). With p̂
  both should be lower; K0's H-rule count on p̂ (§7.2) is reported beside the table at the freeze and
  changes no rule.
- **W:** {0.875, 0.906, 0.922, 0.938, 0.953}. TASK-081's W reached 118/128 (0.922) under C1-M (one
  run, one flagged seed); no W has run under U-sat.
- **The ceiling:** 0.97–1.0 (development 32/32 under U-sat; TASK-081 126/128 under C1-M).
- **The twins:** TASK-081's S rates (N 73/128, L-shuf 39/128, L-mean 59/128, L-rand 20/128) and a
  stronger N at 0.75.

**What Stage 0 computes** (salt 8412 sub-key 3; 20 000 trials per cell, 10 000 resamples per
(k+, k−) cell; overlap / half / independent couplings, as TASK-081's Stage 0):
1. **G-bar:** P(X ≥ 112 of 128) at each W (exact binomial; TASK-081 §8: 0.57 at 0.875, 0.91 at 0.906,
   0.98 at 0.922, 0.9975 at 0.938).
2. **G-NI** for δ ∈ {8, 12, 16}/128 over the W × C grid with the better-of-two comparator (H-rule at
   C − 0.10, H-sysid at C), the size at the margin (W = C − δ), and L-INFERIOR's false-fire rate at the
   margin. §9.2's rule then fixes δ.
3. **The secondary claim:** the power of the exact one-sided McNemar test of W > C at p < 0.01 over
   the same grid. As a reading of the arithmetic only (not a simulation): at W = 0.938 against
   C = 0.85 on 128 resets the expected net difference is about 11 resets in W's favour; if every C
   success is also a W success (overlap), about 11 discordant pairs all go to W and the test passes
   (0.5¹¹ ≈ 0.0005); with independent outcomes about 18 against 7 discordant pairs are expected, and
   the test sits near its 0.01 threshold (one-sided p about 0.02). At W = C its pass rate is the
   test's size.
4. **The twin tests** at TASK-081's twin rates and at N = 0.75.
5. **L-PASS** at the planning rates (the product of the parts under the overlap coupling).

**Stage 0's numbers** ([record](apple_lewm_unknown_law_v2_stage0.md) §4, R20.20): δ = 16/128; G-bar
0.566 / 0.908 / 0.978 / 0.9975 / 0.9999 at W = 0.875 … 0.953; G-NI at δ = 16 ≥ 0.969 for every
W ≥ 0.906; the secondary claim's power at C = 0.875 is at most 0.38 with independent outcomes (0.88
with overlapping ones at W = 0.953), at C = 0.85 0.41–0.94 at W = 0.938; every twin test ≥ 0.999 at
TASK-081's twin rates; L-PASS's product equals G-bar's power at every planning rate.

**The binding risks** are expected to be G-bar and the training (H-GATE-FAIL, R-NO-BAR), not G-NI: the
comparators are weaker under U-sat than under C1-M. The secondary claim is expected to have low to
moderate power unless W sits within about 0.03 of the ceiling. Stage 0's numbers replace these
readings in the freeze PR.

## 11. Void rule, guards, caps and memory (carried from TASK-077 §10 and TASK-081 §10)

- **Void and repeat:** TASK-077 §10.1 (one repeat per stage after a committed, pushed and recorded
  fix, on the same seeds in a new output directory; a second V of the same stage ends TASK-082 as
  INCONCLUSIVE; Vs in different stages do not add up). **Stage T is voided and repeated per job**
  (R17.16): a second V of the same job ends TASK-082 as INCONCLUSIVE.
- **Guards:** TASK-077 §10.2 and TASK-081 §10 (G-tests in-run or `pre_run_tests.json` for GPU jobs,
  G-sentinel, G-hash, G-frozen, G-repro, G-anchor, G-threads with the pinned environment
  `MKL_DYNAMIC=FALSE`, `OMP_NUM_THREADS=6`, `MKL_NUM_THREADS=6`, `OPENBLAS_NUM_THREADS=16`, G-quiet,
  G-privileged, G-finite, G-memory at 12 GiB process-tree PSS (18 GiB for a Stage T job), G-disk,
  G-GPU, G-solver). G-hash and G-frozen are extended to TASK-077's, TASK-080's and TASK-081's frozen
  blocks and pins, `commit_precision_dev.py`, `plate_law_dev.py` and `plate_law_dev_runtime.py`.
  - **G-law (new):** every attempt and root logs the law it ran under; the runner refuses a record
    whose law is not U-sat at the frozen parameters, and refuses H-rule with any law other than
    C1-M's written κ = −0.5.
  - **G-split:** every fit is on train (Stage O) or train + val (Stage G's readouts) only; val only
    selects; gate-P is opened only after Stage G's `first_outcome_utc` and never fitted on; the
    learned tier is fitted on train only.
  - **G-fresh:** the runner refuses any closed-loop, K0 or corpus seed outside its range, any S run
    without a non-debug D-PASS report of this task under the same frozen block, and any D run
    without a G-PASS report.
- **Caps** (provisional; set at Stage 0 at ≥ 1.5 × the scaled worst case and written here before the
  freeze): K0 7 200 s; Stage C 14 400 s; Stage O featurisation 3 600 s and fits 7 200 s; each Stage T
  job 46 800 s (TASK-077's, carried: its six jobs took 17 180–21 301 s each); Stage G 14 400 s;
  Stage D 7 200 s; Stage S 21 600 s; per closed-loop attempt 300 s.
  **Confirmed at Stage 0 (R20.21)**, each ≥ 1.5 × its scaled worst case: K0 ≈ 910 s (7.9 ×); C
  ≈ 3 100 s with the labelling look-ahead (4.6 ×); O ≈ 500 s and 330 s; T carried; G ≈ 6 500 s from
  the Stage G probe (2.2 ×); D ≈ 600 s; S ≈ 4 600 s (4.7 ×); the slowest attempt 15.9 s. Each stage's
  cap counts from the end of G-quiet's bounded wait (R20.18).

## 12. The abandonment clause and its scope (R20.13)

**It fires on L-NO-GAIN or L-INFERIOR only**, as R17.15 defines them. It does not fire on
CAL-ESCALATE, CORPUS-ESCALATE, O-ARM-KEYED, O-NO-BAR, H-GATE-FAIL, R-VOID-CEILING, R-COMMAND-KEYED,
R-NO-BAR, A-NO-BAR, A-TWIN, L-DEV-STOP, S-VOID-CEILING, L-TWIN-NEAR, L-NEAR, L-BAR, V or
INCONCLUSIVE. Neither the secondary claim nor the learned tier can fire it.

**What closes (a new scope; U-sat is a task change).**

> "LeWM aim selection with a single aim committed at 405 under the declared U-sat plate law
> (A = 8 cm, D = 12 cm, Θ = 0.25 rad, L = 2, s0 = 405, s1 = 525) on v2, with the declared post-pick
> plate move of radius ρ\* = 4 cm (disc), from onboard 112 px frozen DINOv2 pooled tokens, with
> TASK-066-family predictors trained on a corpus under that law."

with TASK-077's declared narrowing (a 4 × 4 pooled latent on a corpus larger than 1 024 roots stays
open; it is still untested). **If the clause fires, the results document states the restored scope
plainly**: one 8 × 8 recipe, three seeds, read by a dual ridge fitted on its own stand-in predictions
and committed by affine_local, closes under U-sat every other pooled grid (except that 4 × 4 case),
every readout of the predicted latent and every post-grid solver for the committed aim from the
grid's predictions, not only those that were run.

**What does not close:** C1-M and every result under it (TASK-081's L-PASS stands); U-play, U-speed,
U-cue, U-hist, a physical plate mechanism and every other law; other law parameters; necessity
studies against learned baselines (design note §3.4); the full unpooled token grid; history longer
than one, `action_chunk` and `predictor_step_embedding`; a fine-tuned or another encoder; other views
or resolutions; other commit steps, move distributions or radii; TASK-076's results; the LeWM
backend; v2; the product goal.

## 13. Risks and disclosures, stated now

- **The condition and the cohort size were chosen after TASK-081, and the condition after reading
  development counts (R20.14).** U-sat was proposed after TASK-081's L-PASS and chosen over U-play
  after the design note's development check had read the hand-written arms' counts under both laws
  (seeds 72000–72355, true plate, one run). The choice favours room for the secondary claim against
  the hand-written arms; it is a researcher choice of condition, made in the open, not a selection of
  gated data, and every gated count here comes from fresh seeds. The law's parameters were fixed
  before that check. The 128-reset cohort is TASK-081's, itself chosen after TASK-080's L-NEAR.
- **H-rule's written law is a choice.** C1-M's κ = −0.5 is the natural prior an engineer would write
  after C1-M, and it is declared as such; a different written law could score differently.
  H-rule-fit (κ fitted) is reported to show how much of H-rule's loss a fitted gain recovers.
- **The learned tier is expected to tie or beat W.** A result in which W passes the primary claim and
  the secondary claim but is detectably inferior to H-sysid-krr is possible and must be reported as
  such.
- **W has to learn the swirl over 60 recursive steps** from 1 500 train roots. H-GATE-FAIL,
  R-NO-BAR or A-NO-BAR are live outcomes. The budget and G1's bars are carried, not re-measured
  (§4.4).
- **τ_commit may fall** under U-sat (the swirl changes where the apple lands for a given aim error);
  at 0.5 cm R1 would likely fail (TASK-080's e_S upper bounds were 0.61–0.66 cm under C1-M).
- **The box may bind** near its sideways edge for the ceiling and for W (§7.2's stop; §9.6).
- **The scorer's margin is small:** counts near the ceiling depend on more than the aim (TASK-081
  §7.3 item 4); the ceiling itself scored 30/32 under C1-M's law in the development check.
- **The primary seed may be flagged** (`last_two_triggered`), as W-66800 was.
- **Cost.** Stage T is about 31 h of GPU queue time; a V in Stage T is expensive, and the void rule
  allows one repeat per job.
- **The scorer and the condition are imposed simulator laws**; nothing here transfers as such to
  Arena or the real G1.

## 14. R7 (R20.15)

- **Unchanged by this draft.** R7 states TASK-081's gated success under C1-M with its qualifiers;
  nothing in TASK-082 changes it until a reviewed ruling after a result.
- **An L-PASS under U-sat** would need its own reviewed ruling before R7 states it, with every
  qualifier of R19.21 and the condition's name (a declared simulation-only law that no hand-written
  arm is given; a task change; the recipe retrained). Its wording must also say that G-NI was
  against the hand-written arms only (H-rule with C1-M's written law, the linear H-sysid), a declared
  narrowing of R9.8, and state the learned tier's reading, including any detectable inferiority of W
  to H-sysid-krr or P-aim.
- **A secondary pass** ("LeWM needed" against H-rule with C1-M's written law and the linear H-sysid)
  would also need its own ruling, and its wording must name both comparator forms and the learned
  tier's reading; it would not make "LeWM needed" true in general.
- TASK-079's precondition (R19.22) is not affected by this task.

## 15. Compute estimate (R20.16)

| stage | estimate | basis |
|---|---|---|
| Stage 0 | debug smokes minutes each; a tiny GPU training run of a few minutes; power simulation minutes | TASK-077's and TASK-081's Stage 0 |
| K0 | about 15 min CPU | about 512 attempts on 6 workers; TASK-077's K0 took 579 s for 352 |
| C | about 1–1.5 h CPU | TASK-077's Stage C took 1 234 s for 2 000 roots on 6 workers; the labelling look-ahead adds about 8 s per root (H-final(commit)'s median attempt in TASK-081's D, 8.23 s, as an upper scale) |
| O | about 5 min GPU (featurisation; TASK-077's probe 212.7 s for 1 995 roots), 20–40 min CPU (fits, the learned tier, O1–O4) | TASK-077 §12 |
| T | **about 31.4 h on the GPU queue** | TASK-077's six model jobs at U = 95 000 took 113 131 s in all (17 180–21 301 s each); no calibration jobs (saves about 5.2 h) |
| G | about 1.5–2.5 h CPU | TASK-080's Stage R (3 593 s: fits on 1 995 roots × 3 seeds, offline aims) and TASK-077's Stage G (2 369 s: G1–G4 on 250 roots × 6 models) |
| D | about 15 min CPU | TASK-081's D: 512 s for seven arms |
| S | about 1.5 h CPU | TASK-081's S: 3 318 s for eleven arms on 128 resets; W-frozen is dropped, three cheap arms (H-sysid-krr, P-aim, H-rule-fit) and two more W seeds are added, about 30 s more per reset over 4 workers |
| **total** | **about 31.5 h GPU and 5–7 h CPU**, about three to four days of wall time with the reviews and GOs | |

Disk: frames and features as TASK-077 (about 1.2 GB of roots, 12.8 GB of 8 × 8 features, 0.8 GB of
full tokens at 405); the main checkout's free space is checked by G-disk at the start of Stages C
and O.

## 16. The design note's open questions, resolved (R20.3–R20.12)

1. **The comparator set for G-NI:** hand-written only (H-rule, H-sysid); the learned tier is
   reported (§6.1, §6.2).
2. **The written law for H-rule:** C1-M's κ = −0.5, declared as a prior; H-rule-fit reported only
   (§6.3).
3. **The learned baseline's class and data:** H-sysid-krr (primary learned baseline) and P-aim, both
   kernel ridges, on the train split with R-plate's readings; no MLP (§6.2).
4. **τ_commit, r and the margin:** re-measured in K0 on 64 resets (§7.2); δ by §9.2's rule from
   Stage 0's power simulation.
5. **The horizon:** K0's r_K must be ≤ 465, or K0 stops (§7.2).
6. **Swirl and the box:** K0's clip-binding stop and the reported clip-binding fractions (§7.2,
   §9.6).
