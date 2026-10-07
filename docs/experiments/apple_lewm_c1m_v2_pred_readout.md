# Apple→Plate LeWM committed aim under C1-M, read by a readout fitted on W's predicted latents (TASK-080)

**STATUS: DRAFT** (R18.1). Not frozen. Nothing in this document has been run: no seed of its
block (65000–65999) has been simulated, no readout has been fitted, and no corpus root has been
collected. The freeze follows TASK-077's sequence: this draft's independent review; Stage 0
(code, frozen block, manifest, tests, debug smokes, scale probes and one development dry run);
K0′ on a reported GO; the freeze merged on an independent reviewer's reported APPROVE; then each
later stage on its own reported GO. A later K0′ or freeze needs its own GO or APPROVE; approving
this draft approves neither.

- **Admitted by:** the TASK-077 decomposition record's row **D-READOUT**
  ([apple_lewm_c1m_v2_decomposition.md](apple_lewm_c1m_v2_decomposition.md) §2, R17.53–R17.54,
  #152). That row names "a new preregistration whose controller reads the plate with a readout
  fitted on W's predicted latents under the stand-in at r, on fresh gated roots". This document
  is that preregistration, in DRAFT.
- **Rulings:** R18.1–R18.14 in [DECISIONS.md](../DECISIONS.md), decision 2026-10-07. Each is
  **decided by Claude under owner delegation (2026-09-30)**. They build on R9.8 and R9.9 (the
  claim split and the random-choice test; [design note](apple_lewm_next_v2_design.md) §3), R15
  and R16 (C1-M), and R17 (TASK-077, whose protocol is the template here).
- **Task card:** `.mc/tasks/todo/TASK-080-lewm-committed-aim-under-c1m-read-by-a-readout-fitted-on-predicted-latents.md`.
- **Template:** TASK-077's frozen protocol, [apple_lewm_c1m_v2.md](apple_lewm_c1m_v2.md). Where
  this document says "carried", the TASK-077 section named is in force unchanged; only the
  differences are written out here.
- **TASK-078 and TASK-079 are reserved** in [PLAN.md](../PLAN.md) (Arena perception transfer and
  the LeWM place in Arena); this task is TASK-080, the next free number (checked on 2026-10-07:
  no card, branch or document uses TASK-080).

The canonical status sentence (DECISIONS 2026-10-02, R7), verbatim. **This draft does not change
it** (R18.12):

> Learned Apple→Plate on the frozen v1 MVP benchmark (TASK-020) is 0/150 per backend
> (`native_jepa` and LeWM). On `apple-to-plate-v2`, the behaviour-cloning/DAgger policy P-3 (an
> MLP on a frozen DINOv2 readout, trained on demonstrations from the privileged scripted expert
> e9; not a world model) scored 40/40 counted successes on the held-out cohort C against 39/40
> for its random-init encoder control R-3 (one run, one training seed per arm, 40 resets, one
> camera at 112 px onboard, a narrow reset distribution), so TASK-072 M2 is M2-FAIL on G3
> (encoder pretraining contributed nothing measurable), and cohort C is no longer held out. No
> LeWM-driven controller has run in closed loop on v2 yet; LeWM's only closed-loop Apple→Plate
> runs are on v1, with 0 successes. Scripted-expert, privileged-ceiling, oracle and GR00T
> successes are not project-learned results.

---

## 0. What changes from TASK-077, stated first

TASK-077 ended **G-NO-BAR** at Stage G ([results](apple_lewm_c1m_v2_results.md), R17.50–R17.52):
G1–G4 passed on all three seeds, and G5 (a) failed on all three, because R8, a plate readout fitted
on *encoded* frames, read W's *predicted* latent at r at 1.581, 1.617 and 1.873 cm (upper bounds
1.697, 1.794 and 2.076 cm) against τ_commit = 1.0 cm, and at 3.09–3.63 cm under the stand-in
commands W ranks in closed loop. The decomposition record then found, on train and val only, that
a dual ridge refitted on W's own stand-in predictions reads the plate at r at **0.578, 0.564 and
0.556 cm** in val median (upper bounds 0.636, 0.625 and 0.612 cm; 0.34–0.38 of N's refit error),
against an encoded ceiling of 0.410 cm. That reading is optimistic (§3), and it is the reason for
this task.

TASK-080 keeps **everything** of TASK-077's design except three things:

1. **The readout.** W's controller reads the predicted plate with **R-S**, a dual ridge fitted on
   W's own predicted latents at r under the kinematic stand-in's chunks, not with R8 (§4). The
   action-blind twin N gets **R-N**, fitted the same way on N's predictions.
2. **Fresh roots.** A new corpus of 500 roots, never seen by any fit, selection or gate, carries a
   new offline gate split (§5, §8). TASK-077's gate split has been read and is used here only as
   readout-fit data.
3. **No training.** The six Stage T checkpoints of TASK-077 are reused as they are (§3, R18.2).

LeWM still makes **one decision**: the place aim committed at step 405. W encodes the onboard
112 px frame at 405 with the frozen DINOv2, rolls each candidate aim's 60 stand-in commands to
r = 465, reads the predicted plate with R-S, and picks the aim whose predicted plate is nearest to
the aim. P-3 (behaviour cloning, not a world model) picks before 405; e9's scripted place
primitive executes the committed aim after 405, with no re-aim. The claim, if any, is **"LeWM-driven
aim selection at 405"** under C1-M, not "a LeWM policy".

## 1. The question and the two claims (R9.8, R9.9; carried from TASK-077 §1)

**Question.** Under C1-M (v2's reset, P-3's pick, a post-pick plate move at step 300 uniform over
a 4 cm disc, cell A's reactive-plate rule from 405, one aim committed at 405), does W's choice,
read by a readout fitted on W's predicted latents under the stand-in, reach the primary claim on a
fresh gated cohort?

**Primary claim, "LeWM-driven closed-loop success"** (R9.8; all of the following, on cohort S):
1. W's predictor makes the decision from the encoded current frame, with no privileged read at
   run time (`task_truth_in_controller` = 0, as TASK-077).
2. **The bar:** W(S) ≥ 56/64 (G-bar, §9.5).
3. **Non-inferiority:** W is non-inferior to the better of H-rule and H-sysid on S within
   δ = 8/64 (G-NI). δ is **the design note's proposal, an allocation, not calibrated** (R15.6's
   label, R17.8).
4. **W beats the action-blind twin N, both scene-blind twins (L-shuf, L-mean) and a random
   choice (L-rand)**, each by an exact one-sided McNemar test at p < 0.01.

Items 2 and 3 must both hold (R9.8). Item 4 shows that the prediction is used.

**Secondary claim, "LeWM needed"**: W detectably better than the better of H-rule and H-sysid
(exact one-sided McNemar p < 0.01). **Reported only, never a gate, and not expected** (both scored
30/32 in C1-M's M-F6, development, and both capture the rule). A primary pass never supports a
"LeWM needed" statement.

**What a pass would show:** "LeWM-driven aim selection at 405, followed by e9's place, reaches the
calibrated bar, is non-inferior to the best hand-written arm within the allocated δ and beats its
action-blind, scene-blind and random twins, under a declared reactive-plate rule with a post-pick
plate move on v2; LeWM ranks aims inside a box built from a non-LeWM single-frame readout (R-plate)
and the proprioceptive palm, its refinement is clipped to that box, and its predicted plate is read
by a ridge fitted on its own predictions under the kinematic stand-in's commands". That would be
the first LeWM-driven closed-loop success on v2. It would change R7's sentence only through its
own reviewed ruling.

**What it would not show:** that LeWM is needed; a LeWM policy; anything about v1's 0/150, Arena
or the real G1; that pretraining matters (no random-init floor is run for W); that a readout of
the encoded frame reads W's predictions (TASK-077 showed it does not); anything about other
conditions, grids or horizons.

## 2. What is carried, with its source

### 2.1 The condition and the code (TASK-077 §2.1, §2.2)

The condition is C1-M exactly as TASK-077 §2.1 records it: v2's `wide_reset_values`; P-3
(`7988162d…60be8`) with G-repro's post-look estimates; the move at the observation of step 300,
ρ\* = 4 cm (disc), off-table re-draw and blocked move as R16.3; cell A's `CellMotion` (κ = −0.5,
L = 2, s0 = 405, s1 = 525); one aim committed at c = 405 and executed by e9's place primitive with
no re-aim; the 147-candidate box g = p̂ + a·(h − p̂) + b·n, a ∈ [−0.5, 0.5] in steps of 0.05,
b ∈ [−3, +3] cm in steps of 1 cm; **r = 465, h = 60**. The only change to the condition is the move
draw's salt (§7), so that the cohorts are fresh.

Code: TASK-077's modules (`lewm_c1m_v2.py`, `_runtime`, `_offline`, `_train`) and everything they
reuse are **imported, not edited**; TASK-077's frozen block and its pin
(`f28e5e2c…548d`) must still match (G-frozen). The Stage-0 PR adds new modules only
(`lewm_pr_v2.py` with this task's frozen block, `lewm_pr_v2_runtime.py`, `lewm_pr_v2_offline.py`,
`scripts/run_lewm_pr_v2.py`; names provisional).

### 2.2 Artifacts reused, each checked by sha256 before use (G-hash)

| artifact | source | sha256 (as recorded) |
|---|---|---|
| corpus `apple-c1m-v2` (1 995 roots: 1 495 train, 250 val, 250 TASK-077 gate) | TASK-077 Stage C | sealed manifest `ad8974b2…43fb` |
| its featurisation (8 × 8 at every kept frame; full tokens at 405) | TASK-077 Stage O | featurise report `f187c7c8…e9ee`; every file against its `files_sha256` (R17.26) |
| normalisation moments | TASK-077 Stage O | content `5415eea4…de6d` |
| R8 (encoded-frame readout at r; **used only for the encoded ceiling, §9.4**) | TASK-077 Stage O | content `62a5ea8a…3b4a` |
| R-plate (the 405 plate reading that builds every grid) | TASK-077 Stage O | content `08bde901…9eb9` |
| L-mean's mean 405 latent | TASK-077 Stage O | content `85da6336…81bd` |
| H-sysid | TASK-077 Stage O | `sysid.json` file `671841cd…121fc` |
| W and N, seeds 66800, 66801, 66802 | TASK-077 Stage T (§7.8) | W `891d2664…76b8`, `27aeadab…c193`, `ba2614ec…97b4`; N `51b51035…d2c2`, `15f5dd41…2d37`, `4f92a8fe…bfef` |
| K0's τ curve and ceiling | TASK-077 K0 (§7.1), report `9be44fd9…f235` | 32, 29, 29, 18, 11, 2 of 32 at 0, 0.5, 1, 1.5, 2, 3 cm; N_K(0) = 32/32; r_K = 460 |

The artifacts stay where TASK-077 left them (the `task077-stageo`, `task077-staget` and
`task077-staget2` worktrees, with evidence copies under `~/develop/emai/evidence/`). They are
read only.

## 3. Reuse the six checkpoints or retrain them (R18.2)

**Decision: reuse.** The six kept Stage T models of TASK-077 are used unchanged; nothing is
trained in TASK-080.

**For reuse.**
- **The predictor is not what failed.** On TASK-077's gate split, then unread, G1–G4 passed on all
  three seeds: no collapse, action sensitivity (wrong/true lower bounds 3.49–3.62, zero/true
  5.93–15.64), and W below copy-last (upper bounds 0.148–0.151) and below N (0.377–0.404). G5 (b),
  W's predicted plate against N's under R8, passed too. G5 (a) failed on a readout fitted on a
  different distribution, and the decomposition record measured that a readout fitted on W's own
  predictions removes most of that gap (§0).
- **Retraining the same recipe would test nothing new.** The recipe, corpus and budget rule are
  frozen in TASK-077; new seeds of the same recipe would give models of the same kind, and a
  changed recipe would be a different question that needs its own reason. None is indicated by
  the measurements.
- **Cost.** The six model jobs took 113 131 s of wall time on the GPU queue (about 31.4 h, protocol
  §7.8; the sum of the jobs' `total_seconds`, not pure GPU time) plus the two
  calibration jobs. Reuse costs nothing on the GPU except the fresh corpus's featurisation.

**Against reuse, and how each point is handled.**
- **The checkpoints were selected on val** (TASK-077 §4.4: the earliest point within 1 % of the
  minimum of the val latent MSE), and G1's bars were calibrated on val. Selection used a latent MSE
  of the executed-command roll-out, not a plate reading and not the stand-in. Every gate of
  TASK-080 reads **fresh roots** that no fit, selection or calibration has seen (§5), so the val
  selection biases only the val numbers already published, not any number this task gates on.
- **TASK-077's gate split has been read** (results §3, caveat 5). It is not a gate here. It is
  used only as readout-fit data (§4.2), which is not a selection: a fit on it carries nothing about
  the fresh roots.
- **W-66800's `last_two_triggered` flag is true**, and all three W curves have their raw minimum at
  their last point (results §3, caveats 1 and 2). The models may be slightly under-trained. That
  would, if anything, weaken W against N (a reading, not a measurement, as results §3 says); it is
  disclosed beside every seed-66800 W-versus-N
  number, as in TASK-077.
- **The predictions on train roots are in sample for W.** R-S is fitted mostly on them (§4.2).
  The fresh-root gate (§9.4) measures exactly what that costs out of sample; the decomposition
  record's train and val readings of S_60 were close (train 0.603, 0.590, 0.607; val 0.578, 0.564,
  0.556 cm), so the cost is not expected to be large, which is a reading, not a measurement.

**The primary seed** for the offline aims and the closed loop is **66800**, by TASK-077's rule
(§4.4: the W with the lowest kept val criterion, 0.350399), carried unchanged so that no choice
here is made after the decomposition record's numbers (R18.2). Its flag is stated beside every
seed-66800 comparison. (Open question 1, §15.)

## 4. The readouts on predicted latents (R18.3)

### 4.1 What each readout is

All three are dual ridges of Stage O's form (`lewm_c1m_v2_offline.ridge`: the relative λ grid,
5 inner folds grouped by root, salt 8205), from the raw 8 × 8 predicted latent at r (24 576-d, the
feature space R8 reads) to the true plate (x, y) at r, one per model seed s:

| readout | fitted on | read by |
|---|---|---|
| **R-S^s** | W^s's predictions at r from each fit root's encoded 405 frame under **the stand-in chunk of that root's own committed aim** | W, L-shuf and L-mean (the scene-blind twins are W rolled from another latent; they keep W's readout, as they kept R8 in TASK-077) |
| **R-N^s** | N^s's predictions at r from the same 405 frames under zero commands | N |
| **R-L^s** (diagnostic, §5.3) | W^s's predictions at r from **L-mean's mean 405 latent** under the same stand-in chunks | nothing at run time; Stage R's command-keyed check only |

The stand-in chunk is the kinematic stand-in's 60 commands (405–464) for the root's own committed
aim, from its logged 405 state (`lewm_c1m_v2_runtime.run_chunks_task`,
`place_planner.primitive_chunks`, horizon 60), exactly as the decomposition record and TASK-077's
Stage G computed them. A root whose chunk is infeasible is counted and kept (the decomposition
record had 0 of 1 745). Every roll-out runs on the CPU, float32, one torch thread per process.

### 4.2 The fit set

**All 1 995 roots of `apple-c1m-v2`** (train, val and TASK-077's read gate split). Reasons:
- none of them is a gate in TASK-080; the gate is fresh (§5);
- every readout learning curve so far was still falling (TASK-077 O1: 0.625, 0.515, 0.446,
  0.404 cm at 1/4, 1/2, 3/4 and all of each fold's fit roots), so more fit roots should help;
- the val and old gate roots are out of sample for W's training (only val was used for selection),
  so including them makes the fit set closer to the fresh-root distribution than train alone.

Reported in Stage R (not gated): each readout cross-fitted over the 1 995 roots in 5 outer folds
(salt 8204), by split; and a nested learning curve on the fresh gate split (1/4, 1/2, 3/4 and all
of the fit roots, salt 8208). (Open question 2, §15: the decomposition record's rule read val
with the gate split unopened; adding the old gate roots to the fit is a change from what it
measured.)

### 4.3 Why the stand-in, and not the executed commands

The closed loop ranks candidates by W's roll-out under their **stand-in** chunks; the executed
commands do not exist yet at 405. The decomposition record measured that a readout fitted on
executed-command predictions reads stand-in predictions at **5.45, 9.47 and 4.62 cm** (h = 60,
val), while one fitted on stand-in predictions reads them at 0.556–0.578 cm. The readout must be
fitted on predictions made under the kind of commands it will read. Because the corpus's aims are
uniform over the box (§2.1), each root's own stand-in chunk is one draw from the candidates' chunk
distribution, so the fit covers the box as the closed loop's grid does.

### 4.4 The encoded ceiling

R8 (Stage O's encoded-frame readout, unchanged) on the **encoded** 8 × 8 frame at r of each fresh
gate root is the ceiling for Stage R (§9.4, R-VOID-CEILING), as E_60 was in the decomposition record
(0.410 cm on val) and c_plate in TASK-077's Stage O (0.404 cm).

## 5. Fresh roots and the corpus-aim confound (R18.4, R18.5)

### 5.1 The confound

TASK-077's corpus commits each root to an aim drawn uniformly over the box built from the **true**
plate p and palm h at 405 (salt 8102). The closed loop builds the same box from **p̂**, R-plate's
reading of the 405 frame (cross-fitted median error 0.148 [0.142, 0.153] cm on the 1 745 train and
val roots, TASK-077 Stage O, protocol §7.5). A stand-in chunk encodes its aim g,
and g = p + a·(h − p) + b·n carries information about the true p. R-S, fitted on corpus roots, may
therefore read part of the plate from the commands' relation to the true p, a relation the closed
loop carries only through p̂ (decomposition record §2.7, point 4). If it does, R-S would read
corpus roots better than closed-loop candidates, and every number on corpus-built aims would be
optimistic.

### 5.2 The fresh corpus `apple-c1m-v2-f` (Stage C′)

500 fresh roots (seeds 65300–65799), collected by TASK-077's privileged scripted collector with
the same per-root record (frames 403–467, executed commands, true plate and palm, a, b, the 405
state, the plate-hidden render at r), split by root before collection (salt 8203) into two halves
of 250 that differ **only in how the committed aim is built**:

| split | the committed aim at 405 | role |
|---|---|---|
| **gate-P** | g = p̂ + a·(h − p̂) + b·n, with **p̂ = R-plate's reading of the onboard 405 frame** (CPU DINOv2 full tokens, exactly as the closed loop computes it) and (a, b) uniform over the box (salt 8202) | **the offline gate split** (§9.4): the aims are drawn from the closed loop's own grid distribution |
| **contrast-T** | g = p + a·(h − p) + b·n with the **true** p, as TASK-077's corpus | reported only: the same statistics on fresh roots with the corpus's aim construction |

Both halves log p̂ at 405. The collector still knows the true plate (it records it as the target
and scores the attempt), so the corpus is privileged-collector data, not a learned result; no
privileged read reaches any arm.

### 5.3 How the confound is handled

1. **The gate reads gate-P only.** Its aims are built from p̂ as the closed loop builds its grid,
   so R1, R2 and the offline aims (§9.4) are measured on the closed loop's aim distribution, not
   the corpus's.
2. **Its size is measured** (reported): the median of e_S on gate-P minus that on contrast-T, with a
   root-bootstrap interval (two independent root sets, salt 8206), per seed. Contrast-T against the
   decomposition record's val numbers (0.556–0.578 cm) also shows the fresh-root cost of the val
   selection with the aim construction held fixed (reported; different root sets, so it is a
   comparison of medians only).
3. **A commands-alone screen gates** (R3, §9.4): R-L, fitted on predictions from L-mean's mean
   latent (no scene information) under the same stand-in chunks, must **not** read the plate within
   τ_commit on gate-P (its lower bound > τ_commit), as O4's plate-hidden check did for the frame. R3
   is a sufficient screen, not a full test of keying: it asks only whether the commands alone read
   the plate, not whether R-S leans on the commands given the scene. On gate-P the plate
   information in the commands comes from p̂, which the closed loop also has legitimately; if the
   commands alone read the plate within τ_commit, the prediction would add nothing measurable, and
   the row is R-COMMAND-KEYED (escalate, no clause).
4. **The split detects only a small confound, and A1–A2 cover the larger threat.** gate-P and
   contrast-T differ only by p̂ − p at 405 (about 0.148 cm in median), so their difference can show
   an effect of about that size only. The larger threat is a readout that echoes each candidate's
   own aim across the 147 candidates, which a one-aim-per-root reading cannot see. A1 and A2 read
   the argmin over the whole grid, so such an echo shows up as missed offline aims, not as a pass.
   **Reported in Stage R:** per gate-P root, the slope of the predicted plate p̃(g) against the aim
   g across the candidates (a least-squares fit over the feasible grid), beside the slope κ = −0.5
   that the rule's fixed-point form implies (g\* = (p − κh)/(1 − κ) is the fixed point of
   g = p + κ(g − h)); a slope near 1 would be the echo.
5. **The closed loop is the final check**: on cohort S, every grid is built from p̂, so a readout
   that relied on the true p would show up as missed bars, never as a pass.

## 6. Arms (carried from TASK-077 §5, with the readouts of §4)

| arm | what it is | readout |
|---|---|---|
| **W** | TASK-077 §5.1's controller form unchanged: the p̂-and-h grid of 147 candidates; each candidate's stand-in chunk rolled by W from the encoded 405 frame to r; score |p̃(g) − g|, lowest wins (ties to the lowest row-major index); at most 10 refinements g ← clip(p̃(g)), stopping at a move ≤ τ_commit/4 | **R-S** (primary seed 66800) |
| **N** (action-blind) | the trained N of the primary seed, zero commands; its p̃ is the same for every candidate | **R-N** |
| **L-shuf** (scene-blind) | W from the encoded 405 frame of the next reset in cohort order that reached 405, with this reset's candidate chunks (R17.20) | R-S |
| **L-mean** (scene-blind) | W from L-mean's mean 405 latent, with this reset's candidate chunks | R-S |
| **L-rand** | a feasible grid candidate drawn uniformly (salt 8209), no refinement | none |
| **H-rule** | `RuleCommit`: the rule's fixed point on p̂ and h, single commit, clipped | none (knows the rule) |
| **H-sysid** | `SysidAim` with TASK-077 Stage O's fit, inverted with W's controller form, clipped | none |
| **H-final(commit)** | the ceiling: `LookaheadAim` once at 405 in cloned state, unclipped; **privileged, not learned** | — |
| **H-read**, **H-now** | reported only, as TASK-077 §5.3; **privileged** | R8 (H-read) |

**The twins keep W's readout.** L-shuf and L-mean are "W without the scene": the same model and the
same readout, rolled from a latent that carries another or no scene. Fitting them their own
readouts would change two things at once. Their own-readout strength is measured offline by R-L
(§5.3). Infeasible candidates are dropped for every arm alike; if all are infeasible the arm aims
at p̂ (a counted attempt). Each arm's clip-binding fraction is reported. The non-inferiority
comparator is the better of H-rule and H-sysid on S (a tie goes to H-rule).

## 7. Seeds, cohorts and salts (R18.9)

**Block 65000–65999.** It lies outside every forbidden range of TASK-077
(`lewm_c1m_v2.FORBIDDEN_RANGES`: every earlier task's block, C1's 57000–57999, R15's 58000–58999
and C1-M's 63000–64999) and outside **TASK-077's whole block 66000–68999**, including its D
(66100–66115) and S (66200–66263) ranges, which were declared but never simulated: they stay
unused rather than being taken over.

| seeds | use |
|---|---|
| 65000–65031 | **K′**: K0′ (32 resets) |
| 65100–65115 | **D**: the development closed loop (16 resets) |
| 65200–65263 | **S**: the gated cohort (64 resets) |
| 65300–65799 | **F**: the fresh corpus `apple-c1m-v2-f` (500 roots: gate-P and contrast-T, 250 each, §5.2) |
| 65900–65999 | **debug**: runner mechanics only; nothing in it is read |
| 66800–66802 | TASK-077's model seeds (the reused checkpoints; nothing is trained) |
| everything else in 65000–65999 | reserved; any use needs its own ruling |

| salt | use |
|---|---|
| 8201 | the move draw per reset: `default_rng(SeedSequence([8201, seed, k]))`, k the re-draw index |
| 8202 | the fresh corpus's uniform (a, b) per root |
| 8203 | the fresh corpus's split into gate-P and contrast-T |
| 8204 | the readouts' reported cross-fit over the 1 995 fit roots (5 outer folds by root) |
| 8205 | λ selection (inner CV grouped by root) in every ridge |
| 8206 | every bootstrap (10 000 resamples) and every power or feasibility simulation |
| 8207 | τ_commit's planted-error direction per reset (K0′) |
| 8208 | the readouts' nested learning-curve subsample |
| 8209 | L-rand's draw |
| 8210 | G4's wrong-command permutation across fresh roots (reported dynamics, §9.4) |
| 8211–8212 | reserved |

**The search** (2026-10-07, at `edb603d`; R18.9). `git grep -w` over all 104 local and remote refs
(paths `src`, `scripts`, `tests`, `configs`, `docs`, `benchmarks`, `.mc`): **65000–65999: only
`65000`**, the GR00T release's training step in `docs/ARENA.md`, `docs/DECISIONS.md`,
`docs/experiments/README.md` and `.mc/tasks/todo/TASK-025-…`, not a seed. The working files of all 38 worktrees under
`/home/huhn/develop/emai/worktrees/` add only `65000` (the same). Salts: no `82xx` value appears on
any ref in `src`, `scripts`, `tests` or `configs`, nor in any worktree's `src` or `scripts`.
(59000–62999 was avoided: seed-like values in manifests and docs, as C1-M's search found; 69000+ was
avoided as R17.11 did.) No R18 label exists on any ref or worktree. Stage 0's test re-checks the
block against the forbidden ranges in code, with TASK-077's block added to them.

## 8. Stages and the staged GO flow

Nothing from K′, D, S or F is simulated before the reported GO of its stage. Every stage runs from
a clean worktree of the merged revision (`scripts/new_worktree.sh --run --from <rev>`), after
G-tests. The coordinator is told before each stage starts, and the main session posts a notice in
chat before the gated stage.

1. **Stage 0 (the preregistration PR, after this draft's review).** The stage code in new modules,
   the manifest (DRAFT until the freeze), the tests (seed ranges and salts against every forbidden
   range including 66000–68999; the reused artifacts' sha256 checks; that W reads R-S and N reads
   R-N and that neither ever reads R8; that a gate-P root's aim is built from p̂ and never from the
   true plate; the `"not evaluated"` sentinel; no runner imports; `install_guards`,
   `assert_local_import`; no privileged read in W or a twin; every row in both directions), debug
   smokes (one root per fresh split, one D-style attempt per arm), the scale probes that set §11's
   caps, and the power simulation with K0′'s curve slot. **Plus one development dry run
   (R18.13), reported only, setting no bar:** the offline aims of §9.4 (A1, A2) on TASK-077's 250
   **val** roots, with R-S, R-N and R-L fitted on the other 1 745 old roots (train and the old
   gate split), mapped through K0's curve. Its only use is to size the power table before the
   freeze (§10). It is disclosed as optimistic twice over (val selected the checkpoints, and val's
   aims are built from the true plate).
2. **K0′ (on a reported GO; CPU only; before the freeze)** (R18.6). On K′'s 32 fresh resets, at
   ρ\* = 4 cm: H-final(commit)'s converged aim plus a planted error of 0, 0.5, 1, 1.5, 2 or 3 cm in
   a per-reset uniform direction (salt 8207), every level on the same 32 resets; the ceiling
   N_K′(0); r_K′ and the palm speed at 405. **τ_commit for TASK-080** is TASK-076 K0's rule applied
   to the **pooled 64 resets of K and K′**: the largest level such that every level up to it
   reaches ≥ 56/64. The pooled τ curve is the one the offline aims map through.
   - **Stops, all CAL-ESCALATE** (escalate, no clause; nothing is frozen): K′ level 0 < 28/32
     (carried; implied by the next stop, since both count level 0); N_K′(0) < 30/32; r_K′ later
     than 465 or undefined; median palm speed at 405 > 0.5 cm per step; the pooled τ_commit
     undefined.
   - **If the pooled τ_commit falls to 0.5 cm**, every bar tied to it tightens with it (R1's bar
     becomes 0.5 cm, and the decomposition record's upper bounds of 0.61–0.64 cm would not meet
     it). That is the honest consequence of a measured tolerance and is not an escalation by
     itself.
3. **The freeze.** Status FROZEN with the frozen-block sha pin, K0′'s values in the block, merged
   on an independent reviewer's reported APPROVE.
4. **Stage C′, the fresh corpus (on a GO; CPU).** §5.2. 500 roots; rows **CORPUS-ESCALATE** (more
   than 2 % of the roots excluded, or more than 2 % of either half; escalate, no clause) and
   **CORPUS-SEALED**. The sealed manifest's sha256 is checked by every later stage.
5. **Stage R, the fresh offline gate (on a GO; GPU for featurisation only, then CPU).**
   - Featurisation of the fresh corpus's kept frames (8 × 8 at every kept frame; full tokens at
     405) with TASK-077's featurise code on CUDA through `scripts/gpu_run.sh --wait
     --min-free-gib 8 --board --who oej:task080-featurise`, with G-anchor.
   - The roll-outs for the fits on the 1 995 old roots, the fits R-S^s, R-N^s and R-L^s, and their
     reported cross-fit.
   - `first_outcome_utc` is written, and only then is the fresh corpus's gate-P split opened: R0–R3
     on all three seeds, then A1 and A2 on the primary seed (§9.4).
   - Reported beside them: contrast-T, the confound's size, G1–G4 at h = 60 on gate-P (with N,
     copy-last and G4's wrong and zero commands), the learning curve, and the offline aims of
     every arm including H-rule and H-sysid.
   - Rows: §9.4.
6. **Stage D, the development closed loop (on a GO after R-PASS; CPU).** W, N, L-shuf, L-mean and
   H-final(commit) on D's 16 resets; **L-DEV-STOP** (escalate, no clause) if W < 12/16, or
   H-final(commit) < 14/16, or W − max(N, L-shuf, L-mean) < +3/16 (carried from TASK-077 §7
   step 8, with its false-stop probabilities). Nothing is refitted on D.
7. **Stage S, gated (on a reported GO after D-PASS; CPU).** Cohort S, every arm of §6 once per
   reset, paired, with TASK-077's determinism re-run of W on S's first four resets (reading gated
   at 0.1 cm, commit target at 0.6 cm, success identical; R17.21, R17.27, R17.38). Rows: §9.5.
8. **Results PR.** Every arm is reported, privileged arms labelled as not learned, and an
   independent reviewer checks every restated number.

## 9. Gates, bars and rows

Intervals are root- or reset-clustered bootstrap percentile intervals, 10 000 resamples, 95 %,
salt 8206, with TASK-077's estimators (`lm.median_ci`, `lm.median_ratio_ci`,
`lm.median_difference_ci`). Every bar is labelled **measured**, **definitional**, **carried** or
**allocation**, as TASK-077 §8.

### 9.1 K0′

As §8 step 2: **CAL-ESCALATE** or **K0′-PASS**.

### 9.2 Stage C′

**CORPUS-ESCALATE** or **CORPUS-SEALED** (§8 step 4).

### 9.3 Stage D

**L-DEV-STOP** or **D-PASS** (§8 step 6).

### 9.4 Stage R, the fresh offline gate (gate-P, 250 roots; h = 60)

e_X denotes the plate error at r, in cm, of readout X on its own model's prediction from the
root's encoded 405 frame under the root's own stand-in chunk (zero commands for N).

| gate | passes when | seeds | source |
|---|---|---|---|
| **R0** ceiling | the upper bound of the median error of R8 on the **encoded** frame at r ≤ τ_commit | (one; no model) | **measured** (τ_commit) |
| **R1** precision | the upper bound of median e_S ≤ τ_commit | all three | **measured** (τ_commit; G5 (a)'s form with R-S in place of R8) |
| **R2** against N | the upper bound of median e_S / median e_N (R-N on N's prediction; paired by root) < 1.0 | all three | **definitional** (G5 (b)'s form) |
| **R3** commands-alone screen | the **lower** bound of median e_L (R-L on W rolled from the mean latent under the same chunks) > τ_commit | all three | **measured** (O4's form against τ_commit) |
| **A1** offline aims, the bar | W's predicted count ≥ 56/64 | primary (66800) | **measured** (the pooled τ curve) against G-bar's 56/64 |
| **A2** offline aims, the twins | W's predicted count − each of N's, L-shuf's, L-mean's and L-rand's ≥ +7/64 | primary | **a necessary floor**, not definitional: +7/64 is the minimum number of discordant pairs at which the McNemar test can pass at all (R17.15); A2 applies it to differences of predicted counts, which are means, not discordant pairs |

**The offline aims** (TASK-077 §7 step 7's form, carried): on every gate-P root's logged 405 state,
each arm's own controller code with the stand-in chunks, the grid from the p̂ the collector logged,
no fallbacks; the aim error against the rule's fixed point g\* = (p − κh)/(1 − κ); the predicted
count of 64 maps each root's aim error through the pooled τ curve (TASK-077's mapping) and scales
the mean to 64. Each arm's median aim error with its interval, 87.5th percentile, clip-binding
fraction and predicted count are reported, H-rule and H-sysid included (reported only: their
predicted count against W's is the offline reading of G-NI). **A predicted count is a prediction,
not a closed-loop count**, and is never reported as one.

**Why A1 and A2 gate.** TASK-077's offline aims were reported only; they predicted W at 22.6/64,
far below the bar, and the closed loop never ran. Here they are the ranking check the decomposition
record asked for (§2.7, point 3): R1 measures the predicted plate at each root's own aim, A1 and A2
measure whether W's argmin over 147 candidates lands near the fixed point and well away from the
twins'. A closed loop that the offline aims already predict below the bar or within the McNemar
test's minimum separation is not run.

**Rows, first match:**

| row | condition | consequence |
|---|---|---|
| **V** | the void rule (§11) | one repeat after a recorded fix |
| **R-VOID-CEILING** | R0 fails (the fresh encoded frame does not read the plate within τ_commit) | escalate, no clause; nothing below is informative |
| **R-COMMAND-KEYED** | R3 fails on any seed | escalate, no clause: the commands alone read the plate, so R-S may be keyed by the aim's construction (§5.3) |
| **R-NO-BAR** | R1 or R2 fails on any seed | escalate, no clause: the decomposition's reading does not hold on fresh roots with the closed loop's aim construction |
| **A-NO-BAR** | A1 fails | escalate, no clause: the readout reads the plate but the ranking does not predict the bar |
| **A-TWIN** | A2 fails for any of the four | escalate, no clause: the offline aims leave no room for the declared twin test |
| **R-PASS** | otherwise | Stage D may get its GO |

None of these rows fires the clause, closes anything or changes TASK-077's row.

**Reported only in Stage R.** Contrast-T's R1–R3 numbers; the confound's size (§5.3); each
readout's cross-fit on the 1 995 old roots by split and its learning curve on gate-P; G1–G4 at
h = 60 on gate-P (TASK-077 §8.2's definitions and bars, which these checkpoints passed on the old
gate split; here reported, not gated, R18.7); every statistic at 1.5 τ_commit beside τ_commit
(TASK-077 §8.2's scaled tolerance); R-plate's error at 405 on F.

### 9.5 Stage S, the gated rows (carried from TASK-077 §8.4 unchanged)

- **G-bar:** W(S) ≥ 56/64 (τ_commit's bar fraction, TASK-076 G1's form). **S-VOID-CEILING** if
  H-final(commit)(S) < 56/64.
- **G-NI:** the lower bound of the paired 95 % interval of W − C > −8/64, C the better of H-rule
  and H-sysid on S; δ = 8/64 an **allocation**.
- **G-N, G-shuf, G-mean, G-rand:** W > arm, exact one-sided McNemar p < 0.01 on the paired resets.
- **"Detectably"** as R17.15: a twin test passes at McNemar p < 0.01; W is detectably no better than
  an arm when that test fails and the upper bound of W − arm < +7/64; W is detectably inferior when
  the upper bound of W − C < −8/64.

| row (first match) | condition | consequence |
|---|---|---|
| **V** | the void rule | one repeat after a recorded fix |
| **S-VOID-CEILING** | H-final(commit)(S) < 56/64 | escalate, no clause, no claim |
| **L-NO-GAIN** | for at least one of N, L-shuf, L-mean, L-rand, the McNemar test fails **and** W is detectably no better | **the clause fires** (§12) |
| **L-INFERIOR** | W is detectably inferior beyond δ | **the clause fires** (§12) |
| **L-PASS** | G-bar, G-NI and the four McNemar tests all pass | **the primary claim, "LeWM-driven closed-loop success"**; "LeWM needed" is then reported |
| **L-TWIN-NEAR** | at least one McNemar test fails, and every failed one is a miss within noise | escalate, no clause, no claim |
| **L-NEAR** | G-NI fails, but W is not detectably inferior beyond δ | escalate, no clause, no claim |
| **L-BAR** | otherwise (G-bar fails, with G-NI and the tests passing) | escalate, no clause, no claim |

Reported in every row: every arm's count, every paired difference with its discordant counts and
interval, H-read's and H-now's counts, the Stage R predicted counts against the S counts, and the
secondary claim's test. Any repeat after L-TWIN-NEAR, L-NEAR or L-BAR needs fresh seeds and its own
ruling.

## 10. Power (R18.11)

**G-bar** (exact binomial, P(X ≥ 56 of 64)): 0.59 at a true rate of 0.875, 0.73 at 0.89, 0.86 at
0.906, 0.94 at 0.922, 0.98 at 0.9375 and > 0.99 at 0.969 (recomputed for this draft; the values TASK-077 §8.4
gives agree, and 0.922 is added here).

**G-NI** is carried with its estimator, so TASK-077's Stage-0 simulation applies unchanged
(R17.19): size at the margin 2.4–3.8 % with one comparator and 0.9–3.5 % with the better of two
(nominal 2.5 %); power with the better of two 0.69–1.00 at W = C, 0.35–0.86 at W = C − 2/64 and
0.14–0.43 at W = C − 4/64 (C at 30–32/32). **Non-inferiority stays demanding**: W must be about as
precise as the rule.

**The twin tests.** The refit readout makes N stronger too: R-N reads the plate at 1.48–1.68 cm on
val, against 2.29–2.58 cm under R8 on the old gate split, so G-N, not the scene-blind twins, is the
likeliest binding test. Exact one-sided McNemar at p < 0.01 on 64 paired resets (this draft;
200 000 trials per cell, salt 8206; "nested": every N success is a W success; "independent":
outcomes independent given the rates; "half": halfway between in discordant probabilities):

| W's true rate | N's true rate | nested | half | independent |
|---|---|---|---|---|
| 0.875 | 0.70 | 0.95 | 0.61 | 0.46 |
| 0.875 | 0.75 | 0.70 | 0.32 | 0.23 |
| 0.906 | 0.60 | 1.00 | 0.99 | 0.95 |
| 0.906 | 0.70 | 0.99 | 0.80 | 0.67 |
| 0.906 | 0.75 | 0.89 | 0.54 | 0.42 |
| 0.906 | 0.80 | 0.53 | 0.25 | 0.18 |
| 0.9375 | 0.70 | 1.00 | 0.93 | 0.86 |
| 0.9375 | 0.75 | 0.97 | 0.77 | 0.66 |
| 0.9375 | 0.80 | 0.79 | 0.49 | 0.38 |

**Reading.** The design has good power only if W sits near 0.91–0.94 **and** N sits at or below
about 0.70. N's offline predicted count under R-N is unknown today; if it lands near 48/64 (0.75),
the probability of L-PASS is about one half at best, and a twin miss within noise (L-TWIN-NEAR) is a
likely row. A2's +7/64 is the minimum at which the test can pass, not a power guarantee. The
clause's false-fire probabilities at the boundaries are those of TASK-077's Stage 0 (2.3–3.1 % per
twin at +7/64, and 9–12 % if all four twins sat exactly at the boundary at once; 0.9–3.1 % for
L-INFERIOR at δ), since the tests and estimators are unchanged.

**The offline gate.** On 250 val roots the decomposition record's upper bounds sat 0.056–0.061 cm
above medians of 0.556–0.578 cm; gate-P has as many roots (250). R1 therefore passes unless the
fresh-root median rises by about 60 % (to about 0.9 cm); R2 passes unless the ratio e_S / e_N rises
by a factor of about 2.2 (its upper bounds were 0.398–0.451). These are readings from val, not
measurements on fresh roots.

**Stage 0's dry run** (§8 step 1) replaces these unknowns with development readings of every arm's
offline predicted count on val, and the freeze PR re-states this section with them. If the dry run
shows W − N below +10/64, the freeze PR must say so and the reviewer decides whether the 64-reset
cohort is kept (open question 3).

## 11. Void rule, guards, caps and memory (carried from TASK-077 §10)

- **Void and repeat:** TASK-077 §10.1 (R16.10's form; one repeat per stage after a committed, pushed
  and recorded fix; a second V of the same stage ends TASK-080 as INCONCLUSIVE; Vs in different
  stages do not add up). There is no Stage T.
- **Guards:** TASK-077 §10.2 (G-tests, G-sentinel, G-hash and G-frozen with **TASK-077's frozen
  block and pins added** to the checked pins, G-repro, G-anchor, G-threads with the same pinned
  environment, G-quiet, G-split, G-privileged, G-finite, G-memory at 12 GiB process-tree PSS,
  G-disk at 10 GiB free, G-GPU with `gpu_run.sh --wait --min-free-gib 8 --board`). **G-split here:**
  every fit used by a later stage (R-S, R-N, R-L) is fitted on the 1 995 old roots only; the fresh
  corpus is read by Stage R's gates and reports only and is never fitted on; gate-P is opened only
  after `first_outcome_utc`; contrast-T is reported only. **G-fresh (new):** the runner refuses any
  fit whose roots include a fresh-corpus root, and any closed-loop or K0′ seed outside its range.
- **Caps:** set by Stage 0's scale probes at ≥ 1.5 × the measured worst case and written here
  before the freeze. Working values for this draft: K0′ 7 200 s; Stage C′ 7 200 s; Stage R
  featurisation 3 600 s and the rest 14 400 s; Stage D 7 200 s; Stage S 21 600 s; per closed-loop
  attempt 300 s.

## 12. The abandonment clause and its scope (R18.10)

**It fires on L-NO-GAIN or L-INFERIOR only**, as TASK-077 §11 and R17.15 define them. It does not
fire on CAL-ESCALATE, CORPUS-ESCALATE, R-VOID-CEILING, R-COMMAND-KEYED, R-NO-BAR, A-NO-BAR, A-TWIN,
L-DEV-STOP, S-VOID-CEILING, L-TWIN-NEAR, L-NEAR, L-BAR, V or INCONCLUSIVE.

**What closes: TASK-077's scope (R17.10) unchanged.** TASK-077's clause did not fire, so its scope
is still open, and TASK-080 tests a member of it:

> "LeWM aim selection with a single aim committed at 405 under the declared reactive-plate rule
> (κ = −0.5, L = 2, s1 = 525) on v2, with the declared post-pick plate move of radius ρ\* = 4 cm
> (the disc, which also covers the −y half-disc, at radii ≤ 4 cm), from onboard 112 px frozen
> DINOv2 pooled tokens, with TASK-066-family predictors."

with TASK-077's one declared narrowing (a 4 × 4 pooled latent on a corpus larger than 1 024 roots
stays open). **If the clause fires, the results document states the restored scope plainly**: one
8 × 8 run read by a dual ridge fitted on its own stand-in predictions closes, under this condition,
every other pooled grid (except that 4 × 4 case) and every readout of the predicted latent,
including trained readout heads and readouts fitted on executed-command or other predictions, not
only the readout that was run. **What does not close** is TASK-077 §11's list unchanged: the full,
unpooled token grid; history longer than one, `action_chunk` and `predictor_step_embedding`; a
fine-tuned or another encoder; other views or resolutions; other κ, L, commit steps, move
distributions or radii above 4 cm; C1 without the move; C2; TASK-076's results; the LeWM backend;
v2; the product goal.

## 13. Compute estimate (R18.11)

| stage | estimate | basis |
|---|---|---|
| K0′ | about 10 min CPU | TASK-077 K0: 579 s for about 350 attempts on 6 workers |
| C′ | about 10–15 min CPU | TASK-077 Stage C: 1 234 s for 2 000 roots; gate-P adds one CPU DINOv2 encoding and one R-plate read per root |
| R | about 2–5 min GPU (featurisation), then about 1–2 h CPU | 500 × 65 frames against TASK-077's 212.7 s probe for 1 995 roots; about 18 000 fit roll-outs (1 995 roots × 3 seeds × 3 models) and about 4 500 fresh ones at 0.055 s each on one thread; the decomposition record's 812 s for 15 700 roll-outs and 135 ridges; the offline aims (TASK-077 Stage G: 2 369 s with its offline aims) |
| D | about 15 min CPU | TASK-077 §12 |
| S | 1–2 h CPU | TASK-077 §12 |
| Stage 0's dry run | about 30–60 min CPU | the decomposition record's scale plus the offline aims on 250 val roots |

No training: the GPU need is the featurisation only, against about 31 h for retraining the six
models (§3). Disk: the fresh corpus about 0.3 GB of frames and about 3.2 GB of 8 × 8 features
(500 × 65 × 98 KB); the main checkout had 75 GB free on 2026-10-07.

## 14. Risks, stated now

- **Fresh roots may read worse.** The decomposition record's numbers are optimistic (checkpoints
  selected on val; readouts fitted mostly on in-sample predictions; the true-plate aim
  construction). R-NO-BAR is a live outcome.
- **The ranking is unmeasured.** R1 reads the plate at each root's own aim; whether the argmin over
  147 candidates lands near the fixed point is A1's question, measured for the first time in Stage 0's
  dry run (development) and Stage R (gated).
- **N is a stronger twin under its own readout** (§10). L-TWIN-NEAR is a likely row if N's
  closed-loop rate lands near 0.75.
- **Non-inferiority is demanding**, as in TASK-077 (§10).
- **τ_commit's margins were thin in K0** (the 0.5 and 1.0 cm levels each one success above 28/32);
  the pooled K0′ rule may move τ_commit to 0.5 cm, which would make R1 likely to fail (§8 step 2).
- **W-66800 is flagged** (`last_two_triggered`); all W curves were lowest at their last point.
- **The scorer and condition are imposed simulator laws**, as TASK-077 §13 says; nothing here
  transfers as such to Arena or the real G1.

## 15. Open questions for the freeze

1. **The primary seed.** Carried rule (66800, flagged) or a rule on the readouts' cross-fitted
   error over the 1 995 old roots (decided in Stage R before gate-P opens)? The draft carries the
   rule to avoid a choice made after the decomposition record's numbers.
2. **The fit set.** All 1 995 old roots (the draft) or the 1 745 train and val roots the
   decomposition record read?
3. **The cohort size and A2's separation.** 64 resets and +7/64 (carried) or, if Stage 0's dry run
   puts W − N below about +10/64, a larger S (for example 96 resets with the bar at the same
   fraction) or a higher A2 bar, declared before the freeze?
4. **τ_commit.** Pooled K ∪ K′ (the draft) or TASK-077's carried 1.0 cm with K0′ as a ceiling check
   only?
5. **G1–G4 on fresh roots.** Reported (the draft; they passed on the old gate split with these
   checkpoints) or re-gated?
6. **TASK-079's precondition.** The draft proposes that a TASK-080 L-PASS counts as "TASK-077 L-PASS
   (or its equivalent row)" in PLAN.md (R18.12); TASK-078's pass and an owner ruling on Arena are
   still required.
