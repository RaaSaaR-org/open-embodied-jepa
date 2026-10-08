# Apple→Plate LeWM committed aim under C1-M, read by a readout fitted on W's predicted latents (TASK-080)

**STATUS: FROZEN** (after K0′-PASS; R18.22, decided by Claude under owner delegation). The frozen
block is `src/embodied_jepa/lewm_pr_v2.py`; its sha256
**`0fc095dc947f0ac74ebf6d1c541098a1e6fec1897592256f97ac8962b97be064`** is pinned in
`tests/test_lewm_pr_v2.py` and in the manifest (`benchmarks/manifests/apple-lewm-pr-v2.json`),
with the manifest's file pins and this document's sha256. **The freeze takes effect when it is
merged on an independent reviewer's reported APPROVE.** Each later stage (C′, R, D, S) still needs
its own reported GO (§8). K0′'s record is §8.1, Stage C′'s §8.2, Stage R's plan §8.3,
Stage R's result §8.4, Stage D's plan §8.5, Stage D's result §8.6 and Stage S's plan §8.7.

History. The DRAFT (R18.1–R18.14) was reviewed independently (#153). Stage 0 (R18.15–R18.21,
[Stage-0 record](apple_lewm_c1m_v2_pred_readout_stage0.md), #154) added the code, debug smokes on
65900–65999 only, the caps and R18.13's development dry run on TASK-077's val roots, which sets no
bar; §15's questions 1–5 are ruled there. K0′ then ran once on a reported GO, and this freeze
writes its values into the frozen block. Stage C′ then sealed the fresh corpus (§8.2, R18.23), and
Stage R ended R-PASS (§8.4, R18.25), an offline result: its predicted counts are predictions, not
closed-loop counts. Stage D, the non-gating development closed loop on 16 resets, then ended
D-PASS (§8.6, R18.27): W 16/16, N 8/16, L-shuf 5/16, L-mean 9/16, the privileged ceiling
H-final(commit) 16/16; a development result, not the gated one. No seed of S or F has been
simulated in closed loop. A later stage's GO approves that stage only.

- **Admitted by:** the TASK-077 decomposition record's row **D-READOUT**
  ([apple_lewm_c1m_v2_decomposition.md](apple_lewm_c1m_v2_decomposition.md) §2, R17.53–R17.54,
  #152). That row names "a new preregistration whose controller reads the plate with a readout
  fitted on W's predicted latents under the stand-in at r, on fresh gated roots". This document
  is that preregistration, frozen at R18.22.
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

The canonical status sentence (DECISIONS 2026-10-02, R7), verbatim **as it stood at the
freeze**. This protocol did not change it (R18.12) until Stage D's record: **R18.28** (decision
2026-10-08 (c)) replaces its v2 LeWM clause with Stage D's development counts, as a named
exception to R18.12 (§8.6). The quote below is kept as frozen:

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
| **R-L^s** (diagnostic, §5.3) | W^s's predictions at r from **L-mean's mean 405 latent** under the same stand-in chunks | nothing at run time; Stage R's commands-alone screen (R3) only |

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
   g across the candidates, as the 2 × 2 least-squares matrix over the feasible grid, compared with
   κI, κ = −0.5, the slope that the rule's fixed-point form implies (g\* = (p − κh)/(1 − κ) is the
   fixed point of g = p + κ(g − h)); a matrix near I would be the echo.
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

### 8.1 K0′'s result: K0′-PASS (R18.22)

K0′ ran once, on the reviewer's reported GO (#154,
<https://github.com/RaaSaaR-org/open-embodied-jepa/pull/154#issuecomment-6047218319>), at
`931281a`, on a clean tree with the DRAFT frozen sha `d3ebc26b…54fe`, in the worktree
`/home/huhn/develop/emai/worktrees/task080-k0`. It ran on the CPU with 6 workers and without the
GPU lock, from 21:34:57 to 21:43:13 UTC (496 s). In-run G-tests passed (2066 passed, 37 skipped).
G-repro passed 8 of 8 checks. There were no render disagreements. Load at the start was
0.67 / 1.76 (1- and 5-minute; the G-quiet bar is 2.0 / 2.0). Peak process-tree PSS was 8.08 GiB
(cap 12).
- **Report:** `outputs/task080-k0-1/report.json` (git-ignored), sha256
  `a0939e3eea7b695d1c3029e733970bcd123ea62db76192104c69479ad9314c16`; its log
  (`outputs/task080-k0-1.log`, sha256 `f677d2dd…3d02`) and its stdout capture (empty, 0 bytes) are
  beside it. A copy of all three is in `~/develop/emai/evidence/task080-k0/`, with the manifest
  `_checksums/task080-k0.sha256`.

**τ_commit**: counted successes of 32 for H-final(commit)'s aim plus a planted error (direction
salt 8207), every level on the same 32 resets K′ = 65000–65031, beside TASK-077's K (66000–66031):

| planted error (cm) | 0 | 0.5 | 1 | 1.5 | 2 | 3 |
|---|---|---|---|---|---|---|
| K′ counted successes / 32 | 32 | 29 | 31 | 14 | 18 | 5 |
| K (TASK-077 K0) / 32 | 32 | 29 | 29 | 18 | 11 | 2 |
| **pooled K ∪ K′ / 64** | **64** | **58** | **60** | 32 | 29 | 7 |
| K′ median landing miss (cm) | 0.08 | 0.75 | 1.49 | 2.24 | 2.98 | 4.48 |

- **Pooled τ_commit = 1.0 cm** (every level up to it reaches ≥ 56/64; 1.5 cm reaches 32/64). It
  is TASK-077's value, so nothing tightens (§8 step 2): R0's, R1's and R3's bars stay at 1.0 cm,
  and the offline aims (A1, A2) map through the pooled curve above.
- **The ceiling N_K′(0) = 32/32.**
- **r_K′ = 459**, so the frozen r = 465 stands.
- **The history check.** At 405 the palm speed had median 0.00277 cm per step (maximum 0.0141),
  and m2 had median 0.00217 cm (maximum 0.0135); the plate's remaining distance at 405 had median
  6.52 cm.
- **No stop fired** (K′ level 0 32 ≥ 28; N_K′(0) 32 ≥ 30; r_K′ 459 ≤ 465; palm speed 0.00277 ≤
  0.5 cm per step; pooled τ defined). There were no refusals, the clip-binding fraction was 0 and
  there were no fallbacks at any level.
- **Failed seeds:** at 0.5 cm 65008, 65014, 65024; at 1.0 cm 65001; the frozen block lists every
  level's.

**Margins, disclosed (R18.22).** Each passed its rule; each is stated with its distance to the bar:
- **The pooled 0.5 cm and 1.0 cm levels sit 2 and 4 successes above 56/64** (58 and 60), against
  1 and 1 of 28/32 in K alone. Pooling did what R18.19 asked of it: the 1.0 cm level is now
  measured on 64 resets, not 32.
- **The K′ curve is not monotone** (1.0 cm 31 > 0.5 cm 29; 2.0 cm 18 > 1.5 cm 14), and K′ is more
  lenient than K at 2 and 3 cm (18 against 11, 5 against 2). The two non-monotone steps within
  K′ (2 and 4 of 32, paired on the same resets) are of the size binomial noise gives on 32 resets
  per level; the K′-against-K gaps (7 and 3 of 32, on different resets) are within chance too
  (Fisher two-sided p ≈ 0.13 and 0.43; #155 review, nit 1) and sit above τ_commit, so they do not
  move it. The mapping uses the pooled counts as measured, without smoothing, as declared
  (TASK-077's mapping: linear between the planted levels).
- **r_K′ = 459 against r = 465:** 6 steps of margin (K: 5).
- **What this means for G-bar.** A W whose aim error sits at about τ_commit would succeed at about
  60/64 ≈ 0.94 by the pooled curve, where G-bar passes with probability about 0.98 (§10), and
  at 0.5 cm at about 58/64 ≈ 0.906 (probability 0.86). These are curve readings, not predictions
  of W.

**The dry run re-mapped through the pooled curve** (R18.19; `simulate --dryrun … --k0-prime …`,
development, optimistic, setting no bar; primary seed 66800, flagged `last_two_triggered`;
predicted counts are not closed-loop counts): W **57.35** of 64 (56.88 through K alone), N
28.36, L-shuf 23.79, L-mean 28.02, L-rand 11.62; H-rule 58.62 and H-sysid 54.49 (reported). W − N
is **+29.0** (N is the closest twin; L-mean +29.3, L-shuf +33.6, L-rand +45.7). A1's form (≥ 56) holds by **1.35 of 64**
(0.88 through K); A2's form (≥ +7) holds for every twin. G-bar's power at W's predicted rate
(57.35/64 ≈ 0.896) is **0.78** (0.72 through K); the twin tests' power is at least 0.999 in every
coupling (lowest 0.99946, N, independent). Report `outputs/task080-simulate-k0p-1/report.json` (sha256 `5e97ac85…48cb`), run at
`931281a` from the K0′ worktree after G-tests (2066 passed, 37 skipped); copied with K0′'s files to
`~/develop/emai/evidence/task080-k0/`. These numbers come from TASK-077's val roots with aims built
from the true plate; Stage R measures them on fresh gate-P roots.

### 8.2 Stage C′'s result: CORPUS-SEALED (R18.23)

Stage C′ ran once, on the reviewer's reported GO (#155,
<https://github.com/RaaSaaR-org/open-embodied-jepa/pull/155#issuecomment-6048275092>, posted
22:36:26 UTC), at `514110d`, the freeze's merge commit, on a clean tree with STATUS FROZEN and the
frozen sha `0fc095dc…be064`, in the worktree `/home/huhn/develop/emai/worktrees/task080-stagec`,
with the command the GO named. It ran on the CPU with 6 workers and without the GPU lock, from
22:37:16 to 22:45:01 UTC on 2026-10-07 (465 s; cap 7 200 s).
- **Guards.** In-run G-tests passed at `514110d` on a clean tree (2070 passed, 37 skipped, with
  CUDA hidden; 22:37:16–22:39:58). G-hash checked TASK-076's 84 pins, TASK-077's 13 pins and
  this task's 7 own pins, with the protocol document's pin `5b28c2d0…e78c`; the revision at the end
  was still `514110d` and the tracked tree stayed clean. G-repro passed 8 of 8 checks. Load at the
  start was 0.21 / 0.13 (bar 2.0 / 2.0); MemAvailable 24.8 GiB; 68.3 GiB free on disk.
- **Memory.** Peak process-tree PSS was **9.78 GiB** against the 12 GiB ceiling (RSS 12.01 GiB,
  11 processes).
- **Row: CORPUS-SEALED.** 499 of the 500 roots 65300–65799 are kept. The one excluded root,
  **65688** (gate-P), was excluded as `no_decision` (no commit decision at 405), so the excluded
  fractions are 0.2 % overall, 0.4 % in gate-P and 0 % in contrast-T, against the 2 % bar in each.
  The split, drawn before collection (salt 8203), was 250 and 250; the kept halves are **gate-P
  249** and **contrast-T 250**. The clause does not fire.
- **Render disagreements.** None in the corpus. In G-repro's re-render of TASK-072's seed 51307,
  one of the three renders differed from the other two in 4 pixels by at most 1 level, with equal
  states; the majority frame was used, as declared (TASK-077 §7.3 saw the same kind on two corpus
  seeds).
- **The log** has one line per 60-root chunk of the form `fresh corpus 65300-65359: 0/60`, and a
  last line for the 20-root chunk, `fresh corpus 65780-65799: 0/20` (#156 review, nit d). The
  count is the scorer's `success` field, which is false for every corpus attempt because the
  attempt stops at `STOP_STEP`, before the place; TASK-077's Stage C log reads the same
  (`corpus 67000-67059: 0/60`). It is not an exclusion count.
- **Sealed manifest:** `outputs/task080-corpus-1/corpus/manifest.json` in that worktree
  (git-ignored; 499 root files and the manifest, 302 MB), sha256
  **`deebd83db6de53e23dbde0b921ae7f7c1cf79cb24c6dd27bfae066c2f5017c4e`**, with provenance
  `privileged_scripted_collector: true`, `learned_control: false`, revision `514110d` and resets
  digest `e7f5dbf1…bf27`. Every later stage requires it (`--corpus-sha256`, §8 step 4).
- **Report:** `outputs/task080-corpus-1/report.json`, sha256
  `63b085df119f59219c24057ad4f592a59d20f43ff5ca210c32fd0e570bf6b0ca`; its log
  (`outputs/task080-corpus-1.log`, sha256 `540c2ee5…4751`) and its stdout capture (empty, 0 bytes)
  are beside it. A copy of all three and of the sealed manifest is in
  `~/develop/emai/evidence/task080-stagec/`, with the manifest `_checksums/task080-stagec.sha256`
  (sha256 `89779255…49dc`). The root files themselves are not copied, as TASK-077's corpus was
  not; they stay in the `task080-stagec` worktree, which is not edited and must not be removed
  before TASK-080's results PR.

At this record (before Stage R), none of Stage R's readouts (R-S, R-N, R-L, R8) had been fitted on
or evaluated against a fresh root, and no gate statistic had been computed; gate-P opens only
inside Stage R after `first_outcome_utc` (§8 step 5, §11 G-split). By design (§5.2), R-plate
read every fresh root's onboard 405 frame during collection: gate-P's aims are built from p̂, and
both halves log `p_hat405`. (Wording corrected in R18.25, #156 review, nit b; it read "Nothing in
the corpus has been featurised or read: no readout has seen a fresh root".)

### 8.3 Stage R's plan (R18.24)

Stage R runs once, on its own reported GO at this record's merge commit, from a fresh clean
worktree of that commit (`scripts/new_worktree.sh <dir> --run --from <merge sha>`). One GO covers
its three invocations, in this order, each from the worktree root:

1. **`tests`** (CPU, a few minutes): the full suite at HEAD with CUDA hidden, written as a stage
   report. It is the G-tests record the GPU stage requires (`--tests-record`: same revision,
   exit 0, clean tree, finished after HEAD's commit). It simulates and reads nothing.
2. **`featurise`** (GPU) through `scripts/gpu_run.sh --wait --min-free-gib 8 --board --who
   oej:task080-featurise`, with `--corpus` the sealed folder of §8.2 and `--corpus-sha256
   deebd83d…7c4e`. It writes the 8 × 8 features and the full tokens at 405 for both halves
   (about 3.3 GB, scaled from TASK-077's 13 GB for 1 995 roots), checks G-anchor on 256 frames and
   records the GPU memory peak. Cap 3 600 s (§11); expected about a minute.
3. **`rgate`** (CPU) with the same corpus and sha, `--features` the featurise output, TASK-077's
   featurisation, Stage O fits, six job reports and T-PLANNED plan (each checked by sha256), and
   TASK-076's evidence root (G-repro). It runs its own in-run G-tests and G-quiet, the roll-outs
   and fits on the 1 995 old roots with their cross-fit, writes `first_outcome_utc`, and only then
   opens gate-P and contrast-T. Cap 14 400 s (§11); Stage 0 scaled it at ≤ 4 200 s (record §4).
   The dry run's peak PSS was 9.96 GiB against the 12 GiB ceiling.

If one of the three ends in anything other than its expected outcome (TESTS-PASS, FEATURISED,
or a row of §9.4 for `rgate`), nothing further is launched and the case is ruled under §11 before
anything else runs. The exact commands are in the GO.

### 8.4 Stage R's result: R-PASS (R18.25)

Stage R ran once, on the reviewer's reported GO (#156,
<https://github.com/RaaSaaR-org/open-embodied-jepa/pull/156#issuecomment-6048908259>, posted
23:26:58 UTC), at `33cea5c`, the record's merge commit, on a clean tree with STATUS FROZEN and the
frozen sha `0fc095dc…be064`, in the worktree `/home/huhn/develop/emai/worktrees/task080-stager`,
with the three commands the GO named, in order. **Everything below is offline**: readouts and
offline aims on logged 405 states of fresh roots. **A predicted count is a prediction, not a
closed-loop count; no closed loop of this task has run on a counted cohort, and no LeWM-driven
controller has run in closed loop on v2 (Stage 0's four debug resets aside, which are not read).**

| invocation | UTC (2026-10-07) | seconds (cap) | outcome | peak PSS (GiB) |
|---|---|---:|---|---:|
| `tests` (CPU, CUDA hidden) | 23:28:11–23:30:54 | 163 (–) | **TESTS-PASS**: 2070 passed, 37 skipped, at `33cea5c`, clean | 3.05 |
| `featurise` (GPU, `gpu_run.sh --wait --min-free-gib 8 --board`, lock held) | 23:30:57–23:31:54 | 58 (3 600) | **FEATURISED**: gate-P 249, contrast-T 250 roots; G-anchor max \|Δ\| 1.53 × 10⁻⁴ on 256 frames (bound 10⁻³); GPU peak 0.49 GiB allocated, 0.55 GiB reserved | 1.54 |
| `rgate` (CPU) | 23:32:55–00:32:48 (2026-10-08) | 3 593 (14 400) | **R-PASS** | 9.73 (RSS 11.14, 11 processes; cap 12) |

- **Guards (`rgate`).** G-quiet: load 1.10 / 0.89 at the start (bar 2.0 / 2.0). In-run G-tests
  passed at `33cea5c` on a clean tree (2070 passed, 37 skipped; 23:32:55–23:35:34). G-hash checked
  TASK-076's 84 pins, TASK-077's 13 pins and this task's 7 own pins, with the protocol document's
  pin `6ceca231…9849`; TASK-077's 21 old feature files (13.84 GB), its Stage O fits and the six
  checkpoints matched; the 14 fresh feature files (3.46 GB) matched the featurise report. The
  revision at the end was still `33cea5c` and the tracked tree stayed clean. No non-finite field.
  G-split: the fits used only the 1 995 old roots (train 1 495, val 250, old gate 250);
  `first_outcome_utc` was written at 23:50:31 UTC, after the fits and before gate-P and
  contrast-T were opened. TASK-076's evidence root supplied the stand-in workers' P-3 checkpoint.
- **The fits** (dual ridge, 5 inner folds, salt 8205; on the 1 995 old roots): R-S at λ_rel 0.001
  on every seed, R-N at 0.001, R-L at 0.1. Their 5-fold cross-fitted medians on the old roots
  (reported): R-S 0.556 / 0.574 / 0.591 cm, R-N 1.325 / 0.993 / 1.291 cm, R-L 2.434 / 2.430 /
  2.434 cm (seeds 66800 / 66801 / 66802). The fits are in `outputs/task080-rgate-1/fits/`; their
  content sha256s are in the report, and Stage D reads R-S and R-N of the primary seed from there.
- **The log** holds the progress lines (three fit lines, seven offline-aim lines, the outcome) and,
  besides them, MuJoCo EGL `__del__` tracebacks from the stand-in workers' interpreter teardown, as
  in TASK-077's logs; they are not errors of the run.

**The gates on gate-P (249 roots; h = 60; τ_commit = 1.0 cm).** Medians in cm with their 95 %
root-bootstrap intervals (10 000 resamples, salt 8206) and 87.5th percentiles:

| | seed 66800 (primary) | seed 66801 | seed 66802 |
|---|---|---|---|
| **R0** ceiling (R8 on the encoded frame at r; no model) | 0.455 [0.403, 0.504], p87.5 0.914 | (same) | (same) |
| e_S (R-S on W's prediction) | 0.547 [0.497, **0.610**], p87.5 1.137 | 0.584 [0.517, **0.657**], p87.5 1.135 | 0.589 [0.536, **0.643**], p87.5 1.159 |
| e_N (R-N on N's prediction) | 1.756 [1.608, 1.921], p87.5 3.269 | 1.574 [1.360, 1.753], p87.5 3.191 | 1.529 [1.381, 1.701], p87.5 3.058 |
| e_L (R-L on W from the mean latent) | 2.329 [**2.125**, 2.542], p87.5 4.165 | 2.446 [**2.214**, 2.629], p87.5 4.132 | 2.402 [**2.174**, 2.621], p87.5 4.186 |
| e_S / e_N (paired by root) | 0.311 [0.275, **0.357**] | 0.371 [0.319, **0.434**] | 0.385 [0.332, **0.438**] |
| e_S − ceiling | +0.092 [0.029, 0.170] | +0.129 [0.054, 0.201] | +0.134 [0.064, 0.215] |
| frozen R8 on W's prediction (reported) | 3.784 [3.452, 4.120] | 3.222 [2.826, 3.546] | 3.168 [2.898, 3.596] |

- **R0** passes (upper bound 0.504 ≤ 1.0). **R1** passes on every seed (upper bounds 0.610, 0.657,
  0.643 ≤ 1.0). **R2** passes on every seed (upper bounds 0.357, 0.434, 0.438 < 1.0). **R3** passes
  on every seed (lower bounds 2.125, 2.214, 2.174 > 1.0): the commands alone, from the mean latent,
  do not read the plate within τ_commit.
- R-S on W's prediction stays 0.09–0.13 cm above the encoded ceiling in median, and each of those
  differences' intervals excludes 0. Its 87.5th percentiles (1.135–1.159 cm) are above τ_commit;
  the fraction of gate-P roots with e_S ≤ 1.5 cm is 0.960 / 0.940 / 0.940 (e_N 0.382 / 0.474 /
  0.482; e_L 0.213 / 0.229 / 0.233), reported at 1.5 τ_commit.
- The frozen R8 read the same predictions at 3.17–3.78 cm, as in the decomposition record (R8 was
  fitted on encoded frames; it is used only for the ceiling).

**The offline aims, A1 and A2** (primary seed 66800, flagged `last_two_triggered`: W-66800 selected
one of its last two checkpoints and every W curve was lowest at its last point, TASK-077 results
§3, caveats 1–2). Each arm's own controller code on gate-P's logged 405 states, with the grid from
the logged p̂ and the stand-in chunks, no fallbacks (0 in every arm); the aim error against the
rule's fixed point; the predicted count maps each root's error through the pooled K ∪ K′ τ curve
(64 / 58 / 60 / 32 / 29 / 7 of 64 at 0 / 0.5 / 1 / 1.5 / 2 / 3 cm) and scales the mean to 64:

| arm | median aim error (cm) [95 %] | p87.5 (cm) | clip-binding | predicted count of 64 | W − arm |
|---|---|---:|---:|---:|---:|
| **W** | 0.637 [0.592, 0.677] | 1.020 | 0.024 | **56.92** | – |
| N | 2.118 [1.962, 2.315] | 3.046 | 0.165 | 27.79 | **+29.12** |
| L-shuf | 2.894 [2.700, 3.064] | 4.056 | 0.378 | 19.67 | **+37.25** |
| L-mean | 2.043 [1.852, 2.232] | 3.155 | 0.088 | 28.95 | **+27.97** |
| L-rand | 6.918 [5.952, 8.156] | 14.061 | 0 | 11.85 | **+45.06** |
| H-rule (hand-coded rule; reported) | 0.495 [0.485, 0.513] | 0.623 | 0.261 | 58.64 | −1.73 |
| H-sysid (reported) | 0.749 [0.672, 0.872] | 1.402 | 0 | 53.93 | +2.99 |

- **A1 passes by a thin margin: W's predicted count is 56.92 of 64 against the bar of 56, a margin
  of 0.92 of 64 (less than one reset).** The report gives the predicted count as a point, without
  an interval; its uncertainty in the report is that of W's aim errors (median 0.637 cm, 95 %
  interval 0.592–0.677 cm; 87.5th percentile 1.020 cm; 33 of 249 roots above τ_commit). For this
  record only, and not as a gate, the count was resampled by root from the report's per-root aim
  errors through the same curve (10 000 resamples, salt 8206): **95 % interval 55.84–57.88 of 64,
  with 4.6 % of the resamples below 56.** That interval does not include the τ curve's own
  uncertainty (64 resets per level; the curve is not monotone, 58 at 0.5 cm and 60 at 1.0 cm).
  The dry run, on TASK-077's val roots with aims from the true plate, had put W at 57.35 (§8.1); the
  fresh gate-P roots read 0.43 lower.
- **A2 passes for every twin by a wide margin** (+27.97 to +45.06 against +7). The closest twin is
  L-mean (+27.97), then N (+29.12).
- **What A1's margin means for Stage S.** At W's predicted rate, 56.92 / 64 ≈ 0.889, G-bar
  (≥ 56 of 64) passes with probability about **0.73** (exact binomial; §10 and R18.18 had 0.78 at
  the dry run's 0.896). The twin tests are not the binding risk at these offline counts. **G-bar is
  the binding risk: at this rate it fails about one time in four.** H-rule's offline count (58.64)
  sits above W's by 1.73 of 64, within G-NI's δ = 8/64 as a point; G-NI is a closed-loop test on
  cohort S, not an offline one.

**Reported only.**
- **Contrast-T** (250 roots, aims from the true plate): ceiling 0.390 [0.356, 0.432]; e_S 0.512
  [0.461, 0.568], 0.573 [0.527, 0.630], 0.542 [0.480, 0.607]; e_N 1.681, 1.452, 1.640; e_L 2.425,
  2.435, 2.342; e_S / e_N 0.305 [0.267, 0.351], 0.394 [0.340, 0.458], 0.331 [0.291, 0.386]. Every
  R-gate form would pass on contrast-T as well. Seed by seed against the decomposition record's
  val medians (0.578 / 0.564 / 0.556 cm), contrast-T's e_S medians are 0.066 lower (66800), 0.009
  higher (66801) and 0.014 lower (66802); as ranges, 0.512–0.573 cm is not above 0.556–0.578 cm.
  So the val selection shows no consistent cost at the median on fresh roots (different root sets;
  a comparison of medians only). Reworded in R18.27 after #157's review: the earlier sentence held
  for the ranges, not seed by seed.
- **The corpus-aim confound** (§5.3 point 2): median e_S on gate-P minus contrast-T is +0.035
  [−0.043, +0.121], +0.011 [−0.075, +0.082] and +0.047 [−0.039, +0.134] cm by seed. Every interval
  includes 0; the split can detect only an effect of about p̂'s error (R-plate's error at 405 on
  the fresh roots: 0.160 [0.146, 0.170] cm on gate-P, 0.154 [0.138, 0.169] cm on contrast-T), so
  this bounds a small confound only, as §5.3 point 4 says.
- **The echo slope** (§5.3 point 4; W, primary seed, 249 roots): the per-root 2 × 2 slope of W's
  predicted plate against the aim over the feasible grid has median matrix
  [[−0.126, 0.022], [−0.013, −0.410]], half-trace −0.273 [−0.283, −0.266]; its Frobenius distance
  to κI (κ = −0.5) is 0.398 [0.391, 0.405] and to I (the echo) 1.815 [1.804, 1.829]. **The
  predicted plate does not echo the aim**: its slope is negative, like the rule's κ, about a quarter
  of κ in the first axis (−0.126) and close to κ in the second (−0.410).
- **G1–G4 at h = 60 on gate-P** (TASK-077 §8.2's definitions and bars, reported, not gated): all
  pass on all three seeds. G1's effective-rank ratio 0.284 / 0.278 / 0.281 (no collapsed
  dimension); G2's ratio 0.140 / 0.144 / 0.142; G3's 0.346 / 0.370 / 0.379; G4's wrong-command
  ratio 3.99 / 3.78 / 3.83 and zero-command ratio 6.42 / 15.74 / 6.13.
- **The learning curve on gate-P** (R-S at 25 / 50 / 75 / 100 % of the 1 995 fit roots): 0.760 →
  0.595 → 0.548 → 0.547 cm (66800), 0.791 → 0.673 → 0.619 → 0.584 (66801), 0.823 → 0.691 → 0.618
  → 0.589 (66802). It is flat at the end for 66800 and still falling for 66801 and 66802; R-N falls
  slowly (1.53–1.79 cm) and R-L is flat (2.32–2.45 cm).
- p̂ recomputed from the stored CUDA tokens equals the collector's logged p̂ to within
  3.1 × 10⁻⁵ cm.
- Neither half had a stand-in-infeasible root.

**Row: R-PASS** (first match; R0–R3, A1 and A2 all pass; no V). The clause does not fire, nothing
is closed, and TASK-077's row is unchanged. **Stage D may get its GO** (§8 step 6).

**Evidence.**
- Reports (git-ignored, in the worktree): `outputs/task080-tests-1/report.json` (sha256
  `acbe897d…7464`), `outputs/task080-featurise-1/report.json` (sha256 `cda16651…83f2`) and
  `outputs/task080-rgate-1/report.json` (sha256
  **`bedb896692a9718ac598fa93c5d7dcc199090bf4cfa13d14fa0bd0fc53848ea7`**), with their logs
  (`task080-rgate-1.log` sha256 `836bdd4a…b98c`) and stdout captures (all three empty, 0 bytes, as
  `--log` redirects the process after it opens).
- A copy of the three reports, logs and stdout captures and of the nine fits is in
  `~/develop/emai/evidence/task080-stager/` (18 files, 7.0 MB), with the manifest
  `_checksums/task080-stager.sha256` (sha256 `ef2af17d…35d0`). The fresh feature files (14 files,
  3.46 GB) are not copied, as TASK-077 Stage O's were not; they stay in the `task080-stager`
  worktree, and the featurise report's `files_sha256` identifies them.
- The `task080-stager` worktree is not edited and must not be removed before TASK-080's results
  PR: Stage D and Stage S read its rgate report and fits.

### 8.5 Stage D's plan (R18.26)

Stage D runs once, on its own reported GO at this record's merge commit, from a fresh clean
worktree of that commit (`scripts/new_worktree.sh /home/huhn/develop/emai/worktrees/task080-staged
--run --from <merge sha>`). It is one invocation, from the worktree root, on the CPU (no GPU lock;
the stage creates no CUDA context):

- **`closed --cohort D`**: the 16 development resets **65100–65115**, arms **W, N, L-shuf,
  L-mean and H-final(commit)** (§6), W and N reading R-S and R-N of the primary seed 66800 from
  Stage R's fits (each checked against its content sha256 in the R-PASS report), with TASK-077's
  six checkpoint reports, featurisation, Stage O fits and TASK-076's evidence root (G-repro). The
  runner requires `--stage-r` to be a non-debug R-PASS report and the corpus sha256 to equal the
  one Stage R read (G-split), and it runs its own in-run G-tests and G-quiet first, so no separate
  `tests` record is needed (that is only for the GPU stage). Nothing is refitted on D.
- **Rows (§8 step 6, §9.3):** **L-DEV-STOP** (escalate, no clause) if W < 12/16, or
  H-final(commit) < 14/16, or W − max(N, L-shuf, L-mean) < +3/16; otherwise **D-PASS**, after
  which Stage S may get its own GO. H-final(commit) is privileged (it looks ahead in cloned
  simulator state) and is a ceiling, not a learned arm.
- **Caps and resources (§11):** Stage D 7 200 s and 300 s per attempt (Stage 0 scaled D at
  ≤ 400 s from debug medians of 7–16 s per attempt on 4 world-model workers); G-memory 12 GiB
  process-tree PSS (debug D peaked at 9.3 GiB); MemAvailable at start ≥ the ceiling plus its
  headroom; G-quiet load ≤ 2.0 / 2.0 at the start; G-disk ≥ 10 GiB free.
- **What D is and is not.** D is a development cohort of 16 resets that gates only the GO of
  Stage S; it is not a gated claim. It will be the first closed-loop run of a LeWM-driven
  controller on v2 on a cohort whose counts are read (Stage 0's debug closed loops ran on four
  debug resets and are not read): whatever its row, the clause of R7's canonical sentence that
  says no LeWM-driven controller has run in closed loop on v2 will no longer be literally true once
  D has run. R18.12 keeps R7 unedited by this task except through a reviewed ruling after an
  L-PASS, so Stage D's record must state D's counts beside R7 as development results and propose
  the factual wording that R7 needs as its own reviewed ruling.

If the invocation ends in anything other than a §9.3 row, nothing further is launched and the case
is ruled under §11 first. The exact command is in the GO.

### 8.6 Stage D's result: D-PASS (R18.27)

Stage D ran once, on the reviewer's reported GO (#157,
<https://github.com/RaaSaaR-org/open-embodied-jepa/pull/157#issuecomment-6050425196>, posted
01:41:24 UTC), at `db34adc`, the record's merge commit, on a clean tree with STATUS FROZEN and the
frozen sha `0fc095dc…be064`, in the worktree `/home/huhn/develop/emai/worktrees/task080-staged`,
with the command the GO named, once. **D is a non-gating development cohort of 16 resets, one run,
one model seed, in simulation only. It gates only the GO of Stage S and is not the gated result.**
It is the first LeWM-driven closed loop on v2 whose counts are read (R18.28 below; Stage 0's debug
closed loops ran on debug seeds and are not read). In every arm, P-3's learned pick and e9's
scripted place are the same; the arms differ only in the single place aim committed at step 405,
under the declared simulation-only condition C1-M.

| | UTC (2026-10-08) | seconds |
|---|---|---:|
| start; G-quiet passed (load 0.15 / 0.15, bar 2.0 / 2.0) | 01:42:36 | – |
| in-run G-tests: 2070 passed, 37 skipped, at `db34adc`, clean | 01:42:36–01:45:21 | 163 |
| G-repro (8 of 8 checks; 170 train roots decoded, no test split) | – | 42 |
| `first_outcome_utc` | 01:46:05 | – |
| arms, in order (log): W, N, L-shuf, L-mean, H-final(commit) done at | 01:47:03, 01:47:30, 01:48:27, 01:49:22, 01:49:54 | – |
| end, report written: **D-PASS** | 01:49:55 | **439** (cap 7 200) |

- **Guards.** G-hash checked TASK-076's 84 pins, TASK-077's 13 pins and this task's 7 own pins,
  with the protocol document's pin `96298e16…0ecc`; the six checkpoints matched their sha256s; the
  runner accepted the Stage R report the GO named (`bedb8966…48ea7`) as a non-debug R-PASS report
  that read the same sealed corpus (`deebd83d…7c4e`; G-split); R-S and R-N of seed 66800 loaded
  against their content sha256s. The revision at the end was still `db34adc` and the tracked tree
  stayed clean. No non-finite field. G-memory:
  peak process-tree PSS 9.70 GiB (RSS 11.17 GiB, 9 processes; cap 12), MemAvailable 24.6 GiB at
  the start. G-disk: at least 64.3 GiB free throughout. Threads as pinned (MKL and OMP 6, OpenBLAS
  16), 4 world-model workers. No render disagreement between arms and no render retry. Every
  attempt ran its 725 steps to `policy_complete`; the post-pick plate move was applied on every
  attempt, under the reactive rule κ = −0.5, L = 2; no contact with the plate before s1 was
  recorded on any attempt. Every attempt passed the privileged
  read check, with 0 task-truth reads in any controller. No determinism re-run is part of D
  (`"not evaluated"`).
- **The GO's G-quiet note.** The frozen block's G-quiet text says a busy machine "makes the stage
  V before any render"; the code instead polls for up to 4 h before raising. The machine was quiet
  at launch (load 0.15 / 0.15), so the stage did not wait. This is a wording mismatch only; the
  frozen block is not edited, and Stage S's GO repeats the launch-time load check.

**Counts** (counted success: the apple at rest on the plate under `apple_at_rest_v0` after a
latched grasp and a latched place, T71-R1/R2). Exact binomial 95 % intervals on 16 resets are
wide:

| arm | count | 95 % (exact) | W − arm | resets W only / arm only |
|---|---:|---|---:|---|
| **W** (R-S, seed 66800, flagged `last_two_triggered`) | **16/16** | 0.794–1.000 | – | – |
| N (action-blind; R-N) | 8/16 | 0.247–0.753 | +8 | 8 / 0 |
| L-shuf (scene-blind; R-S) | 5/16 | 0.110–0.587 | +11 | 11 / 0 |
| L-mean (scene-blind; R-S) | 9/16 | 0.299–0.802 | +7 | 7 / 0 |
| H-final(commit) (**privileged** look-ahead ceiling, not learned) | 16/16 | 0.794–1.000 | 0 | 0 / 0 |

**Rows (§8 step 6, §9.3).** W 16 ≥ 12; H-final(commit) 16 ≥ 14; W − max(N, L-shuf, L-mean) = 16 −
9 = **+7 ≥ +3**. No stop holds: **D-PASS**, `clause_fires` false. Every twin's success is also a
W success on the same reset (no reset where a twin succeeded and W failed). D runs no McNemar
test; those discordant counts are reported, not tested, and D's bars are not S's.

**The committed aims.** The aim error is the commit target's distance from the rule's fixed point
g\* = (p − κh)/(1 − κ) at 405 (the quantity Stage R's offline aims mapped through the τ curve);
the landing miss is its distance from where the plate actually was at s1 = 525:

| arm | median aim error (cm) | p87.5 | max | aims > τ_commit | median landing miss (cm) | p87.5 | clip-binding | fallbacks | refused before 405 | seconds per attempt (median / max) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| W | 0.546 | 0.782 | 1.108 | 1 of 16 | 0.475 | 1.049 | 0 | 0 | 0 | 13.85 / 16.00 |
| N | 2.313 | 2.779 | 3.679 | 14 of 16 | 3.234 | 3.930 | 0 | 0 | 0 | 6.77 / 7.06 |
| L-shuf | 2.748 | 3.862 | 4.237 | 15 of 16 | 4.213 | 5.663 | 0.25 | 0 | 0 | 14.02 / 14.55 |
| L-mean | 1.964 | 3.460 | 3.633 | 13 of 16 | 2.596 | 4.616 | 0 | 0 | 0 | 13.92 / 14.35 |
| H-final(commit) | 0.456 | 0.505 | 0.511 | 0 of 16 | 0.085 | 0.111 | 0 | 0 | 0 | 7.74 / 8.31 |

- W's one aim above τ_commit (65112, 1.108 cm) still succeeded. W's refinement converged on 13 of
  16 resets and stopped at the 10-refinement cap on 3 (65104, 65105, 65111), all of which
  succeeded. In W, N and both scene-blind twins all 147 grid candidates were feasible on every
  reset. W's decision took 9.7 s in median (148–157 roll-outs per decision).
- H-final(commit)'s aims sit 0.41–0.51 cm from the analytic fixed point but land within 0.12 cm
  of the plate at s1: the fixed point g\* is the rule's idealisation, and the look-ahead aims at the
  plate the simulator actually produces. W's landing misses (median 0.475 cm) are close to its
  aim errors.
- The twins' successes come with aim errors up to 3.27 cm (L-shuf on 65104) and 2.56 cm (L-mean on
  65111): the success tolerance is looser than τ_commit at some resets, as the τ curve's 32/64 at
  1.5 cm and 29/64 at 2 cm already show.
- L-shuf on 65111 ended at rest on the plate without a latched place, so it is not counted. On
  four L-shuf resets (65103, 65108, 65110, 65113) the apple ended 1.16–1.72 m from the plate.

**Per reset** (S = counted success, – = not; the number is the aim error in cm):

| reset | W | N | L-shuf | L-mean | H-final(commit) |
|---|---|---|---|---|---|
| 65100 | S 0.53 | – 2.50 | – 0.89 | S 1.32 | S 0.47 |
| 65101 | S 0.54 | – 2.34 | – 2.93 | S 0.92 | S 0.42 |
| 65102 | S 0.77 | S 0.48 | – 2.56 | S 1.15 | S 0.51 |
| 65103 | S 0.25 | S 1.77 | – 3.89 | S 2.51 | S 0.50 |
| 65104 | S 0.74 | – 2.98 | S 3.27 | – 3.63 | S 0.45 |
| 65105 | S 0.77 | – 2.28 | – 2.01 | – 2.25 | S 0.43 |
| 65106 | S 0.55 | S 1.16 | – 1.25 | S 1.58 | S 0.42 |
| 65107 | S 0.21 | S 1.46 | S 2.48 | S 1.67 | S 0.41 |
| 65108 | S 0.60 | S 2.05 | – 2.93 | S 1.18 | S 0.51 |
| 65109 | S 0.56 | – 2.47 | S 4.24 | – 3.28 | S 0.46 |
| 65110 | S 0.49 | S 2.51 | – 2.48 | – 0.78 | S 0.51 |
| 65111 | S 0.86 | – 2.50 | – 3.03 | S 2.56 | S 0.50 |
| 65112 | S 1.11 | S 1.15 | – 3.86 | – 3.45 | S 0.45 |
| 65113 | S 0.44 | S 0.59 | – 3.38 | S 0.77 | S 0.51 |
| 65114 | S 0.23 | – 3.68 | S 2.48 | – 3.54 | S 0.45 |
| 65115 | S 0.38 | – 2.75 | S 1.50 | – 2.69 | S 0.45 |

**What D does and does not show.**
- It does not test G-bar, G-NI or any McNemar test: those are Stage S's, on 64 fresh resets. 16/16
  on 16 resets is compatible with any true rate from about 0.79 upward; Stage R's offline prediction
  for W (56.92 of 64, about 0.889) is inside that range, so D neither confirms nor refutes it, and
  G-bar's power at that predicted rate is still about 0.73 (§8.4).
- The non-inferiority comparators, H-rule (the hand-coded rule) and H-sysid, and L-rand did not run
  in D. Whether W is as good as a rule that knows the law is S's G-NI question; offline, H-rule
  read 1.73 of 64 above W (§8.4).
- The ceiling H-final(commit) also scored 16/16, so D does not separate W from the privileged
  ceiling either.
- Everything is one model seed (66800, flagged `last_two_triggered`: W-66800 selected one of its
  last two checkpoints), one camera, one encoder, one condition imposed by the simulator (C1-M), in
  MuJoCo only.

**Beside R7** (§8.5): the canonical sentence's clause "No LeWM-driven controller has run in
closed loop on v2 yet; LeWM's only closed-loop Apple→Plate runs are on v1, with 0 successes" is
no longer true. R18.28 ([DECISIONS.md](../DECISIONS.md), decision 2026-10-08 (c)) replaces it
with D's counts, labelled as a development result, as its own ruling and as a named exception to
R18.12. The quote at the top of this document is R7 as it stood when the protocol was frozen.

**Row: D-PASS.** The clause does not fire, nothing is closed, and TASK-077's row is unchanged.
**Stage S may get its GO** (§8 step 7, §8.7).

**Evidence.**
- Report (git-ignored, in the worktree): `outputs/task080-d-1/report.json`, sha256
  **`1602447a08523f49d3924f6b38d10275a417d1ccd4a90bc8cd70d22ef79a0611`**; log
  `outputs/task080-d-1.log` (sha256 `13af1196…0662`; the five arm lines and the outcome, nothing
  else); stdout capture `outputs/task080-d-1.stdout` (empty, 0 bytes, as `--log` redirects the
  process after it opens).
- A copy of the three is in `~/develop/emai/evidence/task080-staged/` with the manifest
  `_checksums/task080-staged.sha256` (sha256 `dabbfac8…6f75`), verified against the source and
  the copy.
- The `task080-staged` worktree is not edited and must not be removed before TASK-080's results
  PR: Stage S reads its D-PASS report (`--stage-d`).

### 8.7 Stage S's plan (R18.29)

Stage S runs once, on its own reported GO at this record's merge commit, from a fresh clean
worktree of that commit (`scripts/new_worktree.sh /home/huhn/develop/emai/worktrees/task080-stages
--run --from <merge sha>`). It is one invocation, from the worktree root, on the CPU (no GPU lock;
the stage creates no CUDA context):

- **`closed --cohort S`**: the 64 gated resets **65200–65263**, every arm of §6 once per reset,
  paired: **W, N, L-shuf, L-mean, L-rand, H-rule, H-sysid, H-final(commit), H-read and H-now**
  (H-read and H-now reported only; H-final, H-read and H-now are privileged), then TASK-077's
  determinism re-run of W on S's first four resets (reading 0.1 cm, commit target 0.6 cm, success
  identical; a difference is a G-determinism guard error). The inputs are D's: the six checkpoint
  reports, TASK-077's featurisation and Stage O fits, TASK-076's evidence root, the sealed corpus at
  `deebd83d…7c4e` and the Stage R report, **plus `--stage-d` set to the D-PASS report above**: the
  runner refuses any `--stage-d` that is not a non-debug D-PASS report, or that read another
  corpus. The GO also checks that report's sha256 (`1602447a…0611`) before launch, which the runner
  does not.
- **Rows:** §9.5, first match (V, S-VOID-CEILING, L-NO-GAIN, L-INFERIOR, L-PASS, L-TWIN-NEAR,
  L-NEAR, L-BAR). Only L-NO-GAIN and L-INFERIOR fire the clause (§12). L-PASS is the primary claim,
  "LeWM-driven closed-loop success", and only it lets R7 state a gated LeWM result.
- **Caps and resources (§11):** Stage S 21 600 s and 300 s per attempt (Stage 0 scaled S at
  ≤ 1 600 s); G-memory 12 GiB process-tree PSS (debug S peaked at 9.7 GiB, D at 9.70 GiB);
  MemAvailable at start ≥ 16 GiB; G-quiet load ≤ 2.0 / 2.0 at the start; G-disk ≥ 10 GiB free.
- **Expected time.** From D's per-attempt medians and debug S's for the five arms D did not run
  (about 84 s per reset over the ten arms), 64 resets on 4 workers take about 23 min, plus the
  determinism re-run, G-tests (about 165 s) and G-repro (about 42 s): **about 25–35 min**.
- **Launch-time checks** (the GO lists them): the worktree's HEAD is the merge commit with a clean
  tracked tree and no `outputs/task080-s*` anywhere; the 1- and 5-minute load averages ≤ 2.0 and no
  other heavy CPU job running or about to start (`hz.Clock` starts before G-quiet's wait, so a long
  wait counts against the cap); MemAvailable ≥ 16 GiB; ≥ 10 GiB free; the Stage R report
  `bedb8966…48ea7` and its fits unchanged; the D report `1602447a…0611`; no seed of 65200–65263
  simulated before.

If the invocation ends in anything other than a §9.5 row, nothing further is launched and the case
is ruled under §11 first. The exact command is in the GO. After S, the results PR reports every arm
with privileged arms labelled, and an independent reviewer checks every restated number (§8
step 8).

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
| **R-COMMAND-KEYED** | R3 fails on any seed | escalate, no clause: the commands alone (with no scene latent) read the plate within τ_commit, so W's prediction would add nothing measurable over them (§5.3, point 3) |
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

**At the freeze (R18.22).** K0′ left τ_commit at 1.0 cm, and the dry run re-mapped through the pooled
curve (§8.1) puts W at 57.35 of 64 and N at 28.36 (development, optimistic; primary seed 66800,
flagged `last_two_triggered`; predictions, not closed-loop counts). The twin tests are not the
binding risk (power ≥ 0.999 at those rates in every coupling). **The binding risk is G-bar**
(R18.18): at W's predicted rate of about 0.896 it passes with probability about 0.78; R18.18 found that a
larger cohort at the same fraction does not change this materially (0.72 at 64 resets, 0.73 at
96 and 0.75 at 128, at a rate of 0.889). A1 (W ≥ 56/64 offline) is the gate
most likely to stop Stage R if the fresh roots read even slightly worse than val.

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
  before the freeze. Stage 0's caps (R18.15), kept at the freeze: K0′ 7 200 s; Stage C′ 7 200 s; Stage R
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
  (K0′ left it at 1.0 cm, §8.1.)
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

**Rulings at Stage 0 (R18.16–R18.21; [record](apple_lewm_c1m_v2_pred_readout_stage0.md) §7),
informed by R18.13's dry run** (development and optimistic, setting no bar; primary seed 66800,
flagged `last_two_triggered`; W's offline predicted count 56.88/64, N 25.94, W − N +30.9; offline
predicted counts are not closed-loop counts): 1 — the carried rule, 66800 (R18.16); 2 — all 1 995 old roots (R18.17); 3 — 64 resets and
+7/64 unchanged, since W − N is far above +10/64; the binding risk is G-bar, which a larger cohort
does not fix (R18.18); 4 — the pooled K ∪ K′ rule (R18.19); 5 — reported, not gated (R18.20);
6 — a recommendation to the owner that a TASK-080 L-PASS counts as the equivalent row, not ruled
(R18.21).
