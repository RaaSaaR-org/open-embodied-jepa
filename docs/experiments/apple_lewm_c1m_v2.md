# Apple→Plate LeWM committed aim under C1-M: LeWM chooses the single place aim at 405 from the encoded current frame (TASK-077)

**STATUS: FROZEN** (after K0-PASS; R17.25–R17.29, decided by Claude under owner delegation).
The frozen block is `src/embodied_jepa/lewm_c1m_v2.py`; its sha256
**`f28e5e2cd23d110f40ff043c7308e0bb9b3b71f46a2a4b6940bc536cc5e3548d`** is pinned in
`tests/test_lewm_c1m_v2.py` and in the manifest (`benchmarks/manifests/apple-lewm-c1m-v2.json`),
with the manifest's file pins and this document's sha256. **The freeze takes effect when it is
merged on an independent reviewer's reported APPROVE.** Each later stage still needs its own
reported GO (§7).

History. Revision 2 of the draft followed the independent review of #142 at `26894ca` (REQUEST
CHANGES; R17.15–R17.18). Stage 0 (R17.19) added the stage code, the manifest, the tests, the
simulations of §8.4, the storage fix of §12, the scale probe and smokes on the debug range
66900–66999 only; its record is [apple_lewm_c1m_v2_stage0.md](apple_lewm_c1m_v2_stage0.md). The
independent review of #143 (REQUEST CHANGES at `0dea22b`) led to R17.20–R17.24, and its APPROVE at
`306fbdc` gave K0 its GO. **K0 ran once at `306fbdc` and ended K0-PASS** (§7.1): τ_commit =
1.0 cm, N_K(0) = 32/32, r_K = 460. Its values are in the frozen block (`K0_MEASURED`). Two of
them sit close to their bars, and this is disclosed in §7.1. K, a cohort of 32 resets, is the only
cohort simulated. No seed of D, S or the corpus has been simulated.
**Stage C ran once at `862d63c` and ended V on G-memory** during seed preparation, before any
root was collected. **Erratum 2026-10-05** (§7.2; R17.30–R17.34) records the cause and the fix,
re-pins the changed files and this document, and leaves the frozen block and its sha256
unchanged. **Stage C's one repeat ran at `4f30fbb` and ended CORPUS-SEALED** (§7.3, R17.35):
1 995 of 2 000 roots, sealed manifest sha256 `ad8974b2…43fb`. **Erratum 2026-10-05 (b)** (§7.4;
R17.36–R17.39) corrects the bit-identity claim of §7.2, records the GPU featurisation scale probe
that Stage O's GO needed, gates the determinism re-run's commit target at a derived 0.6 cm, and
makes G-anchor check the frozen 256 frames. It re-pins the changed files and this document; the frozen block and its sha256 are unchanged. Its 0.6 cm gate (R17.38) is a post-freeze **amendment**, a tightening, not an erratum (R17.43).
**Stage O ran once at `8721516` and ended O-PASS** (§7.5, R17.40): c_plate 0.404 cm
[0.385, 0.417] against τ_commit = 1.0 cm; two command and report anomalies are disclosed
(R17.41). **Before Stage T's GO** (§7.6, R17.41–R17.45): the runner records peak GPU memory and
finds Stage O's fits from another worktree; W's CPU roll-out measured bit-identical across
processes (debug and random-init models, synthetic commands, 4 debug roots); Stage T's scale
re-probed at 50 kept states; the eight jobs run from a committed driver script that checks each
step's outcome (R17.45).
**Stage T started at `215fcce`** (§7.7, R17.46–R17.48): the calibration jobs, the plan
(T-PLANNED, U = 95 000) and W-66800 completed. N-66800 ended **V** when the operator stopped it
at the owner's request, to pause Stage T and free the shared GPU. That V voids N-66800 only. Its
one repeat, its last allowed attempt, needs §7.7's record of the cause and its prevention, merged,
and its own GO. Five jobs remain; nothing has been launched since the pause.

- **Admitted by:** the C1-M feasibility record's row **M-PROCEED**
  ([apple_lewm_next_v2_c1m_feasibility.md](apple_lewm_next_v2_c1m_feasibility.md), R16, #141).
  That row admits **only the drafting of a preregistration** (R16.13). This document is that
  preregistration, frozen after K0-PASS (R17.25).
- **Rulings:** R17.1–R17.18 in [DECISIONS.md](../DECISIONS.md), decision 2026-10-05 (b). Each is
  **decided by Claude under owner delegation (2026-09-30)**. They build on R9 (the design note,
  [apple_lewm_next_v2_design.md](apple_lewm_next_v2_design.md); the claim split R9.8 and the
  random-choice test R9.9), R13 (TASK-076's results), R14 (the C1 record), R15 (the C1-M
  direction, [apple_lewm_next_v2_direction.md](apple_lewm_next_v2_direction.md)) and R16.
- **Task card:** `.mc/tasks/todo/TASK-077-lewm-committed-aim-under-c1m-lewm-chooses-the-single-place-aim-from-the-encoded-frame.md`.
- **Templates:** TASK-066 for the model, its gates and its calibration
  ([apple_token_dynamics_v1.md](apple_token_dynamics_v1.md)); TASK-074 for the budget lessons
  ([apple_lewm_planner_v2.md](apple_lewm_planner_v2.md) Addendum A2,
  [apple_lewm_planner_v2_results.md](apple_lewm_planner_v2_results.md)); TASK-076 for the
  protocol form, the freeze and the staged GO flow ([apple_plate_twin_v2.md](apple_plate_twin_v2.md)).

The canonical status sentence (DECISIONS 2026-10-02, R7), verbatim. **This protocol does not
change it** (R17.14):

> Learned Apple→Plate on the frozen v1 MVP benchmark (TASK-020) is 0/150 per backend
> (`native_jepa` and LeWM). On `apple-to-plate-v2`, the behaviour-cloning/DAgger policy P-3 (an
> MLP on a frozen DINOv2 readout, trained on demonstrations from the privileged scripted expert
> e9; not a world model) scored 40/40 counted successes on the held-out cohort C against 39/40
> for its random-init encoder control R-3 (one run, one training seed per arm, 40 resets, one
> camera at 112 px onboard, a narrow reset distribution), so TASK-072 M2 is M2-FAIL on G3
> (encoder pretraining contributed nothing measurable), and cohort C is no longer held out. No
> LeWM-driven controller has run in closed loop on v2 yet; LeWM's only closed-loop Apple→Plate
> runs are on v1, with 0 successes. Scripted-expert, privileged-ceiling, oracle and GR00T
> successes are not project-learned results. (LeWM also ran in closed loop on
> the TASK-014 development reach pilot, a reach task, not Apple→Plate: its v2 target-space
> selector reached 1/5 goals, with intervals overlapping the 0/5 controls.)

---

## 0. Where LeWM sits, stated first

LeWM makes **one decision**: the place aim committed at step 405. W, a TASK-066-family LeWM token
predictor, encodes the current onboard 112 px frame at 405 with the frozen DINOv2. It then rolls
each candidate aim's 60 commands forward to the read step r = 465. A plate readout reads the
predicted plate position, and W picks the aim whose predicted plate is nearest to where the apple
would land. Before 405, P-3 (behaviour cloning, not a world model) picks the apple. After 405,
e9's scripted place primitive executes the committed aim, with no re-aim.

The claim, if any, is named by that decision: **"LeWM-driven aim selection at 405"** under the
C1-M condition. It is not "a LeWM policy". Every other arm is either a trained twin of W (N,
L-shuf, L-mean), a random choice (L-rand), a hand-written non-world-model arm (H-rule, H-sysid) or
a privileged reference (H-final(commit), H-read, H-now). Nothing privileged reaches W or its twins
at run time (§5).

## 1. The question and the two claims (R9.8, R9.9)

**Question.** Under C1-M (v2's reset, P-3's pick, a post-pick plate move at step 300 uniform over
a 4 cm disc, cell A's reactive-plate rule from 405, and a single aim committed at 405), does
W's choice reach the primary claim on a fresh gated cohort?

**Primary claim, "LeWM-driven closed-loop success"** (R9.8; all of the following, on cohort S):
1. W's predictor makes the decision from the encoded current frame, with no privileged read at
   run time (§5.1; checked by the `task_truth_in_controller` count, as in C1-M).
2. **The bar:** W(S) ≥ 56/64 (§8.4, G-bar).
3. **Non-inferiority:** W is non-inferior to the best non-world-model arm (the better of H-rule
   and H-sysid on S) within δ = 8/64 (§8.4, G-NI). δ is labelled as R15 labels it: **the design
   note's proposal, an allocation, not calibrated** (R17.8).
4. **W beats the action-blind twin N, both scene-blind twins (L-shuf and L-mean) and a random
   choice (L-rand)**, each by an exact one-sided McNemar test at p < 0.01 (§8.4, G-N, G-shuf,
   G-mean, G-rand).

Items 2 and 3 must both hold (R9.8). Item 4 is what shows that the prediction is used.

**Secondary claim, "LeWM needed"**: W detectably better than the better of H-rule and H-sysid
(exact one-sided McNemar p < 0.01). **Reported only, never a gate, and not expected**: H-rule and
H-sysid scored 30/32 each in the C1-M record's M-F6 (development, one run), and both capture the
rule (R15 §5.7). A primary pass never supports a "LeWM needed" statement.

**What a pass would show:** "LeWM-driven aim selection at 405, followed by e9's place, reaches the
calibrated bar, is non-inferior to the best hand-written arm within the allocated δ and beats its
action-blind, scene-blind and random twins, under a declared reactive-plate rule with a post-pick
plate move on v2; LeWM ranks aims inside a box built from a non-LeWM single-frame readout (R-plate)
and the proprioceptive palm, and its refinement is clipped to that box". That would be the first LeWM-driven closed-loop success on v2. It would change
R7's sentence only through its own reviewed ruling (R15 §5.7).

**What it would not show:** that LeWM is needed; a LeWM policy (P-3 picks and e9 places);
anything about v1's 0/150, Arena or the real G1; that pretraining matters (no random-init floor is
run for W); anything about 4 × 4 latents or other conditions.

## 2. What is carried, with its source

### 2.1 The condition (C1-M, fixed by R15 and the C1-M record)

| quantity | value | source |
|---|---|---|
| task, reset, success | `apple-to-plate-v2`; `wm_critic_v2.wide_reset_values(seed)` (plate (0.49, −0.09) ± 2 cm, apple (0.34, −0.18) ± 3 cm, uniform per axis); counted success = at rest after a latched grasp and place (T71-R1/R2) | v2, TASK-071 |
| pick | P-3 (`checkpoints/task072-first-policy-v2-linux/run-1/P-3.pt`, sha256 `7988162d…60be8`), with G-repro's post-look estimates | TASK-072, TASK-076 |
| the move | at the observation of step 300, `plate_shift.move_plate` by an offset ρ·√u·(cos 2πv, sin 2πv), ρ\* = **4 cm (disc)**, u, v from TASK-077's salt 8101 (§6); off-table re-draw rule and blocked move as R16.3 | R15.5, M-F1 |
| the rule | cell A's `CellMotion`: κ = −0.5, L = 2, s0 = 405, s1 = 525 | R9.2, R14.2 |
| the commitment | one aim committed at c = 405 (`CommitController`); e9's place primitive (`RestingPlaceExpert(release_pitch_rad=0.45, release_dx=0.015)`, `place_planner.PlacePrimitive`) executes it with no re-aim | R9.2 |
| the box and grid | g = p̂ + a·(h − p̂) + b·n, a ∈ [a_lo, 0.5] at steps of 0.05, b ∈ [−3, +3] cm at steps of 1 cm: **147 candidates**; a_lo = **−0.5** | design note §4.1, M-F2 |
| the read step | **r = 465**, so the roll-out horizon is **h = r − 405 = 60** steps | M-F2, R16.13 |
| τ_commit | 1.0 cm in the record; **re-measured in K0** on fresh seeds, and K0's value is the one used | M-F4, R16.13 |
| the ceiling bar | H-final(commit) ≥ 30/32, re-measured in K0 | R9.6 |

**The history check (R8.9), stated from the record.** At 405 the median palm speed was
0.0031 cm per step and m2 had median 0.0025 cm (maximum 0.019 cm) (M-F2). With the 8 × 8
readout's median 0.478 cm at r (M-F5a), c_plate + m2\* is far below τ_commit = 1.0 cm, so a
history-one predictor qualifies. K0 re-measures the palm speed. If its median exceeds 0.5 cm per
step (the design note's rule), K0 stops with CAL-ESCALATE, because a history-two predictor would be
a different model from the one declared here.

### 2.2 Code reused unchanged (the Stage-0 PR adds new modules only)

TASK-076's worker, hook, `AimController`, `LookaheadAim` (H-final), `RuleAim`, `PlateBrancher`,
harness, G-repro and pins; C1's `C1Motion`, `CommitController`, `GeometricAim`, `RuleCommit`,
`SysidAim`, `clip_to_box` and the box; C1-M's move hook, attempt, H-read and planted aim
(`lewm_next_c1m_runtime`); `place_planner.primitive_chunks` (the kinematic stand-in's command
chunk per candidate, here with `horizon = 60`); `models.frozen_tokens.frozen_token_model` with
`token_grid: 8` (already supported; no model file changes, so no implementation hash of an earlier
checkpoint changes); `run_tools` (`check_budget`, `select_checkpoint`, `last_two_triggered`,
`gpu_guard`, `install_guards`, `assert_local_import`, `scale_probe`, `MemoryWatch`). No
hash-pinned TASK-073–076 file and no C1 or C1-M file is edited. The training loop is new shared
code in `src/embodied_jepa/` (the new runner may not load `scripts/train_apple_token_dynamics.py`;
`tests/test_no_runner_imports.py`).

## 3. Caveats carried from the C1-M record and its review (#141), stated before any number

These come from the independent review of #141
(<https://github.com/RaaSaaR-org/open-embodied-jepa/pull/141#issuecomment-5986034811>) and the
record's §2.10. They bind this protocol's wording and design.

1. **60/64 is not a margin, and no bar here is derived from it.** The 8 × 8 oracle-dynamics arm
   H-read scored 60/64 against a 64/64 ceiling: +4/64, exactly at its 4/64 allowance, with an
   interval of [+1, +8] that reaches the whole proposed δ. One more failure would have made the
   row M-NO-BAR-DATA. ρ\* = 4 cm was set by one reset (5 cm: 29/32; effectively 29/31, because
   seed 63031 failed in P-3's pick at every radius). H-read's fixed point converged on only 21/64
   attempts, so its count includes the stopping rule's choice of the last iterate. K0 and Stage O
   re-measure everything they use on fresh seeds and a new corpus; nothing cites 60/64 as room.
2. **4 × 4 on a larger corpus is untested.** 8 × 8 was chosen because 4 × 4 missed H-read's
   allowance (+6/64) while its own readout curve was still falling (guard lower bound +0.037 cm);
   R15 allowed that. Whether 4 × 4 would pass on a larger corpus was not tested. Stage O reports
   the 4 × 4 readout on this task's 2 000-root corpus **as a reported-only number**; it does not
   reopen the grid choice and decides nothing.
3. **8 × 8 dynamics have never been gated.** TASK-066 gated 4 × 4 only, at h = 8 and 16, on a
   static apple, on the train split. This protocol gates its own 8 × 8 dynamics at h = 60 (§8.2),
   with no reach-back over TASK-066.
4. **The full test suite runs on the exact commit before any record or gated run** (review note 1:
   the C1-M record ran at a revision that failed a repository test). This is a guard here, G-tests
   (§10.2).
5. **The `early_verdict` placeholder bug is fixed in the new runner** (review note 2: the C1-M
   report stored `M-NO-TAU` for a ladder evaluated before τ_commit existed). A field this protocol's
   runner has not evaluated holds the literal sentinel `"not evaluated"`, never a row name, and no
   row ladder is evaluated with a missing input. A test checks both (§10.2).
6. **The record's proxies and H-read are privileged calculations, not trained twins.** The real N,
   L-shuf and L-mean are trained models here (§5.2), and their closed-loop counts are measured only
   in this protocol.

## 4. The model and its training

### 4.1 The latent and the predictor (TASK-066's, at 8 × 8)

- **The latent:** the frozen pretrained DINOv2 ViT-S/14's final patch tokens of the onboard 112 px
  frame (upsampled to 224 px as in TASK-063–076), average-pooled to an **8 × 8 grid**
  (`frozen_tokens.pool_tokens`, grid 8): 64 tokens × 384 = **24 576-d**, standardised per dimension
  by train-only moments (std floored at 1e-3). The encoder never trains.
- **The predictor:** TASK-066's pinned upstream LeWM predictor adapted to a token sequence
  (`frozen_token_model("leworldmodel")`), bidirectional within the frame, **history one**, with
  TASK-066's capacity and settings unchanged: depth 2, 2 heads × 24, MLP 128, zero dropout,
  SIGReg weight 0, recursive multistep loss weight 1.0, learning rate 3e-4 constant, weight decay
  1e-4, gradient clip 1.0. Its configuration is TASK-066's `MODEL_CONFIG` with `token_grid: 8` and
  `latent_dim: 24576` (`max_horizon` 64 ≥ 60).
- **The backend swap** stays one key (`BACKEND = "leworldmodel"`); native code is not run.

### 4.2 Three departures from TASK-066's recipe, and why (R17.3, R17.17)

TASK-066 trained on windows of 16 transitions with batch 64. Here:
- **Training windows of T = 60 transitions**, the gated horizon. TASK-066 gated only inside its
  training window; its extended horizons beyond it degraded (W read the static apple at
  1.09–1.18 cm at h = 32 and 1.18–1.52 cm at h = 64, against 0.53–0.59 cm for the encoded grid;
  TASK-066 results, reported only). The horizon is this task's main risk (R15 §5.8), so the
  recursion is trained over exactly the horizon it is used at.
- **Batch 16 windows**, because batch 64 at T = 60 does not fit the 16 GB GPU: the cost probe ran
  out of memory there (§15). Batch 16 × 60 = 960 transitions per update, against TASK-066's
  64 × 16 = 1 024, so the data per update is about the same.
- **BatchNorm's batch statistics change with the batch.** The upstream predictor's `pred_proj` MLP
  carries BatchNorm, whose training statistics are computed over the batch (here 16 windows × 64
  tokens = 1 024 rows per step, against TASK-066's 64 × 16 = 1 024; the same row count, a different
  mix of windows and tokens). It needs batch ≥ 2, which batch 16 satisfies. Declared as the third
  departure; nothing else in the recipe changes.

**The decisive argument for T = 60 (from the review of #142, R17.17).** Training windows start at
403–407 and the corpus keeps frames 403–467 only. With T = 16, no training transition after step
423 would ever be seen, so the plate's motion from 423 to 465, the phase the decision depends on,
would be untrained recursion in a regime the model never trained on. TASK-066's h = 64 readings
(1.18–1.52 cm) show how such extrapolation degrades.

Measured cost (§15, synthetic features, one run): 0.161 s per update at 8 × 8, T = 60, batch 16
(5.48 GiB peak), against 0.110 s for TASK-066's T = 16, batch 64 at 8 × 8. The alternative (T = 16,
batch 64, extrapolated to h = 60) is cheaper by about a third and is rejected for the reasons
above (settled, R17.17).

### 4.3 The corpus, `apple-c1m-v2` (privileged scripted collector; Stage C)

- **2 000 roots, seeds 67000–68999** (§6), each: P-3 picks; the move at 300 (salt 8101); at 405
  e9's aim is committed to (a, b) drawn uniformly over the box from the **true** p and h (salt 8102;
  the design note's privileged scripted collector, as in M-F5a). W is therefore never asked to rank
  aims outside its training aims.
- **Kept per root:** the onboard 112 px frames of every step 403–467 (65 frames), the executed
  normalised 14-D commands of 403–466, the true plate and palm at every kept step, a, b, the robot
  state at 405, a plate-hidden render at r (TASK-075's renderer) and the full-token features'
  source frame at 405. A root runs to step 467 and stops there (nothing after 467 is used).
- **Exclusions:** a root that does not reach 467 (a guard refusal or a fallback) is excluded and
  counted. More than 2 % excluded is **CORPUS-ESCALATE** (escalate, no clause); the C1-M corpus lost
  0.6 % (6/1 024, all in P-3's pick).
- **Splits by root, fixed before collection** (one permutation, salt 8103): **train 1 500, val
  250, gate 250**. Train fits everything (normalisation, W, N, the readouts, H-sysid, L-mean's
  mean latent); val selects checkpoints and sets G1's calibrated bars; **gate is used only by the
  offline gates (§8.2) and never for a fit or a selection.** The corpus roots are disjoint from
  every closed-loop cohort.
- **Why 2 000:** both readout learning curves were still falling at 1 024 roots (M-F5a), and W
  needs more transitions than a readout. Its cost is CPU time (about 45–60 min, *estimate*) and
  disk (§12).

### 4.4 Training (Stage T, GPU, per-job slots)

- **Arms trained:** W (true executed commands) and N (every command zero, in training and in every
  evaluation), **three model seeds each (66800, 66801, 66802)**, on the train split. W and N of a
  seed draw the same windows in the same order (sampler `SeedSequence([8109, seed])`).
- **Windows:** 60 transitions (61 frames) starting at a step drawn uniformly from 403–407 of a
  train root, so every window covers the decision step's neighbourhood; batch 16.
- **Normalisation:** per-dimension moments over every kept frame of the train split, once, before
  the first update. The metric scale for the gates is the same moments' std (floored at 1e-3),
  identical for every model.
- **Device:** CUDA for training only (strict determinism, `devices.configure_determinism("cuda",
  strict=True)`). **Every prediction that a gate or a closed-loop arm reads is computed on the CPU,
  float32, one torch thread**, from the selected checkpoint. The cost probe measured 7.4 s for 147
  candidates × 60 steps on one CPU thread (§15), so the closed loop needs no GPU.
- **Selection:** the val criterion is TASK-065's normalised MSE of the recursive prediction over
  steps 1…60 from frame 405 of every val root (250 windows; for N with zero commands). It is
  computed at 20 evenly spaced points of each model's budget. **The kept checkpoint is
  `run_tools.select_checkpoint(curve, 0.01)`**, the earliest point within 1 % of the curve's
  minimum (not the raw argmin; TASK-074's lesson).
- **The primary seed** for the closed loop is the W seed with the lowest val criterion at its kept
  checkpoint (TASK-073/074's rule), fixed before any closed-loop attempt; N uses the same seed.

### 4.5 The budget block and saturation handling (R17.5; TASK-074's lessons)

TASK-074 escalated twice on its budget: its cap equalled its calibration length with a factor of
2 (cap 60 000 < 2 × 60 000, so every saturation above 30 000 escalated by its own arithmetic), and
its last-two rule used a raw argmin, so noise in a flat tail read as unsaturated. Here:

| key | value |
|---|---|
| `calibration_runs` | (W, 66810), (N, 66810), on the train split, val criterion on val |
| `calibration_updates` | 50 000 |
| `calibration_select_every` | 1 000 (50 points) |
| `saturation_tolerance` | 0.01 (u_sat = `select_checkpoint(curve, 0.01)`) |
| `factor` | 2.0 |
| `step` | 5 000 |
| `min` | 10 000 |
| `cap` | **100 000** (= factor × calibration_updates) |
| `selection_points` | 20 |

- **U = clamp(5 000 · ⌈2 · max(u_sat(W), u_sat(N)) / 5 000⌉, 10 000, 100 000).** The block passes
  `run_tools.check_budget` (checked on this draft's revision: `consistent: True`, `max_request`
  100 000 = `min_cap`); a Stage-0 test asserts it. Because a saturation update is at most 50 000,
  the formula never asks for more than the cap, so **no budget row exists**: nothing in this
  protocol escalates on a budget by design.
- **The last-two rule never raises a budget and never escalates.** `run_tools.last_two_triggered`
  (curve, 0.01) is recorded per model as a saturation flag. An unsaturated N would bias G3 and
  G1 (iii) towards W (the TASK-066 owner ruling); so an N model flagged unsaturated is **reported
  beside every W-versus-N comparison of its seed**, and the results document states which passes
  rest on it. It does not change a row. Mitigation, declared now: U uses the larger of W's and N's
  calibration saturation, with factor 2.
- **G1's calibrated bars** come from the calibration W at its kept checkpoint, on val, at h = 60
  (§8.2). If its rank reference is below 0.10, the stage ends **CAL-T-ESCALATE** (the recipe may be
  collapsing; escalate, no clause), as TASK-066's rule.

## 5. Arms

### 5.1 W and its controller form (declared under R8.8)

At 405, with the robot's own joint state and the onboard frame only:
1. **Grid.** The plate reading p̂ = R-plate's reading of the 405 frame (full DINOv2 tokens, dual
   ridge: TASK-076's readout *form*, **refitted on this corpus's train split** at 405, not
   TASK-076's fitted weights) and the proprioceptive palm h
   define the 147-candidate grid of §2.1. Infeasible candidates (stand-in refusal or unreachable
   release pose) are dropped for every arm alike. If all are infeasible, the arm aims at p̂ (a
   counted attempt; expected never, a_lo's reach check was 31/32 complete).
2. **Commands.** Each candidate's 60 commands (405–464) come from the kinematic stand-in
   (`place_planner.primitive_chunks`, horizon 60), from the observed state, with the inputs the
   executed primitive receives. They are the commands e9's primitive would send if the arm tracked
   ideally; the executed commands differ (the corpus logs both), and W is trained on executed ones.
3. **Roll-out and score.** W encodes the 405 frame and rolls every candidate's commands to r;
   R8, a dual-ridge readout from the 8 × 8 latent to the plate fitted on the train split's encoded
   frames at r, reads the predicted plate p̃(g). The score is |p̃(g) − g|; the lowest score wins
   (ties to the lowest row-major index of (a, b)).
4. **Refinement.** At most 10 single-candidate roll-outs g ← clip(p̃(g)), stopping when the move is
   ≤ τ_commit/4; clip = C1's `clip_to_box`. With the rule's |κ| = 0.5 the iteration contracts.
5. e9's primitive executes the final g with no re-aim. P-3 controls everything before 405.

**The plate reading p̂ builds the grid for every candidate-choosing arm** (W, N, L-shuf, L-mean,
L-rand). It is a non-privileged single-frame readout. For the scene-blind twins it leaks scene
information only through the box (R15.2: the clip raised the record's proxy counts by at most
about +1/32 at ρ = 6 cm, development). That makes them stronger twins, so the twin tests are
conservative, and "every arm shares the grid and the clip" (design note §4.1) is kept (settled,
R17.17). **Each arm's clip-binding fraction** (the share of attempts whose chosen or refined aim
was clipped to the box) is reported in K0 (for the proxies), D and S, so that "scene-blind in
prediction only" is visible in the results.

### 5.2 The twins: trained models, not privileged proxies (R17.7)

| arm | what it is | expected (*from the record's privileged proxies; not a bar*) |
|---|---|---|
| **N** (action-blind) | the trained N of the primary seed, rolled from this reset's encoded 405 frame with zero commands; its p̃ is the same for every candidate, so it picks the candidate nearest its prediction and the refinement converges to it | the N-proxy scored 8/32 (M-F3) |
| **L-shuf** (scene-blind) | W from the encoded 405 frame of the **next reset in the cohort's order**, (i + 1) mod n, with **this** reset's candidate commands (TASK-074's `shuf`) | the shuf-proxy scored 8/32 |
| **L-mean** (scene-blind) | W from the **mean encoded 405 latent of the train split** (raw feature space, computed in Stage O from the train split's 405 features and fixed there), with this reset's candidate commands (R9.10) | the mean-proxy scored 12/32 |
| **L-rand** | a candidate drawn uniformly from the feasible grid (salt 8111), no refinement | low |

The foreign 405 frame for L-shuf is the logged 405 frame of W's attempt on reset (i + 1) mod n.
Every arm runs P-3 identically to 405, so that frame matches across arms **up to the renderer's
nondeterminism** (EGL worker history; R17.21): Stage 0's smokes saw two 405 readings differ
between arms on the same reset by about 6e-5 m (seeds 66920 and 66910). G-repro fixes the
post-look frame, not the 405 frame.

**A reset refused before 405 (R17.20).** P-3's pick is the shared prefix of every arm, so a reset
whose attempts end before 405 (a guard refusal in the pick; C1-M saw 1 of 32 resets and 6 of
1 024 roots) is a failure for every arm: a concordant fail-fail pair. It stays in the cohort's
denominator (G-bar counts it as a miss) and never voids a stage. L-shuf's foreign frame is then
the logged 405 frame of the **next reset in cohort order, cyclically, whose W attempt reached
405**. The determinism re-run checks a refused reset only for being refused identically (the same
termination reason and executed steps). K0's and every closed-loop stage's reports count the
resets refused before 405 per arm; the corpus already excludes and counts them (§4.3).

### 5.3 Comparators, ceilings and reported arms

| arm | role | privileged? |
|---|---|---|
| **H-rule** | `RuleCommit`: the rule's fixed point on p̂ and h, single commit, clipped | no (it knows the rule, a simulator law) |
| **H-sysid** | `SysidAim`: plate(r) ≈ c + A·p̂ + B·h + C·g fitted on the train split with out-of-fold readings, inverted with W's controller form, clipped | no |
| **H-final(commit)** | the ceiling: `LookaheadAim` once at 405 in cloned state, unclipped (as C1-M) | **yes**; not learned |
| **H-read** | reported only: H-final(commit)'s look-ahead with plate(r) read by R8 from the frame rendered at r in the clone (perfect dynamics, real readout); g0 = the true plate (disclosed) | **yes** |
| **H-now** | reported only: the true plate at 405 | **yes** |

The non-inferiority comparator is **the better of H-rule and H-sysid on S** (the larger count; a
tie goes to H-rule). Choosing the better one after S is conservative for W. The re-aiming arm
H-now-reaim is not run (a different controller structure; design note §3).

## 6. Seeds, cohorts and salts (R17.11)

**Block 66000–68999**, outside every forbidden range of TASK-076, C1 and C1-M
(`lewm_next_c1m.FORBIDDEN_RANGES` plus C1-M's block 63000–64999; checked in code on this draft's
revision) and outside 58000–58999 and 63000–64999 as required.

| seeds | use |
|---|---|
| 66000–66031 | **K**: K0 (32 resets) |
| 66100–66115 | **D**: the development closed loop (16 resets) |
| 66200–66263 | **S**: the gated cohort (64 resets) |
| 66800–66802 | model seeds of W and N (torch; not resets) |
| 66810 | the calibration model seed (torch; not a reset) |
| 66900–66999 | **debug**: runner mechanics only; nothing in it is read. The cost probe (§15) used 66990 and 66991 as torch seeds on synthetic data |
| 67000–68999 | **corpus**: 2 000 roots |
| everything else in the block | reserved; any use needs its own ruling |

| salt | use |
|---|---|
| 8101 | the move draw per reset: `default_rng(SeedSequence([8101, seed, k]))`, k the re-draw index |
| 8102 | the corpus's uniform (a, b) per root |
| 8103 | the corpus split (train / val / gate) |
| 8104 | Stage O's outer folds by root |
| 8105 | λ selection (inner CV grouped by root) in every ridge |
| 8106 | every bootstrap (10 000 resamples) and every feasibility resampling |
| 8107 | τ_commit's planted-error direction per reset (K0) |
| 8108 | Stage O's nested learning-curve subsample |
| 8109 | the training window sampler: `SeedSequence([8109, model seed])` |
| 8110 | G4's wrong-command permutation across gate roots |
| 8111 | L-rand's draw |
| 8112 | reserved |

**The search** (2026-10-05, at `e35bd91`; R17.11). `git grep -w` over all 74 local and remote refs
(every path): **66000–66999: no match.** 67000–68999: only `67548` and `68658` (wheel byte sizes in
`uv.lock`) and `68791` (row counts in two manifests); none is a seed. 69000–69999 was not taken,
because `69112` is a recorded RNG value ("seed 49112, rng 69112",
`docs/reviews/apple_wide_grasp_closure_v3_review.md`). The working files (`src`, `scripts`,
`tests`, `configs`, `docs`, `benchmarks`, `.mc`) of every worktree under
`/home/huhn/develop/emai/worktrees/`, the main checkout and `wt-task076-freeze-prep` add only
`66526`, a byte count in two non-repository worktrees (`ar-v1`, `t1-cluster`). Salts: 8101–8112
appear on no ref in `src`, `scripts`, `tests` or `configs` (the only `81xx` matches are `8192`). The
Stage-0 test re-checks the block against the forbidden ranges in code.

## 7. Stages and the staged GO flow

Nothing from K, D, S or the corpus is simulated before the reported GO of its stage. Every stage
runs from a clean worktree of the merged revision (`scripts/new_worktree.sh --run --from <rev>`),
after G-tests (§10.2). The coordinator is told before each stage starts, and the main session posts
a notice in chat before the gated stage.

1. **Stage 0 (the preregistration PR, after this draft's review).** The stage code in new modules
   (`lewm_c1m_v2.py` with the frozen block as module constants, `lewm_c1m_v2_runtime.py`,
   `lewm_c1m_v2_offline.py`, a shared training module, `scripts/run_lewm_c1m_v2.py`), the manifest
   (`benchmarks/manifests/apple-lewm-c1m-v2.json`, DRAFT until the freeze), the tests (seed ranges
   and salts; `check_budget` on the block; `select_checkpoint` and `last_two_triggered` used and no
   raw argmin; the `"not evaluated"` sentinel; no runner imports; `assert_local_import`,
   `install_guards`, `gpu_guard`; no privileged read in W or a twin; every row in both directions),
   and smokes on the debug range: one corpus root per split, a tiny training run (a few hundred
   updates) on the real cache layout through `gpu_run.sh`, one D-style attempt per arm. **Stage 0
   also runs the scale probe** (`run_tools.scale_probe` on the stage's own functions): the real
   per-update time with the real cache and batch gather, and the train job's peak PSS. The wall-time
   caps of §10.3 are then set to at least 1.5 × the measured worst case and written into the
   protocol before the freeze.
2. **K0 (on a reported GO; CPU only; before the freeze).** On K's 32 resets, at ρ\* = 4 cm (disc):
   - **τ_commit:** H-final(commit)'s converged aim plus a planted error of 0, 0.5, 1, 1.5, 2 or 3 cm
     in a per-reset uniform direction (salt 8107), every level on the same 32 resets (TASK-076 K0's
     rule: the largest level such that every level up to it reaches ≥ 28/32; undefined if level 0
     misses).
   - **The ceiling** N_K(0) = the level-0 count.
   - **r and the history check** from the level-0 attempts: r_K, the earliest step at which ≥ 28/32
     are within 0.1 cm of plate(525) (R14.3); the palm speed and m2 at 405.
   - **The proxies** (reported): H-now, N-proxy, shuf-proxy and mean-proxy on K (R16.5's formulae,
     clipped), with their paired headroom intervals and each scene-blind proxy's McNemar feasibility
     for the 64-reset test (R8.12's form, salt 8106). A feasibility below 0.8 is disclosed in the
     freeze PR as a known risk, not a stop.
   - **Stops, all CAL-ESCALATE** (escalate, no clause; nothing is frozen): level 0 < 28/32
     (τ_commit undefined); N_K(0) < 30/32; r_K later than 465 or undefined (the frozen h = 60
     would read the plate before it settles); the median palm speed at 405 > 0.5 cm per step.
   - **K0-PASS:** τ_commit, the τ curve, N_K(0), r_K, the proxies and their feasibility enter the
     frozen block. About 350 attempts, about 10–15 min on 6 workers (*estimate* from C1-M's
     9.7 s maximum per attempt).
3. **The freeze.** Status FROZEN and the frozen-block sha pin, merged on an independent
   reviewer's reported APPROVE. **Prepared (R17.25)**: K0's values are in `K0_MEASURED`, the pin
   is set, and the manifest pins the stage's files and this document. The pre-freeze items of the
   #143 approval are settled: the feature-file check (R17.26), the determinism rule (R17.27), and
   the scale probes of Stage O's readouts and Stage C (R17.28). The thread environment was checked
   (R17.29). A non-debug `k0` is refused once FROZEN.
4. **Stage C, the corpus (on a GO; CPU).** §4.3. Rows: CORPUS-ESCALATE (> 2 % excluded; escalate,
   no clause), CORPUS-SEALED. The sealed manifest's sha256 is checked by every later stage
   (R17.22): the runner requires it, and ties the artifacts to it. The featurise report must be
   FEATURISED and the readouts report O-PASS, each made from the same corpus manifest; the
   moments, R8, R-plate and L-mean's mean latent are checked against Stage O's recorded sha256s;
   every model's `normalisation_sha256` and corpus sha256 against those; a debug report never
   feeds a real stage.
5. **Stage O, offline admission (on a GO; GPU for featurisation only).** Featurisation of every
   kept frame (8 × 8; 4 × 4 at 405 and r, reported; full tokens at 405) on CUDA through
   `scripts/gpu_run.sh --wait`, with a CPU anchor check (G-anchor). `first_outcome_utc` is written
   before the first fit. Then O1, O3, O4 (§8.1), the learning curve, and the train-only fits used
   downstream: R-plate, H-sysid, R8, the normalisation moments and L-mean's mean latent (computed
   here and fixed; Stage T only reads it). Rows: O-ARM-KEYED, O-NO-BAR, O-PASS (§8.1).
   **Before Stage O's GO (R17.24): done (R17.28, §10.3).** The scale probe of Stage O's readouts
   ran on synthetic features at the real sizes (1 750 roots, the 24 576-d and 98 304-d ridges,
   the train moments). Stage C's per-root cost was measured on debug seeds. Both caps are kept:
   each is far above 1.5 × the measured worst case.
   **Feature files (R17.26).** Every stage that reads Stage O's featurisation (readouts, train,
   plan, gates) first hashes every file of each split it reads: the 8 × 8 store, the commands, the
   roots, the hidden render, the 4 × 4 pool, the full tokens and the table. It compares them with
   the featurise report's `files_sha256`. A missing, unrecorded or different file is V. The check
   costs about 8 s for the 12 GB of the train and val splits (R17.28).
   **The featurisation's scale probe (R17.37, Erratum 2026-10-05 (b)): done** (§7.4, §10.3). It
   ran the runner's own `featurise` stage on a synthetic corpus of the sealed corpus's size
   (1 995 roots) through `scripts/gpu_run.sh --wait`, without reading the corpus.
6. **Stage T, training (on a GO; GPU, one `gpu_run.sh --wait` job per model: the two
   calibration runs and the six models, eight jobs).** The calibration (W and N, 66810), the budget rule, G1's bars and the truncation
   controls on val, then the six models. Rows: CAL-T-ESCALATE, T-DONE. No budget row (§4.5).
   (R17.24: each job's report ends T-JOB-DONE; the `plan` step ends CAL-T-ESCALATE or T-PLANNED;
   Stage T's row T-DONE is decided by `decide_t` when Stage G reads all eight jobs, and is
   recorded in Stage G's report.) The exact commands, with `gpu_run.sh --wait --min-free-gib 8
   --board`, are in §7.6.
7. **Stage G, the offline gates (on a GO; CPU).** §8.2 on the gate split, every gate on all three
   seeds. Also written before any closed loop (reported only): each arm's offline aim error from
   the gate roots' 405 states (stand-in chunks, the same controller code) against the rule's fixed
   point g\* = (p − κh)/(1 − κ), and its τ-curve-mapped predicted count with K0's curve. Rows:
   H-GATE-FAIL, G-NO-BAR, G-PASS.
8. **Stage D, the development closed loop (on a GO; CPU).** W, N, L-shuf, L-mean and
   H-final(commit) on D's 16 resets. **L-DEV-STOP** (escalate, no clause) if W < 12/16, or
   H-final(commit) < 14/16, or W − max(N, L-shuf, L-mean) < +3/16. Nothing is refitted on D.
   (False-stop probabilities, exact binomial: W < 12/16 at a true rate of 0.875 is 0.041, at 0.9375
   is 0.002; H-final < 14/16 at 0.9375 is 0.074, at 0.969 is 0.012.)
9. **Stage S, gated (on a reported GO after D-PASS; CPU).** Cohort S, every arm of §5 once per
   reset, paired, with a determinism re-run of W on S's first four resets. **Its rule (R17.21,
   amended by R17.27).** The R-plate reading of the 405 frame is gated at 0.1 cm: more than 15 ×
   the renderer noise Stage 0 saw (about 6e-5 m), and 10 × below τ_commit. The success outcome
   must also be the same in both runs. The commit target's difference is **reported, not gated**
   *(amended by R17.38, a post-freeze amendment recorded with Erratum 2026-10-05 (b), R17.43: it is gated at 0.6 cm, the bound that the
   argument below gives; see §7.4)*. The target comes from an argmin over 147 candidates followed by a refinement that stops at
   τ_commit/4. A renderer flake that meets a near-tie, or that stops the refinement one iterate
   earlier, can therefore move the target by more than 0.1 cm with no difference in what W saw.
   With the rule's contraction |κ| = 0.5, stopping at a move of 0.25 cm leaves the iterate within
   about 0.25 cm of the fixed point, so two runs can end about 0.5 cm apart. The re-run is V in any
   of these cases: a committed reset whose reading differs beyond 0.1 cm; a reset whose success
   differs between the runs; a reset that commits in one run and not the other; *(R17.38)* a
   committed reset whose commit target differs by more than 0.6 cm. A reset refused before 405
   must be refused identically. Rows: §8.4.
10. **Results PR.** Every arm is reported, the privileged arms are labelled as not learned, and an
    independent reviewer checks every restated number.

### 7.1 K0's result: K0-PASS (R17.25)

K0 ran once, on the reviewer's reported GO (#143,
<https://github.com/RaaSaaR-org/open-embodied-jepa/pull/143#issuecomment-5988386225>), at
`306fbdc`, on a clean tree with the DRAFT frozen sha `89359ed3…5b12`. It ran on the CPU with
6 workers and without the GPU lock. In-run G-tests passed (1990 passed, 37 skipped). G-repro
passed 8 of 8 checks. There were no render disagreements. The run took 579 s. Load at the start
was 0.22 / 0.49 (1- and 5-minute). Peak process-tree PSS was 8.0 GiB.
- **Report:** `outputs/task077-k0-1/report.json` in the worktree
  `/home/huhn/develop/emai/worktrees/task077-k0` (git-ignored), sha256
  `9be44fd93996c03cacc6cde089225f0f986676020f2ceb2138f062131f1af235`. Its log is beside it.

**τ_commit**: counted successes of 32 for H-final(commit)'s aim plus a planted error, on the same
32 resets:

| planted error (cm) | 0 | 0.5 | 1 | 1.5 | 2 | 3 |
|---|---|---|---|---|---|---|
| counted successes / 32 | 32 | 29 | 29 | 18 | 11 | 2 |
| median landing miss (cm) | 0.09 | 0.75 | 1.49 | 2.24 | 2.98 | 4.49 |

- **τ_commit = 1.0 cm**, equal to the C1-M record's value. It enters O1, O4 and G5 (a).
- **The ceiling N_K(0) = 32/32.**
- **r_K = 460**, so the frozen r = 465 stands.
- **The history check.** At 405 the palm speed had median 0.00227 cm per step (maximum 0.024),
  and m2 had median 0.00227 cm (maximum 0.020). A history-one predictor qualifies.
- **No stop fired.** There were no refusals, the clip-binding fraction was 0 and there were no
  fallbacks in any arm.

**The proxies** (reported; privileged calculations, not the trained twins):

| proxy | count / 32 | ceiling minus proxy (95 % interval) | McNemar feasibility at the ceiling / at W's bar |
|---|---|---|---|
| H-now | 0 | +32 [+32, +32] | — |
| N-proxy | 13 | +19 [+14, +25] | — |
| shuf-proxy | 10 | +22 [+17, +27] | 1.000 / 1.000 (b = 22 / 18, c = 0) |
| mean-proxy | 15 | +17 [+11, +22] | 1.000 / 1.000 (b = 17 / 13, c = 0) |

No scene-blind feasibility is below 0.8, so no known risk is disclosed on that ground. The
proxies scored higher than in the C1-M record (8, 8 and 12 of 32 there). A higher proxy count is
also what the trained twins could reach (§13).

**Thin margins, disclosed (R17.25).** Each of these values passed its stop rule, but each is close
to a bar:
- **r_K = 460 against the frozen r = 465:** 5 steps of margin. If S's resets settle later than K's
  did, W reads a plate that has not settled. That would lower every arm's success, including
  H-final(commit). S-VOID-CEILING covers the condition's ceiling on S, but r is not re-measured.
- **The 0.5 cm and 1.0 cm levels each sit only one success above the 28/32 bar (29/32).** Their
  failures fall on disjoint resets (66013, 66014, 66017 and 66018, 66019, 66028). One more failure
  at 1.0 cm would have made τ_commit = 0.5 cm. One more at 0.5 cm would have made it 0 cm. So
  τ_commit = 1.0 cm is an upper reading of the tolerance on 32 resets, not a margin.
- **What this means for G-bar.** G-bar is 56/64, which is 28/32. A W whose aim error sits at
  about τ_commit would succeed at about 29/32 ≈ 0.906, by this curve. At that rate G-bar passes
  with probability 0.86 (§8.4's power table), not with certainty.
- **What this means for O1 and G5 (a).** Both use τ_commit = 1.0 cm as their bar. A τ_commit of
  0.5 cm would have halved both bars.

### 7.2 Stage C's first run: V on G-memory, and Erratum 2026-10-05 (R17.30–R17.34)

**Erratum 2026-10-05** (decided by Claude under owner delegation; DECISIONS 2026-10-05 (b),
R17.30–R17.34). This erratum changes code and report fields only. The frozen block, its sha256
`f28e5e2c…548d`, and every bar, seed, salt, cap, ceiling and row are unchanged.

- **The void.** Stage C ran once at `862d63c` under the GO (#144, issuecomment-5989316868). It
  ended **V on G-memory**: process-tree PSS reached 12.22 GiB against the 12.00 GiB ceiling, with
  a peak of 12.896 GiB. The V came during `Cohorts.seeds("corpus")`, after the corpus seeds'
  post-look frames were rendered and before any root was collected. Nothing in the run is read.
  - Report: `outputs/task077-corpus-1/report.json` in the `task077-corpus` worktree, sha256
    `734fa771875110637748c00303e0aee6f65577c9b75558493df161f14194a6c3`.
- **The cause (R17.30).** TASK-076's pinned `cohort_estimates` held all 2 000 seeds' full DINOv2
  tokens at once in the main process:
  - the tokens themselves, float64 [2 000, 98 304]: 1.47 GiB;
  - two float64 copies made by `info_ceiling.cross_gram`: 2.93 GiB more.

  That is about 4.4 GiB on top of a 7.6 GiB baseline (the main process 1.1 GiB, six workers of
  1.07 GiB each). A probe of the pinned path at 2 000 seeds reproduced the void with a 12.94 GiB
  peak. R17.28's scale probe had run this step on 100 seeds only.
- **The fix (R17.30).** The runner's own `cohort_estimates` renders exactly as before. It then
  streams the tokens 128 frames at a time through the pinned `cross_gram`, and runs the pinned
  kernel-ridge `predict` once on all rows.
  - The estimates are **bit-identical**, both by construction (32-row-aligned blocks) and as
    measured: on the debug seeds' real tokens, on 2 000 synthetic rows and at the full step.
    *(Corrected by R17.36, §7.4: "by construction" overstated it. Bit-identity is measured, and
    it fails when the final chunk has exactly one row; that cannot occur in TASK-077, and such a
    chunk is now refused.)*
  - Stage C's seed preparation at 2 000 seeds now peaks at **7.75 GiB**. *(R17.36: that is the
    estimate step with stand-in frames, not the stage's peak; the stage's peak of about 8.5 GiB
    was an expectation, and the repeat measured 8.38 GiB, §7.3.)*
  - No pinned TASK-073–076, C1 or C1-M file changes.
- **The same pattern in Stage O (R17.31).** `featurise_corpus`'s memory-mapped stores held the
  written pages of the whole train split in PSS: 8.82 GiB, measured on a synthetic 2 000-root
  corpus with a stand-in encoder, before the encoder's own memory. They are now re-mapped every 32
  roots, which measured 0.99 GiB, and the written files are byte-identical.
- **Preflight (R17.32).**
  - Once FROZEN, preflight checks this document's sha256 directly against the manifest.
  - Tests cover the FROZEN-only paths.
  - The report's top-level load average is renamed `load_average_after_preflight`, because it is
    taken after G-tests. G-quiet's reading is `quiet_machine.load_average_at_start`.
  - The runner creates `outputs/` when it is missing, and takes `--log <file>`.
- **Disclosures (R17.33).** The `task077-cprobe-3` crash and the probe's 2.21 five-minute load are
  disclosed. CPU stages render with EGL and take no GPU lock (§10.2).
- **The repeat (§10.1).** Stage C gets its one repeat under these conditions:
  - at this erratum's merge commit (the fix commit);
  - on a reported GO that names that commit, the cause and the void report's sha256;
  - from a fresh clean worktree;
  - on the same seeds 67000–68999;
  - into a new output directory.

  A second V of Stage C ends TASK-077 as INCONCLUSIVE.

### 7.3 Stage C's repeat: CORPUS-SEALED (R17.35)

Stage C's one repeat ran at `4f30fbb`, the merge commit of the erratum (#145), on the reviewer's
reported GO (#145, <https://github.com/RaaSaaR-org/open-embodied-jepa/pull/145#issuecomment-5990368000>).
It ran from a fresh clean worktree (`task077-corpus2`) on the CPU with 6 workers and without the
GPU lock, with the command the GO named, on the corpus seeds 67000–68999.
- **Guards.** G-quiet read 0.10 / 0.08 at the start. In-run G-tests passed at `4f30fbb` on a
  clean tree (2001 passed, 37 skipped). G-hash checked TASK-076's 84 pins and TASK-077's 13 own
  pins, with the frozen sha `f28e5e2c…548d` and the protocol document's pin. G-repro passed 8 of 8
  checks. MemAvailable at the start was 25.4 GiB.
- **Memory.** Peak process-tree PSS was **8.38 GiB** against the 12.00 GiB ceiling (RSS
  10.05 GiB). §7.2's "about 8.5 GiB" was an expectation (the #145 approval's note 2); this is the
  measured peak.
- **Time.** 1 234 s from start to end (cap 14 400 s), from 07:53:59Z to 08:14:33Z.
- **Row: CORPUS-SEALED.** 1 995 of 2 000 roots are kept. The 5 excluded roots are 67692, 67888,
  68047, 68414 and 68774. Each was excluded as `no_decision`: no commit decision was made at 405.
  All five are train seeds, so the kept split is 1 495 train, 250 val and 250 gate roots. The
  excluded fraction is 0.25 %, against the 2 % bar.
- **Render disagreements.** Two seeds, 67133 and 68878, had one of their three post-look renders
  differ from the other two in 4 pixels, by at most 1 level, with equal states. The majority frame
  was used, as declared.
- **Sealed manifest:** `outputs/task077-corpus-2/corpus/manifest.json` in the worktree
  `/home/huhn/develop/emai/worktrees/task077-corpus2` (git-ignored), sha256
  **`ad8974b2a8b560bb974c6e0b4f90bd3f1fc79a535ebe46ef6c409bde7e4343fb`**. Every later stage
  requires it (§7 step 4).
- **Report:** `outputs/task077-corpus-2/report.json` in the same worktree, sha256
  `2f84515bbdd612c75f56439355eff580f0617f0b5a7e64652a3626653dd07e47`. Its log is beside it.
- **The void first run is kept.** Its report (`outputs/task077-corpus-1/report.json` in the
  `task077-corpus` worktree, sha256 `734fa771…94a6c3`) stays the record of the V (§7.2). Neither
  corpus worktree is edited.

### 7.4 Erratum 2026-10-05 (b): bit-identity, the featurisation's scale probe, the determinism target and G-anchor's frames (R17.36–R17.39)

**Erratum 2026-10-05 (b)** (decided by Claude under owner delegation; DECISIONS 2026-10-05 (b),
R17.36–R17.39). It changes code, tests and this document only. The frozen block, its sha256
`f28e5e2c…548d`, and every seed, salt, cap, ceiling and row are unchanged, and **no bar changes
except R17.38's added 0.6 cm determinism gate, a post-freeze tightening**. R17.38 is therefore an
**amendment**, not an erratum in the TASK-058/076 sense (R17.43, the #146 approval's note 1). The
manifest re-pins the changed files and this document.

- **Bit-identity, corrected (R17.36; the #145 approval's note 1).** §7.2 and R17.30 said the
  streamed estimates are bit-identical "by construction". That overstated it.
  - Bit-identity is **measured**, at every size that was tried. It **fails when the final chunk
    has exactly one row**: `cross_gram`'s norm line (`einsum("ij,ij->i")`) takes a different
    reduction path on a one-row array, and the estimates then differ by about 1e-15. The reviewer
    measured this at N = 1 985 (chunks of 32 or 64), N = 97 and N = 33.
  - **The multiple-of-32 rule is not the real condition.** A chunk of 100 was also bit-identical
    at N = 2 000. The runner keeps the rule as a conservative one.
  - **This cannot occur in TASK-077.** The corpus's 2 000 seeds leave a final chunk of 80 at 128,
    and K, D and S are one chunk each. The runner now also refuses a one-row final chunk after
    another chunk (a GuardError), and a test covers it.
  - **The 8.5 GiB.** §7.2's 7.75 GiB measured only the estimate step, with stand-in frames, and
    the stage's "about 8.5 GiB" was an expectation (the #145 approval's notes 2 and 3). The repeat
    measured 8.38 GiB (§7.3).
- **The featurisation's scale probe (R17.37; the #144 approval's note 5).** This was the open item
  before Stage O's GO.
  - **What ran.** `scripts/probe_task077_featurise.py` (development only, not pinned) wrote a
    synthetic corpus of the sealed corpus's size: 1 995 roots, 1 495 train, 250 val and 250 gate,
    with uniform-noise frames under the labels 900000–901999. Nothing was simulated and the real
    corpus was not read. It then ran the runner's own `featurise` stage on that corpus, with
    `--debug`, through `scripts/gpu_run.sh --wait`. That is the real stage code: preflight,
    G-memory at 12 GiB of PSS, the 3 600 s cap, `gpu_guard` with 8 GiB free, the pretrained
    DINOv2 encoder on CUDA under strict determinism, the re-mapped stores and G-anchor. Around the
    stage it sampled the process tree's PSS and this process's GPU memory, and read torch's peak
    allocation at the end. Noise frames compress worse than rendered ones (2.49 MB a root against
    about 0.61 MB), so reading and hashing the roots is slower than it will be on the real corpus.
    The encoder's cost does not depend on the content.
  - **Result** (`task077-fscale-3`, at `ab5935f`, clean tree, G-tests from a `tests` record at
    the same revision: 2002 passed, 37 skipped, sha256 `536e9968…c60a`):

    | measure | probe | Stage O's cap or guard |
    |---|---|---|
    | stage wall time | 212.7 s (featurisation 207.4 s, G-anchor on 256 frames included) | 3 600 s (17 ×) |
    | process-tree PSS, peak | 1.76 GiB | 12.00 GiB (G-memory) |
    | GPU memory, this process (`nvidia-smi`) | 0.90 GiB (torch's peak allocation 0.49 GiB, reserved 0.55 GiB) | 8 GiB required free at the start (`gpu_guard`); 16 GiB card |
    | G-anchor | max difference 7.4e-5 on 256 frames | 1e-3 |
    | feature files written | 13.8 GB | 25 GiB of free disk at the start |

  - **The 3 600 s cap is kept.** It is about 17 × the measured stage, far above the 1.5 × rule.
  - **Reports:** `outputs/task077-fscale-3/probe.json`, sha256 `07663fc18f722daef3c6c0df071d434248ebbd56542e6c2333140f0475133fbd`, and the stage's own
    report `outputs/task077-fscale-3/stage/report.json`, sha256 `f9ebedc9b6094f5071291b595fdb21aa2e33a197f774d40d8ad70f65447a9494`, in the worktree
    `/home/huhn/develop/emai/worktrees/task077-stageo-prep` (git-ignored). The synthetic corpus and
    features were removed afterwards.
  - **Two earlier runs, disclosed.** Both ran at `badbbe1`, which checked G-anchor on the first
    root's 32 frames only (see R17.39 below).
    - `task077-fscale-1` ended **V on G-hash**: documentation edits were made in the probe's
      worktree while it ran, so the tracked tree changed during the run. The featurisation itself
      completed (201.6 s; peak PSS 1.74 GiB; GPU 0.90 GiB). That is the R14.9 lesson again, in a
      development probe. Probe report sha256 `342ccf01…2304`, stage report `c326d7c2…91ad5`.
    - `task077-fscale-2` ran clean and ended FEATURISED-DEBUG in 206.8 s, with a peak PSS of
      1.72 GiB and 0.90 GiB of GPU memory. Probe report sha256 `02e546e0…298e`, stage report
      `c3706582…36ad`.
- **The determinism re-run's commit target is gated (R17.38, an amendment; the #144 approval's note 1).** The
  target's difference was reported, not gated (R17.27). It is now **gated at 0.6 cm**, a bound
  derived from §7 step 9's own argument:
  - The refinement stops once a move is at most τ_commit/4 = 0.25 cm, and it returns that
    iterate. With the rule's contraction |κ| = 0.5, that iterate is within
    |κ|/(1 − |κ|) × 0.25 = 0.25 cm of the fixed point. Two runs can therefore end at most 0.5 cm
    apart.
  - A frame difference that passes the 0.1 cm reading gate is taken to move W's predicted plate by
    about as much. That moves the fixed point by at most 0.1/(1 − κ) = 0.067 cm (κ = −0.5).
  - 0.5 + 0.067 = 0.567 cm, rounded up to **0.6 cm**.
  - **Why gate rather than explain.** A W roll-out or R8 that differs between runs while the frame
    is the same would move the target, and without this gate it would be caught only if one of
    the four resets flipped its success. With the gate, real nondeterminism voids rather than
    hides.
  - **What it assumes, disclosed.** The bound assumes that W's learned map contracts like the
    rule's (|κ| = 0.5), and that the refinement converged. If W's map contracts less, or a
    refinement stops on an infeasible step or after its 10 roll-outs, two runs can end further
    apart, and the gate would void S on a renderer flake. Identical frames give
    identical targets on the CPU, so the gate can only bind when a frame differs. Stage 0 saw two
    such differences in 405 readings on debug seeds, and Stage C two in 2 000 post-look renders
    (§7.3). The risk of a false V is small, and a false V is a repeat, not a lost result.
  - **Where it lives.** `DETERMINISM_TARGET_BOUND_M = 0.006` and its rule text sit in
    `lewm_c1m_v2.py` **outside** the frozen block, and `determinism_check` gates on them. The
    frozen block's `determinism_rule` text ("reported, not gated") is unchanged and is superseded
    for the target only by this erratum. The frozen sha256 does not change. This tightens Stage S
    only, and it is settled before Stage S's GO, as the approval asked.
- **G-anchor's frames (R17.39; found while preparing this probe).** The frozen block carries
  TASK-076's `FEATURE_ANCHOR` = {frames 256, bound 1e-3}. `featurise_corpus` compared only the
  first root's first 32 frames, as TASK-075's and TASK-076's runners did. It now compares the
  first 256 kept frames, in split and root order (the first four train roots), with the CPU path.
  The bound is unchanged. A test checks the 256. `task077-fscale-3` ran with it.

### 7.5 Stage O's result: O-PASS (R17.40, R17.41)

Stage O ran once at `8721516`, the merge commit of #146, on the reviewer's reported GO
(<https://github.com/RaaSaaR-org/open-embodied-jepa/pull/146#issuecomment-5991796440>), from the
fresh clean worktree `/home/huhn/develop/emai/worktrees/task077-stageo`, with the commands the GO
named. Its reports are in that worktree (git-ignored); it is not edited.

- **G-tests record.** `outputs/task077-o-tests-1/report.json`, sha256
  `80fc8344b128e03f7604481364ead13bba1fbba2bef0d535a0b950df3e3784b4`: TESTS-PASS at `8721516`,
  clean tree, 2002 passed and 37 skipped, 09:32:21Z–09:34:55Z.
- **Featurisation (GPU).** `outputs/task077-featurise-1/report.json`, sha256
  `f187c7c8179ffe0d1883739be0f30ae960c6033e4c1d088d800075490ef3e9ee`: **FEATURISED**, corpus
  sha256 `ad8974b2…43fb`, 1 495 / 250 / 250 roots. It ran under the GPU lock (`gpu_lock_held`
  true; the lock log has its start and end at 09:35:08Z and 09:38:40Z) with 14.98 GiB of GPU
  memory free at the start. The stage took 211.3 s against the 3 600 s cap (featurisation
  205.5 s). G-anchor: maximum difference 1.0e-4 on 256 frames, against 1e-3. Peak process-tree
  PSS 1.73 GiB against 12.00 GiB. The feature files' sha256s are in the report's
  `files_sha256`; every later stage checks them.
- **Readouts (CPU, no lock).** `outputs/task077-readouts-1/report.json`, sha256
  `452045d22c21b067c8bfe78cb72f4e842fc61f3c44a9e17b6fae860c244f8b78`: **O-PASS**. G-quiet read
  1.62 / 1.52 at the start (bar 2.0). In-run G-tests passed (2002 passed, 37 skipped,
  09:39:13Z–09:41:41Z). The feature-file check verified the 14 train and val files (12.1 GB) in
  9.4 s. `first_outcome_utc` 09:41:50Z. The stage took 194.2 s against the 7 200 s cap, with a
  peak PSS of 2.97 GiB.

**The gates** (median plate error in cm over the 1 745 train + val roots, cross-fitted on 5 outer
folds; 95 % root-clustered intervals):

| gate | measured | bar | result |
|---|---|---|---|
| O1: c_plate, the 8 × 8 readout at r | 0.404 [0.385, 0.417] | upper bound ≤ τ_commit = 1.0 | pass |
| O3: c_plate / the constant prior (2.644 [2.549, 2.724]) | 0.153 [0.145, 0.160] | upper bound < 1.0 | pass |
| O4: the same readout on the plate-hidden renders at r | 4.579 [4.443, 4.673] | lower bound > τ_commit = 1.0 | pass |

**Reported only.** The learning curve is **still falling**: median 0.625, 0.515, 0.446 and
0.404 cm at 1/4, 1/2, 3/4 and all of each fold's fit roots, and the guard's 3/4-minus-all
difference is 0.041 [0.027, 0.060] cm, above 0. The 87.5th percentile of c_plate is 0.805 cm. The
4 × 4 readout at r reads 0.584 [0.564, 0.602] cm, R-plate at 405 reads 0.148 [0.142, 0.153] cm,
and H-sysid at r reads 0.554 [0.539, 0.574] cm. The plate moves 8.43 [8.33, 8.49] cm from 405 to
r. C1-M's record read 0.478 cm (8 × 8) on 1 024 roots; that was a scale reference, not a bar.

**The train-only fits** that later stages read (1 495 train roots; `outputs/task077-readouts-1/
fits/`). The content sha256s below are what the runner checks; the file sha256s are given too:

| fit | content sha256 (checked) | file sha256 |
|---|---|---|
| normalisation moments | `5415eea4b5eda30712176b4f23c0022886a0f9cf1ac60cdf742863393550de6d` | `8d2417d61dbd91618c7a2c290f53f3387d85e1df6716ccd4cd94b826cb7b344e` |
| R8 (dual ridge, λ_rel 0.01) | `62a5ea8abe348d2e7cbdb3379a22ecddcabd6a3f57c65a6f38f8ae57b8e43b4a` | `c1b2c3e3778dcbfe91a0905584180fe4ffdb6d3c0490c235db2222152e96d768` |
| R-plate (dual ridge, λ_rel 0.01) | `08bde901ca75ab0ee4bc1c74bf4da91117ba57c0491828d81bc9db921cc49eb9` | `4f33bc31eb331a80c4c6bbe42477bd742eab3c1c4455b1d0b9664117a64977a8` |
| L-mean's mean latent | `85da63360c6d3c308d4c760cfc43c2391c0beab1c3fefd3ceb5017c7759d53bd` | `699463e1bc6f489a220da1bb577736bfdacb6b29e59e147af77dcd0bc8fca81c` |
| H-sysid (coefficients in the report) | — | `671841cd581d86844c0ad14c03839cf965e912f628dd875b89c7adf3b46121fc` (`sysid.json`) |

**What O-PASS admits.** Stage T may get its own reported GO (§7 step 6). Stage O is done; it is
not repeated.

**Two anomalies, disclosed (R17.41).** Neither changes a number or the row.
- **The featurisation's command lacked `--min-free-gib 8 --board`.** §10.2 names
  `scripts/gpu_run.sh --wait --min-free-gib 8 --board --who oej:task077-<stage>`. The GO's
  command (copied from the #146 approval) had `--wait --who oej:task077-stageo` only.
  - So the shared `~/.local/state/gpu/BOARD.md` did not show the job as the holder while it ran
    (its holder line still named `task077-fscale`'s release at 08:44:07Z).
  - The lock itself was held: `gpu_run.sh` took the flock, the lock log has the start and end
    records, and the runner recorded `gpu_lock_held: true`.
  - `gpu_run.sh`'s own free-memory check used its default of 4 GiB, not 8. The runner's
    `gpu_guard` still required 8 GiB free and measured 14.98 GiB, so G-GPU's declared bound held.
  - The fix: every documented GPU command now uses the full form (§7.6, the runner's docstring,
    the featurisation probe's docstring), and each Stage T command in §7.6 is written out with it.
- **The featurise report records no peak GPU memory.** The runner did not read it. The scale
  probe of the same code at 1 995 synthetic roots measured 0.90 GiB for the process
  (`nvidia-smi`) and a torch peak allocation of 0.49 GiB (§7.4). Encoding does not depend on
  content, so the real run is expected to be the same; that is an inference, not a measurement.
  The runner now records `gpu_memory` (torch's `max_memory_allocated` and
  `max_memory_reserved`) for every GPU stage, a V included (R17.41).

### 7.6 Before Stage T's GO: fixes, the #146 approval's two items, and the Stage T plan (R17.41–R17.45)

Decided by Claude under owner delegation (DECISIONS 2026-10-05 (b), R17.41–R17.45). Code, tests,
a development probe and this document change. The frozen block and its sha256 `f28e5e2c…548d` are
unchanged, and no bar, seed, salt, cap, ceiling or row changes. The manifest re-pins the runner,
the tests and this document.

**The fixes for the GPU stages (R17.41, R17.42).**
- **Peak GPU memory (R17.41).** The runner records `gpu_memory` for every GPU stage
  (`featurise`, `train`, `scale`): torch's `max_memory_allocated` and `max_memory_reserved` at the
  stage's end, read in `run()`'s `finally`, so a V records it too. The CUDA context's own memory
  (about 0.4 GiB) is not in either number. Tests cover the reading and the V path.
- **`--board` (R17.41).** Every documented GPU command uses
  `scripts/gpu_run.sh --wait --min-free-gib 8 --board --who oej:task077-<stage>` (§10.2).
- **Artifacts from another worktree (R17.42; found while preparing this section).** Stage O wrote
  its fits' paths relative to its own worktree (`outputs/task077-readouts-1/fits/moments.npz`, and
  so on), and a Stage T job writes its checkpoint's path relative to its own. A Stage T job run
  from a fresh worktree of a later revision could not open them: the first job would have crashed
  (a V). The runner now finds each fit by its file name in the folder `--fits` names, and each
  job's checkpoint beside the job's report. Every sha256 check is unchanged: the moments, R8,
  R-plate and the mean latent are still checked against Stage O's recorded sha256s, and every
  checkpoint against its job's. `sysid.json` is only required to exist (it has no recorded
  sha256 and is not read; H-sysid's coefficients come from the O-PASS report), and the lookup
  keeps only the file name, which suits the flat `fits/` and job folders (R17.45). Checked on the real Stage O artifacts from this PR's worktree
  (read only): the chain resolved to the `task077-stageo` worktree's `fits/` folder, verified the
  14 train and val feature files (12.1 GB) in 11.4 s, and the moments (`5415eea4…`), R8, R-plate
  and the mean latent all matched.

**The #146 approval's non-blocking items 1 and 2, settled before Stage S's GO (R17.43).**
- **(1) R17.38 is an amendment.** §7.4's header and R17.35–R17.39's header said that no bar
  changes. R17.38 adds a gated bar, the 0.6 cm determinism target gate in Stage S's V rule, so
  both now say "no bar changes except R17.38's added 0.6 cm determinism gate, a post-freeze
  tightening", and R17.38 is called an **amendment**, not an erratum. The precedent the reviewer
  named is accepted: code outside the frozen block may change a gated rule without moving the
  frozen sha256 only as a disclosed, dated, data-blind tightening, like this one.
- **(2) W's CPU roll-out is bit-reproducible on identical input.** Measured, as the approval suggested, before any W exists.
  - **How.** `scripts/probe_task077_rollout_determinism.py` (development only, not pinned) runs
    the closed loop's own functions from an identical frame to the commit target:
    `lewm_c1m_v2_runtime.encode` (DINOv2 on the CPU), R-plate's reading, the 147-candidate grid,
    `rollout_plates` and `choose_from_grid` with the frozen refinement tolerance (0.25 cm). Its
    inputs are the Stage-0 debug chain only (`task077-smoke-9e772e0`; debug seeds): the 405
    frames of the debug val roots 66953, 66965, 66967 and 66975, the debug-trained `W-66992` and
    `N-66992`, and the debug R-plate, R8 and mean latent, each checked against its debug
    report's sha256; plus a **random-init model of the real architecture** (torch seed 66995,
    debug range). The arms are W, N and L-mean on the debug models and W on the random-init
    model. The candidates' commands are synthetic but deterministic (each root's logged commands
    from 405, offset by the candidate); the kinematic stand-in (MuJoCo) is not exercised, since
    the simulator's determinism is G-repro's.
  - **Processes and threads.** Seven fresh interpreters, each with the runner's thread
    environment set before NumPy loads (`MKL_DYNAMIC=FALSE`, `OMP_NUM_THREADS=6`,
    `MKL_NUM_THREADS=6`, `OPENBLAS_NUM_THREADS=16`) and one torch thread
    (`WORKER_TORCH_THREADS`), with `CUDA_VISIBLE_DEVICES` empty: three one after another, then
    four at once, as Stage S's four workers run. Each made two passes, so 14 observations.
  - **Result: IDENTICAL.** For every root and arm, all 14 observations gave the same sha256 of
    the DINOv2 tokens, the pooled latent, the reading, every roll-out's predicted plates (148 or
    149 roll-outs per decision) and the commit target; the largest target difference is 0. Every
    decision converged; two were clipped to the box (the debug N and the random-init W, both on root 66953).
  - **Report** `outputs/task077-rdet-1/report.json` in this PR's worktree (git-ignored), sha256
    `5f94100b1c62d38b09e7e5698b45f85817b82985ef1911009edb603b4bf26fde`, at `afaf50d` on a clean
    tree (clean again at the end), 662 s; the load rose from 2.53 to 3.97 during the concurrent
    processes. The scratch models were removed.
  - **What it covers.** This is the same code path on debug models and synthetic commands, not
    the trained W on S. It supports the premise of R17.38's gate: with identical frames the
    target is identical, so the 0.6 cm gate can only bind when a frame differs. It does not
    measure the stand-in's chunks or a renderer flake.

**Stage T's code path at full scale (R17.44).** The only Stage T code changes since Stage 0's
scale probe (`9e772e0`) are the feature-file check before a job reads the stores (R17.26), the
path lookup above and the GPU memory record; `lewm_c1m_v2_train.py`, the model and the frozen
block's training values are unchanged. The scale probe was run again at this PR's code
(`afaf50d`), through `scripts/gpu_run.sh --wait --min-free-gib 8 --board --who
oej:task077-tscale`, on synthetic features at the real sizes (1 500 train roots in RAM, 250 val
roots), with **50 val selections**, as many as a calibration job keeps (a job holds one copy of
the weights per selection point, 34.6 MB each, which Stage 0's 2-point probe did not show):

| measure | `task077-tscale-1` (`afaf50d`) | Stage 0 (`f4b6b48`, `a98d893`, `9e772e0`) |
|---|---|---|
| per update, median / 95th percentile | 0.169 s / 0.250 s | 0.166–0.223 s / 0.232–0.277 s |
| gather alone, one batch | 6.3 ms (in the prefetch thread; the loop waited 0.14 s in 1 000 updates) | 3.6–3.7 ms |
| one val selection (250 roots) | 3.9–12.3 s (50 selections) | 14.3–17.4 s (run 3) |
| peak process-tree PSS | **13.09 GiB** (probe 13.02 GiB, 50 kept states), against 18 GiB | 12.64–12.74 GiB (2 kept states) |
| peak GPU memory (torch) | **5.48 GiB allocated, 6.45 GiB reserved** of 15.45 GiB | 5.48 GiB (cost probe, §15) |
| the same N job twice (200 updates) | bit-identical | bit-identical |

Report `outputs/task077-tscale-1/report.json` in the worktree
`/home/huhn/develop/emai/worktrees/task077-staget-prep` (git-ignored), sha256
`b409e339bde149f2e502d7156045f7d84469aa1d1d95443e200c72d594852e9d`; its `tests` record
`outputs/task077-tprep-tests-1/report.json` (2006 passed, 37 skipped, at `afaf50d`), sha256
`fb11c0c92adc1e028c35f9ab635f867c61e3a78bac2fab36159283200bf601a6`. Stage 552 s; the synthetic
store was removed. Nothing changed the per-update cost.

**Confirmed against the frozen block and the code (R17.44).**
- **Eight jobs, each its own `gpu_run.sh --wait` slot:** cal-W and cal-N (model seed 66810,
  50 000 updates, a selection every 1 000), then, after `plan`, W and N of seeds 66800, 66801 and
  66802 at the planned budget. W and N of a seed draw the same windows
  (`SeedSequence([8109, seed])`); N trains on zero commands.
- **The budget:** `plan` sets U = clamp(5 000 · ⌈2 · max(u_sat(W), u_sat(N)) / 5 000⌉, 10 000,
  100 000) with u_sat = `select_checkpoint(curve, 0.01)` of each calibration run, and a selection
  every U/20. Checked: u_sat 1 000 gives 10 000 (every 500); u_sat 50 000 gives 100 000 (every
  5 000). No budget row. `plan` also sets G1's bars from cal-W on val at h = 60, and ends
  CAL-T-ESCALATE if the rank reference is below 0.10 (escalate, no clause), else T-PLANNED.
- **Checkpoint and selection:** each job keeps `select_checkpoint(curve, 0.01)`, the earliest
  point within 1 % of the curve's minimum, records the raw argmin and `last_two_triggered`
  beside it, and saves that checkpoint with its sha256, the corpus sha256 and the moments'
  sha256. The primary seed is chosen in Stage G (the W seed with the lowest kept val criterion).
- **The void rule, per job (§10.1, R17.16):** a V voids that job only; it is repeated once after
  a committed, pushed and recorded fix, in a new output directory; completed jobs are kept; a
  second V of the same job ends TASK-077 as INCONCLUSIVE.
- **Caps and memory:** each job's cap is 46 800 s. The worst job is a model job at U = 100 000:
  100 000 × 0.277 s + 20 × 17.4 s = 28 048 s, so the cap is 1.67 × (a calibration job's worst case
  is 14 720 s). G-memory is 18 GiB of PSS per job; the measured peak with 50 kept states is
  13.09 GiB. The machine has 30 GiB of RAM; other projects' processes are not in a job's tree.
- **Expected GPU time** (per update 0.166–0.223 s, a selection 4–17 s, plus about a minute per
  job for the checks and loading the 9.6 GB store):
  - the two calibration jobs: 2 × (50 000 updates + 50 selections) ≈ **4.7–6.7 h**;
  - the six model jobs at U = 10 000: ≈ 2.9–4.3 h, so Stage T ≈ **7.6–11.0 h** in all;
  - at U = 100 000: ≈ 27.8–37.8 h, so Stage T ≈ **32.5–44.5 h** in all (about 55 h at the 95th
    percentile update time).

  Each job holds the shared lock only for itself, so the V2D queue and other agents get the GPU
  between jobs (`--wait` queues on the flock; `--board` shows the holder).

**The Stage T commands a GO would name** (R17.44, revised by R17.45 after the #147 review). `M`
is this PR's merge commit. From a fresh clean worktree of `M`, made with `scripts/new_worktree.sh
/home/huhn/develop/emai/worktrees/task077-staget --run --from M`, run from its root, with no edit
in the worktree while any job runs:

```sh
bash scripts/run_task077_staget.sh
```

`scripts/run_task077_staget.sh` (a committed development helper, not hash-pinned) runs under
`set -euo pipefail` and checks each step's report **outcome**, not only its exit status, before
the next step starts:

| step | command (runner stage) | GPU | must end |
|---|---|---|---|
| 0 | `tests --output outputs/task077-t-tests-1` | no lock | TESTS-PASS |
| 1–2 | `train --job cal-W`, `train --job cal-N` | one `gpu_run.sh --wait --min-free-gib 8 --board --who oej:task077-staget-<job>` slot each | T-JOB-DONE |
| 3 | `plan --cal-w … --cal-n …` | no lock (G-quiet, in-run G-tests) | **T-PLANNED** |
| 4–9 | `train --job W-66800`, `N-66800`, `W-66801`, `N-66801`, `W-66802`, `N-66802`, each with `--plan outputs/task077-t-plan-1/report.json` | one slot each | T-JOB-DONE |

Every `train` and `plan` step passes `--corpus` (the sealed corpus in `task077-corpus2`),
`--corpus-sha256 ad8974b2a8b560bb974c6e0b4f90bd3f1fc79a535ebe46ef6c409bde7e4343fb`, `--features
/home/huhn/develop/emai/worktrees/task077-stageo/outputs/task077-featurise-1/features` and `--fits
/home/huhn/develop/emai/worktrees/task077-stageo/outputs/task077-readouts-1/fits`; each GPU job
also passes `--tests-record outputs/task077-t-tests-1/report.json`; every step writes
`outputs/task077-t-<step>-1/` and a sibling `.log`.

- **Why the outcome check (R17.45, the #147 review's blocking finding).** The runner exits 0 for
  every row except V, so `plan` exits 0 on **CAL-T-ESCALATE**. A script that only checked exit
  statuses would have started `W-66800`, which would take the lock, pass `gpu_guard` and then
  end V on the plan check, leaving a spurious V under the per-job void rule. The script stops
  instead, before any model job takes the lock. Likewise a `tests` step that does not end
  TESTS-PASS stops the script before `cal-W` takes the lock.
- **Tested.** `tests/test_run_task077_staget.py` runs the script with a fake runner and a fake
  `gpu_run.sh`: the eight jobs run in order, each through `gpu_run.sh --wait --min-free-gib 8
  --board`; CAL-T-ESCALATE stops it after `plan` with no model job started; a V or any other
  outcome at `tests`, a calibration job or a model job stops it there.
- Each job must end T-JOB-DONE; any other outcome or a non-zero exit stops the chain (a V is
  repeated per job, only after a recorded fix and under its own GO, §10.1). CAL-T-ESCALATE stops
  Stage T and escalates.
- One `tests` record serves every job: G-tests for a GPU job checks that its revision is HEAD
  and that it finished after HEAD's commit time.
- The two Stage O folders and the corpus are read, never written; the jobs check their sha256s.
  `sysid.json` must exist in `--fits` but is not read: H-sysid's coefficients come from the
  O-PASS readouts report itself, so it carries no sha256 check (R17.45).
- Stage T's row T-DONE is decided in Stage G's report, which reads all eight jobs (R17.24).

### 7.7 Stage T's interim record and the pause; N-66800's V and its prevention (R17.46–R17.48)

Recorded by Claude under owner delegation (DECISIONS 2026-10-05 (b), R17.46–R17.48). The pause
itself was the owner's instruction, not a ruling. The frozen block and its sha256 `f28e5e2c…548d`
are unchanged, and no bar, seed, salt, cap, ceiling or row changes. The runner, the stage modules
and the test file the manifest pins are unchanged; the manifest re-pins this document only. **No
Stage T result is claimed here.** A validation loss is a selection criterion, not a success
measure, and no LeWM controller has run in closed loop under this protocol.

**The run (R17.46).** Stage T started at `215fcce` on the reviewer's reported GO
(<https://github.com/RaaSaaR-org/open-embodied-jepa/pull/147#issuecomment-5993481678>, 11:25:50Z),
from the fresh clean worktree `/home/huhn/develop/emai/worktrees/task077-staget`, with the GO's
command `env -u C -u CS -u F -u R -u SUFFIX -u PY -u GPU_RUN bash scripts/run_task077_staget.sh`
(the R17.45 driver). The first step started at 11:26:04Z (13:26 CEST). The reports are in that
worktree (git-ignored) and are not edited. All times below are UTC; the job logs print local
time (CEST, UTC + 2).

| step | outcome | span (UTC) | report sha256 |
|---|---|---|---|
| tests | TESTS-PASS (2014 passed, 37 skipped, clean tree, at `215fcce`) | 10-05 11:26:04–11:28:39 | `928277d365522ef5a372cc44074c41c8a5bf76955294495843ea140fe17ae761` |
| cal-W (W, 66810, 50 000 updates) | T-JOB-DONE | 11:28:41–14:03:42 | `2339f5f1dc4447af6c04a7ea40e9d22708aec849bece080c57d5301a7e270cb8` |
| cal-N (N, 66810, 50 000 updates) | T-JOB-DONE | 14:03:47–16:43:37 | `42ad6cf892f14c6338608d3d9b148eff34db8e797892a98ac5d46ce08e984423` |
| plan | **T-PLANNED** | 16:43:40–16:47:26 | `7c7147601c156a793c238e34d00ee74e9855f29d19e4ee0d653882335d67cdfb` |
| W-66800 (95 000 updates) | T-JOB-DONE | 16:47:28–21:54:13 | `42195ded1ac1c41a77ed0b47e2031e4066f9bc2c2a61637ea5aa17f08e05757b` |
| N-66800 | **V** (`StageInterrupted: received SIGTERM`) | 21:54:36–22:20:42 | `bb2c442a3a610ea89c10ad22d089a58b3d27d9e6df22d2c02a16b9f3a7ec3355` |

- **The calibration jobs.** The kept checkpoint is `select_checkpoint(curve, 0.01)`, the earliest
  point within 1 % of the curve's minimum (§4.4).
  - cal-W kept update 46 000 (val criterion 0.372733; raw argmin 47 000; last point 0.375754 at
    50 000). Checkpoint `W-66810.pt`, sha256
    `5a2bbf6cba68ae3f3cd74502d76f1f9646a0fa48d389a0291d23f75d0c7b3b1c`.
  - cal-N kept update 38 000 (0.736775; raw argmin 38 000; last point 0.804274 at 50 000).
    Checkpoint `N-66810.pt`, sha256
    `c1f4778eee7bee6dc3030decdfb28d5634e43bd542411c077740ddacf26c3297`.
  - Neither job was flagged by `last_two_triggered`.
- **The plan.** u_sat(W) = 46 000 and u_sat(N) = 38 000, so 2 × 46 000 = 92 000 and **U = 95 000
  updates, a selection every 4 750** (20 points). U is below the 100 000 cap, so nothing escalates
  (no budget row exists, §4.5). G1's calibrated bars come from cal-W on val at h = 60: the rank
  reference is **0.254** (effective rank 6.55 predicted against 25.77 encoded), above the 0.10
  CAL-T-ESCALATE floor, so the rank bar is 0.12 (its floor 0.10); the std reference is 0.774, so
  the std bar is 0.38 (its floor 0.25). The rank-1, 2 and 4 truncations of cal-W's val
  predictions all fail G1 (i)–(ii), so the bar is not raised (`bar_raised` false). Their rank
  ratios are 0.039, 0.077 and 0.119. **The rank-4 truncation misses the 0.12 bar by only
  0.0013** (its std ratio, 0.692, would pass), a thin margin disclosed now. Row T-PLANNED; the
  clause does not fire.
  G-quiet read 0.90 / 1.88 at the start (bar 2.0), and the in-run G-tests passed (2014 passed).
- **W-66800.** 95 000 updates in 18 405 s, about 5.1 h (median 0.172 s per update, 95th
  percentile 0.242 s). It kept update 95 000 (val criterion 0.350399), which is also the raw
  argmin. Checkpoint `W-66800.pt`, sha256
  `891d26640e82ab8fce62c32b7d4a7a9eba99ae1babb8614777c668fe498876b8`.
  - **Disclosed: `last_two_triggered` is true for W-66800.** Its minimum is the curve's last
    point (0.362693 at 90 250, then 0.350399 at 95 000), so by the saturation flag the curve may
    still have been falling at U.
  - By §4.5 the flag never raises a budget and never changes a row; §4.5's reporting duty is
    written for an N model. It is recorded here and will be stated beside every W-versus-N
    comparison of seed 66800 in Stage G's record and the results document. That a W can be
    unsaturated at U was not foreseen in the plan (U is twice the calibration's saturation).
- **Resources.** Peak process-tree PSS 13.11, 13.17 and 13.15 GiB for the three completed GPU
  jobs, against the 18 GiB ceiling. Torch's GPU peak was 5.48 GiB allocated and 6.45 GiB
  reserved in every job. The lock log (`~/.local/state/gpu/oej-gpu_run.log`) has each job's start
  and end; each started with 15 612 MiB of 16 303 MiB free.

**N-66800's V and the pause (R17.46).** The driver started N-66800 automatically 23 s after
W-66800 ended (lock start 21:54:36Z). It logged update 4 750 with val criterion 0.837403 at
22:12:43Z (its first of 20 selection points; not a result). The owner then asked for a pause
after the first model job, to free the shared GPU for another agent ("lets do a pause after that
first run ... lets free the gpu, so other agent can use it"). The operator, the main session,
sent SIGTERM to the driver and to `gpu_run.sh` at 22:20:41Z (00:20 CEST, 2026-10-06). The runner
caught it and wrote outcome **V** (`void_reason` `StageInterrupted: received SIGTERM (SIGTERM)`,
1 566 s). It saved no checkpoint. The lock was released at 22:20:44Z (`status 1 after 1568 s`).
No Stage T process has run since, and the GPU is free.
- **Under §10.1 the V voids N-66800 only.** cal-W, cal-N, plan and W-66800 are complete; their
  reports, checkpoints and sha256s are intact and they are kept. The driver checks them on resume
  (R17.47).
- **N-66800 has one repeat left.** It needs this record of the cause and its prevention
  (a non-code cause, §10.1), merged, and its own reported GO.

**The cause and its prevention (R17.47).** **The cause was not a fault in the run.** The owner
asked for a pause after the first model job. The R17.45 driver chains every job with no point at
which it can stop between jobs, so it had already started the next one, N-66800, by the time the
pause was carried out. A stop signal is a V by §10.1, so the pause voided a job that had barely
begun. The prevention changes the driver
`scripts/run_task077_staget.sh` (a development helper, not hash-pinned). With no option, an
empty `KEPT` and no earlier attempt, it behaves as under R17.45. With its default `KEPT` (the four
pins below), a step pinned there must be found and kept, so neither a plain run nor `--resume`
without `--resume-from` can start Stage T again from scratch (the #148 review).
- **A pause file.** If `outputs/task077-staget.pause` exists when a step is about to start, the
  driver exits 0 before that step and starts nothing. `touch outputs/task077-staget.pause`
  therefore pauses between jobs, never inside one; a job that has started runs to its end.
  `--stop-after STEP` stops cleanly after a named step. The driver never removes the file: the
  operator runs `rm outputs/task077-staget.pause` before resuming. **Killing a job that holds the
  lock is a V of that job.** Stopping `gpu_run.sh` while it waits for the lock is not a safe
  pause either, because it can take the lock as it is being stopped; the pause file is the way to
  pause.
- **Resume mode** (`--resume`, `--resume-from DIR`). A step whose earlier attempt ended as
  required is kept, not re-run. A kept job's checkpoint must exist beside its report with the
  recorded sha256. The reports of cal-W, cal-N, plan and W-66800 are pinned by their sha256 in the driver. A kept plan
  must have been made from the kept calibration reports. The G-tests record is kept only from
  the worktree's own `outputs/` and only at HEAD.
- **The repeat.** A job with one V stops the driver unless `--repeat JOB` names it. The repeat
  then writes a **new folder with the next run number** (`task077-t-N-66800-2`); a V's folder is
  never reused. A second V of the same job stops the driver with "TASK-077 ends INCONCLUSIVE",
  and nothing more runs. Any other earlier outcome, a folder without a report, or an earlier
  CAL-T-ESCALATE also stops it. A search folder named twice is searched once. All of this is
  checked before the G-tests record or any job starts.
- **`--dry-run`** resolves every step and prints keep, run or stop; it starts nothing.
- **Tested.** `tests/test_run_task077_staget.py` runs the driver with a fake runner and a fake
  `gpu_run.sh`. It covers the R17.45 cases unchanged and adds these: a resume keeps the completed
  jobs and repeats the V job in `-2`; a V without `--repeat` stops; a second V stops as
  INCONCLUSIVE and refuses a third attempt; a changed checkpoint, a report that differs from its
  pin, or a plan made from another calibration stops the resume; the pause file stops between
  jobs and a resume continues; `--stop-after` and `--dry-run` work; bad arguments are refused.
- **Checked on the real reports (read only).** A dry run from this PR's worktree,
  `--resume-from /home/huhn/develop/emai/worktrees/task077-staget/outputs --repeat N-66800
  --dry-run`, kept cal-W, cal-N, plan and W-66800 (each report's sha256 matched its pin; each
  checkpoint matched its report; the plan matched both calibrations). It would run the G-tests
  record, then N-66800 into `task077-t-N-66800-2`, then W and N of 66801 and 66802 into `-1`.
  Without `--repeat` it stops at N-66800.

**What is left, and the rule that binds it (R17.48).** Five jobs remain: **N-66800's repeat (its
last allowed attempt)**, then W-66801, N-66801, W-66802 and N-66802, each at U = 95 000 with the
kept plan. At W-66800's 18 405 s each, that is about **25.6 h of GPU** (five slots of about 5.1 h;
the 46 800 s cap per job is unchanged).
- **A second V of N-66800 ends TASK-077 as INCONCLUSIVE** (§10.1), whatever its cause, a pause
  included. A first V of any other job voids only that job and leaves it one repeat, after its own
  record.
- **Pauses from now on** use the pause file or `--stop-after`, never a signal to a running job.
- **A preflight failure is a V too.** The runner turns every guard failure into V, including
  G-tests, G-memory and the GPU guard at a job's start. A guard failure at the start of
  N-66800's repeat would therefore be its second V. Before that GO, the operator confirms the
  following: the new worktree's G-tests record is TESTS-PASS at `M`, the tree is clean, nothing
  else holds GPU memory (at least 8 GiB free), the machine's other jobs leave room for the
  18 GiB PSS ceiling, and no pause file is left in the new worktree.
- **The command a GO would name.** `M` is this PR's merge commit. First make a fresh worktree:
  `scripts/new_worktree.sh /home/huhn/develop/emai/worktrees/task077-staget2 --run --from M`.
  Then run, from its root:
  `env -u C -u CS -u F -u R -u SUFFIX -u PY -u GPU_RUN -u REV -u KEPT -u PAUSE_FILE bash
  scripts/run_task077_staget.sh --resume-from
  /home/huhn/develop/emai/worktrees/task077-staget/outputs --repeat N-66800`.
  - The new worktree writes its own G-tests record at `M`, because G-tests checks HEAD. The
    repeat runs at `M` "from scratch at the fix's revision on the same seeds, in a new output
    directory" (§10.1).
  - The runner, the stage modules, the training module and the frozen block are byte-identical
    at `215fcce` and at `M`, so the repeat and the later jobs train with the same code as the
    kept jobs. Only the recorded protocol-document sha256 differs, as it has between stages.
  - The old worktree `task077-staget` is read only, and is not removed until Stage G has read
    its jobs.
- **When it runs is the owner's decision.** The owner wants the GPU left free now. The repeat
  needs this PR merged, a reported GO naming the command above, and the owner's go-ahead for the
  GPU. Nothing has been launched.

## 8. Gates, bars and rows

Intervals are reset- (or root-) clustered bootstrap percentile intervals, 10 000 resamples, 95 %,
salt 8106. Every bar is labelled by its source: **measured** (a tolerance or ceiling measured in
this task), **definitional** (a ratio below 1, a test level), **carried** (a TASK-065/066 bar
reused unchanged; an allocation, not calibrated here) or **allocation** (a declared margin).

### 8.1 Stage O (train + val roots, 5 outer folds, salt 8104)

| gate | condition | source |
|---|---|---|
| **O1** precision | the upper 95 % bound of the cross-fitted 8 × 8 readout's median plate error at r (= c_plate) ≤ τ_commit | **measured** (τ_commit from K0) |
| **O3** prior | the upper bound of median(e_8×8) / median(e_constant prior at r) < 1.0 | **definitional** |
| **O4** plate hidden | the lower bound of the same fold's readout's median error on the plate-hidden renders at r > τ_commit | **measured** (O4's form against τ_commit) |

Reported only: the learning curve (1/4, 1/2, 3/4, all of each fold's fit roots, nested, salt 8108)
with TASK-075 §7's falling-curve guard; the 87.5th percentile; **the 4 × 4 readout at r on the same
folds (§3, caveat 2)**; R-plate's error at 405; H-sysid's error at r. Rows, first match: V;
**O-ARM-KEYED** (O4 fails: escalate, no clause); **O-NO-BAR** (O1 or O3 fails: escalate, no clause;
with a still-falling curve it says so); **O-PASS**. The record read 0.478 cm (8 × 8) on 1 024 roots;
that is a scale reference, not a bar.

### 8.2 Stage G, the 8 × 8 dynamics at h = 60 (gate split; every gate on all three seeds)

The window set **E60** is one window per gate root: the encoded frame at 405 rolled forward with
that root's 60 executed commands, against the encoded frames 406–465. Ratios are of summed
normalised errors (TASK-065's MSE in metric-scale units), bootstrapped over gate roots.

| gate | passes when | source |
|---|---|---|
| **G1** no collapse | (i) collapsed fraction (dimensions with std < 0.01 of the metric scale) ≤ 0.05; (ii) effective-rank ratio ≥ B_rank and std ratio ≥ B_std of W's predicted against the encoded latents at h = 60; (iii) the lower 95 % bound of (rank ratio W − rank ratio N) > 0 | (i) **carried** (TASK-065); (ii) **measured**: B = max(floor, ½ × the calibration W's ratio on val at h = 60, rounded down to 0.01), floors 0.10 and 0.25 (TASK-066's rule); the combined G1 must fail rank-1, 2 and 4 truncations of the calibration W's val predictions, or the bar is raised and disclosed (TASK-066's owner condition); (iii) **definitional** |
| **G2** copy-last | upper bound of MSE(W) / MSE(copy-last) ≤ 0.8 | **carried** (TASK-065/066), an allocation; copy-last's own numbers are reported (at h = 60 the plate moves about 8.5 cm, so copy-last is weak and G2 is expected to be easy) |
| **G3** no-action | upper bound of MSE(W) / MSE(N) < 1.0 | **definitional** |
| **G4** action sensitivity | lower bounds of MSE(W, wrong) / MSE(W, true) and MSE(W, zero) / MSE(W, true) ≥ 1.10 (wrong = another gate root's commands, salt 8110) | **carried** (TASK-065), an allocation |
| **G5** the predicted plate at r | (a) the upper 95 % bound of R8's median plate error on W's predicted latent at r ≤ τ_commit; (b) the upper bound of median(e_W) / median(e_N) < 1.0 | (a) **measured** (τ_commit); (b) **definitional** |

- **The effective rank** on E60 is computed from the centred 250 × 250 Gram matrix (the same
  nonzero eigenvalues as the covariance, so it is exact and cheap; with 250 windows it is capped at
  249 for predictions and targets alike). TASK-066's 256-direction projection is not needed at this
  window count; the comparative bootstrap (iii) resamples gate roots (2 000 resamples, salt 8106)
  and recomputes both ratios.
- **Why G5 (a) uses τ_commit unscaled.** Under the rule, a constant error ε in the predicted plate
  moves the converged aim by ε/(1 − κ) = ε/1.5, and τ_commit is measured in aim error, so the
  matching tolerance on ε would be about 1.5 τ_commit. Using τ_commit itself is conservative by
  that factor; the scaled number is reported, not gated.
- **G5 on the stand-in chunks (reported beside G5, not gated):** the same statistic with W rolled
  from each gate root's 405 frame under the **stand-in** chunk of that root's own committed aim
  (`primitive_chunks` from the logged 405 state), the commands W ranks in closed loop, against the
  executed commands G5 uses. The difference between the two is the stand-in's cost to W.
- **Reported only:** every gate at h = 16 and 30 from 405; the encoded readout's error on the gate
  split (c_plate there) and W minus it; the constant prior; copy-last's reading; each offline aim
  error and predicted count (§7, step 7).

Rows, first match: V; **H-GATE-FAIL** (any of G1–G4 fails on any seed: escalate, no clause; it
closes nothing and never reaches back over TASK-066's 4 × 4 result; the design note's declared
remedy is a reviewed amendment with `action_chunk` or `predictor_step_embedding`, each needing its
own preregistration); **G-NO-BAR** (G1–G4 pass on all seeds and G5 fails on any: escalate, no
clause); **G-PASS**.

### 8.3 Stage D

As §7 step 8: **L-DEV-STOP** or **D-PASS**.

### 8.4 Stage S, the gated rows

**The bars and tests.**
- **G-bar:** W(S) ≥ 56/64. This is τ_commit's own bar fraction, 28/32, which an aim error within
  τ_commit keeps by τ_commit's definition (TASK-076 G1's form). Its feasibility is K0's
  N_K(0) ≥ 30/32. Power (exact binomial, P(X ≥ 56 of 64)): 0.59 at a true rate of 0.875, 0.73 at
  0.89, 0.86 at 0.906, 0.98 at 0.9375 and > 0.99 at 0.969.
- **G-NI:** the lower bound of the paired 95 % interval of W − C > −8/64, where C is the better of
  H-rule and H-sysid on S. **δ = 8/64 is an allocation** (the design note's proposal, R15.6's
  label; not calibrated). Simulated power (this draft; 2 000 trials, 2 000 resamples, C at 30/32):
  at W = C, 0.78 with independent outcomes and 1.00 with maximally shared outcomes; at W = C − 2/64,
  0.48 and 0.85; at W = C − 4/64, 0.20 and 0.42. **Non-inferiority is demanding**: W must be about
  as precise as H-rule. **Two corrections from the review of #142, which Stage 0's simulation must
model:** (i) the test's size: at the margin (W = C − 8/64) the paired percentile bootstrap rejected
3.3 % of the time (3.0–3.5 % for C at 30–32/32), against the nominal one-sided 2.5 %, so it is
slightly anti-conservative at n = 64 near the ceiling; (ii) the comparator is the better of two
arms chosen after S, which lowers the power: with two comparators at 30/32 sharing half their
outcomes, 0.72, 0.39 and 0.16 at 0, −2 and −4/64 (independent outcomes; the reviewer's
simulation). Stage 0 repeats the simulation with K0's counts, the max-of-two comparator and the
size at the margin, and reports both; an exact or score interval may replace the percentile
interval only by a declared change before the freeze.
  **Stage 0's simulation (R17.19; 10 000 trials per configuration, the real salt-8106 estimator;
  [stage-0 record](apple_lewm_c1m_v2_stage0.md) §2).** At the margin (W = C − 8/64) G-NI passed in
  **2.4–3.8 %** of trials with one comparator and **0.9–3.5 %** with the better of two chosen after
  S (C at 30–32/32, independent or nested outcomes, the two comparators sharing none, half or all
  of their outcomes), against the nominal 2.5 %. Power with the better of two: 0.69–1.00 at
  W = C, 0.35–0.86 at −2/64, 0.14–0.43 at −4/64. The percentile interval is kept; no exact or
  score interval is declared. K0 runs no comparator arm, so "K0's counts" could enter only through
  its ceiling, and K0-PASS needs N_K(0) ≥ 30/32, which makes min(30/32, N_K(0)) always 30/32: a
  re-run in K0 would repeat a configuration simulated here. It is therefore not run (R17.24); the
  Stage-0 grid at 30–32/32 is the simulation.
- **G-N, G-shuf, G-mean, G-rand:** W > arm, exact one-sided McNemar p < 0.01 on the paired resets.
  The minimum separation is 7 discordant pairs, all W's (TASK-076 R8.7). From the record's proxies
  (8, 8, 12 of 32 against a 32/32 ceiling) every test's feasibility was 1.000 at the ceiling; W
  will sit below the ceiling, so K0 re-states it with W's bar in place of the ceiling.
- **The voids:** **S-VOID-CEILING** if H-final(commit)(S) < 56/64 (the condition's ceiling fell
  below the bar on S).
- **"Detectably", defined (R17.15).** Every twin row reads the **same declared test**, the exact
  one-sided McNemar test of W > arm at p < 0.01, plus the paired reset-clustered 95 % interval of
  W − arm (§8, salt 8106):
  - a twin test **passes** when its McNemar p < 0.01;
  - W is **detectably no better** than an arm when that arm's McNemar test fails **and** the upper
    bound of W − arm is below **+7/64**, the minimum separation at which the McNemar test can pass
    at all (7 discordant pairs, all W's). W's advantage is then detectably smaller than any
    advantage the declared test could certify;
  - a failed test whose upper bound is ≥ +7/64 is a **miss within noise**.

  Non-inferiority uses the same interval: W is **detectably inferior** when the upper bound of
  W − C is below −δ = −8/64. A clause row therefore needs a failed declared test **and** a
  detectable shortfall, and **no clause row can fire on a run whose four McNemar tests and G-NI all
  pass**. Stage 0 simulates the clause's false-fire probability at a true twin advantage of
  exactly +7/64 and at δ, as R8.14 did, and writes it into the protocol before the freeze.
  **Stage 0's values (R17.19):** at a true twin advantage of exactly +7/64 (W at 56 or 60/64, 0–2
  reversed pairs per 64), L-NO-GAIN fires falsely in **2.3–3.1 %** of trials per twin, and in
  9–12 % if all four twins sat exactly at the boundary at once; at a true W − C of exactly −8/64,
  L-INFERIOR fires falsely in **0.9–2.0 %** with one comparator and **0.9–3.1 %** with the better
  of two. At +7/64 the McNemar test itself passes only 29–56 % of the time per twin, so a twin at
  the boundary most often gives L-TWIN-NEAR (an escalation).

| row (first match) | condition | consequence |
|---|---|---|
| **V** | the void rule (§10.1) | one repeat of the stage after a recorded fix |
| **S-VOID-CEILING** | H-final(commit)(S) < 56/64 | escalate, no clause, no claim |
| **L-NO-GAIN** | for at least one of N, L-shuf, L-mean, L-rand, the McNemar test fails **and** W is detectably no better (upper bound of W − arm < +7/64) | **the clause fires** (§11) |
| **L-INFERIOR** | W is detectably inferior beyond δ (upper bound of W − C < −8/64) | **the clause fires** (§11) |
| **L-PASS** | G-bar, G-NI and the four McNemar tests all pass | **the primary claim, "LeWM-driven closed-loop success"**; the secondary claim is then reported |
| **L-TWIN-NEAR** | at least one McNemar test fails, and every failed one is a miss within noise (upper bound ≥ +7/64) | escalate, no clause, no claim (the TWIN-NEAR / PRED-NEAR convention, R17.15) |
| **L-NEAR** | G-NI fails, but W is not detectably inferior beyond δ | escalate, no clause, no claim |
| **L-BAR** | otherwise (G-bar fails, with G-NI and the tests passing) | escalate, no clause, no claim: the place family itself sits below the bar on S |

**Declared departures from the design note's clause trigger (§4.1, "W fails a twin or random
test").** The note fires the clause on any failed twin test. Here a failed test fires it only when
the shortfall is detectable; a miss within noise is L-TWIN-NEAR and escalates (R17.15, the
TWIN-NEAR/PRED-NEAR convention of TASK-076 and R8.14). The first draft's L-TWIN-WEAK and its
bootstrap-lower-bound trigger are withdrawn.

Any repeat after L-TWIN-NEAR, L-NEAR or L-BAR needs fresh seeds and its own ruling. Reported in
every row: every arm's count, every paired difference with its discordant counts and interval,
H-read's and H-now's counts, the offline predictions of §7 step 7 against the S counts, and the
secondary claim's test.

## 9. Blind and prior-only baselines against every bar

| bar | baseline | where it sits |
|---|---|---|
| O1, O4 | none; τ_commit is measured | O4's plate-hidden renders test the readout's source |
| O3 | the constant prior at r | the record: 2.66 cm |
| G2 | copy-last | measured in the run |
| G3, G1 (iii), G5 (b) | N | trained in the run |
| G4 | W with wrong or zero commands | measured in the run |
| G-N, G-shuf, G-mean, G-rand | the trained twins and the random choice | measured on S |
| G-NI | H-rule, H-sysid | measured on S (30/32 each in M-F6, development) |
| G-bar, S-VOID-CEILING | H-final(commit) | privileged; the ceiling of the arm family |

## 10. Void rule, guards, caps and memory

### 10.1 Void and repeat (R16.10's form)

- A stage that ends **V** (any guard below, a crash, a cap, a stop signal, a CUDA allocation
  failure) is void. Its report is kept, its sha256 recorded and the cause disclosed.
- **At most one repeat per stage**, only after a fix that is committed, pushed and recorded (the
  cause, the fix commit, the void report's sha256), from scratch at the fix's revision on the same
  seeds, in a new output directory; a non-code cause still needs a committed record of the cause and
  its prevention. **A second V of the same stage ends TASK-077 as INCONCLUSIVE** (escalate to the
  owner). Vs in different stages do not add up (R17.16).
- **Stage T is voided and repeated per job** (R17.16). Each of its eight jobs (the two
  calibration runs, the six models) is its own unit with its own report: a V voids that job only,
  which is repeated once after a recorded fix; completed jobs whose reports, checkpoints and
  sha256s are intact are kept (strict CUDA determinism makes a re-run of an identical job
  bit-identical; Stage 0's scale probe shows this on a debug seed). A second V of the **same job**
  ends TASK-077 as INCONCLUSIVE. The budget rule runs only after both calibration jobs complete.
- A V after a stage's `first_outcome_utc` is never read as an outcome.
- A completed stage is the record; no stage is repeated for its result; no tracked file is edited
  in a run's worktree during a run (R14.9's lesson).
- **Debug runs** happen only on committed code, only on 66900–66999, and nothing in them is read.

### 10.2 Guards (any failure is V)

- **G-tests (new; #141 review note 1):** for CPU stages the runner **runs the full suite itself**
  (`pytest` from the worktree root, at HEAD) before its first simulation and records the summary
  line, exit status and timestamps; any failure is V before the outcome boundary. For GPU jobs,
  where running the suite would hold the lock, the runner **verifies** `pre_run_tests.json`
  instead: the recorded revision equals HEAD, the exit status is 0, the summary shows no failure
  or error, and its finish time is later than HEAD's commit time. The GO comment also cites a green
  CI run for that commit.
- **G-sentinel (new; #141 review note 2):** every report field for a check or row the stage has not
  reached holds `"not evaluated"`; no row ladder is evaluated with a missing input.
- **G-hash and G-frozen:** the tracked tree is clean at the start and the end; the frozen-block sha
  and every pinned file match (including TASK-076's 84 pins and C1's and C1-M's code files).
- **G-repro, G-anchor, G-threads, G-quiet** (TASK-076's: G-repro's eight facts; the DINOv2 anchor;
  pinned thread settings; 1- and 5-minute load ≤ 2.0 at start for CPU stages).
  - **The pinned threads (R17.29).** They are `MKL_DYNAMIC=FALSE`, `OMP_NUM_THREADS=6`,
    `MKL_NUM_THREADS=6` and **`OPENBLAS_NUM_THREADS=16`**.
  - The 16 is not a typo for 6. It is the owner ruling of 2026-09-29 (TASK-073,
    `wm_critic_v2.THREAD_ENV`): OpenBLAS's default of one thread per logical CPU on this 16-thread
    PC, pinned so that it cannot drift.
  - TASK-074, 075, 076 (including its K0) and C1-M ran with the same environment. K0 ran with it
    as declared.
  - A test checks that the runner sets exactly this environment before NumPy loads.
- **G-split:** every featurisation file a stage reads matches the featurise report's sha256
  (R17.26); every fit used downstream (normalisation, W, N, R-plate, R8, H-sysid, L-mean's mean)
  is fitted on train only; val only selects and calibrates; the gate split is read only by Stage G;
  the corpus manifest's sha matches. **The one declared exception:** Stage O's *admission*
  estimates (O1, O3, O4 and the learning curve) are cross-fitted over train + val roots (5 outer
  folds), as in the C1-M record; no fold of them is used by a later stage.
- **G-privileged:** `task_truth_in_controller` is 0 for W, N, L-shuf, L-mean, L-rand, H-rule and
  H-sysid on every attempt.
- **G-finite, G-memory, G-disk:** every feature, loss, prediction and statistic finite; process-tree
  PSS ≤ 12 GiB (≤ 18 GiB for a Stage T job, whose train-split feature cache is about 9.6 GB);
  ≥ 10 GiB free disk throughout (≥ 25 GiB at the start of Stages C and O).
- **G-GPU:** every GPU job runs through `scripts/gpu_run.sh --wait --min-free-gib 8 --board --who
  oej:task077-<stage>`, and the runner calls `run_tools.gpu_guard(report, min_free_gib=8,
  require_lock=True)` and records torch's peak GPU memory in `gpu_memory` (R17.41). Resident services (the GR00T server, other projects' queues) are never
  stopped or reconfigured. **CPU stages and EGL (R17.33, Erratum 2026-10-05):** K0, C, D, S, the
  readouts, plan and gates render with EGL on the GPU's driver but run no CUDA job; they do not
  take the shared GPU lock and are not GPU jobs. "No GPU" in a GO means no CUDA job and no lock.

### 10.3 Caps (wall time; a cap is a V, never an escalation)

Kept by Stage 0 (R17.19) and by the pre-freeze probes (R17.28). Each is at least 1.5 × the
measured or scaled worst case ([stage-0 record](apple_lewm_c1m_v2_stage0.md) §3–§4 and §7):
- **Stage T:** the job worst case at the cap is 27 700 s, so 46 800 s is 1.69 ×.
- **Stage O's readouts:** `readouts_core` at the real sizes took 98.9 s on synthetic features,
  plus 7.9 s for the feature-file check, with a peak PSS of 2.96 GiB. The whole probe stage,
  including G-tests, took 285 s. 7 200 s is 67 × the core.
- **Stage C:** the worst case for 2 000 roots, scaled from 100 debug roots, is about 1 400 s. That
  is 2 000 roots at the slowest attempt, 3.24 s per root on 6 workers, plus writing, the cohort
  estimates, setup and G-tests. 14 400 s is 10 ×. Peak PSS was 8.46 GiB, against the 12 GiB
  ceiling. **Erratum 2026-10-05 (R17.30):** that peak covered the cohort estimates at 100 seeds
  only. At 2 000 seeds, the unstreamed estimates reached 12.9 GiB and voided the first run (§7.2).
  With the streamed estimates, the seed preparation measures 7.75 GiB at 2 000 seeds.
  *(R17.35–R17.36:)* Stage C's repeat measured a peak of 8.38 GiB over the whole stage, and took
  1 234 s, 0.09 of the cap (§7.3).
- **Stage O's featurisation (R17.37, Erratum 2026-10-05 (b)):** the runner's own `featurise`
  stage on a synthetic corpus of the sealed corpus's size (1 995 roots) took 212.7 s on CUDA,
  G-anchor included, with a peak PSS of 1.76 GiB and 0.90 GiB of GPU memory (§7.4). 3 600 s is
  17 × that, so the cap stays.

The caps:
K0 7 200 s; Stage C 14 400 s; Stage O featurisation 3 600 s and readouts 7 200 s; **each Stage T
job 46 800 s** (13 h; one calibration run, or one W or N model: at most 100 000 updates, whose
worst case without the Stage-0 storage fix is 30 000 s at 0.30 s per update, so the cap is
1.56 × that; with the fix, measured in Stage 0, 27 700 s at the 95th-percentile update time,
§12); Stage G 7 200 s; **Stage D 7 200 s and Stage S 21 600 s, each its own cap** (R17.23);
per closed-loop attempt 300 s.

## 11. The abandonment clause and its scope (R17.10)

**It fires on L-NO-GAIN or L-INFERIOR only**, that is, only when W is detectably no better than a
twin or random choice, or detectably inferior to the best non-world-model arm by more than δ, as
§8.4 defines "detectably" with the declared tests (R17.15). It does not fire on CAL-ESCALATE,
CORPUS-ESCALATE, O-ARM-KEYED, O-NO-BAR, CAL-T-ESCALATE, H-GATE-FAIL, G-NO-BAR, L-DEV-STOP,
S-VOID-CEILING, L-TWIN-NEAR, L-NEAR, L-BAR, V or INCONCLUSIVE.

**What closes.** R15.8's form: the design note's §4.1 scope, **extended by the move**:

> "LeWM aim selection with a single aim committed at 405 under the declared reactive-plate rule
> (κ = −0.5, L = 2, s1 = 525) on v2, **with the declared post-pick plate move of radius ρ\* = 4 cm
> (the disc, which also covers the −y half-disc, at radii ≤ 4 cm)**, from onboard 112 px frozen
> DINOv2 pooled tokens, with TASK-066-family predictors."

**One declared narrowing, with its reason (a departure from R15.8; R17.10).** The scope does **not**
cover a 4 × 4 pooled latent on a corpus larger than C1-M's 1 024 roots. Reason: this protocol never
runs W on 4 × 4; 8 × 8 was chosen only because 4 × 4 missed H-read's allowance on 1 024 roots while
its readout curve was still falling (#141 caveat 2), so a clause fired on an 8 × 8 W would close a
4 × 4 W that was never tested on adequate data. Every other pooled grid, every readout of the
predicted latent and every TASK-066-family recipe variant (capacity, depth, training horizon,
batch, budget, loss weighting, selection) stays inside the note's scope, as R15.8 has it. The first
draft's narrowing to "rolled 60 steps and read by a dual-ridge plate readout" is withdrawn: it had
no stated reason.

**If the clause fires, the results document states the restored scope plainly** (the #142
approval's note 3). With the design note's scope restored (R15.8), one 8 × 8 run read by a
dual-ridge readout closes, under this condition, **every other pooled grid** (except 4 × 4 on a
corpus larger than 1 024 roots, above) **and every readout of the predicted latent, including
trained readout heads**, not only the 8 × 8 grid and the dual-ridge readout that were run. The
results document says this in those words, beside the row.

**What does not close:** a 4 × 4 latent on a larger corpus (above); the full, unpooled token grid
(not "pooled tokens"); history longer than one, `action_chunk` or `predictor_step_embedding` (the
design note's declared remedies, outside "TASK-066-family" as TASK-066 defined it: history one, no
chunking); a fine-tuned or another encoder; other views or resolutions; other κ, L, commit steps,
move distributions or radii above 4 cm; C1 without the move (its own record ended
C1-TWINS-ESCALATE, which closes nothing); C2; TASK-076's results; the LeWM backend; v2; the product
goal.

## 12. Compute estimate

| stage | estimate | basis |
|---|---|---|
| K0 | **9.7 min CPU, measured** (579 s) | 352 attempts plus G-tests and G-repro on 6 workers; attempts took about 4.2 s (proxies) to 8.2 s (look-ahead arms) median |
| C | **about 20–23 min CPU, scaled from 100 debug roots (R17.28)** | 0.48 s of wall time per root on 6 workers (attempts 2.6 s median, 3.24 s maximum), plus the cohort estimates (3.6 s per 100 seeds), setup and G-tests; about 0.6 MB per root, 1.2 GB on disk. The draft's 45–60 min was an estimate |
| O | 10–20 min GPU, 20–40 min CPU (*estimate*) | about 130 000 frames through DINOv2 on CUDA; ridges at 24 576-d by dual form |
| T | *(R17.44, with the selections: about 7.6–11.0 h at U = 10 000 and 32.5–44.5 h at U = 100 000; §7.6)* **a scenario band, not bounds: about 7–10 h to about 32–43 h of GPU in eight jobs, with the Stage-0 storage fix, measured** (R17.19; up to about 58 h without it) | **Stage 0's scale probe** (Stage T's own code, synthetic features at the real sizes, two runs): 0.166–0.223 s per update median, 0.232–0.277 s at the 95th percentile, the slice gather 3.7 ms per batch in the prefetch thread. **Low end:** U = 10 000: 160 000 updates, about 7.4–9.9 h. **At the cap** U = 100 000: 700 000 updates, about 32–43 h (54 h at the 95th percentile). The draft's band (7–8 h to 31–33 h, from the probe's 0.161 s compute alone) is superseded. **Without the fix**, the probe's fancy-indexed gather (0.55 s for batch 64, about 0.14 s for batch 16) without overlap gives about 0.30 s per update and up to about 58 h at the cap; at U = 60 000, 22–38 h (460 000 updates at 0.17–0.30 s) |
| G | 20–40 min CPU | 250 gate roots × 6 models × (true, wrong, zero) roll-outs on one CPU thread each (0.055 s per 60-step roll-out, §15) |
| D | about 15 min CPU | 80 attempts; a W-family attempt adds about 7–10 s of CPU roll-outs and the stand-in chunks |
| S | 1–2 h CPU | 64 resets × 10 arms, H-read the slowest |

**The storage fix, implemented in Stage 0 (R17.18).** The probe's gather fancy-indexed 383 MB
at about 0.7 GB/s. Stage 0 stores each root's 65 frames contiguously, so a training window is one
slice, and fills the next batch in a prefetch thread while the GPU trains; its scale probe measures
the real gather path, and the caps are reset from it.

**Stage T is the main cost and is several times PLAN.md's 4 × 4 sizing** (6.5 h, 10–11 h worst
case). It comes from the 8 × 8 grid (2.6 × the 4 × 4 update at T = 16, §15) and from training at
the gated horizon (§4.2). It runs one job at a time under the shared lock, so other projects get
turns between jobs. Disk: frames about 2–5 GB, the 8 × 8 band cache 12.8 GB (2 000 × 65 × 98 KB),
full tokens at 405 0.8 GB; the main checkout has 95 GB free.

## 13. Risks, stated now

- **The horizon.** 60 recursive steps of an 8 × 8 latent whose dynamics no task has gated; G5 and
  the non-inferiority test need W's predicted plate to be about as good as a perfect roll-out read
  by the same readout. H-GATE-FAIL or G-NO-BAR is a live outcome.
- **Non-inferiority is demanding.** H-rule and H-sysid sat at 30/32 in development, and at a true
  W two resets per 64 below them G-NI passes only 48–85 % of the time (§8.4). L-NEAR is a likely
  row if W is slightly less precise than the rule.
- **H-read's own margin was zero.** Perfect dynamics with the 8 × 8 readout met the 4/64 allowance
  exactly on one run (§3). W can only add error. The larger corpus may help the readout (both
  curves were still falling), but that is a hope, not a measurement.
- **The secondary claim is not expected** (R15 §5.7).
- **Cost.** Stage T is about 7–43 h of GPU with the storage fix (measured in Stage 0; about 54 h
  at the 95th-percentile update time), up to about 58 h without it (a
  scenario band, §12); a V in Stage T is expensive, and the void rule
  allows one repeat per stage, and per job in Stage T (R17.16).
- **The proxies were privileged.** The trained twins may be stronger than the proxies (L-mean
  through a mean latent that still encodes a "typical" plate, N through the box), though every
  proxy cleared +8/32 by at least 7 resets (lower bounds +15 to +32).
- **The scorer and the condition are imposed**: the single commitment and the reactive rule are
  simulator laws; neither transfers as such to Arena or the real G1 (design note §4.1).

## 14. Points settled after the first review (R17.17, decided by Claude under owner delegation)

The first draft left five points open. The independent review of #142 (REQUEST CHANGES at
`26894ca`) recommended each, and they are settled:

1. **T = 60 with batch 16 is kept** (§4.2). The decisive argument, from the review: with T = 16
   and windows starting at 403–407 in a corpus that keeps frames 403–467, no transition after
   step 423 would ever be trained, so the plate's motion from 423 to 465 would be untrained
   recursion.
2. **Three model seeds are kept** (TASK-065/066/074's practice; every gate on all seeds). If cost
   ever forces a cut, it goes through the gather (§12), not the seeds.
3. **The shared twin grid is kept**, with each arm's clip-binding fraction reported (§5.1) and the
   claim wording of §1 ("ranks inside a box built from a non-LeWM readout").
4. **δ = 8/64 is kept as an allocation** (R15.6's label). Its only anchor is its τ-curve reading,
   about one τ of extra aim error. The Stage-0 simulation models the test's size and the
   max-of-two comparator (§8.4).
5. **G2 and G4 stay gated**, escalating without the clause (H-GATE-FAIL). G4 is the
   action-sensitivity diagnostic the repository's rules require; G2 at h = 60 is cheap. Both are
   also reported at h = 16 and 30.

## 15. Development measurement made for this draft (not a result; gates nothing)

**What:** `scripts/probe_task077_cost.py`, the GPU and CPU cost of TASK-066's token predictor at
`token_grid` 8 on **synthetic** features and commands (random normal features, uniform commands):
no corpus, no rendered frame, no reset seed. Torch seeds 66990 and 66991 (the debug range). It sets
no bar; it sizes §4.2, §10.3 and §12.

| run | revision | where | outcome |
|---|---|---|---|
| `outputs/task077-cost-1` | `39c6967` | GPU, through `gpu_run.sh --wait` | **crashed** (CUDA out of memory at 8 × 8, T = 60, batch 64); no report written; disclosed. The probe was changed to record such a configuration and go on |
| `outputs/task077-cost-2/report.json` | `2dc6106`, clean tree | RTX 5080, through `gpu_run.sh --wait` | sha256 `0669bf24d9ae4200fed434e9d8ce0b7266e2470fe3e0dd451c97901fd750378b` |
| `outputs/task077-cost-cpu-1/report.json` | `5218857`, clean tree | CPU, one torch thread, no GPU | sha256 `f760f32fc96cdfad621752fee3957510791f5f463cb6ec6642000aba879a28e1` |

(All in the worktree `/home/huhn/develop/emai/worktrees/task077-prereg`, git-ignored.)

| configuration (training update, LeWM token predictor) | seconds per update | peak GPU memory |
|---|---|---|
| 4 × 4, T = 16, batch 64 (TASK-066's recipe) | 0.042 | 1.59 GiB |
| 8 × 8, T = 16, batch 64 | 0.110 | 5.79 GiB |
| 8 × 8, T = 60, batch 64 | **out of memory** (> 14.7 GiB allocated) | — |
| 8 × 8, T = 60, batch 32 | 0.234 | 10.86 GiB |
| **8 × 8, T = 60, batch 16 (this draft)** | **0.161** | **5.48 GiB** |
| 8 × 8, T = 30, batch 32 | 0.119 | 5.48 GiB |
| 4 × 4, T = 60, batch 64 | 0.170 | 5.70 GiB |

Roll-outs (60 steps, 8 × 8): GPU 0.377 s for 147 candidates, 0.032 s for one, 0.605 s for 256;
**CPU, one thread: 7.43 s for 147 candidates, 0.055 s for one.** A host batch gather from a
20 000-frame float32 table took 0.55 s for 64 windows of 61 frames and 0.033 s for 64 windows of
17 frames (no prefetch). The update timings reuse one batch, so the gather is not in them. TASK-066
measured 0.313 s per 8 × 8 update on MPS (T = 16, batch 64); the CUDA figure here is about a third
of that.
