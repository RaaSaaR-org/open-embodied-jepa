# Apple→Plate plate-readout twin v2: the place aims at the plate read from the image, plus a moving-plate headroom check (TASK-076)

**Status: DRAFT, not frozen.** This document is a preregistration draft for review. Nothing in it
has run. It has no frozen block, no manifest and no code yet. Nothing from any cohort below may be
simulated before the conditions in §5 are met: K0 runs only on a reviewer's reported GO for K0,
and nothing after K0 runs before the frozen protocol is merged on an independent reviewer's
reported APPROVE. Every bar below is either a formula fixed here whose value K0 measures, or a
measured quantity. No bar is carried from an earlier task without being measured again.

- Task card: `.mc/tasks/todo/TASK-076-*.md`. Rulings: [DECISIONS.md](../DECISIONS.md), decision
  2026-10-02, R1–R4 and R7, and decision 2026-10-02 (b), R8 (this draft's scope).
- Plan context: [docs/PLAN.md](../PLAN.md).
- Learned Apple→Plate status (the canonical sentence, R7): learned Apple→Plate on the frozen v1
  MVP benchmark (TASK-020) is 0/150 per backend (`native_jepa` and LeWM). On `apple-to-plate-v2`,
  the behaviour-cloning/DAgger policy P-3 (an MLP on a frozen DINOv2 readout, trained on
  demonstrations from the privileged scripted expert e9; not a world model) scored 40/40 counted
  successes on the held-out cohort C against 39/40 for its random-init encoder control R-3, so
  TASK-072 M2 is M2-FAIL on G3, and cohort C is no longer held out. No LeWM-driven controller has
  run in closed loop on v2 yet; LeWM's only closed-loop Apple→Plate runs are on v1, with 0
  successes.

---

## 0. Where LeWM sits, stated first

**LeWM is not in this task's control loop.** No arm of TASK-076 runs a world model. The
hypothesis arm H-twin aims e9's scripted place primitive at a ridge readout of the plate position
from frozen DINOv2 tokens of the current onboard frame. P-3, a behaviour-cloned policy, does the
pick. Nothing in this task can produce, or be reported as, a LeWM-driven success.

**What LeWM gets from it.** The frozen DINOv2 patch tokens that R-plate reads are the encoded
latent of TASK-066's LeWM token predictor (`token_dynamics.MODEL_CONFIG`; TASK-074 used the same
latent, `apple_lewm_planner_v2.md` §1). So this task measures three things that the next LeWM task
needs and cannot honestly guess:
1. **The encoded-latent plate ceiling** c_plate (Stage O). This is the readout error before any
   prediction, on the latent a LeWM planner would roll forward. A later LeWM task's
   predicted-latent plate bar B must satisfy c_plate ≤ B ≤ τ_re, the rule of TASK-075 §6.3. This
   is the TASK-074 lesson: bars come from measured ceilings.
2. **The perception baseline** H-twin's closed-loop count. Any later LeWM place planner must tie
   it, as TASK-074's bar T did.
3. **Whether a condition exists in which the place target must be predicted** (Stage K-pred).
   Under ruling R2, a LeWM plate-target task is admissible only when paired with such a
   condition. K-pred measures, with no world model, whether a moving-plate condition leaves room
   for prediction over the best non-predicting arm, with a feasible ceiling.

**Can Option 1 alone move the LeWM goal?** No. With a static, visible plate a world model is not
needed for the place (TASK-074 protocol §9b: "A pass would not show that a world model is
needed"). An H-twin pass would show that a frozen-encoder readout can aim the place under
TASK-074's condition, and that is all. That is why this draft adds K-pred (ruling R8, decided by
Claude under owner delegation). With K-pred, TASK-076's outcome decides whether TASK-077, the
first LeWM controller in closed loop on v2, can be preregistered and under which condition. See §9.

## 1. The question and the claim

**Primary question (Stages O, D, S/U).** On `apple-to-plate-v2` under TASK-074's condition (the
plate moves 9 cm to the robot's right at step 300), does P-3's pick followed by e9's place aimed
at a frozen-DINOv2 single-frame plate readout of the onboard 112 px camera (H-twin) succeed at the
place family's bar, and does it beat an image-free prior?

**Why this quantity.** TASK-075 found that no view reads the apple-minus-plate offset within
τ = 1.0 cm, but its reported-only plate readout reads the plate position to a median of
0.493–0.682 cm over the four views (onboard 112: 0.493 [0.455, 0.511] cm). The offset error tracks
the apple term, 2.211–3.122 cm (`apple_obs_ceiling_v2_results.md` §2.3, §3.4). τ is defined as an
error in the aimed plate target (§2 there). Two development results point the same way, and
neither is gated:
- TASK-074's Probe C: a kernel-ridge plate readout trained with far frames read 0.55–0.69 cm in
  median, and e9's place aimed at it scored 31/32 (9 cm) and 30/32 (12 cm). That probe used
  P-truth frames and readouts fitted on other seeds of the same probe cohort, on spent
  development seeds 54700–54999 (`apple_lewm_planner_v2.md` §2).
- TASK-074's train-stage R-plate (dual ridge on full tokens at the decision frames, 1 350 rows):
  val median 0.417 cm, p90 0.816 cm, reported only (`apple_lewm_planner_v2_results.md` §1.5).

**The claim a TWIN-PASS would make.** "Under TASK-074's 9 cm condition, P-3's pick followed by
e9's scripted place aimed at a frozen-DINOv2 ridge readout of the plate in the current onboard
112 px frame reaches the calibrated place bar on fresh gated resets and beats an image-free
prior." **It would not say:**
- anything about LeWM or any world model;
- that the pick, or the whole episode, is learned end to end (the pick is P-3, the place is
  scripted);
- that pretrained features are needed (the random-init floor is reported, not gated);
- anything about v1, its 0/150, Arena or the real G1.

**Secondary question (Stage K-pred; no world model).** Is there a moving-plate condition on v2
in which aiming at the plate's *final* position succeeds, while aiming at where the plate is *now*
fails by a clear margin, even with perfect perception? That margin is the room a predicting
controller could have.

## 2. Carried unchanged

- The v2 scene, e9 (`RestingPlaceExpert(release_pitch_rad=0.45, release_dx=0.015)`) and its
  schedule `(130, 80, 45, 150, 100, 50, 30, 50, 30, 60)`: the transfer starts at 405, the lower at
  505 (the target is frozen from here) and the place ends at 725 (`lewm_planner_v2.py`
  `TRANSFER_START`, `LOWER_START`, `PLACE_END`).
- The counted success (T71-R1/R2: at rest after a latched grasp and place) and the attempt of 740
  commands plus the settle.
- P-3: `checkpoints/task072-first-policy-v2-linux/run-1/P-3.pt`, sha256 `7988162d…60be8`, with
  its post-look readout refitted exactly as TASK-072 run-1 fitted it (G-repro).
- TASK-074's condition: `lewm_planner_v2.shift_vector` (the −y arc 225–315°, salt 7413),
  `condition_reset` (re-draw salt 7425), the plate-shift hook at step 300 and `SHIFT_BLOCKED`.
  These salts are part of the condition's definition and are carried; the seeds are new, so the
  draws are new.
- `place_planner.PlannerController` and `PlacePrimitive`, with the decision steps 405, 421, 437,
  453, 469 and 485. Mode `twin` (H-twin) and mode `truth` (H-handover) exist and are used
  unchanged. New modes (H-clock, H-stale, H-floor, and K-pred's arms) are added in new modules;
  no file pinned by an earlier manifest is edited.
- The sealed views store `apple-far-shift-v2-views` (manifest sha256 `ea627a8f…4d77`) and its
  source corpus `apple-far-shift-v2` (manifest sha256 `fe7ab915…b134bd`): 270 train + val roots,
  of which 253 are read. The 30 test roots are never simulated, rendered or decoded.

## 3. Arms

### 3.1 The primary closed loop (cohorts D, S, U)

| arm | what the place is aimed at, at each decision 405–485 | role | gating |
|---|---|---|---|
| **H-twin** | R-plate's reading of the current onboard 112 px frame (pretrained DINOv2 full tokens, dual ridge) | hypothesis | yes |
| H-handover | the true moved plate (privileged) | the family ceiling | feasibility only |
| **H-clock** | the per-decision-step median of the true plate position over K0's resets under the same condition (fitted once in K0; reads no image at run time) | the image-free control | yes (G2) |
| H-stale | P-3's post-look plate estimate (read before the move; no later image) | the condition's check | S-VOID-CONDITION |
| H-floor | R-plate refitted on the seed-0 random-init DINOv2 (`pretrained_encoder.random_init`) | the encoder floor | reported |
| P-stale | P-3 unchanged | the incumbent | U no-harm (G3) |

- **R-plate for the closed loop** is fitted once, after the Stage O boundary, on the full tokens
  of every read root's decision frames that exist (225 train + 28 val roots), λ by 5-fold inner
  CV grouped by root. It is TASK-074's `lewm_planner_v2_offline.fit_r_plate` recipe on more rows.
  Its sha256 is recorded, and every worker checks it.
- **Encoding in the loop** is on the CPU in the worker, batch size 1, as in TASK-073/074.
- **H-clock** is the strongest image-free target we could name for this condition: it knows the
  condition's distribution, from privileged labels on development resets, but reads nothing at
  run time. It replaces the card's "clock-prior arm". It is fitted on K0, so it is not fitted on
  any gated reset.

### 3.2 K-pred, the moving-plate headroom check (cohorts M-a, M-b; no world model)

**The condition family M.** A harness hook moves the plate every step from step s0 to step s1,
at constant velocity, along a direction drawn by TASK-074's −y arc rule (salt 7606), by a total
distance D. It uses the same `body_pos` write plus `mj_forward` as the step-300 hook, outside
every controller's `act()`. Moves are refused, as in TASK-073's hook, when the plate would touch
anything but the table; a refused move ends the attempt as a counted failure. The plate is static
after s1.

| cell | s0 | s1 | D | plate distance still to go at the last decision (485) |
|---|---|---|---|---|
| M-a | 405 | 525 | 12 cm | 12 × 40/120 = 4.0 cm |
| M-b | 405 | 525 | 9 cm | 9 × 40/120 = 3.0 cm |

s1 = 525 lies inside the lower phase (505–555), before the open phase (585–635). **To be
confirmed in the Stage-0 smokes:** no apple–plate contact before s1 on any smoke attempt. If a
smoke finds contact, s1 moves earlier and D is rescaled to keep the same distance to go. That
change is made before the freeze and disclosed.

| arm | aimed at, at each decision 405–485 | role |
|---|---|---|
| H-final | the plate's true position at s1 (privileged) | the ceiling |
| H-now | the plate's true position at the decision step (privileged, non-predicting) | the cost of not predicting, with perfect perception |
| H-twin | R-plate's reading of the current frame | the cost of not predicting, with this perception |
| H-cv | a least-squares constant-velocity line through H-twin's readings so far, extrapolated to s1 | the non-world-model predictor |

- H-cv knows the condition's declared form: constant velocity, and the stop step s1. That is
  privileged knowledge of the condition's structure, not of the state, in the same way that
  H-clock knows the condition's distribution. A world model would have to learn it from data.
- At 405 H-cv has one reading and aims at it (H-twin's choice). From 421 on it fits a line.

## 4. Seeds and cohorts

A repository search on 2026-10-02 (src, scripts, docs, benchmarks, tests, configs, `.mc`, and the
open and archived remote branches) found no seed use in 56000–56999. The only hits for 56xxx were a
float in a manifest. It also found no use of 7601–7620 as a seed constant. Every range below is
disjoint from every range in `obs_ceiling_v2.FORBIDDEN_RANGES` (20000–20049 up to the TASK-074
block 54000–54999, including cohort C 45300–45339 and the TASK-071/072 block 51000–52199), from
TASK-075's block 55000–55999, and from the Arena seeds of #123 (50200–50231, in TASK-070's block).

| seeds | cohort | stage | use |
|---|---|---|---|
| 56000–56031 | K | K0 (development, before the freeze) | τ re-measured; H-handover's ceiling; H-clock fitted; H-stale and H-clock levels |
| 56040–56071 | M-a | K-pred (development) | the 12 cm moving-plate cell |
| 56080–56111 | M-b | K-pred (development) | the 9 cm moving-plate cell |
| 56120–56135 | D | D (development, stop rule only) | H-twin and H-handover, 16 resets |
| 56200–56263 | S | gated | 64 shifted resets, every §3.1 arm once, paired |
| 56300–56331 | U | gated | 32 unshifted resets: H-twin, P-stale, H-handover |
| 56900–56999 | smoke | Stage 0 | mechanics only; nothing in them is read |

**RNG salts** (new): 7601 planted-error direction in K0's τ curve; 7602 outer folds; 7603 inner
folds; 7604 bootstrap; 7605 learning-curve subsets; 7606 moving-plate direction. Carried as part
of the condition: 7413 (shift direction) and 7425 (reset re-draw). Cohort C is not used.

## 5. Stages and stop rules

1. **Stage 0 (the preregistration PR).** The protocol, the frozen block, the code, the tests and
   the smokes on 56900–56999. The tests include: every seed range above against the forbidden
   ranges; no iterative training anywhere in the stage code (§7); the scale probe calls the
   stage's own function (`run_tools.scale_probe`, F15); the runner imports no other runner script
   (`tests/test_no_runner_imports.py`); and `run_tools.assert_local_import`, `install_guards` and
   `gpu_guard` are used.
2. **Stage K0, calibration (development; after a reviewer's reported GO for K0, before the
   freeze).** It runs on cohort K, the simulator only, on the CPU:
   - **τ_re.** TASK-075's pre-committed re-measurement (`TAU_REMEASURE`, its §6.3): the same arm
     (H-handover with a planted target error), condition, levels {0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5}
     cm, direction rule (salt 7601 here) and rule. τ_re is the largest level L such that every
     level ≤ L reaches ≥ 28/32. That is 288 attempts; TASK-075's τ took 265 s for the same number
     on 6 workers.
   - **The ceiling** N_K(0) is the level-0 count.
   - **H-clock** is fitted on K's true plate positions at the decision steps, then run on K, with
     H-stale beside it (64 attempts).
   - **Stop: CAL-ESCALATE** if level 0 is below 28/32 (τ_re undefined) or N_K(0) < 30/32 (the
     gated bar of 56/64 is then not feasible). Nothing is frozen; the owner, or Claude under the
     owner's delegation, rules.
   - The K0 values (τ_re, the τ curve, N_K(0), H-clock's fitted targets, H-clock's and H-stale's K
     counts) are written into the frozen block. K0 runs no hypothesis arm, so no bar sees an
     H-twin number.
3. **Freeze.** The PR is merged on an independent reviewer's reported APPROVE.
4. **Stage O, offline plate admission (after the freeze, on a GO).** It reads the sealed views
   store: no render and no new simulation.
   - **Featurisation** on CUDA through `scripts/gpu_run.sh` (§8), with a CPU anchor check.
   - **Cross-fitted R-plate** (5 outer folds by root, salt 7602), on each read root's onboard 112
     px decision frames, gives per-frame plate errors. The same is done for R-plate-floor, the
     random-init features.
   - **The offline clock prior** is the per-step median plate position over the outer-training
     roots.
   - **The plate-hidden check** uses the stored plate-hidden onboard frames at the eval steps
     421–501.
   - **`first_outcome_utc`** is written before the first fit.
   - Then the gates O1–O4 (§6.1), and the closed-loop R-plate fit (§3.1).
   - **Reported only:** the 87.5th percentile, the per-step errors, the signed mean along x and y
     (τ is anisotropic), and the τ-curve prediction (TASK-075 §6.4's mapping, with K0's curve).
     The prediction is written before any closed-loop attempt.
5. **Stage D, development closed loop (on a GO; it requires the recorded O-PASS).** H-twin and
   H-handover run on cohort D. **Stop: TWIN-DEV-STOP** (escalate, no clause) if H-twin < 12/16 or
   H-handover < 14/16. Nothing is refitted on D.
6. **Stage S/U, gated (on a GO; it requires the recorded D-PASS).** Cohorts S and U, every arm
   once per reset, paired, from a clean worktree of the merged revision. It includes a determinism
   re-run of H-twin on the first four S seeds; any difference is a V (TASK-074 §8.3).
7. **Stage K-pred (on a GO; it requires the recorded O-PASS; independent of D and S/U).** Cohorts
   M-a and M-b, all four §3.2 arms. It produces its own row (§6.3).
8. **Results PR.** Every arm is reported, and the privileged ceilings and H-clock are labelled as
   not learned. An independent reviewer checks every restated number.

## 6. Gates and rows

Intervals are reset-clustered bootstrap percentile intervals: 10 000 resamples, 95 % (salt
7604). Paired closed-loop differences are reported with their discordant counts.

### 6.1 Stage O, offline admission (all four must hold)

| | condition | why |
|---|---|---|
| **O1** precision | c_plate := the upper 95 % bound of cross-fitted R-plate's median error ≤ τ_re | the readout is within the measured tolerance (TASK-075 A1's form) |
| **O2** the decision tail | the upper bound of the 87.5th-percentile error ≤ 2 τ_re | for scale, the hand crop's 87.5th percentile is 1.36 cm against a 0.682 cm median (`apple_white_plate_dev.md`); this bounds the tail to a level the τ curve still tolerates at about 23/32 (2 cm) if τ_re = 1.0. The draft asks the reviewer whether to keep O2 gating or make it reported |
| **O3** clock prior | the upper bound of median(e_R-plate) / median(e_clock) < 1.0 | the image adds over a prior that reads none |
| **O4** plate hidden | the lower bound of R-plate's median error on the plate-hidden frames > τ_re | the readout reads the plate, not e9's arm, which in this corpus is aimed at the (possibly mis-aimed) plate. **This matters more in closed loop than offline:** from 405 on, the arm moves towards H-twin's own last reading, so a readout keyed on arm pose would confirm itself |

Reported, not gating: R-plate / R-plate-floor (the random-init floor tied P-3 in M2, and this task
does not ask whether pretraining helps); per-step errors; the signed means; the τ-curve
prediction.

**Rows, first match:** V; **TWIN-OFF-ARM** if O4 fails (escalate, no clause: the readout's
source is unclear); **TWIN-OFF-FAIL** if O1, O2 or O3 fails (the clause fires, §7); otherwise
**O-PASS**.

### 6.2 Stage S/U, the gated rows (first match)

The bars:
- **G1 (the place bar):** H-twin(S) ≥ 56/64. This is τ's own bar, 28/32 = 87.5 %, which a target
  error within τ_re keeps by τ's definition, measured in K0. Its feasibility is K0's N_K(0) ≥ 30/32.
  The paired difference H-twin − H-handover is reported beside it, with its interval.
- **G2 (the image-free control):** H-twin(S) > H-clock(S), one-sided exact McNemar p < 0.01 on the
  paired resets. There is no tuned margin, as with TASK-075's ratio bars.
- **G3 (no harm, U):** H-twin(U) ≥ P-stale(U) − 2/32 (TASK-074's G5).

| row | condition | consequence |
|---|---|---|
| **V** | the void rule (§7) | one repeat after a reviewed fix |
| **S-VOID-CONDITION** | H-stale(S) > 8/64, or H-handover(S) < 56/64 | escalate: the condition does not need a reading, or the ceiling fell below the bar on S |
| **TWIN-HARM** | G3 fails | the clause fires (§7) |
| **TWIN-PASS** | G1 and G2 pass | **the only claim row** (§1) |
| **TWIN-PRIOR** | G1 passes, G2 fails | escalate: an image-free prior is about as good, so this condition does not test perception |
| **TWIN-FAIL** | otherwise | the clause fires (§7) |

### 6.3 Stage K-pred, its own row (first admitted cell, in the order M-a, M-b)

A cell is admitted if all three hold:
- **K-P1 (feasible ceiling):** H-final ≥ 30/32 (TASK-074 K1's ceiling bar).
- **K-P2 (prediction matters with perfect perception):** H-final − H-now ≥ +8/32 (TASK-074 K1's
  headroom bar).
- **K-P3 (prediction matters with this perception):** H-final − H-twin ≥ +8/32.

Reported, never bars: H-cv, and H-final − H-cv with its interval.

| row | condition | consequence (for the plan, not for TASK-076's claim) |
|---|---|---|
| **PRED-ADMIT(cell)** | a cell passes K-P1–K-P3 | TASK-077 may be preregistered under that cell, as a task change declared under R2 |
| **PRED-NONE** | no cell passes | no moving-plate LeWM place task is preregistered on v2 under this family (§7) |

If H-cv ≥ H-final − 2/32 on the admitted cell, a hand-written extrapolator already reaches the
ceiling. TASK-077 could then only claim that LeWM *can drive* the place under a prediction
condition and tie the extrapolator, not that it is needed. This is stated now, before any number.

### 6.4 Blind and prior-only baselines against every bar

| bar | baseline | where it sits |
|---|---|---|
| O1, O2 | none; τ_re is measured | the plate is in view, and TASK-075 reported 0.493 [0.455, 0.511] cm on onboard 112 |
| O3 | the offline clock prior | fitted on the outer-training roots |
| O4 | plate-hidden frames | stored in the views store |
| G1 | H-handover | privileged; it is an upper bound for the arm family |
| G2 | H-clock | measured in the run |
| S-VOID-CONDITION | H-stale | TASK-074 K1: P-stale 0/32 at 9 cm |
| K-P2, K-P3 | H-now, H-twin | measured in the run |

## 7. Budgets, void rule and abandonment clause

**Budgets.** This task trains nothing iteratively. Every readout is a closed-form ridge with λ
chosen by inner cross-validation, so there is no update count, no saturation rule and no
last-two rule. TASK-074's escalations came from exactly such a rule: its BUDGET block (cap 60 000,
calibration 60 000 updates, factor 2), the same as TASK-073's, fails
`run_tools.check_budget` with "cap 60000 < 120000: factor 2 x calibration 60000 can be requested,
so the rule escalates by design" (checked on main at f41b94f). The frozen code will carry
`TRAINING_BUDGET = None`, and a test will assert it. If a reviewed amendment ever adds iterative
training, its BUDGET block must pass `run_tools.check_budget`, select checkpoints with
`run_tools.select_checkpoint` (the earliest point within tolerance), and apply the last-two rule
only through `run_tools.last_two_triggered`. A test will check that too.

**Caps** (wall time; exceeding one is a V, never an escalation): 7 200 s per runner invocation,
300 s per attempt (TASK-074's), and 1 800 s for Stage O's featurisation. Memory: process-tree
PSS ≤ 12 GiB (`run_tools.MemoryWatch`). GPU: §8. Each cap is at least 5 times the estimate
in §10. No stage has a row that a budget can trigger: every escalation in §6 comes from a measured
result.

**Void rule.** A stage is V on any of: a stop signal (`run_tools.install_guards`, which records
the first signal); a cap; a CUDA allocation failure; a pin or frozen-sha mismatch; a decoded test
root; a privileged read in a non-privileged arm; a failed determinism re-run; G-repro failing. A V
after the stage's outcome boundary is not read as an outcome. One repeat from scratch is allowed
after a reviewed fix (TASK-074 §7 precedent). A second V escalates.

**The abandonment clause** fires on TWIN-OFF-FAIL, TWIN-FAIL or TWIN-HARM. **Its scope:**
"aiming e9's place primitive, after P-3's pick, at a single-frame frozen-DINOv2 ridge readout of
the plate from the onboard 112 px camera, under TASK-074's 9 cm condition on `apple-to-plate-v2`:
no further readout variant of this formulation (pooled or full tokens, λ grid, crop, colour) is
preregistered without new evidence of a different kind (a place primitive with its own feedback,
temporal aggregation of readings, or a new view or sensor)." Then the next step is TASK-075's
Option 2 (a place servo; R3).

**K-pred's own clause.** PRED-NONE closes preregistering a LeWM plate-target place task on v2
under the tested moving-plate family (constant velocity from 405 to s1, D ∈ {9, 12} cm) without
new evidence of a different kind. It does not close the LeWM backend, the v2 task or the product
goal.

**Never closed by this task:** the LeWM backend, DINOv2 as an encoder, the v2 task, Arena and the
product goal.

## 8. Platform, GPU plan under the shared lock

- **Platform.** The Linux PC (RTX 5080, 16 GB), MuJoCo 3.13.0, `MUJOCO_GL=egl`, strict CUDA
  determinism in the main process. Simulation on the CPU with 6 workers, 1 torch thread each.
  The pinned thread environment (G-threads) and the quiet-machine rule (G-quiet: 1- and 5-minute
  load ≤ 2.0 at start) are carried.
- **What uses the GPU.** Only Stage O's DINOv2 featurisation: about 3 000 onboard 112 px frames
  (253 roots × 6 visible decision frames, plus 6 plate-hidden frames per root at the eval steps). TASK-075 featurised 10 626 frames
  per view, plus its full-token Gram, in 76–92 s, with a peak of 0.85 GiB allocated and 0.96 GiB
  reserved against a 3.0 GiB cap (`apple_obs_ceiling_v2_results.md` §1.3). Estimate: under 2
  minutes and under 1 GiB.
- **How.** `scripts/gpu_run.sh --wait --min-free-gib 4 --board --who oej:task076-O -- uv run
  --no-sync python scripts/run_plate_twin_v2.py offline …` (the script name is a placeholder until
  the code exists). The runner calls `run_tools.gpu_guard(report, min_free_gib=1.0,
  process_cap_gib=3.0, require_lock=True)`, so it refuses to start outside the lock. The lock is
  the machine-wide flock `~/.local/state/gpu/lock`, shared with other projects; `--wait` queues
  behind the current holder instead of failing.
- **Everything else runs on the CPU** (K0, D, S/U, K-pred: simulation, plus CPU encoding with
  batch 1 in the workers, as in TASK-073/074). These stages do not take the GPU lock. They are
  started only when G-quiet holds; a neighbour's long CPU job delays them, and does not void them.
- **The resident GPU services** (another project's queue, the GR00T server) are never stopped,
  killed or reconfigured.

## 9. What would count as "a LeWM-driven closed-loop success", and how this task feeds it

**Definition** (fixed here, for TASK-076 and every later task in [PLAN.md](../PLAN.md)):

> A **LeWM-driven closed-loop success** is a counted success (T71-R1/R2, `apple_at_rest_v0` after
> a latched grasp and place) on a gated, fresh cohort, in an arm in which every control decision
> of the phase LeWM is said to drive is chosen from the output of a LeWM predictor rolled forward
> from the encoded current observation, with no privileged read at run time. The arm's row must
> also beat, with a preregistered test, the same controller with an action-blind predictor, a
> scene-blind predictor and a random choice. It must be reported beside the best non-world-model
> arm for the same phase. The claim names the phase: "LeWM-driven place-target selection" is not
> "a LeWM policy".

**TASK-076 produces none.** No arm in it uses a predictor. A TWIN-PASS is a frozen-encoder
perception result, labelled as such. It must not be described as progress in LeWM successes.

**What it hands the next task (TASK-077, conditional):**
- c_plate and τ_re, which bound TASK-077's predicted-latent plate bar: c_plate ≤ B ≤ τ_re;
- H-twin's and H-cv's counts on the admitted cell, as the tie bars TASK-077 must not fall
  clearly below;
- the admitted cell, or PRED-NONE.

If the row is PRED-NONE, Option 1 has not advanced the LeWM goal beyond calibration. The
recommended next step (PLAN.md, branch B) is then a task change in which the target's motion
depends on the robot's own action, so that an action-conditioned predictor has something to
predict that no hand-written extrapolator gets for free.

## 9b. Limitations (stated plainly)

- **A static plate needs no world model.** The primary question cannot say anything about LeWM.
- **G1 is a point bar on 64 resets.** τ = 1.0 cm rested on one run at exactly its bar (28/32,
  Wilson [0.72, 0.95]); K0 re-measures it once, on 32 resets.
- **The readout is fitted on e9 corpus frames and run on P-3-pick frames.** The corpus is TASK-074's
  e9 plan with mis-aims and noise, not P-3's carry. Probe C's 5.53 cm in-distribution-only error
  shows how much the training frames matter. Stage D is the cheap check for this before the gated
  seeds are spent.
- **The closed loop has feedback.** From 405 on, the arm moves towards H-twin's last reading, and
  the frames then show the arm there. O4 checks offline that the readout reads the plate. It
  cannot check the closed loop's own feedback; the per-decision readings are logged.
- **τ is anisotropic** (−y errors fail earlier). The signed errors are reported, not gated.
- **K-pred's H-cv knows the motion model.** A pass for a later world model against it would be a
  tie at best on this family, which §6.3 states in advance.
- **One encoder, one view, one input size, one floor seed, one corpus.**

## 10. Compute estimate (from measured stage times)

| stage | attempts or work | device | estimate | basis |
|---|---|---|---|---|
| K0 | 288 + 64 attempts | CPU, 6 workers | about 6 min | TASK-075 τ: 288 attempts in 265 s |
| O | featurisation; cross-fitted ridge on at most 1 518 rows (253 × 6) | GPU < 2 min, then CPU | about 10 min | TASK-075 readouts: featurisation 76–92 s per view; cross-fit 221–228 s per view for more readouts |
| D | 32 attempts | CPU | about 2 min | as K0, plus the CPU encoding |
| S/U | 384 + 96 attempts | CPU | about 10 min | as K0, plus six encodings per H-twin and H-floor attempt |
| K-pred | 2 × 32 × 4 = 256 attempts | CPU | about 5 min | as K0 |

Total machine time is under 1 h. The GPU is used for under 2 minutes.

## 11. Deviations from the card's proposed shape (disclosed)

| card | this draft | why |
|---|---|---|
| closed loop on "32 fresh development resets", bar 28/32 | D: 16 development resets as a stop rule; then gated S (64) and U (32) | the gated count must come from fresh seeds after a development check; 64 resets narrow the interval at the same bar fraction by about √2; the cost is minutes |
| "an image-free clock-prior arm" | H-clock fitted on K0's resets under the same condition | the strongest image-free target; a corpus-wide clock prior mixes unshifted and 3–12 cm roots |
| bar 28/32 carried | G1 56/64 = τ's bar, feasibility from K0's measured ceiling; O1's bar is τ_re measured in K0 | the TASK-074 lesson: bars come from measured ceilings |
| — | Stage K-pred | ruling R8: without it, TASK-076 cannot inform the LeWM goal beyond calibration |

## 12. Open questions for the reviewer (to be resolved before the freeze)

1. O2: keep it gating (the tail bar of 2 τ_re), or make it reported?
2. K-pred's cells: is s1 = 525 safe from apple–plate contact (the Stage-0 smoke decides), and
   should a third cell use a later start (s0 = 300)?
3. Should U also run H-handover (now included: 32 more attempts), or only H-twin and P-stale?
4. Whether a TWIN-PASS should be followed by a hand-crop variant: not planned. The hand crop's
   plate tail is worse (87.5th percentile 1.36 cm against 0.91 cm on onboard 112).
