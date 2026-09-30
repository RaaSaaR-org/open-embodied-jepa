# Apple→Plate LeWM planner v2: LeWM chooses where the place primitive puts the apple after a far plate move (TASK-074)

**Status: preregistration (Stage 0).** No TASK-074 cohort seed (K1, R, D3, corpus, S, U) has
been simulated. Only these have run:
- the design probes of §2, on development seeds 54700–54999, which are now spent;
- the smokes of §10, on smoke seeds 54650–54699, which are mechanics only.

As in M2 and TASK-073, every stage re-renders (frame only, nothing scored) the post-look frames of
TASK-072 run-1's perception seeds and D2. It refits run-1's readouts and reproduces run-1's facts
(G-repro). That is M2's accepted precedent, not a use of those seeds.

**Status lines**
- **Learned Apple→Plate on the frozen benchmark is still 0 successes** (the v1 MVP benchmark,
  0/150 per model). Nothing in this task changes that number.
- **No control line is primary.** This protocol tests one formulation: a LeWM world model
  choosing where a scripted place primitive puts the apple, after P-3's learned pick. It does not
  make LeWM, sampling-based planning or behaviour cloning the project's control approach.
- **The plate move is a simulation-only diagnostic condition on `apple-to-plate-v2`**, as in
  TASK-073. It is not a new benchmark, and a teleported plate is not a real-world disturbance.
- **What an L-plan success would be:** "P-3's learned pick (BC/DAgger on privileged e9 labels)
  followed by a LeWM planner that chooses where e9's scripted place primitive puts the apple". It
  is not LeWM driving the whole episode.
- **What even the best row, L-PASS, would show:** that a LeWM planner beats P-3 and the blind
  world-model twins under this condition, while staying within 6/64 of a readout-only twin. It
  would not show that LeWM is necessary: the design probes predict that the readout twin does
  about as well (§2).

Manifest: `benchmarks/manifests/apple-lewm-planner-v2.json` (the frozen block, the stored cohort
values and digests, the corpus plan's digest, the pins, the smoke record).
- **Design:** `src/embodied_jepa/lewm_planner_v2.py`.
- **New modules:**
  - `place_planner.py` (the place primitive, the planner controller, the ranking branch, P-far's
    controller, the re-run comparison);
  - `lewm_planner_v2_runtime.py` (the simulation workers);
  - `lewm_planner_v2_offline.py` (features, training, statistics; a copy of TASK-073's offline
    module with this protocol's constants).
- **Runner:** `scripts/run_lewm_planner_v2.py`, with the modes `preflight`, `smoke`, `k1`,
  `corpus`, `train`, `pfar`, `rank`, `decide`, `d3` and `gated`.
- **Tests:** `tests/test_lewm_planner_v2.py`.
- **No existing file is modified.** Every file behind an existing implementation hash, an M2 pin
  or a TASK-073 pin keeps its bytes; a test compares all of TASK-073's pins. The run guards come
  from `src/embodied_jepa/run_guards.py` (#111).

---

## 0. The owner's decisions and what was decided under delegation

**The owner's rulings of 2026-09-30, on the design proposal, verbatim:**

| decision | the owner's answer | what it fixes here |
|---|---|---|
| D1, the claim | **"Beat P-3, tie readout (Recommended)"** | L-plan must beat P-3 even when P-3 is given the true plate (G1). It must beat the action-blind, scene-blind and random twins (G2–G4). It must stay no worse than the readout arm H-twin by more than 6/64 (T). Beating H-twin is not required |
| D2, what LeWM plans over | **"Put-down spots (Recommended)"** | a global grid of release targets, executed by e9's place phases (§3.3) |
| D3+D4, condition and scope | **"Right-side move, place only (Recommended)"** | the plate moves to the robot's right (−y, §3.2) at step 300, by 9 or 12 cm chosen by K1. P-3 does the pick, and LeWM plans the place |
| D5, the BC control | **"Yes, add it (Recommended)"** | the 300-root corpus plus P-far, a retrained BC policy, as a reported control (§3.5) |

**Owner delegation (2026-09-30).** The owner delegated every remaining ruling. Each choice below
is the option the design would mark (Recommended), and is labelled **"decided by Claude under
owner delegation, 2026-09-30"** (`lewm_planner_v2.DELEGATED_CHOICES`):

- **The cost.** J(g) = ‖R_off(ẑ_{t+16}) − o_f‖, where o_f = o*(501), e9's median release offset.
  TASK-073 used a per-step o*(t + 16). That is an unshifted-trajectory prior, and under a 9–12 cm
  move it rewards chunks that imitate the unshifted path rather than reach the plate. TASK-073
  K0's privileged H-sim with that cost already fell below P-truth at 6 cm (20 against 26).
- **The grid.** Coarse 7 × 7 at 3 cm, then fine 5 × 5 at 0.75 cm: 74 candidates per decision.
  The design proposal's 2 cm coarse grid did not cover a 12 cm move.
- **The decisions.** Six, every 16 commands from 405 to 485. The target is frozen from 505, the
  lower phase.
- **The re-draw rule for resets** (§3.2). It replaces 1–2 % of resets on which the plate cannot
  move to the right.
- **The bars.** K1's, D3's and the gated bars in §6.
- **The P-far recipe** (§3.5) and **the re-run rule** (§8.3).
- **The memory measurement at full corpus scale**, taken in the train smoke (§8.2).

## 1. The question and the claim

**Question.** On `apple-to-plate-v2`, after P-3 has picked the apple and the plate has moved 9 or
12 cm to the robot's right at step 300: does a LeWM predictor over frozen DINOv2 4 × 4 patch
tokens produce more counted successes when it chooses where e9's place primitive puts the apple?
It is TASK-066's latent, unchanged, at its validated horizon h = 16. The comparisons are:
- P-3 itself, given the true moved plate (P-truth);
- the same planner with an **action-blind** predictor (L-N, the no-action model: with the tie rule
  it keeps the stale target);
- a **scene-blind** predictor (L-shuf: W started from another reset's latent);
- a **random choice** among the same 74 targets (L-rand);
- and without harming unshifted resets.

Beside these, a **readout-only twin** (H-twin: the same primitive aimed at a direct plate readout
of the frame, with no world model) is reported and bounds the claim (T).

**The claim a pass would make.** "A LeWM planner choosing where a scripted place primitive puts
the apple beats the learned policy P-3 under a far plate move, and the world model's prediction,
not the choice mechanism, carries the gain." It carries the labels of the status lines. **It
would not say:**
- that LeWM drives the pick or the whole episode;
- that LeWM is needed rather than a readout;
- anything about v1, its 0/150, a real disturbance, another camera or another policy.

## 2. The design probes (disclosed: the bars were set after seeing them)

The design proposal ran cheap probes on **development seeds 54700–54999**. **These seeds are
spent and are refused by every TASK-074 guard.** The bars and the −y restriction in §6 were set
**after** these numbers had been seen. That is allowed for development seeds, and it is disclosed
here, as TASK-073 did in its §10.

**How the probes were run:**
- The probe code was scratch: the design worktree at `4d3e596`, not committed.
- It used TASK-073's direction rule with the direction salt 7401, and one run of 32 resets per
  cell, with no interval.
- P-3's post-look estimates came from a readout fitted on 192 new in-distribution frames, not
  run-1's refit.
- The e9 place in the probes was e9 shadowed from step 0 and handed over at 405. In this
  protocol, the place primitive is built fresh at 405 from declared scene constants (§3.3).

**Probe A: larger moves and far plates, any direction** (cohort 54900–54931)

| condition | e9 (true plate) | P-truth | P-stale (readout) | e9 aimed at the reset readout | readout plate error, median |
|---|---|---|---|---|---|
| far plate at reset, 6 / 9 / 12 cm | 29 / 19 / 15 | 23 / 5 / 0 | 7 / 0 / 0 | 6 / 0 / 0 | 3.5 / 7.1 / 10.7 cm |
| shift at step 300, 8 / 10 / 12 cm | 21 / 19 / 15 | 14 / 7 / 3 | 0 / 0 / 0 | — | — |

Split by direction, e9 fails on the +y side (towards the midline), for example 1/18 on +y at
12 cm, but succeeds on every −y reset (14/14 at 12 cm). Hence the right-side (−y) condition.

**Probe B: the −y side** (cohort 54932–54963)

| condition | e9 (true plate) | P-stale | P-truth | P-3 pick → e9 place to the true plate |
|---|---|---|---|---|
| shift at 300, 9 cm | 32 | 0 | 20 | **32** |
| shift at 300, 12 cm | 32 | 0 | 7 | **32** |
| far plate at reset, 9 / 12 cm | 32 / 32 | 0 / 0 | 10 / 0 | 12 / 8 (P-3's grasp fails: 12 / 8 grasps) |

A far plate at reset breaks P-3's pick, so the condition is the mid-episode move (owner D3+D4).

**Probe C: the readout twin.** A kernel-ridge plate readout of the frames at steps 320–400 after
the move behaves as follows:
- **trained in-distribution only:** 5.53 cm median error on far frames;
- **trained with far frames of other seeds:** 0.55–0.69 cm median, 1.7 cm p90.
- **In closed loop,** P-3's pick followed by e9's place aimed at that readout scored **31/32
  (9 cm) and 30/32 (12 cm)**, against 32/32 with the true plate.

This is an approximation: the frames came from P-truth trajectories, and the readout was fitted
on other seeds of the same probe cohort.

**The reading that shaped D1.** The room over P-3 is large (+12/32 and +25/32 over P-truth), but
a readout trained on data that covers the condition leaves at most +1 to +2 of 32 for a world
model. The claim is therefore superiority over P-3 and the blind twins, and non-inferiority to
the readout twin.

## 3. The design

### 3.1 Carried unchanged

- The v2 scene, e9 (`RestingPlaceExpert(release_pitch_rad=0.45, release_dx=0.015)`) and its
  schedule `(130, 80, 45, 150, 100, 50, 30, 50, 30, 60)`: the transfer starts at 405 and the
  lower at 505, and the place ends at 725.
- The counted success of T71-R1/R2, and the attempt of 740 commands plus the settle.
- P-3 (`checkpoints/task072-first-policy-v2-linux/run-1/P-3.pt`, sha256 `7988162d…60be8`), with
  its post-look readout refitted exactly as TASK-072 run-1 fitted it (G-repro).
- TASK-066's LeWM token predictor configuration (`token_dynamics.MODEL_CONFIG`).
- TASK-073's direction rule (restricted here to −y), its plate-shift hook, its e9 collector
  (`wm_critic_v2_runtime.run_collect_task`), its kinematic stand-in, its brancher and its renderer
  rules.

### 3.2 The condition: a right-side plate move at step 300

**Sign convention** (frozen, `SIGN_CONVENTION`):
- The world frame is the fixed pelvis's base frame: +x forward from the pelvis, +y to the robot's
  left, +z up.
- The right shoulder is at y = −0.10 m, and the reset plate centre at y = −0.09 ± 0.02 m.
- So **−y is the robot's right, away from the midline y = 0.**
- A "right-side move" is a shift whose direction lies in **225–315°**, measured from +x towards
  +y. Its y component is therefore at most −|d| cos 45°.

**Direction rule.**
- TASK-073's rule is kept: e9's plate-relative release point stays within 0.5 cm of its reset
  value, the plate stays on the tabletop, and the reset rule's plate–apple clearance holds.
- It is restricted to the 225–315° arc.
- One uniform draw per reset, `default_rng(SeedSequence([7413, seed]))`, picks from the eligible
  set at every |d|.
- Every shift vector of every cohort is stored in the manifest with a digest.

**Re-draw rule** (decided by Claude under owner delegation, 2026-09-30).
- On about 1–2 % of TASK-047 resets, no −y direction is eligible at some |d|, because the apple's
  reset position blocks it.
- So every TASK-074 seed, in every role including the corpus and U, uses `condition_reset`:
  - the seed's own wide-jitter reset if the plate can move right by every |d| in
    {3, 6, 9, 12} cm;
  - otherwise the same distribution re-drawn from `SeedSequence([seed, 7425, k])`,
    k = 1, 2, …
- 3 of the 176 stored cohort resets are re-drawn (K1 54011, R 54064, S 54502), and 3 of the 300
  corpus roots. `redraw` stores k.

**The move.** A harness hook moves the plate just before the observation of step 300, as in
TASK-073: a `body_pos` write plus `mj_forward`, outside every controller's `act()`. The grasp
has latched by then (about step 265 in the probes). The shift is |d| = 9 cm, or 12 cm if K1
chooses it.

### 3.3 The planner: P-3 picks, LeWM chooses the put-down spot

- **Before 405:** P-3 runs unchanged, on its post-look estimates.
- **The place primitive** (`place_planner.PlacePrimitive`) is e9's own six plate-dependent
  phases (transfer, lower, steady, open, clear, retreat), built by `RestingPlaceExpert` for a
  target xy g instead of the plate.
  - It keeps e9's phase clock from 405, and it re-targets without resetting the clock.
  - It reads only the robot's own joint state: the palm pose comes from the controller's private
    kinematic model (`PalmFK`), and the heights from the scene's declared constants
    (`a4_truth`).
  - A target whose release pose is out of the arm's reach is infeasible.
- **Decisions:** at 405, 421, 437, 453, 469 and 485 (six). From 505 (lower) the target is
  frozen.
- **The target grid:**
  - **coarse:** 49 targets, the post-look plate estimate plus dx ∈ {−9, −6, −3, 0, 3, 6, 9} cm and
    dy ∈ {−15, −12, −9, −6, −3, 0, 3} cm;
  - **fine:** 25 targets at 0.75 cm (±1.5 cm) around the coarse choice.
  - The coarse (0, 0), the stale post-look estimate, is the incumbent; the fine centre is the
    coarse choice.
- **Chunks:** each target's first 16 commands come from TASK-073's kinematic stand-in
  (object-free, ideal tracking, the embodiment's own acceptance calculation), rolled from the
  measured state.
- **The critic:** W rolls all the chunks of a stage in one batch (h = 16), and
  `models.latent_critic.LatentCritic` returns J(g) = ‖R_off(ẑ_{t+16}) − o_f‖ in cm, where
  o_f = o*(501) = (0.5731, −0.1121) cm.
  - o* is TASK-073's frozen statistic (the 112 counted-success train roots of
    `apple-look-v2-linux` run-1), recomputed at this protocol's steps. It reproduces TASK-073's
    values at the shared steps exactly.
  - The critic returns costs only: the selector never sees a latent.
- **Tie rule.** The incumbent wins every tie within 1e-6 cm, and an infeasible chunk costs +∞.
  With an action-blind critic every cost ties, so L-N keeps the stale target at every decision.
- **Fallback (frozen):**
  - if every candidate is infeasible, the planner keeps its previous target (at 405: the
    anchor);
  - every fallback is logged;
  - no decision is skipped for time; G6 measures latency instead;
  - there is no fallback to P-3 after 405: the probes show that P-3 cannot execute the far place.
- **Architecture.**
  - The controller holds no simulator handle, and there are no backend branches: the world model
    is anything with `predict_features`, so the LeWM/native swap stays one key.
  - `VisualModel.defaults` is unchanged, and so is every file behind an implementation hash.

### 3.4 Arms

| arm | rung | role |
|---|---|---|
| **L-plan** (the W seed with the lowest val criterion, fixed before any closed loop) | L1 | primary |
| L-plan-s1, L-plan-s2 (the other W seeds) | L1 | secondary: per-seed sign agreement |
| L-N (the no-action model N) | L1 | action-blind twin: every cost ties, so it keeps the stale target |
| L-shuf (W from H-twin's latent of reset (i + 1) mod n at the same step) | L1 | scene-blind twin |
| L-rand (a seeded uniform coarse, then fine choice among reachable targets) | L1 | random-choice twin; it reads no decision-time image |
| **H-twin** (e9's place aimed at R-plate's reading of the current frame, at every decision) | L1 | the perception twin (T, owner D1) |
| P-stale (P-3 unchanged) | L1 | the incumbent |
| P-far (P-3's recipe retrained on the new corpus, plate re-read by R-plate) | L1 | the data-matched BC control, reported (owner D5) |
| P-truth (P-3 given the true moved plate at step 300) | L4 | the incumbent's perception ceiling (G1) |
| H-handover (P-3 pick, then e9's place aimed at the true plate) | L4 | the family ceiling (K1, L-NO-HEADROOM) |
| B-oracle-shift, B-hold, B-random | L4 | harness checks |

The unshifted cohort U runs P-stale, H-twin and L-plan.

### 3.5 P-far, the data-matched BC control (owner D5; it gates nothing)

- **Recipe:** P-3's recipe, `first_policy_v2_model.train` with `first_policy_v2.TRAINING`: the
  same network, optimiser, loss, selection and model-init seed. The sampler seed is 7424.
- **BC rows:** the complete, non-mis-aimed train roots of `apple-far-shift-v2`, with
  `bc_rows`' mask; the labels are e9's clean commands.
- **Inputs:**
  - the apple from run-1's P readout of the post-look frame;
  - the plate from that readout's post-look estimate before 405, and from 405 the true plate at
    the latest decision step.
- **DAgger:** three iterations of 40, 50 and 60 rollouts on the corpus train seeds (plan order,
  disjoint slices), under their planned shifts.
  - At run time the plate is re-read by R-plate at the decision steps.
  - The labeller is e9 built from the post-shift truth and advanced on the executed results
    (TASK-072's shadow expert; privileged, training time only).
  - The rows are the executed steps' raw inputs as run.
- **At run time:** P-3's controller class with R-plate's re-read at the decision steps.

## 4. Seeds and cohorts

A repository search of 2026-09-30 (src, scripts, docs, benchmarks, tests, configs, `.mc`) found no
use of 54000–54999 before the design probes, and no use of 7410–7425 as a seed constant. Every
range is disjoint from every other and from every forbidden range: TASK-067's list, 51000–52199
including D2, cohort C, TASK-073's 53000–53999, and the spent design probes 54700–54999.

| seeds | cohort | use |
|---|---|---|
| 54000–54031 | K1 | the condition gate (development) |
| 54040–54071 | R | offline ranking and regret (O3, O4); never fitted on |
| 54100–54115 | D3 | development closed loop, stop rule only |
| 54200–54499 | `apple-far-shift-v2` corpus | 240 / 30 / 30 by reset; the test split is never decoded |
| 54500–54563 | S | gated, 64 shifted resets |
| 54600–54631 | U | gated no-harm, 32 unshifted resets |
| 54650–54699 | smoke | mechanics only (§10) |
| 54700–54999 | design probes | **spent** (§2); refused by every guard |

- **Model seeds:** 7410–7412 (W and N, one of each per seed).
- **RNG seeds:**

  | seed | use |
  |---|---|
  | 7413 | shift direction |
  | 7414 | mis-aim and unshifted draw |
  | 7415 | corpus split |
  | 7416 | corpus plan salt |
  | 7417 | L-rand |
  | 7418 | B-random |
  | 7419 | bootstrap |
  | 7420 | readout folds |
  | 7421 | W/N sampler |
  | 7422 | wrong-action shuffle |
  | 7423 | comparative rank |
  | 7424 | P-far sampler |
  | 7425 | reset re-draw |

- 7401 was the design probes' salt.

## 5. Stages and stop rules

Nothing from K1, R, D3, the corpus, S or U is simulated before this PR is merged **and** a pre-run
reviewer's GO is reported. The coordinator is told before each stage starts.

1. **Stage 0 (this PR).** The protocol, the frozen modules, the tests, the stored cohort values
   and the smokes (§10).
2. **Stage 1, K1** (`run_lewm_planner_v2.py k1`; simulator only, no world model).
   - For |d| = 9 and then 12 cm, the arms run on the 32 K1 resets in the order B-oracle-shift,
     H-handover, P-stale, P-truth. A |d| ends at its first failed bar:
     1. B-oracle-shift ≥ 30/32;
     2. H-handover ≥ 30/32;
     3. P-stale ≤ 4/32;
     4. H-handover − P-truth ≥ +8/32.
   - The smallest passing |d| is chosen. **Stop: L-NO-CONDITION** (escalate).
   - O5 (the stand-in's fidelity for the primitive's chunks) is recorded on the chosen cell's
     H-handover attempts.
3. **Stage 2, corpus** (`corpus`). 300 e9 roots from the frozen plan:
   - 25 % unshifted, and 75 % moved at step 300 on the −y rule, with |d| drawn from
     {3, 6, 9, 12} cm (independent of K1's choice);
   - 50 % mis-aimed by an extra release offset in a 4 cm disc, so that the world model sees wrong
     targets;
   - noise levels 0–3.

   It uses TASK-073's collector and episode schema, including the plate trajectory. The train and
   val roots of `apple-look-v2-linux` run-1 (190) join W's training set.
4. **Stage 3, train** (`train`, train split only).
   - **Features:** band frames 384–560, DINOv2 on CUDA, with the anchor check (G-anchor).
   - **Readouts:** R_off, on every 8th band frame, and R-plate, on the full tokens at the decision
     steps. λ is chosen from {1e-3 … 1e3} by 5-fold inner CV grouped by root.
   - **`first_outcome_utc`** is written after the train-only fits and before any number on a val
     root (the void rule's boundary).
   - **Blind baselines are written before any W number exists.**
   - **Budget:** TASK-073's calibration, saturation and escalation rules (U between 10 000 and
     60 000 updates). There are 3 seeds each of W and N.
   - **O1 and O2.**
5. **Stage 3b, P-far** (`pfar`): §3.5.
6. **Stage 4, ranking** (`rank`, cohort R, K1's |d|). H-twin runs each R reset. At each decision
   step:
   - the 49 coarse targets and the 25 fine targets around the best coarse target by true cost
     are each run for 16 commands in cloned state (true outcome, privileged, scoring only);
   - they are also turned into stand-in chunks.

   The blind rankers are scored and written (`blind.json`) before any W ranker.
7. **Stage 5, decide** (`decide`). The offline decision, first match:
   - any W seed fails O1 → **L-NO-DYNAMICS** (escalate);
   - N passes O2 (i) → **L-O2-VOID** (escalate);
   - O2 fails → **L-G2A** (TASK-054's criterion fails again: the clause fires);
   - a void O3 → **L-O3-VOID** (escalate);
   - O3 or O4 fails → **L-NO-RANK** (clause);
   - O5 fails → **L-PROPOSAL** (escalate);
   - else **OFFLINE-PASS**.
8. **Stage 6, D3** (16 development resets, non-gating; it requires the recorded OFFLINE-PASS and
   P-far). The arms are H-twin (first, for L-shuf's latents), L-plan, L-N, L-shuf, L-rand,
   P-stale, P-truth, H-handover, B-oracle-shift and P-far.
   - **L-NO-HEADROOM** if H-handover − P-truth < +4/16 (escalate).
   - **L-DEV-STOP** if L-plan − max(L-N, L-shuf, L-rand) < +3/16 (escalate).
   - Nothing on D3 is refitted.
9. **Stage 7, gated.** It runs only under a separate owner authorisation record pinned in the
   manifest (`gated_authorization`). The runner refuses it until then, and the authorisation PR
   writes `stage_gated`.
   - Cohorts S (64) and U (32), each arm once per reset, paired.
   - A clean worktree of the merged revision.
   - `decide_gated` produces the row.
10. **Stage 8, results PR.** Every arm is reported, and the privileged ceilings are labelled as
    not learned. An independent reviewer checks every restated number.

## 6. Gates

Intervals are session-(reset-)clustered bootstrap percentile intervals: 10 000 resamples, 95 %.
Every offline gate must pass on all three W seeds.

### 6.1 O1, dynamics (TASK-066's set, on the val roots)

At h = 8 and h = 16, on two window sets: **overall** (every val window in the band, stride 4) and
**shifted** (the shifted val roots). The bars:
- rank ratio ≥ 0.16, std ratio ≥ 0.39, collapsed fraction ≤ 0.05;
- the W − N comparative-rank lower bound > 0, with the rank-truncated controls at k = 1, 2, 4
  failing;
- the W / copy-last upper bound ≤ 0.8;
- the W / N upper bound < 1.0;
- the shuffled-action / W and zero-action / W lower bounds ≥ 1.10.

### 6.2 O2, readability and the G2a re-test

The windows are every val root's windows starting at a decision step whose true offset moves by
at least 1 cm over the 16 commands (TASK-054's moving cohort). All must hold:
- R_off on encoded latents at t + 16: median ≤ 1.0 cm;
- R_off on W's h = 16 predictions: median ≤ 1.5 cm;
- **(i)** the upper bound of the ratio rollout error / persistence ≤ 0.8 (TASK-054's G2a);
- **(iii)** the upper bound of rollout error / clock prior (o*(t + 16)) ≤ 0.8.

**(ii)** The same ratio (i) for N must fail, or the gate is void.

### 6.3 O3 and O4, ranking and regret (cohort R)

- **Groups** are the coarse (49) and fine (25) groups of §5.6.
- **O3:**
  - the lower bound of the median Spearman ρ_W ≥ 0.5;
  - the lower bound of the median of (ρ_W − the best blind ranker's ρ) ≥ 0.3.
  - The blind rankers are copy-last, N, L-shuf and prior-distance (the target's distance from the
    stale estimate).
  - A median ρ ≥ 0.5 of copy-last, N or L-shuf voids the gate.
  - **Twin-distance** (the target's distance from R-plate's reading, i.e. the readout as a ranker)
    is **reported only**: by D1, beating the readout is not required.
- **O4:**
  - the median true regret of W's choice within each group ≤ 1.0 cm;
  - the upper bound of the paired regret difference against the group's incumbent < 0.
  - A choice whose branch stopped counts as the group's worst finite outcome.
- **O5:** on K1's chosen cell's H-handover attempts, the stand-in's chunk for the executed target
  against the executed commands. The median relative error must be ≤ 0.25.

### 6.4 The gated rows on S and U (first match)

| row | condition |
|---|---|
| **VOID** | any of: B-oracle-shift < 60/64; a grasp by B-hold or B-random; a privileged read in an L1 arm; a failed determinism re-run (§8.3); a CUDA allocation failure |
| **S-VOID-CONDITION** | P-stale > 8/64 |
| **L-NO-HEADROOM** | H-handover − P-truth < +16/64 (escalate; no clause) |
| **L-HARM** | G5 fails (the clause fires) |
| **L-PASS** | G1–G6 and T |
| **L-PASS-TWIN-BETTER** | G1–G6 pass, T fails: the claim is made with the label that the readout beat LeWM |
| **L-SLOW** | G1–G5 pass, G6 fails (no clause) |
| **L-SCENE-BLIND** | G1 and G3 pass, G2 fails (no clause) |
| **L-NO-GAIN** | otherwise (the clause fires) |

The gates:
- **G1:** L-plan − P-truth ≥ +10 of 64, with one-sided exact McNemar p < 0.01.
- **G2–G4:** the same bar against L-shuf, L-N and L-rand.
- **G5 (U):** L-plan ≥ P-stale(U) − 2 of 32.
- **G6:** the median decision latency ≤ 0.8 s, one chunk at 20 Hz. Its scope is L-plan's decisions
  in all 64 S attempts, each decision's wall time inside `act()`, in the gated run's H workers
  (4 × 4 threads).
- **T (owner D1):** L-plan ≥ H-twin − 6 of 64.

Secondary, not changing the row: L-plan-s1 and -s2 show the same sign against P-truth, L-shuf
and L-N. P-far's paired difference is reported.

### 6.5 Blind and prior-only baselines against every bar

| gate | blind or prior-only baseline | where it sits |
|---|---|---|
| G1 | P-truth | privileged, so it is an upper bound for any P-3 variant |
| G2 | L-shuf | measured in the run |
| G3 | L-N | ties every cost, so it keeps the stale target; ≈ P-stale's place under the move |
| G4 | L-rand | measured in the run |
| T | H-twin | expected near the family ceiling (the probes: 31/32, 30/32) |
| O1 | copy-last; N | copy-last fails W/copy ≤ 0.8 (ratio 1.0); N fails W/N < 1 |
| O2 | persistence; clock prior | recomputed on this corpus in the train stage (`baselines.json`), before any W number |
| O3 | copy-last and N (constant costs: ρ = 0); prior-distance | prior-distance ranks by the stale estimate, which is wrong under the move; the margin carries the gate |
| O4 | the incumbent | a regret difference of 0 is not < 0, so it fails |

## 7. Void rule, abandonment clause and consequences

**Void rule** (frozen, `VOID_RULE`).
- A guard, a crash, a cap or a CUDA allocation failure makes the stage V, and nothing in it is
  read. Batches are never shrunk to fit.
- **A V before a stage's boundary is not a spent attempt**: it may be repeated as-is, recorded,
  without a fix. The boundary is:
  - `cohort_first_render_utc` for a simulating stage (K1, corpus, ranking, D3, P-far, gated);
  - **`first_outcome_utc` for the train stage** (the first number computed on val roots).
- After the boundary:
  - a gated V goes to the owner, and a second V closes TASK-074 as INCONCLUSIVE;
  - a development stage may be repeated once from scratch after a reviewed fix.
- Nothing is re-thresholded, retrained or re-selected after its numbers are seen.

**Abandonment clause** (fixed now).
- It fires on **L-G2A**, **L-NO-RANK**, **L-HARM** or **L-NO-GAIN**.
- It closes "a LeWM frozen-DINOv2-token planner choosing the place target of e9's place primitive
  after P-3's pick, on `apple-to-plate-v2` at 112 px". No further grid, cost, horizon or proposal
  variant is preregistered on `apple-far-shift-v2` without new evidence of a different kind.
- The LeWM backend, the token latent, the v2 task and the product goal stay open. The next step
  is the task owner's choice of a data or hardware change, not another planner variant.

**Every row's consequence** is in `ROW_CONSEQUENCES`:
- **the clause fires:** L-G2A, L-NO-RANK, L-HARM, L-NO-GAIN;
- **escalate:** L-NO-CONDITION, L-NO-DYNAMICS, L-O2-VOID, L-O3-VOID, L-PROPOSAL, L-NO-HEADROOM,
  L-DEV-STOP, VOID, S-VOID-CONDITION, and the budget rows;
- **close without the clause:** L-PASS, L-PASS-TWIN-BETTER, L-SLOW, L-SCENE-BLIND, INCONCLUSIVE.

## 8. Platform, run guards, memory and the re-run rule

### 8.1 Platform, devices, caps

As TASK-073:
- the Linux PC, `MUJOCO_GL=egl`, MuJoCo 3.13.0, and the main process in strict CUDA determinism;
- simulation on the CPU with 6 workers (1 torch thread each);
- arms that make world-model decisions, and the ranking stage, on 4 workers (4 torch threads for
  decisions, 1 for ranking);
- closed-loop encoding and W rollouts on the CPU in the worker; corpus featurisation, W/N and
  P-far training on CUDA.
- **The GPU** is shared with a resident service (about 7 GB) and with short Isaac containers.
  Before a GPU stage the operator checks `nvidia-smi` and waits if memory is short. A CUDA
  allocation failure is a V.
- **Caps:** 43 200 s global, 300 s per attempt, 5 400 s per model.
- **The pinned thread environment** (G-threads) and **the quiet-machine rule** (G-quiet: 1- and
  5-minute load ≤ 2.0 at the start) are carried.

### 8.2 The run guards (#108 and its review follow-ups, fixed in #111)

- **The pool** is `run_guards.BoundedPool`. It raises at once on a dead worker, and its close is
  bounded: pidfd-based kills, and `Pool.terminate` in a daemon thread with a timeout. The close
  record is written to every report.
- **The memory watch** samples the whole process tree with `run_guards.process_tree_memory`: PSS,
  falling back to VmRSS for a process whose PSS cannot be read. The fallback is counted and
  reported. The ceiling is 12 GiB summed PSS, and a stage starts only with MemAvailable ≥ 16 GiB
  (G-memory).
- **The signals:** V reports are written on SIGTERM, SIGINT and SIGHUP, by TASK-073's signal-safe
  handler (carried).
- **The render majority rules** are carried from TASK-073's owner rulings: a vote only on bitwise
  equal simulation state, with a difference of at most one level in at most 16 pixels.
- **The train stage's memory at full corpus scale** was measured before freezing, in the train
  smoke's `train-scale` probe (§10). The probe builds the train stage's feature table at the real
  size: 460 roots (the new corpus's 270 train and val roots plus the 190 of
  `apple-look-v2-linux`) × 177 band frames, 81 420 frames, with smoke features cycled. It then fits R-plate on the real
  row count (1 440 × 98 304 tokens), trains W and N briefly, and computes O1 and O2 on the real
  number of val windows.

### 8.3 The determinism re-run and its margin

- After the gated arms, **L-plan and H-twin** run again on the first four S seeds (54500–54503).
- Both read images, so both are compared by `run_guards.RerunRule` (`lewm_planner_v2.
  rerun_rule()`), which is validated at construction: every compared field is exact or has a
  tolerance, including a command tolerance.
  - **Exact:** success, grasp, at rest, termination reason, executed steps, the chosen-index
    sequence.
  - **Tolerances:**
    - final distance: 1.0 cm;
    - first grasp step and first place step: 40 each;
    - `target_cm_max` (the largest per-decision target difference): 1.0 cm;
    - `max_abs_command_difference`: 0.8.
  - A field missing from both records, missing commands, or a NaN fails.
- **TASK-073's thin margin (0.87 cm measured against its 1.0 cm incumbent tolerance, on H-rand) is
  handled explicitly:**
  1. TASK-074's random twin L-rand reads no decision-time image (its candidates are anchored on
     the post-look estimate). It is not a re-run arm, which removes the arm class that produced
     the thin margin.
  2. `RERUN_MARGIN_RULE`: each tolerance is at least twice the largest difference of that field
     measured between two runs of the same arm and smoke seed (the d3 smoke's re-run, §10). A
     gated difference above half a tolerance is reported in the results; one above it is a V.

## 9. Why this is different in kind from the abandoned lines

- **TASK-054 (CEM over the world-model cost).**
  - There is no sampling optimiser and no search over 14-D actions. The world model scores a
    fixed, low-dimensional, non-learned proposal set: 74 put-down spots per decision, each
    executed by a scripted primitive.
  - The latent is frozen external DINOv2 tokens, which passed TASK-066's dynamics gates. It is
    not the in-corpus encoder whose rollouts failed G2a five times.
  - The task is v2, where the scripted expert is feasible. v1 was infeasible even for e9
    (TASK-068), so the MVP's 0/150 is not evidence against planning as such.
  - TASK-054's G2a is re-tested head-on (O2 (i)). If it fails, the run stops before any closed
    loop and the clause fires.
- **TASK-073 (a critic re-ranking P-3's local aims).**
  - The executor is not P-3. The probes show why it must not be: P-3 aimed at the true far plate
    scores 20/32 and 7/32.
  - The grid is global (up to 15 cm from the stale estimate), not ±2 cm around an incumbent.
  - The condition was chosen where the family ceiling (32/32) is far above P-3's perception
    ceiling. K0 had +2/32 and +3/32 of room; here it is +12/32 and +25/32.
- **TASK-057 (BC on `apple-wide-v1` at 112 px).** No policy under test is trained. P-far is a
  reported control on a new v2 corpus, not `apple-wide-v1`.
- **TASK-062/065.** No encoder is trained, and the latent is patch tokens, not CLS.

## 10. Stage-0 smokes (smoke seeds 54650–54699 only; nothing in them is read)

**Setup.**
- All on the Linux PC, in `outputs/task074-scratch/` (git-ignored, never committed), on the
  working tree of this PR over `main` at `9a8b0c0`.
- Each report pins the file hashes it ran with (`pinned_hashes_at_preflight`/`_at_end`).
- The K1 smoke ran before `SHIFT_BLOCKED` was added (below). Every later smoke ran at this PR's
  final code, with one exception: after them, a root count in two texts was corrected, from
  "300 + 190" to "270 + 190" read roots. The texts are `MEMORY["train_scale_rule"]` (so the
  frozen hash changed) and the runner's `train_scale_probe` docstring. No code path changed.
  Otherwise only this document, the tests and the manifest's smoke block changed after the
  smokes.
- Every smoke started with the 1- and 5-minute load at or below 2.0 and MemAvailable ≥ 25 GiB,
  except the `decide` smoke (a JSON read with no simulation).
- The GPU's resident service was running throughout (6.6–7.2 GB).
- **The rows the smokes printed are meaningless and are not read:** the models are
  200-update smoke models on 10 train roots, and the cohorts are 3–4 stand-in seeds. The
  mechanics facts are these:

| smoke (report sha256) | what it checked | result |
|---|---|---|
| `smoke` (render check; `d7d9ed8c…`) | G-repro; the GO's render check, 32 seeds × 4 renders over 6 workers | G-repro 8/8; **IDENTICAL** (128 renders); 43 s; peak PSS 8.01 GiB |
| `k1 --smoke` (`85f16bdf…`) | K1's arms on 4 seeds at 9 cm; H-handover with chunk logging | every arm ran; **O5 median relative chunk error 0.210** (bar 0.25; this is the stand-in's fidelity for the place primitive, and it is read by nothing); 60 s; peak PSS 8.14 GiB; pool closed by joins, pidfd on |
| `corpus --smoke` (`f7e29487…`) | the collector on 16 smoke roots | sealed; 14 complete; **2 moves blocked** (see below); 22 s; peak PSS 7.48 GiB |
| `train --smoke` (`02b5bedf…`) | features, anchor, R-plate, R_off, `first_outcome_utc`, the budget code, W/N × 3, O1, O2; then **`train-scale`** | anchor max difference 4.4e-5 (bound 1e-3); **train-scale: 81 420 frames, 460 roots, R-plate on 1 440 × 98 304 rows, W and N briefly trained, O1 on 1 230 val windows; peak tree PSS 8.77 GiB against the 12 GiB ceiling**; the probe took 627 s. The whole smoke took 3 035 s: the O1/O2 statistics on the small smoke table took about 38 min while another agent's job loaded the machine (load about 15) |
| `pfar --smoke` (`778c87b4…`) | P-far: BC on 6 roots, 3 DAgger iterations of 2 rollouts, 4 CUDA trainings | complete; rows 3 960 → 7 803; 60 s; peak PSS 9.22 GiB |
| `rank --smoke` (`c04d02a2…`) | 3 R stand-in seeds, 6 decision steps, coarse and fine groups, the brancher, the blind and W rankers | 36 groups; `blind.json` written first; 97 s; peak PSS 7.31 GiB |
| `decide --smoke` (`db3421ea…`) | the offline decision's code path | the row is computed (meaningless with smoke models) |
| `d3 --smoke` (`e5e70019…`) | all 10 D3 arms on 4 seeds (H-twin first, L-shuf from its latents); latency; **the re-run** | every arm ran, with no fallback and no privileged read; **L-plan decision latency, 4 workers × 4 threads with the GPU's resident service running: median 0.518 s, p90 0.558 s, max 0.602 s over 24 decisions (G6's bar 0.8 s)**; **re-run of L-plan and H-twin on 2 seeds: every field identical** (command difference 0, target difference 0); 111 s; peak PSS 8.74 GiB |

**What the smokes changed** (decided by Claude under owner delegation, 2026-09-30). The corpus
smoke V'd once: TASK-073's plate-shift guard refused a move that would have put the plate onto the
apple.
- **What happened:** on two noisy roots e9's grasp had failed, so the apple still lay near its
  reset position at step 300.
- **The fix, `SHIFT_BLOCKED`:**
  - a move that would leave the plate touching anything but the table is not made;
  - an evaluated attempt then ends as a counted failure (`shift_blocked`), not a V;
  - a corpus root is collected again without the move and flagged;
  - blocked moves are reported per arm.
- **Why it is needed:** without it, one failed grasp would void a whole stage. No bar was changed.

**The re-run margin (`RERUN_MARGIN_RULE`).** The largest measured differences were 0 for every
field, so every frozen tolerance is at least twice the measured difference. A single run of 2
seeds cannot see the renderer's rare one-level difference (about 1 frame in 1 000). That is why
the tolerances are not 0.

## 11. Compute estimate

On the Linux PC, with 6 simulation workers and the GPU shared with the resident service:

| stage | estimate | basis |
|---|---|---|
| **K1** | **about 5–15 min wall** (at most 2 cells × 4 arms × 32 resets = 256 attempts; fewer if a bar fails early) | the K1 smoke took 60 s for 16 attempts including G-repro and the cohort renders; the probes ran 128 attempts in about 90 s on 6 workers. 6 CPU workers, no GPU; the smoke's peak tree PSS was 8.14 GiB (ceiling 12) |
| corpus | about 30–40 min | CPU |
| train | about 5–7 h | featurisation about 20 min; W/N calibration plus 6 models on CUDA dominate; O1/O2 at full scale took about 10 min in the train-scale probe, but up to about 40 min under load in the smoke |
| P-far | about 30–45 min | |
| rank | about 30 min | 4 workers |
| D3 | about 30–45 min | |
| gated | about 2–3 h | |
| **total** | **about 10–12 h**, within the 43 200 s cap | |

## 12. Deviations from the design proposal

| # | proposal | here | why |
|---|---|---|---|
| X-1 | cost against o*(t + 16) | against o_f = o*(501) | §0: the per-step prior misleads under a far move (delegated) |
| X-2 | coarse grid 7 × 7 at 2 cm | 7 × 7 at 3 cm, anchored on the post-look estimate and reaching 15 cm to the right | coverage of a 12 cm move (delegated) |
| X-3 | resets as TASK-047's | the re-draw rule (§3.2) | 1–2 % of resets have no −y direction (delegated) |
| X-4 | decisions 405–485 "every 16" | exactly 405, 421, 437, 453, 469, 485 | fixed in code |
| X-5 | the P-far "recipe" | §3.5, with the plate's run-time re-read and the BC rows' true plate at the latest decision step | a run-time plate input under the move (delegated) |
