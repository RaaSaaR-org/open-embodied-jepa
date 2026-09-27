# Apple→Plate control formulation proposal v1: a route to a first learned success (TASK-067)

**Status: PROPOSAL. This is not a preregistration.** Nothing here is frozen: no gate, threshold,
cohort, seed range, arm or budget. The task owner chooses whether any of it becomes a protocol,
and in what form. No controller was trained, no closed-loop attempt was run, and nothing was
simulated. The only new number comes from one label-only probe (§2.2, §13).

**Status lines.**
- **Learned Apple→Plate is still 0 successes.** This document does not change that.
- **TASK-066's gated run is in progress.** Nothing here reads, anticipates or depends on its
  numbers. §8 says how the recommendation branches on each of its pre-declared rows.
- The test splits of `apple-wide-v1` and `apple-look-v1` were not decoded. Cohorts C
  (45300–45339) and D (45000–45007, 45100–45107) were not simulated. `exemption_spent`
  (`benchmarks/manifests/apple-policy-diagnostics-v1.json`) stays `false`.
- Since TASK-057 the project has **no** primary control line. This document proposes one for the
  owner to accept or reject. It does not present behaviour cloning or sampling-based planning as
  the project's current control approach.

---

## 0. Summary

**Recommended primary line (P): information-matched behaviour cloning.** A learned policy that
receives, at run time, the same information the scripted expert acts on, and nothing privileged:
- a **learned perception bottleneck**: apple and plate xy, read once from the frozen DINOv2
  features of the post-look frame (TASK-061's look, TASK-063/064's encoder);
- the robot's **own step counter** (the time since the look ended);
- proprioception, plus the right palm pose computed from it by forward kinematics.

It is trained by behaviour cloning on the `apple-look-v1` train roots, then by up to three DAgger
iterations labelled by the privileged scripted expert on fresh training resets.

**Recommended fallback (F): a phase-decomposed, partially learned controller.** Learned
per-phase motor heads on the same inputs, with the phase switched by a scripted clock. It runs
only if P fails the first milestone while the perception diagnostic succeeds (§7.2). Any success
it produces is labelled *partially learned* and is never counted as a learned Apple→Plate success.

**Why.** The two abandoned lines failed for different proximate reasons (§1, §2), but the
behaviour-cloning failure has a concrete, checkable information gap behind it:
- The scripted expert's command is a deterministic function of the apple and plate positions at
  reset, the palm pose, and **its own command counter** (`scripted.py`).
- The TASK-056 arms saw one 112 px frame plus 86-D joint positions and velocities: **no clock**,
  and (TASK-059) **no readable apple position in the decision frame**.
- The new probe (§2.2) shows why the missing clock matters. The expert spends 43.3 % of its
  `orient` steps parked, commanding almost nothing (79.3 % on the 43 noise-level-0 roots). 91 of 170
  `orient`→`descend` switches are taken from such a parked state, on the counter alone. A
  memoryless policy is taught "do nothing" there for tens of steps and "go" for one. The TASK-056
  arms stalled at exactly that point.
- Since then, the look (TASK-061) and the frozen DINOv2 encoder (TASK-063/064) have made the
  apple readable in the decision frame, **offline**. The clock is free.

P is the smallest formulation that closes both gaps. It is independent of the world model, so it
does not wait on TASK-066. The world model enters at stage 3, as a critic of P's samples, and
only if TASK-066 passes (§8).

**First milestone (M1), cheap and development-only.** On the 16 development resets D, P reaches
at least 1/16 full Apple→Plate successes, with zero privileged reads and nothing substituted.
Named controls and diagnostics run on the same resets (§7.2). M1 is an **existence** bar, not a
gate. A non-learned open-loop demonstration replay has reached 5/16 on D (§7.2), so M1 cannot
show that P is better than replay. That comparison belongs to the gated stage (M2, cohort C).
Estimated compute is about 3–6 h, after TASK-066's run has finished (§10).

**Owner decisions needed**: §12. The first is whether a control formulation on `apple-look-v1`
is inside or outside the TASK-057 clause (§9).

---

## 1. Why CEM over the world-model cost failed (TASK-054)

Source: [`apple_world_model_v4_results.md`](apple_world_model_v4_results.md) and its TASK-058
errata; [`claim_audit_v1.md`](claim_audit_v1.md); `docs/DECISIONS.md` (2026-09-24).

**The record.**
- All four v4 arms failed the 14 preregistered gates: E0 10/14; E1, E2 and E3 9/14 each.
  - Erratum S4-09: the count is not a quality ranking. The difference is G6a at h = 16, and
    three shared passes are cleared without a learned predictor (G4 by a constant, G3 and G5 by a
    true-state copy-last).
- **The primary gate G2a** (rollout readout error ÷ the model's own persistence readout, ≤ 0.8)
  was 0.8763 / 0.8814 / 0.8635 / 0.9036. It has never passed (0.835 at v2, 0.831 at best in v3).
  - Erratum S4-02: the rollout does beat its own persistence readout in every v4 arm, by
    10–14 %, but never by the required 20 %. It has never beaten the true-state copy-last
    baseline (2.49 cm).
- **G1** (palm–apple, moving windows, ≤ 1.5 cm) was 3.65 / 3.66 / 3.48 / 4.41 cm. E0's encoded
  target alone was 2.5903 cm, above the threshold, so a perfect predictor on that encoder would
  still have failed G1.
- None of the three redesigns improved the rollout term. E2 − E0 on the rollout was −0.173 cm
  [−0.804, +0.132] (clustered 95 %; it crosses zero), and E3 made it worse (+0.761 cm
  [+0.155, +1.199]). Erratum S4-13 withdraws the
  claim about where E2's gain sits: that is not measured.
- **Candidate ranking**, the one quantity a CEM directly uses, was fragile: G6a 0.5438 (E0)
  against 0.2865 / 0.3296 / 0.4141, at the gated h = 16 only, on one seed, with no interval.
  Erratum S4-10: at the planner's h = 8 (10 groups, below the 12-group cohort rule), ρ was
  E0 0.36, E1 0.50, E2 0.36, E3 0.30.
- At h = 4 the E0 rollout was worse than persistence (U2 = 1.0461;
  [`apple_policy_v1_results.md`](apple_policy_v1_results.md) §4).
- The earlier frozen MVP benchmark (TASK-020, image-goal CEM) recorded 0/50 on each of three
  seeds per backend, on the same 50 resets, so not 150 independent trials; every learned episode
  ended on a joint-rate guard stop after a few commands (erratum S4-25, `claim_audit_v1.md`).

**Diagnosis** (reading, not measurement):
1. **The planner's cost was a readout of a rollout that did not beat persistence by the margin
   the protocol required**, and its within-state candidate ranking was not stable across arms
   or horizons. A sampling planner amplifies exactly those errors (the "planner exploits model
   errors" risk in `docs/DECISIONS.md`).
2. **The encoder term alone exceeded the task tolerance** (2.5903 cm against 1.5 cm) on the
   moving windows, so the cost could not resolve the positions the grasp turns on.
3. **An information limit found later applies to the same camera and encoder.** TASK-059 found
   that the 112 px onboard reset frame, read out by the preregistered readouts (kernel ridge,
   n = 190), does not give the apple position beyond a prior:
   occluded by the wrist on 111/190 resets; on the 79 visible ones raw pixels reach a ratio of
   0.964 [0.751, 1.207] to the prior, and E0 reads 2.067 cm against the prior's 1.720 cm (ratio 1.202 [0.977, 1.458]). That
   result postdates TASK-054 and was not one of its gates. It is stated as a contributing
   constraint, not as the cause.
4. **The goal is hard to state as an image.** An image goal for "apple on this reset's plate"
   needs a goal frame for the reset's own positions. That is privileged or has to be generated.
   This is structural, and it applies to latent MPC as well (§4, option O5).

## 2. Why behaviour cloning failed (TASK-056, TASK-057)

### 2.1 The record

Sources: [`apple_policy_v1_results.md`](apple_policy_v1_results.md),
[`apple_policy_diagnostics_v1_results.md`](apple_policy_diagnostics_v1_results.md),
[`task056_handover.md`](task056_handover.md), each with its TASK-058 errata.

- **0 full successes in 64 development attempts**: four arms × the 16 resets of D. D never
  gates, and n = 16 per arm with deterministic simulation. TASK-057 reproduced every arm
  unsubstituted (B2 exact).
  - Grasp: A0 (no image) 0/16, A1 (random-init encoder) 0/16, A2 (frozen E0, primary) 1/16
    (45100), A3 (fine-tuned E0) 1/16 (45006).
  - Every attempt the embodiment did not stop ran to the 1 000-step cap. A3 had two guard stops
    (45001 at 157 steps, 45003 at 143) (erratum S6-04).
- **Offline, the controls won.** Selection scores (lower is better): A1 0.01106 < A0 0.01313 <
  A2 0.01793 < A3 0.02340. A3 trained at batch 32 against 256 for the others (erratum S5-16).
- **The approach is where it broke.** On 61/64 unsubstituted attempts the palm never came 1 cm
  closer to the apple than at reset. Context, as TASK-057 gives it: the expert's own `orient`
  target is 0.13 m above the apple, and on every attempt the distance at step 130 exceeds its
  reset value by +0.014 to +0.049 m. So the statistic partly reflects that shared first phase.
- **On the non-descending resets the loop was a fixed point.** Representative trace A2/45001:
  from step 130 to 1 000 the palm–apple distance is constant at 0.1385 m, and the policy commands
  a constant (+0.040, −0.028, −0.028) with standard deviation ≤ 0.002. The shadow expert, on its
  own clock, commands bang-bang ±0.4 in each translation component. TASK-057 records the
  mechanism as unmeasured. Its declared candidate is that a policy which reads motion from joint
  velocities can continue motion but cannot start it.
- **Clause (a) did not hold.** D1-pipeline passed for all four arms: the loop's observation is
  byte-identical to the training observation, within D1-pipeline's stated scope.
- **No single command channel carries the failure.** G-SUB's best candidate reached 1/16
  against thresholds of 8–12 (privileged-substitution diagnostics). The clock-only blind
  controller alone reached 2/16 grasps, against A2's 1/16.
- **Where the approach did happen, grasp and lift worked and carry failed.** On A2/45100 and
  A3/45006 the hand held the apple on 784/784 and 804/804 steps after the latch, lifting it about
  17 cm, but `transport` never latched and the apple ended 0.218 m and 0.193 m from the plate.
  This is n = 2, on resets known before the run.
- **The expert succeeds on the same resets.** `scripted_oracle` was 16/16 grasp and 16/16
  success on D (B1); `hold` and `random` 0/16.
- **The reset frame gave no apple position beyond a prior to the preregistered readouts**
  (TASK-059, §1 item 3; "little apple-position signal that these readouts can use"). After the first
  commands proprioception does: a proprioception-only probe reads the `orient`-phase apple
  position to 1.01 cm, and to 0.163 cm when fitted on `orient` rows only, because the collector
  servos the palm to an apple-relative station (erratum S5-04). A policy that never makes the
  first apple-dependent move never reaches that shortcut.
- **Withdrawn, not used here:** the "smooth-L1 hedging makes dz under-shoot" story (errata S6-10,
  S6-12). Its statistics were |dz| medians and cannot show under-shoot.

### 2.2 A label-only probe: the demonstrations are mostly parked, and phases switch on a counter

`scripted.OracleManipulationPolicy` advances its phase when a **command counter** reaches the
phase's budget (`orient` 130, `descend` 80, `close` 45, `lift` 150, …; `advance()` in
`scripted.py`). It does not advance on an observable event. The P-servo reaches its target long
before the budget runs out, and then it waits. How much of the demonstration is that wait?

The probe `scripts/probe_expert_dwell.py` answers from the label sidecars of the 170
`apple-look-v1` **train roots** only (provenance: §13). It counts a step as **parked** when the
palm moves less than 1 mm and every commanded translation component is below 0.07 (about 1 mm at
15 mm per unit command).

| phase | steps | parked | parked, noise-level-0 roots (43, of which 8 have an aim offset) | switches to the next phase taken from a parked step |
|---|---|---|---|---|
| `orient` | 22 100 | **43.3 %** | **79.3 %** (of 5 590) | **91 / 170** |
| `descend` | 13 600 | 0.0 % | 0.0 % | 0 / 170 |
| `close` | 7 028 | 0.0 % | 0.0 % | 0 / 150 |
| `lift` | 21 539 | 7.4 % | 4.8 % | 20 / 143 |
| `transfer` | 8 530 | 1.5 % | 1.1 % | 8 / 142 |
| `release_high` | 9 763 | 33.1 % | 64.9 % | 17 / 58 |
| `lower_open` | 5 668 | 9.7 % | 21.0 % | 6 / 53 |
| `retreat` | 4 240 | 16.3 % | 58.3 % | n/a (no next phase) |
| all phases | 92 468 | 17.1 % | | |

- In `orient`, the first parked step comes at a median of step 16 of 130 (p90 50, max 113; on
  169 of the 170 roots).
- **Not interpreted:** `descend` and `close`. There the palm moves less than 1 mm on 79.6 % and
  61.1 % of steps while the command is saturated (median 0.4). A 1 mm-per-step rule cannot tell
  slow motion from a blocked palm, so no dwell claim is made for those phases.
- **Scope.** This is `apple-look-v1`. TASK-056 trained on `apple-wide-v1`, which TASK-064 records
  as collected with the same collector settings, so the same phase budgets apply. The probe did
  not read `apple-wide-v1`, so for TASK-056's own training data this is an inference.

What this means for a memoryless policy (reading): at a parked `orient` state the training set
pairs almost the same observation with "command ≈ 0" for tens of steps and with "start `descend`"
for one. Least-squares imitation of that mixture is a small command, and a small command keeps the
palm parked. That is the fixed point A2/45001 shows from step 130 onward. It fits the evidence and
sharpens TASK-057's declared candidate. **It is not a demonstrated mechanism:** no experiment has
separated it from the alternatives. P tests it by construction (the clock is an input), and the
C-noimg control (§7.2) keeps the clock while removing the image.

The same probe gives two outcome tables that set the precision P needs. These are privileged
scorer labels of the scripted collector, not learned results.

| train roots | noise level 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| no aim offset: success / roots | **35/35** | 27/37 | 15/28 | 7/34 |
| aim offset of 1.5–3.0 cm by design: success / roots | **1/8** | 0/9 | 0/10 | 0/9 |

- **An aim error of 1.5–3.0 cm is close to fatal:** 1/36 successes, against 84/134 without the
  offset. So the apple estimate has to stay well inside 1.5 cm on nearly every reset, not only at
  the median.
- **The expert is sensitive to action noise:** success falls from 35/35 to 7/34 as the injected
  OU noise rises (translation σ 0.04 per level, plus bursts; `look_corpus.py`). A learned head's
  imitation error acts like such noise. That is the case for DAgger.

### 2.3 Reading: the same information gap, twice

- **CEM (TASK-054):** the cost could not resolve the apple to the task tolerance (the encoder term
  alone was 2.59 cm), and ranking was unstable. Later evidence adds that the decision frame
  carried no apple position.
- **BC (TASK-056/057):** the policy could not know where the apple was at the decision frame
  (TASK-059), and could not know when to leave a parked state (§2.2). The expert knew both.

In both cases the controller was asked to act on information it did not have. Behaviour cloning
is only realisable when the expert's action is a function of what the learner observes. Neither
line checked that before training. P is built to satisfy it, and its first gate measures that it
does (§7.1).

## 3. What changed since, and what did not

| task | outcome | what it gives a controller | caveats that travel with it |
|---|---|---|---|
| TASK-061 | O-LOOK-RAW | A reset-independent 8-command look makes the apple readable in the 112 px onboard frame: raw pixels 0.469 cm [0.419, 0.528], dx sign 180/190 | E0 fails on the same frame (1.273 cm; ratio upper bound 0.779) and is no better than a random-init encoder on T1 (+0.149 cm [−0.162, 0.347]). Offline, kernel ridge, n = 190. |
| TASK-063 | O-PT-POOLED | Frozen DINOv2 ViT-S/14 reads the post-look apple: CLS 0.538 cm [0.487, 0.611], 183/190; tokens 0.394 cm [0.363, 0.447], 187/190 | CLS not detectably different from raw pixels (+0.068 cm [−0.010, 0.153]). Both random-init floors meet the bars. One encoder, one input size, one floor seed. |
| TASK-064 | C-ACCEPT | `apple-look-v1`: 200 roots, 599 branches, 170/20/10 train/val/test sessions; P-cls 0.550 cm [0.478, 0.642], 178/190 on fresh resets | Built by the privileged scripted collector (103/200 root successes are scripted). P-cls not detectably different from raw pixels (−0.034 cm [−0.125, 0.054]); R-tok met every bar (0.572 cm); McNemar P-cls vs R-cls 11/5, p = 0.210. Readability is not prediction. |
| TASK-065 | WM-NO-DYNAMICS | A LeWM predictor on frozen CLS latents beats copy-last (0.746 at h = 8), a no-action predictor (0.879–0.910) and wrong actions (1.78–1.98×), on all seeds | Fails the uncalibrated rank bar (ratio 0.365–0.399 against 0.5) and G5 (0.94–1.19 cm predicted against 0.68–0.83 cm encoded; the non-inferiority margin fails at h = 16 on every seed, and G5 passed only for seed 2 at h = 8). At h = 1 copy-last beats it (W / copy-last 1.227–1.250). Clause fired for pooled CLS only. |
| TASK-066 | in progress | Patch-token (4 × 4 pooled) predictor; G1 bars calibrated (rank 0.16, std 0.39, W-over-N comparative) | Not read here. Rows: §8. |

**What did not change.** The 112 px onboard camera; the single-strategy scripted collector; the
grasp's sensitivity to aim (§2.2); `scripted_oracle` 16/16 on D; learned Apple→Plate 0.

## 4. Options considered

Each chance below is **the author's prior, not a measurement**. It is the probability of at least
one success of the option's own label on the 16 resets of D, within its first milestone and
assuming correct software. This project's record argues for humility: TASK-056 passed its
pre-flight and scored 0/64. The ranges are wide on purpose, and each gives its main reason.

| | option | label of a success | chance of ≥ 1/16 on D | cost to first D result (M-series Mac) | data | main risks |
|---|---|---|---|---|---|---|
| **O1** | End-to-end visuomotor BC on per-frame DINOv2 features, with a diffusion or chunked (ACT-style) head, on `apple-look-v1` as it is: no clock, no bottleneck | learned | **10–25 %** | per-frame features for about 172 k train transitions at 14.3 ms/frame on CPU (≈ 40 min); 1–3 h training on MPS; D eval ≈ 10 min | apple-look-v1 suffices in volume, but has only about 134 distinct apple positions (non-aim train roots) for an image→action map | The parked-state aliasing of §2.2 remains. `orient` has 130 commands and parks from a median step of 16, far longer than any practical chunk. Overfitting 134 positions. |
| **O2 = P** | Information-matched BC: perception bottleneck (apple, plate xy from the post-look DINOv2 feature), own step counter, proprioception + FK palm pose; BC then ≤ 3 DAgger iterations from the privileged clock-expert | learned (§6, rung L1) | **50–70 %** | ≈ 3–6 h compute in total (§10); about 1–2 days of implementation and review | apple-look-v1 train roots for BC and the readout; DAgger adds fresh resets and their post-look frames; a small fresh perception and calibration set (§7.1) | Plate readability is unmeasured. Perception tails against the 1.5 cm cliff. Imitation noise (35/35 → 27/37 at noise level 1). Clock desynchronisation under DAgger. The "learned" label could read as cosmetic (§6). |
| **O3 = F** | Phase-decomposed: learned per-phase heads on the same inputs, phase switched by the collector's counter | partially learned (rung L2) | **55–75 %** | as P, minus most DAgger | as P | Shares P's perception risks. A success is not a learned-policy success. Converting it needs a learned phase switch. |
| **O4** | World model as critic: rerank K samples of P's stochastic head by a token-WM rollout | learned, and WM-in-the-loop only if a no-critic ablation is worse | **not a first-success route**: it can only reorder samples from a policy that already has support near success | needs per-step DINOv2 tokens (14.3 ms/frame CPU) plus K × h token-WM steps (2048 windows × 16 steps take 0.85 s on MPS in TASK-066 §14, so K = 16, h = 8 is a few ms) | on-policy rollouts from P for ranking gates; TASK-066's latent | Needs TASK-066 to pass. TASK-065's latent lost to copy-last at h = 1. Ranking was the fragile quantity at TASK-054. |
| **O5** | Latent MPC (CEM) over DINOv2 tokens with a goal latent | learned, WM-driven | **5–10 %** | a replan ≈ 0.2 s (768 rollouts × 8 steps, scaled from TASK-066 §14's prediction timing), plus per-step features; about 1 min per attempt when replanning every 4 steps (≈ 2–3 min when replanning every step), ≈ 16–45 min per 16-reset pass | TASK-066's latent; **a goal latent per reset**, which is not available without privileged information or a learned goal generator | Repeats the TASK-054 structure: planner exploits model error, no stable ranking, plus goal specification. Only after O4 shows ranking. |
| (diag.) | **A4-look**: learned perception → the scripted `apple_collector_policy` built from the estimates | **not learned** (learned perception, scripted control) | 70–85 % *of this non-learned kind* | about 1 h | as P's bottleneck | It decomposes failures; it is never a learned result. TASK-056's A4, never built, with the look. |
| (hw.) | Overview camera (TASK-061 O-raw 0.183 cm [0.168, 0.207], 185/190) | n/a | n/a | a new corpus | new | A hardware or workspace change on the robot. It is the owner's decision, and the §7 route if perception fails in the loop. |

Why O1 is not the primary, although it is the most "end-to-end": it keeps both information gaps
of §2.3 except the camera one. The dwell problem is structural, and chunking addresses short
pauses, not 100-step counters. With about 134 distinct apple positions, an image→action network
has too few independent examples of the one quantity that decides the grasp. P fixes both, and
O1-like end-to-end rungs can follow once a success exists (§6).

## 5. Recommendation

### 5.1 Primary P: information-matched behaviour cloning with DAgger

**Run-time inputs, all non-privileged:**
1. **Perception, once.** After the fixed 8-command look (TASK-061; identical on every reset,
   post-look state spread 0.0), the post-look onboard 112 px frame goes through the pinned frozen
   DINOv2 (TASK-063's preprocessing). A **learned readout** (TASK-063's kernel-ridge family, or a
   small MLP) outputs apple xy and plate xy. It is trained on privileged reset labels of
   training resets only, the same discipline as every readout head in the repo. The estimate is
   held for the episode, as the expert holds its one reset read.
2. **The policy's own step counter** since the look ended. It is a clock, not simulator truth.
3. **Proprioception** (86-D joint positions and velocities, train-normalised) and the **right
   palm pose from forward kinematics** of those joints. It is robot-derived, the `robot__`
   class in `training_labels.py`.

**Head.** An MLP over (estimates ‖ clock features ‖ proprioception ‖ FK pose) → the 7 free action
dimensions, assembled into the 14-D contract and passed through the unchanged
`project_candidates` guard, as in TASK-056. Chunk size 1 in the primary. A chunked or diffusion
head is **not** in the primary: given the inputs above, the expert's command is a deterministic
function, so the multimodality those heads address is expected to be absent (it is a hypothesis;
the offline action error on val roots is reported against it).

**Training.**
- **BC-0:** non-aim `apple-look-v1` train roots (about 73 k transitions, taking 134/170 of the
  probe's 92 468 policy steps as the rough share), label `collector__base_action`, post-displacement
  mask as TASK-056. Branch episodes stay excluded (their base policy is corrupted). The
  perception inputs during training are **cross-fitted estimates**, not truth, so the head sees
  realistic estimate error.
- **DAgger-k, k = 1..3:** the learner in command on fresh training resets (proposed range
  46000–46999, disjointness to be checked at preregistration), about 128 resets per iteration.
  Every visited state is labelled by the privileged expert on the learner's clock (TASK-057's
  corrected shadow expert, `scripted.apple_collector_policy`). Data is aggregated and retrained.
  The new resets' post-look frames also grow the readout's training set.
- Selection on the 20 val roots by offline action error, with TASK-056's eligibility rule
  (per-dimension output std ≥ 0.02). Its known weakness is declared again: val action error is a
  weak proxy for closed-loop success.

**Why this is the smallest change that can work.** Every input the expert uses now has a
non-privileged counterpart:
- the apple and plate come from the image (the offline readability of the apple is established;
  the plate's is the first thing measured, §7.1);
- the clock is the robot's own;
- the palm pose comes from the robot's own kinematics.

What is left to learn is a smooth map from those inputs to the command, plus robustness to its
own errors, which is what DAgger is for.

### 5.2 Fallback F: phase-decomposed, partially learned controller

This is the same perception bottleneck, inputs and data as P. The difference is one learned head
per collector phase, with the phase chosen by the collector's own counter schedule, a scripted
switch. It removes the burden of learning the timing and the phase boundaries.

- **It is triggered only by M1-MOTOR** (§7.2): P scores 0/16 on D while A4-look succeeds. F then
  runs once, on the same resets.
- **Its label is fixed now: partially learned.** A success of F is reported as "learned motor
  control with a scripted phase schedule". It is never counted in "learned Apple→Plate
  successes".
- **To promote it**, replace the scripted switch with a learned phase classifier (from the image
  and proprioception) in a separately preregistered step.

### 5.3 Diagnostic arms (not lines, never learned results)

- **A4-look:** learned perception → scripted `apple_collector_policy` constructed from the
  *estimated* apple and plate. The scene constants (`container_surface_z`,
  `object_support_height`) come from the committed scene description, with zero `task_truth`
  reads, as TASK-056 §3.2 specified. It is the arm that separates a perception failure from a
  motor-learning failure.
- **D-oracle-perc:** P's motor head fed the **true** apple and plate. This is privileged; it runs
  in its own report and is never mixed with learned arms. It bounds what better perception could
  give.

## 6. What counts as "learned"

TASK-056 lost a frozen-cohort decision to an undefined term, "learned arm" (its §15). So the
proposal defines the ladder by what runs at evaluation time, and a preregistration must enumerate
its arms by name against it.

| rung | at evaluation time | reported as |
|---|---|---|
| **L1: learned policy** | Every command after the fixed look prefix comes from a trained network. Its inputs are the onboard RGB frames, proprioception, forward kinematics of proprioception, and the controller's own step counter. There is no scripted controller or phase machine, no demonstration replay, no substitution, and **zero** reads of `sim.task_truth()`, of any `privileged__` or `collector__` label, or of the scorer (counted per attempt). Training may use privileged labels as targets. | **learned Apple→Plate success** |
| **L2: partially learned** | Learned motor heads, but a scripted component (for example a clock schedule) chooses the phase or head. | partially learned; never counted as a learned success |
| **L3: learned perception, scripted control** | A trained perception module feeds a scripted controller (A4-look). | not learned |
| **L4: privileged, scripted or substituted** | Any run-time privileged read, the scripted collector, demonstration replay, `scripted_oracle`, G-SUB-style substitution, D-oracle-perc | not learned |

Four things are allowed in L1, and each needs the owner's explicit acceptance (§12):
- **The look prefix.** It is fixed and reset-independent (applied = requested; post-look state
  spread 0.0 on every root in TASK-061 and TASK-064), like a home pose. It carries no reset
  information of its own: with the apple hidden, every readout falls to the prior (TASK-061).
- **The step counter.** It is non-privileged. It does let a policy reproduce the expert's
  schedule, which is why the C-noimg control keeps the clock and removes the image.
- **Forward-kinematics palm pose.** It is computed from the robot's own joint readings, not from
  simulator truth.
- **A perception bottleneck trained on privileged reset labels.** It is trained with labels and
  run without them, the same discipline as the readout heads since TASK-050.

**Honest limits of an L1 success at M1.**
- **The learned motor map imitates a known analytic P-servo** on nearly explicit inputs, so the
  claim is "a first learned success". It is not "a hard skill learned".
- **It is not "LeWM driving the robot".** It uses the frozen DINOv2 encoder, which is also
  TASK-066's latent, but no world-model predictor. A success counts as world-model-driven only
  when the world model's output changes the executed command **and** a no-world-model ablation
  scores worse on the same resets (stage 3).
- **Later rungs remove the crutches one at a time**, each separately preregistered: the
  bottleneck (end-to-end features), the FK input, and the look-once restriction (per-step
  features).

## 7. Staged plan, with preregistrable gates

Every threshold marked *(calibrate)* is set before the freeze, from the stage-0 calibration, and
never after an outcome is seen.

### 7.1 Stage 0: pre-freeze calibration and offline gates (disjoint resets; about 1 h of compute)

- **C0, the expert's tolerance curve (scripted, privileged; calibration only).** On about 32
  fresh calibration resets, with the look and noise level 0, run `apple_collector_policy`
  constructed from truth plus an injected estimate error. Apple error 0 / 0.5 / 1.0 / 1.5 cm and
  plate error 0 / 1 / 2 / 3 cm, each in a random direction. The output is success against error.
  §2.2 already shows the cliff (1/36 at 1.5–3.0 cm aim offset); C0 locates it. It sets the
  perception bars below.
- **S0-P, perception on fresh resets (gate).** Fit the readout on the `apple-look-v1` train roots
  plus a fresh perception-train set. Evaluate it on fresh held-out resets that neither the corpus
  nor any fit has touched, **not** on the corpus val roots, which are gate-evaluation roots of
  earlier tasks.
  - Apple xy: median and **p90** within the C0-derived budget *(calibrate)*.
  - Plate xy: median and p90 within budget *(calibrate)*. **The plate has never been read out
    from the post-look frame in this project.** If the plate fails while the apple passes, stop
    and report: a per-step plate readout during `transfer` is a design change for the owner.
  - The apple-hidden spurious check, as TASK-061.
- **S0-D1, pipeline equivalence** (TASK-057's D1-pipeline, extended). The closed-loop look
  reproduces the stored look. The live post-look frame and its features equal the offline ones
  byte for byte (or within the recorded float32 batch effect, ≤ 1e-3 as TASK-066's G-cache). The
  live `act()` equals the batched offline prediction to 1e-4.
- **S0-A, offline action error** on val roots, P against C-noimg. Reported, not gating: it is a
  weak proxy.

### 7.2 Stage 1: milestone M1 on the development cohort D (about 1–2 h of compute)

**Arms, enumerated by name.** "Learned arm" in this stage means exactly P-0, P-1, P-2, P-3 and
C-noimg.

| arm | what | label |
|---|---|---|
| **P-k**, k = 0..3 | the primary after k DAgger iterations | learned (L1) |
| **C-noimg** | P-0's recipe with the perception inputs fixed to the train-mean apple and plate; clock, proprioception and FK kept | learned control (L1) |
| **A4-look** | §5.3 | not learned (L3) |
| **D-oracle-perc** | §5.3; own report | privileged (L4) |
| **B-oracle** | `scripted_oracle` with the look | harness: ≥ 14/16 success |
| **B-hold, B-random** | as TASK-057 B1 | harness: 0/16 grasp |
| **B-replay** | open-loop replay of the successful non-aim train root whose post-look DINOv2 CLS is nearest | non-learned reference |

- **Each arm runs each of the 16 resets exactly once.** P-k is evaluated on D once per k. The
  DAgger rollouts never use D. All four evaluations are reported, whatever they show.
- **The non-learned bar is already above 0 on D.** `demo_replay`, open-loop replay of a TRAIN
  demonstration retrieved by frame-0 RGB distance, reached 5/16 successes on D: 2/8 on
  45000–45007 and 3/8 on 45100–45107 (`apple_wide_object_ceiling_results_v2.md`, as TASK-056
  §5.2 assembles it; pooled over 24 resets, 6/24 = 25.0 % [12.0, 44.9]). So M1 can establish
  existence only. Beating replay is M2's job.

**M1 outcome rows (first match):**

| row | condition | reading | next |
|---|---|---|---|
| **M1-VOID** | B-oracle < 14/16, or B-hold or B-random > 0 grasps, or S0-D1 fails, or any privileged read by a learned arm | nothing is read | fix the harness; one repeat |
| **M1-PASS** | some P-k reaches ≥ 1/16 full successes | **the first learned (L1) Apple→Plate success, on the non-gating development cohort** | M2 (owner authorization for cohort C). The P-k carried to M2 is the one with the most D successes, ties to the later k. This is a selection on D, declared as one; C is the unbiased test. |
| **M1-MOTOR** | every P-k scores 0/16; A4-look ≥ its calibrated threshold *(calibrate from C0 and S0-P)* | perception is adequate in the loop, and motor learning is the failing component | F once (§5.2) |
| **M1-PERCEPTION** | every P-k scores 0/16; A4-look below its threshold; B-oracle ≥ 14/16 | the estimates do not survive the closed loop, or are not accurate enough | the abandonment clause (§7.6) |

Rules a preregistration must add (the rows above do not yet cover them):
- **S0-P fails on the apple** (the readout misses its calibrated budget on fresh resets): stop
  before any closed loop and report to the owner. The apple estimate is what the whole line rests
  on, so this is the §7.6 route unless the owner rules otherwise.
- **A second M1-VOID** closes M1 as INCONCLUSIVE; the owner decides.
- **C-noimg reaching ≥ 1/16.** The clock-only blind controller already grasped 2/16 on D
  (TASK-057), so this is possible. It is reported and changes no row, since M1 is about P. But a
  P-k success then says little about vision until M2's G2 (P − C-noimg ≥ +8) is read.
- **F reaching ≥ 1/16** after M1-MOTOR is a partially learned success, reported as such. The
  next step is the owner's: M2 for F (under its label) and/or a learned phase switch (§5.2).

### 7.3 Stage 2: M2, the gated test on cohort C (about 3 h of compute; separate authorization)

This reuses `apple_policy_v1.md` §5.2 with its lessons applied:
- **Arms named by enumeration.** The stop rule (0/16 grasp on D → does not run on C) names its
  arms.
- **The precedence field is cited.** The gate section cites `exemption_spent` (currently `false`),
  as `apple_policy_diagnostics_v1.md` requires of any successor protocol.
- **Cohort values come from the stored manifest**, never recomputed from `wide_reset`. This is
  the blocking stored-values debt of `task056_handover.md` §7.

| gate | proposed condition (TASK-056's, where it exists) |
|---|---|
| G1 | ≥ 17/40 full successes **and** more than B-replay on the identical 40 resets |
| G2 | P − C-noimg ≥ +8 (exact McNemar, with realised n_d and p reported) |
| G3 | P − (the same pipeline on seed-0 random-init DINOv2 features) ≥ +8. **Declared reading if it fails:** "a learned visuomotor policy works; encoder pretraining contributes nothing measurable". R-tok met every readability bar (TASK-064), so this is a live possibility. |
| G4 | grasp ≥ 20/40 |
| G5 | harness: hold and random 0/40, `scripted_oracle` with the look ≥ 38/40 |
| G6 | zero privileged reads by any learned arm (counted per attempt); a violation voids the run |
| G7 | median control time ≤ 100 ms per command, including the one DINOv2 forward pass at the look |

### 7.4 Stage 3: the world model as a critic (conditional on TASK-066 passing; §8)

- **What it is.** P's head becomes stochastic (for example Gaussian, or a small diffusion head
  over a short chunk). At each replan, K samples are rolled through the TASK-066 token predictor
  for h steps and scored by a readout of the predicted grid against the policy's own current
  sub-target. The best sample is executed.
- **Offline gate before any closed loop.** Within-state ranking ρ ≥ 0.5 over sibling branches.
  `apple-look-v1` has 473 RGB-distinct sibling pairs at step 16 (A8), but those are train + val,
  and val roots are gate-evaluation roots of earlier tasks; a preregistration must say which
  siblings it reads, or use fresh branches. Also a ranking gate on P's own on-policy rollouts
  (fresh resets). This is v4's G6a idea on the new latent, with an interval
  this time.
- **Closed-loop gate.** P + critic beats P alone on the same resets by a pre-declared margin, on
  D first and then on a fresh cohort. Only then is the success "world-model-driven" (§6).

### 7.5 Stage 4: latent MPC over tokens (conditional on stage 3)

Only if the critic's ranking holds. Sampling is centred on P's proposals (policy-prior MPC), not
from a broad prior, and the goal is P's own sub-target readout, not a goal image. Stage 4 is not
proposed for preregistration now.

### 7.6 Abandonment clause for TASK-067 (proposed)

**It fires on M1-PERCEPTION, or on M1-MOTOR followed by F scoring 0/16.**
- **What closes:** learned control on `apple-look-v1` at the 112 px onboard camera. No further
  head, loss, DAgger or bottleneck variant on this corpus and this camera without new evidence of
  a different kind.
- **The conclusion is the one `apple_policy_v1.md` §7 pre-declared.** Any results document quotes it
  verbatim rather than paraphrasing it. In substance: a 112 px onboard camera plus a single-mode
  scripted-collector corpus does not support learned Apple→Plate on this platform, and the
  product goal needs a data or hardware change, "not another model". The overview camera (TASK-061 O-raw) is the change
  on record, and it is the owner's decision.
- **What does not close:** the LeWM backend, DINOv2 as an encoder, the product goal, and the
  corpus (sealed; test split unread).

## 8. How the recommendation branches on TASK-066

**Stages 0–2 do not depend on TASK-066 at any row.** P uses DINOv2 as a frozen perception
encoder, not a world-model latent. That is deliberate: no outcome of TASK-066 blocks the route to
a first learned success.

| TASK-066 row | stages 0–2 | stage 3 (critic) and stage 4 (latent MPC) |
|---|---|---|
| **WM-TOK-DYNAMICS** (every seed passes G1–G5) | unchanged | Preregistrable after M1-PASS. TASK-066's row itself says "No control formulation is implied; control needs its own preregistration and must answer the TASK-054 and TASK-057 clauses". That row's own recommendation (a held-out test-split confirmation, and/or a harder world-model test) should come first or run alongside. Stage 3's offline ranking gate is itself a harder, control-relevant test. |
| **WM-TOK-UNSTABLE** (one or two seeds) | unchanged | Wait. The clause does not fire; the owner decides about added seeds under that row's rule. |
| **WM-TOK-CEILING** (the pooled grid misses the T1 bar) | unchanged | Wait. The clause does not fire; the pooling is the finding, and a finer grid needs its own compute plan. |
| **WM-TOK-APPLE-LOST / WM-TOK-COLLAPSE / WM-TOK-NO-DYNAMICS** (clause fires) | unchanged | **Off on this corpus.** The frozen-pretrained-DINOv2-latent predictor line on `apple-look-v1` is then closed at both read-out points (TASK-065 and TASK-066 clauses). A world model re-enters only through a route with new evidence of a different kind. Recorded as untested by TASK-065/066: a fine-tuned encoder, other encoders, or the overview camera. A predictor on the same frozen latents trained on P's on-policy DAgger rollouts is **a clause-scope question, not a route**: it is arguably a variant the clause closes, and the owner rules on it. Each is the owner's choice. The product goal is not met by a stage-2 success alone, and this should be said plainly. |
| **V** | unchanged | Wait: TASK-066's own rule is one from-scratch repeat (its §13). |
| **INCONCLUSIVE** (a second V) | unchanged | The clause does not fire; the owner decides. |

**Operational coupling.** Every stage-0 to stage-2 step that uses the CPU or MPS should start only
after TASK-066's gated run has finished. Its caps (featurisation 5 400 s, 4 500 s per model,
57 600 s global) are wall-clock, and contention could void it. Preregistration work (documents,
code, review) can go in parallel.

## 9. How this answers the TASK-054 and TASK-057 clauses

- **TASK-054** abandoned CEM over *this* world-model cost as the primary line, and preregisters no
  further predictor-architecture protocol of that kind. P is not sampling-based planning. Stage 4
  (latent MPC) is conditional, not primary, and uses a different latent and a policy prior.
- **TASK-057** fired `apple_policy_v1.md` §7: *"this line stops. No third control formulation is
  preregistered on this corpus and this camera."* The next task had to be a perception/data task:
  *"and if that does not move the closed-loop number either,"* the product goal needs a data or
  hardware change.
  - The perception/data tasks since then (TASK-059, 061, 062, 063, 064) changed the corpus design
    (the look) and the encoder. **None has measured the closed-loop number.** Without such a
    measurement, the clause's own second sentence cannot be evaluated.
  - P runs on `apple-look-v1`, a corpus built by that perception/data task, with an encoder that
    passed readability where E0 failed. It keeps the same **camera** (112 px onboard).
  - **The executing agent does not decide whether that is inside the clause.** The reading
    proposed here is that TASK-067 is the closed-loop measurement §7 anticipates, rather than a
    third formulation "on this corpus". §7.6 re-imposes the §7 conclusion if it fails. **The owner
    rules (§12, decision 1).**
  - Also answered: `apple_policy_diagnostics_v1.md` §5.1's "no further loss, head or
    output-parameterisation variant" is read here as applying to the stopped line (E0 features
    on `apple-wide-v1`); P is not such a variant. That reading is also the owner's to confirm.
    Cohort C stays unconsumed until a separate authorization. `exemption_spent` stays `false` and is cited in M2's gate section.

## 10. Cost and schedule on the M-series Mac (MPS/CPU)

Estimates are anchored on committed timings: DINOv2 14.3 ms/frame on 6 CPU threads (TASK-066
§6.5); TASK-064 collected 799 episodes in 844 s on 12 workers, plus 395 s assembly; TASK-057 ran
256 closed-loop attempts in 2 792 s on CPU; TASK-056's policy control time was 12–14 ms per
command.

| step | estimate |
|---|---|
| C0 tolerance curve: 7 conditions (apple error levels with the plate exact, plate levels with the apple exact, sharing the 0/0 cell) × 32 resets | ≈ 10 min on 12 workers |
| Fresh perception and calibration resets (look plus one frame, about 512) | ≈ 5 min, plus about 10 s of features |
| Readout fit and S0-P | minutes |
| BC-0 training (MLP on low-dimensional inputs) | ≤ 10 min |
| DAgger: 3 × (128 rollouts + retraining) | ≈ 30–90 min |
| M1 on D: about 10 arm-evaluations × 16 attempts | ≈ 1 h |
| **Stages 0–1 in total** (the rows above, with margin for a void-harness rerun and the sequential parts) | **≈ 3–6 h compute**, after TASK-066 finishes |
| M2 on C (separate authorization) | ≈ 3 h |
| Implementation: look-aware runner, policy, DAgger loop, controls, three PRs with review | 1–2 days of agent time |

## 11. Risks, named before any number exists

1. **The plate's readability is unmeasured.** It is S0-P's first job (§7.1).
2. **Perception tails against the aim cliff.** The medians are about 0.5 cm, but the aim offset of
   1.5–3.0 cm gives 1/36 successes. The p90, not the median, decides the grasp.
3. **Imitation error acts like injected noise.** The expert falls from 35/35 to 27/37 at noise
   level 1. DAgger is the mitigation, and its limit is that the clock-expert is a weak corrector
   far from its own path: 7/34 at noise level 3.
4. **No recovery behaviour in the labeller.** The scripted expert has no re-grasp: after a
   failed or slipped grasp its labels continue the schedule, so DAgger cannot teach recovery.
   A first success does not need recovery; a high success rate may.
5. **Clock desynchronisation.** If P falls behind the schedule, the clock-expert's DAgger labels
   switch phase anyway. The mitigation is an event-triggered relabelling expert (a phase advances
   on convergence). It must itself pass ≥ 14/16 on D in command before its labels are used, and
   it would be a declared variant, not a silent swap.
6. **The "learned" label could read as cosmetic** (§6). The answer is the ladder and C-noimg, not
   stronger words.
7. **Grasp physics.** The close-phase lateral drift that ejects the apple (grasp-closure v3) is
   handled by the expert at noise level 0 (35/35), but not necessarily by a learner's deviations.
8. **Reuse of D.** D has been consumed many times, n = 16, and simulation is deterministic, so M1
   is existence only. `demo_replay` reached 5/16 there.
9. **Compute contention with TASK-066** (§8).
10. **The history of forecasts.** Every earlier generation expected more than it got. The chance
   ranges in §4 are priors, and M1 is sized so that a wrong prior costs hours, not a frozen cohort.

## 12. Owner decisions needed

1. **The clause.** Is a control formulation on `apple-look-v1` (new corpus design, same 112 px
   onboard camera) the closed-loop measurement `apple_policy_v1.md` §7 anticipates, or a "third
   control formulation on this corpus and this camera"? This proposal reads the former (§9).
2. **The ladder (§6)** and the four L1 allowances: the look prefix, the step counter, the FK palm
   pose, and a perception bottleneck trained on privileged reset labels.
3. **The primary and the fallback** (§5). This includes that F's successes are never counted as
   learned.
4. **M1's design**: up to four evaluations of P on D, all reported; the arm carried to M2 selected
   on D; the named controls.
5. **Scheduling**: compute starts only after TASK-066's gated run has finished.
6. **The product-goal framing**: a stage-1 or stage-2 success is a learned policy on DINOv2, not
   yet "LeWM driving the robot". The world model enters at stage 3, or through an owner-chosen
   route if TASK-066's clause fires.
7. **A new seed range** for calibration, perception and DAgger resets (proposed 46000–46999;
   disjointness checked at preregistration).
8. **Cohort C** is used only by a separate authorization, as always.

## 13. The probe behind §2.2 (provenance), and what was deliberately not run

| item | value |
|---|---|
| script | `scripts/probe_expert_dwell.py`, committed in this PR; the cited run is at `7c9d08a` |
| command | `uv run --no-sync python scripts/probe_expert_dwell.py --dataset data/apple-look-v1 --output outputs/task067-dwell/run-4` |
| reads | the label sidecars of the 170 `apple-look-v1` train roots only, each checked against its recorded sha256; the corpus manifest (sha256 `81d760d1…db64`, checked). No frame decoded; no val, test or branch episode opened. |
| report | `outputs/task067-dwell/run-4/report.json` (git-ignored), sha256 `b4197fac6ad188feeaeefaefe81a34b6332fa70f10913714083a4079afd87eba` |
| device, time | CPU, a few seconds |
| earlier runs | `run-1` and `run-2` were uncommitted drafts. `run-3` (at `04fd767`, sha256 `ee342baa…c383`) was the first committed run; the PR #77 review found that it counted parked switches against phase runs rather than switches (the last phase has no switch). `7c9d08a` adds the `switches` field and the noise-level-0 root count (43), and `run-4` is cited. Every field `run-4` shares with `run-3` is identical, and the reviewer's own re-run of `04fd767` reproduced `run-3` byte for byte. |

**Not run, on purpose.**
- **No DINOv2 featurisation and no plate readout.** TASK-066's gated run was featurising on the
  same machine under a wall-clock cap. A plate readout on the corpus's 190 train + val roots would
  also be a pre-freeze read of gate-evaluation roots for a protocol that does not exist yet. It is
  S0-P's job, on fresh resets.
- **No simulation.** C0, A4-look and everything closed-loop belong to a preregistration.

## 14. Sources

`docs/DECISIONS.md` (2026-09-24, 2026-09-25, 2026-09-26, 2026-09-27);
[`apple_world_model_v4_results.md`](apple_world_model_v4_results.md);
[`apple_policy_v1.md`](apple_policy_v1.md) §3, §5, §7;
[`apple_policy_v1_results.md`](apple_policy_v1_results.md);
[`apple_policy_diagnostics_v1_results.md`](apple_policy_diagnostics_v1_results.md);
[`task056_handover.md`](task056_handover.md);
[`claim_audit_v1.md`](claim_audit_v1.md);
[`apple_info_ceiling_v1_results.md`](apple_info_ceiling_v1_results.md);
[`apple_observation_reprobe_v1_results.md`](apple_observation_reprobe_v1_results.md);
[`apple_encoder_study_v1_results.md`](apple_encoder_study_v1_results.md);
[`apple_pretrained_encoder_v1_results.md`](apple_pretrained_encoder_v1_results.md);
[`apple_look_corpus_v1_results.md`](apple_look_corpus_v1_results.md);
[`apple_latent_dynamics_v1_results.md`](apple_latent_dynamics_v1_results.md);
[`apple_token_dynamics_v1.md`](apple_token_dynamics_v1.md) (protocol only; no result read);
[`apple_wide_object_ceiling_v1.md`](apple_wide_object_ceiling_v1.md) (`demo_replay` retrieval);
`src/embodied_jepa/scripted.py`, `src/embodied_jepa/task.py`, `src/embodied_jepa/look_corpus.py`.
