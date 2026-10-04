# The next LeWM task on apple-to-plate-v2: candidate designs (DRAFT design note)

**Status: DRAFT design note. It is not a protocol and nothing in it is frozen.** No run was made
for it, no seed was simulated and no bar below is gated. It answers PLAN.md's Branch B entry for
PRED-INFEASIBLE: "a design note for a different action-dependent target, with its own K0-style
headroom check and no world model".

Every choice in it is **decided by Claude under owner delegation (2026-09-30)**. The rulings are
R9.1–R9.10 in [DECISIONS.md](../DECISIONS.md), decision 2026-10-04. This revision answers the
independent reviews of #135 at `c387b20` and `008a1bf` (both REQUEST CHANGES) and applies R9.8 and
R9.9. Any task the note
leads to needs, before a single cohort seed is simulated:
- its own feasibility record;
- its own preregistration;
- an independent review.

The canonical status sentence (DECISIONS 2026-10-02, R7), verbatim:

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

Nothing in this note is a result. Every number it quotes names its source. Numbers marked
*estimate* are back-of-envelope figures derived here. They are not measured, and the feasibility
smokes exist to check them.

---

## 1. Where we start (checked against the repository on 2026-10-04, at `702a7d9`)

**TASK-076's status.**
- **The protocol is frozen.** #137 merged as `702a7d9`, and R8.19 is in effect.
- **K0 ran and ended K0-PASS** (R8.19), with τ_re = 1.0 cm. The τ curve is 32, 31, 28, 23, 17, 18,
  16, 8 and 5 of 32 at 0, 0.5, 1, 1.5, 2, 2.5, 3, 4 and 5 cm.
- **K-pred** is PRED-INFEASIBLE if Stage O records O-PASS, and PRED-NOT-RUN otherwise. Both rows
  escalate without a clause and lead to PLAN.md's Branch B, which this note answers.
- **Reported-only data since the freeze** (run at `702a7d9`; no row is read from them here).
  - **Stage D** ended D-PASS: H-twin 15/16 and H-handover 16/16 (`task076-D-1/report.json`,
    sha256 `a84c00e2…e93f`).
  - **K-pred's constant-velocity cells.** These are reported only and never admit. The plate moves
    12 cm (M-a) or 9 cm (M-b), with 4.0 cm and 3.0 cm still to go at the last decision. Counted
    successes of 32 (`task076-Kpred-Ma-1/report.json`, sha256 `d512a712…a106e`;
    `task076-Kpred-Mb-1/report.json`, sha256 `0c976384…a9130`):

    | arm | M-a | M-b |
    |---|---|---|
    | H-final (privileged ceiling) | 32 | 32 |
    | H-now (privileged, the current plate) | 7 | 1 |
    | H-twin (perception, the current plate) | 11 | 4 |
    | H-cv (perception, constant-velocity extrapolation) | 24 | 24 |

  - **What they show.**
    - Not predicting costs most of the ceiling.
    - A hand-written extrapolator fed perception readings of a *moving* plate still sat 8/32
      below the ceiling in both cells, even though it knew the motion's form and stop step.
  - **What they do not show.** These cells do not depend on the action, so an action-blind
    predictor is expected to tie in them (TASK-076 §3.2). They say nothing about an
    action-conditioned predictor.
- **The primary question** (Stages O, D, S/U) is unaffected by this note.

**Why cell A was removed.** Source: the Stage-0 record
[apple_plate_twin_v2_stage0.md](apple_plate_twin_v2_stage0.md) §3.2–§3.3, ruling R8.16.
- Under H-final, cell A's plate moved a median of 0.060 cm (maximum 0.076 cm) after the last
  decision at 485. The rule needed 2 cm.
- The declared remedies (L = 1, then s1 = 535 … 605) gave 0.022–0.063 cm.
- The mechanism: e9's transfer brings the palm to its aim before 483. The palm's xy moved a
  median of 0.12 cm between 483 and 523, and the lower (505–555) is vertical.
- Before 485 the rule did move the plate a lot: over 405–525 it moved 4.9–7.4 cm in −x and
  2.1–4.1 cm in −y. These are smoke mechanics, with re-decisions at every step, and are not read
  as a result.

**The lessons that bind any next design.**

| lesson | source | what it requires here |
|---|---|---|
| Readout bars are calibrated from measured ceilings, on the representation that is gated | TASK-074 results §5; PLAN.md | every precision bar comes from a ceiling measured in this condition, on TASK-066's pooled 4 × 4 latent (R8.2) |
| A critic or planner needs measurable headroom first | TASK-073 results (S-NO-CONDITION: at most +3/32 against +4) | a K0-style check with no world model comes before any LeWM training |
| The consequence must depend on the robot's action | R8.3 | the action-blind twin must be expected to lose, from a stated calculation |
| An action-blind predictor can still tie | R8.8 | declare the controller form, the training-action distribution relative to the answer, and the action-blind twin's expected result |
| A hand-written arm that knows a declared rule ties | TASK-076 §3.2 and §6.3 (H-rule) | under R9.8 this is no longer disqualifying (§3); it decides only the secondary claim |
| A budget rule must not escalate by design | TASK-074; `run_tools.check_budget` | any training block passes `check_budget` |
| Only horizons up to h = 16 are gated | TASK-066 (h = 8 and h = 16, history one, train split only) | the next task gates its own horizon offline before any closed loop, with a declared failure row |

**Closed scopes that stay closed.** The verbatim scopes are in the cited results documents.
- **TASK-054:** CEM over that world-model cost as the primary control line.
- **TASK-057:** no third control formulation is preregistered on `apple-wide-v1` and the
  112 px onboard camera.
- **TASK-062:** in-corpus encoder training on `apple-wide-v1`.
- **TASK-065:** an action-conditioned predictor on frozen pooled CLS latents on `apple-look-v1`.
- **TASK-075:** a LeWM planner or critic for the place phase of v2, under TASK-074's condition, on
  its four views with frozen DINOv2 features. It stays closed without new evidence of a different
  kind; a task or condition change is one such kind.
- **Not fired:** TASK-073's and TASK-074's clauses. TASK-076's clauses have not been reached.

**Assets, with their qualifiers.**
- **P-3** (`checkpoints/task072-first-policy-v2-linux/run-1/P-3.pt`, sha256 `7988162d…60be8`).
  - It scored 40/40 on cohort C for the whole v2 episode.
  - That is one run with one training seed and 40 resets, on one camera at 112 px onboard, under
    v2's narrow reset distribution: plate (0.49, −0.09) ± 2 cm and apple (0.34, −0.18) ± 3 cm
    (`wm_critic_v2.RESET_CENTERS`, `WIDE_JITTER_M`).
- **e9:** `RestingPlaceExpert(release_pitch_rad=0.45, release_dx=0.015)`.
  - Its pick phases are orient, descend, close and lift (0–405).
  - Its place primitive is `place_planner.PlacePrimitive`: transfer 405–505, lower 505–555, open
    585–635, end at 725.
  - In TASK-070 it left the apple at rest on 32/32 resets with the plate exact, and on 30/32 at
    1.0 cm plate error.
- **TASK-066's token predictor:** pooled 4 × 4 DINOv2 latent, gated at h = 8 and h = 16, history
  one, train split only, with four caveats.
- **TASK-076's code:**
  - the cell-A hook (`plate_twin_v2_runtime.CellMotion`, `plate_twin_v2.palm_driven_xy`);
  - H-final's cloned-state look-ahead (`PlateBrancher` over `sim_selector.Brancher`);
  - H-rule (`RuleAim`);
  - R-plate and R-plate-pool;
  - `place_planner`'s `twin`, `truth`, `shuf` and `rand` modes;
  - `run_tools`.
- **e9-arena** (#123): 13/16 at rest on fresh Arena seeds (development, TASK-025). It matters only
  for the later cross-simulator step (PLAN.md, TASK-078/079).

## 2. What cell A teaches beyond "too late"

Both facts below are checked against the implemented code.

1. **A rule linear in palm velocity does not depend on the path.**
   - `palm_driven_xy` gives plate(s1) = plate(s0) + κ·(palm(s1 − L) − palm(s0)). So only the
     palm's net displacement matters.
   - Choosing a transfer "path" therefore reduces to choosing its end point. If the palm ends
     near the aim g, the plate stops near p + κ(g − h). Here p is the plate and h the palm at the
     decision.
   - The fixed point is g\* = (p − κh)/(1 − κ) (R8.8). `RuleAim` computes it from the reading
     and the stand-in's palm path.
2. **Re-aiming at the current plate already solves for the fixed point.**
   - With a decision every 16 steps, aiming each time at where the plate is now moves the aim
     towards g\* by about a factor |κ| per decision. That is the same contraction H-final's
     look-ahead uses.
   - So whenever the controller may re-decide after seeing the consequence, feedback ties with
     prediction.

**Consequences.**
- A world model's prediction is *used* only if the decision is committed before its consequence
  can be seen and corrected, and the consequence still decides the episode.
- Whether a hand-written arm can match the prediction is a separate question. Under R9.8 (§3) it
  decides the secondary claim, not the primary one.
- The first version of this note tried to defeat hand-written arms by moving to contact physics
  (C2). The review showed that with this scene's constants that does not work (§4.2).

**Rejected without a candidate section.**
- **More constant-velocity cells.** The motion does not depend on the action. An action-blind
  predictor and H-cv are expected to tie (TASK-076 §3.2).
- **A plate that leaves the view during the carry** (R2's other example). Remembering a static
  plate does not depend on the action, so an action-blind predictor, or simply holding the last
  reading, ties.

## 3. The two claims (R9.8, with R9.9)

**Ruling R9.8** (decided by Claude under owner delegation). The product goal is a working
LeWM-driven policy. A design that must show LeWM beats every hand-written model may have no
feasible task in this simulator. So the claim is split in two, and both names are used exactly as
below:

- **Primary claim, "LeWM-driven closed-loop success".** All of the following hold:
  1. LeWM's predictor makes the named decision from the encoded current frame, with no
     privileged read at run time.
  2. The arm meets a calibrated success bar on a fresh gated cohort.
  3. The arm is non-inferior to the best non-world-model arm for the same decision, within a
     declared margin and test.
  4. The arm beats the action-blind and scene-blind twins with a preregistered test. Beating them
     shows that LeWM's prediction is used.
  5. **The arm also beats a random choice with a preregistered test** (ruling R9.9). TASK-076 §9
     fixed this test "for every later task". R9.8 adds requirements to it and supersedes nothing
     in it.

  Items 2 and 3 must **both** hold. Meeting the bar without non-inferiority is not a primary pass,
  and neither is non-inferiority without the bar.
- **Secondary claim, "LeWM needed".** LeWM is detectably better than the best hand-written arm.
  It is reported only and never gates anything.

A primary pass never supports a "LeWM needed" statement. R9.8 and R9.9 refine TASK-076 §9 for
tasks after TASK-076. They do not change TASK-076.

**Proposed test forms.** These are proposals. Each is set at the preregistration, with its power
simulated before the freeze, as R8.14 did.
- **Success bar.** The ceiling measured in K0, minus a declared margin, in the τ family of
  TASK-076's G1.
- **Twin and random tests.** An exact one-sided McNemar test at p < 0.01 (G2's form), for W
  against N, W against L-shuf, and W against L-rand.
- **Non-inferiority.**
  - Test: the lower bound of the paired, reset-clustered 95 % bootstrap interval of W minus the
    best non-world-model arm must lie above −δ.
  - Proposal: δ = 8/64 (that is, 4/32) on a 64-reset gated cohort.
  - Why 8/64:
    - It is half of the +8/32 headroom scale that C1-F3 requires between the ceiling and each
      twin. A W at the margin therefore still sits above every twin the check separated from the
      ceiling.
    - It equals the 4/32 stop margin K0 already uses (R8.7).
    - On TASK-076's measured τ curve (R8.19: 32, 31 and 28 of 32 at 0, 0.5 and 1.0 cm), 4/32 is
      about the cost of one τ_re (1.0 cm) of extra aim error. So W may be at most about one
      tolerance less precise than the best hand-written arm.
  - Power is simulated at true differences of 0 and −4/64.
- **The non-inferiority comparators are single-commit arms only.** The re-aiming arm
  (H-now-reaim) uses a different controller structure, so it is reported and is not a comparator.

## 4. Candidates

All candidates keep the following:
- v2's counted success (T71-R1/R2: at rest after a latched grasp and place).
- The same arm names:
  - **W:** TASK-066's predictor family on the pooled 4 × 4 latent. It is rolled out once per
    candidate, and the plate is read from the predicted latent with R-plate-pool.
  - **N:** the action-blind twin, an equally trained predictor that gets no actions.
  - **L-shuf:** the scene-blind twin, W rolled out from another reset's encoded frame (TASK-074's
    `shuf`).
  - **L-rand:** a random candidate.
  - **H-sim / H-final:** the privileged ceiling, simulated in cloned state.
- **One ceiling bar everywhere:** ≥ 30/32 (K-P1's form), in the smokes and in the protocol (review
  item 7).
- **What is attributable:** every arm shares the same pick and the same downstream primitive, so
  only differences *between arms* are attributable to the decision (review item 5).

### 4.1 Candidate C1: a single committed aim under the reactive-plate rule (recommended)

**A departure from R8's Branch B wording, stated plainly** (R9.2). R8 recommended "a design note
for a *different* action-dependent target". C1 keeps cell A's target and rule and changes only the
controller, to a single aim committed at 405.

This is not a reopening of cell A:
- **Why cell A was removed.** It was removed for one reason: under re-decisions, the decision at
  485 came too late to move its target (R8.16). C1 changes the decision time, which is the factor
  that failed.
- **No clause is in the way.** PRED-INFEASIBLE and PRED-NOT-RUN carry no clause, and K-pred's
  clause (PRED-NONE) was never reached.
- **It is declared as what it is.** C1 is a condition and controller change under R2, declared as
  such, never presented as a silent reopening.

**The condition.**
- **Reset.** v2 with its own reset, unchanged: plate ± 2 cm, apple ± 3 cm, and no step-300 shift.
  No jitter knob is added (review item 9).
- **The plate rule.** Cell A's rule as implemented in `CellMotion`: κ = −0.5, L = 2, s0 = 405,
  s1 = 525. |κ| ≥ 1 stays forbidden (R8.10).
- **One controller change.** A single aim is committed at **c = 405**. e9's primitive then runs
  the transfer, lower and open with no re-aim.
  - §2 shows why the commitment is required: with re-aims, feedback ties.
  - The commitment is an imposed restriction, and any claim is limited to single committed aims.
- **Why c = 405 and not later.**
  - At 405 the plate has not yet moved: palm velocity before s0 counts as zero (R8.15). So for H-rule and
    H-sysid, R-plate reads a static plate inside its training distribution.
  - **The missed-history term has no effect at 405, because of the clamp, though it is not literally
    zero.**
    - The rule uses palm(max(t − L, s0)), so palm(403) never enters plate(s1).
    - m2 = |κ|·‖palm(405) − palm(403)‖ itself need not be zero.
  - **What the frame cannot show at 405 is the palm's own velocity at the end of P-3's handover.** It
    shapes the palm path after 405. C1-F1 records it.
  - At 437 or 453 the palm is moving fast. The clamp no longer applies, and the 2-step term could
    reach |κ| × 2 steps × 1.5 cm, up to 1.5 cm (*estimate*). That is beyond what TASK-066's
    history-one predictor sees.

**The question.** With a single aim committed at 405 under the reactive-plate rule, does W reach
the primary claim, "LeWM-driven closed-loop success"? W predicts each candidate aim's plate
outcome. The primary claim requires all of the following:
- the calibrated bar;
- non-inferiority to the best of H-rule and H-sysid on the reading;
- beating N, L-shuf and L-rand.

The secondary claim, "LeWM needed", is reported only: whether W is detectably better than H-rule
and H-sysid.

**Where LeWM sits.** At step 405 only.
1. W encodes the onboard 112 px frame.
2. It scores every candidate in the declared grid (below) by |predicted plate(r) − the candidate's
   landing point|, where r is the read step.
3. It refines with at most 10 single-candidate roll-outs, g ← predicted plate(r | g), stopping at
   τ/4. Each refined aim is **clipped to the grid's box**. This mirrors H-final and H-rule. Every
   step is an action-conditioned roll-out of one candidate.
4. P-3 picks before 405, and e9's primitive executes the committed aim.

The claim names the decision: "LeWM-driven aim selection at 405".

**The read step r.** r is the earliest step at which H-final's plate is within 0.1 cm of
plate(s1) on at least 87.5 % of the smoke attempts. C1-F1 measures it.
- Stage-0's median of 0.060 cm of movement after 485 suggests r ≤ 485 (*estimate*).
- That gives a horizon of r − 405 ≤ 80 steps, about 5 recursive chunks of 16.
- **At r the plate has moved** about 5–8 cm towards the robot (Stage-0: 4.9–7.4 cm in −x,
  2.1–4.1 cm in −y), and the hand and apple are over the aim. That is where W's readout must work
  (see "Ceilings").

**The candidate grid and the training aims** (R8.8; review item 1b; declared now, before any
smoke). Coordinates are relative to the plate reading p̂ and the proprioceptive palm h at 405:
g = p̂ + a·(h − p̂) + b·n, where n is the unit vector perpendicular to h − p̂.
- **The box.** a ∈ [a_lo, a_hi] and b ∈ [−3, +3] cm.
  - a_hi = 0.5. It covers g\*, which lies at a = 1/3, with a margin.
  - a_lo is the most negative of −0.5, −0.4, −0.3 and −0.2 whose candidates are reachable: at
    least 31/32 transfers complete in C1-F2's reach check.
  - a_lo is set by reach only, never by a headroom number.
- **The controller's grid** is the box at steps of 0.05 in a and 1 cm in b. The refinement is
  clipped to the box. Every arm that chooses from candidates uses the same grid and the same clip.
- **The training aims.** A privileged scripted collector commits e9's aim uniformly over the same
  box, using the true p and h. W is therefore never asked to extrapolate outside its training
  aims.
- **What N does.**
  - The aims' mean along h − p is a_m = (a_lo + 0.5)/2. N predicts the corpus-mean outcome,
    which lies at a = 0.5·(1 − a_m), for every candidate.
  - Its controller therefore picks a_N = min(0.5·(1 − a_m), a_hi). With a_hi = 0.5 the clip never
    binds, because a_m ≥ 0.
  - N's landing miss is 1.5·|a_N − 1/3|·|p − h|.

| a_lo (set by reach) | a_m | a_N | N's miss (*estimate*, \|p − h\| ≈ 16–25 cm) |
|---|---|---|---|
| −0.5 | 0 | 0.5 | 0.25·\|p − h\| ≈ 4–6 cm |
| −0.4 | 0.05 | 0.475 | 0.21·\|p − h\| ≈ 3.4–5.3 cm |
| −0.3 | 0.1 | 0.45 | 0.175·\|p − h\| ≈ 2.8–4.4 cm |
| −0.2 | 0.15 | 0.425 | 0.14·\|p − h\| ≈ 2.2–3.4 cm |

**The expected partial tie, stated now.** The less reach allows behind the plate, the closer the
training mean sits to g\*, and the closer N gets to it. At a_lo = −0.2, N's miss of 2.2–3.4 cm maps
on TASK-076's τ curve (R8.19; caveats below) to roughly 13–17/32. That is a marginal headroom.
C1-F3's N-proxy uses the same box, the same clip and the same a_lo.

**What each arm is expected to do** (*estimates*). They come from the rule, taking the palm to end
near the aim; C1-F3 checks each one. The counts are mapped through TASK-076's K0 τ curve
(R8.19): 32, 31, 28, 23, 17, 18, 16, 8 and 5 of 32 at 0, 0.5, 1, 1.5, 2, 2.5, 3, 4 and 5 cm.

**That curve carries four caveats:**
- it is one run;
- it was measured under TASK-074's condition with the step-300 shift;
- the aim was re-aimed at a planted target, not committed;
- it is non-monotone at 2–2.5 cm.

C1's own τ_commit is measured in K0.

| arm | what it aims at | miss of the landing point (*estimate*) | expected count (*estimate*) |
|---|---|---|---|
| H-now | the plate now (H-cv is the same at 405, because the plate is static) | (1 − κ)·\|g\* − p\| = \|p − h\|/2 ≈ 8–12 cm | about 0–5/32 |
| N | the corpus-mean outcome (table above) | 1.5·\|a_N − 1/3\|·\|p − h\| | about 5–8/32 at a_lo = −0.5; about 13–17/32 at a_lo = −0.2 |
| L-shuf | W rolled out from the foreign reset's latent with *this* reset's commands (`place_planner` `shuf`). It predicts p′ + κ(g − h), so it lands at the foreign fixed point (p′ − κh)/(1 − κ) | exactly \|p − p′\| = \|Δp\| | median about 1.9–2.0 cm; about 20/32 |
| L-rand | a uniformly random candidate in the box | large | low |
| H-rule | `RuleAim` on the R-plate reading, single commit | about the reading error | near the ceiling |
| H-sysid | a linear regression of plate(r) on the R-plate reading, the proprioceptive palm and the aim, fitted on the same corpus as W | about the reading error | near the ceiling |
| W | its own roll-outs | the predicted-plate error B | non-inferior if B is close to c_plate |

How these are derived:
- **The palm-to-plate distance.** |p − h| = 1.5·|g\* − h|. Stage-0's plate travel (4.9–7.4 cm in
  −x, 2.1–4.1 cm in −y) is |κ| times the palm's travel. So the palm travelled about 11–17 cm, and
  |p − h| ≈ 16–25 cm.
- **L-shuf, as implemented** (review item 1a).
  - The candidate commands are this reset's end-effector deltas from h.
  - Rolled out from the foreign latent, they give a predicted plate of p′ + κ(g − h). Its fixed
    point lands the apple at a miss of exactly |Δp|, whatever the palms.
  - With v2's ± 2 cm plate jitter, Δp is the difference of two uniform draws per axis. In a
    simulation of 10⁶ draws (computed for this note) its 2-D median is 2.05 cm; a Rayleigh
    approximation gives 1.92 cm.
  - Integrating the τ curve over those draws gives an expected L-shuf of **about 20/32**.
- **The H-rule and H-sysid ties depend on the readout at 405.**
  - In Stage-0's smokes, H-rule scored 0/16 because the smoke R-plate (fitted on 20 static-plate
    roots) was 2–6 cm off on cell A. These are mechanics and are not read.
  - At 405 the plate is static and in distribution, but the tie still assumes R-plate's
    precision.
  - **K-pred's reported-only M cells (§1) show the realistic size of a perception-fed
    extrapolator's loss on a moving plate.** H-cv, using R-plate readings, reached 24/32 in both
    cells against a ceiling of 32/32.
    - C1's hand-written arms read the plate once, at 405, while it is still static, so the M cells
      do not transfer to them directly.
    - W, however, reads the plate on a predicted frame at r, after it has moved. The M cells are
      a caution that readout errors on moved plates are not small: H-twin reached only 11/32 and
      4/32 there. Most of that is the cost of not predicting, which H-now's 7/32 and 1/32 bound.
    - C1-F5 measures that readout error at r directly.
- **The claim-row competitors** are H-rule and H-sysid on the R-plate reading, the same
  non-privileged reading W gets (review item 8). Their true-plate versions are reported as their
  ceilings.
- **Feedback, for information.** H-now-reaim (TASK-076's H-now with decisions at 405–485) is
  reported only. It is not a comparator (§3).

**The scene-blind contrast is marginal, and it is kept as implemented** (ruling R9.10, decided by
Claude under owner delegation).
- **The margin.** Against a ceiling of 30–32/32, L-shuf at about 20/32 leaves a headroom of about
  10–12/32. That is marginal against C1-F3's +8/32 bar with R8.14's noise guard on 32 smoke
  resets.
- **The gated test has more room than the smoke check.** On 64 gated resets, a W near the ceiling
  against an L-shuf near 40/64 gives an expected 18 or so discordant pairs one way. McNemar at
  p < 0.01 needs 7 (R8.7).
- **No stronger scene-blind control is added.** A mean-latent twin would land at the centre of
  the reset distribution, miss by |p − p̄| (median about 1.4 cm, *estimate*), and so be weaker.
- **The reset jitter is not widened.** It is the free knob that review item 9 asked to keep
  fixed, and widening it would manufacture the contrast.
- **So C1-F3 must show** (bars in the next table):
  - the shuf-proxy at least 8/32 below the ceiling;
  - G2's predicted feasibility for the 64-reset twin test, from the proxy's paired counts as in
    R8.12, at a pass probability of at least 0.8.

  Otherwise C1 stops before any protocol, as a recorded design failure with no clause.

**Feasibility checks first.** CPU smokes with no world model, on a newly declared smoke block that
is checked against every range on main. The code is TASK-076's, plus a commit-at-405 mode for
H-final, H-now, H-rule and the proxies.

| check | what is measured | bar |
|---|---|---|
| C1-F1: remaining motion and history | under H-final(commit): the median \|plate(s1) − plate(405)\|; r (as defined above); and the palm's speed at 405 | remaining motion ≥ 2 cm (TASK-076's bar). r and the palm speed are recorded. If the median palm speed at 405 exceeds 0.5 cm per step (declared now), the protocol gives W and N the last 2 executed commands (R8.7's alternative) |
| C1-F2: ceiling and reach | H-final(commit) on 32 smoke resets, with refused moves counted as failures; the reach check for a_lo | ceiling ≥ 30/32; refusals ≤ 8/32 (TASK-076's removal rule); a_lo set by the reach rule |
| C1-F3: the twins lose (privileged proxies, no world model) | **N-proxy:** aim at a_N, using the same box and clip, from the true p and h. **Shuf-proxy:** the foreign fixed point (p′ − κh)/(1 − κ), with the true p′ of reset (i + 1) mod n and *this* reset's h. **H-now** | each ≤ ceiling − 8/32, with R8.14's guard. For the shuf-proxy, G2-style predicted feasibility ≥ 0.8 |
| C1-F4: the best non-world-model arm | H-rule and H-sysid on a smoke R-plate (single commit), with their true-plate versions; H-now-reaim (reported only) | reported, no bar. They set the non-inferiority comparator |
| C1-F5: readout at r | cross-fitted R-plate-pool on corpus frames **at r** (moved plate, hand and apple over the aim); the same readout on plate-hidden renders of those states (O4's form); also R-plate at 405, for H-rule and H-sysid | at r: median error ≤ τ/2 (an estimate; c_plate itself comes from the protocol's Stage O). Plate-hidden at r: the lower bound of the median error > τ (O4's form) |
| C1-F6: cost | an H-final(commit) attempt, with at most 10 look-ahead roll-outs at a single decision | ≤ 60 s, as in TASK-076 |

If a check fails:
- **C1-F1 or C1-F2:** C1 is infeasible. Escalate, no clause.
- **C1-F3** (a proxy not detectably below its bar, or the shuf feasibility below 0.8): stop before
  any protocol, as a recorded design failure with no clause.
- **C1-F5:**
  - If the plate-hidden check fails, the readout at r may be keyed on the arm. Escalate, no
    clause.
  - If the readout at r misses its estimate, no B is feasible (NO-BAR). Escalate, no clause.

**Ceilings that calibrate its bars.**
- **The count ceiling:** H-final(commit), ≥ 30/32.
- **τ_commit:** the place's tolerance to a planted error in the committed aim. It is re-measured
  in the protocol's K0, as TASK-075's τ was. The committed aim gets no late correction, so
  τ_commit may differ from τ_re.
- **c_plate:** measured on the pooled latent **at r, on the declared-aim corpus's frames**, in
  O1's form (R8.2).
  - This is where the plate has moved 5–8 cm, outside R-plate's static-plate training
    distribution: Stage-0 §3.2's coverage caveat, restored here.
  - The hand and apple are over the aim, which is TASK-076's O4 risk. A readout keyed on the arm
    would return about the aim for every candidate. W's ranking would then confirm itself and
    degenerate.
  - So O4's plate-hidden check runs at r too, and must hold before any B is set.
- **B:** the predicted-latent plate bar, with c_plate(r) ≤ B ≤ τ_commit. Any allowance above
  c_plate is calibrated on development data.
- **R-plate at 405** bounds H-rule and H-sysid, not W.
- **The non-inferiority comparator:** the better of H-rule and H-sysid on the reading, single
  commit.

**The horizon gate and its row** (review item 6).
- W's dynamics gates run offline at h = r − 405, before any closed loop:
  - no collapse;
  - beats copy-last;
  - beats an equally trained N;
  - action-sensitive.
- **On failure: row H-GATE-FAIL.** It escalates without a clause, closes nothing, and never
  reaches back over TASK-066's h ≤ 16 result.
- A remedy would be a reviewed amendment that trains with `action_chunk` or
  `predictor_step_embedding`. Both are off by default in `VisualModel.defaults`, and either needs
  its own preregistration.

**The abandonment clause and its scope.**
- **When it fires:**
  - W fails a twin or random test (L-NO-GAIN);
  - W is detectably inferior to the best single-commit non-world-model arm: the upper bound of
    the paired interval lies below −δ.
- **No clause, escalate:**
  - W misses non-inferiority within noise (L-NEAR);
  - H-GATE-FAIL;
  - NO-BAR, or a failed plate-hidden check at r;
  - a failed ceiling;
  - a budget escalation.
- **Scope:** "LeWM aim selection with a single aim committed at 405 under the declared
  reactive-plate rule (κ = −0.5, L = 2, s1 = 525) on v2, from onboard 112 px frozen DINOv2 pooled
  tokens, with TASK-066-family predictors". It never closes the backend, v2 or the product goal.

**GPU.** This is PLAN.md's TASK-077 sizing:
- a new corpus under the rule, about 4 min of CPU;
- DINOv2 featurisation, about 20 min of GPU;
- W and N on three seeds.

That totals about 6.5 h of GPU at TASK-074's measured times (2 664 s of calibration for 60 000
updates; 3 431–3 582 s per model at 80 000), and 10–11 h worst case at a `check_budget`-passing cap
of 120 000. It runs in per-job slots under the shared lock. L-shuf needs no extra model, and the
smokes need no GPU.

**Risks.**
- **The horizon.** About 80 steps (*estimate*), five recursive chunks beyond TASK-066's gate. This
  is the main risk to the primary claim.
- **The readout at r.** The plate has moved off R-plate's static distribution, and the hand and
  apple are over the aim. c_plate(r) may be much worse than TASK-076's static-plate number, and
  the plate-hidden check may fail.
- **Marginal twin contrasts.**
  - L-shuf is expected at about 20/32 (R9.10).
  - N comes close to g\* if reach forces a_lo up towards −0.2.
- **Non-inferiority is demanding.** H-rule and H-sysid are expected near the ceiling, so W must be
  nearly as precise, with B close to c_plate(r) after an 80-step roll-out.
- **The secondary claim is expected to fail.** "LeWM needed" is not expected here, because a
  linear regression captures the rule. This is stated now.
- **The commitment and the rule are imposed.** The single commitment exists to make prediction
  matter, and the rule is a simulator law; neither transfers as such to Arena or the real G1. What
  transfers is the machinery: a LeWM decision in closed loop, its gates and its twins.

**Earlier clauses.**
- **TASK-075.** Its scope is the place phase under TASK-074's condition. C1 is a place-phase
  decision under a different, declared condition: no step-300 shift, and a reactive plate. A
  condition change is one of that scope's own kinds of new evidence. C1 must be declared under R2
  as a task change and never presented as a reopening.
- **TASK-076.** Its K-pred clause fires only on PRED-NONE, which was never reached. The departure
  from R8's "different target" wording is stated at the top of this section.
- **TASK-057.** C1 uses the same 112 px onboard camera but a new corpus on v2. It is therefore
  outside a clause scoped to `apple-wide-v1` on that camera, the same reasoning that admitted
  TASK-071 to TASK-076.
- **TASK-054.** C1 ranks one decision and then refines it, with a plate-position cost. That is not
  CEM over that cost as a primary line, and the note does not present sampling-based planning as
  the project's control approach.
- **TASK-062 and TASK-065.** C1 uses frozen DINOv2 pooled patch tokens, so it involves neither
  in-corpus encoder training nor CLS.

### 4.2 Candidate C2: a single pre-pick push of a free plate (re-assessed with the scene's constants; not recommended)

**The condition, as first proposed.**
- The plate becomes a free body. Today it is a static body without a joint (`plate_shift.py`).
- The plate resets outside P-3's zone, with v2's own ± 2 cm jitter around a declared start
  centre (review item 9).
- Before the look, the right hand approaches a pre-contact pose. **That pose is computed from the
  R-plate reading of the start frame, never from simulator truth** (review item 2), so the reading
  error is part of the condition.
- The hand then executes one committed push from a declared grid and returns home. Then the v2
  episode runs.

**The physics, with the scene's real constants** (review item 3; *estimate*):
- **Friction.** The plate's friction is `1 .01 .001`, and the table geom declares none, so it has
  MuJoCo's default sliding coefficient of 1. The combined μ is 1.
- **Hand speed.** The end-effector translation is capped at 0.015 m per step at control_dt 0.05 s
  (`configs/g1_sim_action.json`), so v ≤ 0.30 m/s.
- **Slide after release.** At most v²/(2μg) = 0.09 / 19.6 ≈ 0.46 cm, stopping within 0.03 s
  (under one control step). This does not depend on the plate's mass, so the mass is irrelevant to
  the slide.
- **Workspace.** The right hand's workspace ends at y = −0.03 m in the base frame, which limits
  pushes from the left.
- **Push duration.** A 10 cm push lasts at least about 7 steps at the cap, and about 7–20 steps at
  typical speeds. So h_push sits near TASK-066's gated range. The first version's "40–100 steps"
  was unsupported and is withdrawn (review item 6).

**What follows, stated plainly.**
- The rest pose is set by the hand's end point and the contact geometry, to within about 0.5 cm,
  well below any useful zone (C2-F3's 2 cm minimum).
- The pre-contact pose is set relative to the reading. So the displacement is nearly a function of
  the candidate alone, Δ ≈ f(candidate). A per-candidate displacement table, or a regression on
  (plate reading, candidate), measured on development pushes with W's data budget (**H-sysid**,
  review item 1), solves the ranking with no world model.
- With a fixed world pose instead, f also depends on the start reading. That is a low-dimensional
  regression that about 300 pushes fit.
- **So C2's headroom over H-sysid collapses**, and the first version's main reason ("its
  consequence comes from contact physics, so no hand-written arm is handed the answer") does not
  hold for this scene. C2 is withdrawn as the recommendation.

**Under R9.8, C2 could still give a primary claim, but it is a worse route than C1.**
- The twins should lose: N because the training pushes are uniform over the grid, and L-shuf
  because of the start jitter.
- Non-inferiority to H-sysid is plausible.
- But it costs a new scene variant and a push primitive (about 3–5 days), it puts the plate where
  P-3 may be off its narrow distribution, and its consequence is still a learnable closed form of
  the frame.
- A closed-loop servo push (push while re-reading, stop at the zone) is a standard visual-servo
  task. It would be reported beside H-kin-2 (push, re-read, push), and a pass would be limited to
  open-loop single pushes (review item 4).
- P-3's own re-read of the plate before its place is a second feedback stage, which r_zone
  absorbs.

**If C2 were ever run**, its smokes would keep the first version's C2-F0 to C2-F6 with these
changes:
- C2-F4's maximum is over H-kin, H-kin-cal, **H-sysid** and H-fixed, all on the R-plate reading
  for the claim row, with true-plate versions reported. The ceiling bar is the single ≥ 30/32.
- C2-F1 records h_push with an H-GATE-FAIL row as in C1.
- The start jitter is v2's ± 2 cm.

**Clause and scope:** as in the first version, plus H-sysid in the comparator. **GPU:** about
6.5 h, or 10–11 h worst case, after admission.

**As a later "LeWM needed" study, C2 does not make sense with these constants.** A necessity
study needs a consequence that no low-dimensional table of (reading, action) captures. Examples
are state the frame shows but a hand-written feature set does not name: several interacting
objects, a deformable, or an articulated object. No such condition is designed here. It is left
as an open question for after a primary claim exists (R9.5).

### 4.3 Candidate C3: a contact-sensitive grasp approach (not taken)

**The condition.** A declared grasp-phase disturbance in which the descending hand's contact moves
the apple. Examples:
- an apple resting against a low stop;
- a lower rolling friction, which changes v2's defining friction and so would be a new variant,
  as v2 was from v1.

The pick is e9's scripted pick, with an aim offset chosen at the start of the descend (about
step 130) instead of P-3.

**The question.** When finger contact during the descend moves the apple, does W's ranking of
approach offsets by the predicted apple position at closure reach the primary claim?

**Where LeWM sits.** At about 130: W rolls out each candidate's descend-and-close commands (about
125 steps) and reads the predicted apple position relative to the palm.

**Expected results.**
- N cannot rank.
- An arm that assumes the apple stays put loses whenever contact moves it.
- H-sysid (a table of apple displacement per offset) is expected to be strong, for the same reason
  as in C2.
- **The pick's tolerance to a disturbance is unmeasured.** e9's 32/32 and P-3's 40/40 measure end
  to end success under v2's narrow reset, not grasp tolerance (review item 10).

**Feasibility checks.**

| check | what is measured | bar |
|---|---|---|
| C3-F1 | the disturbance moves the apple under an aim at the current apple, with H-sim ≥ 30/32 | ≥ 2 cm in median |
| C3-F2 | H-sim minus the best of the hand-written arms, H-sysid included | ≥ +8/32 |
| C3-F3 | a pre-grasp apple readout ceiling on the pooled latent, against a measured grasp tolerance curve | below the tolerance |

**Ceilings, clause, GPU.**
- **Ceilings:** H-sim, the grasp tolerance curve, and the apple readout ceiling.
- **Clause:** fires on L-NO-GAIN, or when W is detectably inferior. Its scope is LeWM
  grasp-approach ranking under that disturbance.
- **GPU:** about 6.5 h, or 10–11 h worst case.

**Risks.**
- The apple is this project's weakest readout quantity. TASK-075 reported, without gating, an
  apple target readout of 2.2–3.1 cm, and TASK-074's offset failure tracked the apple term.
- The learned grasp lines on v1 failed their gates (for example TASK-043, learned grasp 0/4),
  although TASK-056's A2 and A3 grasped, lifted and held the apple.
- The disturbance would probably need tuning until the pick breaks.
- It replaces P-3's pick with e9's scripted pick.

**Earlier clauses.** These are respected formally: the pick phase is outside TASK-075's
place-phase scope, and frozen pooled tokens avoid TASK-062 and TASK-065.

## 5. Comparison under R9.8

| | **C1: committed aim, reactive plate** | C2: single push, free plate | C3: contact-sensitive grasp |
|---|---|---|---|
| decision | the aim at 405 | the push before the look | the approach at about 130 |
| consequence from | a declared kinematic rule | quasi-static pushing (slide ≤ 0.5 cm, *estimate*) | finger contact |
| N (action-blind) | misses by 0.14–0.25·\|p − h\| ≈ 2.2–6 cm, depending on reach (*estimate*) | loses: uniform training pushes | loses: cannot rank |
| L-shuf (scene-blind) | misses by \|Δp\|, median about 2 cm; about 20/32, which is marginal (*estimate*) | loses by the start jitter | loses by the apple jitter |
| best non-world-model arm | H-rule / H-sysid, near the ceiling | H-sysid, near the ceiling | H-sysid, probably strong |
| primary claim plausible? | yes, if W holds the moved plate through about 80 steps and the readout at r passes its plate-hidden check | yes, at a higher cost | uncertain: the apple readout is weak |
| secondary claim ("LeWM needed") | not expected | not expected | not expected |
| new development | small: TASK-076's code plus a commit mode | free-plate variant and push primitive, about 3–5 days | disturbance and a pick-aim interface |
| cheapest decisive check | C1-F1 to C1-F3, under 1 h of CPU (*estimate*) | C2-F4, under 1 h of CPU after the development | C3-F2, under 1 h of CPU after the development |
| GPU after admission | about 6.5 h; 10–11 h worst case | the same | the same |

## 6. Recommendation: C1 (decided by Claude under owner delegation, R9.2 as revised)

**Main reasons.**
1. **C1 is the likeliest route to a first primary claim, "LeWM-driven closed-loop success", on
   v2.**
   - The twins' expected losses follow from the rule and v2's fixed reset, not from tuning
     (*estimates*, checked in C1-F3):
     - N misses by 0.14–0.25·|p − h|, depending on reach.
     - L-shuf misses by |Δp|, about 20/32. That is marginal and is said so (R9.10).
   - The comparator arms are known and cheap: H-rule exists, and H-sysid is a regression.
2. **It is the cheapest.** The rule, the look-ahead, H-rule, the readouts and the guards exist in
   TASK-076's code. C1 adds a commit-at-405 mode, and its decisive checks run on the CPU in about
   an hour (*estimate*). No scene variant is needed.
3. **The decision is clean at 405.**
   - The plate is static and inside R-plate's training distribution, for H-rule and H-sysid.
   - The rule's clamp removes the 2-step palm term, so TASK-066's history-one predictor is enough,
     subject to the palm speed at 405 that C1-F1 records.
   - The plate's remaining motion is large (Stage-0 mechanics: several centimetres over 405–525).
4. **Its main risks are measurable before training.**
   - The roughly 80-step horizon has its own offline gate and row (H-GATE-FAIL, no clause).
   - The readout at r, on the moved plate with plate-hidden frames, is checked in C1-F5.
5. **It is honest about what it cannot show.** The secondary claim, "LeWM needed", is expected to
   fail, because H-sysid captures the rule. That is stated now, and it is the reason R9.8 split
   the claim.

**What a primary pass would and would not show.**
- **It would show:** "LeWM-driven aim selection at 405, followed by e9's place, reaches the
  calibrated bar, is non-inferior to the best hand-written arm and beats its blind twins, under a
  declared reactive-plate rule on v2".
- **It would not show:** that LeWM is needed, a LeWM policy, anything about v1 or its 0/150,
  Arena or the real G1.

**Fallbacks.**
- **C2:** only if C1 is infeasible for a reason a push condition would not share, at the cost of
  the scene variant.
- **C3:** not taken (R9.4).
- **A "LeWM needed" study:** later, and only with a condition whose consequence no
  low-dimensional table captures (§4.2).

## 7. Next steps (none started by this note)

1. Re-review of this note (docs only).
2. A development feasibility record for C1, in the style of TASK-076's Stage-0 record:
   - a commit-at-405 mode for H-final, H-now, H-rule, the N-proxy and the shuf-proxy (as
     implemented: the foreign p′ with this reset's h), using the declared box and clip;
   - H-sysid;
   - C1-F1 to C1-F6, on a newly declared smoke block and salts, checked against every range on
     main before any seed is simulated.
3. Only if C1-F1 to C1-F3 pass: a preregistration with these stages, in order:
   - K0 (τ_commit and the ceiling);
   - Stage O (c_plate at r on the declared-aim corpus, the plate-hidden check at r, and R-plate at
     405 for the hand-written arms);
   - the corpus with the declared aim distribution;
   - training under a `check_budget`-passing block;
   - the offline horizon and dynamics gates (H-GATE-FAIL row);
   - a 16-reset development closed loop with a stop rule;
   - the gated cohort, with the two claims of R9.8 and R9.9 and their declared tests.
4. TASK-076's primary question continues independently.
