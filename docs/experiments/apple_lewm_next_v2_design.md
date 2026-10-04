# The next LeWM task on apple-to-plate-v2: candidate designs (DRAFT design note)

**Status: DRAFT design note. It is not a protocol and nothing in it is frozen.** No run was made
for it, no seed was simulated and no bar below is gated. It answers PLAN.md's Branch B entry for
PRED-INFEASIBLE: "a design note for a different action-dependent target, with its own K0-style
headroom check and no world model".

Every choice in it is **decided by Claude under owner delegation (2026-09-30)**. The rulings are
R9.1–R9.8 in [DECISIONS.md](../DECISIONS.md), decision 2026-10-04. This revision answers the
independent review of #135 at `c387b20` (REQUEST CHANGES) and applies R9.8. Any task the note
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

## 1. Where we start (checked against the repository on 2026-10-04, at `2d0bdb7`)

**TASK-076's status.** These lines are written to stay true whether or not #137 (the TASK-076
freeze) merges first.
- **K0:** K0 ran with outcome K0-PASS; see R8.19 once #137 merges.
- **K-pred:** PRED-INFEASIBLE if Stage O passes, PRED-NOT-RUN otherwise. Both rows escalate
  without a clause and lead to PLAN.md's Branch B, which this note answers.
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

## 3. The two claims (R9.8)

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
  4. The arm beats the action-blind and scene-blind twins, which shows that LeWM's prediction is
     used. TASK-076 §9's random-choice arm is kept beside them.
- **Secondary claim, "LeWM needed".** LeWM is detectably better than the best hand-written arm.
  It is reported only and never gates anything.

A primary pass never supports a "LeWM needed" statement. R9.8 refines the definition of
TASK-076 §9 for tasks after TASK-076 and does not change TASK-076.

**The proposed test forms.** They are proposals, set at the preregistration together with their
power.
- **Success bar:** the K0-measured ceiling minus a declared margin, in the τ family of TASK-076's
  G1.
- **Twin tests:** exact one-sided McNemar, p < 0.01, W against N and W against L-shuf (G2's form).
- **Non-inferiority:** the lower bound of the paired, reset-clustered 95 % bootstrap interval of
  W − best non-world-model arm lies above −δ. The proposal is δ = 8/64 on a 64-reset gated
  cohort. Its power at true differences of 0 and −4/64 is simulated before the freeze, as R8.14
  did.

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

**The condition.** It is declared under R2 as a condition change, not a reopening.
- **Reset.** v2 with its own reset unchanged: plate ± 2 cm, apple ± 3 cm, no step-300 shift. There
  is no added jitter knob (review item 9).
- **The plate rule.** Cell A's rule as implemented: κ = −0.5, L = 2, s0 = 405, s1 = 525,
  `CellMotion`. |κ| ≥ 1 stays forbidden (R8.10).
- **One controller change.** A single aim is committed at **c = 405**, and e9's primitive then
  runs the transfer, lower and open with no re-aim. §2 shows why the commitment is required: with
  re-aims, feedback ties. It is an imposed restriction, and any claim is limited to single
  committed aims.
- **Why c = 405, and not a later commit step:**
  - At 405 the plate has not yet moved (palm velocity before s0 counts as zero, R8.15). So
    R-plate reads a static plate inside its training distribution.
  - The missed-history term is exactly zero: m2 = |κ|·‖palm(405) − palm(403)‖ is zero by the
    rule.
  - At 437 or 453 the palm is moving fast. The 2-step palm term the frame cannot show could then
    reach |κ| × 2 steps × 1.5 cm (*estimate*), up to 1.5 cm, beyond what TASK-066's history-one
    predictor sees.

**The question.** With a single aim committed at 405 under the reactive-plate rule, does W, which
predicts each candidate aim's plate outcome, reach the primary claim?
- **Primary claim, "LeWM-driven closed-loop success":** calibrated bar, non-inferior to the best
  of H-rule and H-sysid, beats N and L-shuf.
- **Secondary claim, "LeWM needed"** (reported only): is W detectably better than H-rule and
  H-sysid?

**Where LeWM sits.** At step 405 only.
1. W encodes the onboard 112 px frame.
2. It ranks a declared candidate grid around the plate reading by |predicted plate(r) −
   candidate's landing point|. Here r is the read step, defined below.
3. It refines with at most 10 single-candidate roll-outs, g ← predicted plate(r | g), stopping
   at τ/4. This mirrors H-final and H-rule. Every step is an action-conditioned roll-out of one
   candidate.
4. P-3 picks before 405, and e9's primitive executes the committed aim.

The claim names the decision: "LeWM-driven aim selection at 405".

**The read step r.** r is the earliest step at which H-final's plate is within 0.1 cm of plate(s1)
on at least 87.5 % of the smoke attempts. It is measured in C1-F1.
- Stage-0's 0.060 cm median after 485 suggests r ≤ 485 (*estimate*).
- That gives a horizon of r − 405 ≤ 80 steps, about 5 recursive chunks of 16.

**The training actions** (R8.8, declared now). A privileged scripted collector commits e9's aim at
p₄₀₅ + u, where p₄₀₅ is the true plate at 405 and u is uniform over a box that covers the candidate
grid. The aims are therefore centred on the current plate, not on g\*.

**Why each arm should win or lose, with expected values** (*estimates* from the rule; the palm is
taken to end near the aim; each is checked in C1-F3):

| arm | what it aims at | miss of the landing point (*estimate*) | expected |
|---|---|---|---|
| H-now | the plate now (H-cv is the same at 405, because the plate is static) | (1 − κ)·\|g\* − p\| = \|h − p\|/2 | about 8–12 cm: fails |
| N | the mean outcome over the training aims, (p + h)/2 | \|p − h\|/4 | about 4–6 cm: fails |
| L-shuf | another reset's fixed point | \|Δp + Δh/2\| | median about 2.4 cm (below): mostly fails |
| H-rule | `RuleAim` on the R-plate reading | about the reading error | near the ceiling |
| H-sysid | a linear regression of plate(r) on (R-plate reading, proprio palm, aim), fitted on the same corpus as W | about the reading error | near the ceiling |
| W | its own roll-outs | the predicted-plate error B | non-inferior if B ≈ c_plate |

How the estimates are derived:
- **The palm-to-plate distance.** |p − h| = 1.5·|g\* − h|. Stage-0's plate travel of 4.9–7.4 cm
  in −x and 2.1–4.1 cm in −y equals |κ| × the palm's travel. So the palm travelled about
  11–17 cm, and |p − h| is about 16–25 cm (*estimate*).
- **L-shuf.** Δp has a per-axis standard deviation of about 1.6 cm, from two ± 2 cm draws. Δh/2
  has about 1.2 cm, if the palm at 405 tracks the apple's ± 3 cm reset. The 2-D median is then
  about 2.4 cm, against e9's 30/32 at 1.0 cm plate error (TASK-070).
- **The H-rule and H-sysid ties depend on the readout.** Stage-0's smoke H-rule scored 0/16
  because its smoke R-plate, fitted on 20 static-plate roots, was 2–6 cm off on cell A (mechanics,
  not read). At 405 the plate is static and in distribution, but the tie still assumes R-plate's
  precision.
- **The claim-row competitors.** H-rule and H-sysid on the R-plate reading, the same non-privileged
  reading W gets, are the claim-row arms (review item 8). Their true-plate versions are reported as
  their ceilings.
- **Feedback for information.** The re-aiming arm H-now-reaim (TASK-076's H-now with decisions at
  405–485) is reported, to show what feedback alone achieves without the commitment.

**Feasibility checks first.** These are CPU smokes with no world model, run on a newly declared
smoke block checked against every range on main. The code is TASK-076's, plus a commit-at-405 mode
for H-final, H-now, H-rule and the proxies.

| check | what is measured | bar |
|---|---|---|
| C1-F1, remaining motion | under H-final(commit), the median \|plate(s1) − plate(405)\|; r as defined above | ≥ 2 cm (TASK-076's bar) |
| C1-F2, ceiling | H-final(commit) on 32 smoke resets, refused moves counted as failures | ≥ 30/32, and refusals ≤ 8/32 (TASK-076's removal rule) |
| C1-F3, the twins lose (privileged proxies, no world model) | **N-proxy**: aim at the mean outcome of the declared training aims, from the true p and h, with κ. **Shuf-proxy**: H-rule's fixed point from reset (i + 1) mod n's true p and h | each ≤ H-final − 8/32 (TASK-074 K1's headroom bar); also H-now ≤ H-final − 8/32 |
| C1-F4, the best non-world-model arm | H-rule and H-sysid on a smoke R-plate; their true-plate versions; H-now-reaim | reported, no bar; they set the non-inferiority comparator |
| C1-F5, readout | cross-fitted R-plate-pool on the 405 frames of the smoke corpus | median error ≤ τ/2 (an estimate; c_plate itself comes from the protocol's Stage O) |
| C1-F6, cost | an H-final(commit) attempt, with at most 10 look-ahead roll-outs at a single decision | ≤ 60 s, as in TASK-076 |

If a check fails:
- **C1-F1 or C1-F2 fails:** C1 is infeasible. Escalate, no clause.
- **A C1-F3 proxy is not detectably below its bar** (TASK-076's R8.14 guard): the primary claim's
  twin tests have no expected room. Stop before any protocol, as a recorded design failure with no
  clause.

**Ceilings that calibrate its bars.**
- **The count ceiling:** H-final(commit), ≥ 30/32.
- **τ_commit:** the place's tolerance to a planted error in the committed aim, re-measured in the
  protocol's K0 as TASK-075's τ was. The committed aim gets no late correction, so it may differ
  from τ.
- **c_plate:** on the pooled latent at the 405 frames, in O1's form (R8.2).
- **B:** the predicted-latent plate bar, with c_plate ≤ B ≤ τ_commit. Any allowance above c_plate
  is calibrated on development data.
- **The non-inferiority comparator:** the best of H-rule and H-sysid on the reading.

**The horizon gate and its row** (review item 6).
- W's dynamics gates run offline at h = r − 405 before any closed loop: no collapse, beats
  copy-last, beats an equally trained N, action-sensitive.
- **On failure: row H-GATE-FAIL.** It escalates without a clause, closes nothing, and never
  reaches back over TASK-066's h ≤ 16 result.
- A remedy would be a reviewed amendment that trains with `action_chunk` or
  `predictor_step_embedding`. Both are off by default in `VisualModel.defaults`, and either needs
  its own preregistration.

**The abandonment clause and its scope.**
- **When it fires:**
  - W fails a twin test (L-NO-GAIN);
  - W is detectably inferior to the best non-world-model arm: the upper bound of the paired
    interval lies below −δ.
- **No clause, escalate:** W misses non-inferiority within noise (L-NEAR), H-GATE-FAIL, a failed
  ceiling, or a budget escalation.
- **Scope:** "LeWM aim selection with a single aim committed at 405 under the declared
  reactive-plate rule (κ = −0.5, L = 2, s1 = 525) on v2, from onboard 112 px frozen DINOv2 pooled
  tokens, with TASK-066-family predictors". It never closes the backend, v2 or the product goal.

**GPU.** This is PLAN.md's TASK-077 sizing.
- A new corpus under the rule, about 4 min of CPU.
- DINOv2 featurisation, about 20 min of GPU.
- W and N on three seeds.

That totals about 6.5 h of GPU at TASK-074's measured times (2 664 s of calibration for 60 000
updates; 3 431–3 582 s per model at 80 000), and 10–11 h worst case at a `check_budget`-passing cap
of 120 000. It runs in per-job slots under the shared lock. L-shuf needs no extra model, and the
smokes need no GPU.

**Risks.**
- **The horizon.** About 80 steps (*estimate*), five recursive chunks beyond TASK-066's gate. This
  is the main risk to the primary claim.
- **Non-inferiority is demanding.** H-rule and H-sysid are expected near the ceiling, so W must be
  nearly as precise. B has to sit close to c_plate after an 80-step roll-out.
- **The secondary claim is expected to fail.** "LeWM needed" is not expected here: a linear
  regression captures the rule. That is stated now.
- **The commitment and the rule are imposed.** The single commitment exists to make prediction
  matter, and the rule is a simulator law. Neither transfers as such to Arena or the real G1. What
  transfers is the machinery: a LeWM decision in closed loop, its gates and its twins.
- **P-3 hands over at 405 with the hand's pose varying.** The palm's spread at 405 feeds the
  L-shuf estimate; C1-F3 measures it.

**Earlier clauses.**
- **TASK-075:** its scope is the place phase under TASK-074's condition. C1 is a place-phase
  decision under a different, declared condition (no step-300 shift; a reactive plate). A
  condition change is one of the scope's own kinds of new evidence. It must be declared under R2
  as a task change and never presented as a reopening.
- **TASK-076:** its K-pred clause fires only on PRED-NONE and was never reached.
- **TASK-057:** C1 uses the same 112 px onboard camera but a new corpus on v2, so it is outside a
  clause scoped to `apple-wide-v1` on that camera. The same reasoning admitted TASK-071–076.
- **TASK-054:** this is a ranking of one decision followed by refinement, with a plate-position
  cost. It is not CEM over that cost as a primary line, and the note does not present
  sampling-based planning as the project's control approach.
- **TASK-062 and TASK-065:** frozen DINOv2 pooled patch tokens, so neither in-corpus encoder
  training nor CLS.

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
| N (action-blind) | misses by about \|p − h\|/4 ≈ 4–6 cm (*estimate*) | loses: uniform training pushes | loses: cannot rank |
| L-shuf (scene-blind) | misses by about 2.4 cm in median (*estimate*) | loses by the start jitter | loses by the apple jitter |
| best non-world-model arm | H-rule / H-sysid, near the ceiling | H-sysid, near the ceiling | H-sysid, probably strong |
| primary claim plausible? | yes, if W holds the plate through about 80 steps | yes, at a higher cost | uncertain: the apple readout is weak |
| secondary claim ("LeWM needed") | not expected | not expected | not expected |
| new development | small: TASK-076's code plus a commit mode | free-plate variant and push primitive, about 3–5 days | disturbance and a pick-aim interface |
| cheapest decisive check | C1-F1 to C1-F3, under 1 h of CPU (*estimate*) | C2-F4, under 1 h of CPU after the development | C3-F2, under 1 h of CPU after the development |
| GPU after admission | about 6.5 h; 10–11 h worst case | the same | the same |

## 6. Recommendation: C1 (decided by Claude under owner delegation, R9.2 as revised)

**Main reasons.**
1. **C1 is the likeliest route to a first primary claim, "LeWM-driven closed-loop success", on
   v2.**
   - The twins' expected losses follow from the rule, not from tuning: N misses by about
     |p − h|/4, and L-shuf by about 2.4 cm in median (*estimates*, checked in C1-F3).
   - The comparator arms are known and cheap: H-rule exists, and H-sysid is a regression.
2. **It is the cheapest.** The rule, the look-ahead, H-rule, the readouts and the guards exist in
   TASK-076's code. C1 adds a commit-at-405 mode, and its decisive checks run on the CPU in about
   an hour (*estimate*). No scene variant is needed.
3. **The decision is clean at 405.**
   - The plate is static and in R-plate's distribution.
   - The missed-history term is exactly zero, so TASK-066's history one is enough.
   - The plate's remaining motion is large (Stage-0 mechanics: several centimetres over 405–525).
4. **Its main risk is measurable before training.** The risk is the roughly 80-step horizon. It
   has its own offline gate and its own row (H-GATE-FAIL, no clause).
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
   - a commit-at-405 mode for H-final, H-now, H-rule, the N-proxy and the shuf-proxy;
   - H-sysid;
   - C1-F1 to C1-F6, on a newly declared smoke block and salts, checked against every range on
     main before any seed is simulated.
3. Only if C1-F1 to C1-F3 pass: a preregistration with these stages, in order:
   - K0 (τ_commit and the ceiling);
   - Stage O (c_plate at 405);
   - the corpus with the declared aim distribution;
   - training under a `check_budget`-passing block;
   - the offline horizon and dynamics gates (H-GATE-FAIL row);
   - a 16-reset development closed loop with a stop rule;
   - the gated cohort, with R9.8's two claims and their declared tests.
4. TASK-076's primary question continues independently.
