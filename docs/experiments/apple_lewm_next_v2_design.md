# The next LeWM task on apple-to-plate-v2: candidate designs (DRAFT design note)

**Status: DRAFT design note. It is not a protocol and nothing in it is frozen.** No run was made
for it, no seed was simulated and no bar below is gated. It answers PLAN.md's Branch B entry for
PRED-INFEASIBLE: "a design note for a different action-dependent target, with its own K0-style
headroom check and no world model". Every choice in it is **decided by Claude under owner
delegation (2026-09-30)**; the rulings are R9.1–R9.7 in [DECISIONS.md](../DECISIONS.md), decision
2026-10-04. Any task it leads to needs its own feasibility record, its own preregistration and an
independent review before a single cohort seed is simulated.

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

Nothing in this note is a result. Where it quotes a number, it gives the source document.

---

## 1. Where we start (checked against the repository on 2026-10-04, at `2d0bdb7`)

**TASK-076's Stage 0 removed cell A.** Record: [apple_plate_twin_v2_stage0.md](apple_plate_twin_v2_stage0.md)
§3.2–§3.3, ruling R8.16.
- Under H-final, cell A's plate moved a median of 0.060 cm (maximum 0.076 cm) after the last
  decision at 485, against the 2 cm the rule needed. The declared remedies (L = 1, then
  s1 = 535 … 605) gave 0.022–0.063 cm. Cell A was removed, so K-pred's row is PRED-INFEASIBLE
  (escalate, no clause), and PLAN.md's Branch B applies.
- The mechanism, from the same record: e9's transfer brings the palm to its aim before 483 (the
  palm's xy moved a median of 0.12 cm between 483 and 523), and the lower (505–555) is vertical.
  So, within e9's place phase structure, the last aim cannot move the target.
- Before 485 the rule did move the plate a lot: over 405–525 it moved 4.9–7.4 cm in −x and
  2.1–4.1 cm in −y (smoke mechanics, not read as a result).

**The lessons that bind any next design.**

| lesson | source | what it requires here |
|---|---|---|
| Readout bars are calibrated from measured ceilings, on the representation that is gated | TASK-074 results §5; PLAN.md | every precision bar is set from a ceiling measured in this condition, on TASK-066's pooled 4 × 4 latent (R8.2) |
| A critic or planner needs measurable headroom first | TASK-073 results (S-NO-CONDITION: +3/32 at most against +4) | a K0-style headroom check with no world model comes before any LeWM training |
| The consequence must depend on the robot's action | R8.3 | an action-blind predictor must be expected to lose, by construction, under the declared controller |
| An action-blind predictor can still tie | R8.8 | the controller ranks candidates by their individual predicted outcomes; the training actions are not centred on the answer; the action-blind twin's expected result is declared |
| A hand-written arm that knows a declared rule ties | TASK-076 §3.2 (H-rule), §6.3 | see §2: a declared kinematic rule hands the answer to a hand-written arm |
| A budget rule must not escalate by design | TASK-074; `run_tools.check_budget` | any training block passes `check_budget` (cap ≥ factor × calibration) |
| Horizons beyond h = 16 are ungated | TASK-066 (h = 8 and h = 16 only, history one) | the next task gates its own horizon offline before any closed loop |

**Closed scopes that stay closed** (verbatim scopes are in the cited results documents):
- TASK-054: CEM over that world-model cost as the primary control line.
- TASK-057: the behaviour-cloning line on `apple-wide-v1` at 112 px onboard.
- TASK-062: in-corpus encoder training on `apple-wide-v1`.
- TASK-065: an action-conditioned predictor on frozen pooled CLS latents on `apple-look-v1`.
- TASK-075: a LeWM planner or critic for the **place phase** of v2 **under TASK-074's condition**,
  on its four views with frozen DINOv2 features, without new evidence of a different kind.
- TASK-073 and TASK-074 did not fire their clauses; TASK-076's two clauses have not been reached
  (its primary stages wait for K0's GO, and K-pred is PRED-INFEASIBLE, which carries no clause).

**Assets.**
- P-3 (`checkpoints/task072-first-policy-v2-linux/run-1/P-3.pt`, sha256 `7988162d…60be8`): 40/40
  on cohort C, the whole v2 episode (pick and place).
- e9 (`RestingPlaceExpert(release_pitch_rad=0.45, release_dx=0.015)`), its pick phases (orient,
  descend, close, lift; 0–405) and its place primitive (`place_planner.PlacePrimitive`; transfer
  405–505, lower 505–555, open 585–635, end 725).
- TASK-066's token predictor: pooled 4 × 4 DINOv2 latent, gated at h = 8 and h = 16, history one,
  train split only, with four caveats.
- TASK-074/076 machinery: cloned-state look-ahead (`sim_selector.Brancher`), the plate hooks
  (`plate_shift.move_plate`), `place_planner` modes `twin`, `truth`, `shuf` and `rand`, R-plate and
  R-plate-pool, `run_tools`.
- e9-arena (#123): 13/16 at rest on fresh Arena seeds (development, TASK-025). Not used by any
  candidate below; it matters only for the later cross-simulator step (PLAN.md TASK-078/079).

## 2. What cell A teaches beyond "too late"

Two structural facts follow from the cell-A record, and both apply to any rule of its kind:

1. **A rule linear in palm velocity is path-independent.** Stage 0's discretisation is
   plate(t) = plate(s0) + κ·(palm(max(t − L, s0)) − palm(s0)) (R8.15). So the plate's final
   position depends only on the palm's net displacement, not on the path. Choosing a "transfer
   path" collapses to choosing an end point, and the end point's fixed point,
   g\* ≈ (p − κh)/(1 − κ), is a closed form of the current frame and proprioception (R8.8).
   H-rule computes it with no world model.
2. **Re-aiming at the current plate is itself a fixed-point solver.** With decisions every 16
   steps, aiming each time at where the plate is now contracts towards g\* by about |κ| per
   decision, the same contraction H-final's look-ahead uses. Feedback substitutes for prediction
   whenever the controller may re-decide after it sees the consequence.

So the design requirement is sharper than R8.3. A world model earns a decision only if **(i)** the
consequence depends on the action, **(ii)** it is generated by physics the controller is not
handed as a formula (otherwise a hand-written arm that knows the formula ties), **(iii)** it is
committed before it can be observed and corrected (otherwise feedback ties), and **(iv)** it
happens early enough that its consequence still decides the episode. Any declared kinematic rule
fails (ii) by construction; that is why only physics-borne consequences are taken further below.
A rule could be made nonlinear (a speed threshold, a dead zone), but any declared rule is still a
formula a hand-written arm can be given.

**Rejected without a candidate section:**
- **More constant-velocity cells.** Action-independent; an action-blind predictor and H-cv are
  expected to tie (TASK-076 §3.2). Fails (i).
- **A plate that leaves the view during the carry** (R2's other example). Memory of a static plate
  is action-independent; an action-blind predictor (or holding the last reading) ties. Fails (i).

## 3. Candidates

Each candidate keeps v2's counted success (T71-R1/R2: at rest after a latched grasp and place) and
the definition of "a LeWM-driven closed-loop success" in [apple_plate_twin_v2.md](apple_plate_twin_v2.md)
§9. Each would be a task or condition change declared under R2, never a silent reopening.

Shared arm names:
- **H-sim** — privileged ceiling: picks among the declared candidates by simulating each in cloned
  state (TASK-073's H-sim, TASK-076's H-final).
- **N** — the action-blind twin: the same controller with an equally trained predictor that gets
  no actions.
- **L-shuf** — the scene-blind twin: W rolled from another reset's encoded frame (TASK-074's
  `shuf`).
- **L-rand** — a uniformly random candidate.
- **W** — the LeWM arm: TASK-066's predictor family on the pooled 4 × 4 latent, ranking every
  candidate by its own predicted outcome (one roll-out per candidate).

### 3.1 Candidate C1 — early commitment under a reactive plate (direction (a))

**The condition.** Cell A's rule (κ = −0.5, L = 2, s0 = 405, s1 = 525; |κ| ≥ 1 stays forbidden,
R8.10), with one change to the controller structure: **one decision at 405 commits the transfer
aim**, with no re-aim at 421–485. By §2, the decision has to be a commitment; otherwise feedback
ties.

**The question.** Under a declared action-dependent plate rule, with a single committed aim at the
start of the transfer, does LeWM's ranking of candidate aims by predicted plate position at s1
give more counted successes than N, L-shuf, L-rand, H-now and H-cv?

**Where LeWM sits.** At step 405 only: W encodes the current frame, rolls each candidate aim's
commands (405 → 523, about 118 steps) forward, reads the predicted plate with R-plate-pool, and
commits the aim whose predicted plate lies closest to its own landing point. P-3 picks; e9's
primitive executes the transfer, lower and open.

**Why the blind and hand-written arms should lose, and the expected ties.**
- N predicts one plate for every candidate, so it cannot rank; it loses only because the
  controller ranks and the training aims are spread uniformly, not centred on g\* (R8.8).
- H-now (aim at the plate now) and H-cv (constant-velocity extrapolation) ignore that the plate's
  future depends on the aim; they should lose by the size of the remaining motion.
- **Expected tie, stated now: H-rule reaches about the ceiling.** By §2 the target is
  g\* = (p − κh)/(1 − κ), a closed form of the frame and proprioception. W can at best tie H-rule;
  any gap between them is only the primitive's tracking error. The claim could never be that a
  world model is needed (the TASK-076 §6.3 wording carries).

**Feasibility checks first (CPU smokes, no world model, smoke seeds only).** The cell-A hook and
H-final's look-ahead exist (`plate_twin_v2.py`), so these are small.
- C1-F1, remaining motion: under H-final with a single commit at 405, the median
  |plate(s1) − plate(405)| ≥ 2 cm (TASK-076's bar). The Stage-0 mechanics suggest it will pass
  (4.9–7.4 cm in −x over 405–525), but those were with re-decisions.
- C1-F2, ceiling: H-final(commit) ≥ 30/32 on 32 smoke resets.
- C1-F3, refusals: at most 1/4 of moves refused (TASK-076's removal rule).
- C1-F4, the tie that kills it: H-rule(commit) − H-final(commit) is reported; if
  H-rule ≥ H-final − 2/32, C1 can produce no claim beyond "LeWM can drive it".

**Ceilings that calibrate its bars.** H-final(commit) (the count ceiling); c_plate on the pooled
latent for cell A's displaced plates (R8.2; the static-plate c_plate of TASK-076 does not cover
the −x displacements, Stage-0 §3.2's coverage caveat); a re-measured τ under the commit (the
committed aim has no late correction, so τ may shrink); the predicted-latent plate bar B with
c_plate ≤ B ≤ τ.

**Abandonment clause and scope.** Fires if W fails to beat N, L-shuf and L-rand (L-NO-GAIN), or
falls clearly below H-rule (clause, no claim; TASK-074's L-TWIN-BETTER precedent). Scope: "LeWM
aim selection with a single committed aim at 405 under the declared reactive-plate rule (κ, L, s1
as run) on v2, with TASK-066-family pooled-token predictors". Never the backend, v2 or the product
goal.

**GPU.** PLAN.md's TASK-077 sizing: a new e9 corpus under the rule (about 4 min CPU), DINOv2
featurisation (about 20 min GPU), calibration plus six models (TASK-074: 2 664 s for 60 000
updates; 3 431–3 582 s per model at 80 000), about 10–11 h worst case at a `check_budget`-passing
cap of 120 000. The horizon gate adds offline evaluation only.

**Risks.**
- The decisive one: H-rule ties at the ceiling, so C1 cannot show that a world model is needed.
- The horizon is about 118 steps, about 7 recursive chunks of 16, far beyond TASK-066's gate.
- The rule is an artificial simulator law, and its single-commit structure is imposed only to
  defeat feedback; neither transfers to Arena or the real G1.
- R-plate coverage of displaced plates is unmeasured.

**Earlier clauses.** Respected formally: it is a place-phase task, but not under TASK-074's
condition (cell A has no step-300 shift), so TASK-075's scope does not cover it, and TASK-076's
K-pred clause never fired. It must still be declared as a task change under R2, and it sits close
enough to the closed place-phase scopes that a reviewer would rightly ask what is new beyond the
commit. Not a CEM primary line (TASK-054): a discrete ranker for one decision.

### 3.2 Candidate C2 — a single pre-pick push of a free plate into P-3's zone (directions (b) and (c); recommended)

**The condition** (a declared scene variant of v2, provisional name `apple-to-plate-v2-push`,
separate from v2 as v2 is from v1):
- The plate becomes a free body (today it is a static body without a joint, `plate_shift.py`),
  with its existing geometry and the scene's own declared plate friction `1 .01 .001`, and a mass
  fixed from a physical prior (a small saucer, about 0.2 kg) **before any smoke, never tuned on a
  headroom number**.
- The plate resets in a declared start region outside P-3's working zone (P-3 was trained and
  tested with the plate at (0.49, −0.09) ± 2 cm, `wm_critic_v2.RESET_CENTERS`, `WIDE_JITTER_M`).
  For example, nearer the robot and to the left of the apple, from where a push away from the
  robot brings it into the zone. The region's centre and jitter are set in the feasibility record
  (C2-F1, C2-F3), before C2-F4, and not tuned on C2-F4.
- **One push per episode, committed before it is executed.** Before the look, the right hand (idle
  until the pick) runs a scripted approach to a pre-contact pose behind the plate (independent of
  the decision), then executes one candidate push, then returns to its home pose. Then the
  ordinary v2 episode runs: the look, P-3's pick and P-3's place (or e9's place aimed at the
  R-plate reading; see risks).
- A push candidate is a short command sequence from a declared grid: lateral contact offset ×
  push direction × push length × speed. Fast candidates leave the plate sliding after the hand
  stops, so the plate's rest pose is set by contact and friction, not by the hand's end point.

**The question.** When the plate's end position after a single push is set by contact physics,
does a LeWM predictor that ranks candidate pushes by their own predicted plate rest positions
give more counted successes than the action-blind, scene-blind and random twins (the claim row,
§9's definition)? And does it also beat the best hand-written push model, H-kin-cal (a second,
stronger claim row: a calibrated hand-written model does not suffice)?

**Where LeWM sits.** One decision, at the pre-contact pose, about step 0 of the episode's
controlled part. W encodes the current onboard 112 px frame, rolls each candidate's commands plus
a declared settle (until the plate is at rest) forward, reads the predicted plate with
R-plate-pool, and executes the candidate whose predicted rest position is nearest the zone's
centre. Everything after the push is the existing v2 episode, so a counted success means the push
put the plate where the solved remainder works. The claim would name the phase: "LeWM-driven push
selection, followed by P-3", never "a LeWM policy".

**Why the blind and hand-written arms should lose, and the expected ties.**
- **N** predicts one rest position for every candidate, so the ranking carries no information; it
  should land near L-rand or near the best fixed push. Declared now (R8.8): the training pushes
  are drawn uniformly from the candidate grid, not from a privileged expert centred on good
  pushes, and the controller ranks.
- **L-shuf** knows what pushes do but not where this plate starts; it should land near the best
  fixed push. That only loses if the start jitter is wide enough that no single push serves most
  starts, which C2-F4 checks (H-fixed below).
- **H-fixed** (the best single candidate on development resets, image-free) is the clock-prior
  analogue. Expected to lose by construction of the start jitter, checked in C2-F4.
- **H-kin** (a hand-written quasi-static push model: the plate moves along the push direction by
  the push length beyond first contact, from the true start plate) and **H-kin-cal** (the same
  with a slide gain and a lateral-slip term fitted on development pushes, a system-identification
  arm) are the hand-written extrapolators. They should lose only where contact physics departs
  from them: off-centre contact on a disc, post-release sliding, finger compliance, rim contact.
- **Expected ties, stated now.** If the useful pushes are slow and centred, H-kin is close to
  exact and ties the ceiling, and there is no room for a world model. H-kin-cal may also tie if
  sliding is regular enough to fit with two parameters. C2-F4 measures exactly this, and a tie
  stops the line before any world model is trained.
- **Feedback is excluded by the single push**, and that is a declared limitation, not hidden: a
  two-push controller that pushes, re-reads and pushes again (H-kin-2) is reported in C2-F4 for
  information. If it reaches the ceiling, the claim is limited to single-shot pushes.

**Feasibility checks first (CPU smokes, no world model, fresh smoke seeds, development only).**
Each numeric bar is a development screen, not a gate; the gated versions belong to the protocol.
- **C2-F0, the free plate leaves v2 intact.** Plate free, reset in P-3's zone, no push: P-3 ≥
  30/32 and e9 ≥ 30/32 counted successes; median plate displacement during the episode ≤ 0.5 cm
  (the place must not shove the plate).
- **C2-F1, the push primitive works.** On 32 attempts spread over the candidate grid: ≥ 30/32 with
  no plate tilt above 10°, no plate off the table, no hand–apple or plate–apple contact; the
  median displacement of the longest candidates covers the 87.5th percentile of the
  start-to-zone distance; the 87.5th percentile of the steps from the decision to plate rest
  (h_push) is recorded.
- **C2-F2, the prefix does not disturb P-3.** The push prefix runs, then the plate is put back at
  a zone reset by a state write before the look: P-3 ≥ 30/32.
- **C2-F3, the zone is measured (τ_push).** P-3's counted successes with the plate placed at
  offsets of 0, 2, 4, 6 and 8 cm from the zone centre (4 directions, 16 resets per level, pooled):
  r_zone is the largest level with ≥ 14/16. It must be at least 2 cm (else no push can hit it) and
  at most half the median start-to-zone distance (else a fixed push suffices).
- **C2-F4, the headroom pre-check (K-push, development).** On 32 development resets: H-sim
  ≥ 28/32; H-sim − max(H-kin, H-kin-cal, H-fixed) ≥ +8/32 (TASK-074 K1's headroom bar, carried as
  in TASK-076's K-P2–K-P4) to proceed to a protocol; below +4/32 the line stops before any
  protocol (recorded, no clause, since nothing is gated); in between, a ruling. H-rand, H-kin-2
  and H-kin on the R-plate reading are reported. The scene's physical constants and the grid are
  frozen before C2-F4, and C2-F4 runs once.
- **C2-F5, the view.** The plate is wholly in the onboard 112 px frame at the decision on ≥ 31/32
  resets; a cross-fitted R-plate-pool on smoke roots reads the start plate with median error ≤
  r_zone / 2 (an estimate; c_plate itself comes from the protocol's Stage O).
- **C2-F6, cost.** An H-sim attempt (every candidate simulated to plate rest in cloned state) takes
  ≤ 60 s, as TASK-076's cap-review rule.

**Ceilings that calibrate its bars.**
- The count ceiling: H-sim on the gated condition (≥ 30/32, K-P1's form).
- The zone tolerance τ_push = r_zone (C2-F3, re-measured in the protocol's K0), the analogue of τ.
- c_plate_push: the upper 95 % bound of cross-fitted R-plate-pool's median plate error on this
  condition's frames (O1's form, R8.2). A predicted-latent plate bar B must satisfy
  c_plate_push ≤ B ≤ τ_push; if c_plate_push > τ_push, the row is NO-BAR (escalate).
- The tie bars: H-kin-cal's and H-fixed's measured counts; the best non-world-model arm is
  reported beside W.
- The horizon: h_push from C2-F1. W's dynamics gates (no collapse, beats copy-last, beats an
  equally trained N, action-sensitive) are run offline at h_push before any closed loop. At the
  decision the plate is at rest, so no hidden velocity term like cell A's m2 exists, and
  TASK-066's history one is enough for the start state.

**Abandonment clauses and scope.**
- K-push's own clause fires on PUSH-NONE (the ceiling holds and a headroom is detectably below
  +8/32, with R8.14's noise guard; PUSH-NEAR escalates without a clause). Scope: "a LeWM
  push-outcome ranker for one pre-pick push of a free plate on `apple-to-plate-v2-push`, with the
  declared candidate grid, constants and start region, on onboard 112 px frozen DINOv2 pooled
  tokens: no further grid, constant or start-region variant is preregistered without new evidence
  of a different kind".
- The later LeWM protocol's clause fires on L-NO-GAIN (W fails to beat N, L-shuf and L-rand) and,
  with no claim, when W is clearly below H-kin-cal. Same scope, for the predictor.
- PUSH-INFEASIBLE (C2-F0 to C2-F3 fail) and NO-BAR escalate without a clause.
- Never closed: the LeWM backend, DINOv2 as an encoder, v2, Arena and the product goal.

**GPU.**
- Feasibility and K-push: none (CPU; an attempt is about 5.5 worker-seconds as in TASK-076 §10,
  plus the push prefix; H-sim's look-ahead adds at most a few attempts' worth per decision).
- The LeWM stage: a push corpus (about 300 resets with uniformly drawn candidates; only the push
  prefix and its frames are needed, minutes of CPU), DINOv2 featurisation (under 20 min), then W
  and N on three seeds: about 6.5 h of GPU at TASK-074's measured per-model times, about 10–11 h
  worst case at a `check_budget`-passing cap (PLAN.md's TASK-077 sizing), in per-job slots under
  the shared lock. L-shuf needs no
  extra model. The closed loop runs on the CPU.
- Development effort before any GPU: the free-plate scene variant, the push primitive and the
  smokes, about 3–5 days.

**Risks.**
- **H-kin or H-kin-cal ties the ceiling** (slow, centred pushes are nearly kinematic). The most
  likely way for C2 to end, and it ends cheaply, at C2-F4, before any GPU.
- **A new scene variant.** A free plate is a task change in the sense of TASK-069/070 (a separate
  benchmark), and the physical constants must be fixed before any number. Tuning them on
  headroom would manufacture the result.
- **The single push is a declared restriction.** If H-kin-2 reaches the ceiling, the claim is
  limited to single-shot pushes.
- **The horizon** (h_push, likely 40–100 steps) is beyond TASK-066's gated 16; the offline horizon
  gate may fail.
- **P-3 outside its distribution.** A plate at the zone's edge, or a frame after an arm excursion,
  may drop P-3 below its 40/40; C2-F2 and C2-F3 measure it. If P-3's own place is the weak link,
  e9's place aimed at R-plate (TASK-076's H-twin) is the declared fallback, which then depends on
  TASK-076's twin result.
- **Push quality on Dex3.** Finger compliance, plate tilt at the rim (9 mm high), and apple contact on
  the way back.
- **Corpus coverage.** The readout and the predictor see a plate in new places (start region,
  sliding); c_plate_push must be measured on those frames.

**Earlier clauses.** Respected:
- TASK-075 covers the place phase under TASK-074's condition; C2's LeWM decision is a pre-pick
  push under a different, declared condition, so it is outside that scope by phase and by
  condition. It does not read the apple-minus-plate offset at all.
- R2 holds: the target (the plate's rest position) must be predicted, from the robot's own
  candidate action, before it exists.
- TASK-065 (pooled CLS) and TASK-062 (in-corpus encoder training) are not touched: frozen DINOv2,
  pooled patch tokens.
- TASK-057 is not touched: P-3 is TASK-071/072's v2 policy, not the `apple-wide-v1` line.
- TASK-054: a discrete ranker for one declared decision in one phase, with a plate-position cost,
  not CEM over that cost as a primary line. This note does not present sampling-based planning as
  the project's control approach; no control line is primary.

### 3.3 Candidate C3 — a contact-sensitive grasp approach (direction (b), grasp phase)

**The condition.** A declared grasp-phase disturbance in which the descending hand's contact moves
the apple: for example, an apple resting against a low stop, or a reduced rolling friction for the
apple (the latter changes v2's defining friction, so it would be a new variant, as v2 was from
v1). The pick is e9's scripted pick phases (orient, descend, close, lift; built from the apple
only) with an aim offset chosen at the start of the descend, instead of P-3.

**The question.** When the finger contact during the descend moves the apple, does LeWM's ranking
of candidate approach offsets by the predicted apple position at closure give more latched grasps,
and so counted successes, than N, L-shuf, L-rand and aiming at the apple's current reading?

**Where LeWM sits.** At the start of the descend (about step 130): W rolls each candidate's
descend-and-close commands (about 125 steps) forward and reads the predicted apple position
relative to the palm. e9's place (or P-3's) finishes the episode.

**Why the blind and hand-written arms should lose, and the expected ties.** N cannot rank; a
"the apple stays put" aim loses whenever contact moves it. **Expected tie, stated now:** the pick
is tolerant today (e9 32/32 at rest in TASK-070; P-3 40/40 on cohort C), so aiming at the current
apple is expected to be near the ceiling unless the disturbance is made strong; a disturbance strong
enough to open headroom is the kind of tuned condition this note refuses for C2.

**Feasibility checks first.** C3-F1: the disturbance moves the apple ≥ 2 cm in the median under
an aim at the current apple, with H-sim ≥ 30/32; C3-F2: H-sim − H-now-pick ≥ +8/32; C3-F3: an
on-table apple readout's ceiling (from TASK-072's post-look readout) below the grasp tolerance,
measured as τ is.

**Ceilings.** H-sim on the grasp; a grasp tolerance curve (latched grasps against a planted apple
aim error); the pooled-latent apple readout's ceiling before closure.

**Abandonment clause and scope.** Fires on L-NO-GAIN; closes "LeWM grasp-approach ranking under the
declared contact disturbance on v2 with pooled frozen DINOv2 tokens".

**GPU.** As C2: about 6.5 h, 10–11 h worst case, for W and N on three seeds, after a corpus of e9
picks with spread aims.

**Risks.**
- The apple is the weak quantity in this project's readouts: the in-hand apple term was
  2.2–3.1 cm in TASK-075, and TASK-074's offset failure came from it. TASK-063/064 found the
  post-look apple readable offline (readability gates against a floor, not a centimetre
  ceiling), and no pooled-latent apple ceiling under contact exists.
- Every learned grasp line on v1 failed (for example TASK-043's learned grasp, 0/4), and grasp
  contact is the hardest physics to predict from a 112 px frame.
- The headroom probably requires tuning the disturbance until the pick breaks.

**Earlier clauses.** Respected formally: the pick phase is outside TASK-075's place-phase scope;
frozen DINOv2 pooled tokens avoid TASK-062/065. It would replace P-3's pick with e9's scripted pick,
which moves the episode further from the learned policy that solved v2.

## 4. Comparison

| | C1 early commit, reactive plate | **C2 single push of a free plate** | C3 contact-sensitive grasp |
|---|---|---|---|
| decision | aim at 405 | push at the start | approach at about 130 |
| consequence from | a declared kinematic rule | contact and friction (physics) | finger contact (physics) |
| action-blind N | loses only by the ranking and corpus spread (R8.8) | loses by the ranking and uniform pushes | loses by the ranking |
| best hand-written arm | **H-rule, expected at the ceiling** | H-kin / H-kin-cal, **may tie** (measured in C2-F4) | aim at the current apple, expected near the ceiling |
| feedback excluded by | an imposed single commit | one push, natural for a non-prehensile slide | the commitment of a descend |
| remainder of the episode | e9's place | the solved v2 episode (P-3) | e9's or P-3's place |
| new development | small (Stage-0 code exists) | free-plate variant, push primitive (3–5 days) | disturbance, pick-aim interface |
| cheapest decisive check | C1-F4 (minutes) | C2-F4 (under 1 h of CPU) | C3-F2 (under 1 h of CPU) |
| GPU after admission | about 10–11 h worst case | about 6.5 h, 10–11 h worst case | about 6.5 h, 10–11 h worst case |
| what a pass could claim | LeWM *can* drive it; not that it is needed | LeWM-driven push selection beats its blind twins; on the second row, also a calibrated hand-written push model | LeWM-driven grasp approach under that disturbance |

## 5. Recommendation: C2 (decided by Claude under owner delegation, R9.2)

**Main reasons.**
1. **It is the only candidate whose best hand-written arm is not expected to tie by
   construction or by the task's current tolerance.** C1's consequence is a declared formula, so H-rule has the answer (§2). C2's is
   contact physics; a hand-written model has to approximate it, and C2-F4 measures how well,
   with a calibrated system-identification arm (H-kin-cal) as the strongest competitor.
2. **The decision comes first and decides the episode.** The push is the first controlled act,
   and everything after it is the v2 episode P-3 already solves 40/40. A counted success is then
   attributable to the push, and the downstream is a learned, non-privileged policy.
3. **No hidden-state term at the decision.** The plate is at rest when W encodes the frame, so
   TASK-066's history one suffices for the start (unlike cell A's m2).
4. **It stays outside every closed scope.** It is not the place phase, not TASK-074's condition,
   not the offset readout, not CEM as a primary line.
5. **Its failure modes are cheap and early.** If H-kin ties, C2-F4 says so on the CPU, before any
   corpus or GPU, which is the TASK-073 lesson applied before the protocol rather than inside it.

**What it does not buy.** A pass would show that LeWM can choose a single non-prehensile push
better than its blind twins and a calibrated hand-written model, on one simulated scene variant. It
would not show a LeWM policy, anything about v1 or its 0/150, Arena or the real G1. And it is a
new scene variant: its numbers do not transfer to v2.

**C1 is kept as the fallback** only if C2 ends at PUSH-INFEASIBLE for engineering reasons (the
free plate or the push primitive cannot be made to work), because it is cheap and its code
exists, with its expected H-rule tie written into its claim row in advance. **C3 is not taken**
(R9.4).

## 6. Next steps (none started by this note)

1. Review of this note (docs only).
2. A development feasibility record for C2, in the style of TASK-076's Stage-0 record: the
   free-plate scene variant behind an opt-in flag (v1 and v2 code untouched, as
   `apple_to_plate_v2.apply_v2_scene` does for v2), the push primitive and the arms H-sim, H-kin,
   H-kin-cal, H-fixed, H-kin-2, then C2-F0 to C2-F6 on a newly declared smoke seed block. The
   block and salts are declared in that record before any seed is simulated and checked against
   every range on main.
3. Only if C2-F4 passes: a preregistration with K0 (τ_push, the ceiling), Stage O (c_plate_push),
   K-push on fresh seeds with R8.14's noise guard, then the LeWM corpus, training under a
   `check_budget`-passing block, the offline horizon and dynamics gates, a 16-reset development
   closed loop with a stop rule, and the gated cohort.
4. TASK-076's primary question (Stages K0, O, D, S/U) is independent of this note and continues on
   its reviewer's GO.
