# Apple→Plate plate-readout twin v2: the place aims at the plate read from the image, plus a prediction-headroom check (TASK-076)

**Status: FROZEN** (K0-PASS; frozen block sha256 `011dff8ac65dde7341850ac4c4d9c569b399f1e3a8d2a1e978d3a3b9bbb7570b`,
pinned in `tests/test_plate_twin_v2.py` and the manifest). The freeze takes effect when this
revision is merged on an independent reviewer's reported APPROVE (§13.2 step 4). Stage 0 is done
(§13.2 steps 1–2, R8.15–R8.17): the stage code, the frozen block as module constants
(`src/embodied_jepa/plate_twin_v2.py`), the manifest (`benchmarks/manifests/apple-plate-twin-v2.json`),
the §5 tests and the smokes on smoke seeds 56900–56999. Their record is
[apple_plate_twin_v2_stage0.md](apple_plate_twin_v2_stage0.md). The smokes removed cell A
(§3.2). K-pred therefore cannot reach PRED-ADMIT(A): its row will be PRED-INFEASIBLE if Stage O
records O-PASS, and PRED-NOT-RUN if it does not (`decide_kpred` checks Stage O's row first; §6.3).
Both escalate with no clause. K0 ran once on cohort K (56000–56031), on a reviewer's reported GO
for K0 at `2d0bdb7`, and ended **K0-PASS** with τ_re = 1.0 cm (§5.1, R8.19). No other cohort seed
has been simulated. Nothing after K0 runs before this frozen protocol is merged on an independent
reviewer's reported APPROVE, and each later stage then runs only on its own reported GO (§5).
(A damaged TASK-072 evidence file found in Stage 0 has been restored, and G-repro passes.)

Every precision bar below (O1, O4, K-P5) is τ_re, measured in K0, and every ratio or test bar is
a definitional constant (O3's 1.0, G2's p < 0.01). The count bars and stop thresholds are
declared decision margins, with these sources (§13.1 lists each):
- G1's 56/64 and K0's level-0 stop (28/32) are τ's own bar;
- K-P1's 30/32, the +8/32 headroom bars (K-P2–K-P4), K0's N_K(0) < 30/32 and H-stale(K) > 4/32
  stops and S-VOID-CONDITION's H-stale(S) > 8/64 are carried from TASK-074's K1 (the last one
  doubled), and G3's −2/32 from TASK-074's G5;
- K0's H-clock stop, H-clock(K) ≥ N_K(0) − 4/32, is G2's minimum separation (§6.2), not a
  TASK-074 bar;
- Stage D's 12/16 and 14/16 (TWIN-DEV-STOP) are stops declared here; TASK-074 has no such bars.

None of them is a precision bar on a readout, which is the kind TASK-074's lesson is about.
§13.1 gives, for each, its source and either its feasibility check against a ceiling measured in
this task or its role as a stop. Everything else is reported only.

- Task card: `.mc/tasks/todo/TASK-076-*.md`. Rulings: [DECISIONS.md](../DECISIONS.md), decision
  2026-10-02, R1–R4 and R7, and decision 2026-10-02 (b), R8 (this draft's scope; its review
  rulings R8.1–R8.7; the freeze-preparation rulings R8.8–R8.14; Stage 0, R8.15–R8.18; K0 and
  the freeze, R8.19).
- Plan context: [docs/PLAN.md](../PLAN.md).
- Learned Apple→Plate status, the canonical sentence (DECISIONS 2026-10-02, R7), verbatim:

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

**LeWM is not in this task's control loop.** No arm of TASK-076 runs a world model. The
hypothesis arm H-twin aims e9's scripted place primitive at a ridge readout of the plate position
from frozen DINOv2 tokens of the current onboard frame. P-3, a behaviour-cloned policy, does the
pick. Nothing in this task can produce, or be reported as, a LeWM-driven success.

**Two representations, kept apart.**
- **H-twin's readout, R-plate,** reads the *full* DINOv2 patch tokens (98 304-d). It is TASK-074's
  `lewm_planner_v2_offline.fit_r_plate`, a dual ridge on full tokens. This is **not** the LeWM
  latent.
- **The LeWM latent** of TASK-066's token predictor is the *pooled 4 × 4 grid*
  (`token_dynamics.TOKEN_GRID = 4`: each pooled token is the mean of a 4 × 4 block of patches;
  6 144-d), the latent TASK-074's R_off and TASK-075's reported plate target readout read.
  **R-plate-pool**, a ridge from that pooled latent to the plate position, is measured in Stage O
  only so that a later LeWM task gets a ceiling on the representation it would actually roll
  forward (ruling R8.2).

**What LeWM gets from TASK-076.** It gets three things the next LeWM task needs and cannot
honestly guess:
1. **The encoded-latent plate ceiling** c_plate, measured on the pooled 4 × 4 latent (R-plate-pool,
   Stage O). This is the readout error before any prediction, on the latent a LeWM planner would
   roll forward. A later LeWM task's predicted-latent plate bar B must satisfy c_plate ≤ B ≤ τ_re.
   This extends TASK-075 §6.3's downstream rule, stated there for the offset readout c_V\*, to the
   plate; the extension is ours, declared here. It is the TASK-074 lesson: bars come from measured
   ceilings on the representation that is gated.
2. **The perception baseline,** H-twin's closed-loop count. Any later LeWM place planner is
   reported beside it, as TASK-074's bar T was.
3. **Whether a condition exists in which the place target must be predicted, and predicted from
   the robot's own actions** (Stage K-pred). Under ruling R2 a LeWM plate-target task is
   admissible only when paired with a condition where the target must be predicted. Under §9 the
   LeWM arm must also beat an action-blind predictor. So only an action-dependent cell can admit
   TASK-077 (ruling R8.3).

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
place family's bar? And does it beat an image-free prior?

**Why this quantity.** TASK-075 found that no view reads the apple-minus-plate offset within
τ = 1.0 cm. Its reported-only plate target readout, though, read the plate position to a median of
0.493–0.682 cm over the four views (onboard 112: 0.493 [0.455, 0.511] cm). That readout was a
pooled-token ridge (the same pooled 4 × 4 features as R_off), cross-fitted, evaluated at the frames
t + 16 = 421, …, 501 on O2's 915 moving windows. The offset error tracks the apple term, 2.211–3.122
cm (`apple_obs_ceiling_v2_results.md` §2.3, §3.4). τ is defined as an error in the aimed plate
target (§2 there). Two development results point the same way. Neither is gated:
- **TASK-074's Probe C.** A kernel-ridge plate readout of the frames at steps 320–400 after the
  move. Trained in-distribution only, it read 5.53 cm in median on far frames. Trained with far
  frames of other seeds, it read 0.55–0.69 cm in median, with a p90 of 1.7 cm. In closed loop, e9's
  place aimed at it after P-3's pick scored 31/32 (9 cm) and 30/32 (12 cm), against 32/32 with the
  true plate. Caveats: the frames came from P-truth trajectories, the readout was fitted on other
  seeds of the same probe cohort, and the seeds were spent development seeds 54700–54999
  (`apple_lewm_planner_v2.md` §2).
- **TASK-074's train-stage R-plate** (a dual ridge on full tokens at the decision frames, 1 350
  rows): val median 0.417 cm, p90 0.816 cm, reported only (`apple_lewm_planner_v2_results.md`
  §1.5).

**The claim a TWIN-PASS would make.** "Under TASK-074's 9 cm condition, P-3's pick followed by
e9's scripted place aimed at a frozen-DINOv2 ridge readout of the plate in the current onboard
112 px frame reaches the calibrated place bar on fresh gated resets and beats an image-free
prior." **It would not say:**
- anything about LeWM or any world model;
- that the pick, or the whole episode, is learned end to end (the pick is P-3, the place is
  scripted);
- that pretrained features are needed (the random-init floor is reported, not gated);
- anything about v1, its 0/150, Arena or the real G1.

**Secondary question (Stage K-pred; no world model).** Is there a condition on v2 in which the
plate keeps moving, *driven by the robot's own action*, such that the following both hold?
- Aiming at the plate's *final* position succeeds.
- Aiming at where the plate is *now*, or at a constant-velocity extrapolation of it, fails by a
  clear margin.

That margin is the room an action-conditioned predictor could have.

## 2. Carried unchanged

- **The v2 scene, e9 and its schedule.** e9 is
  `RestingPlaceExpert(release_pitch_rad=0.45, release_dx=0.015)` with the schedule
  `(130, 80, 45, 150, 100, 50, 30, 50, 30, 60)`. The transfer starts at 405, the lower at 505 (the
  target is frozen from here), and the place ends at 725 (`lewm_planner_v2.py` `TRANSFER_START`,
  `LOWER_START`, `PLACE_END`).
- **The counted success** (T71-R1/R2: at rest after a latched grasp and place), and the attempt
  of 740 commands plus the settle.
- **P-3:** `checkpoints/task072-first-policy-v2-linux/run-1/P-3.pt`, sha256 `7988162d…60be8`, with
  its post-look readout refitted exactly as TASK-072 run-1 fitted it (G-repro).
- **TASK-074's condition** (Stages K0, D and S/U only): `lewm_planner_v2.shift_vector` (the −y arc
  225–315°, salt 7413), `condition_reset` (re-draw salt 7425), the plate-shift hook at step 300 and
  `SHIFT_BLOCKED`. These salts are part of the condition's definition and are carried. The seeds
  are new, so the draws are new.
- **`place_planner.PlannerController` and `PlacePrimitive`,** with the decision steps 405, 421,
  437, 453, 469 and 485. Mode `twin` (H-twin) and mode `truth` (H-handover) exist and are used
  unchanged. New modes are added in new modules: H-clock, H-stale, H-floor and K-pred's arms. No
  file pinned by an earlier manifest is edited.
- **The sealed views store** `apple-far-shift-v2-views` (manifest sha256 `ea627a8f…4d77`) and its
  source corpus `apple-far-shift-v2` (manifest sha256 `fe7ab915…b134bd`): 270 train + val roots,
  of which 253 are read. The 30 test roots are never simulated, rendered or decoded. Both are
  identified by name and manifest sha256, which every stage checks before reading. They live in the
  main checkout's `data/`, and a run worktree made with `scripts/new_worktree.sh --run` reaches
  them through its `data/` symlink.

## 3. Arms

### 3.1 The primary closed loop (cohorts D, S, U)

| arm | what the place is aimed at, at each decision 405–485 | role | gating |
|---|---|---|---|
| **H-twin** | R-plate's reading of the current onboard 112 px frame (pretrained DINOv2 full tokens, dual ridge) | hypothesis | yes |
| H-handover | the true moved plate (privileged) | the family ceiling | feasibility; TWIN-NEAR |
| **H-clock** | the per-decision-step median of the true plate position over K0's resets under the same condition (fitted once in K0; reads no image at run time) | the image-free control | yes (G2) |
| H-stale | P-3's post-look plate estimate (read before the move; no later image) | the condition's check | S-VOID-CONDITION |
| H-floor | R-plate refitted on the seed-0 random-init DINOv2 (`pretrained_encoder.random_init`) | the encoder floor | reported |
| P-stale | P-3 unchanged | the incumbent | U no-harm (G3) |

- **R-plate for the closed loop** is fitted once, after the Stage O boundary, on the full tokens
  of every read root's decision frames that exist (225 train + 28 val roots). λ is chosen by
  5-fold inner CV grouped by root. It is TASK-074's `lewm_planner_v2_offline.fit_r_plate` recipe
  on more rows. Its sha256 is recorded, and every worker checks it.
- **Encoding in the loop** is on the CPU in the worker, batch size 1, as in TASK-073/074.
- **H-clock** is the strongest image-free target we could name for this condition.
  - It knows the condition's distribution, from privileged labels on development resets, but reads
    nothing at run time.
  - It replaces the card's "clock-prior arm".
  - It is fitted on K0, so it is not fitted on any gated reset. Its K count is in-sample (§5).

### 3.2 K-pred, the prediction-headroom check (cohorts M-a, M-b, A; no world model)

TASK-074's step-300 shift does **not** apply in any K-pred cell. The plate stays at its reset
position until 405; only the cell's own hook moves it. A plate move is refused, as in TASK-073's
hook, when the plate would touch anything but the table. A refused move ends the attempt as a
counted failure.

Every hook uses the same `body_pos` write plus `mj_forward` as the step-300 hook, outside every
controller's `act()`. The plate is static after s1 = 525. s1 lies inside the lower phase
(505–555), before the open phase (585–635).

**To be confirmed in the Stage-0 smokes:**
- no apple–plate contact before s1 on any smoke attempt. If a smoke finds contact, s1 moves
  earlier, and the cells are rescaled to keep their distance to go;
- for cell A, the size of the plate's remaining motion at 485, measured under H-final (below).

Any such change is made before the freeze and disclosed.

**Stage-0 result (R8.16; record: [apple_plate_twin_v2_stage0.md](apple_plate_twin_v2_stage0.md)).**
- **Contact.** None of the 144 base-setting smoke attempts (cell A: 5 arms × 16 resets; M-a and
  M-b: 4 arms × 8 resets) touched the plate with the apple before s1 = 525. The earliest
  apple–plate contact was at step 607, in the open phase. s1 = 525 stays, and M-a and M-b are
  unchanged.
- **Cell A is removed.** Under H-final the median remaining motion |plate(s1) − plate(485)| was
  0.060 cm (maximum 0.076 cm; H-now 0.148 cm), against the 2 cm the rule needs, with no refused
  move. The declared remedies, in order (L = 1; then s1 = 535, 545, …, 605, strictly before the
  contact limit of 607), gave medians of 0.022–0.063 cm, none refused. No allowed setting
  reaches 2 cm, so cell A is removed before the freeze, and K-pred's row is PRED-INFEASIBLE if
  Stage O records O-PASS and PRED-NOT-RUN otherwise (both escalate, no clause; §6.3). The reason is the one §3.2's remedy limits foresaw: e9's transfer
  brings the palm to the aim before 483, and the lower moves the palm by about 0.1 cm in xy. The
  sections on cell A below are kept as the design that was tested.
- **m2** (H-final, L = 2): median 0.0012 cm, maximum 0.0019 cm, on 16 smoke attempts.
- **A full-look-ahead H-final(A) attempt** took 15.0 s in median and 16.4 s at most (25.0 s at
  most over the remedy settings); every one of the 96 decisions converged, in at most 7
  iterations. That is below §10's 60 s, so the 300 s per-attempt cap is not reviewed.
- These smokes ran P-3 on the reset truth as its post-look estimate (a smoke stand-in, because
  G-repro cannot run; §13.2), with a placeholder τ_re of 1.0 cm. Neither changes the plate's
  remaining motion after 485, which the place primitive's palm path sets.

**Constant-velocity cells (reported only; ruling R8.3).** The plate moves at constant velocity from
s0 = 405 to s1 = 525, along a direction drawn by TASK-074's −y arc rule (salt 7606), by a total
distance D.

| cell | D | plate distance still to go at the last decision (485) |
|---|---|---|
| M-a | 12 cm | 12 × 40/120 = 4.0 cm |
| M-b | 9 cm | 9 × 40/120 = 3.0 cm |

**Expected in advance, before any number.** The plate's motion in M does not depend on the
robot's action. So, for the plate target:
- an action-blind predictor is expected to tie an action-conditioned one;
- H-cv, which knows the constant-velocity form and s1, is expected to reach about H-final.

So M cannot admit a task whose claim row needs to beat an action-blind twin (§9). M is reported
to measure how much a non-predicting aim loses, and nothing more.

**The action-dependent cell A (the only admitting cell; rulings R8.3 and R8.7).** This is a
declared, simple, simulator-only rule:
- **The rule.** From 405 to s1 = 525, the plate's xy velocity at step t is κ times the right
  palm's xy velocity at step t − L.
  - The palm velocity comes from robot kinematics (`PalmFK` on the executed joint state), so it
    is the robot's own executed motion.
  - The constants are **κ = −0.5 and L = 2 steps**. Palm velocity before 405 counts as zero.
- **Direction, stated plainly.** κ < 0, so the plate moves against the palm's velocity, at half
  its speed. As the hand approaches the plate, the plate moves back towards the approaching hand.
  - This risks moves that the hook refuses, because the plate would touch the hand, the apple or
    the arm. A refused move is a counted failure.
  - Refusals show up in H-final's count, and so in K-P1 and PRED-INFEASIBLE (§6.3).
- **Why the last decision matters** (the R8.7 redesign).
  - The plate's motion after the last decision at 485 is
    κ · [palm(s1 − L) − palm(485 − L)] = κ · [palm(523) − palm(483)].
  - All but the first 2 steps of that palm motion happen after 485, so they are caused by the aim
    chosen at 485. The palm keeps moving towards it through the rest of the transfer, to 505.
  - The aim that lands on the plate is therefore the fixed point of "where the plate will be,
    given the palm motion my own aim causes". If the palm ends near the aim, the plate stops at
    about p + κ(g − h) for an aim g, where p is the plate and h the palm at the decision. The fixed
    point is then about g\* = (p − κh)/(1 − κ) = (p + 0.5h)/1.5. Both p and h are in the current
    frame (and h in proprioception).
- **What each predictor is given, and when an action-blind one could tie** (R8.8).
  - An action-conditioned predictor, given each candidate aim's commands, predicts each
    candidate's own outcome. It is given the information to rank candidates.
  - An action-blind predictor is not. It predicts one plate(s1) for all candidates: in effect the
    mean over the aims of the behaviour policy it was trained on.
  - **It can still tie.** If the training aims are centred on g\* (for example e9 aimed at a
    privileged fixed point, with zero-mean mis-aims), that mean is about g\*. A controller that
    simply aims at the single predicted plate would then tie with either predictor.
  - The action-blind twin is expected to lose only if (a) the controller ranks candidate aims by
    their individual predicted outcomes, or (b) the training aims are not centred on g\*.
  - **TASK-077 must therefore declare three things in advance:**
    1. its controller form, which should rank candidate aims by their individual predicted
       outcomes (one roll-out per candidate), not aim at one predicted plate;
    2. the spread of its training aims relative to g\*: the corpus's aim distribution, its centre
       and its spread;
    3. the expected result of the action-blind twin under that controller and corpus.
  - H-rule's expected near-ceiling result (below) is the same point seen from the
    non-world-model side.
  - Under the earlier L = 40, that motion was κ · [palm(485) − palm(445)], already fixed at 485,
    so action conditioning added nothing. That design is withdrawn.
- **Each predictor's required history, declared now** (R8.7).
  - **Action-conditioned predictor.** It needs one current frame plus the candidate aim's
    commands. The palm motion that the frame cannot show is palm(485) − palm(483): 2 executed
    steps.
    - **The 2-step term, m2.** Per attempt, m2 = |κ| · ‖palm_xy(485) − palm_xy(483)‖, from the
      executed joint states (`PalmFK`), at the last decision 485. It is a bias that adds to the
      readout error, not a substitute for it.
    - **It is measured, not bounded** (R8.9). The clip bound, |κ| × 2 steps × about 0.6 cm per
      step and axis (the primitive's per-axis clip, `apple_lewm_planner_v2.md` §9b), is about
      0.85 cm in xy, but the clip is on commands, and m2 uses executed palm motion. So:
      - the Stage-0 smoke measures m2 under H-final on its cell-A attempts and records its median
        and maximum (the pre-freeze estimate; nothing is decided from it);
      - Stage K-pred logs m2 on all 32 of cell A's H-final attempts and reports m2\*, the upper
        95 % bound of its median (bootstrap, salt 7604), and its 87.5th percentile.
    - **TASK-066's history-one predictor** (`token_dynamics.py`, the comment above
      `MODEL_CONFIG`) qualifies for TASK-077 only if **c_plate + m2\* ≤ τ_re**, or if TASK-077's
      bar B has a calibrated allowance that covers m2\* (c_plate + m2\* ≤ B ≤ τ_re).
    - Otherwise TASK-077 must declare a history of L + 1 = 3 frames, or give its predictor the
      last 2 executed commands. Recorded now, before any number.
  - **The action-blind twin** gets the same observation history (one current frame) and no
    actions.
- **Look-ahead to s1.** The plate's stop at s1 = 525 lies 40–120 steps after the decisions, so the
  look-ahead goes beyond one 16-step chunk. TASK-077 must gate its own horizon (§9).

The rule is a scene variant, not a claim about any real plate. **Its feasibility in this scene is
unverified.**
- **The smoke measures under H-final.** The Stage-0 smoke reports, under H-final, the median
  |plate(s1) − plate(485)| on the smoke resets: the plate's remaining motion after the last
  decision. It also reports the same quantity under H-now, for scale.
- **The remedy only makes the target depend more on the last aim** (R8.7). If the median under
  H-final is below 2 cm:
  1. first, L is lowered to 1;
  2. then s1 moves later, in steps of 10, up to the contact limit found by the same smoke.

  L is never raised.
- **The remedy's limits, stated in advance** (R8.10).
  - Lowering L from 2 to 1 changes almost nothing: the remaining motion becomes
    κ · [palm(524) − palm(484)] instead of κ · [palm(523) − palm(483)], of nearly the same size.
  - Moving s1 later adds remaining motion only if the palm still moves in xy during the lower
    (505–555). If the lower is mostly vertical, it adds little.
  - κ is not a remedy. It stays at −0.5. **Any |κ| ≥ 1 is forbidden** in this task and in any
    amendment of it: H-final(A)'s look-ahead contracts by about |κ| per iteration, so at |κ| ≥ 1 it
    would stop converging, and H-rule's fixed point with it.
  - So a shortfall is the expected way for cell A to fail. It is handled by removing cell A, which
    makes the row PRED-INFEASIBLE (escalate, no clause), not by stretching the rule.
- **If no allowed setting gives a median of at least 2 cm,** or if more than a quarter of the
  smoke attempts are refused, cell A is declared infeasible and removed before the freeze. In
  that case PRED-ADMIT is unreachable: constant velocity alone cannot admit (§6.3).

**K-pred arms.** Every cell runs all of them, except H-rule, which runs on A only.

| arm | aimed at, at each decision 405–485 | role |
|---|---|---|
| H-final | the plate's position at s1 (privileged). In M it is the scheduled final position. In A it is a fixed point found by privileged look-ahead (below) | the ceiling |
| H-now | the plate's true position at the decision step (privileged, non-predicting) | the cost of not predicting, with perfect perception |
| H-twin | R-plate's reading of the current frame | the cost of not predicting, with this perception |
| H-cv | a least-squares constant-velocity line through H-twin's readings so far, extrapolated to s1 | the hand-written, action-blind extrapolator |
| H-rule (A only) | the fixed point of the declared rule, computed from H-twin's reading, the robot's own last L executed palm steps, and the kinematic stand-in's palm path under each candidate aim | the rule-knowing non-world-model arm |

**H-final(A)'s look-ahead iterates to a declared tolerance** (R8.7). At each decision:
- start from g₀ = the true current plate;
- for each k, simulate the rest of the place in cloned state to s1 under aim g_k, and set
  g_{k+1} = the simulated plate at s1;
- stop when |g_{k+1} − g_k| ≤ τ_re/4 (0.25 cm if τ_re = 1.0 cm), with at most 10 iterations.

With κ = −0.5 and a palm that follows the aim roughly one to one, the gap contracts by about half
per iteration: from 10 cm, 6 iterations reach 0.16 cm. That is an estimate, to be checked in the
smoke.
- **If an attempt does not converge** at any decision within 10 iterations, it is logged, aimed at
  the last iterate, and counted as an H-final failure if it does not succeed. So
  non-convergence lowers H-final, and can make K-P1 fail and the row PRED-INFEASIBLE.
- **The tolerance and the iteration cap are frozen with the protocol.**

Further notes on the K-pred arms:
- **Privileged knowledge of structure.** H-cv knows the M family's declared form (constant
  velocity, the stop step s1). H-rule knows the A rule (κ, L, s1). That is privileged knowledge
  of the condition's structure, not of the state, in the same way that H-clock knows the
  condition's distribution. A world model would have to learn it from data.
- **Expected in advance:** on A, H-rule is expected to come close to H-final. A later LeWM arm
  could then at best *tie* the best non-world-model arm on A. §9 requires only that it be
  reported beside that arm, and PLAN.md states the claim TASK-077 could make.
- **H-cv's first decision.** At 405 it has one reading and aims at it (H-twin's choice). From 421
  on, it fits a line.
- **Coverage of R-plate.** R-plate is fitted on static-plate corpus frames. In A, the plate can be
  displaced in directions the corpus did not cover (the corpus moves are on the −y arc). H-twin's
  error is reported per cell.

## 4. Seeds and cohorts

A repository search on 2026-10-02 found no seed use in 56000–56999. It covered src, scripts,
docs, benchmarks, tests, configs, `.mc`, and the open and archived remote branches. The only hits
for 56xxx were a float in a manifest. It also found no use of 7601–7620 as a seed constant. The
independent review of #129 confirmed both on all 50 local and remote refs.

Every range below is disjoint from:
- every range in `obs_ceiling_v2.FORBIDDEN_RANGES`, from 20000–20049 up to the TASK-074 block
  54000–54999, including cohort C 45300–45339 and the TASK-071/072 block 51000–52199;
- TASK-075's block 55000–55999;
- the Arena seeds of #123 (50200–50231, in TASK-070's block).

| seeds | cohort | stage | use |
|---|---|---|---|
| 56000–56031 | K | K0 (development, before the freeze) | τ re-measured; H-handover's ceiling; H-clock fitted; H-stale's and H-clock's counts |
| 56040–56071 | M-a | K-pred (development) | the 12 cm constant-velocity cell (reported) |
| 56080–56111 | M-b | K-pred (development) | the 9 cm constant-velocity cell (reported) |
| 56160–56191 | A | K-pred (development) | the action-dependent cell (the only admitting cell) |
| 56120–56135 | D | D (development, stop rule only) | H-twin and H-handover, 16 resets |
| 56200–56263 | S | gated | 64 shifted resets, every §3.1 arm once, paired |
| 56300–56331 | U | gated | 32 unshifted resets: H-twin, P-stale, H-handover |
| 56900–56999 | smoke | Stage 0 | mechanics only; nothing in them is read |

**Resets.** Every cohort, including U, M-a, M-b and A, draws its reset with
`lewm_planner_v2.condition_reset` (the right-eligible re-draw, salt 7425), as TASK-074's cohorts
did. So all cohorts share one reset distribution. The step-300 shift (salt 7413) applies to K, D
and S only; U and the K-pred cells have none (§3.2).

**RNG salts** (R8.11 completes the list).
- New:
  - 7601: the planted-error direction in K0's τ curve;
  - 7602: the outer folds (Stage O's cross-fitted R-plate, R-plate-floor and R-plate-pool);
  - 7603: the inner folds (λ selection in every ridge, including the closed-loop R-plate fit);
  - 7604: every bootstrap (§6 intervals, m2\*, and K0's G2 feasibility resampling, §5);
  - 7606: the moving-plate direction (M cells).
- Reserved and unused: 7605 (it was listed for learning-curve subsets, but no stage fits a
  learning curve). Any use needs a reviewed amendment.
- Carried as part of the condition: 7413 (shift direction) and 7425 (reset re-draw).
- Fixed seeds, not salts: the random-init floor uses `pretrained_encoder.random_init` seed 0
  (H-floor, R-plate-floor). Cell A draws nothing: its plate motion is a deterministic function of
  the executed palm motion. H-final(A)'s look-ahead is deterministic on cloned state.
- Cohort C is not used.

## 5. Stages and stop rules

1. **Stage 0 (the preregistration PR).** The protocol, the frozen block, the code, the tests and
   the smokes on 56900–56999. The tests include:
   - every seed range above against the forbidden ranges;
   - no iterative training anywhere in the stage code (§7);
   - the scale probe calls the stage's own function (`run_tools.scale_probe`, F15);
   - the runner imports no other runner script (`tests/test_no_runner_imports.py`);
   - `run_tools.assert_local_import`, `install_guards` and `gpu_guard` are used.

   The smokes also settle the K-pred mechanics questions of §3.2, and they measure the 2-step
   term m2 under H-final on their cell-A attempts (§3.2, R8.9). Their results, and any change
   they force (s1, L or cell A's removal), are written into the protocol before the freeze.
2. **Stage K0, calibration (development; after a reviewer's reported GO for K0, before the
   freeze).** It runs on cohort K, the simulator only, on the CPU.
   - **τ_re.** It uses **the procedure of** TASK-075's `TAU_REMEASURE` (its §6.3): the same arm
     (H-handover with a planted target error), condition, levels {0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5}
     cm, direction rule (salt 7601 here) and rule. τ_re is the largest level L such that every
     level ≤ L reaches ≥ 28/32.
     - TASK-075 pre-committed that re-measurement only for an OBS-ONBOARD or OBS-EXTRA outcome
       (`obs_ceiling_v2.py`), and TASK-075 ended OBS-NONE. This is therefore an adoption of the
       procedure, not that commitment.
     - Its use as the plate readout's tolerance is our extension, declared here (ruling R8.6).
     - It is 288 attempts. TASK-075's τ took 265 s for the same number on 6 workers.
   - **The ceiling** N_K(0) is the level-0 count.
   - **H-clock** is fitted on K's true plate positions at the decision steps, then run on K, with
     H-stale beside it (64 attempts). **H-clock(K) is in-sample** (fitted and run on the same
     resets), so it is optimistic for H-clock. That makes the stop below conservative.
   - **Stops, all CAL-ESCALATE** (escalate, no clause; nothing is frozen; the owner, or Claude
     under the owner's delegation, rules). Ruling R8.4 added the last two:
     - level 0 is below 28/32 (τ_re undefined);
     - N_K(0) < 30/32 (the gated bar of 56/64 is then not feasible);
     - H-stale(K) > 4/32 (the condition does not need a reading; TASK-074 K1's P-stale ≤ 4 bar);
     - H-clock(K) ≥ N_K(0) − 4/32 (an image-free prior is near the ceiling, so G2 is not
       feasible).
   - **G2's predicted feasibility, reported beside the stop** (R8.12; reported only, it stops
     nothing). K0 runs H-handover at level 0 and H-clock on the same 32 resets, so their paired
     outcomes give discordant counts b (H-handover succeeds, H-clock fails) and c (the reverse).
     The two numbers below are not bounds, because two effects pull in opposite directions
     (R8.17): H-handover stands in for H-twin at its ceiling, which is optimistic for G2; but
     H-clock(K) is in-sample (fitted and run on K's resets), which flatters H-clock and so is
     pessimistic for G2:
     - **the scaled point prediction:** the exact one-sided McNemar p-value on (2b, 2c), the K0
       counts doubled to S's 64 resets;
     - **the predicted pass probability:** the fraction of 10 000 resamples of 64 resets, drawn
       with replacement from K0's 32 pairs (salt 7604), whose exact one-sided McNemar p is
       below 0.01.
     Both are written into the frozen block with the K0 values. If the stop does not fire but the
     predicted pass probability is below 0.5, that is disclosed in the freeze PR as a known risk
     to G2, not as a stop.
   - The K0 values are written into the frozen block: τ_re, the τ curve, N_K(0), H-clock's fitted
     targets, H-clock's and H-stale's K counts, b and c, and G2's predicted feasibility. K0 runs
     no hypothesis arm, so no bar sees an H-twin number.
   - **K0-PASS** (no stop fires): the K0 values are frozen with the protocol (step 3).
3. **Freeze.** The PR is merged on an independent reviewer's reported APPROVE.
4. **Stage O, offline plate admission (after the freeze, on a GO).** It reads the sealed views
   store: no render and no new simulation.
   - **Featurisation** on CUDA through `scripts/gpu_run.sh` (§8), with a CPU anchor check: full
     tokens, and their pooled 4 × 4 grid.
   - **Cross-fitted R-plate** (full tokens; 5 outer folds by root, salt 7602), on each read root's
     onboard 112 px decision frames, gives per-frame plate errors. The same is done for
     R-plate-floor, on random-init features.
   - **Cross-fitted R-plate-pool** (the pooled 4 × 4 LeWM latent; TASK-075's `RidgeReadout`
     settings) is computed on the same frames and folds. It gates nothing in TASK-076. It defines
     c_plate for TASK-077 (§9).
   - **The offline clock prior** is the per-step median plate position over the outer-training
     roots.
   - **The plate-hidden check** uses the stored plate-hidden onboard frames at the eval steps
     421–501.
   - **`first_outcome_utc`** is written before the first fit.
   - Then the gates O1, O3 and O4 (§6.1), and the closed-loop R-plate fit (§3.1).
   - **Reported only:**
     - the 87.5th percentile (former O2, ruling R8.5);
     - the per-step errors;
     - the signed mean along x and y (τ is anisotropic);
     - c_plate on R-plate-pool;
     - the τ-curve prediction: TASK-075 §6.4's mapping with K0's curve, read against 28/32.
     The prediction is written before any closed-loop attempt.
5. **Stage D, development closed loop (on a GO; it requires the recorded O-PASS).** H-twin and
   H-handover run on cohort D. **Stop: TWIN-DEV-STOP** (escalate, no clause) if H-twin < 12/16 or
   H-handover < 14/16. Nothing is refitted on D.
6. **Stage S/U, gated (on a GO; it requires the recorded D-PASS).** Cohorts S and U, every arm
   once per reset, paired, from a clean worktree of the merged revision. It includes a determinism
   re-run of H-twin on the first four S seeds; any difference is a V (TASK-074 §8.3).
7. **Stage K-pred (on a GO; it requires the recorded O-PASS; independent of D and S/U).** Cohorts
   M-a, M-b and A (if A survived Stage 0), with the arms of §3.2, one runner invocation per cell
   (§7). It produces its own row (§6.3). If Stage O does not record O-PASS, K-pred does not run,
   and its row is PRED-NOT-RUN (§6.3).
8. **Results PR.** Every arm is reported, and the privileged ceilings, H-clock, H-cv and H-rule
   are labelled as not learned. An independent reviewer checks every restated number.

### 5.1 K0 results (K0-PASS)

K0 ran once, on a reviewer's reported GO for K0 (#134, issuecomment-5982693642), with
`scripts/run_plate_twin_v2.py k0 --output outputs/task076-k0-1 --evidence
/home/huhn/develop/emai/worktrees/task076-evidence`, from the main checkout at `2d0bdb7` (clean
tracked tree; the DRAFT frozen block's sha256 was `fe23e917…112f`). It ran on the CPU with 6
workers and without the GPU lock (§8), from 2026-10-04 17:52:07 to 17:57:27 UTC (319.75 s; first
cohort render 17:52:52 UTC; start load averages 0.21 / 0.66; peak process-tree PSS 8.20 GiB of the
12 GiB ceiling). G-repro passed all eight of TASK-072 run-1's reproduction checks inside the run,
and no render or re-render disagreed. No hypothesis arm and no world model ran.

- **Report:** `outputs/task076-k0-1/report.json` (Linux PC, main checkout; git-ignored), sha256
  `ef4b541067dc969ee5ebfc9ee78e2c719a4b84e4d3ed0d71a167feb6fcd0c876`. Every number below is copied
  from it, and the frozen block's `K0_MEASURED` carries the same values.

**The τ curve** (H-handover with a planted target error; counted successes of 32 on cohort K,
seeds 56000–56031; at rest is reported only):

| planted error (cm) | 0 | 0.5 | 1 | 1.5 | 2 | 2.5 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|---|
| counted successes / 32 | 32 | 31 | 28 | 23 | 17 | 18 | 16 | 8 | 5 |
| ≥ 28/32 | yes | yes | yes | no | no | no | no | no | no |
| at rest / 32 (reported) | 32 | 31 | 28 | 23 | 17 | 18 | 16 | 9 | 10 |

- **τ_re = 1.0 cm:** every level up to 1 cm reaches 28/32, and 1.5 cm does not (23/32). It equals
  TASK-075's τ (1.0 cm, on other seeds).
- **N_K(0) = 32/32** (the level-0 count). No level-0 attempt failed.
- **H-clock(K) = 24/32**; it failed on seeds 56000, 56007, 56012, 56017, 56018, 56023, 56028 and
  56029. Its fitted targets are the same at every decision step (405–485), because the plate is
  static after the shift at step 300: plate xy = (0.48204, −0.18196) m (full precision in
  `K0_MEASURED["clock_targets"]`). H-clock(K) is in-sample (§5 step 2).
- **H-stale(K) = 0/32.**
- **The stops** (none fired): level 0 = 32 ≥ 28; N_K(0) = 32 ≥ 30; H-stale(K) = 0 ≤ 4;
  H-clock(K) = 24 < N_K(0) − 4 = 28. The row is **K0-PASS**: K0's values enter the frozen block.
- **G2's predicted feasibility** (R8.12; reported only, it stops nothing): b = 8 (H-handover at
  level 0 succeeds, H-clock fails), c = 0. The exact one-sided McNemar p on (2b, 2c) = (16, 0) is
  1.53 × 10⁻⁵, and 9 983 of 10 000 resamples of 64 pairs (salt 7604) have p < 0.01, a predicted
  pass probability of 0.9983. It is not below 0.5, so no risk to G2 is disclosed on this
  ground. These numbers are not bounds: H-handover stands in for H-twin at its ceiling, which is
  optimistic for G2, and H-clock(K) is in-sample, which is pessimistic for G2 (§5 step 2). G2
  needs H-twin itself to keep most of H-handover's 8-reset margin over H-clock.

## 6. Gates and rows

Intervals are reset-clustered bootstrap percentile intervals: 10 000 resamples, 95 % (salt
7604). Paired closed-loop differences are reported with their discordant counts.

### 6.1 Stage O, offline admission (O1, O3 and O4 must hold)

| | condition | why |
|---|---|---|
| **O1** precision | the upper 95 % bound of cross-fitted R-plate's (full-token) median error ≤ τ_re | H-twin's readout is within the measured tolerance (TASK-075 A1's form) |
| **O3** clock prior | the upper bound of median(e_R-plate) / median(e_clock) < 1.0 | the image adds over a prior that reads none |
| **O4** plate hidden | the lower bound of R-plate's median error on the plate-hidden frames > τ_re | the readout reads the plate, not e9's arm, which in this corpus is aimed at the (possibly mis-aimed) plate. **This matters more in closed loop than offline:** from 405 on, the arm moves towards H-twin's own last reading, so a readout keyed on arm pose would confirm itself |

**O2 is reported only** (ruling R8.5). It is the 87.5th percentile of R-plate's error, with its
interval, plus the τ-curve-mapped predicted count, using K0's curve against 28/32. A tail bar of
"2 τ_re" would not come from a measured ceiling. K0's one-run τ curve (§5.1) is non-monotone
(17/32 at 2 cm, 18/32 at 2.5 cm), and every level from 1.5 cm up (23/32 at 1.5 cm) is below 28/32.
(Corrected after the freeze, in the results PR, ruling R10.2: this sentence had quoted TASK-075's
curve, 22/32 at 1.5 cm and 23/32 at 2 cm. Reported-only text; the frozen block is unchanged.)

Reported, not gating: R-plate / R-plate-floor (the random-init floor tied P-3 in M2, and this task
does not ask whether pretraining helps); per-step errors; the signed means; **c_plate on
R-plate-pool**.

**Rows, first match:**
1. V;
2. **TWIN-OFF-ARM** if O4 fails (escalate, no clause: the readout's source is unclear);
3. **TWIN-OFF-FAIL** if O1 or O3 fails (the clause fires, §7);
4. otherwise **O-PASS**.

### 6.2 Stage S/U, the gated rows (first match)

**The bars.**
- **G1 (the place bar):** H-twin(S) ≥ 56/64. This is τ's own bar, 28/32 = 87.5 %, which a target
  error within τ_re keeps by τ's definition, measured in K0. Its feasibility is K0's
  N_K(0) ≥ 30/32. The paired difference H-twin − H-handover is reported beside it, with its
  interval.
  - **Power** (exact binomial, P(X ≥ 56 of 64) at H-twin's true rate p): 0.59 at p = 0.875
    (28/32), 0.73 at p = 0.89, 0.86 at p = 0.906 (29/32), 0.98 at p = 0.9375 (30/32), and
    > 0.99 at p = 0.969 (31/32).
  - So an arm whose true rate sits exactly at the bar fails G1 about 4 times in 10. The TWIN-NEAR
    row below keeps such a miss from firing the clause (ruling R8.4).
- **G2 (the image-free control):** H-twin(S) > H-clock(S), one-sided exact McNemar p < 0.01 on
  the paired resets. There is no tuned margin, as with TASK-075's ratio bars.
  - **G2's minimum separation, stated** (R8.7). An exact one-sided McNemar test reaches p < 0.01
    only with at least 7 discordant pairs, all in H-twin's favour (0.5⁷ = 0.0078; 6 give 0.0156).
    With one reversed pair it needs 10 of 11 (p = 0.0059); 9 of 10 gives 0.0107 and fails.
  - So G2 needs H-twin − H-clock ≥ 7/64, about 3.5/32, at the very least. K0's stop at
    H-clock(K) ≥ N_K(0) − 4/32 is set at that separation. H-clock(K) is in-sample, and so
    optimistic, which makes the stop conservative.
- **G3 (no harm, U):** H-twin(U) ≥ P-stale(U) − 2/32 (TASK-074's G5). Its feasibility is checked
  on U itself: H-handover runs on U (R8.13), and the U-VOID-CEILING row below keeps a ceiling
  failure on U from firing the clause through TWIN-HARM.

| row | condition | consequence |
|---|---|---|
| **V** | the void rule (§7) | one repeat after a reviewed fix |
| **S-VOID-CONDITION** | H-stale(S) > 8/64, or H-handover(S) < 56/64 | escalate: the condition does not need a reading, or the ceiling fell below the bar on S |
| **U-VOID-CEILING** | H-handover(U) < P-stale(U) − 2/32 | escalate, no clause (R8.13): G3 is not feasible on U, because even the true plate falls below its bar, so a G3 miss would measure the place family, not the readout |
| **TWIN-HARM** | G3 fails | the clause fires (§7) |
| **TWIN-PASS** | G1 and G2 pass | **the only claim row** (§1) |
| **TWIN-PRIOR** | G1 passes, G2 fails | escalate: an image-free prior is about as good, so this condition does not test perception |
| **TWIN-NEAR** | G1 fails, and the 95 % interval of the paired H-twin − H-handover difference includes 0 | escalate, no clause, no claim: H-twin is not detectably below the ceiling, so the miss may be noise at n = 64. Any repeat needs fresh seeds and its own ruling |
| **TWIN-FAIL** | otherwise (G1 fails, and H-twin is detectably below H-handover) | the clause fires (§7) |

### 6.3 Stage K-pred, its own row

**Only cell A can admit** (ruling R8.3). The M cells are reported against the same quantities and
never admit.

**The bars on A:**
- **K-P1 (feasible ceiling):** H-final(A) ≥ 30/32 (TASK-074 K1's ceiling bar).
- **K-P2 (prediction matters with perfect perception):** H-final(A) − H-now(A) ≥ +8/32 (TASK-074
  K1's headroom bar).
- **K-P3 (prediction matters with this perception):** H-final(A) − H-twin(A) ≥ +8/32.
- **K-P4 (an action-blind extrapolator does not suffice):** H-final(A) − H-cv(A) ≥ +8/32.
- **K-P5 (a TASK-077 bar is feasible):** c_plate ≤ τ_re. **c_plate** is the upper 95 % bound of
  cross-fitted R-plate-pool's median plate error on Stage O's frames and folds (O1's form,
  TASK-075's A1 form c_V), with the bootstrap of §6.

Reported, never bars: H-rule(A), and H-final − H-rule with its interval; every M-cell count.

**The headroom's noise guard** (R8.14). Each headroom (K-P2, K-P3, K-P4) is a paired difference on
A's 32 resets, reported with its paired interval: the reset-clustered bootstrap percentile
interval of §6 (10 000 resamples, 95 %, salt 7604). A failed headroom bar is:
- **detectably below** +8/32 when that interval's upper bound is below 8/32;
- **near** +8/32 otherwise (the interval includes +8/32).

PRED-NONE needs at least one failed headroom bar that is detectably below. A failure that is only
near gives PRED-NEAR, as G1's TWIN-NEAR does in §6.2.

- **PRED-NONE's false-fire probability at a true headroom of exactly 8/32** is about 2–3 % per
  bar: 2.6 %, 3.0 % and 2.2 % in a simulation (4 000 trials of 32 paired resets, 4 000 bootstrap
  resamples each) with 0, 1 and 2 reversed pairs per 32 expected. Without the guard it was the
  bar's own miss rate, 42–45 % in the same simulation. If all three headrooms sit exactly at 8/32,
  the union bound is about 9 %.
- **The guard's power, stated in advance** (R8.17, from the independent review of #133, its own
  simulation). At a true headroom of 4/32 (half the bar), PRED-NONE fires only about 42 % of the
  time (about 36 % when that headroom comes as 5 wins with 1 reversed pair per 32); at 2/32 it
  fires about 87 % of the time. So at an intermediate headroom PRED-NEAR is the likely row, and
  K-pred then ends in an escalation rather than in its clause.

| row (first match) | condition | consequence (for the plan, not for TASK-076's claim) |
|---|---|---|
| **PRED-NOT-RUN** | Stage O did not record O-PASS, so K-pred did not run (§5) | no clause of its own; PLAN.md's Branch B applies, together with Stage O's own row (TWIN-OFF-ARM escalates, TWIN-OFF-FAIL fires its clause) |
| **V** | the void rule | one repeat after a reviewed fix |
| **PRED-INFEASIBLE** | cell A was removed at Stage 0 as infeasible, or K-P1 fails on A | **escalate, no clause** (ruling R8.4): the ceiling, not the room for prediction, failed. It is a design failure, as at TASK-073/074's failed ceilings |
| **PRED-NONE** | K-P1 passes, and at least one of K-P2, K-P3 and K-P4 fails with its paired upper bound below 8/32 (detectably below) | the K-pred clause fires (§7) |
| **PRED-NEAR** | K-P1 passes, and at least one of K-P2–K-P4 fails, but every failed one's paired interval includes +8/32 | **escalate, no clause, no claim** (R8.14): the headroom is not detectably below its bar, so the miss may be noise at n = 32. Any repeat needs fresh seeds and its own ruling |
| **PRED-NO-BAR** | K-P1–K-P4 pass, K-P5 fails | escalate, no clause: there is room for prediction, but the pooled LeWM latent does not read the plate within τ_re, so no TASK-077 bar is feasible |
| **PRED-ADMIT(A)** | K-P1–K-P5 pass | TASK-077 may be preregistered under cell A, as a task change declared under R2, with the three declarations below |

**Stated in advance.**
- If H-rule(A) ≥ H-final(A) − 2/32, a rule-knowing non-world-model arm reaches the ceiling.
  TASK-077 could then claim only that LeWM *can drive* the place target under an action-dependent
  prediction condition and that it beats the action-blind, scene-blind and random twins. It could
  not claim that a world model is needed.
- On M, an action-blind predictor is expected to tie, and H-cv to come near H-final (§3.2).
- On A, an action-blind predictor can also tie, if TASK-077's training aims are centred on the
  fixed point g\* and its controller aims at one predicted plate (§3.2, R8.8). So PRED-ADMIT(A)
  admits TASK-077 only with three declarations made before its freeze: its controller form
  (which should rank candidate aims by their individual predicted outcomes), the spread of its
  training aims relative to g\*, and the action-blind twin's expected result.
- Whether TASK-066's history-one predictor suffices is decided by c_plate + m2\* ≤ τ_re (§3.2,
  R8.9), reported with the row.

### 6.4 Blind and prior-only baselines against every bar

| bar | baseline | where it sits |
|---|---|---|
| O1 | none; τ_re is measured | the plate is in view. For scale only: TASK-075's pooled-token plate target readout (a different readout at different frames, §1) reported 0.493 [0.455, 0.511] cm on onboard 112 |
| O3 | the offline clock prior | fitted on the outer-training roots |
| O4 | plate-hidden frames | stored in the views store |
| G1 | H-handover | privileged; it is an upper bound for the arm family |
| G2 | H-clock | measured in the run (its K0 count is in-sample) |
| S-VOID-CONDITION, K0 stop | H-stale | TASK-074 K1: P-stale 0/32 at 9 cm |
| K-P2, K-P3, K-P4 | H-now, H-twin, H-cv | measured in the run |
| (reported) | H-rule | measured in the run |

## 7. Budgets, void rule and abandonment clause

**Budgets.** This task trains nothing iteratively. Every readout is a closed-form ridge with λ
chosen by inner cross-validation. So there is no update count, no saturation rule and no last-two
rule.
- **TASK-074's escalations came from exactly such a rule.** Its BUDGET block (cap 60 000,
  calibration 60 000 updates, factor 2) is the same as TASK-073's. Both fail
  `run_tools.check_budget` with "cap 60000 < 120000: factor 2 x calibration 60000 can be
  requested, so the rule escalates by design" (checked on main at f41b94f).
- **The frozen code will carry `TRAINING_BUDGET = None`,** and a test will assert it.
- **If a reviewed amendment ever adds iterative training,** its BUDGET block must pass
  `run_tools.check_budget`. It must select checkpoints with `run_tools.select_checkpoint` (the
  earliest point within tolerance), and apply the last-two rule only through
  `run_tools.last_two_triggered`. A test will check that too.

**Caps** (wall time; exceeding one is a V, never an escalation):
- 7 200 s per runner invocation. K0, O, D and S/U are one invocation each; K-pred is one
  invocation per cell (M-a, M-b, A), so no invocation's estimate exceeds 20 minutes (§10);
- 300 s per attempt (TASK-074's). The heaviest attempt is an H-final(A) attempt with the full
  look-ahead (§10);
- 1 800 s for Stage O's featurisation.

Memory: process-tree PSS ≤ 12 GiB (`run_tools.MemoryWatch`). GPU: §8. Each cap is at least 5
times the corresponding estimate in §10 (checked there). No stage has a row that a budget can trigger: every escalation in §6
comes from a measured result.

**Void rule.** A stage is V on any of:
- a stop signal (`run_tools.install_guards`, which records the first signal);
- a cap;
- a CUDA allocation failure;
- a pin or frozen-sha mismatch;
- a decoded test root;
- a privileged read in a non-privileged arm;
- a failed determinism re-run;
- G-repro failing.

A V after the stage's outcome boundary is not read as an outcome. One repeat from scratch is
allowed after a reviewed fix (TASK-074 §7 precedent). A second V escalates.

**The abandonment clause** fires on TWIN-OFF-FAIL, TWIN-FAIL or TWIN-HARM. It does not fire on
TWIN-NEAR, TWIN-PRIOR, TWIN-OFF-ARM, S-VOID-CONDITION, U-VOID-CEILING, TWIN-DEV-STOP or
CAL-ESCALATE. **Its scope:**

> "aiming e9's place primitive, after P-3's pick, at a single-frame frozen-DINOv2 ridge readout of
> the plate from the onboard 112 px camera, under TASK-074's 9 cm condition on `apple-to-plate-v2`:
> no further readout variant of this formulation (pooled or full tokens, λ grid, crop, colour) is
> preregistered without new evidence of a different kind (a place primitive with its own feedback,
> temporal aggregation of readings, or a new view or sensor)."

The next step is then TASK-075's Option 2 (a place servo; R3).

**K-pred's own clause.** It fires on PRED-NONE only, not on PRED-NEAR, PRED-INFEASIBLE,
PRED-NO-BAR or PRED-NOT-RUN. It
closes preregistering a LeWM plate-target place task on v2 under the tested action-dependent rule
(cell A: κ = −0.5 and the L and s1 it ran with) and under the constant-velocity family (M-a, M-b), without
new evidence of a different kind. It does not close the LeWM backend, the v2 task or the product
goal.

**Never closed by this task:** the LeWM backend, DINOv2 as an encoder, the v2 task, Arena and the
product goal.

## 8. Platform, GPU plan under the shared lock

- **Platform.** The Linux PC (RTX 5080, 16 GB), MuJoCo 3.13.0, `MUJOCO_GL=egl`, strict CUDA
  determinism in the main process. Simulation runs on the CPU with 6 workers, 1 torch thread each.
  The pinned thread environment (G-threads) and the quiet-machine rule (G-quiet: 1- and 5-minute
  load ≤ 2.0 at start) are carried.
- **What uses the GPU.** Only Stage O's DINOv2 featurisation: about 3 000 onboard 112 px frames.
  That is 253 roots × 6 visible decision frames, plus 6 plate-hidden frames per root at the eval
  steps.
  - TASK-075 featurised 10 626 frames per view, plus its full-token Gram, in 76–92 s. Its peak
    was 0.85 GiB allocated and 0.96 GiB reserved, against a 3.0 GiB cap
    (`apple_obs_ceiling_v2_results.md` §1.3).
  - Estimate: under 2 minutes and under 1 GiB.
- **How.** `scripts/gpu_run.sh --wait --min-free-gib 4 --board --who oej:task076-O -- uv run
  --no-sync python scripts/run_plate_twin_v2.py offline …` (the runner exists since Stage 0).
  - The runner calls `run_tools.gpu_guard(report, min_free_gib=1.0, process_cap_gib=3.0,
    require_lock=True)`, so it refuses to start outside the lock.
  - The lock is the machine-wide flock `~/.local/state/gpu/lock`, shared with other projects.
    `--wait` queues behind the current holder instead of failing.
- **Everything else runs on the CPU:** K0, D, S/U and K-pred. That is simulation plus CPU
  encoding with batch 1 in the workers, as in TASK-073/074. K-pred's privileged look-ahead for
  H-final(A) also runs on the CPU. These stages do not take the GPU lock. They are started only
  when G-quiet holds; a neighbour's long CPU job delays them and does not void them.
- **Rendering without the lock.** The simulating stages (K0, D, S/U, K-pred) render their frames
  with MuJoCo's EGL backend (`MUJOCO_GL=egl`) in the CPU workers, and that rendering does **not**
  take the GPU lock, as in TASK-073, TASK-074 and TASK-075. K0 ran this way (§5.1). This is kept
  deliberately (R8.19, on note 3 of the K0 GO review): rendering under the lock would be a change
  of this plan and would need its own ruling. `gpu_run.sh` and `gpu_guard(require_lock=True)`
  apply to Stage O's featurisation only.
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

**What it hands the next task (TASK-077, only on PRED-ADMIT(A)):**
- c_plate, measured on the pooled 4 × 4 LeWM latent (R-plate-pool), and τ_re. They bound
  TASK-077's predicted-latent plate bar: c_plate ≤ B ≤ τ_re. Any allowance above c_plate is
  calibrated on development data before TASK-077's freeze.
- H-twin's, H-cv's and H-rule's counts on cell A. LeWM is reported beside them, and TASK-077
  declares its own tie bar against the best of them.
- Cell A's rule, as a task change declared under R2.
- m2\* and the history rule: TASK-066's history-one predictor qualifies only if
  c_plate + m2\* ≤ τ_re, or B's calibrated allowance covers m2\* (§3.2, R8.9).
- Three obligations, declared before TASK-077's freeze (§3.2, R8.8): its controller form (which
  should rank candidate aims by their individual predicted outcomes), the spread of its training
  aims relative to the fixed point g\*, and the action-blind twin's expected result.

**A horizon caveat for TASK-077.** On cell A, the plate's final position at s1 = 525 lies 40 to
120 steps after the decisions at 405–485. TASK-066 gated its predictor only at h = 8 and h = 16. A
recursive roll-out to s1 is therefore ungated. TASK-077 must gate its own horizon, offline, before
any closed loop, with TASK-066's no-collapse, copy-last, no-action and action-sensitivity gates at
that horizon.

**If the row is PRED-NONE, PRED-NEAR, PRED-INFEASIBLE, PRED-NO-BAR or PRED-NOT-RUN,** Option 1 has not advanced the LeWM
goal beyond calibration. PLAN.md's Branch B applies.

## 9b. Limitations (stated plainly)

- **A static plate needs no world model.** The primary question cannot say anything about LeWM.
- **G1 is a point bar on 64 resets.** It has power 0.59 at a true rate of exactly 87.5 % (§6.2),
  and TWIN-NEAR handles a miss within noise. τ = 1.0 cm rested on one run at exactly its bar
  (28/32, Wilson [0.72, 0.95]); K0 re-measures it once, on 32 resets.
- **The readout is fitted on e9 corpus frames and run on P-3-pick frames.** The corpus is
  TASK-074's e9 plan with mis-aims and noise, not P-3's carry. Probe C's 5.53 cm
  in-distribution-only error shows how much the training frames matter. Stage D is the cheap check
  for this before the gated seeds are spent.
- **The closed loop has feedback.** From 405 on, the arm moves towards H-twin's last reading, and
  the frames then show the arm there. O4 checks offline that the readout reads the plate. It
  cannot check the closed loop's own feedback; the per-decision readings are logged.
- **τ is anisotropic** (−y errors fail earlier). The signed errors are reported, not gated.
- **Cell A is a declared simulator rule.** With L = 2, the target depends on the aim chosen at the
  decision that sets it (R8.7). Its feasibility is unverified until Stage 0, and moves against the
  approaching hand may be refused. TASK-066's history-one predictor misses the last 2 executed
  palm steps; that term, m2, is measured from executed palm motion (about 0.85 cm in xy if the
  commands sit at the clip), and it adds to c_plate in the history rule (§3.2). A hand-written arm
  that knows the rule (H-rule) is expected to come close to the ceiling, which §6.3 states in
  advance. An action-blind predictor can tie on A if the training aims are centred on the fixed
  point and the controller aims at one predicted plate (§3.2, R8.8).
- **The remedies for too little remaining motion are weak** (R8.10). Lowering L to 1 changes
  almost nothing, and moving s1 later helps only if the palm still moves in xy during the lower.
  |κ| ≥ 1 is forbidden. A shortfall ends in PRED-INFEASIBLE.
- **H-cv and H-rule know the condition's structure.** A later world model would have to learn it.
- **One encoder, one view, one input size, one floor seed, one corpus.**

## 10. Compute estimate (from measured stage times)

| stage | attempts or work | device | estimate | basis |
|---|---|---|---|---|
| K0 | 288 + 64 attempts | CPU, 6 workers | about 6 min | TASK-075 τ: 288 attempts in 265 s |
| O | featurisation; cross-fitted ridges (full and pooled) on at most 1 518 rows (253 × 6) | GPU < 2 min, then CPU | about 15 min | TASK-075 readouts: featurisation 76–92 s per view; cross-fit 221–228 s per view for more readouts |
| D | 32 attempts | CPU | about 2 min | as K0, plus the CPU encoding |
| S/U | 384 + 96 attempts | CPU | about 10 min | as K0, plus six encodings per H-twin and H-floor attempt |
| K-pred, M-a and M-b (one invocation each) | 32 × 4 = 128 attempts per cell | CPU | at most about 10 min per cell | as K0 (about 5.5 worker-seconds per attempt: 265 s × 6 workers / 288), plus H-twin's and H-cv's CPU encodings |
| K-pred, A (one invocation) | 32 × 5 = 160 attempts | CPU | at most about 15 min | as M, plus H-final(A)'s look-ahead: at most 10 cloned roll-outs to s1 per decision, of 120, 104, 88, 72, 56 and 40 steps, so at most 4 800 extra steps per attempt, about 6.5 attempts of 740 commands (about 36 worker-seconds) |

Total machine time is about 1–1.5 h. The GPU is used for under 2 minutes.

**The caps against these estimates (§7).** Per invocation (7 200 s): the largest estimate is
Stage O's and cell A's 15 minutes, a factor of 8. Per attempt (300 s): an ordinary attempt is about
5.5 worker-seconds, and the worst H-final(A) attempt about 42 (5.5 + 36), a factor of about 7.
Featurisation (1 800 s): under 2 minutes, a factor of at least 15. The Stage-0 smokes measure
the per-attempt time of a full-look-ahead H-final(A) attempt; if it exceeds 60 s, the per-attempt
cap is reviewed before the freeze.

## 11. Deviations from the card's proposed shape (disclosed)

| card | this draft | why |
|---|---|---|
| closed loop on "32 fresh development resets", bar 28/32 | D: 16 development resets as a stop rule; then gated S (64) and U (32) | the gated count must come from fresh seeds after a development check. 64 resets narrow the interval at the same bar fraction by about √2, and the cost is minutes |
| "an image-free clock-prior arm" | H-clock fitted on K0's resets under the same condition | the strongest image-free target; a corpus-wide clock prior mixes unshifted and 3–12 cm roots |
| bar 28/32 carried | G1 56/64 = τ's bar, feasibility from K0's measured ceiling; O1's bar is τ_re, measured in K0 | the TASK-074 lesson: bars come from measured ceilings |
| — | Stage K-pred, with an action-dependent cell as the only admitting cell | rulings R8 and R8.3: without it, TASK-076 cannot inform the LeWM goal beyond calibration |

## 12. Review record and remaining open questions

The independent review of #129 at `314d843` (REQUEST CHANGES, eleven findings) is addressed in
this revision. The rulings it needed are R8.1–R8.6 in DECISIONS 2026-10-02 (b). The re-review at `3d8560e`
(one blocking finding: under L = 40, cell A's target ignored the last aim) is addressed by R8.7.
The third review at `ee3f3c9` (APPROVE as a DRAFT, four non-blocking items to settle before the
freeze) is addressed by R8.8–R8.10 and R8.12; the freeze-readiness check of §13 added R8.11 and
R8.13, and R8.14 added the PRED-NEAR noise guard. All are in DECISIONS 2026-10-02 (b).

Settled:
- **U runs H-handover** (32 more attempts; R8.13). It is the feasibility check for G3 through
  U-VOID-CEILING.
- **No hand-crop variant follows a TWIN-PASS in this task** (R8.13). In the white-plate
  development record (`apple_white_plate_dev.md`, blue plate), the hand crop's plate readout has
  an 87.5th percentile of 1.36 cm, against 0.91 cm on onboard 112. A hand-crop follow-up would be
  a new view, and so needs its own preregistration.

Settled by the Stage-0 smokes (§3.2's Stage-0 result; R8.16):
1. s1 = 525 is safe from apple–plate contact: no contact before s1 on any of the 144
   base-setting smoke attempts; the earliest contact was at step 607.
2. Cell A's rule does not give the remaining motion: the median under H-final was 0.060 cm at
   L = 2, and 0.022–0.063 cm under every declared remedy (L = 1; s1 = 535–605), with no refused
   move. Cell A is removed; K-pred's row is PRED-INFEASIBLE after an O-PASS (PRED-NOT-RUN
   without one).
3. m2: median 0.0012 cm, maximum 0.0019 cm. A full-look-ahead H-final(A) attempt: 15.0 s in
   median, 16.4 s at most (25.0 s over the remedies), below §10's 60 s.

## 13. Freeze readiness (checked 2026-10-04, R8.11–R8.14)

### 13.1 Every bar: its source, and its feasibility check or role

| bar | value | source | feasibility check, or role |
|---|---|---|---|
| O1 | upper bound of R-plate's median error ≤ τ_re | τ_re measured in K0 | precision bar on a measured tolerance; TASK-075's onboard-112 pooled plate readout (0.493 cm) is the scale reference (§6.4) |
| O3 | ratio upper bound < 1.0 | definitional | image versus a prior that reads none |
| O4 | plate-hidden lower bound > τ_re | τ_re measured in K0 | a check on the readout's source |
| O2 | 87.5th percentile and τ-curve prediction | — | **reported only** (R8.5) |
| G1 | H-twin(S) ≥ 56/64 | τ's own bar fraction, 28/32 | feasible only if N_K(0) ≥ 30/32 (K0 stop); power stated (§6.2); TWIN-NEAR covers a miss within noise |
| G2 | exact one-sided McNemar p < 0.01 | definitional | K0 stop H-clock(K) ≥ N_K(0) − 4/32; G2's predicted feasibility from K0's paired counts, reported beside it (R8.12) |
| G3 | H-twin(U) ≥ P-stale(U) − 2/32 | declared margin, carried from TASK-074's G5 | U-VOID-CEILING when H-handover(U) itself misses it (R8.13) |
| S-VOID-CONDITION | H-stale(S) > 8/64; H-handover(S) < 56/64 | TASK-074 K1's P-stale ≤ 4/32, doubled; G1's bar | a row, not a bar on the hypothesis |
| K0 stops | level 0 < 28/32; N_K(0) < 30/32; H-stale(K) > 4/32; H-clock(K) ≥ N_K(0) − 4/32 | τ's bar; TASK-074 K1's ceiling and P-stale bars; G2's minimum separation | stops (CAL-ESCALATE) |
| TWIN-DEV-STOP | H-twin(D) < 12/16 or H-handover(D) < 14/16 | declared stop | at true rates of 87.5 % and 93.75 %, these fire with probability 0.04 and 0.07 (exact binomial); a stop, no clause |
| K-P1 | H-final(A) ≥ 30/32 | TASK-074 K1's ceiling bar | itself the ceiling; a miss is PRED-INFEASIBLE |
| K-P2–K-P4 | +8/32 each | TASK-074 K1's headroom bar | feasible whenever K-P1 passes (it leaves room for an 8/32 gap); a miss within noise is PRED-NEAR, so PRED-NONE fires falsely about 2–3 % of the time per bar at a true 8/32 (R8.14, §6.3) |
| K-P5 | c_plate ≤ τ_re | c_plate (O1's form) and τ_re both measured | a miss is PRED-NO-BAR |
| H-final(A) tolerance | τ_re/4, at most 10 iterations | declared (R8.7) | the contraction estimate is checked in the smoke |
| m2\*, c_plate + m2\* ≤ τ_re | — | measured (R8.9) | **reported**; it decides TASK-077's history obligation, not a TASK-076 row |
| every other number in §6 | — | — | **reported only** |

The +8/32 headroom bars have a near-noise escalation row, PRED-NEAR (R8.14), like G1's TWIN-NEAR.
Without it, a true headroom of exactly 8/32 would have missed its bar, and fired the clause,
42–45 % of the time.

### 13.2 What must happen before the freeze, in order

1. **Stage 0, in the preregistration PR** (the task card's acceptance criterion): the stage code
   in new modules, the frozen block as module constants with its sha256 pinned in a test, the
   manifest under `benchmarks/manifests/`, and the tests of §5 (seed ranges against
   `obs_ceiling_v2.FORBIDDEN_RANGES` and TASK-075's block; `TRAINING_BUDGET is None`; the scale
   probe through `run_tools.scale_probe`; no runner imports; `assert_local_import`,
   `install_guards` and `gpu_guard`).
   **Done** (R8.15): `plate_twin_v2.py` (the frozen block), `plate_twin_v2_runtime.py`,
   `plate_twin_v2_offline.py`, `plate_twin_v2_harness.py`, `scripts/run_plate_twin_v2.py`,
   `tests/test_plate_twin_v2.py` and the manifest (DRAFT until step 4, which set the pin).
2. **The Stage-0 smokes** on 56900–56999, which settle §12's open items. Their results, and any
   change they force (s1, L, or cell A's removal), are written into this protocol. If a
   full-look-ahead H-final(A) attempt takes more than 60 s in the smoke, the per-attempt cap
   (300 s) is reviewed before the freeze (§10).
   **Done** (R8.15–R8.17): s1 = 525 stays; cell A is removed (PRED-INFEASIBLE after an O-PASS,
   PRED-NOT-RUN without one); the look-ahead attempt took at most 25.0 s, so the cap is not
   reviewed.
   **Found in Stage 0, and cleared:** TASK-072 run-1's corpus root `look2-51171.npz` was
   unreadable on the archive disk, so the smokes ran P-3 on a stand-in estimate (§3.2). It has
   since been restored bit-identically (sha256 `e13c97cb…c296`). A real G-repro check then
   passed all eight of run-1's recorded facts (R8.18; Stage-0 record §2a).
3. **K0**, on a reviewer's reported GO for K0. Its values, including G2's predicted feasibility,
   are written into the frozen block (§5). A CAL-ESCALATE stops here.
   **Done** (R8.19): the GO was given on #134 at `2d0bdb7`; K0 ran once there and ended
   **K0-PASS** (§5.1; report sha256 `ef4b541067dc969ee5ebfc9ee78e2c719a4b84e4d3ed0d71a167feb6fcd0c876`).
   Its values are `K0_MEASURED` in the frozen block.
4. **The freeze:** status FROZEN and the frozen-block sha pin, merged on an independent
   reviewer's reported APPROVE.
   **Prepared** (R8.19): status FROZEN; the frozen block's sha256
   `011dff8ac65dde7341850ac4c4d9c569b399f1e3a8d2a1e978d3a3b9bbb7570b` is pinned in
   `tests/test_plate_twin_v2.py` (`FROZEN_SHA256_PIN`) and in the manifest
   (`frozen_sha256_pin`). It takes effect on the merge.

Steps 1–3 are done. Step 4 is this revision; nothing after K0 runs before it is merged.

### 13.3 The other freeze checks

- **Seeds and salts** are declared in §4: every cohort, the smoke range, every salt with its use,
  the reserved 7605, the fixed floor seed and the reset rule. The ranges were searched on all 50
  local and remote refs in the review of #129; the Stage-0 test re-checks them in code.
- **Budgets.** No stage trains iteratively, so there is no `BUDGET` block for
  `run_tools.check_budget` to check (it takes a TASK-073/074-style dict with a cap, factor and
  calibration count). The frozen code carries `TRAINING_BUDGET = None`, and a test asserts it
  and requires any later `BUDGET` to pass `check_budget` (§7). The wall-time caps are checked
  against §10's estimates.
- **GPU.** Only Stage O's featurisation uses it, through `scripts/gpu_run.sh --wait
  --min-free-gib 4 --board --who oej:task076-O`, with `run_tools.gpu_guard(...,
  require_lock=True)` inside (§8). The flags and the guard's signature exist on main.
- **Every row maps to an action:**

| stage | row | action |
|---|---|---|
| K0 | CAL-ESCALATE | escalate, no clause; nothing is frozen |
| K0 | K0-PASS | K0's values enter the frozen block; freeze review |
| O | TWIN-OFF-ARM | escalate, no clause; K-pred is PRED-NOT-RUN |
| O | TWIN-OFF-FAIL | the clause fires; next step Option 2 (R3); K-pred is PRED-NOT-RUN |
| O | O-PASS | Stage D and Stage K-pred may run, each on a GO |
| D | TWIN-DEV-STOP | escalate, no clause |
| D | D-PASS | Stage S/U may run on a GO |
| any | V | one repeat after a reviewed fix; a second V escalates |
| S/U | S-VOID-CONDITION, U-VOID-CEILING, TWIN-PRIOR, TWIN-NEAR | escalate, no clause, no claim |
| S/U | TWIN-HARM, TWIN-FAIL | the clause fires; next step Option 2 (R3) |
| S/U | TWIN-PASS | the claim of §1; results PR; the next LeWM task is chosen by K-pred's row |
| K-pred | PRED-NOT-RUN, PRED-INFEASIBLE, PRED-NO-BAR | escalate, no clause; PLAN.md's Branch B |
| K-pred | PRED-NONE | K-pred's clause fires; Branch B |
| K-pred | PRED-NEAR | escalate, no clause, no claim; any repeat needs fresh seeds and its own ruling |
| K-pred | PRED-ADMIT(A) | TASK-077 may be preregistered under cell A, with R8.8's three declarations and R8.9's history rule (Branch A) |
