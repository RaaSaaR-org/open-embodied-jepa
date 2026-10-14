# TASK-093 — a rate-projected planner: project each CEM candidate's grasp channel onto the embodiment's rate limit before the model scores it, then a closed-loop grasp-and-lift comparison against the unprojected planner on TASK-087's unchanged G

**Status: FROZEN at the Stage 0 PR's merge** (R27.115–R27.131 as the DRAFT, #220; Stage 0 record
§16, R27.132–R27.138; decided by Claude under owner delegation, 2026-10-14; DECISIONS (bd), (be)).
Stage 0 ended **S0-PASS**: every degeneracy check of §6.1, the projector's equivalence tests and
the power rule pass, and Stage P's bars are **C8@R1 ≥ 39 / 64 and H8@R2 ≥ 48 / 64**. Stage 0
changes none of the other two items §14 lets it change. Every later stage runs at the merged
revision from a clean tree with `scripts/run_task093_stage.sh`; Stage BS runs only after an
independent GO on Stage BD's record. **Stage P** (gated, offline, 64 roots, one run) ended
**P-PASS** (§17, R27.139–R27.141), and **Stage BD** (development, not gated, 32 resets, one run)
set the bar at **0.40** (§18, R27.142–R27.146). Stage BS has not run; it waits for an independent
GO on §18.

This is the planner-side fix that R27.104 and R27.113 route to after D-RATE′ read D-FLAT. It is a
planner variant over TASK-087's checkpoints, so it falls under R26.33. R27.115 (§2) rules that
R26.33's condition is met for this one variant, and only for it.

**This is not an Apple→Plate experiment.** It plans grasp-and-lifts of the play scene's objects
with G1's right arm and right Dex3, in MuJoCo only. **R7's canonical status sentence does not
change, whatever row this task ends in** (R27.131).

## 1. What the records say

Every number below carries the qualifiers of its record.

### 1.1 The closed-loop base rate of the unchanged G under TASK-089's planner

- **TASK-089 Stage BS** (one run, simulation only, 64 fresh gated resets; one model seed gated,
  two reported): G-chg-87100 **0 / 64** (exact 95 % 0.000–0.056), G-chg-87101 8 / 64
  (0.056–0.232) and G-chg-87102 7 / 64 (0.045–0.212), both reported only. The IK follower scored
  48 / 64.
- **TASK-091 Stage BS** (one run, simulation only, 128 fresh gated resets): M-v1-87100 **1 / 128**
  (0.000–0.043, reported only). ik scored **82 / 128** (0.551–0.723).
- **TASK-091 Stage BD** (development, 32 resets): M-v1-87100 2 / 32. ik scored 26 / 32.
- **D-RATE′** (development, one run, simulation only, 32 resets, not gated): M-v1-87100 0 / 32
  (0.000–0.109).

So the unprojected arm is near the floor, and its seeds differ. 87100 scored 3 of 256 over its
four records (about 0.01). 87101 and 87102 have one record each, 8/64 and 7/64 (0.125 and 0.109,
TASK-089 BS, reported only). Pooled, that is 15/192 ≈ 0.08 in TASK-089 BS and 18/384 ≈ 0.05 over
every record above (one run each, simulation only; records of different sizes and cohorts, pooled
for orientation only). The IK follower uses the same goals and no world model, and it lifts
on about two thirds of resets. It is the reference, not a competitor.

### 1.2 The planner queries the model off its grasp-rate support

- **The rate limit.** The embodiment clips the right-grasp synergy target to the previous target
  ± `joint_speed_limit_rad_s × control_dt / max slope` = 2.0 × 0.05 / 0.55 = **2 / 11 per command**
  (`embodiment.py`, `_prepare_side`; `configs/g1_sim_action.json`). From an open hand (−1) the
  executed channel first exceeds 0.5 at the 9th closing command.
- **Training support** (TASK-092 §1.2 and §15.2; development, then Stage 0 check). In `play-v1`
  train no executed grasp step exceeds the limit (largest |Δg| 0.181818), and 7.3 % sit at it.
- **The planner's queries** (TASK-092 §1.2; development). TASK-089's planner samples without
  candidate projection. In a first CEM iteration 83 % of within-plan grasp steps exceed 2 / 11,
  and no 8-step candidate is rate-feasible.
- **Chosen plans in closed loop.**
  - TASK-091 Stage BS run reports (one run, simulation only; a development reading of them, not
    gated): 28–45 % of chosen plan steps per world-model seed were off support.
  - D-RATE′ (development, one run, simulation only, 32 resets): 0.47 of M-v1-87100's chosen plan
    steps were off support, and the clip changed the grasp of 0.78 of its grasp-phase first
    commands.
- **What G predicts there** (TASK-092 Stage R, one run, offline, simulation only, `play-v1` test,
  821 R-roots). Under saturated requests, which the embodiment executes identically to the
  recorded commands, TASK-087's G predicts the right-hand state **15.0, 20.3 and 15.1 times** worse
  at h = 8 than under the recorded commands (ρ_ctrl, seeds 87100–87102). On val this was 7.4–9.5
  (development).
- **What G predicts under its own plans** (TASK-092 Stage A‴ record, reported, one run, offline,
  64 roots). At R1, the plan's executed grasp from the open hand is −0.818, −0.636, −0.455,
  −0.273, −0.091, … (the maximum rate). G-v1-87100 predicts the hand synergy at −0.69, −0.31,
  0.10, 0.46, 0.62, 0.67, 0.67, 0.61, about twice as fast as the hand can close; the other two
  seeds are alike.

### 1.3 What TASK-092 showed and did not show

- G-req, trained on rate-equivalent requests, removed the prediction gap (Stage R: ratio about 1;
  R-PASS on all three seeds; one run, offline, simulation only).
- But its plans did not become rate-feasible. In D-RATE′ (development, one run, simulation only,
  32 resets), 0.56 of M-req-87100's chosen plan steps were still off support, and the clip changed
  0.72 of its grasp-phase first commands. M-req-87100 and M-v1-87100 both scored 0 / 32.
- TASK-092 §10 and §16.3 name the reason as a hypothesis: a rate-aware model gives the same cost
  to every request beyond the limit (a flat cost), so CEM has no reason to prefer a feasible one.

### 1.4 Reading

Two fixes remove the mismatch between what the planner asks and what the model knows. One is on
the model side: G-req. It is installed (R-PASS) but left the flat cost. The other is on the
planner side: score only what the embodiment will execute. Projecting each candidate's grasp
channel onto the rate set does that by construction. The model then sees only rate-feasible grasp
sequences, which is the support it was trained on. The cost then ranks executed grasp sequences,
not requests.

**That the mismatch causes the closed-loop failures is a hypothesis.** The records show the
mismatch (§1.2). They do not show that it causes the failures. The failures also include low
grasp contact (39–87 of 128 per world-model seed in TASK-091 BS, against ik's 122; one run,
simulation only), and the arm channels may cause that. This task does not change the arm channels.

## 2. R26.33 and the other closed scopes

**R26.33** (TASK-089 §9, fired at B-FAIL): "CEM planning of grasp-and-lift over TASK-087's
checkpoints (trained on `play-v1` as collected) is closed; no further planner, cost, horizon or
action-parametrisation variant over these checkpoints is preregistered for grasping without new
evidence of a different kind."

**R27.115 — R26.33's condition is met for exactly one variant: grasp-channel rate projection
(§4).** The reasons:

1. **The evidence is of a different kind.** R26.33 fired on closed-loop counts (0 / 64 on the
   primary seed) after an offline diagnosis that passed (A1, A2). Those offline measures compared
   executed, rate-feasible branches, and they could not see the planner's queries.
   - The new evidence measures the interface between the planner and the model: what the planner
     asks, and what the model answers there.
   - First, the planner queries the model where it was never trained (§1.2, development).
   - Second, G's hand-state error there is 15–20 times its error on the recorded commands. That
     comes from TASK-092 Stage R, a gated offline stage on held-out roots (one run, simulation
     only).
   - Third, G's own plans predict a hand that closes about twice as fast as it can (the A‴
     record).
   - None of this existed when R26.33 fired. It names a specific mechanism that a specific
     planner change removes.
2. **The variant is the one change the evidence names.** It changes which candidate sequences the
   model scores, and nothing else. The cost, horizon, λ, CEM budget, model and action space stay
   as frozen in TASK-089 (§4.3).
3. **The admission is narrow.** It does not admit any other planner, cost, horizon or
   action-parametrisation variant over TASK-087's checkpoints. Examples it does not admit:
   - tie-breaking towards executed requests;
   - full-embodiment candidate projection (the arm channels through the IK surrogate);
   - a different horizon or λ;
   - the held grasp, or the h = 16 setting.
   Each of these would need its own ruling. R26.33 is otherwise neither widened nor narrowed.

**R27.116 — the other closed scopes do not cover it.**

- **R27.71** (TASK-091's clause): closes more scripted grasp-and-lift play of
  `play_policy_grasp_v0`'s kind as the grasp fix. This task adds no data.
- **TASK-092 §8:** did not fire (A‴-FAIL, no clause; R27.100). Its "not closed" list names
  planner-side candidate projection.
- **TASK-062 and TASK-065:** encoder training on `apple-wide-v1` and predictors on pooled CLS on
  `apple-look-v1`. Neither applies.
- **TASK-054** (CEM over that apple world-model cost as the *primary* control line): this task
  makes no claim that CEM is the primary line. It is a test within the Phase 3 grasp line, as TASK-088 to
  TASK-092 were.

## 3. Planner-side options, ranked

| Rank | Option | For | Against, cost | Scope |
| --- | --- | --- | --- | --- |
| **1 (chosen)** | **Grasp-channel rate projection** of every CEM candidate before the model scores it (§4). | It removes the off-support queries by construction. The cost then ranks executed grasp sequences, so the flat cost goes. It changes only which sequences are scored, and its implementation is NumPy over 300 × 8 values per iteration (negligible next to the model call). It reuses `CEMPlanner`'s existing `projector` hook. | It covers the grasp channel only; the arm channels are still executed through backtracking and the IK, unscored. | Admitted by R27.115. |
| 2 | **Tie-breaking towards executed requests** (#216 review): among equal-cost candidates, prefer the one whose requests are feasible. | Small change. | It only breaks ties. For G-v1 the costs of off-support requests are not tied but wrong (§1.2), so tie-breaking does not touch G-v1's mismatch. | Not admitted (R27.115 (3)). |
| 3 | **Full-embodiment projection** (`CEMConfig.project_candidates` with `robot.project_candidates`, the IK surrogate). | It would project the arm channels too. | It runs the IK about 300 × 8 × 10 times per command. Not measured; TASK-092 §2 judged it likely far above the 7 200 s per-job cap at 400 commands. It is also two changes at once. | Not admitted (R27.115 (3)). |
| 4 | **Projection plus G-req** as a further arm. | It is consistent on both sides. | Under projection every query is rate-feasible. On those queries Stage R's R2 found G-req within δ of G-v1 under the recorded commands, so the arm adds about 6.4 GPU h at BS for little information. G-req also stopped at A‴-FAIL. | Not taken (R27.120). It is closed with the clause (§8). |
| 5 | **Projection over TASK-090's six G checkpoints** (M-v2g, M-v1p). | Their recorded rates are slightly higher (6–11 / 128 per seed in TASK-091 BS; one run, simulation only). | The evidence of §1.2 that R27.115 rests on (Stage R's ratios, the A‴ record) was measured on TASK-087's G. Two corpora would add a factor. TASK-091's comparison is closed. | Not taken. |

**Why rank 1.** It is the planner change that the measured mechanism names. It is a single
change with an exactly paired control: the same checkpoints, resets and CEM seeds, and the same
first-iteration samples (§4.2). It needs no exception to the simulator-truth rule, because the
rate limit is the robot's own declared command limit.

**Judgement, not measurement.** The drafter judges that a gated B-PASS (the plan's 0.40) is very
unlikely (under 5 %), and that a measurable gain over the unprojected planner (B-PASS or B-GAIN)
has about a 15–30 % chance. Neither is a number this task relies on.

## 4. The planner change

### 4.1 The projector

For each candidate sequence u₁ … u_H (H = 8) of normalised 14-D commands, only the right-grasp
channel (index 13, `zero_shot_grasp.GRASP_DIM`) changes. Every other channel is passed through
bitwise.

- **Previous targets.** Let T₀ be the embodiment's last accepted right-hand actuator targets: the
  `previous_targets` that `_prepare_side` reads on the next command.
  - In closed loop, T₀ is read from the embodiment at plan time. It is the embodiment's own
    command state, the quantity `project_candidates` already uses. It is not simulator truth: no
    object, contact or future state enters.
  - Offline (Stage P), T₀ is the synergy target of the root's grasp: −1 at R1 and +1 at R2, the
    demonstrator's grasp before each root (as R27.91 read it for A‴), through the synergy line and
    `_joint_commands`.
- **Per step k = 1 … H.**
  - The bounds [ℓ_k, υ_k] come from `_prepare_side`'s own per-joint formula over the seven
    right-hand actuators, with prior targets T_{k−1} and `max_delta = joint_speed_limit_rad_s ×
    control_dt`. Each joint with a non-zero slope gives the interval (T − max_delta − centre) /
    slope … (T + max_delta − centre) / slope. The bounds are the intersection with [−1, 1].
  - The projected grasp is g_k = clip(float(u_k[13]), ℓ_k, υ_k), computed in float64 as
    `_prepare_side` does. The candidate stores float32(g_k). T_k is computed from the float64 g_k,
    `opened + (g_k + 1) × 0.5 × (closed − opened)`, and passed through `_joint_commands`, which
    casts it to float32 and clips it to the command limits; that float32 T_k is the next step's
    prior.
  - So the clip is cumulative: each step is clipped around the previous *projected* step, from
    the current executed hand state.
- **Equivalence.** In closed loop, step 1's projected grasp is the grasp the embodiment will
  execute for that request. Stage 0 tests this bitwise (§14).
- **Feasibility.** A grasp clip always has a feasible value, so every candidate stays feasible. If
  `_prepare_side`'s "no rate-feasible synergy" condition arises, the projector raises, as the
  embodiment would. That is a B-VOID or P-VOID, never a silent fallback.
- **Code placement.** The projector lives in a new opt-in module (NumPy at import), which reads
  the manifest values from `configs/g1_sim_action.json` and the MJCF-derived open and closed
  targets. `embodiment.py` is not edited, so every frozen hash stays.
  - The architecture rule says the embodiment owns command limits. The new module copies the
    formula of the one limit it needs, and Stage 0 checks the copy against the embodiment's own
    projection (§14).
  - This is disclosed as a risk (§11). It is not a second definition of the limit.

### 4.2 How the planner uses it

TASK-089's planner (TASK-088's `CEMPlanner` with TASK-089's step-averaged `StepMeanModel`) is used
through its existing hook: `CEMConfig.project_candidates = True` with the projector of §4.1 passed
as `projector`. `planning.py` is not edited. In that existing code path:

- candidates are sampled exactly as before (the same RNG calls, so with the same CEM seed both
  arms draw identical first-iteration samples);
- the projected candidates are rolled out and scored;
- the elites are refitted on the **requests**, as the existing code does ("since clipping absolute
  grasps is not invertible");
- the returned plan is the best **projected** sequence, so the command sent to the embodiment
  already lies inside the rate set, and the embodiment's grasp clip does not change it.

The step-averaged cost, λ = 2.0552 and the CEM budget are unchanged (§4.3). The CEM seed of each
reset is TASK-088's `cem_seed(reset)`, as in the control.

### 4.3 What stays frozen

- **Checkpoints.** TASK-087's G-87100, G-87101 and G-87102 (`checkpoints/task087-run/`), under
  their existing pins and loaders.
- **Latent and encoder.** TASK-087's frozen DINOv2 pooled latent and its projection.
- **Cost.** The step-averaged cost with λ = 2.0552 × the standardised joint-position MSE.
- **CEM.** H = 8, 300 samples, 10 iterations, 30 elites, minimum std 0.05, MPC every command, the
  left arm and left grasp fixed (`FIXED_LOWER` and `FIXED_UPPER`).
- **Task.** TASK-088's scene and grasp goals (its demonstrator, four subgoals), subgoal switching,
  the 400-command budget, execution through the embodiment's projection and the 5 rad/s velocity
  stop.
- **Scorer.** The target ≥ 5 cm above its start height and in grasp contact for 20 consecutive
  commands, with no latching.

**No other planner knob changes, now or after the freeze, whatever any stage shows.**

## 5. Question, hypotheses and arms

**Question.** With TASK-087's G, TASK-089's step-averaged planner and every other setting held
fixed, does projecting CEM's candidates onto the grasp-rate set make the closed loop grasp and
lift more often than the unprojected planner, and does it reach the plan's bar?

- **H1 (offline, no regression).** In executed terms, the projected planner still closes at R1
  and lifts while holding at R2 (Stage P).
- **H2 (closed loop).** M-proj (three seeds pooled) beats M-v1 (the same three checkpoints,
  unprojected) on grasp-and-lift on the same resets, and reaches the plan's bar.

| Arm | Model | Planner | Role |
| --- | --- | --- | --- |
| **M-proj-{87100, 87101, 87102}** | TASK-087's G | TASK-089's G-chg **with the grasp-rate projector** (§4) | **primary, pooled** |
| M-v1-{87100, 87101, 87102} | TASK-087's G | TASK-089's G-chg, unprojected (as TASK-089, TASK-091 and TASK-092 ran it) | **gated control** |
| ik | — | TASK-088's IK follower with e9's press | reference; its BD rate sets the bar |
| ik-nopress | — | the same without the press | reported |
| hold, random | — | TASK-088's | floors |

- **M-v1 is a control re-run, not a variant under R26.33** (as in TASK-090 §3, TASK-091 and
  TASK-092 §5). It carries no claim of its own.
- **G-req is not an arm** (R27.120, §3 rank 4).
- **Simulator truth.** Object truth is read by the demonstrator, the root builder, the offline
  labels and the scorer only. It never enters a model input, the projector or the planning cost.

## 6. Stages and gates

### 6.1 Stage 0 calibration of Stage P (development roots; before the freeze)

Stage P's gate uses **executed** plan measures, as R27.98 and R27.104 require. TASK-092 showed how
such a measure can be degenerate: at R2 the executed grasp after one clipped step is at least
0.818 whatever the request, so that half of A‴'s post-hoc reading could not fail. So Stage 0
checks every measure for degeneracy, and sets its bar on **development roots** that no gated stage
uses: **916000–916063**.

- **Measures** (per root; binary; executed = the plan's grasp after the cumulative clip of §4.1
  from the root's grasp, which is the plan itself for a projected plan):
  - **C8@R1, the executed close:** at R1 (goal: the grasp subgoal), the chosen 8-step plan's
    executed grasp reaches the maximum-rate value at step 8 (≥ −1 + 16 / 11 − 10⁻⁶, about 0.455).
    That is, it closes at the maximum rate through the whole horizon.
  - **H8@R2, the executed lift while holding:** at R2 (goal: the lift subgoal), the chosen plan's
    executed grasp stays above 0.5 at all eight steps, **and** its first command's right vertical
    command (dimension 8) is above 0.
- **References** on the same development roots, all through the same clip:
  - the control: each M-v1 seed's chosen plan (unprojected planner);
  - the projected planner on each seed (reported only, so a reader can see whether the bar is in
    reach; it does not set the bar);
  - **open/release:** requests −1 at every step at R1 and at R2, with the vertical command +1 at R2;
    - **random:** CEM's first-iteration distribution, N(0, 1) clipped to the bounds, with salt 9011;
  - **closed, vertical −1** (H8 only): requests +1 at every step with the vertical command −1 at
    R2. It must score 0 / 64, which shows that H8's vertical half can fail on its own.
- **Bar rule** (per measure m): **bar_m = min(48, ⌈0.75 × median over the three M-v1 seeds of
  their development count⌉)** out of 64. The projected planner may lose at most a quarter of the
  control's executed rate, and the bar is capped at TASK-091's 48 / 64.
- **Degeneracy checks** (per measure; all must hold, or Stage 0 escalates before the freeze, and
  no measure is silently replaced):
  - (a) open/release scores **0 / 64**, so the measure can fail, and for H8 the closed,
    vertical −1 reference also scores 0 / 64;
  - (b) random scores **≤ ⌊bar_m / 2⌋**, so it is not met by chance;
  - (c) the control's median development count is **≥ 16 / 64**, so a no-regression bar means
    something.
- **Context, not calibration.** In TASK-092's post-hoc A‴ reading (reported only, not gated, A‴'s
  roots, which are read), "maximum rate through step 8" at R1 held on 44, 56 and 53 of 64 for
  G-v1. That motivates C8 as a measure that can fail. It sets nothing: the bar comes only from the
  development roots above.

### 6.2 Stage P — executed-measure no-regression gate (offline, gated)

- **Roots: 917000–917063**, 64 fresh roots, built with TASK-089's root builder through TASK-091's
  frozen modules (the target object, R1 and R2 per root, as A″ and A‴ did). The rejected draws are
  recorded.
- **Per seed:** one CEM call per root at R1 and at R2, with TASK-089's A2 seed, for M-proj-i and
  M-v1-i. These use the same seed for both arms, so their first-iteration samples are identical.
- **Gate per M-proj seed:** C8@R1 ≥ bar_C8 and H8@R2 ≥ bar_H8 (the bars frozen at Stage 0).
- **Plumbing checks** (each a P-VOID if it fails):
  - every M-proj plan's grasp steps are inside the rate set (|Δg| ≤ 2 / 11 + 10⁻⁶ from the root's
    grasp);
  - M-proj and M-v1 drew identical first-iteration samples;
  - the root, checkpoint, λ and frozen-file hashes match.
- **Reported only:**
  - M-v1's counts;
  - **P-MECH:** the chosen plan's predicted right-hand synergy (`synergy_grasp` on predicted
    joints) against its executed grasp path, as mean |gap| over steps 1–8 at R1, per seed and arm;
  - the share of M-proj's *requests* beyond the limit;
  - close@R1 and lift@R2 in TASK-089's request terms;
  - the A1 comparisons. These are model-only and unaffected by the planner, so they are not
    re-run.
- **Rows:** P-VOID; **P-FAIL** if two or more M-proj seeds fail the gate (escalate, no clause);
  otherwise **P-PASS**. A seed that fails alone stays in M-proj's pooled count, and its Stage P
  results are reported beside its closed-loop count.

### 6.3 Stage BD, GO and Stage BS — TASK-091 §5's closed loop with M-proj in the treatment's place

- **Stage BD (development, not gated):** 32 fresh resets, **91900–91931**.
  - Arms: ik, ik-nopress, hold, random, M-proj-87100 and M-v1-87100.
  - **bar = min(0.40, 0.9 × r)**, where r is ik's BD rate. If r < 0.30 the bar is not set and the
    task escalates.
  - BD can show a defect, whose fix needs its own ruling. It changes no knob by itself.
- **GO:** an independent reviewer posts GO or NO-GO on BD's record before Stage BS. After a NO-GO,
  BD is recorded again on fresh development seeds named by a ruling.
- **Stage BS (gated): n = 128 fresh resets, 918000–918127**, every arm of §5 (six world-model
  runs), at the frozen revision from a clean tree.
  - Each world-model run is two 64-reset jobs (918000–918063 and 918064–918127) under the 7 200 s
    cap, combined before any summary.
  - **Determinism re-run:** M-proj-87100 on 918000–918003, which must match bitwise.
- **Notation.** k_p is M-proj's pooled successes of 384, and k_c is M-v1's. The pooled bar is
  3 × ⌈bar × 128⌉ and the per-seed floor ⌈⌈bar × 128⌉ / 2⌉; at bar 0.40 they are **156 / 384**
  and **26 / 128**.
- **Gating tests (reset level),** as TASK-090 §7.2 and TASK-092 §6.4. For each reset r, P_r is
  M-proj's successes over its three seeds, C_r is M-v1's and R_r is random's.
  - "M-proj beats M-v1": the exact one-sided sign-flip test at p < 0.01 over d_r = P_r − C_r.
  - "M-proj beats random": the same over d_r = P_r − 3 R_r (approximate and conservative, as
    there).
  - Reported: the pooled exact McNemar test, and the reset-cluster bootstrap 95 % interval of
    (k_p − k_c) / 384 (10 000 resamples, salt 9006).
- **Rows** (first match):
  1. **B-VOID** — any of these:
     - a run did not complete under the rules (including the cap);
     - a check failed (goal, checkpoint, frozen-file hash, λ, seed, the projector equivalence of §4.1 on any executed command (request channel 13 bitwise
       equal to the applied channel 13 returned by `zs.execute`));
     - the determinism re-run did not match;
     - a cohort was run twice.
  2. **B-PASS** — all of: k_p ≥ the pooled bar; every M-proj seed ≥ the per-seed floor; M-proj
     beats random; M-proj beats M-v1.
  3. **B-GAIN** — M-proj beats M-v1, but k_p is below the pooled bar, a seed is below the floor,
     or M-proj does not beat random. Escalate; no clause.
  4. **B-BAR** — k_p ≥ the pooled bar and M-proj beats random, but M-proj does not beat M-v1.
     Escalate; no clause; no Phase 4 by itself.
  5. **B-FAIL** — all of: M-proj does not beat M-v1; k_p is below the pooled bar or M-proj does not
     beat random; the bootstrap 95 % upper bound of (k_p − k_c) / 384 is below **δ = +0.10**.
     **The clause fires** (§8).
  6. **B-UNDER** — as B-FAIL, but that upper bound is ≥ δ. Escalate; no clause.
- **Reported with every row:**
  - each seed's count with its exact 95 % interval;
  - ik against M-proj;
  - per-object results;
  - grasp contact, maximum rise, and subgoals reached or timed out;
  - stopped episodes and planning seconds;
  - TASK-091 §5.3's three deferral measures per seed;
  - the share of grasp-phase first commands whose applied channel 13 differs from the requested
    one (0 by construction for M-proj; any non-zero value is a B-VOID check);
  - the share of chosen plan steps off the training support, and for M-proj the share of
    *requests* beyond the limit;
  - Stage P's results beside each seed;
  - two MP4 replays (one success if any, one failure).
- **"LeWM needed" is not claimed by any row.** The IK follower uses the same goals and no world
  model.
- **B-PASS admits a Phase 4 preregistration** (pick-and-place with subgoals). It does not
  preregister it.

### 6.4 Power (development; Stage 0 recomputes it)

These figures use TASK-090's simulation (`grasp_corpus_wm.power`: independent per-arm, per-reset
N(0, SD) logit effects shared by an arm's three seeds, and the exact sign-flip test at p < 0.01).
They are development figures, 4 000 simulations each, CPU only (§12).

| Per-seed rates, M-proj against M-v1 | n | SD 0 | SD 1 | SD 2 |
| --- | --- | --- | --- | --- |
| **0.15 vs 0.05 (declared alternative)** | **128** | 0.988 | **0.970** | 0.913 |
| 0.12 vs 0.05 | 128 | 0.861 | 0.787 | 0.664 |
| 0.10 vs 0.05 | 128 | 0.567 | 0.492 | 0.393 |
| 0.20 vs 0.05 | 128 | 1.000 | 1.000 | 0.998 |
| 0.15 vs 0.08 | 128 | 0.722 | 0.630 | 0.510 |
| 0.10 vs 0.02 | 128 | 0.992 | 0.986 | 0.950 |
| 0.10 vs 0.05 | 192 | 0.793 | 0.721 | 0.581 |
| 0.12 vs 0.05 | 192 | 0.970 | 0.943 | 0.859 |
| 0.10 vs 0.05 | 256 | 0.913 | 0.864 | 0.742 |

- **Declared alternative: 0.15 against 0.05 per seed,** as in TASK-092. The control rate 0.05 is about the pool of
  every M-v1 record in §1.1 (18/384); TASK-089 BS alone pools to about 0.08.
- **What n = 128 can detect.**
  - At SD 1 it detects a tripling, 0.05 → 0.15, with power 0.97.
  - It detects about 0.05 → 0.12 with power 0.79. That is the smallest effect it detects at about
    0.80.
  - It **cannot** reliably detect a doubling, 0.05 → 0.10 (power 0.49).
  - If the control's rate is 0.08 (TASK-089 BS's three seeds pooled, 15/192), 0.15 is detected
    with power only 0.63.
- **Why not more resets.** n = 256 would detect 0.05 → 0.10 with power 0.86, at about twice
  BS's cost (about 25 GPU h). At n = 128 a true doubling to 0.10 is missed about half the time, and
  then usually ends in B-FAIL, so **the clause can fire on a real doubling** (about 0.48 in a
  development row simulation, SD 1, 400 simulations; Stage 0 recomputes it with 20 000). This is
  accepted by declaration: a per-seed 0.10 is a quarter of the plan's 0.40 bar and about a sixth
  of ik's 82/128, and δ = +0.10 declares it below the gain this line is kept open for. So n = 128
  stands.
- **The clause risk at true gains below δ** (TASK-090's `b_row`, SD 1, n = 128, 400 simulations per
  cell; development only, from the #220 review's `rows.py`, archived in
  `~/develop/emai/evidence/task093-draft/`). **δ = +0.10 is the smallest pooled gain per seed worth
  keeping this line open for**, given that ik scores 82 / 128 (TASK-091 BS; one run, simulation
  only). A true gain below it can end in B-FAIL, and then the clause fires:

  | Per-seed rates, M-proj against M-v1 | n | B-GAIN | B-FAIL (clause) | B-UNDER |
  | --- | --- | --- | --- | --- |
  | 0.10 vs 0.05 | 128 | 0.52 | **0.48** | 0.01 |
  | 0.10 vs 0.05 | 256 | 0.86 | 0.14 | — |
  | 0.12 vs 0.05 | 128 | 0.78 | **0.21** | 0.02 |
  | 0.15 vs 0.08 | 128 | 0.64 | **0.26** | 0.11 |
  | 0.05 vs 0.05 | 128 | 0.01 | 0.99 | — |

  Stage 0 recomputes these row rates (§14).
- **Honest reading.** A B-FAIL at n = 128 rules out, at the bootstrap's 95 % level, only a pooled
  gain of +0.10 or more per seed. Any smaller gain, up to its recorded upper bound, stays possible.
  A true gain below about +0.07 per seed is detected with power below 0.80.
- **Stage 0** recomputes the table with 20 000 simulations. If the power at the declared
  alternative and SD 1 is below 0.80 at n = 128, it discloses this and escalates before the
  freeze.

## 7. Rows of the task, in order

| Row | When | Next step |
| --- | --- | --- |
| S0-ESCALATE | Stage 0: a degeneracy check of §6.1 fails, the projector fails its equivalence tests (§14), or the power rule of §6.4 fails | a ruling before the freeze; nothing is gated yet |
| P-VOID | a Stage P check fails | fix under its own ruling; fresh roots |
| P-FAIL | two or more M-proj seeds fail the Stage P gate | escalate; no clause |
| BD-ESCALATE | ik's BD rate r < 0.30 | escalate; no clause |
| B-VOID … B-UNDER | §6.3 | as §6.3 |

- After any VOID the stage runs again only on fresh seeds named by its ruling. A voided cohort is
  never re-run.
- Escalations go to a ruling in `docs/DECISIONS.md`. Each closes the task with a recommendation,
  unless the ruling says otherwise.

## 8. Abandonment clause

**Fires on B-FAIL only.**

**It closes:** grasp-channel rate projection of CEM's candidates as the grasp fix, under TASK-089's
step-averaged planner. That means projecting each candidate's right-grasp channel onto the
embodiment's rate set before the model scores it, over TASK-087's G checkpoints (TASK-087's recipe,
latent and `play-v1`) and, for the reason in §3 rank 4, over TASK-092's G-req checkpoints. After it
fires, no further variant of it is preregistered for grasping over these checkpoints without new
evidence of a different kind. Variants include:

- the refit space (requests or projected); the projected refit is closed without being tested;
- the elite count or sample count under projection;
- a projection of only the first steps;
- a softened or margin-shifted rate set;
- adding G-req.

**Not closed:**

- LeWM, DINOv2 and CEM in general;
- full-embodiment candidate projection that includes the arm channels: it stays under R26.33 for
  TASK-087's checkpoints and needs its own ruling;
- every model change of TASK-092 §2 not yet tested: a grasp-outcome head and cost, a hand-crop or
  higher-resolution latent, a weighted loss, and any other model change, including models on
  other corpora;
- the corpora (`play-v1`, play-v2g, play-v1p) as bases, and the kinds of data R27.71 leaves open;
- reach planning (TASK-088's Z3-REACH stands);
- Phase 4 as a goal, and the product goal.

**G-req is closed here on an argument, not a test:** under projection every query is on support,
where Stage R's R2 (recorded commands only) found G-req within δ of G-v1 (§3 rank 4). No arm runs
G-req with the projector.

**It does not fire on** S0-ESCALATE, P-VOID, P-FAIL, BD-ESCALATE, B-VOID, B-GAIN, B-BAR or
B-UNDER. M-v1's counts are recorded with the clause. They are not a way around R26.33.

## 9. Seeds and salts

| Use | Seeds |
| --- | --- |
| Model checkpoints | TASK-087's **G-87100, G-87101, G-87102** (no training) |
| Stage 0 calibration roots (development) | **916000–916063** |
| Stage P roots (gated) | **917000–917063** |
| Stage BD resets (development) | **91900–91931** |
| Stage BS resets (gated) | **918000–918127** (two jobs: 918000–918063 and 918064–918127) |
| Determinism re-run | 918000–918003 (inside BS) |
| Debug (Stage 0 plumbing and smoke; never counted) | **91940–91999** |

**Salts.**

- **New:** **9011**, the random reference of §6.1.
- **Reused:** TASK-088's 8801 and 8802 (layouts and goals), 8803 (CEM, through `cem_seed`) and
  8804 (random), with the new seeds; TASK-090's 9006 (the reset-cluster bootstrap).
- **Rejected draws:** a root's or reset's rejected draws are further attempts under the same seed.

**Checked (DRAFT, 2026-10-14).**

- **Remote branches.** On all 114 remote-tracking refs (after `git fetch --all`), over `src/`,
  `scripts/`, `docs/`, `.mc/`, `tests/`, `benchmarks/` and `configs/`, `git grep -w` finds no
  token in 916000–918999, in 91900–91999 or for the salt 9011.
  - 91412 (a `branch` field in two apple manifests) is why 91400–91599 was not used.
  - 91797–91839 occur as `rows_added` counts in two manifests, which is why 91700–91899 was not
    used.
  - 9010 occurs only inside `9010{0,1,2}` (TASK-090's model seeds), so it was skipped as a salt.
- **Evidence.** In `~/develop/emai/evidence/` (`*.json`, `*.txt`, `*.log`, `*.md`, `*.csv`,
  `*.yaml`, `*.py`, `*.sh`):
  - tokens in 916000–918999 occur only as digits of decimal fractions (for example `0.916039`),
    not as seeds;
  - tokens in 91900–91999 are decimal fractions (for example `11.91914`), except 91900, which
    occurs as a training-update index in TASK-077's Stage T loss curves
    (`task077-staget2/…/report.json`). None is a seed or reset.
- **Nearest ranges in use:**
  - TASK-092: 908000–908063 (A‴), 909000–909127 (BS, unused), 90400–90431 (BD, unused),
    90500–90599 (debug), 90810–90841 (D-RATE, spent) and 90850–90881 (D-RATE′);
  - TASK-091: 906000–906063, 907000–907127, 90300–90331 and 90700–90799;
  - TASK-090: 903000–905063, 90200–90231, 910000–913199 and 920000–923199.
- **Stage 0** repeats both checks at its own revision and records them.

## 10. Runtime, disk, paths and evidence

**GPU.** Every GPU use is one `scripts/gpu_run.sh --wait --min-free-gib 8 --board` job, under the
shared lock per job and the 7 200 s per-job cap.

- Before each job the `## Requests` section of `~/.local/state/gpu/BOARD.md` is checked; a
  `- v2d: …` line means the job waits.
- The V2D session is told in advance about BD and BS.
- The GR00T server, `/mnt/data/emai/{archive,models,cache}` and other worktrees are not touched.

| Step | Estimate | Basis |
| --- | --- | --- |
| Stage 0 (calibration roots and six CEM calls per root, references, equivalence tests, debug smoke, timing) | about 0.6 GPU h | TASK-092 Stage 0 0.39 h; A‴ 457 s for six models on 64 roots |
| Stage P (64 roots, six models) | about 0.15 h | A‴ |
| Stage BD (2 world-model runs × 32 resets, plus baselines) | about 1.2 h | D-RATE′ 2 155–2 237 s per run; TASK-091 BD |
| Stage BS (6 world-model runs × 128 resets as 12 jobs, plus baselines and the determinism re-run) | about 11–14 GPU h; at most about 16 at the full 400-command budget | TASK-091 BS: 3 184–4 218 s per 64-reset job; TASK-092 Stage 0's projection 4 786 s |
| **Total** | **about 13–16 GPU h** (ceiling about 18); about 4–6 days wall-clock with reviews | |

The projector adds NumPy work on 300 × 8 values per CEM iteration. Stage 0 times it. If a 64-reset
BS job projects above 5 400 s, Stage 0 may split BS into more, smaller jobs (§14).

**Disk.** No new corpus, feature store or checkpoint. Goals, roots, runs and MP4s go to
`outputs/task093-*` on the data disk through the repository's links.

**Paths that must stay.** TASK-087's feature store and projection in
`worktrees/task087-run/outputs/`, and `checkpoints/task087-run/`. **`worktrees/task087-run` must
not be archived or removed until TASK-093 closes.**

**Evidence.** `~/develop/emai/evidence/task093-*/` with `SHA256SUMS`. Existing evidence under
`data/`, `checkpoints/`, `outputs/` and `~/develop/emai/evidence/task08*` and `task09[012]-*` is
read and never written.

## 11. What this cannot show, and its risks

- **Scope.** One scene, arm, hand and camera at 112 px, simulation only. One encoder, latent,
  corpus, recipe, budget and cost. Goals come from a scripted demonstrator (biased to layouts a
  scripted pick can do; almost no banana). Seeds are pooled.
- **Near the floor.** The control's recorded rate is about 0.01 for 87100 and about 0.11–0.125 for
  87101 and 87102 (§1.1), and ik's is
  about 0.64.
  - Even a large relative gain may leave M-proj far below the 0.40 bar. That is B-GAIN, not
    B-PASS.
  - n = 128 does not reliably detect (power below 0.80) a gain below about +0.07 per seed, and
    B-FAIL can fire at such a gain (§6.4).
- **The mechanism is a hypothesis.** A B-GAIN or B-PASS would be consistent with "the planner
  scored off-support requests". A B-FAIL would say that removing the grasp-rate mismatch on the
  planner side does not, on its own, move the closed loop. The cause would then lie elsewhere:
  the arm channels, contact, the latent's resolution or the cost.
- **Grasp channel only.** The arm channels are still executed through backtracking and the IK.
  The planner does not score that. The records cannot show how often the arm requests were
  reduced, because `plan_log` keeps the grasp only. The contact gap (TASK-091: 39–87 against 122
  of 128) may come from the arm.
- **The refit is on requests.** The CEM distribution is refitted on the requested values, as the
  existing code does. Many requests map to one projected grasp, so the distribution can drift
  towards saturated requests without changing what is scored. This is the existing planner's behaviour, so it is not changed. It is CEM over
  u ↦ cost(P(u)), a valid optimisation of the projected cost; refitting on projected values would
  need a planner subclass, a second change. P-MECH and the request-level off-support share are reported, so
  a reader can see it.
- **A copied limit.** The projector re-implements the grasp half of `_prepare_side` instead of
  calling the embodiment, because full projection is too slow (§3 rank 3) and `embodiment.py` is
  frozen. This is a deliberate exception to "the embodiment owns command limits".
  - Stage 0's equivalence tests (§14) bound the risk: they compare against the embodiment's own
    `project_candidates` on random sequences and against executed commands in closed loop.
  - Any later change to the embodiment's grasp limit would invalidate the copy, and the frozen-file
    pins detect that.
- **Stage P's bars are no-regression bars** calibrated from the control, not from closed-loop
  success. Offline plan measures have not predicted the closed loop before (TASK-089: A1 and A2
  passed, then 0 / 64). Stage P only stops a broken planner from spending BS's budget.
- **What a pass shows.** A B-PASS would be the first gated world-model grasp-and-lift at the plan's
  bar in this project, on these terms only. It would not show "LeWM needed".

## 12. Disclosed development before this draft

All of it is development and **not gated**: CPU only, at `1be28a1`. No GPU job, simulation,
rendering, model load or training was run for this draft.

- **Code reading.** `embodiment.py` `_prepare_side` (the grasp-rate formula) and
  `project_candidates`; `planning.py` `CEMPlanner.plan` (the existing `projector` hook, the refit
  on requests and the returned projected plan); `zero_shot_runtime.py` and
  `zero_shot_grasp_runtime.py` (TASK-089's controller and step-averaged model).
- **Power.** The table of §6.4 comes from `grasp_corpus_wm.power` (TASK-090's code, unchanged)
  with 4 000 simulations per cell. The script and its output are archived in
  `~/develop/emai/evidence/task093-draft/` (`power093.py`, `power093.json`, and the #220 review's row simulation `rows.py`; `SHA256SUMS` sha256
  `0a9d4306…ad81`).
- **Seed checks.** §9's checks.
- **The numbers of §1.** These are quoted from merged records (TASK-089, TASK-091, TASK-092 §1.2,
  §15–§19). None was recomputed.

These choices set the candidate, the measures and the declared alternative. They set no bar: the
Stage P bars come from Stage 0's development roots (§6.1), and the closed-loop bar comes from ik's
BD rate.

## 13. Order

1. This DRAFT, reviewed and merged.
2. A Stage 0 PR (§14), reviewed. **The protocol is frozen at its merge.**
3. Stage P, then Stage BD, at the frozen revision from a clean tree with
   `scripts/run_task093_stage.sh`. Each record is a PR with a DECISIONS entry.
4. An independent GO on BD's record.
5. Stage BS, then its record PR.

## 14. Stage 0, freeze and what Stage 0 may change

**Stage 0** is its own PR, merged before the freeze. It adds only opt-in modules and new scripts,
and changes no existing file except CLAUDE.md's entry points and this record. So `planning.py`,
`embodiment.py`, TASK-087's implementation hash and TASK-088–TASK-092's frozen code stay as they
are.

- **A new module** (NumPy at import), for example `grasp_rate_projected.py`:
  - the projector of §4.1;
  - Stage P's measures, references, bar rule, degeneracy checks and rows;
  - the B rows (reusing TASK-090's sign-flip test, bootstrap and power code);
  - the cohorts and seeds;
  - sha256 pins of every earlier file it imports.
- **A torch module**, for example `grasp_rate_projected_runtime.py`:
  - a controller subclassing TASK-091's `RecordingController` (itself TASK-089's
    `ChangeController`, so `plan_log` and the deferral measures are kept), which builds the
    `CEMPlanner` with `project_candidates=True`, reads T₀ from the embodiment at each command and
    passes the projector;
  - Stage P's per-root plans for both arms;
  - the closed loop through TASK-091's frozen runtime.
- **Scripts:** `scripts/{run,summarize}_task093.py` and `scripts/run_task093_stage.sh`.
- **Tests:**
  - **Projector against the embodiment:** on random request sequences from debug states, with zero arm deltas and
    with non-zero arm deltas, the projector's grasp channel equals `robot.project_candidates`' grasp
    channel at every step, bitwise in float32 (with non-zero deltas, on the sequences the embodiment
    marks feasible).
  - **Projector in closed loop:** on the debug smoke, for every M-proj command, the applied action returned by `zs.execute` has
    channel 13 bitwise equal to the request's channel 13. The embodiment's reason string is not
    used, because `zs.execute` pre-projects the request, so `robot.execute` never reports a grasp
    clip. A negative control: on the same smoke, M-v1 shows at least one command whose channel 13
    differs.
  - **Pairing:** with an identity projector, `plan` returns bitwise the unprojected planner's plan
    for the same seed, and the first-iteration samples of both arms are identical.
  - The cumulative clip on synthetic sequences.
  - The bar rule and the degeneracy checks on synthetic counts.
  - The rows, the seed ranges and their disjointness, and the frozen-file pins.
  - The core-import check.

**Stage 0 also runs:**

- the development calibration of §6.1 on 916000–916063, recording each measure's counts for the
  control, the projected planner and the two references, the bars and the degeneracy checks;
- §9's seed checks and the hash checks (TASK-087's checkpoints, feature report and projection,
  λ, the rate limit);
- the power simulation of §6.4 with 20 000 simulations;
- the row distribution (B-GAIN, B-FAIL, B-UNDER) at 0.10, 0.12 and 0.15 against 0.05 and at 0.15
  against 0.08, n = 128, SD 0/1/2, with TASK-090's `b_row`, 20 000 simulations;
- a debug smoke on 91940–91999 of Stage P and every closed-loop arm, timed per command; it also
  confirms that passing `CandidateProjection`'s read-only arrays to `torch.as_tensor` has no
  behavioural effect.

**Stage 0 may change only:**

- the per-run process count;
- the BS job split (more, smaller jobs) if a 64-reset job projects above 5 400 s;
- the bar values that §6.1's rule computes (the rule itself is fixed).

Each change is recorded. A failed degeneracy, equivalence or power check is S0-ESCALATE, not a
change. The protocol is frozen at the Stage 0 merge.

## 15. Rulings this document records (R27.115–R27.131; DRAFT; decided by Claude under owner delegation)

- **R27.115** — **R26.33's condition ("new evidence of a different kind") is met for exactly one
  planner variant**: grasp-channel rate projection of CEM's candidates over TASK-087's
  checkpoints (§2). No other planner, cost, horizon or action-parametrisation variant is admitted.
- **R27.116** — R27.71, TASK-092 §8, TASK-062, TASK-065 and TASK-054 do not cover it (§2).
- **R27.117** — **TASK-093 is that variant.** The ranking of §3 is recorded: tie-breaking,
  full-embodiment projection, projection plus G-req, and projection over TASK-090's checkpoints are
  not taken.
- **R27.118** — **The projector** (§4.1): a cumulative clip of the right-grasp channel from the
  embodiment's last accepted hand targets, with `_prepare_side`'s per-joint formula. It is used
  through `CEMPlanner`'s existing hook, with the refit on requests and the projected plan executed
  (§4.2). `planning.py` and `embodiment.py` are not edited.
- **R27.119** — **Everything else is frozen** (§4.3): TASK-087's G-87100–87102, the latent, the
  step-averaged cost, λ = 2.0552, the CEM budget, the goals, the budget of 400 commands and the
  scorer.
- **R27.120** — **Arms** (§5): M-proj ×3 is primary and pooled. M-v1 ×3 (unprojected) is the gated
  control and a control re-run. ik, ik-nopress, hold and random keep their roles. G-req is not an
  arm.
- **R27.121** — **Stage 0 calibrates Stage P** on development roots 916000–916063 (§6.1). It sets
  the bars by the declared rule min(48, ⌈0.75 × the control's median count⌉) and runs the
  degeneracy checks (a)–(c). A failed check is S0-ESCALATE.
- **R27.122** — **Stage P** (§6.2): C8@R1 and H8@R2 in executed terms on fresh roots
  917000–917063. P-FAIL needs two or more failing seeds. P-MECH is reported only.
- **R27.123** — **The closed loop** (§6.3):
  - BD on 91900–91931, then an independent GO;
  - BS with n = 128 on 918000–918127;
  - the bar min(0.40, 0.9 r), the pooled bar and the per-seed floor;
  - the reset-level sign-flip tests;
  - rows B-VOID, B-PASS, B-GAIN, B-BAR, B-FAIL (δ = +0.10) and B-UNDER.
- **R27.124** — **Power** (§6.4): the declared alternative is 0.15 against 0.05 per seed, with
  power 0.97 at SD 1 and n = 128 (development, 4 000 simulations). The smallest effect detected at
  about 0.80 is 0.05 → 0.12, and 0.05 → 0.10 is not reliably detected. n = 128 stands by declaration: a true per-seed 0.10 is below δ, and at n = 128 it ends in
  B-FAIL (the clause fires) about half the time (development row simulation); Stage 0 recomputes
  this. δ = +0.10 is the smallest pooled gain worth keeping the line open for, given ik's 82 / 128;
  the clause risk below δ is disclosed (0.48 at 0.05 → 0.10, 0.21 at 0.12 vs 0.05, 0.26 at 0.15 vs
  0.08; SD 1, 400 simulations per cell, development only).
- **R27.125** — **Rows** (§7) and their order.
- **R27.126** — **The abandonment clause** (§8) fires on B-FAIL only, with the scope stated there.
- **R27.127** — **Seeds and salts** (§9), checked on 114 remote-tracking refs and in the evidence
  (where the tokens in these ranges are decimal-fraction digits or, for 91900, a TASK-077
  training-update index; none is a seed).
  The new salt is 9011.
- **R27.128** — **Budget** (§10): about 13–16 GPU h, ceiling about 18. Every job runs under the
  7 200 s cap and the shared lock. `worktrees/task087-run` stays until TASK-093 closes.
- **R27.129** — **Disclosed development** (§12) is recorded.
- **R27.130** — **Status.** DRAFT in this PR. A Stage 0 PR freezes it at its merge. Gated stages
  run only after that, and BS only after an independent GO on BD's record.
- **R27.131** — **R7 does not change**, whatever row TASK-093 ends in. TASK-093 is not an
  Apple→Plate experiment.

## 16. Stage 0 record (R27.132–R27.138; decided by Claude under owner delegation)

Everything below is Stage 0: checks, the development calibration on cal-roots 916000–916063 and a
debug smoke on debug seeds 91940–91999. No gated stage ran; roots 917000–917063 and resets
91900–91931 and 918000–918127 were not touched. Nothing here is evidence of any outcome. Evidence:
`~/develop/emai/evidence/task093-stage0/` (`SHA256SUMS` sha256 `3e64082f…7adc`); large
artefacts (root frames, goal frames) in `outputs/task093-stage0/` on the data disk.

**Revisions** (from the reports and `gpu_run.log`; every report records `tracked_tree_dirty:
false`, and every job ran in the worktree root, where `third_party/le-wm` is):

- `b7af860` (the code): `verify`, `seeds`, `power` and `rows` (CPU).
- `c18bdae` (a debug-only flag added to `plans`): `equivalence`, the debug roots and the first
  debug `plans` job, which exited VOID on its rate-set plumbing check (below).
- `c7cf7f4` (the rate-set check corrected): the second debug `plans` job, `smoke-checks`, and the
  **calibration** (cal-roots `roots` and `plans`).
- `0055acd` (Stage P's bars written into `grasp_rate_projected.P_BARS`): the debug Stage P smoke,
  every debug closed-loop run, the determinism re-run and the opt-in render tests.
- The record commit changes documentation only.
- The review fixes (after every Stage 0 run; no Stage 0 number depends on them) reorder
  `cmd_plans` so the ledger claim comes after `load_roots` and after the model loads, add TASK-087's
  DINOv2 encoder (with its digest check) to the preflight of `plans` and `run`, let
  `run_task093_stage.sh` summarise a stage whose job exited VOID, and correct a comment and a
  docstring. The Stage 0 reports record the earlier module and runner hashes.

### 16.1 Code

- `src/embodied_jepa/grasp_rate_projected.py` (NumPy at import): the projector of §4.1
  (`RateGeometry`, `project_grasp`, `GraspRateProjector`: `_prepare_side`'s per-joint float64
  bounds, its float64 `np.clip`, the synergy line and `_joint_commands`' range check, float32 cast
  and command-limit clip, vectorised over candidates; T₀ read from the embodiment's
  `_projection_targets` and checked against its live targets), Stage P's measures, references
  (random with salt 9011), bar rule, degeneracy checks and rows, the S0 row, the B rows (TASK-090's
  `b_row` logic with B-COMP named B-GAIN and B-SIZE named B-BAR; a test checks them against
  TASK-090's), the closed-loop equivalence and request measures, the power and row-rate recompute,
  the cohorts and seeds, and sha256 pins of every earlier file TASK-093 imports: TASK-092's pins
  (TASK-087–TASK-091), TASK-092's modules and scripts, and `embodiment.py`,
  `configs/g1_sim_action.json`, `constraints.py`, `contracts.py`, `play_corpus.py` and
  `simulation.py`. Every runner refuses to start when one of them differs.
- `src/embodied_jepa/grasp_rate_projected_runtime.py` (torch): `projected_planner`
  (`zgr.make_planner`'s configuration with only `project_candidates=True`), Stage P's per-root plan
  for both arms with the first-iteration samples' sha256, P-MECH's predicted synergy, and
  `ProjectedController`, TASK-091's `RecordingController` whose planner is the projected one (T₀
  read at every command; `plan_log` also keeps the requested grasp).
- `scripts/run_task093.py` (`verify`, `seeds`, `power`, `rows`, `equivalence`, `roots`, `plans`,
  `smoke-checks`, `goals`, `run`, `combine`, `video`), `scripts/summarize_task093.py` (stages 0, p,
  bd, bs, timing) and `scripts/run_task093_stage.sh` (stages p, bd, bs). Every job refuses another
  working directory or a missing `third_party/le-wm` (R27.107's lesson). Before any ledger claim,
  `plans` loads and checks its roots and `run` its goals, and both load their G checkpoints and
  TASK-087's DINOv2 encoder with its projection (preflight, on the CPU); a CPU test fails if a claim
  comes first. A job that exits VOID does not stop `run_task093_stage.sh` before its summary; the
  stage then exits VOID itself.
- `tests/test_grasp_rate_projected.py`, 27 tests: the projector against `_prepare_side` bitwise with
  zero and non-zero arm deltas (no rendering), against `robot.project_candidates` bitwise (graphics
  opt-in), the cumulative clip on synthetic sequences, the hook contract, the empty interval, the
  identity-projector pairing and the shared first-iteration samples, the runtime planner's
  configuration, the measures and references, the bar rule and degeneracy checks on synthetic
  counts, the frozen bars, the P and S0 rows, the B rows against TASK-090's, the closed-loop
  equivalence measure (bitwise), the request measure, the power rule and row-rate plumbing, the seed
  ranges and their disjointness with earlier tasks, the admitted runs, the frozen-file pins, the
  core-import check, the working-directory guard, the ledger claim's place after the roots or goals
  and the model loads, and the summarizer's refusals. The whole suite at the review head: 2 457
  passed, 34 skipped (graphics opt-in, MPS unavailable and the LeRobot-source opt-in; 2 455 and 34
  at `0055acd`); with `JEPA_TEST_RENDER=1` this file's 25 tests of `0055acd` passed (the two added
  after review are CPU-only).
- No existing file changes except CLAUDE.md's entry points and this record.

### 16.2 Checks

| Check | Result |
| --- | --- |
| **Hashes** (`verify`) | TASK-087's G-87100–87102 against their pins, reports and evidence copies (kind `job`, T-JOB-DONE); the feature report and every train/val store file; the projection; the pinned moments; TASK-087's implementation hash; λ = 2.0552; the rate limit 2 / 11; CEM (8, 300, 10, 30, 0.05); every frozen file: **all pass** |
| **Seeds** (`seeds`, 116 remote-tracking refs and HEAD) | no token in 916000–918999, 91900–91999 or 9011 outside TASK-093's own files. In the evidence (15 tokens): 91900 six times as a TASK-077 training-update index, and 91907 and 916172–918979 as frame offsets in TASK-087's episode table (`task087-stagef/episodes_train.json`); none is a seed |
| **Projector against `robot.project_candidates`** (`equivalence`; 4 debug layouts 91940–91943, 40 states with T₀'s grasp from −1 to +1, 24 random sequences of 8 per state, zero and N(0, 0.5) right-arm deltas) | 15 360 grasp steps compared, 12 792 of them clipped, **0 mismatches** (bitwise float32); every other channel bitwise the request; all 960 non-zero-delta sequences feasible after backtracking |
| **Projector against `_prepare_side`** (tests, no rendering) | bitwise on both parametrisations |
| **Read-only arrays** (`smoke-checks`, G-87100, 4 debug roots × R1/R2) | `torch.as_tensor` on `CandidateProjection`'s read-only arrays warns once and gives **bitwise** the costs of a writable copy |
| **Pairing on the real model** (`smoke-checks`) | with an identity projector the projected planner returns **bitwise** the unprojected plan and costs (8 of 8 calls) |
| **Pairing in the calibration** (`plans`, 384 root calls) | M-proj and M-v1 drew identical first-iteration samples (sha256) on every call; every M-v1 plan equals `pr.a2_plan`'s bitwise; every M-proj plan is in the rate set |
| **Closed-loop equivalence** (debug smoke) | M-proj-87100: applied channel 13 bitwise the request on **all 2 400** commands (600 in the grasp phase) and on all 1 600 of the determinism re-run. Negative control M-v1-87100: **1 454 of 2 206** commands differ (454 of 507 in the grasp phase) |
| **Projector cost** | 3.5 ms per command (10 CEM iterations of 300 × 8), negligible against about 0.5–1.1 s per command |

**The first debug `plans` job exited VOID** on its rate-set plumbing check (debug root 91943, G-87102,
R2). The check as first coded also required a stored projected plan to be bitwise its own
re-projection. That is not exact: the plan stores float32(g_k) while the cumulative clip carries
the float64 g_k, so a re-projection can differ by one float32 ulp (here 1.2 × 10⁻⁷ at step 8). The
§6.2 check is the |Δg| ≤ 2 / 11 + 10⁻⁶ bound only, which that plan met (largest step 0.18181831).
`c7cf7f4` restores the check to §6.2's wording. The closed-loop equivalence is unaffected: each
command's first step is clipped from the same T₀ by the same formula, as the smoke confirms.

### 16.3 Calibration of Stage P on cal-roots 916000–916063 (§6.1; GPU, at `c7cf7f4`, clean tree)

64 roots (apple 19, can 21, cube 23, banana 1; 71 rejected draws). Counts of 64; executed terms:

| Planner | G-87100 | G-87101 | G-87102 |
| --- | --- | --- | --- |
| M-v1 (control), C8@R1 | 33 | 58 | 51 |
| M-v1 (control), H8@R2 | 64 | 64 | 61 |
| M-proj (reported), C8@R1 | 36 | 53 | 47 |
| M-proj (reported), H8@R2 | 57 | 61 | 55 |

| Reference | C8@R1 | H8@R2 |
| --- | --- | --- |
| open / release | 0 | 0 |
| closed, vertical −1 | — | 0 |
| random (salt 9011) | 0 | 6 |
| maximum rate, vertical 0 (reported, not a §6.1 reference) | 64 | 0 |

- **Bars** (the rule of §6.1): **bar_C8 = min(48, ⌈0.75 × 51⌉) = 39**, **bar_H8 = min(48,
  ⌈0.75 × 64⌉) = 48**.
- **Degeneracy checks: all pass.** (a) open/release 0 / 64 on both, and closed-vertical-−1 0 / 64
  on H8. (b) random 0 ≤ 19 on C8 and 6 ≤ 24 on H8. (c) the control's median 51 and 64, both
  ≥ 16.
- **Executed path.** §6.1 calls a projected plan's executed path "the plan itself"; the code
  re-projects the stored plan, which can differ from it by about one float32 ulp (about 10⁻⁷,
  §16.2). C8's 10⁻⁶ slack absorbs this; H8's > 0.5 test could change only for a path value within
  about 10⁻⁷ of 0.5.
- **G-87102's checkpoint has `last_two_triggered: true`** (TASK-087's training flag, recorded in
  `plans-cal-roots.json`); G-87100's and G-87101's do not.
- **Reported, development only** (not a gate): on these roots M-proj-87100's C8 (36) is below
  bar_C8, and the other five M-proj counts are at or above their bars. So a Stage P seed failing
  alone (which does not make P-FAIL) is plausible on the evidence of these roots.
- **P-MECH (reported)**, mean |predicted − executed hand synergy| over steps 1–8 at R1: M-proj
  0.285, 0.292 and 0.282; M-v1 0.461, 0.517 and 0.411. At step 8 the mean executed grasp is
  0.37–0.43 under M-proj's plans and 0.34–0.45 under M-v1's, while G predicts 0.07–0.11 under
  M-proj's plans and 0.59–0.64 under M-v1's. So (development roots, reported only) under
  projection G predicts a hand that closes more *slowly* than executed, where
  under its own unprojected plans it predicts one that closes faster.
- **M-proj's requests (reported):** 490–503 of 512 requested grasp steps at R1 and 173–226 of 512
  at R2 lie beyond the limit from the executed path. The refit on requests drifts towards
  saturated requests, as §11 discloses; what is scored and executed is the projected path.
- In TASK-089's request terms (the first command), close@R1 is 0 for M-proj (its first projected
  grasp from the open hand is −0.818 by construction) and 62–64 for M-v1; lift@R2 is 64 for M-proj
  and 36–61 for M-v1.

### 16.4 Power and row rates (§6.4, 20 000 simulations each, CPU, at `b7af860`)

Power of "M-proj beats M-v1" (reset-level exact sign-flip test, p < 0.01):

| Per-seed rates | n | SD 0 | SD 1 | SD 2 |
| --- | --- | --- | --- | --- |
| **0.15 vs 0.05 (declared)** | **128** | 0.988 | **0.976** | 0.915 |
| 0.12 vs 0.05 | 128 | 0.857 | 0.797 | 0.655 |
| 0.10 vs 0.05 | 128 | 0.572 | 0.502 | 0.383 |
| 0.20 vs 0.05 | 128 | 1.000 | 1.000 | 0.997 |
| 0.15 vs 0.08 | 128 | 0.726 | 0.642 | 0.501 |
| 0.10 vs 0.02 | 128 | 0.993 | 0.987 | 0.949 |
| 0.10 vs 0.05 | 192 | 0.788 | 0.723 | 0.584 |
| 0.12 vs 0.05 | 192 | 0.972 | 0.946 | 0.862 |
| 0.10 vs 0.05 | 256 | 0.908 | 0.861 | 0.738 |

The power at the declared alternative and SD 1 is 0.976 ≥ 0.80, so **n = 128 stands** and the
power rule does not escalate. The DRAFT's 4 000-simulation figures are reproduced within
simulation error.

B-row rates at n = 128, bar 0.40, random at 0 successes (TASK-090's `b_row` through
`grasp_rate_projected.row_rates`; B-PASS and B-BAR never occurred at these rates):

| Per-seed rates | SD | B-GAIN | B-FAIL (clause) | B-UNDER |
| --- | --- | --- | --- | --- |
| 0.10 vs 0.05 | 0 / 1 / 2 | 0.563 / 0.506 / 0.380 | 0.437 / **0.486** / 0.524 | 0.000 / 0.008 / 0.097 |
| 0.12 vs 0.05 | 0 / 1 / 2 | 0.855 / 0.795 / 0.659 | 0.144 / **0.190** / 0.221 | 0.001 / 0.016 / 0.121 |
| 0.15 vs 0.05 | 0 / 1 / 2 | 0.989 / 0.972 / 0.911 | 0.011 / **0.021** / 0.030 | 0.001 / 0.007 / 0.059 |
| 0.15 vs 0.08 | 0 / 1 / 2 | 0.730 / 0.635 / 0.499 | 0.226 / **0.229** / 0.192 | 0.045 / 0.136 / 0.310 |

At SD 1 the clause risk below δ is 0.49 at 0.05 → 0.10, 0.19 at 0.12 vs 0.05 and 0.23 at 0.15 vs
0.08 (the DRAFT's 400-simulation 0.48, 0.21 and 0.26). R27.124's disclosure stands with these
values; nothing escalates.

### 16.5 Debug smoke (debug seeds only; never counted; not evidence)

- **Stage P plumbing** on 6 debug roots (91940–91945, 1 rejected draw). The evidence holds two
  debug Stage P summaries, both P-VOID and both plumbing only:
  - `debug/summary-p-smoke.json` read the second debug `plans` file (at `c7cf7f4`, run before the
    bars were frozen and without the gate) with roots made at `c18bdae`. It is P-VOID on the mixed
    revisions and because its bars are not the frozen ones.
  - `debug/p-smoke/summary-p-smoke.json` read the `--stage-p-smoke` run at `0055acd` with the
    frozen bars. Every check ran and the gate and row were computed. It is P-VOID only because the
    debug roots were made at an earlier revision, which is the revision check working.
- **Closed loop** on 6 debug resets (91960–91965, 3 rejected draws): ik 4 / 6, ik-nopress 4 / 6,
  hold 0 / 6, random 0 / 6, M-v1-87100 0 / 6 (one episode stopped by the joint-velocity guard),
  M-proj-87100 0 / 6 (run as two 3-reset jobs, combined). The **determinism re-run** of M-proj on
  the first four resets matched the combined run **bitwise** (applied commands and outcomes),
  across a different job split and process count. The deferral, rate, request and equivalence
  measures were recorded on every world-model episode.
- **Timing.** At 6 processes M-v1-87100 took **1.108 s per command per process**, so a 64-reset
  job projects to at most **4 873 s** at the full 400-command budget, below the 5 400 s trigger.
  M-proj ran at 3 and 4 processes (0.55 and 0.72 s per command per process; projections 2 424 and
  3 171 s); the projector adds 3.5 ms per command. So the BS split stays two 64-reset jobs.
- **GPU use:** 18 jobs, **1 944 s (0.54 GPU h)**, all under `gpu_run.sh --wait --board` after a
  BOARD.md check (no `- v2d:` request was present), one of them the 4 s opt-in render-test run. One job exited
  VOID (the first debug `plans`, above); none hit the cap. The GR00T server and
  `/mnt/data/emai/{archive,models,cache}` were not touched.

### 16.6 Projected budget

Stage P about 0.1–0.15 GPU h (the calibration's two jobs took 555 s); Stage BD about 1.2; Stage BS
about 11–14 (12 world-model jobs at most about 4 873 s each, plus baselines and the determinism
re-run). Total about 13–16 GPU h, as §10.

### 16.7 Rulings (R27.132–R27.138)

- **R27.132 — Stage 0 is recorded** (§16.1–§16.5) **and ends S0-PASS**: every degeneracy check of
  §6.1 passes, the projector is bitwise equal to the embodiment's own projection and to the
  executed commands (with M-v1 as negative control), the identity-projector pairing is bitwise,
  and the power at the declared alternative is 0.976 at SD 1.
- **R27.133 — Stage P's bars are frozen: C8@R1 ≥ 39 / 64 and H8@R2 ≥ 48 / 64**, by §6.1's rule
  from the control's development counts (C8 33, 58, 51; H8 64, 64, 61). They are in
  `grasp_rate_projected.P_BARS`, and Stage P refuses to run without them.
- **R27.134 — Stage 0 changes none of the other §14 items.** Process counts stay 8 for roots,
  goals and baselines, 6 for world-model runs and 4 for the determinism re-run. The BS split stays
  two 64-reset jobs (projection 4 873 s ≤ 5 400 s). n stays 128.
- **R27.135 — readings of points the DRAFT left open:**
  - the debug split 91940–91959 (roots) and 91960–91999 (resets);
  - Stage P's rate-set check is the |Δg| bound of §6.2 only, since a stored float32 plan is not
    bitwise its own re-projection (§16.2);
  - T₀ in closed loop is `_projection_targets` at the observation, checked equal to the live
    targets;
  - the random reference draws R1's sequence and then R2's from `default_rng([9011, root seed])`,
    N(0, 1) clipped to TASK-088's bounds;
  - the deterministic references (open/release, closed-vertical-−1, maximum-rate) keep every
    unused channel at 0 and the left grasp at −1; the random reference draws every channel from
    N(0, 1) clipped to TASK-088's bounds (left arm and left grasp fixed);
  - Stage P's plumbing checks also require every M-v1 plan to equal TASK-091's `pr.a2_plan`
    bitwise (the control is the recorded A2 planner); a mismatch is a P-VOID. It held on all 384
    calibration root calls;
  - "S0-PASS" names the absence of S0-ESCALATE;
  - the row-rate simulation puts random at 0 successes, as the DRAFT's `rows.py` did, with its own
    generator tag (93);
  - the request-terms close@R1 and lift@R2 read the plan's first command, as TASK-089 did.
- **R27.136 — the debug smoke is not evidence** and changes nothing in §2–§12. The development
  calibration counts (including M-proj's, reported) are not a Stage P result.
- **R27.137 — frozen at the Stage 0 PR's merge.** Stage P, BD and BS run at the merged revision
  from a clean tree with `scripts/run_task093_stage.sh`; BS only after an independent GO on BD's
  record. The V2D session is told before BD and BS.
- **R27.138 — R7 does not change.** TASK-093 is not an Apple→Plate experiment.

## 17. Stage P record (gated, offline; R27.139–R27.141; decided by Claude under owner delegation)

**Run conditions.** Stage P ran once on 2026-10-14 (15:41–15:46 UTC) with
`scripts/run_task093_stage.sh p outputs/task093-p ~/develop/emai/evidence/task093-stage0/summary-0.json`,
at the frozen revision `4d3491e` (#221's merge), from a clean tracked tree in
`worktrees/task093-run` (detached at `4d3491e`; the job ran in that tree's root, where
`third_party/le-wm` is; its `.venv` was synced from `uv.lock` with the same package set as the
Stage 0 worktree's, checked by `uv pip freeze`). Before each of the two jobs the `## Requests`
section of `~/.local/state/gpu/BOARD.md` had no `- v2d:` line, and each job took the shared lock
with `gpu_run.sh --wait --board`. Evidence: `~/develop/emai/evidence/task093-p/` (`SHA256SUMS`
sha256 `9446499b…e2cd`): the roots report, the plans report, the summary, the stage log, the two
ledger entries and the `gpu_run` log lines. The root frames (`roots-p-roots.npz`) stay in
`outputs/task093-p/` on the data disk.

**Jobs and GPU time.** `roots` (64 roots, 917000–917063, 62 rejected draws; targets cube 28,
apple 18, can 18) 65 s; `plans` (six models × 64 roots × R1 and R2) 216 s. In total 2 jobs,
**281 s (about 0.08 GPU h)**, no cap hit, no VOID exit.

**The summarizer's output, as `summarize_task093.py --stage p` printed it:**

```
{
 "stage": "p",
 "row": "P-PASS",
 "bar": null,
 "bars": {
  "C8": 39,
  "H8": 48
 },
 "problems": []
}
```

**Per seed** (64 fresh gated roots; executed terms; bars frozen at Stage 0, R27.133: C8@R1 ≥ 39,
H8@R2 ≥ 48; one run, offline, simulation only):

| Seed | M-proj C8@R1 | M-proj H8@R2 | Gate | M-v1 C8@R1 (reported) | M-v1 H8@R2 (reported) |
| --- | --- | --- | --- | --- | --- |
| G-87100 | **39** | 58 | pass (C8 at the bar) | 43 | 62 |
| G-87101 | 48 | 61 | pass | 55 | 64 |
| G-87102 ¹ | 54 | 58 | pass | 54 | 62 |

¹ `last_two_triggered: true` (TASK-087's training flag, recorded in `plans-p-roots.json`).

- **Why this is the row.** No M-proj seed fails the gate, so the P-FAIL rule (two or more
  failing seeds) does not apply; the summary lists no problem, so the plumbing checks of §6.2
  (every M-proj grasp step inside the rate set, identical first-iteration samples for both arms,
  the root, checkpoint, λ and frozen-file hashes, and R27.135's bitwise M-v1 = `pr.a2_plan`)
  held, and the row is not P-VOID. **P-PASS.** M-proj-87100's C8@R1 is exactly at its bar
  (39 / 64): it passes by zero roots of margin.
- **References on these roots** (reported; the degeneracy checks are Stage 0's): open/release
  C8 0, H8 0; closed, vertical −1 H8 0; random (salt 9011) C8 0, H8 3; maximum rate, vertical 0
  C8 64, H8 0.
- **P-MECH (reported)**, mean |predicted − executed hand synergy| over steps 1–8 at R1: M-proj
  0.286, 0.285 and 0.290; M-v1 0.450, 0.514 and 0.405 (G-87100, 87101, 87102). As on the
  calibration roots, under projection G predicts a hand that closes more slowly than executed
  (mean predicted synergy at step 8: 0.07–0.11, executed grasp 0.35–0.44), and under its own
  unprojected plans one that closes faster (predicted 0.60–0.63, executed 0.38–0.44).
- **M-proj's requests beyond the limit (reported):** 500, 500 and 503 of 512 requested grasp
  steps at R1, 192, 212 and 178 of 512 at R2.
- **TASK-089's request terms (reported, first command):** close@R1 0 / 64 for every M-proj seed
  (its first projected grasp from the open hand is −0.818 by construction) and 62, 64 and 64 for
  M-v1; lift@R2 64 for every M-proj seed and 62, 46 and 60 for M-v1.
- The A1 comparisons are model-only and were not re-run (§6.2).

**What this is and is not.** One run, offline, simulation only, 64 fresh roots. Stage P is an
executed-measure **no-regression gate** on the chosen plans at two fixed states per root. It is not
a closed-loop result; it does not show that M-proj closes, grasps or lifts in closed loop, and no
claim of this task rests on it. R7 does not change.

- **R27.139 — Stage P is recorded.** It ran once at `4d3491e` from a clean tree in two jobs
  (281 s), with no cap hit, no VOID exit and no problem in the summary.
- **R27.140 — P-PASS.** M-proj's C8@R1 39, 48, 54 (bar 39) and H8@R2 58, 61, 58 (bar 48) of 64
  on G-87100, 87101, 87102: every seed passes, G-87100 at the C8 bar exactly. Reported: M-v1's
  C8@R1 43, 55, 54 and H8@R2 62, 64, 62. One run, offline, simulation only, 64 gated roots; a
  no-regression gate, not a closed-loop result.
- **R27.141 — Stage BD runs next** (§6.3, development resets 91900–91931), at the same revision
  from the same clean tree.

## 18. Stage BD record (development, not gated; R27.142–R27.146; decided by Claude under owner delegation)

**Run conditions.** Stage BD ran once on 2026-10-14 (15:46–16:59 UTC) with
`scripts/run_task093_stage.sh bd outputs/task093-bd outputs/task093-p/summary-p.json`, at
`4d3491e` from the same clean tracked tree as §17 (the summarizer checks that Stage P ended
P-PASS cleanly at the same revision; no problem). Before each of the seven jobs the stage script
read BOARD.md's `## Requests`; before the M-v1-87100 job a `- v2d:` request was present, and the
job waited 120 s until it was gone (16:22:48–16:24:48 UTC), as §10 requires. Every job took the
shared lock with `gpu_run.sh --wait --board`. The V2D session was not sent a separate message by
this run; the BOARD.md protocol was its channel. Evidence: `~/develop/emai/evidence/task093-bd/`
(`SHA256SUMS` sha256 `08452d04…ab6f`): the goals report, the six run reports, the summary, the
stage log, the seven ledger entries and the `gpu_run` log lines. The goal frames
(`goals-bd-grasp.npz`) stay in `outputs/task093-bd/` on the data disk.

**Jobs and GPU time.** Goals (32 resets, 91900–91931, 28 rejected draws; targets can 12, cube 11,
apple 8, banana 1) 17 s; ik, ik-nopress, hold and random 11–12 s each; M-proj-87100 2 070 s and
M-v1-87100 2 070 s (wall, 6 processes each). No job reached the 7 200 s cap; every report is
`completed: true`, `cap_hit: false`. In total 7 jobs, **4 203 s (about 1.17 GPU h)**.

**The summarizer's output, as `summarize_task093.py --stage bd` printed it:**

```
{
 "stage": "bd",
 "row": null,
 "bar": 0.4,
 "bars": null,
 "problems": []
}
```

BD has no row (it is not gated). **The bar is set:** ik reached 21 / 32 (r = 0.656, at least
0.30), so bar = min(0.40, 0.9 × 0.656) = **0.40**; at BS's n = 128 the pooled bar is
**156 / 384** and the per-seed floor **26 / 128** (`summary-bd.json`: `pooled_bar_at_bs` 156,
`seed_floor_at_bs` 26, `escalate: false`). The summary lists no problems, so BD-ESCALATE does
not apply.

**Per arm** (32 development resets, one run, simulation only; exact 95 % intervals; contact is
thumb plus index or middle on the target; rise is the maximum target rise; "stopped" is the
embodiment's measured-joint-velocity stop):

| Arm | Successes | 95 % interval | Grasp contact | Median max rise | Stopped | Plan s / command |
| --- | --- | --- | --- | --- | --- | --- |
| ik | **21 / 32** | 0.468–0.814 | 31 | 6.1 cm | 2 | — |
| ik-nopress | 21 / 32 | 0.468–0.814 | 30 | 5.8 cm | 2 | — |
| hold | 0 / 32 | 0.000–0.109 | 0 | 0.0 cm | 0 | — |
| random | 0 / 32 | 0.000–0.109 | 0 | 0.0 cm | 0 | — |
| **M-proj-87100** | **0 / 32** | 0.000–0.109 | 16 | 0.27 cm | 5 | 0.98 |
| **M-v1-87100** | **0 / 32** | 0.000–0.109 | 13 | 0.19 cm | 5 | 1.00 |

- **Per object.** ik succeeded on can 11 / 12, cube 10 / 11, apple 0 / 8 and banana 0 / 1
  (ik-nopress the same). Neither world-model arm succeeded on any object. The largest maximum
  rise was 4.4 cm for M-proj-87100 (next 0.95 cm) and 2.1 cm for M-v1-87100 (next 0.89 cm).
- **Final subgoal reached:** the lift subgoal (3) on 27 of 32 resets for both world-model arms,
  the grasp subgoal (2) on 5; ik reached the lift subgoal on 31.
- **Executed close per phase** (share of executed commands with the right-grasp channel > 0.5):
  grasp phase 0.75 (M-proj-87100), 0.71 (M-v1-87100), 0.88 (ik); lift phase 0.89, 0.85 and 1.0.
- **The projector equivalence (a B-VOID check at BS, recorded here):** M-proj-87100's applied
  channel 13 equalled the request on **all 11 650** commands (2 750 in the grasp phase); for the
  unprojected control M-v1-87100 it differed on 6 803 of 11 659 (2 192 of 2 743 in the grasp
  phase).
- **Support and requests (reported):** M-proj-87100's chosen plan steps off the training support
  0.0 (every plan wholly on support), and 0.528 of its *requested* grasp steps lie beyond the
  limit; M-v1-87100's plan steps off support 0.453 (plans wholly on support 0.099), its first
  grasp step clipped on 0.58 of commands (0.80 in the grasp phase).
- **Velocity stops.** 5 per world-model arm and 2 per ik arm, all the embodiment's frozen
  measured-joint-velocity rule, a declared failure, not a harness defect.

**§6.3's deferral measures (TASK-091 §5.3; reported only, no row reads them; per reset over the
32 episodes; medians with the 10th–90th percentiles):**

| Measure | M-proj-87100 | M-v1-87100 |
| --- | --- | --- |
| (1) closing commands before first contact, episodes with contact | median 3 (p10 0.5, p90 20, max 33); n = 16 | median 3 (p10 1, p90 27.4, max 72); n = 13 |
| (1) episodes with no contact (counted to the end, flagged) | 16 | 19 |
| (2) first executed close in the grasp phase, command index | median 8 (p10 8, p90 9, max 11) | median 8 (p10 8, p90 11, max 12) |
| (2) never closed in the grasp phase / grasp phase not reached | 3 / 0 | 3 / 0 |
| (3) share of grasp-phase re-plans with k = 1, per reset | median 0.785 (p10 0.26, p90 0.85); pooled 0.750 | median 0.59 (p10 0.53, p90 0.75); pooled 0.585 |
| (3) median planned close step k, per reset | median 1 (p90 3.95, max 4.5) | median 1 (max 1) |
| (3) longest run of re-plans with k > 1, per reset | median 7.5 (p10 7, p90 8, max 34) | median 3 (p10 2, p90 5, max 6) |

As TASK-091's BD record read it, the 2 / 11 rate limit lets an executed close register from an
open hand only after about 8–9 grasp-phase commands, so both arms' median first close (command 8)
is near the earliest possible; that deferral caused either arm's failures is not shown.

**What this is and is not.** A development cohort of **32 resets, one run, one model seed
(G-87100) per world-model arm, simulation only, not gated, not Stage BS**. It sets the bar (0.40)
and shows that the plumbing and the arms ran under the rules, including the projector
equivalence that BS checks. It carries no row and no claim, and its counts are not a prediction of
BS. Both world-model arms scored 0 / 32 (exact 95 % interval 0.000–0.109 each), so BD shows no
difference between M-proj and M-v1 on these resets: at BS's per-seed floor rate (26 / 128 =
0.203) a seed would expect about 6.5 successes on 32 resets. The other two seeds of each arm did
not run in BD. ik's 21 / 32 is below TASK-091 BD's 26 / 32 on other resets, with apple 0 / 8 here.
No MP4 replay was made in BD (the stage script makes none). No knob changes (§6.3), and R7 does
not change.

**What the GO review decides** (§6.3): an independent reviewer reads this record and posts GO or
NO-GO before Stage BS. BD can show a defect, whose fix needs its own ruling; it changes no knob by
itself. The bar is set (r ≥ 0.30), so the bar rule does not escalate. After a NO-GO, BD is
recorded again on fresh development seeds named by a ruling. This record does not decide GO.

**Next.** Stage BS (`scripts/run_task093_stage.sh bs <out> outputs/task093-bd/summary-bd.json
outputs/task093-p/summary-p.json`, 918000–918127) runs only after an independent GO on this
record, at `4d3491e` from a clean tree, after the V2D session is told. It has not run.

- **R27.142 — Stage BD is recorded.** It ran once at `4d3491e` from a clean tree in seven jobs
  (4 203 s, about 1.17 GPU h), with no cap hit, every report complete and no problem in the
  summary; one job waited 120 s for a V2D request on BOARD.md.
- **R27.143 — The bar is 0.40.** ik 21 / 32 (r = 0.656 ≥ 0.30); pooled bar at BS 156 / 384,
  per-seed floor 26 / 128. Counts, development only (one run, 32 resets, simulation only):
  M-proj-87100 0 / 32, M-v1-87100 0 / 32, ik-nopress 21 / 32, hold 0 / 32, random 0 / 32.
- **R27.144 — the projector equivalence held in closed loop:** M-proj-87100's applied channel 13
  equalled its request on all 11 650 commands (development).
- **R27.145 — Stage BS waits for an independent GO** on this record (R27.137); nothing of §2–§12
  changes.
- **R27.146 — R7 does not change.**
