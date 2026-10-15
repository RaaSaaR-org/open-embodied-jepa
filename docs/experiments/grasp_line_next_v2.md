# After TASK-094's K-NULL: where the grasp line goes next (design note)

**Status: DESIGN NOTE, not a preregistration** (R27.200–R27.208, DECISIONS (bn); decided by
Claude under owner delegation, 2026-10-15). It records an offline CPU read of existing run
reports, reconciles the earlier "19 / 179 within 0.5 cm" figure with TASK-094's K-NULL, weighs
the options that every fired clause leaves open, and chooses TASK-095, a measurement-only
placement-resolution diagnostic. TASK-095 gets its own DRAFT preregistration, with seeds, bars and
rows fixed before anything runs. No GPU was used, nothing was trained and no simulator episode
ran for this note.

**This is not an Apple→Plate experiment.** R7's canonical status sentence does not change.

## 1. Where things stand

- **TASK-094 closed at K-NULL** (R27.194–R27.197; gated, one run, simulation only, the scripted IK
  follower on 128 paired resets 941000–941127, no model). At the follower's exact pose, the lift
  rates were K-1.0 88 / 128 (exact 95 % 0.600–0.766), K-s\* 84 (0.567–0.738) and K-0.6 81
  (0.543–0.716). The losses against +1.0 were +0.031 (reset-bootstrap 95 % −0.031 to +0.094) and
  +0.055 (−0.016 to +0.125). Both upper bounds are below δ_K = 0.15.
- **Placement mattered more** (reported beside the row; a scripted dose of two directions of one
  size, not the world-model arms' errors). A rigid +y palm offset of 1.2 cm lifted 31 / 128
  (0.171–0.326; Δ +0.445, +0.352 to +0.539). A +x offset of 1.2 cm lifted 82 (Δ +0.047, −0.039 to
  +0.133, not significant).
- **Post hoc, not preregistered** (review of #228; no row, no bar): under the +x offset, holding at
  s\* lost +0.102 against +1.0 (82 against 69; discordant 17 / 4, exact one-sided McNemar
  p = 0.0036, reset bootstrap 95 % +0.039 to +0.172). K-NULL holds at an exact pose; it does not
  show that partial closure is harmless when the palm is misplaced.
- **The routing.** Under R27.157 the grasp line paused, and the owner was told (2026-10-15). D's
  work, for example a Phase 4 precursor with a scripted closure rule, needs its own ruling, because
  Phase 4 is not admitted (R25.27, R27.148). No grasp controller over TASK-087's checkpoints is
  proposed until a ruling cites evidence of a different kind (R27.157).
- **Reach works at its bar** (Z3-REACH, R25.26; TASK-088, gated, one run, simulation only, 64 fresh
  resets, one model seed gated). G-87100 reached 62 / 64 within 5 cm (exact 95 % 0.892–0.996). Its
  median final distance was **2.3 cm** and its median final orientation error 0.16 rad; 53 / 64
  were within 3 cm. The follower's median final distance was 0.1 cm. P-87100 reached 53 / 64 (bar
  52; margin one reset), with a median of 3.7 cm and 0.84 rad. "LeWM needed" is not shown.

### 1.1 The fired clauses (scopes quoted from their rulings, not widened)

| Clause | It closes (short form) | It leaves open (relevant here) |
| --- | --- | --- |
| TASK-054, TASK-057 §7, TASK-062 §10, TASK-065 §10, TASK-075 §7 | the apple-line items listed in [grasp_next_step_v1.md §1](grasp_next_step_v1.md) | not grasp-line items |
| **R26.33** (TASK-089 §9) | CEM planning of grasp-and-lift over TASK-087's checkpoints: no further planner, cost, horizon or action-parametrisation variant over them for grasping **without new evidence of a different kind** | LeWM, DINOv2, `play-v1` as a base, CEM in general, reach planning, Phase 4 as a goal |
| **R27.71** (TASK-090 §9) | more scripted grasp-and-lift play of `play_policy_grasp_v0`'s kind | model changes; other kinds of data (real G1, a different demonstrator or pick recipe, **other sensors**) |
| **R27.149** (TASK-093 §8) | grasp-channel rate projection of CEM's candidates and its variants, over TASK-087's G and TASK-092's G-req | full-embodiment projection (own ruling under R26.33); TASK-092 §2's untested model changes; the corpora; reach; Phase 4 as a goal |

Not fired: TASK-092 §8 (A‴-FAIL, R27.100) and TASK-094 (no clause, §7.2). The owner stopped the
moving-plate line after TASK-083 (JEPA_ZERO_SHOT_PLAN.md, interview of 2026-10-08: "Stop the
moving-plate line after TASK-083? Yes, stop"). Phase 4 is not admitted (R25.27, R26.34, R27.148).
The TASK-094 card records that a pause ruling and any Phase 4 precursor ruling are the owner's;
this note makes no Phase 4 precursor ruling.

## 2. The offline read (R27.200)

**What was read.** The same inputs as [grasp_next_step_v1.md §2](grasp_next_step_v1.md): the
Stage BS run reports of TASK-089, TASK-091 and TASK-093. That is 1 856 world-model (WM) episodes,
every arm that ran TASK-089's step-averaged planner over a G checkpoint, and 320 IK-follower (ik)
episodes on the same resets. The read also took TASK-094's gated K reports and goals, and two
arithmetic checks described in §2.5.

**How.** CPU only. Scripts and their verbatim outputs are in
`~/develop/emai/evidence/grasp-next-v2/` (`SHA256SUMS` sha256 `2990e3d5…10a3`): `read_placement_failures.py`,
`read_lift_phase.py`, `read_close_phase.py`, `read_held_placement.py`, `read_close_jitter.py`,
`read_by_object.py`, `pose_term_resolution.py` and `exact_intervals.py`, each with a `.txt`
output. Phases are rebuilt from the subgoal-switch log, as in the earlier read: 0 hover,
1 descend (open), 2 close, 3 lift. Each boundary may be off by one command.

**Qualifiers that apply to every number below.**
- Each stage is one run, in simulation only, on its own gated cohort.
- Pooled numbers pool arms, model seeds, checkpoints and cohorts. They are for orientation only.
- Every link between a measure and success is an **association, not an intervention**.
- This read was chosen after the results were known.
- The run reports log the palm's *distance* to the current subgoal's palm only. They do not log
  its direction, its orientation or the object's pose, so none of those is measured here.
- ik commands its palm 5 cm below the goal palm on the grasp subgoal (e9's press), so part of
  its distance during the close is the press against the object or table.

### 2.1 The failure is in acquiring the grasp, not in holding it during the lift

"Established grasp" means grasp contact (thumb plus index or middle on the target) at the last
command of the close phase.

| | WM | ik |
| --- | --- | --- |
| Touched the target in the close | 982 / 1 848 | 302 / 320 |
| **Established grasp at the end of the close** | **194 / 1 848** (0.105; exact 0.091–0.120) | **236 / 320** (0.738; 0.686–0.785) |
| Success given an established grasp | 103 / 194 (0.531; 0.458–0.603) | 220 / 236 (0.932; 0.892–0.961) |
| Success without one | 5 / 1 654 | 1 / 84 |
| Close subgoal "reached" (palm 1 cm and 0.17 rad, hand 0.15 rad RMS, 3 steps) | 67 / 1 856 | 153 / 320 |

- **Touch, then lose.** In failing episodes that touched the target, contact was present on a
  median of 5 % of the close commands after the first touch (WM) and 7 % (ik). In successes it
  was present on about 100 %.
- **The lift itself is not where it fails.** In the lift phase, the palm of the failing WM
  episodes came within 6 cm of the lift subgoal's palm (12 cm above the grasp) on 1 400 / 1 435.
  It rose with no contact: the hand lifted empty.
- **So** the WM arms lose most of their episodes before the lift, by not establishing a grasp
  (10.5 % against 73.8 %). The conversion of an established grasp is a second, smaller gap
  (0.53 against 0.93).

### 2.2 Reconciling 19 / 179 with K-NULL (R27.201)

The earlier read found that WM episodes whose minimum close-phase palm distance was below 0.5 cm
succeeded only 19 / 179 (0.106; exact 0.065–0.161), against ik's 160 / 241 (0.664; 0.600–0.723).
Read alone, that suggested that placement is not the main gap. With K-NULL beside it, the figures
fit together as follows.

1. **The minimum distance is a single-step pass, not a held placement.**
   - Over the close phase from the first touch, the WM palm sat a median **2.62 cm** from the
     grasp subgoal's palm (p25–p75 1.96–3.37), against ik's 1.23 cm (0.70–1.73).
   - Over the last 10 close commands, the medians were 2.67 cm against 1.30 cm.
   - In the 179 episodes, the WM palm was within 1 cm for a median of 8 (successes) and 9
     (failures) of about 100 close commands. At the end of the close it was a median 2.03 cm away
     in the failures.
   - Only 20 of the 179 met the close subgoal's switch condition (9 succeeded, against 10 / 159
     that did not meet it).
   - The WM arms' typical held error is therefore about twice the 1.2 cm dose that, in the +y
     direction, cut the follower's lift rate from 88 to 31 of 128 (K).
2. **Of the 160 failures among the 179:** 39 never touched the target, 96 touched but never
   raised it 1 cm, 11 raised it under 5 cm, 8 raised it 5 cm without the 20-command hold, and 6
   were stopped. That is the touch-then-lose pattern of §2.1, not a closure that held and then
   slipped.
3. **Distance is a scalar; K shows that direction matters.** At 1.2 cm, +y cost 0.445 and +x
   0.047. The direction and the orientation of the WM placement errors are not logged. So even
   at a matched held distance, WM and ik are not matched placements: in the held-distance bin
   below 1 cm, WM succeeded 13 / 53 (0.245; 0.138–0.383) and ik 106 / 126 (0.841; 0.766–0.900),
   and at 1–1.5 cm, 17 / 151 against 74 / 86. Unlogged direction and orientation errors are one
   reading of the residual gap. It is **not measured**.
4. **Closure acts with misplacement, not alone.** At the exact pose, closure barely mattered (K).
   Under the +x offset, s\* lost 0.10 against +1.0 (post hoc). In the 179, the commanded grasp at
   the step of minimum distance was ≥ 0.9 in 9 episodes (4 succeeded) and below 0.7 in 133
   (12 succeeded). This is consistent with a compounding effect, on few episodes, and is an
   association.

**Reconciliation.** 19 / 179 does not contradict K-NULL. It conditions on a transient minimum,
not on a held placement. The WM palm's held error during the close (median about 2.6 cm, with
unknown direction and orientation) lies in the range where K showed that placement, not closure,
decides the lift. Both readings point to placement held through the close as the main locus, with
partial closure as a secondary, compounding factor once the palm is off.

### 2.3 Is anything besides closure and placement implicated? (R27.202)

| Candidate | What the logs show | Reading |
| --- | --- | --- |
| **Holding the palm still while closing** | In the first 20 commands after the first touch, the WM arms commanded a median lateral path of 12.3 cm (successes) and 12.7 cm (failures), with a net displacement over path of 0.34 and 0.32. ik commanded 3.9–4.8 cm, with 0.86. These are commands, not measured palm motion. | **The WM arms move the arm back and forth while closing**, about three times the follower's commanded path. Within WM, this does not separate successes from failures, so it is **not shown** to cause the failures. It is part of "held placement", which TASK-095 measures. |
| **The object** | Success by target: cube WM 14 / 592 (0.024) against ik 86 / 104 (0.827); can 64 / 707 (0.091) against 113 / 127 (0.890); apple 30 / 536 (0.056) against 22 / 86 (0.256). In K, K-1.0 lifted the apple 6 / 31, the can 39 / 46 and the cube 43 / 51. +y 1.2 cm cut the can to 5 / 46 and the cube to 25 / 51. | **Yes, as a ceiling.** The scripted follower itself lifts the apple rarely, so the apple caps every arm judged against it. The WM gap is largest on the cube, where the follower does best. A cube's faces may need yaw alignment, and the switch's 0.17 rad orientation condition is near G's median reach orientation error of 0.16 rad. That orientation causes the cube gap is **inferred, not measured**. |
| **Orientation** | Not logged in the grasp reports. In TASK-088's reach, G's median final orientation error was 0.16 rad. | **Plausible, unmeasured.** TASK-095 adds a yaw dose. |
| **The press** | ik-nopress lifted 50 / 64 against ik's 50 (TASK-088), 46 against 48 (TASK-089) and 89 against 91 (TASK-093). After the first touch, WM commanded a median dz of −0.21 (ik −0.40). | **Not implicated** for the follower. The WM arms also press down. |
| **Approach timing** | While the palm was more than 1 cm away in the close phase, the share of commands with a closing grasp (> 0) was a median 0.93–0.95 (WM) and 0.87–0.91 (ik). The WM close phase ends by its 100-command timeout in almost every episode. | **Not separable** from placement in these logs. No evidence that the WM arms close too early relative to the follower. |
| **The lift** | Failing WM palms reach the lift pose without the object (§2.1). Successful WM episodes command the lift like ik (median dz 0.34–0.38 against 0.35). | **Not implicated.** The grasp is already lost. |

### 2.4 Summary of the locus

Measured: the WM arms fail mainly by not establishing a grasp during the close. Their held palm
error is about twice the follower's, and in a range where K's scripted offsets cut the lift rate
sharply in one direction. Object geometry caps the follower on the apple. Inferred: unlogged
direction and orientation errors, together with arm motion during the close, explain why the WM
arms rarely establish a grasp even at a small scalar distance.

### 2.5 Can the frozen cost resolve a cm-scale placement? (arithmetic, CPU; R27.203)

TASK-088's cost (`zero_shot_runtime.PlannerModel.distance`) is the latent MSE plus λ times the
mean squared *standardised* difference of the 14 right arm and hand joint positions (λ = 2.0552).
The standardisation uses G's training moments. G-87100's arm joint standard deviations are
0.25–0.70 rad, the spread of the whole play workspace.

For each of TASK-094's 128 gated K goals, `pose_term_resolution.py` linearised the arm at the
grasp subgoal. It used MuJoCo forward kinematics on the CPU (no simulation step, no rendering, no
model run) and took the minimum-norm joint change that moves the palm by v with its orientation
fixed. It assumes a perfect state prediction, so it is a best case for G.

| Change at the grasp subgoal | λ-weighted pose-term change, p10 / median / p90 |
| --- | --- |
| palm +y 1.2 cm (K's costly direction) | 0.0048 / **0.0065** / 0.018 |
| palm +x 1.2 cm | 0.0043 / 0.0081 / 0.080 |
| palm +z 1.2 cm | 0.0090 / 0.0146 / 0.064 |
| palm +y 0.5 cm | 0.0008 / 0.0011 / 0.0032 |
| palm +y 2.5 cm | 0.021 / 0.028 / 0.079 |
| palm yaw 0.17 rad | 0.062 / 0.073 / 0.153 |
| hand: the goal's measured hand → fully closed synergy | 0.91 / 1.09 / 1.90 |

For scale: in TASK-094 M (reported, not read), the ceiling cost on the Rs discordant roots had a
median total of 0.778 for the squeeze against 0.852 for s\*.

**Reading.** In the pose term, a 1.2 cm palm error is worth about 1 % of a typical cost and
under 1 % of a full hand-closure change. The pose term is therefore close to blind to the
placement that K shows matters. Whether the *latent* term resolves it (4 × 4 pooled DINOv2 tokens
of a 112 px frame, where 1.2 cm is a few pixels) is **not measured**. That is the question TASK-095
answers. Reach at a 5 cm bar does not answer it: G's median reach error was 2.3 cm.

## 3. Candidates (R27.204)

Every candidate was checked against §1.1. GPU hours are estimates from earlier task records.
"Chance" is a judgement on existing evidence, not a measurement.

| | Option | Scope legality | Cost | Chance it moves closed-loop grasp success | Measure that would show progress |
| --- | --- | --- | --- | --- | --- |
| **P (chosen)** | **TASK-095: placement-resolution diagnostic** (§4). Measurement only. A scripted placement dose that maps the follower's tolerance (direction, size, yaw), and an offline ranking: do the frozen cost (on real frames, the ceiling) and G's predicted cost rank an exact placement above a costly offset from a root at the start of the close? | Legal. It runs no controller, no training and no world-model closed loop. It preregisters no planner, cost, horizon or action-parametrisation variant, so R26.33, R27.71 and R27.149 are not engaged. It proposes no grasp controller (R27.157). It reuses TASK-094's K and M machinery. Phase 4 is not touched. | **≈ 0.5 GPU h** (ceiling 1.0) plus CPU | **None directly.** It is a routing step. Every fix it routes to costs 15 GPU h or more. It decides which family can work: search/execution, cost specification or model/sensor. | The share of discordant roots on which the ceiling and G rank the exact placement first; the tolerance envelope (the offset size per direction at which the follower loses 0.15 of its lifts) |
| D | **Phase 4 precursor:** the WM plans the reach to the grasp pose, and a scripted closure-and-lift rule does the grasp. | Phase 4 is not admitted (R25.27, R27.148), so it needs its own ruling, which the TASK-094 card reserves to the owner. It is also an action-parametrisation variant over TASK-087's checkpoints for (a part of) grasping, because the planner no longer commands the grasp channel. So **R26.33 applies**, and its condition is not met. | ≈ 10–15 GPU h plus a preregistration | **Low.** It inherits the WM arms' held placement error (median 2.6 cm), and the scripted closure was not the bottleneck at an exact pose (K-NULL). +y 1.2 cm alone left the follower at 31 / 128. It would not show "LeWM needed". | Grasp-and-lift rate of WM reach plus scripted closure, against ik and against a no-WM placement baseline |
| X | **Return to the v2 Apple→Plate LeWM line**, which has gated successes (TASK-081 L-PASS; TASK-083 REP-PASS). | **Not legal under delegation.** Every gated LeWM success on v2 is under the simulation-only moving-plate condition C1-M, and the owner stopped the moving-plate line after TASK-083. A later LeWM task on a plate target counts as a task change only if it is paired with a condition where the target must be predicted, a moving plate or one that leaves the view (DECISIONS 2026-10-02, R1–R4). TASK-075 §7 closes a LeWM place planner under TASK-074's condition on those views. It needs the owner to reverse the stop. | ≈ 10–20 GPU h per gated task | Moves no grasp result, and adds little to the owner's goal of one general model. On v2, "LeWM needed" is not shown (H-rule). | Showing "LeWM needed" (beating H-rule) under a new condition |
| B | **Grasp-outcome head and cost** (TASK-092 §2 rank 2), a model change. | Open (TASK-092 §2; R27.71 and R27.149 list it as not closed). It needs its own ruling on simulator-truth labels in a learned cost. | ≈ 17–18 GPU h | **Low now.** It targets the closure decision. K-NULL says closure is not the primary locus, and §2 says the grasp is lost at placement and acquisition. | Closed-loop lift rate at n = 128 |
| C′ | **Cartesian palm term:** replace or extend the joint-space pose term with a palm-position term normalised at cm scale. | A cost variant over TASK-087's checkpoints, so **R26.33 applies**. §2.5 is a design fact about the cost, not outcome evidence. TASK-095's R-COST or R-RESOLVED row would be evidence of a different kind, under its own ruling. | ≈ 0.5 GPU h offline, plus ≈ 14 GPU h closed loop | Plausible if TASK-095 shows that the cost, not G, lacks placement resolution. Unknown now. | Held close-phase palm distance; established grasps; lift rate |
| F | **Full-embodiment candidate projection** (the arm channels). | Open under R26.33, with its own ruling. Its condition is not met. | ≈ 15 GPU h; the IK surrogate was judged likely to exceed the job cap (TASK-092 §2; not measured) | Addresses execution only. Unknown. | Plan-to-execution palm error; lift rate |
| W | **A sensor or data change:** a wrist camera, a higher resolution or a finer latent grid, with a new corpus and retraining. | Open (R27.71: "other sensors"; other kinds of data). Needs its own preregistration. | ≈ 30–60 GPU h plus collection | Plausible if TASK-095 reads R-MODEL or R-COST on the latent term. Large and slow. | Offline placement resolution first, then the closed loop |
| Z | **Stay paused**, with no new grasp work. | Legal (R27.157). | 0 | None | — |

## 4. TASK-095 in outline (its preregistration fixes the details) (R27.205)

**Question.** The WM arms hold the palm a median 2.6 cm from the grasp pose while closing (§2.2),
and a 1.2 cm +y offset already costs the follower 0.445 of its lifts (K). Can the frozen
step-averaged cost resolve a placement error that costs lifts? If it can, does G's prediction
resolve it?

**Parts.** Each runs once, on fresh development or gated seeds and roots from a block that the
preregistration checks. Every GPU job runs under `gpu_run.sh --wait --board`.

1. **K′, the placement dose (scripted, no model).**
   - **Arms.** TASK-094's K-1.0 follower (the closing command at +1.0) on 128 fresh paired resets
     and TASK-088 goals. The palm target is rigidly displaced, on all four subgoals, by v ∈
     {±x, ±y} × {0.5, 1.2, 2.5 cm} (pelvis frame), plus a yaw offset of ±0.17 rad about the
     palm's vertical axis.
   - **What it reads (reported; it sets the routing bar of part R).** The follower's tolerance
     envelope: per direction, the smallest offset whose lift loss against K-1.0 has a
     reset-bootstrap lower bound above 0.15 (salt fixed by the preregistration). The primary
     offset for part R is fixed **before** K′ runs: +y 1.2 cm, TASK-094's costly offset. K′
     reports whether it still costs at least 0.15 on fresh resets.
   - Scripted probes, not learned results.
2. **R, the ranking (offline, frozen G, no training).**
   - **Roots.** Fresh roots at the demonstrator's state at the end of its descent: palm at the
     pre-grasp pose, hand open, before any close (TASK-089's R1). Placement is decided there.
   - **Branches.** The follower's close-and-lift with the palm target at the goal (exact) and at
     each offset of K′, simulated to the true outcome and recorded as real frames and measured
     states.
   - **Costs.** Each branch is scored by TASK-089's frozen step-averaged cost in two ways: (i)
     from G-87100–87102's predictions given the branch's commands from the root; (ii) from the
     branch's real encoded frames and measured states (the ceiling). The primary window is the
     planner's own horizon (8 commands); 16 commands are reported. The latent and pose terms are
     reported separately.
   - **What it reads.** On discordant roots (the exact branch lifts and the primary offset branch
     does not), the share on which the cost ranks the exact branch below the offset branch, for
     the ceiling and for each G seed.
3. **S′, the search (offline, frozen planner; reported).** TASK-089's frozen planner at the same
   roots: G's cost of its chosen plan against the exact branch's commands, the miss as in TASK-094
   S, and the palm displacement that G predicts for its chosen plan.

**Rows (sketch; the preregistration declares bars and margins before any run).**
- **R-VOID:** a declared check fails. Examples: K′'s K-1.0 outside its band; a positive control (a
  branch that does not close) ranked first; seeds, digests or determinism.
- **R-UNREAD:** too few discordant roots to read (TASK-094 used 16).
- **R-COST:** the ceiling ranks the exact branch first on fewer than 0.75 of the discordant roots.
  The goal and cost cannot express the placement the lift needs, even with a perfect prediction,
  so no planner, search or horizon variant over this cost can fix it.
- **R-MODEL:** the ceiling does rank it first, and G does not, on two of three seeds.
- **R-RESOLVED:** both rank it first.

**Routing only.** Each row routes to a ruling. None of them admits a controller by itself.
- **R-COST** routes to a cost-specification change (C′) or a sensor change (W). C′ is a cost
  variant over TASK-087's checkpoints, so it goes only through a ruling that finds R26.33's
  condition met. The preregistration states whether R-COST is put forward as such evidence.
- **R-MODEL** routes to a model or sensor change (W, or TASK-092 §2's open kinds).
- **R-RESOLVED** routes to the search and execution side (F, or a CEM-noise or horizon variant)
  under R26.33. The preregistration states whether R-RESOLVED with S′'s miss is put forward as
  evidence of a different kind.
- D stays the owner's ruling whatever TASK-095 reads. K′'s envelope gives D the placement target
  it would have to meet.

**What it cannot show.** It runs no closed-loop controller, so it cannot show a grasp success.
Its roots are the demonstrator's states, not the WM arms' states. Its offsets are scripted, rigid
and of chosen directions. It covers one scene, camera, encoder and corpus, in simulation only. A
ranking from a root is necessary for the planner to place well, but not sufficient.

**Disclosed expectation.** §2.5 makes it likely that the pose term alone does not resolve 1.2 cm.
So R-COST or R-MODEL are the likelier rows, unless the latent term carries the placement. This is
stated now, so the row cannot be read as a surprise.

**Budget.** K′ has 15 arms at about 30–45 s each, from TASK-094's K jobs: about 0.15 GPU h. R and
S′ come to about 0.15–0.25 GPU h; TASK-094's roots, ranking and search jobs took 391 + 29 + 151 s.
Stage 0 adds about 0.1–0.2 GPU h. That is **≈ 0.5 GPU h**, with a ceiling of 1.0 GPU h.

## 5. Why P, and not D or X (R27.205–R27.206)

- **P is decisive and cheap.** Every fix on the table costs 15 GPU h or more of closed loop.
  Every world-model grasp closed loop so far has failed: TASK-088's grasp part, TASK-089,
  TASK-091 and TASK-093 (gated) and D-RATE′ (development), three of the gated ones after offline
  gates that passed. The open question that splits the fixes is
  whether the cost and G can resolve a placement that costs lifts. P answers it in about
  0.5 GPU h. It reuses TASK-094's machinery, and it engages no clause.
- **D is not chosen now.** It is Phase 4 work (not admitted). It engages R26.33 as an
  action-parametrisation variant over TASK-087's checkpoints. And on existing evidence it would
  most likely fail on the very placement error that P measures. If P reads R-COST or R-MODEL, D
  inherits that limit. If P reads R-RESOLVED, a planner fix comes before D. P's envelope is the
  precision target D would need.
- **X is not open to this ruling.** The owner stopped the moving-plate line, and every gated v2
  LeWM success is under that condition. Reopening it is the owner's call. It is noted for the
  owner, not proposed.
- **B, C′, F and W** wait for P's row, which tells which of them is worth its cost.

## 6. Rulings (R27.200–R27.208)

See DECISIONS (bn).
