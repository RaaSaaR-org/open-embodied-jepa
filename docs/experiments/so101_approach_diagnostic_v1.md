# TASK-102 Δa: why CEM over J does not deliver the pre-grasp pose (development decomposition)

**Status: DECLARED (measures, cohort, readings and rows fixed before any measurement; this commit
runs nothing).** Development diagnostic of TASK-102's Stage 0 row S0-APPROACH
([record](so101_hybrid_lift_v1_stage0.md), DECISIONS 2026-10-20 (bz), R28.114–R28.116), declared
by Claude under owner delegation (DECISIONS 2026-10-20 (ca), R28.119–R28.125). It is not a gate,
freezes nothing, makes no claim and admits no controller variant. Simulation only,
`so101_cube_lift_v1`, not Apple→Plate. R7 does not change. The world model is `native_jepa`
J-100101 (TASK-100), not LeWM.

## 0. Why this runs, and what it decides

TASK-102's Stage 0 found that TASK-101's CEM over J, aimed at the pre-grasp image subgoal
(subgoal 1), handed over by tolerance (τ_h = 0.08 rad arm RMS, jaw within 0.15 rad, for
3 commands) on 0/32 of cohort A's resets in each of its four configurations; the median closest
arm RMS to subgoal 1 was 0.354 rad for the chosen H16-last. TASK-101's K′ looked the same
(0.26–0.48 rad). TASK-101's Δ showed that from far states CEM's first 8 commands move the arm
toward the subgoal open-loop (m3 0.69–0.94 at t = 0, 10) and that J ranks the demonstration's
commands above random ones on subgoal 1. So J's cost is informative locally, yet the closed loop
stalls far from the pose.

S0-APPROACH names three candidates (§7.7 row 3: R28.57's LeWM retrain, a cost or goal change, a
planner variant). Choosing among them without knowing **which component binds** would be the
"second variant before a decomposition" that TASK-100/101's consequence rules forbid. This
diagnostic separates the three:

- the **cost geometry** of the image-subgoal cost itself (the latent error between the current
  frames' true encoding and subgoal 1's latent, with no model in between);
- the **model** (J's predicted cost against that true-frame cost);
- the **planner** (CEM's budget and horizon, and whether CEM reaches the pose at all when the cost
  is perfect).

Its rows (§6) recommend the next study. They do not start it: the owner, or Claude under
delegation, rules.

## 1. Cohort, seeds and salts (fresh; disjoint from every SO-101 range used)

- **Cohort L: 104000–104015** (16 resets), reserve **104016–104031**. Each seed's goal is the
  v1 goal demonstration (follower v1 with no offset, TASK-101 §4's keep rule, TASK-101's
  `demonstrate` unchanged): a rejected seed's slot takes the next reserve seed. The goal is
  subgoal 1 (the frames and measured joints after command 40), encoded with TASK-100's live
  featuriser (`so101_closed_loop_runtime.LiveFeaturiser`, fit-report and moments hashes checked).
- **Λ2's closed loops run on L's slots 0–7** (the first eight kept seeds); Λ1 runs on all 16.
- **Debug seeds 104900–104909**: plumbing and timing only, never counted or reported as a reading.
- **Salts: 10116** (Λ2's planner seed, `10116 · 10⁶ + reset seed`, the same for every Λ2 arm on a
  reset) and **10117** (Λ1's random candidates, `default_rng([10117, seed, state index])`).
- **Ranges used before, all disjoint from the above** (checked on every ref on 2026-10-20 with
  `git grep` over `src/`, `scripts/`, `docs/`, `benchmarks/`, `tests/`, `configs/`; the only
  match of `104xxx` is the end bound of TASK-102's debug range in a test): S2 development seeds
  0–63, 100–163, 1000–1015; the play-corpus streams 990000–990027, 990100–990163 and
  991000–993999; TASK-100's model and debug seeds 100100–100102, 100900–100920; TASK-101's
  101010–101999 (K, K′, D, S, D′, X, debug, reserves) and 102000–102299 (the envelope's E, K″,
  D″); TASK-102's 103000–103999 (P, P′, D, S, A, D′, A′, debug, reserves). Salts used before:
  9801, 9802, 9901, 9902, 10101–10108, 10110–10115.

## 2. Λ0 — Stage 0's own approach logs, re-read (reported only; CPU; no new episode)

A's 128 approach episodes (`outputs/task102-stage0/A-episodes/`, per-step `q`, `latents_raw16`,
logged planned `cost`; goals `A-goals`) are re-read with no simulation and no model: for each
step, the **true-frame cost** C_true = mean squared difference between the step's standardised
latent and subgoal 1's latent (12 288 values, float64; wrist half and overview half separately),
against the arm RMS to subgoal 1. Reported per configuration: the median over episodes of the
within-episode Spearman ρ(C_true, arm RMS); C_true at the reset, at the closest-RMS state and at
the last state; and the ratio of the logged planned cost to the true-frame cost at the same step
(how much cheaper J believes its plan's end state is than the frames it then sees). These are
A's spent counts' logs: Λ0 reads them, it selects nothing and no row reads it.

## 3. Λ1 — the cost landscape (16 seeds; no closed loop)

### 3.1 States

On each L seed, two families of settled states, with the cube at its declared reset:

- **Demonstration path:** the goal demonstration is replayed with rendering (its executed actions
  must equal the goal store's: guard G-replay), and the states after commands
  t ∈ {0, 4, 8, 12, 16, 20, 24, 32, 40} are taken (t = 40 is subgoal 1 itself).
- **Straight line:** q(d) = q₁ + (d / d₀)(q₀ − q₁) for all six joints (jaw included), where q₀ is
  the reset's measured joints, q₁ subgoal 1's measured joints and d₀ = arm RMS(q₀ − q₁); nominal
  distances **d ∈ {0, 0.02, 0.04, 0.08, 0.12, 0.16, 0.24, 0.32, 0.40, 0.48, 0.64} rad with d < d₀,
  plus d₀** (the reset pose itself). A state is placed in a fresh simulation after the task's
  reset: the six joint positions are set to q(d), velocities to 0, joint targets and controls to
  q(d), then 10 hold commands (targets q(d)) settle it. The settled arm RMS d_meas is what is
  reported; the nominal d indexes the bands. A state whose cube moved more than 2 mm during
  placement is excluded from every reading (counted and reported).

### 3.2 Costs at each state

- **C_true** (the true-frame cost): the state's two frames encoded live, latent error to
  subgoal 1's latent, full (12 288) and per camera (wrist tokens 0–15, overview tokens 16–31;
  Λ3).
- **C_J-hold**: J's predicted cost (TASK-101's `LatentCost`, unchanged) of H hold commands from the
  state's latent, in the forms `last` and `mean`, H ∈ {8, 16}; full and per camera.

### 3.3 Candidates at five line states

At nominal d ∈ {0.08, 0.16, 0.32, 0.48} and at d₀, **64 candidate sequences of 16 commands**:
the pose-law approach (TASK-101's H-pose law toward q₁, chained over its own targets), hold, and
62 random sequences uniform on [−1, 1]. Each is projected (TASK-101's chained projector) and then
**executed in a copy of the simulation** from the state, with both frames rendered and encoded
after every command (the copy's transport arithmetic equals the real one's: guard G-rollout,
§5). For each candidate: the true-frame cost of its executed frames, J's predicted cost from the
state's latent, in `last` and `mean` at H = 8 (its first 8 commands) and H = 16, and the arm RMS
to q₁ after command H (d_end). Per state and form:

- ρ_true = Spearman ρ(true-frame cost, d_end) over the 64 candidates;
- ρ_J = Spearman ρ(J's cost, d_end);
- ρ_JT = Spearman ρ(J's cost, true-frame cost);
- regret: d_end of the candidate each cost ranks first, minus the smallest d_end.

### 3.4 Readings

- **Band ordering (line paths).** Bands between consecutive nominal distances from 0.08 upward:
  (0.08, 0.12), (0.12, 0.16), (0.16, 0.24), (0.24, 0.32), (0.32, 0.40), (0.40, 0.48),
  (0.48, 0.64). For a cost c, a band's ordering accuracy is the fraction of seeds (both states
  present and not excluded) with c(upper) > c(lower); the band is **resolved** when it is at least
  0.75. **d_flat(c)** is the lower edge of the first unresolved band going outward from 0.08 rad
  (> 0.64 when all are resolved): the distance beyond which the cost's change stops exceeding its
  seed-to-seed noise in sign. Reported also: the bands below 0.08, each band's median increment
  and its ratio to the increments' interquartile range, and the median per-seed Spearman ρ(c, d)
  over each line path and over each demonstration path.
- **Candidate reading (H16-last, the configuration Stage 0 chose):** the medians over seeds of
  ρ_true and ρ_J at the far states (nominal d ≥ 0.16, i.e. 0.16, 0.32, 0.48 and d₀; 64 state
  readings), and **Δρ = median ρ_true − median ρ_J** there. Every other form and horizon is
  reported.

## 4. Λ2 — the planner and an oracle cost, in closed loop (L slots 0–7)

Phase 1 alone, as TASK-102's A ran it: the arm is driven by CEM toward subgoal 1 and the episode
ends at the hand-over (`so101_hybrid.HandoverRule`, τ_h = 0.08 rad, jaw 0.15 rad, 3 commands) or
at the T₁ = 60-command timeout. Every arm uses `SchemaCEMPlanner` unchanged (`minimum_std` 0.05,
no action penalty, candidates projected by TASK-101's projector, the first command of the best
candidate executed, replanned every command), one planner per episode seeded by salt 10116.

| arm | cost the planner minimises | samples × iterations (elites) | H, form |
|---|---|---|---|
| **J-ref** | J's predicted latent cost (TASK-101's `LatentCost`) | 256 × 6 (32) | 16, last |
| **J-big16** | the same | 1 024 × 12 (128) | 16, last |
| **J-big32** | the same | 1 024 × 12 (128) | 32, last |
| **O-img** (privileged) | the **true-frame cost**: each candidate is executed in a copy of the simulation from the current state, its final frames rendered and encoded, latent error to subgoal 1 | 256 × 6 (32) | 16, last |
| **O-pose** (privileged) | the **true pose cost**: each candidate executed in a copy, (arm RMS to q₁ / 0.08)² + (jaw error / 0.15)² after command 16 | 256 × 6 (32) | 16, last |

J-ref is Stage 0's chosen configuration re-run on fresh seeds. O-img is a "perfect model" for the
image-goal cost: it keeps the cost, the goal, the planner and its budget and replaces only J's
prediction by the simulator's. O-pose keeps the planner and budget and replaces the image cost by
the quantity the hand-over tests. The oracle arms read simulator state; they are ceilings and
diagnostic instruments, never controllers. For each arm: h = hand-overs by tolerance of 8, the
median closest arm RMS to q₁ and the jaw error there, commands used, and seconds per command.

## 5. Guards

G-replay (the demonstration replay equals the goal store's actions); G-rollout (after every
oracle command, the executed command re-simulated in the copy from the pre-command state equals
the real transport's joint positions and velocities bit for bit; and, in a test, chained
candidate execution in the copy equals `SO101Embodiment.execute` on the same state); G-projection
(the executed action equals the scored projected one, as in TASK-101/102); G-finite (no
non-finite cost); the TASK-100 frozen-files check, the checkpoint and implementation hashes, the
fit-moments and encoder digests (TASK-101's loaders, unchanged); the seed guard (every seed inside
L, its reserve or debug); no output overwritten. Every GPU job runs under
`scripts/gpu_run.sh --wait --board` (`--who oej:task102-approach-diag`), respecting other
holders' `- v2d:` request lines. O-pose and Λ0 need no GPU and run without the lock.

## 6. Rows (first match) and what each leads to

1. **Δa-VOID** — a guard failed, fewer than 12 of L's 16 slots were kept (or fewer than 8 for
   Λ2), a job did not complete, or the budget cap (§7) was reached before every declared measure
   completed. Fixed under its own record; a rerun after any count was read uses fresh seeds named
   by a ruling.
2. **Δa-NOT-REPRODUCED** — J-ref hands over by tolerance on at least 4/8. Stage 0's approach
   failure does not reproduce on fresh seeds; escalate with the record (no component is named).
3. **Δa-PLANNER-BOUND** — O-pose hands over on fewer than 6/8. Even with the exact pose cost,
   this CEM at TASK-101's budget does not deliver the pose within 60 commands, so the planner (its
   budget, horizon, re-initialisation or the timeout) binds before any cost question. Leads to: a
   planner change first (warm start, more iterations, longer T₁), re-measured with O-pose.
4. **Δa-GEOMETRY** — O-img hands over on fewer than 6/8. A perfectly predicted image-subgoal cost,
   with the planner that O-pose shows can reach the pose, does not carry CEM to it: **the
   image-goal cost geometry is the limit**, and no model, LeWM included, can fix it by better
   prediction. Leads to: a goal or cost change for the approach, for example a proprioceptive or
   pose-readout cost (a readout of the arm pose from the latent, trained on corpus train data), a
   per-camera or re-weighted cost if Λ3 shows one camera carries the pose, or denser intermediate
   image subgoals along the demonstration path; Λ1's d_flat and Λ3 say which.
5. **Δa-PLANNER** — max(h(J-big16), h(J-big32)) ≥ 6/8. J's cost suffices given a larger planner.
   Leads to: an amendment of phase 1's planner budget (and horizon) in TASK-102, J unchanged.
6. **Δa-MODEL** — every J arm hands over on at most 2/8, and Δρ ≥ 0.20 (J ranks candidates by
   their end distance clearly worse than the true-frame cost does at far states). The true-frame
   cost works, J's prediction of it does not. Leads to: a model change, for example R28.57's LeWM
   longer-budget retrain or a J retrain, with the image goal and planner kept.
7. **Δa-MIXED** — otherwise. Escalate with the record; no variant is admitted by default.

The thresholds are declared, not calibrated. With 8 seeds, 6/8 has an exact one-sided 95 % lower
bound of 0.40 and 2/8 an upper bound of 0.60, so a row names the binding component only coarsely.

## 7. Budget and evidence

**Cap: 2.0 GPU h of lock time** for every job of this diagnostic (goals, debug, Λ1, Λ2), read from
`~/.local/state/gpu/oej-gpu_run.log` (`--who oej:task102-approach-diag`). CPU jobs without the
lock are reported with their wall time. Outputs under the shared checkout's
`outputs/task102-approach-diag/`; the evidence (reports, summaries, logs, the GPU ledger lines)
under `~/develop/emai/evidence/task102-approach-diag/` with `SHA256SUMS`. Nothing earlier is
overwritten.

## 8. What this cannot show

- A development reading on 16 (Λ1) and 8 (Λ2) seeds, one run, simulation only. It is not a gate,
  and nothing here is a result for or against J, LeWM, the hybrid or the task.
- The oracle arms read simulator truth. They show what this planner can do with a perfect cost;
  they are not controllers and are never compared with learned arms as results.
- Λ1's line states are placed by setting joints, not reached by a controller; the demonstration
  path and Λ2's closed loops cover reached states.
- One goal frame per seed (the demonstration's subgoal 1), one encoder (DINOv2 g4), one latent
  error (mean squared difference over standardised tokens). A different goal frame, pooling or
  distance is not tested.
