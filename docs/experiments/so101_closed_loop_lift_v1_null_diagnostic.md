# TASK-101 Δ: the S0-NULL diagnostic on cohort X (record)

**Status: development diagnostic record (§18.6 of [so101_closed_loop_lift_v1.md](so101_closed_loop_lift_v1.md),
R28.72). Row: Δ-MIXED, escalated with this record.** It is not a gate and makes no claim. It ran
because Stage 0′ ended S0-NULL ([record](so101_closed_loop_lift_v1_stage0b.md)). Simulation only,
`so101_cube_lift_v1`, not Apple→Plate. R7 does not change. Recorded by Claude under owner
delegation (DECISIONS 2026-10-19 (bw), R28.79).

## What ran

- **Cohort X 101700–101715** (reserve 101716–101731 unused): the v1 goal demonstration (follower
  v1, §4's rejection rule) kept 16 of 16 drawn seeds. Goal store `82a67c57…`.
- **Configuration H8-mean**, the configuration §7.3's rule chose on K′ (all four tied at 0/32).
- **States:** on each seed, the demonstration is replayed with rendering (its executed actions
  must equal the goal store's, a guard; they did on all 16). At the states after commands
  t ∈ {0, 10, 20, 30} (toward subgoal 1, the pre-grasp pose) and t ∈ {40, 50, 60} (toward subgoal
  2, the descent) the planner is **queried, not executed**, with the demonstration's own subgoal
  latent as the goal: 112 states.
- **Candidates per state:** a_demo, the demonstration's next 8 executed commands; a_cem, W's CEM
  best (256 × 6, planner seed 10108 · 10⁶ + 1 000 · (seed − 101700) + t), projected; and 255 random
  candidates uniform on [−1, 1], projected by §5.2's projector (`default_rng([10108, seed, t])`).
  Each is scored by J-100101's step-mean latent cost as W scores it. At H = 8 no a_demo window
  crosses a subgoal boundary, so the "without boundary windows" reading equals the full one.
- **m3 (reported only):** in a fresh copy of the simulation run to the same state, a_cem's 8
  requested commands are executed open-loop (each executed action equalled the projected a_cem,
  112/112), and the arm RMS to the subgoal's joints is compared before and after.
- **Budget:** one GPU job, 36 s of compute and 66 s of lock time, against the 1.5 GPU h allowance.
  Revision `88a014b`, the same as K′'s.

## Measures and row

| reading | states | m1: J(a_demo) < J(a_cem) | m2: median normalised rank of a_demo among 255 random | m3: a_cem lowers the arm RMS (reported) |
|---|---|---|---|---|
| **all (the rows read this)** | 112 | **0.116** (13/112) | **0.000** | 0.509 (57/112) |
| subgoal 1 (t = 0–30) | 64 | 0.188 | 0.000 | 0.406 |
| subgoal 2 (t = 40–60) | 48 | 0.021 | 0.137 | 0.646 |

**Rows, first match (§18.6):** Δ-VOID no (no guard failed, 16 ≥ 12 kept, within budget); Δ-MODEL
no (m2 = 0.000 < 0.25); Δ-PLANNER no (m2 ≤ 0.10, but m1 = 0.116 < 0.50); **Δ-MIXED.** Per §18.6:
escalate with the record; no variant is admitted by default.

Per state (medians over the 16 seeds; costs are J's step-mean latent error):

| t | subgoal | m1 | rank of a_demo | J(a_demo) | J(a_cem) | best random | median random | m3 | arm RMS before → after a_cem (rad) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 0.375 | 0.000 | 1.114 | 1.102 | 1.157 | 1.261 | 0.94 | 0.838 → 0.721 |
| 10 | 1 | 0.000 | 0.000 | 0.517 | 0.513 | 0.529 | 0.562 | 0.69 | 0.188 → 0.163 |
| 20 | 1 | 0.188 | 0.000 | 0.048 | 0.043 | 0.073 | 0.147 | 0.00 | 0.000 → 0.048 |
| 30 | 1 | 0.188 | 0.000 | 0.048 | 0.044 | 0.076 | 0.145 | 0.00 | 0.000 → 0.055 |
| 40 | 2 | 0.000 | 0.388 | 0.946 | 0.769 | 0.833 | 0.963 | 0.94 | 0.242 → 0.152 |
| 50 | 2 | 0.000 | 0.137 | 0.687 | 0.598 | 0.635 | 0.746 | 0.88 | 0.164 → 0.118 |
| 60 | 2 | 0.063 | 0.002 | 0.437 | 0.430 | 0.447 | 0.499 | 0.13 | 0.080 → 0.122 |

## A development reading (not a row; nothing selects on it)

- **J's cost does rank commands relative to random ones.** On subgoal 1 the demonstration's
  commands beat every random candidate at the median state. At t = 40, the first step of the
  descent, they rank only 0.39, near chance.
- **CEM is not failing to minimise J's cost; it finds commands J scores as cheaper than the
  demonstration's** on 99 of 112 states. That is the opposite of Δ-PLANNER's premise. The
  margins are small on subgoal 1 (about 1–10 %) and larger at t = 40–50 (about 13–19 %).
- **Where the arm is already at the subgoal pose (t = 20, 30: the demonstration waits at the
  pre-grasp pose, RMS 0.000), CEM's cheaper commands move it away** (to 0.048–0.055 rad, above the
  0.04 tolerance) and still score lower than staying put. From farther away (t = 0, 10, 40, 50),
  CEM's commands mostly move the arm toward the subgoal open-loop (m3 0.69–0.94). So the cost's
  minimum near the goal pose lies off the goal pose, at a distance of the order of the switching
  tolerance. That is consistent with K′'s closed loop coming within 0.26–0.48 rad in median but
  never holding 0.04 rad for 3 commands. It is a reading of 16 seeds, from states on the
  demonstration's own path, which §18.6 notes favours the model.
- **What Δ cannot show (§18.6):** it is a development reading on 16 seeds with declared, not
  calibrated, cut-offs; it scores open-loop queries, not closed loops; it uses one configuration.

## Evidence

`outputs/task101-stage0b/delta/delta.json` (`5fd62bed…15b0`, every state's costs, ranks and
branch readings) and `X-goals/` (`82a67c57…`); copied with the logs into
`~/develop/emai/evidence/task101-stage0b/` (`SHA256SUMS` `e29b2cd3…fbe71`).
