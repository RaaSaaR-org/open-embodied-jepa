# TASK-089 results — zero-shot grasp-and-lift: offline diagnosis (Part A) and closed loop with the step-averaged cost (Part B)

Protocol: [zero_shot_grasp_v1.md](zero_shot_grasp_v1.md) (DRAFT #197; FROZEN at `c40919d`, #198;
Stage A at #199; Stage BD at #200 with an independent GO). Rulings on this record: DECISIONS
2026-10-11 (al), R26.31–R26.36, decided by Claude under owner delegation.

**This is a simulation result on G1's right arm and right Dex3, not an Apple→Plate experiment.**
**R7's canonical status sentence does not change** (R26.35).

## 1. Row: A-PASS(step-mean), then B-FAIL; the abandonment clause fires

- **Part A (offline, 64 fresh roots): A-PASS(step-mean).** G-87100's predictions separated a closing
  from an open hand and a lifting from a releasing or staying hand at the object on 64 of 64 roots at
  h = 8 and 16 (exact 95 % 0.944–1.000; bars 59–61 of 64 from the simulator's measured ceiling;
  action-shuffled control 27–33). With TASK-088's last-step cost a single CEM call at those states
  chose to close on 4 and to lift closed on 17 of 64; with the step-averaged cost on 62 and 61. So
  Part B ran with the step-averaged cost.
- **Part B (closed loop, 64 fresh gated resets): B-FAIL.** The primary arm **G-chg-87100** (G, CEM
  with the step-averaged cost) lifted **0 / 64** (0.000–0.056) against the bar of 26 / 64 and did not
  beat random (0 / 64; no discordant reset, p = 1). By §8–§9 the row is **B-FAIL** and **the
  abandonment clause fires** (R26.33).
- **Reported, not the row:** the same planner with G's two other model seeds lifted **8 / 64**
  (87101; 0.056–0.232) and **7 / 64** (87102; 0.045–0.212). These are the first world-model
  grasp-and-lifts on gated resets in this project, but they are replication runs that cannot change
  the row, and both are far below the bar.

## 2. Numbers (Stage BS, gated)

64 resets 891000–891063 (75 rejected demonstrator draws; targets can 32, cube 23, apple 9, banana
0), every arm on the same resets (paired), at `c40919d` from a clean tree, one GPU lock per run,
2026-10-11 05:07–10:46 UTC. Success: the target at least 5 cm above its start height and in grasp
contact for 20 consecutive commands (TASK-088's scorer). Exact 95 % intervals.

| Arm | grasp-and-lift | 95 % | by object (can / cube / apple) | grasp contact | pre-grasp left by "reached" | stopped | share of commands > 0.5 (grasp, lift phase) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **G-chg-87100 (primary)** | **0 / 64** | 0.000–0.056 | 0 / 0 / 0 | 39 | 45 | 8 | 0.70, 0.79 |
| G-base-87100 (TASK-088's cost) | 0 / 64 | 0.000–0.056 | 0 / 0 / 0 | 12 | 0 | 8 | 0.00, 0.03 |
| P-chg-87100 | 0 / 64 | 0.000–0.056 | 0 / 0 / 0 | 5 | 4 | 4 | 0.11, 0.42 |
| G-chg-87101 (replication) | 8 / 64 | 0.056–0.232 | 7 / 0 / 1 | 43 | 47 | 8 | 0.41, 0.73 |
| G-chg-87102 (replication) | 7 / 64 | 0.045–0.212 | 5 / 0 / 2 | 40 | 49 | 15 | 0.51, 0.70 |
| ik (ceiling) | 48 / 64 | 0.626–0.850 | 27 / 19 / 2 | 61 | 63 | 4 | 0.87, 1.00 |
| ik-nopress | 46 / 64 | 0.592–0.824 | 26 / 19 / 1 | 61 | 63 | 5 | 0.87, 1.00 |
| hold | 0 / 64 | 0.000–0.056 | | 0 | 0 | 0 | |
| random | 0 / 64 | 0.000–0.056 | | 0 | 0 | 0 | |

- Paired (exact one-sided McNemar, reported): G-chg-87100 against G-base, P-chg and random: no
  discordant reset (p = 1); ik against G-chg-87100: 48 to 0 (p = 3.6 × 10⁻¹⁵). Reported only:
  G-chg-87101 and G-chg-87102 against random 8 to 0 and 7 to 0 (p = 0.004 and 0.008); they share 3
  successful resets (12 resets in their union), and the IK follower lifted on 9 of those 12.
- The primary arm's highest rise was 3.5 cm (reset 891031); the replications' successes raised the
  target by 6.8–11.8 cm at most. Their 20-command lift dwell completed at commands 157–347. No world-model arm left the grasp subgoal
  by "reached" except G-chg-87101 (6) and G-chg-87102 (5); ik did on 29.
- Stopped episodes were the measured-velocity stop or "projection infeasible" (G-chg-87100 7 and 1,
  G-chg-87102 14 and 1, ik 4 and 0). G-87102's checkpoint carries TASK-087's `last_two_triggered`
  flag; 87100's and 87101's do not.
- Planning about 0.97–1.01 s per command with 6 processes; world-model runs 3 623–4 321 s each.
- **Bar:** 0.40, 26 / 64, from ik's Stage BD rate 28 / 32 (min(0.40, 0.9 × 0.875); the declared
  value, ik's rate did not bind).

Stage A and Stage BD are recorded in the protocol (§17, §18). In short: A1 passed for G at both
horizons (P-87100 passed at h = 8 only); A2 selected the step-averaged cost; Stage BD (32
development resets) gave G-chg 2 / 32, G-base 0, P-chg 0, ik 28.

## 3. What the numbers say, and what they do not

- **The models see the grasp, and the changed cost makes the planner choose it.** Offline, G's
  roll-outs rank closing and lifting above the alternatives as reliably as the real frames do, and
  with the step-averaged cost the planner commands a closed hand on the grasp and lift subgoals
  (70–79 % of G-chg-87100's commands there, against 0–3 % with TASK-088's cost) and reaches the
  pre-grasp subgoal far more often (45 against 0 of 64). Part of A1's separation is the robot's own
  predicted joints (G's pose term), not the object.
- **It still does not grasp and lift reliably.** With the primary seed the hand closed on the target
  (grasp contact on 39 of 64) but lifted nothing; with the other two seeds it lifted 8 and 7. The IK
  follower, given the same goal poses and no world model, lifted 48. So the remaining gap is in
  executing a stable grasp (where the palm is while the hand closes and the arm rises), not in
  choosing to close. Why is not tested here.
- **Seed matters.** The gated row rests on one model seed (87100), which here was the weakest of the
  three; the protocol fixed it before any closed loop. Summed over the three seeds the planner lifted
  15 of 192 (reported only).
- **"LeWM needed" is not shown:** the IK follower does far better with the same goal information.
- Simulation only, one scene, one camera at 112 px, one encoder and latent, one corpus, goals from a
  scripted demonstrator (biased to layouts a scripted pick can do; no banana), one planner setting.

## 4. Run and evidence

- Stage A: `~/develop/emai/evidence/task089-a/` (`SHA256SUMS` sha256 `9b4e8953…6c98`). Stage BD:
  `~/develop/emai/evidence/task089-bd/` (`b72db5d7…4e60`). Stage BS:
  `~/develop/emai/evidence/task089-bs/` (`SHA256SUMS` sha256 `7266621e…d1da`; `summary-bs.json`
  sha256 `d8209868…4ed7`). Stage 0 and A-K0: `~/develop/emai/evidence/task089-stage0/`; the
  pre-draft prototype: `~/develop/emai/evidence/task089-prototype/`.
- Replays (`~/develop/emai/evidence/task089-bs/videos/`, own `SHA256SUMS`), each replayed from the
  stored commands and checked against the run's palm distances (all matched):
  - `task089-gated-grasp-G-chg-87100-891031-failure.mp4` — the primary arm's failure that came
    closest (largest rise, 3.5 cm), gated;
  - `task089-gated-grasp-G-chg-87101-891016-success.mp4` — a replication success (the one whose
    success step is nearest the replication's median), gated cohort, reported arm;
  - `task089-dev-grasp-G-chg-87100-89118-success.mp4` — a Stage BD (development) success of the
    primary model, not gated.
- Every GPU use ran under `scripts/gpu_run.sh --wait --min-free-gib 8 --board`, one job per run;
  the replays ran under their own locks between runs.

## 5. What changes

- **R26.31 — the run** as above; no refusal, error or re-run; the void checks found nothing.
- **R26.32 — the task's row is B-FAIL** (after A-PASS(step-mean)): G-chg-87100 0 / 64 against the
  bar of 26 / 64, not above random.
- **R26.33 — the abandonment clause fires** (§9): CEM planning of grasp-and-lift over TASK-087's
  checkpoints (trained on `play-v1` as collected) is closed; no further planner, cost, horizon or
  action-parametrisation variant over these checkpoints is preregistered for grasping without new
  evidence of a different kind. The replication seeds' 8 and 7 of 64 are recorded with the clause,
  not as a way around it. Not closed: LeWM, DINOv2, `play-v1` as a base, CEM in general, reach
  planning (TASK-088's Z3-REACH stands), Phase 4 as a goal, and the product goal.
- **R26.34 — recommendation only.** Phase 4 is not admitted. The next grasp step changes the data or
  the model, as §9 says: for example more grasp-and-lift in the play corpus (only about 10 % of
  `play-v1`'s episodes lift anything, and within 16 commands even the demonstrator's lift raised the
  object 5 cm on only 21 of 64 roots), or a model trained to predict the grasp's outcome (contact and
  object rise) over the planning horizon; the step-averaged cost and this task's offline probes can
  be reused as the planner and the diagnostic. It needs its own preregistration.
- **R26.35 — R7 does not change.** Not Apple→Plate.
- **R26.36 — TASK-089 closes.**
