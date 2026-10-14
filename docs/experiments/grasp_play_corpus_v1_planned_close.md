# TASK-091 — the planned close within the plan: TASK-090's A′ on fresh roots with A2 redefined, then TASK-090's closed-loop grasp-and-lift comparison on fresh resets

**Status: FROZEN at the merge of its Stage 0 PR** (R27.45–R27.60, decided by Claude under owner
delegation, 2026-10-13; DECISIONS (as) and (at); the DRAFT was reviewed and merged at #209,
`940de4e`; Stage 0's record is §14). After that merge nothing in §2–§11 changes without its own
reviewed ruling. Stage 0 changed none of the three items §12 allows it to change (R27.56): the
process counts stay, the BS split stays at two 64-reset jobs per world-model run, and §5.3's third
deferral measure is kept. **Stage A″ ran once and ended A″-PASS** (§15, R27.61–R27.64);
Stage BD ran once and set the bar at 0.40 (§16, R27.65–R27.68). **Stage BS ran once, after an
independent GO, and ended B-FAIL; the §7 clause fires and TASK-091 closes** (§17, R27.69–R27.73;
one run, simulation only).

This is the next task that R27.43 recommends. TASK-090 ([grasp_play_corpus_v1.md](grasp_play_corpus_v1.md))
closed at **A′-FAIL** (R27.34, R27.45): escalate, no clause. Its Stage BD and Stage BS were never
run, so no closed loop ran and nothing compared the two corpora on grasping. The development
diagnosis D-A2 (§20 there, R27.42) then read **D-TIMING**. That reading is development only, not
gated: one run, offline, simulation only, on 64 roots 905000–905063. In it, the three A2-failing
models (G-v2g-90100, G-v1p-90100, G-v1p-90102) mostly did not close on the first planned command, but mostly planned the close at step 2–4 of the 8-step plan, stably across five CEM seeds (largest spread
5 / 64). All nine G models closed by plan step k ≤ 4 on 64 / 64 roots.

**This is not an Apple→Plate experiment.** It plans grasp-and-lifts of the play scene's objects
with G1's right arm and right Dex3, in MuJoCo only. **R7's canonical status sentence does not
change, whatever row this task ends in** (R27.54).

## 1. Question

TASK-090's question, unchanged (its §1): with TASK-087's recipe, latent and budget and TASK-089's
planner and cost held fixed, does a model trained with **grasp-enriched** play episodes (M-v2g)
grasp and lift more often in closed loop than one trained with an equal number of **unchanged**
play episodes (M-v1p)? And does it reach the plan's bar?

What changes is only the offline no-regression gate in front of the closed loop. A2 counted only
the **first** planned command. But every model plans the close by step ≤ 4 (D-A2, development).
A receding-horizon controller re-plans every command and executes only the first. Whether it then
actually closes and lifts is exactly what the closed loop measures. So the offline gate checks
the **planned close within the plan**, and the closed loop decides (R27.46).

## 2. What changes from TASK-090, and what does not

| Item | TASK-091 | Source |
| --- | --- | --- |
| Corpora, featurisation, feature stores | TASK-090's, unchanged (`play-v1`, play-v2g, play-v1p; `data/task090-features-{v2g,v1p}` and TASK-087's store) | TASK-090 §2, §18, §19.1 |
| Models | TASK-090's checkpoints and TASK-087's G, unchanged, **no retraining** (§3) | TASK-090 §3, §19.2 |
| Pinned projection, pinned `play-v1` moments, λ = 2.0552 | unchanged | TASK-090 §2.2 |
| Planner | TASK-089's G-chg, unchanged: `CEMPlanner` H = 8, 300 samples, 10 iterations, 30 elites, min std 0.05, MPC every command executing the first command, left arm fixed, step-averaged cost | TASK-090 §2.2 |
| Scene, goals, subgoal switching, budget, stop rule, success scorer | unchanged | TASK-090 §2.2 |
| Stage C, Stage T, Stage E | **not re-run**; C-PASS, T-DONE and E-PASS are carried over as recorded for these checkpoints | TASK-090 §18, §19 |
| Offline gate | **Stage A″** (§4): A′ on fresh roots with **A2's close bar redefined** | new |
| Closed loop | Stage BD, GO and Stage BS as TASK-090 §7, **on fresh resets** (§5) | TASK-090 §7 |
| Reported only | adds receding-horizon deferral measures (§5.3) | new |
| Rows | A″-FAIL, then TASK-090's B rows (§6) | TASK-090 §7.2, §8 |
| Per-job cap | 7 200 s on every GPU job (the figure of R27.24's training cap, extended here); BS world-model runs split into two 64-reset jobs; a cap hit is B-VOID in BS and A″-VOID in A″, and in BD the job is recorded and re-run on fresh development seeds (§5.4, §6) | new |
| Abandonment clause | TASK-090 §9 in its scope, on B-FAIL only (§7) | TASK-090 §9 |

No planner knob, no cost term, no horizon and no λ changes. That holds now and after the freeze,
whatever Stage BD shows (R27.46, R27.50).

## 3. Models (reused, not retrained)

| Model | Corpus | Checkpoints | Role |
| --- | --- | --- | --- |
| **M-v2g** | play-v2g | G-v2g-90100, G-v2g-90101, G-v2g-90102 | **primary** (all three gated, pooled) |
| M-v1p | play-v1p | G-v1p-90100, G-v1p-90101, G-v1p-90102 | size control (gated comparison) |
| M-v1 | `play-v1` | TASK-087's G-87100, G-87101, G-87102 | reported only; in Stage BS 87100 alone (R27.25) |
| N-v2g, N-v1p | each corpus | N-v2g-90100, N-v1p-90100 | A1's no-action twins, reported |

These are the nine G checkpoints and two N checkpoints that TASK-090's A′ and D-A2 ran. Their
sha256 are those recorded in TASK-090 §19.2 (for example G-v2g-90100 `6ca90525…c375`,
G-v1p-90100 `1a8a2991…63b2`; the full hashes are in each `report.json` and in
`~/develop/emai/evidence/task090-t/`), and TASK-087's in its record. TASK-090's loader guards
apply unchanged: the implementation hash, the pinned moments (sha256 `1e960227…81b5c`) and the
feature-report pins (`play-v1` `f5648a35…7eae`, play-v2g `d8afb236…e988`, play-v1p
`00625a44…822e`). The `last_two_triggered` flags (G-v2g-90102, G-v1p-90100, G-v1p-90101) are
reported, as in TASK-090 §2.2.

Pairing, pooling and M-v1's role are as in TASK-090 §3.

## 4. Stage A″ — TASK-090's A′ on fresh roots, with A2's close bar redefined

**What runs.** TASK-090 §6.2's Stage A′ exactly, for every model of §3 (TASK-089 §5.1–§5.5's code
via TASK-090's frozen runtime), on **64 fresh roots, 906000–906063**, drawn by A′'s root sampler.
That covers A1 with G's cost at h = 8 and 16, A2 with the step-averaged cost, the action-shuffled
control (salt 9002), and the reported diagnostics: the readability probe (salt 9004), latent-only
wins and the lift branch's rise. A-K0 is not re-run: A1's bars are A-K0's frozen `A1_BARS`
(61 / 64 on each gated comparison at h = 8). The ceiling on A″'s own roots is computed and
reported only, as at A′.

**A2, redefined.** The A2 CEM call is unchanged: one call at each R1 and R2 root, CEM seed
`cem_seed(root) = 8803 · 10⁷ + root`, step-averaged cost and λ = 2.0552. What changes is that the
whole normalised 8-step plan is kept, not only its first command, as D-A2 did. Its first command
must equal `first_planned`'s bitwise on every root. If it does not, A″ stops: row A″-VOID (§6).
"Closes" means the plan's right-grasp channel (`GRASP_DIM`, index 13) is > 0.5.

- **close≤4@R1 (the redefined close bar):** the A2-seed plan at R1 closes at some plan step
  k ≤ 4. **Bar: ≥ 48 / 64 roots.**
- **lift@R2 (kept, unchanged):** the first planned command at R2 has right grasp > 0.5 and right
  palm dz (dimension 8) > 0. **Bar: ≥ 48 / 64.**
- **close@R1 (reported only):** the first planned command at R1 closes. This was A′'s close bar.
- Also reported: the first closing step k at R1 per root (a plan that never closes has no k), the
  share of roots closing by k ≤ 2, 4 and 8, and the mean right-grasp channel per plan step at R1,
  as D-A2's (b) and (d).

**Pass rule.** A″ passes for a model when it meets TASK-089 §5.5's A1 pass rule at h = 8 (as A′)
**and** both bars, close≤4@R1 and lift@R2.

**A″-FAIL** has the same rule as A′-FAIL. It applies when M-v2g fails A″ on two or more of its
three seeds, **or** M-v1p fails A″ on two or more of its three seeds. The task then escalates and
no clause fires. Below that, a seed that fails A″ alone stays in its arm's pooled count, and its
A″ result is reported beside its closed-loop count. M-v1's A″ results are reported only.

**Disclosed: the threshold k ≤ 4 was chosen after seeing D-A2's development numbers** on roots
905000–905063 (TASK-090 §20). On those roots all nine G models closed by k ≤ 4 on 64 / 64. For
that reason:

- A″ uses fresh roots, disjoint from A′'s and D-A2's;
- A″ is a **no-regression gate only**. It checks that, on fresh roots, the planner still plans a
  close early in the plan at R1 and still lifts at R2. It is not evidence that the models grasp,
  and a pass is expected from the development numbers;
- no claim of this task rests on A″.

**Asymmetry, disclosed.** The close bar now reads the plan, but lift@R2 still reads the first
command. That is deliberate: on A′'s roots every new model met lift@R2 (55–63 of 64), and only the
close bar failed. Redefining only the bar that D-A2 diagnosed keeps the change to what the
diagnosis supports. TASK-087's G-87101 failed A′ on lift@R2 (37 / 64; 39 at D-A2) and is
reported only.

## 5. Closed loop — TASK-090 §7 on fresh resets

### 5.1 Arms, stages, bar and rows: as TASK-090 §7

Stage BD, the GO and Stage BS run exactly as TASK-090 §7.1–§7.2 states them, with these
consequences:

- **Arms.** The same arms: M-v2g ×3, M-v1p ×3, M-v1 (87100 only in Stage BS, R27.25), ik,
  ik-nopress, hold and random.
- **Bar.** bar = min(0.40, 0.9 × r), with r ik's success rate on this task's BD. If r < 0.30, the
  bar is not set and the task escalates.
- **Pooled bar and per-seed floor.** The pooled bar is 3 × ⌈bar × n⌉ and the per-seed floor is
  ⌈⌈bar × n⌉ / 2⌉; at bar 0.40 and n = 128 they are 156 / 384 and 26 / 128.
- **Gating tests.** The reset-level exact one-sided sign-flip tests at p < 0.01, "M-v2g beats
  M-v1p" and "M-v2g beats random". The pooled McNemar and the reset-cluster bootstrap (salt 9006)
  are reported.
- **Rows.** B-VOID, B-PASS, B-COMP, B-SIZE, B-FAIL (with δ = +0.10) and B-UNDER, first match.
- **Reported with every row.** TASK-090 §7.2's list, including the determinism re-run of
  M-v2g-90100 on the first four BS resets, which must match bitwise. A mismatch is B-VOID.
- **"LeWM needed" is not claimed by any row.**
- **Phase 4.** B-PASS admits a Phase 4 preregistration; it does not preregister it.

### 5.2 Seeds and size

- **Stage BD (development, not gated):** 32 fresh resets, **90300–90331**. Its arms are
  TASK-090's BD arms: ik, ik-nopress, hold, random, seed 90100 of M-v2g and of M-v1p, and
  M-v1-87100.
- **GO.** An independent reviewer posts GO (or NO-GO) on BD's record before Stage BS. After a
  NO-GO, BD is recorded again on fresh development seeds named by a ruling.
- **Stage BS (gated):** **n = 128** fresh resets, **907000–907127**. n stays at 128, the size
  TASK-090's power rule chose (R27.22: power 0.974 at SD 1 logit effects at the declared
  alternative). The power estimate is not recomputed, because neither the alternative nor the test
  has changed. The determinism re-run uses 907000–907003.
- TASK-090's declared cohorts 90200–90231 (BD) and 904000–904127 (BS) were never run. They are not
  reused here. They stay unused, so no reader can confuse the two tasks' resets.

### 5.3 Added, reported only: receding-horizon deferral

The risk of R27.46's rationale (§9) is that a plan that closes at step 3 on every re-plan may put
the close off indefinitely in closed loop. To expose that, every world-model episode also records
the following. They are reported per arm and seed, with their distribution over resets.

1. **Closing commands before first contact.** The number of executed commands whose right-grasp
   channel is > 0.5, counted before the first grasp contact with the target (thumb plus index or
   middle, TASK-085's contact definition). Episodes with no contact are counted to their end and
   flagged.
2. **First executed close in the grasp phase.** The command index, counted from the start of the
   grasp subgoal phase, at which the executed right-grasp channel first exceeds 0.5. An episode
   that never closes in the grasp phase, or never reaches it, is reported as such.
3. **The planned close step on each re-plan.** For each re-plan in the grasp phase, the first plan
   step k at which the plan's right-grasp channel exceeds 0.5 (none if the plan never closes).
   Summary: the share of re-plans with k = 1, the median k, and the longest run of consecutive
   re-plans with k > 1. Stage 0 implements this only if the full plan can be read without changing
   the planner's computation; the first executed command must stay bitwise identical. Otherwise
   it is dropped, and the drop is recorded.

These measures are not gated, and no row reads them. They let a reader see whether a seed's
closed-loop count tracks deferral. They are reported for the same reason in Stage BD, where they
may show the risk. BD by itself changes no knob (§2).

### 5.4 Per-job cap

Every GPU job is limited to 7 200 s: the figure of R27.24's training-job cap (TASK-087's rule), extended here to every GPU job. One BS
world-model run over 128 resets projects to about 2.2 GPU hours (R27.25), above the cap. So each
BS world-model run is split into two jobs of 64 resets, 907000–907063 and 907064–907127. They use
the same arm, model and frozen code, and their outputs are combined into one run before any
summary. TASK-089's 64-reset world-model runs took 3 623–4 321 s. A job that reaches the cap stops,
and its run did not complete under the rules, which is B-VOID (TASK-090 §7.2, row 1). Stage 0's
smoke times per-command planning and may split the jobs further (§12). A cap hit in Stage BD,
which is not gated, is recorded, and that job is run again on fresh development seeds named by a
ruling, under the same rules, as after a NO-GO; it is not a B-VOID of Stage BS.

## 6. Rows of the task, in order

| Row | When | Next step |
| --- | --- | --- |
| A″-VOID | a check of §4 fails (checkpoint, implementation hash, pinned moments, feature report, λ, seed, or the first command against `first_planned`), a job did not complete under the rules (including reaching the 7 200 s cap), or a root set was run twice | fix under its own ruling; re-run on fresh roots named by that ruling |
| A″-FAIL | §4's rule | escalate; no clause |
| B-VOID | TASK-090 §7.2 row 1 | fix under its own ruling |
| B-PASS | TASK-090 §7.2 row 2 | admits a Phase 4 preregistration |
| B-COMP | TASK-090 §7.2 row 3 | escalate; no clause |
| B-SIZE | TASK-090 §7.2 row 4 | escalate; no clause |
| B-FAIL | TASK-090 §7.2 row 5 | the clause fires (§7) |
| B-UNDER | TASK-090 §7.2 row 6 | escalate; no clause |

- After any VOID, the stage runs again only on fresh seeds named by its ruling. A voided cohort is
  never re-run.
- Escalations go to a ruling in `docs/DECISIONS.md`. Each closes the task with a recommendation
  unless the ruling says otherwise.

## 7. Abandonment clause

TASK-090 §9 is carried over with its scope unchanged, as follows.

**Fires on B-FAIL only.**

**It closes:** adding more scripted grasp-and-lift play to `play-v1` (e9's pick recipe run more
often, held longer or retried, as `play_policy_grasp_v0` does) as the grasp fix, under TASK-087's
recipe and latent and TASK-089's step-averaged planner. After it fires, no further variant of this
kind is preregistered for grasping without new evidence of a different kind. That covers the skill
mix, mode mix, hold length, retries, target weighting or added amount of this collector.

**The next grasp step is then a model change.** For example, a model trained to predict the grasp's
outcome (contact and object rise) over the planning horizon, the model-first design of TASK-090
§15. It needs its own preregistration.

**Not closed:**

- LeWM, DINOv2 and CEM;
- `play-v1` and play-v2g as bases;
- other kinds of data: real G1 data, a different demonstrator or pick recipe, other sensors;
- model changes;
- reach planning (TASK-088's Z3-REACH stands);
- Phase 4 as a goal, and the product goal.

**It does not fire on** A″-VOID, A″-FAIL, B-VOID, B-COMP, B-SIZE or B-UNDER. TASK-090's A′-FAIL
fired nothing (R27.45), and this task's own A″ result fires nothing either. M-v1's counts and
the replication of TASK-089's seeds are recorded with the clause, not as a way around R26.33.

## 8. Seeds and salts

| Use | Seeds |
| --- | --- |
| Model seeds | none new: TASK-090's 90100–90102 and TASK-087's 87100–87102, as trained |
| Stage A″ roots | **906000–906063** |
| Stage BD resets | **90300–90331** |
| Stage BS resets | **907000–907127** (two jobs: 907000–907063, 907064–907127) |
| Determinism re-run | 907000–907003 (inside BS) |
| Debug (Stage 0 plumbing and smoke; never counted) | **90700–90799** |

**Salts.** There is no new salt. Layouts and goals use TASK-088's 8801 and 8802, CEM 8803 and
random 8804, with the new seeds, as TASK-089 and TASK-090 did. A″'s shuffled deal uses 9002, the
readability probe 9004 and the reset-cluster bootstrap 9006, all TASK-090's. A root's or reset's
rejected draws are further attempts under the same seed, so each range is the whole range used.

**Checked (DRAFT, 2026-10-13).**

- **Remote branches.** Every 5- and 6-digit number from 90000 to 999999 was listed on all 104
  remote branches, over `src/`, `scripts/`, `docs/`, `.mc/`, `tests/`, `benchmarks/` and
  `configs/`. None of them falls in 906000–906063, 907000–907127, 90300–90331 or 90700–90799. The
  closest ranges are A′'s 903000–903063, BS's declared 904000–904127, D-A2's 905000–905063 (with
  905064 and 905999 as end bounds of its check) and the debug range 90900–90999. No declared range
  spans the new ranges: the largest 6-digit range below them is TASK-077's labels 900000–901999.
- **Evidence.** In `~/develop/emai/evidence/` (14 GB; `*.json`, `*.txt`, `*.log`, `*.md`,
  `*.csv`, `*.yaml`, `*.py`, `*.sh`), the only tokens in these ranges are decimal fractions
  (for example `0.907226` in a TASK-080 dry-run report, `4.90724` in a TASK-080 smoke report) and
  update counts (`90300`, `90700` in TASK-077's training curves). None is a seed.
- **Stage 0.** Stage 0 repeats the check on the remote branches and in the evidence at its own
  revision, and records it.

## 9. What this cannot show, and its risks

- **Everything of TASK-090 §11** still applies: one scene, arm, hand and camera at 112 px,
  simulation only; one encoder, latent, recipe, budget and planner; the same privileged scripted
  pick recipe; the bundle; pooled seeds; cohort bias; no object, contact or tactile input; the
  velocity stop; undertraining. The manipulation is weaker than designed: every restatement of a B
  row quotes Stage C's C1 2.38 and C2 1.69 beside it.
- **Receding-horizon deferral (the named risk).** A plan that closes at step 3 on every re-plan
  may put the close off indefinitely in closed loop, because only the first command is executed
  and the next plan may again close at step 3. A″ cannot see this. Stage BD shows it, and Stage BS
  counts it (§5.3). **No planner knob may change after the freeze** to counter it, not even if BD
  shows it: that would be a planner change, which R27.46 excludes and which R26.33's clause on
  TASK-087's checkpoints also bears on. If deferral sinks the closed loop, the row records it.
- **The deferral risk is not balanced across the arms.** Two of M-v1p's three seeds
  (G-v1p-90100, G-v1p-90102) and one of M-v2g's (G-v2g-90100) failed close@R1 at A′ and, in D-A2
  (development, not gated), planned the close later in the plan (G-v1p-90102 mostly at step 2, the
  other two mostly at step 3–4). If late planned closes lower closed-loop success, M-v1p's pooled count is hit
  on more seeds, which favours "M-v2g beats M-v1p" for a reason unrelated to the composition. A
  B-PASS or B-COMP would not separate the two causes. The per-seed counts and §5.3's per-seed
  deferral measures are reported so that a reader can see whether the comparison tracks the late
  seeds. They do not correct for it. Any restatement of a B-PASS or B-COMP quotes this caveat.
  The confound can also act the other way: if deferral sinks G-v2g-90100, it can push k_g below
  the pooled bar, giving B-FAIL (the clause fires) or B-UNDER for a planner-timing reason rather
  than a data reason. Any restatement of a B-FAIL or B-UNDER therefore quotes §5.3's per-seed
  deferral measures beside it.
- **A″'s threshold is post hoc.** k ≤ 4 was chosen after seeing D-A2's development numbers (§4).
  A″ passing on fresh roots is a no-regression check, not evidence.
- **First-command A2 did not predict closed loop before.** In TASK-089, G-87100 met the
  first-command A2 (62 and 61 of 64) and then lifted 0 / 64 in its gated closed loop (simulation
  only, one gated model seed; replication seeds 8 / 64 and 7 / 64, reported only). An offline plan
  measure, first command or within the plan, is a weak predictor of closed-loop grasping.
- **What a pass shows.** A B-PASS would be the first gated world-model grasp-and-lift at the plan's
  bar in this project, on these terms only. It would not show "LeWM needed", because the IK follower
  uses the same goal information and no world model.

## 10. Runtime, disk, paths and evidence

**GPU.** Every GPU use is one `scripts/gpu_run.sh --wait --min-free-gib 8 --board` job, with the
shared lock taken per job and the per-job 7 200 s cap (§5.4). Before each job, the `## Requests`
section of `~/.local/state/gpu/BOARD.md` is checked. If it has a `- v2d: …` line, the job starts
with `gpu_run --wait`, so that the V2D session's checkpoint export goes first, per the 2026-10-12
agreement. The V2D session is told in advance about any multi-hour stage (BD and BS).

**Estimates:**

| Step | Estimate | Basis |
| --- | --- | --- |
| Stage 0 (checks, debug smoke) | under 1 GPU h | TASK-090 §17: about 0.7 h with training |
| Stage A″ (roots, A1 and A2 for every model of §3, probes) | **under 1 GPU h** | A′'s roots and A′ took 0.12 h; D-A2 0.26 h |
| Stage BD (3 world-model runs × 32 resets plus baselines) | about 2 h | TASK-090 §12 |
| Stage BS (7 world-model runs × 128 resets as 14 jobs, plus baselines and the determinism re-run) | **about 15.5 GPU h** | R27.25 |
| **Total** | **about 19–20 GPU h**; about 3–5 days wall-clock with reviews | |

**Disk.** No new corpus, feature store or checkpoint. Goals, runs and MP4s are small, and they go
to `outputs/task091-*` on the data disk through the repository's link.

**Paths that must stay.** TASK-087's feature store and projection live in
`worktrees/task087-run/outputs/` (TASK-090 §17). **That worktree must not be archived or removed
until TASK-091 closes.** Neither may `checkpoints/task090-run/` or `data/task090-features-*`.

**Evidence.** Stored in `~/develop/emai/evidence/task091-*/` with `SHA256SUMS`. Existing evidence
under `data/`, `checkpoints/`, `outputs/` and `~/develop/emai/evidence/task090-*` is read and
never written.

## 11. Disclosed development before this draft

TASK-090's D-A2 (§20 there; R27.36–R27.44) is the only development behind this draft. It was
preregistered by ruling, its reading order was fixed before it ran, and it is recorded. It is
**development, not gated**: one run, offline, simulation only, 64 roots (905000–905063), at
`53f0b72`.

- close@R1: the three A2-failing models scored 0, 2 and 15 of 64; the other six scored 57–64.
- Close by plan step k ≤ 4: all nine models on 64 / 64.
- The CEM-seed spread for a failing model was at most 5 / 64.
- lift@R2 (reported): 56–63 for the six new models; TASK-087's 87101 scored 39.

§4's k ≤ 4 threshold was read off these numbers, and that is why A″ uses fresh roots and is a
no-regression gate only. No GPU job, simulation or training was run for this draft.

## 12. Stage 0, freeze and what Stage 0 may change

**Stage 0** is its own PR, merged before the freeze. It adds only opt-in modules and new scripts.
It changes no existing file, so TASK-087's implementation hash and TASK-088's, TASK-089's and
TASK-090's frozen code stay as they are. TASK-090's runners are hash-pinned by its protocol and
are not edited. It adds:

- **A new module** (NumPy at import), for example `grasp_corpus_wm_planned_close.py`. It holds
  TASK-091's cohorts and seeds (§8), the A″ measures and pass rule, the A″-FAIL rule, and the
  deferral summaries. For everything else it imports TASK-090's frozen `grasp_corpus_wm` rules
  (bars, sign-flip tests, bootstrap, rows).
- **A runtime wrapper.** It reuses `grasp_corpus_wm_runtime` and TASK-089's `zgr.wrap` and planner
  construction unchanged. It keeps the whole A2 plan, as D-A2's development script did, and checks
  the plan's first command against `first_planned` bitwise. It records §5.3's measures without
  changing any executed command.
- **Scripts** `scripts/{run,summarize}_task091.py` and `scripts/run_task091_stage.sh` (A″, BD and
  BS, with each BS world-model run split into jobs of 64 resets).
- **Tests:** the redefined A2 measure on synthetic plans, the A″ pass and fail rules, the seed
  ranges and their disjointness from TASK-090's, the deferral counters, the job split and recombine (on debug seeds, a split run's per-reset
  outcomes must equal an unsplit run's bitwise),
  and the core-import check.

**Stage 0 also runs:**

- §8's seed checks at its own revision;
- the hash checks: the nine G and two N checkpoints against TASK-090 §19.2's and TASK-087's
  reports, the feature reports and the projection, after the data-disk layout (TASK-090 §12, §17);
- a debug smoke on debug seeds (90700–90799) of A″ and of every closed-loop arm, including the
  bitwise first-command check and §5.3's measures, timed per command to confirm §5.4's job split.

**Stage 0 may change only:**

- the per-run process count;
- the BS job split (more, smaller jobs), if its timing projects a 64-reset job above 5 400 s
  (0.75 × the cap);
- whether §5.3's third measure is kept (§5.3).

Each such change is recorded. The protocol is frozen at the Stage 0 merge. Every later stage runs at
the merged revision from a clean tree.

## 13. Rulings this document records (R27.45–R27.54; DRAFT; decided by Claude under owner delegation)

- **R27.45** — **TASK-090 closes.** Its final row is **A′-FAIL** (R27.34): escalate, no clause.
  Stage BD and Stage BS were never run. The §9 clause did not fire, and nothing in its scope is
  closed or refuted. R7 does not change. D-A2's D-TIMING (development, not gated; R27.42) is the
  reason for the next task. TASK-090 moves to done in `.mc`.
- **R27.46** — **The next task is TASK-091**, R27.43's mapped recommendation, taking the
  "corrected A2 definition" branch, **not a planner change**. TASK-089's step-averaged CEM
  (λ = 2.0552, receding horizon, executing the first command) and everything else in TASK-090
  §2–§7 stay frozen, so no planner change confounds the composition comparison M-v2g against M-v1p (the deferral
  confound of R27.50 and §9 remains).
  Rationale: A2 counted only the first planned command, while every model plans the close by step
  ≤ 4 (D-A2, development). Whether a receding-horizon controller actually closes and lifts is
  exactly what the closed loop measures. So the gate checks the planned close within the plan, and
  the closed loop decides.
- **R27.47** — **Models and corpora are reused unchanged.** TASK-090's six G and two N
  checkpoints and TASK-087's three G checkpoints are used with no retraining, at the hashes
  recorded in TASK-090 §19.2 and TASK-087's record, under TASK-090's loader guards (§3). TASK-090's
  corpora and feature stores are reused too. C-PASS, T-DONE and E-PASS carry over and are not
  re-run.
- **R27.48** — **Stage A″** (§4). It is TASK-090's A′ on 64 fresh roots, 906000–906063, with A2's
  close bar redefined: the A2-seed plan's grasp channel (dim 13) > 0.5 at some plan step k ≤ 4, on
  ≥ 48 / 64 roots. close@R1 is reported, and lift@R2 ≥ 48 / 64 is kept. A1 and the A′-FAIL rule
  are unchanged, as A″-FAIL. The k ≤ 4 threshold was chosen after seeing D-A2's development
  numbers on 905000–905063. That is why A″ uses fresh roots, and why it is a no-regression gate
  only.
- **R27.49** — **The closed loop** (§5). Stage BD, the GO and Stage BS run exactly as TASK-090 §7:
  the same arms, bar rule, n = 128, reset-level sign-flip tests and rows B-VOID…B-UNDER. They run
  on fresh resets: BD 90300–90331 and BS 907000–907127, disjoint from TASK-090's declared but unrun
  90200–90231 and 904000–904127. The §9 clause is carried over with its scope unchanged, and fires
  on B-FAIL only (§7).
- **R27.50** — **Deferral is reported, and the risk is named** (§5.3, §9). Reported only: closing
  commands executed before first contact; the first executed close in the grasp phase; and, if
  Stage 0 can read it without changing the plan, the planned close step on each re-plan. The risk
  is that a plan closing at step 3 on each re-plan may defer the close indefinitely in closed loop.
  BD shows it, and BS counts it. No planner knob may change after the freeze. That the risk falls
  on more M-v1p seeds than M-v2g seeds is disclosed as a confound of the composition comparison.
- **R27.51** — **Seeds and salts** (§8). A″ uses 906000–906063, BD 90300–90331, BS
  907000–907127 and debug 90700–90799. There is no new model seed and no new salt. The uniqueness
  check over all remote branches and the evidence is recorded, and Stage 0 repeats it.
- **R27.52** — **Budget and process** (§10). A″ takes under 1 GPU hour, BD about 2 hours and BS
  about 15.5 GPU hours (R27.25). The 7 200 s figure of R27.24's training-job cap applies to every
  GPU job, so each BS world-model run is split into two 64-reset jobs (§5.4). The shared GPU lock applies, one
  `gpu_run.sh --wait` job each, and so does the V2D agreement: BOARD.md's Requests are checked
  before each job. `worktrees/task087-run` stays until TASK-091 closes.
- **R27.53** — **Status.** This protocol is DRAFT in this PR. A Stage 0 PR follows (§12) and freezes
  it at its merge, as TASK-090's did. Gated runs (A″, BD, BS) take place only after that freeze,
  BS only after an independent GO on BD's record.
- **R27.54** — **R7 does not change**, whatever the row.

## 14. Stage 0 record (R27.55–R27.60; decided by Claude under owner delegation)

Stage 0 ran on 2026-10-12/13 (23:14–00:03 UTC) in the worktree of the Stage 0 PR (branch
`feat/task091-stage0`, from `940de4e`), every GPU use one `scripts/gpu_run.sh --wait` job under
the shared lock, BOARD.md's Requests read before each job (no `- v2d:` line was present at any
of them). Evidence: `~/develop/emai/evidence/task091-stage0/` (`SHA256SUMS`; the smoke's outputs
are copied there from `outputs/task091-stage0/` on the data disk). **Everything below on debug
seeds is debug plumbing, not a result, and is not read as an outcome of any stage.**

**Code** (opt-in; no registry entry; `import embodied_jepa` stays torch- and MuJoCo-free, checked
by a test; no existing file changes, so TASK-087's implementation hash `e2ec0096…49b1` and
TASK-088's, TASK-089's and TASK-090's frozen code are untouched):

- `src/embodied_jepa/grasp_corpus_wm_planned_close.py` (NumPy at import): §8's cohorts and the
  debug split of 90700–90799 (roots 90700–90739, resets 90740–90799); the pins of the eleven
  checkpoints (§3, the full hashes of TASK-090 §19.2's table and TASK-087's reports), the three
  feature reports, the moments and λ; the sha256 pins of TASK-090's frozen files at `53f0b72`
  (`grasp_corpus_wm*`, `zero_shot*`, `planning.py`, `run_task090.py`, `summarize_task090.py`),
  checked before any job runs; A″'s measure (`a2_second`), pass rule and rows (A″-VOID, A″-FAIL,
  A″-PASS); the bitwise first-command check; the 7 200 s cap, the 5 400 s split trigger, the BS
  job split and the recombination of split runs (`combine_parts`); §5.3's deferral measures and
  their per-arm summary. The bars, sign-flip tests, bootstrap and B rows are TASK-090's
  `grasp_corpus_wm` functions, unchanged.
- `src/embodied_jepa/grasp_corpus_wm_planned_close_runtime.py`: the A2 plan (`zgr.wrap` and
  `zgr.make_planner(root, 8)`, the very call `first_planned` makes, keeping the whole plan) and
  the recording controller (TASK-089's `ChangeController`, built as TASK-090's `gr.controller`
  builds it, whose planner is wrapped by a recorder that returns the planner's plan object
  untouched and keeps its right-grasp channel per step).
- `src/embodied_jepa/grasp_corpus_wm_planned_close_run.py`: the parts of TASK-090's frozen runner
  that TASK-091's runner needs (its report and guards, `verify`, the feature pins, the encoded
  roots, the ceiling and the shuffled deal), copied, because the project's audit rule F16
  (`tests/test_no_runner_imports.py`) forbids a new runner to load another script. A test checks
  the copied shuffled deal against the frozen runner's on fresh draws.
- `scripts/run_task091.py` (`verify`, `seeds`, `roots`, `a-second`, `goals`, `run` with `--part`,
  `combine`, `video`), `scripts/summarize_task091.py` (stages `a`, `bd`, `bs`, and Stage 0's
  `timing`), `scripts/run_task091_stage.sh` (`a`, `bd`, `bs`) and
  `tests/test_grasp_corpus_planned_close.py` (19 tests, one graphics opt-in).

**§12's checks** (`verify/verify-task091.json`, all pass; at `7c49117`, clean tree):

- TASK-090's own §12 check, run by the copied code (`verify/verify-task090-checks.json`): the
  `data/`, `checkpoints/` and `outputs/` links resolve to the data disk; `play-v1` re-verifies
  (every shard re-hashed); TASK-087's feature report `f5648a35…7eae`, store files and projection
  `c196fc48…e33f`; its 18 checkpoints against their reports and the evidence copies; the pinned
  moments equal G-87100's buffers bitwise (`1e960227…81b5c`); TASK-087's implementation hash.
- TASK-090's eight checkpoints: each `model.pt` equals its report's sha256 and §19.2's pin, each
  report is byte-identical to its copy in `evidence/task090-t/t/`, and each records the three
  feature reports and the pinned moments. The `last_two_triggered` flags are set for exactly
  G-v2g-90102, G-v1p-90100 and G-v1p-90101 (reported, §3).
- TASK-087's G-87100, G-87101 and G-87102 equal their pins; the feature reports are
  `play-v1` `f5648a35…7eae`, play-v2g `d8afb236…e988`, play-v1p `00625a44…822e`; λ = 2.0552; the
  frozen files equal their pins.

**§8's seed check at this revision** (`seeds-check.json`; `git fetch`, then all 105 remote
branches and this branch, over `src/`, `scripts/`, `docs/`, `.mc/`, `tests/`, `benchmarks/` and
`configs/`): no number in 906000–906063, 907000–907127, 90300–90331 or 90700–90799 appears in any
file other than TASK-091's own (this protocol, its `.mc` task, `docs/DECISIONS.md` (as) and the
Stage 0 code). In the evidence the only tokens in these ranges are the update counts 90300 and
90700 in TASK-077's training curves (12 occurrences in six `report.json` files); decimal fractions
are not counted as tokens. A first run of the check (kept as `seeds-check-hexpattern.json`) also
counted digit runs inside hex hashes (for example `…e0c90716cd…` in a revision); its pattern was
fixed before the recorded run.

**Debug smoke** (debug seeds and the real checkpoints; at `df5941f`, the recorder comparison at
`5620808`, and a second pass at `7c49117`; **debug, not results**):

- **A″ on 8 debug roots** (90700–90707; 10 rejected draws; all eleven models). Every A2-seed plan's
  first command equalled `first_planned`'s bitwise, at R1 and R2 of every root, for all nine G
  models. The redefined measure ran: every G model's plan closed by k ≤ 4 on 8 / 8 roots at R1;
  the first command closed on 0–8 of 8 (G-v2g-90100 0, G-v1p-90100 1, G-v1p-90102 4, the others
  8). The row printed is A″-FAIL, as expected and meaningless on 8 roots: A1's bars are counts of
  64 (61 / 64), so no model can pass on 8 roots.
- **Closed loop on 6 debug resets** (90740–90745; 12 rejected draws): ik 5 / 6, ik-nopress 5 / 6,
  hold 0 / 6, random 0 / 6, M-v2g-90100 0 / 6, M-v1p-90100 1 / 6, M-v1-87100 0 / 6. The BD
  summary (bar from ik's rate) and an MP4 replay (one debug episode; the replay matched the run's
  palm distances) worked.
- **§5.3's measures** were recorded for every world-model episode, and the summary reports their
  distribution per arm (debug, not read).
- **The split equals the unsplit run, per reset, bitwise** (`compare-smoke.json`): M-v2g-90100 run
  as two jobs of three resets each and combined gives the same executed commands, success,
  distances, rise, contact, plan logs and deferral measures as the unsplit run on all six resets.
  The graphics opt-in test does the same for the random and ik arms on three debug resets.
- **§5.3's third measure leaves the executed commands unchanged** (`compare-smoke.json`):
  M-v2g-90100 with the recorder against TASK-090's `gr.controller` without it, on the first three
  resets: executed commands, success, distances, rise, contact and costs equal bitwise.
- **The second pass** (`compare-smoke2.json`; after moving the runner helpers into the module for
  F16): A″ on the same 8 debug roots (re-drawn, arrays equal) gives equal A1 wins, plans, A2
  measures, ceiling and probes, and ik and M-v2g-90100 on the first two debug resets give equal
  episodes.

**Timing and the job split** (`smoke/summary-timing-debug-grasp.json`). At 6 processes a
world-model run's wall time per command per process was 1.075–1.094 s (planning 0.868–0.892 s of
it); at 3 processes 0.544 s. With every episode at the full 400 commands (an upper bound: successes
and velocity stops end episodes early), a 64-reset job at 6 processes projects to at most 4 814 s,
below the 5 400 s trigger and the 7 200 s cap. TASK-089's measured 64-reset runs took
3 623–4 321 s.

| Stage | Jobs | Projection |
| --- | --- | --- |
| A″ | roots, then one measurement job | about 0.25 GPU h (8 debug roots: 28 s and 75 s) |
| BD | goals, 4 baselines, 3 world-model runs of 32 resets (one job each, at most about 2 600 s) | at most about 2.3 GPU h |
| BS | goals, 4 baselines, 7 world-model runs as 14 jobs of 64 resets (each at most about 4 800 s), the determinism re-run | about 14–17 GPU h at TASK-089's rate, at most about 19 |

**Compute.** Stage 0 used about 0.52 GPU hours in 22 jobs (no lock wait) and a few CPU minutes
(the two `verify` runs, the seed checks and the tests).

**Incidents (disclosed; debug only).** The smoke script passed `--suffix -norecord`, which
argparse read as a flag; the recorder comparison then hit a runner bug (the overwrite check used
the name before the suffix). Both were fixed (`5620808`) and that comparison re-run; nothing else
was re-run. The full test suite then caught the runner loading TASK-090's runner (F16), so its
helpers were moved into `grasp_corpus_wm_planned_close_run.py` (`7c49117`) and the second smoke
pass above checked the result against the first.

- **R27.55 — Stage 0 is recorded.** The code, the hash checks and the seed check are as above;
  every check passes.
- **R27.56 — Stage 0 changes none of §12's three items.** The per-run process counts stay (8 for
  baselines, 6 for world-model runs, 4 for the determinism re-run). The BS split stays at two
  64-reset jobs per world-model run (907000–907063, 907064–907127): the worst projected job is
  4 814 s at the full budget, below 5 400 s. §5.3's third measure is kept: reading the plan
  leaves every executed command bitwise unchanged.
- **R27.57 — readings of points the DRAFT left open** (each the reading closest to TASK-090's
  code):
  1. Debug seeds: roots 90700–90739, resets 90740–90799 (inside §8's debug range). Debug runs use
     the real checkpoints of §3 (there are no debug models).
  2. A″ reads A1's bars from A-K0's frozen `A1_BARS`; it needs no ceiling file. It keeps both
     A2-seed plans (R1 and R2); lift@R2 reads the R2 plan's first command. It stops at the first
     plan whose first command differs from `first_planned` (A″-VOID).
  3. §5.3 (1): "before first contact" counts the commands issued while no contact had been
     observed, up to and including the command after which contact is first seen; "executed" is
     the applied command (after the embodiment's projection); the requested count is reported
     beside it. (2): the grasp phase is the commands whose controller call planned for the grasp
     subgoal (index 2), and the index is 1-based from its first command. (3): a plan that never
     closes counts as k > 1 in the longest run, and the median k is over plans that close.
  4. The cap: each job stops itself at 7 200 s and writes its report marked `completed: false`,
     `cap_hit: true` (exit 4); `timeout(1)` 5 minutes later is a backstop. A summary that finds an
     incomplete report, a part without its combined run or a combined run without its parts is
     VOID. "Run twice" is enforced by a ledger (`outputs/task091-ledger/`) that a non-debug
     cohort's roots, goals or run claim once.
  5. BOARD.md: while the Requests section has a `- v2d:` line not marked `[handled]`, the stage
     script waits up to 15 minutes before taking the lock with `gpu_run.sh --wait`.
- **R27.58 — the debug smoke is not evidence.** It changes nothing in §2–§11.
- **R27.59 — frozen at this PR's merge.** Stage A″, BD and BS run at the merged revision from a
  clean tree with `scripts/run_task091_stage.sh` (`a`, then `bd`; `bs` only after an independent
  GO on BD's record), evidence in `~/develop/emai/evidence/task091-*`. The V2D session is told
  before BD and before BS.
- **R27.60 — R7 does not change.**

## 15. Stage A″ record (R27.61–R27.64; decided by Claude under owner delegation)

**Run conditions.** Stage A″ ran once on 2026-10-13 (10:39–10:50 UTC) with
`scripts/run_task091_stage.sh a outputs/task091-a`, at the frozen revision `dbab1a5` (the merge
of #210), from a clean tracked tree in a fresh worktree (`worktrees/task091-run`, its `data/`,
`checkpoints/` and `outputs/` linked to the data disk as in the main checkout). Before any job
claimed the cohort ledger, the CPU checks ran at the same revision and passed:
`run_task091.py verify` (`all_pass: true`: TASK-090's §12 check, the eleven checkpoints, the
three feature reports, the pinned moments, the implementation hash, λ and the frozen files) and
`run_task091.py seeds` (106 remote branches; no number in 906000–906063, 907000–907127,
90300–90331 or 90700–90799 outside TASK-091's own files; in the evidence the only tokens are the
same 12 TASK-077 update counts as at Stage 0), and the 19 Stage 0 tests (18 passed, 1 graphics
opt-in skip). This order answers the review's note (c) on #210 in practice: a guard failure could
not burn the A″ cohort, because the guards had passed before the ledger was claimed. BOARD.md's
Requests had no `- v2d:` line before either job. Each job took the shared lock with
`gpu_run.sh --wait` (about 1.5 minutes' wait behind a V2D job). Evidence:
`~/develop/emai/evidence/task091-a/` (`SHA256SUMS`): the roots, the A″ report, the summary, the
checks, the stage log, the two ledger entries and the `gpu_run` log lines.

**Jobs and GPU time.** The roots job (64 roots, 906000–906063, 83 rejected draws; targets: cube
29, apple 18, can 16, banana 1) took 68 s; the A″ job took 566 s. Neither reached the 7 200 s
cap, both reports are `completed: true`, and no plan's first command differed from
`first_planned`'s (`first_command_mismatches: []`). In total about 0.18 GPU hours (634 s).

**The row, as `summarize_task091.py --stage a` printed it:**
`{'row': 'A″-PASS', 'void_reason': '', 'fails': {'v2g': 0, 'v1p': 1}, 'failing_arms': []}`.
The summary lists no problems.

**Per model** (64 roots; A1 at h = 8 with G's cost, bar 61 / 64 on each of the three
comparisons; close≤4@R1 and lift@R2, bar 48 / 64; close@R1 reported only):

| Model | A1 h = 8 (close-vs-open / lift-vs-release / lift-vs-stay) | close≤4@R1 | lift@R2 | close@R1 (reported) | A″ |
| --- | --- | --- | --- | --- | --- |
| G-v2g-90100 | 64 / 64 / 64 | 64 | 56 | 0 | pass |
| G-v2g-90101 | 64 / 64 / 64 | 64 | 52 | 53 | pass |
| G-v2g-90102 ¹ | 64 / 64 / 64 | 64 | 61 | 55 | pass |
| G-v1p-90100 ¹ | 64 / 64 / 64 | 64 | 61 | 3 | pass |
| G-v1p-90101 ¹ | 64 / 64 / 64 | 64 | 63 | 60 | pass |
| G-v1p-90102 | 64 / 64 / **59** | 64 | 59 | 11 | **fail (A1)** |
| G-v1-87100 (reported) | 64 / 64 / 64 | 64 | 61 | 63 | pass |
| G-v1-87101 (reported) | 64 / 64 / 64 | 64 | **34** | 64 | fail (lift@R2) |
| G-v1-87102 (reported) ¹ | 64 / 64 / 64 | 64 | 57 | 63 | pass |

¹ `last_two_triggered` set (reported, §3).

- **Why this is the row.** M-v2g fails A″ on none of its three seeds and M-v1p on one
  (G-v1p-90102), so neither arm fails on two or more seeds, which is §4's A″-FAIL rule. The checks
  of §4 all hold, so the row is not A″-VOID. A″ passes.
- **G-v1p-90102's failure is on A1, not on the redefined bar.** Its lift-vs-stay comparison at
  h = 8 won on 59 of 64 roots, below A-K0's bar of 61 (at A′, on other roots, it won 63). It met
  close≤4@R1 (64) and lift@R2 (59). By §4, that seed stays in M-v1p's pooled count in the closed
  loop, and its A″ result is reported beside its closed-loop count. M-v1's results are reported
  only; G-v1-87101 fails lift@R2 again (34; 37 at A′, 39 at D-A2).
- **Reported, the plan at R1.** Every G model's A2-seed plan closed by plan step k ≤ 4 on 64 / 64
  roots, and none never closed. The first closing step: G-v2g-90100 at k = 3 on 40 roots and
  k = 4 on 24; G-v1p-90100 at k = 1, 2, 3, 4 on 3, 3, 45, 13; G-v1p-90102 at k = 1 on 11 and
  k = 2 on 53; the other six models at k = 1 on 53–64. Share closing by k ≤ 2: 0 / 64
  (G-v2g-90100), 6 / 64 (G-v1p-90100), 61–64 / 64 for the rest. The mean right-grasp channel per
  plan step is in the summary (`mean_grasp_per_step_R1`).
- **Reported, the controls and probes.** The action-shuffled control at h = 8 won 26–37 of 64 on
  each comparison for every G model. The N twins' latent-only wins at h = 8 are 0 on all three
  comparisons. The ceiling on A″'s roots (latent+pose, h = 8): 64 / 63 / 64. The readability
  probe's proprio floor (AUROC 0.98 at h = 8 and 16) sits above its frame ceiling (0.84 at h = 8,
  0.76 at h = 16), as at A′. The scripted lift branch's maximum rise at R2 was at least 5 cm on
  18 of 64 roots.

**What this is and is not.** One run, offline, simulation only, on 64 fresh roots. A″ is a
first-command and plan check, a **no-regression gate only** (§4): its close bar's threshold
k ≤ 4 was chosen after D-A2's development numbers, and a pass was expected from them. It is not a
closed-loop result. It does not show that any model closes, grasps or lifts in closed loop, and
no claim of this task rests on it. Whether a receding-horizon controller that plans the close at
step 3–4 on each re-plan actually closes (the deferral risk of R27.50 and §9, which falls on
G-v2g-90100 and G-v1p-90100 here) is what Stage BD and Stage BS measure. R7 does not change.

**Known notes from the review of #210, unchanged at the freeze** (disclosed here; none was acted
on, because the code is frozen):

- (a) A first-command mismatch stops A″ after the model with the mismatch, not at the first plan;
  the row (A″-VOID) is the same. No mismatch occurred.
- (b) A missing report makes the summarizer crash instead of writing a VOID summary; a
  hand-written VOID record under a ruling would cover it. Both reports were present.
- (c) The A″ job claims the ledger before its cheap guards (`load_roots`, the feature pins), so a
  guard failure would burn the cohort. Mitigated in this run by passing `verify` and the seed
  check first; no guard failed.
- (d) The cap's self-stop timer starts at job entry, not process start (a gap of seconds, bounded
  by the `timeout` backstop). Neither job came near the cap.
- (e) A BD re-run after a cap hit or a NO-GO needs a new cohort entry, so it is necessarily a
  post-freeze code change under its own ruling.
- (f) Stage 0's GPU log has a 23rd job (the render tests at `d61115b`, 39 s) outside §14's
  window; "22 jobs" holds for that window.

**Next.** By R27.59, Stage BD runs next (`scripts/run_task091_stage.sh bd`, development resets
90300–90331), after the V2D session is told; Stage BS runs only after an independent GO on BD's
record. Neither has run.

- **R27.61 — Stage A″ is recorded.** The checks passed before the cohort was claimed; the stage
  ran once at `dbab1a5` from a clean tree, in two jobs (68 s and 566 s, about 0.18 GPU h), with no
  first-command mismatch and no cap hit.
- **R27.62 — A″-PASS.** The summarizer's row is `A″-PASS` (`fails: {'v2g': 0, 'v1p': 1}`).
  Every G model met close≤4@R1 (64 / 64), and every gated G model met lift@R2 (52–63; the
  reported-only G-v1-87101 scored 34);
  G-v1p-90102 failed A1 at h = 8 (lift-vs-stay 59 < 61) and stays in M-v1p's pooled count, with
  its A″ result reported beside its closed-loop count. A no-regression gate only: one run,
  offline, simulation only; not a closed-loop result.
- **R27.63 — Stage BD is next** (R27.59), after the V2D session is told; BS only after an
  independent GO on BD's record.
- **R27.64 — R7 does not change.**

## 16. Stage BD record (development, not gated; R27.65–R27.68; decided by Claude under owner delegation)

**Run conditions.** Stage BD ran once on 2026-10-13 (11:35–13:14 UTC) with
`scripts/run_task091_stage.sh bd outputs/task091-bd`, at the frozen revision `dbab1a5` (#210's
merge, the revision A″ ran at), from a clean tracked tree in `worktrees/task091-run` (detached;
`main` at `cff67c5` differs from `dbab1a5` only in `docs/` and `.mc/`, checked with
`git diff --stat`). It ran at `dbab1a5` rather than at `cff67c5` because Stage BS's summarizer
requires BD and BS to share one revision (`summarize_task091.py`, "Stage BD ran at another
revision"), and A″ ran there too. Before any job claimed the cohort ledger, the CPU checks ran at
the same revision and passed: `run_task091.py verify` (`all_pass: true` in both
`verify-task091.json` and `verify-task090-checks.json`: TASK-090's §12 check, the eleven
checkpoints, the `last_two_triggered` flags, the three feature reports, the pinned moments, the
implementation hash, λ and the frozen files), `run_task091.py seeds` (107 remote branches; no
number in TASK-091's ranges outside TASK-091's own files; in the evidence the same 12 TASK-077
update counts as before), and the 19 Stage 0 tests (18 passed, 1 graphics opt-in skip; that test log is not archived in the
evidence). The V2D
session was told before the run. BOARD.md's Requests had no `- v2d:` line before any of the eight
jobs, and no job waited for the lock. Evidence: `~/develop/emai/evidence/task091-bd/`
(`SHA256SUMS`): the goals, the seven run reports, the summary, the checks, the stage log, the
eight ledger entries and the `gpu_run` log lines.

**Jobs and GPU time.** Goals (32 resets, 90300–90331, 31 rejected draws; targets can 15, cube 11,
apple 6) 13 s; ik, ik-nopress, hold and random 10–13 s each; M-v2g-90100 2 061 s, M-v1p-90100
1 674 s, M-v1-87100 2 079 s (wall, 6 processes each). No job reached the 7 200 s cap; every report
is `completed: true`, `cap_hit: false`. In total 8 jobs, 5 873 s, about **1.63 GPU hours**.

**The summarizer's output, as `summarize_task091.py --stage bd` printed it:**

```
{
 "stage": "bd",
 "row": null,
 "bar": 0.4,
 "problems": []
}
```

BD has no row (it is not gated). **The bar is set:** ik reached 26 / 32 (r = 0.8125, at least
0.30), so bar = min(0.40, 0.9 × 0.8125) = **0.40**; at BS's n = 128 the pooled bar is
**156 / 384** and the per-seed floor **26 / 128** (`summary-bd.json`: `pooled_bar_at_bs` 156,
`seed_floor_at_bs` 26, `escalate: false`). The summary lists no problems.

**Per arm** (32 resets; exact 95 % intervals; contact is thumb plus index or middle on the target;
rise is the maximum target rise; "stopped" is the embodiment's measured-joint-velocity stop):

| Arm | Successes | 95 % interval | Grasp contact | Median max rise | Stopped | Plan s / command |
| --- | --- | --- | --- | --- | --- | --- |
| ik | **26 / 32** | 0.636–0.928 | 32 | 6.6 cm | 1 | — |
| ik-nopress | 24 / 32 | 0.566–0.885 | 32 | 6.6 cm | 3 | — |
| hold | 0 / 32 | 0.000–0.109 | 0 | 0.0 cm | 0 | — |
| random | 0 / 32 | 0.000–0.109 | 0 | 0.0 cm | 0 | — |
| M-v2g-90100 | **5 / 32** | 0.053–0.328 | 23 | 0.70 cm | 1 | 0.97 |
| M-v1p-90100 | **3 / 32** | 0.020–0.250 | 18 | 0.32 cm | 11 | 0.96 |
| M-v1-87100 (reported) | 2 / 32 | 0.008–0.208 | 21 | 0.45 cm | 3 | 0.95 |

- **Successful resets.** M-v2g-90100: 90302, 90318, 90320, 90321, 90326 (can 3, cube 1, apple 1).
  M-v1p-90100: 90301, 90312, 90321 (one of each). M-v1-87100: 90302, 90326 (cube 1, can 1). ik
  succeeded on can 15 / 15, cube 10 / 11 and apple 1 / 6.
- **Executed close per phase** (share of executed commands with the right-grasp channel > 0.5):
  grasp phase 0.47 (M-v2g-90100), 0.56 (M-v1p-90100), 0.73 (M-v1-87100), 0.86 (ik); lift phase
  0.79–0.81 for the world-model arms, 1.0 for ik. Final subgoal reached: the lift subgoal (3) on
  31, 21 and 29 resets; the rest ended in the grasp subgoal (2).
- **Velocity stops.** All 11 of M-v1p-90100's stops came in the grasp subgoal, 9 before any
  contact; the run reports do not log which joint tripped. The stop is the embodiment's frozen
  5 rad/s rule (§2), a declared failure, not a harness defect.

**§5.3's deferral measures** (reported only; no row reads them; per reset over the 32 episodes;
medians with the 10th–90th percentiles):

| Measure | M-v2g-90100 | M-v1p-90100 | M-v1-87100 |
| --- | --- | --- | --- |
| (1) closing commands before first contact, episodes with contact | median 1 (p90 4.6, max 43); n = 23 | median 1 (p90 5.3, max 72); n = 18 | median 2 (p90 15, max 90); n = 21 |
| (1) episodes with no contact (counted to the end, flagged) | 9 | 14 | 11 |
| (2) first executed close in the grasp phase, command index | median 10 (p10 8, p90 17.9, max 22) | median 8 (p10 8, p90 13, max 26) | median 9 (p10 8, p90 11, max 13) |
| (2) never closed in the grasp phase / grasp phase not reached | 0 / 0 | 1 / 0 | 3 / 0 |
| (3) share of grasp-phase re-plans with k = 1, per reset | median 0.50 (p10 0.43, p90 0.85); pooled 0.545 | median 0.53 (p10 0.49, p90 0.70); pooled 0.546 | median 0.59 (p10 0.54, p90 0.88); pooled 0.619 |
| (3) median planned close step k, per reset | median 1.75 (max 3) | median 1 (max 4) | median 1 (max 2) |
| (3) longest run of re-plans with k > 1, per reset | median 5 (p90 6, max 58) | median 4.5 (p90 6, max 100) | median 3 (p90 4, max 5) |

- **Reading, reported only.** M-v2g-90100 did not put the close off indefinitely: every one of its
  episodes closed in the grasp phase (median at its 10th command, at most its 22nd); M-v1p-90100
  never closed in the grasp phase on 1 reset and M-v1-87100 on 3. The longest runs of re-plans
  planning the close later than step 1 are about 4–6 re-plans long on most resets for M-v2g-90100
  and M-v1p-90100, and 3–4 (maximum 5) for M-v1-87100. One
  reset per new arm had a long run (58 for M-v2g-90100, 100 for M-v1p-90100, on reset 90319 for
  both). On about half of the grasp-phase re-plans of M-v2g-90100 and M-v1p-90100 the plan closes
  at step 1 (A″'s R1 plan at the A2 seed closed at step 1 on 0 / 64 and 3 / 64 roots for these two
  models; that is a different state and a different measure). M-v2g-90100's median first executed
  close came at command 10, against 9 for M-v1-87100 and 8 for M-v1p-90100. The projection's
  joint-rate limit moves the executed grasp channel by about 0.18 per command, so from an open hand
  a close cannot register before about grasp-phase command 8–9 (observed minimum 7; p10 8 in every
  arm): these medians are about 0–2 commands after the earliest possible, not 8–10 commands of
  deferral, and that deferral caused any gap is not shown. A step-1 planned close is not an
  executed close: on reset 90319 M-v1-87100 planned the close at step 1 on 46 % of re-plans but
  never executed one (its requests alternated close and open, and the rate limit never let the
  channel build up); on
  reset 90319, M-v1p-90100 never closed in the grasp phase, and M-v2g-90100 first closed at its 9th
  grasp-phase command but planned the close later than step 1 on all but 3 % of its grasp-phase
  re-plans there (both failed). Whether that lowers success is not shown by 32 resets. On the five
  M-v2g-90100 successes the per-reset share at k = 1 was 0.50–0.95 and the first close came at
  command 8–18. Over all 32 resets the visible gap to ik is the rise: M-v2g-90100 reached grasp
  contact on 23 / 32 resets (ik 32), but its median maximum rise was 0.70 cm against ik's 6.6 cm.
- **The named confound (§9).** Both world-model seeds in BD are late planners at A′/D-A2 and A″:
  G-v2g-90100 (M-v2g's one late seed of three) and G-v1p-90100 (one of M-v1p's two late seeds of
  three). So BD's M-v2g-against-M-v1p comparison is between two late-planning seeds and cannot show
  how the 1-of-3 against 2-of-3 imbalance acts at BS. At BS, a B-PASS or B-COMP is restated with
  §9's caveat, and a B-FAIL or B-UNDER with §5.3's per-seed deferral measures beside it.

**What this is and is not.** A development cohort of **32 resets, one run, simulation only, not
gated, not Stage BS**. It sets the bar (0.40) and shows the plumbing and the arms ran under the
rules. It carries no row and no claim, and its counts are not a prediction of BS: M-v2g-90100's
5 / 32 (0.156; interval 0.053–0.328) is below the bar of 0.40 and below the BS per-seed floor's
rate (26 / 128 = 0.203), and M-v2g-90100 is the one late-planning seed of M-v2g; the other two M-v2g
seeds did not run in BD. M-v1-87100's 2 / 32 is reported only (TASK-089's gated G-87100 lifted
0 / 64). Every restatement quotes TASK-090 Stage C's C1 2.38 and C2 1.69 (§9). No MP4 replay was
made in BD (the stage script makes none; TASK-090 §7.2's two MP4s are reported with a BS row).
No knob changes (§2, §9), and R7 does not change.

**What the GO review decides** (§5.2, TASK-090 §7.2, TASK-089 §7.2's reading): an independent
reviewer reads this record and posts GO or NO-GO before Stage BS. BD can show a defect, whose fix
needs its own ruling, but it changes no knob of the protocol by itself, and §9 forbids a planner
change to counter deferral even if BD shows it. The bar is set (r ≥ 0.30), so the bar rule does
not escalate. After a NO-GO, BD is recorded again on fresh development seeds named by a ruling
(a post-freeze cohort entry, §15 note (e)). This record does not decide GO.

**Next.** Stage BS (`scripts/run_task091_stage.sh bs <out> outputs/task091-bd/summary-bd.json
outputs/task091-a/summary-a.json`, 907000–907127) runs only after an independent GO on this
record, at `dbab1a5` from a clean tree, after the V2D session is told. It has not run.

- **R27.65 — Stage BD is recorded.** The checks passed before the cohort was claimed; the stage
  ran once at `dbab1a5` from a clean tree, in eight jobs (5 873 s, about 1.63 GPU h), with no cap
  hit, every report complete and no problem in the summary.
- **R27.66 — The bar is 0.40.** ik 26 / 32 (r = 0.8125 ≥ 0.30); pooled bar at BS 156 / 384,
  per-seed floor 26 / 128. Counts, development only: M-v2g-90100 5 / 32, M-v1p-90100 3 / 32,
  M-v1-87100 2 / 32 (reported), ik-nopress 24 / 32, hold 0 / 32, random 0 / 32.
- **R27.67 — Stage BS waits for an independent GO** on this record (R27.59); nothing of §2–§11
  changes.
- **R27.68 — R7 does not change.**

## 17. Stage BS record (gated; R27.69–R27.73; decided by Claude under owner delegation)

**Run conditions.** Stage BS ran once, from 2026-10-13 13:46 UTC to 2026-10-14 04:41 UTC, with
`scripts/run_task091_stage.sh bs outputs/task091-bs outputs/task091-bd/summary-bd.json
outputs/task091-a/summary-a.json`. It ran at the frozen revision `dbab1a5`, the revision of A″
and BD, from a clean tracked tree in `worktrees/task091-run` (detached). That was after an
independent reviewer posted **GO** on BD's record (#212, review of head `d679a01`; #212 has since
merged as `ac77c7d`, which differs from `dbab1a5` only in `docs/` and `.mc/`). Before any job
claimed the cohort ledger, the CPU checks ran at the same revision and passed:
`run_task091.py verify` (every key true: TASK-090's §12 check, the eleven checkpoints, the
`last_two_triggered` flags, the three feature reports, the pinned moments, the implementation
hash, λ and the frozen files), and `run_task091.py seeds` (108 remote branches; no number in
TASK-091's ranges outside TASK-091's own files; in the evidence, the same 12 TASK-077 update
counts as before). The Stage 0 tests were not re-run before BS; they ran before BD at the same
revision. The V2D session was told before the run. BOARD.md's Requests had a `- v2d:` line twice,
before the second job of M-v2g-90100 and before the second job of M-v1p-90102. The stage script
waited both times (5 minutes and 1 minute) until the line was gone and then took the lock. No other
job waited. Evidence: `~/develop/emai/evidence/task091-bs/` (`SHA256SUMS`, 60 files): the goals,
the 11 combined run reports, the 14 part reports, the determinism re-run, the summary, the checks,
the stage log, the 20 ledger entries, the `gpu_run` log lines and the two MP4 replays.

**Jobs and GPU time.** Goals (128 resets, 907000–907127, 151 rejected draws; targets apple 47,
can 41, cube 37, banana 3) 52 s. ik, ik-nopress, hold and random took 31–42 s each. The 14
world-model jobs (two 64-reset parts per run, 6 processes each) took 3 184–4 218 s each, and each
run's two parts were combined into one run before the summary. The determinism re-run took 287 s.
**No job reached the 7 200 s cap** (the longest was M-v1-87100's first part, 4 218 s). Every run
and part report is `completed: true`, `cap_hit: false`. In total 20 jobs, 53 260 s, about
**14.8 GPU hours**. After the stage, the two MP4 replays took 2 jobs of 5 s each. They replay
recorded commands, claim no ledger entry and are not counted.

**The summarizer's output, as `summarize_task091.py --stage bs` printed it:**

```
{
 "stage": "bs",
 "row": "B-FAIL",
 "bar": 0.4,
 "problems": []
}
```

`summary-bs.json` gives `clause_fires: true`, `lewm_needed_claimed: false`,
`determinism_matches: true` and `problems: []`.

**Per arm** (128 resets; exact 95 % intervals; contact means thumb plus index or middle on the
target; rise is the maximum target rise; "stopped" is the embodiment's measured-joint-velocity
stop; the executed-close share is the share of executed commands in that phase with right grasp
> 0.5):

| Arm | Successes | 95 % interval | Grasp contact | Median max rise | Stopped | Final subgoal = 3 | Executed close, grasp / lift phase | Plan s / command | A″ (§15) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ik | **82 / 128** | 0.551–0.723 | 122 | 6.2 cm | 12 | 120 | 0.86 / 1.00 | — | — |
| ik-nopress | 78 / 128 | 0.519–0.694 | 121 | 6.0 cm | 13 | 120 | 0.86 / 1.00 | — | — |
| hold | 0 / 128 | 0.000–0.028 | 0 | 0.0 cm | 0 | 128 | 0 / 0 | — | — |
| random | 0 / 128 | 0.000–0.028 | 0 | 0.0 cm | 1 | 128 | 0.04 / 0.04 | — | — |
| M-v2g-90100 ¹ | **8 / 128** | 0.027–0.119 | 58 | 0.32 cm | 13 | 115 | 0.46 / 0.87 | 1.01 | pass |
| M-v2g-90101 | **8 / 128** | 0.027–0.119 | 39 | 0.18 cm | 43 | 88 | 0.29 / 0.66 | 0.96 | pass |
| M-v2g-90102 ² | **6 / 128** | 0.017–0.099 | 60 | 0.29 cm | 31 | 97 | 0.41 / 0.87 | 0.99 | pass |
| M-v1p-90100 ¹ ² | **11 / 128** | 0.044–0.149 | 87 | 0.32 cm | 34 | 94 | 0.58 / 0.89 | 0.98 | pass |
| M-v1p-90101 ² | **7 / 128** | 0.022–0.109 | 53 | 0.25 cm | 45 | 83 | 0.54 / 0.74 | 0.97 | pass |
| M-v1p-90102 ¹ | **8 / 128** | 0.027–0.119 | 71 | 0.34 cm | 13 | 115 | 0.46 / 0.72 | 1.00 | fail (A1) |
| M-v1-87100 (reported) | 1 / 128 | 0.000–0.043 | 51 | 0.23 cm | 19 | 110 | 0.72 / 0.85 | 1.00 | pass |

¹ A late planner at A′/D-A2 (§9's confound): G-v2g-90100, G-v1p-90100 and G-v1p-90102.
² `last_two_triggered` set (reported, §3).

- **Successful resets.** M-v2g-90100: 907025, 907040, 907064, 907075, 907099, 907103, 907107,
  907113. M-v2g-90101: 907005, 907067, 907077, 907078, 907083, 907090, 907099, 907101.
  M-v2g-90102: 907015, 907018, 907049, 907061, 907078, 907107. M-v1p-90100: 907000, 907005,
  907017, 907061, 907073, 907077, 907087, 907102, 907105, 907115, 907118. M-v1p-90101: 907038,
  907053, 907091, 907093, 907099, 907101, 907118. M-v1p-90102: 907014, 907033, 907052, 907077,
  907083, 907097, 907118, 907121. M-v1-87100: 907017.
- **Per object** (successes / episodes, pooled over seeds): M-v2g apple 8 / 141, can 10 / 123,
  cube 4 / 111, banana 0 / 9. M-v1p apple 12 / 141, can 9 / 123, cube 5 / 111, banana 0 / 9.
  M-v1-87100 apple 1 / 47, others 0. ik apple 12 / 47, can 38 / 41, cube 32 / 37, banana 0 / 3.

**The row's inputs** (§5.1, TASK-090 §7.2; bar 0.40 from BD, R27.66):

| Quantity | Value | Bar or test |
| --- | --- | --- |
| k_g, M-v2g pooled | **22 / 384** (0.057; exact 95 % 0.036–0.085) | pooled bar **156 / 384**: not met |
| k_c, M-v1p pooled | 26 / 384 (0.068; 0.045–0.098) | — |
| M-v2g per seed | 8, 8, 6 of 128 | per-seed floor **26 / 128**: no seed meets it |
| "M-v2g beats M-v1p" (reset-level exact one-sided sign-flip over d_r = G_r − C_r) | p = 0.767 | p < 0.01: **not met** |
| "M-v2g beats random" (same, d_r = G_r − 3 R_r) | p = 1.9 × 10⁻⁶ | p < 0.01: met |
| Reset-cluster bootstrap of (k_g − k_c) / 384 (10 000 resamples, salt 9006) | −0.010; 95 % −0.047 to **+0.023** | B-FAIL needs the upper bound below δ = +0.10: met |
| Pooled exact one-sided McNemar (reported) | p = 0.774 | not gating |
| Determinism re-run, M-v2g-90100 on 907000–907003 | matches bitwise (executed commands and success; 0 / 4, as in the run) | a mismatch is B-VOID: matched |

- **Why this is the row.** Every check held and every run completed under the cap, so the row is
  not B-VOID. M-v2g does not beat M-v1p, so it is not B-PASS, B-COMP or B-SIZE. M-v2g does not beat
  M-v1p and k_g is below the pooled bar. The bootstrap's upper bound, +0.023, is below δ = +0.10,
  so the cohort rules out a composition gain of δ at that interval. That is **B-FAIL** (TASK-090
  §7.2 row 5), not B-UNDER.
- **Reported tests** (TASK-090 §7.2): M-v2g against M-v1-87100 (×3), sign-flip p = 7.1 × 10⁻⁴,
  bootstrap +0.049 (95 % +0.021 to +0.081); M-v1p against M-v1-87100 (×3), p = 2.5 × 10⁻⁵,
  +0.060 (+0.031 to +0.091). Both of TASK-090's added play sets did better than `play-v1` alone,
  each against one M-v1 seed. ik against M-v2g (ik ×3 per reset), p = 1.4 × 10⁻²³.

**§5.3's deferral measures, per seed** (reported only; no row reads them; per reset over the 128
episodes; medians with p90 and max). §9 requires them beside any restatement of a B-FAIL.

| Measure | M-v2g-90100 ¹ | M-v2g-90101 | M-v2g-90102 | M-v1p-90100 ¹ | M-v1p-90101 | M-v1p-90102 ¹ | M-v1-87100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Successes | 8 | 8 | 6 | 11 | 7 | 8 | 1 |
| (1) closing commands before first contact, episodes with contact: median (p90, max); n | 1 (6, 44); 58 | 1 (8.6, 31); 39 | 3 (6.2, 65); 60 | 1 (6, 90); 87 | 1 (12.6, 99); 53 | 1 (10, 61); 71 | 2 (13, 97); 51 |
| (1) episodes with no contact | 70 | 89 | 68 | 41 | 75 | 57 | 77 |
| (2) first executed close in the grasp phase, command index: median (p90, max) | 10 (17.1, 66) | 8 (11.7, 28) | 9 (15, 97) | 8 (13, 34) | 9 (11, 27) | 10 (15.8, 66) | 9 (11, 18) |
| (2) never closed in the grasp phase / grasp phase not reached | 7 / 1 | 14 / 0 | 8 / 0 | 10 / 0 | 17 / 0 | 5 / 0 | 6 / 2 |
| (3) share of grasp-phase re-plans with k = 1, per reset: median (p10–p90); pooled | 0.49 (0.40–0.62); 0.516 | 0.54 (0.47–1.00); 0.542 | 0.56 (0.44–0.93); 0.560 | 0.52 (0.43–0.65); 0.531 | 0.64 (0.52–1.00); 0.602 | 0.53 (0.49–0.83); 0.541 | 0.58 (0.53–0.83); 0.582 |
| (3) median planned close step k, per reset: median (max) | 2 (2.5) | 1 (2) | 1 (3) | 1 (3) | 1 (2) | 1 (2) | 1 (2) |
| (3) longest run of re-plans with k > 1: median (p90, max) | 5 (6, 36) | 4 (6, 10) | 3 (9, 88) | 5 (7, 100) | 4 (7.3, 18) | 4 (6, 38) | 3 (5, 7) |

- **Reading, reported only.** As in BD, the projection's joint-rate limit moves the executed grasp
  channel by at most about 0.18 per command. So from an open hand a close cannot register before
  about grasp-phase command 8–9. The per-seed medians of measure (2), 8–10, are therefore about
  0–2 commands after the earliest possible, in every arm, not 8–10 commands of deferral. Of the
  three late planners, G-v2g-90100 has the latest median planned close step (2; median per-reset
  k = 1 share 0.49) and the latest median first executed close (10, tied with G-v1p-90102). It
  scored 8 / 128, the same as G-v2g-90101 and two more than G-v2g-90102, which planned the close at
  step 1 more often. The late M-v1p seeds scored 11 and 8, and its early seed 7. In this one run,
  the per-seed counts do not order by late against early planning. That does not show that
  deferral played no part.
- **The confound in the opposite direction (§9).** §9 names the risk that deferral sinks
  G-v2g-90100 and pushes k_g below the pooled bar, giving B-FAIL for a planner-timing reason rather
  than a data reason. Here k_g is 22 / 384 against a bar of 156. No M-v2g seed reached a third of
  the per-seed floor (best 8 / 128 against 26), including the two seeds that were not late
  planners. M-v2g also did not beat M-v1p, whose pooled count holds two late seeds. The measures
  are given so a reader can judge this. They do not correct for it, and this record does not claim
  that planner timing played no part.
- **Where the world-model arms fall short of ik.** Grasp contact on 39–87 of 128 resets per seed
  (ik 122), and a median maximum rise of 0.18–0.34 cm (ik 6.2 cm). Between 41 and 89 episodes per
  seed never touched the target with thumb plus index or middle. Velocity stops number 13–45 per
  seed (ik 12). This describes what happened. It is not a diagnosis of the cause.
- **MP4 replays** (TASK-090 §7.2): `videos/M-v2g-90100-907025.mp4` (a success) and
  `videos/M-v2g-90100-907000.mp4` (a failure). Each replays the run's recorded executed commands
  from its reset, and each replay matched the run's palm distances (`replay_matches: true`).

**What this is and is not.** One run, **simulation only**, on 128 fresh gated resets, with one
scene, arm, hand and camera at 112 px, one encoder, latent, recipe, budget and planner, and the
same privileged scripted pick recipe in both corpora (§9, TASK-090 §11). The data gate behind the
corpora was weaker than designed: TASK-090 Stage C's **C1 2.38 and C2 1.69** are quoted with every
restatement of this row. Seeds are pooled, and the bar comes from ik's development rate. "LeWM
needed" is not claimed by any row (the IK follower uses the same goal information and no world
model), and Phase 4 is not admitted. M-v1-87100's 1 / 128 is reported only. The determinism
re-run's four resets were all failures in both runs.

**The abandonment clause fires (§7, TASK-090 §9; B-FAIL only), with its scope unchanged.** It
closes, in §7's words:

> adding more scripted grasp-and-lift play to `play-v1` (e9's pick recipe run more often, held
> longer or retried, as `play_policy_grasp_v0` does) as the grasp fix, under TASK-087's recipe and
> latent and TASK-089's step-averaged planner. After it fires, no further variant of this kind is
> preregistered for grasping without new evidence of a different kind. That covers the skill mix,
> mode mix, hold length, retries, target weighting or added amount of this collector.

> **The next grasp step is then a model change.** For example, a model trained to predict the
> grasp's outcome (contact and object rise) over the planning horizon, the model-first design of
> TASK-090 §15. It needs its own preregistration.

**Not closed** (§7): LeWM, DINOv2 and CEM; `play-v1` and play-v2g as bases; other kinds of data
(real G1 data, a different demonstrator or pick recipe, other sensors); model changes; reach
planning (TASK-088's Z3-REACH stands); Phase 4 as a goal, and the product goal. M-v1's counts are
recorded with the clause, not as a way around R26.33.

- **R27.69 — Stage BS is recorded.** It ran once, after the GO, at `dbab1a5` from a clean tree,
  after `verify` and the seed check passed. It ran as 20 jobs (53 260 s, about 14.8 GPU h) with no
  cap hit. Every run and part report is complete, the summary lists no problems, and the
  determinism re-run matched bitwise.
- **R27.70 — B-FAIL.** The summarizer's row is `B-FAIL`. M-v2g 22 / 384 (per seed 8, 8 and 6 of
  128) against M-v1p 26 / 384 (11, 7 and 8). The pooled bar of 156 / 384 is not met, and no seed
  meets the per-seed floor of 26 / 128. "M-v2g beats M-v1p": p = 0.767, not met. "M-v2g beats
  random": p = 1.9 × 10⁻⁶. The bootstrap's 95 % upper bound of (k_g − k_c) / 384 is +0.023, below
  δ = +0.10. ik scored 82 / 128. The per-seed deferral measures are above. One run, simulation
  only. TASK-090 Stage C's C1 2.38 and C2 1.69 are quoted with every restatement.
- **R27.71 — The §7 clause fires**, with the scope quoted above and nothing wider. More scripted
  grasp-and-lift play of this collector's kind is closed as the grasp fix under TASK-087's recipe
  and latent and TASK-089's step-averaged planner. The next grasp step is a model change, which
  needs its own preregistration.
- **R27.72 — TASK-091 closes at B-FAIL** with this record's merge (§6: B-FAIL's next step is the
  clause). This record chooses no next task.
- **R27.73 — R7 does not change** (R27.54: whatever row this task ends in).
