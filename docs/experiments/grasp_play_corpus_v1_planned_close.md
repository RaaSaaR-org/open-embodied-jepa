# TASK-091 — the planned close within the plan: TASK-090's A′ on fresh roots with A2 redefined, then TASK-090's closed-loop grasp-and-lift comparison on fresh resets

**Status: DRAFT** (R27.45–R27.54, decided by Claude under owner delegation, 2026-10-13; DECISIONS
(as)). A Stage 0 PR follows: it adds the code for the redefined A2 measure, checks the seeds and
runs a debug smoke. The protocol is frozen at that PR's merge, as TASK-090's was (§12). No gated
stage runs before that freeze.

This is the next task that R27.43 recommends. TASK-090 ([grasp_play_corpus_v1.md](grasp_play_corpus_v1.md))
closed at **A′-FAIL** (R27.34, R27.45): escalate, no clause. Its Stage BD and Stage BS were never
run, so no closed loop ran and nothing compared the two corpora on grasping. The development
diagnosis D-A2 (§20 there, R27.42) then read **D-TIMING**. That reading is development only, not
gated: one run, offline, simulation only, on 64 roots 905000–905063. In it, the three A2-failing
models (G-v2g-90100, G-v1p-90100, G-v1p-90102) did not close on the first planned command, but
planned the close at step 2–4 of the 8-step plan, stably across five CEM seeds (largest spread
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
must equal `first_planned`'s bitwise on every root. If it does not, A″ stops and is reported.
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

Every GPU job is limited to 7 200 s (R27.24's per-job cap, applied here to every job). One BS
world-model run over 128 resets projects to about 2.2 GPU hours (R27.25), above the cap. So each
BS world-model run is split into two jobs of 64 resets, 907000–907063 and 907064–907127. They use
the same arm, model and frozen code, and their outputs are combined into one run before any
summary. TASK-089's 64-reset world-model runs took 3 623–4 321 s. A job that reaches the cap stops,
and its run did not complete under the rules, which is B-VOID (TASK-090 §7.2, row 1). Stage 0's
smoke times per-command planning and may split the jobs further (§12).

## 6. Rows of the task, in order

| Row | When | Next step |
| --- | --- | --- |
| A″-VOID | a check of §4 fails (checkpoint, implementation hash, pinned moments, feature report, λ, seed, or the first command against `first_planned`) | fix under its own ruling; re-run on fresh roots named by that ruling |
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
fired nothing (R27.45), and this task's own A″ result fires nothing either.

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
  (G-v1p-90100, G-v1p-90102) and one of M-v2g's (G-v2g-90100) planned the close late at A′ and in
  D-A2 (development). If late planned closes lower closed-loop success, M-v1p's pooled count is hit
  on more seeds, which favours "M-v2g beats M-v1p" for a reason unrelated to the composition. A
  B-PASS or B-COMP would not separate the two causes. The per-seed counts and §5.3's per-seed
  deferral measures are reported so that a reader can see whether the comparison tracks the late
  seeds. They do not correct for it. Any restatement of a B-PASS or B-COMP quotes this caveat.
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
  ranges and their disjointness from TASK-090's, the deferral counters, the job split and recombine,
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
  §2–§7 stay frozen, so the composition comparison M-v2g against M-v1p stays unconfounded.
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
  90200–90231 and 904000–904127. The §9 clause is carried over in its scope, verbatim, and fires
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
  about 15.5 GPU hours (R27.25). The per-job 7 200 s cap (R27.24) applies to every job, so each BS
  world-model run is split into two 64-reset jobs (§5.4). The shared GPU lock applies, one
  `gpu_run.sh --wait` job each, and so does the V2D agreement: BOARD.md's Requests are checked
  before each job. `worktrees/task087-run` stays until TASK-091 closes.
- **R27.53** — **Status.** This protocol is DRAFT in this PR. A Stage 0 PR follows (§12) and freezes
  it at its merge, as TASK-090's did. Gated runs (A″, BD, BS) take place only after that freeze,
  BS only after an independent GO on BD's record.
- **R27.54** — **R7 does not change**, whatever the row.
