# Apple→Plate first learned policy v2: TASK-067's design on apple-to-plate-v2, with the e9 expert and a fresh corpus (TASK-071)

**Status: preregistration.** No corpus episode has been collected, no readout has been fitted, no
policy has been trained, and no gated seed has been simulated. Only the CPU smoke runs of §18 have
run, on the smoke seeds, with noise targets, and nothing in them was read.

**Status lines.**
- **Learned Apple→Plate is still 0 successes.**
- This protocol carries TASK-067's preregistered design
  ([`apple_first_policy_v1.md`](apple_first_policy_v1.md), rulings R1–R7) over to the
  `apple-to-plate-v2` task. **Every change from v1 is listed in §3.** Anything not listed there
  is v1's, unchanged.
- The test split of the new corpus `apple-look-v2` is never decoded. The `apple-look-v1` and
  `apple-wide-v1` corpora are not read at all.
- The development cohort is the fresh D2 (52000–52015). It never gates. Cohort C (45300–45339)
  is not opened by this protocol's gated run; M2 (§12) needs a separate authorization.
- `exemption_spent` (`benchmarks/manifests/apple-policy-diagnostics-v1.json`,
  `precedence_rule_D1_over_G_SUB`) reads `false` at this preregistration, and this protocol does
  not claim the exemption.

> **Disclosed prominently, as v1's ruling R1 requires: the camera and its resolution are
> unchanged.** This task uses the same 112 px `onboard_rgb` camera as TASK-056, TASK-057 and
> TASK-067. What changed since TASK-067 is the **task and the demonstrator**:
> - the task is `apple-to-plate-v2` (v1 plus the apple's contact at condim 6; owner ruling R12);
> - success is `apple_at_rest_v0`, not the latched v1 scorer;
> - the demonstrator and DAgger expert is the frozen e9, which passed TASK-070's at-rest gate;
> - the corpus is new: e9's demonstrations under v2.
>
> The TASK-057 clause (BC on `apple-wide-v1` at 112 px) is not reopened: this protocol reads
> neither that corpus nor `apple-look-v1`. TASK-067's clause did not fire (its row was
> CAL-ESCALATE).

Manifest: `benchmarks/manifests/apple-first-policy-v2.json`. Code that fixes the design:
`src/embodied_jepa/first_policy_v2.py`, which holds the seeds, the corpus plan, e9's schedule,
the arms, labels, calibration rule, gates, rows and caps. A test,
`tests/test_first_policy_v2.py`, pins the manifest's `frozen` block to it. Run-time pieces:
`first_policy_v2_runtime.py` and `first_policy_v2_model.py`. The runner is
`scripts/run_first_policy_v2.py`. v1's modules (`first_policy*.py`, `run_first_policy.py`) are
not modified.

---

## 0. The owner's brief (TASK-071, received 2026-09-28 via the coordinator)

The coordinator's brief, as it bears on the design, restated item by item:

1. **Goal:** train the project's first learned Apple→Plate policy that succeeds.
2. **Carry v1's design over, with only these changes:**
   1. **Task:** apple-to-plate-v2. Success is `apple_at_rest_v0` (the gate). Report the latched
      v1 scorer beside it.
   2. **Demonstrator and DAgger expert:** e9, frozen.
   3. **Corpus:** a fresh demonstration corpus collected with e9 under v2. `apple-look-v1` was
      collected by the old collector under v1 physics and is not valid for BC here. Record hashes
      and splits by episode or session.
   4. **Perception bars:** set from e9's measured plate-error tolerance, using the calibration in
      TASK-070's results; re-derived through a calibration stage if the protocol needs it,
      following v1's calibration logic, with no looser bars.
   5. **Seeds:** fresh for everything, checked disjoint from every declared or spent range,
      including 46000–46999, 47000–47199, 50000–50299 and 50600–50631.
3. **Stay as in v1 unless a change is justified for the owner's ruling:** the development
   milestone (≥ 1/16 at rest on the development cohort); claims need cohort C against open-loop
   replay of the expert; the DAgger rounds; the budget (about 36 000 s global on MPS); the
   render-determinism check at the gated worker count.
4. **Flag every deviation from v1 in a table** (§3).

v1's rulings R1–R7 are carried as recorded verbatim in
[`apple_first_policy_v1.md`](apple_first_policy_v1.md) §0 and
[`apple_first_policy_v1_results.md`](apple_first_policy_v1_results.md) §0. R4 is carried
word for word: a first success counts as a **"learned policy with a DINOv2 encoder"**, never
"LeWM driving the robot".

## 1. The question, and the claim it can make

**Question.** Give a learned policy, at run time, the information e9 acts on and nothing
privileged:
- apple and plate xy, read once from the post-look frame by a learned readout on frozen DINOv2
  tokens;
- its own step counter;
- proprioception;
- the palm pose by forward kinematics of that proprioception.

Train it by behaviour cloning on e9's `apple-to-plate-v2` demonstrations and three DAgger
iterations labelled by e9. Does it pick the apple and leave it **at rest on the plate**
(`apple_at_rest_v0` after a latched grasp; T71-R1)
on **at least one** of the 16 development resets D2?

**Claim scope, fixed now.**
- **What M1-PASS would say:** a learned policy (rung L1, §7) with a frozen DINOv2 encoder left
  the apple at rest on the plate at least once on the non-gating development cohort of
  `apple-to-plate-v2`, with zero privileged reads at run time. It would be the project's first
  learned Apple→Plate success, as an existence result on `apple-to-plate-v2`.
- **What it would not say:**
  - That the policy works, or beats open-loop replay. Claims of that kind come only from M2 on
    cohort C (§12; v1 ruling R3).
  - That LeWM, or any world model, drives the robot (R4).
  - Anything about the v1 task, whose history and 0/150 benchmark are unchanged; the test split;
    another camera; or other resets.

Why this can work where TASK-056 failed is v1's argument (proposal §2, v1 §1) and is not
repeated. What v2 adds is that the demonstrator now actually rests the apple on the plate:
e9 passed TASK-070's gate at 32/32 (plate exact) and 30/32 (1.0 cm plate error) at rest on fresh
seeds, where v1's collector rested it on 4/32 (TASK-067 landing diagnosis).

## 2. What is carried from v1 unchanged

| piece | v1 source, reused as is |
|---|---|
| the look | `observation_reprobe.look_sequence` / `execute_look` (8 commands), through `first_policy_runtime.reset_and_look` |
| reset rule | `wide_reset` from `scripts/evaluate_apple.py`, imported, never reimplemented |
| encoder | pinned DINOv2 ViT-S/14 (digest `3a697b87…af27`; floor seed-0 digest `3d305f9c…c9db`), 112 → 224 px bicubic, CPU float32, read-out point `tokens` |
| readout | `first_policy_perception` (TASK-063's probe: linear and RBF kernel ridge, family and λ by inner CV), fit on 426 post-look frames, 10-fold cross-fitted estimates for the BC roots, frozen before any closed loop; DAgger frames are not added |
| R-arm floor | the same pipeline on the seed-0 random-init DINOv2's tokens |
| policy inputs | 132-d: estimates (4), clock (33), proprioception (86), palm pose by FK on the controller's own MuJoCo model (9); standardised by BC-0 train moments with std floor 1e-3 |
| head | LayerNorm → Linear(132, 512) → SiLU → Linear(512, 512) → SiLU → Linear(512, 512) → SiLU → Linear(512, 7); MSE to the clipped expert label |
| training | MPS, from scratch on each aggregate, 30 000 updates, AdamW 3e-4 cosine to 3e-5, wd 1e-4, clip 1.0, batch 256; selection every 1 000 updates by val MSE with per-dimension output std ≥ 0.02 |
| DAgger | 3 iterations × 128 fresh resets, β = 0, labels for every visited state with t < the expert's budget, retrain from scratch on the aggregate, select on val; C and R run their own DAgger on the same seeds |
| arms | P-0…P-3, C-3, R-3 (L1); F (L2); A4-look (L3); D-oracle-perc, B-oracle, B-hold, B-random, B-replay (L4) |
| embodiment path | clip to `configs/apple_wm_v4.yaml` bounds, `project_candidates`, `execute` |
| calibration logic | C0: nine conditions × 32 resets; apple 0.5/0.8/1.0/1.2 cm, plate 1.0/1.5/2.0/2.5 cm; the ≥ 28/32 rule and v1's caps (`first_policy.c0_bars`) |
| gates and rows | S0-P, S0-D1, the A4-look threshold (with the rate cap), `decide_m1`'s rows, the harness condition, the clause rows, M2's G1–G7 |
| guards | G-hash, G-weights, G-seeds, G-look, G-privileged, G-device, G-finite, G-cap, Q-split |
| devices, workers | training on MPS; features and readouts on CPU (6 threads); 8 CPU simulation workers, 1 torch thread each |
| v1 amendments 1–10 and the PR 2 clarifications | carried: the A4 rate cap, the counter installed after reset and scorer, the main-process estimate path with the G-frame re-render check on every attempt, no-eligible-checkpoint propagation, S0-D1 after BC-0 with the P→C→R fallback, B-random seeded 6703-style with one fixed sequence (seed now 7103), the per-root C-3 constant, TASK-056's 3-D-drift BC mask, G-look (b) on every attempt, the 300 s attempt cap as a G-cap |

## 3. Every deviation from v1 (for the owner's ruling)

Rows 1–5 are the brief's required changes. Rows 6–21 are consequences of them that this
preregistration had to fix, each with its reason. No row loosens a bar: with owner ruling
T71-R1 on row 19, the counted success requires both at rest and a latched grasp, which is
stricter than v1's latched success.

**Owner ruling T71-R1 (verbatim, 2026-09-28 UTC, via the coordinator):**

> **Accepted as written:** rows 7 (fresh D2), 9 (740 commands then settle), 14 (settle allowance f), 16, 20 and 21. So is everything you listed as unchanged.
>
> **Row 19 is accepted only in a tightened form.**
> - An at-rest success without a latched grasp is recorded and reported, with its count per arm.
> - It does NOT count toward the ≥ 1/16 development milestone, toward any M2 gate or claim, or toward "learned Apple→Plate successes".
> - For those purposes, a success is at rest (`apple_at_rest_v0`) AND a latched grasp before release.
> - Reason: a push or nudge that leaves the apple on the plate is not the pick-and-place the product claims. This tightens v1's condition; it does not loosen it.
> - With this change, the §3 preamble's "No row loosens a bar" becomes true. Make sure it reads correctly.

**How T71-R1 is implemented.** "Before release" is read as: the latched grasp stage is reached
at some step of the attempt's commands, before the task's settle (whose hands are open). The
same counted success is used everywhere a success is counted: C0 and the bars, S0's A4
threshold, the corpus's B-replay library, the harness condition, every M1 row and every M2 gate.
Wherever this document says "at rest" for a count that decides something, it means this
counted success; the plain at-rest count is reported beside it. The frozen block defines C0's
and M2's success as this counted success (`COUNTED_SUCCESS`).

**A case the literal reading admits, for the owner to confirm or tighten.** An attempt that
grasps, drops the apple off the plate and then pushes it onto the plate would count: the grasp
was latched before the settle and the apple ends at rest. The ruling's condition is met; its
stated reason (a push is not pick-and-place) is only partly served. Requiring the latched
`place` stage as well would exclude it. This protocol implements the ruling as written and
reports, for every counted success, its first grasp step and latched stages, so such a case would
be visible.

| # | item | v1 (TASK-067) | v2 (this protocol) | why |
|---|---|---|---|---|
| 1 | task | v1 scene, apple condim 3 | `apple-to-plate-v2`: apple condim 6, friction 1 / 0.01 / 0.001 (R12), applied by `apple_to_plate_v2.apply_v2_scene` in every worker | brief 2.1 |
| 2 | success (every arm, C0, M1, M2) | latched `AppleToPlateTask` success; the attempt ends at the first latched success | `apple_at_rest_v0` after the task's 60-step settle; the latched scorer is recorded beside it and never decides a row | brief 2.1 |
| 3 | demonstrator, DAgger labeller, B-oracle, A4-look, C0 expert | `scripted.apple_collector_policy` (745 commands, 8 phases) | e9 = `RestingPlaceExpert(release_pitch_rad=0.45, release_dx=0.015)`, frozen, 725 commands in 10 phases (orient 130, descend 80, close 45, lift 150, transfer 100, lower 50, steady 30, open 50, clear 30, retreat 60) | brief 2.2 |
| 4 | corpus | `apple-look-v1` (TASK-064; manifest `81d760d1…`), 134 non-aim train roots | a fresh `apple-look-v2`, collected by the gated run itself (§5): e9 under v2 on 200 fresh resets, split 170 / 20 / 10 by reset, sealed with per-episode sha256; 170 train roots in BC-0 | brief 2.3 |
| 5 | perception bars | C0 alone, caps apple 0.75 / 1.2 cm, plate 1.5 / 2.5 cm (median / p90) | C0 re-run with e9, at rest, on fresh seeds, with v1's rule and caps; **then the plate p90 bar is capped at 1.5 cm**, the level v1's rule gives on TASK-070's gated counts (30/32 at 1.0 cm, 28/32 at 1.5 cm); the plate median bar is at most the p90 bar (§8.1) | brief 2.4: e9's apple tolerance has never been measured, and TASK-070 measured the plate only up to 1.5 cm |
| 6 | seeds | 46000–46999 block; C0 directions `default_rng(6700)`; readout folds 6701; sampler 6702; random controller 6703; model 0 | 51000–51999 for every gated role, D2 52000–52015, smoke 52100–52199; C0 directions 7100, folds 7101, sampler 7102, random controller 7103, model 7104, corpus split 7105, noise-seed salt 7106 (§4) | brief 2.5 |
| 7 | development cohort | D = 45000–45007, 45100–45107 | **D2 = 52000–52015**, fresh | brief 2.5 ("fresh seeds for everything"): D was decoded by many arms in TASK-056/057 under v1 physics. **The milestone itself is unchanged** (≥ 1/16, now counted successes: at rest after a latched grasp, T71-R1). If the owner prefers v1's D, it is a small change before the gated run (the seed guard, which now forbids 45000–45207 as v1's did, the runner's whitelist and the frozen block), through a reviewed amendment. |
| 8 | cohort C (M2 only) | 45300–45339, stored values | **unchanged** | the brief keeps "cohort C against open-loop replay" |
| 9 | policy step budget per attempt | 800 policy steps, expert budget 745 | at most **740** policy commands (`resting_expert.EXPERT_BUDGET`, the budget e9 was gated under), then the task's 60-step settle: 800 steps in all | the 740-command budget TASK-070 gated e9 under, plus its settle (e9 itself used 725 + 60 = 785 steps); the at-rest check needs the settle inside the attempt. As in v1 (745–799), the policy's last steps (725–739) lie beyond e9's clock and carry no DAgger label |
| 10 | clock | t / 745 plus sin/cos at 16 periods | t / 725 plus the same sin/cos | e9's budget is 725 |
| 11 | F (fallback) | one head per collector phase (8) | one head per e9 phase (10), chosen by e9's clock schedule; still rung L2 | e9's schedule |
| 12 | B-replay library | successful non-aim train roots of `apple-look-v1` (latched) | train roots of `apple-look-v2` that ended in a **counted success** (T71-R1); replays their executed policy commands (no settle), then the task's settle | corpus and success change |
| 13 | G-privileged (2) | total `task_truth` calls = the scorer's evaluations | total = the scorer's evaluations + the at-rest records (two harness reads per executed step); calls inside `act()` must still be 0 | the at-rest check reads truth once per step, outside the controller |
| 14 | ladder | allowances (a)–(e) | adds **(f): the task's settle** (arm still, hands open, 60 steps) after the policy's commands. It is part of `apple_at_rest_v0`'s scoring procedure, identical for every arm including B-hold, and never learned | the settle is the task's, not the controller's |
| 15 | caps | as v1 | as v1, plus a corpus-collection cap of 3 600 s | the corpus is collected inside the run |
| 16 | the corpus's noise and plan | TASK-064: noise levels 0–3, aim offsets on every fifth root, 3 branches per root | noise levels 0–3 (root i gets level i mod 4; TASK-048's `Perturber`, loaded unchanged); **no aim offsets and no branches** | v1's BC-0 excluded aim-offset roots and branches, so they would add collection time and nothing that BC-0 reads |
| 17 | D2 harness condition | B-oracle (collector) ≥ 14/16 latched successes | B-oracle (e9) ≥ 14/16 counted successes (T71-R1); B-hold, B-random 0 latched grasps | rows 2–3 |
| 18 | M2 G5 | B-oracle ≥ 38/40 | B-oracle (e9) ≥ 38/40 counted successes (T71-R1) | rows 2–3 |
| 19 | what counts as a success (owner ruling T71-R1, below) | a latched success, which implied the latched grasp stage; `0 ≤ success ≤ grasp ≤ 16` | a **counted success** is `apple_at_rest_v0` **and** the latched scorer's grasp stage reached during the attempt's commands, before the settle (`first_policy_v2.counted_success`). An at-rest attempt without that grasp is recorded and reported, with its count per arm (`at_rest_without_grasp`), and never counts. v1's `0 ≤ success ≤ grasp ≤ 16` check is kept | T71-R1: a push or nudge that leaves the apple on the plate is not the pick-and-place the product claims. This tightens v1's condition |
| 20 | DAgger grasp label in `open`, `clear`, `retreat` | the collector's release had no state-dependent ramp | e9's grasp label there is `max(−1, accepted_grasp − 0.04)`, where `accepted_grasp` is the **learner's** last applied grasp, because the labeller advances on the executed result (as v1's) | a consequence of row 3 and v1's labeller semantics, disclosed |
| 21 | per-attempt wall cap for corpus roots | no corpus in the run | the corpus collector is covered by the corpus-stage cap (3 600 s) only; the 300 s per-attempt G-cap applies to the controller attempts (C0, DAgger, S0-D1, M1) | the collector is privileged data collection, not an evaluated attempt |

**What is deliberately not changed** (the brief's "stay as in v1" list): the milestone
(≥ 1/16, at rest, on the development cohort), the M2 gates including G1's "strictly more than
B-replay on the identical resets", the 3 DAgger rounds, the global budget of 36 000 s on MPS, and
the render-determinism check at 8 workers (§15).

## 4. Seeds

| use | seeds |
|---|---|
| corpus `apple-look-v2` (200 roots; 170 train / 20 val / 10 test) | 51000–51199 |
| perception-train | 51200–51455 (256) |
| perception-held-out (S0-P; the first 8 also S0-D1) | 51456–51583 (128) |
| C0 calibration | 51584–51615 (32) |
| DAgger iteration 1 / 2 / 3 (shared by P, C and R) | 51616–51743 / 51744–51871 / 51872–51999 (128 each) |
| development cohort D2 (M1) | 52000–52015 |
| reserved, unused | 52016–52099 |
| smoke runs and the render check only (nothing from them is read) | 52100–52199 |
| cohort C (M2 only; separate authorization) | 45300–45339, from stored manifest values |

RNG seeds: C0 error directions `default_rng(7100)` (one apple and one plate direction per C0
seed); readout folds 7101; training batches 7102; the random harness controller 7103 (one fixed
sequence per attempt, as v1 amendment 1 item 8); torch model initialisation 7104; the corpus split
7105; per-root noise seeds `default_rng(SeedSequence([seed, 7106]))`.

### 4.1 The seed-overlap check, recorded

It ran on 2026-09-28 at `fe696a0` (main after #93). The command, over `src`, `scripts`, `tests`,
`configs`, `docs`, `benchmarks` and `.mc`:
`grep -rnoE '\b5[12][0-9]{3}\b' --include='*.py' --include='*.json' --include='*.md' --include='*.yaml' --include='*.yml' --include='*.sh'`.
- **51000–51999: no hit at all.**
- 52000–52999: seven hits, none a seed and none in 52000–52199: 52890 and 52780 (distances in cm
  × 10⁵ in `apple_world_model_v4.md`, `claim_audit_v1.md` and `apple-world-model-v4.json`),
  52242 (candidates per second, `clean-reproduction.json`) and 52512 (candidate evaluations,
  `apple-control-commitment-v1.json`).
- No local corpus manifest (`data/*/meta/jepa_manifest.json`) records a reset seed in 51000–52999.
- The nearest declared ranges: 50000–50999 below (TASK-068 50000–50099, TASK-069 50100–50199,
  TASK-070 50200–50299 and 50600–50631, and 50500–50531, named and not used), which this
  protocol forbids as a whole block; nothing is declared above 50999.
- `first_policy_v2.check_seed_ranges` re-checks, as a guard, that every TASK-071 range lies
  inside 51000–52199 (the gated ones inside 51000–51999), is pairwise disjoint, and is off
  cohort C and every forbidden range: v1's list (which includes 45000–45207 with v1's D,
  47000–47199 and 48000–48999), 46000–46999 and 50000–50999. The runner also whitelists every
  simulated seed by role and refuses cohort C first.

## 5. The corpus `apple-look-v2` (stage 1 of the gated run)

- **Plan** (`first_policy_v2.corpus_plan`, fixed now). One root per seed 51000–51199: the
  `wide_reset(seed)` reset, split by reset (session) with `default_rng(7105)` — 20 val, 10 test,
  170 train — and noise level i mod 4 for the i-th root.
- **Episode.** Reset, the look, then e9 built from the reset truth (privileged; this is the
  collector), its every command passed through TASK-048's seeded OU-plus-burst `Perturber` at the
  root's level, clipped to the configured bounds, projected and executed. e9 advances on the
  executed result. After e9's 725 commands, the task's 60-step settle. The episode is scored by
  `apple_at_rest_v0` and by the latched scorer.
- **Stored per root** (`first_policy_v2_runtime.EPISODE_ARRAYS`, one `.npz` and one `.json`):
  every 112 px `onboard_rgb` frame from the post-look frame on (frame 0 is the post-look frame,
  and its sha256 must equal the worker's post-look render), proprioception, e9's clean command
  (`base`, the BC label source), the perturbed and the executed commands, e9's phase index, and,
  as labels only, the apple position, the dropped flag and hand contact. The settle is stored with
  phase index 10.
- **Sealing.** `manifest.json` holds the plan, the splits, every episode's sha256 and the run's
  provenance. Its sha256 is recorded in `report.json`. `CorpusReader` verifies every hash before
  decoding and refuses any episode outside train and val before a file is opened (Q-split);
  `test_split_decoded` is recorded. Q-split covers decoding stored episodes: the collector's own
  collection-time summaries of the test roots (at rest, latched, termination) are reported with
  the corpus counts, and no stored test episode is ever opened.
- **Location.** `data/apple-look-v2/run-<k>`; the runner refuses to overwrite it. A repeat after
  a void collects again into a new directory from the same plan and seeds (§14).
- **The corpus is privileged scripted data.** Its at-rest counts are reported by noise level and
  split. They are not learned results.

## 6. The controller and its training

**Perception.** As v1 §5.1, with the corpus's 170 train roots' frame 0 in place of
`apple-look-v1`'s frame 8: the readout is fit on 426 post-look frames (170 corpus train roots,
labels from the reset truth, plus the 256 perception-train resets). Estimates for val, DAgger
rollouts and every evaluation come from the full fit, computed in the main process at batch size
1 from the frame a worker renders on the attempt's own reset path; every attempt re-renders that
frame and must match its sha256 (G-frame).

**BC-0.** Every train root's policy steps (t < 725; the settle is never a row). The label is e9's
clean command, restricted to the 7 free dimensions and clipped to the configured bounds. The mask
is TASK-056's (`cloning.sample_mask`): a step is dropped when the apple's 3-D drift from frame 0
exceeds 1 cm while e9's phase is before `close`, or when the apple is dropped. Val: the 20 val
roots, prepared the same way, for selection only. The per-criterion accounting goes to
`report.json`.

**The policy P, C-3, R-3, their training and selection, and DAgger** are v1's (§2), with e9's
clock and budget: each attempt goes reset → the reset truth captured for the labeller → the look
→ the estimates → up to 740 policy commands → the settle. Every visited state with t < 725 is
labelled by e9 in shadow (`first_policy_v2_runtime.ShadowExpert`: e9 advanced on the executed
result, on the learner's clock). **The labeller is privileged, at training time only** (R2 (e)).

**The attempt** (`first_policy_v2_runtime.run_attempt`): at most 740 commands. A latched success
is recorded and does **not** end the attempt. A controller that exhausts (e9 after 725, a replay
at its end) moves on to the settle. A guard refusal, an infeasible command or a rejected command
ends the attempt as **not at rest** — a failure, not a void (as v1 and TASK-070). The 300 s wall
cap is a G-cap (a void). For e9 in command, this loop reproduces TASK-070's harness
(`resting_expert.run_attempt`) command for command and verdict for verdict; a graphics test pins
that.

## 7. What counts as "learned" (v1's ladder, with one allowance added)

| rung | at evaluation time | reported as |
|---|---|---|
| **L1: learned policy** | Every command after the look and before the task's settle comes from a trained network whose inputs are the onboard RGB frames, proprioception, FK of proprioception and the step counter. No scripted controller or phase switch, no replay, no substitution, **zero** privileged reads by the controller. | "learned policy with a DINOv2 encoder" (R4) |
| **L2: partially learned** | learned motor heads, but a scripted component selects the head or phase (F) | **partially learned**; never counted as a learned success |
| **L3: learned perception, scripted control** | a trained readout feeds a scripted controller (A4-look: e9) | not learned |
| **L4: privileged, scripted, replayed or substituted** | any run-time privileged read, e9 from truth, replay, D-oracle-perc | not learned |

**Allowed in L1, each labelled:** (a) the fixed look prefix; (b) the step counter; (c) the palm
pose by FK; (d) a perception readout trained on privileged reset labels, with zero privileged
reads at evaluation; (e) privileged-expert DAgger labels at training time only (v1's, by R2);
and **(f) the task's 60-step settle** after the policy's commands: arm still, hands open,
identical for every arm, part of `apple_at_rest_v0`'s scoring procedure (TASK-068/070), never
learned. **Not allowed:** no L1 arm's controller may read `sim.task_truth()`, any label, or the
scorers. G-privileged enforces this.

## 8. Stage 0: calibration and offline gates (inside the gated run, before any closed loop)

### 8.1 C0: e9's tolerance curve, at rest (scripted, privileged; calibration only)

On the 32 C0 seeds, with the look, e9 is built from the reset truth plus an injected xy error of
fixed size, one apple and one plate direction per seed from `default_rng(7100)`, with no action
noise. Nine conditions × 32 attempts: the reference; apple error 0.5, 0.8, 1.0 and 1.2 cm with the
plate exact; plate error 1.0, 1.5, 2.0 and 2.5 cm with the apple exact. **Success is the counted
success (at rest after a latched grasp; T71-R1).** The plain at-rest count, the
latched count and each attempt's final distance are reported beside it.

**The rule** (`first_policy_v2.c0_bars`):
1. v1's rule, unchanged (`first_policy.c0_bars`): the p90 bar is the largest level at which that
   level and every smaller one reach at least 28/32; the median bar is min(cap, p90 bar); caps
   0.75 / 1.2 cm (apple) and 1.5 / 2.5 cm (plate); **CAL-ESCALATE** if the reference or the
   smallest level of either quantity is below 28/32.
2. **The TASK-070 plate ceiling.** v1's rule applied to e9's gated counts in TASK-070 (report
   sha256 `27543757…099f`: 30/32 at 1.0 cm and 28/32 at 1.5 cm, both at rest, on 50600–50631)
   gives 1.5 cm, the largest level measured there. The plate p90 bar is min(the C0 result,
   1.5 cm), and the plate median bar is at most the p90 bar. With v1's plate median cap of 1.5 cm,
   **both plate bars are therefore at most 1.5 cm.** This can only tighten: v1's plate p90 cap
   was 2.5 cm, and no evidence exists that e9 tolerates more than 1.5 cm on fresh seeds.
3. The apple bars come from C0 alone: e9 has never been calibrated for apple error.

The bars can only tighten below v1's caps, never loosen. C0 is not re-run on these seeds after a
CAL-ESCALATE.

### 8.2 The A4-look threshold, S0-P and S0-D1

As v1 §7.1a–§7.3, on C0's counted-success counts (T71-R1): T = ⌈0.5 × 16 × the mean predicted A4-look rate⌉,
at least 1, with each per-reset rate capped at 1.0; S0-P passes when the median and p90 of the
held-out apple and plate errors are each at or below their bars (apple fails → S0-APPLE-FAIL;
apple passes and plate fails → S0-PLATE-FAIL); S0-D1 checks, on the first 8 held-out seeds, that
the runner's live input and command equal the offline ones within 1e-4.

## 9. Stage 1: M1 on the development cohort D2

After S0 passes and the three DAgger pipelines have finished, **each M1 arm runs each of the 16
D2 resets exactly once**: P-0 to P-3, C-3, R-3, A4-look, D-oracle-perc, B-oracle (e9 from truth),
B-hold, B-random and B-replay. P is evaluated on D2 four times (k = 0–3), as R3 allows, and every
evaluation is reported. DAgger never uses D2. The seed guard whitelists D2; every other seed is
refused, cohort C first.

## 10. Pre-declared outcomes (first matching row; `first_policy_v2.decide_m1` for M1)

`success` below is the **counted success** (at rest after a latched grasp before the settle;
T71-R1); "grasps" are the latched scorer's grasp stage. At-rest attempts without a grasp are
reported per arm and never counted.

| row | condition | reading | next (a recommendation; the owner chooses) |
|---|---|---|---|
| **V** | a guard fails (§13), a crash or cap, S0-D1 fails, or the harness fails on D2 (B-oracle < 14/16 at rest, or B-hold or B-random > 0 grasps) | nothing is read | one from-scratch repeat (§14) |
| **CAL-ESCALATE** | §8.1 | e9 is not as tolerant as assumed | stop; the owner decides; the clause does not fire |
| **S0-APPLE-FAIL** | §8.2 | the post-look frame, through this readout, does not give the apple to e9's tolerance on fresh resets | **the clause fires** (§11) |
| **S0-PLATE-FAIL** | §8.2 | the apple is read, the plate is not | stop; the owner decides; the clause does not fire |
| **M1-PASS** | some P-k reaches a counted success (at rest after a latched grasp, T71-R1) on ≥ 1/16 of D2 | **the first learned (L1) Apple→Plate success, on the non-gating development cohort of apple-to-plate-v2: an existence result** (R3), a "learned policy with a DINOv2 encoder" (R4) | M2 on cohort C, under a separate authorization. The carried arm is the P-k with the most D2 successes, ties to the later k: a selection on D2, declared as one. |
| **M1-MOTOR** → F | every P-k 0/16; A4-look ≥ T | perception is adequate in the loop; motor learning fails | F is trained and runs once on D2 |
| **M1-MOTOR-F-PARTIAL** | as above, and F ≥ 1/16 | a **partially learned** (L2) success; never counted as learned | the owner decides |
| **M1-MOTOR-F-NONE** | as above, and F 0/16 | neither the learned policy nor the phase-decomposed one succeeds | **the clause fires** |
| **M1-PERCEPTION** | every P-k 0/16; A4-look < T | the estimates do not survive the closed loop, or are not accurate enough in it | **the clause fires** |

**Reported whatever the row:** the C-3, R-3, D-oracle-perc and B-replay counts (C-3 can reach
≥ 1/16; a P-k success then says little about vision until M2's G2); every arm's at-rest, latched
and grasp counts, stage occupancy, termination reasons, step counts, final distances and signed
command statistics; S0-P against its bars and C0's full table; the corpus's counts; the DAgger
data sizes, every val curve and selected update. `M1-MOTOR` is an intermediate state, never a
final row. **A 0/16 on every P-k is reported as 0/16.**

## 11. Abandonment clause

**It fires on S0-APPLE-FAIL, M1-MOTOR-F-NONE or M1-PERCEPTION.**
- **What closes:** learned control on `apple-look-v2` (e9's demonstrations under
  `apple-to-plate-v2`) at the 112 px onboard camera. No further head, loss, input, DAgger or
  readout variant is preregistered on this corpus and camera without new evidence of a different
  kind.
- **The conclusion is the one `apple_policy_v1.md` §7 pre-declared, quoted verbatim in the
  results:** "the honest published conclusion is that a 112 px onboard camera plus a single-mode
  scripted-collector corpus does not support learned Apple→Plate on this platform, and the
  product goal needs a data or hardware change — not another model."
- **What does not close:** the LeWM backend, DINOv2 as an encoder, the product goal, v2 as a task
  definition, and the corpus (sealed; test split unread).

## 12. M2: the gated test on cohort C (preregistered now; run only under a separate authorization)

v1 §11, carried, with success the counted success of T71-R1 ("at rest" in the table means it):
the 40 cohort-C resets from
`benchmarks/manifests/apple-policy-v1.json`'s stored values, never recomputed; arms the carried
P-k, C-3, R-3, B-replay (nearest counted-success `apple-look-v2` train root), B-oracle (e9 from truth),
B-hold, B-random; the stop rule (the carried P-k does not run on C with 0/16 grasps on D2, and
M2 fails; the controls always run).

| gate | condition |
|---|---|
| G1 | the carried P-k ≥ 17/40 counted successes **and** strictly more than B-replay on the identical resets |
| G2 | P − C-3 ≥ +8 counted successes (exact McNemar, realised n_d and p reported) |
| G3 | P − R-3 ≥ +8. Declared reading if it fails: "a learned visuomotor policy works; encoder pretraining contributes nothing measurable" |
| G4 | the carried P-k ≥ 20/40 grasps |
| G5 | B-hold and B-random 0/40 grasps; B-oracle (e9) ≥ 38/40 counted successes. A failure invalidates the run, not the arms |
| G6 | zero controller privileged reads by any L1 arm |
| G7 | median control time ≤ 100 ms per command, including the one DINOv2 forward pass |

Rows: M2-VOID (G5 or G6), M2-PASS (all), M2-FAIL-VISION (G1 and G4 pass, G2 fails), M2-FAIL
(anything else). A gate that cannot be evaluated counts as failed. `exemption_spent` is `false`
and not claimed. A pass would mean "a learned policy with a DINOv2 encoder works on Apple→Plate on
this cohort under apple-to-plate-v2", not that LeWM drives the robot.

## 13. Guards (any failure is V)

v1's guards (§2), with these v2 specifics:

| guard | condition |
|---|---|
| **G-frozen / G-hash** | the manifest's `frozen` block equals the module; every pinned file (the manifest's `hashes`) matches at preflight and at the end; the tracked tree is clean at both; MuJoCo is 3.13.0 |
| **G-scene** | every simulation worker's robot is `apple_to_plate_v2` (apple condim 6, friction 1 / 0.01 / 0.001), checked when the worker starts |
| **G-data** | the corpus has 170 / 20 / 10 roots by the plan; every read episode matches its sealed sha256; every read root's frame 0 equals its post-look render |
| **Q-split** | only train and val corpus episodes are decoded; `test_split_decoded: false` is recorded |
| **G-weights** | the DINOv2 files and the pretrained and floor digests match their pins |
| **G-seeds** | `check_seed_ranges`, and every simulated seed lies in its role's range; the D2 whitelist; cohort C refused |
| **G-look** | on every attempt, the applied look commands equal the requested ones, and the post-look joint state equals the run's first attempt's within 1e-6 |
| **G-privileged** | for every evaluation attempt of an L1 (and F) arm: 0 `task_truth` calls while `act()` is on the stack, and the total equals the scorer's evaluations plus the at-rest records; L1 controllers hold no robot or simulator handle; FK runs on the controller's own model |
| **G-frame** | every attempt's post-look frame equals the frame its estimates came from |
| **G-device, G-finite, G-cap** | as v1; the caps are §15's |

## 14. Void rule

- **A run that stops early other than at a declared early-stop row (CAL-ESCALATE, S0-APPLE-FAIL,
  S0-PLATE-FAIL) is V.** That covers a guard, a crash or a cap. The run records a
  `void_reason`, and nothing in it is read. A declared early-stop row is an outcome, not a void,
  and it is never repeated.
- **Exactly one from-scratch repeat** is allowed, into `run-2` (output, checkpoints and a new
  corpus directory), with the same seeds, caps and device. The corpus is collected again from the
  same plan.
- **A second V closes TASK-071 as INCONCLUSIVE.**
- **Any guard change needs an owner ruling.** A fix between runs is otherwise limited to runner
  mechanics, goes through a reviewed PR and a fresh pre-run GO, and is disclosed.

## 15. Budget, devices, caps

**Devices.** Training on **MPS**; DINOv2 features, readouts and statistics on CPU (6 threads,
float32); simulation and rollout inference in **8** CPU workers, one torch thread each.

| cap | seconds |
|---|---|
| global | 36 000 |
| corpus collection | 3 600 |
| each training | 1 800 |
| each rollout batch | 3 600 |
| C0 | 3 600 |
| perception collection | 1 800 |
| each attempt | 300 |

**Expected time: about 2–3 h.** Anchored on TASK-070's gate (96 e9 attempts of 785 steps in
301 s on 8 workers) and one measured e9 attempt through this runner's loop (about 10 s on one
core):
- the corpus, 200 roots with frame storage, about 5–10 min;
- perception frames, a few minutes; C0, 288 attempts, about 6 min;
- 12 trainings, a few minutes each on MPS;
- DAgger, 1 152 rollouts on 8 workers, about 30–40 min;
- M1, about 200 attempts, about 5 min; F only on M1-MOTOR.

**The render-determinism check** (v1 ruling R7, carried): before the pre-run GO, the pre-run
reviewer runs `scripts/check_first_policy_v2_render.py` at the gated worker count (8) on the smoke
seeds and confirms the verdict `IDENTICAL` (every seed one sha256 across four renders on at least
two workers, including after full-length attempts; the negative control raises G-frame). If it is
not identical, the run does not start and the issue goes back to the owner. It is not relaxed
silently.

**Output.** `outputs/task071-first-policy-v2/run-<k>/`, checkpoints in
`checkpoints/task071-first-policy-v2/run-<k>/`, the corpus in `data/apple-look-v2/run-<k>/`. The
runner refuses to overwrite any of them. Nothing is committed except manifests and hashes.

## 16. Process

1. **This PR** holds the protocol, the manifest, the design, runtime and model modules, the
   runner, the render check, their tests and the card, and records the smoke runs (§18). It
   merges on an independent reviewer's reported APPROVE and green CI; the owner merges.
2. **The gated run** starts from a clean checkout of `main`, and only on a fresh pre-run
   reviewer's **reported** GO, delivered as a message, after the render check (§15):

   ```sh
   uv run --no-sync python scripts/run_first_policy_v2.py run \
       --output outputs/task071-first-policy-v2/run-1 \
       --checkpoints checkpoints/task071-first-policy-v2/run-1 \
       --corpus data/apple-look-v2/run-1
   ```
3. **The results PR** holds the results document and the results manifest. It states the outcome
   plainly, including 0/16, and carries every restated number with its count. A reviewer checks
   every number against `report.json`. If the clause fires, it also adds a `docs/DECISIONS.md`
   entry.
4. **One task, one agent.** A defect found after the freeze goes to the owner and is fixed only
   through a disclosed amendment. Nothing is re-thresholded, retrained or re-selected after
   numbers are seen, beyond what §9 and §10 declare.

## 17. Not done (declared)

- No world model is in the loop: no critic and no latent MPC.
- No per-step image features, no end-to-end image head, no chunked or diffusion head.
- No apple-hidden spurious check for the readout; S0-P measures its accuracy on fresh resets.
- No readability gate on `apple-look-v2` separate from S0-P (TASK-064's gate was on
  `apple-look-v1`). S0-P on 128 fresh held-out resets is the perception gate here, as in v1.
- No test-split decode, and no cohort-C attempt in this gated run.
- Nothing about the v1 task changes: its scorer, its history and the 0/150 MVP benchmark are
  untouched.

## 18. Smoke runs (runner checks; nothing in them is read)

All ran on CPU in the author's worktree, on the smoke seeds only (corpus 52100–52105, the other
roles two seeds each from 52110 on, D2's stand-in 52170–52171), with 2 workers, 20 updates and
noise targets for the readout and the policy labels. Their outputs are under
`outputs/task071-scratch/` (git-ignored). **Nothing in them is read**; they check the runner.

| run | revision, tree | policy steps | wall | end | what it exercised |
|---|---|---|---|---|---|
| smoke-a | `fe696a0` + uncommitted draft | 40 | 55 s | V (harness) | every stage end to end |
| smoke-b | `8bda8c5`, clean at start | 740 | 369 s | V (G-hash at the end) | full-length attempts; the doc was edited during the run, and the end-of-run G-hash caught it, as designed |
| smoke-c | `be74726`, clean | 40 | 55 s | V (harness) | superseded by smoke-e (before T71-R1) |
| smoke-d | `be74726`, clean | 740 | 369 s | V (harness) | superseded by smoke-f (before T71-R1) |
| smoke-e | `401403b`, clean | 40 | 55 s | V (harness) | superseded by smoke-g (frozen-block fix after re-review) |
| smoke-f | `401403b`, clean | 740 | 370 s | V (harness) | superseded by smoke-h (frozen-block fix after re-review) |
| **smoke-g** | **`82cb197`, clean** | 40 | 55 s | V (harness) | every stage; report sha256 `fed5eb93…b84a` |
| **smoke-h** | **`82cb197`, clean** | 740 | 369 s | V (harness) | every stage at full length; report sha256 `323b745a…91d0` |

`401403b` implements owner ruling T71-R1 (§3, row 19); `82cb197` makes the frozen block define
C0's and M2's success as the counted success, after re-review. In every smoke the row is V
because the harness condition needs B-oracle ≥ 14/16 and a smoke has two D2 stand-in resets
(v1's smokes ended the same way). What smoke-g and smoke-h show about the mechanics, at
`82cb197`:
- the corpus stage wrote, sealed and read back 6 roots; the reader decoded 4 train + 1 val
  episodes and `test_split_decoded` is false; every read root's frame 0 matched its post-look
  render;
- at full length (smoke-h) e9 through the collector ran all 725 commands on all 6 roots; 4 of 6
  ended in a counted success (at rest after a latched grasp) under noise levels 0–3; C0 was a
  counted success on 2 of 2 in every condition (smoke counts are scaled to 32), so the bars came out at the caps with the
  TASK-070 plate ceiling applied (plate 1.5 / 1.5 cm); B-oracle was a counted success on 2 of 2
  and B-replay on 1 of 2; no arm had an at-rest attempt without a grasp;
- S0-D1: live and offline inputs identical, commands within 3e-8;
- G-privileged held on every learned attempt (controller reads 0; total = scorer evaluations +
  at-rest records); the end-of-run pins (30) and revision matched; no non-finite field.

**Render check** (author's, not the pre-run reviewer's): `scripts/check_first_policy_v2_render.py`
at `82cb197`, 8 workers, clean tree: **IDENTICAL** — 32 of 32 seeds rendered on two or more
workers, 8 distinct workers, 106 s; report sha256 `08b32d89…7024`
(`outputs/task071-scratch/render-determinism-3`). Earlier checks at `be74726` and `401403b` were
also IDENTICAL. The pre-run reviewer re-runs it (§15).

**Tests.** `tests/test_first_policy_v2.py`; its two graphics tests (`JEPA_TEST_RENDER=1`, both
passed at `82cb197`) check that the v2 attempt loop reproduces TASK-070's harness for e9 command
for command and verdict for verdict, and that the collector at noise level 0 issues exactly e9's
commands with frame 0 equal to the post-look frame. A stand-in-robot test checks that an attempt
whose apple rests on the plate is counted only when the latched grasp is reached before the
settle (T71-R1), and a runner test that `count()` counts only counted successes and reports
at-rest attempts without a grasp separately.

After `82cb197` only this section and the manifest's pin of this document changed.

## 19. Amendment log

*Empty.*

**Learned Apple→Plate is still 0 successes.**
