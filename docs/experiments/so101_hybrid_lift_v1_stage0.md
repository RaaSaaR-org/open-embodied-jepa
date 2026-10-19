# TASK-102 Stage 0: G_loc trained, every check passed before any count, τ_h = 0.08 rad from P, and no configuration hands over by tolerance on A (S0-APPROACH) (record)

**Status: Stage 0 record. Row: S0-APPROACH (§7.7 row 3, by its first clause: the configuration
chosen on A handed over by tolerance on 0 of 32 of A's resets, below 16/32). D did not run. The
protocol is not frozen, and there is no gated stage. Escalated.**
Protocol: [so101_hybrid_lift_v1.md](so101_hybrid_lift_v1.md) §7. Rulings R28.110–R28.118 in
[DECISIONS.md](../DECISIONS.md), decision 2026-10-20 (bz), **recorded by Claude under owner
delegation**. They record the row and the deviations. They do not choose the next study:
S0-APPROACH escalates.

**Nothing here is a result.** Every closed loop ran on development cohorts P (103000–103031) and A
(103500–103531) or on debug seeds (103900–103905). D's 64 goal demonstrations (103100–103163) were
drawn, rendered and checked (7.3.2), but no arm ran on D. P′, A′ and D′ were not used. No gated seed
(103200–103327, reserve 103400–103463) was drawn, rendered or run. This is a simulation-only SO-101
study on `so101_cube_lift_v1`, not Apple→Plate. R7 does not change (R28.9). Its world model is
`native_jepa` (J-100101), not LeWM.

## Summary

- **G_loc trained as declared** (§5.4, §7.2). 3 029 train windows (210 022 frames) and 168 val
  windows (11 594 frames) from `so101-play-v1`'s train and val splits; the test split was never
  opened. All three seeds had eligible checkpoints. The selected val MSEs were 0.02712 (G-0,
  update 7 000), 0.02762 (G-1, update 7 000) and 0.02756 (G-2, update 5 000). **The primary G_loc is
  G-0.** No seed set `last_two_triggered`: every val curve was lowest early and rose afterwards
  (0.0344–0.0350 at update 30 000).
- **Every §7.3 check passed at the running revision `c586eb1`, before P's first episode, and the
  runner enforced it.** The live latent equalled the stored latent on 64 of 64 frames (0 ulps). The
  projector agreed on 1 000 of 1 000 sequences. G_loc's live path matched the batched prediction
  to 7.2 × 10⁻⁷ for all three seeds. Determinism held even in the literal form (all six pairs bit
  identical). The longest command took 0.98 s against the 30 s deadline. R-enc read the cube at
  0.951, 0.968 and 0.970 cm in median on P's, A's and D's subgoal frames, against
  B_cube = 2.397 cm.
- **P: G_loc lifts after an exact privileged approach.** S+G lifted **30/32** at r = 0 (exact 95 %
  interval 0.792–0.992), 31/32 at 0.04 rad (0.838–0.999), 28/32 at 0.08 rad (0.710–0.965) and 19/32
  at 0.12 rad (0.406–0.763). The rule gives **τ_h = 0.08 rad**: k(0.08) = 28 ≥ k(0) − 3 = 27, and
  k(0.12) = 19 < 27. k(0) ≥ 16, so S0-GLOC-WEAK does not fire on P.
- **A: CEM over J never handed over by tolerance.** Phase 1 alone, in each of TASK-101's four
  configurations, ran to the 60-command timeout on all 32 of A's resets. So **a_A = 0/32 in every
  configuration** (exact 95 % interval 0.000–0.109). The tie-break chose H16-last, which had the
  lowest median closest arm RMS to subgoal 1 (0.354 rad). Its a_A = 0 < 16, so **the row is
  S0-APPROACH and D does not run** (§7.5, §7.7 row 3).
- **The demonstration was kept on 32/32 of P's, 32/32 of A's and 64/64 of D's drawn seeds** (and on
  6/6 debug seeds). No seed was rejected.
- **GPU: 1.246 GPU h in Stage 0** (23 lock-holding jobs), against the 5 GPU h Stage 0 cap and the
  12 GPU h hard cap. This total includes a superseded first round of checks (0.115 GPU h) and two
  debug plumbing runs (0.107 GPU h).

## 1. Code (new files only; the run tree `worktrees/so101-hybrid-stage0-run` was at `c586eb1`)

| file | what it is |
|---|---|
| `src/embodied_jepa/so101_hybrid.py` (**new**, NumPy) | cohorts, seeds and salts; `HandoverRule` (§5.2); `tau_env_rule` and `tau_h_rule`; P's offsets; G_loc's clock and non-latent inputs (§5.3); `grasp_windows` (§5.4); G_loc's selection and primary rule; the arms' composition (§6); A's configuration rule (§7.5); the bar (§7.6); the Stage 0 and gated rows (§7.7, §9); the ledger verdict with this task's checks; the S projection (§13) |
| `src/embodied_jepa/so101_grasp_policy.py` (**new**, torch) | G_loc's head (P-3's), input moments, `act` (clipped) and `predict`, the training loop and checkpoints (implementation hash = this file's sha256) |
| `src/embodied_jepa/so101_hybrid_runtime.py` (**new**) | the window builder (hash checks, the alignment guard and the latent cache), the privileged aims (follower v1's `_pick` aims at the reset), S+G's scripted approach, W+F's `FollowerTail`, phase 1 (TASK-101's planner settings, imported), `HybridController`, the episode loop, G-zero for the follower path, the worker pool, the determinism pairs, 7.3.1 |
| `scripts/train_task102.py`, `scripts/run_task102.py`, `scripts/summarize_task102.py`, `scripts/run_task102_stage.sh` (**new**) | `windows`, `train`, `select`; `goals`, `check`, `episodes` (the ledger is enforced for P, A and D before any controller, simulator or seed is built); `tau`, `config`, `stage0`; one `gpu_run.sh --wait --board` lock per GPU job |
| `tests/test_so101_hybrid.py`, `tests/test_so101_grasp_policy.py` (**new**) | 44 tests (§7.1's list), including the window builder on a synthetic corpus, G-zero on debug seeds and the episode loop for every kind of arm |
| `benchmarks/manifests/so101-hybrid-lift-v1.json` (**new**) | this task's pins, status **CANDIDATE** (not frozen) |
| `CLAUDE.md` | one sentence listing the entry points |

**Byte-identical (tests check them):** every G1 byte-pinned file, every TASK-100 file
(`frozen_files_check`), every file and composed file pinned by TASK-101's manifest (still
CANDIDATE), `so101_switch_envelope.py`, `schema_planning.py`, `so101.py`, `so101_simulation.py`,
`so101_tasks.py`, `so101_followers.py`, `so101_play_corpus.py` and `config.py`. No existing test
needed an exemption. Full suite at `c586eb1`: 2 725 passed and 36 skipped (graphics opt-in and
similar). Ruff check and format are clean.

## 2. G_loc's training (§5.4, §7.2)

**Windows** (CPU job at `4d81c56`; 486 s). The job checked the corpus pins (`corpus.json` and all
30 shard manifests against TASK-099's manifest), TASK-100's feature report (`afe605d5…`, the
sha256 in TASK-100's evidence) and every train and val store file. Each episode file and sidecar
was checked against its shard manifest as it was read. The store-corpus alignment guard held on
every window.

| split | episodes | attempts | windows | frames | not grasped | `early_close` | no `lift_end_step` | window commands (min / median / max) | holds (median) | windows with the task's success inside |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 2 700 | 5 781 | **3 029** | **210 022** | 1 355 | 275 | 1 122 | 39 / 68 / 111 | 13 | 1 901 (62.8 %) |
| val | 150 | 302 | 168 | 11 594 | 72 | 15 | 47 | 45 / 68 / 97 | 12 | 100 (59.5 %) |

Train and val windows share no episode. Test-split files were never opened: the loader refuses the
split before touching any file, and the corpus check hashes only the shard manifests and split
files.

**Training** (one GPU job per seed, at `07bb8b0`; 228 s, 230 s and 221 s of lock time).

| seed | init seed (from `default_rng([10115, i])`) | selected update | val MSE | eligible points | `last_two_triggered` | val MSE at 30 000 | checkpoint sha256 |
|---|---|---|---|---|---|---|---|
| **G-0 (primary)** | 2423198619069167376 | 7 000 | **0.027120** | 30 of 30 | no | 0.034413 | `656f26d5…` |
| G-1 | 1485459417062414512 | 7 000 | 0.027616 | 30 of 30 | no | 0.034564 | `c38a2409…` |
| G-2 | 2548269044083312364 | 5 000 | 0.027556 | 30 of 30 | no | 0.034961 | `7ba6b715…` |

The curves fall to their minimum within 5 000–7 000 updates and then rise, which is overfitting of
the train windows. Selection on val takes the early point, as declared. In 62.8 % of the training
windows the cube rises at least 5 cm and is held for 20 steps inside the window. So the corpus does
contain task-level successes, and P's S+G count below confirms that G_loc learned to use them.

## 3. Checks (§7.3; every one at `c586eb1`, from a clean tracked tree)

| check | finished (UTC) | outcome |
|---|---|---|
| 7.3.1 live latent = stored | 21:49:18 | **pass.** 64/64 frames bit identical (0 ulps) |
| 7.3.3 projector (P's reset states) | 21:49:30 | **pass.** 1 000/1 000 equal to chained `preview` and to `execute`, 0 stopped |
| 7.3.4 G_loc's live path | 21:49:34 | **pass.** On 256 val frames, the live `act` (inputs rebuilt from the joints, velocities, targets and clock) against the batched prediction: max |Δ| 6.9 × 10⁻⁷ (G-0), 7.2 × 10⁻⁷ (G-1, G-2). The rebuilt inputs equalled the stored ones |
| 7.3.5 determinism (conditional form) | 21:55:25 | **pass, and the literal form holds too.** H and S+G each run twice; four H episodes in a pool of 1 and of 6: all six pairs bit identical (latents, actions, outcome) |
| 7.3.6 timing | 21:55:25 | **pass.** Phase 1: 0.180 s per command (median; 95th percentile 0.181) per worker. Phase 2: 0.148 s (0.151). With 6 workers, 0.128 s of lock time per command. Longest command 0.98 s, against the 30 s deadline. Projection (upper bounds at full budgets): P 0.89, A 0.29, D 2.71, S 7.25 GPU h |
| 7.3.2 R-enc on P's goals | 21:55:39 | **pass.** 0.951 cm median (per subgoal 0.91, 1.01, 1.20, 0.88; reset frames 1.51) |
| 7.3.2 on A's goals | 21:55:52 | **pass.** 0.968 cm (0.97, 1.05, 1.07, 0.84; reset frames 1.46) |
| 7.3.2 on D's goals | 21:56:14 | **pass.** 0.970 cm (0.98, 0.98, 0.98, 0.90; reset frames 1.37) |

**Ordering:** P's episodes started at 21:56:17Z, after the latest required check (21:55:39Z for P's
goal store). A's started at 22:19:02Z, after 21:55:52Z. The runner refused any ledger cohort
without the checks, and the summariser confirmed both orders before reading any count.

## 4. P: G_loc's tolerance at the hand-over, and τ_h (§5.2, §7.4)

S+G on P's 32 resets: the scripted approach drives to subgoal 1's IK target plus δ (RMS r) for 40
commands, then G_loc runs from command 41. **G-zero held:** at r = 0 the approach executed exactly
the demonstration's first 40 commands on all 32 resets.

| r (rad) | lifts of 32 | exact 95 % | achieved hand-over arm RMS (median; range) | failures |
|---|---|---|---|---|
| 0 | **30** | 0.792–0.992 | 2 × 10⁻⁸ | 2 timeouts |
| 0.04 | **31** | 0.838–0.999 | 0.0400 (0.0400–0.0400) | 1 timeout |
| 0.08 | **28** | 0.710–0.965 | 0.0800 (0.057–0.0800) | 4 timeouts |
| 0.12 | **19** | 0.406–0.763 | 0.1200 (0.065–0.1200) | 13 timeouts |

(Where the achieved RMS is below r, a joint limit clipped the offset.) No episode stopped and no
cube was lost.

**τ_h, mechanically.** Step 1: TASK-101 A2's "pre" column (`envelope.json`, `0355aba9…`) lifted
32/32 at every level, so τ_env = 0.16 rad and **τ_max = 0.12 rad**. Step 2: k(0) − 3 = 27.
k(0.04) = 31 and k(0.08) = 28 qualify, and k(0.12) = 19 does not. So **τ_h = 0.08 rad** (no
fallback). k(0) = 30 ≥ 16, so S0-GLOC-WEAK does not fire.

*Reading, not a row:* the learned grasp works from the pre-grasp pose and tolerates about 0.08 rad
there. That is less than the privileged follower's 0.16 rad, as §5.2's margin anticipated, and it
falls off at 0.12 rad.

## 5. A: phase 1's configuration (§7.5)

Phase 1 alone on A's 32 resets in each configuration, at τ_h = 0.08 rad, until the hand-over fired
(128 approach episodes over J-100101).

| configuration | a_A (by tolerance, of 32) | exact 95 % | commands used (median) | closest arm RMS to subgoal 1, median (min) | hand-over arm RMS, median (min) | timeouts |
|---|---|---|---|---|---|---|
| H8-last | **0** | 0.000–0.109 | 60 | 0.384 (0.039) | 0.530 (0.071) | 32 |
| H8-mean | **0** | 0.000–0.109 | 60 | 0.482 (0.333) | 0.880 (0.441) | 32 |
| H16-last | **0** | 0.000–0.109 | 60 | **0.354** (0.104) | 0.436 (0.104) | 32 |
| H16-mean | **0** | 0.000–0.109 | 60 | 0.519 (0.247) | 0.858 (0.347) | 32 |

**The rule:** all four tie at a_A = 0. The first tie-break, the lower median closest RMS, chooses
**H16-last**. Its a_A = 0 < 16/32, so **S0-APPROACH**, and D does not run.

*Reported only (from A's per-step logs; no row reads these):* across the 128 episodes, the arm
came within τ_h = 0.08 rad of subgoal 1's joints in only 3 episodes (all H8-last; 12 commands in
total). In none of them was the jaw within 0.15 rad at the same time. Subgoal 1's jaw is open
(1.745 rad in median). Phase 1's jaw stayed far from it: at the closest arm state the median jaw
error was 0.64–0.95 rad by configuration. So the approach failed on the arm, by 0.35–0.52 rad in
median, and also on the jaw. These readings match K′'s in TASK-101 (closest arm RMS 0.26–0.48 rad
in median). The fresh cohort and the looser tolerance did not change the picture.

## 6. The row (§7.7, first match)

1. S0-VOID: no check, guard or job failed. Every run completed (one superseded check round; §8
   deviation 1).
2. S0-GLOC-WEAK: every G_loc seed had an eligible checkpoint, and k(0) = 30 ≥ 16 on P. It does
   not fire (S+G on D did not run).
3. **S0-APPROACH: the configuration chosen on A handed over by tolerance on 0/32 < 16/32. It
   fires.** D does not run, so neither a_H, W+F, the bar nor the power is computed.

**Row: S0-APPROACH. Escalate. No gated stage.** The candidates §7.7 names, each needing its own
amendment or preregistration, are R28.57's LeWM retrain, a cost or goal change for the approach,
and a planner variant. This record does not choose among them.

## 7. GPU hours (§13; lock time from `~/.local/state/gpu/oej-gpu_run.log`, `--who oej:task102-s0`)

| jobs | revision | GPU h |
|---|---|---|
| G_loc training, 3 seeds | `07bb8b0` | 0.190 |
| goals: debug (6), P, A, D | `07bb8b0` | 0.021 |
| debug plumbing (2 episode jobs on 103900–103903; never counted) | `07bb8b0` | 0.107 |
| checks, first round (superseded; §8 deviation 1) | `e0412e5` | 0.115 |
| checks, second round | `c586eb1` | 0.118 |
| P (128 S+G episodes) | `c586eb1` | 0.373 |
| A (128 approach episodes) | `c586eb1` | 0.321 |
| **total** | | **1.246** (cap: Stage 0 ≤ 5, overall ≤ 12) |

The window builder (486 s) and the projector check ran on the CPU without the lock.

## 8. Deviations (disclosed; recorded under owner delegation, R28.113)

1. **A first round of checks ran at `e0412e5` and was superseded before any count.** The full suite
   then found `test_no_runner_imports` failing, because `run_task102.py` loaded `run_task101.py` to
   reuse TASK-101's 7.2.1. The check was restated in `so101_hybrid_runtime.check_live_latent` (same
   logic). Every check was rerun at `c586eb1`. The superseded entries, all passing, stay in the
   ledger. Only entries at the running revision admit episodes. No count had been read at that
   point.
2. **7.3.5 ran at H8-mean with τ_h = 0.04 rad (plumbing values).** §7.3 requires the checks before
   P's first episode, while τ_h comes from P and the configuration from A. H8-mean is TASK-101's
   precedent for its 7.2.4, and 0.04 rad is §5.2's fallback. 7.3.6 was read from the same episodes
   and written as its own ledger entry, and its projection is an upper bound (every episode at its
   full budget).
3. **7.3.4 compares `act` against `predict` on stored val latents.** Live encoding is 7.3.1's
   subject. The live path still rebuilds the 35 non-latent inputs from the joints, velocities,
   targets and clock, as the controller does.
4. **The window rule's boundaries, as implemented:** the window takes commands s … L − 1 (the last
   contiguous `descend` run's first command s up to the attempt's `lift_end_step` L, which is the
   first command after the lift), plus the `hold` commands from L on, at most 40. Grasped means the
   corpus's per-attempt fact is `True`.
5. **S+G's approach is follower v1's own drive toward q_pre + δ.** It is not a replay of stored
   commands, because re-executing a rate-limited stored command is not bit-exact. At r = 0 a guard
   requires equality with the demonstration's first 40 commands, and it held on all 32 of P's
   resets. W+F's follower (not run here) aims from the reset's simulator truth, as the envelope's
   "pre" path did. G-zero for it holds on debug seeds (tests).
6. **D's goal demonstrations were drawn and rendered, and 7.3.2 ran on them,** before A's row was
   known (the stage order prepared all goal stores first). No arm ran on D. If a later ruling reuses
   D, this record says that its demonstrations and frames exist and that R-enc read them. No
   closed-loop count on D exists.
7. **Debug plumbing:** two episode jobs on debug seeds 103900–103903 ran in the development tree.
   The first used a random-init stand-in G_loc. The second used the trained G-0, with H8-last and
   τ_h = 0.08 or 0.12 named directly. Debug seeds are never counted. Their lock time is in §7.
8. **The summariser was edited after the run** (`stage0` now tallies GPU hours when D is absent).
   The rules it applies are `so101_hybrid.py` at `c586eb1`, unchanged.

## 9. What this does not show

- Simulation only, `so101_cube_lift_v1`, development cohorts, one run. Not Apple→Plate. R7 is
  unchanged.
- **S+G's 30/32 is not a hybrid result.** Its approach is privileged: the follower's own IK target
  from simulator truth, replayed for 40 commands. It shows that G_loc, cloned from privileged
  scripted play and reading no truth itself, grasps and lifts after an exact approach on 32
  development resets.
- **A says nothing about whether J's approach could work under another cost, goal or planner.** It
  shows that TASK-101's four declared configurations, aimed at subgoal 1 with τ_h = 0.08 rad and a
  0.15 rad jaw tolerance, never reached the hand-over by tolerance within 60 commands on 32 fresh
  resets each.
- No test of LeWM. No world-model claim in either direction.

## 10. Evidence

`~/develop/emai/evidence/task102-stage0/` with `SHA256SUMS`: the windows report and window tables,
the three training reports, `gloc.json`, the four goal reports, every check report, the ledger,
P's and A's reports and progress logs, `tau.json` (`38ecccae…`), `config.json` (`bcc7f620…`),
`summary.json` (`51898127…`), the stage logs and the GPU ledger lines. The per-episode logs and
checkpoints are under `outputs/task102-stage0/` and `checkpoints/task102-stage0/` (HDD). Manifest:
`benchmarks/manifests/so101-hybrid-lift-v1.json`, **CANDIDATE**.
