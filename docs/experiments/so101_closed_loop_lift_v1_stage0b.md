# TASK-101 Stage 0′: every check passed, K′ null in all four configurations (S0-NULL), diagnostic Δ-MIXED (record)

**Status: Stage 0′ record. Row: S0-NULL (§7.6 row 3 as amended by A1, §18.6). Δ's row: Δ-MIXED,
escalated with this record. The protocol is not frozen, and there is no gated stage.**
Protocol: [so101_closed_loop_lift_v1.md](so101_closed_loop_lift_v1.md) §7 and §18 (amendment A1).
Diagnostic record: [so101_closed_loop_lift_v1_null_diagnostic.md](so101_closed_loop_lift_v1_null_diagnostic.md).
Rulings R28.76–R28.82 in [DECISIONS.md](../DECISIONS.md), decision 2026-10-19 (bw), **recorded by
Claude under owner delegation**. They record the rows and the deviations. They do not choose the
next study: Δ-MIXED admits no variant by default and escalates.

**Nothing here is a result.** Every closed loop ran on the fresh calibration cohort K′
(101060–101091) or on debug seeds. The planner queries of Δ ran on the diagnostic cohort X
(101700–101715) and executed nothing in closed loop. D′ (101600–101663) was **not drawn, rendered or
run**: under S0-NULL, D′ does not run (§18.6). No gated seed was drawn, rendered or run. This is a
simulation-only SO-101 study on `so101_cube_lift_v1` (the lift scene with its inert target marker
drawn inside the play corpus's marker range), not Apple→Plate. R7 does not change (R28.9).

## Summary

- **Every check of §7.2 passed before any K′ count existed, at one revision (`88a014b`), and the
  runner enforced it.** The check ledger held passing 7.2.0, 7.2.1, 7.2.3, 7.2.4 and 7.2.6 reports
  and a passing 7.2.2 report for K′'s goal store (`e5c6ec56…`) at that revision. The latest of them
  finished at 17:44:54Z, and K′'s episodes job started at 17:45:10Z. The summariser read K′'s counts
  only after confirming that order. Deviation 7 of Stage 0 did not recur.
- **The scene check now holds: 7.2.2 reads 0.912 cm** in median on K′'s 128 subgoal frames, against
  B_cube = 2.397 cm. Stage 0 read 6.106 cm on v0's frames. Per subgoal: 0.83, 1.05, 0.94 and 0.78 cm.
  On the 32 reset frames it reads 1.21 cm. 7.2.0 (the scene comparison) found every declared element
  inside the corpus's support. 7.2.6 found the marker inert: on 16 debug seeds, every named array
  was bit-identical between v0 and v1 after the reset and after every command, and the marker sat
  exactly at its drawn point.
- **K′: 0/32 in every configuration** (H8-last, H8-mean, H16-last and H16-mean; 128 episodes; exact
  95 % interval 0.000–0.109 each). As in Stage 0's K, **no episode reached a single subgoal by the
  switching tolerance**: all 3 × 128 switches were timeouts. On H8-mean the median closest arm RMS to
  subgoal 1 was 0.45 rad, against the 0.04 rad tolerance. The best single episode came within
  0.035 rad (H8-last). By §7.3's tie rule the configuration is H8-mean, and the row is
  **S0-NULL**. So moving the marker into the corpus's range, which fixed the readout, did not move
  the closed loop. That settles R28.67's open question in one direction: K's null was not the
  marker's doing alone.
- **Δ (§18.6), on 16 X seeds at H8-mean, read Δ-MIXED.** m2 = 0.000: at the median state, J's cost
  ranks the demonstration's own next 8 commands ahead of **all** 255 projected random candidates,
  so the cost does point toward the subgoal relative to random commands. m1 = 0.116, below 0.50:
  CEM's best candidate is cheaper under J than the demonstration's commands on 99 of 112 states, so
  CEM does find commands as cheap. The row is neither Δ-MODEL (m2 ≥ 0.25) nor Δ-PLANNER (needs
  m1 ≥ 0.50). Reported only, m3 = 0.509: executing CEM's 8 commands open-loop lowered the arm RMS to
  the subgoal on 57 of 112 states. The diagnostic record has the per-state readings and a
  development reading of what the pattern suggests. That reading is not a row.
- **The demonstration was kept on 32/32 drawn K′ seeds and 16/16 drawn X seeds** (and 4/4 debug
  seeds). No seed was rejected.
- **GPU: 1.21 GPU h in Stage 0′ (9 lock-holding jobs), 2.87 GPU h cumulative** against the Stage 0
  cap of 6 GPU h. Δ took 66 s of lock time against its 1.5 GPU h allowance. K′'s measured
  throughput would project Stage S at 5.86 GPU h at H8-mean (6.94 at H = 16) for all six CEM arms,
  inside the cap's remainder. That projection is recorded only: no row reads it, because there is
  no gated stage under S0-NULL.

## 1. Code (commit `88a014b`; the run tree `worktrees/task101-stage0b-run` was checked out there)

| file | what changed |
|---|---|
| `src/embodied_jepa/so101_tasks_lift_v1.py` (**new**) | `so101_cube_lift_v1` exactly as §18.2: v0's cube draws, the marker from `default_rng([10107, seed])` on `so101_play_corpus.TARGET_REGION`, at least `MIN_CUBE_TARGET_M` from the cube (imported, not copied), `SO101CubeLiftTaskV1` overriding only `name` and `sample_reset`; NumPy only; no registry entry |
| `src/embodied_jepa/so101_closed_loop.py` | cohort X, spent cohorts K and D (refused for goals and episodes), salts 10107 and 10108, `LIFT_TASK_NAME`, the check ledger and its verdict (§18.5 (2)), the ordering reading (§18.5 (3)), the D′ projection (§18.7), Δ's states, seeds, ranks, measures and rows (§18.6) |
| `src/embodied_jepa/so101_closed_loop_runtime.py` | `lift_task` (every demonstration, episode, F and check runs v1; v0 only inside 7.2.6 and Stage 0's `marker` diagnostic); Δ's planner queries and open-loop branches |
| `scripts/run_task101.py` | `goals` writes only the goal store (no `--evidence`); `episodes` refuses K′ and D′ unless the ledger admits them, before any controller, simulator or seed is built, and refuses goal stores not made on v1; every `check` appends a ledger entry; new checks `scene` (7.2.0) and `inert` (7.2.6); `renc` per cohort with the reset-frame reading; `projector` on K′; `delta` |
| `scripts/summarize_task101.py` | `--stage0b`: the ledger table, K′ with its decomposition and closest-RMS reading, the D′ projection, Δ, D′ (unused here), the row; it refuses any cohort whose episodes started before its latest required check |
| `scripts/run_task101_stage.sh` | Stage 0′ jobs on `outputs/task101-stage0b`; CPU arms' evidence episodes in their own locked episodes job; CPU-only checks outside the lock |
| tests | `test_so101_tasks_lift_v1.py` (8 tests), `test_so101_closed_loop_ledger.py` (13, 16 cases) and one Δ plumbing test in `test_so101_closed_loop.py`; see §5 for the change to `test_so101_wm.py` |

`so101_tasks.py`, `so101_simulation.py` (both pinned by `so101-play-v1`), `config.py`,
`schema_planning.py`, `so101.py`, `so101_followers.py`, every TASK-100 pinned file and every G1
byte-pinned file keep their bytes (tests check them). Full suite on the branch: 2 670 passed and 36
skipped (graphics opt-in and similar), after the deviation-1 change below.

## 2. Checks (§7.2 as amended; every one at `88a014b` from a clean tracked tree)

| check | finished (UTC) | outcome |
|---|---|---|
| 7.2.0 scene comparison | 17:36:33 | **pass.** Marker box and distance identical to the corpus's constants (the same objects); cube box x 0.18–0.23, \|y\| ≤ 0.10 inside the corpus's x 0.16–0.25, \|y\| ≤ 0.12; yaw ±45° on both; arm `home_rad` inside the corpus's ±0.06 rad jitter box; jaw 0.6 rad inside 0.2–1.2; `so101_simulation.py`, `so101_sim_action.json`, `assets/so101_manifest.json` and the MJCF `so101.xml` equal to their pins; cameras and 224 px equal. All 364 K′, D′, S and X seeds (main and reserve) drawn: cube equal to v0's bit for bit, inside the corpus's cube box, marker inside the box and at least 8 cm away. Motion: not compared by construction |
| 7.2.6 the marker is inert | 17:36:38 | **pass.** 16 debug seeds (101900–101915), v0 against v1, rendering off, 123 states each: `qpos`, `qvel`, the targets, the executed commands and every named `task_truth()` array bit-identical; the marker exactly at `(target_xy, 0)` at every step (v1's drawn point, v0's (0.30, −0.20)); every demonstration lifted under both |
| 7.2.1 live latent = stored | 17:37:05 | **pass.** 64/64 frames bit-identical (0 ulps) |
| 7.2.4 determinism (conditional) | 17:44:03 | **pass** on the conditional form; literal form false. W and N reruns bit-identical; pool 1 against pool 6 on four debug episodes at H8-mean: two bit-identical, on 101901 the latent first differed at step 82 (3 steps) with identical actions, on 101903 at step 76 (50 steps) with the actions first differing at step 161, the same outcome on all four |
| 7.2.3 projector (K′'s reset states) | 17:44:41 | **pass.** 1 000/1 000 equal to chained `preview` and to `execute`, 0 stopped |
| 7.2.2 R-enc on K′'s goal store `e5c6ec56…` | 17:44:54 | **pass.** 0.912 cm median on 128 frames (B_cube 2.397 cm); per subgoal 0.83 / 1.05 / 0.94 / 0.78 cm; reset frames 1.21 cm (32) |
| ties (reported, as in Stage 0) | 17:44:59 | pass: N_J-100101's costs bit-identical across 256 candidates at H8/H16 × last/mean, the plan is the initial mean |
| 7.2.5 timing (from K′) | — | planning s per command, median / 95th percentile: H8-last 0.234 / 0.290, H8-mean 0.233 / 0.291, H16-last 0.394 / 0.468, H16-mean 0.393 / 0.467; longest whole command 1.47 s against the 30 s deadline; every configuration eligible |

7.2.2 for D′ did not run, because D′ did not run.

## 3. K′ (§7.3 on K′; goal store `e5c6ec56…`, 32 kept of 32 drawn)

W (J-100101), pool of 6 workers, one fresh process per episode, 3 839 s of wall time (1.07 h of lock
time) for 128 episodes. Episodes started 17:45:10Z, after the last required check (17:44:54Z).

| configuration | W | exact 95 % | subgoal switches (1→2, 2→3, 3→4) | any gripper contact | both jaws | max rise median | rise ≥ 5 cm | closest arm RMS to subgoal 1, median (min) | commands clipped |
|---|---|---|---|---|---|---|---|---|---|
| H8-last | 0/32 | 0.000–0.109 | 32/32/32 timeouts | 15 | 2 | 0.16 cm | 0 | 0.35 (0.035) rad | 4 412 |
| H8-mean | 0/32 | 0.000–0.109 | 32/32/32 timeouts | 6 | 2 | −0.01 cm | 0 | 0.45 (0.21) rad | 4 838 |
| H16-last | 0/32 | 0.000–0.109 | 32/32/32 timeouts | 23 | 1 | 0.42 cm | 0 | 0.26 (0.083) rad | 4 139 |
| H16-mean | 0/32 | 0.000–0.109 | 32/32/32 timeouts | 12 | 2 | −0.01 cm | 1 | 0.48 (0.14) rad | 4 742 |

No episode stopped; one cube was lost (H8-last). Every executed action equalled the projected
action the model scored. **By §7.3 every configuration is tied at 0, so the choice is H8-mean (mean
over last, then H = 8), and every configuration scored 0/32: S0-NULL.** On the two `last`
configurations the closest RMS came nearer (0.26–0.35 rad in median) than on the `mean` ones
(0.45–0.48 rad). This is a development reading on 32 resets, and nothing selects on it.

Compared with Stage 0's K under the void condition (0/128; H8-mean's median closest RMS to subgoal 1
0.41 rad), K′ is the same pattern. The marker was not what kept W from the first subgoal.

## 4. Δ (§18.6; record [so101_closed_loop_lift_v1_null_diagnostic.md](so101_closed_loop_lift_v1_null_diagnostic.md))

Goal store X `82a67c57…`, 16 kept of 16 drawn. One GPU job, 66 s of lock time, at H8-mean.
**m1 = 0.116, m2 = 0.000, m3 = 0.509 (reported); row Δ-MIXED.** Escalated with the record. No
variant is admitted by default.

## 5. Deviations (disclosed; recorded under owner delegation, R28.80)

1. **TASK-100's test `test_g1_byte_pinned_files_match_origin_main_when_available` was narrowed.**
   It required every change under `src/` against `origin/main` to be an addition. A1 changes
   TASK-101's own Stage 0 modules (`so101_closed_loop.py` and its runtime), which no G1 or TASK-100
   manifest pins and whose own manifest is CANDIDATE. The test now exempts exactly the files in
   TASK-101's manifest while that manifest is not FROZEN. Every other path must still be an
   addition. The test file is not pinned by any manifest. §7.1 said "new files only", and A1 added
   the new module; editing the runner and the closed-loop modules was not a new file. They are this
   task's own unfrozen code, and the alternative was duplicating them.
2. **7.2.4 ran at H8-mean** (Stage 0 used H16-mean, the check's former default). Neither is
   declared. H8-mean is the tie-rule configuration and halves the lock time.
3. **7.2.2's reset-frame reading** re-renders each goal demonstration's reset state from its
   declared reset rather than storing the frame in the goal store. The goal-store format is
   unchanged. It is reported only.
4. **The CPU-only checks (7.2.0, 7.2.6, 7.2.3) ran outside the GPU lock**, as §13 allows for CPU
   work; they render nothing.
5. **Δ's random candidates use `default_rng([10108, seed, t])`**, and the normalised rank counts
   ties as half. §18.6 names the salt 10108 and "rank (0 = best, chance 0.5)" without the stream's
   other keys or a tie convention. No exact tie occurred in m2's reading (m2 = 0 with the
   demonstration strictly cheapest at the median state).
6. **The ledger also requires a clean tracked tree** for each required report and for the episodes
   job. That is stricter than §18.5 (2).
7. **Δ's first launch was refused by `gpu_run.sh`** (GPU utilisation 56 % at 18:50Z, transient;
   nothing started and no output was written). It was relaunched 20 s later.

## 6. What this leaves open (the escalation)

The row is S0-NULL with Δ-MIXED. §18.6 says: "escalate with the record; no variant is admitted by
default". This record does not choose the next study. What a ruling has in front of it:

- The checks now pass, so the pipeline reads the scene as TASK-100's gates did. The null is a
  controller result on the in-distribution scene, not a pipeline defect, at this configuration,
  this sample budget and this goal form.
- Δ says J's cost prefers the demonstration's commands over random ones (m2 = 0), and that CEM
  finds commands J scores as even cheaper (m1 = 0.116). Neither declared one-sided reading applies.
  The diagnostic record gives the per-state numbers, including where CEM's cheaper commands still
  move the arm toward the subgoal open-loop and where they do not.
- Budget: 2.87 of Stage 0's 6 GPU h used; 17.1 of the 20 GPU h total remain.
- H-pose, which reads only the subgoal joints, lifted on 8/8 D slots in Stage 0 (the v0 scene; the
  marker plays no part in H-pose). That is the measured non-trivial reference for any redesign.

## 7. Evidence

Outputs: `outputs/task101-stage0b/` (shared checkout, HDD). Evidence
`~/develop/emai/evidence/task101-stage0b/` (796 KB: every report, the ledger, logs, progress files,
both summaries, the GPU-lock log lines and the SHA-256 of all 143 episode and goal `.npz` files),
**`SHA256SUMS` `e29b2cd30ac858f920e1870c4f9e9d26f93ac7607fed7badb2308c14d13fbe71`**. Summary
`stage0b_summary.json` `69974ac4…5cec9`; Δ `delta.json` `5fd62bed…15b0`. Goal stores: debug
`f628823c…`, K′ `e5c6ec56…`, X `82a67c57…`. TASK-100's checkpoints were read only. Nothing under
`data/`, `checkpoints/` or an earlier `outputs/` path was written; Stage 0's `outputs/task101-stage0/`
is untouched.
