# Apple→Plate first learned policy v2, M2: the gated test of P-3 on cohort C (TASK-072, M2 step)

**Status: authorization record and run plan; no cohort-C seed has been simulated.** Only the
preflight readiness check and the smoke run of §9 have run. Neither simulates a cohort-C seed,
and nothing in them is read.

**Status lines.**
- **Learned Apple→Plate on the frozen benchmark is still 0 successes.** The frozen MVP benchmark
  (v1 task, 0/150 per model) is not what M2 tests, and M2 does not change it.
- **M2 was preregistered by TASK-071** ([`apple_first_policy_v2.md`](apple_first_policy_v2.md)
  §12, which carries [`apple_first_policy_v1.md`](apple_first_policy_v1.md) §11). Its question,
  cohort, arms, gates G1–G7, rows and success definition are that section's, **unchanged**. This
  document adds only what §12 left to "a separate authorization" (§2), and it names every choice
  that §12 does not settle (§3).
- **The owner gave the go-ahead on 2026-09-28** (via the orchestrator) for M2 on cohort C,
  through the normal pre-run review.
- **The policy is P-3 from TASK-072 run-1** on the Linux PC
  ([`apple_first_policy_v2_linux_results.md`](apple_first_policy_v2_linux_results.md)): a
  behaviour-cloning/DAgger MLP on a frozen DINOv2 readout. **It is not LeWM**, and no world model
  is in the loop.
- **Expected, from D2 (not a prediction that gates anything):** on D2 the random-init floor R-3
  tied P-3 at 16/16 on the same resets, in both TASK-071 and TASK-072. If cohort C agrees, G3
  (P − R-3 ≥ +8) fails, and the row is M2-FAIL whatever P-3 scores (§3.2).

Manifest: `benchmarks/manifests/apple-first-policy-v2-m2.json`. Design:
`src/embodied_jepa/first_policy_v2_m2.py`, which imports M2 from `first_policy_v2` unchanged.
Runner: `scripts/run_first_policy_v2_m2.py`, which reuses TASK-072's runner
(`scripts/run_first_policy_v2_linux.py`, pinned and unmodified) for the worker, the checkpoint
loader and the report helpers. Tests: `tests/test_first_policy_v2_m2.py`. **No file of
TASK-071 or TASK-072 is modified**, and all 37 TASK-072 pins still hold (a test checks it).

---

## 0. The go-ahead

The orchestrator relayed, on 2026-09-28: the owner gave explicit go-ahead for M2 on cohort C to
proceed through normal pre-run review, for the carried policy P-3, using the P-3 checkpoint of
TASK-072 gated run-1 (`checkpoints/task072-first-policy-v2-linux/run-1/`). TASK-072's own
protocol ([`apple_first_policy_v2_linux.md`](apple_first_policy_v2_linux.md) §0.6, §9) kept M2
out of its scope; this document is the separate authorization record that TASK-071 §12 requires.

## 1. M2, as preregistered (TASK-071 §12; v1 §11), unchanged

**Question.** Does the carried learned policy leave the apple at rest on the plate, after a
carried placement, on the frozen held-out cohort C, clearly more often than the no-image
control, the random-init floor and open-loop replay of the expert?

- **Cohort C:** the 40 resets 45300–45339, from the stored values in
  `benchmarks/manifests/apple-policy-v1.json` (`cohorts.C_frozen_gating.resets`, sealed by
  `cohort_sha256` `4f888154…533e` over the stored decimals), **never recomputed**.
- **Success:** the counted success of T71-R1/R2: `apple_at_rest_v0` after the task's 60-step
  settle, **and** the latched scorer's `grasp` and `place` stages both reached during the
  attempt's commands, before the settle. At-rest attempts without both are reported per arm and
  count toward nothing. "Grasps" are the latched scorer's grasp stage.
- **Arms**, each run on each of the 40 resets exactly once: the carried P-k (**P-3**), C-3 (the
  no-image control), R-3 (the random-init encoder floor), B-replay (the nearest counted-success
  train root of the corpus, replayed open loop), B-oracle (e9 from the reset truth), B-hold and
  B-random.
- **Stop rule:** the carried P-k does not run on C if it had 0/16 grasps on D2 (M2 then fails);
  the controls always run. P-3 had 16/16 grasps on D2 in TASK-072 run-1, so it runs.

| gate | condition |
|---|---|
| G1 | the carried P-k ≥ 17/40 counted successes **and** strictly more than B-replay on the identical resets |
| G2 | P − C-3 ≥ +8 counted successes (exact McNemar, realised n_d and p reported) |
| G3 | P − R-3 ≥ +8. Declared reading if it fails: "a learned visuomotor policy works; encoder pretraining contributes nothing measurable" |
| G4 | the carried P-k ≥ 20/40 grasps |
| G5 | B-hold and B-random 0/40 grasps; B-oracle (e9) ≥ 38/40 counted successes. A failure invalidates the run, not the arms |
| G6 | zero controller privileged reads by any L1 arm |
| G7 | median control time ≤ 100 ms per command, including the one DINOv2 forward pass |

**Rows (first match):** **M2-VOID** (G5 or G6 fails), **M2-PASS** (every gate passes),
**M2-FAIL-VISION** (G1 and G4 pass, G2 fails: no evidence the image is used), **M2-FAIL**
(anything else; the claim is not made and the owner decides). A gate that cannot be evaluated
counts as failed. `exemption_spent` (`apple-policy-diagnostics-v1.json`) is `false` and is not
claimed.

**What a pass would mean:** "a learned policy with a DINOv2 encoder works on Apple→Plate on this
cohort under apple-to-plate-v2". **It would not mean** that LeWM (or any world model) drives the
robot, anything about the v1 task or its 0/150 benchmark, another camera, or another cohort.
M2 has no abandonment clause of its own; TASK-071 §11's clause fires only on M1 rows.

## 2. What this authorization fixes

| item | fixed as | why |
|---|---|---|
| the policy | TASK-072 run-1's checkpoints, by sha256: P-3 `7988162d…60be8`, C-3 `d1820db4…1289`, R-3 `bea0d422…fd3` (full values in the design module and the manifest), under `checkpoints/task072-first-policy-v2-linux/run-1/` | the owner's go-ahead names the TASK-072 P-3; C-3 and R-3 are its controls from the same run |
| run-1's evidence | its `report.json` (sha256 `87745284…ec78`, revision `db65816`, outcome M1-PASS, carried P-3) and its corpus `data/apple-look-v2-linux/run-1` (manifest sha256 `67c342f6…4f54`), read from the checkout that holds them (the runner's `--evidence`) and checked by hash (G-evidence) | the runner reads them; it writes nothing there |
| the readouts | refitted exactly as run-1 fitted them, because run-1 did not store them; they must reproduce run-1's recorded facts exactly (**G-repro**, below), or the run is V before any cohort-C seed is simulated | the policy was trained and evaluated on these readouts' estimates; a different readout would be a different policy input |
| C-3's input | run-1's constant estimate (the mean of the cross-fitted P estimates over the 170 train roots), recomputed and required to equal run-1's `C_mean_estimates` exactly | as in M1 |
| B-replay | run-1's library: the 112 train roots of `apple-look-v2-linux` with a counted success; nearest by standardised DINOv2 CLS of the post-look frame, as in M1; each replays its executed policy commands (no settle), then the task's settle | TASK-071 §12 ("nearest counted-success `apple-look-v2` train root"); the Linux corpus is the one P-3 was trained on |
| the cohort | the stored values, passed to the robot's reset unchanged; G-cohort checks the digest and that the seeds are exactly 45300–45339; G-seeds is a whitelist of exactly those 40 ints, in order, and refuses non-int seeds before any coercion | v1 §11 and `task056_handover.md` §7: the stored-values debt is discharged by a behavioural test (§5) |
| G7 | per command of P-3's 40 attempts: the wall time of the controller's `act()` in its worker (1 torch thread), with the attempt's one DINOv2 forward pass plus the readout (main process, batch size 1, 6 threads) added to its first command; G7 compares the median with 0.100 s | §12 says "including the one DINOv2 forward pass"; §3.4 |
| platform | the Linux PC as in TASK-072: G-platform (Linux, x86-64, `MUJOCO_GL=egl`), 16 simulation workers, rollouts and features on the CPU; the main process is put in run-1's strict deterministic state (`devices.require("cuda", strict=True)`); no CUDA work is done | run-1's process state for the refit |
| caps | global 7 200 s; each rollout batch 3 600 s; the re-render 1 800 s; each attempt 300 s | TASK-071's per-stage caps; a smaller global cap for a stage-only run |
| output | `outputs/task072-m2-cohort-c/run-<k>/report.json`; the runner refuses to overwrite it and writes nothing else | never overwrite evidence |

**G-repro, in full** (every item exact; any mismatch is V, and nothing on cohort C is
simulated):
1. The P and R readouts are refitted on the same 426 rows: frame 0 of run-1's 170 sealed train
   roots plus the 256 perception-train post-look frames (51200–51455), re-rendered here.
2. Each readout's selection (family, λ_rel, inner MSE) equals run-1's report.
3. The P readout's per-reset apple and plate errors on the 128 re-rendered held-out frames
   (51456–51583) equal run-1's S0-P errors (float equality after the JSON round trip).
4. C-3's constant estimate equals run-1's `C_mean_estimates`.
5. B-replay's retrieval, re-run on the 16 re-rendered D2 post-look frames (52000–52015), picks
   run-1's `b_replay_nearest` roots from the same 112-root library.

Re-rendering these spent seeds is a reset and the 8-command look only: no controller runs and
nothing about them is scored. The corpus is read through its train split only (val and test are
not decoded).

## 3. Choices §12 does not settle (for the pre-run reviewer and the owner)

1. **Which P-3.** TASK-071 §12 carried "the P-k with the most D2 successes" of TASK-071's run;
   that run was on the Mac (MPS). The owner's go-ahead names TASK-072's Linux P-3, which is the
   platform's own replicated run; TASK-071's Mac checkpoints are not used here.
2. **G3 and the rows.** If every gate passes except G3, the first matching row is **M2-FAIL**,
   and G3's declared reading ("a learned visuomotor policy works; encoder pretraining
   contributes nothing measurable") is reported beside it. The results document will state both
   and not choose between them; the owner decides. D2 makes this case likely (status lines).
3. **G2 and G3 are difference gates.** The exact McNemar p-values (and n_d) are reported, as
   §12 says; they do not gate.
4. **G7's measurement** is §2's. A conservative variant (the forward pass added to every
   command), p90 and max are reported beside it, and do not gate.
5. **G6 is a gate here, not a guard.** In the M1 runner a privileged read raised (V). Here, as
   §12 says, a G6 failure is the row M2-VOID.
6. **Arms not in §12** (A4-look, D-oracle-perc, F, P-0…P-2) do not run on cohort C.
7. **The void rule** (§12 has none of its own): any guard, crash or cap is **V** and nothing is
   read. A V before the first cohort-C frame is rendered may be repeated after a reviewed fix. A
   V after that point is reported to the owner, and a repeat (`run-2`, cohort C simulated again)
   needs the owner's ruling; a second V closes M2 as INCONCLUSIVE. M2-VOID is an outcome, not a
   V, and is not repeated without the owner's ruling. Nothing is re-thresholded, retrained or
   re-selected.

## 4. Stages

1. **Preflight:** G-frozen (the manifest's frozen block equals the module), G-hash (every pin,
   the tracked tree clean), G-platform, G-device (strict), MuJoCo 3.13.0, G-weights (pretrained
   `3a697b87…af27`, Linux floor `546b9011…29d2`).
2. **G-evidence** and the stop rule; the three checkpoints load.
3. **Run-1's corpus, train split only** (170 roots, hashes verified; frame 0 is the post-look
   frame).
4. **Re-render** the 256 + 128 perception frames and the 16 D2 frames (frame only).
5. **Readouts and G-repro.** The `preflight` mode stops here.
6. **Cohort C frames:** the 40 post-look frames from the stored resets; the P and R estimates
   (DINOv2 tokens, batch size 1); C's constant; B-replay's nearest roots; the forward-pass
   timings.
7. **The arms**, each on each reset once, every attempt's re-rendered post-look frame checked
   against the frame its estimates came from (G-frame), every attempt's post-look state against
   the first's (G-look).
8. **`decide_m2`**, then G-hash and G-device again at the end.

## 5. Guards (any failure is V)

TASK-071's guards as they apply to a stage without training (G-hash, G-frozen, G-scene, G-weights,
G-look, G-frame, G-device, G-finite, G-cap, Q-split), TASK-072's G-platform, and four for this
step: **G-evidence** (run-1's report, checkpoints and corpus manifest by sha256; the report is
the recorded M1-PASS run, clean, not a smoke, test split not decoded), **G-repro** (§2),
**G-cohort** (the stored values, their digest and exactly 40 seeds) and **G-seeds** (the
whitelists: exactly cohort C for the cohort stage; exactly run-1's seeds for each re-rendered
role, never cohort C). G-privileged is G6 (§3.5).

The tests check, among other things, that the cohort stage never calls `wide_reset`, that the
value reaching the robot's `reset` is the stored one (through the runner's `run_task`), that the
string seed "45300" is refused, that the timing wrapper keeps privileged reads visible to the
counter, and each row of `decide_m2` in both directions.

## 6. Process

1. **This PR** holds this document, the manifest, the design module, the runner, its tests and
   the card. It merges on an independent reviewer's reported APPROVE (posted on the PR) and
   green CI.
2. **The gated run** starts from a clean checkout of the merged revision on the Linux PC, only
   after a fresh pre-run reviewer's **reported** GO, and after the orchestrator has been told:

   ```sh
   uv run --no-sync python scripts/run_first_policy_v2_m2.py run \
       --output outputs/task072-m2-cohort-c/run-1 \
       --evidence /home/huhn/develop/emai/worktrees/task072-run
   ```

   It runs under `nohup`, so it survives an ssh disconnect. The GPU is shared with another
   service; this run does no CUDA work.
3. **The results PR** states the row, every gate with its numbers and every arm's count, with
   a results manifest generated from `report.json` by a committed script, and every restated
   number checked by an independent reviewer.

## 7. Budget

Measured by the preflight and the smoke (§9). TASK-072 run-1's whole pipeline took 1 235 s on
this PC; M2 is its perception refit plus 280 attempts (7 arms × 40) of 800 steps on 16 workers.

## 8. Not done (declared)

- No training, retraining or reselection; no new seed range; no world model in the loop.
- No arm beyond §12's; nothing on the v1 task or the test split.
- No claim beyond cohort C under `apple-to-plate-v2`, one camera (112 px onboard), one run.

## 9. Preflight readiness check and smoke (nothing in them is read)

*To be recorded before the pre-run review.*

## 10. Amendment log

*Empty.*

**Learned Apple→Plate on the frozen benchmark is still 0 successes.**
