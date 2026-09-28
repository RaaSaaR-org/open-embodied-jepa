# Apple→Plate first learned policy v2, M2 on cohort C: results (TASK-072, M2 step)

**Row: M2-FAIL. The only failing gate is G3.** P-3 reached **40/40** counted successes on
cohort C, and the random-init encoder floor R-3 reached **39/40**. P − R-3 is +1, where G3
needs at least +8. Every other gate passed, including G2 (P-3 40 against the no-image control
C-3 at 12) and G1 (40 against open-loop replay B-replay at 28).

The authorization record fixed before the run ([`apple_first_policy_v2_m2.md`](apple_first_policy_v2_m2.md)
§3.2) says what to do in exactly this case: report the row and G3's declared reading side by
side, and do not choose between them. **The owner has since ruled (§7):** M2-FAIL stays the
recorded row, and G3's declared reading is adopted as the interpretation.

| the frozen row (first match, TASK-071 §12) | G3's declared reading (TASK-071 §12, v1 §11) |
|---|---|
| **M2-FAIL**: "any other failing gate; the claim is not made; the owner decides" | "a learned visuomotor policy works; encoder pretraining contributes nothing measurable" |

**Read this first.**
- **This is the held-out cohort.** Cohort C (45300–45339, the stored values of
  `apple-policy-v1.json`) was simulated for the first time, once, by this run.
- **The official learned Apple→Plate count on the frozen benchmark is still 0.** That benchmark
  is the v1 task's MVP benchmark (0/150 per model). M2 is on `apple-to-plate-v2` and does not
  change it.
- **The policy is not LeWM.** It is a behaviour-cloning/DAgger MLP on a frozen DINOv2 readout
  (TASK-072 run-1's P-3). No world model is in the loop.
- **Pretrained vision shows no measurable contribution.** R-3 reads random-init DINOv2 tokens
  through the same pipeline. It failed only one reset (45307): there the robot's
  measured-joint-velocity guard refused a command before any grasp. That is a counted failure
  of the attempt, not a void of the run (§3). This matches D2, where R-3 tied P-3 in both
  earlier runs.
- **The image is used.** C-3 has the same pipeline but a constant estimate instead of the image.
  It reached 12/40, and P-3 beat it on 28 resets with none the other way (exact McNemar
  p = 7.5e-9).
- **The scale is small.** This is one run with one training seed per arm on 40 resets, one
  camera (112 px onboard) and one encoder.

Authorization record: [`apple_first_policy_v2_m2.md`](apple_first_policy_v2_m2.md), merged in
PR #102 as `23e2593`. M2 itself was preregistered by TASK-071
([`apple_first_policy_v2.md`](apple_first_policy_v2.md) §12). The results manifest is
`benchmarks/manifests/apple-first-policy-v2-m2-results.json`. It was generated from
`report.json` by `scripts/summarize_first_policy_v2_m2.py`. That script refuses to write if the
report's hash, the stored cohort values, the per-arm records, the recomputed `decide_m2` or the
evidence hashes disagree. This document does not amend the authorization record.

---

## 1. Provenance

| item | value |
|---|---|
| revision | `23e25937b1f195d7cf5acf6693fe42334021ac69` (main after #102), from a fresh clean worktree `worktrees/task072-m2-run` with its own `.venv`. `revision_at_end` is the same, and `tracked_tree_dirty: false` |
| command | `MUJOCO_GL=egl nohup uv run --no-sync python scripts/run_first_policy_v2_m2.py run --output outputs/task072-m2-cohort-c/run-1 --evidence /home/huhn/develop/emai/worktrees/task072-run` |
| pre-run GO | The independent pre-run reviewer **reported GO**, posted on PR #102 as comment 5879303699. The orchestrator was told, and told the owner, before the run started. The reviewer's render check at 16 workers was **IDENTICAL** (report `4a72bb2e…ef2c`), and its preflight at `23e2593` was PREFLIGHT-READY with G-repro 8/8 (report `9953997c…9c8f`). An earlier REQUEST CHANGES at `714b9d0` was fixed in `3350f3d` before the GO |
| before starting | `nvidia-smi` and `ps` showed only the long-running GR00T server (about 6.6 GB). No Isaac container process was running. The run itself allocated no GPU memory |
| machine | the Linux PC: Python 3.12.3, torch 2.14.0+cu130, MuJoCo 3.13.0, `MUJOCO_GL=egl`, 16 simulation workers. Rollouts and DINOv2 run on the CPU. The main process was in run-1's strict deterministic state at the start and the end |
| started / ended | 2026-09-28T21:44:49Z to 21:47:39Z (UTC), **169 s** of the 7 200 s cap. `cohort_first_render_utc` is 21:45:30Z |
| stage times | preflight 5 s, re-render 2 s, readout refit and G-repro 25 s, cohort frames and estimates 3.5 s, the seven arms 125 s |
| report | `outputs/task072-m2-cohort-c/run-1/report.json` in that worktree, sha256 `aa274cddbdb31a3641bc7bc55abc7a43e3ceb0162d60b573b7c68d8d2e301baa`. It is git-ignored, on the Linux PC. The log is `worktrees/task072-m2-run-1.log` (one line: report written, outcome M2-FAIL) |
| guards | All run guards passed (a V on any of them would have voided the run): G-frozen; G-hash (43 pins, equal at the preflight and at the end); clean tree; G-platform; G-device (strict); G-weights; G-evidence (run-1's report, the P-3/C-3/R-3 checkpoints and the corpus manifest, all recomputed from the files again by the summarizer); G-cohort (digest `4f888154…533e`); G-seeds; G-look; G-frame; G-cap. `non_finite_fields` is empty. **`test_split_decoded` is false** (170 train episodes decoded). These are distinct from the embodiment's per-command guard (the "guard" terminations in §3), which ends one attempt as a failure and never voids the run |
| G-repro | **8/8 exact.** Both readout selections, the 426 fit rows, the 128 S0-P error pairs, C-3's constant, the 112-root B-replay library and the 16 D2 nearest roots all reproduce run-1 |
| cohort | seeds 45300–45339 in order. The executed resets equal the stored `apple-policy-v1.json` values exactly (checked by the summarizer) |
| void | none; one run |

## 2. The gates (as `decide_m2` produced them; recomputed by the summarizer)

A **counted success** is `apple_at_rest_v0` after the 60-step settle, **and** the latched `grasp`
and `place` stages, both reached during the attempt's commands (T71-R1/R2). Counts are of 40.

| gate | condition | measured | result |
|---|---|---|---|
| G1 | P-3 ≥ 17/40 and strictly more than B-replay | P-3 40, B-replay 28 | **pass** |
| G2 | P-3 − C-3 ≥ +8 | 40 − 12 = **+28**; discordant 28 vs 0, n_d 28, exact McNemar p = 7.45e-9 | **pass** |
| G3 | P-3 − R-3 ≥ +8 | 40 − 39 = **+1**; discordant 1 vs 0, n_d 1, exact McNemar p = 1.0 | **fail** |
| G4 | P-3 ≥ 20/40 grasps | 40 | **pass** |
| G5 | B-hold and B-random 0 grasps; B-oracle ≥ 38/40 | 0, 0; B-oracle 40 | **pass** |
| G6 | zero controller privileged reads by any L1 arm | P-3, C-3, R-3: none | **pass** |
| G7 | median control time ≤ 100 ms per command, including the one DINOv2 forward pass | **0.42 ms** median over 29 600 commands (p90 0.45 ms, max 41 ms). The forward pass plus readout alone had a median of 31 ms (max 40 ms). The conservative variant, with the forward pass added to every command, had a median of 32 ms | **pass** |

**Row: M2-FAIL** (first match). G5 and G6 pass, so it is not M2-VOID. G3 fails, so it is not
M2-PASS. G2 passes, so it is not M2-FAIL-VISION.

## 3. Every arm on cohort C (40 resets, each arm once)

The 95 % intervals are Wilson intervals for k of 40. The final distance is the apple-to-plate
distance at the end of the attempt, over all 40 attempts.

| arm | rung | counted success | 95 % CI | latched grasp | at rest, not counted | latched v1 success | terminations | final distance, cm (q10 / q50 / q90) | failed seeds |
|---|---|---|---|---|---|---|---|---|---|
| **P-3 (carried)** | L1 | **40/40** | 0.91–1.00 | 40 | 0 | 40 | step limit 40 | 2.28 / 2.82 / 3.27 | none |
| R-3 (random-init floor) | L1 | **39/40** | 0.87–1.00 | 39 | 0 | 39 | step limit 39, guard 1 | 2.32 / 2.91 / 3.37 | 45307 |
| C-3 (no-image control) | L1 | **12/40** | 0.18–0.45 | 21 | 0 | 19 | guard 18, step limit 22 | 3.13 / 4.61 / 18.91 | 28 seeds |
| B-replay (nearest corpus root, open loop) | L4 | 28/40 | 0.55–0.82 | 32 | 0 | 32 | policy complete 32, guard 8 | 2.74 / 3.47 / 17.40 | 12 seeds |
| B-oracle (e9 from truth) | L4 | 40/40 | 0.91–1.00 | 40 | 0 | 40 | policy complete 40 | 3.01 / 3.46 / 3.83 | none |
| B-hold | L4 | 0/40 | 0.00–0.09 | 0 | 0 | 0 | step limit 40 | 15.25 / 17.68 / 19.80 | all |
| B-random | L4 | 0/40 | 0.00–0.09 | 0 | 0 | 0 | step limit 40 | 15.25 / 17.68 / 19.80 | all |

- **Timing of the learned successes.** They latched grasp at steps 264–272 and place at steps
  609–638, all before the settle. P-3 was 265–269 and 611–635. No at-rest attempt went uncounted
  in any arm.
- **R-3's one failure (45307).** The robot's measured-joint-velocity guard
  (`GUARD_REFUSALS = ("measured joint velocity limit exceeded",)`,
  `first_policy_runtime.py:30`) refused the command at step 223 of 740. Steps are 0-based, so
  223 commands had been executed. This happened before any grasp; the only latched stage was
  `reach`. `run_attempt` counts this as a failed attempt, not a void. The apple ended 20.3 cm
  from the plate. The "guard" terminations of C-3 and B-replay in the table are the same
  per-command guard.
- **B-replay's 12 failures** were 45301, 45302, 45306, 45308, 45315, 45320, 45321, 45323, 45324,
  45326, 45336 and 45338. Its 40 resets were served by 31 distinct library roots.
- **The readout on cohort C (descriptive, never gating).** The P readout's error against the
  reset truth had a median of 0.21 cm for the apple (p90 0.52, max 0.81) and 0.05 cm for the
  plate (p90 0.11, max 0.21). These are within the S0-P bars fixed in TASK-072 run-1.
- **Paired, on the identical resets:**
  - P-3 vs C-3: 28 vs 0 (p = 7.45e-9).
  - P-3 vs R-3: 1 vs 0 (p = 1.0).
  - P-3 vs B-replay: 12 vs 0 (p = 0.00049).

## 4. Reading (interpretation, not measurement)

1. **The frozen row is M2-FAIL, and the claim "a learned policy with a DINOv2 encoder works on
   Apple→Plate on this cohort" is not made by the rule.** The only failing gate is G3.
   Preregistration attached a declared reading to exactly that failure: "a learned visuomotor
   policy works; encoder pretraining contributes nothing measurable". This document reports
   both and does not choose between them (authorization record §3.2). The owner has since ruled
   (§7): the row stays, and G3's declared reading is adopted as the interpretation.
2. **What the other gates show, as measured.**
   - The carried learned policy succeeded on every held-out reset.
   - It beat open-loop replay of the nearest expert demonstration (G1, 40 vs 28; 12 vs 0 paired).
   - It beat the no-image control by 28 (G2), so the per-reset image estimates are used.
   - It grasped on all 40 (G4) and ran well inside the latency bound (G7).
   - The harness was valid (G5), and no learned controller read simulator truth (G6).
3. **What G3 shows.** Replacing the pretrained DINOv2 with its random-init counterpart, with the
   readout and policy retrained on its features, changed one reset out of 40. On these 112 px
   frames the readout gets the apple and plate positions from random-init tokens as well. TASK-063
   and TASK-064 had already found that random-init features read the apple on their own. So this
   pipeline gives no evidence that the pretrained weights matter. It does not show that they
   never matter: a harder cohort, a different camera or a different task could separate the
   two arms.
4. **Why the success rate is so high.** The policy acts on two xy estimates read once from the
   post-look frame, plus proprioception and a clock. It imitates a scripted expert (e9) that
   itself reaches 40/40 here. The task is narrow: one apple, one plate, a ±3 cm / ±2 cm reset
   distribution, and a fixed look.
5. **The development cohort predicted this.** On D2, P-3 and R-3 were 16/16 on identical sets in
   both TASK-071 and TASK-072. The authorization record's status lines stated beforehand that
   G3 would likely fail if cohort C agreed.

## 5. Caveats kept in the record

- **The frozen benchmark is unchanged.** Learned Apple→Plate on the frozen v1 MVP benchmark is
  still 0 successes (0/150 per model). This result is on `apple-to-plate-v2` only.
- **Not LeWM.** The pipeline is DINOv2 features, then a linear readout, then a BC/DAgger MLP. No
  world model is in the loop, and nothing here shows that LeWM or any world model can drive the
  robot.
- **The row is M2-FAIL.** The claim of §12 is not made by the rule; the owner rules on G3's
  declared reading.
- **Pretraining shows no measurable contribution** (R-3 39/40).
- **Small scale.** One run, one training seed per arm, one encoder and one floor seed, one camera
  (112 px onboard), 40 resets, and a narrow reset distribution.
- **Privileged elements.** The arms that read truth or replay demonstrations (B-oracle,
  B-replay) are not learned results. The DAgger labeller was privileged at training time only.
  The learned arms made zero privileged reads at run time.
- **Cohort C is no longer held out.** It has now been simulated. `apple_policy_v1.md`
  (line 731) says cohort C is "never reused"; any reuse needs a new, disclosed protocol
  and the owner's ruling.

## 6. Process

- The pre-run reviewer's **reported** GO was delivered as a message and posted on PR #102. The
  orchestrator told the owner before the run started.
- The run was made exactly once, from a fresh clean checkout of the merged revision `23e2593`.
  It was not void.
- Nothing was re-thresholded, retrained or re-selected after the numbers were seen.

## 7. Owner ruling (2026-09-28)

After the row and G3's declared reading were reported side by side, the orchestrator recommended
how to read them. The owner answered, on 2026-09-28, verbatim: **"yes, do your recommendations"**. That makes the
recommendation the owner's ruling:

- **M2-FAIL stays the recorded row.** The frozen rule is not rewritten.
- **G3's declared reading is adopted as the interpretation.** On held-out `apple-to-plate-v2`
  resets, a learned visuomotor policy (DINOv2 + BC/DAgger, trained on e9 demonstrations) works,
  and encoder pretraining contributes nothing measurable.
- **This does not change the v1 benchmark** (still 0/150), and **it is not a LeWM result**.

The ruling interprets the recorded result. It does not change any count, gate or row above, and
the caveats of §5 still apply to the adopted reading.

**Learned Apple→Plate on the frozen benchmark is still 0 successes.** On cohort C of
`apple-to-plate-v2`, the carried learned policy P-3 reached 40/40 and M2's row is **M2-FAIL** on
G3 alone. The random-init floor reached 39/40. G3's declared reading is reported beside the row;
the owner's ruling (§7) keeps the row and adopts that reading as the interpretation.
