# Apple→Plate WM critic v2: results (TASK-073)

**Row: S-NO-CONDITION.** The K0 condition calibration found no plate-shift condition that passes
all four preregistered bars. TASK-073 ends at K0: the corpus, training, ranking, D3 and the gated
S/U stages do not run, and **the LeWM critic was never trained or run**.

- **The protocol's wording for this row** ([`apple_wm_critic_v2.md`](apple_wm_critic_v2.md) §5,
  stage 1): "BC plus perception leaves no measurable room for a world-model critic under this
  disturbance on v2".
- **The abandonment clause does not fire.** It fires only on WMC-G2A, WMC-NO-RANK, HYB-HARM and
  HYB-NO-GAIN (§7.2).
- **The owner's D7 fallback is authorised.** S-NO-CONDITION is one of the fallback rows (§7.3,
  `FALLBACK_ROWS`). The fallback is TASK-074, the LeWM-only planner (`lewm-planner`), which needs
  its own preregistration. This document does not design it.
- **Learned Apple→Plate on the frozen benchmark is still 0 successes** (the v1 MVP benchmark,
  0/150 per model). Nothing here changes it. No control line is primary.

Protocol: [`apple_wm_critic_v2.md`](apple_wm_critic_v2.md). It was preregistered in #106 (merged as
`b4df3f0`) and amended after K0 run-1's void by #108 (merged as `35772e5`, protocol §14–§15).
Manifest: `benchmarks/manifests/apple-wm-critic-v2.json` (frozen hash `c7a3eb21…0023`).

---

## 1. What K0 asks, and the bars

K0 checks whether the diagnostic plate-shift condition (owner D1: simulation only, on
`apple-to-plate-v2`) leaves room for a critic. It uses only the simulator, no world model, on the
32 development resets of cohort K0 (53000–53031).

The runner works through the steps in order, shift step 300 first and then the one retry at 480,
and |d| from 3 to 6 cm. In each cell the arms run in the order below, and the cell stops at the
first bar that fails:

1. **B-oracle-shift ≥ 28/32.** B-oracle-shift is e9 built from the post-shift truth: the harness
   and the oracle.
2. **P-stale ≤ 8/32.** P-stale is P-3 with its post-look estimates only. The bar requires that the
   shift actually breaks the unaided policy.
3. **P-truth ≥ 20/32.** P-truth is P-3 handed the true post-shift plate at the shift step: the
   perception ceiling.
4. **H-sim − P-truth ≥ +4/32.** H-sim picks among the 25 aims around the true plate by
   privileged 16-command look-ahead in cloned simulator state: the selection ceiling.

The chosen cell would have been the smallest |d| passing all four. None did.

## 2. Results (K0 run-2)

Counted successes of 32 per arm, one attempt per reset. "—" means the arm did not run because
the cell had already failed an earlier bar.

| shift step / \|d\| | B-oracle-shift (≥ 28) | P-stale (≤ 8) | P-truth (≥ 20) | H-sim | H-sim − P-truth (≥ +4) | failed bar |
|---|---|---|---|---|---|---|
| 300 / 3 cm | 32 | 4 | 27 | 29 | +2 | 4 |
| 300 / 4 cm | 32 | 4 | 28 | 31 | +3 | 4 |
| 300 / 5 cm | 32 | 1 | 29 | 28 | −1 | 4 |
| 300 / 6 cm | 31 | 0 | 26 | 20 | −6 | 4 |
| 480 / 3 cm | 32 | 4 | 29 | 31 | +2 | 4 |
| 480 / 4 cm | 32 | 5 | 29 | 29 | 0 | 4 |
| 480 / 5 cm | 32 | 18 | — | — | — | 2 |
| 480 / 6 cm | 31 | 20 | — | — | — | 2 |

`k0_select` returns S-NO-CONDITION. O5, the stand-in chunk check, is computed only on a passing
cell, so it was not computed.

## 3. Reading

These are interpretation, not measurement. Each count is one run of 32 resets.

1. **The condition works as a disturbance.** A shift at step 300 breaks the unaided policy
   (P-stale 0–4/32), while e9 rebuilt for the moved plate still succeeds (31–32/32). At step 480
   the smaller shifts also break it (P-stale 4–5/32). At 5–6 cm, P-stale reaches 18–20/32, so
   that condition is too weak for bar 2.
2. **Perception alone recovers almost everything.** Given the true post-shift plate, P-3 reaches
   26–29/32 in every cell where it ran, although it never saw its estimates change mid-episode in
   training.
3. **Privileged aim selection adds at most +3/32 over P-truth**, and at 5–6 cm (step 300) it does
   worse than P-truth. In this setting, choosing among nearby aims by 16-command look-ahead toward
   e9's median offset does not improve on simply aiming at the true plate, by the preregistered
   margin.
4. **So the question the critic was meant to answer does not arise here.** Even a perfect
   16-command outcome predictor, choosing among these aims, gains less than the +4/32 bar over the
   perception ceiling. A LeWM critic using the same cost J (the distance to e9's median offset
   o*), the same 25 candidates and the same 16-command horizon could at best match H-sim. It
   could not match an arbitrary selector. H-sim optimises that proxy, not success, and it can do
   worse than simply aiming at the true plate: 20 against P-truth's 26 at 300 / 6 cm.

## 4. Caveats (read these before citing the row)

- **This is a headroom finding, not a test of the LeWM critic.** H-sim is privileged look-ahead in
  cloned simulator state, not LeWM. The world model, its readouts and its offline gates were never
  trained or run. The row says nothing about how well LeWM predicts or ranks.
- **One disturbance family.** The plate is teleported sideways, mostly along ±y, per the
  preregistered direction rule. There is one task (v2), one camera (112 px onboard), one policy
  (P-3) and one target definition (e9's median offset, o*). A different disturbance, target or
  selection horizon could leave room that this one does not.
- **32 resets per cell, one run.** The differences at bar 4 (+2 and +3) are within a few resets.
  No interval is claimed, and a larger cohort could land on either side of the bar.
- **The 480 / 5 cm and 480 / 6 cm cells stopped at bar 2** (P-stale 18/32 and 20/32). Their
  P-truth and H-sim were not run, so nothing is known about headroom there.
- **H-sim's counts carry renderer noise when H-sim reads images.** That applies in D3 and S only;
  in K0, H-sim is centred on the true plate and reads no image. The owner's ruling on this is in
  §6.
- **One K0 reset caps the P-3 arms at 31/32.** On reset 53018, every P-3 attempt (P-stale,
  P-truth and H-sim, in every cell) ended at step 231 with the embodiment's joint-velocity guard
  refusing a command (`guard_refusal`). That is before the shift at step 300 or 480, so it is
  independent of the condition. B-oracle-shift succeeded there in every cell.
- **The privileged elements are not learned results.** These are B-oracle-shift, P-truth, H-sim,
  e9 and P-3's privileged training labels. P-3 is a behaviour-cloning/DAgger policy trained on e9's
  demonstrations. It is not LeWM.

## 5. Provenance

### 5.1 K0 run-1: V (not read)

K0 run-1 ran on 2026-09-29 from a clean worktree at `b4df3f0`, after the pre-run GO.

- **What happened:** after its first cohort render, an external host agent sent SIGTERM to its
  whole process group, because of memory pressure. Its 16 simulation workers had taken the
  launcher's cgroup to a peak of 25.9 GiB. No report was written.
- **Status:** recorded as V (protocol §15.1); nothing from it is read.
- **Owner rulings:** "Fix PR, then repeat (Recommended)" and "I'll tell Hank to leave gated runs
  alone".
- **The fix, #108:**
  - 6 simulation workers, and a 12 GiB process-tree ceiling on summed PSS, with G-memory at the
    start and a runtime guard;
  - V reports on SIGTERM, SIGINT and SIGHUP, with a bounded pool close and signal-safe cleanup;
  - `cohort_first_render_utc` for the corpus stage, and a train boundary;
  - the renderer guards (§7 below).
- **The owner's rulings on #108's review**, verbatim in protocol §15.5:
  - "Accept, record every retry (Recommended)";
  - "Yes, count it as a pass (Recommended)";
  - "Match decisions and results (Recommended)".

### 5.2 K0 run-2: the recorded run

| item | value |
|---|---|
| GO | the pre-run reviewer's reported GO, [#108 comment 5901205401](https://github.com/RaaSaaR-org/open-embodied-jepa/pull/108#issuecomment-5901205401): render check MAJORITY-PASS (one odd render, 53954, 1 pixel, 1 level, same state), preflight PREFLIGHT-READY. The owner was told before the start |
| revision | `35772e5e8564ed2c116cf90d1202854aa7c006c9` (main after #108), a fresh worktree `worktrees/task073-k0-2`; `revision_at_end` the same; `tracked_tree_dirty: false` |
| command | `nohup env MUJOCO_GL=egl UV_PROJECT_ENVIRONMENT=…/.venv uv run --no-sync python scripts/run_wm_critic_v2.py k0 --output outputs/task073-k0/run-2 --evidence /home/huhn/develop/emai/worktrees/task072-run` |
| report | `outputs/task073-k0/run-2/report.json` in that worktree (git-ignored, on the Linux PC), sha256 `f760af401cd92b974f7c99f74f7a0bce3d6deaf00ca9000173e92e3f860a27f2` |
| time | started 03:13:46 local (`uptime`); first cohort render 2026-09-30T01:14:28Z; 2 015 s |
| machine | load 0.23 / 0.08 / 0.02 at the start (G-quiet: ≤ 2.0; `load_average_at_start` in the report); 4.60 / 5.23 / 4.78 at the end (the run's own), **from the operator's `uptime` at 03:47:34, not in `report.json`**; MemAvailable 25.4 GiB at the start (G-memory: ≥ 16 GiB, in the report); no other heavy job; the GPU's resident service untouched |
| guards | every guard passed (the run ended without a V; this list of guard names is the protocol's §8, not a field of `report.json`): G-frozen, G-hash (61 pins), G-platform, G-threads, G-quiet, G-memory, G-device (strict), G-weights, G-evidence, G-cohort (stored values), G-seeds, G-look, G-frame, G-shift, G-cap; G-repro 8/8 |
| memory | peak of the whole process tree: **PSS 8.42 GiB** (the guarded measure; ceiling 12), RSS 10.08 GiB; main process 2.00 GiB, largest worker 1.21 GiB; 6 workers |
| render events | none: no disagreement in G-repro's re-render or the cohort frames, no G-frame retry in any attempt, no signal during cleanup |
| void | none |

## 6. The owner's ruling on H-sim (moot here)

Before D3, the owner ruled on H-sim's renderer sensitivity. In D3 and S, H-sim's candidates are
centred on R-mid's reading of the frame, and a smoke showed one counted success flip between
runs. The ruling, verbatim: **"Accept as recorded (Recommended)"**. It meant three things:
- each count is one run of the frozen procedure;
- the reports list H-sim's odd renders;
- the results say its counts are reproducible only up to renderer noise, with no code change.

It is moot for this result: D3 and S never ran, and in K0 H-sim reads no image.

## 7. The renderer finding

The EGL renderer on the Linux PC sometimes returns a different frame for an identical simulator
state. The difference is one intensity level in 1–7 pixels, on the order of 1 frame in 1 000 in
this task's smokes; the reviewer's probe measured 0.02–0.03 %. It caused intermittent G-repro and
G-frame voids in smokes. Since #108:
- votes and re-renders are guarded: the full simulation state must be bitwise equal, and the
  difference must be at most 1 level in at most 16 pixels;
- every event is recorded;
- a sealed corpus is not bitwise reproducible across collections.

K0 run-2 had no event. Image-reading closed-loop arms can diverge between runs because of it. That
is why the re-run rule for those arms ("Match decisions and results") and the H-sim ruling exist.

## 8. Not fixed (needed only if this code is reused for a gated run)

The independent review of #108 listed non-blocking follow-ups that are **not fixed**, because
TASK-073 ends at K0:
- `rerun_matches` treats a field missing from both records as a match, and does not compute
  `max_abs_command_difference`;
- the protocol cites 0.45 cm as the largest incumbent difference, but the reviewer measured
  **0.87 cm** (H-rand, 53960, verify-3) against the 1.0 cm tolerance, a thin margin that the
  protocol does not record;
- `Pool.close` is not fully bounded: `pool.terminate()` itself joins without a timeout, so a worker
  stuck in uninterruptible sleep could still block it;
- the memory watch has no RSS fallback when PSS cannot be read.

- `first_outcome_utc`, the train stage's boundary, is described in protocol §15.5 but is not in
  the frozen `VOID_RULE` text;
- the train stage's memory on the real corpus has never been measured. The full-size figure
  (10.75 GiB PSS) comes from a synthetic probe.

**Known inconsistencies, left in place because the files are pinned.** Editing them would change
their pinned hashes after the fact:
- protocol §14 and §15.3 still give the frozen hash as `cd9e8723…` (the value after the first
  #108 commit); the manifest's `frozen_sha256`, which the runner checks, is `c7a3eb21…0023`;
- the `k0_select` docstring in `wm_critic_v2.py` says S-NO-CONDITION "escalates to the owner
  under D7", while `ROW_CONSEQUENCES` and protocol §7.0/§7.3 give "fallback". The code path that
  decided this run is the frozen `ROW_CONSEQUENCES` and `FALLBACK_ROWS`.

If the runner or the re-run rule is reused for a gated run, for example by TASK-074, these need
fixing and review first. The gated harness (`stage_gated`) was never written.

**Addendum (2026-09-30, TASK-074 follow-up fix PR).**
- **Where the fixes are.** Three of the follow-ups above are fixed in a new module,
  `src/embodied_jepa/run_guards.py`, with tests in `tests/test_run_guards.py`:
  - `rerun_matches` now fails a field that is missing from both records, requires the commands,
    and computes `max_abs_command_difference`;
  - `BoundedPool.close` is bounded;
  - `pss_bytes` falls back to RSS.

  TASK-074 and later runners import that module.
- **TASK-073's pinned code is unchanged.** `wm_critic_v2.py`, `scripts/run_wm_critic_v2.py` and
  every other file in `apple-wm-critic-v2.json`'s `hashes` keep the bytes K0 run-2 ran at
  `35772e5`, and a test checks the two central ones. The fixed code is therefore not what K0
  run-2 ran, and TASK-073's record, frozen hash and G-hash stay reproducible as recorded.
- **Three follow-ups are not in that module; each moves to TASK-074's preregistration:**
  - `first_outcome_utc` in `VOID_RULE` belongs to each protocol's frozen text, so TASK-074
    writes it into its own `VOID_RULE` from the start;
  - the train-stage memory on a real corpus is measured in TASK-074's preregistration;
  - the thin re-run margin. The reviewer measured 0.87 cm for H-rand (53960, verify-3) against
    the 1.0 cm incumbent tolerance, which the protocol does not record. TASK-074 records its own
    re-run tolerances with the largest difference measured on its smoke seeds and that margin.
    `RerunRule` now also requires a command tolerance for every image-reading arm.

## 9. Process

- #106 (protocol) merged on an independent APPROVE. K0 run-1 started on a reported GO, became a V,
  and went to the owner.
- #108 (the fix) merged on an independent review, after REQUEST CHANGES and the owner's rulings.
- K0 run-2 ran exactly once, from a fresh worktree of the merged revision, on the reported GO,
  with the owner told before the start. No other stage started.
- Nothing was re-thresholded, re-selected or rerun after the numbers were seen.

**Learned Apple→Plate on the frozen benchmark is still 0 successes.** TASK-073 ends S-NO-CONDITION
at K0, a headroom finding under privileged look-ahead. The LeWM critic never ran, and the D7
fallback (TASK-074) is authorised.
