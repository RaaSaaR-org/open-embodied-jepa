# Apple→Plate LeWM planner v2: results (TASK-074)

**Outcome: INCONCLUSIVE ("close without the clause").** The train stage (Stage 3) ended
**ESCALATE-BUDGET-LAST-TWO** on its run-3 under Addendum A2. Under a ruling decided by Claude under
owner delegation on 2026-10-01 (§3), there is no further budget raise and no further train run,
and TASK-074 closes INCONCLUSIVE. P-far, ranking (O3/O4), the offline decision, D3 and the gated
S/U stages never ran. **No LeWM controller ran in closed loop.**

- **The §7 abandonment clause does not fire.** INCONCLUSIVE is "close without the clause" in §7's
  consequence list. The clause's scope, "a LeWM frozen-DINOv2-token planner choosing the place
  target of e9's place primitive after P-3's pick, on `apple-to-plate-v2` at 112 px", is **not
  refuted and not closed**. It was not tested.
- **Learned Apple→Plate on the frozen benchmark is still 0 successes** (the v1 MVP benchmark,
  0/150 per model). Nothing here changes that number. No control line is primary.
- **The binding finding is on the observation side, not the predictor.** O2's encoded-readout bar
  (≤ 1.0 cm) sits below the measured error of the readout it applies to: R_off on *encoded*
  pooled DINOv2 tokens at t + 16 has a median error of **2.872 cm** on O2's moving windows (90
  windows, 22 val roots), against 4.351 cm for persistence and 3.123 cm for the clock prior, a
  prior that reads no image. That statistic does not depend on W or on the budget, so
  OFFLINE-PASS was unreachable whatever W did (protocol A1.5, A2.7). The bar was carried from
  TASK-073 and never calibrated (§5).
- **A descriptive, non-gating diagnostic** (§4) computed O1 and O2 with the frozen functions on
  run-3's selected checkpoints. Descriptive, not a gate, N-7412 unsaturated, and no row is read
  from it: every O1 bar is met on all three W seeds, and W's predicted offset readout on O2's
  windows has a median of 2.47–2.62 cm, below the encoded readout's 2.872 cm but far above O2's
  1.0 and 1.5 cm bars; the action-blind N's ratio against persistence overlaps W's.
- **The next task is a measurement-first observation-ceiling study, to be preregistered
  separately.**

Protocol: [`apple_lewm_planner_v2.md`](apple_lewm_planner_v2.md), preregistered in #113 (merged as
`9d9b03c`), with Addendum A1 (#114, merged as `f52c905`) and Addendum A2 (#115, merged as
`5e53ef3`). Manifest: `benchmarks/manifests/apple-lewm-planner-v2.json` (frozen sha
`2cf80f5aa54d509e3801bcb3934409407da9b6a0600e6856f319ddeda1d2e36a`, unchanged through both
addenda). The pre-run GOs and the operators' stage records are comments on PR #113.

---

## 1. Stage-by-stage record

Every stage ran once per attempt, from a fresh detached worktree of the merged revision with a
clean tracked tree, on an independent pre-run reviewer's reported GO posted on PR #113. Reports
are git-ignored and stay on the Linux PC under `/home/huhn/develop/emai/worktrees/<run worktree>/`.
Times are UTC from each `report.json`.

| stage / attempt | revision | run worktree | report (sha256) | started → ended (s) | outcome |
|---|---|---|---|---|---|
| K1 | `9d9b03c` | `task074-run` | `outputs/task074-k1/run-1/report.json` (`25701293c54fd585274d7734018de7485d65f92915060ed7aa034439db8aad88`) | 2026-09-30T21:31:11Z → 21:33:35Z (144) | **K1-PASS** |
| corpus | `9d9b03c` | `task074-run` | `outputs/task074-corpus/run-1/report.json` (`16e6417eee7596e9553cf65f02afc4c3784224406782f685067cb83ddda702ac`) | 2026-09-30T21:41:56Z → 21:45:48Z (232) | **CORPUS-SEALED** |
| train run-1 | `9d9b03c` | `task074-run` | `outputs/task074-train/run-1/report.json` (`78137de0c28211833be458f956f7e1f03c2b5c58f75998fb8d42fa2f45ce80df`) | 2026-09-30T22:00:02Z → 22:09:00Z (538) | **V** (G-memory, after the boundary) |
| train run-2 (A1's repeat) | `f52c905` | `task074-run2` | `outputs/task074-train/run-2/report.json` (`f5974cd290f739e7cc20944e68fe8a9dfe485f46cf400d7c89960fa608f20638`) | 2026-09-30T23:58:15Z → 2026-10-01T01:30:22Z (5 527) | **ESCALATE-BUDGET** |
| train run-3 (A2's continuation) | `5e53ef3` | `task074-run3` | `outputs/task074-train/run-3/report.json` (`25cb1bf3bbcbe89d300fb3a702da0f0c56b20bafc24dff896a48ee5d184b562c`) | 2026-10-01T03:05:50Z → 10:30:26Z (26 676) | **ESCALATE-BUDGET-LAST-TWO** |

GO comments on PR #113: K1
[5920077599](https://github.com/RaaSaaR-org/open-embodied-jepa/pull/113#issuecomment-5920077599),
corpus
[5920219476](https://github.com/RaaSaaR-org/open-embodied-jepa/pull/113#issuecomment-5920219476),
train run-1
[5920453351](https://github.com/RaaSaaR-org/open-embodied-jepa/pull/113#issuecomment-5920453351),
train run-2
[5921870076](https://github.com/RaaSaaR-org/open-embodied-jepa/pull/113#issuecomment-5921870076),
train run-3
[5923894699](https://github.com/RaaSaaR-org/open-embodied-jepa/pull/113#issuecomment-5923894699).
Operator records: run-2
[5922899985](https://github.com/RaaSaaR-org/open-embodied-jepa/pull/113#issuecomment-5922899985),
run-3
[5929560762](https://github.com/RaaSaaR-org/open-embodied-jepa/pull/113#issuecomment-5929560762).

### 1.1 K1 (simulator only, no world model; cohort K1, 54000–54031)

The first |d|, 9 cm at step 300, passed all four bars, so 12 cm was not run (§5 stage 1).

| arm (32 resets, one attempt each) | count | bar |
|---|---|---|
| B-oracle-shift | 32 | ≥ 30 |
| H-handover | 32 | ≥ 30 |
| P-stale | 0 | ≤ 4 |
| P-truth | 19 | — |
| H-handover − P-truth | +13 | ≥ +8 |

- No move was blocked in any arm. G-repro reproduced every TASK-072 run-1 check; no render or
  re-render disagreement was recorded.
- O5 (the stand-in's fidelity for the primitive's chunks, on the 9 cm H-handover attempts):
  median relative error 0.209 over 192 decisions, bar ≤ 0.25, passes.
- Memory: peak tree PSS 8.45 GiB, 8 processes. Load at start 0.32 / 1.10.
- B-oracle-shift, H-handover and P-truth are privileged (they read the true moved plate); they
  are ceilings and harnesses, not learned results.

### 1.2 Corpus `apple-far-shift-v2` (privileged scripted collector e9; corpus seeds 54200–54499)

- 300 roots (240 train, 30 val, 30 test), corpus manifest sha256
  `fe7ab9150216887f5759521dcc7fe7c7f54d2e5e8c36fecf6ed713a000b134bd`.
- 196 moves applied, 12 blocked; 281 complete; 156 counted successes (at rest). These are e9
  collector outcomes, including the 150 deliberately mis-aimed roots: **scripted-collector
  outcomes, not learned results**, and not a benchmark.
- Memory: peak tree PSS 7.60 GiB, 8 processes. Load at start 0.21 / 1.13.

### 1.3 Train run-1: V on G-memory, after the boundary

Recorded in full in protocol A1.1–A1.2. The runner's log, verbatim:
`[00:09:00] VOID: StageInterrupted: G-memory: process-tree PSS 12.04 GiB > 12.00 GiB (SIGUSR1)`.
`first_outcome_utc` (22:04:01Z) had been written, so the V came after the train stage's
boundary, and nothing in it is read as an outcome.

- **Cause** (measured, A1.2): R-plate's decision frames were kept as NumPy views `frames[t]`,
  each keeping its episode's whole decoded frame array (about 27 MiB) alive: 270 far episodes,
  about 7.1 GiB, held for the whole stage. The pre-freeze probe had cycled 14 decoded smoke
  episodes, so it did not scale that term.
- **Fix** (A1.3, #114): `featurise_sources` copies what it keeps and frees each episode; the
  train-scale probe calls the same function at the real split sizes, with a 2.0 GiB margin
  check. No computed number changes (976 of 998 smoke leaf values identical, the 22 others are
  timing and path fields; the readouts, baselines and checkpoints byte-identical).
- §7 allowed one repeat from scratch after a reviewed fix: run-2.

### 1.4 Train run-2: ESCALATE-BUDGET (first escalation)

Recorded in full in protocol A2.1. Calibration (seed 7410, 60 000 updates, a point every 1 000):
W-7410 saturated at 25 000 and N-7410 at 40 000. The frozen rule gave `wanted` 80 000 against
the frozen cap of 60 000, so the stage escalated before training any seed model. Peak tree PSS
5.25 GiB (the A1 fix held). `baselines.json` and `r_off.npz` were byte-identical to run-1's.

- **Ruling (A2.2):** the train stage's cap was raised from 60 000 to 80 000, exactly the value
  the frozen formula asked for, as a recorded override outside the frozen block. A last-two
  selection at U = 80 000 would still escalate (ESCALATE-BUDGET-LAST-TWO), and A2 left that row
  "escalate" without ruling on it in advance.

### 1.5 Train run-3: ESCALATE-BUDGET-LAST-TWO (second escalation)

- **A2's calibration check reproduced run-2:** saturation W-7410 25 000, N-7410 40 000
  (`reproduces_run_2: true`); calibration selected updates W-7410 39 000 (val criterion 0.34597)
  and N-7410 54 000 (0.37868). U = 80 000 (`escalate: false`).
- **The six seed models** (80 000 updates, a selection point every 4 000, so the last two points
  are 76 000 and 80 000):

| model | selected update | selected val criterion | in last two points | seconds | checkpoint sha256 |
|---|---|---|---|---|---|
| W-7410 | 32 000 | 0.34708 | no | 3 582 | `678a5f31…cd10e` |
| W-7411 | 48 000 | 0.34766 | no | 3 431 | `cf58c807…81a9` |
| W-7412 | 48 000 | 0.34975 | no | 3 537 | `49c94c4d…7f22` |
| N-7410 | 72 000 | 0.37859 | no | 3 513 | `9c0ee697…6c15` |
| N-7411 | 72 000 | 0.37877 | no | 3 477 | `f33f0beb…cd505` |
| N-7412 | **80 000** | 0.37536 | **yes** | 3 512 | `8e679609…8dbf` |

- N-7412 selected its last point, so `train_last_two_raise(80 000, 0)` returned no raise (U was
  already at A2's cap) and the stage returned **ESCALATE-BUDGET-LAST-TWO** before O1 and O2. The
  report has no O1 or O2 record and no `critic` block.
- The checkpoints (each the model at its selected update) are in
  `task074-run3/checkpoints/task074-train/run-3/`.
- Readouts (train roots only, fitted before the boundary): R_off primal ridge, 5 175 rows, 225
  groups, sha256 `260af3f7…`; R-plate dual ridge, 1 350 rows, val median error 0.417 cm (p90
  0.816 cm), reported only.
- Blind baselines (`baselines.json`, identical to runs 1 and 2): encoded 2.872 cm, persistence
  4.351 cm, clock prior 3.123 cm; 90 moving windows on 22 val roots.
- Memory: peak tree PSS 5.65 GiB (ceiling 12), 1 process, no RSS fallback. Load at start 0.07 /
  1.02. No test root was decoded. The log is one line, with no traceback.

**Reading of the curves (interpretation, not a gate).** W's val criterion selected at 32 000–48 000
and was flat to slightly worse afterwards (0.347–0.356 from 60 000 on). N kept improving slowly;
N-7412's last six points (60 000–80 000) lie within about 0.01 of each other (0.3754–0.3852), so
"selected at 80 000" is a selection among near-equal points, not evidence that N had far to go.
On every seed W's selected val criterion (0.347–0.350) is below N's (0.375–0.379). That is the
training criterion, one statistic, not O1's comparison or its bar.

## 2. Why run-3's row needed a ruling

- §7 lists the budget rows under **escalate**. A2.2 and A2.6 left ESCALATE-BUDGET-LAST-TWO at
  U = 80 000 "escalate" and "not ruled in advance".
- The protocol has no further run defined for this row. A further train run would need another
  budget override (a cap above 80 000) and a new run.

## 3. The ruling (decided by Claude under owner delegation, 2026-10-01)

Recorded verbatim:

> 1. No further budget raise and no further train run. TASK-074 closes **INCONCLUSIVE** ("close
>    without the clause").
> 2. Rationale:
>    - OFFLINE-PASS was already unreachable, because O2's encoded-readout bar (≤ 1.0 cm) sits
>      below the measured readout error of 2.87 cm, whatever W does (A1.5, A2.7).
>    - Another raise would cost about 9 h or more to reach, at best, L-G2A. The rules already make
>      that outcome uninformative about W.
>    - The binding constraint is the observation and readout, not the predictor. The next task is
>      therefore a measurement-first observation study, which is a data/hardware-side change.
> 3. The §7 abandonment clause **does not fire**. Its scope stays open and must be stated as not
>    refuted and not closed.
> 4. A descriptive, non-gating diagnostic is allowed, but only if the protocol text does not
>    forbid reading an escalated stage's models. Check §5, §7 and A1/A2 first. If it is allowed:
>    - compute O1 and O2 descriptively on run-3's existing checkpoints with the frozen code paths
>      (`o2_statistics` and the O1 function the runner uses), at the selected updates;
>    - write the output to a NEW output dir, e.g.
>      task074-run3/outputs/task074-train/run-3-descriptive. Never overwrite run-3;
>    - label every number "descriptive, not a gate, N-7412 unsaturated", and read no row from it;
>    - run it only when the 1-minute load is ≤ 2.0, on the GPU at its normal footprint (≥ 4 GB
>      free; do not touch the GR00T server's ~6.6 GB);
>    - do it in your own worktree at main HEAD, which must include 5e53ef3. Use symlinks to the
>      run-3 inputs as task074-run3 does.
>
>    If the protocol forbids it, skip it and say why.

<a id="note-l-g2a"></a>**Note on "at best, L-G2A"** (factual, added after the §4 diagnostic; the ruling and its
verbatim text are unchanged).
- The phrase names the furthest offline row the frozen rules permit. It is not a prediction of
  the row.
- The descriptive diagnostic (§4) later found that, on run-3's selected models, the frozen
  `o2_passes` returns `void: true` on W seeds 7411 and 7412: N's ratio against persistence has
  upper bounds 0.756 and 0.699, both < 0.8.
- In `decide_offline`'s first-match order (§5 item 7), that void row (L-O2-VOID, escalate) is
  checked before L-G2A.
- So, read as gates, these models would not have reached L-G2A.
- No row is read from the diagnostic (descriptive, not a gate, N-7412 unsaturated), and the
  ruling is unchanged.

The delegation it rests on is the owner's of 2026-09-30, quoted verbatim in protocol §0.

**What the ruling does and does not do.**
- It does not re-threshold, retrain or re-select anything (§7: "Nothing is re-thresholded,
  retrained or re-selected after its numbers are seen").
- It does not read run-3's escalation as an L-row. No offline row (L-NO-DYNAMICS, L-O2-VOID,
  L-G2A, …) was produced, and none is inferred.
- It does not change the bar. O2's 1.0 cm bar stays as frozen, and §5 below records why it was
  wrong.

## 4. The descriptive diagnostic on run-3's checkpoints

**Every number in this section is descriptive, not a gate, N-7412 unsaturated. No row is read
from it.**

**Allowed?** Checked against §5, §7, A1 and A2 before running. The protocol's only "nothing is
read" rule is the void rule's ("A guard, a crash, a cap or a CUDA allocation failure makes the
stage V, and nothing in it is read"); run-3 is an escalation, not a V. §7 forbids
re-thresholding, retraining and re-selection after numbers are seen; the diagnostic does none of
these (it uses the frozen bars only to label pass/fail fields that the frozen functions return,
the models as selected, and no new fit). A1.5's ruling asks the results document for O2's
W-dependent components for every W seed if the clause fires on L-G2A; that did not happen, but
the same components are reported here, labelled descriptive. Nothing in §5, §7, A1 or A2 forbids
reading an escalated stage's models.

**How it ran.**
- From the results worktree at `5e53ef3` (main HEAD, clean tracked tree), `MUJOCO_GL=egl`,
  `PYTHONPATH=src`, the main `.venv`; the corpus and DINOv2 weights by symlink, as `task074-run3`
  does.
- The frozen code paths of `scripts/run_lewm_planner_v2.py`: `featurise_sources` (the same 432
  roots and 76 464 frames as run-3), `train_context`, and `offline_gates`, which calls
  `lewm_planner_v2_offline.o1_statistics` and `o2_statistics`. The six run-3 checkpoints were
  loaded as saved (each the model at its selected update; file sha256s checked against run-3's
  report), with run-3's saved `r_off.npz` (readout sha256 checked).
- **Reproduction checks, all passed:** the six checkpoint file sha256s and R_off's readout sha256
  equal run-3's records; the featurisation gave run-3's 432 roots and 76 464 frames; the blind
  baselines equal run-3's `blind_baselines` exactly; and each loaded model's val criterion,
  recomputed with `val_criterion`, equals run-3's `selected_val_criterion` bit for bit (all six).
  No test root was decoded.
- A first launch stopped before decoding any frame or computing any statistic (it had hashed the
  checkpoints and opened the corpus manifest), because the results worktree lacked the
  `third_party` symlink to the DINOv2 weights. It had created only the empty output directory,
  which was removed. The second launch is the one recorded.
- The script, as it ran, is committed verbatim (not pinned, not part of the runner) as
  [`apple_lewm_planner_v2_results/descriptive_o1o2.py`](apple_lewm_planner_v2_results/descriptive_o1o2.py),
  sha256 `37f02f6e816bfd200e46f93130284f21b14b58561f454bdc2866d3c9b9ff0deb`. It ran with `WT=<results worktree>`,
  `RUN3=/home/huhn/develop/emai/worktrees/task074-run3` and
  `OUT=<RUN3>/outputs/task074-train/run-3-descriptive`.
- Output: a new directory, `task074-run3/outputs/task074-train/run-3-descriptive/descriptive.json`
  (sha256 `7c75d02392a23d17e5d11f6dfc7602289cda260a4aa0ace0242908174629b324`). Run-3's own
  output and checkpoints were not written.
- 2026-10-01T10:33:54Z → 10:43:36Z (581 s). Load at start 0.15 / 1.01 (1-minute ≤ 2.0); GPU at
  its normal footprint, the resident GR00T service untouched.

**O1 (descriptive, not a gate, N-7412 unsaturated).** The frozen `o1_seed_passes` returns
`passes: true` for all three W seeds, on all four cells (overall and shifted windows, h = 8 and
h = 16). The ranges over the three seeds:

| quantity (frozen bar) | overall h = 8 | overall h = 16 | shifted h = 8 | shifted h = 16 |
|---|---|---|---|---|
| effective-rank ratio (≥ 0.16) | 0.365–0.379 | 0.349–0.365 | 0.374–0.387 | 0.359–0.374 |
| std ratio (≥ 0.39) | 0.822–0.838 | 0.822–0.839 | 0.822–0.839 | 0.820–0.841 |
| predicted collapsed fraction (≤ 0.05) | 0.000 | 0.000 | 0.000 | 0.000 |
| W − N comparative-rank lower bound (> 0) | 0.048–0.059 | 0.064–0.080 | 0.053–0.064 | 0.072–0.090 |
| truncation controls k = 1, 2, 4 (must fail) | all fail | all fail | all fail | all fail |
| W / copy-last upper bound (≤ 0.8) | 0.706–0.712 | 0.533–0.540 | 0.725–0.731 | 0.545–0.553 |
| W / N upper bound (< 1.0) | 0.939–0.954 | 0.870–0.889 | 0.945–0.961 | 0.870–0.898 |
| shuffled-action / W lower bound (≥ 1.10) | 1.424–1.448 | 1.736–1.760 | 1.420–1.440 | 1.730–1.756 |
| zero-action / W lower bound (≥ 1.10) | 1.861–2.325 | 2.570–3.512 | 1.889–2.376 | 2.596–3.561 |

Windows: 1 148 overall (28 val roots) and 902 shifted (22 val roots). The W / N upper bounds at h = 8
(0.939–0.961) are the narrowest margins.

**O2 (descriptive, not a gate, N-7412 unsaturated).** 90 moving windows on 22 val roots; cm are
median Euclidean errors of the apple-minus-plate offset at t + 16; ratios are cluster-median
ratios with reset-clustered 95 % intervals.

| W seed | encoded median (≤ 1.0) | predicted median (≤ 1.5) | (i) W / persistence | (iii) W / clock prior | (ii) N / persistence | frozen `o2_passes` |
|---|---|---|---|---|---|---|
| 7410 | 2.872 | 2.542 | 0.584 [0.399, 0.826] | 0.814 [0.575, 1.024] | 0.565 [0.429, 0.831] | passes false, void false |
| 7411 | 2.872 | 2.473 | 0.568 [0.405, 0.800] | 0.792 [0.574, 0.982] | 0.610 [0.519, 0.756] | passes false, void true |
| 7412 | 2.872 | 2.623 | 0.603 [0.480, 0.757] | 0.840 [0.672, 0.965] | 0.572 [0.472, 0.699] | passes false, void true |

What these numbers show, descriptively and without reading a row:
- **Every O1 bar is met on all three seeds:** the collapse bars are met (effective-rank ratio
  0.35–0.39 against the bar of 0.16, calibrated in TASK-066 on `apple-look-v1` pilot data and
  carried unchanged), W beats copy-last and the action-blind N, and W is action-sensitive.
- **On O2's moving windows, W's predicted readout (2.47–2.62 cm) is below the encoded readout
  (2.872 cm) and the clock prior (3.123 cm), but nowhere near 1.0 or 1.5 cm.** The readout, not
  the predictor, sets this floor.
- **The action-blind N reads the offset about as well as W on these windows:** N's ratio against
  persistence (0.565–0.610) overlaps W's (0.568–0.603). On two seeds N's upper bound is < 0.8 (0.756 and 0.699),
  which the frozen function labels void (O2 (ii)). So on O2's windows the readout does not
  separate W from N, even though O1 does in latent space.
- N-7412 selected its last point, so its N numbers (in O1's W / N and comparative-rank columns
  for seed 7412, and O2 (ii) for seed 7412) come from a model that may not have saturated.

## 5. O2's uncalibrated bar, and the lesson

- **The bar.** O2 (§6.2) requires R_off on *encoded* latents at t + 16 to have a median error of
  at most 1.0 cm on the moving windows. That component does not depend on W.
- **Where it came from.** It was carried verbatim from TASK-073
  (`wm_critic_v2.O2["encoded_median_max_cm"]`). Neither protocol records a calibration. TASK-073's
  O2 baselines used the true offset as a stand-in for a perfect readout, so the encoded readout's
  own error was never measured before either freeze. TASK-074's design probes measured a
  different readout (Probe C: a kernel-ridge plate readout on full tokens, 0.55–0.69 cm), not
  R_off on pooled latents (A1.5).
- **The signs were there before the freeze.** The frozen-code train smoke read 3.21 cm (12
  windows, 3 roots) and TASK-073's development smokes 2.31 cm (9 windows, 2 roots), both labelled
  "nothing is read" and both above 1.0 cm. The closest earlier project figure, TASK-054's
  encoded-target readout error, was 2.47–2.59 cm (a different encoder, horizon and quantity).
- **The measured value.** 2.872 cm, byte-identical in runs 1, 2 and 3. The clock prior, which
  reads no image, is 3.123 cm; the encoded readout beats it by about 0.25 cm.
- **Consequence.** From A1.5 on, the furthest offline row the frozen rules permitted was L-G2A,
  which would have fired the §7 clause on a bar the readout could not meet, not because of
  anything W does. On the models run-3 actually trained, the void check that precedes L-G2A
  already holds on two seeds, descriptively; see the [note in §3](#note-l-g2a).
- **Lesson.** A bar on a readout must be calibrated from that readout's measured ceiling on
  representative data *before* the freeze. Carrying a bar from another protocol, or setting it
  against a perfect-readout stand-in, is not calibration. When a smoke shows a W-independent
  component above its bar, that is a design defect to resolve before the freeze, even if the
  smoke is labelled "nothing is read". The next task should measure the observation's readout
  ceiling first and set any readout bar from it.

## 6. What is and is not concluded

**Concluded.**
- TASK-074 is **INCONCLUSIVE**. The train stage escalated twice on its budget rules
  (ESCALATE-BUDGET at the frozen cap, then ESCALATE-BUDGET-LAST-TWO at A2's cap) and did not
  reach O1 or O2 as a gate. No offline row, no D3 and no gated row exists.
- Under the frozen bars, OFFLINE-PASS was unreachable from A1.5 on, because of a W-independent
  readout ceiling (2.872 cm against a 1.0 cm bar).
- On this corpus, from 112 px onboard frames, a linear readout of the apple-minus-plate offset
  from pooled 4 × 4 DINOv2 tokens is only about 0.25 cm better than a clock prior that reads no
  image. That is the binding constraint the next task addresses.

**Not concluded.**
- **Nothing about LeWM as a planner.** No ranking (O3/O4), no D3 and no gated arm ran; no LeWM
  controller ran in closed loop. The clause's scope is **not refuted and not closed**.
- Nothing about W's dynamics as a gate: O1 was not computed by the stage. The descriptive numbers
  in §4 are not O1 and are not read.
- Nothing about whether a longer budget would saturate N-7412, or change any row.
- Nothing about other observations (cameras, resolution, readouts, encoders), which the next task
  measures.
- **Learned Apple→Plate on the frozen benchmark is still 0 successes** (0/150 per model). The K1
  and corpus counts are privileged or scripted-collector outcomes, not learned results.

## 7. Next step

The next task is a measurement-first observation-ceiling study, to be preregistered separately.
It is a data/hardware-side change in the sense of §7 ("the next step is the task owner's choice
of a data or hardware change, not another planner variant"). The LeWM backend, the token latent,
the v2 task and the product goal stay open.

## 8. Process

- #113 (protocol) merged on an independent APPROVE. K1, the corpus and train run-1 each ran once
  on its own reported GO.
- Train run-1 ended V after its boundary; #114 (A1, the reviewed fix) merged; run-2 ran once on
  its GO and escalated; #115 (A2, the budget override) merged; run-3 ran once on its GO and
  escalated again.
- Nothing was re-thresholded, retrained or re-selected after numbers were seen. The descriptive
  diagnostic (§4) used the selected checkpoints as saved and wrote only to a new directory.
