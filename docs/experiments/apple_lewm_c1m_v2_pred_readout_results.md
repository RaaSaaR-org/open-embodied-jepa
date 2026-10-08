# Apple→Plate LeWM committed aim under C1-M, read by a readout fitted on W's predicted latents: results (TASK-080)

**Outcome: L-NEAR at Stage S.** "G-NI fails, W not detectably inferior: escalate, no clause, no
claim" (protocol §9.5). **The abandonment clause (§12) does not fire**, nothing in its scope is
closed, and **no claim is made**: neither the primary claim, "LeWM-driven closed-loop success", nor
the secondary, "LeWM needed". TASK-080 closes here, at protocol §8 step 8, with outcome L-NEAR
(DECISIONS R18.30–R18.32, decided by Claude under owner delegation).

On the gated cohort S (64 fresh resets, 65200–65263; one run, one model seed, in simulation only,
under the declared simulation-only plate condition C1-M), P-3's learned pick and e9's scripted
place are the same in every arm; the arms differ only in the single place aim committed at step
405. W, the LeWM 8 × 8 token predictor on frozen DINOv2 features read by R-S (a ridge fitted on
W's own predicted latents under the kinematic stand-in's commands), chose that aim:

- **What passed.** G-bar: **W 58/64** ≥ 56/64. The four twin tests: W beat its action-blind twin
  N (26/64), its scene-blind twins L-shuf (21/64) and L-mean (30/64) and a random grid aim L-rand
  (11/64), each by an exact one-sided McNemar test at p < 0.01 (p from 8.9 × 10⁻¹⁴ to
  9.7 × 10⁻⁷). W's controller read no task truth at run time (0 reads on every attempt).
- **What failed.** G-NI: against the better of the two hand-written non-world-model arms, H-rule
  (63/64; H-sysid 62/64), the paired difference W − H-rule is **−5/64** with a reset-bootstrap
  95 % interval of **[−10, 0]/64**; its lower bound is below −δ = −8/64 (δ an allocation, not
  calibrated), so non-inferiority is not shown. Its upper bound (0) is not below −8/64, so W is not
  detectably inferior beyond δ, and the row is L-NEAR, not L-INFERIOR.
- **The determinism re-run passed.** The log's "W determinism: 3/4" is the success count of W's
  re-run on S's first four resets (65203 failed in both runs), not a mismatch count: all four
  resets matched with reading and commit-target differences of exactly 0.0 (§2.3).

**What this does not show.** It does not show "LeWM-driven closed-loop success" in the protocol's
sense (R9.8: G-bar **and** G-NI must both hold, beside the four twin tests); it does not show that
LeWM is needed (W was not better than H-rule: 1 reset against 6, p = 0.992); it does not show that
W is worse than the rule beyond the allocated margin either. Nothing here is about v1's 0/150, a
LeWM policy (LeWM picks only the place aim; the pick is P-3's and the place primitive is e9's),
Arena or the real G1, other conditions, other seeds, or whether pretraining matters (no
random-init floor ran for W).

**The canonical status sentence** (DECISIONS 2026-10-02, R7) is updated by **R18.31**, as a named
factual correction (its clause "TASK-080's gated Stage S has not run" became false). It now states
Stage S's counts and its L-NEAR row; it states no LeWM success claim (§5).

Protocol: [`apple_lewm_c1m_v2_pred_readout.md`](apple_lewm_c1m_v2_pred_readout.md) (STATUS FROZEN;
frozen block `src/embodied_jepa/lewm_pr_v2.py`, sha256
`0fc095dc947f0ac74ebf6d1c541098a1e6fec1897592256f97ac8962b97be064`, unchanged through every
stage). Stage 0 record: [`apple_lewm_c1m_v2_pred_readout_stage0.md`](apple_lewm_c1m_v2_pred_readout_stage0.md).
Manifest: `benchmarks/manifests/apple-lewm-pr-v2.json`. Rulings: R18.1–R18.32 in
[DECISIONS.md](../DECISIONS.md), each decided or recorded by Claude under owner delegation
(2026-09-30). The records of K0′, C′, R and D are protocol §8.1–§8.6; this document restates
their outcomes and records Stage S, read from its report.

---

## 1. Stage-by-stage record

Every stage ran once, from a fresh, clean worktree of a merged revision, on an independent
reviewer's reported GO, after a full `pytest` at that revision (G-tests). No stage was voided or
repeated. Reports are git-ignored and stay on the Linux PC (RTX 5080), with copies in the SSD
evidence store. Times are UTC.

| stage | revision | worktree (under `/home/huhn/develop/emai/worktrees/`) | report | report sha256 | outcome |
|---|---|---|---|---|---|
| K0′ (before the freeze) | `931281a` | `task080-k0` | `outputs/task080-k0-1/report.json` | `a0939e3eea7b695d1c3029e733970bcd123ea62db76192104c69479ad9314c16` | **K0′-PASS** (pooled τ_commit = 1.0 cm; §8.1, R18.22) |
| C′ | `514110d` | `task080-stagec` | `outputs/task080-corpus-1/report.json` | `63b085df119f59219c24057ad4f592a59d20f43ff5ca210c32fd0e570bf6b0ca` | **CORPUS-SEALED** (499 of 500 roots; §8.2, R18.23) |
| R, G-tests | `33cea5c` | `task080-stager` | `outputs/task080-tests-1/report.json` | `acbe897da8361186d3319c1e200b33495fd735492e0835f77a58cb56ea267464` | TESTS-PASS |
| R, featurisation (GPU) | `33cea5c` | `task080-stager` | `outputs/task080-featurise-1/report.json` | `cda16651d5c2fdcb30f8ab6fd7841dbbabddeb4d2e17c6d7d8518d052d5883f2` | FEATURISED |
| R, offline gate | `33cea5c` | `task080-stager` | `outputs/task080-rgate-1/report.json` | `bedb896692a9718ac598fa93c5d7dcc199090bf4cfa13d14fa0bd0fc53848ea7` | **R-PASS** (offline; §8.4, R18.25) |
| D (development, 16 resets) | `db34adc` | `task080-staged` | `outputs/task080-d-1/report.json` | `1602447a08523f49d3924f6b38d10275a417d1ccd4a90bc8cd70d22ef79a0611` | **D-PASS** (§8.6, R18.27) |
| **S (gated, 64 resets)** | `bfae0fa` | `task080-stages` | `outputs/task080-s-1/report.json` | **`53cfdec19d4c570cf50af00bf33fc236847b4e520cba4489e4991b8bf079393e`** | **L-NEAR** (§2 here, R18.30) |

Outcomes of the earlier stages, as their records give them (not re-measured here): K0′ left
τ_commit at 1.0 cm with a ceiling of 32/32; Stage R's gates R0–R3, A1 and A2 passed, with W's
offline predicted count 56.92 of 64 against A1's 56 (a prediction, not a closed-loop count); Stage
D, a non-gating development cohort, scored W 16/16, N 8/16, L-shuf 5/16, L-mean 9/16 and the
privileged ceiling H-final(commit) 16/16.

## 2. Stage S: L-NEAR (protocol §8 step 7, §8.7, §9.5)

### 2.1 The run

Stage S ran once, on the reviewer's reported GO (#158,
<https://github.com/RaaSaaR-org/open-embodied-jepa/pull/158#issuecomment-6051252854>, posted
02:58:24 UTC on 2026-10-08), at `bfae0fa`, #158's merge commit, on a clean tracked tree with STATUS
FROZEN and the frozen sha `0fc095dc…be064`, in the worktree
`/home/huhn/develop/emai/worktrees/task080-stages`, with the command the GO named, once, on the CPU.

| | UTC (2026-10-08) | seconds |
|---|---|---:|
| start; G-quiet passed (load 0.06 / 0.08, bar 2.0 / 2.0; no wait) | 02:58:59 | – |
| in-run G-tests: 2070 passed, 37 skipped, at `bfae0fa`, clean | 02:58:59–03:01:42 | 163 |
| G-repro (8 of 8 checks; 170 train roots decoded, no test split) | – | 41 |
| first render; `first_outcome_utc` | 03:02:25; 03:02:28 | – |
| arms, in order (log lines, converted from CEST): W, N, L-shuf, L-mean, L-rand, H-rule, H-sysid, H-final(commit), H-read, H-now done at | 03:06:12, 03:08:02, 03:11:46, 03:15:28, 03:17:17, 03:18:29, 03:19:39, 03:21:49, 03:23:56, 03:25:04 | – |
| W's determinism re-run (65200–65203) done | 03:25:19 | – |
| end, report written: **L-NEAR** | 03:25:20 | **1 581** (cap 21 600) |

- **Guards.** G-hash checked TASK-076's 84 pins, TASK-077's 13 pins and this task's 7 own pins,
  with the protocol document's pin `b454d616…1b50`; the six checkpoints matched their sha256s; the
  runner accepted the Stage R report (`bedb8966…48ea7`) as a non-debug R-PASS report and the Stage D
  report (`1602447a…0611`, whose sha256 the GO checked) as a non-debug D-PASS report, both of the
  sealed corpus `deebd83d…7c4e` (G-split). The revision at the end was still `bfae0fa` and the
  tracked tree stayed clean. No non-finite field. G-memory: peak process-tree PSS 9.65 GiB (RSS
  11.11 GiB, 9 processes; cap 12), MemAvailable 25.2 GiB at the start. G-disk: at least 64.2 GiB
  free throughout. Threads as pinned (MKL and OMP 6, OpenBLAS 16), 4 world-model workers; the pool
  closed with all 4 workers joined and none killed.
- **Every attempt** (10 arms × 64 resets) ran its 725 steps to `policy_complete`, with a latched
  grasp; no reset was refused before 405 in any arm; no fallback in any arm; the post-pick plate
  move was applied and no contact with the plate before s1 was recorded on any attempt. No render
  disagreement between arms and no render retry. `task_truth_in_controller` was 0 on every attempt
  of W, N, L-shuf, L-mean, L-rand, H-rule and H-sysid (G-privileged), and every attempt passed the
  privileged-read check.
- **One G-repro re-render disagreement, resolved by its majority rule.** In G-repro's re-render of
  the carried TASK-072 run-1 seeds, seed 51533's three renders gave two identical frames and a
  third differing by one level in 4 pixels, with equal simulator states; the majority frame was
  used and all 8 reproduction checks passed. This is the renderer mitigation carried from TASK-073
  (`render_majority`: a third render decides, no majority would be a V, every disagreement is
  recorded; [apple_wm_critic_v2.md](apple_wm_critic_v2.md) §15), and TASK-076's M-a had the same
  case (root 51235). It is not a V, and it touches no cohort-S frame (`render_disagreements` for S
  is empty).

### 2.2 Counts

Counted success: the apple at rest on the plate under `apple_at_rest_v0` after a latched grasp and
a latched place (T71-R1/R2). Exact (Clopper–Pearson) 95 % intervals on 64 resets. Paired
differences W − arm, with the reset-bootstrap 95 % percentile interval (10 000 resamples, salt
8206; `lewm_pr_v2.paired_interval`) and the discordant resets:

| arm | what it is | count | 95 % (exact) | W − arm [95 %] | W only / arm only |
|---|---|---:|---|---|---|
| **W** | LeWM (seed 66800, flagged `last_two_triggered`), R-S | **58/64** | 0.807–0.965 | – | – |
| N | action-blind twin (zero commands), R-N | 26/64 | 0.285–0.536 | +32 [+23, +40] | 34 / 2 |
| L-shuf | scene-blind twin (another reset's 405 frame), R-S | 21/64 | 0.216–0.457 | +37 [+27, +46] | 41 / 4 |
| L-mean | scene-blind twin (the mean 405 latent), R-S | 30/64 | 0.343–0.598 | +28 [+18, +37] | 32 / 4 |
| L-rand | a feasible grid aim drawn at random, no refinement | 11/64 | 0.089–0.287 | +47 [+39, +54] | 48 / 1 |
| **H-rule** | hand-written rule, not learned (see below) | **63/64** | 0.916–0.9996 | **−5 [−10, 0]** | 1 / 6 |
| H-sysid | hand-written linear system-identification fit, not a world model | 62/64 | 0.892–0.996 | −4 [−10, +1] | 2 / 6 |
| H-final(commit) | **privileged** look-ahead ceiling, not learned | 62/64 | 0.892–0.996 | −4 [−10, +1] | 2 / 6 |
| H-read | **privileged**, reported only | 64/64 | 0.944–1.000 | −6 [−11, −2] | 0 / 6 |
| H-now | **privileged**, reported only (the true plate at 405) | 1/64 | 0.0004–0.084 | +57 [+52, +62] | 57 / 0 |

**What H-rule and H-sysid use** (TASK-077 §5.3, carried by §6). Both are non-privileged at run time
(0 task-truth reads) and neither is a world model or learned end to end; both read the plate at
405 with p̂, the same R-plate ridge readout of the encoded 405 frame (frozen DINOv2 features) from
which W's candidate box is built, and the palm position h from proprioception. **H-rule**
(`RuleCommit`) computes the declared reactive-plate rule's fixed point g\* = (p̂ − κh)/(1 − κ)
(κ = −0.5) and commits it once, clipped to the box: **it is given the simulator's plate law**, a
hand-written law of the imposed condition, which W must learn from data. **H-sysid** (`SysidAim`)
uses a linear fit plate(r) ≈ c + A·p̂ + B·h + C·g on TASK-077's train split, inverted with W's
controller form, clipped. H-final(commit), H-read and H-now are privileged (simulator clones or
the true plate) and are never learned results.

### 2.3 The determinism re-run (R17.21, R17.27, R17.38)

The rule, as the report records it: W is re-run on S's first four resets; a reset refused before
405 must be refused identically; a committed reset must commit again with its R-plate reading
within 0.1 cm (R17.21/R17.27) and the same success outcome, and its commit target within 0.6 cm
(R17.38, an amendment that gates the target; it supersedes the frozen block's older text "target
… reported, not gated" for the target only); **any gated mismatch is V** (a G-determinism guard
error; §8.7).

| reset | committed (run, re-run) | success (run, re-run) | reading difference | commit-target difference |
|---|---|---|---:|---:|
| 65200 | yes, yes | S, S | 0.0 cm | 0.0 cm |
| 65201 | yes, yes | S, S | 0.0 cm | 0.0 cm |
| 65202 | yes, yes | S, S | 0.0 cm | 0.0 cm |
| 65203 | yes, yes | –, – | 0.0 cm | 0.0 cm |

All four match exactly (`ok` true; largest target difference 0.0 m). The log line "W determinism:
3/4" counts the re-run's successes: 65203 is one of W's six failures in the gated run and failed
identically in the re-run. **The stage is not void**; no quantity differed.

### 2.4 The rows (§9.5, first match)

| row | condition | here |
|---|---|---|
| V | the void rule | no guard error; determinism matched |
| S-VOID-CEILING | H-final(commit) < 56/64 | 62 ≥ 56: no |
| L-NO-GAIN | a twin test fails and W is detectably no better | all four tests pass: no |
| L-INFERIOR | upper bound of W − C < −8/64 | upper bound 0: no |
| L-PASS | G-bar, G-NI and the four tests all pass | G-NI fails: no |
| L-TWIN-NEAR | a twin test fails within noise | none fails: no |
| **L-NEAR** | **G-NI fails, W not detectably inferior beyond δ** | **yes** |

C = H-rule (63 > H-sysid's 62). `clause_fires` false. The ladder was recomputed from the report's
per-reset outcomes with the frozen block's `decide_s` and gives the same row, intervals and p
values.

### 2.5 The two claims (R9.8, protocol §1)

| claim | what the protocol requires | on S | status |
|---|---|---|---|
| **Primary, "LeWM-driven closed-loop success"** | (1) W decides from the encoded frame, no privileged read; (2) G-bar W ≥ 56/64; (3) G-NI against the better of H-rule and H-sysid within δ = 8/64; (4) W beats N, L-shuf, L-mean and L-rand (exact one-sided McNemar p < 0.01). Items 2 and 3 must both hold. | (1) holds (0 task-truth reads); (2) passes, 58 ≥ 56; (3) **fails** (lower bound −10 < −8); (4) passes for all four | **Not made.** Only L-PASS makes it. |
| **Secondary, "LeWM needed"** (reported only, never a gate) | W detectably better than the better of H-rule and H-sysid (exact one-sided McNemar p < 0.01) | W only 1, H-rule only 6; p = 0.992 | **Not shown** (and not expected, §1). |

So under L-NEAR the record is: a LeWM-driven controller's aim choice reached the bar and beat its
action-blind, scene-blind and random twins on the gated cohort, but did not meet the preregistered
non-inferiority to the hand-written rule; **no claim is made**. "W beat its twins" is the result
of the declared twin tests, not a claim: the protocol attaches a claim only to L-PASS.

### 2.6 The committed aims

The aim error is the commit target's distance from the rule's fixed point g\* at 405 (the quantity
Stage R's offline aims mapped through the τ curve); the landing miss is its distance from where the
plate actually was at s1 = 525. τ_commit = 1.0 cm.

| arm | median aim error (cm) | p87.5 | max | aims > τ_commit | median landing miss (cm) | p87.5 | max | clip-binding | seconds per attempt (median / max) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| W | 0.575 | 0.968 | 1.452 | 5 of 64 | 0.537 | 0.945 | 1.956 | 0 | 13.77 / 16.21 |
| N | 2.474 | 3.219 | 4.078 | 59 of 64 | 3.376 | 4.461 | 5.961 | 0.219 | 6.83 / 7.10 |
| L-shuf | 2.962 | 4.190 | 5.573 | 61 of 64 | 4.394 | 6.017 | 9.063 | 0.328 | 13.88 / 14.40 |
| L-mean | 1.803 | 3.092 | 4.066 | 53 of 64 | 2.400 | 4.729 | 5.400 | 0.078 | 13.77 / 14.42 |
| L-rand | 7.170 | 13.241 | 18.720 | 61 of 64 | 10.279 | 17.809 | 21.405 | 0 | 6.67 / 7.20 |
| H-rule | 0.493 | 0.587 | 0.734 | 0 of 64 | 0.197 | 0.301 | 0.694 | 0.344 | 4.50 / 4.55 |
| H-sysid | 0.766 | 1.419 | 1.881 | 23 of 64 | 0.782 | 1.485 | 2.110 | 0 | 4.31 / 4.39 |
| H-final(commit) | 0.478 | 0.508 | 0.517 | 0 of 64 | 0.087 | 0.117 | 0.131 | 0 | 8.27 / 8.51 |
| H-read | 0.478 | 0.797 | 1.729 | 3 of 64 | 0.522 | 1.025 | 2.057 | 0 | 8.16 / 8.44 |
| H-now | 6.357 | 7.533 | 7.924 | 64 of 64 | 9.263 | 10.225 | 11.357 | 0 | 4.22 / 4.43 |

- **W's six failures** (65203, 65212, 65225, 65237, 65245, 65247) had aim errors of 0.29–1.06 cm and
  landing misses of 0.66–1.96 cm; on each, the apple ended 4.17–4.55 cm from the plate (the
  report's `final_distance_cm`), and H-rule, H-sysid and H-final(commit) all succeeded on all six.
  H-rule's one failure (65262; aim error 0.53 cm, landing miss 0.42 cm) is a W success.
- **Reported, post hoc, not a test:** all six of W's failures are among the 17 resets on which W's
  refinement stopped at its 10-refinement cap instead of converging (11 of those 17 succeeded; 47
  of 47 converged resets succeeded). This is an observation on one run, not a measured cause; any use of
  it needs its own preregistration on fresh seeds.
- **The ceiling is not perfect.** H-final(commit) landed within 0.131 cm of the plate on every
  reset but failed on 65234 and 65238 (the apple ended 4.00 and 4.02 cm from the plate), both W
  successes; H-final's count (62) is below H-rule's (63). Near the scorer's boundary the outcome
  depends on more than the aim.
- W's refinement ran 148–157 roll-outs per decision, all 147 grid candidates were feasible on every
  reset for W, N, L-shuf and L-mean, and W's decision took 9.5 s in median. W's clip-binding
  fraction is 0 (H-rule's box clip bound on 22 of 64 resets).

### 2.7 Stage R's offline predictions against Stage S's counts (reported, §9.5)

Stage R's predicted counts map gate-P's offline aim errors through the pooled τ curve; they are
predictions, not closed-loop counts (§8.4).

| arm | Stage R predicted (of 64) | Stage S count | S − predicted |
|---|---:|---:|---:|
| W | 56.92 | 58 | +1.08 |
| N | 27.79 | 26 | −1.79 |
| L-shuf | 19.67 | 21 | +1.33 |
| L-mean | 28.95 | 30 | +1.05 |
| L-rand | 11.85 | 11 | −0.85 |
| H-rule | 58.64 | 63 | +4.36 |
| H-sysid | 53.93 | 62 | +8.07 |

The offline prediction was within about two resets for W and the four twins, and under-predicted
both hand-written arms (by 4.36 and 8.07). Offline, W − H-rule was −1.73 of 64; on S it is −5.

### 2.8 Per reset

S = counted success, – = not. W's column adds its aim error in cm, and "c" marks a reset where W's
refinement stopped at the 10-refinement cap.

| reset | W (aim error, cm) | N | L-shuf | L-mean | L-rand | H-rule | H-sysid | H-final(commit) | H-read | H-now |
|---|---|---|---|---|---|---|---|---|---|---|
| 65200 | S 0.31 | – | – | – | – | S | S | S | S | – |
| 65201 | S 1.45 c | – | – | – | – | S | S | S | S | – |
| 65202 | S 0.64 | – | – | – | S | S | S | S | S | – |
| 65203 | – 1.06 c | – | S | S | – | S | S | S | S | – |
| 65204 | S 0.73 | – | – | S | – | S | S | S | S | – |
| 65205 | S 0.35 | S | – | – | – | S | S | S | S | – |
| 65206 | S 0.30 | – | – | – | – | S | S | S | S | – |
| 65207 | S 0.65 c | – | S | S | – | S | S | S | S | – |
| 65208 | S 0.54 | S | S | S | S | S | S | S | S | – |
| 65209 | S 0.67 | – | S | S | S | S | S | S | S | – |
| 65210 | S 0.57 | – | – | S | S | S | S | S | S | – |
| 65211 | S 1.06 c | S | – | – | – | S | S | S | S | – |
| 65212 | – 0.89 c | – | S | S | S | S | S | S | S | – |
| 65213 | S 0.58 c | – | – | – | – | S | S | S | S | – |
| 65214 | S 0.97 c | – | S | – | – | S | S | S | S | – |
| 65215 | S 0.79 | – | S | – | – | S | S | S | S | – |
| 65216 | S 0.44 | – | – | S | S | S | S | S | S | – |
| 65217 | S 0.78 | – | S | S | – | S | S | S | S | – |
| 65218 | S 0.49 c | S | S | S | – | S | S | S | S | – |
| 65219 | S 0.53 | S | S | – | – | S | S | S | S | – |
| 65220 | S 0.37 | S | – | – | – | S | S | S | S | – |
| 65221 | S 0.48 | S | – | S | – | S | S | S | S | – |
| 65222 | S 0.46 | – | – | – | S | S | S | S | S | – |
| 65223 | S 0.61 | S | – | S | – | S | S | S | S | – |
| 65224 | S 0.55 | S | – | – | – | S | S | S | S | – |
| 65225 | – 0.29 c | S | – | – | – | S | S | S | S | – |
| 65226 | S 0.54 | – | – | – | – | S | S | S | S | – |
| 65227 | S 0.97 | – | – | – | – | S | S | S | S | – |
| 65228 | S 0.56 | S | – | – | – | S | S | S | S | – |
| 65229 | S 0.69 | S | S | S | – | S | S | S | S | – |
| 65230 | S 0.43 c | – | S | – | – | S | S | S | S | – |
| 65231 | S 0.90 | S | – | S | – | S | S | S | S | – |
| 65232 | S 0.56 | – | – | – | – | S | S | S | S | – |
| 65233 | S 0.51 | – | S | S | – | S | S | S | S | – |
| 65234 | S 0.67 | – | – | – | – | S | S | – | S | – |
| 65235 | S 0.67 | S | S | S | – | S | S | S | S | – |
| 65236 | S 0.59 | – | S | S | – | S | S | S | S | – |
| 65237 | – 0.38 c | S | – | S | – | S | S | S | S | – |
| 65238 | S 1.15 | – | – | – | – | S | S | – | S | – |
| 65239 | S 0.78 | – | – | – | – | S | S | S | S | – |
| 65240 | S 0.67 | – | – | – | – | S | S | S | S | – |
| 65241 | S 0.53 | S | – | S | – | S | S | S | S | – |
| 65242 | S 0.98 | – | – | – | – | S | S | S | S | – |
| 65243 | S 0.61 | S | – | – | – | S | S | S | S | – |
| 65244 | S 1.16 | – | – | S | S | S | S | S | S | – |
| 65245 | – 0.86 c | – | S | – | – | S | S | S | S | – |
| 65246 | S 0.31 | – | – | – | – | S | S | S | S | – |
| 65247 | – 0.98 c | – | S | S | – | S | S | S | S | – |
| 65248 | S 0.53 | – | – | S | – | S | S | S | S | – |
| 65249 | S 0.46 | S | S | – | – | S | S | S | S | – |
| 65250 | S 0.18 c | S | – | S | – | S | – | S | S | – |
| 65251 | S 0.55 | – | S | S | – | S | S | S | S | – |
| 65252 | S 0.90 | – | – | – | – | S | S | S | S | – |
| 65253 | S 0.61 | S | S | S | – | S | S | S | S | – |
| 65254 | S 0.33 | – | – | – | S | S | S | S | S | – |
| 65255 | S 0.50 c | S | – | – | S | S | S | S | S | – |
| 65256 | S 0.71 | S | – | – | – | S | S | S | S | – |
| 65257 | S 0.22 | S | – | S | S | S | – | S | S | S |
| 65258 | S 0.42 c | – | S | S | – | S | S | S | S | – |
| 65259 | S 0.85 | S | – | S | – | S | S | S | S | – |
| 65260 | S 0.38 c | – | – | – | – | S | S | S | S | – |
| 65261 | S 0.77 | S | – | S | – | S | S | S | S | – |
| 65262 | S 0.53 | – | – | – | – | – | S | S | S | – |
| 65263 | S 0.56 | S | – | S | – | S | S | S | S | – |

## 3. Caveats

1. **One run, one model seed.** Everything is seed 66800's W and N, one run of 64 resets. **W-66800
   is flagged `last_two_triggered`**: it selected one of its last two checkpoints, and every W
   curve was lowest at its last point (TASK-077 results §3, caveats 1–2), so W may have been
   slightly under-trained.
2. **Simulation only, under an imposed condition.** C1-M (a post-pick plate move at step 300,
   uniform over a 4 cm disc, then cell A's reactive-plate rule κ = −0.5, L = 2 from 405, one aim
   committed at 405) is a simulator law, declared for this test; nothing transfers as such to
   Arena or the real G1 (§14).
3. **LeWM picks only the place aim.** P-3 (behaviour cloning on a frozen DINOv2 readout, not a
   world model) picks; e9's scripted place primitive executes the committed aim with no re-aim.
   W ranks aims inside a box built from R-plate's single-frame reading and the proprioceptive palm,
   and refines within it. This is not a LeWM policy.
4. **The non-inferiority comparator knows the law.** H-rule is given the simulator's plate law
   and needs no learning beyond R-plate; W must learn the law's effect from data. That is the
   design (R9.8: the primary claim is non-inferiority to the best hand-written arm, not
   superiority), and H-rule scored 63/64 on S, above Stage R's offline 58.64.
5. **δ = 8/64 is an allocation, not calibrated** (R15.6's label). G-NI's power was demanding by
   design: with the better of two comparators, 0.69–1.00 at W = C and 0.14–0.43 at W = C − 4/64
   (§10). The observed difference is −5/64.
6. **The privileged ceiling is not a hard ceiling** on this scorer: H-final(commit) failed twice
   with landing misses under 0.14 cm (§2.6).
7. **The cohort S has now been read.** Its 64 resets (65200–65263) are no longer fresh for these
   models, readouts or arms. The protocol allows a repeat after L-NEAR only on fresh seeds and with
   its own ruling (§9.5).
8. **The refinement-cap observation (§2.6) is post hoc** and is not evidence of a cause.

## 4. Rows and what follows

- **Row: L-NEAR** (escalate, no clause, no claim). The clause fires only on L-NO-GAIN or
  L-INFERIOR (§12); neither holds. Nothing in TASK-077's scope (R17.10) is closed; TASK-077's row
  is unchanged.
- **TASK-079's precondition is not met.** R18.21 recommended that a TASK-080 L-PASS count as
  TASK-077's "equivalent row"; L-NEAR is not L-PASS.
- **Recommendation (R18.32, recorded only; not a preregistration and not a GO).** The protocol's
  escalation text for L-NEAR is "escalate, no clause, no claim", and "any repeat after L-TWIN-NEAR,
  L-NEAR or L-BAR needs fresh seeds and its own ruling" (§9.5). Recommended: **do not repeat
  Stage S** with this design on fresh seeds. At the observed W − H-rule of −5/64, G-NI's simulated
  power is below one half (0.14–0.43 at −4/64, §10), so a repeat would most likely end L-NEAR
  again and would spend fresh seeds without a different question. Escalate to the owner with
  TASK-080 closed at L-NEAR; any next LeWM task on v2 is drafted first as a design note, with its
  own preregistration and fresh seeds. Two directions, for the owner's choice, neither ruled here:
  (a) a condition in which no hand-written arm is given the plate law (so that "LeWM needed" is a
  testable question rather than "not expected"), declared as a task change; (b) W's precision at
  the commit, where §2.6's post-hoc cap observation would be a hypothesis to preregister, not a
  finding. The product goal, the LeWM backend and v2 are not affected by this row.

## 5. R7 (R18.31)

R18.28's sentence said "TASK-080's gated Stage S has not run", which is now false. R18.31 replaces
the v2 LeWM part of R7 with Stage S's counts and row, as a second named exception to R18.12 (a
factual correction, not a success claim; only an L-PASS with its own reviewed ruling could let R7
state a gated LeWM success, and L-NEAR is not one). The sentence in force is quoted in
[DECISIONS.md](../DECISIONS.md) (decision 2026-10-08 (d), R18.31) and in the entry documents.

## 6. Process and evidence

- Report (git-ignored, in the worktree `task080-stages`): `outputs/task080-s-1/report.json`,
  sha256 **`53cfdec19d4c570cf50af00bf33fc236847b4e520cba4489e4991b8bf079393e`**; log
  `outputs/task080-s-1.log` (sha256 `70952ee2…9edc`; the ten arm lines, the determinism line and
  the outcome); stdout capture `outputs/task080-s-1.stdout` (empty, 0 bytes, as `--log` redirects
  the process after it opens).
- A copy of the three is in `~/develop/emai/evidence/task080-stages/` with the manifest
  `_checksums/task080-stages.sha256` (sha256 `a71a5da3…4fdf`), verified against the source and
  the copy, and an entry in the evidence README.
- Every number in §2 is read from that report (counts, per-reset outcomes, decision block,
  determinism block, attempt records); the exact intervals are computed from the counts, and the
  paired intervals and p values were recomputed with the frozen block's functions and match the
  report. Nothing was re-run.
- The run worktrees `task080-k0`, `task080-stagec`, `task080-stager`, `task080-staged` and
  `task080-stages` are not edited; their outputs are copied to the SSD evidence store
  (`~/develop/emai/evidence/task080-{k0,stagec,stager,staged,stages}/`). They can be archived like
  the others (copy, checksum, manifest row, verify before removing).
