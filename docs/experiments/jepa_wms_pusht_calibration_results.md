# TASK-084 results — Phase 0 calibration on Meta's released JEPA-WM (Push-T); LeWM action-sensitivity baseline

Protocol: [jepa_wms_pusht_calibration.md](jepa_wms_pusht_calibration.md) (FROZEN at `176258b`,
#178; R22.1–R22.9). Rulings on this record: DECISIONS 2026-10-09 (t), R22.10–R22.13, decided by
Claude under owner delegation.

**This is not an Apple→Plate result. R7's canonical status sentence does not change** (R22.9,
R22.12).

## 1. Part A — Meta's released JEPA-WM on Push-T, upstream's planner and ours (gated)

**Row: P0-PASS** (protocol §4.5, first match; `decision.json` sha256 `d50c62b3…d7f9`).

### 1.1 Run

- At `176258b` (the frozen protocol's merge), worktree `task084-run`, runtime as protocol §3
  (upstream `13cf1d9`, Python 3.10.21, torch 2.7.0+cu128, RTX 5080, fp32), every hash pin
  checked by the runner. Smoke first (debug seed 9902, 2 episodes per arm: 2/2 and 2/2), then the
  eight gated jobs, one GPU lock each, 2026-10-09 00:01–08:19 UTC; every job exited 0, **no
  re-run**, all eight on one repository and one upstream revision; both arms planned from the same
  96 (initial, goal) pairs (the decision script's check). About 155 s per plan in both arms
  (median 155.5 s upstream, 154.6 s ours; 62 min per job).
- Evidence: `~/develop/emai/evidence/task084-run/` (per-run `episodes.jsonl`, `summary.json`,
  upstream's logs and CSV, the job logs, the queue script, `runtime-freeze.txt`,
  `decision.json`), `SHA256SUMS` sha256 `77541878…a4bf9a` over 644 files; the pre-freeze plumbing
  in `~/develop/emai/evidence/task084-plumbing/` with its own `SHA256SUMS`.

### 1.2 Counts

| Arm | Successes / 96 | Rate | Wilson 95 % | Per meta seed (1 · 2 · 3 · 4, of 24) |
| --- | --- | --- | --- | --- |
| upstream (Meta's CEM) | **67** | 69.8 % | 60.0–78.1 % | 17 · 15 · 16 · 19 |
| ours (`embodied_jepa.planning.CEMPlanner`) | **83** | 86.5 % | 78.2–91.9 % | 21 · 20 · 20 · 22 |
| published (arXiv 2512.24497, 3 seeds × 10 epochs × 96) | — | 70.2 % (2.8) | — | — |

- **G-REPRO passes:** upstream − published = −0.4 points (window ±10).
- **G-PLAN-ABS passes:** ours 86.5 % ≥ 60.2 %. **G-PLAN-REL passes:** ours − upstream =
  **+16.7 points**, paired bootstrap 95 % interval **[+7.3, +26.0]** (salt 8602, 20 000
  resamples); the lower bound is above −10, so the pass is "clear", not only on the point
  estimate. Discordant episodes: 20 solved only by ours, 4 only by upstream (exact two-sided
  McNemar p = 0.0015, reported).

### 1.3 What it shows, and what it does not

- **Upstream's pipeline reproduces the published Push-T number on this machine**: 69.8 % for one
  released checkpoint on 96 episodes against 70.2 % averaged over 30 checkpoint evaluations. The
  Wilson interval (60.0–78.1 %) is wide; this is agreement within the window, not a precise match.
  The runtime deviations of protocol §3.1 (one process, chunked roll-outs, no decoder heads, newer
  tensordict/torchrl, torch 2.7.0 on an RTX 5080) did not move it out of the window.
- **Our planner meets the Phase 0 gate**, so the plan's "else fix it first" does not apply and
  `embodied_jepa.planning.CEMPlanner` can be used as is where the plan needs a CEM. It was not
  worse than Meta's planner here; on these 96 episodes it scored higher.
- **"Ours is better" is not a preregistered claim** and is not made: the protocol tested only
  non-inferiority, the superiority interval and McNemar p are reported, one released checkpoint,
  one task, one planning setting (one 30-step open-loop plan per episode). Why it scored higher was
  not tested. Design differences that could explain it (protocol §4.1): ours returns the best
  candidate seen rather than the final elite mean, starts with a wider sampling distribution (about
  3.3–3.7 z-units against 1), keeps a 0.05 standard-deviation floor and carries the best candidate
  between iterations; any of them is a hypothesis, not a finding.
- **Scope.** Push-T is a 2-D pusher with a 2-D relative action; nothing here is about LeWM,
  G1 + Dex3, 14-D dexterous actions, the MuJoCo apple task or a closed loop with replanning. P0-PASS
  says the CEM implementation is not the bottleneck where a JEPA world model is known to plan; it
  does not say this repository's world models are plannable (part B and Phase 2 are about that).

## 2. Part B — action-sensitivity baseline (reported only; no gate, no row)

Run once at `176258b` (clean tree), CPU, 2026-10-08 23:51–23:55 UTC (271 s), on the 250 val roots
of `apple-c1m-v2`; TASK-077's W checkpoints as they are (sha256 `891d2664…76b8` for 66800,
flagged `last_two_triggered`; `27aeadab…c193` for 66801; `ba2614ec…97b4` for 66802). Report
sha256 `abd88f74…c799`; evidence `~/develop/emai/evidence/task084-probe/` with `SHA256SUMS`.

Per seed, from frame 405 with the root's own 60 executed commands; ratios are of summed
normalised MSE to the encoded frame at h with root-bootstrap 95 % intervals; the ranking is over 16
candidate command sequences (the true one and 15 other val roots'), chance top-1 0.0625 and chance
normalised rank 0.5.

| h | wrong / true (66800 · 66801 · 66802) | zero / true | true / copy-last | top-1 of 16 | normalised rank of the true sequence |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.04 · 1.04 · 1.04 (each [1.03, 1.05]) | 1.43 · 7.15 · 1.92 | 0.62 · 0.63 · 0.63 | 0.180 · 0.140 · 0.188 | 0.271 · 0.286 · 0.278 |
| 2 | 1.10 · 1.10 · 1.09 | 1.77 · 11.5 · 3.18 | 0.48 · 0.50 · 0.49 | 0.160 · 0.184 · 0.232 | 0.212 · 0.220 · 0.217 |
| 4 | 1.13 · 1.13 · 1.13 | 2.09 · 14.8 · 4.96 | 0.41 · 0.42 · 0.42 | 0.292 · 0.308 · 0.376 | 0.161 · 0.145 · 0.139 |
| 8 | 1.37 · 1.36 · 1.36 | 2.65 · 18.1 · 5.73 | 0.29 · 0.30 · 0.30 | 0.440 · 0.396 · 0.428 | 0.099 · 0.101 · 0.081 |
| 16 | 1.84 · 1.74 · 1.77 | 3.93 · 17.8 · 5.06 | 0.19 · 0.20 · 0.20 | 0.700 · 0.608 · 0.656 | 0.030 · 0.053 · 0.038 |
| 30 | 2.80 · 2.67 · 2.66 | 5.95 · 16.6 · 5.47 | 0.14 · 0.15 · 0.15 | 0.916 · 0.876 · 0.900 | 0.006 · 0.010 · 0.008 |
| 60 | 3.77 · 3.59 · 3.66 | 6.79 · 16.2 · 6.15 | 0.15 · 0.15 · 0.15 | 0.924 · 0.900 · 0.940 | 0.005 · 0.007 · 0.004 |

Intervals for every cell are in the report. The top-1 intervals at h = 1 are [0.136, 0.228],
[0.100, 0.184] and [0.140, 0.236], all above chance (0.0625); at h = 8 they are [0.380, 0.500],
[0.336, 0.456] and [0.364, 0.492].

What this baseline says, within its scope:

- **The models are action-sensitive at every horizon, but weakly at short ones.** One step after
  frame 405, replacing a root's commands with another root's raises the prediction error by only
  4 % (wrong / true 1.04, the interval excludes 1), and the true command sequence is ranked first
  among 16 on 14–19 % of roots (about 2–3 × chance). The discrimination grows with the horizon:
  about 40–44 % top-1 at h = 8 and 88–94 % at h = 30–60, where whole 30–60-step command
  sequences, not single steps, are being told apart.
- **The zero-command ratio differs widely between seeds** (1.43–7.15 at h = 1, 6.2–16.2 at
  h = 60), while the wrong-command ratio and the ranking agree closely. An all-zero command
  sequence is probably far from the corpus's executed commands (not checked here), so this ratio
  says more about how a model extrapolates than about how it uses actions; the wrong-command ratio
  and the ranking are the more comparable numbers for Phase 2.

Caveats: offline only, not a planning or closed-loop result; the val roots are not out of sample
for checkpoint selection (TASK-077 selected on val); one corpus (`apple-c1m-v2`, the C1-M place
phase from frame 405 with the corpus's executed commands), one encoder (frozen DINOv2 tokens pooled to
8 × 8), one recipe; the ranking scores against the true future frame, which a planner does not
have. This is the plain-LeWM reference Phase 2's preregistration may compare against.

## 3. What changes

- TASK-084 closes with **P0-PASS** (R22.11). Phase 0's two items are done: the plate line was
  closed by TASK-083 and the calibration gate is met. Phase 1 (the play corpus) needs its own
  preregistration and ruling; nothing here preregisters it.
- **R7 does not change** (R22.12): this is not an Apple→Plate result, and the canonical sentence
  stays as R21.20 left it.
