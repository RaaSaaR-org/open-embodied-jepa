# TASK-087 results — Phase 2: robot-grounded LeWM on `play-v1`, offline

Protocol: [grounded_lewm_v1.md](grounded_lewm_v1.md) (FROZEN at `dab7475`, #188; Stage F's reader
fixed at `bcda883`, #189, R24.20). Rulings on this record: DECISIONS 2026-10-10 (ac),
R24.21–R24.25, decided by Claude under owner delegation.

**This is an offline world-model comparison, not a control result and not an Apple→Plate
experiment.** No planner or closed loop ran. **R7's canonical status sentence does not change.**

## 1. Row: P2-FAIL

The gate needs an eligible grounded arm (S: + robot state; G: + state, joint-change and
inverse-dynamics losses) to beat plain LeWM (P) of the same seed on **all of** (a) action
sensitivity, (b) top-1 ranking, (c) accuracy within δ and (d) beating copy-last, on all three
seeds, at h = 1, 4 and 8 (§8). Neither did:

- **At h = 4 and h = 8, S and G both meet (a)–(d) on every seed.**
- **At h = 1, both fail (a):** their wrong / true ratio is *lower* than P's on every seed
  (G: Δsens −0.007, −0.016, −0.010, every 97.5 % interval below 0; S: −0.005, −0.018, −0.003, two
  of three intervals below 0). S also fails (b) at h = 1 on two seeds. So P2-PASS and P2-SHORT (which
  needs h = 1) are both out, and the row is **P2-FAIL**.
- Row 2's validity checks passed on every seed (P beats copy-last at h = 1, 4, 8 and the no-action
  model N at h = 8, and P's sensitivity at h = 8 is above 1), so the comparison is meaningful.

`test_report.json` sha256 `aed62c91…ca18`.

## 2. The numbers (test split, 5 634 roots in 160 episodes)

Per seed 87100 · 87101 · 87102. sens = Σ wrong-command error / Σ true-command error; top-1 of 16
candidate command sequences (chance 0.0625); copy = Σ true error / Σ copy-last error.

| h | arm | sens | top-1 | copy |
| --- | --- | --- | --- | --- |
| 1 | P | 1.101 · 1.110 · 1.111 | 0.345 · 0.345 · 0.346 | 0.869 · 0.867 · 0.864 |
| 1 | S | 1.096 · 1.092 · 1.108 | 0.344 · 0.356 · 0.381 | 0.864 · 0.866 · 0.866 |
| 1 | G | 1.094 · 1.094 · 1.101 | **0.409 · 0.402 · 0.413** | 0.856 · 0.855 · 0.850 |
| 4 | P | 1.445 · 1.474 · 1.485 | 0.636 · 0.639 · 0.642 | 0.691 · 0.690 · 0.687 |
| 4 | S | 1.540 · 1.516 · 1.524 | 0.677 · 0.676 · 0.670 | 0.656 · 0.662 · 0.662 |
| 4 | G | 1.508 · 1.522 · 1.546 | 0.704 · 0.697 · 0.703 | 0.667 · 0.668 · 0.661 |
| 8 | P | 1.752 · 1.778 · 1.801 | 0.743 · 0.745 · 0.742 | 0.595 · 0.592 · 0.592 |
| 8 | S | 1.884 · 1.861 · 1.879 | 0.793 · 0.785 · 0.794 | 0.557 · 0.561 · 0.561 |
| 8 | G | 1.862 · 1.897 · 1.902 | 0.802 · 0.800 · 0.815 | 0.563 · 0.562 · 0.558 |

Paired differences against P of the same seed (97.5 % episode-cluster bootstrap):

| | h | Δsens | Δtop-1 | acc ratio X / P | criteria met |
| --- | --- | --- | --- | --- | --- |
| G | 1 | −0.007 [−0.011, −0.003] · −0.016 [−0.021, −0.012] · −0.010 [−0.015, −0.006] | +0.065 [0.049, 0.081] · +0.056 [0.040, 0.074] · +0.068 [0.051, 0.084] | 0.985 · 0.987 · 0.983 | (b)(c)(d), not (a) |
| G | 4 | +0.062 · +0.049 · +0.061 (all > 0) | +0.068 · +0.058 · +0.061 (all > 0) | 0.966 · 0.968 · 0.962 | all |
| G | 8 | +0.110 · +0.119 · +0.101 (all > 0) | +0.059 · +0.055 · +0.072 (all > 0) | 0.947 · 0.950 · 0.942 | all |
| S | 1 | −0.005 [−0.009, −0.001] · −0.018 [−0.023, −0.015] · −0.003 [−0.007, 0.002] | −0.001 [−0.016, 0.015] · +0.011 [−0.005, 0.027] · +0.035 [0.019, 0.052] | 0.994 · 0.999 · 1.002 | (c)(d); (b) on 87102 only |
| S | 4 | +0.095 · +0.042 · +0.039 (all > 0) | +0.041 · +0.037 · +0.029 (all > 0) | 0.949 · 0.960 · 0.964 | all |
| S | 8 | +0.132 · +0.083 · +0.077 (all > 0) | +0.049 · +0.040 · +0.052 (all > 0) | 0.937 · 0.948 · 0.947 | all |

δ from val (§6.4): 0.0065 (h = 1), 0.0099 (h = 4), 0.0114 (h = 8). Every (c) upper bound of S and
G is ≤ 1 + δ (G's accuracy ratio is below 1 everywhere: G predicts the true-command future better
than P). Full intervals for every cell are in the report.

**Why (a) fails at h = 1.** G and S predict one step ahead slightly *better* than P under the true
commands (G's true-command error is 1.3–1.7 % lower) and also under other episodes' commands, so the ratio
falls; in absolute terms, too, the gap between wrong and true commands is smaller for them at
h = 1 (reported Σ wrong − Σ true, per root: G −0.002 to −0.004 against P, every interval below
0). Our reading, not tested here: with the joint velocities as input, the next frame is largely
predictable from the robot's own motion, so the one-step prediction leans less on the command.
At h = 4 and 8 the opposite holds: the gap grows by +0.008 to +0.041 per root for S and G on every seed (at h = 16, outside the training horizon, it grows for G but falls for S on two seeds).

## 3. Reported only

- **Controls.** C (state fusion + palm readout, the apple WM v2–v4 recipe adapted) is *worse than
  copy-last* at h = 1 on every seed (copy ratio 1.11–1.14) and below P on sensitivity at every gated
  horizon; it would fail (a) everywhere. I (inverse dynamics only) meets (a)–(d) at h = 1 on two seeds
  (Δsens +0.003 to +0.005, Δtop-1 +0.016 to +0.017) but not on 87102, and not at h = 8 on any
  seed; it does not meet all criteria on all seeds at any horizon. N's sens is 1 and top-1 0 by
  construction.
- **The inverse-dynamics caveat fixed in §9 applies.** G's and I's inverse-dynamics heads read the
  seven played command dimensions from *predicted* next latents with R² 0.98–0.99, against 0.58 for
  a ridge probe on *encoded* frame pairs (and 0.41–0.42 for the heads themselves on encoded pairs).
  So the predicted latents carry action information well beyond what one encoded frame pair shows;
  G's sensitivity and ranking gains may partly be action information placed in the prediction
  rather than better dynamics. S, which has no inverse-dynamics loss, shows gains of similar size at
  h = 4 and 8, which argues against that being the whole story, but this is not separated here.
- **G's state head** predicts the standardised right arm and hand state with MSE 0.012–0.013 at
  h = 1 and 0.036–0.037 at h = 16, against 0.18 and 1.02 for "no change".
- **Ranking ceiling.** No candidate's commands equal the root's own over any h (tie share 0 at every
  h, at start frame 0 and later), so the ceiling is top-1 = 1.
- **Collapse diagnostic.** Effective rank of the predicted latents at h = 8: P 679–682, S 660–684,
  G 681–684, C 671–687, I 667–683, N 614–640, against 1 051 for the encoded latents at the same
  frames. No arm collapsed; every predictor is lower-rank than the frames, as expected of a mean
  prediction.
- **h = 2 and 16.** At h = 16 (outside the training horizon) G has the highest top-1 (0.83–0.85
  against P's 0.76–0.79) and the lowest error. Against TASK-084 part B's plain-LeWM baseline (another
  corpus and recipe, not comparable as numbers): there, top-1 at h = 1 was 0.14–0.19 and wrong /
  true 1.04; here plain LeWM on `play-v1` reaches 0.35 and 1.10.
- **Training.** U = 60 000 updates (the probe's u\* 50 000 × 1.5 = 75 000, capped at 60 000 by
  §6.1); kept updates 36 000–57 000; `last_two_triggered` on S-87102 and G-87102 (flags only, §6.2);
  31–37 ms per update; 18 jobs, about 10 h of GPU time, each under its own lock; no job failed or
  was re-run. Parameters: P and N 3.12 M, S 3.18 M, C 3.97 M, I 6.27 M, G 6.39 M.

## 4. Run

- Stage F attempt 1 at `dab7475` stalled on memory and was stopped (§16 of the protocol, R24.20);
  evidence `~/develop/emai/evidence/task087-stagef-attempt1/`. Since the review of #189: in that
  attempt the log printed progress every 100 episodes and showed none after the first train
  episode, so fewer than 101 had been processed.
- Stage F re-run at `bcda883` (clean tree), 1 804 s, resident memory about 1.7 GB in one `ps` reading during pass B
  (not a peak measurement): 705 fit-sample episodes (259 085 frames), **0.943** of the standardised
  variance and **0.875** of the frame-to-frame change variance kept at k = 192; the projection's
  sha256 (`c196fc48…e33f`) and file sha256 equal attempt 1's. Train 2 874 and val 160 episodes;
  6.5 GB.
- Stage B (probe, P-87900, 60 000 updates): P's val copy ratio at h = 1 was 0.874; U = 60 000,
  cap 5 575 s per job, projected 9.3 GPU hours (no escalation).
- Stage T: 18 jobs at `bcda883`, 2026-10-09 19:34 – 2026-10-10 05:41 UTC.
- Stage V: val report sha256 `dbc05e79…ac38`, recorded at 05:49:58 UTC before Stage F′ started
  (05:50:23). Stage F′: 160 test episodes, 73 s. Stage E: 622 s.
- Evidence (each folder has `SHA256SUMS`): `~/develop/emai/evidence/task087-stagef/`,
  `task087-probe/`, `task087-val/`, `task087-test/`. Checkpoints in
  `checkpoints/task087-run/`, features in the run worktree's `outputs/task087-features/` (both
  git-ignored).

## 5. Caveats

- Offline only: ranking against the true future frame is not planning (a planner does not have
  that frame) and nothing here is a closed loop.
- One corpus (scripted play, right arm and hand, simulation), one camera at 112 px, one frozen
  encoder with one 4 × 4 pooling and 192-component projection, one architecture, one budget (which
  hit its cap; the val curves were still falling slowly), three seeds.
- The grounded arms have more parameters (G twice P's) and G and I more compute per update; a gain
  is not separated from capacity.
- The h = 1 failure is not only an artefact of the ratio: the absolute gap between wrong- and
  true-command errors is also smaller than P's at h = 1 (G on every seed, S on two). The same G
  models nevertheless rank the executed commands first more often than P at h = 1. Both facts
  stand.

## 6. What changes

- TASK-087 closes with **P2-FAIL** (R24.22). Under the stop rule (§8, R24.11), the grounded recipes
  S and G as specified here are **not carried into Phase 3 as "the grounded model"**; what Phase 3
  uses needs its own preregistration and ruling. Nothing else closes: LeWM, DINOv2, `play-v1` and
  the product goal are not abandoned.
- **R7 does not change** (R24.24).
