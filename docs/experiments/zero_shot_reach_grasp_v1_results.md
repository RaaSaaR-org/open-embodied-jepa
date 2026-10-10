# TASK-088 results — Phase 3: zero-shot reach and grasp-and-lift by short-horizon planning

Protocol: [zero_shot_reach_grasp_v1.md](zero_shot_reach_grasp_v1.md) (FROZEN at `028e6f3`, #192;
Stage D recorded at #193). Rulings on this record: DECISIONS 2026-10-10 (ag), R25.25–R25.29,
decided by Claude under owner delegation.

**This is a simulation result on G1's right arm and right Dex3, not an Apple→Plate experiment.**
**R7's canonical status sentence does not change** (R25.28).

## 1. Row: Z3-REACH

On the 64 + 64 fresh gated resets, the primary arm **P** (plain LeWM, TASK-087's P-87100, CEM to
the goal image only) reached **53 / 64** (exact 95 % interval 0.713–0.911) against the reach bar
of 52 / 64, and lifted **0 / 64** objects (0.000–0.056) against the grasp bar of 26 / 64. Reach
meets its bar (by one reset); grasp-and-lift does not.

Row 3 of §8, **Z3-REACH**: reach k = 53 ≥ max(⌈0.80 × 64⌉, 32) = 52, grasp k = 0 < 26. The
automatic void check (every run of §5.4 present, 64 episodes on the cohort's seeds, one revision,
clean trees, λ) found nothing, so the row is not Z3-VOID. The second arm **G** (+ state, joint-change
and inverse-dynamics losses; image + pose cost) **meets the reach bar** (62 / 64, 0.892–0.996) and
**not the grasp bar** (0 / 64). By §8, Z3-REACH escalates to a ruling; the plan's stop rule (reach
under 50 % for both P and G) does not apply.

## 2. Numbers (Stage S, gated)

All arms ran on the same 64 + 64 resets (paired). Exact 95 % intervals.

**Reach** (gated-reach, 880000–880063; success = palm within 5 cm for 10 consecutive commands):

| Arm | Success | 95 % | ≤ 3 cm | ≤ 8 cm | median final distance | median final orientation error |
| --- | --- | --- | --- | --- | --- | --- |
| **P-87100 (primary)** | **53 / 64** | 0.713–0.911 | 38 | 58 | 3.7 cm | 0.84 rad |
| **G-87100 (second arm)** | **62 / 64** | 0.892–0.996 | 53 | 64 | 2.3 cm | 0.16 rad |
| G-lat-87100 (G, image cost only) | 57 / 64 | 0.788–0.955 | 45 | 60 | 2.9 cm | 0.25 rad |
| G-pose-87100 (G, pose cost only) | 54 / 64 | 0.731–0.922 | 32 | 61 | 3.2 cm | 0.12 rad |
| P-87101 (replication) | 55 / 64 | 0.750–0.934 | 40 | 59 | 3.3 cm | |
| P-87102 (replication) | 54 / 64 | 0.731–0.922 | 36 | 63 | 4.4 cm | |
| G-87101 (replication) | 63 / 64 | 0.916–1.000 | 58 | 64 | 1.9 cm | |
| G-87102 (replication) | 63 / 64 | 0.916–1.000 | 57 | 64 | 2.1 cm | |
| ik (scripted IK follower, ceiling) | 64 / 64 | 0.944–1.000 | 64 | 64 | 0.1 cm | 0.01 rad |
| ik-nopress | 64 / 64 | 0.944–1.000 | 64 | 64 | 0.1 cm | |
| hold | 0 / 64 | 0.000–0.056 | 0 | 0 | 15.7 cm | |
| random | 0 / 64 | 0.000–0.056 | 0 | 4 | 14.7 cm | |

Paired (exact McNemar, reported): P against random 53 to 0 discordant (one-sided p = 1.1 × 10⁻¹⁶);
G against random 62 to 0; **G against P 9 to 0 (two-sided p = 0.0039)**; ik against P 11 to 0
(p = 0.001); ik against G 2 to 0 (p = 0.5). Stopped episodes ("projection infeasible"): G-87100 1
and P-87102 1 after their dwell had completed (counted as successes, since success is when the dwell
first completes), and P-87102 1 before it (a failure).

**Grasp-and-lift** (gated-grasp, 881000–881063; success = target ≥ 5 cm above its start and in grasp
contact for 20 consecutive commands; targets cube 32, can 18, apple 13, banana 1):

| Arm | Success | 95 % | grasp contact (episodes) | max rise (median / max) | palm ≤ 1 cm of a subgoal | stopped |
| --- | --- | --- | --- | --- | --- | --- |
| **P-87100 (primary)** | **0 / 64** | 0.000–0.056 | 12 | 0.2 / 1.6 cm | 41 | 0 |
| **G-87100 (second arm)** | **0 / 64** | 0.000–0.056 | 13 | 0.2 / 0.9 cm | 52 | 3 |
| G-lat-87100 | 0 / 64 | 0.000–0.056 | 24 | 0.2 / 3.0 cm | 56 | 0 |
| G-pose-87100 | 0 / 64 | 0.000–0.056 | 3 | 0.1 / 0.8 cm | 26 | 6 |
| ik (ceiling) | 50 / 64 | 0.660–0.875 | 62 | 6.6 cm (median) | 64 | 2 |
| ik-nopress | 50 / 64 | 0.660–0.875 | 61 | 6.6 cm (median) | 64 | 4 |
| hold | 0 / 64 | 0.000–0.056 | 0 | 0 | 1 | 0 |
| random | 0 / 64 | 0.000–0.056 | 0 | 0 / 0.7 cm | 2 | 0 |

Subgoal switching (§5.3): the world-model arms left the hover subgoal by its 100-command timeout on
64 / 64 resets each; only P (2) and G (1) ever left a subgoal (pre-grasp) by "reached". ik left
hover and pre-grasp by "reached" on 64 / 64 and the grasp subgoal on 30. ik by object: cube 29 / 32,
can 17 / 18, apple 4 / 13, banana 0 / 1. ik and ik-nopress differ on 2 resets (1 each way).

Why grasp episodes stopped: G 1 "projection infeasible" and 2 measured-velocity stops, G-pose 2
and 4, ik 0 and 2, ik-nopress 0 and 4.

Planning: about 0.95 s (P) and 1.0 s (G) per command with 6 processes sharing the GPU.

## 3. What the numbers say, and what they do not

- **Reach works from images alone, at the plan's example bar.** Plain LeWM (P), trained once on
  task-agnostic play and never on reaching, brought the palm within 5 cm of new goal poses, given
  as an image, on 53 of 64 fresh resets (and 55 and 54 with its two other model seeds). The margin
  over the bar is one reset. At 3 cm it reached 38 / 64, and it largely ignored the goal's palm
  orientation (median final error 0.84 rad, about 48°, against 1.07 for hold and 0.16 for G): the
  latent cost places the palm and turns it far less than G does.
- **G reaches better than P** on the same resets (62 against 53; 9 discordant resets, all in G's
  favour, p = 0.004), and its two other seeds reach 63 / 64. Most of the gain is not the pose term:
  G with the image cost alone (G-lat) reached 57, G with the pose cost alone (G-pose) 54. So the
  grounded model's latent is also a better reach cost than P's (57 vs 53, not tested), and the two
  terms together do best. This is the first closed-loop comparison of TASK-087's models; TASK-087's
  P2-FAIL (offline, one-step sensitivity) stands as recorded.
- **Grasp-and-lift fails for every world-model arm (0 / 64 each).** The arms bring the palm close to
  the subgoal poses (within 1 cm at some point on 41–56 of 64 resets, except G-pose with 26), but
  never meet the switch's orientation and hand-joint conditions, so every subgoal ends by timeout,
  and the hand closes on the object on only 3–24 resets and never lifts it (maximum rise 3 cm).
  The non-learned follower, given the same goal poses, lifts 50 / 64, so the goals are attainable;
  the planners do not produce the closing-and-lifting motion. Why is not tested here; candidates
  are the short horizon (8 commands) against a grasp that needs the hand closed for many steps
  before the lift, a cost that does not separate closed-hand-at-object from open-hand-at-object
  in the latent, and the corpus (only about 10 % of `play-v1` episodes lift anything, TASK-085).
- **"LeWM needed" is not shown** for either task: the hand-written follower does at least as well
  on reach (64 / 64) and far better on grasp.
- The plan's stop rule does not fire (P and G reach far above 50 %). Phase 4 (pick-and-place with
  subgoals) is not admitted by this row; Z3-REACH escalates (§6).

## 4. Run

- At `028e6f3` (the frozen revision, clean tree), after the independent GO on #193; NVIDIA EGL on
  the RTX 5080; `scripts/run_task088_stage.sh gated`, one `gpu_run.sh` lock per run (22 jobs:
  λ, two goal sets, 20 runs), 2026-10-10 09:59–18:36 UTC; no refusal, error or re-run. λ
  recomputed 2.0552. Goals: reach 64 (16 rejected draws), grasp 64 (57 rejected draws).
- Two replay MP4s were rendered from the stored commands after their runs (each replay reproduced
  the run's palm distances exactly): a P-87100 reach on reset 880022 (success; chosen as the success
  whose dwell completed nearest the median, at command 38) and a P-87100 grasp attempt on reset
  881023 (**failure**; chosen as the episode with the median closest palm approach, 0.6 cm). They
  ran under their own GPU locks between two gated runs.
- Evidence: `~/develop/emai/evidence/task088-gated/` (`SHA256SUMS` sha256 `cdb3a39b…900f`;
  `summary-gated.json` sha256 `b18af4db…4a33`; `videos/task088-reach-P-87100-880022-success.mp4`,
  `videos/task088-grasp-P-87100-881023-failure.mp4`). Earlier stages:
  `~/develop/emai/evidence/task088-stage0/`, `task088-dev/`.

## 5. Caveats

- Simulation only, one scene (`play-v1`'s), the right arm and hand, one onboard camera at 112 px,
  one frozen encoder and latent, one corpus, one planner setting. The gated row rests on one model
  seed per arm (87100); 87101 and 87102 were run on reach only and are reported.
- The goals are the demonstrator's own reached states on the same reset, so every goal is
  reachable and the grasp cohort is biased to layouts a scripted pick can do (57 rejected grasp
  draws for 64 goals). The goal pose gives the palm target; for grasp it carries where the object
  is.
- "LeWM needed" is not shown: the scripted IK follower, with the same goal information and no
  world model, reached 64 / 64 and lifted 50 / 64.
- Reach success is position within 5 cm held for 10 commands; P's final orientation error was
  large (median 0.84 rad), and at 3 cm P reached 38 / 64.
- The reach pass margin is one reset (53 against 52); at a true rate equal to the bar the chance
  of meeting it is about one half.

## 6. What changes

- **R25.26 — the row is Z3-REACH** (P 53 / 64 reach, 0 / 64 grasp; G 62 / 64 and 0 / 64). By §8
  it escalates to a ruling; the plan's stop rule does not apply.
- **R25.27 — the ruling on the escalation** (decided by Claude under owner delegation): Phase 4
  (pick-and-place with subgoals) is **not** preregistered on this result, because grasp-and-lift,
  its first step, failed for every world-model arm. The next task is a grasp-focused one, to be
  preregistered separately; recommendation only: diagnose offline first (does P's or G's predicted
  latent or state separate a closed-and-lifting hand from an open hand at the object over the
  planning horizon?), then a closed-loop grasp test with a change of a different kind
  (longer horizon or a grasp-phase cost, or more grasp data in the play corpus). G (not P) is the
  better reach planner here and should be the default model of that task unless a ruling says
  otherwise.
- **R25.28 — R7 does not change.** This is not Apple→Plate.
- **R25.29 — TASK-088 closes.** Nothing is abandoned: LeWM, DINOv2, `play-v1`, CEM and the product
  goal stay open.
