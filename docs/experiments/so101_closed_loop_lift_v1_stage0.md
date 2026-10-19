# TASK-101 Stage 0: code, checks and K, stopped at S0-VOID before D's closed loop (record)

**Status: Stage 0 record. Row: S0-VOID, escalated for a ruling. The protocol is not frozen.**
Protocol: [so101_closed_loop_lift_v1.md](so101_closed_loop_lift_v1.md) §7. Rulings R28.59–R28.64
in [DECISIONS.md](../DECISIONS.md), decision 2026-10-19 (bu), **decided by Claude under owner
delegation**. **Nothing here is a result.** Every closed loop ran on calibration seeds K
(101010–101041) or debug seeds (101900–101999). D's (101100–101163) goal demonstrations and the
first 8 episodes of its CPU arms ran; **W's closed loop on D did not run, and no D count of a CEM arm
exists.** No gated seed was drawn, rendered or run. This is a simulation-only SO-101 study, not
Apple→Plate. R7 does not change (R28.9).

## Summary

- **Check 7.2.2 failed: the live latent does not read the cube on the lift task's frames.** TASK-100's
  pretrained R-enc readout, applied to the live-encoded subgoal frames of D's 64 goal
  demonstrations (256 frames), reads the cube at **6.106 cm in median**, against
  B_cube = 2.397 cm. The protocol says a miss "means a pipeline or rendering mismatch, not a model
  result", and the cause is a rendering mismatch of the scene. `so101_cube_lift_v0` puts its
  (inert) blue target marker at a fixed x = 0.30 m, y = −0.20 m. The play corpus J was trained on
  (`so101-play-v1`) drew the marker inside x ∈ [0.16, 0.22] m, |y| ≤ 0.14 m on every episode. Re-rendered with
  **only the marker moved** into the corpus's box, the same 256 states read at **1.048 cm**
  (diagnostic `check marker`). Re-rendered as declared, they reproduce the goal store's latents
  256/256 and read 6.106 cm again. The readout error is mostly a constant bias (+2.8 cm in x,
  −5.3 cm in y). So the frames the controller would see in this task lie outside the
  distribution TASK-100's model and readout were fitted on, in one scene element.
- **This is S0-VOID (§7.6 row 1). It cannot be fixed in code without changing the declared scene or
  observation** (§2: the scene and cameras are unchanged; R28.43). A fix is a ruling on a
  condition change, so the row is escalated and Stage 0 stops here.
- **K ran before the cause was found, and every configuration scored 0/32** (H8-last, H8-mean,
  H16-last and H16-mean; 128 episodes; exact 95 % interval 0.000–0.109 each). No episode reached a
  single subgoal by the switching tolerance: all 3 × 128 switches were timeouts. On H8-mean the
  median closest arm RMS to subgoal 1 was 0.41 rad, against 0.04 required. By §7.3's tie rule the
  configuration would be H8-mean, with K "null". **K ran under the void condition, so its 0/128 is
  not attributable to J or to the planner**; the marker shift may account for some or all of it.
  K's counts have been read, so any rerun after a fix uses K′ (§7.6 row 1).
- **Every other check passed**, three of them after a fix made under §7.6 row 1 before any count was
  read (items 1–3 of the deviations below): the live latent equals the stored latent bit for bit
  (64/64 frames), the adapter equals TASK-100's `predict_features` + latent error, N's costs tie
  exactly on CUDA, the projector equals `preview` and `execute` (1 000/1 000), determinism holds in
  the conditional form, and every command finished within 1.6 s (the deadline is 30 s).
- **Measured follower check:** the goal demonstration was kept on 32/32 drawn K seeds and 64/64
  drawn D seeds (no rejection; S0-CEILING would need 8). F replayed the demonstration exactly on
  every episode it ran.
- **GPU: 1.66 GPU h of Stage 0's 6 GPU h** (27 lock-holding jobs, including the diagnostics). K's
  measured throughput projects Stage S at about 5.9 GPU h at H = 8 or 6.9 at H = 16 for all six CEM
  arms, which fits the remaining 18.3 GPU h. That projection assumes every CEM episode runs its full
  200 commands, as all of K's did. It is recorded, not used by any row, because the row is S0-VOID.

## 1. Code (new files only)

| file | what |
|---|---|
| `src/embodied_jepa/so101_closed_loop.py` | NumPy: cohorts and seed guards, salts, the switching rule, the derangement, the vectorised chained-`preview` projector, H-pose's law, the goal store (hashed), §4's replacement, exact intervals, McNemar, the reset bootstrap, the bar rule and power, K's configuration rule, the Stage 0 and gated rows, the S projection |
| `src/embodied_jepa/so101_closed_loop_runtime.py` | torch/MuJoCo, lazily: the live g4 latent (TASK-100's featuriser, fit moments and weight pins checked), the checkpoint loader (file and implementation hashes), the cost adapter (last / step-mean; N's broadcast roll-out; W-mean's zero latent), the projector, the controllers, the truth sentinel, the goal demonstration, the episode loop, F, the worker pool |
| `scripts/run_task101.py` | `goals`, `episodes`, `check` (`latent`, `adapter`, `ties`, `determinism`, `render`, `renc`, `projector`, `marker`) |
| `scripts/summarize_task101.py` | Stage 0 summary: counts, intervals, paired tests, decomposition, bar, power, timing, GPU hours, projection, row |
| `scripts/run_task101_stage.sh` | every Stage 0 and gated job, one `gpu_run.sh --wait --board` lock per GPU job |
| `tests/test_so101_closed_loop.py` | 34 tests, CPU, tiny synthetic models (below) |
| `benchmarks/manifests/so101-closed-loop-lift-v1.json` | pins of the files above and of the composed files; status **CANDIDATE** (not frozen) |

No existing file changed (`git diff --name-status origin/main` lists additions, this record and the
documents). `config.py`, `schema_planning.py`, `planning.py`, `constraints.py`, `so101.py`,
`so101_simulation.py`, `so101_tasks.py`, `so101_followers.py`, every TASK-100 pinned file
(`frozen_files_check` passes) and every G1 byte-pinned file keep their bytes. A test checks the
named SO-101 files against their `origin/main` hashes.

**Tests:** the cohorts and their disjointness, seeds outside a cohort refused, the planner and rand
seeds; the switching rule in both directions (three consecutive near commands, the arm and jaw
tolerances, the 60/45/30 timeouts, one switch per command, the single-goal variant); the
derangement (no fixed point, fixed by its salt); H-pose's law; exact intervals and McNemar; the bar
rule reproducing §7.5's table; the bootstrap and the power simulation fixed by their salts; K's
configuration rule (ties, eligibility, all-zero); every Stage 0 row and every gated row in both
directions; the S projection dropping replication first; goal stores refused on a hash mismatch
and never overwritten; §4's replacement from the reserve; the demonstration verdict; a cohort not
run twice into one output, and S refused without `--gated`; the core-import check; TASK-100's pins;
**the vectorised projector bit-identical to chained `preview` and to `execute`'s applied actions**
(including joint-limit clips and gripper rate limits); **F's executed actions equal to the goal
demonstration's**, and a diverging demonstration refused; **the truth sentinel** (a controller
reaching `task_truth` or `gripper_contacts` raises); the cost adapter against `predict_features`
and the mean wrapper; **N's exact ties keep the planner's initial mean** (the requested zero
command, executed as its projection); checkpoints refused on a hash mismatch; the determinism-pair
reader.

## 2. Checks (§7.2)

| check | revision | outcome |
|---|---|---|
| 7.2.1 live latent = stored latent | `a7332ce` | **pass**: 64 frames per camera (8 val episodes × 8 frames), **64/64 bit-identical** (0 ulps), after the fix of deviation 1 |
| 7.2.2 the live latent reads the cube | `a7ea306` | **FAIL**: R-enc median 6.106 cm on D's 256 subgoal frames (per subgoal 6.14, 5.91, 5.91, 6.29 cm) against B_cube 2.397 cm; S0-VOID |
| 7.2.2 diagnostic (`check marker`, reported) | `42117c4` | declared scene 6.106 cm; only the marker moved into the corpus's box (x 0.16–0.22, \|y\| ≤ 0.14, ≥ 8 cm from the cube; rng 0) **1.048 cm**; declared re-renders equal the goal store's latents 256/256 |
| 7.2.3 projector | `a7ea306` | **pass**: 1 000 random 16-step sequences (every 4th saturated) from D's reset states: vectorised = chained `preview` 1 000/1 000, = `execute`'s applied 1 000/1 000, 0 stopped |
| ties (§6, §7.1) on CUDA, N_J-100101 | `a7332ce` | **pass** at H8/H16 × last/mean: costs bit-identical across 256 candidates, the requested plan is the initial mean, the executed action its projection; the un-broadcast batched roll-out's rows are bit-identical too |
| adapter vs `predict_features` (reported) | `a7332ce` | max relative difference 4.4 × 10⁻¹⁶, identical rankings |
| 7.2.4 determinism (conditional form, deviation 3) | `6048169` | **pass**: W and N reruns bit-identical; pool 1 vs pool 6 bit-identical on 3 of 4 episodes; on 101903 the live latent first differs at step 105 (10 steps differ), the executed actions not at all, the same outcome |
| renderer stability (reported, `check render`) | `a7ea306` | 1 288 states × 2 cameras, rendered 3 times each: 12 wrist and 1 overview states gave different images (up to 19 pixels); follower states 0 of 488 |
| 7.2.5 timing (K, pool of 6) | `a7ea306` | planning s per command, median / 95th percentile: H8-last 0.233 / 0.295, H8-mean 0.231 / 0.294, H16-last 0.390 / 0.461, H16-mean 0.391 / 0.463; longest whole command (observe, encode, plan, execute) 1.50 s against the 30 s deadline |

## 3. K (§7.3; ran under the void condition)

Goal store `d6a891a3…5c255c`: 32 kept of 32 drawn (no rejection). W (J-100101), pool of 6 workers,
one fresh process per episode, 1.07 h of lock time for 128 episodes.

| configuration | W | exact 95 % | subgoal switches (1→2, 2→3, 3→4) | any gripper contact | both jaws | max rise median / max | commands clipped |
|---|---|---|---|---|---|---|---|
| H8-last | 0/32 | 0.000–0.109 | 32/32/32 timeouts | 29 | 1 | 0.27 / 2.19 cm | 64.7 % |
| H8-mean | 0/32 | 0.000–0.109 | 32/32/32 timeouts | 13 | 3 | −0.01 / 5.13 cm | 73.0 % |
| H16-last | 0/32 | 0.000–0.109 | 32/32/32 timeouts | 26 | 1 | 0.36 / 1.66 cm | 54.6 % |
| H16-mean | 0/32 | 0.000–0.109 | 32/32/32 timeouts | 18 | 5 | 0.28 / 7.15 cm | 67.0 % |

No episode stopped. "Clipped" counts commands whose applied action differs from the requested one
(mostly the gripper's 0.25 rad rate limit); every executed action equalled the projected action the
model scored (the guard of deviation 6). By §7.3 the configuration would be **H8-mean** (all tied at
0, mean over last, then H = 8), and S0-NULL's condition holds on these numbers. **Neither is a
finding**, because the row is S0-VOID, which comes first, and K ran under the void condition.

## 4. D (§7.4; stopped before the closed loop)

Goal store `d98a3b89…a1f5eed0`: **64 kept of 64 drawn, no rejection** (S0-CEILING needs ≥ 8). The goal
job's evidence episodes on D's first 8 slots read **F 8/8 (each equal to the demonstration),
H-pose 8/8, rand 0/8, hold 0/8**. These arms read no image, so the scene finding does not touch
them. They are disclosed because they are D counts that have now been read. **W, N, W-shuf, W-mean
and the single-goal diagnostic did not run on D**, so no D count of a CEM arm has been read. The bar
(§7.5) needs W(D) and is not computed. The rerun rule (§7.6 row 1) therefore lets a fix rerun D's
CEM arms on D's seeds; whether the eight CPU-arm counts read on D affect that is left to the ruling
(R28.63).

## 5. Deviations from the DRAFT (disclosed; decided by Claude under owner delegation, R28.60–R28.62)

1. **The live latent uses TASK-100's fixed featuriser batch of 64 (§3, §7.2.1).** The first run of
   7.2.1 (`check-latent-void1`, revision `5b2c01d`) encoded one frame per camera per call and failed.
   98.4 % of values matched, but the worst differed from the stored float16 by 221 float16 ulps.
   The large ulp counts were all near zero: the largest absolute difference was 0.004 against a
   raw-value standard deviation of 1.9, every value of magnitude ≥ 0.25 was within 1 ulp, and the
   latent error was below 10⁻⁶. In TASK-100's own batches of 64 the same frames reproduce the store
   bit for bit. The difference is GPU arithmetic that depends on the batch size, not a pipeline
   mismatch. TASK-100's `Featuriser` runs "a fixed batch of 64 frames per camera". The live path now
   pads every call to whole batches of 64 (the frame repeated) and keeps the first rows. The check's
   frames are 8 per episode, evenly spaced over the frames the store encoded in full batches (0–383).
   TASK-100 encoded each episode's last 17 frames in a batch of 17, and frame 400 was the only miss
   of a padded diagnostic. Cost: about 0.13 s of DINOv2 per command instead of 0.005 s.
2. **One fresh worker process per episode (§13).** The first determinism check
   (`check-determinism-void1`, `a7332ce`) found pool 1 and pool 6 diverging on one of four episodes.
   The pool now uses `maxtasksperchild=1` for every pool size, 1 included. That did not remove the
   divergence (item 3), but it is kept: it costs little and removes process history as a variable.
3. **Check 7.2.4 is read in a conditional form. This changes the check, not only the code.** With
   fresh processes the divergence remained (`check-determinism-void2`, `ae70d1e`). The physics state
   (joints, velocities, targets, cube pose, fingertip site) was bit-identical up to the divergence,
   but the rendered frames were not. **On this RTX 5080, MuJoCo's EGL renderer returns 2–3 distinct
   images when one physics state is rendered 30 times in one process**, on some states (one pixel to
   19 pixels, one intensity level), and one image on others. Short of changing the observation (for
   example turning off multisampling, which would move the frames away from the corpus's rendering),
   no change makes such frames bit-stable, so identical executed sequences across reruns and pool
   sizes cannot be guaranteed. The committed check (`6048169`) reports the literal form and passes on
   the conditional one: **in every pair the executed actions never diverge before the live latent
   (the rendered frames) does, and with identical latents at every step the actions and the outcome
   are identical.** In other words, the controller and the physics are deterministic, and the renderer
   is the only nondeterministic input. A rerun of an episode can differ after an unstable frame. S
   runs every arm once and no row reads a rerun. F reads no frames and must still replay the
   demonstration exactly.
4. **`SchemaCandidateProjection`.** `constraints.CandidateProjection` validates actions against the
   14-D `ee_delta_grasp_v0` only. The projector returns an identical value object validated against
   `joint_delta_so101_v0` (new code; `constraints.py` unchanged).
5. **The projector is a vectorised transcription of `preview`**, not a Python loop calling it once
   per candidate step. Tests and check 7.2.3 hold it bit-identical to chained `preview` and to
   `execute`.
6. **W sends the best candidate's requested first action, and a guard requires `execute`'s applied
   action to equal the projected first action the model scored, bit for bit.** `planning.MPC`
   sends the projected action instead. Because `execute` applies the same `preview`, sending the
   requested action makes the executed action exactly the scored one. The guard never fired in
   K's 25 600 commands.

The plumbing outputs that predate these fixes are kept and are not results:
`debug-goals-a`, `debug-ep-a-crashed`, `debug-ep-b`, `debug-goals-batch1`, `check-*-void*`,
`check-render-follower-only` and `diag-pool`.

## 6. The escalation (R28.63, recommendation only; the owner rules)

The VOID's cause is a scene element: the lift task's inert target marker sits where the corpus never
put it. Options for the ruling:

- **(a) Recommended: a declared scene amendment for TASK-101.** In the lift scene, put the marker
  inside the play corpus's marker box, drawn per reset by the corpus's rule (at least 8 cm from the
  cube) from a new salt, or fixed at one declared in-box point. The marker has no contact and plays
  no role in the lift scorer. The amendment must be preregistered and named in every restatement
  ("the lift scene with its target marker inside the corpus's range"). Then:
  - Re-run 7.2.2 on regenerated D goals. Pass bar unchanged (the diagnostic read 1.048 cm).
  - Run K on **K′** (K's counts are read).
  - Run D on D's seeds. W's D counts are unread; the ruling decides whether the eight read CPU-arm
    counts force D′.
  - Every check that reads frames is re-run.
- **(b) Remove the marker from the lift scene.** This is also outside the corpus's distribution
  (every corpus episode shows a marker), so it is not recommended without its own check.
- **(c) Close TASK-101 as INCONCLUSIVE** and carry the lesson into a redesign: a task's scene must be
  checked against the training corpus's scene before Stage 0.

**Lesson (for RESUME.md's list): check that the closed-loop task renders frames from the training
corpus's distribution before Stage 0. Here an inert scene element was enough to move a linear
readout by 5 cm.**

## 7. Evidence

Outputs: `outputs/task101-stage0/` (shared checkout, HDD). Evidence
`~/develop/emai/evidence/task101-stage0/` (832 KB: every report, log and progress file, the
summary, the GPU-lock log lines and the SHA-256 of all 212 episode and goal `.npz` files),
**`SHA256SUMS` `9ee6638dc23dae5d660119f639e8a8a50a486979d10f3269b12ba261e1e6ea0b`**. Summary
`stage0_summary.json` `c93251d5…a8bc7`. Goal stores: debug `a4dec9f0…`, K `d6a891a3…`, D
`d98a3b89…`. TASK-100's checkpoints were read only, and nothing under `data/`, `checkpoints/` or an
earlier `outputs/` path was written.
