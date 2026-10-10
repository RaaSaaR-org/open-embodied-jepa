# TASK-088 — Phase 3: zero-shot reach and grasp-and-lift by short-horizon planning

**Status: DRAFT** (R25.1–R25.16, decided by Claude under owner delegation). It is **frozen when
its Stage 0 PR merges**; after that merge nothing in §2–§10 changes without its own reviewed
ruling. Stage 0 may fix only what §13 lists, each by a recorded rule.
This is Phase 3 of the proposal [JEPA_ZERO_SHOT_PLAN.md](../JEPA_ZERO_SHOT_PLAN.md) ("Zero-shot
reach and grasp"), after TASK-087's P2-FAIL ([grounded_lewm_v1_results.md](grounded_lewm_v1_results.md),
R24.22–R24.23).

**This is not an Apple→Plate experiment.** It plans reaches and grasp-and-lifts of `play-v1`'s
objects, in simulation only. **R7's canonical status sentence does not change, whatever row this
task ends in** (R25.16); a change would need its own ruling.

## 1. Question

With world models trained once on task-agnostic play (`play-v1`, TASK-085) and never on these
tasks, does short-horizon CEM planning (our `embodied_jepa.planning.CEMPlanner`, which met
TASK-084's Phase 0 gate) move G1's right arm and right Dex3 to **new goals given as an image plus
the robot's own goal pose**, on fresh resets never seen in training:

- **reach:** the right palm to a goal pose, and
- **grasp-and-lift:** one of `play-v1`'s objects (apple, cube, banana, can) picked up and held,
  given as a short sequence of four image + pose subgoals (hover, pre-grasp, grasp, lifted)?

"Zero-shot" here means: no training, fine-tuning or selection on these tasks, these goals or these
resets. It does not mean new objects, a new robot or a new scene: everything is from `play-v1`'s
scene, and the goals come from a scripted demonstrator (§4).

## 2. Models (R24.23 adopted)

R24.23 recommended that Phase 3 take plain LeWM P as its primary model, may take G as a second
arm, and not gate on the uncalibrated one-step wrong / true ratio. **This task adopts that
recommendation as its ruling (R25.1, decided by Claude under owner delegation).**

- **Reused, not retrained:** TASK-087's kept checkpoints (`checkpoints/task087-run/<arm>-<seed>/`
  in the worktree `task087-run`, each loaded only if its report is `T-JOB-DONE`, its
  `model.pt` matches its recorded sha256, it was trained on TASK-087's features (feature report
  sha256 `f5648a35…7eae`) and by the current implementation of `grounded_wm_model.py`). No model is
  trained, fine-tuned or selected here: TASK-087's val selection stands, and nothing in this task
  justifies retraining (the plan's Phase 3 asks whether the Phase 2 models plan; R25.2).
- **Latent:** TASK-087's: the frozen DINOv2 ViT-S/14 tokens of the onboard 112 px frame, pooled to
  4 × 4, standardised and projected with TASK-087's train-fitted projection
  (`outputs/task087-features/projection.npz`, sha256 `c196fc48…e33f`) to `[16, 192]`.
- **P** (plain LeWM, latent and action in, latent out) is the **primary** model; **G** (+ the
  28-D right arm and hand state as a token, joint-change and inverse-dynamics losses; it also
  predicts the standardised state) is the **second arm**. The primary model seed is **87100**
  (fixed now, before any closed loop on it is read); 87101 and 87102 are run on reach as
  reported-only replications (§5.4). S, C, I and N are not used.
- The one-step wrong / true ratio is not a gate here; nothing offline is.

## 3. Scene, robot and resets

- `play_scene_v0` unchanged (TASK-085 §2): fixed-pelvis G1, the table, the static plate and the
  four movable objects with v2's contact; the onboard camera at 112 × 112; NVIDIA EGL on the
  RTX 5080 (the renderer `play-v1` was recorded with; frames are not bit-identical run to run,
  TASK-085 §14).
- **Actions:** `ee_delta_grasp_v0` (14-D) and `configs/g1_sim_action.json` (20 Hz) unchanged; the
  left arm is held (commands 0, left grasp −1), as in `play-v1`; the right arm's 6 and the right
  grasp are planned.
- **Execution, for every arm alike:** each command is projected by the embodiment's own
  backtracking (`project_candidates`: 1, 1/2, …, 1/64, 0) on the current observation and then
  executed, as `play-v1` was recorded. If the projection is infeasible or raises (the 5 rad/s
  measured-velocity stop) or the command is rejected, the episode **ends as a failure** ("stopped").
- **Reset:** `play-v1`'s reset (arm at its default pose, both hands open) and its 10 settle
  commands; the right palm then starts at about (0.318, −0.149, 0.071) m in the pelvis frame.
- **Layouts** (fresh, salt 8801 with the reset seed and an attempt index): the plate uniform in
  `play-v1`'s plate region; every object uniform in **TASK-085's Phase 3 test object region**
  (world x 0.30–0.46, y −0.28 to −0.04), with 1 cm clearance, uniform yaw.
  - reach: 1–3 of the four objects on the table (uniform count, uniform subset);
  - grasp: the **target** uniform over the four objects, plus 0–2 distractors.

## 4. Goals and what each arm may read

Goals are made by the task, not by the controllers: a **demonstrator** (privileged scripted code
that reads object truth) acts on the same reset in the simulator, and the goal is what it reached.
The demonstrator is never a counted arm.

- **Reach goal** (§4.1, salt 8802): a palm target uniform in x 0.24–0.44, y −0.30 to −0.02,
  z 0.10–0.26 m (pelvis frame; inside TASK-085's Phase 3 hand region, whose floor is raised to
  0.10 so that reach goals stay above the objects), at least 10 cm from the start palm, with the
  palm turned top-down and a yaw uniform in ±0.3 rad. The demonstrator drives e9's tracking law
  (`play-v1`'s scripted law: 1.5 cm per unit, clipped at 0.4; rotation clipped at 0.8) for up to
  120 commands until the palm is within 0.5 cm, then 10 hold commands. The draw is **rejected**
  (and the next layout and target drawn, up to 30 attempts) if it does not converge, settles more
  than 2 cm off the target, is stopped, moves any object by more than 5 mm or touches one. The goal
  is its final frame and pose.
- **Grasp goal** (§4.2, salt 8802): e9's pick recipe without noise on the target (palm aim = the
  object's centre + e9's offset (−1.5, 0, 5.2) cm, a yaw uniform in ±0.2 rad): hover 8 cm above
  the aim (60 commands), descend open (50), close to 1.0 (35), lift by 12 cm (50) and hold (20).
  The draw is rejected (next layout, up to 30 attempts) if it is stopped or if the demonstrator
  itself does not meet §6.2's lift criterion. The four **subgoals** are its frame and pose at the
  end of the hover, of the descent, of the close and of the hold.
- **The goal pose** of a (sub)goal is the robot's own state there: the 7 right-arm and 7 right-hand
  joint positions and their velocities (the 28-D `play-v1` / TASK-087 state), and the right palm's
  position and orientation in the pelvis frame as the embodiment's forward kinematics gives them
  for those joints. **It is proprioception, which a controller may read** (R25.5): it is a target
  for the robot's own body, not object truth.

**What each arm reads.** Every controller arm sees only: the current onboard frame; the robot's
measured joint state (and the palm pose by forward kinematics of the measured joints); and the
goal's image(s) and pose(s). **No controller and no planning cost reads object truth.** Object
truth from the simulator is read only by the demonstrator (to make goals) and by the scorer (§6).

**What the goal pose gives away (fixed now, R25.5).** A grasp subgoal's pose places the palm at the
object, so grasp success does not show that a planner *localised the object visually*: the
pose-following IK arm (§5.2) needs no image at all. P's planning cost reads only images, but its
subgoal switching (§5.3) reads the goal pose. The decomposition arms G-lat and G-pose (§5.1)
separate G's image and pose terms.

## 5. Arms

All arms run on exactly the same goals and resets (paired). Per reach episode at most **150**
commands (7.5 s); per grasp episode at most **300** (15 s). An episode ends at success (§6), at
"stopped" (§3) or at the budget.

### 5.1 World-model arms (CEM)

- **Planner:** `embodied_jepa.planning.CEMPlanner`, unchanged: horizon **H = 8** (TASK-087's
  training horizon), **300** samples, **10** iterations, **30** elites, minimum std 0.05, the
  left arm's dimensions fixed at 0 and its grasp at −1 by the bounds, the right arm and grasp in
  [−1, 1], no candidate projection; replanned from scratch every command (MPC), the first action
  of the best candidate executed (§3). CEM seed `8803·10⁷ + reset seed`, fixed per episode and the
  same for every arm. (TASK-084 used 300 samples, H = 6 and 30 iterations; 10 iterations here keep
  the run inside about a day of GPU time, §13.)
- **The model contract** (`zero_shot_runtime.PlannerModel`): `predict` rolls every candidate out
  from the encoded current frame (and, for G, the measured 28-D state) with the TASK-087
  predictor; `distance` returns per candidate and step
  `w_lat · latent MSE(ẑ_h, z_goal) + w_pose · λ · MSE(ŝ_h[:14], s_goal[:14])`, where ẑ is the
  predicted `[16, 192]` latent, z_goal the current subgoal's encoded frame, ŝ G's predicted
  standardised state and s_goal the subgoal's state standardised by the same train moments (the 14
  joint positions only). The planner reads the last step (h = 8).
- **λ** is fixed by a rule, not tuned: the mean latent MSE over 10 000 pairs of val frames from
  different `play-v1` val episodes (salt 8806) divided by the mean standardised joint-position MSE
  over the same pairs, so both terms have the same mean on unrelated frames. On TASK-087's val
  store it is **λ = 2.0385** (the same for every G seed, which share the train moments; recorded
  by `run_task088.py lambda`).

| Arm | Model | w_lat | w_pose | Role |
| --- | --- | --- | --- | --- |
| **P** | P-87100 | 1 | 0 | **primary** (image goal only) |
| **G** | G-87100 | 1 | 1 | **second arm** (image + pose goal) |
| G-lat | G-87100 | 1 | 0 | decomposition, reported only |
| G-pose | G-87100 | 0 | 1 | decomposition, reported only |
| P-87101, P-87102, G-87101, G-87102 | as P / G | | | replication on reach only, reported |

### 5.2 Baselines (non-learned)

- **hold:** right arm 0, right grasp −1 every command (floor).
- **random:** the right arm's six and the right grasp uniform in [−1, 1], fresh every command
  (salt 8804) (floor).
- **ik — the scripted IK follower:** e9's tracking law to the current subgoal's palm position and
  orientation (the goal pose); grasp +1 when the synergy value nearest the subgoal's 7 hand joints
  is above 0, else −1, ramped by 0.2 a command; on the grasp subgoal it commands the palm 5 cm
  below the goal palm (e9's press, §12). It reads proprioception and the goal pose only, no image
  and no object truth. It is the **calibration ceiling** (§7): a hand-written controller with the
  same goal information, not a learned result.

### 5.3 Subgoal switching (grasp; every arm alike)

The controller works on subgoal k (from 0) and moves to k + 1 when, for **3 consecutive
commands**, the measured palm (forward kinematics of the measured joints) is within **1 cm** and
**0.17 rad** of subgoal k's palm pose and the 7 measured hand joints are within **0.15 rad RMS**
of subgoal k's hand joints; or after **100** commands on subgoal k. The last subgoal (lifted) is
kept to the end. The rule reads proprioception and the goal pose only.

### 5.4 Runs

Gated cohorts (§8): reach — P, G, G-lat, G-pose, P-87101, P-87102, G-87101, G-87102, ik, hold,
random; grasp — P, G, G-lat, G-pose, ik, hold, random.

## 6. Success (scorer: simulator truth, after every executed command)

- **6.1 Reach:** success when the right palm (`right_ee` site, pelvis frame) has been within
  **τ = 5 cm** of the goal palm for **10 consecutive commands** (0.5 s). A shorter crossing never
  counts (TASK-067's lesson: no latching of transient crossings). Reported: success at 3 and 8 cm
  with the same dwell, the final and minimum distance.
- **6.2 Grasp-and-lift:** success when the target's centre has been at least **5 cm above its
  start height** *and* in grasp contact with the right hand (thumb plus index or middle, as
  `play-v1`'s sidecar defines it) for **20 consecutive commands** (1 s). Reported: maximum rise,
  any grasp contact, the subgoals reached and how (reached or timeout).
- τ = 5 cm, the dwells and the 5 cm lift were set in code before any world-model closed loop was
  run (§12).

## 7. Calibration K0 and the bars (Stage 0, before the freeze)

TASK-074's lesson: bars must be achievable by something given the same information. **K0** runs
ik, hold and random on 32 + 32 **calibration resets** (reach 88000–88031, grasp 88100–88131;
never used again) before the freeze. For each task:

**bar = min(the plan's example bar, 0.9 × ik's K0 success rate)**; the plan's bars are reach 0.80
and grasp-and-lift 0.40. The bar counts on the gated cohort are **⌈bar × 64⌉**, written into §8 by
the Stage 0 record. If ik's K0 rate is below 0.5 on a task, that task's bar is not set and the
task escalates to a ruling before the freeze (the task, not the model, would be in doubt).

## 8. Stages, cohorts and rows

- **Stage D — development, not gated** (after the freeze): 16 reach (88200–88215) and 16 grasp
  (88300–88315) resets, P, G, G-lat, G-pose (87100), ik, hold and random. Its numbers are read and
  recorded; it can show a defect (a defect fix needs its own ruling before S) but changes no bar
  or knob of §2–§7 by itself.
- **GO:** an independent reviewer reads Stage D's record and posts **GO** (or NO-GO) as a PR
  comment before Stage S starts.
- **Stage S — gated:** **64 reach (880000–880063) and 64 grasp (881000–881063)** fresh resets,
  every run of §5.4, at the frozen revision from a clean tree. **Power:** at 64 resets a true rate
  of 0.90 meets an 0.80 bar (52 / 64) with probability about 0.99, and a true rate of 0.55 meets a
  0.40 bar (26 / 64) with probability about 0.99; at a true rate equal to the bar it is about one
  half.
- **Rows** (first match), on the primary arm **P** (model seed 87100); k = P's successes:
  1. **Z3-VOID** — a run did not complete under the rules, a goal, checkpoint, projection or seed
     check failed, or a cohort was run twice.
  2. **Z3-PASS** — reach k ≥ ⌈bar_reach × 64⌉ **and** grasp k ≥ ⌈bar_grasp × 64⌉.
  3. **Z3-REACH** — reach meets its bar, grasp does not.
  4. **Z3-LOW** — reach below its bar, but P or G reaches at least 32 / 64 (50 %).
  5. **Z3-STOP-CANDIDATE** — P and G both reach fewer than 32 / 64.
- **The second arm G** is reported against the same bars ("G meets the reach / grasp bar"); it does
  not change the row. G against P: paired exact two-sided McNemar test, reported.
- **What each row leads to (the plan's stop rule):** Z3-PASS — Phase 4 may be preregistered.
  Z3-REACH, Z3-LOW — escalate to a ruling. **Z3-STOP-CANDIDATE** — JEPA planning gets **one fix**
  (its own preregistration and fresh resets); if reach is still under 50 % after it, **JEPA
  planning stops** as the plan's primary line. Nothing else closes: LeWM, DINOv2, `play-v1` and
  the product goal are not abandoned by any row.
- Intervals: exact (Clopper–Pearson) 95 % for every count.

## 9. Seeds and salts

Calibration 88000–88031 (reach) and 88100–88131 (grasp); development 88200–88215 and 88300–88315;
gated 880000–880063 and 881000–881063; debug 88900–88999 (88900–88949 reach, 88950–88999 grasp;
plumbing only, never counted). Salts: 8801 layouts, 8802 goals, 8803 CEM, 8804 random, 8805
reserved (bootstrap), 8806 λ's pairs. Checked free in `src/`, `scripts/`, `docs/` and `.mc/` on
every branch of the repository (`git grep` over all refs; only `range(87900, 88000)`'s end bound
matched).

## 10. Reported only

Per arm and task: successes with exact intervals; reach success at 3 / 5 / 8 cm; final and minimum
distance; grasp maximum rise, grasp contact, subgoals reached / timed out; stopped episodes and
why; planning seconds per command; per-object grasp results; the demonstrator's rejection counts
per task and object (the accepted resets are those a scripted pick could do, which biases the
grasp cohort towards feasible layouts); the replication seeds; ik, hold and random; paired
McNemar tests of P and G against each other and against random. One MP4 of a representative P
or G reach and one grasp attempt (success or failure, labelled), replayed from the stored
commands.

## 11. What this cannot show

One scene, one robot arm and hand, one camera at 112 px, simulation only, one encoder and latent,
one corpus, one model seed gated (two more on reach, reported), one planner setting. The goals are
made by a scripted demonstrator, so they are reachable by construction and the grasp cohort is
biased to layouts a scripted pick can do. The goal pose carries the object's location for grasp
(§4). A pass would show that these Phase 2 models plan short steps to such goals; it would not
show "JEPA needed" (the ik follower uses the same goal information and no world model), and it is
not Apple→Plate or pick-and-place.

## 12. Disclosed development before this draft

On debug seeds 88900–88999 (never counted; `outputs/task088-debug` in the Stage 0 worktree),
while writing the code:

- **Goals:** reach goals were accepted on 50 / 50 debug seeds with 11 rejected draws; grasp goals
  on 50 / 50 with 32 rejected draws (mostly "demonstrator did not lift").
- **The IK follower** first scored 0 / 16 on debug grasp. Two causes were found and fixed before
  this draft: (1) switching subgoals before the palm had turned (the switch now also needs the
  orientation within 0.17 rad, and the palm within 1 cm instead of 2 cm), and (2) **e9's press**:
  on 40 debug layouts, e9's recipe lifted on 26 with its palm commanded below where the table stops
  it while closing, and on 7 when the palm target was frozen at the reached palm while closing; so
  this hand grasps only while pressed down, which a target equal to the reached pose does not do.
  The ik follower therefore commands its palm 5 cm below the goal palm on the grasp subgoal. A
  fourth subgoal (pre-grasp, open) was added so that a follower does not close while descending.
  After these changes ik scored 34 / 50 on debug grasp and 16 / 16 on debug reach; hold and random
  0 / 16 on both.
- **World-model plumbing:** P-87100 ran 4 debug reach episodes (3 / 4 met the 5 cm criterion;
  minimum distances 2.5–4.9 cm; about 0.36–0.64 s of planning per command with 4 processes) and
  G-87100 8 debug grasp episodes (0 / 8; no hover subgoal reached within its 100 commands; about
  1.4 s per command with 8 processes sharing the GPU). τ, the dwells and the lift height had been
  written before these runs and were not changed after them. These are plumbing numbers, not
  results.
- One short prototype render (about a second of EGL use) ran before the GPU lock was used; every
  later render and CUDA use ran under `scripts/gpu_run.sh`.

## 13. Stage 0, freeze, runtime and evidence

- **Stage 0** (its own PR, before the freeze): the opt-in modules `zero_shot.py` (NumPy at import)
  and `zero_shot_runtime.py` (torch), `scripts/run_task088.py` (goals, λ, runs, video) and a
  summary script; tests (layouts inside the regions, the dwell scorers do not latch transient
  crossings, the switch rule, the controllers read no object truth, the bar rule and the rows, the
  exact intervals, the cohort seed checks, the planner contract's shapes and costs, the
  core-import check); λ; a debug smoke of every arm; **K0** (§7) and the bars it gives. Stage 0
  may change only the CEM iteration count (10 → 5 if the smoke projects Stage S above 30 GPU hours)
  and the per-run process count, each recorded.
- **Frozen** at the Stage 0 merge; D and S run at the merged revision from a clean tree, each run
  one `scripts/gpu_run.sh --wait --min-free-gib 8 --board` job.
- **Runtime estimate** (from the debug runs): about 8 planned commands a second over all
  processes; Stage S about 4–5 GPU hours, Stage D about 1.5.
- **Disk:** about 16 GB free; nothing is written below 10 GiB free (the runner refuses). Outputs
  in `outputs/task088-*` of the run worktree; evidence in `~/develop/emai/evidence/task088-*/`
  with `SHA256SUMS`. The TASK-087 checkpoints and features are read, never written.

## 14. Rulings this document records (R25.1–R25.16, DRAFT; decided by Claude under owner delegation)

- **R25.1** — R24.23 is adopted: P is the primary model, G the second arm; the one-step wrong /
  true ratio is not a gate.
- **R25.2** — TASK-087's checkpoints are reused unchanged; nothing is retrained or selected here.
- **R25.3** — tasks: reach and grasp-and-lift of `play-v1`'s objects, on fresh layouts in
  TASK-085's test object region (§3).
- **R25.4** — goals are made by a privileged scripted demonstrator on the same reset; grasp goals
  are four image + pose subgoals (§4).
- **R25.5** — the goal pose (the robot's own joint state and palm pose at the goal) is
  proprioception and may be read by controllers and costs; object truth never is (§4).
- **R25.6** — the planner, its budget and the cost as §5.1; λ by its rule.
- **R25.7** — the baselines hold, random and the ik follower (with e9's press) as §5.2.
- **R25.8** — subgoal switching as §5.3.
- **R25.9** — success as §6 (5 cm with 10 commands' dwell; 5 cm lift with grasp contact for 20).
- **R25.10** — K0 and the bar rule as §7.
- **R25.11** — Stage D (development), an independent GO, then Stage S (64 + 64) as §8.
- **R25.12** — rows and the stop rule as §8, on P-87100; G reported against the same bars.
- **R25.13** — seeds and salts as §9.
- **R25.14** — the reported-only quantities of §10, including the two MP4s.
- **R25.15** — Stage 0, freeze, runtime and evidence as §13.
- **R25.16** — R7 does not change, whatever the row.
