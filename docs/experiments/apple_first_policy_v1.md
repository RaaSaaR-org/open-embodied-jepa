# Apple→Plate first learned policy v1: can an information-matched learned policy reach a first success on the development cohort? (TASK-067)

**Status: preregistration.** No policy has been trained, no readout has been fitted, and nothing
has been simulated for this task. The design comes from the proposal
[`control_formulation_proposal_v1.md`](control_formulation_proposal_v1.md) (PR #77), as the
owner accepted it on 2026-09-27 (§0). What it adds is the frozen numbers, rows and guards.

**Status lines.**
- **Learned Apple→Plate is still 0 successes.**
- The test split of `apple-look-v1` is never decoded, and the `apple-wide-v1` test split is not
  touched.
- Cohort D (45000–45007, 45100–45107) is the development cohort; it never gates. Cohort C
  (45300–45339) is not opened by this protocol's gated run. M2 (§11) needs a separate
  authorization.
- `exemption_spent` (`benchmarks/manifests/apple-policy-diagnostics-v1.json`,
  `precedence_rule_D1_over_G_SUB`) reads `false` at this preregistration, and this protocol does
  not claim the exemption.
- **No compute before TASK-066 ends.** The gated run, and every calibration or training step,
  starts only after TASK-066's gated run has finished (the owner's schedule: GPU hold until about
  2026-09-28T00:10Z, hard cap 04:08Z) and only on the pre-run reviewer's reported GO (§15).

> **Disclosed prominently, by owner ruling R1: the camera and its resolution are unchanged.** This
> task uses the same 112 px `onboard_rgb` camera as TASK-056/057. What changed is the
> **information** the controller receives:
> - a different corpus, `apple-look-v1`;
> - a different observation protocol, the fixed look prefix (TASK-061);
> - a different encoder, the frozen pretrained DINOv2, which reads the post-look apple offline
>   (TASK-063/064);
> - the robot's own step counter as an input.
>
> This is an information change, not a retry of the TASK-056 formulation.

Manifest: `benchmarks/manifests/apple-first-policy-v1.json`. Code that fixes the design:
`src/embodied_jepa/first_policy.py`, which holds the seeds, arms, labels, schedule, calibration
rule, gates, rows and caps. A test, `tests/test_first_policy.py`, pins the manifest's `frozen`
block to it.

---

## 0. Owner rulings (verbatim, received 2026-09-27T12:43Z via the coordinator)

> R1 (clause scope): allowed. The TASK-057 clause stops BC on `apple-wide-v1` + 112 px onboard.
> TASK-067 uses a different corpus (`apple-look-v1`), a different observation protocol (the look
> prefix) and a different encoder (DINOv2, readable per TASK-063/064), so it is not a third
> formulation on that corpus. Disclose prominently that the camera and resolution are unchanged,
> and state the rationale as an information change, not a retry of the formulation.
>
> R2 (learned inputs): accepted. The fixed look prefix, the step counter and the palm pose from
> forward kinematics all count as legitimate non-privileged inputs. A perception head trained on
> privileged labels is fine, provided the evaluation makes zero privileged reads.
> Privileged-expert DAgger labelling is fine at training time only. Label all of these explicitly
> in the "learned" ladder.
>
> R3 (primary, fallback, milestone): accepted as proposed. Up to 4 dev-cohort evaluations, with
> the carried policy selected on dev. The dev result is an existence result only. Any "learned
> policy works" claim must come from the gated cohort-C test against open-loop replay (5/16 on
> dev) and the named controls. Preregister the fallback trigger exactly as you proposed.
>
> R4: accepted. A first success counts as "learned policy with a DINOv2 encoder", never "LeWM
> driving the robot". The docs keep that distinction.
>
> R5: seeds 46000–46999 are accepted, on condition that you grep the repo, including the
> manifests, and confirm no overlap. Record the check.

**Owner ruling R6, verbatim (received 2026-09-27T13:13Z via the coordinator, also posted on
#78):**

> Owner ruling R6 (2026-09-27 13:13Z), also posted on #78: both §17 departures are accepted.
> C-3 gets its own 3 DAgger rounds, and R-3 is trained and run on the development cohort. Both
> make the controls stronger, so G2 and G3 become harder to pass, not easier.

A later correction from the coordinator changed only the year of the TASK-066 GPU hold
("about 00:10Z on 2026-09-28 (hard cap 04:08Z), not 2028. Nothing else changes."). The check R5
asks for is §4.1. The rulings are also recorded, verbatim, in the manifest's `owner_rulings`.

## 1. The question, and the claim it can make

**Question.** Give a learned policy, at run time, the information the scripted expert acts on
and nothing privileged:
- apple and plate xy, read once from the post-look frame by a learned readout on frozen DINOv2
  tokens;
- its own step counter;
- proprioception;
- the palm pose by forward kinematics of that proprioception.

Train it by behaviour cloning and three DAgger iterations. Does it reach **at least one full
Apple→Plate success** on the 16 development resets D?

**Claim scope, fixed now.**
- **What M1-PASS would say:** a learned policy (rung L1, §6) with a frozen DINOv2 encoder
  reached at least one full Apple→Plate success on the non-gating development cohort, with zero
  privileged reads at run time.
- **What it would not say:**
  - That the policy works, or beats open-loop replay. Replay reached 5/16 on D. Claims of that
    kind come only from M2 on cohort C (§11; ruling R3).
  - That LeWM, or any world model, drives the robot (R4).
  - Anything about the test split, another camera, or other resets.

The proposal's argument for why this can work where TASK-056 failed is its §2, and it is not
repeated here. The central facts:
- The expert's command is a function of the reset apple and plate positions, the palm pose and
  its command counter (`scripted.py`).
- It is parked on 43.3 % of `orient` steps, and 79.3 % on the 43 noise-level-0 roots. It takes
  91/170 `orient`→`descend` switches from a parked state. This comes from the label-only probe at
  `7c9d08a`, run-4.
- The TASK-056 arms had no clock, and the preregistered readouts found no apple position in their
  decision frame beyond a prior (TASK-059).

## 2. What is reused unchanged

| piece | source |
|---|---|
| corpus | `data/apple-look-v1`, manifest sha256 `81d760d1…db64` (TASK-064). Train and val only, through TASK-065's `TrainValReader` (it refuses every other episode id before a file is opened). |
| the look | `observation_reprobe.look_sequence` / `execute_look` (8 commands, sha256 `17bfb070…587c`), with the collection bounds |
| reset rule | `wide_reset` from `scripts/evaluate_apple.py`, imported, never reimplemented (as `evaluate_policy.py`) |
| encoder | pinned DINOv2 ViT-S/14 (`pretrained_encoder`, digest `3a697b87…2af27`; floor seed-0 digest `3d305f9c…7c9db`), 112 → 224 px bicubic, CPU float32, read-out point `tokens` (256 × 384) |
| readout | TASK-063's probe (`info_ceiling.gram`, `select`, `split_fit`: linear and RBF kernel ridge, family and λ by inner CV) |
| expert and DAgger labeller | `scripted.apple_collector_policy` through `policy_diagnostics.ShadowExpert` (TASK-057 Amendment 1's corrected shadow expert, advanced on the executed result) |
| scorer | `task.AppleToPlateTask`, unchanged |
| embodiment path | clip to the configured bounds of `configs/apple_wm_v4.yaml` (±0.5 on the right arm; left arm and left grasp pinned), then the unchanged `project_candidates`, then `execute`. Guard refusals end an attempt as a non-success (`evaluate_policy.GUARD_REFUSALS`). |
| harness controllers | `policy_diagnostics` `ScriptedOracleController`, `hold_controller` and `RandomController` |

## 3. Data (fixed now)

- **BC-0.** Every **non-aim** train root of `apple-look-v1`: `aim_offset_applied` is false,
  which leaves 134 of the 170 roots. Policy steps only (phase index ≥ 0; the 8 look steps are
  excluded).
  - The label is `collector__base_action`, restricted to the 7 free dimensions and clipped to
    the configured bounds.
  - The mask is TASK-056's post-displacement mask: exclude steps where the apple has moved
    more than 1 cm in xy from its frame-0 position before `grasp` latched, or where
    `privileged__apple_dropped` is set.
  - Branch episodes are excluded, because their base policy is corrupted. Aim-offset roots are
    excluded because their labels servo to a wrong target.
- **Val.** The non-aim val roots, prepared the same way. They are used for selection only.
- **Perception sets (new, rendered by the run).** Each is a reset from `wide_reset(seed)`, the
  look, the post-look `onboard_rgb` frame, and the reset truth of apple and plate xy
  (privileged, as training targets and scoring references only).
  - perception-train: seeds 46000–46255;
  - perception-held-out: seeds 46256–46383.
- Privileged labels are training targets and scoring references only. They are never a run-time
  input.

## 4. Seeds

| use | seeds |
|---|---|
| perception-train | 46000–46255 (256) |
| perception-held-out (S0-P) | 46256–46383 (128) |
| C0 calibration | 46384–46415 (32) |
| DAgger iteration 1 / 2 / 3 | 46416–46543 / 46544–46671 / 46672–46799 (128 each; shared by P, C and R) |
| reserved, unused | 46800–46899 |
| smoke runs only (amendment 1; nothing from them is read) | 46900–46999 |
| development cohort D (M1) | 45000–45007, 45100–45107 |
| cohort C (M2 only; separate authorization) | 45300–45339, from stored manifest values |

Model seed 0; readout folds `default_rng(6701)`; C0 error directions `default_rng(6700)`;
training batches `default_rng(6702)`; the random harness controller seed 6703.

### 4.1 The seed-overlap check (ruling R5), recorded

This ran on 2026-09-27 at `f2e9f63`. The command:
`grep -rnoE '\b46[0-9]{3}\b' --include='*.py' --include='*.json' --include='*.md' --include='*.yaml' --include='*.yml' --include='*.sh' .`
It covers `src`, `scripts`, `tests`, `configs`, `docs` and `benchmarks/manifests`. Nine hits in
five files, none of them a seed use:
- the proposal, which names this range;
- `tests/test_policy_preflight.py` (twice) and `apple_policy_v1_results.md`, which use 46000 as
  an example of a seed a guard must refuse;
- `reach_pilot_v2.md` (0.46038, a decimal) and `apple_control_results_v2_resumed1.md`
  (0.46615 m, a distance).

Every declared seed range in `src`, `scripts`, `tests` and the manifests was also listed. The
declared ranges nearest to it are 45000–45107, 45200–45207 and 45300–45339 below, and
47000–47199, 47900–47931 and 48000–48999 above; none contains any of 46000–46999. No local corpus manifest (`data/*/meta/jepa_manifest.json`) records a reset seed in
that range. **Result: no overlap.** `first_policy.check_seed_ranges` re-checks, as a guard, that
every task range lies inside 46000–46999, is pairwise disjoint, and is off every forbidden
range.

## 5. The controller and its training (fixed now)

### 5.1 Perception

- **Feature.** The post-look frame goes through the pinned DINOv2 on CPU, read at `tokens`.
- **Readout.** TASK-063's probe maps the tokens to apple xy and to plate xy (world frame, m).
  - **Fit** on 426 post-look frames: those of the 170 `apple-look-v1` train roots (frame 8,
    labels at frame 0) and of the 256 perception-train resets.
  - **Cross-fitted estimates** for the BC-0 roots come from 10 folds over those 426 frames
    (`default_rng(6701)`). The head therefore trains on realistic estimate error.
  - **The full fit**, frozen before any closed loop, supplies the estimates for val, DAgger
    rollouts and every evaluation.
  - DAgger frames are not added to the readout.
- **R-arm floor.** The same pipeline, fitted identically on the seed-0 random-init DINOv2's
  tokens.

### 5.2 The policy P

- **Inputs, 132-d.** Each is standardised by BC-0 train moments, with the std floored at
  1e-3, so a constant column (C-3's estimates) maps to 0, never NaN. The moments are fixed
  thereafter.
  - Estimates: apple xy and plate xy (4). They are fixed for the attempt.
  - Clock: `clock_features(t)` (33), where t is the post-look step. It is t / 745 plus sin and cos
    at 16 geometric periods from 4 to 2048 steps. (A period of 2 would make sin(πt) pure
    rounding noise.)
  - Proprioception: 86-D joint positions and velocities.
  - Palm pose: the right palm position plus the first two rotation columns (9). It is computed
    by forward kinematics of the observed joint positions on the controller's own MjModel and
    MjData (`ee_pose(side, data=scratch)`; the pelvis is fixed in this scene), never read from
    the live simulator.
- **What P does not get, and why that is not missing information.** The expert's command also
  depends on the apple's reset height (its targets are offsets from `obj`) and on its own last
  applied grasp (the release ramp). The apple rests on the table at a height that is a scene
  constant across `wide_reset` resets, and the release ramp is a function of the clock. The head
  learns both from the labels.
- **Head.** LayerNorm → Linear(132, 512) → SiLU → Linear(512, 512) → SiLU → Linear(512, 512) →
  SiLU → Linear(512, 7). The outputs are the 7 free dimensions, assembled into the 14-D contract
  with the pinned values.
- **Loss.** Mean squared error to the clipped expert label.
- **Training.** Each training runs on MPS, from scratch on its aggregate, for 30 000 updates:
  AdamW, learning rate 3e-4 cosine to 3e-5, weight decay 1e-4, gradient clip 1.0, batch 256,
  seed 0.
- **Selection.** Every 1 000 updates, by val mean squared error. The eligibility rule is a
  per-dimension output std ≥ 0.02 on val and update > 0. If no checkpoint is eligible, the arm
  scores 0/16 on D and is reported as `no_eligible_checkpoint`.
- **Declared weakness.** Val action error is a weak proxy for closed-loop success.

### 5.3 DAgger

For k = 1, 2, 3, the selected P-(k−1) runs **in command** (β = 0) on the 128 seeds of iteration
k. Each attempt goes:
1. reset;
2. capture the reset truth for the labeller, as the collector does, before the look;
3. the look;
4. the estimates;
5. up to 800 policy steps.

Every visited state with t < 745 is labelled by the shadow expert on the learner's clock. The
labels are aggregated with all earlier data, and P-k is retrained from scratch and selected on
val.

- **The labeller is privileged, at training time only** (R2).
- **C and R run their own DAgger** on the same seeds with their own policies.
- **Rollout inference runs on CPU** in 8 simulation workers, with one torch thread each.

### 5.4 Arms (the full list; "learned arm" means exactly P-0, P-1, P-2, P-3, C-3 and R-3)

| arm | what | rung (§6) |
|---|---|---|
| **P-0 … P-3** | P after 0–3 DAgger iterations | L1 |
| **C-3** | no-image control: P's recipe with the estimates fixed to their BC-0 train mean; its own DAgger × 3 | L1 |
| **R-3** | random-init floor: P's recipe on the seed-0 random-init DINOv2 tokens and their own readout; its own DAgger × 3 | L1 |
| **F** | fallback, trained only on M1-MOTOR (§9): one head per collector phase, same architecture, the head chosen by `scheduled_phase(t)` (the collector's clock schedule, a scripted switch); trained on P-3's final aggregate, 30 000 updates, selected on val | L2 |
| **A4-look** | readout estimates → the scripted `apple_collector_policy`, built from a dict with the estimated apple and plate xy. The apple and plate heights, `container_surface_z`, `object_support_height` and the base pose are declared constants of the committed scene. They are read from the compiled scene model (static geometry) at construction, never from `task_truth()` at run time, and a PR 2 test checks them against `task_truth()` on a non-cohort reset. | L3 |
| **D-oracle-perc** | the carried P-k's head (P-3 when no P-k succeeds) fed the **true** reset xy; its own report | L4 |
| **B-oracle** | `scripted_oracle` with the look | L4, harness |
| **B-hold**, **B-random** | as TASK-057 B1 (random seed 6703) | L4, harness |
| **B-replay** | open-loop replay, after the live look, of the successful non-aim train root whose post-look DINOv2 CLS is nearest (Euclidean on train-standardised features). It replays that root's executed `action` column from its post-look step 0 (frame 8) to its end or to 800 steps, through the same clip and projection. | L4 |

Every attempt:
- runs at most 800 policy steps after the look, with a 300 s wall cap;
- counts a guard refusal as a non-success;
- is scored for `grasp` and `success` by the unchanged scorer.

## 6. What counts as "learned" (the ladder, with ruling R2's allowances labelled)

| rung | at evaluation time | reported as |
|---|---|---|
| **L1: learned policy** | Every command after the look comes from a trained network. Its inputs are the onboard RGB frames, proprioception, forward kinematics of proprioception and the step counter. There is no scripted controller or phase switch, no replay, no substitution, and **zero** privileged reads by the controller. | "learned policy with a DINOv2 encoder" (R4) |
| **L2: partially learned** | Learned motor heads, but a scripted component selects the head or phase | partially learned; never counted as a learned success |
| **L3: learned perception, scripted control** | A trained readout feeds a scripted controller | not learned |
| **L4: privileged, scripted, replayed or substituted** | Any run-time privileged read, the collector, `scripted_oracle`, replay, D-oracle-perc | not learned |

**Allowed in L1 by ruling R2, each labelled.**
- **(a) The fixed look prefix.** It is reset-independent and non-privileged.
- **(b) The step counter.** It is non-privileged.
- **(c) The palm pose.** It is forward kinematics of the robot's own joint readings.
- **(d) A perception readout trained on privileged reset labels.** It is allowed because the
  evaluation makes zero privileged reads.
- **(e) Privileged-expert DAgger labels.** They are used at training time only.

**Not allowed.** No L1 arm's controller may read `sim.task_truth()`, any `privileged__` or
`collector__` label, or the scorer. G-privileged (§12) enforces this.

## 7. Stage 0: calibration and offline gates (inside the gated run, before any closed loop)

### 7.1 C0, the expert's tolerance curve (scripted, privileged; calibration only)

On the 32 C0 seeds, with the look, `apple_collector_policy` is built from the reset truth plus an
injected estimate error, with no action noise. The error has a fixed size and a direction drawn
from `default_rng(6700)` per seed. Nine conditions × 32 attempts:
- the reference, with no error;
- apple error 0.5, 0.8, 1.0 and 1.2 cm, with the plate exact;
- plate error 1.0, 1.5, 2.0 and 2.5 cm, with the apple exact.

**The rule** (`first_policy.c0_bars`). The p90 bar is the largest level at which that level and
every smaller one reach at least 28/32 successes. The median bar is min(cap, p90 bar).
- The caps are 0.75 / 1.2 cm for the apple and 1.5 / 2.5 cm for the plate (median / p90).
- **The bars can only tighten below the caps, never loosen.**
- **CAL-ESCALATE** is the row if the reference is below 28/32 or if the smallest level of either
  quantity is below 28/32.

### 7.1a The A4-look threshold (calibrated from C0 and S0-P, as the proposal specified)

This follows `first_policy.a4_threshold`. For each of the 128 held-out perception resets, the
predicted A4-look success probability is the C0 success fraction at the smallest tested level at
or above its apple error (0 beyond the largest), times the same for its plate error, divided by
the C0 reference fraction, and capped at 1.0 (amendment 1). The threshold T is
⌈0.5 × 16 × the mean prediction⌉, and at least 1.
The reading is that perception is "adequate in the loop" when A4-look reaches at least half the
successes that C0 and S0-P predict for it. T is computed before any D attempt and recorded in
the report.

### 7.2 S0-P, perception on fresh held-out resets (gate)

On the 128 perception-held-out resets, the frozen readout's apple and plate xy errors are
measured, in cm (`first_policy.s0_perception`). **Passes:** the median and the p90 are each at or
below the C0 bar, separately for the apple and for the plate. The corpus val roots are not used
here.
- Apple fails → **S0-APPLE-FAIL**.
- Apple passes and plate fails → **S0-PLATE-FAIL**.

### 7.3 S0-D1, pipeline equivalence (a guard; a failure is V)

On the first 8 perception-held-out seeds:
- the post-look frame from the closed-loop runner's own path is byte-identical to the perception
  set's frame for that seed;
- the live estimates equal the offline reference within 1e-4 m. The reference is computed through
  the same worker code path at batch size 1 and one thread, so the batch-size effect TASK-065
  measured (≤ 3.8e-5 in features) cannot enter. The tolerance is 0.01 cm, two orders below
  any bar;
- a live `act()` equals the batched offline prediction on the same inputs within 1e-4.

## 8. Stage 1: M1 on the development cohort D

After S0 passes and the three DAgger pipelines have finished, **each M1 arm runs each of the 16
D resets exactly once**. The M1 arms are P-0 to P-3, C-3, R-3, A4-look, D-oracle-perc, B-oracle,
B-hold, B-random and B-replay. P is evaluated on D four times (k = 0–3), as ruling R3 allows,
and every evaluation is reported. DAgger never uses D. The cohort guard whitelists D; every
other seed is refused, cohort C first.

## 9. Pre-declared outcomes (first matching row; `first_policy.decide_m1` for M1)

| row | condition | reading | next (a recommendation; the owner chooses) |
|---|---|---|---|
| **V** | a guard fails (§12), a crash or cap, S0-D1 fails, or the harness fails on D (B-oracle < 14/16 successes, or B-hold or B-random > 0 grasps) | nothing is read | one from-scratch repeat (§13) |
| **CAL-ESCALATE** | §7.1 | the expert is not as tolerant as assumed, or the harness is off | stop; the owner decides; the clause does not fire |
| **S0-APPLE-FAIL** | §7.2 | the post-look frame, through this readout, does not give the apple to the expert's tolerance on fresh resets | **the clause fires** (§10) |
| **S0-PLATE-FAIL** | §7.2 | the apple is read, the plate is not | stop; the owner decides on a design change (for example a per-step plate readout during `transfer`); the clause does not fire |
| **M1-PASS** | some P-k reaches ≥ 1/16 full successes on D | **the first learned (L1) Apple→Plate success, on the non-gating development cohort: an existence result** (R3). It is "learned policy with a DINOv2 encoder" (R4). | M2 on cohort C, under a separate authorization. The carried arm is the P-k with the most D successes, ties to the later k: a selection on D, declared as one. |
| **M1-MOTOR** → F | every P-k scores 0/16; A4-look ≥ T successes (§7.1a) | perception is adequate in the loop, and motor learning is the failing component | F is trained and runs once on D (the trigger proposed and accepted under R3) |
| **M1-MOTOR-F-PARTIAL** | as above, and F ≥ 1/16 | a **partially learned** (L2) success; never counted as learned | the owner decides: M2 for F under its label, and/or a learned phase switch |
| **M1-MOTOR-F-NONE** | as above, and F 0/16 | neither the learned policy nor the phase-decomposed one succeeds | **the clause fires** |
| **M1-PERCEPTION** | every P-k scores 0/16; A4-look < T | the estimates do not survive the closed loop, or are not accurate enough in it | **the clause fires** |

**Reported whatever the row; none of them changes it:**
- the C-3, R-3, D-oracle-perc and B-replay counts. C-3 can reach ≥ 1/16: the clock-only blind
  controller already grasped 2/16 on D in TASK-057. A P-k success then says little about vision
  until M2's G2 is read.
- every arm's grasp and success counts, stage occupancy, termination reasons, step counts, and
  per-dimension command statistics (signed);
- S0-P's values against its bars, and C0's full table;
- the DAgger data sizes and every training's val curve and selected update.

`M1-MOTOR` is an intermediate state of `decide_m1` (train and run F), never a final row.

## 10. Abandonment clause

**It fires on S0-APPLE-FAIL, M1-MOTOR-F-NONE or M1-PERCEPTION.**
- **What closes:** learned control on `apple-look-v1` at the 112 px onboard camera. No further
  head, loss, input, DAgger or readout variant is preregistered on this corpus and this camera
  without new evidence of a different kind.
- **The conclusion is the one `apple_policy_v1.md` §7 pre-declared, quoted verbatim in the
  results:** "the honest published conclusion is that a 112 px onboard camera plus a single-mode
  scripted-collector corpus does not support learned Apple→Plate on this platform, and the
  product goal needs a data or hardware change — not another model."
- **The observation change on record** is the fixed overview camera (TASK-061 O-raw). It is a
  hardware or workspace decision, and it belongs to the owner.
- **What does not close:** the LeWM backend, DINOv2 as an encoder, the product goal, and the
  corpus (sealed; test split unread).

## 11. M2: the gated test on cohort C (preregistered now; run only under a separate authorization)

The test is on the 40 cohort-C resets. The values come from `benchmarks/manifests/apple-policy-v1.json`'s stored resets and are never recomputed. That discharges the stored-values debt of `task056_handover.md` §7, which is blocking for any runner that opens C.

**Arms.** The carried P-k, C-3, R-3, B-replay, B-oracle, B-hold and B-random.

**Stop rule.** It applies to the carried P-k only. If it has 0/16 grasps on D it does not run on
C, and M2 fails. The controls (C-3, R-3, B-replay) and the harness always run on C, so a
control's D result cannot fail P's gates.

**Precedence field, cited as `apple-policy-diagnostics-v1.json` requires of any successor's gate
section.** `precedence_rule_D1_over_G_SUB.exemption_spent` is `false` at this preregistration,
and this protocol does not claim the exemption.

| gate | condition |
|---|---|
| G1 | the carried P-k reaches ≥ 17/40 full successes **and** strictly more than B-replay on the identical resets |
| G2 | P − C-3 ≥ +8 successes (the exact McNemar test, with realised n_d and p reported) |
| G3 | P − R-3 ≥ +8. **Declared reading if it fails:** "a learned visuomotor policy works; encoder pretraining contributes nothing measurable" (R-tok met every readability bar in TASK-064). |
| G4 | the carried P-k reaches ≥ 20/40 grasps |
| G5 | B-hold and B-random 0/40 grasps; B-oracle ≥ 38/40 successes. A failure invalidates the run, not the arms. |
| G6 | zero controller privileged reads by any L1 arm; a violation voids the run |
| G7 | median control time ≤ 100 ms per command, including the one DINOv2 forward pass |

**Rows (first match).**
- **M2-VOID:** G5 or G6 fails; the run is invalid, not the arms.
- **M2-PASS:** every gate passes.
- **M2-FAIL-VISION:** G1 and G4 pass and G2 fails. There is no evidence the image is used.
- **M2-FAIL:** any other failing gate. The claim is not made, and the owner decides.

A gate that cannot be evaluated counts as failed. **What a pass would mean:** "a learned policy with a DINOv2 encoder works on Apple→Plate on this cohort". It
would not mean that LeWM drives the robot (R4).

## 12. Guards (any failure is V)

| guard | condition |
|---|---|
| **G-hash** | every pinned file (the manifest's `hashes`, added in PR 2) matches at preflight and at the end; the tracked tree is clean at both |
| **G-data** | the corpus manifest is `81d760d1…db64`; its splits are 170 / 20 / 10 sessions |
| **Q-split** | only train and val episodes are decoded; `test_split_decoded: false` is recorded |
| **G-weights** | DINOv2 files and the pretrained and floor digests match their pins |
| **G-seeds** | `check_seed_ranges`, plus every simulated seed lies in its declared range; the D whitelist; cohort C refused |
| **G-look** | on every attempt, the applied look commands equal the requested ones, and the post-look joint state equals the first attempt's within 1e-6 |
| **G-privileged** | Applies to every **evaluation** attempt of an L1 arm. DAgger rollouts read truth through the labeller, before the look and outside `act()`, by design (R2 e). (1) `task_truth()` is wrapped with a counter that records any call made while a controller's `act()` is on the stack; that count must be 0. (2) The counter is installed after the reset and after the scorer's construction (both read `task_truth()`), so the attempt's total count must equal the scorer's `evaluate()` calls (amendment 1). (3) L1 controllers are constructed without any robot or simulator handle, and receive only the `Observation` and the step; a PR 2 test asserts this. (4) FK runs on the controller's own MjModel and MjData. |
| **G-device** | training on MPS; features and rollouts on CPU |
| **G-finite** | every feature, estimate, loss, prediction and count is finite |
| **G-cap** | every cap in §14 |

S0-D1 (§7.3) and the harness condition on D (§9) also lead to V. **`exemption_spent`**
(`apple-policy-diagnostics-v1.json`, `precedence_rule_D1_over_G_SUB`) is `false` and is not
claimed; it is cited here as that manifest requires.

## 13. Void rule

- **A run that stops early other than at a declared early-stop row (CAL-ESCALATE,
  S0-APPLE-FAIL, S0-PLATE-FAIL) is V.** That covers a guard, a crash or a cap. The run records a
  `void_reason`, and nothing in it is read. A declared early-stop row is an outcome, not a
  void, and it is never repeated.
- **Exactly one from-scratch repeat** is allowed, into `run-2`, with the same seeds, caps and
  device.
- **A second V closes TASK-067 as INCONCLUSIVE.**
- **Any guard change needs an owner ruling.** A fix between runs is otherwise limited to runner
  mechanics, goes through a reviewed PR and a fresh pre-run GO, and is disclosed.

## 14. Budget, devices, caps

**Devices.**
- Training runs on **MPS**.
- DINOv2 features, readouts and statistics run on CPU (6 threads, float32, as TASK-063 to
  TASK-066).
- Simulation and rollout inference run in 8 CPU workers, with one torch thread each.

**Caps.**

| cap | seconds |
|---|---|
| global | 36 000 |
| each training | 1 800 |
| each rollout batch | 3 600 |
| C0 | 3 600 |
| perception collection | 1 800 |
| each attempt | 300 |

**Expected time: about 2–3 h.** Anchored on committed timings (DINOv2 14.3 ms per frame on CPU;
TASK-057's 256 closed-loop attempts in 2 792 s), the parts are:
- perception collection, a few minutes;
- C0, 288 attempts, about 15 min;
- 12 trainings, a few minutes each;
- DAgger, 1 152 rollouts on 8 workers, about 50 min;
- M1, 192 attempts, about 10 min;
- F, only on M1-MOTOR.

**Output.** `outputs/task067-first-policy/run-<k>/`, with checkpoints in
`checkpoints/task067-first-policy/run-<k>/`. The runner refuses to overwrite either. Nothing is
committed except manifests and hashes.

## 15. Process

1. **PR 1 (this one).** It holds this document, the manifest, `first_policy.py` and its tests,
   and the card. It merges on an independent reviewer's reported APPROVE and green CI.
2. **PR 2.** The runner, the policy and training code, the A4-look constants, the tests (every
   guard and row in both directions, the privileged-read counter, the D whitelist and C refusal,
   S0-D1) and the pins. It also carries tiny CPU smoke checks only, with noise labels. It merges
   on a reported APPROVE and green CI.
3. **The gated run.** It starts only after TASK-066's run has finished and the coordinator
   releases the GPU, and only on a fresh pre-run reviewer's **reported** GO, delivered as a
   message.
4. **PR 3.** The results document and the results manifest. A reviewer checks every restated
   number against `report.json`. If the clause fires, PR 3 also adds a `docs/DECISIONS.md`
   entry.
5. **One task, one agent.** A defect found after the freeze goes to the owner and is fixed only
   through a disclosed amendment. Nothing is re-thresholded, retrained or re-selected after
   numbers are seen, beyond what §8 and §9 declare.

## 16. Not done (declared)

- No world model is in the loop: no critic and no latent MPC. The world model's role after M1 is
  the proposal's stage 3, conditional on TASK-066 passing, with its own preregistration.
- No per-step image features, no end-to-end image head, no chunked or diffusion head.
- No apple-hidden spurious check for the readout. The readout's accuracy on fresh held-out
  resets is what the controller depends on, and S0-P measures it for both the apple and the
  plate. TASK-061 to TASK-064's spurious checks covered the apple only; no such check exists
  for the plate.
- No test-split decode, and no cohort-C attempt in this gated run.

## 17. Differences from the accepted proposal (disclosed; the two substantive ones confirmed by R6)

The owner accepted the proposal "as proposed" (R3). These changes were made while writing the
preregistration.

**Confirmed by owner ruling R6 (2026-09-27T13:13Z):**
1. **The no-image control is C-3, not the proposal's C-noimg.** The proposal's C-noimg was
   "P-0's recipe" (behaviour cloning only). C-3 gets its own three DAgger iterations on the same
   seeds. The reason is that M2's G2 compares the carried P-k, which has had DAgger, with this
   control, and a control without DAgger would make G2 easier to pass.
2. **R-3 is added to M1 and to the learned arms.** The proposal had the random-init floor only
   in M2's G3. It must be trained and run on D for M2's arms to exist, so it is listed now.

**Tightenings or clarifications, recorded:**
3. C0 has 9 conditions (four levels each, plus the reference) instead of 7, and its caps are
   0.75 / 1.2 cm for the apple and 1.5 / 2.5 cm for the plate.
4. S0-APPLE-FAIL fires the clause outright. The proposal said "unless the owner rules
   otherwise", an escape hatch this removes.
5. DAgger frames are **not** added to the readout. The proposal said they would grow its
   training set; the readout is instead frozen before any closed loop.
6. S0-D1 compares estimates, within 1e-4 m at batch size 1, rather than features.
7. The readout uses the `tokens` read-out point.
8. The apple-hidden check is not run (§16).
9. M2 rows are added; the stop rule applies to the carried P-k only; controls always run.

The A4-look trigger follows the proposal: it is calibrated from C0 and S0-P (§7.1a). An earlier
draft of this PR fixed it at an uncalibrated 8/16. Review caught that it departed from R3, and it
was replaced before any merge.

## 18. Amendment log

**Amendment 1 (PR 2, before any gated run, before any compute).** Nothing here was changed after
any number existed; no policy had been trained and nothing had been simulated.
1. **A4-look threshold: each per-reset predicted rate is capped at 1.0** (`A4_RATE_CAP`). Review
   of PR #78 found that `ra × rp / ref` can exceed 1 when a C0 level succeeds more often than the
   reference by sampling. A probability cannot exceed 1. The frozen block gains
   `a4_threshold.rate_cap`.
2. **`D_RESETS_FOR_A4` is removed.** `a4_threshold` now uses `D_RESETS`, defined before it. This
   is a code-only change, with no effect on any value.
3. **G-privileged (2).** The counter is installed after the reset and after the scorer's
   construction, both of which read `task_truth()` (`embodiment.reset`, `AppleToPlateTask`), so
   the expected total is the scorer's `evaluate()` calls. The earlier wording would have voided
   every attempt.
4. **Smoke seeds.** Smoke runs use 46900–46999, split off the reserved range, which is now
   46800–46899. Nothing from a smoke is read.
5. **Estimate path.** Estimates are computed in the main process, at batch size 1, from the
   post-look frame a worker renders on the attempt's own reset path. Every attempt re-renders that
   frame and must match its sha256 (G-frame, part of S0-D1). Live and offline estimates are
   therefore identical by construction, and S0-D1's 1e-4 m estimate check holds trivially. S0-D1
   still compares the live `act()` against the batched offline prediction.
6. **No eligible checkpoint.** If a family's checkpoint at iteration k has none eligible, that
   family's later iterations are not rolled out, and every later checkpoint of the family also
   scores 0/16 on D (`no_eligible_checkpoint`). This extends §5.2's rule to DAgger, which needs a
   policy to roll out.
7. **S0-D1's timing.** Its `act()` comparison needs a trained P-0, so it runs right after BC-0
   training. That is still before any DAgger rollout and before any D attempt. The frame
   comparison runs on every attempt (item 5).
