# Apple wide-jitter object-aware ceiling v2: preregistration

This is the prospective TASK-049 protocol. It is committed before any attempt on
the fresh development resets 45100–45107 and before any v2 attempt on the
TASK-047 resets 45000–45007. **Every arm is a NON-LEARNED diagnostic:**

- `privileged_object` (v2) is an exact MuJoCo-rollout planner whose cost, phase
  transitions and release predictor read simulator object state.
- `demo_replay` is open-loop replay of a retrieved TRAIN demonstration.
- `scripted_oracle` is the privileged scripted collector.

No outcome here is a learned result, and none counts toward TASK-033/TASK-034.
Learned Apple→Plate stays at zero successes.

## Why

TASK-047's object-aware ceiling v1 grasped on 5/8 wide-jitter development resets
(45000–45007; gate ≥ 6) while the scripted collector succeeded on 8/8, so the
preregistered reading was `ceiling_inadequate_task_feasible`: redesign the cost
and phase transitions under a new preregistration. The v1 post-hoc diagnosis
(`apple_wide_object_ceiling_results_v1.md`) found only geometric failures;
rollout parity was exact throughout.

The ceiling matters beyond itself. The product goal is a learned LeWM world model
that plans with an object-aware cost computed from learned heads. This ceiling
tests whether the *cost and phase design* works when dynamics and object state
are perfect; the learned controller will reuse it. Every v2 cost term and
transition is therefore stated in a quantity a learned head could predict
(listed at the end).

## Question and hypothesis

**Hypothesis (primary gate).** On the 8 fresh wide-jitter development resets
45100–45107, the v2 ceiling reaches

- full task success on **≥ 6/8**, **and**
- the scorer's `grasp` stage (reach, a ≥ 5 cm lift with hand contact) on
  **≥ 7/8**,

with exact rollouts (zero robot- and full-state parity mismatches over
non-vacuous checks), every primary ceiling attempt counted and valid provenance.

## The v2 design (`src/embodied_jepa/object_ceiling_v2.py`)

v2 subclasses v1 and changes only the items below. The v1 module keeps its
behaviour (a behaviour-preserving refactor moved the progress append into
`_record_progress`; all v1 tests pass unchanged) and the v1 plan is unchanged.
Twin, planner (horizon 6, 24 candidates, 2 rounds, std 0.3→0.05 floor, single
elite, hold and warm-start candidates, commitment 1), arm bounds, grasp schedule
of the other phases, phase order, approach and lift/retreat costs, stops and
the rejected-step convention are v1's.

| Item | v1 (TASK-047) | v2 (this protocol) | v1 failure it addresses |
|---|---|---|---|
| Descend cost | ‖p − g(0.052)‖ (3-D, dominated by the blocked height) | 3 · ‖(p − g)_xy‖ + \|p_z − g_z\| + apple displacement | 45003/45004: xy drift unpenalised |
| Blocked descent | height stalled AND xy < 1.0 cm | height stalled AND (xy < 5 mm, OR xy stalled — < 1 mm gain over 10 commands — AND xy < 1.2 cm) | 45003/45004: 1.05–1.25 cm, never "blocked" |
| Close cost | ‖p − g_start‖ + apple displacement | v1 term + 1.0 · max(0, ‖p_xy − p_xy,start‖ − 1 cm) | 45006: palm drifted 2.5 cm during close |
| Transport cost | ‖a_xy − plate‖ + carry + slip + contact | 4 · ‖a_xy − plate‖ + carry + slip + contact | — (centring priority) |
| Lower cost | 2 · ‖a_xy − plate‖ + \|a_z − z_rest+1.5 cm\| + slip + contact | 4 · ‖a_xy − plate‖ + same | 45000/45007: xy traded for height |
| Hand-over to release | transport reached (< 1.5 cm) or blocked inside 4 cm → lower; lower reached or blocked → release | **placement predictor**: from transport or lower, only when the carried apple is in hand contact within **3 cm** of the plate centre *and* the exactly probed release predicts a placement; blocked transport → lower; blocked lower keeps planning | 45000/45007: released 4.5–4.6 cm off-centre |
| Release | 40 commands, CEM hold, grasp bound −1 (fingers snap open) | **60 commands**, exactly the probed sequence: arms held (zero-width bounds), right grasp opening by 0.08/command (the collector's ramp), then held open | apples flung and rolled to the plate rim |

**Placement predictor.** In transport and lower, once per command, the twin
executes the release sequence (60 commands, arms held, right grasp 1 − 0.08·k
clipped at −1, left hand open) from a copy of the live state and records the
apple's path, speed and hand contact. A placement is predicted when, for **4
consecutive commands** (0.2 s ≥ the scorer's 0.15 s dwell), the apple is out of
hand contact, within ±6 mm of the plate support height (the table lies only
1.2 cm lower), within 3 cm of the plate centre and slower than 0.1 m/s. The probe
records no parity candidate and no planning diagnostic. Because the release phase
then executes exactly that sequence, an accepted prediction is what the live
release does under exact dynamics. Each probe is recorded in the trace
(`predicted_landing`, with `placed_at`). The carried-over-the-plate condition
exists because the scorer's `transport` stage needs the lifted apple in contact
over the plate before a placement counts.

## Development-tuning disclosure (TRAIN-side tuning resets only)

v2 was designed and tuned on **tuning resets 49000–49015**, drawn with the same
wide-jitter rule (`evaluate_apple.wide_reset`), plus TRAIN reset 42000 for the
smoke. The range 49000–49015 is used nowhere else in the repository (checked
against docs, manifests, scripts, src, tests and `.mc`); it is disjoint from the
TRAIN/VAL/TEST collection 42000–42031, the narrow development 43000–43004, the
final cohort 44000–44019, the TASK-047 resets 45000–45007 and the fresh resets
45100–45107. No 45000–45007 or 45100–45107 reset was simulated during design. The
v1 traces of 45000–45007 were read only for the post-hoc diagnosis already
published with TASK-047 (and, like v1, the design fixes target those diagnosed
mechanisms), so 45000–45007 are reported only as a secondary, non-gating
cohort.

Design probes ran in-process (an uncommitted scratch harness: same embodiment,
twin, controller and unchanged scorer, no rendering) into the worktree's ignored
`outputs/task049-scratch/`. In order:

| Probe | Change tested | Tuning-reset outcome | What it drove |
|---|---|---|---|
| baseline | v1 on 49000–49007; collector on 49000–49007 | v1 grasp 7/8, success 3/8 (49005 descent stall at 0.85 cm xy; 4 releases rolled to the plate rim); collector 8/8 | confirmed the v1 mechanisms on tuning draws |
| v2a | xy-weighted descent + xy-keyed block; close holding the achieved 3-D pose (xy weight 3); landing-end release gate 1.5 cm aiming at a learned release offset | stopped early: descent now centred on 49005, but the close pushed the apple out of the hand | close must keep downward pressure |
| v2b | close: achieved xy (weight 3) + downward pressure | grasp 5/8; every transport stalled (the release-offset aim absorbed table landings 13–25 cm away) | landing must be on the plate surface |
| scans | close xy weight 1.0 and 0.3 (to 230 commands) | grasp 7/8 each (49001 fails) | weight 1.0 |
| v2c | weight 1.0; landing needs plate support ±6 mm | grasp 12/16, success 0/16: one on-plate landing set a 10–12 cm "release offset" | drop the release-offset aim; aim at the centre |
| v1 check | v1 on 49008–49015 (to 260 commands) | grasp 7/8 | v1's close pressure grasps more reliably than v2's |
| v2d | aim at centre; release tolerance 2.5 cm | success 1/16: releases from a moving arm flung the apple, and the live CEM release differed from the probe | release must be the exact probed sequence |
| v2e | release = zero-width bounds, gradual opening, exactly probed | stopped early (superseded) | — |
| v2f | close = v1 pressure + 1 cm dead-band drift guard | grasp 15/16, success 2/16: releases accepted with the carried apple 11 cm off the plate (no transport stage) and landings that later rolled to the rim | gate on the carried apple over the plate and on a predicted resting placement |
| **v2g (frozen)** | placement predicted as ≥ 4 consecutive resting, on-plate, out-of-contact, < 3 cm commands during a 60-command probed release; carried apple within 3 cm of the centre and in contact | **full success 15/16** (272–304 commands); 49001 lost the apple during the close; 0 full-state parity mismatches | frozen |

The v2g configuration is exactly the defaults of `ObjectCeilingV2Config`
(recorded in the plan as `object_ceiling`). No parameter was changed after v2g.

These are design runs, not evidence. Because of them, a pass shows adequacy on
new draws from the distribution the tuning resets spanned.

## Fresh development resets (new; nothing else changes)

Same distribution and rule as TASK-047 (apple ±3 cm, plate ±2 cm around the
unchanged centres; `rng = default_rng(seed)`), new seeds **45100–45107**. All 16
v2 resets pass the simulator's reset-envelope check. The offsets (cm) are
computed from the rule, not outcomes:

| Reset | Apple dx, dy | Plate dx, dy | Apple offset from centre |
|---|---|---|---:|
| 45100 | −3.00, −1.12 | +1.64, +1.92 | 3.20 |
| 45101 | −0.28, +1.56 | +0.75, −0.20 | 1.58 |
| 45102 | −1.27, +2.43 | −0.81, −0.27 | 2.74 |
| 45103 | +2.52, +2.90 | −0.02, −1.64 | 3.84 |
| 45104 | −0.75, +1.63 | +1.72, −1.11 | 1.80 |
| 45105 | −0.46, +1.26 | −0.42, −0.10 | 1.35 |
| 45106 | +1.32, +0.99 | −1.25, −1.66 | 1.65 |
| 45107 | −1.46, +2.77 | −1.43, +1.66 | 3.13 |

The narrow development cohort, the final cohort 44000–44019, TEST, TASK-034's
criteria and every earlier plan are untouched (tests pin this).

## Arms (one command, 48 attempts, mode-major)

| Order | Mode | Resets | Role |
|---|---|---|---|
| 1 | `demo_replay` | 45100–45107, then 45000–45007 | reported; never gates |
| 2 | `scripted_oracle` | same | feasibility reference; never gates |
| 3 | `privileged_object` (v2) | same | **gate** on 45100–45107 only |

`demo_replay` retrieval, `scripted_oracle` and the guarded open-loop projection
are exactly TASK-047's.

## Frozen inputs (unchanged from TASK-043–047)

- Corpus `data/apple-branches-v1`, manifest SHA-256
  `6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`.
- Checkpoint `checkpoints/apple-branches-h16-sensor-v1/sensor.pt`, SHA-256
  `0192b99a60abf1d426d127505680570f238401b5a80f6270dd4859d6e64aa5a5` (retrieval
  only).
- Action manifest `configs/g1_sim_action.json`; pinned G1/Dex3 assets.
- Regenerated TRAIN artifacts must be byte-identical to TASK-045/046/047:
  `state_goals.npz` `839190fc…f88d`, `state_calibration.json` `d8dae051…8d75`.

## Budget (frozen)

- 48 attempts, no retries, replacement seeds or reruns.
- ≤ 1,000 commands; **1,200 s per attempt**; **20,400 s global** including
  preparation and finalization; `min(attempt cap, global remaining)` allocation;
  CPU, Torch ≤ 4 threads.
- **Per-command observe+plan deadline 10 s** (v1: 5 s). The run shares the machine
  with a parallel data-collection job; the ceiling is not a real-time claim, and a
  deadline miss would be an uncounted attempt. The change is declared here, before
  any outcome.
- Worst case: 32 × ~15 s + 16 × 1,200 s ≈ 19,700 s < 20,400 s, so no ceiling
  attempt can be shortened by the global cap unless preparation exceeds ~700 s.
- Expected: ~0.8–1.2 s per ceiling command, 250–500 commands per attempt, about
  1.5–2.5 h in total.

## Primary gate (`report.json["object_ceiling_v2_gate"]`)

`primary_gate_passed` is true only when, on the primary resets 45100–45107:

- `privileged_object` full successes ≥ 6 AND grasp resets ≥ 7;
- zero robot-state and zero full-state rollout parity mismatches in counted
  ceiling attempts, over non-vacuous checks (≥ executed commands − 1 per
  attempt);
- all 8 primary ceiling attempts are counted;
- provenance is valid.

Counted attempts, clean terminations and zero-stage handling are TASK-047's.
`demo_replay`, `scripted_oracle` and the secondary resets never change the gate.

## Readings and next steps (fixed now; `readings.outcome`)

| Outcome | Condition | Reading | Next step |
|---|---|---|---|
| `ceiling_adequate` | gate passed | Under exact dynamics and perfect object state the v2 cost/phase design grasps, lifts, transports and places on fresh draws: the design is adequate as the target for a learned controller | **T4**: pair this design with a learned world model whose heads predict its terms (below), trained on the wide-jitter TRAIN data (TASK-048), under its own preregistration; keep this ceiling as the upper reference |
| `grasp_adequate_place_inadequate` | grasp ≥ 7, success < 6 | The grasp design suffices; transport/release does not | Redesign only transport/release under a new preregistration; T4 may target the grasp+lift sub-goal meanwhile |
| `ceiling_inadequate_task_feasible` | grasp < 7, scripted successes ≥ 6 | The scripted collector succeeds where v2 does not grasp | Diagnose and redesign under a new preregistration; do not pair with a learned model |
| `ceiling_inadequate_scripted_also_fails` | grasp < 7, scripted successes < 6 | These draws may exceed the grasp mechanics | Audit the fresh resets and distribution before any cost redesign |
| `ceiling_inadequate_feasibility_reference_inconclusive` | grasp < 7, scripted arm incomplete | Cause unattributed | Repair the reference arm; repeat into a new output directory |
| `inconclusive` | a primary ceiling attempt uncounted, provenance invalid or rollouts inexact | — | Fix the defect; rerun into a new versioned output; keep this run as a recorded failure |

Non-gating diagnostics reported with the result:

- `replay_separated_diagnostic`: `demo_replay` grasp resets ≤ ceiling grasp
  resets − 3 on the primary resets (TASK-047's decision rule, as a count only).
- The secondary cohort 45000–45007 for all three arms (v1 got 5/8 grasp,
  3/8 success there).
- Per reset and arm: termination, ordered stages, phase log with reached /
  blocked / `release_predicted` ends, release probes, parity counts, wall time.

With n = 8 the thresholds are preregistered decision rules, not significance
tests.

## Cost terms a learned model must predict for T4

| Term | Used in | Learned-head quantity |
|---|---|---|
| Palm–apple offset (3-D), vs the grasp offset | approach, descend, close, blocked-descent rule | predicted palm–apple relative position |
| Palm pose (position, top-down angle) | every phase, close drift guard | proprioception / forward kinematics (known) |
| Apple displacement from phase start | descend, close | predicted apple xy change |
| Apple rise above its start | lift, transport carry, lift→transport | predicted apple height |
| Palm–apple offset change (slip) | lift, transport, lower | predicted relative offset change |
| Hand–apple contact | lift/transport/lower, lift→transport | predicted contact / grasp-state head |
| Apple–plate xy offset | transport, lower | predicted apple–plate relative position |
| Placement after the scripted gradual release | transport/lower → release gate | predicted "apple comes to rest on the plate within 3 cm" after the release sequence (a release-outcome head) |

## Recorded TRAIN smoke (reset 42000 only)

The smoke runs `evaluate_apple.run` with the frozen v2 arguments and
`WIDE_V2_COHORT` patched to `(42000,)`, `WIDE_V2_SECONDARY` to `()` and
`wide_reset` to the narrow TRAIN reset, through the complete supervisor,
snapshot, preparation and worker path (scratch output
`outputs/task049-scratch/smoke-42000-a` in the TASK-049 worktree, uncommitted
working tree of this protocol's code). It is a runtime check, not evidence.

- Report `completed`, provenance valid, 3/3 attempts counted, 217.3 s global.
- TRAIN artifacts byte-identical to TASK-045/046/047: `state_goals.npz`
  `839190fc…f88d`, `state_calibration.json` `d8dae051…8d75`.
- `demo_replay` succeeded (501 commands, 8.4 s); `scripted_oracle` succeeded
  (501 commands, 8.1 s).
- `privileged_object` v2 succeeded in 278 commands (180.9 s, 0.65 s/command):
  approach 0, descend 110, close 126 (descent blocked), lift 171, transport 199,
  release 242 (`release_predicted`); 43 release probes; 277/277 robot-state and
  277/277 full-state parity checks exact.
- The one-reset gate object computed `ceiling_adequate` (gate code exercised only).

## Frozen run command

Execute **once**, from a clean checkout of the reviewed commit, into a new
directory under the main checkout's `outputs/`:

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_apple.py \
  --dataset data/apple-branches-v1 \
  --checkpoint checkpoints/apple-branches-h16-sensor-v1/sensor.pt \
  --output <main checkout>/outputs/apple-wide-object-ceiling-v2 \
  --stage development --goal-kind object --object-ceiling-version 2 --no-proposals \
  --horizon 6 --stride 16 --dwell 1 --candidates 24 --iterations 2 \
  --commitment-steps 1 --max-steps 1000 --attempt-max-seconds 1200 \
  --max-seconds 20400 --control-timeout 10
```

Seeds default to 45100–45107 then 45000–45007; modes to
`demo_replay scripted_oracle privileged_object`. Every outcome, including
failures, stalls and timeouts, goes into `apple_wide_object_ceiling_results_v2.md`
and `benchmarks/manifests/apple-wide-object-ceiling-v2.json`.

## Known risks declared before outcomes

- **Privileged and non-learned.** A pass says the design suffices *given exact
  dynamics and perfect object state*; nothing about whether a learned model can
  supply the terms.
- **Release predictor is exact only here.** It runs the true simulator; a learned
  model must predict the release outcome, which is likely the hardest head.
- **Tuning.** 16 tuning resets from the same distribution shaped v2; the gate uses
  fresh draws but the same distribution.
- **Shared machine.** A parallel job may slow commands; the 10 s deadline and
  1,200 s cap are sized for that, but a deadline miss or timeout would make the
  reading inconclusive.
- **Short run.** n = 8 primary resets; every count has wide uncertainty.
