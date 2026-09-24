---
id: TASK-033
aliases:
- TASK-033
title: Train and validate apple world-model dynamics and closed-loop development control
slug: train-and-validate-apple-world-model-dynamics-and-closed-loop-development-contro
status: in-progress
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- training
sprint: ''
depends_on:
- "[[TASK-030]]"
- "[[TASK-031]]"
- "[[TASK-032]]"
due_date: ''
created: 2026-09-20
updated: 2026-09-24
---


# Train and pass dynamics and closed-loop development gates

## Acceptance criteria

- Preregister each bounded training attempt with immutable data/split/action/source hashes, seed, objective, validation selection, memory and runtime caps. Initial budget ceiling: 1800 wall seconds per attempt; no unlimited search.
- Use training/validation only; measure prediction versus persistence and shuffled actions, multi-step drift, collapse where relevant, and state/action sensitivity.
- Run frozen development controls and learned policy with all attempts counted. Require apple grasp/transport/release success with unchanged evaluator before final testing.
- Measure a dynamics-disabled or shuffled-dynamics ablation to establish the learned model contributes to control.
- Preserve negative attempts and declare subsequent hypothesis changes before execution; never retune the sealed final cohort.

## Authorization and workflow

User requested completion toward a working world-model apple pick-and-place MVP on 2026-09-20, with subagents and end-to-end delivery. Coordinator owns Git and task state. Experiments are bounded and recorded before execution.
%% mc-links: [[TASK-030]] [[TASK-031]] [[TASK-032]] %%

## Execution evidence

Initial training protocol and supervised runner committed before execution. Planned3000updates,B16,H8,seed0,CPU4,1800s totalwall. Data/source/protocol hashes mandatory. Physical learned ApplePlate remains unproven; all negative results retained.

First sensor model completed3000 updates and failed both learned development runs (0/6 comparison placements, including three errored controls). Matched-branch retraining improved causal assignment but missed the frozen primary H16 error-reduction gate. Preserve all artifacts and keep this task in progress. H16 balanced-intervention training subsequently passed the unchanged primary causal gate (89.40% assignment; 81.41% endpoint-error reduction). The first physical v2 attempt was externally paused after574 acknowledged commands. An unchanged preregistered resume comparison completed one learned failure at1000 commands (reach only), timed out persistence at846 commands, and left four attempts unstarted under its600-second total cap. No final-cohort evaluation has run. A prospective saved-state forecast audit will investigate the observed goal5 stall before any further controller change.

The separate four-command comparison (TASK-039) subsequently started all six attempts: two learned runs completed 1,000 commands, four controls timed out, and no physical stage was reached. Forecast audit and controller implementation tasks can be delivered independently of this unmet physical gate. A paired arrival-feedback diagnostic and a goal-metric research review follow; final evaluation remains untouched.

TASK-041 passed an offline image-goal screen; TASK-042 then missed the fixed combined-cost gate (+4.30pp versus required +5pp) and stopped before physics. These offline results do not meet this task's physical acceptance. The user requested pausing work after TASK-042 delivery; leave this task open and the final cohort untouched.

TASK-043 ran its single preregistered demonstration-state-goal comparison (source a90bf5b, resets 43000-43003, 16 attempts, 130 s): primary gate failed. All 12 learned/dynamics_shuffle/persistence attempts stalled on goal 0 with zero scorer stages; the non-learned open-loop demo_replay reference succeeded on 3/4 resets. Learned Apple->Plate is still unsuccessful; this task stays open and the final cohort untouched. Evidence: docs/experiments/apple_state_goal_control_results_v1.md, benchmarks/manifests/apple-state-goal-control-v1.json.

TASK-044 privileged MuJoCo-rollout ceiling (non-learned, source 733bfdc, resets 43000-43003): with exact dynamics the same state-goal scaffold also stalled (goal 0/1, 0/4 grasp, 0 stages), so the scaffold, not only the learned dynamics, blocks progress. Evidence: docs/experiments/apple_privileged_ceiling_results_v1.md.

TASK-045 privileged trajectory-tracking ceiling (non-learned, source bae0838, resets 43000-43003): tracking the demo's arm+hand trajectory reached reach/contact on 4/4 but grasp on 0/4 (apple dropped off the table 4/4); gate failed, learned stage not run. Evidence: docs/experiments/apple_trajectory_tracking_results_v1.md.

TASK-046 privileged hybrid phase ceiling (non-learned, source f8dac63, resets 43000-43003): tracked approach + open-loop demo close grasped 0/4 with handoff at the close row (primary gate failed) and 2/4 (full success) with handoff one horizon earlier; demo_replay reproduced 4/4 grasp. Evidence: docs/experiments/apple_hybrid_phase_results_v1.md.

TASK-047 wide-jitter object-aware privileged ceiling (non-learned, source 613ecf8, NEW wide development resets 45000-45007, apple +-3 cm, plate +-2 cm): object-aware exact-rollout ceiling grasped 5/8 (gate needed >=6/8; failed, conclusive), demo_replay 2/8, scripted collector 8/8 -> outcome ceiling_inadequate_task_feasible. Evidence: docs/experiments/apple_wide_object_ceiling_results_v1.md.

TASK-048 wide-jitter TRAIN corpus (privileged scripted collector, source 32865f5): all preregistered acceptance checks A1-A13 passed; data/apple-wide-v1 = 797 episodes / 205,519 transitions, dataset manifest SHA-256 028e130576..., accepted as the world-model-v2 training corpus. Collection outcomes only, not learned results. Evidence: docs/experiments/apple_wide_collection_results_v1.md, benchmarks/manifests/apple-wide-collection-v1.json.

TASK-049 object-aware ceiling v2 (non-learned, source 7b94f06, fresh wide resets 45100-45107): full task success 5/8 (>=6 required) and grasp 5/8 (>=7 required) -> primary gate failed, conclusive; scripted collector 8/8 on the same resets, so outcome ceiling_inadequate_task_feasible again. Evidence: docs/experiments/apple_wide_object_ceiling_results_v2.md, benchmarks/manifests/apple-wide-object-ceiling-v2.json.

TASK-050 world model v2 (first offline gate set on the new corpus; three arms, 15,000 steps each, MPS, data 028e130576...): ALL THREE ARMS FAILED. G1 palm-apple readout 3.65 / 4.08 / 5.84 cm against <=1.5 cm, and G2a (rollout error / the model's OWN persistence readout) 0.835 / 0.858 / 0.844 against <=0.8 - no arm beat its own persistence baseline, so by the pre-declared reading the closed loop did not start. No collapse anywhere (G8 passed on all three). Offline evaluation on recorded validation data only, as the v2 protocol declares its scope. Evidence: docs/experiments/apple_world_model_v2_results.md, benchmarks/manifests/apple-world-model-v2.json.

TASK-051 grasp-closure ceiling v3 (non-learned, source b673f3d, fresh wide resets 45200-45207): the preregistered primary gate PASSED - full task success 8/8 (>=6) and grasp 8/8 (>=7), outcome ceiling_adequate, conclusive true, with exact rollouts (0 mismatches) and the TASK-049 close-phase ejection not occurring once. This supersedes the reading recorded above from TASK-044 and TASK-047 that "the scaffold, not only the learned dynamics, blocks progress": under exact dynamics and perfect object state the cost and phase design is adequate as a target for a learned controller. It is a non-learned diagnostic and establishes nothing about a learned policy. Evidence: docs/experiments/apple_wide_grasp_closure_results_v3.md, benchmarks/manifests/apple-wide-grasp-closure-v3.json.

TASK-052 world model v3 (four arms, source e2f8227, same corpus/splits/hashes as v2): ALL FOUR ARMS FAILED. G1 6.62 / 3.65 / 6.11 / 3.30 cm against <=1.5 cm; G2a 1.010 / 0.876 / 0.940 / 0.831 against <=0.8, so again no arm beat its own persistence readout and the closed loop did not start. BOTH headline hypotheses were wrong in the direction predicted to help: a second camera made readout precision markedly WORSE, and the v3 readout shaping (motion-weighted loss plus auxiliary position targets) made the CEM-relevant ranking metrics WORSE. A follow-up (merged 7c85c04) split the gate error into an encoder term and a rollout term on identical moving windows: v3's one-camera encoders reached 2.47-2.59 cm against v2's 3.26 cm (a 21-24% improvement) while the rollout's excess error roughly doubled to tripled - the error moved into the action-conditioned prediction step. On that evidence the merged document's branch-4 inference (that the evidence pointed at the DATA and an information ceiling) was WITHDRAWN and branch 2 (redesign the action conditioning) became the primary line; that is also what the protocol's own earliest-failing-group rule had selected, since G7a failed on arms A and B. No information ceiling was demonstrated. Evidence: docs/experiments/apple_world_model_v3_results.md, benchmarks/manifests/apple-world-model-v3.json.

TASK-054 world model v4 (four arms, one-factor isolation on the prediction step, trained at 46d62eb/c9cf9a6, same corpus/splits/hashes): ALL FOUR ARMS FAILED the 14-gate preregistered set - no arm passed every gate; E0 control passed 10/14, E1 action-chunk 9/14, E2 tail-weighting 9/14, E3 step-embedding 9/14, so THE UNTOUCHED CONTROL PASSED MORE GATES THAN EVERY INTERVENTION. Primary gate G2a 0.8763 / 0.8814 / 0.8635 / 0.9036 against <=0.8; no arm passed and none came close. E0 was verified as a bit-for-bit control (all 170 weight tensors torch.equal against v3 arm B), so the three new config keys are provably inert at their defaults and any arm difference is attributable to the one option each changed, NOT to the revision. Episode-clustered paired bootstrap (20,000 resamples, seed 20540) against E0 on the rollout term: E1 +0.010 cm [-0.407, +0.367] and E2 -0.173 cm [-0.804, +0.132] both cross zero; E3 +0.761 cm [+0.155, +1.199] excludes zero. E3's encoder damage is the separate encoded-target contrast, +1.023 cm [+0.485, +1.435]. This makes five failed attempts on the rollout gap (second camera, motion-weighted readouts, action chunking, tail weighting, non-shared predictor step). The protocol's pre-declared Outcome B therefore fired and was not deferred again: CEM OVER THIS WORLD-MODEL COST IS ABANDONED as the primary control line, and no further predictor-architecture protocol is preregistered. What carries forward: the encoder (palm-apple offset read to 2.35-2.59 cm, still improving), apple_held AUROC 0.9992-0.9997 on the lift cohort (weak evidence - the shuffled-action AUROC on the same cohort is 0.714-0.812), no collapse (effective rank 6.74-8.23), and the corpus/splits/hashes, with test still never decoded. LIMITS, as the results document records them: ONE SEED PER ARM, so every arm difference is a single-run difference; E0 is a revision control at arm B's seed, not a replication, and cannot distinguish "the intervention did nothing" from "this seed is unlucky"; the bootstrap intervals bound cohort sampling for fixed checkpoints and say nothing about run-to-run variation; selection and the gates both used val; and no interval is available for the rollout-excess (G9) contrast. Offline evaluation only; no closed loop was run and learned Apple->Plate remains at 0 successes. Evidence: docs/experiments/apple_world_model_v4_results.md, benchmarks/manifests/apple-world-model-v4.json.

This task's physical acceptance criteria remain UNMET and are unchanged: no final-cohort evaluation has run, and learned Apple->Plate is still 0 successes. The task stays open and the final cohort stays untouched. The control formulation for the next attempt changed at TASK-054; whether this umbrella task closes is a decision for the user.
