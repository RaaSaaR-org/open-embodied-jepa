# Experiment index

Every protocol in this directory with its results document, outcome row and manifest, newest
first. Protocols and results are the frozen historical record: they are not rewritten, and
failures and negative results stay in them. Read each outcome with the caveats in its results
document; the one-line outcomes below are pointers, not claims.

**Status (canonical sentence, [DECISIONS.md](../DECISIONS.md), decision 2026-10-02, R7, as updated
on 2026-10-08 by R18.28, R18.31, R19.18 and R19.21).**
Learned Apple→Plate on the frozen v1 MVP benchmark (TASK-020) is 0/150 per backend (`native_jepa`
and LeWM). On `apple-to-plate-v2`, the behaviour-cloning/DAgger policy P-3 (an MLP on a frozen
DINOv2 readout, trained on demonstrations from the privileged scripted expert e9; not a world model)
scored 40/40 counted successes on the held-out cohort C against 39/40 for its random-init encoder
control R-3 (one run, one training seed per arm, 40 resets, one camera at 112 px onboard, a narrow
reset distribution), so TASK-072 M2 is M2-FAIL on G3 (encoder pretraining contributed nothing
measurable), and cohort C is no longer held out. LeWM's first gated closed-loop Apple→Plate success
is TASK-081's, on v2 (one run, one model seed whose `last_two_triggered` training flag is set, in
simulation only, under the declared simulation-only plate condition C1-M), in which a LeWM token
predictor on frozen DINOv2 features chose only the single place aim between P-3's learned pick and
e9's scripted place, committing it with a local affine fixed-point solver. On 128 fresh gated resets
it reached 118/128 counted successes (exact 95 % interval 0.861–0.962), above the 112/128 bar, ahead
of its action-blind twin (73/128), its two scene-blind twins (39/128 and 59/128) and a random aim
(20/128) by exact one-sided McNemar tests at p < 0.01, and non-inferior within the allocated margin
of −16/128 to the non-learned hand-written rule controller H-rule, which is given the simulator's
plate law (125/128; paired difference −7/128, 95 % interval −13/128 to −2/128). The row is L-PASS,
the preregistered primary claim "LeWM-driven closed-loop success". "LeWM needed" is not shown:
H-rule scored measurably higher than LeWM (the interval excludes 0), and the privileged look-ahead
ceiling scored 126/128. That the new solver mattered is not shown either: TASK-080's solver, run on
the same resets and reported only, scored 117/128 (paired difference +1/128, 95 % interval −5/128
to +7/128). TASK-080's earlier gated run of the same model under the same condition with that solver
(58/64 against H-rule's 63/64) stays L-NEAR, and TASK-081's 128-reset cohort was chosen after that
result. LeWM's only closed-loop Apple→Plate runs on v1 have 0 successes. Scripted-expert,
privileged-ceiling, oracle and GR00T successes are not project-learned results.

Manifests are under [`benchmarks/manifests/`](../../benchmarks/manifests/). Run outputs are
git-ignored and stay on the machine that made them; several finished run worktrees have been
archived to external storage (archived, see docs/STORAGE.md). A results document's own paths are
not rewritten when its worktree is archived.

## Gated and preregistered tasks (newest first)

| Task | Protocol | Results | Outcome (pointer only) | Manifest |
|---|---|---|---|---|
| TASK-083 | [apple_lewm_seed_replication_v2](apple_lewm_seed_replication_v2.md) (**FROZEN**, R21) ([Stage 0](apple_lewm_seed_replication_v2_stage0.md)) | – | **Not run.** TASK-081's L-PASS replicated with W's two other model seeds of the same training run (66801, 66802; each with its own N, R-S and R-N; nothing trained or fitted) under C1-M, on one fresh cohort of 128 resets (75200–75327) shared by both seeds and the seed-independent arms. TASK-081's arms, affine_local, G-bar 112/128, G-NI within 16/128 and the McNemar tests per seed; the replication claim needs both seeds at L-PASS (REP-PASS), else REP-ONE / REP-NONE; no clause; W-66800 reported only. Stage 0 power at W = 0.922: each seed passes with probability 0.52–0.79, both 0.27–0.63 | `apple-lewm-rep-v2.json` (FROZEN, R21.17); Stage S next on its GO |
| TASK-082 | [apple_lewm_unknown_law_v2](apple_lewm_unknown_law_v2.md) (**DRAFT**, R20; never frozen) | CAL-ESCALATE | **No LeWM run.** Preregistration under U-sat, C1-M with the plate law after 405 replaced by a saturating, swirling law that no hand-written arm is given (a declared task change). TASK-077's recipe retrained on a new 2 000-root corpus (W and N × 3 seeds, about 31 h GPU); K0 on 64 resets; Stage G gates dynamics and refitted readouts on gate-P; D 16 and S 128 resets; primary claim per R9.8 with G-NI against H-rule (C1-M's written law) and the linear H-sysid; "LeWM needed" reported with Stage-0 power; a learned-baseline tier reported only (not expected to leave headroom) | **Closed at K0: CAL-ESCALATE**, no clause (R20.23–R20.25, [K0 record](apple_lewm_unknown_law_v2_k0.md); Stage 0 [record](apple_lewm_unknown_law_v2_stage0.md)). K0 on 64 resets (privileged and hand-written arms only, one run): τ curve 64, 55, 55, 34, 22, 21 of 64, so τ_commit = 0 (0.5 cm one below 56/64); r_K = 466 > 465; ceiling 64/64; 0/64 aims outside the box; H-rule 54/64 reported. Nothing frozen, no model trained, no LeWM arm run |
| TASK-081 | [apple_lewm_commit_precision_v2](apple_lewm_commit_precision_v2.md) (**FROZEN**, R19) ([Stage 0](apple_lewm_commit_precision_v2_stage0.md)) | [results](apple_lewm_commit_precision_v2_results.md) | **L-PASS** at the gated Stage S (R19.20; 128 fresh resets, one run, one model seed flagged `last_two_triggered`, simulation only, C1-M; LeWM chooses only the place aim): W 118/128 (0.861–0.962) passed G-bar (112) and beat N 73, L-shuf 39, L-mean 59 and L-rand 20 (McNemar p < 0.01); G-NI passed against the non-learned H-rule 125/128 (W − H-rule −7/128, 95 % [−13, −2] against −16/128). The primary claim "LeWM-driven closed-loop success"; **"LeWM needed" not shown** (H-rule measurably better); **solver effect not shown** (W-frozen 117/128, W − W-frozen +1, [−5, +7]); privileged ceiling 126/128. Earlier: Stage D **D-PASS** (R19.17; development, 16 resets): W 16/16, W-frozen 15/16, N 5/16, L-shuf 5/16, L-mean 8/16, H-rule 16/16, privileged H-final(commit) 16/16. W's solver after the grid becomes affine_local (the design note's tested variant; no retraining, no refit); twins carry it; W-frozen reported only; no K0 or offline gate; D 16 resets, S 128 resets with G-bar 112/128 and G-NI within 16/128 | `apple-lewm-cp-v2.json` (FROZEN) |
| TASK-080 | [apple_lewm_c1m_v2_pred_readout](apple_lewm_c1m_v2_pred_readout.md) (FROZEN, R18) ([Stage 0](apple_lewm_c1m_v2_pred_readout_stage0.md)) | [results](apple_lewm_c1m_v2_pred_readout_results.md) | **L-NEAR** at the gated Stage S (64 resets, one run, one model seed, simulation only): W 58/64 passed G-bar (56) and beat N 26, L-shuf 21, L-mean 30 and L-rand 11 (McNemar p < 0.01), but failed G-NI against the non-learned rule controller H-rule 63/64 (W − H-rule −5/64, 95 % [−10, 0] against −8/64; not detectably inferior); escalate, no clause, **no claim**; privileged ceiling 62/64. Earlier: K0′-PASS, CORPUS-SEALED, R-PASS offline, Stage D D-PASS (W 16/16, development) | `apple-lewm-pr-v2.json` (FROZEN) |
| TASK-077 | [apple_lewm_c1m_v2](apple_lewm_c1m_v2.md) (FROZEN, R17) ([Stage 0](apple_lewm_c1m_v2_stage0.md)) | [results](apple_lewm_c1m_v2_results.md) | **G-NO-BAR** at Stage G (Stage T T-DONE): G1–G4 pass on all three seeds, G5 (a) fails on all three (predicted-plate readout 1.58–1.87 cm against τ_commit 1.0 cm); escalate, no clause; D and S not run, so no LeWM closed loop | `apple-lewm-c1m-v2.json` |
| TASK-076 | [apple_plate_twin_v2](apple_plate_twin_v2.md) ([Stage 0](apple_plate_twin_v2_stage0.md)) | [results](apple_plate_twin_v2_results.md) | **TWIN-PASS** (H-twin 64/64 vs H-clock 51/64; random-init floor also 64/64; no world model); K-pred **PRED-INFEASIBLE** (cell A removed), so Branch B; no clause fires | `apple-plate-twin-v2.json` |
| TASK-075 | [apple_obs_ceiling_v2](apple_obs_ceiling_v2.md) | [results](apple_obs_ceiling_v2_results.md) | **OBS-NONE**; the clause fires (next: a task or condition change) | `apple-obs-ceiling-v2.json` |
| TASK-074 | [apple_lewm_planner_v2](apple_lewm_planner_v2.md) | [results](apple_lewm_planner_v2_results.md) | **INCONCLUSIVE**, closed without the clause after two budget escalations; no LeWM controller ran | `apple-lewm-planner-v2.json` |
| TASK-073 | [apple_wm_critic_v2](apple_wm_critic_v2.md) | [results](apple_wm_critic_v2_results.md) | **S-NO-CONDITION** at K0; the critic never ran; the clause does not fire | `apple-wm-critic-v2.json` |
| TASK-072 (M2) | [apple_first_policy_v2_m2](apple_first_policy_v2_m2.md) | [results](apple_first_policy_v2_m2_results.md) | **M2-FAIL on G3 alone** (P-3 40/40, R-3 39/40 on cohort C); G3's reading adopted by owner ruling | `apple-first-policy-v2-m2{,-results}.json` |
| TASK-072 | [apple_first_policy_v2_linux](apple_first_policy_v2_linux.md) | [results](apple_first_policy_v2_linux_results.md) | **REPLICATED** on Linux (M1-PASS, P-3 16/16 on the development cohort) | `apple-first-policy-v2-linux{,-results}.json` |
| TASK-071 | [apple_first_policy_v2](apple_first_policy_v2.md) | [results](apple_first_policy_v2_results.md) | **M1-PASS** on the development cohort (P-3 16/16, R-3 16/16) | `apple-first-policy-v2{,-results}.json` |
| TASK-070 | [apple_to_plate_v2_expert](apple_to_plate_v2_expert.md) (protocol and results) | same | **PASS**: e9 at rest 32/32 (plate exact), 30/32 at 1.0 cm; a scripted-expert result | `apple-to-plate-v2-expert-gate-v1.json` |
| TASK-067 | [apple_first_policy_v1](apple_first_policy_v1.md); [proposal](control_formulation_proposal_v1.md) | [results](apple_first_policy_v1_results.md); [release probe](apple_first_policy_v1_release_probe.md); [landing diagnosis](apple_first_policy_v1_landing_diagnosis.md) | **CAL-ESCALATE** at C0; probe P-CANDIDATE-FAIL; closed under the owner's fallback | `apple-first-policy-v1{,-results}.json` |
| TASK-066 | [apple_token_dynamics_v1](apple_token_dynamics_v1.md) | [results](apple_token_dynamics_v1_results.md) | **WM-TOK-DYNAMICS** (train split only; four caveats) | `apple-token-dynamics-v1{,-results}.json` |
| TASK-065 | [apple_latent_dynamics_v1](apple_latent_dynamics_v1.md) | [results](apple_latent_dynamics_v1_results.md) | **WM-NO-DYNAMICS**; the clause fires (pooled-CLS predictor line only) | `apple-latent-dynamics-v1{,-results}.json` |
| TASK-064 | [apple_look_corpus_v1](apple_look_corpus_v1.md) | [results](apple_look_corpus_v1_results.md) | **C-ACCEPT**: `apple-look-v1` accepted | `apple-look-corpus-v1{,-results}.json` |
| TASK-063 | [apple_pretrained_encoder_v1](apple_pretrained_encoder_v1.md) | [results](apple_pretrained_encoder_v1_results.md) | **O-PT-POOLED** (offline readability only) | `apple-pretrained-encoder-v1{,-results}.json` |
| TASK-062 | [apple_encoder_study_v1](apple_encoder_study_v1.md) | [results](apple_encoder_study_v1_results.md) | **O-ENC-ARCH**; the clause fires (in-corpus encoder training closed) | `apple-encoder-study-v1{,-results}.json` |
| TASK-061 | [apple_observation_reprobe_v1](apple_observation_reprobe_v1.md) | [results](apple_observation_reprobe_v1_results.md) | **O-LOOK-RAW** | `apple-observation-reprobe-v1{,-results}.json` |
| TASK-059 | [apple_info_ceiling_v1](apple_info_ceiling_v1.md) | [results](apple_info_ceiling_v1_results.md) | **O-OCC-NONE** | `apple-info-ceiling-v1{,-results}.json` |
| TASK-058 | [claim_audit_v1](claim_audit_v1.md) | same | numerical-claim audit; corrections are errata in the audited documents | — |
| TASK-057 | [apple_policy_diagnostics_v1](apple_policy_diagnostics_v1.md) | [results](apple_policy_diagnostics_v1_results.md) | **Outcome X**; the behaviour-cloning clause fires | `apple-policy-diagnostics-v1{,-results}.json` |
| TASK-056 | [apple_policy_v1](apple_policy_v1.md); [handover](task056_handover.md) | [results](apple_policy_v1_results.md) | **FAILED** on the development stop rule (every arm 0/16) | `apple-policy-v1.json` |
| TASK-054 | [apple_world_model_v4](apple_world_model_v4.md) | [results](apple_world_model_v4_results.md) | all four arms fail the 14 gates; the clause fires (CEM over this cost abandoned as the primary line) | `apple-world-model-v4.json` |
| TASK-052 | [apple_world_model_v3](apple_world_model_v3.md) | [results](apple_world_model_v3_results.md) | all four arms fail the gate set | `apple-world-model-v3.json` |
| TASK-051 | [apple_wide_grasp_closure_v3](apple_wide_grasp_closure_v3.md); [diagnosis](apple_grasp_closure_diagnosis.md) | [results](apple_wide_grasp_closure_results_v3.md) | primary gate passed (privileged ceiling, 8/8, `ceiling_adequate`) | `apple-wide-grasp-closure-v3.json` |
| TASK-050 | [apple_world_model_v2](apple_world_model_v2.md) | [results](apple_world_model_v2_results.md) | FAIL: accurate on still frames, not where the offset moves | `apple-world-model-v2.json` |
| TASK-049 | [apple_wide_object_ceiling_v2](apple_wide_object_ceiling_v2.md) | [results](apple_wide_object_ceiling_results_v2.md) | primary gate failed, conclusively (privileged ceiling, 5/8) | `apple-wide-object-ceiling-v2.json` |
| TASK-048 | [apple_wide_collection_v1](apple_wide_collection_v1.md) | [results](apple_wide_collection_results_v1.md) | every acceptance check passed; `apple-wide-v1` accepted (scripted collection) | `apple-wide-collection-v1.json` |
| TASK-047 | [apple_wide_object_ceiling_v1](apple_wide_object_ceiling_v1.md) | [results](apple_wide_object_ceiling_results_v1.md) | `ceiling_inadequate_task_feasible` (privileged ceiling) | `apple-wide-object-ceiling-v1.json` |
| TASK-046 | [apple_hybrid_phase_v1](apple_hybrid_phase_v1.md) | [results](apple_hybrid_phase_results_v1.md) | primary gate failed (non-learned diagnostic) | `apple-hybrid-phase-v1.json` |
| TASK-045 | [apple_trajectory_tracking_v1](apple_trajectory_tracking_v1.md) | [results](apple_trajectory_tracking_results_v1.md) | stage-1 gate failed (privileged ceiling) | `apple-trajectory-tracking-v1.json` |
| TASK-044 | [apple_privileged_ceiling_v1](apple_privileged_ceiling_v1.md) | [results](apple_privileged_ceiling_results_v1.md) | primary gate failed (privileged ceiling) | `apple-privileged-ceiling-v1.json` |
| TASK-043 | [apple_state_goal_control_v1](apple_state_goal_control_v1.md) | [results](apple_state_goal_control_results_v1.md) | primary gate failed (learned grasp 0/4) | `apple-state-goal-control-v1.json` |
| TASK-042 | [apple_aligned_control_v1](apple_aligned_control_v1.md) | [results](apple_aligned_control_results_v1.md) | offline primary gate failed (+4.30 points against +5) | `apple-aligned-cost-v1.json` |
| TASK-041 | [apple_goal_alignment_v1](apple_goal_alignment_v1.md); [data audit](apple_goal_alignment_data_audit.md) | [results](apple_goal_alignment_results_v1.md) | offline image-goal screen passed (offline only) | `apple-goal-alignment-v1.json` |
| TASK-040 | [apple_arrival_feedback_v1](apple_arrival_feedback_v1.md) | [results](apple_arrival_feedback_results_v1.md) | negative paired result | `apple-arrival-feedback-v1.json` |
| TASK-039 | [apple_control_commitment_v1](apple_control_commitment_v1.md) | [results](apple_control_commitment_results_v1.md) | negative, time-limited development result | `apple-control-commitment-v1.json` |
| TASK-036 | [joint_limit_precision_v1](joint_limit_precision_v1.md) | — | joint-target precision investigation | — |
| TASK-029/030 | [feasibility_v1](feasibility_v1.md), [feasibility_v2](feasibility_v2.md), [projection optimisation](projection_optimization_v1.md) | [v1 results](feasibility_results_v1.md), [v2 results](feasibility_results_v2.md) | candidate-feasibility diagnostics | `feasibility-results-v{1,2}.json` |
| TASK-024 | [jepa_wms_spike](jepa_wms_spike.md) | same | optional JEPA-WMs adapter: CPU compatibility only, no manipulation claim | `jepa-wms-spike.json` |
| TASK-019/020 | [mvp_final](mvp_final.md), [mvp_evaluation](mvp_evaluation.md) | [training](mvp_training_results.md), [results](mvp_results.md) | the frozen v1 MVP benchmark: **0/150 per model** for `native_jepa` and LeWM | `mvp-{corpus-v0,goals-v0,leakage-audit,results-v0}.json` |
| TASK-014 | [reach_pilot](reach_pilot.md), [v1](reach_pilot_v1.md), [v2](reach_pilot_v2.md) | [results](reach_results.md) | development reaching: the v2 selectors reached 1/5 each against 0/5 for the controls (intervals overlap); native v0 reached 3/5 with a collapsed representation | `reach-pilot-v0*.json`, `reach-development-results.json` |

## Earlier apple development records (TASK-030 to TASK-038, sensor world model)

These predate the per-task index above; their task numbers are in the MC cards and the documents
themselves. Plan and diagnosis: [apple_mvp_plan](apple_mvp_plan.md),
[apple_wm_diagnosis](apple_wm_diagnosis.md), [apple_mechanics_probe](apple_mechanics_probe.md),
[waypoint_design](waypoint_design.md). Collection: [apple_collection_v1](apple_collection_v1.md)
→ [results](apple_collection_results_v1.md). Sensor model:
[apple_sensor_training_v1](apple_sensor_training_v1.md) →
[results](apple_sensor_training_results_v1.md);
[apple_sensor_diagnostics_v1](apple_sensor_diagnostics_v1.md) →
[results](apple_sensor_diagnostics_results_v1.md). Matched branches:
[apple_branches_v1](apple_branches_v1.md) → [results](apple_branches_results_v1.md),
[review](apple_branch_review.md); [apple_branch_training_v1](apple_branch_training_v1.md) →
[results](apple_branch_training_results_v1.md);
[apple_branch_diagnostics_v1](apple_branch_diagnostics_v1.md) →
[results](apple_branch_diagnostics_results_v1.md);
[apple_branch_training_h16_v1](apple_branch_training_h16_v1.md) →
[H16 diagnostics](apple_branch_h16_diagnostics_results_v1.md). Control:
[apple_control_development_v1](apple_control_development_v1.md) →
[results](apple_control_results_v1.md);
[apple_control_development_v2](apple_control_development_v2.md),
[interruption](apple_control_interruption_v2.md), [resume](apple_control_development_v2_resume.md)
→ [results](apple_control_results_v2_resumed1.md);
[apple_control_forecast_audit_v1](apple_control_forecast_audit_v1.md) →
[results](apple_control_forecast_results_v1.md). Every learned control attempt here ended without
meeting its gate.

Earlier manipulation and reproduction records:
[manipulation_pilot](manipulation_pilot.md),
[manipulation_controller_v1](manipulation_controller_v1.md),
[manipulation_controller_v2](manipulation_controller_v2.md) (scripted collection, not learned),
and [clean_reproduction](clean_reproduction.md) (software reproduction only).

## Development, not gated

No preregistration and no gate. Nothing here is a project-learned result.

| Date | Record | What it found |
|---|---|---|
| 2026-10-08 | [apple_lewm_unknown_law_v2_design](apple_lewm_unknown_law_v2_design.md) (**DRAFT** design note for TASK-082, R19.24–R19.25) | After TASK-081's L-PASS, direction (a): a plate law that no hand-written arm is given (C1-M with the law after 405 replaced; a task change). Development check (seeds 72000–72355, CPU, no world model, one run, arms on the true plate): U-sat (saturating, swirling) ceiling 32/32, H-rule with C1-M's written law 23/32, linear H-sysid 28/32, kernel-ridge sysid on 256 roots 32/32; U-play (hysteretic) leaves H-rule and H-sysid at 31/32. "LeWM needed" testable against hand-written arms; against a learned non-LeWM baseline it is not expected (no headroom). Recommends a preregistration under U-sat with TASK-077's recipe retrained (about 31 h GPU) |
| 2026-10-08 | [apple_lewm_commit_precision_v2_design](apple_lewm_commit_precision_v2_design.md) (**DRAFT** design note for TASK-081, R18.33–R18.35) | After TASK-080's L-NEAR, direction (b), W's commit precision: the capped refinement oscillates on a predicted-plate map that is rough below about 1 cm. Development check (seeds 70000–70063, CPU, one run, W-66800 and R-S unchanged): cap bound on 19/64; closed loop W:frozen 57/64, W:damped 58/64, **W:affine_local 62/64** (fixed point of a local affine fit to W's own grid predictions), H-rule 64/64, privileged H-final(commit) 63/64; affine_local's median aim error 0.243 cm against frozen's 0.382 (both against H-final's aim). Selection among three executed variants; not gated. Recommends a preregistration with δ = 8/64 unchanged and n = 128 |
| 2026-10-07 | [apple_lewm_c1m_v2_decomposition](apple_lewm_c1m_v2_decomposition.md) (development record after TASK-077, R17.53–R17.54) | Where W's G5 miss comes from (CPU; TASK-077's six kept checkpoints; train and val only, never the gate split; no training, no closed loop): **D-READOUT** under the rule declared before the run. A readout refitted on W's predicted latents under the stand-in reads the plate at r at 0.556–0.578 cm in val median (upper bounds 0.61–0.64 against τ_commit 1.0 cm; 0.34–0.38 of N's), against 2.86–3.41 cm for the frozen R8; a readout fitted on executed-command predictions does not transfer to stand-in ones (4.6–9.5 cm). Optimistic (checkpoints selected on val); next: a new preregistration on fresh roots. No LeWM closed loop |
| 2026-10-05 | [apple_lewm_next_v2_c1m_feasibility](apple_lewm_next_v2_c1m_feasibility.md) (development record, R16) | C1-M's feasibility checks (CPU, no world model, one run, block 63000–64999; debug 64900–64979 subsets, not read): **M-PROCEED**. ρ\* = 4 cm (disc; H-final(commit) 30, 30, 29/32 at 3, 4, 5 cm); fresh ceiling 32/32 and every proxy clears the strict +8/32 bar (mean-proxy 12/32, +20 [+15, +25]); τ_commit = 1.0 cm; H-read 4 × 4 misses (58/64, +6) and 8 × 8 meets the 4/64 allowance exactly (60/64, +4 [+1, +8], inside the noise); both readout learning curves still falling; H-rule and H-sysid 30/32 on the reading. Admits only drafting a preregistration, committed to an 8 × 8 latent whose dynamics are ungated. Privileged proxies, not trained twins; no LeWM result |
| 2026-10-04 | [apple_lewm_next_v2_direction](apple_lewm_next_v2_direction.md) (**DRAFT** ruling, R15) | After C1-TWINS-ESCALATE: a scene-blind twin misses by p̄ − p for any κ, L or commit step, so only plate spread at the decision helps; recommends C1-M (C1 plus a post-pick plate move over a disc, radius set by the ceiling; the strict twin bar is projected to clear only from 5 cm) with the readout judged by an oracle-dynamics arm (H-read). Development re-analysis of C1 run-2 only (no simulation; reset values on 58000–58511). No run, no protocol |
| 2026-10-04 | [apple_lewm_next_v2_c1_feasibility](apple_lewm_next_v2_c1_feasibility.md) (development record, R14) | C1's feasibility checks (CPU, no world model, seeds 57000–57031 and 57100–57355; debug 57900–57905 and 57910–57939, not read): **C1-TWINS-ESCALATE**. F1, F2, F6 pass (H-final(commit) 31/32, r = 460, a_lo = −0.5); F3 fails on the mean-proxy (+6/32 headroom, feasibility 0.728); F5's readout at r misses the note's provisional 0.5 cm (0.886 cm; plate-hidden passes). Escalate, no clause; no protocol |
| 2026-10-04 | [apple_lewm_next_v2_design](apple_lewm_next_v2_design.md) (**DRAFT** design note, R9) | Branch B after TASK-076's cell A was removed: three candidate designs for the next LeWM task under R9.8's two claims; recommends C1, a single aim committed at 405 under the reactive-plate rule, with a CPU-only feasibility check first. No run |
| 2026-10-02 | [apple_white_plate_dev](apple_white_plate_dev.md) | an opt-in white plate leaves TASK-075 at OBS-NONE; any colour effect on the offset's median error is bounded to about 0.90–1.11 (1 of 12 intervals excludes 1.0), though colour does shift the plate-hidden check; colour is not what limits the readout |
| 2026-10-01 | [ARENA.md](../ARENA.md) §8 | GR00T reference baseline in Isaac Lab-Arena (client-only; NVIDIA's GN1x-Tuned release (GR00T N1.7, step 65000), not the tutorial's checkpoint-20000): 16/30 and 10/30 under Arena's rule, 0/30 under the strict at-rest rule (stale PhysX velocity, inferred; post-hoc position check 6/30). GR00T's result, not ours |
| 2026-10-01 | [ARENA.md](../ARENA.md) §1–§7 | the GR00T-tutorial Arena scene; arm kinematics agree, Dex3 fingertips differ by up to 3.1 cm; e9 at rest 0/16 in Arena |
| 2026-09-30 | [ISAAC_E9_REPLAY.md](../ISAAC_E9_REPLAY.md) | e9 in Isaac/Newton: actions match, at rest 24–26/32 per cell against 31/32 in MuJoCo |
| 2026-09-29 | [ISAAC_NEWTON_SPIKE.md](../ISAAC_NEWTON_SPIKE.md), [ISAAC_V2_SCENE.md](../ISAAC_V2_SCENE.md) | Newton reproduces MuJoCo's contact behaviour on five scripted cases; PhysX does not roll like MuJoCo |
| 2026-09-28 | [ISAAC_BRINGUP_SPIKE.md](../ISAAC_BRINGUP_SPIKE.md), [ISAAC_MJCF_TRANSPORT.md](../ISAAC_MJCF_TRANSPORT.md) | Isaac Sim bring-up, MJCF→USD conversion and joint-level parity |
| 2026-09-28 | [apple_to_plate_v2_feasibility](apple_to_plate_v2_feasibility.md) (TASK-069) | the scan that led to `apple-to-plate-v2` |
| 2026-09-28 | [apple_resting_expert_v1](apple_resting_expert_v1.md) (TASK-068) | closed on its development finding: 0 at rest in 269 attempts under v1 |
