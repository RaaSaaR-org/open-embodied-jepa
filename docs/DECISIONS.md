# Decisions, assumptions, and risks

## Working decisions

| Decision | Rationale | Revisit when |
| --- | --- | --- |
| Embedded MissionControl at `.mc/` | Keep plans with the project and preserve `tasks/` for robot tasks | Repository workflow changes |
| MuJoCo-first on the available Mac, stabilized G1 tabletop | Explicit user constraint; local simulation and training before other platforms | Future Isaac/hardware access |
| Native JEPA + LeWM first | Native reference plus one external backend meets MVP scope | TASK-002 finds a compatibility blocker |
| JEPA-WMs optional research follow-up | Keep restrictive upstream components out of required core | Separate license/usage review |
| Image goals and shared CEM/MPC | Direct implementation of the PRD comparison contract | **Revisited 2026-09-24**: CEM over the world-model cost is no longer the primary control line — see the pivot below |
| One grasp synergy per hand initially | Limit action complexity while preserving both hands in the API | Grasp coverage proves insufficient |
| Local planning before environment installation | Mac is confirmed; dataset, exact assets, MPS operator support, and future robot access need validation | M0 readiness inventory |

A candidate starting point for `native_jepa` is a compact RGB encoder, robot-state/action conditioning, latent dynamics predictor, an EMA target encoder, and explicit variance/covariance regularization. This is a project proposal, not an assertion about LeWM's loss. TASK-011 must test collapse prevention and recursive rollout behavior. No pixel decoder is required.

## Open decisions with resolving tasks

The status column was swept on 2026-09-24. The original assumptions are kept as written so
the record shows what was assumed before each was resolved.

| Unknown | Assumption as recorded | Resolve in | Status |
| --- | --- | --- | --- |
| Local compute | Observed arm64 macOS 26.5.1, 48 GiB memory; MPS support and free storage still to measure | TASK-001, 003 | **Resolved** — measured in [RESOURCES.md](RESOURCES.md) |
| Actual G1 EDU4 joint/hand/camera configuration | Dual Dex3 as specified, precise calibration pending | TASK-001, 008 | **Simulation only** — MJCF verified in [MUJOCO_SPIKE.md](MUJOCO_SPIKE.md); physical calibration still unset |
| Existing demonstrations and rights | No usable dataset assumed yet | TASK-001, 010 | **Resolved by collection** — all corpora are locally collected simulation data; no external demonstrations |
| Python/framework versions | Choose macOS arm64 compatible pins after MuJoCo and LeWM spike | TASK-002, 003, 005 | **Resolved** — pinned in `uv.lock` |
| External checkpoint suitability | Train on G1 canonical data; no transferable checkpoint assumed | TASK-002, 015 | **Resolved as assumed** — no upstream pretrained weights are used |
| Action frequency/scales and IK implementation | Unset until tested in simulation | TASK-004, 008 | **Resolved for simulation** — `configs/g1_sim_action.json`; hardware values remain unset |
| Owners, staffing, delivery date | Unassigned; effort ranges only | Assign when execution starts | Still open |
| Isaac and physical execution | Future ports; local preparation now, commissioning when resources exist | TASK-022, 025, 026 | Still open — TASK-025/026 are in the backlog |

## Risk register

| Risk | Early signal | Mitigation / task |
| --- | --- | --- |
| Upstream data/actions mismatch | Adapter needs future state or hardcoded action dimensions | Canonical-batch spike before full integration; 002/004 |
| MuJoCo G1/Dex3 asset is missing or differs from EDU4 | Joint ordering/camera/hand mismatch | Audit or compose MJCF/meshes and verify dual-hand collisions, inertias, and actuators; 003/008 |
| Representation collapse | Low loss but near-zero latent variance or no action effect | Regularization and action ablations; 011/014 |
| Planner exploits model errors | Predicted goal improves while physical outcomes regress | Short horizon, bounded actions, diverse failure data; 010/013 |
| Dexterous grasping exceeds pilot coverage | Reach works but stable lift fails | Scripted controller baseline, expand grasp data; 018 |
| Mac memory, MPS support, or latency limits | Candidate batches fail or MPC misses deadlines | Adapter chunking, measured image resolution/horizon; 012/017 |
| Simulation-to-real mismatch | Calibrations, dynamics, or camera shifts | Replay, mocks, staged commissioning; 021/022 |
| Evaluation leakage | Shared sessions/goals enter training | Versioned split manifests and held-out combinations; 007/019 |
| Unclear asset/checkpoint terms | No explicit license tied to exact artifact | Track independently; keep optional and unresolved until verified; 002/023 |

Negative research outcomes should generate a diagnosis and reproducible report. They should not be hidden by changing splits or relaxing outcome definitions after evaluation.

## Apple goal-alignment investigation (TASK-041)

The H16 sensor model passed matched action prediction, while committed control and arrival-triggered replanning failed to produce manipulation. These results do not identify one unique cause. Before another physical controller variant, test whether an image-only goal estimate provides useful arm-position information on held-out parent sessions and whether that information survives frozen model forecasts.

Compare a TRAIN mean, TRAIN-only image retrieval and one small neural encoder. Retrieval is a legitimate learned nonparametric candidate; a neural head failing to outperform it does not by itself reject goal alignment. Neither candidate may receive goal joint values or frame identifiers at inference. Measured positions are TRAIN supervision and VAL assessment labels only. Preserve the frozen image-goal contract and existing model checkpoints.

This is an offline feasibility screen for seven named arm positions, not a complete manipulation representation: the same arm pose may correspond to a held or dropped apple. Even a passing result requires a separately reviewed visual-plus-pose cost, TRAIN-only calibration and matched physical model ablations before any final acceptance attempt. Keep the original pixel metric as a frozen baseline. The versioned TASK-041 protocol will specify sampling, metrics and stopping rules before fitting.

The planning-time [data audit](experiments/apple_goal_alignment_data_audit.md) confirms 26 TRAIN and three VAL original parents, with all VAL parents using the closure-burst collection variant. It also finds saved examples with nearly identical arm positions but substantially different apple heights. These metadata/state inspections inform the protocol and are disclosed; they are not prospective model-evaluation outcomes or goal inputs.

**Overtaken.** TASK-042's combined visual-and-pose cost reached 77.68% against the
preregistered five-point gate (a 4.30-point gain) and stopped before physical control; see
[apple_aligned_control_results_v1.md](experiments/apple_aligned_control_results_v1.md). The
line this investigation belongs to — image-goal costs consumed by a sampling planner — was
subsequently abandoned as the primary control line by the decision below. The record above
is kept as written.

## Decision 2026-09-26 — the in-corpus encoder-training line is closed (TASK-062)

**Decision.** This applies the clause that `apple_encoder_study_v1.md` §10 pre-declared for the
rows O-ENC-ARCH and O-ENC-NONE. The run ended in O-ENC-ARCH.
- **What is closed:** the line "train a LeWM-family encoder on `apple-wide-v1` train-split frames
  so that its frozen features expose the post-look apple". No further readout-point,
  regularisation, objective or supervision variant of the TASK-054 recipe is preregistered on this
  corpus without new evidence of a different kind.
- **Recorded as untested:** capacity, input handling and the shared distribution confound.
- **Not abandoned:** the LeWM backend, the encoder as a component, and the product goal.
- **The look-prefix corpus is not collected.** The owner's precondition, a frozen encoder that
  exposes the post-look apple beyond its random init, is not met.
- **Recommended next task (the owner chooses):** one preregistered test of an externally
  pretrained frozen encoder against its random-init floor, on the same probe.

**Evidence.** [apple_encoder_study_v1_results.md](experiments/apple_encoder_study_v1_results.md),
with manifest `benchmarks/manifests/apple-encoder-study-v1-results.json`.
- Four cross-fitted arms were tested on held-out roots. None passed, and all were evaluated.
- A-tok (the recipe's patch tokens) met every bar: 0.463 cm [0.422, 0.536], 178/190. But it did
  not beat its random-init token floor F-tok, which reached 179/190 by itself.
- The three pooled-latent changes (image-feature SIGReg, pixel reconstruction, readout heads off)
  all failed the bars.

Learned Apple→Plate remains at 0 successes.

## Decision 2026-09-25 — the behaviour-cloning abandonment clause fired (TASK-057)

**Decision.** This applies the clause that `apple_policy_v1.md` §7 pre-declared and
`apple_policy_diagnostics_v1.md` §5.1 wired to G-SUB. Its consequence applies as written,
as quoted in the results §6:
- **This behaviour-cloning line stops on this corpus (`apple-wide-v1`) and this camera (112 px
  onboard).** No third control formulation is preregistered on them, and no further loss, head
  or output-parameterisation variant either.
- The next task is a perception/data task: resolution, camera placement or corpus design. If that
  task does not move the closed-loop number, the §7 conclusion is that the product goal needs a
  data or hardware change, not another model. The task owner selects the task; the executing
  agent does not.
- Cohort C remains unconsumed, and `exemption_spent` remains `false`.

The product goal is unchanged: LeWM on G1 + dual Dex3. So are the LeWM backend, the encoder and
the backend-swap invariant.

**Evidence.** [apple_policy_diagnostics_v1_results.md](experiments/apple_policy_diagnostics_v1_results.md),
with manifest `benchmarks/manifests/apple-policy-diagnostics-v1-results.json`.
- B1, B2 and B3 passed. B3 reached 16/16. B2 reproduced TASK-056's A2 report exactly.
- D1-pipeline passed for all four arms. The images were byte-identical for A1–A3 (A0 takes no
  image), and the state difference was 0.0. D1-grasp fired for none. So clause (a) does not
  hold, within D1-pipeline's stated scope.
- G-SUB failed on the non-gating development cohort, n = 16 per configuration. The best candidate
  reached 1/16, against thresholds of 8–12. These are privileged-substitution diagnostics.
- With A2 as the complement, every candidate scored below the same substitution with a
  clock-only complement, by 1–9 attempts. That is a single attempt on `dz`, `dpitch` and
  `grasp`, and 9 only on `dy`.
- The result is Outcome X.

## Decision 2026-09-24 — abandon CEM over this world-model cost as the primary control line (TASK-054)

**Decision.** Sampling-based planning (CEM/MPC) over the learned world model's cost is
**abandoned as the primary control line**. Behaviour cloning with the world model as a
critic or residual becomes the primary line. No further predictor-architecture protocol is
preregistered. This applies the abandonment clause that
[apple_world_model_v4.md](experiments/apple_world_model_v4.md) pre-declared as Outcome B,
before any v4 number was seen; the clause was recorded as TRIGGERED by TASK-052 and
deferred by exactly one protocol, which was v4.

**Evidence.** [apple_world_model_v4_results.md](experiments/apple_world_model_v4_results.md),
manifest `benchmarks/manifests/apple-world-model-v4.json`, merged as `c5ec88c`.

- All four v4 arms **failed** the 14 preregistered gates: E0 control 10/14, E1 action-chunk
  9/14, E2 tail-weighting 9/14, E3 step-embedding 9/14. The untouched control passed more
  gates than every intervention.
- The primary gate **G2a** (rollout readout error ÷ the model's own persistence readout,
  threshold ≤ 0.8) came in at **0.8763 / 0.8814 / 0.8635 / 0.9036**. No arm passed, and
  none came close. G2a has never passed: 0.835 at v2, 0.831 at best in v3.
- The error decomposition (median, moving windows, h = 8, cm; encoded target / rollout /
  excess) shows the encoder improving while the rollout term did not, with the excess
  roughly tripling between v2 and v4: v2 3.26 / 3.65 / 0.39; E0 2.5903 / 3.6481 / 1.0578;
  E1 2.5403 / 3.6579 / 1.1176; E2 2.5723 / 3.4754 / 0.9031; E3 3.6130 / 4.4087 / 0.7957.
- Episode-clustered paired bootstrap (20,000 resamples, seed 20540) against E0 on the
  rollout term: E1 **+0.010 cm [−0.407, +0.367]** (crosses zero), E2 **−0.173 cm
  [−0.804, +0.132]** (crosses zero), E3 **+0.761 cm [+0.155, +1.199]** (excludes zero —
  E3 damaged the encoder). No interval is available for the G9 rollout-excess contrast;
  the bootstrap field is a different estimand.
- **Five attempts to close the rollout gap have now failed**: a second camera (v3, made
  readout precision worse), motion-weighted readout shaping (v3, no help and it hurt
  candidate ranking), action chunking (v4 E1, no effect), tail weighting (v4 E2, a 14.6%
  reduction of the excess where 49.9% was required, and not distinguishable from cohort
  sampling), and a non-shared per-step predictor (v4 E3, damaged the encoder).
- All three v4 interventions also **cost candidate ranking** — G6a 0.5438 for the control
  against 0.2865 / 0.3296 / 0.4141 — and the control is the only v4 arm that passes the one
  metric a CEM directly needs.

**What is explicitly NOT abandoned.**

- **The LeWM backend.** Both backends and the one-line backend swap are unchanged, and the
  next line runs on the same LeWM backend.
- **The encoder.** It is the component that works and is still improving (palm–apple offset
  read to 2.35–2.59 cm, against 3.13 cm at v2), with no representation collapse (effective
  rank 6.74–8.23). It is unfinished, not solved: 2.5903 cm alone exceeds the 1.5 cm G1
  threshold.
- **The product goal.** LeWM controlling G1 + dual Dex3 is unchanged. Only the control
  formulation changed.
- **The CEM/MPC implementation, the frozen benchmark and the result schema.** They stay in
  the repository, under test, as the shared planner and evaluation contracts. Nothing was
  deleted and no historical result was revised.

**What this does not establish.** That the prediction step cannot be fixed. v4 tested three
specific designs at one seed and one budget. What it establishes is that the pre-declared
decision rule's condition was met and the rule is being followed rather than renegotiated
after the fact. Learned Apple→Plate remains at 0 successes.

**Erratum 2026-09-25 (TASK-058).** The decision stands. Its trigger, G2a, is correctly computed,
and neither a copy-last nor a constant predictor passes it. The entry text above is kept as
written. The audit [claim_audit_v1.md](experiments/claim_audit_v1.md) corrects how several of
the supporting statements are read:

- **"The untouched control passed more gates than every intervention" (S4-09).** The count is
  right, 10 against 9. The difference is G6a at h = 16. At least three passes that every arm
  shares are also cleared without a learned predictor: G4 by a constant zero-height
  predictor, and G3 and G5 by copying the true simulator state forward.
- **"All three v4 interventions also cost candidate ranking" (S4-10).** This holds at the gated
  h = 16 only, on one seed and with no interval. At the planner's h = 8 (10 groups), E1 has the
  highest ρ (0.50 against E0's 0.36).
- **"The excess roughly tripling between v2 and v4" (S4-12, S3-07, S2-08).** The excess is the
  difference between two medians. The v2 figures (3.26 / 0.39 cm) were never committed; a
  TASK-058 re-run reproduces them. The median of the per-window excess is 0.47–0.53 cm in all
  four v4 arms, including the control.
- **"E3 +0.761 cm … E3 damaged the encoder" (S4-17).** The interval quoted is the rollout
  contrast. The encoder evidence is the encoded-target contrast, +1.023 cm [+0.485, +1.435].
- **"Motion-weighted readout shaping (v3, no help and it hurt candidate ranking)" (S4-24).** This
  was one bundled factor: motion weighting together with auxiliary position targets. "Hurt
  ranking" is a lower G6a on one seed.
- **"Palm–apple offset read to 2.35–2.59 cm, against 3.13 cm at v2" (S4-18, S2-08).**
  - The figure is correctly an encode-then-readout number, not a rollout.
  - It covers the moving validation windows only. That split was also used for checkpoint
    selection.
  - The encoding includes fused proprioception.
  - 2.35–2.40 cm is the start frame and 2.54–2.59 cm the target frame.
  - The v2 3.13 cm is carried over, not recomputed. It is also not citable as recorded.
