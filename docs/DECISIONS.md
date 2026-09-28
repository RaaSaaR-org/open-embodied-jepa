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

## Decision 2026-09-28 — apple-to-plate-v2 is defined, and its privileged expert passes the at-rest gate (TASK-069, TASK-070)

**Decision.** `apple-to-plate-v2` is defined by owner ruling R12, which is recorded verbatim in
[apple_to_plate_v2_expert.md](experiments/apple_to_plate_v2_expert.md) §0.
- **Definition:** v1 plus the apple's contact at condim 6, with the v1 scene's own declared
  friction `1 / 0.01 / 0.001`. Nothing else changes.
- **Friction source (R12 §3 (i)):** the values were declared before any scan and were not chosen
  from it. No other value was tried under v2.
- **Code:** `src/embodied_jepa/apple_to_plate_v2.py`, scene version `apple_to_plate_v2`, applied at
  run time.
- **v2 is a separate benchmark from v1.** v1, its scorer, its history and the 0/150 MVP benchmark
  are unchanged. `simulation.py`, `task.py` and `scripted.py` are untouched.
- **What R12 rejected:** the TASK-069 scan's b2 option (plates moved into reach) and b1 option
  (the waist in the IK), and choosing friction values from the scan.

**The expert gate passed.** TASK-070's preregistered gate was PR #92. The run was made on the
pre-run reviewer's reported GO, from `main` at `1ba0557`.
- **The expert:** the privileged scripted expert e9, `RestingPlaceExpert(release_pitch_rad=0.45,
  release_dx=0.015)`.
- **The result on 32 fresh gated seeds, 50600–50631:** at rest (`apple_at_rest_v0`) on **32/32**
  with the plate exact and **30/32** at 1.0 cm plate error. The bar was 28/32 at each gated level, so
  the row is **PASS**.
- **Reported alongside:** 28/32 at 1.5 cm, 32/32 latched at every level, and 0 guard stops.
- **The six misses** all rested 4.02–4.61 cm from the plate centre, outside the 4 cm radius.
- **Qualifiers:**
  - e9 was selected on the development seeds (50200–50295). Only the 32 gated seeds are
    independent evidence for it.
  - The sample is small.
  - v2's friction values are the scene's declared ones, not measured apple data.

**What this is not.** It is a scripted-expert result under privileged truth, not a learned one.
**Learned Apple→Plate is still 0 successes.** R12 §5 names the next step: a task that trains the
first learned policy on this expert's v2 demonstrations. That task is the owner's to open.

**Evidence.**
- [apple_to_plate_v2_expert.md](experiments/apple_to_plate_v2_expert.md) §4 (the development
  log), §5 (the frozen gate) and §8 (the results, with SHA maps).
- The report sha256 is `27543757…099f`.
- [apple_to_plate_v2_feasibility.md](experiments/apple_to_plate_v2_feasibility.md) (TASK-069).

## Decision 2026-09-28 — TASK-068 closes on its development finding: under the frozen v1 task the privileged expert cannot rest the apple on the plate (TASK-068)

**Decision.** Owner ruling R11 closes TASK-068 on its development finding, with no gated run.
- **Option (a) is declined.** No gated run is made to put an expected failure on record.
- **The development log is the record.** The privileged scripted expert rested the apple on the
  plate in **0 of 269 completed development attempts**, across 15 design-by-level cells (304
  attempts; 35 ended early on the joint-velocity guard). The development seeds were
  50000–50099.
- **The at-rest check** is `apple_at_rest_v0` (`src/embodied_jepa/at_rest.py`). It reads 20
  steps at the end of a 60-step settle. On every one of them the apple must be within 4 cm,
  supported, moving at ≤ 0.001 m/s, and free of hand contact.
- **Next:** TASK-069, a development-only feasibility scan toward an `apple-to-plate-v2` task
  definition. The v1 task, its history and the 0/150 benchmark stay untouched and separately
  named.
- **Learned Apple→Plate is still 0 successes.** This is privileged scripted engineering, not a
  learned result.

**Why, under the frozen v1 task.**
1. **The place pose is beyond the fixed-pelvis arm's reach.**
   - Measured: no right-arm configuration within the joint limits brings the palm within
     3.1 cm of any of the eight place targets, whatever the palm orientation (the smallest
     reachable distance is 3.1–6.2 cm per target).
   - The pelvis is fixed, and the IK uses the 7 arm joints only.
   - Measured: when the hand starts to open, the held apple is 8.0–23.2 cm above its resting
     height.
2. **The apple, a sphere with no rolling resistance in this scene, keeps rolling.**
   - Measured: a 0.002 m/s roll persisted unchanged for 10 s.
   - Measured in the traces examined: the hand's opening rolls the apple off the thumb. No development cell's median
     landing speed was below 0.08 m/s.
   - Measured: most grasped apples end rolling at or along the rim, 4.5–4.6 cm from the centre.

**The latch, again.** The best design, d12, scored 32/32 and 29/32 on the latched scorer, and
0/32 at rest at both levels. Latched scripted counts are not evidence of an apple left on the
plate.

**Evidence.**
- [apple_resting_expert_v1.md](experiments/apple_resting_expert_v1.md):
  - §4, the descent diagnosis;
  - §5, the development log;
  - §8, R11 verbatim and the map from the commit SHAs cited before the merge to the SHAs on
    `main`.
- The tag `task068-dev-evidence-pre-merge` (`a8004c8`) keeps the SHAs cited before the merge
  reachable.

## Decision 2026-09-28 — TASK-067 closes as CAL-ESCALATE; the scripted expert does not rest the apple on the plate, and the scorer latches transient crossings (TASK-067)

**Decision.** Owner ruling R10 closes TASK-067 under the R9 fallback, with outcome
**CAL-ESCALATE**:
- **The gated run** stopped at the C0 calibration before any policy was trained. The privileged
  scripted expert scored 21/32 at 1.0 cm plate error, below the 28/32 bar.
- **The release-point probe** ended **P-CANDIDATE-FAIL**.
- **The one place redesign** ended **FAIL, with its premise untested**. The hand never reached
  the place pose within its 100-command budget. It was still descending at about 0.44 mm/step
  with the command saturated, and it made no hand–plate contact during `lower_closed`. Why the descent is so slow is
  not identified.
- **No abandonment clause fires.** The task, the plate geometry, the scorer radius and the 28/32
  bar were not changed.
- **Next:** the follow-up task, the expert's placement and success at rest, goes to a new agent.
- **Not abandoned:** the LeWM backend, the encoder as a component and the product goal.
- **Learned Apple→Plate is still 0 successes.**

**Two findings that matter beyond this task.**

1. **The scripted collector does not rest the apple on the plate.**
   - It opens the hand at the transfer height, so the apple falls about 15.5 cm and lands at
     about 1.28 m/s.
   - The apple picks up horizontal velocity while the hand opens. It is carried and rolls to the
     rim, touching it on 32/32 attempts.
   - After a 60-step settle it is outside the 4 cm radius, slowly rolling along the rim at
     4.52–4.60 cm from the centre, on most resets.
   - **At rest the count is 4/32 with the plate exact, and 1/32 at 1.0 cm plate error.** This is
     measured on the 32 spent probe seeds 46800–46831.
   - A rim-supported apple sits about 4.6 cm from the centre, so by geometry it always fails the
     4 cm radius.
2. **The success scorer latches on transient crossings.**
   - How `AppleToPlateTask` (thresholds `tabletop_proxy_v0`) counts success:
     - its per-step `success` needs 0.15 s inside 4 cm, supported, at ≤ 0.1 m/s and with no hand
       contact;
     - its `place` and `release` stages latch the first such window;
     - runners that stop at the first success, such as C0 and the probes, count the same way.
   - A slowly rolling apple that crosses the disc for 0.15 s therefore counts as a success.
   - On the same attempts, success at any step was 32/32 and 21/32, against 4/32 and 1/32 at
     rest.
   - **Scripted-collector success counts measured with this scorer therefore overstate how often
     the apple actually ends on the plate.**

**Past results are not rewritten.**
- **Flag:** scripted-collector and privileged success numbers elsewhere in this repository,
  where they were scored with `AppleToPlateTask`, were measured with this latching scorer. An
  example is the 103/200 root successes in `apple-look-v1`, which were. Read such numbers as
  "reached the plate under the latch", not "rested on the plate".
- **Not audited or re-measured here:**
  - which past numbers were scored with `AppleToPlateTask`;
  - which of those include transient crossings.
- **Learned counts are unaffected.** A latch can only over-count, and the learned count is 0.

**Evidence.**
- [apple_first_policy_v1_results.md](experiments/apple_first_policy_v1_results.md): C0 and
  CAL-ESCALATE, with the dated erratum in §4.2.
- [apple_first_policy_v1_release_probe.md](experiments/apple_first_policy_v1_release_probe.md):
  P-CANDIDATE-FAIL, with R8 and R9 verbatim.
- [apple_first_policy_v1_landing_diagnosis.md](experiments/apple_first_policy_v1_landing_diagnosis.md):
  - §A, the diagnosis, with its dated correction;
  - §B–§C, the redesign and its FAIL row;
  - §D, R10 verbatim.
- Artifact report sha256:
  - `outputs/task067-landing-diagnosis/run-1`: `d9966669…1f46`;
  - `outputs/task067-place-probe/run-1`: `a5af156c…acb9`.

## Decision 2026-09-28 — the patch-token predictor passes its world-model gates on the train split (TASK-066)

**Decision.** `apple_token_dynamics_v1.md` §11 row **WM-TOK-DYNAMICS** matched: every seed passes
G1–G5 at h = 8 and h = 16, and the encoded 4 × 4 grid meets the T1 bar. **No abandonment clause
fires.** The row records a result and does not choose a line.
- **What it says:** on `apple-look-v1`, on 170 cross-fitted train sessions, the pinned upstream
  LeWM predictor over frozen DINOv2 patch tokens pooled to 4 × 4 learns action-conditioned latent
  dynamics that pass a calibrated no-collapse gate, beat copy-last and an equally budgeted
  no-action predictor, use the actions, and keep the post-look apple readable through 16-step
  prediction.
- **Recommended next task (the owner chooses):** a separately preregistered **held-out
  confirmation on the corpus's test split** (never decoded until then). Fix the saturation caveat
  first, with a budget or schedule under which N reaches a plateau on train/val, and keep this
  task's gates and bars unchanged.
- **Not implied:** any control formulation. Control use needs its own preregistration and must
  answer the TASK-054 and TASK-057 clauses.

**Evidence.** [apple_token_dynamics_v1_results.md](experiments/apple_token_dynamics_v1_results.md),
with manifest `benchmarks/manifests/apple-token-dynamics-v1-results.json` (run-1 report sha256
`e6b28e07…0b60`).
- G1: rank ratio 0.325–0.344 (bar 0.16), std ratio 0.82–0.84 (bar 0.39), and W − N rank lower
  bounds 0.038–0.076 (bar > 0).
- G2: W / copy-last upper bounds 0.790–0.792 at h = 8 and 0.652–0.654 at h = 16 (bar ≤ 0.8).
- G3: W / N upper bounds 0.915–0.916 at h = 8 and 0.873–0.876 at h = 16 (bar < 1.0).
- G4: wrong / W and zero / W lower bounds 1.67–2.14 (bar ≥ 1.10).
- G5: W reads the apple at 0.75–0.93 cm (median CI upper bounds ≤ 1.03 cm), against 0.63 and
  0.50 cm encoded. The ratio-to-B-occ
  upper bounds are 0.469–0.581 (bar 0.6), and the excess-over-encoded upper bounds 0.335–0.423 cm
  (bar 0.5).

**Caveats, stated plainly.**
1. **The budget did not saturate.** 9 of 12 models selected one of their last two checkpoints,
   and every N model did. An under-trained N may bias G3 and G1 (iii) towards W.
2. **Narrow margins.** G2 at h = 8 (upper bounds 0.790–0.792 against 0.8), and G5 for seed 2 at
   h = 8 (ratio upper bound 0.581 against 0.6).
3. **The token rank ratio (0.325–0.344) is not better than TASK-065's CLS ratio (0.365–0.399).**
   G1 passes because its bar is calibrated, not because tokens lose less rank.
4. **Train split only; no control claim.** One corpus, a static apple, 0.4 s and 0.8 s horizons.

This is a world-model test only. Learned Apple→Plate remains at 0 successes.

## Decision 2026-09-27 — the predictor-on-frozen-pooled-CLS line is closed (TASK-065)

**Decision.** This applies the clause that `apple_latent_dynamics_v1.md` §10 pre-declared for
the rows WM-APPLE-LOST and WM-NO-DYNAMICS. Run-2 ended in **WM-NO-DYNAMICS**.
- **What is closed:** the line "an action-conditioned LeWM-family predictor on frozen,
  externally pretrained **pooled (CLS)** latents on `apple-look-v1`". No further predictor
  variant on this corpus with frozen pooled pretrained latents is preregistered without new
  evidence of a different kind: not history length, action chunking, step embeddings, loss
  weighting, capacity, residual parameterisation or budget.
- **Not abandoned:** the LeWM backend, DINOv2 as an encoder, patch-token latents, the product
  goal, and the corpus (sealed; test split unread).
- **Recommended next task (the owner chooses):** a preregistered patch-token latent predictor
  (P-tok, in the style of DINO-WM) on `apple-look-v1`. Its training budget and any rank/collapse
  bar are to be calibrated on train/val before the freeze.

**Evidence.** [apple_latent_dynamics_v1_results.md](experiments/apple_latent_dynamics_v1_results.md),
with manifest `benchmarks/manifests/apple-latent-dynamics-v1-results.json`.
- **The row name overstates the failure.** G2–G4 passed on all three seeds at h = 8 and 16:
  - the predictor beats copy-last (0.746 at h = 8, 0.647–0.652 at h = 16);
  - it beats an equally trained no-action predictor (0.879–0.910);
  - wrong actions cost it 1.78–1.98× the error.
- **It fails the preregistered no-collapse rank gate and the apple-readability gate.**
  - **G1:** the effective-rank ratio is 0.365–0.399 against an uncalibrated bar of 0.5; std ratio
    and collapsed fraction pass.
  - **G5:** the apple readout from predicted latents is 0.94–1.19 cm against 0.68–0.83 cm encoded.
    The non-inferiority margin fails at h = 16 on every seed.
- **Why this row:** WM-APPLE-LOST needs G1–G4 on two seeds, so WM-NO-DYNAMICS matched. Both rows
  fire the clause.
- **Run-1 was V at G-cache.** A float32 batch-size effect caused it, before any model existed. The
  guard was amended by owner ruling under §15.5, beyond §12's "runner mechanics" limit, before any
  outcome data existed. Run-2 was the single repeat.

This is a world-model test only; no control formulation is preregistered or implied. Learned
Apple→Plate remains at 0 successes.

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
