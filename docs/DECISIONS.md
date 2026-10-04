# Decisions, assumptions, and risks

## Working decisions

| Decision | Rationale | Revisit when |
| --- | --- | --- |
| Embedded MissionControl at `.mc/` | Keep plans with the project and preserve `tasks/` for robot tasks | Repository workflow changes |
| MuJoCo-first on the available Mac, stabilized G1 tabletop | Explicit user constraint; local simulation and training before other platforms | Future Isaac/hardware access. **Revisited 2026-10-02**: the Linux PC (RTX 5080) is the working platform since TASK-072 (2026-09-28) and the Mac is the archive; MuJoCo stays the reference simulator, and Isaac/Arena exist only as development cross-sim checks (TASK-025) |
| Native JEPA + LeWM first | Native reference plus one external backend meets MVP scope | TASK-002 finds a compatibility blocker |
| JEPA-WMs optional research follow-up | Keep restrictive upstream components out of required core | Separate license/usage review |
| Image goals and shared CEM/MPC | Direct implementation of the PRD comparison contract | **Revisited 2026-09-24**: CEM over the world-model cost is no longer the primary control line — see the pivot below |
| One grasp synergy per hand initially | Limit action complexity while preserving both hands in the API | Grasp coverage proves insufficient |
| Local planning before environment installation | Mac is confirmed; dataset, exact assets, MPS operator support, and future robot access need validation | M0 readiness inventory. **Revisited 2026-10-02**: the working platform is now the Linux PC with CUDA (TASK-072); the Mac remains supported |

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
| External checkpoint suitability | Train on G1 canonical data; no transferable checkpoint assumed | TASK-002, 015 | **Resolved as assumed** — no upstream pretrained weights are used. **Revisited 2026-10-02**: frozen, externally pretrained DINOv2 ViT-S/14 weights are used since TASK-063 (`scripts/fetch_dinov2.py`); NVIDIA's GN1x-Tuned GR00T N1.7 release (step 65000) is used only as an external reference baseline in Arena (#121), never as a project model |
| Action frequency/scales and IK implementation | Unset until tested in simulation | TASK-004, 008 | **Resolved for simulation** — `configs/g1_sim_action.json`; hardware values remain unset |
| Owners, staffing, delivery date | Unassigned; effort ranges only | Assign when execution starts | Still open |
| Isaac and physical execution | Future ports; local preparation now, commissioning when resources exist | TASK-022, 025, 026 | Still open — TASK-025/026 are in the backlog. **Revisited 2026-10-02**: Isaac (PhysX and Newton) and Isaac Lab-Arena run as development cross-sim checks under TASK-025, not admitted as benchmarks; physical execution is still open (TASK-026, `hardware.py` mock-only) |

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

## Decision 2026-10-04 — Branch B after cell A's removal: a DRAFT design note for the next LeWM task (R9; stub)

**Decided by Claude under owner delegation (2026-09-30).** No run was made for it, and nothing in
it is frozen. It answers PLAN.md's Branch B entry for PRED-INFEASIBLE (TASK-076's Stage 0 removed
cell A, R8.16). The note is
[apple_lewm_next_v2_design.md](experiments/apple_lewm_next_v2_design.md) (DRAFT). This entry is a
stub: each ruling is provisional until the note's review, and none changes TASK-076.

- **R9.1 — form.** The next LeWM step is first a docs-only design note with three candidates, not
  a protocol. No task number is assigned and no MC task is created until a candidate passes its
  feasibility record.
- **R9.2 — recommendation: C2**, a single committed pre-pick push of a free plate into P-3's zone,
  with LeWM ranking candidate pushes by their predicted plate rest positions (one roll-out per
  candidate). Reasons: its consequence comes from contact physics, so no hand-written arm is
  handed the answer by construction; the decision is the first controlled act and the remainder
  is the v2 episode P-3 already solves; the plate is at rest at the decision, so history one
  suffices; it lies outside every closed scope (TASK-054/057/062/065/075).
- **R9.3 — a design requirement drawn from cell A** (note §2). A rule linear in palm velocity is
  path-independent and its fixed point is a closed form, so a hand-written arm that knows any
  declared kinematic rule ties; and re-aiming at the current target is itself a fixed-point
  solver. A world model earns a decision only if its consequence depends on the action, comes
  from physics the controller is not handed as a formula, is committed before it can be observed
  and corrected, and still decides the episode.
- **R9.4 — C3 (a contact-sensitive grasp approach) is not taken**; its tolerant pick would need a
  tuned disturbance to open headroom, and its quantity is the apple, this project's weakest
  readout. **C1 (an early committed aim under cell A's rule)** is kept as a fallback only if C2
  ends PUSH-INFEASIBLE for engineering reasons, with its expected H-rule tie written into its
  claim row in advance.
- **R9.5 — the free plate is a declared task change** (a scene variant separate from v2, as v2 is
  from v1, declared under R2). Its physical constants (plate mass, the scene's own plate friction)
  and the candidate grid are fixed before any smoke that reads headroom and are never tuned on a
  headroom number. The single push is a declared restriction; a two-push feedback arm is reported.
- **R9.6 — the next step is a development feasibility record for C2** (C2-F0 to C2-F6 in the
  note; CPU, no world model, a newly declared smoke seed block). A preregistration follows only if
  C2-F4 shows a ceiling of at least 28/32 and a headroom of at least +8/32 over the best of H-kin,
  H-kin-cal and H-fixed; below +4/32 the line stops before any protocol; in between, a ruling.
- **R9.7 — scope.** TASK-076's primary question (K0, O, D, S/U) is unchanged and continues on its
  reviewer's GO. The LeWM backend, DINOv2 as an encoder, v2 and the product goal are not affected.

## Decision 2026-10-02 — after TASK-075: the plate-readout perception twin is next (TASK-076); what counts as a task change; one canonical status sentence

Six rulings, each **decided by Claude under owner delegation (2026-09-30)**. They follow the
TASK-075 clause (decision 2026-10-01 below), whose next step is "a task or condition change", and
the four options in §7 of [apple_obs_ceiling_v2_results.md](experiments/apple_obs_ceiling_v2_results.md).
No run was made for them.

- **R1 — the next task is Option 1, the plate-readout perception twin (TASK-076).** The place
  reads the plate position from the image, not the apple-minus-plate offset. The reported-only
  plate readout is 0.49–0.68 cm in median against τ = 1.0 cm (its 87.5th percentile reaches
  1.35–1.36 cm on the hand crop, so the tails matter), and the offset error comes from the apple
  term (2.2–3.1 cm). The closed loop (P-3's pick plus e9's place aimed at the frozen-DINOv2 plate
  readout, beside a true-plate ceiling and an image-free clock-prior control) runs no world model,
  so the TASK-075 clause does not close it. TASK-076 needs its own preregistration and review
  before any run; the preregistration is pending. It uses fresh development seeds, not cohort C.
  Option 3 (a larger plate or other τ relaxation) is not taken: a larger τ alone admits nothing
  under the frozen admission, and the image-free clock prior would then succeed too.
- **R2 — a later LeWM task on a plate target counts as a task change only if it is paired with a
  condition where the target must be predicted**, for example a plate that keeps moving during
  the place or one that leaves the view during the carry. With a static, visible plate a world
  model is not needed for the place (TASK-074 protocol §9b), so a LeWM plate-target task without
  such a condition is not admissible under the clause. Any such task must declare itself as a
  task change in its preregistration; it must not be presented as a silent reopening.
- **R3 — a new place primitive (Option 2) is a task change if it is declared as one.** Developing
  and measuring the primitive (with its own τ curve) is not a LeWM task and is not closed by the
  clause. A later LeWM task built on it must declare the change, and must use targets that an
  image-free prior cannot guess (the clock prior already reads 2.35 cm).
- **R4 — Option 4 (a trained readout head on the offset) is deferred.** It sits close to the
  closed TASK-062 in-corpus encoder-training line, and RBF kernel ridge, already nonlinear, reached
  only 2.30–2.52 cm. It is not preregistered now.
- **R6 — merged remote branches are not deleted.** Squash merges leave the run-provenance commits
  (run worktrees, pre-run review revisions, uncommitted-then-committed diagnostics) reachable only
  from those branches, and results documents cite them. Local topic branches may still be removed.
  This replaces "remove the merged topic branch when safe" in AGENTS.md for remote branches.
- **R7 — one canonical status sentence replaces "Learned Apple→Plate is 0 successes" repo-wide**
  in the entry documents (README, AGENTS.md, CLAUDE.md and the docs that restate the status).
  Frozen protocols and results documents are not rewritten. The sentence, checked against
  [apple_first_policy_v2_m2_results.md](experiments/apple_first_policy_v2_m2_results.md) and
  [mvp_results.md](experiments/mvp_results.md):

  > Learned Apple→Plate on the frozen v1 MVP benchmark (TASK-020) is 0/150 per backend
  > (`native_jepa` and LeWM). On `apple-to-plate-v2`, the behaviour-cloning/DAgger policy P-3 (an
  > MLP on a frozen DINOv2 readout, trained on demonstrations from the privileged scripted expert
  > e9; not a world model) scored 40/40 counted successes on the held-out cohort C against 39/40
  > for its random-init encoder control R-3 (one run, one training seed per arm, 40 resets, one
  > camera at 112 px onboard, a narrow reset distribution), so TASK-072 M2 is M2-FAIL on G3
  > (encoder pretraining contributed nothing measurable), and cohort C is no longer held out. No
  > LeWM-driven controller has run in closed loop on v2 yet; LeWM's only closed-loop Apple→Plate
  > runs are on v1, with 0 successes. Scripted-expert, privileged-ceiling, oracle and GR00T
  > successes are not project-learned results. (LeWM also ran in closed loop on
  > the TASK-014 development reach pilot, a reach task, not Apple→Plate: its v2 target-space
  > selector reached 1/5 goals, with intervals overlapping the 0/5 controls.)

(R5, any reuse of cohort C, is unchanged: it needs a new, disclosed protocol and the owner's ruling, as
the 2026-09-28 M2 decision says.)

## Decision 2026-10-02 (b) — TASK-076's scope: the perception twin alone cannot advance the LeWM goal, so a moving-plate headroom check is added (R8)

**Decided by Claude under owner delegation (2026-09-30).** No run was made for it. It refines R1
and does not replace it.

- **R8 — finding, stated plainly.** Option 1 (R1) runs no world model. With a static, visible
  plate, a world model is not needed for the place (TASK-074 protocol §9b). So an H-twin pass
  cannot be, or lead directly to, a LeWM-driven success. On its own it only calibrates things a
  later LeWM task needs: the plate ceiling on TASK-066's pooled 4 × 4 LeWM latent, τ re-measured,
  and the perception baseline.
- **R8 — ruling.** TASK-076 keeps Option 1 as its primary, gated question and adds **Stage
  K-pred**: a simulator-only check, with no world model, of whether a moving-plate condition on v2
  leaves room for prediction. The plate keeps moving after the pick and stops after the place
  target freezes. The check compares aiming at the plate's final position, its current true
  position, its current read position, and extrapolations of the readings. K-pred's row decides
  the next LeWM task:
  - PRED-ADMIT means TASK-077, a LeWM plate-prediction place planner, is declared as a task change
    under R2.
  - Otherwise PLAN.md's Branch B applies.
- **R8.1–R8.6 — rulings on the independent review of #129** (REQUEST CHANGES at `314d843`). Each
  was decided by Claude under owner delegation.
  - **R8.1.** Entry documents quote R7's canonical sentence verbatim and in full. A shortened
    version must not carry the "canonical" label.
  - **R8.2.** c_plate, the ceiling that bounds a later LeWM task's bar (c_plate ≤ B ≤ τ_re), is
    measured on the representation that task would read. That is TASK-066's pooled 4 × 4 latent
    (R-plate-pool). The full-token R-plate stays as H-twin's readout only.
  - **R8.3.** K-pred gets an action-dependent cell, A. In it, the plate's velocity is κ = −0.5
    times the robot's own palm velocity L steps earlier, from 405 to 525 (L = 40 at
    first; L = 2 since R8.7). This is a declared,
    simple, simulator-only rule. **Only cell A can admit TASK-077.** The definition of a LeWM-driven
    success requires beating an action-blind predictor, and with action-independent plate motion
    that predictor is expected to tie.
    - The constant-velocity cells are reported only. Their expected action-blind tie and H-cv tie
      are stated in advance.
    - If cell A proves infeasible in this scene at Stage 0, it is removed, and PRED-ADMIT is
      unreachable.
  - **R8.4.** Three changes so that a design failure or noise does not fire a clause:
    - PRED-INFEASIBLE (cell A removed, or its ceiling fails) escalates without a clause.
    - K0 stops early (CAL-ESCALATE) when H-stale(K) > 4/32, or when the in-sample H-clock(K) is
      within 2/32 of the ceiling (4/32 since R8.7).
    - TWIN-NEAR (G1 missed, but H-twin not detectably below H-handover) escalates without a
      clause. G1's power is stated: 0.59 at a true rate of exactly 87.5 %.
  - **R8.5.** The tail bar O2 becomes reported only. A tail bar of 2 τ_re did not come from a
    measured ceiling.
  - **R8.6.** τ_re uses the procedure of TASK-075's TAU_REMEASURE. That re-measurement was
    pre-committed only for OBS-ONBOARD and OBS-EXTRA, and its use for the plate readout is an
    extension, disclosed in the protocol.
- **R8.7 — rulings on the re-review of #129** (REQUEST CHANGES at `3d8560e`; one blocking
  finding). Each was decided by Claude under owner delegation.
  - **The flaw.** With L = 40, cell A's motion after the last decision was
    κ · [palm(485) − palm(445)], already fixed at 485. So action conditioning added nothing, and
    TASK-066's history-one predictor could not recover palm(445).
  - **The redesign.**
    - L = 2 steps, so the plate's motion after 485 is κ · [palm(523) − palm(483)], caused almost
      entirely by the aim chosen at 485. The aim that lands on the plate is the fixed point of
      "where the plate will be given the palm motion my aim causes".
    - The remedy for too little remaining motion lowers L (to 1) or moves s1 later. It never
      raises L.
  - **Required history, declared.**
    - The action-conditioned predictor needs one frame plus the candidate commands. TASK-066's
      history-one predictor qualifies if the missed 2-step term (at most about 0.85 cm in xy at
      the clip) is within τ_re. Otherwise TASK-077 must declare 3 frames or the last 2 executed
      commands.
    - The action-blind twin gets the same one frame and no actions.
  - **H-final(A)'s fixed point** iterates to |g_{k+1} − g_k| ≤ τ_re/4, at most 10 iterations.
    Non-convergence lowers H-final and so can make the row PRED-INFEASIBLE.
  - **Wording.** With κ < 0, the plate moves against the palm's velocity, back towards the
    approaching hand, so refused moves are expected. The Stage-0 smoke measures the remaining
    motion under H-final.
  - **K0's G2 stop** is now H-clock(K) ≥ N_K(0) − 4/32, superseding R8.4's 2/32. An exact
    one-sided McNemar test at p < 0.01 needs at least 7 one-way discordant pairs out of 64.
- **R8.8–R8.13 — freeze preparation** (the four non-blocking items of the third review of #129,
  APPROVE as a DRAFT at `ee3f3c9`, and a freeze-readiness check; 2026-10-04). Each was decided by
  Claude under owner delegation. No run was made for them; none changes a seed, an arm or a gated
  bar's value.
  - **R8.8 — an action-blind predictor can tie on cell A.** The draft's "an action-blind
    predictor cannot [find the fixed point]" is withdrawn. If the palm ends near the aim, the fixed
    point is about g\* = (p − κh)/(1 − κ), from the plate p and palm h in the current frame. An
    action-blind predictor predicts the mean over its training policy's aims; if those aims are
    centred on g\*, a controller that aims at one predicted plate ties. It is expected to lose
    only if the controller ranks candidate aims by their individual predicted outcomes, or the
    training aims are not centred on g\*. So PRED-ADMIT(A) admits TASK-077 only with three
    declarations made before its freeze: its controller form (which should rank candidates by
    their individual predicted outcomes), the spread of its training aims relative to g\*, and
    the action-blind twin's expected result.
  - **R8.9 — the history rule uses the measured 2-step term.** It supersedes R8.7's "qualifies if
    0.85 cm ≤ τ_re". TASK-066's history-one predictor qualifies for TASK-077 only if
    c_plate + m2\* ≤ τ_re, or B's calibrated allowance covers m2\*. m2 = |κ| ·
    ‖palm_xy(485) − palm_xy(483)‖ from executed joint states under H-final. The Stage-0 smoke
    measures it (an estimate); K-pred reports m2\*, the upper 95 % bound of its median on cell A's
    32 H-final attempts. The 0.85 cm clip bound is no longer used: the clip is on commands.
    c_plate is defined in O1's form (the upper 95 % bound of cross-fitted R-plate-pool's median
    error).
  - **R8.10 — the remedy's limits.** Lowering L to 1 changes almost nothing; moving s1 later helps
    only if the palm still moves in xy during the lower (505–555). κ is not a remedy, and any
    |κ| ≥ 1 is forbidden, because the look-ahead would stop converging. A shortfall removes cell A
    and gives PRED-INFEASIBLE (escalate, no clause).
  - **R8.11 — seeds, salts and resets completed.** Every cohort draws its reset with
    `condition_reset` (salt 7425). The salts' uses are listed (7603 also covers the closed-loop
    R-plate's λ; 7604 also covers m2\* and the G2 feasibility resampling). 7605 is reserved and
    unused. 7601–7609 appear nowhere in src, scripts, tests, configs or benchmarks on main
    (re-checked 2026-10-04).
  - **R8.12 — G2's predicted feasibility at K0.** Reported beside the −4/32 stop, from K0's paired
    H-handover (level 0) and H-clock outcomes: the exact one-sided McNemar p on the doubled
    discordant counts, and the fraction of 10 000 resamples of 64 pairs (salt 7604) with p < 0.01.
    It stops nothing; a predicted pass probability below 0.5 is disclosed in the freeze PR.
  - **R8.13 — rows completed.** U-VOID-CEILING (H-handover(U) < P-stale(U) − 2/32; escalate, no
    clause) keeps a ceiling failure on U from firing the clause through TWIN-HARM, so U keeps its
    H-handover arm. PRED-NOT-RUN (Stage O without O-PASS; no clause; Branch B) covers K-pred not
    running. K-pred runs one invocation per cell, so each cap is at least 5 times its estimate.
    No hand-crop variant follows a TWIN-PASS in TASK-076. The protocol's §13 lists every bar's
    source and every row's action, and what remains before the freeze (Stage-0 code and smokes,
    K0, then an independent APPROVE).
  - **R8.14 — a noise guard for the headroom bars** (2026-10-04, decided by Claude under owner
    delegation). New row PRED-NEAR: K-P1 passes and a headroom bar (K-P2, K-P3 or K-P4) fails,
    but every failed one's paired interval (§6's reset-clustered bootstrap, 95 %, salt 7604)
    includes +8/32. It escalates, with no clause and no claim, as TWIN-NEAR does for G1.
    PRED-NONE now needs a failed headroom that is detectably below +8/32 (paired upper bound
    < 8/32). At a true headroom of exactly 8/32, PRED-NONE then fires falsely about 2–3 % of the
    time per bar (a simulation of 4 000 trials of 32 paired resets with 4 000 resamples each:
    2.6 %, 3.0 % and 2.2 % with 0, 1 and 2 reversed pairs per 32 expected), or at most about 9 % if
    all three sit at 8/32. Before, it was the bar's miss rate, 42–45 %.
- **R8.15–R8.17 — Stage 0** (2026-10-04, each decided by Claude under owner delegation; record:
  [apple_plate_twin_v2_stage0.md](experiments/apple_plate_twin_v2_stage0.md)). Smoke seeds
  56900–56999 only; no cohort seed simulated; no world model.
  - **R8.15 — the Stage-0 code.** The frozen block is `plate_twin_v2.py` (DRAFT manifest
    `apple-plate-twin-v2.json`; the sha pin is `None` until the freeze), with workers, Stage O,
    a runner on `run_tools`' guards and the §5 tests. The pieces of the pinned runner chain that
    the runner needs (G-evidence, G-repro, the render majority, report I/O) are ported into
    `plate_twin_v2_harness.py` (as verbatim copies since R8.18), since a new runner may load no other script. Cell A's
    rule is discretised as plate(t) = plate(s0) + κ·(palm(max(t − L, s0)) − palm(s0)); the
    remedies' contact limit is the earliest apple–plate contact of the base-setting smokes, with
    s1 strictly before it, and a later s1 applies to cell A only.
  - **R8.16 — the Stage-0 smokes settle §12, and cell A is removed.** No apple–plate contact
    before s1 = 525 on 144 base-setting attempts (earliest contact: step 607), so s1 stays. Under
    H-final, cell A's plate moved a median of 0.060 cm after 485 (2 cm needed); L = 1 and s1 =
    535–605 gave 0.022–0.063 cm; nothing was refused. By the declared removal rule cell A is
    removed before the freeze, so K-pred's row is PRED-INFEASIBLE (escalate, no clause; PLAN.md's
    Branch B). m2: median 0.0012 cm, maximum 0.0019 cm. A full-look-ahead H-final(A) attempt took
    at most 25.0 s, below 60 s, so the per-attempt cap is not reviewed. The smokes ran P-3 on
    the reset truth as a stand-in estimate, because TASK-072 run-1's corpus root
    `look2-51171.npz` was unreadable on the archive disk and G-repro could not run. That
    blocked K0 until the file was restored; it now is (R8.18).
  - **R8.17 — the four non-blocking items of the #133 approval** are folded into the protocol:
    the header now gives each count bar's real source (D's 12/16 and 14/16 are declared stops,
    K0's H-clock stop is G2's minimum separation); R8.12's two numbers are not "both optimistic"
    (H-handover for H-twin is optimistic for G2, the in-sample H-clock(K) pessimistic); the
    noise guard's power is stated (PRED-NONE fires about 42 % of the time at a true headroom of
    4/32 and about 87 % at 2/32, so PRED-NEAR is the likely row at intermediate headrooms); and
    §10's "> 60 s per attempt" cap-review rule is part of §13.2 step 2.
- **R8.18 — the harness is a verbatim port** (2026-10-04, decided by Claude under owner
  delegation, on the review of #134). R8.15 called the ported harness "unchanged", but several
  functions differed from their sources. Option (a) is taken.
  - The pinned runners' pieces in `plate_twin_v2_harness.py` are now exact copies of their
    sources' text, and a test pins each one to its source function.
  - The only adaptations:
    1. the aliases `R65`, `M2R` and `LIN` are rebound to the harness's own copies;
    2. `LIN.load_wide_reset` returns `wm_critic_v2.wide_reset_values` instead of loading
       `scripts/evaluate_apple.py`, tested equal on every seed 51000–52199.
  - The truth stand-in moved out of the ported `cohort_estimates` into the runner's
    smoke-only `smoke_truth_estimates`, which `estimates_for` and `run()` refuse outside
    smokes.
  - The damaged evidence file `look2-51171.npz` has been restored bit-identically. A real G-repro
    check at `75d79ae` passed all eight reproduction checks, so K0's evidence blocker is cleared.
- **Why not skip to a LeWM task now.** No calibrated plate bar and no measured prediction headroom
  exist yet. Starting without them would repeat TASK-074 (an uncalibrated bar) and TASK-073 (no
  headroom). K-pred costs about 416 CPU attempts.
- **A definition fixed now** for every later task: "a LeWM-driven closed-loop success". See
  [apple_plate_twin_v2.md](experiments/apple_plate_twin_v2.md) §9 and [PLAN.md](PLAN.md).

The draft preregistration is [apple_plate_twin_v2.md](experiments/apple_plate_twin_v2.md) (DRAFT,
not frozen). The plan is [PLAN.md](PLAN.md).

## Development record 2026-10-01/02 — Arena cross-simulator checks, a GR00T reference baseline and the white plate (not gated)

Development only: no preregistration, no gate, and nothing here is a project-learned result.

- **Arena e9 (#120, TASK-025).** In Isaac Lab-Arena's GR00T-tutorial scene, the arm kinematics of
  NVIDIA's G1 USD agree with our MJCF (wrist within 0.8 mm), but the Dex3 fingertips differ by up
  to 3.1 cm. The MuJoCo-tuned scripted expert e9 left the apple at rest on **0/16** development
  seeds (16/16 in MuJoCo on the same seeds): it never grasps, because the arms stall without a
  gravity offset and, with one, the floating base steps back during the close. A scripted expert
  does not transfer as is. See [ARENA.md](ARENA.md) §7.
- **GR00T reference baseline (#121, TASK-025).** NVIDIA's GN1x-Tuned release (GR00T N1.7, step 65000), not
  the tutorial's checkpoint-20000, run client-only against the owner's server in that scene: **16/30** and **10/30** under Arena's loose contact-and-speed rule
  in two processes, and **0/30** under our stricter `apple_at_rest_arena_v0`. The strict check
  used PhysX's reported apple velocity, which is stale for a resting apple (inferred); a post-hoc
  position-based diagnostic gives 6/30. This is GR00T's result, an external reference, not ours.
  See [ARENA.md](ARENA.md) §8.
- **White plate (#122).** An opt-in white plate (`plate_color.py`) leaves TASK-075 at
  OBS-NONE. The paired intervals bound any colour effect on the offset's median error to about
  0.90–1.11 (a bound, not a finding of no difference; 1 of 12 intervals excludes 1.0), the apple
  term is unchanged, and colour does shift the plate-hidden check (overview224 R_off 7.86 → 4.26
  cm), a real colour effect on a gating check that still passes. The plate's colour is not what limits the readout. See
  [apple_white_plate_dev.md](experiments/apple_white_plate_dev.md).

## Decision 2026-10-01 — TASK-075 ends OBS-NONE: no view reads the place offset within τ; the clause fires; the next step is a task or condition change

**Outcome: OBS-NONE** (`apple_obs_ceiling_v2`). τ, the place tolerance, was measured before the
freeze at 1.0 cm: the largest planted target error at which P-3's pick plus e9's place still
rests ≥ 28/32 under TASK-074's condition. The render stage sealed the four views on
`apple-far-shift-v2`'s 270 train + val roots (VIEWS-SEALED). The readouts stage reproduced
TASK-074's reference numbers exactly (G-repro-off), then read 915 moving windows on 218 roots per
view.

- **No view is admitted.** The R_off c_upper values are: hand crop 2.655, onboard 112 3.394,
  onboard 224 3.321 and overview 224 3.438 cm, against τ = 1.0 cm.
- **A3 fails on every view.** No pooled-token readout beats a clock prior that reads no image
  (2.350 cm; in median, and detectably on three of four views). So, for the frozen R_off as
  fitted, no τ would admit a view under the frozen rule, and nothing is admitted at 0.5 or
  1.5 cm.
- **The representation checks fail too.** R_full and R_pix fail B on every view. The best value
  is R_pix on overview 224, at c_upper 2.022 cm.
- **The learning curve is not still falling** on any view; on three views the 100 % fit is worse
  than the 50 % fit.
- **The §7 abandonment clause fires.** It closes preregistering a LeWM planner or critic for the
  place phase under TASK-074's condition on these four views with frozen DINOv2 features,
  without new evidence of a different kind (a new view, a new readout family, or a task or
  condition change). The LeWM backend, the v2 task and the product goal stay open.
- **Scope.** Only linear and kernel-ridge readouts of single frames on frozen features were
  tested. Trained detectors, fine-tuned features, temporal aggregation and other sensors were not.
- **Reported only, not gated.** The same pooled-token ridge reads the plate position to
  0.49–0.68 cm and the apple position to 2.2–3.1 cm. The offset error tracks the apple term.
- **Next:** "a task or condition change" (§7). The results document lists four options. It
  recommends a gated perception-twin closed loop first: the place aimed at the frozen-DINOv2
  *plate* readout, with no world model. The choice is made by the owner,
  or Claude under the owner's delegation of 2026-09-30. Whether a later LeWM task on a plate
  target (or on a new place primitive) counts as a "task change" under the
  clause needs an explicit ruling by them.
- **Learned Apple→Plate on the frozen benchmark is still 0 successes.** No control line is
  primary.

**Evidence.** [apple_obs_ceiling_v2_results.md](experiments/apple_obs_ceiling_v2_results.md);
readouts report sha256 `96ab76b2…7b3f`, render report `753e05f2…a58c`.

## Decision 2026-10-01 — TASK-074 closes INCONCLUSIVE after two budget escalations; the next task measures the observation first

**Outcome: INCONCLUSIVE ("close without the clause")** (`apple_lewm_planner_v2`). K1 passed at
9 cm and the `apple-far-shift-v2` corpus was sealed. The train stage then escalated twice on its
budget rules: run-2 ESCALATE-BUDGET (the frozen rule wanted 80 000 updates, above the 60 000 cap),
and run-3, under Addendum A2's 80 000 cap, ESCALATE-BUDGET-LAST-TWO (N-7412 selected its last
point). The stage returned before O1 and O2.

- **Ruling (decided by Claude under owner delegation, 2026-10-01):** no further budget raise and
  no further train run. OFFLINE-PASS was already unreachable, because O2's encoded-readout bar
  (≤ 1.0 cm) sits below the measured readout error of 2.87 cm, whatever W does; another raise
  would cost about 9 h or more to reach, at best, L-G2A, which the rules already make
  uninformative about W; the binding constraint is the observation and readout, not the
  predictor.
  - *Note (factual, added after the descriptive diagnostic; the ruling is unchanged):* "at best,
    L-G2A" names the furthest row the rules permit. The diagnostic later found that the frozen
    `o2_passes` marks W seeds 7411 and 7412 void (N / persistence upper bounds 0.756 and 0.699,
    < 0.8). In `decide_offline`'s first-match order that void row comes before L-G2A, so, read
    as gates, these models would not have reached L-G2A. No row is read from the diagnostic.
- **The abandonment clause does not fire.** Its scope is not refuted and not closed.
- **No LeWM controller ran in closed loop.** Ranking, D3 and the gated stages never ran.
- **Lesson:** readout bars must be calibrated from measured ceilings before a freeze. O2's bar
  was carried from TASK-073 and never calibrated.
- **Next:** a measurement-first observation-ceiling study, to be preregistered separately.
- **Learned Apple→Plate on the frozen benchmark is still 0 successes.** No control line is
  primary.

**Evidence.** [apple_lewm_planner_v2_results.md](experiments/apple_lewm_planner_v2_results.md);
run-3 report sha256 `25cb1bf3…562c`.

## Decision 2026-09-30 — TASK-073 ends at K0 with S-NO-CONDITION; the D7 fallback (TASK-074) is authorised

**Outcome: S-NO-CONDITION** (`apple_wm_critic_v2`). K0, the preregistered condition calibration
(simulator only, no world model, 32 development resets per cell), found no mid-episode plate-shift
condition on `apple-to-plate-v2` that passes all four bars. In every cell where P-truth ran, P-3
given the true post-shift plate reached 26–29/32. Privileged 16-command look-ahead among the 25
nearby aims (H-sim) added at most +3/32 over that, against a bar of +4. The margins (+2 and +3
at best) are within a few resets on 32 resets and one run, so a larger cohort could land on
either side of the bar. At step 480 with a 5–6 cm
shift, the unaided policy still succeeded too often (18–20/32 against ≤ 8). In the protocol's
words, "BC plus perception leaves no measurable room for a world-model critic under this
disturbance on v2".

- **The LeWM critic was never trained or run.** This is a headroom finding under privileged
  look-ahead. It is not a test of LeWM and says nothing about how well LeWM predicts or ranks.
- **The abandonment clause does not fire.** The owner's D7 fallback applies: TASK-074, the
  LeWM-only planner, is authorised and needs its own preregistration.
- **Process:** K0 run-1 was a V: an external SIGTERM after memory pressure, with 16 workers
  reaching 25.9 GiB. It was fixed in #108 under the owner's rulings. K0 run-2 ran once, from
  `35772e5`, on a reported GO, with no void and no render event.
- **Learned Apple→Plate on the frozen benchmark is still 0 successes.** No control line is
  primary.

**Evidence.** [apple_wm_critic_v2_results.md](experiments/apple_wm_critic_v2_results.md); report
sha256 `f760af40…27f2`.

## Decision 2026-09-28 — M2 on cohort C: P-3 40/40, R-3 39/40; row M2-FAIL on G3 alone (TASK-072)

**Outcome: M2-FAIL. The only failing gate is G3.** M2 was preregistered by TASK-071 §12. It was
authorized by the owner's go-ahead and PR #102 (`apple_first_policy_v2_m2`), and run once from
`main` at `23e2593` on the pre-run reviewer's reported GO (render check IDENTICAL). The
orchestrator told the owner before the run started. It ran on the 40 stored cohort-C resets
(45300–45339), which had never been simulated before, and took 169 s.

- **Counted successes of 40:** **P-3 40**, R-3 39, C-3 12, B-replay 28, B-oracle 40, B-hold 0,
  B-random 0.
- **Gates:**
  - G1 pass (40 ≥ 17, and 40 > 28).
  - G2 pass (+28; 28 vs 0 discordant, p = 7.45e-9).
  - **G3 fail (+1; 1 vs 0, p = 1.0)**.
  - G4 pass (40 grasps).
  - G5 pass.
  - G6 pass.
  - G7 pass (median 0.42 ms per command; the DINOv2 forward pass plus readout had a median of
    31 ms).

**Two readings, reported side by side as fixed beforehand (authorization record §3.2).**
- **The frozen row, M2-FAIL:** "the claim is not made; the owner decides".
- **G3's declared reading:** "a learned visuomotor policy works; encoder pretraining contributes
  nothing measurable".

This log does not choose between them; the owner's ruling below does.

**What it is.** The carried BC/DAgger policy on a frozen DINOv2 readout succeeded on every
held-out reset. It beat the no-image control and open-loop replay of the expert on identical
resets, and it made no privileged read. The random-init encoder floor did as well within one
reset, so pretrained vision shows no measurable contribution here.

**What it is not.**
- **The official learned Apple→Plate count on the frozen v1 benchmark is still 0** (0/150 per
  model).
- **Not LeWM:** no world model is in the loop.
- **Not free of privileged training data:** the policy was trained on the privileged scripted
  expert e9's demonstrations, with a privileged DAgger labeller at training time. It made no
  privileged read at run time.
- **Small scale:** one run, one seed per arm, one camera (112 px onboard) and a narrow reset
  distribution.
- **Cohort C is no longer held out.** It has now been simulated. `apple_policy_v1.md` says it
  is "never reused"; any reuse needs a new, disclosed protocol and the owner's ruling.

**Owner ruling (2026-09-28).** The owner answered the orchestrator's recommendation verbatim:
"yes, do your recommendations". The ruling:
- **M2-FAIL stays the recorded row.** The frozen rule is not rewritten.
- **G3's declared reading is adopted as the interpretation.** On held-out apple-to-plate-v2
  resets, a learned visuomotor policy (DINOv2 + BC/DAgger, trained on e9 demonstrations) works,
  and encoder pretraining contributes nothing measurable.
- **This does not change the v1 benchmark** (still 0/150), and **it is not a LeWM result**.

**Evidence.** [apple_first_policy_v2_m2_results.md](experiments/apple_first_policy_v2_m2_results.md);
`benchmarks/manifests/apple-first-policy-v2-m2-results.json`; report sha256 `aa274cdd…01baa`.

## Decision 2026-09-28 — TASK-071's development result replicates on the Linux PC (TASK-072)

**Outcome: REPLICATED (M1-PASS, P-3 16/16).** TASK-072's preregistered run (PR #99,
`apple_first_policy_v2_linux`) re-ran TASK-071's whole pipeline on the Linux PC, from `main` at
`db65816`, on the pre-run reviewer's reported GO. The orchestrator was told before the run
started. The pipeline was corpus, readouts, BC, three DAgger iterations and M1 on D2. The run used
strict-deterministic CUDA training, NVIDIA EGL rendering and 16 simulation workers, and took
1 235 s.
- **The declared reading** (fixed before the run) was REPLICATED iff M1-PASS and P-3 ≥ 14/16.
- **Counted successes on D2 (of 16; Mac run-1 in brackets where different):** P-0 2 (4),
  P-1 11 (9), P-2 15, **P-3 16**, R-3 16, C-3 7 (3), B-replay 10 (9); A4-look, D-oracle-perc and
  B-oracle 16; B-hold and B-random 0.
- **P-3 succeeded on the same 16 resets** as run-1.
- **The scripted stages match run-1 outcome by outcome.** The corpus (129/200) and C0 agree root
  by root and attempt by attempt.

**What this is.** A cross-platform replication of an existence result on the non-gating
development cohort, which run-1 had already used. In the protocol's words, the result does
not depend on the Mac, its renderer or MPS. **The official learned Apple→Plate count on the
frozen benchmark is still 0.** Cohort C is untouched, and M2 needs a separate owner go-ahead. The policy is a
BC/DAgger MLP on a frozen DINOv2 readout, not LeWM.

**What limits it.**
- **R-3 = P-3 = 16/16 again**, on identical success sets. There is no evidence that pretrained
  vision helps.
- **The oracle arms are at the ceiling again.**
- **C-3 moved from 3 to 7** (paired on the same resets 4 vs 0, exact McNemar p = 0.125). Two runs
  cannot measure its spread, but any single C-3 number should be read with that in mind.
- One run, one seed per arm, 16 reused resets.

**Recorded, not amended.** The pre-run reviewer raised two non-blocking items:
- Stale TASK-071 wording in the frozen `void_rule` ("closes TASK-071") and in
  `REPLICATION_RULE["V"]` ("§14", where this protocol's section is §6).
- The flash and memory-efficient attention backends are enabled but unused.

**Evidence.** [apple_first_policy_v2_linux_results.md](experiments/apple_first_policy_v2_linux_results.md);
`benchmarks/manifests/apple-first-policy-v2-linux-results.json`; report sha256 `87745284…ec78`.

## Decision 2026-09-28 — a learned policy reaches its first successes on the development cohort of apple-to-plate-v2 (TASK-071)

**Outcome: M1-PASS.** TASK-071's preregistered run (PR #96, `apple_first_policy_v2`) trained
TASK-067's design (behaviour cloning plus 3 DAgger iterations, the frozen DINOv2 readout of the
post-look frame, the step counter and proprioception) on a fresh corpus of the e9 expert's
`apple-to-plate-v2` demonstrations. The run was made on the pre-run reviewer's reported GO, from
`main` at `9e23ced`, on the Mac (MPS), 6 215 s.
- **Counted success** (owner rulings T71-R1, T71-R2): `apple_at_rest_v0` and the latched grasp
  and place stages before the settle.
- **On the 16 development resets D2 (52000–52015):** P-0 4/16, P-1 9/16, P-2 15/16, **P-3 16/16**
  (carried; a selection on D2). No at-rest attempt went uncounted.
- **Controls on D2:** R-3 (random-init DINOv2 floor) 16/16, C-3 (no image) 3/16, B-replay 9/16;
  D-oracle-perc, A4-look and B-oracle 16/16; B-hold and B-random 0/16.

**What this is.** An existence result on the non-gating development cohort: the project's first
learned (L1) Apple→Plate successes, a "learned policy with a DINOv2 encoder", not LeWM driving the
robot. **The official learned Apple→Plate count on the frozen benchmark is still 0**; cohort C
is untouched, and M2 needs a separate authorization.

**What limits it.**
- **R-3 = P-3 = 16/16:** on D2, encoder pretraining shows no measurable contribution. M2's G3 is
  expected to fail if cohort C agrees.
- **The oracle arms gain nothing over the readout** (16/16 each, the ceiling): perception is not
  the binding constraint on this cohort. Only C-3 (3/16) shows that the per-reset estimates matter.
- One run, one seed, 16 resets, one camera; v2 only.

**Process note.** The pre-run GO reached the task agent as the reviewer's hand-back message and
was not relayed to the coordinator before the run started. From now on the coordinator is told
before any gated run starts.

**Platform note.** The project has since moved to a Linux PC (RTX 5080). There the frozen v2
runner refuses to start: the random-init floor encoder's weight digest differs on x86
(`546b9011…` vs the pinned `3d305f9c…`), non-smoke runs require MPS, and post-look frame hashes
differ between Apple GL and NVIDIA EGL; physics outcomes match within 2 µm. A Linux replication
needs a new, separately reviewed protocol version (a separate task). Evidence:
`outputs/linux-bringup-1/` on the Linux PC, development-only.

**Evidence.** [apple_first_policy_v2_results.md](experiments/apple_first_policy_v2_results.md);
`benchmarks/manifests/apple-first-policy-v2-results.json`; report sha256 `77aae207…d90b`.

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
