# Next-steps plan (from 2026-10-02)

This is the working plan of the next tasks toward the product goal: a LeWM-driven closed-loop
policy for the Unitree G1 with Dex3 hands. It is a plan, not a result. Each task still needs its
own preregistration and review, and every outcome below is open. [MVP_PLAN.md](MVP_PLAN.md) is
the original plan of record and is not rewritten. Rulings are in [DECISIONS.md](DECISIONS.md),
and outcomes are in the [experiment index](experiments/README.md).

**Where we stand.** This is the canonical sentence (DECISIONS 2026-10-02, R7), verbatim:

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

**What the last three tasks taught.**
- **TASK-073 (S-NO-CONDITION).** Under a mid-episode plate shift, privileged look-ahead added at
  most +3/32 over P-3 given the true plate (bar +4). There was no measurable room for a critic.
- **TASK-074 (INCONCLUSIVE).** Its encoded-readout bar (1.0 cm) sat below the readout's measured
  error (2.872 cm), and its budget rule escalated by design. **Lesson: readout bars are calibrated
  from measured ceilings, on the representation that is gated, and budget rules are checked with
  `run_tools.check_budget`.**
- **TASK-075 (OBS-NONE, the clause fired).** No view reads the apple-minus-plate offset within
  τ = 1.0 cm. Its reported-only pooled-token plate readout reads the plate to 0.49–0.68 cm in
  median, and the offset error tracks the apple term (2.2–3.1 cm). The tails matter: in the
  white-plate development record (`experiments/apple_white_plate_dev.md`), the plate readout's
  87.5th percentile reaches 1.35–1.36 cm on the hand crop.

**What "a LeWM-driven closed-loop success" means here.** It is a counted success on a gated,
fresh cohort, in an arm whose decisions for a named phase come from a LeWM predictor rolled
forward from the encoded current observation, with no privileged read.
- It must beat action-blind, scene-blind and random twins.
- It must be reported beside the best non-world-model arm for the same phase.
- The claim names the phase: "LeWM-driven place-target selection" after P-3's pick is not "a LeWM
  policy".

The full definition is in [apple_plate_twin_v2.md](experiments/apple_plate_twin_v2.md) §9.
For tasks after TASK-076, R9.8 (DECISIONS 2026-10-04) splits it into a primary claim, "LeWM-driven
closed-loop success" (a calibrated bar, non-inferiority to the best non-world-model arm, and
beating the action-blind and scene-blind twins), and a secondary claim, "LeWM needed" (detectably
better than the best hand-written arm; reported only, never a gate).
Because the LeWM arm must beat an action-blind predictor, any condition that admits a LeWM task
must make the target depend on the robot's own action (ruling R8.3). That is necessary, not
sufficient: an action-blind predictor can still tie if the training aims are centred on the
target and the controller aims at one predicted plate (R8.8).

## The tasks

The order is fixed by dependencies. Branches A and B after TASK-076 are alternatives, and
TASK-076's K-pred row picks one.

### TASK-076: the plate-readout perception twin, plus a prediction-headroom check

Protocol: [apple_plate_twin_v2.md](experiments/apple_plate_twin_v2.md) (FROZEN after K0-PASS,
in force once merged on an independent APPROVE).

**Stage-0 update (2026-10-04, R8.16):** the smokes removed cell A, because its plate barely moves
after the last decision (median 0.06 cm under H-final, against the 2 cm needed, and no declared
remedy reached 2 cm). K-pred's row is therefore PRED-INFEASIBLE if Stage O records O-PASS, and
PRED-NOT-RUN if it does not; either way Branch B applies, with no clause. The primary question
(Stages O, D, S/U) is unchanged. The G-repro evidence is restored and its check passes
([Stage-0 record](experiments/apple_plate_twin_v2_stage0.md)).

**K0 update (2026-10-04, R8.19):** K0 ended K0-PASS: τ_re = 1.0 cm, N_K(0) = 32/32, H-clock(K) =
24/32, H-stale(K) = 0/32, G2's predicted pass probability 0.9983 (protocol §5.1). Its values are
in the frozen block, and the protocol is FROZEN. Stage O is next, on a GO after the merge.

- **Question.** Two parts:
  - Can e9's place, after P-3's pick, aim at a frozen-DINOv2 single-frame plate readout and still
    reach the place bar under TASK-074's 9 cm condition?
  - (K-pred) Is there a condition where the plate keeps moving, *driven by the robot's own
    action*, such that aiming at its current or extrapolated position fails but aiming at its
    final position succeeds? That is cell A, a declared simulator rule: the plate moves against
    the palm's velocity of 2 steps earlier, so the target depends on the aim chosen at the last
    decision (R8.7).
  - Two constant-velocity cells are reported only. There, an action-blind predictor is expected
    to tie, and the constant-velocity extrapolator H-cv to reach about the ceiling.
- **LeWM's role.** None in the loop. The task measures what TASK-077 needs (ruling R8):
  - the plate ceiling c_plate, on the pooled 4 × 4 LeWM latent;
  - the perception baseline;
  - the prediction headroom.
- **GPU.** Under 2 minutes and under 1 GiB (Stage O featurisation), through `scripts/gpu_run.sh`.
  Everything else runs on the CPU. Machine time is about 1 h in total. Protocol, code and review
  take about 1–2 days.
- **Depends on.**
  - the sealed `apple-far-shift-v2-views` store (manifest sha256 `ea627a8f…4d77`) and
    `apple-far-shift-v2` corpus (manifest sha256 `fe7ab915…b134bd`), restored from the USB disk to
    the main checkout's `data/` on 2026-10-04, every file checked against the archived sha256; a
    run worktree reaches them through `scripts/new_worktree.sh --run` (see [STORAGE.md](STORAGE.md));
  - the P-3 checkpoint;
  - an independent review.
- **Stop / abandon.**
  - **K0:** CAL-ESCALATE on any of: the ceiling below 30/32; τ_re undefined; H-stale above 4/32;
    H-clock within 4/32 of the ceiling. G2's predicted feasibility (exact McNemar on K0's paired
    counts) is reported beside the last stop (R8.12).
  - **Development closed loop:** TWIN-DEV-STOP.
  - **Escalate, no clause:** TWIN-NEAR (a G1 miss within noise of the ceiling), TWIN-PRIOR and
    U-VOID-CEILING (the true plate itself misses G3's bar on U; R8.13).
  - **Clause:** it fires on TWIN-OFF-FAIL, TWIN-FAIL or TWIN-HARM, and closes single-frame
    frozen-readout variants of this place. The next step is then TASK-075's Option 2, a place
    servo (R3).
  - **K-pred:**
    - PRED-INFEASIBLE (cell A was removed at Stage 0, or its ceiling fails), PRED-NO-BAR and
      PRED-NOT-RUN (Stage O did not pass, so K-pred did not run) escalate without a clause.
    - PRED-NEAR (a headroom bar missed, but not detectably below +8/32; R8.14) escalates
      without a clause or a claim.
    - PRED-NONE (the ceiling holds, and a headroom is detectably below +8/32) fires K-pred's
      clause. At a true headroom of exactly 8/32 it fires falsely about 2–3 % of the time per bar.

### Branch A: TASK-077, a LeWM plate-prediction place planner on v2 (only if K-pred returns PRED-ADMIT(A))

- **Question.** Take cell A, the action-dependent plate, declared as a task change under R2. Does
  a LeWM token predictor choosing where e9's place puts the apple produce more counted successes
  than the non-predicting perception twin (H-twin), the constant-velocity extrapolator (H-cv), and
  the action-blind, scene-blind and random twins? The predictor is TASK-066's pooled 4 × 4 latent,
  rolled forward to the plate's stop step. This would be the first LeWM controller in closed loop
  on v2.
- **The horizon must be gated first.** From the decisions at 405–485 to the stop step s1 = 525 is
  40–120 steps. TASK-066 gated only h = 8 and h = 16, so the recursive roll-out is ungated.
  TASK-077 gates its own horizon offline, with TASK-066's dynamics gates at that horizon, before
  any closed loop.
- **Declared before the freeze** (R8.8, R8.9):
  - the controller form, which should rank candidate aims by their individual predicted
    outcomes (one roll-out per candidate), not aim at one predicted plate;
  - the spread of the training aims relative to the fixed point g\* = (p − κh)/(1 − κ);
  - the action-blind twin's expected result under that controller and corpus;
  - the predictor's history: TASK-066's history-one predictor qualifies only if
    c_plate + m2\* ≤ τ_re (m2\*, the measured 2-step palm term, from TASK-076's K-pred), or if
    B's calibrated allowance covers m2\*; otherwise 3 frames or the last 2 executed commands.
- **Bars, calibrated before the freeze.** The predicted-latent plate bar B satisfies
  c_plate ≤ B ≤ τ_re. c_plate is measured on the pooled latent in TASK-076. Any allowance above
  c_plate is calibrated on development data. The tie bar comes from TASK-076's measured H-twin,
  H-cv and H-rule counts on cell A.
- **GPU.** The main cost:
  - a new e9 corpus under cell A (about 4 min on the CPU; TASK-074's corpus took 232 s);
  - DINOv2 featurisation (about 20 min, TASK-074's estimate);
  - W/N calibration (TASK-074: 2 664 s for 60 000 updates on W-7410);
  - six models (TASK-074 run-3: 3 431–3 582 s each at 80 000 updates).

  The budget block must pass `run_tools.check_budget`. For example, calibration 60 000, factor 2,
  step 5 000, min 10 000, cap 120 000 passes, while TASK-073/074's cap of 60 000 fails. At that
  cap the worst case is about 10–11 h of GPU. It runs in per-job slots under the shared lock
  (`gpu_run.sh --wait`), so other projects get turns between jobs. The closed loop runs on the
  CPU.
- **Depends on.**
  - TASK-076's PRED-ADMIT(A) row;
  - c_plate (pooled) and τ_re;
  - the H-twin, H-cv and H-rule counts on cell A;
  - the review of TASK-076's results.
- **Stop / abandon.**
  - **Gates before the gated cohort:** an offline gate on the predicted plate reading (B), at the
    gated horizon; then the TASK-066 dynamics gates (no collapse, beats copy-last and an equally
    trained no-action model, action-sensitive); then a development closed loop on 16 resets with
    a stop rule.
  - **No gain:** the clause fires if LeWM fails to beat H-twin, H-cv and the blind twins
    (L-NO-GAIN). It closes LeWM plate prediction under cell A.
  - **Below the best non-world-model arm:** if LeWM beats those arms but falls clearly below the
    best non-world-model arm (H-rule), the row has no claim **and fires the clause**, as TASK-074's
    L-TWIN-BETTER did ("clause; no claim").
  - **What a pass shows:** "LeWM can drive the place target under an action-dependent prediction
    condition, beating the action-blind, scene-blind and random twins". **It does not show that
    LeWM is needed if H-rule is at the ceiling** (stated in TASK-076 §6.3 before any number).

### Branch B: if K-pred returns PRED-NONE, PRED-NEAR, PRED-INFEASIBLE, PRED-NO-BAR or PRED-NOT-RUN (or TASK-076's clause fires)

- **PRED-INFEASIBLE.** The declared action-dependent rule could not be run, or its ceiling failed.
  This is a design failure, so it escalates without a clause.
  - **Recommended (decided by Claude under owner delegation, R8):** a design note for a different
    action-dependent target, with its own K0-style headroom check and no world model. One example
    is a plate the hand must push into a marked zone before the place.
  - GPU: none for the design and the K0; TASK-077-sized afterwards.
  - **Design note (DRAFT, 2026-10-04, R9):**
    [apple_lewm_next_v2_design.md](experiments/apple_lewm_next_v2_design.md), revised after the
    review of #135. Three candidates, assessed under R9.8's two claims ("LeWM-driven closed-loop
    success", the primary; "LeWM needed", reported only): C1, a single aim committed at 405 under
    cell A's reactive-plate rule (recommended, decided by Claude under owner delegation); C2, a
    single pre-pick push of a free plate (not recommended: with the scene's constants pushing is
    quasi-static, so a displacement table ties); C3, a contact-sensitive grasp approach (not
    taken). Next: a CPU-only feasibility record for C1 (C1-F1 to C1-F6), before any protocol or
    GPU.
- **PRED-NONE.** The tested rule left no room for prediction, and K-pred's clause closes it. A
  different action-dependent condition then needs new evidence of a different kind, as the clause
  says.
- **PRED-NEAR.** A headroom bar missed within noise (R8.14). Escalate, no clause, no claim. Any
  repeat of cell A needs fresh seeds and its own ruling.
- **PRED-NO-BAR.** The pooled LeWM latent does not read the plate within τ_re, so no TASK-077 bar
  is feasible. Escalate. This is a representation question for the owner, close to TASK-075's
  OBS-REPRESENTATION. It is not a silent reopening.
- **TWIN-FAIL / TWIN-OFF-FAIL.** The next step is TASK-075's Option 2: a place primitive with its
  own local feedback, with its τ curve re-measured (R3).
  - That is a development task on the CPU: several days, then about 5 min per τ curve.
  - A later LeWM task on it must use targets that an image-free prior cannot guess (the offset's
    clock prior already reads 2.35 cm).

### TASK-078: Arena perception transfer, offline (ARENA.md §5 step 4; development)

- **Question.** Does the frozen DINOv2 plate readout transfer to Isaac Lab-Arena's head camera
  (640 × 480, pitched 35°, against our 112 px at 60°) within an Arena place tolerance measured the
  same way as τ?
- **GPU.** Isaac containers: one Arena process at a time, about 2 GiB of headroom next to the GR00T
  server (ARENA.md §6), with 5.5–7 min of start-up per run. Run it through
  `gpu_run.sh --wait --container oej-isaac- …`. Estimate: 1–3 h of GPU over a few container runs.
- **Depends on.** #123 (e9-arena) merged after its independent review (F26), so that Arena resets
  exist with a working scripted place. It also depends on F14 (shared `arena_truth`, so that the
  at-rest check does not use PhysX's stale velocity), on mirrored, hashed Arena assets, and on
  TASK-076 (the readout recipe and c_plate).
- **Stop.** If e9-arena cannot rest the apple on fresh seeds at a bar like TASK-070's, there is no
  Arena τ, and this task stops at development. If the Arena plate readout's upper bound exceeds
  the Arena τ, no Arena closed loop is planned.

### TASK-079: the LeWM place in Arena (cross-simulator; only after TASK-077's claim row and TASK-078's pass)

- **Question.** Does the LeWM place planner keep its gated result when the simulator, camera and
  robot asset change? That is the last step before hardware commissioning.
- **GPU.** Isaac plus LeWM fine-tuning or re-training on Arena frames; to be estimated from
  TASK-077's measured times.
- **Depends on.** TASK-077 L-PASS (or its equivalent row), TASK-078's pass, and an owner ruling on
  Arena as a gated benchmark (it is development-only today, TASK-025).
- **Stop.** A failure on Arena's own ceiling arm (scripted e9-arena with the true plate) voids the
  comparison. A failure of LeWM against Arena's perception twin ends the cross-simulator claim.

**The real G1 (TASK-026)** stays future work. `hardware.py` is mock-only, and physical execution
needs commissioning and the owner. Nothing in this plan runs on hardware.

## Open follow-ups (not research tasks)

| item | state | needs |
|---|---|---|
| **F10** integration workflow (timeout 20 → 30 min; per-module loop moved to nightly and `workflow_dispatch`) | ready as commit `7eb696e` on the local branch `fix/ci-integration-workflow`; the push was refused because the token lacks the `workflow` scope (#124) | the owner runs `gh auth refresh -s workflow`, then the branch is pushed and reviewed. A Linux integration job is still not added |
| **F11** SIGTERM race flake (`void_reason` "RuntimeError: release unlocked lock" instead of "received SIGTERM", main run 36793229744; the stage still ends V) | known flake, still open. `tests/test_wm_critic_v2.py` is hash-pinned by the TASK-073/074/075 manifests, so its assertion cannot be relaxed. New runners use `run_tools.install_guards` (#125). Seen again on 2026-10-04 in #133's macos-integration job (run 37208326249, `test_a_process_group_sigterm_with_a_live_pool_writes_the_v_report[0]`, in the per-module order check); it passed on a rerun of the failed job. A red macos-integration on this test alone is this flake, not a regression, but it must still be rerun to green, not ignored | either an amendment that re-pins the test file, or accepting the known flake |
| **Test-order bug** | `pytest tests/test_apple_evaluation.py tests/test_privileged_rollout.py` gives 6 errors in `test_privileged_rollout.py` ("privileged rollouts require the live G1 MuJoCo embodiment"); each file passes alone, and so does the full suite. Pre-existing on main (#124 review) | find the shared state that the first file leaves behind |
| **F14** shared `arena_truth.py` (finite-difference speed, `at_rest_arena`, blank first frame) | not started: wait for #123 to merge, to avoid conflicts | after #123; it is a prerequisite of TASK-078 |
| **F26** independent review of #123 (e9-arena) | open, unreviewed | a CPU-only review of the code and of its pre-run declaration |
| **`oej-isaac-newton` archiving** (2.8 GB, `feat/task-025-isaac-newton`) | skipped by the archive pass: a `tail -F …/outputs/isaac-newton-scripted-dev-3/log.txt` process holds a file open (docs/STORAGE.md) | end that process, then archive it like the others (copy, checksum, MANIFEST row, verify before removing) |

## GPU etiquette (all tasks)

Every GPU job goes through `scripts/gpu_run.sh` (the machine-wide flock
`~/.local/state/gpu/lock`, shared with other projects; see `~/.local/state/gpu/BOARD.md`). New
runners call `run_tools.gpu_guard(require_lock=True)`. Long work takes the lock per job, not per
queue. Resident services (other projects' queues, the GR00T server) are never stopped or
reconfigured. CPU-only stages do not take the GPU lock.
