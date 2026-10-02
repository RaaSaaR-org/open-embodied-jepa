# Next-steps plan (from 2026-10-02)

This is the working plan of the next tasks toward the product goal: a LeWM-driven closed-loop
policy for the Unitree G1 with Dex3 hands. It is a plan, not a result. Each task still needs its
own preregistration and review, and every outcome below is open. [MVP_PLAN.md](MVP_PLAN.md) is
the original plan of record and is not rewritten. Rulings are in [DECISIONS.md](DECISIONS.md),
and outcomes are in the [experiment index](experiments/README.md).

**Where we stand** (the canonical sentence, DECISIONS 2026-10-02, R7). Learned Apple→Plate on the
frozen v1 MVP benchmark (TASK-020) is 0/150 per backend (`native_jepa` and LeWM). On
`apple-to-plate-v2`, the behaviour-cloning/DAgger policy P-3 (an MLP on a frozen DINOv2 readout,
trained on demonstrations from the privileged scripted expert e9; not a world model) scored 40/40
counted successes on the held-out cohort C, against 39/40 for its random-init encoder control R-3,
so TASK-072 M2 is M2-FAIL on G3, and cohort C is no longer held out. No LeWM-driven controller has
run in closed loop on v2 yet; LeWM's only closed-loop Apple→Plate runs are on v1, with 0
successes.

**What the last three tasks taught.**
- TASK-073 (S-NO-CONDITION): under a mid-episode plate shift, privileged look-ahead added at most
  +3/32 over P-3 given the true plate (bar +4). There was no measurable room for a critic.
- TASK-074 (INCONCLUSIVE): its encoded-readout bar (1.0 cm) sat below the readout's measured
  error (2.872 cm), and its budget rule escalated by design. **Lesson: readout bars are calibrated
  from measured ceilings, and budget rules are checked with `run_tools.check_budget`.**
- TASK-075 (OBS-NONE, the clause fired): no view reads the apple-minus-plate offset within
  τ = 1.0 cm. The plate itself reads to 0.49–0.68 cm in median (reported only; 87.5th percentile
  up to 1.36 cm on the hand crop), and the offset error tracks the apple term (2.2–3.1 cm).

**What "a LeWM-driven closed-loop success" means here.** A counted success on a gated, fresh
cohort, in an arm whose decisions for a named phase come from a LeWM predictor rolled forward from
the encoded current observation, with no privileged read. It must beat action-blind, scene-blind
and random twins, and be reported beside the best non-world-model arm for the same phase. The
full definition is in [apple_plate_twin_v2.md](experiments/apple_plate_twin_v2.md) §9. The
claim names the phase. "LeWM-driven place-target selection" after P-3's pick is not "a LeWM
policy".

## The tasks

The order is fixed by dependencies. Branches A and B after TASK-076 are alternatives: TASK-076's
K-pred row picks one.

### TASK-076: the plate-readout perception twin, plus a moving-plate headroom check

Protocol draft: [apple_plate_twin_v2.md](experiments/apple_plate_twin_v2.md) (DRAFT).

- **Question.** Can e9's place, after P-3's pick, aim at a frozen-DINOv2 single-frame plate
  readout and still reach the place bar under TASK-074's 9 cm condition? And (K-pred) is there a
  moving-plate condition where aiming at where the plate *is* fails but aiming at where it *will
  be* succeeds, so that prediction has room?
- **LeWM's role.** None in the loop. It measures the encoded-latent plate ceiling, the perception
  baseline and the prediction headroom that TASK-077 needs (ruling R8).
- **GPU.** Under 2 minutes, under 1 GiB (Stage O featurisation), through `scripts/gpu_run.sh`.
  Everything else is CPU. Machine time is under 1 h in total; protocol, code and review take about
  1–2 days.
- **Depends on.** The sealed `apple-far-shift-v2-views` store and `apple-far-shift-v2` corpus
  (the run worktrees `task075-run` and `task074-run`, which stay on the SSD), the P-3 checkpoint,
  and an independent review.
- **Stop / abandon.** CAL-ESCALATE at K0 if the ceiling is below 30/32 or τ_re is undefined.
  TWIN-DEV-STOP at the development closed loop. The clause fires on TWIN-OFF-FAIL, TWIN-FAIL or
  TWIN-HARM, and closes single-frame frozen-readout variants of this place. Next: TASK-075's
  Option 2, a place servo (R3). K-pred's PRED-NONE closes the tested moving-plate family as a
  LeWM place condition.

### Branch A: TASK-077, a LeWM plate-prediction place planner on v2 (only if K-pred returns PRED-ADMIT)

- **Question.** Under the admitted moving-plate cell, does a LeWM token predictor (TASK-066's
  latent, h = 16, rolled recursively to the plate's stop step) choosing where e9's place puts the
  apple produce more counted successes than the non-predicting perception twin (H-twin) and the
  action-blind, scene-blind and random twins? And is it not clearly below the hand-written
  constant-velocity extrapolator (H-cv)? This would be the first LeWM controller in closed loop on
  v2. It is a task change declared under R2.
- **Bars, calibrated before the freeze.** The predicted-latent plate bar B satisfies
  c_plate ≤ B ≤ τ_re (from TASK-076). Any allowance above c_plate is calibrated on development
  data. The tie bars come from TASK-076's measured H-twin and H-cv counts.
- **GPU.** The main cost. A new e9 corpus under the moving plate (about 4 min on the CPU;
  TASK-074's corpus took 232 s). DINOv2 featurisation (about 20 min, TASK-074's estimate). W/N
  calibration (TASK-074: 2 664 s for 60 000 updates on W-7410). Six models (TASK-074 run-3: 3 431–3 582
  s each at 80 000 updates). The budget block must pass `run_tools.check_budget`. For example,
  calibration 60 000, factor 2, step 5 000, min 10 000, cap 120 000 passes, while TASK-073/074's
  cap of 60 000 fails. At that cap the worst case is about 10–11 h of GPU, in per-job slots under
  the shared lock (`gpu_run.sh --wait`), so other projects get turns between jobs. Closed loop on
  the CPU.
- **Depends on.** TASK-076's PRED-ADMIT row, c_plate and τ_re, and the H-twin and H-cv counts.
  It also depends on the review of TASK-076's results.
- **Stop / abandon.** An offline gate on the predicted plate reading (B), with the TASK-066
  dynamics gates (no collapse, beats copy-last and an equally trained no-action model,
  action-sensitive). A development closed loop on 16 resets with a stop rule. Then the gated
  cohort. The clause fires if LeWM fails to beat H-twin and the blind twins (L-NO-GAIN), and
  closes LeWM plate prediction under that cell. If it beats them but falls clearly below H-cv, the
  row is a no-claim descriptive note, as TASK-074's L-TWIN-BETTER was. **A pass shows "LeWM can
  drive the place target under a prediction condition". It does not show that LeWM is needed if
  H-cv is at the ceiling** (stated in TASK-076 §6.3 before any number).

### Branch B: if K-pred returns PRED-NONE (or TASK-076's clause fires)

- **PRED-NONE.** The place phase of v2 then has no admissible LeWM role under the tested family.
  **Recommended (decided by Claude under owner delegation, R8):** preregister a task change in
  which the target's motion depends on the robot's own action, so that an action-conditioned
  predictor has something to predict that a hand-written extrapolator does not get for free. One
  example is a plate the hand must push into a marked zone before the place. A design note comes
  first, with a K0 headroom check like K-pred (no world model). GPU: none for the design and K0;
  TASK-077-sized afterwards.
- **TWIN-FAIL / TWIN-OFF-FAIL.** TASK-075's Option 2: a place primitive with its own local
  feedback, with its τ curve re-measured (R3). That is a development task on the CPU (several days,
  then about 5 min per τ curve). A later LeWM task on it must use targets that an image-free prior
  cannot guess (the offset's clock prior already reads 2.35 cm).

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
| **F11** SIGTERM race flake (`void_reason` "RuntimeError: release unlocked lock" instead of "received SIGTERM", main run 36793229744; the stage still ends V) | `tests/test_wm_critic_v2.py` is hash-pinned by the TASK-073/074/075 manifests, so its assertion cannot be relaxed. New runners use `run_tools.install_guards` (#125) | either an amendment that re-pins the test file, or accepting the known flake |
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
