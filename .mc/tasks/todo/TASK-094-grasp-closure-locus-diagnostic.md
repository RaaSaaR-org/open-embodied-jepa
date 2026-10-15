---
id: TASK-094
aliases:
- TASK-094
title: 'Grasp-closure locus diagnostic: does the partial closure the world-model arms command cause the grasp failures, and is it the goal and cost, the model or the search (measurement only, no controller)'
slug: grasp-closure-locus-diagnostic
status: todo
priority: 2
owner: ''
projects: []
customers: []
tags:
- world-model
- lewm
- diagnostic
- preregistration
- phase-3
sprint: ''
depends_on:
- "[[TASK-093]]"
due_date: ''
created: 2026-10-15
updated: 2026-10-15
---


# Grasp-closure locus diagnostic: does the partial closure the world-model arms command cause the grasp failures, and is it the goal and cost, the model or the search (measurement only, no controller)

## Description

**DRAFT.** Chosen by R27.156 (DECISIONS (bh); design note
[grasp_next_step_v1.md](../../../docs/experiments/grasp_next_step_v1.md)). DRAFT preregistration:
[grasp_closure_locus_v1.md](../../../docs/experiments/grasp_closure_locus_v1.md) (R27.160–R27.176,
DECISIONS (bi)); merged (#225). **Stage 0 ended S0-ESCALATE** (§16, R27.177–R27.182, DECISIONS
(bj)) on §14's n_d rule: 6 (Rs) and 1 (R2) of 32 development roots discordant; every other check
passed. **Escalation ruling** (§17, R27.183–R27.186, DECISIONS (bk)): option 2, extended — M and S
move to 128 gated root seeds 946000–946127 (942000–942063 retired unused), R2 stays reported, S's ℓ
is the median executed right grasp over window steps 3–7 and its search miss compares the plan with
its right-grasp requests replaced by the squeeze branch's; K, every bar and row unchanged. A
Stage 0′ (no new development cohort) checks this and freezes the protocol at its merge. **Stage 0′
ended S0-PASS** (§18, R27.187–R27.192, DECISIONS (bl)): projected n_d 24 at Rs and 4 at R2 at 128
roots, every check passes, the gated stage projects to 0.352 GPU h; the protocol is frozen at the
Stage 0′ merge. **Gated stage** (§19, R27.193–R27.199, DECISIONS (bm); gated, one run, simulation
only): **K-NULL** by §4.3's text — K-1.0 88/128 (exact 95 % 0.600–0.766), K-s\* 84, K-0.6 81;
losses +0.031 (bootstrap 95 % −0.031 to +0.094) and +0.055 (−0.016 to +0.125), both below 0.15.
The summarizer reads K-VOID only on a size check of the four-reset determinism re-run reports,
which §4.3 does not contain (disclosed; the review must confirm). Palm offset +y 1.2 cm cost
+0.445 (31/128). The roots job exited VOID at root 946073, so M and S are L-VOID and reported
only (G-87102 `last_two_triggered`). The grasp line pauses and the owner is told; R7 unchanged.

**Why.** An offline read of the Stage BS reports of TASK-089, TASK-091 and TASK-093 (R27.152)
has these qualifiers: one run per stage, simulation only, pooled, an association, and read after
the results. It finds:

- **Contact without a hold.** The world-model arms touch the target in the close phase on
  982 / 1 856 episodes, but succeed on only 105 of those. ik succeeds on 220 / 302.
- **The closure they command.** They command a partial closure (median lift-phase mean 0.53–0.73
  per run, against ik's 1.00). It sits near the demonstrator's *measured*, object-blocked hand
  (median synergy about 0.68), not near its +1 command.
- **Success by commanded closure.** Given contact, successes are 21 / 739 below 0.7 and 17 / 23
  at 0.9 or above (commanded before the object rose).
- **The corpus.** Partial closures at contact are rare in `play-v1` (97.1 % of contact
  commands are ≥ 0.8).

**The question.** Is that closure the physical cause of the failures? If it is, does the failure
sit in the goal and cost (b), in G (a) or in the search (c)?

**Parts.** Each runs once, on fresh seeds that the preregistration checks.

- **K, the physics dose (scripted, no model).**
  - TASK-088's IK follower with its closing command held at +1.0, 0.9, 0.8, 0.7 and the close
    subgoal's measured synergy.
  - +1.0 and the measured synergy are also run with a palm-target offset of about 1.2 cm.
- **M, the ranking (offline).**
  - At fresh roots at the end of the demonstrator's close, scripted branches follow the lift at
    each hold level and are simulated to their true outcome.
  - Each branch is scored by TASK-089's frozen step-averaged cost, both from G-87100–87102's
    predictions and from the branch's real frames and states (the ceiling).
- **S, the search (offline).** TASK-089's frozen planner at the same roots: its chosen hold level,
  and G's cost of that plan against the squeeze branch.

**Rows** (declared with bars and margins in the preregistration; routing only, no claim):
- K-VOID: the run is void, for example when ik at c = +1.0 does not reproduce its frozen BS
  rate within a declared margin, or when a degeneracy check fails.
- K-NULL (no lift loss of 0.15 or more at s\* or at 0.6): if TASK-094 reads K-NULL, the grasp line pauses and the owner is told. D's work (for
  example a Phase 4 precursor with a scripted closure rule) still needs its own ruling, because
  Phase 4 is not admitted (R25.27, R27.148) (R27.157). The palm-offset arm is reported beside
  K-NULL, so that a placement-dominant result is visible to the pause ruling.
- K-CAUSAL is followed by one of:
  - L-UNREAD (no root type read);
  - L-COST (b);
  - L-MODEL (a);
  - L-SEARCH (c);
  - L-MIXED.

  Each routes to a ruling.

**Limits.**
- No world-model closed loop and no training.
- No planner, cost, horizon or action-parametrisation variant is preregistered for grasping, so
  R26.33, R27.71 and R27.149 are not engaged.
- The scripted probes are not learned results.
- Not Apple→Plate; R7 does not change.

**Budget:** about 0.5 GPU h (ceiling 1.0) plus CPU, with every GPU job under
`gpu_run.sh --wait --board`.

## Acceptance criteria

- [x] DRAFT preregistration (`docs/experiments/grasp_closure_locus_v1.md`; written, R27.160–R27.176), reviewed
  independently and merged. It fixes the seeds (checked by the seed check), the hold levels, the
  palm offset, the roots, the bars and margins and the row order before anything runs.
- [x] Stage 0: opt-in code with no frozen file edited, tests, the degeneracy checks of M's
  branches on development roots, and a debug smoke. The protocol is frozen at the Stage 0 merge.
  (Code, tests, checks and smoke done; ended S0-ESCALATE on the n_d rule, so not frozen; R27.177–R27.182.
  The ruling R27.183–R27.186 adds a Stage 0′, which freezes the protocol at its merge.)
- [x] Stage 0′ (R27.183 (5)): code and tests for the amendment, ℓ recomputed, the comparator's
  costs on the development plans, the seed check of 946000–946127 and the projection; ended
  S0-PASS (R27.187–R27.192); the protocol is frozen at its merge.
- [x] K, M and S recorded at the frozen revision from a clean tree, with evidence in
  `~/develop/emai/evidence/task094-*` (`task094-gated/`; §19).
- [x] Row recorded with a next-step ruling (K-NULL pauses the grasp line; D still needs its own
  ruling); R7 unchanged. (K-NULL by §4.3's text, R27.194–R27.197; the summarizer's K-VOID is
  disclosed and the reading needs the review's confirmation, R27.195.)

## Notes

- 2026-10-15, created as DRAFT by R27.156 (branch `docs/grasp-next-step-ruling`). The offline read
  behind it is in `~/develop/emai/evidence/grasp-next-step/`; it used the CPU only.
- 2026-10-15, DRAFT preregistration written (R27.160–R27.176, DECISIONS (bi); branch
  `docs/task094-prereg`). Fixed there:
  - **K:** TASK-088's IK follower with its closing command held at +1.0, 0.9, 0.8, 0.7, s\*
    (the goal's measured close-subgoal synergy) and 0.6 (reported, and part of K-NULL's condition),
    K-open (c = −1, degeneracy reference), and palm offsets v_x = (+1.2, 0, 0) cm and v_y = (0, +1.2, 0) cm (pelvis frame, a
    rigidly displaced goal) at +1.0 and s\*; 128 paired gated resets 941000–941127.
  - **M and S:** 64 gated root seeds 942000–942063, two root types: Rs (8 of the demonstrator's 35
    close commands in; roots in the close: yes, R27.163) and R2 (the end of the close); branches at
    +1.0, 0.9, 0.8, 0.7, s\* and a control, simulated to their true outcome; the frozen
    step-averaged cost on G-87100–87102 (G-87102 `last_two_triggered`) and on real frames; S is
    TASK-089's frozen, unprojected planner.
  - **Rows and bars:** K-VOID (K-1.0 outside 63–114 of 128; K-open > 3 / 128; checks), **K-NULL**
    (the reset-bootstrap upper bounds of both the K-1.0 minus K-s\* and the K-1.0 minus K-0.6 lift
    rates below δ_K = 0.15; salt 9014), K-CAUSAL (loss ≥ 0.15 and McNemar p < 0.01),
    K-UNCLEAR; after K-CAUSAL, first match: L-VOID, L-UNREAD (no root type read), L-COST (ceiling
    < 0.75 on discordant roots), L-MODEL (G below ⌈(0.5 + 0.75 (ĉ − 0.5)) n_d⌉ on two of three seeds), L-SEARCH (search misses
    on ≥ a quarter of discordant roots, two of three seeds), L-MIXED; the positive control at
    56 / 64 (28 / 32 at Stage 0). Routing only; no controller. K-CAUSAL with L-COST is put forward,
    not ruled met, as R26.33 evidence for option C; the record names the root type(s) at which
    L-COST was read (Rs's window holds no lift).
  - **Seeds:** Stage 0 development K 94100–94131 and roots 940000–940031; debug 94500–94599; new
    salt 9014 (9015 reserved).
  - **Budget:** about 0.5 GPU h (0.4–0.7), ceiling 1.0; Stage 0 escalates above a 0.6 GPU h
    projection for the gated stage.
  - Development behind it (CPU only): `~/develop/emai/evidence/task094-draft/` (power simulation).
  - Next: independent review and merge of the DRAFT, then the Stage 0 PR (code, tests,
    development checks), which freezes the protocol.
- 2026-10-15, independent review of PR #225 (REQUEST-CHANGES, textual) applied in the DRAFT: K-NULL
  also requires the K-0.6 contrast's bound below 0.15 (else K-UNCLEAR); the bootstrap bound is the
  97.5th percentile of the two-sided 95 % percentile interval; Rs's window holds no lift, and the
  record names the root type of an L-COST. Non-blocking suggestions adopted: positive control
  56 / 64 (28 / 32 at Stage 0), the seed-independence approximation disclosed, a named L-UNREAD
  row, ik's contact 302 / 320 in δ_K's basis, gated K at 0.09–0.13 GPU h. The power evidence in
  `task094-draft/` is unchanged; Stage 0's recompute simulates K-NULL's joint condition.
- 2026-10-15, Stage 0 run and recorded (branch `feat/task094-stage0`; §16; R27.177–R27.182,
  DECISIONS (bj); evidence `~/develop/emai/evidence/task094-s0/`). Development, one run,
  simulation only: K on 94100–94131 (all 11 arms plus the frozen `ik`; plumbing passes on every arm
  after the check's correction, K-1.0 bitwise `ik`, K-1.0 23 / 32, K-open 0 / 32), M and S on
  940000–940031 (roots and squeeze bitwise, positive control 32 / 32 at both root types), power
  0.997 / 0.978, 0.108 GPU h used and 0.257 projected. **S0-ESCALATE:** n_d 6 (Rs) and 1 (R2) of
  32, below 8 at both. The protocol is not frozen; gated cohorts untouched. Next: a ruling on the
  options of §16.6 (run as preregistered with M likely unread, amend M, K alone, or close).
- 2026-10-15, independent review of PR #226 (REQUEST-CHANGES, textual) applied in the record:
  K-off-y-s\* interval 0.11–0.43; §16.5 states that S's figures do not show G preferring a partial
  closure (CEM's best candidate cost less than the squeeze branch on 192 / 192 calls, so §6's miss
  could not fire; ℓ read at step 8); R27.181's last bullet reworded; the pose term beside its hand
  part; a "grasp subgoal timed out" column in the K table.
- 2026-10-15, escalation ruling recorded (R27.183–R27.186, DECISIONS (bk); protocol §17): option 2,
  extended, adopting the review's amendment items 1–7 word for word. Disclosed: the S definition
  changes are motivated by development plans; the development K pattern makes K-UNCLEAR the likely
  gated K row; the +y placement effect is larger than the closure effect. Protocol stays DRAFT.
  Next: the Stage 0′ PR (code and tests for S's ℓ and comparator, ℓ recomputed on CPU, one GPU job
  for the substituted costs on 940000–940031, seed check of 946000–946127, budget projection), which
  freezes the protocol at its merge.
- 2026-10-15, Stage 0′ (branch `feat/task094-stage0prime`; R27.187–R27.192, DECISIONS (bl)):
  S0-PASS. No new development cohort; one 6 s GPU job on Stage 0's stored plans; projected n_d
  24 (Rs) and 4 (R2) at 128 roots; gated stage projected at 0.352 GPU h. Evidence
  `~/develop/emai/evidence/task094-s0p/`. The protocol freezes at the PR's merge.
- 2026-10-15, gated stage run once at `2eacacf` from the clean run tree `worktrees/task094-run`
  (branch `docs/task094-gated-record`; §19; R27.193–R27.199, DECISIONS (bm); evidence
  `~/develop/emai/evidence/task094-gated/`). 17 jobs, 0.286 GPU h (task total 0.395). The stage exited
  3 on two events, recorded and not fixed: the roots job VOID at root 946073 (Rs re-run not bitwise;
  L-VOID), and the summary's K-VOID on a size check of the determinism re-run reports. By §4.3's text
  K reads K-NULL, so the grasp line pauses and the owner is told. Next: the independent review
  confirms or rejects the K-NULL reading; a pause ruling, and any Phase 4 precursor ruling, are the
  owner's.
