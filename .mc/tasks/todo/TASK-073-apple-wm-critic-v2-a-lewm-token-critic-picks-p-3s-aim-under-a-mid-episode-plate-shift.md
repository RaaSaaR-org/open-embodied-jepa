---
id: TASK-073
aliases:
- TASK-073
title: 'Apple WM critic v2: a LeWM token critic picks P-3''s aim under a mid-episode plate shift'
slug: apple-wm-critic-v2-a-lewm-token-critic-picks-p-3s-aim-under-a-mid-episode-plate-shift
status: review
priority: 1
owner: ''
projects: []
customers: []
tags:
- world-model
- lewm
- critic
- preregistration
sprint: ''
depends_on:
- "[[TASK-066]]"
- "[[TASK-072]]"
due_date: ''
created: 2026-09-29
updated: 2026-09-30
---


# Apple WM critic v2: a LeWM token critic picks P-3's aim under a mid-episode plate shift

## Description

The first step that puts a LeWM world model into the control loop (`apple_wm_critic_v2`). On
`apple-to-plate-v2`, with the plate moved a few cm after the grasp latches (a simulation-only
diagnostic condition, owner D1), TASK-072 run-1's P-3 proposes 25 aims around a mid-episode plate
re-read, a LeWM predictor over frozen DINOv2 4 × 4 patch tokens (TASK-066's latent, h = 16)
predicts where each leads, and the aim whose predicted apple-minus-plate offset is nearest the
frozen target is executed. Credit only against the same controller with an action-blind
(P-reread / H-N), scene-blind (H-shuf) or random (H-rand) chooser, on fresh resets.

Owner decisions of 2026-09-29: D7 "Critic first, planner fallback" (TASK-074, the LeWM-only
planner, pre-authorised as the fallback if TASK-073 ends S-NO-CONDITION, H-NO-HEADROOM or
NO-HEADROOM; not designed here); D1/D2 "Accept, diagnostic only"; D3/D4/D6 "Use the defaults".

Stages, each a separate reviewed step: Stage 0 preregistration PR (this card's first PR); K0
condition calibration (development); `apple-shift-v2` corpus; W/N training and offline gates
O1-O5 (+ O0); D3 development closed loop; gated S/U only on a separate owner authorisation
record; results PR.

Learned Apple→Plate on the frozen benchmark is still 0 successes. An H-LeWM success would be a
learned policy trained on privileged expert labels plus a LeWM critic, not LeWM driving the robot,
and not the v1 benchmark. No control line is primary.

## Acceptance Criteria
- [x] Stage 0: protocol `docs/experiments/apple_wm_critic_v2.md`, manifest
      `benchmarks/manifests/apple-wm-critic-v2.json` (frozen block, stored cohort values and
      digests, pins), modules, scripts and tests; smokes on 53950-53999 only; merged on an
      independent reviewer's reported APPROVE.
- [x] K0 on a fresh pre-run reviewer's reported GO, orchestrator told first (run-1 V; run-2 after
      the #108 fix): **S-NO-CONDITION**; results PR pending review.
- [x] Corpus, training, ranking, offline decision, D3, gated S/U: **not run by rule** (TASK-073
      ends at K0 with S-NO-CONDITION; the D7 fallback TASK-074 is authorised).
- [ ] Results PR, reviewed; every restated number checked.

## Notes
- 2026-09-29: card opened; branch `feat/task-073-wm-critic-v2` from `70f1358` (main after #104).
  `mc` is not installed on the Linux PC; MC validation was not run.
- 2026-09-29 facts checked in code: EXPERT_BUDGETS as the outline; e9's pick phases invariant
  under a plate shift; `lewm.py` 14-D `Embedder`, `max_horizon` 64 (no implementation-hash
  change); 53000-53999 and 7300-7313 unused; P-3 checkpoint sha256 `7988162d…60be8` equals M2's
  pin; M2's P-3 final distance q10/q50/q90 2.28/2.82/3.27 cm in the merged results manifest.
- 2026-09-29 design corrections (protocol §10): the outline's "non-clamping arc" is empty (e9's
  release is always pulled back to its reach sphere), so shifts keep e9's plate-relative release
  geometry (sideways arcs); o* frozen now from apple-look-v2-linux; O2 on TASK-054's moving
  cohort plus a clock-prior ratio (the clock prior passes G2a's ratio by itself); O0 headroom
  precondition before the ranking gates (on smoke seeds with the true plate as incumbent the
  prior-distance ranker reaches median ρ 0.63 and the incumbent's regret is 0.03 cm).
- 2026-09-29: PR #106 REQUEST CHANGES at c8ea8a5 (macOS CI red on an exact-float cohort test; H-shuf
  stage void risk; G7/determinism scope; GO procedure; N1-N9). Owner rulings the same day:
  R-NO-HEADROOM is a fallback row; G7 bar kept, G7-only failure = HYB-SLOW; pin threading + quiet
  machine, a pre-render V is repeatable as-is; D-8/D-17/D-19 accepted as set after smoke data;
  HYB-HARM checked first and fires the clause. All implemented in the follow-up commit.
- 2026-09-29: PR #106 merged (`b4df3f0`); pre-run GO reported. **K0 run-1 V**: an external host agent
  SIGTERMed the run after memory pressure (16 workers, cgroup peak 25.9 GiB), after the first cohort
  render; no report was written. Owner: "Fix PR, then repeat (Recommended)"; "I'll tell Hank to leave
  gated runs alone". Fix PR `fix/task-073-memory`: 6 simulation workers, a 12 GiB process-tree
  ceiling with G-memory and a runtime guard, V reports on SIGTERM/SIGINT/SIGHUP, the corpus's
  first-render boundary, and render-only retries/majority for the renderer's rare one-level pixel
  differences (protocol §15).
- 2026-09-30: #106 merged as `b4df3f0`, #108 as `35772e5`. **K0 run-2** from `35772e5` (GO: #108 comment
  5901205401), 2 015 s, report sha256 `f760af40…27f2`: **S-NO-CONDITION**. Best cell: 300/4 cm
  B-oracle-shift 32, P-stale 4, P-truth 28, H-sim 31 (+3 < +4). The abandonment clause does not fire;
  the D7 fallback (TASK-074, LeWM-only planner) is authorised. The LeWM critic never ran. Results:
  `docs/experiments/apple_wm_critic_v2_results.md` (branch `docs/task-073-results`). `mc` is not
  installed; MC validation was not run.
