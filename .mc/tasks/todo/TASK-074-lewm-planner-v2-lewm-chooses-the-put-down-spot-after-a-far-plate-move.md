---
id: TASK-074
aliases:
- TASK-074
title: 'LeWM planner v2: LeWM chooses the put-down spot after a far plate move'
slug: lewm-planner-v2-lewm-chooses-the-put-down-spot-after-a-far-plate-move
status: review
priority: 1
owner: ''
projects: []
customers: []
tags:
- world-model
- lewm
- planner
- preregistration
sprint: ''
depends_on:
- "[[TASK-066]]"
- "[[TASK-072]]"
- "[[TASK-073]]"
due_date: ''
created: 2026-09-30
updated: 2026-09-30
---


# LeWM planner v2: LeWM chooses the put-down spot after a far plate move

## Description

TASK-074 is the owner's D7 fallback after TASK-073 ended S-NO-CONDITION (`apple_lewm_planner_v2`,
`lewm-planner`).

**The formulation.** On `apple-to-plate-v2` the plate moves 9 or 12 cm to the robot's right (−y)
at step 300. This is a simulation-only diagnostic condition. P-3 does the pick. From step 405 a
LeWM predictor over frozen DINOv2 4 × 4 patch tokens (TASK-066's latent, h = 16) chooses where
e9's scripted place primitive puts the apple:
- the candidates are 74 put-down spots per decision (coarse, then fine);
- the cost is the predicted apple-minus-plate offset against e9's release offset.

**Credit** is only against:
- P-3 given the true plate (P-truth);
- the action-blind (L-N), scene-blind (L-shuf) and random (L-rand) choosers;
- and within 6/64 of a readout-only twin (H-twin).

P-far, a retrained BC policy, is a reported control.

**The owner's decisions of 2026-09-30, verbatim:**
- D1 "Beat P-3, tie readout (Recommended)";
- D2 "Put-down spots (Recommended)";
- D3+D4 "Right-side move, place only (Recommended)";
- D5 "Yes, add it (Recommended)".

**The owner's explicit delegation of 2026-09-30, verbatim:** "do the work without me - if you
have decidions, choose your recommandation. do the work in subagents, use this chat just for
updates. use subagents and workflows. goal is to continue working and try to get a real LeWM for
the Unitree G1 without my help".

Every other ruling is therefore delegated, and is labelled "decided by Claude under owner
delegation, 2026-09-30". This includes the gated authorisation:
- the gated stage may run on an authorisation record signed by the task owner, or by Claude under
  this delegation;
- either way, only after a pre-run reviewer's reported GO;
- the main session posts a notice in chat before each gated run.

**The design probes** used development seeds 54700–54999, which are now spent. The bars were set
after seeing them, and the protocol discloses this.

**Stages,** each a separate reviewed step:
1. the Stage 0 preregistration PR (this card's first PR);
2. K1, the condition gate (development);
3. the `apple-far-shift-v2` corpus;
4. W/N training, P-far, ranking and the offline decision (O1–O5);
5. the D3 development closed loop;
6. gated S/U, only on a separate owner or delegated authorisation record, after a reported
   reviewer GO, with a chat notice before each gated run;
7. the results PR.

Learned Apple→Plate on the frozen benchmark is still 0 successes. An L-plan success would be
"P-3's learned pick plus a LeWM planner choosing where a scripted place primitive puts the
apple", not LeWM driving the whole episode, and not the v1 benchmark. No control line is primary.

## Acceptance Criteria
- [ ] Stage 0: protocol `docs/experiments/apple_lewm_planner_v2.md` and manifest
      `benchmarks/manifests/apple-lewm-planner-v2.json`, with the frozen block, the stored cohort
      values and digests, the corpus plan digest and the pins.
- [ ] Stage 0 code: the modules, the runner and the tests. The #108 follow-ups are fixed first
      (#111) and used through `run_guards`.
- [ ] Stage 0 smokes on 54650–54699 only, with the train stage's memory measured at full corpus
      scale.
- [ ] Stage 0 merged on an independent reviewer's reported APPROVE.
- [ ] K1 on a fresh pre-run reviewer's reported GO, with the coordinator told first.
- [ ] Corpus, training, P-far, ranking and the offline decision, each on its own reported GO.
- [ ] D3, on its own reported GO.
- [ ] Gated S/U only after the owner or delegated authorisation record is in the manifest,
      validated by `check_authorisation`, after a reported reviewer GO and a chat notice.
- [ ] The results PR, reviewed with every restated number checked.
