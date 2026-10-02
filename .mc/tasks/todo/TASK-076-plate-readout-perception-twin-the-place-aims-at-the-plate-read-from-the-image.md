---
id: TASK-076
aliases:
- TASK-076
title: 'Plate-readout perception twin: the place aims at the plate read from the image'
slug: plate-readout-perception-twin-the-place-aims-at-the-plate-read-from-the-image
status: todo
priority: 1
owner: ''
projects: []
customers: []
tags:
- perception
- apple-pnp
- preregistration
sprint: ''
depends_on:
- "[[TASK-075]]"
due_date: ''
created: 2026-10-02
updated: 2026-10-02
---

# Plate-readout perception twin: the place aims at the plate read from the image

## Description

TASK-075 ended OBS-NONE and its clause fired: no view reads the apple-minus-plate offset within
the place tolerance τ = 1.0 cm, and the next step is "a task or condition change". Ruling R1
(`docs/DECISIONS.md`, decision 2026-10-02, decided by Claude under owner delegation 2026-09-30)
takes the results document's §7 Option 1: change what the place consumes. The place target
becomes the plate position read from the image; the apple stays where e9's place primitive puts
it by kinematics.

Why: the reported-only plate readout is 0.49–0.68 cm (median), under τ, while the offset error
comes from the apple term (2.2–3.1 cm). τ is defined as an error in the aimed plate target. The
white-plate development diagnostic (#122) found that the plate's colour does not limit the
readout. The plate's 87.5th-percentile error reaches 1.35–1.36 cm on the hand crop, so the tails
matter.

This is not a LeWM task and runs no world model, so the TASK-075 clause does not close it. A
pass would not show that a world model is needed (TASK-074 protocol §9b); under R2, a later LeWM
task on a plate target counts as a task change only if it is paired with a condition where the
target must be predicted (a moving plate, or a plate that leaves the view), declared as such.

**The preregistration is a draft:** [`docs/experiments/apple_plate_twin_v2.md`](../../../docs/experiments/apple_plate_twin_v2.md)
(DRAFT, not frozen). Nothing may run before it is merged on an independent reviewer's reported
APPROVE, except its K0 calibration (development seeds 56000–56031), which needs a reviewer's
reported GO for K0 and runs before the freeze. Ruling R8 (DECISIONS 2026-10-02 (b)) adds a
moving-plate headroom check, Stage K-pred, because Option 1 alone cannot advance the LeWM goal.
The plan of the next tasks is [`docs/PLAN.md`](../../../docs/PLAN.md).

## Proposed shape (to be fixed by the preregistration, not by this card)

1. *Offline:* the plate readout's own floor, clock-prior and plate-hidden checks, with the frozen
   τ, on the sealed `apple-far-shift-v2-views` store (no new render).
2. *Closed loop:* H-twin, P-3's pick plus e9's place aimed at the frozen-DINOv2 plate readout of
   the onboard 112 px frame, on 32 fresh development resets under TASK-074's 9 cm condition, bar
   28/32, beside H-handover (true plate, the ceiling) and an image-free clock-prior arm (the
   control). Fresh seeds, not cohort C.
3. The budget rule and scale probe must be consistent with the frozen stage code (the TASK-066 and
   TASK-074 budget escalations are the lesson); the device is declared explicitly.

## Acceptance Criteria
- [ ] Preregistration PR (protocol, manifest, code, tests, smoke on smoke seeds only), merged on
      an independent reviewer's reported APPROVE.
- [ ] Each gated stage on its own reported GO, from a clean checkout of the merged revision.
- [ ] Results PR, reviewed with every restated number checked.

## Notes
- 2026-10-02: card opened under ruling R1. Not started; the preregistration is pending.
  Learned Apple→Plate status: see `docs/DECISIONS.md` decision 2026-10-02, R7.
- 2026-10-02: preregistration draft `docs/experiments/apple_plate_twin_v2.md` and `docs/PLAN.md`
  (branch `docs/task076-prereg-draft`); ruling R8 recorded. Status stays todo until the draft is
  reviewed and the code exists. No run, no GPU.
- 2026-10-02: the independent review of #129 (REQUEST CHANGES at `314d843`) is addressed in a
  follow-up commit. Rulings R8.1–R8.6: the canonical sentence is quoted verbatim; c_plate is measured on
  the pooled LeWM latent; an action-dependent K-pred cell is the only admitting cell;
  PRED-INFEASIBLE, TWIN-NEAR and the K0 early stops are added; O2 is reported only.
