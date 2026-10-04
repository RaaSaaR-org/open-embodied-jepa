---
id: TASK-076
aliases:
- TASK-076
title: 'Plate-readout perception twin: the place aims at the plate read from the image'
slug: plate-readout-perception-twin-the-place-aims-at-the-plate-read-from-the-image
status: done
priority: 1
owner: ''
projects: []
customers: []
tags:
- perception
- apple-pnp
- preregistration
- twin-pass
sprint: ''
depends_on:
- "[[TASK-075]]"
due_date: ''
created: 2026-10-02
updated: 2026-10-04
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

**The preregistration:** [`docs/experiments/apple_plate_twin_v2.md`](../../../docs/experiments/apple_plate_twin_v2.md)
(FROZEN after K0-PASS, merged as `702a7d9`, #137). **Results:**
[`docs/experiments/apple_plate_twin_v2_results.md`](../../../docs/experiments/apple_plate_twin_v2_results.md):
**TWIN-PASS**, and K-pred **PRED-INFEASIBLE** (Branch B). Ruling R8 (DECISIONS 2026-10-02 (b)) adds a
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
- [x] Preregistration PR (protocol, manifest, code, tests, smoke on smoke seeds only), merged on
      an independent reviewer's reported APPROVE (#134 Stage 0; #137 K0 values and the freeze,
      `702a7d9`).
- [x] Each gated stage on its own reported GO, from a clean checkout of the merged revision
      (O, D, K-pred M-a/M-b and S/U at `702a7d9`, each once).
- [ ] Results PR, reviewed with every restated number checked (branch `results/task076`).

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
- 2026-10-02: the re-review at `3d8560e` is addressed by R8.7. Cell A is redesigned with L = 2, so
  the target depends on the last aim. The required predictor history is declared, H-final iterates
  to tolerance, and K0's G2 stop is now 4/32.
- 2026-10-04: freeze preparation (branch `docs/task076-freeze-prep`; R8.8–R8.13, decided by
  Claude under owner delegation). The third review's four non-blocking items are settled: an
  action-blind predictor can tie on cell A, so TASK-077 must declare its controller form, its
  training-aim spread and the twin's expected result; the history rule is c_plate + the measured
  2-step term ≤ τ_re; the remedy's limits are stated and |κ| ≥ 1 is forbidden; G2's predicted
  feasibility is reported at K0. The protocol's new §13 lists every bar's source and every row's
  action. Not frozen: the Stage-0 code and smokes, K0 and an independent APPROVE remain.
- 2026-10-04: R8.14 (decided by Claude under owner delegation): PRED-NEAR escalates without a
  clause when a headroom bar misses within noise; PRED-NONE now needs a headroom detectably below
  +8/32 (false-fire about 2–3 % per bar at a true 8/32).
- 2026-10-04: Stage 0 (§13.2 steps 1–2) on branch `feat/task076-stage0`: the stage code, the
  frozen block as module constants, the DRAFT manifest, the §5 tests and the smokes on smoke seeds
  56900–56999 ([record](../../../docs/experiments/apple_plate_twin_v2_stage0.md); rulings
  R8.15–R8.17). s1 = 525 is free of apple–plate contact; cell A is removed (median remaining
  plate motion 0.06 cm under H-final, 2 cm needed; no remedy reached it), so K-pred's row is
  PRED-INFEASIBLE (after an O-PASS; PRED-NOT-RUN without one).
- 2026-10-04: review of #134 addressed (R8.18): the harness is a verbatim, test-pinned port; the
  truth stand-in is smoke-only and guarded twice. `look2-51171.npz` was restored bit-identically,
  and a real G-repro check passes all eight facts, so K0's evidence blocker is cleared. K0 waits
  for a reviewer's GO.
- 2026-10-04: K0 ran once at `2d0bdb7` on the reviewer's GO (#134) and ended **K0-PASS**
  (τ_re = 1.0 cm; N_K(0) = 32/32; H-clock(K) 24/32; H-stale(K) 0/32; G2's predicted pass
  probability 0.9983; report sha256 `ef4b5410…c876`). Freeze PR (branch `prereg/task076-freeze`,
  R8.19, decided by Claude under owner delegation): `K0_MEASURED` in the frozen block, status
  FROZEN, the sha pin set; the GO review's non-blocking notes 3, 4 and 6 folded in. Stage O
  waits for the merge on an independent APPROVE and its own GO.
- 2026-10-04: every post-freeze stage ran once at `702a7d9` on its own reported GO (#137): Stage O
  **O-PASS** (report `77ff3758…05fd`; R-plate median 0.403 cm, upper bound 0.416 cm; the
  random-init floor reads 0.278 cm; c_plate 0.662 cm), Stage D **D-PASS** (`a84c00e2…e93f`;
  15/16, 16/16), K-pred M-a and M-b KPRED-DONE (reported only; `d512a712…106e`,
  `0c976384…9130`), and the gated Stage S/U **TWIN-PASS** (`242655f0…74326`): on S H-twin 64,
  H-handover 64, H-clock 51, H-stale 0, H-floor 64, P-stale 0; on U H-twin 31, P-stale 30,
  H-handover 31; G2 b = 13, c = 0, p = 1.22e-4. K-pred's row is **PRED-INFEASIBLE** (cell A
  removed at Stage 0): escalate, no clause, Branch B. No world model ran. Results PR on branch
  `results/task076` with rulings R10.1–R10.3 (decided by Claude under owner delegation): R7 is
  unchanged; the protocol's O2 sentence is corrected; the next step is Branch B through PR #135.
  Card moved to done.
