---
id: TASK-082
aliases:
- TASK-082
title: "LeWM under a plate law no hand-written arm is given (direction (a))"
slug: lewm-under-a-plate-law-no-hand-written-arm-is-given
status: todo
priority: 1
owner: ''
projects: []
customers: []
tags:
- lewm
- apple-pnp
- design-note
- preregistration
sprint: ''
depends_on:
- "[[TASK-081]]"
due_date: ''
created: 2026-10-08
updated: 2026-10-08
---

# LeWM under a plate law no hand-written arm is given (direction (a))

## Description

TASK-081 closed **L-PASS** under C1-M (W 118/128), but "LeWM needed" was not shown: the
non-learned H-rule, which is given the simulator's plate law, scored measurably higher (125/128).
R19.24 (decided by Claude under owner delegation) pursues R19.23's direction (1): a condition in
which no hand-written arm is given the plate law, declared as a task change.

**The design note (DRAFT):**
[`docs/experiments/apple_lewm_unknown_law_v2_design.md`](../../../docs/experiments/apple_lewm_unknown_law_v2_design.md).
Development check (R19.25; seeds 72000–72355, CPU, no world model, one run, true-plate arms):
under **U-sat** (C1-M with a saturating, swirling law after 405) the privileged ceiling scored
32/32, H-rule with C1-M's written law 23/32, the linear H-sysid 28/32 and a kernel-ridge sysid on
256 roots 32/32; **U-play** (hysteretic) left H-rule and H-sysid at 31/32. "LeWM needed" becomes
testable against hand-written arms (R9.8); against a learned non-LeWM baseline it is not
expected (no headroom).

**Recommended next step:** a preregistration under U-sat: R9.8's primary claim with G-NI against
the better of H-rule and H-sysid on R-plate's reading; R9.8's secondary claim reported, with its
power simulated in Stage 0; a reported tier against learned baselines (kernel-ridge or MLP sysid,
P-aim) as non-inferiority only. TASK-077's recipe retrained unchanged on a new 2 000-root corpus,
three seeds per arm (about 31 h on the GPU queue; one pair about 10 h), plus about 4–7 h of CPU.
Seed block 72000–74999 (72032–72099 and 72356–74899 free for the protocol), salts 8401–8412 (8405–8412 free).

**The preregistration (DRAFT, R20.1–R20.16, decided by Claude under owner delegation):**
[`docs/experiments/apple_lewm_unknown_law_v2.md`](../../../docs/experiments/apple_lewm_unknown_law_v2.md).
Condition U-sat (C1-M with the plate law after 405 replaced; a declared task change). Primary claim
per R9.8 (G-bar 112/128; G-NI against the better of H-rule with C1-M's written law and the linear
H-sysid on R-plate's reading, δ fixed at the freeze from {8, 12, 16}/128 by a declared power rule;
four McNemar tests at p < 0.01); "LeWM needed" against those hand-written arms reported only, with
power simulated in Stage 0; a reported tier of learned non-LeWM baselines (H-sysid-krr, P-aim), against
which "LeWM needed" is not expected. TASK-077's recipe retrained unchanged on a new 2 000-root
U-sat corpus (train 1 500, val 250, gate-P 250 with aims from p̂), W and N × 3 seeds (72360–72362),
U = 95 000 and G1's bars carried (no calibration jobs); readouts R-S/R-N/R-L refitted; affine_local
solver. Stages: 0 → K0 (64 resets, τ_commit, r ≤ 465, ceiling, clip-binding) → freeze → C → O →
T (about 31.4 h GPU) → G (G1–G4, R0–R3, A1–A2 on gate-P) → D (16) → S (128). Seeds: K 72400–72463,
D 72500–72515, S 72600–72727, corpus 72800–74799, debug 74800–74899; salts 8405–8412. Compute about
31.5 h GPU and 5–7 h CPU.

## Acceptance Criteria

- [x] DRAFT design note with a development feasibility check (R19.24–R19.25)
- [x] Owner direction on the recommendation (or a ruling under delegation): R20.1–R20.16
- [ ] DRAFT preregistration, independently reviewed, before any cohort seed is simulated (drafted, R20; review on its PR)
- [ ] Stage 0, freeze, and the gated stages on their GOs

## Notes

Parallel options from R19.23, not part of this card: TASK-079's precondition (an owner ruling, no
compute) and two extra W seeds under C1-M (CPU only; R-S already exists for 66801 and 66802).
