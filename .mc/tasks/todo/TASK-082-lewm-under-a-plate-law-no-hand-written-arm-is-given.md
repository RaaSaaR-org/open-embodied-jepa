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
testable against hand-written arms (R9.8), not against a learned non-LeWM baseline.

**Recommended next step:** a preregistration under U-sat: R9.8's primary claim with G-NI against
the better of H-rule and H-sysid on R-plate's reading; R9.8's secondary claim reported, with its
power simulated in Stage 0; a reported tier against learned baselines (kernel-ridge or MLP sysid,
P-aim) as non-inferiority only. TASK-077's recipe retrained unchanged on a new 2 000-root corpus,
three seeds per arm (about 31 h on the GPU queue; one pair about 10 h), plus about 4–7 h of CPU.
Seed block 72000–74999 (72356–74899 free for the protocol), salts 8401–8412 (8405–8412 free).

## Acceptance Criteria

- [x] DRAFT design note with a development feasibility check (R19.24–R19.25)
- [ ] Owner direction on the recommendation (or a ruling under delegation)
- [ ] DRAFT preregistration, independently reviewed, before any cohort seed is simulated
- [ ] Stage 0, freeze, and the gated stages on their GOs

## Notes

Parallel options from R19.23, not part of this card: TASK-079's precondition (an owner ruling, no
compute) and two extra W seeds under C1-M (CPU only; R-S already exists for 66801 and 66802).
