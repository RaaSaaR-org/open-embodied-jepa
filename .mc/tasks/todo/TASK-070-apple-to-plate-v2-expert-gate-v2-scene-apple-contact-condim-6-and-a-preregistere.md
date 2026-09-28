---
id: TASK-070
aliases:
- TASK-070
title: 'apple-to-plate-v2 expert gate: v2 scene (apple contact condim 6) and a preregistered at-rest expert gate'
slug: apple-to-plate-v2-expert-gate-v2-scene-apple-contact-condim-6-and-a-preregistere
status: in-progress
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- scripted-expert
- task-definition
sprint: ''
depends_on:
- "[[TASK-069]]"
due_date: ''
created: 2026-09-28
updated: 2026-09-28
---

# apple-to-plate-v2 expert gate: v2 scene (apple contact condim 6) and a preregistered at-rest expert gate

## Description

Owner ruling R12 is recorded verbatim in `docs/experiments/apple_to_plate_v2_expert.md`. It
defines **apple-to-plate-v2 = v1 + apple contact condim 6, and nothing else**.
- The plates stay as in v1. b2 and b1 are rejected.
- The friction values are the v1 scene's own declared values, `1 .01 .001` (R12 §3 (i)). They
  are not chosen from any scan.
- No cited-source replacement is taken, so no other value is tried under v2 (R12 §3 (ii)–(iii)).

TASK-070 does four things:
1. Add the v2 scene option in code, separately named and versioned (`apple_to_plate_v2`). v1
   code and history stay untouched.
2. Improve the privileged scripted expert on a fresh development seed range under the fixed v2
   physics. d12 is the starting point, and every design is logged.
3. Open a preregistration PR that freezes the expert, the v2 physics with its source,
   `apple_at_rest_v0` and fresh gated seeds.
   - The gate: at rest ≥ 28/32 at plate exact AND ≥ 28/32 at 1.0 cm plate error.
   - The latched scorer and the 1.5 cm level are reported beside the gate.
4. Run the gate only on the pre-run reviewer's reported GO, then open a results PR.

This is privileged scripted engineering, not a learned result. **Learned Apple→Plate is still 0
successes.**

## Seeds
- **Development: 50200–50299** (`apple_to_plate_v2.DEV_SEEDS`). Plate-error directions come from
  `default_rng(6850)`. Both are declared here, before any TASK-070 episode.
- The range is disjoint from:
  - TASK-068 (50000–50099) and TASK-069 (50100–50199);
  - every range in `first_policy.FORBIDDEN_RANGES`;
  - the 46000–46999 block.
- **Gated seeds:** declared in the preregistration PR. They will be fresh and checked.

## Acceptance Criteria
- [x] v2 scene option in code, with tests. The v1 files are byte-identical to `main`.
- [x] Development log of every design, with counts.
- [x] Preregistration PR merged on an independent reviewer's reported APPROVE (PR #92,
      rebase-merged; `1ba0557` on `main`).
- [x] Gated run started only on the pre-run reviewer's reported GO (reviewer's own worktree at
      `1ba0557`). Outcome **PASS**: 32/32 and 30/32 at rest.
- [ ] Results PR merged on a reported APPROVE.
- [x] The PR #91 old→new SHA map is added (R12 §1), once #91 is merged.

## Notes
- 2026-09-28: card opened under owner ruling R12 §4. v2 physics = R12 §3 (i), accepted by the
  owner. Development designs e1–e11 on 50200–50295 (protocol §4); e9 frozen. Preregistration
  PR opened with gated seeds 50600–50631. Learned Apple→Plate is still 0 successes.
- 2026-09-28: PR #92 rebase-merged (`1ba0557`), and the pre-run reviewer reported GO. Run-1 ran
  from a clean checkout of `main` at `1ba0557`: 96 attempts, 0 errors, 0 early stops.
  - At rest: **32/32 at plate exact and 30/32 at 1.0 cm → PASS**; 28/32 at 1.5 cm (reported
    only).
  - Latched: 32/32 at every level.
  - Report sha256 `27543757…`.
  - Results in protocol §8.
  - e9 was selected on the development seeds. This is a privileged scripted expert, not a
    learned result. Learned Apple→Plate is still 0 successes.

%% mc-links: [[TASK-069]] %%
