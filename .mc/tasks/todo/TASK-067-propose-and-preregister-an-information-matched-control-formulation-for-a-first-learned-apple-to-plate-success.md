---
id: TASK-067
aliases:
- TASK-067
title: Preregister and run an information-matched learned policy for a first learned Apple->Plate success on the development cohort
slug: propose-and-preregister-an-information-matched-control-formulation-for-a-first-learned-apple-to-plate-success
status: in-progress
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- control
- policy
sprint: ''
depends_on:
- "[[TASK-064]]"
due_date: ''
created: 2026-09-27
updated: 2026-09-27
---


# Preregister and run an information-matched learned policy for a first learned Apple->Plate success on the development cohort

## Description

The proposal (`docs/experiments/control_formulation_proposal_v1.md`, PR #77, merged as `f2e9f63`)
was accepted by owner rulings R1–R5 on 2026-09-27T12:43Z. They are recorded verbatim in the
protocol's §0 and in the manifest.

**Protocol:** `docs/experiments/apple_first_policy_v1.md`.
**Manifest:** `benchmarks/manifests/apple-first-policy-v1.json`.
**Design module:** `src/embodied_jepa/first_policy.py` (pinned by `tests/test_first_policy.py`).

**What is tested.** A learned policy P gets, at run time, what the scripted expert acts on and
nothing privileged:
- apple and plate xy, read once from the post-look frame by a kernel-ridge readout on frozen
  DINOv2 tokens;
- its own step counter;
- proprioception;
- the palm pose by forward kinematics.

P is trained by behaviour cloning on the non-aim `apple-look-v1` train roots, then by 3 DAgger
iterations labelled by the privileged expert, at training time only.

**The camera and its resolution are unchanged** (112 px onboard). The change is in the
information the controller receives (R1).

**Stages.**
- **C0:** the expert's tolerance curve, which sets the perception bars. They can only tighten.
- **S0-P:** perception on 128 fresh held-out resets.
- **S0-D1:** pipeline equivalence.
- **M1 on cohort D:** P-0 to P-3, the controls C-3 and R-3, and the diagnostics A4-look,
  D-oracle-perc and B-oracle/hold/random/replay.
- **M2 on cohort C:** preregistered now, run only under a separate authorization.

M1-PASS is an existence result on the development cohort. It is labelled "learned policy with a
DINOv2 encoder", never "LeWM driving the robot" (R4). **Learned Apple->Plate is still 0
successes.**

## Acceptance Criteria
- [ ] PR 1 (the protocol, the manifest, `first_policy.py`, its tests and this card) merges on an
      independent reviewer's reported APPROVE and green CI.
- [ ] PR 2 (the runner, policy and training code, guard and row tests, pins) merges on a reported
      APPROVE and green CI.
- [ ] The gated run starts only after TASK-066's run has finished and the coordinator has
      released the GPU, on a fresh pre-run reviewer's reported GO, from a clean tree.
- [ ] PR 3 states the row and every quantity. A reviewer checks every restated number against
      `report.json`. If the clause fires, PR 3 adds a DECISIONS entry.
- [ ] The test split is never decoded. Cohort C is not opened. `exemption_spent` stays false.

## Rulings and log (UTC)
- 2026-09-27: PR #77 (proposal) merged as `f2e9f63`.
- **2026-09-27T12:43Z, owner rulings R1–R5**, received via the coordinator and recorded verbatim
  in the protocol's §0.
  - R1: allowed; disclose that the camera is unchanged; the rationale is an information change.
  - R2: the learned inputs are accepted, labelled in the ladder.
  - R3: the primary, fallback and milestone are accepted, and the dev result is existence only.
  - R4: "learned policy with a DINOv2 encoder", never "LeWM driving the robot".
  - R5: seeds 46000–46999, subject to a recorded overlap check.
- A later coordinator correction changed only the year of the TASK-066 GPU hold: about
  2026-09-28T00:10Z, hard cap 04:08Z. No compute runs before that hold is released.
- 2026-09-27: the R5 seed-overlap check at `f2e9f63` found no overlap (protocol §4.1).
- **Differences from the proposal: protocol §17 and the manifest's
  `differences_from_proposal`.**
  - **Two need owner confirmation before PR 1 merges:**
    - C-3 gets its own DAgger × 3, where the proposal's C-noimg was BC-only;
    - R-3 is added to M1.
  - The A4-look trigger is calibrated from C0 and S0-P, as proposed. A first-draft 8/16 was
    replaced after the PR #78 review.
