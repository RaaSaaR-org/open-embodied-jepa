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
- 2026-09-27: PR #78 (preregistration) merged as `ecb1300`.
- **2026-09-27T13:13Z, owner ruling R6**, received via the coordinator and also posted on #78:
  both §17 departures are accepted (C-3's own DAgger ×3; R-3 on D). Recorded verbatim in the
  protocol's §0 and in the manifest in PR 2.
- PR 2 (implementation) adds:
  - `first_policy_runtime.py`, `first_policy_model.py`, `first_policy_perception.py` and
    `scripts/run_first_policy.py`;
  - `tests/test_first_policy_runtime.py`;
  - the manifest's `hashes` and `encoder_digests`;
  - protocol §18, amendment 1: the a4 rate cap, the counter after the reset, `D_RESETS`, the
    smoke seeds, the estimate path, no-eligible cascading and S0-D1 timing.
- CPU smokes only (`outputs/task067-scratch/smoke-*`, seeds 46900–46999, noise targets); nothing
  in them is read. No MPS training or evaluation before the GPU release.
- **Differences from the proposal: protocol §17 and the manifest's
  `differences_from_proposal`.**
  - **Two need owner confirmation before PR 1 merges:**
    - C-3 gets its own DAgger × 3, where the proposal's C-noimg was BC-only;
    - R-3 is added to M1.
  - The A4-look trigger is calibrated from C0 and S0-P, as proposed. A first-draft 8/16 was
    replaced after the PR #78 review.
- 2026-09-27T13:56Z, **owner ruling R7** (posted on #79): amendment 1 items 4–10 accepted, with a
  condition on item 5 (render bit-identity at the gated worker count before GO). PR #80 added
  `scripts/check_first_policy_render.py`, merged as `9b0c580`.
- Pre-run reviewer's reported verdict: **PRE-RUN: GO** at `9b0c580`. Its R7 render check was
  IDENTICAL (8 workers, 32/32 seeds on ≥ 2 workers, negative control raised G-frame; sha256
  `b5de81a7…336f`).
- 2026-09-27T23:06:14Z–23:17:33Z UTC: gated run-1 (`outputs/task067-first-policy/run-1`,
  report sha256 `4da883de…eb3b`). **Outcome CAL-ESCALATE** at C0: the plate at 1.0 cm scored
  21/32, below 28/32. The clause does not fire. No policy was trained; D and C were not simulated.
  The owner decides next (results §5).
- 2026-09-27T23:36Z, **owner ruling R8**: option (a), probe first. Recorded verbatim in the probe
  document.
- Release-point probe (PR #85), declared at `e694a45` before running. Seeds 46800–46831; seeds
  46832–46863 are reserved for any v2 calibration. Row **P-CANDIDATE-FAIL**:
  - The current collector scores 32/32 at the reference, landing a median 2.7 cm short. At a
    1.0 cm plate error it scores 21/32.
  - The mechanism counts as confirmed by the declared rule only. The per-attempt data contradict
    it as the cause: at 1.0 cm every failure of the current collector lands forward or sideways
    at the rim (4.53–4.62 cm), not short.
  - The release-at-centre candidate scores 15/32 at the reference and 23/32 at 1.0 cm, below
    28/32.
  - Under R8 the work stops and goes back to the owner.
- R9 step 2 (PR #87): landing diagnosis at `b664ff6`. The mechanism was identified from the
  logs:
  - the apple drops about 15.5 cm, is carried and rolls to the rim within about 0.8 s, and ends
    rolling slowly along the rim at 4.52–4.60 cm, outside the 4 cm radius (corrected after
    review: not a bounce);
  - the scorer's successes are mostly transient crossings: 32/32 and 21/32 at any step, against
    4/32 and 1/32 at rest.
- R9 step 3: place-then-open at the plate centre, declared at `9088236` before running, on seeds
  46864–46895. **Row FAIL:** 25/32 with the plate exact and 20/32 at 1.0 cm, both below 28/32.
  - The palm did not reach the place pose within the 100-command budget (median lowest point
    0.113 m against a 0.035 m target). It was still descending at about 0.44 mm/step with the
    command saturated, and there was no hand–plate contact while lowering, so the declared risk
    did not occur and the intended place was never tested.
  - The apple still fell about 8 cm and reached the rim on 32/32.
  - Under R9 the work stops; closing TASK-067 is the fallback.
