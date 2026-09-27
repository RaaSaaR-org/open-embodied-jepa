---
id: TASK-067
aliases:
- TASK-067
title: Propose, then preregister on the owner's ruling, an information-matched control formulation for a first learned Apple->Plate success (DRAFT)
slug: propose-and-preregister-an-information-matched-control-formulation-for-a-first-learned-apple-to-plate-success
status: backlog
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- control
- proposal
sprint: ''
depends_on:
- "[[TASK-064]]"
due_date: ''
created: 2026-09-27
updated: 2026-09-27
---


# Propose, then preregister on the owner's ruling, an information-matched control formulation for a first learned Apple->Plate success (DRAFT)

## Description

**Draft card for a proposal. Nothing here is preregistered.** Since TASK-057 the project has no
primary control line: CEM over the world-model cost was abandoned at TASK-054, and the
behaviour-cloning line stopped at TASK-057. **Learned Apple->Plate is still 0 successes.**

The proposal is `docs/experiments/control_formulation_proposal_v1.md` (status: PROPOSAL). Its
reading of the two failures is that, both times, the controller had to act on information the
scripted expert had and it did not:
- **The apple position in the decision frame.** TASK-059 found it missing; the look (TASK-061)
  and the frozen DINOv2 encoder (TASK-063/064) now make it readable offline.
- **The expert's command counter.** A new label-only probe finds the expert parked on 43.3 % of
  `orient` steps (79.3 % on noise-level-0 roots). 91 of 170 `orient`→`descend` switches are taken
  from a parked state, on the counter alone.

It recommends:
- **Primary P: information-matched behaviour cloning.**
  - Inputs: a learned perception bottleneck (apple and plate xy from the post-look DINOv2
    feature), the robot's own step counter, proprioception and the forward-kinematics palm pose.
  - Training: BC on the `apple-look-v1` train roots, then up to 3 DAgger iterations labelled by
    the privileged expert on fresh resets.
- **Fallback F: a phase-decomposed, partially learned controller.** It runs only on M1-MOTOR, and
  its successes are never counted as learned.
- **The world model enters at stage 3**, as a critic of P's samples, and only if TASK-066 passes.
  Stages 0–2 do not depend on TASK-066.

The first milestone, **M1**, is development-only: P reaches at least 1/16 full successes on cohort
D with zero privileged reads. It is an existence bar. A non-learned demonstration replay has
reached 5/16 on D, so beating replay is left to the gated stage M2 on cohort C.

## Acceptance Criteria (of the proposal stage)
- [ ] The proposal PR merges on an independent reviewer's **reported** verdict and green CI. The
      reviewer checks every restated number against its source document, including the
      uncertainty qualifiers.
- [ ] The owner rules on the proposal's §12 decisions, first of all on whether a control
      formulation on `apple-look-v1` falls inside the TASK-057 clause.
- [ ] Only after that ruling: a preregistration PR (protocol, manifest, calibration C0, and the
      S0 gates with their thresholds frozen from calibration). The arms are enumerated by name,
      and `exemption_spent` is cited in the gate section.

## Notes
- **No compute before TASK-066 finishes.** Its gated run has wall-clock caps, so no stage-0 to
  stage-2 compute starts until that run completes. Documents and code can proceed in parallel.
- **Probe provenance.** `scripts/probe_expert_dwell.py`, run-3 at `04fd767`. It read the label
  sidecars of the 170 train roots only, with hashes checked. Report
  `outputs/task067-dwell/run-3/report.json`, sha256 `ee342baa…c383`.
- **Not run, on purpose.** No DINOv2 featurisation, no plate readout and no simulation. These are
  S0-P's and the preregistration's jobs.
- **Standing rules.** The test splits are never decoded. Cohort C needs a separate authorization.
  `exemption_spent` stays `false`.
