---
id: TASK-056
aliases:
- TASK-056
title: Close the loop on the encoder's perception with behaviour cloning and a readout-driven controller
slug: close-the-loop-on-the-encoders-perception-behaviour-cloning-plus-a-read
status: in-progress
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- learning
- evaluation
sprint: ''
depends_on:
- "[[TASK-054]]"
due_date: ''
created: 2026-09-24
updated: 2026-09-24
---


# Close the loop on the encoder's perception with behaviour cloning and a readout-driven controller

## Description

TASK-054 fired Outcome B: all four arms failed the primary gate G2a (0.8763 / 0.8814 / 0.8635 /
0.9036 against ≤ 0.8), the untouched control passed more gates (10/14) than every intervention
(9/14), and the clause carried from TASK-052 fired as written — CEM over this cost is abandoned
and behaviour cloning with the world model as a critic or residual becomes the primary line.

This task implements the behaviour cloning. It **deliberately does not implement the critic**, and
the preregistration says why before any number is seen: a critic is `Q(s, a)`, a counterfactual
action-conditioned value, and every action-conditioned quantity in this model has failed its gate.
Building one re-imports the failed component under a new name.

The basis for the task is a narrow, checkable **hypothesis**: *the encoder's perception on a
nearly-current frame is good enough for a reactive controller, and it was never the gated
quantity.* **It is not yet supported by a direct measurement, and the pre-flight is its first
test.** The only directly-encoded (horizon-0) numbers in the committed record are E0's
`encoded_start`/`encoded_target` of **2.3536 / 2.5903 cm on the moving cohort** — exactly the
2.35–2.59 cm README and DECISIONS publish — and **both exceed the pre-flight's 1.5 cm threshold**.
The low figures often quoted (0.78 cm all-windows, 0.98 cm grasp cohort, `apple_held` AUROC
0.99949) are readouts of **h-step action-conditioned rollouts**, not perception measurements; an
independent review caught an earlier draft of this protocol presenting them as the latter. Only
the h = 1 figures bear on the hypothesis, and only by inference across a one-step predictor.

The preregistration carries the refutation of the wider claim in the same section, so no later
reader can mistake this for a dynamics result: on the moving cohort the model beats its own
persistence readout by only 5 % at h = 1, and `held_lift_shuffled_auroc` is 0.99737 at h = 1 —
a readout computable just as well from a *different window's* actions is reading the frame, not the
dynamics. **Both control formulations in this task consume only the h ≈ 0 perception and neither
ever asks the model what would happen under a counterfactual action.**

Protocol: [docs/experiments/apple_policy_v1.md](../../docs/experiments/apple_policy_v1.md).
Frozen manifest: `benchmarks/manifests/apple-policy-v1.json`.

## Scope

Five arms on one frozen cohort of 40 fresh wide-jitter resets (seeds 45300–45339, cohort sha256
`4f888154c055bcbbbc0df333442e6886d8f65c2389596fe1aedde1d2cb0b533e`, never simulated), plus four
non-learned references (`scripted_oracle`, `demo_replay`, `hold`, `random`).

- **A0** proprioception-only cloned policy — control: is the policy just a clock?
- **A1** cloned policy on a frozen *randomly initialized* encoder — control: do any visual features
  suffice, or does pretraining matter?
- **A2 (PRIMARY)** behaviour cloning on the frozen E0 encoder's `image_features` + proprioception.
- **A3** the same, fine-tuned end-to-end.
- **A4** the existing `scripted.apple_collector_policy` phase machine driven by the model's own
  readouts instead of `initial_truth`. Trains nothing. **Not a learned policy and never reported as
  one** — learned perception driving a scripted controller.

New code: `src/embodied_jepa/policy.py`, `src/embodied_jepa/cloning.py`,
`scripts/evaluate_policy.py`, `scripts/measure_policy_preflight.py`, a `POLICIES` registry,
`configs/apple_policy_v1.yaml` + one-line arm files.

## Acceptance criteria

- Preregistration and frozen manifest merged **before** the gated run, carrying the cohort resets,
  gate thresholds, budget, arm configs, dataset/split/action/checkpoint hashes, code revision,
  seeds and device.
- **Pre-flight P1/P2/P3 measured before any training starts.** P1 (close-phase palm–apple readout
  ≤ 1.5 cm) is a **binding abort**: on failure the task stops, the per-phase table is recorded, the
  outcome is E, the runner exits non-zero, and the agent stops and reports to the coordinator.
  "P1 failed but we proceeded" is not an available outcome. **P3 is a hard precondition for A4**: on
  failure A4 does not run, recorded as a pre-run protocol change with its reason, and Outcome D is
  reported as reached on partial evidence.
- Development stop rule honoured: an arm reaching grasp on 0/16 development resets does not open
  cohort C.
- Gates evaluated as preregistered and published for every arm whatever they say: **G1 ≥ 17/40 full
  successes and > demo_replay on C (primary); G2 ≥ +8 over A0; G3 ≥ +8 over A1; G4 ≥ 20/40 grasp;
  G5 hold 0 / random 0 / scripted_oracle ≥ 38; G6 zero privileged reads; G7 ≤ 100 ms median control,
  zero deadline misses.** A gate that cannot be evaluated counts as failed.
- Exclusions applied and the surviving counts asserted against the manifest: aim-offset roots
  (40 total, 33 train), all 597 branches, post-displacement frames → **137 / 15 / 8 surviving roots**.
- Risk R1 recorded: ≈ 6 165 close-phase training transitions, and the branch exclusion removes the
  corpus's only dense close-phase coverage.
- `src/embodied_jepa/models/base.py` and `models/lewm.py` unmodified, with a branch CI check that
  loads `checkpoints/task054-wm-v4/leworldmodel_baseline.pt` and asserts implementation hash
  `4ad0a6856d847aaf724daba5e721c078ee599658d4733357ab8703eccbb15a92`.
- `scripts/evaluate_policy.py` imports `wide_reset` and the scorer rather than copying them, covered
  by the existing equality test extended to the new module.
- Backend swap remains a one-line `world_model.backend` change; robot, dataset, task, scorer, action
  schema and embodiment untouched.
- `scripts/measure_policy_preflight.py` committed, and it re-derives the four
  `unverified_pending_P0` quantities (U1–U4) that currently exist only in a git-ignored run
  artifact, writing them to a committed report.
- Budget respected: ≤ 28 800 s global, ≤ 300 s per attempt, ≤ 7 200 s per training arm.
- No prior evidence overwritten under `data/`, `checkpoints/`, `outputs/`; the `apple-wide-v1` test
  split never decoded; **TASK-034's cohort 44000–44019 and thresholds not touched**; cohort C
  consumed once and never reused.
- Results document `docs/experiments/apple_policy_v1_results.md` records the outcome, all arms'
  numbers, the realized McNemar discordance and exact p for G2/G3, every failure and negative
  result, and the limitations — including that every difference between arms is a single-run
  difference.
- Independent pre-run review, independent post-run review, PR and verified merge/CI.

## Pre-declared outcomes

E (P1 fails → stop before training) takes precedence over all; A (G1+G2+G4 pass → proceed to
TASK-034's fresh-cohort validation) takes precedence over B (grasps but does not place → hand
place to the v3 phase machine, do not retrain). C (A4 succeeds, A2 does not → the demonstration set
is the binding constraint; collect ≥ 500 clean roots with close oversampled). D (both fail →
**the line stops; no third control formulation is preregistered on this corpus and this camera**).

## Progress

- **2026-09-24** — preregistration and frozen manifest merged as `3006b3b` (PR #33), after an
  independent review that returned BLOCK on the first draft (six findings, all confirmed and
  fixed) and APPROVE WITH NON-BLOCKING COMMENTS on the second (six further items, all fixed).
  CI green on macOS and Linux.
- **2026-09-24** — coordinator authorized the gated run through the **development stage only**:
  pre-flight P0–P3, then (if P1 passes) feature precompute and training of A0–A3, then the
  16-reset cohort-D pre-check, then stop and report. **Cohort C (45300–45339) is NOT authorized**
  and requires a third authorization. Standing constraints: `models/base.py` and `models/lewm.py`
  off-limits; no threshold may be adjusted in response to anything the pre-flight or development
  stage reports.
- **Carried forward as a blocking item for `scripts/evaluate_policy.py`:** it must instantiate
  each cohort-C reset from the stored manifest values and must not recompute them from
  `wide_reset`. numpy's `Generator.uniform` can differ by one ULP across platforms, so a
  recomputing runner would let two machines execute subtly different cohorts while both passing
  the digest check.

## Authorization and workflow

Design reviewed and accepted by the coordinator on 2026-09-24 with three decisions (A4 in; G1 asks
for the first learned success, not the MVP; the P1 abort rule is binding) and four required
corrections, all applied. **Correction C2 changed a threshold:** the design's `demo_replay` null of
5/24 was wrong; the value verified from committed results documents is **6/24 = 25.0 %**, so G1
moved from 14/40 (p = 0.1032 against the corrected null) to **17/40** (p = 0.0116), and G2/G3 from
+6 (McNemar p = 0.073–0.132) to **+8**. The collision with the coordinator's own decision was
surfaced rather than resolved silently; the coordinator recomputed independently, confirmed every
figure, and let the correction stand. The manifest keeps a `correction_note` recording that an
earlier draft said 5/24 and was wrong.

Rebased onto `main` = `ac9625d` (TASK-055, PR #31). Two numbers that record corrected and this
protocol states correctly: **best-ever G2a is 0.831 (v3 arm D)** — 0.8635 is only the best in v4 —
and the frozen wall-clock caps differ by generation (**6,000 s per run at v2**, 10,800 s per arm at
v3 and v4). The pivot itself is cited from `docs/DECISIONS.md` rather than restated.

An independent review returned **BLOCK** on the first draft with six findings, all confirmed and
fixed: the section-1 evidence table presented rollout readouts as perception; the pre-flight had no
no-vision control and P2/P3 were passable by a model that never reads the image (a constant-centre
predictor scores 2.394 cm under `wide_reset`, above both); the P1 provenance republished an
ejection range the cited document explicitly retracted; an h=16 row was mislabelled; P2 had no
declared consequence, leaving a loophole around the abandonment clause; and R1 misattributed the
close-phase failure mechanism. P2 and P3 were tightened from 2.5 and 3.0 cm to **1.5 cm**, and a
shuffled-frame control with a ≤ 0.7 ratio condition was added to every pre-flight gate.

The two prior frozen manifests are **not amended** by this task, even additively; the `demo_replay`
full-success null is cited from the committed results documents by file and line.

**Three separate authorizations, in order, none implying the next:** (1) merge the
preregistration, on an independent reviewer's reported APPROVE; (2) start the gated run, whose
first step is the pre-flight; (3) open frozen cohort C, after the pre-flight values, training
reports and development numbers have been reported.

**A gated run starts only on the pre-run reviewer's REPORTED verdict, delivered as a message —
never on a review file read from disk. The same rule applies to merging. One task, one agent.**
If P1 fails, stop and report; do not select a replacement plan.
