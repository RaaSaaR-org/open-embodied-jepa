---
id: TASK-035
aliases:
- TASK-035
title: Collect matched action branches and test causal world-model predictions
slug: collect-matched-action-branches-and-test-causal-world-model-predictions
status: in-progress
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- counterfactual-data
- world-model
sprint: ''
depends_on:
- "[[TASK-031]]"
due_date: ''
created: 2026-09-21
updated: 2026-09-21
---

# Matched action branches and causal prediction

## Motivation

The first sensor model predicts motion better than persistence but has only ~1%
aggregate benefit from true versus shuffled actions, with inconsistent phase-level
benefit. Early bounded control stalls before reach. The next experiment must
improve counterfactual action coverage rather than assume more epochs solve it.

## Acceptance criteria

- Preregister a bounded branch-collection protocol: exact parent reset/root cohort,
  action alternatives, branch horizon, sample count, resource limits and stopping
  rule before physics. No changes to mechanics, limits or success scoring.
- Collect multiple actual executed-action futures from identical restored simulator
  roots. Privileged simulator snapshots are collector internals only; canonical
  model inputs remain RGB and measured robot state. Validate identical root
  observations and replay/restore semantics, including previous actuator targets.
- Keep all sibling branches and nominal parent trajectories in the same session
  partition. Never move historical test episodes into training; retain failed or
  rejected branches and missing-attempt accounting. Preserve full nominal training
  demonstrations for image goals.
- Record applied actions, complete T+1 sensor sequences, branch provenance and
  physical future differences. Demonstrate that interventions alter observable
  futures, rather than merely changing command parameters.
- Train under a newly registered budget and assess action-conditioned prediction
  against shuffled/persistence controls plus actual paired candidate ranking.
  Keep v1 checkpoints/data/control results immutable.
- Review, tests, reproducible manifests, PR and verified authorized merge. Scientific
  failures remain evidence; mark implementation and research gates separately.

## Ownership and scope

Collector/phase-diagnostic agent owns the prospective branch-data design and new
collector files. Coordinator owns MC/Git and resource scheduling. No full branch
collection starts before protocol review/commit or while the initial control
diagnostic is using the same CPU. Current task-specific physics is MuJoCo only.
