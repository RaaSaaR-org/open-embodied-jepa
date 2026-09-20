---
id: TASK-031
aliases:
- TASK-031
title: Establish apple pick-and-place mechanics and collect a targeted development corpus
slug: establish-apple-pick-and-place-mechanics-and-collect-a-targeted-development-corp
status: done
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- data
- mechanics
sprint: ''
depends_on: []
due_date: ''
created: 2026-09-20
updated: 2026-09-21
---



# Establish reproducible apple mechanics and a targeted corpus

## Acceptance criteria

- Run a preregistered exploratory apple mechanics probe, at most 180 CPU seconds; preserve all failures and use unchanged physics/action guards/task thresholds.
- Demonstrate reproducible scripted Apple→Plate success before representing demonstrations as successful training data. Privileged control remains explicitly separate from learned inference.
- Collect a new, immutable development corpus with episode-grouped train/validation splits, executed actions, RGB and measured robot state, provenance and coverage.
- Preserve original held-out Apple→Plate corpus/results. New apple training is an explicit task-specific learning experiment, not zero-shot generalization.
- Define collection count and wall-time cap before collection. If mechanical success fails, record the stage diagnosis and create a bounded corrective experiment.

## Authorization and workflow

User requested completion toward a working world-model apple pick-and-place MVP on 2026-09-20, with subagents and end-to-end delivery. Coordinator owns Git and task state. Experiments are bounded and recorded before execution.

## Execution evidence

Collected and audited32/32episodes,31oracle successes,16132transitions. Task-specific splits26/3/3,successes25/3/3. No test image decoding during audit; source/payload hashes and train-only normalization verified. DatasetSHA d70edd9daaeec58e80a4e6a743980c049335a03f2b32b7d42bbac5ec5bc579df. See benchmarks/manifests/apple-collection-v1.json. Merge pending PR5.

Coordinator acceptance: deliverable criteria met with the recorded tests and complete bounded artifact evidence. PR5 merge remains pending; no learned pick-and-place success is inferred.
