---
id: TASK-035
aliases:
- TASK-035
title: Collect matched action branches and test causal world-model predictions
slug: collect-matched-action-branches-and-test-causal-world-model-predictions
status: done
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

## Collection evidence

At frozen source `662a1ab`, all528 planned H16 branches completed with8448 new transitions, zero rejections/replay failures,386.535 supervisor wall seconds within600. The sealed corpus includes560 episodes/24580 transitions, TRAIN410/VAL147/TEST3, retaining every original partition. Contrast screen passes48/48TRAIN and18/18VAL roots. Dataset manifest SHA256 `6e9a5bcb38a42a27ce1118e102865db985e8e490f37d10de0075c437676f0331`. No TEST images decoded. Independent artifact audit and preregistered training/causal comparison remain pending. Source/protocol bytes stayed frozen; unrelated new audit/diagnostic files briefly affected Git's dirty flag and were moved outside the tree before final identity verification, which passed.
Independent audit passed in6.13 seconds: all528 branches and8448 new transitions, exact sibling roots, original payloads and TEST bytes, applied/raw actions, TRAIN normalization and both contrast screens verified. Training completed3000 updates in32.71 seconds, selected2700. Frozen matched diagnostic completed66/66 roots: VAL H16 ranking73.87%, raw error reduction6.26% (95% interval4.80–8.81%), below10% primary gate. H8 secondary passed. This task delivered the experiment and retained its negative primary outcome; successful manipulation remains TASK033/034. PR #5 review/delivery pending.
%% mc-links: [[TASK-031]] %%

## Verified delivery

PR #5 merged as `95bbba0f6e7bc39ca03c1a5b28edb30a8de046d0` after Linux/macOS core and optional macOS integration passed. The task's collection, training and causal assessment deliverables are complete; its primary prediction gate was negative and remains so. TASK033/034 retain physical success acceptance. TASK037 separately tests horizon-matched balanced-intervention training. No historical result or threshold is replaced.
