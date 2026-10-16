---
id: TASK-099
aliases:
- TASK-099
title: 'SO-101 S3: privileged scripted play collector and the so101-play-v1 corpus, measured before any training'
slug: so101-play-corpus
status: todo
priority: 2
owner: ''
projects: []
customers: []
tags:
- so101
- data
- simulation
sprint: ''
depends_on:
- "[[TASK-098]]"
due_date: ''
created: 2026-10-16
updated: 2026-10-16
---


# SO-101 S3: privileged scripted play collector and the so101-play-v1 corpus, measured before any training

## Description

Phase S3 of [SO101_PLAN.md](../../../docs/SO101_PLAN.md) (DECISIONS 2026-10-16 (bp), R28.6–R28.7).
A privileged scripted collector, modelled on TASK-085's `play_corpus.py` and TASK-090's
`play_corpus_grasp.py` (neither edited), records reach, approach, close, lift, place and free-play
segments with injected perturbations (action noise, bursts, random detours, failed grasps and
regrasps) on the S1 scene, through `joint_delta_so101_v0` (the stored action is the executed one),
with both cameras, into the schema-sized `DatasetStore` (`schema_data.py`, S0), with
episode-grouped train/val/test splits and a corpus manifest with hashes and the code revision.
Development data only: the collector reads simulator truth, its grasps and placements are never
learned results, and nothing here is gated.

## Acceptance Criteria

- [x] Collector, collection and measurement scripts in new files; every G1 hash-pinned file
  byte-identical; the core-import check passes; tests on a tiny CPU corpus.
- [x] A development pilot is collected and timed.
- [x] Corpus measures before any training (the plan's S3 check): action-range coverage per joint,
  gripper open/close balance, lift and place window counts, camera frame checks, split leakage
  check (plus the successful-grasp fraction and storage per episode).
- [x] If the pilot's projection fits the time and disk budget, `so101-play-v1` is collected under
  `data/` through `scripts/gpu_run.sh --wait --board` and measured.
- [ ] Results in `docs/experiments/so101_play_corpus_v1.md`; ruff and pytest green; PR reviewed
  independently.

## Notes

- 2026-10-16, created for S3 (branch `feat/so101-play-corpus`).
%% mc-links: [[TASK-098]] %%
- 2026-10-16, in review on branch `feat/so101-play-corpus`: `so101_play_corpus.py`
  (`so101_play_policy_v0`), `collect_so101_play.py`, `measure_so101_play.py`, tests on a tiny CPU
  corpus, `benchmarks/manifests/so101-play-v1.json` and
  `docs/experiments/so101_play_corpus_v1.md`. Pilot 64 episodes (66 s, 2 workers); corpus
  `so101-play-v1` 3 000 episodes (866 s, 12 workers, 26.5 GiB, collected at `bf12d37`), measured
  before training. Development data from a privileged scripted collector; gates nothing. Every
  G1 hash-pinned file is byte-identical.
