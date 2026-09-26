---
id: TASK-064
aliases:
- TASK-064
title: Preregister and collect the look-prefix 112 px Apple corpus with the frozen DINOv2 encoder as candidate image encoder
slug: preregister-and-collect-the-look-prefix-112-px-apple-corpus-with-the-frozen-dino
status: in-progress
priority: 1
owner: ''
projects: []
customers: []
tags:
- apple-pnp
- data
- perception
sprint: ''
depends_on:
- "[[TASK-063]]"
due_date: ''
created: 2026-09-26
updated: 2026-09-26
---


# Preregister and collect the look-prefix 112 px Apple corpus with the frozen DINOv2 encoder as candidate image encoder

## Description

TASK-063 ended in **O-PT-POOLED**: the pinned frozen DINOv2 ViT-S/14 CLS token reads the
post-look apple (0.538 cm [0.487, 0.611], 183/190) beyond its seed-0 random init. Its row
pre-declared this task: **preregister the look-prefix 112 px corpus, with this frozen encoder as
the candidate image encoder.**

**Learned Apple->Plate is still 0 successes.** This is a data task: it builds and validates one
corpus from the **privileged scripted collector** (scripted successes are not learned results).
It trains nothing, fits no world model and preregisters no control formulation; the TASK-054
(CEM) and TASK-057 (BC) clauses hold.

- Corpus: 200 fresh wide-jitter resets (seeds 47000-47199), TASK-048's collector mechanics
  unchanged, every root episode starting with TASK-061's constant 8-command look (frame 8 = the
  post-look decision frame), `onboard_rgb` at 112 px only, three grasp-phase branches per root.
- Splits: whole-reset, 170 train / 20 val / 10 test (split seed 64). **The test split is never
  decoded** during collection QA; all-split facts are recorded at collection time.
- Acceptance checks: TASK-048's A1-A13 thresholds unchanged, plus L1 (the look on every root).
- Readability gate on the 190 train + val roots: TASK-063's P-cls against R-cls under the
  unchanged TASK-061 probe (bars, floor rule, p_arm <= 0.025, apple-hidden check).
- Rows: V, C-ACCEPT, C-READ-FAIL (abandonment clause), C-DATA-FAIL; a second V is INCONCLUSIVE.
- TASK-063 caveats carried forward: P-cls not detectably different from raw pixels; both floors
  met the bars alone; one encoder, one floor seed; cause (data, width, input size) unidentified.

Protocol: `docs/experiments/apple_look_corpus_v1.md`.
Manifest: `benchmarks/manifests/apple-look-corpus-v1.json`.
Design code: `src/embodied_jepa/look_corpus.py`.

## Acceptance Criteria
- [ ] PR 1 (protocol, manifest, `look_corpus.py`, tests, task card) merges on an independent
      reviewer's reported APPROVE and green CI, before any corpus seed is simulated.
- [ ] PR 2 (collector, readability gate, guard tests, script pins) merges on a reported APPROVE
      and green CI.
- [ ] The gated run starts only on the pre-run reviewer's reported verdict, from a clean tree.
- [ ] PR 3 (results document, results manifest with the dataset manifest hash, summarize script)
      states the outcome row, every acceptance quantity, the readability numbers with CIs against
      floor and baselines, every negative result and a recommended (not chosen) next task; a
      reviewer checks every restated number against `collection_report.json` by script. The data
      is never committed.
- [ ] The test split is never decoded; cohorts C and D are untouched; `exemption_spent` stays
      false.

## Notes
- Void rule: an early stop is V (with a void_reason). One from-scratch repeat into
  `data/apple-look-v1-run2` with the same seeds, caps and device; a second void closes the task
  as INCONCLUSIVE. Owner rulings are recorded with a UTC timestamp before the run they affect.
- Pilot (disclosed, not evidence): seeds 47900-47931, `outputs/task064-scratch/pilot-a/`,
  readability in smoke mode (noise targets). 21/32 root successes, look spread 0.0; thresholds
  kept at TASK-048's values.
- Worktree: `data`, `outputs`, `checkpoints`, `third_party` and `.venv` are ignored
  convenience symlinks to the main checkout; nothing under them is overwritten.

%% mc-links: [[TASK-063]] %%
