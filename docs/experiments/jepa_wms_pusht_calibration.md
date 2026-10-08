# TASK-084 — Phase 0 calibration: Meta's released JEPA-WM on Push-T, with its planner and ours; LeWM action-sensitivity baseline

**Status: FROZEN when this document's PR merges** (R22.1–R22.9, decided by Claude under owner
delegation). Nothing in §4–§5 may change after the merge without its own reviewed ruling. This is
Phase 0 of the proposal [JEPA_ZERO_SHOT_PLAN.md](../JEPA_ZERO_SHOT_PLAN.md) ("Close and
calibrate"); the plate line was already closed by TASK-083 (#177) and is not touched here.

**This is not an Apple→Plate experiment.** Its outcome does not change R7's canonical status
sentence (DECISIONS 2026-10-02, R7, as last updated by R21.20), whatever row it ends in.

## 1. Questions

- **A (gated).** On this machine (Linux PC, RTX 5080), does Meta's released JEPA-WM checkpoint for
  Push-T, run through Meta's own planning evaluation, reproduce the published Push-T success rate
  within 10 percentage points; and does this repository's planner, `embodied_jepa.planning.CEMPlanner`
  ("ours"), reach within 10 points of Meta's planner with the same model, cost, episodes and
  execution? The plan's gate: "our planner within about 10 points of the released result on one
  task, else fix it first".
- **B (reported only).** How action-sensitive are our existing LeWM checkpoints (TASK-077's W,
  model seeds 66800–66802, the models TASK-080, TASK-081 and TASK-083 ran in closed loop) at short
  horizons, and how well do they rank the executed command sequence among alternatives? A baseline
  for Phase 2; it gates nothing.

## 2. What is reproduced (verified citation)

- **Paper.** B. Terver, T.-Y. Yang, J. Ponce, A. Bardes, Y. LeCun, *What Drives Success in
  Physical Planning with Joint-Embedding Predictive World Models?*, arXiv
  [2512.24497](https://arxiv.org/abs/2512.24497). The number is identical in v1 (2025-12-30,
  Table 1 and Table S5.1) and v4 (2026-09-02, Table 2 and Table 11); both PDFs were read on
  2026-10-09.
- **Number.** JEPA-WM ("Ours"), Push-T, planner CEM with the L2 objective: **70.2 % (2.8)**.
  DINO-WM on the same row: 66.0 (4.7).
- **What the number is.** Per the paper (v1 §5.1 "Statistical significance"; v4 §F): each model
  is trained with 3 seeds; at each epoch, e = 96 episodes are evaluated; success is averaged over
  the last n = 10 epochs and the three seeds; the parenthesis is the standard deviation across the
  last epochs' success rates. So 70.2 % is an average over 30 checkpoint evaluations, not the
  success rate of one released checkpoint.
- **Released checkpoint.** `jepa_wm_pusht.pth.tar` from Hugging Face `facebook/jepa-wms`
  (CC BY-NC 4.0), sha256 `9beca3ea…aaaa0eb` (equal to the Hub's LFS oid), 211 639 615 bytes. It
  loads as **epoch 50** of one training run (the loader's log); which training seed it is, the
  repository does not say.
- **Planning setting.** The repository's own Push-T evaluation config,
  `configs/evals/simu_env_planning/pt/jepa-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml`
  at upstream revision `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0` (upstream `HEAD` on 2026-10-09,
  the revision TASK-024 pinned): CEM, N = 300 samples, H = 6, K = 10 elites, J = 30 iterations,
  m = 6 actions stepped, frameskip 5, objective L2 on the last predicted step with the
  proprioception term weighted α = 0.1, goal source `dset` (initial and goal state from a
  validation-split trajectory segment whose expert reaches the goal in 6 planning steps), so an
  episode is **one plan of 6 × 5 = 30 environment steps**, executed open loop. These match the
  paper's Table S4.1 / Table 10 row for Push-T.
- **Success** (upstream `PushTWrapper.eval_state`, unchanged): after the 30 steps, the agent and
  T-block position difference to the goal state is < 20 (pixels) and the block angle difference
  < π/9.

## 3. Set-up (all of it in git-ignored `third_party/jepa-wms-runtime/`)

| Item | Value |
| --- | --- |
| Upstream code | a local clone of `third_party/jepa-wms` at `13cf1d9…`, so the pinned TASK-024 source stays untouched (upstream's `setup_macros.py` writes `macros.py` into its tree) |
| Runtime | separate Python 3.10.21 venv (upstream requires `>=3.10,<3.11`); torch 2.7.0+cu128 and torchvision 0.22.0+cu128 (upstream pins torchvision 0.22.0; cu128 wheels include sm_120 for the RTX 5080); the rest of upstream's dependency list except the PointMaze, Metaworld and RoboCasa stacks (d4rl, mujoco-py, metaworld, robosuite, robocasa), resolved on 2026-10-09: e.g. tensordict 0.14.3, torchrl 0.14.0, timm, pymunk 6.8.0, gym 0.23.1. `uv pip freeze`: 170 lines, sha256 `3c392117…3c1e` (kept with the evidence) |
| This repository on that path | `PYTHONPATH=<checkout>/src`; `embodied_jepa.planning` (with `contracts`, `constraints`) imports on Python 3.10 unchanged; nothing of upstream enters the core package |
| Encoder | frozen DINOv2 ViT-S/14 as upstream loads it from torch hub: weights `dinov2_vits14_pretrain.pth`, sha256 `b938bf1b…0cd9`, **the same file this repository already pins** (DEPENDENCIES, TASK-063); hub code tree (`facebookresearch/dinov2` `main` at download time) sha256 `5c0d48ca…abbc9` over its sorted `*.py` hashes; the runner refuses other values |
| Data | `pusht_noise` (DINO-WM's Push-T dataset). The `facebook/jepa-wms` Hub dataset is gated and this account has no access, so it is taken from DINO-WM's own OSF release (`osf.io/bmw48`, `pusht_noise.zip`, 2 785 304 515 bytes, the size the Hub lists), which upstream's README says it re-hosts unmodified; sha256 `442f5dee…2da08`, as OSF publishes it; the runner refuses another zip |
| Device | the RTX 5080 (16 GB), fp32 as upstream's evaluation (no autocast, TF32 off) |

### 3.1 Deviations from upstream's evaluation (both arms equally)

1. **One process instead of 8 GPUs.** Upstream's config runs 8 ranks with rank-shifted seeds; here
   world size 1. Episode sampling is upstream's (its `local_generator`, seeded by `meta.seed`), but
   the episodes are not the ones Meta evaluated; neither set is published.
2. **Chunked roll-outs.** One 300-sample roll-out needs more than the RTX 5080's 16 GB (it ran
   out of memory). The runner splits the sample axis of upstream's `EncPredWM.unroll` into chunks
   of 100 and concatenates; every sample's roll-out is independent of the others, so only memory
   and floating-point reduction order change.
3. **No decoder heads, no decoding, no optional plots.** The config's state head is not released
   and the image head is a 3.6 GB download; planning never reads them (`decode_each_iteration`
   only feeds visualisation).
4. **Episodes split over four evaluation seeds** (§4.2) so that no GPU job holds the shared lock
   for more than about an hour.
5. Newer `tensordict`/`torchrl` than upstream's minimums (§3); torch 2.7.0 rather than whatever
   Meta ran.

## 4. Part A — protocol (gated)

### 4.1 Arms

- **upstream.** Upstream's `CEMPlanner`, unchanged, as the config builds it.
- **ours.** `embodied_jepa.planning.CEMPlanner`, unchanged, behind a bridge
  (`scripts/run_jepa_wms_calibration.py`, `make_bridge`) that gives it upstream's planner
  interface. Only the optimiser differs; the model, encoder, goal encoding, cost, episode sampling,
  action execution and success check are upstream's.
  - `CEMConfig(horizon = min(6, steps_left) = 6, samples = 300, iterations = 30, elites = 10,
    seed = 84 000 000 + 1000·meta_seed + plan index)`, every other field its default
    (`minimum_std` 0.05, `action_penalty` 0, bounds ±1 on all 14 dimensions, no projection).
  - The planner's cost is the model contract it already calls: `predict` rolls the 300 candidates
    out with upstream's `unroll`, `distance` returns upstream's L2 objective (α = 0.1) per predicted
    step, and the planner reads the last step, as upstream's objective does.
  - **Action mapping** (declared here, before any gated episode): the planner's normalised
    actions are 14-D in [−1, 1]; the model's Push-T action is 10-D (2-D relative motion × frameskip
    5) in upstream's z-scored units. The bridge uses the first 10 dimensions (the other 4 are
    sampled and ignored) and maps dimension d to `b_d · a_d`, with `b` = the per-raw-dimension
    99.5th percentile of |z| over the train split's relative actions, z-scored with upstream's
    fixed constants after upstream's division by the environment's action scale (100):
    **b = (3.703, 3.301)**, tiled five times. This is the project's convention (the normalised
    range is the action's operating range; here the data's, since the relative action has no
    physical bound) and is part of "our planner" as tested.
  - Known differences that remain, by design (they are what is being tested): ours returns the
    best candidate seen, not the final elite mean; its initial standard deviation is 1 in
    normalised units (about 3.3–3.7 z-units, against upstream's 1 z-unit); it clips candidates to
    the bounds; it keeps a 0.05 floor on the standard deviation; it draws from NumPy, not the GPU
    generator.

### 4.2 Episodes and seeds

- `meta.seed` ∈ {1, 2, 3, 4}, 24 episodes each: **96 episodes per arm**, the paper's per-evaluation
  count. Both arms run the same `meta.seed` values, so they plan from the same 96 (initial, goal)
  pairs; the runner records both states per episode and the decision voids the run if any pair
  differs (§4.5).
- Debug seeds (plumbing only, never counted): 9901–9999.
- Bootstrap salt for §4.4: 8602. Part B's candidate salt: 8601 (both new; 8401–8412 and 8501–8512
  belong to TASK-082 and TASK-083).

### 4.3 Order and budget

Plumbing smoke at the merged revision (both arms, debug seed 9902, 2 episodes), then the eight
gated jobs, one GPU job each through `scripts/gpu_run.sh --wait --min-free-gib 8 --board`, in the
order upstream-s1, ours-s1, upstream-s2, ours-s2, …, ours-s4. One plan costs about 155 s on the
RTX 5080 (plumbing), so a job is about 62 min and the run about 8.5 GPU-hours, the lock released
between jobs. A job that fails for a reason outside the protocol (a crash, a lock timeout) is
re-run once with the same seed into a new directory, recorded; a second failure voids the run.

### 4.4 Measurements

`scripts/summarize_task084.py` (NumPy, this repository's environment):

- success counts and rates per arm with Wilson 95 % intervals;
- the paired difference d = ours − upstream over the 96 episodes, with a paired bootstrap 95 %
  interval over episodes (20 000 resamples, salt 8602) and the exact two-sided McNemar p on the
  discordant pairs (reported);
- per-seed counts, seconds per episode, and ours − 70.2 (reported).

### 4.5 Rows (first match) and what each means

| Row | Condition | Meaning and consequence |
| --- | --- | --- |
| **P0-VOID** | an arm has fewer than 96 episodes after the one allowed re-run, a hash check failed, or any episode's (initial, goal) differs between the arms | no row; the cause is recorded; a new run needs fresh meta seeds (5–8) and its own ruling |
| **P0-PASS** | **G-REPRO**: upstream's rate is within ±10 points of 70.2 % (60.2 ≤ 100·p̂_up ≤ 80.2), **and G-PLAN**: 100·(p̂_ours − p̂_up) ≥ −10 | upstream's pipeline reproduces the published number within the window on this machine, and our planner is within 10 points of Meta's planner with the same model. Phase 0's planner gate is met; our planner may be used as is for Phase 3. Also reported: whether the paired interval's lower bound is above −10 ("clear") or only the point estimate is ("within on the point estimate only") |
| **P0-PLANNER-GAP** | G-REPRO passes, G-PLAN fails | the pipeline reproduces, our planner does not: fix it first (candidate causes are listed in §4.1), in its own reviewed PR, and re-test both arms on fresh meta seeds 5–8 under its own ruling; Phase 3 does not start with the unfixed planner |
| **P0-REPRO-FAIL** | G-REPRO fails (either direction) | Meta's own planner does not reproduce the number here within the window, so the calibration target is not established on this machine; G-PLAN is still reported (it says whether ours matches theirs here) but the gate is not met. Investigate the runtime and the §3.1 deviations before anything relies on it |

The window is about 2 standard errors at n = 96 (p ≈ 0.7: SE ≈ 4.7 points; the paired
difference's SE is smaller when the arms agree on most episodes). The published number averages
30 checkpoint evaluations of three training runs, the released file is one checkpoint, and the
episodes differ (§3.1), so a ±10-point window is the tolerance the plan named, not a test of
equality.

### 4.6 What part A does not show

Nothing about LeWM, G1 + Dex3, dexterous actions, MuJoCo or Apple→Plate. One task (Push-T, a 2-D
pusher), one released checkpoint, one planning setting (one plan per episode, open loop), 96
episodes per arm, simulation only. P0-PASS says our CEM implementation is not the bottleneck on a
task where a JEPA world model is known to plan; it does not say our world models are plannable.

## 5. Part B — action-sensitivity baseline (reported only)

- **Models.** TASK-077's W checkpoints for model seeds 66800, 66801 and 66802 (the files the
  decomposition record verified by their Stage T sha256), as they are: no training. W-66800 is
  flagged `last_two_triggered`.
- **Data.** The **val** split of `apple-c1m-v2` (250 roots; corpus `ad8974b2…43fb`, features
  verified against the featurise report). The gate split is never opened; the train split is not
  read. From frame 405 of each root, with its 60 executed commands.
- **Horizons.** h ∈ {1, 2, 4, 8, 16, 30, 60} steps.
- **Statistics** (`scripts/probe_task084_action_sensitivity.py`, CPU), per seed and h:
  - `wrong_over_true`, `zero_over_true`: TASK-077's G4 statistic (ratio of summed normalised MSE
    to the encoded frame at h; wrong = a derangement of the val roots, salt 8110; root bootstrap,
    salt 8106), here at short horizons;
  - `true_over_copy`: the same against copy-last (TASK-077's G2 form);
  - ranking: 16 candidate command sequences per root (its own and 15 other val roots', drawn once,
    salt 8601), each rolled out from the root's frame and scored by normalised squared distance of
    the predicted latent at h to the encoded frame at h; top-1 accuracy (chance 1/16) and mean
    normalised rank of the true sequence (0 best, chance 0.5; ties count against it), root
    bootstrap.
- **No gate, no row.** It is the plain-LeWM reference for Phase 2's comparison, which needs its
  own preregistration. The val roots are not out of sample for checkpoint selection (TASK-077
  selected on val), which the results must say.
- A smoke on the first 20 val roots ran before this document was frozen (§6); its numbers are not
  results.

## 6. Plumbing before the freeze (disclosed; not results)

All on debug seeds, at the branch's working revision (base `f7fef9c`), evidence in
`~/develop/emai/evidence/task084-plumbing/`:

| Run | Episodes | Successes | Note |
| --- | --- | --- | --- |
| upstream, seed 9901 | 2 | 2 | first two runs failed before any episode: no process group, then out of memory (→ §3.1 item 2) |
| ours, seed 9901, **wrong scale** | 2 | 0 | bug: the scale omitted upstream's division by the action scale 100, so b ≈ (369, 330) z-units; fixed before anything else ran |
| ours, seed 9901 | 2 | 2 | same (initial, goal) pairs as the upstream run |
| ours, seed 9903 | 1 | 0 | after adding the hash pins (§3) |
| part B smoke, 20 val roots | — | — | runs in 19 s; numbers not used |

About 155 s per plan in both arms. These episodes are not part of any count.

## 7. Evidence, guards and cost

- Runs: `~/develop/emai/evidence/task084-run/{upstream,ours}-s{1..4}/` (`episodes.jsonl`,
  `summary.json`, upstream's logs and CSV), the job logs, `decision.json` from §4.4, part B's
  `report.json` in `~/develop/emai/evidence/task084-probe/`, the runtime freeze, and a
  `SHA256SUMS` over all of it. On the SSD; nothing under `data/`, `checkpoints/` or `outputs/` is
  written or overwritten.
- The runner refuses an existing output directory, an upstream checkout at another revision, and
  any checkpoint, encoder, hub-code or dataset hash other than §3's.
- GPU only through `scripts/gpu_run.sh`; the GR00T server, its containers and other processes are
  not touched. Disk is kept above 10 GB free (the runtime is about 18 GB).
- Licences: upstream code, the released checkpoint (CC BY-NC 4.0) and the data stay in ignored
  directories and are not redistributed; DINOv2 is the Apache-2.0 file the repository already uses.

## 8. Rulings this document records (R22.1–R22.9, decided by Claude under owner delegation)

The rulings are in [DECISIONS.md](../DECISIONS.md), decision 2026-10-09 (s). In short: the task
(Push-T), the checkpoint, the published number and its citation, the arms, the action mapping, the
episodes and seeds, the gate and its rows, part B's design, and that R7 does not change.
