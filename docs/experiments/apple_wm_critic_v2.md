# Apple→Plate WM critic v2: a LeWM token critic picks P-3's aim under a mid-episode plate shift (TASK-073)

**Status: preregistration (Stage 0).** No TASK-073 seed outside the smoke range 53950–53999 has
been simulated. Only the smokes and the blind-baseline checks of §9 have run; §9 says exactly what
they read. As in M2, every `preflight`, `smoke` and stage run re-renders (frame only, nothing
scored) the post-look frames of TASK-072 run-1's perception seeds (51200–51583) and D2
(52000–52015) to refit run-1's readouts and reproduce its facts (G-repro); that is M2's accepted
precedent (`apple_first_policy_v2_m2.md` §2), not a use of those seeds.

**Status lines.**
- **Learned Apple→Plate on the frozen benchmark is still 0 successes** (the v1 MVP benchmark,
  0/150 per model). Nothing in this task changes that number.
- **No control line is primary.** This protocol tests one component, a world-model critic, inside
  a controller that stays P-3 (TASK-072 run-1's behaviour-cloning/DAgger policy). It does not
  make LeWM, sampling-based planning or behaviour cloning the project's control approach.
- **The plate shift is a simulation-only diagnostic condition on `apple-to-plate-v2`**, not a new
  benchmark (owner D1). A teleported plate is not a real-world disturbance.
- **An H-LeWM success would be "a learned policy trained on privileged expert labels plus a LeWM
  critic"** (owner D2): P-3 was trained on e9's demonstrations with a privileged DAgger labeller,
  and the critic's corpus is collected by the privileged scripted expert. It is not LeWM driving
  the robot alone.
- This is the first step that puts a LeWM world model into the control loop. Even the best row,
  HYB-PASS, would be evidence for "a world model as a critic measurably helps a learned policy
  under a disturbance", on one task, one camera (112 px onboard) and one policy.

Manifest: `benchmarks/manifests/apple-wm-critic-v2.json` (the frozen block, the stored cohort
values and digests, the pins). Design: `src/embodied_jepa/wm_critic_v2.py`. New modules:
`plate_shift.py`, `hybrid_selection.py`, `sim_selector.py`, `models/latent_critic.py`,
`wm_critic_v2_runtime.py` (simulation workers) and `wm_critic_v2_offline.py` (features, training,
statistics). Scripts: `scripts/run_wm_critic_v2.py` (preflight, smoke, K0, train, D3, gated),
`scripts/collect_apple_shift_v2.py`, `scripts/rank_wm_critic_v2.py` and
`scripts/summarize_wm_critic_v2.py`. Tests: `tests/test_wm_critic_v2.py`. **No existing file is
modified**: every file an existing implementation hash or M2 pin covers keeps its bytes (a test
compares the 36 shared pins with M2's).

---

## 0. Owner decisions (2026-09-29, answered in the design session, relayed by the orchestrator)

The design came from a judge-panel workflow (three candidate formulations, three lenses). The
winner, `policy-plus-wm` (6 / 8 / 6.5 on product, rigour and feasibility), is this protocol. The
owner answered its open decisions as follows; the answers are quoted as relayed.

| decision | the owner's answer | what it fixes here |
|---|---|---|
| **D7**, path | **"Critic first, planner fallback"** | TASK-073 is the hybrid critic. **TASK-074, the LeWM-only planner (`lewm-planner`), is pre-authorised as the fallback if TASK-073 ends S-NO-CONDITION, H-NO-HEADROOM or NO-HEADROOM.** TASK-074 needs its own preregistration; this protocol does not design it (§7.3) |
| **D1/D2**, condition | **"Accept, diagnostic only"** | the mid-episode plate shift is a simulation-only diagnostic condition, not a benchmark; H-LeWM successes are labelled "a learned policy trained on privileged expert labels plus a LeWM critic", and they do not change the v1 0/150 headline |
| **D3/D4/D6**, defaults | **"Use the defaults"** | the |d| grid 3–6 cm; shift steps 300 and 480; the four K0 bars; G1–G4 at +10/64 with p < 0.01; the O3 ρ bar of 0.5; the HYB-SCENE-BLIND row; DINOv2 featurisation on the GPU with a CPU anchor check |
| **D5**, PR #104 | moot | #104 merged as `70f1358` before this protocol. Every M2 number cited here was checked against `benchmarks/manifests/apple-first-policy-v2-m2-results.json` (§12) |

**Owner rulings on the pre-registration review (2026-09-29, relayed by the orchestrator; the
questions abbreviated, the answers verbatim).**

| question | the owner's answer | what it fixes here |
|---|---|---|
| "R-NO-HEADROOM … trigger the pre-authorized LeWM-planner fallback like the other no-headroom / no-condition rows?" | **"Yes, same as fallback rows (Recommended)"** | R-NO-HEADROOM is in `FALLBACK_ROWS` (D-17) |
| "Decision-latency gate G7 …" | **"Keep bar, failure = HYB-SLOW only (Recommended)"** | G7 stays at 0.25 s; a G7-only failure is HYB-SLOW, which does not fire the clause (D-18) |
| "G-repro voids under load" | **"Pin threading + quiet machine (Recommended)"** | the runners pin `MKL_DYNAMIC=FALSE`, `OMP_NUM_THREADS=6`, `MKL_NUM_THREADS=6`, `OPENBLAS_NUM_THREADS=16` before NumPy or torch loads (G-threads), and refuse to start on a busy machine (G-quiet, §13.2); a V before a stage's first cohort render may be repeated as-is and does not use up the one development repeat (§7.1) |
| "Post-smoke D-17/D-8 and D-19" | **"Accept both (Recommended)"** | D-8, D-17 and D-19 stay; they were set after smoke data (smoke seeds only; §9.3, §9.5), and §10 says so |
| "Harm rule" | **"Harm also closes the line (Recommended)"** | HYB-HARM is checked before HYB-SLOW, HYB-SCENE-BLIND and HYB-NO-GAIN, and fires the abandonment clause (§6.6, §7.2) |

## 1. The question and the claim

**Question.** On `apple-to-plate-v2`, with the plate moved a few cm after the grasp latches, does a
LeWM predictor over frozen DINOv2 4 × 4 patch tokens (TASK-066's latent, unchanged, run at
TASK-066's validated horizon h = 16 with the per-step 14-D action adapter) rank P-3's candidate
aims well enough that choosing the aim by it gives more counted successes than the identical
controller with the critic replaced by:

- an **action-blind** predictor (copy-last, or the equally trained no-action model N; with the
  tie rule either reduces exactly to **P-reread**);
- a **scene-blind** predictor (W started from another reset's latent, **H-shuf**);
- a **random choice** among the same 25 candidates (**H-rand**);

without harming undisturbed resets?

**The claim a pass would make.** "A world-model component measurably improves a learned policy
under a disturbance", with the labels of the status lines. **It would not say** that LeWM drives
the robot alone, anything about the v1 task or its 0/150, anything about a real-world disturbance,
another camera, or another policy.

## 2. Why this is not a rerun of an abandoned line

- **TASK-054 (CEM over the world-model cost).** There is no sampling optimiser over 14-D actions:
  the world model makes a 1-of-25 choice among proposals from a policy that succeeds on unshifted
  v2 resets (P-3 40/40 on cohort C, in an M2 run whose row was M2-FAIL, where the random-init
  floor R-3 reached 39/40), the
  critic role `DECISIONS.md` left open ("behaviour cloning with the world model as a critic").
  The latent is frozen external DINOv2 tokens, not the in-corpus encoder; the corpus and task are
  new (v2). TASK-054's primary criterion G2a is re-measured head-on as O2 (i), on TASK-054's own
  moving-window definition; if it fails the run stops before any closed loop.
- **TASK-057 §7 (behaviour cloning on `apple-wide-v1` at 112 px).** No policy is trained; P-3 is
  carried frozen and `apple-wide-v1` is not read. The camera is still the 112 px onboard one, and
  this protocol says so.
- **TASK-062 (in-corpus encoder training).** No encoder is trained.
- **TASK-065 (predictor on pooled CLS latents).** The latent is patch tokens, TASK-066's line,
  which passed. `apple-look-v1` is not read.

## 3. The design

### 3.1 Carried unchanged

The v2 scene; e9 (`RestingPlaceExpert(release_pitch_rad=0.45, release_dx=0.015)`), its schedule
`EXPERT_BUDGETS = (130, 80, 45, 150, 100, 50, 30, 50, 30, 60)` (transfer starts at step 405,
steady ends at step 585); the counted success of T71-R1/R2 (at rest after the 60-step settle
**and** the latched grasp and place stages before it); the attempt of 740 commands plus the
settle; the 132-D policy input `input_vector(estimates, step, state, palm9)`; P-3 itself
(`checkpoints/task072-first-policy-v2-linux/run-1/P-3.pt`, sha256 `7988162d…60be8`, verified);
its post-look readout, refitted exactly as TASK-072 run-1 fitted it and required to reproduce
run-1's eight recorded facts (M2's G-repro, reused unchanged) before any TASK-073 seed is
rendered; TASK-066's LeWM token predictor configuration (`token_dynamics.MODEL_CONFIG`).

### 3.2 The condition: a mid-episode plate shift (owner D1)

A harness hook (`plate_shift.PlateShift`) moves the static plate once, just before the observation
of post-look step s ∈ {300, 480} is taken: a `body_pos` write plus `mj_forward`, logged, outside
every controller's `act()`. The observation of step s is the first to show the moved plate; the
latched scorer and the at-rest record read the plate's live position, so scoring after the shift
uses the post-shift plate (tested). A move that leaves the plate touching anything other than the
table is refused (a void).

**The direction rule (a correction to the outline).** The outline asked for shifts "over the arc
where `release_pose` does not clamp". Checked in code, e9's release pose **at the reset** is pulled
back to its reach sphere (x ≈ 0.48 m) on every v2 reset (1000/1000 in 53000–53999; plate x
0.47–0.51 m plus release_dx 1.5 cm). Shift directions that leave the *moved* plate's release
unclamped do exist (towards the robot; for 3683 of the 4000 reset × |d| pairs in 53000–53999),
but on those the release point moves relative to the plate, which changes the plate-relative
release geometry that e9 was validated with, and a shift along +x moves the plate but not the
release. The stored rule
instead keeps e9's release geometry **relative to the plate**: a direction θ (whole degrees) is
eligible for a reset and a |d| when e9's release point relative to the shifted plate is within
0.5 cm of its value at the reset, the shifted plate stays on the tabletop, and it keeps the reset
rule's plate-apple clearance from the apple's reset position. In practice the eligible directions
are mostly near ±y, but not exactly sideways: the eligible sets reach 111–196 of 360 directions
on some resets, and 4 of the 64 stored S shifts at 4 cm point at 132–135° or 219–231°. One uniform draw per reset (`default_rng(SeedSequence([7303, seed]))`)
picks from the eligible set at every |d|. Every shift vector of every cohort and every |d| is
stored in the manifest.

**Why e9 is the exact post-shift oracle.** e9's pick phases (orient, descend, close, lift) are
built from the apple only; its transfer, lower, steady, open, clear and retreat phases from the
plate. So e9 built at reset from `perturbed_truth(truth, 0, d)` issues exactly the unshifted
commands until step 405 and then transfers to the moved plate (tested: `phases[:4]` targets,
grasps, budgets and rotation are invariant under a plate shift). It is **B-oracle-shift** and the
corpus collector.

### 3.3 The controller

P-3 is unchanged: `LearnedController` with its mutable `estimates` (apple xy, plate xy), the
post-look estimates until the first decision. **Decision steps**: every 16 commands from 304 to
576 (18 decisions). At each decision:

1. Encode the current frame on the CPU at batch size 1 with frozen DINOv2 (the P readout's
   path): the full 256 × 384 tokens for R-mid and their 4 × 4 pool for the critic.
2. **R-mid** re-reads the plate: a linear ridge readout on the full tokens, fitted on the
   `apple-shift-v2` train roots' decision-step frames. Its reading is the **incumbent**.
3. Form K = 25 aims: the incumbent plus the 5 × 5 grid at 1 cm (±2 cm); index 12 is the
   incumbent.
4. For each aim, generate a 16-command chunk by **re-querying P-3 at every step against a
   kinematic stand-in** (`hybrid_selection.KinematicStandIn`): the embodiment's own per-side
   acceptance calculation (workspace clip, IK, joint- and grasp-rate limits, with
   `project_candidates`' backtracking) on a private copy of the kinematics, joints assumed to reach
   the accepted targets within the step. It is object-free and reads no simulator state. The
   transport's previous joint targets are reconstructed from what the controller knows: measured
   arm joints, and each hand on the synergy of its own last executed grasp command (a closed hand
   is blocked by the apple and sits off the synergy; using the measured hand joints made every
   chunk infeasible in the first mechanics check).
5. Roll all 25 chunks through W in one batch (`predict_features`, h = 16).
6. Score each: `J_k = ‖R_off(ẑ_{t+16}) − o*(t+16)‖` in cm. `R_off` is a linear ridge readout from
   the pooled latent to the apple-minus-plate offset; `o*` is §3.5's frozen target.
7. Execute the argmin for 16 closed-loop commands (P-3 with that aim as its plate estimate). The
   incumbent wins every tie within 1e-6 cm, and an infeasible chunk costs +∞.

**Architecture.** The critic (`models/latent_critic.LatentCritic`) returns costs only; the
selector never sees a latent. There are no backend branches (the world model is anything with
`predict_features`, so the LeWM/native swap stays one key). `VisualModel.defaults` is unchanged,
and so is every file behind an implementation hash: W is `frozen_token_model("leworldmodel")`
with TASK-066's configuration, and its action encoder is the pinned 14-D `Embedder`, stepped 16
times.

### 3.4 Arms

| arm | rung | role |
|---|---|---|
| **H-LeWM** (the W seed with the lowest val criterion, fixed before any closed loop) | L1 | primary |
| P-stale (P-3 unchanged, post-look estimates throughout) | L1 | blind: no re-perception, no world model |
| P-reread (R-mid every 16 steps) | L1 | the exact action-blind twin (equals H-copy and H-N by construction; tested) |
| H-N (the no-action model, run closed loop) | L1 | action-blind world model, run rather than assumed |
| H-shuf (W started from P-reread's latent of reset (i + 1) mod n at the same step) | L1 | scene-blind |
| H-rand (a seeded uniform pick among the same 25 candidates) | L1 | "any aim change" control |
| H-LeWM-s1, H-LeWM-s2 (the other two W seeds) | L1 | secondary: per-seed sign agreement |
| P-truth (P-3 given the true post-shift plate at the shift step) | L4 | perception ceiling |
| H-sim (the same 25 candidates, chosen by privileged cloned-state rollouts) | L4 | selection ceiling and headroom |
| B-oracle-shift, B-hold, B-random | L4 | harness checks |

H-sim (`sim_selector.SimSelector`) saves the live `MjData` and the transport and embodiment
fields, runs each aim for 16 closed-loop commands through the exact attempt path (with the image
renderer replaced by a blank frame, since P-3 reads no image), reads the true offset, and restores
everything bit for bit (tested). The latched scorer and the at-rest record are never called in a
branch. In K0 its candidates are centred on the true post-shift plate (R-mid does not exist yet);
from D3 on, on R-mid's incumbent. The undisturbed cohort U runs P-stale, P-reread and H-LeWM.

### 3.5 The frozen target o*

`o*(t)` is the per-step median apple-minus-plate offset (cm) over the **112 counted-success train
roots of `apple-look-v2-linux` run-1** (e9, unshifted; manifest `67c342f6…4f54`; run-1's B-replay
library), at the 18 steps t + 16. It is computed now and frozen in the module and the manifest
(for example `o*(480) = (0.35, −0.15)`, `o*(544) = (−1.58, −0.14)`). The outline took o* from the
new corpus; freezing it here removes K0's dependency on a corpus that does not yet exist (H-sim
needs it) and one researcher degree of freedom. The offset in the steady phase is the plate-relative
release geometry, which the direction rule keeps unchanged.

## 4. Seeds and cohorts

The repository search of 2026-09-29 found no seed use in 53000–53999 (src, scripts, docs,
benchmarks, tests, configs, `.mc`), and no use of 7300–7313 as a seed constant. Every range lies
in 53000–53999, is disjoint from every other and from every forbidden range (TASK-067's list,
51000–52199 including D2, and cohort C, which M2 has now simulated). The runner's G-seeds refuses
anything else as a TASK-073 attempt seed, including cohort C, D2 and string seeds. The frame-only
re-render of run-1's perception and D2 seeds for G-repro (status lines) is the one exception, and
it goes through M2's own whitelist (`check_rerender_seeds`).

| seeds | cohort | use |
|---|---|---|
| 53000–53031 | K0 | condition calibration (development) |
| 53040–53071 | R | offline ranking and regret (O3, O4); never fitted on |
| 53100–53115 | D3 | development closed loop, stop rule only |
| 53400–53699 | `apple-shift-v2` corpus | 240 / 30 / 30 by reset; the test split is never decoded |
| 53800–53863 | S | gated, 64 shift resets |
| 53900–53931 | U | gated no-harm, 32 undisturbed resets |
| 53950–53999 | smoke | mechanics only; §9 says what was read |

Model seeds 7300–7302 (W and N, one model each per seed). RNG seeds: 7303 shift direction, 7304
mis-aim, 7305 corpus split, 7306 corpus plan salt, 7307 H-rand, 7308 B-random, 7309 bootstrap,
7310 readout folds, 7311 W/N sampler, 7312 wrong-action shuffle, 7313 comparative-rank bootstrap.

**Stored values.** Every cohort's resets (TASK-047's wide-jitter distribution, pinned equal to
`scripts/evaluate_apple.wide_reset`) and, for K0, R, D3 and S, the shift vector at every |d| of
the grid are stored in the manifest with a sha256 digest (`cohorts.<role>.sha256` over the
sorted-key compact JSON). Runners read the stored values and check the digest and the whitelist
(G-cohort); they never recompute them. Exact regeneration of the stored floats is Linux-only: on
macOS a few `object_xy` values differ in the last ulp (seeds 53007, 53026, 53028, 53030), so the
test compares the floats within 1e-12, the chosen shift directions exactly (whole degrees) and the
digest exactly against the stored JSON.

## 5. Stages and stop rules

Nothing from K0, R, D3, the corpus, S or U is simulated before this PR is merged **and** a
pre-run reviewer's GO is reported, and the orchestrator is told before each stage starts.

1. **Stage 0 (this PR).** Protocol, frozen modules, tests, stored cohort values, smokes (§9).
2. **Stage 1, K0** (`run_wm_critic_v2.py k0`; simulator only, no world model; about 1 h CPU;
   ≤ 2 h with the retry). For s = 300 and each |d| ∈ {3, 4, 5, 6} cm in order, the arms run on
   the 32 K0 resets in the order B-oracle-shift, P-stale, P-truth, H-sim, and a |d| ends at its
   first failed bar: (1) B-oracle-shift ≥ 28/32; (2) P-stale ≤ 8/32; (3) P-truth ≥ 20/32;
   (4) H-sim − P-truth ≥ +4/32. The smallest |d| passing all four is chosen; only if none passes
   at step 300, the same at step 480 (the one retry). **Stop: S-NO-CONDITION** ("BC plus perception
   leaves no measurable room for a world-model critic under this disturbance on v2"); the fallback
   of §7.3 applies. No third condition without a new preregistration. O5's chunk errors are
   recorded on the chosen cell's P-truth attempts.
3. **Stage 2, corpus** (`collect_apple_shift_v2.py`). 300 e9 roots from the frozen plan: 75 %
   shifted at K0's choice; 50 % mis-aimed by an extra release offset drawn in a 3 cm disc (these
   are never used for behaviour cloning; nothing here is); noise levels 0–3 by TASK-048's
   perturber. The episode schema is `apple-look-v2`'s plus the plate trajectory (`plate`, a
   declared new corpus schema). The report counts contacts, at-rest and off-plate roots per split,
   shift and mis-aim. The train and val roots of `apple-look-v2-linux` run-1 (190) join W's
   training set.
4. **Stage 3, models** (`run_wm_critic_v2.py train`; train split only).
   - Features: frames 240–660 of every root (every decision window lies inside), DINOv2 on CUDA
     (owner D6), with the anchor check G-anchor: 256 frames through the closed loop's CPU path
     must agree with the CUDA pooled features within 1e-3 (max absolute difference).
   - Readouts R_off and R-mid (§3.3), λ from {1e-3 … 1e3} by 5-fold inner CV grouped by root.
   - **Blind baselines are computed and written before any W number exists**: the persistence
     critic, the clock prior o*(t + 16) and the encoded readout on O2's windows
     (`baselines.json`); in the ranking stage the copy-last, N and prior-distance rankers are
     written (`blind.json`) before any W ranker is scored. **Limitation:** both happen inside one
     process with no external checkpoint, so this is a record of order, not a pre-commitment
     anyone outside the process can verify; the numbers are fully determined by frozen code and
     data, so they cannot be steered by the W numbers either way.
   - Budget: calibration runs W-7300 and N-7300 for 60 000 updates with the val criterion every
     1 000; `u_sat` by `token_dynamics.saturation_update`; U = clamp(5 000 · ⌈2 · max u_sat / 5 000⌉,
     10 000, 60 000), escalating (no freeze) above the cap. W and N × 3 seeds at U, selection at
     20 points. If any model selects one of its last two points, U is raised once to min(2U,
     60 000) and all six retrained; if that still happens at the cap, escalate.
   - The primary W seed is the lowest val criterion at its selected checkpoint.
5. **Stage 4, offline gates** (O1, O2 in the train stage; O3, O4 in `rank_wm_critic_v2.py` on
   cohort R; O5 from K0; the decision by `summarize_wm_critic_v2.py offline`). First match:
   any W seed fails O1 → **WMC-NO-DYNAMICS**; N passes O2 (i) on any seed → **WMC-O2-VOID**
   (escalate); O2 fails → **WMC-G2A** (TASK-054's criterion still fails here); R-mid's incumbent
   already near-best on cohort R (O0 fails) → **R-NO-HEADROOM** (the offline form of
   NO-HEADROOM; fallback, no abandonment); a void O3 → **WMC-O3-VOID** (escalate); O3 or O4
   fails → **WMC-NO-RANK**; O5 fails → **WMC-PROPOSAL** (proposal-harness escalation); else
   **OFFLINE-PASS**.
6. **Stage 5, D3** (16 development resets; non-gating; `run_wm_critic_v2.py d3`, which requires
   the recorded OFFLINE-PASS). Arms P-stale, P-reread, H-LeWM, H-N, H-shuf, H-rand, H-sim,
   B-oracle-shift. **NO-HEADROOM** if H-sim − P-reread < +2/16 (fallback applies);
   **WMC-DEV-STOP** if H-LeWM − max(P-reread, H-shuf) < +2/16. Nothing on D3 is refitted.
7. **Stage 6, gated** — only on a separate owner authorisation record, following the TASK-072 M2
   pattern (the runner refuses the gated stage until that record is pinned in the manifest's
   `gated_authorization`). Cohorts S (64) and U (32), each arm once per reset, paired; a clean
   worktree of the merged revision; `decide_gated` produces the row and the summarizer recomputes
   it and checks the evidence hashes.
8. **Stage 7, results PR.** Every arm reported; privileged ceilings labelled as not learned; an
   independent reviewer checks every restated number.

## 6. Gates

Intervals are session-(reset-)clustered bootstrap percentile intervals, 10 000 resamples, 95 %.
Every offline gate must pass on all three W seeds.

### 6.1 O1, dynamics (TASK-066's set, on the val roots)

At h = 8 and h = 16, on two window sets: **overall** (every val window inside the band, stride 4)
and **post-shift** (shifted val roots, windows starting at or after the shift step):
rank ratio ≥ 0.16, std ratio ≥ 0.39, collapsed fraction ≤ 0.05; the W − N comparative-rank lower
bound > 0 (the 256-direction train basis), with the rank-truncated W predictors at k = 1, 2, 4
failing the same bars; W / copy-last upper bound ≤ 0.8; W / N upper bound < 1.0; shuffled-action /
W and zero-action / W lower bounds ≥ 1.10.

### 6.2 O2, critic readability and the G2a re-test

Windows: every val root, starting at the decision steps t ≥ max(416, shift step) **whose true
offset moves by at least 1 cm over the 16 commands** (TASK-054's moving cohort,
`world_model_v2.MOVING_THRESHOLD_M`; a correction, §10 D-6). All must hold: R_off on encoded
latents at t + 16 has median ≤ 1.0 cm; R_off on W's h = 16 predictions has median ≤ 1.5 cm;
(i) the upper bound of the ratio of medians rollout error / persistence (R_off of the start latent
used as the answer at t + 16) ≤ 0.8 (TASK-054's G2a bar); (iii) the upper bound of rollout error /
clock-prior error (o*(t + 16) used as the answer) ≤ 0.8 (added, §10 D-7). (ii) The same ratio (i)
for N must fail (upper bound ≥ 0.8), or the gate is void.

### 6.3 O3, ranking (cohort R)

Each R reset runs P-reread with the shift; at the five R points (416, 448, 480, 512, 544 for a
shift at 300; 480, 496, 512, 528, 544 for 480) the 25 real candidates around R-mid's incumbent are
scored against their true 16-command outcomes (privileged, scoring only). Per group, Spearman's ρ
between a ranker's costs and the true costs over the feasible candidates. The lower bound of the
median ρ_W ≥ 0.5, and the lower bound of the median of (ρ_W − the best blind ranker's ρ in the
group) ≥ 0.3. Blind rankers: copy-last, N, L-shuf (each W seed started from the latent of reset
(i + 1) mod n at the same point) and prior-distance (the aim's distance from the incumbent).
**Pinned rule:** a median ρ ≥ 0.5 of copy-last, N or L-shuf voids the gate. Prior-distance enters
through the margin only (§10 D-8).

**O0, headroom for the ranking gates (added, D-17).** The median over cohort R's groups of the
incumbent's true regret (its true cost minus the group's best) must be at least 0.5 cm, or the
row is R-NO-HEADROOM. Without it, an accurate R-mid makes O3's margin and O4's paired bar
impossible for any critic, and the abandonment clause would fire on what is a headroom finding.
0.5 cm is about 40 % of the median spread of true costs across the 25 aims measured on smoke
seeds (1.27 cm, §9.5), and about the margin between e9's final distance on cohort C in M2
(B-oracle q10/q50/q90 3.01/3.46/3.83 cm, M2 results manifest) and
the 4 cm at-rest radius.

### 6.4 O4, regret; 6.5 O5, the proposal harness

**O4:** the median true-outcome regret of W's top-1 choice (true cost of the chosen aim minus the
group's best) ≤ 1.0 cm, and the upper bound of the median paired difference regret_W −
regret_incumbent < 0. **O5:** on K0's chosen cell's P-truth attempts, at every decision from the
shift on, the stand-in's chunk for the executed aim against the commands P-3 actually executed:
mean over the 16 commands of ‖stand-in − executed‖ (free 7-vectors), divided by the attempt's
median command norm; the median over decisions ≤ 0.25. Its distribution is reported.

### 6.6 The gated rows on S and U (first match)

| row | condition |
|---|---|
| **VOID** | B-oracle-shift < 58/64; B-hold or B-random has a grasp; any L1 arm made a privileged read; a determinism re-run on 4 S resets differs; a CUDA allocation fails (batches are never shrunk to fit) |
| **S-VOID-CONDITION** | P-stale > 24/64 |
| **H-NO-HEADROOM** | H-sim − P-reread < +12/64 (the world model is neither credited nor blamed) |
| **HYB-PASS** | all of G1–G7 |
| **HYB-HARM** | G6 fails (checked before the rows below; fires the clause; owner ruling) |
| **HYB-SLOW** | G1–G6 pass, G7 fails (added, D-18; the clause does not fire; owner ruling) |
| **HYB-SCENE-BLIND** | G1 and G3 pass, G2 fails |
| **HYB-NO-GAIN** | otherwise |

G1–G4: H-LeWM minus P-reread / H-shuf / H-N / H-rand ≥ +10 of 64, each with one-sided exact
McNemar p < 0.01. G5: H-LeWM − P-stale ≥ +16. G6 (U): H-LeWM ≥ P-stale(U) − 2 of 32. G7: the
median decision latency ≤ 250 ms with the GPU's resident service running. **G7's scope, frozen
(`GATED`):** H-LeWM only (the primary W seed); every decision record of its 64 S attempts; each
record's wall time of the decision inside `act()` (encoding, R-mid, the stand-in chunks, the W
rollout and the choice), in the gated run's H workers (4 workers × 4 torch threads, all attempts
concurrent); the median over those records. **The determinism re-run, frozen:** after the gated
arms, H-LeWM and P-reread run again on the first four S seeds (53800–53803) in the same run. A
non-image arm "differs" if any attempt's executed command array is not bitwise equal, or its
counted success, termination reason or chosen-aim sequence differs from the first run of that arm
and seed. **Image-reading arms (owner ruling of 2026-09-29, §15.5):** they "differ" when
`rerun_matches` fails — the fields and tolerances of `RERUN_IMAGE_ARMS`; every difference is
logged. Both re-run arms read images.
Secondary, not
changing the row: H-LeWM-s1 and -s2 show the same sign against P-reread, H-shuf and H-N.

### 6.7 Blind baselines against every bar (checked before freezing)

| gate | blind or prior-only baseline | where it sits |
|---|---|---|
| G1, G3 | P-reread (= H-copy = H-N by the tie rule) | difference 0 < +10 by construction (tested: identical commands) |
| G2 | H-shuf | measured in the run; the gate needs +10 over it |
| G4 | H-rand | measured in the run |
| G5 | P-reread − P-stale | **can exceed +16 by itself**: G5 alone is not blind-proof; G1 carries that in the conjunction |
| O1 | copy-last; N | copy-last passes the rank/std bars (ratio ≈ 1, as in TASK-066) but fails W/copy ≤ 0.8 (1.0); N fails W/N < 1 (1.0) and the action-sensitivity bars (1.0 < 1.10) |
| O2 absolute bars | persistence; clock prior | on `apple-look-v2-linux` moving windows (e9, perfect readout, val / train): persistence 3.48 / 3.83 cm (fails 1.5 cm); **the clock prior 1.49 / 1.54 cm sits at the 1.5 cm bar**, so the absolute bar is not blind-proof alone |
| O2 (i) | persistence; clock prior | persistence 1.0 by definition (fails ≤ 0.8); **the clock prior's ratio to persistence is 0.43 / 0.40 and would pass (i)**; hence (iii), which the clock prior fails at 1.0 |
| O3 | copy-last, N; prior-distance | copy-last and N give constant costs: ρ = 0 by definition; **prior-distance with the incumbent at the true plate reaches a median ρ of 0.63 on smoke seeds (§9.5), above the 0.5 bar**: the absolute bar is not blind-proof alone; the margin (ρ_W − best blind ≥ 0.3) is, and O0 separates "no headroom" from "the critic cannot rank" |
| O4 | the incumbent | regret difference 0, not < 0 (fails); its absolute regret is 0.03 cm on smoke seeds with the true plate (passes the 1.0 cm bar by itself; the paired bar and O0 carry the gate) |
| K0, D3 | — | condition and harness bars, not blind-proofing bars |

## 7. Void rule, abandonment clause, fallback

### 7.0 Every terminal row's consequence (`ROW_CONSEQUENCES`, frozen)

| consequence | rows |
|---|---|
| the abandonment clause fires | WMC-G2A, WMC-NO-RANK, HYB-HARM, HYB-NO-GAIN |
| the D7 fallback (TASK-074) is authorised | S-NO-CONDITION, R-NO-HEADROOM, NO-HEADROOM, H-NO-HEADROOM |
| escalate to the owner | WMC-NO-DYNAMICS (no closed loop runs), WMC-O2-VOID, WMC-O3-VOID, WMC-PROPOSAL (no closed loop runs), WMC-DEV-STOP (no gated stage is proposed), VOID, S-VOID-CONDITION, ESCALATE-BUDGET, ESCALATE-BUDGET-LAST-TWO |
| next stage (after its own reported GO) | K0-PASS → corpus; OFFLINE-PASS → D3; D3-GO → the owner's authorisation record |
| close without the clause | HYB-PASS (the claim, with its labels), HYB-SLOW (the owner decides), HYB-SCENE-BLIND (reported as a scene-blind gain), INCONCLUSIVE |

### 7.1 Void rule

A guard, a crash, a cap or a CUDA allocation failure makes the stage V and nothing in it is read;
batches are never shrunk to fit. **A V before a stage's first cohort render**
(`cohort_first_render_utc`; for example G-repro, G-quiet, G-threads or any preflight guard) **is not
a spent attempt**: it may be repeated as-is, with the V report kept, and needs no fix (owner
ruling). After the first render: a gated (S or U) V goes to the owner and a second V closes
TASK-073 as INCONCLUSIVE; a development stage (K0, corpus, training, ranking, D3) may be repeated
once from scratch after a reviewed fix. Nothing is re-thresholded, retrained or re-selected after
its numbers are seen.

### 7.2 Abandonment clause (fixed now)

It fires on **WMC-G2A**, **WMC-NO-RANK**, **HYB-HARM** (owner ruling) or **HYB-NO-GAIN**
(HYB-NO-GAIN and HYB-HARM are reached only with headroom present). It closes "P-3 proposals plus LeWM frozen-DINOv2-token critic selection on
`apple-to-plate-v2` at 112 px": no further critic, K, horizon or cost variant is preregistered on
`apple-shift-v2` or `apple-look-v2-linux` without new evidence of a different kind. The LeWM
backend, the token latent, the v2 task and the planner fallback stay open.

### 7.3 The owner's fallback (D7)

If TASK-073 ends **S-NO-CONDITION**, **R-NO-HEADROOM** (owner ruling 2026-09-29),
**H-NO-HEADROOM** or **NO-HEADROOM**, TASK-074, the
LeWM-only planner (`lewm-planner`), is pre-authorised as the next task. It needs its own
preregistration, with its own condition and floors; this protocol does not design it.

## 8. Platform, devices, caps

The Linux PC as in TASK-072: G-platform (Linux, x86-64, `MUJOCO_GL=egl`), MuJoCo 3.13.0, the main
process in strict CUDA determinism (`devices.require("cuda", strict=True)`). Simulation on the CPU
with 6 workers (1 torch thread each; 16 until the K0 run-1 V, §15); arms that make world-model
decisions (and H-sim), and the ranking stage, on 4 workers × 4 torch threads (ranking: 1 thread); closed-loop encoding and W rollouts on the CPU in the worker; corpus featurisation and
W/N training on CUDA. The GPU is shared with a resident service (about 6.6 GB) and short Isaac
containers: before a GPU stage the operator checks `nvidia-smi` and waits if memory is short; a
CUDA allocation failure is a V, never a smaller batch. Caps: 43 200 s global, 300 s per attempt,
5 400 s per model. Disk: the corpus is about 2 GB and the feature table about 5 GB (frames
240–660 only), inside the 10 GB free-space floor. Guards: G-frozen, G-hash (every pin, clean
tree), G-platform, G-device, G-weights, G-evidence (TASK-072 run-1's report, P-3/C-3/R-3, corpus
manifest), G-repro, G-cohort, G-seeds, G-look, G-frame, G-shift, G-anchor, G-finite, G-cap,
Q-split (asserted again at the end of the train stage), G-authorisation, G-threads (the pinned
thread environment, §0) and G-quiet (§13.2).

## 9. Stage-0 smokes and blind-baseline checks

All on the Linux PC, on smoke seeds 53950–53999 only, in `outputs/task073-scratch/` (git-ignored,
never committed). **Nothing in them informs a threshold or a choice**; each result below is a
mechanics check, and §9.5 is the blind-baseline check the orchestrator asked for. No bar was
changed after any of these numbers was seen.

### 9.1 smoke-1 (mechanics; at the pre-commit code, 150 s)

`run_wm_critic_v2.py smoke`. Its end check voided on G-hash because the design module was edited
during the run; the stages it had already written are mechanics and are kept only as such.
smoke-2 (§9.3) repeats them on the committed revision.

- **G-repro 8/8 exact** (44 s): TASK-072 run-1's readouts refitted and reproduced.
- **Render check IDENTICAL**: 32 smoke seeds × 4 post-look renders over 16 workers.
- **Proposal generator**: 4 P-truth attempts with a 4 cm shift at step 300 (applied on all 4);
  72 decisions; median relative chunk error 0.203.
- **Feature anchor**: CUDA vs CPU pooled features, max |difference| 2.8e-5 (bound 1e-3).
- **W training on strict CUDA, twice**: 500 updates each on an 8-root smoke corpus;
  **BIT-IDENTICAL** weights (sha256 `f211460e…`), losses and val curves; about 0.044 s per update
  (so 60 000 calibration updates take about 45 min per model); peak CUDA allocation 1.7 GB.
- **Decision latency** (4 H-LeWM attempts on 8 workers × 2 threads, the GPU's resident service
  running): median 0.214 s, p90 0.327 s, max 0.573 s over 72 decisions; no privileged read.

### 9.2 Stage smokes (every stage's code on smoke seeds, tiny budgets)

`--smoke` runs each stage on stand-in seeds with forced K0 selection (300, 4 cm) and 200-update
models, end to end: `k0` (4 seeds, all four arms; 232 s) → `collect_apple_shift_v2.py` (12 roots
53962–53973; 9 complete, 7 counted successes, 2 off the plate) → `train` (78 766 band frames incl.
`apple-look-v2-linux`'s 190 roots; anchor 2.4e-5; 669 s) → `rank_wm_critic_v2.py` (4 seeds × 5
points, 20 groups; 65 s) → `summarize offline` → `d3` (4 seeds, 8 arms; 766 s) → `summarize
results`. Every stage wrote its report; none voided. The rows they printed are meaningless (the
models are 200-update smoke models on 6 train roots) and are not read. Mechanics facts: H-sim
decisions take about 3.3 s (25 × 16 branch commands); H-N chose the incumbent at all 72 decisions
(the action-blind reduction, live); in the D3 smoke the H-LeWM decision median was 0.217 s
(p90 0.243 s), but H-N's was 0.395 s and H-shuf's 0.638 s, with the machine's load average above
the worker thread count, unexplained (hence D-18).

### 9.3 Runs at the committed revisions

- **smoke-2** at `114d8b8`: **V on G-repro** before anything else ran (the P and R readout
  selections, the S0-P errors and C-3's constant did not reproduce run-1's). It started just after
  the full test suite, with the machine's 5-minute load average near 13. An immediate
  **preflight-1** at the same revision (report `94bf03fe…cd39`) passed G-repro 8/8, and so did every
  other refit (smoke-1, smoke-3 and the four stage smokes: 7 of 8). The refit is CPU numerics in
  PyTorch (MKL, dynamic threading by default) and NumPy (OpenBLAS); a load-sensitive thread count
  is the likely cause, not established. **G-repro is checked before any cohort frame is rendered**,
  so such a V costs nothing; the runner's record of it stays, and an operator starts a stage only
  on a quiet machine (no other CPU-heavy job; the GPU's resident service may run).
- **smoke-3** at `114d8b8`, clean tree (report `5e6c0750…aa75a`, 146 s): G-repro 8/8; render check
  IDENTICAL (128 renders, 32 seeds, 16 workers); proposal generator median relative chunk error
  0.203 over 72 decisions (the same as smoke-1); feature anchor 2.8e-5; W training twice on strict
  CUDA **BIT-IDENTICAL** (`f211460e…`, 21.9 s and 22.6 s); decision latency with 8 concurrent
  H-LeWM attempts on 8 workers × 2 threads **median 0.295 s**, p90 0.340 s, max 0.699 s (144
  decisions; no privileged read), above G7's 0.25 s.
- **Latency probe** (scratch, same smoke seeds and smoke model, all attempts concurrent): 4 workers
  × 4 threads median **0.230 s** (p90 0.244 s); 8 × 1, 0.310 s. Hence D-19. The bar is unchanged.
- **Thread pins (owner ruling, after the review):** no simulation. On the 170 stored train roots'
  post-look frames of `apple-look-v2-linux` run-1 (the train split only), the DINOv2 features, the
  P readout's selection and its predictions are bitwise identical with and without the pinned
  environment (§0). The full G-repro under the pins is first checked by the GO's fresh preflight
  (§13.2); the pins and G-quiet were not exercised by a smoke, because the orchestrator restricted
  this round to smoke seeds and a preflight re-renders run-1's perception and D2 seeds.
- **smoke-4** at `86d2da8` (the code of this PR; only this section and the manifest's pin of this
  document changed after it), clean tree, report `1a5f1166…1bae`, 142 s: G-repro 8/8; render
  check IDENTICAL; proposal generator 0.203; anchor 2.8e-5; W training BIT-IDENTICAL; decision
  latency with 4 concurrent H-LeWM attempts on 4 workers × 4 threads, the GPU's resident service
  running: **median 0.230 s**, p90 0.331 s, max 0.643 s (72 decisions; no privileged read).

### 9.4 O2's blind baselines (stored data, no simulation)

On `apple-look-v2-linux`'s complete roots (e9, unshifted), with the true offset standing in for a
perfect readout, on the O2 decision steps 416–576: all windows — persistence 0.50 cm (train) /
0.65 cm (val), clock prior 1.37 / 1.19 cm; **moving windows** (≥ 1 cm motion; 685 / 86 windows)
— persistence 3.83 / 3.48 cm, clock prior 1.54 / 1.49 cm, clock prior ÷ persistence 0.40 / 0.43.
These drove D-6 and D-7 and are restated in §6.7.

### 9.5 O3/O4 blind baselines (smoke seeds 53970–53977, simulated)

P-3 with the **true** post-shift plate as its estimate (4 cm shift at step 300); at 416, 448, 480,
512 and 544 the 25 aims around the true plate were run for 16 commands in cloned state (35
groups; one seed ended early on the joint-velocity guard). The prior-distance ranker's median ρ
against the true costs was **0.63** (q10 −0.20, q90 0.67; −0.25 at 416, 0.63–0.66 later); the
incumbent's median regret was **0.03 cm**; the median spread of true costs across the 25 aims
was 1.27 cm. With the incumbent at the truth this is an upper bound for the prior ranker and a
lower bound for the incumbent's regret. It drove D-8 and D-17. (In the rank stage smoke, with a
6-root R-mid whose val error is 3.8 cm, prior-distance's median ρ was 0.02 and the incumbent's
regret 0.79 cm: the prior is strong exactly when perception is good.)



## 10. Deviations from the design outline

| # | outline | here | why |
|---|---|---|---|
| D-1 | shift directions "over the arc where `release_pose` does not clamp" | directions that keep e9's release point relative to the plate within 0.5 cm (§3.2) | e9's release is clamped at every v2 reset (tested); directions that unclamp the moved plate's release exist (3683 of 4000 reset × |d| pairs) but change the plate-relative release geometry |
| D-2 | the stand-in's previous targets implicit | measured arm joints; hands on the synergy of the controller's own last executed grasp | measured closed-hand joints lie off the synergy, and every chunk was infeasible in the first mechanics check |
| D-3 | o* from the new corpus's train split | o* frozen now from `apple-look-v2-linux`'s 112 counted-success train roots | K0's H-sim needs o* before the corpus exists; freezing it removes a degree of freedom |
| D-4 | "PR #104 is still open"; P-3's q50/q90 from the proposal | #104 merged (`70f1358`); q10/q50/q90 2.28 / 2.82 / 3.27 cm verified in the M2 results manifest | fact check |
| D-5 | H-sim candidates around R-mid's incumbent | in K0 around the true post-shift plate | R-mid does not exist before the corpus |
| D-6 | O2 on "post-shift windows" | TASK-054's moving cohort (≥ 1 cm true offset motion over 16 commands), t ≥ max(416, s) | on all windows the steady phase dominates and persistence is near-perfect (median 0.50–0.65 cm); G2a was defined on moving windows |
| D-7 | O2 (i) against persistence only | adds (iii), W against the clock prior o*(t + 16) ≤ 0.8 | the clock prior passes (i) by itself (0.40–0.43) |
| D-8 | "if any blind ranker passes 0.5, the gate is void" | voiding rankers are copy-last, N and L-shuf; prior-distance enters through the margin | prior-distance ranks by the incumbent, which is informative by design when R-mid is good; **set after smoke data** (§9.5, smoke seeds only); accepted by the owner |
| D-9 | O5 on "16 D3-style development P-truth attempts" | on K0's chosen cell's P-truth attempts (32) | the same controller and condition, available before any world model, without extra seeds |
| D-10 | "`R_off`, `R-mid` ridge readouts" | R_off primal on the 6144-d pooled latent; R-mid dual on the full 98 304-d tokens; λ relative to the mean linear-kernel diagonal in both | the closed loop evaluates both per decision; kernel readouts with 4 000+ reference rows would not fit in the workers' memory |
| D-11 | "`R-mid` on train frames at the decision steps" | the same, on the corpus's train roots, full tokens through the closed loop's CPU path | the P readout's feature and path |
| D-12 | L-shuf's start latent "from reset (i + 1) mod n" | in the closed loop, P-reread's recorded latent of reset (i + 1) mod n at the same decision step (P-reread runs first). **Fallback, preregistered (`shuf_latents`):** if that attempt ended before the step, its latest latent before the step; if it reached no decision step, the next reset in cohort order that did; never the reset's own; every substitution is recorded in the stage report | the other reset's latent must exist when the decision is made, and a P-reread attempt that ends early (a guard stop) must not void the stage |
| D-13 | featurise "about 390 k frames" | frames 240–660 of every root (about 190 k) | every decision window lies inside; the full table would need ~10 GB of disk |
| D-14 | module list | adds `wm_critic_v2_runtime.py` and `wm_critic_v2_offline.py` | spawned workers need an importable module; the offline statistics are testable apart from the script |
| D-15 | decision latency "with GR00T resident" | measured in the gated workers inside `act()`, all attempts concurrent | the arm's own configuration is what G7 bounds |
| D-19 | worker layout unspecified | world-model arms on 4 workers × 4 torch threads | at 8 × 2 the concurrent median was 0.295 s (smoke-3), above the 0.25 s bar; at 4 × 4, 0.230 s (a latency probe on the same smoke seeds and smoke model); a layout, not a threshold, and the margin stays thin (§9.3) |
| D-16 | O2 (ii) void → unspecified | a void O2 or O3 is its own row (WMC-O2-VOID, WMC-O3-VOID) that escalates to the owner | "void, not passed" needs a row |
| D-17 | O3/O4 directly after O2 | O0 first: the incumbent's median true regret on cohort R ≥ 0.5 cm, else **R-NO-HEADROOM**, a D7 fallback row (owner ruling) | on smoke seeds with the incumbent at the true plate its regret is 0.03 cm and prior-distance reaches ρ 0.63: without O0, a headroom finding would read as WMC-NO-RANK and fire the abandonment clause. **Set after smoke data** (smoke seeds only); accepted by the owner |
| D-18 | a G7 failure falls to HYB-NO-GAIN and fires the abandonment clause | a row HYB-SLOW when G1–G6 pass and only G7 fails; no abandonment (owner ruling) | a latency overrun is not evidence that the critic does not help. The smokes put H-LeWM's median at 0.214–0.295 s depending on layout and load: 0.214 s (smoke-1, 4 attempts on 8 × 2), 0.217 s (D3 stage smoke), **0.295 s** (smoke-3, 8 concurrent on 8 × 2) and 0.230 s at the chosen 4 × 4 layout (smoke-4: p90 0.331 s, max 0.643 s); H-N and H-shuf measured 0.395 s and 0.638 s in the D3 stage smoke under unexplained load (§9) |
| D-20 | O4 unspecified in how W's top-1 is chosen | W chooses among the candidates the stand-in found feasible (what the closed loop knows); a chosen candidate whose true branch stopped counts as the group's worst finite outcome | the earlier code filtered W's choice with the privileged truth (reviewer N3) |
| D-21 | — | a frozen `ROW_CONSEQUENCES` table (§7.0), G7's scope and the determinism re-run (§6.6), and the authorisation PR's scope (§13.3) | reviewer findings 3 and N5 |

## 11. Risks and caveats

1. **No headroom is the likeliest outcome** (P-reread may equal P-truth and H-sim). K0 and D3
   catch it within about an hour, before any GPU time. K0's bar 4 (H-sim − P-truth ≥ +4/32) also
   requires P-truth ≤ 28/32 in effect. On smoke seeds, with the incumbent at the true plate, the
   incumbent was already the best or near-best of the 25 aims (median regret 0.03 cm, §9.5): a
   perfect plate estimate leaves little for a critic, which is what K0's bar 4 and O0 test.
2. **P-3 has never seen its estimates change mid-episode.** P-truth may fall below 20/32 (then
   S-NO-CONDITION). A DAgger fix would need its own preregistration.
3. **The proposal stand-in may diverge from the executed commands**; O5 bounds it, and the
   H-sim vs H-LeWM gap mixes world-model error with stand-in error.
4. **Readability of an in-hand, moving apple through prediction is unproven.** TASK-066 showed a
   static apple only; O2 gates it. G2a has never passed in this project (0.83–0.90).
5. **The condition is artificial**, and the direction rule makes it mostly sideways (§3.2).
6. **Power.** n = 64 at a +10 bar can miss a true +8 effect.
7. **Pretrained vs random-init tokens** stays untested here (TASK-072 found no measurable
   contribution of pretraining for P-3's readout).
8. **Product reach.** Even HYB-PASS leaves P-3 as the controller.
9. **The o* target** is e9's unshifted median; under a shift the lift-phase offsets differ by d and
   no aim can change them there, so early decisions choose among near-equal costs.

## 12. Facts verified in code (2026-09-29)

- `EXPERT_BUDGETS = (130, 80, 45, 150, 100, 50, 30, 50, 30, 60)`; transfer starts at 405, steady
  ends at 585 (tested).
- e9's pick phases are invariant under a plate shift (tested); its release is pulled back to the
  reach sphere on every v2 reset (tested; D-1).
- `lewm.py` builds `Embedder(input_dim=14)`; the frozen-token model's action encoder is
  `EE_DELTA_GRASP_V0.dimension` = 14 wide and `max_horizon` is 64 ≥ 16: W runs at h = 16 as 16
  per-step 14-D actions, and no file behind an implementation hash changes (tested).
- Seeds 53000–53999 and 7300–7313 are free (search above; tested disjointness).
- P-3's checkpoint `checkpoints/task072-first-policy-v2-linux/run-1/P-3.pt` (under
  `worktrees/task072-run`) has sha256 `7988162df0b6106db5d007e322b824a3725b74cb103ba5a8de9f552912060be8`,
  equal to M2's pin; run-1's report `87745284…ec78` and corpus manifest `67c342f6…4f54` also match.
- M2 (from the merged results manifest): P-3 40/40 on cohort C, final distance q10/q50/q90
  2.28 / 2.82 / 3.27 cm; R-3 39/40; row M2-FAIL on G3 alone.

## 13. Process

### 13.1 This PR

Stage 0 merges on an independent reviewer's reported APPROVE, posted on the PR, and green CI.

### 13.2 The GO procedure (every stage from K0 on)

Each later stage (K0, corpus, train, rank, offline, D3, gated) starts from a clean worktree of the
merged revision on the Linux PC, and only after all of the following:

1. **The orchestrator is told before the stage starts.** K0 and every later stage count as gated
   for this purpose.
2. **A fresh pre-run reviewer's reported GO**, delivered as a message and posted on the PR, never
   read from a file. The GO includes:
   - the pre-run reviewer's own render-determinism check at the **frozen worker count (6)** on smoke seeds
     (`run_wm_critic_v2.py smoke`, its `render_check`), with verdict **IDENTICAL** or, by the
     owner's ruling of 2026-09-29 (§15.5), **MAJORITY-PASS** (every seed has a majority frame and
     every odd render has the same simulation state and differs by at most one level in at most
     16 pixels; the odd renders are listed in the GO comment); any other verdict means the stage
     does not start and the issue goes to the owner;
   - a fresh `run_wm_critic_v2.py preflight` from the merged revision into a new output name,
     ending **PREFLIGHT-READY** (G-repro 8/8, G-threads, G-quiet, G-cohort, every pin).
3. **The quiet-machine rule (`QUIET_MACHINE`, checked by the runner as G-quiet):** at the stage's
   start, `os.getloadavg()`'s 1- and 5-minute load averages are both at most 2.0; the idle machine
   with the GPU's resident service measures about 0.1–0.4. No other CPU-heavy job may be started
   during the stage. A G-quiet V is pre-render and may be repeated as-is.
4. **Memory (§15):** at the stage's start `MemAvailable` (`/proc/meminfo`) is at least 16 GiB, the
   12 GiB process-tree ceiling plus 4 GiB of headroom (checked by the runner as G-memory, before
   any render); during the stage the runner voids cleanly, with its report, if the summed RSS of
   its process tree exceeds 12 GiB.
5. **For a GPU stage (train; the smoke's training part):** `nvidia-smi` is checked immediately
   before the start and the stage waits while less than 4 GB of GPU memory is free (the smokes'
   peak allocation was 1.7 GB; the resident service holds about 6.6 GB). A CUDA allocation failure
   is a V, never a smaller batch.
6. The stage runs under `nohup` and writes its report, also on SIGTERM, SIGINT or SIGHUP (§15); the author reports the outcome to the
   orchestrator, and nothing is read or decided beyond the frozen rules.

### 13.3 The gated stage

It additionally needs the owner's authorisation record, a reviewed PR that pins it in the
manifest's `gated_authorization`. **Its scope is frozen (`AUTHORISATION_SCOPE`):** it may add the
gated harness (the runner's `stage_gated` and its tests), the record and that manifest block; it
may not change `wm_critic_v2.py`, the frozen block, `frozen_sha256` or the stored cohort values.
`tests/test_wm_critic_v2.py` pins the frozen hash as a literal and is itself pinned in the
manifest.

## 14. Amendment log

- 2026-09-29, after the K0 run-1 V (owner ruling "Fix PR, then repeat"): §15. The frozen block
  changes only in `sim_workers` (16 → 6) and the new `memory` block; `frozen_sha256` `f476eb9a…` →
  `cd9e8723…`. No gate, threshold, cohort, arm or row changes.

## 15. Addendum (2026-09-29): the K0 run-1 V and its fixes

### 15.1 What happened

K0 run-1 started at 12:44:37 on 2026-09-29 from a clean worktree at `b4df3f0`, after the pre-run
reviewer's reported GO (PR #106 comment 5888594328) and with the quiet-machine rule met (load
0.12 / 0.68 / 4.86). It passed its preflight guards and G-repro and rendered its first cohort frames.
At about 12:53:56 **an external host agent sent SIGTERM to its whole process group** after memory
pressure: the run's 16 simulation workers (about 1.45–1.60 GiB each) took the launcher's cgroup to
a peak of **25.9 GiB** on the 30 GiB machine, next to the GPU's resident service. No kernel OOM kill
occurred. The runner wrote no report (a Python `finally` does not run on an unhandled SIGTERM).
**K0 run-1 is recorded as V**, after the first cohort render; nothing from it is read, and its log
is kept untouched (`/home/huhn/develop/emai/worktrees/task073-k0-run-1.log`).

### 15.2 Owner rulings (2026-09-29, relayed by the orchestrator, verbatim)

| question | the owner's answer |
|---|---|
| the K0 run-1 V | **"Fix PR, then repeat (Recommended)"** |
| the host agent | **"I'll tell Hank to leave gated runs alone"** |

The one K0 repeat follows this PR's merge, a new reported GO and the owner's notice.

### 15.3 Fixes (this PR)

1. **Memory.** The frozen block changes in exactly two fields: `sim_workers` 16 → **6**, and a new
   `memory` block (`MEMORY`: a 12 GiB ceiling on the runner's process tree — summed RSS in the
   first version of this PR, summed PSS since the review (§15.5) —
   4 GiB headroom, sampled every 0.5 s). `h_workers` (4 × 4 threads) is unchanged; the ranking
   stage now uses it instead of `sim_workers`, because its workers load the encoder.
   `frozen_sha256` is now `cd9e8723…44f83` (was `f476eb9a…01b5e`); the pins and the test's literal
   are updated. **No gate, threshold, cohort, arm or row changes.** Runtime changes outside the
   frozen block: workers load the DINOv2 encoder and the kinematic stand-in on first use (a K0 or
   corpus worker needs neither); R-mid's dual ridge works from the linear kernel built in column
   blocks, with the full tokens stored as float32 (the same estimator, without a float64 copy of
   the 98 304-d rows); O1 keeps only the gated horizons' predictions. Guards: **G-memory** at the
   start (`MemAvailable` ≥ 16 GiB, pre-render) and the **runtime ceiling** (a watch thread sends
   SIGUSR1 above 12 GiB; the stage ends V with its report). Every report records `memory` (the
   peaks of the tree, the main process and the largest child).
2. **Signals.** SIGTERM, SIGINT and SIGHUP (and the ceiling's SIGUSR1) end every runner's stage as
   V with the report written: `void_reason` names the signal and `interrupted_utc` its time; the
   pool is terminated first. Tested with a live subprocess for each signal and for the ceiling;
   also exercised by a real SIGTERM on a slow train smoke (§15.4).
3. **N-d, the pre-render boundary.** The corpus stage now records `cohort_first_render_utc` just
   before its first root is collected: a corpus V before it is not a spent attempt (§7.1). The
   **train stage renders no seed** (it reads stored frames only), so every train V is pre-render
   and may be repeated as-is. The ranking and closed-loop stages already recorded it.
4. **The renderer (found while measuring; not in the owner's brief).** The EGL renderer
   occasionally returns, for an identical simulator state, a frame that differs by one intensity
   level in a few pixels (up to 7 pixels; about 1 frame in 1 000 in the corpus smoke; states,
   commands and every other array identical). It explains the load-independent G-repro Vs (one
   re-rendered frame of 400 changes run-1's reproduced facts), one G-frame V in a 6-worker K0 smoke,
   and a render check with verdict DIFFERENT (smoke-b below). Mitigations, all recorded in the
   reports, none of them a tolerance:
   - **G-repro's re-render and every cohort's post-look frames are rendered twice** in separate
     tasks; if the two differ a third render decides by majority, and no majority is a V
     (`render_majority`; disagreements recorded);
   - **G-frame keeps its bitwise check**, but on a mismatch the attempt's worker renders the same
     post-look state again, at most twice, and proceeds only if a re-render matches exactly
     (`check_post_look_frame`; every re-render hash recorded);
   - the render check now also reports `seeds_with_differing_renders`, `odd_renders` and
     `every_seed_has_a_majority`.

   K0's arms read no image after the post-look frame (P-3's estimates come from the cohort frame),
   so with these K0 is unaffected. **Open for the owner, before D3 and the gated stage:** arms that
   read images at decision steps (P-reread, H-LeWM, H-N, H-shuf, H-rand) can see a glitched frame,
   so their commands are not guaranteed bitwise reproducible (one such difference appeared in a D3
   smoke: 3 of 32 attempts differed between 4 and 2 workers in verify-1, none in verify-2), and the
   gated determinism re-run (§6.6, "differs" bitwise) could then make the run VOID. Also, the GO's
   render check (§13.2) says "IDENTICAL, otherwise the issue goes to the owner"; it can now fail on
   a single odd render. Whether a DIFFERENT verdict with a majority for every seed may count for
   the GO is the owner's call; this PR does not change the rule.

### 15.4 Measurements (smoke seeds 53950–53999 only; worktree at this PR's code)

Peak summed RSS of the whole process tree (GiB), each stage type at the new counts
(`verify-2`, reports in `outputs/task073-fix/verify-2/`, git-ignored):

| stage | workers | peak tree | main | largest child | report sha256 |
|---|---|---|---|---|---|
| smoke (render check, proposals, 2 × W training, latency) | 6, then 4 × 4 | 11.07 / 11.04 | 3.96 / 3.58 | 1.94 | `b20ee33a…`, `48b36d0c…` |
| K0 (all four arms, H-sim included) | 6 | 9.85 | 1.99 | 1.45 | `86ed8524…` |
| corpus (12 roots) | 6 | 9.05 | 0.83 | 1.39 | `edc82799…` |
| ranking (4 seeds × 5 points) | 4 | 9.02 | 2.23 | 1.72 | `29ac4e7a…` |
| D3 (8 arms) | 4 × 4 | 9.52 | 1.82 | 2.06 | `57bd0019…` |
| train (smoke scale: 78 766 band frames, 200-update models) | main only | 8.21 | 8.21 | — | killed by a test SIGTERM (V report written) |

The full-scale train stage cannot be measured on smoke seeds: its feature table is about 4.8 GB
instead of 1.9 GB, so it should peak near 11 GiB. The runtime ceiling voids it cleanly if not,
and that V is pre-render (item 3). A train smoke under another user's CPU-heavy job ran about five
times slower than before (OpenBLAS threads oversubscribed); the G-quiet rule prevents that at GO.

**Worker-count equivalence** (the same smoke seeds, the same code):
- K0 at 6 and 2 workers: all 16 attempts (four arms × four seeds) identical in success,
  termination, executed steps, final distance, grasp and place steps, H-sim's chosen aims, the
  post-look frame hash and **the executed command array (sha256)**, in two separate verification
  rounds. Against the 16-worker K0 smoke of Stage 0 (commands were not hashed then): every other
  field and every H-sim choice identical, and the cohort estimates identical.
- D3 at 4 and 2 workers (all eight arms): identical, commands included, in verify-2; in verify-1,
  3 of 32 image-reading attempts differed (item 4).
- Corpus at 6 workers against Stage 0's 16-worker corpus smoke: all 12 episodes' metadata
  identical, 10 of 12 array files identical; the two others differ only in 1–3 frames by one
  intensity level (item 4).
- smoke render check: IDENTICAL (smoke) and DIFFERENT with one odd render and a majority for every
  seed (smoke-b); W training BIT-IDENTICAL in both; decision latency median 0.232 / 0.228 s.

### 15.5 The review of #108 and the owner's rulings on it (2026-09-29)

The independent review of this PR (REQUEST CHANGES at `787ffac`) found two blocking defects and
asked for owner rulings on the renderer mitigations. **Owner rulings (verbatim):**

| question | the owner's answer |
|---|---|
| the renderer mitigations (majority of three for G-repro and cohort frames; the G-frame re-render) | **"Accept, record every retry (Recommended)"** |
| may a DIFFERENT GO render check count as a pass? | **"Yes, count it as a pass (Recommended)"** |
| the determinism re-run for image-reading arms | **"Match decisions and results (Recommended)"** |

**How each ruling is made operational (frozen: `RENDER`, `IMAGE_READING_ARMS`,
`RERUN_IMAGE_ARMS`).**
- **Renderer votes and retries** (`render_majority`, `check_post_look_frame`) are allowed only when
  (a) the whole simulation state — `qpos`, `qvel`, `act`, `ctrl` and `time`, hashed
  (`state_sha256`) — is bitwise equal across the disagreeing renders (for G-frame: equal to the
  state the cohort frame was rendered from, and unchanged by the re-renders), and (b) every odd
  frame differs from the reference frame by at most **1 intensity level in at most 16 pixels**
  (`RENDER`; the largest odd frame seen was 7 pixels in this PR's smokes, 4 in the reviewer's
  10 100-render probe). Anything else is a V, with the pixel differences in the report. Every
  disagreement and re-render is recorded with its pixel count and largest level
  (`render_disagreements`, `rerender_disagreements`, `render_retries`).
- **The GO render check** (`render_check_verdict`): IDENTICAL; or **MAJORITY-PASS** — every seed
  has a majority frame and every odd render passes (a) and (b) — which counts as a pass and whose
  odd renders are listed in the GO comment; otherwise DIFFERENT, and the owner decides.
- **The determinism re-run** (§6.6): non-image arms stay bitwise. For image-reading arms
  (`IMAGE_READING_ARMS`: P-reread and every H arm except H-sim) `rerun_matches` requires, on each
  re-run seed: **exactly equal** counted success, grasp, at-rest verdict, termination reason,
  executed steps and the chosen-aim index at every decision; and **within tolerance** the final
  apple-to-plate distance (±1.0 cm), the first grasp and first place steps (±40 steps each) and
  R-mid's incumbent at each decision (±1.0 cm). The executed commands are not required to match;
  their hash and every other difference are logged. The tolerances cover, with margin, the
  largest differences seen between re-runs of image-reading arms on smoke seeds (verify-1 D3,
  three attempts: final distance 0.41 cm, place step 25, incumbent 0.45 cm; one of the three,
  H-shuf on 53959, also chose a different aim at one decision, which this rule counts as a
  difference). Gates, thresholds, cohorts, arms and rows are otherwise unchanged.

**Blocking fixes.**
- **B1 (a process-group SIGTERM could hang the runner).** The handler used `SIG_IGN` against
  re-entry; respawned pool workers inherited the ignored SIGTERM, so `pool.terminate()` +
  `join()` could block forever. Now: re-entry is a module flag (a second signal is recorded, not
  raised); `worker_init` resets SIGTERM/SIGINT/SIGHUP/SIGUSR1 to `SIG_DFL`; `Pool.close` terminates,
  joins each worker with a 10 s timeout and then SIGKILLs any survivor it owns. Tested with a live
  spawn pool and `killpg` (three trials per run; 12 of 12 local trials wrote the V report).
- **B2 (a first signal in except/finally lost the report).** `finish_guards` is the first call of
  every runner's `finally` (run, collect, rank), and `void` sets the flag first: from there on
  signals are recorded (`signals_during_cleanup`), and the report is always written; a signal that
  interrupts the except clause itself is still turned into the V. Tested with a signal delivered
  during a slow pool close.

**Non-blocking findings.**
1. **A dead worker** (killed alone) now voids `map` within about a second (`G-worker`, the pid and
   exit code) instead of waiting for the stage cap. Tested.
2. **Memory measure.** The guard now uses the summed **PSS** of the process tree
   (`/proc/<pid>/smaps_rollup`), which divides shared pages among the processes sharing them; summed
   RSS, which counts each shared library once per worker, is recorded beside it. The 12 GiB ceiling
   is unchanged, and PSS tracks the physical memory the pane's cgroup limit (MemoryHigh 14 GiB)
   actually meters. Both are measured below (§15.6). The per-pid walk no longer drops a pid's
   children when one of its threads exits mid-sample. The guard remains a sampled check (a long
   NumPy call in the main thread delays its signal), not a hard limit.
3. **The train boundary.** The train stage records `first_outcome_utc` immediately before the
   first number computed on val roots (R-mid's val error, then the blind baselines, the budget
   curves, O1 and O2). A train V before it may be repeated as-is; a V after it uses the stage's one
   repeat (§7.1). The corpus stage's boundary is `cohort_first_render_utc` (item 3 of §15.3).
4. **Ranking equivalence** at 4 vs 2 workers is in §15.6. **A sealed corpus is not bitwise
   reproducible across collections**: its frames are not voted, so a re-collection can differ in a
   few frames by one intensity level; the sealed corpus's hashes are what later stages pin.

### 15.6 Measurements of the review round

_Filled in from the verification at this PR's code (see the PR comment)._

**Learned Apple→Plate on the frozen benchmark is still 0 successes.**
