# After C1-TWINS-ESCALATE: the next direction for a LeWM decision on v2 (DRAFT ruling, R15)

**Status: DRAFT ruling. It is not a protocol and nothing in it is frozen.** No gated run was made
for it, no cohort seed was simulated, and no bar below is in force until an independent review
approves this ruling. It answers the escalation of the C1 feasibility record
([apple_lewm_next_v2_c1_feasibility.md](apple_lewm_next_v2_c1_feasibility.md), C1-TWINS-ESCALATE,
R14.10: "the next step is the owner's choice among the escalation options").

Every choice in it is **decided by Claude under owner delegation (2026-09-30)**. The rulings are
R15.1–R15.10 in [DECISIONS.md](../DECISIONS.md), decision 2026-10-04 (d). R1–R14 are taken; a search of
every local and remote ref and every worktree's `docs` on 2026-10-04 found no R15.

The development measurements in §3 are **development only**: a CPU re-analysis of the C1 record's
run-2 artifacts and reset-value arithmetic, with no episode simulated, no world model and no GPU
(§3.1). None of them is a result, a gate or a learned outcome.

The canonical status sentence (DECISIONS 2026-10-02, R7) is unchanged by this ruling, verbatim:

> Learned Apple→Plate on the frozen v1 MVP benchmark (TASK-020) is 0/150 per backend
> (`native_jepa` and LeWM). On `apple-to-plate-v2`, the behaviour-cloning/DAgger policy P-3 (an
> MLP on a frozen DINOv2 readout, trained on demonstrations from the privileged scripted expert
> e9; not a world model) scored 40/40 counted successes on the held-out cohort C against 39/40
> for its random-init encoder control R-3 (one run, one training seed per arm, 40 resets, one
> camera at 112 px onboard, a narrow reset distribution), so TASK-072 M2 is M2-FAIL on G3
> (encoder pretraining contributed nothing measurable), and cohort C is no longer held out. No
> LeWM-driven controller has run in closed loop on v2 yet; LeWM's only closed-loop Apple→Plate
> runs are on v1, with 0 successes. Scripted-expert, privileged-ceiling, oracle and GR00T
> successes are not project-learned results. (LeWM also ran in closed loop on
> the TASK-014 development reach pilot, a reach task, not Apple→Plate: its v2 target-space
> selector reached 1/5 goals, with intervals overlapping the 0/5 controls.)

---

## 1. Where we stand

The C1 record (R14; CPU, no world model, one run, 32 paired development resets) escalated for two
reasons that any next design must answer:

- **The scene-blind twin is too close to the ceiling.** The mean-proxy (L-mean's stand-in) scored
  25/32 against H-final(commit)'s 31/32: a headroom of +6/32, interval [+1, +11], below the
  +8/32 bar and *near* under R8.14's guard, with a McNemar feasibility of 0.728 (bar 0.8). The
  shuf-proxy (21/32, +10, [+4, +16]) passed only under R14.5's point reading.
- **The readout at the read step is not shown to be precise enough.** At r = 460 the pooled 4 × 4
  readout of the encoded true frame read the moved plate to 0.886 cm in median ([0.778, 0.993])
  against the note's *provisional* 0.5 cm, a bar that was never calibrated from a tolerance
  measured under a committed aim (record §2.6).

What passed, and stays useful: the rule leaves 6.72 cm of plate motion after 405; the privileged
ceiling holds (31/32, none refused); every aim in the box is reachable (a_lo = −0.5); the readout at
r is not keyed on the arm (plate-hidden lower bound 3.46 cm); the action-blind and no-prediction
proxies lose clearly (N-proxy 12/32, H-now 0/32).

**Three LeWM designs in a row hit the same wall.** TASK-073 (S-NO-CONDITION: privileged look-ahead
added at most +3/32 over P-3 given the true plate), TASK-074 (INCONCLUSIVE; its readout bar sat
below the measured readout error, and under its condition an image-free clock prior later scored
51/64 in TASK-076's closed loop) and C1. Each time the task turned out too
regular for a world model's prediction to be measurable: against an arm that does not predict
(TASK-073), or against an image-free or scene-blind twin (TASK-074's condition, C1). §2 names the cause, §3 measures it, and §4–§5 choose a direction that removes it by a
declared condition change rather than another tweak.

## 2. The common cause, stated as a calculation (R15.2)

**The scene-blind miss does not depend on the rule's constants.** Under a rule linear in palm
displacement, plate(s1) = p + κ·(g − h) when the palm ends at the aim g (design note §2; p the
plate and h the palm at the decision). A twin that predicts with the mean plate p̄ instead of p
aims at its fixed point g_m = (p̄ − κh)/(1 − κ). The plate then stops at p + κ(g_m − h), so the
landing miss is

> g_m − p − κ(g_m − h) = p̄ − p,

whatever κ, L and the commit step are. The same algebra gives L-shuf a miss of p′ − p (the note's
§4.1 already stated both for κ = −0.5). The identity assumes, as the note does, that the palm ends
at the aim and that the fixed point lies inside the box (a = −κ/(1 − κ) ∈ [a_lo, 0.5]). So **at a
fixed tolerance curve, the headroom over the scene-blind twins is set only by the spread of the
plate at the decision step, mapped through the committed place's tolerance.** That curve is
measured only at κ = −0.5, L = 2, s0 = 405 (§3.2); another κ, L or commit step could change the
curve itself, which is unmeasured. Two consequences:

- R9.10's remedy 3 ("a different commit step or rule"), as far as it means another commit step,
  L or κ under a rule of this linear form, does not change L-mean's miss while the plate is static
  before the decision; it could help only through a sharper tolerance curve, which nothing
  measured suggests, so it is not expected to remedy C1-F3. (A rule of another form is a new design and
  would need its own Stage −1, below.) Remedy 2 ("a larger gated
  cohort") raises power but not the headroom itself: the point estimate (+6, interval [+1, +11],
  *near* under R8.14) and the Stage −1 projection (25.4/32, about +5.6 against a 31/32 ceiling)
  both sit below +8, and a larger cohort narrows the interval without being expected to move it. Only remedy 1, more plate spread at the decision, acts on the cause.
- The same holds for TASK-074's condition, and §3.3 shows why its 9 cm move did not widen the
  spread either.

**The rule this adds (R15.2).** Before any later LeWM design reaches a feasibility record, it
computes, from its declared reset and condition distributions alone (no simulation), each twin's
expected miss and maps it through the best measured tolerance curve then available. A design
whose projected headroom over any gating twin is below its bar, **read as the bar will be read**,
is not drafted: under a strict (interval) reading the projection minus the expected interval
half-width (about 5/32 at n = 32, from C1's paired intervals) must reach the bar. This "Stage −1"
is arithmetic on reset values and costs seconds; §3.3 is its first instance.

## 3. Development measurements made for this ruling (not gated; no new simulation)

### 3.1 How they were made

- **Script:** `scripts/r15_direction_dev.py`, one invocation, CPU only (`CUDA_VISIBLE_DEVICES`
  empty; no GPU lock, no CUDA context), 29 s. Report `outputs/r15-dev-1/report.json` in the
  `next-direction-r15` worktree (git-ignored), sha256
  `d3d95e04d2d95a83ca545b607833321d9fd2cc70c7744944314423c36ab06984`. It was made on the
  uncommitted script; a repeat at the committed `ca8fb99` (`outputs/r15-dev-2/report.json`) is
  byte-identical (the only edit between them moved the CPU-only environment setting into `main`), and so is a repeat at `a926405` after
  comment-only script edits (`outputs/r15-dev-3/report.json`).
- **Inputs:** the C1 record's run-2 `report.json` and `corpus.npz`, checked against the sha256 the
  record lists (`7779709c…fcef6`, `93e96c3f…cdb5`) before anything is read. No episode was
  simulated; run-2's frames are re-encoded with the pinned DINOv2 ViT-S/14 on the CPU.
- **Seeds (R15.9):** 58000–58511, used **only** to draw reset values (and, for TASK-074's move, its
  own direction rule and salt 7413) for the spread arithmetic. Nothing was simulated on them.
  R15 declares the block 58000–58999; the rest of it is unused. A search on 2026-10-04 of every
  local and remote ref and every worktree's `src` and `scripts` found 58000–58999 only as byte
  counts and row counts (`uv.lock`, two manifests' `rows_added`), never as a seed.
- **Salts:** 7801 (the learning curve's subsample), 7802 (the disc and half-disc draws). 7801–7809
  appear nowhere in `src`, `scripts`, `tests` or `configs` on any ref.

### 3.2 The committed place's tolerance, a development estimate

Counted success against the landing miss |committed aim − plate(525)| over all 607 completed
single-commit attempts of run-2 (the ceiling, the four proxies and two offset variants, the four
single-commit comparators, and the 255 corpus roots with aims uniform over the box):

| landing miss (cm) | 0–0.25 | 0.25–0.5 | 0.5–1 | 1–1.5 | 1.5–2 | 2–2.5 | 2.5–3 | 3–4 | 4–5 | 5–7 | ≥ 7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| attempts | 90 | 31 | 51 | 28 | 39 | 33 | 26 | 42 | 39 | 46 | 182 |
| counted successes | 89 | 28 | 45 | 24 | 33 | 19 | 13 | 20 | 7 | 4 | 0 |
| rate | 0.99 | 0.90 | 0.88 | 0.86 | 0.85 | 0.58 | 0.50 | 0.48 | 0.18 | 0.09 | 0 |

- **Check.** Mapping each arm's own landing misses through this curve gives 23.9/32 for the
  mean-proxy (measured 25), 22.0 for the shuf-proxy (21), 9.0 for the N-proxy (12) and 31.6 for
  H-final (31). It is within about ±3/32.
- **Caveats.** The check is partly circular (those attempts are in the curve). The attempts share
  32 resets and are correlated within a reset; the curve ignores the miss's direction; and the
  plateau of about 0.86–0.90 from 0.25 to 2 cm may be a few resets that fail at any small miss
  (seed 57014 failed even under H-final) rather than a smooth tolerance. **It is not τ_commit**,
  which a planted-error measurement must give (§5.3, M-F4).
- **What it says.** Under a committed aim the place tolerates about 2 cm at roughly the 85 % level,
  more than TASK-076's re-aimed τ curve (17/32 at 2 cm). That is why a scene-blind twin missing by
  a median 1.6 cm reaches 25/32: v2's ±2 cm plate jitter is inside the committed place's
  tolerance.

### 3.3 The plate's spread at the decision, and the scene-blind twins it implies

Reset values for seeds 58000–58511; expected counts are the curve of §3.2 applied to |p − p̄|
(mean twin) and |p − p′| with p′ the next seed's plate (shuf twin). These are projections, not
measurements of an arm.

| plate at the decision | median \|p − p̄\| (cm) | median \|p − p′\| (cm) | mean twin (of 32) | shuf twin (of 32) |
|---|---|---|---|---|
| v2's reset, no move (C1) | 1.64 | 2.06 | 25.4 (measured 25) | 21.5 (measured 21) |
| TASK-074's −y move, 9 cm | 1.61 | 2.14 | 24.6 | 20.8 |
| TASK-074's −y move, 12 cm | 1.61 | 2.14 | 24.8 | 20.9 |
| disc move, ρ = 3 cm | 2.31 | 3.16 | 20.1 | 15.5 |
| disc move, ρ = 4 cm | 2.93 | 4.10 | 16.2 | 11.6 |
| disc move, ρ = 5 cm | 3.49 | 4.94 | 13.5 | 8.4 |
| disc move, ρ = 6 cm | 4.24 | 5.48 | 10.8 | 7.1 |
| −y half-disc move, ρ = 3 cm | 2.09 | 2.89 | 21.5 | 16.7 |
| −y half-disc move, ρ = 4 cm | 2.45 | 3.20 | 18.9 | 14.5 |
| −y half-disc move, ρ = 5 cm | 2.84 | 3.76 | 16.5 | 12.2 |
| −y half-disc move, ρ = 6 cm | 3.17 | 4.49 | 14.7 | 10.1 |

- **TASK-074's move widens nothing.** Its direction rule (TASK-073's eligibility, restricted to
  225–315°) leaves only a few whole-degree directions per reset; the drawn directions lie at
  261–270° (9 cm) and 260–267° (12 cm) between their 5th and 95th percentiles. The move is almost a
  fixed translation, so the spread around the mean is v2's own. This is consistent, after the
  fact, with TASK-076's image-free H-clock reaching 51/64 under that condition.
- **A disc move of a few centimetres does.** At ρ = 4 cm the projected mean twin falls to about
  16/32 (headroom about +15 over a 31/32 ceiling), at ρ = 3 cm to about 20/32 (+11).
- "Disc" draws the offset uniformly over a disc of radius ρ around the reset's plate; "−y
  half-disc" over its −y half. ρ runs over TASK-073's |d| grid (3, 4, 5, 6 cm), used as radii.

### 3.4 The readout at r, by token grid, with a learning curve

Run-2's 255 corpus frames at r = 460 (and their plate-hidden renders), cross-fitted exactly as
C1-F5 (5 outer folds by root, salt 7702; λ by inner CV, salt 7703; dual ridge):

| latent read | median error (cm) | 95 % interval | 87.5th pct | plate-hidden median [interval] | curve-mapped count (of 32) |
|---|---|---|---|---|---|
| pooled 2 × 2 | 1.013 | [0.901, 1.132] | 1.86 | 2.87 [2.66, 3.09] | 27.1 |
| **pooled 4 × 4 (W's latent; C1-F5)** | **0.886** | [0.778, 0.993] | 1.52 | 3.87 [3.46, 4.28] | 27.8 |
| pooled 8 × 8 | 0.642 | [0.596, 0.730] | 1.21 | 2.88 [2.62, 3.10] | 28.3 |
| full 16 × 16 | 0.670 | [0.600, 0.735] | 1.20 | 2.46 [2.19, 2.63] | 28.5 |

The 4 × 4 row reproduces C1-F5's number and interval exactly (the anchor check).

| fit rows per fold | 51 | 102 | 153 | 204 |
|---|---|---|---|---|
| 4 × 4 median (cm) | 1.449 | 1.142 | 0.968 | 0.886 |
| 8 × 8 median (cm) | 1.289 | 0.988 | 0.726 | 0.642 |

- **The readout is data-limited.** Both curves still fall from 153 to 204 fit rows (by 8–12 %).
  The smoke corpus had 255 roots; a protocol corpus is larger. How far it falls is not measured.
- **An 8 × 8 grid reads the moved plate about 28 % better** than 4 × 4 on the same 255 rows at
  full data (the learning curve's subsamples are not paired between grids or nested across
  fractions), and as
  well as the full tokens. `frozen_tokens.token_grid` already supports it, but TASK-066's dynamics
  gates cover only 4 × 4.
- **A centimetre bar is the wrong instrument here.** Mapped through §3.2's curve, every grid's
  readout costs about 3–4.5/32 against the curve-mapped ceiling (31.6), nearly independent of its
  median, because the
  curve is flat from 0.25 to 2 cm. Whether that cost is real, or the plateau's few hard resets that
  every arm shares, can only be told by a paired closed-loop measurement, which §5.3 adds (H-read).

## 4. The options (R15.3)

| | option | acts on the scene-blind headroom? | acts on the readout at r? | assessment |
|---|---|---|---|---|
| **O1** | **C1-M: C1 plus a declared post-pick plate move over a disc (radius set by the ceiling), with the readout judged by a paired oracle-dynamics arm (H-read) on a larger corpus, 4 × 4 then 8 × 8** | yes: projected +11 to +20 over L-mean (§3.3) | yes: a calibrated, paired criterion, more data, a declared finer grid | **recommended** |
| O2 | R9.10 remedy 1 literally: widen v2's reset jitter for the plate | yes | no | not recommended: a plate far from P-3's training range at reset breaks P-3's pick (TASK-074 design Probe B: with a far plate at reset, P-3 grasped 12/32 and 8/32 at 9 and 12 cm); the move after the grasp latches (O1) gives the same spread without touching the pick |
| O3 | TASK-074's −y move as the declared "existing distribution" | no (§3.3: median 1.61 cm, unchanged) | no | rejected by computation |
| O4 | R9.10 remedy 2: a larger gated cohort | no (power only) | no | not taken: the point headroom (+6) and its Stage −1 projection (about +5.6) are below +8 |
| O5 | R9.10 remedy 3: another commit step, L or κ | not at a fixed tolerance curve (§2: the miss is p̄ − p for any κ, L or commit step) | changes r only | not taken: not expected to help (derivation; the curve's dependence on κ, L and the commit step is unmeasured) |
| O6 | a per-reset κ signalled by a visible cue (for example plate colour, `plate_color.py`) | yes: a mean twin then also misses by \|p − h\|·\|κ̄ − κ\|/(1 − κ̄), about 2.7–4.2 cm for κ ∈ {−0.25, −0.75} (*estimate*) | no; and colour shifts the plate-hidden check (white-plate record) | fallback only: contrived, and it adds a second perception risk |
| O7 | C2 (a single pre-pick push) or a pause of LeWM on v2 in favour of Arena (TASK-078) | C2: yes, at 3–5 days' development (design note §4.2) | no | not now: C2 stays R9.5's fallback; Arena is development-only and admits no LeWM claim |

## 5. Recommendation: O1, "C1-M" (R15.4)

### 5.1 The condition, declared as a condition change under R2

C1 unchanged (v2's own reset; P-3's pick; cell A's rule, κ = −0.5, L = 2, s0 = 405, s1 = 525; a
single aim committed at 405 on the box a ∈ [a_lo, 0.5], b ∈ [−3, +3] cm; e9's place primitive)
**plus one declared change: a post-pick plate move** (R15.5).

- **When and how.** Just before the observation of step 300, after the grasp has latched (about
  step 265), by TASK-073's harness hook (`plate_shift.py`: a `body_pos` write plus `mj_forward`,
  outside every controller), as TASK-073 and TASK-074 did. The plate is static from 300 to 405,
  then follows the rule.
- **Its distribution.** An offset drawn uniformly over a disc of radius ρ\* around the reset's
  plate, one draw per reset with its own declared salt, stored per cohort with a digest. No
  direction-eligibility rule: that rule existed to keep e9's stale release comparable (TASK-073),
  and the ceiling check below covers feasibility here. A move that leaves the plate off the
  tabletop is re-drawn by a declared rule; a `SHIFT_BLOCKED` move counts as a failure in every
  arm.
- **ρ\* is set by the ceiling only, never by a headroom number** (as a_lo is set by reach only).
  ρ\* is the largest radius in TASK-073's |d| grid {3, 4, 5, 6} cm such that it and every smaller
  radius reach H-final(commit) ≥ 30/32, refused and blocked moves counted as failures (R9.6's
  ceiling bar, in the monotone form of R14.4). If 3 cm fails, the same rule runs once on the −y half-disc (Probe A of
  TASK-074's design: e9 failed towards +y at 8–12 cm). If neither family passes at 3 cm, the
  record ends M-INFEASIBLE.
- **Why prediction from the current frame is then needed.** The decision at 405 needs both the
  plate's moved position, which only a frame after the move shows, and its action-dependent future,
  which only a model of the rule gives:
  - H-now (no prediction) misses by about |p − h|/2 (C1: 0/32);
  - N (action-blind) misses by 1.5·|a_N − 1/3|·|p − h| (C1: 12/32);
  - a stale twin that reads the plate before the move misses by the move's offset: projected
    about 21.7, 18.9, 14.2 and 10.7/32 at ρ = 3, 4, 5 and 6 cm through §3.2's curve (10⁵ draws,
    review computation; *estimate*), slightly above the mean twin at the same ρ. It is reported in
    M-F6 (H-rule-stale), not gated;
  - L-mean and L-shuf (scene-blind) miss by p̄ − p and p′ − p, now 2.3–4.2 cm and 3.2–5.5 cm in
    median (§3.3);
  - H-rule and H-sysid, hand-written with the rule, read the static plate at 405 and are expected
    near the ceiling. They decide only the secondary claim, which is not expected (R9.8).

### 5.2 The readout at the decision point (R15.6)

- **The provisional 0.5 cm bar (τ_re/2) is withdrawn.** It was never calibrated from a tolerance
  measured under a committed aim, and §3.4 shows a centimetre bar does not predict the count.
- **It is replaced by two calibrated quantities, both measured in the feasibility record:**
  - **τ_commit**, the committed place's tolerance to a planted aim error (M-F4), in TASK-076 K0's
    rule form;
  - **H-read, the oracle-dynamics readout arm:** H-final(commit)'s cloned-state look-ahead in
    which, at every iteration, plate(r) is not read from simulator truth but by the pooled
    readout from the onboard frame rendered at r in the cloned roll-out. It is W with perfect
    dynamics and the real readout: the ceiling W can reach on that latent. Its readout is fitted on
    the declared-aim corpus only, whose roots are disjoint from the paired resets.
- **The readout gates, which resolves the inconsistency R14.10 recorded.** A preregistration
  needs H-read within half of the protocol's non-inferiority margin δ = 4/32 of the ceiling
  (ceiling − H-read ≤ 2/32, paired, a point reading), leaving the other half for the predictor's
  roll-out error. **The 2/32 is an allocation of δ, not a bar calibrated from a measured
  ceiling**; H-read itself is the measured ceiling it is applied to. δ is measured in the protocol
  against the best hand-written arm (C1: H-rule and H-sysid at 30/32), not against the ceiling, so
  the allocation is conservative when that arm sits below the ceiling.
- **Order of latents, declared now:** 4 × 4 first (TASK-066's gated latent). 8 × 8 is measured
  only if 4 × 4 misses, and choosing it commits the protocol to gating its own dynamics at 8 × 8
  (TASK-066's gates at h = r − 405, no reach-back over TASK-066).
- **More data first:** the declared-aim corpus is 1 024 roots (C1's was 256), and the learning
  curve is reported at 1/4, 1/2, 3/4 and all of it.

### 5.3 The feasibility record (CPU, no world model; before any protocol)

Code: C1's runner and TASK-076's worker, hook and arms, plus the move hook, a ρ loop, a
planted-error mode for H-final(commit) and H-read. Its seed block, salts and debug range are
declared before any seed is simulated and checked against every ref; R15.9's block 58000–58999
serves this ruling's arithmetic only.

| check | what is measured | bar and its source |
|---|---|---|
| M-F0 (done, §3.3) | projected twin counts from reset values | headroom over each scene-blind twin ≥ +8/32 under the strict reading at some ρ in the grid (R15.2); projected to pass from ρ = 4 cm; ρ = 3 cm (disc +11) and the −y half-disc at 3 cm (+9.5) are marginal |
| M-F1: ρ\* and the ceiling | H-final(commit) on 32 resets at each ρ, smallest first | ≥ 30/32, refused and blocked moves counted as failures (R9.6's ceiling bar); ρ\* by the monotone rule above |
| M-F2: motion, r and reach at ρ\* | C1-F1's and C1-F2's measurements at ρ\* | remaining motion ≥ 2 cm; r recorded; a_lo by R14.4's rule |
| M-F3: the twins lose at ρ\* | H-now, N-proxy, shuf-proxy and mean-proxy (p̄ = the move distribution's mean plate) on the 32 resets | **the strict reading** (R15.7): each proxy's paired headroom interval lies at or above +8/32 (lower bound ≥ +8); each scene-blind proxy's McNemar feasibility for the 64-reset test ≥ 0.8 (R8.12's form) |
| M-F4: τ_commit | H-final(commit)'s converged aim plus a planted error of 0, 0.5, 1, 1.5, 2 or 3 cm in a per-reset uniform direction, 32 resets each | τ_commit = the largest level such that every level up to it reaches ≥ 28/32 (TASK-076 K0's rule); undefined if level 0 misses |
| M-F5a: the readout offline | 1 024-root declared-aim corpus at ρ\*; cross-fitted pooled readout at r; plate-hidden renders; learning curve | plate-hidden lower bound > τ_commit (O4's form, against the measured tolerance); the median is reported |
| M-F5b: H-read | H-read on the 32 resets, 4 × 4 (8 × 8 only if 4 × 4 misses) | paired ceiling − H-read ≤ 2/32 (point reading; an allocation of δ, not calibrated) |
| M-F6: comparators | H-rule, H-sysid on the reading and on the true plate; H-now-reaim; H-rule-stale (H-rule on the pre-move reading) | reported, no bar; they set the non-inferiority comparator |
| M-F7: cost | H-final(commit) and H-read attempts | ≤ 60 s each, or the cap is reviewed (TASK-076's rule) |

Machine time, *estimate* from C1's run (910 s on 6 workers for about 1 000 attempts): about 4 × 32
ceiling attempts, 6 × 32 planted-error attempts, the proxies and comparators (about 350) and 1 024
corpus roots, so roughly 30–45 min of CPU, plus H-read's renders and readouts. No GPU.

### 5.4 Rows, in order (R15.8)

1. M-F1 has no passing ρ in either family, or M-F2 fails → **M-INFEASIBLE**: escalate, no clause.
2. M-F3: a scene-blind proxy's headroom is *detectably below* +8/32 (paired upper bound < +8) →
   **M-TWINS-NONE: the clause fires** (§5.5). Any other M-F3 failure (near, or feasibility < 0.8)
   → **M-TWINS-ESCALATE**: escalate, no clause.
3. M-F4: τ_commit undefined → **M-NO-TAU**: escalate, no clause.
4. M-F5a: the plate-hidden check fails → **M-ARM-KEYED**: escalate, no clause.
5. M-F5b: H-read misses on 4 × 4 and on 8 × 8 → if on both grids the paired interval of
   ceiling − H-read lies entirely above δ = 4/32 (lower bound > 4/32), **M-READ-NONE: the clause
   fires**; otherwise **M-NO-BAR**: escalate, no clause.
6. Otherwise → **M-PROCEED**: a preregistration may be drafted under R9.8 and R9.9, with K0
   (τ_commit and the ceiling re-measured on fresh seeds), Stage O (c_plate at r on the chosen grid),
   a `check_budget`-passing training block, the offline horizon and dynamics gates at h = r − 405
   (H-GATE-FAIL, no clause), a 16-reset development closed loop with a stop rule, and the gated
   cohort with R9.8's two claims.

### 5.5 The abandonment clause and its scope (R15.8)

**At the feasibility stage, it fires on M-TWINS-NONE or M-READ-NONE only.**

- **M-TWINS-NONE** closes "LeWM selection of a single committed place aim under cell A's
  reactive-plate rule (κ = −0.5, L = 2, s0 = 405, s1 = 525) on v2, with a post-pick plate move drawn
  uniformly over a disc or −y half-disc of radius ≤ ρ\* (C1 and C1-M)". It does not close other
  move distributions (a ring has more spread at the same radius), radii above ρ\* (shown
  ceiling-infeasible for e9's place, not refuted), or other κ, L or commit steps (§2: their
  tolerance curves are unmeasured).
- **M-READ-NONE** closes "LeWM place-target prediction on v2 read from onboard 112 px frozen
  DINOv2 tokens pooled to 4 × 4 or 8 × 8 at the read step, under C1-M". The next step is then a
  perception change (a view or a resolution, TASK-075's views), which by TASK-057's §7 logic is a
  data or hardware change, not another model.

**After admission, the protocol carries the note's clause** (§4.1: L-NO-GAIN, or W detectably
inferior to the best single-commit non-world-model arm by more than δ), with its scope extended by
"with the declared post-pick move of radius ρ\*". H-GATE-FAIL, L-NEAR, NO-BAR, a failed ceiling
and a budget escalation stay escalations without a clause.

**Nothing here closes** the LeWM backend, v2, the product goal, TASK-076's results or C2.

### 5.6 GPU and cost after admission

None for the ruling or the feasibility record. After M-PROCEED, the design note's sizing applies
at 4 × 4 (about 6.5 h of GPU, 10–11 h worst case at a `check_budget`-passing cap of 120 000
updates), plus featurising a corpus about four times larger than TASK-074's (*estimate*: about
1 h). An 8 × 8 latent has four times the tokens; its cost is not measured and its calibration
run must be, before its budget block is set. Every GPU job runs through `scripts/gpu_run.sh --wait`,
per job.

### 5.7 What a pass would and would not show

- **It would show** (the primary claim): "LeWM-driven aim selection at 405, followed by e9's place,
  reaches the calibrated bar, is non-inferior to the best hand-written arm and beats its
  action-blind, scene-blind and random twins, under a declared reactive-plate rule with a
  post-pick plate move on v2". That would be the first LeWM-driven closed-loop success on v2, and
  it would change R7's sentence only through its own reviewed ruling.
- **It would not show** that LeWM is needed (H-rule and H-sysid capture the rule; stated in
  advance), a LeWM policy, anything about v1's 0/150, Arena or the real G1.

### 5.8 Risks, stated now

- **The readout may still bind.** §3.4's crude map puts the 4 × 4 readout's cost at about 3–4/32,
  above the 2/32 that H-read allows. If H-read confirms it on both grids, the line ends in
  M-NO-BAR or M-READ-NONE before any GPU, which is the purpose of the check.
- **A small ρ\* gives no decision.** If the ceiling holds only at 3 cm, the projected disc
  headroom (+11) minus the expected interval half-width (about 5) is below +8, so M-TWINS-ESCALATE
  is the expected row.
- **τ_commit may be set by noise.** §3.2's plateau (0.86–0.90, that is 27.5–28.8 of 32) sits at
  K0's 28/32 bar, so τ_commit may land anywhere from about 0.5 to 2 cm by chance, and M-F5a's
  plate-hidden check moves with it (the 8 × 8 and full-token lower bounds are 2.62 and 2.19 cm,
  §3.4).
- **The ceiling under the move is unmeasured.** P-3 continues from 300 to 405 towards its stale
  plate estimate, so h at 405 shifts; e9's committed place from there is untested at 3–6 cm in
  every direction.
- **τ_commit may be wider than τ_re**, as §3.2 suggests. That helps every arm, the twins included;
  M-F3 measures the net effect on the paired resets.
- **The horizon** (r − 405, 55 steps in C1) stays the main risk to the primary claim; its gate is
  in the protocol.

## 6. The rulings (summary; full text in DECISIONS 2026-10-04 (d))

- **R15.1** form: a DRAFT ruling; no task number or MC task until M-PROCEED (R9.1 continues).
- **R15.2** the common cause (the scene-blind miss is p̄ − p for any κ, L or commit step) and the
  Stage −1 rule.
- **R15.3** the options and why O2–O7 are not taken now.
- **R15.4** the recommendation: O1, C1-M.
- **R15.5** the move: a disc over TASK-073's |d| grid, ρ\* by the ceiling only, the −y half-disc
  once as the declared second family.
- **R15.6** the readout: the 0.5 cm bar withdrawn; τ_commit and H-read; the readout gates; 4 × 4
  then 8 × 8; 1 024 roots.
- **R15.7** M-F3's reading is the strict one (it resolves R14.5's ambiguity for this record).
- **R15.8** the rows and the clause, with their scopes.
- **R15.9** seeds and salts of §3; the feasibility record declares its own.
- **R15.10** R7 unchanged; the wording nit in R14.2 fixed.

## 7. Next steps (none started by this ruling)

1. An independent review of this ruling (docs and the development script only).
2. On approval: the C1-M feasibility record (§5.3), with its declarations committed before any
   seed of its block is simulated, and its own record document.
3. Only on M-PROCEED: a preregistration, its own K0 and an independent review, then the GPU stages.
4. Nothing in TASK-076's results, the C1 record or the design note is rewritten; this ruling
   supersedes the note's C1-F5 bar (τ_re/2) for C1-M only.
