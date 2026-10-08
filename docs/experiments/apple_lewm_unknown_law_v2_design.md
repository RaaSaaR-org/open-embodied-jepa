# A plate law no hand-written arm is given: the next LeWM task on apple-to-plate-v2 (TASK-082, DRAFT design note)

**Status: DRAFT design note. It is not a protocol and nothing in it is frozen.** No bar below is
gated. It carries one development feasibility check (§4), run on development seeds only
(72000–72031 and 72100–72355, plus debug seeds 74900–74925 for mechanics), CPU only, with **no
world model**: every arm in it is privileged (the look-ahead ceiling, H-now, the corpus collector),
hand-written (H-rule) or a system-identification fit on simulator truth (H-sysid, H-sysid-krr).
No seed of any range used or reserved by TASK-073 to TASK-081 was simulated, and no model was
trained.

Every choice in it is **decided by Claude under owner delegation (2026-09-30)**. The rulings are
R19.24–R19.25 in [DECISIONS.md](../DECISIONS.md), decision 2026-10-08 (k):

- **R19.24** pursues R19.23's direction (1) (R18.32's (a)): a condition in which no hand-written
  arm is given the plate law, declared as a task change and drafted first as this design note;
  it opens the card TASK-082 (todo) and takes the seed block 72000–74999 and the salts 8401–8412
  for it;
- **R19.25** records §4's development numbers and the recommendation of §7 (a recommendation, not
  a ruling on any preregistration).

Any task the note leads to needs, before a single cohort seed is simulated:
- its own preregistration;
- an independent review;
- fresh seeds.

The canonical status sentence (DECISIONS 2026-10-02, R7, as changed by R19.21) is unchanged by
this note. It is quoted in full in DECISIONS.md (R19.21). Its v2 LeWM part says that TASK-081's
W reached 118/128 under C1-M (L-PASS, the primary claim "LeWM-driven closed-loop success") and
that "LeWM needed" is not shown, since the non-learned H-rule, which is given the simulator's plate
law, scored measurably higher (125/128). **Nothing here is a result.** Every development number
below is one run, on 32 check resets (256 corpus roots), in simulation only, with arms that read
the true plate. These are not gated counts and not project-learned results.

---

## 1. Where we start

TASK-081 closed **L-PASS** at Stage S
([results](apple_lewm_commit_precision_v2_results.md) §2; R19.20–R19.23). On 128 fresh gated
resets under C1-M: W 118/128, H-rule 125/128, H-sysid 121/128, the privileged ceiling
H-final(commit) 126/128; W − H-rule −7/128, 95 % [−13, −2]. The secondary claim "LeWM needed"
(R9.8: LeWM detectably better than the best hand-written arm) was not shown, and H-rule was
measurably better.

**Why "LeWM needed" cannot be tested under C1-M.** C1-M's plate law is cell A's linear rule
(`plate_twin_v2.palm_driven_xy`): after the commit step 405 the plate moves by
κ (palm(t − L) − palm(405)), κ = −0.5, L = 2, until s1 = 525. H-rule is handed that law and
solves its fixed point on the kinematic stand-in; H-sysid fits a 7-coefficient linear model of the
plate on (plate, palm, aim), which contains the law exactly. A learned world model can at best
tie arms that already hold the law. R19.23 (1) therefore asks for a condition in which **no
hand-written arm is given the law**, declared as a task change.

## 2. What "LeWM needed" can honestly mean here

Three comparisons are possible. They differ in what the comparator is allowed to know.

| tier | comparator | what it is given | status in R9.8 |
|---|---|---|---|
| hand-written | **H-rule** with a written law that is not the simulator's (C1-M's κ = −0.5 is the natural prior an engineer would write); **H-sysid**, TASK-080's linear fit on the corpus | a closed form, or a fixed low-capacity form fitted on the corpus | R9.8's secondary claim, "LeWM needed": W detectably better than the best of these |
| learned, non-LeWM | **H-sysid-krr / -mlp**: a flexible regression of plate(s1) on (p̂, h, g), fitted on the **same corpus** as W, inverted with W's controller form; **P-aim**: a model-free regression from (p̂, h) to the privileged ceiling's aim on the same corpus resets (behaviour cloning, as P-3 is) | the same corpus and the same run-time inputs as W, but a hand-chosen low-dimensional state (R-plate's reading p̂, the palm h, the aim g) | not in R9.8; proposed here as a **reported** tier |
| privileged | H-final(commit), H-now | the simulator | the ceiling and the "plate must be predicted" check |

**The honest test of necessity is the second tier, and this task family cannot pass it.** The
decision is a single committed aim. Under any plate law whose outcome is a deterministic function
of the plate at 405, the palm at 405 and the aim (a static law, or a history-dependent one driven
by e9's deterministic primitive), a flexible regressor on (p̂, h, g) fitted on the same corpus is a
forward model of the same quantity with the same labels. It should match W in the limit, and the
development check shows that 256 roots already suffice for the candidate laws (§4: H-sysid-krr
32/32 under U-sat, aim error 0.072 cm in median). LeWM can only be *needed* against such a
baseline where the decision-relevant information is in the pixels and not in (p̂, h): a
consequence that no low-dimensional table of (reading, action) captures, as the first design note
already said ([apple_lewm_next_v2_design.md](apple_lewm_next_v2_design.md) §4.2). Examples:
a per-reset law parameter set by a visible scene property that no hand-written feature names, a
parameter revealed only by the plate's motion before the decision (which needs W to take history),
several interacting objects, or an articulated or deformable object. None is designed here (§3.4).

So the attainable statements under a withheld law of this family are:
1. **"LeWM-driven closed-loop success"** (R9.8 primary), unchanged in form;
2. **"LeWM needed" in R9.8's sense**, now testable for the first time, because no hand-written arm
   holds the law: W detectably better than the best of H-rule (wrong written law) and H-sysid
   (linear);
3. **"LeWM matches a learned non-LeWM baseline"**: W non-inferior to H-sysid-krr/-mlp and P-aim
   within a declared margin, reported only. "LeWM needed" against this tier is **not expected**
   and must be said so in advance.

A pass on 2 would say that a *learned* model is needed under this law and that LeWM is one that
suffices from pixels. It would not say that LeWM is needed rather than any learned model.

## 3. Candidate conditions

Every candidate keeps C1-M except the law after 405: P-3's pick, the declared post-pick plate move
at step 300 (disc, ρ = 4 cm), the single aim committed at 405, e9's place primitive, s1 = 525,
L = 2, the scorer. The plate's displacement from its position at 405 is a function F of the
executed palm path d(u) = palm(u) − palm(405), u = t − L. The parameters below were fixed in
`src/embodied_jepa/plate_law_dev.py` from TASK-077's recorded plate motion (about 8.4 cm from 405
to r, so a palm travel of about 17 cm at κ = −0.5) **before any seed of the development block was
simulated**.

### 3.1 U-sat: a saturating, swirling static law (checked)

F(d) = −A tanh(|d|/D) R(Θ |d|/D) d/|d|, with A = 8 cm, D = 12 cm, Θ = 0.25 rad. The plate comes
toward the hand, as under C1-M, but its travel saturates at A and it turns sideways by an angle
that grows with the palm's travel (about 20° and 7.1 cm at a 17 cm travel; 4.6 cm sideways at
30 cm). Memoryless and smooth; by arithmetic its Lipschitz constant stays below 1 at the box's
travels, so the fixed point should be unique, and the look-ahead converged on 32/32 resets (§4).
Against it, H-rule's written C1-M law aims 1.15 cm from the ceiling's aim in median (§4.2).

### 3.2 U-play: a hysteretic play operator (checked)

The drive z(u) = κ d(u), κ = −0.6, passes through a two-dimensional play (backlash) of width
w = 3 cm: the plate stays put until |z − y| > w, then is dragged so that |z − y| = w. The final
plate depends on the path, not only on its end. Along e9's nearly monotone path the result is
close to an affine law (|y| ≈ 0.6 |d| − 3 cm), so H-rule's written law is off by only about 1 cm
(1.10 cm in median, §4.2).

### 3.3 Not checked, recorded for completeness

- **U-speed**, a speed-gated drag (the plate moves only while the palm is faster than v0). Its
  outcome depends on e9's speed profile; v0 would have to be set from palm speeds no record logs.
- **A physical mechanism** (the plate on damped slide joints driven through a spring or a
  frictional contact in MuJoCo). It has no closed form at all, but it changes the scene model, so
  it is a v2 variant, not a condition, and needs its own expert gate as TASK-070 did.

### 3.4 Conditions that could test necessity against learned baselines (not designed)

- **U-cue**: a law parameter (for example the sign of Θ) set per reset by a visible property of
  the scene that the low-dimensional state does not name (a plate colour via the opt-in
  `plate_color.py`). A learned baseline given only (p̂, h, g) fails half the resets; one given an
  image feature does not, so the comparator set must be argued in advance. R-plate and the readouts
  would need refits (the white-plate record saw colour shift the plate-hidden check).
- **U-hist**: a per-reset gain that the plate reveals only by moving before 405. W's recipe takes
  one frame (history one); this needs a recipe change and is close to a new world-model study.

Both are later options; neither is checked here.

## 4. Development feasibility check (development seeds, CPU; not gated)

### 4.1 The run

- **Code** (commit `ae680ca`, made before the record run; reviewed with this note):
  `src/embodied_jepa/plate_law_dev.py` (the laws, seeds, salts, the move draw, the kernel ridge,
  the estimators; NumPy only), `src/embodied_jepa/plate_law_dev_runtime.py` (C1-M's worker with
  the hook's law replaced; every arm's code reused unchanged), `scripts/dev_plate_law_feasibility.py`
  and `tests/test_plate_law_dev.py`. No TASK-073–081 file is edited.
- **Laws:** `c1m` (C1-M's own rule, the within-run reference), `sat` (U-sat), `play` (U-play).
- **Seeds (R19.24):** check cohort F = 72000–72031 (every arm under every law, the same 32 resets
  in the same order); corpus = 72100–72355 (256 roots, the same roots and aims under every law);
  debug 74900–74999 (mechanics only). The block 72000–74999 lies outside every forbidden range of
  TASK-081's list and outside TASK-081's block 70000–71999 (`plate_law_dev.check_seed_ranges`,
  tested). **Salts:** 8401 the move draw (C1-M's disc rule at ρ = 4 cm, with a fresh salt), 8402
  the corpus's uniform box aims, 8403 every bootstrap, 8404 the sysid folds; 8405–8412 are reserved
  by R19.24 for a protocol (not named in the development code).
- **Arms** (every one reads the **true plate**; each count is an optimistic bound for an arm that
  reads the plate from the image):
  - **H-final(commit)**, the privileged look-ahead ceiling (TASK-076's `LookaheadAim`, unchanged;
    branches run the same law);
  - **H-now**, the true plate at 405 (privileged);
  - **H-rule-true**, the hand-written rule handed **C1-M's** law (κ = −0.5) and the true plate;
    under U-sat and U-play that law is the wrong one;
  - **H-sysid-true**, TASK-080's linear form fitted on the 256-root corpus under the same law;
  - **H-sysid-krr-true**, an RBF kernel ridge of plate(s1) − p on standardised (p, h, g), length and
    λ chosen by 5-fold CV on the corpus, inverted with the same controller form
    (`lewm_next_c1.choose_aim`: grid, refinement, clip to the box).
- **Run:** one invocation, 2026-10-08, at `ae680ca`, tracked tree clean, CPU only (6 workers, one
  torch thread each), 1 609 s; G-repro and the cohort estimates passed; no render disagreement.
  Report `outputs/task082-dev-1/report.json` (sha256
  `2b1dd16b205f91dab9fca600d2ffb1101691b472e49bc3a38883322fa336b58a`), git-ignored, with an
  evidence copy in `~/develop/emai/evidence/task082-dev/` (manifest
  `_checksums/task082-dev.sha256`, sha256 `9677ff30…1cef`).
- **Disclosed:** two debug runs on debug seeds came first (the first stopped on a missing report
  key before any attempt; the second, `task082-dev-debug-2`, ran 4 + 16 debug seeds per law). Their
  counts were seen. **No law parameter, arm or seed was changed after them**, and none after the
  record run.

### 4.2 Numbers

Counted successes on F (n = 32, exact 95 % intervals); each arm's paired difference from the
ceiling (ceiling − arm, reset bootstrap, salt 8403); aim error = |aim − H-final(commit)'s aim| on
the same reset (median with bootstrap interval, 87.5th percentile, maximum).

**U-sat**

| arm | successes | exact 95 % | ceiling − arm [95 %] | aim error median [95 %] / p87.5 / max (cm) |
|---|---:|---|---|---|
| H-final(commit) (privileged) | **32** | 0.891–1.000 | – | – (look-ahead converged 32/32) |
| H-now (privileged) | 2 | 0.008–0.208 | +30 [+27, +32] | 6.30 [6.17, 6.48] / 7.19 / 7.40 |
| H-rule-true (C1-M's law: wrong) | **23** | 0.533–0.863 | **+9 [+4, +14]** | 1.15 [1.01, 1.33] / 2.45 / 3.03 |
| H-sysid-true (linear) | **28** | 0.710–0.965 | **+4 [+1, +8]** | 0.36 [0.28, 0.43] / 0.72 / 1.05 |
| H-sysid-krr-true (learned, flexible) | **32** | 0.891–1.000 | 0 [0, 0] | 0.072 [0.056, 0.083] / 0.15 / 0.27 |

**U-play**

| arm | successes | exact 95 % | ceiling − arm [95 %] | aim error median [95 %] / p87.5 / max (cm) |
|---|---:|---|---|---|
| H-final(commit) | **32** | 0.891–1.000 | – | – (converged 32/32) |
| H-now | 2 | 0.008–0.208 | +30 [+27, +32] | 5.31 [5.09, 5.67] / 7.66 / 8.45 |
| H-rule-true (wrong law) | 31 | 0.838–0.999 | +1 [0, +3] | 1.10 [1.06, 1.14] / 1.25 / 1.32 |
| H-sysid-true | 31 | 0.838–0.999 | +1 [0, +3] | 0.50 [0.30, 0.63] / 0.73 / 1.18 |
| H-sysid-krr-true | 30 | 0.792–0.992 | +2 [0, +5] | 0.16 [0.11, 0.19] / 0.24 / 0.51 |

**C1-M's own law (reference)**

| arm | successes | ceiling − arm [95 %] | aim error median / p87.5 / max (cm) |
|---|---:|---|---|
| H-final(commit) | 30 | – | – (converged 32/32) |
| H-now | 0 | +30 [+27, +32] | 6.28 / 8.26 / 8.97 |
| H-rule-true (the right law) | 32 | −2 [−5, 0] | 0.060 / 0.067 / 0.103 |
| H-sysid-true | 30 | 0 [−3, +3] | 0.46 / 0.95 / 1.41 |
| H-sysid-krr-true | 32 | −2 [−5, 0] | 0.13 / 0.19 / 0.31 |

**The corpus** (255 of 256 roots kept under each law; root 72144 ended in a guard refusal before
the commit step under every law, as reset 70302 did in TASK-081): the
collector's uniform box aims succeeded on 38/256 (U-sat), 53/256 (U-play) and 38/256 (C1-M). The
plate moved 7.17 cm (U-sat), 7.63 cm (U-play) and 8.71 cm (C1-M) from 405 to s1 in median over the
corpus; the palm travelled 18.92 cm in median. Cross-fitted offline errors of plate(s1), median
[95 %] / 87.5th percentile in cm:

| law | linear sysid | kernel ridge | C1-M's rule, open form p + κ (g − h) |
|---|---|---|---|
| U-sat | 0.325 [0.299, 0.358] / 0.59 | 0.122 [0.115, 0.130] / 0.26 | 3.87 [3.37, 4.54] / 8.19 |
| U-play | 0.651 [0.557, 0.728] / 1.23 | 0.240 [0.216, 0.267] / 0.51 | 1.49 [1.27, 2.11] / 4.97 |
| C1-M | 0.545 [0.472, 0.589] / 1.05 | 0.201 [0.181, 0.238] / 0.43 | 0.71 [0.71, 0.84] / 3.82 |

(The open form assumes the palm ends at the aim; H-rule itself uses the stand-in's palm path,
which is why it is exact under C1-M in closed loop while the open form is not.)

### 4.3 Reading

- **Both laws admit the privileged ceiling** (32/32 each, ≥ 30/32), the look-ahead converged on
  every reset, and the plate still has to be predicted (H-now 2/32 under each).
- **U-sat degrades the hand-written arms materially; U-play does not.** Under U-sat, H-rule with
  the written C1-M law falls to 23/32 (+9 [+4, +14] below the ceiling) and the linear H-sysid to
  28/32 (+4 [+1, +8]); under U-play both stay at 31/32, inside the noise. U-play's hysteresis is
  nearly affine along e9's monotone path, as §3.2 expected.
- **A learned low-dimensional baseline recovers the ceiling under U-sat from 256 roots**
  (H-sysid-krr 32/32, aim error 0.072 cm in median on the true plate; for scale, W's was 0.295 cm
  under C1-M in TASK-081 on R-plate's reading, so the two are not like for like). This is the measured form of §2's argument: under these laws a flexible regression on
  (p, h, g) is enough, so "LeWM needed" against learned baselines is not testable here.
- **The linear sysid's degradation under U-sat is real but thin.** Its offline error (0.325 cm) is
  even below its C1-M value (0.545 cm); its closed-loop loss comes from the tail (87.5th
  percentile aim error 0.72 cm, maximum 1.05 cm, against τ_commit = 1.0 cm in C1-M). With R-plate's
  reading instead of the true plate every arm here loses some resets (TASK-081: H-sysid 121/128
  on the reading against H-final(commit) 126/128 under C1-M).
- **Caveats.** One run, 32 resets, true-plate arms, one parameter set per law, chosen by arithmetic
  before the run but not calibrated. The C1-M ceiling scored 30/32 while H-rule-true scored 32/32:
  near the ceiling the scorer's 4 cm radius adds noise to every count (TASK-081 results §3.2), so
  counts of 30–32 of 32 do not order arms.

## 5. What a TASK-082 under U-sat would need

### 5.1 What must be retrained or refitted

- **The corpus.** W-66800 and every TASK-077 checkpoint learned C1-M's law from a corpus under
  C1-M. A new corpus under U-sat is needed, collected as TASK-077's Stage C (2 000 roots, the same
  box draw, splits fixed before collection, fresh seeds in 72000–74999).
- **W and N.** TASK-077's recipe unchanged (8 × 8 pooled frozen DINOv2 tokens of the onboard 112 px
  frame, horizon 60, the selection rule and caps), retrained on the new corpus. Three seeds per arm
  as TASK-077 (about 31.4 h on the GPU queue, the measured sum of TASK-077's six jobs, 113 131 s),
  or one seed pair (about 10 h). The calibration jobs need not be repeated if the recipe and corpus
  size are unchanged (to be argued in the preregistration).
- **The readouts.** R-plate (the plate at 405) is unchanged: the plate is static from 300 to 405
  and its appearance does not change, so TASK-076's R-plate and its offline error carry. R-S (the
  ridge on W's own predicted latents under the stand-in, TASK-080's Stage R) must be refitted on
  the new corpus, and its fresh-root gate re-run, because the plate's positions at r now include
  the swirl's sideways component (the ceiling's aim lay up to 2.1 cm sideways in §4, more off the
  fixed point) that C1-M's corpus never showed. r itself (the step after which the plate is within 0.1 cm of plate(s1)) and
  τ_commit must be re-measured in a K0 under U-sat; the saturation changes the plate's speed
  profile.
- **The solver** stays affine_local (TASK-081), whose effect was not shown but which is the frozen
  form; no further solver work (R19.23 (4)).
- **The comparators.** H-rule keeps C1-M's law as the written prior (declared as such); H-sysid is
  fitted on the new corpus's train split with R-plate's reading; the learned tier (H-sysid-krr or a
  small MLP, and P-aim) is fitted on the same split. P-aim needs the privileged ceiling's aims on
  the corpus resets (one H-final(commit) attempt per root).

### 5.2 Compute estimate (estimates, from TASK-077/080/081's measured times and §4)

| stage | estimate |
|---|---|
| Stage 0 (code, tests, debug smokes, G-NI and secondary power simulations) | CPU, minutes per smoke |
| K0 under U-sat (τ_commit, r, ceiling, proxies on 32 resets) | about 10–15 min CPU (TASK-077: 9.7 min) |
| C, 2 000 roots | about 30 min CPU (TASK-077: 20–23 min; §4's corpus ran at 0.84–0.94 s of wall time per root on 6 workers) |
| P-aim's labels (H-final(commit) on the train roots) | about 1 h CPU (10–12 s per attempt on 6 workers) |
| O, featurisation | 10–30 min GPU |
| T, W and N × 3 seeds | **about 31 h on the GPU queue** (one seed pair: about 10 h) |
| R (R-S, R-N refits, fresh-root gate) | 1–3 h CPU |
| D (16 resets) and S (128 resets, about 13 arms; non-primary seeds' W in closed loop adds about 15 min) | about 20 min and 1–1.5 h CPU |
| **total** | **about 31 h GPU (or 10 h) and 4–7 h CPU**, about three days of wall time with the reviews |

## 6. The other R19.23 options, in parallel

- **(2) TASK-079's precondition.** Whether TASK-081's L-PASS counts as "TASK-077 L-PASS (or its
  equivalent row)" is an **owner ruling** (R19.10, R19.22); it costs no compute. It does not by
  itself start TASK-079: TASK-078's pass (Isaac containers, about 1–3 h of GPU over a few runs,
  after #123's review and F14), mirrored Arena assets and an owner ruling on Arena as a gated
  benchmark stay required, and TASK-079 itself needs LeWM retraining or fine-tuning on Arena frames
  (to be estimated from TASK-077's times; of the order of the 31 h above). Independent of TASK-082.
- **(3) Robustness of TASK-081's result under C1-M.**
  - *More model seeds.* W-66801 and W-66802 exist (TASK-077 Stage T) and TASK-080's Stage R fitted
    R-S for both seeds, so a closed-loop run of each needs no training: a preregistration, fresh
    seeds, and about 128 resets × (W plus H-rule and the ceiling) per seed, about 20–30 min of CPU
    per seed. It speaks to seed 66800's `last_two_triggered` flag. Cheapest of the options.
  - *A random-init floor for W.* Featurise the corpus with the random-init encoder (minutes of
    GPU), train W (and N) on it (about 5 h per job, so about 10 h for one seed pair), refit R-S on
    its predictions (CPU, hours), then a closed loop (about 1 h CPU). It asks whether DINOv2
    pretraining matters for W, which no run has asked.
  - *Broader conditions* (other move radii, κ, commit steps): each is its own K0 and, for a
    changed κ, a retrained W; costs as §5.2.
  All three can run beside TASK-082; the GPU queue (one job at a time under the shared lock) is the
  shared constraint, so (3)'s seed runs (CPU only) fit anywhere, while the floor and TASK-082's
  Stage T compete for the queue.

## 7. Recommendation (R19.25; for the owner)

1. **Preregister TASK-082 under U-sat**, declared as a task change (C1-M with the plate law after
   405 replaced by U-sat's, parameters as in §3.1, withheld from every non-privileged arm). U-play
   is not recommended: it leaves the hand-written arms at 31/32.
2. **Claims, stated in advance:**
   - primary: R9.8's "LeWM-driven closed-loop success", with G-NI against the better of H-rule
     (written C1-M law) and H-sysid (linear), both on R-plate's reading, as in TASK-081;
   - secondary, reported only: R9.8's "LeWM needed" (W detectably better than that best
     hand-written arm). The development check gives it room under U-sat (ceiling − H-sysid +4/32
     [+1, +8] on the true plate), but Stage 0 must simulate its power at plausible rates before the
     freeze; it is a live chance, not an expectation;
   - a new reported tier: W against the learned non-LeWM baselines (H-sysid-krr or -mlp, P-aim) on
     the same corpus, as non-inferiority only, with the statement that **"LeWM needed" against
     learned baselines is not testable under this law** (§2, §4.3: H-sysid-krr 32/32 from 256 roots).
3. **Retrain TASK-077's recipe unchanged on a new 2 000-root corpus**, three seeds per arm (about
   31 h of GPU), with seed 66800's role (primary, gating) taken by a declared new primary seed and
   the other two run in closed loop as reported arms. That also gives the seed-to-seed spread that
   R19.23 (3) asks for, under the new condition.
4. **Do not count a pass as necessity.** A pass on the secondary claim would say that a learned
   model is needed under U-sat and that LeWM suffices from pixels, not that LeWM is needed. A
   necessity study against learned baselines needs a pixel-only consequence (§3.4) and its own
   design note.
5. **In parallel:** ask the owner for the TASK-079 ruling ((2), no compute), and consider the two
   extra W seeds under C1-M ((3), CPU only, about an hour in all) as the cheapest robustness check.

## 8. Open questions for the preregistration

- **The comparator set for G-NI.** R9.8 says "the best non-world-model arm". P-aim is a learned,
  model-free arm, so by that wording it could join G-NI; its labels are privileged (as P-3's are).
  Recommended: keep G-NI on the hand-written arms (comparable with TASK-080/081) and report P-aim's
  tier, but the preregistration must argue it.
- **The written law for H-rule.** C1-M's κ = −0.5 is a declared prior, not the law. An H-rule-fit
  (C1-M's form with κ fitted on the corpus) is close to the linear H-sysid; whether to add it is a
  choice for Stage 0.
- **The learned baseline's class and data.** KRR on 256 roots sufficed here; an MLP is the more
  usual baseline. Both should be fitted on the same train split as W, with R-plate's reading at fit
  and run time.
- **τ_commit, r and the margin.** τ_commit and r must be re-measured under U-sat; δ (16/128 in
  TASK-081, an allocation) should be re-argued with a power simulation, since the ceiling and the
  hand-written arms move.
- **Does W's 60-step horizon still cover the plate's motion?** Under U-sat the plate's speed profile
  changes; K0's r decides the readout step.
- **Swirl and the box.** Under U-sat the ceiling's aim lay 0.95 cm sideways of the plate–palm line
  in median (maximum 2.1 cm; 0.25 and 0.20 cm under C1-M and U-play), inside the box's ±3 cm, but
  closer to its edge. Stage 0 should record how often the clip binds for H-sysid and W.
