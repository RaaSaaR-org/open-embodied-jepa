# TASK-094 — grasp-closure locus diagnostic: does the partial closure the world-model arms command cause their grasp failures, and does the failure sit in the goal and cost, the model or the search (measurement only; no controller)

**Status: DRAFT** (R27.160–R27.176; decided by Claude under owner delegation, 2026-10-15;
DECISIONS (bi)). Nothing here has run. A Stage 0 PR freezes this protocol at its merge (§14). The
gated stage runs after that, at the merged revision, from a clean tree.

This is the diagnostic that R27.156 chose after TASK-093 closed at B-FAIL (design note
[grasp_next_step_v1.md](grasp_next_step_v1.md), §4–§5). **It is measurement only.** It runs no
world-model closed loop, trains nothing, and preregisters no planner, cost, horizon or
action-parametrisation variant. Its scripted probes are not learned results, and no row admits a
controller.

**This is not an Apple→Plate experiment.** It measures grasp-and-lifts of the play scene's objects
with G1's right arm and right Dex3, in MuJoCo only. **R7's canonical status sentence does not
change, whatever row this task ends in** (R27.176).

## 1. What the records say

Every number below keeps the qualifiers of its record. Those of the offline read (R27.152) apply
throughout §1.1: each stage is one run, in simulation only, on its own gated cohort; pooled numbers
pool arms, model seeds, checkpoints and cohorts and are for orientation only; every link between a
measure and success is an **association, not an intervention**; and the read was chosen after the
results were known.

### 1.1 The offline read (design note §2; R27.152)

- **Contact without a hold.** The world-model arms (WM: every arm that ran TASK-089's
  step-averaged planner over a G checkpoint in TASK-089, TASK-091 and TASK-093 Stage BS; 1 856
  episodes) made grasp contact in the close phase on 982 and succeeded on 105 of those. The
  scripted IK follower (ik, the same resets and goals, 320 episodes) made contact on 302 and
  succeeded on 220 of those.
- **The closure they command.** The WM arms' median lift-phase mean commanded right grasp was
  0.53–0.73 per run, against ik's 1.00. The demonstrator's *measured* hand synergy at the close
  subgoal has a median of 0.675–0.683 per cohort (p10–p90 0.647–0.694), blocked by the object while
  it commanded +1.0.
- **Success by commanded closure, given contact** (commanded before the object rose): 21 / 739
  below 0.7, 67 / 220 at 0.7–0.9 and 17 / 23 at 0.9 or above (exact 95 % 0.52–0.90; 6 of the 23
  from one run). Reverse causation is possible, and conditioning on contact selects on a variable
  the closure may itself affect.
- **Palm placement.** The WM arms' median minimum palm distance to the close subgoal in the close
  phase was 0.89–1.61 cm per run, against ik's 0.21–0.24 cm. WM episodes placed within 0.5 cm still
  succeeded only 19 / 179, against ik's 160 / 241. No WM episode beyond 2 cm succeeded.
- **G's own cost.** In the lift phase, G's predicted cost of the chosen plan was lower in succeeding
  than in failing episodes (medians 0.58–0.97 against 1.17–1.62 per run). That is a state-level
  separation. Whether G ranks the *actions* from one contact state is not measured.
- **The corpus.** At grasp contact in `play-v1`, 46 675 of 48 050 commands (97.1 %) are ≥ 0.8;
  402 of 27 980 lifted steps have a command below 0.8.

### 1.2 The reference rates this task leans on

- **ik at Stage BS** (TASK-088's IK follower with e9's press; one run per stage, simulation only):
  48 / 64 (TASK-089), 82 / 128 (TASK-091) and 91 / 128 (TASK-093), pooled **221 / 320 = 0.69**
  (three cohorts, pooled for orientation). At Stage BD (development, 32 resets each): 28, 26 and
  21 of 32.
- **TASK-089 A-K0** (32 calibration roots, offline, simulation only): on real frames, G's cost
  family (latent + pose) separated close-vs-open, lift-vs-release and lift-vs-stay at h = 8 on
  32, 32 and 31 of 32 roots. The latent term alone separated lift-while-opening from
  lift-while-closed on 15 / 32 only. TASK-089 gated a comparison when its real-frame ceiling was
  at least 0.75.
- **TASK-089 A2** (64 gated roots, offline): with the step-averaged cost, the frozen planner's
  first command closed at R1 on 62 / 64 and lifted closed at R2 on 61 / 64.

### 1.3 Where the failure most plausibly sits (R27.153), and what is not measured

The most plausible locus is the objective at the close-and-lift transition: (b) the goal's hand
term is the demonstrator's object-blocked measured hand, which a partial close also satisfies,
together with (a) G having little data on partial closes at contact. That this causes the
failures is **inferred**. Not measured: whether a partial closure at an exact pose loses the lift
(physics), whether the cost on real frames or G ranks a squeeze above a partial close from one
contact state (the action-level ranking), and whether the frozen planner finds the squeeze when G
prefers it (search at contact states). TASK-094 measures each of these separately.

## 2. The fired clauses, and why TASK-094 engages none of them

| Clause | It closes (short form) | TASK-094 |
| --- | --- | --- |
| TASK-054 | CEM over the apple world-model cost as the *primary* control line | No control line is proposed; S runs the frozen planner offline at fixed roots, as TASK-089 A2 and TASK-093 Stage P did. |
| TASK-057 §7 | further BC formulations on `apple-wide-v1` at 112 px | No BC, no `apple-wide-v1`. |
| TASK-062 §10 | training a LeWM-family encoder on `apple-wide-v1` | No training. |
| TASK-065 §10 | the predictor on frozen pooled CLS on `apple-look-v1` | Not that predictor or corpus. |
| TASK-075 §7 | a LeWM place planner or critic under TASK-074's condition on those views | Not a place task. |
| **R26.33** (TASK-089 §9) | CEM planning of grasp-and-lift over TASK-087's checkpoints: no further planner, cost, horizon or action-parametrisation variant for grasping without new evidence of a different kind | **Not engaged.** M evaluates the frozen step-averaged cost (λ = 2.0552) unchanged; its decomposition into latent and pose terms is a reported analysis, as TASK-089 A1 reported G-lat, not a cost variant. S runs TASK-089's frozen planner unchanged, offline. No variant is preregistered for use. |
| **R27.71** (TASK-090 §9) | more scripted grasp-and-lift play of `play_policy_grasp_v0`'s kind | No play data is collected. M's branches are evaluation data at fixed roots; nothing trains on them. |
| **R27.149** (TASK-093 §8) | grasp-channel rate projection of CEM's candidates and its variants, and G-req | No projection is used. S runs the unprojected planner (M-v1's setting). The projector's arithmetic is used only to *read* a plan's executed grasp (§6), which changes nothing that is scored. |

Not fired and not engaged: TASK-092 §8 (A‴-FAIL, no clause; R27.100) and TASK-082 (CAL-ESCALATE).
The moving-plate line is stopped by the owner, and Phase 4 is not admitted (R25.27, R27.148).

**R27.154 stands:** R26.33's condition is not ruled met by this document for any variant. §7.3
states what this task's rows put forward to a later ruling.

**Simulator truth.** Object truth is read by the goal demonstrator, the root builder, the branch
outcome labels (§5.4) and the scorer only. It never enters a model input, a cost or the planner.
The IK follower reads proprioception and the goal pose only, as frozen.

## 3. Question, parts and what stays frozen

**Question.** The WM arms command a partial closure at contact. (K) Is that closure, at an exact
pose, the physical cause of lost lifts? If it is, (M, S) does the failure come from the goal and
cost (b), from G (a) or from the search (c)?

| Part | What runs | Model | Cohort |
| --- | --- | --- | --- |
| **K, the physics dose** | TASK-088's IK follower with its closing command held at a level c, ± a palm offset | none | 128 fresh gated resets |
| **M, the ranking** | scripted hold-level branches from fresh roots, simulated to their true outcome, scored by the frozen step-averaged cost on G's predictions and on the branch's real frames (the ceiling) | G-87100, G-87101, G-87102 (frozen) | 64 fresh gated root seeds, two root types |
| **S, the search** | TASK-089's frozen step-averaged CEM, one call per root | the same | the same roots as M |

**Frozen, unchanged:**
- **Checkpoints.** TASK-087's **G-87100, G-87101 and G-87102** (`checkpoints/task087-run/`),
  under their existing pins and loaders; **G-87102 carries TASK-087's `last_two_triggered` flag.**
  No training.
- **Latent and encoder.** TASK-087's frozen DINOv2 ViT-S/14 tokens of the onboard 112 px frame,
  4 × 4 pooled and projected.
- **Cost.** The step-averaged cost: the mean over h = 1 … 8 of latent MSE + λ × the standardised
  joint-position MSE of the 14 right arm and hand joints, λ = **2.0552**.
- **Planner (S only).** TASK-089's G-chg setting as TASK-091 and TASK-093's M-v1 ran it:
  `CEMPlanner` unchanged, H = 8, 300 samples, 10 iterations, 30 elites, minimum std 0.05, the left
  arm and left grasp fixed, **no candidate projection**, CEM seed `zero_shot.cem_seed(root seed)`.
- **Task.** TASK-088's scene, objects, layouts (salt 8801), grasp demonstrator and goals (salt
  8802; four subgoals), subgoal switching (1 cm, 0.17 rad, 0.15 rad RMS for 3 commands, or 100
  commands), the 400-command budget, execution through the embodiment's projection, the 5 rad/s
  velocity stop and the scorer (target ≥ 5 cm above its start height and in grasp contact for 20
  consecutive commands; no latching).

`planning.py`, `embodiment.py`, `zero_shot*.py` and every TASK-087–TASK-093 module are imported,
never edited (§14).

## 4. Part K — the physics dose (scripted; no model)

### 4.1 Arms

Every K arm is TASK-088's `IKController` (e9's tracking law to the current subgoal's palm pose;
on the grasp subgoal the palm commanded 5 cm below the goal palm, e9's press; the grasp ramped by
0.2 per command), with one change of its grasp target: where the frozen follower wants +1 (the
subgoal's hand synergy is above 0, that is on the grasp and lift subgoals), the K arm wants **c**.
Elsewhere it wants −1, as frozen. All arms run on the **same 128 resets and goals** (paired).

| Arm | Held closing command c | Palm target | Role |
| --- | --- | --- | --- |
| **K-1.0** | +1.0 (ik as frozen) | the goal's | **reference; its rate is the K-VOID check** |
| K-0.9, K-0.8, K-0.7 | 0.9, 0.8, 0.7 | the goal's | the dose, reported |
| **K-s\*** | s\* | the goal's | **the contrast that sets K's row** |
| K-0.6 | 0.6 | the goal's | reported (below s\*, inside the WM arms' per-run range) |
| K-open | −1 (never closes) | the goal's | degeneracy reference: the lift needs a closure |
| K-off-x-1.0, K-off-x-s\* | +1.0, s\* | displaced by **v_x** | palm offset, reported beside every row |
| K-off-y-1.0, K-off-y-s\* | +1.0, s\* | displaced by **v_y** | palm offset, reported beside every row |

- **s\*** is, per reset, the right-hand synergy (`zero_shot.synergy_grasp`) of the goal's grasp
  subgoal hand joints: the demonstrator's measured, object-blocked hand at the end of its close
  (about 0.68 in earlier cohorts). It comes from the goal pose, which the follower already reads; no
  object truth enters.
- **The palm offset.** v_x = (+1.2, 0, 0) cm and v_y = (0, +1.2, 0) cm in the pelvis frame (the
  frame of the subgoal palms; +x away from the robot, +y towards its left). The offset is added to
  the palm of **all four subgoals** before the run, so the follower *and* the switching rule see a
  rigidly displaced goal. The timing of the subgoal switches then stays like K-1.0's, and only the
  placement changes.
  - **Why 1.2 cm.** The WM arms' per-run medians of the minimum close-phase palm distance are
    0.89–1.61 cm, and the pooled median lies in the 1.0–1.5 cm bin (§1.1; 671 of 1 848 episodes
    below 1 cm and 1 190 below 1.5 cm). 1.2 cm is the value the design note fixes.
  - **Why two directions.** The direction of the WM arms' placement error is not logged (the run
    reports keep the palm distance only), so no single vector can be justified from the records.
    Two orthogonal horizontal directions show whether the cost of a 1.2 cm error depends on its
    direction. Both are reported only; a minimum distance understates the typical error, so the
    offset is a lower-end dose.

### 4.2 Measures

Per arm (of 128; exact 95 % intervals): grasp-and-lift successes by the frozen scorer; grasp
contact; maximum rise; the measured hand synergy over the grasp and lift subgoal phases (to show
the blocking); contact lost after a rise of at least 1 cm (slip); subgoals reached or timed out;
stopped episodes.

**Paired contrasts** against K-1.0 on the same resets: Δ_c = (k_{1.0} − k_c) / 128, with the exact
one-sided McNemar test on the discordant resets and the reset-bootstrap 95 % interval (10 000
resamples, **salt 9014**). The offset arms are also paired against their no-offset twins.

### 4.3 K's rows (first match; K-NULL and K-CAUSAL are disjoint)

Let Δ = Δ_{s\*}.

1. **K-VOID** — any of:
   - K-1.0's count lies outside **63–114 of 128**, that is outside the pooled earlier BS rate
     0.69 ± 0.20 (§1.2; §8.1 gives the false-VOID risk);
   - K-open scores **more than 3 / 128** (the scorer would not need a closure);
   - a plumbing check fails: on every K-c command of the grasp and lift subgoal phases after the
     ramp, the requested right grasp equals c (float32, bitwise) and the applied one equals it once
     the rate limit allows; the offset arms' goals are the K-1.0 goals displaced by exactly v; goal
     digests; the frozen-file, checkpoint and λ hashes; the seeds; a cohort run twice; the
     determinism re-run (K-1.0 and K-s\* on 941000–941003, bitwise); a job over its cap;
   - or the budget stop of §10.2.
2. **K-NULL** — the bootstrap 95 % upper bound of Δ is **below δ_K = 0.15**: at an exact pose,
   holding the closure at s\* loses less than 0.15 of the lifts against the +1 squeeze.
3. **K-CAUSAL** — Δ ≥ δ_K **and** "K-1.0 beats K-s\*" by the exact one-sided McNemar test at
   **p < 0.01**.
4. **K-UNCLEAR** — anything else (for example a significant Δ below 0.15 whose interval reaches
   0.15, or a Δ ≥ 0.15 that is not significant).

**Why δ_K = 0.15.** Given contact, the WM arms succeed on 10.7 % and ik on 72.8 % (§1.1), a gap of
about 0.62. A physical loss below 0.15 could explain at most about a quarter of it, so it would not
be the primary cause.

## 5. Part M — the ranking (offline; frozen G; no training)

### 5.1 Roots, and the start of the close (R27.163)

For each root seed, the demonstrator's draw is made exactly as `zero_shot.make_grasp_goal` makes it
(the same layout, yaw and acceptance; a rejected draw is replaced and counted), as TASK-089's root
builder does. Two **root types** are taken on its trajectory, each re-run from the reset with the
same commands and checked **bitwise** against the draw's recorded state there (a mismatch is a
VOID):

- **Rs, in the close:** the demonstrator's state after **8 of its 35 close commands**. Its last
  accepted grasp target is −1 + 8 × 2 / 11 = 0.4545, below every s\* recorded so far, so the hand
  has not yet been commanded past the object-blocked level. **Goal:** the grasp subgoal (the end of
  the close), the subgoal the closed-loop planner holds during the close.
- **R2, the end of the close** (TASK-089's R2; the grasp subgoal's state): the hand has squeezed
  at +1 for the rest of the close. **Goal:** the lift subgoal (the end of the hold).

**Decision: M and S use roots in the close as well as at its end** (the brief's default). The WM
arms choose their closure during the close, from an open or closing hand (§1.1: the closure
commanded before the rise is below 0.7 in 739 of 982 contact episodes). Roots at the end of the
demonstrator's full close start from a hand that has already squeezed, so they alone could not show
how the cost ranks a closure that is being chosen.

**Why 8 commands in, not TASK-089's R1 (the end of the open descent).** The embodiment limits the
executed right grasp to 2 / 11 per command (`_prepare_side`). From R1's open hand, every hold level
c ≥ 0.4545 executes **identically** for the first 8 commands, so the branches c = +1.0 and c = s\*
have bitwise the same executed commands, frames and states over the whole planning window, and
neither the cost nor G could rank them. Eight close commands in is the earliest point at which the
squeeze and s\* branches differ at **7 or more of the 8 window steps** for every s\* below 0.818
(they differ from step 2, and from step 1 when s\* is below 0.636; every recorded cohort's p10–p90
is 0.647–0.694). Seven commands in gives 6 of 8.

### 5.2 Branches

From each root, scripted branches run e9's tracking law, as the demonstrator does: the palm target
is the demonstrator's continuation from that root, and the requested grasp moves towards the
branch's level c by at most 0.2 per command from the demonstrator's current request (0.6 at Rs,
+1 at R2).

| Root | Palm targets | Branch levels c | Length |
| --- | --- | --- | --- |
| Rs | the aim for the remaining 27 close commands, then aim + 12 cm (lift 50, hold 20) | **+1.0**, 0.9, 0.8, 0.7, **s\***, and the control −1 (re-open) | 97 commands |
| R2 | aim + 12 cm (lift 50, hold 20) | **+1.0**, 0.9, 0.8, 0.7, **s\***, and the control −1 (release) | 70 commands |

- **The squeeze branch (c = +1.0) is the demonstrator's own continuation** at both root types (the
  same requests), so it lifts by construction. Its state at the end must equal the draw's recorded
  lift-subgoal state bitwise (a plumbing check), and at Rs its state after 27 commands must equal
  the grasp subgoal's.
- **Stored per branch:** the applied commands, the onboard 112 px frames and measured 28-D states at
  steps 1–8 (the planning window), and the target's rise and grasp contact at every step.
- **A branch's true outcome** is the frozen lift criterion over the branch's own steps (rise
  ≥ 5 cm and in grasp contact for 20 consecutive commands). These are labels for selecting and
  reporting; they never enter a cost.
- A branch stopped by the embodiment ends that root seed's draw; the draw is replaced (counted), as
  TASK-089 did.

### 5.3 Costs

For a branch b at root r with goal g (§5.1), the **step-averaged cost** is computed in two ways,
with the planner's own `distance` and λ = 2.0552:

- **(i) G's prediction**, per seed: G-i rolled out from the root's encoded frame and measured state
  under b's **applied** commands, the cost averaged over h = 1 … 8 (TASK-089 A1's roll-out with the
  step-averaged cost).
- **(ii) The ceiling**: the same cost on the encoded **real** frames and measured states of b at
  steps 1 … 8.

**Reported only:** the latent term and the pose term separately, and within the pose term the
seven hand joints separately (which term, if any, is flat); the per-step cost at h = 8.

### 5.4 The ranking measure

- **Discordant roots.** At a root type, a root is *discordant* when the squeeze branch lifts and
  the s\* branch does not. n_d(t) is their number at root type t (of 64).
  - The ranking is read **on discordant roots only**. Where both branches lift, neither ranking is
    wrong, and including those roots would cap any cost's win rate below what it can show.
  - **A root type is read only when n_d(t) ≥ 16.** Below that, its rows are reported, not read.
- **A win** at a discordant root: cost(squeeze) < cost(s\*), strictly; ties are not wins.
  - **Ceiling win rate** ĉ(t) = ceiling wins / n_d(t).
  - **G-i wins** per seed, out of n_d(t).
- **Positive control** (every root, not only discordant ones): the ceiling ranks the squeeze branch
  above the control branch (re-open at Rs, release at R2), which never lifts. TASK-089's A-K0 found
  similar contrasts at 1.000 on real frames (close-vs-open from R1 and lift-vs-release, the per-step
  cost at h = 8).
- **Reported only:** for every other level c, the same wins of the squeeze over c on the roots
  discordant for that pair; the lift rate of each level's branch at each root type (a second
  physics read, at the demonstrator's exact states); the term decomposition of §5.3.

### 5.5 M's bars

- **The ceiling ranks the squeeze above s\*** at t when **ĉ(t) ≥ 0.75**. This is the threshold at
  which TASK-089 counted a contrast as seen by the cost on real frames (A-K0; R26.7), declared, not
  calibrated.
- **G ranks the squeeze above s\*** at t when at least **two of the three seeds** have wins
  ≥ **⌈(0.5 + 0.75 × (ĉ(t) − 0.5)) × n_d(t)⌉**: G recovers at least three quarters of the ceiling's
  margin above chance on the same roots.
  - **Why 0.75 and not TASK-089's 0.9.** TASK-089's factor was used where the ceiling was about 1
    and G was expected to match it. With a ceiling of 1.0 and n_d = 32, the 0.9 rule passes a G
    whose three seeds each have a true rate of 0.9 with probability 0.07 (two of three seeds;
    §8.2). A model that ranks the squeeze first on 90 % of contact states is not where a planner's
    partial closures come from, so that rule would misread a near-ceiling G as L-MODEL. With 0.75
    such a G passes with probability 0.89, and one whose seeds are at 0.8 with 0.11.
- **The positive control holds** at t when the ceiling ranks the squeeze above the control branch
  on at least **48 / 64** roots (0.75). Otherwise M is void (§7.1).

## 6. Part S — the search (offline; the frozen planner)

At every Rs and R2 root, one CEM call per seed with the frozen planner of §3 (goal as §5.1). This
is TASK-089's A2 call with the step-averaged cost, the call TASK-093 Stage P ran for M-v1.

- **The chosen hold level ℓ:** the chosen plan's right grasp at step 8 **as executed**, that is
  after the cumulative rate clip from the root's last accepted grasp target (0.4545 at Rs, +1 at
  R2). The clip is computed with TASK-093's frozen projector arithmetic (`grasp_rate_projected`),
  used only to read the plan; the plan itself is unprojected, as M-v1's were. **The plan squeezes**
  when ℓ ≥ 0.9 (§1.1's top bin), and **holds partial** otherwise.
- **G's cost of the chosen plan:** the cost CEM's best candidate received (its requests), against
  G-i's cost of the squeeze branch's applied commands over the same window (§5.3 (i)).
- **A search miss** at a root: the plan holds partial **and** G-i rates it worse than the squeeze
  branch (cost(plan) > cost(squeeze)). CEM then failed to find a sequence that its own model rates
  at least as well as one that squeezes.
- **The search misses** at root type t when at least **two of the three seeds** miss on
  ≥ **⌈0.25 × n_d(t)⌉** of the discordant roots.
  - **Why 0.25.** At TASK-089's R1 and R2 the same planner closed and lifted closed on 62 and 61 of
    64 roots (§1.2). A search failure that matters for a closed-loop rate near 0.1 against ik's
    0.7 must be common; a quarter of the decisions at contact is the smallest such share this n
    can tell from a rare one (§8.3).
- **Reported only:** ℓ's distribution; the share of plans that squeeze; the first command's
  requested and executed grasp; *partial preferred* (the plan holds partial and G rates it at least
  as well as the squeeze branch); all of these on every root and on discordant roots.
- **Determinism:** one CEM call per seed re-run on two roots, bitwise.

## 7. Rows of the task, routing and what is put forward

### 7.1 The L rows (read only after K-CAUSAL; first match over the read root types)

1. **L-VOID** — an M or S check failed: a root's bitwise re-run, the squeeze branch's bitwise
   reproduction of the demonstrator, the executed-difference check (more than 8 of 64 roots at a
   root type where the squeeze and s\* branches differ at fewer than 7 of the 8 window steps; such
   a root is otherwise excluded from the ranking and counted), the positive control (§5.5), a
   checkpoint, λ or seed check, the determinism re-run of S, or a cohort run twice. K's row stands.
   M and S run again only on fresh roots named by a ruling.
2. **L-COST (b)** — at some read root type the ceiling does **not** rank the squeeze above s\*
   (ĉ(t) < 0.75).
3. **L-MODEL (a)** — the ceiling ranks it at every read root type, and G does not at some read
   root type.
4. **L-SEARCH (c)** — the ceiling and G rank it at every read root type, and the search misses at
   some read root type.
5. **L-MIXED** — any other pattern, including: no root type is read (n_d < 16 at both); or the
   ceiling and G rank the squeeze and the search does not miss at every read root type (the
   offline roots do not reproduce the closed loop's choice).

**Why this order.** It follows the causal chain from the objective to the plan. A cost that is
flat on real frames at a root type lets any planner choose a partial closure there, whatever the
model or the search do, so a flat ceiling at either root type is read first. The record states
each root type's own row beside the task's row, so a split (for example L-COST at Rs and L-MODEL at
R2) is visible to the ruling.

### 7.2 Every row, in order

| Row | When | Routes to |
| --- | --- | --- |
| **S0-ESCALATE** | Stage 0: a degeneracy or positive-control check fails on development roots or resets, the power rule fails, or the gated stage projects above its budget (§10, §14) | a ruling before the freeze; nothing is gated yet |
| **K-VOID** | §4.3 row 1 | fix under its own ruling; the gated stage runs again on fresh seeds named by it |
| **K-NULL** | §4.3 row 2 | **the grasp line pauses and the owner is told** (R27.157). D's work (for example a Phase 4 precursor with a scripted closure rule) still needs its own ruling, because Phase 4 is not admitted (R25.27, R27.148). The palm-offset arms are reported beside it, so that a placement-dominant result is visible to the pause ruling. |
| **K-CAUSAL + L-VOID** | §7.1 row 1 | M and S again on fresh roots under a ruling |
| **K-CAUSAL + L-COST** | §7.1 row 2 | **option C** (a goal-specification change over TASK-087's G), only through a ruling that finds R26.33's condition met (§7.3); or **option B with a grasp term** (a model change, open under TASK-092 §2 and R27.71, with its own ruling on simulator-truth labels in a learned cost) |
| **K-CAUSAL + L-MODEL** | §7.1 row 3 | **option B**, or a kind of data that R27.71 leaves open (real G1, a different demonstrator or pick recipe, other sensors); R26.33 is not engaged by a model or data change |
| **K-CAUSAL + L-SEARCH** | §7.1 row 4 | a planner ruling under R26.33 (§7.3) |
| **K-CAUSAL + L-MIXED** | §7.1 row 5 | a ruling |
| **K-UNCLEAR** | §4.3 row 4 | a ruling; no pause by itself; M and S are reported |

- **No row admits a controller, a training run or a closed loop.** Each routes to a ruling in
  `docs/DECISIONS.md`.
- **Reported with every row:** every K arm (§4.2), including both offset pairs; M's ceiling and
  G wins per root type and seed with n_d and exact intervals; S's measures; the per-root-type L
  rows; the term decomposition.
- **No abandonment clause.** TASK-094 preregisters no controller or variant, so there is nothing
  for a clause to close. K-NULL's pause is R27.157's fallback, not a clause.
- **"LeWM needed" is not claimed by any row**, and no row claims a grasp success.
- After a VOID, the stage runs again only on fresh seeds named by its ruling. A voided cohort is
  never re-run.

### 7.3 What K-CAUSAL with L-COST puts forward under R26.33 (R27.167)

**K-CAUSAL with L-COST is put forward as candidate "new evidence of a different kind" for option C.
This document does not rule it met.** The ruling after the record decides, and it may find it not
met.

- **Why it would be of a different kind.** R26.33 fired on closed-loop counts after offline
  separations of coarse contrasts that passed (TASK-089 A1, A2). R27.154 found the offline read of
  §1.1 not enough, because it is an association read after the results. TASK-094's K is a paired
  **intervention** on the closure level at an exact pose, and M's ceiling measures, on real frames
  at fresh contact roots, the **action-level ranking** that R27.153 records as not measured. Neither
  existed when R26.33 fired.
- **What it could admit, at most.** A goal-specification change that targets the measured flatness:
  the goal's hand term taken from the demonstrator's *command*, or the hand joints taken out of the
  pose term with a closure floor (design note §4 C). Each would still need its own preregistration.
  No other planner, cost, horizon or action-parametrisation variant is put forward.
- **K-CAUSAL with L-SEARCH** is put forward in the same way, and only for a planner ruling that
  targets the search at the close.
- **K-CAUSAL with L-MODEL** needs no R26.33 finding: a model or data change is open (TASK-092 §2,
  R27.71, R27.149's "not closed").
- **K-NULL, K-UNCLEAR, L-MIXED and the VOID rows put nothing forward.**

## 8. Power and sample size (development; Stage 0 recomputes K's table)

These figures are development figures, CPU only (§12). They come from `power094.py` in
`~/develop/emai/evidence/task094-draft/` (2 000 simulations per K cell with 2 000 bootstrap
resamples each; exact binomial sums for M and S).

### 8.1 K (n = 128 paired resets)

The model: K-1.0 lifts with probability 0.69 (§1.2); the cells "K-1.0 lifts, K-s\* fails" and
"K-1.0 fails, K-s\* lifts" have probabilities Δ + d₀ and d₀, with discordant noise d₀ = 0.03 or
0.08.

| True Δ | d₀ | K-CAUSAL | K-NULL | K-UNCLEAR |
| --- | --- | --- | --- | --- |
| 0.00 | 0.03 / 0.08 | 0.000 / 0.000 | **1.000 / 0.986** | 0.000 / 0.015 |
| 0.05 | 0.03 / 0.08 | 0.000 / 0.006 | 0.920 / 0.698 | 0.080 / 0.296 |
| 0.10 | 0.03 / 0.08 | 0.058 / 0.126 | 0.349 / 0.240 | **0.593 / 0.633** |
| 0.15 | 0.03 / 0.08 | 0.473 / 0.478 | 0.037 / 0.033 | 0.489 / 0.489 |
| 0.20 | 0.03 / 0.08 | 0.886 / 0.826 | 0.001 / 0.001 | 0.113 / 0.174 |
| 0.30 | 0.03 / 0.08 | **1.000 / 0.997** | 0.000 | 0.000 / 0.003 |
| 0.50 | 0.03 / 0.08 | 1.000 / 1.000 | 0.000 | 0.000 |

- **What n = 128 does.** A true Δ of 0.30 or more reads K-CAUSAL with probability ≥ 0.997; no
  loss reads K-NULL with ≥ 0.986. A true Δ near δ_K = 0.15 splits between K-CAUSAL and K-UNCLEAR,
  and a true Δ of about 0.10 mostly reads K-UNCLEAR. K-NULL at a true Δ of 0.10 has probability
  0.24–0.35: the pause can be triggered by a real but small loss, which δ_K declares below the
  size that matters (§4.3).
- **Why not n = 64.** At n = 64 a true Δ = 0.20 reads K-CAUSAL with only 0.59–0.80, and no loss
  reads K-NULL with 0.84–0.99. K costs well under 0.1 GPU h at n = 128 (§10), so the larger cohort
  is cheap.
- **K-VOID band (63–114 of 128).** If K-1.0's true rate is 0.60–0.80, the band voids the run with
  probability at most 0.005; at 0.55 with 0.08, and at 0.875 (TASK-089's BD rate on 32 resets)
  with 0.26. A rate that far from every BS record (0.64–0.75) would question the harness, not the
  dose.
- **K-open (≤ 3 / 128).** If an open hand lifted 1 % of the time, the check would void with
  probability 0.04.

### 8.2 M (n_d discordant roots of 64 per root type)

- **L-COST misreads.** The probability that the ceiling falls below 0.75 at a true ceiling rate r:

  | n_d | r = 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
  | --- | --- | --- | --- | --- | --- | --- |
  | 16 | 0.962 | 0.833 | 0.550 | 0.202 | 0.017 | 0.000 |
  | 32 | 0.996 | 0.943 | 0.656 | 0.175 | 0.003 | 0.000 |
  | 48 | 1.000 | 0.978 | 0.720 | 0.148 | 0.001 | 0.000 |

  A flat ceiling (r ≈ 0.5, the hypothesis of (b)) reads L-COST with probability ≥ 0.96 at n_d ≥ 16.
  A ceiling near the bar (0.7–0.8) is misread often; the record reports ĉ with its exact interval.
- **G's bar** (two of three seeds), for example at n_d = 32: with a ceiling of 1.0, a G whose
  three seeds each have a true rate of 0.9 passes with probability 0.89, and one at 0.8 with 0.11;
  with a ceiling of 0.9, 0.996 and 0.55. TASK-089's 0.9 rule would give 0.07 and 0.00 at a ceiling of 1.0.
- **n = 64 root seeds** gives n_d = 32 if s\* fails on half the roots. n_d is not known before the
  run; it is the one quantity that can leave a root type unread, and Stage 0 reports it on
  development roots. A root type with n_d < 16 is reported, not read (§5.4).

### 8.3 S

The probability that the search is read as missing (two of three seeds at ≥ ⌈0.25 n_d⌉), at a
true per-root miss rate m:

| n_d | m = 0.05 | 0.10 | 0.15 | 0.25 | 0.35 | 0.50 |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 0.000 | 0.013 | 0.114 | 0.641 | 0.951 | 1.000 |
| 32 | 0.000 | 0.000 | 0.026 | 0.601 | 0.981 | 1.000 |

A rare miss (≤ 0.10) is not read as a search failure; a miss on a third of the decisions or more
is read as one with probability ≥ 0.95.

### 8.4 Stage 0's power rule

Stage 0 recomputes §8.1's table with 20 000 simulations. If K-CAUSAL's probability at a true Δ of
0.30 (d₀ = 0.08) is below 0.95, or K-NULL's at Δ = 0 is below 0.95, it discloses this and
escalates before the freeze (S0-ESCALATE).

## 9. Seeds and salts

| Use | Seeds |
| --- | --- |
| Model checkpoints | TASK-087's **G-87100, G-87101, G-87102** (no training; G-87102 `last_two_triggered`) |
| Stage 0 development K resets | **94100–94131** |
| Stage 0 development roots (M, S) | **940000–940031** |
| Gated K resets | **941000–941127** |
| Gated M and S root seeds | **942000–942063** |
| Determinism re-runs | K-1.0 and K-s\* on 941000–941003; S on 942000–942001 (inside the gated cohorts) |
| Debug (Stage 0 plumbing and smoke; never counted) | **94500–94599** |

**Salts.**

- **New:** **9014**, K's reset bootstrap; **9015** reserved.
- **Reused:** TASK-088's 8801 and 8802 (layouts and goals; also the root draws, as TASK-089's root
  builder) and 8803 (CEM, through `cem_seed`), with the new seeds.
- **Rejected draws:** a reset's or root's rejected draws are further attempts under the same seed,
  as TASK-088 and TASK-089.

**Checked (DRAFT, 2026-10-15).**

- **Branches.** On all 120 remote-tracking refs and every local branch (after `git fetch --all`),
  over `src/`, `scripts/`, `docs/`, `.mc/`, `tests/`, `benchmarks/` and `configs/`, `git grep -w`
  finds no token in 94100–94131, 94500–94599, 940000–940031, 941000–941127 or 942000–942063, and
  none for the salts 9014 and 9015.
  - Nearby tokens that were avoided: 92400 and 92800 (apple manifests), 925329, 930000–935996
    (several), 9012 (the end bound of TASK-093's salt range) and 9013 (an apple results document).
- **Evidence.** In `~/develop/emai/evidence/` (`*.json`, `*.txt`, `*.log`, `*.md`, `*.csv`,
  `*.yaml`, `*.py`, `*.sh`), standalone tokens in these ranges occur only as 94100 and 94500,
  training-update indices in TASK-077's Stage T loss curves (`task077-staget*/…/report.json`). None
  is a seed or reset.
- **Ranges already used nearby** (not reused): TASK-093 916000–918127, 91900–91931 and 91940–91999;
  TASK-092 908000–909127, 90400–90599, 90810–90841 (spent) and 90850–90881 (D-RATE′); TASK-091
  906000–907127, 90300–90331 and 90700–90799; TASK-090 903000–905063, 90200–90231, 910000–913199
  and 920000–923199; TASK-089 89000–89999 and 890000–891063; TASK-088 88000–88999 and
  880000–881063.
- **Stage 0** repeats both checks at its own revision and records them.

## 10. Runtime, budget, stop rules, paths and evidence

### 10.1 GPU and sharing

Every GPU use (rendering, encoding, roll-outs, CEM) is one `scripts/gpu_run.sh --wait
--min-free-gib 8 --board` job, under the shared lock per job.

- Before each job the `## Requests` section of `~/.local/state/gpu/BOARD.md` is checked; a
  `- v2d: …` line means the job waits.
- **The V2D session is told in advance** about Stage 0's GPU jobs and about the gated stage.
- The GR00T server, `/mnt/data/emai/{archive,models,cache}` and other worktrees are not touched.

### 10.2 Budget and stop rules

| Step | Estimate | Basis |
| --- | --- | --- |
| Stage 0 (development K on 32 resets, development M and S on 32 roots with the squeeze, s\* and control branches only, debug smoke, timing, opt-in render tests) | about 0.15–0.2 GPU h | TASK-089 Stage 0; ik ran 128 resets in 29–43 s per job |
| Gated K (128 goals; 11 arms × 128 resets; the determinism re-run) | about 0.1 GPU h | as above; the goals as TASK-093 BS |
| Gated M (64 root seeds; 2 root types × 6 branches of 70–97 commands, each re-run from its reset; frames encoded at 8 steps; three G roll-outs per branch) | about 0.2–0.3 GPU h | TASK-089 Stage A: 64 roots with 5 branches and 9 models in about 6 minutes; M simulates about three times as many commands |
| Gated S (3 seeds × 128 root calls; re-run) | about 0.05 GPU h | TASK-093 Stage P: 768 calls in 281 s |
| **Total** | **about 0.5 GPU h (0.4–0.7); ceiling 1.0 GPU h** | |

- **Per-job cap 1 800 s.** A job that reaches it exits; in the gated stage that is K-VOID (or
  L-VOID for an M or S job).
- **Stage 0's projection.** Stage 0 times its jobs and projects the gated stage. If the projection
  exceeds **0.6 GPU h**, or Stage 0 itself exceeds 0.4 GPU h, it escalates before the freeze
  (S0-ESCALATE).
- **The ceiling.** If the GPU time of all TASK-094 jobs (Stage 0 and gated) reaches **1.0 GPU h**
  before the gated stage completes, no further job starts, and the stage is K-VOID (budget) for a
  ruling.

### 10.3 Runs, ledger and guards

- Every job refuses another working directory or a missing `third_party/le-wm` (R27.107's lesson).
- Before its ledger claim, each job loads and checks its goals or roots and, for M and S, its G
  checkpoints and TASK-087's DINOv2 encoder with its projection (preflight, on the CPU), as TASK-093
  §16.1. One ledger entry per job; a cohort is never run twice.
- The gated stage runs once, at the frozen revision, from a clean tree, with a stage script, in
  the order K, M, S, summary. M and S run whatever K's row; their L rows are read only after
  K-CAUSAL (§7.1). No GO between the parts is needed, because nothing in the stage controls a
  robot in closed loop and every reading rule is fixed here.

### 10.4 Disk, paths and evidence

- **Disk.** No corpus, feature store or checkpoint. Goals, roots, branches (frames at 8 steps only)
  and reports go to `outputs/task094-*` on the data disk through the repository's links (a few
  hundred MB at most). Nothing is written below 10 GiB free.
- **Paths that must stay.** TASK-087's feature store and projection in `worktrees/task087-run/outputs/`
  and `checkpoints/task087-run/`. **`worktrees/task087-run` must not be archived or removed until
  TASK-094 closes.**
- **Evidence.** `~/develop/emai/evidence/task094-*/` with `SHA256SUMS`. Existing evidence under
  `data/`, `checkpoints/`, `outputs/` and `~/develop/emai/evidence/task08*`, `task09[0-3]-*` and
  `grasp-next-step/` is read and never written.

## 11. What this cannot show, and its risks

- **No grasp success.** It runs no closed-loop world-model controller, so it cannot show that any
  change would grasp. A K-CAUSAL with L-COST says where a fix should aim, not that it would work.
- **The demonstrator's states, not the WM arms'.** ik's median minimum palm distance to the close
  subgoal was 0.21–0.24 cm (§1.1), and M's and S's roots lie on the demonstrator's own trajectory. The WM arms reach
  contact states that are less precise, with other arm motion. Rs answers the question of the start
  of the close only for the demonstrator's closing hand; a WM arm that hesitates earlier in the
  close is not covered.
- **Scripted contrasts.** The hold levels are constant per branch and per arm; the WM arms' commands
  vary from step to step. A held level is a dose, not a replay.
- **The offset is two directions of one size.** The WM arms' error direction is not logged; other
  directions and larger errors are untested.
- **One of each.** One scene, arm, hand and camera at 112 px; simulation only; one encoder, latent,
  corpus, recipe and cost; three model seeds of one training run (G-87102 `last_two_triggered`);
  goals from a scripted demonstrator, biased to layouts a scripted pick can do (almost no banana).
- **Near the bars.** δ_K, the ceiling's 0.75 and S's 0.25 are declared. Near each, a row can be
  misread (§8); the record reports every quantity with its interval so the ruling can see it.
- **Read only on discordant roots.** If s\* lifts at the demonstrator's exact states, n_d is small
  and M is not read, even if K reads K-CAUSAL (K's follower is not the demonstrator). That case is
  L-MIXED, and the per-level lift rates at the roots are reported.
- **Cost on requests against cost on executed commands (S).** CEM scores its candidates' requests,
  as in the closed loop; the squeeze branch is scored on its applied commands. Both are what the
  frozen planner and the closed loop actually compare, but they are not the same object. The
  record reports the plan's executed grasp and the request beside each other.
- **The ceiling is the frozen cost on real frames.** It measures what this cost can rank, not what
  any cost could.

## 12. Disclosed development before this draft

All of it is development and **not gated**: CPU only, at `4afa1cd`. No GPU job, simulation,
rendering, model load, planner call or training was run for this draft.

- **Code reading.** `zero_shot.py` (the IK follower, its press and ramp, the grasp demonstrator,
  `synergy_grasp`, the switching rule and the scorer), `zero_shot_grasp.py` (TASK-089's root builder
  and branches), the rate limit in `embodiment.py` and `configs/g1_sim_action.json`, and the run
  reports' fields in `~/develop/emai/evidence/task093-bs/` (which showed that the palm *direction*
  is not logged).
- **The numbers of §1.** Quoted from merged records and the design note; none was recomputed.
- **Power.** §8's tables come from `power094.py` (2 000 simulations per K cell; exact binomial sums
  for M and S), archived in `~/develop/emai/evidence/task094-draft/` (`power094.py`,
  `power094.json`, `README.txt`; `SHA256SUMS` sha256 `f1b5b608…4ba4`).
  - The draft first used TASK-089's 0.9 factor for G's bar and a K-VOID band of ± 0.15. The
    simulation showed the first would misread a near-ceiling G (§5.5) and the second would void a
    rate of 0.80 with probability 0.13. Both were changed to the values above before this draft was
    submitted; the archived script is the final one.
- **Seed checks.** §9's checks.

These choices set the parts, the levels, the offsets and the bars. No bar was set from a measured
TASK-094 quantity.

## 13. Order

1. This DRAFT, reviewed independently and merged.
2. A Stage 0 PR (§14), reviewed. **The protocol is frozen at its merge.**
3. The gated stage (K, M, S) at the frozen revision from a clean tree, with
   `scripts/run_task094_stage.sh`, after the V2D session is told.
4. Its record PR, with the row and a DECISIONS entry with the next-step ruling (K-NULL pauses the
   grasp line; every other row routes as §7.2).

## 14. Stage 0, freeze and what Stage 0 may change

**Stage 0** is its own PR, merged before the freeze. It adds only opt-in modules and new scripts,
and changes no existing file except CLAUDE.md's entry points and this record. `planning.py`,
`embodiment.py`, TASK-087's implementation hash and TASK-088–TASK-093's frozen code stay as they
are, checked against sha256 pins before any job runs.

- **A new module** (NumPy at import), for example `grasp_closure_locus.py`: the cohorts and seeds;
  the K arms (a follower that subclasses TASK-088's `IKController` with the held level, and the
  displaced goal); K's paired statistics and rows; the root types and branch definitions; M's
  discordant roots, wins, bars and positive control; S's measures; the L rows and the task row; the
  power recompute; sha256 pins of every earlier file it imports.
- **A torch module**, for example `grasp_closure_locus_runtime.py`: M's roll-outs and ceiling costs
  through the frozen planner's own `distance` (TASK-089's A1 code path with the step-averaged cost);
  S's CEM call (TASK-093's `a2_plan` path for M-v1).
- **Scripts:** `scripts/{run,summarize}_task094.py` and `scripts/run_task094_stage.sh`.
- **Tests:** the held level reaches the request after the ramp and nothing else in the follower
  changes (bitwise against `IKController` at c = +1.0); the goal displacement; the root types' step
  counts and the executed-difference arithmetic (≥ 7 of 8 steps at Rs for every s\* below 0.818;
  8 at R2); the squeeze branch equals the demonstrator's continuation (graphics opt-in); the win, bar
  and row rules on synthetic inputs, including first-match order and the n_d < 16 case; the K rows,
  the band and the bootstrap; the seed ranges and their disjointness from earlier tasks; the
  frozen-file pins; the working-directory guard; the ledger claim's place after the goals or roots
  and the model loads; the core-import check.

**Stage 0 also runs** (on development and debug seeds only):

- the hash and seed checks (§9);
- **development K** on 94100–94131, every arm: the plumbing checks of §4.3, K-1.0's and K-open's
  counts, and every arm's counts (reported, not gated; they set nothing);
- **development M and S** on 940000–940031, with the squeeze, s\* and control branches only:
  - the bitwise root re-runs and the squeeze branch's reproduction of the demonstrator;
  - the executed-difference check at every root;
  - **the positive control** at both root types: the ceiling ranks the squeeze above the control on
    at least 24 / 32 roots;
  - n_d at each root type (reported; if n_d < 8 of 32 at both root types, Stage 0 discloses this
    and escalates, because M would most likely not be read);
  - G's wins and S's measures (reported, not gated; they set nothing);
- the power recompute of §8.4;
- a debug smoke on 94500–94599 of every step, timed, for §10.2's projection.

**Stage 0 may change only:** the per-run process count, and the split of a step into more, smaller
jobs. Each change is recorded. A failed check of the list above is S0-ESCALATE, not a change.
**Every level, offset, root type, bar, margin and row is frozen here** and is not changed by any
development number. The protocol is frozen at the Stage 0 merge.

## 15. Rulings this document records (R27.160–R27.176; DRAFT; decided by Claude under owner delegation)

- **R27.160** — **TASK-094's DRAFT preregistration is this document.** It is measurement only: no
  controller, no training, no cost or planner variant, no world-model closed loop. Checked against
  every fired clause (§2), it engages none; R27.154 stands.
- **R27.161** — **Part K** (§4.1): TASK-088's IK follower with its closing command held at +1.0,
  0.9, 0.8, 0.7, s\* and 0.6 (reported), K-open as a degeneracy reference, and the palm offsets
  v_x and v_y of 1.2 cm at +1.0 and s\*, on 128 paired gated resets. The level 0.6 and the second
  offset direction are added to R27.156's list as reported arms.
- **R27.162** — **Part M** (§5): 64 root seeds; branches at +1.0, 0.9, 0.8, 0.7, s\* and a control,
  simulated to their true outcome; the frozen step-averaged cost on G-87100–87102's predictions and
  on the real frames (the ceiling); the ranking read on discordant roots, a root type read only at
  n_d ≥ 16.
- **R27.163** — **Roots in the close: yes** (§5.1). M and S use Rs, 8 of the demonstrator's 35
  close commands in, beside R2. TASK-089's R1 is not used, because the rate limit makes every hold
  level ≥ 0.4545 execute identically over its 8-step window.
- **R27.164** — **Part S** (§6): the frozen, unprojected step-averaged planner, one call per root
  and seed; the executed hold level, the search miss and its bar.
- **R27.165** — **K's rows and bars** (§4.3): K-VOID (K-1.0 outside 63–114 of 128, K-open above
  3 / 128, or a check), K-NULL (bootstrap upper bound of Δ below δ_K = 0.15), K-CAUSAL (Δ ≥ 0.15
  and p < 0.01), K-UNCLEAR.
- **R27.166** — **The L rows and bars** (§5.5, §6, §7.1): the ceiling at 0.75 (TASK-089's
  threshold); G at ⌈(0.5 + 0.75 (ĉ − 0.5)) n_d⌉ on two of three seeds; the search miss at
  ⌈0.25 n_d⌉ on two of three seeds; the positive control at 48 / 64; L-VOID, L-COST, L-MODEL,
  L-SEARCH, L-MIXED, first match over the read root types.
- **R27.167** — **Routing** (§7.2, §7.3). No row admits a controller. K-NULL pauses the grasp line
  (R27.157). **K-CAUSAL with L-COST is put forward as candidate R26.33 evidence for option C**, and
  K-CAUSAL with L-SEARCH for a planner ruling; neither is ruled met here.
- **R27.168** — **No abandonment clause** (§7.2).
- **R27.169** — **Power** (§8; development): n = 128 for K, 64 root seeds for M and S, with the
  misread rates disclosed; Stage 0 recomputes K's table with 20 000 simulations.
- **R27.170** — **Seeds and salts** (§9), checked on 120 remote-tracking refs and in the evidence;
  new salt 9014, 9015 reserved.
- **R27.171** — **Budget and sharing** (§10): about 0.5 GPU h, ceiling 1.0; per-job cap 1 800 s;
  Stage 0 escalates if it projects the gated stage above 0.6 GPU h; every job under
  `gpu_run.sh --wait --board` after a BOARD.md check; the V2D session told in advance;
  `worktrees/task087-run` stays until TASK-094 closes.
- **R27.172** — **Stage 0 and the freeze** (§14), and what Stage 0 may change.
- **R27.173** — **Disclosed development** (§12) is recorded.
- **R27.174** — **What it cannot show** (§11) is part of every restatement of its row.
- **R27.175** — **Status.** DRAFT in this PR, with an independent review. A Stage 0 PR freezes it
  at its merge; the gated stage runs only after that.
- **R27.176** — **R7 does not change**, whatever row TASK-094 ends in. TASK-094 is not an
  Apple→Plate experiment.
