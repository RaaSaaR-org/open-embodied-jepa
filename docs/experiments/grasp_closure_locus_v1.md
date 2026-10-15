# TASK-094 — grasp-closure locus diagnostic: does the partial closure the world-model arms command cause their grasp failures, and does the failure sit in the goal and cost, the model or the search (measurement only; no controller)

**Status: FROZEN at the merge of the Stage 0′ PR** (R27.160–R27.176; Stage 0 record §16,
R27.177–R27.182; escalation ruling §17, R27.183–R27.186; Stage 0′ record §18, R27.187–R27.192;
decided by Claude under owner delegation, 2026-10-15; DECISIONS (bi), (bj), (bk), (bl)). Stage 0
ended S0-ESCALATE on §14's n_d rule (6 of 32 development roots discordant at Rs, 1 at R2). The
ruling (§17) amended M and S before the freeze: 128 gated root seeds 946000–946127 (942000–942063
retired, never generated), the positive control at 112 / 128 and the executed-difference limit at
16 / 128, S's ℓ the median executed right grasp over window steps 3–7, and S's search miss read
against the plan with the squeeze branch's right-grasp requests. **Stage 0′ ended S0-PASS** (§18):
with no new development cohort, the development rates project n_d = 24 at Rs (4 at R2) at 128
roots, every check passes, and the gated stage projects to 0.352 GPU h. §5, §6, §9, §10.2 and §14
carry the amendment; K is unchanged. **The gated stage ran once (§19; R27.193–R27.199): K-NULL** by
§4.3's text (the summary's K-VOID is a size-check defect; K-NULL binding by R27.195), with the
roots job VOID at one root (L-VOID; M and S reported, not read). The grasp line pauses and the owner
is told (§7.2), and TASK-094 closes at K-NULL.

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
| **M, the ranking** | scripted hold-level branches from fresh roots, simulated to their true outcome, scored by the frozen step-averaged cost on G's predictions and on the branch's real frames (the ceiling) | G-87100, G-87101, G-87102 (frozen) | 128 fresh gated root seeds (R27.183), two root types |
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
| K-0.6 | 0.6 | the goal's | reported, and part of K-NULL's condition (§4.3) (below s\*, inside the WM arms' per-run range) |
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
one-sided McNemar test on the discordant resets and the two-sided 95 % reset-bootstrap percentile
interval (10 000 resamples, **salt 9014**; its upper bound is the 97.5th percentile). The offset
arms are also paired against their no-offset twins.

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
2. **K-NULL** — the upper end of the two-sided 95 % reset-bootstrap percentile interval (§4.2) of
   Δ is **below δ_K = 0.15**, **and so is that of Δ_{0.6}**: at an exact pose, holding the closure
   at s\* or at 0.6 (inside the WM arms' per-run range of 0.53–0.73) loses less than 0.15 of the
   lifts against the +1 squeeze. If Δ's bound is below δ_K and Δ_{0.6}'s is not, the row is
   K-UNCLEAR.
3. **K-CAUSAL** — Δ ≥ δ_K **and** "K-1.0 beats K-s\*" by the exact one-sided McNemar test at
   **p < 0.01**.
4. **K-UNCLEAR** — anything else (for example a significant Δ below 0.15 whose interval reaches
   0.15, or a Δ ≥ 0.15 that is not significant).

**Why δ_K = 0.15.** Given contact, the WM arms succeed on 10.7 % and ik on 72.8 % (§1.1), a gap of
about 0.62 (ik makes contact on 302/320, so the two scales are close). A physical loss below 0.15
could explain at most about a quarter of it, so it would not be the primary cause.

**Why K-NULL also asks for 0.6.** K-NULL pauses the grasp line. s\* (about 0.68) lies at the top of
the WM arms' per-run lift-phase levels (0.53–0.73), and 739 of 982 contact episodes commanded below
0.7 before the rise, so a null at s\* alone would not show that the closures the WM arms command are
harmless.

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
  the s\* branch does not. n_d(t) is their number at root type t (of 128; R27.183).
  - The ranking is read **on discordant roots only**. Where both branches lift, neither ranking is
    wrong, and including those roots would cap any cost's win rate below what it can show.
  - **A root type is read only when n_d(t) ≥ 16.** Below that, its rows are reported, not read.
    If neither root type is read, the task's L row is L-UNREAD (§7.1).
  - **R2 is expected to be unread** (R27.183 (2)). At Stage 0's development rate (1 of 32 roots
    discordant) its expected n_d at 128 roots is 4. It stays: every R2 check (the bitwise roots,
    the squeeze branch's reproduction, the executed-difference check, the positive control) still
    voids M (L-VOID), and its rows and per-level lift rates are reported. The read rule is not
    changed for it: it is read only if its n_d reaches 16.
- **A win** at a discordant root: cost(squeeze) < cost(s\*), strictly; ties are not wins.
  - **Ceiling win rate** ĉ(t) = ceiling wins / n_d(t).
  - **G-i wins** per seed, out of n_d(t).
- **Positive control** (every root, not only discordant ones): the ceiling ranks the squeeze branch
  above the control branch (re-open at Rs, release at R2), which never lifts. TASK-089's A-K0 found
  similar contrasts at 1.000 on real frames (close-vs-open from R1 and lift-vs-release, the per-step
  cost at h = 8); its bar (§5.5) is set accordingly.
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
  on at least **112 / 128** roots (0.875; the DRAFT's 56 / 64 in the same proportion, R27.183).
  Otherwise M is void (§7.1).
  - **Why 0.875.** It is a plumbing check, not a measurement: TASK-089's A-K0 found the analogous
    contrasts at **1.000** on real frames, so a ceiling that misranks a branch that never lifts on
    more than one root in eight points at the harness, and 0.75 would be lenient.

## 6. Part S — the search (offline; the frozen planner)

At every Rs and R2 root, one CEM call per seed with the frozen planner of §3 (goal as §5.1). This
is TASK-089's A2 call with the step-averaged cost, the call TASK-093 Stage P ran for M-v1.

- **The chosen hold level ℓ** (amended by R27.183 (3)): the **median of the chosen plan's
  executed right grasp over window steps 3–7**, the executed grasp being the plan's requests after
  the cumulative rate clip from the root's last accepted grasp target (0.4545 at Rs, +1 at R2). The
  clip is computed with TASK-093's frozen projector arithmetic (`grasp_rate_projected`), used only
  to read the plan; the plan itself is unprojected, as M-v1's were. **The plan squeezes** when
  ℓ ≥ 0.9 (§1.1's top bin), and **holds partial** otherwise.
  - **Why steps 3–7.** At Rs steps 1–2 are bound by the rate limit for every plan that closes.
    Step 8 is the end of the horizon, where Stage 0's development plans' requests were lowest and
    most dispersed (median 0.52–0.69, IQR 0.17–0.31, against about 0.8–0.9 at steps 3–7; §16.5).
    Step 8's executed grasp (the DRAFT's ℓ) and the full executed path are reported.
- **The comparator** (amended by R27.183 (4)): the **substituted plan** is the chosen plan with its
  right-grasp requests over steps 1–8 replaced by the squeeze branch's right-grasp requests
  (`grasp_closure_locus.squeeze_requests`: the demonstrator's 0.2 ramp continued from the root, 0.8
  then +1 at Rs, +1 at R2), every other channel the plan's own. G-i's step-averaged cost of the
  plan and of the substituted plan are computed **on requests, as CEM scores its candidates**, in
  one roll-out call (§5.3 (i)'s roll-out). CEM's own best-candidate cost is reported beside them.
- **A search miss** at a root: the plan holds partial **and** the substituted plan's cost is
  **strictly lower** than the plan's. CEM then kept a partial closure although its own model rates
  the same plan with the squeeze's closing requests better.
- **Partial preferred**: the plan holds partial and the substituted plan costs at least as much.
- **Why it changed.** CEM optimises all 14 channels and is scored on its requests, so its best
  candidate cost less than the scripted squeeze branch's applied commands on every development
  call (192 / 192; §16.5), and the DRAFT's miss (cost(plan) > cost(squeeze branch)) could not fire.
  The substituted plan differs from the plan in the right grasp alone, so the comparison isolates
  the closure. **The DRAFT's comparison** (CEM's best-candidate cost against G-i's cost of the
  squeeze branch's applied commands) is reported.
- **The search misses** at root type t when at least **two of the three seeds** miss on
  ≥ **⌈0.25 × n_d(t)⌉** of the discordant roots (unchanged).
  - **Why 0.25.** At TASK-089's R1 and R2 the same planner closed and lifted closed on 62 and 61 of
    64 roots (§1.2). A search failure that matters for a closed-loop rate near 0.1 against ik's
    0.7 must be common; a quarter of the decisions at contact is the smallest such share this n
    can tell from a rare one (§8.3).
- **Reported only:** ℓ's distribution; the share of plans that squeeze; step 8's executed grasp and
  the full executed path; the first command's requested and executed grasp; *partial preferred*;
  the DRAFT's comparison; the substituted plan's cost relative to the plan's; all of these on
  every root and on discordant roots. A plumbing reading, reported: the squeeze branch's requests,
  executed through the same clip, reproduce its recorded applied right grasp (checked on the
  development roots at Stage 0′, §18).
- **Determinism:** one CEM call per seed re-run on two roots (946000–946001), bitwise, with the
  comparator's costs.

## 7. Rows of the task, routing and what is put forward

### 7.1 The L rows (read only after K-CAUSAL; first match over the read root types)

1. **L-VOID** — an M or S check failed: a root's bitwise re-run, the squeeze branch's bitwise
   reproduction of the demonstrator, the executed-difference check (more than 16 of 128 roots, 1 / 8,
   at a root type where the squeeze and s\* branches differ at fewer than 7 of the 8 window steps;
   such a root is otherwise excluded from the ranking and counted; R27.183), the positive control (§5.5), a
   checkpoint, λ or seed check, the determinism re-run of S, or a cohort run twice. K's row stands.
   M and S run again only on fresh roots named by a ruling.
2. **L-UNREAD** — no root type is read (n_d < 16 at both; §5.4). M and S are reported, with the
   per-level lift rates at the roots, but they cannot locate the failure.
3. **L-COST (b)** — at some read root type the ceiling does **not** rank the squeeze above s\*
   (ĉ(t) < 0.75).
4. **L-MODEL (a)** — the ceiling ranks it at every read root type, and G does not at some read
   root type.
5. **L-SEARCH (c)** — the ceiling and G rank it at every read root type, and the search misses at
   some read root type.
6. **L-MIXED** — any other pattern, including: the ceiling and G rank the squeeze and the search
   does not miss at every read root type (the offline roots do not reproduce the closed loop's
   choice).

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
| **K-CAUSAL + L-UNREAD** | §7.1 row 2 | a ruling, as L-MIXED; it is named separately so that an unread M is not restated as a mixed pattern |
| **K-CAUSAL + L-COST** | §7.1 row 3 | **option C** (a goal-specification change over TASK-087's G), only through a ruling that finds R26.33's condition met (§7.3); or **option B with a grasp term** (a model change, open under TASK-092 §2 and R27.71, with its own ruling on simulator-truth labels in a learned cost) |
| **K-CAUSAL + L-MODEL** | §7.1 row 4 | **option B**, or a kind of data that R27.71 leaves open (real G1, a different demonstrator or pick recipe, other sensors); R26.33 is not engaged by a model or data change |
| **K-CAUSAL + L-SEARCH** | §7.1 row 5 | a planner ruling under R26.33 (§7.3) |
| **K-CAUSAL + L-MIXED** | §7.1 row 6 | a ruling |
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
- **Which root type.** The record names the root type(s) at which L-COST was read. An L-COST at Rs
  alone (the ceiling ranks the squeeze at R2) shows that the goal's hand term is flat during the
  close, not that the cost cannot see a failed hold; the ruling weighs it with the per-root-type
  rows and may find it narrower evidence than an L-COST at R2.
- **K-CAUSAL with L-SEARCH** is put forward in the same way, and only for a planner ruling that
  targets the search at the close.
- **K-CAUSAL with L-MODEL** needs no R26.33 finding: a model or data change is open (TASK-092 §2,
  R27.71, R27.149's "not closed").
- **K-NULL, K-UNCLEAR, L-UNREAD, L-MIXED and the VOID rows put nothing forward.**

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

The table is for Δ_{s\*} alone; K-NULL also requires Δ_{0.6}'s bound (§4.3), and Stage 0's
recompute (§8.4) simulates that joint condition with Δ_{0.6} ≥ Δ_{s\*}.

- **What n = 128 does.** A true Δ of 0.30 or more reads K-CAUSAL with probability ≥ 0.997; no
  loss reads K-NULL with ≥ 0.986. A true Δ near δ_K = 0.15 splits between K-CAUSAL and K-UNCLEAR,
  and a true Δ of about 0.10 mostly reads K-UNCLEAR. K-NULL at a true Δ of 0.10 has probability
  0.24–0.35: the pause can be triggered by a real but small loss, which δ_K declares below the
  size that matters (§4.3).
- **Why not n = 64.** At n = 64 a true Δ = 0.20 reads K-CAUSAL with only 0.59–0.80, and no loss
  reads K-NULL with 0.84–0.99. K costs about 0.09–0.13 GPU h at n = 128 (§10), so the larger cohort
  is cheap.
- **K-VOID band (63–114 of 128).** If K-1.0's true rate is 0.60–0.80, the band voids the run with
  probability at most 0.005; at 0.55 with 0.08, and at 0.875 (TASK-089's BD rate on 32 resets)
  with 0.26. A rate that far from every BS record (0.64–0.75) would question the harness, not the
  dose.
- **K-open (≤ 3 / 128).** If an open hand lifted 1 % of the time, the check would void with
  probability 0.04.

### 8.2 M (n_d discordant roots of 128 per root type; R27.183)

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
  These two-of-three probabilities treat the three seeds as independent. They are not: the seeds
  are scored on the same roots and come from one training run, so their wins are correlated, and
  the figures are an approximation.
- **The root count** (amended by R27.183 from the DRAFT's 64). n_d is not known before the run;
  it is the one quantity that can leave a root type unread (both unread: L-UNREAD). Stage 0
  measured it on 32 development roots: 6 at Rs and 1 at R2 (§16.5). At 128 root seeds and Rs's
  development rate, the expected n_d is 24 and P(n_d ≥ 16) is 0.977 at the point rate, 0.84 with
  the rate's uncertainty integrated (Jeffreys; 0.96 at 192 roots); at R2 the expected n_d is 4
  (§18). A root type with n_d < 16 is reported, not read (§5.4).

### 8.3 S

The probability that the search is read as missing (two of three seeds at ≥ ⌈0.25 n_d⌉), at a
true per-root miss rate m:

| n_d | m = 0.05 | 0.10 | 0.15 | 0.25 | 0.35 | 0.50 |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 0.000 | 0.013 | 0.114 | 0.641 | 0.951 | 1.000 |
| 32 | 0.000 | 0.000 | 0.026 | 0.601 | 0.981 | 1.000 |

A rare miss (≤ 0.10) is not read as a search failure; a miss on a third of the decisions or more
is read as one with probability ≥ 0.95. As in §8.2, the two-of-three figures treat the three seeds
as independent, which they are not (the same roots, one training run); they are an approximation.

### 8.4 Stage 0's power rule

Stage 0 recomputes §8.1's table with 20 000 simulations. If K-CAUSAL's probability at a true Δ of
0.30 (d₀ = 0.08) is below 0.95, or K-NULL's (the joint condition of §4.3) at Δ = Δ_{0.6} = 0 is
below 0.95, it discloses this and escalates before the freeze (S0-ESCALATE).

## 9. Seeds and salts

| Use | Seeds |
| --- | --- |
| Model checkpoints | TASK-087's **G-87100, G-87101, G-87102** (no training; G-87102 `last_two_triggered`) |
| Stage 0 development K resets | **94100–94131** |
| Stage 0 development roots (M, S) | **940000–940031** |
| Gated K resets | **941000–941127** |
| Gated M and S root seeds | **946000–946127** (R27.183 (1)) |
| Retired, never generated or run, not reused | 942000–942063 (the DRAFT's gated root seeds; every seed check refuses them) |
| Determinism re-runs | K-1.0 and K-s\* on 941000–941003; S on 946000–946001 (inside the gated cohorts) |
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
- **Stage 0** repeats both checks at its own revision and records them; **Stage 0′** repeats them
  with 946000–946127 and the retired 942000–942063 (§18).

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
| Gated K (128 goals; 11 arms × 128 resets; the determinism re-run) | about 0.09–0.13 GPU h (11 jobs × 29–43 s), before goal generation | as above; the goals as TASK-093 BS |
| Gated M (128 root seeds since R27.183; 2 root types × 6 branches of 70–97 commands, each re-run from its reset; frames encoded at 8 steps; three G roll-outs per branch) | the DRAFT's 0.2–0.3 GPU h at 64 roots; Stage 0′ projects it from Stage 0's jobs (§18) | TASK-089 Stage A: 64 roots with 5 branches and 9 models in about 6 minutes; M simulates about three times as many commands |
| Gated S (3 seeds × 256 root calls with the comparator's roll-outs; re-run) | projected by Stage 0′ (§18) | TASK-093 Stage P: 768 calls in 281 s |
| **Total** | **the DRAFT's about 0.5 GPU h (0.4–0.7); Stage 0′'s projection in §18; ceiling 1.0 GPU h** | |

- **Per-job cap 1 800 s.** A job that reaches it exits; in the gated stage that is K-VOID (or
  L-VOID for an M or S job).
- **Stage 0's projection.** Stage 0 times its jobs and projects the gated stage. If the projection
  exceeds **0.6 GPU h**, or Stage 0 itself exceeds 0.4 GPU h, it escalates before the freeze
  (S0-ESCALATE). **Stage 0′** (§17, §18) projects again at 128 root seeds from Stage 0's
  development jobs (K's goals and arms × 4; M's roots and ranking × 8, for 4 × the roots and 2 ×
  the branches; S × 4, plus 4 × Stage 0′'s comparator job) with the same 0.6 GPU h limit, and
  counts Stage 0's and Stage 0′'s GPU time together against 0.4 GPU h.
- **The ceiling.** If the GPU time of all TASK-094 jobs (Stage 0, Stage 0′ and gated) reaches **1.0 GPU h**
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
- **Rs's window holds no lift.** At Rs the 8-step window lies inside the close (27 close commands
  remain), so the object does not rise in it; the ranking at Rs is of the hand and scene during the
  close, not of a held lift. Only R2's window covers the start of the lift.
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
  L-UNREAD (when neither root type is read), and the per-level lift rates at the roots are reported.
- **Cost on requests (S).** CEM scores its candidates' requests, as in the closed loop, and the
  comparator (R27.183 (4)) scores the plan and the substituted plan on requests too; the executed
  grasp (ℓ) is read through the rate clip. The DRAFT compared CEM's cost on requests with the
  squeeze branch's cost on applied commands, which are not the same object; that comparison is
  reported only. The substituted plan keeps the plan's arm channels, so a miss says that G prefers
  the squeeze's closing requests *with this plan's arm motion*, not that it prefers the scripted
  squeeze branch. G was trained on `play-v1`, whose grasp requests never exceed the 2 / 11 rate
  limit, so at Rs both the plan's early requests (first request median 0.94–0.97, executed 0.636)
  and the substituted plan's (0.8, then +1, executed 0.636, 0.818, then 1.0) lie off G's training
  support. The comparison is symmetric in this respect, but a miss or a partial preference at Rs
  is G's reading of above-clip requests, not of the executed grasp.
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
2. A Stage 0 PR (§14), reviewed; it ended S0-ESCALATE (§16), and the ruling (§17) added a Stage 0′
   PR (§18), reviewed. **The protocol is frozen at the Stage 0′ merge** if Stage 0′ passes.
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
    at least 28 / 32 roots;
  - n_d at each root type (reported; if n_d < 8 of 32 at both root types, Stage 0 discloses this
    and escalates, because M would most likely not be read);
  - G's wins and S's measures (reported, not gated; they set nothing);
- the power recompute of §8.4;
- a debug smoke on 94500–94599 of every step, timed, for §10.2's projection.

**Stage 0 may change only:** the per-run process count, and the split of a step into more, smaller
jobs. Each change is recorded. A failed check of the list above is S0-ESCALATE, not a change.
**Every level, offset, root type, bar, margin and row is frozen here** and is not changed by any
development number. The protocol was to freeze at the Stage 0 merge, but Stage 0 ended S0-ESCALATE
(§16); the ruling (§17, R27.183) amended M's root count and S's ℓ and comparator before the freeze.

**Stage 0′** (R27.183 (5); §18) is its own PR. It draws **no new development cohort** and runs:

- the code and tests for S's ℓ and comparator, the 128 gated root seeds 946000–946127, the retired
  range 942000–942063 (refused by every seed check), the positive control at 112 / 128 and the
  executed-difference limit at 16 / 128;
- ℓ recomputed on Stage 0's stored development plans (CPU);
- the comparator's costs on Stage 0's stored development plans at 940000–940031: one GPU job,
  `run_task094.py substitute` (reported; it sets nothing), with the squeeze branch's requests
  checked against its recorded applied grasp;
- the hash checks (`verify`) and the seed check with 946000–946127 on every ref and in the evidence;
- the budget projection at 128 root seeds (§10.2) and TASK-094's GPU time so far;
- **§14's n_d rule read at the gated root count:** Stage 0′ escalates (S0-ESCALATE) only if the
  development rates project n_d below 16 at both root types at 128 roots (Rs: 6 / 32 × 128 = 24),
  or if a check above fails, Stage 0 and Stage 0′ together exceed 0.4 GPU h, or the gated stage
  projects above 0.6 GPU h. **The protocol is frozen at the Stage 0′ merge** if Stage 0′ passes.

## 15. Rulings this document records (R27.160–R27.176; DRAFT; decided by Claude under owner delegation)

- **R27.160** — **TASK-094's DRAFT preregistration is this document.** It is measurement only: no
  controller, no training, no cost or planner variant, no world-model closed loop. Checked against
  every fired clause (§2), it engages none; R27.154 stands.
- **R27.161** — **Part K** (§4.1): TASK-088's IK follower with its closing command held at +1.0,
  0.9, 0.8, 0.7, s\* and 0.6 (reported, and part of K-NULL's condition), K-open as a degeneracy
  reference, and the palm offsets v_x and v_y of 1.2 cm at +1.0 and s\*, on 128 paired gated
  resets. The level 0.6 and the second offset direction are added to R27.156's list as reported
  arms.
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
  3 / 128, or a check), K-NULL (the upper ends of the two-sided 95 % reset-bootstrap percentile
  intervals (§4.2) of both Δ and Δ_{0.6}, the K-1.0 minus K-s\* and the K-1.0 minus K-0.6 lift
  rates, below δ_K = 0.15; salt 9014), K-CAUSAL (Δ ≥ 0.15 and p < 0.01), K-UNCLEAR (including Δ's
  bound below δ_K with Δ_{0.6}'s not).
- **R27.166** — **The L rows and bars** (§5.5, §6, §7.1): the ceiling at 0.75 (TASK-089's
  threshold); G at ⌈(0.5 + 0.75 (ĉ − 0.5)) n_d⌉ on two of three seeds; the search miss at
  ⌈0.25 n_d⌉ on two of three seeds; the positive control at 56 / 64 (28 / 32 at Stage 0; TASK-089's
  A-K0 found the analogous contrasts at 1.000); L-VOID, L-UNREAD, L-COST, L-MODEL, L-SEARCH,
  L-MIXED, first match over the read root types.
- **R27.167** — **Routing** (§7.2, §7.3). No row admits a controller. K-NULL pauses the grasp line
  (R27.157). **K-CAUSAL with L-COST is put forward as candidate R26.33 evidence for option C**, and
  K-CAUSAL with L-SEARCH for a planner ruling; neither is ruled met here.
- **R27.168** — **No abandonment clause** (§7.2).
- **R27.169** — **Power** (§8; development): n = 128 for K, 64 root seeds for M and S, with the
  misread rates disclosed (the two-of-three seed figures assume independent seeds, which they are
  not, so they are an approximation); Stage 0 recomputes K's table with 20 000 simulations,
  including K-NULL's joint condition.
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

## 16. Stage 0 record (R27.177–R27.182; decided by Claude under owner delegation)

Everything below is Stage 0: checks, development K on resets 94100–94131, development M and S on
roots 940000–940031 (the squeeze, s\* and control branches only), the power recompute and a debug
smoke on debug seeds 94500–94599. **Stage 0 ends S0-ESCALATE** on one rule (§14: n_d below 8 of 32
at both root types). The gated K resets 941000–941127 and root seeds 942000–942063 were not
touched. Every number here is development: one run, simulation only, 32 resets or roots, scripted
probes; none is a learned result, none sets a bar, and K's development counts are reported only.
Evidence: `~/develop/emai/evidence/task094-s0/` (`SHA256SUMS` sha256 `5c944848…77b1`); root and
goal frames in `outputs/task094-s0{,-debug}/` on the data disk.

**Revisions** (from the reports and `gpu_run.log`; every report records `tracked_tree_dirty:
false`, and every job ran in its worktree's root, where `third_party/le-wm` is):

- `bf6dc58` (the code): the debug smoke (from the implementation worktree), the first opt-in render
  test, and, from a clean run tree at that commit (`worktrees/task094-s0run`, with the data,
  checkpoint, `third_party` and output links of `worktrees/task093-run`), `verify`, `seeds`,
  `power` and the development stage (`scripts/run_task094_stage.sh dev`).
- `e94e2bf` (K's plumbing check corrected; §16.4): the first Stage 0 summary, kept as
  `summary-0.at-e94e2bf-projection-incomplete.json`.
- `8bd74ca` (the projection counts completed jobs that exited VOID): `summary-0.json`, the record's
  source; the full test suite and the opt-in render tests.
- The record commit changes documentation only.

### 16.1 Code

- `src/embodied_jepa/grasp_closure_locus.py` (NumPy at import): the cohorts and seeds; the K arms
  (`HeldIKController`, TASK-088's `IKController` with its closing target replaced by the arm's
  level, the frozen code's arithmetic otherwise; the displaced goal); K's plumbing check, measures,
  paired contrasts (exact one-sided McNemar; one shared set of 10 000 reset resamples from
  `default_rng([9014, n])`, `np.percentile`'s linear percentiles) and rows; the root types, the
  branch builder (TASK-089's `_Demo` run from the reset, re-checked bitwise, frames and states kept
  for the 8 window steps only), the executed-difference arithmetic, the discordant roots, wins,
  bars and positive control; S's executed level (TASK-093's `project_grasp` from the root's
  recorded last accepted hand targets), measures and bar; the L rows, the task row and the S0 row;
  the power recompute; sha256 pins of every earlier file TASK-094 imports (TASK-093's pins,
  TASK-093's modules and scripts, `run_tools.py`, `devices.py`).
- `src/embodied_jepa/grasp_closure_locus_runtime.py` (torch): M's roll-out and ceiling costs
  through the planner model's own `distance`, with the latent, pose, arm, hand and per-hand-joint
  terms (checked to add up to the planner's cost); S's CEM call, `pr.a2_plan`'s call with the best
  candidate's cost kept.
- `scripts/run_task094.py` (`verify`, `seeds`, `power`, `goals`, `k-run`, `roots`, `rank`,
  `search`), `scripts/summarize_task094.py` (stages 0 and gated) and `scripts/run_task094_stage.sh`
  (stages gated, dev, debug; the gated stage stops at the 1.0 GPU h ceiling, counting Stage 0's
  time from its summary). Every job refuses another working directory or a missing
  `third_party/le-wm`, loads and checks its goals or roots and, for M and S, the three G
  checkpoints and TASK-087's DINOv2 encoder (preflight, CPU) before its ledger claim; CPU tests fail
  if a claim comes first.
- `tests/test_grasp_closure_locus.py`, 38 tests: the follower bitwise `IKController` at +1.0, the
  held level after the ramp with every other channel unchanged, the displaced goal, the plumbing
  check (including an episode stopped before the hand reaches c), K's rows in first-match order,
  the band and the bootstrap, the root types' step counts and the executed-difference arithmetic
  (≥ 7 of 8 at Rs for every s\* below 0.818, 8 at R2), the squeeze branch's reproduction of the
  demonstrator (graphics opt-in), wins, bars, positive control, S's measures and executed level, the
  L rows (first match, n_d < 16), the S0 and task rows, the power model's marginals and rule, the
  runtime's decomposition and the A2 call (small model, CPU), the seed ranges, the pins, the
  working-directory guard, the ledger claim's place and the core-import check. The whole suite at
  `8bd74ca`: 2 494 passed, 35 skipped (graphics opt-in, MPS unavailable, the LeRobot-source
  opt-in); with `JEPA_TEST_RENDER=1` this file's 38 tests passed.
- No existing file changes except CLAUDE.md's entry points and this record.

### 16.2 Checks

| Check | Result |
| --- | --- |
| **Hashes** (`verify`) | G-87100–87102 against their pins, reports and evidence copies; the feature report and store files; the projection; the pinned moments; TASK-087's implementation hash; λ = 2.0552; the step-mean setting; the rate limit 2 / 11; CEM (8, 300, 10, 30, 0.05); the levels, offsets, branches and Rs's 8 commands; every frozen file: **all pass**. G-87102 has `last_two_triggered: true` |
| **Seeds** (`seeds`; 121 remote-tracking refs, 93 local branches and HEAD) | no token in 94100–94131, 94500–94599, 940000–940031, 941000–941127, 942000–942063, 9014 or 9015 outside TASK-094's files. In the evidence: 12 tokens, 94100 and 94500 as TASK-077 training-update indices; none in a seed field |
| **K plumbing** (every arm, 32 development resets; §16.4) | requests bitwise the frozen ramp towards c; after the ramp every request equals c; the applied grasp within 2.3 × 10⁻⁷ of the rate-limited request path and bitwise c on every command once that path reaches c (8–10 commands into the grasp subgoal); the closing decision exactly the grasp and lift subgoals: **all 11 arms pass** |
| **K-1.0 against TASK-088's frozen follower** (an added Stage 0 check) | `ik` run on the same 32 resets: applied commands **bitwise equal** to K-1.0's on all 32 |
| **Displaced goals** | every offset arm's goals are the K-1.0 goals with every subgoal palm moved by exactly v, everything else bitwise the same: pass |
| **K-1.0 and K-open** (§4.3's band at Stage 0's proportions, 16–28 and 0 of 32) | K-1.0 23 / 32, K-open 0 / 32: pass |
| **Roots and the squeeze** (M) | every re-run root bitwise the draw's state and frame at both root types; every squeeze branch bitwise the demonstrator's grasp-subgoal state (Rs, after 27 commands) and lift-subgoal state (end): **0 problems** on 32 roots. Rs's root: the demonstrator's request 0.6 and last accepted target synergy 0.4545 on every root |
| **Executed difference** | squeeze and s\* differ at 7 (Rs) and 8 (R2) of the 8 window steps on all 32 roots; none excluded |
| **Positive control** (bar 28 / 32) | the ceiling ranks the squeeze above re-open (Rs) and release (R2) on **32 / 32** and **32 / 32** |
| **Ceiling across seeds** | identical under the three seeds' (pinned) standardisations |
| **S determinism** (2 roots × 2 root types × 3 seeds) | 12 / 12 re-runs bitwise |
| **Debug smoke** (4 debug resets and 4 debug roots with all six branches; not results) | every K arm and the frozen `ik` ran; plumbing and displacement pass; K-1.0 bitwise `ik`; K-1.0's and K-s\*'s determinism re-runs bitwise; M and S ran with every branch; S's plans equal `pr.a2_plan`'s bitwise (`--check-a2`) |
| **Power** (§8.4) | passes (§16.3) |
| **n_d** (§14: escalate if below 8 of 32 at both root types) | **Rs 6, R2 1: fails** |
| **Budget** (§10.2) | Stage 0: 36 `gpu_run` jobs, 387 s (0.108 GPU h; the summary's 371 s precede the final render-test run), below 0.4. The gated stage projected from the development jobs (K's goals and arms × 4, two determinism jobs, M's roots and ranking × 4, S × 2): 924 s, **0.257 GPU h**, below 0.6 |

### 16.3 Power (§8.4; 20 000 simulations per cell, 10 000 bootstrap resamples each, CPU)

The joint condition needs a model of the three arms together, which §8.4 does not fix. Stage 0 uses
K-1.0 lifting with 0.69, the cells (Δ + d₀, d₀) and (Δ_0.6 + d₀, d₀), and K-s\* and K-0.6
**independent given K-1.0's outcome** (the rule's model), and reports the most correlated case
(**nested**: K-0.6 fails wherever K-s\* fails) beside it. Generator
`default_rng([9014, 94, n, …])`.

| Δ = Δ_0.6 | d₀ | K-CAUSAL | K-NULL (joint) | K-UNCLEAR |
| --- | --- | --- | --- | --- |
| 0.00 | 0.03 / 0.08 | 0.000 / 0.000 | **1.000 / 0.978** | 0.000 / 0.022 |
| 0.05 | 0.03 / 0.08 | 0.001 / 0.005 | 0.841 / 0.528 | 0.158 / 0.467 |
| 0.10 | 0.03 / 0.08 | 0.066 / 0.119 | 0.133 / 0.071 | 0.801 / 0.810 |
| 0.15 | 0.03 / 0.08 | 0.475 / 0.478 | 0.002 / 0.002 | 0.523 / 0.519 |
| 0.20 | 0.03 / 0.08 | 0.877 / 0.827 | 0.000 / 0.000 | 0.123 / 0.173 |
| 0.30 | 0.03 / 0.08 | 0.999 / **0.997** | 0.000 / 0.000 | 0.001 / 0.003 |
| 0.50 | 0.03 / 0.08 | 1.000 / 1.000 | 0.000 / 0.000 | 0.000 / 0.000 |

- **The rule passes:** K-CAUSAL at Δ = 0.30 (d₀ = 0.08) 0.997 ≥ 0.95; joint K-NULL at
  Δ = Δ_0.6 = 0 1.000 (d₀ = 0.03) and 0.978 (d₀ = 0.08) ≥ 0.95. Nested model: 1.000 and 0.987, and
  K-CAUSAL 0.999 and 0.997.
- **What the joint condition costs** (disclosed): with a small real loss at both levels K-NULL is
  rarer than §8.1's s\*-only table says (0.841 / 0.528 at 0.05 against 0.920 / 0.698; 0.133 /
  0.071 at 0.10 against 0.349 / 0.240). With Δ_0.6 = Δ + 0.10 K-NULL falls to 0.350 / 0.223 at
  Δ = 0, and K-CAUSAL is unchanged (it reads Δ only).

### 16.4 Incidents and corrections

- **K's plumbing check (job reading).** At `bf6dc58` the job's check also required every
  episode's applied grasp to *reach* c. Nine development episodes of the four offset arms
  (K-off-x-1.0: 94111, 94120; K-off-x-s\*: 94111; K-off-y-1.0: 94102, 94111, 94120; K-off-y-s\*:
  94102, 94111) were stopped by the embodiment's measured-joint-velocity stop 9–10 commands into the
  grasp subgoal, before the rate limit could let the hand reach c, so those four jobs exited VOID
  (3). §4.3 asks that the applied grasp equal c "once the rate limit allows", which these episodes
  meet with no such command. `e94e2bf` makes the check follow that wording (the rate-limited path of
  the requests decides when c is allowed; the applied grasp must follow that path and then equal c
  bitwise), and the summarizer re-reads the check from each episode's recorded requests, subgoal
  indices and applied commands (`replumb`), reporting the job's reading beside it. On every
  development and debug episode of every arm the applied grasp reached c on exactly the command the
  path did. No run was repeated and no count changed. **The reviewer should confirm that this is a
  correction of the check, not a failed check** (§14 makes a failed check S0-ESCALATE); Stage 0
  escalates on the n_d rule either way.
- **The projection.** At `e94e2bf` the summarizer's projection skipped jobs that exited VOID, so it
  was incomplete (and the summary escalated on it as well); `8bd74ca` counts them, since they ran to
  completion.
- **Stage 0 readings the protocol leaves open** (disclosed, set nothing): §4.3's band and K-open
  maximum at Stage 0's 32 resets are the same proportions rounded inwards (16–28 and 0); the
  executed-difference check at Stage 0 allows 4 of 32 excluded roots (8 of 64); the joint power
  model (§16.3); and the added check of K-1.0 against TASK-088's frozen follower. The debug smoke
  ran M with all six branches, to time them.
- No job reached its cap; no `- v2d:` request was on BOARD.md before any job; the GR00T server,
  `/mnt/data/emai/{archive,models,cache}`, `worktrees/task087-run` and other worktrees were not
  touched. Process counts: 8 workers for K and the roots (4 in the debug smoke), as planned.

### 16.5 Development measurements (reported; one run, simulation only; they set nothing)

**K** (resets 94100–94131, 32 paired resets; exact 95 % intervals; s\* median 0.674, p10–p90
0.651–0.693):

| Arm | Lifts / 32 | Interval | Contact | Stopped | Slip | Measured hand, close and lift (median) | Grasp subgoal timed out |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K-1.0 (`ik` 23 / 32, bitwise the same) | 23 | 0.53–0.86 | 32 | 1 | 1 | 0.66 | 19 |
| K-0.9 | 23 | 0.53–0.86 | 32 | 1 | 1 | 0.60 | 21 |
| K-0.8 | 24 | 0.57–0.89 | 32 | 0 | 0 | 0.54 | 23 |
| K-0.7 | 23 | 0.53–0.86 | 32 | 0 | 0 | 0.48 | 29 |
| K-s\* | 21 | 0.47–0.81 | 32 | 0 | 1 | 0.47 | 30 |
| K-0.6 | 19 | 0.41–0.76 | 32 | 1 | 2 | 0.43 | 30 |
| K-open | 0 | 0.00–0.11 | 0 | 0 | 0 | −1.00 | 32 |
| K-off-x-1.0 | 19 | 0.41–0.76 | 30 | 4 | 2 | 0.66 | 14 |
| K-off-x-s\* | 16 | 0.32–0.68 | 30 | 3 | 2 | 0.47 | 28 |
| K-off-y-1.0 | 7 | 0.09–0.40 | 23 | 6 | 2 | 1.00 | 23 |
| K-off-y-s\* | 8 | 0.11–0.43 | 25 | 3 | 0 | 0.66 | 28 |

The switching rule's hand-RMS test makes lower held levels time out on the grasp subgoal more
often, so the dose arms also differ in grasp-phase duration (§4.2 measure).

- K-1.0 minus K-s\*: 0.0625 (3 against 1 discordant; reset-bootstrap 95 % −0.0625 to 0.1875);
  K-1.0 minus K-0.6: 0.125 (5 against 1; 0.000 to 0.281); K-1.0 minus K-off-y-1.0: 0.50 (16
  against 0). These are 32 development resets; they are not K's row and §4.3's rows are not read
  on them.

**M** (roots 940000–940031; squeeze, s\* and control branches; 43 rejected draws; targets cube 17,
can 9, apple 6):

| Root type | Squeeze lifts | s\* lifts | Control lifts | **n_d** | Ceiling wins on discordant | G wins (87100, 87101, 87102) | Ceiling squeeze < s\* on all roots | Positive control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rs | 32 | 26 | 0 | **6** | 5 / 6 | 6, 6, 3 of 6 | 30 / 32 | 32 / 32 |
| R2 | 32 | 31 | 0 | **1** | 1 / 1 | 1, 1, 1 of 1 | 28 / 32 | 32 / 32 |

- Every s\* branch that failed kept grasp contact; at Rs the target rose 0.7–2.2 cm, at R2 (one
  root) 6.0 cm without holding 20 commands.
- Term decomposition (medians on the discordant roots, reported): at Rs the ceiling's cost of the
  squeeze and s\* branches is 0.76 against 0.86, the hand part of the pose term 0.15 against 0.29
  (the pose term 0.17 against 0.31), the latent term 0.60 against 0.58.

**S** (the frozen planner, one call per root, type and seed): no search miss on any discordant root
at either type for any seed; the plan holds partial and G rates it at least as well as the squeeze
branch ("partial preferred") on 5, 6 and 6 of Rs's 6 discordant roots and on R2's one. Plans that
squeeze (ℓ ≥ 0.9) on all 32 roots: Rs 6, 4, 1; R2 4, 6, 0. Median ℓ: Rs 0.70, 0.72, 0.70; R2 0.69,
0.77, 0.71. At Rs the first command requests a median grasp of 0.94–0.97 and executes 0.636 (the
rate limit).

These S figures do not show that G prefers a partial closure. CEM's best candidate, scored on its
requests over all 14 channels, cost less than the scripted squeeze branch's applied commands on
every call (192 / 192; ratio median 0.62–0.78, maximum 0.94), so a miss as §6 defines it could not
occur, and "partial preferred" here means only ℓ < 0.9. ℓ is read at step 8, the end of the
horizon, where the plans' requests were lowest and most dispersed (median 0.52–0.69, IQR
0.17–0.31, against about 0.8–0.9 at steps 3–7). By the median executed grasp over steps 3–7, 4, 3
and 2 Rs plans and 14, 16 and 14 R2 plans of 32 squeeze. The median ℓ of about 0.70 is not a
reproduction of the WM arms' 0.53–0.73 (§1.1).

### 16.6 The escalation and what the ruling decides (R27.181)

§14's n_d rule fired: at the demonstrator's exact states the s\* branch lifts where the squeeze
does on 26 of 32 Rs roots and 31 of 32 R2 roots, so the discordant roots M reads are rare (6 and 1
of 32). At these rates 64 gated root seeds would give about 12 (Rs) and 2 (R2), below the 16 at
which a root type is read, so M and S would most likely end L-UNREAD after a K-CAUSAL. K is not
part of this rule; its development numbers are reported only. As §7.2 says, S0-ESCALATE routes to a
ruling before the freeze. This record does not decide it. The options it sees, without ranking
them:

- run the gated stage as preregistered, accepting that M is most likely unread (K still answers
  the physics question; L-UNREAD is a named row);
- amend M before the freeze, for example more root seeds (at Rs's development rate, 128 roots would
  give about 24 discordant roots, at roughly 0.1 GPU h more) or roots where s\* is marginal, each a
  protocol change with its own power and seed checks and a new Stage 0 reading of n_d;
- run K alone and drop M and S;
- close TASK-094 without a gated stage.

Spent: the development cohorts 94100–94131 and 940000–940031 (ledgered). Untouched: 941000–941127
and 942000–942063; of the debug range 94500–94599 nine seeds were used (94500–94503,
94550–94553 and 94599, the render test), and debug seeds are never counted.

**R7 does not change** (R27.176). Nothing here is a grasp success or a learned result.

## 17. Stage 0 escalation ruling (R27.183–R27.186; decided by Claude under owner delegation)

S0-ESCALATE (§16.6) is resolved by **option 2, extended** (DECISIONS (bk)), adopting items 1–7 of
the amendment proposed in the independent review of PR #226 word for word (R27.183):

1. **Root seeds.** M and S use 128 gated root seeds **946000–946127**; 942000–942063 are retired
   (named, never generated, never run, not reused). S's determinism re-run moves to
   946000–946001. n_d(t) ≥ 16 is unchanged; positive control 112 / 128, at most 16 / 128 roots
   excluded by the executed-difference check.
2. **R2 stays, reported.** Every R2 check still voids M (L-VOID); at the development rate its
   expected n_d is 4, so it is expected to be unread.
3. **S's ℓ** is the median of the plan's executed right grasp over window steps 3–7 (cumulative
   clip as before); step 8 and the full path are reported; squeeze at ℓ ≥ 0.9, unchanged.
4. **S's search miss**: the plan holds partial and G-i's cost of the plan with its right-grasp
   requests replaced by the squeeze branch's over steps 1–8 is strictly lower than the plan's cost;
   *partial preferred* otherwise when partial. Bar ⌈0.25 n_d⌉ on two of three seeds, unchanged;
   the previous comparison is reported.
5. **Stage 0′**, with no new development cohort: code and tests for 3–4, ℓ recomputed on the stored
   development plans, the substituted costs on 940000–940031 (one GPU job, reported), the seed
   check of 946000–946127, and the budget projection (about 0.35 GPU h, within 0.6). It escalates
   only if the development rates project n_d below 16 at both root types at 128 roots.
6. **K unchanged** (941000–941127, every arm, band and bar).
7. **R7 unchanged.**

**Why (R27.184).** Before the freeze, before any gated data, on development evidence, through the
ruling §7.2 names; no level, offset, root type, bar, margin or row changes, and §14's frozen list
does not include the root count. **Disclosed:** items 3 and 4 are motivated by the development plans
(step-8 dispersion; CEM's best candidate below the squeeze branch's cost on 192 / 192 calls).

**For the gated ruling to weigh (R27.185).** The development K pattern (Δ_s\* 0.0625 [−0.06, 0.19],
Δ_0.6 0.125 [0, 0.28]) makes K-UNCLEAR the likely gated K row, in which case M and S are reported,
not read; and the +y palm-placement effect (−16 / 32) is larger than any closure effect. Both are
development numbers and set nothing.

**Status (R27.186).** The protocol stays **DRAFT, not frozen**. The edits of §5, §6, §9, §10.2 and
§14 that carry items 1–5 are made in the Stage 0′ PR, which is reviewed independently and freezes
the protocol at its merge. **R7 does not change** (R27.176).

## 18. Stage 0′ record (R27.187–R27.192; decided by Claude under owner delegation)

Everything below is Stage 0′ (§14, §17; R27.183 (5)). **No new development cohort was drawn.** It
ran CPU checks and **one GPU job** on Stage 0's stored development plans and roots (940000–940031),
which are reported and set nothing. **Stage 0′ ends S0-PASS, and the protocol is frozen at the merge
of its PR.** The gated cohorts 941000–941127 and 946000–946127 were not touched; 942000–942063 are
retired and were never generated. Every development number here is one run, simulation only,
32 roots, scripted branches and three model seeds of one training run (G-87102
`last_two_triggered`); none is a learned result and none sets a bar. Evidence:
`~/develop/emai/evidence/task094-s0p/` (`SHA256SUMS` sha256 `86e5f46f…2075`); the reports also stay in
`outputs/task094-s0p/` on the data disk.

**Revisions** (every report records `tracked_tree_dirty: false`; the jobs ran in the root of a clean
run tree, `worktrees/task094-s0prun`, detached, with the data, checkpoint, `third_party` and output
links of `worktrees/task094-s0run`):

- `ef89953` (the code and the amendment of §5, §6, §9, §10.2 and §14): the first `verify` and
  `seeds`, kept as `verify-task094.at-ef89953.json` and `seeds.at-ef89953.json` (§18.4).
- `f9d78e5` (the seed check reads a line that names TASK-094 as TASK-094's own): `verify`, `seeds`,
  the GPU job `run_task094.py substitute` (through `scripts/run_task094_stage.sh 0p`) and
  `summary-0p.json`, the record's source.
- The record commit changes documentation only.

### 18.1 Code

- `grasp_closure_locus.py`: the gated root seeds 946000–946127 (`GATED_ROOTS_N` 128) and
  `RETIRED_ROOTS` 942000–942063, which `check_seed` refuses under every cohort name; the positive
  control 112 / 128 (unchanged 7 / 8) and the executed-difference limit 1 / 8 (16 / 128;
  `exec_diff_limit`); `window_level` (ℓ, the median executed grasp over window steps 3–7);
  `squeeze_requests` and `substituted_plan` (the comparator); `search_measures` with the new miss
  and, reported, the DRAFT's comparison; `s0p_row` (§14's n_d rule at 128 roots); the read
  probabilities (reported).
- `grasp_closure_locus_runtime.py`: `substitution_costs`, G-i's step-averaged cost of the plan and
  of the substituted plan, on requests, from one roll-out call.
- `scripts/run_task094.py`: `search` records ℓ, step 8's executed grasp, the comparator's costs and
  (reported) the squeeze requests' path deviation, and its determinism re-run includes the
  comparator's costs; `seeds` checks 946000–946127 and 942000–942063; `roots` refuses a retired
  seed; the new `substitute` job (Stage 0′ only, development roots only; it loads the stored roots
  and search, checks their digests and the models before its ledger claim).
- `scripts/summarize_task094.py --stage 0p`; the gated summary needs Stage 0′'s S0-PASS.
  `scripts/run_task094_stage.sh 0p`; the gated stage reads Stage 0′'s summary and counts Stage 0's
  and Stage 0′'s GPU time towards the 1.0 GPU h ceiling.
- `tests/test_grasp_closure_locus.py`: 42 tests (four new: the squeeze requests and the substituted
  plan, the comparator's costs against the planner's step-mean cost on a small CPU model, the
  Stage 0′ row and read probabilities, and the `substitute` job's ledger claim after its loads;
  the seed, ℓ, miss and bar tests extended). The whole suite at `f9d78e5`: 2 498 passed, 35 skipped (graphics opt-in, MPS unavailable, the LeRobot-source opt-in); the opt-in render tests were not re-run (the root builder is unchanged).

### 18.2 Checks

| Check | Result |
| --- | --- |
| **Hashes** (`verify` at `f9d78e5`) | every check passes (G-87100–87102 against their pins, reports and evidence copies; the feature store and projection; the moments; TASK-087's implementation hash; λ = 2.0552; the setting; the rate limit; CEM; the levels and offsets; every frozen file); G-87102 `last_two_triggered: true` |
| **Seeds** (`seeds`; 123 remote-tracking refs, 94 local branches and HEAD) | no token in 94100–94131, 94500–94599, 940000–940031, 941000–941127, **946000–946127**, the retired 942000–942063, 9014 or 9015 outside TASK-094's files and lines; the only line elsewhere is TASK-094's own row in `docs/JEPA_ZERO_SHOT_PLAN.md` (946000 and 946127, R27.183's text; §18.4). In the evidence: the same 12 tokens as at Stage 0 (94100 and 94500, TASK-077 training-update indices), none in a seed field |
| **Seed ranges in the code** | gated roots exactly 946000–946127; the retired range refused; both disjoint from every cohort and every earlier task's ranges |
| **The comparator's plumbing** (192 root, type and seed entries) | the squeeze branch's requests (`squeeze_requests`: 0.8 then +1 at Rs, +1 at R2), executed through the cumulative clip from each root's T₀, reproduce the squeeze branch's recorded applied right grasp over the window: maximum deviation 1.6 × 10⁻⁸ (tolerance 10⁻⁵) |
| **The plan's cost** | G-i's step-averaged cost of each stored plan, re-evaluated in the comparator's roll-out call, equals CEM's recorded best-candidate cost within 3.6 × 10⁻⁷ on all 192 |
| **Stage 0's records** | the stored roots and search are the ones Stage 0's summary read (digests); n_d re-read: 6 (Rs) and 1 (R2) |
| **§14's n_d rule at 128 roots** (escalate only if below 16 at both root types) | projected n_d **24 at Rs** (6 / 32 × 128) and 4 at R2: **passes** |
| **Budget** (§10.2) | Stage 0′: 1 `gpu_run` job, 6 s. Stage 0 and Stage 0′ together: 37 jobs, 393 s (**0.109 GPU h**, below 0.4). The gated stage projected at 128 root seeds: 1 266 s, **0.352 GPU h** (below 0.6): K's goals 72 s and arms 512 s (× 4), K's determinism 22 s, M's roots 424 s and ranking 56 s (× 8), S 156 s (× 4) and the comparator 24 s (4 × Stage 0′'s job) |

### 18.3 Development measurements (reported; one run, simulation only, 32 roots; they set nothing)

**n_d at 128 roots** (§8.2): at Rs's development rate (6 / 32) P(n_d ≥ 16) is 0.977, or 0.84 with
the rate's uncertainty integrated (Jeffreys posterior); at R2's (1 / 32) the expected n_d is 4 and
P(n_d ≥ 16) is about 0.05 with the uncertainty integrated, so R2 is expected to be unread, as R27.183
(2) says.

**ℓ recomputed** on Stage 0's stored plans (per seed G-87100, G-87101, G-87102; plans that squeeze,
ℓ ≥ 0.9, of 32 roots; the DRAFT's step-8 reading beside it):

| Root type | Squeeze by ℓ (steps 3–7) | Median ℓ | Squeeze by step 8 (DRAFT) | Median step 8 | Discordant roots that squeeze by ℓ |
| --- | --- | --- | --- | --- | --- |
| Rs | 4, 3, 2 | 0.849, 0.836, 0.826 | 6, 4, 1 | 0.70, 0.72, 0.70 | 0, 0, 0 of 6 |
| R2 | 14, 16, 14 | 0.890, 0.900, 0.894 | 4, 6, 0 | 0.69, 0.77, 0.71 | 1, 1, 1 of 1 |

At Rs the first command still requests a median 0.94–0.97 and executes 0.636 (the rate limit).

**The comparator** (the plan against the same plan with the squeeze branch's right-grasp requests,
G-i's step-averaged cost on requests):

| Root type | Misses on discordant roots (bar ⌈0.25 n_d⌉ on two of three seeds, not read here) | Misses on all 32 roots | Substituted plan cheaper, all 32 (any ℓ) | Substituted / plan cost, median |
| --- | --- | --- | --- | --- |
| Rs (n_d 6) | **0, 0, 3** of 6 | 0, 1, 6 | 1, 1, 6 | 1.024, 1.053, 1.015 |
| R2 (n_d 1) | 0, 0, 0 of 1 (the one root's plans squeeze) | 8, 4, 8 | 16, 11, 16 | 0.997, 1.005, 0.998 |

- **The DRAFT's comparison** (reported): CEM's best candidate cost less than the squeeze branch's
  applied commands on 192 / 192 calls, so it gave no miss on any root, as §16.5 recorded; by the
  step-8 ℓ it read "partial preferred" on 5, 6, 6 of Rs's discordant roots and 1, 1, 1 at R2.
- **What these show, and do not.** The new miss can fire: it did so on development roots, mostly
  for G-87102 (3 of Rs's 6 discordant roots, 6 of 32 Rs roots). At Rs G-87100 and G-87101 gave the
  substituted plan a higher cost than the plan (rated it worse) on all but one root (median cost
  ratios 1.02 and 1.05), so on
  these roots their plans' partial closures are mostly what G-i prefers given the plans' arm
  motion; at R2 the ratio is near 1. These are 32 development roots and 6 discordant ones; they do
  not estimate the gated rates, and no row is read on them.

### 18.4 Incidents, readings and deviations

- **The seed check (job reading).** At `ef89953` `seeds` classified hits by file only, so TASK-094's
  own row in `docs/JEPA_ZERO_SHOT_PLAN.md`, which quotes R27.183's 946000–946127, counted as a hit
  outside TASK-094 and would have escalated Stage 0′ on a check of TASK-094's own text. `f9d78e5`
  reads a matching line that names TASK-094 as TASK-094's own and reports it apart
  (`hits_on_lines_naming_task094`); `verify` and `seeds` were re-run there, and the first reports
  are kept. No GPU job ran at `ef89953`. The review should confirm that this is a correction of
  the check, not a loosening: a line naming TASK-094 elsewhere is a reference to TASK-094.
- **Readings this record makes** (they set nothing; the review should confirm them):
  - "The plan's cost" in R27.183 (4) is G-i's step-averaged cost of the plan **re-evaluated in the
    same roll-out call as the substituted plan**, so the two are compared under identical
    arithmetic; CEM's recorded best-candidate cost is reported beside it (they agree within
    3.6 × 10⁻⁷ on the development plans).
  - R2 is read only if its n_d reaches 16 (§5.4's rule, unchanged); R27.183 (2) expects it unread.
  - The comparator's plumbing check (the squeeze requests against the squeeze branch's applied
    grasp) is a Stage 0′ check. In the gated search job it is recorded per root and reported, not
    added to L-VOID's list (no row trigger is added).
  - The projection multiplies M's roots and ranking by 8 (4 × the roots, 2 × the branches), S by 4,
    and adds 4 × Stage 0′'s comparator job; it counts whole job durations, start-up included.
  - TASK-094's GPU time so far counts every `oej:task094-*` job in `gpu_run`'s log: Stage 0's 36
    (387 s) and Stage 0′'s one (6 s).
- No job reached its cap; no `- v2d:` request was on BOARD.md before the job; the GR00T server,
  `/mnt/data/emai/{archive,models,cache}`, `worktrees/task087-run` and other worktrees and evidence
  were not touched. Nothing was moved to the data disk, so `MANIFEST.tsv` was not written. Ledger:
  one new key, `substitute-dev-roots`.

### 18.5 The freeze

**Stage 0′ ends S0-PASS. The protocol is frozen at the merge of this PR**, with §5, §6, §9, §10.2
and §14 as amended (R27.183) and K as preregistered. The gated stage then runs once, at the merge
revision from a clean tree, with `scripts/run_task094_stage.sh gated <out>
outputs/task094-s0p/summary-0p.json`, after the V2D session is told (§10.1, §13). R27.185 stands:
the development K pattern makes K-UNCLEAR the likely gated K row, in which case M and S are
reported, not read; the ruling after the gated stage weighs it, and the palm-placement effect, then.

**R7 does not change** (R27.176). Nothing here is a grasp success or a learned result.

## 19. Gated stage record (R27.193–R27.199; decided by Claude under owner delegation)

The gated stage ran **once**, on 2026-10-15 between 14:07 and 14:25 UTC, after the V2D session had
paused and acknowledged it. It ran at the frozen revision `2eacacf` (#227) from a clean, detached
run tree (`worktrees/task094-run`, with the data, checkpoint, `third_party` and output links of
`worktrees/task094-s0prun` and its own `.venv`). The command was §18.5's,
`scripts/run_task094_stage.sh gated outputs/task094-gated outputs/task094-s0p/summary-0p.json`.
Before it, a CPU `verify` at `2eacacf` passed every check. The gated cohorts were K resets
941000–941127 and M and S root seeds 946000–946127. 942000–942063 stay retired and were never
generated.

Every number below is **gated, one run, simulation only**. K is a scripted IK follower with no
model. M and S use three model seeds of one training run (**G-87102 `last_two_triggered`**).
Nothing here is a grasp success by a world-model controller or a learned result.

Evidence: `~/develop/emai/evidence/task094-gated/` (`SHA256SUMS` sha256 `1569fe63…0e31`). The
reports also stay in `outputs/task094-gated/`. The record's source is `summary-gated.json`, read
as §19.1 sets out.

### 19.1 What the stage reported, and the two VOID events

The stage script exited **3 (`STAGE-gated-VOID`)**. As the script is written, it went on to the
summary after a VOID job. There were two events, and both are recorded, not fixed:

1. **The roots job exited VOID (3).** At root seed **946073** (an apple; the third draw, the
   first two rejected), all six re-runs from the reset to **Rs** differed from the draw's recorded
   Rs record (`same_record`: the measured 28-D state and the onboard frame, bitwise). The report
   does not say which of the two differed.
   - At the same root, the squeeze branch's own checks passed. Its state after 27 commands equals
     the grasp subgoal, and its end state equals the lift subgoal, bitwise. R2's re-runs matched.
   - So the trajectory from the re-run Rs reproduced the draw's later states. That the
     difference is confined to the Rs record (for example its frame) is **inferred, not
     measured**.
   - **Inferred, not measured:** `state28` is the measured joints, a function of the simulator
     state, so a bitwise-identical continuation makes a differing frame (EGL rendering) the
     likelier cause; no other GPU application ran (`gpu_run.log`). The record stores only the last
     re-run's Rs record, so the evidence cannot separate frame from state.
   - Every other root (127 of 128) and every other M or S check passed (below).
   - This is an L-VOID trigger of §7.1 ("a root's bitwise re-run"). §7.1 says that K's row
     stands, and that M and S run again only on fresh roots named by a ruling.
2. **The summary's K row reads K-VOID, for one reason only: "K-1.0: n = 4; K-s\*: n = 4".**
   `summarize_task094.py` applies its cohort-size check (n = 128) to every K report, including
   the two determinism re-run reports. §4.3 and §9 define those re-runs on exactly four resets
   (941000–941003). Both re-runs matched bitwise on all four resets.
   - §4.3's text lists the determinism re-run as a K-VOID trigger only if it is not bitwise. It
     does not list a size check of the re-run reports.
   - Run on the 11 arm reports and the goals alone, the same function finds no problem. So does
     the same function run on the determinism reports without the size argument.
   - Stage 0 ran its K summary without that check (`expect_n=None`), and the debug smoke was
     never summarised as gated. So the defect was not exercised before the gated stage.
   - The summarizer is part of the frozen code and is **not edited** here.

**K's row by §4.3's frozen text is K-NULL** (§19.2). **The summarizer's mechanical row is K-VOID.**
This record reads K-NULL, because the protocol's text defines the rows and its code implements
them; the size check is not in §4.3, contradicts §9, and fires whatever K's counts are. The
independent review of #228 confirmed the reading (R27.195): the frozen summarizer re-run on the CPU
reproduces `summary-gated.json` byte for byte, and a copy patched only at that check reads K-NULL
with every other field unchanged (`~/develop/emai/evidence/task094-gated-recompute/`; `SHA256SUMS`
sha256 `3d3b0380…3d63`).

### 19.2 K (128 paired gated resets 941000–941127; exact 95 % intervals)

The goals were drawn with 146 rejected draws. Targets: cube 51, can 46, apple 31. s\* median
0.679 (p10–p90 0.649–0.692).

| Arm | Lifts / 128 | Interval | Contact | Stopped | Slip | Measured hand, close and lift (median) | Grasp subgoal timed out |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **K-1.0** | **88** | 0.600–0.766 | 121 | 10 | 4 | 0.66 | 60 |
| K-0.9 | 89 | 0.608–0.774 | 121 | 8 | 2 | 0.60 | 69 |
| K-0.8 | 88 | 0.600–0.766 | 119 | 7 | 2 | 0.54 | 79 |
| K-0.7 | 82 | 0.551–0.723 | 118 | 10 | 3 | 0.48 | 105 |
| **K-s\*** | **84** | 0.567–0.738 | 118 | 9 | 2 | 0.47 | 111 |
| **K-0.6** | **81** | 0.543–0.716 | 119 | 7 | 1 | 0.43 | 118 |
| K-open | 0 | 0.000–0.028 | 0 | 0 | 0 | −1.00 | 128 |
| K-off-x-1.0 | 82 | 0.551–0.723 | 120 | 3 | 4 | 0.67 | 82 |
| K-off-x-s\* | 69 | 0.449–0.628 | 120 | 4 | 6 | 0.48 | 115 |
| K-off-y-1.0 | 31 | 0.171–0.326 | 91 | 14 | 2 | 1.00 | 105 |
| K-off-y-s\* | 26 | 0.137–0.283 | 99 | 12 | 5 | 0.66 | 111 |

**Paired contrasts against K-1.0** (Δ is the rate difference. "Discordant" counts the resets
where only K-1.0 lifted, then those where only the other arm lifted. p is the exact one-sided
McNemar test. The interval is the two-sided 95 % reset-bootstrap percentile interval, 10 000
resamples, salt 9014.)

| Other arm | Δ | Discordant | p | Bootstrap 95 % |
| --- | --- | --- | --- | --- |
| K-0.9 | −0.008 | 4 / 5 | 0.75 | −0.055 to +0.039 |
| K-0.8 | 0.000 | 5 / 5 | 0.62 | −0.047 to +0.047 |
| K-0.7 | +0.047 | 11 / 5 | 0.11 | −0.016 to +0.109 |
| **K-s\*** | **+0.031** | **10 / 6** | **0.23** | **−0.031 to +0.094** |
| **K-0.6** | **+0.055** | **14 / 7** | **0.095** | **−0.016 to +0.125** |
| K-open | +0.688 | 88 / 0 | 3 × 10⁻²⁷ | +0.609 to +0.766 |
| K-off-x-1.0 | +0.047 | 18 / 12 | 0.18 | −0.039 to +0.133 |
| K-off-x-s\* | +0.148 | 29 / 10 | 0.0017 | +0.055 to +0.242 |
| K-off-y-1.0 | +0.445 | 60 / 3 | 5 × 10⁻¹⁵ | +0.352 to +0.539 |
| K-off-y-s\* | +0.484 | 64 / 2 | 3 × 10⁻¹⁷ | +0.391 to +0.570 |

The offset arms against their s\* twin: K-s\* minus K-off-x-s\* is +0.117 (27 / 12; p = 0.012;
+0.023 to +0.211), and K-s\* minus K-off-y-s\* is +0.453 (61 / 3; p = 2 × 10⁻¹⁵; +0.359 to +0.547).
The +1.0 pairs are in the table above (K-1.0 against K-off-x-1.0 and K-off-y-1.0).

**§4.3, first match:**

1. **K-VOID — no condition holds.**
   - K-1.0 lifted 88, inside the band of 63–114.
   - K-open lifted 0, not above 3.
   - The plumbing passed on all 11 arms, both in the job's reading and in the summarizer's
     re-reading. The offset goals are the K-1.0 goals displaced by exactly v.
   - The goal digests, the revision, the clean tree and the frozen files matched.
   - No cohort ran twice, no job hit its cap, and there was no budget stop.
   - The determinism re-runs of K-1.0 and K-s\* on 941000–941003 matched bitwise.
   - The summarizer's "n = 4" is §19.1 (2).
2. **K-NULL — holds.** Δ_{s\*}'s bootstrap upper bound is 0.094 and Δ_{0.6}'s is 0.125. Both are
   below δ_K = 0.15.

**K's row: K-NULL** (by §4.3's text, confirmed, R27.195; the frozen summarizer printed K-VOID on a
size-check defect, §19.1). At the scripted follower's exact pose, holding the closure at s\* or at
0.6 lost less than 0.15 of the lifts against the +1 squeeze on these 128 resets. The point estimates are +0.031 and +0.055, and
neither is significant.

**The palm-offset arms, reported beside K-NULL as §7.2 asks.** A rigid 1.2 cm palm offset in +y
cost far more than any closure level:
- **+y:** K-1.0 minus K-off-y-1.0 is +0.445 (+0.352 to +0.539), and K-s\* minus K-off-y-s\* is
  +0.453.
- **+x:** at +1.0 the offset cost +0.047 (−0.039 to +0.133, not significant). Combined with s\*
  it cost +0.148 against K-1.0 (+0.055 to +0.242; p = 0.0017), and +0.117 against K-s\*
  (+0.023 to +0.211).

On these resets the loss from placement is larger than the loss from closure. It also depends on
the direction of the offset.

Computed by the review of #228, post hoc and not preregistered (no row, no bar): under the +x
offset, holding at s\* lost +0.102 against +1.0 (K-off-x-1.0 82 against K-off-x-s\* 69; discordant
17 / 4, exact one-sided McNemar p = 0.0036, reset bootstrap 95 % +0.039 to +0.172, salt 9014); under
+y +0.039 (6 / 1, p = 0.063, 0.000 to +0.078). K-NULL is a finding at the follower's exact pose; it
does not show that partial closure is harmless when the palm is misplaced, and the pause ruling
should weigh the two together.

**What these figures do not show:**
- The offset is a dose of two directions of one size, imposed on the demonstrator's goals. It is
  not a measurement of the world-model arms' placement errors (§4.1, §11).
- K is one follower at exact poses, not the world-model arms.
- The switching rule makes lower held levels time out on the grasp subgoal more often (60 at
  +1.0 against 111 at s\*), so the arms also differ in grasp-phase duration.

### 19.3 M (128 gated root seeds 946000–946127; reported, not read)

M is reported only. K is not K-CAUSAL, so §7.1's L rows are not read. The roots job's VOID at
946073 would in any case make the L row L-VOID.

The roots were drawn with 242 rejected draws. Targets: cube 71, can 31, apple 26. s\* median
0.680. The counts include root 946073.

| Root type | Squeeze lifts | 0.9 | 0.8 | 0.7 | s\* | Control | **n_d** | Excluded (< 7 differing steps) | Ceiling wins on discordant | G wins (87100, 87101, 87102) | Positive control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rs | 128 | 127 | 110 | 101 | 102 | 0 | **26** | 2 (limit 16) | **25 / 26** (0.962; exact 0.804–0.999) | 24, 23, 15 of 26 (exact 0.749–0.991, 0.698–0.976, 0.369–0.766); bar 22 | 128 / 128 (need 112) |
| R2 | 128 | 127 | 127 | 127 | 127 | 0 | **1** | 0 | 1 / 1 (0.025–1.000) | 1, 1, 1 of 1 | 128 / 128 |

Other M checks:
- S's determinism re-run on 946000–946001 matched.
- The ceiling is identical across the three seeds.

**Reported only:**
- At Rs the squeeze also beat 0.7 on 26 of 27 discordant roots (G: 24, 22, 14) and 0.8 on 13 of
  18 (G: 16, 16, 1).
- Term medians on the Rs discordant roots (ceiling, squeeze against s\*): total 0.778 against
  0.852; the hand part of the pose term 0.138 against 0.272; the latent term 0.622 against 0.604.
  At R2, on one root: total 1.889 against 2.146.

**Each root type's own reading (reported; not read):**
- **Rs (n_d 26 ≥ 16):** the ceiling ranks the squeeze above s\* (0.962 ≥ 0.75), and G ranks it
  on two of three seeds (24 and 23 ≥ 22; G-87102 15). The search does not miss (below). The
  pattern is therefore that of **L-MIXED**.
- **R2:** unread (n_d 1), as R27.183 (2) expected.

### 19.4 S (the frozen planner, one call per root, root type and seed; reported, not read)

| Root type | Plans that squeeze (ℓ ≥ 0.9), of 128 | Median ℓ | Misses on discordant roots (need ⌈0.25 n_d⌉) | Misses on all 128 | Partial preferred, discordant | Substituted / plan cost, median |
| --- | --- | --- | --- | --- | --- | --- |
| Rs | 28, 21, 23 | 0.839, 0.859, 0.833 | **1, 0, 8 of 26** (need 7; one seed) | 6, 4, 25 | 19, 19, 14 | 1.030, 1.047, 1.022 |
| R2 | 67, 53, 85 | 0.903, 0.893, 0.915 | 0, 0, 0 of 1 | 33, 20, 21 | 0, 1, 0 | 0.994, 1.022, 0.997 |

- At Rs the first command requested a median grasp of 0.949–0.976 and executed 0.636 (the rate
  limit).
- G-i's re-evaluated plan cost equals CEM's recorded cost within 4.8 × 10⁻⁷ on every call.
- The DRAFT's comparison (reported) gave no miss anywhere: CEM's best candidate cost less than
  the squeeze branch on 128 of 128 calls per seed and root type.
- On the Rs discordant roots, G-87100 and G-87101 mostly rated the plan's own partial closure at
  least as well as the substituted squeeze ("partial preferred" 19 and 19 of 26). G-87102 missed
  on 8 of 26.

### 19.5 Budget, incidents and deviations

**Budget.**
- 17 `gpu_run` jobs, 1 029 s (**0.286 GPU h**). The job durations were 391 s for the roots, 151 s
  for the search, 29 s for the ranking, 61 s for the goals, 27–44 s per K arm and 3–4 s per K
  determinism re-run.
- TASK-094's total with Stage 0 and Stage 0′ is 1 422 s (**0.395 GPU h**), under the 0.6 GPU h
  limit and the 1.0 GPU h ceiling.
- No job reached the 1 800 s cap.

**Incidents.** The two VOID events of §19.1. Nothing was improvised: no file was edited, no job
was re-run and no cohort was touched twice.

**Ledger.** 17 new keys: `goals-gated-k`, 11 `k-gated-k-*` arm keys, 2 determinism keys, and
`roots-`, `rank-` and `search-gated-roots`.

**What was not touched.**
- No `- v2d:` request was on BOARD.md before any job.
- The GR00T server, `/mnt/data/emai/{archive,models,cache}`, `worktrees/task087-run` and other
  worktrees and evidence were not touched.
- Nothing was moved to the data disk, so `MANIFEST.tsv` was not written.

**After the run.** The only change made after the run is the comment on line 4 of
`scripts/run_task094_stage.sh` ("the Stage 0 merge" becomes "the Stage 0′ merge, 2eacacf"), as the
review of #227 asked. It changes no behaviour, and it lands after the gated stage. TASK-094 does not
hash-pin the script; its sha256 is
`ec205e1f2437bdc72af1b1f03ef4c5b1cdd85431edc15f2d0a061ae40d5a54da` at `2eacacf` and
`9f4562b32d0f1651e67887800ab9caf71056934df400148d4083306240cc2b79` after the edit.

### 19.6 Row and routing

**K-NULL** (by §4.3's text, confirmed, R27.195; the frozen summarizer printed K-VOID on a size-check
defect, §19.1). M and S are reported, not read. Their own state is L-VOID (root 946073), and Rs's reading has the L-MIXED pattern.
§7.2 routes K-NULL as follows: **the grasp line pauses and the owner is told** (R27.157). D's work
(for example a Phase 4 precursor with a scripted closure rule) still needs its own ruling, because
Phase 4 is not admitted (R25.27, R27.148). The palm-offset arms are reported beside the row
(§19.2) for that pause ruling to weigh.

R27.185's development expectation (K-UNCLEAR likely; Δ_s\* 0.0625 [−0.06, 0.19], Δ_0.6 0.125 [0,
0.28] on 32 resets) did not hold: at 128 gated resets both upper bounds are below 0.15. The
development +y palm effect (−16 / 32) recurred in direction on the gated resets (−57 / 128).

TASK-094 closes at K-NULL; no M or S re-run follows (§7.2 routes one only after K-CAUSAL + L-VOID).

**No row admits a controller, a training run or a closed loop.** "LeWM needed" is not claimed,
and no grasp success is claimed. There is no abandonment clause (§7.2). **R7 does not change**
(R27.176).
