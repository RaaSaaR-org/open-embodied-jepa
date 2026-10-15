# After TASK-093: where the grasp failure sits, and the next step (design note)

**Status: DESIGN NOTE, not a preregistration** (R27.152–R27.159, DECISIONS (bh); decided by Claude
under owner delegation, 2026-10-15). It records an offline read of existing run reports, the
options that every fired clause leaves open, and the choice of TASK-094, a measurement-only
diagnostic. TASK-094 gets its own DRAFT preregistration, with seeds, bars and rows fixed before
anything runs. Nothing here is a closed-loop result, nothing here was trained, and no GPU was used.

**This is not an Apple→Plate experiment.** R7's canonical status sentence does not change.

## 1. What is closed

Every option in §4 was checked against each clause below. The scopes are quoted from their
rulings, not widened.

| Clause | Fired at | It closes (short form) | It leaves open (relevant here) |
| --- | --- | --- | --- |
| TASK-054 | 2026-09-24 | CEM over the apple world-model cost as the *primary* control line | CEM as a tool inside a phase test |
| TASK-057 §7 | 2026-09-25 | further BC formulations on `apple-wide-v1` at 112 px | — (other corpora) |
| TASK-062 §10 | 2026-09-26 | training a LeWM-family encoder on `apple-wide-v1` | frozen DINOv2 |
| TASK-065 §10 | 2026-09-27 | the predictor on frozen pooled CLS on `apple-look-v1` | patch tokens, other corpora |
| TASK-075 §7 | 2026-10-01 | a LeWM place planner or critic under TASK-074's condition on those four views | — (not a grasp task) |
| **R26.33** (TASK-089 §9) | 2026-10-11 | CEM planning of grasp-and-lift over TASK-087's checkpoints: no further planner, cost, horizon or action-parametrisation variant over them for grasping **without new evidence of a different kind** | LeWM, DINOv2, `play-v1` as a base, CEM in general, reach planning, Phase 4 as a goal |
| **R27.71** (TASK-090 §9, fired by TASK-091's B-FAIL) | 2026-10-14 | more scripted grasp-and-lift play of `play_policy_grasp_v0`'s kind (skill mix, mode mix, hold length, retries, target weighting, amount) | model changes; other kinds of data (real G1, a different demonstrator or pick recipe, other sensors) |
| **R27.149** (TASK-093 §8) | 2026-10-15 | grasp-channel rate projection of CEM's candidates and its variants (refit space, elites or samples, first steps only, a softened rate set, adding G-req), over TASK-087's G and TASK-092's G-req | full-embodiment projection (own ruling under R26.33); every untested model change of TASK-092 §2; the corpora as bases; reach; Phase 4 as a goal |

Not fired: TASK-092 §8 (A‴-FAIL, R27.100); TASK-082 (CAL-ESCALATE at K0). The owner stopped the
moving-plate line after TASK-083 (JEPA_ZERO_SHOT_PLAN.md, interview of 2026-10-08), so it is not
an option here. Phase 4 is not admitted (R25.27, R26.34, R27.148).

## 2. The offline read (R27.152)

**What was read.** These are the Stage BS run reports of the three gated grasp-and-lift closed
loops that log per command: TASK-089 (`~/develop/emai/evidence/task089-bs/`), TASK-091
(`task091-bs/bs/`) and TASK-093 (`task093-bs/`). The read also took the goals files and
`play-v1`'s actions, joint states and sidecars, without images.

**How.** CPU only. No model, simulator or planner ran. The scripts and their verbatim outputs are
in `~/develop/emai/evidence/grasp-next-step/` (`read_grasp_logs.py`, `read_play_corpus.py`,
`grasp_logs.txt`, `play_corpus.txt`, `SHA256SUMS`). They were run against this branch's
`configs/g1_sim_action.json`.

**Definitions.**
- **Phases** are rebuilt from each episode's subgoal-switch log: 0 hover, 1 descend (open),
  2 close, 3 lift-and-hold. Each boundary may be off by one command.
- **WM** means every world-model arm that ran TASK-089's step-averaged planner over a G
  checkpoint:
  - TASK-089: G-chg × 3;
  - TASK-091: M-v1-87100, M-v1p × 3 and M-v2g × 3;
  - TASK-093: M-proj × 3 and M-v1 × 3.

  That is 1 856 episodes in all. ik is the scripted IK follower on the same resets and goals
  (320 episodes).
- **Contact** is the scorer's grasp contact: thumb plus index or middle on the target.

**Qualifiers that apply to every number below.**
- Each stage is one run, in simulation only, on its own gated cohort.
- Pooled numbers pool arms, model seeds, checkpoints and cohorts. They are for orientation only.
- Every link between a measure and success is an **association, not an intervention**.
- This read was chosen after the results were known.

### 2.1 What the WM arms do differently from ik (measured)

| Measure | WM (per run, or pooled) | ik |
| --- | --- | --- |
| Episodes with contact in the close phase | 982 / 1 856 pooled | 302 / 320 |
| Successes among them | **105 / 982** (10.7 %) | **220 / 302** (72.8 %) |
| Close subgoal "reached" (palm within 1 cm and 0.17 rad, hand within 0.15 rad RMS, for 3 steps) | 0–9 of 128 per run (0–6 of 64 in TASK-089) | 29 / 64, 58 / 128, 66 / 128 |
| Median of the minimum palm distance to the close subgoal, close phase | 0.89–1.61 cm per run | 0.21–0.24 cm |
| Median lift-phase mean **commanded** right grasp | **0.53–0.73** per run | **1.00** |
| Demonstrator's **measured** hand synergy at the close subgoal | median 0.675–0.683 per cohort (p10–p90 0.647–0.694); the demonstrator **commanded +1.0** | (same goals) |

So the WM arms touch the target in about half of their episodes, but they rarely hold it. They
command a partial closure that sits near the demonstrator's *measured* hand. That hand stopped at
about 0.68 because the object blocked it. They do not command the +1 squeeze that the
demonstrator and ik send.

### 2.2 Success against the commanded closure, given contact (measured; association)

WM episodes with contact in the close phase. Successes / episodes in each bin of the commanded
right grasp:

| Commanded grasp | < 0.7 | 0.7–0.9 | ≥ 0.9 |
| --- | --- | --- | --- |
| **Before the object rises** (the 10 commands before its first 1 cm rise, or lift start + 10) | **21 / 739** | 67 / 220 | **17 / 23** |
| Mean over the lift phase | **6 / 685** | 48 / 213 | **51 / 65** |
| Per stage, before the rise: TASK-089 | 5 / 94 | 7 / 16 | 2 / 3 |
| Per stage, before the rise: TASK-091 | 12 / 285 | 29 / 107 | 8 / 11 |
| Per stage, before the rise: TASK-093 | 4 / 360 | 31 / 97 | 7 / 9 |

- **Most common bin.** In 739 of the 982 WM contact episodes (75 %), the closure commanded
  before the rise is below 0.7.
- **Their success rate.** Those episodes almost never succeed.
- **The rare full squeeze.** The 23 episodes that command at least 0.9 succeed 17 times (exact
  95 % interval 0.52–0.90; 6 of the 23 come from one run, M-v1p-90102). That is near ik's rate
  given contact (72.8 %, 0.67–0.78), on few episodes.
- **Reverse causation.** Reverse causation is possible. A grasp that is already slipping may change
  what CEM selects, so this table does not show that the partial command causes the failure.
- **Window and selection.** The pre-rise window ends at the first 1 cm rise when there is one, and
  at lift start + 10 otherwise, so it is defined differently for episodes that rose. Conditioning
  on contact selects on a variable the closure may itself affect. In each of the 16 WM runs, the
  < 0.7 bin's success rate is at most the 0.7–0.9 bin's, so pooling across arms does not create
  the direction, but the counts per run are small.

### 2.3 Palm placement (measured; association)

Pooled successes / episodes by the minimum palm distance to the close subgoal in the close phase:

| Distance | < 0.5 cm | 0.5–1 cm | 1–1.5 cm | 1.5–2 cm | ≥ 2 cm |
| --- | --- | --- | --- | --- | --- |
| WM | 19 / 179 | 56 / 492 | 27 / 519 | 6 / 358 | 0 / 300 |
| ik | 160 / 241 | 61 / 72 | 0 / 4 | 0 / 0 | 0 / 3 |

The rows cover 1 848 of the 1 856 WM episodes; 8 never enter the close phase in the
reconstruction. 3 of the 108 WM successes have no contact in the reconstructed close phase, which
is why §2.1 counts 105.

- **Placement matters.** No WM episode succeeds beyond 2 cm.
- **But it is not the main gap.** WM episodes placed as well as ik (under 0.5 cm) succeed 19 / 179,
  against ik's 160 / 241.

### 2.4 G's own cost in the lift phase (measured)

- **The measure.** The median cost of the chosen plan, as G predicts it, over the lift phase.
- **The values.** It is lower in succeeding episodes than in failing ones in every run that has a
  success: 0.58–0.97 against 1.17–1.62.
- **What it shows.** G's cost does tell a held, rising object from a failed hold *as a state*.
- **What it does not show.** It does not show that G ranks the *actions* that lead to each outcome
  from the same contact state. That is not measured.

### 2.5 The training corpus (measured)

`play-v1`: 3 194 episodes and 1 156 589 transitions, every split read.

- **At grasp contact (48 050 steps).**
  - The commanded right grasp is ≥ 0.8 on **46 675 (97.1 %)**, with a median of 0.924.
  - The measured synergy is median 0.633 (p10–p90 0.584–0.807). So the corpus does show the
    blocking: a near-full command gives a hand at about 0.63 on an object.
  - Commands of 0.5–0.8 at contact occur on 730 steps, and commands below 0.5 on 645.
- **Lifted steps (27 980; ≥ 5 cm and in contact).** The command is below 0.8 on **402 (1.4 %)**.

So the partial closures at contact that the WM arms command are rare in G's training data
(1 375 of 48 050 contact steps below 0.8, 2.9 %; 402 of 27 980 lifted steps, 1.4 %). *Inferred:*
G has had little data from which to learn what a partial close does to the object.

## 3. Where the failure most plausibly sits (R27.153)

| Locus | Measured | Inferred |
| --- | --- | --- |
| **(a) G cannot predict grasp outcomes** | **Coarse contrasts.** G separates coarse contrasts at the demonstrator's states: closing or lifting against open or non-lifting branches, 64 / 64 roots (TASK-089 A1; offline). **State level.** G's cost separates a held state from a failed one in the closed loop (§2.4). **Data.** The closure levels the planner picks at contact are rare at contact in `play-v1` (§2.5). | **Partial closes.** G is unlikely to know that a partial close at contact does not hold. The action-level ranking from one contact state (squeeze against partial close) is **not measured**. |
| **(b) the cost or goal cannot express a grasp** | **The goal.** Its hand term is the demonstrator's *measured* hand, blocked by the object at about 0.68 while it commanded +1 (§2.1). **The commands.** The planner's commands sit near that value (§2.1–§2.2). **The corpus.** A squeeze on an object reaches a hand state close to the one a partial close reaches (§2.5). **A related contrast.** On real frames the latent term alone did not separate lift-while-opening from lift-while-closed at h = 8 (TASK-089 A-K0: 15 / 32). With the pose term, which G's cost includes, it did on 32 / 32, because an opening hand's posture differs. A partial close and a squeeze on an object-blocked hand reach nearly the same posture, so that contrast is not measured. | **Flat cost.** The pose term is about equally satisfied by a partial close and by a squeeze. Before the object rises, nothing in the goal asks for squeeze. **The most plausible primary locus, together with (a).** |
| **(c) CEM cannot find plans that G rates well** | **Push-T.** Our CEM is not the bottleneck there (TASK-084 P0-PASS). **The roots.** CEM chooses to close and lift on 62 and 61 of 64 roots (TASK-089 A2; offline). | **Unlikely to be primary.** The closures CEM chooses sit where the goal's hand term is lowest, so CEM appears to find what the cost asks for. **Not measured** at contact states. |
| **(d) a mismatch between plan and execution** | **Grasp channel.** TASK-093 removed its mismatch: the plans were wholly on the training support, and 0 of 139 825 commands differed. There was no gain (p = 0.810). That mismatch is therefore **not the bottleneck** (one run, simulation only, 128 resets). **Palm precision.** It is worse than ik's (§2.1, §2.3). | **Arm channels.** The arm's plan against its execution is **not measured**: `plan_log` keeps the grasp only. Palm imprecision contributes, but it is secondary (§2.3). |

**Reading.** The failure most plausibly sits in the objective at the close-and-lift transition,
in (b) together with (a).
- **The goal.** It specifies the hand's measured posture, and a partial close also satisfies that
  posture.
- **The model.** G has little data on what a partial close at contact does.
- **The result.** The planner commands closures of about 0.55–0.65, and those do not hold.

The chain from these commands to the failure is **inferred**. The individual measures are listed
above, and the link to success is an association. TASK-094 is designed to test each link
separately.

## 4. Candidate next tasks (R27.155)

Every candidate was checked against §1.

| | Task | What it decides or tries | Scope check | Cost |
| --- | --- | --- | --- | --- |
| **A (chosen)** | **TASK-094: grasp-closure locus diagnostic** (§5). Measurement only, with no controller, no training and no world-model closed loop. | It tests whether partial closure causes the failure in physics, and whether G or the cost ranks a squeeze above a partial close at contact. It also tests whether the frozen planner finds the squeeze when G prefers it. This decides between (a), (b) and (c). | It preregisters no planner, cost, horizon or action-parametrisation variant for grasping, so R26.33 is not engaged. The frozen planner runs only offline at fixed roots, as in TASK-093 Stage P. It uses no new play data (R27.71) and no rate projection variant (R27.149). The scripted probes are not learned results. | **≈ 0.5 GPU h** (ceiling 1.0) plus CPU |
| B | **Grasp-outcome head and cost** (TASK-092 §2 rank 2): a model change that predicts contact and rise over the horizon, with a cost on it. | It tries a fix on the model side. | Open (TASK-092 §2, R27.149's "not closed"). It needs its own ruling on simulator-truth labels in a learned cost (TASK-090 §15 reason 3). | ≈ 2–3 GPU h for training and the offline gate, plus ≈ 14–15 GPU h for BD and BS at n = 128; **≈ 17–18 GPU h** |
| C | **Goal specification change** over TASK-087's G: the goal's hand term taken from the demonstrator's *command*, or the hand joints taken out of the pose term and given a closure floor. | It tries a fix on the cost side. | A cost variant over TASK-087's checkpoints, so **R26.33 applies**. This ruling does not find R26.33's condition met: §2 is an association read after the results. TASK-094's physics and ceiling parts could supply evidence of a different kind, under their own ruling. | ≈ 0.5 GPU h offline, plus ≈ 14 GPU h closed loop; **≈ 15 GPU h** |
| D | **Pause the grasp line** and return to what is demonstrated (Z3-REACH). For example, a Phase 4 precursor in which the world model plans reach and transport and a scripted closure rule does the grasp. | It gets progress elsewhere. | Phase 4 is not admitted (R25.27), so this needs its own ruling. It would not show "LeWM needed", and the owner wants one general model. The moving-plate line is stopped by the owner. | Preregistration, plus ≈ 10–15 GPU h |

**Considered and not proposed.**
- **Full-embodiment candidate projection** (the arm channels): it is open under R26.33, with its
  own ruling. §2.3 measures palm precision as secondary, and the IK surrogate is judged likely to
  exceed the job cap (TASK-092 §2; not measured).
- **More grasp play of the `play_policy_grasp_v0` kind**: closed (R27.71).
- **Any rate-projection variant, or G-req**: closed (R27.149).

**Why A.**
- **Why it fits.** §2 names a specific mechanism that is cheap to test, and every fix of it (B, C,
  or a data change) costs about 15–18 GPU h of closed loop. Four closed loops have failed after
  offline gates that passed. Spending 0.5 GPU h first to learn which locus is real is the
  measurement-first step.
- **Not D yet.** The evidence does not yet point to pausing, because §2's lead has not been
  tested. If TASK-094 reads K-NULL, the grasp line pauses and the owner is told. D's work (for
  example a Phase 4 precursor with a scripted closure rule) still needs its own ruling, because
  Phase 4 is not admitted (R25.27, R27.148) (R27.157).

## 5. TASK-094 in outline (its preregistration fixes the details)

**Question.** The WM arms command a partial closure at contact. Is that closure the cause of their
grasp failures? If it is, does the failure come from the goal and cost (b), from G (a) or from the
search (c)?

**Parts.** Each runs once, on fresh development or gated roots and resets from a seed block that
the preregistration checks.

1. **K, the physics dose (scripted, no model).**
   - **Arms.** TASK-088's IK follower on fresh resets and the TASK-088 goals, with the closing
     command held at c ∈ {+1.0 (ik as frozen), 0.9, 0.8, 0.7, s\*}. Here s\* is the close
     subgoal's measured synergy.
   - **Palm offset.** c = +1.0 and c = s\* are also run with a fixed palm-target offset of about
     the WM arms' median (≈ 1.2 cm; the preregistration fixes the vector).
   - **What it reads.** Whether a partial closure at an exact pose loses the lift, and what a
     1.2 cm placement error costs on its own.
   - **Status.** These are scripted probes, not learned results.
2. **M, the ranking (offline, frozen G, no training).**
   - **Roots.** Fresh roots at the demonstrator's state at the end of the close, before the lift,
     like TASK-089's R2.
   - **Branches.** Scripted branches that follow the demonstrator's lift with the hold level at
     each c of K, simulated to their true outcome (lifted and held, or not).
   - **What it reads.** Each branch is scored with the frozen step-averaged cost in two ways:
     - (i) from G's predictions (the model);
     - (ii) from the branch's real encoded frames and measured states (the ceiling, as in
       TASK-089's A-K0).
   - The read is how often the squeeze outranks s\*, per root and per seed.
3. **S, the search (offline, frozen planner).**
   - **Run.** At the same roots, TASK-089's frozen step-averaged CEM over G-87100–87102.
   - **What it reads.** The chosen hold level, and G's cost of the chosen plan against the c = +1.0
     branch.

**Reading rule (sketch; the preregistration declares it with bars before any run).**
- **K-VOID.** The run is void, for example when ik at c = +1.0 does not reproduce its frozen BS
  rate within a declared margin, or when a degeneracy check fails (the preregistration declares
  the conditions).
- **K-NULL.** At an exact pose, c = s\* lifts within a declared margin of c = +1.0. Then §2.2 is
  not a physics cause. If TASK-094 reads K-NULL, the grasp line pauses and the owner is told. D's
  work (for example a Phase 4 precursor with a scripted closure rule) still needs its own ruling,
  because Phase 4 is not admitted (R25.27, R27.148). The palm-offset arm (c = +1.0 and c = s\*
  at about 1.2 cm) is reported beside K-NULL, so that a placement-dominant result is visible to
  the pause ruling.
- **K-CAUSAL**, and then the first matching row of M and S applies:
  - **L-COST (b):** the ceiling does not rank the squeeze above s\* on the declared bar of roots.
  - **L-MODEL (a):** the ceiling does, and G does not.
  - **L-SEARCH (c):** G does, but CEM's chosen plans hold partial and G rates them worse than the
    squeeze branch.
  - **L-MIXED:** any other pattern.
- **Routing only.** Each row routes to a ruling. None of them admits a controller by itself:
  - L-COST routes to C or to B with a grasp term. C is a cost variant over TASK-087's
    checkpoints, so it goes only through a ruling that finds R26.33's condition (new evidence of a
    different kind) met; TASK-094's preregistration states whether a K-CAUSAL with L-COST reading
    is put forward as such evidence. B is a model change, open under TASK-092 §2 and R27.71, with
    its own ruling on simulator-truth labels;
  - L-MODEL routes to B, or to a data kind that R27.71 leaves open;
  - L-SEARCH routes to a planner ruling under R26.33;
  - L-MIXED routes to a ruling.

**What it cannot show.** It runs no closed-loop world-model controller, so it cannot show a grasp
success. Its roots are the demonstrator's states, not the WM arms' states. Its contrasts are
scripted. It covers one scene, camera, encoder and corpus, in simulation only. The WM arms
choose their closure during the close, from an open or closing hand. M's and S's roots at the end
of the demonstrator's full close start from a hand that has already squeezed. The preregistration
states whether M and S also use roots at the start of the close (as TASK-089's R1), and if not,
why.

**Budget.** K costs well under 0.1 GPU h: ik ran 128 resets in 29–43 s per job. M and S together
cost about 0.2–0.3 GPU h; TASK-089's Stage A took 6 minutes. Stage 0 checks add about 0.2 GPU h.
That is **≈ 0.5 GPU h**, with a ceiling of 1.0 GPU h. Every GPU job runs under
`gpu_run.sh --wait --board`.

## 6. Rulings (R27.152–R27.159)

See DECISIONS (bh).
