# Apple→Plate policy diagnostics v1: locate the open-loop/closed-loop gap (TASK-057)

**Status: preregistration DRAFT. Nothing in this document has been run.** No probe has executed,
no cohort has been opened, and **cohort C (seeds 45300–45339) is not touched by this task at any
stage** — this protocol preregisters no use of it.

Predecessor: [`apple_policy_v1.md`](apple_policy_v1.md) / [`_results.md`](apple_policy_v1_results.md)
(TASK-056), which failed on its development stop rule: four arms, 64 development attempts, **0 full
successes**. Handover: [`task056_handover.md`](task056_handover.md).

**Two attempts were not total failures, and this protocol leads with them (§1.3): A2 on seed 45100
and A3 on seed 45006 each reached the scorer's `grasp` stage, lifted the apple ~17 cm, and were
still holding it undropped when the step cap ended the attempt.** Neither reached `transport`. The
two control arms A0 and A1 never came within 6.5 cm of the apple on any of their 32 attempts.
A3 additionally had two `guard_refusal` terminations, which are physical stops rather than cap
hits, so **"every attempt ran to the cap" is false and appears nowhere in this document.**

This is **not** a new control formulation. It trains nothing and proposes no model change. It is a
diagnostic task whose only purpose is to locate the gap between TASK-056's near-perfect open-loop
imitation (median action error 0.0067–0.0303 per dimension, §9) and a closed loop that reaches
`grasp` on 2 of 64 attempts and `transport` on none — and whose gate can fire the abandonment
clause that `apple_policy_v1.md` §7 already preregistered.

---

## 1. Why this task exists, and why it is not the task that was first proposed

### 1.1 The mechanism TASK-056 recorded is contradicted by its own artifacts

`apple_policy_v1_results.md` §12 and §16, and `task056_handover.md` §5, record a candidate
mechanism: the expert's `right_dz` sits at exactly 0.400 on 44.89 % of commands, a smooth-L1
regression head hedges toward the interior of that bimodal target, and the resulting under-shoot
means the policy never reaches the object inside 1000 steps. It is recorded there — correctly —
as **a consistent story, not a demonstrated mechanism**.

It is now contradicted. Three findings, each traced to the code that computes it:

**(a) The expert's `right_dz` is a signed, near-symmetric ±0.400 bang-bang whose mean is zero.**
`src/embodied_jepa/scripted.py` computes
`action[6:9] = np.clip((phase.target_base - position) / translation_per_step_m, -0.4, 0.4)`
— a saturating P-servo whose sign flips by phase. Reproducing the BC cohort from the committed
rules in `src/embodied_jepa/cloning.py` returns 137 surviving train roots and **68,791 rows**,
matching `benchmarks/manifests/apple-policy-v1.json` `/frozen/.../train_rows_sampled` exactly, and
**|dz| ≥ 0.400 on 44.886 %**, matching the committed 44.89 %. Decomposed:

| statistic | value |
|---|---|
| `right_dz` = +0.400 | 18.45 % |
| `right_dz` = −0.400 | 26.44 % |
| \|`right_dz`\| = 0.400 | 44.89 % |
| **mean `right_dz`** | **−0.0012** |
| median \|`right_dz`\| | 0.1968 |

The committed 44.89 % is an **absolute-value** rate. A conditional-mean regressor that learned
nothing at all would emit ≈ 0 on this target, so "the expert commands 0.400, the policy commands
0.03" was never a comparison of the same quantity.

**(b) The trained heads do not hedge. Offline they reproduce the saturated expert command.**
Re-derived by executing `cloning.evaluate_policy` against the four TASK-056 checkpoints on the val
split; §9 carries the full table and records agreement with the training-time record to float32
round-trip. Median absolute `right_dz` error against the expert is **0.01333 / 0.01569 / 0.01992 /
0.02832** (A1 / A0 / A2 / A3) — against an expert whose median |dz| is **0.1968** and which sits at
±0.400 on 44.89 % of rows — while predicted `right_dz` standard deviation is **0.2656–0.2735**. On
the near-binary grasp dimension the median error is **0.0073–0.0111** and the predicted standard
deviation is **0.971–0.979**.

A head that hedged toward the interior of a bimodal target would show a *compressed* output
standard deviation and a *large* error on the saturated rows. It shows neither.
**Declared risk R5 of `apple_policy_v1.md` §8 — "a regression head on a near-binary target emits
interior values" — did not occur.**

Incidentally: `cloning.py` calls `torch.nn.functional.smooth_l1_loss` with the default
`beta = 1.0`, and every residual here is well inside |x| < 1, so the objective was **pure L2
throughout** and its minimizer is the conditional *mean*. The "smooth-L1 hedging" framing describes
a regime the run never entered.

**(c) The closed-loop *translation* commands are near-constant. The rotation commands are not.**
Across all 64 attempts in `outputs/task056-cohort-d/*.json`, median across attempts of each
attempt's own |·| quantiles:

| | q50 | q90 | q99 | q90/q50 |
|---|---|---|---|---|
| `dx` | 0.0138–0.0342 | — | — | **1.00–1.22** |
| `dz` | 0.0125–0.0857 | — | — | **1.01–1.32** |
| `droll` | **0.179–0.243** | **0.478–0.498** | **0.503–0.513** | **2.05–2.67** |

A head whose validation output standard deviation is 0.27 produced a *translation* command whose
90th percentile equals its median over a thousand consecutive steps. **Roll varies across its full
range in the same attempts.**

**Stated at exactly that strength and no further.** This is **not** roll running away: `droll`'s
out-of-distribution rate is 5–7 % and its q90 sits at the 0.5 configured bound, which is **less**
saturation than the expert's own measured ~27 % at |max|
(`benchmarks/manifests/apple-policy-v1.json`, `no_pooled_figure_is_produced`). The observation is
narrower and stranger: **the one dimension with real online variance is rotational, and it is the
dimension with the *smallest* offline error in Table A** (0.00726–0.01873, the smallest of all
seven for every arm). That is the largest online/offline behavioural gap in the run, and it is not
in the dimension the dz hypothesis pointed at. **D2 reports all seven dimensions and privileges
none of them.**

### 1.2 What is NOT known, and an instrumentation finding that produced this protocol's best probe

`scripts/evaluate_policy.py` computes every stored command statistic from
`free = np.abs(commands[:, FREE_INDICES])`. **Not one negative number exists anywhere in
`command_statistics` across all 64 attempts.** Therefore:

> **The sign of every commanded dimension, in every attempt of TASK-056, is unrecoverable from the
> run artifacts.** This is not specific to `grasp`. `|dz| q50 = 0.0125` is equally consistent with
> descending slowly, ascending slowly, and oscillating about zero; `|grasp| q50 = 1.0` is equally
> consistent with a hand held fully open (which is what the expert does through `orient` and
> `descend`) and one held fully closed.

Consequences, stated because they constrain what this protocol may claim:

- **No directional claim about TASK-056's commands is available from its record**, including the
  under-shoot reading in `apple_policy_v1_results.md` §16 and `task056_handover.md` §5. That
  reading is not refuted by this observation; it is shown to be **unmeasured**.
- **This is why probe D1-grasp is the most informative measurement in this protocol.** The
  expert's step-zero `right_grasp` is exactly −1.0 on every reset (`scripted.py`, `orient` phase),
  so a signed step-zero comparison has a known-correct reference and needs no threshold — and it
  would be the first sign measurement of a learned command anyone in this project has taken.
- Restoring signed per-dimension logging is a **blocking requirement** on the runner change (§9).

### 1.3 Both learned arms grasped and lifted the apple. This comes first.

**A2 and A3 each reached the scorer's `grasp` stage on 1/16, and the two attempts are the highest-
information resets in the cohort.** `AppleToPlateTask` requires `reach ∧ lifted ∧ hand_contact`
with `lift_height_m = 0.05`:

| arm | seed | `grasp` | final `object_height_m` | lift vs rest (0.766633 m) | `hand_contact` at cap | `dropped` | `transport` |
|---|---|---|---|---|---|---|---|
| A2 | **45100** | **True** | 0.934384 | **+0.1678 m** | **True** | False | False |
| A3 | **45006** | **True** | 0.935996 | **+0.1694 m** | **True** | False | False |

Both arms contacted the apple, closed on it, lifted it about **17 cm — more than three times the
5 cm gate** — and **were still holding it, undropped, when the 1000-step cap ended the attempt.**
Neither reached `transport`, which additionally requires the apple to be within the plate radius
in xy. **So the A2/A3 failure is not "never approaches the object". It is "grasps, holds, never
transports."**

**Provenance of this correction, stated precisely.** The 1/16 counts **are** in the committed
record — `apple_policy_v1_results.md` §14's table and §16 ("A2 ... and A3 each reached grasp once
in sixteen"). What is *not* anywhere in the record, in either document, is the lift height, the
sustained contact, or the undropped state at the cap. And `task056_handover.md` — the document
written to carry TASK-056 forward — states only the controls' 0/16 and never gives A2's or A3's
grasp count at all. **The compression into the handover removed the run's single most encouraging
measurement.** That is a finding about handover practice and it is recorded as one.

**Consequences carried into this protocol:**
- Seeds **45100 (A2)** and **45006 (A3)** are named targets for D2 (§3), because they are the only
  two attempts where the loop got far enough for a divergence trace to show anything other than
  immediate collapse.
- **No statement of the form "every attempt ran to the step cap" appears in this protocol**, and
  none may. A3 had **two `guard_refusal` terminations, on seeds 45001 (157 executed steps) and
  45003 (143)**; they are physical stops, not cap hits (`apple_policy_v1_results.md` §17).

### 1.4 What is established about the control arms, and its exact scope

Two claims that do not route through that absolute value, both confirmed directly.

**A0 and A1 never approached the apple, and never moved it.** `AppleToPlateTask.evaluate`
(`src/embodied_jepa/task.py`) latches `reach` when `‖palm − apple‖ ≤ 6.5 cm or hand_contact`, and
is evaluated after every executed command. Across **32 attempts** (A0 and A1 × 16 development
seeds) `reach` is false 16/16 for each arm, `hand_contact` is false at every attempt's end, the
final `object_plate_distance_m` matches the report's own stored reset geometry to within 0.1 mm on
16/16 for both, and `object_height_m` takes exactly one distinct value (0.766633 m) across all 32.
Contact is the only route by which the robot can move the apple, so `reach = 0/32` establishes this
independently of the distance statistic. The expert covers the whole approach in ~210 commands;
these arms had 1000.

**A2 and A3 are reported separately and are NOT covered by the statement above.** A2: `reach`
1/16, final geometry unchanged on 15/16. **A3: `reach` 4/16** — it contacted the apple on four
resets — final geometry unchanged on 12/16, and `object_height_m` takes five distinct values.
§1.3 is the substantive result for these two arms. (An earlier draft of this analysis extended the
"never touched the apple" statement to A3 from a mis-scoped count; it does not hold for A3.)

### 1.5 The two live hypotheses

Near-perfect open-loop imitation (Table A, §9) with a closed loop that emits a near-constant
*translation* command; for the two control arms it never approaches the object at all, and for the
two encoder arms it reaches, grasps and holds on one reset each but never transports.

- **H-pipeline.** The observation `scripts/evaluate_policy.py` hands `ClonedPolicy.act` is not the
  object `cloning.py`'s `arrays.states` / `arrays.frames` were. The policy sees something outside
  its training distribution, emits a near-constant command, the robot barely moves, the state stays
  constant, and the loop is a fixed point. **Nothing in TASK-056 verified that the loop's
  observation equals the training observation.**
- **H-shift.** The observation is correct, but the reset state or the first commands put the policy
  off the demonstration manifold and it collapses toward the marginal. Compounding covariate
  shift — what `apple_policy_v1.md` §7 Outcome D already names.

Both predict a near-constant translation command. They differ in **when** the command first departs from what
the expert would have issued, and that is measurable at step zero, before anything can compound.

---

## 2. Definitions — the domain of every scope term, at introduction

`apple_policy_v1_results.md` §15 records that a single undefined term ("learned arm", used thirteen
times across the frozen artifacts, defined in none) decided whether an irreplaceable cohort would be
consumed. Every scope term used in a gate or stop rule below is enumerated here by name.

- **Arm** — ranges over exactly **{A0, A1, A2, A3}**, the four checkpoints
  `checkpoints/task056-policy-v1/{a0,a1,a2,a3}.pt`. Nothing else is an arm in this protocol.
  **A4 is not an arm here**; it is not built and this protocol does not propose building it.
- **Control arm** — exactly {A0, A1}. **Primary arm** — exactly {A2}. Used with no other extension.
- **Development cohort** — exactly the 16 seeds 45000–45007 and 45100–45107.
- **Attempt** — one closed-loop episode of one arm on one development seed under one named
  configuration, capped at 1000 executed commands.
- **Arm dimension** — one of the six {dx, dy, dz, droll, dpitch, dyaw}. **`grasp` is never included
  in an arm-dimension aggregate** and is always reported separately.
- **Offline error of arm X in dimension d** — the per-dimension median absolute action error of X's
  selected checkpoint on the val split, from the re-derivation of §9. These are **fixed numbers
  transcribed into this document and the manifest before any probe runs.**
- **Step-zero command** — the policy's 7-vector at the first `act` call after `robot.reset`, before
  any `execute`.
- **Shadow expert command** — `OracleManipulationPolicy(sim.task_truth()).action(robot)` at indices
  (6,7,8,9,10,11,13), the instance advanced by `advance(result)` on every executed step exactly as
  the collector advanced it, and computed strictly **after** `policy.act` returns.
- **Departure at step t** — median over the six arm dimensions of |policy − shadow expert|.
- **Configuration** — the named subset of the seven free dimensions taken from the shadow expert.
  `none` is the empty set; `full` is all seven.
- **Grasp reset** — a development seed on which the latched `score["grasp"]` is true at any point.

---

## 3. Probes

No probe trains anything. All three run on the development cohort only.

### D1 — the step-zero contrast

For each arm and each of the 16 seeds, capture the step-zero command **with sign** and the shadow
expert command at the same instant. At step zero no error has compounded and the state is the reset
state, drawn from the distribution the corpus was collected from. If the input pipeline is sound,
the step-zero command must be right.

### D1-grasp — the one measurement with a ground truth

The expert's step-zero `right_grasp` is **−1.0 exactly, on every reset**, because `orient` carries
`grasp = -1.0` in `scripted.py`'s phase table. For each arm, report the count out of 16 on which the
step-zero policy `right_grasp` is **> 0**. No threshold: the reference is known-correct and the
result is a count from 0 to 16. Reported separately from D1 and never folded into it.

### D2 — the divergence trace

All four arms × 16 seeds × ≤ 1000 steps, with additive per-step logging: signed policy command (7),
signed shadow expert command (7), `‖palm − apple‖`, the scorer's stage, executed step index.

**Named traces, declared in advance.** The results document reports the full per-step trace for
**A2/45100** and **A3/45006** — the two attempts that reached `grasp` (§1.3) — each alongside a
**failing reset of the same arm** for contrast, fixed here before the run as **A2/45000** and
**A3/45000** (the lowest-numbered seed, chosen by a rule rather than by inspection so it cannot be
picked after the traces are seen). These four traces are reported whatever they show.

**All seven dimensions are reported. No dimension is privileged**, and in particular the protocol
does not report dz preferentially: §1.1(c) records that the largest online/offline behavioural gap
is in `droll`, not in translation.

### D3 — per-dimension expert substitution, primary arm only

At each step, dimensions in the configuration come from the shadow expert; the rest come from the
policy. Nine configurations on the 16 seeds: `none`, `all_translation`, and one per dimension.

Pre-committing to the **sweep** rather than to a single dimension is deliberate: the mechanism
TASK-056 recorded turned out to be wrong (§1.1), so a probe aimed at it would have confirmed an
error rather than tested it. The sweep costs the same and tests seven hypotheses.

---

## 4. Cohort discipline

- **Every probe: the development cohort only.** `scripts/evaluate_policy.py`'s existing cohort-D
  whitelist enforces it and is not widened. The whitelist form is load-bearing —
  `apple_policy_v1_results.md` §8 records a type coercion that would otherwise have consumed the
  frozen cohort — and a guard that enumerates what is permitted fails closed.
- **Cohort C (45300–45339) is not touched by this task at any stage.** No gate references it. The
  handover's blocking stored-values debt attaches to a runner that opens C; **this protocol builds
  no such runner, so that debt does not land here and must not be discharged speculatively.**
- **No selection on D.** D is a mechanism instrument, not a scoreboard. No checkpoint is chosen,
  retrained or tuned against a D outcome.
- **Every substituted configuration puts a privileged controller partly in command.** Every number
  D3 produces is a diagnostic, never a learned-policy result and never a manipulation result, and
  is labelled so in the table itself and not only in a footnote.

---

## 5. Gates

**D1 — step-zero contrast.** D1 passes for arm X iff, for **every** one of the six arm dimensions,
the median across the 16 seeds of |step-zero policy − shadow expert| is ≤ **3 ×** X's own offline
error in that dimension.

*Derivation of 3×, fixed before the numbers exist.* The offline error is measured on recorded val
frames at arbitrary phase; a reset frame is one specific, in-distribution phase. The claim tested is
"same order", not "equal". The per-dimension offline errors themselves span roughly a 4.5× range
across arms and dimensions, so a factor below 3 cannot be distinguished from ordinary per-dimension
variation, while 10× or more is unambiguous. It is a judgement call, not a derivation, and the
realized ratios are reported per dimension whatever they are. **No threshold moves in response to
anything measured.**

**D1-grasp.** A reported count, 0–16 per arm. No threshold (§3).

**D2 — a reading rule, explicitly not a gate, and it determines no gate outcome.** Stated this way
because `apple_policy_v1.md` §5.3 announced a rule was "not a gate" and then decided gate outcomes
through the missing-values rule (`apple_policy_v1_results.md` §15). D2 feeds nothing.
Reading fixed in advance: median `first_departure_step` ≥ 50 → **gradual**; ≤ 5 → **immediate**;
otherwise → **inconclusive, and reported as inconclusive**.

**G-SUB — the only gate. Per-dimension expert substitution, A2 only.**
Metric: grasp resets out of 16, per configuration.
Fixed reference points, all before the run: pure-policy A2 = **1/16**
(`outputs/task056-cohort-d/a2.json`); `scripted_oracle` = **24/24 pooled** on three prior wide
cohorts (`apple_wide_grasp_closure_results_v3.md`, `apple_wide_object_ceiling_results_v2.md`, as
cited in `apple_policy_v1.md` §5.2 G5); `demo_replay` pooled grasp = **8/24 = 33.3 %**
(`apple_policy_v1.md` §5.2).

> **G-SUB passes iff at least one configuration reaches ≥ 8/16 grasp resets.**

*Derivation of 8/16.* It is the midpoint between the pure-policy 1/16 = 6.25 % and the 100 %
ceiling, and it sits above the `demo_replay` null of 33.3 %, so a configuration that merely matches
open-loop replay does not pass. **This is a magnitude threshold, not a significance threshold, and
is not presented as one:** against p₀ = 1/3 the one-sided binomial probability of ≥ 8/16 is 0.189.
D is a non-gating development cohort and no p-value from it is a result. The requirement is that
the effect be at least half the distance to the ceiling; a smaller effect would not justify
building anything.

**Missing values.** A gate that cannot be evaluated counts as failed. D1 and G-SUB are the only
gates.

### 5.1 Abandonment clause

> **If G-SUB fails — no configuration, including `all_translation`, reaches 8/16 grasp resets —
> then no single command channel carries the failure, and the preregistered Outcome D abandonment
> clause of `apple_policy_v1.md` §7 fires as written.** This control line stops on this corpus and
> this camera. No further loss, head or output-parameterisation variant is preregistered on it. The
> next task is a perception/data task. Cohort C remains unconsumed. The executing agent stops and
> reports and does not select the follow-up itself.

### 5.2 Success clause

The line continues iff **either**:

- **(a)** D1 fails for ≥ 3 of the 4 arms, **or** D1-grasp returns a step-zero grasp sign error on
  ≥ 8/16 seeds for ≥ 3 of the 4 arms. Either is a concrete, locatable discrepancy between the
  loop's observation pipeline and the trainer's. The follow-up is a pipeline audit and a
  re-measurement — **not** a model change and **not** a retrain.
- **(b)** G-SUB passes, naming the failing channel.

In both cases the follow-up is a **new preregistration with its own gates and controls**, never a
build started inside this task. That follow-up is the task that may ask for cohort C.

### 5.3 Committed in advance so it cannot be argued later

A rescue under expert substitution injects privileged information at 50 Hz, which the policy does
not get. **It shows which channel the failure flows through; it does not show that a learned
version of that channel is attainable.** No D3 number may be read as evidence that a fix will work,
only as evidence about where to aim one.

---

## 6. Baselines and controls

- **B1 — harness soundness on D.** `scripted_oracle` must reach grasp on ≥ 14/16 and succeed on
  ≥ 12/16; `hold` and `random` must be 0/16. The oracle is 24/24 on three prior wide cohorts of 8;
  14/16 permits two harder resets without admitting a broken harness. `hold`/`random` at 0 is the
  measured 0/50 on apple→plate (`docs/experiments/mvp_results.md`, via `apple_policy_v1.md` §5.2).
  **If B1 fails, every number in this task is void**, exactly as G5 works in the parent protocol.
- **B2 — the zero-substitution control.** The `none` configuration must reproduce
  `outputs/task056-cohort-d/a2.json`: the same single grasp seed (45100), the same 16 termination
  reasons, the same executed-step counts. If the harness changes the pure-policy result, the harness
  is a confound and the sweep is void. This is the control that makes G-SUB interpretable and the
  one most likely to be skipped.
- **B3 — the inert-probe tripwire, run FIRST.** `apple_policy_v1_results.md` §13 instance 9: a
  control whose output is indistinguishable from a real pass **conceals** the class it was built to
  catch. The `full` configuration — all seven dimensions from the shadow expert — must reach
  ≥ 14/16 grasp resets. If it does not, the substitution channel is not reaching the robot and the
  sweep is void before it starts. This probe exists only to prove the mechanism is connected and it
  **fails loudly when inert**.
- **B4 — the shadow expert is not an input.** A behavioural test that the shadow command is computed
  strictly after `policy.act` returns and is never an argument to it; plus the `ast.walk` form
  (its text-grep predecessor produced a false positive on a correct runner — §13 instance 8)
  asserting `policy.py` imports no label or truth module. **G6's zero-privileged-reads rule applies
  to the controller; in a substituted configuration the expert *is* part of the commanded action**,
  hence §4's labelling rule.
- **B5 — the standing offline anchor, carried not re-measured.** A1 0.01106 < A0 0.01313 <
  A2 0.01793 < A3 0.02340: the primary is third of four, beaten by the arm that never sees an
  image. Any claim that vision does work must survive it.
- **B6 — does the learned part do work?** From B5 and §1.3 the honest answer is **not
  demonstrated, on any metric.** This task does not attempt to establish it. Every probe is
  diagnostic, and no result here may be reported as progress toward a learned result.

**Every debt-discharging test is written by injecting its own violation and watching it fail
first** — the standard `apple_policy_v1_results.md` §9 adopted after two tests claimed debts they
did not discharge.

---

## 7. Pre-declared outcomes

- **Outcome P — D1 or D1-grasp fails (success clause (a)).** A locatable pipeline discrepancy.
  Next: an audit of the observation path from `G1Embodiment.observe` to `ClonedPolicy.act` against
  `cloning.py`'s training path, then a re-measurement. No model change, no retrain.
- **Outcome S — D1 passes and G-SUB passes (success clause (b)).** The failing channel is named.
  Next: a new preregistration for a targeted fix, with its own controls. That task may ask for C.
- **Outcome X — G-SUB fails.** §5.1 fires. The line stops.
- **Outcome V — B1, B2 or B3 fails.** The run is void. No arm numbers are reported as results;
  the defect and the void are recorded.

In every outcome: all numbers are published whatever they say; failures and negative results stay
in the versioned record; the `apple-wide-v1` test split is never decoded; nothing is retuned and
re-reported after the gates are read.

---

## 8. Budget

MPS/CPU only, Apple M5 Pro 48 GB. No training. No corpus decoding beyond §9's re-derivation.

| Stage | Basis | Estimate |
|---|---|---|
| B1 baselines + B3 inert probe | oracle 7.7–8.0 s per attempt (`apple_policy_v1.md` §6) | ≤ 20 min |
| D1 + D1-grasp | 4 arms × 16 seeds × 1 step; dominated by simulator construction | ≤ 10 min |
| D2 traces | 4 arms × 16 seeds × ≤ 1000 steps | ≤ 45 min |
| D3 sweep (A2) | 9 configurations × 16 seeds; passing configurations terminate early | ≤ 60 min |
| B2 control + slack for one repeat | | ≤ 60 min |
| **Total** | | **≈ 3.5 h** |

**Caps: 6 h global, 300 s per attempt, 1000 steps per attempt.** At the cap the task stops and
reports what it has. The parent protocol's 28 800 s cap is not carried; each protocol freezes its
own (`docs/RESOURCES.md`).

---

## 9. The offline error table, and the standard it was held to

`apple_policy_v1.md` §9's practice is followed: a quantity that exists only in a git-ignored run
artifact is not citable. The per-dimension offline errors that D1's threshold is defined against
existed only in `checkpoints/task056-policy-v1/*.run.json`. They are **re-derived by executing
`cloning.evaluate_policy` against the four checkpoints on the val split** and transcribed here and
into the manifest **before this protocol is frozen**.

> **Pre-committed, before the re-derivation's answer was known:** if the re-derived table disagrees
> with the training-time record, §1.1(b) is **withdrawn, not patched**, and this protocol's premise
> is re-argued from the corrected numbers. No argument is entertained about whether the discrepancy
> "would have mattered."

**The re-derivation was performed and it reproduces the training-time record exactly.** Executed
`cloning.evaluate_policy` on the val split (**7 723 rows, 15 surviving val roots** — matching
`benchmarks/manifests/apple-policy-v1.json` `val_rows_sampled` = 7723), rebuilding each arm's
feature source from its own checkpoint provenance by the same procedure
`scripts/evaluate_policy.py` uses. Maximum absolute disagreement with the training-time
`validation[best_step]` record: **3.6e-7, 6.6e-7, 6.9e-7, 1.3e-7** for A0–A3 on the per-dimension
errors and ≤ 6e-8 on the output standard deviations — float32 round-trip. Selection scores
reproduce: A1 0.011059 < A0 0.013129 < A2 0.017935 < A3 0.023404. **Nothing is withdrawn.**
Total 16.1 s on CPU.

**Table A — median |predicted − expert| per dimension, val split, 7 723 rows. Frozen.**

| dimension | A0 | A1 | A2 | A3 |
|---|---|---|---|---|
| `right_dx` | 0.01896 | 0.02099 | 0.02715 | 0.03029 |
| `right_dy` | 0.01862 | 0.02504 | 0.02886 | 0.02806 |
| `right_dz` | 0.01569 | 0.01333 | 0.01992 | 0.02832 |
| `right_droll` | 0.00985 | 0.00726 | 0.01248 | 0.01873 |
| `right_dpitch` | 0.01178 | 0.00970 | 0.02007 | 0.02776 |
| `right_dyaw` | 0.01005 | 0.00673 | 0.01474 | 0.02150 |
| `right_grasp` | 0.01043 | 0.00731 | 0.00931 | 0.01105 |

**Table B — predicted output standard deviation per dimension, same rows.**

| dimension | A0 | A1 | A2 | A3 |
|---|---|---|---|---|
| `right_dz` | 0.27349 | 0.27348 | 0.27084 | 0.26564 |
| `right_droll` | 0.22866 | 0.23183 | 0.22644 | 0.22069 |
| `right_grasp` | 0.97942 | 0.97760 | 0.97368 | 0.97093 |
| `right_dx` / `dy` / `dpitch` / `dyaw` | 0.160 / 0.154 / 0.190 / 0.105 | 0.164 / 0.150 / 0.188 / 0.105 | 0.161 / 0.154 / 0.195 / 0.103 | 0.150 / 0.140 / 0.179 / 0.086 |

Table A is the sole source of D1's thresholds (§5), which are **3 × the cell**, and those 24
numbers are frozen in the manifest before any probe runs.

**The 4.5× spread quoted in §5's derivation is from Table A and is now verified rather than
estimated:** the per-dimension errors range from **0.00673** (A1 `right_dyaw`) to **0.03029**
(A3 `right_dx`), a ratio of **4.50**.

---

## 10. Engineering constraints

- **Ordering constraint, blocking.** `ClonedPolicy.implementation_sha256` is `sha256(policy.py)` and
  `ClonedPolicy.load` refuses a mismatch, which is demonstrated rather than theoretical
  (`apple_policy_v1_results.md` §9: a source fix made all four checkpoints unloadable).
  **D1, D2 and D3 run to completion before any edit to `src/embodied_jepa/policy.py`.** If a head
  change is ever preregistered, it goes in a new module.
- **`models/base.py` and `models/lewm.py` are untouched by this protocol**, so the E0 checkpoint and
  every pre-flight number stay valid. The branch CI check that loads the checkpoint and asserts the
  hash must pass.
- **Blocking requirements on the additive runner change**, each discharged by writing the violation
  first and watching the test fail:
  1. **Signed per-dimension command logging — landed with this preregistration.**
     `command_statistics` now records a `signed` block per dimension (mean, q10/q50/q90, min, max,
     rate at each signed maximum, rate positive) **alongside** the absolute fields, which keep
     their exact former values so reports stay comparable with TASK-056's. `run_attempt` also
     records `first_command`, the signed step-zero 7-vector D1 consumes.
     Discharged by two tests that **fail on the pre-change runner**: one exhibits the defect by
     asserting that a pure descent and a pure ascent produce byte-identical absolute statistics
     and are separated only by the signed block; the other does the same for an oscillating versus
     a steady command, whose |dz| median and max are equal.
  2. **`clipped_commands` is a single count over any dimension — landed.**
     `clipped_commands_per_dimension` is recorded alongside the any-dimension counter, which keeps
     its former meaning exactly. `apple_policy_v1_results.md` §14.1 records the single count as an
     instrument limitation.
  3. **A third requirement was proposed and is WITHDRAWN, because it was wrong.** The claim was
     that `stages[t]` is one step stale. It is not. `scorer.evaluate()` is the **last** statement of
     the loop body and `robot.observe()` the **first** of the next iteration, with nothing mutating
     the simulator between them, so the stage label is contemporaneous with the observation the
     command was computed from. The claim was made by reading the code and retracted by tracing the
     ordering — the same distinction this protocol is about — and it is recorded here rather than
     quietly dropped. (The stages *are* latched by `|=`, so the label means "the furthest stage
     reached by this step", which is what `apple_policy_v1_results.md` §14 already states.)
- **Preserved-property rule, for every fix above.** Name the property that existed before the fix
  and must still hold after it, and say how it was checked. Not "the tests pass" — the named
  property and the check (`apple_policy_v1_results.md` §7).
- Everything else in `scripts/evaluate_policy.py` is kept unchanged: the cohort-D whitelist with its
  type-boundary coercion, the configured-bounds clipping and pinned-component check,
  `frozen_flag_for`, `GUARD_REFUSALS`, the render-size assertion, the arm/metadata cross-check.
- New artifacts under `outputs/task057-diagnostics/`. **Nothing under `data/`, `checkpoints/` or
  `outputs/` is overwritten**; the runner refuses to write over an existing path.

---

## 11. Not done in this task (declared)

- **Any model, loss, head or output-parameterisation change.** §1.1 is why.
- **Any retraining of any arm.** The handover's standing rule: an arm that died on development is
  recorded `dead_on_development` and left standing.
- **A4**, the readout-driven scripted controller. Still deferred, still not dropped.
- **Cohort C.** Not opened, not referenced by any gate, not requested.
- **The test split.** Never decoded.
- **The on-policy-relabeling argument for a follow-up.** Recorded as a named candidate in the
  results document with its reasoning, and deliberately **not** preregistered here.

---

## 12. Process

- One MC task (**TASK-057**), one agent, resumable. No second agent is spawned on this task.
- This document and its manifest are merged **before** any probe runs. The merge requires an
  independent reviewer's **reported** APPROVE.
- **Two separate authorizations, in order, neither implying the other:** merge the preregistration;
  start the diagnostic run. There is no third, because cohort C is not opened by this task.
- **A gated run starts only on the pre-run reviewer's REPORTED verdict, delivered as a message —
  never on a review file read from disk.** The same rule applies to merging.
- On any voiding control (B1, B2, B3) the executing agent stops and reports rather than repairing
  as a reflex.
