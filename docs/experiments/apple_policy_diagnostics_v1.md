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

### 1.1 The mechanism TASK-056 recorded is not supported by the statistics it was read off

`apple_policy_v1_results.md` §12 and §16, and `task056_handover.md` §5, record a candidate
mechanism: the expert's `right_dz` sits at exactly 0.400 on 44.89 % of commands, a smooth-L1
regression head hedges toward the interior of that bimodal target, and the resulting under-shoot
means the policy never reaches the object inside 1000 steps. It is recorded there — correctly —
as **a consistent story, not a demonstrated mechanism**.

**It is not supported by the statistics it was read off.** Three findings, each traced to the code
that computes it. The verdict language is deliberately weaker than an earlier draft's
("contradicted", "refuted", "did not occur"): §1.2 sets the standard that a quantity a statistic
*cannot* compute is **unmeasured** rather than refuted, and §1.1 has to meet its own standard.

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

**(b) The hedging signature is absent offline, measured directly for the first time.**

**The statistic it was originally read off cannot test it, and that is recorded first.**
`cloning.action_error` (`cloning.py`) computes **one unconditional per-dimension median over all
rows**. There is no saturated-row conditional anywhere in the file. Saturation on val is
**44.503 %** — below half — so the unconditional median necessarily sits inside the
*non-saturated* group, and the cell is **mathematically incapable** of testing "large error on the
saturated rows". Likewise `output_std_per_dimension` is the **predictions only**: the *target*
standard deviation is never computed anywhere, so "compressed" had **no referent**. An earlier
draft of this section drew a conclusion from both. That conclusion is withdrawn.

**The three conditionals that can test it were computed for this protocol** (§9.1), on the same
7 723 val rows, and they are reported whatever they say:

| quantity | A0 | A1 | A2 | A3 |
|---|---|---|---|---|
| median \|err\| `right_dz`, **rows where \|target dz\| ≥ 0.4** | **0.01240** | 0.01283 | 0.01349 | 0.02529 |
| median \|err\| `right_dz`, rows where it is **not** saturated | 0.01793 | 0.01362 | 0.02582 | 0.03292 |
| **mean predicted `dz` on target = +0.400 rows** | **+0.3750** | +0.3719 | +0.3722 | +0.3537 |
| **mean predicted `dz` on target = −0.400 rows** | **−0.3869** | −0.3903 | −0.3760 | −0.3736 |
| predicted ÷ **target** `right_dz` std (target std **0.27905**) | 0.980 | 0.980 | 0.971 | 0.952 |
| predicted ÷ **target** `right_grasp` std (target std **0.98009**) | 0.999 | 0.997 | 0.993 | 0.991 |

On the rows where the expert saturates, the heads are **more** accurate, not less; conditioned on
the target being at +0.400 they predict **+0.35 to +0.38**, conditioned on −0.400 they predict
**−0.37 to −0.39**; and the predicted standard deviation is **95–100 % of the target's**, not a
fraction of it.

**What this licenses, stated exactly, and it is deliberately only half the mechanism.**

> **Measured on the validation split, the hedging signature is absent offline: saturated-row error
> is lower than unsaturated, conditional means reach 88–98 % of the boundary on both signs, and the
> predicted-to-target standard-deviation ratio is 0.952–0.980.**

**The online half is unmeasured and stays unmeasured.** Whether dz behaviour caused the closed-loop
failure cannot be read from the TASK-056 record at all, because §1.2 establishes that the signs of
every closed-loop command are unrecoverable. The prediction in `apple_policy_v1.md` §12 and declared
risk R5 of its §8 range over **both** halves; only one is measured, so the words "refuted",
"contradicted" and "did not occur" are **not used unqualified anywhere in this protocol** — they
would claim the unmeasured half. Every statement here carries "offline" or "on the validation
split", and that qualifier is load-bearing rather than cautious phrasing.

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
  the collector advanced it, and computed strictly **after** `policy.act` returns. **Two properties
  of this object are declared here because both affect a gate:**
  - **It is exhausted after 805 commands.** The phase table sums to
    130 + 80 + 45 + 150 + 160 + 100 + 60 + 80 = **805**, and `action()` **raises** once `done`.
    Every TASK-056 attempt that was not stopped by the embodiment ran to 1000, so on nearly every
    attempt the shadow expert dies around step 805 — and `run_attempt`'s bare `except BaseException`
    would turn that into a dead run with no report. §2.1 preregisters the behaviour.
  - **It is a hybrid reference after roughly step 130, not the expert's own trajectory.**
    `advance(result)` is called on **the policy's** executed result, so the expert's phase counter
    advances on the policy's clock rather than on its own progress. A policy that has gone nowhere
    still drives the reference into `descend`, then `lift`. **The shadow expert is therefore
    "expert targets on a policy-driven clock", and is named that way wherever it is used.**
- **Departure at step t** — median over the six arm dimensions of |policy − shadow expert|.
- **Configuration** — one of the ten named below. Nothing else is a configuration.
  **The eight G-SUB candidates**, which are the only configurations G-SUB ranges over:
  `all_translation` (= {dx, dy, dz}), `dx`, `dy`, `dz`, `droll`, `dpitch`, `dyaw`, `grasp`.
  **Two controls, which are configurations but are NOT G-SUB candidates:** `none` (the empty set,
  B2's control) and `full` (all seven dimensions, B3's control).

  > **Why the controls are excluded by name.** An earlier draft defined a configuration as "the
  > named subset of the seven free dimensions" and let G-SUB range over *any* configuration. Under
  > that wording `full` **is** a configuration, B3 requires `full` to reach ≥ 14/16, and G-SUB
  > passes if any configuration reaches ≥ 8/16 — so **B3 passing implied G-SUB passing, while B3
  > failing voided the run, and the gate wired to the abandonment clause had no failing path at
  > all.** The candidate list was also present only in §3's prose, never in this section and never
  > in the gate: an unenumerated scope term deciding an outcome, which is verbatim the TASK-056
  > defect this section exists to prevent. It is enumerated here and in the manifest.

### 2.1 Shadow-expert exhaustion — preregistered behaviour

- The runner catches the exhaustion explicitly, by the enumerated message, and **never** by a broad
  `except`. An unrelated `ContractError` still propagates and still kills the run loudly.
- **In D1 and D2** (diagnostic, no substitution): the shadow expert stops being recorded at
  exhaustion, the attempt continues to its cap, and the report records
  `shadow_expert_exhausted_at_step`. Departure is undefined past that step and is recorded as
  `null`, never as zero.
- **In D3** (substitution, and therefore the gate): the attempt **terminates** at exhaustion with
  termination reason `shadow_expert_exhausted`, and **counts as a non-success.** This is not a
  concession: the oracle's own budget is 805 commands and `scripted_oracle` succeeds inside it
  (24/24 pooled), so a configuration that cannot reach `grasp` within the expert's own budget has
  not been cut short by the rule.
- `shadow_expert_exhausted_at_step` is reported for every attempt of every configuration, so a
  reader can see how many attempts the rule bound.
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
`grasp = -1.0` in `scripted.py`'s phase table.

**The primary report is the signed step-zero `right_grasp` value, per arm per seed — the full
4 × 16 table of numbers, not a count.** No threshold: the reference is known-correct.

**Why a count was replaced by the values.** An earlier draft reported "the count of seeds where the
step-zero grasp is `> 0`". That criterion is **one-sided, and an inert policy passes it**: a head
emitting exactly `0.0` — commanding neither open nor closed — scores **zero** sign errors and reads
as correct, while being wrong by a full unit against a reference that is known exactly. That is the
blind-baseline failure `apple_policy_v1.md` §5.1 was rewritten to remove, recurring in a new place,
and it is the **second** threshold in this design with that shape (the first was the original
G-SUB, §2).

The derived summary, defined two-sidedly against the known reference, is reported alongside the
values: **the count of seeds where |step-zero `right_grasp` − (−1.0)| ≥ 1.0**, equivalently where
the commanded value is ≥ 0.0. It fires on an inert `0.0` (error exactly 1.0) and on a fully closed
`+1.0` (error 2.0), and not on `−1.0` (0.0) or `−0.5` (0.5). Its reading: *the policy is not
commanding the hand open at the one instant in the episode where the correct command is known and
unambiguous.*

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
policy. **Ten configurations run on the 16 seeds**, and §2 enumerates them: the **eight G-SUB
candidates** (`all_translation` and one per dimension) plus the **two controls** `none` (B2) and
`full` (B3). **The controls are not candidates**, so neither can pass the gate — see §2's box for
the failure that wording produced.

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

> **G-SUB passes iff at least one of the eight G-SUB CANDIDATE configurations enumerated in §2
> reaches ≥ 8/16 grasp resets.** `none` and `full` are controls and are **not** candidates; a
> passing `full` therefore cannot pass the gate.

*The null this gate is defended against is **the pure policy's own rate**, 1/16*
(`outputs/task056-cohort-d/a2.json`, A2's `result.grasp_resets`). That is the rate the substitution
has to beat for the substitution to have done anything, and it is the only null with a defensible
claim to being this experiment's. Against p₀ = 1/16 the one-sided binomial probability of ≥ 8/16 is
**1.90 × 10⁻⁶**, and the family-wise probability across the eight candidates is
**1 − (1 − 1.90 × 10⁻⁶)⁸ = 1.5 × 10⁻⁵**. The gate is strong against the null that matters.

*Derivation of 8/16.* It is the midpoint between the pure-policy 1/16 = 6.25 % and the 100 %
ceiling: the effect must be at least half the distance to the ceiling, because a smaller one would
not justify building anything. **It is a magnitude threshold, not a significance threshold**, and D
is a non-gating development cohort from which no p-value is a result.

> **Correction carried in the open, because a frozen manifest holding a wrong number is the thing
> this project has bled over most.** An earlier draft defended this gate against a `demo_replay`
> null of p₀ = 1/3 and quoted "the one-sided binomial probability of ≥ 8/16 is 0.189". **0.189 is
> wrong**: `P(X ≥ 8 | n = 16, p = 1/3) = 0.126501`. The figure 0.189 corresponds to p₀ = 0.363344,
> a value nothing in this protocol uses. Worse, at the correct value the **family-wise** probability
> across the candidates is **1 − (1 − 0.126501)⁸ = 0.661**, so under that null the gate would have
> been close to a coin flip rather than the screen it was presented as. **The `demo_replay` null is
> dropped entirely**; it was the wrong reference anyway, since `demo_replay` is an open-loop
> baseline and G-SUB asks whether substitution moves *this policy*. Both corrected numbers are in
> the manifest.

*Fixed reference points, all before the run:* pure-policy A2 = **1/16**; `scripted_oracle` =
**24/24 pooled** on three prior wide cohorts (`apple_wide_grasp_closure_results_v3.md`,
`apple_wide_object_ceiling_results_v2.md`, via `apple_policy_v1.md` §5.2 G5).

**Missing values.** A gate that cannot be evaluated counts as failed. D1 and G-SUB are the only
gates.

### 5.1 Abandonment clause

> **If G-SUB fails — none of the eight G-SUB candidate configurations enumerated in §2,
> including `all_translation`, reaches 8/16 grasp resets —
> then no single command channel carries the failure, and the preregistered Outcome D abandonment
> clause of `apple_policy_v1.md` §7 fires as written.** This control line stops on this corpus and
> this camera. No further loss, head or output-parameterisation variant is preregistered on it. The
> next task is a perception/data task. Cohort C remains unconsumed. The executing agent stops and
> reports and does not select the follow-up itself.

> **Precedence over §5.2, and it is the D1 result that takes it.** If D1 fails for ≥ 3 of the 4
> arms, the observation the controller receives is not the one the trainer used. G-SUB's result is
> then **uninterpretable rather than negative** — a sweep conducted through a broken channel
> measures the channel. In that case the abandonment clause does **not** fire; the task reports a
> pipeline defect, and the line continues for the purpose of fixing it.
>
> **This exemption is available exactly once.** Once the defect is fixed, a re-run whose G-SUB
> fails with D1 passing fires the clause as written. **The exemption is recorded as spent in the
> results document the first time it is claimed**, so a second claim is visibly unavailable.
>
> The "once" is the load-bearing half. Without it this is an unbounded escape from abandonment,
> which is worse than the contradiction it replaces.

### 5.2 Success clause

**§5.1's precedence rule governs this section.** If D1 fails for ≥ 3 of the 4 arms *and* G-SUB
fails, **clause (a) applies and the abandonment clause does not fire** — once. An earlier revision
left §5.1 saying "stop" and §5.2 saying "continue" on that joint outcome, with §7 listing the two
outcomes side by side and no precedence, which left the executing agent free to choose **after
seeing the numbers**. That is the defect a preregistration exists to make impossible.

The line continues iff **either**:

- **(a)** D1 fails for ≥ 3 of the 4 arms, **or** D1-grasp's two-sided count — seeds where
  |step-zero `right_grasp` − (−1.0)| ≥ 1.0 — is ≥ 8/16 for ≥ 3 of the 4 arms. Either is a concrete, locatable discrepancy between the
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

**And the substituted channel is not the expert's own trajectory — it is expert targets on a
policy-driven clock (§2).** `advance(result)` steps the phase counter on the policy's executed
result, so after roughly step 130 the reference has moved into `descend` and later `lift`
regardless of where the palm actually is. **A D3 configuration can therefore inject a `lift`-phase
ascent into an attempt whose hand has never closed on anything.** That is a confound on the only
gate, it is declared here before any number exists, and two things follow:

- `shadow_expert_phase` is logged per step in D2 and D3, so the hybrid is visible in the trace
  rather than inferred.
- **A G-SUB pass whose attempts spent most of their commands past the expert's `close` phase is
  reported as such**, next to the pass, because a rescue driven by a mistimed reference is a
  different finding from a rescue driven by a correct one. This is a reporting requirement, not a
  threshold, and it does not alter the gate.

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
- **Outcome X — G-SUB fails *and* D1 passes for ≥ 2 of the 4 arms.** §5.1 fires. The line stops.
- **Outcome P/X precedence — G-SUB fails *and* D1 fails for ≥ 3 of the 4 arms.** **Outcome P takes
  precedence and Outcome X does not fire**, once: G-SUB conducted through a broken observation
  channel measures the channel, so its failure is uninterpretable rather than negative. The
  exemption is **recorded as spent** in the results document, and a re-run after the fix whose
  G-SUB fails with D1 passing fires §5.1 as written.
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
| `right_dx` | 0.018958 | 0.020990 | 0.027149 | 0.030289 |
| `right_dy` | 0.018620 | 0.025044 | 0.028862 | 0.028059 |
| `right_dz` | 0.015693 | 0.013327 | 0.019921 | 0.028324 |
| `right_droll` | 0.009851 | 0.007255 | 0.012483 | 0.018732 |
| `right_dpitch` | 0.011776 | 0.009703 | 0.020074 | 0.027760 |
| `right_dyaw` | 0.010054 | 0.006734 | 0.014743 | 0.021500 |
| `right_grasp` | 0.010430 | 0.007306 | 0.009306 | 0.011050 |

**Table B — predicted output standard deviation per dimension, same rows.** (Predicted only; the ratio against the *target* std is Table D, §9.1.)

| dimension | A0 | A1 | A2 | A3 |
|---|---|---|---|---|
| `right_dx` | 0.160344 | 0.164477 | 0.161114 | 0.150127 |
| `right_dy` | 0.153921 | 0.150140 | 0.154288 | 0.139683 |
| `right_dz` | 0.273488 | 0.273484 | 0.270837 | 0.265636 |
| `right_droll` | 0.228657 | 0.231827 | 0.226441 | 0.220689 |
| `right_dpitch` | 0.190188 | 0.188090 | 0.195052 | 0.178829 |
| `right_dyaw` | 0.105481 | 0.105059 | 0.102762 | 0.085868 |
| `right_grasp` | 0.979420 | 0.977598 | 0.973685 | 0.970933 |

**The manifest is authoritative for every cell above.** `benchmarks/manifests/apple-policy-diagnostics-v1.json` `frozen_offline_error_table_A.values` holds these numbers at 6 decimal places, and D1's 24 thresholds are exactly **3 ×** those cells. The tables here print the manifest's own values at the manifest's own precision, so the document and the manifest cannot disagree — and `tests/test_policy_preflight.py` parses this table and asserts cell-by-cell agreement, because a test that compares the manifest against itself cannot see document drift.

> **Why this is spelled out.** An earlier revision printed Table A rounded to 5 decimal places while the manifest carried 6, so **22 of 28 cells disagreed** and the sentence "D1's thresholds are 3 × the cell" was false of the document's own printed numbers: 3 × 0.01896 = 0.05688 against a stored threshold of 0.056874. The manifest was internally perfect throughout and its self-consistency test passed, which is exactly why the drift was invisible: **a consistency check that compares an artifact with itself certifies nothing about the artifact a reader actually reads.**

**The spread quoted as the derivation of the 3× factor is computed from these cells:** **0.006734** (A1 `right_dyaw`) to **0.030289** (A3 `right_dx`), a ratio of **4.498**. Verified, not estimated.

### 9.1 The conditionals the unconditional median cannot compute (F2), and the phase-restricted error (F7)

Both are published **before the run**, on the same 7 723 val rows, because D1's thresholds are defined against Table A and a reader has to be able to see what Table A does and does not test.

**Val saturation, stated because Table A's 44.886 % is the TRAIN cohort:** on val, |target `right_dz`| ≥ 0.400 on **44.503 %** of rows (+0.400 on 17.662 %, −0.400 on 26.842 %), target mean -0.0041.

**Table C — `right_dz` error split by whether the expert saturates, and the conditional means.**

| quantity | A0 | A1 | A2 | A3 |
|---|---|---|---|---|
| median \|err\|, saturated rows (\|target\| ≥ 0.4) | 0.01240 | 0.01283 | 0.01349 | 0.02529 |
| median \|err\|, unsaturated rows | 0.01793 | 0.01362 | 0.02582 | 0.03292 |
| median \|err\|, unconditional (Table A) | 0.01569 | 0.01333 | 0.01992 | 0.02832 |
| **mean predicted dz** \| target = +0.400 | +0.3750 | +0.3719 | +0.3722 | +0.3537 |
| **mean predicted dz** \| target = −0.400 | -0.3869 | -0.3903 | -0.3760 | -0.3736 |
| median predicted dz \| target = +0.400 | +0.3953 | +0.3904 | +0.3912 | +0.3714 |
| median predicted dz \| target = −0.400 | -0.4018 | -0.4048 | -0.3982 | -0.4036 |

**Table D — predicted ÷ TARGET standard deviation.** Table B gave predictions only, which is why "compressed" had no referent.

| dimension | target std | A0 | A1 | A2 | A3 |
|---|---|---|---|---|---|
| `right_dx` | 0.16382 | 0.979 | 1.004 | 0.983 | 0.916 |
| `right_dy` | 0.16008 | 0.962 | 0.938 | 0.964 | 0.873 |
| `right_dz` | 0.27905 | 0.980 | 0.980 | 0.971 | 0.952 |
| `right_droll` | 0.23148 | 0.988 | 1.001 | 0.978 | 0.953 |
| `right_dpitch` | 0.19233 | 0.989 | 0.978 | 1.014 | 0.930 |
| `right_dyaw` | 0.10637 | 0.992 | 0.988 | 0.966 | 0.807 |
| `right_grasp` | 0.98009 | 0.999 | 0.997 | 0.993 | 0.991 |

**Table E — median absolute error by collector phase (F7), all seven dimensions.** Reported so a reader can see where in the episode the offline fit is weakest before any closed-loop number exists.

*A0*

| phase | rows | dx | dy | dz | droll | dpitch | dyaw | grasp |
|---|---|---|---|---|---|---|---|---|
| `close` | 675 | 0.0175 | 0.0161 | 0.0207 | 0.0192 | 0.0130 | 0.0106 | 0.0168 |
| `descend` | 1200 | 0.0213 | 0.0153 | 0.0068 | 0.0085 | 0.0100 | 0.0075 | 0.0268 |
| `lift` | 2181 | 0.0160 | 0.0192 | 0.0139 | 0.0121 | 0.0096 | 0.0071 | 0.0042 |
| `lower` | 699 | 0.0218 | 0.0182 | 0.0220 | 0.0163 | 0.0138 | 0.0203 | 0.0167 |
| `orient` | 1950 | 0.0225 | 0.0213 | 0.0161 | 0.0034 | 0.0130 | 0.0084 | 0.0102 |
| `release` | 178 | 0.0351 | 0.0407 | 0.0889 | 0.0488 | 0.0169 | 0.0336 | 0.0151 |
| `transfer` | 840 | 0.0123 | 0.0146 | 0.0210 | 0.0149 | 0.0136 | 0.0213 | 0.0089 |

*A1*

| phase | rows | dx | dy | dz | droll | dpitch | dyaw | grasp |
|---|---|---|---|---|---|---|---|---|
| `close` | 675 | 0.0242 | 0.0261 | 0.0219 | 0.0108 | 0.0130 | 0.0141 | 0.0099 |
| `descend` | 1200 | 0.0196 | 0.0243 | 0.0091 | 0.0114 | 0.0090 | 0.0063 | 0.0236 |
| `lift` | 2181 | 0.0227 | 0.0271 | 0.0116 | 0.0091 | 0.0076 | 0.0054 | 0.0051 |
| `lower` | 699 | 0.0232 | 0.0158 | 0.0163 | 0.0074 | 0.0094 | 0.0072 | 0.0211 |
| `orient` | 1950 | 0.0215 | 0.0242 | 0.0140 | 0.0024 | 0.0116 | 0.0053 | 0.0038 |
| `release` | 178 | 0.0507 | 0.0614 | 0.0737 | 0.0295 | 0.0555 | 0.0170 | 0.0177 |
| `transfer` | 840 | 0.0112 | 0.0190 | 0.0140 | 0.0072 | 0.0082 | 0.0108 | 0.0088 |

*A2*

| phase | rows | dx | dy | dz | droll | dpitch | dyaw | grasp |
|---|---|---|---|---|---|---|---|---|
| `close` | 675 | 0.0248 | 0.0294 | 0.0265 | 0.0161 | 0.0226 | 0.0181 | 0.0093 |
| `descend` | 1200 | 0.0234 | 0.0217 | 0.0084 | 0.0185 | 0.0188 | 0.0112 | 0.0173 |
| `lift` | 2181 | 0.0270 | 0.0300 | 0.0168 | 0.0161 | 0.0175 | 0.0120 | 0.0057 |
| `lower` | 699 | 0.0352 | 0.0314 | 0.0270 | 0.0169 | 0.0196 | 0.0217 | 0.0254 |
| `orient` | 1950 | 0.0258 | 0.0323 | 0.0256 | 0.0048 | 0.0264 | 0.0162 | 0.0080 |
| `release` | 178 | 0.0613 | 0.0448 | 0.0925 | 0.0452 | 0.0428 | 0.0322 | 0.0433 |
| `transfer` | 840 | 0.0258 | 0.0233 | 0.0271 | 0.0176 | 0.0147 | 0.0158 | 0.0112 |

*A3*

| phase | rows | dx | dy | dz | droll | dpitch | dyaw | grasp |
|---|---|---|---|---|---|---|---|---|
| `close` | 675 | 0.0436 | 0.0300 | 0.0253 | 0.0309 | 0.0260 | 0.0219 | 0.0113 |
| `descend` | 1200 | 0.0203 | 0.0276 | 0.0155 | 0.0340 | 0.0213 | 0.0127 | 0.0445 |
| `lift` | 2181 | 0.0283 | 0.0245 | 0.0289 | 0.0171 | 0.0265 | 0.0229 | 0.0082 |
| `lower` | 699 | 0.0287 | 0.0260 | 0.0385 | 0.0325 | 0.0355 | 0.0214 | 0.0276 |
| `orient` | 1950 | 0.0306 | 0.0273 | 0.0326 | 0.0082 | 0.0335 | 0.0195 | 0.0084 |
| `release` | 178 | 0.1205 | 0.0660 | 0.1742 | 0.0668 | 0.0765 | 0.0308 | 0.0196 |
| `transfer` | 840 | 0.0403 | 0.0322 | 0.0326 | 0.0196 | 0.0160 | 0.0306 | 0.0086 |

**Reading rule for Table E, fixed in advance:** it is an OFFLINE error on recorded frames. A phase with a low error here is not thereby a phase the controller handles — that is the covariate-shift gap this whole protocol exists to measure, and `apple_policy_v1.md` §2 already records that a readout measured on recorded frames is not one that survives the state distribution its own controller induces.

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
     reached by this step", which is what `apple_policy_v1_results.md` §14 already states. And
     `stages[0]` is an **absent** label rather than a stale one: at the first command the scorer
     has not run, so no stage has been evaluated yet, which is a different thing from a label
     lagging behind one that exists.)
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
