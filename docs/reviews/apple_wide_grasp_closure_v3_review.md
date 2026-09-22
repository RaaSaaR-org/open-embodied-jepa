# TASK-051 pre-run review: object-aware ceiling v3 (caged grasp closure)

Fresh-context reviewer subagent, independent of the author, read-only, on branch
`feat/task-051-grasp-closure-v3`. Review started at `dc44f42`; this is the final
version, checked against **`b673f3d` (R1)**. The gated evaluation was **not**
executed and no 45xxx reset was stepped by this review.

> **This document supersedes the draft recorded in `b673f3d`.** While the review
> was still in progress the author committed an interim draft of it (verdict
> "CLEAR TO RUN") together with an R1 revision applying its non-blocking items.
> The verification of the diagnosis against the raw scratch outputs finished after
> that commit and changed the verdict. R1 resolved what is now **B1**; **B2, B3
> and B4 are still present at `b673f3d`**, and R1 in fact propagated B3 into a new
> risk bullet. The `b673f3d` commit message's "Verdict CLEAR TO RUN, no blocking
> defect" should be corrected in the results document.

## Verdict: BLOCK — narrowly, on three remaining factual corrections to the record

**The machinery is ready to run.** The v3 change is confined to `_bounds()` in the
`close` phase; it cannot produce an empty, degenerate or contract-violating
candidate set; it does not reduce candidate feasibility relative to v2; and the CEM
still runs on every close command, so the runtime parity checks stay non-vacuous.
v1 and v2 plan and gate output were verified **byte-identical in content** to
`origin/main` by direct comparison, not by the branch's own tests. The gate cohort
45200–45207 has never been stepped anywhere on this filesystem. Simulator physics
and contact parameters are untouched. The gate, thresholds, cohorts, budget and
frozen command are unambiguous, mutually consistent, and reproduce the manifest
exactly. **The v3 design itself is sound and well supported by the paired closure
experiment.**

**What blocks is the record, not the run.** Three statements in the two committed
documents are factually wrong or contradicted by the raw scratch data I
re-aggregated. Two of them (B2, B3) appear in the *preregistration* itself, and two
(B3, B4) are repeated verbatim in the shipped `object_ceiling_v3.py` module
docstring. A preregistration is frozen at a commit and cited by the results
document, so these would become permanent.

Clearing this block is a **documentation edit only**. No code, seed, threshold,
cohort, budget, parameter or command needs to change; no re-tuning is needed; the
TRAIN smoke does not need to be repeated. Once B2–B4 are applied, this is CLEAR TO
RUN — I found nothing else that would make the run's conclusion unsound or
unreproducible.

## Blocking findings

All were reproduced by me directly from
`/Users/sebastian/develop/emai/experiments/JEPA/open-embodied-jepa/outputs/task051-scratch/`
and the harness sources. All are document-only.

### B1. The protocol's guard-refusal risk was wrong by ~6× — RESOLVED in R1

At `dc44f42` the protocol's "Known risks" said *"one of **twelve** `cage` replays
stopped on the embodiment's measured joint-velocity guard (5 rad/s)."*
Re-aggregating every `cage` replay across `scan-a/b/c` and `fork-smoke*`:

```
cage    72 runs   24 seeds   70 grasped   {grasped: 70, guard_refused: 1, budget: 1}
        guard refusal: 1 of 72  (seed 49112, rng 69112)
```

The true rate is **1 of 72**, not 1 of 12 — an ~6× overstatement in the
preregistration's only quantified risk, and the number that would have been cited
if the gate failed at 7/8. **R1 (`b673f3d`) corrected this** to "one of 72 … and
one more reached the replay budget: 2/72 ≈ 2.8% per attempt", with the resulting
≈ 0.98 probability of clearing the ≥ 7/8 threshold stated. Verified against the
raw data: correct as now written. Recorded here for the audit trail only.

### B2. "The scripted collector reaches 16/16 on the same resets" is the grasp stage, not success

This sentence appears in the **preregistration** (`apple_wide_grasp_closure_v3.md`,
"What changes from v2, and only that": *"Simulator physics and contact parameters
are **unchanged**. … the scripted collector reaches 16/16 on the same resets under
the same physics."*) and again in the diagnosis's Conclusion. It is load-bearing:
it is the protocol's justification that the physics needs no scrutiny.

But the forensics harness truncates the scripted arm shortly after the close —
`outputs/task051-scratch/probe.py:204`:

```python
if stop_after_close and close_start is not None and step > close_start + 85:
```

so **full task success was never measured on 49100–49115**. The 16/16 is the
scorer's *grasp stage* / "no ejection", and the recorded traces carry
`success: False` throughout because the run stops before transport and place.
Restate as "16/16 grasp stage with no ejection" in both documents.

### B3. The "flat cost" mechanism is contradicted in its vertical half

The diagnosis (Finding 3, bullet 2), the **shipped module docstring** of
`src/embodied_jepa/object_ceiling_v3.py`, and the commit message all assert that
the close cost is nearly flat in the palm command, specifically: *"the descent is
blocked, so a candidate that commands −0.5 in z and one that commands 0 reach
almost the same palm position over the six-step horizon."*

The recorded commands contradict this. Over the eleven ramp commands of all
sixteen v2 closes, the **mean applied dz is −0.22 to −0.41 on every single seed**,
whereas undirected clipped noise (σ 0.3, bounds ±0.5, zero mean) has expectation
**0.000**. The CEM is demonstrably *selecting* descent, so the cost is not flat in
z. At the measured ~0.15 cm realised sink per command, a six-step horizon separates
a full-descent from a zero-descent candidate by ~0.9 cm of palm z — about 12× the
0.078 cm lateral sensitivity the same bullet computes.

The flatness claim **is** supported laterally, and that is the half that justifies
v3's lateral pin: mean absolute lateral command 0.219–0.313 against the ~0.227
expectation of pure clipped noise, i.e. statistically indistinguishable from
undirected noise. Narrow the claim to the lateral command.

Present at `b673f3d` in four places, one of them newly introduced by R1:

- `docs/experiments/apple_grasp_closure_diagnosis.md:83` ("that cost is nearly
  flat") and `:89` ("reach almost the same palm position over the six-step
  horizon");
- `docs/experiments/apple_wide_grasp_closure_v3.md:17` ("the v2 close cost is
  nearly flat in the palm command");
- **`docs/experiments/apple_wide_grasp_closure_v3.md:272`** — *new in R1*: the
  risk bullet "**v3 forbids rising but does not require sinking.** The close cost
  is nearly flat in z while the palm is blocked…". This was added in response to
  my earlier non-blocking item, which I had written from the diagnosis's own
  (incorrect) premise. The *risk* is worth declaring and the bullet's empirical
  conclusion is right; only its "nearly flat in z" premise needs correcting — the
  measured evidence is that the planner already selects descent (mean applied dz
  −0.22…−0.41 on all sixteen v2 closes under bounds that permitted rising).

### B4. Finding 3's travel figures are at first contact, not "during the ramp", and the gloss is inverted

Finding 3 says: *"The consequence is the palm travel **during the ramp**: the
collector sinks 0.97–0.99 cm and drifts 0.20–0.22 cm laterally on all sixteen
resets, while v2 sinks 0.40–1.85 cm and drifts 0.05–1.22 cm."* The same two ranges
are quoted in the v3 module docstring and the commit message.

I recomputed all four ranges. They are travel **at the first hand–apple touch
command**, which is what the harness computes
(`outputs/task051-scratch/correlate.py:39-41`, `sink_at_touch_cm` /
`drift_at_touch_cm`):

| statistic | v2 ceiling | scripted |
|---|---|---|
| over a fixed 11-command ramp (`rows[0]→rows[11]`) | 0.94–2.31 / 0.05–1.13 cm | 0.97–0.99 / 0.20–0.22 cm |
| **at first touch** | **0.40–1.85 / 0.05–1.22 cm** | **0.97–0.99 / 0.20–0.22 cm** |

The doc's figures are the second row. For the collector the two coincide (it always
touches at command 11); for v2 the touch index varies 4–11, so the window is
**defined by the outcome being explained** and is 4–11 commands long depending on
the run. Comparing v2's 4-command window with the collector's 11-command window as
"travel during the ramp" is not like-for-like.

The following sentence — *"the runs at the fast end of that spread drive the open
hand into the apple early, and the runs at the slow end let the hand shut first"* —
is then inverted for accumulated travel: the two **smallest** v2 numbers (49115 at
0.40 cm, 49114 at 0.71 cm) are both ejections, small precisely because contact came
early. It is also wrong for the vertical component in rate terms: the ejecting
closes commanded *less* descent than the holding ones (mean dz −0.31…−0.22 vs
−0.41…−0.32).

**The lateral half does hold, completely.** Lateral drift per command up to first
touch: ejections 0.078–0.204 cm/command (49112, 49114, 49103, 49115) vs holds
0.004–0.074 cm/command, **no overlap** on n = 16. Relabel the window, give the
per-command lateral rate, and restate the gloss in lateral terms only.

## Non-blocking findings

These were raised against `dc44f42`. **R1 (`b673f3d`) applied items 1–8 and 10**;
they are kept here as the record of what was found and what changed. Item 9 is
still outstanding, and item 7's wording needs the B3 correction (R1 adopted it
with the incorrect "nearly flat in z" premise).

1. **The MC task card contradicted the frozen scope — fixed in R1.**
   At `dc44f42`, `.mc/tasks/todo/TASK-051-...md` "Scope and resources" said *"48
   attempts (16 each …; primary fresh range, then the secondary 45100-45107
   cohort)"*, while the frozen protocol, manifest and `make_object_plan` all say
   **72 attempts, 24 each, over three cohorts** (45200–45207, 45100–45107,
   45000–45007). The card was amended in the working tree while this review was in
   progress and now matches (72 attempts, 24 each, three cohorts, budgets, and the
   tuning range recorded). Recorded here only so the change is traceable; **commit
   it with the B1–B4 corrections** so the frozen revision is self-consistent.

2. **"Worst case" in the budget section is an assumption, not a worst case.**
   `48 × ~20 s + 24 × 1,200 s ≈ 29,760 s` is arithmetically right, but 20 s per
   `demo_replay`/`scripted_oracle` attempt is the *expected* cost (the smoke
   measured 12.6 s and 12.4 s); the cap on those attempts is also 1,200 s, so the
   true worst case is 86,400 s. The margin is nevertheless large: for the **eight
   primary** ceiling attempts to be squeezed by the global cap, the 48 non-gating
   attempts would have to average **> 475 s each**, i.e. ~38× the measured cost.
   Recommend calling it the planning case and stating that margin. Related
   structural note, worth one line in the protocol: because the order is
   mode-major, all 48 non-gating attempts run *before* any ceiling attempt, so any
   overrun in the non-gating arms is absorbed by the gating arm. The mitigation
   that matters — and which the plan does get right — is that within
   `privileged_object` the **primary seeds run first** (verified: attempts 49–56
   of 72 are 45200…45207), so budget pressure reaches the secondary cohorts before
   the gate.

3. **The descent-only half of the change is not supported by the paired
   experiment, and the documents do not say so.** The diagnosis's own evidence for
   *"never rise"* is the `zonly` row — frozen lateral, CEM free in ±0.5 z — which
   grasped **2/2** with 0.56–0.60 cm of apple motion, i.e. indistinguishable from
   `cage`'s 70/72 at n = 2. Finding 4 states the honest version ("What matters is
   removing the *lateral* freedom, not the vertical one"), but the protocol's
   headline framing, the module docstring and the manifest's `changed_from_v2` all
   present the descent-only bound as load-bearing. If the gate passes, credit
   cannot be assigned between the two sub-changes. Recommend one sentence in the
   protocol: *the lateral/rotational pin is the change the paired experiment
   supports; the descent-only restriction is a conservative addition with n = 2
   evidence and is not separately attributable.* (Keep the restriction — it is
   harmless and it is what was tuned.)

4. **"A frozen palm ejects it by 17 cm" is n = 1.** The `hold` row is 1 run on 1
   seed (16.85 cm) and `narrow` is 2 runs on 1 seed (same 16.85 cm). That number
   appears as a bare fact in `object_ceiling_v3.py`'s module docstring, in the
   commit message and in the protocol. The *general* claim it supports — that the
   descent bound must be large — is well supported (`cage_g0` 0/24 over 8 seeds at
   bound 0.12, `cage_g25` 5/24 over 8 seeds at 0.25, against `cage` 70/72 at 0.5).
   Attribute the 17 cm figure to a single replay.

5. **"On the same entry states" is true for only 16 of `cage`'s 24 seeds — but the
   effect survives the matched restriction.** `cage` ran 72 replays on 24 seeds,
   the `v2` control 53 replays on 19 seeds (52/18 in JSON plus one log-only line
   for 49117); the overlap is **16 seeds**, so 24 of the 72 `cage` runs
   (49124–49131) have no `v2` counterpart. The Limits section does disclose this
   ("a screen, not a matched trial"), but the bolded conclusion — *"`cage` grasped
   on 70 of 72 … against v2's 49/53 **on the same entry states**"* — reads as
   matched. Recommend dropping "on the same entry states" or quoting the matched
   pair. **I recomputed the matched comparison: on the 16 common seeds, `cage`
   47/48 (97.9%) vs `v2` 46/50 (92.0%)** — essentially the same separation as the
   headline 70/72 (97.2%) vs 49/53 (92.5%). So this is a wording fix, not a
   weakening of the evidence. (A second verification pass reported the matched
   figure as 44/48 for v2; my recount from the raw `scan-*` JSON does not reproduce
   that, and I am reporting my own aggregation.)

6. **The close phase remains 45 commands while the synergy finishes in 11, and the
   CEM can keep commanding descent for the remaining ~34.** That is v2's behaviour
   and is deliberately unchanged, but it is the mechanism behind the one observed
   v3 tuning failure (the measured joint-velocity guard at 5 rad/s, 1 of 12 `cage`
   replays). The protocol does declare this risk. Quantitatively it is the tightest
   part of the gate: a per-attempt closure failure rate of ~2.8% (2/72) implies
   roughly a 98% chance of clearing the ≥ 7/8 grasp threshold, and the gate
   tolerates **exactly one** grasp failure in eight. Worth stating explicitly so a
   6/8 or 7/8 outcome is not read as a surprise.

7. **v3 forbids rising but never *requires* sinking — the data says this is fine.**
   In the close, candidate 0 is always the all-zero hold and ties break to the
   lowest index, so nothing in the design guarantees descent; the diagnosis's own
   claim that a frozen palm ejects the apple would then be a live failure mode. I
   checked, and it is not a concern: per B3, the v2 CEM already selects a mean
   applied dz of −0.22…−0.41 on every one of the sixteen closes under bounds that
   *permitted* rising, and the paired `cage` experiment grasped 70/72 under exactly
   v3's `[-0.5, 0]` bound. Worth one line in the risks list for completeness, but
   the empirical answer is already in hand. (This item also shows why B3 matters:
   the diagnosis's "flat in z" wording, if taken literally, would predict the
   opposite.)

8. **Pre-existing, order-dependent test failure (not introduced by this branch).**
   `tests/test_apple_evaluation.py::test_demo_replay_is_open_loop_non_learned_and_exhausts`
   **fails deterministically when that module is run alone** (`1 failed, 71
   passed`), and passes in the full suite. Traced to
   `project_open_loop` → lazy `from embodied_jepa.object_ceiling import
   GUARD_REFUSALS` → `privileged_rollout.py:50 class
   _BlindSimulation(MuJoCoSimulation)` raising `TypeError: function() argument
   'code' must be code, not str` because the test fixture has monkeypatched
   `MuJoCoSimulation`. The lazy import, `project_open_loop` and the failing test
   are all **identical to `origin/main`** (the branch's hunks in
   `scripts/evaluate_apple.py` are at lines 83, 450–630, 1568–1895, 2111 and 2697 —
   none near `project_open_loop`), so this is inherited, not caused here. It does
   not affect the run. Worth a follow-up task: pre-import the module in the fixture
   so the suite is not order-dependent.

9. **Further diagnosis-wording items, from a second independent verification pass
   over the same scratch data.** I did not re-derive each of these myself; they are
   recorded so the author can check them, and none affects the gate. (a) *"all
   three digits load together at command 10–11 … 7.5/6.8/6.8 N on the collector"* —
   those forces are at close command **14**; at the collector's first contact
   (command 11) only index and middle load, 2.6–3.2 N, and the thumb arrives at
   command 13. The 21.0/9.7/11.8 N figure for the ceiling on 49100 **is** exact at
   command 10 (I confirmed that one). (b) "ejected" / "held" in the Finding 2 table
   is an undocumented ≥ 1.5 cm apple-xy threshold from `correlate.py:31`; define
   it, and note that 49103 (2.45 cm, listed as ejected) did latch the scorer's
   grasp stage. (c) *"32 tuning resets from the same distribution shaped v3"* and
   *"24 for the paired closure experiment"* — **27 distinct seeds were actually
   stepped** (49100–49117, 49120, 49124–49131); 49118, 49119 and 49121–49123 were
   never run. The manifest's 32-entry `tuning_seeds_not_evaluated` over-declares,
   which is the safe direction. (d) the Method section's *"the requested and
   applied 14-D action"* — `probe.py` logs `decision.action`, the post-projection
   committed action, so the pre-projection CEM request is never recorded. (e) the
   `hold` mechanism sentence ("the fingers curl around the apple above its equator
   and squeeze it out sideways") has no logged pose or contact data behind it —
   `fork.py` records only outcome scalars — so mark it a conjecture.

10. **`planning_dynamics` stays `privileged_mujoco_rollout_object_v1` for v3** (the
   twin is unchanged and v3 does not override it), exactly as for v2. Note it in
   the results so a reader does not misread the v3 records as v1 dynamics. Also,
   `plan["object_ceiling_phases"]` is still v1's `PHASES` — correct, the phase list
   is unchanged — but the v3 records are only distinguishable by
   `object_ceiling_version` / `ceiling_version` / `result_label`.

## What was verified, and how

### 1. Correctness of the v3 change

`ObjectCeilingV3Controller._bounds()` (`src/embodied_jepa/object_ceiling_v3.py:110`)
delegates to v2 and, **only** when `self.phase.name == "close"`, pins
`lower[6:8] = upper[6:8] = 0`, `lower[9:12] = upper[9:12] = 0` and sets
`(lower[8], upper[8]) = (-close_descent_bound, 0)`. Checked against the chain it
overrides:

- **No aliasing.** `ObjectCeilingController._bounds` (`object_ceiling.py:464`)
  allocates two fresh `np.zeros(14, np.float32)` on every call, so the in-place
  mutation cannot corrupt config or shared state. v2's `_bounds`
  (`object_ceiling_v2.py:245`) returns `super()._bounds()` for every phase but
  `release`, and `lower, lower.copy()` for `release` — v3 returns that untouched.
- **Never empty or zero-width where it matters.** `close_descent_bound` is
  validated in `(0, arm_bound]`, so `upper[8] = 0 > lower[8]` always; the other
  free-dim collapses are exact-point bounds, which the CEM already handles (v2's
  `release` collapses *all fourteen* dims this way). `hold = np.clip(zeros, lower,
  upper)` yields a valid all-zero-arm / grasp = +1 candidate, and
  `mean = np.clip(self.warm, …)` is clipped to the new bounds. `self.warm` is
  additionally reset to `None` on every phase entry (`object_ceiling.py:345`), so
  no stale lateral command survives into the close.
- **No new infeasibility risk.** `G1Embodiment.project_candidates`
  (`embodiment.py:344-399`) backtracks each 6-D arm delta through
  `1.0 … 0.015625, 0.0`; the terminal factor `0.0` is exactly the zero arm delta,
  which is what v3's pinned dims already are. A v3 candidate is therefore
  infeasible only in cases where every v2 candidate would also be infeasible. The
  `ContractError("no feasible object-ceiling candidate sequence")` path is not made
  more reachable.
- **Candidate diversity is not collapsed.** Simulated the close-phase sampling with
  the frozen `candidates = 24`, `horizon = 6`, `proposal_std = 0.3 → 0.15`: 22 and
  23 of 24 candidates remain distinct in the free z dimension (only the first
  executed step is half-clipped to 0). The search is a genuine 1-D line search, not
  a degenerate one.
- **Parity checks stay non-vacuous.** `step()` is unchanged, so a full CEM search
  (projection + twin rollouts + `previous_search_first_step_*` diagnostics) runs on
  every close command. The committed real-physics test asserts exactly 5 checks
  over 6 commands, and the recorded TRAIN smoke reports 282/282 robot-state and
  282/282 full-state checks over 283 commands — comfortably above the gate's
  `>= executed commands - 1`.
- **Emitted actions stay inside the plan's declared controller bounds**
  (`(0,)*6 + (∓0.5,)*6 + (-1, ∓1)`): the close emits `0` laterally/rotationally,
  `[-0.5, 0]` in z, `-1` / `+1` for the grasps.
- Scale sanity from `configs/g1_sim_action.json`: `translation_per_step_m 0.015`,
  `control_dt_s 0.05`, `joint_speed_limit_rad_s 2.0`,
  `measured_joint_velocity_stop_rad_s 5.0`. These reproduce the diagnosis's
  arithmetic: `2.0 × 0.05 = 0.1 rad/command`, `0.1 / 0.55 = 0.1818 = 2/11` →
  **eleven commands** open-to-closed; a 1 cm lateral change against a 6.4 cm height
  error moves the 3-D distance term by `√(6.4² + 1²) − 6.4 = 0.078 cm`; normalized
  lateral 0.219–0.313 → 0.33–0.47 cm/command at 1.5 cm per unit. All correct.
- `close_descent_bound = 0.5 = arm_bound` in both the manifest and the plan the
  frozen command generates — i.e. the shipped default is the `cage` design that was
  actually tuned, and there is **no CLI flag** that could change it during the run.

### 2. v1 / v2 are untouched

Verified by direct comparison against `origin/main`, not by the branch's own
(self-referential) assertions:

- `git diff origin/main..HEAD --name-only` touches 8 files; **neither
  `src/embodied_jepa/object_ceiling.py` nor `object_ceiling_v2.py` is among them.**
  v3 is a pure subclass.
- Loaded `origin/main`'s `scripts/evaluate_apple.py` and this branch's side by side
  and compared serialized output: **v1 plan identical: True; v2 plan identical:
  True.**
- `object_ceiling_v2_gate` compared over **50 randomized synthetic record sets**
  (varying scores, provenance, terminations, parity counts): **identical in every
  case**, both with the default `version=2` and with `version=2` passed explicitly.
  `gate_summary` for a v1 plan and for a v2 plan is identical, and the v2 key is
  still `object_ceiling_v2_gate`. The version parameterisation is therefore inert
  for version 2, as claimed.
- The gate body itself is unchanged in substance: thresholds
  (`6 if n == 8 else ceil(0.75n)`, `7 if n == 8 else ceil(0.875n)`), outcome
  precedence, `conclusive = provenance_valid and exact and complete`, and the
  whole-run `provenance_valid` semantics are v2's. The only differences are the
  `v{version}` string interpolations.
- `.mc`, `benchmarks/manifests/*` for v1/v2 and `docs/experiments/*v2*` are not in
  the diff; recorded evidence cannot change.

### 3. Leakage and cohort hygiene

- **45200–45207 have never been stepped.** Enumerating every recorded seed across
  every `outputs/` tree on this filesystem (main checkout and all sibling
  worktrees) yields `0–4, 7, 2000–2005, 3000–3003, 3072, 5000–5011, 10000–10004,
  19000–19001, 20000–20049, 42000, 43000–43003, 45000–45007, 45100–45107,
  48900–48931, 49100–49116, 49120, 49124–49131` — **no 452xx**. No
  `outputs/*/attempts/452xx-*` directory exists. The only 45200-range text in
  `outputs/` is the inert `gate_rule` string and the archived evaluator source
  inside the 42000 smoke.
- Every in-repo occurrence of 45200–45207 is a declaration (evaluator constants,
  manifest `resets`/`primary_seeds`, protocol tables, plan-shape tests). The one
  test that names 45204 (`test_object_v3_worker_builds_the_v3_ceiling`) runs
  against a fully stubbed `Robot` whose `reset()` is a no-op and a stubbed
  controller that returns `action=None` immediately — no MuJoCo, no command.
- **49100–49131 is used nowhere else** and is disjoint from 42000–42031,
  43000–43004, 44000–44019, 45000–45007, 45100–45107, 48000–48199, 48900–48931 and
  TASK-049's own 49000–49015. All declared ranges in the repository are pairwise
  disjoint as integer sets.
- The 42000 smoke does **not** use `wide_reset(42000)`; it patches `wide_reset` to
  the TRAIN collector's reset rule, so it is a TRAIN episode's own reset.
- Reproduced `wide_reset` for all 103 seeds across all seven cohorts: **no exact
  draw collisions**, and the manifest's 24 frozen `resets` records reproduce
  bit-for-bit. The protocol's offset table (`45200: −1.70, +2.91 … 45207: +1.64,
  −2.80`) matches the generator to the printed precision, and is computable without
  any simulation.
- **Soft finding (disclosed here, not a blocker):** because the tuning range is
  drawn from the same distribution and is 4× larger, several gate draws sit close
  to already-tuned draws. Nearest tuning neighbour by apple xy: 45206 ↔ 49129 at
  **1.8 mm** (plate 3.3 cm away); minimum joint L∞ over the 4-D reset vector is
  0.80 cm (45204 ↔ 49115), tighter than gate-vs-45000 (1.09 cm), gate-vs-45100
  (1.09 cm) and the gate cohort's own internal separation (1.09 cm). Freshness is
  not broken — the resets are distinct and the protocol already frames a pass as
  *"adequacy on new draws from the distribution the tuning resets spanned"* — but
  seed-integer disjointness is doing less work than it appears, and per-seed
  results on 45204/45206 are near design-set neighbours.
- **Test gap:** the only draw-level disjointness assertion
  (`tests/test_apple_evaluation.py`, primary vs secondary) is an *exact-equality*
  set intersection on `object_xy` only. It never compares the gate cohort to the
  tuning range, never checks `plate_xy`, and has no minimum-distance threshold, so
  it would pass even at 0.01 mm separation. Nothing about this run depends on it,
  but do not cite it as evidence of independence.

### 4. Preregistration quality

- **The frozen command is exactly runnable and reproduces the manifest.** Executed
  `make_plan` with precisely the protocol's arguments and compared to
  `benchmarks/manifests/apple-wide-grasp-closure-v3.json`: `gate_rule`,
  `privileged_rule`, `label`, `primary_seeds`, `secondary_seeds` and the whole
  `object_ceiling` config match (the single difference is `palm_offset` as a JSON
  list vs a Python tuple). 72 attempts; mode-major
  `demo_replay → scripted_oracle → privileged_object`; seeds
  45200–45207 → 45100–45107 → 45000–45007 within each mode; first attempt
  `45200-demo_replay`, last `45007-privileged_object`. `--max-seconds 32400` is
  exactly the v3 wall limit (inclusive). `--output` must not exist
  (`FileExistsError`), so the run cannot overwrite prior evidence.
- **The gate is fixed and not reinterpretable.** `gate_summary` emits
  `object_ceiling_v3_gate` for `object_ceiling_version == 3`; thresholds are 6 and
  7 of 8, all six outcomes are enumerated in the manifest and the protocol's
  readings table, and the outcome precedence in code matches that table
  line-for-line. Uncounted attempts are defined by `FAILED_TERMINATIONS =
  ("runtime_error", "deadline_miss", "attempt_timeout")`, so a `phase_stall`,
  `step_limit` or `guard_refused` attempt is **counted with zero stages** — which
  is what the protocol says about the declared guard-refusal risk.
  `provenance_valid` is computed over all records, primary and secondary, and the
  protocol states that.
- Budget arithmetic checks out (see non-blocking item 2). The declared per-command
  deadline (10 s) against the measured 0.91 s/command is an ~11× margin.
- Frozen inputs (dataset manifest, checkpoint, TRAIN-artifact hashes) are the same
  strings as TASK-045/046/047/049.
- `mc validate` → **All checks passed**.

### 5. Evidence discipline

- The diagnosis is headed **"EXPLORATORY. This document is a diagnosis, not gate
  evidence."** and its Limits section volunteers the four things that matter: the
  screen-not-matched-trial caveat, the scratch-code provenance, the n = 16
  separation with wide uncertainty, and that "grasped" is the scorer's grasp stage
  within a 200-command budget, not full success.
- Internal arithmetic is consistent: the `runs` column sums to **289**, matching
  "289 replays"; 17 table rows = 16 closure designs plus the `v2` control.
- The v2 reference numbers in the manifest (`45100–45107`: 5 and 5; `45000–45007`:
  7 and 7, v1 5 and 3) match
  `docs/experiments/apple_wide_object_ceiling_results_v2.md` exactly, as does "all
  four failures (three primary, one secondary)".
- TASK-051 is precisely the preregistered next step v2 recorded for its
  `ceiling_inadequate_task_feasible` outcome ("diagnose the close-phase ejection
  and redesign the grasp closure under a new preregistration; do not pair this
  ceiling with a learned model yet") — the follow-up honours its own prior
  registration.
- NON-LEARNED labelling is consistent everywhere (module label, plan
  `result_label`, gate label, both documents, the commit message), and both
  documents restate that learned Apple→Plate remains at zero successes.

**What reproduces exactly from the raw scratch data** (my own re-aggregation of
`forensics-a/*.json`, `scan-*/*.json` and `fork-smoke*/*.json`):

- Finding 2's whole table, cell for cell: first-touch index ≤ 8 → {49103: 8,
  49115: 4, 49112: 6, 49114: 5} with max apple xy {2.45, 3.12, 7.79, 9.12} cm;
  first touch ≥ 10 → the other 12 seeds, 0.19–0.78 cm; scripted → touch 11 on all
  16, 0.71–0.74 cm. The separation on first-touch index is complete, with no seed
  touching at 9.
- Finding 3's command statistics: v2 mean absolute lateral command over the eleven
  ramp commands **0.219–0.313**; collector **0.029–0.047**; collector dz ≡ −0.400.
- The eleven-command derivation, from `configs/g1_sim_action.json`:
  `2.0 rad/s × 0.05 s = 0.1 rad`, `0.1 / 0.55 = 0.1818 = 2/11`, so `2 / 0.1818 = 11`
  commands from fully open to fully closed.
- The 6.4 cm / 0.078 cm geometry: `√(6.4² + 1²) − 6.4 = 0.0776 cm`.
- All 17 Finding 4 rows (runs / seeds / grasped / motion ranges), including
  `cage` 72/24/70, `v2` 53/19/49, `zonly` 2/1/2, `cage_g0` 24/8/0, `cage_g25`
  24/8/5, and the `runs` column summing to 289.
- All 24 reset offsets for 45200–45207, regenerated from `wide_reset` in pure
  NumPy with no simulation.

**Over-claims found** are B1–B4 (blocking) and non-blocking items 3, 4, 5 and 9.
Four numbers are ones the code or the described method could **not** have produced
as stated: the 1-of-12 guard rate (B1), the 16/16 "success" (B2), the flat-in-z
claim (B3), and the "during the ramp" travel window (B4).

### 6. Honesty about the physics

Confirmed from the diff: the branch touches exactly
`.mc/tasks/todo/TASK-051-….md`, `benchmarks/manifests/apple-wide-grasp-closure-v3.json`,
`docs/experiments/apple_grasp_closure_diagnosis.md`,
`docs/experiments/apple_wide_grasp_closure_v3.md`, `scripts/evaluate_apple.py`,
`src/embodied_jepa/object_ceiling_v3.py`, `tests/test_apple_evaluation.py`,
`tests/test_object_ceiling_v3.py`. **No change to `src/embodied_jepa/simulation.py`,
`src/embodied_jepa/embodiment.py`, `configs/`, `assets/` or `third_party/`.** The
claim that no simulator physics or contact parameter changed is true.

### 7. Tests

The new tests were **mutation-tested** rather than read: three in-memory mutations
were applied and the suite re-run.

| Mutation | Result |
|---|---|
| Revert `_bounds` to v2's (i.e. v3 == v2) | **2 failed** — `test_the_close_cages_the_apple_and_only_lets_the_palm_sink` and the real-physics `test_the_live_close_holds_the_palm_laterally_and_only_sinks` |
| Cage lateral/rotational only, keep v2's symmetric ±0.5 z (the `zonly` design) | **1 failed** — the descent-only half is genuinely pinned |
| Remove `ObjectCeilingV3Config.__post_init__` validation | **1 failed** — `test_v3_config_rejects_an_unusable_descent_bound` |

So no v3 behavioural assertion is vacuous. The suite exercises real behaviour:
`test_the_live_close_holds_the_palm_laterally_and_only_sinks` runs **real MuJoCo**
(mujoco 3.13.0, pinned assets present — it ran, it did not skip) and checks the
executed commands, the 2/11 rate-limited synergy increment, that the palm holds
laterally within 2 mm and sinks, and that 5 parity checks over 6 commands are all
exact. `test_every_other_phase_keeps_the_v2_bounds` compares all seven other phases
against a live v2 controller.

Caveat worth recording: the three evaluator-side v3 tests in
`test_apple_evaluation.py` are **wiring** tests — under the "revert `_bounds`"
mutation all of them still passed. They correctly test the plan, gate and worker
dispatch, not controller behaviour, which `test_object_ceiling_v3.py` covers.
Also, the "v1 and v2 plans unchanged" assertions inside that file are
self-referential (they compare the branch to itself); the independent comparison
against `origin/main` in §2 is what actually establishes that claim — same
observation as item 11 of the v2 review.

### 8. Checks run

Run at `dc44f42` and again at `b673f3d`, identical results both times:

```
.venv/bin/python -m ruff check src tests scripts        → All checks passed!
.venv/bin/python -m ruff format --check src tests scripts → 107 files already formatted
.venv/bin/python -m pytest -q -p no:randomly            → 676 passed, 13 skipped
mc validate                                             → All checks passed!
```

R1's only change under `src`, `scripts`, `tests` or `configs` is the
`object_ceiling_v3.py` **module docstring** (9 insertions, 5 deletions, all inside
the docstring): `git diff dc44f42..b673f3d -- src scripts tests configs` touches
that one file. So every code, plan, gate and cohort finding verified at `dc44f42`
carries over to `b673f3d` unchanged, including the v1/v2 byte-identity comparison
(`scripts/evaluate_apple.py` is not in the R1 diff).

The 13 skips are the tolerated ones (graphics opt-in ×3, `LEROBOT_SOURCE` ×3) plus
7 `timm`-missing skips, which are an artifact of this worktree's venv lacking the
`lewm` extra, not a branch property. Per-module isolated runs:
`test_object_ceiling.py` 10 passed, `test_object_ceiling_v2.py` 12 passed,
`test_object_ceiling_v3.py` 7 passed, `test_apple_evaluation.py` **1 failed**
(pre-existing, non-blocking item 8).

## What I could not verify

- **That the reset-envelope check on 45200–45207 was actually executed.** The
  protocol says those resets were "only *reset* (no command executed)" for the
  envelope check and the offset table. No artifact of that check survives on disk,
  so the claim is unfalsifiable from evidence. It is consistent with everything
  observed (a reset-only check writes no `"seed":` record), and the offset table
  demonstrably needed no simulation at all — I regenerated it from `wide_reset`
  alone.
- **Per-seed accounting of the 289 tuning replays.** Surviving scratch results
  cover 49100–49116, 49120 and 49124–49131; 49117–49119 and 49121–49123 have no
  files. The manifest conservatively declares all 32 seeds as tuning (the safe
  direction), but "289 replays on 49100–49131" cannot be reconciled seed by seed
  from the surviving files.
- **Whether the paired experiment's later variants are order-independent.** For
  each seed, `fork.py` runs the variants sequentially against one reused
  simulation, robot and twin. Only `v2` (always first) has a verified
  cross-process reproduction (49100: 0.5197330928 cm from the fork vs
  0.5197330925 cm from the independent full `probe.py` run — agreement to nine
  significant figures). I did not establish that `cage`, run later in the same
  process, is equally reproducible from a cold start.
- Anything about whether a **learned** model can supply the cost terms. This run is
  a NON-LEARNED privileged ceiling; a pass says the design suffices given exact
  dynamics and perfect object state, and nothing more. The protocol says this
  first.

## What clears the block

Three edits, all documentation, against `b673f3d`:

| | Where | Change |
|---|---|---|
| **B2** | `apple_wide_grasp_closure_v3.md:47`, `apple_grasp_closure_diagnosis.md:172` | "the collector reaches 16/16 on the same resets" → "16/16 **grasp stage**, with no ejection"; note the forensics runs stop at `close_start + 85` so full success was never measured there |
| **B3** | `apple_grasp_closure_diagnosis.md:83,89`; `apple_wide_grasp_closure_v3.md:17,272`; `object_ceiling_v3.py` docstring | narrow "the close cost is nearly flat" to the **lateral** command; drop or correct "−0.5 in z and 0 in z reach almost the same palm position" |
| **B4** | `apple_grasp_closure_diagnosis.md:100–103`; `object_ceiling_v3.py:18` | relabel the travel figures as **at first contact** (or restate over the fixed 11 commands), and restate the "fast end / slow end" gloss in lateral-rate terms |

Then re-run `ruff` + `pytest` + `mc validate`. None of these edits touches a code
path, so no behaviour change is expected. Nothing else is required: **no seed,
threshold, cohort, budget, parameter, config value or command changes, the TRAIN
smoke does not need repeating, and no re-tuning is needed.** The v3 controller,
the plan, the gate, the manifest and the frozen command are all correct as they
stand, and B1 and non-blocking items 1–8 and 10 were already applied in R1.

Also worth folding into the same commit: the protocol's "Known risks" still says
*"**32** tuning resets from the same distribution shaped v3"*
(`apple_wide_grasp_closure_v3.md:285`), which now contradicts the tuning
disclosure two sections above that R1 corrected to 27 seeds actually stepped.

Non-blocking item 9 remains outstanding as wording; item 8 is worth a separate
follow-up task since it is inherited from `main`.
