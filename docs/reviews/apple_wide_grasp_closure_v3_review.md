# TASK-051 pre-run review: object-aware ceiling v3 (caged grasp closure)

Fresh-context reviewer subagent, independent of the author, read-only, on branch
`feat/task-051-grasp-closure-v3` at `dc44f42` (two commits on `origin/main`
`6739427`). The gated evaluation was **not** executed and no 45xxx reset was
stepped by this review.

## Verdict: CLEAR TO RUN

No blocking defects. The v3 change is confined to `_bounds()` in the `close`
phase; it cannot produce an empty, degenerate or contract-violating candidate set;
it does not reduce candidate feasibility relative to v2; and the CEM still runs on
every close command, so the runtime parity checks stay non-vacuous. v1 and v2 plan
and gate output were verified **byte-identical in content** to `origin/main` by
direct comparison, not by the branch's own tests. The gate cohort 45200–45207 has
never been stepped anywhere on this filesystem. Simulator physics and contact
parameters are untouched. The preregistration is unambiguous and the frozen
command reproduces the manifest exactly.

Ten non-blocking items below. Item 1 is a **factual error in a committed
document and in the shipped module docstring** and must be corrected in the
record; it does not block the run, because no gate number, threshold, budget or
design decision depends on it. Items 2–4 are worth applying as wording/record
fixes before the run. None of them changes a seed, threshold, budget, parameter
or command.

## Blocking findings

None.

## Non-blocking findings

1. **A mislabelled measurement window in the diagnosis (fix the record; the
   conclusion survives).** `apple_grasp_closure_diagnosis.md` Finding 3 says:
   *"The consequence is the palm travel **during the ramp**: the collector sinks
   0.97–0.99 cm and drifts 0.20–0.22 cm laterally on all sixteen resets, while v2
   sinks 0.40–1.85 cm and drifts 0.05–1.22 cm."* I recomputed all four ranges from
   `outputs/task051-scratch/forensics-a/`. They are **not** travel over the ramp —
   they are travel **at the first hand–apple touch command**, which is what the
   scratch harness actually computes (`correlate.py:39-41`,
   `sink_at_touch_cm` / `drift_at_touch_cm`):

   | statistic | v2 ceiling | scripted |
   |---|---|---|
   | sink / drift over a fixed 11-command ramp (`rows[0]→rows[11]`) | 0.94–2.31 / 0.05–1.13 cm | 0.97–0.99 / 0.20–0.22 cm |
   | **sink / drift at first touch** | **0.40–1.85 / 0.05–1.22 cm** | **0.97–0.99 / 0.20–0.22 cm** |

   The doc's figures are the second row. For the collector the two coincide (it
   always touches at command 11); for v2 the touch index varies 4–11, so the
   window is **defined by the outcome being explained** and is 4–11 commands long
   depending on the run. Comparing v2's 4-command window with the collector's
   11-command window as "the palm travel during the ramp" is not like-for-like,
   and the following sentence — *"the runs at the fast end of that spread drive
   the open hand into the apple early"* — is inverted for accumulated travel: the
   two smallest v2 numbers (49115 at 0.40 cm, 49114 at 0.71 cm) are both
   **ejections**, small precisely because contact came early.

   **The conclusion is nevertheless correct**, on the rate rather than the
   accumulated distance. Lateral drift *per command* up to first touch separates
   the sample completely — ejections 0.078–0.204 cm/command (49112, 49114, 49103,
   49115), holds 0.004–0.074 cm/command, **no overlap** — and the fixed-window
   evidence that actually carries Finding 3 (mean absolute lateral command over
   the eleven ramp commands: v2 **0.219–0.313** vs collector **0.029–0.047**) I
   reproduced **exactly**. Recommend: relabel the sentence as travel *at first
   contact*, give the per-command rate alongside it, and drop the "fast
   end / slow end" gloss or restate it in rate terms. The same two ranges are
   quoted in `src/embodied_jepa/object_ceiling_v3.py`'s module docstring ("the
   palm moves 0.05-1.22 cm laterally and 0.40-1.85 cm vertically while the fingers
   are still opening") and in the commit message; fix the docstring, and note the
   commit-message wording in the results document.

2. **The MC task card contradicts the frozen scope.**
   `.mc/tasks/todo/TASK-051-...md` "Scope and resources" still says *"48 attempts
   (16 each …; primary fresh range, then the secondary 45100-45107 cohort)"*. The
   frozen protocol, manifest and `make_object_plan` all say **72 attempts, 24 each,
   over three cohorts** (45200–45207, 45100–45107, 45000–45007). Update the card
   before the run so the frozen record is self-consistent. (This is the same class
   of item as #6 in the v2 review.)

3. **"Worst case" in the budget section is an assumption, not a worst case.**
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

4. **The descent-only half of the change is not supported by the paired
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

5. **"A frozen palm ejects it by 17 cm" is n = 1.** The `hold` row is 1 run on 1
   seed (16.85 cm) and `narrow` is 2 runs on 1 seed (same 16.85 cm). That number
   appears as a bare fact in `object_ceiling_v3.py`'s module docstring, in the
   commit message and in the protocol. The *general* claim it supports — that the
   descent bound must be large — is well supported (`cage_g0` 0/24 over 8 seeds at
   bound 0.12, `cage_g25` 5/24 over 8 seeds at 0.25, against `cage` 70/72 at 0.5).
   Attribute the 17 cm figure to a single replay.

6. **The `cage` vs `v2` headline is across different seed sets.** `cage` ran 72
   replays on 24 seeds, the `v2` control 53 replays on 19 seeds. The Limits
   section discloses this ("a screen, not a matched trial"), but the bolded
   conclusion — *"`cage` grasped on 70 of 72 … against v2's 49/53 on the same entry
   states"* — reads as matched. Either restrict the headline comparison to the
   common seeds or drop "on the same entry states" from that sentence.

7. **The close phase remains 45 commands while the synergy finishes in 11, and the
   CEM can keep commanding descent for the remaining ~34.** That is v2's behaviour
   and is deliberately unchanged, but it is the mechanism behind the one observed
   v3 tuning failure (the measured joint-velocity guard at 5 rad/s, 1 of 12 `cage`
   replays). The protocol does declare this risk. Quantitatively it is the tightest
   part of the gate: a per-attempt closure failure rate of ~2.8% (2/72) implies
   roughly a 98% chance of clearing the ≥ 7/8 grasp threshold, and the gate
   tolerates **exactly one** grasp failure in eight. Worth stating explicitly so a
   6/8 or 7/8 outcome is not read as a surprise.

8. **The descent during the close is chosen by a cost the diagnosis itself shows is
   nearly flat in z while the palm is blocked.** v3 forbids rising but does not
   *require* sinking; candidate 0 is always the all-zero hold, and ties break to
   the lowest index. Once the fingers curl off the table the height term stops
   being flat and descent strictly reduces cost, and the paired experiment (70/72)
   is direct empirical evidence that the CEM does choose enough descent under this
   bound — so this is not a defect, but it is a residual mechanism the risks list
   does not name.

9. **Pre-existing, order-dependent test failure (not introduced by this branch).**
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
- Over-claims found are items 3, 4 and 5 above. No number was found that the code
  or the described method could not have produced.

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

```
.venv/bin/python -m ruff check src tests scripts        → All checks passed!
.venv/bin/python -m ruff format --check src tests scripts → 107 files already formatted
.venv/bin/python -m pytest -q -p no:randomly            → 676 passed, 13 skipped (12.11s)
mc validate                                             → All checks passed!
```

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
- Anything about whether a **learned** model can supply the cost terms. This run is
  a NON-LEARNED privileged ceiling; a pass says the design suffices given exact
  dynamics and perfect object state, and nothing more. The protocol says this
  first.
