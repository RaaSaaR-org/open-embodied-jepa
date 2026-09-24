# Apple→Plate policy v1 results (TASK-056) — FAILED on the development stop rule

Protocol: [apple_policy_v1.md](apple_policy_v1.md). Manifest:
`benchmarks/manifests/apple-policy-v1.json`, merged as `3006b3b` (PR #33).

> **TASK-056 FAILED.** Four arms were trained and run on the 16 development resets. **Every arm
> scored 0 full successes.** Both controls reached the scorer's `grasp` stage on 0/16 and are
> `dead_on_development`. **Cohort C (seeds 45300–45339) was never opened and will not be opened
> for this task** — the user ruled it stays unconsumed, permanently. Learned Apple→Plate remains
> at **0 successes**, as it has been for every generation of this line.
>
> **No manipulation result is claimed anywhere in this document.** The pre-flight measured
> perception on recorded validation frames and passed; that is a perception measurement, not a
> control result, and §1 says so at length. Everything after §14 is a development-cohort number,
> which the protocol states "never gates and is never reported as a result."

**Why cohort C was not opened, recorded before anyone asks whether it should have been.** Under
*either* reading of the protocol's ambiguous stop rule (§15), cohort C could not have produced a
pass. If the controls are subject to the stop rule they are dead, G2 and G3 lose their reference
terms, and §5.2's missing-values rule converts an unevaluable gate to failed. If they are exempt,
A2 would have needed 17/40 successes on fresh resets after 0/16 on development with every attempt
timing out at the step cap. Spending 40 irreplaceable frozen resets to document a determined
outcome buys nothing; preserving them for a line with a real chance is worth more. **The decision
did not require resolving the ambiguity, and the ambiguity was deliberately not resolved** — see
§15, which is a finding in its own right.

---

## 1. Pre-flight — all three gates PASS

Artifact `outputs/apple-policy-v1/preflight.json` (git-ignored, as all run artifacts are).
Runner `scripts/measure_policy_preflight.py` at the merged revision; frozen feature source
`checkpoints/task054-wm-v4/leworldmodel_baseline.pt` (E0), loaded read-only. Val split only, 15
surviving val roots, **7 734 rows scored**, 985 rows dropped (apple off the table, where the
regression targets are undefined). 9.5 s on MPS.

| Gate | Value | Threshold | P0a shuffled-frame control | Ratio | ≤ 0.7 | Result |
|---|---|---|---|---|---|---|
| **P1** close palm–apple | **0.491 cm** | ≤ 1.5 cm | 1.314 cm | **0.374** | ✓ | **PASS** |
| **P2** approach palm–apple | **0.456 cm** | ≤ 1.5 cm | 0.669 cm | **0.682** | ✓ | **PASS** |
| **P3** orient apple-position | **0.873 cm** | ≤ 2.0 cm | 2.269 cm | **0.385** | ✓ | **PASS** |

P0b, the analytic blind prior (always emit the reset centre): **2.394 cm**.

### Per-phase, directly encoded, horizon 0

| phase | rows | palm–apple (cm) | apple-position (cm) |
|---|---|---|---|
| orient | 1 950 | 0.490 | 0.873 |
| descend | 1 200 | 0.404 | 0.754 |
| **close** | 675 | **0.491** | 0.794 |
| lift | 2 181 | 0.356 | 0.936 |
| transfer | 840 | 0.246 | 0.982 |
| release_high | 709 | 1.579 | 1.810 |
| lower_open | 179 | 4.084 | 4.071 |

### What this establishes

**The protocol's premise was a hypothesis and it is now supported.** Close-phase h≈0 perception
is **0.491 cm**, against the 2.3536 cm that the committed record's only directly-encoded
measurement (the *moving* cohort) reported. 0.491 cm sits at the **bottom of the measured
grasp/hold band** (holds 0.49–1.35 cm, `apple_wide_grasp_closure_results_v3.md`).

**Declared risk R2 is not borne out in the form it was written.** R2 was that the encoder might
only localize the apple once the palm is already beside it, making the close-phase number a
shortcut. Approach phases read **0.404–0.490 cm**, so the accuracy is not confined to the close.

**This result means what it does because the threshold was written before the number existed**,
and because the control could have caught the alternative. It is not evidence about dynamics —
see §4 — and it is not evidence that any policy works.

---

## 2. The P2 ratio, recorded as an observation

**P2's ratio is 0.682 against a 0.7 threshold — a margin of 0.018.** It passes. **No threshold
was adjusted in response to it**, and none may be: the protocol forbids it and this document
records the value rather than the absence of one.

What it means: on approach frames a **foreign** image still yields 0.669 cm, so most of the
**relative** palm–apple accuracy is available without looking at the apple at all. In hindsight
this is expected — the collector servos the palm toward a target derived from the apple, so the
palm's own pose encodes the apple's position, and `state_fusion: true` puts proprioception in the
latent. **It is the R2 shortcut signature, partially present, on the relative readout.**

Two consequences, both recorded in advance of the arms being built:

- **At the first draft's P2 of 2.5 cm with no control this would have been completely
  invisible.** The no-vision control was added at an independent reviewer's insistence; without
  it the protocol would have passed a gate while being unable to see the thing the gate existed
  to detect.
- **A0 is therefore the most informative arm in the protocol, not a formality.** If relative
  palm–apple accuracy is largely recoverable from proprioception, a proprioception-only policy
  may be strong, and **G2 (best learned arm − A0) becomes the gate that decides whether anything
  visual is happening at all.** A0 is to be built and trained with the same care as A2. **If A0
  turns out strong, that is a finding, not an inconvenience**, and it will be reported as one.

## 3. P3 is the cleanest measurement in the run, and it is stated separately

P3's control lands at **2.269 cm**, essentially on the analytic blind prior of 2.394 cm. That is
the point: **absolute apple position genuinely cannot be inferred from the palm's own pose**, so
P3 is the one measurement here with **no shortcut available to it**. The model reads **0.873 cm**
against that control, a ratio of 0.385.

This is the result that most supports **A4** (learned perception driving the scripted phase
machine), whose entire front end is a single reset-time absolute estimate. It is reported apart
from P1/P2 because its evidential status is different and stronger.

### Flagged in advance: the place phase is where A4 is most likely to fail

`release_high` at **1.579 cm** and `lower_open` at **4.084 cm** are far worse than every other
phase — the apple is at the plate and partly out of the onboard view. **A4's place phase runs
exactly there**, and the v3 ceiling's release predictor, which converted every grasp into a
success under exact dynamics, is the component with **no learned substitute**.

**Recorded now, before any arm runs:** if A4 grasps reliably and then fails at the plate, that is
the failure this table predicted, and this paragraph is the pre-registration of that reading. It
is neither gated nor a reason to change any gate.

## 4. U1–U4 promoted from unverified to established

The protocol listed four quantities as `unverified_pending_P0` because they existed only in a
git-ignored run artifact. The pre-flight re-derived all four by calling the v4 evaluator's own
`window_metrics` on its own window cohort (4 807 windows). **Every value reproduces exactly**;
the committed script is now their provenance.

| h | U1 grasp cohort (cm) | U2 rollout ÷ persistence | U3 shuffled AUROC | U3 real AUROC | U4 all windows (cm) |
|---|---|---|---|---|---|
| 1 | 0.8336 | 0.9524 | 0.99737 | 0.99949 | 0.7101 |
| 4 | 0.9013 | **1.0461** | 0.84996 | 0.99949 | 0.7315 |
| 8 | 0.9834 | 0.8763 | 0.76765 | 0.99949 | 0.7818 |
| 16 | 1.1469 | 0.7878 | 0.70736 | 0.99928 | 0.9131 |

**Kept because it is inconvenient and unimportant, which is exactly how such things get
dropped:** at **h = 4 the rollout is *worse* than persistence** (ratio **1.0461**). The protocol
quoted only h = 1, 8 and 16, where the ratio is below 1, so this was not visible. It changes no
gate — G2a is scored at h = 8 and has already failed in every generation — and it is one more
data point against the abandoned line rather than a new finding. It stays in the record.

**These numbers remain rollout readouts, not perception measurements** (`world_model_v2.py`
L381: `palm = error(predicted, …)`). Their re-derivation establishes that the protocol quoted
them accurately; it does not change what they measure. The directly-encoded numbers are §1's.

## 5. A research-integrity guard was widened during this protocol — deliberately, and here is why

`tests/test_apple_wide_collection.py::test_model_planner_and_evaluation_code_never_import_training_labels`
forbids every module in `src/embodied_jepa/` from importing a label module, with a short
allow-list. **`src/embodied_jepa/cloning.py` was added to that allow-list in stage 2.** A guard
against leakage was widened during an active protocol, by the agent running that protocol, so the
reasoning belongs in the record rather than in a diff a future auditor has to reconstruct.

**What changed.** The allow-list went from `{training_labels.py, readout_labels.py,
world_model_v2.py, world_model_v3.py, world_model_v4.py}` to the same set plus `cloning.py`.

**Why the boundary is where it is.** The rule the repo already uses is *training runners may build
targets from labels; control-loop code may not.* `world_model_v2.py` is on the list because it
builds readout targets. `cloning.py` is on it for the identical reason: it reads
`collector__base_action` as the behaviour-cloning target, under the same explicit
`acknowledge_privileged_training_labels` gate. **This applies the existing rule to a new trainer;
it does not relax the rule.**

**`src/embodied_jepa/policy.py` is excluded by design.** The policy runs *inside the control
loop*, where a label would be a genuine leak — it is the module whose inputs become the robot's
inputs. It imports no label module and must not.

**Which check is load-bearing, stated so nobody later "simplifies" the wrong one.** Two things
guard `policy.py` and they are not equivalent:

- `assert "policy.py" not in allowed` is a **tautology against a literal set**. It can only fail
  when someone edits that set, which is exactly and only its job: it makes a future widening
  fail loudly rather than pass silently.
- **The load-bearing check is the loop beneath it**, which parses every non-allow-listed module
  and asserts it neither mentions nor imports a label module. `policy.py` is covered by that loop
  *precisely because* it is not on the allow-list.

Removing the loop on the grounds that the assertion covers it would delete the real protection and
leave a tautology behind. Both are kept.

## 6. Withdrawal clause for the stage-1 result — pre-committed, before the answer is known

**Committed on 2026-09-24, while the alignment review of the runner was still running and its
answer was unknown to everyone involved.** It is written now precisely because it is cheap to
commit to in advance and expensive to argue about once a passing result is in hand and a day of
work sits behind it.

The pre-flight's numbers depend on a row/position correspondence that nobody has yet verified:
`shuffled_frame_control` deliberately uses **two different row arrays** — `image_rows` for the
image, `rows` for the proprioception — and compares the resulting readouts against
`truth[valid_rows]`. The result in §1 was accepted on its *values*; the correspondence underneath
those values was not checked. That is the same shape of gap as the rollout-versus-perception error
this protocol's §1 was rewritten to fix, which is why it is being checked at all.

**If the alignment check finds a defect in `shuffled_frame_control`, or anywhere else the
pre-flight depends on, then:**

0. **The executing agent STOPS AND REPORTS BEFORE FIXING ANYTHING.** Added 2026-09-24 as a
   refinement that makes this clause stricter, not looser: the original wording went straight to
   "the fix lands", which would have let a repair happen as a reflex inside a merge. A defect
   here would have implications wider than one script, and that response is decided deliberately
   with the coordinator rather than chosen by whoever finds it.
1. **The stage-1 result is WITHDRAWN, not patched.** §1's gate table stops being the result.
2. Once the response is agreed, the fix lands and **the runner re-runs from the same frozen
   checkpoint**. The new numbers stand on their own — **including the possibility that a gate
   which passed now fails**, and with it the possibility that Outcome E fires and this task
   stops.
3. **The 0.491 / 0.456 / 0.873 figures are not carried forward alongside a corrected runner.**
   They are reported as withdrawn, next to what replaced them.
4. **No argument is entertained about whether the defect "would have mattered."** A result whose
   correctness was established by reasoning after the fact is not a measurement. This clause
   exists to remove that conversation, not to have it more carefully.
5. The withdrawal, the defect and the re-run are all recorded here, **with both sets of numbers**,
   so the record shows what we believed and what corrected it.

This is not expected to fire. If it does not, this section stays in the document as a record of
what was committed to while the answer was still open.

**A related exposure is named here and deliberately NOT acted on.** The same gap — that verifying
a number's *value* does not verify *what produced it* — applies in principle to every accepted
numerical claim in this experiment record, and the claims nobody has had reason to re-examine
carry it most, since "nobody re-examined it" is not evidence of anything. Re-auditing the record
from inside a running experiment is how a protocol stops being a protocol, so it is not done here
and no task is opened for it by this protocol. It is recorded so that it is carried rather than
forgotten.

## 7. A discipline adopted mid-protocol, and what it caught immediately

**Every blocking finding in the stage-2 review was in code written while fixing a previous
finding.** Round 1 found three defects in the fresh implementation; round 2 found two in round
1's fixes; round 3 found one in round 2's fix. The findings narrowed, but the location did not
move.

The mechanism is not carelessness about the original code. **A fix is written under the belief
that the surrounding area has just been understood — and that belief is precisely what suppresses
the check.** The specific miss was identical every time: *the fix was verified to do the new
thing, and not verified to still do the old one.*

So, adopted for every fix from here on, and stated before a reviewer sees it:

> **Name the property that existed before the fix and must still hold after it, and say how you
> checked that it does.** Not "the tests pass" — the named property and the check. If you cannot
> name it, that is the signal you do not yet understand what you are changing.

**It found something within minutes of being adopted.** Applying it retroactively to the round-3
`load`-ordering fix, the preserved property is *"an A2 head still refuses an A1 encoder."* The
suite said yes. Checking it directly on the real backend also said yes — but surfaced that the
refusal message said *"weights were not restored exactly"* for a frozen arm where **nothing was
restored**, which would send a debugger to the wrong place.

**And repairing that message broke the thing again, in the same way.** Folding two nested
conditions into one `if saved_digest is not None and …` re-attached the `else` branch to the
combined expression, so a *matching* digest began raising "carries no encoder digest to verify".
A correct A3 checkpoint became unloadable a second time, by a different route, inside a
two-line cosmetic edit. The round-trip test caught it — but only because the previous round had
given the test stand-in a real digest, without which it would have been invisible again.

That is the fourth occurrence of the same shape, and the most instructive, because the edit was
too small to feel like it needed verification. The nesting now carries a comment saying why it
must stay nested. All four load properties were then verified directly on the real backend:
A2→A2 loads, A2→A1 refuses, A3→fresh-E0 loads, A3→frozen refuses.

## 8. The most serious defect found in this task, and its class

**A type coercion could have silently consumed the frozen cohort.**

`scripts/evaluate_policy.py` refused cohort C by testing `set(seeds) & set(COHORT_C)` — **before**
coercing the seeds to `int`. `"45300"` is not equal to `45300`, so the intersection was empty, and
the coercion that followed produced the frozen reset. Verified: the string form returned exactly
the manifest's stored `resets["45300"]`.

**This is categorically worse than every other defect in this protocol.** The others produce a
wrong number or a failed load — visible, recoverable, costing a day. This one consumes cohort C,
which is **unrecoverable**: there is no re-running a cohort whose whole value is never having been
seen. Cohort C is gated behind three separate authorizations precisely because opening it cannot be
undone, and a type coercion could have walked through all three without any of them being asked.

**The class, which is the part worth carrying:**

> **A guard that enumerates what is FORBIDDEN fails open. A guard that enumerates what is
> PERMITTED fails closed.**
>
> And it does not merely fail open: because the report hard-codes the cohort label, an
> unenumerated seed would have been **filed as development data**. It fails open *and*
> mislabels, which is what makes it silent rather than merely permissive.

The original guard was a blacklist of cohort C. It would equally have admitted 45400, 46000, or
any seed nobody thought to forbid — and the report hard-codes `"cohort": "D_development_never_gating"`,
so such a run would have been *filed as development data*. The replacement is a whitelist of
cohort D, and it is right not because it happens to catch the string case but because **refusal is
its default**.

Both are now tested at the type boundary — `int`, `str`, `float`, `np.int64`, `np.int32`, `bool`,
whitespace — including that legitimate non-canonical forms of a *permitted* seed are still
accepted, since a guard that refuses valid input is a different defect rather than extra safety.
Each guard was verified by **injecting its own violation and watching the test fail**: restoring
the blacklist, and restoring membership-before-coercion, each break the test.

**No cohort-C seed has ever been simulated.** The defect was found by independent review before the
runner was used for anything.

## 9. Two demonstrations that the integrity machinery works

**The implementation-hash guard refused to let a source change re-interpret historical
checkpoints.** The clean fix for the mutating-digest defect was a helper in `models/../policy.py`;
applying it changed that file's `implementation_sha256`, and **all four trained checkpoints
immediately became unloadable.** That is the guard doing exactly its job — a later source fix does
not make a historical checkpoint compatible. Found by attempting the refactor, not by reasoning
about it. The minimal non-mutating fix was used instead and the clean refactor is recorded as owed,
belonging with a retrain.

**A test that claimed to discharge a preregistered debt did not, and an injected violation proved
it.** The clipping test exercised a helper, never the control loop. The reviewer changed
`run_attempt` to execute the raw command while still counting the clip: the robot received twice
the configured delta, the report's counter actively lied, and **the full suite stayed green at 782
passed**. That is now twice in this protocol a test has claimed a debt it did not discharge.

> **Standard adopted for every debt-discharging test from here: write the violation, watch the test
> fail, then fix it.** A test that has only been seen to pass is not known to discriminate.

## 10. Process notes

- **A bug was found by smoking the runner on two episodes before the real run.**
  `world_model_v2._write_json` writes through a sibling `.tmp` and does not create directories,
  so the first execution failed on a missing `outputs/apple-policy-v1/`. Cost: one minute. Fixed
  in PR #35.
- A second two-episode smoke then tripped the runner's own ≥ 95 % re-pairing assertion —
  **correctly**, because one surviving root leaves no foreign frame for the control to pair with.
  The smoke was resized to 20 episodes / 4 roots.
- **No threshold was adjusted** in response to anything measured here.
- `models/base.py` and `models/lewm.py` untouched; the implementation-hash CI check passes.
- Nothing under `data/`, `checkpoints/` or `outputs/` was overwritten.
- **Cohort C (45300–45339) has not been simulated**, not even once, and opening it is a separate
  authorization.

## 11. How the stage-3 results will be read — committed before they exist

Same class as the withdrawal clause of §6, and recorded for the same reason: it is cheap to
commit to now and expensive to argue about once results are in hand.

**Two independent readings.** When the arms have run, this author produces a reading, and a
**fresh agent that has not seen it** produces its own from the artifacts and the protocol alone.
The coordinator compares them. The point is not redundancy — it is that the history which makes
this author a good interpreter (why `image_features` and not `encode`, what the A0 contrast
actually measures, why P2's 0.682 control ratio matters) is the same history that makes the
reading hard to un-bias from the inside.

**Therefore the results report separates numbers from reading, and labels both.**

- A **numbers** section that stands alone: gate values, cohort sizes, per-arm counts, control
  ratios, seeds, provenance. **No interpretive language** — including no "as expected", "only",
  "already" or "still", which are readings wearing the clothes of description. It must be
  runnable as the sole input to the second reader.
- A **reading** section, separate and labelled, which states explicitly what in it is inference
  rather than measurement — the way §1 now distinguishes the h=1 rollout inference from the
  measurements around it.

**This is a commitment about presentation, not about conclusions.** It constrains how the result
is written down, whatever it says, and it applies equally to a result that supports the
hypothesis and one that refutes it.

## 12. Stage 3: training — declared caveats recorded before the closed-loop numbers exist

**A3's result cannot distinguish "fine-tuning hurts" from "fine-tuning had not finished."**
A3's best step is its **final** step (15 000 of 15 000), so there is no evidence it converged —
it was still improving when its step budget ended. Every other arm's best step sits comfortably
inside its run (42 000/50 000, 40 000/50 000, 26 000/50 000). A3 used 207 s of a 7 200 s cap, so
**the binding constraint was the declared step count, not wall clock.**

`steps` and `a3_steps` are in the frozen training block. **They were not changed, and A3 was not
retrained** — altering either after seeing the ordering is exactly what the protocol forbids.
This is recorded as a limitation that travels with A3's reading wherever it is used.

**Both controls beat the primary arm.** The offline selection scores, lower better, are
**A1 0.01106 < A0 0.01313 < A2 0.01793 < A3 0.02340**. So A2, the preregistered primary, is
third of four — beaten not only by the random-encoder control but by **A0, which sees no image
at all**. A0 versus the best learned arm is the offline shadow of gate G2, and on this metric it
points the wrong way for the hypothesis. Stated here because an earlier version of this section
named only the A1 result and printed no scores, so a reader could not see where A2 sat.

**The A1 < A2 ordering is left as an open question.** A randomly initialized frozen encoder
scoring better than the trained one on the offline metric is not explained here. A plausible
story exists — the head may ignore the image in both cases, and A1's less structured features
may be less misleading — but it is untested, and an unexplained measurement is more useful to
the next generation than a tidy account of it.

**Declared expected failure mode: the policies will under-shoot the translation boundary.**
The expert corpus is saturated against its own limits — `right_dz` sits at exactly 0.400 in
**44.89%** of commands — and the arms are trained with a smooth-L1 objective, which is
minimized by hedging toward the interior of a bimodal target rather than committing to either
mode. The prediction, recorded here **before the development numbers exist**, is that the
learned arms' dz distributions will be *compressed relative to the expert's*: a lower rate at
or near the maximum, and mass pulled toward zero. If that happens it is a **declared expected
failure mode, not a discovery**, and it must not be reported as an insight found in the data.

This changes nothing about the gates. It is recorded so that the observation cannot later be
presented as explanatory. Two things will be reported alongside the cohort-D numbers so the
prediction is falsifiable rather than decorative: **the policies' own per-dimension dz
distribution in the same form as the expert's** (rate at maximum, rate in the out-of-distribution
band, quantiles), and **per-attempt step counts with cap hits**, since a policy that
under-shoots dz reaches the object later or not at all, and an attempt that ends on the step cap
is a different failure from one that ends on a bad grasp.

**A4 is deferred, not abandoned.** The readout-driven scripted controller is not built yet
because the development pre-check does not need it. It becomes *more* important if the learned
arms fail on development, not less: A4 is what distinguishes **Outcome D** ("perception is
inadequate in the loop") from **Outcome C** ("cloning is the failing component"). If all four
learned arms die on development, A4 is the next thing built and the protocol still decomposes
the failure. A future reader should not see it absent here and conclude it was dropped.

## 13. Next

Stage 2 was **not** a training launch. `src/embodied_jepa/policy.py`,
`src/embodied_jepa/cloning.py`, the `POLICIES` registry and the arm configs did not exist; the
protocol declared them as to-be-built. They are built, independently reviewed and merged
**before any arm is trained**, because a bug in the feature pipeline or the masking rules would
produce numbers that could not be trusted and would not announce themselves until the gates read
strangely.

That review returned **BLOCK** on the first submission, and it was right to. Three defects, all
in the **A3 fine-tune arm** and the **selection rule**, none of them touching the feature/target
alignment:

- **A3's validation was scored against stale features.** The val feature cache was built once,
  before training, from the pre-training encoder — so a head trained on live features was scored
  against an input distribution frozen at initialization. A3's val curve, `best_step` and
  selected checkpoint would have been meaningless *while still looking like a merely-bad curve*.
- **A3's fine-tuned encoder was never saved.** `FrozenEncoder` is a plain object, not an
  `nn.Module`, so it is absent from `state_dict()`; the trained encoder was silently discarded and
  the reloaded head would have been paired with the original E0 encoder.
- **The untrained step-0 policy was selectable**, and the run then reported `completed`. A random
  head passes the collapse rule — a LayerNorm-MLP at initialization has healthy per-dimension
  output std — so it could be written as `best`, never beaten, and fed to the gates.
  `world_model_v2.selectable` guards against exactly this with `step > 0` and says so in a
  comment; this runner had omitted it.

All three are fixed, each with a regression test. The reviewer's observation that **a two-step
test on `--encoder finetune` would have caught the first two** is correct, and that test now
exists.

### Why the review gate sits before training and not after it

**The smokes proved the pipeline runs. They could not prove it computes the right thing.**

Both stage-2 smokes — A0 and A2, on 20 episodes — completed successfully and reported
`status: "completed"` **while all three of the defects above were live.** The point is sharper
than "the smokes missed them": **two of the three are A3-only, and the A3 path was never smoked
at all.** The smokes did not fail to notice B1 and B2 — they never touched the code containing
them, and nothing about a green A0/A2 run said so. And had they been smoked, neither would have
announced itself:

- the stale-feature defect would have produced a plausible, merely-bad validation curve for A3;
- the step-0 selection defect would have produced a checkpoint that passes the collapse rule,
  because a LayerNorm-MLP at initialization has healthy per-dimension output std.

Both would have fed the G-gates numbers that look like results. The only thing that caught them
was an independent review reading the code before any arm trained.

The sharpened form, which is the one worth carrying: **a green run tells you nothing about the
code it did not execute, and it does not announce which code that was.**

**This is the third distinct instance of the same lesson in this protocol's short history**, and
they are worth listing together because the shape repeats and the surface changes:

1. §1 of the protocol originally presented **rollout readouts as perception measurements**. The
   values were correct and were independently verified; what nobody checked was *what code
   produced them*.
2. The frozen cohort's generator check asserted **bit-exact float equality**, which passed on the
   machine that wrote it and failed on Linux by one ULP. A green local suite is not evidence for
   a cross-platform claim.
3. The stage-2 smokes **ran to completion with three defects live**. Running is not measuring.
4. A round-1 fix added a **compatibility check that could not do what its own comment said it
   did**. `FrozenEncoder.provenance()` was entirely weight-independent —
   `model_implementation_sha256` hashes *source files*, `model_parameters` is a count — so the
   randomly-initialized encoder (A1) and the loaded E0 checkpoint (A2) produced **byte-identical
   provenance**, and `ClonedPolicy.load` could not distinguish the two arms whose difference is
   the entire content of gate G3. A code comment asserted it could, and the test named for that
   case passed only because two *unrelated* stand-ins differ in `kind`. The location is the
   point: **this was committed inside the fix for the finding about claims of protection.** The
   general form: *a comment asserting that a check is meaningful is not a check — it is an
   assertion the reader will believe and nobody will test.* The repair was a real digest
   (`model_weights_sha256`), a comment that explains **why** the check is meaningful rather than
   asserting **that** it is, and a test that exercises the A1/A2 case.
5. The first version of the tripwires below used **`pytest.skip`**, which reads as "do nothing
   here" — and would have failed the integration job on *every* run, because that job rejects
   any skip outside two allowed messages. The one mechanism that looks inert was the one that
   could not be inert here. They use a bare `return` instead, and the tripwires were then
   verified to fire by writing a probe runner that violates each constraint and watching both
   assertions trip — rather than assuming they were connected.
6. **The fix for instance 4 made the check meaningful and simultaneously made it always fail.**
   Adding the weights digest to `provenance()` closed the A1/A2 gap for real — and `load()`
   compared the whole provenance *before* restoring the encoder, so an A3 checkpoint's saved
   digest (post-training) could never match the live one (freshly loaded E0). **B2 went from
   "the trained encoder is discarded at save" to "saved and unreachable"**: the arm still could
   not be reproduced from its checkpoint, which is the property B2 existed to establish. The
   suite stayed green because the only test covering that path used a stand-in whose
   `provenance()` was hardcoded without a digest — **a test whose name covered a case its
   fixtures could not reach**, which is instance 4 again one level down. The repair splits the
   comparison around the restore, which is also strictly stronger: afterwards the digest asserts
   *"the encoder in memory is byte-identical to the one that trained this head"*.
7. The runner's new behavioural tests were added to `tests/test_policy_preflight.py`, which is
   **deliberately torch-free** so it runs in the core CI job. Importing the runner pulls in
   `embodied_jepa.policy` and therefore torch, so the whole file failed on both core jobs while
   the local suite was green at 782 — because this machine has the optional `learning` extra
   installed and the core job does not. Same family as instance 2: **a green local suite is not
   evidence for a claim about an environment you are not in, and you are in fewer environments
   than you think.** The durable fix is not "remember this": the core environment is now
   reproducible locally by blocking `torch` through `sys.meta_path`, which takes seconds and
   would have caught it before the push. *The commit that fixed this claimed in its message to
   have recorded the lesson here and did not — the renumber landed and the entry did not. An
   independent review caught the gap. That is itself the pattern, so it is recorded rather than
   silently repaired.*
8. One of those tripwires then **fired on a *correct* runner**, because a text grep matched a
   docstring explaining what the runner does *not* do. A false positive is not a weaker version
   of the right check; it is a different and worse thing, because it punishes whoever got it
   right and the cheapest way out is to delete the test. Replaced with an `ast.walk` over `Call`
   nodes and **verified in both directions** — correct runner passes, violating runner fails.
   The AST form had been available the whole time; the constraint that seemed to rule it out
   ("nothing stronger exists for a file that does not exist") was about *importing* the file and
   was generalized too far. The AST walk now also catches an aliased import and a binding to a
   local, and **names the hole it cannot close** — reimplementing the reset arithmetic inline
   references nothing — so whoever inherits it knows what they are inheriting rather than
   believing the parse is airtight.

9. The fix for instance 7 was a `sys.meta_path` blocker that made the core environment
   reproducible locally. It **did not work, and it reported success twice.** The blocker class
   defined `find_module`/`load_module` — the legacy import hook API **removed in Python 3.12** —
   so the interpreter never consulted it and `import torch` succeeded straight through it. Both
   "torch-free" verifications it produced (PR #38 and the first push of PR #39) were **false
   greens**, and the second one shipped a genuinely torch-dependent test into the torch-free
   file, which an independent review caught rather than CI. A control built specifically to
   prevent a failure class was itself non-functional, and because its output was
   indistinguishable from a real pass it *concealed* the class instead of catching it —
   strictly worse than having no control, which would at least have left the question open.
   The durable fix is not a better blocker but a **blocker that fails loudly when it is inert**:
   it now asserts `"torch" not in sys.modules` before installing itself and then attempts the
   import, exiting non-zero if it succeeds. Verified in both directions against the pre-fix
   file: **`1 failed, 17 passed`** with exactly the expected `ImportError`, versus **18 passed**
   after the split. The first negative control written for this was *also* worthless — it ran
   the pre-fix copy from `/tmp`, where all 18 tests failed on `FileNotFoundError` before
   reaching any import — which is instance 8's lesson (a test failing is not a test failing
   *for the reason you think*) recurring inside the fix for instance 9.

10. The runner crashed A3's entire development run on a physical guard stop (§17) — and the code
    **had a comment naming that exact exception and that exact consequence.** `run_attempt`'s
    `finally` block reads: *"G1Embodiment raises ContractError("measured joint velocity limit
    exceeded") from inside project_candidates, and without this the simulator is left actuating
    and the whole run dies with no report."* The author identified the failure, wrote it down,
    and then handled only half of it: the robot gets stopped, the attempt does not get a
    termination reason, and the run dies with no report exactly as the comment says. Two
    established files, `object_ceiling.py` and `hybrid_phase.py`, already carried the tuple that
    solves it. This is a different family from instances 1–9, and it is the more uncomfortable
    one: **the others are failures of verification, this is a failure to finish a thought that
    was already correct.** Naming a hazard in a comment produces the feeling of having handled
    it. The durable form: when a comment describes a failure mode, it must also say which line
    handles it — and if no line does, the comment is a bug report against its own file, not
    documentation. A secondary form: **precedent in the repository was not consulted.** Two files
    had solved this, and a grep for the error string — which is how it was eventually found —
    would have surfaced both before the runner was written.

The common form of instances 1–9: **verifying that something produces a plausible value does not
verify that it computes the quantity you believe it computes.** Every one of those was caught by
someone checking construction rather than output, and none would have been caught by more careful
reading of the numbers. **Instance 10 is not of that family** — it was caught by a robot refusing
to move — and it is left outside the generalization rather than folded into it, because a summary
that covers every instance by widening its own claim is the same error the list is about.

That is the argument for the ordering this protocol uses — build, review, merge, *then* train —
and it is recorded here rather than left in a message, because the next person to run a stage of
this protocol will be tempted to treat a passing smoke as a green light. It is not one.

**The withdrawal clause of §6 did not fire.** The reviewer verified the row/position
correspondence by execution rather than by reading — a synthetic corpus in which every quantity
carries a fingerprint of its own row, checked so that positions and rows genuinely diverge past
an excluded episode — and found the chain correct, including in `shuffled_frame_control`, on
which the stage-1 result depends. **Nothing is withdrawn.**

---

## 14. Stage 4: the development cohort — the numbers

Cohort D, 16 resets (45000–45007, 45100–45107), all four trained arms, `cpu`. Reports:
`outputs/task056-cohort-d/{a0,a1,a2,a3}.json` (git-ignored; hashes and seeds inside).

**This section contains no interpretation.** The reading is §16.

| arm | role | grasp resets | full successes | `dead_on_development` | cap hits | guard stops |
|---|---|---|---|---|---|---|
| `A0_proprio_only` | control | **0/16** | **0/16** | **true** | 16/16 | 0 |
| `A1_random_encoder` | control | **0/16** | **0/16** | **true** | 16/16 | 0 |
| `A2_bc_frozen_e0` | **PRIMARY** | **1/16** | **0/16** | false | 16/16 | 0 |
| `A3_bc_finetuned_e0` | — | **1/16** | **0/16** | false | 14/16 | **2** |

No attempt of any arm terminated on task success. No attempt terminated on task failure. Every
attempt that was not stopped by the embodiment ran to the 1000-step cap.

Stage occupancy, in commands issued while the scorer reported that stage:

| arm | `none` | `reach` | `grasp` |
|---|---|---|---|
| A0 | 16000 | 0 | 0 |
| A1 | 16000 | 0 | 0 |
| A2 | 15140 | 77 | 783 |
| A3 | 12593 | 906 | 803 |

Grasp occurred on seed 45006 for A2 and for A3. A3's two guard stops were seeds **45001** and
**45003**, at 143 steps and later.

Median control time per command: A0 12.1 ms, A1 13.6 ms, A2 13.6 ms, A3 13.8 ms. G7's threshold
is 100 ms and **is not enforced on development**; this is the same quantity measured, recorded so
the budget claim is checkable.

### 14.1 The policies' own dz distribution, against the expert's

Declared in §12 before these numbers existed. Expert corpus: `right_dz` sits at exactly **0.400
in 44.89 %** of the 68 791 BC target commands.

| arm | dz q50 | dz q90 | dz q99 | dz max\|·\| | dz out-of-distribution rate |
|---|---|---|---|---|---|
| A0 | 0.0125 | 0.0165 | 0.3586 | 0.4312 | 0.0070 |
| A1 | 0.0544 | 0.0548 | 0.3743 | 0.4136 | 0.0011 |
| A2 | 0.0317 | 0.0343 | 0.3649 | 0.6070 | 0.0353 |
| A3 | 0.0857 | 0.0890 | 0.3131 | 0.6490 | 0.0057 |

Rotation, the one family where the expert maximum and the configured bound coincide at 0.500 so
**the out-of-distribution rate *is* the clip rate**: `droll` 0.0510 / 0.0726 / 0.0602 / 0.0856
for A0–A3.

**No figure pooled across dimensions appears in this document**, for the reason the manifest
gives: translation saturation is unprecedented in the demonstrations while rotation saturation is
normal, and the grasp dimension sits at its maximum 91.78 % of the time by design, so a pooled
rate blends three incomparable things. A pooled clip count was present in the first draft of the
aggregation used to produce this table and was removed before the table was written.

**A per-dimension clip rate is not recoverable for translation from these reports.** The runner
stores `clipped_commands` as a single count of commands where *any* dimension was clipped. For
rotation the coincidence above makes it recoverable; for translation it is not. Recorded as a
limitation of the instrument rather than left as an absence.

## 15. The protocol gap — a term that decided the outcome and was never defined

**This is the most durable finding in TASK-056**, and it is about preregistration practice rather
than about this robot.

The stop rule (§5.3 of the protocol, `/stop_rule_protecting_cohort_C` in the manifest) reads:

> every learned arm runs the 16 development resets D; an arm reaching the scorer's `grasp` stage
> on 0/16 does **not** run on C

**"Learned arm" appears in four places across the two frozen artifacts and is defined in none of
them.** The manifest's `/arms` block independently tags A0 `"role": "control: the schedule
alone"` and A1 `"role": "control: do any visual features suffice"`, while also tagging both
`"trains": true`. **The stop rule keys off neither field.** So when both controls came back 0/16,
the frozen text could not say whether they were subject to the rule — not because it said
something ambiguous, but because it never addressed the question.

Neither A0 nor A1 is named in the stop rule in either artifact. The only in-text signal about
what "learned arm" ranges over is G6, which says "any **learned arm's** controller" and then
states separately that "**The assertion covers A4**" — which speaks to A4's inclusion, not to the
controls'. G2 and G3 contain no clause of any kind about what happens if their reference arm does
not run. §7's pre-declared outcomes contain no outcome, sub-case or clause conditioned on a
control being dead on development; its only sentence about an arm not running is "only P3 can
drop an arm," and P3 drops A4.

Two internal tensions, recorded verbatim because both exist under either reading:

- **`best_learned_arm_rule` selects "most full successes on C."** If "learned arm" has the same
  referent there as in the stop rule, the domain of that selection overlaps G2's subtrahend — the
  rule would admit A0 as a candidate for the arm G2 measures *against* A0.
- **The stop rule is headed "declared in advance, not a gate"** (§5.3's own heading), yet its
  stated consequence is that gate rows become `None`, and §5.2's unqualified missing-values rule —
  "A gate that cannot be evaluated **counts as failed**" — converts that to failed. A rule that
  announces it is not a gate determines gate outcomes.

**What a future protocol must do differently.** Define the domain of every scope term that
appears in a stop rule or a gate, at the point the term is introduced, by enumerating the arms it
covers by name. "Every learned arm" reads as precise and is not: it silently assumes the reader
shares an unstated partition of the arm list. Had §5.3 said "A0, A1, A2 and A3 each run the 16
development resets," there would have been nothing to rule on.

**This is not an embarrassment and is not recorded as one.** This protocol was written in
advance, gated, independently reviewed five times, and carries a frozen manifest with cohort
hashes — and it still contained a single undefined term that decided whether an irreplaceable
frozen cohort would be consumed. That is the finding: **specificity, review depth and freezing do
not by themselves close definitional holes**, because a reviewer checking whether thresholds are
justified does not naturally ask whether the nouns are defined.

**The ambiguity was deliberately not resolved.** Resolving it after seeing results — with the
fate of the frozen cohort riding on the answer — is the forbidden move in its purest form, and
the executing agent declining to make it is what allowed the decision to be taken on the ground
that *the outcome is the same either way*. That reasoning is only available to someone who has
not already picked a reading.

## 16. Reading

**Learned Apple→Plate remains at 0 successes.** Four arms, 64 development attempts, zero.

**No arm distinguished itself from the no-vision control on task outcome.** A2, the preregistered
primary, and A3 each reached grasp once in sixteen; A0 and A1 never left stage `none`. The
offline selection scores had already put A2 third of four, behind both controls (§12). Cohort D
does not predict cohort C and no gate was evaluated, so this is not a gate result — but there is
no development evidence that the primary arm does anything the blind control does not.

**The dz prediction was made in advance and is borne out, at that strength and no further.**
Every arm's median dz sits between 0.012 and 0.086 against an expert that spends 44.89 % of its
commands at exactly 0.400. The policies do not approach the boundary their demonstrations are
saturated against, which is the smooth-L1 hedging toward the interior of a bimodal target that
§12 predicted before the numbers existed. It is **a declared expected failure mode, not a
discovery**, and it must not be reported as an insight found in the data.

**It is a consistent story, not a demonstrated mechanism.** Under-shooting descent is compatible
with every attempt exhausting the step cap — a policy commanding a tenth of the expert's descent
plausibly does not reach the object in 1000 steps — but **that causal link is not shown here.**
Every arm hitting the cap on every attempt is equally compatible with several other mechanisms,
including ones in which dz is irrelevant. No experiment in TASK-056 separates them.

**A3's inversion is recorded as an open observation, not an explanation.** A3 has the highest
median dz (0.0857), the highest rotation clip rate (0.0856) and the largest dz excursion
(0.6490), and it is the only arm the embodiment physically stopped. The fine-tuned arm commands
the most aggressive motion and is the only one the platform refused. Why fine-tuning produces
that rather than better-scaled motion is not established here.

**A3's reading still carries its §12 limitation.** Its best step was its **final** step
(15 000/15 000), so its result cannot distinguish "fine-tuning hurts" from "fine-tuning had not
finished." It was not retrained and its step count was not raised.

## 17. A defect found by the run, and a fix authorized after the numbers were seen

**Stated plainly because it matters more than the defect: this fix was written and authorized
*after* the development numbers were seen.** It is recorded that way so no future reader has to
reconstruct the order from commit timestamps.

A3's first development run **crashed and produced no report at all**:
`ContractError: measured joint velocity limit exceeded`, raised from `embodiment.py:363` inside
`project_candidates`. One violation on one seed aborted all sixteen attempts.

`G1Embodiment` has two paths for the same physical condition. `execute` **returns** it through
`reject()` (`embodiment.py:425`); `project_candidates` **raises** it (`embodiment.py:363`). The
runner calls the raising one. Established precedent treats this refusal as a physical stop rather
than a software failure: `object_ceiling.py:59` and `hybrid_phase.py:31` both carry
`GUARD_REFUSALS = ("measured joint velocity limit exceeded",)` and terminate the attempt on it,
as TASK-046.

The fix records `guard_refusal` as a termination reason, counts the attempt as a **non-success**,
and lets the arm continue. **It moves no threshold, no limit and no gate.** The velocity stop
stays exactly where it is; an arm the platform refuses still fails that attempt. The catch
enumerates what **may** be caught rather than catching `ContractError` broadly, so it **fails
closed**: a malformed command, a stale observation or a pinned-component breach still propagates
and still kills the run loudly.

**Why changing results-producing code after seeing results was acceptable here, and when it would
not be.** Cohort D is the development cohort. It exists so arms can be exercised without
consuming frozen evidence, it was already consumed before this task began, and the protocol
states it "never gates and is never reported as a result." Re-running A3 on D is not re-running a
result, because D is not a result. **The hazard this rule guards against bites when the numbers
are evidence; these are not.** The same change against cohort C would have been refused.

**Unplanned cross-check.** Before the fix existed, A3 was probed as 16 independent single-seed
runs to find which seeds tripped the stop. The fixed runner's 16-attempt report reproduces those
per-seed outcomes exactly — guard stops on 45001 and 45003, grasp on 45006 — which is evidence
the change altered error handling and nothing else.

