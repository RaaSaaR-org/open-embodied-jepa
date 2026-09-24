# Apple→Plate policy v1 results (TASK-056) — stage 1: pre-flight

Protocol: [apple_policy_v1.md](apple_policy_v1.md). Manifest:
`benchmarks/manifests/apple-policy-v1.json`, merged as `3006b3b` (PR #33).

**Status: the pre-flight has run and passed. No arm has been trained. Cohort C (seeds
45300–45339) has never been simulated. Learned Apple→Plate remains at 0 successes.**

Nothing in this document is a manipulation result. It reports a perception measurement on
recorded validation data.

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

## 8. Process notes

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

## 9. How the stage-3 results will be read — committed before they exist

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

## 10. Stage 3: training — declared caveats recorded before the closed-loop numbers exist

**A3's result cannot distinguish "fine-tuning hurts" from "fine-tuning had not finished."**
A3's best step is its **final** step (15 000 of 15 000), so there is no evidence it converged —
it was still improving when its step budget ended. Every other arm's best step sits comfortably
inside its run (42 000/50 000, 40 000/50 000, 26 000/50 000). A3 used 207 s of a 7 200 s cap, so
**the binding constraint was the declared step count, not wall clock.**

`steps` and `a3_steps` are in the frozen training block. **They were not changed, and A3 was not
retrained** — altering either after seeing the ordering is exactly what the protocol forbids.
This is recorded as a limitation that travels with A3's reading wherever it is used.

**The A1 < A2 ordering is left as an open question.** A randomly initialized frozen encoder
scoring better than the trained one on the offline metric is not explained here. A plausible
story exists — the head may ignore the image in both cases, and A1's less structured features
may be less misleading — but it is untested, and an unexplained measurement is more useful to
the next generation than a tidy account of it.

**A4 is deferred, not abandoned.** The readout-driven scripted controller is not built yet
because the development pre-check does not need it. It becomes *more* important if the learned
arms fail on development, not less: A4 is what distinguishes **Outcome D** ("perception is
inadequate in the loop") from **Outcome C** ("cloning is the failing component"). If all four
learned arms die on development, A4 is the next thing built and the protocol still decomposes
the failure. A future reader should not see it absent here and conclude it was dropped.

## 11. Next

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
7. One of those tripwires then **fired on a *correct* runner**, because a text grep matched a
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

The common form: **verifying that something produces a plausible value does not verify that it
computes the quantity you believe it computes.** Every instance was caught by someone checking
construction rather than output, and none would have been caught by more careful reading of the
numbers.

That is the argument for the ordering this protocol uses — build, review, merge, *then* train —
and it is recorded here rather than left in a message, because the next person to run a stage of
this protocol will be tempted to treat a passing smoke as a green light. It is not one.

**The withdrawal clause of §6 did not fire.** The reviewer verified the row/position
correspondence by execution rather than by reading — a synthetic corpus in which every quantity
carries a fingerprint of its own row, checked so that positions and rows genuinely diverge past
an excluded episode — and found the chain correct, including in `shuffled_frame_control`, on
which the stage-1 result depends. **Nothing is withdrawn.**
