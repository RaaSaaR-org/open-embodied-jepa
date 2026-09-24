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

1. **The stage-1 result is WITHDRAWN, not patched.** §1's gate table stops being the result.
2. The fix lands, and **the runner re-runs from the same frozen checkpoint**. The new numbers
   stand on their own — **including the possibility that a gate which passed now fails**, and
   with it the possibility that Outcome E fires and this task stops.
3. **The 0.491 / 0.456 / 0.873 figures are not carried forward alongside a corrected runner.**
   They are reported as withdrawn, next to what replaced them.
4. **No argument is entertained about whether the defect "would have mattered."** A result whose
   correctness was established by reasoning after the fact is not a measurement. This clause
   exists to remove that conversation, not to have it more carefully.
5. The withdrawal, the defect and the re-run are all recorded here, **with both sets of numbers**,
   so the record shows what we believed and what corrected it.

This is not expected to fire. If it does not, this section stays in the document as a record of
what was committed to while the answer was still open.

## 7. Process notes

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

## 8. Next

Stage 2 is **not** a training launch. `src/embodied_jepa/policy.py`,
`src/embodied_jepa/cloning.py`, the `POLICIES` registry and the arm configs do not exist; the
protocol declares them as to-be-built. They are built, reviewed independently and merged
**before any arm is trained**, because a bug in the feature pipeline or the masking rules would
produce numbers that could not be trusted and would not announce themselves until the gates read
strangely.
