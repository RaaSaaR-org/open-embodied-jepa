# Apple→Plate first learned policy v1: results (TASK-067)

**Outcome: CAL-ESCALATE.** The run stopped at the C0 calibration, as §7.1 of the protocol
prescribes. The expert's tolerance to plate error fell below 28/32 at the smallest preregistered
plate level (1.0 cm → 21/32), so no feasible plate bar exists (the apple bar was feasible). This
is a declared early-stop row. It is **not** V: nothing is repeated. **The abandonment clause does
not fire; the owner decides what follows** (§9).

**Status lines.**
- **Learned Apple→Plate is still 0 successes.** No policy was trained: the run stopped before
  BC-0. No development-cohort (D) reset was simulated, and cohort C was not touched.
- The test split of `apple-look-v1` was not decoded (`test_split_decoded: false`). The run read
  190 train and val episodes.
- `exemption_spent` stays `false`.
- Every number below comes from C0, a **scripted, privileged calibration** (rung L4). None of it
  is a learned result.

Protocol: [`apple_first_policy_v1.md`](apple_first_policy_v1.md) (#78, #79). Results manifest:
`benchmarks/manifests/apple-first-policy-v1-results.json`, produced from `report.json` by
`scripts/summarize_first_policy.py`. §1–§3 give the numbers; §4 gives the reading.

---

## 0. Owner rulings (verbatim)

R1–R5 were received 2026-09-27T12:43Z via the coordinator:

> R1 (clause scope): allowed. The TASK-057 clause stops BC on `apple-wide-v1` + 112 px onboard.
> TASK-067 uses a different corpus (`apple-look-v1`), a different observation protocol (the look
> prefix) and a different encoder (DINOv2, readable per TASK-063/064), so it is not a third
> formulation on that corpus. Disclose prominently that the camera and resolution are unchanged,
> and state the rationale as an information change, not a retry of the formulation.
>
> R2 (learned inputs): accepted. The fixed look prefix, the step counter and the palm pose from
> forward kinematics all count as legitimate non-privileged inputs. A perception head trained on
> privileged labels is fine, provided the evaluation makes zero privileged reads.
> Privileged-expert DAgger labelling is fine at training time only. Label all of these explicitly
> in the "learned" ladder.
>
> R3 (primary, fallback, milestone): accepted as proposed. Up to 4 dev-cohort evaluations, with
> the carried policy selected on dev. The dev result is an existence result only. Any "learned
> policy works" claim must come from the gated cohort-C test against open-loop replay (5/16 on
> dev) and the named controls. Preregister the fallback trigger exactly as you proposed.
>
> R4: accepted. A first success counts as "learned policy with a DINOv2 encoder", never "LeWM
> driving the robot". The docs keep that distinction.
>
> R5: seeds 46000–46999 are accepted, on condition that you grep the repo, including the
> manifests, and confirm no overlap. Record the check.

R6 was received 2026-09-27T13:13Z and also posted on #78:

> Owner ruling R6 (2026-09-27 13:13Z), also posted on #78: both §17 departures are accepted.
> C-3 gets its own 3 DAgger rounds, and R-3 is trained and run on the development cohort. Both
> make the controls stronger, so G2 and G3 become harder to pass, not easier.

R7 was posted on #79 at 2026-09-27T13:56:08Z:

> Owner ruling R7 (2026-09-27 13:56Z) on amendment 1 items 4–10: all are accepted as implemented,
> with one condition on item 5. Known render nondeterminism (1–2 px under 12 workers, recorded
> earlier) could make the post-look frame sha256 re-render check void attempts. Before GO, the
> pre-run reviewer must confirm, from a smoke at the gated worker count, that the re-render is
> bit-identical. If it is not, the run does not start and the issue comes back to the owner. It
> must not be relaxed silently. Items 6 and 10 are accepted as conservative: a skip scores 0/16
> or makes the run void, never a pass. Item 8 is accepted: B-random is a sanity control, and a
> fixed sequence keeps it reproducible. Record R7 verbatim in PR 3.

## 1. Provenance

| item | value |
|---|---|
| revision | `9b0c5806ef8f4da1c4d68903290ece0ec5bdf5e3` (main after #80), detached, `tracked_tree_dirty: false` |
| command | `scripts/run_first_policy.py run --output outputs/task067-first-policy/run-1 --checkpoints checkpoints/task067-first-policy/run-1`, run by the main checkout's `.venv` Python (the environment `uv run --no-sync` uses), from the task agent's worktree at that revision |
| pre-run GO | the pre-run reviewer's **reported** verdict was `PRE-RUN: GO`. Its R7 render check (`outputs/task067-scratch/prerun-go-render-1/report.json`, sha256 `b5de81a7…336f`) was IDENTICAL: 8 of 8 workers, 32/32 seeds rendered on ≥ 2 workers, 0 errors, and the negative control raised G-frame. |
| started / ended | 2026-09-27T23:06:14Z to 23:17:33Z (UTC, machine clock); 679 s |
| report | `outputs/task067-first-policy/run-1/report.json`, sha256 `4da883de…eb3b` |
| environment | macOS 26.5.1 arm64, Python 3.12.13, NumPy 2.5.3, torch 2.14.0, MPS available; 8 simulation workers |
| guards | preflight passed: G-frozen, G-hash (21 pins), clean tree, G-seeds, G-device, G-data (manifest `81d760d1…db64`, 170/20 train/val roots), Q-split, G-weights (digests `3a697b87…af27` and `3d305f9c…c9db`); G-look held on every attempt; the end-of-run G-hash held on all 21 pins; `non_finite_fields` empty |
| seeds simulated | perception-train 46000–46255 (256), perception-held-out 46256–46383 (128), C0 46384–46415 (32). No D, DAgger or cohort-C seed was simulated. |

## 2. Stages completed before the stop

- **Perception frames.** 384 fresh resets were rendered after the look. The 170 train and 20 val
  roots' post-look frames were decoded.
- **Readouts.** The P (pretrained) and R (floor) readouts were fitted on 426 frames. Both chose
  the linear family, with λ_rel 0.01 (P) and 0.001 (R).
- **S0-P was not evaluated.** It comes after C0, and CAL-ESCALATE comes first. The held-out
  errors were computed in memory but never written or seen. **This run says nothing about
  perception accuracy.**

## 3. C0: the expert's tolerance curve (scripted, privileged; successes of 32)

`apple_collector_policy` was built from the reset truth plus an injected xy error of a fixed size.
Each seed gets its own direction, drawn from `default_rng(6700)`. There is no action noise, and
the attempt runs with the look.

| condition | successes |
|---|---|
| reference, no error | **32** |
| apple 0.5 cm | 32 |
| apple 0.8 cm | 32 |
| apple 1.0 cm | 28 |
| apple 1.2 cm | 20 |
| **plate 1.0 cm** | **21** |
| plate 1.5 cm | 21 |
| plate 2.0 cm | 17 |
| plate 2.5 cm | 10 |

**The rule** (`first_policy.c0_bars`, protocol §7.1) gives `{"escalate": "C0: the smallest plate
level is below 28/32"}`. `summarize_first_policy.py` recomputes the same result from the counts.
For the apple, the rule would have given a p90 bar of 1.0 cm and a median bar of 0.75 cm, but the
plate part has no feasible level, so CAL-ESCALATE is the row.

## 4. Reading (interpretation, not measurement)

1. **The expert tolerates apple estimate error up to about 1 cm.** It succeeds 32/32 at 0.5 and
   0.8 cm, 28/32 at 1.0 cm and 20/32 at 1.2 cm. This fits the aim-offset evidence (1/36 successes
   at 1.5–3.0 cm, proposal §2.2). Apple perception on the order of TASK-064's 0.46–0.55 cm medians
   would have been within budget, subject to its tail.
> **Erratum, 2026-09-28 (owner ruling R9).** The "releases 3.5 cm short" explanation in item 2
> below is **contradicted** by the R8 release-point probe
> ([`apple_first_policy_v1_release_probe.md`](apple_first_policy_v1_release_probe.md), row
> P-CANDIDATE-FAIL):
> - With the plate exact, the current collector lands a median 2.7 cm short of the centre, not
>   3.5 cm.
> - A landing that short stays inside the 4 cm radius under a 1.0 cm plate error.
> - All 11 failures of the current collector at 1.0 cm land forward or sideways, at 4.53–4.62 cm
>   from the centre (the rim), not short.
>
> The cause of the failures is unidentified. The text below is kept as written.

2. **The expert does not tolerate plate error: 1.0 cm already costs 11 of 32 successes.** A
   likely mechanism can be derived from the collector's code, but **it is not measured here**,
   because the report stores C0 counts, not per-attempt apple end positions:
   - `apple_collector_policy` puts the palm 1.5 cm behind the apple in x (`obj − 0.03 + 0.015`).
   - It releases at `container_x − 0.03 + 0.015 − 0.035 = container_x − 0.05`.
   - If the held apple keeps its grasp offset, it lands about 3.5 cm short of the plate centre in
     x, which is only about 0.5 cm inside the scorer's 4 cm plate radius (`task.py`).
   - A plate error whose x component is about −0.39 cm or less (pointing toward the robot) then
     moves the landing point outside the radius. For a 1 cm error in a uniformly random
     direction, that happens with probability of about 0.37, which is consistent with 11/32 = 0.34
     failures.
   - **The simple model fits unevenly across levels.** With the run's actual `default_rng(6700)`
     plate directions (review of #83), it predicts 21 / 18 / 17 / 14 successes at 1.0 / 1.5 /
     2.0 / 2.5 cm, against 21 / 21 / 17 / 10 observed. It is a candidate mechanism, not a fitted
     one.
   - Under this reading the collector's own placement has almost no x margin, and a learned policy
     that imitates it inherits that margin. **Whether a plate readout can reach about 0.5 cm in x
     is not known from this run:** S0-P was never evaluated, and no absolute plate-xy readout
     from the post-look frame has been measured. Earlier protocols measured apple–plate
     relative readouts on other corpora (world-model v2–v4, gate G3); those are not comparable.
   - **The harness alternative in protocol §9 ("the harness is off") is not supported.** The
     reference condition scored 32/32 and the two smallest apple levels 32/32, on the same seeds
     and code path.
3. **What this does and does not show.**
   - **It shows** that TASK-067 as preregistered cannot proceed past C0. The clause does not fire,
     because nothing about perception or learning was tested.
   - **It does not show** that a learned policy cannot place the apple. It does not show that the
     plate is unreadable. It says nothing about the approach, the grasp or the carry.

## 5. What the owner decides (recommendations; the owner chooses)

The run stops here. Nothing is re-thresholded, and C0 is not re-run on these seeds. The options,
each needing its own disclosed amendment or a new protocol version:

- **(a) Move the collector's release point toward the plate centre, and collect a corpus v2.**
  For example, change `transfer_x_shift` so the apple lands near the plate centre, not about
  3.5 cm short of it. The BC labels come from the collector, so this is a data change:
  - a new look-prefix corpus, about 20 min of collection at TASK-064's rate;
  - a repeated readability gate;
  - C0 re-run on fresh calibration seeds.

  It addresses the mechanism in §4.2 directly. **Recommended**, once the mechanism is confirmed by
  a small, disclosed probe that records C0's per-attempt apple landing positions on fresh
  calibration seeds.
- **(b) Add plate levels below 1.0 cm to C0** (for example 0.25 and 0.5 cm). This would at best
  set a plate bar of about 0.5 cm or tighter. Whether a readout can meet that is unknown, so
  S0-PLATE-FAIL would be a possible next row. It also leaves the collector's small x margin in
  place, which is why (a) is recommended first.
- **(c) Stop TASK-067 here** and record the calibration finding.

**Context, not a basis for this decision.** TASK-066 has since ended as WM-TOK-DYNAMICS on the
train split, with four caveats (#81). That affects the proposal's stage 3, not this row.

## 6. Caveats kept in the record

- **C0 counts are per condition, over 32 seeds, with one direction per seed.** The same seed
  directions are reused across levels, so the level counts are not independent samples. No
  interval is attached.
- **The report does not keep per-attempt C0 records.** The §4.2 mechanism is derived from code
  constants and an assumption, not measured.
- **The run is one run.** CAL-ESCALATE is a declared outcome, not a void, and it is not repeated.
