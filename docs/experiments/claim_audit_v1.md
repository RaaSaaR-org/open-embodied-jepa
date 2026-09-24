# Numerical-claim audit v1: accepted claims against the code that computes them (TASK-058)

**Date:** 2026-09-25. **Repository revision audited:** `8306633` (main). **Scope:** TASK-048 to
TASK-057. That covers every accepted numerical claim in the TASK-048+ documents under
`docs/experiments/`, the matching manifests under `benchmarks/manifests/`, the `.mc` evidence logs of
TASK-048 to TASK-056, `docs/DECISIONS.md`, the README status section, and the TASK-048+ result
restatements in `docs/MODELS.md` and `docs/EVALUATION.md`. The MVP "0/150" headline is included
because the current top-level documents restate it. `docs/experiments/apple_policy_diagnostics_v1.md`
(TASK-057) and its manifest are cross-referenced and **not edited**; their owner is amending them
separately.

**What this audit is not.** It does not check that numbers match their JSON. That check has passed
on every defect this project has found and has caught none of them. For every claim, the audit
names the function that computes the number, reads the function body (not its docstring), and
records five things:

- what the function actually measures;
- the population it ranges over;
- whether the entities the prose names match the provenance;
- where a threshold is involved, whether a blind baseline passes it. The baselines are a constant or
  train-prior, a copy-last or persistence predictor, the model's own non-predictive readout, and a
  proprioception-only probe.
- where an absolute value, a pooled median or an unconditional aggregate destroys the distinction
  the claim needs. Such a claim is recorded as **unmeasured**, which is not the same as refuted.

## Classes

| Class | Meaning |
|---|---|
| **confirmed** | The code computes what the prose says, over the population the prose implies. The entities match, and any gate is not passable blind, or the text already says it is. |
| **mislabelled** | The number is right, but it is described as a different quantity or population, or read for more than the statistic can carry. The unmeasured cases fall here. |
| **wrong** | The number, entity or verdict is incorrect given the code and artifacts. |
| **unverifiable** | The claim cannot be traced to code plus a committed artifact. This includes claims that exist only in a git-ignored run artifact or only in prose, which are **not citable** under the standard of `apple_policy_diagnostics_v1.md` §9 (credited there to `apple_policy_v1.md` §9). |

## Result

145 rows. Six corpus slices were audited independently, then adjudicated here (see
"Adjudications"). The per-slice counts are recounted from the rows below, after the
S4-07 reclassification. Slice 5's own summary said 9 confirmed / 7 mislabelled; its rows give
10 / 6.

| Slice | Corpus | confirmed | mislabelled | wrong | unverifiable | rows |
|---|---|---:|---:|---:|---:|---:|
| S1 | TASK-048/049/051/053: wide collection, object ceiling v2, grasp-closure diagnosis and v3 | 9 | 6 | 1 | 4 | 20 |
| S2 | TASK-050: world model v2 | 15 | 6 | 1 | 3 | 25 |
| S3 | TASK-052: world model v3 | 13 | 6 | 0 | 5 | 24 |
| S4 | TASK-054/055, DECISIONS, README, MODELS, EVALUATION, MVP 0/150 | 11 | 12 | 3 | 0 | 26 |
| S5 | TASK-056 pre-flight, training, preregistration | 10 | 6 | 3 | 0 | 19 |
| S6 | TASK-056 closed loop, handover, TASK-057 cross-reference | 13 | 9 | 3 | 4 | 29 |
| L | Lead rows (README / CLAUDE.md top-level wording) | 1 | 1 | 0 | 0 | 2 |
| **Total** | | **72** | **46** | **11** | **16** | **145** |

One of the 11 **wrong** rows, S4-19 (the pilots' "failing … every other gate"), was already
withdrawn in its source before this audit.

## What changes, and what does not

**Unchanged.** No preregistered gate verdict flips, and no pre-declared outcome changes:

- the TASK-048 acceptance;
- the v2 ceiling's 5/8 FAIL and the v3 ceiling's 8/8 PASS;
- every world-model v2, v3 and v4 gate verdict;
- the TASK-054 Outcome B, i.e. the abandonment of CEM over this world-model cost. It is keyed on G2a,
  which is correctly computed, and no blind predictor passes it;
- the TASK-056 pre-flight PASS verdicts and the TASK-056 FAIL.

The "0 learned successes" headline stands.

**Changed.** These readings of past results are corrected or withdrawn. In each case the number was
right and its construction did not support the reading.

1. **TASK-057's shadow expert is the wrong policy (S6-29, wrong entity; cross-reference only).** The
   demonstrations and BC targets come from `scripted.apple_collector_policy(sim.task_truth())`, not
   from `OracleManipulationPolicy` (`collect_apple_wide.py:run_root` L465, `scripted.py` L192–207).
   The collector policy is `EarlyReleaseOracleManipulationPolicy(opening_ramp=0.08)` with +0.015 m
   palm-x on every phase and −0.035 m transfer-x on phases ≥ 4. Its phase budget is **745**
   commands (orient 130, descend 80, close 45, lift 150, transfer 60, release_high 100, lower_open
   100, retreat 80), not 805. The repository's own `scripted_oracle` is also
   `apple_collector_policy` (`evaluate_apple.py` L2171).
   - **Why D1 is affected.** The step-zero commands were rebuilt from labels. The rebuild reproduces
     the recorded `collector__base_action` exactly on all 152 roots. Applied to cohort D, it gives a
     step-zero `dx` difference between the two experts of median 0.048 over 16 seeds, and more than
     0.057 on 7/16 seeds, with the sign reversed on three. The dx threshold is 3 × A0's Table-A error
     = 0.057, so the median gap is **84 %** of it.
   - **So D1 can fail for the wrong reason.** A perfect imitator of the collector would still
     pass D1-dx on every arm: its median gap of 0.048 is below the lowest dx threshold, A0's
     0.057. But the expert mismatch alone uses up 84 % of A0's dx threshold and 59 % of A2's.
     That leaves an imperfect imitator little room, and it can fail D1-dx, and so count
     towards the abandonment clause, because the reference is the wrong policy rather than
     because it imitates badly. The extent is not quantified.
   - **What is not affected.** `dy`, `dz`, rotation and grasp are identical at step zero, so
     D1-grasp is unaffected.
   - **Caveat.** This is a label-based reconstruction. It assumes cohort D's stored reset `object_xy`
     is the apple's world xy. It is not a simulation.
   - **A second defect with the same cause (S5 and S6 cross-references).**
     `scripts/measure_policy_offline_conditionals.py` L50 takes its `PHASES` names from
     `OracleManipulationPolicy`. So Table E of `apple_policy_diagnostics_v1.md` and
     `apple-policy-diagnostics-v1.json` label collector phase 5 (699 val rows, really
     `release_high`) as `lower`, and phase 6 (178 rows, really `lower_open`) as `release`. The
     numbers are right; the names are wrong.
2. **P3's "no shortcut available" is wrong (S5-04).** The claim is that "absolute apple position
   genuinely cannot be inferred from the palm's own pose".
   - **What was measured.** A linear least-squares probe from the 86-D proprioceptive state (no
     image, no object state, no phase), fit on train and scored on val, reads orient-phase apple
     position to **1.01 cm**. Fit on orient rows only, it reads **0.163 cm**, against E0's 0.873 cm.
     The reason is that during `orient` the collector servos the palm to a fixed apple-relative
     station.
   - **What stands.** P3's PASS stands, because the gate is a ratio against E0's own foreign-image
     control.
   - **What is withdrawn.** The reading "the cleanest measurement … stronger evidential status …
     the result that most supports A4" is withdrawn. The pre-flight shows that E0's readout
     *depends on* the image. It does not show that the image carries information beyond
     proprioception on these cohorts.
   - **P2 (S5-06, S5-05).** P2's threshold rationale is also wrong. It borrowed the 2.394 cm prior
     for absolute apple position. For P2's palm-minus-apple quantity, a per-phase constant scores
     0.10–0.35 cm, better than E0's 0.456 cm. So P2's absolute threshold is passable by a
     phase-conditioned prior (a phase-free global constant scores 1.569 cm and narrowly fails), and "R2 not borne out" is unmeasured rather than refuted.
3. **The closed-loop dz "under-shoot" is withdrawn as unmeasured (S6-12).** Every statistic behind it
   passed through `np.abs`. It also compared a per-attempt median of |dz| over mostly pre-reach
   commands with an all-phase expert saturation rate. The expert's own median |dz| in `orient` is
   0.057, inside the policies' range of 0.013–0.086. `apple_policy_diagnostics_v1.md` §1.1–§1.2
   already records this as unmeasured, but the source documents still carried "borne out" until
   this audit's errata.
4. **The v3 grasp closure is not prediction-free (S1-13, wrong).** The claim "The close phase needs
   no prediction: v3's closure is a fixed schedule" is contradicted by the code:
   - `object_ceiling_v3._bounds` pins lateral and rotation, and leaves the vertical command to the
     CEM on every close command;
   - the CEM runs over exact MuJoCo rollouts, with a cost built from the true apple pose;
   - in the 24 recorded closes the vertical command ranged from 0 to −0.5 (std 0.17–0.21);
   - every fixed-schedule closure in the diagnosis failed.

   This matters for the BC line. The close is not a solved open-loop primitive. The only closure in
   the record that works without prediction is the scripted collector's position servo, and it has
   been shown only from its own entry state.
5. **The "ejection" mechanism is overstated (S1-09, S1-16, S1-17).** The traces show three things:
   - the 2–3.5 cm early push is transient: the apple is re-caged and still in contact 4–6 commands
     later, and is lost 10–40 commands after that;
   - the "complete separation" is 3/4 lost against 0/12, not 4/4 ejected;
   - "statistically indistinguishable from proposal noise" used the wrong null, and no test was run.

   `apple_policy_v1.md` §2.3 (R1) restates this mechanism as proven.
6. **Gate counts include a gate a constant passes (S2-06, S3-10, S4-05).**
   - **G4.** In v2, v3 and v4, G4 (apple height, grasp cohort, ≤ 1.0 cm) is passed by predicting
     a constant height. On v4's identical 2,398 windows, height 0 scores 0.36 cm. On v2, the
     train-median constant scores 0.047 cm, better than every arm.
   - **G3 and G5.** These are passed by copying the **true** (simulator) start state forward: 0.74 cm
     and AUROC 0.956. That baseline is privileged, not a blind predictor. G5 is also passed by the
     model's own no-prediction readout: v2 AUROC 0.996 (S2-07).
   - **v4's "the control passed more gates (10 of 14) than every intervention" (S4-09).** Once these
     gates are set aside, the claim reduces to E0 alone clearing G6a at h = 16.
   - **"Every intervention hurt candidate ranking" (S4-10).** This holds only at h = 16. At the
     planner's own h = 8, E1 scores highest (0.50 against E0's 0.36). That is on 10 groups, below
     the 12-group cohort rule, so it is also not established.
7. **The encoder/rollout decomposition is a difference and a ratio of medians (S2-08, S3-05–S3-07,
   S4-12, S4-13).**
   - **v2.** v2's "3.13 / 3.26 cm, 89 % of the error is in the encoding" was never written to any
     artifact. A TASK-058 re-run with the frozen checkpoint reproduces it. On 46 % of the moving
     windows, the rollout is closer to the truth than the directly encoded target.
   - **v4.** The median of the per-window rollout excess is 0.47–0.53 cm in all four v4 arms,
     including the control. The published G9 values are 0.80–1.12 cm.
   - **E2.** E2's "90 % of its gain is in the targeted term" is withdrawn: the per-window change is
     −0.010 cm [−0.134, +0.134].
   - **v3.** v3's "ranking improved over v2" (S3-12) is unmeasured: the v3 and v2 figures are
     different statistics.
   - **Decisions.** No pre-declared outcome depended on G9 or on the decomposition. The branch-2
     choice at TASK-052 fired on G7a.
8. **Top-level restatements (S4-15, S4-18, S4-24, S4-25, L-01).**
   - **S4-15 (wrong).** The README's "about 15 % of the needed change" is 29 % (under a third).
   - **S4-18.** The README's encoder figure drops that it covers moving windows only, is fused with
     proprioception, and uses val, which was also the selection split.
   - **S4-24.** "Motion-weighted readout shaping" was a bundled factor.
   - **S4-25.** "0/150 per model" drops two things: every learned MVP episode ended on a joint-rate
     guard stop, and the same 50 resets were reused across seeds. Per run, the mean is about
     5–22 commands per episode, and the per-run median is 3–15 commands. Per episode the range is
     0 to 170.
   - **L-01.** "Every task-specific apple control attempt since has failed its declared gate" is
     true of learned attempts only. The privileged v3 ceiling passed its gate.

**Findings for the behaviour-cloning line.** These are not verdict changes. Gate any world-model
critic readout against the model's own encoded-start readout, a train-prior constant, and a
proprioception-only probe; several v2 to v4 passes did not survive those baselines. On the BC
selection metric, a phase-only constant (0.01685) beats the primary arm A2 (0.01793) and A3
(0.02340) (S5-14). A3 trained on about 27× fewer samples than the other arms (batch 32 against 256),
which the record does not state (S5-16). The learned-arm lift is confirmed as an end state only:
A2/45100 +0.168 m and A3/45006 +0.169 m, in contact and undropped at step 1000. The runner stores
no per-step contact, so "held throughout" and "sustained contact" are unmeasured (S6-07). A3
reached the `reach` stage on 4/16 seeds, which no results document states (S6-05).

## Adjudications made when merging the slices

- **S4-07** (README and v4 "most of that [`apple_held`] signal comes from the encoded state and fused
  proprioception") was classed *confirmed* by slice 4 and is reclassified **mislabelled**. It stays
  consistent with S3-11, which applied the same reasoning to the same sentence in v3. A
  shuffled-action control gives *wrong* actions, not absent ones, so it cannot divide the AUROC
  between state and action. The attribution is unmeasured. The direction is plausible: copying the
  true start state reaches AUROC 0.956 (held flag) and 0.997 (height).
- **G4 baselines.** Slices 3 and 4 used the constant "height = 0" (0.36 cm). Slice 2 used the
  train-median constant (0.047 cm). Both pass ≤ 1.0 cm. The rest label is about −3.4 mm, not 0, so a
  better constant does better still.
- Slice 1's S1-12, the v3 `close_phase_apple_xy_cm` manifest field, reproduces from traces, but no
  committed code computes it. It stays **unverifiable**, not wrong.

## Lead rows

| ID | Claim | Where | Computing code | What it measures / population | Class | Note |
|---|---|---|---|---|---|---|
| L-01 | "every task-specific apple control attempt since has failed its own declared gate" | README.md:11-12; CLAUDE.md "Research-evidence rules" | gate evaluators per protocol (`evaluate_apple.py:object_ceiling_v2_gate` etc.) | All TASK-032+ apple control protocols | mislabelled | The privileged v3 grasp-closure ceiling passed its primary gate 8/8 (S1-11), and the TASK-056 pre-flight passed P1–P3. The statement holds for **learned** control attempts. The README text is corrected by erratum. CLAUDE.md is not edited by this audit; see "Not done". |
| L-02 | "No representation collapse: effective rank 6.74–8.23, collapsed fraction 0.000, mean latent std 0.850–0.873 on all four v4 arms" | README.md:44-45; v4 results :325, :479 | `world_model_v2.collapse_metrics` → `models/base._statistics` | Encoded **fused** (image + proprioception) val latents | confirmed | G8 on a fused latent cannot see an image-encoder collapse (S2-17). The v4 results report image-only rank 7.19 (E0) and 6.00 (E3), above G8b's 4, so the claim holds for the arms with an image-only figure. |

## Provenance recorded by this audit

- **TASK-056 policy checkpoints.** No committed file held these hashes before this audit (S6-01).
  Each matches the `checkpoint_sha256` in the git-ignored `outputs/task056-cohort-d/a?.json`.
  - a0 `960dd6f094b2948a99c48017b36047791ba9778e65565c91d32309ddc33189f4`
  - a1 `6e9912b82cbb9a03e2868b2682c59def68a77f11c2e8d94366571026884bad46`
  - a2 `0dc2fb701862d418beb12fb9a0c9f0aec86509eeee8f99966a21c908cf6b8735`
  - a3 `e61e2e090c3130f2555b9c87f4abae415489a9724d7870a663f836fe903614c1`
- **Cohort-D code revision.** The cohort-D reports record revision `103b14f`, which is **not an
  ancestor of `main`**. It survives only on `origin/feat/task-056-clip-instrumentation`. `src/` is
  identical to `6e728ce`. Do not delete that branch while these reports are cited (S6-25).
- **Scope of `python_source_sha256`.** This field hashes `src/` only, not `scripts/`. The dirty
  runner that produced `a3.json` is identified only by the later commit `6e728ce` (S6-24).

## Reproduction

The audit's computed numbers are themselves claims, and they are held to the same standard. The
scripts that produced them are committed under `benchmarks/audits/task058/`:

- `s1_*.py`: ceiling traces and collection labels;
- `s2_baselines*.py`, `s2_decompose_v2.py` and the output `s2_decompose_lewm_onboard_mps.json`;
- `s3_labels.py` (writes `labels.pkl`), then `s3_analyse.py`;
- `s4_baselines.py`;
- `s5_blind*.py`, `s5_runs.py`;
- `s6_*.py`.

**Running them.** Run them with `JEPA_ROOT` pointing at a checkout that holds the git-ignored
`data/apple-wide-v1`, `outputs/` and `checkpoints/`. They read label sidecars, the
`observation.state` column and existing reports. No images are decoded, except that
`s2_decompose_v2.py` runs inference with the frozen TASK-050 checkpoint and needs the `3b6af0b` source
tree extracted with `git archive` next to it as `v2tree/`. No simulation or training was run.

**What they are.** These are audit scratch scripts, not project code. They are outside the linted
`src tests scripts` tree, are not imported, and carry no tests. Their inputs are git-ignored, so the
numbers they print are reproducible from committed code but are not committed measurements. **Every
correction below rests on the code reading. The computed numbers size the effect.**

## Corrections applied (errata, not silent edits)

Every affected experiment document keeps its original text. A dated **"Errata — 2026-09-25
(TASK-058)"** block was added near its top, listing each divergence with its audit row ID.
`docs/DECISIONS.md` and `docs/MODELS.md` keep their text and get an appended or inline dated
erratum. `README.md` is a living summary, so its affected sentences were corrected in place,
with a dated note saying so. The earlier wording is in git history. The `.mc` evidence logs of the affected
done tasks got an appended erratum pointer. Everything is visible in the git history of this PR.

| Source document | Rows |
|---|---|
| `docs/experiments/apple_wide_collection_results_v1.md` | S1-03, S1-04, S1-05, S1-06 |
| `docs/experiments/apple_wide_object_ceiling_results_v2.md` | S1-09, S1-19 (tuning figures) |
| `docs/experiments/apple_grasp_closure_diagnosis.md` | S1-09, S1-16, S1-17, S1-18 |
| `docs/experiments/apple_wide_grasp_closure_v3.md` | S1-13 (amendment note to a frozen prereg) |
| `docs/experiments/apple_wide_grasp_closure_results_v3.md` | S1-10, S1-12, S1-13 |
| `docs/experiments/apple_world_model_v2.md` | S2-18, S2-22, S2-24 |
| `docs/experiments/apple_world_model_v2_results.md` | S2-05, S2-06, S2-07, S2-08, S2-09, S2-10, S2-11, S2-13, S2-17, S2-25 |
| `docs/experiments/apple_world_model_v3.md` | S3-06, S3-13 |
| `docs/experiments/apple_world_model_v3_results.md` | S3-04, S3-05, S3-06, S3-07, S3-10, S3-11, S3-12, S3-14, S3-16, S3-17, S3-18 |
| `docs/experiments/apple_world_model_v4.md` | S3-05, S3-07, S4-02, S4-05, S4-12, S4-20 |
| `docs/experiments/apple_world_model_v4_results.md` | S4-02, S4-05, S4-07, S4-09, S4-10, S4-11, S4-12, S4-13 |
| `docs/experiments/apple_policy_v1.md` | S1-13/16/17 (R1), S3-04 (citation location), S3-10, S5-06, S5-10, S5-15 |
| `docs/experiments/apple_policy_v1_results.md` | S5-04, S5-05, S5-07, S5-11, S5-16, S5-17/S6-10, S6-02, S6-04, S6-09, S6-11, S6-12, S6-13, S6-14, S6-16, S6-24, S6-25 |
| `docs/experiments/task056_handover.md` | S6-04, S6-07, S6-10, S6-12, S6-16, S6-17, S6-22, S6-23 |
| `docs/DECISIONS.md` | S2-08, S3-07, S4-09, S4-10, S4-12, S4-17, S4-18, S4-24 |
| `README.md` | L-01, S4-07, S4-09, S4-15, S4-18, S4-24, S4-25 |
| `docs/MODELS.md` | S4-09 |
| `.mc` TASK-049, 050, 051, 052, 054, 055, 056 | pointer to the rows above |

## Not citable: consolidated list

These claims rest only on git-ignored artifacts, scratch code or prose. They are **not citable** as
recorded, whatever their class. Several reproduce exactly; reproducing a figure does not make it
citable.

- **Classed unverifiable:**
  - S1-05, S1-06: collection report tables.
  - S1-12: v3 `close_phase_apple_xy_cm`. The field is committed, but no committed code computes it.
  - S1-18: grasp-closure paired experiment.
  - S2-08: v2 3.13 / 3.26 cm and 89 %. Reproduced by this audit in
    `benchmarks/audits/task058/s2_decompose_lewm_onboard_mps.json`.
  - S2-10: v2 h = 1/4/16 rows.
  - S2-24: v2 pilots.
  - S3-07: v2 anchor of the v2→v3 comparison.
  - S3-13: G6 null pass rates.
  - S3-16: second agent's resampling.
  - S3-17: v3 validation curves.
  - S3-18: v3 other horizons.
  - S6-17: mass-at-max test.
  - S6-22: retracted ejection figures.
  - S6-24: the crashed A3 run and its cross-check.
  - S6-27: train dz mean.
- **Other classes, but resting on git-ignored or scratch sources:**
  - S1-09, S1-10 (v2 side), S1-16, S1-17: ceiling traces and TASK-051 forensics.
  - S1-19: tuning 15/16.
  - S2-11: other-arm h = 1 figures.
  - S2-13: the 190 G6 rows.
  - S3-14: sibling start-label mismatch.
  - S4-21: the 170-tensor weight identity. It rests on checkpoints; only their hashes are
    committed.
  - S4-23: trade-off Spearman. Prose only, but its inputs are committed.
  - S4-26: runner-log eval times.
  - S5-17 / S6-10: train |dz| = 0.4 rate.
  - S6-11, S6-15: §14.1 policy |dz| and `droll` tables.
  - S6-16: `droll` ~27 % and |grasp| 91.78 %.
- **Also not citable:** every number computed by this audit's own scripts. They are
  reproducible from committed code, but their inputs are git-ignored.

## Not done here, deliberately

- **`apple_policy_diagnostics_v1.md`, its manifest, and the TASK-057 task body.** These are not
  edited, because their owner is amending them. Their divergences are S6-26 (the inherited
  "sits inside the non-saturated group" wording, whose correct form is *bracketed by*:
  `min(non-sat) ≤ median ≤ max(non-sat)`), S6-29 (the wrong expert and the 805 budget, which
  should be `apple_collector_policy` and 745), the Table-E phase names, the §1.1(c)/§1.5
  "near-constant translation command" read off |·| quantiles (only the magnitude is measured), and
  S6-27 (the train mean −0.0012, which has no committed code). The amendment text is in slice 6's
  "Divergences" and "Cross-references" below.
- **Manifests are not edited.** Their prose fields that repeat a corrected claim are listed per row.
  Examples are `apple-policy-v1.json` `dz_under_shoot` and `P3_is_the_cleanest_measurement`, and
  `apple-world-model-v4.json` `control_passed_more_gates_than_every_intervention`. Read them
  through this audit.
- **Source docstrings and comments are not edited.** This covers `object_ceiling_v3.py` L5–6,
  L16–25 and L55–57, and `measure_policy_preflight.py` L85–89 and L405–414. Editing `src/` would
  change the implementation hash that historical checkpoints enforce. The wrong wording in them is
  recorded in S1-13, S1-16, S1-17 and S5-06.
- **CLAUDE.md is not edited.** It repeats the S4-25 and L-01 wordings. It is agent configuration, so
  the change is left to the maintainer.
- **No number was patched.** Where a quantity was misnamed, the name was corrected or the claim
  withdrawn. Where a number was wrong (S2-22 "242" → 240, S4-15, S4-20, S6-23), the erratum gives
  the corrected value next to the original.

---

The six slice audits follow verbatim, with scratch paths rewritten to `benchmarks/audits/task058/`.
Line references are to revision `8306633`, before this audit's errata blocks were inserted.

## TASK-058 audit — slice 1 (TASK-048, 049, 051, 053)

Scope read: `apple_wide_collection_v1.md` / `_results_v1.md`, `apple_wide_object_ceiling_v2.md` /
`_results_v2.md`, `apple_grasp_closure_diagnosis.md`, `apple_wide_grasp_closure_v3.md` /
`_results_v3.md`, the three manifests, `.mc/tasks/done/TASK-04{8,9}*`, `TASK-05{1,3}*`.
Code read: `scripts/collect_apple_wide.py` (only commit `57f3169`), `src/embodied_jepa/task.py`
(`9afff34`, unchanged since), `simulation.py:task_truth`, `scripts/evaluate_apple.py`
(`_object_arm`, `object_ceiling_v2_gate`, attempt loop; unchanged since `ba11756`),
`object_ceiling.py` (CEM, full-state parity), `object_ceiling_v2.py`, `object_ceiling_v3.py`,
`privileged_rollout.py` @ `b2e0789` and `35852c0`.
Artifacts read (git-ignored, main checkout): `data/apple-wide-v1/meta/jepa_manifest.json` +
`labels/*.npz`, `data/apple-wide-v1-work/collection_report.json`,
`outputs/apple-wide-object-ceiling-v2/attempts/*/trace.jsonl`,
`outputs/apple-wide-grasp-closure-v3/attempts/*/trace.jsonl`,
`outputs/task051-scratch/forensics-a/*.json`. Check scripts: `benchmarks/audits/task058/s1_*.py`.
Nothing was simulated or trained.

| ID | Claim (short, quoted number) | Where (file:line, all restatements) | Artifact field | Computing code (file:function:line @commit) | What it measures / population | Entities & baseline check | Class | Note / proposed correction |
|---|---|---|---|---|---|---|---|---|
| S1-01 | Corpus size: "797" episodes (200 root + 597 branch), "205,519" transitions, splits "677/80/40" episodes, 170/20/10 sessions | collection_results_v1.md:37-38, 90-96, 111; TASK-048:26 | `apple-wide-collection-v1.json` results.episodes / transitions / split_episodes | `collect_apple_wide.py:acceptance` L712-760 (`len(rows)`, `sum(length-1)`, `split_counts`) @57f3169 | every stored episode in the sealed dataset; transitions = length−1 per episode | seeds 48000-48199, split seed 48 match plan; not a controller gate | confirmed | — |
| S1-02 | Root full-task successes "115/200"; root grasps "(contact lift ≥ 5 cm) 126/200"; A3 ≥ 80 passed | collection_results_v1.md:3, 54-55, 104; TASK-048:26 | results.outcomes.root_full_success / root_grasp; acceptance.A3 | `acceptance` L733-734 over `outcome()` L413-426, which reads the LAST row of latched stages from `AppleToPlateTask.evaluate` (task.py L86-87: grasp latches when reach ∧ apple rise ≥ 0.05 m ∧ hand contact at the same instant) | 200 roots, stage flags latched over the whole episode | PRIVILEGED scripted collector (stated in doc L6-8, manifest `label`); A3 is a data-quality threshold on the scripted collector, not a learned-control gate | confirmed | "contact lift" is exactly the code's condition; doc already says it is not force closure. |
| S1-03 | "Grasp-phase successes 329", "failures 468 (0.587)"; A4/A5 pass; read as "a dense set of grasp-phase outcomes on both sides of the success boundary: 329 grasps and 468 failures" | collection_results_v1.md:57-58, 105-106, 128-131; TASK-048:26 | results.outcomes.grasp_phase_* | `acceptance` L725-739; `outcome()` L422: failure = attempted ∧ ¬grasp at episode end | all 797 stored episodes (every root reached the pre-grasp frame); failure = "grasp stage never latched before the episode ended", for any reason | Computed from dataset manifest: of the 468 failures, **154 ended `guard_refused`** (136 branches, 18 roots) and **96 of those branches were stopped before the 45-command close finished**. By kind, 44 of the `early_lift` failures are guard stops | mislabelled | Gate verdict unaffected (definition = preregistration). The interpretive sentence counts truncated episodes as grasp "outcomes". Minor. |
| S1-04 | "Branches dropped after grasp: 44" | collection_results_v1.md:60 | results.outcomes.branch_dropped_after_grasp | `outcome()` L416: `grasped.any() and rows["dropped"][-1]`; `dropped` = `simulation.task_truth` L351 `apple[2] < 0.70` (tabletop is at 0.74 m) | branches that latched grasp AND whose apple is **below the tabletop, i.e. off the table** at the final frame | Computed from label sidecars: all 44 end with apple z = 0.0266 m (resting on the floor) 0.7-2.0 m from where they started; 39/44 are `open_during_lift` | mislabelled | Reads as "released after grasp". It counts "grasped, then the apple ended on the floor". Releases that land back on the table are not counted. Physics note: the apple is a free sphere that rolls at a constant ~0.7 cm/command once released (also visible in the v2 traces, S1-09). Default condim 3 gives it no rolling resistance (reasoned from the MJCF in `simulation.py`: no `condim` set). |
| S1-05 | A6 "min 0.063 (+, dim 9) and 0.095 (−, dim 11)"; A7 "0.874"; A8 "499/499"; right-arm std "0.151…0.118", grasp std 0.865, "34.7%" interior; left arm std 0 | collection_results_v1.md:107-109, 116-121 | manifest holds only booleans A6-A8. Values only in `data/apple-wide-v1-work/collection_report.json` acceptance.action_coverage / branch_pair_rgb_distinct_at_16 | `acceptance` L815-834 @57f3169 | all transitions of all 797 episodes (applied actions); A8 = sibling branch pairs, onboard RGB at step 16 | Values match the report exactly. A7/A8 are near-trivially true by construction, and the protocol says so (L240-245) | unverifiable | **Not citable**: the numbers exist only in a git-ignored report. Code and definition agree. |
| S1-06 | Descriptive tables: by level "41/50 … 15/50", by kind "73/118 … 39/120", by split "99/170 …", terminations "115/457/160/65", first missing stage | collection_results_v1.md:62-96 | terminations in manifest; the level/kind/failure-stage tables only in collection_report.json | `acceptance` L741-760 | roots per noise level; branches per kind | descriptive, never gating (protocol L250-252) | unverifiable | Not citable except the terminations. The level and kind values match the report. The per-split root/branch rows were not re-derived. |
| S1-07 | v2 primary gate **failed**: success "5/8" (≥6), grasp "5/8" (≥7), `ceiling_inadequate_task_feasible`, conclusive; scripted "8/8"; demo_replay grasp "4/8" | object_ceiling_results_v2.md:10-19, 87-88, 109-122; TASK-049:31; TASK-051:5; diagnosis.md:13-15 | `apple-wide-object-ceiling-v2.json` result.primary | `evaluate_apple.py:object_ceiling_v2_gate` L1776-1893 over `_object_arm` L1593-1671 (latched scorer stages of the final `score`; counted = completed ∧ not runtime_error/deadline/timeout ∧ provenance) @6bb7e0c | 8 fresh primary resets 45100-45107, privileged_object arm | privileged ceiling, labelled NON-LEARNED everywhere. Blind baselines: demo_replay 4/8 grasp, hold would be 0 (reasoned); scripted passes (it is the feasibility reference) | confirmed | — |
| S1-08 | "rollouts exact": v2 "0 mismatches in 2,169 robot-state and 2,169 full-state"; v3 "2,378 + 2,378", "6,935 + 6,935" | results_v2.md:15-17, 113-116; results_v3.md:14-16, 162-166, 298-301; TASK-049:31; TASK-051:19 | manifests run.primary_parity_checks, run.all_ceiling_parity_checks, result.primary.rollouts_exact | `privileged_rollout.encode` L154-183 @b2e0789 (robot qpos/qvel) + `object_ceiling.encode` L187-199 (full `qpos,qvel` incl. apple); summed in `evaluate_apple` attempt loop L2280-2292; non-vacuity in `_object_arm` L1648-1661 | one check per executed command after the first: does the live next state equal *some* feasible candidate's simulated **first** step bit-exactly (min over candidates) | — | confirmed | Accurate as "one-step twin/live parity". Horizon steps 2-6 are not checked. They are deterministic, so this is fine, but "rollouts exact" means first-step parity. |
| S1-09 | v2 failure mechanism: "the fingers pushed the apple out of the hand: 8–9 commands into the close the apple had moved 0.8–3.1 cm, peaking at 2.7–3.5 cm … and it then rolled off the table"; "All four ceiling failures are one mode: the closing fingers eject the apple 8-9 commands into the close" | results_v2.md:132-138, 172-174; TASK-049:33; TASK-051:5; diagnosis.md:15-17; object_ceiling_v3.py:5-6 | none committed (post-hoc; per-attempt traces git-ignored) | ad-hoc, no committed code. Re-derived here from `outputs/apple-wide-object-ceiling-v2/attempts/*-privileged_object/trace.jsonl` (`live_apple`, `live_contact` on `result` events) | the 4 failing v2 attempts (45100, 45103, 45105 primary; 45000 secondary) | Trace (computed): the 2.7-3.5 cm push at close cmds 7-10 is **transient**. By cmd 13-15 the apple is back within 0.0-1.0 cm with `live_contact` True. Contact is lost only at close cmd ~20-27 (45100, 45105, 45000), after which the apple rolls away at ~0.7 cm/cmd. **On 45103 the apple stayed within 0.2 cm and in contact for the whole close; contact was lost at lift cmd ~4-6.** The same transient occurs on the *successful* 45002 (1.9 cm) and 45005 (1.6 cm) | mislabelled | The "8-9 commands in" numbers describe a transient that the hand recovered from. The loss of the apple happens 10-40 commands later. Early contact may still predict failure, but "pushed out of the hand at 8-9" does not describe these traces. Not citable either way. |
| S1-10 | v2's gated failures "2.70–19.10 cm" on the "close apple xy" measure; "six" of v2's 16 gated closes ≥ 1.5 cm "ejected"; "the threshold this project used to call an ejection is 1.5 cm" | results_v3.md:26-31, 177-187, 375-386; TASK-051:20, 24; apple_policy_v1.md:201-202 | none for v2 (the v3 manifest has only v3's own close_phase_apple_xy_cm) | no committed code (verifier's own). Recomputed here as max ‖apple_xy − apple_xy@close0‖ over the 45 close commands: 45103 2.70, 45105 10.90, 45000 11.04, 45100 19.10; 45002 1.92, 45005 1.61. Values reproduce | max over the close window. It pools the transient push with the later roll-away | 19.10/11.04/10.90 are distance rolled after contact loss at cmd 20-27. 2.70 (45103) is the transient only, and that apple was lost in the lift. 45002/45005 "ejected" yet succeeded | mislabelled | The numbers are right. "Ejection" is a max-displacement statistic, and the max cannot tell a recovered transient from a lost apple (S1-09, S1-16). Ranking "how badly v2 failed" by it is unmeasured. |
| S1-11 | v3 primary gate **passed**: "8/8" success (≥6), "8/8" grasp (≥7), `ceiling_adequate`; secondary 16/16; demo_replay "2/8"; replay_separated true | results_v3.md:10-17, 35-41, 128-129, 158-170; TASK-051:19, 21 | `apple-wide-grasp-closure-v3.json` result.primary / secondary | `object_ceiling_v2_gate(version=3)` L1776-1893 @ba11756 (script unchanged since `36025a9`, before prereg) | 8 fresh primary resets 45200-45207, privileged_object v3 | privileged, NON-LEARNED, stated. Baselines: demo_replay 2/8 grasp, hold 0 (reasoned). The scripted collector also passes 8/8, so the gate separates the ceiling from replay but not from the scripted collector. That is by design | confirmed | — |
| S1-12 | v3 close apple xy "0.49–1.35 cm", "0" ejections over 24 attempts | results_v3.md:26-27, 117-150, 177-179, 325-326; TASK-051:20 | v3 manifest result.close_phase_apple_xy_cm {min 0.49, max 1.35, ejections_at_1_5cm 0} | **no committed computing code** (the field was written by hand from the verifier's own script) | max apple xy displacement over each of the 24 v3 closes | recomputed here from traces: 0.49-1.35, per-reset values match the table | unverifiable | Correct as recomputed, but the committed field has no committed function. Same measure caveat as S1-10 (harmless here: all 24 succeeded). |
| S1-13 | "The close phase needs **no** prediction: v3's closure is a fixed schedule in the commanded palm delta and the grasp command"; protocol: "fixed schedule in proprioception", table row "Nothing / close / the closure is an open-loop proprioceptive primitive"; "The closure needs no prediction of any kind" | results_v3.md:230-234; grasp_closure_v3.md:208-211, 225; object_ceiling_v3.py:55-57; TASK-051:27 | none: the claim is about the controller | `object_ceiling_v3._bounds` L104-112: lateral/rotational pinned, **vertical planned by CEM in [−0.5, 0] every close command**; the cost is `object_ceiling_v2.phase_costs` L374-381: ‖palm − grasp target‖ (target from true apple pose) + drift guard + **disturbance of the rollout-predicted apple xy**, scored on exact MuJoCo rollouts; CEM `object_ceiling.py` L499-537 | the executed v3 close commands | Trace (computed, all 24 v3 attempts): applied vertical command per close: mean −0.110 to −0.166, **std 0.165-0.207, range 0 to −0.5, only 3-8 of 45 at the bound**. The command is not constant or scheduled, and it is chosen through exact-rollout prediction on a privileged cost. The project's own diagnosis contradicts "fixed schedule": every fixed-schedule closure failed on every run (`press`, `press_half`, `sched06/12/12f/20`; diagnosis.md:139-144, 171-180, "A fixed press is brittle"), and the v3 docstring itself (L37-40) says a fixed press trips the guard | **wrong** | v3's closure is a rollout-planned vertical command under a privileged cost, with lateral and rotation pinned. It is not a fixed schedule. No run tested a prediction-free closure from the ceiling's pre-grasp state. The scripted collector's close is a proprioceptive position servo (`scripted.py` L53) and holds 16/16, but that is a different controller from a different entry state. The inference "T4 needs no close head" does not follow from this run. |
| S1-14 | "8/8 is consistent with p above roughly 0.69 (one-sided 95%)" | results_v3.md:209-211, 346-347 | — | arithmetic: 0.05^(1/8) = 0.688 | — | — | confirmed | — |
| S1-15 | Placement: "every such release placed the apple: 5/5 primary, 12/12 pooled" (v2); "24/24" (v3); descents "blocked 15–23 commands after 108–117" | results_v2.md:149-157, 168-172; results_v3.md:188-193 | per-attempt phase logs, in manifests attempts_detail and the git-ignored reports | `object_ceiling_v2._advance` L308-358 (release only on `release_predicted`, which rests on the exact simulator probe `release_probe` L136) | the successful ceiling attempts | privileged: the release predictor runs the true simulator, and the docs say so (results_v3.md:233-234) | confirmed | "placed" = scorer success after a release_predicted hand-over (grasp and success sets coincide in the tables). |
| S1-16 | Diagnosis Finding 2: "first contact ≤ 8 → 2.45, 3.12, 7.79, 9.12 cm, all four ejected; ≥ 10 → all twelve held … The separation is complete"; mechanism "the apple is pushed away … leaving the hand before it shuts"; "sweeps it out of the hand" | diagnosis.md:63-83, 184-191; object_ceiling_v3.py:21-25; TASK-051:32; apple_policy_v1.md:196-200 | none committed; scratch `outputs/task051-scratch/forensics-a/*.json` (git-ignored); harness `probe.py` not in repo | scratch harness, not in repo. Re-derived here from forensics rows (`apple`, `contact`, `phase_commands`) | 16 v2 closes on TRAIN tuning resets 49100-49115 | Forensics (computed): 49103 (contact c8, 2.45 cm, counted "ejected") re-centred to ≤ 0.2 cm by c13 and **held through close and lift** (contact throughout; the doc's own parenthesis concedes it latched grasp). On 49112/49114/49115 the apple returned to 0.0-0.3 cm **in contact** by c13-c15. Contact was lost at c29-30 (49112, 49114) and lift l5 (49115), 20+ commands after the transient | mislabelled | In outcome terms early contact → 3/4 lost, 0/12 lost among late contact. The label "ejected" (max ≥ 1.5 cm) is 4/4 only by construction. "Leaving the hand before it shuts" is contradicted by the same data: the apple is re-caged and lost much later. Not citable (scratch). |
| S1-17 | Finding 3: lateral mean \|cmd\| "0.219–0.313" is "statistically indistinguishable" from "0.228" for "pure clipped proposal noise (σ 0.3, ±0.5)", so "the noise is steering" | diagnosis.md:97-115; object_ceiling_v3.py:16-20; apple_policy_v1.md:197-199 | none committed (scratch) | baseline 0.228 = E\|clip(N(0,0.3),±0.5)\| (checked analytically ≈ 0.228). The CEM (`object_ceiling.py` L499-537, warm start L606): the proposal mean is the **previous best sequence shifted**, round 2 is σ 0.15 around round-1's best, and candidates 0/1 are hold/mean | 11 ramp commands × 16 v2 closes | Under a flat cost the executed command is a clipped **random walk** from the warm start, not a zero-mean σ 0.3 draw, so 0.228 is not the CEM's null. No test was run ("statistically" has no statistic) | mislabelled | Unmeasured, not refuted: the comparison uses the wrong null (reasoned, not computed). The flat-cost argument (0.078 cm per 1 cm lateral) and the lateral-drift-rate split (0.078-0.204 vs 0.004-0.074 cm/cmd) are the actual support, and both are not citable. |
| S1-18 | Finding 4 paired experiment: `cage` "70/72" over 24 seeds, `v2` "49/53", matched "47/48 vs 44/48", bound 0.12 "0/28", 0.25 "5/24" | diagnosis.md:133-170, 213-215; object_ceiling_v3.py:35-36; results_v3.md:212-214, 357-359 | none committed; scan logs git-ignored; `fork.py` not in repo | scratch harness, not in repo | replays from snapshotted close-entry states, TRAIN tuning seeds; "grasped" = scorer grasp within 200 commands | design screen, stated as such (diagnosis.md:225-240) | unverifiable | Not citable (code and data are scratch). The post-run verifier re-derived the counts from the logs, but that is not a committed artifact. |
| S1-19 | v2 secondary "7/8 grasp, 7/8 success" vs v1 "5/8, 3/8"; v2 tuning "15/16", "1/16" ejection on tuning | results_v2.md:27-33, 103, 173-174; TASK-049:24, 32 | secondary: manifest result.secondary + task_047_v1_secondary_reference; tuning: prose only | `object_ceiling_v2_gate` secondary_arms L1854-1857 | 45000-45007, non-independent (stated) | — | confirmed | Secondary counts are confirmed. The tuning figures (15/16, 1/16) are prose only and not citable. |
| S1-20 | TASK-053: "No experimental behaviour, result, manifest or frozen value changes: in production nothing replaces that attribute, so the twin is the same class built from the same base" | TASK-053:19-21, 41-43 | — | `privileged_rollout._blind_simulation_class` L52 @35852c0 vs module-level class @b2e0789 | — | — | confirmed | Reasoned from the code: only a monkeypatched `MuJoCoSimulation` changes the base. |

### Divergences needing correction

**S1-13 (wrong, strong: code + all 24 traces + the project's own diagnosis).**
- `docs/experiments/apple_wide_grasp_closure_results_v3.md:230-234`: "The close phase needs
  **no** prediction: v3's closure is a fixed schedule in the commanded palm delta and the grasp
  command, so the learned controller has to decide *when* to start the close and whether it
  succeeded, not how to steer during it."
  → Erratum: "v3's closure is **not** prediction-free. The lateral and rotational commands are
  pinned to zero. The vertical command is chosen on every close command by the CEM, over exact
  MuJoCo rollouts, under a cost that reads the true apple pose (palm-to-grasp-point distance and
  predicted apple displacement). In the 24 recorded closes it varied between 0 and −0.5 (std
  0.17–0.21). Every fixed-schedule closure in the diagnosis failed. Whether a prediction-free
  closure works from this entry state is untested."
- `docs/experiments/apple_wide_grasp_closure_v3.md:208-211` ("the close phase no longer needs any
  prediction at all, because v3's closure is a fixed schedule in proprioception") and `:225`
  ("**Nothing** | **close** | the closure is an open-loop proprioceptive primitive").
  → Amendment note (frozen prereg): the close row should read "vertical palm command: the
  rollout-predicted palm-to-grasp-point height and apple disturbance (lateral and rotation
  pinned)".
- `src/embodied_jepa/object_ceiling_v3.py:55-57` docstring: same correction.
- `.mc/tasks/done/TASK-051-…md:27` "The close phase needs no prediction at all." → withdraw.

**S1-09 (mislabelled, medium: git-ignored traces, 4 attempts).**
- `apple_wide_object_ceiling_results_v2.md:132-138`: "the fingers pushed the apple out of the
  hand: 8–9 commands into the close the apple had moved 0.8–3.1 cm in xy, peaking at 2.7–3.5 cm
  one or two commands later, and it then rolled off the table".
  → "8–9 commands into the close the apple was pushed 2.7–3.5 cm, then returned to within
  1 cm while still in hand contact. The hand lost contact later: at close commands ~20–27 on
  45100, 45105 and 45000, and in the first lift commands on 45103, where the apple stayed
  centred through the whole close. The apple then rolled off the table. The same transient push
  (1.6–1.9 cm) occurred on the successful 45002 and 45005."
- The same line in `.mc/tasks/done/TASK-049…md:33` and `apple_grasp_closure_diagnosis.md:15-17`
  needs the same qualifier.

**S1-10 (mislabelled).** `apple_wide_grasp_closure_results_v3.md:28-29` and `:180-187`
("v2's four gated failures were **2.70–19.10 cm** on the same measure"; "six closes reached
1.5 cm").
→ Add: "The measure is the largest apple displacement over the whole 45-command close. For
45100, 45105 and 45000 it is dominated by the apple rolling away after contact was lost
(10.9–19.1 cm). For 45103 it is the transient push only (2.70 cm); that apple was lost during
the lift. The measure does not separate a recovered push from a lost apple."

**S1-16 (mislabelled, medium: git-ignored scratch forensics).**
`apple_grasp_closure_diagnosis.md:67-73` ("all four ejected … The separation is complete") and
`:81-83` ("leaving the hand before it shuts").
→ "Early first contact (≤ 8) coincided with the ≥ 1.5 cm displacement label on 4/4 and with
eventual loss of the apple on 3/4. 49103 held and latched grasp. In the three lost cases the
apple was re-centred (≤ 0.3 cm) and in contact by close command 13–15, and was lost at close
command 29–30 or lift command 5, not before the hand shut." The same wording appears in
`object_ceiling_v3.py:21-25` and `TASK-051:32`.

**S1-17 (mislabelled / unmeasured; reasoned, not computed).** `apple_grasp_closure_diagnosis.md:105-107`
("statistically indistinguishable from undirected proposal noise").
→ "comparable to the mean absolute value of a single zero-mean σ 0.3 draw (0.228). That is not
the CEM's null under a flat cost: the proposal is warm-started from the previous plan, so it
performs a clipped random walk. No test was run. The flat-cost argument and the lateral drift
rate carry the claim." The same wording appears in `object_ceiling_v3.py:16-20`.

**S1-04 (mislabelled, strong: label sidecars).** `apple_wide_collection_results_v1.md:60`
"Branches dropped after grasp | 44".
→ "Branches that grasped and ended with the apple off the table (below 0.70 m; all 44 on the
floor) | 44".

**S1-03 (mislabelled, minor).** `apple_wide_collection_results_v1.md:128-131` ("grasp-phase
outcomes on both sides of the success boundary: 329 grasps and 468 failures").
→ Add: "154 of the 468 failures are episodes that the velocity guard stopped (96 branches
before the close finished), not completed failed grasps."

**Not citable (unverifiable) rows: S1-05, S1-06, S1-12, S1-18.**
- These list only in the audit's not-citable table; no prose change is needed.
- S1-12 needs its computing function committed, or a note that the manifest field was computed
  outside the repository.

### Cross-references (TASK-057 / behaviour-cloning line)

- **`docs/experiments/apple_policy_v1.md:196-203` (§2.3 R1)** restates S1-16 and S1-17 as
  established:
  - "Grasp-closure v3 **proved** the task is decided in the close phase"
  - "statistically indistinguishable … (0.219–0.313 against 0.228)"
  - "the palm walks off the apple, and the still-open thumb strikes it first"
  - "It is **not** a 'closed too early' mechanism"

  The traces show the apple is re-caged after the early strike and lost 10–40 commands later.
  The noise comparison uses the wrong null. v3 passing its gate shows that the caged closure
  works. It does not prove the mechanism. The BC close-phase risk (R1) and any TASK-057
  diagnostic that assumes "lateral drift ejects at contact" should cite this row rather than
  the diagnosis.
- **S1-13 affects the BC line directly.**
  - The claim "the close needs no prediction / is an open-loop proprioceptive primitive"
    described the privileged ceiling, whose close is rollout-planned.
  - The only closure in the record that actually works without prediction is the scripted
    collector's proprioceptive servo (`scripted.py` L53), which is what BC clones.
  - Its success comes from its own entry state (first contact at command 11 on 16/16). It says
    nothing about closing from a learned policy's entry state.
  - Any TASK-057 reasoning that treats the close as a solved primitive inherits S1-13.
- **`apple_policy_v1.md:168`** calls all 597 branches "deliberately corrupted", but the 118
  `noise_only` branches have no modification: their base action is the unmodified script
  (plus the root's aim offset on 24 of them). Reasoned from `collect_apple_wide.py` branch
  kinds; not computed. Minor, and outside this slice.
- **S1-04 / S1-09 physics.** A released or pushed apple rolls at a constant ~0.7 cm per command
  until it falls off the table. That turns every lost grasp in this simulator into `dropped` /
  `object_dropped`. Keep it in mind when reading drop counts on the BC line. The cause
  (default condim 3, so no rolling resistance) is reasoned, not verified.

## TASK-058 audit — slice 2: TASK-050, world model v2

Corpus: `docs/experiments/apple_world_model_v2.md` (protocol, "P"), `docs/experiments/apple_world_model_v2_results.md`
("R"), `benchmarks/manifests/apple-world-model-v2.json` ("M"), `.mc/tasks/done/TASK-050-*.md` ("T").
Code audited at the producing revision `3b6af0b` (`src/embodied_jepa/world_model_v2.py`, "WM2"; `models/base.py`,
"BASE"). `3b6af0b`→`acf52a0` changed WM2 only by a guard and a chunk pass-through (no metric change); `9fa3d9f`
(arms 2 and 3) is a `.mc`-only child of `3b6af0b` (verified: ancestor, and the diff touches only `.mc`).

Audit computations (all read-only, all in `benchmarks/audits/task058/`):
- `s2_baselines*.py` — label-only blind baselines on val sidecar labels, reimplementing the `window_metrics` cohorts
  (stride-4, horizon-16 windows). Reproduces the published cohort counts exactly (4,807 windows; valid 3,961;
  moving 1,462; true displacement 2.488 cm; 190 G6 rows; train 175,451 obs / 164,822 h16 windows; 32,961 dropped frames).
- `s2_decompose_v2.py` → `s2_decompose_lewm_onboard_mps.json` — inference only, with the `3b6af0b` source tree
  (`git archive`) and the frozen LeWM/onboard checkpoint on MPS. **Reproduces G1 (0.036453066 m) and the persistence
  median (0.043650571 m) bit-for-bit**, so the cohort and model are the gate's own. Adds the encoder/rollout split and
  the model's own no-prediction ("persistence") readout for G3–G6.

Key: "own persistence" = readout of the *directly encoded start* latent scored against the *target* truth — exactly
WM2:358's G2a control, extended to the other gates. "Truth persistence" = copy the true start value (oracle, not
available to a model). "Train prior" = constant fit on train labels only.

| ID | Claim (short, quoted number) | Where (file:line, all restatements) | Artifact field | Computing code (file:function:line @commit) | What it measures / population | Entities & baseline check | Class | Note / proposed correction |
|---|---|---|---|---|---|---|---|---|
| S2-01 | "All three arms FAIL"; "TASK-051 does not start the closed loop" | R:3-4, R:57, R:132-134; P:8-9; T:73-77; M `overall.any_arm_passed=false` | M `results.*.gates.all_passed` | WM2:`evaluate_gates`:517-586 @3b6af0b (thresholds from M `frozen.gates`) | AND of 12 gate checks, h=8 (G7 h=16), val | Arms/checkpoints/revisions match M provenance. Failing gates G1/G2a/G6 are not blind-passable in the failing direction (see rows below). | confirmed | — |
| S2-02 | G1 palm–apple moving h=8: "3.65 / 4.08 / 5.84 cm FAIL" (≤1.5 cm) | R:45, R:64, R:71, R:179-180; T:73-74; restated v3/v4 docs, DECISIONS | M `gates.G1_palm_apple_moving_h8.value`, `key_metrics_h8.palm_apple_moving_median_m` | WM2:`window_metrics`:357,382 via `predicted_readouts`:296-304 (encode→**predict**→readout) | Median ‖pred−true‖ of palm_minus_apple at h=8, **rollout** readout, val, "moving" = valid & true displacement ≥1 cm (1,462 of 3,961 valid) | Label "palm–apple error, moving windows" correct and the R table header "predicted" is correct. Blind baselines fail 1.5 cm: train-mean constant 10.43 cm; truth persistence 2.49 cm (≥1 cm per window by the cohort definition). Recomputed bit-for-bit. | confirmed | — |
| S2-03 | G2a ÷ persistence: "0.835 / 0.858 / 0.844 FAIL"; "model barely beats its own persistence readout" | R:46, R:85-86, R:146-151, R:180; T:74-75 ("0.84 / 0.86 / 0.84") | M `gates.G2a_vs_persistence_h8.value`; `palm_apple_moving_persistence_median_m` 4.37/4.76/6.92 cm | WM2:`window_metrics`:339,358,383; `evaluate_gates`:532-535 | Ratio of medians: rollout median ÷ median ‖readout(encode(start)) − true(target)‖, moving cohort | "Own persistence readout" is the right name (R:82-83 says so). The control carries the start-frame encoding error (3.13 cm, S2-08), so it is not truth persistence (2.49 cm) — R:78-83 discloses this. A model ignoring actions returns ratio ≈1 → fails. | confirmed | — |
| S2-04 | G2b ÷ shuffled: "0.699 / 0.684 / 0.614 PASS"; "beats its shuffled-action control by a wide margin" | R:47, R:84-85, R:147-148; T:76 | M `gates.G2b_vs_shuffled_actions_h8.value` | WM2:`shuffled_pairing`:312-326; `window_metrics`:337,384-386 | Rollout median ÷ rollout-with-another-episode's-actions median, moving cohort | Action-ignoring model gives 1.0 → fails; not blind-passable. Note: shuffled (5.22) and zero-action (5.31) are both *worse than the model's own no-prediction persistence* (4.37 cm, LeWM), so G2b partly measures that wrong actions damage the rollout, not only that right actions help; R's "not enough" reading (R:84-86) already carries this. | confirmed | — |
| S2-05 | G3 apple–plate valid h=8: "1.93 cm PASS" (LeWM/onboard), 2.24 / 2.16 FAIL | R:48, R:135-136, R:123; T (omitted) | M `gates.G3_apple_plate_h8.value` | WM2:`window_metrics`:392 (rollout, all valid windows) | Median ‖pred−true‖ apple_minus_plate at h=8 over **all** 3,961 valid windows, ~63 % non-moving | **Passable without prediction:** LeWM/onboard's own no-prediction readout scores **1.55 cm** (encoded target 1.20 cm) against the rollout's 1.93 cm (computed, `s2_decompose…json`). Truth persistence 0.74 cm; train-mean constant 7.24 cm (fails). The PASS therefore certifies static apple/plate localisation, and the 8-step rollout makes it *worse*; it is not evidence of predicting the offset. | mislabelled | Gate value/verdict right; what a PASS shows is unmeasured by this gate. See Divergences. |
| S2-06 | G4 apple height, grasp cohort: "0.24 / 0.16 / 0.16 cm PASS"; hand crop "slightly better on apple height" | R:49, R:121-122; T:76 | M `gates.G4_apple_height_grasp_h8.value` | WM2:`window_metrics`:360,394 | Median \|pred−true\| apple_height at h=8, target phase close/lift, valid (2,398 windows) | **Blind-passable, and the models lose to a constant.** 59.8 % of the cohort's targets are an apple at rest (\|h+3.4 mm\|<1 cm), only 36.7 % are lifted ≥2 cm, so the median is bracketed by the resting rows. Train-median constant: **0.047 cm**; constant −3.4 mm: 0.044 cm (computed on val labels); LeWM own no-prediction readout 0.22 cm < its rollout 0.24 cm. The statistic cannot measure height prediction on lifted rows. | mislabelled | PASS is vacuous; the arm comparison "hand crop better on apple height" is a 0.08 cm difference on the resting majority — unmeasured. |
| S2-07 | G5 `apple_held` AUROC lift cohort: "0.9996 / 0.9997 / 0.9995 PASS"; "partly trivial… passed but weak evidence" | R:50, R:108-112; T:76 | M `gates.G5_held_auroc_lift_h8.value` | WM2:`window_metrics`:361,376,397; `auroc`:235-243 | AUROC of predicted held prob at h=8 vs held truth, target phase=lift, valid (879 pos / 866 neg) | LeWM's own no-prediction readout (encoded start): **0.9962** — passes without any prediction (computed). Truth apple height at start: 0.997. R self-qualifies as weak evidence, but "partly" understates it: the gate is fully passable by the non-predictive readout. | confirmed | Self-qualified; suggest strengthening "partly trivial" → "passable by the model's own non-predictive readout (0.996)". |
| S2-08 | "Most of the error is in the encoding": encoded start "3.13 cm", encoded target "3.26 cm" vs rollout 3.65; "about 89 % of the rollout's error is already in the encoding"; "(Recomputed independently… agree)" | R:87-94, R:142-143, R:158-159; T:91-93; carried into v3 results :236 ("carried over"), v4 :22/:170 ("not recomputed"), DECISIONS:87,108, README:39, `.mc` TASK-052:29-31, TASK-054:33, TASK-033:69 | **None.** Not in M, not in any `outputs/task050-wm-v2` report; the post-run verifier's recomputation left no artifact | No committed code at the time. Equivalent later code: `scripts/decompose_wm_v3_readout.py:decompose` (median of per-window errors; share = ratio of medians). Audit re-ran it against the v2 checkpoint at `3b6af0b` | encoded start = readout(encode(start)) vs **start** truth; encoded target = readout(encode(target)) vs target truth; moving cohort | **Numbers reproduce** (3.1297 / 3.2607 cm, with G1 matching bit-for-bit), but they are not citable. "89 %" is 3.26/3.65, a **ratio of medians**, not a share of per-window error: on 46.4 % of the moving windows the rollout is *more* accurate than the direct encoding of the target (computed), so rollout error is not "encoding error + a rollout term" window by window. "A perfect predictor could reach 3.26" holds only for a predictor that reproduces the encoded target latent. | unverifiable | Not citable (no artifact); plus mislabelled share. Every downstream "v2 3.13/3.26/0.39/89 %" restatement inherits this. See Divergences. |
| S2-09 | "The models are accurate on still frames… At h=8 the palm–apple error over *all* valid windows is 0.77 cm / 0.86 / 0.85" | R:61-63; R:158-159 ("all-window/moving-window gap (0.8 cm versus 3.6 cm)… readout precision under motion"); restated TASK-052:29, v3 protocol :20, v3 results :287 | M `key_metrics_h8.palm_apple_median_m` | WM2:`window_metrics`:357,380 — **rollout** (encode→predict→readout) | h=8 rollout median over all 3,961 valid windows, of which 36.9 % are moving | Not "still frames": an 8-step prediction over a mixed cohort. Computed LeWM: rollout on non-moving windows 0.39 cm; own no-prediction readout on all valid 0.90 cm; **truth persistence over all valid windows 0.50 cm** (beats every arm). The underlying point (good when nothing moves) survives, but the quoted number is neither a still-frame nor a perception figure. This is the v2 origin of the v3 "0.78 cm perception" defect. | mislabelled | Rename population; note truth persistence 0.50 cm. |
| S2-10 | h-table: h=1/4/16 rows ("2.75/3.35/3.49/2.88/1.30" etc.) | R:67-72 | Only in git-ignored `outputs/task050-wm-v2/leworldmodel-onboard-val-gates.json` (hash-pinned in M `gate_report_sha256`); M holds only h=8 | WM2:`window_metrics`:342-420 | Moving-cohort medians per h, LeWM only | Values match the report exactly (checked). h=8 row confirmed via M. | unverifiable | Not citable for h=1/4/16 (git-ignored; hash-pinned). |
| S2-11 | "(The other two arms behave the same way; … native is uniformly worse and the hand crop is between.)" | R:74-76 | Git-ignored gate reports only | WM2:`window_metrics` | Moving-cohort palm–apple, all h | At **h=1 both other arms are worse than their own persistence readout** (hand crop 3.57 vs 3.45 cm; native 5.40 vs 4.84), while LeWM/onboard beats its (2.75 vs 3.35). "Same way" is false at h=1. The native-worst / crop-between ordering holds on the moving cohort, but not on the all-valid median (native 0.850 < crop 0.863 cm). | mislabelled | Minor; not citable either (git-ignored). |
| S2-12 | "An oracle 'nothing moves' predictor would be better": 2.49 cm vs 3.65 | R:78-83 | M `palm_apple_moving_true_displacement_median_m` 0.024884 | WM2:`window_metrics`:348,390 | Median true displacement on moving windows = truth-persistence error by definition | Correct; recomputed 2.488 cm from labels. | confirmed | — |
| S2-13 | G6 approach-cost calibration: "1.132 / 1.075 / 0.802 FAIL" (≤ ln 1.5); 190 approach windows | R:51, R:101-107, R:121-123, R:186-187; T:75 | M `gates.G6_approach_cost_calibration_h8.value`; 190 only in git-ignored report | WM2:`window_metrics`:362-369,410-411 | Median \|ln(pred cost/true cost)\|, cost = ‖pma − (−1.5,0,13) cm‖, target phase orient/descend, true cost ≥2 cm | FAIL verdict right. But the gate is **passable by a train-prior constant: 0.131** (constant 2.74 cm; val true costs span 2.02–7.67 cm, median 2.62, IQR 2.31–2.85) — computed. LeWM predicts a median cost of 0.85 cm (3× too close); its own no-prediction readout scores 0.81. "Poorly calibrated" understates: every arm is worse than a constant. | confirmed | FAIL correct; record that the gate had no blind-baseline protection and the arms lose to a constant. 190 recomputed from labels (not in M). |
| S2-14 | Approach-cost Spearman "0.16 / 0.16 / 0.06"; "ρ ≈ 0.16 is the number to beat" | R:103-107, R:124, R:162-166 | M `key_metrics_h8.approach_cost_spearman` | WM2:`window_metrics`:417; `spearman`:260-267 | Cross-window rank correlation, rollout cost vs true cost, 190 rows | R correctly scopes it as cross-window, not within-state. True-cost range is narrow (IQR 0.54 cm), so ρ is dominated by small differences; descriptive only. | confirmed | — |
| S2-15 | G7a own<swapped h=16: "0.679 / 0.689 / 0.745" of "106 qualifying ordered pairs"; medians "1.76/1.41/1.38 vs 2.15/1.85/2.14 cm" | R:52, R:95-100, R:136-137; T:75-76 | M `siblings_h16.*` | WM2:`sibling_metrics`:424-503 (own/swap 476-480) | All siblings rolled from the root's encoded branch frame (start-label mismatch 0.0 m), ordered pairs with true divergence ≥1 cm, not dropped at h | Action-ignoring model: pa=pb → own<swap never strictly true → 0.0 → fails. Not blind-passable. | confirmed | — |
| S2-16 | G7b divergence ρ h=16: "0.589 / 0.502 / 0.737 PASS" "over 103 unordered pairs" | R:53, R:95; T:76 | M `siblings_h16.divergence_spearman`, `unordered_pairs` | WM2:`sibling_metrics`:481-483,493 | Spearman of predicted vs true pairwise divergence, unordered pairs (no divergence floor) | Constant/action-ignoring predictor → zero predicted divergence → Spearman None → fail. | confirmed | — |
| S2-17 | G8 "No collapse. Every arm passes G8" (0.000; rank 8.13/8.00/14.74; std 0.907/0.935/1.055); image-only rank "7.60 / 7.19 / **2.19**" | R:54-56, R:113-117; T:76; M | M `collapse.*`, `collapse.image_only.*` | WM2:`collapse_metrics`:506-513; BASE:`_statistics`:39-57 (exp-entropy of centred SV energy), `latent_statistics`:325, `image_embedding_statistics`:335 | Encoded **fused** (image+proprio) latents of every 4th val frame (5,102, dropped frames included) | Formula matches P:271-273. But G8 on the fused latent is passable with a collapsed image encoder (proprio MLP alone supplies variance) — reasoned, and native demonstrates it: image-only rank 2.19 (<4), image std min 0.064. "No collapse" is stronger than G8 supports for native; R:114-117 discloses the image-only number next to it. | mislabelled | Qualify: "no collapse of the fused latent"; native's image pathway alone would fail G8b. |
| S2-18 | Grasp outcome from pre-grasp state, h=64: AUROC "0.846 / 0.819 / 0.911", 63 siblings, 29 positive | R:118-119; P:294-296 ("max predicted apple_held over h = 32…64") | M `grasp_outcome_h64.*` | WM2:`sibling_metrics`:458-462,497-502 | Max predicted held over predicted steps `[32:64]` = **h=33…64**, vs scorer grasp stage label, siblings with ≥64 steps | Descriptive. Off-by-one in the stated window (h=33…64, not 32…64); immaterial. No blind baseline computed. | confirmed | Nit: P:295 "h = 32…64" → "h = 33…64". |
| S2-19 | `apple_dropped` AUROC over all windows "0.976 / 0.947 / 0.931" | R:120 | M `key_metrics_h8.dropped_all_windows_auroc` | WM2:`window_metrics`:407-409 | AUROC of predicted dropped prob at h=8 vs dropped at target, **all** 4,807 windows (846 positive) | Truth persistence (dropped at start) scores **0.966** (computed): two of three arms are below a copy-the-start baseline. Stated as a bare number, no claim leans on it. | confirmed | Optional: add the 0.966 persistence reference. |
| S2-20 | Run table: steps, selected step 13,000/11,000/12,000, selection score "3.63 / 3.89 / 5.24 cm", params, wall clock, CPU, RSS, checkpoint/impl hashes, revisions | R:13-30, R:181-182; T:69-72 | M `selected_step`, `selection_score_m`, `model_parameters`, `elapsed_seconds`, `process_cpu_seconds`, `peak_host_rss_bytes`, `checkpoint_sha256`, `provenance.*` | WM2:`selection_score`:597-617 (rollout moving median, 1,024-window selection cohort), `selectable`:628 | Val selection cohort (not the gate cohort) | Selection replay from git-ignored `metrics.jsonl`: argmin of eligible steps ≥1,000 = 13,000 / 11,000 / 12,000 (step 0 excluded) — confirmed. Gate-report hashes in M match files on disk. | confirmed | — |
| S2-21 | Data: "677 train episodes (175,451 observations, 164,822 horizon-16 windows) and 80 val (20,406)"; "`test_episodes_decoded: 0`… the test split was never opened" | R:33-37; P:43-46; M `test_episodes_decoded` | M | WM2:`load_split`:113-135 (refuses any split but train/val, checks test/holdout overlap); `evaluate`:969 | — | Counts recomputed exactly. `test_episodes_decoded` is a **literal 0** (WM2:745, :969), not a counter; the claim holds by the `load_split` guard, not by measurement. | confirmed | Optional wording: "enforced by the loader; the field is a constant". |
| S2-22 | "32,961 of 175,451 frames (19 %, in 242 episodes) have `apple_dropped`" | P:81-83 | None (prose) | `readout_labels.targets` (dropped = `privileged__apple_dropped`) | Train split, per-frame | Frames reproduce exactly (32,961; 18.8 %). Episodes with any dropped frame: **240** (also 240 dropped at the final frame; 140 distinct roots) — recount from sidecar labels. Method behind "242" unknown. | wrong | Minor, design context only. "242" → "240". |
| S2-23 | Protocol "true persistence already has a median h=8 error of 0.9 cm (train labels)" (basis for "all-window gate would be vacuous") | P:236-239 | None (prose) | — (recomputed) | Train, all windows **including dropped** | Recomputed 0.85 cm (stride-4 h16 windows) / 0.89 cm (stride-1) with dropped windows included; **0.46 cm** over valid windows only. The figure is right for the population it implicitly uses; the vacuity argument is, if anything, stronger. | confirmed | — |
| S2-24 | Pilots: pilot-a "about 25 cm"; pilot-b "7.4–8.3 cm… persistence 7.8–9.2… shuffled 9.1–11.1… AUROC 0.95–1.00… rank 4.2–5.6"; smoke-full rank "12–15 → 2.2–3.2" (basis for the rank floor 2) | P:338-379; T:62-65 | Only git-ignored `outputs/task050-scratch/*/metrics.jsonl` | WM2:`selection_score` (selection cohort) | Selection cohort, steps 500–2,000 | pilot-a/pilot-b values match the git-ignored logs (pilot-b score max 8.2 at 1-dp vs "8.3", rounding). smoke-full not checked. | unverifiable | Not citable (git-ignored). |
| S2-25 | Reading: "the *follow-up* targets readout precision instead, because the measured decomposition shows the error is in the encoding rather than the rollout"; "readout is too imprecise on moving and occluded states" | R:138-151, R:155-161; T:91-94 | None (rests on S2-08) | — | — | Rests on the uncited S2-08 decomposition and on the ratio-of-medians share. "Occluded" is unmeasured: no occlusion variable exists in WM2 or the labels. v3 later reversed the attribution (encoder improved, rollout excess grew; v3 results :246-257) but did not revisit this v2 reading. | mislabelled | Mark the diagnosis as resting on an uncited, ratio-of-medians decomposition; drop "occluded". |

Class counts: confirmed 15 (S2-01, 02, 03, 04, 07, 12, 13, 14, 15, 16, 18, 19, 20, 21, 23); mislabelled 6 (S2-05, 06, 09, 11, 17, 25); wrong 1 (S2-22); unverifiable 3 (S2-08, 10, 24).

### Divergences needing correction

1. **S2-08 (unverifiable + mislabelled share; load-bearing)** — R:87-94: "the readout of a *directly encoded*
   observation is 3.13 cm from the truth when that observation is the window's start frame and 3.26 cm when it is the
   target frame, against 3.65 cm for the 8-step rollout from the start frame. The target-frame figure is what a perfect
   predictor could reach with this encoder and head, so about 89 % of the rollout's error is already in the encoding.
   (Recomputed independently during post-run verification; the numbers agree to the reported digits.)"
   Proposed erratum: "*Erratum (TASK-058):* these figures were never written to a committed or run artifact and are not
   citable as recorded. A TASK-058 re-run with the `3b6af0b` code and the frozen checkpoint (reproducing G1 bit-for-bit)
   gives 3.1297 / 3.2607 cm, so the values stand, but '89 %' is a ratio of two medians (3.26 ÷ 3.65), not a share of
   per-window error: on 46 % of these windows the rollout is closer to the truth than the direct encoding of the target.
   Read it as 'the directly encoded target is almost as far off as the rollout', not as an additive split."
   Same note at T:91-93, and every downstream "v2 3.13 / 3.26 / 0.39 cm / 89 %" row (v3 results :236, v4 :22 and :170,
   DECISIONS:87 and :108, README:39, TASK-052:29-31, TASK-054:33, TASK-033:69) should cite the erratum.
2. **S2-09** — R:61-63: "The models are accurate on still frames and inaccurate exactly where planning needs them. At h
   = 8 the palm–apple error over *all* valid windows is 0.77 cm". Proposed: "…At h = 8 the **8-step rollout's**
   palm–apple error over all valid windows (37 % of them moving) is 0.77 cm (0.39 cm on the non-moving windows) — a
   prediction figure, not a still-frame or perception figure; copying the true start offset scores 0.50 cm on the
   same windows." Same qualifier at R:158-159 ("0.8 cm versus 3.6 cm" are both rollout medians).
3. **S2-05** — R:135-136: "G3 fails on the hand-crop and native arms; the primary LeWM/onboard arm passes it at 1.93
   cm." Add: "The model's own no-prediction readout (encoded start frame) scores 1.55 cm on the same windows, so this
   pass shows static apple–plate localisation, not prediction; the rollout makes it worse."
4. **S2-06** — R:49 / R:121-122 (G4 PASS; "slightly better on apple height"). Add: "G4 is vacuous: 60 % of the grasp
   cohort is an apple at rest, and a constant predictor (train median) scores 0.047 cm, 3–5× better than every arm. The
   arm difference on apple height is not measured by this statistic."
5. **S2-17** — R:113: "**No collapse.** Every arm passes G8." Proposed: "**No collapse of the fused latent.** Every
   arm passes G8, which is measured after proprioception is added and so cannot detect an image-encoder collapse: the
   native arm's image pathway alone has effective rank 2.19, below G8b's 4."
6. **S2-11** — R:74-76: "The other two arms behave the same way". Proposed: "The other two arms are worse throughout,
   and at h = 1 both are worse than their own persistence readout (3.57 vs 3.45 cm; 5.40 vs 4.84 cm)." (Also note the
   h = 1/4/16 rows are only in the git-ignored reports.)
7. **S2-25** — R:149-150: "the readout is too imprecise on moving and occluded states for the ratio to clear 0.8".
   Proposed: drop "and occluded" (never measured) and add "(this rests on the uncited decomposition in 'What the numbers
   say'; see the erratum)". R:141-143 "because the measured decomposition shows the error is in the encoding rather
   than the rollout" → "because a descriptive, uncommitted decomposition suggested…".
8. **S2-22** — P:81-82: "32,961 of 175,451 frames (19 %, in 242 episodes)" → "in 240 episodes" (recount from the
   sidecars; frame count unchanged).
9. **S2-13 (addition, verdict unchanged)** — R:101-102: add "A constant cost equal to the train-split median (2.74 cm)
   would score 0.131 and pass G6; every arm is worse than a constant (the arms predict the palm about 3× too close)."
10. **S2-07 (strengthen self-qualification)** — R:108: "partly trivial" → "passable without prediction: the model's own
    encoded-start readout scores 0.996 on the same cohort".
11. **S2-18 (nit)** — P:295: "over h = 32…64" → "over h = 33…64" (code slices predicted steps 32:64).
12. **S2-10 / S2-24 (not citable)** — R:67-72 (h = 1/4/16 rows) and P:338-379 (pilot numbers) exist only in git-ignored
    artifacts; mark them "not citable" (the gate reports are hash-pinned in M, which does not make them committed).

### Cross-references

- **v3 "0.78 cm perception" defect origin.** The v2 all-valid rollout median (0.77 cm, S2-09) is the quantity that
  v3 protocol :20 and v3 results :287 carry forward. R already mis-describes it as "still frames"; the v3 slice's
  "perception" label is a hardening of this v2 wording.
- **v3/v4/DECISIONS/README comparator.** "3.13 cm at v2" (README:39, DECISIONS:108) is the encoded *start* vs start
  truth, while the v4 figure it is set against ("2.35–2.59 cm", README:37-39) is described as "start to target", mixing
  the start (3.13) and target (3.26) definitions; the v4 slice should check which is being compared. The v2 values
  are numerically sound (reproduced here) but not citable, and the "excess 0.39 cm" / "89 %" are median differences
  and ratios (S2-08).
- **Gate design lesson for TASK-057 / the BC line.** Three TASK-050 gates were passable without the capability they
  name: G3 and G5 by the model's own non-predictive readout, G4 and G6 by a train-prior constant. Any
  world-model-as-critic readout reused in the behaviour-cloning line (held / height / cost) should be gated against
  the model's own encoded-start readout and a train-prior constant, not only against truth persistence. This
  matches the TASK-056/057 pattern that a pooled median over a mixed population (here, 60 % resting apples in G4)
  cannot measure the minority rows the claim is about.
- No entity mismatches (arm, checkpoint, revision, dataset/split/action hash) were found in this slice.

## TASK-058 audit — slice 3: TASK-052 (world model v3)

Corpus: `docs/experiments/apple_world_model_v3.md` (protocol, "P"), `docs/experiments/apple_world_model_v3_results.md`
("R"), `benchmarks/manifests/apple-world-model-v3.json` ("M"), `.mc/tasks/done/TASK-052-*.md` ("T52").
Code at the run revision: `world_model_v2.py` / `world_model_v3.py` are unchanged between `e2f8227` (run
revision) and HEAD (`git diff e2f8227 HEAD` over both is empty). `scripts/decompose_wm_v3_readout.py` and
`scripts/bootstrap_wm_v3_contrasts.py` were added in `7c85c04` and not changed since.

Audit recomputations (read-only, labels + committed/ignored artifacts, no model run) are in
`benchmarks/audits/task058/s3_labels.py` (rebuilds val/train label arrays exactly as `load_split` does; reproduces
20,406 val / 175,451 train observations, 3,961 valid / 1,462 moving windows, true-displacement median
0.02488404791802168 m, and the moving mask in the `-errors.npz` dumps byte-for-byte) and
`benchmarks/audits/task058/s3_analyse.py`.

| ID | Claim (short, quoted number) | Where (file:line, all restatements) | Artifact field | Computing code (file:function:line @commit) | What it measures / population | Entities & baseline check | Class | Note / proposed correction |
|---|---|---|---|---|---|---|---|---|
| S3-01 | G1 "6.62 / 3.65 / 6.11 / 3.30 cm FAIL"; "best arm 3.30 cm, 2.2× the threshold" | R:3-5, R:83, R:116, R:149, R:168, R:281; T52:54; M `overall.best_G1_m` | M `results.<arm>.gates.G1_palm_apple_moving_h8.value`, `key_metrics_h8.palm_apple_moving_median_m` | `world_model_v2.py:window_metrics:353-406` (`palm = error(predicted, …)`, `predicted_readouts` L320 = encode→predict→readout) → `world_model_v3.py:evaluate_gates:223-227` @e2f8227 | 8-step **rollout** readout error, median over the 1,462 moving val windows (valid & true offset displacement ≥ 1 cm), 80 val eps (67 contribute moving windows), stride-4 | Arms/checkpoints match M (`<arm>.pt`, best step). Baselines (computed): oracle true-offset persistence 2.49 cm FAIL; blind train-mean prior 22.1 cm FAIL. Not passable blind. | confirmed | Prose calls it "readout error" and "palm–apple error" — correct for a rollout; no "perception" label in v3. |
| S3-02 | G2a "1.010 / 0.876 / 0.940 / 0.831 FAIL"; "no arm beats its own persistence"; "Arm A is worse than doing nothing (6.62 vs 6.55 cm)"; control-formulation clause "TRIGGERED" | R:5-6, R:84, R:117, R:345-348, R:438-442; T52:54-55, T52:96; M `overall.arms_passing_G2a`, `pre_declared_branch` | M `gates.G2a_vs_persistence_h8`, `key_metrics_h8.palm_apple_moving_persistence_median_m` | `window_metrics:382` (`persistence = ‖readout(encode(start)) − truth(target)‖`), `evaluate_gates:228-232` | ratio of medians: rollout ÷ model's own encoded-start readout held for 8 steps, moving cohort | Self-baselined (persistence is the baseline). Values and clause wording (P:370-377) match. | confirmed | — |
| S3-03 | G2b "0.765/0.666/0.745/0.683 PASS"; zero-action "8.59 / 5.21 / 8.94 / 4.68 cm"; "The actions carry information" | R:85, R:349-351; P:24 (v2 0.699) | M `gates.G2b_…`, `key_metrics_h8.palm_apple_moving_{shuffled,zero_action}_median_m` | `window_metrics:356-362` (`shuffled_pairing` L336: window i gets actions of window (i+N/2) from another episode; zero = all-zero actions) | rollout with wrong / zero actions, same moving cohort | Controls are wrong-action controls: they show action-sensitivity, not accuracy — which is all the prose claims. | confirmed | — |
| S3-04 | "palm–apple, all valid windows 1.26 / **0.78** / 0.87 / 0.92 cm" (v2 "0.77 cm"); "the all-window / moving-window gap … is still there" | R:119, R:287, R:352-356; P:20; T52:29-30. Restated: `apple_world_model_v4_results.md:261,322`; `apple_policy_v1.md:73,780`; `TASK-056…md:45`; `apple-policy-v1.json:1441,1445,1478` | M `results.B_lewm_onboard_only.key_metrics_h8.palm_apple_median_m` = 0.00781842041760683 (M:583) | `window_metrics:404` `_median(palm[valid])`, `palm` from `predicted_readouts` → **8-step rollout** | rollout readout over all 3,961 valid windows; 63.1 % of them have true offset displacement < 1 cm (computed) | Oracle true-offset persistence over the same windows: **0.50 cm** (computed) — beats every arm; blind train-mean prior 22.8 cm. | confirmed (within v3) | v3 text never calls it perception, but never says "rollout" either. The perception misreading (TASK-056 draft) is already corrected at TASK-056:45-46, `apple_policy_v1.md:66-73` and `apple-policy-v1.json:838,1441`. Remaining nit (other slice): those corrections cite `apple_world_model_v4_results.md L261/L322` (prose) as "committed"; the committed JSON field is M:583 — the v4 manifest does not contain 0.0078. Suggest adding "(8-step rollout)" to R:119/R:287/R:353. |
| S3-05 | encoded TARGET "6.31 / 2.59 / 5.45 / 2.47 cm" = "what a *perfect* predictor could achieve"; "A *perfect* predictor on v3's best encoder would still fail G1" | R:234-243, R:261-265, R:433-435; T52:78-80, T52:95-96; M `decomposition.encoded_target_median_m`, `overall.pre_declared_branch`; restated `apple_world_model_v4.md:20-24,38-39` | M `results.<arm>.decomposition_h8.encoded_target_median_m`; per-window `outputs/task052-decomposition-final/<arm>-errors.npz` (ignored) | `scripts/decompose_wm_v3_readout.py:decompose:92-94` @7c85c04 (`readout_rows(targets)` = encode→readout, **no predict**) | direct encoding of the target frame (image + proprio), moving cohort | Construction is genuinely h=0 (confirmed). But it is **not a floor**: per-window, the rollout is closer to truth than the encoded target in **47 / 32 / 49 / 36 %** of moving windows (A/B/C/D, computed from the npz). | mislabelled | "What a perfect predictor could achieve" holds only for a predictor that reproduces the encoded-target latent; it is not a lower bound on G1. The numeric fact (2.47 cm > 1.5 cm) stands. |
| S3-06 | "rollout excess 0.31 / 1.06 / 0.66 / 0.83 cm — the prediction step's own contribution"; "encoder share 95/71/89/75 %"; v2 "about **89 %** of the rollout error is already in the encoding" | R:234-247; P:27-29; T52:31-32, T52:80; v2 results L91-93 | M `decomposition_h8.rollout_excess_median_m`, `encoder_share_of_rollout` | `decompose_wm_v3_readout.py:decompose:116-117` (`rollout_median − encoded_target_median`; `encoded_target_median / rollout_median`) | **difference / ratio of two medians** over the moving cohort, not a per-window attribution; errors are Euclidean norms and not additive | Per-window median of (rollout − encoded_target): **0.11 / 0.48 / 0.05 / 0.38 cm** (computed) — roughly half of the published "excess" for B and D. | mislabelled | "share of error" / "own contribution" read an additive decomposition into a ratio/difference of medians. The distinction was recognised later for v4 (`bootstrap_wm_v4_contrasts.py:171-175`, `apple_world_model_v4.md:336`) but the v3 text was not amended. |
| S3-07 | "The encoder improved 21 % / 24 % (2.59 / 2.47 vs v2's **3.26 cm**)"; "rollout excess went from **0.39 cm** in v2 to 1.06 / 0.83 cm — roughly two to three times"; "the error moved out of the encoder and into the prediction" → branch-4 inference withdrawn, branch 2 primary | R:250-259, R:421-427; T52:77-82, T52:86-93; M `overall.pre_declared_branch`, `decomposition.v2_reference_source`; restated `apple_world_model_v4.md:20-30`, `docs/DECISIONS.md:87` | v3 side: M `decomposition_h8`. v2 side: M `decomposition.encoded_target_median_m.v2_lewm_onboard` = 0.0326, sourced to `apple_world_model_v2_results.md:88-93` prose ("recomputed independently during post-run verification") | v3: `decompose` (above). v2 3.26 / 0.39: **no committed computing code** — the decompose script postdates TASK-050 and cannot load v2 checkpoints (M `v2_reference_source`) | v3 terms as S3-05/06; v2 term unknown construction, stated to be on the same cohort | Arithmetic from the quoted values is right (20.6 %, 24.2 %; 2.7×, 2.1×). The "two-to-three-fold" growth is a difference-of-medians comparison (S3-06); per-window v3 excess is 0.48 / 0.38 cm, and the v2 per-window value is unavailable. | unverifiable | v2 anchor has no committed code or JSON field. The pivot itself does **not** depend on it: branch 2 already fires by the pre-declared ordering on G7a (S3-08). Suggest an erratum that the v2→v3 growth claim is a difference-of-medians comparison against an uncommitted v2 figure. |
| S3-08 | Branch reading: "Branch 2 … triggered for arms A and B, where G7a fails (0.613, 0.698 against ≥ 0.70)"; branch-4 premise for C, D ("best v3 G2a is D's 0.831, a 0.5 % relative change" vs v2 0.835) | R:393-427; T52:86-93; M `overall.pre_declared_branch` | M `gates.G7a_…` (106 qualifying pairs for B), `gates.G2a_…`; v2 G2a `apple-world-model-v2.json` 0.8351 | `world_model_v2.py:sibling_metrics:463-534` (own < swapped over ordered pairs with true divergence ≥ 1 cm, h=16), `evaluate_gates:258-263` | ordering rule applied as P:350-369 states | Blind check (reasoned): an action-blind or persistence model predicts identical own/swapped → strict `<` never holds → fraction 0; not passable. | confirmed | — |
| S3-09 | G3 "2.79 / 1.97 / 2.95 / 1.80 cm" (B, D PASS) | R:86, R:118, R:154, R:170, R:284 | M `gates.G3_apple_plate_h8`, `key_metrics_h8.apple_plate_median_m` | `window_metrics:431` (3-D ‖rollout apple_minus_plate − truth‖, median over valid) | rollout over all 3,961 valid windows | Blind train-median prior 7.30 cm FAIL, train-mean 20.8 cm FAIL; oracle true persistence 0.74 cm would PASS (not blind; no model-persistence control exists for G3). | confirmed | Note: G3 cannot separate readout from dynamics; no v3 sentence claims it does. |
| S3-10 | G4 "0.346 / 0.149 / 0.224 / 0.175 cm PASS" for all arms; counted in "8 / 10 / 10 / 10 of 13"; used in A↔B "worse on nine of the thirteen gates" and A↔C "improved five of the six rows" (G4 0.346 → 0.224) | R:87, R:96, R:126, R:150, R:156; T52:53; M `overall.gates_passed_of_13`; restated `apple_policy_v1.md:75` (0.1485 cm PASS) | M `gates.G4_apple_height_grasp_h8`, `key_metrics_h8.apple_height_grasp_median_abs_m` (2,398 windows) | `window_metrics:384,433` (grasp = valid & phase(target) ∈ {CLOSE, LIFT}; \|rollout apple_height − truth\|) | rollout apple height above rest, grasp cohort (27 % CLOSE, 73 % LIFT at target) | **Passable blind**: constant 0 (apple at its rest height) scores **0.362 cm** on the identical 2,398-window cohort (computed) — passes ≤ 1.0 cm and beats arm A's 0.346 by only 0.016 cm. The untrained step-0 checkpoints score 0.130 / 0.130 / 0.130 / 0.148 cm on the selection cohort (`checkpoints/task052-wm-v3/*.metrics.jsonl`, ignored). | mislabelled | Gate verdict is arithmetically right but carries no evidence of learning; it is reported as a pass and counted in the per-arm totals and one-factor tallies without that disclosure. |
| S3-11 | G5 "≈ 1.0 PASS … but weak evidence"; "shuffled-action AUROC 0.619 / 0.768 / 0.708 / 0.764 … **Most of that AUROC comes from the encoded state and the fused proprioception, not from action-conditioned prediction**" | R:88, R:365-369 | M `gates.G5_…`, `key_metrics_h8.held_lift_shuffled_auroc` | `window_metrics:419-424` (AUROC of rollout `apple_held` at h=8 on lift cohort; shuffled = rollout with another episode's actions) | rollout AUROC, lift cohort 879 pos / 866 neg | Shuffled actions are *wrong* actions, not absent ones, so they cannot isolate the state-only share; above-chance AUROC surviving the shuffle is 24 / 54 / 42 / 53 % (A/B/C/D) — under half for A and C. No h=0 (encoded-start) AUROC is computed in v3. Oracle truth persistence (apple height at start) AUROC **0.997** (computed) — the gate is passable by copy-current-state using truth. | mislabelled | Gate verdict and "weak evidence" confirmed. The attribution sentence is UNMEASURED by the control it cites (direction plausible given the 0.997 oracle, reasoned). |
| S3-12 | "This is the one place where v3 measures something better than v2"; "G6a passes for B and C (0.544, 0.516) **against v2's cross-window ρ of 0.165**"; T52: "the CEM-relevant ranking metric **improved a lot over v2**"; "the models order candidate actions far better than they localise the apple" | R:300-301, R:320-324, R:340-341; T52:70-74; M `overall.one_factor_readings` (n/a), `baselines.v2_lewm_onboard_cross_window_cost_spearman` | M `results.<arm>.ranking_h16.within_state_spearman_pooled`; v2 `apple-world-model-v2.json` `approach_cost_spearman` 0.1647 | v3: `world_model_v3.py:candidate_ranking_metrics:77-141`, `_summarize_ranking:143-185`. v2's 0.165: `window_metrics:437` (`spearman(cost_pred[cost_rows], cost_true[cost_rows])`) | v3: within-group pooled rank correlation over 18 sibling groups at h=16. v2: cross-window Spearman over approach-phase windows with true cost ≥ 2 cm at h=8 — **a different statistic on a different population** | v2 models were never scored on within-state ranking; on v2's own statistic the v3 arms score 0.005 / 0.097 / 0.064 / 0.228 (M), i.e. no better. | mislabelled | G6a values and verdicts confirmed by code. "Improved over v2" is UNMEASURED; the metric is new, not better. "far better than they localise" compares ρ to cm. |
| S3-13 | G6 null: "G6a pass rate 1.0e-4, G6b 5.9 %, pair 1.0e-4 … the G6a+G6b pair is not passable by chance" | P:287-303; R:317-324; M `overall.caveats[3]` | none — prose only; `outputs/task052-scratch/` holds only smoke evals | **no committed code** for the Monte-Carlo | "uniformly random ranker over the real val cohort, 20,000 draws" (stated) | Only a random-ranker null was considered. A model-free ranker that integrates the commanded end-effector deltas from the shared start state was not evaluated (reasoned, not computed): candidates differ only in actions, and the cost is palm–apple offset, which those actions largely determine pre-grasp. | unverifiable | Not citable. "Not passable by chance" ≠ "not passable without a learned model". |
| S3-14 | "`start_state_max_label_mismatch_m` is 0.0 for every arm: the siblings' start states coincide exactly, so the only thing differing between candidates is the actions" | R:331-332; P:255-256 | gate JSON `metrics.siblings.start_state_max_label_mismatch_m` (outputs/, ignored; not in M) | `world_model_v2.py:sibling_metrics:477-478` (max \|palm_minus_apple(start_m) − palm_minus_apple(start_anchor)\|) | one 3-vector label at the sibling start rows | Checks the palm–apple offset only, not the full state; model inputs coincide anyway because every candidate is encoded from the anchor row (`candidate_ranking_metrics:101`). | mislabelled | Minor: "start states coincide exactly" → "the siblings' palm–apple offsets at the start coincide exactly (0.0 m)". Also not citable (not in M). |
| S3-15 | Paired bootstrap: "A − B +2.97 cm [+2.39, +3.46]; C − A −0.51 [−0.96, +0.12]; D − B −0.34 [−0.63, −0.14]"; "least conservative design … lower bound" | R:193-212; M `decomposition.paired_bootstrap_contrasts`, `overall.caveats[6]` | M `decomposition.paired_bootstrap_contrasts.*_rollout` | `scripts/bootstrap_wm_v3_contrasts.py:paired_intervals:53-80` @7c85c04 (iid resample of the 1,462 **windows**, paired, diff of medians, 20,000 × seed 20520) | window-level iid bootstrap for two fixed checkpoints | Doc labels it correctly as understating variance. Audit **episode-clustered** recomputation (67 episodes with moving windows, 5,000 resamples): A − B +2.97 [+1.72, +4.52]; C − A −0.51 [−1.38, +0.40]; D − B −0.34 [−1.13, +0.08]. | confirmed | The adopted conservative reading (only A↔B exceeds sampling noise) is supported. Extra (not claimed in doc): clustered encoded-target C − A = −0.86 cm [−1.55, −0.15] excludes zero — dropping the readout shaping improved the *encoder* term. |
| S3-16 | "A second agent's independent, more conservative resampling … produced intervals wide enough that both C − A and D − B cross zero. That disagreement is unresolved" | R:208-212, R:522-524; M `overall.caveats[6]` | none — no committed code or report | none committed | unknown design | Audit clustered recomputation (S3-15) reproduces the qualitative result. | unverifiable | Not citable as published; the "unresolved" disagreement can now be closed by committing a clustered interval. |
| S3-17 | B↔D "should probably be read as noise … arm D's selection score swings 7.50 → 12.34 → 15.41 → 4.99 cm"; step-budget readings (B 5.35 → 6.96, 4.53 → 4.85; D 3.35 → 3.54; A 5.99–6.73; C 6.27–7.07); "Train B longer is the one live option" | R:180-189, R:373-380 | only `checkpoints/task052-wm-v3/<arm>.metrics.jsonl` (ignored); M has only `selection_score_m`, `selected_step` | `world_model_v2.py:selection_score:628-640` (median rollout error, moving windows of the 1,024-window selection cohort) | per-checkpoint validation curve, selection cohort (not the gate cohort) | All values match the ignored artifact exactly. | unverifiable | Not citable. Also compares selection-cohort, cross-checkpoint jitter with a gate-cohort, between-arm gap (loose, not wrong). |
| S3-18 | Multi-horizon table for B (h = 1 / 4 / 16: 3.10 / 4.25 / 3.17 cm etc.); ranking ρ at h = 8 (0.26/0.36/0.46/0.38) and h = 32; "The planner's own horizon is 8, and no arm reaches ρ ≥ 0.5 there" | R:333-338, R:358-363, R:571-572 | only `outputs/task052-wm-v3/<arm>-val-gates.json` (ignored); M carries h=8 window metrics and h=16 ranking only | `window_metrics` (all horizons), `candidate_ranking_metrics` | as S3-01/S3-12 at other horizons | Values match the ignored artifact exactly. | unverifiable | Not citable. |
| S3-19 | Descriptive: grasp outcome AUROC 0.877/0.822/0.790/0.941 (h = 64, 63 siblings, 29 pos); `apple_dropped` AUROC 0.969–0.975; v2 calibration 1.048/0.906/1.027/0.772; mean per-group ρ; mean regret 3.47/3.30/3.24/3.02 mm; shuffled ρ and regret; random-choice 8.44 mm | R:303-316, R:381-385 | M `grasp_outcome_h64`, `key_metrics_h8.*`, `ranking_h16.*` | `sibling_metrics:486-494` (max rollout `apple_held` over readout indices 32..63 = h 33–64), `_summarize_ranking` | as named | All match M. | confirmed | Trivial: P:337 says "h = 32…64"; code covers h = 33…64. |
| S3-20 | One-factor arithmetic: A↔B "worse on nine of the thirteen gates", "1.7× memory, 1.5× wall clock"; A↔C "five of the six rows"; B↔D "0.34 cm (9 %) … 2.1×"; A − B = 16,384 and D − B = −33,792 params; 21,662 s = 6.02 h | R:24-47, R:109-189; T52:59-66 | M `results.<arm>.*` | arithmetic over M | — | All recompute. The nine-of-13 and five-of-6 tallies include G4 (S3-10); without G4 they are 8/12 and 4/5 — reading unchanged. | confirmed | — |
| S3-21 | v2 comparison: "B reproduces v2's G1 to within 0.03 mm (3.6481 vs 3.6453 cm)", G2a 0.876 vs 0.835; "three of the four v3 arms are worse than v2's 0.165" (cross-window ρ) | R:272-296; T52:67-68 | M `gates`, `key_metrics_h8.approach_cost_spearman`; `apple-world-model-v2.json` L290 (0.036453066), L232 (0.83511), L286 (0.16471) | as S3-01, S3-02, `window_metrics:437` | like-for-like on the v2 statistic | Matches. | confirmed | — |
| S3-22 | G8 "No collapse anywhere"; effective rank 5.12/7.74/5.36/7.54; image-only 4.53/7.19/4.75/7.09 | R:93-95, R:370-372, R:134-138 | M `results.<arm>.collapse` | `collapse_metrics:537-545` (encoded val latents, stride 4) | encoded latents, all val rows | Untrained step-0 encoders also pass G8's thresholds on the selection cohort (rank 13.6–13.9, std 0.100–0.110; ignored metrics.jsonl), so G8 shows non-collapse only — which is all the prose claims. | confirmed | — |
| S3-23 | "A predictor that saw the offset exactly and then assumed it froze would still beat every arm on the moving windows" (true displacement 2.49 cm) | R:352-356; also v2 results L80-81 | M `key_metrics_h8.palm_apple_moving_true_displacement_median_m` 0.024884 | `window_metrics:370,414` | oracle persistence of true offset, moving cohort | Recomputed from labels: 2.488 cm < 3.30 cm (best arm). | confirmed | — |
| S3-24 | "`test_episodes_decoded: 0` … a literal … the real guarantee is structural" | R:64-72, R:471-474 | M `results.<arm>.test_episodes_decoded` | `load_split:143-152` (refuses test/holdout) | — | Self-qualified. | confirmed | — |

### Divergences needing correction

Proposed wording is for an erratum/amendment block; the original sentence stays in history.

1. **S3-05** — `apple_world_model_v3_results.md:242-244`: "**encoded TARGET** is the readout of a *directly encoded*
   target-frame observation: what a *perfect* predictor could achieve with that encoder and head." Also R:262-263
   ("A *perfect* predictor on v3's best encoder would still fail G1"), R:434-435, T52:78-79, T52:95-96,
   `apple_world_model_v4.md:38-39`.
   → "…the readout of a directly encoded target-frame observation (encode → readout, no prediction). It is what a
   predictor that reproduced the encoded target latent exactly would score; it is **not a lower bound** on G1 — the
   rollout is closer to the truth than the encoded target in 32–49 % of moving windows."

2. **S3-06** — R:243-244: "**Rollout excess** is the gate value minus it — the prediction step's own contribution."
   and the "encoder share" column (R:234); P:27-28 "About **89 %** of the rollout error is already in the encoding";
   T52:31-32 "so about 89 % of the rollout error is in the encoder/head rather than the dynamics".
   → "Rollout excess is the difference between two cohort medians (G1 minus the encoded-target median); it is not a
   per-window attribution, and errors are norms that do not add. The median of the per-window difference is
   0.11 / 0.48 / 0.05 / 0.38 cm. 'Encoder share' is the ratio of the two medians, not a share of the error."

3. **S3-07** — R:257-259: "The rollout excess went from 0.39 cm in v2 to 1.06 cm (B) and 0.83 cm (D) — roughly two to
   three times. The error moved out of the encoder and into the prediction." Also R:250-252 ("21 % and 24 % better"),
   R:423-426, T52:80-82, M `overall.pre_declared_branch`.
   → Add: "The v2 terms (3.26 cm, 0.39 cm) are quoted from `apple_world_model_v2_results.md`; no committed code or
   JSON field computes them, so the v2→v3 comparison is not citable. On v3 alone, the growth claim is a
   difference-of-medians comparison. The branch-2 decision does not depend on it: branch 2 fires by the pre-declared
   ordering on G7a."

4. **S3-10** — R:87 (G4 PASS row) and the per-arm totals R:96 / T52:53 / M `overall.gates_passed_of_13`; R:126 and R:156
   tallies; `apple_policy_v1.md:75`.
   → Add under the gate table: "G4 is passable without a model: predicting the apple at its rest height (constant 0)
   scores 0.36 cm on the same 2,398 windows, and the untrained step-0 checkpoints score 0.13–0.15 cm on the selection
   cohort. A G4 pass is not evidence of learned readout or dynamics." (Verdicts unchanged; the frozen threshold stays.)

5. **S3-11** — R:367-369: "Most of that AUROC comes from the encoded state and the fused proprioception, not from
   action-conditioned prediction."
   → "The shuffled-action control cannot apportion the AUROC between state and actions (shuffled actions are wrong,
   not absent), and no h = 0 AUROC was computed. The attribution is unmeasured. For scale, a copy-current-state
   baseline on simulator truth (apple height at the start frame) already reaches AUROC 0.997 on this cohort."

6. **S3-12** — R:300-301 "This is the one place where v3 measures something better than v2"; R:320
   "G6a passes for B and C (0.544, 0.516) against v2's cross-window ρ of 0.165"; T52:70-72 "the CEM-relevant ranking
   metric improved a lot over v2"; R:340 "the models order candidate actions far better than they localise the apple".
   → "G6a is a new statistic (within-state ranking over 18 sibling groups). v2 was never scored on it, so no v2→v3
   improvement is measured. On v2's own cross-window statistic, the v3 arms score 0.005 / 0.097 / 0.064 / 0.228,
   against v2's 0.165."

7. **S3-13** — P:287-297, R:321-324: the null pass rates "1.0e-4 / 5.9 %" and "the G6a+G6b pair is not passable by
   chance".
   → Mark as not citable (no committed code or artifact). Add: "Only a random ranker was tested; a model-free
   action-integration ranker was not."

8. **S3-14** — R:331-332: "the siblings' start states coincide exactly".
   → "the siblings' palm–apple offsets at the start coincide exactly (the check compares that one label)."

9. **S3-16 / S3-17 / S3-18** — not citable: R:208-212 (second agent's resampling), R:180-189 and R:373-380
   (validation curves), R:333-338 and R:358-363 (other horizons). Either commit the underlying numbers (for example
   the per-horizon `windows` block and `metrics.jsonl` validation scores into the manifest, and a clustered bootstrap
   report) or mark them "from git-ignored run artifacts; not citable".

10. **S3-04** (optional clarification) — R:119, R:287, R:353: add "(8-step rollout)" to "palm–apple, all valid windows",
    and note that an oracle frozen-offset predictor scores 0.50 cm on those windows, so the figure is dominated by
    static windows.

### Cross-references

- **TASK-056 / behaviour-cloning line** (`apple_policy_v1.md`, `apple-policy-v1.json`, TASK-056 task body): the 0.78 cm
  rollout-as-perception mislabel is already corrected there (TASK-056:45-46, `apple_policy_v1.md:66-73`,
  `apple-policy-v1.json:838,1441`). Remaining nit: `apple_policy_v1.md:73,780` and `apple-policy-v1.json:1441` give
  `apple_world_model_v4_results.md L261/L322` (prose) as the committed location. The committed field is
  `apple-world-model-v3.json` `results.B_lewm_onboard_only.key_metrics_h8.palm_apple_median_m` (M:583). The v4
  manifest does not contain it.
- **`apple_policy_v1.md:75`** lists G4 "0.1485 cm PASS" in the §1 evidence table. Per S3-10, G4 is passable by a
  constant and by untrained checkpoints, so it is not evidence about the encoder. It is listed there as a rollout
  quantity and not used as load-bearing, but it should carry the same caveat.
- **`apple_policy_v1.md` §1 "the only directly-encoded numbers … 2.3536 / 2.5903 cm"** (also README:38-40,
  DECISIONS:108-109): the construction is confirmed as h = 0 (`decompose:91-94`; the v4 equivalent is cited). But the
  2.59 cm "encoded target" is *not* a floor for a predictor (S3-05). README:38 calls the population "held-out
  validation windows", yet val also drove checkpoint selection (R:557). That belongs to another slice.
- **`docs/DECISIONS.md:87`** and **`apple_world_model_v4.md:20-30`** restate the v2 → v3 → v4 "rollout excess roughly
  tripling (0.39 → 1.06)". That rests on the uncommitted v2 figure and on a difference of medians (S3-06/S3-07). The
  v4 G9 gate reads the same difference of medians. The v4 slice should check whether any v4 verdict reads it as a
  per-window contribution.
- The TASK-054 → behaviour-cloning pivot traces back to the clause pre-declared at R:450-452. That clause is keyed on
  G2a (S3-02, confirmed), not on the decomposition, so the audit findings here do not undermine the pivot's trigger.

## TASK-058 audit — slice 4 (TASK-054, TASK-055, top-level record)

Corpus: `docs/experiments/apple_world_model_v4.md` (protocol, "P"), `apple_world_model_v4_results.md`
("R"), `benchmarks/manifests/apple-world-model-v4.json` ("M"), `.mc/tasks/done/TASK-054*` / `TASK-055*`,
`docs/DECISIONS.md`, `README.md` Status, `docs/MODELS.md`, `docs/EVALUATION.md`; the MVP 0/150 restatement.

Code revision: every v4 evaluation ran at `c9cf9a6`. `git diff c9cf9a6 HEAD` is empty for
`world_model_v2.py`, `world_model_v3.py`, `world_model_v4.py` and `scripts/bootstrap_wm_v4_contrasts.py`,
so the line numbers below (at HEAD) are the lines that produced the numbers.

Computed checks (read-only, labels only, no images, no model): the script is
`benchmarks/audits/task058/s4_baselines.py`. It rebuilds the gate cohorts from `data/apple-wide-v1` val labels
and matches the reports exactly: 4,807 windows, 3,961 valid, 1,462 moving, 67 moving episodes, true-displacement
median 0.024884 m, 2,398 grasp windows, 879/866 lift classes, 18 ranked groups, random-choice regret 8.444 mm.
It then scores blind baselines on those identical cohorts. Per-window estimands come from the git-ignored
`outputs/task054-wm-v4/*-errors.npz`. The weight identity comes from the git-ignored checkpoints, whose SHA-256
hashes are committed in M.

Shared construction facts (read from the function bodies):
- **G1** is `world_model_v2.window_metrics` (v2:353), `palm = error(predicted, …)` (v2:381). `predicted` is
  `predicted_readouts` = encode → **predict** → readout, so G1 is a **rollout**. It is taken over `moving`
  (v2:373): valid windows whose **true** palm–apple displacement over 8 steps is ≥ 1 cm.
- **G2a** is `ratio(moving rollout median, moving persistence median)` (v2 `evaluate_gates`:548, v3:192). Here
  `persistence = ‖readout(encode(START)) − truth(TARGET)‖` (v2:382): the model's **own encoded-start readout**,
  which includes that model's encoder error. It is **not** the true-state copy-last.
- **G9** is `rollout_median − encoded_target_median` (v4 `decompose`:140, gate at v4:203). It is a **difference
  of two medians**, not a median of per-window differences.
- The **"encoded"** quantities are `readout_rows` = encode → readout, with no prediction. The v4 config has
  `state_fusion: true` (`configs/apple_wm_v4.yaml:32`), so every "encoded" readout also sees fused
  proprioception.

| ID | Claim (short, quoted number) | Where (file:line, all restatements) | Artifact field | Computing code (file:function:line @commit) | What it measures / population | Entities & baseline check | Class | Note / proposed correction |
|---|---|---|---|---|---|---|---|---|
| S4-01 | G2a "0.8763 / 0.8814 / 0.8635 / 0.9036" > 0.8 → no arm passes → **Outcome B fires, CEM over this cost abandoned** | R:3-10, R:102, R:378-392; TASK-054:66-71; TASK-055:10-15; DECISIONS:66-84; README:17-21; CLAUDE.md "control line changed" | M `results.*.gates.G2a_vs_persistence_h8.value`, `overall.primary_gate_values`, `overall.outcome_fired = B_no_arm_passes_g2a` | v3 `evaluate_gates`:192 → G2a ratio; numerator v2 `window_metrics`:381, denominator v2:382 @c9cf9a6 | Ratio of the moving-window (1,462 windows, 67 val episodes) median rollout error to the median of the model's own encoded-start readout error, h = 8, val | The arms and checkpoints match M (`checkpoint_sha256`). The outcome rule matches `pre_declared_outcomes.B`. **Blind baselines fail it:** an identity/copy-last rollout gives exactly 1.0, and a train-mean constant gives 10.43 cm / 4.16 cm ≈ 2.5 (computed). | confirmed | The decision rests on the frozen definition and is correct. The denominator is the model's own persistence readout (4.02–4.88 cm), which is much worse than the true-state copy-last (2.49 cm). See S4-02. |
| S4-02 | "beat the persistence baseline" (title); "The persistence baseline is the thing to beat and it has **never been beaten**: G2a is 0.8635 at best"; "every arm still **loses to** its own persistence readout by the required margin"; Outcome A "beats the model's own persistence readout **for the first time**"; H "bring the rollout error **below** the model's own persistence readout (G2a < 0.8)" | TASK-054 title / R:493-495; R:311-312; P:371-372; P:60-61 | same as S4-01 | same as S4-01 | G2a < 1 means the rollout median **is below** the own-persistence median. That holds in **every** v4 arm (0.86–0.90) and every v3 arm except A (1.010). Per window, rollout < own persistence on 58.1 / 55.3 / 55.1 / 58.3 % of moving windows (computed from the error dumps). Against the **true-state** copy-last (2.49 cm), the rollout (3.48–4.41 cm) has indeed never been lower. | The prose names the 0.8-margin gate as "beating persistence". The rollout does beat its own persistence readout, by 10–14 %, just not by the required 20 %. | mislabelled | "Never beaten" is true only for true-state copy-last, a quantity no gate computes. The cited number (G2a) shows the opposite. README:49-51 phrases it correctly ("by the required margin"); copy that wording. |
| S4-03 | G1 "3.65 / 3.66 / 3.48 / 4.41 cm" FAIL ≤ 1.5 cm | R:101, R:171-174, DECISIONS:87-88 | M `results.*.gates.G1_palm_apple_moving_h8.value` | v2 `window_metrics`:381 (`predicted_readouts`) | A **rollout** readout over the moving cohort (true displacement ≥ 1 cm). Val was also used for checkpoint selection (declared at P:218, R:514). | True-state copy-last scores 2.49 cm and train-mean 10.43 cm; both fail (computed). | confirmed | Correctly called a rollout ("rollout (G1)") throughout. |
| S4-04 | G2b "0.666–0.757 PASS"; "every arm still beats its shuffled-action and zero-action controls … zero-action 5.21 / 5.66 / 5.24 / 5.24 cm … The actions carry information" | R:103, R:308-312 | M `gates.G2b_*`; reports `palm_apple_moving_{shuffled,zero_action}_median_m` | v2 `window_metrics`:367-369 (`shuffled_pairing` derangement, zero actions) | Rollout under another episode's actions or under zero actions, same cohort | Copy-last gives 1.0 and fails. Note that the shuffled rollout (5.05–5.82 cm) and the zero-action rollout (5.21–5.66 cm) are both **worse than the model's own persistence** (4.02–4.88 cm). The controls are harder to lose to than "do nothing". | confirmed | The values are correct. "Actions carry information" is supported. G2b is a weaker test than G2a; say so if it is ever cited as evidence of dynamics. |
| S4-05 | G4 apple height, grasp cohort, "0.15 / 0.13 / 0.17 / 0.26 cm PASS" (≤ 1.0 cm); counted in "10 of 14" | R:105; P:233; pilots P:515-519 ("G4 on a 120-step model says nothing") | M `gates.G4_apple_height_grasp_h8.value`; report `apple_height_grasp_median_abs_m` | v2 `window_metrics`:418 (`error(predicted,"apple_height")[grasp]`); height = apple z − rest z (`readout_labels.targets`:65) | Median abs error of the predicted height over 2,398 windows whose **target** phase is CLOSE or LIFT | **A zero-information constant (height = 0, "apple at rest") scores 0.36 cm and passes** (computed on the identical 2,398 windows). True copy-last scores 0.02 cm. 61 % of grasp-cohort targets have \|height\| < 1 cm. | mislabelled | The gate cannot distinguish a height predictor from a constant, so the PASS is unmeasured as a test of prediction. P implies G4 is uninformative only for 120-step models; it is uninformative for every model. Add: "G4 is passed by a constant (0.36 cm); it carries no information." |
| S4-06 | G3 apple–plate "1.97 / 1.84 / 1.85 cm PASS, E3 2.39 FAIL" | R:104 | M `gates.G3_apple_plate_h8.value` | v2 `window_metrics`:416 (predicted readout, **all valid** windows) | Predicted apple–plate 3-D vector error at h = 8 over 3,961 valid windows | True-state copy-last scores **0.74 cm** and would pass (computed). A train-mean constant scores 7.24 cm and fails. The model's own copy-last was not computed. It would carry the model's encoder error, and the gap between 0.74 cm and the models' 1.8–2.4 cm suggests the gate mostly measures readout (perception) error. That last point is reasoned, not computed. | confirmed | The values are correctly named. The note matters only for the gate-count reading (S4-09): E3's G3 FAIL is consistent with its degraded encoder, not with its prediction step. |
| S4-07 | G5 `apple_held` AUROC "0.9992–0.9997"; "shuffled-action AUROC … 0.714–0.812 … most of it comes from the encoded state and fused proprioception" | R:106, R:481-485; README:41-43 | reports `held_lift_auroc`, `held_lift_shuffled_auroc` (0.768 / 0.714 / 0.812 / 0.761) | v2 `window_metrics`:421 | Predicted held-probability AUROC over the lift cohort (879 positive / 866 negative) | True copy-last `held` scores AUROC 0.956, and true start-height alone scores 0.997. Both would pass ≥ 0.85 (computed). | mislabelled (lead adjudication, see Adjudications; slice 4 said confirmed) | The docs already self-qualify it as "weak evidence" and say it must not be credited to prediction. The blind-baseline numbers strengthen that qualification. |
| S4-08 | G6b top-1 regret "0.0 mm" (3 arms), random-choice "8.44 mm", "null pass rate 5.9 %" | R:108, R:318-320, R:486-492 | report `ranking.16.top1_regret_median_m`, `random_choice_regret_median_m` | v3 `_summarize_ranking`:160-162 | Median over 18 ranked sibling groups at h = 16 | A constant predictor ties every candidate, and argmin picks the first sorted name, which is the root in 19/19 groups. Its median regret is **8.13 mm, which fails 4 mm** (computed). The 5.9 % null rate is quoted from the v3 prereg and was not recomputed. | confirmed | |
| S4-09 | "The control passed more gates than every intervention" — "**10 of 14**" vs "9" | R:12-13, R:115; TASK-054:74-76; TASK-055:12-13; DECISIONS:79-81; README:56-57; MODELS.md:89-90; M `overall.control_passed_more_gates_than_every_intervention` | M `results.*.gates_passed` | v4 `evaluate_gates`:161-212 (count of `passed`) | A count of 14 heterogeneous gates | The arithmetic is right (E0 fails G1, G2a, G7a, G9; E1/E2 fail G1, G2a, G6a, G7a, G9; E3 fails G1, G2a, G3, G6a, G9). Among the passes, **G4 is passable by a constant** (S4-05), **G3 and G5 by a true copy-last** (S4-06/07), and G8a/G8c are floors (R:518-519 says so). The E0 vs E1/E2 difference is **G6a alone**, a single h = 16 statistic on 18 groups (see S4-10). | mislabelled | The count is read as the control being better overall. What it actually records is "E0 alone clears G6a at h = 16". Correction: "E0 passes 10 and E1–E3 pass 9. The difference is G6a (h = 16) for E1/E2, and G6a/G3 against G7a for E3. At least three of the passes shared by all arms (G3, G4, G5) are cleared by a copy-last or constant predictor." |
| S4-10 | "Every intervention hurt candidate ranking … the most consistent pattern"; "the one metric a CEM directly needs"; E1 "markedly worse at the metric a CEM actually uses" (G6a 0.5438 → 0.2865 / 0.3296 / 0.4141) | R:219-221, R:313-317, R:496-497; DECISIONS:99-101; README (via "10 of 14"); MODELS.md:88-90 (indirect) | report `ranking.{8,16,32}.within_state_spearman_pooled` | v3 `candidate_ranking_metrics`:77, `_summarize_ranking`:167; gate uses h = 16 only (v3:250) | Pooled within-state Spearman over 18 groups at **h = 16**. The planner horizon is **8** (P:580-581). | At h = 8, the horizon a CEM uses (10 ranked groups, below the 12-group cohort rule), ρ is **E0 0.36, E1 0.50, E2 0.36, E3 0.30**. At h = 32 (17 groups) it is E0 0.359, **E1 0.368, E3 0.417**, E2 0.307. The "all three interventions hurt ranking" ordering holds **only at h = 16**. It is single-seed with no interval. A constant predictor gives ρ = None and fails (computed). | mislabelled | Correction: "At the gated h = 16, all three interventions scored lower G6a than the control (single seed, no interval). The ordering does not hold at h = 8 (E1 highest, 10 groups) or h = 32, so it is not established as a property of the metric a horizon-8 CEM uses." |
| S4-11 | G7a "74 / 67 / 70 / 77 of 106"; E3 the only pass; "Making the predictor horizon-conditioned **did improve** how much the prediction follows the executed actions" | R:109, R:281-301; TASK-054:96-97 | report `siblings.16.own_beats_swapped_fraction`, `qualifying_ordered_pairs` = 106 (all arms) | v2 `sibling_metrics`:463, fraction `(own < swap).mean()` at v2:520 | Ordered sibling pairs at h = 16 whose true offsets diverge by ≥ 1 cm, rolled out from the shared anchor frame | The counts are right (×106). Copy-last gives own == swap, the strict `<` makes the fraction 0, so it fails (reasoned from code). | mislabelled | The numbers are confirmed. "Did improve" is a causal claim on +3 pairs of 106, one seed, no interval (the doc itself says "thin"). Correction: "E3 won 3 more of 106 pairs than the control (77 vs 74). No interval, one seed, so no improvement is established." |
| S4-12 | G9 "1.058 / 1.118 / 0.903 / 0.796 cm FAIL"; the rollout excess is "**the prediction step's own contribution**"; "E3 has the lowest rollout excess"; v2 → v3 excess "doubled to tripled" | R:114, R:122-127, R:168-180, R:274-279; P:26-28, P:258-287; TASK-054:37, 75-76; DECISIONS:85-88 | M `results.*.decomposition_h8.rollout_excess_median_m` | v4 `decompose`:140 (difference of medians); gate v4:203 | median(rollout) − median(encoded target) over the same 1,462 windows | The **median of the per-window difference** is **0.479 / 0.527 / 0.469 / 0.526 cm**, and every arm, including the control, would be ≤ 0.53 cm under that estimand. On 32–35 % of moving windows the rollout is **better** than the encoded target. Under the per-window estimand E3 is not the lowest (E2 is). The identity rollout gives persistence − encoded target = 1.57 cm and fails (confirmed; S4-19). | mislabelled | The FAIL verdicts are correct for the frozen definition. The prose meaning is not what is computed: a difference of medians is not a per-window "contribution" and does not decompose additively. R:206-213 discloses the estimand gap only for the bootstrap, not for the gate's interpretation. No pre-declared outcome depends on G9 (P:315-316), so the decision is unaffected. Correction: call G9 "the difference between the rollout and encoded-target medians". Add that the per-window median excess is 0.47–0.53 cm in all four arms. |
| S4-13 | E2: "of the 0.173 cm rollout improvement, **0.155 cm is rollout excess** and only 0.018 cm is the encoder … The mechanism behaved as designed"; "**90 % of its gain is in the targeted term**" | R:233-236; TASK-054:90-91 | M `contrasts_difference_of_medians_m.E2_*` | Arithmetic on v4 `decompose`:140 medians | Differences of medians | Under the per-window estimand, E2's excess change is **−0.010 cm** (M/bootstrap `E2_…_rollout_excess.point_m` = −0.000103 m), which is **6 %** of the 0.173 cm rollout gain. Its clustered interval is [−0.134, +0.134] cm. | mislabelled | The attribution is an artefact of subtracting medians and is unmeasured. R itself reports −0.010 cm at R:209-210 but keeps the attribution. Withdraw "almost all of its gain is in the term this task attacked", "the mechanism behaved as designed" and "90 %". |
| S4-14 | E2 "14.6 % reduction where 49.9 % was required"; "about 3.4× … G9 and 5.9× … G2a"; "7.9 % above" | R:180, R:240-244, R:401; TASK-054:90; DECISIONS:97 | M decomposition fields, G2a values | Arithmetic on the S4-12 fields | Gate-consistent difference of medians | 0.1547 / 1.0578 = 14.6 %; 0.5278 / 0.1547 = 3.41; (0.8763 − 0.8) / 0.0128 = 5.95; 0.8635 / 0.8 = 1.079 (computed) | confirmed | "5.9×" is 5.95, a rounding nit. The same estimand caveat as S4-12 applies. |
| S4-15 | README: tail-weighted loss "**about 15 % of the needed change**" | README:54 | as S4-14 | as S4-14 | The 14.6 % is the reduction **of the excess**. The needed reduction was 49.9 %, so E2 delivered 0.1547 / 0.5278 = **29 % of the needed change** (R:401 says "under a third"). | — | wrong | Correction: "a 14.6 % reduction of the excess where 49.9 % was needed (under a third of the required change)". |
| S4-16 | Bootstrap intervals: E1 +0.010 [−0.407, +0.367], E2 −0.173 [−0.804, +0.132], E3 +0.761 [+0.155, +1.199], encoded target E3 [+0.485, +1.435]; 67 clusters; clustered "1.46×–2.28× wider (1.66×–2.28× over six shown)" | R:187-204, R:245-248; TASK-054:91-94; DECISIONS:89-93 | `outputs/task054-wm-v4/paired-bootstrap.json` (git-ignored); point estimates in M `contrasts_difference_of_medians_m`; M `paired_bootstrap` | `scripts/bootstrap_wm_v4_contrasts.py` `paired_intervals` (percentile of the paired median difference; the cluster design resamples episodes) | Cohort sampling for fixed checkpoints | Recomputed width ratios 1.456–2.283 (nine) and 1.664–2.283 (six) ✓. 67 episodes ✓. | confirmed | These are not run-to-run intervals (declared). |
| S4-17 | DECISIONS: "E3 **+0.761 cm [+0.155, +1.199]** (excludes zero — **E3 damaged the encoder**)" | DECISIONS:91-92 | as S4-16 | as S4-16 | The quoted interval is the **rollout** contrast | The evidence for encoder damage is the **encoded-target** contrast +1.023 [+0.485, +1.435] (R:204, R:268-269) | mislabelled | Minor. Cite the encoded-target interval for "damaged the encoder", or change the parenthetical to "E3's rollout got worse". |
| S4-18 | Encoder "reads the palm–apple offset to **2.35–2.59 cm** median on **held-out validation windows** (start to target, three v4 arms whose encoder was intact), against 3.13 cm at v2"; "the encoder is the component that works" | README:37-40; DECISIONS:107-110; R:474-478 | M `decomposition_h8.encoded_start_median_m` (2.3536 / 2.3548 / 2.3978), `encoded_target_median_m` (2.5903 / 2.5403 / 2.5723) | v4 `decompose`:111-114 (`readout_rows` = encode → readout, no prediction) | A direct encode → readout, correctly **not** a rollout (unlike the v3 0.78 cm defect). **But:** only the 1,462 **moving** windows (37 % of valid windows, chosen by true motion); val was also the checkpoint-selection split; "encoded" includes **fused proprioception** (`state_fusion: true`); the range mixes start-frame and target-frame errors; the v2 3.13 cm was carried over, not recomputed (P:42-44). | A train-mean constant scores 10.43 cm on this cohort (computed), so the encoder clearly beats a prior-only predictor | mislabelled | The qualifiers are dropped. Correction: "…to 2.35–2.40 cm (start frame) and 2.54–2.59 cm (target frame), median over the moving validation windows (val also used for checkpoint selection), from image plus fused proprioception; v2's 3.13 cm is carried over, not recomputed". |
| S4-19 | G9 "not passable by a do-nothing predictor … 1.57 cm (B), 1.51 cm (D)"; pilots "all four PASSED G9 … 0.109 / 0.337 / 0.161 / 0.122 cm", smoke-a "−1.66 cm, 14.13 cm"; pilots "failing G2a **and every other gate**" | P:295-306; v4 docstring; M `g9_threshold_derivation.passable_by_an_undertrained_model_measured_in_the_pilots` | M `baselines.v3_arm_{B,D}_{persistence,encoded_target}_m`; pilot fields in M | v4 `decompose`:140 on the pilot reports `outputs/task054-scratch/*-gates.json` | — | 4.1628 − 2.5903 = 1.5725 ✓; 3.9771 − 2.4695 = 1.5077 ✓. Pilot values re-read from the scratch reports ✓. The pilots passed 5 gates each (smoke-a passed 4: G5, G8a, G8c, G9). | wrong (the "every other gate" clause) — already withdrawn in R:448-457 and M `preregistration_defects_found_after_launch` | The rest is confirmed. The pilot pass sets (G4, G5) are consistent with S4-05/S4-07: these gates are passed without a trained predictor. |
| S4-20 | G9 derivation: "1.05× the C↔A and 1.54× the D↔B contrast" (body) vs pre-run review item 3: "it is **1.04× and 1.56×**" | P:273-276 vs P:543-544 | M `g9_threshold_derivation` | Arithmetic | 0.52780 / 0.50484 = 1.0455; 0.52780 / 0.34376 = 1.535 (computed) | — | wrong | Minor internal inconsistency in the frozen prereg. 1.56× is wrong (it is 1.54×), and 1.04× is a truncation of 1.0455. Erratum only. |
| S4-21 | "E0 reproduces v3 arm B bit-for-bit … all **170 weight tensors** bit-identical"; 13 gates exact floats; 11 decomposition quantities | R:130-159; TASK-054:78-82; MODELS.md:70-72; M `overall.e0_reproduces_v3_arm_b_bit_for_bit` | M prose string plus committed checkpoint SHA-256s. No committed script computed it. | None committed. Recomputed here with `torch.equal` over `envelope['weights']` of `checkpoints/task052-wm-v3/leworldmodel_onboard.pt` vs `checkpoints/task054-wm-v4/leworldmodel_baseline.pt`. | Model weights | **170/170 equal** (computed). The G1/G2a floats equal M `baselines.v3_arm_B_*`. | confirmed | Evidence lives in git-ignored checkpoints. Only the claim string and the hashes are committed, so it is reproducible but not citable as a committed measurement. |
| S4-22 | "A predictor that saw the offset exactly and then assumed it froze would still beat every arm on the moving windows" (true displacement 2.49 cm) | R:321-324 | report `palm_apple_moving_true_displacement_median_m` = M `decomposition_h8.true_displacement_median_m` | v2 `window_metrics`:372/402; v4 `decompose`:96 | Median ‖truth(target) − truth(start)‖ over moving windows, which equals the true-state copy-last error | 2.49 < 3.48 (best arm) ✓ (computed). The cohort is selected by displacement ≥ 1 cm, so this baseline's error is ≥ 1 cm by construction. | confirmed | Over **all valid** windows, true copy-last is 0.50 cm against the rollout's 0.78–1.23 cm, so the conclusion also holds there. |
| S4-23 | Trade-off Spearman "−0.107", "+0.084", "−0.400", "+0.800" | R:350-355; TASK-054:116-118 | Prose only (inputs from M and the v3 manifest) | `world_model_v2.spearman`:286 | Rank correlation across 7 or 8 arms and 4-arm subsets | Recomputed all four ✓ | confirmed | The inputs are committed. The statistic itself is prose-only, but it is reproducible. |
| S4-24 | "Five attempts … **motion-weighted readout shaping (v3, no help and it hurt candidate ranking)**" | DECISIONS:95-96; README:52-53 (MODELS.md:88 says "the readout shaping", which is correct) | v3 manifest arm A vs C | v3 contrast A ↔ C | Arm C drops **both** the motion weighting **and** the auxiliary position readouts (`apple_world_model_v3_results.md`:142-144, 513: "A ↔ C is not 'the motion weighting' on its own"). "Hurt ranking" is G6a 0.328 (A) vs 0.5155 (C), single seed, no interval. | The named entity does not match the manipulated factor | mislabelled | Correction: "v3 readout shaping (motion weighting plus auxiliary position targets, one bundled factor): no G1 help; lower G6a on one seed". |
| S4-25 | "**0/150 per model** on the frozen unseen-pair benchmark … 400 attempts including hold and random controls"; "0/50 in every one of the eight runs … no backend beating the hold or random controls" | README:9-11; CLAUDE.md:73; EVALUATION.md:44; origin `mvp_results.md`:3, 11-20 (pre-TASK-048) | `benchmarks/manifests/mvp-results-v0.json` `evaluation[*].summary.{episodes,successes,termination_reasons}`, `total_executed_steps`, `limit_rejections` | `benchmark.summarize`:52-55 (sum of `score.success`) via `scripts/summarize_mvp.py` | 6 CEM runs (2 backends × 3 seeds, 3,000 updates on the 184-episode `mvp-v0` corpus, horizon 4, task `tabletop_proxy_v0`) × the **same** 50 resets | 6 × 50 + 50 + 50 = 400 ✓; successes all 0 ✓. **Every one of the 300 model episodes terminated `stopped`, with 50 `limit_rejections` per run.** Commands executed: 1,117 / 239 / 620 / 706 / 670 / 461 per 50 episodes (≈ 5–22 per episode). Hold ran all 805 steps. Origin (`mvp_results.md`:20-22) says the 0/150 is "not 150 independent environments", gets "no pooled binomial interval", and that all model episodes "stopped on right joint rate limit". | mislabelled | The count is right, but the top-level restatements drop that every learned episode ended on a joint-rate guard stop after a few commands (a command-feasibility failure, not a manipulation attempt that failed), and that the same 50 resets were reused across seeds. Correction for README/CLAUDE.md: "0/50 on each of three seeds per backend on the same 50 resets (every learned episode ended on a joint-rate guard stop, after a per-run mean of ~5–22 commands (per-episode range 0–170; lead correction after review); not 150 independent trials)". The headline "0 successes" stands. |
| S4-26 | Budget "14,470 s = 4.02 h"; wall clocks 3,512 / 3,884 / 3,553 / 3,521 s; eval 16/17/16/17 s (log), 14.0–15.6 s (report); "No arm hit the cap" | R:44-50, R:73-78; TASK-054:105 | M `overall.measured_training_seconds_total` = 14,469.78; `results.*.elapsed_seconds`; `outputs/task054-runner.log` | Run clocks (`RunClock`) | — | Checked against M and the runner log timestamps (16/17/16/17 s) ✓ | confirmed | The runner-log basis is git-ignored. The M fields are committed. |

### Divergences needing correction

1. **S4-02** (mislabelled). There are four locations.
   - `apple_world_model_v4_results.md:493-495`: "The persistence baseline is the thing to beat and it has never been beaten: G2a is 0.8635 at best here, 0.831 at best in v3, 0.835 in v2."
     → "The rollout does beat the model's own persistence readout (G2a < 1 in every v4 arm), but never by the required margin: G2a is 0.8635 at best here, 0.831 in v3, 0.835 in v2, against ≤ 0.8. It has never beaten the true-state copy-last baseline (2.49 cm on the moving windows)."
   - `apple_world_model_v4_results.md:311-312`: "But every arm still loses to its own persistence readout by the required margin"
     → "But no arm beats its own persistence readout by the required margin".
   - `apple_world_model_v4.md:371-372` (frozen prereg, erratum only): "demonstrably beats the model's own persistence readout for the first time in this line"
     → "beats the model's own persistence readout by the required 20 % margin for the first time (G2a < 1 already held for v2 and three v3 arms)".
   - `apple_world_model_v4.md:60-61`: "below the model's own persistence readout (G2a < 0.8)"
     → "at least 20 % below the model's own persistence readout (G2a ≤ 0.8)".
   - TASK-054 title: note in the erratum that "beat the persistence baseline" means G2a ≤ 0.8.
2. **S4-05** (mislabelled). At `apple_world_model_v4_results.md:105` (G4 row), add a note: "G4 is passed by a constant zero-height predictor (0.36 cm median abs on the identical 2,398 grasp windows); it carries no information about prediction." At `apple_world_model_v4.md:519` ("G4 on a 120-step model says nothing"), the erratum should say that G4 says nothing for any model.
3. **S4-09** (mislabelled). Affects `apple_world_model_v4_results.md:12-13` ("The control passed more gates than every intervention"), TASK-054:74-76, TASK-055:12-13, `DECISIONS.md:80-81`, `README.md:56-57` and `MODELS.md:89-90`.
   Add: "The difference is G6a at h = 16 (E1, E2), and G6a and G3 against G7a (E3). At least three of the shared passes (G3, G4, G5) are also cleared by a copy-last or constant predictor, so the gate count is not a quality ranking."
4. **S4-10** (mislabelled).
   - `apple_world_model_v4_results.md:220-221`: "the arm became markedly worse at the metric a CEM actually uses"
     → "…at the gated h = 16 ranking (at the planner's h = 8, on 10 groups, E1's ρ is 0.50 against E0's 0.36)".
   - `apple_world_model_v4_results.md:313-317` ("Every intervention hurt candidate ranking … the most consistent pattern") and `DECISIONS.md:99-101` ("the control is the only v4 arm that passes the one metric a CEM directly needs")
     → qualify both as "at h = 16, single seed, no interval; not reproduced at h = 8 or h = 32".
5. **S4-11** (mislabelled).
   - `apple_world_model_v4_results.md:283-284`: "Making the predictor horizon-conditioned *did* improve how much the prediction follows the executed actions."
     → "E3 won 3 more of 106 sibling pairs than the control; with one seed and no interval this is not an established improvement."
6. **S4-12** (mislabelled). Affects `apple_world_model_v4.md:27-28` and `results:168-180` ("the prediction step's own contribution"), and `DECISIONS.md:85-88`.
   Add: "G9 is the difference of the rollout and encoded-target medians. The median per-window excess is 0.479 / 0.527 / 0.469 / 0.526 cm, at or below the 0.53 cm threshold for every arm including the control, so the size of 'the term that grew', and any reading of G9 as a per-window contribution, depends on the estimand." The verdicts are unchanged.
7. **S4-13** (mislabelled; withdraw).
   - `apple_world_model_v4_results.md:233-236`: "almost all of its gain is in the term this task attacked: of the 0.173 cm rollout improvement, 0.155 cm is rollout excess and only 0.018 cm is the encoder. The mechanism behaved as designed."
   - TASK-054:90-91: "90 % of its gain is in the targeted term".
   - Both → "Split by differences of medians, 0.155 cm of the 0.173 cm is excess. The per-window excess changed by only −0.010 cm [−0.134, +0.134], so where the gain sits is not measured."
8. **S4-15** (wrong).
   - `README.md:54`: "a tail-weighted multistep loss (about 15% of the needed change, …)"
     → "(a 14.6 % reduction of the excess where 49.9 % was needed — under a third of the required change, …)".
9. **S4-17** (mislabelled, minor).
   - `DECISIONS.md:91-92`: "E3 **+0.761 cm [+0.155, +1.199]** (excludes zero — E3 damaged the encoder)"
     → "(excludes zero; E3's encoded-target error also rose, +1.023 cm [+0.485, +1.435])".
10. **S4-18** (mislabelled).
    - `README.md:37-40` and `DECISIONS.md:107-108` ("reads the palm–apple offset to 2.35–2.59 cm median on held-out validation windows … against 3.13 cm at v2") → see the row's correction. It needs four qualifiers: moving windows only, start frame vs target frame, image plus fused proprioception, and val used for selection with v2 not recomputed.
11. **S4-19**. Already withdrawn at `apple_world_model_v4_results.md:448-457`. No new action.
12. **S4-20** (wrong, minor).
    - `apple_world_model_v4.md:543-544`: "it is 1.04× and 1.56× those two contrasts"
      → erratum: "1.05× and 1.54× (as stated at :274)".
13. **S4-24** (mislabelled).
    - `DECISIONS.md:95-96`: "motion-weighted readout shaping (v3, no help and it hurt candidate ranking)"
    - `README.md:52-53`: "motion-weighted readout shaping (no help, and it hurt candidate ranking)"
    - Both → "v3's bundled readout shaping (motion weighting together with auxiliary position targets; not separable) — no G1 help and a lower G6a on one seed".
14. **S4-25** (mislabelled).
    - `README.md:9-11` and `CLAUDE.md:73` ("0/150 per model on the frozen unseen-pair benchmark")
      → add "on the same 50 resets per seed; every learned episode ended on a joint-rate guard stop, typically within a few commands (per-run medians 3–15, maximum 170) (`mvp_results.md` §Physical outcomes)".
    - `EVALUATION.md:44` is accurate as written.

### Cross-references

- **Behaviour-cloning line and TASK-057.** The pivot (Outcome B, S4-01) is correctly computed, and the decision
  stands. Several of the handoff facts BC inherits are restated with qualifiers dropped:
  - The "encoder works, 2.35–2.59 cm" figure (S4-18) is a moving-window-only, proprio-fused, start-and-target
    mixed figure.
  - "Candidate ranking is fragile / interventions break it" (S4-10) holds only at h = 16.
  - "Held readouts near-perfect" (S4-07) is passable by a true-state copy-last (AUROC 0.956).
  - If `apple_policy_diagnostics_v1.md` or any BC-critic protocol cites the world model as a critic on the
    strength of G5, G4, G3 or the gate count, those citations inherit S4-05, S4-07 and S4-09.
  - I did not read `apple_policy_diagnostics_v1.md` (owned by another agent). It should be checked for these
    restatements.
- **The same estimand pattern as the known defects.** The G9 excess (S4-12/13) is a fifth instance of the
  "aggregate destroys the distinction" pattern. A difference of medians was read as a per-window contribution.
  Under the per-window estimand, all four arms, including the control, sit at or below the 0.53 cm threshold.
  This does not change any decision, because no outcome depended on G9.
- **The CLAUDE.md research-evidence rule** repeats the 0/150 without the guard-stop qualifier (S4-25). CLAUDE.md
  is outside this worker's edit authority. It is flagged for the lead.

## TASK-058 audit, slice 5: TASK-056 offline, pre-flight and training stages

Corpus: `docs/experiments/apple_policy_v1.md` (prereg, load-bearing numbers only),
`docs/experiments/apple_policy_v1_results.md` §1–§13 (L1–598), `benchmarks/manifests/apple-policy-v1.json`
(offline parts). Repo at `8306633`. The pre-flight ran at `e7dc6e8`. `git diff e7dc6e8 HEAD` is empty for
`scripts/measure_policy_preflight.py`, `world_model_v2.py` and `models/base.py`, so current line numbers
apply. The arms trained at `919a75b`. `cloning.py` last changed in `9b6d4b9`, before that.

Artifacts read (git-ignored, main checkout): `outputs/apple-policy-v1/preflight.json`,
`checkpoints/task056-policy-v1/a{0,1,2,3}.run.json`, and `data/apple-wide-v1` (only the label sidecars
and the `observation.state` parquet column; no images decoded).

Blind baselines were computed by `benchmarks/audits/task058/s5_blind.py`, `s5_blind2.py` and `s5_blind3.py`,
which are read-only and take under 1 s. They use the same cohort as the pre-flight: 15 surviving val
roots, 8 719 rows, 985 apple-dropped, 7 734 scored, all reproduced exactly. The train side is the 137
surviving train roots, and the BC mask reproduces 68 791 train / 7 723 val rows exactly. The
"proprio-linear" probe is an ordinary least-squares map from the 86-D `observation.state` (joint
positions and velocities only, no object state) plus a bias to the target. It is fit on train rows
and scored on val. It uses no image.

| ID | Claim (short, quoted number) | Where (file:line, all restatements) | Artifact field | Computing code (file:function:line @commit) | What it measures / population | Entities & baseline check | Class | Note / proposed correction |
|---|---|---|---|---|---|---|---|---|
| S5-01 | Per-phase table "directly encoded, horizon 0" (orient 0.490, descend 0.404, close 0.491, lift 0.356, transfer 0.246, release_high 1.579, lower_open 4.084 cm; apple-position column) | results L45–55; manifest `results.preflight.by_phase` (~L1223ff); protocol §5 | `preflight.json: measurement.by_phase.*` (git-ignored); values copied into the committed manifest | `measure_policy_preflight.py:per_phase_errors:144` → `world_model_v2.readout_rows:314` → `encode_rows:305` → `VisualModel.encode` (`models/base.py:365`) → `readout` (`base.py:535`) @e7dc6e8 | **Encode then readout only. There is no `predict` call, so this is horizon 0.** `encode` adds `state_features(proprio)` because `state_fusion: true`, so the latent fuses the image with proprioception. Population: rows of the 15 surviving val roots minus apple-dropped rows, split by `collector__phase_index`. The median of the 3-D norm is taken per phase. | Phase names 5 and 6 (`release_high`, `lower_open`) are correct for the collector actually used (`EarlyReleaseOracleManipulationPolicy`). Row counts 1950/1200/675/2181/840/709/179 reproduced. | confirmed | "Directly encoded" is literally true. Readers should know that the encoded latent contains proprioception, so a "perception" number here is not image-only. The arms A2 and A3 consume `image_features`, which do not include it. |
| S5-02 | P1 PASS: close palm–apple **0.491 cm** ≤ 1.5 cm, P0a 1.314 cm, ratio **0.374** | results L39, L59–62, L206; manifest `results.preflight.gates` | `preflight.json: gates.P1_close_palm_apple` | `measure_policy_preflight.py:evaluate_preflight_gates:281`, `shuffled_frame_control:199`, `control_errors:247` @e7dc6e8 | Median palm–apple readout error on close-phase rows (675), and the same model's readout with a foreign same-phase image and its own proprio. | **Blind check:** a model that ignores the image scores a ratio of ≈1.0 against its own control, so the gate as defined **cannot be passed blind** (reasoned). The absolute value can be approached without vision: a phase-conditioned proprio-linear probe gets **0.808 cm** on close; a per-phase train-median constant gets 1.19 cm, and a per-phase train mean gets 1.68 cm. E0 beats the proprio-linear probe by 0.32 cm. | confirmed | The verdict stands. It shows that E0's close readout depends on the image and beats a linear proprio probe. It does not show how much of the 0.491 cm comes from pixels rather than from state (unmeasured). |
| S5-03 | P2 PASS: approach palm–apple **0.456 cm** ≤ 1.5 cm, P0a 0.669 cm, ratio **0.682** (margin 0.018) | results L40, L74–84; manifest `results.observations.P2_ratio_margin` | `preflight.json: gates.P2_approach_palm_apple` | same as S5-02 @e7dc6e8 | Orient and descend rows (3 150) pooled into one median. | Not passable blind (ratio ≈1 for an image-ignoring model, reasoned). **The absolute value is worse than a zero-information prior:** the per-phase train-median constant gets **0.102 cm**, the per-phase mean 0.353 cm, and the phase-conditioned proprio-linear probe 0.240 cm, against E0's 0.456 cm. | confirmed | The verdict and ratio are computed as described. §2 already says most of the accuracy is available without the apple; the prior shows it is **all** available, and more. |
| S5-04 | §3: "P3's control lands at **2.269 cm**, essentially on the analytic blind prior of 2.394 cm … **absolute apple position genuinely cannot be inferred from the palm's own pose**, so P3 is the one measurement here with **no shortcut available to it**"; model **0.873 cm**, ratio 0.385; "the result that most supports A4" | results L98–107; manifest `results.observations.P3_is_the_cleanest_measurement` | `preflight.json: gates.P3_orient_apple_position`, `measurement.no_vision_control.cohorts.P3_orient_apple_position` | same as S5-02 @e7dc6e8 | P0a is E0's readout when **fed a foreign image** with its own proprio. It measures how E0 behaves under a conflicting image. It does not measure what proprioception contains. | **Computed:** a proprio-linear probe with **no image and no phase knowledge** (one fit over all train rows) gets **1.01 cm** on orient-phase apple position. That passes P3's 2.0 cm absolute threshold, and at 0.45× P0a it would pass the ratio. Fit on orient rows only, it gets **0.163 cm**, 5× better than E0. The reason: in `orient` the collector servos the palm to `obj + [-0.03+0.015, 0, 0.13]`, and on 74.5 % of val orient frames the palm moves less than 1 mm per step, so the palm pose encodes the apple's absolute position. | **wrong** | The information claim is false, and the "cleanest / stronger evidential status" reading falls with it. The 0.873 cm value and the PASS verdict stand, since the ratio is model-relative. |
| S5-05 | §1: "**Declared risk R2 is not borne out** … Approach phases read **0.404–0.490 cm**, so the accuracy is not confined to the close" | results L64–66; manifest `results.observations.R2_not_borne_out_as_written` | `preflight.json: measurement.by_phase.{orient,descend}` | `per_phase_errors:144` @e7dc6e8 | Unconditional per-phase medians over all orient and descend rows. | **Computed:** the per-phase train-mean constant gets **0.367 cm** on orient and **0.313 cm** on descend, better than E0 on both. 74.5 % of orient and 85.7 % of descend val frames are station-keeping (palm step under 1 mm) at a fixed apple-relative offset. | mislabelled | The statistic cannot distinguish localization from the phase prior, so R2 is **unmeasured** by these numbers, not "not borne out". |
| S5-06 | Prereg P2 threshold provenance: 2.5 cm "was inside the 2.394 cm blind baseline", "1.5 cm is 0.63 × that baseline, so it requires the model to remove ≥ 37 % of the pure-prior error"; "a pure-prior model reads ~2.4 cm on orient/descend" | apple_policy_v1.md L425–430, L442; manifest `preflight.no_vision_control.what_it_fixed`, `preflight.thresholds.P2`; code comment `measure_policy_preflight.py:85–89`, `threshold_provenance` L405–414 | `P0b_analytic_prior_median_m` 0.023937 | Analytic value; 2e6-sample check reproduces 2.3937 cm | P0b is the prior for **absolute apple position**. P2 scores **palm − apple**, a relative quantity. | **Computed:** the pure-prior error on P2's quantity is **0.10 cm** (per-phase median) to 0.35 cm (mean), not ~2.4 cm. So P2's 1.5 cm absolute threshold **is passable by a pure prior**, and the "remove ≥ 37 %" claim does not hold for P2. Only the ratio condition protects P2. | **wrong** | The P2 verdict is unaffected because the ratio carries it, but the stated threshold rationale is wrong for the relative quantity. The P0b 2.394 cm itself is correct as an absolute-position prior; the empirical per-orient train mean gives 2.385 cm. |
| S5-07 | P0a called the "no-vision control"; "a readout computable from someone else's FRAME is reading the prior"; §2 "on approach frames a **foreign** image still yields 0.669 cm, so most of the relative accuracy is available without looking at the apple" | results L37 table header, L80–84; code field `measurement.no_vision_control` (`control_errors:247`); prereg L402–418 | `preflight.json: measurement.no_vision_control` | `shuffled_frame_control:199` @e7dc6e8 | This is a **wrong-image** control: a foreign same-phase image plus own proprio, read by E0. It is not a no-vision predictor. | Measured no-vision predictors beat it on all three cohorts: 0.808 < 1.314 (P1), 0.10–0.24 < 0.669 (P2), 0.163 < 2.269 (P3). Prereg L405–410 admits that P0a is inflated, but only in the ratio-leniency sense. | mislabelled | P0a is valid as the ratio's denominator, where it tests whether the readout uses the image. It is not an estimate of blind error, and it cannot be read as "what is available without vision". |
| S5-08 | Cohort: 15 surviving val roots, **7 734 rows scored**, **985 dropped** (apple off table), 9.5 s on MPS | results L32–35; manifest `results.preflight` | `preflight.json: cohort`, `measurement.rows_*`, `elapsed_seconds` 9.516 | `surviving_rows:133`, `per_phase_errors:150` (`apple_dropped` = `privileged__apple_dropped`, apple centre below 0.70 m) | All rows of the 15 non-branch, non-aim-offset val roots. | Reproduced from labels: 8 719 / 985 / 7 734. | confirmed | — |
| S5-09 | "0.491 cm, against the **2.3536 cm** that the committed record's only directly-encoded measurement (the moving cohort) reported" → "premise … now supported" | results L59–61; prereg L449–455; manifest `evidence_framing.only_committed_h0_numbers` | `apple-world-model-v4.json: results.E0_baseline_arm_b_rerun.decomposition_h8.encoded_start_median_m` (committed; the same value also sits in the v3 manifest arm B) | `world_model_v4.py` L111–115 (`readout_rows` at window start) | Same checkpoint (E0). The population is the h=8 **moving** windows (1 462 of 3 961 valid) at their start frame. | The entity matches and the cohort is named in the text. The gap is at least partly **population**: approach phases are about 75–86 % station-keeping frames, and moving windows are the minority where E0 reads 2.35 cm. | confirmed | Note for the reading: "premise supported" rests on stationary-dominated phase cohorts. On frames where the palm actually moves, E0's h=0 error is 2.35 cm. |
| S5-10 | Prereg: "during `orient` and `descend` **essentially every frame is a moving window** … The moving cohort is therefore approximately the approach phases"; hence "a P2 failure is EXPECTED" | apple_policy_v1.md L445–455; manifest `preflight.P3_is_the_threshold_most_likely_to_bite`, `P2_is_expected_to_fail…` | none (reasoning) | — | — | **Computed:** only 25.5 % of orient and 14.3 % of descend val frames have a palm step ≥ 1 mm. Committed v4: 1 462 moving of 3 961 valid windows over all phases. | **wrong** | This is a design premise that was falsified. P2 passed, and the results doc never revisits the premise. Relevant to how the E0 2.35 cm figure is used in S5-09. |
| S5-11 | "0.491 cm sits at the **bottom of the measured grasp/hold band** (holds 0.49–1.35 cm)" | results L61–62; prereg L441 (P1 provenance); manifest `preflight.thresholds.P1` | `apple-wide-grasp-closure-v3` hold apple xy | — | The band is `close_phase_apple_xy_cm`, **how far the apple moved** during ceiling holds. 0.491 cm is a **3-D readout estimation error**. | The prereg itself states that the bridge between the two is "plausible … stated rather than assumed". | mislabelled | The results sentence places an estimation error on a physical-displacement band as though they were the same axis. It needs the prereg's qualifier. |
| S5-12 | U1–U4 "re-derived … **Every value reproduces exactly**", 4 807 windows; table (e.g. U1 h=1 0.8336 cm, U3 h=1 0.99737/0.99949, U4 h=16 0.9131 cm); "remain rollout readouts, not perception" | results L120–142; manifest `unverified_pending_P0.*.status`, `rederivation` block ~L1450ff | `preflight.json: rederivation.by_horizon` (values copied into the committed manifest) | `measure_policy_preflight.py:rederive_unverified:327` → `world_model_v2.window_metrics:353` → `predicted_readouts:320` (encode→**predict**→readout) @e7dc6e8 | Rollout readouts on all 80 val episodes (branches and aim-offset included) with windows at stride 4. U4 uses the "valid" windows (apple not dropped), not literally all windows. | Every table value matches the artifact. Labelled correctly as rollout. | confirmed | Minor: the "U4 all windows" header means all **valid** windows. The U cohort differs from the P1–P3 cohort (80 episodes against 15 roots). |
| S5-13 | "at **h = 4 the rollout is *worse* than persistence** (ratio **1.0461**)" | results L134–138; manifest `new_to_the_record` | `rederivation.by_horizon.4.U2_rollout_over_persistence_moving` | `rederive_unverified:336–340`: median(rollout on moving) / median(persistence on moving); persistence = the **encoded** start readout against the target | Ratio of medians on moving windows at h=4. | — | confirmed | — |
| S5-14 | "**A1 0.01106 < A0 0.01313 < A2 0.01793 < A3 0.02340**", "Both controls beat the primary arm", "A0 versus the best learned arm is the offline shadow of gate G2" | results L372–377; manifest `results.training.arms.*.selection_score` (~L1337–1358) | `a*.run.json: best_selection_score` (git-ignored; copied into the manifest) | `cloning.py:train.validate:426` → `evaluate_policy:248` → `action_error:232` (`np.median` over the flattened [N,7] \|error\|) @919a75b | Pooled median absolute error over all 7 free dims × 7 723 masked val rows, in normalized units, at each arm's best step. | Values reproduced. The ordering also holds under a median of per-dim medians (0.0097/0.0118/0.0199/0.0278). **Blind check (computed):** a per-phase train-median constant, which needs phase only, scores **0.01685**. It beats A2 and A3. Global constants score 0.080–0.122. | confirmed | Numbers and ordering are correct. A reader should know that a phase-only constant already beats the primary arm on this metric, so "offline shadow of G2" carries little. |
| S5-15 | Selection metric = "**median per-dimension** expert action error" | apple_policy_v1.md L265–266; manifest `training.selection_metric` | — | `action_error:240` returns `median_abs_error` = the pooled median | A pooled median over dims and rows, not a per-dimension median. | Already disclosed in manifest `training.selection_metric_as_implemented` (L1406). The prose docs do not carry the disclosure. | mislabelled | Already disclosed in the manifest. The results doc §12 should say "pooled median". |
| S5-16 | A3: "best step is its **final** step (15 000 of 15 000) … **it was still improving** when its step budget ended"; others 42 000/40 000/26 000 of 50 000; "207 s of a 7 200 s cap" | results L362–366; manifest `results.training.A3_did_not_converge` | `a3.run.json: best_step, completed_steps, elapsed_seconds` 206.9, `validation[]` | `cloning.py:train:292` | Val score every 2 000 steps. | Last A3 validations: 10k 0.02449, 12k 0.02370, **14k 0.02428**, 15k 0.02340. That is non-monotone. A3 also ran at **batch 32** against 256 for the others: 0.48 M samples against 12.8 M, about 27× fewer. | mislabelled | Best-at-final is equally consistent with noise around a plateau. "Still improving" is not shown. "No evidence it converged" is correct. The ~27× smaller sample budget is an unstated second reason A3 is not comparable. |
| S5-17 | "`right_dz` **sits at exactly 0.400** in **44.89%** of commands", "a bimodal target", smooth-L1 "hedging toward the interior" | results L385–392; restated at results L635, L759 (slice 6), manifest `dz_under_shoot` L1213, `task056_handover.md:90`, TASK-056 .mc L173, diagnostics_v1 L31 | none committed for the train figure (`measure_policy_offline_conditionals.py:203` computes the **val** \|dz\| rate, ~44.5 %) | No committed function computes 44.89 %. Reproduced here as `mean(isclose(\|dz\|,0.4))` over the 68 791 train BC target rows | An **absolute-value** rate. Signed: **dz = +0.400 on 18.45 %**, **−0.400 on 26.44 %**, interior on 55.1 %. | The value reproduces exactly, but only as \|dz\|. The hedging mechanism is a reasoned prediction and is labelled as one. `cloning.action_error` (unconditional per-dim median plus `predictions.std`) cannot test it: see the TASK-058 known pattern and diagnostics §1.1(b). | mislabelled | The number is right and the named quantity is wrong. "Bimodal" means two clip modes **of opposite sign** plus a 55 % interior. The train-population figure has no committed computing code, so it is not citable as it stands. |
| S5-18 | BC cohort: surviving **137/15/8** roots; **68 791** train / **7 723** val sampled rows; R1 "at most **6 165** close-phase" (actual 6 033); 150/200 roots noise-injected; 597 branches excluded | prereg L141–196; results §13; manifest `dataset.*`, `declared_risks.R1_*` | `a*.run.json: data.{train,val}` | `cloning.py:load_bc_split:168`, `sample_mask:136`, `is_surviving_root:94` | Surviving roots, minus the terminal row, apple-dropped rows and pre-close post-displacement rows (>1 cm from frame 0, collector phase < close). | Reproduced 68 791 / 7 723 exactly. Noise levels 0/1/2/3 are 50 roots each. | confirmed | Prereg "before the grasp stage" is implemented as collector phase < `close`, not the scorer's grasp stage. |
| S5-19 | Expert / teacher behind the BC labels: `collector__base_action` from `scripted.apple_collector_policy`; "labels on-policy-optimal" | prereg L147–152, L230, L273–281; results L155–159; `cloning.py` docstring; manifest `bc_labels.target` | label sidecars `collector__base_action` | `scripts/collect_apple_wide.py:465` (`apple_collector_policy(sim.task_truth())`), `Controller.command:268` (base = the unperturbed policy action; the perturbation is applied after) | Clean scripted command at the visited, possibly noise-perturbed, state. | **Correct expert:** `EarlyReleaseOracleManipulationPolicy(initial_truth, opening_ramp=0.08)` with every phase target shifted by `palm_x_offset = +0.015 m` in x and phases ≥ 4 by a further `transfer_x_shift = −0.035 m` (`scripted.py:192–207`). Phases are orient 130, descend 80, close 45, lift 150, transfer 60, release_high 100, lower_open 100, retreat 80, **745 total**. This corpus names it correctly. | confirmed | "On-policy-optimal" is prose, not numeric: the labels come from a scripted P-servo to initial-truth targets, with 115/200 root successes. |

### Divergences needing correction

- **S5-04 (wrong).** results L100–102, "**absolute apple position genuinely cannot be inferred from
  the palm's own pose**, so P3 is the one measurement here with **no shortcut available to it**";
  also L106–107, "its evidential status is different and stronger"; manifest
  `results.observations.P3_is_the_cleanest_measurement`.
  Proposed erratum: "P3's control (2.269 cm) measures E0's readout under a *foreign* image, not what
  proprioception contains. A proprioception-only linear probe (no image, no phase) reads 1.01 cm on
  orient apple position, and 0.163 cm when fit on orient rows, because the collector servos the palm
  to an apple-relative station during `orient`. P3's ratio shows that E0's readout depends on the
  image. It does not show that the image adds information beyond proprioception. Withdrawn: 'no
  shortcut available' and 'stronger evidential status'."
- **S5-05 (mislabelled).** results L64–66, "Declared risk R2 is not borne out in the form it was
  written … Approach phases read 0.404–0.490 cm, so the accuracy is not confined to the close."
  Proposed: "R2 is not tested by these numbers. On approach phases a zero-information per-phase
  constant reads 0.31–0.37 cm (per-phase mean) or 0.10 cm (per-phase median), better than E0, because
  most approach frames are station-keeping at a fixed apple-relative offset."
- **S5-06 (wrong).** apple_policy_v1.md L442 (P2 row), "1.5 cm is 0.63 × that baseline, so it requires
  the model to remove ≥ 37 % of the pure-prior error"; L427, "a pure-prior model reads ~2.4 cm on
  orient/descend"; manifest `preflight.no_vision_control.what_it_fixed` and `preflight.thresholds.P2`;
  the code comment and `threshold_provenance` in `measure_policy_preflight.py` L85–88 and L412–414.
  Proposed amendment note: "2.394 cm is the prior for absolute apple position. P2 scores palm − apple,
  whose phase-conditioned prior error is 0.10–0.35 cm, so P2's absolute threshold is passable by a pure
  prior. P2 is protected only by its ratio condition."
- **S5-07 (mislabelled).** results L37 ("P0a shuffled-frame control" is fine) and L80–81, "so most of
  the relative palm–apple accuracy is available without looking at the apple at all"; field name
  `no_vision_control`. Proposed: call P0a a "foreign-image control" wherever "no-vision" appears, and
  add: "P0a is a ratio denominator, not an estimate of blind error. Measured no-image predictors beat
  it on all three cohorts."
- **S5-10 (wrong).** apple_policy_v1.md L448–450, "during `orient` and `descend` essentially every frame
  is a moving window". Proposed amendment: "Measured afterwards: 25.5 % of orient and 14.3 % of descend
  val frames move ≥ 1 mm per step. Approach phases are dominated by station-keeping, so the moving
  cohort is not approximately the approach phases."
- **S5-11 (mislabelled).** results L61–62, "0.491 cm sits at the bottom of the measured grasp/hold band
  (holds 0.49–1.35 cm …)". Proposed: add "…a band of how far the apple *moved* in ceiling holds, not
  an estimation error. The bridge between them is stated, not measured (protocol §5.1)."
- **S5-15 (mislabelled, already disclosed in manifest L1406).** apple_policy_v1.md L265,
  "metric = median per-dimension expert action error". Proposed: carry the manifest's
  `selection_metric_as_implemented` note into results §12: "a pooled median over the flattened
  [N,7] error array".
- **S5-16 (mislabelled).** results L363–364, "there is no evidence it converged — it was still
  improving when its step budget ended." Proposed: "…there is no evidence it converged. Its val score
  was not monotone over the last 5 000 steps (0.0237 → 0.0243 → 0.0234), so best-at-final does not
  show continued improvement. A3 also trained at batch 32 against 256, about 27× fewer samples."
- **S5-17 (mislabelled).** results L386–387, "`right_dz` sits at exactly 0.400 in **44.89%** of
  commands", and L388, "bimodal target". Proposed: "`|right_dz|` = 0.400 on 44.89 % of the 68 791
  train BC targets (+0.400 on 18.45 %, −0.400 on 26.44 %): two clip modes of opposite sign plus a 55 %
  interior." Also commit the computation, since `measure_policy_offline_conditionals.py` computes only
  the val rate. The same wording recurs at results L635/L759 (slice 6), `task056_handover.md:90` and
  the TASK-056 .mc body L173.

### Cross-references

- **The correct expert for TASK-057** (`apple_policy_diagnostics_v1.md` §2 shadow expert; not edited)
  is `scripted.apple_collector_policy(initial_truth)` = `EarlyReleaseOracleManipulationPolicy(
  initial_truth, opening_ramp=0.08)` with `palm_x_offset=0.015` added to every phase target's x and
  `transfer_x_shift=-0.035` added for phase index ≥ 4, with phase budgets summing to **745** (plain
  `OracleManipulationPolicy`: 805). Source: `scripts/collect_apple_wide.py:451,465`,
  `src/embodied_jepa/scripted.py:157–207`. The recorded phase counts in the BC data confirm it: val
  phase 5 = 699 rows and phase 6 = 178 rows exist, and indices 5/6 are only `release_high` /
  `lower_open` on the EarlyRelease variant. (Aim-offset roots additionally get `shifted(policy,
  (0,1,2,3), aim_offset)`, but they are excluded.)
- **New, same root cause:** `scripts/measure_policy_offline_conditionals.py:50` defines
  `PHASES = ("orient","descend","close","lift","transfer","lower","release","retreat")` "from
  `scripted.OracleManipulationPolicy.phases`", which is the plain policy's names. As a result,
  **Table E of `apple_policy_diagnostics_v1.md` (L762–800) and `apple-policy-diagnostics-v1.json`
  (~L215, L235) label phase 5 (699 rows = `release_high`) as `lower` and phase 6 (178 rows =
  `lower_open`) as `release`.** The numbers are right and the names are wrong. This belongs in the
  TASK-057 amendment.
- **Behaviour-cloning line:** the pre-flight shows that E0's readouts depend on the image (ratios
  0.37–0.68). It does not show that the image carries information beyond proprioception on these
  cohorts. A phase-only constant beats E0 on approach palm–apple (0.10 vs 0.456 cm) and on the BC
  selection metric beats A2/A3 (0.01685 vs 0.01793/0.02340). A proprio-linear probe beats E0 on
  orient apple position (0.163 vs 0.873 cm). This bears on A4's rationale (S5-04) and on any reading
  of A2 − A0 as "vision helps or hurts".
- The TASK-058 known pattern (unconditional median, and no target std in `cloning.py`) was
  confirmed in code: `action_error:232–245` and `evaluate_policy:268` (`predictions.std(axis=0)` only).
  In §1–§13 the only claim it touches is §12's hedging prediction, which is framed as a prediction.
  Its test in §16 belongs to slice 6.

## TASK-058 audit — slice 6: TASK-056 closed loop, handover, TASK-057 cross-reference

Auditor: slice-6 worker. Repo at `8306633` (worktree, read-only). Artifacts read-only from the main
checkout: `outputs/task056-cohort-d/{a0,a1,a2,a3}.json`, `outputs/apple-policy-v1/`,
`checkpoints/task056-policy-v1/`, `data/apple-wide-v1` (labels only, no image decode).

Scratch checks (not committed; in `benchmarks/audits/task058/`): `s6_dump.py` (per-attempt dump and
re-aggregation of all 64 attempts), `s6_expert.py` / `s6_expert2.py` (labels-only reproduction of
the BC target distribution, same exclusion rules as `cloning.load_bc_split`: 137 train roots,
**68,791 rows** and 15 val roots, **7,723 rows**, both exactly matching the manifest),
`s6_step0.py` (step-zero `apple_collector_policy` vs `OracleManipulationPolicy` commands,
reconstructed from privileged labels; the reconstruction reproduces the recorded
`collector__base_action[0, 6:9]` with max error **0.0** on all 152 roots), `s6_cohortD.py` (the same
reconstruction applied to cohort D's stored reset `object_xy`), `s6_ckpt.py` (checkpoint hashes).

Code revisions. The four cohort-D reports record `source.revision = 103b14f` (`a3.json`
`dirty: true`). `103b14f` is **not an ancestor of `main`**. It exists only on
`origin/feat/task-056-clip-instrumentation`, because the PR was squash-merged as `6e728ce`.
`src/` is identical between `103b14f` and `6e728ce`. `scripts/evaluate_policy.py` differs only by
the guard-refusal catch and the `guard_refusals` counter, which is A3's uncommitted fix.
`command_statistics` **did not exist at `8956e42`**. It was added on the feature branch and is
present at `103b14f` and `6e728ce`. At both of those revisions every field it writes is computed
from `free = np.abs(commands[:, FREE_INDICES])` (`evaluate_policy.py:197` @`6e728ce`): `q50`, `q90`,
`q99`, `max_abs`, `out_of_distribution_rate` and `by_stage.translation_out_of_distribution_rate`.
**None are signed.** A signed block was only added at `4c72ffa`.

| ID | Claim (short, quoted number) | Where (file:line, all restatements) | Artifact field | Computing code (file:function:line @commit) | What it measures / population | Entities & baseline check | Class | Note / proposed correction |
|---|---|---|---|---|---|---|---|---|
| S6-01 | Cohort-D outcome table: grasp resets A0 **0/16**, A1 **0/16**, A2 **1/16**, A3 **1/16**; full successes 0/16 each; `dead_on_development` T/T/F/F; cap hits 16/16,16/16,16/16,**14/16**; guard stops 0/0/0/**2** | results `apple_policy_v1_results.md:606-611`, :748; handover `task056_handover.md:15-23`; `.mc/.../TASK-056*.md:148-151`; manifest `apple-policy-v1.json:1172-1210` (`/results/cohort_D_development`) | `a{0..3}.json` `/result/{grasp_resets,full_successes,dead_on_development,guard_refusals}` (git-ignored); transcribed into manifest `/results/cohort_D_development/arms` | `scripts/evaluate_policy.py:evaluate` (grasps += `bool(attempt["score"]["grasp"])`) @`103b14f`; `task.py:AppleToPlateTask.evaluate:86-87` latches `grasp |= reach ∧ lifted(≥0.05 m) ∧ hand_contact` | Per attempt, the latched stage at the final evaluated step, so it means "grasp at any point". Over 16 cohort-D seeds per arm. Reproduced exactly from the 64 attempts. | Arms match `checkpoint_arm_metadata`. Report `checkpoint_sha256` equals the sha256 of `checkpoints/task056-policy-v1/a?.pt`, **but none of the four policy checkpoint hashes (960dd6f0…, 6e9912b8…, 0dc2fb70…, e61e2e09…) appears in any committed file**, and the code revision `103b14f` is not on `main`. Not a gate. Hold/random would score 0/16 here. | confirmed | Provenance gap: CLAUDE.md requires checkpoint hashes in the record. The counts themselves are right. |
| S6-02 | "No attempt terminated on task failure." | results :613; manifest :1209 `no_attempt_terminated_on_task_success_or_task_failure: true` | per-attempt `termination_reason` | `evaluate_policy.py:run_attempt:243-315` @`103b14f`/`6e728ce` | `run_attempt` **has no task-failure termination path**. Its reasons are `step_limit`, `success`, `guard_refusal`, `infeasible_command`, the `execute` rejection reason, `deadline_miss` and `error`. A dropped apple does not end an attempt. | Structurally guaranteed, so it is not an observation. | mislabelled | Low weight. It is a property of the runner, not an outcome. Say so. |
| S6-03 | "Every attempt that was not stopped by the embodiment ran to the 1000-step cap"; A3 guard stops on **45001 (157 steps)** and **45003 (143)** | results :613-614, :625-626, :818; `.mc` TASK-056 :151; manifest :1208 | `attempts[*].termination_reason`, `executed_steps` | as S6-02 | 62/64 `step_limit` at 1000 executed steps. A3 45001 `guard_refusal` at 157, 45003 at 143. | — | confirmed | — |
| S6-04 | "every attempt hitting the 1000-step cap" / "Every arm hitting the cap on every attempt" | results :765-768; handover :101-102; manifest :1213 (`dz_under_shoot`: "every attempt hitting the step cap") | as S6-03 | as S6-03 | A3 is **14/16** cap hits, as §14's own table says two sections earlier. | — | wrong | Already flagged in `apple_policy_diagnostics_v1.md:15-16` but not corrected at source. The true form is "every attempt that the embodiment did not stop (62/64)". |
| S6-05 | Stage occupancy: A0/A1 16000 `none`; A2 15140/77/783; A3 12593/906/803 | results :616-623 | `attempts[*].command_statistics.by_stage[*].commands` | `evaluate_policy.py:command_statistics:211-222`; stage label from `run_attempt:258-267` | Commands labelled by the latched stage from the **previous** executed step. A3's two guard-refused commands are counted (158 + 144 commands for 157 + 143 executed steps). | — | confirmed | Grasp-stage commands are almost entirely one seed per arm (A2/45100 783, A3/45006 803). **A3 reached `reach` on 4/16 seeds (45001, 45003, 45006, 45100). A2 reached it on 1/16. No results document states this.** |
| S6-06 | Grasp on seed **45100** (A2) and **45006** (A3) | results :625; manifest; `apple_policy_diagnostics_v1.md:11-13,169-172` | `attempts[seed].score.grasp` | as S6-01 | — | — | confirmed | — |
| S6-07 | Learned-arm lift/hold: A2/45100 **+0.1678 m**, A3/45006 **+0.1694 m**, `hand_contact` True, `dropped` False at the cap | diagnostics :11-13, :169-178; TASK-057 body :44-47; TASK-058 description ("sustained contact"); absent from results and handover | `attempts[seed].score.{object_height_m,hand_contact,dropped}` (git-ignored); prose in diagnostics manifest :53 | `task.py:AppleToPlateTask.evaluate:109-117` returns the **final** step's truth. `run_attempt` stores only the last `score`. | End state at step 1000: object heights 0.934384 and 0.935996 against 0.766633. That rest height is the identical value in all untouched attempts, because `initial_height` is not stored. | — | confirmed | Confirmed **as an end state only**. The runner logs no per-step contact or height. "Held throughout" and TASK-058's "sustained contact" are **unmeasured**. The 783/803 grasp-stage commands show that the latch was set, not that contact was continuous. The handover omits the learned-arm grasp count and the lift entirely (known compression). |
| S6-08 | 2,894 numeric values in `command_statistics` across 64 attempts, **0 negatives** | TASK-058 description; diagnostics :142-150 | all `attempts[*].command_statistics` | `evaluate_policy.py:command_statistics:197` `free = np.abs(...)` @`103b14f`/`6e728ce` | All fields of all 64 blocks | — | confirmed | Recounted: 2894 / 0. |
| S6-09 | Median control time per command: A0 **12.1**, A1 **13.6**, A2 **13.6**, A3 **13.8 ms** | results :628-630 | `/result/g7_reference_only/median_control_seconds` | `evaluate_policy.py:evaluate`: `np.median` over the **per-attempt medians**; each per-step time runs from `observe` to `execute`, excluding `scorer.evaluate` | A median of 16 per-attempt medians, not a median over all commands. Per-step times are not stored. | G7 is not enforced on D. | mislabelled | Minor, and the numbers are right. Should read "median over attempts of each attempt's median control time". |
| S6-10 | Expert "`right_dz` sits at **exactly 0.400 in 44.89 %** of the **68 791** BC target commands" | results :386-387 (§12), :634-635 (§14.1), :759-760 (§16); handover :89-91; manifest :1213; `.mc` TASK-056 :172-173 | **none committed**. No field holds the train rate. The diagnostics manifest :51 has "44.886% TRAIN" in prose only. | No committed code computes it (`measure_policy_offline_conditionals.py` is val-only). Reproduced in scratch from `collector__base_action` under `cloning.load_bc_split`'s exclusions. | It is an **absolute-value** rate: \|dz\| ≥ 0.4 = **0.44887**. Signed: **+0.400 on 18.45 %**, −0.400 on 26.44 %, mean −0.00118. | "sits at exactly 0.400" reads as +0.400, and +0.400 is only 18.45 %. Row count 68,791 reproduced. | mislabelled | Not citable (§9 standard). Value reproduced. Reword as "\|`right_dz`\| = 0.400 on 44.89 % (+0.400 on 18.45 %, −0.400 on 26.44 %)". |
| S6-11 | §14.1 policy dz table: q50 0.0125/0.0544/0.0317/0.0857; q90 0.0165/0.0548/0.0343/0.0890; q99 0.3586/0.3743/0.3649/0.3131; max 0.4312/0.4136/0.6070/0.6490; OOD 0.0070/0.0011/0.0353/0.0057 | results :637-646; handover :95; manifest :1213; `.mc` TASK-056 :172-173; diagnostics :116 | `attempts[*].command_statistics.per_dimension.dz.*` (git-ignored). Medians only as manifest prose. | `command_statistics:197-208` over `np.abs`. Aggregation: median over 16 attempts of each attempt's quantile, max over attempts, OOD command-weighted (reproduced exactly). | Every column is a **\|dz\|** statistic, over all commands of each attempt. Those commands are mostly scorer stage `none`: 100 % for A0/A1, ≥ 79 % for A2/A3. | Headers "dz q50/q90/q99" name signed dz. Only "max\|·\|" says absolute. | mislabelled | Relabel the columns "\|dz\| q50" and so on. Not citable (git-ignored only). |
| S6-12 | "The dz prediction … is **borne out**"; "Every trained arm **under-shoots** it"; "Under-shooting **descent**"; "a policy commanding **a tenth of the expert's descent**" | results :758-763, :765-767; handover :95, :97-103; manifest :1213; `.mc` TASK-056 :172-174 | S6-10 + S6-11 | S6-10 + S6-11 | Compares a per-attempt **median of \|dz\|** (policy, mostly pre-reach) with a **saturation rate of \|dz\|** over all expert phases. These are different statistics over different populations, and the sign is destroyed, so "descent" is unreadable. Expert all-phase median \|dz\| = **0.1968**. In the expert's pre-reach `orient` phase (its first 130 commands), median \|dz\| = **0.0573**, \|dz\| = 0.4 on only 12.7 %, and mean dz is **+0.052** (ascending). At step zero, dz = **+0.4 on every root**. | The policy medians 0.0125–0.0857 bracket the expert's own orient-phase median. §12 (:394-396) promised the policy distribution "in the same form as the expert's (rate at maximum …, quantiles)". §14.1 gives no policy rate-at-maximum and no expert quantiles. | mislabelled | **UNMEASURED, not refuted.** The directional reading is unsupported, and even the magnitude contrast compares mismatched populations. Already recorded as unmeasured in diagnostics §1.1–§1.2 and the TASK-057 body. The source documents still carry it. |
| S6-13 | "The policies **do not approach the boundary** their demonstrations are saturated against" | results :761-762 | S6-11 | S6-11 | Holds at q50/q90 only. Every arm's **max\|dz\| 0.4136–0.6490 exceeds the expert maximum 0.400**, q99 is 0.31–0.37, and **0.11–3.53 %** of commands (A2 3.53 %) fall beyond anything demonstrated. | — | mislabelled | Stated of the distribution, but true only of its median. Say "median \|dz\| is far from the boundary; the upper tail crosses it (OOD 0.11–3.53 %)". |
| S6-14 | A3 "highest median dz (**0.0857**), highest rotation clip rate (**0.0856**), largest dz excursion (**0.6490**) … commands **the most aggressive motion**" | results :771-775 | S6-11, droll OOD | S6-11 | All three figures are \|·\|. The "rotation clip rate" is **`droll` only**. | "most aggressive" is selective. A3 has the **lowest** dz q99 (0.3131) and a dz OOD rate of 0.0057 against A2's 0.0353. | mislabelled | Keep the numbers. Name `droll` and drop "most aggressive motion", or qualify it by statistic. |
| S6-15 | droll OOD = clip rate: **0.0510 / 0.0726 / 0.0602 / 0.0856** | results :648-650 | `per_dimension.droll.out_of_distribution_rate` × commands (git-ignored) | `command_statistics:203`. Configured bound ±0.5 equals expert max 0.5 (`configured_bounds` in each report). | Command-weighted \|droll\| > 0.5 + 1e-6. Equals the per-dimension clip rate up to the 1e-6 tolerance. | — | confirmed | Not citable (git-ignored). |
| S6-16 | Pooling rationale: "rotation saturation is normal at **~27 %**"; "grasp sits at its maximum **91.78 %**" | handover :107-109; results :652-655; manifest :163; `evaluate_policy.py:190-192` | none committed (prose only) | none committed. Reproduced in scratch. | 27.06 % is **`droll` alone**. Other rotation dims: dpitch 4.19 %, dyaw 2.31 %, 11.19 % per element, 29.90 % any-of-three. 91.78 % is \|grasp\| = 1, i.e. open (−1) **or** closed (+1). | — | mislabelled | Not citable. Say "`droll` at \|max\| 27 %" and "\|grasp\| = 1.0 on 91.78 %". |
| S6-17 | Mass-at-max test: "**9×–242×** more mass exactly on the maximum than in the 5 % band below it, with **q99 = q999 = max** for every dimension" | handover :92-93; manifest :165; `evaluate_policy.py:68-71` | none committed | none committed | Reproduced: ratio 9.2× (dyaw) to 242.8× (grasp) with band [0.95·max, max), and q99 = q999 = max for all 7 dims | — | unverifiable | **Not citable** (no committed code or field). The value reproduces, so the only issue is citability. |
| S6-18 | "**1/16 against 0/16**, a single reset each … no interval that would separate it from noise"; "A0 and A1 never left stage `none`"; A2 "third of four" | results :750-756 | S6-01, S6-05, manifest `selection_score` | as S6-01/S6-05/S6-19 | — | Correctly hedged | confirmed | — |
| S6-19 | Offline selection scores **A1 0.01106 < A0 0.01313 < A2 0.01793 < A3 0.02340**; "the controls already won" | handover :80-82; results :372-374; diagnostics B5 | manifest `/training`/arms `selection_score` (e.g. :1337, :1344); diagnostics re-derivation | `cloning.action_error:232-245`: `median_abs_error = np.median(error)` over the flattened [N,7] (includes grasp) | Val split, 7,723 rows, 15 surviving roots, selected checkpoint | The manifest discloses the flattened-median reading. A single seed, no interval. | confirmed | "won" is a one-run ordering. |
| S6-20 | Handover §2: close-phase palm–apple passes P1 (**≤ 1.5 cm**); the shuffled-frame control "confirms the encoder is reading the current frame" | handover :34-37 | manifest `/results/preflight/gates/P1_close_palm_apple` (value 0.004908, control 0.013138, ratio 0.3736) | `scripts/measure_policy_preflight.py:per_phase_errors:144` → `world_model_v2.readout_rows:314` (encode → readout, h = 0); `shuffled_frame_control:199` | Directly encoded val frames, close phase, apple-dropped rows removed | Blind baseline: **the shuffled-frame control (1.31 cm) itself passes the absolute 1.5 cm threshold.** Only the ratio ≤ 0.7 excludes it. `encode()` fuses proprioception (state_fusion), so the "perception" figure includes proprioception, and the ratio addresses that. | confirmed | Sound because P1 is the conjunction. Anyone quoting "(≤ 1.5 cm)" alone quotes the blind-passable half. |
| S6-21 | Handover §2: moving-cohort persistence ratio **0.9524 at h = 1**; `held_lift_shuffled_auroc` **0.99737 at h = 1**; "a predictor that simply repeats the current position is very nearly as good" | handover :41-44; `.mc` TASK-056 :51-53 | manifest `/unverified_pending_P0/rederived_values/1/{U2_rollout_over_persistence_moving=0.95240, U3_held_lift_shuffled_auroc=0.99737}` | `measure_policy_preflight.rederive_unverified:327` → `world_model_v2.window_metrics` (ratio of **medians**: rollout palm–apple error / persistence error over moving windows; persistence = the encoded-start readout held) | TASK-054 E0, val windows at stride 4 | The gloss fits the ratio (persistence of the encoded readout). The AUROC is a shuffled-**action** rollout, not a persistence predictor. | confirmed | Minor. The gloss belongs to the 0.9524 only. The manifest also records h = 4 at **1.046** (worse than persistence), which the handover omits. |
| S6-22 | Handover §2: retracted ejection figures. Gated failures **2.70–19.10 cm**, lowest crossing **1.61 cm** | handover :51-54 | `apple_wide_grasp_closure_results_v3.md:375-386` (prose); underlying `outputs/apple-wide-object-ceiling-v2/` (git-ignored) | outside this slice | — | — | unverifiable | Consistent with its cited source. Not citable, since it rests on git-ignored traces. Construction belongs to the TASK-049/050 slice. |
| S6-23 | "**Ten** instances are recorded in full in **§7 and §8** of the results document"; "Ten recurring-lesson instances"; suite "green at **782** passed" | handover :220, :236; `.mc` TASK-056 :177-178 | results :409-585 | — | The results doc lists **eleven** instances, in **§13** ("Why the review gate sits before training", :439-585). §7/§8 are other sections. "782 passed" has no artifact. | — | wrong | Low weight. Correct to "eleven, in §13". 782 is unverifiable. |
| S6-24 | §17: A3's first run "crashed …: one violation on one seed aborted all sixteen attempts"; "the fixed runner's 16-attempt report reproduces those per-seed outcomes exactly" (unplanned cross-check) | results :787-789, :816-819; manifest `runner_defect_fixed_after_numbers_were_seen` | no artifact of the crashed run or of the 16 single-seed probes survives (only `a0..a3.json` in `outputs/task056-cohort-d/`) | `a3.json` `source.python_source_sha256` hashes `src/embodied_jepa/**.py` only (`training.source_identity:53-56`), **not `scripts/`**, so the dirty runner that produced `a3.json` is not hash-identified | — | Consistent with `a3.json` (45001 is the lowest guard seed) | unverifiable | Record that the cross-check's evidence was not retained, and that the runner's identity for `a3.json` rests on the later commit `6e728ce`. |
| S6-25 | "all four record revision `103b14f`"; "`a3.json` carries `source.dirty: true`" | results :812-814 | `source.{revision,dirty}` | `training.source_identity` | — | `103b14f` is **not an ancestor of main**, and is reachable only from `origin/feat/task-056-clip-instrumentation` | confirmed | Add that the revision survives only on a remote feature branch. If that branch is deleted, the recorded revision is unresolvable. |
| S6-26 | TASK-057 body: "an unconditional median necessarily **sits inside the non-saturated group** at **44.503 %** val saturation" | TASK-057 :37-39; diagnostics :64-67 and manifest :51, :657 | diagnostics manifest `val_saturation.dz_abs_at_max_rate = 0.44503` | `measure_policy_offline_conditionals.measure`, val rows | 44.503 % reproduced (labels-only 0.44503) | — | mislabelled | Known inherited wording (TASK-058). Correct form: with fewer than half the rows saturated, min(non-sat) ≤ median ≤ max(non-sat), i.e. the median is **bracketed by** the non-saturated rows. The verdict is unchanged. |
| S6-27 | TASK-057 body: expert `right_dz` "mean is **−0.0012**" | TASK-057 :41-42; diagnostics :54 | none committed. The val mean (−0.0041) is in the diagnostics manifest `dz_target_mean`. The **train** −0.0012 is prose only. | none committed | Train cohort, 68,791 rows. Reproduced: −0.00118. | — | unverifiable | Not citable. The value reproduces. Name the population (train). |
| S6-28 | TASK-057 body: conditional means reach **88–98 %** of the boundary; pred/target std ratio **0.952–0.980** | TASK-057 :34-36; diagnostics :79-93 | diagnostics manifest `frozen_offline_conditionals…` | `measure_policy_offline_conditionals.conditionals:128-160` (mean predicted dz on target ≥ +0.4 / ≤ −0.4 rows; `targets[:, i].std()` on the same val rows) | Val, 7,723 rows. 0.3537/0.4 = 88.4 % … 0.3903/0.4 = 97.6 %. 0.265636/0.27905 = 0.952 … 0.273488/0.27905 = 0.980. | — | confirmed | — |
| S6-29 | Entity: the "scripted expert" / **shadow expert = `OracleManipulationPolicy(sim.task_truth())`**, "exhausted after **805** commands" | TASK-057 :50-52 ("contrast against the scripted expert"); diagnostics :256-269, :302, :307-312; diagnostics manifest :126-130 | — | Demonstrations: `scripts/collect_apple_wide.py:run_root:449-468` builds `scripted.apple_collector_policy(sim.task_truth())` (`scripted.py:192-208`). That is **`EarlyReleaseOracleManipulationPolicy`** with opening_ramp 0.08, **+0.015 m palm-x offset on every phase** and **−0.035 m transfer-x shift on phases ≥ 4**. BC target: `collector__base_action` = the unperturbed `policy.action(robot)` (`Controller.command:268`). `cloning.is_surviving_root` drops aim-offset roots and all branches, so no aim offset or branch modification is in the BC target. The repo's `scripted_oracle` is also `apple_collector_policy` (`evaluate_apple.py:2171-2174`). | Collector phases: orient 130, descend 80, close 45, lift 150, **transfer 60, release_high 100, lower_open 100**, retreat 80 = **745**, not 805. `preflight` (`measure_policy_preflight.py:95-104`) uses these names. | **Wrong expert.** Step-zero effect on cohort D, reconstructed exactly from labels and reset `object_xy`. dy, dz, rotation and grasp are identical at step zero. **dx differs on 7/16 seeds by > 0.057**. On 45000, 45101 and 45105 the sign is opposite (collector 0.157 to +0.4 against Oracle −0.4). The median over 16 of \|collector − Oracle\| is **0.0479**, i.e. **84 % of A0's dx D1 threshold** (3 × 0.018958 = 0.0569) and 59 % of A2's (0.0814), **for a policy that imitates its own expert perfectly**. | wrong | The shadow expert per code should be `scripted.apple_collector_policy(sim.task_truth())` with no aim offset and no branch. Its budget is 745. Phases 4–7 differ in duration, grasp schedule (release_high opens at the transfer height) and targets. D2 and D3 references after step ~130 diverge further (reasoned). |

### Divergences needing correction

1. **S6-04** (wrong).
   - Source: `apple_policy_v1_results.md:765-768`: "compatible with every attempt exhausting the step cap … Every arm hitting the cap on every attempt".
   - Source: `task056_handover.md:101`: "every attempt hitting the 1000-step cap".
   - Source: manifest `/results/dz_under_shoot`: "every attempt hitting the step cap".
   - Proposed erratum: "every attempt the embodiment did not stop (62/64; A3 was guard-stopped on 45001 and 45003) ran to the 1000-step cap".
2. **S6-10** (mislabelled).
   - Source: results :386-387, :634-635, :759-760; handover :89-90; manifest :1213; `.mc` TASK-056 :172-173: "`right_dz` sits at exactly 0.400 in 44.89 %".
   - Proposed erratum: "|`right_dz`| = 0.400 on 44.89 % of the 68 791 BC target commands (+0.400 on 18.45 %, −0.400 on 26.44 %, mean −0.0012). The figure is an absolute-value rate and is not citable: no committed code computes it on the train cohort."
3. **S6-11** (mislabelled).
   - Source: results :637 table header "dz q50 | dz q90 | dz q99".
   - Proposed erratum: "|dz| q50 | |dz| q90 | |dz| q99". Add: "every statistic in this table is computed from `np.abs` (`evaluate_policy.py:197`); no sign survives".
4. **S6-12** (mislabelled, UNMEASURED).
   - Source: results :758: "The dz prediction was made in advance and is borne out, at that strength and no further."
   - Source: results :765-766: "Under-shooting descent … a policy commanding a tenth of the expert's descent".
   - Source: handover :95: "Every trained arm under-shoots it."
   - Source: manifest `dz_under_shoot`: "borne out"; `.mc` TASK-056 :172: "predicted in advance and borne out".
   - Proposed erratum: "**Withdrawn as unmeasured.** The reported statistics are per-attempt medians of |dz|. They carry no sign, so 'descent' and 'under-shoot' cannot be read from them. They were also compared with a saturation rate over all expert phases rather than with the same statistic over the matching phase. The expert's own median |dz| in `orient`, the phase before reach, is 0.057, inside the policies' 0.013–0.086. The §12 prediction is neither borne out nor refuted by TASK-056's record (see `apple_policy_diagnostics_v1.md` §1.1–§1.2)."
5. **S6-13** (mislabelled).
   - Source: results :761-762: "The policies do not approach the boundary their demonstrations are saturated against".
   - Proposed erratum: "The policies' median |dz| is far from 0.400. Their upper tail crosses it: max |dz| 0.41–0.65 for every arm, and 0.11–3.53 % of commands exceed the expert maximum."
6. **S6-14** (mislabelled).
   - Source: results :771-774: "the highest rotation clip rate (0.0856) … The fine-tuned arm commands the most aggressive motion".
   - Proposed erratum: "the highest `droll` clip rate (0.0856) … A3 has the largest |dz| median and excursion but the lowest |dz| q99 (0.3131) and a lower dz OOD rate than A2 (0.0057 vs 0.0353), so 'most aggressive' holds only per statistic."
7. **S6-16** (mislabelled).
   - Source: handover :108-109 and manifest :163: "rotation saturation is normal at ~27 %, and grasp sits at its maximum 91.78 %".
   - Proposed erratum: "`droll` saturation is normal at ~27 % (dpitch 4 %, dyaw 2 %), and |grasp| = 1.0 on 91.78 % (open or closed). Neither figure is citable from committed code."
8. **S6-09** (mislabelled, minor).
   - Source: results :628: "Median control time per command".
   - Proposed erratum: "Median over the 16 attempts of each attempt's median control time (observe → execute, scorer excluded)".
9. **S6-02** (mislabelled, minor).
   - Source: results :613: "No attempt terminated on task failure."
   - Proposed erratum: append "(the runner has no task-failure termination; this is structural, not observed)".
10. **S6-23** (wrong, minor).
    - Source: handover :220: "Ten instances are recorded in full in §7 and §8 of the results document."
    - Source: `.mc` TASK-056 :177: "Ten recurring-lesson instances".
    - Proposed erratum: "Eleven instances are recorded in §13 of the results document".
11. **S6-26** (mislabelled, known inherited).
    - Source: TASK-057 :37-39; diagnostics :66-67; diagnostics manifest :51, :657: "necessarily sits inside the non-saturated group".
    - Proposed erratum: "is necessarily bracketed by the non-saturated rows, min(non-sat) ≤ median ≤ max(non-sat)".
    - This is an amendment to a frozen prereg, handled by the diagnostics owner.
12. **S6-29** (wrong entity). Handled by the agent that owns the diagnostics doc.
    - Source: diagnostics :256: "`OracleManipulationPolicy(sim.task_truth()).action(robot)`".
    - Source: diagnostics :260-261 and manifest :130: "exhausted after 805 commands … 130 + 80 + 45 + 150 + 160 + 100 + 60 + 80".
    - Proposed: "`scripted.apple_collector_policy(sim.task_truth())` (the collector that generated `collector__base_action`, and the repo's `scripted_oracle`). Its budget is 130 + 80 + 45 + 150 + 60 + 100 + 100 + 80 = 745."
13. **Not-citable rows** (S6-10, S6-16, S6-17, S6-22, S6-24, S6-27; also S6-11 and S6-15, which exist only in git-ignored reports).
    - Record them as not citable. The expert-corpus figures would become citable if a committed train-split variant of `measure_policy_offline_conditionals.py` (labels only, no image decode, about a minute) wrote them to a manifest.
14. **Provenance** (S6-01, S6-25).
    - Add the four policy checkpoint sha256s to the manifest:
      - a0 `960dd6f094b2…`
      - a1 `6e9912b82cbb…`
      - a2 `0dc2fb701862…`
      - a3 `e61e2e090c31…`
    - Note that report revision `103b14f` is off-main.

### Cross-references (TASK-057 / `apple_policy_diagnostics_v1.md`, read-only; behaviour-cloning line)

**Which expert generated the demonstrations (established from code).**
- `scripts/collect_apple_wide.py:run_root` uses `scripted.apple_collector_policy`, i.e. `EarlyReleaseOracleManipulationPolicy(opening_ramp=0.08)` with palm_x_offset +0.015 m on all phases and transfer_x_shift −0.035 m on phases ≥ 4.
  - Each fifth root additionally has an aim offset (`shifted(policy, (0,1,2,3), aim)`).
  - Branches modify the base policy.
  - The BC cohort (`cloning.is_surviving_root`) excludes both, so the BC target is the unmodified `apple_collector_policy` command.
- **The shadow expert must be `apple_collector_policy(sim.task_truth())`, not `OracleManipulationPolicy`.**
  - Its budget is **745**, not 805.
  - Its phases 4–7 are `transfer`(60), `release_high`(100, opening ramp), `lower_open`(100), `retreat`(80).
  - This matches the repo's own `scripted_oracle` (`evaluate_apple.py:2171-2174`), which B1/B3's 24/24 reference was measured on.

**Diagnostics premises and their standing:**

- **§1.1(a).** The numbers are right: I reproduced 44.887 %, +0.4 on 18.45 %, −0.4 on 26.44 %, mean −0.00118, median |dz| 0.1968, 68,791 rows. The train figures are **not citable**, because no committed code computes the train cohort (S6-10, S6-27).
- **§1.1(b).** "sits inside the non-saturated group" is inherited wording (S6-26). The val conditionals are confirmed (S6-28).
- **§1.1(c) and §1.5 rest on a non-confirmed reading.** "The closed-loop translation commands are near-constant", and both hypotheses' shared prediction of "a near-constant translation command … the robot barely moves", are read off **|·| quantiles** (q90/q50 ≈ 1).
  - A constant |dx| is equally consistent with a sign-alternating (chattering) command.
  - So "near-constant command" is **unmeasured**. Only "near-constant magnitude" is measured.
  - This is the same class of error §1.2 names. The source is git-ignored only (not citable).
  - `droll` OOD "5–7 %" matches a median-over-attempts aggregation (5.1–7.3 %). Command-weighted, as in results §14.1, it is 5.1–8.6 %. The aggregation should be named.
- **§1.3.** Confirmed as end state only (S6-07). "held … when the step cap ended" is correct, but nothing in the record measures continuity of contact.
- **§1.4.** Confirmed: A0/A1 reach 0/32; A3 reach 4/16; A2 1/16; A3 five distinct final heights. "~210 commands" to approach holds for either expert.
- **§2 / §2.1 shadow-expert definition.** Wrong entity (S6-29).
  - The 805 exhaustion step, and the B2 void argument built on it, should use 745. The B2 fix (keying on the substitution set) is still needed.
  - "full … 405 inside 805" still holds for 745.
- **§5 D1.** The thresholds are 3 × Table A, and Table A is error against `apple_collector_policy`'s commands.
  - With the Oracle as reference, a perfect imitator's step-zero dx departure on cohort D is already 0.048 (median over 16).
  - That consumes 84 % of A0's and 59 % of A2's dx threshold, and 7/16 seeds individually exceed A0's threshold.
  - So D1 can fail, and the abandonment clause can fire, **by the wrong route**. This is reconstructed from labels, exact on 152 training roots, and assumes the reset `object_xy` is the apple's world xy.
  - dy/dz/rotation/grasp are unaffected at step zero, so D1-grasp (−1.0 reference) is valid under either expert.
- **D2/D3/B3.** Under the Oracle, the "full" substitution drives a controller that aims 1.5 cm behind the trained grasp point, transfers for 160 not 60 steps, and never runs `release_high`.
  - B3's ≥ 14/16 grasp threshold is justified by 24/24 for `apple_collector_policy`, a different controller. That is reasoned, not computed.
  - D2 departure is measured against a reference the policy was never trained on.
- **§9 Table E.** `scripts/measure_policy_offline_conditionals.py:PHASES` uses Oracle phase names. The rows labelled **`lower` (699 rows) and `release` (178)** are the collector's phase 5 **`release_high`** and phase 6 **`lower_open`** (val phase-5 count 699 and phase-6 count 178, reproduced). They are mislabelled.
- **Table A (§9).** The values are consistent with the BC target (the collector). The defect is only that D1 pairs them with a different expert.

**Behaviour-cloning line.**
- The directional dz mechanism (S6-12) should not be carried forward as a candidate cause. Its magnitude comparison was also population-mismatched: the expert's own pre-reach |dz| is as small as the policies'.
- The honest TASK-056 closed-loop signal is:
  - A0/A1 never reached;
  - A3 reached on 4/16 but guard-stopped twice;
  - A2/A3 each grasped, lifted ~17 cm and held to the cap on one seed;
  - no transport.
- The policy checkpoint hashes should be committed before any further citation of these runs.
