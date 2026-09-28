# apple-to-plate-v2: the v2 scene and a preregistered at-rest expert gate (TASK-070)

**Status (2026-09-28): gated run complete. Outcome: PASS** (§8).
- At rest: 32/32 at plate exact and 30/32 at 1.0 cm, on the fresh gated seeds 50600–50631.
- Privileged scripted expert, under the v2 physics. The expert (e9) was selected on the
  development seeds.

§1–§7 are the preregistration, merged in PR #92 and unchanged since.

**Scope.**
- This is privileged scripted engineering: the expert reads simulator truth. It is not a learned
  result, and no learned policy is trained or run here.
- No corpus is read, and the test split is not decoded.
- **Learned Apple→Plate is still 0 successes.**
- The v1 task, its scorer, its history and the 0/150 MVP benchmark are untouched.
  `simulation.py`, `task.py` and `scripted.py` are byte-identical to `main`.

## 0. Owner ruling R12 (verbatim, received 2026-09-28T04:11Z UTC via the coordinator)

> #90 is merged as faf7932, a squash merge. #91 now targets main but still carries #90's pre-squash commits.
>
> Owner ruling R12, 2026-09-28T04:11Z UTC. Record it verbatim.
>
> 1. **Rebase #91** onto main so that only the TASK-069 commits remain, and keep the tag task069-dev-evidence-pre-merge at 4599296 for the cited SHAs. Have a reviewer in its own temporary worktree confirm the rebase is content-neutral against 4599296. Then send me the new head and the verdict. I'll do a rebase merge as you asked, and you add the old→new SHA map in the TASK-070 PR, as you did for #89.
>
> 2. **apple-to-plate-v2 = v1 + apple contact condim 6, and nothing else.** b2 and b1 are rejected. The plates stay as in v1.
>
> 3. **The friction values are NOT chosen from the scan.** b3b passed the bar, but choosing it for that reason would tune the environment to the expert. Your own caveat says it is unsourced and makes the apple "effectively stuck", which is too forgiving. So:
>    - (i) The default is the scene's own declared values (b3a: rolling 0.001, torsional 0.01). They were declared before any scan.
>    - (ii) You may replace them only with values backed by a cited physical source, such as rolling resistance for fruit or soft spheres on ceramic or a hard surface. The source must be quoted in the protocol and fixed before any further runs. If sources give a range, use its midpoint, not the value that passes.
>    - (iii) No other values may be tried under v2.
>
> 4. **TASK-070, which you own, runs the v2 expert gate.**
>    - Add the v2 scene option in code as a separately named, versioned task or scene. v1 code and history stay untouched.
>    - Improve the expert on a fresh dev seed range under the fixed v2 physics. d12 is the starting point; log every design.
>    - Then open a preregistration PR. It freezes: the expert, v2 physics with its source, apple_at_rest_v0, and fresh gated seeds.
>      - The gate is at rest ≥ 28/32 at plate exact AND ≥ 28/32 at 1.0 cm.
>      - Report the latched scorer and the 1.5 cm error beside it.
>    - Get it reviewed. I merge it.
>    - The gated run starts only on the pre-run reviewer's reported GO.
>    - Then a results PR.
>
> 5. If the v2 expert passes, the next task trains the first learned policy on its demonstrations. That is the product goal.
>
> Rules, verbatim:
> 1. "A gated run starts only on the pre-run reviewer's REPORTED verdict, delivered as a message — never on a review file read from disk. Same rule for merging: only on the reviewer's reported APPROVE."
> 2. "One task, one agent. Do not launch a closeout/follow-up agent while the original task agent can still resume."
> 3. Reviewers use their own temporary worktree, never yours.
>
> Do not merge. Learned Apple→Plate is still 0 successes. Keep restating that.
>
> Address this before completing your current task.

The coordinator then relayed, before any TASK-070 episode ran:

> Your choice of R12 §3(i) (1 .01 .001, with no source guessed) is accepted.

## 1. Provenance of PR #91 (R12 §1)

PR #91 (TASK-069) was rebased onto `main` and then rebase-merged. Its SHAs were rewritten twice.
- The tag `task069-dev-evidence-pre-merge` (at `4599296`) keeps the SHAs that
  `apple_to_plate_v2_feasibility.md` cites reachable.
- In each row below, the three commits have the **identical git tree**. This was checked with
  `git log --format='%h %T'` on the tag, on `21b5810` and on `main`.

| cited in the TASK-069 doc (tag) | after the R12 rebase | on `main` | tree |
|---|---|---|---|
| `a0f922b` | `d57e9e8` | `552ebd2` | `9abcfd9e…` |
| `d9920f1` | `a23f293` | `0ad8f2a` | `4ab6ae1c…` |
| `13dcef8` | `7ad873d` | `95b9992` | `d6d804a7…` |
| `1fbd43d` | `c91a35e` | `f237df7` | `a229da6d…` |
| `216f41a` | `8371d66` | `d9c2b6e` | `d41516af…` |
| `4599296` | `21b5810` | `d545ee0` | `0943d528…` |

## 2. The v2 task (R12 §2–§3)

**apple-to-plate-v2 = v1 plus the apple's contact at condim 6, with nothing else changed.**
- **Source of the values** (R12 §3 (i)).
  - The friction is the v1 scene's own declaration, `friction="1 .01 .001"` on `apple_geom`
    (`simulation._scene`): sliding 1, torsional 0.01 m, rolling 0.001 m. It was committed long
    before TASK-068 and TASK-069.
  - v1 leaves `condim` at MuJoCo's default 3, where only sliding friction acts. v2 sets condim 6,
    so the declared torsional and rolling coefficients act.
  - No other source was sought or used (R12 §3 (ii)). No other value is tried under v2 (R12 §3
    (iii)).
- **Code.** `src/embodied_jepa/apple_to_plate_v2.py`, scene version `apple_to_plate_v2`.
  - `apply_v2_scene` edits one simulator instance's compiled model at run time.
  - It first refuses any model whose apple is not the v1 apple: condim 3, friction
    `1 / 0.01 / 0.001`.
  - Only the apple geom changes. MuJoCo uses the larger condim of two contacting geoms, and the
    element-wise maximum friction.
  - Tests: `tests/test_apple_to_plate_v2.py`.
- **Unchanged from v1:**
  - the robot, the fixed pelvis, the action schema `ee_delta_grasp_v0`;
  - the plate geometry and the TASK-047 wide-jitter reset distribution;
  - the 4 cm radius and the look.
- **Success is `apple_at_rest_v0`** (`at_rest.py`, TASK-068). On each of the last 20 steps of a
  60-step settle (arm still, hands open), the apple must be:
  - within 4 cm of the plate centre;
  - supported within 1.2 cm;
  - moving at ≤ 0.001 m/s;
  - out of contact with the hand.

## 3. Seeds

- **Development: 50200–50299** (`apple_to_plate_v2.DEV_SEEDS`), with plate-error directions
  from `default_rng(6850)`. Both were declared on the TASK-070 card before any TASK-070 episode.
  - Used so far: 50200–50231, 50232–50295, and 50298–50299 for the runner smoke test.
- **Gated: 50600–50631** (`GATE_SEEDS`, 32 seeds), with plate-error directions from
  `default_rng(6860)`.
  - Fresh: never simulated before the gated run.
  - `check_gate_seeds()` refuses any overlap with TASK-068/069/070 development seeds (50000–50299),
    `first_policy.FORBIDDEN_RANGES`, the 46000–46999 block and cohorts C and D.
- **The 2026-09-28 search.** A grep of `src`, `scripts`, `tests`, `configs`,
  `benchmarks/manifests`, `docs` and `.mc/tasks` for 50600–50631 and 6860 found only hash digits
  and decimals.
- **50500–50531 is avoided.** It was named, and not used, as an example range in
  `apple_resting_expert_v1.md` §6 (a).

## 4. Development log (every design, every count)

**Setup.**
- Script: `scripts/develop_v2_expert.py`, every design in its `DESIGNS` dict, on the v2 physics
  only.
- The expert is `resting_expert.RestingPlaceExpert`: the collector's pick, then a reachable
  release pose, a steady hold, a slow opening, a clear hold and a retreat (TASK-068 §5).
- Each attempt is TASK-068's harness (`resting_expert.run_attempt`): the wide-jitter reset, the
  look, the expert (≤ 740 commands), then the 60-step settle.
- CPU, 8 workers. Every run had a clean tracked tree and 0 errors.
- Artifacts are in `outputs/task070-dev/<design>-run-<n>/report.json` (git-ignored).

"Final" is the q10 / q50 / q90 distance from the plate centre at the end, in cm. "Land v" (the
plate-exact cell's value) is
the median horizontal speed at first plate-base contact, in m/s.

| design | parameters (vs. RestingPlaceExpert defaults) | revision | seeds | at rest: exact / 1.0 cm / 1.5 cm | latched: exact / 1.0 / 1.5 | final at 1.0 cm | land v | sha256 |
|---|---|---|---|---|---|---|---|---|
| e1 (= TASK-068 d12) | pitch 0.45 rad | `a008163` | 50200–231 | 31 / 29 / 23 of 32 | 32 / 32 / 32 | 2.91 / 3.36 / 3.90 | 0.076 | `2934fa80` |
| e2 | pitch 0.55 | `8714750` | 50200–231 | 24 / 22 / 19 | 32 / 32 / 32 | 1.55 / 3.09 / 4.54 | 0.016 | `a8c9e286` |
| e3 | pitch 0.6 | `8714750` | 50200–231 | 19 / 18 / 21 | 32 / 31 / 30 | 3.30 / 3.93 / 4.29 | 0.039 | `396eca3b` |
| e4 | pitch 0.45, release 2 cm further back (dx −0.035) | `8714750` | 50200–231 | 22 / 24 / 21 | 32 / 32 / 32 | 3.06 / 3.63 / 4.21 | 0.060 | `7e43518b` |
| e5 | pitch 0.45, release 1 cm further forward (dx −0.005) | `8714750` | 50200–231 | **32 / 31 / 27** | 32 / 32 / 32 | 2.57 / 3.25 / 3.84 | 0.075 | `736b5316` |
| e6 | pitch 0.45, opening ramp 0.02 | `8714750` | 50200–231 | 17 / 8 / 12 | 32 / 32 / 32 | 3.70 / 4.13 / 4.50 | 0.047 | `d1a3afcb` |
| e7 | pitch 0.5 | `8714750` | 50200–231 | 14 / 14 / 17 | 32 / 32 / 32 | 3.60 / 4.10 / 4.42 | 0.044 | `67d4561b` |
| e8 | pitch 0.45, dx +0.005 | `daee1f0` | 50200–231 | 31 / 30 / 25 | 32 / 32 / 32 | 2.82 / 3.16 / 3.86 | 0.080 | `8503b1be` |
| e9 | pitch 0.45, dx +0.015 | `daee1f0` | 50200–231 | **32 / 30 / 25** | 32 / 32 / 31 | 2.62 / 3.21 / 3.85 | 0.081 | `a8a91d83` |

**Held-back check.** e1, e5 and e9 were then run on the untouched development seeds 50232–50295
(64 each). e10 and e11 went straight to the held-back seeds.

| design | parameters | revision | at rest: exact / 1.0 cm / 1.5 cm (of 64) | latched | final at 1.0 cm | sha256 |
|---|---|---|---|---|---|---|
| e1 | pitch 0.45 | `daee1f0` | 64 / **54** / 45 | 64 / 64 / 64 | 2.82 / 3.38 / 4.09 | `e27cc857` |
| e5 | pitch 0.45, dx −0.005 | `daee1f0` | 64 / **56** / 54 | 64 / 64 / 64 | 2.73 / 3.39 / 4.06 | `ef39838d` |
| **e9** | pitch 0.45, dx +0.015 | `daee1f0` | 64 / **59** / 57 | 64 / 63 / 64 | 2.39 / 3.29 / 3.85 | `01eb5641` |
| e10 | pitch 0.45, dx +0.025 | `8e1ca80` | 64 / **59** / 58 | 64 / 63 / 64 | 2.66 / 3.29 / 3.94 | `8759deff` |
| e11 | pitch 0.40, dx +0.015 | `8e1ca80` | 64 / **56** / 46 | 64 / 60 / 56 | 1.91 / 2.78 / 4.05 | `2587a248` |

There were no guard stops in any TASK-070 development run.

**What the development logs show.**
- **Where the apple ends.** Under v2 the released apple lands near the plate centre. In every e1
  attempt it touches the rim after the opening, then ends at a median 3.2–3.4 cm from the centre
  at plate exact and 1.0 cm (3.5–3.6 cm at 1.5 cm)
  (development cells of e1, e5, e8, e9).
- **Slower landings did not help.** e2, e3, e6 and e7 land slower, but end nearer the rim and
  rest less often.
- **Shifting the release forward helped slightly.** On the held-back seeds, e9 is at 59/64 at
  1.0 cm, against 54/64 for e1.
- **e10 mostly repeats e9.** On the held-back seeds, 170 of its 192 release targets are
  identical to e9's (the reach sphere pulls the release back in x), and 191 of 192 at-rest
  outcomes agree.

**The frozen expert is e9**, `RestingPlaceExpert(release_pitch_rad=0.45, release_dx=0.015)`.
- **Reasons:** it has the best held-back count at 1.0 cm (tied with e10, which is nearly the
  same expert and was run only on the held-back seeds), and 30/32 on its development block.
- **Its development record at 1.0 cm plate error is 89/96, and 96/96 at plate exact.**
- **An estimate, not a promise.** A per-attempt rate of 0.92–0.93 at 1.0 cm (59/64 held-back,
  89/96 overall) gives roughly a 90 % chance of ≥ 28/32 on 32 fresh seeds (binomial); at 0.90
  it would be about 79 %. The gate can fail.
- **This is not independent evidence.** The design was chosen on these development seeds, so
  they are not a check of the chosen design.

## 5. The frozen gate

**Everything below is pinned by `benchmarks/manifests/apple-to-plate-v2-expert-gate-v1.json`.**
- That manifest holds the sha256 of 17 files the run depends on: the v2 module, the expert and
  harness, `at_rest.py`, the v1 scene, the task, the embodiment, the runtime, the look, the reset
  script, the gate runner, the action config and the planner bounds config.
- The runner refuses to start if any pin differs.

| item | frozen value |
|---|---|
| scene | `apple_to_plate_v2`: apple condim 6, friction 1 / 0.01 / 0.001 |
| expert | `RestingPlaceExpert(release_pitch_rad=0.45, release_dx=0.015)` (e9), ≤ 740 commands after the look |
| attempt | wide-jitter reset (`scripts/evaluate_apple.py:wide_reset`), the look, the expert with the believed plate xy shifted by the level in a fixed direction per seed, then a 60-step settle |
| plate error | TASK-067's C0 perturbation (`first_policy_runtime.perturbed_truth`); directions from `default_rng(6860)` |
| seeds | 50600–50631 (32), each run at 0, 1.0 and 1.5 cm (96 attempts) |
| success | `apple_at_rest_v0` (`at_rest.AtRestThresholds` defaults) |
| **gate** | **at rest ≥ 28/32 at plate exact AND ≥ 28/32 at 1.0 cm → PASS; otherwise FAIL** |
| reported, not gated | at rest at 1.5 cm; the latched v1 scorer (per-step `AppleToPlateTask` success at any step) at each level; guard stops; landing speed; final distance |
| device | CPU, 8 spawned workers; MuJoCo 3.13.0 (recorded in the manifest) |
| guard and other stops | a joint-velocity stop (refused at projection or rejected at execution), an infeasible command or any other early stop ends that attempt as not at rest; it is a failure, not a void |

**Rows** (`apple_to_plate_v2.gate_row`, tested):
- **VOID** if any attempt raised an exception, or a gated cell does not have 32 attempts.
- **PASS** if at rest is ≥ 28 at 0 cm and ≥ 28 at 1.0 cm.
- **FAIL** otherwise.
- The 1.5 cm level never decides the row.

**Void rule.** A first VOID allows one from-scratch repeat, with the same seeds, the same caps
and the same device, into a new output directory. A second VOID makes the result
**INCONCLUSIVE**.
- **A run that ends without a `report.json`** (a crash outside an attempt, or a hang stopped by
  hand) is a VOID.
- **A known risk, stated before the run.** The simulation is deterministic, so an exception that
  the expert itself raises would recur in the repeat and give INCONCLUSIVE, not FAIL. There were
  0 exceptions in 1088 development attempts. The rule is not changed here.

**No second chance.** No expert or physics change and no seed change follow a FAIL within
TASK-070. A FAIL is reported plainly, and the next step is the owner's.

## 6. How the gated run is made (only on the pre-run reviewer's reported GO)

```sh
uv run --no-sync python scripts/run_v2_expert_gate.py --output outputs/task070-gate/run-1
```

The runner (`scripts/run_v2_expert_gate.py`) refuses to start on any of:
- an existing output directory;
- a dirty tracked tree;
- any pinned sha256 mismatch;
- a gated-seed overlap.

It records the revision at start and at end, the pins, the device, the worker count and the wall
time. Its `--smoke` mode runs 2 development seeds (50298–50299) with no verdict. It was run
twice before this preregistration, to check the runner, and neither run read a gated seed:
- with an earlier version of the runner, before the MuJoCo-version check was added;
- at `48c2df3` on a clean tree: 6 attempts, 0 errors, and 2 / 2 / 0 at rest at 0 / 1.0 /
  1.5 cm.

Both runs' outputs are in the author's scratch directory, not under `outputs/`.

**The pre-run reviewer checks, in their own worktree:**
- that `main` holds this preregistration;
- that the pins match;
- that the tree is clean;
- that the smoke mode runs.

They then report GO or NO-GO as a message. The run starts only on a reported GO.

## 7. Reproduction

```sh
uv run --no-sync pytest tests/test_apple_to_plate_v2.py tests/test_at_rest.py tests/test_resting_expert.py
uv run --no-sync python scripts/develop_v2_expert.py --design e9 --count 32 --levels 0,1.0,1.5 \
    --output outputs/task070-dev/<new>
```

## 8. Results

*§1–§7 above are unchanged since the preregistration merged.*

### 8.1 Outcome: **PASS**

On the 32 gated seeds, 50600–50631, the frozen expert e9 under the v2 physics left the apple at
rest (`apple_at_rest_v0`) on:
- **32/32 at plate exact**;
- **30/32 at 1.0 cm plate error**.

Both meet the bar of ≥ 28/32. The row is **PASS** (`gate_row`).

**Read with these qualifiers:**
- **e9 was selected on the development seeds** (§4). Only these 32 gated seeds are independent
  evidence for it.
- **This is a privileged scripted expert:** it reads the reset truth of the apple and plate,
  perturbed by the plate error. It is not a learned result.
- **It is v2, not v1.** v2 is the v1 task with the apple's contact at condim 6 (R12). The v1
  task, and its 0/150 benchmark, are unchanged and separate.
- **The sample is small.** The 1.0 cm level passed with 2 failures against a limit of 4.
- **Learned Apple→Plate is still 0 successes.**

### 8.2 Provenance

- **Pre-run review.** The reviewer worked in its own temporary worktree at `1ba0557` and
  reported **GO**. It had checked all 17 pins, MuJoCo 3.13.0, the physics and seed
  disjointness, and that no gated seed had been run. It also ran its own smoke on development
  seeds (6 attempts, 0 errors) and the tests. The run started only after that reported GO.
- **The run.** `uv run --no-sync python scripts/run_v2_expert_gate.py --output
  outputs/task070-gate/run-1`.
  - It ran from a clean detached checkout of `main` at
    **`1ba05578c8da883218d390d733b8ff396cd473d4`**. That is recorded as both `revision` and
    `revision_at_end`, with `tracked_tree_dirty: false`.
  - The runner's pin check passed, including MuJoCo 3.13.0.
- **Environment.** CPU, 8 spawned workers, 301 s wall time. Python 3.12.13, numpy 2.5.3,
  macOS-26.5.1-arm64.
- **Report.** `outputs/task070-gate/run-1/report.json` (git-ignored), sha256
  `27543757f0f1d7d3a00b2f6b58a7e394d526342d14a407c018a0a27816ad099f`. Per-attempt logs are in
  `attempts/*.npz`.
- **One run, no void.** Run-1 had 96 attempts, 0 errors, and 0 guard or other early stops. No
  repeat was needed or made.

### 8.3 Numbers

| plate error | **at rest (gated)** | latched v1 scorer (reported) | guard / other early stops | final distance, cm (q10 / q50 / q90) | landing speed, m/s (q10 / q50 / q90) |
|---|---|---|---|---|---|
| 0 cm | **32 / 32** | 32 / 32 | 0 / 0 | 2.85 / 3.37 / 3.71 | 0.054 / 0.081 / 0.082 |
| 1.0 cm | **30 / 32** | 32 / 32 | 0 / 0 | 2.43 / 3.11 / 3.80 | 0.060 / 0.078 / 0.085 |
| 1.5 cm (reported only) | 28 / 32 | 32 / 32 | 0 / 0 | 2.61 / 3.30 / 4.26 | 0.059 / 0.076 / 0.087 |

**What the logs show.**
- **All 6 attempts that are not at rest fail on the 4 cm radius only.** In each, the apple is
  still, supported and free of the hand in the final window, but it rests 4.02–4.61 cm from the
  plate centre:
  - at 1.0 cm: seeds 50619 (4.07 cm) and 50628 (4.02 cm);
  - at 1.5 cm: seeds 50616, 50622, 50626 and 50627 (4.29–4.61 cm).
- **The latched scorer is 32/32 at every level.** It counts the transient crossings that
  TASK-067 identified, so it again overstates the at-rest count at 1.0 and 1.5 cm.
- **The gated numbers are consistent with the development record.** e9 had 89/96 at rest at 1.0 cm in
  development, and 30/32 here.

### 8.4 What this does and does not show

**It shows** that, under the owner-defined v2 physics, a privileged scripted expert can leave the
apple at rest on the plate on at least 28 of 32 fresh resets, both at plate exact and with
1.0 cm of plate-position error. That is the precondition R12 §5 set for training the first
learned policy on its demonstrations.

**It does not show:**
- anything learned;
- anything about v1;
- robustness beyond 32 seeds and these plate-error levels;
- physical validity of v2's friction values. They are the scene's declared values, not measured
  apple data.

**Next (the owner's call, R12 §5):** a task that trains the first learned policy on this
expert's v2 demonstrations. **Learned Apple→Plate is still 0 successes.**

### 8.5 PR #92 SHA map

PR #92 was rebase-merged, so its SHAs were rewritten. The tag `task070-prereg-pre-merge` (at
`80d3309`) keeps every pre-merge SHA of PR #92 reachable (§3–§6 cite five of them). Each pair below has the **identical git
tree**, checked with `git log --format='%h %T'`.

| pre-merge (tag) | on `main` | tree |
|---|---|---|
| `a008163` | `134265b` | `29497e10…` |
| `8714750` | `38658d0` | `c8330739…` |
| `daee1f0` | `4d04179` | `20f7eacc…` |
| `8e1ca80` | `e3bd77a` | `b350c74b…` |
| `48c2df3` | `5ebbb72` | `4b47a27c…` |
| `f893cd2` | `8868221` | `be025833…` |
| `80d3309` | `1ba0557` | `963b269f…` |

The development runs recorded their revisions as the pre-merge SHAs. The gated run recorded
`1ba0557`, which is on `main`.

**Learned Apple→Plate is still 0 successes.**
