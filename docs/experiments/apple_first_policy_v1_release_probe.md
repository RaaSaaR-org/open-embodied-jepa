# Apple→Plate first learned policy v1: release-point probe (TASK-067, owner ruling R8)

**Status: diagnostic probe, declared before it runs (§1–§4 are written first).** No gate. It is
scripted and privileged only: no learned policy, no corpus read, no test-split decoding.
**Learned Apple→Plate is still 0 successes.**

## 0. Owner ruling R8 (verbatim, received 2026-09-27T23:36Z UTC via the coordinator)

> TASK-067 owner ruling R8, 2026-09-27T23:36Z UTC: option (a), probe first. PR #83 is merged as
> c8a9263.
>
> Record R8 verbatim in the next PR. Here is the ruling:
>
> 1. **Probe (diagnostic, disclosed, no gate).**
>    - Before anything else, record the apple's landing position relative to the plate centre on
>      fresh calibration seeds.
>    - Choose the seeds and write them into the PR before running. They must be disjoint from:
>      - 46000–46415, already spent;
>      - every seed allocated to cohort D, cohort C, the smoke range 46900–46999, and any future
>        v2 calibration.
>    - Record positions under the reference and under plate errors of 1.0 and 1.5 cm, for two
>      arms:
>      - the current collector;
>      - a candidate collector that releases at the plate centre.
>    - Keep the probe small (≤ 32 resets per condition) and privileged/scripted only. No learned
>      policy, and no test-split decoding.
>    - Report the landing-position distributions and success counts. The probe confirms or
>      refutes the "release is 3.5 cm short" mechanism.
> 2. **Stop conditions after the probe:**
>    - If the candidate collector does not reach ≥ 28/32 at 1.0 cm plate error in the probe, stop
>      and report back.
>    - If the mechanism is refuted, stop and report back.
>    - In both cases, do not go on to (b), and do not change any bar.
> 3. **Amendment PR, if the probe supports the fix.** Open a preregistered v2 amendment PR:
>    - The collector fix, as a code change with tests.
>    - A new v2 corpus, collected with the fixed expert. Record its hashes.
>    - The open-loop replay baseline regenerated on the new expert.
>    - Recalibration on fresh seeds, with the same calibration levels and the same ≥ 28/32 bar.
>    - Everything else frozen as in v1, including R1–R7, the 36,000 s budget, MPS and the cohorts'
>      definitions.
>    - Nothing may be tuned on the probe seeds' outcomes beyond the release-point choice itself.
>      Disclose this.
>    - It goes through independent review. I merge it once the reviewer reports APPROVE.
> 4. **The gated v2 run** starts only on the pre-run reviewer's reported GO, and that reviewer
>    re-runs the render check. Then PR 3, as before.

## 1. Disclosed before running: the committed record already bears on the mechanism

The "release is 3.5 cm short" mechanism was the results document's inference (§4.2), derived from
code constants. It assumed the held apple lands where it is released. The committed record says
otherwise. That record is TASK-031's landing refinement
([`apple_mechanics_probe.md`](apple_mechanics_probe.md) §"V1 landing results"): an earlier,
narrow-jitter distribution with two resets per setting.
- With no transfer shift, the apples landed about **+3.6 to +3.7 cm** from the plate centre in X
  (the V0 note says "+0.036m/+0.039m in X").
- With the selected −0.035 m shift, the final xy error was **2.88–2.89 cm**, with a mean success
  margin of 1.1 cm.

So after release the apple travels **forward**, beyond the release point. The shift of −0.035 m
was chosen to compensate for that. If this holds on the wide distribution, the mechanism is
**refuted**: the apple lands about 2.9 cm from the centre, not 3.5 cm short. The candidate below
(release over the centre) would then land about 3.6 cm beyond it, with less margin than the
current collector. This prediction is recorded here before any number exists. The probe measures
it on the current distribution, with the look, on fresh seeds.

## 2. Seeds (fixed before running)

- **Probe seeds: 46800–46831** (32). They come from TASK-067's reserved range (46800–46899,
  inside the owner-approved 46000–46999 block; R5's overlap check covers it).
  - They are disjoint from the spent 46000–46415, from cohorts C (45300–45339) and D
    (45000–45007, 45100–45107), and from the smoke range 46900–46999.
  - `check_seeds()` asserts this, and also checks every range in `first_policy.FORBIDDEN_RANGES`.
- **Reserved now for any future v2 calibration: 46832–46863.** The probe never touches them.
- **Plate-error directions:** one per seed, from `default_rng(6810)`. The same direction is used
  for both arms and both non-zero levels, so the conditions are paired.

## 3. Design

| | |
|---|---|
| arms | `current`: `apple_collector_policy`, `transfer_x_shift = −0.035` (as `apple-look-v1`). `candidate`: `transfer_x_shift = 0.0`. With it the palm targets `container_x − 0.015`, and the held apple, 1.5 cm ahead of the palm, is released over the plate centre. |
| conditions | plate error 0 (reference), 1.0 cm and 1.5 cm; the apple is exact. That makes 2 arms × 3 levels × 32 seeds = 192 attempts. |
| path | the runner's worker pool (8 workers, CPU): reset, then the look, then the scripted arm, up to 800 steps (`first_policy_runtime`) |
| recorded | from simulator truth at each attempt's end: the apple xy minus the **true** plate-centre xy, landing distance, success, grasp, transport, dropped, termination and steps |
| script | `scripts/probe_first_policy_release.py`; output `outputs/task067-release-probe/run-1/report.json` |

`transfer_x_shift = 0.0` is the only candidate. It is fixed here by geometry before any probe
number exists. The probe's outcomes choose nothing.

## 4. Pre-declared reading (`decide()`, first match)

- **Mechanism confirmed** requires both of the following:
  - under `current` at the reference, the median landing dx (apple − plate, x) lies in
    [−4.5, −2.5] cm;
  - under `current` at 1.0 cm, at least 80 % of the failures transported the apple and ended with
    it outside the 4 cm radius.
- **Rows:**

  | row | condition |
  |---|---|
  | **P-VOID** | any attempt raised |
  | **P-MECH-REFUTED** | the mechanism is not confirmed |
  | **P-CANDIDATE-FAIL** | the mechanism is confirmed, and the candidate is below 28/32 at 1.0 cm |
  | **P-SUPPORTED** | the mechanism is confirmed, and the candidate reaches ≥ 28/32 at 1.0 cm |

- **R8's stop conditions apply as written.** On P-MECH-REFUTED or P-CANDIDATE-FAIL the work stops
  and is reported back. There is no option (b) and no bar change. Only P-SUPPORTED leads to the v2
  amendment PR.
- **Always reported:** every cell's attempts, successes, grasps and transports, the landing dx and
  distance quantiles (q10/q50/q90), and the failure breakdown.

## 5. Results

*To be filled from `report.json` after the run; nothing below §4 is written before it.*
