# Apple→Plate first learned policy v2, Linux replication: TASK-071 on the RTX 5080 PC with CUDA training (TASK-072)

**Status: preregistration.** No gated seed has been simulated on Linux under this protocol. Only
the smoke runs and the author's render check of §10 have run, on the smoke seeds, and nothing in
them is read.

**Status lines.**
- **Learned Apple→Plate on the frozen benchmark is still 0 successes.** TASK-071 run-1 reached
  M1-PASS on the non-gating development cohort D2 (P-3 16/16, an existence result); that is not
  the frozen benchmark, cohort C is untouched, and it is not LeWM driving the robot.
- This protocol **replicates TASK-071** ([`apple_first_policy_v2.md`](apple_first_policy_v2.md),
  merged at `9e23ced`; results [`apple_first_policy_v2_results.md`](apple_first_policy_v2_results.md),
  #97) on the Linux PC, which the owner made the working platform on 2026-09-28. **Every
  deviation from TASK-071 is in §3.** Anything not listed there is TASK-071's, unchanged.
- The development cohort is D2 (52000–52015), as in TASK-071. It never gates. **Cohort C
  (45300–45339) is not opened**, and M2 is not part of this task; it needs a separate owner
  go-ahead.
- The claim this can make is about **reproducibility across platforms** of the development
  result (§1). It cannot make a claim about cohort C, the frozen benchmark, LeWM or the v1 task.

Manifest: `benchmarks/manifests/apple-first-policy-v2-linux.json`. Design:
`src/embodied_jepa/first_policy_v2_linux.py`, which imports TASK-071's `first_policy_v2`
unchanged and declares only the changes below. Runner: `scripts/run_first_policy_v2_linux.py`;
render check: `scripts/check_first_policy_v2_linux_render.py`. Both are TASK-071's files plus a
declared list of textual substitutions, which `tests/test_first_policy_v2_linux.py` lists and
pins. **No TASK-071 file is modified**, and every TASK-071 pin still holds (a test checks it).

---

## 0. The brief (TASK-072, via the orchestrator, 2026-09-28)

1. Replicate TASK-071 on the Linux PC with CUDA training and 16 CPU simulation workers.
2. The same seeds, arms, budget cap (36 000 s), corpus plan, readout, DAgger schedule and M1 gate
   on D2 (52000–52015).
3. Name every deviation from v2 with its reason: device, platform, the floor digest re-pinned on
   Linux, frame-hash expectations now platform-specific, and the worker count.
4. Set new pins from measurements on Linux; keep v2's pins unchanged.
5. Declare before running how a match or mismatch with run-1 is read. The brief's example:
   P-3 counted success ≥ 14/16 counts as "replicated".
6. Cohort C stays untouched; M2 is out of scope.
7. Owner note on Stage B (2026-09-28): the replication's main process runs CUDA in **strict**
   deterministic mode. If strict mode fails for the MLP policy, stop and report; never fall back
   to warn-only silently.

## 1. The question, and the claim it can make

**Question.** Does TASK-071's development result reproduce when the whole pipeline runs again on
the Linux PC: a fresh `apple-look-v2` corpus collected there (NVIDIA EGL rendering, x86-64
physics), readouts on DINOv2 features computed there, and the policies trained on CUDA in strict
deterministic mode instead of MPS, with every seed, arm, cap and gate unchanged?

**What "replicated" would say.** The M1-PASS existence result, and a high P-3 rate, do not depend
on the Mac, its renderer or MPS: the pipeline gives the same development-cohort outcome on the
project's new working platform. **What it would not say:** anything about cohort C or the frozen
benchmark (still 0), that a learned policy "works" (only M2 on cohort C can say that), that LeWM
or any world model drives the robot, or that the two runs are bit-identical (they cannot be; §3).

**What a mismatch would say.** Something in the pipeline is sensitive to the platform (renderer,
CPU numerics in the DINOv2 features or the readout, CUDA vs MPS training), or run-1's 16/16 was
favourable variation from one run. One replication run cannot tell these apart; the results
document says so and the owner decides.

## 2. What is carried from TASK-071 unchanged

Everything in TASK-071's protocol §2 (itself v1's design) and §4–§14, including: the task
`apple-to-plate-v2` and the counted success (at rest after the settle **and** the latched grasp
and place stages before it; owner rulings T71-R1/R2); the e9 expert and its budgets; the corpus
plan (200 roots, seeds 51000–51199, split 170 / 20 / 10 by `default_rng(7105)`, noise level
i mod 4, the corpus name `apple-look-v2` inside its manifest); perception, C0, DAgger and D2 seeds;
every RNG seed (7100–7106); the readout (TASK-063 probe on frozen DINOv2 tokens, fit on 426
post-look frames); the policy inputs, head, training schedule (30 000 updates, AdamW, cosine,
batch 256, selection every 1 000 on val) and seeds; three DAgger iterations of 128 resets for P,
C and R; the arms P-0…P-3, C-3, R-3, F, A4-look, D-oracle-perc, B-oracle, B-hold, B-random,
B-replay; C0's rule and caps with the TASK-070 plate ceiling; S0-P, S0-D1 and the A4-look
threshold; the rows of `decide_m1`, the harness condition and the abandonment clause; every guard;
the void rule; and the caps (36 000 s global; 3 600 s corpus; 1 800 s per training; 3 600 s per
rollout batch and for C0; 1 800 s perception collection; 300 s per attempt). Features, readouts and
rollout inference stay on the CPU (6 torch threads in the main process, 1 per worker).

**Seeds are reused on purpose.** A replication simulates exactly TASK-071's seeds. They were
spent by run-1 on the Mac; here they are simulated again on a different platform. D2 is therefore
not a fresh cohort for this protocol, which is fine for a replication and is why nothing here is
a claim beyond D2.

## 3. Every deviation from TASK-071

| # | item | TASK-071 | this protocol | reason |
|---|---|---|---|---|
| 1 | platform | Mac: macOS 26.5.1 arm64, Python 3.12.13, Apple GL rendering | the Linux PC: Ubuntu 24.04.5 x86-64 (Ryzen 7 9800X3D), Python 3.12.3, **NVIDIA EGL** rendering (`MUJOCO_GL=egl`, set by the runner); a new guard **G-platform** refuses anything else | brief 1: the owner's platform switch |
| 2 | training device | MPS (`first_policy.TRAIN_DEVICE`) | **CUDA** (RTX 5080, torch 2.14.0+cu130), through `embodied_jepa.devices.require("cuda", strict=True)` before any CUDA work: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms in **strict** mode, cuDNN deterministic without benchmarking, TF32 off. G-device is also checked at the end (the process must still be strict). A kernel without a deterministic variant raises, which makes the run V; there is no warn-only fallback | brief 1 and the owner's note (§0.7). The MLP policy trains strictly and bit-identically on CUDA, for one head and for F's ten heads (a CUDA test and the smokes, §10) |
| 3 | simulation workers | 8 | **16** (1 torch thread each) | brief 1: the PC has 16 hardware threads. Attempts are per-seed deterministic and independent of the worker that runs them; the render check (row 6) checks the one place where workers could matter |
| 4 | floor-encoder digest pin | `3d305f9c…c9db` (measured on the Mac) | **`546b9011…29d2`** (full value in the manifest), measured on the Linux PC at `4f85f24`, the same at 1, 6 and 16 torch threads | brief 3: the seed-0 random-init DINOv2 differs in low-order float bits on x86-64 (TASK-071 results §7). The pretrained digest `3a697b87…af27` is unchanged and still pinned |
| 5 | frame hashes | post-look frames rendered by Apple GL | frames rendered by NVIDIA EGL; about a third of channels differ from the Mac's by about 2/255 (the bring-up) | brief 3: no protocol pins a frame hash across platforms. G-frame compares every attempt's re-render with the frame its estimates came from **within this run**, as in TASK-071. So the corpus (and its manifest sha256), the features, the readouts and every trained policy are necessarily different from run-1's; only outcomes can be compared |
| 6 | render check | `check_first_policy_v2_render.py` at 8 workers on the Mac | `check_first_policy_v2_linux_render.py`: the same check, at **16** workers, on the Linux PC through the Linux runner's pool, with G-platform | rows 1 and 3 |
| 7 | pins | 30 pins (TASK-071's files) | TASK-071's 30 pins, **unchanged and still checked**, plus this protocol's files: `first_policy_v2_linux.py`, `devices.py`, the two scripts, this document and TASK-071's manifest | brief 4 |
| 8 | smoke runs | CPU training | the smokes train on CUDA in strict mode too (runner checks only; nothing read) | to exercise row 2's path before the gated run |
| 9 | report | TASK-071's `report.json` | the same, plus `platform`, `environment.accelerator`, `environment.determinism`, `determinism_at_end` and `replication` (the declared reading of §5) | rows 1, 2 and §5 |
| 10 | output locations | `outputs/task071-first-policy-v2/run-<k>`, `checkpoints/task071-…`, `data/apple-look-v2/run-<k>` | `outputs/task072-first-policy-v2-linux/run-<k>`, `checkpoints/task072-first-policy-v2-linux/run-<k>`, `data/apple-look-v2-linux/run-<k>`; the runner refuses to overwrite any of them | never overwrite run-1's evidence (copied to the PC) |

Disclosed, not a deviation: the corpus's own `manifest.json` is written by TASK-071's unchanged
corpus store and so records protocol `apple_first_policy_v2` / TASK-071; the run's `report.json`
records `apple_first_policy_v2_linux` / TASK-072 and the corpus path.

## 4. Stages and outcomes

As TASK-071 §5–§10: preflight (now with G-platform and strict CUDA) → the corpus → perception
frames → readouts → C0 → S0-P and the A4-look threshold → BC-0, S0-D1 and three DAgger iterations
→ M1 on D2, each M1 arm once per reset → `decide_m1`'s first matching row (V, INCONCLUSIVE,
CAL-ESCALATE, S0-APPLE-FAIL, S0-PLATE-FAIL, M1-PASS, M1-MOTOR-F-PARTIAL, M1-MOTOR-F-NONE,
M1-PERCEPTION). The abandonment clause is TASK-071's, and fires on the same rows; the results
document reports it if it does.

## 5. The declared reading against TASK-071 run-1 (fixed now)

**The comparison target** is run-1 as recorded in
`benchmarks/manifests/apple-first-policy-v2-results.json` (report sha256 `77aae207…d90b`,
revision `9e23ced`, the Mac, MPS, 8 workers): outcome M1-PASS, carried P-3; counted successes of
16 on D2: P-0 4, P-1 9, P-2 15, **P-3 16**, C-3 3, R-3 16, A4-look 16, D-oracle-perc 16,
B-oracle 16, B-replay 9, B-hold 0, B-random 0; corpus 129/200 counted successes. A test pins
these numbers to that manifest.

**The replication row** (`first_policy_v2_linux.read_replication`, first match; written into
`report.json` as `replication`):

| row | condition | reading |
|---|---|---|
| **V** | the run is void (a guard, a crash, a cap, S0-D1, or the harness on D2) | nothing is read; one from-scratch repeat (§6) |
| **REP-EARLY-STOP** | the run ends at CAL-ESCALATE, S0-APPLE-FAIL or S0-PLATE-FAIL | run-1 passed those stages: **not replicated**, before any policy was evaluated; the owner decides |
| **REPLICATED** | outcome M1-PASS **and** P-3 ≥ **14/16** counted successes | the development result replicates on the Linux PC |
| **REPLICATED-M1-ONLY** | M1-PASS, P-3 < 14/16 | the existence result replicates (some P-k ≥ 1/16); P-3's rate does not |
| **NOT-REPLICATED** | any other M1 row | the development result does not replicate on this platform |

**Why 14/16.** It is the brief's bar, and the protocol's own bar for a working scripted expert on
D2 (B-oracle ≥ 14/16). Run-1's P-3 had 16/16 (Wilson 95 % interval 0.81–1.00, whose lower end is
13/16), so ≥ 14 asks for a rate in that range with one attempt of slack beyond the interval's
edge. It is fixed now and not revisited after the run.

**Reported whatever the row, descriptively (one run each; no test):** every M1 arm's counted
successes against run-1's, as signed differences; P-3's per-reset success set against run-1's on
the same 16 seeds (how many agree); whether R-3 is within 2 of P-3 and whether C-3 is at least 8
below P-3 (run-1's two vision findings), as observed facts, not gates; the corpus's counted
successes against 129/200 by noise level; the C0 table, the bars, S0-P's errors, the DAgger
rollout counts and each training's selected update against run-1's.

## 6. Void rule

TASK-071's: a run that stops early other than at a declared early-stop row is **V**; exactly one
from-scratch repeat into `run-2` (output, checkpoints and a new corpus directory), with the same
seeds, caps and device; **a second V closes this replication as INCONCLUSIVE**. A strict-mode
failure (a CUDA kernel without a deterministic variant) is a V like any other crash, and it is
reported to the owner before a repeat is started; the repeat is never made in warn-only mode.
Any guard change needs an owner ruling.

## 7. Process

1. **This PR** holds the protocol, the manifest, the design module, the two derived scripts, their
   tests and the card, and records the smokes and the author's render check (§10). It merges on an
   independent reviewer's reported APPROVE (posted on the PR) and green CI; the orchestrator
   merges.
2. **The gated run** starts from a clean checkout of the merged revision on the Linux PC, only
   after a fresh pre-run reviewer's **reported** GO (including its own render check at 16 workers,
   verdict IDENTICAL), and only after the orchestrator has been told of the GO:

   ```sh
   uv run --no-sync python scripts/run_first_policy_v2_linux.py run \
       --output outputs/task072-first-policy-v2-linux/run-1 \
       --checkpoints checkpoints/task072-first-policy-v2-linux/run-1 \
       --corpus data/apple-look-v2-linux/run-1
   ```

   It runs under `nohup` (or `systemd-run --user`), so it survives an ssh disconnect. The GPU is
   shared with another service; the policy MLP needs well under 1 GB.
3. **The results PR** states the M1 row and the replication row plainly, with counts, and every
   restated number is checked against `report.json` by an independent reviewer.
4. **One task, one agent.** Nothing is re-thresholded, retrained or re-selected after numbers are
   seen.

## 8. Budget

TASK-071 run-1 took 6 215 s of its 36 000 s cap on the Mac with 8 workers and MPS training. The
cap is unchanged. On the Linux PC the simulation stages run on 16 workers; the corpus, C0 and
DAgger rollouts dominate the time.

## 9. Not done (declared)

- No cohort-C attempt and no M2; no test-split decode.
- No attempt to make Linux and Mac runs bit-identical; outcomes are compared, not bytes.
- No second seed or second run beyond the void rule's repeat; no world model in the loop.

## 10. Smoke runs and the author's render check (nothing in them is read)

*Filled in below after the runs, before review.*

## 11. Amendment log

*Empty.*

**Learned Apple→Plate on the frozen benchmark is still 0 successes.**
