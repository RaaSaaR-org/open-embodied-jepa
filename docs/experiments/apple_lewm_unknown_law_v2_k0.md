# TASK-082 K0 under U-sat: CAL-ESCALATE (record)

**Status: K0 record; the row is CAL-ESCALATE (escalate, no clause); nothing is frozen.** Protocol:
[apple_lewm_unknown_law_v2.md](apple_lewm_unknown_law_v2.md) §7.2 (STATUS DRAFT; Stage 0 record
[apple_lewm_unknown_law_v2_stage0.md](apple_lewm_unknown_law_v2_stage0.md)). Rulings R20.23–R20.25
in [DECISIONS.md](../DECISIONS.md), decision 2026-10-08 (n), each **decided by Claude under owner
delegation (2026-09-30)**. K0 is a calibration of privileged and hand-written arms only: no world
model, no learned controller and no LeWM arm ran, and nothing here is a learned or LeWM result. The
canonical status sentence (R7) is unchanged.

## 1. The run

| item | value |
|---|---|
| GO | independent pre-launch review, "K0 GO at c0b2df6dedfa61cb091a021382e6f1e3aea3f03a" ([#170 comment](https://github.com/RaaSaaR-org/open-embodied-jepa/pull/170#issuecomment-6063308314)) |
| revision | `c0b2df6dedfa61cb091a021382e6f1e3aea3f03a` at start and end, tracked tree clean; frozen-block candidate sha256 `bb29b38b…39b9` (DRAFT, no pin) |
| worktree | `/home/huhn/develop/emai/worktrees/task082-k0` (`scripts/new_worktree.sh … --run --from c0b2df6`, its own `.venv`) |
| command | `MUJOCO_GL=egl MKL_DYNAMIC=FALSE OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=16 uv run --no-sync python scripts/run_lewm_ul_v2.py k0 --output outputs/task082-k0-1 --evidence /home/huhn/develop/emai/worktrees/task076-evidence --r-plate-fits /home/huhn/develop/emai/worktrees/task077-stageo/outputs/task077-readouts-1/fits --log outputs/task082-k0-1.log` |
| cohort | K, 64 resets 72400–72463, under U-sat (A 8 cm, D 12 cm, Θ 0.25 rad, L 2, s0 405, s1 525), ρ\* = 4 cm; planted direction salt 8408 |
| device | CPU only, 6 workers, no GPU lock (the protocol runs K0 on the CPU); Linux x86_64, MuJoCo 3.13.0, EGL |
| guards | in-run G-tests at the revision (2 213 passed, 37 skipped); G-quiet (load 0.10 / 0.63 at start, no wait); G-memory: 25.6 GiB available at start, peak process-tree PSS **9.67 GiB** against the 12 GiB ceiling; G-hash on TASK-076's evidence and TASK-077's R-plate (`08bde901…9eb9`); G-law on every record; 0 refusals, 0 fallbacks, no non-finite fields |
| time | 16:41:41–16:54:58 UTC, **797 s** against the 7 200 s cap |
| report | `outputs/task082-k0-1/report.json`, sha256 `ceeece61b60d22f9662549b657aa7f76bec57036588164dbc5c25cdd8f27956c` |
| evidence copy | `~/develop/emai/evidence/task082-k0/` (report, log, launcher stdout, README, `SHA256SUMS`) |

The run started only after another session's Isaac Sim development jobs had released their memory
(about 10 GiB available while they ran; the launch waited until at least 18 GiB was available and
the load was low, as the reviewer recommended). K0 ran **once**; it is not repeated (R20.1, §7.2).

## 2. The numbers (one run, 64 resets, simulation only)

**The τ curve** (H-final(commit)'s converged aim plus a planted error, counted successes of 64; bar
56/64):

| planted error | 0 | 0.5 cm | 1.0 cm | 1.5 cm | 2.0 cm | 3.0 cm |
|---|---:|---:|---:|---:|---:|---:|
| counted successes | **64** | **55** | 55 | 34 | 22 | 21 |
| landing miss, median (max), cm | 0.05 (0.12) | 0.68 (0.84) | 1.35 (1.61) | 2.02 (2.37) | 2.69 (3.14) | 4.05 (4.67) |

- **τ_commit = 0** (0.5 cm reaches 55/64, **one success below** the 56/64 bar). Level 0 is 64/64.
- The 9 failures at 0.5 cm and the 9 at 1.0 cm are **disjoint resets** (18 distinct), so the
  failures at these two levels do not grow with the planted error; the count at 0.5 cm reads as a
  rate near the bar on 64 resets (55/64 = 0.859), not a sharp tolerance edge. Under C1-M TASK-077's
  32-reset K0 read 29/32 (0.906) at both levels (R17.25). The landing miss at 0.5 cm planted is 0.68
  cm in median, larger than the planted error: under U-sat the plate's response to the hand moves
  the landing further than the aim error alone.
- **The ceiling N_K(0) = 64/64** (bar 60/64).
- **r_K = 466** against the frozen read step r = 465, **one step late**: 466 is the earliest step at
  which ≥ 56/64 level-0 attempts have the plate within 0.1 cm of plate(525). The recipe rolls out
  exactly 60 steps from 405, so r = 465 cannot be kept.
- **The history check:** median palm speed at 405 **0.0039 cm per step** (max 0.025; bar 0.5). The
  plate moves 6.62 cm in median after 405 (max 7.41).
- **The aim-box clip-binding check:** **0 of 64** ceiling aims outside the box built from p̂ and h
  (64 level-0 aims had p̂ logged; limit 4). |b| 1.17 cm in median, 2.29 cm at most (bound 3 cm);
  |a| 0.336 in median, 0.394 at most (bound 0.5).
- **Reported only** (gate nothing): H-rule (C1-M's written law κ = −0.5, on p̂; not learned)
  **54/64**, ceiling − H-rule +10/64 (95 % interval +5 to +16; 10 discordant resets, all in the
  ceiling's favour), its own clip-binding fraction 0.28; H-now (the plate's position at 405; not
  learned) **1/64**, ceiling − H-now +63/64 (95 % interval +61 to +64).

## 3. The row

`decide_k0` returned **CAL-ESCALATE** with two stops (§7.2), each by one unit:

| stop | fired | value |
|---|---|---|
| level 0 < 56/64 | no | 64/64 |
| N_K(0) < 60/64 | no | 64/64 |
| **τ_commit = 0** | **yes** | 0.5 cm: 55/64 |
| **r_K later than 465** | **yes** | 466 |
| median palm speed at 405 > 0.5 cm/step | no | 0.0039 |
| more than 4/64 ceiling aims outside the box | no | 0/64 |

The clause does not fire (it fires only on L-NO-GAIN or L-INFERIOR, §12). The τ < 1.0 cm ruling
point for Stage T's GO (§7.2) is **not reached**: it applies to a K0-PASS with τ_commit = 0.5 cm,
whereas τ_commit = 0 is itself CAL-ESCALATE (O1, R0 and R1 would be unattainable).

## 4. What follows (R20.25)

- **Nothing is frozen**, and no later stage of this protocol (C, O, T, G, D, S) is launched. K's
  seeds 72400–72463 are spent; K0 is not repeated, and neither stop is relaxed after the fact: the
  56/64 rule and r ≤ 465 were declared before K0, and both misses are measured facts on fresh resets.
- **TASK-082 closes at K0 as CAL-ESCALATE, without the clause.** Nothing in its scope is closed or
  refuted: no LeWM model was trained or run under U-sat. The DRAFT protocol, its Stage 0 code and
  δ = 16/128 stay as recorded.
- **What a successor would need** (recommendation only; any of it is a new task with its own design
  note, preregistration, review and fresh seeds): a horizon that covers U-sat's settling (r ≥ 466,
  so h ≥ 61, a change to TASK-077's recipe and a retrain on that horizon), and a commit tolerance
  that holds at U-sat's measured τ curve, either by a law whose landing amplification is smaller or
  by bars declared for τ_commit = 0.5 cm from the start with the risk to R1 stated. The design
  note's development check already measured U-sat's ceiling at 32/32 and H-rule at 23/32 on the
  true plate; K0 measured H-rule at 54/64 on p̂, so the headroom over H-rule under this law on these
  resets is +10/64.

## 5. Caveats

One run, 64 resets, one condition, simulation only; τ_commit and r_K both miss by one unit, so a
different 64-reset draw could have passed either check, and the row is the declared reading of this
draw, not a tolerance measured to within one reset. H-final(commit) is privileged; H-rule and H-now
are hand-written and not learned. Nothing here is a learned, LeWM or world-model result.
