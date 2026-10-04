# TASK-076 Stage 0: the stage code and the smokes (development record)

**What this is.** The record of TASK-076's Stage 0 (protocol
[apple_plate_twin_v2.md](apple_plate_twin_v2.md) §5 step 1 and §13.2 steps 1–2). It ran on smoke
seeds 56900–56999 only. Nothing in a smoke is read as a result; the smokes check mechanics and
settle the K-pred questions the protocol leaves to them (§3.2, §12). No cohort seed (56000–56331)
was simulated. No arm runs a world model. Learned Apple→Plate on the frozen v1 MVP benchmark is
still 0/150 per backend.

Every decision below was decided by Claude under owner delegation (2026-09-30); the rulings are
R8.15–R8.17 in [DECISIONS.md](../DECISIONS.md), decision 2026-10-02 (b).

## 1. Results in one paragraph

- **Contact.** No smoke attempt had apple–plate contact before s1 = 525. The earliest contact was
  at step 607, in the open phase. s1 stays at 525.
- **Cell A is removed.** Under H-final the plate moved a median of 0.060 cm after the last
  decision, against the 2 cm the rule needs. No declared remedy reached 2 cm (L = 1, then
  s1 = 535 … 605: medians 0.022–0.063 cm). No move was refused. By the protocol's removal rule,
  cell A is removed before the freeze, and K-pred's row is **PRED-INFEASIBLE** (escalate, no
  clause).
- **m2** (H-final, L = 2): median 0.0012 cm, maximum 0.0019 cm.
- **A full-look-ahead H-final(A) attempt** took 15.0 s in median and 16.4 s at most (25.0 s at
  most over the remedy settings). That is below 60 s, so the 300 s per-attempt cap is not
  reviewed.
- **The K0 evidence blocker is cleared.** During Stage 0 one file of TASK-072 run-1's evidence
  was unreadable on the archive disk, so the smokes ran P-3 on a stand-in estimate (§2). The
  coordinator has since restored the file bit-identically, and a real G-repro check now passes
  all eight of run-1's recorded readout facts (§2a).

## 2. Setup, provenance and the evidence problem

- **Worktree:** `/home/huhn/develop/emai/worktrees/task076-stage0`, made with
  `scripts/new_worktree.sh --run` on branch `feat/task076-stage0` from `origin/main` at `0b92aea`.
  `data/` and `checkpoints/` are symlinks to the main checkout; every output is under its own
  git-ignored `outputs/`.
- **The code under test:**
  - `14b23b0` for every smoke in `outputs/task076-smoke-2` except the scale probe;
  - `bf3902b` for the scale probe (`offline-scale-2`), after the fix in §3.4.
  - Every report records `tracked_tree_dirty: false`.
  - After the smokes, only these changed: the Stage-0 constants in `plate_twin_v2.py` (`CELL_A`'s
    status and `STAGE0_SMOKES`), `cohort_digest` (floats rounded to 10 decimals, as
    `plan_digest`, after macOS CI computed a different last ulp; the draws themselves are
    unchanged), the tests, the manifest and the documents. The stage functions are unchanged.
  - After the review of #134 (R8.18), the harness was replaced by verbatim copies of the pinned
    runners' functions (`75d79ae`), and the truth stand-in moved into the runner as
    `smoke_truth_estimates`. The stand-in path the smokes took produces the same estimates and
    frame checks as before: the reset truth, after the same render-majority rule.
- **The machine:** the Linux PC; MuJoCo 3.13.0 with EGL; 6 CPU workers with 1 torch thread each.
  - Only the two offline smokes used the GPU, each through
    `scripts/gpu_run.sh --wait --board --who oej:task076-O-smoke` (or `-scale`), with
    `gpu_lock_held: true` in its report.
  - Each held the lock for under a minute. No BOARD request was posted.
  - The 1-minute load at the starts was 0.1–5.1, from the smokes themselves. G-quiet does not
    apply to smokes.
- **The evidence root (`--evidence`).** TASK-072 run-1's evidence (the `task072-run` worktree)
  was removed from the SSD on 2026-10-04 after archiving (docs/STORAGE.md). Its three parts were
  restored from the archive disk into a new directory,
  `/home/huhn/develop/emai/worktrees/task076-evidence/` (not a git worktree):
  - `outputs/task072-first-policy-v2-linux/run-1`;
  - `checkpoints/task072-first-policy-v2-linux/run-1`;
  - `data/apple-look-v2-linux/run-1`.

  Every restored file was checked against `checksums/task072-run.sha256`. 413 of 414 match:
  run-1's report, the P-3, C-3 and R-3 checkpoints (P-3 sha256 `7988162d…60be8`) and the corpus
  manifest all pass `check_evidence`.
- **One file is damaged.** `data/apple-look-v2-linux/run-1/episodes/look2-51171.npz`, a train
  root, cannot be read:
  - two 512-byte sectors (file bytes 785 408–786 431, inside `frames.npy`) return I/O errors on
    every retry;
  - no other copy was found on the SSD or on the archive disk;
  - no repair was attempted.

  G-repro reads every train root's post-look frame through `CorpusReader`, which checks each
  file's sha256 first, so G-repro could not pass during the smokes.
  - **Cleared after the smokes.** The coordinator restored the file bit-identically (sha256
    `e13c97cb…c296`; 414 of 414 run-1 files now verify). The procedure is in
    `/home/huhn/develop/emai/worktrees/task076-evidence/RESTORE-51171.md`. That includes K0 and every closed-loop
  stage.
- **The smoke stand-in.** Every closed-loop smoke ran with `--truth-estimates`: P-3's post-look
  estimates are the reset truth, not its refitted readout. The runner accepts this flag in smokes
  only, and each report says so (`estimates_source`). G-repro was not run, so no TASK-072 seed
  was rendered.
  - This does not change the K-pred answers. The plate's motion after 485 is set by the place
    primitive's palm path, which follows the aim, not P-3's estimate.
  - It may change P-3's grasp rate, which the smokes do not read.
- **τ_re** was a placeholder of 1.0 cm (TASK-075's τ) in every smoke. It sets H-final(A)'s
  look-ahead tolerance of 0.25 cm.

## 2a. The real G-repro path, checked (after the restore)

`scripts/run_plate_twin_v2.py preflight` ran at `75d79ae`, on a clean tree and on the CPU, with
`--evidence /home/huhn/develop/emai/worktrees/task076-evidence`. It ran the preflight guards,
G-evidence and G-repro. G-repro refits P-3's post-look readout exactly as TASK-072 run-1 did.
- It re-renders only run-1's own post-look frames (the perception and D seeds of TASK-072's
  block). No new seed was used and no attempt was run.
- Outcome: **PREFLIGHT-READY**, in 46 s, with a peak of 8.0 GiB PSS.
- All eight reproduction checks pass:
  - the P and R readout selections;
  - the fit rows;
  - S0-P's apple and plate errors;
  - C's mean estimates;
  - the B-replay library;
  - D2's nearest roots.
- 170 train roots were decoded, and no test root.
- One re-render disagreed and was settled by the render-majority rule: seed 51235, one level in
  4 pixels, with equal states.
- Report: `outputs/task076-grepro-1/report.json`, sha256
  `8e52397a563fdd0ee25ac8cf3e46ab69f5a7d4e975d7f7f3e35b9d5300ecceb0`.

## 3. The smokes (outputs/task076-smoke-2, revision 14b23b0)

Seeds per stage:

| part | seeds |
|---|---|
| K0 | 56900–56903 |
| M-a | 56910–56917 |
| M-b | 56920–56927 |
| A | 56930–56945 |
| D | 56946–56949 |
| S | 56950–56955 |
| U | 56956–56959 |
| source corpus | 56960–56983 (16 train, 6 val, 2 test) |

Every remedy run reused cell A's 16 seeds, so the settings are compared on the same resets.

### 3.1 Reports

| part | outcome | seconds | peak PSS (GiB) | report sha256 |
|---|---|---|---|---|
| source | SOURCE-SEALED | 30.8 | 7.5 | `7b79f2e73690bff222c392977c85b712e0858bcd8495fd823ba3368f7a019a77` |
| render | VIEWS-SEALED | 25.3 | 7.9 | `8b17bd1ec6203a903ade64fc7a40dc2dad8f450dc368f4babe15af40b83bbba2` |
| offline | OFFLINE-SMOKE | 9.5 | 1.8 | `6b30297f1f9b64fb8a8c77597ecb05581d54a4a6d8101f6ee939eeda99258f9f` |
| kpred-A | KPRED-SMOKE | 112.3 | 9.1 | `cdbc046a73960a2bc504ac01d08f77e00790087e15a53546316b445698b287b2` |
| kpred-M-a | KPRED-SMOKE | 44.7 | 8.3 | `e13c89fcd89be5f734e626294d9e26bfbe6d92f9fadfd226503f64f563e36390` |
| kpred-M-b | KPRED-SMOKE | 44.6 | 8.3 | `a522d53d196a864bcf2ccc83410e59bd09e26be438ed249a1950f05ecb9d8ecd` |
| k0 | K0-SMOKE | 23.6 | 7.0 | `25a5ee609c9c5d533575e2af546d4e3eb40e161ca6e694a659df622f2a25b109` |
| dev | DEV-SMOKE | 17.5 | 7.9 | `295c08a25aff11c7e23f82dbd036413279db50542946c884663776c8eb93390c` |
| gated | GATED-SMOKE | 53.3 | 8.8 | `e23a7c4ab07432a1ed5bc64f2d21a0241df8d36d75b58df813fd6814bbbccc2c` |
| offline-scale | **V** (§3.4) | 9.6 | 1.8 | `65e759b9434a2923b7eec778cb51e1545d8ed4fdcee78847f7f63067205afc5d` |
| offline-scale-2 (`bf3902b`) | OFFLINE-SMOKE | 49.0 | 4.1 | `0f1a61f5906d7a13e300c5185ffdd48cbef194f2a7395b4d7b95dbe968da535b` |

The remedy reports are listed in §3.3.

### 3.2 K-pred at the base setting (κ = −0.5, L = 2, s1 = 525)

Counts and grasp rates are mechanics only and are not read: they ran with the stand-in estimates
and a smoke R-plate fitted on 20 smoke roots.

| cell | arm | count | refused | contact before s1 | earliest contact | median remaining (cm) | median attempt time (s) |
|---|---|---|---|---|---|---|---|
| A | H-final | 16/16 | 0 | 0 | 607 | 0.060 (max 0.076) | 15.0 (max 16.4) |
| A | H-now | 14/16 | 0 | 0 | 607 | 0.148 | 4.2 |
| A | H-twin | 2/16 | 0 | 0 | 609 | 0.283 | 4.8 |
| A | H-cv | 6/16 | 0 | 0 | 608 | 0.387 | 4.8 |
| A | H-rule | 0/16 | 0 | 0 | 609 | 0.115 | 5.5 |
| M-a | H-final / H-now / H-twin / H-cv | 8, 1, 3, 2 of 8 | 0 | 0 | 610 | 4.00 (as declared) | 4.2–6.3 |
| M-b | H-final / H-now / H-twin / H-cv | 8, 0, 0, 1 of 8 | 0 | 0 | 610 | 3.00 (as declared) | 4.2–6.3 |

- **The contact check holds.** None of the 144 base-setting attempts had apple–plate contact
  before s1, and none was refused. The earliest contact over all of them was step 607. That is
  the contact limit for the remedies.
- **What the contact check covers.** The hook checks contact from step 405 (s0) onward, once per
  control step, at that step's observation. So "no contact before s1" means no contact at any
  observed step from 405 to 524. Physics substeps between two observations are not checked.
  Every plate move is separately refused by `plate_shift.move_plate` if the moved plate touches
  anything but the table.
- **Why cell A's plate barely moves after 485.** Under H-final, the palm's xy moved a median of
  0.12 cm (maximum 0.15 cm) between 483 and 523. e9's transfer brings the palm to the aim well
  before the last decision, and the lower is vertical. Before 485, though, the rule moved the
  plate a lot: over 405–525 it moved 4.9–7.4 cm in −x and 2.1–4.1 cm in −y, towards the robot.
  That is outside the corpus's −y move arc, which is the coverage caveat of §3.2.
- **H-final(A)'s look-ahead.** All 96 decisions converged to 0.25 cm, in at most 7 iterations
  (median 4). From a first gap of 7–10 cm the gap about halves per iteration, as §3.2 estimated.
  At most 2 480 branch steps were simulated per attempt.
- **m2** (H-final): median 0.0012 cm, maximum 0.0019 cm.
- **H-rule's mechanics.** The fixed-point iteration also contracts by about half per iteration
  and converged at every decision. Its 0/16 comes from the smoke R-plate's readings, which were
  2–6 cm off on this cell: that readout was fitted on 20 static-plate smoke roots. Nothing is read
  from it.

### 3.3 Cell A's remedies, in the declared order

The rule (§3.2): first L = 1, then s1 later in steps of 10 up to the contact limit.
- The contact limit is the earliest contact of the base smokes, step 607.
- s1 stays strictly before it, so the candidates are 535–605.

Each setting ran H-final and H-now on cell A's 16 seeds.

| setting | H-final median remaining (cm) | max (cm) | refused | H-final count | report sha256 |
|---|---|---|---|---|---|
| L = 1, s1 = 525 | 0.0626 | 0.0744 | 0 | 16/16 | `f71251de3d908bbd0d51902d5f72f049493d2647a06483a8599406261628ddbe` |
| L = 1, s1 = 535 | 0.0491 | 0.0565 | 0 | 14/16 | `306c4092989a5d8af59f576d7767fc7f68fb5375150743e13ca7bbfe0202b0c2` |
| L = 1, s1 = 545 | 0.0315 | 0.0388 | 0 | 14/16 | `1f11f3bc342f83d9ce8c9ddcabc3b14a4a760e68e1b368c86124683231873029` |
| L = 1, s1 = 555 | 0.0252 | 0.0350 | 0 | 15/16 | `08b7a5e838fb307adc6b80289f8dcbcbbeda7a216388b0a5a3372967d67faf96` |
| L = 1, s1 = 565 | 0.0230 | 0.0340 | 0 | 14/16 | `7ffcef4fed686ff3b8c95059525b6944f4b6fcc1690868e5d50896fc48c7150f` |
| L = 1, s1 = 575 | 0.0222 | 0.0336 | 0 | 15/16 | `b6fc93d150f54346322fd9aa27ae27ad1fe32aaf5640cfeb515f0712aba11ba6` |
| L = 1, s1 = 585 | 0.0235 | 0.0335 | 0 | 15/16 | `a35c3aa18b72bf8f40fcfc180d1dce64db6a0b8e97af7ebce32a745b4871303f` |
| L = 1, s1 = 595 | 0.0235 | 0.0335 | 0 | 15/16 | `8f0cc2b3ed4fb811e7d42626f050d42f33363584eba813f8c336681df02f7893` |
| L = 1, s1 = 605 | 0.0233 | 0.0334 | 0 | 15/16 | `475e2b4c4b923259049861a3331f5103741ef935fa2a1d223d19b537dbf1ae1d` |

- No setting reaches a median of 2 cm; the best is 0.063 cm, about 30 times short.
- **Cell A is removed before the freeze**, by the protocol's rule: "If no allowed setting gives a
  median of at least 2 cm … cell A is declared infeasible and removed before the freeze". κ was
  not touched, and L was never raised.
- No remedy attempt had contact before its own s1, and none was refused.
- H-final's attempt time rose with s1, to 25.0 s at most at s1 = 605. H-now's remaining motion
  (0.15–0.19 cm in median) is reported for scale.
- The remaining motion shrinks as s1 moves later. The look-ahead then aims at a plate that has
  already settled, and the palm's xy hardly moves during the lower and the steady phase.

### 3.4 The other stages' mechanics (nothing read)

- **K0** (levels 0 and 3 cm on 4 seeds):
  - the planted-error H-handover arm ran at both levels and the counts were assembled;
  - H-clock's targets were fitted by the per-step median rule on the smoke seeds' schedules (a
    stand-in for K's);
  - H-stale and H-clock ran;
  - the G2 feasibility calculation ran (b = 1, c = 0 on 4 pairs; mechanics only).
- **Stage O** on the smoke corpus (20 read roots, 120 decision rows, 120 plate-hidden rows):
  - featurisation ran under the GPU lock, with a CPU anchor difference of 2.8e-5 (bound 1e-3) and
    a GPU peak of 0.57 GiB reserved;
  - the 5-fold cross-fit of R-plate, R-plate-floor, R-plate-pool and the clock prior ran;
  - the closed-loop R-plate and R-plate-floor were saved and sha256-checked by the workers.
- **The scale probe.**
  - The first attempt (`offline-scale`) was **V**. `run_tools.CycledReader` by default pools
    every split, so a slot was served by a smoke test root. The reader refused that root before
    opening it (Q-split), so no test root was decoded.
  - Fixed in `bf3902b`: the probe cycles train and val roots only, and a test checks this.
  - The repeat (`offline-scale-2`) ran the Stage O core itself (`run_tools.scale_probe` on
    `offline_core`) at 240 train + 30 val slots: 246 slots read, 1 476 decision rows, 2 952
    frames.
  - Its peak was 4.13 GiB process-tree PSS, against the 12 GiB ceiling less the 2 GiB margin.
    Featurisation took 6.3 s on the GPU, and the probe 39 s in all.
- **Stage D and the gated stage** ran every arm (H-twin, H-handover, H-clock, H-stale, H-floor,
  P-stale on S; H-twin, P-stale, H-handover on U). H-twin's determinism re-run matched on all 4
  seeds.

### 3.5 Earlier runs (superseded, kept)

These runs gave the same K-pred numbers as smoke-2:
- `outputs/task076-debug-1`: one cell A run on seeds 56990–56991, on uncommitted code before
  `1deb139` (report `bf5a7202…a035`).
- `outputs/task076-smoke-1`, at `1deb139`:
  - source `3fb946b2…42bd`;
  - render `dcf8a5af…50d2`;
  - offline `77db907d…408c`;
  - kpred-A `c440afe5…939b`;
  - kpred-M-a `f9415682…0669`;
  - kpred-M-b `08d36bbf…4c7541`.

  Smoke-2 is the record, because its revision has the absolute readout paths.

## 4. Decisions made in Stage 0 (R8.15–R8.17)

- **Cell A's discretisation.** For s0 < t ≤ s1, plate(t) = plate(s0) + κ·(palm(max(t − L, s0)) −
  palm(s0)). Palm velocity at or before s0 counts as zero, and the palm is the right palm's xy
  from `PalmFK` on each step's observed joint state. So plate(s1) − plate(485) =
  κ·(palm(s1 − L) − palm(485 − L)), as §3.2 states.
- **The plate hooks** move the plate at every step's observation, before the image, through
  `plate_shift.move_plate`, which refuses a contact. In M cells the plate moves 0.1 cm (M-a) or
  0.075 cm (M-b) per step.
- **H-final(A)'s look-ahead.**
  - It clones the state with `sim_selector.Brancher`, plus the plate's `body_pos` and the hook's
    history.
  - It aims at g_{k+1} once |g_{k+1} − g_k| is within the tolerance.
  - A branch whose plate move is refused ends there, and its iterate is the plate where it
    stopped. This never happened in the smokes.
- **H-rule** uses palm(max(d − L, s0)) from the controller's own executed states, and the
  kinematic stand-in's palm at s1 − L under each candidate aim.
- **H-cv** fits a least-squares line per axis through H-twin's readings, and aims at the
  reading at the first decision.
- **The contact limit** for the remedies is the earliest apple–plate contact step over every
  base-setting smoke attempt (all cells and arms), with s1 strictly before it. The remedies'
  later s1 applies to cell A only; M-a and M-b keep s1 = 525.
- **The smoke stand-ins** were the truth estimates (G-repro unavailable) and τ_re = 1.0 cm.
- **The harness pieces of the pinned runner chain** (G-evidence, G-repro, the render majority,
  report I/O) are copied into `plate_twin_v2_harness.py`, because a new runner may load no
  other script (R8.18, after the review of #134).
  - **Verbatim.** Each copied definition has exactly its source's text, docstrings included.
    `test_the_harness_ports_are_verbatim` compares them with `ast.get_source_segment`; nothing
    is stripped.
  - **Adapted, the only deviations:**
    1. The module aliases the copied text uses (`R65`, `M2R`, `LIN`) are rebound to the
       harness's own copies instead of loading the scripts.
    2. `LIN.load_wide_reset()` returns `wm_critic_v2.wide_reset_values`, the NumPy copy of
       `scripts/evaluate_apple.wide_reset`, instead of loading that script. A test checks that
       the two are equal on every seed 51000–52199, which is where G-repro's re-rendered seeds
       lie. These are pure draws, and nothing is simulated.
  - **TASK-076's own code, not ports:** `utc`, the bounded `Pool` (the TASK-074/075 pattern on
    TASK-076's worker), `seal_far_corpus` and `strip`.
  - **The truth stand-in** is not in the harness. It is the runner's `smoke_truth_estimates`,
    and both `estimates_for` and `run()` refuse it unless `--smoke` is set, behind the parser's
    own check (tested).
  - The guards come from `run_tools`.
- **The protocol text items of the #133 approval** (R8.17):
  - the header's bar sources;
  - R8.12's two effects pulling in opposite directions;
  - the guard's power (about 42 % at 4/32, about 87 % at 2/32);
  - §10's 60 s cap-review rule added to §13.2 step 2.

## 5. What blocks K0

1. ~~**The G-repro evidence.**~~ **Cleared.** `look2-51171.npz` is restored with its recorded
   sha256 (`e13c97cb…c296`). See `RESTORE-51171.md` in the evidence root for the procedure. The
   real G-repro path passes (§2a).
2. **A reviewer's reported GO for K0**, as before. K0 runs with
   `--evidence /home/huhn/develop/emai/worktrees/task076-evidence`.
3. Not a blocker: cell A's removal makes K-pred's row PRED-INFEASIBLE in advance. Whether the
   K-pred M cells (reported only) still run is unchanged by this: §5 step 7 runs them after
   O-PASS.
