# Storage: SSD and the archive disk

The Linux PC's internal SSD (`/`, 442 GB) is shared and nearly full; keep at least 10 GB free on it
(see [RESOURCES.md](RESOURCES.md)). Large files that do not need to live on the SSD go to an external
USB disk.

## The archive disk

- Intenso USB HDD, 466 GB, label `INTENSO`, FAT32 (vfat), device `/dev/sda1`.
- Mounted by udisks at `/media/huhn/INTENSO` (`udisksctl mount -b /dev/sda1` if it is not mounted).
- Slow (USB HDD, mounted with `flush`): copy in the background with `nice` and `ionice -c3`.

```
/media/huhn/INTENSO/
  README.md                         what the disk is, layout, FAT32 caveats, restore steps
  emai/
    open-embodied-jepa/
      README.md                     project-level index
      archive/
        worktrees/<worktree-name>/  verbatim copy of a removed worktree (minus symlinks and the .git file)
        MANIFEST.md                 one row per archived item: path, branch, HEAD, PR, bytes, files,
                                    split files, skipped symlinks, checksum file, time, removed yes/no
        checksums/<name>.sha256     sha256 of every archived file, paths relative to the worktree root
        run-logs/                   loose run-launcher *.log/*.txt from ~/develop/emai/worktrees/
                                    (checksums/run-logs.sha256)
        patches/<branch>/           git format-patch of local-only commits that could not be pushed
        deleted-remote-branches-<date>.txt   origin branches deleted after their PRs merged
      datasets/                     large corpora (empty for now)
      checkpoints/                  model checkpoints (empty for now)
      runs/                         generated run outputs (empty for now)
```

Nothing on the disk is the only copy of source: every archived worktree's HEAD is reachable on
GitHub, but since the merged branches were deleted from origin (2026-10-02) mostly not as a branch.
Each HEAD is on `main`, on an `archive/*` branch (`task072-a`'s detached WIP commit was pushed as
`archive/task072-a-wip` for this), or only as `refs/pull/<N>/head` of the PR in the tables below.
The disk holds the ignored artefacts git does not track (`data/`, `outputs/`, `checkpoints/`,
`.venv/`, logs). As with `data/`, `checkpoints/` and `outputs/`, never overwrite evidence on it:
add, do not replace.

### The disk is failing; the evidence is back on the SSD (2026-10-04)

The archive disk is degrading: a read-only audit on 2026-10-04 found four unreadable files
(SCSI medium errors) and a growing number of bad sectors spread over the surface, beyond the
clusters first seen on 2026-10-02. Its record is `archive/INTEGRITY-2026-10-04.md` on the disk
(`/media/huhn/INTENSO/emai/open-embodied-jepa/archive/INTEGRITY-2026-10-04.md`). Only one of the
four is cited evidence: `data/apple-look-v2-linux/run-1/episodes/look2-51171.npz` of TASK-072
run-1 (the train root of seed 51171). It was restored bit-identically (sha256 `e13c97cb…c296`,
the archived value) at
`/home/huhn/develop/emai/worktrees/task076-evidence/data/apple-look-v2-linux/run-1/episodes/look2-51171.npz`,
from the disk's readable bytes plus the two unreadable 512-byte sectors refilled from a
deterministic re-collection of that seed; the method and checks are in
`/home/huhn/develop/emai/worktrees/task076-evidence/RESTORE-51171.md`. The other three are Warp
and Omniverse caches, not evidence.

Because the disk had become the only copy, all archived evidence was copied back to the SSD the
same day, at **`/home/huhn/develop/emai/evidence/`**: 42 items, 20 387 files, 10 049 048 059
bytes (about 10.05 GB, 9.36 GiB), counted from the tree without `_checksums/` and its
`README.md`.

- One directory per archived worktree, **40 in all**, named as in the table below, plus
  `_archive-extras/` (`run-logs/` and `patches/fix-ci-integration-workflow/`).
- The 14 worktrees with run evidence (`task072-run`, `task072-m2-run`, `task072-m2`, `task072-a`,
  `task073-go`, `task073-go-2`, `task073-fix`, `task074-run`, `task075-run`, `arena-e9`,
  `arena-spike`, `arena-gr00t`, `white-plate-dev`, `isaac-usd-threads`) are copied whole, minus
  `.venv/`, `home_cache/` and tool caches. The other 26 hold only `data/`, `outputs/`,
  `checkpoints/`, `benchmarks/` and `docs/experiments/`.
- TASK-072 run-1 (`task072-run/data/apple-look-v2-linux/run-1/`, 401 files: 200 episode `.npz`,
  200 `.json` and `manifest.json`) is included. It was added after the first pass, which had
  skipped it because of the copy in `task076-evidence`. Its 401 files verify against the archive
  checksums (the damaged `look2-51171.npz` is the restored one), so `evidence/task072-run/`
  holds 1 038 files.
- **Verified:** every file checked with `sha256sum -c` against the disk's
  `archive/checksums/<name>.sha256`, filtered to the copied scope; every count matched, with no
  failure. The filtered lists are in `_checksums/` (42 files): `<name>.sha256` for each
  worktree, plus `run-logs.sha256` and `patches.sha256`.
- `README.md` there records the method and the per-worktree table.

Re-verify a worktree, and the two extras, with:

```sh
E=/home/huhn/develop/emai/evidence
cd "$E/<name>" && sha256sum -c --quiet ../_checksums/<name>.sha256
cd "$E/_archive-extras/run-logs" && sha256sum -c --quiet ../../_checksums/run-logs.sha256
cd "$E/_archive-extras/patches/fix-ci-integration-workflow" \
  && sha256sum -c --quiet ../../../_checksums/patches.sha256
```

It is a plain directory tree, not a git worktree: each worktree's code is at its HEAD in the table
below. **Read evidence from there, not from the disk.** The path mapping is uniform. For an old
SSD path in a results document, replace `/home/huhn/develop/emai/worktrees/<name>/` (or
`/home/huhn/develop/emai/<name>/` for the three `oej-isaac-*` rows) with
`/home/huhn/develop/emai/evidence/<name>/`. That includes TASK-072 run-1, at
`evidence/task072-run/data/apple-look-v2-linux/run-1/`. The archive path in the table is the
second copy, and a loose run log is at `evidence/_archive-extras/run-logs/<file>`. Never
overwrite anything under `evidence/`; add, do not replace. `.venv/` and caches were not copied,
so the archive is still the only copy of those (none is cited evidence). Use the disk only to
read, and copy anything new to the SSD as well.

**A second SSD copy of TASK-072 run-1: `/home/huhn/develop/emai/worktrees/task076-evidence/`.**
It is a plain directory (not a git worktree, and not in `git worktree list`) holding TASK-072
run-1's `data/`, `checkpoints/` and `outputs/` and `RESTORE-51171.md`. TASK-076's runner reads it
through `--evidence` and pins its hashes (G-evidence and G-repro: report, P-3, R-3, C-3 and the
corpus manifest; `plate_twin_v2.EVIDENCE`). K0 ran on it. **Do not clean it up, move or edit it
while TASK-076 is open**; `scripts/remove_worktree.sh` does not apply to it.

**TASK-076's run worktree: `/home/huhn/develop/emai/worktrees/task076-run`** (at `702a7d9`). Its
`outputs/` holds the post-freeze reports of Stages O, D, K-pred (M-a, M-b) and S/U and the
closed-loop readouts (`task076-O-1/r_plate*.npz`), which the workers loaded by absolute path; the
results document cites their sha256 ([apple_plate_twin_v2_results.md](experiments/apple_plate_twin_v2_results.md) §1).
K0's report is in the main checkout's `outputs/task076-k0-1`. The run worktree was archived on
2026-10-10 (below); K0's report and the `--evidence` root are still only on the SSD.

**TASK-077's run worktrees** (closed 2026-10-07, G-NO-BAR; [results](experiments/apple_lewm_c1m_v2_results.md)
§1): `task077-k0`, `task077-corpus`, `task077-corpus2` (the sealed corpus), `task077-stageo`
(the featurisation and Stage O's fits), `task077-staget` and `task077-staget2` (the eight Stage T
jobs and their checkpoints) and `task077-stageg`. Stage T's and Stage G's outputs are copied to
the SSD evidence store (`~/develop/emai/evidence/task077-stage{t,t2,g}/`, sha256 manifests under
`_checksums/`). `task077-k0`, `task077-corpus`, `task077-corpus2` and `task077-stageo` were archived on
2026-10-10 and `task077-staget`, `task077-staget2` and `task077-stageg` removed (below).

**TASK-080's run worktrees** (closed 2026-10-08, L-NEAR;
[results](experiments/apple_lewm_c1m_v2_pred_readout_results.md) §1): `task080-k0`,
`task080-stagec` (the sealed fresh corpus), `task080-stager` (the fresh featurisation and the
readout fits R-S, R-N and R-L), `task080-staged` and `task080-stages`. Their reports, logs and the
Stage R fits are copied to the SSD evidence store
(`~/develop/emai/evidence/task080-{k0,stagec,stager,staged,stages}/`, sha256 manifests under
`_checksums/`); the fresh feature files (3.46 GB) are not copied and stay in `task080-stager`.
Stages D and S read the Stage R report and fits from these worktrees. On 2026-10-10 `task080-stagec` and `task080-stager` were archived (the
feature files only on the USB disk) and the other three removed (below).

**TASK-081's run worktrees** (closed 2026-10-08, L-PASS;
[results](experiments/apple_lewm_commit_precision_v2_results.md) §1): `task081-design` (the
design note's development check), `task081-staged` and `task081-stages`. Their reports, logs and
stdouts are copied to the SSD evidence store
(`~/develop/emai/evidence/task081-{dev,staged,stages}/`, sha256 manifests under `_checksums/`).
TASK-081's Stages D and S read TASK-077's featurisation and fits (`task077-stageo`) and TASK-080's
Stage R report and fits (`task080-stager`); with TASK-081 closed nothing reads them now, but any
re-run or follow-up on W-66800 and R-S does, so archive them only with their checksums verified. Done on 2026-10-10 (below); the three
TASK-081 worktrees themselves held nothing beyond the evidence store and were removed.

## Cleanup of 2026-10-10

`/` had 28.3 GB free (94 %) before this pass. The Phase 3 run (TASK-088) was running, so nothing
it reads was touched: `worktrees/task088-stage0`, `worktrees/task087-run` (TASK-087's checkpoints
and `outputs/task087-features`), the main checkout's `data/play-v1`, `.venv`, `checkpoints/` and
`third_party/` (lerobot, le-wm, dinov2-small, unitree_mujoco, jepa-wms, `jepa-wms-runtime`), the
GPU lock and `worktrees/task076-evidence`.

- **Removed, nothing archived (53 worktrees).** Every worktree whose ignored content was only
  caches, `.venv/`, `.mc/data/` index files and symlinks, or outputs whose every file's sha256
  is already in `~/develop/emai/evidence/`, removed with `git worktree remove` (no `--force`):
  the review, prereg, freeze, record and results worktrees of TASK-076 to TASK-086, plus
  `task077-stageg`, `task077-staget`, `task077-staget2`, `task077-decomp`, `task080-k0`,
  `task080-staged`, `task080-stages`, `task081-design`, `task081-staged`, `task081-stages`,
  `task082-k0`, `task083-stage0`, `task083-stages`, `task084-run`, `task085-run`, `task086-run`,
  `fix-ci-pr-a`, `task076-prereg-draft` and `wt-task076-freeze-prep`. The list is
  `archive/removed-worktrees-2026-10-10.txt` on the disk and
  `evidence/_archive-extras/removed-worktrees-2026-10-10.txt`. Their 38 merged local branches
  were deleted (`deleted-local-branches-2026-10-10.txt` in both places, `name sha #PR`; every
  tip is a merged PR's head on GitHub). The local branch `ci/integration-timeout-30` holds an
  unpushed commit (`abaafdb`, the macOS-timeout change, needs the `workflow` token scope) and was
  kept; its patch is in `archive/patches/ci-integration-timeout-30/` and
  `evidence/_archive-extras/patches/ci-integration-timeout-30/`.
- **Archived, SSD originals kept (30 worktrees).** `task076-run`, `task076-stage0`,
  `task077-{k0,corpus,corpus2,cprobe,freeze,memfix,prereg,stage0,stageo,stageo-prep,staget-prep}`,
  `task080-{stage0,stagec,stager}`, `task081-stage0`, `task082-{design,s0chain,s0run,stage0}`,
  `task085-{prereg,stage0}`, `task087-stage0`, `c1-feasibility`, `c1m-run`, `c1m-debug`,
  `next-direction-r15`, `arena-e9b` and `/home/huhn/develop/emai/oej-isaac-newton`. Each was
  copied to `archive/worktrees/<name>/` (without `.venv/`, tool caches, symlinks and the
  Isaac/Omniverse `home_cache/` directories), every file re-read from the disk after the copy and
  matched; 23.1 GB, one split file (`task077-stageo`'s `features8_train.npy`, 9.55 GB). The
  failing disk is not the only copy of anything cited: their `outputs/`, `data/`,
  `checkpoints/`, `assets/isaac/` and two untracked files (3.0 GB) are also on the SSD at
  **`/home/huhn/develop/emai/evidence/worktrees/<name>/`**, mirroring the worktree's paths
  (manifests `evidence/_checksums/worktrees/<name>.sha256`). Only on the disk: TASK-077's and
  TASK-080's frozen-DINOv2 feature arrays (`outputs/task0{77,80}-featurise-1/features/`, 13.8 GB
  and 3.5 GB; regenerable featurisation output, their reports, logs and fits are on the SSD) and
  Stage 0 smoke/debug files of 10 MB or more (`task077-stage0`, `task082-s0chain`,
  `task087-stage0`; 2.4 GB). The rows are in the disk's `MANIFEST.md`, "Third pass". **Removing
  these 30 worktrees from the SSD is left to the owner** (it was not permitted in this session);
  until then read them in place. Once removed, an old path
  `/home/huhn/develop/emai/worktrees/<name>/X` maps to `evidence/worktrees/<name>/X` (outputs) or
  `archive/worktrees/<name>/X` (everything). The `tail -F` process in `oej-isaac-newton` (pid
  260505) only reads a log; it does not block archiving.
- **Second copies of data on the disk, SSD originals kept:** `datasets/real-g1-v1/` (TASK-086's
  converted stores, unused since R23.28, 2.0 GB; the protocol allows deleting the SSD copy if
  space is needed, recorded) and `datasets/jepa-wms-runtime-downloads/` (TASK-084's pinned
  `pusht_noise.zip` and checkpoint). `third_party/jepa-wms-runtime/dset/` (7.0 GB) matches the
  pinned zip member for member (CRC-32 of all 18 718 files, none extra), so it can be deleted and
  regenerated with `unzip -q downloads/pusht_noise.zip -d dset`; it was not deleted. (The
  `downloads/SHA256SUMS` file lists another value for the zip; the zip on disk has the runner's
  pin, `442f5dee…2da08`.)
- After this pass `/` had about 30 GB free. Removing the 30 archived worktrees would free about
  30 GB more (28.6 GiB by `du`, with files hard-linked to kept checkouts counted against those),
  and deleting `real-g1-v1` and `jepa-wms-runtime/dset` about 9 GB.

## Old path to archive path

Results documents cite worktree paths on the SSD. Each worktree below was copied, every file
re-hashed on the disk and matched against the source, and then removed from the SSD with
`git worktree remove --force` (branches were kept). Replace the old prefix with the archive prefix:

| old path (removed from the SSD) | archive path | branch @ HEAD | PR |
|---|---|---|---|
| `/home/huhn/develop/emai/worktrees/task074-results-review/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task074-results-review/` | detached (on origin/docs/task-074-results) @ `75b1eeee33ad` | #116 |
| `/home/huhn/develop/emai/worktrees/task075-results-review/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task075-results-review/` | detached (on origin/docs/task-075-results) @ `082b1570d2c0` | #118 |
| `/home/huhn/develop/emai/worktrees/arena-e9-review/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/arena-e9-review/` | detached (on origin/feat/arena-e9-crosssim) @ `6b22ebdeb53c` | #120 |
| `/home/huhn/develop/emai/worktrees/task075-prereg-review/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task075-prereg-review/` | detached (on origin/docs/task-075-obs-ceiling-prereg) @ `c9163ac6dd01` | #117 |
| `/home/huhn/develop/emai/worktrees/task074-a2-review/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task074-a2-review/` | detached (on origin/fix/task-074-a2-budget) @ `f75b9febb141` | #115 |
| `/home/huhn/develop/emai/worktrees/task074-memfix-review/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task074-memfix-review/` | detached (on origin/fix/task-074-memfix) @ `96934ea63ae3` | #114 |
| `/home/huhn/develop/emai/worktrees/arena-spike-review/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/arena-spike-review/` | detached (on origin/feat/arena-g1-pick-place-spike) @ `cf58361e43b9` | #119 |
| `/home/huhn/develop/emai/worktrees/task072-results/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task072-results/` | task072-results @ `eb103b4ffb1b` | #100 |
| `/home/huhn/develop/emai/worktrees/task073-results/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task073-results/` | docs/task-073-results @ `e0ab18d23e59` | #109 |
| `/home/huhn/develop/emai/worktrees/task073-k0/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task073-k0/` | detached (on origin/HEAD->origin/main, origin/archive/arena-gr00t-baseline-runs, origin/backup/arena-e9-adapted-declared-3146533) @ `b4df3f041bf3` | #106 |
| `/home/huhn/develop/emai/worktrees/task074-design/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task074-design/` | design/task074-lewm-planner @ `4d3e59622d92` | #109 |
| `/home/huhn/develop/emai/worktrees/task073-k0-2/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task073-k0-2/` | detached (on origin/HEAD->origin/main, origin/archive/arena-gr00t-baseline-runs, origin/backup/arena-e9-adapted-declared-3146533) @ `35772e5e8564` | #108 |
| `/home/huhn/develop/emai/worktrees/task074-run2/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task074-run2/` | detached (on origin/HEAD->origin/main, origin/archive/arena-gr00t-baseline-runs, origin/backup/arena-e9-adapted-declared-3146533) @ `f52c905a423a` | #114 |
| `/home/huhn/develop/emai/worktrees/task074-fix/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task074-fix/` | fix/task074-run-guards @ `d2c2b138dd47` | #111 |
| `/home/huhn/develop/emai/worktrees/task074-results/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task074-results/` | docs/task-074-results @ `75b1eeee33ad` | #116 |
| `/home/huhn/develop/emai/worktrees/task075-results/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task075-results/` | docs/task-075-results @ `082b1570d2c0` | #118 |
| `/home/huhn/develop/emai/worktrees/task073-go/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task073-go/` | detached (on origin/HEAD->origin/main, origin/archive/arena-gr00t-baseline-runs, origin/backup/arena-e9-adapted-declared-3146533) @ `b4df3f041bf3` | #106 |
| `/home/huhn/develop/emai/worktrees/task073-go-2/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task073-go-2/` | detached (on origin/HEAD->origin/main, origin/archive/arena-gr00t-baseline-runs, origin/backup/arena-e9-adapted-declared-3146533) @ `35772e5e8564` | #108 |
| `/home/huhn/develop/emai/worktrees/isaac-e9/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/isaac-e9/` | feat/isaac-e9-replay @ `fad9780afb9e` | #110 |
| `/home/huhn/develop/emai/worktrees/task074-run3/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task074-run3/` | detached (on origin/HEAD->origin/main, origin/archive/arena-gr00t-baseline-runs, origin/backup/arena-e9-adapted-declared-3146533) @ `5e53ef37e749` | #115 |
| `/home/huhn/develop/emai/worktrees/task074-a2/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task074-a2/` | fix/task-074-a2-budget @ `f75b9febb141` | #115 |
| `/home/huhn/develop/emai/worktrees/task074-memfix/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task074-memfix/` | fix/task-074-memfix @ `96934ea63ae3` | #114 |
| `/home/huhn/develop/emai/worktrees/task075-prereg/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task075-prereg/` | docs/task-075-obs-ceiling-prereg @ `c9163ac6dd01` | #117 |
| `/home/huhn/develop/emai/oej-isaac-spike/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/oej-isaac-spike/` | isaac-bringup-spike @ `dbc05bfc241b` | #101 |
| `/home/huhn/develop/emai/oej-isaac-v2/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/oej-isaac-v2/` | feat/task-025-isaac-v2-scene @ `d1cb782919eb` | #105 |
| `/home/huhn/develop/emai/oej-isaac-mjcf/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/oej-isaac-mjcf/` | feat/task-025-isaac-mjcf-usd @ `935405b29cfe` | #103 |
| `/home/huhn/develop/emai/worktrees/arena-e9/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/arena-e9/` | feat/arena-e9-crosssim @ `6b22ebdeb53c` | #120 |
| `/home/huhn/develop/emai/worktrees/task074-prereg/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task074-prereg/` | docs/task074-lewm-planner-prereg @ `1b939279823c` | #113 |
| `/home/huhn/develop/emai/worktrees/task073-fix/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task073-fix/` | fix/task-073-memory @ `fa390abfafe3` | #108 |
| `/home/huhn/develop/emai/worktrees/arena-spike/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/arena-spike/` | feat/arena-g1-pick-place-spike @ `cf58361e43b9` | #119 |
| `/home/huhn/develop/emai/worktrees/task072-m2/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task072-m2/` | task072-m2-results @ `c61f9a8475ed` | #104 |
| `/home/huhn/develop/emai/worktrees/task072-m2-run/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task072-m2-run/` | detached (on origin/HEAD->origin/main, origin/archive/arena-gr00t-baseline-runs, origin/backup/arena-e9-adapted-declared-3146533) @ `23e25937b1f1` | #102 |
| `/home/huhn/develop/emai/worktrees/task072-a/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task072-a/` | detached (on origin/archive/task072-a-wip) @ `4ae387c2f612` | - |
| `/home/huhn/develop/emai/worktrees/isaac-usd-threads/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/isaac-usd-threads/` | fix/isaac-usd-thread-limit @ `561f257a70fd` | #112 |
| `/home/huhn/develop/emai/worktrees/docs-pr-b/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/docs-pr-b/` | docs/post-task075-sweep @ `5c78ec2b45f2` | #126 |
| `/home/huhn/develop/emai/worktrees/task075-run/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task075-run/` | detached (on origin/main) @ `34345382cbf2` | TASK-075 run (results #118) |
| `/home/huhn/develop/emai/worktrees/white-plate-dev/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/white-plate-dev/` | feat/white-plate-variant @ `f88edde1b170` | #122 |
| `/home/huhn/develop/emai/worktrees/arena-gr00t/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/arena-gr00t/` | feat/arena-gr00t-baseline @ `7b387df3b461` | #121 |
| `/home/huhn/develop/emai/worktrees/task074-run/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task074-run/` | detached (on origin/main) @ `9d9b03cff448` | TASK-074 run (results #116) |
| `/home/huhn/develop/emai/worktrees/task072-run/` | `/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task072-run/` | detached (on origin/main) @ `db65816eb56d` | TASK-072 run (results #100, #104) |

For example `/home/huhn/develop/emai/worktrees/task074-run3/outputs/...` is now
`/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task074-run3/outputs/...`.

The last six rows (`docs-pr-b` to `task072-run`) were copied and verified on 2026-10-02 but kept on
the SSD until 2026-10-04, when each was re-verified against its checksum file (file count and
sha256) immediately before removal, which left 111 GB free on `/`.
`task075-run/data/apple-far-shift-v2` was a symlink to `task074-run/data/apple-far-shift-v2`; on the
disk the corpus is under `worktrees/task074-run/data/`, and the `apple-far-shift-v2-views` store
under `worktrees/task075-run/data/`. Both were restored from the disk to the main checkout on
2026-10-04 for TASK-076, as `/home/huhn/develop/emai/open-embodied-jepa/data/apple-far-shift-v2`
(601 files) and `.../data/apple-far-shift-v2-views` (271 files), every file checked against the
archived sha256 (each `manifest.json` matches the protocol's hash). A worktree that reads them
symlinks them from there (`scripts/new_worktree.sh --run` links `data/`) rather than copying them.
`task072-run` was the `--evidence` root of the TASK-074/075 stages.

### Other items on the disk

- `run-logs/`: the 16 loose run-launcher logs (`task072-*.log` to `task075-*.log`,
  `task073-k0-*uptime.txt`) that sat directly in `~/develop/emai/worktrees/`; verified against
  `checksums/run-logs.sha256` and removed from the SSD. Results documents that cite
  `~/develop/emai/worktrees/<file>.log` now find it at `archive/run-logs/<file>.log`.
- `patches/fix-ci-integration-workflow/`: `git format-patch` of the local branch
  `fix/ci-integration-workflow`, whose commit `7eb696e` (the F10 integration-workflow change) could
  not be pushed for lack of `workflow` token scope. The branch is kept in the local repository.
- `deleted-remote-branches-2026-10-02.txt`: the 83 origin branches deleted after their PRs merged,
  as `name sha`. Several HEADs in the tables above are given as "on origin/<branch>"; those branches
  are gone from origin, but each commit is still on GitHub as `refs/pull/<N>/head` (the PR column),
  and the local branches were kept. Fetch one with
  `git fetch origin refs/pull/<N>/head:refs/remotes/origin/pr/<N>`.

### Not archived

Still only on the SSD, as of 2026-10-04 (a snapshot, superseded by the cleanup of 2026-10-10
above; run `git worktree list` in the main checkout for the current set):

- the main checkout;
- `arena-e9b` (PR #123; a run is in progress in it);
- `oej-isaac-newton` (below);
- `fix-ci-pr-a` (`fix/ci-stabilise-imports` @ `51c6600`, squash-merged as #124);
- `task076-prereg-draft` (`docs/task076-prereg-draft`, PR #129, merged), which holds only generated
  `.mc/data/*.json` index files beyond git: `scripts/remove_worktree.sh` refuses it without
  `--force-ignored`, so it was kept;
- `storage-archive-3` (this document's PR);
- `/home/huhn/develop/emai/wt-task076-freeze-prep` (`docs/task076-freeze-prep`, another session's);
- `/home/huhn/develop/emai/worktrees/task076-evidence` (a plain directory, not a worktree): the
  `--evidence` root that TASK-076's runner pins; keep it until TASK-076 is closed (above);
- directories of other projects (`ar-v1`, `t1-cluster`, `v2d-toolkit-8c778f9`).

The review and PR worktrees `docs-review`, `fix-ci-review`, `followup-nits`, `tooling-pr-c`,
`tooling-review`, `white-plate-review`, `arena-gr00t-review`, `fix-test-leak` and
`storage-archive-2` held no evidence of their own (every HEAD is on GitHub) and were removed with
`scripts/remove_worktree.sh` on 2026-10-04, without archiving.

`/home/huhn/develop/emai/oej-isaac-newton` (`feat/task-025-isaac-newton`) was archived on
2026-10-10 (above; its 2.8 GB were mostly `home_cache/`, not archived).

## Restore

Restore from the SSD copy (`/home/huhn/develop/emai/evidence/<name>/`, above) when it holds what
you need; it is verified and does not stress the failing disk. The steps below restore a whole
worktree, `.venv/` included, from the disk.

Restore to the row's recorded original path (the table's first column, or MANIFEST.md's
"original path"), not to a fixed prefix: most rows lived under `~/develop/emai/worktrees/`, but
`oej-isaac-spike`, `oej-isaac-v2` and `oej-isaac-mjcf` lived directly under `~/develop/emai/`.

```sh
NAME=task074-run3                                        # worktree name from the table
DEST=/home/huhn/develop/emai/worktrees/task074-run3      # that row's original path
SHA=<HEAD from archive/MANIFEST.md>
A=/media/huhn/INTENSO/emai/open-embodied-jepa/archive
git -C ~/develop/emai/open-embodied-jepa worktree add --detach "$DEST" $SHA
rsync -rt "$A/worktrees/$NAME/" "$DEST/"
cd "$DEST"
# rejoin any split file listed in MANIFEST.md:  cat big.bin.part-* > big.bin && rm big.bin.part-*
# recreate skipped symlinks from MANIFEST.md:   ln -s <target> <path>   (e.g. third_party)
sha256sum -c "$A/checksums/$NAME.sha256"                 # verify every file
```

Recreated symlinks point back at the SSD, so they resolve only while their sources there exist:
for example `data/apple-far-shift-v2` (to `task074-run/data/apple-far-shift-v2`) or
`outputs/task073-scratch` (to the main checkout's `outputs/task073-scratch`). If a source has since been archived or removed, point the link at its
new location or restore the source first.

`.venv/` is restored without executable bits (FAT32 has no permissions); recreating it with
`uv sync --locked --extra ...` (see [SETUP.md](SETUP.md)) is simpler than repairing it. A rebuilt
`.venv/` does not match the archived hashes, so in that case leave it out of the check (only some
checksum files list `.venv/` at all):

```sh
grep -v '  \.venv/' "$A/checksums/$NAME.sha256" | sha256sum -c -
```

## FAT32 caveats

- Files of 4 GiB or more do not fit: they are stored split with `split -b 3900M -d -a 3` as
  `<file>.part-000`, `<file>.part-001`, ... and listed in MANIFEST.md; rejoin with `cat`.
  (No archived worktree so far had such a file.)
- No symlinks: they are skipped and recorded in MANIFEST.md as `path → target`.
- No unix permissions or ownership; executable bits are lost.
- Case-insensitive names, and `: * ? " < > | \` are forbidden; a renamed file would be recorded in
  MANIFEST.md (none so far).
- 2-second timestamp resolution.
