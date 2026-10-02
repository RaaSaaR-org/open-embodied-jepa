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
      datasets/                     large corpora (empty for now)
      checkpoints/                  model checkpoints (empty for now)
      runs/                         generated run outputs (empty for now)
```

Nothing on the disk is the only copy of source: every archived worktree's HEAD is on origin
(`task072-a`'s detached WIP commit was pushed as `archive/task072-a-wip` for this). The disk holds
the ignored artefacts git does not track (`data/`, `outputs/`, `checkpoints/`, `.venv/`, logs).
As with `data/`, `checkpoints/` and `outputs/`, never overwrite evidence on it: add, do not replace.

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

For example `/home/huhn/develop/emai/worktrees/task074-run3/outputs/...` is now
`/media/huhn/INTENSO/emai/open-embodied-jepa/archive/worktrees/task074-run3/outputs/...`.

Not archived, still on the SSD, as of 2026-10-02 (a snapshot; run `git worktree list` in the main
checkout for the current set): the main checkout; `task072-run` (the `--evidence` root of the
TASK-074/075 stages); `task074-run` (its `data/apple-far-shift-v2` is the source corpus that
`task075-run` and other worktrees symlink to); `task075-run`; worktrees in active use
(`white-plate-dev`, `arena-e9b`, `arena-gr00t`, `arena-gr00t-review`); and directories of other
projects (`ar-v1`, `t1-cluster`).

Skipped for now, still on the SSD: `/home/huhn/develop/emai/oej-isaac-newton` (2.8 GB,
`feat/task-025-isaac-newton`). A process (`tail -F .../outputs/isaac-newton-scripted-dev-3/log.txt`)
holds a file inside it open, so it was not archived; archive it the same way once that process ends.

## Restore

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
