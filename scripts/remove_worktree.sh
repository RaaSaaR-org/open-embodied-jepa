#!/usr/bin/env bash
# Remove a git worktree only when nothing in it can be lost (the 2026-10-02 audit, F5).
#
# Usage: scripts/remove_worktree.sh <dir> [--dry-run]
#
# Refuses (exit 1) when the worktree
#   * is the main worktree,
#   * has modified, staged or untracked files (`git status --porcelain`),
#   * has commits that are on no remote-tracking branch (`git rev-list HEAD --not --remotes`;
#     run `git fetch` first so the remote refs are current), or
#   * holds run evidence of its own: data/, checkpoints/ or outputs/ as a real, non-empty
#     directory (a symlink to the shared checkout is fine). Archive evidence first.
# Otherwise it runs `git worktree remove <dir>`, which deletes the symlinks made by
# scripts/new_worktree.sh but never what they point to. The branch is kept.
set -euo pipefail

usage() { sed -n '2,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//' >&2; exit 2; }
DIR="" DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    -h|--help) usage ;;
    -*) echo "unknown option $1" >&2; usage ;;
    *) [ -z "$DIR" ] || usage; DIR=$1; shift ;;
  esac
done
[ -n "$DIR" ] || usage
[ -d "$DIR" ] || { echo "refusing: $DIR is not a directory" >&2; exit 1; }
DIR=$(cd -- "$DIR" && pwd -P)
TOP=$(git -C "$DIR" rev-parse --show-toplevel 2>/dev/null) \
  || { echo "refusing: $DIR is not a git worktree" >&2; exit 1; }
[ "$(cd -- "$TOP" && pwd -P)" = "$DIR" ] || { echo "refusing: $DIR is not a worktree root" >&2; exit 1; }
COMMON=$(cd -- "$DIR" && cd -- "$(git rev-parse --git-common-dir)" && pwd -P)
GITDIR=$(cd -- "$DIR" && cd -- "$(git rev-parse --git-dir)" && pwd -P)
[ "$COMMON" != "$GITDIR" ] || { echo "refusing: $DIR is the main worktree" >&2; exit 1; }

reasons=()
dirty=$(git -C "$DIR" status --porcelain)
[ -z "$dirty" ] || reasons+=("uncommitted or untracked files:"$'\n'"$dirty")
unpushed=$(git -C "$DIR" rev-list HEAD --not --remotes)
[ -z "$unpushed" ] || reasons+=("$(printf '%s\n' "$unpushed" | wc -l) commit(s) on no remote branch: $(git -C "$DIR" log --oneline HEAD --not --remotes | head -5 | tr '\n' ';')")
for d in data checkpoints outputs; do
  if [ -d "$DIR/$d" ] && [ ! -L "$DIR/$d" ] && [ -n "$(ls -A "$DIR/$d")" ]; then
    reasons+=("$d/ holds this worktree's own files; archive them first")
  fi
done

if [ ${#reasons[@]} -gt 0 ]; then
  echo "refusing to remove $DIR:" >&2
  printf '  - %s\n' "${reasons[@]}" >&2
  exit 1
fi
if [ "$DRY" = 1 ]; then
  echo "would remove $DIR ($(git -C "$DIR" rev-parse --short HEAD), nothing unpushed)"
  exit 0
fi
git -C "$COMMON" worktree remove "$DIR"
echo "removed $DIR"
