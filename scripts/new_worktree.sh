#!/usr/bin/env bash
# Create a git worktree that runs its own code (the 2026-10-02 audit, F5).
#
# Usage: scripts/new_worktree.sh <dir> [--from <rev>] [--branch <name>] [--run] [--no-sync]
#
#   <dir>            the new worktree; must not exist.
#   --from <rev>     what to check out (default origin/main, fetched first). A local branch name
#                    without --branch checks that branch out; anything else is a detached HEAD.
#   --branch <name>  create <name> at <rev> and check it out (git worktree add -b).
#   --run            also symlink data/ and checkpoints/ from the shared checkout, for a run that
#                    reads existing evidence. outputs/ is never linked: every run writes its own.
#   --no-sync        skip `uv sync` (the import check is skipped too).
#
# Always symlinked from the shared checkout (SHARED_ROOT, default: the repository's main
# worktree): third_party/ (required; fetched pinned upstream sources and the G1/Dex3 MuJoCo
# assets) and, when present there, assets/isaac/ (converted USDs). Then it runs
# `uv sync --locked` with every extra into the worktree's own .venv, so the editable install
# points at <dir>/src (a shared venv's install points at another checkout; audit F1), and checks
# that `import embodied_jepa` resolves to <dir>/src. Remove a worktree with
# scripts/remove_worktree.sh.
set -euo pipefail

usage() { sed -n '2,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//' >&2; exit 2; }

DIR="" FROM="" BRANCH="" RUN=0 SYNC=1
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM=${2:?--from needs a revision}; shift 2 ;;
    --branch) BRANCH=${2:?--branch needs a name}; shift 2 ;;
    --run) RUN=1; shift ;;
    --no-sync) SYNC=0; shift ;;
    -h|--help) usage ;;
    -*) echo "unknown option $1" >&2; usage ;;
    *) [ -z "$DIR" ] || usage; DIR=$1; shift ;;
  esac
done
[ -n "$DIR" ] || usage
[ -e "$DIR" ] && { echo "refusing: $DIR already exists" >&2; exit 1; }

REPO=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
COMMON=$(cd -- "$REPO" && cd -- "$(git rev-parse --git-common-dir)" && pwd -P)
SHARED_ROOT=${SHARED_ROOT:-$(dirname -- "$COMMON")}
[ -d "$SHARED_ROOT/third_party" ] || {
  echo "refusing: $SHARED_ROOT/third_party is missing; run the scripts/fetch_*.py there first" >&2
  exit 1
}

if [ -z "$FROM" ]; then
  git -C "$REPO" fetch --quiet origin main
  FROM=origin/main
fi
git -C "$REPO" rev-parse --verify --quiet "$FROM^{commit}" > /dev/null \
  || { echo "refusing: $FROM is not a revision" >&2; exit 1; }

if [ -n "$BRANCH" ]; then
  git -C "$REPO" worktree add -b "$BRANCH" "$DIR" "$FROM"
elif git -C "$REPO" show-ref --verify --quiet "refs/heads/$FROM"; then
  git -C "$REPO" worktree add "$DIR" "$FROM"
else
  git -C "$REPO" worktree add --detach "$DIR" "$FROM"
fi
DIR=$(cd -- "$DIR" && pwd -P)

link() {  # link <relative path>: <dir>/<path> -> <shared>/<path>
  mkdir -p "$(dirname -- "$DIR/$1")"
  ln -s "$SHARED_ROOT/$1" "$DIR/$1"
  echo "linked $1 -> $SHARED_ROOT/$1"
}
link third_party
if [ -d "$SHARED_ROOT/assets/isaac" ]; then link assets/isaac; else echo "no assets/isaac in $SHARED_ROOT (Isaac USDs not linked)"; fi
if [ "$RUN" = 1 ]; then
  for d in data checkpoints; do
    if [ -d "$SHARED_ROOT/$d" ]; then link "$d"; else echo "no $d/ in $SHARED_ROOT (not linked)"; fi
  done
fi

echo "worktree: $DIR"
echo "revision: $(git -C "$DIR" rev-parse HEAD) ($(git -C "$DIR" rev-parse --abbrev-ref HEAD))"
if [ "$SYNC" = 1 ]; then
  (cd -- "$DIR" && uv sync --locked --extra learning --extra sim --extra lewm --extra data \
    --extra compatibility --extra pretrained --extra jepa-wms)
  FILE=$(cd -- "$DIR" && uv run --no-sync python -c \
    'import embodied_jepa, pathlib; print(pathlib.Path(embodied_jepa.__file__).resolve())')
  echo "embodied_jepa: $FILE"
  case "$FILE" in
    "$DIR/src/embodied_jepa/"*) echo "check: OK, this worktree imports its own src" ;;
    *) echo "check: FAILED, embodied_jepa is not imported from $DIR/src" >&2; exit 1 ;;
  esac
else
  echo "check: skipped (--no-sync); run 'uv sync --locked' with the extras before using it"
fi
