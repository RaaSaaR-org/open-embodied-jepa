#!/usr/bin/env bash
# TASK-094: run a stage (protocol docs/experiments/grasp_closure_locus_v1.md §10.3, §13), each GPU
# use its own scripts/gpu_run.sh job under the 1 800 s per-job cap, then summarise. Run from a
# clean tree at the frozen revision (the Stage 0′ merge, 2eacacf) for the gated stage.
#   scripts/run_task094_stage.sh gated <out> <summary-0p.json>  K, M, S, summary (gated cohorts)
#   scripts/run_task094_stage.sh dev <out>                      Stage 0's development K, M and S
#   scripts/run_task094_stage.sh 0p <out> <stage-0-dir>         Stage 0′'s comparator job on the
#                                                               stored development plans (§17)
#   scripts/run_task094_stage.sh debug <out>                    Stage 0's debug smoke (not counted)
# Every output file is refused if it exists, and the runner's ledger refuses a cohort run twice.
# The runner starts in this checkout's root (where third_party/le-wm is; R27.107) and loads its
# goals or roots and models before any ledger claim. Before each GPU job the "## Requests" section
# of ~/.local/state/gpu/BOARD.md is read; while it has a "- v2d: ..." line the job waits (up to
# 15 min), and every job takes the lock with gpu_run.sh --wait. In the gated stage no job starts
# once TASK-094's GPU time (Stage 0's and Stage 0′'s, from Stage 0′'s summary, plus this stage's
# jobs) reaches 1.0 GPU h:
# the stage writes <out>/BUDGET-STOP and its summary reads K-VOID (budget) (§10.2).
set -euo pipefail
STAGE=${1:?stage}; OUT=${2:?output dir}
cd "$(dirname "$0")/.."
[ -d third_party/le-wm ] || { echo "third_party/le-wm is missing in $(pwd)" >&2; exit 1; }
if [ "$STAGE" != debug ]; then
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "tracked tree is dirty" >&2; exit 1; }
fi
OUT=$(realpath -m "$OUT")
mkdir -p "$OUT"
PYB=$(pwd)/.venv/bin/python
PY=(env MUJOCO_GL=egl "$PYB" "$(pwd)/scripts/run_task094.py")
gl() { "$PYB" -c "import sys; sys.path.insert(0, 'src'); from embodied_jepa import grasp_closure_locus as g; $1"; }
CAP=$(gl "print(g.JOB_CAP_SECONDS)")
CEILING=$(gl "print(int(g.CEILING_GPU_HOURS * 3600))")
BOARD=${GPU_BOARD:-$HOME/.local/state/gpu/BOARD.md}
USED=0
if [ "$STAGE" = gated ]; then
  S0=${3:?Stage 0 summary}
  USED=$("$PYB" -c "import json,sys; s=json.load(open(sys.argv[1])); assert s['stage']=='0p' and s['row']=='S0-PASS', s['row']; print(int(s['gpu']['stage0_seconds']))" "$S0")
fi
v2d_waiting() {
  [ -f "$BOARD" ] || return 1
  awk '/^## /{s=($0=="## Requests")} s' "$BOARD" | grep -E '^- v2d:' | grep -vq '\[handled\]'
}
board_check() {
  local waited=0
  while v2d_waiting && [ "$waited" -lt 900 ]; do
    [ "$waited" -eq 0 ] && echo "[$(date -u +%FT%TZ)] BOARD.md has a v2d request; letting it go first" >&2
    sleep 30; waited=$((waited + 30))
  done
  echo "[$(date -u +%FT%TZ)] board check: v2d request $(v2d_waiting && echo present || echo none)" >&2
}
STOPPED=0
# gpu_run.sh exits 76 when it refuses to start (VRAM / utilisation check, before the command
# runs); only that is retried, up to 10 times a minute apart. Any other failure stops the stage.
gpu() {
  local tries=0 code t0
  if [ "$STAGE" = gated ] && [ "$USED" -ge "$CEILING" ]; then
    echo "[$(date -u +%FT%TZ)] TASK-094 GPU time $USED s reached the ceiling; no further job" >&2
    echo "$USED" > "$OUT/BUDGET-STOP"; STOPPED=1; return 0
  fi
  while :; do
    board_check
    t0=$(date +%s)
    set +e
    scripts/gpu_run.sh --wait --min-free-gib 8 --board --samples 5 --who "oej:task094-$STAGE" -- \
      timeout --kill-after=60 $((CAP + 300)) "$@" </dev/null
    code=$?
    set -e
    USED=$((USED + $(date +%s) - t0))
    printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$code" "$*" >> "$OUT/jobs.tsv"
    [ "$code" -ne 76 ] && return "$code"
    tries=$((tries + 1)); [ "$tries" -ge 10 ] && return 76
    sleep 60
  done
}
# A job that exits VOID (3: a plumbing, bitwise or determinism check failed) does not stop the
# stage before its summary: the stage records the VOID, goes on, summarises, and exits 3 itself.
VOID=0
gpuv() {
  local code=0
  gpu "$@" || code=$?
  if [ "$code" -eq 3 ]; then
    echo "[$(date -u +%FT%TZ)] job exited VOID (3): $*" >&2; VOID=3
  elif [ "$code" -ne 0 ]; then
    exit "$code"
  fi
}
ARMS=$(gl "print(' '.join(g.arm_slug(a) for a in g.K_ARMS))")
case "$STAGE" in
  gated)
    gpu "${PY[@]}" goals --cohort gated-k --output "$OUT" --workers 8
    for a in $ARMS; do
      gpuv "${PY[@]}" k-run --cohort gated-k --arm "$a" --goals "$OUT" --output "$OUT" --workers 8
    done
    for a in K-1.0 K-sstar; do
      gpuv "${PY[@]}" k-run --cohort gated-k --arm "$a" --determinism --goals "$OUT" \
        --output "$OUT" --workers 4
    done
    gpuv "${PY[@]}" roots --cohort gated-roots --output "$OUT" --workers 8
    if [ -f "$OUT/roots-gated-roots.json" ]; then
      gpuv "${PY[@]}" rank --cohort gated-roots --roots "$OUT" --output "$OUT"
      gpuv "${PY[@]}" search --cohort gated-roots --roots "$OUT" --output "$OUT"
    fi
    "$PYB" scripts/summarize_task094.py --stage gated --runs "$OUT" --s0-summary "$S0" ;;
  dev)
    gpu "${PY[@]}" goals --cohort dev-k --output "$OUT" --workers 8
    for a in $ARMS ik; do
      gpuv "${PY[@]}" k-run --cohort dev-k --arm "$a" --goals "$OUT" --output "$OUT" --workers 8
    done
    gpuv "${PY[@]}" roots --cohort dev-roots --output "$OUT" --workers 8
    gpuv "${PY[@]}" rank --cohort dev-roots --roots "$OUT" --output "$OUT"
    gpuv "${PY[@]}" search --cohort dev-roots --roots "$OUT" --output "$OUT" ;;
  0p)
    S0DIR=$(realpath -m "${3:?Stage 0 development folder}")
    gpu "${PY[@]}" substitute --cohort dev-roots --roots "$S0DIR" \
      --search "$S0DIR/search-dev-roots.json" --output "$OUT" ;;
  debug)
    N=${3:-4}
    gpu "${PY[@]}" goals --cohort debug-k --output "$OUT" --workers 4 --limit "$N"
    for a in $ARMS ik; do
      gpuv "${PY[@]}" k-run --cohort debug-k --arm "$a" --goals "$OUT" --output "$OUT" \
        --workers 4 --limit "$N"
    done
    for a in K-1.0 K-sstar; do
      gpuv "${PY[@]}" k-run --cohort debug-k --arm "$a" --determinism --goals "$OUT" \
        --output "$OUT" --workers 4 --limit "$N"
    done
    gpuv "${PY[@]}" roots --cohort debug-roots --output "$OUT" --workers 4 --limit "$N"
    gpuv "${PY[@]}" rank --cohort debug-roots --roots "$OUT" --output "$OUT"
    gpuv "${PY[@]}" search --cohort debug-roots --roots "$OUT" --output "$OUT" --check-a2 ;;
  *) sed -n '2,15p' "$0" >&2; exit 2 ;;
esac
[ "$STOPPED" -ne 0 ] && { echo "STAGE-$STAGE-BUDGET-STOP" >&2; exit 5; }
[ "$VOID" -ne 0 ] && { echo "STAGE-$STAGE-VOID" >&2; exit "$VOID"; }
echo "STAGE-$STAGE-DONE"
