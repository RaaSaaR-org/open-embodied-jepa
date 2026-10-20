#!/usr/bin/env bash
# TASK-102 A1 (docs/experiments/so101_hybrid_lift_v1.md §18): one GPU lock per GPU job
# (scripts/gpu_run.sh --wait --board); O-pose jobs run on the CPU and take no lock.
#
# Usage: scripts/run_task102_a1_stage.sh <stage> [args]
#   goals <Q|debug> [--limit N]               goal demonstrations and goal store (GPU)
#   pose <goals-dir> <tag> <config...>        §18.3 step 1: O-pose (CPU)
#   j <goals-dir> <tag> <config...>           §18.3 steps 2-3: J (GPU)
#   debug <goals-dir> <tag> <timeout> <pose|J> <config...>   timing on debug seeds only
#   summary                                   the readings, the selection and the row
#
# Outputs: the shared checkout's outputs/task102-a1/. Every job refuses an existing output.
# MUJOCO_GL=egl for every rendering job.
set -euo pipefail
REPO=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cd "$REPO"
SHARED=$(dirname -- "$(cd -- "$(git rev-parse --git-common-dir)" && pwd -P)")
OUT=${TASK102_OUT:-$SHARED/outputs}/task102-a1
PY=.venv/bin/python
WORKERS=${A1_WORKERS:-8}
JWORKERS=${A1_J_WORKERS:-6}
GPU=(scripts/gpu_run.sh --wait --min-free-gib 8 --board --who oej:task102-a1 --)
export MUJOCO_GL=egl
mkdir -p "$OUT"

sha() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))['goals_sha256'])" "$1/goals_report.json"; }

stage=${1:-}; shift || true
case "$stage" in
  goals)
    cohort=${1:?cohort}; shift
    "${GPU[@]}" $PY scripts/run_task102_a1.py goals --cohort "$cohort" "$@" \
      --output "$OUT/$cohort-goals" </dev/null 2>&1 | tee -a "$OUT/goals-$cohort.log" ;;
  pose)
    goals=${1:?goals}; tag=${2:?tag}; shift 2
    $PY scripts/run_task102_a1.py grid --goals "$goals" --goals-sha256 "$(sha "$goals")" \
      --output "$OUT/$tag" --workers "$WORKERS" --device none --cost pose --configs "$@" \
      </dev/null 2>&1 | tee -a "$OUT/$tag.log" ;;
  j)
    goals=${1:?goals}; tag=${2:?tag}; shift 2
    "${GPU[@]}" $PY scripts/run_task102_a1.py grid --goals "$goals" \
      --goals-sha256 "$(sha "$goals")" --output "$OUT/$tag" --workers "$JWORKERS" --cost J \
      --configs "$@" </dev/null 2>&1 | tee -a "$OUT/$tag.log" ;;
  debug)
    goals=${1:?goals}; tag=${2:?tag}; timeout=${3:?timeout}; cost=${4:?cost}; shift 4
    dev=cuda; if [ "$cost" = pose ]; then dev=none; fi
    run=($PY scripts/run_task102_a1.py grid --goals "$goals" --goals-sha256 "$(sha "$goals")" \
      --output "$OUT/$tag" --workers "$WORKERS" --device "$dev" --cost "$cost" \
      --debug-timeout "$timeout" --configs "$@")
    if [ "$dev" = cuda ]; then run=("${GPU[@]}" "${run[@]}"); fi
    "${run[@]}" </dev/null 2>&1 | tee -a "$OUT/$tag.log" ;;
  summary)
    $PY scripts/summarize_task102_a1.py --root "$OUT" --output "$OUT/summary.json" \
      </dev/null 2>&1 | tee -a "$OUT/summary.log" ;;
  *)
    echo "unknown stage: $stage" >&2; exit 2 ;;
esac
