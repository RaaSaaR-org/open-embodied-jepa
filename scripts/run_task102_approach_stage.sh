#!/usr/bin/env bash
# TASK-102 Δa (docs/experiments/so101_approach_diagnostic_v1.md): one GPU lock per GPU job
# (scripts/gpu_run.sh --wait --board); CPU jobs (lambda0, O-pose) take no lock.
#
# Usage: scripts/run_task102_approach_stage.sh <stage> [args]
#   goals <L|debug> [--limit N]          goal demonstrations and goal store (GPU)
#   lambda0                              §2: Stage 0's A logs re-read (CPU)
#   landscape <goals-dir> [slots] [tag]  §3: Λ1 on the kept slots (GPU)
#   approach <goals-dir> <tag> <arm...>  §4: Λ2 GPU arms (J-ref J-big16 J-big32 O-img)
#   pose <goals-dir> <tag>               §4: Λ2 O-pose (CPU)
#   debug-approach <goals-dir> <tag> <timeout> <arm...>   timing on debug seeds only
#   summary                              the readings and the row
#
# Outputs: the shared checkout's outputs/task102-approach-diag/. Every job refuses an existing
# output. MUJOCO_GL=egl for every rendering job.
set -euo pipefail
REPO=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cd "$REPO"
SHARED=$(dirname -- "$(cd -- "$(git rev-parse --git-common-dir)" && pwd -P)")
OUT=${TASK102_OUT:-$SHARED/outputs}/task102-approach-diag
PY=.venv/bin/python
WORKERS=${DA_WORKERS:-8}
GPU=(scripts/gpu_run.sh --wait --min-free-gib 8 --board --who oej:task102-approach-diag --)
export MUJOCO_GL=egl
mkdir -p "$OUT"

sha() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))['goals_sha256'])" "$1/goals_report.json"; }

stage=${1:-}; shift || true
case "$stage" in
  goals)
    cohort=${1:?cohort}; shift
    "${GPU[@]}" $PY scripts/run_task102_approach.py goals --cohort "$cohort" "$@" \
      --output "$OUT/$cohort-goals" </dev/null 2>&1 | tee -a "$OUT/goals-$cohort.log" ;;
  lambda0)
    $PY scripts/run_task102_approach.py lambda0 --stage0 "$SHARED/outputs/task102-stage0" \
      --output "$OUT/lambda0.json" </dev/null 2>&1 | tee -a "$OUT/lambda0.log" ;;
  landscape)
    goals=${1:?goals}; slots=${2:-}; tag=${3:-landscape}
    extra=(); if [ -n "$slots" ]; then extra=(--slots "$slots"); fi
    "${GPU[@]}" $PY scripts/run_task102_approach.py landscape --goals "$goals" \
      --goals-sha256 "$(sha "$goals")" --output "$OUT/$tag" --workers "$WORKERS" "${extra[@]}" \
      </dev/null 2>&1 | tee -a "$OUT/$tag.log" ;;
  approach)
    goals=${1:?goals}; tag=${2:?tag}; shift 2
    "${GPU[@]}" $PY scripts/run_task102_approach.py approach --goals "$goals" \
      --goals-sha256 "$(sha "$goals")" --output "$OUT/$tag" --workers "$WORKERS" --arms "$@" \
      </dev/null 2>&1 | tee -a "$OUT/$tag.log" ;;
  pose)
    goals=${1:?goals}; tag=${2:?tag}
    $PY scripts/run_task102_approach.py approach --goals "$goals" --goals-sha256 "$(sha "$goals")" \
      --output "$OUT/$tag" --workers "$WORKERS" --device none --arms O-pose \
      </dev/null 2>&1 | tee -a "$OUT/$tag.log" ;;
  debug-approach)
    goals=${1:?goals}; tag=${2:?tag}; timeout=${3:?timeout}; shift 3
    dev=cuda; if [ "$*" = "O-pose" ]; then dev=none; fi
    run=($PY scripts/run_task102_approach.py approach --goals "$goals" \
      --goals-sha256 "$(sha "$goals")" --output "$OUT/$tag" --workers "$WORKERS" --device "$dev" \
      --debug-timeout "$timeout" --arms "$@")
    if [ "$dev" = cuda ]; then run=("${GPU[@]}" "${run[@]}"); fi
    "${run[@]}" </dev/null 2>&1 | tee -a "$OUT/$tag.log" ;;
  summary)
    $PY scripts/summarize_task102_approach.py --root "$OUT" --output "$OUT/summary.json" \
      </dev/null 2>&1 | tee -a "$OUT/summary.log" ;;
  *)
    echo "unknown stage: $stage" >&2; exit 2 ;;
esac
