#!/usr/bin/env bash
# TASK-101's stages (protocol docs/experiments/so101_closed_loop_lift_v1.md §7, §8, §13), one GPU
# lock per GPU job (scripts/gpu_run.sh --wait --board); CPU arms without frames take no lock.
#
# Usage: scripts/run_task101_stage.sh <stage> [args]
#  Stage 0 (development; calibration K, development D, debug seeds; never S):
#   goals <cohort> [evidence]        goal demonstrations (+ the first N CPU-arm episodes, rendered)
#   check <name> [goals-dir]         §7.2: latent | adapter | ties | determinism | renc | projector
#   k <goals-dir>                    §7.3: W (J-100101) in the four configurations on K
#   d-cem <goals-dir> <config>       §7.4: W, N, W-shuf, W-mean on D, then W-single on D's first 16
#   d-cpu <goals-dir>                §7.4: rand, hold, F, H-pose on D's slots 8-63 (CPU, no frames)
#  Gated (a clean run tree at the frozen revision; refused before the freeze):
#   s-goals | s-cem <goals-dir> <config> [arms] | s-cpu <goals-dir>
#
# Outputs go to the shared checkout's outputs/ ($OUT/task101-stage0/..., $OUT/task101-run/...);
# every job refuses an existing output. MUJOCO_GL=egl for every rendering job.
set -euo pipefail
REPO=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cd "$REPO"
SHARED=$(dirname -- "$(cd -- "$(git rev-parse --git-common-dir)" && pwd -P)")
OUT=${TASK101_OUT:-$SHARED/outputs}
PY=.venv/bin/python
WORKERS=${TASK101_WORKERS:-6}
GPU=(scripts/gpu_run.sh --wait --min-free-gib 6 --board --who oej:task101 --)
S0=$OUT/task101-stage0
RUN=$OUT/task101-run
export MUJOCO_GL=egl

sha() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))['goals_sha256'])" "$1/goals_report.json"; }

stage=${1:-}; shift || true
case "$stage" in
  goals)
    cohort=${1:?cohort}; ev=${2:-0}
    mkdir -p "$S0"
    "${GPU[@]}" $PY scripts/run_task101.py goals --cohort "$cohort" --evidence "$ev" \
      --output "$S0/$cohort-goals" </dev/null 2>&1 | tee -a "$S0/goals-$cohort.log" ;;
  check)
    name=${1:?check}; goals=${2:-}
    extra=()
    if [ -n "$goals" ]; then extra=(--goals "$goals" --goals-sha256 "$(sha "$goals")"); fi
    "${GPU[@]}" $PY scripts/run_task101.py check "$name" "${extra[@]}" --workers "$WORKERS" \
      --output "$S0/check-$name" </dev/null 2>&1 | tee -a "$S0/check-$name.log" ;;
  k)
    goals=${1:?goals}
    "${GPU[@]}" $PY scripts/run_task101.py episodes --cohort K --goals "$goals" \
      --goals-sha256 "$(sha "$goals")" --arms W --configs H8-last,H8-mean,H16-last,H16-mean \
      --workers "$WORKERS" --output "$S0/K-episodes" </dev/null 2>&1 | tee -a "$S0/K.log" ;;
  d-cem)
    goals=${1:?goals}; cfg=${2:?config}
    "${GPU[@]}" $PY scripts/run_task101.py episodes --cohort D --goals "$goals" \
      --goals-sha256 "$(sha "$goals")" --arms W,N,W-shuf,W-mean --configs "$cfg" \
      --workers "$WORKERS" --output "$S0/D-cem" </dev/null 2>&1 | tee -a "$S0/D-cem.log"
    "${GPU[@]}" $PY scripts/run_task101.py episodes --cohort D --goals "$goals" \
      --goals-sha256 "$(sha "$goals")" --arms W-single --configs "$cfg" --slots 0:16 \
      --workers "$WORKERS" --output "$S0/D-single" </dev/null 2>&1 | tee -a "$S0/D-single.log" ;;
  d-cpu)
    goals=${1:?goals}
    $PY scripts/run_task101.py episodes --cohort D --goals "$goals" --goals-sha256 "$(sha "$goals")" \
      --arms rand,hold,F,H-pose --slots 8:64 --evidence 0 --device none --workers "$WORKERS" \
      --output "$S0/D-cpu" </dev/null 2>&1 | tee -a "$S0/D-cpu.log" ;;
  s-goals)
    mkdir -p "$RUN"
    "${GPU[@]}" $PY scripts/run_task101.py goals --cohort S --gated --evidence 8 \
      --output "$RUN/S-goals" </dev/null 2>&1 | tee -a "$RUN/goals.log" ;;
  s-cem)
    goals=${1:?goals}; cfg=${2:?config}; arms=${3:-W,W-100100,W-100102,N,W-shuf,W-mean}
    "${GPU[@]}" $PY scripts/run_task101.py episodes --cohort S --gated --goals "$goals" \
      --goals-sha256 "$(sha "$goals")" --arms "$arms" --configs "$cfg" --workers "$WORKERS" \
      --output "$RUN/S-cem" </dev/null 2>&1 | tee -a "$RUN/cem.log" ;;
  s-cpu)
    goals=${1:?goals}
    $PY scripts/run_task101.py episodes --cohort S --gated --goals "$goals" \
      --goals-sha256 "$(sha "$goals")" --arms rand,hold,F,H-pose --slots 8:128 --evidence 0 \
      --device none --workers "$WORKERS" --output "$RUN/S-cpu" </dev/null 2>&1 | tee -a "$RUN/cpu.log" ;;
  *) sed -n '2,/^set -euo/p' "$0" | grep '^#' >&2; exit 2 ;;
esac
