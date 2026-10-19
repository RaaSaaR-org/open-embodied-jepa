#!/usr/bin/env bash
# TASK-101's stages (protocol docs/experiments/so101_closed_loop_lift_v1.md §7, §8, §13, §18), one
# GPU lock per GPU job (scripts/gpu_run.sh --wait --board); CPU jobs without frames take no lock.
# Every job runs so101_cube_lift_v1 (A1). K and D are spent (A1, §18.5); Stage 0' runs K', D', X.
#
# Usage: scripts/run_task101_stage.sh <stage> [args]
#  Stage 0' (development; K', D', X and debug seeds; never S). Order per cohort (§18.5's rule,
#  enforced by the runner's ledger): goals -> checks -> episodes.
#   goals <cohort> [--limit N]       goal demonstrations only (Kp | Dp | X | debug)
#   check <name> [goals-dir] [tag]   §7.2: scene | latent | renc | projector | determinism | inert
#                                    (reported: ties | adapter | render)
#   k <goals-dir>                    §7.3 on K': W (J-100101) in the four configurations
#   d-cem <goals-dir> <config>       §7.4 on D': W, N, W-shuf, W-mean
#   d-single <goals-dir> <config>    §7.4's development diagnostic on the first 16 D' resets
#   d-cpu-ev <goals-dir>             rand, hold, F, H-pose on D' slots 0-7, frames kept (locked)
#   d-cpu <goals-dir>                rand, hold, F, H-pose on D' slots 8-63 (CPU, no frames)
#   delta <goals-dir> <config>       §18.6's diagnostic on X (only under S0-NULL)
#  Gated (a clean run tree at the frozen revision; refused before the freeze):
#   s-goals | s-cem <goals-dir> <config> [arms] | s-cpu-ev <goals-dir> | s-cpu <goals-dir>
#
# Outputs go to the shared checkout's outputs/ ($OUT/task101-stage0b/..., $OUT/task101-run/...);
# every job refuses an existing output. The check ledger is $OUT/task101-stage0b/ledger.jsonl.
# MUJOCO_GL=egl for every rendering job.
set -euo pipefail
REPO=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cd "$REPO"
SHARED=$(dirname -- "$(cd -- "$(git rev-parse --git-common-dir)" && pwd -P)")
OUT=${TASK101_OUT:-$SHARED/outputs}
PY=.venv/bin/python
WORKERS=${TASK101_WORKERS:-6}
GPU=(scripts/gpu_run.sh --wait --min-free-gib 6 --board --who oej:task101-s0b --)
S0=$OUT/task101-stage0b
RUN=$OUT/task101-run
export MUJOCO_GL=egl

sha() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))['goals_sha256'])" "$1/goals_report.json"; }

stage=${1:-}; shift || true
case "$stage" in
  goals)
    cohort=${1:?cohort}; shift
    mkdir -p "$S0"
    "${GPU[@]}" $PY scripts/run_task101.py goals --cohort "$cohort" "$@" \
      --output "$S0/$cohort-goals" </dev/null 2>&1 | tee -a "$S0/goals-$cohort.log" ;;
  check)
    name=${1:?check}; goals=${2:-}; tag=${3:-$name}
    extra=()
    if [ -n "$goals" ]; then extra=(--goals "$goals" --goals-sha256 "$(sha "$goals")"); fi
    mkdir -p "$S0"
    case "$name" in
      scene|inert|projector)  # CPU only, no frames: no lock
        $PY scripts/run_task101.py check "$name" "${extra[@]}" --device cpu \
          --output "$S0/check-$tag" </dev/null 2>&1 | tee -a "$S0/check-$tag.log" ;;
      *)
        "${GPU[@]}" $PY scripts/run_task101.py check "$name" "${extra[@]}" --workers "$WORKERS" \
          --output "$S0/check-$tag" </dev/null 2>&1 | tee -a "$S0/check-$tag.log" ;;
    esac ;;
  k)
    goals=${1:?goals}
    "${GPU[@]}" $PY scripts/run_task101.py episodes --cohort Kp --goals "$goals" \
      --goals-sha256 "$(sha "$goals")" --arms W --configs H8-last,H8-mean,H16-last,H16-mean \
      --workers "$WORKERS" --output "$S0/Kp-episodes" </dev/null 2>&1 | tee -a "$S0/Kp.log" ;;
  d-cem)
    goals=${1:?goals}; cfg=${2:?config}
    "${GPU[@]}" $PY scripts/run_task101.py episodes --cohort Dp --goals "$goals" \
      --goals-sha256 "$(sha "$goals")" --arms W,N,W-shuf,W-mean --configs "$cfg" \
      --workers "$WORKERS" --output "$S0/Dp-cem" </dev/null 2>&1 | tee -a "$S0/Dp-cem.log" ;;
  d-single)
    goals=${1:?goals}; cfg=${2:?config}
    "${GPU[@]}" $PY scripts/run_task101.py episodes --cohort Dp --goals "$goals" \
      --goals-sha256 "$(sha "$goals")" --arms W-single --configs "$cfg" --slots 0:16 \
      --workers "$WORKERS" --output "$S0/Dp-single" </dev/null 2>&1 | tee -a "$S0/Dp-single.log" ;;
  d-cpu-ev)
    goals=${1:?goals}
    "${GPU[@]}" $PY scripts/run_task101.py episodes --cohort Dp --goals "$goals" \
      --goals-sha256 "$(sha "$goals")" --arms rand,hold,F,H-pose --slots 0:8 --evidence 8 \
      --device none --workers "$WORKERS" --output "$S0/Dp-cpu-ev" </dev/null 2>&1 \
      | tee -a "$S0/Dp-cpu-ev.log" ;;
  d-cpu)
    goals=${1:?goals}
    $PY scripts/run_task101.py episodes --cohort Dp --goals "$goals" --goals-sha256 "$(sha "$goals")" \
      --arms rand,hold,F,H-pose --slots 8:64 --evidence 0 --device none --workers "$WORKERS" \
      --output "$S0/Dp-cpu" </dev/null 2>&1 | tee -a "$S0/Dp-cpu.log" ;;
  delta)
    goals=${1:?goals}; cfg=${2:?config}
    "${GPU[@]}" $PY scripts/run_task101.py delta --goals "$goals" --goals-sha256 "$(sha "$goals")" \
      --config "$cfg" --output "$S0/delta" </dev/null 2>&1 | tee -a "$S0/delta.log" ;;
  s-goals)
    mkdir -p "$RUN"
    "${GPU[@]}" $PY scripts/run_task101.py goals --cohort S --gated \
      --output "$RUN/S-goals" </dev/null 2>&1 | tee -a "$RUN/goals.log" ;;
  s-cem)
    goals=${1:?goals}; cfg=${2:?config}; arms=${3:-W,W-100100,W-100102,N,W-shuf,W-mean}
    "${GPU[@]}" $PY scripts/run_task101.py episodes --cohort S --gated --goals "$goals" \
      --goals-sha256 "$(sha "$goals")" --arms "$arms" --configs "$cfg" --workers "$WORKERS" \
      --output "$RUN/S-cem" </dev/null 2>&1 | tee -a "$RUN/cem.log" ;;
  s-cpu-ev)
    goals=${1:?goals}
    "${GPU[@]}" $PY scripts/run_task101.py episodes --cohort S --gated --goals "$goals" \
      --goals-sha256 "$(sha "$goals")" --arms rand,hold,F,H-pose --slots 0:8 --evidence 8 \
      --device none --workers "$WORKERS" --output "$RUN/S-cpu-ev" </dev/null 2>&1 \
      | tee -a "$RUN/cpu-ev.log" ;;
  s-cpu)
    goals=${1:?goals}
    $PY scripts/run_task101.py episodes --cohort S --gated --goals "$goals" \
      --goals-sha256 "$(sha "$goals")" --arms rand,hold,F,H-pose --slots 8:128 --evidence 0 \
      --device none --workers "$WORKERS" --output "$RUN/S-cpu" </dev/null 2>&1 | tee -a "$RUN/cpu.log" ;;
  *) sed -n '2,/^set -euo/p' "$0" | grep '^#' >&2; exit 2 ;;
esac
