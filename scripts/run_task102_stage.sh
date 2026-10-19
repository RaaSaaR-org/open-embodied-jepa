#!/usr/bin/env bash
# TASK-102's stages (protocol docs/experiments/so101_hybrid_lift_v1.md §7, §8, §13), one GPU lock
# per GPU job (scripts/gpu_run.sh --wait --board); CPU jobs without frames take no lock. Every
# job runs so101_cube_lift_v1 (TASK-101's A1).
#
# Usage: scripts/run_task102_stage.sh <stage> [args]
#  Stage 0 (development; P, A, D and debug seeds; never S). Order (§7.2-§7.5, the runner's ledger
#  enforces the checks before each ledger cohort's first episode):
#   windows                          G_loc's training windows (CPU, no lock)
#   train <i>                        G-i's training (GPU)
#   select                           the primary G_loc -> gloc.json (CPU)
#   goals <cohort> [--limit N]       goal demonstrations (P | A | D | debug; GPU)
#   check <name> [goals-dir] [tag]   §7.3: latent | renc | projector | gloc | determinism
#   p <goals-dir>                    §7.4: S+G at r = 0, 0.04, 0.08, 0.12 on P
#   a <goals-dir> <tau-file>         §7.5: phase 1 alone in the four configurations on A
#   d-gpu <goals-dir> <tau> <cfg> <arms> <tag>   §7.5: D's GPU arms (split into jobs under 1 h)
#   d-cpu-ev <goals-dir>             F, rand, hold on D slots 0-7, frames kept (locked)
#   d-cpu <goals-dir>                F, rand, hold on D slots 8-63 (CPU, no frames)
#
# Outputs go to the shared checkout's outputs/ ($OUT/task102-stage0/...); checkpoints to
# checkpoints/task102-stage0/G-<i>/. Every job refuses an existing output. The check ledger is
# $OUT/task102-stage0/ledger.jsonl. MUJOCO_GL=egl for every rendering job.
set -euo pipefail
REPO=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cd "$REPO"
SHARED=$(dirname -- "$(cd -- "$(git rev-parse --git-common-dir)" && pwd -P)")
OUT=${TASK102_OUT:-$SHARED/outputs}
CKPT=${TASK102_CKPT:-$SHARED/checkpoints}
PY=.venv/bin/python
WORKERS=${TASK102_WORKERS:-6}
GPU=(scripts/gpu_run.sh --wait --min-free-gib 6 --board --who oej:task102-s0 --)
S0=$OUT/task102-stage0
WIN=$S0/gloc-windows
GLOC=$S0/gloc.json
export MUJOCO_GL=egl

sha() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))['goals_sha256'])" "$1/goals_report.json"; }

stage=${1:-}; shift || true
case "$stage" in
  windows)
    mkdir -p "$S0"
    $PY scripts/train_task102.py windows --output "$WIN" </dev/null 2>&1 | tee -a "$S0/windows.log" ;;
  train)
    i=${1:?seed}
    mkdir -p "$CKPT/task102-stage0"
    "${GPU[@]}" $PY scripts/train_task102.py train --seed "$i" --windows "$WIN" \
      --output "$CKPT/task102-stage0/G-$i" </dev/null 2>&1 | tee -a "$S0/train-G-$i.log" ;;
  select)
    $PY scripts/train_task102.py select --runs "$CKPT"/task102-stage0/G-{0,1,2} --output "$GLOC" \
      </dev/null 2>&1 | tee -a "$S0/select.log" ;;
  goals)
    cohort=${1:?cohort}; shift
    mkdir -p "$S0"
    "${GPU[@]}" $PY scripts/run_task102.py goals --cohort "$cohort" "$@" \
      --output "$S0/$cohort-goals" </dev/null 2>&1 | tee -a "$S0/goals-$cohort.log" ;;
  check)
    name=${1:?check}; goals=${2:-}; tag=${3:-$name}
    extra=()
    if [ -n "$goals" ]; then extra=(--goals "$goals" --goals-sha256 "$(sha "$goals")"); fi
    case "$name" in
      projector)  # CPU only, no frames: no lock
        $PY scripts/run_task102.py check "$name" "${extra[@]}" --device cpu \
          --output "$S0/check-$tag" </dev/null 2>&1 | tee -a "$S0/check-$tag.log" ;;
      *)
        "${GPU[@]}" $PY scripts/run_task102.py check "$name" "${extra[@]}" --gloc "$GLOC" \
          --windows "$WIN" --workers "$WORKERS" --output "$S0/check-$tag" </dev/null 2>&1 \
          | tee -a "$S0/check-$tag.log" ;;
    esac ;;
  p)
    goals=${1:?goals}
    "${GPU[@]}" $PY scripts/run_task102.py episodes --cohort P --goals "$goals" \
      --goals-sha256 "$(sha "$goals")" --arms S+G --levels 0,0.04,0.08,0.12 --gloc "$GLOC" \
      --workers "$WORKERS" --output "$S0/P-episodes" </dev/null 2>&1 | tee -a "$S0/P.log" ;;
  a)
    goals=${1:?goals}; tau=${2:?tau-file}
    "${GPU[@]}" $PY scripts/run_task102.py episodes --cohort A --goals "$goals" \
      --goals-sha256 "$(sha "$goals")" --arms H-approach --configs H8-last,H8-mean,H16-last,H16-mean \
      --tau-file "$tau" --gloc "$GLOC" --workers "$WORKERS" --output "$S0/A-episodes" \
      </dev/null 2>&1 | tee -a "$S0/A.log" ;;
  d-gpu)
    goals=${1:?goals}; tau=${2:?tau-file}; cfg=${3:?config-file}; arms=${4:?arms}; tag=${5:?tag}
    "${GPU[@]}" $PY scripts/run_task102.py episodes --cohort D --goals "$goals" \
      --goals-sha256 "$(sha "$goals")" --arms "$arms" --tau-file "$tau" --config-file "$cfg" \
      --gloc "$GLOC" --workers "$WORKERS" --output "$S0/D-$tag" </dev/null 2>&1 \
      | tee -a "$S0/D-$tag.log" ;;
  d-cpu-ev)
    goals=${1:?goals}; tau=${2:?tau-file}; cfg=${3:?config-file}
    "${GPU[@]}" $PY scripts/run_task102.py episodes --cohort D --goals "$goals" \
      --goals-sha256 "$(sha "$goals")" --arms F,rand,hold --slots 0:8 --evidence 8 \
      --tau-file "$tau" --config-file "$cfg" --device none --workers "$WORKERS" \
      --output "$S0/D-cpu-ev" </dev/null 2>&1 | tee -a "$S0/D-cpu-ev.log" ;;
  d-cpu)
    goals=${1:?goals}; tau=${2:?tau-file}; cfg=${3:?config-file}
    $PY scripts/run_task102.py episodes --cohort D --goals "$goals" --goals-sha256 "$(sha "$goals")" \
      --arms F,rand,hold --slots 8:64 --evidence 0 --tau-file "$tau" --config-file "$cfg" \
      --device none --workers "$WORKERS" --output "$S0/D-cpu" </dev/null 2>&1 | tee -a "$S0/D-cpu.log" ;;
  *) sed -n '2,/^set -euo/p' "$0" | grep '^#' >&2; exit 2 ;;
esac
