#!/usr/bin/env bash
# TASK-093: run one stage (protocol docs/experiments/grasp_rate_projected_planner_v1.md §6), each
# GPU use its own scripts/gpu_run.sh job under the 7 200 s per-job cap, then summarise. Run from a
# clean tree at the frozen revision (the Stage 0 merge).
#   scripts/run_task093_stage.sh p <out> <summary-0.json>         Stage P (roots 917000-917063)
#   scripts/run_task093_stage.sh bd <out> <summary-p.json>        Stage BD (development)
#   scripts/run_task093_stage.sh bs <out> <summary-bd.json> <summary-p.json>
#                                                                 Stage BS (gated; after GO)
# Every output file is refused if it exists, and the runner's ledger refuses a cohort run twice.
# The runner starts in this checkout's root (where third_party/le-wm is; R27.107) and loads its
# models before any ledger claim. Before each GPU job the "## Requests" section of
# ~/.local/state/gpu/BOARD.md is read; while it has a "- v2d: ..." line the job waits (up to
# 15 min), and every job takes the lock with gpu_run.sh --wait.
set -euo pipefail
STAGE=${1:?stage}; OUT=${2:?output dir}
cd "$(dirname "$0")/.."
[ -d third_party/le-wm ] || { echo "third_party/le-wm is missing in $(pwd)" >&2; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "tracked tree is dirty" >&2; exit 1; }
OUT=$(realpath -m "$OUT")
mkdir -p "$OUT"
PYB=$(pwd)/.venv/bin/python
PY=(env MUJOCO_GL=egl "$PYB" "$(pwd)/scripts/run_task093.py")
rp() { "$PYB" -c "import sys; sys.path.insert(0, 'src'); from embodied_jepa import grasp_rate_projected as r; $1"; }
CAP=$(rp "print(r.JOB_CAP_SECONDS)")
PARTS=$(rp "print(r.BS_PARTS)")
BOARD=${GPU_BOARD:-$HOME/.local/state/gpu/BOARD.md}
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
# gpu_run.sh exits 76 when it refuses to start (VRAM / utilisation check, before the command
# runs); only that is retried, up to 10 times a minute apart. Any other failure stops the stage.
gpu() {
  local tries=0 code
  while :; do
    board_check
    set +e
    scripts/gpu_run.sh --wait --min-free-gib 8 --board --samples 5 --who "oej:task093-$STAGE" -- \
      timeout --kill-after=60 $((CAP + 300)) "$@" </dev/null
    code=$?
    set -e
    [ "$code" -ne 76 ] && return "$code"
    tries=$((tries + 1)); [ "$tries" -ge 10 ] && return 76
    sleep 60
  done
}
# A job that exits VOID (3: a plumbing or equivalence check failed) does not stop the stage
# before its summary: the stage records the VOID, summarises, and then exits 3 itself.
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
WM_ARMS() {  # arm:seed pairs of a cohort, from the frozen tables
  rp "t = r.BD_SEEDS if '$1' == 'bd' else r.BS_SEEDS; print(' '.join(f'{a}:{s}' for a in ('M-proj', 'M-v1') for s in t[a]))"
}
case "$STAGE" in
  p)
    S0=${3:?Stage 0 summary}
    gpu "${PY[@]}" roots --cohort p-roots --output "$OUT" --workers 8
    gpuv "${PY[@]}" plans --cohort p-roots --roots "$OUT" --output "$OUT"
    "$PYB" scripts/summarize_task093.py --stage p --runs "$OUT" --s0-summary "$S0" ;;
  bd)
    PS=${3:?Stage P summary}
    gpu "${PY[@]}" goals --cohort bd-grasp --output "$OUT" --workers 8
    for a in ik ik-nopress hold random; do
      gpu "${PY[@]}" run --cohort bd-grasp --arm "$a" --goals "$OUT" --output "$OUT" --workers 8
    done
    for pair in $(WM_ARMS bd); do
      gpuv "${PY[@]}" run --cohort bd-grasp --arm "${pair%%:*}" --model-seed "${pair##*:}" \
        --goals "$OUT" --output "$OUT" --workers 6
    done
    "$PYB" scripts/summarize_task093.py --stage bd --runs "$OUT" --p-summary "$PS" ;;
  bs)
    BD=${3:?Stage BD summary}; PS=${4:?Stage P summary}
    gpu "${PY[@]}" goals --cohort bs-grasp --output "$OUT" --workers 8
    for a in ik ik-nopress hold random; do
      gpu "${PY[@]}" run --cohort bs-grasp --arm "$a" --goals "$OUT" --output "$OUT" --workers 8
    done
    for pair in $(WM_ARMS bs); do
      for i in $(seq 1 "$PARTS"); do
        gpuv "${PY[@]}" run --cohort bs-grasp --arm "${pair%%:*}" --model-seed "${pair##*:}" \
          --part "$i/$PARTS" --goals "$OUT" --output "$OUT" --workers 6
      done
      "$PYB" scripts/run_task093.py combine --cohort bs-grasp --arm "${pair%%:*}" \
        --model-seed "${pair##*:}" --runs "$OUT"
    done
    gpuv "${PY[@]}" run --cohort bs-grasp --arm M-proj --model-seed 87100 --determinism \
      --goals "$OUT" --output "$OUT" --workers 4
    "$PYB" scripts/summarize_task093.py --stage bs --runs "$OUT" --bd-summary "$BD" \
      --p-summary "$PS" ;;
  *) sed -n '2,14p' "$0" >&2; exit 2 ;;
esac
[ "$VOID" -ne 0 ] && { echo "STAGE-$STAGE-VOID" >&2; exit "$VOID"; }
echo "STAGE-$STAGE-DONE"
