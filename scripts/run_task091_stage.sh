#!/usr/bin/env bash
# TASK-091: run one stage (protocol docs/experiments/grasp_play_corpus_v1_planned_close.md §4-§5),
# each GPU use its own scripts/gpu_run.sh job under the 7 200 s per-job cap (§5.4), then
# summarise. Run from a clean tree at the frozen revision (the Stage 0 merge).
#   scripts/run_task091_stage.sh a <out>                         Stage A″ (roots 906000-906063)
#   scripts/run_task091_stage.sh bd <out>                        Stage BD (development)
#   scripts/run_task091_stage.sh bs <out> <summary-bd.json> <summary-a.json>
#                                                                Stage BS (gated; after GO)
# Every output file is refused if it exists, and the runner's ledger refuses a cohort run twice.
# Before each GPU job the "## Requests" section of ~/.local/state/gpu/BOARD.md is read; while it
# has a "- v2d: ..." line the job waits (up to 15 min) so that the V2D export goes first, and
# every job takes the lock with gpu_run.sh --wait.
set -euo pipefail
STAGE=${1:?stage}; OUT=${2:?output dir}
cd "$(dirname "$0")/.."
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "tracked tree is dirty" >&2; exit 1; }
mkdir -p "$OUT"
PYB=.venv/bin/python
PY=(env MUJOCO_GL=egl "$PYB" scripts/run_task091.py)
pcl() { "$PYB" -c "import sys; sys.path.insert(0, 'src'); from embodied_jepa import grasp_corpus_wm_planned_close as p; $1"; }
CAP=$(pcl "print(p.JOB_CAP_SECONDS)")
PARTS=$(pcl "print(p.BS_PARTS)")
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
# The runner stops itself at the cap (CAP-HIT, exit 4); timeout(1) is a backstop 5 min later.
gpu() {
  local tries=0 code
  while :; do
    board_check
    set +e
    scripts/gpu_run.sh --wait --min-free-gib 8 --board --samples 5 --who "oej:task091-$STAGE" -- \
      timeout --kill-after=60 $((CAP + 300)) "$@" </dev/null
    code=$?
    set -e
    [ "$code" -ne 76 ] && return "$code"
    tries=$((tries + 1)); [ "$tries" -ge 10 ] && return 76
    sleep 60
  done
}
WM_ARMS() {  # arm:seed pairs of a cohort, from the frozen tables
  pcl "t = p.BD_SEEDS if '$1' == 'bd' else p.BS_SEEDS; print(' '.join(f'{a}:{s}' for a in ('M-v2g', 'M-v1p', 'M-v1') for s in t[a]))"
}
case "$STAGE" in
  a)
    gpu "${PY[@]}" roots --cohort a2-roots --output "$OUT" --workers 8
    gpu "${PY[@]}" a-second --cohort a2-roots --roots "$OUT" --output "$OUT"
    "$PYB" scripts/summarize_task091.py --stage a --runs "$OUT" ;;
  bd)
    gpu "${PY[@]}" goals --cohort bd-grasp --output "$OUT" --workers 8
    for a in ik ik-nopress hold random; do
      gpu "${PY[@]}" run --cohort bd-grasp --arm "$a" --goals "$OUT" --output "$OUT" --workers 8
    done
    for pair in $(WM_ARMS bd); do
      gpu "${PY[@]}" run --cohort bd-grasp --arm "${pair%%:*}" --model-seed "${pair##*:}" \
        --goals "$OUT" --output "$OUT" --workers 6
    done
    "$PYB" scripts/summarize_task091.py --stage bd --runs "$OUT" ;;
  bs)
    BD=${3:?Stage BD summary}; AS=${4:?Stage A″ summary}
    gpu "${PY[@]}" goals --cohort bs-grasp --output "$OUT" --workers 8
    for a in ik ik-nopress hold random; do
      gpu "${PY[@]}" run --cohort bs-grasp --arm "$a" --goals "$OUT" --output "$OUT" --workers 8
    done
    for pair in $(WM_ARMS bs); do
      for i in $(seq 1 "$PARTS"); do
        gpu "${PY[@]}" run --cohort bs-grasp --arm "${pair%%:*}" --model-seed "${pair##*:}" \
          --part "$i/$PARTS" --goals "$OUT" --output "$OUT" --workers 6
      done
      "$PYB" scripts/run_task091.py combine --cohort bs-grasp --arm "${pair%%:*}" \
        --model-seed "${pair##*:}" --runs "$OUT"
    done
    gpu "${PY[@]}" run --cohort bs-grasp --arm M-v2g --model-seed 90100 --determinism \
      --goals "$OUT" --output "$OUT" --workers 4
    "$PYB" scripts/summarize_task091.py --stage bs --runs "$OUT" --bd-summary "$BD" --a-summary "$AS" ;;
  *) sed -n '2,12p' "$0" >&2; exit 2 ;;
esac
echo "STAGE-$STAGE-DONE"
