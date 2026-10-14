#!/usr/bin/env bash
# TASK-092: run one stage (protocol docs/experiments/grasp_rate_aware_wm_v1.md §6), each GPU use
# its own scripts/gpu_run.sh job under the 7 200 s per-job cap, then summarise. Run from a clean
# tree at the frozen revision (the Stage 0 merge).
#   scripts/run_task092_stage.sh t <out>                        Stage T (three training jobs)
#   scripts/run_task092_stage.sh r <out>                        Stage R (play-v1 test)
#   scripts/run_task092_stage.sh a <out> <summary-r.json>       Stage A‴ (roots 908000-908063)
#   scripts/run_task092_stage.sh bd <out>                       Stage BD (development)
#   scripts/run_task092_stage.sh bs <out> <summary-bd.json> <summary-r.json> <summary-a.json>
#                                                               Stage BS (gated; after GO)
# Stage T writes the checkpoints to checkpoints/task092-run/G-req-<seed> and its summary to <out>.
# Every output file is refused if it exists, and the runner's ledger refuses a cohort run twice.
# Before each GPU job the "## Requests" section of ~/.local/state/gpu/BOARD.md is read; while it
# has a "- v2d: ..." line the job waits (up to 15 min), and every job takes the lock with
# gpu_run.sh --wait.
set -euo pipefail
STAGE=${1:?stage}; OUT=${2:?output dir}
cd "$(dirname "$0")/.."
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "tracked tree is dirty" >&2; exit 1; }
mkdir -p "$OUT"
PYB=.venv/bin/python
PY=(env MUJOCO_GL=egl "$PYB" scripts/run_task092.py)
ra() { "$PYB" -c "import sys; sys.path.insert(0, 'src'); from embodied_jepa import grasp_rate_aware as r; $1"; }
CAP=$(ra "print(r.JOB_CAP_SECONDS)")
PARTS=$(ra "print(r.BS_PARTS)")
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
    scripts/gpu_run.sh --wait --min-free-gib 8 --board --samples 5 --who "oej:task092-$STAGE" -- \
      timeout --kill-after=60 $((CAP + 300)) "$@" </dev/null
    code=$?
    set -e
    [ "$code" -ne 76 ] && return "$code"
    tries=$((tries + 1)); [ "$tries" -ge 10 ] && return 76
    sleep 60
  done
}
WM_ARMS() {  # arm:seed pairs of a cohort, from the frozen tables
  ra "t = r.BD_SEEDS if '$1' == 'bd' else r.BS_SEEDS; print(' '.join(f'{a}:{s}' for a in ('M-req', 'M-v1') for s in t[a]))"
}
case "$STAGE" in
  t)
    for seed in 87100 87101 87102; do
      out=checkpoints/task092-run/G-req-$seed
      [ -e "$out/report.json" ] && { echo "skip G-req-$seed (done)"; continue; }
      [ -e "$out" ] && { echo "G-req-$seed did not complete: T-VOID, stop for a ruling" >&2; exit 1; }
      gpu "$PYB" scripts/train_task092.py --seed "$seed" --output "$out" 2>&1 | tee -a "$OUT/train-$seed.log"
    done
    "$PYB" scripts/summarize_task092.py --stage t --runs checkpoints/task092-run
    cp checkpoints/task092-run/summary-t.json "$OUT/" ;;
  r)
    gpu "${PY[@]}" stage-r --output "$OUT/stage-r.json"
    "$PYB" scripts/summarize_task092.py --stage r --runs "$OUT" ;;
  a)
    RS=${3:?Stage R summary}
    gpu "${PY[@]}" roots --cohort a3-roots --output "$OUT" --workers 8
    gpu "${PY[@]}" a-third --cohort a3-roots --roots "$OUT" --output "$OUT"
    "$PYB" scripts/summarize_task092.py --stage a --runs "$OUT" --r-summary "$RS" ;;
  bd)
    gpu "${PY[@]}" goals --cohort bd-grasp --output "$OUT" --workers 8
    for a in ik ik-nopress hold random; do
      gpu "${PY[@]}" run --cohort bd-grasp --arm "$a" --goals "$OUT" --output "$OUT" --workers 8
    done
    for pair in $(WM_ARMS bd); do
      gpu "${PY[@]}" run --cohort bd-grasp --arm "${pair%%:*}" --model-seed "${pair##*:}" \
        --goals "$OUT" --output "$OUT" --workers 6
    done
    "$PYB" scripts/summarize_task092.py --stage bd --runs "$OUT" ;;
  bs)
    BD=${3:?Stage BD summary}; RS=${4:?Stage R summary}; AS=${5:?Stage A‴ summary}
    gpu "${PY[@]}" goals --cohort bs-grasp --output "$OUT" --workers 8
    for a in ik ik-nopress hold random; do
      gpu "${PY[@]}" run --cohort bs-grasp --arm "$a" --goals "$OUT" --output "$OUT" --workers 8
    done
    for pair in $(WM_ARMS bs); do
      for i in $(seq 1 "$PARTS"); do
        gpu "${PY[@]}" run --cohort bs-grasp --arm "${pair%%:*}" --model-seed "${pair##*:}" \
          --part "$i/$PARTS" --goals "$OUT" --output "$OUT" --workers 6
      done
      "$PYB" scripts/run_task092.py combine --cohort bs-grasp --arm "${pair%%:*}" \
        --model-seed "${pair##*:}" --runs "$OUT"
    done
    gpu "${PY[@]}" run --cohort bs-grasp --arm M-req --model-seed 87100 --determinism \
      --goals "$OUT" --output "$OUT" --workers 4
    "$PYB" scripts/summarize_task092.py --stage bs --runs "$OUT" --bd-summary "$BD" \
      --r-summary "$RS" --a-summary "$AS" ;;
  *) sed -n '2,14p' "$0" >&2; exit 2 ;;
esac
echo "STAGE-$STAGE-DONE"
