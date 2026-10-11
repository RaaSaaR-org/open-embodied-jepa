#!/usr/bin/env bash
# TASK-089: run one stage (protocol §5.4, §5.5, §7.2), each GPU use its own scripts/gpu_run.sh job,
# then summarise.
#   scripts/run_task089_stage.sh k0 <out>                      A-K0: roots and the ceiling
#   scripts/run_task089_stage.sh a  <out>                      Stage A: roots, A1, A2
#   scripts/run_task089_stage.sh bd <out> <summary-a.json>     Stage BD (development)
#   scripts/run_task089_stage.sh bs <out> <summary-a.json> <summary-bd.json>   Stage BS (gated)
# Every output file is refused if it exists (run_task089.py and summarize_task089.py).
set -euo pipefail
STAGE=${1:?k0, a, bd or bs}; OUT=${2:?output dir}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
PY=(env MUJOCO_GL=egl .venv/bin/python scripts/run_task089.py)
# gpu_run.sh exits 76 when it refuses to start (VRAM / utilisation check, before the command
# runs); only that is retried, up to 10 times a minute apart. Any other failure stops the stage.
gpu() {
  local tries=0 code
  while :; do
    set +e
    scripts/gpu_run.sh --wait --min-free-gib 8 --board --samples 5 --who "oej:task089-$STAGE" -- "$@" </dev/null
    code=$?
    set -e
    [ "$code" -ne 76 ] && return "$code"
    tries=$((tries + 1)); [ "$tries" -ge 10 ] && return 76
    sleep 60
  done
}
case "$STAGE" in
  k0)
    gpu "${PY[@]}" roots --cohort k0-roots --output "$OUT" --workers 8
    gpu "${PY[@]}" ceiling --cohort k0-roots --roots "$OUT" --output "$OUT"
    ;;
  a)
    gpu "${PY[@]}" lambda --output "$OUT"
    gpu "${PY[@]}" roots --cohort a-roots --output "$OUT" --workers 8
    gpu "${PY[@]}" a1 --cohort a-roots --roots "$OUT" --output "$OUT"
    gpu "${PY[@]}" a2 --cohort a-roots --roots "$OUT" --output "$OUT"
    ;;
  bd|bs)
    CHANGE=${3:?Stage A summary}
    C=bd-grasp; [ "$STAGE" = bs ] && C=bs-grasp
    gpu "${PY[@]}" goals --cohort "$C" --output "$OUT" --workers 8
    for a in ik ik-nopress hold random; do
      gpu "${PY[@]}" run --cohort "$C" --arm "$a" --change-file "$CHANGE" --goals "$OUT" \
        --output "$OUT" --workers 8
    done
    for a in G-chg G-base P-chg; do
      gpu "${PY[@]}" run --cohort "$C" --arm "$a" --model-seed 87100 --change-file "$CHANGE" \
        --goals "$OUT" --output "$OUT" --workers 6
    done
    if [ "$STAGE" = bs ]; then
      for s in 87101 87102; do
        gpu "${PY[@]}" run --cohort "$C" --arm G-chg --model-seed "$s" --change-file "$CHANGE" \
          --goals "$OUT" --output "$OUT" --workers 6
      done
    fi
    ;;
  *) echo "stage must be k0, a, bd or bs" >&2; exit 2 ;;
esac
EXTRA=()
[ "$STAGE" = bs ] && EXTRA=(--bd-summary "${4:?Stage BD summary}")
.venv/bin/python scripts/summarize_task089.py --stage "$STAGE" --runs "$OUT" "${EXTRA[@]}"
echo "STAGE-$STAGE-DONE"
