#!/usr/bin/env bash
# TASK-088: run one stage's runs (protocol §5.4, §8), each its own scripts/gpu_run.sh job, then
# summarise. Usage: scripts/run_task088_stage.sh {dev|gated} <output-dir>
# Refuses an existing output directory's run files (run_task088.py refuses to overwrite).
set -euo pipefail
STAGE=${1:?dev or gated}; OUT=${2:?output dir}
cd "$(dirname "$0")/.."
case "$STAGE" in
  dev) R=dev-reach; Gc=dev-grasp; REPL=() ;;
  gated) R=gated-reach; Gc=gated-grasp; REPL=(87101 87102) ;;
  *) echo "stage must be dev or gated" >&2; exit 2 ;;
esac
mkdir -p "$OUT"
PY=(env MUJOCO_GL=egl .venv/bin/python scripts/run_task088.py)
# gpu_run.sh exits 76 when it refuses to start (VRAM / utilisation check, before the command
# runs); only that is retried, up to 10 times a minute apart. Any other failure stops the stage.
gpu() {
  local tries=0 code
  while :; do
    set +e
    scripts/gpu_run.sh --wait --min-free-gib 8 --board --samples 5 --who "oej:task088-$STAGE" -- "$@"
    code=$?
    set -e
    [ "$code" -ne 76 ] && return "$code"
    tries=$((tries + 1)); [ "$tries" -ge 10 ] && return 76
    sleep 60
  done
}
GPU=(gpu)
"${GPU[@]}" "${PY[@]}" lambda --output "$OUT"
L=(--lambda-file "$OUT/lambda.json")
for c in "$R" "$Gc"; do "${GPU[@]}" "${PY[@]}" goals --cohort "$c" --output "$OUT" --workers 8; done
for c in "$R" "$Gc"; do
  for a in ik ik-nopress hold random; do
    "${GPU[@]}" "${PY[@]}" run --cohort "$c" --arm "$a" --goals "$OUT" --output "$OUT" --workers 8
  done
  for a in P G G-lat G-pose; do
    "${GPU[@]}" "${PY[@]}" run --cohort "$c" --arm "$a" --model-seed 87100 "${L[@]}" \
      --goals "$OUT" --output "$OUT" --workers 6
  done
done
for s in "${REPL[@]}"; do
  for a in P G; do
    "${GPU[@]}" "${PY[@]}" run --cohort "$R" --arm "$a" --model-seed "$s" "${L[@]}" \
      --goals "$OUT" --output "$OUT" --workers 6
  done
done
.venv/bin/python scripts/summarize_task088.py --runs "$OUT" --goals "$OUT" --stage "$STAGE"
echo "STAGE-$STAGE-DONE"
