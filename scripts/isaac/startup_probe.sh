#!/usr/bin/env bash
# Start-up probe for the intermittent Newton start-up hang (development, TASK-025).
# Usage (via `sg docker -c` if needed):
#   scripts/isaac/startup_probe.sh <out-prefix> <count> [limit-seconds] [--physics newton|physx]
# Runs e9_server_isaac.py --startup_only <count> times, one container at a time, through
# run_isaac.sh (so the usual guard and records apply). A run that has not logged
# "transport built" within <limit-seconds> (default 480) is recorded as a hang, its last
# stack dump is kept in its log, and only that run's container (oej-isaac-<name>) is stopped.
# The server pins PXR_WORK_THREAD_LIMIT=1 for Newton by default; add --pxr_work_thread_limit 0
# after --physics newton for Kit's 16 threads (the setting of the original hang rate).
# Writes <out-prefix>-summary.tsv: run, outcome, seconds to "transport built", last stage.
set -euo pipefail
PREFIX=${1:?usage: startup_probe.sh <out-prefix> <count> [limit-seconds] [--physics ...]}
COUNT=${2:?usage: startup_probe.sh <out-prefix> <count> [limit-seconds] [--physics ...]}
LIMIT=${3:-480}
shift $(( $# < 3 ? $# : 3 ))
PHYSICS=${*:---physics newton}
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
U=/oej/usd/g1_29dof_with_hand-ref/run/usd/g1_29dof_with_hand/g1_29dof_with_hand.usda
SUMMARY="$PREFIX-summary.tsv"
[ -e "$SUMMARY" ] && { echo "refusing to overwrite $SUMMARY" >&2; exit 1; }
printf 'run\toutcome\tbuilt_s\tlast_stage\n' > "$SUMMARY"
for i in $(seq 1 "$COUNT"); do
  OUT="$PREFIX-$i"
  NAME="oej-isaac-$(basename "$OUT")"
  # shellcheck disable=SC2086
  "$REPO/scripts/isaac/run_isaac.sh" e9_server_isaac.py "$OUT" --usd "$U" \
    --manifest /oej/configs/isaac/g1_dex3_joint_manifest_v1.json \
    --scene /oej/configs/isaac/apple_to_plate_v2_scene_v1.json $PHYSICS --startup_only \
    > "$OUT.stdout" 2>&1 &
  PID=$!
  START=$(date +%s)
  OUTCOME=hang
  while kill -0 "$PID" 2>/dev/null; do
    if grep -q "transport built" "$OUT/log.txt" 2>/dev/null; then OUTCOME=built; fi
    if [ "$OUTCOME" = hang ] && [ $(( $(date +%s) - START )) -ge "$LIMIT" ]; then
      docker stop -t 5 "$NAME" > /dev/null 2>&1 || true
      break
    fi
    sleep 2
  done
  wait "$PID" || true
  grep -q "transport built" "$OUT/log.txt" 2>/dev/null && OUTCOME=built
  BUILT=$(grep -o '+ *[0-9.]*s [0-9:]*Z\] transport built' "$OUT/log.txt" 2>/dev/null \
    | grep -o '[0-9.]*s' | head -1 || true)
  LAST=$(grep -o 'e9-server .*' "$OUT/log.txt" 2>/dev/null | tail -1 || true)
  printf '%s\t%s\t%s\t%s\n' "$(basename "$OUT")" "$OUTCOME" "${BUILT:--}" "${LAST:--}" \
    >> "$SUMMARY"
  rm -rf "$OUT/home_cache"  # the per-run warp/ov caches (~350 MB each); logs are kept
done
cat "$SUMMARY"
