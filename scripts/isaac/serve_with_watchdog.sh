#!/usr/bin/env bash
# Start the e9 Isaac server with a start-up watchdog and a bounded retry (development, TASK-025).
# Usage (via `sg docker -c` if needed):
#   scripts/isaac/serve_with_watchdog.sh <out-prefix> <limit-seconds> <tries> [server args...]
# e.g. scripts/isaac/serve_with_watchdog.sh outputs/isaac-e9-server-4 300 3 --usd "$U" \
#        --manifest /oej/configs/isaac/g1_dex3_joint_manifest_v1.json \
#        --scene /oej/configs/isaac/apple_to_plate_v2_scene_v1.json --physics newton
#
# Why: Newton start-up hangs intermittently inside UsdPhysics.LoadUsdPhysicsFromRange (about 1 in
# 5 starts; docs/ISAAC_E9_REPLAY.md §4). The hang happens before any episode, so a restart is
# safe. Try k runs `run_isaac.sh e9_server_isaac.py <out-prefix>-t<k> [server args...]` (so the
# usual guard and records apply). If its log has no "transport built" within <limit-seconds>
# (healthy starts: 124.9-151.6 s; 300 s is about twice the slowest), the watchdog stops only
# that try's own container (oej-isaac-<basename of the try dir>), then the launcher process it
# started itself, and starts try k+1, up to <tries> tries. A try that exits before building is
# not retried (that is a different failure). Every try, hang and retry is logged to
# <out-prefix>-watchdog.log. When a try has built, <out-prefix>-ready holds its socket path
# (<try dir>/run/e9.sock, for `e9_replay.py isaac --socket`) and the watchdog waits for that
# server to exit and returns its status. Exit 4: every try hung.
# Test hooks (tests/test_isaac_watchdog.py): ISAAC_WATCHDOG_RUN (launcher), ISAAC_WATCHDOG_STOP
# (container stop command), ISAAC_WATCHDOG_POLL and ISAAC_WATCHDOG_GRACE (seconds).
set -euo pipefail
PREFIX=${1:?usage: serve_with_watchdog.sh <out-prefix> <limit-seconds> <tries> [server args]}
LIMIT=${2:?usage: serve_with_watchdog.sh <out-prefix> <limit-seconds> <tries> [server args]}
TRIES=${3:?usage: serve_with_watchdog.sh <out-prefix> <limit-seconds> <tries> [server args]}
shift 3
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
RUN=${ISAAC_WATCHDOG_RUN:-$REPO/scripts/isaac/run_isaac.sh}
STOP=${ISAAC_WATCHDOG_STOP:-docker stop -t 5}
POLL=${ISAAC_WATCHDOG_POLL:-2}
GRACE=${ISAAC_WATCHDOG_GRACE:-30}
LOG="$PREFIX-watchdog.log"
READY="$PREFIX-ready"
for f in "$LOG" "$READY"; do
  [ -e "$f" ] && { echo "refusing to overwrite $f" >&2; exit 1; }
done
mkdir -p "$(dirname -- "$PREFIX")"
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG" >&2; }
built() { grep -q "transport built" "$1/log.txt" 2>/dev/null; }

log "watchdog: prefix=$PREFIX limit=${LIMIT}s tries=$TRIES"
for k in $(seq 1 "$TRIES"); do
  OUT="$PREFIX-t$k"
  NAME="oej-isaac-$(basename -- "$OUT")"
  log "try $k/$TRIES: starting $OUT (container $NAME)"
  "$RUN" e9_server_isaac.py "$OUT" "$@" > "$OUT.stdout" 2>&1 &
  PID=$!
  START=$(date +%s)
  while kill -0 "$PID" 2>/dev/null && ! built "$OUT"; do
    if [ $(( $(date +%s) - START )) -ge "$LIMIT" ]; then
      break
    fi
    sleep "$POLL"
  done
  if built "$OUT"; then
    log "try $k/$TRIES: transport built after $(( $(date +%s) - START )) s; serving"
    printf '%s\n' "$OUT/run/e9.sock" > "$READY"
    STATUS=0
    wait "$PID" || STATUS=$?
    log "try $k/$TRIES: server exited with status $STATUS"
    exit "$STATUS"
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    STATUS=0
    wait "$PID" || STATUS=$?
    log "try $k/$TRIES: exited with status $STATUS before building; not retried (see $OUT.stdout)"
    [ "$STATUS" -eq 0 ] && STATUS=5
    exit "$STATUS"
  fi
  log "try $k/$TRIES: HANG: no 'transport built' after ${LIMIT} s; stopping own container $NAME"
  # shellcheck disable=SC2086
  $STOP "$NAME" > /dev/null 2>&1 || true
  WAITED=0
  while kill -0 "$PID" 2>/dev/null && [ "$WAITED" -lt "$GRACE" ]; do
    sleep 1
    WAITED=$(( WAITED + 1 ))
  done
  if kill -0 "$PID" 2>/dev/null; then
    log "try $k/$TRIES: launcher $PID still running after ${GRACE} s; killing it (own child)"
    kill "$PID" 2>/dev/null || true
  fi
  wait "$PID" 2>/dev/null || true
  rm -rf "$OUT/home_cache"  # the per-run warp/ov caches (~350 MB); logs and stack dumps are kept
  if [ "$k" -lt "$TRIES" ]; then
    log "try $k/$TRIES: retrying (the hang is before any episode)"
  fi
done
log "every try hung ($TRIES of $TRIES); giving up"
exit 4
