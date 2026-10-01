#!/usr/bin/env bash
# The single entry point for any GPU job on this machine (the 2026-10-02 audit, F13).
#
# Usage: scripts/gpu_run.sh [options] -- <command> [args...]
#
#   --min-free-gib N   refuse unless at least N GiB of GPU memory are free (default 4)
#   --max-util P       refuse when GPU utilisation is above P % (default 30)
#   --max-load L       refuse when the 1-minute CPU load average is above L (default: no check)
#   --samples K        GPU readings, 1 s apart; the worst one counts (default 3)
#   --gpu I            the GPU index (default 0)
#   --wait             wait for the lock instead of failing at once
#   --wait-timeout S   with --wait: give up after S seconds
#   --who NAME         who holds the lock, for the log and the board (default oej:<cwd name>)
#   --log FILE         append start and end records here
#                      (default ~/.local/state/gpu/oej-gpu_run.log)
#   --board            write the BOARD.md "Holder" section while the command runs
#   --request TEXT     with --wait: add "- NAME: TEXT" under the BOARD.md "Requests" section
#                      while waiting; the line is removed once the lock is taken or on exit
#   --grace S          after forwarding a stop signal, SIGKILL the command's process group
#                      when it is still alive S seconds later (default 30)
#   --container P      after the command's group is gone, keep the lock until no running docker
#                      container's name starts with P; after a stop signal, `docker stop` them
#                      (use `--container oej-isaac-` for scripts/isaac/run_isaac.sh)
#
# It takes the machine-wide lock with flock on ~/.local/state/gpu/lock (the convention in
# ~/.local/state/gpu/BOARD.md, shared with other projects' sessions), then, inside the lock,
# checks free VRAM, GPU utilisation and optionally CPU load, logs the GPU's compute apps, and runs
# the command with GPU_RUN_LOCKED=1 in its environment (run_tools.gpu_guard(require_lock=True)
# checks it).
#
# Process group and lock: the command runs in its own process group, without the lock's file
# descriptor, so the lock is held by this script alone and is released when it exits. The script
# exits only after every process in that group is gone (a background grandchild included), and
# only then logs "end" and restores the board. SIGTERM/SIGINT/SIGHUP to this script are sent to
# the whole group as SIGTERM, followed by SIGKILL after --grace seconds. Limits: a descendant
# that leaves the group (setsid, a daemon) is neither waited for nor signalled, and a docker
# container is not a descendant at all. `docker run` without a TTY proxies the TERM to the
# container, but a SIGKILLed docker client leaves its container running on the GPU; --container
# covers that (it waits, and after a stop signal runs `docker stop`). If gpu_run itself is
# SIGKILLed, the lock is released at once while the group may still run, and any board lines it
# wrote stay behind; they carry the tag "[gpu_run pid N]", so a stale line is recognisable.
#
# BOARD.md is never written unless --board or --request is given; every edit holds
# BOARD.md.lock (a separate flock, not the GPU lock) and replaces the file atomically.
#
# Exit status: the command's; 75 when the lock is busy (or the wait timed out); 76 when a GPU or
# load check refused; 2 for a usage error.
# GPU_STATE_DIR (default ~/.local/state/gpu) and NVIDIA_SMI (default nvidia-smi) are overridable,
# for tests.
set -euo pipefail

usage() { sed -n '2,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//' >&2; exit 2; }

STATE=${GPU_STATE_DIR:-$HOME/.local/state/gpu}
SMI=${NVIDIA_SMI:-nvidia-smi}
LOCK=$STATE/lock BOARD=$STATE/BOARD.md
MIN_FREE=4 MAX_UTIL=30 MAX_LOAD="" SAMPLES=3 GPU=0 WAIT=0 WAIT_TIMEOUT="" BOARD_HOLDER=0
REQUEST="" WHO="oej:$(basename -- "$PWD")" LOG="" GRACE=30 CONTAINER=""
DOCKER=${DOCKER:-docker}
while [ $# -gt 0 ]; do
  case "$1" in
    --min-free-gib) MIN_FREE=${2:?}; shift 2 ;;
    --max-util) MAX_UTIL=${2:?}; shift 2 ;;
    --max-load) MAX_LOAD=${2:?}; shift 2 ;;
    --samples) SAMPLES=${2:?}; shift 2 ;;
    --gpu) GPU=${2:?}; shift 2 ;;
    --wait) WAIT=1; shift ;;
    --wait-timeout) WAIT_TIMEOUT=${2:?}; shift 2 ;;
    --who) WHO=${2:?}; shift 2 ;;
    --log) LOG=${2:?}; shift 2 ;;
    --board) BOARD_HOLDER=1; shift ;;
    --request) REQUEST=${2:?}; shift 2 ;;
    --grace) GRACE=${2:?}; shift 2 ;;
    --container) CONTAINER=${2:?}; shift 2 ;;
    --) shift; break ;;
    -h|--help) usage ;;
    *) echo "unknown option $1 (the command goes after --)" >&2; usage ;;
  esac
done
[ $# -gt 0 ] || { echo "no command given" >&2; usage; }
[ -z "$REQUEST" ] || [ "$WAIT" = 1 ] || { echo "--request needs --wait" >&2; exit 2; }
for n in "$MIN_FREE" "$MAX_UTIL" "$SAMPLES" "$GPU" "$GRACE" ${MAX_LOAD:+"$MAX_LOAD"} ${WAIT_TIMEOUT:+"$WAIT_TIMEOUT"}; do
  [[ "$n" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "not a number: $n" >&2; exit 2; }
done
LOG=${LOG:-$STATE/oej-gpu_run.log}
mkdir -p "$STATE" "$(dirname -- "$LOG")"

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '%s %s\n' "$(now)" "$*" >> "$LOG"; }
CMD=$(printf ' %q' "$@")
TAG="[gpu_run pid $$]"

# ----- BOARD.md edits (only with --board / --request) ---------------------------------------------
board_rewrite() {  # board_rewrite <awk program>: atomic rewrite of BOARD.md; values via B_* env
  local prog=$1
  [ -f "$BOARD" ] || { echo "no $BOARD; board not updated" >&2; return 0; }
  {
    flock -w 10 8 || { echo "BOARD.md.lock busy; board not updated" >&2; return 0; }
    local tmp
    tmp=$(mktemp "$BOARD.XXXXXX")
    chmod --reference="$BOARD" -- "$tmp" 2>/dev/null || true
    awk "$prog" "$BOARD" > "$tmp" && mv -f -- "$tmp" "$BOARD" || rm -f -- "$tmp"
  } 8>> "$BOARD.lock"
}
SECTION_SET='
  /^## / { if (insec) { printf "%s", ENVIRON["B_BODY"]; insec = 0 }
           if ($0 == "## " ENVIRON["B_SEC"]) { print; insec = 1; next } }
  insec { next }
  { print }
  END { if (insec) printf "%s", ENVIRON["B_BODY"] }'
REQUEST_ADD='
  { print }
  $0 == "## Requests" { print ENVIRON["B_LINE"]; added = 1 }
  END { if (!added) { print ""; print "## Requests"; print ENVIRON["B_LINE"] } }'
REQUEST_DEL='index($0, ENVIRON["B_TAG"]) == 0 { print }'
REQUEST_ON=0
request_add() {
  B_LINE="- $WHO: $REQUEST $TAG" board_rewrite "$REQUEST_ADD"
  REQUEST_ON=1
  log "request added: $WHO: $REQUEST"
}
request_del() {
  [ "$REQUEST_ON" = 1 ] || return 0
  B_TAG=$TAG board_rewrite "$REQUEST_DEL"
  REQUEST_ON=0
  log "request removed: $WHO"
}
HOLDER_ON=0
holder_set() {
  B_SEC=Holder B_BODY=$1 board_rewrite "$SECTION_SET"
}

CHILD=""
cleanup() {
  request_del
  if [ "$HOLDER_ON" = 1 ]; then
    holder_set "- holder: none (released by $WHO $TAG at $(now))"$'\n\n'
    HOLDER_ON=0
  fi
}
trap cleanup EXIT

# ----- the lock -----------------------------------------------------------------------------------
exec 9>> "$LOCK"
if ! flock -n 9; then
  if [ "$WAIT" != 1 ]; then
    echo "GPU lock $LOCK is busy (see $BOARD); retry later or pass --wait" >&2
    log "busy: $WHO$CMD"
    exit 75
  fi
  [ -z "$REQUEST" ] || request_add
  echo "waiting for the GPU lock $LOCK ..." >&2
  if ! flock ${WAIT_TIMEOUT:+-w "$WAIT_TIMEOUT"} 9; then
    echo "gave up waiting for the GPU lock after ${WAIT_TIMEOUT} s" >&2
    log "wait timed out: $WHO$CMD"
    exit 75
  fi
  request_del
fi

# ----- the checks, inside the lock ----------------------------------------------------------------
worst_free="" worst_util="" total=""
for ((k = 0; k < SAMPLES; k++)); do
  [ "$k" = 0 ] || sleep 1
  line=$("$SMI" -i "$GPU" --query-gpu=memory.free,memory.total,utilization.gpu \
    --format=csv,noheader,nounits) || { echo "nvidia-smi failed" >&2; log "refused: nvidia-smi failed"; exit 76; }
  IFS=', ' read -r free total util <<< "$line"
  if [ -z "$worst_free" ] || [ "$free" -lt "$worst_free" ]; then worst_free=$free; fi
  if [ -z "$worst_util" ] || [ "$util" -gt "$worst_util" ]; then worst_util=$util; fi
done
load=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)
apps=$("$SMI" -i "$GPU" --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader 2>/dev/null | tr '\n' ';' || true)
state="gpu $GPU free ${worst_free} MiB of ${total} MiB, util ${worst_util} %, load ${load}, apps: ${apps:-none}"
refuse=""
awk -v f="$worst_free" -v m="$MIN_FREE" 'BEGIN { exit !(f < m * 1024) }' \
  && refuse="free ${worst_free} MiB < ${MIN_FREE} GiB"
awk -v u="$worst_util" -v m="$MAX_UTIL" 'BEGIN { exit !(u > m) }' \
  && refuse="${refuse:+$refuse; }utilisation ${worst_util} % > ${MAX_UTIL} %"
if [ -n "$MAX_LOAD" ]; then
  awk -v l="$load" -v m="$MAX_LOAD" 'BEGIN { exit !(l > m) }' \
    && refuse="${refuse:+$refuse; }load ${load} > ${MAX_LOAD}"
fi
if [ -n "$refuse" ]; then
  echo "refusing to start: $refuse ($state)" >&2
  log "refused: $WHO$CMD: $refuse; $state"
  exit 76
fi

# ----- run ----------------------------------------------------------------------------------------
REV=$(git rev-parse HEAD 2>/dev/null || echo none)
START=$(date +%s)
log "start: $WHO pid $$ cwd $PWD rev $REV:$CMD; $state"
if [ "$BOARD_HOLDER" = 1 ]; then
  holder_set "- holder: $WHO $TAG"$'\n'"- job:$CMD"$'\n'"- start: $(now)"$'\n'"- log: $LOG"$'\n\n'
  HOLDER_ON=1
fi
group_alive() { kill -0 -- "-$CHILD" 2>/dev/null; }
STOPPED="" KILLER=""
forward() {
  STOPPED=${STOPPED:-$(date +%s)}
  [ -n "$CHILD" ] || return 0
  kill -TERM -- "-$CHILD" 2>/dev/null || true
  if [ -z "$KILLER" ]; then  # SIGKILL the group if it outlives the grace
    (
      trap - TERM INT HUP
      sleep "$GRACE"
      if kill -0 -- "-$CHILD" 2>/dev/null; then
        log "killed: $WHO group $CHILD still alive ${GRACE} s after the stop signal"
        kill -KILL -- "-$CHILD" 2>/dev/null || true
      fi
    ) 9>&- &
    KILLER=$!
  fi
}
trap forward TERM INT HUP
set -m  # the command gets its own process group (its pgid is its pid)
GPU_RUN_LOCKED=1 GPU_RUN_LOCK=$LOCK "$@" 9>&- &
CHILD=$!
set +m
[ -z "$STOPPED" ] || forward  # a signal that arrived before the command started
status=0
while :; do
  wait "$CHILD" && status=0 || status=$?
  kill -0 "$CHILD" 2>/dev/null || break  # a forwarded signal interrupted wait; keep waiting
done
# the leader is gone; wait for the rest of its group (a background grandchild, a sampler, ...)
while group_alive; do sleep 0.2; done
if [ -n "$KILLER" ]; then kill "$KILLER" 2>/dev/null || true; wait "$KILLER" 2>/dev/null || true; fi
if [ -n "$CONTAINER" ]; then
  containers() { "$DOCKER" ps -q --filter "name=^$CONTAINER" 2>/dev/null; }
  if ! "$DOCKER" ps -q > /dev/null 2>&1; then
    log "warning: cannot run '$DOCKER ps'; containers named $CONTAINER* not verified"
    echo "warning: cannot verify that containers named $CONTAINER* have stopped" >&2
  else
    while [ -n "$(containers)" ]; do
      if [ -n "$STOPPED" ]; then
        log "stopping containers: $(containers | tr '\n' ' ')"
        containers | xargs "$DOCKER" stop -t 10 > /dev/null 2>&1 || true
      fi
      sleep 1
    done
  fi
fi
log "end: $WHO pid $$ status $status after $(( $(date +%s) - START )) s"
exit "$status"
