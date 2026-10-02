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
#                      (use `--container oej-isaac-` for scripts/isaac/run_isaac.sh). P must
#                      match ^[A-Za-z0-9][A-Za-z0-9_-]*$ and is compared as a literal prefix
#
# The command runs in a background process group, so it must not read from the terminal (a read
# stops it with SIGTTIN); give it its input from a file or </dev/null.
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
# the whole group as SIGTERM, followed by SIGKILL after --grace seconds. The helper that sends
# that SIGKILL runs in a process group of its own, so a signal to this script's own group (a
# terminal's Ctrl-C, a tool timeout that TERMs the caller's group) cannot kill it while the
# command, which may ignore TERM, lives on; it is started again if it has died anyway. A watchdog,
# also in a group of its own, notices when this script dies without cleaning up (SIGKILL) while
# the command's group still runs, and then sends that group SIGTERM and, --grace seconds later,
# SIGKILL; it is stopped when this script exits normally. Limits: a descendant
# that leaves the group (setsid, a daemon) is neither waited for nor signalled, and a docker
# container is not a descendant at all. `docker run` without a TTY proxies the TERM to the
# container, but a SIGKILLed docker client leaves its container running on the GPU; --container
# covers that (it waits, and after a stop signal runs `docker stop`). If gpu_run itself is
# SIGKILLed, the lock is released at once while the group may still run for up to about
# --grace + 1 seconds (until the watchdog has killed it), and any board lines it wrote stay
# behind; they carry the tag "[gpu_run pid N]", so a stale line is recognisable. A SIGKILL to the
# watchdog's group as well as this script's (or to every process of the user) leaves the
# command's group running: nothing outside it is left to stop it.
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
if [ -n "$CONTAINER" ] && ! [[ "$CONTAINER" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "--container needs a plain name prefix ([A-Za-z0-9][A-Za-z0-9_-]*): $CONTAINER" >&2
  exit 2
fi
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

CHILD="" KILLER="" WATCHDOG=""
stop_helper() {  # stop_helper <pid>: end a helper started in its own group, and its sleep
  [ -n "$1" ] || return 0
  # SIGKILL, not TERM: a helper can receive a signal before it has reset the TERM trap it
  # inherits from this script, and would then run forward() and start a helper of its own
  if ! kill -KILL -- "-$1" 2>/dev/null; then
    pkill -KILL -P "$1" 2>/dev/null || true
    kill -KILL "$1" 2>/dev/null || true
  fi
  wait "$1" 2>/dev/null || true
}
cleanup() {
  # an early exit while the command's group runs leaves the watchdog to stop that group
  if [ -z "$CHILD" ] || ! kill -0 -- "-$CHILD" 2>/dev/null; then
    stop_helper "$WATCHDOG"
    WATCHDOG=""
  fi
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
in_own_group() {  # in_own_group <var> <command...>: start it in a new process group; pid to <var>
  local var=$1 monitor=$-
  shift
  set -m
  "$@" 9>&- < /dev/null > /dev/null 2>&1 &
  printf -v "$var" %s "$!"
  [[ $monitor == *m* ]] || set +m  # a trap can run while the command is being started
}
grace_kill() {  # SIGKILL the command's group if it outlives the grace
  trap - TERM INT HUP
  sleep "$GRACE"
  if kill -0 -- "-$CHILD" 2>/dev/null; then
    log "killed: $WHO group $CHILD still alive ${GRACE} s after the stop signal"
    kill -KILL -- "-$CHILD" 2>/dev/null || true
  fi
}
parent_alive() {  # is this script (pid $1) still running? A zombie counts as gone
  local stat
  stat=$(ps -o stat= -p "$1" 2>/dev/null) && [[ $stat != Z* ]]
}
watchdog() {  # this script died without cleaning up: stop the command's group itself
  trap - TERM INT HUP
  local me=$1
  while parent_alive "$me"; do
    kill -0 -- "-$CHILD" 2>/dev/null || exit 0
    sleep 1
  done
  kill -0 -- "-$CHILD" 2>/dev/null || exit 0
  log "orphaned: gpu_run pid $me is gone while $WHO group $CHILD runs; sending SIGTERM"
  kill -TERM -- "-$CHILD" 2>/dev/null || true
  grace_kill
}
STOPPED=""
forward() {
  STOPPED=${STOPPED:-$(date +%s)}
  [ -n "$CHILD" ] || return 0
  kill -TERM -- "-$CHILD" 2>/dev/null || true
  # a killer of its own group survives a signal to ours; start another if it died all the same
  if [ -z "$KILLER" ] || ! kill -0 "$KILLER" 2>/dev/null; then
    in_own_group KILLER grace_kill
  fi
}
trap forward TERM INT HUP
set -m  # the command gets its own process group (its pgid is its pid)
GPU_RUN_LOCKED=1 GPU_RUN_LOCK=$LOCK "$@" 9>&- &
CHILD=$!
set +m
in_own_group WATCHDOG watchdog "$$"
[ -z "$STOPPED" ] || forward  # a signal that arrived before the command started
status=0
while :; do
  wait "$CHILD" && status=0 || status=$?
  kill -0 "$CHILD" 2>/dev/null || break  # a forwarded signal interrupted wait; keep waiting
done
# A signal can interrupt the last wait just as the command exits; that wait then returns
# 128 + the signal's number, not the command's status. Ask once more (bash keeps the status of
# a reaped background job; 127 means there is nothing more to learn).
final=0
wait "$CHILD" 2>/dev/null || final=$?
[ "$final" = 127 ] || status=$final
# the leader is gone; wait for the rest of its group (a background grandchild, a sampler, ...)
while group_alive; do sleep 0.2; done
stop_helper "$KILLER"
KILLER=""
stop_helper "$WATCHDOG"
WATCHDOG=""
if [ -n "$CONTAINER" ]; then
  containers() {  # ids of running containers whose name starts with $CONTAINER, compared literally
    local id names name
    "$DOCKER" ps --format '{{.ID}} {{.Names}}' 2>/dev/null | while read -r id names; do
      IFS=, read -r -a name <<< "$names"
      for n in "${name[@]}"; do
        if [[ "$n" == "$CONTAINER"* ]]; then echo "$id"; break; fi
      done
    done
  }
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
