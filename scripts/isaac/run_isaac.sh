#!/usr/bin/env bash
# Run one scripts/isaac/*.py in a fresh, self-removing isaaclab_arena container.
# Usage: scripts/isaac/run_isaac.sh <script.py> <out-dir> [extra script args]
# <out-dir> must not exist yet (never overwritten); it is mounted writable at /oej/out and the
# script gets `--output /oej/out/run`. Read-only mounts: scripts/isaac (/oej/scripts/isaac),
# src (/oej/src), configs (/oej/configs), the pinned G1 MJCF directory (/oej/mjcf) and, if
# present, assets/isaac (/oej/usd, converted USDs). Records the image id and code
# revision/status, and samples GPU memory every ~1 s (gpu_apps.csv, gpu_device.csv; sampled
# peaks are lower bounds). GPU coordination is the machine-wide lock (audit F13): run it as
# `scripts/gpu_run.sh [checks] -- scripts/isaac/run_isaac.sh ...` (or inside
# `flock ~/.local/state/gpu/lock ...`). It does not take the lock itself, since a caller's
# `flock` already holds it; started outside gpu_run.sh it prints a reminder. (Before 2026-10-02
# it refused while a TASK-072 process name was running, which missed every gated run under
# another name.)
# Fails fast (exit 1, before any container or sampler starts) when an input is missing: the
# MJCF directory, ISAAC_MODELS_DIR when set, and assets/isaac when an argument names /oej/usd or
# ISAAC_REQUIRE_USD=1 (audit F5; it used to drop the USD mount silently). Whether the USD mount
# was made is recorded in <out-dir>/usd_mount.txt.
# PXR_WORK_THREAD_LIMIT, if set on the host, is passed into the container (OpenUSD's work-pool
# thread limit). Kit overwrites it at start-up, so on its own it has no effect; e9_server_isaac.py
# pins it (default 1 for Newton; docs/ISAAC_E9_REPLAY.md §4).
# ISAAC_MODELS_DIR, if set, is mounted read-only at /models/<its basename>, e.g.
# ~/models/isaaclab_arena -> /models/isaaclab_arena (the GR00T tutorial's /models layout; its
# client config asserts the model path exists but loads no weights). Not /models itself: the
# image's entrypoint chowns /models under `set -e`.
# If your login shell predates your docker group membership, run it via `sg docker -c`.
set -euo pipefail
SCRIPT=${1:?usage: run_isaac.sh <script.py> <out-dir> [args]}
OUT=${2:?usage: run_isaac.sh <script.py> <out-dir> [args]}
shift 2
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
IMAGE=${ISAAC_IMAGE:-isaaclab_arena:latest}
MJCF_DIR=${MJCF_DIR:-$REPO/third_party/unitree_mujoco/unitree_robots/g1}
[ -f "$REPO/scripts/isaac/$SCRIPT" ] || { echo "no scripts/isaac/$SCRIPT" >&2; exit 1; }
[ -e "$OUT" ] && { echo "refusing to overwrite $OUT" >&2; exit 1; }
[ -d "$MJCF_DIR" ] || {
  echo "missing MJCF directory $MJCF_DIR (third_party not linked? run scripts/fetch_assets.py)" >&2
  exit 1
}
if [ -n "${ISAAC_MODELS_DIR:-}" ] && [ ! -d "$ISAAC_MODELS_DIR" ]; then
  echo "ISAAC_MODELS_DIR=$ISAAC_MODELS_DIR is not a directory" >&2
  exit 1
fi
NEED_USD=${ISAAC_REQUIRE_USD:-0}
for arg in "$@"; do [[ "$arg" == */oej/usd* ]] && NEED_USD=1; done
if [ "$NEED_USD" = 1 ] && [ ! -d "$REPO/assets/isaac" ]; then
  echo "missing $REPO/assets/isaac (converted USDs), which this run needs at /oej/usd;" \
    "link it (scripts/new_worktree.sh does) or convert them first" >&2
  exit 1
fi
if [ "${GPU_RUN_LOCKED:-0}" != 1 ]; then
  echo "note: not started via scripts/gpu_run.sh; hold the machine GPU lock" \
    "(flock ~/.local/state/gpu/lock) yourself" >&2
fi
OUT=$(mkdir -p "$OUT" && cd -- "$OUT" && pwd)
mkdir -p "$OUT/home_cache"
docker image inspect "$IMAGE" --format '{{.Id}} {{json .RepoDigests}}' > "$OUT/image.txt"
git -C "$REPO" rev-parse HEAD > "$OUT/code_revision.txt"
git -C "$REPO" status --porcelain > "$OUT/code_status.txt"
EXTRA=""
[ $# -gt 0 ] && EXTRA=$(printf ' %q' "$@")
ENV_PASS=()
[ -n "${PXR_WORK_THREAD_LIMIT+x}" ] && ENV_PASS+=(-e "PXR_WORK_THREAD_LIMIT=$PXR_WORK_THREAD_LIMIT")
echo "${PXR_WORK_THREAD_LIMIT-<unset>}" > "$OUT/pxr_work_thread_limit_host.txt"
USD_MOUNT=()
if [ -d "$REPO/assets/isaac" ]; then
  USD_MOUNT=(-v "$(cd -- "$REPO/assets/isaac" && pwd -P):/oej/usd:ro")
  echo "mounted $(cd -- "$REPO/assets/isaac" && pwd -P)" > "$OUT/usd_mount.txt"
else
  echo "none (no assets/isaac)" > "$OUT/usd_mount.txt"
fi
[ -n "${ISAAC_MODELS_DIR:-}" ] && USD_MOUNT+=(-v "$(cd -- "$ISAAC_MODELS_DIR" && pwd -P):/models/$(basename -- "$ISAAC_MODELS_DIR"):ro")

nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_memory \
  --format=csv,noheader -lms 1000 > "$OUT/gpu_apps.csv" &
SAMPLE_APPS=$!
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -lms 1000 \
  > "$OUT/gpu_device.csv" &
SAMPLE_DEV=$!
trap 'kill "$SAMPLE_APPS" "$SAMPLE_DEV" 2>/dev/null || true' EXIT

docker run --rm --name "oej-isaac-$(basename "$OUT")" \
  --gpus all --runtime=nvidia --ipc=host --net=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e PYTHONPATH=/oej/src \
  -e DOCKER_RUN_USER_ID="$(id -u)" -e DOCKER_RUN_USER_NAME="$(id -un)" \
  -e DOCKER_RUN_GROUP_ID="$(id -g)" -e DOCKER_RUN_GROUP_NAME="$(id -gn)" \
  "${ENV_PASS[@]}" \
  -v "$REPO/scripts/isaac:/oej/scripts/isaac:ro" \
  -v "$REPO/src:/oej/src:ro" \
  -v "$REPO/configs:/oej/configs:ro" \
  -v "$(cd -- "$MJCF_DIR" && pwd -P):/oej/mjcf:ro" \
  "${USD_MOUNT[@]}" \
  -v "$OUT:/oej/out" \
  -v "$OUT/home_cache:/home/$(id -un)/.cache" \
  "$IMAGE" \
  "/isaac-sim/python.sh /oej/scripts/isaac/$SCRIPT --output /oej/out/run$EXTRA 2>&1 | tee /oej/out/log.txt"
