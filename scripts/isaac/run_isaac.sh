#!/usr/bin/env bash
# Run one scripts/isaac/*.py in a fresh, self-removing isaaclab_arena container.
# Usage: scripts/isaac/run_isaac.sh <script.py> <out-dir> [extra script args]
# <out-dir> must not exist yet (never overwritten); it is mounted writable at /oej/out and the
# script gets `--output /oej/out/run`. Read-only mounts: scripts/isaac (/oej/scripts/isaac),
# src (/oej/src), configs (/oej/configs), the pinned G1 MJCF directory (/oej/mjcf) and, if
# present, assets/isaac (/oej/usd, converted USDs). Records the image id and code
# revision/status, and samples GPU memory every ~1 s (gpu_apps.csv, gpu_device.csv; sampled
# peaks are lower bounds). Refuses to start while any non-shell TASK-072 / first_policy
# process (a run, its pytest, ...) is alive, so a development run does not overlap a gated
# run on the shared GPU. Limits: it matches those names only (a gated task under another name
# is not caught) and checks once at start (a gated run started while the container is up is
# not detected). It is a courtesy check, not a lock.
# PXR_WORK_THREAD_LIMIT, if set on the host, is passed into the container (OpenUSD's work-pool
# thread limit; e9_server_isaac.py otherwise sets it to 1 for Newton, docs/ISAAC_E9_REPLAY.md §4).
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
if ps -eo comm=,args= | awk '$1 !~ /^(bash|sh|sg|sleep|tail|grep|awk|ps|nohup)$/ && /first_policy|task072|TASK-072|task-072/' | grep -q .; then
  echo "a TASK-072/first_policy process is running; not starting Isaac" >&2
  exit 3
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
[ -d "$REPO/assets/isaac" ] && USD_MOUNT=(-v "$(cd -- "$REPO/assets/isaac" && pwd -P):/oej/usd:ro")

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
