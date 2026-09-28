#!/usr/bin/env bash
# Run the Isaac Sim G1+Dex3 bring-up spike in a fresh, self-removing container.
# Usage: scripts/isaac/run_bringup.sh <run-name> [extra bringup_g1_dex3.py args]
# Writes to ${OUT_ROOT:-outputs}/<run-name>/ (git-ignored); refuses to overwrite a run.
# If your login shell predates your docker group membership, run it via `sg docker -c`.
set -euo pipefail
NAME=${1:?usage: run_bringup.sh <run-name> [args]}
shift
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
OUT="${OUT_ROOT:-$REPO/outputs}/$NAME"
IMAGE=${ISAAC_IMAGE:-isaaclab_arena:latest}
[ -e "$OUT" ] && { echo "refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT/home_cache"
docker image inspect "$IMAGE" --format '{{.Id}} {{json .RepoDigests}}' > "$OUT/image.txt"
exec docker run --rm --name "oej-isaac-$NAME" \
  --gpus all --runtime=nvidia --ipc=host --net=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e DOCKER_RUN_USER_ID="$(id -u)" -e DOCKER_RUN_USER_NAME="$(id -un)" \
  -e DOCKER_RUN_GROUP_ID="$(id -g)" -e DOCKER_RUN_GROUP_NAME="$(id -gn)" \
  -v "$REPO/scripts/isaac:/oej/scripts/isaac:ro" \
  -v "$OUT:/oej/out" \
  -v "$OUT/home_cache:/home/$(id -un)/.cache" \
  "$IMAGE" \
  "/isaac-sim/python.sh /oej/scripts/isaac/bringup_g1_dex3.py --output /oej/out/run $* 2>&1 | tee /oej/out/log.txt"
