#!/usr/bin/env bash
# TASK-081's controller in closed loop in Isaac Sim (development only, not evidence).
#
# Usage (from the repo root, under the GPU lock):
#   scripts/gpu_run.sh --wait --min-free-gib 8 --board --container oej-isaac- -- \
#     scripts/isaac/run_lewm_cp_closedloop.sh <server-out> <client-out> <physx|newton> \
#     -- <scripts/isaac_lewm_cp_closedloop.py args except --output/--socket>
#
# Starts scripts/isaac/lewm_cp_server_isaac.py in its own container (scripts/isaac/run_isaac.sh,
# container name oej-isaac-<basename of server-out>) and runs the host client against its socket.
# If the client fails, only this run's own container is stopped. Neither directory may exist.
set -euo pipefail
SRV=${1:?server out-dir}
CLI=${2:?client out-dir}
PHYS=${3:?physx or newton}
shift 3
[ "${1:-}" = "--" ] && shift
case "$PHYS" in physx|newton) ;; *) echo "physics must be physx or newton" >&2; exit 2 ;; esac
[ -e "$SRV" ] && { echo "refusing to overwrite $SRV" >&2; exit 1; }
[ -e "$CLI" ] && { echo "refusing to overwrite $CLI" >&2; exit 1; }
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
U=/oej/usd/g1_29dof_with_hand-ref/run/usd/g1_29dof_with_hand/g1_29dof_with_hand.usda
NAME="oej-isaac-$(basename "$SRV")"
sg docker -c "scripts/isaac/run_isaac.sh lewm_cp_server_isaac.py $(printf '%q' "$SRV") \
  --usd $U --manifest /oej/configs/isaac/g1_dex3_joint_manifest_v1.json \
  --scene /oej/configs/isaac/apple_to_plate_v2_scene_v1.json \
  --camera /oej/configs/isaac/onboard_camera_v1.json --physics $PHYS" \
  > "$SRV.server.log" 2>&1 &
SERVER=$!
RC=0
MUJOCO_GL=egl uv run --no-sync python scripts/isaac_lewm_cp_closedloop.py --output "$CLI" \
  --socket "$SRV/run/cp.sock" --physics-label "$PHYS" "$@" || RC=$?
if [ "$RC" != 0 ]; then
  echo "client exited $RC; stopping $NAME" >&2
  sg docker -c "docker stop $NAME" >/dev/null 2>&1 || true
fi
wait "$SERVER" || true
exit "$RC"
