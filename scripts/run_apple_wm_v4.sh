#!/bin/zsh
# Frozen TASK-054 runner: train and evaluate the four world model v4 arms, once each, in
# the preregistered order, then compute the paired contrasts.
#
# It must be launched from a CLEAN COMMITTED checkout: every train and evaluate passes
# --require-clean, which refuses a dirty or unversioned tree. Artifacts are written to
# $ARTIFACTS (absolute), which should be the main checkout's git-ignored checkpoints/ and
# outputs/ so they survive a worktree being removed. train refuses to overwrite existing
# training artifacts and evaluate refuses to overwrite a report, so a second attempt
# cannot silently replace the first.
#
#   scripts/run_apple_wm_v4.sh <repo-root-to-run-from> <artifact-root>
#
# Progress and every arm's gate summary go to $ARTIFACTS/outputs/task054-runner.log.
# Offline only: no simulator reset is opened and no closed loop is run.
set -u
REPO="${1:?repo root to run from}"
ARTIFACTS="${2:?artifact root (holds checkpoints/ and outputs/)}"
MANIFEST="benchmarks/manifests/apple-world-model-v4.json"
LOG="$ARTIFACTS/outputs/task054-runner.log"
CKPT="$ARTIFACTS/checkpoints/task054-wm-v4"
OUT="$ARTIFACTS/outputs/task054-wm-v4"

mkdir -p "$ARTIFACTS/outputs" "$CKPT" "$OUT"
cd "$REPO" || exit 1

log() { print -r -- "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

log "runner start; repo=$REPO revision=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ')"

# name:config, in the preregistered arm order.
ARMS=(
  leworldmodel_baseline:apple_wm_v4_lewm
  leworldmodel_chunk:apple_wm_v4_lewm_chunk
  leworldmodel_tail:apple_wm_v4_lewm_tail
  leworldmodel_step:apple_wm_v4_lewm_step
)

for entry in $ARMS; do
  name="${entry%%:*}"
  cfg="${entry##*:}"
  log "=== arm $name ($cfg): train"
  uv run --no-sync python -m embodied_jepa.world_model_v4 train \
    --config "configs/$cfg.yaml" --protocol-manifest "$MANIFEST" \
    --device mps --workers 8 --require-clean \
    --output "$CKPT/$name.pt" \
    --acknowledge-privileged-training-labels >>"$LOG" 2>&1
  status=$?
  log "=== arm $name: train exited $status"
  if [[ $status -ne 0 ]]; then
    log "=== arm $name: training did not complete; skipping its evaluation"
    continue
  fi
  log "=== arm $name: evaluate"
  uv run --no-sync python -m embodied_jepa.world_model_v4 evaluate \
    --config "configs/$cfg.yaml" --protocol-manifest "$MANIFEST" \
    --device mps --workers 8 --require-clean \
    --checkpoint "$CKPT/$name.pt" \
    --output "$OUT/$name-val-gates.json" \
    --dump-errors "$OUT/$name-errors.npz" \
    --acknowledge-privileged-training-labels >>"$LOG" 2>&1
  log "=== arm $name: evaluate exited $?"
done

log "=== paired contrasts (iid and episode-clustered)"
uv run --no-sync python scripts/bootstrap_wm_v4_contrasts.py \
  --errors "$OUT" --output "$OUT/paired-bootstrap.json" \
  --seed 20540 --resamples 20000 >>"$LOG" 2>&1
log "=== contrasts exited $?"
log "runner done"
