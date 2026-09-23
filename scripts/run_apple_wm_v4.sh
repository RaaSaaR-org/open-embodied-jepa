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
# Both must be absolute: the script creates the artifact directories before it cds into
# the repo, so a relative artifact root would split the log from the artifacts.
case "$REPO" in /*) ;; *) print -u2 "repo root must be absolute"; exit 1 ;; esac
case "$ARTIFACTS" in /*) ;; *) print -u2 "artifact root must be absolute"; exit 1 ;; esac
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

  # The protocol declares the checkpoint to gate under every training status, so the
  # runner must not equate "exit code 0" with "evaluable". world_model_v4 returns 2 for
  # any status other than `completed`, including `selection_failed` (where the protocol
  # says to gate `.latest.pt`) and `time_budget` (where a selected checkpoint may exist
  # and is gated exactly as a completed arm's would be). Only `failed`, or a missing run
  # report, skips the arm.
  report="$CKPT/$name.run.json"
  if [[ ! -f "$report" ]]; then
    log "=== arm $name: no run report at $report; skipping its evaluation"
    continue
  fi
  train_status=$(uv run --no-sync python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' \
    "$report" 2>>"$LOG")
  has_best=$(uv run --no-sync python -c \
    'import json,sys; print("yes" if json.load(open(sys.argv[1])).get("best_checkpoint_exists") else "no")' \
    "$report" 2>>"$LOG")
  log "=== arm $name: run status=$train_status best_checkpoint_exists=$has_best"
  if [[ "$train_status" == "failed" || "$train_status" == "missing" || "$train_status" == "running" ]]; then
    log "=== arm $name: training did not produce a gateable checkpoint; skipping"
    continue
  fi
  if [[ "$has_best" == "yes" ]]; then
    checkpoint="$CKPT/$name.pt"
  else
    checkpoint="$CKPT/$name.latest.pt"
    log "=== arm $name: no eligible checkpoint was selected; gating .latest.pt as declared"
  fi

  log "=== arm $name: evaluate $checkpoint"
  uv run --no-sync python -m embodied_jepa.world_model_v4 evaluate \
    --config "configs/$cfg.yaml" --protocol-manifest "$MANIFEST" \
    --device mps --workers 8 --require-clean \
    --checkpoint "$checkpoint" \
    --output "$OUT/$name-val-gates.json" \
    --dump-errors "$OUT/$name-errors.npz" \
    --acknowledge-privileged-training-labels >>"$LOG" 2>&1
  status=$?
  log "=== arm $name: evaluate exited $status"
done

# The contrasts need every arm's dump; say so plainly rather than dying inside the script.
missing=0
for entry in $ARMS; do
  name="${entry%%:*}"
  [[ -f "$OUT/$name-errors.npz" ]] || { log "=== missing $OUT/$name-errors.npz"; missing=1; }
done
if [[ $missing -eq 1 ]]; then
  log "=== paired contrasts SKIPPED: at least one arm has no error dump; rerun by hand"
else
  log "=== paired contrasts (iid and episode-clustered)"
  uv run --no-sync python scripts/bootstrap_wm_v4_contrasts.py \
    --errors "$OUT" --output "$OUT/paired-bootstrap.json" \
    --seed 20540 --resamples 20000 >>"$LOG" 2>&1
  status=$?
  log "=== contrasts exited $status"
fi
log "runner done"
