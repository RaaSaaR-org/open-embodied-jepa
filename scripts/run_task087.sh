#!/usr/bin/env bash
# TASK-087's gated stages (protocol docs/experiments/grounded_lewm_v1.md §12), one GPU lock per job.
#
# Usage: scripts/run_task087.sh <stage> [args]
#   features                      Stage F: featurise train and val (one job)
#   probe                         Stage B: the budget probe (P, seed 87900, 60 000 updates)
#   train                         Stage T: the 18 jobs, U and the cap from the probe's report
#   val                           Stage V: the val report
#   features-test <val-sha256>    Stage F': featurise test, after the val report
#   test <val-sha256>             Stage E: the gated evaluation
#
# Paths: features outputs/task087-features, models checkpoints/task087-run, reports and logs
# under outputs/task087-run. Every job refuses an existing output, so a finished job is skipped
# by the train loop (its report exists) and a failed one must be re-run by hand as -r2.
set -euo pipefail
cd "$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

PY=.venv/bin/python
FEAT=outputs/task087-features
MODELS=checkpoints/task087-run
RUN=outputs/task087-run
GPU=(scripts/gpu_run.sh --wait --min-free-gib 8 --board --)
mkdir -p "$RUN"

[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "tracked tree is dirty" >&2; exit 1; }

case "${1:-}" in
  features)
    "${GPU[@]}" $PY scripts/featurise_task087.py --stage trainval --output "$FEAT" --workers 8 \
      </dev/null 2>&1 | tee -a "$RUN/features.log" ;;
  probe)
    "${GPU[@]}" $PY scripts/train_task087.py --features "$FEAT" --arm P --seed 87900 \
      --updates 60000 --select-every 2000 --cap-seconds 14400 --output "$RUN/probe-P-87900" \
      --debug --probe </dev/null 2>&1 | tee -a "$RUN/probe.log" ;;
  train)
    read -r U EVERY CAP ESC < <($PY -c "import json;b=json.load(open('$RUN/probe-P-87900/report.json'))['budget'];print(b['updates'],b['select_every'],int(b['cap_seconds'])+1,int(b['escalate']))")
    [ "$ESC" = 0 ] || { echo "the budget escalates (R24.6); stop for a ruling" >&2; exit 1; }
    for seed in 87100 87101 87102; do
      for arm in P S G C I N; do
        out="$MODELS/$arm-$seed"
        [ -e "$out/report.json" ] && { echo "skip $arm-$seed (done)"; continue; }
        "${GPU[@]}" $PY scripts/train_task087.py --features "$FEAT" --arm "$arm" --seed "$seed" \
          --updates "$U" --select-every "$EVERY" --cap-seconds "$CAP" --output "$out" \
          </dev/null 2>&1 | tee -a "$RUN/train-$arm-$seed.log"
      done
    done ;;
  val)
    "${GPU[@]}" $PY scripts/evaluate_task087.py --features "$FEAT" --split val \
      --models "$MODELS"/*-871* --output "$RUN/val_report.json" </dev/null 2>&1 | tee -a "$RUN/val.log"
    sha256sum "$RUN/val_report.json" | tee -a "$RUN/val.log" ;;
  features-test)
    "${GPU[@]}" $PY scripts/featurise_task087.py --stage test --output "$FEAT" --workers 8 \
      --val-report "$RUN/val_report.json" --val-report-sha256 "${2:?val report sha256}" \
      </dev/null 2>&1 | tee -a "$RUN/features-test.log" ;;
  test)
    "${GPU[@]}" $PY scripts/evaluate_task087.py --features "$FEAT" --split test \
      --models "$MODELS"/*-871* --output "$RUN/test_report.json" \
      --val-report "$RUN/val_report.json" --val-report-sha256 "${2:?val report sha256}" \
      </dev/null 2>&1 | tee -a "$RUN/test.log" ;;
  *) sed -n '2,15p' "$0" >&2; exit 2 ;;
esac
