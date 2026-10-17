#!/usr/bin/env bash
# TASK-100's stages (protocol docs/experiments/so101_world_model_v1.md §8.4, §10), one GPU lock per
# job (scripts/gpu_run.sh --wait --board); CPU-only steps (fit, readouts, score) take no lock.
#
# Usage: scripts/run_task100_stage.sh <stage> [args]
#  Stage 0 (development, before the freeze; any tracked tree):
#   s0-features <encoder> <grids>   featurise D0 and the smoke's val episodes (GPU)
#   s0-fit <encoder> <grid>         D0-fit standardisation (and the G1 (iii) basis) (CPU)
#   s0-grid                         §10.3: R-enc on D0-eval for g4 and g8 (CPU)
#   s0-calibrate <grid>             §10.4: calibration L and N_L, 20 000 updates on D0-fit (GPU)
#   s0-cal-eval <grid>              fitpred, readouts, evalpred on D0-eval, score calibrate
#   s0-smoke <grid> <updates>       §10.6: every arm on D0-fit, val standing in for val and test
#   s0-determinism <grid> <updates> §10.6: one debug job trained twice
#  Gated stages (a clean run tree at the frozen revision):
#   features | fit | copy | probe <used-gpu-h> | train | extend | val
#   features-test <val-sha256> | test <val-sha256>
#
# Paths: Stage 0 under $OUT/task100-stage0*; gated features $OUT/task100-features-<encoder> (the
# SSD working copy under $SSD), models $CK/task100-run, reports $OUT/task100-run. $OUT and $CK
# are the shared checkout's outputs/ and checkpoints/ (symlinks to the HDD). Every job refuses
# an existing output.
set -euo pipefail
REPO=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cd "$REPO"
SHARED=$(dirname -- "$(cd -- "$(git rev-parse --git-common-dir)" && pwd -P)")
OUT=${TASK100_OUT:-$SHARED/outputs}
CK=${TASK100_CK:-$SHARED/checkpoints}
SSD=${TASK100_SSD:-$HOME/develop/emai/ssd-work}
PY=.venv/bin/python
GPU=(scripts/gpu_run.sh --wait --min-free-gib 6 --board --)
S0=$OUT/task100-stage0
S0F=$OUT/task100-stage0-features
RUN=$OUT/task100-run
MODELS=$CK/task100-run
SEEDS=(100100 100101 100102)
ARMS=(L J N_L N_J R)

feat_for() {  # <arm> <features root> -> the arm's encoder folder
  case "$1" in R) echo "$2-random_init" ;; *) echo "$2-pretrained" ;; esac
}
grid() { $PY -c "from embodied_jepa import so101_wm as sw; print(sw.frozen_choices()['grid'])"; }
clean() { [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "tracked tree is dirty" >&2; exit 1; }; }
completed() {  # every model's finished job (its extension when T' ran, else the job or its -r2)
  for seed in "${SEEDS[@]}"; do for arm in "${ARMS[@]}"; do
    local base="$MODELS/$arm-$seed"
    for d in "$base-ext" "$base-ext-r2" "$base" "$base-r2"; do
      if [ -e "$d/report.json" ]; then echo "$d"; continue 2; fi
    done
    echo "missing $arm-$seed" >&2; return 1
  done; done
}

stage=${1:-}; shift || true
case "$stage" in
  s0-features)
    enc=${1:?encoder}; grids=${2:?grids}
    mkdir -p "$S0"
    "${GPU[@]}" $PY scripts/featurise_task100.py --stage stage0 --encoder "$enc" --grids "$grids" \
      --output "$S0F-$enc" --workers 8 </dev/null 2>&1 | tee -a "$S0/features-$enc.log" ;;
  s0-fit)
    $PY scripts/featurise_task100.py --stage fit --features "$S0F-${1:?encoder}/${2:?grid}" \
      --view d0-fit 2>&1 | tee -a "$S0/fit-$1-$2.log" ;;
  s0-grid)
    for g in g4 g8; do
      $PY scripts/evaluate_task100.py readouts --purpose grid --fit-view d0-fit \
        --features-pretrained "$S0F-pretrained/$g" --output "$S0/readouts-grid-$g" 2>&1 | tee -a "$S0/grid.log"
    done
    $PY scripts/evaluate_task100.py score --purpose grid --eval-view d0-eval \
      --features-pretrained "$S0F-pretrained/g4" --features-g8 "$S0F-pretrained/g8" \
      --readouts "$S0/readouts-grid-g4" --readouts-g8 "$S0/readouts-grid-g8" \
      --output "$S0/grid_report.json" 2>&1 | tee -a "$S0/grid.log" ;;
  s0-calibrate)
    g=${1:?grid}
    for spec in "L 100900" "N_L 100911"; do
      set -- $spec
      "${GPU[@]}" $PY scripts/train_task100.py --features "$S0F-pretrained/$g" --grid "$g" \
        --arm "$1" --seed "$2" --kind calibration --train-view d0-fit --select-view d0-eval \
        --updates 20000 --select-every 1000 --cap-seconds 14400 --output "$CK/task100-stage0/cal-$1-$2" \
        </dev/null 2>&1 | tee -a "$S0/calibrate.log"
    done ;;
  s0-cal-eval)
    g=${1:?grid}; M=("$CK/task100-stage0/cal-L-100900" "$CK/task100-stage0/cal-N_L-100911")
    "${GPU[@]}" $PY scripts/evaluate_task100.py fitpred --purpose calibrate --fit-view d0-fit \
      --features-pretrained "$S0F-pretrained/$g" --models "${M[@]}" --output "$S0/cal-fitpred" \
      </dev/null 2>&1 | tee -a "$S0/cal-eval.log"
    $PY scripts/evaluate_task100.py readouts --purpose calibrate --fit-view d0-fit \
      --features-pretrained "$S0F-pretrained/$g" --fitpred "$S0/cal-fitpred" \
      --output "$S0/cal-readouts" 2>&1 | tee -a "$S0/cal-eval.log"
    "${GPU[@]}" $PY scripts/evaluate_task100.py evalpred --purpose calibrate --eval-view d0-eval \
      --features-pretrained "$S0F-pretrained/$g" --models "${M[@]}" --readouts "$S0/cal-readouts" \
      --output "$S0/cal-evalpred" </dev/null 2>&1 | tee -a "$S0/cal-eval.log"
    $PY scripts/evaluate_task100.py score --purpose calibrate --eval-view d0-eval \
      --features-pretrained "$S0F-pretrained/$g" --readouts "$S0/cal-readouts" \
      --evalpred "$S0/cal-evalpred" --output "$S0/calibration_report.json" 2>&1 | tee -a "$S0/cal-eval.log" ;;
  s0-smoke)
    g=${1:?grid}; u=${2:?updates}; M=(); seed=100901
    for arm in "${ARMS[@]}"; do
      out="$CK/task100-stage0/smoke-$arm-$seed"
      "${GPU[@]}" $PY scripts/train_task100.py --features "$(feat_for "$arm" "$S0F")/$g" --grid "$g" \
        --arm "$arm" --seed "$seed" --kind debug --train-view d0-fit --select-view smoke-val \
        --updates "$u" --select-every $((u / 4)) --cap-seconds 3600 --output "$out" \
        </dev/null 2>&1 | tee -a "$S0/smoke.log"
      M+=("$out")
    done
    FP=(--features-pretrained "$S0F-pretrained/$g" --features-random "$S0F-random_init/$g")
    "${GPU[@]}" $PY scripts/evaluate_task100.py fitpred --purpose smoke --fit-view d0-fit "${FP[@]}" \
      --models "${M[@]}" --output "$S0/smoke-fitpred" </dev/null 2>&1 | tee -a "$S0/smoke.log"
    $PY scripts/evaluate_task100.py readouts --purpose smoke --fit-view d0-fit "${FP[@]}" \
      --fitpred "$S0/smoke-fitpred" --output "$S0/smoke-readouts" 2>&1 | tee -a "$S0/smoke.log"
    for v in smoke-val smoke-test; do
      "${GPU[@]}" $PY scripts/evaluate_task100.py evalpred --purpose smoke --eval-view "$v" "${FP[@]}" \
        --models "${M[@]}" --readouts "$S0/smoke-readouts" --output "$S0/smoke-evalpred-$v" \
        </dev/null 2>&1 | tee -a "$S0/smoke.log"
      $PY scripts/evaluate_task100.py score --purpose smoke --eval-view "$v" "${FP[@]}" \
        --readouts "$S0/smoke-readouts" --evalpred "$S0/smoke-evalpred-$v" \
        --output "$S0/smoke_report_$v.json" 2>&1 | tee -a "$S0/smoke.log"
    done ;;
  s0-determinism)
    g=${1:?grid}; u=${2:?updates}
    for k in a b; do
      "${GPU[@]}" $PY scripts/train_task100.py --features "$S0F-pretrained/$g" --grid "$g" \
        --arm L --seed 100902 --kind debug --train-view d0-fit --select-view d0-eval \
        --updates "$u" --select-every $((u / 2)) --cap-seconds 3600 \
        --output "$CK/task100-stage0/determinism-$k" </dev/null 2>&1 | tee -a "$S0/determinism.log"
    done
    $PY -c "import json,sys;a,b=(json.load(open('$CK/task100-stage0/determinism-'+k+'/report.json')) for k in 'ab');print('final state', a['final_state_sha256'], b['final_state_sha256'], 'IDENTICAL' if a['final_state_sha256']==b['final_state_sha256'] else 'DIFFERENT')" | tee -a "$S0/determinism.log" ;;
  features)
    clean; g=$(grid); mkdir -p "$RUN"
    for enc in pretrained random_init; do
      "${GPU[@]}" $PY scripts/featurise_task100.py --stage trainval --encoder "$enc" \
        --output "$OUT/task100-features-$enc" --workers 8 </dev/null 2>&1 | tee -a "$RUN/features.log"
    done ;;
  fit)
    clean
    for enc in pretrained random_init; do
      $PY scripts/featurise_task100.py --stage fit --features "$OUT/task100-features-$enc" \
        --view fit-sample 2>&1 | tee -a "$RUN/fit.log"
    done ;;
  copy)  # §8.4: the SSD working copy of train and val, only if / keeps 10 GiB free after it
    clean; mkdir -p "$SSD"
    need=$(du -sb "$OUT"/task100-features-pretrained "$OUT"/task100-features-random_init | awk '{s+=$1} END {print s}')
    free=$(df -B1 --output=avail / | tail -1)
    if [ $((free - need)) -lt $((10 * 1024 * 1024 * 1024)) ]; then
      echo "copy refused: / has $free bytes free, the copy needs $need (10 GiB floor); train from the HDD" | tee -a "$RUN/copy.log"
      exit 0
    fi
    for enc in pretrained random_init; do
      [ -e "$SSD/task100-features-$enc" ] && { echo "refusing to overwrite $SSD/task100-features-$enc" >&2; exit 1; }
      cp -a "$OUT/task100-features-$enc" "$SSD/task100-features-$enc"
      (cd "$SSD/task100-features-$enc" && sha256sum ./*.npy ./*.json ./*.npz > "$RUN/ssd-copy-$enc.sha256")
      (cd "$OUT/task100-features-$enc" && sha256sum -c "$RUN/ssd-copy-$enc.sha256") | tail -1 | tee -a "$RUN/copy.log"
    done ;;
  probe)
    clean; g=$(grid); F=$SSD/task100-features-pretrained; [ -d "$F" ] || F=$OUT/task100-features-pretrained
    "${GPU[@]}" $PY scripts/train_task100.py --features "$F" --grid "$g" --arm L --seed 100920 \
      --kind probe --updates 60000 --select-every 2000 --cap-seconds 28800 --used-gpu-h "${1:?GPU h used}" \
      --output "$MODELS/probe-L-100920" </dev/null 2>&1 | tee -a "$RUN/probe.log" ;;
  train)
    clean; g=$(grid)
    read -r U EVERY CAP ESC < <($PY -c "import json;b=json.load(open('$MODELS/probe-L-100920/report.json'))['budget'];print(b['updates'],b['select_every'],int(b['cap_seconds'])+1,int(b['escalate']))")
    [ "$ESC" = 0 ] || { echo "the budget does not fit the cap at U = 20 000: stop for a ruling (§8.2)" >&2; exit 1; }
    for seed in "${SEEDS[@]}"; do for arm in "${ARMS[@]}"; do
      out="$MODELS/$arm-$seed"
      [ -e "$out/report.json" ] && { echo "skip $arm-$seed (done)"; continue; }
      [ -e "$out" ] && { echo "$arm-$seed did not complete; re-run it once by hand as $out-r2" >&2; exit 1; }
      F=$(feat_for "$arm" "$SSD/task100-features"); [ -d "$F" ] || F=$(feat_for "$arm" "$OUT/task100-features")
      "${GPU[@]}" $PY scripts/train_task100.py --features "$F" --grid "$g" --arm "$arm" --seed "$seed" \
        --kind job --updates "$U" --select-every "$EVERY" --cap-seconds "$CAP" --output "$out" \
        </dev/null 2>&1 | tee -a "$RUN/train-$arm-$seed.log"
    done; done ;;
  extend)  # G6's one extension (Stage T'): all six models of an eligible arm resume to 1.5 U
    clean; g=$(grid)
    read -r U EVERY CAP < <($PY -c "import json;b=json.load(open('$MODELS/probe-L-100920/report.json'))['budget'];print(b['updates'],b['select_every'],int(b['cap_seconds'])+1)")
    for pair in "L N_L" "J N_J"; do
      set -- $pair
      need=$($PY -c "
import json; from embodied_jepa import so101_wm as sw
c = {}
for a in '$1 $2'.split():
    for s in (100100, 100101, 100102):
        d = '$MODELS/%s-%d' % (a, s)
        import os; d = d if os.path.exists(d + '/report.json') else d + '-r2'
        c[(a, s)] = json.load(open(d + '/report.json'))['training']['val_curve']
print(int(sw.extension_needed(c)))")
      [ "$need" = 1 ] || { echo "G6: $1 and $2 saturated; no extension"; continue; }
      for arm in "$1" "$2"; do for seed in "${SEEDS[@]}"; do
        base="$MODELS/$arm-$seed"; [ -e "$base/report.json" ] || base="$base-r2"
        F=$(feat_for "$arm" "$SSD/task100-features"); [ -d "$F" ] || F=$(feat_for "$arm" "$OUT/task100-features")
        "${GPU[@]}" $PY scripts/train_task100.py --features "$F" --grid "$g" --arm "$arm" --seed "$seed" \
          --kind job --updates $((U * 3 / 2)) --select-every "$EVERY" --cap-seconds "$CAP" \
          --resume "$base" --extend-to $((U * 3 / 2)) --output "$MODELS/$arm-$seed-ext" \
          </dev/null 2>&1 | tee -a "$RUN/extend-$arm-$seed.log"
      done; done
    done ;;
  val)
    clean; mapfile -t M < <(completed)
    FP=(--features-pretrained "$OUT/task100-features-pretrained" --features-random "$OUT/task100-features-random_init")
    "${GPU[@]}" $PY scripts/evaluate_task100.py fitpred --purpose val --fit-view fit-sample "${FP[@]}" \
      --models "${M[@]}" --output "$RUN/fitpred" </dev/null 2>&1 | tee -a "$RUN/val.log"
    $PY scripts/evaluate_task100.py readouts --purpose val --fit-view fit-sample "${FP[@]}" \
      --fitpred "$RUN/fitpred" --output "$RUN/readouts" 2>&1 | tee -a "$RUN/val.log"
    "${GPU[@]}" $PY scripts/evaluate_task100.py evalpred --purpose val --eval-view val "${FP[@]}" \
      --models "${M[@]}" --readouts "$RUN/readouts" --output "$RUN/evalpred-val" \
      </dev/null 2>&1 | tee -a "$RUN/val.log"
    $PY scripts/evaluate_task100.py score --purpose val --eval-view val "${FP[@]}" \
      --readouts "$RUN/readouts" --evalpred "$RUN/evalpred-val" --output "$RUN/val_report.json" \
      2>&1 | tee -a "$RUN/val.log"
    sha256sum "$RUN/val_report.json" | tee -a "$RUN/val.log" ;;
  features-test)
    clean; sha=${1:?val report sha256}
    for enc in pretrained random_init; do
      "${GPU[@]}" $PY scripts/featurise_task100.py --stage test --encoder "$enc" \
        --output "$OUT/task100-features-$enc" --val-report "$RUN/val_report.json" \
        --val-report-sha256 "$sha" --workers 8 </dev/null 2>&1 | tee -a "$RUN/features-test.log"
    done ;;
  test)
    clean; sha=${1:?val report sha256}; mapfile -t M < <(completed)
    FP=(--features-pretrained "$OUT/task100-features-pretrained" --features-random "$OUT/task100-features-random_init")
    V=(--val-report "$RUN/val_report.json" --val-report-sha256 "$sha")
    "${GPU[@]}" $PY scripts/evaluate_task100.py evalpred --purpose test --eval-view test "${FP[@]}" "${V[@]}" \
      --models "${M[@]}" --readouts "$RUN/readouts" --output "$RUN/evalpred-test" \
      </dev/null 2>&1 | tee -a "$RUN/test.log"
    $PY scripts/evaluate_task100.py score --purpose test --eval-view test "${FP[@]}" "${V[@]}" \
      --readouts "$RUN/readouts" --evalpred "$RUN/evalpred-test" --output "$RUN/test_report.json" \
      2>&1 | tee -a "$RUN/test.log"
    $PY scripts/summarize_task100.py --report "$RUN/test_report.json" | tee "$RUN/summary.md" ;;
  *) sed -n '2,21p' "$0" >&2; exit 2 ;;
esac
