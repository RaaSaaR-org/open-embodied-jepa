#!/usr/bin/env bash
# TASK-090: run one stage (protocol docs/experiments/grasp_play_corpus_v1.md §4-§7), each GPU use
# its own scripts/gpu_run.sh job, then summarise. Run from a clean tree at the frozen revision.
#   scripts/run_task090_stage.sh collect <out>     both collections (data/play-v2g, data/play-v1p)
#   scripts/run_task090_stage.sh c <out>           Stage C: labels and the data gate (CPU)
#   scripts/run_task090_stage.sh features <out>    featurise both collections (train, val)
#   scripts/run_task090_stage.sh t <out>           Stage T: the eight jobs (needs C-PASS)
#   scripts/run_task090_stage.sh e <out>           Stage E
#   scripts/run_task090_stage.sh a <out> <k0-ceiling.json>   Stage A′
#   scripts/run_task090_stage.sh bd <out>          Stage BD (development)
#   scripts/run_task090_stage.sh bs <out> <summary-bd.json>  Stage BS (gated; after GO)
# Every output file is refused if it exists (the runners and summarize_task090.py).
set -euo pipefail
STAGE=${1:?stage}; OUT=${2:?output dir}
cd "$(dirname "$0")/.."
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "tracked tree is dirty" >&2; exit 1; }
mkdir -p "$OUT"
PYB=.venv/bin/python
PY=(env MUJOCO_GL=egl "$PYB" scripts/run_task090.py)
CAP=$("$PYB" -c "import sys; sys.path.insert(0, 'src'); from embodied_jepa import grasp_corpus_wm as gc; print(int(gc.TRAIN_CAP_SECONDS))")
MMAP=$("$PYB" -c "import sys; sys.path.insert(0, 'src'); from embodied_jepa import grasp_corpus_wm as gc; print('--mmap' if gc.TRAIN_MMAP else '')")
# gpu_run.sh exits 76 when it refuses to start (VRAM / utilisation check, before the command
# runs); only that is retried, up to 10 times a minute apart. Any other failure stops the stage.
gpu() {
  local tries=0 code
  while :; do
    set +e
    scripts/gpu_run.sh --wait --min-free-gib 8 --board --samples 5 --who "oej:task090-$STAGE" -- "$@" </dev/null
    code=$?
    set -e
    [ "$code" -ne 76 ] && return "$code"
    tries=$((tries + 1)); [ "$tries" -ge 10 ] && return 76
    sleep 60
  done
}
WM_ARMS() {  # arm seed pairs of a cohort, from run_task090's tables
  "$PYB" -c "import sys; sys.path.insert(0, 'src'); from embodied_jepa import grasp_corpus_wm as gc; t = gc.BD_SEEDS if '$1' == 'bd' else gc.BS_SEEDS; print(' '.join(f'{a}:{s}' for a in ('M-v2g', 'M-v1p', 'M-v1') for s in t[a]))"
}
case "$STAGE" in
  collect)
    for c in play-v2g play-v1p; do
      gpu env MUJOCO_GL=egl "$PYB" scripts/collect_task090.py --corpus "$c" --out "data/$c" \
        --workers 12 --evidence "$OUT/collect-$c"
    done ;;
  c)
    "$PYB" scripts/measure_task090.py labels --corpus data/play-v1 --split train --output "$OUT/labels-play-v1-train.json"
    for c in play-v2g play-v1p; do
      "$PYB" scripts/measure_task090.py labels --corpus "data/$c" --split train --output "$OUT/labels-$c-train.json"
    done
    "$PYB" scripts/measure_task090.py stage-c --v1 "$OUT/labels-play-v1-train.json" \
      --v2g "$OUT/labels-play-v2g-train.json" --v1p "$OUT/labels-play-v1p-train.json" \
      --output "$OUT/stage-c.json"
    echo "STAGE-c-DONE"; exit 0 ;;
  features)
    for c in play-v2g play-v1p; do
      gpu "${PY[@]}" featurise --corpus "$c" --workers 8
    done ;;
  t)
    row=$("$PYB" -c "import json; print(json.load(open('$OUT/stage-c.json'))['stage_c']['row'])")
    [ "$row" = C-PASS ] || { echo "Stage C is $row: no training (§4)" >&2; exit 1; }
    for job in G:play-v2g:90100 G:play-v2g:90101 G:play-v2g:90102 G:play-v1p:90100 \
               G:play-v1p:90101 G:play-v1p:90102 N:play-v2g:90100 N:play-v1p:90100; do
      IFS=: read -r arm corpus seed <<< "$job"
      tag=${corpus#play-}; out="checkpoints/task090-run/$arm-$tag-$seed"
      [ -e "$out/report.json" ] && { echo "skip $arm-$tag-$seed (done)"; continue; }
      [ -e "$out" ] && { echo "$out did not complete: T-VOID, stop for a ruling (§5)" >&2; exit 1; }
      gpu "$PYB" scripts/train_task090.py --corpus "$corpus" --arm "$arm" --seed "$seed" \
        --cap-seconds "$CAP" $MMAP --output "$out" 2>&1 | tee -a "$OUT/train-$arm-$tag-$seed.log"
    done
    mkdir -p "$OUT/t"; for d in checkpoints/task090-run/*/; do
      n=$(basename "$d"); mkdir -p "$OUT/t/$n"; cp "$d/report.json" "$OUT/t/$n/report.json"; done
    "$PYB" scripts/summarize_task090.py --stage t --runs "$OUT/t"
    echo "STAGE-t-DONE"; exit 0 ;;
  e)
    gpu "${PY[@]}" stage-e --output "$OUT/stage-e.json" ;;
  a)
    K0=${3:?A-K0 ceiling file}
    gpu "${PY[@]}" roots --cohort a-roots --output "$OUT" --workers 8
    gpu "${PY[@]}" a-prime --cohort a-roots --roots "$OUT" --k0 "$K0" --output "$OUT" ;;
  bd|bs)
    C=bd-grasp; [ "$STAGE" = bs ] && C=bs-grasp
    gpu "${PY[@]}" goals --cohort "$C" --output "$OUT" --workers 8
    for a in ik ik-nopress hold random; do
      gpu "${PY[@]}" run --cohort "$C" --arm "$a" --goals "$OUT" --output "$OUT" --workers 8
    done
    for pair in $(WM_ARMS "$STAGE"); do
      gpu "${PY[@]}" run --cohort "$C" --arm "${pair%%:*}" --model-seed "${pair##*:}" \
        --goals "$OUT" --output "$OUT" --workers 6
    done
    if [ "$STAGE" = bs ]; then
      gpu "${PY[@]}" run --cohort "$C" --arm M-v2g --model-seed 90100 --determinism \
        --goals "$OUT" --output "$OUT" --workers 4
    fi ;;
  *) sed -n '2,13p' "$0" >&2; exit 2 ;;
esac
EXTRA=()
[ "$STAGE" = bs ] && EXTRA=(--bd-summary "${3:?Stage BD summary}")
case "$STAGE" in
  e|bd|bs) "$PYB" scripts/summarize_task090.py --stage "$STAGE" --runs "$OUT" "${EXTRA[@]}" ;;
  a) "$PYB" scripts/summarize_task090.py --stage a --runs "$OUT" ;;
esac
echo "STAGE-$STAGE-DONE"
