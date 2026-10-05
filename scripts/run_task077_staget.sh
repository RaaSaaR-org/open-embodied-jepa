#!/usr/bin/env bash
# TASK-077 Stage T driver (protocol docs/experiments/apple_lewm_c1m_v2.md §7.6; R17.44-R17.45).
# A development helper that runs the eight Stage T jobs and the plan in order; not hash-pinned.
#
# Usage, from the root of a fresh clean worktree of the merge commit a GO names:
#   bash scripts/run_task077_staget.sh
#
# Steps, each checked by its report's outcome before the next step starts:
#   0. the G-tests record (CPU, no lock)                    -> must end TESTS-PASS
#   1-2. cal-W, cal-N (GPU, one gpu_run.sh slot each)       -> each must end T-JOB-DONE
#   3. plan (CPU, no lock; G-quiet, in-run G-tests)         -> must end T-PLANNED
#        (CAL-T-ESCALATE exits 0 from the runner, so it is checked here: Stage T stops and
#        escalates, and no model job takes the GPU lock)
#   4-9. W and N of 66800, 66801, 66802 (GPU, one slot each) -> each must end T-JOB-DONE
# Any other outcome, a missing report or a non-zero exit stops the script with a non-zero status.
# The runner refuses an existing output folder, so nothing is overwritten; a V of a job is
# repeated only after a recorded fix, under its own GO (protocol §10.1).
#
# Inputs (fixed; overridable only for tests): C, CS (the sealed corpus and its manifest sha256),
# F (Stage O's features), R (Stage O's fits), SUFFIX (the output folders' run number).
# PY (the Python command, default "uv run --no-sync python") and GPU_RUN (default
# scripts/gpu_run.sh) are overridable for the dry-run test only.
set -euo pipefail

C=${C:-/home/huhn/develop/emai/worktrees/task077-corpus2/outputs/task077-corpus-2/corpus}
CS=${CS:-ad8974b2a8b560bb974c6e0b4f90bd3f1fc79a535ebe46ef6c409bde7e4343fb}
F=${F:-/home/huhn/develop/emai/worktrees/task077-stageo/outputs/task077-featurise-1/features}
R=${R:-/home/huhn/develop/emai/worktrees/task077-stageo/outputs/task077-readouts-1/fits}
SUFFIX=${SUFFIX:-1}
read -r -a PY <<< "${PY:-uv run --no-sync python}"
GPU_RUN=${GPU_RUN:-scripts/gpu_run.sh}
RUNNER=scripts/run_lewm_c1m_v2.py
OUT=outputs
T=$OUT/task077-t-tests-$SUFFIX/report.json

outcome() {  # outcome <report.json>: its "outcome" field (empty when missing or unreadable)
  python3 -c 'import json, sys
try:
    print(json.load(open(sys.argv[1])).get("outcome") or "")
except Exception:
    print("")' "$1"
}

require() {  # require <report.json> <expected outcome> <what>
  local got
  got=$(outcome "$1")
  if [ "$got" != "$2" ]; then
    echo "Stage T stops: $3 ended '${got:-no report}', not $2 ($1)" >&2
    exit 1
  fi
  echo "$3: $2"
}

gpu_job() {  # gpu_job <job> [extra runner args...]
  local job=$1
  shift
  "$GPU_RUN" --wait --min-free-gib 8 --board --who "oej:task077-staget-$job" -- \
    "${PY[@]}" "$RUNNER" train --job "$job" "$@" \
    --output "$OUT/task077-t-$job-$SUFFIX" --tests-record "$T" \
    --corpus "$C" --corpus-sha256 "$CS" --features "$F" --fits "$R" \
    --log "$OUT/task077-t-$job-$SUFFIX.log" < /dev/null
  require "$OUT/task077-t-$job-$SUFFIX/report.json" T-JOB-DONE "$job"
}

# 0. the G-tests record at HEAD
"${PY[@]}" "$RUNNER" tests --output "$OUT/task077-t-tests-$SUFFIX" \
  --log "$OUT/task077-t-tests-$SUFFIX.log"
require "$T" TESTS-PASS "tests"

# 1-2. the calibration jobs
for job in cal-W cal-N; do
  gpu_job "$job"
done

# 3. the plan: the budget, G1's bars, CAL-T-ESCALATE
PLAN=$OUT/task077-t-plan-$SUFFIX/report.json
"${PY[@]}" "$RUNNER" plan --output "$OUT/task077-t-plan-$SUFFIX" \
  --corpus "$C" --corpus-sha256 "$CS" --features "$F" --fits "$R" \
  --cal-w "$OUT/task077-t-cal-W-$SUFFIX/report.json" \
  --cal-n "$OUT/task077-t-cal-N-$SUFFIX/report.json" \
  --log "$OUT/task077-t-plan-$SUFFIX.log"
require "$PLAN" T-PLANNED "plan"

# 4-9. the six models
for job in W-66800 N-66800 W-66801 N-66801 W-66802 N-66802; do
  gpu_job "$job" --plan "$PLAN"
done
echo "Stage T: all eight jobs ended T-JOB-DONE; T-DONE is decided by Stage G (R17.24)"
