#!/usr/bin/env bash
# TASK-077 Stage T driver (protocol docs/experiments/apple_lewm_c1m_v2.md §7.6, §7.7;
# R17.44-R17.47). A development helper that runs the eight Stage T jobs and the plan in order;
# not hash-pinned. Written for bash 3.2 or later.
#
# Usage, from the root of a fresh clean worktree of the merge commit a GO names:
#   bash scripts/run_task077_staget.sh                       # a fresh Stage T (R17.45)
#   bash scripts/run_task077_staget.sh --resume-from DIR --repeat N-66800 [--stop-after STEP]
#   bash scripts/run_task077_staget.sh ... --dry-run         # print what would run; run nothing
#
# Steps, each checked by its report's outcome before the next step starts:
#   0. the G-tests record (CPU, no lock)                    -> must end TESTS-PASS
#   1-2. cal-W, cal-N (GPU, one gpu_run.sh slot each)       -> each must end T-JOB-DONE
#   3. plan (CPU, no lock; G-quiet, in-run G-tests)         -> must end T-PLANNED
#        (CAL-T-ESCALATE exits 0 from the runner, so it is checked here: Stage T stops and
#        escalates, and no model job takes the GPU lock)
#   4-9. W and N of 66800, 66801, 66802 (GPU, one slot each) -> each must end T-JOB-DONE
# Any other outcome, a missing report or a non-zero exit stops the script with a non-zero status.
# The runner refuses an existing output folder, so nothing is overwritten.
#
# Options (R17.47):
#   --resume            resume mode: look for earlier attempts of each step in outputs/ (and in
#                       every --resume-from folder). Without it (and without --resume-from), any
#                       earlier attempt of a step stops the script, as under R17.45.
#   --resume-from DIR   an earlier run's outputs folder (repeatable; implies --resume). Read only.
#   --repeat JOB        allow JOB's one repeat (protocol §10.1): JOB must have exactly one earlier
#                       attempt, and it must have ended V. The repeat writes a new folder with the
#                       next run number (task077-t-JOB-2). Name a job here only after the V's
#                       cause and its fix or prevention are recorded and a GO names the repeat.
#   --stop-after STEP   stop cleanly (exit 0) once STEP has ended as required (or was kept).
#   --dry-run           resolve every step and print keep / run / stop; start nothing.
# In resume mode each step is decided by its earlier attempts:
#   - one attempt ended as required: kept, not re-run (a --repeat of it is then ignored). A kept job's checkpoint must exist beside
#     its report with the recorded sha256; a kept report named in KEPT must have that sha256; a
#     kept plan must have been made from the kept calibration reports. The G-tests record is kept
#     only from outputs/ and only when its revision is HEAD (G-tests checks HEAD).
#   - one V and no --repeat: stop (a V's repeat needs its own record and GO).
#   - two Vs of the same job: stop. TASK-077 ends INCONCLUSIVE (§10.1); nothing more runs.
#   - one V and --repeat: the one repeat, in a new folder. A V's folder is never reused.
#   - any other earlier outcome, or a folder without a report: stop; the operator decides.
#   - CAL-T-ESCALATE at plan: stop (escalate).
# Pause: if the pause file (outputs/task077-staget.pause) exists when a step is about to start,
# the script exits 0 before that step, with nothing started. A job that has started runs to its
# end, so `touch outputs/task077-staget.pause` pauses between jobs, never inside one.
#
# Inputs (fixed; overridable only for tests): C, CS (the sealed corpus and its manifest sha256),
# F (Stage O's features), R (Stage O's fits), SUFFIX (the first run number), KEPT (step=sha256
# pins of kept reports; R17.46). PY (the Python command, default "uv run --no-sync python"),
# GPU_RUN (default scripts/gpu_run.sh), REV (default git HEAD) and PAUSE_FILE are overridable for
# the dry-run tests only.
set -euo pipefail

C=${C:-/home/huhn/develop/emai/worktrees/task077-corpus2/outputs/task077-corpus-2/corpus}
CS=${CS:-ad8974b2a8b560bb974c6e0b4f90bd3f1fc79a535ebe46ef6c409bde7e4343fb}
F=${F:-/home/huhn/develop/emai/worktrees/task077-stageo/outputs/task077-featurise-1/features}
R=${R:-/home/huhn/develop/emai/worktrees/task077-stageo/outputs/task077-readouts-1/fits}
SUFFIX=${SUFFIX:-1}
# The four reports that Stage T's first run (at 215fcce, 2026-10-05) completed (R17.46).
KEPT=${KEPT-"cal-W=2339f5f1dc4447af6c04a7ea40e9d22708aec849bece080c57d5301a7e270cb8
cal-N=42ad6cf892f14c6338608d3d9b148eff34db8e797892a98ac5d46ce08e984423
plan=7c7147601c156a793c238e34d00ee74e9855f29d19e4ee0d653882335d67cdfb
W-66800=42195ded1ac1c41a77ed0b47e2031e4066f9bc2c2a61637ea5aa17f08e05757b"}
read -r -a PY <<< "${PY:-uv run --no-sync python}"
GPU_RUN=${GPU_RUN:-scripts/gpu_run.sh}
RUNNER=scripts/run_lewm_c1m_v2.py
OUT=outputs
PAUSE_FILE=${PAUSE_FILE:-$OUT/task077-staget.pause}
REV=${REV:-$(git rev-parse HEAD 2>/dev/null || true)}
JOBS="cal-W cal-N W-66800 N-66800 W-66801 N-66801 W-66802 N-66802"
STEPS="tests cal-W cal-N plan W-66800 N-66800 W-66801 N-66801 W-66802 N-66802"

RESUME=0
DRY=0
STOP_AFTER=""
REPEATS=" "
SEARCH=()

die() { echo "Stage T stops: $*" >&2; exit 1; }
is_word() { case " $2 " in *" $1 "*) return 0 ;; esac; return 1; }  # is_word <x> "<list>"
repeats() { is_word "$1" "$REPEATS"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --resume) RESUME=1; shift ;;
    --resume-from)
      [ -n "${2:-}" ] && [ -d "$2" ] || die "--resume-from needs an existing folder (got '${2:-}')"
      RESUME=1; SEARCH+=("$2"); shift 2 ;;
    --repeat)
      [ -n "${2:-}" ] && is_word "$2" "$JOBS" || die "--repeat needs one of: $JOBS (got '${2:-}')"
      REPEATS="$REPEATS$2 "; shift 2 ;;
    --stop-after)
      [ -n "${2:-}" ] && is_word "$2" "$STEPS" \
        || die "--stop-after needs one of: $STEPS (got '${2:-}')"
      STOP_AFTER=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) die "unknown argument '$1'" ;;
  esac
done
if [ "$REPEATS" != " " ] && [ "$RESUME" = 0 ]; then
  die "--repeat needs --resume or --resume-from (a repeat follows an earlier attempt)"
fi
SEARCH+=("$OUT")

# One Python helper for the report reads (python3, not sha256sum, which not every runner has).
read -r -d '' PYQ <<'PYEOF' || true
import hashlib, json, os, sys
cmd, path = sys.argv[1], sys.argv[2]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
if cmd == "sha256":
    print(sha(path))
    sys.exit(0)
if cmd == "realpath":
    print(os.path.realpath(path))
    sys.exit(0)
try:
    r = json.load(open(path))
    r = r if isinstance(r, dict) else {}
except Exception:
    r = {}
if cmd == "outcome":
    print(r.get("outcome") or "")
elif cmd == "revision":
    print(r.get("revision") or "")
elif cmd == "checkpoint":  # "ok", or why the job's checkpoint is not intact
    rec = (r.get("fields") or {}).get("record")
    if not isinstance(rec, dict) or not rec.get("checkpoint") or not rec.get("checkpoint_sha256"):
        print("no checkpoint recorded")
    else:
        ck = os.path.join(os.path.dirname(path), os.path.basename(rec["checkpoint"]))
        if not os.path.isfile(ck):
            print(f"checkpoint missing: {ck}")
        elif sha(ck) != rec["checkpoint_sha256"]:
            print(f"checkpoint sha256 differs from its report: {ck}")
        else:
            print("ok")
elif cmd == "input":  # input <report> <flag>: that argument, resolved against the run's root
    argv, flag = r.get("argv") or [], sys.argv[3]
    if flag in argv and argv.index(flag) + 1 < len(argv):
        v = argv[argv.index(flag) + 1]
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(path))))
        print(os.path.realpath(v if os.path.isabs(v) else os.path.join(root, v)))
PYEOF
pyq() { python3 -c "$PYQ" "$@"; }
outcome() { pyq outcome "$1"; }

require() {  # require <report.json> <expected outcome> <what>
  local got
  got=$(outcome "$1")
  [ "$got" = "$2" ] || die "$3 ended '${got:-no report}', not $2 ($1)"
  echo "$3: $2"
}

pinned_sha() {  # the KEPT pin of a step, or empty
  local line
  while IFS= read -r line; do
    if [ "${line%%=*}" = "$1" ]; then echo "${line#*=}"; return 0; fi
  done <<< "$KEPT"
  echo ""
}

after_step() {  # after_step <step>: honour --stop-after
  if [ "$1" = "$STOP_AFTER" ]; then
    echo "Stage T paused after $1 (--stop-after); Stage T is not complete"
    exit 0
  fi
}

before_start() {  # before_start <step> <folder>: the pause file, then the dry run
  if [ -e "$PAUSE_FILE" ]; then
    echo "Stage T paused before $1: pause file $PAUSE_FILE exists; nothing started"
    exit 0
  fi
  if [ "$DRY" = 1 ]; then
    echo "dry run: $1: would run -> $2"
    return 1
  fi
  return 0
}

# resolve <step> <required outcome>: sets KEEP (a kept report) or NEXT (the new attempt's folder
# in outputs/), or stops.
resolve() {
  local step=$1 want=$2 d f n o maxn=0 ndone=0 nvoid=0 nother=0 seen=""
  KEEP="" NEXT=""
  for d in "${SEARCH[@]}"; do
    for f in "$d"/task077-t-"$step"-*; do
      [ -d "$f" ] || continue
      n=${f##*-}
      case "$n" in '' | *[!0-9]*) continue ;; esac
      [ "$n" -gt "$maxn" ] && maxn=$n
      o=$(outcome "$f/report.json")
      seen="$seen $f (${o:-no report})"
      if [ "$o" = "$want" ]; then
        ndone=$((ndone + 1)); KEEP=$f/report.json
      elif [ "$o" = V ]; then
        nvoid=$((nvoid + 1))
      else
        nother=$((nother + 1))
      fi
    done
  done
  if [ "$RESUME" = 0 ] && [ -n "$seen" ]; then
    die "$step has an earlier attempt:$seen; use --resume"
  fi
  [ "$ndone" -le 1 ] || die "$step has more than one completed attempt:$seen; the operator decides"
  if [ "$ndone" = 1 ]; then
    # A --repeat left on the command line after the repeat completed (a resume) is harmless.
    return 0
  fi
  [ "$nother" = 0 ] || die "$step has an earlier attempt with no recognised outcome:$seen; the operator decides"
  [ "$nvoid" -lt 2 ] \
    || die "$step was voided twice:$seen. TASK-077 ends INCONCLUSIVE (protocol §10.1); nothing more runs"
  if [ "$nvoid" = 1 ]; then
    repeats "$step" || die "$step was voided:$seen. Its one repeat needs its recorded cause, its fix or prevention and a GO, then --repeat $step"
    n=$((maxn + 1))
  else
    repeats "$step" && die "--repeat $step, but $step has no V"
    n=$SUFFIX
  fi
  NEXT=$OUT/task077-t-$step-$n
  [ ! -e "$NEXT" ] || die "$NEXT already exists"
  return 0
}

keep() {  # keep <step> <report>: the kept report's checks
  local step=$1 rep=$2 why pin got
  if [ "$step" != plan ]; then
    why=$(pyq checkpoint "$rep")
    [ "$why" = ok ] || die "the kept $step is not intact: $why ($rep)"
  fi
  pin=$(pinned_sha "$step")
  got=$(pyq sha256 "$rep")
  if [ -n "$pin" ] && [ "$got" != "$pin" ]; then
    die "the kept $step report's sha256 $got differs from its recorded $pin ($rep)"
  fi
  echo "$step: kept $rep (sha256 $got)"
}

LAST=""  # the report of the job gpu_job just resolved (kept or completed; empty in a dry run)
gpu_job() {  # gpu_job <job> [extra runner args...]
  local job=$1
  shift
  resolve "$job" T-JOB-DONE
  LAST=""
  if [ -n "$KEEP" ]; then
    keep "$job" "$KEEP"
    LAST=$KEEP
  elif before_start "$job" "$NEXT"; then
    "$GPU_RUN" --wait --min-free-gib 8 --board --who "oej:task077-staget-$job" -- \
      "${PY[@]}" "$RUNNER" train --job "$job" "$@" \
      --output "$NEXT" --tests-record "$T" \
      --corpus "$C" --corpus-sha256 "$CS" --features "$F" --fits "$R" \
      --log "$NEXT.log" < /dev/null
    require "$NEXT/report.json" T-JOB-DONE "$job"
    LAST=$NEXT/report.json
  fi
  after_step "$job"
}

# Before anything starts: resolve every job and the plan, so that a V without --repeat, a
# second V, a --repeat of a job with no V or a kept job that is not intact stops the script
# before the G-tests record or any job runs.
for d in "${SEARCH[@]}"; do
  for f in "$d"/task077-t-plan-*; do
    if [ "$RESUME" = 1 ] && [ -d "$f" ] && [ "$(outcome "$f/report.json")" = CAL-T-ESCALATE ]; then
      die "plan ended CAL-T-ESCALATE ($f): Stage T stops and escalates"
    fi
  done
done
for step in $JOBS plan; do
  if [ "$step" = plan ]; then resolve plan T-PLANNED; else resolve "$step" T-JOB-DONE; fi
  if [ -n "$KEEP" ]; then keep "$step" "$KEEP" > /dev/null; fi
done

# 0. the G-tests record at HEAD (kept only from this worktree's outputs/, and only at HEAD)
T="" maxn=0
for f in "$OUT"/task077-t-tests-*; do
  [ -d "$f" ] || continue
  [ "$RESUME" = 1 ] || die "tests has an earlier attempt ($f); use --resume"
  n=${f##*-}
  case "$n" in '' | *[!0-9]*) continue ;; esac
  [ "$n" -gt "$maxn" ] && maxn=$n
  if [ "$(outcome "$f/report.json")" = TESTS-PASS ] && [ -n "$REV" ] \
    && [ "$(pyq revision "$f/report.json")" = "$REV" ]; then
    T=$f/report.json
  fi
done
if [ -n "$T" ]; then
  echo "tests: kept $T (TESTS-PASS at HEAD $REV)"
else
  n=$SUFFIX
  [ "$maxn" -gt 0 ] && n=$((maxn + 1))
  TD=$OUT/task077-t-tests-$n
  T=$TD/report.json
  if before_start tests "$TD"; then
    "${PY[@]}" "$RUNNER" tests --output "$TD" --log "$TD.log"
    require "$T" TESTS-PASS "tests"
  fi
fi
after_step tests

# 1-2. the calibration jobs
gpu_job cal-W
CAL_W=$LAST
gpu_job cal-N
CAL_N=$LAST

# 3. the plan: the budget, G1's bars, CAL-T-ESCALATE (an earlier CAL-T-ESCALATE stopped above)
resolve plan T-PLANNED
if [ -n "$KEEP" ]; then
  keep plan "$KEEP"
  if [ -n "$CAL_W" ] && [ -n "$CAL_N" ]; then
    [ "$(pyq input "$KEEP" --cal-w)" = "$(pyq realpath "$CAL_W")" ] \
      || die "the kept plan ($KEEP) was not made from the kept cal-W ($CAL_W)"
    [ "$(pyq input "$KEEP" --cal-n)" = "$(pyq realpath "$CAL_N")" ] \
      || die "the kept plan ($KEEP) was not made from the kept cal-N ($CAL_N)"
  elif [ "$DRY" = 0 ]; then
    die "a plan is kept but a calibration job is not"
  fi
  PLAN=$KEEP
else
  PLAN=$NEXT/report.json
  if before_start plan "$NEXT"; then
    "${PY[@]}" "$RUNNER" plan --output "$NEXT" \
      --corpus "$C" --corpus-sha256 "$CS" --features "$F" --fits "$R" \
      --cal-w "$CAL_W" --cal-n "$CAL_N" \
      --log "$NEXT.log"
    require "$PLAN" T-PLANNED "plan"
  fi
fi
after_step plan

# 4-9. the six models
for job in W-66800 N-66800 W-66801 N-66801 W-66802 N-66802; do
  gpu_job "$job" --plan "$PLAN"
done
if [ "$DRY" = 1 ]; then
  echo "dry run: nothing was started"
  exit 0
fi
echo "Stage T: all eight jobs ended T-JOB-DONE; T-DONE is decided by Stage G (R17.24)"
