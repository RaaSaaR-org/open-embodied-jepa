"""The Stage T driver (``scripts/run_task077_staget.sh``, protocol §7.6, R17.45): a dry run with a
fake runner and a fake ``gpu_run.sh`` checks that every step's outcome gates the next one."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_task077_staget.sh"

FAKE_RUNNER = """
import json, os, sys
from pathlib import Path
args = sys.argv[2:]
stage = args[0]
out = Path(args[args.index("--output") + 1])
job = args[args.index("--job") + 1] if "--job" in args else None
key = job or stage
with open(os.environ["FAKE_CALLS"], "a") as f:
    f.write(json.dumps({"stage": stage, "job": job, "argv": args}) + "\\n")
outcomes = json.loads(os.environ["FAKE_OUTCOMES"])
outcome = outcomes.get(key, outcomes.get(stage))
out.mkdir(parents=True)
(out / "report.json").write_text(json.dumps({"outcome": outcome}))
sys.exit(1 if outcome in (None, "V") else 0)
"""

FAKE_GPU_RUN = """#!/usr/bin/env bash
set -euo pipefail
echo "gpu_run $*" >> "$FAKE_GPU_CALLS"
while [ "$1" != "--" ]; do shift; done
shift
exec "$@"
"""

GOOD = {"tests": "TESTS-PASS", "train": "T-JOB-DONE", "plan": "T-PLANNED"}


def _run(tmp_path, outcomes):
    if shutil.which("bash") is None:
        pytest.skip("bash unavailable")
    (tmp_path / "fake_runner.py").write_text(FAKE_RUNNER)
    gpu = tmp_path / "fake_gpu_run.sh"
    gpu.write_text(FAKE_GPU_RUN)
    gpu.chmod(0o755)
    calls, gpu_calls = tmp_path / "calls.jsonl", tmp_path / "gpu_calls.txt"
    env = dict(os.environ) | {
        "PY": f"{sys.executable} {tmp_path / 'fake_runner.py'}",
        "GPU_RUN": str(gpu),
        "FAKE_OUTCOMES": json.dumps(outcomes),
        "FAKE_CALLS": str(calls),
        "FAKE_GPU_CALLS": str(gpu_calls),
    }
    work = tmp_path / "work"
    work.mkdir()
    proc = subprocess.run(["bash", str(SCRIPT)], cwd=work, env=env, capture_output=True, text=True)
    made = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    gpu_lines = gpu_calls.read_text().splitlines() if gpu_calls.exists() else []
    return proc, made, gpu_lines


def test_the_script_parses():
    if shutil.which("bash") is None:
        pytest.skip("bash unavailable")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_all_eight_jobs_run_in_order_through_gpu_run_with_board(tmp_path):
    proc, calls, gpu = _run(tmp_path, GOOD)
    assert proc.returncode == 0, proc.stderr
    order = [c["job"] or c["stage"] for c in calls]
    assert order == [
        "tests", "cal-W", "cal-N", "plan",
        "W-66800", "N-66800", "W-66801", "N-66801", "W-66802", "N-66802",
    ]  # fmt: skip
    assert len(gpu) == 8
    for line in gpu:
        assert "--wait --min-free-gib 8 --board --who oej:task077-staget-" in line
    for c in calls:
        if c["stage"] == "train":
            argv = c["argv"]
            for flag in ("--tests-record", "--corpus", "--corpus-sha256", "--features", "--fits"):
                assert flag in argv
            assert ("--plan" in argv) == (not c["job"].startswith("cal-"))
            assert argv[argv.index("--corpus-sha256") + 1] == (
                "ad8974b2a8b560bb974c6e0b4f90bd3f1fc79a535ebe46ef6c409bde7e4343fb"
            )


def test_cal_t_escalate_stops_before_any_model_job(tmp_path):
    """plan exits 0 on CAL-T-ESCALATE; the script must still stop (the #147 review)."""
    proc, calls, gpu = _run(tmp_path, GOOD | {"plan": "CAL-T-ESCALATE"})
    assert proc.returncode != 0
    assert [c["job"] or c["stage"] for c in calls] == ["tests", "cal-W", "cal-N", "plan"]
    assert len(gpu) == 2 and "CAL-T-ESCALATE" in proc.stderr


@pytest.mark.parametrize(
    ("outcomes", "ran"),
    [
        (GOOD | {"tests": "V"}, ["tests"]),
        (GOOD | {"tests": "TESTS-FAIL"}, ["tests"]),
        (GOOD | {"cal-W": "V"}, ["tests", "cal-W"]),
        (GOOD | {"cal-N": "SOMETHING-ELSE"}, ["tests", "cal-W", "cal-N"]),
        (GOOD | {"W-66801": "V"}, ["tests", "cal-W", "cal-N", "plan", "W-66800", "N-66800",
                                   "W-66801"]),
    ],
)  # fmt: skip
def test_any_other_outcome_stops_the_chain(tmp_path, outcomes, ran):
    proc, calls, _gpu = _run(tmp_path, outcomes)
    assert proc.returncode != 0
    assert [c["job"] or c["stage"] for c in calls] == ran
