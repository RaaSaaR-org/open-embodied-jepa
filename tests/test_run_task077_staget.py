"""The Stage T driver (``scripts/run_task077_staget.sh``, protocol §7.6-§7.7, R17.45, R17.47): dry
runs with a fake runner and a fake ``gpu_run.sh`` check that every step's outcome gates the next
one, and that a resume keeps completed jobs, repeats a V job once in a new folder, stops on a
second V, and pauses cleanly between jobs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_task077_staget.sh"
REV = "f" * 40

FAKE_RUNNER = """
import hashlib, json, os, sys
from pathlib import Path
args = sys.argv[2:]
stage = args[0]
out = Path(args[args.index("--output") + 1])
job = args[args.index("--job") + 1] if "--job" in args else None
key = job or stage
with open(os.environ["FAKE_CALLS"], "a") as f:
    f.write(json.dumps({"stage": stage, "job": job, "argv": args, "cwd": os.getcwd()}) + "\\n")
outcomes = json.loads(os.environ["FAKE_OUTCOMES"])
outcome = outcomes.get(key, outcomes.get(stage))
out.mkdir(parents=True)
report = {"outcome": outcome, "argv": args, "revision": os.environ["REV"]}
if stage == "train" and outcome == "T-JOB-DONE":
    ck = out / f"{job}.pt"
    ck.write_bytes(job.encode())
    report["fields"] = {"record": {"checkpoint": str(ck),
                                   "checkpoint_sha256": hashlib.sha256(job.encode()).hexdigest()}}
(out / "report.json").write_text(json.dumps(report))
if os.environ.get("FAKE_PAUSE_AFTER") == key:  # the operator touches the pause file mid-job
    Path(os.environ["PAUSE_FILE"]).touch()
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
EIGHT = ["W-66800", "N-66800", "W-66801", "N-66801", "W-66802", "N-66802"]
ALL = ["tests", "cal-W", "cal-N", "plan", *EIGHT]


def _run(tmp_path, outcomes, *args, work="work", env_extra=None):
    """Run the driver in ``tmp_path/work`` (made if missing); return the process, the runner calls
    of this invocation and its gpu_run lines."""
    if shutil.which("bash") is None:
        pytest.skip("bash unavailable")
    (tmp_path / "fake_runner.py").write_text(FAKE_RUNNER)
    gpu = tmp_path / "fake_gpu_run.sh"
    gpu.write_text(FAKE_GPU_RUN)
    gpu.chmod(0o755)
    calls, gpu_calls = tmp_path / "calls.jsonl", tmp_path / "gpu_calls.txt"
    for p in (calls, gpu_calls):
        p.unlink(missing_ok=True)
    wd = tmp_path / work
    wd.mkdir(exist_ok=True)
    env = dict(os.environ) | {
        "PY": f"{sys.executable} {tmp_path / 'fake_runner.py'}",
        "GPU_RUN": str(gpu),
        "FAKE_OUTCOMES": json.dumps(outcomes),
        "FAKE_CALLS": str(calls),
        "FAKE_GPU_CALLS": str(gpu_calls),
        "REV": REV,
        "KEPT": "",
        "PAUSE_FILE": str(wd / "outputs" / "task077-staget.pause"),
    }
    env.pop("FAKE_PAUSE_AFTER", None)
    env |= env_extra or {}
    proc = subprocess.run(
        ["bash", str(SCRIPT), *args], cwd=wd, env=env, capture_output=True, text=True
    )
    made = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    gpu_lines = gpu_calls.read_text().splitlines() if gpu_calls.exists() else []
    return proc, made, gpu_lines


def _order(calls):
    return [c["job"] or c["stage"] for c in calls]


def _first_run_paused_by_a_v(tmp_path):
    """Stage T's first run as it happened: four steps complete, then N-66800 ends V."""
    proc, calls, _ = _run(tmp_path, GOOD | {"N-66800": "V"}, work="first")
    assert proc.returncode != 0 and _order(calls) == ALL[:6]
    return tmp_path / "first" / "outputs"


def test_the_script_parses():
    if shutil.which("bash") is None:
        pytest.skip("bash unavailable")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_all_eight_jobs_run_in_order_through_gpu_run_with_board(tmp_path):
    proc, calls, gpu = _run(tmp_path, GOOD)
    assert proc.returncode == 0, proc.stderr
    assert _order(calls) == ALL
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
            assert argv[argv.index("--output") + 1] == f"outputs/task077-t-{c['job']}-1"


def test_cal_t_escalate_stops_before_any_model_job(tmp_path):
    """plan exits 0 on CAL-T-ESCALATE; the script must still stop (the #147 review)."""
    proc, calls, gpu = _run(tmp_path, GOOD | {"plan": "CAL-T-ESCALATE"})
    assert proc.returncode != 0
    assert _order(calls) == ["tests", "cal-W", "cal-N", "plan"]
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
    assert _order(calls) == ran


# ----- R17.47: resume, the one repeat, pause ------------------------------------------------------
def test_without_resume_an_earlier_attempt_stops_the_script(tmp_path):
    proc, _calls, _gpu = _run(tmp_path, GOOD | {"N-66800": "V"})
    assert proc.returncode != 0
    proc, calls, gpu = _run(tmp_path, GOOD)
    assert proc.returncode != 0 and calls == [] and gpu == []
    assert "use --resume" in proc.stderr


def test_resume_keeps_completed_jobs_and_repeats_the_v_job_in_a_new_folder(tmp_path):
    first = _first_run_paused_by_a_v(tmp_path)
    proc, calls, gpu = _run(tmp_path, GOOD, "--resume-from", str(first), "--repeat", "N-66800")
    assert proc.returncode == 0, proc.stderr
    # the new worktree runs its own G-tests record (G-tests checks HEAD), then only the jobs
    # that are not complete; cal-W, cal-N, plan and W-66800 are kept, not re-run
    assert _order(calls) == ["tests", *EIGHT[1:]]
    assert len(gpu) == 5
    for step in ("cal-W", "cal-N", "plan", "W-66800"):
        assert f"{step}: kept {first}/task077-t-{step}-1/report.json" in proc.stdout
    repeat = calls[1]
    assert repeat["job"] == "N-66800"
    out = repeat["argv"][repeat["argv"].index("--output") + 1]
    assert out == "outputs/task077-t-N-66800-2"  # never the V's folder
    assert (first / "task077-t-N-66800-1" / "report.json").exists()  # the V is kept
    for c in calls[1:]:
        argv = c["argv"]
        assert argv[argv.index("--plan") + 1] == f"{first}/task077-t-plan-1/report.json"
        assert argv[argv.index("--tests-record") + 1] == "outputs/task077-t-tests-1/report.json"
    for job in EIGHT[2:]:
        assert f"outputs/task077-t-{job}-1" in " ".join(gpu)


def test_a_v_without_repeat_stops_before_the_job(tmp_path):
    first = _first_run_paused_by_a_v(tmp_path)
    proc, calls, gpu = _run(tmp_path, GOOD, "--resume-from", str(first))
    assert proc.returncode != 0
    assert calls == [] and gpu == []  # stopped before the G-tests record, too
    assert "N-66800 was voided" in proc.stderr and "--repeat N-66800" in proc.stderr


def test_a_second_v_of_the_same_job_ends_inconclusive_and_nothing_more_runs(tmp_path):
    first = _first_run_paused_by_a_v(tmp_path)
    args = ("--resume-from", str(first), "--repeat", "N-66800")
    proc, calls, _gpu = _run(tmp_path, GOOD | {"N-66800": "V"}, *args)
    assert proc.returncode != 0 and _order(calls) == ["tests", "N-66800"]
    # a further resume, even with --repeat, refuses a third attempt
    proc, calls, gpu = _run(tmp_path, GOOD, *args)
    assert proc.returncode != 0 and calls == [] and gpu == []
    assert "voided twice" in proc.stderr and "INCONCLUSIVE" in proc.stderr


def test_repeat_of_a_job_with_no_v_is_refused(tmp_path):
    first = _first_run_paused_by_a_v(tmp_path)
    args = ("--resume-from", str(first), "--repeat", "N-66800", "--repeat", "W-66801")
    proc, calls, gpu = _run(tmp_path, GOOD, *args)
    assert proc.returncode != 0 and calls == [] and gpu == []
    assert "W-66801 has no V" in proc.stderr


def test_a_kept_job_whose_checkpoint_changed_stops_the_resume(tmp_path):
    first = _first_run_paused_by_a_v(tmp_path)
    (first / "task077-t-W-66800-1" / "W-66800.pt").write_bytes(b"changed")
    proc, calls, gpu = _run(tmp_path, GOOD, "--resume-from", str(first), "--repeat", "N-66800")
    assert proc.returncode != 0 and calls == [] and gpu == []
    assert "kept W-66800 is not intact" in proc.stderr


def test_a_kept_report_must_match_its_recorded_sha256(tmp_path):
    first = _first_run_paused_by_a_v(tmp_path)
    good = hashlib.sha256((first / "task077-t-cal-W-1" / "report.json").read_bytes()).hexdigest()
    args = ("--resume-from", str(first), "--repeat", "N-66800")
    proc, _calls, gpu = _run(tmp_path, GOOD, *args, env_extra={"KEPT": f"cal-W={'0' * 64}"})
    assert proc.returncode != 0 and gpu == [] and "differs from its recorded" in proc.stderr
    proc, _calls, gpu = _run(tmp_path, GOOD, *args, work="w2", env_extra={"KEPT": f"cal-W={good}"})
    assert proc.returncode == 0, proc.stderr


def test_the_default_pins_are_the_first_runs_reports():
    """The pins R17.46 records (Stage T's first run at 215fcce)."""
    text = SCRIPT.read_text()
    for step, sha in {
        "cal-W": "2339f5f1dc4447af6c04a7ea40e9d22708aec849bece080c57d5301a7e270cb8",
        "cal-N": "42ad6cf892f14c6338608d3d9b148eff34db8e797892a98ac5d46ce08e984423",
        "plan": "7c7147601c156a793c238e34d00ee74e9855f29d19e4ee0d653882335d67cdfb",
        "W-66800": "42195ded1ac1c41a77ed0b47e2031e4066f9bc2c2a61637ea5aa17f08e05757b",
    }.items():
        assert f"{step}={sha}" in text


def test_a_kept_plan_must_come_from_the_kept_calibrations(tmp_path):
    first = _first_run_paused_by_a_v(tmp_path)
    # a second, different cal-W folder elsewhere: the plan no longer matches what is kept
    other = tmp_path / "other" / "outputs"
    shutil.copytree(first / "task077-t-cal-W-1", other / "task077-t-cal-W-1")
    shutil.rmtree(first / "task077-t-cal-W-1")
    proc, _calls, gpu = _run(tmp_path, GOOD, "--resume-from", str(other), "--resume-from",
                             str(first), "--repeat", "N-66800")  # fmt: skip
    assert proc.returncode != 0 and gpu == []
    assert "not made from the kept cal-W" in proc.stderr


def test_a_cal_t_escalate_plan_stops_a_resume(tmp_path):
    proc, _calls, _gpu = _run(tmp_path, GOOD | {"plan": "CAL-T-ESCALATE"}, work="first")
    proc, calls, gpu = _run(tmp_path, GOOD, "--resume-from", str(tmp_path / "first" / "outputs"))
    assert proc.returncode != 0 and gpu == [] and calls == []
    assert "CAL-T-ESCALATE" in proc.stderr


def test_the_pause_file_stops_between_jobs_and_a_resume_continues(tmp_path):
    proc, calls, gpu = _run(tmp_path, GOOD, env_extra={"FAKE_PAUSE_AFTER": "W-66800"})
    assert proc.returncode == 0, proc.stderr
    assert _order(calls) == ALL[:5] and len(gpu) == 3  # W-66800 ran to its end; N-66800 did not
    assert "paused before N-66800" in proc.stdout and "all eight" not in proc.stdout
    pause = tmp_path / "work" / "outputs" / "task077-staget.pause"
    # with the pause file still there, a resume starts nothing
    proc, calls, gpu = _run(tmp_path, GOOD, "--resume")
    assert proc.returncode == 0 and calls == [] and gpu == []
    pause.unlink()
    proc, calls, gpu = _run(tmp_path, GOOD, "--resume")
    assert proc.returncode == 0, proc.stderr
    assert _order(calls) == EIGHT[1:]  # the G-tests record at HEAD is kept in the same worktree
    assert "tests: kept outputs/task077-t-tests-1/report.json" in proc.stdout
    assert "all eight" in proc.stdout


def test_a_tests_record_at_another_revision_is_not_kept(tmp_path):
    _run(tmp_path, GOOD, "--stop-after", "W-66800")
    proc, calls, _gpu = _run(tmp_path, GOOD, "--resume", env_extra={"REV": "e" * 40})
    assert proc.returncode == 0, proc.stderr
    assert _order(calls)[0] == "tests"
    assert calls[0]["argv"][calls[0]["argv"].index("--output") + 1] == "outputs/task077-t-tests-2"


def test_stop_after_ends_cleanly_after_that_job(tmp_path):
    proc, calls, gpu = _run(tmp_path, GOOD, "--stop-after", "W-66800")
    assert proc.returncode == 0, proc.stderr
    assert _order(calls) == ALL[:5] and len(gpu) == 3
    assert "paused after W-66800" in proc.stdout


def test_dry_run_starts_nothing(tmp_path):
    first = _first_run_paused_by_a_v(tmp_path)
    args = ("--resume-from", str(first), "--repeat", "N-66800", "--dry-run")
    proc, calls, gpu = _run(tmp_path, GOOD, *args)
    assert proc.returncode == 0, proc.stderr
    assert calls == [] and gpu == []
    assert "N-66800: would run -> outputs/task077-t-N-66800-2" in proc.stdout
    assert not (tmp_path / "work" / "outputs").exists()


@pytest.mark.parametrize(
    "args",
    [("--repeat", "N-66800"), ("--repeat", "tests"), ("--stop-after", "nope"), ("--bogus",),
     ("--resume-from", "/nonexistent/folder")],
)  # fmt: skip
def test_bad_arguments_are_refused_before_anything_runs(tmp_path, args):
    proc, calls, gpu = _run(tmp_path, GOOD, *args)
    assert proc.returncode != 0 and calls == [] and gpu == []


def test_a_pinned_step_that_is_not_found_stops_instead_of_running_afresh(tmp_path):
    """With the first run's pins, neither a plain run nor --resume without --resume-from may
    start Stage T again from scratch (the #148 review)."""
    pins = {"KEPT": f"cal-W={'0' * 64}"}
    for args in ((), ("--resume",)):
        proc, calls, gpu = _run(tmp_path, GOOD, *args, env_extra=pins)
        assert proc.returncode != 0 and calls == [] and gpu == []
        assert "cal-W is complete (pinned in KEPT) but was not found" in proc.stderr


def test_a_folder_named_twice_is_searched_once(tmp_path):
    first = _first_run_paused_by_a_v(tmp_path)
    args = ("--resume-from", str(first), "--resume-from", f"{first}/.", "--repeat", "N-66800")
    proc, calls, _gpu = _run(tmp_path, GOOD, *args, "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert calls == [] and "N-66800: would run -> outputs/task077-t-N-66800-2" in proc.stdout
