"""Host tests for the worktree and GPU-lock shell tools (the 2026-10-02 audit, F5 and F13).

No GPU, no network and never the real ``~/.local/state/gpu``: ``gpu_run.sh`` runs against a
temporary state directory and a stand-in ``nvidia-smi``; the worktree scripts run on throwaway
git repositories. Where ``flock(1)`` is missing (macOS), a stand-in built on ``fcntl.flock``
locks the inherited descriptor exactly as util-linux ``flock <fd>`` does."""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GPU_RUN = ROOT / "scripts" / "gpu_run.sh"
BASH = shutil.which("bash") or "/bin/bash"

FAKE_SMI = """#!/bin/sh
# stand-in nvidia-smi: FAKE_FREE / FAKE_UTIL in MiB / %
case "$*" in
  *query-compute-apps*) echo "4242, python, 1024 MiB" ;;
  *) echo "${FAKE_FREE:-12000}, 16303, ${FAKE_UTIL:-3}" ;;
esac
"""

FLOCK_SHIM = f"""#!{sys.executable}
import fcntl, sys, time
args = sys.argv[1:]
nb, wait = False, None
while args[0].startswith("-"):
    flag = args.pop(0)
    if flag == "-n":
        nb = True
    elif flag == "-w":
        wait = float(args.pop(0))
fd = int(args[0])
deadline = None if wait is None else time.monotonic() + wait
while True:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.exit(0)
    except BlockingIOError:
        if nb or (deadline is not None and time.monotonic() > deadline):
            sys.exit(1)
        time.sleep(0.05)
"""

BOARD = """# GPU board (test)

## Holder
- holder: someone else
- job: theirs

## Requests
- other: keep this line
"""


TAG_VAR = "OEJ_GPU_RUN_TEST_TAG"
# a command's busy loop ends by itself after 120 s, even if every cleanup below failed
BOUNDED_LOOP = "while [ $SECONDS -lt 120 ]; do sleep 0.05; done"


def _zombie_or_gone(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return True


def _ps_tokens(*args: str) -> list[tuple[int, str, list[str]]]:
    """(pid, state, whitespace-separated command and environment tokens) from ``ps -E``."""
    out = subprocess.run(
        ["ps", "-ww", "-E", "-o", "pid=,stat=,command=", *args], capture_output=True, text=True
    )
    rows = []
    for line in out.stdout.splitlines():
        pid, stat, rest = (line.split(None, 2) + ["", ""])[:3]
        if pid.isdigit():
            rows.append((int(pid), stat, rest.split()))
    return rows


def _has_proc_environ() -> bool:
    return Path("/proc/self/environ").exists()


def _is_tagged(pid: int, tag: str) -> bool:
    """Does live process ``pid`` carry exactly ``OEJ_GPU_RUN_TEST_TAG=<tag>`` right now?

    On Linux this compares whole entries of its initial environment; elsewhere whole
    whitespace-separated tokens of ``ps -E`` (environment and command line, best effort)."""
    needle = f"{TAG_VAR}={tag}"
    if _has_proc_environ():
        try:
            env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        except OSError:
            return False
        return needle.encode() in env and not _zombie_or_gone(pid)
    return any(
        p == pid and not stat.startswith("Z") and needle in tokens
        for p, stat, tokens in _ps_tokens("-p", str(pid))
    )


def _tagged_pids(tag: str) -> set[int]:
    """Live processes started from one test's environment: every process gpu_run.sh starts
    inherits ``OEJ_GPU_RUN_TEST_TAG=<the test's tmp dir>``."""
    me = os.getpid()
    if _has_proc_environ():
        pids = (int(e.name) for e in Path("/proc").iterdir() if e.name.isdigit())
        return {pid for pid in pids if pid != me and _is_tagged(pid, tag)}
    needle = f"{TAG_VAR}={tag}"
    return {
        pid
        for pid, stat, tokens in _ps_tokens("-ax")
        if pid != me and not stat.startswith("Z") and needle in tokens
    }


def _describe(pids: set[int]) -> str:
    if not pids:
        return ""
    out = subprocess.run(
        ["ps", "-o", "pid,ppid,pgid,stat,command", "-p", ",".join(map(str, sorted(pids)))],
        capture_output=True,
        text=True,
    )
    return out.stdout


def _kill_tagged(pid: int, tag: str, *, ours: bool = False) -> None:
    """SIGKILL ``pid`` if it is tagged right now (or is an unreaped child of ours), and its process
    group only if that group's leader is tagged right now. Never pytest's own group, and never
    a pid that has since been reused by somebody else's process."""
    if not (ours or _is_tagged(pid, tag)):
        return
    with contextlib.suppress(OSError):
        pgid = os.getpgid(pid)
        if pgid != os.getpgrp() and _is_tagged(pgid, tag):
            os.killpg(pgid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGKILL)


@pytest.fixture
def gpu_env(tmp_path):
    """A temporary GPU state directory, a stand-in nvidia-smi and an unconditional cleanup.

    Every process a test starts carries the tag ``OEJ_GPU_RUN_TEST_TAG=<tmp_path>`` in its
    environment. Teardown runs even when the test fails or times out: it waits briefly for the
    tagged processes to end, then SIGKILLs every unreaped tracked ``Popen``, every pid recorded in
    a file listed in ``env["pidfiles"]`` (such as the stubborn child's) and every survivor of a
    fresh scan, each only if it is tagged at that moment, with its process group when that
    group's leader is tagged too; it fails the test if anything from its temp dir outlived it.
    The real ``~/.local/state/gpu`` is never touched."""
    state = tmp_path / "state"
    state.mkdir()
    (state / "BOARD.md").write_text(BOARD)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    smi = bin_dir / "nvidia-smi"
    smi.write_text(FAKE_SMI)
    smi.chmod(0o755)
    path = os.environ["PATH"]
    if shutil.which("flock") is None:
        shim = bin_dir / "flock"
        shim.write_text(FLOCK_SHIM)
        shim.chmod(0o755)
    tag = str(tmp_path)
    env = os.environ | {
        "GPU_STATE_DIR": str(state),
        "NVIDIA_SMI": str(smi),
        "PATH": f"{bin_dir}{os.pathsep}{path}",
        TAG_VAR: tag,
    }
    env.pop("GPU_RUN_LOCKED", None)
    real_state = Path.home() / ".local" / "state" / "gpu"
    assert Path(env["GPU_STATE_DIR"]).resolve() != real_state.resolve()
    procs: list[subprocess.Popen] = []
    pidfiles: list[Path] = []
    try:
        yield {"state": state, "env": env, "tmp": tmp_path, "procs": procs, "pidfiles": pidfiles}
    finally:
        deadline = time.monotonic() + 5
        while (leaked := _tagged_pids(tag)) and time.monotonic() < deadline:
            time.sleep(0.1)
        described = _describe(leaked)
        if leaked and (state / "oej-gpu_run.log").exists():
            described += "gpu_run log:\n" + (state / "oej-gpu_run.log").read_text()
        for proc in procs:
            if proc.poll() is None:  # not reaped yet, so the pid is still this process's
                leaked.add(proc.pid)
                _kill_tagged(proc.pid, tag, ours=True)
        for pidfile in pidfiles:  # the tag is checked again right before each kill
            with contextlib.suppress(OSError, ValueError):
                _kill_tagged(int(pidfile.read_text()), tag)
        for pid in _tagged_pids(tag):  # a fresh scan, never the earlier snapshot
            _kill_tagged(pid, tag)
        for proc in procs:
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
        deadline = time.monotonic() + 10
        while (left := _tagged_pids(tag)) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not left, f"processes from {tmp_path} survive even SIGKILL: {sorted(left)}"
        assert not leaked, f"processes from {tmp_path} outlived the test:\n{described}"


def _spawn(gpu_env, *args, extra_env=None, **kw) -> subprocess.Popen:
    """Start gpu_run.sh in the background; the fixture's teardown kills it whatever happens."""
    proc = subprocess.Popen(
        [BASH, str(GPU_RUN), *args],
        env=gpu_env["env"] | (extra_env or {}),
        cwd=gpu_env["tmp"],
        **kw,
    )
    gpu_env["procs"].append(proc)
    return proc


def _gpu_run(gpu_env, *args, extra_env=None, **kw):
    env = gpu_env["env"] | (extra_env or {})
    return subprocess.run(
        [BASH, str(GPU_RUN), "--samples", "1", *args],
        env=env,
        cwd=gpu_env["tmp"],
        capture_output=True,
        text=True,
        timeout=60,
        **kw,
    )


def _hold_lock(gpu_env):
    """A process holding the test lock until it is killed."""
    code = (
        "import fcntl, sys, time; f = open(sys.argv[1], 'a'); "
        "fcntl.flock(f, fcntl.LOCK_EX); print('held', flush=True); time.sleep(60)"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", code, str(gpu_env["state"] / "lock")],
        stdout=subprocess.PIPE,
        text=True,
        env=gpu_env["env"],
    )
    gpu_env["procs"].append(holder)
    assert holder.stdout.readline().strip() == "held"
    return holder


def test_gpu_run_runs_the_command_inside_the_lock_and_logs(gpu_env):
    out = _gpu_run(gpu_env, "--", "sh", "-c", 'echo "locked=$GPU_RUN_LOCKED"; exit 7')
    assert out.returncode == 7, out.stderr
    assert "locked=1" in out.stdout
    log = (gpu_env["state"] / "oej-gpu_run.log").read_text()
    assert "start:" in log and "free 12000 MiB" in log and "4242, python" in log
    assert "status 7" in log
    assert (gpu_env["state"] / "BOARD.md").read_text() == BOARD  # untouched without flags


def test_gpu_run_refuses_too_little_free_memory_or_a_busy_gpu(gpu_env):
    out = _gpu_run(gpu_env, "--min-free-gib", "14", "--", "true", extra_env={"FAKE_FREE": "8000"})
    assert out.returncode == 76 and "free 8000 MiB < 14 GiB" in out.stderr
    out = _gpu_run(gpu_env, "--max-util", "30", "--", "true", extra_env={"FAKE_UTIL": "95"})
    assert out.returncode == 76 and "utilisation 95 % > 30 %" in out.stderr
    out = _gpu_run(gpu_env, "--max-load", "0", "--", "true")
    if Path("/proc/loadavg").exists() and float(Path("/proc/loadavg").read_text().split()[0]):
        assert out.returncode == 76 and "load" in out.stderr
    assert (gpu_env["state"] / "BOARD.md").read_text() == BOARD


def test_gpu_run_fails_at_once_while_another_job_holds_the_lock(gpu_env):
    holder = _hold_lock(gpu_env)
    try:
        out = _gpu_run(gpu_env, "--", "true")
        assert out.returncode == 75 and "busy" in out.stderr
        out = _gpu_run(gpu_env, "--wait", "--wait-timeout", "0.5", "--", "true")
        assert out.returncode == 75
    finally:
        holder.kill()
        holder.wait()
    assert (gpu_env["state"] / "BOARD.md").read_text() == BOARD
    assert _gpu_run(gpu_env, "--", "true").returncode == 0


def test_a_request_line_is_added_only_while_waiting_and_then_removed(gpu_env):
    board = gpu_env["state"] / "BOARD.md"
    holder = _hold_lock(gpu_env)
    proc = _spawn(
        gpu_env,
        *["--samples", "1", "--wait", "--who", "tester"],
        *["--request", "a short test, 1 s", "--", "true"],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while "tester: a short test" not in board.read_text():
            assert time.monotonic() < deadline, "the request line never appeared"
            time.sleep(0.05)
        lines = board.read_text().splitlines()
        mine = next(i for i, line in enumerate(lines) if "tester: a short test" in line)
        assert lines[mine - 1] == "## Requests" and "- other: keep this line" in lines
    finally:
        holder.kill()
        holder.wait()
    assert proc.wait(timeout=30) == 0, proc.stderr.read()
    assert board.read_text() == BOARD


def test_board_holder_is_written_only_with_the_flag(gpu_env):
    board = gpu_env["state"] / "BOARD.md"
    out = _gpu_run(
        gpu_env, "--board", "--who", "tester", "--", "sh", "-c", f'cat "{board}"', check=True
    )
    assert "- holder: tester" in out.stdout and "someone else" not in out.stdout
    after = board.read_text()
    assert "- holder: none (released by tester" in after
    assert "- other: keep this line" in after and "someone else" not in after


def test_gpu_run_forwards_sigterm_to_the_command(gpu_env):
    marker = gpu_env["tmp"] / "got-term"
    script = f"trap 'touch \"{marker}\"; exit 143' TERM; {BOUNDED_LOOP}"
    proc = _spawn(gpu_env, "--samples", "1", "--", BASH, "-c", script)
    log = gpu_env["state"] / "oej-gpu_run.log"
    deadline = time.monotonic() + 20
    while not (log.exists() and "start:" in log.read_text()):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    time.sleep(0.3)
    proc.terminate()
    assert proc.wait(timeout=20) == 143
    assert marker.exists()
    assert "status 143" in log.read_text()


def _lock_free(gpu_env) -> bool:
    code = (
        "import fcntl, sys; f = open(sys.argv[1], 'a')\n"
        "try:\n fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)\nexcept BlockingIOError:\n"
        " sys.exit(1)"
    )
    return (
        subprocess.run([sys.executable, "-c", code, str(gpu_env["state"] / "lock")]).returncode == 0
    )


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie is gone for our purposes
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return True


def _wait_file(path: Path, seconds: float = 20) -> str:
    deadline = time.monotonic() + seconds
    while not (path.exists() and path.read_text().strip()):
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.05)
    return path.read_text()


def test_a_grandchild_that_outlives_its_parent_keeps_gpu_run_and_the_lock(gpu_env):
    """Review B1: ``sh -c 'sleep & echo'`` must not leave the lock held after gpu_run exits."""
    marker = gpu_env["tmp"] / "grandchild-done"
    started = time.monotonic()
    out = _gpu_run(gpu_env, "--", "sh", "-c", f"(sleep 1.5; touch '{marker}') & echo started")
    assert out.returncode == 0, out.stderr
    assert time.monotonic() - started >= 1.4 and marker.exists()  # waited for the grandchild
    assert _lock_free(gpu_env)  # released the moment gpu_run exited
    log = (gpu_env["state"] / "oej-gpu_run.log").read_text()
    assert log.index("end:") > log.index("start:")


RUN_ISAAC_SHAPE = """#!/usr/bin/env bash
# shaped like scripts/isaac/run_isaac.sh: background samplers, a foreground "docker run"
set -euo pipefail
sleep 120 & SAMPLE_APPS=$!
sleep 120 & SAMPLE_DEV=$!
trap 'kill "$SAMPLE_APPS" "$SAMPLE_DEV" 2>/dev/null || true' EXIT
echo "$SAMPLE_APPS $SAMPLE_DEV" > "$1/samplers"
bash -c 'echo $$ > "$0/client"; exec sleep 120' "$1"
"""


def test_sigterm_reaches_a_run_isaac_shaped_wrapper_and_its_foreground_child(gpu_env):
    wrapper = gpu_env["tmp"] / "fake_run_isaac.sh"
    wrapper.write_text(RUN_ISAAC_SHAPE)
    wrapper.chmod(0o755)
    proc = _spawn(gpu_env, "--samples", "1", "--", str(wrapper), str(gpu_env["tmp"]))
    client = int(_wait_file(gpu_env["tmp"] / "client"))
    samplers = [int(p) for p in _wait_file(gpu_env["tmp"] / "samplers").split()]
    assert not _lock_free(gpu_env)
    proc.terminate()
    assert proc.wait(timeout=30) != 0
    for pid in (client, *samplers):
        assert not _alive(pid), pid  # the whole group got the TERM, none is orphaned
    assert _lock_free(gpu_env)
    assert "end:" in (gpu_env["state"] / "oej-gpu_run.log").read_text()


def _start_stubborn(gpu_env, *options, **kw):
    """gpu_run.sh around a command that ignores SIGTERM; returns (gpu_run, the command's pid)."""
    pidfile = gpu_env["tmp"] / "stubborn"
    gpu_env["pidfiles"].append(pidfile)
    script = f"trap '' TERM; echo $$ > '{pidfile}'; {BOUNDED_LOOP}"
    proc = _spawn(gpu_env, "--samples", "1", *options, "--", BASH, "-c", script, **kw)
    return proc, int(_wait_file(pidfile))


def _wait_gone(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while _alive(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def test_a_child_that_ignores_sigterm_is_killed_after_the_grace(gpu_env):
    proc, pid = _start_stubborn(gpu_env, "--grace", "1")
    proc.terminate()
    assert proc.wait(timeout=30) != 0
    assert not _alive(pid) and _lock_free(gpu_env)
    assert "killed:" in (gpu_env["state"] / "oej-gpu_run.log").read_text()


def test_a_term_to_gpu_runs_whole_group_cannot_stop_the_grace_kill(gpu_env):
    """The 2026-10-02 leak: a TERM to gpu_run, then a TERM to the caller's whole process group (a
    tool timeout, Ctrl-C), killed the helper that was to SIGKILL the command after the grace, and
    gpu_run waited forever on a command that ignores TERM. The helper now has its own group."""
    proc, pid = _start_stubborn(gpu_env, "--grace", "1", start_new_session=True)
    proc.terminate()
    time.sleep(0.3)
    os.killpg(proc.pid, signal.SIGTERM)  # gpu_run's own group, as the caller's would be
    assert proc.wait(timeout=30) != 0
    assert not _alive(pid) and _lock_free(gpu_env)
    assert "killed:" in (gpu_env["state"] / "oej-gpu_run.log").read_text()


def test_the_watchdog_stops_the_command_when_gpu_run_is_sigkilled(gpu_env):
    proc, pid = _start_stubborn(gpu_env, "--grace", "1")
    proc.kill()
    proc.wait(timeout=10)
    assert _wait_gone(pid, 20), "the command outlived a SIGKILLed gpu_run"
    log = (gpu_env["state"] / "oej-gpu_run.log").read_text()
    assert "orphaned:" in log and "killed:" in log and "end:" not in log


FAKE_DOCKER = """#!/bin/sh
# stand-in docker: a container "runs" while $STATE/container exists; stop removes it.
# `ps --format` lists it with a regex-looking neighbour that a literal prefix must not match.
case "$1" in
  ps)
    if [ "$2" = "--format" ]; then
      echo "fff999 oejXisaac-other"
      [ -f "$FAKE_DOCKER_STATE/container" ] && echo "abc123 oej-isaac-run7,alias"
    fi
    exit 0 ;;
  stop) echo "$@" >> "$FAKE_DOCKER_STATE/stopped"; rm -f "$FAKE_DOCKER_STATE/container" ;;
esac
"""


def test_a_surviving_container_keeps_the_lock_and_is_stopped_after_a_signal(gpu_env):
    docker = gpu_env["tmp"] / "bin" / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(0o755)
    (gpu_env["tmp"] / "container").write_text("running")
    env = {"DOCKER": str(docker), "FAKE_DOCKER_STATE": str(gpu_env["tmp"])}
    proc = _spawn(
        gpu_env,
        *["--samples", "1", "--container", "oej-isaac-", "--", BASH, "-c", "sleep 120"],
        extra_env=env,
    )
    log = gpu_env["state"] / "oej-gpu_run.log"
    deadline = time.monotonic() + 20
    while not (log.exists() and "start:" in log.read_text()):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    proc.terminate()  # the client dies at once, the "container" does not
    assert proc.wait(timeout=30) != 0
    stopped = (gpu_env["tmp"] / "stopped").read_text()
    assert "stop -t 10 abc123" in stopped and "fff999" not in stopped
    assert _lock_free(gpu_env)
    text = log.read_text()
    assert text.index("stopping containers") < text.index("end:")


def test_the_watchdog_never_fires_when_it_cannot_read_its_parent(gpu_env):
    """Re-review of #130: a failing ``ps`` (no procps, a transient fork failure) must not read as
    "gpu_run is dead"; the watchdog fires only on a parent it has read and that is not gpu_run."""
    bad = gpu_env["tmp"] / "bad-ps"
    bad.mkdir()
    (bad / "ps").write_text("#!/bin/sh\nexit 1\n")
    (bad / "ps").chmod(0o755)
    path = f"{bad}{os.pathsep}{gpu_env['env']['PATH']}"
    out = _gpu_run(gpu_env, "--grace", "1", "--", BASH, "-c", "sleep 2.5", extra_env={"PATH": path})
    assert out.returncode == 0, out.stderr
    log = (gpu_env["state"] / "oej-gpu_run.log").read_text()
    assert "orphaned:" not in log and "status 0" in log


def test_a_signal_after_the_command_has_exited_starts_no_killer(gpu_env):
    """Review of #130, finding 2: while --container keeps gpu_run waiting after the command has
    exited, a TERM must stop the containers and let gpu_run exit 0 without starting a grace killer
    (whose sleep would outlive gpu_run and later probe a pgid that may have been reused)."""
    docker = gpu_env["tmp"] / "bin" / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(0o755)
    (gpu_env["tmp"] / "container").write_text("running")
    pidfile = gpu_env["tmp"] / "quick"
    env = {"DOCKER": str(docker), "FAKE_DOCKER_STATE": str(gpu_env["tmp"])}
    proc = _spawn(
        gpu_env,
        *["--samples", "1", "--grace", "30", "--container", "oej-isaac-", "--"],
        *[BASH, "-c", f"echo $$ > '{pidfile}'"],
        extra_env=env,
    )
    command = int(_wait_file(pidfile))
    assert _wait_gone(command, 20)
    time.sleep(0.5)  # gpu_run is now in its container wait
    assert proc.poll() is None
    proc.terminate()
    assert proc.wait(timeout=30) == 0
    tag = gpu_env["env"][TAG_VAR]
    deadline = time.monotonic() + 2
    while (left := _tagged_pids(tag)) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not left, _describe(left)  # no grace killer, no sleep, no watchdog
    log = (gpu_env["state"] / "oej-gpu_run.log").read_text()
    assert "stopping containers" in log and "end:" in log and "killed:" not in log


def test_concurrent_request_lines_are_not_lost(gpu_env):
    """Review N2: two waiters edit BOARD.md under BOARD.md.lock; both lines appear."""
    board = gpu_env["state"] / "BOARD.md"
    holder = _hold_lock(gpu_env)
    procs = [
        _spawn(
            gpu_env, "--samples", "1", "--wait", "--who", f"w{i}", "--request", "test", "--", "true"
        )
        for i in range(4)
    ]
    try:
        deadline = time.monotonic() + 20
        while sum(f"- w{i}: test" in board.read_text() for i in range(4)) < 4:
            assert time.monotonic() < deadline, board.read_text()
            time.sleep(0.05)
    finally:
        holder.kill()
        holder.wait()
    assert all(p.wait(timeout=60) == 0 for p in procs)
    assert board.read_text() == BOARD


@pytest.mark.parametrize("prefix", ["oej.isaac", "^oej", "oej isaac", "-oej", "oej*"])
def test_container_prefix_must_be_a_plain_name(gpu_env, prefix):
    out = _gpu_run(gpu_env, "--container", prefix, "--", "true")
    assert out.returncode == 2 and "plain name prefix" in out.stderr


def test_the_commands_status_survives_a_signal_racing_its_exit(gpu_env):
    """Review N8: HUP forwarded as TERM; the command exits 143 on TERM, so gpu_run reports 143,
    never the 129 of its own interrupted wait."""
    for attempt in range(5):
        ready = gpu_env["tmp"] / f"ready-{attempt}"
        script = f"trap 'exit 143' TERM; touch '{ready}'; {BOUNDED_LOOP}"
        proc = _spawn(gpu_env, "--samples", "1", "--", BASH, "-c", script)
        deadline = time.monotonic() + 20
        while not ready.exists():
            assert time.monotonic() < deadline
            time.sleep(0.02)
        proc.send_signal(signal.SIGHUP)
        assert proc.wait(timeout=30) == 143, attempt


def test_gpu_run_usage_errors(gpu_env):
    assert _gpu_run(gpu_env).returncode == 2
    assert _gpu_run(gpu_env, "--request", "x", "--", "true").returncode == 2
    assert _gpu_run(gpu_env, "--min-free-gib", "lots", "--", "true").returncode == 2


# ----- the worktree scripts ----------------------------------------------------------------------
def _git(cwd, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A throwaway origin + primary checkout carrying the two scripts, with a third_party/."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(origin))
    main = tmp_path / "main"
    _git(tmp_path, "clone", str(origin), str(main))
    _git(main, "checkout", "-B", "main")
    (main / "scripts").mkdir()
    for name in ("new_worktree.sh", "remove_worktree.sh"):
        shutil.copy2(ROOT / "scripts" / name, main / "scripts" / name)
    shutil.copy2(ROOT / ".gitignore", main / ".gitignore")  # the repository's real rules
    (main / "README").write_text("x\n")
    _git(main, "add", ".")
    _git(main, "commit", "-m", "init")
    _git(main, "push", "-u", "origin", "main")
    (main / "third_party" / "lewm").mkdir(parents=True)
    (main / "third_party" / "lewm" / "keep").write_text("pinned\n")
    (main / "data" / "corpus").mkdir(parents=True)
    (main / "assets" / "isaac").mkdir(parents=True)
    (main / "assets" / "isaac" / "g1.usda").write_text("#usda\n")
    return main


def _run(script, *args, cwd):
    return subprocess.run(
        [BASH, str(cwd / "scripts" / script), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_new_worktree_links_shared_inputs_and_never_outputs(repo, tmp_path):
    wt = tmp_path / "wt-a"
    out = _run("new_worktree.sh", str(wt), "--no-sync", "--run", cwd=repo)
    assert out.returncode == 0, out.stderr
    assert (wt / "third_party").is_symlink()
    assert (wt / "third_party" / "lewm" / "keep").read_text() == "pinned\n"
    assert (wt / "data").is_symlink() and not (wt / "checkpoints").exists()
    assert (wt / "assets" / "isaac").is_symlink()
    assert _git(wt, "status", "--porcelain") == ""  # the links are ignored (audit review B2)
    assert not (wt / "outputs").exists()
    assert _git(wt, "rev-parse", "HEAD") == _git(repo, "rev-parse", "origin/main")
    assert "check: skipped" in out.stdout
    # without --run no evidence directory is linked; --branch makes a branch
    wt_b = tmp_path / "wt-b"
    out = _run("new_worktree.sh", str(wt_b), "--no-sync", "--branch", "feat/x", cwd=repo)
    assert out.returncode == 0, out.stderr
    assert not (wt_b / "data").exists()
    assert _git(wt_b, "rev-parse", "--abbrev-ref", "HEAD") == "feat/x"
    assert _run("new_worktree.sh", str(wt_b), "--no-sync", cwd=repo).returncode == 1


def test_new_worktree_refuses_without_third_party(repo, tmp_path):
    shutil.rmtree(repo / "third_party")
    out = _run("new_worktree.sh", str(tmp_path / "wt"), "--no-sync", cwd=repo)
    assert out.returncode == 1 and "third_party is missing" in out.stderr
    assert not (tmp_path / "wt").exists()


def test_remove_worktree_refuses_dirty_unpushed_or_evidence(repo, tmp_path):
    wt = tmp_path / "wt"
    assert _run("new_worktree.sh", str(wt), "--no-sync", "--branch", "w", cwd=repo).returncode == 0

    (wt / "notes.txt").write_text("unsaved\n")
    out = _run("remove_worktree.sh", str(wt), cwd=repo)
    assert out.returncode == 1 and "untracked" in out.stderr

    _git(wt, "add", "notes.txt")
    _git(wt, "commit", "-m", "local only")
    out = _run("remove_worktree.sh", str(wt), cwd=repo)
    assert out.returncode == 1 and "on no remote branch" in out.stderr

    _git(wt, "push", "-u", "origin", "w")
    (wt / "outputs").mkdir()
    (wt / "outputs" / "run.json").write_text("{}")
    out = _run("remove_worktree.sh", str(wt), cwd=repo)
    assert out.returncode == 1 and "outputs/" in out.stderr

    shutil.rmtree(wt / "outputs")
    assert _run("remove_worktree.sh", str(wt), "--dry-run", cwd=repo).returncode == 0
    assert wt.exists()
    out = _run("remove_worktree.sh", str(wt), cwd=repo)
    assert out.returncode == 0, out.stderr
    assert not wt.exists()
    assert (repo / "third_party" / "lewm" / "keep").read_text() == "pinned\n"  # target kept


def test_remove_worktree_lists_other_ignored_content_and_needs_force(repo, tmp_path):
    wt = tmp_path / "wt"
    assert _run("new_worktree.sh", str(wt), "--no-sync", cwd=repo).returncode == 0
    (wt / "wandb").mkdir()
    (wt / "wandb" / "run.log").write_text("x")
    (wt / ".venv" / "lib").mkdir(parents=True)  # a known cache: never a reason to refuse
    (wt / "scripts" / "__pycache__").mkdir()
    (wt / "scripts" / "__pycache__" / "x.pyc").write_text("")
    out = _run("remove_worktree.sh", str(wt), "--dry-run", cwd=repo)
    assert out.returncode == 1 and "wandb" in out.stderr
    assert ".venv" not in out.stderr and "__pycache__" not in out.stderr
    assert "third_party" not in out.stderr and "assets" not in out.stderr  # symlinks are fine
    out = _run("remove_worktree.sh", str(wt), "--force-ignored", cwd=repo)
    assert out.returncode == 0, out.stderr
    assert not wt.exists()
    assert (repo / "assets" / "isaac" / "g1.usda").exists()


def test_remove_worktree_refuses_the_main_worktree(repo):
    out = _run("remove_worktree.sh", str(repo), cwd=repo)
    assert out.returncode == 1 and "main worktree" in out.stderr
