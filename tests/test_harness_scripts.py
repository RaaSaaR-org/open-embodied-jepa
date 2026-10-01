"""Host tests for the worktree and GPU-lock shell tools (the 2026-10-02 audit, F5 and F13).

No GPU, no network and never the real ``~/.local/state/gpu``: ``gpu_run.sh`` runs against a
temporary state directory and a stand-in ``nvidia-smi``; the worktree scripts run on throwaway
git repositories. Where ``flock(1)`` is missing (macOS), a stand-in built on ``fcntl.flock``
locks the inherited descriptor exactly as util-linux ``flock <fd>`` does."""

from __future__ import annotations

import os
import shutil
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


@pytest.fixture
def gpu_env(tmp_path):
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
    env = os.environ | {
        "GPU_STATE_DIR": str(state),
        "NVIDIA_SMI": str(smi),
        "PATH": f"{bin_dir}{os.pathsep}{path}",
    }
    env.pop("GPU_RUN_LOCKED", None)
    return {"state": state, "env": env, "tmp": tmp_path}


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
    )
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
    proc = subprocess.Popen(
        [BASH, str(GPU_RUN), "--samples", "1", "--wait", "--who", "tester"]
        + ["--request", "a short test, 1 s", "--", "true"],
        env=gpu_env["env"],
        cwd=gpu_env["tmp"],
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
    script = f"trap 'touch \"{marker}\"; exit 143' TERM; while :; do sleep 0.1; done"
    proc = subprocess.Popen(
        [BASH, str(GPU_RUN), "--samples", "1", "--", BASH, "-c", script],
        env=gpu_env["env"],
        cwd=gpu_env["tmp"],
    )
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
    (main / ".gitignore").write_text(
        "/third_party\n/third_party/\n/data\n/data/\n/checkpoints\n/checkpoints/\n"
        "/outputs\n/outputs/\n/assets/isaac/\n/assets/isaac\n"
    )
    (main / "README").write_text("x\n")
    _git(main, "add", ".")
    _git(main, "commit", "-m", "init")
    _git(main, "push", "-u", "origin", "main")
    (main / "third_party" / "lewm").mkdir(parents=True)
    (main / "third_party" / "lewm" / "keep").write_text("pinned\n")
    (main / "data" / "corpus").mkdir(parents=True)
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


def test_remove_worktree_refuses_the_main_worktree(repo):
    out = _run("remove_worktree.sh", str(repo), cwd=repo)
    assert out.returncode == 1 and "main worktree" in out.stderr
