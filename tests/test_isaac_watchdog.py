"""The e9 server's start-up watchdog (scripts/isaac/serve_with_watchdog.sh), with a fake launcher.

No Isaac, Docker or GPU: ``ISAAC_WATCHDOG_RUN`` points at a stand-in for ``run_isaac.sh`` and
``ISAAC_WATCHDOG_STOP`` at a command that records which container would have been stopped.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/isaac/serve_with_watchdog.sh"

# Stand-in for run_isaac.sh: <script> <out-dir> [args]. Behaviour per call from $PLAN (one word
# per try): "hang" never logs "transport built"; "ok" logs it and exits 0 after a moment;
# "crash" exits 7 without building.
FAKE = """#!/usr/bin/env bash
set -eu
OUT=$2
mkdir -p "$OUT"
n=$(cat "$STATE/calls" 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > "$STATE/calls"
echo "$*" >> "$STATE/args"
what=$(echo "$PLAN" | cut -d' ' -f"$n")
echo "[e9-server +   3.0s] building IsaacTransport (physics=newton)" > "$OUT/log.txt"
case "$what" in
  hang) exec sleep 600 ;;
  ok) echo "[e9-server + 130.0s] transport built" >> "$OUT/log.txt"; sleep 0.5; exit 0 ;;
  crash) exit 7 ;;
esac
"""


def _run(tmp_path, plan: str, tries: int = 3):
    fake = tmp_path / "fake_run_isaac.sh"
    fake.write_text(FAKE)
    fake.chmod(0o755)
    stop = tmp_path / "fake_stop.sh"
    stop.write_text('#!/usr/bin/env bash\necho "$@" >> "$STATE/stopped"\n')
    stop.chmod(0o755)
    state = tmp_path / "state"
    state.mkdir()
    env = {
        **os.environ,
        "ISAAC_WATCHDOG_RUN": str(fake),
        "ISAAC_WATCHDOG_STOP": str(stop),
        "ISAAC_WATCHDOG_POLL": "0.1",
        "ISAAC_WATCHDOG_GRACE": "1",
        "PLAN": plan,
        "STATE": str(state),
    }
    prefix = tmp_path / "out" / "srv"
    proc = subprocess.run(
        [str(SCRIPT), str(prefix), "2", str(tries), "--physics", "newton"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    read = lambda name: (state / name).read_text() if (state / name).exists() else ""  # noqa: E731
    return proc, prefix, read


pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def test_watchdog_retries_a_hung_start_and_serves_the_next(tmp_path):
    proc, prefix, read = _run(tmp_path, "hang ok")
    assert proc.returncode == 0, proc.stderr
    log = Path(f"{prefix}-watchdog.log").read_text()
    assert "try 1/3: HANG" in log and "retrying" in log and "try 2/3: transport built" in log
    assert Path(f"{prefix}-ready").read_text().strip() == f"{prefix}-t2/run/e9.sock"
    assert read("stopped").split() == ["oej-isaac-srv-t1"]  # only the hung try's container
    assert read("calls").strip() == "2"
    assert "e9_server_isaac.py" in read("args") and "--physics newton" in read("args")


def test_watchdog_gives_up_after_the_bounded_number_of_tries(tmp_path):
    proc, prefix, read = _run(tmp_path, "hang hang", tries=2)
    assert proc.returncode == 4
    assert read("stopped").split() == ["oej-isaac-srv-t1", "oej-isaac-srv-t2"]
    assert not Path(f"{prefix}-ready").exists()
    assert "every try hung (2 of 2)" in Path(f"{prefix}-watchdog.log").read_text()


def test_watchdog_does_not_retry_a_start_that_fails_otherwise(tmp_path):
    proc, prefix, read = _run(tmp_path, "crash ok")
    assert proc.returncode == 7
    assert read("calls").strip() == "1" and read("stopped") == ""
    assert "not retried" in Path(f"{prefix}-watchdog.log").read_text()


def test_watchdog_refuses_to_overwrite(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "srv-watchdog.log").write_text("earlier")
    proc, _, read = _run(tmp_path, "ok")
    assert proc.returncode == 1 and read("calls") == ""
