"""Run guards for gated simulation runners, with the #108 review follow-ups fixed (TASK-074).

TASK-073's runner (``scripts/run_wm_critic_v2.py``) and design module (``wm_critic_v2.py``) are
pinned by sha256 in ``benchmarks/manifests/apple-wm-critic-v2.json`` and TASK-073 is closed, so
they are left byte-for-byte as K0 run-2 ran them (revision ``35772e5``). The independent review of
#108 listed follow-ups that must be fixed before any later gated run reuses that code. The fixed
versions live here, for TASK-074 and later runners to import:

1. :func:`rerun_matches`: a compared field missing from **both** records is a failure, not a
   match, and ``max_abs_command_difference`` is computed from the executed commands (required;
   a NaN or a shape mismatch is ``inf``). A :class:`RerunRule` must cover every compared
   quantity of an image-reading arm, including a command tolerance, or it is rejected.
2. :class:`BoundedPool`: ``close`` is bounded. It closes the pool so no worker is respawned and idle
   workers exit, joins each worker with a timeout, SIGKILLs survivors (through a pidfd on Linux,
   so a reused pid is never signalled), and only then calls ``Pool.terminate`` in a daemon thread
   that is itself joined with a timeout; that thread's timeout is what bounds ``close``
   (``Pool.terminate`` joins its workers without a timeout, so a worker stuck in uninterruptible
   sleep could otherwise block the runner forever).
3. :func:`pss_bytes` falls back to RSS when ``/proc/<pid>/smaps_rollup`` cannot be read. The
   fallback is counted per sample, so a report can say its measure was partly RSS.

Standard library and NumPy only; nothing here imports torch or MuJoCo.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ----- 1. the determinism re-run comparison ------------------------------------------------------
RERUN_FIELDS = (
    "success",
    "grasp",
    "at_rest",
    "termination_reason",
    "executed_steps",
    "final_distance_cm",
    "first_grasp_step",
    "first_place_step",
)
COMMAND_KEY = "max_abs_command_difference"
CHOSEN_KEY = "chosen_sequence"


def max_abs_command_difference(first, second) -> float:
    """The largest absolute difference between two executed command arrays. ``inf`` when their
    shapes differ (for example one attempt ran longer) or when any entry is not finite in either
    run (a NaN must never read as "no difference"; reviewer B1 of #111). Bitwise-equal arrays,
    including equal non-finite entries, give 0."""
    a = np.asarray(first, np.float64)
    b = np.asarray(second, np.float64)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0 or np.array_equal(a, b, equal_nan=True):
        return 0.0 if np.isfinite(a).all() else float("inf")
    diff = np.abs(a - b)
    if not np.isfinite(diff).all():
        return float("inf")
    return float(np.max(diff))


@dataclass(frozen=True)
class RerunRule:
    """How a determinism re-run of one arm is compared (validated at construction).

    * ``image_reading=False``: every field in ``fields``, the decision sequence and the executed
      commands must be identical (command difference exactly 0).
    * ``image_reading=True`` (the renderer's rare one-level pixel differences can reach the arm):
      every compared quantity must be covered, either by ``exact`` or by ``tolerances``. That is
      every name in ``fields``, ``"chosen_sequence"`` and ``"max_abs_command_difference"``, so a
      command tolerance is required. A configuration that leaves anything uncovered, or names
      something that is never compared, is rejected (reviewer N1 of #111)."""

    image_reading: bool
    exact: tuple = ()
    tolerances: dict = field(default_factory=dict)
    fields: tuple = RERUN_FIELDS
    decision_key: str = "chosen"

    def __post_init__(self):
        object.__setattr__(self, "exact", tuple(self.exact))
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "tolerances", dict(self.tolerances))
        compared = {*self.fields, CHOSEN_KEY, COMMAND_KEY}
        named = set(self.exact) | set(self.tolerances)
        if set(self.exact) & set(self.tolerances):
            raise ValueError(
                f"rerun rule: {sorted(set(self.exact) & set(self.tolerances))} "
                "are both exact and toleranced"
            )
        if named - compared:
            raise ValueError(f"rerun rule: {sorted(named - compared)} are never compared")
        for key, limit in self.tolerances.items():
            if not np.isfinite(float(limit)) or float(limit) < 0:
                raise ValueError(f"rerun rule: tolerance for {key} must be finite and >= 0")
        if self.image_reading:
            if COMMAND_KEY not in self.tolerances and COMMAND_KEY not in self.exact:
                raise ValueError("rerun rule: an image-reading arm needs a command tolerance")
            if compared - named:
                raise ValueError(f"rerun rule: {sorted(compared - named)} are not covered")
        elif named:
            raise ValueError(
                "rerun rule: a non-image arm is compared bitwise; no exact or "
                "tolerance fields may be given"
            )


def rerun_matches(first: dict, second: dict, rule: RerunRule) -> dict:
    """Compare one (arm, seed) attempt with its determinism re-run under ``rule``.

    Every name in ``rule.fields`` must be present in at least one record: a field missing from
    both fails the comparison (it is not a match). A field present in only one record is a
    difference. Both records must carry ``commands`` (the executed command array); a missing
    array fails. A non-finite command difference (NaN or a shape mismatch) is ``inf`` and fails
    every arm, since no tolerance is infinite.

    Returns ``{"matches", "differences", "missing", "max_abs_command_difference"}``. Every
    difference is returned, within tolerance or not."""
    if not isinstance(rule, RerunRule):
        raise TypeError("rerun_matches needs a RerunRule")
    diffs: dict = {}
    missing: list[str] = []
    for key in rule.fields:
        in_a, in_b = key in first, key in second
        if not in_a and not in_b:
            missing.append(key)
        elif in_a != in_b or not _same(first.get(key), second.get(key)):
            diffs[key] = [first.get(key), second.get(key)]
    choose_a = [d.get(rule.decision_key) for d in first.get("decisions", [])]
    choose_b = [d.get(rule.decision_key) for d in second.get("decisions", [])]
    if choose_a != choose_b:
        diffs[CHOSEN_KEY] = [choose_a, choose_b]
    if "commands" not in first or "commands" not in second:
        missing.append("commands")
        command_diff = None
    else:
        command_diff = max_abs_command_difference(first["commands"], second["commands"])
        if not command_diff == 0:
            diffs[COMMAND_KEY] = command_diff
    result = {"differences": diffs, "missing": missing, COMMAND_KEY: command_diff}
    if missing:
        return result | {"matches": False}
    if not rule.image_reading:
        return result | {"matches": not diffs}
    ok = True
    for key, value in diffs.items():
        if key in rule.exact:
            ok = False
        elif key == COMMAND_KEY:
            ok = ok and float(value) <= float(rule.tolerances[key])
        else:
            a, b = value
            ok = ok and _within(a, b, rule.tolerances[key])
    return result | {"matches": bool(ok)}


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return False  # a NaN result is never a confirmed match
    return a == b


def _within(a, b, limit) -> bool:
    try:
        d = abs(float(a) - float(b))
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(d) and d <= float(limit))


# ----- 3. memory: PSS with an RSS fallback ------------------------------------------------------
def _status_rss(pid: int, proc: Path) -> int | None:
    try:
        with open(proc / str(pid) / "status") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return 0  # a kernel thread or a zombie: no VmRSS line


def pss_bytes(pid: int, proc: Path | str = "/proc") -> tuple[int, str]:
    """``(bytes, source)``: the PSS from ``smaps_rollup``, or, when that file cannot be read or
    has no ``Pss:`` line, the process's VmRSS with ``source == "rss"`` (an upper bound on its
    PSS). ``(0, "gone")`` when the process no longer exists."""
    proc = Path(proc)
    try:
        with open(proc / str(pid) / "smaps_rollup") as rollup:
            for line in rollup:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024, "pss"
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        pass
    rss = _status_rss(pid, proc)
    if rss is None:
        return 0, "gone"
    return rss, "rss"


def process_tree_memory(
    root: int | None = None, per: dict | None = None, proc: Path | str = "/proc"
) -> dict:
    """Summed VmRSS and PSS of a process and all its descendants, from ``/proc`` (Linux).

    ``pss`` uses :func:`pss_bytes`, so a process whose PSS cannot be read counts with its RSS;
    ``pss_fallback_processes`` says how many did. ``per`` collects each pid's
    ``(rss, pss_or_fallback)``. Each pid is read on its own, so one exiting process drops only
    itself from a sample."""
    proc = Path(proc)
    root = os.getpid() if root is None else int(root)
    seen, stack = set(), [root]
    rss_total = pss_total = fallbacks = 0
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        rss = _status_rss(pid, proc)
        if rss is None:
            continue
        pss, source = pss_bytes(pid, proc)
        if source == "gone":
            continue
        fallbacks += source == "rss"
        rss_total, pss_total = rss_total + rss, pss_total + pss
        if per is not None:
            per[pid] = (rss, pss)
        try:
            tasks = os.listdir(proc / str(pid) / "task")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for task in tasks:
            try:
                with open(proc / str(pid) / "task" / task / "children") as children:
                    stack.extend(int(c) for c in children.read().split())
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
    return {"rss": rss_total, "pss": pss_total, "pss_fallback_processes": fallbacks}


# ----- 2. a spawn pool whose close is bounded ---------------------------------------------------
class WorkerDied(RuntimeError):
    """A pool worker exited during a map (G-worker): the stage is V."""


class MapCapExceeded(RuntimeError):
    """A map ran past its wall-time cap (G-cap): the stage is V."""


def _open_pidfd(pid: int) -> int | None:
    """A pidfd for ``pid`` on Linux (``os.pidfd_open``). While it is open, signals sent through
    it reach that process or nothing, never a later process that reuses the pid. ``None`` where
    pidfds are unavailable (macOS) or the process is already gone."""
    if not hasattr(os, "pidfd_open"):
        return None
    try:
        return os.pidfd_open(int(pid))
    except OSError:  # ProcessLookupError included: already reaped
        return None


def _alive(process, pidfd: int | None = None) -> bool:
    """Whether a worker still runs; a reaped pid or a zombie counts as dead. With a pidfd the
    answer is race-free (the pidfd polls readable once the process has exited). Without one:
    ``exitcode``, then ``kill(pid, 0)`` and, on Linux, the zombie state in ``/proc``."""
    if process.exitcode is not None:
        return False
    if pidfd is not None:
        import select

        poller = select.poll()
        poller.register(pidfd, select.POLLIN)
        return not poller.poll(0)
    try:
        os.kill(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{process.pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return not sys.platform.startswith("linux")


def _kill(process, pidfd: int | None) -> str | None:
    """SIGKILL a worker, through its pidfd when there is one. Returns an error string instead of
    raising (``close`` must always return its record)."""
    try:
        if pidfd is not None:
            signal.pidfd_send_signal(pidfd, signal.SIGKILL)
        else:
            os.kill(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return None
    except OSError as exc:  # PermissionError included
        return repr(exc)
    return None


def _wait_dead(process, seconds: float, pidfd: int | None = None) -> bool:
    """Join ``process`` for at most ``seconds``; True once it is dead."""
    deadline = time.monotonic() + float(seconds)
    while True:
        process.join(timeout=min(0.1, max(deadline - time.monotonic(), 0.0)))
        if not _alive(process, pidfd):
            return True
        if time.monotonic() >= deadline:
            return False


class BoundedPool:
    """A spawn pool that raises at once when a worker dies during a map, and whose ``close``
    always returns within about ``2 * join_seconds * workers + terminate_seconds``.

    ``close`` records what it had to do in ``close_record`` (for the run report)."""

    POLL_SECONDS = 1.0

    def __init__(
        self,
        workers: int,
        func: Callable,
        *,
        initializer: Callable | None = None,
        initargs: tuple = (),
        context: str = "spawn",
        join_seconds: float = 10.0,
        terminate_seconds: float = 30.0,
        check: Callable[[list], None] | None = None,
    ):
        self.pool = mp.get_context(context).Pool(
            workers, initializer=initializer, initargs=initargs
        )
        self.func = func
        self.join_seconds = float(join_seconds)
        self.terminate_seconds = float(terminate_seconds)
        self.check = check
        self.pids = self._pids()
        self.close_record: dict | None = None
        # pidfds taken while the workers are known to be ours (just started, not yet reaped)
        self._pidfds = {p.pid: _open_pidfd(p.pid) for p in list(self.pool._pool)}

    def _pids(self) -> set:
        return {p.pid for p in list(self.pool._pool)}

    def map(self, tasks: list, cap: float, what: str) -> list:
        started = time.monotonic()
        result = self.pool.map_async(self.func, tasks, chunksize=1)
        while True:
            remaining = cap - (time.monotonic() - started)
            if remaining <= 0:
                raise MapCapExceeded(f"G-cap: {what} exceeded its {cap} s cap")
            try:
                out = result.get(timeout=min(self.POLL_SECONDS, max(remaining, 0.01)))
                break
            except mp.TimeoutError:
                pass
            dead = [p for p in list(self.pool._pool) if p.exitcode is not None]
            now = self._pids()
            if dead or not now <= self.pids or len(now) < len(self.pids):
                detail = [(p.pid, p.exitcode) for p in dead] or sorted(now ^ self.pids)
                raise WorkerDied(f"G-worker: a worker died during {what}: {detail}")
        if self.check is not None:
            self.check(out)
        return out

    def close(self) -> dict:
        """Bounded shutdown. Idempotent; returns ``close_record``."""
        if self.close_record is not None:
            return self.close_record
        started = time.monotonic()
        record: dict = {"joined": [], "killed": [], "unkillable": [], "terminate_returned": None}
        workers = list(self.pool._pool)
        for p in workers:  # a worker respawned since construction gets its pidfd now
            if p.pid not in self._pidfds:
                self._pidfds[p.pid] = _open_pidfd(p.pid)
        record["pidfd"] = any(fd is not None for fd in self._pidfds.values())
        record["kill_errors"] = {}
        try:
            self.pool.close()  # no new tasks; the worker handler stops respawning once idle
        except Exception as exc:  # noqa: BLE001 - the bounded steps below still run
            record["close_error"] = repr(exc)
        # Idle workers exit on the close sentinel, so no blanket signal is sent first. A worker
        # killed while it holds the task queue's read lock (SIGTERM or the SIGKILL below alike)
        # can make Pool.terminate's lock-draining step hang; what bounds close() is that
        # Pool.terminate runs in a daemon thread joined with a timeout (reviewer N3 of #111).
        for p in workers:
            if _wait_dead(p, self.join_seconds, self._pidfds.get(p.pid)):
                record["joined"].append(p.pid)
        for p in workers:
            if p.pid in record["joined"]:
                continue
            fd = self._pidfds.get(p.pid)
            error = _kill(p, fd)
            if error:
                record["kill_errors"][str(p.pid)] = error
            dead = _wait_dead(p, self.join_seconds, fd)
            record["killed" if dead else "unkillable"].append(p.pid)
        # Pool.terminate joins its workers and handler threads without a timeout; run it where a
        # hang cannot block the runner.
        done = threading.Event()

        def _terminate():
            try:
                self.pool.terminate()
            finally:
                done.set()

        thread = threading.Thread(target=_terminate, daemon=True)
        thread.start()
        thread.join(timeout=self.terminate_seconds)
        record["terminate_returned"] = done.is_set()
        for p in list(self.pool._pool):  # anything respawned meanwhile
            fd = self._pidfds.get(p.pid)
            if _alive(p, fd) and p.pid not in record["killed"]:
                error = _kill(p, fd)
                if error:
                    record["kill_errors"][str(p.pid)] = error
                    record["unkillable"].append(p.pid)
                else:
                    record["killed"].append(p.pid)
        for fd in self._pidfds.values():
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._pidfds = {}
        record["seconds"] = time.monotonic() - started
        self.close_record = record
        return record


# ----- probes for the pool's tests (importable by spawned workers) ------------------------------
def probe_init(ignore_sigterm: bool = False) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN if ignore_sigterm else signal.SIG_DFL)


def probe_task(task: dict) -> dict:
    if task.get("marker_dir"):
        (Path(task["marker_dir"]) / f"started-{os.getpid()}").touch()
    time.sleep(float(task.get("seconds", 0.0)))
    return {"seed": task.get("seed"), "pid": os.getpid()}
