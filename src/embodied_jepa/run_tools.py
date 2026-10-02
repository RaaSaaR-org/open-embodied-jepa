"""Shared harness pieces for new gated runners (the 2026-10-02 audit, PR-C).

``run_guards.py`` is pinned by sha256 in the TASK-074 and TASK-075 manifests, so it is not edited;
the pieces below live here instead. Each one replaces code that the pinned runners copy between
scripts (``scripts/run_wm_critic_v2.py`` -> ``run_lewm_planner_v2.py`` ->
``run_obs_ceiling_v2.py``). New runners import them from here and never load another runner
script (``tests/test_no_runner_imports.py``).

* :func:`assert_local_import` (F1d): the imported ``embodied_jepa`` must be the runner's own
  checkout's ``src``, not a shared venv's editable install of another worktree.
* :func:`install_guards` / :class:`Guards` (F11): stop signals end the stage as V. The handler
  records the *first* signal before it raises, and :meth:`Guards.void` reports that signal even
  when the raise landed inside library code and surfaced as another exception (TASK-073 main run
  36793229744 recorded ``RuntimeError: release unlocked lock`` instead of ``received SIGTERM``).
* :class:`MemoryWatch`: the process-tree memory ceiling (PSS or RSS, via
  ``run_guards.process_tree_memory``), with an optional free-disk floor; it signals SIGUSR1.
* :func:`budget_rule_consistent`, :func:`select_checkpoint`, :func:`last_two_triggered` (F12):
  a saturation budget rule whose cap is below ``factor * calibration_updates`` escalates by
  design (TASK-066, and TASK-074 twice). Future preregistrations check their rule with this.
* :func:`gpu_guard` (F13): free GPU memory at start, an optional per-process cap, and whether
  the machine-wide GPU lock (``scripts/gpu_run.sh``) is held.
* :class:`CycledReader`, :class:`SlotProxy`, :func:`scale_probe` (F15): a memory probe that runs
  the stage's own function at the real stage's sizes on smoke data (TASK-074 addendum A1: a probe
  with its own copy of the stage steps reported 8.77 GiB against a real 12.04 GiB peak).

Standard library and NumPy only at import time; :func:`gpu_guard` imports torch lazily.
"""

from __future__ import annotations

import inspect
import math
import os
import shutil
import signal
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path

GIB = 2**30
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
GPU_LOCK_ENV = "GPU_RUN_LOCKED"  # set to "1" by scripts/gpu_run.sh while it holds the lock


class GuardError(RuntimeError):
    """A run guard refused to start or continue (the stage is V or never starts)."""


# ----- F1d: the import location ------------------------------------------------------------------
def _find_root(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / "src" / "embodied_jepa" / "__init__.py").is_file():
            return candidate
    return None


def assert_local_import(root: Path | str | None = None, report: dict | None = None) -> Path:
    """Require that ``embodied_jepa`` was imported from ``<root>/src``, and record where.

    ``root`` defaults to the checkout that contains the running script (``sys.argv[0]``), else
    the current directory. A worktree whose venv's editable install points at another checkout
    (the 2026-10-02 audit: the shared ``.venv`` pointed at the primary checkout) runs other code
    than its recorded revision; that is refused. Returns the resolved package directory."""
    import embodied_jepa

    if root is None:
        script = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
        root = (_find_root(script.parent) if script else None) or _find_root(Path.cwd())
        if root is None:
            raise GuardError("G-import: no checkout with src/embodied_jepa above the script")
    expected = (Path(root).resolve() / "src" / "embodied_jepa").resolve()
    actual = Path(embodied_jepa.__file__).resolve().parent
    if report is not None:
        report["embodied_jepa_file"] = str(Path(embodied_jepa.__file__).resolve())
        report["import_root"] = str(Path(root).resolve())
    if actual != expected:
        raise GuardError(
            f"G-import: embodied_jepa was imported from {actual}, not {expected}; "
            "sync this checkout's own venv (scripts/new_worktree.sh) or run with "
            "PYTHONPATH=src"
        )
    return actual


# ----- F11: signals that record the first signal before raising ----------------------------------
class StageInterrupted(BaseException):
    """A stop signal or a watch ended the stage: it is V, and the report is written."""

    def __init__(self, reason: str, utc: str, signal_name: str | None = None):
        super().__init__(reason)
        self.reason, self.utc, self.signal_name = reason, utc, signal_name


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Guards:
    """The signal state of one stage. Build it with :func:`install_guards`.

    ``first_signal`` is written by the handler *before* it raises, so it survives a raise that
    lands inside library code (a lock release, a pool's internals) and surfaces as another
    exception; :meth:`void` reports it. Re-entry is guarded by ``stopping``, never by SIG_IGN
    (an ignored disposition would be inherited by respawned pool workers)."""

    def __init__(self, watch=None, signals: Iterable[int] = STOP_SIGNALS, log=None):
        self.watch = watch
        self.signals = tuple(signals)
        self.log = log
        self.stopping = False
        self.first_signal: dict | None = None
        self.late_signals: list[dict] = []
        self._previous: dict = {}

    def handler(self, signum, _frame):
        utc = _utc()
        name = signal.Signals(signum).name
        if self.stopping:
            self.late_signals.append({"signal": name, "utc": utc})
            return
        self.stopping = True
        reason = None
        if signum == signal.SIGUSR1:
            reason = getattr(self.watch, "reason", None)
        reason = reason or f"received {name}"
        # recorded first: the raise below may surface as a different exception
        self.first_signal = {"signal": name, "utc": utc, "reason": reason}
        raise StageInterrupted(f"{reason} ({name})", utc, name)

    def install(self) -> Guards:
        for s in (*self.signals, signal.SIGUSR1):
            self._previous[s] = signal.signal(s, self.handler)
        if self.watch is not None and hasattr(self.watch, "start"):
            self.watch.start()
        return self

    def uninstall(self) -> None:
        """Restore the previous handlers (tests; a runner normally exits instead)."""
        for s, previous in self._previous.items():
            signal.signal(s, previous)
        self._previous = {}

    def stop_handling(self) -> None:
        """Call first in every except/finally: from here on a signal is recorded, not raised."""
        self.stopping = True

    def void(self, report: dict, error: BaseException) -> None:
        """Record a V. A recorded first signal is the reason, whatever exception arrived; the
        exception that did arrive is kept beside it when it is a different one."""
        self.stop_handling()
        report["outcome"] = "V"
        if self.first_signal is not None:
            report["void_reason"] = (
                f"StageInterrupted: {self.first_signal['reason']} ({self.first_signal['signal']})"
            )
            report["interrupted_utc"] = self.first_signal["utc"]
            report["first_signal"] = dict(self.first_signal)
            if not isinstance(error, StageInterrupted):
                report["void_surfaced_as"] = f"{type(error).__name__}: {error}"
        elif isinstance(error, StageInterrupted):
            report["void_reason"] = f"StageInterrupted: {error.reason}"
            report["interrupted_utc"] = error.utc
        else:
            report["void_reason"] = f"{type(error).__name__}: {error}"
        report["traceback"] = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        if self.log is not None:
            self.log(f"VOID: {report['void_reason']}")

    def finish(self, report: dict, pool=None) -> None:
        """The first statements of every runner's ``finally``: stop raising, void a stage whose
        except clause was itself interrupted, close the pool, record memory and late signals."""
        self.stop_handling()
        error = sys.exc_info()[1]
        if report.get("outcome") != "V" and (
            isinstance(error, StageInterrupted) or (error is not None and self.first_signal)
        ):
            self.void(report, error)
        if pool is not None:
            pool.close()
        if self.watch is not None:
            if hasattr(self.watch, "stop"):
                self.watch.stop()
            if hasattr(self.watch, "summary"):
                report["memory"] = self.watch.summary()
        if self.late_signals:
            report["signals_during_cleanup"] = list(self.late_signals)


def install_guards(watch=None, signals: Iterable[int] = STOP_SIGNALS, log=None) -> Guards:
    """Install the stop-signal handlers (and SIGUSR1 for ``watch``) and start ``watch``."""
    return Guards(watch, signals, log).install()


# ----- the memory watch -------------------------------------------------------------------------
class MemoryWatch(threading.Thread):
    """Samples the process tree's memory every ``interval`` seconds; above ``ceiling_bytes`` (on
    ``measure``, "pss" or "rss"), or below ``min_disk_free_bytes`` free on ``disk_path``, it
    records why and sends SIGUSR1 to this process once (whose :class:`Guards` handler ends the
    stage as V)."""

    def __init__(
        self,
        ceiling_bytes: int,
        interval: float,
        measure: str = "pss",
        disk_path: Path | str | None = None,
        min_disk_free_bytes: int | None = None,
        sampler: Callable[..., dict] | None = None,
    ):
        super().__init__(daemon=True)
        if measure not in ("pss", "rss"):
            raise ValueError("measure must be 'pss' or 'rss'")
        if sampler is None:
            from embodied_jepa.run_guards import process_tree_memory as sampler
        self.sampler = sampler
        self.ceiling, self.interval, self.measure = int(ceiling_bytes), float(interval), measure
        self.disk_path = None if disk_path is None else Path(disk_path)
        self.min_disk_free = min_disk_free_bytes
        self.reason: str | None = None
        self.peak_rss = self.peak_pss = self.samples = self.peak_processes = 0
        self.window_peak = 0  # the peak on ``measure`` since the last reset_window()
        self.min_disk_free_seen: int | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def sample(self) -> dict:
        per: dict = {}
        both = self.sampler(per=per)
        with self._lock:
            self.peak_rss = max(self.peak_rss, both["rss"])
            self.peak_pss = max(self.peak_pss, both["pss"])
            self.peak_processes = max(self.peak_processes, len(per))
            self.window_peak = max(self.window_peak, both[self.measure])
            self.samples += 1
        if both[self.measure] > self.ceiling:
            self._trip(
                f"G-memory: process-tree {self.measure.upper()} "
                f"{both[self.measure] / GIB:.2f} GiB > {self.ceiling / GIB:.2f} GiB"
            )
        if self.disk_path is not None and self.min_disk_free is not None:
            free = shutil.disk_usage(self.disk_path).free
            seen = self.min_disk_free_seen
            self.min_disk_free_seen = free if seen is None else min(seen, free)
            if free < self.min_disk_free:
                self._trip(f"G-disk: {free / GIB:.1f} GiB free < {self.min_disk_free / GIB:.1f}")
        return both

    def _trip(self, reason: str) -> None:
        if self.reason is None:
            self.reason = reason
            os.kill(os.getpid(), signal.SIGUSR1)

    def reset_window(self) -> None:
        """Start a new window for ``window_peak`` (a scale probe's own peak)."""
        with self._lock:
            self.window_peak = 0

    @property
    def peak(self) -> int:
        return self.peak_pss if self.measure == "pss" else self.peak_rss

    def run(self):
        while not self._stop_event.wait(self.interval):
            self.sample()

    def stop(self):
        self._stop_event.set()

    def summary(self) -> dict:
        out = {
            "measure": self.measure,
            "ceiling_gib": self.ceiling / GIB,
            "peak_tree_gib": self.peak / GIB,
            "peak_tree_rss_gib": self.peak_rss / GIB,
            "peak_tree_pss_gib": self.peak_pss / GIB,
            "peak_processes": self.peak_processes,
            "samples": self.samples,
            "reason": self.reason,
        }
        if self.min_disk_free_seen is not None:
            out["min_disk_free_gib"] = self.min_disk_free_seen / GIB
        return out


# ----- F12: budget rules that cannot escalate by design ------------------------------------------
def budget_rule_consistent(
    cap: int,
    factor: float,
    calibration_updates: int,
    *,
    step: int | None = None,
    min_updates: int | None = None,
    calibration_select_every: int | None = None,
    selection_points: int | None = None,
    saturation_tolerance: float | None = None,
) -> dict:
    """Whether a saturation budget rule ``U = clamp(step * ceil(factor * max(u_sat) / step),
    min, cap)``, escalating when ``factor * max(u_sat) > cap``, can reach its escalation only by
    a result, not by its own arithmetic.

    A saturation update is at most the calibration's length, so the rule can request up to
    ``factor * calibration_updates``. A cap below that (TASK-074: cap = calibration = 60 000 with
    factor 2, so every saturation above 30 000 escalates) is rejected. Returns ``{"consistent",
    "problems", "max_request", "min_cap"}``; ``problems`` names every violated condition."""
    problems: list[str] = []
    values = {"cap": cap, "calibration_updates": calibration_updates}
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            problems.append(f"{name} must be a positive integer")
    if not (isinstance(factor, int | float) and math.isfinite(factor) and factor >= 1):
        problems.append("factor must be finite and >= 1")
    if problems:
        return {"consistent": False, "problems": problems, "max_request": None, "min_cap": None}
    max_request = factor * calibration_updates
    min_cap = max_request
    if step is not None:
        if not isinstance(step, int) or step < 1:
            problems.append("step must be a positive integer")
        else:
            min_cap = step * math.ceil(max_request / step)
            if cap % step:
                problems.append(f"cap {cap} is not a multiple of step {step}")
    if cap < min_cap:
        problems.append(
            f"cap {cap} < {min_cap:g}: factor {factor:g} x calibration {calibration_updates} "
            "can be requested, so the rule escalates by design"
        )
    if min_updates is not None and min_updates > cap:
        problems.append(f"min {min_updates} > cap {cap}")
    if calibration_select_every is not None:
        if calibration_select_every < 1 or calibration_updates % calibration_select_every:
            problems.append("calibration_updates must be a positive multiple of its select_every")
        elif calibration_updates // calibration_select_every < 3:
            problems.append("a calibration curve needs at least 3 selection points")
    if selection_points is not None and selection_points < 3:
        problems.append("selection_points must be >= 3 (the last-two rule needs an earlier one)")
    if saturation_tolerance is not None and not (
        math.isfinite(saturation_tolerance) and 0 <= saturation_tolerance < 1
    ):
        problems.append("saturation_tolerance must be in [0, 1)")
    return {
        "consistent": not problems,
        "problems": problems,
        "max_request": max_request,
        "min_cap": min_cap,
    }


def check_budget(budget: dict) -> dict:
    """:func:`budget_rule_consistent` on a protocol's ``BUDGET`` block (the TASK-073/074 key
    names: ``cap``, ``factor``, ``calibration_updates``, ``step``, ``min``,
    ``calibration_select_every``, ``selection_points``, ``saturation_tolerance``)."""
    return budget_rule_consistent(
        budget["cap"],
        budget["factor"],
        budget["calibration_updates"],
        step=budget.get("step"),
        min_updates=budget.get("min"),
        calibration_select_every=budget.get("calibration_select_every"),
        selection_points=budget.get("selection_points"),
        saturation_tolerance=budget.get("saturation_tolerance"),
    )


def _curve(curve) -> list[tuple[int, float]]:
    points = [(int(u), float(v)) for u, v in curve]
    if not points or any(not math.isfinite(v) for _, v in points):
        raise ValueError("a selection curve needs finite criteria")
    if [u for u, _ in points] != sorted({u for u, _ in points}):
        raise ValueError("a selection curve must be strictly increasing in updates")
    return points


def select_checkpoint(curve, tolerance: float) -> int:
    """The earliest update whose criterion is within ``tolerance`` (relative) of the curve's
    minimum, not the raw argmin (the TASK-074 last-two rule used a raw argmin, so noise in a flat
    tail read as unsaturated). ``curve`` is ``[[update, criterion], ...]``, lower is better."""
    points = _curve(curve)
    floor = min(v for _, v in points)
    bound = floor + abs(floor) * float(tolerance)
    return next(u for u, v in points if v <= bound)


def last_two_triggered(curve, tolerance: float) -> bool:
    """The last-two rule beyond tolerance: True only when the checkpoint chosen by
    :func:`select_checkpoint` is one of the curve's last two points."""
    points = _curve(curve)
    return select_checkpoint(points, tolerance) in [u for u, _ in points[-2:]]


# ----- F13: the GPU guard ------------------------------------------------------------------------
def gpu_guard(
    report: dict,
    *,
    min_free_gib: float,
    process_cap_gib: float | None = None,
    require_lock: bool = False,
    device: int = 0,
    cuda=None,
) -> dict:
    """Start only with ``min_free_gib`` (plus ``process_cap_gib``, when given) free on the GPU;
    cap this process at ``process_cap_gib``; with ``require_lock``, refuse unless the machine-wide
    GPU lock is held (``scripts/gpu_run.sh`` sets ``GPU_RUN_LOCKED=1``). Generalised from
    TASK-075's runner. ``cuda`` defaults to ``torch.cuda`` (imported here, lazily)."""
    if cuda is None:
        import torch

        cuda = torch.cuda
    locked = os.environ.get(GPU_LOCK_ENV) == "1"
    record = {"gpu_lock_held": locked, "gpu_device": int(device)}
    report.update(record)
    if require_lock and not locked:
        raise GuardError("G-gpu-lock: run GPU jobs through scripts/gpu_run.sh")
    free, total = cuda.mem_get_info(device)
    need = (float(min_free_gib) + float(process_cap_gib or 0.0)) * GIB
    record |= {"gpu_free_gib_at_start": free / GIB, "gpu_total_gib": total / GIB}
    report.update(record)
    if free < need:
        raise GuardError(f"G-gpu: {free / GIB:.2f} GiB free < {need / GIB:.1f} GiB")
    if process_cap_gib is not None:
        cuda.set_per_process_memory_fraction(float(process_cap_gib) * GIB / total, device)
        record["gpu_process_cap_gib"] = float(process_cap_gib)
        report["gpu_process_cap_gib"] = float(process_cap_gib)
    return record


# ----- F15: a scale probe that runs the stage's own code -----------------------------------------
class CycledReader:
    """A smoke corpus reader presented at the real corpus's split sizes. Slot ``j`` of a split
    is served by a smoke episode (cycled over ``read_splits`` of the smoke reader), decoded afresh
    on every read exactly as the wrapped reader does, so the stage holds and frees the same arrays
    as on the real corpus.

    ``reader`` needs ``manifest["splits"]`` and ``episode(episode_id, **kwargs)``. ``counts`` maps
    split -> the real stage's slot count. ``source`` maps each slot to its smoke episode."""

    def __init__(self, reader, counts: dict, read_splits: Iterable[str] | None = None):
        self.reader = reader
        splits_in = reader.manifest["splits"]
        read_splits = list(splits_in) if read_splits is None else list(read_splits)
        pool = [e for s in read_splits for e in splits_in[s]]
        if not pool:
            raise ValueError("the smoke reader has no episodes in the read splits")
        self.source: dict[str, str] = {}
        splits: dict[str, list[str]] = {}
        k = 0
        for split, count in counts.items():
            if int(count) < 0:
                raise ValueError("slot counts must be >= 0")
            splits[split] = []
            for j in range(int(count)):
                slot = f"scale-{split}-{j}"
                self.source[slot] = pool[k % len(pool)]
                splits[split].append(slot)
                k += 1
        self.manifest = {"splits": splits}

    def episode(self, episode_id: str, *args, **kwargs):
        return self.reader.episode(self.source[episode_id], *args, **kwargs)


class SlotProxy:
    """Any other per-episode store seen through a :class:`CycledReader`'s slots: every method's
    first argument (an episode id) is mapped to its smoke episode."""

    def __init__(self, store, slot_source: dict):
        self._store, self._slot_source = store, slot_source

    def __getattr__(self, name):
        target = getattr(self._store, name)
        if not callable(target):
            return target

        def call(episode_id, *args, **kwargs):
            return target(self._slot_source[episode_id], *args, **kwargs)

        return call


def scale_probe(
    stage_core: Callable,
    *args,
    watch,
    ceiling_gib: float,
    margin_gib: float,
    **kwargs,
) -> dict:
    """Run the stage's own function, ``stage_core(*args, probe=True, **kwargs)``, and require
    the probe's memory peak to stay ``margin_gib`` below ``ceiling_gib`` (G-memory-margin).

    The probe must be the stage's code path at the stage's sizes (pass a :class:`CycledReader`
    in ``args``); a probe with its own copy of the stage steps drifts from the stage (TASK-074
    A1). Only the signature is checked (``stage_core`` must accept a ``probe`` keyword); that it
    is the stage's own function is the caller's convention, which the runner's tests should pin.

    The peak judged is the probe's own: with a watch that has ``reset_window()`` and
    ``window_peak`` (:class:`MemoryWatch`), the window is reset before the call and read after it
    (``peak_scope == "probe"``). A watch with only ``peak`` gives its cumulative peak since it
    started (``peak_scope == "cumulative"``), which is an upper bound on the probe's. ``watch``
    may have ``sample()``, which is called before and after. Returns the record; ``result`` is
    what ``stage_core`` returned."""
    if "probe" in kwargs:
        raise TypeError("scale_probe passes probe=True itself")
    try:
        params = inspect.signature(stage_core).parameters
    except (TypeError, ValueError) as exc:
        raise TypeError("scale_probe needs an inspectable stage function") from exc
    if "probe" not in params and not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    ):
        raise TypeError(f"{stage_core!r} has no probe keyword; it is not a stage core")
    started = time.monotonic()
    before = watch.sample() if hasattr(watch, "sample") else None
    windowed = hasattr(watch, "reset_window") and hasattr(watch, "window_peak")
    if windowed:
        watch.reset_window()
    result = stage_core(*args, probe=True, **kwargs)
    if hasattr(watch, "sample"):
        watch.sample()
    peak = (watch.window_peak if windowed else watch.peak) / GIB
    limit = float(ceiling_gib) - float(margin_gib)
    record = {
        "stage_core": f"{getattr(stage_core, '__module__', '?')}."
        f"{getattr(stage_core, '__qualname__', repr(stage_core))}",
        "tree_gib_before": None if before is None else before.get(watch_measure(watch)) / GIB,
        "probe_peak_tree_gib": peak,
        "peak_scope": "probe" if windowed else "cumulative",
        "cumulative_peak_tree_gib": watch.peak / GIB,
        "ceiling_gib": float(ceiling_gib),
        "margin_gib": float(margin_gib),
        "within_margin": bool(peak <= limit),
        "seconds": time.monotonic() - started,
        "result": result,
    }
    if not record["within_margin"]:
        raise GuardError(f"G-memory-margin: the scale peak {peak:.2f} GiB is above {limit:.2f} GiB")
    return record


def watch_measure(watch) -> str:
    return getattr(watch, "measure", "pss")
