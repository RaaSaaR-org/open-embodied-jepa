"""Run Arena's own ``policy_runner.py`` unchanged, with a non-remote policy (development spike).

Container-only, via ``scripts/isaac/run_isaac.sh arena_policy_runner.py outputs/<new> <runner
args>``; ``run_isaac.sh`` appends ``--output``, which is dropped here (the runner writes nothing
but its log, which ``run_isaac.sh`` tees to ``log.txt``). Example: the tutorial's evaluation
command with ``--policy_type zero_action`` in place of the GR00T remote policy, and
``--headless`` in place of ``--viz kit`` (docs/ARENA.md). No GR00T server is started or contacted.
"""

from __future__ import annotations

import os
import runpy
import sys

RUNNER = "/workspaces/isaaclab_arena/isaaclab_arena/evaluation/policy_runner.py"


def strip_output(argv: list[str]) -> list[str]:
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--output":
            skip = True
            continue
        if a.startswith("--output="):
            continue
        out.append(a)
    return out


if __name__ == "__main__":
    sys.path.insert(0, "/oej/src")
    from embodied_jepa.arena_transport import pin_pxr_work_thread_limit

    print(f"[arena] PXR_WORK_THREAD_LIMIT pinned to {pin_pxr_work_thread_limit()}", flush=True)
    args = strip_output(sys.argv[1:])
    if any("remote" in a.lower() or "gr00t" in a.lower() for a in args):
        sys.exit("refusing: this wrapper runs local policies only (no GR00T server)")
    os.chdir("/workspaces/isaaclab_arena")
    sys.argv = [RUNNER, *args]
    runpy.run_path(RUNNER, run_name="__main__")
