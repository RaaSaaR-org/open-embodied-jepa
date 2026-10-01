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
# Arena's registered local policies (release/0.2.1). A dotted --policy_type path could import
# arbitrary code, so only these names are accepted.
ALLOWED_POLICIES = ("zero_action", "replay", "rsl_rl")
# Arena's parser keeps argparse's default allow_abbrev=True, so "--policy_t pkg.X" would reach
# --policy_type past an exact-name check. Every argument that starts with "--pol" must be one of
# the real option names, spelled out (bare or "--name=value").
POLICY_OPTIONS = ("--policy_type", "--policy_config_yaml_path")


def check_args(args: list[str]) -> None:
    """Refuse anything that could reach a policy server (GR00T listens on 5555)."""
    for a in args:
        low = a.lower()
        if "remote" in low or "gr00t" in low or "5555" in low:
            sys.exit(f"refusing {a!r}: this wrapper runs local policies only (no GR00T server)")
        if a.startswith("--pol") and a.split("=", 1)[0] not in POLICY_OPTIONS:
            sys.exit(f"refusing {a!r}: spell out {POLICY_OPTIONS} (no argparse abbreviations)")
    types = [args[i + 1] for i, a in enumerate(args[:-1]) if a == "--policy_type"]
    types += [a.split("=", 1)[1] for a in args if a.startswith("--policy_type=")]
    if not types or any(t not in ALLOWED_POLICIES for t in types):
        sys.exit(f"refusing --policy_type {types}: allowed {ALLOWED_POLICIES}")


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
    check_args(args)
    os.chdir("/workspaces/isaaclab_arena")
    # stdout is a pipe (tee): line-buffer it, or the runner's per-episode and final "Metrics"
    # prints are lost when Kit exits without flushing (seen in arena-runner-zero-1).
    sys.stdout.reconfigure(line_buffering=True)
    sys.argv = [RUNNER, *args]
    try:
        runpy.run_path(RUNNER, run_name="__main__")
    finally:
        print("[arena] policy_runner returned", flush=True)
