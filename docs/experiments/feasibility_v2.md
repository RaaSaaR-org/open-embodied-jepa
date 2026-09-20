# TASK-030 feasibility runtime comparison — preregistration

Before execution, repeat TASK-029's exact 20 planned development episodes and
fixed model checkpoints, original goal pixels/resets, horizon4,64candidates,
3iterations,8elites,100commands and5-second control deadline. The sole new
implementation replaces full scratch forward dynamics with kinematics and center
of mass position updates needed for site poses and Jacobians. Live physics,
actuator guards, model weights, task thresholds and scoring remain unchanged.

Run the existing `scripts/evaluate_feasibility.py` from a detached worktree at the
committed optimized revision with its exact Python source hash, writing fresh
`outputs/feasibility-v2`. No old results are overwritten. Runtime sources and
pinned inputs are checked before and after each attempt by the existing runner.
The runner's existing protocol field references the preserved v1 protocol; this
v2 addendum and its SHA-256 must be stored alongside the new comparison before
execution. All v1 controls, metric definitions and missing-attempt accounting
remain binding. Source and protocol commits are made before starting.

A single600-second true-wall cap includes setup and all20episodes. No retries,
extensions or result-dependent changes. CPU4threads. Other CPU-heavy experiments
will not run concurrently. Record all intended attempts, completed matched pairs,
projection/planning/control timing, guard/deadline stops, scored/accepted command
agreement, physical stages and success. Compare latency to archivedv1 with its
incomplete-cohort limitation; do not claim improvement in learned manipulation
from computational optimization.

Before this run, fixed profiling and permanent regression tests must show
bit-identical projected actions and feasibility to full-forward scratch refresh,
including measured tracking lag. The profile budget is60CPU seconds. The target
is completing20/20attempts within600seconds; report a negative result if not.
