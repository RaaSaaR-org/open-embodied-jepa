---
name: jepa-implement
description: Implement one scoped Open Embodied JEPA task or run its authorized bounded experiment, including validation and reproducible evidence. Use for model, data, planner, MuJoCo, or adapter work; not for planning-only requests or unrequested long training runs.
---

# Implement a scoped JEPA task

Follow [AGENTS.md](../../../AGENTS.md) from its project root. Read the MC task, dependencies, affected contracts, and applicable module instructions. Use the [setup notes](../../../docs/SETUP.md) for actual commands; this repository starts as a planning scaffold, so a placeholder or proposed command is not an executable feature.

## Prepare the working state

Inspect the checkout and current branch before changing files. Preserve unrelated work. Resume the task's existing local or remote topic branch when present; otherwise branch from an updated clean `main`. Do not recreate a published branch from scratch or switch another worker's shared checkout. Move the task to `in-progress` and record the implementation/experiment scope.

Work directly unless authorized delegation materially helps. When delegation is explicitly requested or otherwise authorized, give independent workers bounded, disjoint file ownership and acceptance criteria; only the coordinating agent changes Git state. Do not require unavailable named agents, select models from a task size, or share mutable package environments across incompatible runtimes.

## Implement through the correct boundary

Use the [architecture contract](../../../docs/ARCHITECTURE.md): model preprocessing and losses stay inside the model adapter; robot frames/IK/limits stay inside the embodiment; CEM/MPC remains model independent; canonical data serves all models. Version changed schemas and reject incompatible checkpoints/configurations early.

Use MuJoCo locally and lazy-load future simulator/hardware integrations. Check CPU/MPS capability instead of introducing CUDA as a local prerequisite. Choose a small fixture or smoke cohort before expensive runs. Do not replace missing Dex3 assets, states, actions, or measurements with silently fabricated substitutes.

For an experiment, record the hypothesis and control before execution, then capture code revision, resolved config, input hashes, seeds, hardware/device, resource budget, metrics, and artifact locations. Respect its stopping rule. Report a negative result rather than changing the test split, success rule, or budget until it becomes positive.

## Validate the behavior

Select checks from the changed contract and acceptance criteria:

- Data: time alignment, episode boundaries, executed-action semantics, split leakage, and training-only normalization.
- Models: finite gradients/predictions, save/load, recursive rollout without future-state leakage, collapse diagnostics, and action sensitivity.
- Planner/control: known-dynamics progress, bounded actions, solver failure, stale observations, and deadline behavior.
- Integration: MuJoCo reset/render/execute and image-goal planning, with simulator truth restricted to evaluation.
- Documentation/skills: relevant structure, references, and workflow consistency; do not invent runtime coverage.

Use the [evaluation protocol](../../../docs/EVALUATION.md) for model comparisons. Record actual checks, failures, missing resources, and device fallbacks. Fix findings within scope; repeated failure caused by an unavailable prerequisite should produce a precise blocker, not an endless retry loop or a claim of completion.

## Deliver

Update task evidence and relevant docs, run MC checks for task changes, and make focused commits after reviewing the staged diff. Follow the authorized repository delivery scope: push and prepare a PR when included, using `$jepa-review` for the final diff and acceptance audit. Do not merge merely because implementation is complete. If the user requested end-to-end delivery, continue through review and the authorized merge instead of forcing a new user turn at every skill boundary.
