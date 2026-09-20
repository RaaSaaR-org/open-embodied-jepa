---
name: jepa-grill
description: Pressure-test an Open Embodied JEPA research idea, experiment, or feature before implementation. Use when the user wants assumptions challenged or the research question, scope, and acceptance evidence clarified; not for routine implementation or code review.
---

# Clarify a JEPA research or engineering task

Produce a decision-ready specification, not an implementation. Follow the project's [AGENTS.md](../../../AGENTS.md); use the project root containing that file even when this skill is discovered through a symlink.

## Establish what is known

Read the relevant MC task, its hard dependencies, and affected contracts. Inspect available code, configurations, data manifests, or results before asking questions the repository can answer. Consult the [platform plan](../../../docs/PLATFORMS.md) and [evaluation protocol](../../../docs/EVALUATION.md) when relevant. Separate observed facts, working assumptions, user preferences, and untested hypotheses.

For research, identify the question, baseline, proposed intervention, observation/action inputs, and evidence that could disprove the hypothesis. For engineering, identify the observable behavior and the contract it must preserve. Some tasks need both.

## Resolve consequential decisions

Ask about choices that change the experiment or implementation: success semantics, data availability, held-out combinations, simulator scope, resource limits, or an unresolved user preference. Give a recommendation and its practical tradeoff. Prefer one pivotal question at a time; use the host's available question tool when appropriate. Continue independent fact-finding while an answer is pending.

Use established project choices without reconfirming them: Mac/MuJoCo first, model-independent CEM/MPC, shared canonical data, and isolated future Isaac/SDK2 ports. Do not demand that every research uncertainty be settled in advance. Convert an empirical unknown into a bounded feasibility task with evidence and a stopping condition. Choose routine reversible details and label assumptions.

Do not run training, install a simulator, or execute robot actions merely to clarify a proposal. A small read-only inspection is appropriate; a new experiment needs to fit the requested scope and available budget.

## Write the specification

Create or update the task with `mc`, preserving supported frontmatter. Record:

- The question or behavior, motivation, scope, and affected interfaces.
- Decisions, material alternatives, unresolved assumptions, and prerequisite evidence.
- Checkable engineering acceptance and a separate research outcome to measure.
- For experiments: control, dataset/split, seeds, metric definitions, compute/time budget, and stopping rule.
- Validation strategy and the artifact that will let another contributor reproduce or evaluate the result.

Link lasting architecture decisions from `docs/DECISIONS.md`; keep experiment-specific detail with its task/report. Preserve prior evidence when decisions change. A negative experiment can satisfy a properly defined research task; do not require a favorable result or retrofit thresholds afterward.

Finish when remaining uncertainty is either resolved or assigned to a concrete investigation. Do not add a ceremonial approval step after the user has already supplied the necessary decisions. If a consequential answer is required, keep dependent work pending. Cancel a task only when the user decides not to pursue it. Hand off to `$jepa-plan` for multiple slices or `$jepa-implement` for one ready task; do not start building solely because clarification is complete.
