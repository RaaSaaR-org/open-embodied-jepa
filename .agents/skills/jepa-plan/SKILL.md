---
name: jepa-plan
description: Decompose an Open Embodied JEPA feature, integration, or research experiment into dependency-linked MissionControl tasks with measurable outputs and bounded resource use. Use for task breakdown and experiment planning, not to execute the planned work.
---

# Plan executable research and engineering slices

Follow [AGENTS.md](../../../AGENTS.md) from its project root. Read the task, [MVP plan](../../../docs/MVP_PLAN.md), relevant contracts, and existing implementations or manifests before estimating work. Reuse existing tasks instead of creating duplicates.

## Classify and reduce uncertainty

Distinguish an implementation task, a data/asset preparation task, and an empirical experiment. Name the concrete output for each: an adapter with conformance evidence, a versioned corpus, or a reproducible result that answers a question.

An unanswered engineering fact may warrant a short spike. An unresolved user preference may need a question. An unknown research result is expected and does not prevent planning: define a control, metrics, budget, and stopping condition. Use `$jepa-grill` only when substantive scope or purpose still needs pressure-testing.

Preserve the Mac/MuJoCo critical path. Make Isaac runtime and real-hardware commissioning conditional follow-ups. Verify rather than assume MPS operator support, asset completeness, dataset rights, and action compatibility.

## Slice around verifiable outcomes

Prefer the smallest useful end-to-end result, such as canonical batch → prediction → goal cost, or observe → plan → execute → task outcome. A focused schema, loader, dataset, or experiment slice is also valid when it has independently checkable output; do not force web-app layers into every task.

Keep each slice reviewable with a bounded implementation surface or experiment budget. Size based on inspected code and remaining uncertainty; record a rough effort range and its assumptions in the task body when useful. Do not assign an agent model, reasoning effort, owner, or invented story-point scale automatically.

For each slice specify:

- Outcome, relevant paths/contracts, and what stays outside scope.
- Inputs, provenance, hard prerequisites, and resource needs.
- Acceptance evidence and relevant tests or experiment controls.
- Reproduction artifacts: config, hashes, seeds, metrics, and report location where applicable.
- Estimated effort or run budget, uncertainty, and stopping condition.

Preserve the distinction between software acceptance and scientific success. A backend comparison must freeze the shared data/actions/task/CEM budget and evaluator; include failures and negative results. Do not promise a success rate absent a defined prior target.

## Publish the dependency graph

Create blockers first using `mc new task --depends-on ...`. Use `todo` for ready tasks and `backlog` for tasks with unmet prerequisites. Follow current `mc --help` and the repository's actual fields; do not assume legacy `spe`, `effort`, `parent`, or `deferred` support.

Link a parent and its children explicitly in their task bodies if grouping is useful. Keep hard blockers in `depends_on`; dependency is not ownership and does not imply an epic relationship. Do not change the established task schema just to plan work.

Read the created tasks back, verify references and an acyclic graph, then run `mc validate` and `mc index`. Preserve task IDs already used elsewhere. Follow the repository's branch/PR workflow when publishing the plan within the authorized scope. Identify the first unblocked task and relevant next skill; planning alone does not authorize executing all children or starting a long training job.
