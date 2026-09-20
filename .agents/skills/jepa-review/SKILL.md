---
name: jepa-review
description: Review an Open Embodied JEPA change, task, experiment, or PR for software correctness, acceptance evidence, and scientific validity. Use for review or pre-PR validation; a review-only request does not authorize edits, publishing, or merging.
---

# Review engineering and research evidence

Use [AGENTS.md](../../../AGENTS.md), the task acceptance criteria, and relevant [architecture](../../../docs/ARCHITECTURE.md) and [evaluation](../../../docs/EVALUATION.md) contracts. Work from that project root and preserve the requested review scope.

## Establish the exact subject

Resolve the requested task/PR and its base/head commits. Review the topic diff from its merge-base with the intended base branch; include uncommitted changes only when they are part of the requested review. State the range and whether it is an author review or independent review. A dirty working tree alone is not the whole PR.

Read relevant implementation, tests, configuration, manifests, and claimed result artifacts. Distinguish an absent artifact from a failed check and a successful test from an untested assumption. Do not infer correctness from the PR description or tool exit status alone.

## Review three axes separately

**Software and contracts:** Look for incorrect state/action/time/frame semantics, episode-boundary errors, invalid latents or costs, fragile checkpoint loading, model-specific planner logic, accidental future-state requirements, and hidden dependencies that break Mac execution. Check failure paths and whether tests exercise the actual behavior.

**Task acceptance:** Map each criterion to concrete code, a test, or an accessible artifact. Mark met, unmet, or unverified with the reason. Missing evidence does not become a pass. Verify the same data and planner can support a configuration-only backend change when relevant.

**Scientific validity:** Review controls, split/session/goal leakage, normalization fitting, held-out pairings, action sensitivity, collapse diagnostics, matched planning budgets, device-aware timing, complete failure accounting, sample counts, and uncertainty. Check whether the conclusion follows from the measured outcomes; raw latent losses across different backends are not directly comparable. Distinguish mocked, simulated, and physical evidence.

Apply only the axes relevant to the diff; a prose-only change needs no training run. Use specific counterexamples or observable failure conditions for findings. Label uncertain concerns and the evidence needed to resolve them instead of presenting speculation as a proven defect.

## Report and hand off

Order findings by impact: blocking defect, important correction, optional improvement. Include path/line or artifact, triggering condition, consequence, and a concrete remedy when known. Summarize acceptance coverage, checks actually run, and limitations. A clean review should still identify unverified areas.

For a review-only request, return the findings without editing code, changing tracker state, opening PRs, or posting comments. Within an authorized delivery workflow, fix or route blocking findings, rerun affected checks, and review the changed diff again. Use the existing PR if one already matches the task and branch.

When authorized to publish, use the [PR template](../../../.github/pull_request_template.md), a body file with real newlines, and the task link. Include research evidence when applicable and identify author review honestly. Move the task to `review` when the reviewable deliverable exists. Hand off to `$jepa-ship` only for an authorized landing; never manufacture an independent approval or treat this skill as permission to merge.
