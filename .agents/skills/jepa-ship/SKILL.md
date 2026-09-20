---
name: jepa-ship
description: Land an authorized, reviewed Open Embodied JEPA PR, verify the remote merge, and reconcile MissionControl evidence. Use when asked to ship, merge, or complete an end-to-end delivery; not for robot deployment, artifact releases, or review-only requests.
---

# Land a reviewed JEPA change

Follow [AGENTS.md](../../../AGENTS.md) from its project root. Shipping here means merging a repository change. It does not authorize publishing checkpoints/datasets, deploying services, or executing a robot. Preserve existing user authorization; do not ask again when the requested delivery already includes the merge.

## Resolve and inspect

Find the task's explicit PR link or match its exact branch and task reference. Verify repository, PR number, base, head branch, and head SHA. If multiple candidates remain, resolve the ambiguity before mutating anything; do not choose the first search result. Reuse an existing PR and authenticated `gh`; there is no `gh-bot` or `git-push-bot` dependency here.

Read current reviews, comments, unresolved review threads, and repository merge requirements. Confirm that findings are addressed and applicable approvals exist. Author review is not independent approval. Fix bounded issues and repeat `$jepa-review` within the authorized scope; substantial new scope returns to planning. Answer or post comments only when authorized as part of the delivery, not merely because a comment exists.

## Verify the revision that will merge

Check the current head commit's check runs and commit statuses, including pagination where needed. Recheck the PR head after each push or base update; old green checks do not validate a new SHA. Pending checks are pending, failed required checks block merging, and missing evidence is not success. Follow branch rules and do not use administrative bypasses.

This project initially has no application CI. If no checks are configured or required, say so and use the documented local validation for the change. If configured checks have not registered or are still running, wait within a reasonable bound and report the unresolved state; do not poll forever or claim an empty list means green.

Inspect local status and avoid destructive cleanup. Update the branch from its base when necessary for conflicts, freshness requirements, or validation; do not create a gratuitous merge commit on every invocation. Resolve conflicts carefully, then rerun the affected checks and review. Keep the PR description aligned with the final diff, including task updates.

## Keep task completion truthful

Do not mark a task done just because a merge is planned. If its deliverable-based acceptance criteria are already satisfied, the PR may include `done` plus the actual evidence and an explicit note that merge is still pending. If a criterion depends on the merged state, keep `review` until the merge is verified and update the tracker through a follow-up PR. A tracker-only follow-up for the same task needs no new task that recursively demands its own close-out.

Keep historical specifications and results. Do not delete a task's spec to close it, infer parentage from `depends_on`, or invent unsupported MC fields. Close a documented parent only when every child and the parent's own acceptance criteria are satisfied; a canceled prerequisite is not automatically successful completion.

## Merge and verify

Prefer squash merge for a single task; preserve a deliberate commit series only when justified. Use GitHub's expected-head protection when available, for example `gh pr merge <number> --squash --match-head-commit <verified-sha>`, substituting actual values. If the head changed, review and validate the new revision before retrying. If permission, required approval, or a persistent check blocks the merge, report the concrete condition without bypassing it.

Verify remotely that the PR is merged and record its merge commit. A branch-cleanup error can occur after a successful merge: inspect the PR state before retrying. Fast-forward a clean local `main`, verify any configured checks for the resulting commit, and remove only merged branches/worktrees owned by this task and free of unrelated work. Never force-delete an active worktree to make cleanup appear successful.

Report PR URL, merge SHA, completed validation, MC state, and any remaining limitation. Scientific claims require their experiment evidence even when the code merged cleanly.
