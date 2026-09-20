# Codex workflows for this project

Five project-local skills adapt the workflow concepts from `robot-management-system/.claude/skills` to Open Embodied JEPA. The source skills remain unchanged. The adapted instructions are versioned under `.agents/skills/` and refer to this repository's `AGENTS.md`, MissionControl tasks, and research contracts.

## Choose a workflow

| Source workflow | Codex skill | Output |
| --- | --- | --- |
| `grill` | `$jepa-grill` | A pressure-tested research question or feature specification |
| `plan` | `$jepa-plan` | Bounded engineering/experiment tasks with dependencies and evidence |
| `implement` | `$jepa-implement` | A scoped implementation or authorized experiment with validation |
| `review` | `$jepa-review` | Separate correctness, acceptance, and scientific-validity findings |
| `ship` | `$jepa-ship` | An authorized, verified merge and truthful task completion |

Example prompts:

```text
Use $jepa-grill to pressure-test our action-sensitivity experiment.
Use $jepa-plan to break TASK-003 into Mac-sized feasibility checks.
Use $jepa-implement on TASK-007.
Use $jepa-review to review this branch without changing files.
Use $jepa-ship to merge the reviewed PR for TASK-007.
```

These stages are useful entry points, not mandatory separate conversations. An authorized end-to-end task may continue through them. The skills are normally discoverable automatically; an explicit skill invocation is available when you want a particular workflow.

## Discovery and workspace layout

Open Codex in the `open-embodied-jepa` repository or a subdirectory. Codex discovers repository skills in `.agents/skills` along the path from its working directory to the repository root. Each folder includes `SKILL.md` and small `agents/openai.yaml` UI metadata. The `jepa-` prefix avoids collisions with generic plan/review skills. [Official Codex skill documentation](https://learn.chatgpt.com/docs/build-skills)

If Codex is opened in the parent `JEPA/` workspace instead, that parent does not automatically discover skills below it. For this local workspace, per-skill symlinks under `JEPA/.agents/skills/` point to the versioned project skills. This bridge is outside the repository and is not needed for contributors who open the repository directly. Skill references resolve through the real project path.

New skills should be available on the next turn; if the skill picker does not refresh, reopen Codex in the project directory. No global skills, original Claude files, or global Codex configuration need modification. Codex supports symlinked skill folders. [Official discovery and refresh behavior](https://learn.chatgpt.com/docs/build-skills)

## What changed from the originals

- Replaced Claude paths, slash-command assumptions, npm/Prisma gates, bot Git wrappers, and unavailable named agents with Codex skills, the real project layout, `mc`, and standard Git/GitHub CLI.
- Preserved focused tasks, resumable branches, explicit acceptance evidence, reviewed PRs, and verification of the commit being merged.
- Added hypotheses, controls, bounded run budgets, negative results, seed/config/data provenance, leakage checks, and honest CPU/MPS timing.
- Made MuJoCo on Mac the first platform; Isaac and physical G1/Dex3 remain separate future validations.
- Removed mandatory delegation, fixed reasoning-effort assignments, invented tracker fields, and repeated approval gates. Questions focus on consequential missing decisions; research unknowns become experiments.
- Separated review-only requests from editing/publishing. No skill grants broader execution permissions by matching a request.
- Replaced premature merge claims and forced cleanup with verified remote state and task completion based on actual evidence. An unconfigured CI system is not reported as green.

## Maintenance

`AGENTS.md` remains the shared policy. Keep skill instructions focused on their stage, with relative links to project contracts. Keep names/descriptions precise and normal implicit discovery enabled. Avoid mirroring the same instructions into `.claude/skills` or a global skill directory.

For changes, run the skill-creator's `quick_validate.py` on each skill when available, verify relative references and UI YAML, and review behavior against realistic requests. This structural validation does not prove that a future agent will follow every instruction. Use relevant MC and Git checks, then publish through the standard PR workflow.
