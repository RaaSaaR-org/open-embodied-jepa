---
id: TASK-028
aliases:
- TASK-028
title: Adapt development skills for Codex robotics research workflows
slug: adapt-development-skills-for-codex-robotics-research-workflows
status: done
priority: 2
owner: ''
projects: []
customers: []
tags:
- governance
- skills
sprint: ''
depends_on: []
due_date: ''
created: 2026-09-20
updated: 2026-09-20
---



# Adapt development skills for Codex robotics research workflows

## Description

Adapt the five workflow concepts from robot-management-system/.claude/skills into project-local Codex skills. Preserve the source files and make the workflows appropriate for Mac/MuJoCo development and empirical JEPA research.

## Acceptance Criteria

- [x] Five namespaced skills exist in .agents/skills with valid SKILL.md frontmatter and Codex UI metadata.
- [x] Workflows cover research design, bounded experiments, controls/provenance, CPU/MPS and simulator evidence, scoped review, and verified authorized merges.
- [x] Original Claude-specific dependencies and forced delegation are removed; supported MC fields and current project contracts are used.
- [x] AGENTS.md and usage documentation explain invocation/discovery; current parent workspace symlinks resolve without changing global skills.
- [x] Skill validation, YAML/reference checks, MC validation, and diff checks pass; author review and limitations are recorded.

## Notes

Source workflow concepts: grill, plan, implement, review, ship. Adapted skills: jepa-grill, jepa-plan, jepa-implement, jepa-review, jepa-ship. This is a deliverable-based task: no training, simulator execution, or independent review is claimed.

## Validation and author review

- The skill-creator quick validator passed for all five skills using an isolated `uv run --no-project --with pyyaml` environment; project dependencies were unchanged.
- Parsed all UI YAML; confirmed skill/folder names, supported frontmatter, default implicit discovery, valid local references, and five resolving parent-workspace symlinks.
- `mc validate`, `mc index`, and `git diff --check` passed.
- Author walkthrough: unknown experimental outcomes become bounded investigations; a negative result can meet research acceptance without proving engineering success; review-only prompts do not authorize mutations; missing CI is distinguished from passing checks; a changed PR head requires revalidation; hardware execution is not implied by shipping code.
- Reviewed software/workflow consistency, every acceptance criterion, and the applicability of research evidence. No blocking finding remains. This is author review and structural validation, not independent agent testing, runtime training validation, or a live test of skill selection in a fresh Codex session.

PR: https://github.com/RaaSaaR-org/open-embodied-jepa/pull/1

All deliverable-based acceptance criteria are met on the reviewed branch. Merge is pending at this commit; completion here records the skill/documentation deliverable and does not assert a merged state. The PR provides the authoritative merge record.
