# MVP plan

Prepared 2026-09-20 from PRD sections 1–21, updated with the user’s Mac-only / MuJoCo-first constraint. The copied PRD remains unchanged; this plan records the platform refinement. MissionControl holds execution status; this document holds scope and milestone gates.

> **Status as of 2026-09-24.** The M0–M5 engineering milestones below were delivered and
> audited ([ACCEPTANCE.md](ACCEPTANCE.md)); the research outcome was negative. **Learned
> Apple→Plate is still 0 successes.** Since TASK-054 the planning line in this plan is no
> longer the primary control line: CEM over the world-model cost is abandoned in favour of
> behaviour cloning with the world model as a critic. The current position, what is
> demonstrated and what has been ruled out are summarised in the
> [README status section](../README.md#status--2026-09-24); the decision and its evidence
> are in [DECISIONS.md](DECISIONS.md). The milestone table below is kept as the original
> plan of record and is not rewritten.

## Deliverable

Run goal-image tabletop manipulation in G1 + Dex3 MuJoCo simulation on the local Mac using two trained, action-conditioned world models. Switch only `world_model.backend` to compare them with the same data, action space, CEM/MPC budget, task, seeds, and evaluator. Store reproducible machine-readable results. Prepare the Unitree SDK2 transport mapping and mock-based checks, plus an Isaac Lab/Sim port specification. Executing those integrations requires future access to suitable hardware/compute and is not a prerequisite for the local MVP.

The final simulation task is Apple → Plate. Reach and grasp are intermediate milestones, not substitutes for the final task. Demonstrating a functioning benchmark and measuring its performance is the engineering MVP. Successful zero-shot generalization is a research hypothesis, not a promised outcome.

## Scope

- Required models: project-owned `native_jepa` and first external candidate `leworldmodel`, subject to the compatibility spike.
- Optional follow-up: `jepa_wms` research integration, with its licensing isolated from the core.
- Hardware contract: G1 EDU4, both Dex3 hands, existing onboard RGB, arm/hand state, optional end-effector poses; no new camera requirement.
- Simulation: native MuJoCo Python bindings on macOS, fixed-base or otherwise stabilized tabletop setup, after verifying or composing suitable G1 EDU4/Dex3 MJCF assets. Do not require the upstream Linux DDS simulator launcher. Right-arm-first task curriculum, with the left arm held in a defined pose; both arms remain represented in the API.
- Dataset: one canonical LeRobot-compatible G1 corpus shared across models; simulation collection is the fallback when demonstrations are unavailable.
- Planning: image goal, CEM, receding horizon MPC, short multi-step predictions from the first complete rollout interface.
- Evaluation: common mode for the MVP; native mode must have a distinct result label and is a follow-up.

Exclude locomotion, whole-body control, language goals, video generation, tactile sensing, multi-robot work, production deployment, and large foundation-model pretraining.

## Milestones

Estimates are engineering effort ranges for one experienced contributor, not delivery promises. They exclude hardware procurement, access delays, and unattended training. Re-estimate after M0; model learning and dexterous manipulation may require additional experiments.

| Milestone | Tasks | Estimated effort | Exit evidence |
| --- | --- | --- | --- |
| M0 — Resolve feasibility | 001–004 | 3–5 days | Mac/data inventory, pinned upstream candidates, MuJoCo G1/Dex3 asset/rendering smoke test, frozen v0 contracts |
| M1 — Reproducible foundation | 005–008 | 5–8 days | Installable core, registry, dataset fixture and loader, tested action normalization and limits |
| M2 — First vertical slice | 009–014 | 10–15 days | Simulation data → native training → image-goal CEM/MPC → reach benchmark with stored results |
| M3 — Backend interchangeability | 015–017 | 5–8 days | LeWM trained on the same corpus; config-only swap; identical harness and budget |
| M4 — Apple → Plate benchmark | 018–020 | 8–12 days | Grasp/transport/place curriculum, sealed generalization split, comparative full-task report |
| M5 — Future ports and handoff | 021–023 | 4–7 days | SDK2 mapping checked with mocks, Isaac port spec, physical commissioning procedure, clean-environment reproduction and acceptance audit |

Overall initial estimate: **35–55 engineering days**, with M0 as the first checkpoint. Hardware and Isaac runtime commissioning are follow-ups conditional on access; mark all unexecuted checks explicitly. Optional TASK-024 (JEPA-WMs), TASK-025 (Isaac implementation), and TASK-026 (physical commissioning) do not block the local MVP.

## Execution order

Start TASK-001 (resources), TASK-002 (external adapter feasibility), and TASK-004 (contracts). TASK-003 validates MuJoCo rendering and assets on the local Mac. Build a CPU-testable foundation and measure MPS compatibility before selecting training sizes; no CUDA dependency may enter the local critical path.

The main dependency path is contracts → configuration/data/action foundation → simulation collection → native model + CEM/MPC → LeWM → common benchmark → manipulation curriculum → final evaluation. SDK2 preparation can begin once the embodiment contract and simulation adapter exist. Dependencies are recorded on each MC task; `mc task next` selects work whose prerequisites are complete.

Each task must produce its stated artifacts and pass its acceptance checks before being marked done. Move implemented work through `review` when useful. Keep unavailable resources explicit in task notes instead of pretending a gate passed.

## PRD acceptance mapping

| PRD §19 requirement | Tasks | Required evidence |
| --- | --- | --- |
| G1 + Dex3 embodiment adapter | 008, 009, 021 | Simulation execution, both-hand/arm mapping, SDK2 transport checks |
| LeRobot-compatible G1 data loads | 007, 010 | Canonical transitions/sequences and manifest hashes |
| Native JEPA trains | 011, 014 | Reloadable checkpoint, loss/collapse/action-sensitivity report |
| External backend integrated | 002, 015 | Pinned source/license inventory and trained LeWM adapter |
| Same WorldModel API | 004, 006, 016 | Shared conformance suite passes both backends |
| Action-conditioned prediction | 011, 015, 016 | Changed/shuffled actions alter predictions and affect quality |
| Shared CEM planner | 012, 016 | Same planner implementation and configuration |
| Goal-image planning in simulation | 013, 014, 018, 020 | Closed-loop logs and Apple → Plate rollouts |
| Identical benchmark for both | 017, 020 | Frozen manifest, task, budgets, seeds, and resolved configs |
| Common result format | 017 | Validated per-episode JSONL and aggregate report |
| Configuration-only backend change | 006, 016, 020 | Backend-key-only comparison test and archived configs |

TASK-023 audits all rows. Engineering completion never implies that the hypothesis of reusable physical dynamics was proven.

## After the MVP: the task-specific apple line

The engineering MVP closed with a complete negative benchmark. Work since then is a
separately labelled task-specific mode on the apple corpus, not the historical unseen-pair
benchmark, and it is recorded protocol-by-protocol under
[docs/experiments/](experiments/) with machine-readable manifests under
[benchmarks/manifests/](../benchmarks/manifests/). Every gated **learned** closed-loop attempt in
that line has failed its declared gate, and the fresh 20-reset final cohort (TASK-034) has
not been executed. Some non-learned privileged-ceiling diagnostics did pass their gates;
they are feasibility evidence, not learned-policy results.

The last of those protocols, [world model v4](experiments/apple_world_model_v4_results.md)
(TASK-054), failed all four arms and fired its pre-declared abandonment clause. The next
line is behaviour cloning with the world model as a critic; it is not designed or started
in this plan.
