# Contributing

Open Embodied JEPA is an implemented Mac-first MuJoCo research framework with a negative research record: the software runs, and learned Apple→Plate manipulation has never succeeded. Start with the [status section](README.md#status--2026-09-24), then the [MVP plan](docs/MVP_PLAN.md), [setup notes](docs/SETUP.md), and [AGENTS.md](AGENTS.md). The last defines the shared engineering and research workflow for human and automated contributions.

Choose or create a MissionControl task, work on a topic branch, make focused commits, and open a PR with the supplied template. Include relevant validation and evidence, address review findings, and merge only when the applicable checks and approvals are satisfied. Initial repository bootstrapping is the sole exception to the normal PR path.

For implementation, preserve the separation between world models, planners, embodiments, datasets, and tasks. For experiments, make the hypothesis and controls explicit and publish reproducible configurations and honest results. Small, inspectable contributions are preferred over broad rewrites.

The package is installable and CI runs on Linux and macOS, with a separate macOS integration job. Run the checks for the change you are making; see [SETUP.md](docs/SETUP.md) for the full list and the optional extras:

```sh
git diff --check
mc validate
mc index
uv run --no-sync ruff check src tests scripts
uv run --no-sync ruff format --check src tests scripts
uv run --no-sync pytest
```

Green CI is evidence that the software integrates. It is not evidence that training or robot manipulation works; keep that distinction in PR descriptions, and keep negative results and refuted hypotheses in the record rather than editing them away.

Contributions to original project code and documentation are under [Apache-2.0](LICENSE). Preserve attribution and separately document third-party licenses; do not submit private data, credentials, or artifacts you lack permission to redistribute.
