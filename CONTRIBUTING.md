# Contributing

Open Embodied JEPA is currently a planning scaffold for a Mac-first MuJoCo MVP. Start with the [MVP plan](docs/MVP_PLAN.md), [setup notes](docs/SETUP.md), and [AGENTS.md](AGENTS.md). The latter defines the shared engineering and research workflow for human and automated contributions.

Choose or create a MissionControl task, work on a topic branch, make focused commits, and open a PR with the supplied template. Include relevant validation and evidence, address review findings, and merge only when the applicable checks and approvals are satisfied. Initial repository bootstrapping is the sole exception to the normal PR path.

For implementation, preserve the separation between world models, planners, embodiments, datasets, and tasks. For experiments, make the hypothesis and controls explicit and publish reproducible configurations and honest results. Small, inspectable contributions are preferred over broad rewrites.

Application packaging and CI are planned in TASK-005. Until then, do not assume the example configurations or module directories are executable. Current planning checks are:

```sh
git diff --check
mc validate
mc index
```

Contributions to original project code and documentation are under [Apache-2.0](LICENSE). Preserve attribution and separately document third-party licenses; do not submit private data, credentials, or artifacts you lack permission to redistribute.
