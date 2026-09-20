# Local resources and execution budget

Measured 2026-09-20 with `python scripts/resource_probe.py` after `uv sync --locked --extra learning --extra sim --extra lewm`:

- Apple M5 Pro, arm64, macOS 26.5.1; 48 GiB physical memory.
- 536.6 GiB available storage at the first implementation probe (changes as artifacts accumulate).
- CPython 3.12.13; PyTorch 2.14.0; CPU and MPS matrix forward/backward both produced finite loss/gradients. This is a capability check, not proof every model operation supports MPS.
- Probe peak process RSS: 339,329,024 bytes (macOS process measurement, not GPU allocation).
- No recordings in the project; no supplied demonstration paths, physical robot connection, camera calibration, or remote compute. Inventory scope is the workspace, not a search through unrelated personal files.
- Intended robot: G1 EDU4 with dual Dex3. Simulation model correspondence is documented in `MUJOCO_SPIKE.md`; no claim of verified physical EDU4 calibration.

Use ignored `data/`, `checkpoints/`, and `outputs/` in this repository for local artifacts; `third_party/` holds pinned, unmodified upstream checkouts. Record hashes and small reports in versioned manifests/docs. No artifact upload is implied by the source publishing workflow.

Initial operational limits: one simulator, four Torch CPU threads, up to 16 GiB process RSS, 20 GiB generated data, and 10 minutes per exploratory training run. Begin with small batches and a bounded smoke cohort; measure before scaling. These are conservative limits against measured host capacity, not observed final training requirements. Stop and diagnose nonfinite losses or memory pressure. The final cohort and training budget must be frozen separately before comparative testing.

Reproduce the capability report with `.venv/bin/python scripts/resource_probe.py`; output is `outputs/feasibility/resources.json`. Record dependency metadata with `.venv/bin/python scripts/dependency_inventory.py > outputs/feasibility/dependencies.json`.
