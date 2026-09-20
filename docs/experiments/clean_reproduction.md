# Clean Mac reproduction

A detached checkout of commit `4d2588d` and a new `.venv` reproduced installation, fresh simulation collection, both model training/reload paths, a backend-only configuration swap, and actual image-goal MPC. This used the same Mac and `uv`'s wheel cache; it was not a second machine or a cached dataset/checkpoint. All three upstream source/asset checkouts were fetched independently at their declared revisions.

```sh
uv sync --locked --extra learning --extra sim --extra lewm --extra data --extra compatibility
.venv/bin/python scripts/fetch_assets.py
.venv/bin/python scripts/fetch_lewm.py
.venv/bin/python scripts/fetch_lerobot.py
.venv/bin/python scripts/reproduce_smoke.py --name clean-smoke
JEPA_TEST_RENDER=1 LEROBOT_SOURCE=third_party/lerobot .venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
```

Results: **329 tests passed, zero skips**, including actual graphics, CPU/MPS models and the pinned official LeRobot reader. Ruff lint/format passed. The checkout remained clean. The smoke collected 12 episodes / 336 transitions, trained each backend for 100 updates, then completed two five-step benchmark episodes per backend through the shared planner. Both runs ended at their five-step limit with 0/2 successes, as recorded; this tiny integration smoke is not evidence of learned manipulation.

The smoke uses raw-loss selection explicitly. Research comparisons use the separately preregistered noncollapse selector and full corpus. Smoke tests ran alongside part of the final training orchestration; training wall times therefore describe observed development execution, not an isolated CPU throughput contest. Final evaluation is run serially after training.

Small hashes, commands and results are versioned in `benchmarks/manifests/clean-reproduction.json`. Complete local data, checkpoints, logs, images, test XML and dependency inventory are retained under `outputs/clean-reproduction/`. Original temporary absolute paths remain in metadata as provenance; the manifest names their relocated copies. Reproduction from another machine may have small graphics differences and a different dataset hash; frozen artifact bytes remain hash-verifiable.
