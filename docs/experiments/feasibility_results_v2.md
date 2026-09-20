# Completed feasibility runtime comparison

The preregistered TASK-030 comparison completed **20/20 episodes in 363.41 seconds**, below its 600-second cap. Both unchanged checkpoints completed all five projection-on episodes with 100 commands each. All **1,000/1,000** scored, requested and accepted commands matched exactly. There were no projected guard stops or deadline misses. All ten matched projection-off episodes stopped at joint-rate guards.

| Backend | Off mean commands | On mean commands | On planning p50 / p95 | On control p95 | Reach | Full success |
|---|---:|---:|---:|---:|---:|---:|
| Native JEPA | 9.8 | 100 | 0.104 / 0.110 s | 0.123 s | 1/5 | 0/5 |
| LeWM | 12.8 | 100 | 0.362 / 1.133 s | 1.144 s | 0/5 | 0/5 |

No episode achieved grasp, transport, place or release. The optimization resolves the bounded diagnostic's runtime incompleteness; it does not establish useful learned manipulation or 20 Hz real-time control. The original five reused resets remain development data. All 20 attempts are counted.

Fixed scratch profiling showed 2.13–2.14× speedup with bit-identical projected outputs. The full comparison used the same frozen models, images, reset seeds, task thresholds and candidate budgets as v1. Different completion/timeout patterns mean pooled v1 and v2 timing samples have different coverage; the fixed matched profile is the cleaner speedup estimate. No long training or collection ran concurrently. Brief software tests overlapped the early comparison, so host timing is development instrumentation rather than isolated hardware certification.

Execution revision: `3ddb09243648ad0be06248a45de7442fb1783419`. Python source SHA-256: `84220c56d89db7b609b117a827216ea5cc5794eb7d4a5e76e1cc0eb060d9014e`. Comparison SHA-256: `fe94959def219e59657e8d69c3e0d1e2db4621ef0eee2d12d027313969e5d56b`.

The detached execution worktree is `../jepa-feasibility-v2`; complete traces and manifests remain in its `outputs/feasibility-v2/`. `registration.json` stores the v2 protocol hash written before execution. The reused runner also records its preserved v1 base-protocol hash; the v2 addendum explicitly keeps those controls binding. Original v1 artifacts are untouched. The [compact manifest](../../benchmarks/manifests/feasibility-results-v2.json) retains all 20 metrics and input hashes.

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_feasibility.py \
  --source-sha256 84220c56d89db7b609b117a827216ea5cc5794eb7d4a5e76e1cc0eb060d9014e \
  --output outputs/feasibility-v2
```

Reproduction requires the exact committed execution revision and a fresh output directory.
