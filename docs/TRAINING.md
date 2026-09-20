# Bounded model training

The training runner uses sealed `DatasetStore` train/validation splits and the
same canonical batches for either backend. It never decodes test or held-out
samples for training, model selection, or diagnostics. Opening a dataset still
hash-checks every recorded shard for corpus integrity.

```sh
.venv/bin/python -m embodied_jepa.training \
  --dataset data/pilot --backend native_jepa \
  --output checkpoints/pilot/native.pt \
  --steps 500 --batch-size 16 --horizon 4 --seed 0 \
  --device cpu --max-seconds 600
```

For the external adapter, fetch its pinned source first with
`.venv/bin/python scripts/fetch_lewm.py`, then change `--backend leworldmodel` and
use a distinct output path. Backend-specific architecture defaults are recorded;
the runner uses the registry and contains no backend-specific training logic.
`--model-config path.json` supplies explicit backend config overrides.
Model dependencies, contracts, and baseline differences are in [MODELS.md](MODELS.md).

Python callers can use:

```python
from embodied_jepa.training import train
report = train(dataset, backend, output, steps=500, batch_size=16,
               horizon=4, seed=0, device="cpu", max_seconds=600,
               validation_every=50, validation_batches=4,
               model_config=None, memory_limit_gib=16, selection="raw_mse")
```

## Data and budgets

Episodes from train and validation are decoded once into a bounded cache. Uniform
sampling with replacement over all eligible training windows uses compact
cumulative window counts, not a duplicated image window cache. Every sampled
window stays inside one episode; optional state masks and executed actions pass
through unchanged. Session/episode overlap with validation, test, or holdout is
rejected. The runner does not refit or mutate dataset normalization. Both visual
models estimate their latent-distance variance using training batches only.

The default process host-memory ceiling is 16 GiB; lower it with
`--memory-limit-gib`. A decoded-data estimate reserves three times the raw corpus
size, with additional batch-copy/conversion checks. Oversized corpora fail with a
request for a smaller corpus instead of repeatedly decoding the full dataset.
Host RSS and elapsed time are checked at episode and update/validation-batch
boundaries. These checks are not an OS-enforced allocator limit: an individual
operation can temporarily exceed the limit before the next check. A single model
update/validation batch is not interrupted halfway through. Final checkpoint and
report writes may run after the time budget to preserve evidence.

`max_seconds` counts dataset/model setup, training, and validation after initial
run metadata setup. Steps, batch size, horizon, validation frequency/cohort size,
seed, and resource limits are fixed before training. CPU Torch threads are capped
at four during the run and restored afterward. MPS is explicit, must be available,
and is synchronized around measured model operations. Reported host RSS is the
process-lifetime high-water mark; final MPS allocator/driver bytes are separate
snapshots, not GPU peak-memory estimates.

## Selection and artifacts

An untrained validation baseline is measured at step zero. Validation then runs
every `--validation-every` updates and at the final requested update. Fixed
validation window cohorts are drawn with seeds derived from the training seed
(`seed + 10001 + horizon`); their hashes are logged. No test outcomes affect this
selection. Recursive prediction MSE at the configured training horizon chooses
checkpoints **within each backend**. Raw MSE from different representations must
not be compared as task performance. Horizons 1, 4, and 8 also report model-owned
collapse/action/persistence controls where complete validation windows exist;
missing horizons are explicitly marked unavailable.

An explicit `--selection noncollapsed_relative` alternative uses horizon-4
validation diagnostics. Eligibility requires `collapsed_fraction <= 0.05` and
`latent_std_mean >= 0.1`. Among eligible states it minimizes
`prediction_mse / max(persistence_mse, 1e-12)`, retaining the earlier state on an
exact tie. Every validation event records eligibility, rejection reasons, score,
and whether it became the best checkpoint. Unavailable horizon-4 diagnostics are
ineligible. If no checkpoint qualifies, a completed optimization budget produces
`selection_failed` with no best checkpoint; the latest state and diagnostics
remain available. This explicit selector corrects the collapsed low-MSE state
observed in the development pilot; see the preregistered
[v1 protocol](experiments/reach_pilot_v1.md). The default remains `raw_mse` so
historical commands and v0 results keep their original meaning.

For output `checkpoints/pilot/native.pt`, the runner writes:

- `native.pt`: best eligible checkpoint under the declared selector, possibly
  step zero. It exists only after a complete eligible validation pass.
- `native.latest.pt`: final available model/optimizer state, including a partial
  budget-limited run. A failed run's latest state must not be assumed valid.
- `native.metrics.jsonl`: every completed update and validation event, retaining
  negative results and synchronized update timing.
- `native.run.json`: status, requested/completed budget, best step/error, model
  configuration, hashes, environment, resource measurements, and artifact paths.

Existing artifact paths are never overwritten. Status distinguishes `completed`,
`time_budget`, `memory_budget`, `selection_failed`, and `failed`. Numerical/runtime errors are recorded
then raised; budget-limited CLI runs exit with status 2. When a budget stops the run
between validation events, best refers to the last best measured checkpoint and
latest can be newer. Final partial validation is not presented as a complete
cohort. Model and runner state are kept separate from held-out benchmark results.

Checkpoint provenance contains the dataset manifest hash, split-policy hash,
action-manifest hash, Git revision, and Python source-tree hash. The run records
whether its checkout was dirty and carries dataset/license provenance through to
the report. Each checkpoint includes the model RNG plus the NumPy sampler state
and completed step in `metadata.runner_state`. This supports a future explicit
resume command; the current CLI always starts a new run and does not silently
resume an old output. Whole-experiment replay uses the recorded source, fixed
configuration, and immutable corpus. Checkpoint loading should supply expected
hashes to enforce provenance checks.

`tests/test_training.py` uses labeled procedural software fixtures. It exercises
both actual model adapters, fixed validation cohorts and best/latest association,
no test decoding, one-time episode caching, deterministic sampling, time/memory
limits, preserved failure records, leakage rejection, and the relative selector's
collapse eligibility and within-model persistence normalization. Those checks do not
establish learned robot dynamics, noncollapse on the real corpus, or manipulation
success; those require the subsequent recorded simulation experiment.
