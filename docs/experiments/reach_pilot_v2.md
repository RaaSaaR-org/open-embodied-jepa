# Reach pilot v2 — target-space diagnostics and longer fixed training

Declared before training. V0 selected a collapsed native checkpoint; v1 corrected
that selector but still compared current **online** embeddings with future **EMA
target** embeddings in the persistence baseline. On the selected native v1 state,
that reported baseline was 0.46038; same-target-encoder persistence was 0.00004533,
versus prediction error 0.005207. Thus the apparent persistence advantage did not
establish useful native dynamics. Both v1 learned reaching runs achieved 0/5.
Preserve those negative outcomes and all v0/v1 artifacts.

Hypothesis: a longer fixed optimization budget lets the native EMA representation
stabilize enough for useful action-conditioned prediction. This remains a
research hypothesis; failed prediction controls or reaching are valid outcomes.
No architecture, preprocessing, loss, optimizer, or backend default changes are
included. Both backends remain visual-only, H1, with their prior compact defaults.

## Metric definition version 2

Persistence now compares `goal_embed(current image)` against
`goal_embed(future image)` for every future step. Prediction is scored against the
same future target embeddings. Report online and target embedding mean/minimum
standard deviation and collapsed fraction separately. Collapsed dimensions have
population standard deviation below 0.01 across the validation batch/time samples.
Old `latent_std_*` and `collapsed_fraction` fields remain aliases for **online**
statistics, with explicit version 2 and separate target statistics.

Report target effective rank as the exponential entropy of normalized squared
singular values of centered target embeddings, computed by float64 CPU SVD for
small matrices. An exactly constant matrix has rank zero. Large matrices explicitly
mark rank unavailable rather than silently estimating it; no rank threshold enters
this experiment's checkpoint eligibility.

The explicit `noncollapsed_relative` selector at **horizon four** now requires
both online and target spaces to have mean standard deviation >= 0.1 and collapsed
fraction <= 0.05. Among eligible states minimize corrected
`prediction_mse / max(persistence_mse, 1e-12)`. Ties keep the earlier state. Log the
metric definition version, eligibility, rejection reasons, and score. No eligible
state means selection failure/no best checkpoint; retain latest and diagnostics.
A ratio above one is a negative prediction result, even if the state is selected
as the best among eligible checkpoints. Scores are not comparable between models.

## Frozen run configuration

- Corpus: `data/reach-pilot-v0`, manifest SHA-256
  `9a5256cc2070d2bc8c44f44b741830e428a33682c5baf0719c7973c62231804f`.
- Existing 80/10/10 whole-session split, training seed 0, no test/holdout decoding.
- **3000 updates per backend**, batch 16, horizon 4, CPU four threads, maximum
  **600 seconds per run**. No extensions after observing outcomes.
- Validation at step zero, every **100 updates**, and final step; same fixed four
  batches of 16 windows per available diagnostic horizon 1/4/8. Cohort seeds remain
  `seed + 10001 + horizon`.
- Selector `noncollapsed_relative`, metric definition version 2, fixed guards above.
- Sequential native then LeWM; save under `checkpoints/reach-pilot-v2/`.

```sh
.venv/bin/python -m embodied_jepa.training --dataset data/reach-pilot-v0 \
  --backend native_jepa --output checkpoints/reach-pilot-v2/native_jepa.pt \
  --steps 3000 --batch-size 16 --horizon 4 --seed 0 --device cpu \
  --max-seconds 600 --validation-every 100 --selection noncollapsed_relative
.venv/bin/python -m embodied_jepa.training --dataset data/reach-pilot-v0 \
  --backend leworldmodel --output checkpoints/reach-pilot-v2/leworldmodel.pt \
  --steps 3000 --batch-size 16 --horizon 4 --seed 0 --device cpu \
  --max-seconds 600 --validation-every 100 --selection noncollapsed_relative
```

Reaching evaluation keeps the fixed five development seeds, same goals, controls,
CEM/MPC bounds/budgets, and scoring from prior pilots. Record failure reasons and
prediction/shuffled-action/persistence controls, not only success rates. This is
iterative development validation and cannot be presented as an independent test
of generalization. Apple→Plate remains a separate final manipulation requirement.

Changing diagnostics changes the strict implementation hash, intentionally making
v0/v1 checkpoints incompatible with the new model class. Reproduce historical
model behavior with commit `304abb0` and the preserved v1 runner/selector with
`7676619` (both also precede `0ee5817`). Do not rewrite old checkpoint hashes to
force loading under changed code. Keep old outputs and their original metric
semantics; metric definition version 2 applies only to new runs.

## Post-run portability correction

V2 trained and validated diagnostics on CPU only. Its exact source is preserved
in commit `772b5ab`; both learned reaching runs scored 1/5 on the development
cohort (hold/random 0/5, oracle 5/5), which is not evidence of reliable control.
A later MPS-only diagnostic check found that a fused MPS→CPU float64 conversion
produced corrupted values in the installed Torch/macOS combination. The new code
transfers to CPU **before** casting to float64, with a regression comparing both
backends' MPS effective ranks against an independent NumPy CPU reference. This
changes the strict implementation hash; reproduce V2 checkpoints with `772b5ab`
instead of rewriting them. No architecture/optimizer changes or V2 retraining
were made for this portability fix.

New runner versions also budget `max(wall-clock elapsed, monotonic elapsed)` so
host suspension cannot silently extend a run whose monotonic clock pauses.
Reports separate wall-clock, monotonic, and aggregate process CPU time. Historical
V2 timings retain their original perf-counter definition; they are not retrofitted
with measurements that were not captured.
