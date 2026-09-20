# Final MVP training results

All six preregistered runs completed 3,000 updates, with no runtime failures,
timeouts, retries, or checkpoint-selection failures. The complete experiment took
**770.600 seconds (12 minutes 50.6 seconds)** within the shared 1,080-second cap.
These are validation results; physical manipulation is evaluated separately.

The unchanged training protocol is [mvp_final.md](mvp_final.md). Its recorded
SHA-256 is `2dc397836088932a38ffcd2fc2f39499417654d3d2eff4465ec919be75b63c06`.
Code revision: `e57e28b53eb682102b882dbada937fbcefffb01f`.
Python source hash: `0e9adee4051660dfe03695ef0e7f18e1432a996d6a4833357e580afa06319bb0`.
The checkout was marked dirty because other files were being prepared; the source
and protocol hashes stayed unchanged across all six runs. Corpus manifest SHA-256:
`200246b34bd18cb30c8c711879c42a5ed2f31a62c99b96c1b347e37aa57512b7`.

Each row is the checkpoint selected solely by the fixed horizon-four validation
criterion. Both online and target collapsed fractions were **0.0 for every selected
checkpoint**, and both spaces passed the standard-deviation guard. Predictions,
persistence and shuffled-action errors below are MSE within that model’s learned
target space. Raw errors and ratios must not be used to rank unrelated latent
spaces across backends.

| Backend | Seed | Best step | Prediction | Persistence | Shuffled actions | Pred./persist. | Target effective rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| native_jepa | 0 | 2800 | 0.00648616 | 0.00943985 | 0.00878270 | 0.687104 | 2.9225 |
| leworldmodel | 0 | 1200 | 0.01124475 | 0.00596798 | 0.01408226 | 1.884182 | 2.7118 |
| native_jepa | 1 | 2900 | 0.01020854 | 0.02572132 | 0.01252389 | 0.396890 | 3.4102 |
| leworldmodel | 1 | 2800 | 0.01116172 | 0.00849721 | 0.01312466 | 1.313574 | 3.5588 |
| native_jepa | 2 | 2500 | 0.01086711 | 0.01620965 | 0.01329154 | 0.670410 | 2.6576 |
| leworldmodel | 2 | 1400 | 0.01075260 | 0.01052756 | 0.01265091 | 1.021376 | 2.5948 |

Native JEPA predicts better than persistence and shuffled actions on all three
validation seeds. Its prediction/persistence ratio is 0.584802 ± 0.162950 (mean ±
sample standard deviation, n=3). LeWM predicts better than shuffled actions on all
three seeds but **worse than persistence on all three**: ratio 1.406377 ± 0.438825.
This negative result is preserved; no alternative seed/checkpoint is substituted.
Effective ranks of only 2.59–3.56 show that passing per-dimension collapse guards
does not establish a rich or full-rank representation. Neither validation loss
nor action sensitivity demonstrates successful physical control.

Validation cohorts contain the same fixed 64 windows per available horizon, with
identical cohort hashes between backends for each seed. Training reports record
`test_samples_loaded=false` for all six runs. The assembled corpus contains no
Apple→Plate episodes, and inherited test partitions were not reassigned. No
successful release-v1 episode happens to be in validation; this frozen limitation
was disclosed before final training and was not repaired by reshuffling.

## Compute and artifacts

Elapsed time below is the outer supervisor’s suspension-aware clock, including
imports and checkpoint finalization. RSS is the process-lifetime host high-water
mark in decimal MB. Native runs used approximately 514–515 aggregate process CPU
seconds, LeWM 157–158; aggregate CPU time is not wall time. A short independent
100-update smoke check and tests overlapped part of training, so these timings
are **not isolated throughput measurements**.

| Backend | Seed | Supervised seconds | Peak RSS MB | Online std mean | Target std mean |
|---|---:|---:|---:|---:|---:|
| native_jepa | 0 | 175.836 | 1843.7 | 0.90085 | 0.89927 |
| leworldmodel | 0 | 79.771 | 1901.3 | 0.83485 | 0.83485 |
| native_jepa | 1 | 176.280 | 1850.8 | 0.91647 | 0.92322 |
| leworldmodel | 1 | 79.852 | 1896.0 | 0.90999 | 0.90999 |
| native_jepa | 2 | 176.390 | 1814.7 | 0.85347 | 0.85038 |
| leworldmodel | 2 | 79.519 | 1898.2 | 0.86352 | 0.86352 |

Local artifact root: `checkpoints/mvp-v0/`. Each `seed-N/BACKEND.pt` is the
selected checkpoint; adjacent `BACKEND.latest.pt` preserves update 3000.
`BACKEND.run.json` contains full per-horizon validation, eligibility decisions,
configuration, provenance and resource measurements; `BACKEND.metrics.jsonl`
contains training curves and `BACKEND.log` the subprocess output.
[experiment.json](../../checkpoints/mvp-v0/experiment.json) records all six exact
commands and supervisor outcomes.
[training_summary.json](../../checkpoints/mvp-v0/training_summary.json) captures
the complete selected-checkpoint metrics, source/cohort hashes and both best/latest
checkpoint hashes. These generated local artifacts are ignored by Git, so the
links resolve only in a checkout containing the recorded experiment.

| Selected checkpoint | SHA-256 |
|---|---|
| `seed-0/native_jepa.pt` | `1ad2d1327472b8df28dc57b13ab89c069fa53dfcd7bf81b70a944b5b5310ae9c` |
| `seed-0/leworldmodel.pt` | `0351b6e7e6026c03743c324fc1146ede9d84062c8258c81a64c2a34cd209fe33` |
| `seed-1/native_jepa.pt` | `62125697464d56bcb79ece37008d6afd81b62505f0fab369b5aaa13510b1c33a` |
| `seed-1/leworldmodel.pt` | `ab00894e48c1fac0c862fa5b37cb648f0d673704d6afb839980c2f45611b2dd5` |
| `seed-2/native_jepa.pt` | `54489d27fe4a02d3d3707c916884700252af48041001bcb71266a6ad34454d1d` |
| `seed-2/leworldmodel.pt` | `9d8d0711636b249606cc507bc792e97b2ba485c2c879ffb71105347d3d1aabcd` |
