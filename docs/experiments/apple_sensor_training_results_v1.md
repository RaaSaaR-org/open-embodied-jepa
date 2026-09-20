# First task-specific sensor world-model training result

The fixed run completed all **3,000 updates in 30.46 seconds** (supervised total), below its 1,800-second cap. Raw validation image prediction MSE selected **step 2,100**. No test samples were decoded by training or checkpoint selection. The model uses fixed spatial RGB preprocessing and learned visual/proprioceptive residual dynamics; this is not an unsupervised JEPA or LeWM result.

| Recursive horizon | Selected prediction MSE | Persistence MSE | Shuffled-action MSE | Prediction / persistence | Improvement over shuffle |
|---|---:|---:|---:|---:|---:|
| 1 | 0.00295159 | 0.00854583 | 0.00298610 | 0.345 | 1.16% |
| 4 | 0.00795995 | 0.04389348 | 0.00805476 | 0.181 | 1.18% |
| 8 | 0.01224935 | 0.04049790 | 0.01238151 | 0.302 | 1.07% |

Errors are in this model's training-normalized RGB feature space, not directly comparable with other model latent losses. The fixed validation cohort has 64 windows per horizon. Prediction beats persistence, but the roughly 1% advantage over shuffled actions is weak evidence of useful action conditioning. Measured state/velocity may explain much of the improvement. Phase-specific diagnostics and closed-loop model ablations are required before claiming a working world-model controller. No learned physical task success has been measured by this training run.

This corpus has 26 training episodes, three validation episodes and only one failed training trajectory. Narrow reset distributions, weak action interventions and limited failure-recovery coverage restrict the learned counterfactual dynamics. Fixed-feature inactive background dimensions are not a learned representation-collapse test.

Protocol: [apple_sensor_training_v1.md](apple_sensor_training_v1.md). Exact source, model configuration, dataset/split/action identities, validation cohorts, supervision timings and all artifact hashes are in the [compact training manifest](../../benchmarks/manifests/apple-sensor-training-v1.json). Full local logs/checkpoints remain under `checkpoints/apple-sensor-v1/`.

Execution revision: `27ec3bb` (exact full revision in the manifest). Python source SHA-256: `a5cff9442d959ad0418dcb13e4cb59b9f54bbf46758b9638b9b9a4af85df1796`. Selected checkpoint SHA-256: `3a5c5e7ba77556b6ce4251f05af72dc86df0121807d913ca8b444db60026e3ee`. Best and latest states are both preserved.
