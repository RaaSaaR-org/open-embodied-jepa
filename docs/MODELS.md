# Visual world model adapters

`NativeJEPA` and `LeWM` implement the same frozen `WorldModel` contract. They accept
canonical `SequenceBatch` data and normalized `ee_delta_grasp_v0` actions, predict
all future steps recursively, and return finite NumPy float32 image-goal costs
`[batch, candidates, horizon]`. Both are explicitly **visual-only baselines**:
robot state is validated for batch/schema identity but ignored by the encoder.
Neither needs future robot state or goal proprioception. These implementations
are not evidence of successful learned manipulation.

## Construction and configuration

```python
from embodied_jepa.models import NativeJEPA, LeWM

model = NativeJEPA(
    state_schema,
    device="cpu",  # or explicitly validated MPS; no silent fallback
    seed=0,
    config={"camera": "onboard_rgb", "candidate_chunk_size": 128},
    metadata={"dataset_hash": dataset_hash, "action_hash": action_hash,
              "source_revision": code_revision, "split_hash": split_hash},
)
metrics = model.train_step(batch)  # one optimizer update
current = batch.observation(0)
latent = model.encode(current.images, current.state)
goal = model.encode_goal(goal_images)
prediction = model.predict(latent, candidate_actions)
costs = model.distance(prediction, goal)
```

Constructors are `(state_schema, device='cpu', seed=0, config=None, metadata=None)`.
Classes are exported lazily so importing the package does not require optional
PyTorch or transformers installations. Unknown config keys fail early. Backend
selection belongs to the registry/runner; shared planners must not inspect latent
payloads or branch on backend names.

| Setting | Native default | LeWM default |
| --- | --- | --- |
| camera | `onboard_rgb` | `onboard_rgb` |
| image_size | 64 | 56 |
| latent_dim / hidden_dim | 64 / 128 | 48 / 128 |
| learning_rate / weight_decay | 0.0003 / 0.0001 | 0.0003 / 0.0001 |
| max_horizon / candidate_chunk_size | 64 / 128 | 64 / 128 |
| gradient_clip | 1.0 | 1.0 |
| observation history | one | one |
| multistep_weight | 1.0 | 1.0 |

All resize/channel/normalization operations stay within the model. Actions already
use the shared `[-1,1]` scale; neither adapter fits another action normalization.
Goal distance is mean squared latent error divided by per-feature variance
(clamped at 0.01), estimated with a 0.05 moving-average update on training batches
only and saved in the checkpoint. This makes a backend's feature scaling explicit;
it does not make latent costs comparable across different backends.

Native JEPA uses a three-layer convolutional image encoder retaining a 4×4 spatial
grid, a residual action-conditioned MLP predictor, and a frozen EMA target encoder.
Its loss combines one-step and recursive prediction MSE with embedding variance
and off-diagonal covariance penalties. Defaults: `ema_decay=0.99`,
`variance_weight=1.0`, `covariance_weight=0.01`. Goals use the EMA target encoder;
current observations use the online encoder. Noncollapse penalties are not proof
that collapse is avoided: measure the saved diagnostics on held-out episodes.

## Optional LeWM source

```sh
uv sync --extra learning --extra lewm
.venv/bin/python scripts/fetch_lewm.py
```

The fetcher creates an ignored checkout at the audited revision and refuses to
modify an existing differing checkout. `LeWM` verifies the revision and exact
SHA-256 of its source/license files before importing actual upstream `JEPA`,
`ARPredictor`, `Embedder`, `MLP`, and `SIGReg` classes. It never downloads pretrained
weights or imports upstream planners/datasets/training frameworks. Retain the
upstream MIT license when distributing those sources; see
[the license inventory](DEPENDENCIES.md) and [compatibility spike](LEWM_SPIKE.md).

The small configuration constructs a random Hugging Face ViT with
`patch_size=14`, `encoder_depth=2`, `encoder_heads=3`, `predictor_depth=2`,
`predictor_heads=2`, and `predictor_head_dim=24`. Additional settings are
`sigreg_weight=0.09`, `sigreg_projections=128`, and
`source_path='third_party/le-wm'`. Input size must divide by patch size and latent
dimension by encoder heads. ImageNet pixel statistics match the upstream input
convention. Upstream BatchNorm projectors require at least two sequences per
training batch; inference batch one is supported.

The adapter deliberately uses history one, zero dropout, fewer layers/projections,
and a recursive loss term in addition to the upstream next-embedding loss and
SIGReg. Targets remain attached as in LeWM. These are declared local-budget
research choices, not a reproduction of the paper's architecture/results. Set
`multistep_weight=0` to remove that extra objective term. Action-conditioning gates
start at zero in upstream AdaLN: sensitivity should be checked after optimization.

## Validation, diagnostics, and checkpoints

`validation_error(batch)` reports recursive within-backend latent MSE.
`diagnostics(batch)` reports that MSE plus zero-action, shuffled-action, and
persistence controls; action-effect RMS; mean/minimum embedding standard
deviation; and fraction of dimensions with standard deviation below 0.01.
Shuffling is a deterministic roll across batch/time actions. The result records
whether actions actually changed, since a constant-action batch cannot establish
conditioning. These are descriptive controls; physical success remains the main
common benchmark metric. Validation never updates normalization or model weights.

Inference uses evaluation mode and no gradients, with candidate chunking inside
the adapter. Latents are owned by one model instance and invalidated after a
training update or checkpoint load. This prevents accidental cross-model or stale
latent reuse. Keep CPU/MPS choice explicit and time MPS with synchronization in
the runner. Backend construction/training preserves the process's Torch RNG;
separate internal CPU/MPS states are maintained for reproducible resume.

`save(path)` writes weights, optimizer, EMA/BatchNorm/variance buffers, configuration,
ordered state/action schemas, seed, update count, backend RNG states, upstream
revision, implementation hash, preprocessing identity, and supplied provenance.
Loading uses `torch.load(weights_only=True)` and rejects different backends,
configurations, schemas, upstream revisions, implementation source hashes, and any
caller-declared metadata mismatch before applying weights. Construct with the same
config (including LeWM source path), then call `load`; the device may differ.
Unspecified caller metadata is restored from the checkpoint, so evaluation code
must supply expected dataset/action/split hashes when it needs those checks.
No learning-rate scheduler is used. Data-sampler/global experiment RNG belongs to
the runner and must be recorded separately for full experiment replay.

`tests/test_models.py` exercises both real adapters on synthetic canonical RGB
sequences: recursive shapes and causality, chunk equivalence, finite training,
action sensitivity after updates, masks/state independence, EMA updates,
checkpoint prediction identity and exact CPU optimizer/RNG resume, schema and
provenance rejection, stale-latent rejection, upstream source integrity, and
available MPS forward/backward/checkpoint behavior. It skips optional dependencies
or unavailable MPS explicitly. It does not substitute tests for corpus training,
collapse/generalization assessment, or goal-conditioned manipulation evaluation.
