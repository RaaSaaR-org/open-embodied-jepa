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

## Shared backend-agnostic extensions

`VisualModel.defaults` in `models/base.py` carries options written once in shared code so
that enabling one stays a config change rather than a backend branch. **Every one is
inactive at its default** — the flags are off, and the numeric settings beside them only
take effect once their gating flag is on — so a model written before an option was added
keeps its exact latents, its exact predictor and its exact loss. TASK-054 verified that
end-to-end for the three prediction-step keys: re-training a pre-TASK-054 configuration at
the new revision reproduced all 170 weight tensors bit-for-bit (`torch.equal`).

| Key | Default | Added by | What it does |
| --- | --- | --- | --- |
| `state_fusion` | `False` | TASK-050 | Adds a learned embedding of train-normalized proprioception to every latent |
| `readout_heads`, `readout_weight`, `readout_hidden_dim` | `False`, `1.0`, `256` | TASK-050 | Declared physical readout heads and their loss weight |
| `cameras` | `None` | TASK-052 | Multi-camera fusion; `None` keeps the single `camera` |
| `readout_moving_weight`, `readout_moving_threshold_m`, `readout_auxiliary_weight` | `0.0`, `0.01`, `0.0` | TASK-052 | Extra readout-loss weight on frames whose true offset has moved, plus auxiliary absolute-position readouts |
| `action_chunk` | `1` | TASK-054 | Conditions each rollout step jointly on a chunk of future actions |
| `predictor_step_embedding` | `False` | TASK-054 | Adds a learned per-step vector, making the rollout horizon-conditioned |
| `multistep_tail_weight` | `0.0` | TASK-054 | Ramps the multistep loss towards the late steps at constant total weight |

**None of the five options that have been ablated has been shown to help.** In the
[v3](experiments/apple_world_model_v3_results.md) and
[v4](experiments/apple_world_model_v4_results.md) gated comparisons the second camera made
readout precision worse, the readout shaping hurt candidate ranking, and all three
prediction-step options failed to beat the untouched control, which passed more gates than
every intervention. (*Erratum 2026-09-25, TASK-058
[S4-09](experiments/claim_audit_v1.md):* the control passed one more gate, G6a at h = 16.
At the planner's h = 8 it is not reproduced, on 10 groups, which is below the cohort rule. Of
the gates every arm passed, G4 is cleared by a constant, and G3 and G5 by a true-state
copy-last, so the count is not a quality ranking.
v3's "readout shaping" was one bundled factor, and "hurt ranking" rests on one seed.)
`state_fusion` and `readout_heads` have **not** been ablated on/off in
those protocols — they are untested rather than shown not to help, and `readout_heads` is
the apparatus the physical-readout gates are measured through. All of them are kept because
the negative results are part of the record and the options are inactive by default;
enabling one is a research choice that needs its own preregistration, not a recommended
setting.

### Frozen external encoder (`frozen_encoder`, TASK-065)

`frozen_encoder: dinov2_small_cls` swaps the backend's trainable image encoder for the pinned,
frozen DINOv2 ViT-S/14 CLS feature of `pretrained_encoder` (TASK-063: 112 → 224 px bicubic, no
crop, ImageNet normalisation, `pooler_output`, CPU float32). It was preregistered by
[apple_latent_dynamics_v1.md](experiments/apple_latent_dynamics_v1.md). Its result
([results](experiments/apple_latent_dynamics_v1_results.md)) is outcome WM-NO-DYNAMICS, and the
clause for the pooled-CLS predictor line fired. The latent-error gates G2–G4 passed on all
seeds. G1 failed on an uncalibrated rank bar, and G5 (apple readability through prediction)
failed.

- **It is off unless a run selects it.** It is written once, backend-agnostically, in
  `models/frozen_encoder.py`: `frozen_encoder_model("leworldmodel")` (or `"native_jepa"`) puts a
  mixin in front of the backend class.
  - The plain backends reject the key as unknown, and the frozen class refuses to run without it.
  - It is **not** in `VisualModel.defaults`. `models/base.py`, the backend files, `readout.py` and
    `readout_labels.py` are unchanged. That keeps every earlier checkpoint's
    `implementation_sha256` valid, including TASK-054's E0 checkpoint, which
    `tests/test_policy_preflight.py` guards.

- **The latent is the feature, standardised.** Per-dimension mean and std are fitted once on
  training frames only (`fit_frozen_feature_normalization`, std floored at 1e-3), before any
  update. There is no projector, so a prediction maps back to the raw feature exactly
  (`predict_features`). An external probe fitted on the raw feature can therefore read
  predictions.
- **The encoder never trains.** It is not a submodule, not in the optimizer, not moved to MPS and
  not in the checkpoint's weights. Its weights digest is in the model metadata, which the
  checkpoint saves and `load` compares, so a load under a different frozen encoder is refused.
  `implementation_sha256` also covers `frozen_encoder.py`, the backend file and
  `pretrained_encoder.py`.
- **The backend's training step and predictor are unchanged.** The same configuration trains
  native or LeWM, and the backend swap stays one key.
  - `train_step(batch)` encodes frames on the fly.
  - `train_step_features(features, actions)` hands cached raw features to the backend's own
    `train_step` in place of the frames' features. Because the encoder is frozen, this is the same
    update, bit for bit (tested for both backends).
- The backend's own encoder and projector are still built but receive no gradient.
- It requires `latent_dim` 384 and refuses `state_fusion`, `readout_heads` and a `cameras` list.
- LeWM's SIGReg shapes a trainable encoder's embedding; with a frozen encoder, TASK-065 sets its
  weight to 0.
- Planners still receive only `VisualLatent`. `predict_features` is an evaluation path for
  probes, and it exists only for a frozen encoder.

### Frozen patch-token latent (`frozen_encoder: dinov2_small_tokens`, `token_grid`, TASK-066)

`frozen_encoder: dinov2_small_tokens` with `token_grid: g` makes the latent the pinned DINOv2
ViT-S/14 final patch tokens (TASK-063's `tokens` read-out point), average-pooled from the 16 × 16
patch grid to `g × g` and standardised by train-only moments. The latent is flat,
`g² × 384`-d, row-major over (grid row, grid column, channel). It is preregistered by
[apple_token_dynamics_v1.md](experiments/apple_token_dynamics_v1.md) with `g = 4` (6144-d) and
has **no result yet**.

- **It is off unless a run selects it.** It is written once, in `models/frozen_tokens.py`.
  `frozen_token_model("leworldmodel")` (or `"native_jepa"`) puts `FrozenTokenMixin` in front of
  the backend class.
  - `FrozenTokenMixin` **subclasses the TASK-065 `FrozenEncoderMixin`** and keeps its
    normalisation, its feature injection, `train_step_features` and `predict_features`.
  - The plain backends and the CLS class reject `token_grid` as unknown. The token class refuses
    to run without both keys.
  - No existing file changes, so every earlier `implementation_sha256` stays valid.
- **The prediction step is a token predictor.** It maps the grid of one frame and that frame's
  action to the grid of the next frame. The backend's own `train_step`, `rollout`, losses and
  optimizer run unchanged on the flat latent; only `next_embedding` reshapes it to tokens.
  - **LeWM**: the pinned upstream `ARPredictor`, `Embedder` and `pred_proj`, composed by upstream
    `JEPA.predict`. The token grid is the sequence axis, with one learned position embedding per
    token. The action embedding conditions every token through AdaLN-zero. Attention is
    **bidirectional within the frame**: the upstream attention modules are used with their mask
    fixed to non-causal, and no upstream byte changes.
  - **Native**: a residual token step. One shared MLP reads each token plus its position
    embedding, the frame's mean token and the action.
- **Nothing of the backend's own image pathway is built.** At a 6144-d latent, LeWM's ViT encoder
  would be about 0.9 B parameters. The frozen feature replaces the pathway.
- It refuses `state_fusion`, `readout_heads`, a `cameras` list and every TASK-054 prediction-step
  option.
- **`pool_tokens` is layout-independent, but the tokens it pools are not.** Batched float32 CPU
  inference differs slightly between batch sizes (TASK-065 run-1). A run that caches features
  must check its own cache path for determinism, partial batches included.

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
