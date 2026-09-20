# LeWM Mac feasibility spike

TASK-002, checked 2026-09-20. The unmodified upstream JEPA, action embedding,
autoregressive predictor, and SIGReg classes execute on this Mac's CPU and MPS.
The 14-dimensional action schema is supported. Recursive planning needs only
observed pixels and executed/candidate actions, with no future robot state.
This establishes adapter feasibility, not a trained robotics backend or task success.

## Reproduce

Install the project's LeWM optional dependencies, then use the project environment.
The probe needs PyTorch, einops, transformers, and NumPy; it downloads no weights.
Clone once into the ignored directory and pin the audited revision:

```sh
git clone https://github.com/lucas-maes/le-wm.git third_party/le-wm
git -C third_party/le-wm checkout --detach 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
.venv/bin/python scripts/spikes/lewm/probe.py --profile upstream --output outputs/lewm-upstream-spike.json
.venv/bin/python scripts/spikes/lewm/probe.py --profile upstream --history 1 --output outputs/lewm-h1-spike.json
```

Default `--profile smoke` uses a smaller ViT/predictor and 56px inputs. The
`upstream` profile matches the published tiny encoder and predictor settings
(224px, 192-dimensional embedding, 12 encoder blocks, six predictor blocks),
with action input changed to 14. It has 18,034,518 parameters for history three.
`--steps` accepts 1–10 and defaults to two; this is a bounded compatibility check.
The source revision and clean tracked files are checked before importing it.
Explicit MPS requests fail when unavailable; `all` records unavailable MPS and
still tests CPU. Implicit MPS fallback is rejected.

## Observed evidence

Environment: macOS 26.5.1 arm64, Python 3.12.13, torch 2.14.0,
transformers 4.57.6, einops 0.8.2, NumPy 2.5.3, four CPU threads.
The fixture is deterministic synthetic uint8 RGB, normalized actions, batch two,
seed 3072. No G1 demonstrations, pretrained weights, or simulator truth enter it.

| Check, upstream profile / history three | CPU | MPS |
| --- | --- | --- |
| Two forward/backward/AdamW updates, all existing gradients finite | Pass | Pass |
| Total loss, step one → step two | 0.33385 → 0.20843 | 0.33327 → 0.20814 |
| Four future steps, two candidate sequences, costs `[2,2,4]` | Pass | Pass |
| Upstream rollout equals independently composed recurrence | Pass | Pass |
| Maximum change after changing future actions from −0.5 to +0.5 | 0.000284 | 0.000244 |
| Synchronized two-update duration / rollout duration, seconds | 0.315 / 0.034 | 3.671 / 0.263 |

These single cold, tiny-batch timings are execution evidence, not a hardware
throughput comparison. Loss reduction and nonzero action effects after two
updates do not establish learned physical dynamics, noncollapse, generalization,
or useful planning. MPS and CPU RNG paths differ; numerical losses need not match.
The smaller default smoke profile also passed both devices. JSON artifacts under
ignored `outputs/` record configurations, package versions, fixture and source
hashes, results, and timings; rerun to regenerate them.

The history-one upstream profile also passed every check on CPU and MPS:
18,034,134 parameters, rollout `[2,2,5,192]` including the initial observation,
and four image-goal costs. Action deltas were 0.000451 / 0.000502; two-update
times were 0.215 / 1.919 seconds. This supports the initial single-observation API.

## Adapter mapping

| Project contract | LeWM integration |
| --- | --- |
| Camera uint8 `[B,H,W,3]` | Select documented camera; channels-first float32, resize, ImageNet normalization inside adapter. Sequence training adds time dimension. |
| `encode(observation, robot_state)` | Upstream `JEPA.encode` visual `emb`; explicitly declare unused robot-state fields. Initial integration can configure history one and encode the current image. |
| Action schema A=14 | `Embedder(input_dim=14)`; no fixed action count in predictor. Keep normalized shared actions; frameskip one, no upstream dataset z-score transform. |
| `predict(z, actions[B,K,T,14])` | Tile opaque latents over K; repeatedly apply `action_encoder` and `predict` using only current/predicted embeddings and candidate actions. Return T future embeddings. |
| History three variant | Retain three observed embeddings and two actually executed actions inside adapter history. Upstream `rollout` action length includes those two past actions. Prepend them to T future actions, then discard the three observed embeddings from its result. Reset history on episode boundaries; never pad with hidden future state. |
| `encode_goal(goal)` | Encode the goal image through the same encoder/projector in evaluation mode; no invented target joints. |
| `distance` | Squared embedding distance reduced over D for every future step, shape `[B,K,T]`; shared CEM chooses final-step cost. Upstream `criterion` returns only final costs and is not the shared planner. |
| `train_step` | Canonical T+1 images/T actions, next-embedding MSE plus 0.09 × upstream SIGReg. Targets are not detached in the published objective. AdamW and clipping controlled by documented backend config. |
| `save/load` | Project checkpoint envelope plus weights/config/optimizer/RNG state, schema and split hashes, source revision. Avoid upstream serialized Python-object checkpoints. |

History one is an explicit architecture/config choice supported by the same
upstream classes and avoids expanding the shared `encode` signature. It is not a
claim to reproduce the published history-three benchmark or reuse its checkpoints.
The probe checks history one separately using the same four-step recurrence and
action/goal checks. Temporal aliasing and performance need later experiments.

## Findings to carry into implementation

- Import `jepa.py` and `module.py` from a verified optional checkout. They require
  only torch/einops. Construct the Hugging Face ViT directly; full
  stable-worldmodel environments, planners, Lightning, Hydra, and WandB are not
  needed for the model adapter. The equivalent `vit_hf` factory configuration is
  documented in [stable-pretraining source](https://github.com/galilai-group/stable-pretraining/blob/9aa93f8b6153eebb73f57d4853ccf8a13d848310/stable_pretraining/backbone/utils.py).
- Upstream AdaLN conditioning begins with zero modulation, so action-invariance
  at initialization is expected. Check sensitivity after optimization; do not
  reinterpret nonzero sensitivity as evidence that correct actions help.
- Upstream BatchNorm projectors require more than one item in training; enforce
  adequate effective batch size or document any changed normalization.
- Do not reuse upstream data splitting/normalization blindly: its training script
  computes action statistics before a random split. The project requires grouped
  episode splits and statistics fitted only on training data.
- The probe intentionally skips upstream `get_cost`: it mutates its input,
  assumes upstream keys, and wraps its own rollout convention. Common-mode CEM
  should consume the project adapter's opaque prediction/cost interface.
- Remaining work is the production adapter, checkpoint/conformance tests,
  training on the shared robot corpus, and the frozen manipulation evaluation.

Source evidence: [JEPA](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/jepa.py),
[modules](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/module.py),
[training](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/train.py),
[model config](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/train/model/lewm.yaml).
License and artifact evidence is recorded separately in [DEPENDENCIES.md](DEPENDENCIES.md).
