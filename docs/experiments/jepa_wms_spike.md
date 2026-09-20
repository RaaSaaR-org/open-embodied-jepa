# TASK-024 optional JEPA-WMs compatibility spike

Preregistered 2026-09-21 before execution. Question: can unchanged upstream visual encoder and action-conditioned predictor modules support the canonical 14D common-mode contract on local CPU without requiring the upstream full runtime?

The bounded probe will use random initialization, synthetic canonical RGB/action sequences, a compact 16px image, 8px patches, 16-wide tokens and one block per transformer. Stop after 60 CPU seconds; no pretrained weights, training corpus, evaluation cohort or simulator will be loaded. Checks: recursive prediction, finite train gradients, action sensitivity, EMA target, checkpoint/optimizer resume, generic CEM integration, foreign import namespace preservation and source tamper rejection. These establish software compatibility only, not learned manipulation or paper reproduction. CPU threading is limited to one for the probe. Run after coordinator confirmation that the apple control benchmark is not active.

Implementation uses upstream VisionTransformer and VisionTransformerPredictorAC; disables proprio tokens, so inference never requires future measured robot state. It reuses this project's explicitly separate EMA/variance/covariance loss recipe. One-frame recursive rollout differs from the paper's full context configuration. No upstream training recipe, pretrained DINO encoder or benchmark results are reproduced.

Source audit: revision `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`, [official source](https://github.com/facebookresearch/jepa-wms/tree/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0). The root [LICENSE](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/LICENSE) is CC BY-NC 4.0. The predictor header refers to MIT, inconsistent with the root license; this integration conservatively treats all imported upstream code as CC BY-NC 4.0. Source hashes and notices are pinned in `src/embodied_jepa/models/jepa_wms_source.json`. Retain upstream attribution/license; commercial use is not granted. Checkpoint/weight terms require a separate review before redistribution. Nothing restricted is copied or bundled into the Apache core.

The upstream pyproject requires Python 3.10 and a large training/simulation stack. This minimal-module adapter does not install that package and tests a different, explicit Python3.12 configuration. Only torch, einops, timm and timm's normal dependencies are needed by the imported module closure. Timm1.0.19 is Apache-2.0; keep its bundled NOTICE/license. The isolated probe installs only timm under ignored outputs with `uv pip install --python .venv/bin/python --target outputs/task024-deps/site --no-deps timm==1.0.19`, reusing already available torch/torchvision/huggingface-hub/safetensors/PyYAML. It does not mutate the shared environment. Exact wheel provenance and execution evidence will be recorded after the probe.

Results below establish CPU software compatibility only. CPU is the only advertised device; MPS/CUDA, upstream full framework compatibility and physical/control performance are unverified.

## Completed compatibility evidence

Nine actual-module tests passed in 1.31 seconds; including imports the probe used 1.46 seconds wall and 1.26 seconds CPU. The initial run had two test-fixture errors (a missing RobotState timestamp and constructing an invalid shape before the intended checkpoint rejection); these were corrected, and all nine passed. The compact model trained for two updates, passed finite diagnostics/action sensitivity, recursive causality and image-only goals, resumed the exact next CPU optimizer update, rejected incompatible checkpoints/foreign latents, and ran through the unchanged CEM planner. An existing unrelated `src` module survived upstream loading, and modified source was rejected even after a cached import. Exact package/source/license evidence is in [jepa-wms-spike.json](../../benchmarks/manifests/jepa-wms-spike.json).

Reproduce in a separate optional environment:

```sh
uv sync --extra jepa-wms
uv run python scripts/fetch_jepa_wms.py --accept-noncommercial-source
JEPA_TEST_THREADS=1 uv run pytest -q tests/test_jepa_wms.py tests/test_models.py -k jepa_wms
```

For the original isolated probe, append `outputs/task024-deps/site` to PYTHONPATH along with this worktree's `src`, and set `JEPA_WMS_SOURCE=outputs/task024-source/jepa-wms`. Source remains outside the wheel and normal core imports do not load torch/timm/upstream modules. The fetch script and backend require explicit noncommercial-source selection; this is an optional research configuration, not a relicensing of upstream material.

Use backend `jepa_wms` with device `cpu`, checkpoint mapping `world_model.checkpoints.jepa_wms`, and settings:

```yaml
world_model:
  backend: jepa_wms
  settings:
    jepa_wms:
      source_path: third_party/jepa-wms
      accept_noncommercial_source: true
      image_size: 32
      patch_size: 8
      token_dim: 32
      encoder_depth: 1
      predictor_depth: 1
      attention_heads: 4
```

`latent_dim` is computed as spatial patch count × token width (512 above); an explicitly incompatible value is rejected. Training accepts `--backend jepa_wms` and the same canonical dataset/config API as the original backends. Saved envelopes retain exact source revision, source/adapter/base/training-recipe hashes through implementation identity, schema, configuration, optimizer and RNG. These compatibility tests do not satisfy any apple pick/place success criterion and do not replace the primary sensor-world-model work.
