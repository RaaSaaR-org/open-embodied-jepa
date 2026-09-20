# Upstream and artifact inventory

Initial discovery and feasibility audit checked on 2026-09-20. The overview below records initial candidates; adopted revisions, versions, terms, and execution evidence follow in the audit sections. No third-party source, weights, or robot assets are vendored here.

| Component | Role | Source-code license evidence | Artifact status |
| --- | --- | --- | --- |
| Original framework / native_jepa | Required core | [Apache-2.0](../LICENSE); dependency attribution maintained as components are adopted | Newly trained weights/data terms recorded separately |
| LeWorldModel | First external candidate | [Official repo](https://github.com/lucas-maes/le-wm), [MIT LICENSE](https://raw.githubusercontent.com/lucas-maes/le-wm/main/LICENSE) | Checkpoint and dataset terms unverified |
| JEPA-WMs | Optional research | [Official repo](https://github.com/facebookresearch/jepa-wms), [CC BY-NC 4.0 LICENSE](https://raw.githubusercontent.com/facebookresearch/jepa-wms/main/LICENSE) | Checkpoints and encoders need separate entries |
| LeRobot | Canonical dataset compatibility | [Official source](https://github.com/huggingface/lerobot); selected revision/license audit pending | Dataset provenance is independent of library license |
| MuJoCo Python bindings | Required local simulator | [Official source](https://github.com/google-deepmind/mujoco), [Python documentation](https://mujoco.readthedocs.io/en/stable/python.html); selected revision audit pending | Verify macOS stepping, rendering, and viewer in TASK-003 |
| Unitree MuJoCo assets | Candidate G1/Dex3 MJCF and meshes | [Official source](https://github.com/unitreerobotics/unitree_mujoco); exact asset provenance/terms pending | Complete dual-Dex3/EDU4 compatibility has not been established |
| Unitree Isaac Lab environment | Future G1/Dex3 simulator port | [Official source](https://github.com/unitreerobotics/unitree_sim_isaaclab); full audit pending | G1-29dof-dex3 is listed; EDU4 correspondence and asset terms unverified |
| Unitree SDK2 / Python interface | Physical transport | [SDK2](https://github.com/unitreerobotics/unitree_sdk2), [Python interface](https://github.com/unitreerobotics/unitree_sdk2_python); selected revision audit pending | Firmware/robot compatibility unverified |
| PyTorch, array/config/test libraries | Planned macOS CPU/MPS implementation dependencies | Versions, selected package licenses, and transitive inventory pending TASK-005 | No packages installed by this scaffold |
| Isaac Lab / Isaac Sim and robot assets | Future optional simulator environment | Follow selected upstream environment's supported version matrix; audit separately | Port requirements in TASK-022; runtime verification in TASK-025 |

The official LeWM repository uses its own training and planning support libraries and HDF5 datasets. The adapter spike must determine the smallest integration surface and translate canonical batches internally; it must not force the shared dataset or CEM to change. [LeWM upstream README](https://github.com/lucas-maes/le-wm)

For every adopted component record name, URL, commit/version, SPDX identifier or exact license reference, role, required/optional status, attribution/NOTICE requirements, and review date. For weights and data add hashes, producer, source-model dependencies, redistribution terms, and training-data provenance where known. An upstream source license does not establish the license of its checkpoints.

## Adopted Python environment (2026-09-20)

`uv.lock` records exact versions and artifact hashes, including transitive requirements. Required core: NumPy and PyYAML. Learning, simulation, and minimal LeWM dependencies are separate extras; no mandatory Isaac/SDK2 runtime. CPython 3.12.13 is the tested interpreter.

| Distribution | Tested version | Installed license metadata | Scope |
| --- | --- | --- | --- |
| einops | 0.8.2 | MIT | optional runtime |
| mujoco | 3.13.0 | Apache-2.0 | optional runtime |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | required core |
| pillow | 12.3.0 | MIT-CMU | optional runtime |
| pytest | 9.1.1 | MIT | development |
| PyYAML | 6.0.3 | MIT | required core |
| ruff | 0.16.8 | MIT | development |
| torch | 2.14.0 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | optional runtime |
| transformers | 4.57.6 | Apache 2.0 License | optional runtime |

`scripts/dependency_inventory.py` exports installed metadata for all transitive packages; CI archives this report for each supported core platform. Metadata is evidence for review, not a substitute for each distribution’s bundled license/NOTICE. No third-party package source or binary is vendored. Preserve package license files when redistributing environments. No upstream dataset, pretrained encoder or checkpoint is adopted in the feasibility probes.

## LeWM feasibility audit (TASK-002, 2026-09-20)

The optional checkout is pinned to [`lucas-maes/le-wm` at `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`](https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac). Its [MIT license](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/LICENSE) names Lucas Maes (2026); retain that license/copyright with redistributed source or substantial portions. Source is fetched under ignored `third_party/`, not vendored into the Apache-2.0 core. The probe imports unchanged `jepa.py` and `module.py`; source hashes and device evidence are in the regenerated JSON described in [LEWM_SPIKE.md](LEWM_SPIKE.md).

| Component | Exact revision/version and evidence | Adoption / terms |
| --- | --- | --- |
| LeWM source | `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`, MIT | Optional checkout; core model classes tested on CPU/MPS |
| ViT encoder | transformers 4.57.6, Apache-2.0; architecture constructed from configuration | Random initialization only; no pretrained encoder or weight download |
| stable-pretraining | [`9aa93f8b6153eebb73f57d4853ccf8a13d848310` LICENSE](https://github.com/galilai-group/stable-pretraining/blob/9aa93f8b6153eebb73f57d4853ccf8a13d848310/LICENSE), MIT | Audited encoder factory; not installed/imported/copied. Full trainer dependencies excluded |
| stable-worldmodel | [`4821c8e6a3f0f83b7e6a80da3a757e026ea9026b` LICENSE](https://github.com/galilai-group/stable-worldmodel/blob/4821c8e6a3f0f83b7e6a80da3a757e026ea9026b/LICENSE), MIT | Not installed/imported; its dataset/environment/planner APIs are not needed |
| PushT pretrained LeWM | [`quentinll/lewm-pusht` model card at `22b330c28c27ead4bfd1888615af1340e3fe9052`](https://huggingface.co/quentinll/lewm-pusht/blob/22b330c28c27ead4bfd1888615af1340e3fe9052/README.md), publisher metadata says MIT | Not downloaded/adopted; no weight hash or provenance audit, no claim that source license grants checkpoint rights |
| PushT dataset | [`quentinll/lewm-pusht` dataset card at `655cd446b9929369d7d406001da85c15d1457850`](https://huggingface.co/datasets/quentinll/lewm-pusht/blob/655cd446b9929369d7d406001da85c15d1457850/README.md), publisher metadata says MIT | Not downloaded/adopted; no dataset hash or upstream asset provenance audit |
| Cube / two-rooms / reacher / baseline checkpoints | Linked from the upstream README | Not needed or adopted; artifact terms and provenance remain unverified |

The installed macOS transitive closure of torch/einops/transformers was inspected via `importlib.metadata` requirements with platform markers evaluated and extras disabled. Beyond direct dependencies listed above: tqdm 4.70.1 (MPL-2.0 AND MIT), safetensors 0.8.0 and tokenizers 0.22.2 (Apache), huggingface-hub 0.36.2 and hf-xet 1.6.0 (Apache), typing-extensions 4.16.0 (PSF-2.0), requests 2.34.2 (Apache-2.0), certifi 2026.7.22 (MPL-2.0), urllib3 2.8.0 (MIT), idna 3.20 (BSD-3-Clause), charset-normalizer 3.5.1 (MIT), packaging 26.3 (Apache-2.0 OR BSD-2-Clause), fsspec 2026.9.0 (BSD-3-Clause), filelock 4.0.1 (MIT), regex 2026.9.10 (Apache-2.0 AND CNRI-Python), Jinja2 3.1.6 (BSD), MarkupSafe 3.0.3 (BSD-3-Clause), networkx 3.6.1 (BSD-3-Clause), sympy 1.14.0 and mpmath 1.3.0 (BSD), setuptools 84.0.0 (MIT). PyYAML and NumPy are also in that closure. These are installed package declarations, not a complete legal audit of all bundled native libraries. The lockfile and generated inventory preserve exact dependency evidence; retain bundled license/NOTICE files on redistribution. No source/weight/data is republished by the probe.

## Unitree simulation assets (TASK-003)

The adopted dual-Dex3 asset revision, SHA-256 hashes, and BSD-3-Clause license provenance are recorded in [assets/manifest.json](../assets/manifest.json) and [MUJOCO_SPIKE.md](MUJOCO_SPIKE.md). Native MuJoCo 3.13.0 stepping, deterministic reset, RGB rendering, and the macOS passive viewer passed. EDU4 serial-specific calibration and physical contact fidelity remain unverified. Current upstream rubber-hand assets are not substituted for the pinned articulated model.
