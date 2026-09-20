# Upstream and artifact inventory

Initial discovery checked on 2026-09-20. Links point to upstream sources; exact commits, versions, transitive dependencies, and artifact terms must be captured by TASK-002/003/005 before installation or integration. No third-party source, weights, or robot assets are vendored here.

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
