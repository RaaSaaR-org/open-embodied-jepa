# Open Embodied JEPA — Compact PRD

## 1. Goal

Build an **open-source, modular robotics framework for testing interchangeable JEPA / latent World Models**.

Primary hardware target:

* Unitree G1 EDU4
* Dual Dex3 hands
* Existing G1 cameras
* Unitree SDK2

Core principle:

```text
Dataset + Robot + Task + Planner
              │
              ▼
       WorldModel API
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
 JEPA-WMs    LeWM    Native JEPA
```

Changing the World Model must **not require changing the robot, dataset, planner, task or evaluation code**.

Long-term goal: support additional embodiments such as SO-101 through adapters.

---

## 2. Primary Research Question

Can an action-conditioned latent World Model learn reusable physical dynamics and use planning to solve manipulation tasks without task-specific VLA fine-tuning?

Primary benchmark:

> **Apple → Plate using G1 + Dex3**

---

## 3. Architecture

```text
RGB Camera ───────────┐
Robot State ──────────┤
                      ▼
              WorldModelBackend
                      │
            latent state z(t)
                      │
Candidate Actions ────┤
                      ▼
              WorldModelBackend
                      │
             predicted future
                      │
                      ▼
                CEM / MPC
                  Planner
                      │
                      ▼
             normalized action
                      │
                      ▼
             EmbodimentAdapter
                      │
                      ▼
                G1 + Dex3
```

World Models predict **future latent representations**, not future pixels.

---

## 4. WorldModel API

All models MUST implement a common interface.

```python
class WorldModel:
    def encode(self, observation, robot_state):
        """Return latent state z."""

    def predict(self, z, actions):
        """Predict future latent state(s) conditioned on actions."""

    def encode_goal(self, goal):
        """Encode goal observation into latent representation."""

    def distance(self, predicted_z, goal_z):
        """Return planning cost."""

    def train_step(self, batch):
        """Perform model-specific training step."""

    def save(self, path):
        ...

    def load(self, path):
        ...
```

Model-specific preprocessing MUST remain inside its adapter.

---

## 5. Initial World Model Backends

### Required

`native_jepa`

Minimal permissively licensed implementation owned by this project.

Purpose:

* reference implementation
* easy experimentation
* minimal dependencies
* commercially usable foundation

### Integration targets

`leworldmodel`

Adapter for LeWorldModel / LeWM architecture.

`jepa_wms`

Adapter/reference integration for Meta FAIR JEPA-WMs.

Use only where upstream licenses permit.

### Future

Architecture MUST allow additional backends without changes to planner or embodiment code.

Examples:

```text
V-JEPA-2-AC
DINO-WM
future JEPA models
custom research models
```

---

## 6. JEPA Objective

Canonical model:

```text
z_t = Encoder(observation_t)

predicted_z_t+1 =
    Predictor(
        z_t,
        robot_state_t,
        action_t
    )

target_z_t+1 =
    Encoder(observation_t+1)
```

Train:

```text
predicted_z_t+1 ≈ target_z_t+1
```

Do NOT require future image reconstruction.

Multi-step prediction MUST be supported eventually:

```text
z_t + [a_t ... a_t+n]
        ↓
World Model
        ↓
predicted z_t+n+1
```

---

## 7. G1 + Dex3 Embodiment

Implement:

```text
embodiments/unitree_g1_dex3/
```

Input state SHOULD support:

```text
RGB
arm joint positions
arm joint velocities
Dex3 joint state
end-effector pose
optional IMU
```

Do not require additional cameras for MVP.

### Normalized Action Space

Preferred initial action:

```text
left_ee:
  dx dy dz
  droll dpitch dyaw

right_ee:
  dx dy dz
  droll dpitch dyaw

left_hand:
  grasp parameters

right_hand:
  grasp parameters
```

Adapter converts:

```text
normalized action
      ↓
IK / retargeting
      ↓
joint targets
      ↓
Unitree SDK2
```

Also support direct joint-space actions experimentally.

---

## 8. Embodiment API

```python
class Embodiment:
    def observe(self):
        ...

    def state(self):
        ...

    def execute(self, normalized_action):
        ...

    def normalize_action(self, robot_action):
        ...

    def denormalize_action(self, action):
        ...
```

World Models MUST NOT contain Unitree-specific control logic.

---

## 9. Dataset

Prefer **LeRobot-compatible datasets**.

Canonical transition:

```python
{
    "observation": image_t,
    "robot_state": state_t,
    "action": action_t,
    "next_observation": image_t1,
    "next_robot_state": state_t1
}
```

Sequence datasets MUST support:

```text
observation_t
state_t
action_t
action_t+1
...
action_t+n
observation_t+n+1
```

Sources may include:

* G1 teleoperation
* real demonstrations
* existing G1 datasets
* Isaac Lab simulation

The same dataset MUST be usable by every World Model backend.

---

## 10. Planner

Implement model-independent:

```text
CEM
MPC loop
```

Planner API:

```python
class Planner:
    def plan(
        self,
        world_model,
        current_state,
        goal,
        action_space
    ):
        ...
```

Typical cycle:

```text
observe
  ↓
sample candidate action sequences
  ↓
predict futures using World Model
  ↓
score against goal
  ↓
select best sequence
  ↓
execute first action/chunk
  ↓
observe again
  ↓
replan
```

Planner MUST NOT contain model-specific logic.

---

## 11. Goal Interface

MVP:

```text
goal image → WorldModel.encode_goal() → z_goal
```

Planning objective:

```text
distance(predicted_z, z_goal)
```

Future goals:

* text
* demonstration video
* trajectory
* multimodal goal

---

## 12. Benchmark Modes

### Common Mode

Used for fair model comparison.

Keep constant:

```text
dataset
train/test split
G1 embodiment
action space
task
planner
planning budget
evaluation
```

Change only:

```text
world_model.backend
```

Example:

```text
             SAME G1 DATA
                  │
      ┌───────────┼───────────┐
      ▼           ▼           ▼
   JEPA-WMs      LeWM     Native JEPA
      │           │           │
      └───────────┼───────────┘
                  ▼
              SAME CEM
                  ▼
             G1 + Dex3
```

### Native Mode

Allow each model to use its original recommended:

* encoder
* loss
* planner
* preprocessing
* hyperparameters

Used to measure the model's best/reference configuration.

---

## 13. Primary Benchmark

### Task

**Apple → Plate**

Desired behavior:

```text
observe apple
→ reach
→ grasp
→ transport
→ place on plate
→ release
```

Start development with simpler subtasks:

```text
1. Reach object
2. Reach target
3. Grasp
4. Move object
5. Pick & Place
6. Apple → Plate
```

Prefer simulation before real robot execution.

---

## 14. Generalization Benchmark

Critical experiment:

Train on combinations such as:

```text
cube → target
banana → bowl
object → container
```

Test without task-specific training:

```text
apple → plate
```

Measure whether learned physical dynamics + planning generalize to unseen object/task combinations.

---

## 15. Evaluation

### World Model

Measure:

```text
one-step prediction error
multi-step prediction error
action sensitivity
training time
VRAM
inference latency
planning throughput
```

### Robot

Measure:

```text
reach success
grasp success
transport success
place success
full task success
execution time
number of replans
```

Store all results in a common machine-readable format.

---

## 16. Configuration

Experiments MUST be configuration-driven.

Example:

```yaml
embodiment:
  backend: unitree_g1_dex3

world_model:
  backend: leworldmodel

planner:
  backend: cem
  horizon: 8
  samples: 500

dataset:
  format: lerobot

task:
  name: apple_to_plate
```

Changing:

```yaml
world_model:
  backend: native_jepa
```

MUST NOT require other code changes.

---

## 17. Repository Structure

```text
open-embodied-jepa/

├── models/
│   ├── base.py
│   ├── native_jepa/
│   ├── leworldmodel/
│   └── jepa_wms/
│
├── embodiments/
│   ├── base.py
│   └── unitree_g1_dex3/
│
├── planners/
│   ├── base.py
│   ├── cem.py
│   └── mpc.py
│
├── datasets/
│   └── lerobot.py
│
├── simulation/
│   └── isaac_lab/
│
├── tasks/
│   └── apple_to_plate/
│
├── benchmarks/
├── configs/
├── training/
├── evaluation/
└── deployment/
    └── unitree_sdk2/
```

---

## 18. Licensing Requirements

Project code SHOULD target:

```text
Apache-2.0
```

Requirements:

* Track licenses of every dependency.
* Track model/checkpoint licenses separately from source-code licenses.
* Non-commercial components MUST be optional.
* Core framework MUST NOT depend on NC-licensed code.
* Clearly mark experimental integrations with restrictive licenses.

`jepa-wms` should therefore be treated primarily as a research/reference integration unless licensing permits the intended usage.

---

## 19. MVP Acceptance Criteria

MVP is complete when:

* [ ] G1 + Dex3 embodiment adapter exists.
* [ ] LeRobot-compatible G1 dataset loads.
* [ ] `native_jepa` trains on the dataset.
* [ ] At least one external JEPA backend is integrated.
* [ ] Both implement the same `WorldModel` API.
* [ ] Both support action-conditioned future prediction.
* [ ] CEM planner works with both without model-specific code.
* [ ] Goal-image planning works in simulation.
* [ ] Identical benchmark can run against both models.
* [ ] Results are stored in common format.
* [ ] Backend can be changed through configuration only.

---

## 20. Non-Goals for MVP

Do NOT initially optimize for:

* locomotion
* whole-body manipulation
* language-conditioned planning
* video generation
* tactile sensing
* multi-robot coordination
* production deployment
* training huge foundation models from scratch

Keep the first system focused on **G1 + Dex3 tabletop manipulation**.

---

## 21. Design Rule

When implementing new functionality, preserve this separation:

```text
World Model
    ≠
Planner
    ≠
Embodiment
    ≠
Dataset
    ≠
Task
```

A new World Model should require only a new `WorldModel` adapter.

A new robot should require only a new `Embodiment` adapter.

The framework exists to make both independently interchangeable.

> **One robot stack. One benchmark. Many world models.**
