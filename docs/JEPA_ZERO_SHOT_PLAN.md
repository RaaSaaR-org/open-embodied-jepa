# JEPA Zero-Shot Robot Plan

Status: PROPOSAL (2026-10-08). This is a research-backed plan and recommendation, not a decision record. Each phase still needs its own preregistration and ruling in `docs/DECISIONS.md`. Nothing here changes R7's canonical status sentence. Editable copy: [JEPA Zero-Shot Robot Plan](https://claude.ai/code/artifact/963ec689-7b14-491d-93a8-1fec6faa63d2).

## Short answer

Yes, continue with JEPA, but change the method. Stop the moving-plate line. Build one robot-grounded world model, trained once on broad G1+Dex3 play data, then plan short steps toward goals.

- **No model does new tasks with zero training.** Not JEPA, not GR00T, not pi0. The realistic target is "train once on the robot, then no training per task". That is what Meta's V-JEPA 2-AC shows on Franka arms.
- **Our 0/150 per backend (TASK-020, frozen v1 benchmark) fits the literature.** Replanned image-goal CEM toward a single distant goal image, with no subgoals, fails even for Meta's 1B-parameter model on pick-and-place. Every working recipe uses short planning segments with subgoals. (In TASK-020 every model episode was also stopped by the joint-rate guard, so the failure is confounded.)
- **The owner's idea is the best-supported fix.** Feeding in and predicting the robot's own state (joint angles, hand pose) and action consequences makes latents plannable in several 2026 papers. Those papers are mostly single-group preprints, so we treat them as hypotheses to test.
- **Language comes on top later.** A language model turns "put the apple on the plate" into subgoals; JEPA plans each short step.
- **Honest risk:** no published image-latent planner has worked with dual 14-D dexterous-hand actions. We start with one arm and a simplified grasp, and every phase has a stop rule.

## Progress (2026-10-10)

Each phase ran under its own preregistration and rulings. The rows and numbers below come from the results documents, which give the full caveats. None of them changes R7's canonical status sentence, and none is an Apple→Plate result.

| Phase | Task | Row | What it showed, with its main limits |
| --- | --- | --- | --- |
| 0. Calibrate | TASK-084 ([results](experiments/jepa_wms_pusht_calibration_results.md)) | **P0-PASS** (R22.10) | Meta's released JEPA-WM on Push-T, the same 96 episodes per arm: upstream's CEM 67/96 (69.8 %, Wilson 95 % 60.0–78.1 %) against the published 70.2 %, so the pipeline reproduces on this machine; our `CEMPlanner` 83/96 (86.5 %, 78.2–91.9 %), so our CEM is not the bottleneck where a JEPA world model is known to plan (this does not show that our world models are plannable). "Ours is better" is not a preregistered claim. One released checkpoint, one 2-D task, one planning setting; nothing here is about LeWM, G1 + Dex3 or replanning in a closed loop. |
| 1. Play corpus | TASK-085 ([results](experiments/play_corpus_v1_results.md)) | **P1-PASS** (R23.24) | `play-v1`: 3 194 episodes, 16.06 h, right arm and hand in MuJoCo; an object moved in 55.9 % of train episodes. Collected by a privileged scripted play collector, so it is not a learned result. The bars were close to pass-by-construction, and the row does not say the corpus is good play data for a world model. |
| 1. Real-data arm | TASK-086 ([results](experiments/real_g1_dex3_prestep_results.md)) | **R-DROP-GRASP** (R23.28) | Open real G1 + Dex3 sets: the arm motion converts (5.5 % of steps out of range), but the grasp does not fit our one-scalar Dex3 synergy (pooled median residual 0.487 / 0.461 rad, left / right, against a 0.15 rad bar). The real-data arm is dropped, and Phase 2 used sim play only. The bars were declared, not calibrated. |
| 2. Grounded LeWM, offline | TASK-087 ([results](experiments/grounded_lewm_v1_results.md)) | **P2-FAIL** (R24.22) | On `play-v1`'s test split (5 634 roots, three seeds), the grounded arms S (+ robot state) and G (+ state, joint-change and inverse-dynamics losses) met every criterion against plain LeWM P at h = 4 and 8 on every seed. They did not at h = 1: their one-step action sensitivity was lower than P's on every seed (for G every interval excludes 0; for S two of three). Offline only, with no planner. One corpus, one camera at 112 px, one frozen encoder and one budget; the inverse-dynamics caveat applies. |
| 3. Zero-shot reach and grasp | TASK-088 ([protocol](experiments/zero_shot_reach_grasp_v1.md)) | in progress | Frozen at Stage 0 (#192, R25.17–R25.21); K0 set the bars at reach 52/64 and grasp-and-lift 26/64. Stage D, a development run that is not gated (16 + 16 resets, one model seed, R25.22): reach P 14/16, G 16/16, the scripted IK follower 16/16, hold and random 0/16; grasp-and-lift 0/16 for every world-model arm, against 12/16 for the IK follower. The gated Stage S (64 + 64 resets) was running when this was written, so there is no row yet. |

R25.1 adopted R24.23's recommendation after Phase 2 for Phase 3: plain LeWM P (seed 87100) is the primary model, and G is a declared second arm because it ranks the executed commands better at every horizon. The one-step wrong / true ratio is not used as a gate until it is calibrated for models that take state as input. S and G, as specified, are not carried forward as "the grounded model". LeWM, DINOv2, `play-v1` and the product goal are not abandoned.

## Where the project stands

LeWM has gated successes in simulation (TASK-081, replicated with two more model seeds of the same training run in TASK-083) for one narrow job: predicting where a moving plate will be and choosing the place aim once. Everything else in the loop is a separately trained policy or hand-written code.

| Part of the task | Who does it | Status (MuJoCo sim) |
| --- | --- | --- |
| Grab the apple | P-3, a behaviour-cloning policy on frozen DINOv2 features (not LeWM) | 40/40 on cohort C (TASK-072; one run, one seed per arm; the random-init encoder control scored 39/40, so pretraining is not shown; cohort C is no longer held out) |
| Choose where to place | LeWM token predictor + plate readout + solver | 118/128 gated (TASK-081; one run, one model seed, simulation only, under the declared plate condition C1-M); TASK-083 (REP-PASS) replicated it with W's other two model seeds on 128 fresh gated resets: 124/128 (66801) and 120/128 (66802), each L-PASS (H-rule 123/128 on that cohort, not measurably better than either); one run, simulation only, C1-M, one corpus, recipe and encoder; "LeWM needed" not shown |
| Carry and release | e9, a hand-written expert that reads simulator truth | scripted, not learned |

What this does and does not show:

- In TASK-081, W's aim (118/128) beat its action-blind twin (73/128) and its scene-blind twins (39 and 59).
- "LeWM needed" is not shown: in TASK-081 a hand-written rule given the plate law scored 125/128, measurably better (W − H-rule −7/128, 95 % interval −13 to −2); in TASK-083 it was not measurably better or worse than either seed, and "LeWM needed" is not shown for either. LeWM chooses only the single place aim, on frozen DINOv2 features; TASK-081 used one model seed (66800, flagged `last_two_triggered`), and the three seeds share one training corpus, recipe and encoder. The solver's effect is not shown.
- The moving plate is an artificial condition. DECISIONS R2 required later LeWM plate tasks to use a plate the robot must predict, because with a visible plate no world model is needed for the place; TASK-076 then confirmed it: a plain image readout of the plate scored 64/64 with no world model (under TASK-074's 9 cm plate move; one run, one camera, one encoder; the random-init floor also scored 64/64, the image-free clock prior 51/64).
- LeWM planning the whole motion toward a goal image scored 0/150 per backend on the v1 benchmark (TASK-020).
- In Isaac Sim the learned pick fails on Isaac's images (apple estimate about 11 cm off). With the scripted pick, W rested the apple on 2 of 4 debug seeds under Newton and 0 of 4 under PhysX (development only, not evidence).
- Nothing runs on the real robot; the hardware interface is mock-only.

## JEPA world models: what works today

Zero-shot JEPA robot control works only for short tasks on Franka arms with parallel grippers; pick-and-place needs subgoals. Trial counts are small (5–45 per cell) and nothing has been independently reproduced.

| Work | What it shows | Numbers (with conditions) |
| --- | --- | --- |
| [V-JEPA 2-AC](https://arxiv.org/html/2506.09985) (Meta, 2025-06) | Frozen 1B video encoder + 300M action predictor, trained on ~62 h of unlabeled robot video (DROID); horizon-1 CEM to a goal image | Reach 100%; grasp cup 65%, box 25%; pick-and-place 80% / 65% only with 2 extra subgoal images; 10 trials per cell; 16 s per action on an RTX 4090 |
| [HWM](https://arxiv.org/html/2604.03208v2) (FAIR/NYU, 2026-04) | Hierarchical latent planner | Flat V-JEPA 2-AC 0% on real pick-and-place (cup / box) from a single goal image; hierarchy 70% / 60%; end-effector proprioception is a model input |
| [Planning Limits of Latent WMs](https://arxiv.org/html/2609.39235) (2026-09) | Frozen-encoder WMs rank actions well only ~5–10 steps ahead | Even with the true simulator as predictor, success falls from 92% to 41% as the target moves from 5 to 20 steps ahead of a 5-step rollout; an 81× larger predictor leaves the plannable range unchanged |
| [JEPA-WMs study](https://arxiv.org/html/2512.24497v1) (Meta, 2025-12) | Design choices for planning | Frozen DINOv2 beats V-JEPA in sim; proprioception helps; CEM most robust; Push-T 70.2%, RoboCasa place 30.7% |
| [LeWM](https://arxiv.org/html/2603.19312v1) (our backend, 2026-03) | 15M end-to-end JEPA, fast CEM | Simple sim scenes, ~25-step horizon, no real robot |
| [AD-WM](https://arxiv.org/abs/2609.30264) (2026-09) | Inverse-dynamics / action-recovery losses make WMs plannable | Cube hard 3.7% → 52.0% vs LeWM; real Franka pick-and-place 42% → 71% (45 trials); single group |
| [GAWM](https://arxiv.org/pdf/2609.03565) (2026-09) | Ground latents in robot state | Beats LeWM on 3 of 4 LeWM tasks (100/85/98/87 vs 87/86/96/74; LeWM's numbers copied from its paper, not re-run) |
| [SAGE](https://arxiv.org/html/2607.17973) (2026-07) | Learned subgoals around a frozen LeWM | Push-T at 150 steps 4.7% → 64.7% |
| [Hi-LeWM](https://arxiv.org/html/2607.12547v1) (2026-07) | Naive hierarchy on LeWM | Worse than flat (38.7% vs 52.7%); only a constrained version gains |
| [D-JEPA](https://arxiv.org/pdf/2609.24749) (2026-09) | Latent L2 to goal misranks actions | LeWM top-4 ranking inverted 49% of the time |
| [DexWM](https://arxiv.org/abs/2512.13644) (2025-12) | Only zero-shot latent planner with a multi-finger hand | Real grasp 10/12 (Franka + Allegro); ~900 h pretraining + 4 h exploration |
| [LePlanner](https://arxiv.org/pdf/2609.13845) (2026-09) | Amortized planner trained inside the WM | Matches CEM (92% Cube) with 20 vs 45 000 latent transitions |
| [vjepa2 issue #169](https://github.com/facebookresearch/vjepa2/issues/169) | Released V-JEPA 2-AC in a new Isaac render | Stops cm above the cube, never grasps; anecdotal |

Every working recipe shares three things: diverse task-agnostic interaction data on the same robot, robot state in the model or the goal, and short planning segments with subgoals. This project has not combined them: robot state was fused into the latent before (apple WM v2–v4, `state_fusion` with `readout_heads`; all four v4 arms failed their gates at TASK-054), but never with play data, short-segment subgoals or inverse-dynamics and joint-change losses.

## Robot foundation models (VLAs) and world-model combos

VLAs such as GR00T and pi0 are not zero-shot on a new robot or task. Their "zero-shot" means new scenes for robots and tasks already in their training data. The public G1 apple-to-plate numbers below are all for task-specific fine-tuned checkpoints.

| Work | Role | Numbers (with conditions) |
| --- | --- | --- |
| [GR00T N1.6 G1 apple-to-plate](https://huggingface.co/cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate) | Community fine-tune of GR00T N1.6 on our task family, MuJoCo | 5/10 episodes after fine-tuning on 102 sim demos; the card reports difficulty reproducing NVIDIA's published G1 checkpoints |
| [GR00T on real G1](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/real-robot-workflow/real-deployment.html) | NVIDIA's tutorial, real G1 apple-to-plate | Bundled fine-tuned checkpoint (`nvidia/GR00T-N1.7-ApplePnP-V1`) 68% on 100 real rollouts (25 placements × 4); no zero-shot number reported |
| [GR00T N1](https://arxiv.org/html/2503.14734) / [EgoScale](https://arxiv.org/html/2602.16710v1) | Humanoid VLA | N1 76.8% real with full demos; N1.5 zero-shot novel objects 15% |
| [pi0.5](https://arxiv.org/html/2504.16054) | Scene generalisation, ~400 h data, ~100 homes | Works in unseen homes; parallel grippers only |
| [OpenVLA-OFT](https://arxiv.org/html/2502.19645) | Fine-tuning recipe | LIBERO 97.1% with 500 demos per suite |
| [DreamSteer](https://arxiv.org/abs/2607.02865) (Meta, 2026-07) | DINOv2-latent WM + value model selects among VLA candidates | pi0 23.75% → 66.25% on unseen objects, only with motion primitives added |
| [CheckVLA](https://arxiv.org/abs/2607.26789) (2026-07) | Action-conditioned WM as an execution monitor | RoboCasa 27.6% → 36.1%; sim only |
| [GWM](https://arxiv.org/html/2604.11751) (2026-04) | WM in a frozen VLM embedding space, language goals | Real Franka 55/60 subtasks; 12 H100s |
| [ZeroDex](https://arxiv.org/html/2606.19340) (2026-06) | No world model: VLM subgoals + keypoints + motion generator | Dexterous single-step tasks 4–5/5; tiny n |
| [LEGS](https://arxiv.org/html/2606.01458) (2026-05) | Gaussian-splat backgrounds over MuJoCo, G1+Dex3 | Real 5–10/10 per task; trained on 4×H100 |

On our own task family, our simulation results (P-3 40/40; the P-3 pick + LeWM aim + e9 place chain 118/128, each one run) are not obviously behind these GR00T numbers, though the setups, scorers and reset distributions differ and are not directly comparable. By the owner's answer, GR00T plays no role in the plan. GWM and ZeroDex show the language layer wanted later: a VLM turns a command into subgoals.

## What blocks zero-shot JEPA control, and the fixes

Six blockers explain our results. Each has a published fix we can test, though most fixes are 2026 single-group preprints.

| Blocker | Our symptom | Fix to test |
| --- | --- | --- |
| Horizon: latent WMs rank actions well only for targets ~5–10 steps ahead | 0/150 per backend with replanned CEM (horizon 4, up to 805 steps) toward one distant goal image, no subgoals; every model episode stopped by the joint-rate guard (TASK-020) | Subgoals and short segments ([HWM](https://arxiv.org/html/2604.03208v2), [SAGE](https://arxiv.org/html/2607.17973)); avoid naive hierarchy ([Hi-LeWM](https://arxiv.org/html/2607.12547v1)) |
| Misleading cost: latent L2 to goal misranks actions | Apple-plate offset read only to about 2–3.4 cm (TASK-075) | Add a robot-pose term to cost and goal ([JEPA-WMs](https://arxiv.org/html/2512.24497v1)) |
| Latents not grounded in robot state | Small objects and the hand poorly identifiable | Proprio input, joint-change heads, inverse-dynamics loss ([AD-WM](https://arxiv.org/abs/2609.30264), [GAWM](https://arxiv.org/pdf/2609.03565), [Strohm et al.](https://arxiv.org/abs/2610.03137)): the owner's "predict robot states" idea |
| Data: every result uses tens to hundreds of hours of task-agnostic play on the same robot | Our corpora are mostly scripted or expert-labelled single-task demos with few failures | A MuJoCo play corpus with random, perturbed and failed behaviour ([PLDM](https://arxiv.org/abs/2502.14819)); as an extra arm, open real G1 + Dex3 teleoperation data (see [Open real G1 data](#open-real-g1-data)) |
| Dexterous 14-D actions: no published success | — | Start with one arm, one hand, a reduced grasp synergy ([DexWM](https://arxiv.org/abs/2512.13644)) |
| Visual domain gap | Isaac transfer failed (11 cm apple error) | Fix the camera first; later real-to-sim appearance matching ([LEGS](https://arxiv.org/html/2606.01458)) |

Compute is a seventh, smaller issue: CEM over 14-D actions is slow. Short horizons, fewer action dimensions or an amortized planner ([LePlanner](https://arxiv.org/pdf/2609.13845)) address it, and cloud GPUs are now allowed.

## Options compared

Recommended: B, then A, with C as the language layer on top later. D and E do not lead to the goal.

| Option | What | Zero-shot potential | Cost | Risk | Pick |
| --- | --- | --- | --- | --- | --- |
| **A. Robot-grounded play-data LeWM + short-step planning** | Play corpus with proprio and failures; LeWM with proprio input, joint-change heads, inverse-dynamics loss; CEM on short segments to image + pose goals | Highest within reach: zero-shot reach and grasp near term; pick-and-place with subgoals | 3–5 months; fits the RTX 5080, cloud helps | No published dexterous success; preprint gains may not reproduce | Yes |
| **B. Calibrate on Meta's released jepa-wms models** | Reproduce published numbers on this machine with their planner and ours before any G1 work | None by itself; tells us if our pipeline is sound | 1–2 weeks | Low | Yes, first |
| **C. Language/VLM subgoals + JEPA short-step planner** | A VLM turns a command into subgoals; JEPA plans and checks each short step | Good for long tasks and language commands | +1–2 months on top of A | The VLM may do most of the work; needs ablations | Later |
| D. Continue moving-plate / selector line | More LeWM-picks-an-aim conditions | Very low; per-task and artificial | Ongoing | Months without progress | No, stop |
| E. Switch to GR00T or behaviour cloning | Fine-tune a VLA or extend BC | None in the owner's sense; every task needs demos | Cloud GPUs | Abandons JEPA | No |

## Open real G1 data

Question from the owner (2026-10-09): should Phases 1–2 also use public real data from our robot type (Unitree G1 + Dex3-1), not only MuJoCo play data? Short answer: yes, as an extra arm, not as a replacement. About 40 hours of real G1 + Dex3 teleoperation are public (about 26 h of it under a clear Apache-2.0 or CC-BY-4.0 license, counting Unitree's duplicate set once), one set is our exact task, and the conversion to our action schema is mechanical. But it is task demonstration data, not play, so it cannot replace the play corpus, and nobody has shown that real data helps a world model tested in simulation.

Numbers below were read on 2026-10-09 from each dataset's card and its `meta/info.json` through the Hugging Face API; hours are total frames ÷ fps. Only metadata and one 33 MB parquet file (Unitree PickApple, states and actions) were downloaded.

### Real G1 with Dex3-1 hands (usable)

| Dataset | License | Format | Size | Tasks | Cameras | State / action | Fit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Unitree G1_Dex3_* (13 sets)](https://huggingface.co/datasets?search=unitreerobotics%2FG1_Dex3) | Apache-2.0 | LeRobot v3.0, AV1 video | 3,152 episodes, ≈24.0 h, 30 Hz | One task per set: 7 single-object picks, among them [PickApple](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_PickApple_Dataset) ("Put the apple into the plate.", 201 episodes, ≈1.4 h), plus block stacking, pouring, toast, packaging, object placement | Head stereo pair 640×480; 6 of 13 sets also both wrists 640×480 | 28-D absolute joint positions: 2 × 7 arm + 2 × 7 hand; no end-effector pose, no waist | Best fit: same arms and hands, permissive license, our format family |
| [Humanoid Everyday, G1 part](https://huggingface.co/datasets/USC-PSI-Lab/Humanoid-Everyday-G1) | Unclear: GitHub README says MIT, the full dataset's HF card says Apache-2.0, the G1 repo has no license tag | LeRobot v2.1 | 4,064 episodes, 247 tasks, ≈16.5 h, 30 Hz | Everyday manipulation, interaction and loco-manipulation | Egocentric RealSense D435, 640×480 RGB (depth in the full set) | Arm 14, hand 14 (Dex3), legs; variable-length vectors | Good if the license is confirmed; broadest task variety |
| [NVIDIA GR00T-Teleop-G1](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1) | CC-BY-4.0 | LeRobot v2.1 | Card: 1,000 trajectories; the four folders' metadata: 1,095 episodes, ≈1.7 h, 20 Hz | Pick one of four fruits and place it on the plate | One ego view, 640×480 | 43-D whole-body joints; hand joints named like Dex3 (thumb 0–2, index 0–1, middle 0–1) | Good: close to our task, small |
| [NVIDIA GR00T-N1.7-AppleToPlate](https://huggingface.co/datasets/nvidia/GR00T-N1.7-AppleToPlate) | CC-BY-4.0 | LeRobot v2.1 | 402 episodes, ≈1.6 h, 30 Hz | "move the apple to the plate", real G1, XR teleoperation | One ego view, 640×480 | 43-D whole-body joints, 7 per hand (hand model not named in card or metadata) | Our exact task: keep it as a real-image test set, not training data |
| [mncai G1_Dex3_Trash_LocoManipulation](https://huggingface.co/datasets/mncai/G1_Dex3_Trash_LocoManipulation) | Apache-2.0 | LeRobot v2.1 | 118 episodes, ≈61 min, 50 Hz | Walk, kneel, pick a trash bag, bin it; left hand only | Head D435i, 640×480 | 43-D joints + 14-D wrist pose | Poor: whole-body walking; but 37 episodes contain deliberate failed grasps and recovery |

Data quality notes from the check: Unitree's GraspSquare set has the same episode and frame counts and the same README text as BlockStacking, and its task string reads "camera packaging", so it is probably a duplicate; the PickApple README shows an older v2.1 metadata block (804 videos for 201 episodes) while its `meta/info.json` is v3.0 with 2 cameras; in PickApple, two state rows (episodes 103 and 193) hold impossible values (−1151.6 and 428.9) in one right-hand joint; hand joint order differs between left and right hands and between providers, so mapping must go by name. None of these cards says whether failed attempts were removed; they are teleoperated task demonstrations.

### Other hands and simulated G1 (not recommended now)

- **Other Unitree hands (real):** 26 Inspire whole-body sets with metadata (≈65 h; 3 more have none), 8 + 29 BrainCo sets (≈15 h + ≈70 h) and 108 Dex1 two-finger gripper sets (≈311 h, among them the [DiverseManip](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_DiverseManip_SingleArm_128x128) sets that Unitree describes as a diversity dataset for world models). Different hands, so hand actions do not transfer; the arm and camera parts could. Many newer Unitree sets have no README or license tag; treat them as unlicensed until Unitree says otherwise.
- **Simulated G1:** NVIDIA's [GR00T-X-Embodiment-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim) folder `unitree_g1.LMPnPAppleToPlateDC` (simulated, with MuJoCo state fields in its metadata; 102 trajectories per card, 50 Hz, apple to plate while walking, CC-BY-4.0) and Isaac Lab-Arena sets ([static pick-and-place](https://huggingface.co/datasets/nvidia/Arena-G1-Static-PickNPlace-Task), [loco-manipulation](https://huggingface.co/datasets/nvidia/Arena-G1-Loco-Manipulation-Task); their cards and metadata disagree on episode counts). Small, other simulators and other renderers; they add little over our own MuJoCo play corpus.
- About 100 small community G1 + Dex3 repositories exist on Hugging Face, many copies of Unitree's sets; not checked.

### Precedent

- [V-JEPA 2-AC](https://arxiv.org/html/2506.09985) trained its action-conditioned predictor on less than 62 h of unlabeled real DROID video from one robot type, with a 7-D end-effector-delta action; no simulated data in that stage.
- [DexWM](https://arxiv.org/html/2512.13644) pretrained on 829 h of human egocentric video (EgoDex) plus about 100 h of DROID (parallel gripper), then fine-tuned on about 4 h of random exploration in simulation (RoboCasa), and grasped with a real Franka + Allegro hand: mixing embodiments and real with sim data worked for a latent world model, in one group's work.
- [Sim-and-Real Co-Training](https://arxiv.org/abs/2503.24361) (2025) reports that adding simulation data raised real-world policy success by an average of 38 %, on an arm and a humanoid; that is for policies, not world models.

### Fit with our pipeline, and the conversion

| Our side | Real data | Conversion |
| --- | --- | --- |
| Action `ee_delta_grasp_v0` (`contracts.py`): per arm 6-D end-effector delta, plus one grasp scalar per hand, normalized to [−1, 1] by 1.5 cm and 0.06 rad per 0.05 s step (`configs/g1_sim_action.json`) | Absolute joint targets (28-D or 43-D) at 20–50 Hz | Resample to 20 Hz; forward kinematics with our G1 MJCF to the `*_ee` site (NVIDIA and mncai use our joint names; Unitree's `kLeftShoulderPitch`-style names need a fixed mapping); difference consecutive poses; scale. Steps outside [−1, 1] are flagged and their windows dropped, never clipped. For Unitree's 28-D sets assume waist = 0 and record it. |
| Grasp: one scalar per hand that interpolates 7 Dex3 joints between our open and closed synergy (`embodiment.py`) | 7 Dex3 joint angles per hand | Project onto the open→closed line, by joint name; keep the residual so poorly fitting grasps can be excluded |
| State: joint positions and velocities | Positions only | Velocities by finite difference; missing joints marked invalid |
| Camera: one torso-mounted render, 75° vertical field of view, 112 px | Real head or ego camera, 640×480, different mount, field of view, lighting and background | Centre-crop and resize to 112 px. The viewpoint cannot be matched; store it under its own camera key and treat it as a second domain. The head cameras (Unitree, D435) are the closest match to a later real G1. |
| Storage: local LeRobot v3 profile with PNG frames in Parquet and a JEPA manifest (`data.py`) | LeRobot v3.0 with AV1 video (Unitree) or v2.1 with H.264 | One-off converter: decode, resize, write PNG, record source repository, revision and file hashes in the manifest; split by episode |

What real data can and cannot give: real images, real hand-object contact and real arm motion on our exact robot, and the visual domain a later real G1 sees. It cannot give play data or failures (except the mncai set), its actions cannot be replayed in our simulator, and the Phase 3 test is in simulation, so any gain must show there.

**Recommendation.** Add one arm, "sim play + open real G1/Dex3 data", next to "sim play only". Train both in Phase 2 with the same budget and compare them at Phase 3. Before Phase 2, a short pre-step (about one week): confirm licenses (use Apache-2.0 and CC-BY-4.0 sets only; ask the Humanoid Everyday authors; skip unlicensed sets), convert the 12 distinct Unitree G1_Dex3 sets plus NVIDIA GR00T-Teleop-G1, and Humanoid Everyday G1 if cleared, and hold out NVIDIA GR00T-N1.7-AppleToPlate as a real-image test set. The pre-step's stop rule: if most converted steps fall outside [−1, 1], or the grasp projection fits badly, drop the arm and record why.

## Recommended plan

Six gated phases take about five months to a first zero-shot pick-and-place in simulation. Each phase has a stop rule, so a dead end costs weeks, not months. The first zero-shot claim comes in Phase 3, around week 15. Week counts are estimates for one person with AI agents; bars are examples to be fixed in each phase's preregistration.

| Phase | Weeks | Goal | Gate (stop rule) |
| --- | --- | --- | --- |
| 0. Close and calibrate | 1–2 | Record TASK-083 and close the plate line; reproduce Meta's jepa-wms numbers on this machine | Our planner within about 10 points of the released result on one task, else fix it first |
| 1. Play corpus | 3–6 | One arm and one hand; 10–50 sim hours of random, perturbed and failed behaviour, full robot state logged. In parallel, the open-real-data pre-step: check licenses and convert the open real G1 + Dex3 sets ([Open real G1 data](#open-real-g1-data)) | Covers the test workspace and at least 20% of episodes move an object; the real-data arm is dropped if most converted steps fall outside the action range |
| 2. Robot-grounded LeWM, offline | 7–11 | Plain LeWM vs + robot state vs + joint-change and inverse-dynamics losses, on frozen DINOv2 tokens; the best variant trained twice, on sim play only and on sim play + open real G1/Dex3 data | A grounded version beats plain LeWM on action sensitivity and short-step ranking on all seeds; the data arms are compared at Phase 3, not here |
| **3. Zero-shot reach and grasp** | 12–15 | Short-horizon planning to new image + pose goals never used in training; both data arms (sim only, sim + real) on the same resets | Example bars: reach 80%, grasp-and-lift 40%; stop JEPA planning if reach stays under 50% after one fix |
| 4. Pick-and-place with subgoals | 16–21 | Scripted subgoals first, then learned subgoals from the play corpus; JEPA checks each step and re-plans | Clearly beats subgoal-following without a world model; learned subgoals reach half of scripted |
| 5. Language commands | after Phase 4 | A vision-language model turns a command into subgoals; JEPA plans each short step | New commands and object layouts succeed with no per-task training |

Details that matter:

1. **Phase 0** also runs an action-sensitivity probe on our existing LeWM checkpoints, as a baseline.
2. **Phase 1** collects play data, not expert demos. Coverage and failures matter more than quality. Open real G1 + Dex3 data is an added arm, not a substitute: it is task demonstrations, mostly without failures.
3. **Phase 2** reuses the repo's off-by-default options (`state_fusion`, `readout_heads`); `state_fusion` together with `readout_heads` was tried in apple WM v2–v4 and every arm failed its gate set (v4 at TASK-054), so that combination is a control arm here, not the fix. Each arm gets its own preregistration, as now. With cloud GPUs allowed, a V-JEPA 2.1 encoder arm can be added.
4. **Phase 3** is the first honest zero-shot test. Baselines: hold, random, scripted IK reach and plain LeWM.
5. **Phase 4** compares against P-3 (task-trained) and a VLM + IK baseline (ZeroDex-style), to show what JEPA adds.
6. **Phase 5** adds language. The real robot enters once hardware is available, starting with offline checks on real camera images.

## Interview: owner answers (2026-10-08)

| Question | Recommended | Owner's answer |
| --- | --- | --- |
| Real G1 + Dex3 available? | Later / maybe | Later / maybe |
| What does "no training per task" mean? | Goal image only | Language command ("put the apple on the plate") |
| Must JEPA do the motion itself? | Either is fine | Not sure; wants one general model that does everything, not per-task training |
| Stop the moving-plate line after TASK-083? | Yes, stop | Yes, stop |
| Define the goal as "train once on the robot, then no training per task"? | Yes | Yes |
| Language model turns commands into subgoals, JEPA plans each short step? | Yes, in a later phase | Yes, in a later phase |
| Cloud GPUs? | RTX 5080 only | Yes, allowed |
| Role for GR00T? | External baseline | None |

Consequences for the plan:

- Simulation first; real-robot steps are planned but gated on hardware access.
- Commands are given as language, so a language-to-subgoal step (a VLM) sits in front of the JEPA planner, added in Phase 5.
- The aim is one general model; per-task fine-tuning counts as a cost to minimise, not an acceptable default.
- The moving-plate experiments end once TASK-083 is recorded.
- With cloud GPUs allowed, larger encoders (V-JEPA 2 / 2.1) and bigger play corpora become optional arms. GR00T is dropped from the plan.

## Sources

Pages opened by the research agents on 2026-10-08. Most 2026 entries are preprints from a single group, so their numbers are unreplicated.

- [V-JEPA 2](https://arxiv.org/html/2506.09985)
- [Hierarchical Planning with Latent World Models (HWM)](https://arxiv.org/html/2604.03208v2)
- [The Planning Limits of Latent World Models](https://arxiv.org/html/2609.39235)
- [What Drives Success in Physical Planning with JEPA World Models?](https://arxiv.org/html/2512.24497v1)
- [V-JEPA 2.1](https://arxiv.org/html/2603.14482v1)
- [LeWorldModel](https://arxiv.org/html/2603.19312v1)
- [AD-WM](https://arxiv.org/abs/2609.30264)
- [GAWM](https://arxiv.org/pdf/2609.03565)
- [PSG-JEPA](https://arxiv.org/pdf/2608.06799)
- [Keeping JEPA World Models Plannable When Little of the Frame Moves](https://arxiv.org/abs/2610.03137)
- [SAGE](https://arxiv.org/html/2607.17973)
- [Hierarchical Planning in LeWorldModel (Hi-LeWM)](https://arxiv.org/html/2607.12547v1)
- [D-JEPA](https://arxiv.org/pdf/2609.24749)
- [LePlanner](https://arxiv.org/pdf/2609.13845)
- [PLDM](https://arxiv.org/abs/2502.14819)
- [DexWM](https://arxiv.org/abs/2512.13644)
- [ZeroDex](https://arxiv.org/html/2606.19340)
- [Grounded World Model (GWM)](https://arxiv.org/html/2604.11751)
- [DreamSteer](https://arxiv.org/abs/2607.02865)
- [CheckVLA](https://arxiv.org/abs/2607.26789)
- [Cosmos Policy](https://arxiv.org/abs/2601.16163)
- [Ego-Vision World Model for Humanoid Contact Planning](https://arxiv.org/html/2510.11682)
- [Reconstruction or Semantics? Latent spaces for robotic world models](https://arxiv.org/abs/2605.06388)
- [vjepa2 issue #169](https://github.com/facebookresearch/vjepa2/issues/169)
- [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
- [facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms)
- [le-wm (LeWorldModel code)](https://github.com/lucas-maes/le-wm)
- [HWM_PLDM code](https://github.com/kevinghst/HWM_PLDM)
- [GR00T N1.6](https://research.nvidia.com/labs/gear/gr00t-n1_6/)
- [GR00T-N1.6-G1-PnPAppleToPlate model card](https://huggingface.co/cloudwalk-research/GR00T-N1.6-G1-PnPAppleToPlate)
- [NVIDIA GR00T G1 real deployment](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/real-robot-workflow/real-deployment.html)
- [GR00T N1](https://arxiv.org/html/2503.14734)
- [EgoScale (GR00T N1.7)](https://arxiv.org/html/2602.16710v1)
- [pi0.5](https://arxiv.org/html/2504.16054)
- [openpi](https://github.com/Physical-Intelligence/openpi)
- [OpenVLA-OFT](https://arxiv.org/html/2502.19645)
- [Gemini Robotics On-Device](https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/)
- [VIRAL](https://arxiv.org/html/2511.15200)
- [LEGS](https://arxiv.org/html/2606.01458)
- [Unitree G1_Dex3 ToastedBread dataset](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ToastedBread_Dataset)

Added 2026-10-09 for [Open real G1 data](#open-real-g1-data) (cards and `meta/info.json` read through the Hugging Face API):

- [Unitree datasets on Hugging Face](https://huggingface.co/unitreerobotics) (205 dataset repositories listed on 2026-10-09)
- [Unitree G1_Dex3_PickApple_Dataset](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_PickApple_Dataset)
- [Unitree G1_Dex1_DiverseManip_SingleArm_128x128](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_DiverseManip_SingleArm_128x128)
- [Humanoid Everyday (GitHub)](https://github.com/physical-superintelligence-lab/Humanoid-Everyday), [paper](https://arxiv.org/abs/2510.08807), [full dataset](https://huggingface.co/datasets/USC-PSI-Lab/humanoid-everyday), [G1 part](https://huggingface.co/datasets/USC-PSI-Lab/Humanoid-Everyday-G1)
- [NVIDIA PhysicalAI-Robotics-GR00T-Teleop-G1](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1)
- [NVIDIA GR00T-N1.7-AppleToPlate](https://huggingface.co/datasets/nvidia/GR00T-N1.7-AppleToPlate)
- [mncai G1_Dex3_Trash_LocoManipulation](https://huggingface.co/datasets/mncai/G1_Dex3_Trash_LocoManipulation)
- [NVIDIA PhysicalAI-Robotics-GR00T-X-Embodiment-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim)
- [NVIDIA Arena-G1-Static-PickNPlace-Task](https://huggingface.co/datasets/nvidia/Arena-G1-Static-PickNPlace-Task)
- [NVIDIA Arena-G1-Loco-Manipulation-Task](https://huggingface.co/datasets/nvidia/Arena-G1-Loco-Manipulation-Task)
- [DexWM (HTML, data section)](https://arxiv.org/html/2512.13644)
- [Sim-and-Real Co-Training](https://arxiv.org/abs/2503.24361)
