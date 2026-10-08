# JEPA Zero-Shot Robot Plan

Status: PROPOSAL (2026-10-08). This is a research-backed plan and recommendation, not a decision record. Each phase still needs its own preregistration and ruling in `docs/DECISIONS.md`. Nothing here changes R7's canonical status sentence. Editable copy: [JEPA Zero-Shot Robot Plan](https://claude.ai/code/artifact/963ec689-7b14-491d-93a8-1fec6faa63d2).

## Short answer

Yes, continue with JEPA, but change the method. Stop the moving-plate line. Build one robot-grounded world model, trained once on broad G1+Dex3 play data, then plan short steps toward goals.

- **No model does new tasks with zero training.** Not JEPA, not GR00T, not pi0. The realistic target is "train once on the robot, then no training per task". That is what Meta's V-JEPA 2-AC shows on Franka arms.
- **Our 0/150 per backend (TASK-020, frozen v1 benchmark) fits the literature.** Replanned image-goal CEM toward a single distant goal image, with no subgoals, fails even for Meta's 1B-parameter model on pick-and-place. Every working recipe uses short planning segments with subgoals. (In TASK-020 every model episode was also stopped by the joint-rate guard, so the failure is confounded.)
- **The owner's idea is the best-supported fix.** Feeding in and predicting the robot's own state (joint angles, hand pose) and action consequences makes latents plannable in several 2026 papers. Those papers are mostly single-group preprints, so we treat them as hypotheses to test.
- **Language comes on top later.** A language model turns "put the apple on the plate" into subgoals; JEPA plans each short step.
- **Honest risk:** no published image-latent planner has worked with dual 14-D dexterous-hand actions. We start with one arm and a simplified grasp, and every phase has a stop rule.

## Where the project stands

LeWM has one gated success in simulation, for one narrow job: predicting where a moving plate will be and choosing the place aim once. Everything else in the loop is a separately trained policy or hand-written code.

| Part of the task | Who does it | Status (MuJoCo sim) |
| --- | --- | --- |
| Grab the apple | P-3, a behaviour-cloning policy on frozen DINOv2 features (not LeWM) | 40/40 on cohort C (TASK-072; one run, one seed per arm; the random-init encoder control scored 39/40, so pretraining is not shown; cohort C is no longer held out) |
| Choose where to place | LeWM token predictor + plate readout + solver | 118/128 gated (TASK-081; one run, one model seed, simulation only, under the declared plate condition C1-M); TASK-083, a replication with seeds 66801 and 66802, has no recorded result yet |
| Carry and release | e9, a hand-written expert that reads simulator truth | scripted, not learned |

What this does and does not show:

- In TASK-081, W's aim (118/128) beat its action-blind twin (73/128) and its scene-blind twins (39 and 59).
- "LeWM needed" is not shown: a hand-written rule given the plate law scored 125/128, measurably better (W − H-rule −7/128, 95 % interval −13 to −2). LeWM chooses only the single place aim, on frozen DINOv2 features, with one model seed (66800, flagged `last_two_triggered`).
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
| Data: every result uses tens to hundreds of hours of task-agnostic play on the same robot | Our corpora are mostly scripted or expert-labelled single-task demos with few failures | A MuJoCo play corpus with random, perturbed and failed behaviour ([PLDM](https://arxiv.org/abs/2502.14819)); optionally Unitree's [G1_Dex3 datasets](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ToastedBread_Dataset) |
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

## Recommended plan

Six gated phases take about five months to a first zero-shot pick-and-place in simulation. Each phase has a stop rule, so a dead end costs weeks, not months. The first zero-shot claim comes in Phase 3, around week 15. Week counts are estimates for one person with AI agents; bars are examples to be fixed in each phase's preregistration.

| Phase | Weeks | Goal | Gate (stop rule) |
| --- | --- | --- | --- |
| 0. Close and calibrate | 1–2 | Record TASK-083 and close the plate line; reproduce Meta's jepa-wms numbers on this machine | Our planner within about 10 points of the released result on one task, else fix it first |
| 1. Play corpus | 3–6 | One arm and one hand; 10–50 sim hours of random, perturbed and failed behaviour, full robot state logged | Covers the test workspace and at least 20% of episodes move an object |
| 2. Robot-grounded LeWM, offline | 7–11 | Plain LeWM vs + robot state vs + joint-change and inverse-dynamics losses, on frozen DINOv2 tokens | A grounded version beats plain LeWM on action sensitivity and short-step ranking on all seeds |
| **3. Zero-shot reach and grasp** | 12–15 | Short-horizon planning to new image + pose goals never used in training | Example bars: reach 80%, grasp-and-lift 40%; stop JEPA planning if reach stays under 50% after one fix |
| 4. Pick-and-place with subgoals | 16–21 | Scripted subgoals first, then learned subgoals from the play corpus; JEPA checks each step and re-plans | Clearly beats subgoal-following without a world model; learned subgoals reach half of scripted |
| 5. Language commands | after Phase 4 | A vision-language model turns a command into subgoals; JEPA plans each short step | New commands and object layouts succeed with no per-task training |

Details that matter:

1. **Phase 0** also runs an action-sensitivity probe on our existing LeWM checkpoints, as a baseline.
2. **Phase 1** collects play data, not expert demos. Coverage and failures matter more than quality.
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
