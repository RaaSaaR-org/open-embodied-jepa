# Manipulation controller v1: bounded mechanical reachability probe

Hypothesis declared before execution: pilot velocity stops often occur in the right Dex3 thumb, not the arm. Cube transfer examples had right_hand_thumb_1_joint velocities −5.03 to −5.55 rad/s; banana lift and apple closure reached −6.27 and −5.12 rad/s. Full synergy compression followed by slipping contact may cause these spikes. This probe changes requested motion and closure only; all runtime gains, collision physics, torque limits and the 5 rad/s sampled safety stop remain unchanged.

Six fixed trials, seeds 2000–2005, all start the cube at (0.34, −0.18) and the container at (0.45, −0.10). The current overlap validator must accept every reset. Cube→plate then cube→target each receive the following three configurations in order:

1. Original 805-command oracle.
2. Slow full closure: twice every phase duration, translation amplitude at most 0.2, rotation amplitude at most 0.25, grasp ramp 0.04 per accepted command, closure +1.
3. Slow gentle closure: identical slowed motions, grasp ramp 0.03, closure +0.5.

The total wall budget is 600 seconds, counting host suspension using the maximum of monotonic and civil elapsed clocks. Stop executing at 570 seconds to reserve finalization time. No retries or tuning after outcomes. Apple→plate is excluded. Success uses the frozen ordered `AppleToPlateTask` scorer: real finger-contact lift, transport, release and supported rest. Report all failures. This is privileged simulated mechanical feasibility, not learned control or an estimate of reliability.

Reproduce from the repository root:

```sh
.venv/bin/python scripts/spikes/grasp_probe.py --controller-v1 --output outputs/manipulation_controller_v1
```

The output refuses overwrite and saves a pre-execution plan, exact source snapshots/hashes, all trial traces, phase-end RGB, and complete RGB/state/applied-action transitions at 20 Hz. These exploratory artifacts are separate from the immutable manipulation pilot corpus.

## Observed result

All six predeclared trials ran once in **48.09 true-wall seconds**. Ordered task success was **0/6**. The tested changes do not establish current-code mechanical reachability and do not justify promoting the slow policy as a successful demonstration controller.

| Seed | Pair | Candidate | Accepted commands | Outcome |
|---|---|---|---:|---|
| 2000 | cube→plate | original | 549 | Contact lift and transport; velocity stop during transfer |
| 2001 | cube→plate | slow full | 451 | Velocity stop during closure; no lift |
| 2002 | cube→plate | slow gentle | 1610 | All phases completed; object never grasped |
| 2003 | cube→target | original | 549 | Contact lift and transport; velocity stop during transfer |
| 2004 | cube→target | slow full | 458 | Velocity stop during closure; no lift |
| 2005 | cube→target | slow gentle | 469 | Velocity stop during closure; no lift |

All five stops involved `right_hand_thumb_1_joint`: −5.778, −5.016, −5.778, −5.737 and −5.359 rad/s in trial order excluding seed2002. Original policies reached object center height0.8673m and satisfied the scorer's contact-lift and transport stages. The gentle plate run contains release-phase commands, but no successful object release demonstration. No reset was rejected, no body was welded or moved directly after reset, and no safety threshold was changed. The two container geometries also produce different closure outcomes; this small deterministic probe is not a reliability estimate.

Artifacts: `outputs/manipulation_controller_v1/plan.json`, `report.json`, six `trial_*/trace.json` and `transitions.npz`, and exact executable source snapshots under `sources/`. There are **4086 complete transitions**, with96×96 RGB and86 named proprioceptive state values at20Hz. They remain exploratory NPZ artifacts, separate from the frozen canonical LeRobot corpus. Verified every trial hasT+1 images/states/timestamps forT applied actions,50ms timestamp increments, finite state/raw actions, and matching source snapshot hashes. The probe retains the rejected final request in each failed trial's trace but does not label it as an applied transition.

Plan SHA-256: `5944cf53ed1a3eab06ac9142158e5de4a09abab6f67e466d8d25d7adc58869a0`.
Report SHA-256: `ac7dc9dfb1450d9ab7cd2b28e436c4e26dab0a21a0226b6fdfc32e4b3c601611`.

Validation also exercised slow-policy request limits, accepted-grasp ramp progression, doubled command budget, and rejection termination using an isolated kinematic fixture. Ruff checks pass. This result narrows the issue to contact/hand closure dynamics; additional controller or contact-model development requires a separately declared experiment. Historical successes at reset layouts now rejected by the conservative overlap validator do not close this current-code gate.
