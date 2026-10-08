# TASK-081 demo, replayed in Isaac Sim (illustration, not evidence)

This is a video of the TASK-081 demo episodes ([demo note](apple_lewm_commit_precision_v2_demo_video.md))
shown in the GR00T tutorial's Isaac Lab-Arena shelf scene ([ARENA.md](../ARENA.md)). It is an
**Isaac Sim kinematic replay of MuJoCo episodes, not an Isaac run of the controller**:

- Every attempt ran in MuJoCo, exactly as in the demo. LeWM's arm W was trained and evaluated
  on MuJoCo images, so running it closed-loop in Isaac would be a different experiment that
  has not been validated. That was not done.
- Isaac's physics never runs. The timeline is never played. Each frame sets every robot body,
  the apple and the plate to the pose MuJoCo recorded at that control step, and then renders.
  Contacts, the grasp, the plate's motion and the outcome are all MuJoCo's.
- The controller never saw these images. Its input was MuJoCo's 112 px onboard frame, which
  the Isaac-only video shows in its side panel.

Nothing here is a result. It is not a gated run, it changes no count, and the frozen protocol
and the results document are not edited. For the gated result and the caveats that go with it,
see [apple_lewm_commit_precision_v2_results.md](apple_lewm_commit_precision_v2_results.md).

## What is shown

The clips are the demo's: W on debug seed 71921, then W, N and H-rule on debug seed 71920. They
were chosen by the demo's fixed rule after a screen, so they show contrasts and are not a sample
of typical behaviour. The captions, the green marker (the committed place aim) and the orange
ring (the scorer's 4 cm radius around where the plate stopped) are the same as in the demo. The
outcome banner says "(in MuJoCo)".

The scene:

- **Background.** Arena's `galileo_locomanip` at Arena's pose. The three boxes that Arena's
  `galileo_g1_static_pick_and_place` deactivates are also deactivated here. It is a warehouse
  shelf, not a kitchen. Some of its MDL materials do not resolve in this image ("EntityResolver
  FAILED", as in ARENA.md §4), so some surfaces, such as the bins under the shelf, may not show
  their authored colours.
- **Robot.** Our G1 + Dex3 USD (`g1_29dof_with_hand-ref`, the MJCF conversion of
  [ISAAC_MJCF_TRANSPORT.md](../ISAAC_MJCF_TRANSPORT.md)), with its Physics variant set to
  `none`. NVIDIA's `g1_29dof_with_hand_rev_1_0.usd` from the tutorial is not used, because its
  Dex3 fingertips differ from the MJCF's by up to 3.1 cm (ARENA.md §7). The converted USD
  nests its body prims like the MJCF's body tree. Each frame sets every body prim's local
  transform to MuJoCo's parent-relative body pose, computed from the recorded world poses, so
  the posed robot matches MuJoCo's to float precision. The script refuses to run if the USD
  tree differs from the MJCF's.
- **Apple and plate.** These are Arena's meshes (`apple_01`, the cream HOT3D `clay_plates`),
  rescaled to our sizes: an apple diameter of 5.4 cm and a plate diameter of 14.2 cm. They are
  posed at MuJoCo's apple and plate body poses. They are not our red sphere and blue plate.
- **Frame map.** Our world maps to Arena's env-local frame by a translation of
  (0.09, 0.08, −0.77) m:
  - our table top (z 0.74) goes to Arena's shelf top (z −0.030);
  - our table's front edge (x 0.13) goes to the front edge of Arena's shelf support (x 0.22);
  - y keeps the y of Arena's robot (0.08).

  Our pelvis is fixed, so the table height is matched and the feet hang about 2.5 cm above
  Arena's floor. Our table itself is not drawn; Arena's shelf stands in for it.

## How it was made

There are two new scripts. No pinned file was edited (`run_lewm_cp_v2.py`, `lewm_cp_v2*.py`
and the TASK-073 to TASK-081 runners and modules are unchanged).

1. `scripts/isaac_replay_lewm_cp_v2_demo.py dump` (host, CPU, MuJoCo with EGL).
   - It re-runs the four demo attempts with TASK-081's worker, unchanged, through the demo
     script's render hooks.
   - For every control step it writes the world pose of every MuJoCo body and the committed
     aim. For the rendered steps (every third step, plus steps 300 and 406) it also writes the
     MuJoCo third-person frame and the 112 px onboard frame.
   - It runs no screen. Each outcome is checked against the demo's `report.json`, and all
     four matched.
2. `scripts/isaac/replay_kinematic_arena.py` (container, through `scripts/isaac/run_isaac.sh`).
   - It builds the stage above, with no physics, and renders a 640 × 480 third-person camera
     for each rendered step. The camera is over the robot's right arm, at eye (0.05, −0.65,
     0.65) looking at (0.50, −0.03, −0.05) in Arena's env-local frame, with an 18 mm focal
     length. It was picked from ten candidates rendered at step 405 of one clip. Rendering uses
     RTX real time, with 8 Kit updates per frame and the timeline stopped.
   - `run_isaac.sh` gained an optional read-only input mount for this: `ISAAC_INPUT_DIR` is
     mounted at `/oej/in`.
3. `scripts/isaac_replay_lewm_cp_v2_demo.py compose` (host).
   - It captions the Isaac frames as in the demo and adds the label "Isaac Sim kinematic replay
     of MuJoCo episodes, no Isaac physics - illustration only, not evidence" to every frame,
     including the side-by-side ones. The intro card adds that it is not an Isaac run of the
     controller.
   - It encodes two MP4s with the system `ffmpeg`: Isaac only, and MuJoCo | Isaac side by side.

## The rendered video

The video was rendered on 2026-10-08 on the Linux PC. The dump ran at revision `75835f6` with a
clean tree, on the CPU with EGL rendering, under `scripts/gpu_run.sh`. The Isaac render ran at
`bbc83a8` in the `isaaclab_arena:latest` container (§1 of ARENA.md), under
`scripts/gpu_run.sh --container oej-isaac-`. The only uncommitted file at that point was this
note, which was untracked. The final compose ran in review, after the branch was rebased onto
main, at `e328b46` with a clean tree, from the same dump and Isaac frames (`dump`,
`isaac-replay-1`); that commit only changed the per-frame label. Short hashes before the
rebase name the branch's original commits.

- **Files.** The outputs are in `~/develop/emai/evidence/task081-isaac-replay-video/`, which is
  not committed. `SHA256SUMS` there lists every file.
  - `video-3/task081_lewm_isaac_replay_isaac.mp4`: 58.8 s at 896 × 480 px and 30 fps. It has
    the same layout as the demo, with the Isaac view in place of MuJoCo's. The sha256 is
    `b6d55190271e0e796669395057eb3e5102ff90577a283c5e6dc15e780fbc11dd`.
  - `video-3/task081_lewm_isaac_replay_pair.mp4`: 58.8 s at 1280 × 546 px. MuJoCo is on the
    left and Isaac on the right. The sha256 is
    `e01e12db3ae01f1a4fef7046d8c54092d88df31871cf64511089b96b91a13ebe`.
- **Outcomes.** All four re-run attempts matched the demo's outcome and distance: W on 71921
  at rest, 3.8 cm from the plate centre; W on 71920 at rest, 3.1 cm; N on 71920 a miss,
  4.5 cm; H-rule on 71920 at rest, 3.4 cm. These are debug seeds, and the counts are not
  evidence.
- **Development runs that did not produce the video.** These are kept in the evidence
  directory:
  - `dump-failed-1`: a missing report key, fixed.
  - `isaac-probe-1`: a segmentation fault in the RTX driver library during start-up. It did
    not recur.
  - `isaac-probe-2` and `isaac-probe-3`: an exception on the referenced apple's float scale op.
    It was hidden by Kit's exit, which is why tracebacks are now printed.
  - `isaac-probe-4`: hung in Replicator's `orchestrator.step` with the timeline stopped, then
    stopped by hand (its own container only). This is why rendering now uses Kit updates.
  - `isaac-probe-5` and `isaac-probe-6`: the camera candidates.
  - `video-draft-1`: an earlier compose whose outcome banner was too wide.
  - `video`: the compose at `67bd542`. Its per-frame label said only "not an Isaac run of the
    controller", so the side-by-side frames did not say "illustration only, not evidence";
    superseded by `video-3`.
  - `video-2`: the same compose as `video-3`, run before the label change was committed.

  The containers' asset download caches (`home_cache`, about 1.2 GB each) were deleted after
  the runs.
