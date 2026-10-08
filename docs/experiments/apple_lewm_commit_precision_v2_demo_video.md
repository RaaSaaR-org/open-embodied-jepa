# TASK-081 demo video (illustration, not evidence)

This is a short MP4 of TASK-081's arms running in closed loop. It is there for watching and is
not a result. It is not a gated run, nothing in it is read, and it does not change any count. For
the gated result and the caveats that must come with it, see
[apple_lewm_commit_precision_v2_results.md](apple_lewm_commit_precision_v2_results.md). Neither
that document nor the frozen protocol is edited here.

## What the video shows

The task is `apple-to-plate-v2` under the simulation-only condition C1-M. Each attempt runs in
four steps:

1. P-3, the behaviour-cloned policy, picks up the apple.
2. The plate moves at step 300.
3. At step 405 the arm commits a single place aim.
4. e9's scripted place puts the apple at that aim while the plate reacts to the hand.

The arms shown are:

- **W**: TASK-081's LeWM token predictor. It runs on frozen DINOv2 features with model seed
  66800, the R-S readout and the `affine_local` solver.
- **N**: W's action-blind twin.
- **H-rule**: the hand-written rule. It is given the simulator's plate law and is not learned.

Two markers are drawn on the third-person view:

- **Green marker**: the aim the arm committed.
- **Orange disc**: the scorer's 4 cm radius around the place where the plate stopped (step 525).
  This position comes from the screen run of the same attempt.

The right-hand panel shows the onboard 112 px frame that the controllers read.

## How it was made

`scripts/render_lewm_cp_v2_demo.py` is a new standalone script. It does not edit any pinned file.

- **Library reuse.** The script loads `scripts/run_lewm_cp_v2.py` as a library: the artifact
  checks by sha256, the worker configuration, the P-readout refit and the cohort estimates. It
  runs no stage of that runner.
- **Worker.** It runs TASK-081's worker `lewm_cp_v2_runtime` unchanged.
- **Seeds.** It uses only debug seeds from TASK-081's debug block, 71900–71999. Seeds 71920–71931
  are the default.
- **Screen.** It first screens W, N and H-rule on those seeds without rendering.
- **Clip selection.** The selection rule was fixed in the script before its first run:
  - Seed A is the first screened seed where W succeeds and N misses.
  - Seed B is the first other seed where W succeeds.
  - The clips are W on B, then W, N and H-rule on A.

  Because the clips are chosen this way, they show contrasts and are not a sample of typical
  behaviour.
- **Rendering.** The script re-runs each chosen attempt in-process to render it. For that run it
  replaces two per-attempt helper classes of `lewm_c1m_v2_runtime` in its own process only, so
  that each observation also renders a free third-person camera and reads the committed aim. They
  change no command, plate motion or controller input.
- **Check against the screen.** `report.json` records whether each rendered attempt matched its
  screen run in success and committed aim.
- **Encoding.** Frames are encoded with the system `ffmpeg`. No dependency was added.

Each run writes its outputs, `report.json` and `SHA256SUMS` to a new directory and refuses to
reuse an existing one. The first run went to `~/develop/emai/evidence/task081-demo-video/`, which
is not committed.

## The rendered video

The video was rendered on 2026-10-08 at script revision `8e37870`, with a clean tree, on the
Linux PC, on the CPU with EGL rendering, under `scripts/gpu_run.sh`.

- **File:** `task081_lewm_demo.mp4`, 57.8 s at 896 × 480 px and 30 fps.
- **sha256:** `fbd3f96df6954b8589b95797ecb15feacca4e316893921ba6f3a482172c09f64`. The other
  files are listed in `SHA256SUMS`.
- **Screen.** The screen covered debug seeds 71920–71931. Its counts are not evidence: 12
  debug resets, one run, and resets that are not a test cohort. W succeeded on 12/12, N on 4/12
  and H-rule on 12/12.
- **Clips.** Applying the rule gave seed A = 71920 and seed B = 71921. Every rendered attempt
  matched its screen run.

| Clip | Arm | Seed | Outcome | Apple to plate centre (cm) |
|---|---|---|---|---|
| 01 | W | 71921 | at rest on the plate | 3.8 |
| 02 | W | 71920 | at rest on the plate | 3.1 |
| 03 | N | 71920 | miss | 4.5 |
| 04 | H-rule | 71920 | at rest on the plate | 3.4 |
