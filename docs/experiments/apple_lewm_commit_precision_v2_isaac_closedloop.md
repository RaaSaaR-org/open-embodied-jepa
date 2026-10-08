# TASK-081's controller in closed loop in Isaac Sim (development only, not evidence)

**Isaac Sim closed-loop, development only, not evidence; LeWM chooses only the place aim; trained
on MuJoCo images.** Nothing here is a gated run or a result. It changes no count, it ran on
TASK-081's debug seeds only (71920–71927, in the block 71900–71999 whose outcomes the protocol
never reads), and the counts below are a handful of attempts each, labelled non-evidence. The
frozen protocol, the results document and every hash-pinned runner and module
(`run_lewm_cp_v2.py`, `lewm_cp_v2*.py` and the TASK-073 to TASK-081 chain) are unchanged. For
the gated result and its caveats see
[apple_lewm_commit_precision_v2_results.md](apple_lewm_commit_precision_v2_results.md).

Unlike the [kinematic replay](apple_lewm_commit_precision_v2_isaac_replay.md), here **Isaac steps
the physics and renders every image a controller reads**.

## What ran

- **Isaac (container).** `scripts/isaac/lewm_cp_server_isaac.py` builds `IsaacTransport` with our
  G1 + Dex3 (the MJCF conversion `g1_29dof_with_hand-ref`, canonical tree `cd1fdb27…2fa8`), the
  v2 table, apple and plate (`apple_to_plate_v2_scene_v1.json`) and the pinned onboard camera
  (`onboard_camera_v1.json`, `866c596a…`: `onboard_rgb` on `torso_link`, the same pose, 75°
  fovy and 112 × 112 as MuJoCo's, path traced at 64 spp). It adds a world-fixed third-person
  camera for the video only. Physics: Isaac Lab's Newton backend (MuJoCo-Warp,
  [ISAAC_NEWTON_SPIKE.md](../ISAAC_NEWTON_SPIKE.md)) and, in one run, PhysX. This is our v2
  scene in Isaac, not the Arena warehouse: the onboard camera then sees the same geometry as
  MuJoCo's (ISAAC_V2_SCENE §1: mask IoU ≥ 0.97, but MAD 26–29/255 in shading).
- **Host.** `scripts/isaac_lewm_cp_closedloop.py` runs TASK-081's worker unchanged
  (`lewm_cp_v2_runtime.run_attempt_task`: P-3's pick, the C1-M plate move at step 300 and
  reactive rule, W's or H-rule's single aim at 405, e9's scripted place, `apple_at_rest_v0`) on
  an `isaac_e9.MirrorSimulation` subclass whose physics is Isaac's (the e9-replay mirror,
  [ISAAC_E9_REPLAY.md](../ISAAC_E9_REPLAY.md) §1). The C1-M law teleports the plate in the mirror,
  and the mirror forwards each plate pose to Isaac's kinematic plate before the next render or
  physics step. `render()` returns Isaac's onboard frame.
- **Two in-process replacements** (nothing pinned is edited): the post-look frame check
  (`check_post_look_frame` compares against MuJoCo's sha256) is replaced by computing P-3's
  post-look estimates from Isaac's post-look frame with the M2 P readout (refitted as the demo
  does, G-repro passed, cached in `p_readout_refit-1.pkl`); the per-attempt wall cap is raised
  to 3600 s.
- **Domain-gap record.** The mirror holds Isaac's state in a local MuJoCo model, so MuJoCo can
  render the *same* state. Readouts are computed on both frames; only Isaac's is ever used. At
  405 W's aim is also computed on MuJoCo's frame of the same state (counterfactual, logged,
  never used).
- **Plumbing check** (`--sham`, `sham-1`): the same client against a host MuJoCo endpoint
  reproduced the MuJoCo demo's screen exactly on 71920 (W 3.0524 cm with the same aim, H-rule
  3.4317 cm).

Run at `1f61162` (clean) with `isaaclab_arena:latest` `sha256:2588b526…`, under
`scripts/gpu_run.sh --wait --min-free-gib 8 --board --container oej-isaac-` via
`scripts/isaac/run_lewm_cp_closedloop.sh`. Step A of the domain gap ran at `a8a2d58` (clean).
Under Newton, Isaac took a median 0.03 s per control step and 0.22 s per render, and an attempt
that reached the place ran in 3.4–5.2 min; under PhysX, 0.18 s per step and 5.6–5.9 min.

## Step A: the domain gap, offline (8 debug seeds)

Post-look state (reset + look) and the same state with the C1-M move teleported, 16 frames, Isaac
physics (Newton). Errors against the true positions, median (min–max):

| Readout | Isaac frame | MuJoCo render of the same state |
| --- | --- | --- |
| R-plate (W's p̂ and H-rule's reading), plate, 16 frames | **2.42 cm** (1.35–4.30) | 0.94 cm (0.55–1.66) |
| P-3's post-look apple estimate, 8 frames | **11.4 cm** (8.7–11.9) | 0.42 cm (0.16–1.03) |
| P-3's post-look plate estimate, 8 frames | 2.09 cm (1.05–2.88) | 0.12 cm (0.03–0.17) |
| Mean absolute pixel difference Isaac vs MuJoCo | 26.7/255 | — |

On Isaac's frames the P readout returns nearly the same apple for every seed (x 0.30–0.32,
y −0.05 to −0.09 m against true y −0.16 to −0.20 m). R-plate's error is near the 3 cm line set for
this check; P-3's apple estimate is far beyond it, so **P-3's pick was expected to fail** before step B.
At 405 in step B (the Newton e9-pick run) R-plate read the plate 1.9–3.2 cm off on Isaac's frame against
0.07–0.33 cm on MuJoCo's render of the same state.

## Step B: closed loop (debug seeds 71920–71923)

Per attempt: at rest by `apple_at_rest_v0` on Isaac's state (yes/no), final apple–plate distance.
"Aim miss" is the committed aim's distance from where the plate ended (step 525).

**1. As specified: P-3 picks from Isaac-frame estimates (Newton, `newton-p3-1`).** All 8 attempts
(W and H-rule × 4 seeds) stopped at steps 221–231 on a guard refusal while P-3 reached for the
mis-estimated apple. No grasp, so no place aim was ever committed. 0/4 W, 0/4 H-rule.

**2. Variant: e9's privileged scripted pick, then the same chain (Newton, `newton-e9pick-1`).**
Labelled on every frame "VARIANT: e9's scripted privileged pick". All 8 grasped and placed.

| Seed | W | W aim miss | W aim shift if it had read MuJoCo's frame | H-rule | H-rule aim miss |
| --- | --- | --- | --- | --- | --- |
| 71920 | no, 4.03 cm | 0.33 cm | 0.19 cm | no, 4.36 cm | 2.54 cm |
| 71921 | **yes**, 3.28 cm | 1.01 cm | 0.42 cm | no, 4.57 cm | 1.95 cm |
| 71922 | no, 4.45 cm | 0.54 cm | 0.24 cm | **yes**, 3.60 cm | 2.11 cm |
| 71923 | **yes**, 3.83 cm | 1.08 cm | 0.97 cm | no, 4.21 cm | 3.10 cm |

W 2/4, H-rule 1/4 (non-evidence; four seeds). Every miss is a supported apple 4.0–4.6 cm from the
plate centre, just outside the 4 cm disc, the same failure kind as e9 in Isaac/Newton
(ISAAC_E9_REPLAY §2). In MuJoCo, with P-3's own pick, the same seeds gave W 4/4 (3.05–3.99 cm) and H-rule 4/4
(2.86–3.98 cm) in the demo's screen, also non-evidence and also close to the 4 cm edge.

**3. Diagnostic: P-3 with estimates from MuJoCo's rendering of the same Isaac state
(Newton, `newton-p3mjest-1`).** Labelled "DIAGNOSTIC" on every frame; it is not the Isaac
pipeline, because P-3's estimates do not come from Isaac's image. W only: P-3 grasped on 2 of 4
(71921, 71923); W at rest 1/4 (71923, 3.50 cm; 71921 4.53 cm; the two without a grasp ended
15.7 and 19.7 cm away). So even with good estimates P-3's pick is less reliable under Isaac's
physics than in MuJoCo (where it grasps on all of these seeds).

**4. PhysX: e9-pick variant (`physx-e9pick-1`).** Run at `ec6a2a4` (clean; it differs from `1f61162` only in the video banner). All 8
attempts grasped and placed, and the latched scorer (`AppleToPlateTask`) fired in all 8, but **none
passed `apple_at_rest_v0`: W 0/4, H-rule 0/4**. Final distances: W 4.57, 4.23, 4.66 and
1.33 cm; H-rule 4.63, 3.12, 4.45 and 4.59 cm. Two attempts (W 71923 at 1.33 cm, H-rule 71921
at 3.12 cm) ended inside the 4 cm disc and still failed the at-rest check on another criterion;
the per-criterion detail was not kept in this client's record, so which one is not known.
PhysX has no rolling friction for the apple (ISAAC_V2_SCENE §3, ISAAC_NEWTON_SPIKE), which is
the obvious candidate; that is untested. W's aim missed the plate's final position by 0.4–1.4 cm
(H-rule 1.7–3.0 cm), with R-plate at 405 reading 1.8–3.1 cm off on Isaac's frame.

## What this says (development reading only)

- **The loop runs end to end in Isaac** with Isaac physics and Isaac-rendered onboard frames, at
  about 4 minutes per attempt under Newton (about 6 under PhysX).
- **P-3 does not transfer.** Its post-look apple estimate is about 11 cm off on Isaac's frames,
  so the learned pick fails in every attempt as specified. With MuJoCo-quality estimates it
  still grasped only 2 of 4 under Newton.
- **W's aim transfers better than its inputs suggest.** With e9's pick, W's aim landed 0.3–1.1 cm
  from where the plate ended although R-plate read the plate 2–3 cm off; reading MuJoCo's
  frame of the same state would have moved W's aim by only 0.2–1.0 cm. H-rule, which uses
  R-plate's reading directly, missed by 2.0–3.1 cm. Four seeds; not a comparison.
- **PhysX did worse than Newton** at the place (0/8 at rest against 3/8), consistent with the
  earlier finding that PhysX lets the apple roll where MuJoCo and Newton stop it.
- **The place is the limiting step here**: in the Newton e9-pick runs e9's place left the apple
  3.3–4.6 cm from the plate centre, so a 1 cm aim error decides the 4 cm verdict, as for e9 alone
  in Isaac.

## Caveats

- Four debug seeds per cell, one run each, no repeat; Isaac's own run-to-run spread at the 4 cm
  edge is large (ISAAC_E9_REPLAY §2: 10 of 32 at-rest verdicts flipped between two identical
  runs). None of the counts is a rate or a comparison.
- The e9-pick variant reads simulator truth for the pick (privileged, scripted); only the 405 aim
  is the learned or hand-written arm's. The diagnostic gives P-3 estimates from a MuJoCo
  rendering, so it is not an Isaac-perception run.
- Newton is Isaac Lab's MuJoCo-Warp backend, built by the transport with MuJoCo's contact
  parameters; PhysX is Isaac's native engine and lacks rolling friction. The scene is our v2
  table scene with our MJCF-converted robot, fixed pelvis, not NVIDIA's Arena robot or WBC.
- MuJoCo's rendering "of the same state" uses the mirror's `mj_step` layout (bodies at the start
  of the interval's last 2 ms physics step).
- Project status is unchanged (DECISIONS R7 as updated by R19.21); nothing here is evidence
  about TASK-081's gated result.

## Files

Evidence (not committed): `~/develop/emai/evidence/task081-isaac-closedloop-dev/`, with
`SHA256SUMS` over every file. Videos (1120 × 536, 30 fps; Isaac third-person view with the
committed aim in green and 4 cm around the live plate in orange, Isaac's onboard frame, and
MuJoCo's rendering of the same state for comparison; the label above on every frame):

- `task081_isaac_closedloop_newton_p3_then_e9pick.mp4` (157.7 s, sha256 `83f5b89e…6497`): the
  as-specified run (P-3's pick fails) followed by the e9-pick variant, Newton. Start here.
- `newton-p3-1/task081_isaac_closedloop.mp4` (46.1 s, `9da050a4…43e3`): as specified, Newton.
- `newton-e9pick-1/task081_isaac_closedloop.mp4` (111.6 s, `6da21808…177f`): e9-pick, Newton.
- `newton-p3mjest-1/task081_isaac_closedloop.mp4` (58.8 s, `9b0af587…834c`): the diagnostic.
- `physx-e9pick-1/task081_isaac_closedloop.mp4` (111.6 s, `166de5d5…b234`): e9-pick, PhysX.

Each run directory also has the per-attempt clips, `report.json` (every attempt's outcome, the
committed aim, the post-look and 405 domain-gap rows and W's counterfactual aim) and its own
`SHA256SUMS`; `step_a_frames/` holds the Isaac and MuJoCo onboard frames of step A.

Other development runs, not listed above: `isaac-newton-1` (client stalled in the P readout
refit under heavy CPU contention from another job; stopped, own container stopped by hand; no
video), `isaac-newton-2` (fewer BLAS threads broke the refit's G-repro check; it stopped itself;
no video), and `isaac-newton-3` (step A above, plus a one-attempt check with its own short video:
W on 71920 stopped by the same guard refusal at step 223 as in `newton-p3-1`). `sham-1` is the
plumbing check. `physx-e9pick-1` also recorded its own 4-seed step A under PhysX (consistent with
the table above; not tabulated here). The containers' asset caches (`home_cache`) were deleted after the runs.
