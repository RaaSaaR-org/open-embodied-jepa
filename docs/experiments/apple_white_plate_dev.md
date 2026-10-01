# Does the plate's colour limit how well its position is read? (development, white plate)

**Status: development diagnostic.** This is not a gated run, not a preregistered gate, and not a
TASK-075 result. It was run at the owner's request ("the plate is blue - should be white. not
sure if this is the problem. but could be."). Every number here is development.

- **Learned Apple→Plate on the frozen benchmark is still 0 successes.** Nothing here changes that.
- It trains no world model and no policy, runs no controller, and makes no control claim.

## What changed, and what did not

- **Only the plate's rgba.** `src/embodied_jepa/plate_color.py` adds an opt-in parameter,
  `apply_plate_rgba(model, rgba=None)`. It recolours the compiled scene's 17 plate geoms in place:
  `plate_base`, a cylinder of radius 7 cm, and 16 rim capsules. (The v2 plate is the
  `container_kind="plate"` cylinder. The 14 × 14 cm box is the `target` kind, which v2 does not
  use.)
  - Default, `None`: a no-op. The scene stays blue (`.15 .35 .85 1`), and v2 renders and hashes
    are unchanged. `tests/test_plate_color.py` checks every model array and the render
    hashes.
  - White: `WHITE_PLATE_RGBA = (0.92, 0.92, 0.90, 1.0)`, an off-white like a ceramic plate.
  - `simulation.py` and `apple_to_plate_v2.py` are sha256-pinned by frozen manifests, so the
    parameter is a new module applied at run time, the way `apply_v2_scene` is.
- **Physics, geometry, seeds and trajectories are identical.** The white re-simulation of all 270
  train + val roots of `apple-far-shift-v2` matched every non-image array of the sealed corpus
  exactly. The arrays are `states`, `base`, `requested`, `applied`, `phase`, `apple`, `plate`,
  `dropped`, `hand_contact` and `latched`; 270/270 roots match. The onboard 112 px frames differ
  from the sealed blue frames on 7590/7590 frames, at up to 212 levels, which is the plate.
- **The 30 test roots were never simulated or decoded.** TASK-075's own source checks refuse
  them; `test_split_decoded` is false in every report.

## Harness

`scripts/dev_white_plate.py` imports TASK-075's runner (`scripts/run_obs_ceiling_v2.py`)
unmodified. That gives it TASK-075's guards, preflight, pins, frozen block, cross-fitted
R_off / R_full / R_pix, statistics and τ = 1.0 cm. The harness changes only three things:

- The render workers apply the plate colour (`white_plate_dev_runtime.py`).
- The render stage keeps the re-rendered onboard 112 px frames. TASK-075 reads that view from
  the blue sealed corpus.
- The white readouts read their reference view from that store.

`--reference-from-views` (the reference view's frames read from the harness's own store) was
used by the white readouts and by the blue re-render control readouts. It was not used by the
blue reproduction, which reads TASK-075's sealed frames. The reports of these runs predate the
flag being recorded. The harness now writes it into each report's `development` block
(`reference_from_views`) for any later run; the existing reports were not rewritten.

All runs were at `8d1ac15` on a clean tree, each under the shared heavy lock. Each started with
1- and 5-minute load ≤ 2.0, and at least 7 GiB of the GPU was free. Peak process-tree PSS was
8.6 GiB.

| run | report sha256 | outcome |
|---|---|---|
| render, white plate | `f706c75c219dae4595c36cc743460ecd6b6badd3a820c1797751f963bf18c958` | VIEWS-SEALED, 270/270 arrays equal |
| render, blue (re-render control) | `2b09443b164013832c4f2b53dee813cfd7cfcb8286e758587230729d0618bf68` | VIEWS-SEALED, 270/270 arrays equal; onboard 112: 7579 frames identical, 11 within the render rule |
| readouts, blue, TASK-075's own views (reproduction) | `270645ffc7d9286c29ff388a4c7460dd134cbafef21ad2732ca9b7bb8f63265e` | OBS-NONE; statistics, decision and error-file sha256s **identical** to TASK-075's report `96ab76b2ac7a821868955d77da165034e218fbdc02cd2fc1525ca3423f087b3f` |
| readouts, white plate | `d0e8f0c69ed63b6f3cee7182331c9c412f7edd55511731e55f5343a5e9be1ead` | OBS-NONE |
| readouts, blue re-render (noise control) | `e52b9a4df4708dc0f11d3ee50f14adf7858fd35abf9ca163bcbb76f5f88dc74a` | OBS-NONE; every number within 0.01 cm of TASK-075's |


The comparison files are `compare-white.json` (`9da27d353622b8cb14baf316cf219ae83ad3d5ebf95eed12144918fab31ad233`), `compare-blue-rerender.json`
(`0930f544377977bf5e1a787dc892cffc41e3ce27447f11533a224376c0ba2899`) and `paired-white-over-blue.json` (`a2fe2f2978705d5c74b8b09d311d2dbad3fd41bbfeb21f25120ee4d0ddb50f2a`). Run outputs are under
`outputs/white-plate-dev/` and the stores under `data/white-plate-dev/`, both git-ignored.

The paired ratios are produced by `scripts/dev_white_plate.py paired --blue <TASK-075 readouts
report> --white <white readouts report> --output <file>`. Each entry is
`obs_ceiling_v2.cluster_median_ratio(white_errors, blue_errors, clusters)` over the same 915
windows. The blue errors are TASK-075's own error files, each checked against its report's
sha256. Re-running that mode wrote `paired-white-over-blue-harness.json`, which is byte-identical
(same sha256) to the file above.

**The blue column is TASK-075's report.** The harness re-ran blue on TASK-075's own views and
reproduced every statistic exactly. A full blue re-render then moved no number by more than
0.01 cm. Its own paired ratio is 1.000 [1.000, 1.000]. That bounds the renderer's ±1–2-level pixel
noise, so differences much larger than that come from the colour.

## Results (development; 270 roots, 915 windows, cross-fitted)

**Offset readouts.** The columns are:

- *c_up*: the upper 95 % bound of the median error.
- *p87.5*: the 87.5th percentile of the error.
- *pred*: predicted successes out of 32, from the τ curve.
- *clock*: the upper bound of the median-error ratio against the clock prior.
- *white/blue*: the paired ratio of median errors, with its 95 % root-clustered interval.

All errors are in cm. τ = 1.0 cm.

| view | readout | c_up blue | c_up white | p87.5 blue | p87.5 white | pred blue | pred white | clock blue | clock white | white/blue (95 % CI) |
|---|---|---|---|---|---|---|---|---|---|---|
| onboard112 | R_off | 3.39 | 3.38 | 6.10 | 6.30 | 16.1 | 16.0 | 1.486 | 1.473 | 0.98 [0.93, 1.05] |
| onboard112 | R_full | 2.70 | 2.67 | 5.61 | 5.43 | 18.3 | 18.5 | 1.167 | 1.164 | 1.00 [0.94, 1.04] |
| onboard112 | R_pix | 2.43 | 2.37 | 5.20 | 5.18 | 19.5 | 19.7 | 1.036 | 1.018 | 0.99 [0.95, 1.03] |
| onboard224 | R_off | 3.32 | 3.43 | 6.32 | 6.41 | 16.4 | 16.1 | 1.444 | 1.490 | 1.04 [0.98, 1.11] |
| onboard224 | R_full | 2.61 | 2.56 | 5.73 | 5.75 | 18.8 | 19.0 | 1.122 | 1.103 | 1.00 [0.95, 1.06] |
| onboard224 | R_pix | 2.35 | 2.23 | 5.08 | 5.14 | 19.8 | 20.0 | 1.002 | 0.966 | 0.98 [0.93, 1.01] |
| handcrop | R_off | 2.66 | 2.57 | 4.65 | 4.42 | 18.6 | 18.9 | 1.145 | 1.128 | 0.99 [0.95, 1.03] |
| handcrop | R_full | 2.45 | 2.43 | 4.50 | 4.33 | 19.5 | 19.6 | 1.058 | 1.068 | 1.01 [0.98, 1.05] |
| handcrop | R_pix | 2.94 | 3.02 | 5.35 | 5.34 | 17.4 | 17.3 | 1.282 | 1.309 | 1.02 [0.99, 1.06] |
| overview224 | R_off | 3.44 | 3.34 | 6.63 | 6.70 | 16.2 | 16.4 | 1.486 | 1.453 | 0.98 [0.90, 1.05] |
| overview224 | R_full | 2.73 | 2.66 | 5.67 | 5.31 | 18.5 | 18.7 | 1.174 | 1.145 | 0.98 [0.91, 1.06] |
| overview224 | R_pix | 2.02 | 2.08 | 4.35 | 4.41 | 21.0 | 20.9 | 0.867 | 0.897 | **1.045 [1.00015, 1.093]**, excludes 1.0 |

**The plate's and the apple's own position.** These are TASK-075's reported-only pooled-token
readouts: the same features as R_off, but each regresses one object's world xy. They show
whether colour changes plate legibility, which is the only part a plate colour can affect.

| view | target | median blue | median white | c_up blue | c_up white | p87.5 blue | p87.5 white | white/blue (95 % CI) |
|---|---|---|---|---|---|---|---|---|
| onboard112 | plate | 0.493 | 0.458 | 0.51 | 0.50 | 0.91 | 0.91 | 0.93 [0.89, 1.01] |
| onboard112 | apple | 3.12 | 3.13 | 3.28 | 3.34 | 5.83 | 6.09 | 1.00 [0.95, 1.07] |
| onboard224 | plate | 0.510 | 0.478 | 0.54 | 0.50 | 0.99 | 0.94 | 0.94 [0.88, 1.01] |
| onboard224 | apple | 3.04 | 3.10 | 3.21 | 3.32 | 5.96 | 6.10 | 1.02 [0.96, 1.09] |
| handcrop | plate | 0.682 | 0.654 | 0.72 | 0.70 | 1.36 | 1.35 | 0.96 [0.91, 1.01] |
| handcrop | apple | 2.21 | 2.16 | 2.37 | 2.31 | 4.37 | 4.32 | 0.98 [0.92, 1.03] |
| overview224 | plate | 0.550 | 0.542 | 0.59 | 0.60 | 1.07 | 1.05 | 0.99 [0.91, 1.08] |
| overview224 | apple | 3.07 | 2.99 | 3.24 | 3.24 | 6.36 | 6.46 | 0.97 [0.92, 1.06] |

The row stays OBS-NONE with a white plate: no view, readout or colour reads the offset within
τ = 1.0 cm.

**The plate-hidden check (TASK-075's A4/B4) moves with the colour.** The plate-hidden frames are
the same pixels in both colours: the plate is not drawn, and they differ by at most 1 level. The
readouts are refitted on each colour's visible frames, though, so their plate-hidden error
changes. The table gives the median plate-hidden error in cm, blue → white. The blue re-render
control reproduces the blue column to within 0.01 cm.

| view | R_off | R_full | R_pix |
|---|---|---|---|
| onboard112 | 7.14 → 6.07 | 7.06 → 6.08 | 4.35 → 4.36 |
| onboard224 | 7.23 → 6.26 | 5.65 → 5.80 | 4.18 → 4.17 |
| handcrop | 3.81 → 3.77 | 3.51 → 3.54 | 5.17 → 4.99 |
| overview224 | 7.86 → 4.26 | 5.28 → 4.44 | 3.75 → 3.93 |

Read this as a distribution effect of the readout, not a change in what the frames show. Removing
a saturated blue plate from a light table is further out of the training distribution than
removing an off-white one. A4 and B4 still pass for every view and readout in both colours: they ask only
that the plate-hidden error's lower 95 % bound exceed τ = 1.0 cm. With white, though, the
plate-hidden error of overview224 R_off (4.26 cm) is not far above its visible-frame median
(3.15 cm). It is nonetheless a real colour effect on a
gating check.

## Reading (development)

**On this evidence, the plate's colour is not what limits the readout.**

- The paired intervals bound any colour effect on the offset's median error to roughly
  0.90–1.11 (−10 % to +11 %). This is a bound, not a finding of no difference. The largest shift
  in a median upper bound is 0.12 cm, against a gap to τ of 1.0–2.4 cm.
- 1 of the 12 offset intervals excludes 1.0: overview224 R_pix, 1.045 [1.00015, 1.093]. There,
  white is about 4.5 % worse. The exclusion is marginal and there is no multiplicity correction;
  with 12 intervals, about one such exclusion is expected by chance. The renderer noise does not
  explain it (the control's ratio is 1.000). It does not change OBS-NONE.
- White is nominally better on the plate's own position in the onboard views (−6 to −7 % in the
  median); those intervals reach 1.0. In either colour, the reported-only plate readout's median
  error is 0.46–0.68 cm, at or below τ. That holds for the medians only: the plate's p87.5 reaches
  1.35–1.36 cm on the hand crop (and 1.05–1.07 cm on overview224), above τ.
- The offset error is mostly the apple's: its position is read to 2.2–3.1 cm in both colours.
  That is the TASK-075 results finding, and the plate colour does not move the apple readout
  (every apple interval includes 1.0). The colour does move the plate-hidden check (above),
  which is about how the refitted readout extrapolates, not about how precisely the offset is
  read.

## Caveats

- Development only: one corpus (`apple-far-shift-v2`, train + val roots), one off-white value,
  one renderer and lighting, and no preregistration. The paired intervals are reported, not gated.
- Linear readouts only: R_off and the target readouts are ridge on pooled frozen DINOv2 tokens.
  R_full and R_pix are TASK-075's linear/RBF kernel readouts. A different encoder or a trained
  readout might use colour differently.
- The white plate is on a light-brown table in simulation lighting. Its near-camera shading is
  grey (see the contact sheet). Real-world contrast may differ.
- The white run's reference view comes from its own re-render, whereas TASK-075's comes from the
  sealed frames. The blue re-render control shows that this matters by no more than 0.01 cm.
