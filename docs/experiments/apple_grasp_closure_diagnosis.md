# Apple grasp closure: diagnosis of the v2 close-phase ejection (TASK-051)

**EXPLORATORY. This document is a diagnosis, not gate evidence.** Every run reported
here is a NON-LEARNED privileged diagnostic on **TRAIN-side tuning resets only**
(49100–49131, drawn with `evaluate_apple.wide_reset`, used nowhere else in the
repository). No gated reset range was stepped: not 45000–45007 (TASK-047), not
45100–45107 (TASK-049), not the TASK-051 gate range 45200–45207, not the final cohort
44000–44019, and not TEST. Nothing here is a learned result; learned Apple→Plate
remains at zero successes.

## The question

TASK-049's object-aware exact-rollout ceiling v2 failed its primary gate: 5/8 full
successes and 5/8 grasps on the fresh resets 45100–45107 against the required 6 and 7,
while the scripted collector succeeded 8/8. All four failures (three primary, one
secondary) were a single mode — 8–9 commands into the `close` phase the closing fingers
ejected the apple — and the cause was not isolated. The entry geometry at the first
close command did not separate the failures from the successes.

## Method

Two instrumented harnesses, run in-process against the unchanged embodiment, simulator,
twin, controller and scorer (scratch code under the main checkout's ignored
`outputs/task051-scratch/`; no repository behaviour was modified to produce them).

1. **Forensics** (`probe.py`): the unchanged ceiling v2 and the scripted collector
   (`scripted.apple_collector_policy`) on tuning resets 49100–49115, recording per
   command during the close phase: palm pose, top-down angle, apple pose and velocity,
   every contact on the apple geom with its world-frame force from `mj_contactForce`,
   the right-hand joint targets and measured angles, and the requested and applied
   14-D action.
2. **Paired closure experiment** (`fork.py`): run the unchanged ceiling v2 to the first
   `close` command, snapshot the live `MjData`, the command targets and the controller
   state, then replay several candidate closure designs — and several CEM realisations
   of the same design — **from that identical state**, scoring each by whether the
   unchanged scorer's `grasp` stage latches during the following lift. This isolates the
   closure from everything that precedes it.

`max apple xy` below is the largest horizontal displacement of the apple from its
position at the first close command, over the 45-command close.

## Finding 1: the ejection is caused inside the close, not by where the hand arrived

On tuning reset 49112, three CEM realisations from the **same** close-entry state gave
7.79 cm, 9.28 cm and 0.39 cm of apple motion — two ejections and one clean grasp. The
entry geometry, the apple pose, the palm pose and the hand state were bit-identical in
all three. The ejection is therefore a property of what the controller does during the
close, and it is stochastic in the CEM's proposal noise.

## Finding 2: what separates an ejection is *when* the fingers first touch the apple

The Dex3 synergy is a linear interpolation between the manifest's `right_open_rad` and
`right_closed_rad`. Its largest per-joint travel is 1.1 rad, so the embodiment's rate
limit (`joint_speed_limit_rad_s` 2.0 × `control_dt_s` 0.05 = 0.1 rad per command) caps
the normalized grasp command at 0.1/0.55 ≈ 0.182 per command: **eleven commands from
fully open to fully closed**. Both the ceiling and the collector request a step to +1 at
the first close command and are rate-limited identically.

Across the sixteen v2 closes recorded on 49100–49115:

| first hand–apple contact at close command | attempts | max apple xy (cm) | outcome |
|---|---:|---|---|
| ≤ 8 (4, 5, 6, 8) | 4 | 2.45, 3.12, 7.79, 9.12 | **all four ejected** |
| ≥ 10 (10 or 11) | 12 | 0.19 – 0.78 | all twelve held |
| scripted collector (11 on every reset) | 16 | 0.71 – 0.74 | all sixteen held |

The separation is complete on this sample. A contact before command 10 is a strike by a
hand that is still substantially **open and still moving**; a contact at 10–11 is made
by a hand that has already closed around the apple.

The per-command force traces show the mechanism directly. On a held close, all three
digits load together at command 10–11 (thumb/index/middle 21.0/9.7/11.8 N on 49100,
7.5/6.8/6.8 N on the collector) and the apple moves under a centimetre. On an ejecting
close the **thumb alone** loads first — 1.4 N at command 6 on 49112, 0.1 N at command 5
on 49114, 1.5 N at command 4 on 49115 — and the apple is pushed away from the palm in
+x by 0.71, 1.62, 2.83 cm on consecutive commands, leaving the hand before it shuts.

## Finding 3: why the ceiling's palm keeps moving while the hand closes

The close cost is `‖palm − grasp point‖` (the grasp point fixed at the close start) plus
a palm-drift guard with a 1 cm dead band plus an apple-disturbance term, and the CEM
plans all six right-arm degrees of freedom throughout. Inside the eleven ramp commands
that cost is nearly flat:

- the palm is ~6.4 cm above the (blocked) grasp point, so a 1 cm lateral change alters
  the 3-D distance term by 0.078 cm, and the drift guard exempts the first centimetre
  outright;
- the descent is blocked, so a candidate that commands −0.5 in z and one that commands 0
  reach almost the same palm position over the six-step horizon;
- the apple has not yet moved, so the disturbance term is zero for every candidate.

With the cost flat, the CEM's proposal noise (σ 0.3, bound 0.5) decides. Measured over
the eleven ramp commands of the sixteen v2 closes, the mean absolute lateral command was
**0.219–0.313** (0.33–0.47 cm per command) and the vertical command ranged over the full
[−0.5, +0.5], including commands that lift the palm while the fingers are on the apple
(49112 commanded +0.500 and +0.457 at ramp commands 9 and 10). The collector's close, by
contrast, commands a mean absolute lateral **0.029–0.047** (≈0.05 cm per command) and a
constant saturated −0.400 in z on every reset.

The consequence is the palm travel during the ramp: the collector sinks 0.97–0.99 cm and
drifts 0.20–0.22 cm laterally on all sixteen resets, while v2 sinks 0.40–1.85 cm and
drifts 0.05–1.22 cm. The runs at the fast end of that spread drive the open hand into
the apple early, and the runs at the slow end let the hand shut first.

## Finding 4: which closure designs fix it (paired experiment)

Each row replays one closure design from the same close-entry states as v2; `grasped` is
the unchanged scorer's grasp stage during the following lift.

| Closure design | arm command during close | grasp command | runs | seeds | grasped | max apple xy, all runs (cm) | max apple xy when grasped (cm) |
|---|---|---|---:|---:|---:|---|---|
| `v2` (control) | CEM, all 6 DoF, ±0.5 | step to +1 (rate-limited, 11 cmds) | 53 | 19 | 49 | 0.18 – 9.28 | 0.18 – 2.94 |
| `hold` | frozen at zero | same | 1 | 1 | 0 | 16.85 | — |
| `narrow` | CEM, ±0.05 | same | 2 | 1 | 0 | 16.85 | — |
| `jitter` | ±0.4 lateral square wave, −0.4 z | same | 1 | 1 | 0 (guard refused) | 1.36 | — |
| `press_half` | zero lateral, constant −0.2 | same | 1 | 1 | 0 | 5.23 | — |
| `press` | zero lateral, constant −0.4 | same | 1 | 1 | 0 (guard refused) | 0.42 | — |
| `sched06` | zero lateral, constant −0.06 | linear ramp over 24 cmds | 2 | 2 | 0 | 4.92 – 5.44 | — |
| `sched12` | zero lateral, constant −0.12 | linear ramp over 24 cmds | 2 | 2 | 0 | 5.73 – 6.49 | — |
| `sched12f` | zero lateral, constant −0.12 | same as v2 | 2 | 2 | 0 | 5.45 – 17.85 | — |
| `sched20` | zero lateral, constant −0.20 | linear ramp over 24 cmds | 2 | 2 | 0 (guard refused) | 0.00 – 1.62 | — |
| `ramp` | CEM, all 6 DoF, ±0.5 | linear ramp over 24 cmds | 48 | 16 | 47 | 1.88 – 3.57 | 1.88 – 3.57 |
| `cage_slow` | frozen lateral, CEM in [−0.5, 0] | linear ramp over 24 cmds | 48 | 16 | 42 | 0.33 – 19.04 | 0.33 – 4.53 |
| `cage_gentle` | frozen lateral, CEM in [−0.12, 0] | linear ramp over 24 cmds | 4 | 2 | 0 | 4.37 – 4.49 | — |
| `cage_g0` | frozen lateral, CEM in [−0.12, 0] | same as v2 | 24 | 8 | 0 | 4.96 – 14.86 | — |
| `cage_g25` | frozen lateral, CEM in [−0.25, 0] | same as v2 | 24 | 8 | 5 | 0.28 – 23.43 | 0.28 – 0.67 |
| `zonly` | frozen lateral, CEM in ±0.5 z | same as v2 | 2 | 1 | 2 | 0.56 – 0.60 | 0.56 – 0.60 |
| **`cage`** | **frozen lateral/rotational, CEM in [−0.5, 0]** | **same as v2** | **72** | **24** | **70** | **0.33 – 5.61** | **0.33 – 1.45** |

Four things follow.

- **Freezing the lateral and rotational palm command during the close is what helps.**
  `cage` grasped on 70 of 72 paired runs over 24 tuning resets and kept the apple within
  1.45 cm whenever it grasped, against v2's 49/53 over 19 resets and 0.18–2.94 cm. The two
  designs were not run on the same seed set (see Limits), so this is a screen, not a
  matched comparison. `cage`'s two failures were one guard refusal (below) and one run
  that reached the 200-command budget.
- **Downward pressure is necessary, and it has to be free to be large.** `hold` and
  `narrow`, which let the palm barely sink, ejected the apple by 16.85 cm — one replay
  each, on a single seed, so that figure is anecdotal even though the general claim below
  is not: the fingers curl around the apple above its equator and squeeze it out
  sideways. Tightening
  the descent bound reproduces that failure — `cage_g0` and `cage_gentle` (bound 0.12)
  grasped 0/28 and `cage_g25` (bound 0.25) 5/24, against `cage`'s 70/72 with the full
  bound. What matters is removing the *lateral* freedom, not the vertical one.
- **Slowing the synergy ramp makes things worse, not better.** `ramp` and `cage_slow`
  mostly grasp, but the apple moves up to 3.6 and 4.5 cm instead of under 1.5 cm, and
  every fully scheduled design (`sched06`/`sched12`/`sched12f`/`sched20`) failed on every
  run. A slower ramp means more palm descent happens while the fingers are still open —
  exactly the condition Finding 2 identifies as the ejection. The eleven-command
  rate-limited closure should be kept.
- **A fixed press is brittle.** The fixed-press designs either ejected the apple or
  tripped the embodiment's measured joint-velocity guard (5 rad/s), which a planner free
  to choose a smaller vertical command avoids. Keeping the CEM on the single vertical
  degree of freedom also keeps the ceiling's runtime rollout-parity check non-vacuous.

## Conclusion (the cause)

**The v2 close phase plans the palm while the hand is closing, and its cost cannot see
the difference between the candidates it is choosing among.** The CEM's proposal noise
therefore sets the palm's motion during the eleven commands the Dex3 synergy needs to
shut. When that noise happens to drive the palm down and sideways quickly, the still-open
thumb reaches the apple first and sweeps it out of the hand; when it happens to be gentle
the hand shuts first and cages the apple. The scripted collector never ejects the apple
because its close commands essentially no lateral motion and a constant, contact-limited
downward press, so its first contact is always at command 11, after the hand has closed.

Nothing about the simulator's contact parameters is implicated: the apple
(friction 1/0.01/0.001, `solref` 0.02/1, default `solimp`, 0.08 kg, 2.7 cm sphere) and
the Dex3 fingertip geoms (friction 1/0.005/0.0001, same solver parameters) are the
pinned upstream Unitree values plus the declared procedural apple, and **they were not
changed**. The collector reaches 16/16 on the same resets under the same physics.

## What this implies for the redesign (ceiling v3)

Close the hand with the palm caged and still:

1. pin the lateral and rotational arm deltas to zero for the whole close — the v2
   descent has already centred the palm, and lateral motion during closure can only
   shear the object;
2. keep the vertical degree of freedom under the existing CEM but allow only descent
   (bounded, never rising), so the palm keeps sinking around the apple as the fingers
   curl off the table without a fixed press that trips the velocity guard. Only the
   *bound* is supported by a large sample; the "never rising" half rests on the
   two-replay `zonly` row and is a conservative addition, not separately attributable;
3. keep the eleven-command rate-limited closure; do **not** slow the synergy ramp.

This is `object_ceiling_v3.ObjectCeilingV3Controller`. Every other v2 behaviour — the
xy-weighted descent, the carry costs, the release predictor, the exactly-probed release,
the twin and both parity checks — is inherited unchanged.

## Limits

- Exploratory, on 16 tuning resets for the forensics (49100–49115) and 24 for the paired
  closure experiment, with at most 72 replays per design. The declared tuning range is
  49100–49131; the seeds actually stepped are 49100–49117, 49120 and 49124–49131. The counts are design
  evidence, not a gate, and they were all produced before the v3 preregistration was
  frozen. Designs were not run on identical seed sets: `v2` and `cage` were replayed on
  every seed a process reached, while the clearly-failing designs were stopped early, so
  the run counts differ by design and the comparison is a screen, not a matched trial.
- The paired experiment restores the live `MjData` and the controller state; it
  reproduces the unchanged v2 close bit-for-bit (49100 v2 reproduced 0.52 cm from the
  full run), but it is scratch code and is not part of the repository.
- The separation in Finding 2 is complete on n = 16 v2 closes; with four ejections the
  uncertainty on the rate is wide.
- "Grasped" in Finding 4 is the scorer's grasp stage within a 200-command close+lift
  budget, not full task success. Full success is measured only by the preregistered v3
  run.
