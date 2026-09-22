# TASK-049 pre-run review: object-aware ceiling v2

Fresh-context reviewer subagent, independent of the author, read-only, on commit
`447fce7` (main `61f8ec1`). Recorded verbatim in substance; the author's response
is at the end.

## Verdict: CLEAR TO RUN

No blocking defects. The release phase executes exactly the probed sequence; the
probe leaves the live state and the parity bookkeeping untouched; v1 behaviour and
the v1 plan are unchanged; the gate logic matches the preregistered readings table;
the budgets match the protocol; tests and lint pass.

## Blocking findings

None.

## Non-blocking recommendations

1. The carried-over-the-plate hand-over condition (within 3 cm, in contact,
   placement predicted) is a proxy for the scorer's `transport` stage, which also
   needs a ≥ 5 cm lift while over the plate. From `lower` the apple is low; if
   transport ended blocked without carrying the apple over the plate, a placement
   could be predicted without the scorer counting it. Call it a proxy.
2. The protocol said no 45100–45107 reset was simulated during design, but a
   scratch script reset them (no control steps) for the envelope/offset table.
   Reword.
3. `provenance_valid` is computed over all records, so a secondary attempt with
   invalid provenance fails the gate; say so.
4. Add a multi-step physics test that probe and live release agree (the reviewer's
   own scratch check found bit-exact agreement over 60 release commands).
5. `planning_dynamics` stays `privileged_mujoco_rollout_object_v1` (inherited);
   note it so it is not misread.
6. The MC acceptance text still said "close phase holding the achieved palm pose".
7. The TRAIN smoke ran from the uncommitted tree; the frozen run from the commit is
   the real check.
8. The learned-term list should also say that transitions read the *current*
   apple/plate/contact state, so a learned counterpart needs state estimation.
9. Timeout margin ≈ 1.8×; avoid heavy concurrent load if possible.
10. v2 declares transport blocked with no near-plate condition (v1: within 4 cm);
    state it.
11. The "v1 plan unchanged" unit test is self-referential; the reviewer compared a
    freshly generated v1 plan with the recorded TASK-047 `plan.json`: only run-time
    provenance keys differ.

## What was checked

Probe/transition ordering in one `step()`; release schedule index alignment
(commands 0–59 use `schedule[0..59]`); zero-width bounds make every candidate the
probed command; probe copies full MjData/targets/body_pos and never touches
`_search_first_steps`, `_search_full`, `_pending_parity`; trace JSON; dwell 0.2 s ≥
scorer 0.15 s; all protocol numbers against code and manifest; seeds, primary /
secondary split, outcome precedence, counted-attempt rule, limits (1200 / 20400 /
21600 / 10 s); worker builds v2 classes only for v2 plans; `git grep` for
45100–45107 and 49000–49015; `pytest` (90 passed, physics tests ran), `ruff`, `mc
validate`.

## Author response (revision R1)

Items 1, 2, 3, 8, 10 applied as protocol wording; item 4 added as
`test_live_release_reproduces_the_probed_release_exactly` (apple path bit-exact);
item 6 applied to the MC task; items 5, 7, 9, 11 recorded for the results. No reset,
seed, threshold, budget, parameter or command changed.
