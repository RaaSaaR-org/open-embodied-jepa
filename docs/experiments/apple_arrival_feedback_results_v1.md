# Apple arrival-feedback audit v1: negative paired result

Clearing the commitment at measured arrival **did not complete goal 2's dwell**
in this development state. Both original and arrival-interrupt siblings accepted
all 16 commands. Neither advanced past goal 2; maximum consecutive dwell was
one observation, below the unchanged requirement of three. Every physical task
stage remained false. These are two completed diagnostic negatives, not missing
or budget-truncated outcomes.

The single [preregistered audit](apple_arrival_feedback_v1.md) completed with exit
code 0 in **5.493 supervisor wall seconds / 5.165 worker CPU seconds**, inside its
60/40-second limits. No retry, extension, threshold change, training, or final
cohort execution occurred. Initial replay steps 0–59, root-60's original decision,
and the original sibling's continuation through steps 60–75 matched the frozen
trace at the specified tolerances. Restored root sensors matched exactly, and
post-run input integrity passed.

| Sibling | Accepted slots | Arrival-triggered cache clears | Maximum dwell | Advanced past goal 2 | Physical stages |
|---|---:|---:|---:|---|---|
| Original | 16/16 | 0 | 1 | No | None |
| Arrival interrupt | 16/16 | 1 | 1 | No | None |

The arrival rule cleared plan 58 at offset 2 before the first branch command.
Normal model-ranked planning then chose a perturbation instead of continuing the
demonstration proposal. Its next observed distance was 0.063493, outside the
0.060909 threshold, so dwell reset. The original continuation also left the
threshold, with next distance 0.062281. No additional arrival clear fired because
neither branch subsequently re-entered the threshold during the 16 decisions.
No hold or other action was forced.

## All decision observations

Distances below are measured **before each command**, not after its execution.
The final row is the observation before the 16th command; no later endpoint RGB
measurement or 17th decision is supplied by this audit. Every row uses goal 2,
TRAIN frame 56, and threshold 0.0609087203.

| Branch command (1-based) | Original distance / dwell | Arrival-interrupt distance / dwell |
|---|---:|---:|
| 1 | 0.058320 / 1 | 0.058320 / 1 |
| 2 | 0.062281 / 0 | 0.063493 / 0 |
| 3 | 0.065898 / 0 | 0.064364 / 0 |
| 4 | 0.069094 / 0 | 0.062425 / 0 |
| 5 | 0.067640 / 0 | 0.070532 / 0 |
| 6 | 0.072392 / 0 | 0.072739 / 0 |
| 7 | 0.075107 / 0 | 0.089950 / 0 |
| 8 | 0.079147 / 0 | 0.097619 / 0 |
| 9 | 0.094657 / 0 | 0.099316 / 0 |
| 10 | 0.096398 / 0 | 0.098853 / 0 |
| 11 | 0.103209 / 0 | 0.097281 / 0 |
| 12 | 0.103606 / 0 | 0.108824 / 0 |
| 13 | 0.113481 / 0 | 0.119226 / 0 |
| 14 | 0.121040 / 0 | 0.123733 / 0 |
| 15 | 0.116698 / 0 | 0.124708 / 0 |
| 16 | 0.120463 / 0 | 0.124098 / 0 |

Both siblings used four full searches and twelve cached decisions, at different
command offsets. Original selected-origin labels were twelve perturbations,
two demonstration proposals, and two holds; arrival-interrupt labels were twelve
perturbations and four holds. Cached labels describe their originating selected
plan, not a fresh model evaluation. All 32 projected requests exactly matched
the acknowledged applied actions. There are 32 matched pending/result pairs,
32 accepted slots, zero rejected or unavailable slots, and no uncertain pending
execution. All 32 scorer records have false reach/grasp/transport/place/release
and success flags.

## Provenance and interpretation

Source revision: `b667e19d686df904e0c90716cd63794ed44addc3`. The audit imported the
unchanged captured TASK039 runtime and H16 checkpoint. The
[compact manifest](../../benchmarks/manifests/apple-arrival-feedback-v1.json)
records source/input/script/protocol/environment identities, both slot ledgers,
all decision distances/dwell, selected origins, the cache-clear event and hashes.
Local raw artifacts remain under `outputs/apple-arrival-feedback-v1/`, with a
separate `artifact_hashes.json` inventory. No TEST image was decoded.

This comparison rejects the proposed immediate dwell benefit at this one
correlated development state. It does not show that arrival interruption is
universally ineffective, identify the unique cause of the broader control stall,
or establish a manipulation success rate. The working apple pick-and-place gate
remains unmet; the final 20-reset cohort stays untouched.
