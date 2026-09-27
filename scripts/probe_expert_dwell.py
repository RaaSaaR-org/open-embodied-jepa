"""Exploratory probe for the TASK-067 control-formulation proposal (not a gate, not a result).

Question: how much of the scripted collector's demonstration is *dwell* -- the palm parked at a
converged phase target, commanding almost nothing, while the collector's clock counts down to the
next phase -- and how often does a phase switch happen from such a parked state?

Why it matters: ``scripted.OracleManipulationPolicy`` switches phase on a command counter, not on
an observable event. A memoryless policy that sees one frame and proprioception (the TASK-056
arms had no clock input) receives, at a parked state, the label "do nothing" for every dwell step
and the label "start the next phase" for one step. This probe counts both from the label sidecars.

Reads **only the label sidecars of train root episodes** of ``data/apple-look-v1`` (the split is
taken from the corpus manifest; val, test and branch episodes are never opened). Each sidecar's
sha256 is checked against the manifest before it is read. No frame is decoded, nothing is
trained, nothing is simulated. Privileged labels are used for analysis only.

Usage::

    uv run --no-sync python scripts/probe_expert_dwell.py --output outputs/task067-dwell/run-1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embodied_jepa import training_labels  # noqa: E402

DATASET = ROOT / "data" / "apple-look-v1"
MANIFEST_SHA256 = "81d760d1f5a61834b4f37c3daf12ad1a7175e57fe99fe11d5fb4cecbff8edb64"
PHASES = ("orient", "descend", "close", "lift", "transfer", "release_high", "lower_open", "retreat")
# A step is "parked" when the palm moves less than 1 mm (the action scale is 15 mm per unit
# command, so 1 mm is a command of about 0.07) and the commanded translation is below 0.07.
PARK_MOVE_M = 0.001
PARK_COMMAND = 0.07


def _manifest(dataset: Path) -> dict:
    path = dataset / "meta" / "jepa_manifest.json"
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != MANIFEST_SHA256:
        raise SystemExit(f"manifest sha256 {digest} is not the accepted apple-look-v1 manifest")
    return json.loads(payload)


def _phase_runs(phase):
    """Yield (phase_index, start, stop) for maximal constant runs of a phase-index array."""
    start = 0
    for t in range(1, len(phase) + 1):
        if t == len(phase) or phase[t] != phase[start]:
            yield int(phase[start]), start, t
            start = t


def probe(dataset: Path) -> dict:
    manifest = _manifest(dataset)
    train = set(manifest["splits"]["train"])
    episodes = [
        e
        for e in manifest["episodes"]
        if e["episode_id"] in train and e["metadata"]["kind"] == "root"
    ]
    per_phase = {
        name: {
            "steps": 0,
            "parked": 0,
            "runs": 0,
            "switches": 0,
            "switch_from_parked": 0,
            "still": 0,
        }
        for name in PHASES
    }
    still_commands = {name: [] for name in PHASES}
    per_phase_noise0 = {name: {"steps": 0, "parked": 0, "still": 0} for name in PHASES}
    noise0_roots = 0
    orient_first_park = []
    outcomes = {}
    total_steps = total_parked = 0
    for episode in episodes:
        labels = training_labels.load(
            dataset,
            episode["metadata"]["training_labels"],
            groups=("collector", "privileged"),
            acknowledge_privileged_training_labels=True,
        )
        phase = labels["collector__phase_index"]
        base = labels["collector__base_action"]
        palm = labels["privileged__palm_position_world"]
        move = np.linalg.norm(np.diff(palm, axis=0), axis=1)  # step t -> t + 1
        command = np.abs(base[:, 6:9]).max(axis=1)
        parked = (move < PARK_MOVE_M) & (command < PARK_COMMAND)
        meta = episode["metadata"]
        noise0 = meta["root_noise_level"] == 0
        noise0_roots += int(noise0)
        stages = meta["privileged_outcome_labels"]["stages"]
        key = f"aim_offset={meta['aim_offset_applied']},noise_level={meta['root_noise_level']}"
        cell = outcomes.setdefault(key, {"roots": 0, "grasp": 0, "success": 0})
        cell["roots"] += 1
        cell["grasp"] += int(stages["grasp"])
        cell["success"] += int(stages["success"])
        for index, start, stop in _phase_runs(phase):
            if index < 0:  # the look prefix
                continue
            name = PHASES[index]
            record = per_phase[name]
            record["steps"] += stop - start
            record["parked"] += int(parked[start:stop].sum())
            still = move[start:stop] < PARK_MOVE_M
            record["still"] += int(still.sum())
            still_commands[name].extend(command[start:stop][still].tolist())
            record["runs"] += 1
            if stop < len(phase):  # a switch to the next phase happens at ``stop``
                record["switches"] += 1
                record["switch_from_parked"] += int(parked[stop - 1])
            if noise0:
                per_phase_noise0[name]["steps"] += stop - start
                per_phase_noise0[name]["parked"] += int(parked[start:stop].sum())
                per_phase_noise0[name]["still"] += int(still.sum())
            if name == "orient":
                hits = np.flatnonzero(parked[start:stop])
                if hits.size:
                    orient_first_park.append(int(hits[0]))
        valid = phase >= 0
        total_steps += int(valid.sum())
        total_parked += int((parked & valid).sum())
    for name, record in per_phase.items():
        record["parked_fraction"] = record["parked"] / record["steps"] if record["steps"] else None
        record["still_fraction"] = record["still"] / record["steps"] if record["steps"] else None
        values = still_commands[name]
        record["still_command_median"] = float(np.median(values)) if values else None
    for record in per_phase_noise0.values():
        record["parked_fraction"] = record["parked"] / record["steps"] if record["steps"] else None
        record["still_fraction"] = record["still"] / record["steps"] if record["steps"] else None
    return {
        "question": "dwell and clock-triggered phase switches in the collector's demonstrations",
        "dataset_manifest_sha256": MANIFEST_SHA256,
        "episodes_read": sorted(e["episode_id"] for e in episodes),
        "episodes_read_count": len(episodes),
        "split_read": "train roots only (no val, test or branch episode opened)",
        "park_rule": {
            "palm_move_m_below": PARK_MOVE_M,
            "max_abs_translation_command_below": PARK_COMMAND,
            "still": "palm_move_m_below only, whatever the command",
        },
        "total_policy_steps": total_steps,
        "total_parked_steps": total_parked,
        "parked_fraction_all_phases": total_parked / total_steps,
        "per_phase": per_phase,
        "per_phase_noise_level_0_roots": per_phase_noise0,
        "noise_level_0_roots": noise0_roots,
        "root_outcomes_by_aim_offset_and_noise": dict(sorted(outcomes.items())),
        "orient_first_parked_step": {
            "count": len(orient_first_park),
            "median": float(np.median(orient_first_park)) if orient_first_park else None,
            "p90": float(np.percentile(orient_first_park, 90)) if orient_first_park else None,
            "max": int(max(orient_first_park)) if orient_first_park else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    report = probe(args.dataset)
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {k: v for k, v in report.items() if k != "episodes_read"}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
